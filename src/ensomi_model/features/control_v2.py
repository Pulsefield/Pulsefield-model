from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .control import (
    HitObject,
    OnsetEvent,
    center_of_mass,
    clip01,
    default_beat_length_at,
    group_onsets,
    hand_side_bucket,
    interval_kernel_integral_on_grid,
    iter_cols,
    make_grid,
    merge_intervals,
    movement_bucket,
    normalize_pair_by_leftmost_active,
    popcount,
    rhythm_bucket,
    validate_control_hits,
)


VALUE_FEATURE_NAMES = [
    "density_level",
    "density_burst",
    "hold_occupancy",
    "ln_change_rate",
    "chord_ratio",
    "jack_excess",
    "jack_streak_exposure",
    "hand_balance_signed",
    "hand_imbalance_abs",
    "repeat_exact",
    "repeat_shift",
    "repeat_motion",
    "repeat_rhythm",
]

CONFIDENCE_FEATURE_NAMES = [
    "density_confidence",
    "ln_change_confidence",
    "chord_confidence",
    "jack_confidence",
    "jack_streak_confidence",
    "hand_confidence",
    "repeat_confidence",
    "control_confidence",
]

MODEL_FEATURE_NAMES = VALUE_FEATURE_NAMES + CONFIDENCE_FEATURE_NAMES

FEATURE_NAMES = MODEL_FEATURE_NAMES

DEBUG_ARRAY_NAMES = [
    "density_raw_short",
    "density_raw_med",
    "density_sum_w_short",
    "density_sum_w2_short",
    "density_n_eff_short",
    "density_sum_w_med",
    "density_sum_w2_med",
    "density_n_eff_med",
    "density_confidence",
    "hold_active_columns",
    "ln_change_raw",
    "ln_change_sum_w",
    "ln_change_sum_w2",
    "ln_change_n_eff",
    "ln_change_confidence",
    "chord_num",
    "chord_den",
    "chord_sum_w2",
    "chord_n_eff",
    "chord_confidence",
    "chord_ratio_raw",
    "jack_observed",
    "jack_expected_null",
    "jack_pair_count",
    "jack_sum_w",
    "jack_sum_w2",
    "jack_n_eff",
    "jack_confidence",
    "jack_excess_raw",
    "jack_streak_raw",
    "jack_streak_n_eff",
    "jack_streak_confidence",
    "jack_streak_max",
    "hand_left_load",
    "hand_right_load",
    "hand_n_eff",
    "hand_confidence",
    "hand_balance_raw",
    "repeat_exact_n_eff",
    "repeat_shift_n_eff",
    "repeat_motion_n_eff",
    "repeat_rhythm_n_eff",
    "repeat_exact_top1_freq",
    "repeat_shift_top1_freq",
    "repeat_motion_top1_freq",
    "repeat_rhythm_top1_freq",
    "repeat_exact_pattern_variety",
    "repeat_shift_pattern_variety",
    "repeat_motion_pattern_variety",
    "repeat_rhythm_pattern_variety",
    "repeat_confidence",
    "control_confidence",
    "valid_control_mask",
]


@dataclass
class FeatureConfigV2:
    key_count: int = 4
    grid_step: float = 0.10
    onset_eps_min: float = 0.002
    onset_eps_beat_div: float = 768.0
    min_ln_len: float = 0.03
    density_alpha: float = 1.15
    density_L_short: float = 0.75
    density_L_med: float = 3.0
    density_burst_scale: float = 1.0
    density_n_eff_min: float = 1.0
    density_gate_scale: float = 2.0
    hold_L: float = 2.0
    ln_change_L: float = 1.0
    chord_L: float = 2.0
    ratio_n_eff_min: float = 3.0
    ratio_gate_scale: float = 4.0
    jack_L: float = 1.0
    jack_gap: float = 0.22
    jack_excess_scale: float = 2.0
    jack_streak_cap: int = 4
    hand_L: float = 3.0
    hand_prior: float = 3.0
    repeat_L: float = 3.0
    repeat_max_gap_beats: float = 4.0
    repeat_max_gap_s: float = 3.0


def confidence_from_n_eff(
    n_eff: np.ndarray,
    n0: float = 3.0,
    k: float = 4.0,
) -> np.ndarray:
    if k <= 0:
        raise ValueError(f"confidence scale must be positive: {k}")
    x = (np.asarray(n_eff, dtype=float) - n0) / k
    return clip01(1.0 - np.exp(-np.maximum(0.0, x)))


def confidence_gate(
    value: np.ndarray,
    confidence: np.ndarray,
    neutral_value: float = 0.0,
) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    return confidence * value + (1.0 - confidence) * neutral_value


def robust_tanh(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    if scale <= 0:
        raise ValueError(f"tanh scale must be positive: {scale}")
    return np.tanh(np.asarray(x, dtype=float) / scale)


def kernel_weighted_sum_and_support(
    grid: np.ndarray,
    event_times: Sequence[float],
    weights: Sequence[float],
    L: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if L <= 0:
        raise ValueError(f"kernel support must be positive: {L}")
    if len(event_times) != len(weights):
        raise ValueError(
            f"event_times and weights length mismatch: "
            f"{len(event_times)} vs {len(weights)}"
        )

    weighted = np.zeros_like(grid, dtype=float)
    sum_w = np.zeros_like(grid, dtype=float)
    sum_w2 = np.zeros_like(grid, dtype=float)
    if len(grid) == 0 or len(event_times) == 0:
        return weighted, sum_w, sum_w2, np.zeros_like(grid, dtype=float)

    for center, weight in zip(event_times, weights):
        lo = int(np.searchsorted(grid, center - L, side="left"))
        hi = int(np.searchsorted(grid, center + L, side="right"))
        if hi <= lo:
            continue
        u = grid[lo:hi] - float(center)
        kernel = np.maximum(0.0, 1.0 - np.abs(u) / L) / L
        weighted[lo:hi] += float(weight) * kernel
        sum_w[lo:hi] += kernel
        sum_w2[lo:hi] += kernel * kernel

    n_eff = np.zeros_like(grid, dtype=float)
    np.divide(sum_w * sum_w, sum_w2, out=n_eff, where=sum_w2 > 1e-12)
    return weighted, sum_w, sum_w2, n_eff


def valid_control_mask(grid: np.ndarray, hits: Sequence[HitObject]) -> np.ndarray:
    if len(grid) == 0 or not hits:
        return np.zeros_like(grid, dtype=bool)
    first = min(hit.start for hit in hits)
    last = max(hit.end if hit.end is not None else hit.start for hit in hits)
    return (grid >= first) & (grid <= last)


def edge_neutralize(values: np.ndarray, valid_mask: np.ndarray, neutral: float = 0.0) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    out[~valid_mask] = neutral
    return out


def density_features(
    grid: np.ndarray,
    onsets: Sequence[OnsetEvent],
    cfg: FeatureConfigV2,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    times = [event.t for event in onsets]
    weights = [float(event.chord_size) ** cfg.density_alpha for event in onsets]
    d_short, short_w, short_w2, short_n_eff = kernel_weighted_sum_and_support(
        grid,
        times,
        weights,
        cfg.density_L_short,
    )
    d_med, med_w, med_w2, med_n_eff = kernel_weighted_sum_and_support(
        grid,
        times,
        weights,
        cfg.density_L_med,
    )
    density_level = np.log1p(d_med)
    density_burst = robust_tanh(d_short - d_med, cfg.density_burst_scale)
    density_confidence = confidence_from_n_eff(
        med_n_eff,
        cfg.density_n_eff_min,
        cfg.density_gate_scale,
    )
    features = {
        "density_level": density_level,
        "density_burst": density_burst,
    }
    debug = {
        "density_raw_short": d_short,
        "density_raw_med": d_med,
        "density_sum_w_short": short_w,
        "density_sum_w2_short": short_w2,
        "density_n_eff_short": short_n_eff,
        "density_sum_w_med": med_w,
        "density_sum_w2_med": med_w2,
        "density_n_eff_med": med_n_eff,
        "density_confidence": density_confidence,
    }
    return features, debug


def ln_intervals_by_col(
    hits: Sequence[HitObject],
    cfg: FeatureConfigV2,
) -> list[list[tuple[float, float]]]:
    by_col: list[list[tuple[float, float]]] = [[] for _ in range(cfg.key_count)]
    for hit in hits:
        if hit.end is None:
            continue
        if not (0 <= hit.col < cfg.key_count):
            continue
        if hit.end - hit.start >= cfg.min_ln_len:
            by_col[hit.col].append((hit.start, hit.end))
    return by_col


def hold_and_ln_change_features(
    grid: np.ndarray,
    hits: Sequence[HitObject],
    cfg: FeatureConfigV2,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    by_col = ln_intervals_by_col(hits, cfg)
    total_active = np.zeros_like(grid, dtype=float)
    for col in range(cfg.key_count):
        total_active += interval_kernel_integral_on_grid(
            grid,
            merge_intervals(by_col[col]),
            cfg.hold_L,
        )
    hold_occupancy = clip01(total_active / float(cfg.key_count))

    changes: list[tuple[float, int, int]] = []
    for col, intervals in enumerate(by_col):
        for start, end in intervals:
            changes.append((start, col, 1))
            changes.append((end, col, -1))
    changes.sort(key=lambda item: item[0])

    change_times: list[float] = []
    change_weights: list[float] = []
    active_counts = [0 for _ in range(cfg.key_count)]
    i = 0
    while i < len(changes):
        t = changes[i][0]
        before = 0
        for col, count in enumerate(active_counts):
            if count > 0:
                before |= 1 << col
        while i < len(changes) and abs(changes[i][0] - t) <= 1e-9:
            _, col, delta = changes[i]
            active_counts[col] = max(0, active_counts[col] + delta)
            i += 1
        after = 0
        for col, count in enumerate(active_counts):
            if count > 0:
                after |= 1 << col
        if after != before:
            change_times.append(t)
            change_weights.append(float(popcount(before ^ after)))

    ln_change_raw, change_w, change_w2, change_n_eff = kernel_weighted_sum_and_support(
        grid,
        change_times,
        change_weights,
        cfg.ln_change_L,
    )
    ln_change_rate = np.log1p(ln_change_raw)
    ln_change_confidence = confidence_from_n_eff(
        change_n_eff,
        cfg.density_n_eff_min,
        cfg.density_gate_scale,
    )
    features = {
        "hold_occupancy": hold_occupancy,
        "ln_change_rate": ln_change_rate,
    }
    debug = {
        "hold_active_columns": total_active,
        "ln_change_raw": ln_change_raw,
        "ln_change_sum_w": change_w,
        "ln_change_sum_w2": change_w2,
        "ln_change_n_eff": change_n_eff,
        "ln_change_confidence": ln_change_confidence,
    }
    return features, debug


def chord_ratio_feature(
    grid: np.ndarray,
    onsets: Sequence[OnsetEvent],
    cfg: FeatureConfigV2,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    times = [event.t for event in onsets]
    strengths = [
        (float(event.chord_size) - 1.0) / max(1.0, float(cfg.key_count - 1))
        for event in onsets
    ]
    numer, _, _, _ = kernel_weighted_sum_and_support(grid, times, strengths, cfg.chord_L)
    _, denom, denom_w2, n_eff = kernel_weighted_sum_and_support(
        grid,
        times,
        [1.0 for _ in onsets],
        cfg.chord_L,
    )
    raw = np.zeros_like(grid, dtype=float)
    np.divide(numer, denom, out=raw, where=denom > 1e-12)
    raw = clip01(raw)
    confidence = confidence_from_n_eff(n_eff, cfg.ratio_n_eff_min, cfg.ratio_gate_scale)
    chord_ratio = confidence_gate(raw, confidence, neutral_value=0.0)
    debug = {
        "chord_num": numer,
        "chord_den": denom,
        "chord_sum_w2": denom_w2,
        "chord_n_eff": n_eff,
        "chord_confidence": confidence,
        "chord_ratio_raw": raw,
    }
    return clip01(chord_ratio), debug


def build_jack_terms(
    onsets: Sequence[OnsetEvent],
    cfg: FeatureConfigV2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pair_times: list[float] = []
    observed_weights: list[float] = []
    expected_weights: list[float] = []
    for j, cur in enumerate(onsets):
        for i in range(j - 1, -1, -1):
            prev = onsets[i]
            gap = cur.t - prev.t
            if gap <= 1e-6:
                continue
            if gap >= cfg.jack_gap:
                break
            pair_risk = max(0.0, 1.0 - gap / cfg.jack_gap) ** 2
            observed = float(popcount(prev.mask & cur.mask)) * pair_risk
            expected = (
                float(prev.chord_size)
                * float(cur.chord_size)
                / max(1.0, float(cfg.key_count))
                * pair_risk
            )
            pair_times.append(cur.t)
            observed_weights.append(observed)
            expected_weights.append(expected)

    streak_times: list[float] = []
    streak_weights: list[float] = []
    streak_maxes: list[float] = []
    last_time_by_col = [None for _ in range(cfg.key_count)]
    streak_by_col = [0 for _ in range(cfg.key_count)]
    for event in onsets:
        event_weight = 0.0
        event_max_streak = 0
        for col in iter_cols(event.mask, cfg.key_count):
            last_t = last_time_by_col[col]
            if last_t is None:
                streak_by_col[col] = 1
            else:
                gap = event.t - last_t
                if 1e-6 < gap < cfg.jack_gap:
                    streak_by_col[col] += 1
                    pair_risk = max(0.0, 1.0 - gap / cfg.jack_gap) ** 2
                    streak_factor = min(
                        max(0, streak_by_col[col] - 1),
                        cfg.jack_streak_cap,
                    ) / float(cfg.jack_streak_cap)
                    event_weight += pair_risk * streak_factor
                else:
                    streak_by_col[col] = 1
            last_time_by_col[col] = event.t
            event_max_streak = max(event_max_streak, streak_by_col[col])
        if event_weight > 0.0:
            streak_times.append(event.t)
            streak_weights.append(event_weight)
            streak_maxes.append(float(event_max_streak))

    return (
        np.asarray(pair_times, dtype=float),
        np.asarray(observed_weights, dtype=float),
        np.asarray(expected_weights, dtype=float),
        np.asarray(streak_times, dtype=float),
        np.asarray(streak_weights, dtype=float),
        np.asarray(streak_maxes, dtype=float),
    )


def local_event_max(
    grid: np.ndarray,
    event_times: np.ndarray,
    values: np.ndarray,
    L: float,
) -> np.ndarray:
    out = np.zeros_like(grid, dtype=float)
    if len(grid) == 0 or len(event_times) == 0:
        return out
    for gi, t in enumerate(grid):
        lo = int(np.searchsorted(event_times, t - L, side="left"))
        hi = int(np.searchsorted(event_times, t + L, side="right"))
        if hi > lo:
            out[gi] = float(np.max(values[lo:hi]))
    return out


def jack_features(
    grid: np.ndarray,
    onsets: Sequence[OnsetEvent],
    cfg: FeatureConfigV2,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    (
        pair_times,
        observed_weights,
        expected_weights,
        streak_times,
        streak_weights,
        streak_maxes,
    ) = build_jack_terms(onsets, cfg)
    observed, pair_sum_w, pair_sum_w2, pair_n_eff = kernel_weighted_sum_and_support(
        grid,
        pair_times,
        observed_weights,
        cfg.jack_L,
    )
    expected, _, _, _ = kernel_weighted_sum_and_support(
        grid,
        pair_times,
        expected_weights,
        cfg.jack_L,
    )
    pair_count, _, _, _ = kernel_weighted_sum_and_support(
        grid,
        pair_times,
        [1.0 for _ in pair_times],
        cfg.jack_L,
    )
    raw_excess = np.zeros_like(grid, dtype=float)
    np.divide(
        np.maximum(0.0, observed - expected),
        expected + 1e-9,
        out=raw_excess,
        where=expected > 1e-12,
    )
    confidence = confidence_from_n_eff(
        pair_n_eff,
        cfg.ratio_n_eff_min,
        cfg.ratio_gate_scale,
    )
    jack_excess = confidence_gate(
        robust_tanh(raw_excess, cfg.jack_excess_scale),
        confidence,
        neutral_value=0.0,
    )

    streak_raw, _, _, streak_n_eff = kernel_weighted_sum_and_support(
        grid,
        streak_times,
        streak_weights,
        cfg.jack_L,
    )
    streak_confidence = confidence_from_n_eff(
        streak_n_eff,
        max(2.0, cfg.ratio_n_eff_min - 1.0),
        cfg.ratio_gate_scale,
    )
    jack_streak_exposure = confidence_gate(
        robust_tanh(streak_raw, scale=1.0),
        streak_confidence,
        neutral_value=0.0,
    )

    features = {
        "jack_excess": clip01(jack_excess),
        "jack_streak_exposure": clip01(jack_streak_exposure),
    }
    debug = {
        "jack_observed": observed,
        "jack_expected_null": expected,
        "jack_pair_count": pair_count,
        "jack_sum_w": pair_sum_w,
        "jack_sum_w2": pair_sum_w2,
        "jack_n_eff": pair_n_eff,
        "jack_confidence": confidence,
        "jack_excess_raw": raw_excess,
        "jack_streak_raw": streak_raw,
        "jack_streak_n_eff": streak_n_eff,
        "jack_streak_confidence": streak_confidence,
        "jack_streak_max": local_event_max(grid, streak_times, streak_maxes, cfg.jack_L),
    }
    return features, debug


def hand_balance_features(
    grid: np.ndarray,
    onsets: Sequence[OnsetEvent],
    cfg: FeatureConfigV2,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    times = [event.t for event in onsets]
    mid = cfg.key_count // 2
    left_counts = [
        float(sum(1 for col in iter_cols(event.mask, cfg.key_count) if col < mid))
        for event in onsets
    ]
    right_counts = [float(event.chord_size) - left for event, left in zip(onsets, left_counts)]
    left_load, _, _, hand_n_eff = kernel_weighted_sum_and_support(
        grid,
        times,
        left_counts,
        cfg.hand_L,
    )
    right_load, _, _, _ = kernel_weighted_sum_and_support(
        grid,
        times,
        right_counts,
        cfg.hand_L,
    )
    total = left_load + right_load
    raw_balance = np.zeros_like(grid, dtype=float)
    np.divide(
        left_load - right_load,
        total,
        out=raw_balance,
        where=total > 1e-12,
    )
    raw_balance = np.clip(raw_balance, -1.0, 1.0)
    hand_confidence = np.zeros_like(grid, dtype=float)
    np.divide(
        total,
        total + cfg.hand_prior,
        out=hand_confidence,
        where=(total + cfg.hand_prior) > 1e-12,
    )
    signed = np.clip(confidence_gate(raw_balance, hand_confidence, 0.0), -1.0, 1.0)
    features = {
        "hand_balance_signed": signed,
        "hand_imbalance_abs": np.abs(signed),
    }
    debug = {
        "hand_left_load": left_load,
        "hand_right_load": right_load,
        "hand_n_eff": hand_n_eff,
        "hand_confidence": hand_confidence,
        "hand_balance_raw": raw_balance,
    }
    return features, debug


def build_repeat_tokens(
    onsets: Sequence[OnsetEvent],
    beat_length_at: Callable[[float], float],
    cfg: FeatureConfigV2,
) -> tuple[
    np.ndarray,
    list[tuple[Any, ...]],
    list[tuple[Any, ...]],
    list[tuple[Any, ...]],
    list[tuple[Any, ...]],
]:
    trans_times: list[float] = []
    exact_tokens: list[tuple[Any, ...]] = []
    shift_tokens: list[tuple[Any, ...]] = []
    motion_tokens: list[tuple[Any, ...]] = []
    rhythm_tokens: list[tuple[Any, ...]] = []
    prev_q: int | None = None
    if len(onsets) < 2:
        return (
            np.zeros((0,), dtype=float),
            exact_tokens,
            shift_tokens,
            motion_tokens,
            rhythm_tokens,
        )

    for i in range(1, len(onsets)):
        prev = onsets[i - 1]
        cur = onsets[i]
        gap = cur.t - prev.t
        if gap <= 1e-6:
            continue
        beat_len = beat_length_at(cur.t)
        max_gap = cfg.repeat_max_gap_s
        if beat_len > 1e-9:
            max_gap = min(max_gap, cfg.repeat_max_gap_beats * beat_len)
        if gap > max_gap:
            prev_q = None
            continue

        q = rhythm_bucket(gap, beat_len)
        exact = (prev.mask, cur.mask, q)
        n_prev, n_cur = normalize_pair_by_leftmost_active(prev.mask, cur.mask, cfg.key_count)
        shift = (n_prev, n_cur, q)
        delta = center_of_mass(cur.mask, cfg.key_count) - center_of_mass(prev.mask, cfg.key_count)
        prev_side = hand_side_bucket(prev.mask, cfg.key_count)
        cur_side = hand_side_bucket(cur.mask, cfg.key_count)
        motion = (
            movement_bucket(delta, cfg.key_count),
            prev_side,
            cur_side,
            prev.chord_size,
            cur.chord_size,
            q,
        )
        rhythm = (q,) if prev_q is None else (prev_q, q)

        trans_times.append(cur.t)
        exact_tokens.append(exact)
        shift_tokens.append(shift)
        motion_tokens.append(motion)
        rhythm_tokens.append(rhythm)
        prev_q = q

    return (
        np.asarray(trans_times, dtype=float),
        exact_tokens,
        shift_tokens,
        motion_tokens,
        rhythm_tokens,
    )


def concentration_score_on_grid(
    grid: np.ndarray,
    token_times: np.ndarray,
    tokens: Sequence[tuple[Any, ...]],
    L: float,
    n0: float = 3.0,
    gate_scale: float = 4.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if len(token_times) != len(tokens):
        raise ValueError(
            f"token_times and tokens length mismatch: "
            f"{len(token_times)} vs {len(tokens)}"
        )

    values = np.zeros_like(grid, dtype=float)
    n_eff_values = np.zeros_like(grid, dtype=float)
    top1_freq = np.zeros_like(grid, dtype=float)
    pattern_variety = np.zeros_like(grid, dtype=float)
    top_tokens = np.empty(len(grid), dtype=object)
    top_tokens[:] = None
    if len(grid) == 0 or len(token_times) < 2:
        return values, {
            "n_eff": n_eff_values,
            "top1_freq": top1_freq,
            "pattern_variety": pattern_variety,
            "top_token": top_tokens,
            "confidence": np.zeros_like(grid, dtype=float),
        }

    token_times = np.asarray(token_times, dtype=float)
    confidence = np.zeros_like(grid, dtype=float)
    for gi, t in enumerate(grid):
        lo = int(np.searchsorted(token_times, t - L, side="left"))
        hi = int(np.searchsorted(token_times, t + L, side="right"))
        if hi <= lo:
            continue
        counts: defaultdict[tuple[Any, ...], float] = defaultdict(float)
        total_w = 0.0
        total_w2 = 0.0
        for k in range(lo, hi):
            dt = abs(float(token_times[k]) - float(t))
            weight = max(0.0, 1.0 - dt / L)
            if weight <= 0.0:
                continue
            counts[tokens[k]] += weight
            total_w += weight
            total_w2 += weight * weight
        if total_w2 <= 1e-12 or total_w <= 1e-12:
            continue

        n_eff = (total_w * total_w) / total_w2
        n_eff_values[gi] = n_eff
        probs = [value / total_w for value in counts.values()]
        if not probs:
            continue
        top_token, top_weight = max(counts.items(), key=lambda item: item[1])
        top_tokens[gi] = top_token
        top1_freq[gi] = top_weight / total_w
        concentration = sum(prob * prob for prob in probs)
        baseline = 1.0 / max(n_eff, 1.0)
        concentration_normalized = (concentration - baseline) / max(1e-9, 1.0 - baseline)
        top1_normalized = (top1_freq[gi] - baseline) / max(1e-9, 1.0 - baseline)
        normalized = max(concentration_normalized, top1_normalized)
        normalized = float(np.clip(normalized, 0.0, 1.0))
        confidence[gi] = confidence_from_n_eff(
            np.asarray([n_eff], dtype=float),
            n0,
            gate_scale,
        )[0]
        values[gi] = confidence[gi] * normalized

        if len(probs) > 1:
            entropy = -sum(prob * math.log(max(prob, 1e-12)) for prob in probs)
            pattern_variety[gi] = float(entropy / math.log(len(probs)))

    debug = {
        "n_eff": n_eff_values,
        "top1_freq": top1_freq,
        "pattern_variety": pattern_variety,
        "top_token": top_tokens,
        "confidence": confidence,
    }
    return clip01(values), debug


def repeat_features(
    grid: np.ndarray,
    onsets: Sequence[OnsetEvent],
    beat_length_at: Callable[[float], float],
    cfg: FeatureConfigV2,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    trans_times, exact_tokens, shift_tokens, motion_tokens, rhythm_tokens = build_repeat_tokens(
        onsets,
        beat_length_at,
        cfg,
    )
    exact, exact_debug = concentration_score_on_grid(
        grid,
        trans_times,
        exact_tokens,
        cfg.repeat_L,
        cfg.ratio_n_eff_min,
        cfg.ratio_gate_scale,
    )
    shift, shift_debug = concentration_score_on_grid(
        grid,
        trans_times,
        shift_tokens,
        cfg.repeat_L,
        cfg.ratio_n_eff_min,
        cfg.ratio_gate_scale,
    )
    motion, motion_debug = concentration_score_on_grid(
        grid,
        trans_times,
        motion_tokens,
        cfg.repeat_L,
        cfg.ratio_n_eff_min,
        cfg.ratio_gate_scale,
    )
    rhythm, rhythm_debug = concentration_score_on_grid(
        grid,
        trans_times,
        rhythm_tokens,
        cfg.repeat_L,
        cfg.ratio_n_eff_min,
        cfg.ratio_gate_scale,
    )
    features = {
        "repeat_exact": exact,
        "repeat_shift": shift,
        "repeat_motion": motion,
        "repeat_rhythm": rhythm,
    }
    repeat_confidence = np.mean(
        np.stack(
            [
                exact_debug["confidence"],
                shift_debug["confidence"],
                motion_debug["confidence"],
                rhythm_debug["confidence"],
            ],
            axis=0,
        ),
        axis=0,
    )
    debug = {
        "repeat_exact_n_eff": exact_debug["n_eff"],
        "repeat_shift_n_eff": shift_debug["n_eff"],
        "repeat_motion_n_eff": motion_debug["n_eff"],
        "repeat_rhythm_n_eff": rhythm_debug["n_eff"],
        "repeat_exact_top1_freq": exact_debug["top1_freq"],
        "repeat_shift_top1_freq": shift_debug["top1_freq"],
        "repeat_motion_top1_freq": motion_debug["top1_freq"],
        "repeat_rhythm_top1_freq": rhythm_debug["top1_freq"],
        "repeat_exact_top_token": exact_debug["top_token"],
        "repeat_shift_top_token": shift_debug["top_token"],
        "repeat_motion_top_token": motion_debug["top_token"],
        "repeat_rhythm_top_token": rhythm_debug["top_token"],
        "repeat_exact_pattern_variety": exact_debug["pattern_variety"],
        "repeat_shift_pattern_variety": shift_debug["pattern_variety"],
        "repeat_motion_pattern_variety": motion_debug["pattern_variety"],
        "repeat_rhythm_pattern_variety": rhythm_debug["pattern_variety"],
        "repeat_confidence": repeat_confidence,
    }
    return features, debug


def extract_control_features(
    hits: Sequence[HitObject],
    beat_length_at: Callable[[float], float] | None = None,
    cfg: FeatureConfigV2 | None = None,
    grid: np.ndarray | None = None,
    normalizers: dict[str, Robust01] | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    return_debug: bool = True,
) -> dict[str, Any]:
    if cfg is None:
        cfg = FeatureConfigV2()
    if beat_length_at is None:
        beat_length_at = default_beat_length_at
    hits = list(hits)
    validate_control_hits(hits, cfg.key_count)
    if grid is None:
        grid = make_grid(hits, step=cfg.grid_step, start_time=start_time, end_time=end_time)

    onsets = group_onsets(hits, beat_length_at, cfg)
    valid_mask = valid_control_mask(grid, hits)

    density, density_debug = density_features(grid, onsets, cfg)
    hold_ln, hold_ln_debug = hold_and_ln_change_features(grid, hits, cfg)
    chord_ratio, chord_debug = chord_ratio_feature(grid, onsets, cfg)
    jack, jack_debug = jack_features(grid, onsets, cfg)
    hand, hand_debug = hand_balance_features(grid, onsets, cfg)
    repeat, repeat_debug = repeat_features(grid, onsets, beat_length_at, cfg)

    features = {
        **density,
        **hold_ln,
        "chord_ratio": chord_ratio,
        **jack,
        **hand,
        **repeat,
    }

    if normalizers is not None:
        for name in [
            "density_level",
            "ln_change_rate",
            "jack_excess",
            "jack_streak_exposure",
        ]:
            if name in normalizers:
                features[name] = normalizers[name].transform(features[name])

    confidence_inputs = [
        density_debug["density_confidence"],
        hold_ln_debug["ln_change_confidence"],
        chord_debug["chord_confidence"],
        jack_debug["jack_confidence"],
        hand_debug["hand_confidence"],
        repeat_debug["repeat_confidence"],
    ]
    if len(grid) == 0:
        control_confidence = np.zeros_like(grid, dtype=float)
    else:
        control_confidence = np.mean(np.stack(confidence_inputs, axis=0), axis=0)
    features["density_confidence"] = density_debug["density_confidence"]
    features["ln_change_confidence"] = hold_ln_debug["ln_change_confidence"]
    features["chord_confidence"] = chord_debug["chord_confidence"]
    features["jack_confidence"] = jack_debug["jack_confidence"]
    features["jack_streak_confidence"] = jack_debug["jack_streak_confidence"]
    features["hand_confidence"] = hand_debug["hand_confidence"]
    features["repeat_confidence"] = repeat_debug["repeat_confidence"]
    features["control_confidence"] = clip01(control_confidence)

    for name in list(features):
        features[name] = edge_neutralize(features[name], valid_mask, neutral=0.0)

    if len(grid) == 0:
        x = np.zeros((0, len(MODEL_FEATURE_NAMES)), dtype=float)
    else:
        x = np.stack([features[name] for name in MODEL_FEATURE_NAMES], axis=1)

    debug: dict[str, Any] = {}
    if return_debug:
        debug["onsets"] = onsets
        debug["valid_control_mask"] = valid_mask
        debug["control_confidence"] = clip01(control_confidence)
        debug.update(density_debug)
        debug.update(hold_ln_debug)
        debug.update(chord_debug)
        debug.update(jack_debug)
        debug.update(hand_debug)
        debug.update(repeat_debug)

    return {
        "time": grid,
        "X": x,
        "X_model": x,
        "feature_names": MODEL_FEATURE_NAMES,
        "model_feature_names": MODEL_FEATURE_NAMES,
        "features": features,
        "debug": debug,
    }
