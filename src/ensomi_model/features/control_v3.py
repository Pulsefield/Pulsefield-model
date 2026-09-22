from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .control import (
    HitObject,
    OnsetEvent,
    clip01,
    default_beat_length_at,
    group_onsets,
    interval_kernel_integral_on_grid,
    make_grid,
    merge_intervals,
    popcount,
    validate_control_hits,
)
from .control_v2 import (
    FeatureConfigV2,
    build_repeat_tokens,
    chord_ratio_feature,
    concentration_score_on_grid,
    confidence_from_n_eff,
    density_features,
    edge_neutralize,
    hand_balance_features,
    jack_features,
    kernel_weighted_sum_and_support,
    ln_intervals_by_col,
    valid_control_mask,
)


VALUE_FEATURE_NAMES = [
    "density_level",
    "density_burst",
    "hold_occupancy",
    "ln_change_rate_gated",
    "chord_ratio",
    "jack_excess",
    "jack_streak_exposure",
    "hand_balance_signed",
    "hand_imbalance_abs",
    "repeat_exact",
    "repeat_shift",
    "repeat_motion",
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
    "ln_change_rate_raw",
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
    "repeat_exact_top1_freq",
    "repeat_shift_top1_freq",
    "repeat_motion_top1_freq",
    "repeat_exact_pattern_variety",
    "repeat_shift_pattern_variety",
    "repeat_motion_pattern_variety",
    "repeat_confidence",
    "control_confidence",
    "valid_control_mask",
]


@dataclass
class FeatureConfigV3(FeatureConfigV2):
    ln_change_gate_lo: float = 0.25
    ln_change_gate_hi: float = 0.75


def smooth_conf_gate(
    confidence: np.ndarray,
    *,
    lo: float = 0.25,
    hi: float = 0.75,
) -> np.ndarray:
    if hi <= lo:
        raise ValueError(f"confidence gate hi must be greater than lo: {lo} >= {hi}")
    confidence = np.asarray(confidence, dtype=float)
    return clip01((confidence - lo) / (hi - lo))


def hold_and_ln_change_features(
    grid: np.ndarray,
    hits: Sequence[HitObject],
    cfg: FeatureConfigV3,
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
    ln_change_rate_raw = np.log1p(ln_change_raw)
    ln_change_confidence = confidence_from_n_eff(
        change_n_eff,
        cfg.density_n_eff_min,
        cfg.density_gate_scale,
    )
    ln_change_rate_gated = ln_change_rate_raw * smooth_conf_gate(
        ln_change_confidence,
        lo=cfg.ln_change_gate_lo,
        hi=cfg.ln_change_gate_hi,
    )
    features = {
        "hold_occupancy": hold_occupancy,
        "ln_change_rate_gated": ln_change_rate_gated,
    }
    debug = {
        "hold_active_columns": total_active,
        "ln_change_raw": ln_change_raw,
        "ln_change_rate_raw": ln_change_rate_raw,
        "ln_change_sum_w": change_w,
        "ln_change_sum_w2": change_w2,
        "ln_change_n_eff": change_n_eff,
        "ln_change_confidence": ln_change_confidence,
    }
    return features, debug


def repeat_features(
    grid: np.ndarray,
    onsets: Sequence[OnsetEvent],
    beat_length_at: Callable[[float], float],
    cfg: FeatureConfigV3,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    trans_times, exact_tokens, shift_tokens, motion_tokens, _rhythm_tokens = build_repeat_tokens(
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
    features = {
        "repeat_exact": exact,
        "repeat_shift": shift,
        "repeat_motion": motion,
    }
    repeat_confidence = np.mean(
        np.stack(
            [
                exact_debug["confidence"],
                shift_debug["confidence"],
                motion_debug["confidence"],
            ],
            axis=0,
        ),
        axis=0,
    )
    debug = {
        "repeat_exact_n_eff": exact_debug["n_eff"],
        "repeat_shift_n_eff": shift_debug["n_eff"],
        "repeat_motion_n_eff": motion_debug["n_eff"],
        "repeat_exact_top1_freq": exact_debug["top1_freq"],
        "repeat_shift_top1_freq": shift_debug["top1_freq"],
        "repeat_motion_top1_freq": motion_debug["top1_freq"],
        "repeat_exact_top_token": exact_debug["top_token"],
        "repeat_shift_top_token": shift_debug["top_token"],
        "repeat_motion_top_token": motion_debug["top_token"],
        "repeat_exact_pattern_variety": exact_debug["pattern_variety"],
        "repeat_shift_pattern_variety": shift_debug["pattern_variety"],
        "repeat_motion_pattern_variety": motion_debug["pattern_variety"],
        "repeat_confidence": repeat_confidence,
    }
    return features, debug


def extract_control_features(
    hits: Sequence[HitObject],
    beat_length_at: Callable[[float], float] | None = None,
    cfg: FeatureConfigV3 | None = None,
    grid: np.ndarray | None = None,
    normalizers: dict[str, Robust01] | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    return_debug: bool = True,
) -> dict[str, Any]:
    if cfg is None:
        cfg = FeatureConfigV3()
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
            "ln_change_rate_gated",
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
        jack_debug["jack_streak_confidence"],
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
