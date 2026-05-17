from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from pulsefield_model.osu_core.hitobjects import ManiaHitObject
from pulsefield_model.osu_core.timing import RedTimingPoint


@dataclass
class HitObject:
    col: int
    start: float
    end: float | None = None

    def __post_init__(self) -> None:
        if self.end is None:
            self.end = self.start


@dataclass(frozen=True)
class OnsetEvent:
    t: float
    mask: int
    chord_size: int


@dataclass
class FeatureConfig:
    key_count: int = 4
    grid_step: float = 0.05
    onset_eps_min: float = 0.002
    onset_eps_beat_div: float = 768.0
    min_ln_len: float = 0.03
    density_alpha: float = 1.15
    density_L_short: float = 1.0
    density_L_med: float = 3.0
    density_mix_short: float = 0.70
    hold_L: float = 2.0
    chord_L: float = 2.0
    jack_L: float = 1.0
    jack_gap: float = 0.22
    hand_tau: float = 3.0
    repeat_L: float = 3.0
    repeat_w_exact: float = 0.20
    repeat_w_shift: float = 0.45
    repeat_w_motion: float = 0.35
    repeat_max_gap_beats: float = 4.0
    repeat_max_gap_s: float = 3.0


FEATURE_NAMES = [
    "density_env",
    "hold_occupancy",
    "chord_rate",
    "jack_risk",
    "hand_balance_ema",
    "repeat_risk",
]


def mania_hit_objects_to_control_hits(
    hitobjects: Sequence[ManiaHitObject],
    lane_base: int = 0,
) -> list[HitObject]:
    hits: list[HitObject] = []
    for hitobject in hitobjects:
        start = hitobject.start_time_ms / 1000.0
        end_ms = hitobject.end_time_ms
        end = start if end_ms is None else end_ms / 1000.0
        hits.append(
            HitObject(
                col=hitobject.lane - lane_base,
                start=start,
                end=end,
            )
        )
    return hits


def validate_control_hits(hits: Sequence[HitObject], key_count: int) -> None:
    bad_cols = [hit.col for hit in hits if not (0 <= hit.col < key_count)]
    if bad_cols:
        raise ValueError(
            f"invalid columns for key_count={key_count}: "
            f"min={min(bad_cols)}, max={max(bad_cols)}, count={len(bad_cols)}"
        )
    bad_times = [
        (hit.start, hit.end)
        for hit in hits
        if hit.end is not None and hit.end < hit.start
    ]
    if bad_times:
        raise ValueError(f"found objects with end < start: count={len(bad_times)}")


def red_timing_points_to_beat_length_fn(
    timing_points: Sequence[RedTimingPoint],
    default_beat_len: float = 0.5,
) -> Callable[[float], float]:
    points_sec = [
        (point.offset_ms / 1000.0, point.beat_length_ms / 1000.0)
        for point in timing_points
        if point.beat_length_ms > 0
    ]
    return make_piecewise_beat_length_fn(points_sec, default_beat_len=default_beat_len)


def default_beat_length_at(t: float) -> float:
    return 0.5


def make_piecewise_beat_length_fn(
    timing_points: Sequence[tuple[float, float]],
    default_beat_len: float = 0.5,
) -> Callable[[float], float]:
    pts = sorted(timing_points, key=lambda x: x[0])
    if not pts:
        return lambda t: default_beat_len
    times = np.array([p[0] for p in pts], dtype=float)
    beats = np.array([p[1] for p in pts], dtype=float)

    def beat_length_at(t: float) -> float:
        idx = int(np.searchsorted(times, t, side="right") - 1)
        if idx < 0:
            return default_beat_len
        return float(beats[idx])

    return beat_length_at


def popcount(mask: int) -> int:
    return int(mask).bit_count()


def iter_cols(mask: int, key_count: int = 4):
    for c in range(key_count):
        if mask & (1 << c):
            yield c


def mask_from_cols(cols: Sequence[int]) -> int:
    mask = 0
    for c in cols:
        mask |= 1 << c
    return mask


def center_of_mass(mask: int, key_count: int = 4) -> float:
    cols = list(iter_cols(mask, key_count))
    if not cols:
        return 0.0
    return float(sum(cols) / len(cols))


def hand_side_bucket(mask: int, key_count: int = 4) -> int:
    mid = key_count // 2
    left = sum(1 for c in iter_cols(mask, key_count) if c < mid)
    right = popcount(mask) - left
    if left > right:
        return 1
    if right > left:
        return -1
    return 0


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def make_grid(
    hits: Sequence[HitObject],
    step: float,
    start_time: float | None = None,
    end_time: float | None = None,
) -> np.ndarray:
    if step <= 0:
        raise ValueError(f"grid step must be positive: {step}")
    if not hits and (start_time is None or end_time is None):
        return np.zeros((0,), dtype=float)
    if start_time is None:
        start_time = 0.0
    if end_time is None:
        end_time = max(max(h.start, h.end if h.end is not None else h.start) for h in hits)
    if end_time < start_time:
        return np.zeros((0,), dtype=float)
    return np.arange(start_time, end_time + 0.5 * step, step, dtype=float)


def group_onsets(
    hits: Sequence[HitObject],
    beat_length_at: Callable[[float], float],
    cfg: FeatureConfig,
) -> list[OnsetEvent]:
    valid = [h for h in hits if 0 <= h.col < cfg.key_count]
    valid.sort(key=lambda h: h.start)
    groups: list[list[HitObject]] = []
    cur: list[HitObject] = []
    ref_t: float | None = None

    def eps_at(t: float) -> float:
        return max(cfg.onset_eps_min, beat_length_at(t) / cfg.onset_eps_beat_div)

    for h in valid:
        if not cur:
            cur = [h]
            ref_t = h.start
            continue
        assert ref_t is not None
        eps = max(eps_at(ref_t), eps_at(h.start))
        if h.start - ref_t <= eps:
            cur.append(h)
        else:
            groups.append(cur)
            cur = [h]
            ref_t = h.start
    if cur:
        groups.append(cur)

    events: list[OnsetEvent] = []
    for group in groups:
        t = float(np.median([h.start for h in group]))
        mask = 0
        for hit in group:
            mask |= 1 << hit.col
        chord_size = popcount(mask)
        if chord_size > 0:
            events.append(OnsetEvent(t=t, mask=mask, chord_size=chord_size))
    events.sort(key=lambda e: e.t)
    return events


def event_kernel_sum(
    grid: np.ndarray,
    event_times: Sequence[float],
    weights: Sequence[float],
    L: float,
) -> np.ndarray:
    if L <= 0:
        raise ValueError(f"kernel support must be positive: {L}")
    if len(event_times) != len(weights):
        raise ValueError(
            f"event_times and weights length mismatch: "
            f"{len(event_times)} vs {len(weights)}"
        )
    out = np.zeros_like(grid, dtype=float)
    if len(grid) == 0 or len(event_times) == 0:
        return out
    event_times_array = np.asarray(event_times, dtype=float)
    weights_array = np.asarray(weights, dtype=float)
    for center, weight in zip(event_times_array, weights_array):
        lo = int(np.searchsorted(grid, center - L, side="left"))
        hi = int(np.searchsorted(grid, center + L, side="right"))
        if hi <= lo:
            continue
        u = grid[lo:hi] - center
        kernel = np.maximum(0.0, 1.0 - np.abs(u) / L) / L
        out[lo:hi] += weight * kernel
    return out


def triangular_cdf(x, L: float):
    if L <= 0:
        raise ValueError(f"kernel support must be positive: {L}")
    x = np.asarray(x, dtype=float)
    y = np.empty_like(x, dtype=float)
    left = x <= -L
    mid_l = (x > -L) & (x < 0.0)
    mid_r = (x >= 0.0) & (x < L)
    right = x >= L
    y[left] = 0.0
    y[mid_l] = 0.5 + x[mid_l] / L + (x[mid_l] * x[mid_l]) / (2.0 * L * L)
    y[mid_r] = 0.5 + x[mid_r] / L - (x[mid_r] * x[mid_r]) / (2.0 * L * L)
    y[right] = 1.0
    return y


def interval_kernel_integral_on_grid(
    grid: np.ndarray,
    intervals: Sequence[tuple[float, float]],
    L: float,
) -> np.ndarray:
    if L <= 0:
        raise ValueError(f"kernel support must be positive: {L}")
    out = np.zeros_like(grid, dtype=float)
    if len(grid) == 0:
        return out
    for a, b in intervals:
        if b <= a:
            continue
        lo = int(np.searchsorted(grid, a - L, side="left"))
        hi = int(np.searchsorted(grid, b + L, side="right"))
        if hi <= lo:
            continue
        centers = grid[lo:hi]
        contrib = triangular_cdf(b - centers, L) - triangular_cdf(a - centers, L)
        out[lo:hi] += contrib
    return out


def merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    xs = sorted(intervals, key=lambda x: x[0])
    merged: list[tuple[float, float]] = []
    cur_a, cur_b = xs[0]
    for a, b in xs[1:]:
        if a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            merged.append((cur_a, cur_b))
            cur_a, cur_b = a, b
    merged.append((cur_a, cur_b))
    return merged


def density_env_feature(
    grid: np.ndarray,
    onsets: Sequence[OnsetEvent],
    cfg: FeatureConfig,
) -> np.ndarray:
    times = [e.t for e in onsets]
    weights = [float(e.chord_size) ** cfg.density_alpha for e in onsets]
    d_short = event_kernel_sum(grid, times, weights, cfg.density_L_short)
    d_med = event_kernel_sum(grid, times, weights, cfg.density_L_med)
    raw = cfg.density_mix_short * d_short + (1.0 - cfg.density_mix_short) * d_med
    return np.log1p(raw)


def hold_occupancy_feature(
    grid: np.ndarray,
    hits: Sequence[HitObject],
    cfg: FeatureConfig,
) -> np.ndarray:
    by_col: list[list[tuple[float, float]]] = [[] for _ in range(cfg.key_count)]
    for h in hits:
        if h.end is None:
            continue
        if not (0 <= h.col < cfg.key_count):
            continue
        if h.end - h.start >= cfg.min_ln_len:
            by_col[h.col].append((h.start, h.end))

    total_active = np.zeros_like(grid, dtype=float)
    for col in range(cfg.key_count):
        total_active += interval_kernel_integral_on_grid(grid, merge_intervals(by_col[col]), cfg.hold_L)
    return clip01(total_active / float(cfg.key_count))


def chord_rate_feature(
    grid: np.ndarray,
    onsets: Sequence[OnsetEvent],
    cfg: FeatureConfig,
) -> np.ndarray:
    times = [e.t for e in onsets]
    denom_weights = [1.0 for _ in onsets]
    numer_weights = [(e.chord_size - 1.0) / max(1.0, cfg.key_count - 1.0) for e in onsets]
    numer = event_kernel_sum(grid, times, numer_weights, cfg.chord_L)
    denom = event_kernel_sum(grid, times, denom_weights, cfg.chord_L)
    out = np.zeros_like(grid, dtype=float)
    np.divide(numer, denom, out=out, where=denom > 1e-9)
    return clip01(out)


def build_jack_pairs(
    onsets: Sequence[OnsetEvent],
    cfg: FeatureConfig,
) -> tuple[np.ndarray, np.ndarray]:
    times_by_col: list[list[float]] = [[] for _ in range(cfg.key_count)]
    for event in onsets:
        for col in iter_cols(event.mask, cfg.key_count):
            times_by_col[col].append(event.t)

    pair_times: list[float] = []
    pair_weights: list[float] = []
    for col_times in times_by_col:
        col_times.sort()
        for i in range(1, len(col_times)):
            prev_t = col_times[i - 1]
            cur_t = col_times[i]
            gap = cur_t - prev_t
            if gap <= 1e-6:
                continue
            if gap < cfg.jack_gap:
                risk = max(0.0, 1.0 - gap / cfg.jack_gap) ** 2
                pair_times.append(cur_t)
                pair_weights.append(risk)
    return np.asarray(pair_times, dtype=float), np.asarray(pair_weights, dtype=float)


def jack_risk_feature(
    grid: np.ndarray,
    onsets: Sequence[OnsetEvent],
    cfg: FeatureConfig,
) -> np.ndarray:
    pair_times, pair_weights = build_jack_pairs(onsets, cfg)
    raw = event_kernel_sum(grid, pair_times, pair_weights, cfg.jack_L)
    return np.log1p(raw)


def onset_hand_balance(mask: int, key_count: int = 4) -> float:
    mid = key_count // 2
    left = sum(1 for c in iter_cols(mask, key_count) if c < mid)
    total = popcount(mask)
    right = total - left
    if total <= 0:
        return 0.0
    return float(left - right) / float(total)


def hand_balance_ema_feature(
    grid: np.ndarray,
    onsets: Sequence[OnsetEvent],
    cfg: FeatureConfig,
) -> np.ndarray:
    if len(grid) == 0 or not onsets:
        return np.zeros_like(grid, dtype=float)
    event_times = np.asarray([e.t for e in onsets], dtype=float)
    balances = np.asarray([onset_hand_balance(e.mask, cfg.key_count) for e in onsets], dtype=float)
    ema_values = np.zeros_like(event_times, dtype=float)
    state = 0.0
    last_t: float | None = None
    for i, (t, balance) in enumerate(zip(event_times, balances)):
        if last_t is None:
            state = float(balance)
        else:
            dt = max(0.0, t - last_t)
            lam = math.exp(-dt / cfg.hand_tau)
            state = lam * state + (1.0 - lam) * float(balance)
        ema_values[i] = state
        last_t = float(t)

    idx = np.searchsorted(event_times, grid, side="right") - 1
    out = np.zeros_like(grid, dtype=float)
    valid = idx >= 0
    last_event_t = event_times[idx[valid]]
    last_event_v = ema_values[idx[valid]]
    out[valid] = last_event_v * np.exp(-(grid[valid] - last_event_t) / cfg.hand_tau)
    return np.clip(out, -1.0, 1.0)


def rhythm_bucket(
    gap: float,
    beat_len: float,
    q_min: int = -8,
    q_max: int = 8,
) -> int:
    if gap <= 1e-9 or beat_len <= 1e-9:
        return 0
    q = int(round(2.0 * math.log2(gap / beat_len)))
    return int(max(q_min, min(q_max, q)))


def movement_bucket(d: float, key_count: int = 4) -> int:
    max_bucket = 2 * (key_count - 1)
    bucket = int(round(2.0 * d))
    return int(max(-max_bucket, min(max_bucket, bucket)))


def normalize_pair_by_leftmost_active(
    prev_mask: int,
    cur_mask: int,
    key_count: int = 4,
) -> tuple[int, int]:
    union = prev_mask | cur_mask
    if union == 0:
        return 0, 0
    leftmost = min(iter_cols(union, key_count))
    return prev_mask >> leftmost, cur_mask >> leftmost


def build_repeat_tokens(
    onsets: Sequence[OnsetEvent],
    beat_length_at: Callable[[float], float],
    cfg: FeatureConfig,
) -> tuple[np.ndarray, list[tuple[Any, ...]], list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    trans_times: list[float] = []
    exact_tokens: list[tuple[Any, ...]] = []
    shift_tokens: list[tuple[Any, ...]] = []
    motion_tokens: list[tuple[Any, ...]] = []
    if len(onsets) < 2:
        return np.zeros((0,), dtype=float), exact_tokens, shift_tokens, motion_tokens

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
            continue

        q = rhythm_bucket(gap, beat_len)
        exact = (prev.mask, cur.mask, q)
        n_prev, n_cur = normalize_pair_by_leftmost_active(prev.mask, cur.mask, cfg.key_count)
        shift = (n_prev, n_cur, q)
        d = center_of_mass(cur.mask, cfg.key_count) - center_of_mass(prev.mask, cfg.key_count)
        motion = (
            movement_bucket(d, cfg.key_count),
            hand_side_bucket(prev.mask, cfg.key_count),
            hand_side_bucket(cur.mask, cfg.key_count),
            prev.chord_size,
            cur.chord_size,
            q,
        )
        trans_times.append(cur.t)
        exact_tokens.append(exact)
        shift_tokens.append(shift)
        motion_tokens.append(motion)

    return np.asarray(trans_times, dtype=float), exact_tokens, shift_tokens, motion_tokens


def concentration_score_on_grid(
    grid: np.ndarray,
    token_times: np.ndarray,
    tokens: Sequence[tuple[Any, ...]],
    L: float,
    gate_scale: float = 4.0,
) -> np.ndarray:
    if len(token_times) != len(tokens):
        raise ValueError(
            f"token_times and tokens length mismatch: "
            f"{len(token_times)} vs {len(tokens)}"
        )
    out = np.zeros_like(grid, dtype=float)
    if len(grid) == 0 or len(token_times) < 2:
        return out
    token_times = np.asarray(token_times, dtype=float)
    for gi, t in enumerate(grid):
        lo = int(np.searchsorted(token_times, t - L, side="left"))
        hi = int(np.searchsorted(token_times, t + L, side="right"))
        if hi <= lo:
            continue
        counts = defaultdict(float)
        total_w = 0.0
        total_w2 = 0.0
        for k in range(lo, hi):
            dt = abs(token_times[k] - t)
            weight = max(0.0, 1.0 - dt / L)
            if weight <= 0.0:
                continue
            counts[tokens[k]] += weight
            total_w += weight
            total_w2 += weight * weight
        if total_w2 <= 1e-12:
            continue
        n_eff = (total_w * total_w) / total_w2
        if n_eff <= 2.0:
            continue
        probs = [value / total_w for value in counts.values()]
        concentration = sum(p * p for p in probs)
        baseline = 1.0 / n_eff
        normalized = (concentration - baseline) / max(1e-9, 1.0 - baseline)
        normalized = float(np.clip(normalized, 0.0, 1.0))
        gate = 1.0 - math.exp(-(n_eff - 2.0) / gate_scale)
        out[gi] = gate * normalized
    return clip01(out)


def repeat_risk_feature(
    grid: np.ndarray,
    onsets: Sequence[OnsetEvent],
    beat_length_at: Callable[[float], float],
    cfg: FeatureConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    trans_times, exact_tokens, shift_tokens, motion_tokens = build_repeat_tokens(
        onsets,
        beat_length_at,
        cfg,
    )
    repeat_exact = concentration_score_on_grid(grid, trans_times, exact_tokens, cfg.repeat_L)
    repeat_shift = concentration_score_on_grid(grid, trans_times, shift_tokens, cfg.repeat_L)
    repeat_motion = concentration_score_on_grid(grid, trans_times, motion_tokens, cfg.repeat_L)
    repeat = (
        cfg.repeat_w_exact * repeat_exact
        + cfg.repeat_w_shift * repeat_shift
        + cfg.repeat_w_motion * repeat_motion
    )
    debug = {
        "repeat_exact": repeat_exact,
        "repeat_shift": repeat_shift,
        "repeat_motion": repeat_motion,
    }
    return clip01(repeat), debug


@dataclass(frozen=True)
class Robust01:
    lo: float
    hi: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        denom = max(1e-9, self.hi - self.lo)
        return clip01((x - self.lo) / denom)

    @staticmethod
    def fit(values: Sequence[np.ndarray], q_lo: float = 5.0, q_hi: float = 95.0) -> "Robust01":
        arrays = [np.asarray(value, dtype=float).reshape(-1) for value in values]
        if not arrays:
            return Robust01(0.0, 1.0)
        flat = np.concatenate(arrays)
        flat = flat[np.isfinite(flat)]
        if len(flat) == 0:
            return Robust01(0.0, 1.0)
        lo = float(np.percentile(flat, q_lo))
        hi = float(np.percentile(flat, q_hi))
        if hi <= lo + 1e-9:
            hi = lo + 1.0
        return Robust01(lo, hi)


def extract_control_features(
    hits: Sequence[HitObject],
    beat_length_at: Callable[[float], float] | None = None,
    cfg: FeatureConfig | None = None,
    grid: np.ndarray | None = None,
    normalizers: dict[str, Robust01] | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    return_debug: bool = True,
) -> dict[str, Any]:
    if cfg is None:
        cfg = FeatureConfig()
    if beat_length_at is None:
        beat_length_at = default_beat_length_at
    hits = list(hits)
    validate_control_hits(hits, cfg.key_count)
    if grid is None:
        grid = make_grid(hits, step=cfg.grid_step, start_time=start_time, end_time=end_time)

    onsets = group_onsets(hits, beat_length_at, cfg)
    density_env = density_env_feature(grid, onsets, cfg)
    hold_occupancy = hold_occupancy_feature(grid, hits, cfg)
    chord_rate = chord_rate_feature(grid, onsets, cfg)
    jack_risk = jack_risk_feature(grid, onsets, cfg)
    hand_balance = hand_balance_ema_feature(grid, onsets, cfg)
    repeat_risk, repeat_debug = repeat_risk_feature(grid, onsets, beat_length_at, cfg)

    if normalizers is not None:
        if "density_env" in normalizers:
            density_env = normalizers["density_env"].transform(density_env)
        if "jack_risk" in normalizers:
            jack_risk = normalizers["jack_risk"].transform(jack_risk)

    features = {
        "density_env": density_env,
        "hold_occupancy": hold_occupancy,
        "chord_rate": chord_rate,
        "jack_risk": jack_risk,
        "hand_balance_ema": hand_balance,
        "repeat_risk": repeat_risk,
    }
    if len(grid) == 0:
        x = np.zeros((0, len(FEATURE_NAMES)), dtype=float)
    else:
        x = np.stack([features[name] for name in FEATURE_NAMES], axis=1)

    debug: dict[str, Any] = {}
    if return_debug:
        debug["onsets"] = onsets
        debug.update(repeat_debug)
    return {
        "time": grid,
        "X": x,
        "feature_names": FEATURE_NAMES,
        "features": features,
        "debug": debug,
    }


def extract_raw_for_norm_fit(
    maps: Sequence[Sequence[HitObject]],
    beat_length_fns: Sequence[Callable[[float], float]] | None = None,
    cfg: FeatureConfig | None = None,
) -> dict[str, Robust01]:
    if cfg is None:
        cfg = FeatureConfig()
    density_values: list[np.ndarray] = []
    jack_values: list[np.ndarray] = []
    for i, hits in enumerate(maps):
        beat_fn = beat_length_fns[i] if beat_length_fns is not None else default_beat_length_at
        out = extract_control_features(
            hits,
            beat_length_at=beat_fn,
            cfg=cfg,
            normalizers=None,
            return_debug=False,
        )
        density_values.append(out["features"]["density_env"])
        jack_values.append(out["features"]["jack_risk"])
    return {
        "density_env": Robust01.fit(density_values, 5.0, 95.0),
        "jack_risk": Robust01.fit(jack_values, 5.0, 95.0),
    }
