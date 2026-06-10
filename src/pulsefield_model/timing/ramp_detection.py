from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from pulsefield_model.timing.schema import FittedTimingGrid, TimingSegment


def _require_nonnegative_finite(config: object, fields: tuple[str, ...]) -> None:
    for field in fields:
        value = getattr(config, field)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{field} must be non-negative and finite, got {value!r}")


@dataclass(frozen=True)
class TimingRampDetectionConfig:
    monotonic_eps_bpm: float = 1e-6
    max_continuous_gap_s: float = 20.0
    min_ramp_points: int = 8
    min_ramp_bpm_span: float = 80.0
    min_ramp_duration_s: float = 4.0
    max_continuous_p90_gap_s: float = 20.0
    min_continuous_point_density_per_s: float = 0.08
    smooth_min_linear_r2: float = 0.45
    smooth_max_jumpiness: float = 0.25
    spike_jumpiness_threshold: float = 0.65
    spike_max_linear_r2: float = 0.35

    def __post_init__(self) -> None:
        _require_nonnegative_finite(
            self,
            (
                "monotonic_eps_bpm",
                "max_continuous_gap_s",
                "min_ramp_bpm_span",
                "min_ramp_duration_s",
                "max_continuous_p90_gap_s",
                "min_continuous_point_density_per_s",
                "smooth_min_linear_r2",
                "smooth_max_jumpiness",
                "spike_jumpiness_threshold",
                "spike_max_linear_r2",
            ),
        )
        if self.min_ramp_points <= 1:
            raise ValueError(f"min_ramp_points must be greater than 1, got {self.min_ramp_points!r}")


@dataclass(frozen=True)
class TimingRampRun:
    start_index: int
    end_index: int
    length: int
    start_ms: float
    end_ms: float
    duration_s: float
    start_bpm: float
    end_bpm: float
    bpm_span: float
    direction: str
    linear_r2: float
    point_density_per_s: float
    median_gap_s: float
    p90_gap_s: float
    max_gap_s: float
    median_abs_delta_bpm: float
    p90_abs_delta_bpm: float
    max_abs_delta_bpm: float
    jumpiness: float
    big_jump_count: int
    big_jump_fraction: float
    point_coverage: float
    span_coverage: float
    score: float


@dataclass(frozen=True)
class TimingRampDetection:
    is_ramp: bool
    family: str
    score: float
    reasons: tuple[str, ...]
    best_run: TimingRampRun | None
    segment_count: int


def detect_timing_ramp(
    grid: FittedTimingGrid,
    *,
    config: TimingRampDetectionConfig = TimingRampDetectionConfig(),
) -> TimingRampDetection:
    """Classify whether a timing grid contains a continuous BPM ramp shape."""

    runs = _monotonic_runs(grid.segments, config=config)
    best_run = max(runs, key=_run_rank_key) if runs else None
    if best_run is None:
        return TimingRampDetection(
            is_ramp=False,
            family="none",
            score=0.0,
            reasons=("no_timing_segments",),
            best_run=None,
            segment_count=len(grid.segments),
        )

    reasons = _rejection_reasons(best_run, config=config)
    is_ramp = not reasons
    if is_ramp:
        reasons = ("long_continuous_monotonic_ramp",)
    return TimingRampDetection(
        is_ramp=is_ramp,
        family=_ramp_family(best_run),
        score=float(best_run.score),
        reasons=tuple(reasons),
        best_run=best_run,
        segment_count=len(grid.segments),
    )


def _monotonic_runs(
    segments: Sequence[TimingSegment],
    *,
    config: TimingRampDetectionConfig,
) -> list[TimingRampRun]:
    if not segments:
        return []

    offsets_ms = np.asarray([segment.offset_ms for segment in segments], dtype=np.float64)
    bpms = np.asarray([segment.local_bpm for segment in segments], dtype=np.float64)
    if bpms.size == 1:
        return [_run_payload(offsets_ms, bpms, 0, 0, config=config)]

    runs: list[TimingRampRun] = []
    for direction in (1.0, -1.0):
        start = 0
        for index in range(1, bpms.size):
            diff = float(bpms[index] - bpms[index - 1])
            if abs(diff) <= config.monotonic_eps_bpm or diff * direction > 0.0:
                continue
            runs.extend(_split_run_by_gap(offsets_ms, bpms, start, index - 1, config=config))
            start = index
        runs.extend(_split_run_by_gap(offsets_ms, bpms, start, bpms.size - 1, config=config))
    return runs


def _split_run_by_gap(
    offsets_ms: np.ndarray,
    bpms: np.ndarray,
    start: int,
    end: int,
    *,
    config: TimingRampDetectionConfig,
) -> list[TimingRampRun]:
    if end < start:
        return []

    split_starts = [start]
    for index in range(start + 1, end + 1):
        gap_s = float(offsets_ms[index] - offsets_ms[index - 1]) / 1000.0
        if gap_s > config.max_continuous_gap_s:
            split_starts.append(index)

    return [
        _run_payload(offsets_ms, bpms, run_start, next_start - 1, config=config)
        for run_start, next_start in zip(split_starts, split_starts[1:] + [end + 1])
        if next_start - 1 >= run_start
    ]


def _run_payload(
    offsets_ms: np.ndarray,
    bpms: np.ndarray,
    start: int,
    end: int,
    *,
    config: TimingRampDetectionConfig,
) -> TimingRampRun:
    run_offsets_ms = offsets_ms[start : end + 1]
    run_bpms = bpms[start : end + 1]
    gaps_s = np.diff(run_offsets_ms) / 1000.0
    deltas_bpm = np.diff(run_bpms)
    abs_deltas_bpm = np.abs(deltas_bpm[np.abs(deltas_bpm) > config.monotonic_eps_bpm])

    duration_s = max(0.0, float(run_offsets_ms[-1] - run_offsets_ms[0]) / 1000.0)
    start_bpm = float(run_bpms[0])
    end_bpm = float(run_bpms[-1])
    bpm_delta = end_bpm - start_bpm
    bpm_span = abs(bpm_delta)
    direction = (
        "up"
        if bpm_delta > config.monotonic_eps_bpm
        else "down"
        if bpm_delta < -config.monotonic_eps_bpm
        else "flat"
    )
    big_jump_threshold = max(25.0, bpm_span * 0.20)
    big_jump_count = int(np.sum(abs_deltas_bpm > big_jump_threshold)) if abs_deltas_bpm.size else 0
    big_jump_fraction = float(big_jump_count / abs_deltas_bpm.size) if abs_deltas_bpm.size else 0.0
    whole_span = float(np.max(bpms) - np.min(bpms)) if bpms.size else 0.0

    run = TimingRampRun(
        start_index=int(start),
        end_index=int(end),
        length=int(run_bpms.size),
        start_ms=float(run_offsets_ms[0]),
        end_ms=float(run_offsets_ms[-1]),
        duration_s=duration_s,
        start_bpm=start_bpm,
        end_bpm=end_bpm,
        bpm_span=float(bpm_span),
        direction=direction,
        linear_r2=_linear_r2(run_offsets_ms / 1000.0, run_bpms, eps=config.monotonic_eps_bpm),
        point_density_per_s=float(run_bpms.size / duration_s) if duration_s > 0.0 else 0.0,
        median_gap_s=_percentile_or_zero(gaps_s, 50.0),
        p90_gap_s=_percentile_or_zero(gaps_s, 90.0),
        max_gap_s=float(np.max(gaps_s)) if gaps_s.size else 0.0,
        median_abs_delta_bpm=_percentile_or_zero(abs_deltas_bpm, 50.0),
        p90_abs_delta_bpm=_percentile_or_zero(abs_deltas_bpm, 90.0),
        max_abs_delta_bpm=float(np.max(abs_deltas_bpm)) if abs_deltas_bpm.size else 0.0,
        jumpiness=big_jump_fraction,
        big_jump_count=big_jump_count,
        big_jump_fraction=big_jump_fraction,
        point_coverage=float(run_bpms.size / bpms.size) if bpms.size else 0.0,
        span_coverage=float(bpm_span / whole_span) if whole_span > config.monotonic_eps_bpm else 0.0,
        score=0.0,
    )
    return replace(run, score=_real_ramp_score(run))


def _rejection_reasons(
    run: TimingRampRun,
    *,
    config: TimingRampDetectionConfig,
) -> list[str]:
    reasons: list[str] = []
    if run.length < config.min_ramp_points:
        reasons.append("too_few_points")
    if run.bpm_span < config.min_ramp_bpm_span:
        reasons.append("low_bpm_span")
    if run.duration_s < config.min_ramp_duration_s:
        reasons.append("too_short_duration")
    if (
        run.p90_gap_s > config.max_continuous_p90_gap_s
        and run.point_density_per_s < config.min_continuous_point_density_per_s
    ):
        reasons.append("temporally_disconnected")
    if (
        run.jumpiness > config.spike_jumpiness_threshold
        and run.linear_r2 < config.spike_max_linear_r2
    ):
        reasons.append("spike_like_jumps")
    if run.linear_r2 < config.smooth_min_linear_r2 and run.jumpiness > config.smooth_max_jumpiness:
        reasons.append("weak_ramp_smoothness")
    return reasons


def _ramp_family(run: TimingRampRun) -> str:
    if run.length < 6 or run.bpm_span < 60.0:
        return "short_or_low_span"
    if run.linear_r2 >= 0.85:
        return "linear_ramp"
    if run.linear_r2 >= 0.45:
        return "stepped_ramp"
    if run.jumpiness <= 0.25:
        return "curved_or_exponential_ramp"
    return "spiky_monotonic_run"


def _run_rank_key(run: TimingRampRun) -> tuple[float, float, int, float]:
    return (
        float(run.score),
        float(run.bpm_span),
        int(run.length),
        float(run.duration_s),
    )


def _real_ramp_score(run: TimingRampRun) -> float:
    if run.bpm_span <= 0.0 or run.length <= 1 or run.duration_s <= 0.0:
        return 0.0

    duration_factor = min(1.0, run.duration_s / 8.0)
    r2_factor = 0.40 + 0.60 * run.linear_r2
    continuity_factor = 1.0 if run.p90_gap_s <= 15.0 else max(0.25, 15.0 / run.p90_gap_s)
    jump_factor = 1.0 if run.jumpiness <= 0.25 else max(0.20, 1.0 - (run.jumpiness - 0.25) / 0.75)
    coverage_factor = math.sqrt(max(0.0, min(1.0, run.span_coverage)))
    return float(
        run.bpm_span
        * math.log1p(run.length)
        * duration_factor
        * r2_factor
        * continuity_factor
        * jump_factor
        * coverage_factor
    )


def _linear_r2(times_s: np.ndarray, bpms: np.ndarray, *, eps: float) -> float:
    if times_s.size < 2:
        return 0.0
    if float(np.max(times_s) - np.min(times_s)) <= 0.0:
        return 0.0
    if float(np.max(bpms) - np.min(bpms)) <= eps:
        return 0.0
    slope, intercept = np.polyfit(times_s, bpms, 1)
    predicted = slope * times_s + intercept
    residual = float(np.sum((bpms - predicted) ** 2))
    total = float(np.sum((bpms - float(np.mean(bpms))) ** 2))
    if total <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - residual / total)))


def _percentile_or_zero(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else 0.0
