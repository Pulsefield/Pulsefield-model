from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any

from pulsefield_model.timing.schema import FittedTimingGrid, TimingSegment


DEFAULT_MIN_BPM = 20.0
DEFAULT_MAX_BPM = 1000.0
DEFAULT_MAX_RELATIVE_BPM_ADJUSTMENT = 0.05

PROJECTION_METHOD_PRESERVE_ANCHORS = "preserve_anchors"
PROJECTION_METHOD_PRESERVE_BPM = "preserve_bpm"

REASON_EMPTY_FEASIBLE_COUNT_RANGE = "empty_feasible_count_range"
REASON_COUNT_OUTSIDE_FEASIBLE_RANGE = "count_outside_feasible_range"
REASON_RELATIVE_BPM_ADJUSTMENT_EXCEEDED = "relative_bpm_adjustment_exceeded"
REASON_SOURCE_BPM_OUT_OF_RANGE = "source_bpm_out_of_range"
REASON_SOURCE_INTERVAL_INVALID = "source_interval_invalid"
REASON_SCHEMA_UNAVAILABLE = "timing_v3_schema_unavailable"
REASON_SCHEMA_CONSTRUCTION_FAILED = "timing_v3_schema_construction_failed"


@dataclass(frozen=True)
class TimingV3ProjectionBoundaryDiagnostic:
    boundary_index: int
    source_left_anchor_ms: float
    source_right_anchor_ms: float
    projected_left_anchor_ms: float
    projected_right_anchor_ms: float
    left_anchor_displacement_ms: float
    right_anchor_displacement_ms: float
    implied_beat_count: float
    beat_count: int
    residual_ms: float
    source_left_bpm: float
    source_right_bpm: float
    projected_left_bpm: float
    relative_bpm_adjustment: float
    feasible_min_beat_count: int
    feasible_max_beat_count: int
    failure_reason: str | None = None


@dataclass(frozen=True)
class TimingV3ProjectionDiagnostics:
    method: str
    coverage_start_ms: float
    coverage_end_ms: float
    source_section_count: int
    projected_section_count: int
    source_origin_beat: int
    source_origin_time_ms: float
    grid_origin_beat: int | None
    grid_origin_time_ms: float | None
    first_start_beat: int | None
    last_end_beat: int | None
    boundary_diagnostics: tuple[TimingV3ProjectionBoundaryDiagnostic, ...]
    failure_source_section_index: int | None = None
    failure_reason: str | None = None


@dataclass(frozen=True)
class TimingV3ProjectionResult:
    method: str
    grid: Any | None
    diagnostics: TimingV3ProjectionDiagnostics
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.grid is not None


@dataclass(frozen=True)
class _ProjectionInputs:
    coverage_start_ms: float
    coverage_end_ms: float
    min_bpm: float
    max_bpm: float
    max_relative_bpm_adjustment: float


@dataclass(frozen=True)
class _SectionSpec:
    start_beat: int
    end_beat: int
    bpm: float


@dataclass(frozen=True)
class _IntervalProjection:
    beat_count: int
    projected_bpm: float
    projected_period_ms: float
    projected_right_anchor_ms: float
    diagnostic: TimingV3ProjectionBoundaryDiagnostic
    failure_reason: str | None


def project_preserve_anchors(
    source_grid: FittedTimingGrid,
    *,
    coverage_start_ms: float = 0.0,
    coverage_end_ms: float | None = None,
    frame_count: int | None = None,
    frame_rate_hz: float | None = None,
    min_bpm: float = DEFAULT_MIN_BPM,
    max_bpm: float = DEFAULT_MAX_BPM,
    max_relative_bpm_adjustment: float = DEFAULT_MAX_RELATIVE_BPM_ADJUSTMENT,
) -> TimingV3ProjectionResult:
    """Project a v2 grid to Family B: preserve anchors, adjust left BPMs.

    A projection failure returns ``TimingV3ProjectionResult(grid=None, reason=...)``
    with diagnostics, so callers can explicitly tag a v2 fallback instead of
    treating a failed adapter result as a valid v3 grid.
    """

    inputs = _projection_inputs(
        coverage_start_ms=coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
        frame_count=frame_count,
        frame_rate_hz=frame_rate_hz,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        max_relative_bpm_adjustment=max_relative_bpm_adjustment,
    )
    return _project(
        source_grid,
        method=PROJECTION_METHOD_PRESERVE_ANCHORS,
        inputs=inputs,
    )


def project_preserve_bpm(
    source_grid: FittedTimingGrid,
    *,
    coverage_start_ms: float = 0.0,
    coverage_end_ms: float | None = None,
    frame_count: int | None = None,
    frame_rate_hz: float | None = None,
    min_bpm: float = DEFAULT_MIN_BPM,
    max_bpm: float = DEFAULT_MAX_BPM,
    max_relative_bpm_adjustment: float = DEFAULT_MAX_RELATIVE_BPM_ADJUSTMENT,
) -> TimingV3ProjectionResult:
    """Project a v2 grid to Family A: preserve BPMs, move later anchors.

    This is an explicit whole-grid control projection. It never switches to
    Family B behavior for individual boundaries.
    """

    inputs = _projection_inputs(
        coverage_start_ms=coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
        frame_count=frame_count,
        frame_rate_hz=frame_rate_hz,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        max_relative_bpm_adjustment=max_relative_bpm_adjustment,
    )
    return _project(
        source_grid,
        method=PROJECTION_METHOD_PRESERVE_BPM,
        inputs=inputs,
    )


def _project(
    source_grid: FittedTimingGrid,
    *,
    method: str,
    inputs: _ProjectionInputs,
) -> TimingV3ProjectionResult:
    segments = tuple(source_grid.segments)
    origin_beat = 0
    origin_time_ms = float(segments[0].offset_ms)
    source_bpms = tuple(_bpm(segment) for segment in segments)
    for section_index, source_bpm in enumerate(source_bpms):
        if not _bpm_is_in_range(source_bpm, min_bpm=inputs.min_bpm, max_bpm=inputs.max_bpm):
            return _failure_result(
                method=method,
                inputs=inputs,
                source_section_count=len(segments),
                source_origin_beat=origin_beat,
                source_origin_time_ms=origin_time_ms,
                boundary_diagnostics=(),
                reason=REASON_SOURCE_BPM_OUT_OF_RANGE,
                failure_source_section_index=section_index,
            )
    for section_index, (left, right) in enumerate(zip(segments, segments[1:])):
        delta_ms = float(right.offset_ms - left.offset_ms)
        if not math.isfinite(delta_ms) or delta_ms <= 0.0:
            return _failure_result(
                method=method,
                inputs=inputs,
                source_section_count=len(segments),
                source_origin_beat=origin_beat,
                source_origin_time_ms=origin_time_ms,
                boundary_diagnostics=(),
                reason=REASON_SOURCE_INTERVAL_INVALID,
                failure_source_section_index=section_index,
            )
    anchor_beats = [origin_beat]
    projected_anchor_times_ms = [origin_time_ms]
    section_bpms: list[float] = []
    boundary_diagnostics: list[TimingV3ProjectionBoundaryDiagnostic] = []

    for boundary_index, (left, right) in enumerate(zip(segments, segments[1:])):
        interval = _project_interval(
            boundary_index,
            left,
            right,
            projected_left_anchor_ms=projected_anchor_times_ms[-1],
            method=method,
            inputs=inputs,
        )
        boundary_diagnostics.append(interval.diagnostic)
        if interval.failure_reason is not None:
            return _failure_result(
                method=method,
                inputs=inputs,
                source_section_count=len(segments),
                source_origin_beat=origin_beat,
                source_origin_time_ms=origin_time_ms,
                boundary_diagnostics=tuple(boundary_diagnostics),
                reason=interval.failure_reason,
                failure_source_section_index=boundary_index,
            )
        anchor_beats.append(anchor_beats[-1] + interval.beat_count)
        projected_anchor_times_ms.append(interval.projected_right_anchor_ms)
        section_bpms.append(interval.projected_bpm)

    last_source_bpm = source_bpms[-1]
    section_bpms.append(last_source_bpm)

    first_period_ms = 60000.0 / section_bpms[0]
    first_start_beat = _first_start_beat(
        origin_beat=origin_beat,
        origin_time_ms=origin_time_ms,
        period_ms=first_period_ms,
        coverage_start_ms=inputs.coverage_start_ms,
    )
    grid_origin_beat = origin_beat
    grid_origin_time_ms = origin_time_ms
    last_start_beat = anchor_beats[-1]
    last_anchor_time_ms = projected_anchor_times_ms[-1]
    last_period_ms = 60000.0 / section_bpms[-1]
    last_end_beat = _last_end_beat(
        last_start_beat=last_start_beat,
        last_anchor_time_ms=last_anchor_time_ms,
        period_ms=last_period_ms,
        coverage_end_ms=inputs.coverage_end_ms,
    )
    section_specs = _section_specs(
        anchor_beats=tuple(anchor_beats),
        section_bpms=tuple(section_bpms),
        first_start_beat=first_start_beat,
        last_end_beat=last_end_beat,
    )

    grid, reason = _build_timing_v3_grid(
        section_specs,
        origin_beat=grid_origin_beat,
        origin_time_ms=grid_origin_time_ms,
        coverage_start_ms=inputs.coverage_start_ms,
        coverage_end_ms=inputs.coverage_end_ms,
    )
    diagnostics = TimingV3ProjectionDiagnostics(
        method=method,
        coverage_start_ms=inputs.coverage_start_ms,
        coverage_end_ms=inputs.coverage_end_ms,
        source_section_count=len(segments),
        projected_section_count=len(section_specs),
        source_origin_beat=origin_beat,
        source_origin_time_ms=origin_time_ms,
        grid_origin_beat=grid_origin_beat,
        grid_origin_time_ms=grid_origin_time_ms,
        first_start_beat=first_start_beat,
        last_end_beat=last_end_beat,
        boundary_diagnostics=tuple(boundary_diagnostics),
        failure_source_section_index=None,
        failure_reason=reason,
    )
    return TimingV3ProjectionResult(
        method=method,
        grid=grid,
        diagnostics=diagnostics,
        reason=reason,
    )


def _project_interval(
    boundary_index: int,
    left: TimingSegment,
    right: TimingSegment,
    *,
    projected_left_anchor_ms: float,
    method: str,
    inputs: _ProjectionInputs,
) -> _IntervalProjection:
    delta_ms = float(right.offset_ms - left.offset_ms)
    source_period_ms = float(left.beat_length_ms)
    implied_beat_count = delta_ms / source_period_ms
    beat_count = max(1, math.floor(implied_beat_count + 0.5))
    residual_ms = delta_ms - beat_count * source_period_ms
    feasible_min_beat_count, feasible_max_beat_count = _feasible_beat_count_range(
        delta_ms,
        min_bpm=inputs.min_bpm,
        max_bpm=inputs.max_bpm,
    )

    if method == PROJECTION_METHOD_PRESERVE_ANCHORS:
        projected_period_ms = delta_ms / beat_count
        projected_bpm = 60000.0 / projected_period_ms
        relative_bpm_adjustment = (projected_bpm - _bpm(left)) / _bpm(left)
    elif method == PROJECTION_METHOD_PRESERVE_BPM:
        projected_period_ms = source_period_ms
        projected_bpm = _bpm(left)
        relative_bpm_adjustment = 0.0
    else:
        raise ValueError(f"unsupported projection method {method!r}")

    projected_right_anchor_ms = projected_left_anchor_ms + beat_count * projected_period_ms
    failure_reason = _interval_failure_reason(
        beat_count=beat_count,
        projected_bpm=projected_bpm,
        relative_bpm_adjustment=relative_bpm_adjustment,
        feasible_min_beat_count=feasible_min_beat_count,
        feasible_max_beat_count=feasible_max_beat_count,
        method=method,
        inputs=inputs,
    )
    diagnostic = TimingV3ProjectionBoundaryDiagnostic(
        boundary_index=boundary_index,
        source_left_anchor_ms=float(left.offset_ms),
        source_right_anchor_ms=float(right.offset_ms),
        projected_left_anchor_ms=float(projected_left_anchor_ms),
        projected_right_anchor_ms=float(projected_right_anchor_ms),
        left_anchor_displacement_ms=float(projected_left_anchor_ms - left.offset_ms),
        right_anchor_displacement_ms=float(projected_right_anchor_ms - right.offset_ms),
        implied_beat_count=float(implied_beat_count),
        beat_count=int(beat_count),
        residual_ms=float(residual_ms),
        source_left_bpm=float(_bpm(left)),
        source_right_bpm=float(_bpm(right)),
        projected_left_bpm=float(projected_bpm),
        relative_bpm_adjustment=float(relative_bpm_adjustment),
        feasible_min_beat_count=int(feasible_min_beat_count),
        feasible_max_beat_count=int(feasible_max_beat_count),
        failure_reason=failure_reason,
    )
    return _IntervalProjection(
        beat_count=beat_count,
        projected_bpm=projected_bpm,
        projected_period_ms=projected_period_ms,
        projected_right_anchor_ms=projected_right_anchor_ms,
        diagnostic=diagnostic,
        failure_reason=failure_reason,
    )


def _interval_failure_reason(
    *,
    beat_count: int,
    projected_bpm: float,
    relative_bpm_adjustment: float,
    feasible_min_beat_count: int,
    feasible_max_beat_count: int,
    method: str,
    inputs: _ProjectionInputs,
) -> str | None:
    if feasible_min_beat_count > feasible_max_beat_count:
        return REASON_EMPTY_FEASIBLE_COUNT_RANGE
    if beat_count < feasible_min_beat_count or beat_count > feasible_max_beat_count:
        return REASON_COUNT_OUTSIDE_FEASIBLE_RANGE
    if not _bpm_is_in_range(projected_bpm, min_bpm=inputs.min_bpm, max_bpm=inputs.max_bpm):
        return REASON_SOURCE_BPM_OUT_OF_RANGE
    if (
        method == PROJECTION_METHOD_PRESERVE_ANCHORS
        and abs(relative_bpm_adjustment) > inputs.max_relative_bpm_adjustment
    ):
        return REASON_RELATIVE_BPM_ADJUSTMENT_EXCEEDED
    return None


def _projection_inputs(
    *,
    coverage_start_ms: float,
    coverage_end_ms: float | None,
    frame_count: int | None,
    frame_rate_hz: float | None,
    min_bpm: float,
    max_bpm: float,
    max_relative_bpm_adjustment: float,
) -> _ProjectionInputs:
    if coverage_end_ms is None:
        if frame_count is None or frame_rate_hz is None:
            raise ValueError("coverage_end_ms or frame_count/frame_rate_hz is required")
        if isinstance(frame_count, bool) or frame_count <= 0:
            raise ValueError(f"frame_count must be positive, got {frame_count!r}")
        if not math.isfinite(frame_rate_hz) or frame_rate_hz <= 0.0:
            raise ValueError(f"frame_rate_hz must be positive and finite, got {frame_rate_hz!r}")
        coverage_end_ms = 1000.0 * frame_count / frame_rate_hz
    elif frame_count is not None or frame_rate_hz is not None:
        raise ValueError("coverage_end_ms is mutually exclusive with frame_count/frame_rate_hz")

    if not math.isfinite(coverage_start_ms):
        raise ValueError(f"coverage_start_ms must be finite, got {coverage_start_ms!r}")
    if not math.isfinite(coverage_end_ms) or coverage_end_ms <= coverage_start_ms:
        raise ValueError(
            "coverage_end_ms must be finite and greater than coverage_start_ms, "
            f"got {coverage_end_ms!r} <= {coverage_start_ms!r}",
        )
    if not math.isfinite(min_bpm) or min_bpm <= 0.0:
        raise ValueError(f"min_bpm must be positive and finite, got {min_bpm!r}")
    if not math.isfinite(max_bpm) or max_bpm < min_bpm:
        raise ValueError(f"max_bpm must be finite and >= min_bpm, got {max_bpm!r}")
    if not math.isfinite(max_relative_bpm_adjustment) or max_relative_bpm_adjustment < 0.0:
        raise ValueError(
            "max_relative_bpm_adjustment must be non-negative and finite, "
            f"got {max_relative_bpm_adjustment!r}",
        )
    return _ProjectionInputs(
        coverage_start_ms=float(coverage_start_ms),
        coverage_end_ms=float(coverage_end_ms),
        min_bpm=float(min_bpm),
        max_bpm=float(max_bpm),
        max_relative_bpm_adjustment=float(max_relative_bpm_adjustment),
    )


def _section_specs(
    *,
    anchor_beats: tuple[int, ...],
    section_bpms: tuple[float, ...],
    first_start_beat: int,
    last_end_beat: int,
) -> tuple[_SectionSpec, ...]:
    specs: list[_SectionSpec] = []
    if len(section_bpms) == 1:
        return (_SectionSpec(start_beat=first_start_beat, end_beat=last_end_beat, bpm=section_bpms[0]),)

    for index, bpm in enumerate(section_bpms):
        if index == 0:
            start_beat = first_start_beat
            end_beat = anchor_beats[1]
        elif index == len(section_bpms) - 1:
            start_beat = anchor_beats[index]
            end_beat = last_end_beat
        else:
            start_beat = anchor_beats[index]
            end_beat = anchor_beats[index + 1]
        if end_beat <= start_beat:
            raise ValueError(f"projected section {index} is empty: [{start_beat}, {end_beat})")
        specs.append(_SectionSpec(start_beat=start_beat, end_beat=end_beat, bpm=bpm))
    return tuple(specs)


def _first_start_beat(
    *,
    origin_beat: int,
    origin_time_ms: float,
    period_ms: float,
    coverage_start_ms: float,
) -> int:
    beat = origin_beat + math.floor((coverage_start_ms - origin_time_ms) / period_ms)
    return min(origin_beat, int(beat))


def _last_end_beat(
    *,
    last_start_beat: int,
    last_anchor_time_ms: float,
    period_ms: float,
    coverage_end_ms: float,
) -> int:
    beat_delta = math.ceil((coverage_end_ms - last_anchor_time_ms) / period_ms)
    return max(last_start_beat + 1, last_start_beat + int(beat_delta))


def _feasible_beat_count_range(
    delta_ms: float,
    *,
    min_bpm: float,
    max_bpm: float,
) -> tuple[int, int]:
    min_count = max(1, math.ceil(min_bpm * delta_ms / 60000.0))
    max_count = math.floor(max_bpm * delta_ms / 60000.0)
    return int(min_count), int(max_count)


def _bpm(segment: TimingSegment) -> float:
    return 60000.0 / float(segment.beat_length_ms)


def _bpm_is_in_range(bpm: float, *, min_bpm: float, max_bpm: float) -> bool:
    return math.isfinite(bpm) and min_bpm <= bpm <= max_bpm


def _build_timing_v3_grid(
    section_specs: tuple[_SectionSpec, ...],
    *,
    origin_beat: int,
    origin_time_ms: float,
    coverage_start_ms: float,
    coverage_end_ms: float,
) -> tuple[Any | None, str | None]:
    try:
        schema_module = importlib.import_module("pulsefield_model.timing.v3.schema")
    except ModuleNotFoundError as exc:
        if exc.name == "pulsefield_model.timing.v3.schema":
            return None, REASON_SCHEMA_UNAVAILABLE
        raise

    try:
        section_cls = schema_module.ConstantTimingSection
        grid_cls = schema_module.TimingV3Grid
        sections = tuple(
            section_cls(start_beat=spec.start_beat, end_beat=spec.end_beat, bpm=spec.bpm)
            for spec in section_specs
        )
        grid = grid_cls(
            origin_beat=origin_beat,
            origin_time_ms=origin_time_ms,
            sections=sections,
            coverage_start_ms=coverage_start_ms,
            coverage_end_ms=coverage_end_ms,
        )
    except (AttributeError, TypeError, ValueError):
        return None, REASON_SCHEMA_CONSTRUCTION_FAILED
    return grid, None


def _failure_result(
    *,
    method: str,
    inputs: _ProjectionInputs,
    source_section_count: int,
    source_origin_beat: int,
    source_origin_time_ms: float,
    boundary_diagnostics: tuple[TimingV3ProjectionBoundaryDiagnostic, ...],
    reason: str,
    failure_source_section_index: int | None = None,
) -> TimingV3ProjectionResult:
    diagnostics = TimingV3ProjectionDiagnostics(
        method=method,
        coverage_start_ms=inputs.coverage_start_ms,
        coverage_end_ms=inputs.coverage_end_ms,
        source_section_count=source_section_count,
        projected_section_count=0,
        source_origin_beat=source_origin_beat,
        source_origin_time_ms=source_origin_time_ms,
        grid_origin_beat=None,
        grid_origin_time_ms=None,
        first_start_beat=None,
        last_end_beat=None,
        boundary_diagnostics=boundary_diagnostics,
        failure_source_section_index=failure_source_section_index,
        failure_reason=reason,
    )
    return TimingV3ProjectionResult(method=method, grid=None, diagnostics=diagnostics, reason=reason)


__all__ = [
    "DEFAULT_MAX_BPM",
    "DEFAULT_MAX_RELATIVE_BPM_ADJUSTMENT",
    "DEFAULT_MIN_BPM",
    "PROJECTION_METHOD_PRESERVE_ANCHORS",
    "PROJECTION_METHOD_PRESERVE_BPM",
    "REASON_COUNT_OUTSIDE_FEASIBLE_RANGE",
    "REASON_EMPTY_FEASIBLE_COUNT_RANGE",
    "REASON_RELATIVE_BPM_ADJUSTMENT_EXCEEDED",
    "REASON_SCHEMA_CONSTRUCTION_FAILED",
    "REASON_SCHEMA_UNAVAILABLE",
    "REASON_SOURCE_BPM_OUT_OF_RANGE",
    "REASON_SOURCE_INTERVAL_INVALID",
    "TimingV3ProjectionBoundaryDiagnostic",
    "TimingV3ProjectionDiagnostics",
    "TimingV3ProjectionResult",
    "project_preserve_anchors",
    "project_preserve_bpm",
]
