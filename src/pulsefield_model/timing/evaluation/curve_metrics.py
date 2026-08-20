from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from pulsefield_model.timing.evaluation.exp004_metrics import (
    PHASE_SAMPLING_HOP_MS,
    alias_aware_bpm_error,
    extract_tempo_change_boundaries,
    match_weak_boundaries,
    phase_sample_times_v1,
)
from pulsefield_model.timing.schema import FittedTimingGrid
from pulsefield_model.timing.v3.analytic_curve import PhaseContinuousTimingCurve
from pulsefield_model.timing.v3.schema import TimingV3Grid


WEAK_ORACLE_CURVE_METRICS_SCHEMA = "pulsefield_model.timing_v3_weak_oracle_curve_metrics_v1"

CURVE_CLASS_CONSTANT = "constant"
CURVE_CLASS_JUMP = "jump"
CURVE_CLASS_RAMP = "ramp"
WEAK_ORACLE_CLASS_CONSTANT = "constant"
WEAK_ORACLE_CLASS_JUMP = "jump"
WEAK_ORACLE_CLASS_RAMP_LIKE = "ramp_like"

CurveClass = Literal["constant", "jump", "ramp"]
WeakOracleClass = Literal["constant", "jump", "ramp_like"]


@runtime_checkable
class _AnalyticCurve(Protocol):
    @property
    def start_time_ms(self) -> float: ...

    @property
    def end_time_ms(self) -> float: ...

    def beat_at_time(self, time_ms: float) -> float: ...

    def bpm_at_time(self, time_ms: float) -> float: ...


@dataclass(frozen=True)
class WeakOracleCurveMetrics:
    schema: str
    predicted_class: CurveClass
    weak_oracle_class: WeakOracleClass
    weak_oracle_class_exact: bool
    coverage_start_ms: float
    coverage_end_ms: float
    sample_hop_ms: float
    sample_count: int
    weak_oracle_initial_signed_offset_beats: float
    weak_oracle_initial_signed_offset_ms: float
    weak_oracle_abs_initial_offset_ms: float
    weak_oracle_phase_mean_beats: float
    weak_oracle_phase_p50_beats: float
    weak_oracle_phase_p90_beats: float
    weak_oracle_phase_max_beats: float
    weak_oracle_phase_mean_ms: float
    weak_oracle_phase_p50_ms: float
    weak_oracle_phase_p90_ms: float
    weak_oracle_phase_max_ms: float
    weak_oracle_local_bpm_mae: float
    weak_oracle_local_bpm_p90_abs_error: float
    weak_oracle_direct_bpm_coverage: float
    weak_oracle_local_bpm_alias_mae: float
    weak_oracle_local_bpm_alias_p90_abs_error: float
    weak_oracle_alias_bpm_coverage: float
    weak_oracle_endpoint_relative_drift_beats: float
    weak_oracle_endpoint_relative_drift_ms: float
    weak_oracle_max_abs_prefix_relative_drift_beats: float
    weak_oracle_max_abs_prefix_relative_drift_ms: float
    weak_oracle_predicted_boundary_count: int
    weak_oracle_boundary_count: int
    weak_oracle_boundary_matched_count: int
    weak_oracle_boundary_precision: float | None
    weak_oracle_boundary_recall: float | None
    weak_oracle_boundary_signed_error_mean_ms: float | None
    weak_oracle_boundary_abs_error_mean_ms: float | None
    weak_oracle_boundary_abs_error_p90_ms: float | None
    weak_oracle_boundary_abs_error_max_ms: float | None
    weak_oracle_jump_direction_match_rate: float | None
    weak_oracle_jump_left_bpm_direct_mae: float | None
    weak_oracle_jump_right_bpm_direct_mae: float | None
    weak_oracle_jump_left_bpm_alias_mae: float | None
    weak_oracle_jump_right_bpm_alias_mae: float | None
    weak_oracle_constant_exact_hit: bool | None
    weak_oracle_jump_exact_hit: bool | None
    weak_oracle_ramp_direction_match: bool | None
    weak_oracle_ramp_start_bpm_alias_error: float | None
    weak_oracle_ramp_end_bpm_alias_error: float | None
    weak_oracle_ramp_slope_error_bpm_per_second: float | None
    weak_oracle_ramp_accuracy: None
    maximum_phase_continuity_error_ms: float
    maximum_serialization_seam_error_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _ContinuousGrid:
    grid: FittedTimingGrid

    @property
    def start_time_ms(self) -> float:
        return float(self.grid.segments[0].offset_ms)

    @property
    def end_time_ms(self) -> float:
        return math.inf

    def beat_at_time(self, time_ms: float) -> float:
        segments = self.grid.segments
        first = segments[0]
        if time_ms < first.offset_ms:
            return float((time_ms - first.offset_ms) / first.beat_length_ms)
        index = _active_segment_index(self.grid, time_ms)
        cumulative = 0.0
        for left, right in zip(segments[:index], segments[1 : index + 1]):
            cumulative += (right.offset_ms - left.offset_ms) / left.beat_length_ms
        segment = segments[index]
        return float(cumulative + (time_ms - segment.offset_ms) / segment.beat_length_ms)

    def bpm_at_time(self, time_ms: float) -> float:
        return float(self.grid.segments[_active_segment_index(self.grid, time_ms)].local_bpm)


@dataclass(frozen=True)
class _ContinuousV3Grid:
    grid: TimingV3Grid

    @property
    def start_time_ms(self) -> float:
        return self.grid.start_time_ms

    @property
    def end_time_ms(self) -> float:
        return self.grid.end_time_ms

    def beat_at_time(self, time_ms: float) -> float:
        return self.grid.beat_at_time(time_ms)

    def bpm_at_time(self, time_ms: float) -> float:
        index = self.grid.section_index_at_time(time_ms)
        return float(self.grid.sections[index].bpm)


def evaluate_curve_against_weak_oracle(
    predicted: TimingV3Grid | PhaseContinuousTimingCurve | FittedTimingGrid,
    weak_oracle: FittedTimingGrid,
    *,
    weak_oracle_class: WeakOracleClass,
    coverage_start_ms: float,
    coverage_end_ms: float,
) -> WeakOracleCurveMetrics:
    """Compare a frozen prediction with a post-hoc `.osu` weak oracle.

    The caller is responsible for constructing and selecting ``predicted``
    before loading ``weak_oracle``. This module is evaluation-only.
    """

    if weak_oracle_class not in (
        WEAK_ORACLE_CLASS_CONSTANT,
        WEAK_ORACLE_CLASS_JUMP,
        WEAK_ORACLE_CLASS_RAMP_LIKE,
    ):
        raise ValueError("weak_oracle_class must be constant, jump, or ramp_like")
    start_ms = _finite_float(coverage_start_ms, "coverage_start_ms")
    end_ms = _finite_float(coverage_end_ms, "coverage_end_ms")
    if end_ms <= start_ms:
        raise ValueError("coverage_end_ms must be greater than coverage_start_ms")

    predicted_curve = _continuous_curve(predicted)
    weak_oracle_curve = _ContinuousGrid(weak_oracle)
    if start_ms < predicted_curve.start_time_ms or end_ms > predicted_curve.end_time_ms:
        raise ValueError("requested coverage is outside predicted curve coverage")

    sample_times = np.asarray(
        phase_sample_times_v1(coverage_start_ms=start_ms, coverage_end_ms=end_ms),
        dtype=np.float64,
    )
    if sample_times.size == 0:
        raise ValueError("weak-oracle sample partition is empty")

    predicted_beats = np.asarray(
        [predicted_curve.beat_at_time(float(time_ms)) for time_ms in sample_times],
        dtype=np.float64,
    )
    weak_oracle_beats = np.asarray(
        [weak_oracle_curve.beat_at_time(float(time_ms)) for time_ms in sample_times],
        dtype=np.float64,
    )
    predicted_bpms = np.asarray(
        [predicted_curve.bpm_at_time(float(time_ms)) for time_ms in sample_times],
        dtype=np.float64,
    )
    weak_oracle_bpms = np.asarray(
        [weak_oracle_curve.bpm_at_time(float(time_ms)) for time_ms in sample_times],
        dtype=np.float64,
    )
    weak_oracle_periods_ms = 60_000.0 / weak_oracle_bpms
    signed_phase_errors = ((predicted_beats - weak_oracle_beats + 0.5) % 1.0) - 0.5
    phase_abs_beats = np.abs(signed_phase_errors)
    phase_abs_ms = phase_abs_beats * weak_oracle_periods_ms
    beat_errors = predicted_beats - weak_oracle_beats
    relative_drift_beats = beat_errors - beat_errors[0]
    relative_drift_ms = relative_drift_beats * weak_oracle_periods_ms

    endpoint_time_ms = float(np.nextafter(end_ms, start_ms))
    endpoint_beat_error = (
        predicted_curve.beat_at_time(endpoint_time_ms)
        - weak_oracle_curve.beat_at_time(endpoint_time_ms)
    )
    endpoint_relative_drift_beats = float(endpoint_beat_error - beat_errors[0])
    endpoint_relative_drift_ms = float(
        endpoint_relative_drift_beats * 60_000.0 / weak_oracle_curve.bpm_at_time(endpoint_time_ms)
    )
    prefix_beats = np.append(relative_drift_beats, endpoint_relative_drift_beats)
    prefix_ms = np.append(relative_drift_ms, endpoint_relative_drift_ms)

    bpm_abs_errors = np.abs(predicted_bpms - weak_oracle_bpms)
    bpm_alias_errors = np.asarray(
        [
            alias_aware_bpm_error(float(predicted_bpm), float(reference_bpm))
            for predicted_bpm, reference_bpm in zip(predicted_bpms, weak_oracle_bpms)
        ],
        dtype=np.float64,
    )
    bpm_tolerances = np.maximum(1.0, 0.01 * weak_oracle_bpms)

    predicted_class = classify_predicted_curve(predicted)
    predicted_boundaries = _predicted_boundaries(
        predicted,
        coverage_start_ms=start_ms,
        coverage_end_ms=end_ms,
    )
    weak_oracle_boundaries = extract_tempo_change_boundaries(
        weak_oracle,
        coverage_start_ms=start_ms,
        coverage_end_ms=end_ms,
    )
    boundary_matches = match_weak_boundaries(predicted_boundaries, weak_oracle_boundaries)
    matched_abs_errors = np.asarray(
        [match.abs_error_ms for match in boundary_matches.matches], dtype=np.float64
    )
    matched_signed_errors = np.asarray(
        [match.signed_error_ms for match in boundary_matches.matches], dtype=np.float64
    )
    direction_matches = tuple(
        _direction(match.predicted.left_bpm, match.predicted.right_bpm)
        == _direction(match.weak_redline.left_bpm, match.weak_redline.right_bpm)
        for match in boundary_matches.matches
    )
    left_alias_errors = tuple(
        alias_aware_bpm_error(match.predicted.left_bpm, match.weak_redline.left_bpm)
        for match in boundary_matches.matches
    )
    right_alias_errors = tuple(
        alias_aware_bpm_error(match.predicted.right_bpm, match.weak_redline.right_bpm)
        for match in boundary_matches.matches
    )
    left_direct_errors = tuple(
        abs(match.predicted.left_bpm - match.weak_redline.left_bpm)
        for match in boundary_matches.matches
    )
    right_direct_errors = tuple(
        abs(match.predicted.right_bpm - match.weak_redline.right_bpm)
        for match in boundary_matches.matches
    )

    precision = _rate(boundary_matches.matched_count, boundary_matches.predicted_boundary_count)
    recall = _rate(boundary_matches.matched_count, boundary_matches.weak_redline_boundary_count)
    offset_pass = abs(float(signed_phase_errors[0] * weak_oracle_periods_ms[0])) <= 70.0
    phase_pass = _percentile(phase_abs_ms, 90.0) <= 70.0
    seam_error_ms = maximum_phase_continuity_error_ms(predicted)
    serialization_error_ms = maximum_serialization_seam_error_ms(predicted)

    weak_oracle_constant_exact_hit: bool | None = None
    if weak_oracle_class == WEAK_ORACLE_CLASS_CONSTANT:
        reference_bpm = float(weak_oracle_bpms[0])
        bpm_tolerance = max(0.25, 0.002 * reference_bpm)
        weak_oracle_constant_exact_hit = bool(
            predicted_class == CURVE_CLASS_CONSTANT
            and _percentile(bpm_alias_errors, 90.0) <= bpm_tolerance
            and offset_pass
            and phase_pass
        )

    weak_oracle_jump_exact_hit: bool | None = None
    if weak_oracle_class == WEAK_ORACLE_CLASS_JUMP:
        side_errors = (*left_direct_errors, *right_direct_errors)
        side_bpms = tuple(
            value
            for match in boundary_matches.matches
            for value in (match.weak_redline.left_bpm, match.weak_redline.right_bpm)
        )
        side_bpm_pass = bool(side_errors) and all(
            error <= max(0.25, 0.002 * bpm)
            for error, bpm in zip(side_errors, side_bpms)
        )
        weak_oracle_jump_exact_hit = bool(
            predicted_class == CURVE_CLASS_JUMP
            and precision == 1.0
            and recall == 1.0
            and bool(direction_matches)
            and all(direction_matches)
            and side_bpm_pass
            and offset_pass
            and phase_pass
            and serialization_error_ms <= 5.0
        )

    ramp_direction_match: bool | None = None
    ramp_start_bpm_error: float | None = None
    ramp_end_bpm_error: float | None = None
    ramp_slope_error: float | None = None
    if weak_oracle_class == WEAK_ORACLE_CLASS_RAMP_LIKE:
        predicted_direction = _trend_direction(predicted_bpms)
        weak_oracle_direction = _trend_direction(weak_oracle_bpms)
        ramp_direction_match = bool(
            predicted_direction is not None
            and weak_oracle_direction is not None
            and predicted_direction == weak_oracle_direction
        )
        ramp_start_bpm_error = alias_aware_bpm_error(
            float(predicted_bpms[0]), float(weak_oracle_bpms[0])
        )
        ramp_end_bpm_error = alias_aware_bpm_error(
            float(predicted_bpms[-1]), float(weak_oracle_bpms[-1])
        )
        elapsed_seconds = (sample_times - sample_times[0]) / 1000.0
        ramp_slope_error = abs(
            _linear_slope(elapsed_seconds, predicted_bpms)
            - _linear_slope(elapsed_seconds, weak_oracle_bpms)
        )

    return WeakOracleCurveMetrics(
        schema=WEAK_ORACLE_CURVE_METRICS_SCHEMA,
        predicted_class=predicted_class,
        weak_oracle_class=weak_oracle_class,
        weak_oracle_class_exact=_class_matches(predicted_class, weak_oracle_class),
        coverage_start_ms=start_ms,
        coverage_end_ms=end_ms,
        sample_hop_ms=PHASE_SAMPLING_HOP_MS,
        sample_count=int(sample_times.size),
        weak_oracle_initial_signed_offset_beats=float(signed_phase_errors[0]),
        weak_oracle_initial_signed_offset_ms=float(signed_phase_errors[0] * weak_oracle_periods_ms[0]),
        weak_oracle_abs_initial_offset_ms=abs(float(signed_phase_errors[0] * weak_oracle_periods_ms[0])),
        weak_oracle_phase_mean_beats=float(np.mean(phase_abs_beats)),
        weak_oracle_phase_p50_beats=_percentile(phase_abs_beats, 50.0),
        weak_oracle_phase_p90_beats=_percentile(phase_abs_beats, 90.0),
        weak_oracle_phase_max_beats=float(np.max(phase_abs_beats)),
        weak_oracle_phase_mean_ms=float(np.mean(phase_abs_ms)),
        weak_oracle_phase_p50_ms=_percentile(phase_abs_ms, 50.0),
        weak_oracle_phase_p90_ms=_percentile(phase_abs_ms, 90.0),
        weak_oracle_phase_max_ms=float(np.max(phase_abs_ms)),
        weak_oracle_local_bpm_mae=float(np.mean(bpm_abs_errors)),
        weak_oracle_local_bpm_p90_abs_error=_percentile(bpm_abs_errors, 90.0),
        weak_oracle_direct_bpm_coverage=float(np.mean(bpm_abs_errors <= bpm_tolerances)),
        weak_oracle_local_bpm_alias_mae=float(np.mean(bpm_alias_errors)),
        weak_oracle_local_bpm_alias_p90_abs_error=_percentile(bpm_alias_errors, 90.0),
        weak_oracle_alias_bpm_coverage=float(np.mean(bpm_alias_errors <= bpm_tolerances)),
        weak_oracle_endpoint_relative_drift_beats=endpoint_relative_drift_beats,
        weak_oracle_endpoint_relative_drift_ms=endpoint_relative_drift_ms,
        weak_oracle_max_abs_prefix_relative_drift_beats=float(np.max(np.abs(prefix_beats))),
        weak_oracle_max_abs_prefix_relative_drift_ms=float(np.max(np.abs(prefix_ms))),
        weak_oracle_predicted_boundary_count=boundary_matches.predicted_boundary_count,
        weak_oracle_boundary_count=boundary_matches.weak_redline_boundary_count,
        weak_oracle_boundary_matched_count=boundary_matches.matched_count,
        weak_oracle_boundary_precision=precision,
        weak_oracle_boundary_recall=recall,
        weak_oracle_boundary_signed_error_mean_ms=_mean_or_none(matched_signed_errors),
        weak_oracle_boundary_abs_error_mean_ms=_mean_or_none(matched_abs_errors),
        weak_oracle_boundary_abs_error_p90_ms=_percentile_or_none(matched_abs_errors, 90.0),
        weak_oracle_boundary_abs_error_max_ms=_max_or_none(matched_abs_errors),
        weak_oracle_jump_direction_match_rate=(
            float(sum(direction_matches) / len(direction_matches)) if direction_matches else None
        ),
        weak_oracle_jump_left_bpm_direct_mae=_tuple_mean(left_direct_errors),
        weak_oracle_jump_right_bpm_direct_mae=_tuple_mean(right_direct_errors),
        weak_oracle_jump_left_bpm_alias_mae=_tuple_mean(left_alias_errors),
        weak_oracle_jump_right_bpm_alias_mae=_tuple_mean(right_alias_errors),
        weak_oracle_constant_exact_hit=weak_oracle_constant_exact_hit,
        weak_oracle_jump_exact_hit=weak_oracle_jump_exact_hit,
        weak_oracle_ramp_direction_match=ramp_direction_match,
        weak_oracle_ramp_start_bpm_alias_error=ramp_start_bpm_error,
        weak_oracle_ramp_end_bpm_alias_error=ramp_end_bpm_error,
        weak_oracle_ramp_slope_error_bpm_per_second=ramp_slope_error,
        weak_oracle_ramp_accuracy=None,
        maximum_phase_continuity_error_ms=seam_error_ms,
        maximum_serialization_seam_error_ms=serialization_error_ms,
    )


def classify_predicted_curve(
    predicted: TimingV3Grid | PhaseContinuousTimingCurve | FittedTimingGrid,
) -> CurveClass:
    if isinstance(predicted, PhaseContinuousTimingCurve):
        return predicted.curve_class  # type: ignore[return-value]
    if isinstance(predicted, TimingV3Grid):
        changes = _predicted_boundaries(
            predicted,
            coverage_start_ms=predicted.coverage_start_ms,
            coverage_end_ms=predicted.coverage_end_ms,
        )
        return CURVE_CLASS_JUMP if changes else CURVE_CLASS_CONSTANT
    if isinstance(predicted, FittedTimingGrid):
        return CURVE_CLASS_JUMP if _significant_fitted_changes(predicted) else CURVE_CLASS_CONSTANT
    raise TypeError("predicted must be TimingV3Grid, PhaseContinuousTimingCurve, or FittedTimingGrid")


def maximum_phase_continuity_error_ms(
    predicted: TimingV3Grid | PhaseContinuousTimingCurve | FittedTimingGrid,
) -> float:
    if isinstance(predicted, PhaseContinuousTimingCurve):
        return max((abs(report.phase_discontinuity_ms) for report in predicted.seam_reports), default=0.0)
    if isinstance(predicted, TimingV3Grid):
        return 0.0
    if isinstance(predicted, FittedTimingGrid):
        errors: list[float] = []
        for previous, current in zip(predicted.segments, predicted.segments[1:]):
            phase = (current.offset_ms - previous.offset_ms) / previous.beat_length_ms
            errors.append(abs(phase - round(phase)) * previous.beat_length_ms)
        return max(errors, default=0.0)
    raise TypeError("unsupported predicted curve type")


def maximum_serialization_seam_error_ms(
    predicted: TimingV3Grid | PhaseContinuousTimingCurve | FittedTimingGrid,
) -> float:
    if isinstance(predicted, TimingV3Grid):
        restored: TimingV3Grid | PhaseContinuousTimingCurve = TimingV3Grid.from_dict(predicted.to_dict())
    elif isinstance(predicted, PhaseContinuousTimingCurve):
        restored = PhaseContinuousTimingCurve.from_canonical_bytes(predicted.canonical_bytes())
    else:
        return maximum_phase_continuity_error_ms(predicted)
    return max(
        (
            abs(restored.time_at_beat(section.start_beat) - predicted.time_at_beat(section.start_beat))
            for section in predicted.sections[1:]
        ),
        default=0.0,
    )


def _continuous_curve(
    curve: TimingV3Grid | PhaseContinuousTimingCurve | FittedTimingGrid,
) -> _AnalyticCurve:
    if isinstance(curve, TimingV3Grid):
        return _ContinuousV3Grid(curve)
    if isinstance(curve, PhaseContinuousTimingCurve):
        return curve
    if isinstance(curve, FittedTimingGrid):
        return _ContinuousGrid(curve)
    raise TypeError("unsupported curve type")


def _predicted_boundaries(
    predicted: TimingV3Grid | PhaseContinuousTimingCurve | FittedTimingGrid,
    *,
    coverage_start_ms: float,
    coverage_end_ms: float,
) -> tuple[Any, ...]:
    if isinstance(predicted, PhaseContinuousTimingCurve):
        from pulsefield_model.timing.evaluation.exp004_metrics import TimingBoundary

        boundaries = []
        for index, report in enumerate(predicted.seam_reports, start=1):
            if not coverage_start_ms <= report.time_ms < coverage_end_ms:
                continue
            if abs(math.log2(report.right_bpm / report.left_bpm)) < math.log2(1.005):
                continue
            boundaries.append(
                TimingBoundary(
                    time_ms=report.time_ms,
                    left_period_ms=60_000.0 / report.left_bpm,
                    right_period_ms=60_000.0 / report.right_bpm,
                    left_bpm=report.left_bpm,
                    right_bpm=report.right_bpm,
                    left_index=index - 1,
                    right_index=index,
                )
            )
        return tuple(boundaries)
    return extract_tempo_change_boundaries(
        predicted,
        coverage_start_ms=coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
    )


def _significant_fitted_changes(grid: FittedTimingGrid) -> bool:
    return any(
        abs(math.log2(right.local_bpm / left.local_bpm)) >= math.log2(1.005)
        for left, right in zip(grid.segments, grid.segments[1:])
    )


def _active_segment_index(grid: FittedTimingGrid, time_ms: float) -> int:
    index = 0
    for candidate, segment in enumerate(grid.segments):
        if segment.offset_ms > time_ms:
            break
        index = candidate
    return index


def _class_matches(predicted_class: CurveClass, weak_oracle_class: WeakOracleClass) -> bool:
    if weak_oracle_class == WEAK_ORACLE_CLASS_RAMP_LIKE:
        return predicted_class == CURVE_CLASS_RAMP
    return predicted_class == weak_oracle_class


def _direction(left_bpm: float, right_bpm: float) -> int:
    return 1 if right_bpm > left_bpm else -1


def _trend_direction(values: np.ndarray[Any, np.dtype[np.float64]]) -> int | None:
    if values.size < 2:
        return None
    delta = float(values[-1] - values[0])
    if delta == 0.0:
        return None
    return 1 if delta > 0.0 else -1


def _linear_slope(xs: np.ndarray[Any, np.dtype[np.float64]], ys: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    if xs.size < 2 or float(xs[-1] - xs[0]) <= 0.0:
        return 0.0
    centered = xs - float(np.mean(xs))
    denominator = float(np.dot(centered, centered))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(centered, ys - float(np.mean(ys))) / denominator)


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _percentile(values: np.ndarray[Any, np.dtype[np.float64]], percentile: float) -> float:
    return float(np.percentile(values, percentile))


def _mean_or_none(values: np.ndarray[Any, np.dtype[np.float64]]) -> float | None:
    return None if values.size == 0 else float(np.mean(values))


def _percentile_or_none(
    values: np.ndarray[Any, np.dtype[np.float64]], percentile: float
) -> float | None:
    return None if values.size == 0 else _percentile(values, percentile)


def _max_or_none(values: np.ndarray[Any, np.dtype[np.float64]]) -> float | None:
    return None if values.size == 0 else float(np.max(values))


def _tuple_mean(values: tuple[float, ...]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _finite_float(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result
