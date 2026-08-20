from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from numbers import Integral
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from pulsefield_model.timing.schema import FittedTimingGrid
from pulsefield_model.timing.v3.schema import (
    ConstantTimingSection,
    TimingV3Grid,
    roundtrip_seam_tolerance_ms,
)


JOINT_MIN_BPM = 20.0
JOINT_MAX_BPM = 1000.0
JOINT_MAX_RELATIVE_BPM_ADJUSTMENT = 0.05
JOINT_ADJACENT_BEAT_DISPLACEMENT_LIMIT = 0.5 + 1e-12
JOINT_SOLVER_NORMALIZED_RESIDUAL_LIMIT = 1e-10

PROJECTION_METHOD_JOINT_FIXED_COUNTS = "joint_phase_fixed_counts"
PROJECTION_METHOD_JOINT_NEARBY_COUNTS = "joint_phase_nearby_counts"

REASON_ADJACENT_BEAT_IDENTITY_DISPLACEMENT_EXCEEDED = (
    "adjacent_beat_identity_displacement_exceeded"
)
REASON_BEAT_COUNT_INVALID = "beat_count_invalid"
REASON_COVERAGE_CONSTRUCTION_FAILED = "coverage_construction_failed"
REASON_DIAGNOSTICS_NOT_JSON_SAFE = "diagnostics_not_json_safe"
REASON_FINAL_BPM_CHANGED = "final_bpm_changed"
REASON_JSON_ROUNDTRIP_FAILED = "json_roundtrip_failed"
REASON_PROJECTED_ANCHOR_INVALID = "projected_anchor_invalid"
REASON_PROJECTED_BPM_OUT_OF_RANGE = "projected_bpm_out_of_range"
REASON_RELATIVE_BPM_ADJUSTMENT_EXCEEDED = "relative_bpm_adjustment_exceeded"
REASON_SCHEMA_CONSTRUCTION_FAILED = "timing_v3_schema_construction_failed"
REASON_SEARCH_NOT_CONVERGED = "search_not_converged"
REASON_SECTION_COUNT_INCREASE = "section_count_increase"
REASON_SOLVER_NONFINITE_COEFFICIENT = "solver_nonfinite_coefficient"
REASON_SOLVER_NONPOSITIVE_PIVOT = "solver_nonpositive_pivot"
REASON_SOLVER_RESIDUAL_FAILED = "solver_residual_failed"
REASON_SOURCE_BPM_OUT_OF_RANGE = "source_bpm_out_of_range"
REASON_SOURCE_INTERVAL_INVALID = "source_interval_invalid"

SELECTION_OBJECTIVE_IMPROVED = "objective_improved"
SELECTION_TIE_BREAK_IMPROVED = "tie_break_improved"


@dataclass(frozen=True)
class TimingV3JointSolverDiagnostic:
    variable_count: int
    matrix_inf_norm: float | None
    rhs_inf_norm: float | None
    solution_inf_norm: float | None
    residual_inf_norm: float | None
    normalized_residual: float | None
    passed: bool
    failure_reason: str | None = None


@dataclass(frozen=True)
class TimingV3JointBoundaryDiagnostic:
    boundary_index: int
    source_left_anchor_ms: float
    source_right_anchor_ms: float
    projected_left_anchor_ms: float
    projected_right_anchor_ms: float
    projected_left_anchor_displacement_ms: float
    projected_right_anchor_displacement_ms: float
    source_implied_beat_count: float
    initial_beat_count: int
    beat_count: int
    original_residual_ms: float
    projected_period_ms: float
    source_left_bpm: float
    source_right_bpm: float
    projected_left_bpm: float
    relative_bpm_adjustment: float
    right_anchor_on_left_lattice_displacement_ms: float
    right_anchor_on_left_lattice_displacement_beats: float
    right_anchor_on_right_lattice_displacement_beats: float
    objective_contribution: float
    failure_reason: str | None = None


@dataclass(frozen=True)
class TimingV3JointSearchAttempt:
    attempt_index: int
    sweep_index: int
    boundary_index: int | None
    candidate_beat_count: int | None
    beat_counts: tuple[int, ...]
    accepted_change: bool
    selection_reason: str | None
    objective: float | None
    feasibility_reason: str | None
    mathematical_grid_fingerprint: str | None


@dataclass(frozen=True)
class TimingV3JointProjectionDiagnostics:
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
    initial_beat_counts: tuple[int, ...]
    final_beat_counts: tuple[int, ...]
    projected_anchor_times_ms: tuple[float, ...]
    source_anchor_displacements_ms: tuple[float, ...]
    objective: float | None
    source_surrogate_rms_beats: float | None
    prefix_objective: float | None
    interval_objectives: tuple[float, ...]
    tail_objective: float | None
    maximum_anchor_displacement_adjacent_local_beats: float | None
    maximum_relative_bpm_adjustment: float | None
    changed_count: int
    changed_count_rate: float
    solver: TimingV3JointSolverDiagnostic | None
    boundary_diagnostics: tuple[TimingV3JointBoundaryDiagnostic, ...]
    search_attempts: tuple[TimingV3JointSearchAttempt, ...]
    sweeps_completed: int
    search_converged: bool
    mathematical_grid_fingerprint: str | None
    integer_search_fingerprint: str
    replay_fingerprint: str
    failure_source_section_index: int | None = None
    failure_boundary_index: int | None = None
    failure_reason: str | None = None


@dataclass(frozen=True)
class TimingV3JointProjectionResult:
    method: str
    grid: TimingV3Grid | None
    diagnostics: TimingV3JointProjectionDiagnostics
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.grid is not None


@dataclass(frozen=True)
class _PreparedSource:
    segments: tuple[Any, ...]
    offsets_ms: tuple[float, ...]
    periods_ms: tuple[float, ...]
    bpms: tuple[float, ...]
    deltas_ms: tuple[float, ...]
    implied_beat_counts: tuple[float, ...]
    initial_beat_counts: tuple[int, ...]
    coverage_start_ms: float
    coverage_end_ms: float


@dataclass(frozen=True)
class _PreparationFailure:
    source_section_count: int
    source_origin_time_ms: float
    reason: str
    failure_source_section_index: int | None


@dataclass(frozen=True)
class _Candidate:
    beat_counts: tuple[int, ...]
    displacements_ms: tuple[float, ...]
    projected_anchors_ms: tuple[float, ...]
    projected_bpms: tuple[float, ...]
    objective: float | None
    source_surrogate_rms_beats: float | None
    prefix_objective: float | None
    interval_objectives: tuple[float, ...]
    tail_objective: float | None
    maximum_anchor_displacement_adjacent_local_beats: float | None
    maximum_relative_bpm_adjustment: float | None
    solver: TimingV3JointSolverDiagnostic
    boundary_diagnostics: tuple[TimingV3JointBoundaryDiagnostic, ...]
    grid: TimingV3Grid | None
    mathematical_grid_fingerprint: str | None
    failure_reason: str | None
    failure_boundary_index: int | None

    @property
    def ok(self) -> bool:
        return self.grid is not None and self.failure_reason is None


def project_joint_phase_fixed_counts(
    source_grid: FittedTimingGrid,
    *,
    coverage_start_ms: float = 0.0,
    coverage_end_ms: float | None = None,
    frame_count: int | None = None,
    frame_rate_hz: float | None = None,
) -> TimingV3JointProjectionResult:
    """Project with Family C0's frozen half-up interval counts."""

    return _project_joint(
        source_grid,
        method=PROJECTION_METHOD_JOINT_FIXED_COUNTS,
        search_nearby_counts=False,
        coverage_start_ms=coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
        frame_count=frame_count,
        frame_rate_hz=frame_rate_hz,
    )


def project_joint_phase_nearby_counts(
    source_grid: FittedTimingGrid,
    *,
    coverage_start_ms: float = 0.0,
    coverage_end_ms: float | None = None,
    frame_count: int | None = None,
    frame_rate_hz: float | None = None,
) -> TimingV3JointProjectionResult:
    """Project with Family C1's deterministic N0 +/- 1 coordinate search."""

    return _project_joint(
        source_grid,
        method=PROJECTION_METHOD_JOINT_NEARBY_COUNTS,
        search_nearby_counts=True,
        coverage_start_ms=coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
        frame_count=frame_count,
        frame_rate_hz=frame_rate_hz,
    )


def _project_joint(
    source_grid: FittedTimingGrid,
    *,
    method: str,
    search_nearby_counts: bool,
    coverage_start_ms: float,
    coverage_end_ms: float | None,
    frame_count: int | None,
    frame_rate_hz: float | None,
) -> TimingV3JointProjectionResult:
    coverage_start_ms, coverage_end_ms = _coverage_inputs(
        coverage_start_ms=coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
        frame_count=frame_count,
        frame_rate_hz=frame_rate_hz,
    )
    prepared_or_failure = _prepare_source(
        source_grid,
        coverage_start_ms=coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
    )
    if isinstance(prepared_or_failure, _PreparationFailure):
        return _preparation_failure_result(method, prepared_or_failure, coverage_start_ms, coverage_end_ms)
    source = prepared_or_failure

    candidate = _evaluate_candidate(source, source.initial_beat_counts)
    attempts = [
        _search_attempt(
            attempt_index=0,
            sweep_index=0,
            boundary_index=None,
            candidate_beat_count=None,
            candidate=candidate,
            accepted_change=False,
        )
    ]
    if not candidate.ok:
        return _candidate_result(
            method,
            source,
            candidate,
            attempts=tuple(attempts),
            sweeps_completed=0,
            search_converged=False,
            force_reason=candidate.failure_reason,
        )

    sweeps_completed = 0
    search_converged = True
    if search_nearby_counts and source.initial_beat_counts:
        search_converged = False
        sweep_cap = len(source.segments) + 1
        for sweep_index in range(1, sweep_cap + 1):
            changed_this_sweep = False
            for boundary_index, initial_count in enumerate(source.initial_beat_counts):
                candidates = tuple(
                    count
                    for count in sorted({initial_count - 1, initial_count, initial_count + 1})
                    if count >= 1
                )
                for candidate_count in candidates:
                    if candidate_count == candidate.beat_counts[boundary_index]:
                        continue
                    trial_counts = list(candidate.beat_counts)
                    trial_counts[boundary_index] = candidate_count
                    trial = _evaluate_candidate(source, tuple(trial_counts))
                    selection_reason = (
                        _candidate_selection_reason(
                            trial,
                            candidate,
                            initial_counts=source.initial_beat_counts,
                        )
                        if trial.ok
                        else None
                    )
                    accepted_change = selection_reason is not None
                    attempts.append(
                        _search_attempt(
                            attempt_index=len(attempts),
                            sweep_index=sweep_index,
                            boundary_index=boundary_index,
                            candidate_beat_count=candidate_count,
                            candidate=trial,
                            accepted_change=accepted_change,
                            selection_reason=selection_reason,
                        )
                    )
                    if accepted_change:
                        candidate = trial
                        changed_this_sweep = True
            sweeps_completed = sweep_index
            if not changed_this_sweep:
                search_converged = True
                break
            if sweep_index == sweep_cap:
                return _candidate_result(
                    method,
                    source,
                    candidate,
                    attempts=tuple(attempts),
                    sweeps_completed=sweeps_completed,
                    search_converged=False,
                    force_reason=REASON_SEARCH_NOT_CONVERGED,
                )

    return _candidate_result(
        method,
        source,
        candidate,
        attempts=tuple(attempts),
        sweeps_completed=sweeps_completed,
        search_converged=search_converged,
        force_reason=None,
    )


def _coverage_inputs(
    *,
    coverage_start_ms: float,
    coverage_end_ms: float | None,
    frame_count: int | None,
    frame_rate_hz: float | None,
) -> tuple[float, float]:
    if coverage_end_ms is None:
        if frame_count is None or frame_rate_hz is None:
            raise ValueError("coverage_end_ms or frame_count/frame_rate_hz is required")
        if isinstance(frame_count, bool) or not isinstance(frame_count, Integral) or frame_count <= 0:
            raise ValueError(f"frame_count must be a positive integer, got {frame_count!r}")
        if isinstance(frame_rate_hz, bool) or not math.isfinite(frame_rate_hz) or frame_rate_hz <= 0.0:
            raise ValueError(f"frame_rate_hz must be positive and finite, got {frame_rate_hz!r}")
        coverage_end_ms = 1000.0 * int(frame_count) / float(frame_rate_hz)
    elif frame_count is not None or frame_rate_hz is not None:
        raise ValueError("coverage_end_ms is mutually exclusive with frame_count/frame_rate_hz")

    if isinstance(coverage_start_ms, bool) or not math.isfinite(coverage_start_ms):
        raise ValueError(f"coverage_start_ms must be finite, got {coverage_start_ms!r}")
    if (
        isinstance(coverage_end_ms, bool)
        or not math.isfinite(coverage_end_ms)
        or coverage_end_ms <= coverage_start_ms
    ):
        raise ValueError(
            "coverage_end_ms must be finite and greater than coverage_start_ms, "
            f"got {coverage_end_ms!r} <= {coverage_start_ms!r}",
        )
    return float(coverage_start_ms), float(coverage_end_ms)


def _prepare_source(
    source_grid: FittedTimingGrid,
    *,
    coverage_start_ms: float,
    coverage_end_ms: float,
) -> _PreparedSource | _PreparationFailure:
    segments = tuple(source_grid.segments)
    offsets_ms = tuple(float(segment.offset_ms) for segment in segments)
    periods_ms = tuple(float(segment.beat_length_ms) for segment in segments)
    bpms: list[float] = []
    for section_index, period_ms in enumerate(periods_ms):
        try:
            bpm = 60000.0 / period_ms
        except (OverflowError, ZeroDivisionError):
            bpm = math.inf
        if not math.isfinite(bpm) or not JOINT_MIN_BPM <= bpm <= JOINT_MAX_BPM:
            return _PreparationFailure(
                source_section_count=len(segments),
                source_origin_time_ms=offsets_ms[0],
                reason=REASON_SOURCE_BPM_OUT_OF_RANGE,
                failure_source_section_index=section_index,
            )
        bpms.append(float(bpm))

    deltas_ms: list[float] = []
    implied_counts: list[float] = []
    initial_counts: list[int] = []
    for section_index in range(len(segments) - 1):
        delta_ms = offsets_ms[section_index + 1] - offsets_ms[section_index]
        if not math.isfinite(delta_ms) or delta_ms <= 0.0:
            return _PreparationFailure(
                source_section_count=len(segments),
                source_origin_time_ms=offsets_ms[0],
                reason=REASON_SOURCE_INTERVAL_INVALID,
                failure_source_section_index=section_index,
            )
        implied_count = delta_ms / periods_ms[section_index]
        if not math.isfinite(implied_count) or implied_count <= 0.0:
            return _PreparationFailure(
                source_section_count=len(segments),
                source_origin_time_ms=offsets_ms[0],
                reason=REASON_SOURCE_INTERVAL_INVALID,
                failure_source_section_index=section_index,
            )
        count = max(1, math.floor(implied_count + 0.5))
        deltas_ms.append(float(delta_ms))
        implied_counts.append(float(implied_count))
        initial_counts.append(int(count))

    return _PreparedSource(
        segments=segments,
        offsets_ms=offsets_ms,
        periods_ms=periods_ms,
        bpms=tuple(bpms),
        deltas_ms=tuple(deltas_ms),
        implied_beat_counts=tuple(implied_counts),
        initial_beat_counts=tuple(initial_counts),
        coverage_start_ms=coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
    )


def _evaluate_candidate(source: _PreparedSource, beat_counts: tuple[int, ...]) -> _Candidate:
    if len(beat_counts) != len(source.deltas_ms) or any(
        isinstance(count, bool) or not isinstance(count, Integral) or count < 1 for count in beat_counts
    ):
        return _empty_candidate(beat_counts, REASON_BEAT_COUNT_INVALID)

    residuals_ms: list[float] = []
    for boundary_index, (delta_ms, period_ms, count) in enumerate(
        zip(source.deltas_ms, source.periods_ms, beat_counts)
    ):
        try:
            residual_ms = delta_ms - int(count) * period_ms
        except OverflowError:
            return _empty_candidate(
                beat_counts,
                REASON_SOLVER_NONFINITE_COEFFICIENT,
                failure_boundary_index=boundary_index,
            )
        if not math.isfinite(residual_ms):
            return _empty_candidate(
                beat_counts,
                REASON_SOLVER_NONFINITE_COEFFICIENT,
                failure_boundary_index=boundary_index,
            )
        residuals_ms.append(float(residual_ms))

    solution, solver = _solve_displacements(source, beat_counts, tuple(residuals_ms))
    if solution is None:
        return _empty_candidate(
            beat_counts,
            solver.failure_reason or REASON_SOLVER_RESIDUAL_FAILED,
            solver=solver,
        )

    displacements_ms = (0.0, *(float(value) for value in solution))
    projected_anchors_ms = tuple(
        source_anchor_ms + displacement_ms
        for source_anchor_ms, displacement_ms in zip(source.offsets_ms, displacements_ms)
    )
    if not all(math.isfinite(value) for value in projected_anchors_ms) or any(
        right <= left for left, right in zip(projected_anchors_ms, projected_anchors_ms[1:])
    ):
        return _empty_candidate(
            beat_counts,
            REASON_PROJECTED_ANCHOR_INVALID,
            solver=solver,
            displacements_ms=displacements_ms,
            projected_anchors_ms=projected_anchors_ms,
        )

    projected_bpms: list[float] = []
    projected_periods_ms: list[float] = []
    relative_adjustments: list[float] = []
    boundary_diagnostics: list[TimingV3JointBoundaryDiagnostic] = []
    interval_objectives: list[float] = []
    maximum_adjacent_displacement_beats = 0.0
    failure_reason: str | None = None
    failure_boundary_index: int | None = None

    for boundary_index, count in enumerate(beat_counts):
        projected_interval_ms = projected_anchors_ms[boundary_index + 1] - projected_anchors_ms[boundary_index]
        try:
            projected_period_ms = projected_interval_ms / int(count)
            projected_bpm = 60000.0 / projected_period_ms
        except (OverflowError, ZeroDivisionError):
            projected_period_ms = math.nan
            projected_bpm = math.nan
        if (
            not math.isfinite(projected_period_ms)
            or projected_period_ms <= 0.0
            or not math.isfinite(projected_bpm)
        ):
            return _empty_candidate(
                beat_counts,
                REASON_PROJECTED_ANCHOR_INVALID,
                solver=solver,
                displacements_ms=displacements_ms,
                projected_anchors_ms=projected_anchors_ms,
                failure_boundary_index=boundary_index,
            )

        source_period_ms = source.periods_ms[boundary_index]
        relative_adjustment = source_period_ms / projected_period_ms - 1.0
        right_on_left_lattice_ms = (
            projected_anchors_ms[boundary_index + 1]
            - (source.offsets_ms[boundary_index] + int(count) * source_period_ms)
        )
        right_on_left_lattice_beats = abs(right_on_left_lattice_ms) / source_period_ms
        right_on_right_lattice_beats = (
            abs(projected_anchors_ms[boundary_index + 1] - source.offsets_ms[boundary_index + 1])
            / source.periods_ms[boundary_index + 1]
        )
        maximum_adjacent_displacement_beats = max(
            maximum_adjacent_displacement_beats,
            right_on_left_lattice_beats,
            right_on_right_lattice_beats,
        )

        a_i = displacements_ms[boundary_index]
        b_i = displacements_ms[boundary_index + 1] + residuals_ms[boundary_index]
        weight_i = source.deltas_ms[boundary_index] / (3.0 * source_period_ms * source_period_ms)
        objective_i = weight_i * (a_i * a_i + a_i * b_i + b_i * b_i)
        values = (
            relative_adjustment,
            right_on_left_lattice_ms,
            right_on_left_lattice_beats,
            right_on_right_lattice_beats,
            objective_i,
        )
        if not all(math.isfinite(value) for value in values):
            return _empty_candidate(
                beat_counts,
                REASON_SOLVER_NONFINITE_COEFFICIENT,
                solver=solver,
                displacements_ms=displacements_ms,
                projected_anchors_ms=projected_anchors_ms,
                failure_boundary_index=boundary_index,
            )

        boundary_reason: str | None = None
        if not JOINT_MIN_BPM <= projected_bpm <= JOINT_MAX_BPM:
            boundary_reason = REASON_PROJECTED_BPM_OUT_OF_RANGE
        elif abs(relative_adjustment) > JOINT_MAX_RELATIVE_BPM_ADJUSTMENT:
            boundary_reason = REASON_RELATIVE_BPM_ADJUSTMENT_EXCEEDED
        elif (
            right_on_left_lattice_beats > JOINT_ADJACENT_BEAT_DISPLACEMENT_LIMIT
            or right_on_right_lattice_beats > JOINT_ADJACENT_BEAT_DISPLACEMENT_LIMIT
        ):
            boundary_reason = REASON_ADJACENT_BEAT_IDENTITY_DISPLACEMENT_EXCEEDED
        if boundary_reason is not None and failure_reason is None:
            failure_reason = boundary_reason
            failure_boundary_index = boundary_index

        projected_bpms.append(float(projected_bpm))
        projected_periods_ms.append(float(projected_period_ms))
        relative_adjustments.append(float(relative_adjustment))
        interval_objectives.append(float(objective_i))
        boundary_diagnostics.append(
            TimingV3JointBoundaryDiagnostic(
                boundary_index=boundary_index,
                source_left_anchor_ms=source.offsets_ms[boundary_index],
                source_right_anchor_ms=source.offsets_ms[boundary_index + 1],
                projected_left_anchor_ms=projected_anchors_ms[boundary_index],
                projected_right_anchor_ms=projected_anchors_ms[boundary_index + 1],
                projected_left_anchor_displacement_ms=displacements_ms[boundary_index],
                projected_right_anchor_displacement_ms=displacements_ms[boundary_index + 1],
                source_implied_beat_count=source.implied_beat_counts[boundary_index],
                initial_beat_count=source.initial_beat_counts[boundary_index],
                beat_count=int(count),
                original_residual_ms=residuals_ms[boundary_index],
                projected_period_ms=float(projected_period_ms),
                source_left_bpm=source.bpms[boundary_index],
                source_right_bpm=source.bpms[boundary_index + 1],
                projected_left_bpm=float(projected_bpm),
                relative_bpm_adjustment=float(relative_adjustment),
                right_anchor_on_left_lattice_displacement_ms=float(right_on_left_lattice_ms),
                right_anchor_on_left_lattice_displacement_beats=float(right_on_left_lattice_beats),
                right_anchor_on_right_lattice_displacement_beats=float(right_on_right_lattice_beats),
                objective_contribution=float(objective_i),
                failure_reason=boundary_reason,
            )
        )

    projected_bpms.append(source.bpms[-1])
    prefix_objective, tail_objective = _endpoint_objectives(
        source,
        beat_counts,
        tuple(residuals_ms),
        displacements_ms,
    )
    objective = prefix_objective + math.fsum(interval_objectives) + tail_objective
    if not math.isfinite(objective) or objective < 0.0:
        return _empty_candidate(
            beat_counts,
            REASON_SOLVER_NONFINITE_COEFFICIENT,
            solver=solver,
            displacements_ms=displacements_ms,
            projected_anchors_ms=projected_anchors_ms,
            boundary_diagnostics=tuple(boundary_diagnostics),
        )
    source_surrogate_rms_beats = math.sqrt(
        objective / (source.coverage_end_ms - source.coverage_start_ms)
    )
    maximum_relative_adjustment = max((abs(value) for value in relative_adjustments), default=0.0)

    candidate = _Candidate(
        beat_counts=tuple(int(count) for count in beat_counts),
        displacements_ms=displacements_ms,
        projected_anchors_ms=projected_anchors_ms,
        projected_bpms=tuple(projected_bpms),
        objective=float(objective),
        source_surrogate_rms_beats=float(source_surrogate_rms_beats),
        prefix_objective=float(prefix_objective),
        interval_objectives=tuple(interval_objectives),
        tail_objective=float(tail_objective),
        maximum_anchor_displacement_adjacent_local_beats=float(maximum_adjacent_displacement_beats),
        maximum_relative_bpm_adjustment=float(maximum_relative_adjustment),
        solver=solver,
        boundary_diagnostics=tuple(boundary_diagnostics),
        grid=None,
        mathematical_grid_fingerprint=None,
        failure_reason=failure_reason,
        failure_boundary_index=failure_boundary_index,
    )
    if failure_reason is not None:
        return candidate

    grid, grid_reason = _build_grid(
        source,
        beat_counts=beat_counts,
        projected_anchors_ms=projected_anchors_ms,
        projected_bpms=tuple(projected_bpms),
        projected_periods_ms=tuple(projected_periods_ms),
    )
    if grid is None:
        return replace(candidate, failure_reason=grid_reason)
    fingerprint = _mathematical_grid_fingerprint(
        grid,
        beat_counts=beat_counts,
        projected_anchors_ms=projected_anchors_ms,
        projected_bpms=tuple(projected_bpms),
    )
    return replace(candidate, grid=grid, mathematical_grid_fingerprint=fingerprint)


def _solve_displacements(
    source: _PreparedSource,
    beat_counts: tuple[int, ...],
    residuals_ms: tuple[float, ...],
) -> tuple[tuple[float, ...] | None, TimingV3JointSolverDiagnostic]:
    variable_count = len(beat_counts)
    if variable_count == 0:
        return (), TimingV3JointSolverDiagnostic(
            variable_count=0,
            matrix_inf_norm=0.0,
            rhs_inf_norm=0.0,
            solution_inf_norm=0.0,
            residual_inf_norm=0.0,
            normalized_residual=0.0,
            passed=True,
        )

    diagonal = np.zeros(variable_count, dtype=np.float64)
    off_diagonal = np.zeros(max(0, variable_count - 1), dtype=np.float64)
    linear = np.zeros(variable_count, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for interval_index, (delta_ms, period_ms, residual_ms) in enumerate(
            zip(source.deltas_ms, source.periods_ms, residuals_ms)
        ):
            weight = np.float64(delta_ms) / (
                np.float64(3.0) * np.float64(period_ms) * np.float64(period_ms)
            )
            right_variable = interval_index
            diagonal[right_variable] += weight
            linear[right_variable] += np.float64(2.0) * weight * np.float64(residual_ms)
            if interval_index > 0:
                left_variable = interval_index - 1
                diagonal[left_variable] += weight
                off_diagonal[left_variable] += np.float64(0.5) * weight
                linear[left_variable] += weight * np.float64(residual_ms)

        prefix_length_ms = max(0.0, source.offsets_ms[0] - source.coverage_start_ms)
        prefix_weight = (
            np.float64(prefix_length_ms)
            * np.float64(prefix_length_ms)
            * np.float64(prefix_length_ms)
            / (
                np.float64(3.0)
                * np.float64(beat_counts[0])
                * np.float64(beat_counts[0])
                * np.float64(source.periods_ms[0])
                * np.float64(source.periods_ms[0])
                * np.float64(source.periods_ms[0])
                * np.float64(source.periods_ms[0])
            )
        )
        diagonal[0] += prefix_weight
        linear[0] += np.float64(2.0) * prefix_weight * np.float64(residuals_ms[0])

        tail_length_ms = max(
            0.0,
            source.coverage_end_ms - max(source.offsets_ms[-1], source.coverage_start_ms),
        )
        tail_weight = np.float64(tail_length_ms) / (
            np.float64(source.periods_ms[-1]) * np.float64(source.periods_ms[-1])
        )
        diagonal[-1] += tail_weight
        rhs = np.float64(-0.5) * linear

    arrays: Sequence[NDArray[np.float64]] = (diagonal, off_diagonal, linear, rhs)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        return None, TimingV3JointSolverDiagnostic(
            variable_count=variable_count,
            matrix_inf_norm=None,
            rhs_inf_norm=None,
            solution_inf_norm=None,
            residual_inf_norm=None,
            normalized_residual=None,
            passed=False,
            failure_reason=REASON_SOLVER_NONFINITE_COEFFICIENT,
        )

    matrix_inf_norm = _tridiagonal_inf_norm(diagonal, off_diagonal)
    rhs_inf_norm = float(np.max(np.abs(rhs)))
    solution, solve_reason = _thomas_solve(diagonal, off_diagonal, rhs)
    if solution is None:
        return None, TimingV3JointSolverDiagnostic(
            variable_count=variable_count,
            matrix_inf_norm=matrix_inf_norm,
            rhs_inf_norm=rhs_inf_norm,
            solution_inf_norm=None,
            residual_inf_norm=None,
            normalized_residual=None,
            passed=False,
            failure_reason=solve_reason,
        )

    reconstructed = diagonal * solution
    if variable_count > 1:
        reconstructed[:-1] += off_diagonal * solution[1:]
        reconstructed[1:] += off_diagonal * solution[:-1]
    residual = reconstructed - rhs
    solution_inf_norm = float(np.max(np.abs(solution)))
    residual_inf_norm = float(np.max(np.abs(residual)))
    scale = max(1.0, matrix_inf_norm * solution_inf_norm + rhs_inf_norm)
    normalized_residual = residual_inf_norm / scale
    values = (
        matrix_inf_norm,
        rhs_inf_norm,
        solution_inf_norm,
        residual_inf_norm,
        normalized_residual,
    )
    if not all(math.isfinite(value) for value in values):
        reason = REASON_SOLVER_NONFINITE_COEFFICIENT
    elif normalized_residual > JOINT_SOLVER_NORMALIZED_RESIDUAL_LIMIT:
        reason = REASON_SOLVER_RESIDUAL_FAILED
    else:
        reason = None
    diagnostic = TimingV3JointSolverDiagnostic(
        variable_count=variable_count,
        matrix_inf_norm=matrix_inf_norm,
        rhs_inf_norm=rhs_inf_norm,
        solution_inf_norm=solution_inf_norm,
        residual_inf_norm=residual_inf_norm,
        normalized_residual=normalized_residual,
        passed=reason is None,
        failure_reason=reason,
    )
    if reason is not None:
        return None, diagnostic
    return tuple(float(value) for value in solution), diagnostic


def _thomas_solve(
    diagonal: NDArray[np.float64],
    off_diagonal: NDArray[np.float64],
    rhs: NDArray[np.float64],
) -> tuple[NDArray[np.float64] | None, str | None]:
    modified_diagonal = diagonal.copy()
    modified_rhs = rhs.copy()
    upper = off_diagonal.copy()
    variable_count = len(modified_diagonal)

    for index in range(variable_count):
        pivot = modified_diagonal[index]
        if not np.isfinite(pivot):
            return None, REASON_SOLVER_NONFINITE_COEFFICIENT
        if pivot <= 0.0:
            return None, REASON_SOLVER_NONPOSITIVE_PIVOT
        if index + 1 < variable_count:
            multiplier = off_diagonal[index] / pivot
            modified_diagonal[index + 1] -= multiplier * upper[index]
            modified_rhs[index + 1] -= multiplier * modified_rhs[index]
            if not math.isfinite(float(modified_diagonal[index + 1])) or not math.isfinite(
                float(modified_rhs[index + 1])
            ):
                return None, REASON_SOLVER_NONFINITE_COEFFICIENT

    solution = np.empty(variable_count, dtype=np.float64)
    solution[-1] = modified_rhs[-1] / modified_diagonal[-1]
    for index in range(variable_count - 2, -1, -1):
        solution[index] = (
            modified_rhs[index] - upper[index] * solution[index + 1]
        ) / modified_diagonal[index]
    if not np.all(np.isfinite(solution)):
        return None, REASON_SOLVER_NONFINITE_COEFFICIENT
    return solution, None


def _tridiagonal_inf_norm(
    diagonal: NDArray[np.float64],
    off_diagonal: NDArray[np.float64],
) -> float:
    row_sums = np.abs(diagonal).copy()
    if len(diagonal) > 1:
        row_sums[:-1] += np.abs(off_diagonal)
        row_sums[1:] += np.abs(off_diagonal)
    return float(np.max(row_sums))


def _endpoint_objectives(
    source: _PreparedSource,
    beat_counts: tuple[int, ...],
    residuals_ms: tuple[float, ...],
    displacements_ms: tuple[float, ...],
) -> tuple[float, float]:
    if not beat_counts:
        return 0.0, 0.0
    prefix_length_ms = max(0.0, source.offsets_ms[0] - source.coverage_start_ms)
    prefix_objective = (
        prefix_length_ms**3
        / (
            3.0
            * beat_counts[0] ** 2
            * source.periods_ms[0] ** 4
        )
        * (displacements_ms[1] + residuals_ms[0]) ** 2
    )
    tail_length_ms = max(
        0.0,
        source.coverage_end_ms - max(source.offsets_ms[-1], source.coverage_start_ms),
    )
    tail_objective = tail_length_ms * (
        displacements_ms[-1] / source.periods_ms[-1]
    ) ** 2
    return float(prefix_objective), float(tail_objective)


def _build_grid(
    source: _PreparedSource,
    *,
    beat_counts: tuple[int, ...],
    projected_anchors_ms: tuple[float, ...],
    projected_bpms: tuple[float, ...],
    projected_periods_ms: tuple[float, ...],
) -> tuple[TimingV3Grid | None, str | None]:
    if len(projected_bpms) > len(source.segments):
        return None, REASON_SECTION_COUNT_INCREASE
    if projected_bpms[-1] != source.bpms[-1]:
        return None, REASON_FINAL_BPM_CHANGED

    try:
        anchor_beats = [0]
        for count in beat_counts:
            anchor_beats.append(anchor_beats[-1] + int(count))
        first_period_ms = projected_periods_ms[0] if projected_periods_ms else source.periods_ms[0]
        first_ratio = (source.coverage_start_ms - projected_anchors_ms[0]) / first_period_ms
        if not math.isfinite(first_ratio):
            return None, REASON_COVERAGE_CONSTRUCTION_FAILED
        first_start_beat = min(0, math.floor(first_ratio))

        last_period_ms = source.periods_ms[-1]
        last_ratio = (source.coverage_end_ms - projected_anchors_ms[-1]) / last_period_ms
        if not math.isfinite(last_ratio):
            return None, REASON_COVERAGE_CONSTRUCTION_FAILED
        last_end_beat = max(anchor_beats[-1] + 1, anchor_beats[-1] + math.ceil(last_ratio))

        sections: list[ConstantTimingSection] = []
        if len(projected_bpms) == 1:
            sections.append(
                ConstantTimingSection(
                    start_beat=first_start_beat,
                    end_beat=last_end_beat,
                    bpm=projected_bpms[0],
                )
            )
        else:
            for section_index, bpm in enumerate(projected_bpms):
                start_beat = first_start_beat if section_index == 0 else anchor_beats[section_index]
                end_beat = (
                    last_end_beat
                    if section_index == len(projected_bpms) - 1
                    else anchor_beats[section_index + 1]
                )
                sections.append(
                    ConstantTimingSection(
                        start_beat=start_beat,
                        end_beat=end_beat,
                        bpm=bpm,
                    )
                )
        grid = TimingV3Grid(
            origin_beat=0,
            origin_time_ms=source.offsets_ms[0],
            sections=tuple(sections),
            coverage_start_ms=source.coverage_start_ms,
            coverage_end_ms=source.coverage_end_ms,
        )
    except (OverflowError, TypeError, ValueError):
        return None, REASON_SCHEMA_CONSTRUCTION_FAILED

    if len(grid.sections) > len(source.segments):
        return None, REASON_SECTION_COUNT_INCREASE
    if grid.coverage_start_ms != source.coverage_start_ms or grid.coverage_end_ms != source.coverage_end_ms:
        return None, REASON_COVERAGE_CONSTRUCTION_FAILED
    for boundary_index, expected_anchor_ms in enumerate(projected_anchors_ms[1:], start=0):
        actual_anchor_ms = grid.section_end_time_ms(boundary_index)
        if abs(actual_anchor_ms - expected_anchor_ms) > roundtrip_seam_tolerance_ms(expected_anchor_ms):
            return None, REASON_SCHEMA_CONSTRUCTION_FAILED

    try:
        payload = json.loads(json.dumps(grid.to_dict(), allow_nan=False, sort_keys=True))
        restored = TimingV3Grid.from_dict(payload)
    except (OverflowError, TypeError, ValueError):
        return None, REASON_JSON_ROUNDTRIP_FAILED
    if len(restored.boundary_times_ms) != len(grid.boundary_times_ms):
        return None, REASON_JSON_ROUNDTRIP_FAILED
    for original_ms, restored_ms in zip(grid.boundary_times_ms, restored.boundary_times_ms):
        if abs(restored_ms - original_ms) > roundtrip_seam_tolerance_ms(original_ms):
            return None, REASON_JSON_ROUNDTRIP_FAILED
    return grid, None


def _candidate_selection_reason(
    left: _Candidate,
    right: _Candidate,
    *,
    initial_counts: tuple[int, ...],
) -> str | None:
    assert left.objective is not None
    assert right.objective is not None
    tolerance = max(
        1e-12,
        16.0 * max(math.ulp(left.objective), math.ulp(right.objective)),
    )
    if left.objective < right.objective - tolerance:
        return SELECTION_OBJECTIVE_IMPROVED
    if abs(left.objective - right.objective) > tolerance:
        return None
    if _candidate_tie_key(left, initial_counts) < _candidate_tie_key(right, initial_counts):
        return SELECTION_TIE_BREAK_IMPROVED
    return None


def _candidate_tie_key(
    candidate: _Candidate,
    initial_counts: tuple[int, ...],
) -> tuple[float, int, int, tuple[int, ...]]:
    assert candidate.maximum_anchor_displacement_adjacent_local_beats is not None
    different_count = sum(
        count != initial for count, initial in zip(candidate.beat_counts, initial_counts)
    )
    absolute_count_delta = sum(
        abs(count - initial) for count, initial in zip(candidate.beat_counts, initial_counts)
    )
    return (
        candidate.maximum_anchor_displacement_adjacent_local_beats,
        different_count,
        absolute_count_delta,
        candidate.beat_counts,
    )


def _search_attempt(
    *,
    attempt_index: int,
    sweep_index: int,
    boundary_index: int | None,
    candidate_beat_count: int | None,
    candidate: _Candidate,
    accepted_change: bool,
    selection_reason: str | None = None,
) -> TimingV3JointSearchAttempt:
    return TimingV3JointSearchAttempt(
        attempt_index=attempt_index,
        sweep_index=sweep_index,
        boundary_index=boundary_index,
        candidate_beat_count=candidate_beat_count,
        beat_counts=candidate.beat_counts,
        accepted_change=bool(accepted_change),
        selection_reason=selection_reason,
        objective=candidate.objective,
        feasibility_reason=candidate.failure_reason,
        mathematical_grid_fingerprint=candidate.mathematical_grid_fingerprint,
    )


def _candidate_result(
    method: str,
    source: _PreparedSource,
    candidate: _Candidate,
    *,
    attempts: tuple[TimingV3JointSearchAttempt, ...],
    sweeps_completed: int,
    search_converged: bool,
    force_reason: str | None,
) -> TimingV3JointProjectionResult:
    reason = force_reason or candidate.failure_reason
    grid = candidate.grid if reason is None else None
    integer_search_fingerprint = _fingerprint(
        [asdict(attempt) for attempt in attempts]
    )
    replay_fingerprint = _fingerprint(
        {
            "method": method,
            "mathematical_grid_fingerprint": candidate.mathematical_grid_fingerprint,
            "integer_search_fingerprint": integer_search_fingerprint,
            "final_beat_counts": candidate.beat_counts,
            "reason": reason,
        }
    )
    changed_count = sum(
        current != initial
        for current, initial in zip(candidate.beat_counts, source.initial_beat_counts)
    )
    changed_count_rate = changed_count / len(source.initial_beat_counts) if source.initial_beat_counts else 0.0
    diagnostics = TimingV3JointProjectionDiagnostics(
        method=method,
        coverage_start_ms=source.coverage_start_ms,
        coverage_end_ms=source.coverage_end_ms,
        source_section_count=len(source.segments),
        projected_section_count=len(candidate.projected_bpms) if grid is not None else 0,
        source_origin_beat=0,
        source_origin_time_ms=source.offsets_ms[0],
        grid_origin_beat=grid.origin_beat if grid is not None else None,
        grid_origin_time_ms=grid.origin_time_ms if grid is not None else None,
        first_start_beat=grid.start_beat if grid is not None else None,
        last_end_beat=grid.end_beat if grid is not None else None,
        initial_beat_counts=source.initial_beat_counts,
        final_beat_counts=candidate.beat_counts,
        projected_anchor_times_ms=candidate.projected_anchors_ms,
        source_anchor_displacements_ms=candidate.displacements_ms,
        objective=candidate.objective,
        source_surrogate_rms_beats=candidate.source_surrogate_rms_beats,
        prefix_objective=candidate.prefix_objective,
        interval_objectives=candidate.interval_objectives,
        tail_objective=candidate.tail_objective,
        maximum_anchor_displacement_adjacent_local_beats=(
            candidate.maximum_anchor_displacement_adjacent_local_beats
        ),
        maximum_relative_bpm_adjustment=candidate.maximum_relative_bpm_adjustment,
        changed_count=changed_count,
        changed_count_rate=float(changed_count_rate),
        solver=candidate.solver,
        boundary_diagnostics=candidate.boundary_diagnostics,
        search_attempts=attempts,
        sweeps_completed=sweeps_completed,
        search_converged=search_converged,
        mathematical_grid_fingerprint=candidate.mathematical_grid_fingerprint,
        integer_search_fingerprint=integer_search_fingerprint,
        replay_fingerprint=replay_fingerprint,
        failure_source_section_index=candidate.failure_boundary_index,
        failure_boundary_index=candidate.failure_boundary_index,
        failure_reason=reason,
    )
    try:
        json.dumps(asdict(diagnostics), allow_nan=False, sort_keys=True)
    except (OverflowError, TypeError, ValueError):
        diagnostics = replace(
            diagnostics,
            projected_section_count=0,
            grid_origin_beat=None,
            grid_origin_time_ms=None,
            first_start_beat=None,
            last_end_beat=None,
            projected_anchor_times_ms=(),
            source_anchor_displacements_ms=(),
            objective=None,
            source_surrogate_rms_beats=None,
            prefix_objective=None,
            interval_objectives=(),
            tail_objective=None,
            maximum_anchor_displacement_adjacent_local_beats=None,
            maximum_relative_bpm_adjustment=None,
            solver=None,
            boundary_diagnostics=(),
            mathematical_grid_fingerprint=None,
            failure_reason=REASON_DIAGNOSTICS_NOT_JSON_SAFE,
        )
        return TimingV3JointProjectionResult(
            method=method,
            grid=None,
            diagnostics=diagnostics,
            reason=REASON_DIAGNOSTICS_NOT_JSON_SAFE,
        )
    return TimingV3JointProjectionResult(method=method, grid=grid, diagnostics=diagnostics, reason=reason)


def _preparation_failure_result(
    method: str,
    failure: _PreparationFailure,
    coverage_start_ms: float,
    coverage_end_ms: float,
) -> TimingV3JointProjectionResult:
    integer_search_fingerprint = _fingerprint([])
    replay_fingerprint = _fingerprint(
        {
            "method": method,
            "mathematical_grid_fingerprint": None,
            "integer_search_fingerprint": integer_search_fingerprint,
            "final_beat_counts": (),
            "reason": failure.reason,
        }
    )
    diagnostics = TimingV3JointProjectionDiagnostics(
        method=method,
        coverage_start_ms=coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
        source_section_count=failure.source_section_count,
        projected_section_count=0,
        source_origin_beat=0,
        source_origin_time_ms=failure.source_origin_time_ms,
        grid_origin_beat=None,
        grid_origin_time_ms=None,
        first_start_beat=None,
        last_end_beat=None,
        initial_beat_counts=(),
        final_beat_counts=(),
        projected_anchor_times_ms=(),
        source_anchor_displacements_ms=(),
        objective=None,
        source_surrogate_rms_beats=None,
        prefix_objective=None,
        interval_objectives=(),
        tail_objective=None,
        maximum_anchor_displacement_adjacent_local_beats=None,
        maximum_relative_bpm_adjustment=None,
        changed_count=0,
        changed_count_rate=0.0,
        solver=None,
        boundary_diagnostics=(),
        search_attempts=(),
        sweeps_completed=0,
        search_converged=False,
        mathematical_grid_fingerprint=None,
        integer_search_fingerprint=integer_search_fingerprint,
        replay_fingerprint=replay_fingerprint,
        failure_source_section_index=failure.failure_source_section_index,
        failure_boundary_index=None,
        failure_reason=failure.reason,
    )
    return TimingV3JointProjectionResult(
        method=method,
        grid=None,
        diagnostics=diagnostics,
        reason=failure.reason,
    )


def _empty_candidate(
    beat_counts: tuple[int, ...],
    reason: str,
    *,
    solver: TimingV3JointSolverDiagnostic | None = None,
    displacements_ms: tuple[float, ...] = (),
    projected_anchors_ms: tuple[float, ...] = (),
    boundary_diagnostics: tuple[TimingV3JointBoundaryDiagnostic, ...] = (),
    failure_boundary_index: int | None = None,
) -> _Candidate:
    if solver is None:
        solver = TimingV3JointSolverDiagnostic(
            variable_count=len(beat_counts),
            matrix_inf_norm=None,
            rhs_inf_norm=None,
            solution_inf_norm=None,
            residual_inf_norm=None,
            normalized_residual=None,
            passed=False,
            failure_reason=reason,
        )
    return _Candidate(
        beat_counts=tuple(int(count) for count in beat_counts),
        displacements_ms=displacements_ms,
        projected_anchors_ms=projected_anchors_ms,
        projected_bpms=(),
        objective=None,
        source_surrogate_rms_beats=None,
        prefix_objective=None,
        interval_objectives=(),
        tail_objective=None,
        maximum_anchor_displacement_adjacent_local_beats=None,
        maximum_relative_bpm_adjustment=None,
        solver=solver,
        boundary_diagnostics=boundary_diagnostics,
        grid=None,
        mathematical_grid_fingerprint=None,
        failure_reason=reason,
        failure_boundary_index=failure_boundary_index,
    )


def _mathematical_grid_fingerprint(
    grid: TimingV3Grid,
    *,
    beat_counts: tuple[int, ...],
    projected_anchors_ms: tuple[float, ...],
    projected_bpms: tuple[float, ...],
) -> str:
    return _fingerprint(
        {
            "beat_counts": beat_counts,
            "projected_anchor_times_ms": projected_anchors_ms,
            "projected_bpms": projected_bpms,
            "grid": grid.to_dict(),
        }
    )


def _fingerprint(payload: object) -> str:
    serialized = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


__all__ = [
    "JOINT_ADJACENT_BEAT_DISPLACEMENT_LIMIT",
    "JOINT_MAX_BPM",
    "JOINT_MAX_RELATIVE_BPM_ADJUSTMENT",
    "JOINT_MIN_BPM",
    "JOINT_SOLVER_NORMALIZED_RESIDUAL_LIMIT",
    "PROJECTION_METHOD_JOINT_FIXED_COUNTS",
    "PROJECTION_METHOD_JOINT_NEARBY_COUNTS",
    "REASON_ADJACENT_BEAT_IDENTITY_DISPLACEMENT_EXCEEDED",
    "REASON_BEAT_COUNT_INVALID",
    "REASON_COVERAGE_CONSTRUCTION_FAILED",
    "REASON_DIAGNOSTICS_NOT_JSON_SAFE",
    "REASON_FINAL_BPM_CHANGED",
    "REASON_JSON_ROUNDTRIP_FAILED",
    "REASON_PROJECTED_ANCHOR_INVALID",
    "REASON_PROJECTED_BPM_OUT_OF_RANGE",
    "REASON_RELATIVE_BPM_ADJUSTMENT_EXCEEDED",
    "REASON_SCHEMA_CONSTRUCTION_FAILED",
    "REASON_SEARCH_NOT_CONVERGED",
    "REASON_SECTION_COUNT_INCREASE",
    "REASON_SOLVER_NONFINITE_COEFFICIENT",
    "REASON_SOLVER_NONPOSITIVE_PIVOT",
    "REASON_SOLVER_RESIDUAL_FAILED",
    "REASON_SOURCE_BPM_OUT_OF_RANGE",
    "REASON_SOURCE_INTERVAL_INVALID",
    "SELECTION_OBJECTIVE_IMPROVED",
    "SELECTION_TIE_BREAK_IMPROVED",
    "TimingV3JointBoundaryDiagnostic",
    "TimingV3JointProjectionDiagnostics",
    "TimingV3JointProjectionResult",
    "TimingV3JointSearchAttempt",
    "TimingV3JointSolverDiagnostic",
    "project_joint_phase_fixed_counts",
    "project_joint_phase_nearby_counts",
]
