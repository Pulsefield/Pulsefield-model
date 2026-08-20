from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, asdict, replace

import numpy as np
import pytest

from pulsefield_model.timing.schema import FittedTimingGrid, TimingSegment
from pulsefield_model.timing.v3 import joint_projection
from pulsefield_model.timing.v3.joint_projection import (
    JOINT_SOLVER_NORMALIZED_RESIDUAL_LIMIT,
    PROJECTION_METHOD_JOINT_FIXED_COUNTS,
    PROJECTION_METHOD_JOINT_NEARBY_COUNTS,
    REASON_ADJACENT_BEAT_IDENTITY_DISPLACEMENT_EXCEEDED,
    REASON_PROJECTED_BPM_OUT_OF_RANGE,
    REASON_RELATIVE_BPM_ADJUSTMENT_EXCEEDED,
    REASON_SEARCH_NOT_CONVERGED,
    REASON_SOLVER_NONFINITE_COEFFICIENT,
    REASON_SOLVER_NONPOSITIVE_PIVOT,
    REASON_SOLVER_RESIDUAL_FAILED,
    REASON_SOURCE_BPM_OUT_OF_RANGE,
    REASON_SOURCE_INTERVAL_INVALID,
    SELECTION_OBJECTIVE_IMPROVED,
    SELECTION_TIE_BREAK_IMPROVED,
    project_joint_phase_fixed_counts,
    project_joint_phase_nearby_counts,
)
from pulsefield_model.timing.v3.schema import (
    ConstantTimingSection,
    TimingV3Grid,
    roundtrip_seam_tolerance_ms,
)


def _grid(*segments: tuple[float, float]) -> FittedTimingGrid:
    return FittedTimingGrid(
        tuple(
            TimingSegment(offset_ms=offset_ms, beat_length_ms=beat_length_ms)
            for offset_ms, beat_length_ms in segments
        )
    )


@pytest.mark.parametrize(
    ("project", "method"),
    [
        (project_joint_phase_fixed_counts, PROJECTION_METHOD_JOINT_FIXED_COUNTS),
        (project_joint_phase_nearby_counts, PROJECTION_METHOD_JOINT_NEARBY_COUNTS),
    ],
)
def test_one_section_returns_exact_source_lattice_and_zero_objective(
    project,
    method: str,
) -> None:
    result = project(_grid((250.0, 500.0)), coverage_end_ms=2300.0)

    assert result.ok
    assert result.reason is None
    assert result.method == method
    assert result.grid == TimingV3Grid(
        origin_beat=0,
        origin_time_ms=250.0,
        sections=(ConstantTimingSection(start_beat=-1, end_beat=5, bpm=120.0),),
        coverage_start_ms=0.0,
        coverage_end_ms=2300.0,
    )
    diagnostics = result.diagnostics
    assert diagnostics.initial_beat_counts == ()
    assert diagnostics.final_beat_counts == ()
    assert diagnostics.projected_anchor_times_ms == (250.0,)
    assert diagnostics.source_anchor_displacements_ms == (0.0,)
    assert diagnostics.objective == 0.0
    assert diagnostics.source_surrogate_rms_beats == 0.0
    assert diagnostics.prefix_objective == 0.0
    assert diagnostics.interval_objectives == ()
    assert diagnostics.tail_objective == 0.0
    assert diagnostics.solver is not None
    assert diagnostics.solver.variable_count == 0
    assert diagnostics.solver.normalized_residual == 0.0
    assert diagnostics.solver.passed
    assert diagnostics.search_converged
    assert diagnostics.sweeps_completed == 0


def test_c0_two_section_solution_matches_frozen_quadratic() -> None:
    source = _grid((0.0, 500.0), (2100.0, 400.0))

    result = project_joint_phase_fixed_counts(source, coverage_end_ms=5000.0)

    assert result.ok
    diagnostics = result.diagnostics
    assert diagnostics.initial_beat_counts == (4,)
    assert diagnostics.final_beat_counts == (4,)
    residual_ms = 2100.0 - 4 * 500.0
    interval_weight = 2100.0 / (3.0 * 500.0**2)
    tail_weight = (5000.0 - 2100.0) / 400.0**2
    expected_displacement_ms = -interval_weight * residual_ms / (interval_weight + tail_weight)
    expected_interval_objective = interval_weight * (expected_displacement_ms + residual_ms) ** 2
    expected_tail_objective = tail_weight * expected_displacement_ms**2

    assert diagnostics.source_anchor_displacements_ms == pytest.approx(
        (0.0, expected_displacement_ms)
    )
    assert diagnostics.interval_objectives == pytest.approx((expected_interval_objective,))
    assert diagnostics.tail_objective == pytest.approx(expected_tail_objective)
    assert diagnostics.objective == pytest.approx(
        expected_interval_objective + expected_tail_objective
    )
    boundary = diagnostics.boundary_diagnostics[0]
    assert boundary.original_residual_ms == pytest.approx(100.0)
    assert boundary.objective_contribution > 0.0
    assert boundary.projected_right_anchor_ms == pytest.approx(
        2100.0 + expected_displacement_ms
    )
    assert result.grid.time_at_beat(4) == pytest.approx(boundary.projected_right_anchor_ms)


def test_prefix_term_is_included_in_solver_and_reported_separately() -> None:
    source = _grid((1000.0, 500.0), (3100.0, 400.0))

    result = project_joint_phase_fixed_counts(
        source,
        coverage_start_ms=0.0,
        coverage_end_ms=5000.0,
    )

    assert result.ok
    residual_ms = 100.0
    interval_weight = 2100.0 / (3.0 * 500.0**2)
    prefix_weight = 1000.0**3 / (3.0 * 4**2 * 500.0**4)
    tail_weight = (5000.0 - 3100.0) / 400.0**2
    expected_displacement_ms = -(
        (interval_weight + prefix_weight) * residual_ms
        / (interval_weight + prefix_weight + tail_weight)
    )

    diagnostics = result.diagnostics
    assert diagnostics.source_anchor_displacements_ms[1] == pytest.approx(
        expected_displacement_ms
    )
    assert diagnostics.prefix_objective == pytest.approx(
        prefix_weight * (expected_displacement_ms + residual_ms) ** 2
    )
    assert diagnostics.prefix_objective > 0.0
    assert result.grid.start_time_ms <= 0.0
    assert result.grid.coverage_start_ms == 0.0


def test_multisection_solver_passes_frozen_normalized_residual_check() -> None:
    result = project_joint_phase_fixed_counts(
        _grid(
            (0.0, 500.0),
            (2020.0, 400.0),
            (2830.0, 600.0),
            (5830.0, 480.0),
        ),
        coverage_end_ms=10000.0,
    )

    assert result.ok
    solver = result.diagnostics.solver
    assert solver is not None
    assert solver.variable_count == 3
    assert solver.passed
    assert solver.normalized_residual is not None
    assert solver.normalized_residual <= JOINT_SOLVER_NORMALIZED_RESIDUAL_LIMIT
    assert len(result.diagnostics.interval_objectives) == 3
    assert result.diagnostics.objective == pytest.approx(
        result.diagnostics.prefix_objective
        + math.fsum(result.diagnostics.interval_objectives)
        + result.diagnostics.tail_objective
    )
    assert result.grid.sections[-1].bpm == pytest.approx(125.0)


def test_fixed_count_thomas_solution_matches_independent_dense_reference() -> None:
    source_grid = _grid(
        (750.0, 500.0),
        (2770.0, 400.0),
        (3580.0, 600.0),
        (6580.0, 480.0),
    )
    prepared = joint_projection._prepare_source(
        source_grid,
        coverage_start_ms=0.0,
        coverage_end_ms=10000.0,
    )
    assert isinstance(prepared, joint_projection._PreparedSource)
    counts = prepared.initial_beat_counts
    residuals = tuple(
        delta_ms - count * period_ms
        for delta_ms, count, period_ms in zip(
            prepared.deltas_ms,
            counts,
            prepared.periods_ms,
        )
    )

    variable_count = len(counts)
    dense_q = np.zeros((variable_count, variable_count), dtype=np.float64)
    linear = np.zeros(variable_count, dtype=np.float64)
    for interval_index, (delta_ms, period_ms, residual_ms) in enumerate(
        zip(prepared.deltas_ms, prepared.periods_ms, residuals)
    ):
        weight = delta_ms / (3.0 * period_ms**2)
        dense_q[interval_index, interval_index] += weight
        linear[interval_index] += 2.0 * weight * residual_ms
        if interval_index > 0:
            dense_q[interval_index - 1, interval_index - 1] += weight
            dense_q[interval_index - 1, interval_index] += 0.5 * weight
            dense_q[interval_index, interval_index - 1] += 0.5 * weight
            linear[interval_index - 1] += weight * residual_ms

    prefix_length_ms = prepared.offsets_ms[0] - prepared.coverage_start_ms
    prefix_weight = prefix_length_ms**3 / (
        3.0 * counts[0] ** 2 * prepared.periods_ms[0] ** 4
    )
    dense_q[0, 0] += prefix_weight
    linear[0] += 2.0 * prefix_weight * residuals[0]
    tail_length_ms = prepared.coverage_end_ms - prepared.offsets_ms[-1]
    dense_q[-1, -1] += tail_length_ms / prepared.periods_ms[-1] ** 2
    dense_reference = np.linalg.solve(dense_q, -0.5 * linear)

    thomas_solution, solver = joint_projection._solve_displacements(
        prepared,
        counts,
        residuals,
    )

    assert solver.passed
    assert thomas_solution is not None
    assert thomas_solution == pytest.approx(dense_reference, rel=1e-13, abs=1e-13)


def test_half_up_count_rule_does_not_use_bankers_rounding() -> None:
    result = project_joint_phase_fixed_counts(
        _grid((0.0, 500.0), (1250.0, 500.0)),
        coverage_end_ms=1251.0,
    )

    assert result.ok
    boundary = result.diagnostics.boundary_diagnostics[0]
    assert boundary.source_implied_beat_count == pytest.approx(2.5)
    assert boundary.initial_beat_count == 3
    assert boundary.beat_count == 3
    assert boundary.original_residual_ms == pytest.approx(-250.0)


@pytest.mark.parametrize(
    ("right_anchor_ms", "residual_sign", "displacement_sign"),
    [
        (1920.0, -1, 1),
        (math.nextafter(2000.0, math.inf), 1, -1),
        (math.nextafter(2000.0, -math.inf), -1, 1),
    ],
)
def test_negative_residual_and_one_ulp_edges_keep_half_up_count(
    right_anchor_ms: float,
    residual_sign: int,
    displacement_sign: int,
) -> None:
    result = project_joint_phase_fixed_counts(
        _grid((0.0, 500.0), (right_anchor_ms, 400.0)),
        coverage_end_ms=5000.0,
    )

    assert result.ok
    boundary = result.diagnostics.boundary_diagnostics[0]
    assert boundary.beat_count == 4
    assert math.copysign(1.0, boundary.original_residual_ms) == residual_sign
    assert math.copysign(1.0, boundary.projected_right_anchor_displacement_ms) == (
        displacement_sign
    )
    if right_anchor_ms != 1920.0:
        assert abs(boundary.original_residual_ms) == math.ulp(2000.0)


def test_c0_fails_closed_when_five_percent_bpm_guard_is_exceeded() -> None:
    result = project_joint_phase_fixed_counts(
        _grid((0.0, 500.0), (526.5, 400.0)),
        coverage_end_ms=100000.0,
    )

    assert not result.ok
    assert result.grid is None
    assert result.reason == REASON_RELATIVE_BPM_ADJUSTMENT_EXCEEDED
    assert result.diagnostics.failure_reason == REASON_RELATIVE_BPM_ADJUSTMENT_EXCEEDED
    assert result.diagnostics.failure_boundary_index == 0
    assert result.diagnostics.boundary_diagnostics[0].failure_reason == (
        REASON_RELATIVE_BPM_ADJUSTMENT_EXCEEDED
    )
    assert abs(result.diagnostics.boundary_diagnostics[0].relative_bpm_adjustment) > 0.05
    json.dumps(asdict(result.diagnostics), allow_nan=False)


def test_adjacent_source_lattices_reject_a_moved_anchor_beyond_half_beat() -> None:
    result = project_joint_phase_fixed_counts(
        _grid(
            (0.0, 300.0),
            (6060.0, 60.0),
            (7230.6, 60.0),
        ),
        coverage_end_ms=7231.6,
    )

    assert not result.ok
    assert result.reason == REASON_ADJACENT_BEAT_IDENTITY_DISPLACEMENT_EXCEEDED
    assert result.diagnostics.failure_boundary_index == 1
    boundary = result.diagnostics.boundary_diagnostics[1]
    assert abs(boundary.relative_bpm_adjustment) < 0.05
    assert boundary.right_anchor_on_left_lattice_displacement_beats < 0.5
    assert boundary.right_anchor_on_right_lattice_displacement_beats > 0.5 + 1e-12


def test_c1_coordinate_search_changes_count_and_improves_whole_grid_objective() -> None:
    source = _grid(
        (124.26990834151923, 630.4738687709103),
        (22456.980935883334, 726.1222276455969),
        (37477.522888631196, 713.253311892933),
    )
    coverage_end_ms = 57335.84771997643

    c0 = project_joint_phase_fixed_counts(source, coverage_end_ms=coverage_end_ms)
    c1 = project_joint_phase_nearby_counts(source, coverage_end_ms=coverage_end_ms)

    assert c0.ok and c1.ok
    assert c0.diagnostics.final_beat_counts == (35, 21)
    assert c1.diagnostics.final_beat_counts == (36, 21)
    assert c1.diagnostics.objective < c0.diagnostics.objective
    assert c1.diagnostics.changed_count == 1
    assert c1.diagnostics.changed_count_rate == pytest.approx(0.5)
    assert c1.diagnostics.search_converged
    assert c1.diagnostics.sweeps_completed == 2

    attempts = c1.diagnostics.search_attempts
    assert attempts[0].sweep_index == 0
    assert attempts[0].boundary_index is None
    assert not attempts[0].accepted_change
    first_sweep_boundary_zero = [
        attempt.candidate_beat_count
        for attempt in attempts
        if attempt.sweep_index == 1 and attempt.boundary_index == 0
    ]
    assert first_sweep_boundary_zero == [34, 36]
    accepted_changes = [attempt for attempt in attempts if attempt.accepted_change]
    assert [(attempt.boundary_index, attempt.candidate_beat_count) for attempt in accepted_changes] == [
        (0, 36)
    ]
    assert accepted_changes[0].selection_reason == SELECTION_OBJECTIVE_IMPROVED
    assert any(attempt.feasibility_reason is not None for attempt in attempts)


def test_within_tolerance_selection_is_explicitly_tagged_as_tie_break() -> None:
    prepared = joint_projection._prepare_source(
        _grid((0.0, 500.0), (2100.0, 400.0)),
        coverage_start_ms=0.0,
        coverage_end_ms=5000.0,
    )
    assert isinstance(prepared, joint_projection._PreparedSource)
    current = joint_projection._evaluate_candidate(prepared, prepared.initial_beat_counts)
    assert current.ok
    assert current.maximum_anchor_displacement_adjacent_local_beats is not None
    tied_with_better_anchor_key = replace(
        current,
        maximum_anchor_displacement_adjacent_local_beats=math.nextafter(
            current.maximum_anchor_displacement_adjacent_local_beats,
            -math.inf,
        ),
    )

    reason = joint_projection._candidate_selection_reason(
        tied_with_better_anchor_key,
        current,
        initial_counts=prepared.initial_beat_counts,
    )

    assert reason == SELECTION_TIE_BREAK_IMPROVED


@pytest.mark.parametrize(
    ("better_counts", "worse_counts"),
    [
        pytest.param((11, 10), (11, 9), id="fewer-changed-counts"),
        pytest.param((12, 10), (13, 10), id="smaller-total-delta"),
        pytest.param((9, 11), (11, 9), id="lexicographic-vector"),
    ],
)
def test_tie_break_hierarchy_selects_each_remaining_ordering_level(
    better_counts: tuple[int, ...],
    worse_counts: tuple[int, ...],
) -> None:
    prepared = joint_projection._prepare_source(
        _grid(
            (0.0, 500.0),
            (5000.0, 400.0),
            (9000.0, 600.0),
        ),
        coverage_start_ms=0.0,
        coverage_end_ms=10000.0,
    )
    assert isinstance(prepared, joint_projection._PreparedSource)
    assert prepared.initial_beat_counts == (10, 10)
    base = joint_projection._evaluate_candidate(prepared, prepared.initial_beat_counts)
    assert base.ok
    better = replace(base, beat_counts=better_counts)
    worse = replace(base, beat_counts=worse_counts)

    forward_reason = joint_projection._candidate_selection_reason(
        better,
        worse,
        initial_counts=prepared.initial_beat_counts,
    )
    reverse_reason = joint_projection._candidate_selection_reason(
        worse,
        better,
        initial_counts=prepared.initial_beat_counts,
    )

    assert forward_reason == SELECTION_TIE_BREAK_IMPROVED
    assert reverse_reason is None


def test_c1_replay_is_byte_deterministic_for_grid_and_integer_search() -> None:
    source = _grid(
        (124.26990834151923, 630.4738687709103),
        (22456.980935883334, 726.1222276455969),
        (37477.522888631196, 713.253311892933),
    )
    kwargs = {"coverage_end_ms": 57335.84771997643}

    first = project_joint_phase_nearby_counts(source, **kwargs)
    second = project_joint_phase_nearby_counts(source, **kwargs)

    assert first.ok and second.ok
    assert first.grid.to_dict() == second.grid.to_dict()
    assert first.diagnostics.search_attempts == second.diagnostics.search_attempts
    assert first.diagnostics.mathematical_grid_fingerprint == (
        second.diagnostics.mathematical_grid_fingerprint
    )
    assert first.diagnostics.integer_search_fingerprint == (
        second.diagnostics.integer_search_fingerprint
    )
    assert first.diagnostics.replay_fingerprint == second.diagnostics.replay_fingerprint


def test_c1_fails_instead_of_accepting_a_changing_sweep_at_search_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _grid(
        (124.26990834151923, 630.4738687709103),
        (22456.980935883334, 726.1222276455969),
        (37477.522888631196, 713.253311892933),
    )

    monkeypatch.setattr(
        joint_projection,
        "_candidate_selection_reason",
        lambda left, right, *, initial_counts: SELECTION_OBJECTIVE_IMPROVED,
    )
    result = project_joint_phase_nearby_counts(
        source,
        coverage_end_ms=57335.84771997643,
    )

    assert not result.ok
    assert result.grid is None
    assert result.reason == REASON_SEARCH_NOT_CONVERGED
    assert not result.diagnostics.search_converged
    assert result.diagnostics.sweeps_completed == len(source.segments) + 1
    assert result.diagnostics.search_attempts[-1].sweep_index == len(source.segments) + 1
    assert any(
        attempt.accepted_change
        for attempt in result.diagnostics.search_attempts
        if attempt.sweep_index == len(source.segments) + 1
    )


def test_c1_aborts_before_search_when_initial_c0_solution_is_infeasible() -> None:
    result = project_joint_phase_nearby_counts(
        _grid((0.0, 500.0), (526.5, 400.0)),
        coverage_end_ms=100000.0,
    )

    assert not result.ok
    assert result.reason == REASON_RELATIVE_BPM_ADJUSTMENT_EXCEEDED
    assert result.diagnostics.initial_beat_counts == (1,)
    assert result.diagnostics.final_beat_counts == (1,)
    assert result.diagnostics.sweeps_completed == 0
    assert not result.diagnostics.search_converged
    assert len(result.diagnostics.search_attempts) == 1
    initial_attempt = result.diagnostics.search_attempts[0]
    assert initial_attempt.sweep_index == 0
    assert initial_attempt.boundary_index is None
    assert not initial_attempt.accepted_change
    assert initial_attempt.feasibility_reason == REASON_RELATIVE_BPM_ADJUSTMENT_EXCEEDED


def test_frame_count_and_rate_define_exact_cache_support() -> None:
    result = project_joint_phase_fixed_counts(
        _grid((0.0, 500.0), (1000.0, 400.0)),
        frame_count=100,
        frame_rate_hz=50.0,
    )

    assert result.ok
    assert result.grid.coverage_start_ms == 0.0
    assert result.grid.coverage_end_ms == 2000.0
    assert result.grid.start_time_ms <= 0.0
    assert result.grid.end_time_ms >= 2000.0


def test_schema_and_json_roundtrip_preserve_exact_section_seams() -> None:
    result = project_joint_phase_fixed_counts(
        _grid(
            (250.0, 500.0),
            (2250.0, 400.0),
            (3050.0, 600.0),
        ),
        coverage_end_ms=6000.0,
    )

    assert result.ok
    for section_index in range(len(result.grid.sections) - 1):
        assert result.grid.section_end_time_ms(section_index) == (
            result.grid.section_start_time_ms(section_index + 1)
        )
    restored = TimingV3Grid.from_dict(json.loads(json.dumps(result.grid.to_dict())))
    for original_ms, restored_ms in zip(
        result.grid.boundary_times_ms,
        restored.boundary_times_ms,
    ):
        assert abs(restored_ms - original_ms) <= roundtrip_seam_tolerance_ms(original_ms)


@pytest.mark.parametrize("invalid_section_index", [0, 1])
def test_every_source_bpm_is_guarded_before_solving(invalid_section_index: int) -> None:
    segments = [(0.0, 500.0), (2000.0, 500.0)]
    offset_ms, _ = segments[invalid_section_index]
    segments[invalid_section_index] = (offset_ms, 1e-320)

    result = project_joint_phase_fixed_counts(
        _grid(*segments),
        coverage_end_ms=5000.0,
    )

    assert not result.ok
    assert result.reason == REASON_SOURCE_BPM_OUT_OF_RANGE
    assert result.diagnostics.failure_source_section_index == invalid_section_index
    assert result.diagnostics.solver is None
    json.dumps(asdict(result.diagnostics), allow_nan=False)


def test_overflowing_source_interval_fails_before_quadratic_assembly() -> None:
    with np.errstate(over="ignore"):
        source = _grid((-1e308, 500.0), (1e308, 500.0))

    result = project_joint_phase_fixed_counts(
        source,
        coverage_start_ms=-1000.0,
        coverage_end_ms=1000.0,
    )

    assert not result.ok
    assert result.reason == REASON_SOURCE_INTERVAL_INVALID
    assert result.diagnostics.failure_source_section_index == 0
    json.dumps(asdict(result.diagnostics), allow_nan=False)


def test_nonfinite_quadratic_coefficients_fail_closed() -> None:
    result = project_joint_phase_fixed_counts(
        _grid((0.0, 500.0), (2000.0, 400.0)),
        coverage_start_ms=-1e308,
        coverage_end_ms=3000.0,
    )

    assert not result.ok
    assert result.reason == REASON_SOLVER_NONFINITE_COEFFICIENT
    assert result.diagnostics.solver is not None
    assert not result.diagnostics.solver.passed
    assert result.diagnostics.solver.failure_reason == REASON_SOLVER_NONFINITE_COEFFICIENT
    json.dumps(asdict(result.diagnostics), allow_nan=False)


def test_projected_bpm_out_of_range_precedes_relative_adjustment_guard() -> None:
    result = project_joint_phase_fixed_counts(
        _grid((0.0, 60.0), (59.0, 500.0)),
        coverage_end_ms=1000.0,
    )

    assert not result.ok
    assert result.reason == REASON_PROJECTED_BPM_OUT_OF_RANGE
    assert result.diagnostics.failure_boundary_index == 0
    boundary = result.diagnostics.boundary_diagnostics[0]
    assert boundary.projected_left_bpm > 1000.0
    assert abs(boundary.relative_bpm_adjustment) < 0.05
    assert boundary.failure_reason == REASON_PROJECTED_BPM_OUT_OF_RANGE


def test_solver_residual_failure_is_tagged_and_stops_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        joint_projection,
        "_thomas_solve",
        lambda diagonal, off_diagonal, rhs: (np.zeros_like(rhs), None),
    )

    result = project_joint_phase_fixed_counts(
        _grid((0.0, 500.0), (2100.0, 400.0)),
        coverage_end_ms=5000.0,
    )

    assert not result.ok
    assert result.grid is None
    assert result.reason == REASON_SOLVER_RESIDUAL_FAILED
    solver = result.diagnostics.solver
    assert solver is not None
    assert not solver.passed
    assert solver.failure_reason == REASON_SOLVER_RESIDUAL_FAILED
    assert solver.normalized_residual is not None
    assert solver.normalized_residual > JOINT_SOLVER_NORMALIZED_RESIDUAL_LIMIT


def test_thomas_solver_rejects_nonpositive_pivot() -> None:
    solution, reason = joint_projection._thomas_solve(
        np.asarray([1.0, -1.0], dtype=np.float64),
        np.asarray([0.0], dtype=np.float64),
        np.asarray([1.0, 1.0], dtype=np.float64),
    )

    assert solution is None
    assert reason == REASON_SOLVER_NONPOSITIVE_PIVOT


def test_result_diagnostics_are_immutable_finite_and_json_safe() -> None:
    result = project_joint_phase_nearby_counts(
        _grid((0.0, 500.0), (2100.0, 400.0)),
        coverage_end_ms=5000.0,
    )

    assert result.ok
    serialized = json.dumps(asdict(result.diagnostics), allow_nan=False, sort_keys=True)
    restored = json.loads(serialized)
    assert restored["failure_reason"] is None
    assert len(restored["replay_fingerprint"]) == 64
    assert len(restored["mathematical_grid_fingerprint"]) == 64
    assert len(restored["integer_search_fingerprint"]) == 64

    with pytest.raises(FrozenInstanceError):
        result.reason = "changed"
    with pytest.raises(FrozenInstanceError):
        result.diagnostics.changed_count = 99
    with pytest.raises(AttributeError):
        result.diagnostics.search_attempts.append(result.diagnostics.search_attempts[0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"coverage_end_ms": 0.0},
        {"coverage_end_ms": math.inf},
        {"frame_count": True, "frame_rate_hz": 50.0},
        {"frame_count": 100, "frame_rate_hz": 0.0},
        {"coverage_end_ms": 1000.0, "frame_count": 100, "frame_rate_hz": 50.0},
    ],
)
def test_invalid_coverage_configuration_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        project_joint_phase_fixed_counts(_grid((0.0, 500.0)), **kwargs)
