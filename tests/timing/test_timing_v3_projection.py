from __future__ import annotations

import math
import json
from dataclasses import FrozenInstanceError, asdict

import numpy as np
import pytest

from pulsefield_model.timing.schema import FittedTimingGrid, TimingSegment
from pulsefield_model.timing.v3.projection import (
    PROJECTION_METHOD_PRESERVE_ANCHORS,
    PROJECTION_METHOD_PRESERVE_BPM,
    REASON_COUNT_OUTSIDE_FEASIBLE_RANGE,
    REASON_EMPTY_FEASIBLE_COUNT_RANGE,
    REASON_RELATIVE_BPM_ADJUSTMENT_EXCEEDED,
    REASON_SOURCE_BPM_OUT_OF_RANGE,
    REASON_SOURCE_INTERVAL_INVALID,
    project_preserve_anchors,
    project_preserve_bpm,
)
from pulsefield_model.timing.v3.schema import ConstantTimingSection, TimingV3Grid


def _grid(*segments: tuple[float, float]) -> FittedTimingGrid:
    return FittedTimingGrid(
        tuple(TimingSegment(offset_ms=offset_ms, beat_length_ms=beat_length_ms) for offset_ms, beat_length_ms in segments)
    )


def _time_at_beat(grid: TimingV3Grid, beat: int) -> float:
    return grid.time_at_beat(beat)


def _boundary_times(grid: TimingV3Grid) -> tuple[float, ...]:
    return grid.boundary_times_ms


@pytest.mark.parametrize(
    ("first_offset_ms", "expected_start_beat"),
    [
        (250.0, -1),
        (-250.0, 0),
    ],
)
def test_single_section_extends_first_lattice_and_final_boundary(
    first_offset_ms: float,
    expected_start_beat: int,
) -> None:
    result = project_preserve_anchors(_grid((first_offset_ms, 500.0)), coverage_end_ms=2300.0)

    assert result.ok
    assert result.reason is None
    assert result.method == PROJECTION_METHOD_PRESERVE_ANCHORS
    assert result.diagnostics.boundary_diagnostics == ()
    assert result.diagnostics.first_start_beat == expected_start_beat
    assert result.diagnostics.source_origin_beat == 0
    assert result.diagnostics.source_origin_time_ms == pytest.approx(first_offset_ms)
    assert result.diagnostics.grid_origin_beat == 0
    assert result.diagnostics.grid_origin_time_ms == pytest.approx(first_offset_ms)
    assert result.grid.origin_beat == 0
    assert result.grid.origin_time_ms == pytest.approx(first_offset_ms)
    assert result.grid.sections == (
        ConstantTimingSection(
            start_beat=expected_start_beat,
            end_beat=5 if first_offset_ms > 0.0 else 6,
            bpm=120.0,
        ),
    )
    times = _boundary_times(result.grid)
    assert times[0] <= 0.0
    assert times[-1] >= 2300.0


def test_frame_count_and_rate_derive_cache_support() -> None:
    result = project_preserve_anchors(_grid((0.0, 500.0)), frame_count=100, frame_rate_hz=50.0)

    assert result.ok
    assert result.diagnostics.coverage_start_ms == 0.0
    assert result.diagnostics.coverage_end_ms == 2000.0
    assert result.grid.sections[0].end_beat == 4


def test_on_lattice_jump_preserves_source_anchors_and_last_bpm() -> None:
    result = project_preserve_anchors(_grid((0.0, 500.0), (2000.0, 400.0)), coverage_end_ms=5000.0)

    assert result.ok
    assert result.grid.sections == (
        ConstantTimingSection(start_beat=0, end_beat=4, bpm=120.0),
        ConstantTimingSection(start_beat=4, end_beat=12, bpm=150.0),
    )
    diagnostic = result.diagnostics.boundary_diagnostics[0]
    assert diagnostic.implied_beat_count == pytest.approx(4.0)
    assert diagnostic.beat_count == 4
    assert diagnostic.residual_ms == pytest.approx(0.0)
    assert diagnostic.projected_left_bpm == pytest.approx(120.0)
    assert diagnostic.relative_bpm_adjustment == pytest.approx(0.0)
    assert diagnostic.right_anchor_displacement_ms == pytest.approx(0.0)
    assert _time_at_beat(result.grid, 4) == pytest.approx(2000.0)


@pytest.mark.parametrize(
    ("right_anchor_ms", "expected_residual_ms"),
    [
        (2100.0, 100.0),
        (1920.0, -80.0),
    ],
)
def test_off_lattice_boundaries_adjust_left_bpm_and_keep_anchors(
    right_anchor_ms: float,
    expected_residual_ms: float,
) -> None:
    result = project_preserve_anchors(_grid((0.0, 500.0), (right_anchor_ms, 400.0)), coverage_end_ms=5000.0)

    assert result.ok
    diagnostic = result.diagnostics.boundary_diagnostics[0]
    assert diagnostic.beat_count == 4
    assert diagnostic.residual_ms == pytest.approx(expected_residual_ms)
    assert diagnostic.projected_left_bpm == pytest.approx(60000.0 * 4.0 / right_anchor_ms)
    assert diagnostic.relative_bpm_adjustment == pytest.approx(-expected_residual_ms / right_anchor_ms)
    assert diagnostic.left_anchor_displacement_ms == pytest.approx(0.0)
    assert diagnostic.right_anchor_displacement_ms == pytest.approx(0.0)
    assert _time_at_beat(result.grid, 4) == pytest.approx(right_anchor_ms)


def test_half_up_tie_uses_three_beats_in_control_projection() -> None:
    result = project_preserve_bpm(_grid((0.0, 500.0), (1250.0, 400.0)), coverage_end_ms=3000.0)

    assert result.ok
    diagnostic = result.diagnostics.boundary_diagnostics[0]
    assert diagnostic.implied_beat_count == pytest.approx(2.5)
    assert diagnostic.beat_count == 3
    assert diagnostic.residual_ms == pytest.approx(-250.0)
    assert diagnostic.projected_left_bpm == pytest.approx(diagnostic.source_left_bpm)
    assert diagnostic.relative_bpm_adjustment == pytest.approx(0.0)
    assert diagnostic.projected_right_anchor_ms == pytest.approx(1500.0)
    assert diagnostic.right_anchor_displacement_ms == pytest.approx(250.0)


def test_near_integer_ulp_residual_does_not_change_integer_count() -> None:
    right_anchor_ms = math.nextafter(2000.0, math.inf)

    result = project_preserve_anchors(_grid((0.0, 500.0), (right_anchor_ms, 400.0)), coverage_end_ms=5000.0)

    assert result.ok
    diagnostic = result.diagnostics.boundary_diagnostics[0]
    assert diagnostic.beat_count == 4
    assert abs(diagnostic.residual_ms) < 1e-9
    assert diagnostic.relative_bpm_adjustment == pytest.approx(-diagnostic.residual_ms / right_anchor_ms)
    assert _time_at_beat(result.grid, 4) == pytest.approx(right_anchor_ms)


def test_nonzero_left_anchor_uses_interval_delta_in_projection_formula() -> None:
    result = project_preserve_anchors(
        _grid((1000.0, 500.0), (3100.0, 400.0)),
        coverage_end_ms=5000.0,
    )

    assert result.ok
    diagnostic = result.diagnostics.boundary_diagnostics[0]
    assert diagnostic.implied_beat_count == pytest.approx(4.2)
    assert diagnostic.beat_count == 4
    assert diagnostic.residual_ms == pytest.approx(100.0)
    assert diagnostic.projected_left_bpm == pytest.approx(60000.0 * 4.0 / 2100.0)
    assert diagnostic.relative_bpm_adjustment == pytest.approx(-100.0 / 2100.0)


@pytest.mark.parametrize(
    ("source_period_ms", "right_anchor_ms", "expected_reason"),
    [
        (500.0, 50.0, REASON_EMPTY_FEASIBLE_COUNT_RANGE),
        (60.0, 1000.0, REASON_COUNT_OUTSIDE_FEASIBLE_RANGE),
        (500.0, 526.5, REASON_RELATIVE_BPM_ADJUSTMENT_EXCEEDED),
    ],
)
def test_preserve_anchors_fails_closed_for_short_infeasible_and_distorted_intervals(
    source_period_ms: float,
    right_anchor_ms: float,
    expected_reason: str,
) -> None:
    result = project_preserve_anchors(_grid((0.0, source_period_ms), (right_anchor_ms, 400.0)), coverage_end_ms=5000.0)

    assert not result.ok
    assert result.grid is None
    assert result.reason == expected_reason
    assert result.diagnostics.failure_reason == expected_reason
    assert result.diagnostics.projected_section_count == 0
    assert result.diagnostics.boundary_diagnostics[-1].failure_reason == expected_reason


def test_alias_jump_is_preserved_as_distinct_constant_sections() -> None:
    result = project_preserve_anchors(
        _grid((0.0, 500.0), (1000.0, 250.0), (1500.0, 500.0)),
        coverage_end_ms=3000.0,
    )

    assert result.ok
    assert [section.bpm for section in result.grid.sections] == pytest.approx([120.0, 240.0, 120.0])
    assert [(section.start_beat, section.end_beat) for section in result.grid.sections] == [(0, 2), (2, 4), (4, 7)]
    assert _time_at_beat(result.grid, 2) == pytest.approx(1000.0)
    assert _time_at_beat(result.grid, 4) == pytest.approx(1500.0)


@pytest.mark.parametrize("invalid_section_index", [0, 1])
def test_all_source_bpms_are_guarded_and_extreme_values_fail_closed(
    invalid_section_index: int,
) -> None:
    segments = [(0.0, 500.0), (2000.0, 500.0)]
    offset_ms, _ = segments[invalid_section_index]
    segments[invalid_section_index] = (offset_ms, 1e-320)

    result = project_preserve_anchors(_grid(*segments), coverage_end_ms=5000.0)

    assert not result.ok
    assert result.reason == REASON_SOURCE_BPM_OUT_OF_RANGE
    assert result.diagnostics.failure_source_section_index == invalid_section_index
    assert result.diagnostics.boundary_diagnostics == ()
    json.dumps(asdict(result.diagnostics), allow_nan=False)


def test_family_b_does_not_normalize_an_out_of_guard_source_bpm() -> None:
    source_bpm = 19.9
    source_period_ms = 60000.0 / source_bpm
    right_anchor_ms = 60000.0 / 20.1

    result = project_preserve_anchors(
        _grid((0.0, source_period_ms), (right_anchor_ms, 500.0)),
        coverage_end_ms=5000.0,
    )

    assert not result.ok
    assert result.reason == REASON_SOURCE_BPM_OUT_OF_RANGE
    assert result.diagnostics.failure_source_section_index == 0


def test_overflowing_source_anchor_delta_fails_closed() -> None:
    with np.errstate(over="ignore"):
        source = _grid((-1e308, 500.0), (1e308, 500.0))

    result = project_preserve_anchors(source, coverage_start_ms=-1000.0, coverage_end_ms=1000.0)

    assert not result.ok
    assert result.reason == REASON_SOURCE_INTERVAL_INVALID
    assert result.diagnostics.failure_source_section_index == 0
    json.dumps(asdict(result.diagnostics), allow_nan=False)


def test_preserve_bpm_is_whole_grid_control_and_moves_anchors() -> None:
    result = project_preserve_bpm(
        _grid((0.0, 500.0), (2100.0, 400.0), (2900.0, 500.0)),
        coverage_end_ms=5000.0,
    )

    assert result.ok
    assert result.method == PROJECTION_METHOD_PRESERVE_BPM
    assert [section.bpm for section in result.grid.sections] == pytest.approx([120.0, 150.0, 120.0])
    diagnostics = result.diagnostics.boundary_diagnostics
    assert [diagnostic.beat_count for diagnostic in diagnostics] == [4, 2]
    assert [diagnostic.relative_bpm_adjustment for diagnostic in diagnostics] == pytest.approx([0.0, 0.0])
    assert [diagnostic.projected_left_bpm for diagnostic in diagnostics] == pytest.approx(
        [diagnostics[0].source_left_bpm, diagnostics[1].source_left_bpm]
    )
    assert diagnostics[0].projected_right_anchor_ms == pytest.approx(2000.0)
    assert diagnostics[0].right_anchor_displacement_ms == pytest.approx(-100.0)
    assert diagnostics[1].projected_left_anchor_ms == pytest.approx(2000.0)
    assert diagnostics[1].left_anchor_displacement_ms == pytest.approx(-100.0)
    assert diagnostics[1].projected_right_anchor_ms == pytest.approx(2800.0)
    assert diagnostics[1].right_anchor_displacement_ms == pytest.approx(-100.0)


def test_projection_result_and_diagnostics_are_immutable() -> None:
    result = project_preserve_anchors(_grid((0.0, 500.0), (2000.0, 400.0)), coverage_end_ms=5000.0)

    with pytest.raises(FrozenInstanceError):
        result.reason = "changed"
    with pytest.raises(FrozenInstanceError):
        result.diagnostics.boundary_diagnostics[0].beat_count = 5
    with pytest.raises(AttributeError):
        result.diagnostics.boundary_diagnostics.append(result.diagnostics.boundary_diagnostics[0])
