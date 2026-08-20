from __future__ import annotations

import json
import math

import pytest

from pulsefield_model.timing.evaluation.source_projection import (
    SOURCE_PROJECTION_COMPARISON_SCHEMA,
    compare_timing_v3_projection_to_source,
)
from pulsefield_model.timing.schema import FittedTimingGrid, TimingSegment
from pulsefield_model.timing.v3.schema import ConstantTimingSection, TimingV3Grid


def _grid(*segments: tuple[float, float]) -> FittedTimingGrid:
    return FittedTimingGrid(
        tuple(
            TimingSegment(offset_ms=offset_ms, beat_length_ms=beat_length_ms)
            for offset_ms, beat_length_ms in segments
        )
    )


def test_identical_source_projection_has_zero_displacement() -> None:
    source = _grid((0.0, 500.0), (2000.0, 400.0))

    result = compare_timing_v3_projection_to_source(
        source,
        source,
        frame_count=300,
    )

    assert result.wrapped_phase_rms_beats == pytest.approx(0.0)
    assert result.wrapped_phase_p90_ms == pytest.approx(0.0)
    assert result.local_bpm_mae == pytest.approx(0.0)
    assert result.endpoint_relative_drift_ms == pytest.approx(0.0)
    assert result.active_section_disagreement_frame_count == 0
    assert result.moved_paired_boundary_count == 0
    payload = result.to_dict()
    assert payload["schema"] == SOURCE_PROJECTION_COMPARISON_SCHEMA
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_tempo_change_reports_wrapped_phase_bpm_and_accumulated_drift() -> None:
    source = _grid((0.0, 500.0))
    candidate = _grid((0.0, 505.0))

    result = compare_timing_v3_projection_to_source(
        source,
        candidate,
        frame_count=3000,
    )

    assert result.wrapped_phase_mean_beats > 0.0
    assert result.wrapped_phase_rms_beats >= result.wrapped_phase_mean_beats
    assert result.wrapped_phase_max_beats <= 0.5 + 1e-12
    assert result.local_bpm_mae == pytest.approx(abs(120.0 - 60000.0 / 505.0))
    assert abs(result.endpoint_relative_drift_ms) > 500.0
    assert result.max_abs_prefix_relative_drift_ms >= abs(
        result.endpoint_relative_drift_ms
    )
    assert result.active_section_disagreement_fraction == pytest.approx(0.0)


def test_moved_boundary_reports_exact_active_section_disagreement() -> None:
    source = _grid((0.0, 500.0), (2000.0, 400.0))
    candidate = _grid((0.0, 525.0), (2100.0, 400.0))

    result = compare_timing_v3_projection_to_source(
        source,
        candidate,
        frame_count=200,
    )

    # Frame centers 2010, 2030, 2050, 2070, and 2090 ms lie between
    # the source and candidate change anchors.
    assert result.active_section_disagreement_frame_count == 5
    assert result.active_section_disagreement_fraction == pytest.approx(5.0 / 200.0)
    assert result.moved_paired_boundary_count == 1
    assert result.unmatched_boundary_count == 0


def test_accepts_timing_v3_grid_and_uses_its_derived_boundaries() -> None:
    source = _grid((0.0, 500.0), (2000.0, 400.0))
    candidate = TimingV3Grid(
        origin_beat=0,
        origin_time_ms=0.0,
        coverage_start_ms=0.0,
        coverage_end_ms=4000.0,
        sections=(
            ConstantTimingSection(start_beat=0, end_beat=4, bpm=120.0),
            ConstantTimingSection(start_beat=4, end_beat=9, bpm=150.0),
        ),
    )

    result = compare_timing_v3_projection_to_source(
        source,
        candidate,
        frame_count=200,
    )

    assert result.wrapped_phase_max_beats == pytest.approx(0.0)
    assert result.candidate_section_count == 2


@pytest.mark.parametrize("frame_count", [0, -1, True, 1.5])
def test_rejects_invalid_frame_count(frame_count: object) -> None:
    source = _grid((0.0, 500.0))

    with pytest.raises(ValueError, match="frame_count"):
        compare_timing_v3_projection_to_source(  # type: ignore[arg-type]
            source,
            source,
            frame_count=frame_count,
        )


def test_all_serialized_float_diagnostics_are_finite() -> None:
    source = _grid((17.0, 431.0), (1900.0, 503.0))
    candidate = _grid((17.0, 430.0), (1905.0, 503.0))

    payload = compare_timing_v3_projection_to_source(
        source,
        candidate,
        frame_count=400,
    ).to_dict()

    assert all(
        math.isfinite(value)
        for value in payload.values()
        if isinstance(value, float)
    )
