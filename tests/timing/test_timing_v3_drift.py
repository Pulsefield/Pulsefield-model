from __future__ import annotations

import pytest

from pulsefield_model.timing.evaluation.drift import (
    compare_timing_grid_drift,
    predicted_boundary_discontinuities_ms,
)
from pulsefield_model.timing.schema import FittedTimingGrid, TimingSegment


def _grid(*segments: tuple[float, float]) -> FittedTimingGrid:
    return FittedTimingGrid(
        tuple(TimingSegment(offset_ms=offset_ms, beat_length_ms=beat_length_ms) for offset_ms, beat_length_ms in segments)
    )


def test_identical_grid_has_no_offset_drift_or_boundary_seam() -> None:
    grid = _grid((0.0, 500.0))

    comparison = compare_timing_grid_drift(grid, grid, frame_count=5000)

    assert comparison.initial_signed_phase_error_ms == pytest.approx(0.0)
    assert comparison.endpoint_relative_drift_ms == pytest.approx(0.0)
    assert comparison.max_abs_prefix_relative_drift_ms == pytest.approx(0.0)
    assert comparison.drift_slope_ms_per_minute == pytest.approx(0.0)
    assert comparison.predicted_boundary_count == 0


def test_stable_offset_is_separate_from_relative_drift() -> None:
    oracle = _grid((0.0, 500.0))
    predicted = _grid((25.0, 500.0))

    comparison = compare_timing_grid_drift(predicted, oracle, frame_count=5000)

    assert abs(comparison.initial_signed_phase_error_ms) == pytest.approx(25.0)
    assert comparison.endpoint_relative_drift_ms == pytest.approx(0.0, abs=1e-8)
    assert comparison.max_abs_prefix_relative_drift_ms == pytest.approx(0.0, abs=1e-8)


def test_tempo_error_accumulates_and_window_metrics_observe_it() -> None:
    oracle = _grid((0.0, 500.0))
    predicted = _grid((0.0, 501.0))

    comparison = compare_timing_grid_drift(predicted, oracle, frame_count=6000)

    assert abs(comparison.endpoint_relative_drift_ms) > 200.0
    assert comparison.max_abs_prefix_relative_drift_ms >= abs(comparison.endpoint_relative_drift_ms)
    assert abs(comparison.drift_slope_ms_per_minute) > 50.0
    assert comparison.p90_abs_30s_relative_drift_ms > 40.0
    assert comparison.p90_abs_60s_relative_drift_ms > comparison.p90_abs_30s_relative_drift_ms


def test_boundary_discontinuity_is_measured_on_preceding_lattice() -> None:
    aligned = _grid((0.0, 500.0), (2000.0, 400.0))
    off_lattice = _grid((0.0, 500.0), (2125.0, 400.0))

    assert predicted_boundary_discontinuities_ms(aligned) == pytest.approx((0.0,))
    assert predicted_boundary_discontinuities_ms(off_lattice) == pytest.approx((125.0,))

    comparison = compare_timing_grid_drift(off_lattice, aligned, frame_count=1000)
    assert comparison.predicted_boundary_count == 1
    assert comparison.max_predicted_boundary_discontinuity_ms == pytest.approx(125.0)
