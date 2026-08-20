from __future__ import annotations

import json

import pytest

from pulsefield_model.timing.evaluation.curve_metrics import (
    WEAK_ORACLE_CURVE_METRICS_SCHEMA,
    classify_predicted_curve,
    evaluate_curve_against_weak_oracle,
)
from pulsefield_model.timing.schema import FittedTimingGrid, TimingSegment
from pulsefield_model.timing.v3.analytic_curve import (
    ConstantTempoSection,
    LinearTimeRampSection,
    PhaseContinuousTimingCurve,
)
from pulsefield_model.timing.v3.schema import ConstantTimingSection, TimingV3Grid


def _weak_grid(*segments: tuple[float, float]) -> FittedTimingGrid:
    return FittedTimingGrid(
        tuple(
            TimingSegment(offset_ms=offset_ms, beat_length_ms=beat_length_ms)
            for offset_ms, beat_length_ms in segments
        )
    )


def _v3_grid(*sections: tuple[int, int, float], origin_time_ms: float = 0.0) -> TimingV3Grid:
    payload = tuple(
        ConstantTimingSection(start_beat=start, end_beat=end, bpm=bpm)
        for start, end, bpm in sections
    )
    return TimingV3Grid(
        origin_beat=payload[0].start_beat,
        origin_time_ms=origin_time_ms,
        sections=payload,
        coverage_start_ms=origin_time_ms,
        coverage_end_ms=origin_time_ms
        + sum((end - start) * 60_000.0 / bpm for start, end, bpm in sections),
    )


def test_constant_exact_metrics_are_explicitly_weak_oracle_named() -> None:
    predicted = _v3_grid((0, 12, 120.0))
    metrics = evaluate_curve_against_weak_oracle(
        predicted,
        _weak_grid((0.0, 500.0)),
        weak_oracle_class="constant",
        coverage_start_ms=0.0,
        coverage_end_ms=6000.0,
    )

    assert metrics.schema == WEAK_ORACLE_CURVE_METRICS_SCHEMA
    assert metrics.predicted_class == "constant"
    assert metrics.weak_oracle_class_exact
    assert metrics.weak_oracle_constant_exact_hit is True
    assert metrics.weak_oracle_jump_exact_hit is None
    assert metrics.weak_oracle_initial_signed_offset_ms == pytest.approx(0.0)
    assert metrics.weak_oracle_phase_p90_ms == pytest.approx(0.0)
    assert metrics.weak_oracle_direct_bpm_coverage == pytest.approx(1.0)
    assert metrics.weak_oracle_alias_bpm_coverage == pytest.approx(1.0)
    assert metrics.weak_oracle_ramp_accuracy is None
    payload = metrics.to_dict()
    assert all(
        name.startswith("weak_oracle_")
        for name in payload
        if any(token in name for token in ("phase", "offset", "bpm", "boundary", "jump", "ramp"))
        and name != "maximum_phase_continuity_error_ms"
    )
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_weak_oracle_grid_extrapolates_back_before_first_redline() -> None:
    predicted = _v3_grid((0, 12, 120.0))
    metrics = evaluate_curve_against_weak_oracle(
        predicted,
        _weak_grid((1000.0, 500.0)),
        weak_oracle_class="constant",
        coverage_start_ms=0.0,
        coverage_end_ms=6000.0,
    )

    # A redline offset differs by two whole beats, so phase remains exact.
    assert metrics.weak_oracle_initial_signed_offset_ms == pytest.approx(0.0)
    assert metrics.weak_oracle_constant_exact_hit is True


def test_jump_exact_uses_one_to_one_boundary_direction_and_side_bpms() -> None:
    predicted = _v3_grid((0, 4, 120.0), (4, 14, 150.0))
    metrics = evaluate_curve_against_weak_oracle(
        predicted,
        _weak_grid((0.0, 500.0), (2000.0, 400.0)),
        weak_oracle_class="jump",
        coverage_start_ms=0.0,
        coverage_end_ms=6000.0,
    )

    assert classify_predicted_curve(predicted) == "jump"
    assert metrics.weak_oracle_boundary_matched_count == 1
    assert metrics.weak_oracle_boundary_precision == pytest.approx(1.0)
    assert metrics.weak_oracle_boundary_recall == pytest.approx(1.0)
    assert metrics.weak_oracle_jump_left_bpm_direct_mae == pytest.approx(0.0)
    assert metrics.weak_oracle_jump_right_bpm_direct_mae == pytest.approx(0.0)
    assert metrics.weak_oracle_boundary_abs_error_max_ms == pytest.approx(0.0)
    assert metrics.weak_oracle_jump_direction_match_rate == pytest.approx(1.0)
    assert metrics.weak_oracle_jump_left_bpm_alias_mae == pytest.approx(0.0)
    assert metrics.weak_oracle_jump_right_bpm_alias_mae == pytest.approx(0.0)
    assert metrics.weak_oracle_jump_exact_hit is True
    assert metrics.maximum_phase_continuity_error_ms == pytest.approx(0.0)
    assert metrics.maximum_serialization_seam_error_ms == pytest.approx(0.0)


def test_wrong_jump_direction_fails_exact_even_when_boundary_matches() -> None:
    predicted = _v3_grid((0, 5, 150.0), (5, 13, 120.0))
    metrics = evaluate_curve_against_weak_oracle(
        predicted,
        _weak_grid((0.0, 500.0), (2000.0, 400.0)),
        weak_oracle_class="jump",
        coverage_start_ms=0.0,
        coverage_end_ms=5000.0,
    )

    assert metrics.weak_oracle_boundary_matched_count == 1
    assert metrics.weak_oracle_jump_direction_match_rate == pytest.approx(0.0)
    assert metrics.weak_oracle_jump_exact_hit is False


def test_analytic_ramp_is_diagnostic_and_never_reports_accuracy() -> None:
    predicted = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(LinearTimeRampSection(0, 12, 120.0, 150.0),),
    )
    midpoint_ms = predicted.end_time_ms / 2.0
    reference = _weak_grid(
        (0.0, 500.0),
        (midpoint_ms, 60_000.0 / 135.0),
    )
    metrics = evaluate_curve_against_weak_oracle(
        predicted,
        reference,
        weak_oracle_class="ramp_like",
        coverage_start_ms=0.0,
        coverage_end_ms=predicted.end_time_ms,
    )

    assert metrics.predicted_class == "ramp"
    assert metrics.weak_oracle_class_exact
    assert metrics.weak_oracle_ramp_direction_match is True
    assert metrics.weak_oracle_ramp_accuracy is None
    assert metrics.weak_oracle_constant_exact_hit is None
    assert metrics.weak_oracle_jump_exact_hit is None


def test_analytic_ramp_to_constant_seam_is_a_jump_boundary() -> None:
    ramp = LinearTimeRampSection(0, 8, 120.0, 150.0)
    predicted = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(ramp, ConstantTempoSection(8, 16, 180.0)),
    )
    seam_ms = ramp.duration_ms
    metrics = evaluate_curve_against_weak_oracle(
        predicted,
        _weak_grid((0.0, 500.0), (seam_ms, 60_000.0 / 180.0)),
        weak_oracle_class="ramp_like",
        coverage_start_ms=0.0,
        coverage_end_ms=predicted.end_time_ms,
    )

    assert metrics.weak_oracle_predicted_boundary_count == 1
    assert metrics.weak_oracle_boundary_matched_count == 1


def test_constant_sections_with_subthreshold_bpm_noise_classify_constant() -> None:
    predicted = _v3_grid((0, 4, 120.0), (4, 8, 120.2))
    assert classify_predicted_curve(predicted) == "constant"
