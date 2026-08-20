from __future__ import annotations

import json

import pytest

from pulsefield_model.timing.canonicalization import (
    TIMING_CANONICALIZATION_BPM_80_160,
    TIMING_CANONICALIZATION_NONE,
)
from pulsefield_model.timing.evaluation.exp004_metrics import (
    EXP004_METRICS_SCHEMA,
    GUARD_KILL,
    GUARD_PASS,
    PHASE_SAMPLING_V1,
    ActiveSectionSignatureSpan,
    Exp004DenominatorRow,
    active_section_signature_v1,
    aggregate_weak_boundary_audio_metrics,
    alias_aware_bpm_error,
    canonical_bpm_80_160_for_exp004,
    canonical_bpm_binding_for_exp004,
    classify_exp004_denominators,
    classify_exp004_primary_guards,
    compare_phase_sampling_v1,
    extract_tempo_change_boundaries,
    extract_weak_redline_boundaries,
    match_weak_boundaries,
    phase_sample_times_v1,
    weak_boundary_difficulty_metrics,
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


def _v3_grid() -> TimingV3Grid:
    return TimingV3Grid(
        origin_beat=0,
        origin_time_ms=0.0,
        coverage_start_ms=0.0,
        coverage_end_ms=4000.0,
        sections=(
            ConstantTimingSection(start_beat=0, end_beat=4, bpm=120.0),
            ConstantTimingSection(start_beat=4, end_beat=9, bpm=150.0),
        ),
    )


def test_weak_redline_boundary_extraction_uses_card_threshold_and_half_open_coverage() -> None:
    redlines = _grid(
        (0.0, 500.0),
        (1000.0, 500.0 / 1.004),
        (2000.0, 400.0),
        (4000.0, 500.0),
    )

    extraction = extract_weak_redline_boundaries(
        redlines,
        coverage_start_ms=1000.0,
        coverage_end_ms=4000.0,
    )

    assert extraction.valid_comparator
    assert [boundary.time_ms for boundary in extraction.boundaries] == [2000.0]
    assert extraction.boundaries[0].left_bpm == pytest.approx(60000.0 / (500.0 / 1.004))
    assert extraction.boundaries[0].right_bpm == pytest.approx(150.0)


def test_weak_redline_extraction_rejects_invalid_comparator_without_throwing() -> None:
    extraction = extract_weak_redline_boundaries(
        ({"offset_ms": 0.0, "beat_length_ms": 500.0}, {"offset_ms": 10.0, "beat_length_ms": 0.0}),
        coverage_start_ms=0.0,
        coverage_end_ms=1000.0,
    )

    assert not extraction.valid_comparator
    assert extraction.rejection_reason == "nonpositive_beat_length"
    assert extraction.boundaries == ()


@pytest.mark.parametrize(
    "redlines",
    [
        ({"offset_ms": 100.0, "beat_length_ms": 500.0}, {"offset_ms": 0.0, "beat_length_ms": 400.0}),
        ({"offset_ms": 0.0, "beat_length_ms": 500.0}, {"offset_ms": 0.0, "beat_length_ms": 400.0}),
    ],
)
def test_generic_sequence_extraction_preserves_order_and_rejects_bad_offsets(redlines: object) -> None:
    extraction = extract_weak_redline_boundaries(
        redlines,  # type: ignore[arg-type]
        coverage_start_ms=0.0,
        coverage_end_ms=1000.0,
    )

    assert not extraction.valid_comparator
    assert extraction.rejection_reason == "nonincreasing_offsets"


def test_weak_boundary_matching_is_one_to_one_with_period_bounded_tolerance() -> None:
    predicted = extract_tempo_change_boundaries(
        _grid((0.0, 500.0), (1990.0, 400.0), (2600.0, 500.0)),
        coverage_start_ms=0.0,
        coverage_end_ms=4000.0,
    )
    weak = extract_tempo_change_boundaries(
        _grid((0.0, 500.0), (2000.0, 400.0), (2180.0, 500.0)),
        coverage_start_ms=0.0,
        coverage_end_ms=4000.0,
    )

    summary = match_weak_boundaries(predicted, weak)

    assert summary.matched_count == 1
    assert summary.matches[0].signed_error_ms == pytest.approx(-10.0)
    assert summary.matches[0].abs_error_ms == pytest.approx(10.0)
    assert summary.matches[0].tolerance_ms == pytest.approx(200.0)
    assert summary.unmatched_predicted_boundary_count == 1
    assert summary.unmatched_weak_redline_boundary_count == 1


def test_audio_first_aggregation_uses_valid_difficulty_medians_sums_and_consensus() -> None:
    predicted = extract_tempo_change_boundaries(
        _grid((0.0, 500.0), (2000.0, 400.0), (3200.0, 500.0)),
        coverage_start_ms=0.0,
        coverage_end_ms=5000.0,
    )
    rows = [
        weak_boundary_difficulty_metrics(
            audio_key="audio-a",
            difficulty_key="easy",
            predicted_boundaries=predicted,
            weak_redline_extraction=extract_weak_redline_boundaries(
                _grid((0.0, 500.0), (2005.0, 400.0), (3205.0, 500.0)),
                coverage_start_ms=0.0,
                coverage_end_ms=5000.0,
            ),
        ),
        weak_boundary_difficulty_metrics(
            audio_key="audio-a",
            difficulty_key="hard",
            predicted_boundaries=predicted,
            weak_redline_extraction=extract_weak_redline_boundaries(
                _grid((0.0, 500.0), (1990.0, 400.0)),
                coverage_start_ms=0.0,
                coverage_end_ms=5000.0,
            ),
        ),
        weak_boundary_difficulty_metrics(
            audio_key="audio-a",
            difficulty_key="broken",
            predicted_boundaries=predicted,
            weak_redline_extraction=extract_weak_redline_boundaries(
                (),
                coverage_start_ms=0.0,
                coverage_end_ms=5000.0,
            ),
        ),
    ]

    (audio_metrics,) = aggregate_weak_boundary_audio_metrics(rows)

    assert audio_metrics.valid_difficulty_count == 2
    assert audio_metrics.invalid_difficulty_count == 1
    assert audio_metrics.weak_boundary_matched_count == 3
    assert audio_metrics.weak_boundary_redline_count == 3
    assert audio_metrics.weak_boundary_predicted_count_per_difficulty_mean == pytest.approx(2.0)
    assert audio_metrics.weak_boundary_mean_abs_error_median_ms == pytest.approx(7.5)
    consensus = {
        item.predicted_boundary.time_ms: item.weak_consensus_supported
        for item in audio_metrics.weak_consensus_boundaries
    }
    assert consensus == {2000.0: True, 3200.0: True}


@pytest.mark.parametrize(
    ("predicted_bpm", "reference_bpm"),
    [
        (80.0, 160.0),
        (120.0, 120.0),
        (160.0, 80.0),
        (240.0, 120.0),
        (270.0, 90.0),
    ],
)
def test_alias_aware_bpm_error_accepts_frozen_alias_set(predicted_bpm: float, reference_bpm: float) -> None:
    assert alias_aware_bpm_error(predicted_bpm, reference_bpm) == pytest.approx(0.0)


def test_canonical_bpm_binding_is_the_source_owned_80_160_policy() -> None:
    assert [
        canonical_bpm_80_160_for_exp004(bpm)
        for bpm in (80.0, 120.0, 160.0, 240.0, 270.0)
    ] == [80.0, 120.0, 80.0, 120.0, 135.0]

    binding = canonical_bpm_binding_for_exp004()

    assert binding.canonicalization == TIMING_CANONICALIZATION_BPM_80_160
    assert binding.function_module == "pulsefield_model.timing.canonicalization"
    assert binding.function_qualname == "canonical_bpm_80_160"
    assert len(binding.source_sha256) == 64


def test_phase_sampling_v1_uses_absolute_20ms_lattice_and_excludes_end() -> None:
    assert phase_sample_times_v1(coverage_start_ms=10.0, coverage_end_ms=70.0) == (
        20.0,
        40.0,
        60.0,
    )
    assert phase_sample_times_v1(coverage_start_ms=0.0, coverage_end_ms=40.0) == (
        0.0,
        20.0,
    )


def test_phase_sampling_v1_exact_boundary_belongs_to_right_section_and_end_is_exclusive() -> None:
    predicted = _v3_grid()
    reference = _grid((0.0, 500.0), (2000.0, 400.0))

    comparison = compare_phase_sampling_v1(
        predicted,
        reference,
        coverage_start_ms=2000.0,
        coverage_end_ms=2020.0,
        canonicalization=TIMING_CANONICALIZATION_NONE,
    )

    assert comparison.schema == EXP004_METRICS_SCHEMA
    assert comparison.sampling_version == PHASE_SAMPLING_V1
    assert comparison.sample_count == 1
    assert comparison.local_bpm_mae == pytest.approx(0.0)
    assert comparison.phase_max_ms == pytest.approx(0.0)
    assert comparison.active_section_disagreement_sample_count == 0
    assert json.loads(json.dumps(comparison.to_dict(), allow_nan=False)) == comparison.to_dict()


def test_active_section_disagreement_uses_exact_clipped_signature_not_raw_indices() -> None:
    predicted = _grid((-1000.0, 500.0), (1000.0, 400.0))
    reference = _grid((1000.0, 400.0))

    comparison = compare_phase_sampling_v1(
        predicted,
        reference,
        coverage_start_ms=1000.0,
        coverage_end_ms=1080.0,
        canonicalization=TIMING_CANONICALIZATION_NONE,
    )

    expected_signature = (
        ActiveSectionSignatureSpan(
            start_time_ms=1000.0,
            end_time_ms=1080.0,
            beat_length_ms=400.0,
            bpm=150.0,
        ),
    )
    assert active_section_signature_v1(
        predicted,
        coverage_start_ms=1000.0,
        coverage_end_ms=1080.0,
    ) == expected_signature
    assert comparison.predicted_active_section_signature == expected_signature
    assert comparison.reference_active_section_signature == expected_signature
    assert comparison.active_section_signature_equal
    assert comparison.active_section_disagreement_sample_count == 0


def test_endpoint_drift_uses_coverage_end_left_limit_when_end_is_section_boundary() -> None:
    predicted = _grid((0.0, 501.0))
    reference = _grid((0.0, 500.0), (2000.0, 250.0))

    comparison = compare_phase_sampling_v1(
        predicted,
        reference,
        coverage_start_ms=0.0,
        coverage_end_ms=2000.0,
        canonicalization=TIMING_CANONICALIZATION_NONE,
    )

    expected_endpoint_beats = 2000.0 / 501.0 - 4.0
    assert comparison.sample_count == 100
    assert comparison.endpoint_relative_drift_beats == pytest.approx(expected_endpoint_beats)
    assert comparison.endpoint_relative_drift_ms == pytest.approx(expected_endpoint_beats * 500.0)
    assert comparison.max_abs_prefix_relative_drift_ms == pytest.approx(
        abs(comparison.endpoint_relative_drift_ms)
    )


def test_multi_section_v2_cumulative_beat_drift_does_not_reset_at_later_segments() -> None:
    predicted = _grid((0.0, 500.0))
    reference = _grid((0.0, 500.0), (2100.0, 400.0))

    comparison = compare_phase_sampling_v1(
        predicted,
        reference,
        coverage_start_ms=0.0,
        coverage_end_ms=2500.0,
        canonicalization=TIMING_CANONICALIZATION_NONE,
    )

    expected_reference_endpoint_beats = 2100.0 / 500.0 + (2500.0 - 2100.0) / 400.0
    expected_endpoint_beats = 2500.0 / 500.0 - expected_reference_endpoint_beats
    assert comparison.endpoint_relative_drift_beats == pytest.approx(expected_endpoint_beats)
    assert comparison.endpoint_relative_drift_ms == pytest.approx(expected_endpoint_beats * 400.0)
    assert comparison.max_abs_prefix_relative_drift_ms >= abs(comparison.endpoint_relative_drift_ms)


def test_endpoint_drift_contributes_to_prefix_but_not_sample_partition_slope() -> None:
    predicted = _grid((0.0, 500.0), (21.0, 400.0))
    reference = _grid((0.0, 500.0))

    comparison = compare_phase_sampling_v1(
        predicted,
        reference,
        coverage_start_ms=0.0,
        coverage_end_ms=25.0,
        canonicalization=TIMING_CANONICALIZATION_NONE,
    )

    assert comparison.sample_count == 2
    assert comparison.endpoint_relative_drift_ms != pytest.approx(0.0)
    assert comparison.max_abs_prefix_relative_drift_ms == pytest.approx(
        abs(comparison.endpoint_relative_drift_ms)
    )
    assert comparison.drift_slope_beats_per_minute == pytest.approx(0.0)
    assert comparison.drift_slope_ms_per_minute == pytest.approx(0.0)


def test_phase_sampling_v1_can_use_exact_canonical_80_160_binding() -> None:
    predicted = _grid((0.0, 250.0))
    reference = _grid((0.0, 500.0))

    raw = compare_phase_sampling_v1(
        predicted,
        reference,
        coverage_start_ms=0.0,
        coverage_end_ms=1000.0,
        canonicalization=TIMING_CANONICALIZATION_NONE,
    )
    canonical = compare_phase_sampling_v1(
        predicted,
        reference,
        coverage_start_ms=0.0,
        coverage_end_ms=1000.0,
        canonicalization=TIMING_CANONICALIZATION_BPM_80_160,
    )

    assert raw.local_bpm_mae == pytest.approx(120.0)
    assert raw.local_bpm_alias_mae == pytest.approx(0.0)
    assert raw.endpoint_relative_drift_ms != pytest.approx(0.0)
    assert canonical.local_bpm_mae == pytest.approx(0.0)
    assert canonical.endpoint_relative_drift_ms == pytest.approx(0.0)


def test_exp004_denominators_keep_pure_cj3_separate_from_selected_fallback_safety() -> None:
    summary = classify_exp004_denominators(
        (
            Exp004DenominatorRow(
                audio_key="pure",
                cache_valid=True,
                projection_evaluable=True,
                comparison_eligible=True,
                pure_cj3_grid_produced=True,
                pure_cj3_phase_matched=True,
                current_v2_phase_matched=True,
                selected_safety_phase_scored=True,
                selected_used_fallback=False,
            ),
            Exp004DenominatorRow(
                audio_key="fallback",
                cache_valid=True,
                projection_evaluable=True,
                comparison_eligible=True,
                pure_cj3_grid_produced=False,
                pure_cj3_phase_matched=False,
                current_v2_phase_matched=True,
                selected_safety_phase_scored=True,
                selected_used_fallback=True,
            ),
        )
    )

    assert summary.stage_audio_count == 2
    assert summary.projection_evaluable_count == 2
    assert summary.comparison_eligible_count == 2
    assert summary.pure_CJ3_phase_count == 1
    assert summary.pure_CJ3_phase_coverage == pytest.approx(0.5)
    assert summary.selected_safety_phase_count == 2
    assert summary.selected_fallback_rate == pytest.approx(0.5)

    guards = classify_exp004_primary_guards(
        denominators=summary,
        mean_phase_ratio=1.0,
        p90_phase_ratio=1.0,
    )
    assert guards.mean_phase_ratio == GUARD_PASS
    assert guards.p90_phase_ratio == GUARD_PASS
    assert guards.pure_CJ3_phase_coverage == GUARD_KILL
    assert guards.selected_fallback_rate == GUARD_KILL


def test_comparator_unavailable_selected_fallback_stays_in_fallback_denominator() -> None:
    summary = classify_exp004_denominators(
        (
            Exp004DenominatorRow(
                audio_key="fallback-before-comparator",
                cache_valid=True,
                projection_evaluable=True,
                comparison_eligible=False,
                pure_cj3_grid_produced=False,
                pure_cj3_phase_matched=False,
                current_v2_phase_matched=False,
                selected_safety_phase_scored=False,
                selected_used_fallback=True,
            ),
        )
    )

    assert summary.projection_evaluable_count == 1
    assert summary.comparison_eligible_count == 0
    assert summary.pure_CJ3_phase_count == 0
    assert summary.pure_CJ3_phase_coverage is None
    assert summary.selected_safety_phase_count == 0
    assert summary.selected_fallback_count == 1
    assert summary.selected_fallback_rate == pytest.approx(1.0)


def _denominator_row_kwargs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "audio_key": "row",
        "cache_valid": True,
        "projection_evaluable": False,
        "comparison_eligible": False,
        "pure_cj3_grid_produced": False,
        "pure_cj3_phase_matched": False,
        "current_v2_phase_matched": False,
        "selected_safety_phase_scored": False,
        "selected_used_fallback": False,
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"cache_valid": False, "projection_evaluable": True},
            "projection_evaluable requires cache_valid",
        ),
        (
            {"projection_evaluable": False, "comparison_eligible": True},
            "comparison_eligible requires projection_evaluable",
        ),
        (
            {"projection_evaluable": False, "pure_cj3_grid_produced": True},
            "pure_cj3_grid_produced requires projection_evaluable",
        ),
        (
            {
                "projection_evaluable": True,
                "comparison_eligible": False,
                "current_v2_phase_matched": True,
            },
            "current_v2_phase_matched requires comparison_eligible",
        ),
        (
            {
                "projection_evaluable": True,
                "pure_cj3_grid_produced": True,
                "pure_cj3_phase_matched": True,
            },
            "pure_cj3_phase_matched requires comparison_eligible",
        ),
        (
            {
                "projection_evaluable": True,
                "comparison_eligible": True,
                "pure_cj3_grid_produced": True,
                "pure_cj3_phase_matched": True,
                "current_v2_phase_matched": False,
            },
            "pure_cj3_phase_matched requires current_v2_phase_matched",
        ),
        (
            {
                "projection_evaluable": True,
                "selected_safety_phase_scored": True,
            },
            "selected_safety_phase_scored requires comparison_eligible",
        ),
        (
            {
                "projection_evaluable": True,
                "comparison_eligible": True,
                "current_v2_phase_matched": True,
                "selected_used_fallback": True,
            },
            "comparison-eligible selected_used_fallback requires selected_safety_phase_scored",
        ),
        (
            {
                "projection_evaluable": True,
                "comparison_eligible": True,
                "pure_cj3_grid_produced": True,
                "current_v2_phase_matched": True,
                "selected_safety_phase_scored": True,
                "selected_used_fallback": True,
            },
            "selected_used_fallback excludes pure_cj3_grid_produced",
        ),
        (
            {
                "projection_evaluable": True,
                "comparison_eligible": True,
                "pure_cj3_grid_produced": True,
                "current_v2_phase_matched": True,
                "selected_safety_phase_scored": True,
            },
            "selected_safety_phase_scored without fallback requires pure_cj3_phase_matched",
        ),
    ],
)
def test_exp004_denominator_rows_reject_impossible_states(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        Exp004DenominatorRow(**_denominator_row_kwargs(**overrides))  # type: ignore[arg-type]
