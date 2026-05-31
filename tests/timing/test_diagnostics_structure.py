from dataclasses import asdict

from pulsefield_model.timing.diagnostics import compare_timing_grid_structure, compare_timing_grids
from pulsefield_model.timing.schema import FittedTimingGrid, TimingSegment


def _grid(*segments: tuple[float, float]) -> FittedTimingGrid:
    return FittedTimingGrid(
        tuple(
            TimingSegment(offset_ms=offset_ms, beat_length_ms=60000.0 / bpm)
            for offset_ms, bpm in segments
        )
    )


def test_structural_comparison_collapses_redundant_oracle_same_bpm_segments() -> None:
    oracle = _grid((0.0, 120.0), (1000.0, 120.0), (2000.0, 120.0))
    predicted = _grid((0.0, 120.0))

    comparison = compare_timing_grid_structure(predicted, oracle)

    assert comparison.canonical_oracle_segment_count == 1
    assert comparison.predicted_segment_count == 1
    assert comparison.abs_canonical_segment_count_delta == 0
    assert comparison.redundant_oracle_segment_count == 2


def test_structural_comparison_preserves_distinct_canonical_tempo_regions() -> None:
    oracle = _grid(
        (0.0, 120.0),
        (1000.0, 120.0),
        (2000.0, 150.0),
        (3000.0, 120.0),
    )
    predicted = _grid((0.0, 120.0), (2000.0, 150.0))

    comparison = compare_timing_grid_structure(predicted, oracle)

    assert comparison.canonical_oracle_segment_count == 3
    assert comparison.predicted_segment_count == 2
    assert comparison.abs_canonical_segment_count_delta == 1
    assert comparison.redundant_oracle_segment_count == 1


def test_structural_comparison_reports_predicted_uniques_and_tempo_family_switches() -> None:
    oracle = _grid((0.0, 120.0))
    predicted = _grid(
        (0.0, 60.0),
        (1000.0, 120.0),
        (2000.0, 150.0),
        (3000.0, 75.0),
    )

    comparison = compare_timing_grid_structure(predicted, oracle, predicted_alias_switch_count=1)

    assert comparison.predicted_unique_bpm_count == 4
    assert comparison.predicted_tempo_family_switch_count == 1
    assert comparison.predicted_alias_switch_count == 1


def test_dense_comparison_exposes_structural_metrics() -> None:
    oracle = _grid((0.0, 120.0), (1000.0, 120.0))
    predicted = _grid((0.0, 120.0))

    comparison = compare_timing_grids(predicted, oracle, frame_count=50)
    payload = asdict(comparison)

    assert payload["canonical_oracle_segment_count"] == 1
    assert payload["predicted_segment_count"] == 1
    assert payload["abs_canonical_segment_count_delta"] == 0
    assert payload["redundant_oracle_segment_count"] == 1
