import pytest

from ensomi_model.research.joint_audio_continuation.evaluation import native_diagnostics
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow


def test_empty_generation_remains_an_observed_bos_failure():
    value = native_diagnostics([], coverage_ms=120000, duration_ms=120000)
    assert value['heads'] == 0 and not value['reached30']
    assert value['first_head_ms'] is None and value['first30_head_row_ms'] is None
    assert value['stage_activity']['bos']['duration_ms'] == 120000
    assert value['longest_observed_head_gap_ms'] == 120000
    assert value['tap_tap_rate_per1000_transitions']['10'] is None


def test_early_held_time_and_release_to_head_are_not_tap_repeat_labels():
    rows = [CompleteRow(100, (2, 0, 0, 0)), CompleteRow(500, (3, 0, 0, 0)),
            CompleteRow(505, (1, 0, 0, 0)), CompleteRow(514, (1, 0, 0, 0)),
            CompleteRow(600, (1, 0, 0, 0))]
    value = native_diagnostics(rows, coverage_ms=1000, duration_ms=1000)
    assert value['stage_activity']['bos']['duration_ms'] == 100
    assert value['stage_activity']['early_held']['duration_ms'] == 400
    assert value['stage_activity']['early_free']['duration_ms'] == 500
    assert value['observed_occupied_lane_ms'] == 400
    assert value['occupied_lane_fraction_of_coverage'] == .1
    assert value['release_to_next_head_count'] == 1
    assert value['minimum_release_to_next_head_ms'] == 5
    assert value['eligible_tap_tap_transitions'] == 2
    assert value['tap_tap_counts_le_ms']['10'] == 1
    assert value['tap_tap_rate_per1000_transitions']['10'] == 500
    assert value['closed_ln_duration_ms']['median'] == 400


def test_first30_keeps_crossing_chord_and_partial_hold_has_no_invented_end():
    rows = [CompleteRow(100 * i, (1, 1, 1, 1)) for i in range(1, 8)]
    rows += [CompleteRow(800, (2, 2, 2, 0))]
    value = native_diagnostics(rows, coverage_ms=1000, duration_ms=3000)
    assert value['heads'] == 31 and value['first30_head_row_ms'] == 800
    assert value['stage_activity']['early_free']['heads'] == 27
    assert value['stage_activity']['mature_held']['duration_ms'] == 200
    assert value['observed_occupied_lane_ms'] == 600
    assert value['closed_ln_count'] == 0 and value['closed_ln_duration_ms'] is None
    assert value['open_lanes'] == [True, True, True, False]
    assert sum(v['duration_ms'] for v in value['stage_activity'].values()) == 1000
