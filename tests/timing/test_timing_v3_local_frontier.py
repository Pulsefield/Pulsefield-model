from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass, replace
from typing import Any, Iterable, Sequence

import numpy as np
import pytest

from pulsefield_model.timing.schema import FrameTimingPrediction
from pulsefield_model.timing.v3 import global_constant_jump as exp004
from pulsefield_model.timing.v3 import local_frontier as lf
from pulsefield_model.timing.v3.global_constant_jump import (
    BOUNDARY_CANDIDATE_SCORE_VERSION,
    BoundaryCandidate,
    CANDIDATE_CONTRACT_VERSION,
    GLOBAL_CONSTANT_JUMP_CONSTANTS,
    GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON_SHA256,
    GlobalConstantJumpCandidateDiagnostics,
    GlobalConstantJumpCandidateSet,
    OriginCandidate,
    PULSE_CORRELATION_VERSION,
    TempoCandidate,
    materialize_global_constant_jump_peaks,
)
from pulsefield_model.timing.v3.local_frontier import (
    LocalFrontierConfig,
    LocalFrontierResult,
    LocalFrontierScheduleArm,
    LocalFrontierState,
    block_schedule_for_cut,
    build_tempo_shortlist,
    closure_count_candidates,
    export_frontier,
    fit_local_frontier_constant_jump,
    lattice_time_ms,
    next_beat_on_or_after,
    previous_beat_on_or_before,
    restrict_timing_prediction,
    validate_restricted_prediction,
)
from pulsefield_model.timing.v3.schema import TimingV3Grid


def test_closure_count_candidates_use_half_up_positive_dedup_order() -> None:
    open_section = _frontier_state(
        current_section_start_beat=0,
        current_section_start_time_ms=0.0,
        current_bpm=120.0,
    )

    half_up = closure_count_candidates(open_section, boundary_time_ms=10_250.0)
    assert _closure_counts(half_up) == (21, 20, 22)
    assert _closure_boundary_beats(half_up) == (21, 20, 22)
    assert _closure_boundary_times_ms(half_up) == pytest.approx((10_500.0, 10_000.0, 11_000.0))

    near_origin = closure_count_candidates(open_section, boundary_time_ms=250.0)
    assert _closure_counts(near_origin) == (1, 2)
    assert all(count > 0 for count in _closure_counts(near_origin))


def test_cut_lattice_nextafter_handles_negative_and_large_beat_indices() -> None:
    origin_time_ms = 100.0
    bpm = 120.0

    negative_exact = lattice_time_ms(-3, origin_time_ms=origin_time_ms, bpm=bpm)
    assert negative_exact == pytest.approx(-1400.0)
    assert next_beat_on_or_after(negative_exact, origin_time_ms=origin_time_ms, bpm=bpm) == -3
    assert previous_beat_on_or_before(negative_exact, origin_time_ms=origin_time_ms, bpm=bpm) == -3
    assert next_beat_on_or_after(
        np.nextafter(negative_exact, math.inf),
        origin_time_ms=origin_time_ms,
        bpm=bpm,
    ) == -2
    assert previous_beat_on_or_before(
        np.nextafter(negative_exact, -math.inf),
        origin_time_ms=origin_time_ms,
        bpm=bpm,
    ) == -4

    large_beat = 2**40
    large_exact = lattice_time_ms(large_beat, origin_time_ms=origin_time_ms, bpm=bpm)
    assert next_beat_on_or_after(large_exact, origin_time_ms=origin_time_ms, bpm=bpm) == large_beat
    assert previous_beat_on_or_before(large_exact, origin_time_ms=origin_time_ms, bpm=bpm) == large_beat
    assert next_beat_on_or_after(
        np.nextafter(large_exact, math.inf),
        origin_time_ms=origin_time_ms,
        bpm=bpm,
    ) == large_beat + 1
    assert previous_beat_on_or_before(
        np.nextafter(large_exact, -math.inf),
        origin_time_ms=origin_time_ms,
        bpm=bpm,
    ) == large_beat - 1


@pytest.mark.parametrize("beat", (0, -3, 2**40))
def test_cut_lattice_next_previous_oracle_at_exact_and_one_ulp_edges(beat: int) -> None:
    origin_time_ms = 100.0
    bpm = 120.0

    exact = lattice_time_ms(beat, origin_time_ms=origin_time_ms, bpm=bpm)
    just_after = np.nextafter(exact, math.inf)
    just_before = np.nextafter(exact, -math.inf)

    assert next_beat_on_or_after(exact, origin_time_ms=origin_time_ms, bpm=bpm) == beat
    assert previous_beat_on_or_before(exact, origin_time_ms=origin_time_ms, bpm=bpm) == beat
    assert next_beat_on_or_after(just_after, origin_time_ms=origin_time_ms, bpm=bpm) == beat + 1
    assert previous_beat_on_or_before(just_after, origin_time_ms=origin_time_ms, bpm=bpm) == beat
    assert next_beat_on_or_after(just_before, origin_time_ms=origin_time_ms, bpm=bpm) == beat
    assert previous_beat_on_or_before(just_before, origin_time_ms=origin_time_ms, bpm=bpm) == beat - 1


@pytest.mark.parametrize(
    ("bpm", "expected_alias_family"),
    (
        (37.5, 112.5),
        (60.0, 120.0),
        (75.0, 150.0),
        (120.0, 120.0),
        (160.0, 80.0),
        (250.0, 83.333333),
        (500.0, 125.0),
        (1000.0, 250.0),
    ),
)
def test_alias_family_representative_expected_v1_mappings_are_frozen(
    bpm: float,
    expected_alias_family: float,
) -> None:
    assert exp004._alias_family_representative_v1(  # noqa: SLF001
        bpm,
        GLOBAL_CONSTANT_JUMP_CONSTANTS,
    ) == expected_alias_family


def test_bootstrap_enumerates_downbeat_phases_only_when_downbeat_signal_exists() -> None:
    config = LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S30)
    with_downbeat = fit_local_frontier_constant_jump(
        _constant_prediction(duration_ms=20_000.0, bpm=120.0, downbeat_phase=0),
        config=config,
    )
    without_downbeat = fit_local_frontier_constant_jump(
        _constant_prediction(duration_ms=20_000.0, bpm=120.0, downbeat_phase=None),
        config=config,
    )

    _assert_ok(with_downbeat)
    _assert_ok(without_downbeat)
    assert with_downbeat.diagnostics.bootstrap_downbeat_phases == (0, 1, 2, 3)
    assert without_downbeat.diagnostics.bootstrap_downbeat_phases == (None,)
    assert with_downbeat.diagnostics.bootstrap_state_count == (
        with_downbeat.diagnostics.origin_candidate_count * 4
    )
    assert without_downbeat.diagnostics.bootstrap_state_count == (
        without_downbeat.diagnostics.origin_candidate_count
    )


def test_candidate_set_bootstrap_uses_16_origin_tuple_order_times_4_phases() -> None:
    prediction = _constant_prediction(duration_ms=45_000.0, bpm=120.0, downbeat_phase=0)
    origins = tuple(
        OriginCandidate(
            anchor_id=index,
            time_ms=float(index * 20.0),
            bpm=120.0,
            score=1.0 - index * 0.001,
        )
        for index in range(16)
    )
    candidate_set = _candidate_set_for_prediction(
        prediction,
        tempo_candidates=(TempoCandidate(bpm=120.0, source="test", score=1.0),),
        origin_candidates=origins,
        boundary_candidates=(),
    )

    result = fit_local_frontier_constant_jump(
        prediction,
        config=LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S30),
        candidate_set=candidate_set,
    )

    _assert_ok(result)
    assert result.diagnostics.origin_candidate_count == 16
    assert result.diagnostics.bootstrap_downbeat_phases == (0, 1, 2, 3)
    assert result.diagnostics.bootstrap_state_count == 64
    assert result.diagnostics.bootstrap_replay_keys == tuple(
        (
            origin_index,
            float.hex(origin.time_ms),
            float.hex(origin.bpm),
            phase,
        )
        for origin_index, origin in enumerate(origins)
        for phase in (0, 1, 2, 3)
    )


def test_positive_origin_backward_extension_separates_logical_count_from_serialized_span() -> None:
    prediction = _offset_constant_prediction(
        duration_ms=20_000.0,
        bpm=120.0,
        origin_time_ms=250.0,
        downbeat_phase=None,
    )
    candidate_set = _candidate_set_for_prediction(
        prediction,
        tempo_candidates=(TempoCandidate(bpm=120.0, source="test", score=1.0),),
        origin_candidates=(OriginCandidate(anchor_id=0, time_ms=250.0, bpm=120.0, score=1.0),),
        boundary_candidates=(),
    )

    result = fit_local_frontier_constant_jump(
        prediction,
        config=LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S60),
        candidate_set=candidate_set,
    )

    grid = _assert_ok(result)
    assert grid.origin_beat == 0
    assert grid.origin_time_ms == pytest.approx(250.0)
    assert grid.start_beat == -1
    assert grid.start_time_ms == pytest.approx(-250.0)
    assert result.diagnostics.selected_origin_time_ms == pytest.approx(250.0)
    assert result.diagnostics.selected_serialized_first_start_beat == -1
    assert result.diagnostics.first_section_logical_beat_count == 40
    assert result.diagnostics.first_section_serialized_beat_count == 41
    assert result.diagnostics.first_section_logical_beat_count == (
        grid.sections[0].end_beat - grid.origin_beat
    )
    assert result.diagnostics.first_section_serialized_beat_count == (
        grid.sections[0].end_beat - grid.sections[0].start_beat
    )


def test_tempo_shortlist_reserves_source_quotas_and_keeps_non_alias_jump_candidate() -> None:
    frontier = tuple(
        _frontier_state(
            current_bpm=90.0 + index,
            alias_family=90.0 + index,
            deterministic_replay_key=("incoming", index),
            rank_objective=float(index),
        )
        for index in range(17)
    )
    non_alias_right_bpm = 60000.0 / 437.0
    boundary = _boundary_candidate(
        anchor_id=10,
        time_ms=16_000.0,
        left_period_ms=500.0,
        right_period_ms=437.0,
    )
    global_tempos = tuple(
        TempoCandidate(bpm=bpm, source="global", score=float(index))
        for index, bpm in enumerate((121.25, 188.5, 240.0))
    )

    shortlist = build_tempo_shortlist(frontier, (boundary,), global_tempos)

    assert len(shortlist) <= 64
    assert shortlist[:16] == tuple(state.current_bpm for state in frontier[:16])
    assert _contains_bpm(shortlist, non_alias_right_bpm)
    assert _bpm_index(shortlist, non_alias_right_bpm) < _bpm_index(shortlist, global_tempos[0].bpm)
    assert not _contains_bpm(shortlist[:16], frontier[16].current_bpm)
    assert all(20.0 <= bpm <= 1000.0 and math.isfinite(bpm) for bpm in shortlist)


def test_frontier_export_applies_dominance_before_class_reservation_at_k16() -> None:
    base_states = tuple(
        _frontier_state(
            current_bpm=80.0 + index,
            alias_family=80.0 + index,
            global_downbeat_phase=index % 4,
            rank_objective=10.0 + index,
            committed_objective=20.0 + index,
            deterministic_replay_key=("class", index),
        )
        for index in range(16)
    )
    worse_duplicate = _frontier_state(
        current_bpm=85.0,
        alias_family=85.0,
        global_downbeat_phase=1,
        rank_objective=99.0,
        committed_objective=99.0,
        deterministic_replay_key=("duplicate", "worse"),
    )
    better_duplicate = _frontier_state(
        current_bpm=85.0,
        alias_family=85.0,
        global_downbeat_phase=1,
        rank_objective=0.0,
        committed_objective=0.0,
        deterministic_replay_key=("duplicate", "better"),
    )

    export = export_frontier(
        base_states[:5] + (worse_duplicate, better_duplicate) + base_states[6:],
        cut_time_ms=30_000.0,
        max_states=16,
    )

    assert len(export.states) == 16
    assert len({state.frontier_class_key for state in export.states}) == 16
    assert ("duplicate", "better") in {state.deterministic_replay_key for state in export.states}
    assert ("duplicate", "worse") not in {state.deterministic_replay_key for state in export.states}
    assert export.dominance_pruned_state_count == 1
    assert export.width_pruned_state_count == 0


def test_frontier_export_reserves_best_sixteen_classes_and_width_prunes_rest() -> None:
    states = tuple(
        _frontier_state(
            current_bpm=80.0 + index,
            alias_family=80.0 + index,
            global_downbeat_phase=index % 4,
            rank_objective=float(index),
            committed_objective=float(index),
            deterministic_replay_key=("class", index),
        )
        for index in range(18)
    )

    export = export_frontier(states, cut_time_ms=30_000.0, max_states=16)

    assert tuple(state.deterministic_replay_key for state in export.states) == tuple(
        ("class", index) for index in range(16)
    )
    assert len({state.frontier_class_key for state in export.states}) == 16
    assert export.dominance_pruned_state_count == 0
    assert export.width_pruned_state_count == 2


def test_frontier_export_deduplicates_future_equivalent_paths_by_complete_order() -> None:
    best = _frontier_state(
        current_bpm=128.0,
        alias_family=128.0,
        global_downbeat_phase=2,
        rank_objective=3.0,
        committed_objective=1.0,
        deterministic_replay_key=("same-future", "best"),
    )
    worse_same_future = _frontier_state(
        current_bpm=128.0,
        alias_family=128.0,
        global_downbeat_phase=2,
        rank_objective=3.0,
        committed_objective=2.0,
        deterministic_replay_key=("same-future", "worse"),
    )

    assert best.future_equivalence_key == worse_same_future.future_equivalence_key

    export = export_frontier((worse_same_future, best), cut_time_ms=30_000.0, max_states=16)

    assert tuple(state.deterministic_replay_key for state in export.states) == (
        ("same-future", "best"),
    )
    assert export.dominance_pruned_state_count == 1
    assert export.width_pruned_state_count == 0


def test_frontier_export_collapses_states_that_only_differ_by_serialized_first_start() -> None:
    left_extension = _frontier_state(serialized_first_start_beat=-2)
    origin_aligned = _frontier_state(serialized_first_start_beat=0)

    assert left_extension.future_equivalence_key == origin_aligned.future_equivalence_key

    export = export_frontier((left_extension, origin_aligned), cut_time_ms=30_000.0, max_states=16)

    assert export.states == (left_extension,)
    assert export.dominance_pruned_state_count == 1
    assert export.width_pruned_state_count == 0


def test_constant_track_across_at_least_four_cuts_has_no_artificial_sections() -> None:
    result = fit_local_frontier_constant_jump(
        _constant_prediction(duration_ms=180_000.0, bpm=120.0, downbeat_phase=0),
        config=LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S30),
    )

    grid = _assert_ok(result)
    assert len(grid.sections) == 1
    assert grid.sections[0].bpm == pytest.approx(120.0, abs=0.5)
    assert result.diagnostics.block_count >= 5
    assert max(result.diagnostics.frontier_widths) <= 16
    assert result.diagnostics.local_bucket_width_max <= 64
    _assert_result_json_finite(result)


def test_single_jump_is_supported_after_min_duration_and_rejected_before_it() -> None:
    config = LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S30)
    too_early_prediction = _jump_prediction(
        first_bpm=120.0,
        second_bpm=150.0,
        boundary_ms=7_500.0,
        duration_ms=22_000.0,
        include_downbeats=False,
    )

    too_early = fit_local_frontier_constant_jump(
        too_early_prediction,
        config=config,
        candidate_set=_candidate_set_for_prediction(
            too_early_prediction,
            tempo_candidates=(
                TempoCandidate(bpm=120.0, source="test", score=1.0),
                TempoCandidate(bpm=150.0, source="test", score=0.9),
            ),
            origin_candidates=(OriginCandidate(anchor_id=0, time_ms=0.0, bpm=120.0, score=1.0),),
            boundary_candidates=(
                _boundary_candidate(
                    anchor_id=0,
                    time_ms=7_500.0,
                    left_period_ms=500.0,
                    right_period_ms=400.0,
                ),
            ),
        ),
    )
    early_grid = _assert_ok(too_early)
    assert len(early_grid.sections) == 1

    supported_prediction = _jump_prediction(
        first_bpm=120.0,
        second_bpm=150.0,
        boundary_ms=12_000.0,
        duration_ms=72_000.0,
        include_downbeats=False,
    )
    supported = fit_local_frontier_constant_jump(
        supported_prediction,
        config=config,
        candidate_set=_candidate_set_for_prediction(
            supported_prediction,
            tempo_candidates=(
                TempoCandidate(bpm=120.0, source="test", score=1.0),
                TempoCandidate(bpm=150.0, source="test", score=0.9),
            ),
            origin_candidates=(OriginCandidate(anchor_id=0, time_ms=0.0, bpm=120.0, score=1.0),),
            boundary_candidates=(
                _boundary_candidate(
                    anchor_id=0,
                    time_ms=12_000.0,
                    left_period_ms=500.0,
                    right_period_ms=400.0,
                ),
            ),
        ),
    )
    jump_grid = _assert_ok(supported)
    assert len(jump_grid.sections) == 2
    assert jump_grid.section_end_time_ms(0) == pytest.approx(12_000.0)
    assert jump_grid.section_start_time_ms(1) == jump_grid.section_end_time_ms(0)
    assert jump_grid.sections[0].bpm == pytest.approx(120.0, abs=0.5)
    assert jump_grid.sections[1].bpm == pytest.approx(150.0, abs=0.5)


def test_boundary_exactly_at_cut_is_owned_by_next_core() -> None:
    prediction = _jump_prediction(
        first_bpm=120.0,
        second_bpm=150.0,
        boundary_ms=30_000.0,
        duration_ms=72_000.0,
        include_downbeats=False,
    )
    result = fit_local_frontier_constant_jump(
        prediction,
        config=LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S30),
        candidate_set=_candidate_set_for_prediction(
            prediction,
            tempo_candidates=(
                TempoCandidate(bpm=120.0, source="test", score=1.0),
                TempoCandidate(bpm=150.0, source="test", score=0.9),
            ),
            origin_candidates=(OriginCandidate(anchor_id=0, time_ms=0.0, bpm=120.0, score=1.0),),
            boundary_candidates=(
                _boundary_candidate(
                    anchor_id=0,
                    time_ms=30_000.0,
                    left_period_ms=500.0,
                    right_period_ms=400.0,
                ),
            ),
        ),
    )

    _assert_ok(result)
    exact_cut = _find_record(
        result.diagnostics.boundary_ownership_records,
        "boundary_time_ms",
        30_000.0,
    )
    assert _record_value(exact_cut, "owner_block_index") == 1
    assert _record_value(exact_cut, "committed_by_previous_block") is False


def test_next_core_commits_jump_once_when_short_lookahead_defers_it() -> None:
    prediction = _jump_prediction(
        first_bpm=120.0,
        second_bpm=150.0,
        boundary_ms=32_000.0,
        duration_ms=72_000.0,
        include_downbeats=False,
    )
    candidate_set = _candidate_set_for_prediction(
        prediction,
        tempo_candidates=(
            TempoCandidate(bpm=120.0, source="test", score=1.0),
            TempoCandidate(bpm=150.0, source="test", score=0.9),
        ),
        origin_candidates=(OriginCandidate(anchor_id=0, time_ms=0.0, bpm=120.0, score=1.0),),
        boundary_candidates=(
            _boundary_candidate(
                anchor_id=0,
                time_ms=32_000.0,
                left_period_ms=500.0,
                right_period_ms=400.0,
            ),
        ),
    )

    result = fit_local_frontier_constant_jump(
        prediction,
        config=LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S30),
        candidate_set=candidate_set,
    )

    grid = _assert_ok(result)
    assert len(grid.sections) == 2
    assert grid.section_end_time_ms(0) == pytest.approx(32_000.0)

    blocks = result.diagnostics.block_diagnostics
    assert len(blocks) >= 2
    assert _record_value(blocks[0], "cut_time_ms") == pytest.approx(0.0)
    assert _record_value(blocks[0], "core_end_ms") == pytest.approx(30_000.0)
    assert _record_value(blocks[0], "lookahead_end_ms") == pytest.approx(40_000.0)
    assert _record_value(blocks[1], "cut_time_ms") == pytest.approx(30_000.0)
    assert _record_value(blocks[1], "core_start_ms") == pytest.approx(30_000.0)

    jump_record = _find_record(
        result.diagnostics.boundary_ownership_records,
        "boundary_time_ms",
        32_000.0,
    )
    assert _record_value(jump_record, "provisional_block_indexes") == ()
    assert _record_value(jump_record, "owner_block_index") == 1
    assert _record_value(jump_record, "objective_charge_count") == 1


def test_strong_s90_lookahead_jump_is_recomputed_and_charged_once() -> None:
    prediction = _jump_prediction(
        first_bpm=120.0,
        second_bpm=127.0,
        boundary_ms=92_000.0,
        duration_ms=200_000.0,
        include_downbeats=True,
    )
    result = fit_local_frontier_constant_jump(
        prediction,
        config=LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S90),
        candidate_set=_candidate_set_for_prediction(
            prediction,
            tempo_candidates=(
                TempoCandidate(bpm=120.0, source="test", score=1.0),
                TempoCandidate(bpm=127.0, source="test", score=0.9),
            ),
            origin_candidates=(OriginCandidate(anchor_id=0, time_ms=0.0, bpm=120.0, score=1.0),),
            boundary_candidates=(
                _boundary_candidate(
                    anchor_id=0,
                    time_ms=92_000.0,
                    left_period_ms=500.0,
                    right_period_ms=60000.0 / 127.0,
                ),
            ),
        ),
    )

    grid = _assert_ok(result)
    assert [section.bpm for section in grid.sections] == pytest.approx([120.0, 127.0])
    assert grid.section_end_time_ms(0) == pytest.approx(92_000.0)

    blocks = result.diagnostics.block_diagnostics
    jump_record = _find_record(
        result.diagnostics.boundary_ownership_records,
        "boundary_time_ms",
        92_000.0,
    )
    assert _record_value(jump_record, "provisional_block_indexes") == (0,)
    assert _record_value(jump_record, "owner_block_index") == 1
    assert _record_value(jump_record, "objective_charge_count") == 1

    recompute = _find_record(
        result.diagnostics.lookahead_recompute_records,
        "boundary_time_ms",
        92_000.0,
    )
    assert _record_value(recompute, "previous_block_index") == 0
    assert _record_value(recompute, "next_block_index") == 1
    assert _record_value(recompute, "provisional_trace_fingerprint") == _record_value(
        blocks[0],
        "lookahead_trace_fingerprint",
    )
    assert _record_value(recompute, "recomputed_trace_fingerprint") == _record_value(
        blocks[1],
        "lookahead_trace_fingerprint",
    )
    assert _record_value(recompute, "objective_charge_count") == 1


def test_tail_closure_extends_to_smallest_integer_beat_covering_cache_end() -> None:
    result = fit_local_frontier_constant_jump(
        _constant_prediction(duration_ms=20_120.0, bpm=120.0, downbeat_phase=None),
        config=LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S60),
    )

    grid = _assert_ok(result)
    assert len(grid.sections) == 1
    assert grid.coverage_end_ms == pytest.approx(20_120.0)
    assert grid.end_beat == 41
    assert grid.end_time_ms == pytest.approx(20_500.0)
    assert grid.end_time_ms >= grid.coverage_end_ms


def test_s64_uses_bootstrap_schedule_then_q_ref_beat_scheduled_common_cuts() -> None:
    bootstrap = block_schedule_for_cut(
        0.0,
        coverage_end_ms=300_000.0,
        schedule=LocalFrontierScheduleArm.S64,
    )
    assert bootstrap.core_duration_ms == pytest.approx(90_000.0)
    assert bootstrap.lookahead_duration_ms == pytest.approx(30_000.0)
    assert bootstrap.next_cut_time_ms == pytest.approx(90_000.0)

    fast = block_schedule_for_cut(
        90_000.0,
        coverage_end_ms=300_000.0,
        schedule=LocalFrontierScheduleArm.S64,
        q_ref_bpm=240.0,
    )
    assert fast.core_duration_ms == pytest.approx(30_000.0)
    assert fast.lookahead_duration_ms == pytest.approx(10_000.0)
    assert fast.next_cut_time_ms == pytest.approx(120_000.0)

    slow = block_schedule_for_cut(
        120_000.0,
        coverage_end_ms=300_000.0,
        schedule=LocalFrontierScheduleArm.S64,
        q_ref_bpm=60.0,
    )
    assert slow.core_duration_ms == pytest.approx(64_000.0)
    assert slow.lookahead_duration_ms == pytest.approx(16_000.0)
    assert slow.next_cut_time_ms == pytest.approx(184_000.0)


def test_s64_q_ref_schedule_clamps_to_frozen_min_max_windows() -> None:
    fastest = block_schedule_for_cut(
        90_000.0,
        coverage_end_ms=300_000.0,
        schedule=LocalFrontierScheduleArm.S64,
        q_ref_bpm=1000.0,
    )
    assert fastest.core_duration_ms == pytest.approx(30_000.0)
    assert fastest.lookahead_duration_ms == pytest.approx(10_000.0)
    assert fastest.next_cut_time_ms == pytest.approx(120_000.0)

    slowest = block_schedule_for_cut(
        120_000.0,
        coverage_end_ms=300_000.0,
        schedule=LocalFrontierScheduleArm.S64,
        q_ref_bpm=20.0,
    )
    assert slowest.core_duration_ms == pytest.approx(90_000.0)
    assert slowest.lookahead_duration_ms == pytest.approx(30_000.0)
    assert slowest.next_cut_time_ms == pytest.approx(210_000.0)


def test_local_frontier_pure_contracts_fail_closed_on_invalid_config_nan_and_bpm() -> None:
    with pytest.raises(ValueError, match="unsupported local-frontier schedule arm"):
        LocalFrontierConfig(schedule_arm="S120")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exported_frontier_width"):
        LocalFrontierConfig(exported_frontier_width=17)
    with pytest.raises(ValueError, match="frozen at 64"):
        LocalFrontierConfig(local_beam_width=65)

    with pytest.raises(ValueError, match="current_bpm"):
        _frontier_state(current_bpm=math.nan)
    with pytest.raises(ValueError, match="current_bpm"):
        _frontier_state(current_bpm=19.999)
    with pytest.raises(ValueError, match="previous_bpm"):
        _frontier_state(previous_bpm=1000.001)
    with pytest.raises(ValueError, match="global_downbeat_phase"):
        _frontier_state(global_downbeat_phase=4)

    with pytest.raises(ValueError, match="frozen guard"):
        lattice_time_ms(0, origin_time_ms=0.0, bpm=math.nan)
    with pytest.raises(ValueError, match="frozen guard"):
        lattice_time_ms(0, origin_time_ms=0.0, bpm=1000.001)
    with pytest.raises(ValueError, match="time_ms must be finite"):
        next_beat_on_or_after(math.nan, origin_time_ms=0.0, bpm=120.0)
    with pytest.raises(ValueError, match="q_ref_bpm"):
        block_schedule_for_cut(
            0.0,
            coverage_end_ms=120_000.0,
            schedule=LocalFrontierScheduleArm.S64,
            q_ref_bpm=math.nan,
        )
    with pytest.raises(ValueError, match="cut_time_ms"):
        export_frontier((_frontier_state(),), cut_time_ms=math.nan, max_states=16)


def test_restricted_prediction_removes_source_path_keeps_arrays_read_only_and_shared() -> None:
    prediction = _metadata_trapping_constant_prediction(
        duration_ms=20_000.0,
        bpm=120.0,
        source_path="should-not-enter-core.osu",
    )

    restricted = restrict_timing_prediction(prediction)

    assert restricted.source_path is None
    assert np.shares_memory(restricted.beat_prob, prediction.beat_prob)
    assert np.shares_memory(restricted.downbeat_prob, prediction.downbeat_prob)
    assert not restricted.beat_prob.flags.writeable
    assert not restricted.downbeat_prob.flags.writeable
    validate_restricted_prediction(restricted)
    with pytest.raises(ValueError, match="source_path"):
        validate_restricted_prediction(prediction)

    prediction.arm_metadata_traps()
    result = fit_local_frontier_constant_jump(
        restricted,
        config=LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S30),
    )
    _assert_ok(result)


def test_fit_crosses_array_only_boundary_before_reading_armed_metadata() -> None:
    prediction = _metadata_trapping_constant_prediction(
        duration_ms=20_000.0,
        bpm=120.0,
        source_path="must-not-be-read-before-restriction.osu",
    )
    prediction.arm_metadata_traps()

    result = fit_local_frontier_constant_jump(
        prediction,
        config=LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S30),
    )

    _assert_ok(result)


def test_missing_origin_is_tagged_fallback_without_truncated_success() -> None:
    result = fit_local_frontier_constant_jump(
        _blank_prediction(duration_ms=20_000.0),
        config=LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S30),
    )

    assert isinstance(result, LocalFrontierResult)
    assert not result.ok
    assert result.grid is None
    assert result.reason == "no_origin_candidate"
    assert result.diagnostics.fallback_reason == "no_origin_candidate"
    assert result.diagnostics.fallback_stage == "bootstrap"
    assert result.diagnostics.grid_fingerprint is None
    _assert_result_json_finite(result)


def test_deterministic_projection_and_json_roundtrip_preserve_zero_section_seams() -> None:
    prediction = _jump_prediction(
        first_bpm=120.0,
        second_bpm=180.0,
        boundary_ms=32_000.0,
        duration_ms=72_000.0,
        include_downbeats=True,
    )
    config = LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S30)

    first = fit_local_frontier_constant_jump(prediction, config=config)
    second = fit_local_frontier_constant_jump(prediction, config=config)
    first_grid = _assert_ok(first)
    second_grid = _assert_ok(second)

    assert first_grid.to_dict() == second_grid.to_dict()
    assert first.diagnostics.replay_fingerprint == second.diagnostics.replay_fingerprint
    assert first.diagnostics.grid_fingerprint == second.diagnostics.grid_fingerprint
    assert first.diagnostics.deterministic_projection_sha256 == (
        second.diagnostics.deterministic_projection_sha256
    )

    restored = TimingV3Grid.from_dict(json.loads(json.dumps(first_grid.to_dict(), allow_nan=False)))
    assert restored.to_dict() == first_grid.to_dict()
    for section_index in range(len(restored.sections) - 1):
        assert restored.section_end_time_ms(section_index) == restored.section_start_time_ms(section_index + 1)
    _assert_result_json_finite(first)


def test_diagnostics_report_per_block_cuts_and_no_final_global_rescore() -> None:
    result = fit_local_frontier_constant_jump(
        _constant_prediction(duration_ms=95_000.0, bpm=120.0, downbeat_phase=None),
        config=LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S30),
    )

    _assert_ok(result)
    assert result.diagnostics.final_global_rescore_count == 0
    assert result.diagnostics.block_count == len(result.diagnostics.block_diagnostics)
    assert len(result.diagnostics.frontier_widths) == result.diagnostics.block_count

    previous_core_end_ms = 0.0
    for block_index, block in enumerate(result.diagnostics.block_diagnostics):
        assert _record_value(block, "block_index") == block_index
        assert _record_value(block, "cut_time_ms") == pytest.approx(_record_value(block, "core_start_ms"))
        assert _record_value(block, "core_start_ms") == pytest.approx(previous_core_end_ms)
        assert _record_value(block, "core_start_ms") < _record_value(block, "core_end_ms")
        assert _record_value(block, "core_end_ms") <= _record_value(block, "lookahead_end_ms")
        assert _record_value(block, "frontier_state_count") == result.diagnostics.frontier_widths[block_index]
        assert _record_value(block, "bucket_width_max") <= 64
        previous_core_end_ms = _record_value(block, "core_end_ms")

    assert previous_core_end_ms == pytest.approx(result.diagnostics.coverage_end_ms)


def test_unambiguous_multi_block_constant_grid_is_schedule_invariant() -> None:
    prediction = _constant_prediction(duration_ms=210_000.0, bpm=120.0, downbeat_phase=None)
    candidate_set = _candidate_set_for_prediction(
        prediction,
        tempo_candidates=(TempoCandidate(bpm=120.0, source="test", score=1.0),),
        origin_candidates=(OriginCandidate(anchor_id=0, time_ms=0.0, bpm=120.0, score=1.0),),
        boundary_candidates=(),
    )

    results = {
        arm: fit_local_frontier_constant_jump(
            prediction,
            config=LocalFrontierConfig(schedule_arm=arm),
            candidate_set=candidate_set,
        )
        for arm in LocalFrontierScheduleArm
    }
    grids = {arm: _assert_ok(result).to_dict() for arm, result in results.items()}

    assert len({json.dumps(grid, sort_keys=True) for grid in grids.values()}) == 1
    for result in results.values():
        assert result.diagnostics.selected_section_count == 1
        assert result.diagnostics.boundary_ownership_records == ()
        assert result.diagnostics.lookahead_recompute_records == ()
        assert result.diagnostics.first_section_logical_beat_count == (
            result.grid.sections[0].end_beat - result.grid.origin_beat
        )


@pytest.mark.parametrize(
    ("boundary_ms", "expected_owner_block"),
    (
        (29_500.0, 0),
        (30_000.0, 1),
        (30_500.0, 1),
    ),
)
def test_jump_boundary_at_cut_and_adjacent_beats_has_exact_owner_and_seam(
    boundary_ms: float,
    expected_owner_block: int,
) -> None:
    prediction = _jump_prediction(
        first_bpm=120.0,
        second_bpm=150.0,
        boundary_ms=boundary_ms,
        duration_ms=72_000.0,
        include_downbeats=False,
    )

    result = fit_local_frontier_constant_jump(
        prediction,
        config=LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S30),
        candidate_set=_candidate_set_for_prediction(
            prediction,
            tempo_candidates=(
                TempoCandidate(bpm=120.0, source="test", score=1.0),
                TempoCandidate(bpm=150.0, source="test", score=0.9),
            ),
            origin_candidates=(OriginCandidate(anchor_id=0, time_ms=0.0, bpm=120.0, score=1.0),),
            boundary_candidates=(
                _boundary_candidate(
                    anchor_id=0,
                    time_ms=boundary_ms,
                    left_period_ms=500.0,
                    right_period_ms=400.0,
                ),
            ),
        ),
    )

    grid = _assert_ok(result)
    assert [section.bpm for section in grid.sections] == pytest.approx([120.0, 150.0])
    assert grid.section_end_time_ms(0) == pytest.approx(boundary_ms)
    assert grid.section_start_time_ms(1) == pytest.approx(boundary_ms)

    record = _find_record(
        result.diagnostics.boundary_ownership_records,
        "boundary_time_ms",
        boundary_ms,
    )
    assert _record_value(record, "owner_block_index") == expected_owner_block
    assert _record_value(record, "objective_charge_count") == 1


@pytest.mark.parametrize("source_candidate_time_ms", (29_960.0, 30_040.0))
def test_snapped_t_jump_not_observed_candidate_time_owns_half_open_core(
    source_candidate_time_ms: float,
) -> None:
    prediction = _jump_prediction(
        first_bpm=120.0,
        second_bpm=150.0,
        boundary_ms=30_000.0,
        duration_ms=72_000.0,
        include_downbeats=False,
    )
    # Candidate-contract injection remains strict: add an actual source-owned
    # peak at the observed (deliberately unsnapped) proposal time.
    _write_pulse(
        prediction.beat_prob,
        time_ms=source_candidate_time_ms,
        frame_rate_hz=prediction.frame_rate_hz,
    )
    result = fit_local_frontier_constant_jump(
        prediction,
        config=LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S30),
        candidate_set=_candidate_set_for_prediction(
            prediction,
            tempo_candidates=(
                TempoCandidate(bpm=120.0, source="test", score=1.0),
                TempoCandidate(bpm=150.0, source="test", score=0.9),
            ),
            origin_candidates=(
                OriginCandidate(anchor_id=0, time_ms=0.0, bpm=120.0, score=1.0),
            ),
            boundary_candidates=(
                _boundary_candidate(
                    anchor_id=0,
                    time_ms=source_candidate_time_ms,
                    left_period_ms=500.0,
                    right_period_ms=400.0,
                ),
            ),
        ),
    )

    grid = _assert_ok(result)
    assert [section.bpm for section in grid.sections] == pytest.approx([120.0, 150.0])
    assert grid.section_end_time_ms(0) == pytest.approx(30_000.0)
    record = result.diagnostics.boundary_ownership_records[0]
    assert record.source_candidate_time_ms == pytest.approx(source_candidate_time_ms)
    assert record.boundary_time_ms == pytest.approx(30_000.0)
    assert record.owner_block_index == 1
    assert not record.committed_by_previous_block
    assert record.selected_traceback_objective_charge_count == 1
    assert record.objective_charge_count == 1


def test_interior_beam_width_is_shared_by_current_candidate_and_section_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # These model persistence/jump descendants arriving from two different
    # previous-anchor histories but sharing the same current right boundary
    # and resulting section count.  The old last-selected-anchor buckets would
    # retain 64 from each history (128 total).
    monkeypatch.setattr(
        lf,
        "_path_order_key",
        lambda path, context: (path.replay_key,),  # noqa: ARG005, SLF001
    )
    working = tuple(
        _working_path_for_bucket_test(
            index=index,
            last_boundary_anchor_id=(3 if index % 2 == 0 else 7),
        )
        for index in range(130)
    )

    retained = lf._prune_working_paths(  # noqa: SLF001
        working,
        right_boundary_candidate_id=7,
        context=object(),  # type: ignore[arg-type]
    )

    assert len(retained) == 64
    assert {item.path.last_boundary_anchor_id for item in retained} == {3, 7}
    assert tuple(item.path.replay_key for item in retained) == tuple((index,) for index in range(64))


def test_charge_diagnostics_separate_retained_alternative_from_selected_traceback() -> None:
    prediction = _constant_prediction(duration_ms=20_000.0, bpm=120.0, downbeat_phase=None)
    result = fit_local_frontier_constant_jump(
        prediction,
        config=LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S30),
        candidate_set=_candidate_set_for_prediction(
            prediction,
            tempo_candidates=(
                TempoCandidate(bpm=120.0, source="test", score=1.0),
                TempoCandidate(bpm=150.0, source="test", score=0.9),
            ),
            origin_candidates=(
                OriginCandidate(anchor_id=0, time_ms=0.0, bpm=120.0, score=1.0),
            ),
            boundary_candidates=(
                _boundary_candidate(
                    anchor_id=0,
                    time_ms=12_000.0,
                    left_period_ms=500.0,
                    right_period_ms=400.0,
                ),
            ),
        ),
    )

    grid = _assert_ok(result)
    assert len(grid.sections) == 1
    record = result.diagnostics.boundary_ownership_records[0]
    assert record.retained_frontier_charge_path_count > 0
    assert record.selected_traceback_objective_charge_count == 0
    assert record.objective_charge_count == 0
    assert {
        "retained_frontier_charge_path_count",
        "selected_traceback_objective_charge_count",
        "objective_charge_count",
    } <= asdict(record).keys()


def test_two_jump_fixture_pins_exp005_constant_change_shortcut() -> None:
    prediction = _two_jump_prediction(
        first_bpm=120.0,
        second_bpm=150.0,
        third_bpm=100.0,
        first_boundary_ms=12_000.0,
        second_boundary_ms=36_000.0,
        duration_ms=72_000.0,
        include_downbeats=False,
    )

    result = fit_local_frontier_constant_jump(
        prediction,
        config=LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S30),
        candidate_set=_candidate_set_for_prediction(
            prediction,
            tempo_candidates=(
                TempoCandidate(bpm=120.0, source="test", score=1.0),
                TempoCandidate(bpm=150.0, source="test", score=0.9),
                TempoCandidate(bpm=100.0, source="test", score=0.8),
            ),
            origin_candidates=(OriginCandidate(anchor_id=0, time_ms=0.0, bpm=120.0, score=1.0),),
            boundary_candidates=(
                _boundary_candidate(
                    anchor_id=0,
                    time_ms=12_000.0,
                    left_period_ms=500.0,
                    right_period_ms=400.0,
                ),
                _boundary_candidate(
                    anchor_id=1,
                    time_ms=36_000.0,
                    left_period_ms=400.0,
                    right_period_ms=600.0,
                ),
            ),
        ),
    )

    grid = _assert_ok(result)
    # This is the frozen Experiment 005 negative result.  Experiment 006 has a
    # separately named API and test; changing the default here would erase the
    # comparator that motivated its pair-conditioned transition potential.
    assert [section.bpm for section in grid.sections] == pytest.approx([120.0, 100.0])
    assert grid.section_end_time_ms(0) == pytest.approx(12_000.0)
    assert grid.section_start_time_ms(1) == pytest.approx(12_000.0)
    assert [record.objective_charge_count for record in result.diagnostics.boundary_ownership_records] == [1, 0]


def test_rejected_boundary_proposal_does_not_split_persistence_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction = _constant_prediction(duration_ms=20_000.0, bpm=120.0, downbeat_phase=None)
    boundary_ms = 7_500.0
    candidate_set = _candidate_set_for_prediction(
        prediction,
        tempo_candidates=(
            TempoCandidate(bpm=120.0, source="test", score=1.0),
            TempoCandidate(bpm=150.0, source="test", score=0.9),
        ),
        origin_candidates=(OriginCandidate(anchor_id=0, time_ms=0.0, bpm=120.0, score=1.0),),
        boundary_candidates=(
            _boundary_candidate(
                anchor_id=0,
                time_ms=boundary_ms,
                left_period_ms=500.0,
                right_period_ms=400.0,
            ),
        ),
    )
    intervals: list[tuple[float, float]] = []
    original_local_cost = lf._LocalScoreContext.local_cost  # noqa: SLF001

    def recording_local_cost(self: object, **kwargs: Any) -> float:
        intervals.append((float(kwargs["start_ms"]), float(kwargs["end_ms"])))
        return original_local_cost(self, **kwargs)

    monkeypatch.setattr(lf._LocalScoreContext, "local_cost", recording_local_cost)  # noqa: SLF001

    result = fit_local_frontier_constant_jump(
        prediction,
        config=LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S30),
        candidate_set=candidate_set,
    )

    grid = _assert_ok(result)
    assert len(grid.sections) == 1
    assert _find_record(
        result.diagnostics.boundary_ownership_records,
        "boundary_time_ms",
        boundary_ms,
    ).objective_charge_count == 0
    assert (0.0, 20_000.0) in intervals
    assert all(boundary_ms not in interval for interval in intervals)


def test_resource_cap_exception_returns_tagged_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction = _constant_prediction(duration_ms=20_000.0, bpm=120.0, downbeat_phase=None)
    candidate_set = _candidate_set_for_prediction(
        prediction,
        tempo_candidates=(TempoCandidate(bpm=120.0, source="test", score=1.0),),
        origin_candidates=(OriginCandidate(anchor_id=0, time_ms=0.0, bpm=120.0, score=1.0),),
        boundary_candidates=(),
    )

    def raise_resource_cap(self: object, **kwargs: Any) -> float:
        raise lf._ResourceCapExceeded  # noqa: SLF001

    monkeypatch.setattr(lf._LocalScoreContext, "local_cost", raise_resource_cap)  # noqa: SLF001

    result = fit_local_frontier_constant_jump(
        prediction,
        config=LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S30),
        candidate_set=candidate_set,
    )

    assert not result.ok
    assert result.grid is None
    assert result.reason == "local_frontier_resource_cap_exceeded"
    assert result.diagnostics.fallback_reason == "local_frontier_resource_cap_exceeded"
    assert result.diagnostics.fallback_stage == "local_graph"
    _assert_result_json_finite(result)


@pytest.mark.parametrize(
    ("bpm", "duration_ms", "frame_rate_hz"),
    (
        (20.0, 12_000.0, 100.0),
        (1000.0, 6_000.0, 1000.0),
    ),
)
def test_exact_frozen_bpm_guard_edges_fit_as_valid_constant_grids(
    bpm: float,
    duration_ms: float,
    frame_rate_hz: float,
) -> None:
    prediction = _constant_prediction(
        duration_ms=duration_ms,
        bpm=bpm,
        frame_rate_hz=frame_rate_hz,
        downbeat_phase=None,
    )

    result = fit_local_frontier_constant_jump(
        prediction,
        config=LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S30),
        candidate_set=_candidate_set_for_prediction(
            prediction,
            tempo_candidates=(TempoCandidate(bpm=bpm, source="test", score=1.0),),
            origin_candidates=(OriginCandidate(anchor_id=0, time_ms=0.0, bpm=bpm, score=1.0),),
            boundary_candidates=(),
        ),
    )

    grid = _assert_ok(result)
    assert len(grid.sections) == 1
    assert grid.sections[0].bpm == pytest.approx(bpm)
    assert grid.end_time_ms >= grid.coverage_end_ms
    assert _frontier_state(current_bpm=bpm).current_bpm == pytest.approx(bpm)


def test_candidate_set_is_not_mutated_and_source_hash_is_stable_across_repeats() -> None:
    prediction = restrict_timing_prediction(
        _jump_prediction(
            first_bpm=120.0,
            second_bpm=150.0,
            boundary_ms=32_000.0,
            duration_ms=72_000.0,
            include_downbeats=False,
        )
    )
    candidate_set = _candidate_set_for_prediction(
        prediction,
        tempo_candidates=(
            TempoCandidate(bpm=120.0, source="test", score=1.0),
            TempoCandidate(bpm=150.0, source="test", score=0.9),
        ),
        origin_candidates=(OriginCandidate(anchor_id=0, time_ms=0.0, bpm=120.0, score=1.0),),
        boundary_candidates=(
            _boundary_candidate(
                anchor_id=0,
                time_ms=32_000.0,
                left_period_ms=500.0,
                right_period_ms=400.0,
            ),
        ),
    )
    candidate_payload_before = asdict(candidate_set)
    beat_before = prediction.beat_prob.copy()
    downbeat_before = prediction.downbeat_prob.copy()

    first = fit_local_frontier_constant_jump(
        prediction,
        config=LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S30),
        candidate_set=candidate_set,
    )
    second = fit_local_frontier_constant_jump(
        prediction,
        config=LocalFrontierConfig(schedule_arm=LocalFrontierScheduleArm.S30),
        candidate_set=candidate_set,
    )

    first_grid = _assert_ok(first)
    second_grid = _assert_ok(second)
    assert asdict(candidate_set) == candidate_payload_before
    assert np.array_equal(prediction.beat_prob, beat_before)
    assert np.array_equal(prediction.downbeat_prob, downbeat_before)
    assert not prediction.beat_prob.flags.writeable
    assert not prediction.downbeat_prob.flags.writeable
    with pytest.raises(ValueError):
        prediction.beat_prob[0] = np.float32(0.0)
    assert first.diagnostics.input_signal_sha256 == candidate_set.diagnostics.input_signal_sha256
    assert first.diagnostics.candidate_fingerprint == candidate_set.diagnostics.candidate_fingerprint
    assert first_grid.to_dict() == second_grid.to_dict()
    assert first.diagnostics.replay_fingerprint == second.diagnostics.replay_fingerprint
    assert first.diagnostics.grid_fingerprint == second.diagnostics.grid_fingerprint


def _frontier_state(
    *,
    cut_time_ms: float = 30_000.0,
    next_beat_index: int = 60,
    next_beat_time_ms: float = 30_000.0,
    current_section_start_beat: int = 0,
    current_section_start_time_ms: float = 0.0,
    serialized_first_start_beat: int = 0,
    current_bpm: float = 120.0,
    previous_bpm: float | None = None,
    alias_family: float = 120.0,
    global_downbeat_phase: int | None = None,
    committed_duration_objective_numerator: float = 0.0,
    committed_transition_objective: float = 0.0,
    rank_objective: float = 0.0,
    committed_objective: float = 0.0,
    real_section_count: int = 1,
    alias_switch_count: int = 0,
    max_boundary_displacement_ms: float = 0.0,
    deterministic_replay_key: tuple[Any, ...] = ("state",),
) -> LocalFrontierState:
    return LocalFrontierState(
        cut_time_ms=cut_time_ms,
        next_beat_index=next_beat_index,
        next_beat_time_ms=next_beat_time_ms,
        current_section_start_beat=current_section_start_beat,
        current_section_start_time_ms=current_section_start_time_ms,
        serialized_first_start_beat=serialized_first_start_beat,
        current_bpm=current_bpm,
        previous_bpm=previous_bpm,
        alias_family=alias_family,
        global_downbeat_phase=global_downbeat_phase,
        committed_duration_objective_numerator=committed_duration_objective_numerator,
        committed_transition_objective=committed_transition_objective,
        rank_objective=rank_objective,
        committed_objective=committed_objective,
        real_section_count=real_section_count,
        alias_switch_count=alias_switch_count,
        max_boundary_displacement_ms=max_boundary_displacement_ms,
        open_section_state=(
            current_section_start_beat,
            float.hex(current_section_start_time_ms),
            float.hex(current_bpm),
            float.hex(cut_time_ms),
            "closure_prior_unpaid",
        ),
        prefix_sections_or_backpointer=(),
        deterministic_replay_key=deterministic_replay_key,
    )


def _working_path_for_bucket_test(
    *, index: int, last_boundary_anchor_id: int
) -> lf._WorkingPath:  # noqa: SLF001
    path = lf._Path(  # noqa: SLF001
        origin_time_ms=0.0,
        serialized_first_start_beat=0,
        open_start_beat=index,
        open_start_time_ms=float(index),
        current_bpm=120.0,
        previous_bpm=None,
        global_downbeat_phase=None,
        closed_sections=(),
        duration_objective_numerator=0.0,
        transition_objective=0.0,
        real_section_count=2,
        alias_switch_count=0,
        max_boundary_displacement_ms=0.0,
        replay_key=(index,),
        selected_boundaries=(),
        last_boundary_anchor_id=last_boundary_anchor_id,
    )
    return lf._WorkingPath(path=path, cursor_time_ms=10_000.0)  # noqa: SLF001


def _closure_counts(candidates: Sequence[Any]) -> tuple[int, ...]:
    return tuple(candidate.count for candidate in candidates)


def _closure_boundary_beats(candidates: Sequence[Any]) -> tuple[int, ...]:
    return tuple(candidate.boundary_beat for candidate in candidates)


def _closure_boundary_times_ms(candidates: Sequence[Any]) -> tuple[float, ...]:
    return tuple(candidate.boundary_time_ms for candidate in candidates)


def _boundary_candidate(
    *,
    anchor_id: int,
    time_ms: float,
    left_period_ms: float,
    right_period_ms: float,
) -> BoundaryCandidate:
    return BoundaryCandidate(
        anchor_id=anchor_id,
        time_ms=time_ms,
        source_peak_index=anchor_id,
        source_peak_time_ms=time_ms,
        source_peak_confidence=1.0,
        rank_score=1.0,
        evidence_mode="ordinary",
        left_period_ms=left_period_ms,
        right_period_ms=right_period_ms,
        ordinary_score=1.0,
        super_score=None,
        downbeat_bonus=0.0,
        nearest_downbeat_distance_ms=None,
    )


def _find_record(records: Sequence[Any], field: str, value: float) -> Any:
    for record in records:
        if math.isclose(float(_record_value(record, field)), value, rel_tol=0.0, abs_tol=1e-6):
            return record
    raise AssertionError(f"record with {field}={value!r} not found in {records!r}")


def _record_value(record: Any, field: str) -> Any:
    if isinstance(record, dict):
        return record[field]
    return getattr(record, field)


def _contains_bpm(values: Iterable[float], expected: float) -> bool:
    return any(math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-6) for value in values)


def _bpm_index(values: Sequence[float], expected: float) -> int:
    for index, value in enumerate(values):
        if math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-6):
            return index
    raise AssertionError(f"BPM {expected!r} not found in shortlist {values!r}")


def _assert_ok(result: LocalFrontierResult) -> TimingV3Grid:
    assert isinstance(result, LocalFrontierResult)
    assert result.ok
    assert result.reason is None
    assert result.grid is not None
    assert result.diagnostics.fallback_reason is None
    assert isinstance(result.grid, TimingV3Grid)
    return result.grid


def _assert_result_json_finite(result: LocalFrontierResult) -> None:
    payload = {
        "reason": result.reason,
        "diagnostics": _jsonable(result.diagnostics),
        "grid": None if result.grid is None else result.grid.to_dict(),
    }
    json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_jsonable"):
        return value.to_jsonable()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


class _MetadataTrappingPrediction(FrameTimingPrediction):
    def arm_metadata_traps(self) -> None:
        object.__setattr__(self, "_metadata_traps_armed", True)

    def __getattribute__(self, name: str) -> object:
        if name in {
            "source_path",
            "checkpoint_path",
            "provider",
            "metadata",
            "title",
            "artist",
            "api_bpm",
            "difficulty_name",
        }:
            armed = object.__getattribute__(self, "__dict__").get("_metadata_traps_armed", False)
            if armed:
                raise AssertionError(f"unexpected metadata access: {name}")
        return super().__getattribute__(name)


def _metadata_trapping_constant_prediction(
    *,
    duration_ms: float,
    bpm: float,
    source_path: str,
    frame_rate_hz: float = 50.0,
) -> _MetadataTrappingPrediction:
    base = _constant_prediction(
        duration_ms=duration_ms,
        bpm=bpm,
        frame_rate_hz=frame_rate_hz,
        source_path=source_path,
    )
    return _MetadataTrappingPrediction(
        provider="cached-beatthis",
        checkpoint_path="checkpoint.pt",
        source_path=source_path,
        beat_prob=base.beat_prob,
        downbeat_prob=base.downbeat_prob,
        frame_rate_hz=base.frame_rate_hz,
    )


def _blank_prediction(
    *,
    duration_ms: float,
    frame_rate_hz: float = 50.0,
) -> FrameTimingPrediction:
    frame_count = int(round(duration_ms * frame_rate_hz / 1000.0))
    return FrameTimingPrediction(
        provider="cached-beatthis",
        checkpoint_path="checkpoint.pt",
        source_path=None,
        beat_prob=np.zeros(frame_count, dtype=np.float32),
        downbeat_prob=np.zeros(frame_count, dtype=np.float32),
        frame_rate_hz=frame_rate_hz,
    )


def _offset_constant_prediction(
    *,
    duration_ms: float,
    bpm: float,
    origin_time_ms: float,
    frame_rate_hz: float = 50.0,
    downbeat_phase: int | None = 0,
) -> FrameTimingPrediction:
    frame_count = int(round(duration_ms * frame_rate_hz / 1000.0))
    beat_prob = np.zeros(frame_count, dtype=np.float32)
    downbeat_prob = np.zeros(frame_count, dtype=np.float32)
    period_ms = 60000.0 / bpm
    beat = 0
    time_ms = origin_time_ms
    while time_ms < duration_ms:
        _write_pulse(beat_prob, time_ms=time_ms, frame_rate_hz=frame_rate_hz)
        if downbeat_phase is not None and beat % 4 == downbeat_phase:
            _write_pulse(downbeat_prob, time_ms=time_ms, frame_rate_hz=frame_rate_hz)
        beat += 1
        time_ms += period_ms
    return FrameTimingPrediction(
        provider="cached-beatthis",
        checkpoint_path="checkpoint.pt",
        source_path=None,
        beat_prob=beat_prob,
        downbeat_prob=downbeat_prob,
        frame_rate_hz=frame_rate_hz,
    )


def _constant_prediction(
    *,
    duration_ms: float,
    bpm: float,
    frame_rate_hz: float = 50.0,
    downbeat_phase: int | None = 0,
    source_path: str | None = None,
) -> FrameTimingPrediction:
    frame_count = int(round(duration_ms * frame_rate_hz / 1000.0))
    beat_prob = np.zeros(frame_count, dtype=np.float32)
    downbeat_prob = np.zeros(frame_count, dtype=np.float32)
    period_ms = 60000.0 / bpm
    beat = 0
    time_ms = 0.0
    while time_ms < duration_ms:
        _write_pulse(beat_prob, time_ms=time_ms, frame_rate_hz=frame_rate_hz)
        if downbeat_phase is not None and beat % 4 == downbeat_phase:
            _write_pulse(downbeat_prob, time_ms=time_ms, frame_rate_hz=frame_rate_hz)
        beat += 1
        time_ms += period_ms
    return FrameTimingPrediction(
        provider="cached-beatthis",
        checkpoint_path="checkpoint.pt",
        source_path=source_path,
        beat_prob=beat_prob,
        downbeat_prob=downbeat_prob,
        frame_rate_hz=frame_rate_hz,
    )


def _jump_prediction(
    *,
    first_bpm: float,
    second_bpm: float,
    boundary_ms: float,
    duration_ms: float,
    frame_rate_hz: float = 50.0,
    include_downbeats: bool = True,
    downbeat_phase: int = 0,
) -> FrameTimingPrediction:
    frame_count = int(round(duration_ms * frame_rate_hz / 1000.0))
    beat_prob = np.zeros(frame_count, dtype=np.float32)
    downbeat_prob = np.zeros(frame_count, dtype=np.float32)
    beat = 0
    time_ms = 0.0
    first_period_ms = 60000.0 / first_bpm
    while time_ms <= boundary_ms + 1e-9:
        _write_pulse(beat_prob, time_ms=time_ms, frame_rate_hz=frame_rate_hz)
        if include_downbeats and beat % 4 == downbeat_phase:
            _write_pulse(downbeat_prob, time_ms=time_ms, frame_rate_hz=frame_rate_hz)
        beat += 1
        time_ms += first_period_ms

    second_period_ms = 60000.0 / second_bpm
    time_ms = boundary_ms + second_period_ms
    while time_ms < duration_ms:
        _write_pulse(beat_prob, time_ms=time_ms, frame_rate_hz=frame_rate_hz)
        if include_downbeats and beat % 4 == downbeat_phase:
            _write_pulse(downbeat_prob, time_ms=time_ms, frame_rate_hz=frame_rate_hz)
        beat += 1
        time_ms += second_period_ms

    return FrameTimingPrediction(
        provider="cached-beatthis",
        checkpoint_path="checkpoint.pt",
        source_path=None,
        beat_prob=beat_prob,
        downbeat_prob=downbeat_prob,
        frame_rate_hz=frame_rate_hz,
    )


def _two_jump_prediction(
    *,
    first_bpm: float,
    second_bpm: float,
    third_bpm: float,
    first_boundary_ms: float,
    second_boundary_ms: float,
    duration_ms: float,
    frame_rate_hz: float = 50.0,
    include_downbeats: bool = True,
    downbeat_phase: int = 0,
) -> FrameTimingPrediction:
    frame_count = int(round(duration_ms * frame_rate_hz / 1000.0))
    beat_prob = np.zeros(frame_count, dtype=np.float32)
    downbeat_prob = np.zeros(frame_count, dtype=np.float32)
    beat = 0

    def write_segment(start_ms: float, end_ms: float, bpm: float, include_start: bool) -> float:
        nonlocal beat
        period_ms = 60000.0 / bpm
        time_ms = start_ms if include_start else start_ms + period_ms
        while time_ms <= end_ms + 1e-9 and time_ms < duration_ms:
            _write_pulse(beat_prob, time_ms=time_ms, frame_rate_hz=frame_rate_hz)
            if include_downbeats and beat % 4 == downbeat_phase:
                _write_pulse(downbeat_prob, time_ms=time_ms, frame_rate_hz=frame_rate_hz)
            beat += 1
            time_ms += period_ms
        return period_ms

    write_segment(0.0, first_boundary_ms, first_bpm, True)
    write_segment(first_boundary_ms, second_boundary_ms, second_bpm, False)
    write_segment(second_boundary_ms, duration_ms, third_bpm, False)

    return FrameTimingPrediction(
        provider="cached-beatthis",
        checkpoint_path="checkpoint.pt",
        source_path=None,
        beat_prob=beat_prob,
        downbeat_prob=downbeat_prob,
        frame_rate_hz=frame_rate_hz,
    )


def _write_pulse(
    signal: np.ndarray,
    *,
    time_ms: float,
    frame_rate_hz: float,
) -> None:
    frame = int(round(time_ms * frame_rate_hz / 1000.0))
    if 0 <= frame < signal.shape[0]:
        signal[frame] = max(signal[frame], np.float32(1.0))
    if 0 <= frame - 1 < signal.shape[0]:
        signal[frame - 1] = max(signal[frame - 1], np.float32(0.1))
    if 0 <= frame + 1 < signal.shape[0]:
        signal[frame + 1] = max(signal[frame + 1], np.float32(0.1))


def _candidate_set_for_prediction(
    prediction: FrameTimingPrediction,
    *,
    tempo_candidates: tuple[TempoCandidate, ...],
    origin_candidates: tuple[OriginCandidate, ...],
    boundary_candidates: tuple[BoundaryCandidate, ...],
) -> GlobalConstantJumpCandidateSet:
    beat_peaks = materialize_global_constant_jump_peaks(
        prediction.beat_prob,
        frame_rate_hz=prediction.frame_rate_hz,
    )
    downbeat_peaks = materialize_global_constant_jump_peaks(
        prediction.downbeat_prob,
        frame_rate_hz=prediction.frame_rate_hz,
    )
    boundary_candidates = tuple(
        _boundary_candidate_with_materialized_source_peak(boundary, beat_peaks)
        for boundary in boundary_candidates
    )
    beat_signal = np.asarray(prediction.beat_prob, dtype=np.float64)
    downbeat_signal = np.asarray(prediction.downbeat_prob, dtype=np.float64)
    input_signal_sha256 = exp004._input_signal_sha256(beat_signal, downbeat_signal)
    candidate_fingerprint = exp004._candidate_fingerprint(
        tempo_candidates=tuple(tempo_candidates),
        origin_candidates=tuple(origin_candidates),
        boundary_candidates=tuple(boundary_candidates),
        beat_peaks=tuple(beat_peaks),
        downbeat_peaks=tuple(downbeat_peaks),
        input_signal_sha256=input_signal_sha256,
    )
    frame_count, frame_rate_hz, coverage_start_ms, coverage_end_ms, min_period_frames, max_period_frames = (
        exp004._prediction_geometry(
            beat_signal,
            prediction.frame_rate_hz,
            GLOBAL_CONSTANT_JUMP_CONSTANTS,
        )
    )
    diagnostics = GlobalConstantJumpCandidateDiagnostics(
        candidate_contract_version=CANDIDATE_CONTRACT_VERSION,
        constants_json_sha256=GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON_SHA256,
        pulse_correlation_version=PULSE_CORRELATION_VERSION,
        boundary_candidate_score_version=BOUNDARY_CANDIDATE_SCORE_VERSION,
        frame_count=frame_count,
        frame_rate_hz=frame_rate_hz,
        coverage_start_ms=coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
        min_period_frames=min_period_frames,
        max_period_frames=max_period_frames,
        beat_peak_count=len(beat_peaks),
        downbeat_peak_count=len(downbeat_peaks),
        tempo_candidate_count=len(tempo_candidates),
        origin_candidate_count=len(origin_candidates),
        boundary_candidate_count=len(boundary_candidates),
        input_signal_sha256=input_signal_sha256,
        candidate_fingerprint=candidate_fingerprint,
    )
    return GlobalConstantJumpCandidateSet(
        beat_peaks=tuple(beat_peaks),
        downbeat_peaks=tuple(downbeat_peaks),
        tempo_candidates=tuple(tempo_candidates),
        origin_candidates=tuple(origin_candidates),
        boundary_candidates=tuple(boundary_candidates),
        diagnostics=diagnostics,
    )


def _boundary_candidate_with_materialized_source_peak(
    boundary: BoundaryCandidate,
    beat_peaks: tuple[Any, ...],
) -> BoundaryCandidate:
    for peak_index, peak in enumerate(beat_peaks):
        if peak.time_ms == boundary.source_peak_time_ms or peak.time_ms == boundary.time_ms:
            return replace(
                boundary,
                source_peak_index=peak_index,
                source_peak_time_ms=peak.time_ms,
                source_peak_confidence=peak.confidence,
            )
    raise AssertionError(
        "boundary candidate must reference an exact materialized beat peak: "
        f"anchor_id={boundary.anchor_id!r} time_ms={boundary.time_ms!r} "
        f"source_peak_time_ms={boundary.source_peak_time_ms!r}"
    )
