from dataclasses import FrozenInstanceError, asdict, replace
from itertools import product

import pytest

from ensomi_model.research.oracle_time_continuation.engine import ContinuationState, PredictionInput, prefill
from ensomi_model.research.oracle_time_continuation.replay import ExactReplayState, commit, legal_rows
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow, TimeSkeleton
from ensomi_model.research.scoped_style_modeling.dataset import ContractError


@pytest.mark.parametrize("occupied", list(product((False, True), repeat=4)))
@pytest.mark.parametrize("terminal", (False, True))
def test_support_and_commit_match_all_v3_transitions(occupied, terminal):
    state = ExactReplayState()
    if any(occupied):
        state = commit(state, CompleteRow(0, tuple(2 if held else 0 for held in occupied)))
    expected = set()
    for actions in product(range(4), repeat=4):
        valid = any(actions)
        after = []
        for held, action in zip(occupied, actions):
            valid &= (held, action) in {(False, 0), (True, 0), (False, 1), (False, 2), (True, 3)}
            after.append(held if action == 0 else action == 2)
        valid &= not terminal or not any(after)
        if valid:
            expected.add(actions)
            updated = commit(state, CompleteRow(1, actions), is_terminal=terminal)
            assert updated.occupancy == tuple(after)
            assert updated.row_count == state.row_count + 1
        elif any(actions):
            with pytest.raises(ContractError, match="occupancy|terminal"):
                commit(state, CompleteRow(1, actions), is_terminal=terminal)
    assert set(legal_rows(state, is_terminal=terminal)) == expected
    assert expected


def test_simultaneous_pre_post_clocks_release_only_rows_and_long_gap():
    state = commit(ExactReplayState(), CompleteRow(0, (2, 1, 0, 0)))
    saved = asdict(state)
    before = state.clocks_at(1_000_000)
    assert before.ln_age_ms == (1_000_000, None, None, None)
    assert before.lane_attack_ms == (1_000_000, 1_000_000, None, None)
    assert before.hand_attack_ms == (1_000_000, None)
    assert before.lane_release_ms == (None,) * 4
    assert state.clocks_at(2_000_000).ln_age_ms[0] == 2_000_000
    assert asdict(state) == saved
    after = commit(state, CompleteRow(1_000_000, (3, 2, 1, 2)))
    assert after.occupancy == (False, True, False, True)
    assert after.open_ln_start_ms == (None, 1_000_000, None, 1_000_000)
    clocks = after.clocks_at(1_000_000)
    assert clocks.previous_row_ms == 0
    assert clocks.ln_age_ms == (None, 0, None, 0)
    assert clocks.lane_release_ms == (0, None, None, None)
    assert clocks.hand_attack_ms == (0, 0)
    assert clocks.hand_release_ms == (0, None)
    assert after.note_count == 5
    closed = commit(after, CompleteRow(1_000_001, (0, 3, 0, 3)), is_terminal=True)
    assert closed.note_count == 5
    assert closed.occupancy == (False,) * 4
    assert closed.is_complete
    with pytest.raises(ContractError, match="completed"):
        commit(closed, CompleteRow(1_000_002, (1, 0, 0, 0)))


def test_bos_missing_predecessors_are_distinct_from_known_zero():
    state = ExactReplayState()
    clocks = state.clocks_at(0)
    assert clocks.previous_row_ms is None and clocks.since_first_row_ms is None
    assert clocks.lane_attack_ms == clocks.lane_release_ms == clocks.ln_age_ms == (None,) * 4
    assert state.occupancy == (False,) * 4
    after = commit(state, CompleteRow(0, (2, 0, 0, 0)))
    assert after.clocks_at(0).ln_age_ms == (0, None, None, None)
    assert after.clocks_at(0).previous_row_ms == 0
    with pytest.raises(FrozenInstanceError):
        after.note_count = 99


def test_illegal_commit_cannot_partially_update_or_recommit_a_row():
    state = commit(ExactReplayState(), CompleteRow(1, (0, 2, 0, 0)))
    original = asdict(state)
    with pytest.raises(ContractError, match="occupancy"):
        commit(state, CompleteRow(2, (2, 1, 0, 0)))
    assert asdict(state) == original
    for time in (0, 1):
        with pytest.raises(ContractError, match="strictly"):
            commit(state, CompleteRow(time, (1, 0, 0, 0)))
    with pytest.raises(ContractError, match="precedes"):
        state.clocks_at(0)
    with pytest.raises(ContractError, match="boolean"):
        legal_rows(state, is_terminal=1)


def test_query_is_pure_and_chunk_boundaries_do_not_close_lns(chord_source):
    source = chord_source
    state = source.prefix_state()
    assert all(state.replay.occupancy)
    original = asdict(state)
    assert state.query() == state.query()
    assert asdict(state) == original
    states = {state.next_index: state}
    queries = []
    for row in source.targets[state.next_index:]:
        queries.append(state.query())
        state = state.commit(row)
        states[state.next_index] = state
    for cut, checkpoint in states.items():
        assert prefill(source.skeleton, source.targets[:cut]) == checkpoint
        resumed = checkpoint
        for index in range(cut, len(source.targets)):
            assert resumed.query() == queries[index - source.minimum_seed().seed_row_count]
            resumed = resumed.commit(source.targets[index])
        assert resumed == state
    assert state.finished and state.replay.is_complete
    with pytest.raises(ContractError, match="exhausted"):
        state.query()


def test_window_end_is_not_terminal_and_true_end_has_nonempty_support():
    skeleton = TimeSkeleton((0, 10, 20))
    state = prefill(skeleton, (CompleteRow(0, (2, 2, 2, 2)),))
    assert not state.query().is_terminal
    state = state.commit(CompleteRow(10, (3, 0, 0, 0)))
    assert state.replay.occupancy == (False, True, True, True)
    assert state.query().is_terminal
    assert set(state.query().legal_actions) == {(0, 3, 3, 3), (1, 3, 3, 3)}
    with pytest.raises(ContractError, match="terminal"):
        state.commit(CompleteRow(20, (2, 3, 3, 3)))
    assert state.commit(CompleteRow(20, (1, 3, 3, 3))).finished
    with pytest.raises(ContractError, match="next skeleton time"):
        prefill(skeleton, (CompleteRow(10, (1, 0, 0, 0)),))
    with pytest.raises(ContractError, match="next skeleton time"):
        state.commit(CompleteRow(10, (3, 0, 0, 0)))


def test_committed_continuation_owns_seed_ln_endpoints(chord_source):
    source = chord_source
    original = asdict(source)
    state = source.prefix_state()
    generated = []
    while not state.finished:
        query = state.query()
        if query.is_terminal:
            actions = tuple(3 if held else 0 for held in query.history.occupancy)
        else:
            actions = (0, 0, 0, 3 if query.history.occupancy[3] else 1)
            assert query.history.open_ln_start_ms[0] == 800
        row = CompleteRow(query.time_ms, actions)
        generated.append(row)
        state = state.commit(row)
    assert generated[0].actions == (0, 0, 0, 3)
    assert generated[-1].time_ms == 1300
    assert generated[-1].actions == (3, 3, 3, 0)
    assert state.replay.occupancy == (False,) * 4
    assert asdict(source) == original


def test_mirroring_exchanges_hand_clocks_and_preserves_support():
    rows = [CompleteRow(0, (2, 1, 0, 0)), CompleteRow(10, (0, 0, 1, 2)),
            CompleteRow(20, (3, 2, 1, 0)), CompleteRow(100, (0, 3, 0, 3))]
    state = mirrored = ExactReplayState()
    for index, row in enumerate(rows):
        terminal = index == len(rows) - 1
        assert {actions[::-1] for actions in legal_rows(state, is_terminal=terminal)} == set(
            legal_rows(mirrored, is_terminal=terminal))
        state = commit(state, row, is_terminal=terminal)
        mirrored = commit(mirrored, CompleteRow(row.time_ms, row.actions[::-1]), is_terminal=terminal)
        assert state.occupancy[::-1] == mirrored.occupancy
        assert state.open_ln_start_ms[::-1] == mirrored.open_ln_start_ms
        assert state.last_lane_attack_ms[::-1] == mirrored.last_lane_attack_ms
        assert state.last_lane_release_ms[::-1] == mirrored.last_lane_release_ms
        assert state.clocks_at(row.time_ms).hand_attack_ms[::-1] == mirrored.clocks_at(row.time_ms).hand_attack_ms
        assert state.clocks_at(row.time_ms).hand_release_ms[::-1] == mirrored.clocks_at(row.time_ms).hand_release_ms


def test_execution_rejects_mismatched_replay_boundary():
    state = commit(ExactReplayState(), CompleteRow(10, (1, 0, 0, 0)))
    with pytest.raises(ContractError, match="boundary"):
        ContinuationState(TimeSkeleton((0, 10, 20)), state)
    with pytest.raises(ContractError, match="boundary"):
        ContinuationState(TimeSkeleton((10,)), state)
    with pytest.raises(ContractError, match="committed prefix"):
        replace(state, last_lane_attack_ms=(11, None, None, None))
    with pytest.raises(ContractError, match="Prediction time"):
        PredictionInput(10, False, state)
    with pytest.raises(ContractError, match="boolean"):
        PredictionInput(20, 1, state)
    with pytest.raises(ContractError, match="Prediction time"):
        PredictionInput(30, True, commit(state, CompleteRow(20, (1, 0, 0, 0)), is_terminal=True))
