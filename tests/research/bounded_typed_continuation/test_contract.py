from itertools import product

import pytest

from pulsefield_model.research.bounded_typed_continuation.contract import Arm, HEAD_ACTIONS, ROW_ACTIONS, Schedule, Timing
from pulsefield_model.research.oracle_time_continuation.schema import CompleteRow
from pulsefield_model.research.scoped_style_modeling.dataset import ContractError


def test_original_pressure_and_typed_skip_are_distinct_tasks():
    raw = Schedule(Arm.R0, Timing((100., 124., 200.)))
    typed = Schedule(Arm.R1, Timing((100., 124., 200.), (True, False, True)))
    raw, _ = raw.advance((1, 0, 0, 0))
    typed, _ = typed.advance((1, 0, 0, 0))
    assert not raw.row_possible((0, 0, 0, 0))
    assert sum(raw.row_support()) == 80
    assert [row for row, legal in zip(ROW_ACTIONS, typed.row_support()) if legal] == [(0, 0, 0, 0)]
    before = typed.replay
    typed, row = typed.advance(None)
    assert row is None and typed.replay is before and typed.index == 2
    assert typed.replay.clocks_at(200.).previous_row_ms == 100.


@pytest.mark.parametrize('arm', [Arm.R1, Arm.O1])
def test_typed_seed_preserves_cross_boundary_plans_and_true_start(arm):
    timing = Timing((0., 100., 300., 9000., 10000.), (True, True, True, True, False))
    rows = (CompleteRow(0., (2, 0, 0, 0)), CompleteRow(100., (0, 1, 0, 0)))
    state = Schedule.from_seed(arm, timing, rows, {0: 4})
    assert state.replay.open_ln_start_ms[0] == 0.
    assert state.replay.clocks_at(300.).ln_age_ms[0] == 300.
    assert timing.times_ms[state.known_ends[0]] - state.time_ms == 9700.
    if arm == Arm.R1:
        assert not state.row_possible((3, 1, 0, 0))
        assert state.row_possible((0, 1, 0, 0))
    else:
        assert not state.head_possible((1, 0, 0, 0))
        with pytest.raises(ContractError):
            state.advance((3, 1, 0, 0))


def test_original_seed_never_receives_future_endpoints():
    timing = Timing((0., 100., 200.))
    rows = (CompleteRow(0., (2, 0, 0, 0)),)
    state = Schedule.from_seed(Arm.R0, timing, rows)
    assert state.known_ends == (None,) * 4 and state.replay.occupancy[0]
    with pytest.raises(ContractError):
        Schedule.from_seed(Arm.R0, timing, rows, {0: 2})
    with pytest.raises(ContractError):
        Schedule(Arm.R0, Timing(timing.times_ms, (True, False, False)))


def test_r1_unknown_holds_can_close_but_seed_obligations_cannot():
    timing = Timing((0., 50., 100., 150.), (True, False, True, False))
    state = Schedule(Arm.R1, timing)
    state, _ = state.advance((2, 0, 0, 0))
    assert state.row_possible((0, 0, 0, 0)) and state.row_possible((3, 0, 0, 0))
    fixed = Schedule.from_seed(Arm.R1, timing, (CompleteRow(0., (2, 0, 0, 0)),), {0: 3})
    assert fixed.row_possible((0, 0, 0, 0)) and not fixed.row_possible((3, 0, 0, 0))


@pytest.mark.parametrize('arm', [Arm.R1, Arm.O1])
def test_next_required_onset_needs_a_strictly_preclosed_lane(arm):
    state = Schedule(arm, Timing((0., 100., 200.), (True, True, False)))
    if arm == Arm.R1:
        assert not state.row_possible((2, 2, 2, 2))
        state, _ = state.advance((2, 2, 2, 0))
        assert not state.row_possible((3, 3, 3, 0))
        assert state.row_possible((3, 0, 0, 1))
    else:
        assert not state.head_possible((2, 2, 2, 2))
        state, _ = state.advance((2, 2, 2, 0), {0: 1, 1: 2, 2: 2})
        assert not state.head_possible((1, 0, 0, 0))
        assert state.head_possible((0, 0, 0, 1))
        state, _ = state.advance((3, 0, 0, 1))


def test_r1_must_use_last_non_onset_release_opportunity_when_all_lanes_held():
    state = Schedule(Arm.R1, Timing((0., 40., 100., 160.), (True, False, True, False)))
    state, _ = state.advance((2, 2, 2, 2))
    assert not state.row_possible((0, 0, 0, 0))
    assert state.row_possible((3, 0, 0, 0))


def test_object_factor_support_has_every_feasible_assignment_and_normalizes():
    state = Schedule(Arm.O1, Timing((0., 40., 100., 160.), (True, False, True, False)))
    heads = (2, 2, 2, 2)
    assert state.head_possible(heads)
    total = 0.
    for choices in product((1, 2, 3), repeat=4):
        feasible = 1 in choices
        for order in ((0, 1, 2, 3), (3, 2, 1, 0)):
            assigned, probability = {}, 1.
            for lane in order:
                support = state.endpoint_support(heads, assigned, lane)
                if not support[choices[lane]]:
                    probability = 0.
                    break
                probability /= sum(support)
                assigned[lane] = choices[lane]
            assert (probability > 0) == feasible
            total += probability / 2
    assert total == pytest.approx(1.)
    assert state.endpoint_support(heads, {0: 3, 1: 2, 2: 3}, 3) == (False, True, False, False)


def test_objects_preserve_independent_ends_and_materialize_only_needed_rows():
    state = Schedule(Arm.O1, Timing((0., 40., 100., 160., 5000., 6000.), (True, True, True, True, False, False)))
    expected = [(2, 2, 0, 0), (0, 0, 1, 0), (3, 0, 2, 0), (1, 3, 0, 0), (0, 0, 3, 0), None]
    assignments = [{0: 2, 1: 3}, {}, {2: 4}, {}, {}, {}]
    observed = []
    for actions, ends in zip(expected, assignments):
        state, row = state.advance(actions, ends)
        observed.append(None if row is None else row.actions)
    assert observed == expected and state.finished
    assert state.replay.row_count == 5 and not any(state.replay.occupancy)
    assert state.replay.last_row.time_ms == 5000.


def test_mirrored_head_and_endpoint_support_match():
    timing = Timing((0., 40., 100., 160., 200.), (True, False, True, False, True))
    a = Schedule.from_seed(Arm.O1, timing, (CompleteRow(0., (2, 0, 0, 0)),), {0: 3})
    b = Schedule.from_seed(Arm.O1, timing, (CompleteRow(0., (0, 0, 0, 2)),), {3: 3})
    a, _ = a.advance(None)
    b, _ = b.advance(None)
    for heads in HEAD_ACTIONS:
        assert a.head_possible(heads) == b.head_possible(heads[::-1])
        if a.head_possible(heads):
            for lane, head in enumerate(heads):
                if head == 2:
                    assert a.endpoint_support(heads, {}, lane) == b.endpoint_support(heads[::-1], {}, 3 - lane)


def test_invalid_plans_and_omitted_due_releases_fail_without_mutation():
    state = Schedule(Arm.O1, Timing((0., 100., 200.), (True, False, True)))
    for ends in ({}, {0: 0}, {0: 3}, {0: True}):
        with pytest.raises(ContractError):
            state.advance((2, 0, 0, 0), ends)
    assert state.index == 0 and state.replay.row_count == 0
    state, _ = state.advance((2, 0, 0, 0), {0: 1})
    assert state.forced_row() == (3, 0, 0, 0)
    with pytest.raises(ContractError):
        state.advance(None)
    assert state.index == 1 and state.replay.occupancy[0]


def test_endpoint_support_keeps_candidates_beyond_any_small_rank_cap():
    state = Schedule(Arm.O1, Timing(tuple(float(i) for i in range(301)), (True,) + (False,) * 300))
    assert state.endpoint_bounds((2, 0, 0, 0), {}, 0) == (1, 301)
    assert state.endpoint_support((2, 0, 0, 0), {}, 0)[300]
    state, _ = state.advance((2, 0, 0, 0), {0: 300})
    with pytest.raises(ContractError, match='more rows'):
        Schedule(Arm.O1, state.timing, 0, state.replay, state.known_ends)


def test_typed_row_and_object_tasks_have_the_same_complete_chart_space():
    timing = Timing((0., 100., 140., 200., 300.), (True, True, False, True, False))
    seed = (CompleteRow(0., (0, 2, 2, 2)),)
    row_state = Schedule.from_seed(Arm.R1, timing, seed, {1: 4, 2: 4, 3: 4})
    object_state = Schedule.from_seed(Arm.O1, timing, seed, {1: 4, 2: 4, 3: 4})

    def row_charts(state, prefix=()):
        if state.finished:
            return {prefix}
        charts = set()
        for actions, legal in zip(ROW_ACTIONS, state.row_support()):
            if legal:
                following, row = state.advance(actions)
                charts |= row_charts(following, prefix + (actions,))
        return charts

    def object_charts(state, prefix=()):
        if state.finished:
            return {prefix}
        if not state.timing.onsets[state.index]:
            actions = state.forced_row()
            following, row = state.advance(actions)
            return object_charts(following, prefix + ((0, 0, 0, 0) if actions is None else actions,))
        charts = set()
        for heads, legal in zip(HEAD_ACTIONS, state.head_support()):
            if not legal:
                continue
            lanes = [c for c, head in enumerate(heads) if head == 2]
            assignments = [{}]
            for lane in lanes:
                assignments = [{**chosen, lane: end} for chosen in assignments for end, possible in
                               enumerate(state.endpoint_support(heads, chosen, lane)) if possible]
            actions = tuple(3 if end == state.index else head for end, head in zip(state.known_ends, heads))
            for chosen in assignments:
                following, row = state.advance(actions, chosen)
                charts |= object_charts(following, prefix + (actions,))
        return charts

    expected = row_charts(row_state)
    assert len(expected) == 4 and object_charts(object_state) == expected
