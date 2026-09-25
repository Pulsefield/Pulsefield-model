import numpy as np

from ensomi_model.research.oracle_time_continuation.replay import ExactReplayState, commit
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule, ControlSpan
from ensomi_model.research.typed_audio_continuation.program import (
    ACTIONS, HEADS, MARKS, MARK_INDEX, Recovery, Resources, bind_row, row_support,
)


def test_cross_column_short_events_keep_capacity_but_not_repeated_triples():
    recovery = Recovery()
    state, _ = Resources().advance(0, MARK_INDEX[3, 0, 0], recovery)
    allowed = state.support(21, 1000, recovery)
    assert allowed[MARK_INDEX[1, 0, 0]]
    assert not allowed[MARK_INDEX[2, 0, 0]]
    assert state.support(37, 1000, recovery)[MARK_INDEX[4, 0, 0]]


def test_any_sampled_geometry_can_continue_the_same_resource_program():
    """The useful invariant: resource feasibility survives arbitrary row choice."""
    rng = np.random.default_rng(1719)
    recovery = Recovery()
    for _ in range(8):
        state, replay, bindings = Resources(), ExactReplayState(), (None,)*4
        now = 0
        for _ in range(150):
            allowed = np.flatnonzero(state.support(now, 100000, recovery))
            if not len(allowed):
                now += 1
                continue
            mark = int(rng.choice(allowed))
            support = row_support(replay, bindings, now, mark, 100000, recovery)
            assert support.any(), (now, MARKS[mark], state, replay)
            choice = int(rng.choice(np.flatnonzero(support)))
            row = CompleteRow(now, tuple(map(int, ACTIONS[choice])))
            state, fresh = state.advance(now, mark, recovery)
            bindings = bind_row(bindings, mark, row.actions, fresh)
            replay = commit(replay, row)
            assert len(state.free_at) + sum(s is not None for s in state.starts) == 4
            now += int(rng.choice([1, 7, 21, 25, 37, 80, 150]))


def test_scoped_partial_control_restores_baseline_and_distinguishes_absence():
    names = ('tech',)
    schedule = ControlSchedule((ControlSpan(0, 1000, stars=3, ln_fraction=.2),
                                ControlSpan(200, 500, stars=5, style={'tech': -1.})), names)
    values = schedule.at([199, 200, 499, 500])
    np.testing.assert_allclose(values[:, :3], [[-.5, -.6, 0], [.5, -.6, -1], [.5, -.6, -1], [-.5, -.6, 0]])
    np.testing.assert_array_equal(values[:, 3:6], [[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 0]])
