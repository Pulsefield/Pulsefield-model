import numpy as np

from ensomi_model.research.oracle_time_continuation.replay import ExactReplayState, commit
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.typed_audio_continuation.program import ACTIONS, MARK_INDEX, Recovery, Resources
from ensomi_model.research.typed_audio_continuation.response_preference import RecoveryPreference


def test_release_clock_and_subset_preferences_preserve_old_hold_opportunity():
    preference = RecoveryPreference()
    state = Resources(starts=(100, 0, None, None), free_at=(0, 0))
    costs = preference.mark_cost(state, 130, 3., 1000)
    assert costs[MARK_INDEX[0, 0, 1]] > costs[MARK_INDEX[0, 0, 2]] == 0
    assert preference.release_clock_cost(state, np.array([130]), 3., Recovery(), 1000)[0] == 0
    young = Resources(starts=(100, None, None, None), free_at=(0, 0, 0))
    times = np.array([[130, 180, 1000]])
    clock = preference.release_clock_cost(young, times, np.array([[3.]]), Recovery(), 1000)
    assert clock[0, 0] > clock[0, 1] == clock[0, 2] == 0
    assert np.all(preference.mark_cost(young, 1000, 3., 1000) == 0)


def test_row_preference_reads_real_recovery_and_stops_reusing_an_old_release():
    preference = RecoveryPreference()
    replay = commit(ExactReplayState(), CompleteRow(0, (2, 0, 0, 0)))
    replay = commit(replay, CompleteRow(100, (3, 0, 0, 0)))
    index = lambda action: int(np.flatnonzero((ACTIONS == action).all(-1))[0])
    costs = preference.row_cost(replay, 130, 3.)
    assert costs[index((1, 0, 0, 0))] > costs[index((0, 1, 0, 0))] == 0
    replay = commit(replay, CompleteRow(130, (1, 0, 0, 0)))
    after_tap = preference.row_cost(replay, 280, 3.)
    assert after_tap[index((1, 0, 0, 0))] == 0
    assert np.all(preference.row_cost(replay, 160, np.nan) == 0)
    assert float(preference.cost(40, 3., 'hh')) > float(preference.cost(40, 5., 'hh'))
