from functools import lru_cache
from itertools import combinations

import numpy as np
import pytest

from ensomi_model.research.bounded_typed_continuation.contract import ROW_ACTIONS
from ensomi_model.research.oracle_time_continuation.replay import ExactReplayState, commit, legal_rows
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.planned_audio_continuation.features import HeadPreview, LNProjection
from ensomi_model.research.planned_audio_continuation.spacing import (
    allowed_rows, check_head_capacity, next_head_earliest, release_limits,
)
from ensomi_model.research.scoped_style_modeling.dataset import ContractError
from ensomi_model.research.typed_audio_continuation.program import Recovery


def replay(rows):
    state = ExactReplayState()
    for time, actions in rows:
        state = commit(state, CompleteRow(time, actions))
    return state


def allowed(state, now, actions, heads=(), end=100, gap=21, complete=True):
    return bool(allowed_rows(state, now, HeadPreview(tuple(heads), complete), end, gap,
                             actions=[actions])[0])


def test_inclusive_twenty_ms_relations_and_ln_terminal_boundary():
    state = replay([(0, (1, 2, 0, 0))])
    assert not allowed(state, 20, (1, 0, 0, 0))
    assert allowed(state, 21, (1, 0, 0, 0))
    assert not allowed(state, 20, (0, 3, 0, 0))
    assert allowed(state, 21, (0, 3, 0, 0))
    state = commit(state, CompleteRow(21, (0, 3, 0, 0)))
    assert not allowed(state, 41, (0, 2, 0, 0))
    assert allowed(state, 42, (0, 2, 0, 0))
    assert not allowed(ExactReplayState(), 80, (2, 0, 0, 0), end=100)
    assert allowed(ExactReplayState(), 79, (2, 0, 0, 0), end=100)
    assert allowed(ExactReplayState(), 100, (1, 0, 0, 0), end=100)


def test_four_column_capacity_keeps_rapid_cross_column_heads():
    assert next_head_earliest([], 21) == 0
    assert next_head_earliest([1, 2, 3, 4], 21) == 22
    check_head_capacity([1, 2, 3, 4, 22, 23, 24, 25], 21)
    with pytest.raises(ContractError, match='four-column'):
        check_head_capacity([1, 2, 3, 4, 21], 21)
    assert allowed(ExactReplayState(), 0, (1, 0, 0, 0), (1, 2, 3, 21, 22, 23, 24))
    assert not allowed(ExactReplayState(), 0, (1, 1, 0, 0), (1, 2, 3))


def test_new_hold_needs_two_gaps_before_repress_and_old_hold_needs_a_release_clock():
    state = ExactReplayState()
    assert not allowed(state, 0, (2, 2, 2, 2), (41,))
    assert allowed(state, 0, (2, 2, 2, 2), (42,))
    assert allowed(state, 0, (2, 2, 2, 0), (1, 42))
    assert not allowed(state, 0, (2, 2, 2, 0), (1, 20))
    old = replay([(0, (2, 2, 2, 2))])
    assert not allowed(old, 50, (0, 0, 0, 0), (71,))
    assert allowed(old, 50, (0, 0, 0, 0), (72,))
    assert allowed(old, 50, (3, 0, 0, 0), (71,))


def test_release_deadline_counts_future_head_cluster_and_reads_only_ln_projection():
    p = HeadPreview((100, 105, 110), True)
    two = LNProjection((0, 50, None, None), 60)
    assert release_limits(two, p, 200, 21) == (61, 89)
    assert release_limits(LNProjection((50, None, 0, None), 60), p, 200, 21) == (61, 89)
    assert release_limits(LNProjection((0, 0, 0, 0), 60), p, 200, 21) == (61, 79)
    assert release_limits(LNProjection((None,)*4, 60), p, 200, 21) == (None, None)
    assert release_limits(LNProjection((59,)*4, 60), HeadPreview((), True), 90, 21) == (80, 90)
    with pytest.raises(ContractError, match='LN-only'):
        release_limits(ExactReplayState(), p, 200, 21)


def oracle(state, now, actions, heads, end, gap):
    """Independent native-clock search over every eligible release subset and TAP.

    No availability-time or cluster-deadline formula is used. Future LNs cannot
    improve existence compared with TAPs. Memoization bounds the exhaustive
    search on the small test grids; release timing is still explicitly chosen.
    """
    starts = tuple(state.open_ln_start_ms)
    hh, rh, hr = (gap, gap, gap) if isinstance(gap, int) else (gap.hh, gap.rh, gap.hr)
    attacks = tuple(-10000 if t is None else int(t) for t in state.last_lane_attack_ms)
    releases = tuple(-10000 if t is None else int(t) for t in state.last_lane_release_ms)
    for lane, action in enumerate(actions):
        if action in (1, 2) and (now-attacks[lane] < hh or now-releases[lane] < rh):
            return False
        if action == 3 and now-starts[lane] < hr:
            return False
    starts = tuple(None if a == 3 else now if a == 2 else s for a, s in zip(actions, starts))
    attacks = tuple(now if a in (1, 2) else t for a, t in zip(actions, attacks))
    releases = tuple(now if a == 3 else t for a, t in zip(actions, releases))
    head_set = set(heads)

    @lru_cache(None)
    def visit(time, held, attack, release):
        if time > end:
            return not any(s is not None for s in held)
        eligible = [i for i,s in enumerate(held) if s is not None and time-s >= hr]
        press = ([i for i,s in enumerate(held) if s is None and
                  time-attack[i] >= hh and time-release[i] >= rh] if time in head_set else [None])
        for n in range(len(eligible)+1):
            for subset in combinations(eligible, n):
                for lane in press:
                    following = tuple(None if i in subset else s for i,s in enumerate(held))
                    a = tuple(time if i == lane else value for i,value in enumerate(attack))
                    r = tuple(time if i in subset else value for i,value in enumerate(release))
                    if visit(time+1, following, a, r):
                        return True
        return False

    return visit(now+1, starts, attacks, releases)


def test_finite_horizon_matches_exhaustive_release_and_head_execution():
    histories = [[], [(1, (2, 2, 2, 2))], [(1, (2, 0, 2, 0)), (8, (0, 1, 0, 2))],
                 [(1, (2, 2, 0, 0)), (8, (3, 0, 1, 0)), (9, (0, 0, 0, 2))]]
    rng = np.random.default_rng(250925)
    now, end, gap = 10, 19, 3
    for history in histories:
        state = replay(history)
        choices = legal_rows(state)
        for _ in range(12):
            heads = tuple(sorted(map(int,rng.choice(np.arange(now+1,end+1),rng.integers(0,8),replace=False))))
            check_head_capacity(heads, gap)
            # The analytic side sees only through its declared finite horizon.
            visible = tuple(t for t in heads if t <= now+2*gap)
            following = next((t for t in heads if t > now+2*gap), None)
            preview = HeadPreview(visible+(() if following is None else (following,)), following is None)
            actual = allowed_rows(state, now, preview, end, gap)
            for actions in choices:
                assert actual[ROW_ACTIONS.index(actions)] == oracle(state,now,actions,heads,end,gap)


def test_wait_deadline_preserves_an_available_next_event():
    gap, end = 3, 22
    histories = [[(1, (2, 2, 2, 2))], [(1, (2, 2, 0, 0))],
                 [(1, (2, 0, 0, 0)), (9, (0, 1, 1, 0))], [(8, (2, 2, 2, 2))]]
    plans = [(14,15,16), (12,13,14,15,18), (15,17,18,19), (22,), ()]
    for history in histories:
        state = replay(history); now = 10
        for heads in plans:
            preview = HeadPreview(heads, True)
            if not allowed(state,now,(0,0,0,0),heads,end,gap):
                continue
            earliest, deadline = release_limits(LNProjection(state.open_ln_start_ms,now),preview,end,gap)
            next_h = heads[0] if heads else None
            event = min(deadline,next_h) if next_h is not None else deadline
            for silent in range(now+1,event):
                assert allowed(state,silent,(0,0,0,0),heads,end,gap)
            future = tuple(t for t in heads if t > event)
            candidates = [a for a in legal_rows(state) if any(v in (1,2) for v in a)==(event==next_h)]
            assert any(allowed(state,event,a,future,end,gap) for a in candidates)
            assert earliest <= event or event == next_h


def test_distinct_recovery_intervals_match_actual_release_and_repress_search():
    profile = Recovery(3, 2, 2)
    states = [ExactReplayState(), replay([(1, (2, 2, 2, 2))]),
              replay([(1, (2, 2, 0, 0)), (8, (3, 0, 1, 0)), (9, (0, 0, 0, 2))])]
    for state in states:
        for heads in ((11, 12, 14, 15, 17), (12, 13, 14, 15), (13, 16, 18), ()):
            preview = HeadPreview(heads, True)
            actual = allowed_rows(state, 10, preview, 19, profile)
            for actions in legal_rows(state):
                assert actual[ROW_ACTIONS.index(actions)] == oracle(state, 10, actions, heads, 19, profile)


def test_incomplete_preview_cannot_hide_future_capacity():
    with pytest.raises(ContractError, match='recovery horizon'):
        allowed(ExactReplayState(),0,(1,0,0,0),(22,),complete=False)
    assert allowed(ExactReplayState(),0,(1,0,0,0),(42,),complete=False)
