from dataclasses import replace

import numpy as np
import pytest
import torch

from ensomi_model.research.bounded_typed_continuation.contract import ROW_ACTIONS
from ensomi_model.research.oracle_time_continuation.replay import ExactReplayState, commit, legal_rows
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.planned_audio_continuation import buffering, generation
from ensomi_model.research.planned_audio_continuation.features import HeadPreview
from ensomi_model.research.planned_audio_continuation.model import PlannedAudioModel
from ensomi_model.research.planned_audio_continuation.row_constraints import (
    NoRowContinuation, allowed_rows, condition_rows,
)
from ensomi_model.research.planned_audio_continuation.session import ContinuationSession
from ensomi_model.research.scoped_style_modeling.dataset import ContractError
from .test_distribution import config


def replay(rows):
    state = ExactReplayState()
    for t, actions in rows:
        state = commit(state, CompleteRow(t, actions))
    return state


def allows(state, now, actions, future=(), mode='preview', complete=True):
    return bool(allowed_rows(state, now, HeadPreview(future, complete), mode)[ROW_ACTIONS.index(actions)])


def test_strict_head_and_inclusive_release_boundaries_do_not_limit_hold_duration():
    state = replay([(0, (1, 2, 0, 0)), (10, (0, 3, 0, 0))])
    assert not allows(state, 19, (1, 0, 0, 0))
    assert allows(state, 20, (1, 0, 0, 0))
    assert not allows(state, 30, (0, 2, 0, 0))
    assert allows(state, 31, (0, 2, 0, 0))
    # A one-ms LN tail is legal when another column can carry the nearby H.
    held = replay([(0, (2, 0, 0, 0))])
    assert allows(held, 1, (3, 0, 0, 0), (2,))
    assert allows(held, 1, (3, 0, 0, 0))


def test_preview_reserves_columns_without_forbidding_close_heads_or_full_ln_chords():
    state = ExactReplayState()
    assert allows(state, 100, (1, 1, 1, 1), (105,), mode='current')
    assert not allows(state, 100, (1, 1, 1, 1), (105,))
    assert allows(state, 100, (1, 1, 0, 0), (101, 102))
    assert not allows(state, 100, (1, 1, 1, 0), (101, 102))
    assert allows(state, 100, (1, 1, 1, 1), (120,))
    assert not allows(state, 100, (2, 2, 2, 2), (120,))
    assert allows(state, 100, (2, 2, 2, 2), (121,))
    assert allows(state, 100, (1, 0, 0, 0), (101, 102, 103, 120))
    assert not allows(state, 100, (1, 0, 0, 0), (101, 102, 103, 119))
    assert allows(state, 100, (1, 1, 1, 1), (120,), complete=False)
    with pytest.raises(ContractError, match='full 20-ms'):
        allows(state, 100, (1, 0, 0, 0), (105,), complete=False)
    assert not allows(state, 100, (1, 0, 0, 0), (101, 102, 103, 104, 105), complete=False)


def test_old_hold_and_current_release_cannot_supply_a_five_ms_joint_clean_head():
    state = replay([(0, (0, 0, 0, 2))])
    for row in ((1, 1, 1, 0), (1, 1, 1, 3)):
        assert allows(state, 100, row, (105,), mode='current')
        assert not allows(state, 100, row, (105,))
    assert allows(state, 100, (1, 1, 0, 0), (105,))
    assert allows(state, 100, (1, 1, 0, 3), (105,))


def exhaustive_taps(state, times):
    if not times:
        return True
    for lane in range(4):
        if state.occupancy[lane]:
            continue
        row = CompleteRow(times[0], tuple(int(i == lane) for i in range(4)))
        if not buffering.close_pairs(state, [row], times[0], screen_release_heads=True):
            if exhaustive_taps(commit(state, row), times[1:]):
                return True
    return False


def test_nested_capacity_matches_exhaustive_ordered_assignments_and_column_permutations():
    histories = [[], [(15, (1, 1, 0, 0))], [(0, (2, 2, 2, 2)), (12, (0, 3, 0, 3))],
                 [(0, (1, 0, 0, 0)), (10, (0, 2, 0, 0)), (19, (0, 3, 1, 0))]]
    rng = np.random.default_rng(912)
    permutation = (2, 0, 3, 1)
    for history in histories:
        state = replay(history)
        permuted = replay([(t, tuple(a[i] for i in permutation)) for t, a in history])
        for n in range(6):
            times = tuple(sorted(int(t) for t in rng.choice(np.arange(21, 41), n, replace=False)))
            observed = allowed_rows(state, 20, HeadPreview(times, True), 'preview')
            reordered = allowed_rows(permuted, 20, HeadPreview(times, True), 'preview')
            for actions in legal_rows(state):
                row = CompleteRow(20, actions)
                immediate = not buffering.close_pairs(state, [row], 20, screen_release_heads=True)
                expected = immediate and exhaustive_taps(commit(state, row), times)
                assert bool(observed[ROW_ACTIONS.index(actions)]) == expected
                other = tuple(actions[i] for i in permutation)
                assert observed[ROW_ACTIONS.index(actions)] == reordered[ROW_ACTIONS.index(other)]


def test_mask_conditions_the_complete_law_and_reports_empty_mass_without_sampling():
    state = replay([(0, (1, 0, 0, 0))])
    preview = HeadPreview((10,), True)
    torch.manual_seed(17)
    logp = torch.randn(256, dtype=torch.float64).log_softmax(-1)
    allowed = allowed_rows(state, 5, preview, 'preview')
    filtered, record = condition_rows(logp, state, 5, preview, 'preview')
    reference = logp.exp() * torch.from_numpy(allowed)
    torch.testing.assert_close(filtered.softmax(-1), reference/reference.sum())
    assert record['retained_probability'] == pytest.approx(float(reference.sum()))
    unchanged, decision = condition_rows(logp, ExactReplayState(), 100, HeadPreview((), True), 'current')
    assert unchanged is logp and decision is None
    full = replay([(0, (1, 1, 1, 1))])
    head_rows = torch.tensor([any(a in (1, 2) for a in r) for r in ROW_ACTIONS])
    with pytest.raises(NoRowContinuation) as error:
        condition_rows(logp.masked_fill(~head_rows, -torch.inf), full, 5, HeadPreview((), True), 'current')
    assert error.value.decision['retained_probability'] == 0
    assert error.value.decision['allowed_rows'] == 0
    # A nonempty conditional must remain sampleable even if its base mass
    # underflows; that is different from mechanically empty support.
    tiny = torch.full((256,), -torch.inf, dtype=torch.float64)
    tiny[ROW_ACTIONS.index((1, 0, 0, 0))] = 0.
    tiny[ROW_ACTIONS.index((0, 1, 0, 0))] = -1000.
    conditional, record = condition_rows(tiny, state, 5, HeadPreview((), True), 'current')
    assert conditional.exp().sum() == 1 and record['log_retained_probability'] == -1000.
    assert int(torch.multinomial(conditional.exp(), 1)) == ROW_ACTIONS.index((0, 1, 0, 0))


def quad_model(monkeypatch, single_logit=-40.):
    model = PlannedAudioModel(replace(config(), lookahead=5))

    def scores(audio, history, exact, legal, *args, **kwargs):
        weights = torch.full((1, 256), -torch.inf)
        weights[0, ROW_ACTIONS.index((1, 1, 1, 1))] = 0.
        for lane in range(4):
            weights[0, ROW_ACTIONS.index(tuple(int(i == lane) for i in range(4)))] = single_logit
        return weights.masked_fill(~legal, -torch.inf).log_softmax(-1)

    monkeypatch.setattr(model, 'planned_row_log_probs', scores)
    return model


def test_native_current_empty_support_preserves_observed_coverage_and_rng(monkeypatch):
    model = quad_model(monkeypatch)
    session = ContinuationSession(model, np.zeros((9, 128), np.float32), 80, seed=17,
        planner_factory=generation.HeadPlanner, head_times=(0, 5, 40), row_constraint='current')
    session.step()
    assert session.rows[0].actions == (1, 1, 1, 1)
    rng = session.row_rng.get_state().clone()
    cache = session.row_cache
    fork = session.fork()
    with pytest.raises(NoRowContinuation):
        fork.step()
    assert torch.equal(fork.row_rng.get_state(), rng)
    assert fork.row_cache is cache and fork.rows == session.rows
    assert fork.cursor == 0 and fork.planner.queue[0] == 5
    assert len(fork.constraint_decisions) == 1 and not session.constraint_decisions
    updates = []
    result = generation.rollout(model, np.zeros((9, 128), np.float32), 80, seed=17,
        head_times=(0, 5, 40), row_constraint='current', on_update=updates.append)
    assert result.stop_reason == 'row_constraint_empty' and not result.completed
    assert result.coverage_ms == 0 and len(result.rows) == 1
    assert max(u.coverage_ms for u in updates) == 0


def test_preview_native_draws_once_per_row_and_buffered_current_rejections_never_publish(monkeypatch):
    model = quad_model(monkeypatch)
    mel = np.zeros((9, 128), np.float32)
    plain = generation.rollout(model, mel, 80, seed=17, head_times=(0, 5, 40))
    explicit = generation.rollout(model, mel, 80, seed=17, head_times=(0, 5, 40), row_constraint='none')
    assert plain.rows == explicit.rows
    original, calls = torch.multinomial, []

    def draw(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(torch, 'multinomial', draw)
        result = generation.rollout(model, mel, 80, seed=17, head_times=(0, 5, 40), row_constraint='preview')
    assert result.completed and len(result.rows) == len(calls) == 3
    assert [r.time_ms for r in result.rows] == [0, 5, 40]
    assert not buffering.close_pairs(ExactReplayState(), result.rows, 80, screen_release_heads=True)
    events, rejected = [], []
    capped = buffering.rollout_buffered(model, mel, 80, seed=17, head_times=(0, 5, 40),
        row_constraint='current', window_ms=1, max_attempts=2, on_update=events.append,
        on_rejected=lambda meta, fork: rejected.append((meta, tuple(fork.rows))))
    assert capped.stop_reason == 'planning_attempt_limit' and capped.coverage_ms == -1
    assert not events and not capped.rows and len(rejected) == 2
    assert all(m['constraint_failure']['time_ms'] == 5 for m, _ in rejected)
    safe = buffering.rollout_buffered(model, mel, 80, seed=17, head_times=(0, 5, 40),
        row_constraint='preview', window_ms=1, on_update=events.append)
    assert safe.completed and safe.rows == result.rows
    assert [u.row for u in events if u.row is not None] == list(safe.rows)


def test_constraint_options_are_explicit_and_terminal_release_has_no_duration_floor():
    model = PlannedAudioModel(config())
    mel = np.zeros((5, 128), np.float32)
    for args in ({'row_constraint': 'unknown'}, {'row_constraint': 'preview'},
                 {'row_constraint': 'current', 'correct_short_attacks': True}):
        with pytest.raises(ContractError):
            generation.rollout(model, mel, 40, seed=17, **args)
    with pytest.raises(ContractError):
        buffering.rollout_buffered(model, mel, 40, seed=17, row_constraint='current', screen_release_heads=False)
    state = replay([(0, (2, 2, 2, 2))])
    assert allows(state, 1, (3, 3, 3, 3))


def test_buffered_empty_support_can_retry_and_publish_a_different_valid_prefix(monkeypatch):
    model = quad_model(monkeypatch, single_logit=0.)
    events, rejected = [], []
    # This fixed seed draws the quad first, then a single after the first retry.
    result = buffering.rollout_buffered(model, np.zeros((9, 128), np.float32), 80,
        seed=1, head_times=(0, 5, 40), row_constraint='current', window_ms=1,
        on_update=events.append, on_rejected=lambda m, s: rejected.append((m, tuple(s.rows))))
    assert result.completed and len(rejected) == 1
    assert rejected[0][1] == (CompleteRow(0, (1, 1, 1, 1)),)
    assert rejected[0][0]['constraint_failure']['time_ms'] == 5
    assert result.rows[0].actions != (1, 1, 1, 1)
    assert [e.row for e in events if e.row is not None] == list(result.rows)
    assert not buffering.close_pairs(ExactReplayState(), result.rows, 80, screen_release_heads=True)
