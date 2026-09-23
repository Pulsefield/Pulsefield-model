import math

import numpy as np
import pytest
import torch

from ensomi_model.research.bounded_typed_continuation.contract import ROW_ACTIONS
from ensomi_model.research.joint_audio_continuation.generation import rollout
from ensomi_model.research.joint_audio_continuation.head_spacing import log_acceptance, sample_row
from ensomi_model.research.oracle_time_continuation.replay import ExactReplayState, commit
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from .test_generation import EveryMillisecond, HoldThenSilence, config
from .test_model import small_model


def after(*rows):
    state = ExactReplayState()
    for time, actions in rows:
        state = commit(state, CompleteRow(time, actions))
    return state


def test_head_age_factor_preserves_fast_distinct_keys_and_ln_tail_reset():
    state = after((0, (2, 1, 0, 0)), (100, (3, 0, 0, 0)))
    assert log_acceptance(state, 101, (1, 0, 0, 0), 27) == 0
    assert log_acceptance(state, 101, (0, 0, 2, 1), 27) == 0
    recent = after((100, (1, 1, 0, 0)))
    assert log_acceptance(recent, 101, (0, 0, 1, 0), 27) == 0
    assert log_acceptance(recent, 101, (1, 1, 0, 0), 27) == pytest.approx(8 * math.log(1/27))
    assert math.isfinite(log_acceptance(recent, 101, (1, 0, 0, 0), 27))
    assert log_acceptance(recent, 127, (1, 1, 1, 1), 27) == 0
    held = after((100, (2, 2, 0, 0)))
    assert log_acceptance(held, 101, (3, 3, 0, 0), 27) == 0


def test_marked_thinning_accounts_for_waiting_and_different_lane_actions():
    state = after((0, (1, 0, 0, 0)))
    repeated, other = (1, 0, 0, 0), (0, 1, 0, 0)
    acceptance = math.exp(log_acceptance(state, 5, repeated, 10))
    assert acceptance == pytest.approx(1/16)
    # A rejected repeat becomes no accepted event at this clock. It does not
    # merely transfer its entire mass to the other lane at the same time.
    hazard, q_repeat, q_other = .8, .25, .75
    event_masses = [hazard*q_repeat*acceptance, hazard*q_other]
    no_event = (1-hazard) + hazard*q_repeat*(1-acceptance)
    assert sum(event_masses) + no_event == pytest.approx(1.)
    assert no_event == pytest.approx(.3875)
    assert event_masses[1] == pytest.approx(.6)


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='requires Apple MPS'))])
def test_zero_scale_preserves_original_row_rng_and_probability_path(device):
    state = after((0, (1, 0, 0, 0)))
    probabilities = torch.linspace(.001, 1., 256, device=device)
    log_probs = probabilities.log_softmax(0)
    baseline = torch.Generator().manual_seed(23)
    current = torch.Generator().manual_seed(23)
    acceptance = torch.Generator().manual_seed(99)
    before = acceptance.get_state()
    for _ in range(10):
        expected = ROW_ACTIONS[int(torch.multinomial(log_probs.exp().cpu(), 1, generator=baseline))]
        actions, accepted, factor = sample_row(log_probs, state, 1, current, acceptance, 0)
        assert actions == expected and accepted and factor == 0
    assert torch.equal(baseline.get_state(), current.get_state())
    assert torch.equal(acceptance.get_state(), before)


def test_forced_terminal_closure_is_reweighted_and_never_rejected():
    state = after((0, (2, 1, 0, 0)))
    close, close_tap = (3, 0, 0, 0), (3, 1, 0, 0)
    logs = torch.full((256,), -torch.inf)
    logs[ROW_ACTIONS.index(close)] = math.log(.1)
    logs[ROW_ACTIONS.index(close_tap)] = math.log(.9)
    action, accepted, _ = sample_row(logs, state, 1, torch.Generator().manual_seed(17),
        torch.Generator().manual_seed(90), 27, forced_terminal=True)
    assert accepted and action == close
    assert commit(state, CompleteRow(1, action), is_terminal=True).is_complete
    result = rollout(HoldThenSilence(config()), np.zeros((13, 128), np.float32), 123,
                     chunk_ms=7, head_spacing_ms=10000)
    assert result.completed and result.rows[-1] == CompleteRow(123, (3, 0, 0, 0))
    assert not result.metrics['rejected_rows']


def test_rejections_advance_time_without_entering_exact_or_learned_history(monkeypatch):
    model = EveryMillisecond(config())
    appended = []
    original_append = model.temporal.append

    def append(cache, raw):
        appended.append(raw.detach().clone())
        return original_append(cache, raw)

    monkeypatch.setattr(model.temporal, 'append', append)
    result = rollout(model, np.zeros((11, 128), np.float32), 100,
                     seed=17, chunk_ms=7, head_spacing_ms=27)
    rejected = result.metrics['rejected_rows']
    assert result.completed and rejected and len(result.rows) < 101
    assert result.metrics['proposed_rows'] == 101
    assert len(appended) == len(result.rows)
    for proposal in rejected:
        prefix = [row for row in result.rows if row.time_ms < proposal['time_ms']]
        assert len(prefix) == proposal['committed_rows']
        assert proposal['last_lane_head_ms'][0] == prefix[-1].time_ms
    with_cap = rollout(EveryMillisecond(config()), np.zeros((11, 128), np.float32), 100,
                       seed=17, max_rows=10, head_spacing_ms=10000)
    assert not with_cap.completed and with_cap.stop_reason == 'proposal_limit'
    assert with_cap.metrics['proposed_rows'] == 10 and len(with_cap.rows) == 1


def test_marked_thinning_preserves_draws_across_scheduler_partitions():
    model = small_model()
    mel = np.random.default_rng(3).normal(size=(101, 128)).astype(np.float32)
    a = rollout(model, mel, 1000, seed=47, chunk_ms=13, head_spacing_ms=270)
    b = rollout(model, mel, 1000, seed=47, chunk_ms=500, head_spacing_ms=270)
    assert a.rows == b.rows
    assert a.metrics['rejected_rows'] == b.metrics['rejected_rows']
