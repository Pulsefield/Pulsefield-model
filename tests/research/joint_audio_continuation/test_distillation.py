from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ensomi_model.research.bounded_typed_continuation.contract import ROW_ACTIONS
from ensomi_model.research.bounded_typed_continuation.data import SourceChart
from ensomi_model.research.joint_audio_continuation.data import JointChart, query
from ensomi_model.research.joint_audio_continuation.distillation import (
    WindowScores, WindowTarget, native_window, next_event_log_mass, score_window,
    teacher_target, window_kl,
)
from ensomi_model.research.joint_audio_continuation.head_spacing import log_acceptance
from ensomi_model.research.joint_audio_continuation.model import JointAudioModel, JointModelConfig
from ensomi_model.research.oracle_time_continuation.data import SourceIdentity
from ensomi_model.research.oracle_time_continuation.storage import ROW_DTYPE


def chart(times=(10, 100, 240), actions=((1, 2, 0, 0), (0, 3, 0, 0), (0, 0, 1, 0)), duration=300):
    rows = np.zeros(len(times), ROW_DTYPE)
    rows['time'], rows['actions'] = times, actions
    identity = SourceIdentity('a' * 64, 'b' * 64, 'train-song', 'train')
    source = SourceChart(identity, rows, minimum_seed_notes=1)
    mel = np.random.default_rng(12).normal(size=((duration + 9) // 10, 128)).astype(np.float32)
    return JointChart(asdict(identity), source, mel, duration)


def config():
    return JointModelConfig(hidden=12, audio_width=8, audio_levels=2, history_levels=2,
                            expansion=2, coupling_rank=3, routing_hidden=20, release_hidden=24)


def test_marked_first_event_distribution_includes_censor_and_changes_hazard():
    scores = WindowScores(torch.zeros(1, 2, dtype=torch.float64),
                          torch.tensor([[[.6, .4], [.6, .4]]], dtype=torch.float64).log())
    valid = torch.ones(1, 2, dtype=torch.bool)
    forced = torch.zeros_like(valid)
    factors = torch.tensor([[[.25, 1.], [.25, 1.]]], dtype=torch.float64).log()
    event, censor = next_event_log_mass(scores, valid, forced, factors)
    expected = torch.tensor([[[.075, .2], [.054375, .145]]], dtype=torch.float64)
    torch.testing.assert_close(event.exp(), expected)
    torch.testing.assert_close(censor.exp(), torch.tensor([.525625], dtype=torch.float64))
    torch.testing.assert_close(event.exp().sum() + censor.exp().sum(), torch.tensor(1., dtype=torch.float64))
    window = SimpleNamespace(inputs=SimpleNamespace(timing_valid=valid, timing_forced=forced))
    target = WindowTarget(expected, censor.exp())
    value = window_kl(scores, window, target)
    student = torch.tensor([.3, .2, .15, .1, .25], dtype=torch.float64)
    teacher = torch.cat((expected.flatten(), censor.exp()))
    torch.testing.assert_close(value[0], (teacher * (teacher / student).log()).sum())


def test_terminal_is_mandatory_and_zero_mass_terms_have_finite_gradients():
    logits = torch.tensor([[-1., 2., 3.]], requires_grad=True)
    row_logits = torch.tensor([[[1., 0.], [1., 0.], [1., 0.]]], requires_grad=True)
    scores = WindowScores(logits, row_logits.log_softmax(-1))
    valid = torch.tensor([[False, True, True]])
    forced = torch.tensor([[False, False, True]])
    factors = torch.tensor([[[.1, 1.], [.1, 1.], [.1, 1.]]]).log()
    window = SimpleNamespace(inputs=SimpleNamespace(timing_valid=valid, timing_forced=forced),
                             log_acceptance=factors)
    target = teacher_target(scores, window)
    assert not target.event.requires_grad and not target.censor.requires_grad
    assert target.censor.item() == 0 and target.event[0, 0].sum() == 0
    torch.testing.assert_close(target.event.sum(), torch.tensor(1.))
    loss = window_kl(scores, window, target).sum()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all() and torch.isfinite(row_logits.grad).all()
    assert logits.grad[0, 0] == 0 and logits.grad[0, -1] == 0
    assert row_logits.grad.abs().sum() > 0


def test_native_inputs_cannot_read_changed_future_or_release_endpoint():
    a = chart()
    b = chart((10, 40, 170), ((1, 2, 0, 0), (0, 3, 1, 0), (0, 0, 0, 1)))
    x = native_window([(a, 10)], config(), horizon_ms=80)
    y = native_window([(b, 10)], config(), horizon_ms=80)
    for name in vars(x.inputs):
        torch.testing.assert_close(getattr(x.inputs, name), getattr(y.inputs, name))
    for name in ('times_ms', 'exact', 'legal', 'log_acceptance'):
        torch.testing.assert_close(getattr(x, name), getattr(y, name))
    assert x.inputs.history_valid.sum() == 1
    assert x.times_ms[x.inputs.timing_valid].tolist() == list(range(11, 91))
    assert not x.inputs.timing_forced.any()
    replay = query(a, 10).replay
    for t in (11, 18, 36, 37, 90):
        index = int((x.times_ms[0] == t).nonzero()[-1])
        for actions in ((1, 0, 0, 0), (0, 3, 1, 0), (0, 0, 2, 0)):
            actual = x.log_acceptance[0, index, ROW_ACTIONS.index(actions)]
            assert float(actual) == pytest.approx(log_acceptance(replay, t, actions, 27), abs=2e-6)
    # Close is independent of the head-age factor, including a very young LN.
    assert x.log_acceptance[0, :, ROW_ACTIONS.index((0, 3, 0, 0))].eq(0).all()


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_native_kl_backpropagates_through_joint_audio_history_and_time(device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS hardware unavailable')
    torch.manual_seed(35)
    model = JointAudioModel(config()).to(device)
    window = native_window([(chart(), 10), (chart(), 96)], model.config,
                           horizon_ms=10, device=device)
    scores = score_window(model, window)
    target = teacher_target(scores, window)
    torch.testing.assert_close(target.event.sum((-2, -1)) + target.censor,
                               torch.ones(2, device=device), atol=2e-6, rtol=2e-6)
    loss = window_kl(scores, window, target).mean()
    assert loss.item() > 0
    loss.backward()
    for module in (model.audio_input, model.temporal, model.exact, model.timing,
                   model.joint, model.route_residual, model.release_residual):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads) > 0


def test_unmodified_teacher_is_zero_kl_and_native_terminal_support_matches_replay():
    source = chart((10, 150, 300), ((0, 2, 0, 0), (1, 0, 0, 0), (0, 3, 0, 0)))
    model = JointAudioModel(config())
    window = native_window([(source, 290)], model.config, horizon_ms=80, scale_ms=0.)
    scores = score_window(model, window)
    target = teacher_target(scores, window)
    torch.testing.assert_close(window_kl(scores, window, target), torch.zeros(1), atol=2e-6, rtol=0.)
    forced = window.inputs.timing_forced[0]
    assert forced.sum() == 1 and window.times_ms[0, forced].item() == 300
    assert target.censor.item() == 0
    supported = window.legal[0, forced].squeeze(0).nonzero().flatten().tolist()
    assert all(ROW_ACTIONS[i][1] == 3 and 2 not in ROW_ACTIONS[i] for i in supported)
