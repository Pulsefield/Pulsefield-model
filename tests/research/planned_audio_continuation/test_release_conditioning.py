from dataclasses import replace
import math

import pytest
import torch
from torch.nn import functional as F

from ensomi_model.research.joint_audio_continuation.intervals import IntervalExample
from ensomi_model.research.planned_audio_continuation.intervals import collate_interval, interval_losses, score_interval
from ensomi_model.research.planned_audio_continuation.model import PlannedAudioModel
from ensomi_model.research.planned_audio_continuation.release import conditioned_release_logits
from .test_distribution import chart, config, source


def event_log_probs(logits):
    survival = torch.cat((logits.new_zeros(1), F.softplus(logits[:-1]).cumsum(0)))
    return torch.cat((F.logsigmoid(logits[:-1]), logits.new_zeros(1))) - survival


@pytest.mark.parametrize('values', [[-1.], [-4., -2., 1., -7.], [-1000.] * 309,
                                  [1000., -1000., 500., -800.], [-45., -41., -40., -39., -15.]])
def test_conditional_probabilities_and_gradients_match_normalized_first_event(values):
    raw = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    log_q = F.logsigmoid(raw) - torch.cat((raw.new_zeros(1), F.softplus(raw[:-1]).cumsum(0)))
    expected = log_q.log_softmax(0)
    actual = event_log_probs(conditioned_release_logits(raw))
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-12)
    for event in sorted({0, len(raw) // 2, len(raw) - 1}):
        a = torch.autograd.grad(actual[event], raw, retain_graph=True)[0]
        b = torch.autograd.grad(expected[event], raw, retain_graph=True)[0]
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=2e-10, rtol=2e-10)
    if len(raw) > 1 and max(values) < 10:
        # Even though the final conditional hazard is certain, its raw value
        # changes normalization of every earlier event and must receive signal.
        assert abs(float(torch.autograd.grad(actual[0], raw)[0][-1])) > 0
    for start in range(1, min(len(raw), 8)):
        torch.testing.assert_close(conditioned_release_logits(raw[start:]),
                                   conditioned_release_logits(raw)[start:], atol=2e-10, rtol=2e-12)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS unavailable')
@pytest.mark.parametrize('mixed', [False, True])
def test_conditional_rare_event_probabilities_and_gradients_on_mps(mixed):
    values = torch.linspace(-1000, -990, 127)
    if mixed:
        values = torch.cat((values, torch.tensor([-42., -40., -38., -7., 4., 80.])))
    result = []
    for device in ('cpu', 'mps'):
        raw = values.to(device).requires_grad_()
        scores = event_log_probs(conditioned_release_logits(raw))
        gradient = torch.autograd.grad(scores[17] + scores[-1], raw)[0]
        assert torch.isfinite(scores).all() and torch.isfinite(gradient).all()
        result.append((scores.detach().cpu(), gradient.cpu()))
    for a, b in zip(*result):
        torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-4)


def test_training_uses_normalized_event_law_and_leaves_terminal_closure_unchanged():
    model = PlannedAudioModel(replace(config(), condition_full_holds=True))
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    with torch.no_grad():
        model.timing[-1].bias.fill_(math.log(.2 / .8))
        model.release_clock[-1].bias.fill_(math.log(.3 / .7))
    c = source()
    batch = collate_interval(IntervalExample(c, 0, 20), model.config)
    scores = score_interval(model, batch.inputs, model.encode_coarse(torch.from_numpy(c.mel)[None]))
    h, r, row, total = interval_losses(scores, batch)
    assert len(batch.inputs.release_waits) == 1
    expected_r = -math.log(.3 * .7 ** 3 / (1 - .7 ** 4)) - 4 * math.log(.7)
    torch.testing.assert_close(r, r.new_tensor(expected_r))
    total.backward()
    assert torch.isfinite(model.release_clock[-1].bias.grad).all()
    assert model.release_clock[-1].bias.grad[4] != 0


def held_chart(release=1501, lane=0):
    close = [0] * 4
    close[lane] = 3
    tap = [0] * 4
    tap[lane] = 1
    tail = [3] * 4
    tail[lane] = 0
    return chart([0, release, 1800, 1900], [(2, 2, 2, 2), close, tap, tail], 2000)


def test_normalizer_sees_full_hypothetical_wait_without_actual_future_releases():
    torch.manual_seed(33)
    model = PlannedAudioModel(replace(config(), condition_full_holds=True)).eval()
    a, b = held_chart(), held_chart(1201, 1)
    coarse = model.encode_coarse(torch.from_numpy(a.mel)[None])
    batches = [collate_interval(IntervalExample(c, 0, 100), model.config) for c in (a, b)]
    assert all(len(batch.inputs.release_waits) == 1 for batch in batches)
    wait = batches[0].inputs.release_waits[0]
    assert int(wait.times.max()) == 1799
    assert len(wait.native_indices) == 1799
    assert batches[0].inputs.base.mel_valid.sum() > 150
    scores = [score_interval(model, batch.inputs, coarse) for batch in batches]
    for x, y in zip(vars(scores[0]).values(), vars(scores[1]).values()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_conditional_interval_censoring_preserves_probability_and_gradient():
    torch.manual_seed(81)
    c = held_chart()
    model = PlannedAudioModel(replace(config(), condition_full_holds=True)).eval()

    def score(width):
        coarse = model.encode_coarse(torch.from_numpy(c.mel)[None])
        total = None
        for index in range(IntervalExample(c, 0, width).count):
            batch = collate_interval(IntervalExample(c, index, width), model.config)
            loss = torch.stack(interval_losses(score_interval(model, batch.inputs, coarse), batch))
            total = loss if total is None else total + loss
        gradient = torch.autograd.grad(total[-1], model.release_clock[-1].bias)[0]
        return total.detach(), gradient

    expected, expected_grad = score(2001)
    actual, actual_grad = score(137)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-6)
    torch.testing.assert_close(actual_grad, expected_grad, atol=3e-5, rtol=3e-6)


def test_no_conditional_wait_for_partial_occupancy_or_absent_future_head():
    for c in (chart([0, 100, 200], [(2, 2, 2, 0), (0, 0, 0, 1), (3, 3, 3, 0)], 300),
              chart([0, 10, 200], [(1, 0, 0, 0), (2, 2, 2, 2), (3, 3, 3, 3)], 300)):
        torch.manual_seed(17)
        model = PlannedAudioModel(config())
        coarse = model.encode_coarse(torch.from_numpy(c.mel)[None])
        scores = []
        for enabled in (False, True):
            model.config = replace(model.config, condition_full_holds=enabled)
            batch = collate_interval(IntervalExample(c, 0, 400), model.config)
            assert not batch.inputs.release_waits
            scores.append(score_interval(model, batch.inputs, coarse))
        for x, y in zip(vars(scores[0]).values(), vars(scores[1]).values()):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
