from dataclasses import replace

import torch

from ensomi_model.research.controlled_audio_continuation.model import ControlledAudioModel
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule, ControlSpan
from ensomi_model.research.joint_audio_continuation.intervals import IntervalExample
from ensomi_model.research.planned_audio_continuation.intervals import collate_interval
from planned_audio_continuation.test_distribution import chart, config


def setup():
    net = ControlledAudioModel(replace(config(), lookahead=16, bounded_head=True,
        condition_full_holds=True, minimum_action_gap_ms=60), style_names=('tech',), count_history_bound=1.)
    source = chart([0, 100, 250, 400], [(1, 0, 0, 0), (0, 2, 0, 0), (0, 0, 0, 1), (0, 3, 0, 0)], 500)
    batch = collate_interval(IntervalExample(source, 0, 501), net.config, recovery=net.recovery)
    schedule = ControlSchedule((ControlSpan(0, 501, stars=3, ln_fraction=.2),), net.style_names)
    control = torch.tensor(schedule.at([0], encoding=net.control_encoding))
    return net, batch.inputs, control


def test_count_history_correction_has_bounded_odds_and_no_logit_offset_meaning():
    torch.manual_seed(74)
    net, _, _ = setup()
    comp = net.composition
    actual = (10*torch.randn(2, len(comp.longs))).requires_grad_()
    raw = torch.randn_like(actual, requires_grad=True)
    active = torch.ones_like(actual, dtype=torch.bool)
    active[0, comp.group == 0] = False
    prior = torch.randn(2, len(comp.group_members), requires_grad=True)
    shift = torch.tensor([-.5, .8])
    result = comp.count_log_probs(raw, actual, active, shift, prior=prior, history_bound=1.)
    repeated = comp.count_log_probs(raw, actual+27., active, shift, prior=prior-11., history_bound=1.)
    torch.testing.assert_close(result, repeated, atol=3e-6, rtol=1e-6)
    members = active[:, None] & comp.group_members[None]
    mass = result[:, None].masked_fill(~members, -torch.inf).logsumexp(-1)
    for i in range(2):
        valid = members[i].any(-1)
        delta = mass[i, valid]-prior[i, valid].log_softmax(-1)
        assert float((delta.max()-delta.min()).detach()) <= 2.00001
        torch.testing.assert_close(result[i].exp().sum(), torch.tensor(1.))
    result[1, active[1]].mean().neg().backward()
    assert torch.isfinite(actual.grad).all() and torch.isfinite(prior.grad).all()


def test_prior_reads_hold_projection_and_consequence_retains_chart_history_without_audio():
    torch.manual_seed(132)
    net, x, control = setup()
    with torch.no_grad():
        net.count_prior.readout[-1].weight.normal_(std=.05)
    audio = torch.randn(1, net.config.conditioned_audio_width)
    exact = x.base.row_exact[:1]
    preview = x.row_preview[:1]
    a = net.count_prior(audio, exact, preview, control)
    changed = exact.clone()
    changed[..., -23:-7] = torch.randn_like(changed[..., -23:-7])
    changed[..., -3:] = torch.randn_like(changed[..., -3:])
    torch.testing.assert_close(a, net.count_prior(audio, changed, preview, control), rtol=0, atol=0)
    torch.testing.assert_close(a, net.count_prior(audio, exact.flip(-2), preview, control))
    changed = exact.clone()
    changed[..., net.count_prior.hold_positions] += 1.
    assert not torch.allclose(a, net.count_prior(audio, changed, preview, control))
    contexts = []
    original = net.row_consequence.score
    def record(hands, local, timing):
        contexts.append(hands.detach().clone())
        return original(hands, local, timing)
    net.row_consequence.score = record
    history = net.temporal.boundary[None]
    def score(a, h):
        return net.planned_row_log_probs(a, h, exact, x.base.row_legal[:1], x.base.occupancy[:1],
            preview, x.consequence_local[:1], x.consequence_timing[:1], control=control,
            response_allowed=x.response_allowed[:1])
    left = score(audio, history)
    right = score(audio+2., history)
    torch.testing.assert_close(contexts[0], contexts[1], rtol=0, atol=0)
    assert not torch.allclose(left, right)
    score(audio, history+1.)
    assert not torch.allclose(contexts[0], contexts[2])
