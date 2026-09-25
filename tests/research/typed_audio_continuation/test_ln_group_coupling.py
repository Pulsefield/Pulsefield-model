import numpy as np
import torch

from ensomi_model.research.planned_audio_continuation.model import PlannedModelConfig
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule, ControlSpan
from ensomi_model.research.typed_audio_continuation.model import TypedAudioModel
from ensomi_model.research.typed_audio_continuation.program import MARKS, MARK_INDEX, Recovery, Resources
from ensomi_model.research.typed_audio_continuation.proportions import GROUP_INDEX, GROUPS, tilt_ln_count


def group_masses(logp):
    membership = torch.tensor(GROUP_INDEX)[None] == torch.arange(len(GROUPS))[:, None]
    return logp.exp() @ membership.to(logp.dtype).T


def test_coupled_groups_keep_actual_marginal_and_reference_ln_preferences():
    torch.manual_seed(38)
    local = torch.randn(2, len(MARKS), requires_grad=True)
    actual = torch.randn(2, len(MARKS), requires_grad=True)
    support = torch.tensor([[r == 0 and 1 <= t+l <= 3 for t,l,r in MARKS]]).expand(2, -1)
    fraction = torch.tensor([.2, .7])
    lp = tilt_ln_count(local, support, fraction, .2, group_raw=actual)
    expected = actual.masked_fill(~support, -torch.inf).log_softmax(-1)
    torch.testing.assert_close(group_masses(lp), group_masses(expected))
    # The two-new-head conditional uses local evidence, not the group's logits.
    indices = [MARK_INDEX[2-l, l, 0] for l in range(3)]
    relative = local[:, indices] + (torch.logit(fraction)-torch.logit(torch.tensor(.2)))[:, None]*torch.arange(3)
    torch.testing.assert_close(lp[:, indices].softmax(-1), relative.softmax(-1))
    (-lp[:, MARK_INDEX[1, 1, 0]].sum()).backward()
    assert bool(torch.isfinite(local.grad).all() and torch.isfinite(actual.grad).all())
    assert float(local.grad.abs().sum()) > 0 and float(actual.grad.abs().sum()) > 0


def test_mark_network_receives_amount_for_groups_without_losing_unknown_semantics():
    torch.manual_seed(81)
    model = TypedAudioModel(PlannedModelConfig(), ln_prior=.2, coupled_ln_groups=True)
    assert model.probability_options()['coupled_ln_groups'] is True
    state = Resources(starts=(0, None, None, None), free_at=(0, 0, 0), previous=0, last_head=0)
    audio = torch.randn(1, model.config.conditioned_audio_width)
    history = torch.randn(1, 2, model.plan_temporal.config.hidden)
    clocks = torch.tensor(state.clocks(np.array([300]), 1000))
    support = torch.tensor(state.support(300, 1000, Recovery())[None])
    def control(fraction):
        return torch.tensor(ControlSchedule((ControlSpan(0, 1000, stars=4, ln_fraction=fraction),)).at([300]))
    low, high = control(.2), control(.7)
    l = model.mark_log_probs(audio, history, clocks, low, support)
    h = model.mark_log_probs(audio, history, clocks, high, support)
    actual = model.mark(model.plan_values(audio, history, clocks, high)).masked_fill(~support, -torch.inf).log_softmax(-1)
    torch.testing.assert_close(group_masses(h), group_masses(actual))
    assert not torch.allclose(group_masses(l), group_masses(h), atol=1e-6, rtol=1e-6)
    unknown = control(None)
    u = model.mark_log_probs(audio, history, clocks, unknown, support)
    expected = model.mark(model.plan_values(audio, history, clocks, unknown)).masked_fill(~support, -torch.inf).log_softmax(-1)
    torch.testing.assert_close(u, expected)
    model.coupled_ln_groups = False
    old_l = model.mark_log_probs(audio, history, clocks, low, support)
    old_h = model.mark_log_probs(audio, history, clocks, high, support)
    torch.testing.assert_close(group_masses(old_l), group_masses(old_h))
