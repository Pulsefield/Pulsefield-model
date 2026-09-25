import numpy as np
import torch

from ensomi_model.research.planned_audio_continuation.model import PlannedModelConfig
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule
from ensomi_model.research.typed_audio_continuation.model import TypedAudioModel
from ensomi_model.research.typed_audio_continuation.program import MARKS, MARK_INDEX, Resources
from ensomi_model.research.typed_audio_continuation.proportions import condition_ln_count, tilt_ln_count


def test_scoped_tilt_can_preserve_a_locally_pure_tap_chord():
    raw = torch.full((1, len(MARKS)), -30., requires_grad=True)
    support = torch.tensor([[t+l == 3 and r == 0 for t,l,r in MARKS]])
    target = MARK_INDEX[3, 0, 0]
    with torch.no_grad():
        raw[0, target] = 30.
    fraction = torch.tensor([.7])
    restricted = condition_ln_count(raw, support, fraction).exp()[0, target]
    revised = tilt_ln_count(raw, support, fraction, .2)
    assert float(restricted.detach()) < .171
    assert float(revised.exp()[0, target].detach()) > .999
    (-revised[0, target]).backward()
    assert bool(torch.isfinite(raw.grad).all())


def test_prior_tilt_is_normalized_and_monotone_with_missing_groups():
    rng = torch.Generator().manual_seed(72)
    raw = torch.randn(3, len(MARKS), generator=rng, requires_grad=True)
    support = torch.tensor([[t+l == 2 and r == 0 for t,l,r in MARKS]]).expand(3, -1)
    count = torch.tensor([l for _,l,_ in MARKS], dtype=raw.dtype)
    lo = tilt_ln_count(raw, support, torch.full((3,), .2), .25).exp()
    hi = tilt_ln_count(raw, support, torch.full((3,), .7), .25).exp()
    torch.testing.assert_close(lo.sum(-1), torch.ones(3))
    torch.testing.assert_close(hi.sum(-1), torch.ones(3))
    assert bool((hi @ count > lo @ count).all())
    hi[:, MARK_INDEX[1, 1, 0]].log().sum().backward()
    assert bool(torch.isfinite(raw.grad).all())


def test_timing_history_fades_without_erasing_base_inputs():
    model = TypedAudioModel(PlannedModelConfig(), bounded_clock=True)
    audio = torch.zeros(1, model.config.conditioned_audio_width)
    control = torch.zeros(1, ControlSchedule().width)
    state = Resources(starts=(0, None, None, None), free_at=(0, 0, 0), previous=0, last_head=0)
    clocks = torch.tensor(state.clocks(np.array([10000]), 20000))
    head_clocks = torch.tensor(state.clocks(np.array([10000]), 20000, head_phase=True))
    history = torch.randn(1, 2, 64)*100
    with torch.no_grad():
        model.clock.bias.fill_(1000.)
    base, modulation, gates = model.clock_parts(audio, history, clocks, control, head_clocks)
    other, changed, _ = model.clock_parts(audio, -history, clocks, control, head_clocks)
    torch.testing.assert_close(base, other)
    assert bool((modulation.abs() <= model.config.head_bound*gates+1e-7).all())
    assert float((modulation-changed).abs().max().detach()) < .001
    # The direct base still receives the actual held key and current controls.
    with torch.no_grad():
        model.clock_base[-1].weight.fill_(.1)
    altered = control.clone();altered[:,0]=1
    b1,_,_ = model.clock_parts(audio, history, clocks, control, head_clocks)
    b2,_,_ = model.clock_parts(audio, history, clocks, altered, head_clocks)
    assert not torch.equal(b1, b2)
