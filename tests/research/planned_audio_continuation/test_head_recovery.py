from dataclasses import replace
import math

import torch

from ensomi_model.research.planned_audio_continuation.features import head_clocks
from ensomi_model.research.planned_audio_continuation.model import PlannedAudioModel
from .test_distribution import config


def test_bound_and_time_advance_remove_historical_veto_without_forcing_a_head():
    model = PlannedAudioModel(replace(config(), bounded_head=True))
    # Stress the residual independently of natural parameter scale. The same
    # audio is queried after several legitimate no-event advances.
    torch.nn.init.normal_(model.timing[-1].weight, std=100)
    audio = torch.randn(1, model.config.conditioned_audio_width).expand(4, -1)
    history = torch.randn(4, 2, model.config.head_hidden) * 100
    clocks = torch.from_numpy(head_clocks([None, 100, 100, 100], [100, 100, 1100, 30100]))
    base, residual, gate = model.head_parts(audio, history, clocks)
    torch.testing.assert_close(gate, torch.tensor([0., 1., math.exp(-1), math.exp(-30)]))
    assert torch.all(residual.abs() <= model.config.head_bound * gate[:, None])
    assert torch.equal(residual[0], torch.zeros_like(residual[0]))
    changed = model.head_parts(audio, -history, clocks)
    torch.testing.assert_close(changed[0], base, rtol=0, atol=0)
    torch.testing.assert_close((base+residual)[3], base[3], rtol=0, atol=1e-10)
    # An audio base can still choose no event; recovery is not a timing floor.
    with torch.no_grad():
        model.head_base.weight.zero_()
        model.head_base.bias.fill_(-100)
    assert model.head_logits(audio, history, clocks).max() < -95


def test_extra_audio_base_does_not_redraw_shared_initial_parameters():
    torch.manual_seed(432)
    old = PlannedAudioModel(config())
    torch.manual_seed(432)
    new = PlannedAudioModel(replace(config(), bounded_head=True))
    for name, value in old.state_dict().items():
        if name == 'timing.2.bias':
            assert torch.equal(new.state_dict()[name], torch.zeros_like(value))
        else:
            torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
    assert sum(p.numel() for p in new.head_base.parameters()) == 10 * (new.config.conditioned_audio_width + 1)
