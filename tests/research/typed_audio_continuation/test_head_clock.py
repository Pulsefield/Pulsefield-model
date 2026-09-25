from dataclasses import replace

import numpy as np
import torch

from ensomi_model.research.planned_audio_continuation.model import PlannedModelConfig
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule
from ensomi_model.research.typed_audio_continuation.model import TypedAudioModel
from ensomi_model.research.typed_audio_continuation.program import CLOCK_DIM, Resources


def test_remaining_availability_does_not_invent_idle_age():
    a = Resources(previous=90, last_head=50, free_at=(0, 20, 100, 140))
    b = replace(a, free_at=(90, 90, 100, 140), previous=95)
    np.testing.assert_array_equal(a.clocks(110, 1000, remaining_availability=True, head_phase=True),
                                  b.clocks(110, 1000, remaining_availability=True, head_phase=True))
    assert not np.array_equal(a.clocks(110, 1000), b.clocks(110, 1000))


def test_release_logit_does_not_renormalize_the_head_probability():
    model = TypedAudioModel(PlannedModelConfig(), head_stream=True)
    audio = torch.zeros(2, model.config.conditioned_audio_width)
    history = torch.zeros(2, 2, 64)
    clocks = torch.zeros(2, CLOCK_DIM)
    control = torch.zeros(2, ControlSchedule().width)
    support = torch.ones(2, 10, 3, dtype=torch.bool)
    support[1, :, 1] = False
    a = model.clock_log_probs(audio, history, clocks, control, support,
                             head_history=history, head_clocks=clocks)
    with torch.no_grad():
        model.clock.bias[1::2].add_(8.)
    b = model.clock_log_probs(audio, history, clocks, control, support,
                             head_history=history, head_clocks=clocks)
    torch.testing.assert_close(a[..., 1], b[..., 1])
    torch.testing.assert_close(b.exp().sum(-1), torch.ones(2, 10))
    assert bool((b[..., 2] > a[..., 2]).all())
