from dataclasses import asdict

import numpy as np
import torch

from ensomi_model.research.planned_audio_continuation.model import PlannedModelConfig
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule
from ensomi_model.research.typed_audio_continuation.generation import rollout
from ensomi_model.research.typed_audio_continuation.model import TypedAudioModel
from ensomi_model.research.typed_audio_continuation.program import Recovery


def test_rollout_uses_one_declared_recovery_profile_for_plan_and_rows():
    torch.manual_seed(80)
    torch.set_num_threads(1)
    model = TypedAudioModel(PlannedModelConfig(history_levels=2, skeleton_levels=2)).eval()
    recovery = Recovery(hh=500, rh=500, hr=500)
    result = rollout(model, np.zeros((600, 128), np.float32), 6000, ControlSchedule(),
                     recovery=recovery, seed=53)
    assert result.completed
    assert result.metrics['recovery'] == asdict(recovery)
    heads, releases, starts = [None]*4, [None]*4, [None]*4
    for row in result.rows:
        for lane, action in enumerate(row.actions):
            if action in (1, 2):
                if heads[lane] is not None:
                    assert row.time_ms-heads[lane] >= recovery.hh
                if releases[lane] is not None:
                    assert row.time_ms-releases[lane] >= recovery.rh
                heads[lane] = row.time_ms
                if action == 2:
                    starts[lane] = row.time_ms
            elif action == 3:
                assert row.time_ms-starts[lane] >= recovery.hr
                starts[lane] = None
                releases[lane] = row.time_ms
    assert starts == [None]*4
