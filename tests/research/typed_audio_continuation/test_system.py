from dataclasses import asdict

import numpy as np
import pytest
import torch

from ensomi_model.research.planned_audio_continuation.model import PlannedModelConfig
from ensomi_model.research.typed_audio_continuation.allocation import LnFeedback
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule, ControlSpan
from ensomi_model.research.typed_audio_continuation.demand import AudioDemand, DemandFeedback
from ensomi_model.research.typed_audio_continuation.generation import TypedSession
from ensomi_model.research.typed_audio_continuation.guidance import StarGuidance
from ensomi_model.research.typed_audio_continuation.model import TypedAudioModel
from ensomi_model.research.typed_audio_continuation.program import Recovery
from ensomi_model.research.typed_audio_continuation.response_preference import RecoveryPreference
from ensomi_model.research.typed_audio_continuation.system import FORMAT, load_system


def test_bundle_roundtrip_keeps_both_models_and_the_actual_sampling_recipe(tmp_path):
    torch.manual_seed(176)
    torch.set_num_threads(1)
    model = TypedAudioModel(PlannedModelConfig(history_levels=2, skeleton_levels=2),
                            style_names=('tech',), ln_prior=.2).eval()
    schedule = ControlSchedule((ControlSpan(0, 6001, stars=3., ln_fraction=.7, style={'tech': 1.}),),
                               model.style_names)
    demand = AudioDemand(model.config.conditioned_audio_width, schedule.width, 3).eval()
    policy = dict(recovery=asdict(Recovery(60, 50, 50)), ln_feedback=asdict(LnFeedback(strength=.6)),
        recovery_preference=asdict(RecoveryPreference(head_pressure=3)),
        star_guidance=asdict(StarGuidance(1.6)),
        demand_feedback=asdict(DemandFeedback(strength=1.5, memory_ms=3000)))
    path = tmp_path/'system.pt'
    torch.save(dict(format=FORMAT, model_config=asdict(model.config), probability_options=model.probability_options(),
        model=model.state_dict(), demand_config=demand.config, demand=demand.state_dict(), sampling=policy), path)
    system = load_system(path)
    for key, value in policy.items():
        assert asdict(getattr(system, key)) == value
    mel = np.zeros((600, 128), np.float32)
    loaded = system.session(mel, 6000, ControlSchedule(schedule.spans), seed=921)
    direct = TypedSession(model, mel, 6000, schedule, seed=921, demand_model=demand,
        recovery=Recovery(**policy['recovery']), ln_feedback=LnFeedback(**policy['ln_feedback']),
        recovery_preference=RecoveryPreference(**policy['recovery_preference']),
        star_guidance=StarGuidance(**policy['star_guidance']),
        demand_feedback=DemandFeedback(**policy['demand_feedback']))
    for session in (loaded, direct):
        session.publish_to(1000)
        session.update_controls(ControlSpan(session.coverage+1, 4000, stars=5., ln_fraction=.2))
        session.publish_to(6000)
    assert loaded.rows == direct.rows
    assert loaded.coverage == direct.coverage == 6000
    assert not any(loaded.replay.occupancy)
    saved = torch.load(path, weights_only=True)
    saved['sampling']['unconsumed_controller'] = {}
    torch.save(saved, path)
    with pytest.raises(ValueError, match='sampling recipe'):
        load_system(path)
