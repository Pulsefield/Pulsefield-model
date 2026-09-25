"""Load the tested model components and sampling recipe as one local system."""
from dataclasses import dataclass

import torch

from ..planned_audio_continuation.model import PlannedModelConfig
from .allocation import LnFeedback
from .controls import ControlSchedule
from .demand import AudioDemand, DemandFeedback
from .generation import TypedSession, rollout
from .guidance import StarGuidance
from .model import TypedAudioModel
from .program import Recovery
from .response_preference import RecoveryPreference

FORMAT = 'typed-audio-system-v1'


@dataclass(frozen=True)
class TypedAudioSystem:
    model: TypedAudioModel
    demand_model: AudioDemand
    recovery: Recovery
    ln_feedback: LnFeedback
    recovery_preference: RecoveryPreference
    star_guidance: StarGuidance
    demand_feedback: DemandFeedback

    def options(self):
        return dict(demand_model=self.demand_model, recovery=self.recovery,
                    ln_feedback=self.ln_feedback, recovery_preference=self.recovery_preference,
                    star_guidance=self.star_guidance, demand_feedback=self.demand_feedback)

    def session(self, mel, duration_ms, controls, *, seed=251925):
        """Start a stateful generator with the bundle's vocabulary and recipe.

Call publish_to only as far as the playback buffer requires. The returned
session supports future control updates while retaining published rows/LNs.
"""
        controls = ControlSchedule(controls.spans, self.model.style_names)
        return TypedSession(self.model, mel, duration_ms, controls, seed=seed, **self.options())

    def generate(self, mel, duration_ms, controls, *, seed=251925, max_seconds=120., on_window=None):
        """Generate an offline candidate using the identical streaming factors."""
        controls = ControlSchedule(controls.spans, self.model.style_names)
        return rollout(self.model, mel, duration_ms, controls, seed=seed,
                       max_seconds=max_seconds, on_window=on_window, **self.options())


def load_system(path, *, device='cpu'):
    """Load a data-only system bundle; a bare core checkpoint is not a bundle.

Both neural components and every sampling policy are required. The loader never
silently drops demand feedback or substitutes a different recovery profile.
"""
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('format') != FORMAT:
        raise ValueError(f'Expected a {FORMAT} bundle')
    model = TypedAudioModel(PlannedModelConfig(**saved['model_config']), **saved['probability_options'])
    model.load_state_dict(saved['model'])
    demand = AudioDemand(**saved['demand_config'])
    demand.load_state_dict(saved['demand'])
    width = ControlSchedule(style_names=model.style_names).width
    if (demand.config['audio_width'] != model.config.conditioned_audio_width or
            demand.config['control_width'] != width or
            demand.config['star_known_index'] != 2+len(model.style_names)):
        raise ValueError('Demand inputs do not match the core audio/control contract')
    model.to(device).eval();demand.to(device).eval()
    policy = saved['sampling']
    types = dict(recovery=Recovery, ln_feedback=LnFeedback, recovery_preference=RecoveryPreference,
                 star_guidance=StarGuidance, demand_feedback=DemandFeedback)
    if policy.keys() != types.keys():
        raise ValueError('System sampling recipe has missing or unconsumed fields')
    return TypedAudioSystem(model, demand, **{key: kind(**policy[key]) for key, kind in types.items()})
