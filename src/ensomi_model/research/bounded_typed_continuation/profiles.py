"""Explicit execution envelopes for the small baseline and the measured R1 teacher."""
from ..scoped_style_modeling.dataset import ContractError
from .contract import Arm


def validate_model_profile(model, profile):
    if profile == 'small':
        if model.hidden > 128 or model.levels > 8 or model.expansion > 4 or model.coupling_rank > 16:
            raise ContractError('Model exceeds the small execution envelope')
    elif profile == 'teacher35m':
        expected = dict(arm=Arm.R1, hidden=512, levels=8, expansion=4, coupling_rank=16,
                        seed_context='observed', long_memory='landmarks', memory_hidden=256,
                        memory_stride=64, endpoint_availability='none', row_consequence='none',
                        head_routing='none', release_routing='none')
        if any(getattr(model, name) != value for name, value in expected.items()):
            raise ContractError('teacher35m requires the clean 512-wide R1 seed/landmark architecture')
    else:
        raise ContractError('execution_profile must be small or teacher35m')


def validate_profile_resources(profile, resources):
    if profile not in ('small', 'teacher35m'):
        raise ContractError('execution_profile must be small or teacher35m')
    if profile == 'teacher35m' and not 512 * 1024**2 <= resources.checkpoint_max_bytes <= 1024**3:
        raise ContractError('teacher35m requires a checkpoint budget between 512 MiB and 1 GiB')
