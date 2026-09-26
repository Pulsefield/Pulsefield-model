"""Tune existing R1 difficulty columns without changing its inference graph."""
import hashlib

import torch
from torch import nn
from torch.nn.utils import parametrize

from ..bounded_typed_continuation.features import TIME_DIM
from ..typed_audio_continuation.controls import ControlSchedule, PER_FIELD_SCOPE


TARGETS = ('row_control', 'composition.readout.0')


class SelectedColumns(nn.Module):
    def __init__(self, weight, columns):
        super().__init__()
        self.register_buffer('columns', torch.tensor(columns, device=weight.device), persistent=False)
        self.values = nn.Parameter(weight.detach().index_select(1, self.columns).clone())

    def forward(self, original):
        return original.index_copy(1, self.columns, self.values)


def difficulty_columns(model):
    if model.control_encoding != PER_FIELD_SCOPE:
        raise ValueError('Difficulty tuning requires independently owned control clocks')
    n = 2+len(model.style_names)
    width = ControlSchedule(style_names=model.style_names).width_for(PER_FIELD_SCOPE)
    selected = (0, n, *range(2*n, 2*n+2*TIME_DIM))
    return {path: tuple(i+model.get_submodule(path).in_features-width for i in selected) for path in TARGETS}


def tune_difficulty(model):
    """Freeze the model and expose only R1's difficulty feature columns.

    Parametrizations protect the untouched columns even with optimizer weight
    decay. Export with folded_state_dict; its ordinary weights load through the
    unchanged inference loader. Audio, timing and history-cache parameters stay
    fixed, so this intervention does not invalidate their cached values.
    """
    columns = difficulty_columns(model)
    if any(parametrize.is_parametrized(model.get_submodule(path), 'weight') for path in TARGETS):
        raise ValueError('Difficulty tuning is already attached')
    for p in model.parameters():
        p.requires_grad_(False)
    for path, indices in columns.items():
        layer = model.get_submodule(path)
        parametrize.register_parametrization(layer, 'weight', SelectedColumns(layer.weight, indices))
    return columns


def folded_state_dict(model):
    """Return independent CPU tensors in the standard inference checkpoint schema."""
    prefixes = tuple(path+'.parametrizations.weight.' for path in TARGETS)
    result = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
              if not name.startswith(prefixes)}
    for path in TARGETS:
        result[path+'.weight'] = model.get_submodule(path).weight.detach().cpu().clone()
    return result


def frozen_digest(model):
    """Hash all effective weights/buffers except the explicitly tuned columns."""
    values = folded_state_dict(model)
    for path, indices in difficulty_columns(model).items():
        values[path+'.weight'][:, indices] = 0
    result = hashlib.sha256()
    for name, value in sorted(values.items()):
        result.update(name.encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(value.contiguous().numpy().tobytes())
    return result.hexdigest()
