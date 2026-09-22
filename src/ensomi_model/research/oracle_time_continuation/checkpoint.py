"""Versioned neural-state serialization without skeletons or process-local IDs."""
from dataclasses import fields, is_dataclass
from pathlib import Path
import platform

import torch

from ..scoped_style_modeling.dataset import ContractError
from .engine import ContinuationState
from .features import PaceState
from .local import LocalEntry, LocalState
from .model import CACHE_SCHEMA
from .relation import RelationNode, RelationState
from .replay import ExactReplayState
from .schema import CompleteRow
from .state import NeuralState
from .storage import file_digest
from .temporal import TemporalState, TemporalToken, TokenPosition

TYPES = {cls.__name__: cls for cls in (PaceState, LocalEntry, LocalState, RelationNode, RelationState,
                                      ExactReplayState, CompleteRow, TemporalState, TemporalToken, TokenPosition)}


def runtime_identity(device):
    root = Path(__file__).parent
    return dict(torch=str(torch.__version__), python=platform.python_version(), platform=platform.platform(),
                device=str(device), threads=torch.get_num_threads(), cache_schema=CACHE_SCHEMA,
                source_files={p.name: file_digest(p) for p in sorted(root.glob('*.py'))})


def pack(value):
    if is_dataclass(value):
        if type(value).__name__ not in TYPES:
            raise ContractError('Unsupported checkpoint state type')
        return {'type': type(value).__name__, 'fields': {f.name: pack(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, tuple):
        return tuple(pack(v) for v in value)
    return value


def unpack(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(unpack(v, device) for v in value)
    if isinstance(value, dict):
        if value.get('type') not in TYPES or set(value) != {'type', 'fields'}:
            raise ContractError('Unsupported checkpoint state schema')
        return TYPES[value['type']](**{k: unpack(v, device) for k, v in value['fields'].items()})
    return value


def pack_state(state):
    return {name: pack(value) for name, value in dict(replay=state.execution.replay, local=state.local,
                                                    relation=state.relation, temporal=state.temporal).items()}


def restore_state(payload, skeleton, model):
    values = {name: unpack(value, next(model.parameters()).device) for name, value in payload.items()}
    state = NeuralState(ContinuationState(skeleton, values['replay']), values['local'], values['relation'],
                        values['temporal'], model.cache_signature(), True)
    cfg = model.config
    if (state.temporal.row_count != state.execution.next_index or
            len(state.temporal.recent) > cfg.recent_capacity or len(state.temporal.coarse) > cfg.coarse_capacity or
            len(state.relation.nodes) > cfg.relation_capacity or
            any(token.projected is None for token in state.temporal.tokens)):
        raise ContractError('Checkpoint memory capacity, cache mode or event position mismatch')
    return state
