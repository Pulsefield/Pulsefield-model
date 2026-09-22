"""Mapper v2.1 sparse-token model."""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "CANONICAL_LANE_ORDER": "vocab",
    "KEY_COUNT": "vocab",
    "SPARSE_LANE_ACTIONS": "vocab",
    "LaneAction": "vocab",
    "MapperV21Vocab": "vocab",
    "MapperV21Config": "model",
    "MapperV21ForwardOutput": "model",
    "MapperV21IncrementalDecodeState": "model",
    "MapperV21IncrementalDecodeOutput": "model",
    "MapperV21Model": "model",
    "MapperV21ModelOutput": "model",
    "MapperV21LossConfig": "loss",
    "MapperV21LossOutput": "loss",
    "MapperV21ModelLoss": "loss",
    "MapperTimepoint": "tokenizer",
    "MapperTokenizationError": "tokenizer",
    "TokenizedMapperWindow": "tokenizer",
    "UnsupportedMapperActionError": "tokenizer",
    "encode_full_chart_tokens": "tokenizer",
    "encode_mapper_window": "tokenizer",
    "final_full_chart_token_before": "tokenizer",
    "hitobjects_to_mapper_timepoints": "tokenizer",
    "ln_carry_state_at": "tokenizer",
    "mapper_chart_end_ms": "tokenizer",
    "quantize_10ms_half_up": "tokenizer",
    "tokenize_hitobjects_window": "tokenizer",
    "window_timepoints": "tokenizer",
    "LNCarryState": "replay",
    "MapperReplayState": "replay",
    "ReplayError": "replay",
    "empty_ln_carry_state": "replay",
    "ln_carry_state_tensors": "replay",
    "replay_tokens": "replay",
    "transition_replay_state": "replay",
    "build_grammar_mask": "grammar",
    "valid_token_mask": "grammar",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
