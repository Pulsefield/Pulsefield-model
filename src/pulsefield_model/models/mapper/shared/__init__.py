"""Shared tuple-token mapper contract used by mapper v2."""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "MapperBatch": "batch",
    "MapperCarryStateBatch": "batch",
    "MapperFragmentState": "batch",
    "MapperTokenContract": "batch",
    "LaneAction": "vocab",
    "MapperTupleVocab": "vocab",
    "MapperTupleConfig": "model",
    "MapperTupleForwardOutput": "model",
    "MapperTupleModelOutput": "model",
    "TupleMapperBase": "model",
    "MapperTupleLossConfig": "loss",
    "MapperTupleLossOutput": "loss",
    "MapperTupleModelLoss": "loss",
    "MapperLossTokenSpec": "loss",
    "LNCarryState": "generation",
    "MapperGeneratedWindow": "generation",
    "RecoveryCEReport": "generation",
    "carry_aware_valid_token_mask": "generation",
    "grammar_constrained_window_generation": "generation",
    "reconstruct_ln_carry_states": "generation",
    "short_rollout_recovery_ce": "generation",
    "strict_match_to_gold_replay": "generation",
    "window_is_complete": "generation",
    "LNCloseAdapter": "adapters",
    "LNCloseAdapterOutput": "adapters",
    "StatePriorAdapter": "adapters",
    "StatePriorAdapterOutput": "adapters",
    "AttentionKVCache": "incremental",
    "IncrementalDecodeState": "incremental",
    "IncrementalDecoderStepOutput": "incremental",
    "IncrementalSelfAttentionKVCache": "incremental",
    "attention_projection_to_heads": "incremental",
    "as_decode_batch_vector": "incremental",
    "as_decode_step_lane_tensor": "incremental",
    "as_decode_step_tensor": "incremental",
    "create_empty_incremental_decode_state": "incremental",
    "decode_position_index": "incremental",
    "incremental_decoder_step": "incremental",
    "normalize_attention_kv_cache": "incremental",
    "transformer_decoder_control_attention_kv_cache": "incremental",
    "validate_incremental_decode_state": "incremental",
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
