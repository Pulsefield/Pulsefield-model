from __future__ import annotations

import gc
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from pulsefield_model.models.control import ControlDemoGlobalEncoder, ControlDemoGlobalEncoderConfig
from pulsefield_model.inference.mapper_protocol import (
    MapperInferenceProfile,
    MapperProfileConfig,
    resolve_mapper_profile,
)
from pulsefield_model.models.mapper.shared.vocab import MapperTupleVocab
from pulsefield_model.models.mapper.v2 import MapperV2Config, MapperV2Model
from pulsefield_model.models.mapper.v2_1 import MapperV21Config, MapperV21Model, MapperV21Vocab
from pulsefield_model.timing.providers.beatthis import (
    DEFAULT_BEATTHIS_CHECKPOINT,
    DEFAULT_BEATTHIS_DEVICE,
    BeatThisTimingProvider,
)


CONTROL_ENCODER_STATE_PREFIX = "control_encoder."


@dataclass(frozen=True)
class ModelRuntimeConfig:
    mapper_checkpoint_path: str | Path
    control_checkpoint_path: str | Path
    mapper_profile: MapperProfileConfig = "auto"
    beatthis_checkpoint: str | Path = DEFAULT_BEATTHIS_CHECKPOINT
    beatthis_device: str | torch.device | None = DEFAULT_BEATTHIS_DEVICE
    device: str = "auto"
    beatthis_float16: bool = False
    eager_load_beatthis: bool = True


@dataclass(frozen=True)
class ModelRuntime:
    device: torch.device
    beatthis_provider: BeatThisTimingProvider
    control_model: ControlDemoGlobalEncoder
    mapper_model: nn.Module
    vocab: MapperTupleVocab | MapperV21Vocab
    mapper_profile: MapperInferenceProfile
    checkpoint_metadata: Mapping[str, Any]

    @classmethod
    def load(cls, config: ModelRuntimeConfig) -> ModelRuntime:
        return load_model_runtime(config)


def load_model_runtime(config: ModelRuntimeConfig) -> ModelRuntime:
    device = _resolve_runtime_device(config.device)
    beatthis_device = _resolve_beatthis_device(config.beatthis_device)
    beatthis_provider = BeatThisTimingProvider(
        checkpoint_path=str(config.beatthis_checkpoint),
        device=beatthis_device,
        float16=bool(config.beatthis_float16),
    )
    if config.eager_load_beatthis:
        _eager_load_beatthis_provider(beatthis_provider)

    control_path = Path(config.control_checkpoint_path)
    control_checkpoint = _load_checkpoint(control_path, checkpoint_kind="control")
    control_config_raw = _required_mapping(control_checkpoint, "model_config", checkpoint_kind="control")
    control_config = ControlDemoGlobalEncoderConfig(**control_config_raw)
    control_model = ControlDemoGlobalEncoder(control_config)
    control_state_raw = _required_state_dict(control_checkpoint, checkpoint_kind="control")
    control_state = _tensor_state_dict(control_state_raw, checkpoint_kind="control")
    control_load_result = control_model.load_state_dict(control_state, strict=True)
    _freeze_for_inference(control_model)
    control_model.to(device)
    control_metadata = {
        "checkpoint_path": control_path.as_posix(),
        "checkpoint_schema_version": control_checkpoint.get("checkpoint_schema_version"),
        "loaded_keys": len(control_state),
        "missing_keys": tuple(control_load_result.missing_keys),
        "unexpected_keys": tuple(control_load_result.unexpected_keys),
        "optimizer_state_loaded": False,
    }
    del control_checkpoint, control_state_raw, control_state
    gc.collect()

    mapper_path = Path(config.mapper_checkpoint_path)
    mapper_checkpoint = _load_checkpoint(mapper_path, checkpoint_kind="mapper")
    mapper_config_raw = _required_mapping(mapper_checkpoint, "model_config", checkpoint_kind="mapper")
    mapper_control_config_raw = _required_mapping(mapper_checkpoint, "control_model_config", checkpoint_kind="mapper")
    if mapper_control_config_raw != control_config_raw:
        raise ValueError("mapper checkpoint control_model_config does not match control checkpoint model_config")

    mapper_state_raw = _required_state_dict(mapper_checkpoint, checkpoint_kind="mapper")
    mapper_state, filtered_control_encoder_keys = _mapper_tensor_state_dict(mapper_state_raw)
    mapper_version = _detect_mapper_checkpoint_version(mapper_checkpoint, mapper_state=mapper_state)
    mapper_profile = resolve_mapper_profile(config.mapper_profile, checkpoint_version=mapper_version)
    if mapper_profile.name == "v2_1_sparse":
        mapper_config = MapperV21Config(**mapper_config_raw)
        vocab = MapperV21Vocab()
        mapper_model = MapperV21Model(mapper_config, vocab=vocab)
    else:
        mapper_config = MapperV2Config(**mapper_config_raw)
        vocab = MapperTupleVocab()
        mapper_model = MapperV2Model(mapper_config, vocab=vocab)

    if int(mapper_config.control_dim) != int(control_config.d_model):
        raise ValueError(
            "mapper checkpoint model_config.control_dim must match control checkpoint model_config.d_model"
        )
    if mapper_model.control_encoder is not None:
        raise RuntimeError("mapper runtime must not embed a control_encoder")

    mapper_load_result = mapper_model.load_state_dict(mapper_state, strict=True)
    _freeze_for_inference(mapper_model)
    mapper_model.to(device)
    mapper_metadata = {
        "checkpoint_path": mapper_path.as_posix(),
        "version": mapper_version,
        "profile": mapper_profile.name,
        "model_family": mapper_profile.model_family,
        "vocab_contract": mapper_profile.vocab_contract,
        "grammar_contract": mapper_profile.grammar_contract,
        "protobuf_capability": mapper_profile.protocol_contract.capability_name,
        "protobuf_token_contract_version": mapper_profile.protocol_contract.token_contract_version,
        "checkpoint_schema_version": mapper_checkpoint.get("checkpoint_schema_version"),
        "loaded_keys": len(mapper_state),
        "filtered_control_encoder_keys": tuple(filtered_control_encoder_keys),
        "missing_keys": tuple(mapper_load_result.missing_keys),
        "unexpected_keys": tuple(mapper_load_result.unexpected_keys),
        "optimizer_state_loaded": False,
    }
    del mapper_checkpoint, mapper_state_raw, mapper_state
    gc.collect()

    checkpoint_metadata = {
        "device": str(device),
        "beatthis": {
            "checkpoint_path": str(config.beatthis_checkpoint),
            "device": beatthis_device,
            "float16": bool(config.beatthis_float16),
            "eager_loaded": bool(config.eager_load_beatthis),
        },
        "control": control_metadata,
        "mapper": mapper_metadata,
    }
    return ModelRuntime(
        device=device,
        beatthis_provider=beatthis_provider,
        control_model=control_model,
        mapper_model=mapper_model,
        vocab=vocab,
        mapper_profile=mapper_profile,
        checkpoint_metadata=checkpoint_metadata,
    )


def release_torch_cache(device: str | torch.device) -> None:
    gc.collect()
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        return
    if torch_device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        if _mps_is_available():
            torch.mps.empty_cache()


def _resolve_runtime_device(device: str | torch.device) -> torch.device:
    requested = str(device)
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def _resolve_beatthis_device(device: str | torch.device | None) -> str:
    if device is None:
        return DEFAULT_BEATTHIS_DEVICE
    return str(device)


def _mps_is_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _load_checkpoint(path: Path, *, checkpoint_kind: str) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError(
            f"{checkpoint_kind} checkpoint could not be loaded safely with weights_only=True: {path}"
        ) from exc
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{checkpoint_kind} checkpoint must contain a mapping: {path}")
    return checkpoint


def _eager_load_beatthis_provider(provider: BeatThisTimingProvider) -> None:
    load_fn = getattr(provider, "_get_audio2frames", None)
    if not callable(load_fn):
        raise TypeError("BeatThisTimingProvider must expose _get_audio2frames for eager runtime loading")
    load_fn()


def _required_mapping(
    checkpoint: Mapping[str, Any],
    key: str,
    *,
    checkpoint_kind: str,
) -> dict[str, Any]:
    value = checkpoint.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{checkpoint_kind} checkpoint missing {key}")
    return dict(value)


def _required_state_dict(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_kind: str,
) -> Mapping[Any, Any]:
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError(f"{checkpoint_kind} checkpoint missing model_state_dict")
    return state


def _tensor_state_dict(state: Mapping[Any, Any], *, checkpoint_kind: str) -> dict[str, torch.Tensor]:
    tensor_state: dict[str, torch.Tensor] = {}
    non_tensor_keys: list[str] = []
    for key, value in state.items():
        key_str = str(key)
        if not isinstance(value, torch.Tensor):
            non_tensor_keys.append(key_str)
            continue
        if key_str in tensor_state:
            raise ValueError(f"{checkpoint_kind} checkpoint model_state_dict has duplicate key after string coercion")
        tensor_state[key_str] = value
    if non_tensor_keys:
        raise ValueError(
            f"{checkpoint_kind} checkpoint model_state_dict contains non-tensor values: {non_tensor_keys}"
        )
    return tensor_state


def _mapper_tensor_state_dict(state: Mapping[Any, Any]) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    mapper_state: dict[str, torch.Tensor] = {}
    filtered_control_encoder_keys: list[str] = []
    non_tensor_keys: list[str] = []
    for key, value in state.items():
        key_str = str(key)
        if key_str.startswith(CONTROL_ENCODER_STATE_PREFIX):
            filtered_control_encoder_keys.append(key_str)
            continue
        if not isinstance(value, torch.Tensor):
            non_tensor_keys.append(key_str)
            continue
        if key_str in mapper_state:
            raise ValueError("mapper checkpoint model_state_dict has duplicate key after string coercion")
        mapper_state[key_str] = value
    if non_tensor_keys:
        raise ValueError(f"mapper checkpoint model_state_dict contains non-tensor values: {non_tensor_keys}")
    return mapper_state, tuple(sorted(filtered_control_encoder_keys))


def _detect_mapper_checkpoint_version(
    checkpoint: Mapping[str, Any],
    *,
    mapper_state: Mapping[str, torch.Tensor],
) -> str:
    raw_version = checkpoint.get("model_version")
    if isinstance(raw_version, str) and raw_version.strip():
        normalized = raw_version.strip().lower().replace(".", "_")
        if normalized in {"v2_1", "mapper_v2_1", "2_1"}:
            return "v2_1"
        if normalized in {"v2", "mapper_v2", "2"}:
            return "v2"

    run_name = str(checkpoint.get("run_name", "")).lower()
    if "v2_1" in run_name or "v2.1" in run_name:
        return "v2_1"

    output_head = mapper_state.get("output_head.weight")
    if isinstance(output_head, torch.Tensor) and int(output_head.shape[0]) == MapperV21Vocab().size:
        return "v2_1"
    return "v2"


def _freeze_for_inference(model: nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


__all__ = [
    "ModelRuntime",
    "ModelRuntimeConfig",
    "load_model_runtime",
    "release_torch_cache",
]
