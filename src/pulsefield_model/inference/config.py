from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pulsefield_model.inference.defaults import (
    DEFAULT_BEATTHIS_CHECKPOINT,
    DEFAULT_BEATTHIS_DEVICE,
    DEFAULT_BEATTHIS_FLOAT16,
    DEFAULT_CONTROL_CHECKPOINT_PATH,
    DEFAULT_DECODER_LEAD_MS,
    DEFAULT_DIFFICULTY,
    DEFAULT_EAGER_LOAD_BEATTHIS,
    DEFAULT_HOST,
    DEFAULT_MAPPER_DECODER_WINDOW_MS,
    DEFAULT_MAPPER_MAX_TOKENS,
    DEFAULT_MAPPER_MODEL_ID,
    DEFAULT_MAPPER_PROFILE,
    DEFAULT_MAPPER_TEMPERATURE,
    DEFAULT_MAPPER_TOP_P,
    DEFAULT_MAX_CONTROL_BATCH_SIZE,
    DEFAULT_PORT,
    DEFAULT_RESET_AFTER_AUDIO_END_MS,
    DEFAULT_RUNTIME_DEVICE,
    DEFAULT_TIMING_MOCK_ALIGN_DECODER_WINDOW,
    DEFAULT_TIMING_MOCK_DECODER_LEAD_MS,
    DEFAULT_TIMING_MOCK_MODEL_ID,
    DEFAULT_TIME_SHIFT_LENGTH_PENALTY,
    DEFAULT_TOKEN_SEND_INTERVAL_S,
    DEFAULT_USE_INCREMENTAL_MAPPER_DECODE,
    DEFAULT_WALL_CLOCK_CHECK_INTERVAL_S,
    SUPPORTED_DIFFICULTY_MAX,
    SUPPORTED_DIFFICULTY_MIN,
    TIMING_CANONICALIZATION_CHOICES,
    TIMING_CANONICALIZATION_NONE,
)
from pulsefield_model.inference.mapper_protocol import (
    DEFAULT_MAPPER_PROTOCOL_CONTRACT,
    MAPPER_PROFILE_AUTO,
    normalize_mapper_profile_name,
    resolve_mapper_profile,
)


@dataclass
class InferenceServerConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    token_send_interval_s: float = DEFAULT_TOKEN_SEND_INTERVAL_S
    decoder_lead_ms: int = DEFAULT_DECODER_LEAD_MS
    timing_mock_decoder_lead_ms: int = DEFAULT_TIMING_MOCK_DECODER_LEAD_MS
    timing_mock_align_decoder_window: bool = DEFAULT_TIMING_MOCK_ALIGN_DECODER_WINDOW
    reset_after_audio_end_ms: int = DEFAULT_RESET_AFTER_AUDIO_END_MS
    wall_clock_check_interval_s: float = DEFAULT_WALL_CLOCK_CHECK_INTERVAL_S


@dataclass
class InferenceRuntimeConfig:
    device: str = DEFAULT_RUNTIME_DEVICE
    beatthis_device: str | None = DEFAULT_BEATTHIS_DEVICE
    beatthis_float16: bool = DEFAULT_BEATTHIS_FLOAT16
    eager_load_beatthis: bool = DEFAULT_EAGER_LOAD_BEATTHIS
    canonicalization: str = TIMING_CANONICALIZATION_NONE
    default_difficulty: float = DEFAULT_DIFFICULTY
    max_control_batch_size: int = DEFAULT_MAX_CONTROL_BATCH_SIZE


@dataclass
class InferenceMapperConfig:
    route: str = "mapper"
    model_id: str = DEFAULT_MAPPER_MODEL_ID
    profile: str = DEFAULT_MAPPER_PROFILE
    checkpoint_path: str | None = None
    control_checkpoint_path: str = DEFAULT_CONTROL_CHECKPOINT_PATH.as_posix()
    decoder_window_ms: int = DEFAULT_MAPPER_DECODER_WINDOW_MS
    max_tokens: int = DEFAULT_MAPPER_MAX_TOKENS
    temperature: float = DEFAULT_MAPPER_TEMPERATURE
    top_p: float | None = DEFAULT_MAPPER_TOP_P
    use_incremental_mapper_decode: bool = DEFAULT_USE_INCREMENTAL_MAPPER_DECODE
    time_shift_length_penalty_alpha: float = DEFAULT_TIME_SHIFT_LENGTH_PENALTY
    seed: int | None = None


@dataclass
class TimingMockConfig:
    enabled: bool = True
    route: str = "timing_mock"
    model_id: str = DEFAULT_TIMING_MOCK_MODEL_ID
    timing_checkpoint_path: str = DEFAULT_BEATTHIS_CHECKPOINT


@dataclass
class InferenceProtocolConfig:
    mapper_capability_name: str = DEFAULT_MAPPER_PROTOCOL_CONTRACT.capability_name
    mapper_token_contract_version: int = DEFAULT_MAPPER_PROTOCOL_CONTRACT.token_contract_version
    mapper_manifest_path: str = DEFAULT_MAPPER_PROTOCOL_CONTRACT.manifest_path.as_posix()


@dataclass
class InferenceServiceConfig:
    server: InferenceServerConfig = field(default_factory=InferenceServerConfig)
    runtime: InferenceRuntimeConfig = field(default_factory=InferenceRuntimeConfig)
    mapper: InferenceMapperConfig = field(default_factory=InferenceMapperConfig)
    timing_mock: TimingMockConfig = field(default_factory=TimingMockConfig)
    protocol: InferenceProtocolConfig = field(default_factory=InferenceProtocolConfig)


def default_inference_service_config() -> InferenceServiceConfig:
    return InferenceServiceConfig()


def inference_service_config_from_mapping(data: Mapping[str, Any] | InferenceServiceConfig) -> InferenceServiceConfig:
    if isinstance(data, InferenceServiceConfig):
        config = data
    else:
        from omegaconf import OmegaConf
        from omegaconf.errors import ConfigAttributeError, ConfigKeyError

        try:
            resolved = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(InferenceServiceConfig), data))
        except (ConfigAttributeError, ConfigKeyError) as exc:
            raise ValueError(_friendly_structured_config_error(exc)) from exc
        if not isinstance(resolved, InferenceServiceConfig):
            raise TypeError(f"inference config must resolve to InferenceServiceConfig, got {type(resolved).__name__}")
        config = resolved
    validate_inference_service_config(config)
    return config


def project_to_ws_endpoint_config(config: InferenceServiceConfig) -> "WsEndpointConfig":
    from pulsefield_model.inference.ws_endpoint import WsEndpointConfig

    validate_inference_service_config(config)
    return WsEndpointConfig(
        host=str(config.server.host),
        port=int(config.server.port),
        token_send_interval_s=float(config.server.token_send_interval_s),
        decoder_lead_ms=int(config.server.decoder_lead_ms),
        timing_mock_decoder_lead_ms=int(config.server.timing_mock_decoder_lead_ms),
        timing_mock_align_decoder_window=bool(config.server.timing_mock_align_decoder_window),
        reset_after_audio_end_ms=int(config.server.reset_after_audio_end_ms),
        wall_clock_check_interval_s=float(config.server.wall_clock_check_interval_s),
        decoder_window_ms=int(config.mapper.decoder_window_ms),
        mapper_model_id=str(config.mapper.model_id),
        timing_mock_model_id=str(config.timing_mock.model_id),
        mapper_checkpoint_path=_mapper_checkpoint_path(config),
        control_checkpoint_path=Path(config.mapper.control_checkpoint_path),
        mapper_profile=resolve_mapper_profile(config.mapper.profile).name,
        device=str(config.runtime.device),
        beatthis_checkpoint=str(config.timing_mock.timing_checkpoint_path),
        beatthis_device=config.runtime.beatthis_device,
        beatthis_float16=bool(config.runtime.beatthis_float16),
        eager_load_beatthis=bool(config.runtime.eager_load_beatthis),
        canonicalization=str(config.runtime.canonicalization),
        default_difficulty=float(config.runtime.default_difficulty),
        max_control_batch_size=int(config.runtime.max_control_batch_size),
        max_tokens=int(config.mapper.max_tokens),
        temperature=float(config.mapper.temperature),
        top_p=config.mapper.top_p,
        use_incremental_mapper_decode=bool(config.mapper.use_incremental_mapper_decode),
        time_shift_length_penalty_alpha=float(config.mapper.time_shift_length_penalty_alpha),
        seed=config.mapper.seed,
    )


def validate_inference_service_config(config: InferenceServiceConfig) -> None:
    if config.mapper.route != "mapper":
        raise ValueError(f"mapper.route must be 'mapper', got {config.mapper.route!r}")
    if config.timing_mock.route != "timing_mock":
        raise ValueError(f"timing_mock.route must be 'timing_mock', got {config.timing_mock.route!r}")
    if config.timing_mock.enabled is not True:
        raise ValueError(
            "timing_mock.enabled must be true because the timing_mock route is always available",
        )
    _require_nonempty_string(config.mapper.model_id, "mapper.model_id")
    if config.mapper.checkpoint_path is not None:
        _require_nonempty_string(config.mapper.checkpoint_path, "mapper.checkpoint_path")
    _require_nonempty_string(config.timing_mock.model_id, "timing_mock.model_id")
    _require_nonempty_string(config.timing_mock.timing_checkpoint_path, "timing_mock.timing_checkpoint_path")
    _require_timing_canonicalization(config.runtime.canonicalization)
    _validate_numeric_bounds(config)

    normalized_profile = normalize_mapper_profile_name(config.mapper.profile)
    if normalized_profile == MAPPER_PROFILE_AUTO:
        raise ValueError("mapper.profile must be explicit; use mapper=v2_tuple or mapper=v2_1_sparse")
    profile = resolve_mapper_profile(normalized_profile)

    protocol = profile.protocol_contract
    if config.protocol.mapper_capability_name != protocol.capability_name:
        raise ValueError("protocol.mapper_capability_name does not match mapper protocol contract")
    if int(config.protocol.mapper_token_contract_version) != int(protocol.token_contract_version):
        raise ValueError("protocol.mapper_token_contract_version does not match mapper protocol contract")
    if Path(config.protocol.mapper_manifest_path).name != protocol.manifest_path.name:
        raise ValueError("protocol.mapper_manifest_path does not match mapper protocol contract")


def _mapper_checkpoint_path(config: InferenceServiceConfig) -> Path:
    if config.mapper.checkpoint_path is not None:
        return Path(config.mapper.checkpoint_path)
    return resolve_mapper_profile(config.mapper.profile).default_checkpoint_path


def _validate_numeric_bounds(config: InferenceServiceConfig) -> None:
    _require_int_range(config.server.port, "server.port", min_value=1, max_value=65_535)
    _require_positive_finite_float(config.server.token_send_interval_s, "server.token_send_interval_s")
    _require_nonnegative_int(config.server.decoder_lead_ms, "server.decoder_lead_ms")
    _require_nonnegative_int(config.server.timing_mock_decoder_lead_ms, "server.timing_mock_decoder_lead_ms")
    _require_nonnegative_int(config.server.reset_after_audio_end_ms, "server.reset_after_audio_end_ms")
    _require_positive_finite_float(config.server.wall_clock_check_interval_s, "server.wall_clock_check_interval_s")

    _require_positive_int(config.mapper.decoder_window_ms, "mapper.decoder_window_ms")
    _require_positive_int(config.mapper.max_tokens, "mapper.max_tokens")
    temperature = _require_finite_float(config.mapper.temperature, "mapper.temperature")
    if temperature < 0.0:
        raise ValueError("mapper.temperature must be >= 0")
    if config.mapper.top_p is not None:
        top_p = _require_finite_float(config.mapper.top_p, "mapper.top_p")
        if top_p <= 0.0 or top_p > 1.0:
            raise ValueError("mapper.top_p must satisfy 0 < top_p <= 1")
    penalty = _require_finite_float(
        config.mapper.time_shift_length_penalty_alpha,
        "mapper.time_shift_length_penalty_alpha",
    )
    if penalty < 0.0:
        raise ValueError("mapper.time_shift_length_penalty_alpha must be >= 0")

    difficulty = _require_finite_float(config.runtime.default_difficulty, "runtime.default_difficulty")
    if difficulty < SUPPORTED_DIFFICULTY_MIN or difficulty > SUPPORTED_DIFFICULTY_MAX:
        raise ValueError(
            "runtime.default_difficulty must be inside the supported "
            f"{SUPPORTED_DIFFICULTY_MIN:.1f}..{SUPPORTED_DIFFICULTY_MAX:.1f} range",
        )
    _require_positive_int(config.runtime.max_control_batch_size, "runtime.max_control_batch_size")


def _require_timing_canonicalization(canonicalization: str) -> str:
    if canonicalization not in TIMING_CANONICALIZATION_CHOICES:
        choices = ", ".join(TIMING_CANONICALIZATION_CHOICES)
        raise ValueError(f"canonicalization must be one of {choices}, got {canonicalization!r}")
    return canonicalization


def _require_nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_positive_int(value: object, field: str) -> int:
    return _require_int_range(value, field, min_value=1)


def _require_nonnegative_int(value: object, field: str) -> int:
    return _require_int_range(value, field, min_value=0)


def _require_int_range(value: object, field: str, *, min_value: int | None = None, max_value: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if min_value is not None and value < min_value:
        if min_value == 0:
            raise ValueError(f"{field} must be >= 0")
        if min_value == 1:
            raise ValueError(f"{field} must be > 0")
        raise ValueError(f"{field} must be >= {min_value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{field} must be <= {max_value}")
    return value


def _require_positive_finite_float(value: object, field: str) -> float:
    number = _require_finite_float(value, field)
    if number <= 0.0:
        raise ValueError(f"{field} must be > 0")
    return number


def _require_finite_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _friendly_structured_config_error(exc: Exception) -> str:
    message = str(exc)
    if message.startswith("Key '") and " not in '" in message:
        key = message.split("'", 2)[1]
        section = message.split(" not in '", 1)[1].split("'", 1)[0]
        return f"unknown {section} key(s): {key}"
    return message


__all__ = [
    "InferenceMapperConfig",
    "InferenceProtocolConfig",
    "InferenceRuntimeConfig",
    "InferenceServerConfig",
    "InferenceServiceConfig",
    "TimingMockConfig",
    "default_inference_service_config",
    "inference_service_config_from_mapping",
    "project_to_ws_endpoint_config",
    "validate_inference_service_config",
]
