from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

__all__ = [
    "AudioCommand",
    "EndOfStreamEvent",
    "ErrorEvent",
    "HitObjectTokenEvent",
    "InferenceRoute",
    "MapperStreamBeginEvent",
    "NodeHelloCommand",
    "ReadyCommand",
    "ReferenceTimeCommand",
    "ServiceCommand",
    "ServiceEvent",
    "StatusEvent",
    "StopCommand",
    "command_to_endpoint_message",
    "command_to_endpoint_payload",
    "event_to_endpoint_payload",
    "service_command_from_endpoint_payload",
    "service_command_from_message",
    "service_event_from_endpoint_payload",
    "service_event_to_payload",
]


InferenceRoute: TypeAlias = Literal["mapper", "timing_mock"]


@dataclass(frozen=True)
class ReadyCommand:
    type: Literal["ready"] = "ready"


@dataclass(frozen=True)
class NodeHelloCommand:
    type: Literal["node_hello"] = "node_hello"


@dataclass(frozen=True)
class AudioCommand:
    session_id: str
    audio_path: str
    audio_length_ms: int | None = None
    difficulty: float | None = None
    route: InferenceRoute = "mapper"
    type: Literal["audio"] = "audio"


@dataclass(frozen=True)
class ReferenceTimeCommand:
    session_id: str
    ref_time_ms: int
    local_host_time_send_ms: int | float
    audio_length_ms: int | None = None
    type: Literal["reference_time"] = "reference_time"


@dataclass(frozen=True)
class StopCommand:
    session_id: str
    reason: str = "client_stop"
    type: Literal["stop"] = "stop"


ServiceCommand: TypeAlias = ReadyCommand | NodeHelloCommand | AudioCommand | ReferenceTimeCommand | StopCommand


@dataclass(frozen=True)
class MapperStreamBeginEvent:
    session_id: str
    token_contract_version: int
    audio_length_ms: int | None = None
    type: Literal["mapper_stream_begin"] = "mapper_stream_begin"


@dataclass(frozen=True)
class HitObjectTokenEvent:
    session_id: str
    token_id: int
    ms_in_ref_audio: int
    token_index: int | None = None
    type: Literal["hit_object_token"] = "hit_object_token"


@dataclass(frozen=True)
class EndOfStreamEvent:
    session_id: str
    complete_through_ms: int
    audio_length_ms: int | None = None
    type: Literal["end_of_stream"] = "end_of_stream"


@dataclass(frozen=True)
class ErrorEvent:
    code: str
    message: str
    session_id: str | None = None
    error: str | None = None
    error_kind: str = "protocol"
    phase: str = ""
    route: InferenceRoute | None = None
    type: Literal["error"] = "error"


@dataclass(frozen=True)
class StatusEvent:
    status: str | None = None
    session_id: str | None = None
    message: str = ""
    from_status: str | None = None
    reason: str = ""
    route: InferenceRoute | None = None
    ref_time_ms: int | None = None
    sender_monotonic_ms: int | None = None
    reset_sender_monotonic_ms: int | None = None
    audio_length_ms: int | None = None
    difficulty: float | None = None
    type: Literal["status"] = "status"


ServiceEvent: TypeAlias = MapperStreamBeginEvent | HitObjectTokenEvent | EndOfStreamEvent | ErrorEvent | StatusEvent


def service_command_from_endpoint_payload(payload: Mapping[str, Any]) -> ServiceCommand:
    payload_type = _payload_type(payload)
    if payload_type == "ready":
        return ReadyCommand()
    if payload_type == "node_hello":
        return NodeHelloCommand()
    if payload_type == "audio":
        return AudioCommand(
            session_id=_required_non_empty_string(payload, "session_id"),
            audio_path=_required_non_empty_string(payload, "audio_path"),
            audio_length_ms=_optional_positive_int(payload, "audio_length_ms"),
            difficulty=_optional_float(payload, "difficulty"),
            route=_route_from_payload(payload),
        )
    if payload_type == "reference_time":
        return ReferenceTimeCommand(
            session_id=_required_non_empty_string(payload, "session_id"),
            ref_time_ms=_required_uint(payload, "ref_time_ms"),
            local_host_time_send_ms=_required_number(payload, "local_host_time_send_ms"),
            audio_length_ms=_optional_positive_int(payload, "audio_length_ms"),
        )
    if payload_type == "stop":
        return StopCommand(
            session_id=_required_non_empty_string(payload, "session_id"),
            reason=_non_empty_string_or_default(payload, "reason", "client_stop"),
        )
    raise ValueError(f"unsupported service command type: {payload_type!r}")


def command_to_endpoint_payload(command: ServiceCommand) -> dict[str, Any]:
    if isinstance(command, ReadyCommand):
        return {"type": "ready"}
    if isinstance(command, NodeHelloCommand):
        return {"type": "node_hello"}
    if isinstance(command, AudioCommand):
        message: dict[str, Any] = {
            "type": "audio",
            "session_id": command.session_id,
            "audio_path": command.audio_path,
            "route": command.route,
        }
        if command.audio_length_ms is not None:
            message["audio_length_ms"] = command.audio_length_ms
        if command.difficulty is not None:
            message["difficulty"] = command.difficulty
        return message
    if isinstance(command, ReferenceTimeCommand):
        message = {
            "type": "reference_time",
            "session_id": command.session_id,
            "ref_time_ms": command.ref_time_ms,
            "local_host_time_send_ms": command.local_host_time_send_ms,
        }
        if command.audio_length_ms is not None:
            message["audio_length_ms"] = command.audio_length_ms
        return message
    if isinstance(command, StopCommand):
        return {"type": "stop", "session_id": command.session_id, "reason": command.reason}
    raise TypeError(f"unsupported service command: {type(command).__name__}")


def service_event_from_endpoint_payload(payload: Mapping[str, Any]) -> ServiceEvent:
    payload_type = _payload_type(payload)
    if payload_type == "mapper_stream_begin":
        return MapperStreamBeginEvent(
            session_id=_required_non_empty_string(payload, "session_id"),
            token_contract_version=_required_positive_int(payload, "token_contract_version"),
            audio_length_ms=_optional_positive_int(payload, "audio_length_ms"),
        )
    if payload_type == "hit_object_token":
        return HitObjectTokenEvent(
            session_id=_required_non_empty_string(payload, "session_id"),
            token_id=_required_uint(payload, "token_id"),
            ms_in_ref_audio=_required_uint(payload, "ms_in_ref_audio"),
            token_index=_optional_uint(payload, "token_index"),
        )
    if payload_type == "end_of_stream":
        return EndOfStreamEvent(
            session_id=_required_non_empty_string(payload, "session_id"),
            complete_through_ms=_required_uint(payload, "complete_through_ms"),
            audio_length_ms=_optional_positive_int(payload, "audio_length_ms"),
        )
    if payload_type == "error":
        message = payload.get("message")
        return ErrorEvent(
            code=_non_empty_string_or_default(payload, "code", "protocol_error"),
            message=str(message if message is not None else payload.get("error") or "protocol error"),
            session_id=_optional_string(payload, "session_id"),
            error=_optional_string(payload, "error"),
            error_kind=_non_empty_string_or_default(payload, "error_kind", "protocol"),
            phase=_string_or_default(payload, "phase", ""),
            route=_optional_route_from_payload(payload),
        )
    if payload_type == "status":
        return StatusEvent(
            status=_optional_string(payload, "status") or _optional_string(payload, "to"),
            session_id=_optional_string(payload, "session_id"),
            message=_string_or_default(payload, "message", ""),
            from_status=_optional_string(payload, "from_status") or _optional_string(payload, "from"),
            reason=_string_or_default(payload, "reason", ""),
            route=_optional_route_from_payload(payload),
            ref_time_ms=_optional_uint(payload, "ref_time_ms"),
            sender_monotonic_ms=_optional_uint(payload, "sender_monotonic_ms"),
            reset_sender_monotonic_ms=_optional_uint(payload, "reset_sender_monotonic_ms"),
            audio_length_ms=_optional_positive_int(payload, "audio_length_ms"),
            difficulty=_optional_float(payload, "difficulty"),
        )
    raise ValueError(f"unsupported service event type: {payload_type!r}")


def event_to_endpoint_payload(event: ServiceEvent | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(event, Mapping):
        return dict(event)
    if isinstance(event, MapperStreamBeginEvent):
        payload: dict[str, Any] = {
            "type": "mapper_stream_begin",
            "session_id": event.session_id,
            "token_contract_version": event.token_contract_version,
        }
        if event.audio_length_ms is not None:
            payload["audio_length_ms"] = event.audio_length_ms
        return payload
    if isinstance(event, HitObjectTokenEvent):
        payload = {
            "type": "hit_object_token",
            "session_id": event.session_id,
            "token_id": event.token_id,
            "ms_in_ref_audio": event.ms_in_ref_audio,
        }
        if event.token_index is not None:
            payload["token_index"] = event.token_index
        return payload
    if isinstance(event, EndOfStreamEvent):
        payload = {
            "type": "end_of_stream",
            "session_id": event.session_id,
            "complete_through_ms": event.complete_through_ms,
        }
        if event.audio_length_ms is not None:
            payload["audio_length_ms"] = event.audio_length_ms
        return payload
    if isinstance(event, ErrorEvent):
        payload = {
            "type": "error",
            "code": event.code,
            "message": event.message,
            "error_kind": event.error_kind,
            "phase": event.phase,
        }
        if event.session_id is not None:
            payload["session_id"] = event.session_id
        if event.error is not None:
            payload["error"] = event.error
        if event.route is not None:
            payload["route"] = event.route
        return payload
    if isinstance(event, StatusEvent):
        payload = {
            "type": "status",
            "message": event.message,
            "reason": event.reason,
        }
        for key, value in (
            ("session_id", event.session_id),
            ("status", event.status),
            ("from_status", event.from_status),
            ("route", event.route),
            ("ref_time_ms", event.ref_time_ms),
            ("sender_monotonic_ms", event.sender_monotonic_ms),
            ("reset_sender_monotonic_ms", event.reset_sender_monotonic_ms),
            ("audio_length_ms", event.audio_length_ms),
            ("difficulty", event.difficulty),
        ):
            if value is not None:
                payload[key] = value
        return payload
    raise TypeError(f"unsupported service event: {type(event).__name__}")


def service_command_from_message(message: Mapping[str, Any]) -> ServiceCommand:
    return service_command_from_endpoint_payload(message)


def command_to_endpoint_message(command: ServiceCommand) -> dict[str, Any]:
    return command_to_endpoint_payload(command)


def service_event_to_payload(event: ServiceEvent | Mapping[str, Any]) -> dict[str, Any]:
    return event_to_endpoint_payload(event)


def _payload_type(payload: Mapping[str, Any]) -> str:
    payload_type = payload.get("type")
    if not isinstance(payload_type, str) or not payload_type:
        raise ValueError("service payload must include a non-empty type")
    return payload_type


def _route_from_payload(payload: Mapping[str, Any]) -> InferenceRoute:
    raw_route = payload.get("route")
    if raw_route is None:
        route = "mapper"
    elif isinstance(raw_route, str):
        route = raw_route.strip().lower().replace("-", "_")
    else:
        raise ValueError("route must be a string")
    if route in {"mapper", "inference_route_mapper"}:
        return "mapper"
    if route in {"timing_mock", "inference_route_timing_mock"}:
        return "timing_mock"
    raise ValueError(f"unsupported inference route: {payload.get('route')!r}")


def _optional_route_from_payload(payload: Mapping[str, Any]) -> InferenceRoute | None:
    if payload.get("route") is None:
        return None
    return _route_from_payload(payload)


def _required_non_empty_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    if not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _string_or_default(payload: Mapping[str, Any], key: str, default: str) -> str:
    value = payload.get(key)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _non_empty_string_or_default(payload: Mapping[str, Any], key: str, default: str) -> str:
    value = payload.get(key)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    if not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_string(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _optional_int(payload: Mapping[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    return _int_from_value(value, key)


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    if key not in payload:
        raise ValueError(f"{key} is required")
    return _int_from_value(payload[key], key)


def _optional_uint(payload: Mapping[str, Any], key: str) -> int | None:
    value = _optional_int(payload, key)
    if value is None:
        return None
    if value < 0:
        raise ValueError(f"{key} must be non-negative")
    return value


def _required_uint(payload: Mapping[str, Any], key: str) -> int:
    value = _required_int(payload, key)
    if value < 0:
        raise ValueError(f"{key} must be non-negative")
    return value


def _optional_positive_int(payload: Mapping[str, Any], key: str) -> int | None:
    value = _optional_int(payload, key)
    if value is None:
        return None
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _required_positive_int(payload: Mapping[str, Any], key: str) -> int:
    value = _required_int(payload, key)
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _int_from_value(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return int(value)


def _optional_float(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    return _float_from_value(value, key)


def _float_from_value(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def _required_number(payload: Mapping[str, Any], key: str) -> int | float:
    if key not in payload:
        raise ValueError(f"{key} is required")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value < 0:
        raise ValueError(f"{key} must be finite and non-negative")
    return int(value) if isinstance(value, int) else numeric_value
