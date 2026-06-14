from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from google.protobuf.message import DecodeError
from pulsefield.protocol.v1 import core_pb2, envelope_pb2, inference_pb2

from pulsefield_model.inference.ws_endpoint import ProtocolError


MODEL_SERVICE_NODE_ID = "pulsefield-model.inference"
MODEL_SERVICE_APP_ID = "pulsefield.model"
MAPPER_TOKEN_CONTRACT_VERSION = 2

try:
    PROTOCOL_VERSION = version("pulsefield-protocol")
except PackageNotFoundError:
    PROTOCOL_VERSION = "unknown"


def parse_envelope_frame(payload: bytes) -> envelope_pb2.Envelope:
    envelope = envelope_pb2.Envelope()
    try:
        envelope.ParseFromString(payload)
    except DecodeError as exc:
        raise ProtocolError("invalid protobuf envelope") from exc
    return envelope


def envelope_to_message(envelope: envelope_pb2.Envelope) -> dict[str, Any]:
    payload_name = envelope.WhichOneof("payload")
    if payload_name is None:
        raise ProtocolError("protobuf envelope must include a payload")

    if payload_name == "node_hello":
        return {"type": "node_hello"}
    if payload_name == "ready":
        return {"type": "ready"}
    if payload_name == "audio":
        _require_session_id(envelope, payload_name)
        return _audio_request_to_message(envelope)
    if payload_name == "reference_time":
        _require_session_id(envelope, payload_name)
        return _reference_time_request_to_message(envelope)
    if payload_name == "stop_session":
        _require_session_id(envelope, payload_name)
        return {
            "type": "stop",
            "session_id": envelope.session_id,
            "reason": envelope.stop_session.reason or "client_stop",
        }

    raise ProtocolError(f"unsupported inbound protobuf payload: {payload_name}")


def outbound_payload_to_envelope(
    payload: Mapping[str, Any],
    *,
    sequence: int,
    source_node_id: str = MODEL_SERVICE_NODE_ID,
) -> envelope_pb2.Envelope:
    envelope = envelope_pb2.Envelope(
        session_id=_payload_session_id(payload),
        sequence=max(0, int(sequence)),
        sent_at_unix_ms=int(time.time() * 1000),
        source_node_id=source_node_id,
        message_id=str(uuid.uuid4()),
    )

    payload_type = payload.get("type")
    if payload_type == "mapper_stream_begin":
        event = envelope.mapper_stream_begin
        event.token_contract_version = _optional_uint(payload, "token_contract_version") or MAPPER_TOKEN_CONTRACT_VERSION
        audio_length_ms = _optional_uint(payload, "audio_length_ms")
        if audio_length_ms is not None:
            event.audio_length_ms = audio_length_ms
        return envelope
    if payload_type == "hit_object_token":
        event = envelope.hit_object_token
        event.token_id = _required_uint(payload, "token_id")
        event.ms_in_ref_audio = _required_uint(payload, "ms_in_ref_audio")
        event.token_index = _optional_uint(payload, "token_index") or 0
        return envelope
    if payload_type == "end_of_stream":
        event = envelope.end_of_stream
        event.complete_through_ms = _required_uint(payload, "complete_through_ms")
        audio_length_ms = _optional_uint(payload, "audio_length_ms")
        if audio_length_ms is not None:
            event.audio_length_ms = audio_length_ms
        return envelope
    if payload_type == "error":
        event = envelope.error
        event.code = str(payload.get("code") or "protocol_error")
        event.message = str(payload.get("message") or payload.get("error") or "protocol error")
        event.error_code = _error_code_from_payload(payload)
        event.phase = str(payload.get("phase") or "")
        event.route = _route_to_proto(payload.get("route"))
        event.error_kind = str(payload.get("error_kind") or "protocol")
        return envelope
    if payload_type == "status":
        event = envelope.status
        event.status = _status_to_proto(payload.get("status") or payload.get("to"))
        event.message = str(payload.get("message") or "")
        event.from_status = _status_to_proto(payload.get("from_status") or payload.get("from"))
        event.reason = str(payload.get("reason") or "")
        event.route = _route_to_proto(payload.get("route"))
        for source_key, target_attr in (
            ("ref_time_ms", "ref_time_ms"),
            ("sender_monotonic_ms", "sender_monotonic_ms"),
            ("reset_sender_monotonic_ms", "reset_sender_monotonic_ms"),
            ("audio_length_ms", "audio_length_ms"),
        ):
            value = _optional_uint(payload, source_key)
            if value is not None:
                setattr(event, target_attr, value)
        difficulty = payload.get("difficulty")
        if difficulty is not None:
            event.difficulty = float(difficulty)
        return envelope

    raise ProtocolError(f"unsupported outbound payload type for protobuf transport: {payload_type!r}")


def node_hello_envelope(*, sequence: int = 1, source_node_id: str = MODEL_SERVICE_NODE_ID) -> envelope_pb2.Envelope:
    envelope = envelope_pb2.Envelope(
        sequence=max(0, int(sequence)),
        sent_at_unix_ms=int(time.time() * 1000),
        source_node_id=source_node_id,
        message_id=str(uuid.uuid4()),
    )
    hello = envelope.node_hello
    hello.node_id = source_node_id
    hello.role = core_pb2.NODE_ROLE_MODEL_SERVICE
    hello.protocol_version = PROTOCOL_VERSION
    hello.app_id = MODEL_SERVICE_APP_ID
    mapper = hello.capabilities.add()
    mapper.kind = core_pb2.NODE_CAPABILITY_KIND_MAPPER
    mapper.name = "mapper.tuple_tokens"
    mapper.version = str(MAPPER_TOKEN_CONTRACT_VERSION)
    mapper.direction = core_pb2.NODE_CAPABILITY_DIRECTION_PRODUCER
    timing = hello.capabilities.add()
    timing.kind = core_pb2.NODE_CAPABILITY_KIND_TIMING
    timing.name = "timing.mock"
    timing.version = "1"
    timing.direction = core_pb2.NODE_CAPABILITY_DIRECTION_PRODUCER
    return envelope


def _audio_request_to_message(envelope: envelope_pb2.Envelope) -> dict[str, Any]:
    request = envelope.audio
    audio = request.audio
    ref_kind = audio.WhichOneof("ref")
    if ref_kind != "local_path":
        raise ProtocolError("AudioRequest.audio.local_path is required by the local inference server")

    message: dict[str, Any] = {
        "type": "audio",
        "session_id": envelope.session_id,
        "audio_path": audio.local_path,
    }
    if audio.HasField("audio_length_ms"):
        message["audio_length_ms"] = int(audio.audio_length_ms)
    if request.HasField("difficulty"):
        message["difficulty"] = float(request.difficulty)
    route = _route_from_proto(request.route)
    message["route"] = route
    return message


def _reference_time_request_to_message(envelope: envelope_pb2.Envelope) -> dict[str, Any]:
    request = envelope.reference_time
    message: dict[str, Any] = {
        "type": "reference_time",
        "session_id": envelope.session_id,
        "ref_time_ms": int(request.ref_time_ms),
        "local_host_time_send_ms": int(request.local_host_time_send_ms),
    }
    if request.HasField("audio_length_ms"):
        message["audio_length_ms"] = int(request.audio_length_ms)
    return message


def _require_session_id(envelope: envelope_pb2.Envelope, payload_name: str) -> None:
    if not envelope.session_id.strip():
        raise ProtocolError(f"{payload_name} envelope requires a non-empty session_id")


def _payload_session_id(payload: Mapping[str, Any]) -> str:
    value = payload.get("session_id")
    return value if isinstance(value, str) else ""


def _route_from_proto(value: int) -> str:
    if value in (
        inference_pb2.INFERENCE_ROUTE_UNSPECIFIED,
        inference_pb2.INFERENCE_ROUTE_MAPPER,
    ):
        return "mapper"
    if value == inference_pb2.INFERENCE_ROUTE_TIMING_MOCK:
        return "timing_mock"
    raise ProtocolError(f"unsupported inference route enum value: {value}")


def _route_to_proto(value: object) -> int:
    if value in (None, "", "mapper", "INFERENCE_ROUTE_MAPPER"):
        return inference_pb2.INFERENCE_ROUTE_MAPPER
    if value in ("timing_mock", "timing-mock", "INFERENCE_ROUTE_TIMING_MOCK"):
        return inference_pb2.INFERENCE_ROUTE_TIMING_MOCK
    return inference_pb2.INFERENCE_ROUTE_UNSPECIFIED


def _error_code_from_payload(payload: Mapping[str, Any]) -> int:
    code = str(payload.get("code") or "").lower()
    if code == "protocol_error":
        return inference_pb2.INFERENCE_ERROR_CODE_PROTOCOL_ERROR
    if code == "invalid_device":
        return inference_pb2.INFERENCE_ERROR_CODE_INVALID_DEVICE
    if code == "audio_not_found":
        return inference_pb2.INFERENCE_ERROR_CODE_AUDIO_NOT_FOUND
    if code == "session_not_found":
        return inference_pb2.INFERENCE_ERROR_CODE_SESSION_NOT_FOUND
    if code == "session_ownership_conflict":
        return inference_pb2.INFERENCE_ERROR_CODE_SESSION_OWNERSHIP_CONFLICT
    if code == "audio_not_prepared":
        return inference_pb2.INFERENCE_ERROR_CODE_AUDIO_NOT_PREPARED
    if code == "unsupported_route":
        return inference_pb2.INFERENCE_ERROR_CODE_UNSUPPORTED_ROUTE
    if code:
        return inference_pb2.INFERENCE_ERROR_CODE_INFERENCE_FAILED
    return inference_pb2.INFERENCE_ERROR_CODE_UNSPECIFIED


def _status_to_proto(value: object) -> int:
    status = str(value or "").lower()
    if status == "ready":
        return inference_pb2.ENDPOINT_STATUS_READY
    if status == "audio_preparing":
        return inference_pb2.ENDPOINT_STATUS_AUDIO_PREPARING
    if status == "audio_ready":
        return inference_pb2.ENDPOINT_STATUS_AUDIO_READY
    if status == "streaming":
        return inference_pb2.ENDPOINT_STATUS_STREAMING
    if status in {"stopped", "stopped/reset"}:
        return inference_pb2.ENDPOINT_STATUS_STOPPED
    if status == "failed":
        return inference_pb2.ENDPOINT_STATUS_FAILED
    if status == "cold":
        return inference_pb2.ENDPOINT_STATUS_COLD
    if status == "no_session":
        return inference_pb2.ENDPOINT_STATUS_NO_SESSION
    return inference_pb2.ENDPOINT_STATUS_UNSPECIFIED


def _required_uint(payload: Mapping[str, Any], key: str) -> int:
    if key not in payload:
        raise ProtocolError(f"{key} is required")
    return _uint_from_value(payload[key], key)


def _optional_uint(payload: Mapping[str, Any], key: str) -> int | None:
    if key not in payload or payload[key] is None:
        return None
    return _uint_from_value(payload[key], key)


def _uint_from_value(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{key} must be an integer")
    value = int(value)
    if value < 0:
        raise ProtocolError(f"{key} must be non-negative")
    return value
