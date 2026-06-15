from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from google.protobuf.message import DecodeError
from pulsefield.protocol.v1 import core_pb2, envelope_pb2, inference_pb2

from pulsefield_model.inference.errors import ProtocolError
from pulsefield_model.inference.service_models import (
    AudioCommand,
    EndOfStreamEvent,
    ErrorEvent,
    HitObjectTokenEvent,
    InferenceRoute,
    MapperStreamBeginEvent,
    NodeHelloCommand,
    ReadyCommand,
    ReferenceTimeCommand,
    ServiceCommand,
    ServiceEvent,
    StatusEvent,
    StopCommand,
    command_to_endpoint_payload,
    service_event_from_endpoint_payload,
)


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


def envelope_to_command(envelope: envelope_pb2.Envelope) -> ServiceCommand:
    payload_name = envelope.WhichOneof("payload")
    if payload_name is None:
        raise ProtocolError("protobuf envelope must include a payload")

    if payload_name == "node_hello":
        return NodeHelloCommand()
    if payload_name == "ready":
        return ReadyCommand()
    if payload_name == "audio":
        _require_envelope_session_id(envelope, payload_name)
        return _audio_request_to_command(envelope)
    if payload_name == "reference_time":
        _require_envelope_session_id(envelope, payload_name)
        request = envelope.reference_time
        return ReferenceTimeCommand(
            session_id=envelope.session_id,
            ref_time_ms=int(request.ref_time_ms),
            local_host_time_send_ms=int(request.local_host_time_send_ms),
            audio_length_ms=int(request.audio_length_ms) if request.HasField("audio_length_ms") else None,
        )
    if payload_name == "stop_session":
        _require_envelope_session_id(envelope, payload_name)
        return StopCommand(
            session_id=envelope.session_id,
            reason=envelope.stop_session.reason or "client_stop",
        )

    raise ProtocolError(f"unsupported inbound protobuf payload: {payload_name}")


def envelope_to_message(envelope: envelope_pb2.Envelope) -> dict[str, Any]:
    return command_to_endpoint_payload(envelope_to_command(envelope))


def outbound_event_to_envelope(
    event: ServiceEvent | Mapping[str, Any],
    *,
    sequence: int,
    source_node_id: str = MODEL_SERVICE_NODE_ID,
) -> envelope_pb2.Envelope:
    service_event = _coerce_service_event(event)
    envelope = _base_envelope(
        session_id=_event_session_id(service_event),
        sequence=sequence,
        source_node_id=source_node_id,
    )
    _fill_envelope_payload(envelope, service_event)
    return envelope


def outbound_payload_to_envelope(
    payload: Mapping[str, Any],
    *,
    sequence: int,
    source_node_id: str = MODEL_SERVICE_NODE_ID,
) -> envelope_pb2.Envelope:
    return outbound_event_to_envelope(payload, sequence=sequence, source_node_id=source_node_id)


def node_hello_envelope(*, sequence: int = 1, source_node_id: str = MODEL_SERVICE_NODE_ID) -> envelope_pb2.Envelope:
    envelope = _base_envelope(session_id="", sequence=sequence, source_node_id=source_node_id)
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


def _audio_request_to_command(envelope: envelope_pb2.Envelope) -> AudioCommand:
    request = envelope.audio
    audio = request.audio
    ref_kind = audio.WhichOneof("ref")
    if ref_kind != "local_path" or not audio.local_path.strip():
        raise ProtocolError("AudioRequest.audio.local_path is required by the local inference server")
    return AudioCommand(
        session_id=envelope.session_id,
        audio_path=audio.local_path,
        audio_length_ms=int(audio.audio_length_ms) if audio.HasField("audio_length_ms") else None,
        difficulty=float(request.difficulty) if request.HasField("difficulty") else None,
        route=_route_from_proto(request.route),
    )


def _coerce_service_event(event: ServiceEvent | Mapping[str, Any]) -> ServiceEvent:
    if isinstance(event, Mapping):
        try:
            return service_event_from_endpoint_payload(event)
        except ValueError as exc:
            raise ProtocolError(str(exc)) from exc
    return event


def _fill_envelope_payload(envelope: envelope_pb2.Envelope, event: ServiceEvent) -> None:
    if isinstance(event, MapperStreamBeginEvent):
        _require_event_session_id(event, "mapper_stream_begin")
        payload = envelope.mapper_stream_begin
        payload.token_contract_version = _positive_int(event.token_contract_version, "token_contract_version")
        if event.audio_length_ms is not None:
            payload.audio_length_ms = _positive_int(event.audio_length_ms, "audio_length_ms")
        return
    if isinstance(event, HitObjectTokenEvent):
        _require_event_session_id(event, "hit_object_token")
        payload = envelope.hit_object_token
        payload.token_id = _uint(event.token_id, "token_id")
        payload.ms_in_ref_audio = _uint(event.ms_in_ref_audio, "ms_in_ref_audio")
        payload.token_index = _uint(event.token_index or 0, "token_index")
        return
    if isinstance(event, EndOfStreamEvent):
        _require_event_session_id(event, "end_of_stream")
        payload = envelope.end_of_stream
        payload.complete_through_ms = _uint(event.complete_through_ms, "complete_through_ms")
        if event.audio_length_ms is not None:
            payload.audio_length_ms = _positive_int(event.audio_length_ms, "audio_length_ms")
        return
    if isinstance(event, ErrorEvent):
        payload = envelope.error
        payload.code = event.code or "protocol_error"
        payload.message = event.message or event.error or "protocol error"
        payload.error_code = _error_code_from_string(event.code)
        payload.phase = event.phase
        payload.route = _route_to_proto(event.route)
        payload.error_kind = event.error_kind or "protocol"
        return
    if isinstance(event, StatusEvent):
        payload = envelope.status
        payload.status = _status_to_proto(event.status)
        payload.message = event.message
        payload.from_status = _status_to_proto(event.from_status)
        payload.reason = event.reason
        payload.route = _route_to_proto(event.route)
        if event.ref_time_ms is not None:
            payload.ref_time_ms = _uint(event.ref_time_ms, "ref_time_ms")
        if event.sender_monotonic_ms is not None:
            payload.sender_monotonic_ms = _uint(event.sender_monotonic_ms, "sender_monotonic_ms")
        if event.reset_sender_monotonic_ms is not None:
            payload.reset_sender_monotonic_ms = _uint(
                event.reset_sender_monotonic_ms,
                "reset_sender_monotonic_ms",
            )
        if event.audio_length_ms is not None:
            payload.audio_length_ms = _positive_int(event.audio_length_ms, "audio_length_ms")
        if event.difficulty is not None:
            payload.difficulty = float(event.difficulty)
        return
    raise ProtocolError(f"unsupported service event: {type(event).__name__}")


def _base_envelope(*, session_id: str, sequence: int, source_node_id: str) -> envelope_pb2.Envelope:
    return envelope_pb2.Envelope(
        session_id=session_id,
        sequence=max(0, int(sequence)),
        sent_at_unix_ms=int(time.time() * 1000),
        source_node_id=source_node_id,
        message_id=str(uuid.uuid4()),
    )


def _event_session_id(event: ServiceEvent) -> str:
    session_id = getattr(event, "session_id", None)
    return session_id if isinstance(session_id, str) else ""


def _require_event_session_id(event: ServiceEvent, payload_name: str) -> None:
    if not _event_session_id(event).strip():
        raise ProtocolError(f"{payload_name} event requires a non-empty session_id")


def _require_envelope_session_id(envelope: envelope_pb2.Envelope, payload_name: str) -> None:
    if not envelope.session_id.strip():
        raise ProtocolError(f"{payload_name} envelope requires a non-empty session_id")


def _route_from_proto(value: int) -> InferenceRoute:
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


def _error_code_from_string(value: object) -> int:
    code = str(value or "").lower()
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


def _uint(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{key} must be an integer")
    value = int(value)
    if value < 0:
        raise ProtocolError(f"{key} must be non-negative")
    return value


def _positive_int(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{key} must be an integer")
    value = int(value)
    if value <= 0:
        raise ProtocolError(f"{key} must be positive")
    return value
