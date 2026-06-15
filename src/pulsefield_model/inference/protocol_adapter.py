from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from pulsefield.protocol.v1 import envelope_pb2

from pulsefield_model.inference.errors import ProtocolError
from pulsefield_model.inference.protobuf_transport import (
    MAPPER_TOKEN_CONTRACT_VERSION,
    envelope_to_command,
    outbound_event_to_envelope,
    parse_envelope_frame,
)
from pulsefield_model.inference.service_models import (
    EndOfStreamEvent,
    ErrorEvent,
    HitObjectTokenEvent,
    MapperStreamBeginEvent,
    ServiceCommand,
    ServiceEvent,
    service_event_from_endpoint_payload,
)


class PulsefieldProtocolAdapter:
    """Stateful @pulsefield/protocol adapter for one peer connection."""

    def __init__(self) -> None:
        self._sequence_by_session_id: dict[str, int] = {}
        self._token_index_by_session_id: dict[str, int] = {}
        self._stream_begun_session_ids: set[str] = set()

    def decode_inbound_frame(self, payload: bytes) -> ServiceCommand:
        return self.decode_inbound_command(payload)

    def decode_inbound_command(self, payload: bytes) -> ServiceCommand:
        return envelope_to_command(parse_envelope_frame(payload))

    def outbound_envelopes_for_event(
        self,
        event: ServiceEvent | Mapping[str, Any],
    ) -> Iterator[envelope_pb2.Envelope]:
        service_event = _coerce_service_event(event)
        session_id = _event_session_id(service_event)

        if isinstance(service_event, (HitObjectTokenEvent, EndOfStreamEvent)) and session_id:
            if session_id not in self._stream_begun_session_ids:
                self._stream_begun_session_ids.add(session_id)
                yield self._next_envelope(
                    MapperStreamBeginEvent(
                        session_id=session_id,
                        token_contract_version=MAPPER_TOKEN_CONTRACT_VERSION,
                        audio_length_ms=service_event.audio_length_ms
                        if isinstance(service_event, EndOfStreamEvent)
                        else None,
                    ),
                )

        if isinstance(service_event, HitObjectTokenEvent) and session_id:
            yield self._next_envelope(
                HitObjectTokenEvent(
                    session_id=session_id,
                    token_id=service_event.token_id,
                    ms_in_ref_audio=service_event.ms_in_ref_audio,
                    token_index=self._next_token_index(session_id),
                ),
            )
            return

        yield self._next_envelope(service_event)
        if _event_resets_stream(service_event) and session_id:
            self.reset_session_stream(session_id)

    def serialize_outbound_event(self, event: ServiceEvent | Mapping[str, Any]) -> Iterator[bytes]:
        for envelope in self.outbound_envelopes_for_event(event):
            yield envelope.SerializeToString()

    def reset_session_stream(self, session_id: str) -> None:
        self._token_index_by_session_id.pop(session_id, None)
        self._stream_begun_session_ids.discard(session_id)

    def _next_envelope(self, event: ServiceEvent) -> envelope_pb2.Envelope:
        session_id = _event_session_id(event)
        sequence = self._sequence_by_session_id.get(session_id, 0) + 1
        self._sequence_by_session_id[session_id] = sequence
        return outbound_event_to_envelope(event, sequence=sequence)

    def _next_token_index(self, session_id: str) -> int:
        token_index = self._token_index_by_session_id.get(session_id, 0)
        self._token_index_by_session_id[session_id] = token_index + 1
        return token_index


def _coerce_service_event(event: ServiceEvent | Mapping[str, Any]) -> ServiceEvent:
    if not isinstance(event, Mapping):
        return event
    try:
        return service_event_from_endpoint_payload(event)
    except ValueError as exc:
        raise ProtocolError(str(exc)) from exc


def _event_session_id(event: ServiceEvent) -> str:
    session_id = getattr(event, "session_id", None)
    return session_id if isinstance(session_id, str) else ""


def _event_resets_stream(event: ServiceEvent) -> bool:
    if isinstance(event, EndOfStreamEvent):
        return True
    return isinstance(event, ErrorEvent) and event.error_kind == "inference"
