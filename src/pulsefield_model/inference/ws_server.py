from __future__ import annotations

import asyncio
import json
import traceback
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pulsefield_model.inference.errors import (
    PeerDisconnected,
    ProtocolError,
    is_expected_socket_disconnect,
)
from pulsefield_model.inference.mapper_protocol import resolve_mapper_profile
from pulsefield_model.inference.protocol_adapter import PulsefieldProtocolAdapter
from pulsefield_model.inference.service_models import (
    AudioCommand,
    ErrorEvent,
    ServiceCommand,
    ServiceEvent,
    StopCommand,
)
from pulsefield_model.inference.ws_framing import (
    accept_websocket_handshake,
    close_writer,
    drain_writer,
    encode_server_binary_frame,
    read_client_binary_frame,
    send_http_error,
)

if TYPE_CHECKING:
    from pulsefield_model.inference.ws_endpoint import (
        InferenceEndpoint,
        InferenceError,
        WsEndpointConfig,
    )


async def serve_forever(endpoint: InferenceEndpoint | None = None) -> None:
    if endpoint is None:
        from pulsefield_model.inference.config import (
            default_inference_service_config,
            project_to_ws_endpoint_config,
        )
        from pulsefield_model.inference.ws_endpoint import InferenceEndpoint

        endpoint = InferenceEndpoint(
            config=project_to_ws_endpoint_config(default_inference_service_config()),
        )
    from pulsefield_model.inference.ws_endpoint import PULSEFIELD_WS_URL

    server = await asyncio.start_server(
        lambda reader, writer: _handle_websocket_client(endpoint, reader, writer),
        host=endpoint.config.host,
        port=endpoint.config.port,
    )
    sockets = ", ".join(str(sock.getsockname()) for sock in server.sockets or ())
    print(f"ws_server listening on {PULSEFIELD_WS_URL} ({sockets})", flush=True)
    async with server:
        await server.serve_forever()


async def _handle_websocket_client(
    endpoint: InferenceEndpoint,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    from pulsefield_model.inference.ws_endpoint import InferenceError

    peer = _WebSocketPeer(writer, config=endpoint.config)
    owner = object()
    owned_session_ids: set[str] = set()
    try:
        try:
            await accept_websocket_handshake(reader, writer)
        except ProtocolError as exc:
            await send_http_error(writer, status=400, reason="Bad Request", body=str(exc))
            return
        while True:
            try:
                payload = await read_client_binary_frame(reader, writer)
            except ProtocolError as exc:
                await peer.send_event(ErrorEvent(code="protocol_error", message=str(exc)))
                break
            if payload is None:
                break
            try:
                command = peer.decode_inbound_frame(payload)
                await endpoint.handle_command(command, peer, owner=owner)
                _reset_peer_stream_if_needed(peer, command)
                _track_owned_session(
                    owned_session_ids,
                    command=command,
                )
            except ProtocolError as exc:
                await peer.send_event(ErrorEvent(code="protocol_error", message=str(exc)))
            except InferenceError as exc:
                _log_inference_error(exc)
                try:
                    await peer.send_event(exc.to_service_event())
                except PeerDisconnected:
                    return
            except PeerDisconnected:
                return
    except Exception as exc:
        if not is_expected_socket_disconnect(exc):
            raise
    finally:
        try:
            await _stop_owned_sessions(endpoint, owned_session_ids, owner=owner)
        finally:
            await close_writer(writer)


class _WebSocketPeer:
    def __init__(self, writer: asyncio.StreamWriter, *, config: WsEndpointConfig) -> None:
        self._writer = writer
        self._send_lock = asyncio.Lock()
        mapper_contract = resolve_mapper_profile(config.mapper_profile).protocol_contract
        self._protocol_adapter = PulsefieldProtocolAdapter(mapper_contract=mapper_contract)

    def decode_inbound_frame(self, payload: bytes) -> ServiceCommand:
        return self._protocol_adapter.decode_inbound_frame(payload)

    def reset_session_stream(self, session_id: str) -> None:
        self._protocol_adapter.reset_session_stream(session_id)

    async def send_event(self, payload: ServiceEvent | Mapping[str, Any]) -> None:
        async with self._send_lock:
            try:
                for frame_payload in self._protocol_adapter.serialize_outbound_event(payload):
                    self._writer.write(encode_server_binary_frame(frame_payload))
                await drain_writer(self._writer)
            except Exception as exc:
                if is_expected_socket_disconnect(exc):
                    raise PeerDisconnected("websocket peer disconnected") from exc
                raise


def _log_inference_error(exc: InferenceError) -> None:
    print(
        "ws_inference_error "
        + json.dumps(
            {
                "session_id": exc.session_id,
                "phase": exc.phase,
                "route": exc.route,
                "code": exc.code,
                "message": str(exc),
            },
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        flush=True,
    )
    traceback.print_exception(type(exc), exc, exc.__traceback__)


def _track_owned_session(
    owned_session_ids: set[str],
    *,
    command: ServiceCommand,
) -> None:
    if isinstance(command, AudioCommand):
        owned_session_ids.add(command.session_id)
        return
    if isinstance(command, StopCommand):
        owned_session_ids.discard(command.session_id)


def _reset_peer_stream_if_needed(peer: _WebSocketPeer, command: ServiceCommand) -> None:
    if isinstance(command, (AudioCommand, StopCommand)):
        peer.reset_session_stream(command.session_id)


async def _stop_owned_sessions(
    endpoint: InferenceEndpoint,
    owned_session_ids: set[str],
    *,
    owner: object,
) -> None:
    for session_id in tuple(owned_session_ids):
        await endpoint.stop_session(
            session_id,
            reason="peer_disconnect",
            owner=owner,
            transition="peer_disconnect",
        )
    owned_session_ids.clear()


def main(argv: Sequence[str] | None = None) -> int:
    from pulsefield_model.inference.hydra_entry import main as hydra_main

    return hydra_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
