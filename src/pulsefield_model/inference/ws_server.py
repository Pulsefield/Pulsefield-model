from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import struct
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pulsefield_model.inference.stream_with_cache import (
    DEFAULT_CONTROL_CHECKPOINT_PATH,
    DEFAULT_MAPPER_CHECKPOINT_PATH,
)
from pulsefield_model.inference.ws_endpoint import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    PULSEFIELD_WS_URL,
    InferenceEndpoint,
    InferenceError,
    PeerDisconnected,
    ProtocolError,
    WsEndpointConfig,
    _is_expected_socket_disconnect,
    infer_message_type,
    parse_json_message,
    require_session_id,
)
from pulsefield_model.timing.canonicalization import (
    TIMING_CANONICALIZATION_BPM_80_160,
    TIMING_CANONICALIZATION_CHOICES,
    TIMING_CANONICALIZATION_NONE,
)
from pulsefield_model.timing.providers.beatthis import DEFAULT_BEATTHIS_DEVICE


async def serve_forever(endpoint: InferenceEndpoint | None = None) -> None:
    config = WsEndpointConfig()
    endpoint = InferenceEndpoint(config=config) if endpoint is None else endpoint
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
    peer = _WebSocketPeer(writer)
    owner = object()
    owned_session_ids: set[str] = set()
    try:
        try:
            await _accept_websocket_handshake(reader, writer)
        except ProtocolError as exc:
            await _send_http_error(writer, status=400, reason="Bad Request", body=str(exc))
            return
        while True:
            message = await _read_client_text_frame(reader, writer)
            if message is None:
                break
            try:
                parsed_message = parse_json_message(message)
                message_type = infer_message_type(parsed_message)
                await endpoint.handle_message(parsed_message, peer, owner=owner)
                _track_owned_session(
                    owned_session_ids,
                    message_type=message_type,
                    message=parsed_message,
                )
            except ProtocolError as exc:
                await peer.send_json({"type": "error", "error": str(exc)})
            except InferenceError as exc:
                _log_inference_error(exc)
                try:
                    await peer.send_json(exc.to_payload())
                except PeerDisconnected:
                    return
            except PeerDisconnected:
                return
    except Exception as exc:
        if not _is_expected_socket_disconnect(exc):
            raise
    finally:
        try:
            await _stop_owned_sessions(endpoint, owned_session_ids, owner=owner)
        finally:
            await _close_writer(writer)


class _WebSocketPeer:
    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self._writer = writer
        self._send_lock = asyncio.Lock()

    async def send_json(self, payload: Mapping[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        async with self._send_lock:
            try:
                self._writer.write(_encode_server_text_frame(data))
                await _drain_writer(self._writer)
            except Exception as exc:
                if _is_expected_socket_disconnect(exc):
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
    message_type: str,
    message: Mapping[str, Any],
) -> None:
    if message_type in {"audio_path", "audio"}:
        owned_session_ids.add(require_session_id(message))
        return
    if message_type == "stop":
        owned_session_ids.discard(require_session_id(message))


async def _stop_owned_sessions(
    endpoint: InferenceEndpoint,
    owned_session_ids: set[str],
    *,
    owner: object,
) -> None:
    for session_id in tuple(owned_session_ids):
        await endpoint.stop_session(session_id, reason="peer_disconnect", owner=owner)
    owned_session_ids.clear()


async def _accept_websocket_handshake(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    request = await reader.readuntil(b"\r\n\r\n")
    header_text = request.decode("latin1")
    lines = header_text.split("\r\n")
    if not lines or not lines[0].startswith("GET "):
        raise ProtocolError("websocket handshake must use GET")
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line or ":" not in line:
            continue
        name, value = line.split(":", 1)
        headers[name.strip().lower()] = value.strip()
    key = headers.get("sec-websocket-key")
    if not key:
        raise ProtocolError("websocket handshake missing Sec-WebSocket-Key")
    accept = base64.b64encode(
        hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest(),
    ).decode("ascii")
    writer.write(
        (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n"
            "\r\n"
        ).encode("ascii"),
    )
    await _drain_writer(writer)


async def _send_http_error(
    writer: asyncio.StreamWriter,
    *,
    status: int,
    reason: str,
    body: str,
) -> None:
    body_bytes = body.encode("utf-8", errors="replace")
    writer.write(
        (
            f"HTTP/1.1 {int(status)} {reason}\r\n"
            "Connection: close\r\n"
            "Content-Type: text/plain; charset=utf-8\r\n"
            f"Content-Length: {len(body_bytes)}\r\n"
            "\r\n"
        ).encode("ascii")
        + body_bytes,
    )
    await _drain_writer(writer)


async def _read_client_text_frame(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> str | None:
    while True:
        header = await reader.readexactly(2)
        first, second = header
        opcode = first & 0x0F
        masked = bool(second & 0x80)
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", await reader.readexactly(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", await reader.readexactly(8))[0]
        mask = await reader.readexactly(4) if masked else b""
        payload = await reader.readexactly(length)
        if masked:
            payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))

        if opcode == 0x8:
            writer.write(b"\x88\x00")
            await _drain_writer(writer)
            return None
        if opcode == 0x9:
            writer.write(_encode_server_frame(payload, opcode=0xA))
            await _drain_writer(writer)
            continue
        if opcode != 0x1:
            raise ProtocolError(f"unsupported websocket opcode: {opcode}")
        return payload.decode("utf-8")


async def _drain_writer(writer: asyncio.StreamWriter) -> None:
    if writer.is_closing():
        raise PeerDisconnected("websocket peer disconnected")
    try:
        await writer.drain()
    except Exception as exc:
        if _is_expected_socket_disconnect(exc):
            raise PeerDisconnected("websocket peer disconnected") from exc
        raise


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    try:
        await writer.wait_closed()
    except Exception as exc:
        if not _is_expected_socket_disconnect(exc):
            raise


def _encode_server_text_frame(payload: bytes) -> bytes:
    return _encode_server_frame(payload, opcode=0x1)


def _encode_server_frame(payload: bytes, *, opcode: int) -> bytes:
    length = len(payload)
    if length < 126:
        prefix = bytes([0x80 | opcode, length])
    elif length <= 0xFFFF:
        prefix = bytes([0x80 | opcode, 126]) + struct.pack("!H", length)
    else:
        prefix = bytes([0x80 | opcode, 127]) + struct.pack("!Q", length)
    return prefix + payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Run Mapper V2 local WS server at {PULSEFIELD_WS_URL}.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--beatthis-device", default=DEFAULT_BEATTHIS_DEVICE)
    parser.add_argument(
        "--canonicalization",
        nargs="?",
        const=TIMING_CANONICALIZATION_BPM_80_160,
        default=TIMING_CANONICALIZATION_NONE,
        choices=TIMING_CANONICALIZATION_CHOICES,
        help="Fold fitted timing BPMs into [80, 160); pass 'none' to leave timing unchanged.",
    )
    parser.add_argument("--difficulty", type=float, default=4.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--mapper-checkpoint-path", type=Path, default=DEFAULT_MAPPER_CHECKPOINT_PATH)
    parser.add_argument("--control-checkpoint-path", type=Path, default=DEFAULT_CONTROL_CHECKPOINT_PATH)
    args = parser.parse_args(argv)

    config = WsEndpointConfig(
        host=args.host,
        port=args.port,
        mapper_checkpoint_path=args.mapper_checkpoint_path,
        control_checkpoint_path=args.control_checkpoint_path,
        device=args.device,
        beatthis_device=args.beatthis_device,
        canonicalization=args.canonicalization,
        default_difficulty=float(args.difficulty),
        max_tokens=int(args.max_tokens),
    )
    asyncio.run(serve_forever(InferenceEndpoint(config=config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
