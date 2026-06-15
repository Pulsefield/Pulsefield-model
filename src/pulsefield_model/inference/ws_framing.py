from __future__ import annotations

import asyncio
import base64
import hashlib
import struct

from pulsefield_model.inference.errors import (
    PeerDisconnected,
    ProtocolError,
    is_expected_socket_disconnect,
)


async def accept_websocket_handshake(
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
    await drain_writer(writer)


async def send_http_error(
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
    await drain_writer(writer)


async def read_client_binary_frame(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> bytes | None:
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
            await drain_writer(writer)
            return None
        if opcode == 0x9:
            writer.write(encode_server_frame(payload, opcode=0xA))
            await drain_writer(writer)
            continue
        if opcode != 0x2:
            raise ProtocolError(f"websocket messages must be binary protobuf frames; got opcode {opcode}")
        return payload


async def drain_writer(writer: asyncio.StreamWriter) -> None:
    if writer.is_closing():
        raise PeerDisconnected("websocket peer disconnected")
    try:
        await writer.drain()
    except Exception as exc:
        if is_expected_socket_disconnect(exc):
            raise PeerDisconnected("websocket peer disconnected") from exc
        raise


async def close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    try:
        await writer.wait_closed()
    except Exception as exc:
        if not is_expected_socket_disconnect(exc):
            raise


def encode_server_binary_frame(payload: bytes) -> bytes:
    return encode_server_frame(payload, opcode=0x2)


def encode_server_frame(payload: bytes, *, opcode: int) -> bytes:
    length = len(payload)
    if length < 126:
        prefix = bytes([0x80 | opcode, length])
    elif length <= 0xFFFF:
        prefix = bytes([0x80 | opcode, 126]) + struct.pack("!H", length)
    else:
        prefix = bytes([0x80 | opcode, 127]) + struct.pack("!Q", length)
    return prefix + payload
