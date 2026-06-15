from __future__ import annotations

import asyncio
import errno


class ProtocolError(ValueError):
    pass


class PeerDisconnected(ConnectionError):
    pass


_EXPECTED_SOCKET_DISCONNECT_ERRNOS = {
    errno.ECONNABORTED,
    errno.ECONNRESET,
    errno.ENOTCONN,
    errno.EPIPE,
    errno.ETIMEDOUT,
}


def is_expected_socket_disconnect(exc: BaseException) -> bool:
    if isinstance(exc, (asyncio.IncompleteReadError, ConnectionError)):
        return True
    if isinstance(exc, OSError) and exc.errno in _EXPECTED_SOCKET_DISCONNECT_ERRNOS:
        return True
    return False
