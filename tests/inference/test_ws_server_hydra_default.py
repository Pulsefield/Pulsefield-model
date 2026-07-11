from __future__ import annotations

import asyncio
from typing import Any

import pytest


def test_serve_forever_default_endpoint_uses_explicit_hydra_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    from pulsefield_model.inference import ws_server

    class StopServing(Exception):
        pass

    class FakeServer:
        sockets: tuple[object, ...] = ()

        async def __aenter__(self) -> "FakeServer":
            return self

        async def __aexit__(self, *_exc: object) -> None:
            return None

        async def serve_forever(self) -> None:
            raise StopServing

    captured: dict[str, Any] = {}

    async def fake_start_server(handler: object, *, host: str, port: int) -> FakeServer:
        captured["host"] = host
        captured["port"] = port
        captured["endpoint"] = handler.__closure__[0].cell_contents  # type: ignore[attr-defined]
        return FakeServer()

    monkeypatch.setattr(asyncio, "start_server", fake_start_server)

    with pytest.raises(StopServing):
        asyncio.run(ws_server.serve_forever())

    assert captured["endpoint"].config.mapper_profile == "v2_1_sparse"
