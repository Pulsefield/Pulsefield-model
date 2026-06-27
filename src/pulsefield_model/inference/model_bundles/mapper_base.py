from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from pulsefield_model.inference.mapper_protocol import MapperInferenceProfile
from pulsefield_model.inference.model_bundles.base import ModelBundle, RouteBackend
from pulsefield_model.inference.service_models import InferenceRoute
from pulsefield_model.inference.stream_with_cache import (
    DecoderWindow,
    HitObjectToken,
    StreamWithCacheConfig,
)


class StreamWithCacheMapperBundle(ModelBundle):
    """Shared lifecycle wrapper for stream-with-cache mapper implementations."""

    profile: MapperInferenceProfile

    def __init__(
        self,
        config: StreamWithCacheConfig,
        *,
        model_id: str,
        backend: RouteBackend,
    ) -> None:
        self.config = config
        self.backend = backend
        super().__init__(model_id=model_id, route="mapper")

    async def _mount_impl(self) -> None:
        await self.backend.startup()

    async def _unmount_impl(self) -> None:
        shutdown = getattr(self.backend, "shutdown", None)
        if callable(shutdown):
            await shutdown()
            return
        if hasattr(self.backend, "models_ready"):
            setattr(self.backend, "models_ready", False)

    async def prepare_audio(
        self,
        *,
        session_id: str,
        audio_path: Path,
        audio_length_ms: int,
        difficulty: float | None,
        route: InferenceRoute = "mapper",
    ) -> None:
        if route != "mapper":
            raise ValueError(f"{type(self).__name__} only supports mapper route, got {route!r}")
        await self.backend.prepare_audio(
            session_id=session_id,
            audio_path=audio_path,
            audio_length_ms=audio_length_ms,
            difficulty=difficulty,
            route=route,
        )

    async def iter_hitobject_tokens(
        self,
        *,
        session_id: str,
        audio_path: Path,
        audio_length_ms: int,
        window: DecoderWindow,
    ) -> AsyncIterator[HitObjectToken]:
        async for token in self.backend.iter_hitobject_tokens(
            session_id=session_id,
            audio_path=audio_path,
            audio_length_ms=audio_length_ms,
            window=window,
        ):
            yield token

    async def reset_session(self, session_id: str) -> None:
        await self.backend.reset_session(session_id)
