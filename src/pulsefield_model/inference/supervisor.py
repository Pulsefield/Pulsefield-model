from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from pathlib import Path

from pulsefield_model.inference.model_bundles.base import ModelBundle, ModelBundleSnapshot
from pulsefield_model.inference.model_bundles.registry import ModelBundleRegistry
from pulsefield_model.inference.service_models import InferenceRoute
from pulsefield_model.inference.stream_with_cache import DecoderWindow, HitObjectToken


class InferenceSupervisor:
    """Long-lived inference control plane behind the endpoint."""

    def __init__(
        self,
        bundles: Iterable[ModelBundle] = (),
        *,
        registry: ModelBundleRegistry | None = None,
    ) -> None:
        self.registry = ModelBundleRegistry(bundles) if registry is None else registry

    @property
    def models_ready(self) -> bool:
        return self.registry.models_ready

    @models_ready.setter
    def models_ready(self, value: bool) -> None:
        self.registry.models_ready = bool(value)

    async def startup(self) -> None:
        await self.registry.startup()

    async def mount_model(self, model_id: str) -> None:
        await self.registry.mount_model(model_id)

    async def unmount_model(self, model_id: str) -> None:
        await self.registry.unmount_model(model_id)

    async def shutdown(self) -> None:
        await self.registry.shutdown()

    async def aclose(self) -> None:
        await self.shutdown()

    async def __aenter__(self) -> InferenceSupervisor:
        await self.startup()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        await self.shutdown()

    def bundle_status(self) -> tuple[ModelBundleSnapshot, ...]:
        return self.registry.bundle_status()

    async def prepare_audio(
        self,
        *,
        session_id: str,
        audio_path: Path,
        audio_length_ms: int,
        difficulty: float | None,
        route: InferenceRoute = "mapper",
    ) -> None:
        await self.registry.prepare_audio(
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
        async for token in self.registry.iter_hitobject_tokens(
            session_id=session_id,
            audio_path=audio_path,
            audio_length_ms=audio_length_ms,
            window=window,
        ):
            yield token

    async def reset_session(self, session_id: str) -> None:
        await self.registry.reset_session(session_id)
