from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pulsefield_model.inference.service_models import InferenceRoute
from pulsefield_model.inference.stream_with_cache import DecoderWindow, HitObjectToken


ModelBundleState = Literal["cold", "loading", "ready", "draining", "unloading", "failed"]


class RouteBackend(Protocol):
    models_ready: bool

    async def startup(self) -> None:
        ...

    async def prepare_audio(
        self,
        *,
        session_id: str,
        audio_path: Path,
        audio_length_ms: int,
        difficulty: float | None,
        route: InferenceRoute = "mapper",
    ) -> None:
        ...

    async def iter_hitobject_tokens(
        self,
        *,
        session_id: str,
        audio_path: Path,
        audio_length_ms: int,
        window: DecoderWindow,
    ) -> AsyncIterator[HitObjectToken]:
        ...

    async def reset_session(self, session_id: str) -> None:
        ...


@dataclass(frozen=True)
class ModelBundleSnapshot:
    model_id: str
    route: InferenceRoute
    state: ModelBundleState
    lease_count: int
    last_error: str | None = None


class ModelBundleLease:
    def __init__(self, bundle: ModelBundle) -> None:
        self._bundle = bundle
        self._released = False

    @property
    def bundle(self) -> ModelBundle:
        return self._bundle

    async def __aenter__(self) -> ModelBundleLease:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        await self.release()

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._bundle._release_lease()


class ModelBundle(ABC):
    """Lifecycle boundary for one mounted inference model implementation."""

    def __init__(self, *, model_id: str, route: InferenceRoute) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        self.model_id = model_id
        self.route = route
        self._state: ModelBundleState = "cold"
        self._last_error: str | None = None
        self._lease_count = 0
        self._lifecycle_lock = asyncio.Lock()
        self._lease_condition = asyncio.Condition()

    @property
    def state(self) -> ModelBundleState:
        return self._state

    @property
    def models_ready(self) -> bool:
        return self._state == "ready"

    @property
    def lease_count(self) -> int:
        return self._lease_count

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def snapshot(self) -> ModelBundleSnapshot:
        return ModelBundleSnapshot(
            model_id=self.model_id,
            route=self.route,
            state=self.state,
            lease_count=self.lease_count,
            last_error=self.last_error,
        )

    async def mount(self) -> None:
        async with self._lifecycle_lock:
            if self._state == "ready":
                return
            if self._state in {"draining", "unloading"}:
                raise RuntimeError(f"model bundle is not accepting mounts while {self._state}: {self.model_id}")
            self._state = "loading"
            self._last_error = None
            try:
                await self._mount_impl()
            except Exception as exc:
                self._state = "failed"
                self._last_error = _error_text(exc)
                async with self._lease_condition:
                    self._lease_condition.notify_all()
                raise
            self._state = "ready"
            async with self._lease_condition:
                self._lease_condition.notify_all()

    async def unmount(self) -> None:
        async with self._lifecycle_lock:
            if self._state == "cold":
                return
            async with self._lease_condition:
                if self._state == "loading":
                    raise RuntimeError(f"model bundle is still loading: {self.model_id}")
                self._state = "draining"
                while self._lease_count > 0:
                    await self._lease_condition.wait()
                self._state = "unloading"
            try:
                await self._unmount_impl()
            except Exception as exc:
                self._state = "failed"
                self._last_error = _error_text(exc)
                async with self._lease_condition:
                    self._lease_condition.notify_all()
                raise
            self._state = "cold"
            self._last_error = None
            async with self._lease_condition:
                self._lease_condition.notify_all()

    async def aclose(self) -> None:
        await self.unmount()

    async def __aenter__(self) -> ModelBundle:
        await self.mount()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        await self.unmount()

    async def acquire_lease(self) -> ModelBundleLease:
        await self.mount()
        async with self._lease_condition:
            if self._state != "ready":
                raise RuntimeError(f"model bundle is not ready: {self.model_id} ({self._state})")
            self._lease_count += 1
        return ModelBundleLease(self)

    async def _release_lease(self) -> None:
        async with self._lease_condition:
            if self._lease_count <= 0:
                raise RuntimeError(f"model bundle lease underflow: {self.model_id}")
            self._lease_count -= 1
            if self._lease_count == 0:
                self._lease_condition.notify_all()

    @abstractmethod
    async def _mount_impl(self) -> None:
        ...

    @abstractmethod
    async def _unmount_impl(self) -> None:
        ...

    @abstractmethod
    async def prepare_audio(
        self,
        *,
        session_id: str,
        audio_path: Path,
        audio_length_ms: int,
        difficulty: float | None,
        route: InferenceRoute = "mapper",
    ) -> None:
        ...

    @abstractmethod
    async def iter_hitobject_tokens(
        self,
        *,
        session_id: str,
        audio_path: Path,
        audio_length_ms: int,
        window: DecoderWindow,
    ) -> AsyncIterator[HitObjectToken]:
        ...

    @abstractmethod
    async def reset_session(self, session_id: str) -> None:
        ...


def _error_text(exc: Exception) -> str:
    detail = str(exc).strip()
    name = type(exc).__name__
    return f"{name}: {detail}" if detail else name
