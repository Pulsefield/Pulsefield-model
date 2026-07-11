from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from pulsefield_model.inference.model_bundles.base import (
    ModelBundle,
    ModelBundleLease,
    ModelBundleSnapshot,
)
from pulsefield_model.inference.service_models import InferenceRoute
from pulsefield_model.inference.stream_with_cache import DecoderWindow, HitObjectToken


@dataclass
class BundleSessionBinding:
    bundle: ModelBundle
    lease: ModelBundleLease


@dataclass
class _SessionLockEntry:
    lock: asyncio.Lock
    ref_count: int = 0


class ModelBundleRegistry:
    """Controls bundle mount/unmount lifecycle and session leases."""

    def __init__(self, bundles: Iterable[ModelBundle] = ()) -> None:
        self.models_ready = False
        self._bundles_by_id: dict[str, ModelBundle] = {}
        self._default_bundle_by_route: dict[InferenceRoute, ModelBundle] = {}
        self._session_bindings: dict[str, BundleSessionBinding] = {}
        self._session_locks: dict[str, _SessionLockEntry] = {}
        for bundle in bundles:
            self.add_bundle(bundle)

    def add_bundle(self, bundle: ModelBundle, *, make_default: bool = True) -> None:
        if bundle.model_id in self._bundles_by_id:
            raise ValueError(f"duplicate model bundle id: {bundle.model_id!r}")
        self._bundles_by_id[bundle.model_id] = bundle
        if make_default or bundle.route not in self._default_bundle_by_route:
            self._default_bundle_by_route[bundle.route] = bundle

    async def startup(self) -> None:
        self.models_ready = True

    async def mount_model(self, model_id: str) -> None:
        await self._bundle_for_model_id(model_id).mount()

    async def unmount_model(self, model_id: str) -> None:
        await self._bundle_for_model_id(model_id).unmount()

    async def shutdown(self) -> None:
        for session_id in tuple(self._session_bindings):
            await self.reset_session(session_id)
        for bundle in tuple(self._bundles_by_id.values()):
            await bundle.unmount()
        self.models_ready = False

    async def aclose(self) -> None:
        await self.shutdown()

    def bundle_status(self) -> tuple[ModelBundleSnapshot, ...]:
        return tuple(bundle.snapshot() for bundle in self._bundles_by_id.values())

    def has_session(self, session_id: str) -> bool:
        """Return whether a prepared session currently holds a bundle lease."""

        return session_id in self._session_bindings

    def has_session_lock(self, session_id: str) -> bool:
        """Return whether work is queued or active for a session."""

        return session_id in self._session_locks

    def bundle_for_route(self, route: InferenceRoute) -> ModelBundle:
        try:
            return self._default_bundle_by_route[route]
        except KeyError as exc:
            raise ValueError(f"unsupported inference route: {route!r}") from exc

    async def prepare_audio(
        self,
        *,
        session_id: str,
        audio_path: Path,
        audio_length_ms: int,
        difficulty: float | None,
        route: InferenceRoute = "mapper",
    ) -> None:
        async with self._session_scope(session_id):
            if not self.models_ready:
                await self.startup()
            await self._reset_session_locked(session_id)

            bundle = self.bundle_for_route(route)
            lease = await bundle.acquire_lease()
            try:
                await bundle.prepare_audio(
                    session_id=session_id,
                    audio_path=audio_path,
                    audio_length_ms=audio_length_ms,
                    difficulty=difficulty,
                    route=route,
                )
            except Exception:
                try:
                    await bundle.reset_session(session_id)
                finally:
                    await lease.release()
                raise
            self._session_bindings[session_id] = BundleSessionBinding(bundle=bundle, lease=lease)

    async def iter_hitobject_tokens(
        self,
        *,
        session_id: str,
        audio_path: Path,
        audio_length_ms: int,
        window: DecoderWindow,
    ) -> AsyncIterator[HitObjectToken]:
        binding = self._session_bindings.get(session_id)
        if binding is None:
            raise RuntimeError(f"session audio has not been prepared: {session_id}")
        async for token in binding.bundle.iter_hitobject_tokens(
            session_id=session_id,
            audio_path=audio_path,
            audio_length_ms=audio_length_ms,
            window=window,
        ):
            yield token

    async def reset_session(self, session_id: str) -> None:
        async with self._session_scope(session_id):
            await self._reset_session_locked(session_id)

    async def _reset_session_locked(self, session_id: str) -> None:
        binding = self._session_bindings.pop(session_id, None)
        if binding is None:
            return
        try:
            await binding.bundle.reset_session(session_id)
        finally:
            await binding.lease.release()

    def _bundle_for_model_id(self, model_id: str) -> ModelBundle:
        try:
            return self._bundles_by_id[model_id]
        except KeyError as exc:
            raise ValueError(f"unknown model bundle id: {model_id!r}") from exc

    @asynccontextmanager
    async def _session_scope(self, session_id: str) -> AsyncIterator[None]:
        entry = self._session_locks.get(session_id)
        if entry is None:
            entry = _SessionLockEntry(lock=asyncio.Lock())
            self._session_locks[session_id] = entry
        entry.ref_count += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.ref_count -= 1
            if entry.ref_count == 0 and session_id not in self._session_bindings:
                self._session_locks.pop(session_id, None)
