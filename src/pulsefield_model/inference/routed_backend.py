from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pulsefield_model.events.canonical import CanonicalTimepoint
from pulsefield_model.inference.stream_with_cache import (
    DecoderWindow,
    HitObjectToken,
    StreamWithCache,
    StreamWithCacheConfig,
)
from pulsefield_model.models.mapper.shared.vocab import MapperTupleVocab
from pulsefield_model.timing.canonicalization import TIMING_CANONICALIZATION_NONE
from pulsefield_model.timing.grid_fitting import GridFitterConfig
from pulsefield_model.timing.mock_osu_export import (
    build_mock_beat_grid_timepoints,
    timing_grid_from_report,
)
from pulsefield_model.timing.providers.beatthis import (
    DEFAULT_BEATTHIS_CHECKPOINT,
    DEFAULT_BEATTHIS_DEVICE,
)
from pulsefield_model.timing.schema import FittedTimingGrid


InferenceRoute = Literal["mapper", "timing_mock"]
TimingFitFn = Callable[..., dict[str, object]]


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
class TimingMockSession:
    audio_path: Path
    audio_length_ms: int
    timing_grid: FittedTimingGrid
    timepoints: tuple[CanonicalTimepoint, ...]
    max_rounding_error_ms: float


@dataclass
class _SessionLockEntry:
    lock: asyncio.Lock
    ref_count: int = 0


class RoutedInferenceBackend:
    """Per-session backend router for mapper and timing-mock streams."""

    def __init__(
        self,
        config: StreamWithCacheConfig,
        *,
        mapper_backend: RouteBackend | None = None,
        timing_mock_backend: RouteBackend | None = None,
    ) -> None:
        self.config = config
        self.models_ready = False
        self.mapper_backend = StreamWithCache(config) if mapper_backend is None else mapper_backend
        self.timing_mock_backend = (
            TimingMockStreamBackend(config) if timing_mock_backend is None else timing_mock_backend
        )
        self._session_backends: dict[str, RouteBackend] = {}
        self._session_locks: dict[str, _SessionLockEntry] = {}
        self._route_startup_locks: dict[InferenceRoute, asyncio.Lock] = {
            "mapper": asyncio.Lock(),
            "timing_mock": asyncio.Lock(),
        }

    async def startup(self) -> None:
        self.models_ready = True

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
            backend = self._backend_for_route(route)
            await self._ensure_route_started(route, backend)
            await backend.prepare_audio(
                session_id=session_id,
                audio_path=audio_path,
                audio_length_ms=audio_length_ms,
                difficulty=difficulty,
                route=route,
            )
            self._session_backends[session_id] = backend

    async def iter_hitobject_tokens(
        self,
        *,
        session_id: str,
        audio_path: Path,
        audio_length_ms: int,
        window: DecoderWindow,
    ) -> AsyncIterator[HitObjectToken]:
        backend = self._session_backends.get(session_id)
        if backend is None:
            raise RuntimeError(f"session audio has not been prepared: {session_id}")
        async for token in backend.iter_hitobject_tokens(
            session_id=session_id,
            audio_path=audio_path,
            audio_length_ms=audio_length_ms,
            window=window,
        ):
            yield token

    async def reset_session(self, session_id: str) -> None:
        async with self._session_scope(session_id):
            backend = self._session_backends.pop(session_id, None)
            if backend is not None:
                await backend.reset_session(session_id)

    def _backend_for_route(self, route: InferenceRoute) -> RouteBackend:
        if route == "mapper":
            return self.mapper_backend
        if route == "timing_mock":
            return self.timing_mock_backend
        raise ValueError(f"unsupported inference route: {route!r}")

    async def _ensure_route_started(self, route: InferenceRoute, backend: RouteBackend) -> None:
        if backend.models_ready:
            return
        async with self._route_startup_locks[route]:
            if not backend.models_ready:
                await backend.startup()

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
            if entry.ref_count == 0 and session_id not in self._session_backends:
                self._session_locks.pop(session_id, None)


class TimingMockStreamBackend:
    """Timing-only backend that streams mock beat-grid hitobject tokens."""

    def __init__(
        self,
        config: StreamWithCacheConfig,
        *,
        timing_fit_fn: TimingFitFn | None = None,
        vocab: MapperTupleVocab | None = None,
    ) -> None:
        self.config = config
        self.models_ready = False
        self._timing_fit_fn = _default_timing_fit_fn if timing_fit_fn is None else timing_fit_fn
        self._vocab = MapperTupleVocab() if vocab is None else vocab
        self._sessions: dict[str, TimingMockSession] = {}

    async def startup(self) -> None:
        self.models_ready = True

    async def prepare_audio(
        self,
        *,
        session_id: str,
        audio_path: Path,
        audio_length_ms: int,
        difficulty: float | None,
        route: InferenceRoute = "timing_mock",
    ) -> None:
        del difficulty
        if route != "timing_mock":
            raise ValueError(f"TimingMockStreamBackend only supports timing_mock route, got {route!r}")
        session = await asyncio.to_thread(
            self._prepare_session,
            audio_path,
            audio_length_ms,
        )
        self._sessions[session_id] = session

    async def iter_hitobject_tokens(
        self,
        *,
        session_id: str,
        audio_path: Path,
        audio_length_ms: int,
        window: DecoderWindow,
    ) -> AsyncIterator[HitObjectToken]:
        del audio_path
        session = self._sessions.get(session_id)
        if session is None:
            raise RuntimeError(f"session audio has not been prepared: {session_id}")
        start_ms = min(max(0, int(window.start_ms)), int(audio_length_ms))
        stream_window = DecoderWindow(start_ms=start_ms, end_ms=int(audio_length_ms))
        for timepoint in _timepoints_in_window(session.timepoints, stream_window):
            yield _hitobject_token_from_timepoint(timepoint, self._vocab)
            interval = max(0.0, float(self.config.token_send_interval_s))
            if interval:
                await asyncio.sleep(interval)

    async def reset_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _prepare_session(self, audio_path: Path, audio_length_ms: int) -> TimingMockSession:
        timing_report = self._timing_fit_fn(
            audio_path,
            checkpoint_path=DEFAULT_BEATTHIS_CHECKPOINT,
            device=str(self.config.beatthis_device or DEFAULT_BEATTHIS_DEVICE),
            float16=bool(self.config.beatthis_float16),
            fitter_config=_grid_fitter_config_for_canonicalization(self.config.canonicalization),
        )
        timing_grid = timing_grid_from_report(timing_report)
        grid_build = build_mock_beat_grid_timepoints(
            timing_grid,
            start_ms=0,
            end_ms=int(audio_length_ms),
        )
        return TimingMockSession(
            audio_path=audio_path,
            audio_length_ms=int(audio_length_ms),
            timing_grid=timing_grid,
            timepoints=grid_build.timepoints,
            max_rounding_error_ms=float(grid_build.max_rounding_error_ms),
        )


def _default_timing_fit_fn(*args: object, **kwargs: object) -> dict[str, object]:
    from pulsefield_model.timing.fit_audio import fit_audio_file

    return fit_audio_file(*args, **kwargs)


def _grid_fitter_config_for_canonicalization(canonicalization: str) -> GridFitterConfig:
    return GridFitterConfig(
        canonicalization=canonicalization,
        canonicalize_tempo_aliases=canonicalization == TIMING_CANONICALIZATION_NONE,
    )


def _timepoints_in_window(
    timepoints: tuple[CanonicalTimepoint, ...],
    window: DecoderWindow,
) -> tuple[CanonicalTimepoint, ...]:
    start_ms = int(window.start_ms)
    end_ms = int(window.end_ms)
    return tuple(
        timepoint
        for timepoint in timepoints
        if start_ms <= int(timepoint.time_ms) < end_ms
    )


def _hitobject_token_from_timepoint(
    timepoint: CanonicalTimepoint,
    vocab: MapperTupleVocab,
) -> HitObjectToken:
    token_id = int(vocab.encode_event(timepoint.lane_actions))
    actions = vocab.decode_event(token_id)
    return HitObjectToken(
        token_id=token_id,
        token_name=vocab.token_name(token_id),
        ms_in_ref_audio=int(timepoint.time_ms),
        actions=tuple(action.value for action in actions),
    )
