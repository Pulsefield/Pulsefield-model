from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path

from pulsefield_model.events.canonical import CanonicalTimepoint
from pulsefield_model.inference.model_bundles.base import ModelBundle, RouteBackend
from pulsefield_model.inference.service_models import InferenceRoute
from pulsefield_model.inference.stream_with_cache import DecoderWindow, HitObjectToken, StreamWithCacheConfig
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


DEFAULT_TIMING_MOCK_MODEL_ID = "timing_mock/default"
TimingFitFn = Callable[..., dict[str, object]]


@dataclass(frozen=True)
class TimingMockSession:
    audio_path: Path
    audio_length_ms: int
    timing_grid: FittedTimingGrid
    timepoints: tuple[CanonicalTimepoint, ...]
    max_rounding_error_ms: float


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

    async def shutdown(self) -> None:
        self._sessions.clear()
        self.models_ready = False

    async def aclose(self) -> None:
        await self.shutdown()

    async def __aenter__(self) -> TimingMockStreamBackend:
        await self.startup()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        await self.shutdown()

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


class TimingMockModelBundle(ModelBundle):
    """Model bundle for the timing_mock route."""

    def __init__(
        self,
        config: StreamWithCacheConfig,
        *,
        model_id: str = DEFAULT_TIMING_MOCK_MODEL_ID,
        backend: RouteBackend | None = None,
    ) -> None:
        self.config = config
        self.backend = TimingMockStreamBackend(config) if backend is None else backend
        super().__init__(model_id=model_id, route="timing_mock")

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
        route: InferenceRoute = "timing_mock",
    ) -> None:
        if route != "timing_mock":
            raise ValueError(f"TimingMockModelBundle only supports timing_mock route, got {route!r}")
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
