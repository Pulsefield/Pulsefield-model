from __future__ import annotations

import asyncio
import inspect
import json
import math
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias

from pulsefield_model.data.control_windows import normalize_difficulty
from pulsefield_model.inference.errors import (
    PeerDisconnected,
    ProtocolError,
    is_expected_socket_disconnect as _is_expected_socket_disconnect,
)
from pulsefield_model.inference.model_bundles import ModelBundleSnapshot
from pulsefield_model.inference.routed_backend import RoutedInferenceBackend
from pulsefield_model.inference.service_models import (
    AudioCommand,
    EndOfStreamEvent,
    ErrorEvent,
    HitObjectTokenEvent,
    InferenceRoute,
    NodeHelloCommand,
    ReadyCommand,
    ReferenceTimeCommand,
    ServiceCommand,
    ServiceEvent,
    StopCommand,
    service_command_from_message,
)
from pulsefield_model.inference.stream_with_cache import (
    DecoderWindow,
    HitObjectToken,
    StreamWithCacheConfig,
    audio_length_ms_from_file,
    clamp_decoder_window_to_audio,
)


PULSEFIELD_WS_URL = "ws://localhost:8765"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8765


class InferenceError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        session_id: str,
        phase: str,
        route: InferenceRoute | None = None,
        code: str = "inference_failed",
    ) -> None:
        super().__init__(message)
        self.session_id = session_id
        self.phase = phase
        self.route = route
        self.code = code

    def to_event(self) -> dict[str, Any]:
        event: dict[str, Any] = {
            "type": "error",
            "error_kind": "inference",
            "session_id": self.session_id,
            "phase": self.phase,
            "code": self.code,
            "message": str(self),
            "error": str(self),
        }
        if self.route is not None:
            event["route"] = self.route
        return event

    def to_service_event(self) -> ErrorEvent:
        return ErrorEvent(
            code=self.code,
            message=str(self),
            session_id=self.session_id,
            error=str(self),
            error_kind="inference",
            phase=self.phase,
            route=self.route,
        )


class EndpointPeer(Protocol):
    async def send_event(self, event: ServiceEvent | Mapping[str, Any]) -> None:
        ...


@dataclass(frozen=True)
class WsEndpointConfig(StreamWithCacheConfig):
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    decoder_lead_ms: int = 2_000
    timing_mock_decoder_lead_ms: int = 0
    timing_mock_align_decoder_window: bool = False
    reset_after_audio_end_ms: int = 2_000
    wall_clock_check_interval_s: float = 0.05


@dataclass(frozen=True)
class ReferenceClock:
    ref_time_ms: int
    local_host_time_send_ms: float
    received_local_host_time_ms: float


@dataclass
class SessionState:
    session_id: str
    owner: object | None = None
    audio_path: Path | None = None
    audio_length_ms: int | None = None
    difficulty: float | None = None
    route: InferenceRoute = "mapper"
    audio_prepared: bool = False
    reference_clock: ReferenceClock | None = None
    stream_task: asyncio.Task[None] | None = None
    wall_clock_reset_task: asyncio.Task[None] | None = None
    decoder_window: DecoderWindow | None = None


SessionStatus: TypeAlias = Literal["no_session", "audio_preparing", "audio_ready", "streaming"]
SessionTransition: TypeAlias = Literal[
    "start_audio",
    "audio_prepared",
    "reference_time",
    "replace_audio",
    "stop",
    "prepare_failed",
    "peer_disconnect",
]


@dataclass(frozen=True)
class SessionTransitionRule:
    allowed_from: frozenset[SessionStatus]
    to_status: SessionStatus


_SESSION_STATUS_ORDER: tuple[SessionStatus, ...] = (
    "no_session",
    "audio_preparing",
    "audio_ready",
    "streaming",
)


SESSION_TRANSITION_RULES: dict[SessionTransition, SessionTransitionRule] = {
    "start_audio": SessionTransitionRule(
        allowed_from=frozenset({"no_session"}),
        to_status="audio_preparing",
    ),
    "audio_prepared": SessionTransitionRule(
        allowed_from=frozenset({"audio_preparing"}),
        to_status="audio_ready",
    ),
    "reference_time": SessionTransitionRule(
        allowed_from=frozenset({"audio_ready", "streaming"}),
        to_status="streaming",
    ),
    "replace_audio": SessionTransitionRule(
        allowed_from=frozenset({"audio_ready", "streaming"}),
        to_status="no_session",
    ),
    "stop": SessionTransitionRule(
        allowed_from=frozenset({"no_session", "audio_preparing", "audio_ready", "streaming"}),
        to_status="no_session",
    ),
    "prepare_failed": SessionTransitionRule(
        allowed_from=frozenset({"audio_preparing"}),
        to_status="no_session",
    ),
    "peer_disconnect": SessionTransitionRule(
        allowed_from=frozenset({"no_session", "audio_preparing", "audio_ready", "streaming"}),
        to_status="no_session",
    ),
}


@dataclass
class _SessionLockEntry:
    lock: asyncio.Lock
    ref_count: int = 0


def session_status(session: SessionState | None) -> SessionStatus:
    if session is None:
        return "no_session"
    if session.reference_clock is not None and session.decoder_window is not None:
        return "streaming"
    if session.audio_prepared:
        return "audio_ready"
    return "audio_preparing"


class EndpointBackend(Protocol):
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


@dataclass
class InferenceEndpoint:
    config: WsEndpointConfig = field(default_factory=WsEndpointConfig)
    backend: EndpointBackend | None = None

    def __post_init__(self) -> None:
        if self.backend is None:
            self.backend = RoutedInferenceBackend(self.config)
        self.sessions: dict[str, SessionState] = {}
        self._startup_lock = asyncio.Lock()
        self._session_locks: dict[str, _SessionLockEntry] = {}

    async def handle_command(
        self,
        command: ServiceCommand,
        peer: EndpointPeer,
        *,
        owner: object | None = None,
    ) -> None:
        if isinstance(command, ReadyCommand):
            await self.startup()
            return
        if isinstance(command, NodeHelloCommand):
            return
        if isinstance(command, AudioCommand):
            await self._handle_audio(command, owner=owner)
            return
        if isinstance(command, ReferenceTimeCommand):
            await self._handle_reference_time(command, peer, owner=owner)
            return
        if isinstance(command, StopCommand):
            await self.stop_session(
                _require_command_session_id(command.session_id),
                reason=_validated_stop_reason(command.reason),
                owner=owner,
            )
            return
        raise ProtocolError(f"unsupported command type: {type(command).__name__}")

    async def handle_message(
        self,
        message: Mapping[str, Any],
        peer: EndpointPeer,
        *,
        owner: object | None = None,
    ) -> None:
        try:
            command = service_command_from_message(message)
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(str(exc)) from exc
        await self.handle_command(command, peer, owner=owner)

    async def startup(self) -> None:
        assert self.backend is not None
        async with self._startup_lock:
            if self.backend.models_ready:
                return
            await self.backend.startup()
            log_ws_status(
                session_id=None,
                from_status="cold",
                to_status="ready",
                reason="ready",
            )

    async def mount_model(self, model_id: str) -> None:
        assert self.backend is not None
        mount = getattr(self.backend, "mount_model", None)
        if not callable(mount):
            raise RuntimeError("configured endpoint backend does not support model mount")
        await mount(model_id)

    async def unmount_model(self, model_id: str) -> None:
        assert self.backend is not None
        unmount = getattr(self.backend, "unmount_model", None)
        if not callable(unmount):
            raise RuntimeError("configured endpoint backend does not support model unmount")
        await unmount(model_id)

    def bundle_status(self) -> tuple[ModelBundleSnapshot, ...]:
        assert self.backend is not None
        status = getattr(self.backend, "bundle_status", None)
        if not callable(status):
            return ()
        return tuple(status())

    async def shutdown(self) -> None:
        assert self.backend is not None
        for session_id in tuple(self.sessions):
            await self.stop_session(session_id, reason="endpoint_shutdown")
        shutdown = getattr(self.backend, "shutdown", None)
        if callable(shutdown):
            await shutdown()

    async def aclose(self) -> None:
        await self.shutdown()

    async def __aenter__(self) -> InferenceEndpoint:
        await self.startup()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        await self.shutdown()

    async def stop_session(
        self,
        session_id: str,
        *,
        reason: str = "client_stop",
        owner: object | None = None,
        transition: SessionTransition = "stop",
    ) -> None:
        async with self._session_scope(session_id):
            await self._stop_session_locked(session_id, reason=reason, owner=owner, transition=transition)

    async def _stop_session_locked(
        self,
        session_id: str,
        *,
        reason: str,
        owner: object | None = None,
        transition: SessionTransition = "stop",
    ) -> None:
        session = self.sessions.get(session_id)
        from_status = enforce_session_transition(transition, session, session_id=session_id)
        if session is None:
            return
        if owner is not None and session.owner is not owner:
            return
        self.sessions.pop(session_id, None)
        await _cancel_task(session.stream_task)
        await _cancel_task(session.wall_clock_reset_task)
        assert self.backend is not None
        await self.backend.reset_session(session_id)
        log_ws_status(
            session_id=session_id,
            from_status=from_status,
            to_status="stopped/reset",
            reason=reason,
        )

    async def _handle_audio(self, command: AudioCommand, *, owner: object | None = None) -> None:
        assert self.backend is not None
        if not self.backend.models_ready:
            raise ProtocolError("send ready before audio")
        session_id = _require_command_session_id(command.session_id)
        if not command.audio_path.strip():
            raise ProtocolError("audio_path must be a non-empty string")
        audio_path = Path(command.audio_path).expanduser()
        audio_length_ms = _validated_optional_audio_length_ms(command.audio_length_ms)
        if audio_length_ms is None:
            audio_length_ms = audio_length_ms_from_file(audio_path)
        if audio_length_ms is None:
            raise ProtocolError("audio_length_ms was omitted and audio duration could not be read from audio_path")
        route = command.route
        difficulty = (
            None
            if route == "timing_mock"
            else _validated_difficulty(command.difficulty, default=self.config.default_difficulty)
        )

        async with self._session_scope(session_id):
            existing = self.sessions.get(session_id)
            if existing is not None:
                if owner is not None and existing.owner is not owner:
                    raise ProtocolError("session is owned by another websocket connection")
                await self._stop_session_locked(
                    session_id,
                    reason="replace_audio",
                    transition="replace_audio",
                )
            enforce_session_transition("start_audio", self.sessions.get(session_id), session_id=session_id)

            session = SessionState(
                session_id=session_id,
                owner=owner,
                audio_path=audio_path,
                audio_length_ms=audio_length_ms,
                difficulty=difficulty,
                route=route,
            )
            self.sessions[session_id] = session
            log_ws_status(
                session_id=session_id,
                from_status="no_session",
                to_status="audio_preparing",
                reason="audio",
                audio_path=str(audio_path),
                audio_length_ms=audio_length_ms,
                difficulty=difficulty,
                route=route,
            )
            try:
                await _prepare_backend_audio(
                    self.backend,
                    session_id=session_id,
                    audio_path=audio_path,
                    audio_length_ms=audio_length_ms,
                    difficulty=difficulty,
                    route=route,
                )
            except InferenceError as exc:
                if self.sessions.get(session_id) is session:
                    enforce_session_transition("prepare_failed", session, session_id=session_id)
                    self.sessions.pop(session_id, None)
                    await self.backend.reset_session(session_id)
                log_ws_status(
                    session_id=session_id,
                    from_status="audio_preparing",
                    to_status="failed",
                    reason=exc.code,
                    phase=exc.phase,
                    route=route,
                    audio_path=str(audio_path),
                    audio_length_ms=audio_length_ms,
                    difficulty=difficulty,
                )
                raise
            except BaseException:
                if self.sessions.get(session_id) is session:
                    enforce_session_transition("prepare_failed", session, session_id=session_id)
                    self.sessions.pop(session_id, None)
                    await self.backend.reset_session(session_id)
                raise
            enforce_session_transition("audio_prepared", session, session_id=session_id)
            session.audio_prepared = True
            log_ws_status(
                session_id=session_id,
                from_status="audio_preparing",
                to_status="audio_ready",
                reason="audio_prepared",
                audio_path=str(audio_path),
                audio_length_ms=session.audio_length_ms,
                difficulty=difficulty,
                route=route,
            )

    async def _handle_reference_time(
        self,
        command: ReferenceTimeCommand,
        peer: EndpointPeer,
        *,
        owner: object | None = None,
    ) -> None:
        session_id = _require_command_session_id(command.session_id)
        async with self._session_scope(session_id):
            session = self.sessions.get(session_id)
            if session is not None and owner is not None and session.owner is not owner:
                raise ProtocolError("session is owned by another websocket connection")
            enforce_session_transition("reference_time", session, session_id=session_id)
            if session is None or session.audio_path is None:
                raise ProtocolError("send audio before reference_time")

            clock = _reference_clock_from_command(command)
            message_audio_length_ms = _validated_optional_audio_length_ms(command.audio_length_ms)
            if message_audio_length_ms is not None:
                if session.audio_length_ms is not None and message_audio_length_ms != session.audio_length_ms:
                    raise ProtocolError("audio_length_ms must match the prepared audio_length_ms")
                session.audio_length_ms = message_audio_length_ms
            if session.audio_length_ms is None:
                raise ProtocolError("audio_length_ms is required or must be readable from audio_path")
            audio_length_ms = session.audio_length_ms
            from_status = session_status(session)
            window_policy = decoder_window_policy_for_route(session.route, self.config)
            window = clamp_decoder_window_for_policy(
                choose_decoder_window(
                    clock,
                    self.config,
                    decoder_lead_ms=window_policy.decoder_lead_ms,
                    align_to_decoder_window=window_policy.align_to_decoder_window,
                ),
                policy=window_policy,
                audio_length_ms=audio_length_ms,
                config=self.config,
            )
            session.reference_clock = clock
            session.decoder_window = window
            await _cancel_task(session.stream_task)
            await _cancel_task(session.wall_clock_reset_task)
            session.stream_task = asyncio.create_task(self._stream_tokens(session, window, peer))
            session.wall_clock_reset_task = asyncio.create_task(self._reset_after_audio_end(session))
            reset_local_host_time_ms = audio_end_reset_host_time_ms(
                reference_clock=clock,
                audio_length_ms=audio_length_ms,
                reset_after_audio_end_ms=self.config.reset_after_audio_end_ms,
            )
            log_ws_status(
                session_id=session_id,
                from_status=from_status,
                to_status="streaming",
                reason="reference_time",
                ref_time_ms=clock.ref_time_ms,
                send_local_host_time_ms=clock.local_host_time_send_ms,
                received_local_host_time_ms=clock.received_local_host_time_ms,
                audio_length_ms=audio_length_ms,
                difficulty=session.difficulty,
                route=session.route,
                reset_local_host_time_ms=reset_local_host_time_ms,
            )

    async def _reset_after_audio_end(self, session: SessionState) -> None:
        if session.reference_clock is None or session.audio_length_ms is None:
            return
        reset_local_host_time_ms = audio_end_reset_host_time_ms(
            reference_clock=session.reference_clock,
            audio_length_ms=session.audio_length_ms,
            reset_after_audio_end_ms=self.config.reset_after_audio_end_ms,
        )
        check_interval_s = max(0.01, float(self.config.wall_clock_check_interval_s))
        while self.sessions.get(session.session_id) is session:
            if host_time_ms_reached(reset_local_host_time_ms):
                await self.stop_session(
                    session.session_id,
                    reason="wall_clock_audio_end",
                    owner=session.owner,
                )
                return
            await asyncio.sleep(check_interval_s)

    async def _stream_tokens(self, session: SessionState, window: DecoderWindow, peer: EndpointPeer) -> None:
        assert self.backend is not None
        assert session.audio_path is not None
        if session.audio_length_ms is None:
            raise RuntimeError("audio duration must be resolved before streaming")
        try:
            async for hitobject in self.backend.iter_hitobject_tokens(
                session_id=session.session_id,
                audio_path=session.audio_path,
                audio_length_ms=session.audio_length_ms,
                window=window,
            ):
                if self.sessions.get(session.session_id) is not session:
                    return
                if int(hitobject.ms_in_ref_audio) >= int(session.audio_length_ms):
                    continue
                await peer.send_event(
                    HitObjectTokenEvent(
                        session_id=session.session_id,
                        token_id=int(hitobject.token_id),
                        ms_in_ref_audio=int(hitobject.ms_in_ref_audio),
                    ),
                )
            if self.sessions.get(session.session_id) is session:
                await peer.send_event(
                    EndOfStreamEvent(
                        session_id=session.session_id,
                        audio_length_ms=session.audio_length_ms,
                        complete_through_ms=session.audio_length_ms,
                    ),
                )
        except PeerDisconnected:
            await self.stop_session(
                session.session_id,
                reason="peer_disconnect",
                owner=session.owner,
                transition="peer_disconnect",
            )
            return
        except Exception as exc:
            if _is_expected_socket_disconnect(exc):
                await self.stop_session(
                    session.session_id,
                    reason="peer_disconnect",
                    owner=session.owner,
                    transition="peer_disconnect",
                )
                return
            if self.sessions.get(session.session_id) is session:
                try:
                    inference_error = _inference_error_from_exception(
                        exc,
                        session_id=session.session_id,
                        phase="stream",
                        route=session.route,
                    )
                    await peer.send_event(inference_error.to_service_event())
                except PeerDisconnected:
                    await self.stop_session(
                        session.session_id,
                        reason="peer_disconnect",
                        owner=session.owner,
                        transition="peer_disconnect",
                    )
                    return
                except Exception as send_exc:
                    if _is_expected_socket_disconnect(send_exc):
                        await self.stop_session(
                            session.session_id,
                            reason="peer_disconnect",
                            owner=session.owner,
                            transition="peer_disconnect",
                        )
                        return
                    raise

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
            if entry.ref_count == 0 and session_id not in self.sessions:
                self._session_locks.pop(session_id, None)


def infer_message_type(message: Mapping[str, Any]) -> str:
    raw_type = message.get("type")
    if isinstance(raw_type, str) and raw_type:
        if raw_type in {"ready", "node_hello", "audio", "reference_time", "stop"}:
            return raw_type
        raise ProtocolError(f"unsupported message type: {raw_type!r}")
    raise ProtocolError("message must include type")


def session_transition_target(transition: SessionTransition, from_status: SessionStatus) -> SessionStatus:
    rule = SESSION_TRANSITION_RULES[transition]
    if from_status not in rule.allowed_from:
        raise ProtocolError(
            "invalid session transition "
            f"{transition!r} from {from_status!r}; expected one of {_format_statuses(rule.allowed_from)}",
        )
    return rule.to_status


def enforce_session_transition(
    transition: SessionTransition,
    session: SessionState | None,
    *,
    session_id: str,
) -> SessionStatus:
    from_status = session_status(session)
    try:
        session_transition_target(transition, from_status)
    except ProtocolError as exc:
        raise ProtocolError(f"session {session_id!r}: {exc}") from exc
    return from_status


def _format_statuses(statuses: frozenset[SessionStatus]) -> str:
    ordered = [status for status in _SESSION_STATUS_ORDER if status in statuses]
    return ", ".join(ordered)


def require_session_id(message: Mapping[str, Any]) -> str:
    session_id = message.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ProtocolError("session_id must be a non-empty string")
    return session_id


def _require_command_session_id(session_id: str) -> str:
    if not isinstance(session_id, str) or not session_id.strip():
        raise ProtocolError("session_id must be a non-empty string")
    return session_id


def audio_path_from_message(message: Mapping[str, Any]) -> str | None:
    value = message.get("audio_path")
    if isinstance(value, str):
        return value
    return None


def inference_route_from_message(message: Mapping[str, Any]) -> InferenceRoute:
    route = _optional_route_from_message(message)
    if route is not None:
        return route
    return "mapper"


def reference_clock_from_message(message: Mapping[str, Any]) -> ReferenceClock:
    return ReferenceClock(
        ref_time_ms=_required_int(message, "ref_time_ms"),
        local_host_time_send_ms=_required_float(message, "local_host_time_send_ms"),
        received_local_host_time_ms=current_host_time_ms(),
    )


def _reference_clock_from_command(command: ReferenceTimeCommand) -> ReferenceClock:
    return ReferenceClock(
        ref_time_ms=_validated_required_int(command.ref_time_ms, "ref_time_ms"),
        local_host_time_send_ms=_validated_required_float(
            command.local_host_time_send_ms,
            "local_host_time_send_ms",
        ),
        received_local_host_time_ms=current_host_time_ms(),
    )


def stop_session_reason_from_message(message: Mapping[str, Any]) -> str:
    reason = message.get("reason")
    return _validated_stop_reason(reason)


def _validated_stop_reason(reason: object) -> str:
    if reason is None:
        return "client_stop"
    if not isinstance(reason, str) or not reason.strip():
        raise ProtocolError("stop reason must be a non-empty string")
    return reason


@dataclass(frozen=True)
class DecoderWindowPolicy:
    decoder_lead_ms: int
    align_to_decoder_window: bool


def decoder_window_policy_for_route(route: InferenceRoute, config: WsEndpointConfig) -> DecoderWindowPolicy:
    if route == "timing_mock":
        return DecoderWindowPolicy(
            decoder_lead_ms=int(config.timing_mock_decoder_lead_ms),
            align_to_decoder_window=bool(config.timing_mock_align_decoder_window),
        )
    return DecoderWindowPolicy(
        decoder_lead_ms=int(config.decoder_lead_ms),
        align_to_decoder_window=True,
    )


def clamp_decoder_window_for_policy(
    window: DecoderWindow,
    *,
    policy: DecoderWindowPolicy,
    audio_length_ms: int,
    config: WsEndpointConfig,
) -> DecoderWindow:
    if policy.align_to_decoder_window:
        return clamp_decoder_window_to_audio(window, audio_length_ms=audio_length_ms, config=config)

    window_ms = int(config.decoder_window_ms)
    if window_ms <= 0:
        raise ValueError("decoder_window_ms must be positive")
    start_ms = min(max(0, int(window.start_ms)), int(audio_length_ms))
    return DecoderWindow(start_ms=start_ms, end_ms=start_ms + window_ms)


def choose_decoder_window(
    clock: ReferenceClock,
    config: WsEndpointConfig,
    *,
    decoder_lead_ms: int | None = None,
    align_to_decoder_window: bool = True,
) -> DecoderWindow:
    elapsed_ms = max(0.0, clock.received_local_host_time_ms - clock.local_host_time_send_ms)
    estimated_ref_ms = max(0, clock.ref_time_ms + elapsed_ms)
    lead_ms = config.decoder_lead_ms if decoder_lead_ms is None else decoder_lead_ms
    target_ms = estimated_ref_ms + max(0, int(lead_ms))
    window_ms = int(config.decoder_window_ms)
    if window_ms <= 0:
        raise ValueError("decoder_window_ms must be positive")
    if align_to_decoder_window:
        start_ms = int((target_ms + window_ms - 1) // window_ms) * window_ms
    else:
        start_ms = int(target_ms)
    return DecoderWindow(start_ms=start_ms, end_ms=start_ms + window_ms)


def current_host_time_ms() -> float:
    return time.monotonic() * 1000.0


def audio_end_reset_host_time_ms(
    *,
    reference_clock: ReferenceClock,
    audio_length_ms: int,
    reset_after_audio_end_ms: int,
) -> float:
    remaining_audio_ms = max(0, int(audio_length_ms) - int(reference_clock.ref_time_ms))
    return float(reference_clock.local_host_time_send_ms) + remaining_audio_ms + max(
        0.0,
        float(reset_after_audio_end_ms),
    )


def host_time_ms_reached(deadline_ms: float, now_ms: float | None = None) -> bool:
    now_ms = current_host_time_ms() if now_ms is None else float(now_ms)
    return now_ms >= float(deadline_ms)


def ws_status_log_payload(
    *,
    session_id: str | None,
    from_status: str,
    to_status: str,
    reason: str,
    **fields: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event": "ws_status",
        "local_host_time_ms": current_host_time_ms(),
        "session_id": session_id,
        "from": from_status,
        "to": to_status,
        "reason": reason,
    }
    for key, value in fields.items():
        if value is None:
            continue
        payload[key] = value
    return payload


def log_ws_status(
    *,
    session_id: str | None,
    from_status: str,
    to_status: str,
    reason: str,
    **fields: Any,
) -> None:
    payload = ws_status_log_payload(
        session_id=session_id,
        from_status=from_status,
        to_status=to_status,
        reason=reason,
        **fields,
    )
    print(f"ws_status {json.dumps(payload, separators=(',', ':'), ensure_ascii=False)}", flush=True)


def audio_length_ms_from_message(message: Mapping[str, Any]) -> int | None:
    value = message.get("audio_length_ms")
    return _validated_optional_audio_length_ms(value)


def _validated_optional_audio_length_ms(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError("audio_length_ms must be an integer")
    value = int(value)
    if value <= 0:
        raise ProtocolError("audio_length_ms must be positive")
    return value


def difficulty_from_message(message: Mapping[str, Any], *, default: float) -> float:
    value = message.get("difficulty", default)
    return _validated_difficulty(value, default=default)


def _validated_difficulty(value: object, *, default: float) -> float:
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError("difficulty must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ProtocolError("difficulty must be finite")
    try:
        normalize_difficulty(value)
    except ValueError as exc:
        raise ProtocolError(str(exc)) from exc
    return value


def _optional_route_from_message(message: Mapping[str, Any]) -> InferenceRoute | None:
    raw_route = message.get("route")
    if raw_route is None:
        return None
    if not isinstance(raw_route, str):
        raise ProtocolError("route must be a string")
    route = raw_route.strip().lower().replace("-", "_")
    if route in {"mapper", "inference_route_mapper"}:
        return "mapper"
    if route in {"timing_mock", "inference_route_timing_mock"}:
        return "timing_mock"
    raise ProtocolError(f"unsupported inference route: {raw_route!r}")


def _required_int(message: Mapping[str, Any], key: str) -> int:
    value = message.get(key)
    return _validated_required_int(value, key)


def _validated_required_int(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{key} must be an integer")
    value = int(value)
    if value < 0:
        raise ProtocolError(f"{key} must be non-negative")
    return value


def _required_float(message: Mapping[str, Any], key: str) -> float:
    value = message.get(key)
    return _validated_required_float(value, key)


def _validated_required_float(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{key} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ProtocolError(f"{key} must be finite and non-negative")
    return value


async def _prepare_backend_audio(
    backend: EndpointBackend,
    *,
    session_id: str,
    audio_path: Path,
    audio_length_ms: int,
    difficulty: float | None,
    route: InferenceRoute,
) -> None:
    kwargs: dict[str, Any] = {
        "session_id": session_id,
        "audio_path": audio_path,
        "audio_length_ms": audio_length_ms,
        "difficulty": difficulty,
    }
    if _call_accepts_keyword(backend.prepare_audio, "route"):
        kwargs["route"] = route
    try:
        await backend.prepare_audio(**kwargs)
    except InferenceError:
        raise
    except Exception as exc:
        raise _inference_error_from_exception(
            exc,
            session_id=session_id,
            phase="prepare_audio",
            route=route,
        ) from exc


def _inference_error_from_exception(
    exc: Exception,
    *,
    session_id: str,
    phase: str,
    route: InferenceRoute,
) -> InferenceError:
    return InferenceError(
        _inference_error_message(exc),
        session_id=session_id,
        phase=phase,
        route=route,
        code=_inference_error_code(exc),
    )


def _inference_error_message(exc: Exception) -> str:
    detail = str(exc).strip()
    name = type(exc).__name__
    return f"{name}: {detail}" if detail else name


def _inference_error_code(exc: Exception) -> str:
    message = _inference_error_message(exc).lower()
    if "device" in message and ("expected one of" in message or "invalid" in message or "auto" in message):
        return "invalid_device"
    if isinstance(exc, FileNotFoundError):
        return "audio_not_found"
    return "inference_failed"


def _call_accepts_keyword(callable_obj: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    return keyword in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


async def _cancel_task(task: asyncio.Task[None] | None) -> None:
    if task is None or task.done() or task is asyncio.current_task():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
