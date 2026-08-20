from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from pulsefield_model.timing.grid_fitting import TimingFitResult
from pulsefield_model.timing.schema import FittedTimingGrid, FrameTimingPrediction
from pulsefield_model.timing.v3.analytic_curve import PhaseContinuousTimingCurve
from pulsefield_model.timing.v3.audio_evidence import (
    RawAudioEvidence,
    extract_raw_audio_evidence,
)
from pulsefield_model.timing.v3.tempo_track import (
    TempoTrackResult,
    generate_timing_candidates,
)


TIMING_MODE_V2 = "v2"
TIMING_MODE_V3_SHADOW = "v3_shadow"
TIMING_INFERENCE_MODES = (TIMING_MODE_V2, TIMING_MODE_V3_SHADOW)
DEFAULT_TIMING_MAX_SUPPORTED_AUDIO_DURATION_SECONDS = 600.0
TIMING_V3_OUTCOME_SCHEMA_VERSION = "pulsefield_model.timing_v3_shadow_outcome_v1"

TimingInferenceMode: TypeAlias = Literal["v2", "v3_shadow"]
TimingV3ShadowStatus: TypeAlias = Literal[
    "disabled",
    "completed",
    "skipped_duration",
    "failed",
]


class TempoTrackGenerator(Protocol):
    def __call__(
        self,
        prediction: FrameTimingPrediction,
        *,
        audio_evidence: RawAudioEvidence | None = None,
    ) -> TempoTrackResult:
        ...


class TimingV3Facade(Protocol):
    def __call__(
        self,
        evidence: TimingEvidenceBundle,
        *,
        v2_fallback_fit: TimingFitResult,
        mode: TimingInferenceMode,
        max_supported_audio_duration_seconds: float,
    ) -> TimingV3Outcome:
        ...


@dataclass(frozen=True)
class TimingEvidenceBundle:
    """Inputs currently available to the production Timing-v3 shadow path.

    ``beatthis_frame_probabilities`` names the post-sigmoid provider output
    precisely. It is not a logits contract; preserving BeatThis logits remains
    a separate provider API change. ``raw_audio_log_mel_10ms`` is the existing
    16 kHz, 10 ms-hop, 80-bin float32 feature view used by raw-audio evidence.
    """

    beatthis_frame_probabilities: FrameTimingPrediction
    audio_duration_seconds: float
    raw_audio_log_mel_10ms: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.beatthis_frame_probabilities, FrameTimingPrediction):
            raise TypeError(
                "beatthis_frame_probabilities must be a FrameTimingPrediction",
            )
        duration = _positive_finite_float(
            self.audio_duration_seconds,
            "audio_duration_seconds",
        )
        mel = self.raw_audio_log_mel_10ms
        if mel is not None:
            mel = _require_log_mel_10ms(mel)
        object.__setattr__(self, "audio_duration_seconds", duration)
        object.__setattr__(self, "raw_audio_log_mel_10ms", mel)


@dataclass(frozen=True)
class TimingV3Telemetry:
    mode: TimingInferenceMode
    status: TimingV3ShadowStatus
    elapsed_ms: float
    candidate_count: int
    selection_status: Literal["v3_accepted", "v2_fallback"] | None
    fallback_reason: str | None
    selected_fingerprint_sha256: str | None
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        _require_timing_mode(self.mode)
        if self.status not in {
            "disabled",
            "completed",
            "skipped_duration",
            "failed",
        }:
            raise ValueError(f"unsupported Timing-v3 telemetry status: {self.status!r}")
        elapsed_ms = _nonnegative_finite_float(self.elapsed_ms, "elapsed_ms")
        if isinstance(self.candidate_count, bool) or not isinstance(self.candidate_count, int):
            raise TypeError("candidate_count must be an integer")
        if self.candidate_count < 0:
            raise ValueError("candidate_count must be non-negative")
        if self.selection_status not in {None, "v3_accepted", "v2_fallback"}:
            raise ValueError(f"unsupported selection_status: {self.selection_status!r}")
        if self.status != "failed" and (self.error_type is not None or self.error_message is not None):
            raise ValueError("only failed telemetry may carry error details")
        if self.status == "failed" and not self.error_type:
            raise ValueError("failed telemetry must carry error_type")
        object.__setattr__(self, "elapsed_ms", elapsed_ms)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "status": self.status,
            "elapsed_ms": self.elapsed_ms,
            "candidate_count": self.candidate_count,
            "selection_status": self.selection_status,
            "fallback_reason": self.fallback_reason,
            "selected_fingerprint_sha256": self.selected_fingerprint_sha256,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class TimingV3Outcome:
    """Non-authoritative Timing-v3 shadow result with an authoritative v2 fallback."""

    mode: TimingInferenceMode
    v2_fallback_fit: TimingFitResult
    telemetry: TimingV3Telemetry
    shadow_result: TempoTrackResult | None = None
    _selected_curve_canonical_bytes: bytes | None = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _require_timing_mode(self.mode)
        if not isinstance(self.v2_fallback_fit, TimingFitResult):
            raise TypeError("v2_fallback_fit must be a TimingFitResult")
        if not isinstance(self.telemetry, TimingV3Telemetry):
            raise TypeError("telemetry must be TimingV3Telemetry")
        if self.telemetry.mode != self.mode:
            raise ValueError("telemetry mode must match outcome mode")
        if self.telemetry.status == "completed":
            if not isinstance(self.shadow_result, TempoTrackResult):
                raise ValueError("completed shadow outcome must carry TempoTrackResult")
        elif self.shadow_result is not None:
            raise ValueError("non-completed shadow outcome cannot carry TempoTrackResult")

        curve = self.selected_shadow_curve
        canonical_bytes: bytes | None = None
        if curve is not None:
            canonical_bytes = curve.canonical_bytes()
            restored = PhaseContinuousTimingCurve.from_canonical_bytes(canonical_bytes)
            if restored.canonical_bytes() != canonical_bytes:
                raise ValueError("selected Timing-v3 curve failed canonical round trip")
            if self.telemetry.selected_fingerprint_sha256 != curve.fingerprint_sha256:
                raise ValueError("selected curve fingerprint does not match telemetry")
        elif self.telemetry.selected_fingerprint_sha256 is not None:
            raise ValueError("telemetry cannot select a fingerprint without a shadow curve")
        object.__setattr__(self, "_selected_curve_canonical_bytes", canonical_bytes)

    @property
    def live_timing_grid(self) -> FittedTimingGrid:
        """The live grid remains v2 for both supported modes."""

        return self.v2_fallback_fit.grid

    @property
    def selected_shadow_curve(self) -> PhaseContinuousTimingCurve | None:
        result = self.shadow_result
        if result is None or result.production_selection is None:
            return None
        if result.production_selection.status != "v3_accepted":
            return None
        curve = result.selected_candidate
        if curve is None:
            raise ValueError("accepted Timing-v3 selection is missing its selected curve")
        return curve

    @property
    def selected_curve_canonical_bytes(self) -> bytes | None:
        return self._selected_curve_canonical_bytes

    def to_observable_dict(self) -> dict[str, object]:
        curve = self.selected_shadow_curve
        return {
            "schema": TIMING_V3_OUTCOME_SCHEMA_VERSION,
            "mode": self.mode,
            "live_timing": "v2",
            "v2_fallback": {
                "score": float(self.v2_fallback_fit.score),
                "segment_count": len(self.v2_fallback_fit.grid.segments),
            },
            "telemetry": self.telemetry.to_dict(),
            "selected_shadow_curve": None if curve is None else curve.to_dict(),
        }


def run_timing_v3_shadow(
    evidence: TimingEvidenceBundle,
    *,
    v2_fallback_fit: TimingFitResult,
    mode: TimingInferenceMode = TIMING_MODE_V2,
    max_supported_audio_duration_seconds: float = (
        DEFAULT_TIMING_MAX_SUPPORTED_AUDIO_DURATION_SECONDS
    ),
    candidate_generator: TempoTrackGenerator = generate_timing_candidates,
    clock: Callable[[], float] = time.perf_counter,
) -> TimingV3Outcome:
    """Run current Timing v3 as a non-authoritative shadow beside v2."""

    if not isinstance(evidence, TimingEvidenceBundle):
        raise TypeError("evidence must be TimingEvidenceBundle")
    if not isinstance(v2_fallback_fit, TimingFitResult):
        raise TypeError("v2_fallback_fit must be a TimingFitResult")
    resolved_mode = _require_timing_mode(mode)
    maximum_duration = _positive_finite_float(
        max_supported_audio_duration_seconds,
        "max_supported_audio_duration_seconds",
    )
    if not callable(candidate_generator):
        raise TypeError("candidate_generator must be callable")
    if not callable(clock):
        raise TypeError("clock must be callable")

    if resolved_mode == TIMING_MODE_V2:
        return TimingV3Outcome(
            mode=resolved_mode,
            v2_fallback_fit=v2_fallback_fit,
            telemetry=TimingV3Telemetry(
                mode=resolved_mode,
                status="disabled",
                elapsed_ms=0.0,
                candidate_count=0,
                selection_status=None,
                fallback_reason=None,
                selected_fingerprint_sha256=None,
            ),
        )
    if evidence.audio_duration_seconds > maximum_duration:
        return TimingV3Outcome(
            mode=resolved_mode,
            v2_fallback_fit=v2_fallback_fit,
            telemetry=TimingV3Telemetry(
                mode=resolved_mode,
                status="skipped_duration",
                elapsed_ms=0.0,
                candidate_count=0,
                selection_status=None,
                fallback_reason="audio_duration_exceeds_timing_v3_limit",
                selected_fingerprint_sha256=None,
            ),
        )

    started = float(clock())
    raw_evidence = (
        None
        if evidence.raw_audio_log_mel_10ms is None
        else extract_raw_audio_evidence(
            evidence.raw_audio_log_mel_10ms,
            audio_duration_seconds=evidence.audio_duration_seconds,
        )
    )
    result = candidate_generator(
        evidence.beatthis_frame_probabilities,
        audio_evidence=raw_evidence,
    )
    if not isinstance(result, TempoTrackResult):
        raise TypeError("candidate_generator must return TempoTrackResult")
    selection = result.production_selection
    selection_status = None if selection is None else selection.status
    fallback_reason = (
        "production_selection_missing"
        if selection is None
        else selection.fallback_reason
    )
    selected_fingerprint = (
        selection.selected_fingerprint_sha256
        if selection is not None and selection.status == "v3_accepted"
        else None
    )
    return TimingV3Outcome(
        mode=resolved_mode,
        v2_fallback_fit=v2_fallback_fit,
        shadow_result=result,
        telemetry=TimingV3Telemetry(
            mode=resolved_mode,
            status="completed",
            elapsed_ms=_elapsed_ms(started, clock),
            candidate_count=len(result.candidates),
            selection_status=selection_status,
            fallback_reason=fallback_reason,
            selected_fingerprint_sha256=selected_fingerprint,
        ),
    )


def unpack_packed_mel_20ms_to_log_mel_10ms(
    packed_mel_20ms: object,
) -> NDArray[np.float32]:
    """Invert the existing [first-10ms, second-10ms] channel packing."""

    if not isinstance(packed_mel_20ms, np.ndarray):
        raise TypeError("packed_mel_20ms must be a NumPy array")
    if packed_mel_20ms.dtype != np.float32:
        raise TypeError(
            f"packed_mel_20ms must have dtype float32, got {packed_mel_20ms.dtype}",
        )
    if packed_mel_20ms.ndim != 2 or packed_mel_20ms.shape[1] != 160:
        raise ValueError(
            f"packed_mel_20ms must have shape [frames, 160], got {packed_mel_20ms.shape}",
        )
    if not np.all(np.isfinite(packed_mel_20ms)):
        raise ValueError("packed_mel_20ms must contain only finite values")
    unpacked = np.empty((packed_mel_20ms.shape[0] * 2, 80), dtype=np.float32)
    unpacked[0::2] = packed_mel_20ms[:, :80]
    unpacked[1::2] = packed_mel_20ms[:, 80:]
    return unpacked


def _require_log_mel_10ms(value: object) -> NDArray[np.float32]:
    if not isinstance(value, np.ndarray):
        raise TypeError("raw_audio_log_mel_10ms must be a NumPy array")
    if value.dtype != np.float32:
        raise TypeError(
            f"raw_audio_log_mel_10ms must have dtype float32, got {value.dtype}",
        )
    if value.ndim != 2 or value.shape[1] != 80:
        raise ValueError(
            f"raw_audio_log_mel_10ms must have shape [frames, 80], got {value.shape}",
        )
    if not np.all(np.isfinite(value)):
        raise ValueError("raw_audio_log_mel_10ms must contain only finite values")
    return value


def _require_timing_mode(value: object) -> TimingInferenceMode:
    if value not in TIMING_INFERENCE_MODES:
        raise ValueError(
            f"timing mode must be one of {TIMING_INFERENCE_MODES}, got {value!r}",
        )
    return value  # type: ignore[return-value]


def _positive_finite_float(value: object, name: str) -> float:
    number = _nonnegative_finite_float(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _nonnegative_finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _elapsed_ms(started: float, clock: Callable[[], float]) -> float:
    finished = float(clock())
    if not math.isfinite(started) or not math.isfinite(finished):
        raise ValueError("clock must return finite values")
    return max(0.0, 1000.0 * (finished - started))


__all__ = [
    "DEFAULT_TIMING_MAX_SUPPORTED_AUDIO_DURATION_SECONDS",
    "TIMING_INFERENCE_MODES",
    "TIMING_MODE_V2",
    "TIMING_MODE_V3_SHADOW",
    "TIMING_V3_OUTCOME_SCHEMA_VERSION",
    "TempoTrackGenerator",
    "TimingEvidenceBundle",
    "TimingInferenceMode",
    "TimingV3Facade",
    "TimingV3Outcome",
    "TimingV3ShadowStatus",
    "TimingV3Telemetry",
    "run_timing_v3_shadow",
    "unpack_packed_mel_20ms_to_log_mel_10ms",
]
