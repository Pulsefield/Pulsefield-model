from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, cast

from pulsefield_model.timing.grid_fitting import GridFitter, GridFitterConfig
from pulsefield_model.timing.grid_fitting.types import TimingFitResult
from pulsefield_model.timing.schema import FrameTimingPrediction
from pulsefield_model.timing.v3.projection import (
    DEFAULT_MAX_BPM,
    DEFAULT_MAX_RELATIVE_BPM_ADJUSTMENT,
    DEFAULT_MIN_BPM,
    PROJECTION_METHOD_PRESERVE_ANCHORS,
    PROJECTION_METHOD_PRESERVE_BPM,
    TimingV3ProjectionResult,
    project_preserve_anchors,
    project_preserve_bpm,
)
from pulsefield_model.timing.v3.schema import TimingV3Grid


class V2TimingGridFitter(Protocol):
    def fit(self, prediction: FrameTimingPrediction) -> TimingFitResult:
        ...


@dataclass(frozen=True)
class TimingV3FitterConfig:
    v2_grid_fitter_config: GridFitterConfig = GridFitterConfig()
    min_bpm: float = DEFAULT_MIN_BPM
    max_bpm: float = DEFAULT_MAX_BPM
    max_relative_bpm_adjustment: float = DEFAULT_MAX_RELATIVE_BPM_ADJUSTMENT

    def __post_init__(self) -> None:
        if not isinstance(self.v2_grid_fitter_config, GridFitterConfig):
            raise TypeError("v2_grid_fitter_config must be a GridFitterConfig")
        if not math.isfinite(self.min_bpm) or self.min_bpm <= 0.0:
            raise ValueError(f"min_bpm must be positive and finite, got {self.min_bpm!r}")
        if not math.isfinite(self.max_bpm) or self.max_bpm < self.min_bpm:
            raise ValueError(f"max_bpm must be finite and >= min_bpm, got {self.max_bpm!r}")
        if not math.isfinite(self.max_relative_bpm_adjustment) or self.max_relative_bpm_adjustment < 0.0:
            raise ValueError(
                "max_relative_bpm_adjustment must be non-negative and finite, "
                f"got {self.max_relative_bpm_adjustment!r}",
            )


@dataclass(frozen=True)
class TimingV3FitResult:
    v2_fit: TimingFitResult
    selected_projection: TimingV3ProjectionResult
    control_projection: TimingV3ProjectionResult
    coverage_start_ms: float
    coverage_end_ms: float
    fallback_v2: bool
    reason: str | None

    def __post_init__(self) -> None:
        if self.selected_projection.method != PROJECTION_METHOD_PRESERVE_ANCHORS:
            raise ValueError("selected_projection must be the Family B preserve-anchors projection")
        if self.control_projection.method != PROJECTION_METHOD_PRESERVE_BPM:
            raise ValueError("control_projection must be the Family A preserve-BPM projection")
        expected_fallback = not self.selected_projection.ok
        if self.fallback_v2 != expected_fallback:
            raise ValueError("fallback_v2 must reflect selected_projection.ok")
        expected_reason = self.selected_projection.reason if expected_fallback else None
        if self.reason != expected_reason:
            raise ValueError("reason must be the selected projection failure reason")
        if not math.isfinite(self.coverage_start_ms):
            raise ValueError(f"coverage_start_ms must be finite, got {self.coverage_start_ms!r}")
        if not math.isfinite(self.coverage_end_ms) or self.coverage_end_ms <= self.coverage_start_ms:
            raise ValueError(
                "coverage_end_ms must be finite and greater than coverage_start_ms, "
                f"got {self.coverage_end_ms!r} <= {self.coverage_start_ms!r}",
            )

    @property
    def selected_v3_grid(self) -> TimingV3Grid | None:
        if self.fallback_v2:
            return None
        return cast(TimingV3Grid, self.selected_projection.grid)


class TimingV3Fitter:
    def __init__(
        self,
        config: TimingV3FitterConfig = TimingV3FitterConfig(),
        *,
        v2_fitter: V2TimingGridFitter | None = None,
    ) -> None:
        self.config = config
        self.v2_fitter = (
            v2_fitter
            if v2_fitter is not None
            else GridFitter(config.v2_grid_fitter_config)
        )

    def fit(self, prediction: FrameTimingPrediction) -> TimingV3FitResult:
        v2_fit = self.v2_fitter.fit(prediction)
        coverage_start_ms = 0.0
        coverage_end_ms = 1000.0 * prediction.frame_count / prediction.frame_rate_hz
        projection_kwargs = {
            "coverage_start_ms": coverage_start_ms,
            "coverage_end_ms": coverage_end_ms,
            "min_bpm": self.config.min_bpm,
            "max_bpm": self.config.max_bpm,
            "max_relative_bpm_adjustment": self.config.max_relative_bpm_adjustment,
        }
        selected_projection = project_preserve_anchors(v2_fit.grid, **projection_kwargs)
        control_projection = project_preserve_bpm(v2_fit.grid, **projection_kwargs)
        fallback_v2 = not selected_projection.ok
        return TimingV3FitResult(
            v2_fit=v2_fit,
            selected_projection=selected_projection,
            control_projection=control_projection,
            coverage_start_ms=coverage_start_ms,
            coverage_end_ms=coverage_end_ms,
            fallback_v2=fallback_v2,
            reason=selected_projection.reason if fallback_v2 else None,
        )


__all__ = [
    "TimingV3FitResult",
    "TimingV3Fitter",
    "TimingV3FitterConfig",
    "V2TimingGridFitter",
]
