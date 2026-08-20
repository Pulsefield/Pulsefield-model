from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from pulsefield_model.timing.evaluation.drift import compare_timing_grid_drift
from pulsefield_model.timing.rendering.dense_timing_v2 import (
    DEFAULT_DENSE_TIMING_V2_CONFIG,
    DenseTimingV2Config,
    active_timing_arrays,
    dense_timing_v2_frame_times,
)
from pulsefield_model.timing.schema import FittedTimingGrid
from pulsefield_model.timing.v3.schema import TimingV3Grid


SOURCE_PROJECTION_COMPARISON_SCHEMA = (
    "pulsefield_model.timing_v3_source_projection_comparison_v1"
)


@dataclass(frozen=True)
class TimingV3SourceProjectionComparison:
    """Source-only displacement diagnostics for a v3 projection.

    The source is the stored v2 grid, not an oracle. Wrapped phase metrics
    measure local same-time disagreement; drift metrics retain the unwrapped
    accumulated difference. This object deliberately has no cache, fitter, or
    ``.osu`` dependency.
    """

    frame_count: int
    input_start_ms: float
    frame_hop_ms: float
    source_section_count: int
    candidate_section_count: int
    wrapped_phase_mean_beats: float
    wrapped_phase_rms_beats: float
    wrapped_phase_p90_beats: float
    wrapped_phase_max_beats: float
    wrapped_phase_mean_ms: float
    wrapped_phase_rms_ms: float
    wrapped_phase_p90_ms: float
    wrapped_phase_max_ms: float
    local_bpm_mae: float
    local_bpm_rmse: float
    local_bpm_p90_abs_error: float
    local_bpm_max_abs_error: float
    local_bpm_relative_mae: float
    local_bpm_relative_p90_abs_error: float
    local_bpm_relative_max_abs_error: float
    initial_signed_phase_error_beats: float
    initial_signed_phase_error_ms: float
    endpoint_relative_drift_beats: float
    endpoint_relative_drift_ms: float
    max_abs_prefix_relative_drift_beats: float
    max_abs_prefix_relative_drift_ms: float
    drift_slope_beats_per_minute: float
    drift_slope_ms_per_minute: float
    active_section_disagreement_frame_count: int
    active_section_disagreement_fraction: float
    moved_paired_boundary_count: int
    unmatched_boundary_count: int

    def __post_init__(self) -> None:
        if isinstance(self.frame_count, bool) or self.frame_count <= 0:
            raise ValueError("frame_count must be a positive integer")
        if self.source_section_count <= 0 or self.candidate_section_count <= 0:
            raise ValueError("section counts must be positive")
        if self.active_section_disagreement_frame_count < 0:
            raise ValueError("active_section_disagreement_frame_count must be non-negative")
        if self.moved_paired_boundary_count < 0 or self.unmatched_boundary_count < 0:
            raise ValueError("boundary counts must be non-negative")
        for name, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SOURCE_PROJECTION_COMPARISON_SCHEMA,
            **asdict(self),
        }


def compare_timing_v3_projection_to_source(
    source_grid: FittedTimingGrid,
    candidate_grid: TimingV3Grid | FittedTimingGrid,
    *,
    frame_count: int,
    input_start_ms: float = 0.0,
    frame_hop_ms: float = DEFAULT_DENSE_TIMING_V2_CONFIG.frame_hop_ms,
    frame_center_offset_ms: float | None = None,
) -> TimingV3SourceProjectionComparison:
    """Compare a candidate to v2 over the exact requested frame support."""

    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise ValueError(f"frame_count must be a positive int, got {frame_count!r}")
    input_start_ms = _finite_float(input_start_ms, "input_start_ms")
    frame_hop_ms = _finite_float(frame_hop_ms, "frame_hop_ms")
    if frame_hop_ms <= 0.0:
        raise ValueError("frame_hop_ms must be positive")
    if frame_center_offset_ms is None:
        frame_center_offset_ms = frame_hop_ms / 2.0
    frame_center_offset_ms = _finite_float(
        frame_center_offset_ms,
        "frame_center_offset_ms",
    )

    projected_grid = (
        candidate_grid.to_fitted_timing_grid()
        if isinstance(candidate_grid, TimingV3Grid)
        else candidate_grid
    )
    if not isinstance(projected_grid, FittedTimingGrid):
        raise TypeError("candidate_grid must be TimingV3Grid or FittedTimingGrid")

    dense_config = DenseTimingV2Config(
        frame_hop_ms=frame_hop_ms,
        frame_center_offset_ms=frame_center_offset_ms,
        frame_count=frame_count,
        pulse_width_ms=DEFAULT_DENSE_TIMING_V2_CONFIG.pulse_width_ms,
    )
    frame_times_ms = dense_timing_v2_frame_times(
        input_start_ms,
        frame_count=frame_count,
        config=dense_config,
    )
    source_offsets_ms, source_periods_ms = active_timing_arrays(
        source_grid,
        frame_times_ms,
    )
    candidate_offsets_ms, candidate_periods_ms = active_timing_arrays(
        projected_grid,
        frame_times_ms,
    )

    source_phase = (frame_times_ms - source_offsets_ms) / source_periods_ms
    candidate_phase = (frame_times_ms - candidate_offsets_ms) / candidate_periods_ms
    signed_wrapped_phase_beats = np.angle(
        np.exp(2j * np.pi * (candidate_phase - source_phase))
    ) / (2.0 * np.pi)
    wrapped_phase_beats = np.abs(signed_wrapped_phase_beats)
    wrapped_phase_ms = wrapped_phase_beats * source_periods_ms

    source_bpms = 60000.0 / source_periods_ms
    candidate_bpms = 60000.0 / candidate_periods_ms
    bpm_abs_error = np.abs(candidate_bpms - source_bpms)
    bpm_relative_abs_error = bpm_abs_error / source_bpms

    drift = compare_timing_grid_drift(
        projected_grid,
        source_grid,
        frame_count=frame_count,
        input_start_ms=input_start_ms,
        config=dense_config,
    )
    source_active_indices = _active_section_indices(source_grid, frame_times_ms)
    candidate_active_indices = _active_section_indices(projected_grid, frame_times_ms)
    disagreement_count = int(np.count_nonzero(source_active_indices != candidate_active_indices))
    moved_paired_boundary_count, unmatched_boundary_count = _boundary_movement_counts(
        source_grid,
        projected_grid,
    )

    return TimingV3SourceProjectionComparison(
        frame_count=frame_count,
        input_start_ms=input_start_ms,
        frame_hop_ms=frame_hop_ms,
        source_section_count=len(source_grid.segments),
        candidate_section_count=len(projected_grid.segments),
        wrapped_phase_mean_beats=_mean(wrapped_phase_beats),
        wrapped_phase_rms_beats=_rms(wrapped_phase_beats),
        wrapped_phase_p90_beats=_percentile(wrapped_phase_beats, 90.0),
        wrapped_phase_max_beats=_max(wrapped_phase_beats),
        wrapped_phase_mean_ms=_mean(wrapped_phase_ms),
        wrapped_phase_rms_ms=_rms(wrapped_phase_ms),
        wrapped_phase_p90_ms=_percentile(wrapped_phase_ms, 90.0),
        wrapped_phase_max_ms=_max(wrapped_phase_ms),
        local_bpm_mae=_mean(bpm_abs_error),
        local_bpm_rmse=_rms(bpm_abs_error),
        local_bpm_p90_abs_error=_percentile(bpm_abs_error, 90.0),
        local_bpm_max_abs_error=_max(bpm_abs_error),
        local_bpm_relative_mae=_mean(bpm_relative_abs_error),
        local_bpm_relative_p90_abs_error=_percentile(bpm_relative_abs_error, 90.0),
        local_bpm_relative_max_abs_error=_max(bpm_relative_abs_error),
        initial_signed_phase_error_beats=float(drift.initial_signed_phase_error_beats),
        initial_signed_phase_error_ms=float(drift.initial_signed_phase_error_ms),
        endpoint_relative_drift_beats=float(drift.endpoint_relative_drift_beats),
        endpoint_relative_drift_ms=float(drift.endpoint_relative_drift_ms),
        max_abs_prefix_relative_drift_beats=float(
            drift.max_abs_prefix_relative_drift_beats
        ),
        max_abs_prefix_relative_drift_ms=float(drift.max_abs_prefix_relative_drift_ms),
        drift_slope_beats_per_minute=float(drift.drift_slope_beats_per_minute),
        drift_slope_ms_per_minute=float(drift.drift_slope_ms_per_minute),
        active_section_disagreement_frame_count=disagreement_count,
        active_section_disagreement_fraction=float(disagreement_count / frame_count),
        moved_paired_boundary_count=moved_paired_boundary_count,
        unmatched_boundary_count=unmatched_boundary_count,
    )


def _active_section_indices(
    grid: FittedTimingGrid,
    frame_times_ms: np.ndarray,
) -> np.ndarray:
    offsets_ms = np.asarray(
        [segment.offset_ms for segment in grid.segments],
        dtype=np.float64,
    )
    indices = np.searchsorted(offsets_ms, frame_times_ms, side="right") - 1
    return np.clip(indices, 0, len(grid.segments) - 1)


def _boundary_movement_counts(
    source_grid: FittedTimingGrid,
    candidate_grid: FittedTimingGrid,
) -> tuple[int, int]:
    source_boundaries = tuple(float(segment.offset_ms) for segment in source_grid.segments[1:])
    candidate_boundaries = tuple(float(segment.offset_ms) for segment in candidate_grid.segments[1:])
    paired_count = min(len(source_boundaries), len(candidate_boundaries))
    moved_count = sum(
        not math.isclose(
            source_boundaries[index],
            candidate_boundaries[index],
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        for index in range(paired_count)
    )
    unmatched_count = abs(len(source_boundaries) - len(candidate_boundaries))
    return int(moved_count), int(unmatched_count)


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _mean(values: np.ndarray) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _max(values: np.ndarray) -> float:
    return float(np.max(np.asarray(values, dtype=np.float64)))
