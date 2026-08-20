from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pulsefield_model.timing.rendering.dense_timing_v2 import (
    DEFAULT_DENSE_TIMING_V2_CONFIG,
    DenseTimingV2Config,
    active_timing_arrays,
    dense_timing_v2_frame_times,
)
from pulsefield_model.timing.schema import FittedTimingGrid


@dataclass(frozen=True)
class TimingDriftComparison:
    frame_count: int
    duration_seconds: float
    initial_signed_phase_error_beats: float
    initial_signed_phase_error_ms: float
    endpoint_relative_drift_beats: float
    endpoint_relative_drift_ms: float
    max_abs_prefix_relative_drift_beats: float
    max_abs_prefix_relative_drift_ms: float
    drift_slope_beats_per_minute: float
    drift_slope_ms_per_minute: float
    p90_abs_30s_relative_drift_ms: float
    max_abs_30s_relative_drift_ms: float
    p90_abs_60s_relative_drift_ms: float
    max_abs_60s_relative_drift_ms: float
    predicted_boundary_count: int
    mean_predicted_boundary_discontinuity_ms: float
    p90_predicted_boundary_discontinuity_ms: float
    max_predicted_boundary_discontinuity_ms: float


def compare_timing_grid_drift(
    predicted_grid: FittedTimingGrid,
    oracle_grid: FittedTimingGrid,
    *,
    frame_count: int,
    input_start_ms: float = 0.0,
    config: DenseTimingV2Config = DEFAULT_DENSE_TIMING_V2_CONFIG,
) -> TimingDriftComparison:
    """Separate stable phase offset, accumulated drift, and v2 boundary seams.

    Phase is unwrapped at the dense-timing frame rate. This observes gradual
    accumulation as long as the frame-to-frame phase-difference change stays
    below half a beat; that condition is easily met at the default 20 ms hop
    for the supported tempo range. The comparator remains `.osu`-relative and
    therefore inherits its uncertainty.
    """
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count!r}")

    frame_times_ms = dense_timing_v2_frame_times(
        input_start_ms,
        frame_count=frame_count,
        config=config,
    )
    predicted_offsets_ms, predicted_beat_lengths_ms = active_timing_arrays(predicted_grid, frame_times_ms)
    oracle_offsets_ms, oracle_beat_lengths_ms = active_timing_arrays(oracle_grid, frame_times_ms)
    predicted_phase = (frame_times_ms - predicted_offsets_ms) / predicted_beat_lengths_ms
    oracle_phase = (frame_times_ms - oracle_offsets_ms) / oracle_beat_lengths_ms

    wrapped_radians = np.angle(np.exp(2j * np.pi * (predicted_phase - oracle_phase)))
    unwrapped_error_beats = np.unwrap(wrapped_radians) / (2.0 * np.pi)
    relative_drift_beats = unwrapped_error_beats - unwrapped_error_beats[0]
    relative_drift_ms = relative_drift_beats * oracle_beat_lengths_ms
    frame_times_minutes = (frame_times_ms - frame_times_ms[0]) / 60000.0
    duration_seconds = float(
        max(config.frame_hop_ms, frame_times_ms[-1] - frame_times_ms[0] + config.frame_hop_ms) / 1000.0
    )

    drift_slope_beats_per_minute = _linear_slope(frame_times_minutes, relative_drift_beats)
    drift_slope_ms_per_minute = _linear_slope(frame_times_minutes, relative_drift_ms)
    drift_30s = _window_relative_drift_ms(
        unwrapped_error_beats,
        oracle_beat_lengths_ms,
        window_seconds=30.0,
        frame_hop_ms=config.frame_hop_ms,
    )
    drift_60s = _window_relative_drift_ms(
        unwrapped_error_beats,
        oracle_beat_lengths_ms,
        window_seconds=60.0,
        frame_hop_ms=config.frame_hop_ms,
    )
    boundary_discontinuities_ms = predicted_boundary_discontinuities_ms(predicted_grid)

    initial_error_beats = float(unwrapped_error_beats[0])
    return TimingDriftComparison(
        frame_count=frame_count,
        duration_seconds=duration_seconds,
        initial_signed_phase_error_beats=initial_error_beats,
        initial_signed_phase_error_ms=float(initial_error_beats * oracle_beat_lengths_ms[0]),
        endpoint_relative_drift_beats=float(relative_drift_beats[-1]),
        endpoint_relative_drift_ms=float(relative_drift_ms[-1]),
        max_abs_prefix_relative_drift_beats=float(np.max(np.abs(relative_drift_beats))),
        max_abs_prefix_relative_drift_ms=float(np.max(np.abs(relative_drift_ms))),
        drift_slope_beats_per_minute=drift_slope_beats_per_minute,
        drift_slope_ms_per_minute=drift_slope_ms_per_minute,
        p90_abs_30s_relative_drift_ms=_percentile_abs(drift_30s, 90.0),
        max_abs_30s_relative_drift_ms=_max_abs(drift_30s),
        p90_abs_60s_relative_drift_ms=_percentile_abs(drift_60s, 90.0),
        max_abs_60s_relative_drift_ms=_max_abs(drift_60s),
        predicted_boundary_count=len(boundary_discontinuities_ms),
        mean_predicted_boundary_discontinuity_ms=_mean(boundary_discontinuities_ms),
        p90_predicted_boundary_discontinuity_ms=_percentile(boundary_discontinuities_ms, 90.0),
        max_predicted_boundary_discontinuity_ms=max(boundary_discontinuities_ms, default=0.0),
    )


def predicted_boundary_discontinuities_ms(grid: FittedTimingGrid) -> tuple[float, ...]:
    """Return phase distance from each new v2 offset to the preceding lattice."""
    discontinuities: list[float] = []
    for previous, current in zip(grid.segments, grid.segments[1:]):
        phase_beats = (current.offset_ms - previous.offset_ms) / previous.beat_length_ms
        wrapped_beats = phase_beats - math.floor(phase_beats)
        distance_beats = min(wrapped_beats, 1.0 - wrapped_beats)
        discontinuities.append(float(distance_beats * previous.beat_length_ms))
    return tuple(discontinuities)


def _window_relative_drift_ms(
    unwrapped_error_beats: NDArray[np.float64],
    oracle_beat_lengths_ms: NDArray[np.float64],
    *,
    window_seconds: float,
    frame_hop_ms: float,
) -> NDArray[np.float64]:
    window_frames = max(1, int(round(window_seconds * 1000.0 / frame_hop_ms)))
    if unwrapped_error_beats.size <= window_frames:
        return np.asarray([], dtype=np.float64)
    delta_beats = unwrapped_error_beats[window_frames:] - unwrapped_error_beats[:-window_frames]
    return np.asarray(delta_beats * oracle_beat_lengths_ms[window_frames:], dtype=np.float64)


def _linear_slope(xs: NDArray[np.float64], ys: NDArray[np.float64]) -> float:
    if xs.size < 2 or float(xs[-1] - xs[0]) <= 0.0:
        return 0.0
    centered_xs = xs - float(np.mean(xs))
    denominator = float(np.dot(centered_xs, centered_xs))
    if denominator <= 0.0:
        return 0.0
    centered_ys = ys - float(np.mean(ys))
    return float(np.dot(centered_xs, centered_ys) / denominator)


def _percentile_abs(values: NDArray[np.float64], percentile: float) -> float:
    return float(np.percentile(np.abs(values), percentile)) if values.size else 0.0


def _max_abs(values: NDArray[np.float64]) -> float:
    return float(np.max(np.abs(values))) if values.size else 0.0


def _mean(values: tuple[float, ...]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _percentile(values: tuple[float, ...], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile)) if values else 0.0
