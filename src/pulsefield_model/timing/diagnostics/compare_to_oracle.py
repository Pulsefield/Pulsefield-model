from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pulsefield_model.timing.grid_fitting.alias import (
    _alias_bpm_mae,
)
from pulsefield_model.timing.rendering.dense_timing_v2 import (
    DEFAULT_DENSE_TIMING_V2_CONFIG,
    DenseTimingV2Config,
    active_timing_arrays,
    dense_timing_v2_frame_times,
    render_dense_timing_v2,
)
from pulsefield_model.timing.schema import FittedTimingGrid


_BEAT_PULSE_CHANNEL = 0
_PHASE_SIN_CHANNEL = 1
_PHASE_COS_CHANNEL = 2
_LOCAL_BPM_CHANNEL = 3

@dataclass(frozen=True)
class TimingGridComparison:
    frame_count: int
    beat_pulse_mae: float
    local_bpm_mae: float
    local_bpm_alias_mae: float
    mean_phase_error_beats: float
    max_phase_error_beats: float
    mean_phase_error_ms: float
    max_phase_error_ms: float


def compare_timing_grids(
    predicted_grid: FittedTimingGrid,
    oracle_grid: FittedTimingGrid,
    *,
    frame_count: int,
    input_start_ms: float = 0.0,
    config: DenseTimingV2Config = DEFAULT_DENSE_TIMING_V2_CONFIG,
) -> TimingGridComparison:
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count!r}")

    predicted_track = render_dense_timing_v2(
        predicted_grid,
        input_start_ms=input_start_ms,
        frame_count=frame_count,
        config=config,
    )
    oracle_track = render_dense_timing_v2(
        oracle_grid,
        input_start_ms=input_start_ms,
        frame_count=frame_count,
        config=config,
    )

    frame_times_ms = dense_timing_v2_frame_times(
        input_start_ms,
        frame_count=frame_count,
        config=config,
    )
    _, oracle_beat_lengths_ms = active_timing_arrays(oracle_grid, frame_times_ms)

    phase_error_beats = _phase_error_beats(predicted_track, oracle_track)
    phase_error_ms = phase_error_beats * oracle_beat_lengths_ms

    return TimingGridComparison(
        frame_count=frame_count,
        beat_pulse_mae=_mean_absolute_channel_delta(
            predicted_track,
            oracle_track,
            _BEAT_PULSE_CHANNEL,
        ),
        local_bpm_mae=_mean_absolute_channel_delta(
            predicted_track,
            oracle_track,
            _LOCAL_BPM_CHANNEL,
        ),
        local_bpm_alias_mae=_alias_bpm_mae(
            predicted_track[:, _LOCAL_BPM_CHANNEL].astype(np.float64),
            oracle_track[:, _LOCAL_BPM_CHANNEL].astype(np.float64),
        ),
        mean_phase_error_beats=float(np.mean(phase_error_beats)),
        max_phase_error_beats=float(np.max(phase_error_beats)),
        mean_phase_error_ms=float(np.mean(phase_error_ms)),
        max_phase_error_ms=float(np.max(phase_error_ms)),
    )


def _phase_error_beats(
    predicted_track: np.ndarray,
    oracle_track: np.ndarray,
) -> np.ndarray:
    angle_delta = np.arctan2(
        predicted_track[:, _PHASE_SIN_CHANNEL] * oracle_track[:, _PHASE_COS_CHANNEL]
        - predicted_track[:, _PHASE_COS_CHANNEL] * oracle_track[:, _PHASE_SIN_CHANNEL],
        predicted_track[:, _PHASE_COS_CHANNEL] * oracle_track[:, _PHASE_COS_CHANNEL]
        + predicted_track[:, _PHASE_SIN_CHANNEL] * oracle_track[:, _PHASE_SIN_CHANNEL],
    )
    return np.abs(angle_delta) / (2.0 * np.pi)


def _mean_absolute_channel_delta(
    predicted_track: np.ndarray,
    oracle_track: np.ndarray,
    channel_index: int,
) -> float:
    return float(np.mean(np.abs(predicted_track[:, channel_index] - oracle_track[:, channel_index])))
