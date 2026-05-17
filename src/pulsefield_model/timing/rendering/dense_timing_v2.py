from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from pulsefield_model.timing.schema import FittedTimingGrid


DENSE_TIMING_V2_VERSION = "dense_timing_v2"
DENSE_TIMING_V2_CHANNELS = (
    "beat_pulse",
    "phase_sin",
    "phase_cos",
    "local_bpm",
)
_BEAT_PULSE_CHANNEL = 0
_PHASE_SIN_CHANNEL = 1
_PHASE_COS_CHANNEL = 2
_LOCAL_BPM_CHANNEL = 3

DenseTimingV2Track: TypeAlias = NDArray[np.float32]


@dataclass(frozen=True)
class DenseTimingV2Config:
    frame_hop_ms: float = 20.0
    frame_center_offset_ms: float = 10.0
    frame_count: int = 600
    pulse_width_ms: float = 40.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.frame_hop_ms) or self.frame_hop_ms <= 0.0:
            raise ValueError(f"frame_hop_ms must be positive and finite, got {self.frame_hop_ms!r}")
        if not np.isfinite(self.frame_center_offset_ms):
            raise ValueError(f"frame_center_offset_ms must be finite, got {self.frame_center_offset_ms!r}")
        if self.frame_count < 0:
            raise ValueError(f"frame_count must be non-negative, got {self.frame_count!r}")
        if not np.isfinite(self.pulse_width_ms) or self.pulse_width_ms <= 0.0:
            raise ValueError(f"pulse_width_ms must be positive and finite, got {self.pulse_width_ms!r}")


DEFAULT_DENSE_TIMING_V2_CONFIG = DenseTimingV2Config()


def render_dense_timing_v2(
    grid: FittedTimingGrid,
    *,
    input_start_ms: float,
    frame_count: int | None = None,
    config: DenseTimingV2Config = DEFAULT_DENSE_TIMING_V2_CONFIG,
) -> DenseTimingV2Track:
    frame_times_ms = dense_timing_v2_frame_times(
        input_start_ms,
        frame_count=frame_count,
        config=config,
    )
    active_offsets_ms, active_beat_lengths_ms = active_timing_arrays(grid, frame_times_ms)
    beat_phase = _beat_phase(frame_times_ms, active_offsets_ms, active_beat_lengths_ms)
    beat_pulse = _beat_pulse(
        beat_phase,
        active_beat_lengths_ms,
        pulse_width_ms=config.pulse_width_ms,
    )
    local_bpm = 60000.0 / active_beat_lengths_ms
    return _dense_timing_track(beat_pulse, beat_phase, local_bpm)


def _dense_timing_track(
    beat_pulse: NDArray[np.float64],
    beat_phase: NDArray[np.float64],
    local_bpm: NDArray[np.float64],
) -> DenseTimingV2Track:
    track = np.empty((beat_pulse.shape[0], len(DENSE_TIMING_V2_CHANNELS)), dtype=np.float32)
    track[:, _BEAT_PULSE_CHANNEL] = beat_pulse.astype(np.float32)
    track[:, _PHASE_SIN_CHANNEL] = np.sin(2.0 * math.pi * beat_phase).astype(np.float32)
    track[:, _PHASE_COS_CHANNEL] = np.cos(2.0 * math.pi * beat_phase).astype(np.float32)
    track[:, _LOCAL_BPM_CHANNEL] = local_bpm.astype(np.float32)
    return track


def dense_timing_v2_frame_times(
    input_start_ms: float,
    *,
    frame_count: int | None = None,
    config: DenseTimingV2Config = DEFAULT_DENSE_TIMING_V2_CONFIG,
) -> NDArray[np.float64]:
    if not np.isfinite(input_start_ms):
        raise ValueError(f"input_start_ms must be finite, got {input_start_ms!r}")
    resolved_frame_count = config.frame_count if frame_count is None else frame_count
    if resolved_frame_count < 0:
        raise ValueError(f"frame_count must be non-negative, got {resolved_frame_count!r}")

    frame_indexes = np.arange(resolved_frame_count, dtype=np.float64)
    return input_start_ms + config.frame_hop_ms * frame_indexes + config.frame_center_offset_ms


def active_timing_arrays(
    grid: FittedTimingGrid,
    frame_times_ms: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    offsets = np.asarray([segment.offset_ms for segment in grid.segments], dtype=np.float64)
    beat_lengths = np.asarray([segment.beat_length_ms for segment in grid.segments], dtype=np.float64)
    active_indices = np.searchsorted(offsets, frame_times_ms, side="right") - 1
    active_indices = np.clip(active_indices, 0, len(grid.segments) - 1)
    return offsets[active_indices], beat_lengths[active_indices]


def _beat_phase(
    frame_times_ms: NDArray[np.float64],
    active_offsets_ms: NDArray[np.float64],
    active_beat_lengths_ms: NDArray[np.float64],
) -> NDArray[np.float64]:
    beat_pos = (frame_times_ms - active_offsets_ms) / active_beat_lengths_ms
    return beat_pos - np.floor(beat_pos)


def _beat_pulse(
    beat_phase: NDArray[np.float64],
    active_beat_lengths_ms: NDArray[np.float64],
    *,
    pulse_width_ms: float,
) -> NDArray[np.float64]:
    distance_beats = np.minimum(beat_phase, 1.0 - beat_phase)
    distance_ms = distance_beats * active_beat_lengths_ms
    return np.maximum(0.0, 1.0 - distance_ms / pulse_width_ms)
