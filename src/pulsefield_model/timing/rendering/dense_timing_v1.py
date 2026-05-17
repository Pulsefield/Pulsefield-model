from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence, TypeAlias

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from pulsefield_model.osu_core.timing import RedTimingPoint


TIMING_TRACK_VERSION = "timing_track_20ms_v1"
TIMING_TRACK_CHANNELS = (
    "beat_pulse",
    "beat_phase_sin",
    "beat_phase_cos",
    "local_bpm_log_norm",
    "timing_confidence",
)

TimingTrack: TypeAlias = npt.NDArray[np.float32]


@dataclass(frozen=True)
class TimingTrackConfig:
    frame_hop_ms: float = 20.0
    frame_center_offset_ms: float = 10.0
    frame_count: int = 600
    pulse_width_ms: float = 40.0
    timing_confidence: float = 1.0


DEFAULT_TIMING_TRACK_CONFIG = TimingTrackConfig()


def render_timing_track_20ms_v1(
    timing_points: Sequence[RedTimingPoint],
    *,
    input_start_ms: float,
    bpm_log_mean: float,
    bpm_log_std: float,
    frame_count: int | None = None,
    config: TimingTrackConfig = DEFAULT_TIMING_TRACK_CONFIG,
) -> TimingTrack:
    if not math.isfinite(bpm_log_mean):
        raise ValueError(f"bpm_log_mean must be finite: {bpm_log_mean}")
    if not math.isfinite(bpm_log_std) or bpm_log_std <= 0:
        raise ValueError(f"bpm_log_std must be positive: {bpm_log_std}")

    frame_times = timing_frame_times_20ms_v1(input_start_ms, frame_count=frame_count, config=config)
    active_offsets, active_beat_lengths = active_red_timing_arrays(timing_points, frame_times)
    beat_phase = _beat_phase(frame_times, active_offsets, active_beat_lengths)
    beat_pulse = _beat_pulse(beat_phase, active_beat_lengths, pulse_width_ms=config.pulse_width_ms)
    bpm_log = np.log(60000.0 / active_beat_lengths)
    local_bpm_log_norm = np.clip((bpm_log - bpm_log_mean) / bpm_log_std, -4.0, 4.0)

    track = np.empty((frame_times.shape[0], len(TIMING_TRACK_CHANNELS)), dtype=np.float32)
    track[:, 0] = beat_pulse.astype(np.float32)
    track[:, 1] = np.sin(2.0 * math.pi * beat_phase).astype(np.float32)
    track[:, 2] = np.cos(2.0 * math.pi * beat_phase).astype(np.float32)
    track[:, 3] = local_bpm_log_norm.astype(np.float32)
    track[:, 4] = np.float32(config.timing_confidence)
    return track


def render_local_bpm_log_20ms_v1(
    timing_points: Sequence[RedTimingPoint],
    *,
    input_start_ms: float,
    frame_count: int | None = None,
    config: TimingTrackConfig = DEFAULT_TIMING_TRACK_CONFIG,
) -> npt.NDArray[np.float64]:
    frame_times = timing_frame_times_20ms_v1(input_start_ms, frame_count=frame_count, config=config)
    _, active_beat_lengths = active_red_timing_arrays(timing_points, frame_times)
    return np.log(60000.0 / active_beat_lengths)


def render_raw_beat_lengths_20ms_v1(
    timing_points: Sequence[RedTimingPoint],
    *,
    input_start_ms: float,
    frame_count: int | None = None,
    config: TimingTrackConfig = DEFAULT_TIMING_TRACK_CONFIG,
) -> npt.NDArray[np.float64]:
    frame_times = timing_frame_times_20ms_v1(input_start_ms, frame_count=frame_count, config=config)
    _, active_beat_lengths = active_red_timing_arrays(timing_points, frame_times)
    return active_beat_lengths


def timing_frame_times_20ms_v1(
    input_start_ms: float,
    *,
    frame_count: int | None = None,
    config: TimingTrackConfig = DEFAULT_TIMING_TRACK_CONFIG,
) -> npt.NDArray[np.float64]:
    resolved_frame_count = config.frame_count if frame_count is None else frame_count
    if resolved_frame_count < 0:
        raise ValueError(f"frame_count must be non-negative: {resolved_frame_count}")
    frame_indexes = np.arange(resolved_frame_count, dtype=np.float64)
    return input_start_ms + config.frame_hop_ms * frame_indexes + config.frame_center_offset_ms


def active_red_timing_arrays(
    timing_points: Sequence[RedTimingPoint],
    frame_times_ms: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    missing_red_timing_error, validate_red_timing_point = _red_timing_helpers()
    sorted_points = sorted(timing_points, key=lambda point: point.offset_ms)
    if not sorted_points:
        raise missing_red_timing_error("cannot render dense timing with no red timing points")
    for point in sorted_points:
        validate_red_timing_point(point)

    offsets = np.array([point.offset_ms for point in sorted_points], dtype=np.float64)
    beat_lengths = np.array([point.beat_length_ms for point in sorted_points], dtype=np.float64)
    active_indices = np.searchsorted(offsets, frame_times_ms, side="right") - 1
    active_indices = np.clip(active_indices, 0, len(sorted_points) - 1)
    return offsets[active_indices], beat_lengths[active_indices]


def _beat_phase(
    frame_times_ms: npt.NDArray[np.float64],
    active_offsets_ms: npt.NDArray[np.float64],
    active_beat_lengths_ms: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    beat_pos = (frame_times_ms - active_offsets_ms) / active_beat_lengths_ms
    return beat_pos - np.floor(beat_pos)


def _beat_pulse(
    beat_phase: npt.NDArray[np.float64],
    active_beat_lengths_ms: npt.NDArray[np.float64],
    *,
    pulse_width_ms: float,
) -> npt.NDArray[np.float64]:
    if pulse_width_ms <= 0:
        raise ValueError(f"pulse_width_ms must be positive: {pulse_width_ms}")
    distance_beats = np.minimum(beat_phase, 1.0 - beat_phase)
    distance_ms = distance_beats * active_beat_lengths_ms
    return np.maximum(0.0, 1.0 - distance_ms / pulse_width_ms)


def _red_timing_helpers() -> tuple[type[ValueError], object]:
    from pulsefield_model.osu_core.timing import MissingRedTimingError, validate_red_timing_point

    return MissingRedTimingError, validate_red_timing_point
