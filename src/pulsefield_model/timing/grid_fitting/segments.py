from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from pulsefield_model.timing.grid_fitting.config import GridFitterConfig
from pulsefield_model.timing.grid_fitting.types import _SegmentFit
from pulsefield_model.timing.schema import TimingSegment


def _weighted_score(segment_fits: Sequence[_SegmentFit]) -> float:
    total_frames = sum(fit.frame_count for fit in segment_fits)
    if total_frames <= 0:
        return -np.inf
    return float(sum(fit.score * fit.frame_count for fit in segment_fits) / total_frames)


def _timing_segments_from_fits(
    segment_fits: Sequence[_SegmentFit],
    frame_times_ms: NDArray[np.float64],
    *,
    config: GridFitterConfig,
) -> tuple[TimingSegment, ...]:
    return _merge_similar_timing_segments(
        _raw_timing_segments_from_fits(segment_fits, frame_times_ms),
        config=config,
    )


def _raw_timing_segments_from_fits(
    segment_fits: Sequence[_SegmentFit],
    frame_times_ms: NDArray[np.float64],
) -> tuple[TimingSegment, ...]:
    segments: list[TimingSegment] = []
    for index, fit in enumerate(segment_fits):
        offset_ms = fit.offset_ms
        if index > 0:
            boundary_time_ms = frame_times_ms[fit.start_frame]
            offset_ms = _nearest_congruent_offset(
                offset_ms,
                beat_length_ms=fit.beat_length_ms,
                target_time_ms=boundary_time_ms,
            )
            while offset_ms <= segments[-1].offset_ms:
                offset_ms += fit.beat_length_ms
        segments.append(TimingSegment(offset_ms=float(offset_ms), beat_length_ms=fit.beat_length_ms))
    return tuple(segments)


def _segment_fits_are_mergeable(
    left_fit: _SegmentFit,
    right_fit: _SegmentFit,
    *,
    frame_times_ms: NDArray[np.float64],
    config: GridFitterConfig,
    allow_alias: bool = False,
    allow_loose_similar: bool = False,
) -> bool:
    segments = _raw_timing_segments_from_fits((left_fit, right_fit), frame_times_ms)
    return len(segments) == 2 and _timing_segments_are_mergeable(
        segments[0],
        segments[1],
        config=config,
        allow_alias=allow_alias,
        allow_loose_similar=allow_loose_similar,
    )


def _merge_similar_timing_segments(
    segments: Sequence[TimingSegment],
    *,
    config: GridFitterConfig,
) -> tuple[TimingSegment, ...]:
    if not config.merge_similar_segments or not segments:
        return tuple(segments)

    merged_segments: list[TimingSegment] = [segments[0]]
    for segment in segments[1:]:
        previous_segment = merged_segments[-1]
        if _timing_segments_are_mergeable(
            previous_segment,
            segment,
            config=config,
        ):
            continue
        merged_segments.append(segment)
    return tuple(merged_segments)


def _timing_segments_are_mergeable(
    previous_segment: TimingSegment,
    segment: TimingSegment,
    *,
    config: GridFitterConfig,
    allow_alias: bool = False,
    allow_loose_similar: bool = False,
) -> bool:
    bpm_tolerance = max(
        config.merge_bpm_tolerance,
        min(previous_segment.local_bpm, segment.local_bpm) * config.merge_relative_bpm_tolerance,
    )
    if allow_loose_similar:
        bpm_tolerance = max(bpm_tolerance, config.merge_many_similar_bpm_tolerance)
    if abs(previous_segment.local_bpm - segment.local_bpm) <= bpm_tolerance:
        if allow_loose_similar:
            return True
        return (
            _phase_error_ms(
                segment.offset_ms,
                offset_ms=previous_segment.offset_ms,
                beat_length_ms=previous_segment.beat_length_ms,
            )
            <= config.merge_phase_tolerance_ms
        )
    if not allow_alias or not _bpms_are_alias_compatible(
        previous_segment.local_bpm,
        segment.local_bpm,
        tolerance_bpm=config.merge_alias_bpm_tolerance,
    ):
        return False
    alias_beat_length_ms = min(previous_segment.beat_length_ms, segment.beat_length_ms)
    return (
        _phase_error_ms(
            segment.offset_ms,
            offset_ms=previous_segment.offset_ms,
            beat_length_ms=alias_beat_length_ms,
        )
        <= config.merge_alias_phase_tolerance_ms
    )


def _bpms_are_alias_compatible(
    first_bpm: float,
    second_bpm: float,
    *,
    tolerance_bpm: float,
) -> bool:
    if first_bpm <= 0.0 or second_bpm <= 0.0:
        return False
    for multiplier in (0.25, 0.5, 2.0, 4.0):
        if abs(first_bpm * multiplier - second_bpm) <= tolerance_bpm:
            return True
    return False


def _phase_error_ms(
    time_ms: float,
    *,
    offset_ms: float,
    beat_length_ms: float,
) -> float:
    phase_ms = float(np.mod(time_ms - offset_ms, beat_length_ms))
    return float(min(phase_ms, beat_length_ms - phase_ms))


def _nearest_congruent_offset(
    offset_ms: float,
    *,
    beat_length_ms: float,
    target_time_ms: float,
) -> float:
    period_count = round((target_time_ms - offset_ms) / beat_length_ms)
    return float(offset_ms + period_count * beat_length_ms)
