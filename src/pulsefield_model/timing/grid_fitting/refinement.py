from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from pulsefield_model.timing.grid_fitting.config import GridFitterConfig
from pulsefield_model.timing.grid_fitting.scoring import _best_bpm_fit
from pulsefield_model.timing.grid_fitting.scoring import _score_grid
from pulsefield_model.timing.grid_fitting.segments import (
    _merge_similar_timing_segments,
    _nearest_congruent_offset,
    _phase_error_ms,
)
from pulsefield_model.timing.schema import TimingSegment


@dataclass(frozen=True)
class _TempoFamily:
    bpm: float
    indices: tuple[int, ...]
    duration_ms: float

    @property
    def segment_count(self) -> int:
        return len(self.indices)


def _refine_timing_segments(
    segments: Sequence[TimingSegment],
    frame_times_ms: NDArray[np.float64],
    *,
    beat_signal: NDArray[np.float64] | None = None,
    config: GridFitterConfig,
) -> tuple[TimingSegment, ...]:
    refined = tuple(segments)
    if not refined:
        return refined

    if config.refine_structural_segments:
        collapsed = _collapse_repeated_dominant_tempo(refined, frame_times_ms, config=config)
        if tuple(collapsed) != tuple(refined):
            collapsed = _refine_single_segment_offset_from_signal(
                collapsed,
                frame_times_ms,
                beat_signal=beat_signal,
                config=config,
            )
        if _refinement_score_guard_accepts(
            refined,
            collapsed,
            frame_times_ms,
            beat_signal=beat_signal,
            config=config,
        ):
            refined = collapsed
        refined = _merge_similar_timing_segments(refined, config=config)

    if config.refine_offset_quantum_ms > 0.0:
        refined = _quantize_segment_offsets(refined, quantum_ms=config.refine_offset_quantum_ms)
        refined = _merge_similar_timing_segments(refined, config=config)

    return refined


def _refine_single_segment_offset_from_signal(
    segments: Sequence[TimingSegment],
    frame_times_ms: NDArray[np.float64],
    *,
    beat_signal: NDArray[np.float64] | None,
    config: GridFitterConfig,
) -> tuple[TimingSegment, ...]:
    if len(segments) != 1 or beat_signal is None or beat_signal.shape[0] != frame_times_ms.shape[0]:
        return tuple(segments)

    centered_signal = beat_signal.astype(np.float64, copy=False) - float(np.mean(beat_signal))
    signal_norm = float(np.linalg.norm(centered_signal))
    if signal_norm == 0.0:
        return tuple(segments)

    segment = segments[0]
    _, offset_ms, _ = _best_bpm_fit(
        centered_signal,
        signal_norm=signal_norm,
        frame_times_ms=frame_times_ms,
        bpm=segment.local_bpm,
        pulse_width_ms=config.pulse_width_ms,
        config=config,
    )
    return (
        TimingSegment(
            offset_ms=float(offset_ms),
            beat_length_ms=segment.beat_length_ms,
            meter=segment.meter,
        ),
    )


def _refinement_score_guard_accepts(
    original_segments: Sequence[TimingSegment],
    proposed_segments: Sequence[TimingSegment],
    frame_times_ms: NDArray[np.float64],
    *,
    beat_signal: NDArray[np.float64] | None,
    config: GridFitterConfig,
) -> bool:
    if tuple(original_segments) == tuple(proposed_segments):
        return True
    if beat_signal is None:
        return False

    original_score = _timing_grid_signal_score(original_segments, frame_times_ms, beat_signal, config=config)
    proposed_score = _timing_grid_signal_score(proposed_segments, frame_times_ms, beat_signal, config=config)
    if original_score is None or proposed_score is None:
        return True
    return proposed_score >= original_score - config.refine_structural_max_score_loss


def _timing_grid_signal_score(
    segments: Sequence[TimingSegment],
    frame_times_ms: NDArray[np.float64],
    beat_signal: NDArray[np.float64],
    *,
    config: GridFitterConfig,
) -> float | None:
    if frame_times_ms.shape[0] == 0 or beat_signal.shape[0] != frame_times_ms.shape[0]:
        return None

    weighted_score = 0.0
    scored_frame_count = 0
    for index, segment in enumerate(segments):
        start_time_ms = float(frame_times_ms[0]) if index == 0 else max(
            float(frame_times_ms[0]),
            float(segment.offset_ms),
        )
        end_time_ms = float(frame_times_ms[-1]) + 1.0
        if index + 1 < len(segments):
            end_time_ms = min(end_time_ms, float(segments[index + 1].offset_ms))

        start_frame = int(np.searchsorted(frame_times_ms, start_time_ms, side="left"))
        end_frame = int(np.searchsorted(frame_times_ms, end_time_ms, side="left"))
        if end_frame - start_frame < 2:
            continue

        signal_slice = beat_signal[start_frame:end_frame].astype(np.float64, copy=False)
        centered_signal = signal_slice - float(np.mean(signal_slice))
        signal_norm = float(np.linalg.norm(centered_signal))
        if signal_norm == 0.0:
            continue

        score = _score_grid(
            centered_signal,
            signal_norm=signal_norm,
            frame_times_ms=frame_times_ms[start_frame:end_frame],
            beat_length_ms=segment.beat_length_ms,
            offset_ms=segment.offset_ms,
            pulse_width_ms=config.pulse_width_ms,
        )
        if not np.isfinite(score):
            continue
        frame_count = end_frame - start_frame
        weighted_score += float(score) * float(frame_count)
        scored_frame_count += frame_count

    if scored_frame_count <= 0:
        return None
    return float(weighted_score / float(scored_frame_count))


def _collapse_repeated_dominant_tempo(
    segments: Sequence[TimingSegment],
    frame_times_ms: NDArray[np.float64],
    *,
    config: GridFitterConfig,
) -> tuple[TimingSegment, ...]:
    if len(segments) < config.refine_dominant_min_segments:
        return tuple(segments)

    durations_ms = _active_segment_durations_ms(segments, frame_times_ms)
    total_duration_ms = float(np.sum(durations_ms))
    if total_duration_ms <= 0.0:
        return tuple(segments)

    families = _tempo_families(segments, durations_ms, config=config)
    if len(families) < 2:
        return tuple(segments)

    dominant = max(families, key=lambda family: family.duration_ms)
    outlier_segment_count = len(segments) - dominant.segment_count
    far_outlier_segment_count = _far_outlier_segment_count(segments, dominant, config=config)
    dominant_ratio = dominant.duration_ms / total_duration_ms
    if dominant.segment_count < config.refine_dominant_min_segment_count:
        return tuple(segments)
    if outlier_segment_count < config.refine_dominant_min_outlier_segments:
        return tuple(segments)
    if far_outlier_segment_count < config.refine_dominant_min_far_outlier_segments:
        return tuple(segments)
    if dominant_ratio < config.refine_dominant_min_duration_ratio:
        return tuple(segments)
    if _first_anchor_ratio(segments, dominant, frame_times_ms) > config.refine_dominant_max_first_anchor_ratio:
        return tuple(segments)
    if _off_lattice_transition_count(segments, config=config) < config.refine_dominant_min_off_lattice_transitions:
        return tuple(segments)
    if _indices_are_contiguous(dominant.indices):
        return tuple(segments)

    dominant_bpm = _snap_bpm_to_integer_prior(dominant.bpm, config=config)
    dominant_beat_length_ms = 60000.0 / dominant_bpm
    offset_ms = _dominant_phase_offset_ms(
        segments,
        dominant.indices,
        durations_ms,
        beat_length_ms=dominant_beat_length_ms,
        config=config,
    )
    offset_ms = _nearest_congruent_offset(
        offset_ms,
        beat_length_ms=dominant_beat_length_ms,
        target_time_ms=segments[0].offset_ms,
    )
    return (
        TimingSegment(
            offset_ms=float(offset_ms),
            beat_length_ms=float(dominant_beat_length_ms),
            meter=segments[0].meter,
        ),
    )


def _active_segment_durations_ms(
    segments: Sequence[TimingSegment],
    frame_times_ms: NDArray[np.float64],
) -> NDArray[np.float64]:
    if frame_times_ms.shape[0] == 0:
        return np.ones(len(segments), dtype=np.float64)

    first_time_ms = float(frame_times_ms[0])
    last_time_ms = float(frame_times_ms[-1])
    durations: list[float] = []
    for index, segment in enumerate(segments):
        start_ms = first_time_ms if index == 0 else max(first_time_ms, float(segment.offset_ms))
        end_ms = last_time_ms
        if index + 1 < len(segments):
            end_ms = min(last_time_ms, float(segments[index + 1].offset_ms))
        durations.append(max(0.0, end_ms - start_ms))
    return np.asarray(durations, dtype=np.float64)


def _tempo_families(
    segments: Sequence[TimingSegment],
    durations_ms: NDArray[np.float64],
    *,
    config: GridFitterConfig,
) -> tuple[_TempoFamily, ...]:
    family_indices: list[list[int]] = []
    family_weights: list[float] = []
    family_bpm_sums: list[float] = []

    for index, segment in enumerate(segments):
        bpm = float(segment.local_bpm)
        duration_ms = float(durations_ms[index])
        family_index = _matching_family_index(bpm, family_bpm_sums, family_weights, config=config)
        if family_index is None:
            family_indices.append([index])
            family_weights.append(duration_ms)
            family_bpm_sums.append(bpm * duration_ms)
            continue
        family_indices[family_index].append(index)
        family_weights[family_index] += duration_ms
        family_bpm_sums[family_index] += bpm * duration_ms

    families: list[_TempoFamily] = []
    for indices, weight, bpm_sum in zip(family_indices, family_weights, family_bpm_sums):
        if weight <= 0.0:
            bpm = float(segments[indices[0]].local_bpm)
        else:
            bpm = float(bpm_sum / weight)
        families.append(_TempoFamily(bpm=bpm, indices=tuple(indices), duration_ms=float(weight)))
    return tuple(families)


def _matching_family_index(
    bpm: float,
    family_bpm_sums: Sequence[float],
    family_weights: Sequence[float],
    *,
    config: GridFitterConfig,
) -> int | None:
    for index, (bpm_sum, weight) in enumerate(zip(family_bpm_sums, family_weights)):
        family_bpm = bpm if weight <= 0.0 else bpm_sum / weight
        tolerance = max(
            config.refine_dominant_bpm_tolerance,
            min(float(bpm), float(family_bpm)) * config.merge_relative_bpm_tolerance,
        )
        if abs(float(bpm) - float(family_bpm)) <= tolerance:
            return index
    return None


def _indices_are_contiguous(indices: Sequence[int]) -> bool:
    if not indices:
        return True
    return max(indices) - min(indices) + 1 == len(indices)


def _far_outlier_segment_count(
    segments: Sequence[TimingSegment],
    dominant: _TempoFamily,
    *,
    config: GridFitterConfig,
) -> int:
    dominant_indices = set(dominant.indices)
    far_tolerance = max(
        config.refine_dominant_far_bpm_tolerance,
        dominant.bpm * config.refine_dominant_far_relative_bpm_tolerance,
    )
    return sum(
        1
        for index, segment in enumerate(segments)
        if index not in dominant_indices and abs(float(segment.local_bpm) - dominant.bpm) > far_tolerance
    )


def _off_lattice_transition_count(
    segments: Sequence[TimingSegment],
    *,
    config: GridFitterConfig,
) -> int:
    return sum(
        1
        for previous_segment, segment in zip(segments, segments[1:])
        if _phase_error_ms(
            segment.offset_ms,
            offset_ms=previous_segment.offset_ms,
            beat_length_ms=previous_segment.beat_length_ms,
        )
        > config.refine_dominant_boundary_phase_tolerance_ms
    )


def _first_anchor_ratio(
    segments: Sequence[TimingSegment],
    dominant: _TempoFamily,
    frame_times_ms: NDArray[np.float64],
) -> float:
    if not dominant.indices:
        return 1.0
    if frame_times_ms.shape[0] == 0:
        return 0.0
    first_time_ms = float(frame_times_ms[0])
    last_time_ms = float(frame_times_ms[-1])
    duration_ms = max(1.0, last_time_ms - first_time_ms)
    first_anchor_ms = max(first_time_ms, float(segments[min(dominant.indices)].offset_ms))
    return float((first_anchor_ms - first_time_ms) / duration_ms)


def _snap_bpm_to_integer_prior(bpm: float, *, config: GridFitterConfig) -> float:
    nearest_integer_bpm = round(float(bpm))
    if abs(float(bpm) - nearest_integer_bpm) <= config.refine_dominant_bpm_snap_tolerance:
        return float(nearest_integer_bpm)
    return float(bpm)


def _dominant_phase_offset_ms(
    segments: Sequence[TimingSegment],
    indices: Sequence[int],
    durations_ms: NDArray[np.float64],
    *,
    beat_length_ms: float,
    config: GridFitterConfig,
) -> float:
    vector_x = 0.0
    vector_y = 0.0
    total_weight = 0.0
    longest_index = indices[0]
    longest_duration_ms = -1.0

    for index in indices:
        duration_ms = float(durations_ms[index])
        if duration_ms > longest_duration_ms:
            longest_index = index
            longest_duration_ms = duration_ms
        phase = float(np.mod(segments[index].offset_ms, beat_length_ms)) / beat_length_ms
        angle = 2.0 * math.pi * phase
        vector_x += duration_ms * math.cos(angle)
        vector_y += duration_ms * math.sin(angle)
        total_weight += duration_ms

    if total_weight <= 0.0:
        return float(np.mod(segments[longest_index].offset_ms, beat_length_ms))

    coherence = math.hypot(vector_x, vector_y) / total_weight
    if coherence < config.refine_circular_phase_min_coherence:
        return float(np.mod(segments[longest_index].offset_ms, beat_length_ms))

    phase_angle = math.atan2(vector_y, vector_x)
    if phase_angle < 0.0:
        phase_angle += 2.0 * math.pi
    return float((phase_angle / (2.0 * math.pi)) * beat_length_ms)


def _quantize_segment_offsets(
    segments: Sequence[TimingSegment],
    *,
    quantum_ms: float,
) -> tuple[TimingSegment, ...]:
    quantized_segments: list[TimingSegment] = []
    previous_offset_ms = -np.inf
    for segment in segments:
        offset_ms = round(float(segment.offset_ms) / quantum_ms) * quantum_ms
        if offset_ms <= previous_offset_ms:
            offset_ms = previous_offset_ms + quantum_ms
        quantized_segments.append(
            TimingSegment(
                offset_ms=float(offset_ms),
                beat_length_ms=segment.beat_length_ms,
                meter=segment.meter,
            )
        )
        previous_offset_ms = float(offset_ms)
    return tuple(quantized_segments)
