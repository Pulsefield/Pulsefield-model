from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from pulsefield_model.timing.grid_fitting.config import GridFitterConfig
from pulsefield_model.timing.grid_fitting.frames import _frame_step_ms
from pulsefield_model.timing.grid_fitting.types import _ChangeTimeCandidate, _SegmentFit, _SplitCandidate


def _detect_change_split_candidates(
    signal: NDArray[np.float64],
    *,
    frame_times_ms: NDArray[np.float64],
    downbeat_signal: NDArray[np.float64] | None,
    fit: _SegmentFit,
    config: GridFitterConfig,
) -> list[_SplitCandidate]:
    segment_signal = signal[fit.start_frame : fit.end_frame]
    segment_frame_times_ms = frame_times_ms[fit.start_frame : fit.end_frame]
    peak_times_ms = _beat_peak_times_ms(segment_signal, frame_times_ms=segment_frame_times_ms, config=config)
    if peak_times_ms.shape[0] < 4:
        return []

    frame_step_ms = _frame_step_ms(frame_times_ms)
    min_segment_frames = max(1, int(round(config.min_segment_duration_ms / frame_step_ms)))
    downbeat_peak_times_ms = _segment_downbeat_peak_times_ms(
        downbeat_signal,
        frame_times_ms=frame_times_ms,
        fit=fit,
        config=config,
    )
    candidate_times = _candidate_change_times_from_peaks(peak_times_ms, fit=fit, config=config)

    candidates_by_frame: dict[int, _SplitCandidate] = {}
    for candidate in candidate_times:
        split_frame = int(np.searchsorted(frame_times_ms, candidate.time_ms, side="left"))
        if not _split_respects_minimum_duration(split_frame, fit=fit, min_segment_frames=min_segment_frames):
            continue
        candidate_score = candidate.score + _downbeat_boundary_bonus(
            candidate.time_ms,
            downbeat_peak_times_ms=downbeat_peak_times_ms,
            fit=fit,
            config=config,
        )
        existing = candidates_by_frame.get(split_frame)
        if existing is None or candidate_score > existing.score:
            candidates_by_frame[split_frame] = _SplitCandidate(frame=split_frame, score=float(candidate_score))

    return sorted(candidates_by_frame.values(), key=lambda candidate: candidate.score, reverse=True)


def _segment_downbeat_peak_times_ms(
    downbeat_signal: NDArray[np.float64] | None,
    *,
    frame_times_ms: NDArray[np.float64],
    fit: _SegmentFit,
    config: GridFitterConfig,
) -> NDArray[np.float64]:
    if downbeat_signal is None:
        return np.asarray([], dtype=np.float64)
    segment_downbeat_signal = downbeat_signal[fit.start_frame : fit.end_frame]
    if float(np.linalg.norm(segment_downbeat_signal - float(np.mean(segment_downbeat_signal)))) == 0.0:
        return np.asarray([], dtype=np.float64)
    return _beat_peak_times_ms(
        segment_downbeat_signal,
        frame_times_ms=frame_times_ms[fit.start_frame : fit.end_frame],
        config=config,
    )


def _downbeat_boundary_bonus(
    candidate_time_ms: float,
    *,
    downbeat_peak_times_ms: NDArray[np.float64],
    fit: _SegmentFit,
    config: GridFitterConfig,
) -> float:
    if downbeat_peak_times_ms.shape[0] == 0:
        return 0.0
    nearest_distance_ms = float(np.min(np.abs(downbeat_peak_times_ms - candidate_time_ms)))
    tolerance_ms = min(fit.beat_length_ms * 0.5, config.min_segment_duration_ms * 0.125)
    if nearest_distance_ms > tolerance_ms:
        return 0.0
    closeness = 1.0 - nearest_distance_ms / max(tolerance_ms, 1e-6)
    return float(config.downbeat_split_score_bonus * closeness)


def _beat_peak_times_ms(
    signal: NDArray[np.float64],
    *,
    frame_times_ms: NDArray[np.float64],
    config: GridFitterConfig,
) -> NDArray[np.float64]:
    if signal.shape[0] < 3:
        return np.asarray([], dtype=np.float64)

    threshold = max(float(np.mean(signal) + np.std(signal)), float(np.max(signal) * 0.35))
    candidate_indices = np.flatnonzero(
        (signal[1:-1] >= signal[:-2])
        & (signal[1:-1] > signal[2:])
        & (signal[1:-1] >= threshold)
    ) + 1
    if candidate_indices.shape[0] == 0:
        return np.asarray([], dtype=np.float64)

    min_peak_distance_ms = config.offset_step_ms
    selected_indices: list[int] = []
    last_selected_time_ms = -np.inf
    for index in candidate_indices:
        time_ms = float(frame_times_ms[index])
        if time_ms - last_selected_time_ms >= min_peak_distance_ms:
            selected_indices.append(int(index))
            last_selected_time_ms = time_ms
        elif selected_indices and signal[index] > signal[selected_indices[-1]]:
            selected_indices[-1] = int(index)
            last_selected_time_ms = time_ms

    return frame_times_ms[np.asarray(selected_indices, dtype=np.int64)]


def _candidate_change_times_from_peaks(
    peak_times_ms: NDArray[np.float64],
    *,
    fit: _SegmentFit,
    config: GridFitterConfig,
) -> list[_ChangeTimeCandidate]:
    intervals_ms = np.diff(peak_times_ms)
    if intervals_ms.shape[0] < 3:
        return []

    interval_change_threshold_ms = max(config.offset_step_ms, config.split_phase_change_threshold_ms * 2.0)
    local_interval_count = 4
    candidates: list[_ChangeTimeCandidate] = []
    phase_errors_ms = _peak_phase_errors_ms(peak_times_ms, fit=fit)

    for split_index in range(local_interval_count, intervals_ms.shape[0] - local_interval_count + 1):
        before_intervals = intervals_ms[split_index - local_interval_count : split_index]
        after_intervals = intervals_ms[split_index : split_index + local_interval_count]
        if before_intervals.shape[0] < 3 or after_intervals.shape[0] < 3:
            continue
        interval_change_ms = abs(float(np.median(after_intervals)) - float(np.median(before_intervals)))
        before_phase_error_ms = float(np.median(phase_errors_ms[split_index - local_interval_count : split_index]))
        after_phase_error_ms = float(np.median(phase_errors_ms[split_index : split_index + local_interval_count]))
        phase_change_ms = abs(after_phase_error_ms - before_phase_error_ms)
        phase_residual_ms = max(before_phase_error_ms, after_phase_error_ms)
        score = (
            interval_change_ms / interval_change_threshold_ms
            + phase_change_ms / max(config.split_phase_change_threshold_ms, 1e-6)
            + phase_residual_ms / max(fit.beat_length_ms, 1e-6)
        )
        if (
            interval_change_ms + 1e-9 >= interval_change_threshold_ms
            or phase_change_ms >= config.split_phase_change_threshold_ms
            or phase_residual_ms >= config.split_phase_change_threshold_ms * 2.0
        ):
            candidates.append(_ChangeTimeCandidate(time_ms=float(peak_times_ms[split_index]), score=float(score)))

    return _merge_candidate_change_times(candidates, min_distance_ms=config.min_segment_duration_ms)


def _merge_candidate_change_times(
    candidates: Sequence[_ChangeTimeCandidate],
    *,
    min_distance_ms: float,
) -> list[_ChangeTimeCandidate]:
    merged_candidates: list[_ChangeTimeCandidate] = []
    for candidate in sorted(candidates, key=lambda value: value.time_ms):
        if not merged_candidates or candidate.time_ms - merged_candidates[-1].time_ms >= min_distance_ms:
            merged_candidates.append(candidate)
            continue
        if candidate.score > merged_candidates[-1].score:
            merged_candidates[-1] = candidate
    return merged_candidates


def _peak_phase_errors_ms(
    peak_times_ms: NDArray[np.float64],
    *,
    fit: _SegmentFit,
) -> NDArray[np.float64]:
    phase_ms = np.mod(peak_times_ms - fit.offset_ms, fit.beat_length_ms)
    return np.minimum(phase_ms, fit.beat_length_ms - phase_ms)


def _split_respects_minimum_duration(
    split_frame: int,
    *,
    fit: _SegmentFit,
    min_segment_frames: int,
) -> bool:
    return (
        split_frame - fit.start_frame >= min_segment_frames
        and fit.end_frame - split_frame >= min_segment_frames
    )
