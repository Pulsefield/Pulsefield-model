from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pulsefield_model.timing.grid_fitting.config import GridFitterConfig
from pulsefield_model.timing.grid_fitting.frames import _frame_rate_hz_from_times
from pulsefield_model.timing.grid_fitting.scoring import (
    _best_bpm_fit,
    _best_grid_candidate,
    _candidate_bpms,
    _candidate_offsets_ms,
    _centered_signal_and_norm,
    _downbeat_rejects_close_tempo_alias,
    _refine_grid_offset_ms,
    _score_grid,
)
from pulsefield_model.timing.grid_fitting.types import _GridCandidate, _SegmentFit


def _fit_segment_range(
    signal: NDArray[np.float64],
    *,
    frame_times_ms: NDArray[np.float64],
    downbeat_signal: NDArray[np.float64] | None = None,
    start_frame: int,
    end_frame: int,
    config: GridFitterConfig,
) -> _SegmentFit:
    segment_signal = signal[start_frame:end_frame]
    centered_signal = segment_signal - float(np.mean(segment_signal))
    signal_norm = float(np.linalg.norm(centered_signal))
    if signal_norm == 0.0:
        return _SegmentFit(
            start_frame=start_frame,
            end_frame=end_frame,
            score=-np.inf,
            beat_length_ms=60000.0 / config.min_bpm,
            offset_ms=0.0,
            half_tempo_score=-np.inf,
            double_tempo_score=-np.inf,
            raw_bpm=config.min_bpm,
            raw_score=-np.inf,
            tempo_multiplier=1.0,
            candidate_count=0,
        )

    segment_frame_times_ms = frame_times_ms[start_frame:end_frame]
    segment_downbeat_signal = None if downbeat_signal is None else downbeat_signal[start_frame:end_frame]
    downbeat_centered_signal, downbeat_signal_norm = _centered_signal_and_norm(segment_downbeat_signal)
    candidate_count = 0
    candidates: list[_GridCandidate] = []

    frame_rate_hz = _frame_rate_hz_from_times(segment_frame_times_ms)
    for bpm in _candidate_bpms(
        centered_signal,
        frame_rate_hz=frame_rate_hz,
        config=config,
    ):
        beat_length_ms = 60000.0 / bpm
        for offset_ms in _candidate_offsets_ms(beat_length_ms, config=config):
            score = _score_grid(
                centered_signal,
                signal_norm=signal_norm,
                frame_times_ms=segment_frame_times_ms,
                beat_length_ms=beat_length_ms,
                offset_ms=offset_ms,
                pulse_width_ms=config.pulse_width_ms,
            )
            candidate_count += 1
            candidates.append(
                _GridCandidate(
                    score=float(score),
                    bpm=float(bpm),
                    beat_length_ms=float(beat_length_ms),
                    offset_ms=float(offset_ms),
                )
            )

    best_score, best_downbeat_score, best_bpm, best_beat_length_ms, best_offset_ms = _best_grid_candidate(
        candidates,
        downbeat_centered_signal=downbeat_centered_signal,
        downbeat_signal_norm=downbeat_signal_norm,
        frame_times_ms=segment_frame_times_ms,
        config=config,
    )
    best_score, best_offset_ms, best_downbeat_score = _refine_grid_offset_ms(
        centered_signal,
        signal_norm=signal_norm,
        downbeat_centered_signal=downbeat_centered_signal,
        downbeat_signal_norm=downbeat_signal_norm,
        frame_times_ms=segment_frame_times_ms,
        beat_length_ms=best_beat_length_ms,
        offset_ms=best_offset_ms,
        score=best_score,
        downbeat_score=best_downbeat_score,
        config=config,
    )

    raw_bpm = best_bpm
    raw_score = best_score
    raw_downbeat_score = best_downbeat_score
    half_tempo_score, _, _ = _best_bpm_fit(
        centered_signal,
        signal_norm=signal_norm,
        frame_times_ms=segment_frame_times_ms,
        bpm=raw_bpm / 2.0,
        pulse_width_ms=config.pulse_width_ms,
        downbeat_centered_signal=downbeat_centered_signal,
        downbeat_signal_norm=downbeat_signal_norm,
        config=config,
    )
    double_tempo_score, double_tempo_offset_ms, double_tempo_downbeat_score = _best_bpm_fit(
        centered_signal,
        signal_norm=signal_norm,
        frame_times_ms=segment_frame_times_ms,
        bpm=raw_bpm * 2.0,
        pulse_width_ms=config.pulse_width_ms,
        downbeat_centered_signal=downbeat_centered_signal,
        downbeat_signal_norm=downbeat_signal_norm,
        config=config,
    )

    tempo_multiplier = 1.0
    if (
        double_tempo_score != -np.inf
        and double_tempo_score >= raw_score * config.double_tempo_score_ratio_threshold
        and not _downbeat_rejects_close_tempo_alias(
            raw_score,
            raw_downbeat_score,
            double_tempo_score,
            double_tempo_downbeat_score,
            config=config,
        )
    ):
        best_bpm = raw_bpm * 2.0
        best_beat_length_ms = 60000.0 / best_bpm
        best_offset_ms = double_tempo_offset_ms
        best_score = double_tempo_score
        tempo_multiplier = 2.0

    return _SegmentFit(
        start_frame=start_frame,
        end_frame=end_frame,
        score=float(best_score),
        beat_length_ms=float(best_beat_length_ms),
        offset_ms=float(best_offset_ms),
        half_tempo_score=float(half_tempo_score),
        double_tempo_score=float(double_tempo_score),
        raw_bpm=float(raw_bpm),
        raw_score=float(raw_score),
        tempo_multiplier=tempo_multiplier,
        candidate_count=candidate_count,
    )
