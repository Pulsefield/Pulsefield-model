from __future__ import annotations

import numpy as np

from pulsefield_model.timing.grid_fitting.alias import (
    _AliasCanonicalizationResult,
    _canonicalize_tempo_aliases,
    _segment_alias_switch_count,
    _tempo_multiplier_distribution,
)
from pulsefield_model.timing.canonicalization import TIMING_CANONICALIZATION_NONE, canonicalize_timing_grid
from pulsefield_model.timing.grid_fitting.config import GridFitterConfig, _effective_config_for_prediction
from pulsefield_model.timing.grid_fitting.scoring import _candidate_period_frame_bounds
from pulsefield_model.timing.grid_fitting.refinement import _refine_timing_segments
from pulsefield_model.timing.grid_fitting.segment_fit import _fit_segment_range
from pulsefield_model.timing.grid_fitting.segments import _timing_segments_from_fits, _weighted_score
from pulsefield_model.timing.grid_fitting.splitting import _split_segment_range
from pulsefield_model.timing.grid_fitting.types import TimingFitDiagnostics, TimingFitResult
from pulsefield_model.timing.schema import FittedTimingGrid, FrameTimingPrediction


class GridFitter:
    def __init__(self, config: GridFitterConfig = GridFitterConfig()) -> None:
        self.config = config

    def fit(self, prediction: FrameTimingPrediction) -> TimingFitResult:
        return fit_timing_grid(prediction, config=self.config)


def fit_timing_grid(
    prediction: FrameTimingPrediction,
    *,
    config: GridFitterConfig = GridFitterConfig(),
) -> TimingFitResult:
    config = _effective_config_for_prediction(
        prediction.frame_count,
        frame_rate_hz=prediction.frame_rate_hz,
        config=config,
    )
    min_period_frames, max_period_frames = _candidate_period_frame_bounds(
        prediction.frame_rate_hz,
        config=config,
    )
    if prediction.frame_count < max_period_frames:
        raise ValueError(
            "prediction is too short to fit the configured tempo range: "
            f"frame_count={prediction.frame_count}, required>={max_period_frames}",
        )

    signal = prediction.beat_prob.astype(np.float64, copy=False)
    downbeat_signal = prediction.downbeat_prob.astype(np.float64, copy=False)
    if float(np.linalg.norm(signal - float(np.mean(signal)))) == 0.0:
        raise ValueError("beat_prob contains no beat signal")

    frame_times_ms = np.arange(prediction.frame_count, dtype=np.float64) / prediction.frame_rate_hz * 1000.0
    initial_fit = _fit_segment_range(
        signal,
        frame_times_ms=frame_times_ms,
        downbeat_signal=downbeat_signal,
        start_frame=0,
        end_frame=prediction.frame_count,
        config=config,
    )
    segment_fits = _split_segment_range(
        signal,
        frame_times_ms=frame_times_ms,
        downbeat_signal=downbeat_signal,
        fit=initial_fit,
        config=config,
        remaining_splits=config.max_segments - 1,
    )
    candidate_count = sum(fit.candidate_count for fit in segment_fits)
    if config.canonicalization == TIMING_CANONICALIZATION_NONE:
        alias_result = _canonicalize_tempo_aliases(
            segment_fits,
            signal,
            frame_times_ms=frame_times_ms,
            downbeat_signal=downbeat_signal,
            config=config,
        )
    else:
        alias_result = _AliasCanonicalizationResult(tuple(segment_fits), alias_candidate_count=0)
    segment_fits = alias_result.segment_fits
    best_score = _weighted_score(segment_fits)
    first_fit = segment_fits[0]
    refined_segments = _refine_timing_segments(
        _timing_segments_from_fits(segment_fits, frame_times_ms, config=config),
        frame_times_ms,
        beat_signal=signal,
        config=config,
    )
    grid = canonicalize_timing_grid(
        FittedTimingGrid(segments=refined_segments),
        canonicalization=config.canonicalization,
    )
    selected_segment = grid.segments[0]

    return TimingFitResult(
        grid=grid,
        score=float(best_score),
        diagnostics=TimingFitDiagnostics(
            fit_score=float(best_score),
            selected_period_frames=float(selected_segment.beat_length_ms / 1000.0 * prediction.frame_rate_hz),
            selected_offset_frames=float(selected_segment.offset_ms / 1000.0 * prediction.frame_rate_hz),
            selected_bpm=float(selected_segment.local_bpm),
            candidate_count=candidate_count,
            half_tempo_score=float(first_fit.half_tempo_score),
            double_tempo_score=float(first_fit.double_tempo_score),
            raw_selected_bpm=float(first_fit.raw_bpm),
            raw_score=float(first_fit.raw_score),
            tempo_multiplier=first_fit.tempo_multiplier,
            segment_alias_switch_count=_segment_alias_switch_count(grid.segments, config=config),
            tempo_multiplier_distribution=_tempo_multiplier_distribution(segment_fits),
            alias_candidate_count=alias_result.alias_candidate_count,
        ),
    )
