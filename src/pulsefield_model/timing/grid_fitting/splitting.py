from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from pulsefield_model.timing.grid_fitting.change_detection import _detect_change_split_candidates
from pulsefield_model.timing.grid_fitting.config import GridFitterConfig
from pulsefield_model.timing.grid_fitting.segment_fit import _fit_segment_range
from pulsefield_model.timing.grid_fitting.segments import _segment_fits_are_mergeable, _weighted_score
from pulsefield_model.timing.grid_fitting.types import _EvaluatedSplit, _SegmentFit


def _split_segment_range(
    signal: NDArray[np.float64],
    *,
    frame_times_ms: NDArray[np.float64],
    downbeat_signal: NDArray[np.float64] | None,
    fit: _SegmentFit,
    config: GridFitterConfig,
    remaining_splits: int,
) -> list[_SegmentFit]:
    if remaining_splits <= 0:
        return [fit]

    fit_cache: dict[tuple[int, int], _SegmentFit] = {(fit.start_frame, fit.end_frame): fit}
    segment_fits: list[_SegmentFit]
    if config.initial_batch_split_candidate_count > 0:
        segment_fits = _batch_split_for_fit(
            signal,
            frame_times_ms=frame_times_ms,
            downbeat_signal=downbeat_signal,
            fit=fit,
            config=config,
            remaining_splits=remaining_splits,
            fit_cache=fit_cache,
        ) or [fit]
        remaining_splits -= len(segment_fits) - 1
    else:
        segment_fits = [fit]

    split_cache: dict[tuple[int, int], _EvaluatedSplit | None] = {}

    while remaining_splits > 0:
        best_split: _EvaluatedSplit | None = None
        for segment_index, segment_fit in enumerate(segment_fits):
            cache_key = (segment_fit.start_frame, segment_fit.end_frame)
            if cache_key not in split_cache:
                split_cache[cache_key] = _best_split_for_fit(
                    signal,
                    frame_times_ms=frame_times_ms,
                    downbeat_signal=downbeat_signal,
                    fit=segment_fit,
                    segment_index=segment_index,
                    config=config,
                    fit_cache=fit_cache,
                )
            cached_split = split_cache[cache_key]
            if cached_split is None:
                continue
            evaluated_split = _EvaluatedSplit(
                segment_index=segment_index,
                candidate=cached_split.candidate,
                left_fit=cached_split.left_fit,
                right_fit=cached_split.right_fit,
                improvement=cached_split.improvement,
            )
            if best_split is None or _split_is_better(evaluated_split, best_split):
                best_split = evaluated_split

        if best_split is None or best_split.improvement < config.split_score_improvement_threshold:
            break

        segment_fits[best_split.segment_index : best_split.segment_index + 1] = [
            best_split.left_fit,
            best_split.right_fit,
        ]
        remaining_splits -= 1

    return segment_fits


def _batch_split_for_fit(
    signal: NDArray[np.float64],
    *,
    frame_times_ms: NDArray[np.float64],
    downbeat_signal: NDArray[np.float64] | None,
    fit: _SegmentFit,
    config: GridFitterConfig,
    remaining_splits: int,
    fit_cache: dict[tuple[int, int], _SegmentFit],
) -> list[_SegmentFit] | None:
    candidates = _detect_change_split_candidates(
        signal,
        frame_times_ms=frame_times_ms,
        downbeat_signal=downbeat_signal,
        fit=fit,
        config=config,
    )
    if len(candidates) < 2:
        return None
    if (
        len(candidates) < config.initial_batch_split_min_candidate_count
        and fit.score > config.initial_batch_split_max_parent_score
    ):
        return None

    selected_count = min(remaining_splits, config.initial_batch_split_candidate_count)
    selected_candidates = sorted(candidates[:selected_count], key=lambda candidate: candidate.frame)
    region_edges = [fit.start_frame, *(candidate.frame for candidate in selected_candidates), fit.end_frame]
    segment_fits = [
        _cached_fit_segment_range(
            signal,
            frame_times_ms=frame_times_ms,
            downbeat_signal=downbeat_signal,
            start_frame=start_frame,
            end_frame=end_frame,
            config=config,
            fit_cache=fit_cache,
        )
        for start_frame, end_frame in zip(region_edges, region_edges[1:])
    ]
    segment_fits = _merge_adjacent_segment_fits(
        segment_fits,
        signal,
        frame_times_ms=frame_times_ms,
        downbeat_signal=downbeat_signal,
        config=config,
        fit_cache=fit_cache,
    )
    if len(segment_fits) < 2:
        return None
    if _weighted_score(segment_fits) - fit.score < config.split_score_improvement_threshold:
        return None
    return segment_fits


def _merge_adjacent_segment_fits(
    segment_fits: Sequence[_SegmentFit],
    signal: NDArray[np.float64],
    *,
    frame_times_ms: NDArray[np.float64],
    downbeat_signal: NDArray[np.float64] | None,
    config: GridFitterConfig,
    fit_cache: dict[tuple[int, int], _SegmentFit],
) -> list[_SegmentFit]:
    merged_fits: list[_SegmentFit] = []
    has_downbeat_signal = (
        downbeat_signal is not None
        and float(np.linalg.norm(downbeat_signal - float(np.mean(downbeat_signal)))) > 0.0
    )
    allow_alias_merge = (
        len(segment_fits) >= config.merge_alias_min_segments
        and (has_downbeat_signal or not config.merge_alias_requires_downbeat_signal)
    )
    allow_loose_similar_merge = len(segment_fits) >= config.merge_many_similar_min_segments
    for fit in segment_fits:
        if not merged_fits:
            merged_fits.append(fit)
            continue
        previous_fit = merged_fits[-1]
        pair_allows_alias_merge = allow_alias_merge and fit.score <= config.merge_alias_max_fit_score
        if not _segment_fits_are_mergeable(
            previous_fit,
            fit,
            frame_times_ms=frame_times_ms,
            config=config,
            allow_alias=pair_allows_alias_merge,
            allow_loose_similar=allow_loose_similar_merge,
        ):
            merged_fits.append(fit)
            continue
        merged_fits[-1] = _cached_fit_segment_range(
            signal,
            frame_times_ms=frame_times_ms,
            downbeat_signal=downbeat_signal,
            start_frame=previous_fit.start_frame,
            end_frame=fit.end_frame,
            config=config,
            fit_cache=fit_cache,
        )
    return merged_fits


def _best_split_for_fit(
    signal: NDArray[np.float64],
    *,
    frame_times_ms: NDArray[np.float64],
    downbeat_signal: NDArray[np.float64] | None,
    fit: _SegmentFit,
    segment_index: int,
    config: GridFitterConfig,
    fit_cache: dict[tuple[int, int], _SegmentFit],
) -> _EvaluatedSplit | None:
    candidates = _detect_change_split_candidates(
        signal,
        frame_times_ms=frame_times_ms,
        downbeat_signal=downbeat_signal,
        fit=fit,
        config=config,
    )
    best_split: _EvaluatedSplit | None = None
    for candidate in candidates[: config.max_split_candidates_per_segment]:
        left_fit = _cached_fit_segment_range(
            signal,
            frame_times_ms=frame_times_ms,
            downbeat_signal=downbeat_signal,
            start_frame=fit.start_frame,
            end_frame=candidate.frame,
            config=config,
            fit_cache=fit_cache,
        )
        right_fit = _cached_fit_segment_range(
            signal,
            frame_times_ms=frame_times_ms,
            downbeat_signal=downbeat_signal,
            start_frame=candidate.frame,
            end_frame=fit.end_frame,
            config=config,
            fit_cache=fit_cache,
        )
        if _segment_fits_are_mergeable(left_fit, right_fit, frame_times_ms=frame_times_ms, config=config):
            continue
        improvement = _weighted_score((left_fit, right_fit)) - fit.score
        evaluated_split = _EvaluatedSplit(
            segment_index=segment_index,
            candidate=candidate,
            left_fit=left_fit,
            right_fit=right_fit,
            improvement=float(improvement),
        )
        if best_split is None or _split_is_better(evaluated_split, best_split):
            best_split = evaluated_split

    return best_split


def _cached_fit_segment_range(
    signal: NDArray[np.float64],
    *,
    frame_times_ms: NDArray[np.float64],
    downbeat_signal: NDArray[np.float64] | None,
    start_frame: int,
    end_frame: int,
    config: GridFitterConfig,
    fit_cache: dict[tuple[int, int], _SegmentFit],
) -> _SegmentFit:
    cache_key = (start_frame, end_frame)
    if cache_key not in fit_cache:
        fit_cache[cache_key] = _fit_segment_range(
            signal,
            frame_times_ms=frame_times_ms,
            downbeat_signal=downbeat_signal,
            start_frame=start_frame,
            end_frame=end_frame,
            config=config,
        )
    return fit_cache[cache_key]


def _split_is_better(candidate: _EvaluatedSplit, incumbent: _EvaluatedSplit) -> bool:
    if candidate.improvement > incumbent.improvement:
        return True
    if candidate.improvement < incumbent.improvement:
        return False
    return candidate.candidate.score > incumbent.candidate.score
