from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from pulsefield_model.timing.grid_fitting.config import GridFitterConfig
from pulsefield_model.timing.grid_fitting.scoring import (
    _best_grid_candidate,
    _centered_signal_and_norm,
    _score_grid,
)
from pulsefield_model.timing.grid_fitting.segments import (
    _bpms_are_alias_compatible,
    _timing_segments_from_fits,
)
from pulsefield_model.timing.grid_fitting.types import _GridCandidate, _SegmentFit
from pulsefield_model.timing.schema import TimingSegment


DEFAULT_TEMPO_ALIAS_MULTIPLIERS = (0.25, 1.0 / 3.0, 0.5, 1.0, 2.0, 3.0, 4.0)


@dataclass(frozen=True)
class _AliasOption:
    fit: _SegmentFit
    local_score: float
    change_kind: str = "same"
    beat_support_ratio: float = np.inf
    has_strong_evidence: bool = True


@dataclass(frozen=True)
class _AliasCanonicalizationResult:
    segment_fits: tuple[_SegmentFit, ...]
    alias_candidate_count: int


def _canonicalize_tempo_aliases(
    segment_fits: Sequence[_SegmentFit],
    signal: NDArray[np.float64],
    *,
    frame_times_ms: NDArray[np.float64],
    downbeat_signal: NDArray[np.float64] | None,
    config: GridFitterConfig,
) -> _AliasCanonicalizationResult:
    if not config.canonicalize_tempo_aliases or not segment_fits:
        return _AliasCanonicalizationResult(tuple(segment_fits), alias_candidate_count=0)

    options_by_segment: list[list[_AliasOption]] = []
    alias_candidate_count = 0
    for fit in segment_fits:
        options, evaluated_candidate_count = _alias_options_for_fit(
            fit,
            signal,
            frame_times_ms=frame_times_ms,
            downbeat_signal=downbeat_signal,
            config=config,
            allow_lower_promoted_aliases=len(segment_fits) > 1,
        )
        options_by_segment.append(options)
        alias_candidate_count += evaluated_candidate_count

    best_options = _best_alias_path(options_by_segment, config=config)
    best_fits = tuple(option.fit for option in best_options)
    if not _alias_path_is_acceptable(
        tuple(segment_fits),
        best_options,
        frame_times_ms=frame_times_ms,
        config=config,
    ):
        best_fits = tuple(segment_fits)
    return _AliasCanonicalizationResult(best_fits, alias_candidate_count=alias_candidate_count)


def _alias_options_for_fit(
    fit: _SegmentFit,
    signal: NDArray[np.float64],
    *,
    frame_times_ms: NDArray[np.float64],
    downbeat_signal: NDArray[np.float64] | None,
    config: GridFitterConfig,
    allow_lower_promoted_aliases: bool,
) -> tuple[list[_AliasOption], int]:
    segment_signal = signal[fit.start_frame : fit.end_frame]
    centered_signal = segment_signal - float(np.mean(segment_signal))
    signal_norm = float(np.linalg.norm(centered_signal))
    if signal_norm == 0.0:
        return [_AliasOption(fit=fit, local_score=fit.score)], 0

    segment_frame_times_ms = frame_times_ms[fit.start_frame : fit.end_frame]
    segment_downbeat_signal = (
        None
        if downbeat_signal is None
        else downbeat_signal[fit.start_frame : fit.end_frame]
    )
    downbeat_centered_signal, downbeat_signal_norm = _centered_signal_and_norm(segment_downbeat_signal)
    current_downbeat_score = -np.inf
    options_by_bpm: dict[float, _AliasOption] = {
        round(fit.bpm, 6): _AliasOption(
            fit=fit,
            local_score=_alias_local_score(
                fit.score,
                bpm=fit.bpm,
                downbeat_score=current_downbeat_score,
                is_current_alias=True,
                config=config,
            ),
        ),
    }
    evaluated_candidate_count = 0

    base_bpm = _alias_base_bpm(fit)
    allow_lower_aliases = allow_lower_promoted_aliases or downbeat_signal_norm > 0.0
    for multiplier in config.alias_tempo_multipliers:
        bpm = base_bpm * float(multiplier)
        if (
            not allow_lower_aliases
            and abs(fit.tempo_multiplier - 1.0) > 1e-9
            and bpm < fit.bpm
        ):
            continue
        if bpm < config.min_bpm or bpm > config.max_bpm or not np.isfinite(bpm):
            continue
        if _bpms_are_close(bpm, fit.bpm):
            continue
        beat_length_ms = 60000.0 / bpm
        candidate_offsets_ms = _alias_candidate_offsets_ms(
            fit,
            candidate_beat_length_ms=beat_length_ms,
        )
        evaluated_candidate_count += int(candidate_offsets_ms.shape[0])
        score, offset_ms, downbeat_score = _best_phase_locked_alias_fit(
            centered_signal,
            signal_norm=signal_norm,
            frame_times_ms=segment_frame_times_ms,
            bpm=bpm,
            beat_length_ms=beat_length_ms,
            candidate_offsets_ms=candidate_offsets_ms,
            downbeat_centered_signal=downbeat_centered_signal,
            downbeat_signal_norm=downbeat_signal_norm,
            config=config,
        )
        semantic_promotion = _alias_is_semantic_promotion(
            fit.bpm,
            bpm,
            score=score,
            current_score=fit.score,
            config=config,
        )
        if not np.isfinite(score) or (
            not _alias_score_is_close(score, fit.score, config=config)
            and not semantic_promotion
        ):
            continue

        change_kind = _alias_change_kind(fit.bpm, bpm)
        candidate_fit = replace(
            fit,
            score=float(score),
            beat_length_ms=float(beat_length_ms),
            offset_ms=float(offset_ms),
            tempo_multiplier=float(multiplier),
        )
        beat_support_ratio = _alias_beat_support_ratio(
            fit,
            candidate_fit,
            segment_signal,
            frame_times_ms=segment_frame_times_ms,
            change_kind=change_kind,
            config=config,
        )
        has_strong_evidence = _alias_change_has_density_evidence(
            change_kind,
            beat_support_ratio=beat_support_ratio,
            config=config,
        )
        if semantic_promotion:
            has_strong_evidence = True
        if not has_strong_evidence:
            continue

        local_score = _alias_local_score(
            score,
            bpm=bpm,
            downbeat_score=downbeat_score,
            is_current_alias=_bpms_are_close(bpm, fit.bpm),
            config=config,
        )
        if semantic_promotion:
            local_score += config.alias_semantic_promotion_bonus
        bpm_key = round(bpm, 6)
        previous = options_by_bpm.get(bpm_key)
        if previous is None or local_score > previous.local_score:
            options_by_bpm[bpm_key] = _AliasOption(
                candidate_fit,
                local_score,
                change_kind=change_kind,
                beat_support_ratio=beat_support_ratio,
                has_strong_evidence=has_strong_evidence,
            )

    if not options_by_bpm:
        return [_AliasOption(fit=fit, local_score=fit.score)], evaluated_candidate_count

    return sorted(options_by_bpm.values(), key=lambda option: option.fit.bpm), evaluated_candidate_count


def _alias_base_bpm(fit: _SegmentFit) -> float:
    if np.isfinite(fit.raw_bpm) and fit.raw_bpm > 0.0:
        return float(fit.raw_bpm)
    if np.isfinite(fit.tempo_multiplier) and fit.tempo_multiplier > 0.0:
        base_bpm = fit.bpm / fit.tempo_multiplier
        if np.isfinite(base_bpm) and base_bpm > 0.0:
            return float(base_bpm)
    return float(fit.bpm)


def _alias_candidate_offsets_ms(
    fit: _SegmentFit,
    *,
    candidate_beat_length_ms: float,
) -> NDArray[np.float64]:
    if candidate_beat_length_ms <= 0.0 or not np.isfinite(candidate_beat_length_ms):
        return np.asarray([], dtype=np.float64)
    current_beat_length_ms = fit.beat_length_ms
    if candidate_beat_length_ms > current_beat_length_ms:
        phase_count = max(1, int(round(candidate_beat_length_ms / current_beat_length_ms)))
        offsets = [
            fit.offset_ms + phase_index * current_beat_length_ms
            for phase_index in range(phase_count)
        ]
    else:
        offsets = [fit.offset_ms]

    unique_offsets: list[float] = []
    seen: set[float] = set()
    for offset_ms in np.mod(np.asarray(offsets, dtype=np.float64), candidate_beat_length_ms):
        key = round(float(offset_ms), 6)
        if key in seen:
            continue
        seen.add(key)
        unique_offsets.append(float(offset_ms))
    if not unique_offsets:
        return np.asarray([0.0], dtype=np.float64)
    return np.asarray(sorted(unique_offsets), dtype=np.float64)


def _best_phase_locked_alias_fit(
    centered_signal: NDArray[np.float64],
    *,
    signal_norm: float,
    frame_times_ms: NDArray[np.float64],
    bpm: float,
    beat_length_ms: float,
    candidate_offsets_ms: NDArray[np.float64],
    downbeat_centered_signal: NDArray[np.float64] | None,
    downbeat_signal_norm: float,
    config: GridFitterConfig,
) -> tuple[float, float, float]:
    candidates: list[_GridCandidate] = []
    for offset_ms in candidate_offsets_ms:
        score = _score_grid(
            centered_signal,
            signal_norm=signal_norm,
            frame_times_ms=frame_times_ms,
            beat_length_ms=beat_length_ms,
            offset_ms=float(offset_ms),
            pulse_width_ms=config.pulse_width_ms,
        )
        candidates.append(
            _GridCandidate(
                score=float(score),
                bpm=float(bpm),
                beat_length_ms=float(beat_length_ms),
                offset_ms=float(offset_ms),
            )
        )
    best_score, best_downbeat_score, _, _, best_offset_ms = _best_grid_candidate(
        candidates,
        downbeat_centered_signal=downbeat_centered_signal,
        downbeat_signal_norm=downbeat_signal_norm,
        frame_times_ms=frame_times_ms,
        config=config,
    )
    return float(best_score), float(best_offset_ms), float(best_downbeat_score)


def _bpms_are_close(lhs: float, rhs: float) -> bool:
    return abs(float(lhs) - float(rhs)) <= max(1e-6, abs(float(rhs)) * 1e-9)


def _alias_score_is_close(score: float, current_score: float, *, config: GridFitterConfig) -> bool:
    if not np.isfinite(current_score):
        return True
    if score + config.alias_score_tie_margin >= current_score:
        return True
    if current_score > 0.0 and score >= current_score * config.alias_score_ratio_threshold:
        return True
    return False


def _alias_is_semantic_promotion(
    current_bpm: float,
    candidate_bpm: float,
    *,
    score: float,
    current_score: float,
    config: GridFitterConfig,
) -> bool:
    if current_bpm <= 0.0 or candidate_bpm <= current_bpm:
        return False
    below_preferred_band = current_bpm < config.alias_preferred_min_bpm
    in_half_time_trap_band = (
        config.alias_semantic_promotion_in_band_min_bpm
        <= current_bpm
        <= config.alias_semantic_promotion_current_max_bpm
    )
    if not below_preferred_band and not in_half_time_trap_band:
        return False
    if not (config.alias_preferred_min_bpm <= candidate_bpm <= config.alias_preferred_max_bpm):
        return False
    if not np.isfinite(score):
        return False
    if not np.isfinite(current_score) or current_score <= 0.0:
        return score > 0.0

    score_ratio = float(score / current_score)
    if below_preferred_band:
        return (
            current_score <= config.alias_semantic_promotion_low_bpm_max_fit_score
            and score_ratio >= config.alias_semantic_promotion_score_ratio_threshold
        )
    if score_ratio >= config.alias_semantic_promotion_strong_score_ratio_threshold:
        return True
    return (
        current_score <= config.alias_semantic_promotion_low_confidence_max_fit_score
        and candidate_bpm <= config.alias_semantic_promotion_low_confidence_max_candidate_bpm
        and score_ratio >= config.alias_semantic_promotion_low_confidence_score_ratio_threshold
    )


def _alias_local_score(
    score: float,
    *,
    bpm: float,
    downbeat_score: float,
    is_current_alias: bool,
    config: GridFitterConfig,
) -> float:
    local_score = float(score)
    if np.isfinite(downbeat_score):
        local_score += config.alias_downbeat_score_weight * float(downbeat_score)
    if config.alias_preferred_min_bpm <= bpm <= config.alias_preferred_max_bpm:
        local_score += config.alias_preferred_band_bonus
    if is_current_alias:
        local_score += config.alias_current_tempo_bonus
    return float(local_score)


def _alias_change_kind(current_bpm: float, candidate_bpm: float) -> str:
    if _bpms_are_close(candidate_bpm, current_bpm):
        return "same"
    if candidate_bpm > current_bpm:
        return "promotion"
    return "demotion"


def _alias_change_has_density_evidence(
    change_kind: str,
    *,
    beat_support_ratio: float,
    config: GridFitterConfig,
) -> bool:
    if change_kind == "same":
        return True
    if not np.isfinite(beat_support_ratio):
        return False
    if change_kind == "promotion":
        return beat_support_ratio >= config.alias_promotion_inserted_support_ratio_threshold
    if change_kind == "demotion":
        return beat_support_ratio <= config.alias_demotion_dropped_support_ratio_threshold
    return False


def _alias_beat_support_ratio(
    fit: _SegmentFit,
    candidate_fit: _SegmentFit,
    signal: NDArray[np.float64],
    *,
    frame_times_ms: NDArray[np.float64],
    change_kind: str,
    config: GridFitterConfig,
) -> float:
    if change_kind == "same":
        return np.inf
    if frame_times_ms.shape[0] == 0:
        return np.inf

    if change_kind == "demotion":
        current_beat_times_ms = _grid_beat_times_ms(
            frame_times_ms,
            beat_length_ms=fit.beat_length_ms,
            offset_ms=fit.offset_ms,
        )
        if current_beat_times_ms.shape[0] == 0:
            return np.inf
        kept_mask = _beat_times_match_grid(
            current_beat_times_ms,
            beat_length_ms=candidate_fit.beat_length_ms,
            offset_ms=candidate_fit.offset_ms,
            config=config,
        )
        kept_support = _beat_support_values(
            signal,
            frame_times_ms=frame_times_ms,
            beat_times_ms=current_beat_times_ms[kept_mask],
            pulse_width_ms=config.pulse_width_ms,
        )
        changed_support = _beat_support_values(
            signal,
            frame_times_ms=frame_times_ms,
            beat_times_ms=current_beat_times_ms[~kept_mask],
            pulse_width_ms=config.pulse_width_ms,
        )
    elif change_kind == "promotion":
        candidate_beat_times_ms = _grid_beat_times_ms(
            frame_times_ms,
            beat_length_ms=candidate_fit.beat_length_ms,
            offset_ms=candidate_fit.offset_ms,
        )
        if candidate_beat_times_ms.shape[0] == 0:
            return np.inf
        kept_mask = _beat_times_match_grid(
            candidate_beat_times_ms,
            beat_length_ms=fit.beat_length_ms,
            offset_ms=fit.offset_ms,
            config=config,
        )
        kept_support = _beat_support_values(
            signal,
            frame_times_ms=frame_times_ms,
            beat_times_ms=candidate_beat_times_ms[kept_mask],
            pulse_width_ms=config.pulse_width_ms,
        )
        changed_support = _beat_support_values(
            signal,
            frame_times_ms=frame_times_ms,
            beat_times_ms=candidate_beat_times_ms[~kept_mask],
            pulse_width_ms=config.pulse_width_ms,
        )
    else:
        return np.inf

    if kept_support.shape[0] == 0 or changed_support.shape[0] == 0:
        return np.inf
    kept_mean = float(np.mean(kept_support))
    if kept_mean <= 1e-9:
        return np.inf
    return float(np.mean(changed_support) / kept_mean)


def _grid_beat_times_ms(
    frame_times_ms: NDArray[np.float64],
    *,
    beat_length_ms: float,
    offset_ms: float,
) -> NDArray[np.float64]:
    if frame_times_ms.shape[0] == 0 or beat_length_ms <= 0.0:
        return np.asarray([], dtype=np.float64)
    first_frame_time_ms = float(frame_times_ms[0])
    last_frame_time_ms = float(frame_times_ms[-1])
    first_beat_index = int(np.ceil((first_frame_time_ms - offset_ms) / beat_length_ms))
    last_beat_index = int(np.floor((last_frame_time_ms - offset_ms) / beat_length_ms))
    if last_beat_index < first_beat_index:
        return np.asarray([], dtype=np.float64)
    return offset_ms + np.arange(first_beat_index, last_beat_index + 1, dtype=np.float64) * beat_length_ms


def _beat_times_match_grid(
    beat_times_ms: NDArray[np.float64],
    *,
    beat_length_ms: float,
    offset_ms: float,
    config: GridFitterConfig,
) -> NDArray[np.bool_]:
    if beat_times_ms.shape[0] == 0 or beat_length_ms <= 0.0:
        return np.zeros(beat_times_ms.shape, dtype=np.bool_)
    phase_ms = np.mod(beat_times_ms - offset_ms, beat_length_ms)
    distance_ms = np.minimum(phase_ms, beat_length_ms - phase_ms)
    return distance_ms <= config.alias_beat_match_tolerance_ms


def _beat_support_values(
    signal: NDArray[np.float64],
    *,
    frame_times_ms: NDArray[np.float64],
    beat_times_ms: NDArray[np.float64],
    pulse_width_ms: float,
) -> NDArray[np.float64]:
    if signal.shape[0] == 0 or frame_times_ms.shape[0] == 0 or beat_times_ms.shape[0] == 0:
        return np.asarray([], dtype=np.float64)
    first_frame_time_ms = float(frame_times_ms[0])
    if frame_times_ms.shape[0] == 1:
        frame_step_ms = pulse_width_ms
    else:
        frame_step_ms = float(
            (float(frame_times_ms[-1]) - first_frame_time_ms) / float(frame_times_ms.shape[0] - 1)
        )
        if frame_step_ms <= 0.0:
            return np.asarray([], dtype=np.float64)

    radius_frames = int(np.ceil(pulse_width_ms / frame_step_ms)) + 2
    relative_frame_indices = np.arange(-radius_frames, radius_frames + 1, dtype=np.int64)
    center_frame_indices = np.rint((beat_times_ms - first_frame_time_ms) / frame_step_ms).astype(np.int64)
    frame_indices = center_frame_indices[:, np.newaxis] + relative_frame_indices[np.newaxis, :]
    valid_mask = (frame_indices >= 0) & (frame_indices < signal.shape[0])
    if not np.any(valid_mask):
        return np.zeros(beat_times_ms.shape, dtype=np.float64)

    clamped_indices = np.where(valid_mask, frame_indices, 0)
    candidate_frame_times_ms = frame_times_ms[clamped_indices]
    distances_ms = np.abs(candidate_frame_times_ms - beat_times_ms[:, np.newaxis])
    weights = np.maximum(0.0, 1.0 - distances_ms / pulse_width_ms)
    support = np.where(valid_mask, signal[clamped_indices] * weights, 0.0)
    return np.max(support, axis=1)


def _best_alias_path(
    options_by_segment: Sequence[Sequence[_AliasOption]],
    *,
    config: GridFitterConfig,
) -> tuple[_AliasOption, ...]:
    if not options_by_segment:
        return ()

    scores = [option.local_score for option in options_by_segment[0]]
    paths = [[index] for index in range(len(options_by_segment[0]))]
    for segment_index in range(1, len(options_by_segment)):
        current_options = options_by_segment[segment_index]
        next_scores: list[float] = []
        next_paths: list[list[int]] = []
        for current_index, current_option in enumerate(current_options):
            best_score = -np.inf
            best_path: list[int] | None = None
            for previous_index, previous_score in enumerate(scores):
                previous_option = options_by_segment[segment_index - 1][previous_index]
                score = (
                    previous_score
                    + current_option.local_score
                    - _alias_continuity_penalty(previous_option.fit.bpm, current_option.fit.bpm, config=config)
                )
                if score > best_score:
                    best_score = score
                    best_path = [*paths[previous_index], current_index]
            next_scores.append(float(best_score))
            next_paths.append(best_path if best_path is not None else [current_index])
        scores = next_scores
        paths = next_paths

    best_final_index = int(np.argmax(np.asarray(scores, dtype=np.float64)))
    best_path = paths[best_final_index]
    return tuple(
        options_by_segment[segment_index][option_index]
        for segment_index, option_index in enumerate(best_path)
    )


def _alias_path_is_acceptable(
    original_fits: Sequence[_SegmentFit],
    proposed_options: Sequence[_AliasOption],
    *,
    frame_times_ms: NDArray[np.float64],
    config: GridFitterConfig,
) -> bool:
    if len(original_fits) != len(proposed_options):
        return False
    proposed_fits = tuple(option.fit for option in proposed_options)
    changed_options = [
        option
        for original_fit, option in zip(original_fits, proposed_options)
        if not _bpms_are_close(original_fit.bpm, option.fit.bpm)
    ]
    if not changed_options:
        return True
    if any(not option.has_strong_evidence for option in changed_options):
        return False
    if len(original_fits) > config.alias_semantic_promotion_low_bpm_max_segments and any(
        original_fit.bpm < config.alias_preferred_min_bpm and option.fit.bpm > original_fit.bpm
        for original_fit, option in zip(original_fits, proposed_options)
    ):
        return False

    original_segments = _timing_segments_from_fits(original_fits, frame_times_ms, config=config)
    proposed_segments = _timing_segments_from_fits(proposed_fits, frame_times_ms, config=config)
    if len(proposed_segments) > len(original_segments):
        return False
    if len(proposed_segments) < len(original_segments) and not _alias_score_ratio_is_at_least(
        _weighted_fit_score(proposed_fits),
        _weighted_fit_score(original_fits),
        threshold=config.alias_collapse_score_ratio_threshold,
    ):
        return False
    if _segment_alias_switch_count(proposed_segments, config=config) > _segment_alias_switch_count(
        original_segments,
        config=config,
    ):
        return False

    original_first_fit = original_fits[0]
    proposed_first_fit = proposed_fits[0]
    if (
        proposed_first_fit.bpm < original_first_fit.bpm
        and config.alias_preferred_min_bpm <= original_first_fit.bpm <= config.alias_preferred_max_bpm
    ):
        return False
    return True


def _weighted_fit_score(segment_fits: Sequence[_SegmentFit]) -> float:
    total_frames = sum(max(0, fit.frame_count) for fit in segment_fits)
    if total_frames <= 0:
        return np.nan
    weighted_score = sum(max(0, fit.frame_count) * fit.score for fit in segment_fits)
    return float(weighted_score / total_frames)


def _alias_score_ratio_is_at_least(score: float, current_score: float, *, threshold: float) -> bool:
    if not np.isfinite(score):
        return False
    if not np.isfinite(current_score) or current_score <= 0.0:
        return True
    return score >= current_score * threshold


def _alias_continuity_penalty(previous_bpm: float, bpm: float, *, config: GridFitterConfig) -> float:
    if previous_bpm <= 0.0 or bpm <= 0.0:
        return 0.0
    octave_distance = abs(float(np.log2(bpm / previous_bpm)))
    return float(config.alias_continuity_penalty * octave_distance)


def _segment_alias_switch_count(
    segments: Sequence[TimingSegment],
    *,
    config: GridFitterConfig,
) -> int:
    switch_count = 0
    for previous, segment in zip(segments, segments[1:]):
        if abs(previous.local_bpm - segment.local_bpm) <= config.merge_bpm_tolerance:
            continue
        if _bpms_are_alias_compatible(
            previous.local_bpm,
            segment.local_bpm,
            tolerance_bpm=config.merge_alias_bpm_tolerance,
        ):
            switch_count += 1
    return switch_count


def _tempo_multiplier_distribution(segment_fits: Sequence[_SegmentFit]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for fit in segment_fits:
        key = _format_multiplier(fit.tempo_multiplier)
        distribution[key] = distribution.get(key, 0) + 1
    return distribution


def _format_multiplier(multiplier: float) -> str:
    if abs(multiplier - round(multiplier)) < 1e-9:
        return str(int(round(multiplier)))
    return f"{multiplier:g}"


def _alias_bpm_abs_error(
    predicted_bpm: float,
    oracle_bpm: float,
    *,
    multipliers: Sequence[float] = DEFAULT_TEMPO_ALIAS_MULTIPLIERS,
) -> float:
    if predicted_bpm <= 0.0 or oracle_bpm <= 0.0:
        return abs(float(predicted_bpm) - float(oracle_bpm))
    return float(
        min(
            abs(float(predicted_bpm) * float(multiplier) - float(oracle_bpm))
            for multiplier in multipliers
        )
    )


def _alias_bpm_mae(
    predicted_bpm: NDArray[np.float64],
    oracle_bpm: NDArray[np.float64],
    *,
    multipliers: Sequence[float] = DEFAULT_TEMPO_ALIAS_MULTIPLIERS,
) -> float:
    alias_errors = np.stack(
        [
            np.abs(predicted_bpm * float(multiplier) - oracle_bpm)
            for multiplier in multipliers
        ],
        axis=0,
    )
    return float(np.mean(np.min(alias_errors, axis=0)))


def _distribution_or_default(
    distribution: Mapping[str, int] | None,
    *,
    tempo_multiplier: float,
    segment_count: int,
) -> dict[str, int]:
    if distribution is not None:
        return dict(distribution)
    return {_format_multiplier(tempo_multiplier): int(segment_count)}
