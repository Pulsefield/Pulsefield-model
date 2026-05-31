from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from pulsefield_model.timing.grid_fitting.config import GridFitterConfig
from pulsefield_model.timing.grid_fitting.types import _GridCandidate

_FRACTIONAL_BPM_SEARCH_RADIUS = 0.25
_MIN_FRACTIONAL_BPM = 80.0
_FRACTIONAL_BPM_SCORE_MARGIN = 0.08
_FRACTIONAL_BPM_PARTS = tuple(
    sorted(
        {
            *(fraction / 9.0 for fraction in range(1, 9)),
            *(fraction / 8.0 for fraction in (1, 3, 5, 7)),
            1.0 / 3.0,
            2.0 / 3.0,
        }
    )
)


def _candidate_period_frame_bounds(
    frame_rate_hz: float,
    *,
    config: GridFitterConfig,
) -> tuple[int, int]:
    min_period_frames = int(np.ceil(frame_rate_hz * 60.0 / config.max_bpm))
    max_period_frames = int(np.floor(frame_rate_hz * 60.0 / config.min_bpm))
    if min_period_frames <= 0 or max_period_frames < min_period_frames:
        raise ValueError(
            "configured tempo range does not produce valid frame periods: "
            f"frame_rate_hz={frame_rate_hz}, min_bpm={config.min_bpm}, max_bpm={config.max_bpm}",
        )
    return min_period_frames, max_period_frames


def _candidate_bpms(
    centered_signal: NDArray[np.float64],
    *,
    frame_rate_hz: float,
    config: GridFitterConfig,
) -> NDArray[np.float64]:
    candidate_bpms = _autocorrelation_candidate_bpms(
        centered_signal,
        frame_rate_hz=frame_rate_hz,
        config=config,
    )
    if candidate_bpms.shape[0] == 0:
        candidate_bpms = np.arange(
            config.min_bpm,
            config.max_bpm + config.bpm_step * 0.5,
            config.bpm_step,
            dtype=np.float64,
        )
    coarse_candidate_bpms = _limit_bpm_candidates_by_grid_count(candidate_bpms, config=config)
    return _with_fractional_bpm_candidates(coarse_candidate_bpms, config=config)


def _autocorrelation_candidate_bpms(
    centered_signal: NDArray[np.float64],
    *,
    frame_rate_hz: float,
    config: GridFitterConfig,
) -> NDArray[np.float64]:
    min_period_frames, max_period_frames = _candidate_period_frame_bounds(frame_rate_hz, config=config)
    max_period_frames = min(max_period_frames, centered_signal.shape[0] - 1)
    if max_period_frames < min_period_frames:
        return np.asarray([], dtype=np.float64)

    lag_scores: list[tuple[float, int]] = []
    for lag in range(min_period_frames, max_period_frames + 1):
        left = centered_signal[:-lag]
        right = centered_signal[lag:]
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator == 0.0:
            continue
        lag_scores.append((float(np.dot(left, right) / denominator), lag))
    if not lag_scores:
        return np.asarray([], dtype=np.float64)

    lag_scores.sort(reverse=True)
    selected_lags = [lag for _, lag in lag_scores[: config.autocorrelation_candidate_count]]

    exact_candidates: list[float] = []
    expanded_candidates: list[float] = []
    for lag in selected_lags:
        base_bpm = 60.0 * frame_rate_hz / lag
        for multiplier in (1.0, 2.0, 0.5, 3.0, 4.0, 1.0 / 3.0, 0.25):
            candidate_bpm = base_bpm * multiplier
            exact_candidates.append(candidate_bpm)
            expanded_candidates.extend(_expanded_bpm_window(candidate_bpm, config=config))

    return _ordered_unique_bpms([*exact_candidates, *expanded_candidates], config=config)


def _expanded_bpm_window(bpm: float, *, config: GridFitterConfig) -> list[float]:
    if not np.isfinite(bpm) or bpm < config.min_bpm or bpm > config.max_bpm:
        return []

    window_radius_bpm = max(config.bpm_search_window_min_bpm, bpm * config.bpm_search_window_ratio)
    start_bpm = max(config.min_bpm, _quantize_bpm(bpm - window_radius_bpm, config=config))
    end_bpm = min(config.max_bpm, _quantize_bpm(bpm + window_radius_bpm, config=config))
    return [
        float(value)
        for value in np.arange(
            start_bpm,
            end_bpm + config.bpm_step * 0.5,
            config.bpm_step,
            dtype=np.float64,
        )
    ]


def _quantize_bpm(bpm: float, *, config: GridFitterConfig) -> float:
    steps = round((bpm - config.min_bpm) / config.bpm_step)
    return float(config.min_bpm + steps * config.bpm_step)


def _ordered_unique_bpms(bpms: Sequence[float], *, config: GridFitterConfig) -> NDArray[np.float64]:
    seen: set[float] = set()
    ordered: list[float] = []
    for bpm in bpms:
        quantized_bpm = _quantize_bpm(float(bpm), config=config)
        if quantized_bpm < config.min_bpm or quantized_bpm > config.max_bpm:
            continue
        if quantized_bpm in seen:
            continue
        seen.add(quantized_bpm)
        ordered.append(quantized_bpm)
    return np.asarray(ordered, dtype=np.float64)


def _with_fractional_bpm_candidates(
    candidate_bpms: NDArray[np.float64],
    *,
    config: GridFitterConfig,
) -> NDArray[np.float64]:
    seen: set[float] = set()
    expanded: list[float] = []
    for bpm in candidate_bpms:
        for candidate_bpm in _fractional_bpm_candidates_near(float(bpm)):
            rounded_bpm = round(candidate_bpm, 6)
            if rounded_bpm < config.min_bpm or rounded_bpm > config.max_bpm:
                continue
            if rounded_bpm in seen:
                continue
            seen.add(rounded_bpm)
            expanded.append(float(rounded_bpm))
    return np.asarray(expanded, dtype=np.float64)


def _fractional_bpm_candidates_near(bpm: float) -> list[float]:
    candidates = [float(bpm)]
    if bpm < _MIN_FRACTIONAL_BPM:
        return candidates
    integer_part = int(np.floor(bpm))
    for candidate_integer_part in (integer_part - 1, integer_part, integer_part + 1):
        for fractional_part in _FRACTIONAL_BPM_PARTS:
            candidate_bpm = float(candidate_integer_part) + fractional_part
            if abs(candidate_bpm - bpm) <= _FRACTIONAL_BPM_SEARCH_RADIUS:
                candidates.append(candidate_bpm)
    return sorted(candidates, key=lambda candidate_bpm: (abs(candidate_bpm - bpm), candidate_bpm))


def _limit_bpm_candidates_by_grid_count(
    candidate_bpms: NDArray[np.float64],
    *,
    config: GridFitterConfig,
) -> NDArray[np.float64]:
    selected: list[float] = []
    grid_candidate_count = 0
    for bpm in candidate_bpms:
        beat_length_ms = 60000.0 / float(bpm)
        offset_count = max(1, int(np.ceil(beat_length_ms / config.offset_step_ms)))
        if selected and grid_candidate_count + offset_count > config.max_grid_candidates_per_segment:
            continue
        selected.append(float(bpm))
        grid_candidate_count += offset_count
    if not selected:
        return candidate_bpms[:1]
    return np.asarray(selected, dtype=np.float64)


def _candidate_offsets_ms(beat_length_ms: float, *, config: GridFitterConfig) -> NDArray[np.float64]:
    offsets = np.arange(0.0, beat_length_ms, config.offset_step_ms, dtype=np.float64)
    if offsets.shape[0] == 0:
        return np.asarray([0.0], dtype=np.float64)
    return offsets


def _best_grid_candidate(
    candidates: Sequence[_GridCandidate],
    *,
    downbeat_centered_signal: NDArray[np.float64] | None,
    downbeat_signal_norm: float,
    frame_times_ms: NDArray[np.float64],
    config: GridFitterConfig,
) -> tuple[float, float, float, float, float]:
    if not candidates:
        return -np.inf, -np.inf, config.min_bpm, 60000.0 / config.min_bpm, 0.0

    sorted_candidates = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
    best_candidate = _best_grid_candidate_with_fractional_margin(sorted_candidates, config=config)
    if downbeat_centered_signal is None or downbeat_signal_norm == 0.0:
        return (
            float(best_candidate.score),
            -np.inf,
            float(best_candidate.bpm),
            float(best_candidate.beat_length_ms),
            float(best_candidate.offset_ms),
        )

    best_score = -np.inf
    best_downbeat_score = -np.inf
    best_bpm = best_candidate.bpm
    best_beat_length_ms = best_candidate.beat_length_ms
    best_offset_ms = best_candidate.offset_ms
    for candidate in sorted_candidates[: config.downbeat_refine_candidate_count]:
        downbeat_score, downbeat_offset_ms = _best_downbeat_grid_fit(
            downbeat_centered_signal,
            downbeat_signal_norm=downbeat_signal_norm,
            frame_times_ms=frame_times_ms,
            beat_length_ms=candidate.beat_length_ms,
            offset_ms=candidate.offset_ms,
            config=config,
        )
        if _grid_candidate_is_better(
            candidate.score,
            downbeat_score,
            best_score,
            best_downbeat_score,
            config=config,
        ):
            best_score = candidate.score
            best_downbeat_score = downbeat_score
            best_bpm = candidate.bpm
            best_beat_length_ms = candidate.beat_length_ms
            best_offset_ms = downbeat_offset_ms

    return (
        float(best_score),
        float(best_downbeat_score),
        float(best_bpm),
        float(best_beat_length_ms),
        float(best_offset_ms),
    )


def _best_bpm_fit(
    centered_signal: NDArray[np.float64],
    *,
    signal_norm: float,
    frame_times_ms: NDArray[np.float64],
    bpm: float,
    pulse_width_ms: float,
    downbeat_centered_signal: NDArray[np.float64] | None = None,
    downbeat_signal_norm: float = 0.0,
    config: GridFitterConfig,
) -> tuple[float, float, float]:
    if bpm < config.min_bpm or bpm > config.max_bpm:
        return -np.inf, 0.0, -np.inf

    beat_length_ms = 60000.0 / bpm
    candidates: list[_GridCandidate] = []
    for offset_ms in _candidate_offsets_ms(beat_length_ms, config=config):
        score = _score_grid(
            centered_signal,
            signal_norm=signal_norm,
            frame_times_ms=frame_times_ms,
            beat_length_ms=beat_length_ms,
            offset_ms=offset_ms,
            pulse_width_ms=pulse_width_ms,
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
    best_score, best_offset_ms, best_downbeat_score = _refine_grid_offset_ms(
        centered_signal,
        signal_norm=signal_norm,
        downbeat_centered_signal=downbeat_centered_signal,
        downbeat_signal_norm=downbeat_signal_norm,
        frame_times_ms=frame_times_ms,
        beat_length_ms=beat_length_ms,
        offset_ms=best_offset_ms,
        score=best_score,
        downbeat_score=best_downbeat_score,
        config=config,
    )
    return float(best_score), best_offset_ms, float(best_downbeat_score)


def _refine_grid_offset_ms(
    centered_signal: NDArray[np.float64],
    *,
    signal_norm: float,
    downbeat_centered_signal: NDArray[np.float64] | None,
    downbeat_signal_norm: float,
    frame_times_ms: NDArray[np.float64],
    beat_length_ms: float,
    offset_ms: float,
    score: float,
    downbeat_score: float,
    config: GridFitterConfig,
) -> tuple[float, float, float]:
    if config.offset_refine_step_ms <= 0.0 or config.offset_refine_radius_ms <= 0.0:
        return float(score), float(offset_ms), float(downbeat_score)

    step_ms = float(config.offset_refine_step_ms)
    radius_ms = float(config.offset_refine_radius_ms)
    center_offset_ms = round(float(offset_ms) / step_ms) * step_ms
    step_count = int(np.ceil(radius_ms / step_ms))

    best_score = float(score)
    best_downbeat_score = float(downbeat_score)
    best_offset_ms = float(offset_ms)
    downbeat_period_ms = beat_length_ms * config.downbeat_period_beats
    for offset_index in range(-step_count, step_count + 1):
        candidate_offset_ms = center_offset_ms + float(offset_index) * step_ms
        candidate_score = _score_grid(
            centered_signal,
            signal_norm=signal_norm,
            frame_times_ms=frame_times_ms,
            beat_length_ms=beat_length_ms,
            offset_ms=candidate_offset_ms,
            pulse_width_ms=config.pulse_width_ms,
        )
        if downbeat_centered_signal is None or downbeat_signal_norm == 0.0:
            candidate_downbeat_score = -np.inf
        else:
            candidate_downbeat_score = _score_grid(
                downbeat_centered_signal,
                signal_norm=downbeat_signal_norm,
                frame_times_ms=frame_times_ms,
                beat_length_ms=downbeat_period_ms,
                offset_ms=candidate_offset_ms,
                pulse_width_ms=config.pulse_width_ms,
            )
        if _grid_candidate_is_better(
            candidate_score,
            candidate_downbeat_score,
            best_score,
            best_downbeat_score,
            config=config,
        ):
            best_score = float(candidate_score)
            best_downbeat_score = float(candidate_downbeat_score)
            best_offset_ms = float(candidate_offset_ms)

    return float(best_score), float(best_offset_ms), float(best_downbeat_score)


def _centered_signal_and_norm(
    signal: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64] | None, float]:
    if signal is None:
        return None, 0.0
    centered_signal = signal - float(np.mean(signal))
    signal_norm = float(np.linalg.norm(centered_signal))
    if signal_norm == 0.0:
        return None, 0.0
    return centered_signal, signal_norm


def _best_downbeat_grid_fit(
    downbeat_centered_signal: NDArray[np.float64] | None,
    *,
    downbeat_signal_norm: float,
    frame_times_ms: NDArray[np.float64],
    beat_length_ms: float,
    offset_ms: float,
    config: GridFitterConfig,
) -> tuple[float, float]:
    if downbeat_centered_signal is None or downbeat_signal_norm == 0.0:
        return -np.inf, float(offset_ms)

    downbeat_period_ms = beat_length_ms * config.downbeat_period_beats
    best_score = -np.inf
    best_offset_ms = float(offset_ms)
    for beat_index in range(config.downbeat_period_beats):
        candidate_offset_ms = float(offset_ms + beat_index * beat_length_ms)
        score = _score_grid(
            downbeat_centered_signal,
            signal_norm=downbeat_signal_norm,
            frame_times_ms=frame_times_ms,
            beat_length_ms=downbeat_period_ms,
            offset_ms=candidate_offset_ms,
            pulse_width_ms=config.pulse_width_ms,
        )
        if score > best_score:
            best_score = score
            best_offset_ms = candidate_offset_ms
    return float(best_score), best_offset_ms


def _best_grid_candidate_with_fractional_margin(
    sorted_candidates: Sequence[_GridCandidate],
    *,
    config: GridFitterConfig,
) -> _GridCandidate:
    best_candidate = sorted_candidates[0]
    if not _uses_fractional_bpm_candidate(best_candidate.bpm, config=config):
        return best_candidate

    best_coarse_candidate = next(
        (
            candidate
            for candidate in sorted_candidates
            if not _uses_fractional_bpm_candidate(candidate.bpm, config=config)
        ),
        None,
    )
    if best_coarse_candidate is None:
        return best_candidate
    if best_candidate.score >= best_coarse_candidate.score + _FRACTIONAL_BPM_SCORE_MARGIN:
        return best_candidate
    return best_coarse_candidate


def _uses_fractional_bpm_candidate(bpm: float, *, config: GridFitterConfig) -> bool:
    return abs(_quantize_bpm(bpm, config=config) - bpm) > 1e-6


def _grid_candidate_is_better(
    score: float,
    downbeat_score: float,
    best_score: float,
    best_downbeat_score: float,
    *,
    config: GridFitterConfig,
) -> bool:
    if score > best_score + config.downbeat_tie_score_margin:
        return True
    if score < best_score - config.downbeat_tie_score_margin:
        return False
    if np.isfinite(downbeat_score) or np.isfinite(best_downbeat_score):
        if downbeat_score > best_downbeat_score + config.downbeat_tie_score_margin:
            return True
        if downbeat_score < best_downbeat_score - config.downbeat_tie_score_margin:
            return False
    return score > best_score


def _downbeat_rejects_close_tempo_alias(
    raw_score: float,
    raw_downbeat_score: float,
    alias_score: float,
    alias_downbeat_score: float,
    *,
    config: GridFitterConfig,
) -> bool:
    if not np.isfinite(raw_downbeat_score) or not np.isfinite(alias_downbeat_score):
        return False
    if alias_score > raw_score + config.downbeat_tie_score_margin:
        return False
    return raw_downbeat_score > alias_downbeat_score + config.downbeat_tie_score_margin


def _score_grid(
    centered_signal: NDArray[np.float64],
    *,
    signal_norm: float,
    frame_times_ms: NDArray[np.float64],
    beat_length_ms: float,
    offset_ms: float,
    pulse_width_ms: float,
) -> float:
    frame_count = frame_times_ms.shape[0]
    if frame_count == 0:
        return -np.inf

    template_sum, template_sum_squares, signal_template_dot = _pulse_template_stats(
        centered_signal,
        frame_times_ms=frame_times_ms,
        beat_length_ms=beat_length_ms,
        offset_ms=offset_ms,
        pulse_width_ms=pulse_width_ms,
    )
    centered_template_sum_squares = template_sum_squares - (template_sum * template_sum / float(frame_count))
    if centered_template_sum_squares <= 0.0:
        return -np.inf
    return float(signal_template_dot / (signal_norm * float(np.sqrt(centered_template_sum_squares))))


def _pulse_template_stats(
    centered_signal: NDArray[np.float64],
    *,
    frame_times_ms: NDArray[np.float64],
    beat_length_ms: float,
    offset_ms: float,
    pulse_width_ms: float,
) -> tuple[float, float, float]:
    if beat_length_ms <= pulse_width_ms * 2.0:
        return _dense_pulse_template_stats(
            centered_signal,
            frame_times_ms=frame_times_ms,
            beat_length_ms=beat_length_ms,
            offset_ms=offset_ms,
            pulse_width_ms=pulse_width_ms,
        )
    return _sparse_pulse_template_stats(
        centered_signal,
        frame_times_ms=frame_times_ms,
        beat_length_ms=beat_length_ms,
        offset_ms=offset_ms,
        pulse_width_ms=pulse_width_ms,
    )


def _dense_pulse_template_stats(
    centered_signal: NDArray[np.float64],
    *,
    frame_times_ms: NDArray[np.float64],
    beat_length_ms: float,
    offset_ms: float,
    pulse_width_ms: float,
) -> tuple[float, float, float]:
    phase_ms = np.mod(frame_times_ms - offset_ms, beat_length_ms)
    distance_ms = np.minimum(phase_ms, beat_length_ms - phase_ms)
    weights = np.maximum(0.0, 1.0 - distance_ms / pulse_width_ms)
    return (
        float(np.sum(weights)),
        float(np.dot(weights, weights)),
        float(np.dot(centered_signal, weights)),
    )


def _sparse_pulse_template_stats(
    centered_signal: NDArray[np.float64],
    *,
    frame_times_ms: NDArray[np.float64],
    beat_length_ms: float,
    offset_ms: float,
    pulse_width_ms: float,
) -> tuple[float, float, float]:
    first_frame_time_ms = float(frame_times_ms[0])
    last_frame_time_ms = float(frame_times_ms[-1])
    if frame_times_ms.shape[0] == 1:
        frame_step_ms = pulse_width_ms
    else:
        frame_step_ms = float(
            (last_frame_time_ms - first_frame_time_ms) / float(frame_times_ms.shape[0] - 1)
        )
        if frame_step_ms <= 0.0:
            return _dense_pulse_template_stats(
                centered_signal,
                frame_times_ms=frame_times_ms,
                beat_length_ms=beat_length_ms,
                offset_ms=offset_ms,
                pulse_width_ms=pulse_width_ms,
            )

    first_beat_index = int(np.floor((first_frame_time_ms - offset_ms - pulse_width_ms) / beat_length_ms))
    last_beat_index = int(np.ceil((last_frame_time_ms - offset_ms + pulse_width_ms) / beat_length_ms))
    beat_times_ms = offset_ms + np.arange(first_beat_index, last_beat_index + 1, dtype=np.float64) * beat_length_ms
    if beat_times_ms.shape[0] == 0:
        return 0.0, 0.0, 0.0

    radius_frames = int(np.ceil(pulse_width_ms / frame_step_ms)) + 2
    if beat_times_ms.shape[0] * (2 * radius_frames + 1) >= frame_times_ms.shape[0]:
        return _dense_pulse_template_stats(
            centered_signal,
            frame_times_ms=frame_times_ms,
            beat_length_ms=beat_length_ms,
            offset_ms=offset_ms,
            pulse_width_ms=pulse_width_ms,
        )

    relative_frame_indices = np.arange(-radius_frames, radius_frames + 1, dtype=np.int64)
    center_frame_indices = np.rint((beat_times_ms - first_frame_time_ms) / frame_step_ms).astype(np.int64)
    frame_indices = center_frame_indices[:, np.newaxis] + relative_frame_indices[np.newaxis, :]
    valid_frame_mask = (frame_indices >= 0) & (frame_indices < frame_times_ms.shape[0])
    if not np.any(valid_frame_mask):
        return 0.0, 0.0, 0.0

    clamped_frame_indices = np.where(valid_frame_mask, frame_indices, 0)
    candidate_frame_times_ms = frame_times_ms[clamped_frame_indices]
    distances_ms = np.abs(candidate_frame_times_ms - beat_times_ms[:, np.newaxis])
    weights = 1.0 - distances_ms / pulse_width_ms
    support_mask = valid_frame_mask & (weights > 0.0)
    if not np.any(support_mask):
        return 0.0, 0.0, 0.0

    support_weights = weights[support_mask]
    support_frame_indices = frame_indices[support_mask]
    return (
        float(np.sum(support_weights)),
        float(np.dot(support_weights, support_weights)),
        float(np.dot(centered_signal[support_frame_indices], support_weights)),
    )
