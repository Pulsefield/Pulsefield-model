from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from pulsefield_model.timing.ramp_detection import TimingRampDetection, detect_timing_ramp
from pulsefield_model.timing.schema import FittedTimingGrid, FrameTimingPrediction, TimingSegment


def _require_positive_finite(config: object, fields: tuple[str, ...]) -> None:
    for field in fields:
        value = getattr(config, field)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{field} must be positive and finite, got {value!r}")


def _require_nonnegative_finite(config: object, fields: tuple[str, ...]) -> None:
    for field in fields:
        value = getattr(config, field)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{field} must be non-negative and finite, got {value!r}")


def _require_finite_values(values: tuple[tuple[str, float], ...]) -> None:
    for name, value in values:
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")


@dataclass(frozen=True)
class BeatMaterializationConfig:
    min_bpm: float = 80.0
    max_bpm: float = 500.0
    min_peak_prob: float = 0.25
    peak_quantile: float = 0.75
    local_max_radius_frames: int = 2
    tempo_continuity_penalty: float = 1.75
    min_sequence_beats: int = 8
    min_ramp_points: int = 8
    min_ramp_duration_s: float = 8.0
    min_hint_ramp_duration_s: float = 4.0
    min_ramp_bpm_span: float = 80.0
    min_linear_r2: float = 0.45
    min_sign_consistency: float = 0.62
    max_p90_residual_bpm: float = 42.0
    bpm_smoothing_window: int = 3
    max_candidate_window_points: int = 96
    max_output_points: int = 64
    hint_time_radius_ms: float = 0.0
    hint_time_step_ms: float = 250.0
    hint_bpm_radius: float = 0.0
    hint_bpm_step: float = 10.0
    hint_phase_steps: int = 16
    hint_probability_radius_ms: float = 60.0
    hint_baseline_radius_ms: float = 220.0
    min_hint_probability_score: float = 0.05

    def __post_init__(self) -> None:
        _require_positive_finite(
            self,
            (
                "min_bpm",
                "max_bpm",
                "min_sequence_beats",
                "min_ramp_points",
                "min_ramp_duration_s",
                "min_hint_ramp_duration_s",
                "min_ramp_bpm_span",
                "max_candidate_window_points",
                "max_output_points",
                "hint_time_step_ms",
                "hint_bpm_step",
                "hint_phase_steps",
                "hint_probability_radius_ms",
                "hint_baseline_radius_ms",
            ),
        )
        _require_nonnegative_finite(
            self,
            (
                "min_peak_prob",
                "peak_quantile",
                "tempo_continuity_penalty",
                "min_linear_r2",
                "min_sign_consistency",
                "max_p90_residual_bpm",
                "hint_time_radius_ms",
                "hint_bpm_radius",
                "min_hint_probability_score",
            ),
        )
        if self.min_bpm >= self.max_bpm:
            raise ValueError("min_bpm must be less than max_bpm")
        if self.min_peak_prob > 1.0:
            raise ValueError("min_peak_prob must be in [0, 1]")
        if self.peak_quantile > 1.0:
            raise ValueError("peak_quantile must be in [0, 1]")
        if self.local_max_radius_frames < 1:
            raise ValueError("local_max_radius_frames must be positive")
        if self.bpm_smoothing_window < 1:
            raise ValueError("bpm_smoothing_window must be positive")
        if self.min_sequence_beats < 2:
            raise ValueError("min_sequence_beats must be at least 2")
        if self.min_ramp_points < 2:
            raise ValueError("min_ramp_points must be at least 2")


@dataclass(frozen=True)
class MaterializedBeatSequence:
    frame_indexes: tuple[int, ...]
    beat_times_ms: tuple[float, ...]
    beat_probabilities: tuple[float, ...]
    score: float

    @property
    def beat_count(self) -> int:
        return len(self.beat_times_ms)

    @property
    def start_ms(self) -> float:
        return self.beat_times_ms[0] if self.beat_times_ms else 0.0

    @property
    def end_ms(self) -> float:
        return self.beat_times_ms[-1] if self.beat_times_ms else 0.0

    @property
    def median_bpm(self) -> float:
        intervals_ms = np.diff(np.asarray(self.beat_times_ms, dtype=np.float64))
        if intervals_ms.size == 0:
            return 0.0
        return float(np.median(60000.0 / intervals_ms))

    @property
    def mean_peak_prob(self) -> float:
        if not self.beat_probabilities:
            return 0.0
        return float(np.mean(np.asarray(self.beat_probabilities, dtype=np.float64)))


@dataclass(frozen=True)
class BeatMaterializationResult:
    sequence: MaterializedBeatSequence | None
    peak_count: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RampBeatGridHint:
    start_ms: float
    end_ms: float
    start_bpm: float
    end_bpm: float

    def __post_init__(self) -> None:
        _require_finite_values(
            (
                ("start_ms", self.start_ms),
                ("end_ms", self.end_ms),
                ("start_bpm", self.start_bpm),
                ("end_bpm", self.end_bpm),
            )
        )
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        if self.start_bpm <= 0.0 or self.end_bpm <= 0.0:
            raise ValueError("start_bpm and end_bpm must be positive")


@dataclass(frozen=True)
class RampBeatGridCandidate:
    start_ms: float
    end_ms: float
    duration_s: float
    start_bpm: float
    end_bpm: float
    bpm_span: float
    direction: str
    point_count: int
    beat_count: int
    linear_r2: float
    sign_consistency: float
    p90_residual_bpm: float
    mean_peak_prob: float
    probability_score: float
    phase: float | None
    hypothesis_kind: str
    score: float
    grid: FittedTimingGrid
    ramp_detection: TimingRampDetection


@dataclass(frozen=True)
class RampBeatGridResult:
    accepted: bool
    seconds: float
    reasons: tuple[str, ...]
    materialization: BeatMaterializationResult
    candidate_count: int
    candidate: RampBeatGridCandidate | None


def materialize_beats(
    prediction: FrameTimingPrediction,
    *,
    config: BeatMaterializationConfig = BeatMaterializationConfig(),
) -> BeatMaterializationResult:
    """Decode frame-level beat probabilities into one ordered beat sequence."""

    peak_indexes = _local_peak_indexes(prediction.beat_prob, config=config)
    if peak_indexes.size == 0:
        return BeatMaterializationResult(
            sequence=None,
            peak_count=0,
            reasons=("no_candidate_peaks",),
        )

    sequence = _decode_peak_sequence(
        peak_indexes,
        prediction.beat_prob[peak_indexes],
        frame_rate_hz=prediction.frame_rate_hz,
        config=config,
    )
    if sequence is None:
        return BeatMaterializationResult(
            sequence=None,
            peak_count=int(peak_indexes.size),
            reasons=("no_dp_path",),
        )
    if sequence.beat_count < config.min_sequence_beats:
        return BeatMaterializationResult(
            sequence=sequence,
            peak_count=int(peak_indexes.size),
            reasons=("too_few_materialized_beats",),
        )
    return BeatMaterializationResult(
        sequence=sequence,
        peak_count=int(peak_indexes.size),
        reasons=("materialized_beats",),
    )


def fit_ramp_beat_grid(
    prediction: FrameTimingPrediction,
    *,
    config: BeatMaterializationConfig = BeatMaterializationConfig(),
    hint: RampBeatGridHint | None = None,
    allow_no_hint: bool = False,
) -> RampBeatGridResult:
    start_seconds = time.perf_counter()
    if hint is None and not allow_no_hint:
        return RampBeatGridResult(
            accepted=False,
            seconds=float(time.perf_counter() - start_seconds),
            reasons=("no_hint_fitting_disabled",),
            materialization=BeatMaterializationResult(
                sequence=None,
                peak_count=0,
                reasons=("no_hint_fitting_disabled",),
            ),
            candidate_count=0,
            candidate=None,
        )

    materialization = materialize_beats(prediction, config=config)
    candidates: list[RampBeatGridCandidate] = []
    if hint is not None:
        candidates = _fit_hint_ramp_candidates(prediction, hint=hint, config=config)
    elif (
        allow_no_hint
        and materialization.sequence is not None
        and not _has_failure_reason(materialization.reasons)
    ):
        candidates = _fit_ramp_candidates(materialization.sequence, config=config)

    candidate = max(candidates, key=lambda item: item.score) if candidates else None
    reasons: tuple[str, ...]
    accepted = False
    if candidate is None:
        if hint is not None:
            reasons = ("no_trusted_hint_candidates",)
        else:
            reasons = (
                materialization.reasons
                if _has_failure_reason(materialization.reasons)
                else ("no_ramp_candidates",)
            )
    elif not candidate.ramp_detection.is_ramp:
        reasons = candidate.ramp_detection.reasons
    elif hint is not None and candidate.probability_score < config.min_hint_probability_score:
        reasons = ("low_hint_probability_score",)
    else:
        accepted = True
        reasons = (
            ("trusted_hint_ramp_beat_grid",)
            if hint is not None
            else ("materialized_ramp_beat_grid",)
        )

    return RampBeatGridResult(
        accepted=accepted,
        seconds=float(time.perf_counter() - start_seconds),
        reasons=reasons,
        materialization=materialization,
        candidate_count=len(candidates),
        candidate=candidate,
    )


def ramp_beat_grid_report(result: RampBeatGridResult) -> dict[str, object]:
    sequence = result.materialization.sequence
    report: dict[str, object] = {
        "accepted": result.accepted,
        "seconds": result.seconds,
        "reasons": list(result.reasons),
        "peak_count": result.materialization.peak_count,
        "candidate_count": result.candidate_count,
        "sequence": None,
        "candidate": None,
    }
    if sequence is not None:
        report["sequence"] = {
            "beat_count": sequence.beat_count,
            "start_ms": sequence.start_ms,
            "end_ms": sequence.end_ms,
            "median_bpm": sequence.median_bpm,
            "mean_peak_prob": sequence.mean_peak_prob,
            "score": sequence.score,
        }
    if result.candidate is not None:
        report["candidate"] = _candidate_report(result.candidate)
    return report


def _candidate_report(candidate: RampBeatGridCandidate) -> dict[str, object]:
    return {
        "start_ms": candidate.start_ms,
        "end_ms": candidate.end_ms,
        "duration_s": candidate.duration_s,
        "start_bpm": candidate.start_bpm,
        "end_bpm": candidate.end_bpm,
        "bpm_span": candidate.bpm_span,
        "direction": candidate.direction,
        "point_count": candidate.point_count,
        "beat_count": candidate.beat_count,
        "linear_r2": candidate.linear_r2,
        "sign_consistency": candidate.sign_consistency,
        "p90_residual_bpm": candidate.p90_residual_bpm,
        "mean_peak_prob": candidate.mean_peak_prob,
        "probability_score": candidate.probability_score,
        "phase": candidate.phase,
        "hypothesis_kind": candidate.hypothesis_kind,
        "score": candidate.score,
        "ramp": {
            "is_ramp": candidate.ramp_detection.is_ramp,
            "family": candidate.ramp_detection.family,
            "score": candidate.ramp_detection.score,
            "reasons": list(candidate.ramp_detection.reasons),
            "segment_count": candidate.ramp_detection.segment_count,
        },
        "segments": [
            {
                "offset_ms": segment.offset_ms,
                "beat_length_ms": segment.beat_length_ms,
                "bpm": segment.local_bpm,
                "meter": segment.meter,
            }
            for segment in candidate.grid.segments
        ],
    }


def _local_peak_indexes(
    beat_prob: NDArray[np.float32],
    *,
    config: BeatMaterializationConfig,
) -> NDArray[np.int64]:
    threshold = max(config.min_peak_prob, float(np.quantile(beat_prob, config.peak_quantile)))
    radius = config.local_max_radius_frames
    peak_indexes: list[int] = []
    for index in range(beat_prob.size):
        value = float(beat_prob[index])
        if value < threshold:
            continue
        start = max(0, index - radius)
        end = min(beat_prob.size, index + radius + 1)
        window = beat_prob[start:end]
        if value < float(np.max(window)):
            continue
        if index > start and value == float(beat_prob[index - 1]):
            continue
        peak_indexes.append(index)
    return np.asarray(peak_indexes, dtype=np.int64)


def _decode_peak_sequence(
    peak_indexes: NDArray[np.int64],
    peak_probabilities: NDArray[np.float32],
    *,
    frame_rate_hz: float,
    config: BeatMaterializationConfig,
) -> MaterializedBeatSequence | None:
    min_interval_frames = int(math.ceil(frame_rate_hz * 60.0 / config.max_bpm))
    max_interval_frames = int(math.floor(frame_rate_hz * 60.0 / config.min_bpm))
    if min_interval_frames <= 0 or max_interval_frames < min_interval_frames:
        return None

    predecessors = _predecessor_lists(peak_indexes, min_interval_frames, max_interval_frames)
    scores: dict[tuple[int, int], float] = {}
    lengths: dict[tuple[int, int], int] = {}
    previous_state: dict[tuple[int, int], tuple[int, int] | None] = {}

    best_state: tuple[int, int] | None = None
    best_rank: tuple[int, float] | None = None
    strengths = peak_probabilities.astype(np.float64)
    for current_index, current_predecessors in enumerate(predecessors):
        for previous_index in current_predecessors:
            state = (previous_index, current_index)
            interval_frames = int(peak_indexes[current_index] - peak_indexes[previous_index])
            best_score = float(strengths[previous_index] + strengths[current_index])
            best_length = 2
            best_previous: tuple[int, int] | None = None

            for before_previous_index in predecessors[previous_index]:
                parent = (before_previous_index, previous_index)
                parent_score = scores.get(parent)
                if parent_score is None:
                    continue
                previous_interval_frames = int(
                    peak_indexes[previous_index] - peak_indexes[before_previous_index]
                )
                transition_penalty = config.tempo_continuity_penalty * abs(
                    math.log(interval_frames / previous_interval_frames)
                )
                candidate_score = parent_score + float(strengths[current_index]) - transition_penalty
                candidate_length = lengths[parent] + 1
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_length = candidate_length
                    best_previous = parent

            scores[state] = best_score
            lengths[state] = best_length
            previous_state[state] = best_previous
            rank = (best_length, best_score)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_state = state

    if best_state is None:
        return None

    sequence_indexes = _reconstruct_peak_sequence(best_state, previous_state)
    frame_indexes = tuple(int(peak_indexes[index]) for index in sequence_indexes)
    beat_times_ms = tuple(float(frame_index / frame_rate_hz * 1000.0) for frame_index in frame_indexes)
    beat_probabilities = tuple(float(peak_probabilities[index]) for index in sequence_indexes)
    return MaterializedBeatSequence(
        frame_indexes=frame_indexes,
        beat_times_ms=beat_times_ms,
        beat_probabilities=beat_probabilities,
        score=float(scores[best_state]),
    )


def _predecessor_lists(
    peak_indexes: NDArray[np.int64],
    min_interval_frames: int,
    max_interval_frames: int,
) -> list[list[int]]:
    predecessors: list[list[int]] = []
    for current_index, current_frame in enumerate(peak_indexes):
        current_predecessors: list[int] = []
        for previous_index in range(current_index - 1, -1, -1):
            interval_frames = int(current_frame - peak_indexes[previous_index])
            if interval_frames < min_interval_frames:
                continue
            if interval_frames > max_interval_frames:
                break
            current_predecessors.append(previous_index)
        predecessors.append(current_predecessors)
    return predecessors


def _reconstruct_peak_sequence(
    state: tuple[int, int],
    previous_state: dict[tuple[int, int], tuple[int, int] | None],
) -> list[int]:
    reversed_indexes = [state[1], state[0]]
    cursor = previous_state[state]
    while cursor is not None:
        reversed_indexes.append(cursor[0])
        cursor = previous_state[cursor]
    return list(reversed(reversed_indexes))


def _fit_ramp_candidates(
    sequence: MaterializedBeatSequence,
    *,
    config: BeatMaterializationConfig,
) -> list[RampBeatGridCandidate]:
    beat_times_ms = np.asarray(sequence.beat_times_ms, dtype=np.float64)
    intervals_ms = np.diff(beat_times_ms)
    if intervals_ms.size < config.min_ramp_points:
        return []

    offsets_ms = beat_times_ms[:-1]
    raw_bpms = 60000.0 / intervals_ms
    smoothed_bpms = _median_filter(raw_bpms, config.bpm_smoothing_window)
    probabilities = np.asarray(sequence.beat_probabilities, dtype=np.float64)
    candidates: list[RampBeatGridCandidate] = []
    max_window = min(config.max_candidate_window_points, smoothed_bpms.size)
    for start in range(0, smoothed_bpms.size - config.min_ramp_points + 1):
        last_end = min(smoothed_bpms.size - 1, start + max_window - 1)
        for end in range(start + config.min_ramp_points - 1, last_end + 1):
            candidate = _fit_ramp_window(
                offsets_ms,
                smoothed_bpms,
                probabilities,
                start,
                end,
                config=config,
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def _fit_ramp_window(
    offsets_ms: NDArray[np.float64],
    bpms: NDArray[np.float64],
    probabilities: NDArray[np.float64],
    start: int,
    end: int,
    *,
    config: BeatMaterializationConfig,
) -> RampBeatGridCandidate | None:
    window_offsets_ms = offsets_ms[start : end + 1]
    window_bpms = bpms[start : end + 1]
    duration_s = float((window_offsets_ms[-1] - window_offsets_ms[0]) / 1000.0)
    if duration_s < config.min_ramp_duration_s:
        return None

    fitted_bpms, linear_r2 = _linear_fit(window_offsets_ms / 1000.0, window_bpms)
    start_bpm = float(fitted_bpms[0])
    end_bpm = float(fitted_bpms[-1])
    bpm_delta = end_bpm - start_bpm
    bpm_span = abs(bpm_delta)
    if bpm_span < config.min_ramp_bpm_span:
        return None
    if linear_r2 < config.min_linear_r2:
        return None

    direction_sign = 1.0 if bpm_delta > 0.0 else -1.0
    deltas = np.diff(window_bpms)
    sign_consistency = (
        float(np.mean(deltas * direction_sign >= -1e-6)) if deltas.size else 0.0
    )
    if sign_consistency < config.min_sign_consistency:
        return None

    residuals = np.abs(window_bpms - fitted_bpms)
    p90_residual_bpm = float(np.percentile(residuals, 90.0)) if residuals.size else 0.0
    if p90_residual_bpm > config.max_p90_residual_bpm:
        return None

    output_indexes = _output_indexes(window_offsets_ms.size, config.max_output_points)
    output_offsets_ms = window_offsets_ms[output_indexes]
    output_bpms = fitted_bpms[output_indexes]
    if np.any(output_bpms <= 0.0) or not np.all(np.isfinite(output_bpms)):
        return None

    grid = FittedTimingGrid(
        tuple(
            TimingSegment(offset_ms=float(offset_ms), beat_length_ms=float(60000.0 / bpm))
            for offset_ms, bpm in zip(output_offsets_ms, output_bpms)
        )
    )
    ramp_detection = detect_timing_ramp(grid)
    mean_peak_prob = float(np.mean(probabilities[start : end + 2]))
    score = (
        ramp_detection.score
        + linear_r2
        + sign_consistency
        + bpm_span / 120.0
        + duration_s / 20.0
        + mean_peak_prob
        - p90_residual_bpm / 100.0
    )
    return RampBeatGridCandidate(
        start_ms=float(output_offsets_ms[0]),
        end_ms=float(output_offsets_ms[-1]),
        duration_s=duration_s,
        start_bpm=start_bpm,
        end_bpm=end_bpm,
        bpm_span=float(bpm_span),
        direction="up" if bpm_delta > 0.0 else "down",
        point_count=len(grid.segments),
        beat_count=int(end - start + 2),
        linear_r2=float(linear_r2),
        sign_consistency=sign_consistency,
        p90_residual_bpm=p90_residual_bpm,
        mean_peak_prob=mean_peak_prob,
        probability_score=mean_peak_prob,
        phase=None,
        hypothesis_kind="materialized_beats",
        score=float(score),
        grid=grid,
        ramp_detection=ramp_detection,
    )


def _fit_hint_ramp_candidates(
    prediction: FrameTimingPrediction,
    *,
    hint: RampBeatGridHint,
    config: BeatMaterializationConfig,
) -> list[RampBeatGridCandidate]:
    candidates: list[RampBeatGridCandidate] = []
    start_values = _search_values(
        hint.start_ms,
        radius=config.hint_time_radius_ms,
        step=config.hint_time_step_ms,
    )
    end_values = _search_values(
        hint.end_ms,
        radius=config.hint_time_radius_ms,
        step=config.hint_time_step_ms,
    )
    start_bpm_values = _search_values(
        hint.start_bpm,
        radius=config.hint_bpm_radius,
        step=config.hint_bpm_step,
    )
    end_bpm_values = _search_values(
        hint.end_bpm,
        radius=config.hint_bpm_radius,
        step=config.hint_bpm_step,
    )
    phases = np.arange(config.hint_phase_steps, dtype=np.float64) / float(config.hint_phase_steps)

    for start_ms in start_values:
        for end_ms in end_values:
            if end_ms <= start_ms:
                continue
            for start_bpm in start_bpm_values:
                for end_bpm in end_bpm_values:
                    if start_bpm <= 0.0 or end_bpm <= 0.0:
                        continue
                    for phase in phases:
                        candidate = _fit_hint_ramp_candidate(
                            prediction,
                            start_ms=float(start_ms),
                            end_ms=float(end_ms),
                            start_bpm=float(start_bpm),
                            end_bpm=float(end_bpm),
                            phase=float(phase),
                            config=config,
                        )
                        if candidate is not None:
                            candidates.append(candidate)
    return candidates


def _fit_hint_ramp_candidate(
    prediction: FrameTimingPrediction,
    *,
    start_ms: float,
    end_ms: float,
    start_bpm: float,
    end_bpm: float,
    phase: float,
    config: BeatMaterializationConfig,
) -> RampBeatGridCandidate | None:
    duration_s = (end_ms - start_ms) / 1000.0
    bpm_span = abs(end_bpm - start_bpm)
    if duration_s < config.min_hint_ramp_duration_s or bpm_span < config.min_ramp_bpm_span:
        return None

    beat_times_ms = _render_ramp_beat_times(
        start_ms=start_ms,
        end_ms=end_ms,
        start_bpm=start_bpm,
        end_bpm=end_bpm,
        phase=phase,
    )
    if len(beat_times_ms) < config.min_sequence_beats:
        return None

    probability_samples = np.asarray(
        [
            _local_probability_score(
                prediction.beat_prob,
                frame_rate_hz=prediction.frame_rate_hz,
                time_ms=time_ms,
                radius_ms=config.hint_probability_radius_ms,
                baseline_radius_ms=config.hint_baseline_radius_ms,
            )
            for time_ms in beat_times_ms
        ],
        dtype=np.float64,
    )
    probability_score = float(
        np.mean(probability_samples) + 0.15 * np.percentile(probability_samples, 25.0)
    )

    point_count = min(
        config.max_output_points,
        max(config.min_ramp_points, int(round(duration_s * 2.0))),
    )
    grid = _grid_from_linear_ramp(
        start_ms=start_ms,
        end_ms=end_ms,
        start_bpm=start_bpm,
        end_bpm=end_bpm,
        point_count=point_count,
    )
    ramp_detection = detect_timing_ramp(grid)
    score = (
        ramp_detection.score
        + probability_score
        + bpm_span / 120.0
        + duration_s / 20.0
    )
    return RampBeatGridCandidate(
        start_ms=start_ms,
        end_ms=end_ms,
        duration_s=duration_s,
        start_bpm=start_bpm,
        end_bpm=end_bpm,
        bpm_span=bpm_span,
        direction="up" if end_bpm > start_bpm else "down",
        point_count=len(grid.segments),
        beat_count=len(beat_times_ms),
        linear_r2=1.0,
        sign_consistency=1.0,
        p90_residual_bpm=0.0,
        mean_peak_prob=float(np.mean(probability_samples)),
        probability_score=probability_score,
        phase=phase,
        hypothesis_kind="trusted_hint",
        score=float(score),
        grid=grid,
        ramp_detection=ramp_detection,
    )


def _render_ramp_beat_times(
    *,
    start_ms: float,
    end_ms: float,
    start_bpm: float,
    end_bpm: float,
    phase: float,
) -> list[float]:
    start_s = start_ms / 1000.0
    duration_s = (end_ms - start_ms) / 1000.0
    start_beats_per_s = start_bpm / 60.0
    end_beats_per_s = end_bpm / 60.0
    total_beats = (start_beats_per_s + end_beats_per_s) * duration_s / 2.0
    beat_times_ms: list[float] = []
    beat_index = float(phase)
    while beat_index <= total_beats + 1e-9:
        elapsed_s = _solve_linear_ramp_elapsed_seconds(
            beat_index,
            duration_s=duration_s,
            start_beats_per_s=start_beats_per_s,
            end_beats_per_s=end_beats_per_s,
        )
        if 0.0 <= elapsed_s <= duration_s:
            beat_times_ms.append((start_s + elapsed_s) * 1000.0)
        beat_index += 1.0
    return beat_times_ms


def _solve_linear_ramp_elapsed_seconds(
    beat_index: float,
    *,
    duration_s: float,
    start_beats_per_s: float,
    end_beats_per_s: float,
) -> float:
    acceleration = (end_beats_per_s - start_beats_per_s) / duration_s
    if abs(acceleration) < 1e-12:
        return beat_index / start_beats_per_s
    discriminant = start_beats_per_s * start_beats_per_s + 2.0 * acceleration * beat_index
    if discriminant < 0.0:
        return float("nan")
    sqrt_discriminant = math.sqrt(discriminant)
    roots = (
        (-start_beats_per_s + sqrt_discriminant) / acceleration,
        (-start_beats_per_s - sqrt_discriminant) / acceleration,
    )
    valid_roots = [root for root in roots if 0.0 <= root <= duration_s]
    return valid_roots[0] if valid_roots else float("nan")


def _local_probability_score(
    beat_prob: NDArray[np.float32],
    *,
    frame_rate_hz: float,
    time_ms: float,
    radius_ms: float,
    baseline_radius_ms: float,
) -> float:
    frame = int(round(time_ms / 1000.0 * frame_rate_hz))
    radius = max(1, int(round(radius_ms / 1000.0 * frame_rate_hz)))
    baseline_radius = max(radius + 1, int(round(baseline_radius_ms / 1000.0 * frame_rate_hz)))
    start = max(0, frame - radius)
    end = min(beat_prob.size, frame + radius + 1)
    baseline_start = max(0, frame - baseline_radius)
    baseline_end = min(beat_prob.size, frame + baseline_radius + 1)
    if start >= end:
        return 0.0
    peak = float(np.max(beat_prob[start:end]))
    baseline = (
        float(np.median(beat_prob[baseline_start:baseline_end]))
        if baseline_start < baseline_end
        else 0.0
    )
    return peak - 0.35 * baseline


def _grid_from_linear_ramp(
    *,
    start_ms: float,
    end_ms: float,
    start_bpm: float,
    end_bpm: float,
    point_count: int,
) -> FittedTimingGrid:
    offsets_ms = np.linspace(start_ms, end_ms, point_count, dtype=np.float64)
    bpms = np.linspace(start_bpm, end_bpm, point_count, dtype=np.float64)
    return FittedTimingGrid(
        tuple(
            TimingSegment(offset_ms=float(offset_ms), beat_length_ms=float(60000.0 / bpm))
            for offset_ms, bpm in zip(offsets_ms, bpms)
        )
    )


def _search_values(center: float, *, radius: float, step: float) -> NDArray[np.float64]:
    if radius <= 0.0:
        return np.asarray([center], dtype=np.float64)
    offsets = np.arange(-radius, radius + step * 0.5, step, dtype=np.float64)
    return center + offsets


def _linear_fit(
    times_s: NDArray[np.float64],
    values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    centered_times_s = times_s - float(np.mean(times_s))
    slope, intercept = np.polyfit(centered_times_s, values, deg=1)
    fitted = slope * centered_times_s + intercept
    total = float(np.sum((values - np.mean(values)) ** 2))
    if total <= 1e-12:
        return fitted, 0.0
    residual = float(np.sum((values - fitted) ** 2))
    return fitted, float(max(0.0, 1.0 - residual / total))


def _median_filter(values: NDArray[np.float64], window_size: int) -> NDArray[np.float64]:
    if window_size <= 1:
        return values
    if window_size % 2 == 0:
        window_size += 1
    radius = window_size // 2
    smoothed = np.empty_like(values)
    for index in range(values.size):
        start = max(0, index - radius)
        end = min(values.size, index + radius + 1)
        smoothed[index] = np.median(values[start:end])
    return smoothed


def _output_indexes(size: int, max_output_points: int) -> NDArray[np.int64]:
    if size <= max_output_points:
        return np.arange(size, dtype=np.int64)
    indexes = np.linspace(0, size - 1, max_output_points, dtype=np.int64)
    return np.unique(indexes)


def _has_failure_reason(reasons: Sequence[str]) -> bool:
    return bool(reasons and reasons != ("materialized_beats",))
