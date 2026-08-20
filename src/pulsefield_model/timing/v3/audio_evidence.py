from __future__ import annotations

import math
import re
import weakref
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from pulsefield_model.timing.v3.analytic_curve import (
    ConstantTempoSection,
    LinearTimeRampSection,
    PhaseContinuousTimingCurve,
)


RAW_AUDIO_EVIDENCE_UNAVAILABLE_FEWER_THAN_16_COMMON_BEATS = (
    "fewer_than_16_common_retained_beats"
)
RAW_AUDIO_EVIDENCE_UNAVAILABLE_NO_COMPLETE_WINDOW = "no_complete_16_beat_window"
RAW_AUDIO_EVIDENCE_UNAVAILABLE_RETAINED_DOMAIN_RATIO_BELOW_90 = (
    "retained_domain_ratio_below_0_90"
)
_MINIMUM_CANDIDATE_RETAINED_DOMAIN_RATIO = 0.90
_RAW_AUDIO_EVIDENCE_UNAVAILABLE_PRIORITY = (
    RAW_AUDIO_EVIDENCE_UNAVAILABLE_RETAINED_DOMAIN_RATIO_BELOW_90,
    RAW_AUDIO_EVIDENCE_UNAVAILABLE_FEWER_THAN_16_COMMON_BEATS,
    RAW_AUDIO_EVIDENCE_UNAVAILABLE_NO_COMPLETE_WINDOW,
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class RawAudioEvidenceConfig:
    """Frozen Exp009 raw-audio evidence configuration.

    The scorer intentionally has no tunable values in Exp009.  A dataclass is
    retained so results can carry the complete feature/scoring contract.
    """

    sample_rate: int = 16_000
    hop_length: int = 160
    win_length: int = 400
    mel_bins: int = 80
    mel_bands: tuple[tuple[int, int], ...] = (
        (0, 10),
        (10, 25),
        (25, 45),
        (45, 80),
    )
    top_fraction_denominator: int = 4
    normalization_percentile: float = 95.0
    normalization_epsilon: float = 1e-6
    normalized_clip_max: float = 4.0
    event_window_seconds: float = 0.050
    half_beat_offset: float = 0.5
    half_beat_penalty: float = 0.5
    window_weight: float = 0.25
    window_beats: int = 16
    window_quantile: float = 0.10
    minimum_common_beats: int = 16
    maximum_candidates: int = 64


DEFAULT_RAW_AUDIO_EVIDENCE_CONFIG = RawAudioEvidenceConfig()


@runtime_checkable
class AnalyticTimingCandidate(Protocol):
    """The deliberately small candidate interface consumed by this module."""

    @property
    def start_beat(self) -> int: ...

    @property
    def end_beat(self) -> int: ...

    @property
    def fingerprint_sha256(self) -> str: ...

    def time_at_beat(self, beat: float) -> float: ...


@dataclass(frozen=True)
class RawAudioEvidence:
    """Candidate-independent normalized four-band positive spectral flux."""

    frame_center_seconds: NDArray[np.float64]
    band_flux: NDArray[np.float32]
    band_percentile95: tuple[float, float, float, float]
    input_frame_count: int
    valid_frame_count: int
    audio_duration_seconds: float
    config: RawAudioEvidenceConfig = DEFAULT_RAW_AUDIO_EVIDENCE_CONFIG

    def __post_init__(self) -> None:
        _require_frozen_config(self.config)
        centers = np.asarray(self.frame_center_seconds)
        flux = np.asarray(self.band_flux)
        if centers.dtype != np.float64 or centers.ndim != 1:
            raise ValueError("frame_center_seconds must be a 1-D float64 array")
        if flux.dtype != np.float32 or flux.ndim != 2 or flux.shape[1] != 4:
            raise ValueError("band_flux must have float32 shape [valid_frames, 4]")
        if centers.shape[0] != flux.shape[0]:
            raise ValueError("frame centers and band flux must have the same frame count")
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(flux)):
            raise ValueError("raw-audio evidence arrays must contain only finite values")
        if centers.size > 1 and not np.all(np.diff(centers) > 0.0):
            raise ValueError("frame centers must be strictly increasing")
        if not isinstance(self.input_frame_count, Integral) or isinstance(
            self.input_frame_count, bool
        ):
            raise TypeError("input_frame_count must be an integer")
        if not isinstance(self.valid_frame_count, Integral) or isinstance(
            self.valid_frame_count, bool
        ):
            raise TypeError("valid_frame_count must be an integer")
        if int(self.input_frame_count) < 0:
            raise ValueError("input_frame_count must be non-negative")
        if int(self.valid_frame_count) != centers.shape[0]:
            raise ValueError("valid_frame_count must equal the evidence array length")
        if int(self.valid_frame_count) > int(self.input_frame_count):
            raise ValueError("valid_frame_count cannot exceed input_frame_count")
        duration = _require_nonnegative_finite_real(
            self.audio_duration_seconds,
            "audio_duration_seconds",
        )
        if centers.size and (centers[0] < 0.0 or centers[-1] > duration):
            raise ValueError(
                "frame_center_seconds values must lie inside "
                "[0, audio_duration_seconds]"
            )
        if np.any(flux < 0.0) or np.any(flux > self.config.normalized_clip_max):
            raise ValueError("band_flux values must lie inside the frozen normalized range")
        percentiles = tuple(
            _require_nonnegative_finite_real(value, "band_percentile95")
            for value in self.band_percentile95
        )
        if len(percentiles) != 4:
            raise ValueError("band_percentile95 must contain exactly four values")

        canonical_centers = np.array(centers, dtype=np.float64, copy=True)
        canonical_flux = np.array(flux, dtype=np.float32, copy=True)
        canonical_centers.setflags(write=False)
        canonical_flux.setflags(write=False)
        object.__setattr__(self, "frame_center_seconds", canonical_centers)
        object.__setattr__(self, "band_flux", canonical_flux)
        object.__setattr__(self, "band_percentile95", percentiles)
        object.__setattr__(self, "input_frame_count", int(self.input_frame_count))
        object.__setattr__(self, "valid_frame_count", int(self.valid_frame_count))
        object.__setattr__(self, "audio_duration_seconds", duration)


@dataclass(frozen=True)
class CandidateRawAudioScore:
    candidate_index: int
    fingerprint_sha256: str
    raw_score: float | None
    mean_beat_support: float | None
    mean_half_beat_support: float | None
    window_contrast_p10: float | None
    retained_beat_count: int
    complete_window_count: int
    unavailable_reason: str | None
    candidate_domain_beat_count: int | None = None
    complete_window_start_beats: tuple[int, ...] = ()

    @property
    def beat_count(self) -> int:
        return self.retained_beat_count

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None

    @property
    def retained_domain_ratio(self) -> float | None:
        if self.candidate_domain_beat_count is None:
            return None
        if self.candidate_domain_beat_count <= 0:
            return 0.0
        return self.retained_beat_count / self.candidate_domain_beat_count


@dataclass(frozen=True)
class RawAudioEvidenceRanking:
    evidence: RawAudioEvidence
    candidate_scores: tuple[CandidateRawAudioScore, ...]
    ranked_scores: tuple[CandidateRawAudioScore, ...]
    common_beat_indices: tuple[int, ...]
    complete_window_start_beats: tuple[int, ...]
    unavailable_reason: str | None

    @property
    def results(self) -> tuple[CandidateRawAudioScore, ...]:
        return self.candidate_scores

    @property
    def ranked_results(self) -> tuple[CandidateRawAudioScore, ...]:
        return self.ranked_scores

    @property
    def selected_fingerprint_sha256(self) -> str | None:
        if not self.ranked_scores:
            return None
        return self.ranked_scores[0].fingerprint_sha256

    @property
    def selected_candidate_index(self) -> int | None:
        if not self.ranked_scores:
            return None
        return self.ranked_scores[0].candidate_index


@dataclass(frozen=True)
class _CandidateSchedule:
    candidate_index: int
    fingerprint_sha256: str
    beat_window_starts: NDArray[np.int64]
    beat_window_ends: NDArray[np.int64]
    half_window_starts: NDArray[np.int64]
    half_window_ends: NDArray[np.int64]


@dataclass(frozen=True)
class _IndependentCandidateScorePayload:
    fingerprint_sha256: str
    raw_score: float | None
    mean_beat_support: float | None
    mean_half_beat_support: float | None
    window_contrast_p10: float | None
    retained_beat_count: int
    complete_window_count: int
    unavailable_reason: str | None
    candidate_domain_beat_count: int
    complete_window_start_beats: tuple[int, ...]

    def to_score(self, *, candidate_index: int) -> CandidateRawAudioScore:
        return CandidateRawAudioScore(
            candidate_index=candidate_index,
            fingerprint_sha256=self.fingerprint_sha256,
            raw_score=self.raw_score,
            mean_beat_support=self.mean_beat_support,
            mean_half_beat_support=self.mean_half_beat_support,
            window_contrast_p10=self.window_contrast_p10,
            retained_beat_count=self.retained_beat_count,
            complete_window_count=self.complete_window_count,
            unavailable_reason=self.unavailable_reason,
            candidate_domain_beat_count=self.candidate_domain_beat_count,
            complete_window_start_beats=self.complete_window_start_beats,
        )


_INDEPENDENT_SCORE_CACHE_BY_EVIDENCE_ID: dict[
    int,
    tuple[
        weakref.ReferenceType[RawAudioEvidence],
        dict[str, _IndependentCandidateScorePayload],
    ],
] = {}


def extract_raw_audio_evidence(
    log_mel_10ms: object,
    *,
    audio_duration_seconds: float,
    config: RawAudioEvidenceConfig = DEFAULT_RAW_AUDIO_EVIDENCE_CONFIG,
) -> RawAudioEvidence:
    """Extract the frozen four-band Exp009 novelty representation.

    Frames backed by right-padding in ``compute_log_mel_10ms`` are excluded
    using the unpadded waveform duration before either differencing or track
    normalization.
    """

    _require_frozen_config(config)
    mel = _require_log_mel_float32(log_mel_10ms, config=config)
    duration = _require_nonnegative_finite_real(
        audio_duration_seconds,
        "audio_duration_seconds",
    )

    frame_indices = np.arange(mel.shape[0], dtype=np.int64)
    frame_end_samples = frame_indices * config.hop_length + config.win_length
    duration_samples = np.float64(duration) * np.float64(config.sample_rate)
    valid_count = int(np.count_nonzero(frame_end_samples <= duration_samples))
    valid_mel = mel[:valid_count]

    centers = (
        frame_indices[:valid_count].astype(np.float64) * np.float64(config.hop_length)
        + np.float64(config.win_length) / np.float64(2.0)
    ) / np.float64(config.sample_rate)

    differences = np.empty_like(valid_mel, dtype=np.float32)
    if valid_count:
        differences[0] = np.float32(0.0)
    if valid_count > 1:
        np.subtract(valid_mel[1:], valid_mel[:-1], out=differences[1:])
        np.maximum(differences[1:], np.float32(0.0), out=differences[1:])

    band_flux = np.empty((valid_count, len(config.mel_bands)), dtype=np.float32)
    percentiles: list[float] = []
    for band_index, (start, end) in enumerate(config.mel_bands):
        width = end - start
        top_count = math.ceil(width / config.top_fraction_denominator)
        band_differences = differences[:, start:end]
        if valid_count:
            # Stable descending order makes the otherwise score-neutral tie
            # behavior explicit and reproducible.
            top_indices = np.argsort(-band_differences, axis=1, kind="stable")[
                :, :top_count
            ]
            top_values = np.take_along_axis(band_differences, top_indices, axis=1)
            strength = np.mean(
                top_values.astype(np.float64),
                axis=1,
                dtype=np.float64,
            ).astype(np.float32)
            percentile = float(
                np.percentile(
                    strength,
                    config.normalization_percentile,
                    method="linear",
                )
            )
            if percentile == 0.0 and not np.any(strength):
                normalized = np.zeros_like(strength, dtype=np.float32)
            else:
                normalized = np.clip(
                    strength.astype(np.float64)
                    / (percentile + config.normalization_epsilon),
                    0.0,
                    config.normalized_clip_max,
                ).astype(np.float32)
            band_flux[:, band_index] = normalized
        else:
            percentile = 0.0
        percentiles.append(percentile)

    return RawAudioEvidence(
        frame_center_seconds=centers.astype(np.float64, copy=False),
        band_flux=band_flux,
        band_percentile95=tuple(percentiles),  # type: ignore[arg-type]
        input_frame_count=mel.shape[0],
        valid_frame_count=valid_count,
        audio_duration_seconds=duration,
        config=config,
    )


def score_raw_audio_evidence(
    evidence: RawAudioEvidence,
    candidates: tuple[AnalyticTimingCandidate, ...],
    *,
    config: RawAudioEvidenceConfig = DEFAULT_RAW_AUDIO_EVIDENCE_CONFIG,
) -> RawAudioEvidenceRanking:
    """Score candidates with vectorized per-event band reductions."""

    return _score_raw_audio_evidence(
        evidence,
        candidates,
        config=config,
        scalar_reference=False,
    )


def score_raw_audio_evidence_independent(
    evidence: RawAudioEvidence,
    candidates: tuple[AnalyticTimingCandidate, ...],
    *,
    config: RawAudioEvidenceConfig = DEFAULT_RAW_AUDIO_EVIDENCE_CONFIG,
) -> RawAudioEvidenceRanking:
    """Score each candidate over its own physical beat coverage.

    This is the Exp013 raw self-score path.  It deliberately avoids intersecting
    beat indices across unrelated tempo aliases, so a half-time or double-time
    candidate cannot truncate the scoring domain for a direct-tempo candidate.
    """

    _require_frozen_config(config)
    if not isinstance(evidence, RawAudioEvidence):
        raise TypeError("evidence must be RawAudioEvidence")
    if evidence.config != config:
        raise ValueError("evidence and scorer configs must match")

    candidate_tuple, domains, fingerprints = _validate_candidates_independent(
        candidates,
        config=config,
    )
    return _score_raw_audio_evidence_independent_validated(
        evidence,
        candidate_tuple,
        domains,
        fingerprints,
        config=config,
        duplicate_fingerprint_rank_tiebreaker=False,
    )


def score_raw_audio_evidence_independent_with_duplicate_fingerprints(
    evidence: RawAudioEvidence,
    candidates: tuple[AnalyticTimingCandidate, ...],
    *,
    config: RawAudioEvidenceConfig = DEFAULT_RAW_AUDIO_EVIDENCE_CONFIG,
) -> RawAudioEvidenceRanking:
    """Batch-score independent candidates while preserving duplicate rows.

    This keeps the ordinary independent scorer's math and availability rules,
    but it allows canonical duplicate curves and rank-ties them by candidate
    index after score and fingerprint.  It is intended for caller-side batches
    that previously had to score one duplicate-fingerprint candidate at a time.
    """

    _require_frozen_config(config)
    if not isinstance(evidence, RawAudioEvidence):
        raise TypeError("evidence must be RawAudioEvidence")
    if evidence.config != config:
        raise ValueError("evidence and scorer configs must match")

    candidate_tuple, domains, fingerprints = _validate_candidates_independent(
        candidates,
        config=config,
        allow_duplicate_fingerprints=True,
    )
    return _score_raw_audio_evidence_independent_validated(
        evidence,
        candidate_tuple,
        domains,
        fingerprints,
        config=config,
        duplicate_fingerprint_rank_tiebreaker=True,
    )


def _score_raw_audio_evidence_independent_validated(
    evidence: RawAudioEvidence,
    candidate_tuple: tuple[AnalyticTimingCandidate, ...],
    domains: tuple[tuple[int, int], ...],
    fingerprints: tuple[str, ...],
    *,
    config: RawAudioEvidenceConfig,
    duplicate_fingerprint_rank_tiebreaker: bool,
) -> RawAudioEvidenceRanking:
    scores = [
        _score_independent_candidate_cached(
            evidence,
            candidate,
            candidate_index=index,
            start_beat=domains[index][0],
            end_beat=domains[index][1],
            fingerprint_sha256=fingerprints[index],
            config=config,
        )
        for index, candidate in enumerate(candidate_tuple)
    ]

    ranked = tuple(
        sorted(
            (score for score in scores if score.available),
            key=(
                (
                    lambda score: (
                        -_available_score(score),
                        score.fingerprint_sha256,
                        score.candidate_index,
                    )
                )
                if duplicate_fingerprint_rank_tiebreaker
                else (lambda score: (-_available_score(score), score.fingerprint_sha256))
            ),
        )
    )
    return RawAudioEvidenceRanking(
        evidence=evidence,
        candidate_scores=tuple(scores),
        ranked_scores=ranked,
        common_beat_indices=(),
        complete_window_start_beats=(),
        unavailable_reason=None if ranked else _highest_priority_unavailable_reason(scores),
    )


def score_raw_audio_evidence_scalar_reference(
    evidence: RawAudioEvidence,
    candidates: tuple[AnalyticTimingCandidate, ...],
    *,
    config: RawAudioEvidenceConfig = DEFAULT_RAW_AUDIO_EVIDENCE_CONFIG,
) -> RawAudioEvidenceRanking:
    """Slow independent reducer used to verify vectorized score equivalence."""

    return _score_raw_audio_evidence(
        evidence,
        candidates,
        config=config,
        scalar_reference=True,
    )


def score_timing_candidates(
    log_mel_10ms: object,
    candidates: tuple[AnalyticTimingCandidate, ...],
    *,
    audio_duration_seconds: float,
    config: RawAudioEvidenceConfig = DEFAULT_RAW_AUDIO_EVIDENCE_CONFIG,
) -> RawAudioEvidenceRanking:
    """Extract candidate-independent evidence, then score without a prior."""

    evidence = extract_raw_audio_evidence(
        log_mel_10ms,
        audio_duration_seconds=audio_duration_seconds,
        config=config,
    )
    return score_raw_audio_evidence(evidence, candidates, config=config)


# Concise public synonym retained for callers that treat extraction and
# candidate scoring as two distinct stages.
extract_audio_evidence = extract_raw_audio_evidence


def _score_raw_audio_evidence(
    evidence: RawAudioEvidence,
    candidates: tuple[AnalyticTimingCandidate, ...],
    *,
    config: RawAudioEvidenceConfig,
    scalar_reference: bool,
) -> RawAudioEvidenceRanking:
    _require_frozen_config(config)
    if not isinstance(evidence, RawAudioEvidence):
        raise TypeError("evidence must be RawAudioEvidence")
    if evidence.config != config:
        raise ValueError("evidence and scorer configs must match")

    candidate_tuple, start_beat, end_beat, fingerprints = _validate_candidates(
        candidates,
        config=config,
    )
    beat_indices = np.arange(start_beat, end_beat, dtype=np.int64)
    schedules = tuple(
        _build_candidate_schedule(
            candidate,
            candidate_index=index,
            fingerprint_sha256=fingerprints[index],
            beat_indices=beat_indices,
            frame_centers=evidence.frame_center_seconds,
            config=config,
        )
        for index, candidate in enumerate(candidate_tuple)
    )

    common_mask = np.ones(beat_indices.shape[0], dtype=np.bool_)
    for schedule in schedules:
        common_mask &= schedule.beat_window_ends > schedule.beat_window_starts
        common_mask &= schedule.half_window_ends > schedule.half_window_starts
    common_beats_array = beat_indices[common_mask]
    common_positions = np.flatnonzero(common_mask).astype(np.int64, copy=False)
    common_beats = tuple(int(value) for value in common_beats_array)
    complete_windows = _complete_window_positions(
        common_beats_array,
        window_beats=config.window_beats,
    )
    complete_window_starts = tuple(
        int(common_beats_array[positions[0]]) for positions in complete_windows
    )

    unavailable_reason: str | None = None
    if common_beats_array.size < config.minimum_common_beats:
        unavailable_reason = RAW_AUDIO_EVIDENCE_UNAVAILABLE_FEWER_THAN_16_COMMON_BEATS
    elif not complete_windows:
        unavailable_reason = RAW_AUDIO_EVIDENCE_UNAVAILABLE_NO_COMPLETE_WINDOW

    if unavailable_reason is not None:
        unavailable_scores = tuple(
            CandidateRawAudioScore(
                candidate_index=schedule.candidate_index,
                fingerprint_sha256=schedule.fingerprint_sha256,
                raw_score=None,
                mean_beat_support=None,
                mean_half_beat_support=None,
                window_contrast_p10=None,
                retained_beat_count=len(common_beats),
                complete_window_count=len(complete_windows),
                unavailable_reason=unavailable_reason,
                candidate_domain_beat_count=end_beat - start_beat,
                complete_window_start_beats=complete_window_starts,
            )
            for schedule in schedules
        )
        return RawAudioEvidenceRanking(
            evidence=evidence,
            candidate_scores=unavailable_scores,
            ranked_scores=(),
            common_beat_indices=common_beats,
            complete_window_start_beats=complete_window_starts,
            unavailable_reason=unavailable_reason,
        )

    scores: list[CandidateRawAudioScore] = []
    for schedule in schedules:
        mean_beat, mean_half, window_p10, raw_score = _reduce_schedule_score(
            evidence,
            schedule,
            common_positions,
            complete_windows,
            config=config,
            scalar_reference=scalar_reference,
        )
        scores.append(
            CandidateRawAudioScore(
                candidate_index=schedule.candidate_index,
                fingerprint_sha256=schedule.fingerprint_sha256,
                raw_score=raw_score,
                mean_beat_support=mean_beat,
                mean_half_beat_support=mean_half,
                window_contrast_p10=window_p10,
                retained_beat_count=len(common_beats),
                complete_window_count=len(complete_windows),
                unavailable_reason=None,
                candidate_domain_beat_count=end_beat - start_beat,
                complete_window_start_beats=complete_window_starts,
            )
        )

    ranked = tuple(
        sorted(
            scores,
            key=lambda score: (-_available_score(score), score.fingerprint_sha256),
        )
    )
    return RawAudioEvidenceRanking(
        evidence=evidence,
        candidate_scores=tuple(scores),
        ranked_scores=ranked,
        common_beat_indices=common_beats,
        complete_window_start_beats=complete_window_starts,
        unavailable_reason=None,
    )


def _reduce_schedule_score(
    evidence: RawAudioEvidence,
    schedule: _CandidateSchedule,
    retained_positions: NDArray[np.int64],
    complete_windows: tuple[NDArray[np.int64], ...],
    *,
    config: RawAudioEvidenceConfig,
    scalar_reference: bool,
) -> tuple[float, float, float, float]:
    if scalar_reference:
        beat_support = _event_supports_scalar(
            evidence.band_flux,
            schedule.beat_window_starts[retained_positions],
            schedule.beat_window_ends[retained_positions],
        )
        half_support = _event_supports_scalar(
            evidence.band_flux,
            schedule.half_window_starts[retained_positions],
            schedule.half_window_ends[retained_positions],
        )
    else:
        beat_support = _event_supports_vectorized(
            evidence.band_flux,
            schedule.beat_window_starts[retained_positions],
            schedule.beat_window_ends[retained_positions],
        )
        half_support = _event_supports_vectorized(
            evidence.band_flux,
            schedule.half_window_starts[retained_positions],
            schedule.half_window_ends[retained_positions],
        )

    beat_support64 = beat_support.astype(np.float64, copy=False)
    half_support64 = half_support.astype(np.float64, copy=False)
    contrasts = beat_support64 - half_support64
    window_contrasts = np.asarray(
        [
            np.mean(contrasts[positions], dtype=np.float64)
            for positions in complete_windows
        ],
        dtype=np.float64,
    )
    mean_beat = float(np.mean(beat_support64, dtype=np.float64))
    mean_half = float(np.mean(half_support64, dtype=np.float64))
    window_p10 = float(
        np.quantile(window_contrasts, config.window_quantile, method="linear")
    )
    raw_score = float(
        mean_beat
        - config.half_beat_penalty * mean_half
        + config.window_weight * window_p10
    )
    if not all(
        math.isfinite(value) for value in (mean_beat, mean_half, window_p10, raw_score)
    ):
        raise ValueError("raw-audio score reduction produced a nonfinite value")
    return mean_beat, mean_half, window_p10, raw_score


def _build_candidate_schedule(
    candidate: AnalyticTimingCandidate,
    *,
    candidate_index: int,
    fingerprint_sha256: str,
    beat_indices: NDArray[np.int64],
    frame_centers: NDArray[np.float64],
    config: RawAudioEvidenceConfig,
) -> _CandidateSchedule:
    event_beats = np.empty(beat_indices.size * 2, dtype=np.float64)
    event_beats[0::2] = beat_indices.astype(np.float64, copy=False)
    event_beats[1::2] = event_beats[0::2] + np.float64(config.half_beat_offset)
    ordered_times = _candidate_times_seconds(candidate, event_beats)
    beat_times = ordered_times[0::2]
    half_times = ordered_times[1::2]
    ordered_times = np.empty(beat_times.size * 2, dtype=np.float64)
    ordered_times[0::2] = beat_times
    ordered_times[1::2] = half_times
    if ordered_times.size > 1 and not np.all(np.diff(ordered_times) > 0.0):
        raise ValueError(
            f"candidate {candidate_index} beat/half-beat times must be strictly increasing"
        )

    beat_starts, beat_ends = _inclusive_event_windows(
        frame_centers,
        beat_times,
        radius_seconds=config.event_window_seconds,
    )
    half_starts, half_ends = _inclusive_event_windows(
        frame_centers,
        half_times,
        radius_seconds=config.event_window_seconds,
    )
    return _CandidateSchedule(
        candidate_index=candidate_index,
        fingerprint_sha256=fingerprint_sha256,
        beat_window_starts=beat_starts,
        beat_window_ends=beat_ends,
        half_window_starts=half_starts,
        half_window_ends=half_ends,
    )


def _candidate_times_seconds(
    candidate: AnalyticTimingCandidate,
    beats: NDArray[np.float64],
) -> NDArray[np.float64]:
    if isinstance(candidate, PhaseContinuousTimingCurve):
        return _phase_continuous_curve_times_seconds(candidate, beats)
    return np.asarray(
        [_candidate_time_seconds(candidate, float(beat)) for beat in beats],
        dtype=np.float64,
    )


def _phase_continuous_curve_times_seconds(
    candidate: PhaseContinuousTimingCurve,
    beats: NDArray[np.float64],
) -> NDArray[np.float64]:
    times_ms = np.empty(beats.shape[0], dtype=np.float64)
    for section_index, section in enumerate(candidate.sections):
        if section_index == len(candidate.sections) - 1:
            mask = (beats >= section.start_beat) & (beats <= section.end_beat)
        else:
            mask = (beats >= section.start_beat) & (beats < section.end_beat)
        if not np.any(mask):
            continue
        section_beats = beats[mask]
        boundary_time_ms = candidate.boundary_times_ms[section_index]
        if isinstance(section, ConstantTempoSection):
            times_ms[mask] = np.asarray(
                [
                    boundary_time_ms
                    + 1000.0
                    * (60.0 * (float(beat) - section.start_beat) / section.bpm)
                    for beat in section_beats
                ],
                dtype=np.float64,
            )
        elif isinstance(section, LinearTimeRampSection):
            times_ms[mask] = np.asarray(
                [
                    boundary_time_ms
                    if float(beat) == section.start_beat
                    else boundary_time_ms
                    + 1000.0 * section.elapsed_seconds_at_beat(float(beat))
                    for beat in section_beats
                ],
                dtype=np.float64,
            )
        else:
            raise TypeError("unsupported timing curve section")
    return times_ms / 1000.0


def _score_independent_candidate_cached(
    evidence: RawAudioEvidence,
    candidate: AnalyticTimingCandidate,
    *,
    candidate_index: int,
    start_beat: int,
    end_beat: int,
    fingerprint_sha256: str,
    config: RawAudioEvidenceConfig,
) -> CandidateRawAudioScore:
    cache = _independent_score_cache(evidence)
    cached = cache.get(fingerprint_sha256)
    if cached is None:
        cached = _score_independent_candidate_payload(
            evidence,
            candidate,
            start_beat=start_beat,
            end_beat=end_beat,
            fingerprint_sha256=fingerprint_sha256,
            config=config,
        )
        cache[fingerprint_sha256] = cached
    return cached.to_score(candidate_index=candidate_index)


def _score_independent_candidate_payload(
    evidence: RawAudioEvidence,
    candidate: AnalyticTimingCandidate,
    *,
    start_beat: int,
    end_beat: int,
    fingerprint_sha256: str,
    config: RawAudioEvidenceConfig,
) -> _IndependentCandidateScorePayload:
    beat_indices = np.arange(start_beat, end_beat, dtype=np.int64)
    schedule = _build_candidate_schedule(
        candidate,
        candidate_index=0,
        fingerprint_sha256=fingerprint_sha256,
        beat_indices=beat_indices,
        frame_centers=evidence.frame_center_seconds,
        config=config,
    )

    retained_mask = schedule.beat_window_ends > schedule.beat_window_starts
    retained_mask &= schedule.half_window_ends > schedule.half_window_starts
    retained_beats_array = beat_indices[retained_mask]
    retained_positions = np.flatnonzero(retained_mask).astype(np.int64, copy=False)
    complete_windows = _complete_window_positions(
        retained_beats_array,
        window_beats=config.window_beats,
    )
    complete_window_starts = tuple(
        int(retained_beats_array[positions[0]]) for positions in complete_windows
    )
    candidate_domain_beat_count = end_beat - start_beat
    retained_domain_ratio = retained_beats_array.size / candidate_domain_beat_count

    unavailable_reason: str | None = None
    if retained_beats_array.size < config.minimum_common_beats:
        unavailable_reason = RAW_AUDIO_EVIDENCE_UNAVAILABLE_FEWER_THAN_16_COMMON_BEATS
    elif retained_domain_ratio < _MINIMUM_CANDIDATE_RETAINED_DOMAIN_RATIO:
        unavailable_reason = RAW_AUDIO_EVIDENCE_UNAVAILABLE_RETAINED_DOMAIN_RATIO_BELOW_90
    elif not complete_windows:
        unavailable_reason = RAW_AUDIO_EVIDENCE_UNAVAILABLE_NO_COMPLETE_WINDOW

    if unavailable_reason is not None:
        return _IndependentCandidateScorePayload(
            fingerprint_sha256=fingerprint_sha256,
            raw_score=None,
            mean_beat_support=None,
            mean_half_beat_support=None,
            window_contrast_p10=None,
            retained_beat_count=int(retained_beats_array.size),
            complete_window_count=len(complete_windows),
            unavailable_reason=unavailable_reason,
            candidate_domain_beat_count=candidate_domain_beat_count,
            complete_window_start_beats=complete_window_starts,
        )

    mean_beat, mean_half, window_p10, raw_score = _reduce_schedule_score(
        evidence,
        schedule,
        retained_positions,
        complete_windows,
        config=config,
        scalar_reference=False,
    )
    return _IndependentCandidateScorePayload(
        fingerprint_sha256=fingerprint_sha256,
        raw_score=raw_score,
        mean_beat_support=mean_beat,
        mean_half_beat_support=mean_half,
        window_contrast_p10=window_p10,
        retained_beat_count=int(retained_beats_array.size),
        complete_window_count=len(complete_windows),
        unavailable_reason=None,
        candidate_domain_beat_count=candidate_domain_beat_count,
        complete_window_start_beats=complete_window_starts,
    )


def _independent_score_cache(
    evidence: RawAudioEvidence,
) -> dict[str, _IndependentCandidateScorePayload]:
    evidence_id = id(evidence)
    entry = _INDEPENDENT_SCORE_CACHE_BY_EVIDENCE_ID.get(evidence_id)
    if entry is not None:
        evidence_ref, cache = entry
        if evidence_ref() is evidence:
            return cache

    def cleanup(_ref: weakref.ReferenceType[RawAudioEvidence]) -> None:
        current = _INDEPENDENT_SCORE_CACHE_BY_EVIDENCE_ID.get(evidence_id)
        if current is not None and current[0] is _ref:
            del _INDEPENDENT_SCORE_CACHE_BY_EVIDENCE_ID[evidence_id]

    cache: dict[str, _IndependentCandidateScorePayload] = {}
    _INDEPENDENT_SCORE_CACHE_BY_EVIDENCE_ID[evidence_id] = (
        weakref.ref(evidence, cleanup),
        cache,
    )
    return cache


def _inclusive_event_windows(
    frame_centers: NDArray[np.float64],
    event_times: NDArray[np.float64],
    *,
    radius_seconds: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return exact half-open frame slices for literal inclusive abs windows."""

    if event_times.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()

    lower = np.nextafter(event_times - radius_seconds, -np.inf)
    upper = np.nextafter(event_times + radius_seconds, np.inf)
    starts = np.searchsorted(frame_centers, lower, side="left").astype(
        np.int64,
        copy=False,
    )
    ends = np.searchsorted(frame_centers, upper, side="right").astype(
        np.int64,
        copy=False,
    )

    rough_starts = starts.copy()
    nonempty = starts < ends
    if np.any(nonempty):
        nonempty_indices = np.flatnonzero(nonempty)
        start_centers = frame_centers[starts[nonempty_indices]]
        trim_start = (
            np.abs(start_centers - event_times[nonempty_indices]) > radius_seconds
        )
        if np.any(trim_start):
            starts[nonempty_indices[trim_start]] += 1

        end_indices = np.flatnonzero(starts < ends)
        if end_indices.size:
            end_centers = frame_centers[ends[end_indices] - 1]
            trim_end = np.abs(end_centers - event_times[end_indices]) > radius_seconds
            if np.any(trim_end):
                ends[end_indices[trim_end]] -= 1

    no_included = starts >= ends
    if np.any(no_included):
        starts[no_included] = rough_starts[no_included]
        ends[no_included] = rough_starts[no_included]
    return starts, ends


def _event_supports_vectorized(
    band_flux: NDArray[np.float32],
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
) -> NDArray[np.float64]:
    lengths = ends - starts
    if starts.size == 0 or np.any(lengths <= 0):
        raise ValueError("event support windows must be non-empty")
    width = int(np.max(lengths))
    offsets = np.arange(width, dtype=np.int64)
    indices = starts[:, None] + offsets[None, :]
    valid = indices < ends[:, None]
    safe_indices = np.minimum(indices, band_flux.shape[0] - 1)
    gathered = band_flux[safe_indices]
    maxima = np.max(
        np.where(valid[:, :, None], gathered, np.float32(-np.inf)),
        axis=1,
    )
    ordered = np.sort(maxima, axis=1, kind="stable")
    return np.mean(
        ordered[:, -2:].astype(np.float64),
        axis=1,
        dtype=np.float64,
    )


def _event_supports_scalar(
    band_flux: NDArray[np.float32],
    starts: NDArray[np.int64],
    ends: NDArray[np.int64],
) -> NDArray[np.float64]:
    support = np.empty(starts.shape[0], dtype=np.float64)
    for event_index, (start, end) in enumerate(zip(starts, ends)):
        maxima = [0.0, 0.0, 0.0, 0.0]
        for frame_index in range(int(start), int(end)):
            for band_index in range(4):
                value = float(band_flux[frame_index, band_index])
                if value > maxima[band_index]:
                    maxima[band_index] = value
        maxima.sort()
        support[event_index] = (maxima[2] + maxima[3]) / 2.0
    return support


def _complete_window_positions(
    common_beats: NDArray[np.int64],
    *,
    window_beats: int,
) -> tuple[NDArray[np.int64], ...]:
    if common_beats.size == 0:
        return ()
    first_beat = int(common_beats[0])
    terminal_beat = int(common_beats[-1]) + 1
    window_starts = np.arange(first_beat, terminal_beat, window_beats, dtype=np.int64)
    if window_starts.size == 0:
        return ()

    start_positions = np.searchsorted(common_beats, window_starts, side="left").astype(
        np.int64,
        copy=False,
    )
    in_range = start_positions < common_beats.size
    if not np.any(in_range):
        return ()

    candidate_starts = window_starts[in_range]
    candidate_positions = start_positions[in_range]
    enough_values = candidate_positions + window_beats <= common_beats.size
    if not np.any(enough_values):
        return ()

    candidate_starts = candidate_starts[enough_values]
    candidate_positions = candidate_positions[enough_values]
    window_complete = (
        common_beats[candidate_positions] == candidate_starts
    ) & (
        common_beats[candidate_positions + window_beats - 1]
        == candidate_starts + window_beats - 1
    )
    if not np.any(window_complete):
        return ()

    base_positions = candidate_positions[window_complete]
    offsets = np.arange(window_beats, dtype=np.int64)
    return tuple(base + offsets for base in base_positions)


def _validate_candidates(
    candidates: object,
    *,
    config: RawAudioEvidenceConfig,
) -> tuple[tuple[AnalyticTimingCandidate, ...], int, int, tuple[str, ...]]:
    if not isinstance(candidates, tuple):
        raise TypeError("candidates must be an ordered tuple")
    if not candidates:
        raise ValueError("candidates must be non-empty")
    if len(candidates) > config.maximum_candidates:
        raise ValueError(
            f"candidates cannot contain more than {config.maximum_candidates} curves"
        )

    start_beat: int | None = None
    end_beat: int | None = None
    fingerprints: list[str] = []
    for index, candidate in enumerate(candidates):
        candidate_start = _require_integer_beat(
            getattr(candidate, "start_beat", None),
            f"candidate {index} start_beat",
        )
        candidate_end = _require_integer_beat(
            getattr(candidate, "end_beat", None),
            f"candidate {index} end_beat",
        )
        if candidate_end <= candidate_start:
            raise ValueError(f"candidate {index} end_beat must exceed start_beat")
        if not callable(getattr(candidate, "time_at_beat", None)):
            raise TypeError(f"candidate {index} must provide time_at_beat(beat)")
        fingerprint = _candidate_fingerprint(candidate, candidate_index=index)
        if start_beat is None:
            start_beat = candidate_start
            end_beat = candidate_end
        elif candidate_start != start_beat or candidate_end != end_beat:
            raise ValueError("all candidates must share the same integer beat domain")
        fingerprints.append(fingerprint)

    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("candidate fingerprints must be unique")
    assert start_beat is not None and end_beat is not None
    return candidates, start_beat, end_beat, tuple(fingerprints)


def _validate_candidates_independent(
    candidates: object,
    *,
    config: RawAudioEvidenceConfig,
    allow_duplicate_fingerprints: bool = False,
) -> tuple[
    tuple[AnalyticTimingCandidate, ...],
    tuple[tuple[int, int], ...],
    tuple[str, ...],
]:
    if not isinstance(candidates, tuple):
        raise TypeError("candidates must be an ordered tuple")
    if not candidates:
        raise ValueError("candidates must be non-empty")
    if len(candidates) > config.maximum_candidates:
        raise ValueError(
            f"candidates cannot contain more than {config.maximum_candidates} curves"
        )

    domains: list[tuple[int, int]] = []
    fingerprints: list[str] = []
    for index, candidate in enumerate(candidates):
        candidate_start = _require_integer_beat(
            getattr(candidate, "start_beat", None),
            f"candidate {index} start_beat",
        )
        candidate_end = _require_integer_beat(
            getattr(candidate, "end_beat", None),
            f"candidate {index} end_beat",
        )
        if candidate_end <= candidate_start:
            raise ValueError(f"candidate {index} end_beat must exceed start_beat")
        if not callable(getattr(candidate, "time_at_beat", None)):
            raise TypeError(f"candidate {index} must provide time_at_beat(beat)")
        domains.append((candidate_start, candidate_end))
        fingerprints.append(_candidate_fingerprint(candidate, candidate_index=index))

    if not allow_duplicate_fingerprints and len(set(fingerprints)) != len(fingerprints):
        raise ValueError("candidate fingerprints must be unique")
    return candidates, tuple(domains), tuple(fingerprints)


def _candidate_fingerprint(candidate: object, *, candidate_index: int) -> str:
    fingerprint = getattr(candidate, "fingerprint_sha256", None)
    if callable(fingerprint):
        fingerprint = fingerprint()
    if not isinstance(fingerprint, str) or _SHA256_RE.fullmatch(fingerprint) is None:
        raise ValueError(
            f"candidate {candidate_index} fingerprint_sha256 must be 64 lowercase hex characters"
        )
    return fingerprint


def _candidate_time_seconds(candidate: AnalyticTimingCandidate, beat: float) -> float:
    try:
        time_ms = candidate.time_at_beat(beat)
    except Exception as error:
        raise ValueError(f"candidate time_at_beat({beat!r}) failed") from error
    time_value_ms = _require_finite_real(time_ms, "candidate time_at_beat result")
    return time_value_ms / 1000.0


def _require_log_mel_float32(
    value: object,
    *,
    config: RawAudioEvidenceConfig,
) -> NDArray[np.float32]:
    if not isinstance(value, np.ndarray):
        raise TypeError("log_mel_10ms must be a NumPy array")
    if value.dtype != np.float32:
        raise TypeError(f"log_mel_10ms must have dtype float32, got {value.dtype}")
    if value.ndim != 2 or value.shape[1] != config.mel_bins:
        raise ValueError(
            f"log_mel_10ms must have shape [frames, {config.mel_bins}], got {value.shape}"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError("log_mel_10ms must contain only finite values")
    return value


def _require_frozen_config(config: RawAudioEvidenceConfig) -> None:
    if not isinstance(config, RawAudioEvidenceConfig):
        raise TypeError("config must be RawAudioEvidenceConfig")
    if config != DEFAULT_RAW_AUDIO_EVIDENCE_CONFIG:
        raise ValueError("the Exp009 raw-audio evidence config is frozen")


def _require_integer_beat(value: object, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _require_nonnegative_finite_real(value: object, name: str) -> float:
    result = _require_finite_real(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _require_finite_real(value: object, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _available_score(score: CandidateRawAudioScore) -> float:
    if score.raw_score is None:
        raise ValueError("cannot rank an unavailable raw-audio score")
    return score.raw_score


def _highest_priority_unavailable_reason(
    scores: list[CandidateRawAudioScore] | tuple[CandidateRawAudioScore, ...],
) -> str | None:
    reasons = {score.unavailable_reason for score in scores}
    for reason in _RAW_AUDIO_EVIDENCE_UNAVAILABLE_PRIORITY:
        if reason in reasons:
            return reason
    return next((reason for reason in reasons if reason is not None), None)
