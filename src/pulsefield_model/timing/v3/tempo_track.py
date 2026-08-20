from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from pulsefield_model.timing.schema import FrameTimingPrediction
from pulsefield_model.timing.v3 import global_constant_jump as _global
from pulsefield_model.timing.v3.analytic_curve import (
    ConstantTempoSection,
    LinearTimeRampSection,
    PhaseContinuousTimingCurve,
)
from pulsefield_model.timing.v3.audio_evidence import (
    CandidateRawAudioScore,
    RawAudioEvidence,
    RawAudioEvidenceRanking,
    score_raw_audio_evidence,
    score_raw_audio_evidence_independent,
)


TEMPO_TRACK_VERSION = "pulsefield_model.timing_v3_tempo_track_exp021_v1"
TEMPO_TRACK_RESULT_DUMP_SCHEMA_VERSION = (
    "pulsefield_model.timing_v3_tempo_track_result_dump_exp021_v1"
)


_PHASE_ORIGIN_REQUEST_CACHE: ContextVar[
    dict[tuple[object, ...], float] | None
] = ContextVar("timing_v3_phase_origin_request_cache", default=None)

_EVENT_SUPPORT_REQUEST_CACHE: ContextVar[
    dict[tuple[object, ...], NDArray[np.float64]] | None
] = ContextVar("timing_v3_event_support_request_cache", default=None)


@dataclass(frozen=True)
class TempoTrackConfig:
    """Bounded candidate-generation settings for the Timing-v3 fast path."""

    minimum_bpm: float = 80.0
    maximum_bpm: float = 240.0
    local_window_seconds: float = 6.0
    local_hop_seconds: float = 1.0
    minimum_local_strength: float = 0.04
    beat_event_radius_ms: float = 40.0
    minimum_excursion_seconds: float = 2.0
    maximum_excursion_seconds: float = 60.0
    minimum_jump_bpm: float = 5.0
    maximum_boundary_seeds: int = 96
    maximum_pair_seeds: int = 256
    maximum_base_hypotheses: int = 12
    maximum_jump_candidates: int = 44
    maximum_ramp_candidates: int = 8
    maximum_candidates: int = 64
    raw_run_maximum_retained_runs: int = 4
    raw_run_rational_limit: int = 5
    raw_run_primary_snap_fraction: float = 0.0125
    raw_run_minimum_deviation_bpm: float = 8.0
    raw_run_minimum_deviation_fraction: float = 0.05
    raw_run_minimum_observations: int = 3
    raw_run_gap_hop_multiplier: float = 1.5
    raw_run_expansion_ms: float = 1000.0
    raw_run_jump_minimum_overlap_ms: float = 500.0
    raw_run_jump_max_tempo_ratio: float = 1.15
    raw_self_minimum_retained_domain_ratio: float = 0.90
    raw_self_minimum_retained_beats: int = 16

    def __post_init__(self) -> None:
        finite_positive = (
            ("minimum_bpm", self.minimum_bpm),
            ("maximum_bpm", self.maximum_bpm),
            ("local_window_seconds", self.local_window_seconds),
            ("local_hop_seconds", self.local_hop_seconds),
            ("beat_event_radius_ms", self.beat_event_radius_ms),
            ("minimum_excursion_seconds", self.minimum_excursion_seconds),
            ("maximum_excursion_seconds", self.maximum_excursion_seconds),
            ("minimum_jump_bpm", self.minimum_jump_bpm),
            ("raw_run_primary_snap_fraction", self.raw_run_primary_snap_fraction),
            ("raw_run_minimum_deviation_bpm", self.raw_run_minimum_deviation_bpm),
            (
                "raw_run_minimum_deviation_fraction",
                self.raw_run_minimum_deviation_fraction,
            ),
            ("raw_run_gap_hop_multiplier", self.raw_run_gap_hop_multiplier),
            ("raw_run_expansion_ms", self.raw_run_expansion_ms),
            (
                "raw_run_jump_minimum_overlap_ms",
                self.raw_run_jump_minimum_overlap_ms,
            ),
            ("raw_run_jump_max_tempo_ratio", self.raw_run_jump_max_tempo_ratio),
            (
                "raw_self_minimum_retained_domain_ratio",
                self.raw_self_minimum_retained_domain_ratio,
            ),
        )
        for name, value in finite_positive:
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.maximum_bpm <= self.minimum_bpm:
            raise ValueError("maximum_bpm must exceed minimum_bpm")
        if self.maximum_excursion_seconds <= self.minimum_excursion_seconds:
            raise ValueError(
                "maximum_excursion_seconds must exceed minimum_excursion_seconds"
            )
        if (
            not math.isfinite(self.minimum_local_strength)
            or not 0.0 <= self.minimum_local_strength <= 1.0
        ):
            raise ValueError("minimum_local_strength must lie in [0, 1]")
        integer_fields = (
            "maximum_boundary_seeds",
            "maximum_pair_seeds",
            "maximum_base_hypotheses",
            "maximum_jump_candidates",
            "maximum_ramp_candidates",
            "maximum_candidates",
            "raw_run_maximum_retained_runs",
            "raw_run_rational_limit",
            "raw_run_minimum_observations",
            "raw_self_minimum_retained_beats",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.maximum_candidates > 64:
            raise ValueError("maximum_candidates must not exceed the raw scorer cap of 64")
        if not 0.0 < self.raw_run_primary_snap_fraction < 1.0:
            raise ValueError("raw_run_primary_snap_fraction must lie in (0, 1)")
        if not 0.0 < self.raw_run_minimum_deviation_fraction < 1.0:
            raise ValueError("raw_run_minimum_deviation_fraction must lie in (0, 1)")
        if self.raw_run_jump_max_tempo_ratio <= 1.0:
            raise ValueError("raw_run_jump_max_tempo_ratio must exceed 1")
        if not 0.0 < self.raw_self_minimum_retained_domain_ratio <= 1.0:
            raise ValueError("raw_self_minimum_retained_domain_ratio must lie in (0, 1]")


DEFAULT_TEMPO_TRACK_CONFIG = TempoTrackConfig()


@dataclass(frozen=True)
class LocalTempoObservation:
    source: Literal["beatthis", "raw_audio"]
    center_time_ms: float
    window_start_ms: float
    window_end_ms: float
    bpm: float
    strength: float


@dataclass(frozen=True)
class TimingCandidateDiagnostic:
    fingerprint_sha256: str
    curve_class: str
    source: str
    generation_score: float


@dataclass(frozen=True)
class TempoTrackDiagnostics:
    version: str
    beat_peak_count: int
    raw_boundary_count: int
    pair_seed_count: int
    shared_start_beat: int
    shared_end_beat: int
    primary_origin_time_ms: float
    primary_bpm: float
    candidate_count: int
    candidate_cap_pruning_reason: str | None = None


@dataclass(frozen=True)
class TempoTrackRawRunDiagnostic:
    direction: Literal["up", "down"]
    start_time_ms: float
    end_time_ms: float
    expanded_start_time_ms: float
    expanded_end_time_ms: float
    median_bpm: float
    weighted_median_delta_bpm: float
    observation_count: int
    summed_strength: float


@dataclass(frozen=True)
class TempoTrackProductionSelection:
    status: Literal["v3_accepted", "v2_fallback"]
    selected_candidate_index: int | None
    selected_fingerprint_sha256: str | None
    lane: Literal[
        "constant",
        "paired_jump",
        "boundary_nominal_backbone",
        "early_half_primary_prefix_step",
        "localized_ramp",
        "fallback",
    ]
    fallback_reason: str | None
    raw_run: TempoTrackRawRunDiagnostic | None
    eligible_candidate_indices: tuple[int, ...]
    raw_self_rank_by_candidate: tuple[tuple[int, int], ...]
    beatthis_aba_rank_by_candidate: tuple[tuple[int, int], ...]
    paired_raw_gain_by_candidate: tuple[tuple[int, float, float, str], ...] = ()


@dataclass(frozen=True)
class TempoTrackResult:
    observations: tuple[LocalTempoObservation, ...]
    candidates: tuple[PhaseContinuousTimingCurve, ...]
    candidate_diagnostics: tuple[TimingCandidateDiagnostic, ...]
    diagnostics: TempoTrackDiagnostics
    raw_ranking: RawAudioEvidenceRanking | None = None
    raw_self_ranking: RawAudioEvidenceRanking | None = None
    production_selection: TempoTrackProductionSelection | None = None

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("candidates must be non-empty")
        if len(self.candidates) != len(self.candidate_diagnostics):
            raise ValueError("candidate diagnostics must be parallel to candidates")
        if len(self.candidates) > 64:
            raise ValueError("candidate count exceeds the raw scorer cap")
        fingerprints = tuple(candidate.fingerprint_sha256 for candidate in self.candidates)
        if len(fingerprints) != len(set(fingerprints)):
            raise ValueError("candidate fingerprints must be unique")
        for candidate, diagnostic in zip(self.candidates, self.candidate_diagnostics):
            if candidate.fingerprint_sha256 != diagnostic.fingerprint_sha256:
                raise ValueError("candidate diagnostic fingerprint mismatch")

    @property
    def ranked_candidates(self) -> tuple[PhaseContinuousTimingCurve, ...]:
        if self.raw_ranking is None or not self.raw_ranking.ranked_scores:
            return self.candidates
        ranked_indices = tuple(
            score.candidate_index for score in self.raw_ranking.ranked_scores
        )
        return tuple(self.candidates[index] for index in ranked_indices)

    @property
    def selected_candidate_index(self) -> int | None:
        if self.production_selection is not None:
            return self.production_selection.selected_candidate_index
        if self.raw_ranking is not None and self.raw_ranking.selected_candidate_index is not None:
            return self.raw_ranking.selected_candidate_index
        return 0

    @property
    def selected_candidate(self) -> PhaseContinuousTimingCurve | None:
        index = self.selected_candidate_index
        if index is None:
            return None
        return self.candidates[index]


def tempo_track_result_to_dict(
    result: TempoTrackResult,
    *,
    include_observations: bool = True,
) -> dict[str, object]:
    """Return a stable JSON-serializable diagnostic view of a generator run.

    Candidate rows deliberately join generation diagnostics, the complete
    analytic curve (including sections), and raw-audio scoring/rank by the
    immutable candidate index.  This helper does not generate, score, or
    reorder candidates.
    """

    if not isinstance(result, TempoTrackResult):
        raise TypeError("result must be a TempoTrackResult")
    if not isinstance(include_observations, bool):
        raise TypeError("include_observations must be a bool")

    raw_scores_by_index = {}
    raw_rank_by_index = {}
    if result.raw_ranking is not None:
        raw_scores_by_index = {
            score.candidate_index: score
            for score in result.raw_ranking.candidate_scores
        }
        raw_rank_by_index = {
            score.candidate_index: rank
            for rank, score in enumerate(result.raw_ranking.ranked_scores, start=1)
        }
    raw_self_scores_by_index = {}
    raw_self_rank_by_index = {}
    if result.raw_self_ranking is not None:
        raw_self_scores_by_index = {
            score.candidate_index: score
            for score in result.raw_self_ranking.candidate_scores
        }
        raw_self_rank_by_index = {
            score.candidate_index: rank
            for rank, score in enumerate(result.raw_self_ranking.ranked_scores, start=1)
        }
    paired_gain_by_index = {}
    if result.production_selection is not None:
        paired_gain_by_index = {
            index: (raw_gain, collapsed_raw_score, collapsed_fingerprint_sha256)
            for (
                index,
                raw_gain,
                collapsed_raw_score,
                collapsed_fingerprint_sha256,
            ) in result.production_selection.paired_raw_gain_by_candidate
        }
    production_selected_index = (
        result.production_selection.selected_candidate_index
        if result.production_selection is not None
        else result.selected_candidate_index
    )

    candidate_rows: list[dict[str, object]] = []
    for candidate_index, (candidate, diagnostic) in enumerate(
        zip(result.candidates, result.candidate_diagnostics)
    ):
        raw_score = raw_scores_by_index.get(candidate_index)
        raw_self_score = raw_self_scores_by_index.get(candidate_index)
        paired_gain = paired_gain_by_index.get(candidate_index)
        candidate_rows.append(
            {
                "candidate_index": candidate_index,
                "selected": production_selected_index == candidate_index,
                "raw_rank": raw_rank_by_index.get(candidate_index),
                "raw_self_rank": raw_self_rank_by_index.get(candidate_index),
                "fingerprint_sha256": diagnostic.fingerprint_sha256,
                "curve_class": diagnostic.curve_class,
                "source": diagnostic.source,
                "generation_score": diagnostic.generation_score,
                "start_time_ms": candidate.start_time_ms,
                "end_time_ms": candidate.end_time_ms,
                "boundary_times_ms": list(candidate.boundary_times_ms),
                "curve": candidate.to_dict(),
                "raw_audio_score": None
                if raw_score is None
                else {
                    "raw_score": raw_score.raw_score,
                    "mean_beat_support": raw_score.mean_beat_support,
                    "mean_half_beat_support": raw_score.mean_half_beat_support,
                    "window_contrast_p10": raw_score.window_contrast_p10,
                    "retained_beat_count": raw_score.retained_beat_count,
                    "complete_window_count": raw_score.complete_window_count,
                    "unavailable_reason": raw_score.unavailable_reason,
                },
                "raw_audio_self_score": None
                if raw_self_score is None
                else {
                    "raw_score": raw_self_score.raw_score,
                    "mean_beat_support": raw_self_score.mean_beat_support,
                    "mean_half_beat_support": raw_self_score.mean_half_beat_support,
                    "window_contrast_p10": raw_self_score.window_contrast_p10,
                    "retained_beat_count": raw_self_score.retained_beat_count,
                    "candidate_domain_beat_count": raw_self_score.candidate_domain_beat_count,
                    "retained_domain_ratio": raw_self_score.retained_domain_ratio,
                    "complete_window_count": raw_self_score.complete_window_count,
                    "complete_window_start_beats": list(
                        raw_self_score.complete_window_start_beats
                    ),
                    "unavailable_reason": raw_self_score.unavailable_reason,
                },
                "collapsed_constant_counterfactual": None
                if paired_gain is None
                else {
                    "raw_gain": paired_gain[0],
                    "raw_gain_sign": _raw_gain_sign(paired_gain[0]),
                    "collapsed_raw_score": paired_gain[1],
                    "collapsed_fingerprint_sha256": paired_gain[2],
                },
            }
        )

    selected_candidate = result.selected_candidate
    production_selection: dict[str, object] | None = None
    if result.production_selection is not None:
        selection = result.production_selection
        raw_run = selection.raw_run
        production_selection = {
            "status": selection.status,
            "selected_candidate_index": selection.selected_candidate_index,
            "selected_fingerprint_sha256": selection.selected_fingerprint_sha256,
            "lane": selection.lane,
            "fallback_reason": selection.fallback_reason,
            "eligible_candidate_indices": list(selection.eligible_candidate_indices),
            "selection_rank_scope": "eligible_lane",
            "raw_self_rank_by_candidate": [
                {"candidate_index": index, "rank": rank}
                for index, rank in selection.raw_self_rank_by_candidate
            ],
            "beatthis_aba_rank_by_candidate": [
                {"candidate_index": index, "rank": rank}
                for index, rank in selection.beatthis_aba_rank_by_candidate
            ],
            "paired_raw_gain_by_candidate": [
                {
                    "candidate_index": index,
                    "raw_gain": raw_gain,
                    "collapsed_raw_score": collapsed_raw_score,
                    "collapsed_fingerprint_sha256": collapsed_fingerprint_sha256,
                }
                for (
                    index,
                    raw_gain,
                    collapsed_raw_score,
                    collapsed_fingerprint_sha256,
                ) in selection.paired_raw_gain_by_candidate
            ],
            "raw_run": None
            if raw_run is None
            else {
                "direction": raw_run.direction,
                "start_time_ms": raw_run.start_time_ms,
                "end_time_ms": raw_run.end_time_ms,
                "expanded_start_time_ms": raw_run.expanded_start_time_ms,
                "expanded_end_time_ms": raw_run.expanded_end_time_ms,
                "median_bpm": raw_run.median_bpm,
                "weighted_median_delta_bpm": raw_run.weighted_median_delta_bpm,
                "observation_count": raw_run.observation_count,
                "summed_strength": raw_run.summed_strength,
            },
        }

    payload: dict[str, object] = {
        "schema": TEMPO_TRACK_RESULT_DUMP_SCHEMA_VERSION,
        "tempo_track_version": result.diagnostics.version,
        "selected_candidate_index": result.selected_candidate_index,
        "selected_fingerprint_sha256": None
        if selected_candidate is None
        else selected_candidate.fingerprint_sha256,
        "production_selection": production_selection,
        "raw_ranking_unavailable_reason": (
            result.raw_ranking.unavailable_reason
            if result.raw_ranking is not None
            else "raw_audio_evidence_not_supplied"
        ),
        "raw_self_ranking_unavailable_reason": (
            result.raw_self_ranking.unavailable_reason
            if result.raw_self_ranking is not None
            else "raw_audio_evidence_not_supplied"
        ),
        "diagnostics": {
            "beat_peak_count": result.diagnostics.beat_peak_count,
            "raw_boundary_count": result.diagnostics.raw_boundary_count,
            "pair_seed_count": result.diagnostics.pair_seed_count,
            "shared_start_beat": result.diagnostics.shared_start_beat,
            "shared_end_beat": result.diagnostics.shared_end_beat,
            "primary_origin_time_ms": result.diagnostics.primary_origin_time_ms,
            "primary_bpm": result.diagnostics.primary_bpm,
            "candidate_count": result.diagnostics.candidate_count,
            "candidate_cap_pruning_reason": (
                result.diagnostics.candidate_cap_pruning_reason
            ),
        },
        "candidates": candidate_rows,
    }
    if include_observations:
        payload["observations"] = [
            {
                "source": observation.source,
                "center_time_ms": observation.center_time_ms,
                "window_start_ms": observation.window_start_ms,
                "window_end_ms": observation.window_end_ms,
                "bpm": observation.bpm,
                "strength": observation.strength,
            }
            for observation in result.observations
        ]
    return payload


@dataclass(frozen=True)
class _BaseHypothesis:
    origin_time_ms: float
    bpm: float
    score: float


@dataclass(frozen=True)
class _BoundarySeed:
    time_ms: float
    rank_score: float


@dataclass(frozen=True)
class _PairSeed:
    left_time_ms: float
    right_time_ms: float
    rank_score: float
    source: str
    preferred_bpm: float | None = None


@dataclass(frozen=True)
class _CurveProposal:
    curve: PhaseContinuousTimingCurve
    source: str
    score: float
    collapse_bpm: float
    aba_support_delta: float | None = None


@dataclass(frozen=True)
class _LocalizedRampRun:
    observation_start_ms: float
    observation_end_ms: float
    fitted_start_bpm: float
    fitted_end_bpm: float
    r_squared: float
    mean_strength: float
    point_count: int
    generation_score: float


@dataclass(frozen=True)
class _PersistentStepSpec:
    left_bpm: float
    right_bpm: float
    transition_target_ms: float


@dataclass(frozen=True)
class _LocalizedBackboneSelection:
    curve: PhaseContinuousTimingCurve
    raw_rank: int
    beatthis_rank: int
    constant_left_raw_gain: float
    constant_left_beatthis_gain: float
    constant_right_raw_gain: float
    constant_right_beatthis_gain: float

    @property
    def double_supported(self) -> bool:
        return (
            self.constant_left_raw_gain > 0.0
            and self.constant_left_beatthis_gain > 0.0
            and self.constant_right_raw_gain > 0.0
            and self.constant_right_beatthis_gain > 0.0
        )


@dataclass(frozen=True)
class _LocalizedRampTriplet:
    ramp: PhaseContinuousTimingCurve
    step: PhaseContinuousTimingCurve
    no_ramp: PhaseContinuousTimingCurve
    endpoint_delta_scale: float


@dataclass(frozen=True)
class _LocalizedRampObservationGate:
    raw_step_gain: float
    raw_no_ramp_gain: float
    beatthis_step_gain: float
    beatthis_no_ramp_gain: float

    @property
    def eligible(self) -> bool:
        return (
            self.raw_step_gain > 0.0
            and self.raw_no_ramp_gain > 0.0
            and self.beatthis_step_gain > 0.0
            and self.beatthis_no_ramp_gain > 0.0
        )


@dataclass(frozen=True)
class _LocalizedRampChoice:
    proposal: _CurveProposal
    backbone: _LocalizedBackboneSelection
    gate: _LocalizedRampObservationGate
    run: _LocalizedRampRun
    endpoint_delta_scale: float


@dataclass(frozen=True)
class _NominalBackboneChoice:
    proposal: _CurveProposal
    minimum_edge_gain: float
    total_edge_gain: float
    raw_support_counts: tuple[int, ...]
    nominal: bool


@dataclass(frozen=True)
class _EarlyHalfPrimaryPrefixStepChoice:
    proposal: _CurveProposal
    left_boundary_time_ms: float
    right_boundary_time_ms: float
    boundary_time_ms: float
    left_rank_score: float
    right_rank_score: float
    beatthis_support_delta: float
    prefix_observation_count: int


@dataclass(frozen=True)
class _RegionalTempoMode:
    source: Literal["base", "beatthis", "raw_audio"]
    bpm: float
    support: float
    alias_multiplier: float


_REGIONAL_OBSERVATION_MODES_REQUEST_CACHE: ContextVar[
    dict[tuple[object, ...], tuple[_RegionalTempoMode, ...]] | None
] = ContextVar(
    "timing_v3_regional_observation_modes_request_cache",
    default=None,
)


@dataclass(frozen=True)
class _SideSegmentSeed:
    time_ms: float
    rank_score: float
    source: Literal["boundary", "mode_change_midpoint"]


@dataclass(frozen=True)
class _JumpProposalBatch:
    proposals: tuple[_CurveProposal, ...]
    pruning_reason: str | None = None


_JumpRetentionFamily = Literal[
    "short_aba_paired_boundary",
    "persistent",
    "long_aba",
    "multi_step",
    "overflow",
]


_JUMP_RETENTION_FAMILY_QUOTAS: tuple[tuple[_JumpRetentionFamily, int], ...] = (
    ("short_aba_paired_boundary", 14),
    ("persistent", 10),
    ("long_aba", 8),
    ("multi_step", 6),
    ("overflow", 6),
)

_EXP017_SHORT_ABA_MIN_SECONDS = 2.0
_EXP017_SHORT_ABA_MAX_SECONDS = 8.0
_EXP017_LONG_ABA_MAX_SECONDS = 60.0
_EXP026_BOUNDARY_TOLERANCE_MS = 1000.0
_EXP026_RECOMBINED_PROPOSAL_CAP = 6
_EXP026_LOCAL_ALIAS_MIN_BPM = 60.0
_EXP026_LOCAL_ALIAS_MAX_BPM = 300.0
_EXP026_SIDE_SEGMENT_SOURCE = "observation_side_segment"
_EXP026_SIDE_SEGMENT_SEED_CAP = 48
_EXP026_MODE_CHANGE_REGION_CAP = 16
_EXP026_MODE_CHANGE_REGION_SECONDS = 20.0
_EXP026_MODE_CHANGE_BOUNDARY_RADIUS_MS = 3000.0
_EXP026_CLOSED_ABA_MINIMUM_RAW_GAIN = 0.01
_EXP026_TWO_SECTION_MINIMUM_FIRST_DURATION_RATIO = 0.125
_EXP026_SPARSE_MIDDLE_BEAT_CAP = 6
_SHORT_WINDOW_MONOTONE_CHAIN_SOURCE = "short_window_monotone_chain"
_SHORT_CHAIN_WINDOW_SECONDS = 3.0
_SHORT_CHAIN_HOP_SECONDS = 0.5
_SHORT_CHAIN_MAXIMUM_BPM = 300.0
_SHORT_CHAIN_MINIMUM_SECTIONS = 4
_SHORT_CHAIN_MAXIMUM_SECTIONS = 6
_SHORT_CHAIN_SECTION_BEATS = 4
_SHORT_CHAIN_MINIMUM_TOTAL_DELTA_BPM = 12.0
_SHORT_CHAIN_MINIMUM_TRAJECTORY_SECONDS = 4.0
_SHORT_CHAIN_MAXIMUM_TRAJECTORY_SECONDS = 12.0
_SHORT_CHAIN_SOURCE_AGREEMENT_BPM = 3.0
_BOUNDARY_NOMINAL_BACKBONE_SOURCE = "boundary_nominal_backbone"
_NOMINAL_BACKBONE_MINIMUM_RAW_SUPPORT = 12
_NOMINAL_BACKBONE_MAXIMUM_WINNERS = 2
_NOMINAL_BACKBONE_MINIMUM_SEGMENT_SAMPLES = 12
_NOMINAL_BACKBONE_SEED_RADIUS_MS = 4000.0
_NOMINAL_BACKBONE_PHASE_COUNT = 60
_EARLY_HALF_PRIMARY_PREFIX_STEP_SOURCE = "early_half_primary_prefix_step"
_EARLY_HALF_PREFIX_MIN_PRIMARY_BPM = 220.0
_EARLY_HALF_PREFIX_MIN_TIME_MS = 8000.0
_EARLY_HALF_PREFIX_MAX_TIME_MS = 30000.0
_EARLY_HALF_PREFIX_MIN_GAP_MS = 2000.0
_EARLY_HALF_PREFIX_MAX_GAP_MS = 8000.0
_EARLY_HALF_PREFIX_MIN_BOUNDARY_RANK = 350.0
_EARLY_HALF_PREFIX_LOOKBACK_MS = 14000.0
_EARLY_HALF_PREFIX_LEFT_MARGIN_MS = 500.0
_EARLY_HALF_PREFIX_MIN_SOURCE_OBSERVATIONS = 2
_EARLY_HALF_PREFIX_MIN_RAW_GAIN = -0.01
_EARLY_HALF_PREFIX_MIN_BEATTHIS_GAIN = -0.1
_LOCALIZED_BACKBONE_RAMP_SOURCE = "localized_backbone_ramp"
_LOCALIZED_RAMP_ENDPOINT_SCALES = (0.4, 0.5, 0.6, 2.0 / 3.0, 0.75, 1.0)
_LOCALIZED_RAMP_RATIONAL_ALIASES = (
    0.5,
    2.0 / 3.0,
    0.75,
    1.0,
    4.0 / 3.0,
    1.5,
    2.0,
)


@dataclass(frozen=True)
class _ShortChainPoint:
    time_ms: float
    beatthis_bpm: float
    raw_bpm: float
    combined_bpm: float


@dataclass(frozen=True)
class _ShortChainRegion:
    start_time_ms: float
    end_time_ms: float
    direction: int
    score: float


@dataclass(frozen=True)
class _SmoothedRawTempoObservation:
    observation: LocalTempoObservation
    normalized_bpm: float
    smoothed_bpm: float
    delta_bpm: float
    direction: Literal["up", "down"] | None


def generate_timing_candidates(
    prediction: FrameTimingPrediction,
    *,
    audio_evidence: RawAudioEvidence | None = None,
    config: TempoTrackConfig = DEFAULT_TEMPO_TRACK_CONFIG,
) -> TempoTrackResult:
    phase_cache_token = _PHASE_ORIGIN_REQUEST_CACHE.set({})
    event_support_cache_token = _EVENT_SUPPORT_REQUEST_CACHE.set({})
    regional_modes_cache_token = _REGIONAL_OBSERVATION_MODES_REQUEST_CACHE.set(
        {}
    )
    try:
        return _generate_timing_candidates(
            prediction,
            audio_evidence=audio_evidence,
            config=config,
        )
    finally:
        _REGIONAL_OBSERVATION_MODES_REQUEST_CACHE.reset(
            regional_modes_cache_token
        )
        _EVENT_SUPPORT_REQUEST_CACHE.reset(event_support_cache_token)
        _PHASE_ORIGIN_REQUEST_CACHE.reset(phase_cache_token)


def _generate_timing_candidates(
    prediction: FrameTimingPrediction,
    *,
    audio_evidence: RawAudioEvidence | None = None,
    config: TempoTrackConfig = DEFAULT_TEMPO_TRACK_CONFIG,
) -> TempoTrackResult:
    """Generate a small constant/jump/ramp Timing-v3 candidate family.

    The function is metadata-blind: only BeatThis frame activations and, when
    supplied, candidate-independent raw-audio flux are read.  Every returned
    curve is internally phase-continuous on one integer beat axis.  Exp014
    deliberately allows different candidate alternatives to end at different
    beat indices so persistent tempo changes can cover the same physical
    audio duration without forcing an invalid common beat count.
    """

    if not isinstance(prediction, FrameTimingPrediction):
        raise TypeError("prediction must be a FrameTimingPrediction")
    if not isinstance(config, TempoTrackConfig):
        raise TypeError("config must be a TempoTrackConfig")
    if audio_evidence is not None and not isinstance(audio_evidence, RawAudioEvidence):
        raise TypeError("audio_evidence must be RawAudioEvidence or None")

    beat_signal = np.asarray(prediction.beat_prob, dtype=np.float64)
    frame_rate_hz = float(prediction.frame_rate_hz)
    duration_ms = 1000.0 * prediction.frame_count / frame_rate_hz
    global_candidates = _global.extract_global_constant_jump_candidates(prediction)

    beat_observations = _sliding_tempo_observations(
        beat_signal,
        frame_rate_hz=frame_rate_hz,
        source="beatthis",
        config=config,
    )
    raw_observations: tuple[LocalTempoObservation, ...] = ()
    short_observations: tuple[LocalTempoObservation, ...] = ()
    if audio_evidence is not None and audio_evidence.valid_frame_count:
        raw_signal = np.max(audio_evidence.band_flux, axis=1).astype(
            np.float64, copy=False
        )
        raw_rate_hz = _evidence_frame_rate_hz(audio_evidence)
        raw_observations = _sliding_tempo_observations(
            raw_signal,
            frame_rate_hz=raw_rate_hz,
            source="raw_audio",
            config=config,
            time_offset_ms=float(audio_evidence.frame_center_seconds[0] * 1000.0),
        )
        short_config = replace(
            config,
            local_window_seconds=_SHORT_CHAIN_WINDOW_SECONDS,
            local_hop_seconds=_SHORT_CHAIN_HOP_SECONDS,
            maximum_bpm=max(_SHORT_CHAIN_MAXIMUM_BPM, config.maximum_bpm),
        )
        short_observations = tuple(
            sorted(
                _sliding_tempo_observations(
                    beat_signal,
                    frame_rate_hz=frame_rate_hz,
                    source="beatthis",
                    config=short_config,
                )
                + _sliding_tempo_observations(
                    raw_signal,
                    frame_rate_hz=raw_rate_hz,
                    source="raw_audio",
                    config=short_config,
                    time_offset_ms=float(
                        audio_evidence.frame_center_seconds[0] * 1000.0
                    ),
                ),
                key=lambda value: (value.center_time_ms, value.source, value.bpm),
            )
        )
    observations = tuple(
        sorted(
            beat_observations + raw_observations,
            key=lambda value: (value.center_time_ms, value.source, value.bpm),
        )
    )

    bases = _base_hypotheses(
        beat_signal,
        frame_rate_hz=frame_rate_hz,
        duration_ms=duration_ms,
        global_candidates=global_candidates,
        observations=observations,
        config=config,
    )
    primary = bases[0]
    shared_end_beat = _shared_beat_count(
        observations,
        primary=primary,
        duration_ms=duration_ms,
        config=config,
    )

    constant_proposals = tuple(
        _CurveProposal(
            curve=PhaseContinuousTimingCurve(
                origin_beat=0,
                origin_time_ms=base.origin_time_ms,
                sections=(
                    ConstantTempoSection(
                        start_beat=0,
                        end_beat=_terminal_beat_for_constant_tempo(
                            origin_beat=0,
                            origin_time_ms=base.origin_time_ms,
                            duration_ms=duration_ms,
                            bpm=base.bpm,
                        ),
                        bpm=base.bpm,
                    ),
                ),
            ),
            source="global_constant",
            score=base.score,
            collapse_bpm=base.bpm,
        )
        for base in bases
    )

    raw_boundaries = _unmerged_boundary_seeds(global_candidates, config=config)
    pair_seeds = _pair_seeds(
        raw_boundaries,
        observations=observations,
        bases=bases,
        config=config,
    )
    jump_batch = _jump_proposals(
        beat_signal,
        frame_rate_hz=frame_rate_hz,
        duration_ms=duration_ms,
        shared_end_beat=shared_end_beat,
        global_candidates=global_candidates,
        bases=bases,
        observations=observations,
        boundary_seeds=raw_boundaries,
        pair_seeds=pair_seeds,
        config=config,
        short_observations=short_observations,
    )
    jump_proposals = jump_batch.proposals
    nominal_backbone_choices = _boundary_nominal_backbone_choices(
        audio_evidence=audio_evidence,
        beat_signal=beat_signal,
        frame_rate_hz=frame_rate_hz,
        duration_ms=duration_ms,
        boundary_seeds=raw_boundaries,
        short_observations=short_observations,
        config=config,
    )
    nominal_backbone_proposals = tuple(
        choice.proposal for choice in nominal_backbone_choices
    )
    early_half_prefix_choices = _early_half_primary_prefix_step_choices(
        beat_signal=beat_signal,
        frame_rate_hz=frame_rate_hz,
        duration_ms=duration_ms,
        global_candidates=global_candidates,
        primary=primary,
        boundary_seeds=raw_boundaries,
        short_observations=observations,
        config=config,
    )
    early_half_prefix_proposals = tuple(
        choice.proposal for choice in early_half_prefix_choices
    )
    localized_ramp_choice = _localized_backbone_ramp_choice(
        observations,
        beat_observations=beat_observations,
        raw_observations=raw_observations,
        audio_evidence=audio_evidence,
        beat_signal=beat_signal,
        frame_rate_hz=frame_rate_hz,
        duration_ms=duration_ms,
        boundary_seeds=raw_boundaries,
        jump_proposals=jump_proposals,
        config=config,
    )
    localized_ramp_proposals = (
        ()
        if localized_ramp_choice is None
        else (localized_ramp_choice.proposal,)
    )
    ramp_proposals = (
        localized_ramp_proposals
        + early_half_prefix_proposals
        + nominal_backbone_proposals
        + _ramp_proposals(
            observations,
            primary=primary,
            duration_ms=duration_ms,
            shared_end_beat=shared_end_beat,
            config=config,
        )
    )

    proposals = _bounded_proposals(
        constant_proposals,
        jump_proposals,
        ramp_proposals,
        config=config,
    )
    proposal_count_before_cap = (
        len(constant_proposals) + len(jump_proposals) + len(ramp_proposals)
    )
    pruning_reasons: list[str] = []
    if jump_batch.pruning_reason is not None:
        pruning_reasons.append(jump_batch.pruning_reason)
    if (
        len(proposals) >= config.maximum_candidates
        and proposal_count_before_cap > len(proposals)
    ):
        pruning_reasons.append(f"candidate_cap_{config.maximum_candidates}")
    candidate_cap_pruning_reason = ";".join(pruning_reasons) or None
    curves = tuple(proposal.curve for proposal in proposals)
    raw_ranking = (
        score_raw_audio_evidence(audio_evidence, curves)
        if audio_evidence is not None and _candidates_share_beat_domain(curves)
        else None
    )
    raw_self_ranking = (
        score_raw_audio_evidence_independent(audio_evidence, curves)
        if audio_evidence is not None
        else None
    )
    collapsed_counterfactuals = _collapsed_counterfactuals_for_proposals(
        proposals,
        duration_ms=duration_ms,
        config=config,
    )
    collapsed_raw_self_ranking = (
        _score_collapsed_counterfactuals_independent(
            audio_evidence,
            collapsed_counterfactuals,
        )
        if audio_evidence is not None
        else None
    )
    candidate_diagnostics = tuple(
        TimingCandidateDiagnostic(
            fingerprint_sha256=proposal.curve.fingerprint_sha256,
            curve_class=proposal.curve.curve_class,
            source=proposal.source,
            generation_score=proposal.score,
        )
        for proposal in proposals
    )
    jump_or_constant_selection = _select_exp014_production_candidate(
        curves,
        raw_self_ranking=raw_self_ranking,
        collapsed_raw_self_ranking=collapsed_raw_self_ranking,
        collapsed_counterfactuals=collapsed_counterfactuals,
        observations=observations,
        primary=primary,
        beat_signal=beat_signal,
        frame_rate_hz=frame_rate_hz,
        candidate_sources=tuple(proposal.source for proposal in proposals),
        candidate_generation_scores=tuple(proposal.score for proposal in proposals),
        config=config,
    )
    nominal_or_existing_selection = _nominal_backbone_fast_lane_selection(
        proposals,
        choices=nominal_backbone_choices,
        raw_self_ranking=raw_self_ranking,
        fallback=jump_or_constant_selection,
        config=config,
    )
    early_half_or_existing_selection = _early_half_primary_prefix_fast_lane_selection(
        proposals,
        choices=early_half_prefix_choices,
        raw_self_ranking=raw_self_ranking,
        collapsed_raw_self_ranking=collapsed_raw_self_ranking,
        collapsed_counterfactuals=collapsed_counterfactuals,
        fallback=nominal_or_existing_selection,
        config=config,
    )
    production_selection = _localized_ramp_fast_lane_selection(
        proposals,
        choice=localized_ramp_choice,
        raw_self_ranking=raw_self_ranking,
        fallback=early_half_or_existing_selection,
        config=config,
    )
    production_selection = _stable_audio_consensus_veto_selection(
        production_selection,
        curves,
        raw_self_ranking=raw_self_ranking,
        observations=observations,
        primary=primary,
        alias_only_constant_fallback=nominal_or_existing_selection,
        paired_gain_diagnostics=production_selection.paired_raw_gain_by_candidate,
        config=config,
    )
    diagnostics = TempoTrackDiagnostics(
        version=TEMPO_TRACK_VERSION,
        beat_peak_count=len(global_candidates.beat_peaks),
        raw_boundary_count=len(raw_boundaries),
        pair_seed_count=len(pair_seeds),
        shared_start_beat=0,
        shared_end_beat=shared_end_beat,
        primary_origin_time_ms=primary.origin_time_ms,
        primary_bpm=primary.bpm,
        candidate_count=len(curves),
        candidate_cap_pruning_reason=candidate_cap_pruning_reason,
    )
    return TempoTrackResult(
        observations=observations,
        candidates=curves,
        candidate_diagnostics=candidate_diagnostics,
        diagnostics=diagnostics,
        raw_ranking=raw_ranking,
        raw_self_ranking=raw_self_ranking,
        production_selection=production_selection,
    )


def _stable_audio_consensus_veto_selection(
    selection: TempoTrackProductionSelection,
    candidates: tuple[PhaseContinuousTimingCurve, ...],
    *,
    raw_self_ranking: RawAudioEvidenceRanking | None,
    observations: tuple[LocalTempoObservation, ...],
    primary: _BaseHypothesis,
    alias_only_constant_fallback: TempoTrackProductionSelection,
    paired_gain_diagnostics: tuple[tuple[int, float, float, str], ...],
    config: TempoTrackConfig,
) -> TempoTrackProductionSelection:
    if selection.lane in {"constant", "fallback"}:
        return selection
    if raw_self_ranking is None or not raw_self_ranking.ranked_scores:
        return selection

    selected_index = selection.selected_candidate_index
    half_prefix_alias_only = (
        selection.lane == "early_half_primary_prefix_step"
        and selected_index is not None
        and _curve_is_primary_alias_family(
            candidates[selected_index],
            primary_bpm=primary.bpm,
            config=config,
        )
    )
    if half_prefix_alias_only and alias_only_constant_fallback.lane == "constant":
        return alias_only_constant_fallback
    if not half_prefix_alias_only:
        raw_observations = tuple(
            observation for observation in observations if observation.source == "raw_audio"
        )
        beatthis_observations = tuple(
            observation for observation in observations if observation.source == "beatthis"
        )
        if not raw_observations or not beatthis_observations:
            return selection
        if _raw_tempo_runs(raw_observations, base_bpm=primary.bpm, config=config):
            return selection
        if _raw_tempo_runs(beatthis_observations, base_bpm=primary.bpm, config=config):
            return selection

    raw_scores_by_index = {
        score.candidate_index: score for score in raw_self_ranking.candidate_scores
    }
    constant_eligible = tuple(
        index
        for index, candidate in enumerate(candidates)
        if candidate.curve_class == "constant"
        and _raw_self_score_is_production_usable(
            raw_scores_by_index.get(index),
            config=config,
        )
    )
    if not constant_eligible:
        return selection

    constant_raw_rank_by_index = _lane_raw_self_rank_by_index(
        constant_eligible,
        candidates=candidates,
        raw_scores_by_index=raw_scores_by_index,
    )
    best_constant = min(
        constant_eligible,
        key=lambda index: (
            constant_raw_rank_by_index[index],
            -_required_raw_score(raw_scores_by_index[index]),
            candidates[index].fingerprint_sha256,
        ),
    )
    return TempoTrackProductionSelection(
        status="v3_accepted",
        selected_candidate_index=best_constant,
        selected_fingerprint_sha256=candidates[best_constant].fingerprint_sha256,
        lane="constant",
        fallback_reason=None,
        raw_run=None,
        eligible_candidate_indices=constant_eligible,
        raw_self_rank_by_candidate=_rank_items(constant_raw_rank_by_index),
        beatthis_aba_rank_by_candidate=(),
        paired_raw_gain_by_candidate=paired_gain_diagnostics,
    )


def _nominal_backbone_fast_lane_selection(
    proposals: tuple[_CurveProposal, ...],
    *,
    choices: tuple[_NominalBackboneChoice, ...],
    raw_self_ranking: RawAudioEvidenceRanking | None,
    fallback: TempoTrackProductionSelection,
    config: TempoTrackConfig,
) -> TempoTrackProductionSelection:
    """Select only a pre-gated nominal backbone over a constant winner."""

    if fallback.lane != "constant" or not choices or raw_self_ranking is None:
        return fallback
    raw_by_index = {
        value.candidate_index: value for value in raw_self_ranking.candidate_scores
    }
    indexed: list[tuple[int, _NominalBackboneChoice]] = []
    by_fingerprint = {
        choice.proposal.curve.fingerprint_sha256: choice for choice in choices
    }
    for index, proposal in enumerate(proposals):
        if proposal.source != _BOUNDARY_NOMINAL_BACKBONE_SOURCE:
            continue
        choice = by_fingerprint.get(proposal.curve.fingerprint_sha256)
        if choice is None or not _raw_self_score_is_production_usable(
            raw_by_index.get(index), config=config
        ):
            continue
        indexed.append((index, choice))
    if not indexed:
        return fallback
    selected, _choice = max(
        indexed,
        key=lambda value: (
            value[1].total_edge_gain,
            value[1].minimum_edge_gain,
            int(value[1].nominal),
            -len(value[1].proposal.curve.sections),
            value[1].proposal.curve.fingerprint_sha256,
        ),
    )
    curve = proposals[selected].curve
    return TempoTrackProductionSelection(
        status="v3_accepted",
        selected_candidate_index=selected,
        selected_fingerprint_sha256=curve.fingerprint_sha256,
        lane="boundary_nominal_backbone",
        fallback_reason=None,
        raw_run=None,
        eligible_candidate_indices=tuple(index for index, _ in indexed),
        raw_self_rank_by_candidate=((selected, 1),),
        beatthis_aba_rank_by_candidate=(),
        paired_raw_gain_by_candidate=(),
    )


def _early_half_primary_prefix_fast_lane_selection(
    proposals: tuple[_CurveProposal, ...],
    *,
    choices: tuple[_EarlyHalfPrimaryPrefixStepChoice, ...],
    raw_self_ranking: RawAudioEvidenceRanking | None,
    collapsed_raw_self_ranking: RawAudioEvidenceRanking | None,
    collapsed_counterfactuals: tuple[PhaseContinuousTimingCurve, ...],
    fallback: TempoTrackProductionSelection,
    config: TempoTrackConfig,
) -> TempoTrackProductionSelection:
    if (
        fallback.lane != "constant"
        or not choices
        or raw_self_ranking is None
        or collapsed_raw_self_ranking is None
    ):
        return fallback
    raw_by_index = {
        value.candidate_index: value for value in raw_self_ranking.candidate_scores
    }
    collapsed_by_index = {
        value.candidate_index: value
        for value in collapsed_raw_self_ranking.candidate_scores
    }
    by_fingerprint = {
        choice.proposal.curve.fingerprint_sha256: choice for choice in choices
    }
    eligible: list[tuple[int, _EarlyHalfPrimaryPrefixStepChoice, float, float, str]] = []
    for index, proposal in enumerate(proposals):
        if proposal.source != _EARLY_HALF_PRIMARY_PREFIX_STEP_SOURCE:
            continue
        choice = by_fingerprint.get(proposal.curve.fingerprint_sha256)
        if choice is None:
            continue
        raw_score = raw_by_index.get(index)
        collapsed_score = collapsed_by_index.get(index)
        if not (
            _raw_self_score_is_production_usable(raw_score, config=config)
            and _raw_self_score_is_production_usable(collapsed_score, config=config)
        ):
            continue
        assert raw_score is not None
        assert collapsed_score is not None
        raw_gain = _required_raw_score(raw_score) - _required_raw_score(collapsed_score)
        if (
            raw_gain < _EARLY_HALF_PREFIX_MIN_RAW_GAIN
            or not math.isfinite(choice.beatthis_support_delta)
            or choice.beatthis_support_delta < _EARLY_HALF_PREFIX_MIN_BEATTHIS_GAIN
        ):
            continue
        eligible.append(
            (
                index,
                choice,
                float(raw_gain),
                _required_raw_score(collapsed_score),
                collapsed_counterfactuals[index].fingerprint_sha256,
            )
        )
    if not eligible:
        return fallback
    selected, choice, raw_gain, collapsed_score, collapsed_fingerprint = max(
        eligible,
        key=lambda value: (
            value[2],
            value[1].beatthis_support_delta,
            min(value[1].left_rank_score, value[1].right_rank_score),
            value[1].proposal.curve.fingerprint_sha256,
        ),
    )
    curve = proposals[selected].curve
    return TempoTrackProductionSelection(
        status="v3_accepted",
        selected_candidate_index=selected,
        selected_fingerprint_sha256=curve.fingerprint_sha256,
        lane="early_half_primary_prefix_step",
        fallback_reason=None,
        raw_run=None,
        eligible_candidate_indices=tuple(value[0] for value in eligible),
        raw_self_rank_by_candidate=((selected, 1),),
        beatthis_aba_rank_by_candidate=((selected, 1),),
        paired_raw_gain_by_candidate=(
            (selected, raw_gain, collapsed_score, collapsed_fingerprint),
        ),
    )


def _localized_ramp_fast_lane_selection(
    proposals: tuple[_CurveProposal, ...],
    *,
    choice: _LocalizedRampChoice | None,
    raw_self_ranking: RawAudioEvidenceRanking | None,
    fallback: TempoTrackProductionSelection,
    config: TempoTrackConfig,
) -> TempoTrackProductionSelection:
    """Select a pre-gated localized ramp without entering the jump selector."""

    if choice is None or raw_self_ranking is None:
        return fallback
    selected = next(
        (
            index
            for index, proposal in enumerate(proposals)
            if proposal.source == _LOCALIZED_BACKBONE_RAMP_SOURCE
            and proposal.curve.fingerprint_sha256
            == choice.proposal.curve.fingerprint_sha256
        ),
        None,
    )
    if selected is None:
        return fallback
    raw_score = next(
        (
            score
            for score in raw_self_ranking.candidate_scores
            if score.candidate_index == selected
        ),
        None,
    )
    if not _raw_self_score_is_production_usable(raw_score, config=config):
        return fallback
    curve = proposals[selected].curve
    if (
        curve.curve_class != "ramp"
        or len(curve.sections) != 4
        or not isinstance(curve.sections[2], LinearTimeRampSection)
        or any(report.phase_discontinuity_ms > 5.0 for report in curve.seam_reports)
    ):
        return fallback
    return TempoTrackProductionSelection(
        status="v3_accepted",
        selected_candidate_index=selected,
        selected_fingerprint_sha256=curve.fingerprint_sha256,
        lane="localized_ramp",
        fallback_reason=None,
        raw_run=None,
        eligible_candidate_indices=(selected,),
        raw_self_rank_by_candidate=((selected, 1),),
        beatthis_aba_rank_by_candidate=(),
        paired_raw_gain_by_candidate=(),
    )


def _select_exp014_production_candidate(
    candidates: tuple[PhaseContinuousTimingCurve, ...],
    *,
    raw_self_ranking: RawAudioEvidenceRanking | None,
    collapsed_raw_self_ranking: RawAudioEvidenceRanking | None,
    collapsed_counterfactuals: tuple[PhaseContinuousTimingCurve, ...],
    observations: tuple[LocalTempoObservation, ...],
    primary: _BaseHypothesis,
    beat_signal: NDArray[np.float64],
    frame_rate_hz: float,
    candidate_sources: tuple[str, ...] | None = None,
    candidate_generation_scores: tuple[float, ...] | None = None,
    config: TempoTrackConfig,
) -> TempoTrackProductionSelection:
    if raw_self_ranking is None or collapsed_raw_self_ranking is None:
        return _fallback_selection("raw_audio_evidence_not_supplied")
    if len(collapsed_counterfactuals) != len(candidates):
        raise ValueError("collapsed counterfactuals must be parallel to candidates")
    if candidate_sources is not None and len(candidate_sources) != len(candidates):
        raise ValueError("candidate sources must be parallel to candidates")
    if candidate_generation_scores is not None and len(candidate_generation_scores) != len(
        candidates
    ):
        raise ValueError("candidate generation scores must be parallel to candidates")
    source_aware = candidate_sources is not None
    sources = candidate_sources or ("",) * len(candidates)
    generation_scores = candidate_generation_scores or (0.0,) * len(candidates)

    raw_observations = tuple(
        observation for observation in observations if observation.source == "raw_audio"
    )
    raw_scores_by_index = {
        score.candidate_index: score
        for score in raw_self_ranking.candidate_scores
    }
    collapsed_scores_by_index = {
        score.candidate_index: score
        for score in collapsed_raw_self_ranking.candidate_scores
    }
    if not raw_self_ranking.ranked_scores:
        return _fallback_selection("raw_self_scoring_unavailable")
    paired_gain_diagnostics = _paired_raw_gain_diagnostics(
        candidates,
        raw_scores_by_index=raw_scores_by_index,
        collapsed_scores_by_index=collapsed_scores_by_index,
        collapsed_counterfactuals=collapsed_counterfactuals,
    )

    raw_runs = _raw_tempo_runs(
        raw_observations,
        base_bpm=primary.bpm,
        config=config,
    )
    raw_run = raw_runs[0] if raw_runs else None

    constant_eligible = tuple(
        index
        for index, candidate in enumerate(candidates)
        if candidate.curve_class == "constant"
        and _raw_self_score_is_production_usable(
            raw_scores_by_index.get(index),
            config=config,
        )
    )
    if not constant_eligible:
        return _fallback_selection(
            "no_production_eligible_constant",
            raw_run=raw_run,
        )
    constant_raw_rank_by_index = _lane_raw_self_rank_by_index(
        constant_eligible,
        candidates=candidates,
        raw_scores_by_index=raw_scores_by_index,
    )
    best_constant = min(
        constant_eligible,
        key=lambda index: (
            constant_raw_rank_by_index[index],
            -_required_raw_score(raw_scores_by_index[index]),
            candidates[index].fingerprint_sha256,
        ),
    )

    positive_structures: list[
        tuple[int, float, float, str, float, TempoTrackRawRunDiagnostic | None]
    ] = []
    for index, candidate in enumerate(candidates):
        if candidate.curve_class != "jump":
            continue
        source = sources[index] if source_aware else ""
        if source == _BOUNDARY_NOMINAL_BACKBONE_SOURCE:
            continue
        if source == _SHORT_WINDOW_MONOTONE_CHAIN_SOURCE:
            if not _is_short_window_monotone_chain_jump(candidate):
                continue
        elif not _is_phase1_piecewise_constant_jump(candidate):
            continue
        raw_score = raw_scores_by_index.get(index)
        collapsed_score = collapsed_scores_by_index.get(index)
        if not (
            _raw_self_score_is_production_usable(raw_score, config=config)
            and _raw_self_score_is_production_usable(collapsed_score, config=config)
        ):
            continue
        assert raw_score is not None
        assert collapsed_score is not None
        raw_gain = _required_raw_score(raw_score) - _required_raw_score(collapsed_score)
        if source_aware:
            if not _source_has_beatthis_boundary_anchor(source):
                continue
            if source == _SHORT_WINDOW_MONOTONE_CHAIN_SOURCE:
                compatible_run = raw_run
                beatthis_delta = _generalized_support_delta(
                    beat_signal,
                    frame_rate_hz=frame_rate_hz,
                    candidate=candidate,
                    collapsed=collapsed_counterfactuals[index],
                    radius_ms=config.beat_event_radius_ms,
                )
                if (
                    not math.isfinite(beatthis_delta)
                    or beatthis_delta <= 0.0
                    or raw_gain <= 0.0
                ):
                    continue
            elif source == _EXP026_SIDE_SEGMENT_SOURCE:
                if (
                    len(candidate.sections) == 2
                    and not _observation_side_two_section_has_minimum_first_duration_ratio(
                        candidate
                    )
                ):
                    continue
                if (
                    len(candidate.sections) == 2
                    and not _observation_side_single_boundary_has_cross_source_support(
                        candidate,
                        observations=observations,
                        config=config,
                    )
                ):
                    continue
                if (
                    len(candidate.sections) == 3
                    and not _observation_side_closed_aba_has_beatthis_middle_support(
                        candidate,
                        observations=observations,
                        config=config,
                    )
                ):
                    continue
                if (
                    len(candidate.sections) == 3
                    and not _observation_side_closed_aba_has_cross_source_right_support(
                        candidate,
                        observations=observations,
                        config=config,
                    )
                ):
                    continue
                if (
                    len(candidate.sections) == 3
                    and raw_gain < _EXP026_CLOSED_ABA_MINIMUM_RAW_GAIN
                ):
                    continue
                compatible_run = raw_run
                beatthis_delta = _generalized_support_delta(
                    beat_signal,
                    frame_rate_hz=frame_rate_hz,
                    candidate=candidate,
                    collapsed=collapsed_counterfactuals[index],
                    radius_ms=config.beat_event_radius_ms,
                )
                negative_raw_octave_inheritance = (
                    len(candidate.sections) == 2
                    and raw_gain <= 0.0
                    and _has_raw_positive_octave_equivalent_side_sibling(
                        index,
                        candidates=candidates,
                        sources=sources,
                        raw_scores_by_index=raw_scores_by_index,
                        collapsed_scores_by_index=collapsed_scores_by_index,
                    )
                    and beatthis_delta >= 2.0 * abs(raw_gain)
                )
                if (
                    not math.isfinite(beatthis_delta)
                    or beatthis_delta <= 0.0
                    or (raw_gain <= 0.0 and not negative_raw_octave_inheritance)
                ):
                    continue
            else:
                if raw_gain <= 0.0:
                    continue
                compatibility = _raw_run_compatibility(
                    candidate,
                    raw_runs=raw_runs,
                    raw_observations=raw_observations,
                    primary=primary,
                    beat_signal=beat_signal,
                    frame_rate_hz=frame_rate_hz,
                    collapsed=collapsed_counterfactuals[index],
                    config=config,
                )
                if compatibility is None:
                    continue
                compatible_run, beatthis_delta = compatibility
            positive_structures.append(
                (
                    index,
                    float(raw_gain),
                    _required_raw_score(collapsed_score),
                    collapsed_counterfactuals[index].fingerprint_sha256,
                    float(beatthis_delta),
                    compatible_run,
                )
            )
            continue
        if raw_gain <= 0.0:
            continue
        compatibility = _raw_run_compatibility(
            candidate,
            raw_runs=raw_runs,
            raw_observations=raw_observations,
            primary=primary,
            beat_signal=beat_signal,
            frame_rate_hz=frame_rate_hz,
            collapsed=collapsed_counterfactuals[index],
            config=config,
        )
        if compatibility is None:
            continue
        compatible_run, beatthis_delta = compatibility
        positive_structures.append(
            (
                index,
                float(raw_gain),
                _required_raw_score(collapsed_score),
                collapsed_counterfactuals[index].fingerprint_sha256,
                beatthis_delta,
                compatible_run,
            )
        )

    if not positive_structures:
        return TempoTrackProductionSelection(
            status="v3_accepted",
            selected_candidate_index=best_constant,
            selected_fingerprint_sha256=candidates[best_constant].fingerprint_sha256,
            lane="constant",
            fallback_reason=None,
            raw_run=raw_run,
            eligible_candidate_indices=constant_eligible,
            raw_self_rank_by_candidate=_rank_items(constant_raw_rank_by_index),
            beatthis_aba_rank_by_candidate=(),
            paired_raw_gain_by_candidate=paired_gain_diagnostics,
        )

    eligible = tuple(index for index, _, _, _, _, _ in positive_structures)
    positive_structure_by_index = {
        value[0]: value for value in positive_structures
    }
    borda_buckets = _exp026_two_section_octave_borda_buckets(
        eligible,
        candidates=candidates,
        sources=sources,
        raw_scores_by_index=raw_scores_by_index,
        config=config,
    )
    borda_structures = tuple(
        (
            representative,
            max(positive_structure_by_index[index][1] for index in members),
            max(positive_structure_by_index[index][4] for index in members),
            max(generation_scores[index] for index in members),
        )
        for representative, members in borda_buckets
    )
    beatthis_aba_rank_by_index = {
        index: rank
        for rank, (index, _, beatthis_delta, _) in enumerate(
            sorted(
                borda_structures,
                key=lambda value: (-value[2], candidates[value[0]].fingerprint_sha256),
            ),
            start=1,
        )
    }
    paired_gain_rank_by_index = {
        index: rank
        for rank, (index, raw_gain, _, _) in enumerate(
            sorted(
                borda_structures,
                key=lambda value: (-value[1], candidates[value[0]].fingerprint_sha256),
            ),
            start=1,
        )
    }
    generation_rank_by_index: dict[int, int] = {}
    for source in sorted({sources[index] for index, _, _, _ in borda_structures}):
        source_structures = tuple(
            value for value in borda_structures if sources[value[0]] == source
        )
        for rank, (index, _, _, generation_score) in enumerate(
            sorted(
                source_structures,
                key=lambda value: (
                    -value[3],
                    candidates[value[0]].fingerprint_sha256,
                ),
            ),
            start=1,
        ):
            generation_rank_by_index[index] = rank
    compatible_run_by_index = {
        index: compatible_run
        for index, _, _, _, _, compatible_run in positive_structures
    }
    lane_raw_rank_by_index = _lane_raw_self_rank_by_index(
        eligible,
        candidates=candidates,
        raw_scores_by_index=raw_scores_by_index,
    )
    selected = min(
        tuple(representative for representative, _ in borda_buckets),
        key=lambda index: (
            paired_gain_rank_by_index[index]
            + beatthis_aba_rank_by_index[index]
            + generation_rank_by_index[index],
            generation_rank_by_index[index],
            paired_gain_rank_by_index[index],
            beatthis_aba_rank_by_index[index],
            _piecewise_constant_section_count(candidates[index]),
            -_required_raw_score(raw_scores_by_index[index]),
            candidates[index].fingerprint_sha256,
        ),
    )
    selected = _prefer_observed_pair_over_virtual_right(
        selected,
        borda_buckets=borda_buckets,
        candidates=candidates,
        sources=sources,
        generation_scores=generation_scores,
        positive_structure_by_index=positive_structure_by_index,
    )
    return TempoTrackProductionSelection(
        status="v3_accepted",
        selected_candidate_index=selected,
        selected_fingerprint_sha256=candidates[selected].fingerprint_sha256,
        lane="paired_jump",
        fallback_reason=None,
        raw_run=compatible_run_by_index[selected],
        eligible_candidate_indices=eligible,
        raw_self_rank_by_candidate=_rank_items(lane_raw_rank_by_index),
        beatthis_aba_rank_by_candidate=_rank_items(beatthis_aba_rank_by_index),
        paired_raw_gain_by_candidate=paired_gain_diagnostics,
    )


def _prefer_observed_pair_over_virtual_right(
    selected: int,
    *,
    borda_buckets: tuple[tuple[int, tuple[int, ...]], ...],
    candidates: tuple[PhaseContinuousTimingCurve, ...],
    sources: tuple[str, ...],
    generation_scores: tuple[float, ...],
    positive_structure_by_index: dict[
        int,
        tuple[int, float, float, str, float, TempoTrackRawRunDiagnostic | None],
    ],
) -> int:
    """Prefer two observed boundaries when raw fit and proposal evidence agree."""

    if not sources[selected].startswith("virtual_right_"):
        return selected
    selected_curve = candidates[selected]
    if not _is_short_closed_aba(selected_curve):
        return selected
    selected_raw_gain = positive_structure_by_index[selected][1]
    selected_generation_score = generation_scores[selected]
    challengers = tuple(
        representative
        for representative, _members in borda_buckets
        if sources[representative] == "paired_unmerged_boundary"
        and _is_short_closed_aba(candidates[representative])
        and positive_structure_by_index[representative][1] >= selected_raw_gain
        and generation_scores[representative] >= selected_generation_score
    )
    if not challengers:
        return selected
    return max(
        challengers,
        key=lambda index: (
            positive_structure_by_index[index][1],
            generation_scores[index],
            positive_structure_by_index[index][4],
            candidates[index].fingerprint_sha256,
        ),
    )


def _is_short_closed_aba(candidate: PhaseContinuousTimingCurve) -> bool:
    if candidate.curve_class != "jump" or len(candidate.sections) != 3:
        return False
    first, middle, last = candidate.sections
    if not all(
        isinstance(section, ConstantTempoSection)
        for section in (first, middle, last)
    ):
        return False
    assert isinstance(first, ConstantTempoSection)
    assert isinstance(middle, ConstantTempoSection)
    assert isinstance(last, ConstantTempoSection)
    return (
        _EXP017_SHORT_ABA_MIN_SECONDS
        <= middle.duration_seconds
        <= _EXP017_SHORT_ABA_MAX_SECONDS
        and abs(first.bpm - last.bpm) <= max(1.0, 0.01 * first.bpm)
    )


def _fallback_selection(
    reason: str,
    *,
    raw_run: TempoTrackRawRunDiagnostic | None = None,
) -> TempoTrackProductionSelection:
    return TempoTrackProductionSelection(
        status="v2_fallback",
        selected_candidate_index=None,
        selected_fingerprint_sha256=None,
        lane="fallback",
        fallback_reason=reason,
        raw_run=raw_run,
        eligible_candidate_indices=(),
        raw_self_rank_by_candidate=(),
        beatthis_aba_rank_by_candidate=(),
    )


def _rank_items(values: dict[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(values.items(), key=lambda value: (value[1], value[0])))


def _lane_raw_self_rank_by_index(
    indices: tuple[int, ...],
    *,
    candidates: tuple[PhaseContinuousTimingCurve, ...],
    raw_scores_by_index: dict[int, CandidateRawAudioScore],
) -> dict[int, int]:
    return {
        index: rank
        for rank, index in enumerate(
            sorted(
                indices,
                key=lambda index: (
                    -_required_raw_score(raw_scores_by_index[index]),
                    candidates[index].fingerprint_sha256,
                ),
            ),
            start=1,
        )
    }


def _exp026_two_section_octave_borda_buckets(
    eligible: tuple[int, ...],
    *,
    candidates: tuple[PhaseContinuousTimingCurve, ...],
    sources: tuple[str, ...],
    raw_scores_by_index: dict[int, CandidateRawAudioScore],
    config: TempoTrackConfig,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    remaining = set(eligible)
    buckets: list[tuple[int, tuple[int, ...]]] = []
    while remaining:
        seed = min(remaining)
        members = {seed}
        frontier = [seed]
        remaining.remove(seed)
        while frontier:
            current = frontier.pop()
            peers = tuple(
                index
                for index in sorted(remaining)
                if sources[current] == _EXP026_SIDE_SEGMENT_SOURCE
                and sources[index] == _EXP026_SIDE_SEGMENT_SOURCE
                and _exp026_two_section_octave_equivalent(
                    candidates[current],
                    candidates[index],
                )
            )
            for index in peers:
                remaining.remove(index)
                members.add(index)
                frontier.append(index)
        ordered_members = tuple(sorted(members))
        representative = min(
            ordered_members,
            key=lambda index: (
                -_required_raw_score(raw_scores_by_index[index]),
                candidates[index].fingerprint_sha256,
            ),
        )
        buckets.append((representative, ordered_members))
    return tuple(sorted(buckets, key=lambda value: value[1]))


def _has_raw_positive_octave_equivalent_side_sibling(
    candidate_index: int,
    *,
    candidates: tuple[PhaseContinuousTimingCurve, ...],
    sources: tuple[str, ...],
    raw_scores_by_index: dict[int, CandidateRawAudioScore],
    collapsed_scores_by_index: dict[int, CandidateRawAudioScore],
) -> bool:
    candidate = candidates[candidate_index]
    for sibling_index, sibling in enumerate(candidates):
        if sibling_index == candidate_index:
            continue
        if not (
            sources[sibling_index] == _EXP026_SIDE_SEGMENT_SOURCE
            and _exp026_two_section_octave_equivalent(candidate, sibling)
        ):
            continue
        raw_score = raw_scores_by_index.get(sibling_index)
        collapsed_score = collapsed_scores_by_index.get(sibling_index)
        if raw_score is None or collapsed_score is None:
            continue
        if not (raw_score.available and collapsed_score.available):
            continue
        if _required_raw_score(raw_score) > _required_raw_score(collapsed_score):
            return True
    return False


def _exp026_two_section_octave_equivalent(
    left: PhaseContinuousTimingCurve,
    right: PhaseContinuousTimingCurve,
) -> bool:
    if len(left.sections) != 2 or len(right.sections) != 2:
        return False
    if not all(
        isinstance(section, ConstantTempoSection)
        for section in left.sections + right.sections
    ):
        return False
    left_sections = tuple(left.sections)
    right_sections = tuple(right.sections)
    left_boundary_ms = left.time_at_beat(float(left_sections[1].start_beat))
    right_boundary_ms = right.time_at_beat(float(right_sections[1].start_beat))
    if abs(left_boundary_ms - right_boundary_ms) > _EXP026_BOUNDARY_TOLERANCE_MS:
        return False
    ratios = tuple(
        left_section.bpm / right_section.bpm
        for left_section, right_section in zip(left_sections, right_sections)
    )
    return any(
        all(abs(ratio - scale) <= 0.025 * scale for ratio in ratios)
        for scale in (0.5, 2.0)
    )


def _raw_self_score_is_production_usable(
    score: CandidateRawAudioScore | None,
    *,
    config: TempoTrackConfig,
) -> bool:
    if score is None or not score.available or score.raw_score is None:
        return False
    if score.retained_beat_count < config.raw_self_minimum_retained_beats:
        return False
    if score.retained_domain_ratio is None:
        return False
    return score.retained_domain_ratio >= config.raw_self_minimum_retained_domain_ratio


def _required_raw_score(score: CandidateRawAudioScore) -> float:
    if score.raw_score is None:
        raise ValueError("production candidate requires an available raw self-score")
    return score.raw_score


def _paired_raw_gain_diagnostics(
    candidates: tuple[PhaseContinuousTimingCurve, ...],
    *,
    raw_scores_by_index: dict[int, CandidateRawAudioScore],
    collapsed_scores_by_index: dict[int, CandidateRawAudioScore],
    collapsed_counterfactuals: tuple[PhaseContinuousTimingCurve, ...],
) -> tuple[tuple[int, float, float, str], ...]:
    values: list[tuple[int, float, float, str]] = []
    for index, candidate in enumerate(candidates):
        if candidate.curve_class == "constant":
            continue
        raw_score = raw_scores_by_index.get(index)
        collapsed_score = collapsed_scores_by_index.get(index)
        if (
            raw_score is None
            or collapsed_score is None
            or raw_score.raw_score is None
            or collapsed_score.raw_score is None
        ):
            continue
        values.append(
            (
                index,
                raw_score.raw_score - collapsed_score.raw_score,
                collapsed_score.raw_score,
                collapsed_counterfactuals[index].fingerprint_sha256,
            )
        )
    return tuple(sorted(values, key=lambda value: (value[0], value[3])))


def _raw_gain_sign(raw_gain: float) -> str:
    if raw_gain > 0.0:
        return "positive"
    if raw_gain < 0.0:
        return "negative"
    return "zero"


def _source_has_beatthis_boundary_anchor(source: str) -> bool:
    if source.startswith("raw_run_"):
        return False
    return source in {
        "paired_unmerged_boundary",
        "virtual_right_beatthis",
        "virtual_right_raw_audio",
        _EXP026_SIDE_SEGMENT_SOURCE,
        _SHORT_WINDOW_MONOTONE_CHAIN_SOURCE,
    }


def _observation_side_two_section_has_minimum_first_duration_ratio(
    candidate: PhaseContinuousTimingCurve,
) -> bool:
    if len(candidate.sections) != 2:
        return True
    if not all(
        isinstance(section, ConstantTempoSection) for section in candidate.sections
    ):
        return False
    duration_seconds = (candidate.end_time_ms - candidate.start_time_ms) / 1000.0
    if duration_seconds <= 0.0:
        return False
    first_section = candidate.sections[0]
    return (
        first_section.duration_seconds / duration_seconds
        >= _EXP026_TWO_SECTION_MINIMUM_FIRST_DURATION_RATIO
    )


def _observation_side_single_boundary_has_cross_source_support(
    candidate: PhaseContinuousTimingCurve,
    *,
    observations: tuple[LocalTempoObservation, ...],
    config: TempoTrackConfig,
) -> bool:
    if len(candidate.sections) != 2:
        return True
    left_section, right_section = candidate.sections
    if not (
        isinstance(left_section, ConstantTempoSection)
        and isinstance(right_section, ConstantTempoSection)
    ):
        return False

    boundary_time_ms = candidate.time_at_beat(float(right_section.start_beat))
    shoulder_ms = 3.0 * config.local_window_seconds * 1000.0
    left_observations = tuple(
        observation
        for observation in observations
        if boundary_time_ms - shoulder_ms <= observation.center_time_ms
        and observation.window_end_ms <= boundary_time_ms
    )
    right_observations = tuple(
        observation
        for observation in observations
        if observation.window_start_ms >= boundary_time_ms
        and observation.center_time_ms <= boundary_time_ms + shoulder_ms
    )
    if not left_observations:
        left_observations = tuple(
            observation
            for observation in observations
            if boundary_time_ms - shoulder_ms
            <= observation.center_time_ms
            < boundary_time_ms
        )
    if not right_observations:
        right_observations = tuple(
            observation
            for observation in observations
            if boundary_time_ms
            < observation.center_time_ms
            <= boundary_time_ms + shoulder_ms
        )

    return all(
        _regional_source_pool_supports_bpm(
            regional_observations,
            source=source,
            bpm=section.bpm,
            config=config,
        )
        for regional_observations, section in (
            (left_observations, left_section),
            (right_observations, right_section),
        )
        for source in ("beatthis", "raw_audio")
    )


def _observation_side_closed_aba_has_beatthis_middle_support(
    candidate: PhaseContinuousTimingCurve,
    *,
    observations: tuple[LocalTempoObservation, ...],
    config: TempoTrackConfig,
) -> bool:
    if len(candidate.sections) != 3:
        return True
    left_section, middle_section, right_section = candidate.sections
    if not all(
        isinstance(section, ConstantTempoSection)
        for section in (left_section, middle_section, right_section)
    ):
        return False
    outer_tolerance_bpm = max(
        1.0,
        0.01 * max(left_section.bpm, right_section.bpm),
    )
    if abs(left_section.bpm - right_section.bpm) > outer_tolerance_bpm:
        return True

    middle_start_ms = candidate.time_at_beat(float(middle_section.start_beat))
    middle_end_ms = candidate.time_at_beat(float(middle_section.end_beat))
    middle_observations = tuple(
        observation
        for observation in observations
        if observation.window_start_ms >= middle_start_ms
        and observation.window_end_ms <= middle_end_ms
    )
    if not middle_observations:
        middle_observations = tuple(
            observation
            for observation in observations
            if middle_start_ms < observation.center_time_ms < middle_end_ms
        )
    return _regional_source_pool_supports_bpm(
        middle_observations,
        source="beatthis",
        bpm=middle_section.bpm,
        config=config,
    )


def _observation_side_closed_aba_has_cross_source_right_support(
    candidate: PhaseContinuousTimingCurve,
    *,
    observations: tuple[LocalTempoObservation, ...],
    config: TempoTrackConfig,
) -> bool:
    if len(candidate.sections) != 3:
        return True
    left_section, middle_section, right_section = candidate.sections
    if not all(
        isinstance(section, ConstantTempoSection)
        for section in (left_section, middle_section, right_section)
    ):
        return False
    outer_tolerance_bpm = max(
        1.0,
        0.01 * max(left_section.bpm, right_section.bpm),
    )
    if abs(left_section.bpm - right_section.bpm) > outer_tolerance_bpm:
        return True

    right_start_ms = candidate.time_at_beat(float(middle_section.end_beat))
    right_end_ms = candidate.end_time_ms
    right_observations = tuple(
        observation
        for observation in observations
        if observation.window_start_ms >= right_start_ms
        and observation.window_end_ms <= right_end_ms
    )
    if not right_observations:
        right_observations = tuple(
            observation
            for observation in observations
            if right_start_ms < observation.center_time_ms < right_end_ms
        )
    return all(
        _regional_source_pool_supports_bpm(
            right_observations,
            source=source,
            bpm=right_section.bpm,
            config=config,
        )
        for source in ("beatthis", "raw_audio")
    )


def _regional_source_pool_supports_bpm(
    observations: tuple[LocalTempoObservation, ...],
    *,
    source: Literal["beatthis", "raw_audio"],
    bpm: float,
    config: TempoTrackConfig,
) -> bool:
    modes = _regional_tempo_pool(
        tuple(
            observation
            for observation in observations
            if observation.source == source
        ),
        base=None,
        include_base=False,
        fallback_base=False,
        config=config,
    )
    tolerance_bpm = max(1.0, 0.01 * bpm)
    return any(abs(mode.bpm - bpm) <= tolerance_bpm for mode in modes)


def _score_collapsed_counterfactuals_independent(
    evidence: RawAudioEvidence,
    counterfactuals: tuple[PhaseContinuousTimingCurve, ...],
) -> RawAudioEvidenceRanking:
    """Score collapsed constants without requiring unique comparator fingerprints."""

    scores: list[CandidateRawAudioScore] = []
    for candidate_index, counterfactual in enumerate(counterfactuals):
        single = score_raw_audio_evidence_independent(evidence, (counterfactual,))
        score = single.candidate_scores[0]
        scores.append(
            CandidateRawAudioScore(
                candidate_index=candidate_index,
                fingerprint_sha256=score.fingerprint_sha256,
                raw_score=score.raw_score,
                mean_beat_support=score.mean_beat_support,
                mean_half_beat_support=score.mean_half_beat_support,
                window_contrast_p10=score.window_contrast_p10,
                retained_beat_count=score.retained_beat_count,
                complete_window_count=score.complete_window_count,
                unavailable_reason=score.unavailable_reason,
                candidate_domain_beat_count=score.candidate_domain_beat_count,
                complete_window_start_beats=score.complete_window_start_beats,
            )
        )
    ranked = tuple(
        sorted(
            (score for score in scores if score.available),
            key=lambda score: (
                -_required_raw_score(score),
                score.fingerprint_sha256,
                score.candidate_index,
            ),
        )
    )
    return RawAudioEvidenceRanking(
        evidence=evidence,
        candidate_scores=tuple(scores),
        ranked_scores=ranked,
        common_beat_indices=(),
        complete_window_start_beats=(),
        unavailable_reason=None
        if ranked
        else next((score.unavailable_reason for score in scores), None),
    )


def _candidates_share_beat_domain(
    candidates: tuple[PhaseContinuousTimingCurve, ...],
) -> bool:
    if not candidates:
        return False
    first = candidates[0]
    return all(
        candidate.start_beat == first.start_beat and candidate.end_beat == first.end_beat
        for candidate in candidates
    )


def _collapsed_constant_counterfactual(
    candidate: PhaseContinuousTimingCurve,
    *,
    duration_ms: float,
    base_bpm: float,
    config: TempoTrackConfig,
) -> PhaseContinuousTimingCurve:
    return PhaseContinuousTimingCurve(
        origin_beat=candidate.origin_beat,
        origin_time_ms=candidate.origin_time_ms,
        sections=(
            ConstantTempoSection(
                start_beat=candidate.origin_beat,
                end_beat=_terminal_beat_for_constant_tempo(
                    origin_beat=candidate.origin_beat,
                    origin_time_ms=candidate.origin_time_ms,
                    duration_ms=duration_ms,
                    bpm=base_bpm,
                ),
                bpm=base_bpm,
            ),
        ),
    )


def _collapsed_counterfactuals_for_proposals(
    proposals: tuple[_CurveProposal, ...],
    *,
    duration_ms: float,
    config: TempoTrackConfig,
) -> tuple[PhaseContinuousTimingCurve, ...]:
    return tuple(
        _collapsed_constant_counterfactual(
            proposal.curve,
            duration_ms=duration_ms,
            base_bpm=proposal.collapse_bpm,
            config=config,
        )
        for proposal in proposals
    )


def _terminal_beat_for_constant_tempo(
    *,
    origin_beat: int,
    origin_time_ms: float,
    duration_ms: float,
    bpm: float,
) -> int:
    available_seconds = max(0.0, (duration_ms - origin_time_ms) / 1000.0)
    beat_offset = int(math.ceil(max(0.0, available_seconds * bpm / 60.0) - 1e-9))
    return origin_beat + max(1, beat_offset)


def _is_phase1_piecewise_constant_jump(candidate: PhaseContinuousTimingCurve) -> bool:
    return (
        candidate.curve_class == "jump"
        and 2 <= len(candidate.sections) <= 4
        and all(isinstance(section, ConstantTempoSection) for section in candidate.sections)
        and all(
            report.phase_discontinuity_ms <= 5.0
            for report in candidate.seam_reports
        )
    )


def _is_short_window_monotone_chain_jump(
    candidate: PhaseContinuousTimingCurve,
) -> bool:
    if not (
        candidate.curve_class == "jump"
        and _SHORT_CHAIN_MINIMUM_SECTIONS
        <= len(candidate.sections)
        <= _SHORT_CHAIN_MAXIMUM_SECTIONS
        and all(
            isinstance(section, ConstantTempoSection)
            for section in candidate.sections
        )
        and all(
            report.phase_discontinuity_ms <= 5.0
            for report in candidate.seam_reports
        )
    ):
        return False
    bpms = tuple(float(section.start_bpm) for section in candidate.sections)
    deltas = tuple(right - left for left, right in zip(bpms, bpms[1:]))
    strictly_monotone = all(delta > 0.0 for delta in deltas) or all(
        delta < 0.0 for delta in deltas
    )
    return (
        strictly_monotone
        and abs(bpms[-1] - bpms[0]) >= _SHORT_CHAIN_MINIMUM_TOTAL_DELTA_BPM
        and all(
            section.end_beat - section.start_beat >= _SHORT_CHAIN_SECTION_BEATS
            for section in candidate.sections[1:-1]
        )
    )


def _piecewise_constant_section_count(candidate: PhaseContinuousTimingCurve) -> int:
    if not all(isinstance(section, ConstantTempoSection) for section in candidate.sections):
        return 1_000_000
    return len(candidate.sections)


def _raw_run_compatibility(
    candidate: PhaseContinuousTimingCurve,
    *,
    raw_runs: tuple[TempoTrackRawRunDiagnostic, ...],
    raw_observations: tuple[LocalTempoObservation, ...],
    primary: _BaseHypothesis,
    beat_signal: NDArray[np.float64],
    frame_rate_hz: float,
    collapsed: PhaseContinuousTimingCurve,
    config: TempoTrackConfig,
) -> tuple[TempoTrackRawRunDiagnostic, float] | None:
    if not raw_runs or not _is_phase1_piecewise_constant_jump(candidate):
        return None

    section_matches: list[tuple[TempoTrackRawRunDiagnostic, float]] = []
    audio_duration_ms = 1000.0 * beat_signal.size / frame_rate_hz
    for section_index, section in enumerate(candidate.sections):
        assert isinstance(section, ConstantTempoSection)
        if _is_primary_alias_consistent(section.bpm, primary.bpm, config=config):
            continue
        matches = tuple(
            run
            for run in raw_runs
            if _section_matches_raw_run(
                candidate,
                section,
                section_index=section_index,
                raw_run=run,
                raw_observations=raw_observations,
                primary=primary,
                audio_duration_ms=audio_duration_ms,
                config=config,
            )
        )
        if not matches:
            return None
        best = min(
            matches,
            key=lambda run: (
                _primary_canonical_log_distance(
                    section.bpm,
                    run.median_bpm,
                    primary_bpm=primary.bpm,
                    config=config,
                ),
                -_interval_overlap_ms(
                    candidate.time_at_beat(float(section.start_beat)),
                    candidate.time_at_beat(float(section.end_beat)),
                    run.expanded_start_time_ms,
                    run.expanded_end_time_ms,
                ),
                run.start_time_ms,
            ),
        )
        section_matches.append((best, section.bpm))

    if not section_matches:
        return None

    beatthis_delta = _generalized_support_delta(
        beat_signal,
        frame_rate_hz=frame_rate_hz,
        candidate=candidate,
        collapsed=collapsed,
        radius_ms=config.beat_event_radius_ms,
    )
    selected_run = min(
        (run for run, _ in section_matches),
        key=lambda run: (
            -abs(run.weighted_median_delta_bpm),
            run.start_time_ms,
        ),
    )
    return selected_run, beatthis_delta


def _section_matches_raw_run(
    candidate: PhaseContinuousTimingCurve,
    section: ConstantTempoSection,
    *,
    section_index: int,
    raw_run: TempoTrackRawRunDiagnostic,
    raw_observations: tuple[LocalTempoObservation, ...],
    primary: _BaseHypothesis,
    audio_duration_ms: float,
    config: TempoTrackConfig,
) -> bool:
    section_delta = section.bpm - primary.bpm
    if raw_run.direction == "up" and section_delta <= 0.0:
        return False
    if raw_run.direction == "down" and section_delta >= 0.0:
        return False
    if _primary_canonical_log_distance(
        section.bpm,
        raw_run.median_bpm,
        primary_bpm=primary.bpm,
        config=config,
    ) > math.log(config.raw_run_jump_max_tempo_ratio):
        return False

    section_start_ms = candidate.time_at_beat(float(section.start_beat))
    section_end_ms = candidate.time_at_beat(float(section.end_beat))
    if (
        _interval_overlap_ms(
            section_start_ms,
            section_end_ms,
            raw_run.expanded_start_time_ms,
            raw_run.expanded_end_time_ms,
        )
        < config.raw_run_jump_minimum_overlap_ms
    ):
        return False

    anchor_tolerance_ms = max(
        config.raw_run_expansion_ms + 0.5 * 60000.0 / primary.bpm,
        1000.0,
    )
    edge_tolerance_ms = max(
        anchor_tolerance_ms,
        0.5 * config.local_window_seconds * 1000.0 + config.raw_run_expansion_ms,
    )
    scoreable_start_ms = max(0.0, candidate.start_time_ms)
    scoreable_end_ms = min(audio_duration_ms, candidate.end_time_ms)
    touches_candidate_start = section.start_beat == candidate.start_beat
    touches_candidate_end = section.end_beat == candidate.end_beat
    previous_section = (
        candidate.sections[section_index - 1] if section_index > 0 else None
    )
    next_section = (
        candidate.sections[section_index + 1]
        if section_index + 1 < len(candidate.sections)
        else None
    )
    persistent_a_to_b_tail = (
        touches_candidate_end
        and not touches_candidate_start
        and section_index == len(candidate.sections) - 1
        and isinstance(previous_section, ConstantTempoSection)
        and _is_primary_alias_consistent(
            previous_section.bpm,
            primary.bpm,
            config=config,
        )
    )
    persistent_b_to_a_head = (
        touches_candidate_start
        and not touches_candidate_end
        and section_index == 0
        and isinstance(next_section, ConstantTempoSection)
        and _is_primary_alias_consistent(
            next_section.bpm,
            primary.bpm,
            config=config,
        )
    )
    interior_anchor_tolerance_ms = (
        edge_tolerance_ms
        if _is_exp018_closed_aba_middle_section(
            candidate,
            section_index=section_index,
            primary=primary,
            config=config,
        )
        else anchor_tolerance_ms
    )

    if touches_candidate_start:
        if raw_run.start_time_ms > scoreable_start_ms + edge_tolerance_ms:
            return False
    elif abs(section_start_ms - raw_run.start_time_ms) > interior_anchor_tolerance_ms:
        return False

    if persistent_b_to_a_head and _has_primary_consistent_observation_run(
        raw_observations,
        primary=primary,
        before_time_ms=raw_run.start_time_ms,
        config=config,
    ):
        return False

    if persistent_a_to_b_tail:
        if _has_primary_consistent_observation_run(
            raw_observations,
            primary=primary,
            after_time_ms=raw_run.end_time_ms,
            config=config,
        ):
            return False
        return True

    if touches_candidate_end:
        if raw_run.end_time_ms < scoreable_end_ms - edge_tolerance_ms:
            return False
    elif abs(section_end_ms - raw_run.end_time_ms) > interior_anchor_tolerance_ms:
        return False

    return True


def _is_exp018_closed_aba_middle_section(
    candidate: PhaseContinuousTimingCurve,
    *,
    section_index: int,
    primary: _BaseHypothesis,
    config: TempoTrackConfig,
) -> bool:
    if section_index != 1 or len(candidate.sections) != 3:
        return False
    first, middle, last = candidate.sections
    if not (
        isinstance(first, ConstantTempoSection)
        and isinstance(middle, ConstantTempoSection)
        and isinstance(last, ConstantTempoSection)
    ):
        return False
    if (
        middle.start_beat == candidate.start_beat
        or middle.end_beat == candidate.end_beat
    ):
        return False
    if not (
        _EXP017_SHORT_ABA_MIN_SECONDS
        <= middle.duration_seconds
        <= _EXP017_SHORT_ABA_MAX_SECONDS
    ):
        return False
    if _is_primary_alias_consistent(middle.bpm, primary.bpm, config=config):
        return False
    if not (
        _is_primary_alias_consistent(first.bpm, primary.bpm, config=config)
        and _is_primary_alias_consistent(last.bpm, primary.bpm, config=config)
    ):
        return False
    if first.bpm != last.bpm:
        return False
    return all(
        report.phase_discontinuity_ms == 0.0
        for report in candidate.seam_reports
    )


def _has_primary_consistent_observation_run(
    observations: tuple[LocalTempoObservation, ...],
    *,
    primary: _BaseHypothesis,
    config: TempoTrackConfig,
    before_time_ms: float | None = None,
    after_time_ms: float | None = None,
) -> bool:
    if before_time_ms is not None and after_time_ms is not None:
        raise ValueError("primary observation run filter must be one-sided")
    retained = []
    for observation in sorted(observations, key=lambda value: value.center_time_ms):
        if observation.strength < config.minimum_local_strength:
            continue
        if before_time_ms is not None and observation.center_time_ms >= before_time_ms:
            continue
        if after_time_ms is not None and observation.center_time_ms <= after_time_ms:
            continue
        if _is_primary_alias_consistent(
            observation.bpm,
            primary.bpm,
            config=config,
        ):
            retained.append(observation)

    max_gap_ms = 1000.0 * config.local_hop_seconds * config.raw_run_gap_hop_multiplier
    run_length = 0
    previous_time_ms: float | None = None
    for observation in retained:
        if (
            previous_time_ms is not None
            and observation.center_time_ms - previous_time_ms > max_gap_ms
        ):
            run_length = 0
        run_length += 1
        previous_time_ms = observation.center_time_ms
        if run_length >= config.raw_run_minimum_observations:
            return True
    return False


def _generalized_support_delta(
    signal: NDArray[np.float64],
    *,
    frame_rate_hz: float,
    candidate: PhaseContinuousTimingCurve,
    collapsed: PhaseContinuousTimingCurve,
    radius_ms: float,
) -> float:
    duration_ms = 1000.0 * signal.size / frame_rate_hz
    return _physical_curve_support_score(
        signal,
        frame_rate_hz=frame_rate_hz,
        curve=candidate,
        duration_ms=duration_ms,
        radius_ms=radius_ms,
    ) - _physical_curve_support_score(
        signal,
        frame_rate_hz=frame_rate_hz,
        curve=collapsed,
        duration_ms=duration_ms,
        radius_ms=radius_ms,
    )


def _physical_curve_support_score(
    signal: NDArray[np.float64],
    *,
    frame_rate_hz: float,
    curve: PhaseContinuousTimingCurve,
    duration_ms: float,
    radius_ms: float,
) -> float:
    if all(isinstance(section, ConstantTempoSection) for section in curve.sections):
        beat_parts: list[NDArray[np.float64]] = []
        half_parts: list[NDArray[np.float64]] = []
        for section, section_start_time_ms in zip(
            curve.sections,
            curve.section_start_times_ms,
            strict=True,
        ):
            assert isinstance(section, ConstantTempoSection)
            local_beats = np.arange(
                section.end_beat - section.start_beat,
                dtype=np.float64,
            )
            beat_parts.append(
                section_start_time_ms
                + 1000.0 * (60.0 * local_beats / section.bpm)
            )
            half_parts.append(
                section_start_time_ms
                + 1000.0 * (60.0 * (local_beats + 0.5) / section.bpm)
            )
        beat_times_array = np.concatenate(beat_parts)
        half_times_array = np.concatenate(half_parts)
        beat_times_array = beat_times_array[
            (beat_times_array >= 0.0) & (beat_times_array < duration_ms)
        ]
        half_times_array = half_times_array[
            (half_times_array >= 0.0) & (half_times_array < duration_ms)
        ]
    else:
        beat_times = [
            curve.time_at_beat(float(beat))
            for beat in range(curve.start_beat, curve.end_beat)
        ]
        beat_times_array = np.asarray(
            [time_ms for time_ms in beat_times if 0.0 <= time_ms < duration_ms],
            dtype=np.float64,
        )
        half_times = []
        for beat in range(curve.start_beat, curve.end_beat):
            half_beat = float(beat) + 0.5
            if half_beat > curve.end_beat:
                continue
            time_ms = curve.time_at_beat(half_beat)
            if 0.0 <= time_ms < duration_ms:
                half_times.append(time_ms)
        half_times_array = np.asarray(half_times, dtype=np.float64)
    if not beat_times_array.size:
        return -math.inf
    beat_support = _event_support(
        signal,
        beat_times_array,
        frame_rate_hz=frame_rate_hz,
        radius_ms=radius_ms,
    )
    half_support = _event_support(
        signal,
        half_times_array,
        frame_rate_hz=frame_rate_hz,
        radius_ms=radius_ms,
    )
    return float(
        np.mean(beat_support, dtype=np.float64)
        - 0.5
        * (
            np.mean(half_support, dtype=np.float64)
            if half_support.size
            else 0.0
        )
    )


def _select_exp013_production_candidate(
    candidates: tuple[PhaseContinuousTimingCurve, ...],
    *,
    raw_self_ranking: RawAudioEvidenceRanking | None,
    observations: tuple[LocalTempoObservation, ...],
    primary: _BaseHypothesis,
    beat_signal: NDArray[np.float64],
    frame_rate_hz: float,
    config: TempoTrackConfig,
) -> TempoTrackProductionSelection:
    if raw_self_ranking is None:
        return _fallback_selection("raw_audio_evidence_not_supplied")

    raw_scores_by_index = {
        score.candidate_index: score
        for score in raw_self_ranking.candidate_scores
    }
    if not raw_self_ranking.ranked_scores:
        return _fallback_selection("raw_self_scoring_unavailable")

    raw_observations = tuple(
        observation for observation in observations if observation.source == "raw_audio"
    )
    raw_run = _dominant_raw_tempo_run(
        raw_observations,
        base_bpm=primary.bpm,
        config=config,
    )

    if raw_run is None:
        constant_eligible = tuple(
            index
            for index, candidate in enumerate(candidates)
            if candidate.curve_class == "constant"
            and _raw_self_score_is_production_usable(
                raw_scores_by_index.get(index),
                config=config,
            )
        )
        if not constant_eligible:
            return _fallback_selection("no_production_eligible_constant")
        constant_raw_rank_by_index = _lane_raw_self_rank_by_index(
            constant_eligible,
            candidates=candidates,
            raw_scores_by_index=raw_scores_by_index,
        )
        selected = min(
            constant_eligible,
            key=lambda index: (
                constant_raw_rank_by_index[index],
                -_required_raw_score(raw_scores_by_index[index]),
                candidates[index].fingerprint_sha256,
            ),
        )
        return TempoTrackProductionSelection(
            status="v3_accepted",
            selected_candidate_index=selected,
            selected_fingerprint_sha256=candidates[selected].fingerprint_sha256,
            lane="constant",
            fallback_reason=None,
            raw_run=None,
            eligible_candidate_indices=constant_eligible,
            raw_self_rank_by_candidate=_rank_items(constant_raw_rank_by_index),
            beatthis_aba_rank_by_candidate=(),
        )

    eligible_deltas: list[tuple[int, float]] = []
    for index, candidate in enumerate(candidates):
        if not _raw_self_score_is_production_usable(
            raw_scores_by_index.get(index),
            config=config,
        ):
            continue
        support_delta = _eligible_jump_aba_delta(
            candidate,
            raw_run=raw_run,
            primary=primary,
            beat_signal=beat_signal,
            frame_rate_hz=frame_rate_hz,
            config=config,
        )
        if support_delta is not None:
            eligible_deltas.append((index, support_delta))

    if not eligible_deltas:
        return _fallback_selection(
            "no_production_eligible_jump_for_raw_run",
            raw_run=raw_run,
        )

    eligible = tuple(index for index, _ in eligible_deltas)
    lane_raw_rank_by_index = _lane_raw_self_rank_by_index(
        eligible,
        candidates=candidates,
        raw_scores_by_index=raw_scores_by_index,
    )
    beatthis_aba_rank_by_index = {
        index: rank
        for rank, (index, _) in enumerate(
            sorted(
                eligible_deltas,
                key=lambda value: (-value[1], candidates[value[0]].fingerprint_sha256),
            ),
            start=1,
        )
    }
    selected = min(
        eligible,
        key=lambda index: (
            lane_raw_rank_by_index[index] + beatthis_aba_rank_by_index[index],
            -_required_raw_score(raw_scores_by_index[index]),
            candidates[index].fingerprint_sha256,
        ),
    )
    return TempoTrackProductionSelection(
        status="v3_accepted",
        selected_candidate_index=selected,
        selected_fingerprint_sha256=candidates[selected].fingerprint_sha256,
        lane="paired_jump",
        fallback_reason=None,
        raw_run=raw_run,
        eligible_candidate_indices=eligible,
        raw_self_rank_by_candidate=_rank_items(lane_raw_rank_by_index),
        beatthis_aba_rank_by_candidate=_rank_items(beatthis_aba_rank_by_index),
    )


def _dominant_raw_tempo_run(
    observations: tuple[LocalTempoObservation, ...],
    *,
    base_bpm: float,
    config: TempoTrackConfig,
) -> TempoTrackRawRunDiagnostic | None:
    runs = _raw_tempo_runs(observations, base_bpm=base_bpm, config=config)
    return runs[0] if runs else None


def _raw_tempo_runs(
    observations: tuple[LocalTempoObservation, ...],
    *,
    base_bpm: float,
    config: TempoTrackConfig,
) -> tuple[TempoTrackRawRunDiagnostic, ...]:
    smoothed = _smoothed_raw_tempo_observations(
        observations,
        base_bpm=base_bpm,
        config=config,
    )
    if not smoothed:
        return ()

    runs: list[tuple[_SmoothedRawTempoObservation, ...]] = []
    current: list[_SmoothedRawTempoObservation] = []
    max_gap_ms = 1000.0 * config.local_hop_seconds * config.raw_run_gap_hop_multiplier
    for item in smoothed:
        if item.direction is None:
            if len(current) >= config.raw_run_minimum_observations:
                runs.append(tuple(current))
            current = []
            continue
        if (
            current
            and current[-1].direction == item.direction
            and item.observation.center_time_ms - current[-1].observation.center_time_ms
            <= max_gap_ms
        ):
            current.append(item)
            continue
        if len(current) >= config.raw_run_minimum_observations:
            runs.append(tuple(current))
        current = [item]
    if len(current) >= config.raw_run_minimum_observations:
        runs.append(tuple(current))
    if not runs:
        return ()

    diagnostics = tuple(
        _raw_run_diagnostic(run, base_bpm=base_bpm, config=config)
        for run in runs
    )
    return tuple(
        sorted(
            diagnostics,
            key=lambda run: (
                -abs(run.weighted_median_delta_bpm),
                -(run.end_time_ms - run.start_time_ms),
                -run.summed_strength,
                run.start_time_ms,
            ),
        )[: config.raw_run_maximum_retained_runs]
    )


def _smoothed_raw_tempo_observations(
    observations: tuple[LocalTempoObservation, ...],
    *,
    base_bpm: float,
    config: TempoTrackConfig,
) -> tuple[_SmoothedRawTempoObservation, ...]:
    ordered = tuple(sorted(observations, key=lambda value: value.center_time_ms))
    if not ordered:
        return ()
    normalized = np.asarray(
        [
            _snap_raw_observation_bpm_to_primary(
                observation.bpm,
                primary_bpm=base_bpm,
                config=config,
            )
            for observation in ordered
        ],
        dtype=np.float64,
    )
    weights = np.asarray(
        [max(observation.strength, 1e-9) for observation in ordered],
        dtype=np.float64,
    )
    threshold = max(
        config.raw_run_minimum_deviation_bpm,
        config.raw_run_minimum_deviation_fraction * base_bpm,
    )
    values: list[_SmoothedRawTempoObservation] = []
    for index, observation in enumerate(ordered):
        left = max(0, index - 1)
        right = min(len(ordered), index + 2)
        smoothed_bpm = _weighted_median(normalized[left:right], weights[left:right])
        delta_bpm = smoothed_bpm - base_bpm
        direction: Literal["up", "down"] | None
        if delta_bpm >= threshold:
            direction = "up"
        elif delta_bpm <= -threshold:
            direction = "down"
        else:
            direction = None
        values.append(
            _SmoothedRawTempoObservation(
                observation=observation,
                normalized_bpm=float(normalized[index]),
                smoothed_bpm=smoothed_bpm,
                delta_bpm=delta_bpm,
                direction=direction,
            )
        )
    return tuple(values)


def _raw_run_diagnostic(
    run: tuple[_SmoothedRawTempoObservation, ...],
    *,
    base_bpm: float,
    config: TempoTrackConfig,
) -> TempoTrackRawRunDiagnostic:
    deltas = np.asarray([item.delta_bpm for item in run], dtype=np.float64)
    bpms = np.asarray([item.smoothed_bpm for item in run], dtype=np.float64)
    weights = np.asarray(
        [max(item.observation.strength, 1e-9) for item in run],
        dtype=np.float64,
    )
    weighted_delta = _weighted_median(deltas, weights)
    median_bpm = _weighted_median(bpms, weights)
    start_time_ms = run[0].observation.center_time_ms
    end_time_ms = run[-1].observation.center_time_ms
    return TempoTrackRawRunDiagnostic(
        direction="up" if weighted_delta > 0.0 else "down",
        start_time_ms=float(start_time_ms),
        end_time_ms=float(end_time_ms),
        expanded_start_time_ms=float(start_time_ms - config.raw_run_expansion_ms),
        expanded_end_time_ms=float(end_time_ms + config.raw_run_expansion_ms),
        median_bpm=float(median_bpm if math.isfinite(median_bpm) else base_bpm),
        weighted_median_delta_bpm=float(weighted_delta),
        observation_count=len(run),
        summed_strength=float(sum(item.observation.strength for item in run)),
    )


def _weighted_median(values: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    if values.size == 0:
        raise ValueError("weighted median requires at least one value")
    if values.shape != weights.shape:
        raise ValueError("weighted median values and weights must have identical shape")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    total = float(np.sum(sorted_weights, dtype=np.float64))
    if total <= 0.0 or not math.isfinite(total):
        return float(sorted_values[sorted_values.size // 2])
    cumulative = np.cumsum(sorted_weights, dtype=np.float64)
    index = int(np.searchsorted(cumulative, 0.5 * total, side="left"))
    return float(sorted_values[min(index, sorted_values.size - 1)])


def _snap_raw_observation_bpm_to_primary(
    bpm: float,
    *,
    primary_bpm: float,
    config: TempoTrackConfig,
) -> float:
    best = bpm
    best_distance = math.inf
    seen: set[float] = set()
    for numerator in range(1, config.raw_run_rational_limit + 1):
        for denominator in range(1, config.raw_run_rational_limit + 1):
            multiplier = numerator / denominator
            transformed = bpm * multiplier
            if not config.minimum_bpm <= transformed <= config.maximum_bpm:
                continue
            key = round(transformed, 9)
            if key in seen:
                continue
            seen.add(key)
            distance = abs(math.log(transformed / primary_bpm))
            if distance < best_distance:
                best = transformed
                best_distance = distance
    if best_distance <= math.log(1.0 + config.raw_run_primary_snap_fraction):
        return float(primary_bpm)
    return float(bpm)


def _eligible_jump_aba_delta(
    candidate: PhaseContinuousTimingCurve,
    *,
    raw_run: TempoTrackRawRunDiagnostic,
    primary: _BaseHypothesis,
    beat_signal: NDArray[np.float64],
    frame_rate_hz: float,
    config: TempoTrackConfig,
) -> float | None:
    if candidate.curve_class != "jump" or len(candidate.sections) != 3:
        return None
    first, middle, last = candidate.sections
    if not (
        isinstance(first, ConstantTempoSection)
        and isinstance(middle, ConstantTempoSection)
        and isinstance(last, ConstantTempoSection)
    ):
        return None
    if not (
        _is_primary_alias_consistent(first.bpm, primary.bpm, config=config)
        and _is_primary_alias_consistent(last.bpm, primary.bpm, config=config)
    ):
        return None
    middle_delta = middle.bpm - primary.bpm
    if raw_run.direction == "up" and middle_delta <= 0.0:
        return None
    if raw_run.direction == "down" and middle_delta >= 0.0:
        return None
    if _primary_canonical_log_distance(
        middle.bpm,
        raw_run.median_bpm,
        primary_bpm=primary.bpm,
        config=config,
    ) > math.log(config.raw_run_jump_max_tempo_ratio):
        return None

    middle_start_ms = candidate.time_at_beat(float(middle.start_beat))
    middle_end_ms = candidate.time_at_beat(float(middle.end_beat))
    overlap_ms = _interval_overlap_ms(
        middle_start_ms,
        middle_end_ms,
        raw_run.expanded_start_time_ms,
        raw_run.expanded_end_time_ms,
    )
    if overlap_ms < config.raw_run_jump_minimum_overlap_ms:
        return None
    return _aba_support_delta(
        beat_signal,
        frame_rate_hz=frame_rate_hz,
        origin_time_ms=candidate.origin_time_ms,
        base_bpm=first.bpm,
        first_beat=middle.start_beat,
        middle_beats=middle.end_beat - middle.start_beat,
        middle_bpm=middle.bpm,
        radius_ms=config.beat_event_radius_ms,
    )


def _is_primary_alias_consistent(
    bpm: float,
    primary_bpm: float,
    *,
    config: TempoTrackConfig,
) -> bool:
    return (
        _snap_raw_observation_bpm_to_primary(bpm, primary_bpm=primary_bpm, config=config)
        == primary_bpm
    )


def _curve_is_primary_alias_family(
    candidate: PhaseContinuousTimingCurve,
    *,
    primary_bpm: float,
    config: TempoTrackConfig,
) -> bool:
    if candidate.curve_class != "jump" or not candidate.sections:
        return False
    return all(
        isinstance(section, ConstantTempoSection)
        and _is_primary_alias_consistent(section.bpm, primary_bpm, config=config)
        for section in candidate.sections
    )


def _primary_canonical_log_distance(
    left_bpm: float,
    right_bpm: float,
    *,
    primary_bpm: float,
    config: TempoTrackConfig,
) -> float:
    """Return distance after Exp013 primary-snap canonicalization.

    The Exp013 jump gate compares the candidate middle tempo to the raw-run
    representative tempo after the same primary-family canonicalization used
    for raw observations: each BPM maps to the primary alias family only when a
    p/q<=5 transform lands within the frozen 1.25% primary snap tolerance;
    otherwise it remains unchanged.  The final gate is a direct log-ratio check
    between those canonical BPMs.
    """

    left = _snap_raw_observation_bpm_to_primary(
        left_bpm,
        primary_bpm=primary_bpm,
        config=config,
    )
    right = _snap_raw_observation_bpm_to_primary(
        right_bpm,
        primary_bpm=primary_bpm,
        config=config,
    )
    return abs(math.log(left / right))


def _interval_overlap_ms(
    left_start_ms: float,
    left_end_ms: float,
    right_start_ms: float,
    right_end_ms: float,
) -> float:
    return max(0.0, min(left_end_ms, right_end_ms) - max(left_start_ms, right_start_ms))


def _shared_beat_count(
    observations: tuple[LocalTempoObservation, ...],
    *,
    primary: _BaseHypothesis,
    duration_ms: float,
    config: TempoTrackConfig,
) -> int:
    beatthis = tuple(
        observation for observation in observations if observation.source == "beatthis"
    )
    chosen = beatthis if len(beatthis) >= 6 else observations
    available_seconds = max(1e-6, (duration_ms - primary.origin_time_ms) / 1000.0)
    fit = _fit_linear_tempo(chosen, config=config)
    if fit is not None:
        slope, r_squared, _ = fit
        endpoint_delta = slope * available_seconds
    else:
        slope, r_squared, endpoint_delta = 0.0, 0.0, 0.0
    if chosen and r_squared >= 0.60 and abs(endpoint_delta) >= 8.0:
        times = np.asarray(
            [observation.center_time_ms / 1000.0 for observation in chosen],
            dtype=np.float64,
        )
        bpms = np.asarray(
            [
                _normalize_observation_alias(
                    observation.bpm,
                    reference_bpm=150.0,
                    config=config,
                )
                for observation in chosen
            ],
            dtype=np.float64,
        )
        weights = np.asarray(
            [max(observation.strength, 1e-3) for observation in chosen],
            dtype=np.float64,
        )
        intercept = float(np.average(bpms - slope * times, weights=weights))
        start_seconds = primary.origin_time_ms / 1000.0
        end_seconds = duration_ms / 1000.0
        integrated_beats = (
            intercept * (end_seconds - start_seconds)
            + 0.5 * slope * (end_seconds * end_seconds - start_seconds * start_seconds)
        ) / 60.0
        if math.isfinite(integrated_beats) and integrated_beats >= 1.0:
            return max(1, int(round(integrated_beats)))
    if chosen:
        normalized_bpms = np.asarray(
            [
                _normalize_observation_alias(
                    observation.bpm,
                    reference_bpm=primary.bpm,
                    config=config,
                )
                for observation in chosen
            ],
            dtype=np.float64,
        )
        tempo = float(np.median(normalized_bpms))
    else:
        tempo = primary.bpm
    return max(1, int(round(available_seconds * tempo / 60.0)))


def _sliding_tempo_observations(
    signal: NDArray[np.float64],
    *,
    frame_rate_hz: float,
    source: Literal["beatthis", "raw_audio"],
    config: TempoTrackConfig,
    time_offset_ms: float = 0.0,
) -> tuple[LocalTempoObservation, ...]:
    window_frames = max(8, int(round(config.local_window_seconds * frame_rate_hz)))
    hop_frames = max(1, int(round(config.local_hop_seconds * frame_rate_hz)))
    if signal.size < window_frames:
        return ()
    minimum_lag = max(1, int(math.ceil(60.0 * frame_rate_hz / config.maximum_bpm)))
    maximum_lag = min(
        window_frames // 2,
        int(math.floor(60.0 * frame_rate_hz / config.minimum_bpm)),
    )
    if maximum_lag < minimum_lag:
        return ()

    observations: list[LocalTempoObservation] = []
    starts = range(0, signal.size - window_frames + 1, hop_frames)
    for start in starts:
        end = start + window_frames
        values = signal[start:end].astype(np.float64, copy=False)
        centered = values - float(np.mean(values, dtype=np.float64))
        energy = float(np.dot(centered, centered))
        if energy <= 1e-12:
            continue
        scores = np.full(maximum_lag + 1, -np.inf, dtype=np.float64)
        for lag in range(minimum_lag, maximum_lag + 1):
            left = centered[:-lag]
            right = centered[lag:]
            denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
            if denominator > 0.0:
                scores[lag] = float(np.dot(left, right) / denominator)
        finite_lags = np.flatnonzero(np.isfinite(scores))
        if not finite_lags.size:
            continue
        best_score = float(np.max(scores[finite_lags]))
        # A periodic pulse train also correlates at slower integer multiples.
        # Prefer the shortest lag among statistically tied peaks so a complete
        # beat train wins over its half-tempo alias.
        tied = finite_lags[scores[finite_lags] >= best_score - 0.025]
        best_lag = int(np.min(tied))
        refined_lag = float(best_lag)
        if minimum_lag < best_lag < maximum_lag:
            left_score = float(scores[best_lag - 1])
            center_score = float(scores[best_lag])
            right_score = float(scores[best_lag + 1])
            denominator = left_score - 2.0 * center_score + right_score
            if math.isfinite(denominator) and denominator < 0.0:
                displacement = 0.5 * (left_score - right_score) / denominator
                if math.isfinite(displacement) and abs(displacement) <= 0.5:
                    refined_lag += displacement
        bpm = 60.0 * frame_rate_hz / refined_lag
        if not config.minimum_bpm <= bpm <= config.maximum_bpm:
            continue
        strength = float(np.clip(scores[best_lag], 0.0, 1.0))
        if strength < config.minimum_local_strength:
            continue
        start_ms = time_offset_ms + 1000.0 * start / frame_rate_hz
        end_ms = time_offset_ms + 1000.0 * end / frame_rate_hz
        observations.append(
            LocalTempoObservation(
                source=source,
                center_time_ms=0.5 * (start_ms + end_ms),
                window_start_ms=start_ms,
                window_end_ms=end_ms,
                bpm=float(bpm),
                strength=strength,
            )
        )
    return tuple(observations)


def _evidence_frame_rate_hz(evidence: RawAudioEvidence) -> float:
    centers = evidence.frame_center_seconds
    if centers.size < 2:
        return 100.0
    step = float(np.median(np.diff(centers)))
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("raw-audio evidence frame centers have invalid spacing")
    return 1.0 / step


def _base_hypotheses(
    beat_signal: NDArray[np.float64],
    *,
    frame_rate_hz: float,
    duration_ms: float,
    global_candidates: _global.GlobalConstantJumpCandidateSet,
    observations: tuple[LocalTempoObservation, ...],
    config: TempoTrackConfig,
) -> tuple[_BaseHypothesis, ...]:
    raw: list[tuple[float, float]] = []
    for origin in global_candidates.origin_candidates:
        for multiplier in (0.5, 1.0, 2.0):
            bpm = origin.bpm * multiplier
            if config.minimum_bpm <= bpm <= config.maximum_bpm:
                period_ms = 60000.0 / bpm
                normalized_origin = float(origin.time_ms % period_ms)
                raw.append((normalized_origin, float(bpm)))

    # The whole-track autocorrelation shortlist can miss the outer tempo of a
    # short A->B->A excursion.  Add modes from local tempo observations, then
    # estimate their phase directly from activation peaks.
    for bpm in _observation_tempo_modes(observations, config=config):
        origin_time_ms = _phase_origin_for_bpm(
            global_candidates,
            bpm=bpm,
            beat_signal=beat_signal,
            frame_rate_hz=frame_rate_hz,
            duration_ms=duration_ms,
            radius_ms=config.beat_event_radius_ms,
        )
        raw.append((origin_time_ms, bpm))
        peak_aligned_origin = _earliest_peak_aligned_origin_for_bpm(
            global_candidates,
            bpm=bpm,
        )
        if peak_aligned_origin is not None:
            raw.append((peak_aligned_origin, bpm))
        period_ms = 60000.0 / bpm
        for peak in sorted(
            global_candidates.beat_peaks,
            key=lambda value: (-value.confidence, value.time_ms),
        )[:4]:
            raw.append((float(peak.time_ms % period_ms), bpm))

    if not raw:
        fallback_bpm = _fallback_bpm(observations, config=config)
        peak_time = (
            global_candidates.beat_peaks[0].time_ms
            if global_candidates.beat_peaks
            else 0.0
        )
        raw.append((float(peak_time % (60000.0 / fallback_bpm)), fallback_bpm))

    deduped: dict[tuple[int, int], tuple[float, float]] = {}
    for origin_time_ms, bpm in raw:
        key = (int(round(origin_time_ms / 5.0)), int(round(bpm * 4.0)))
        deduped.setdefault(key, (origin_time_ms, bpm))

    peak_times = np.asarray(
        [peak.time_ms for peak in global_candidates.beat_peaks], dtype=np.float64
    )
    hypotheses = [
        _BaseHypothesis(
            origin_time_ms=origin_time_ms,
            bpm=bpm,
            score=_whole_track_grid_score(
                beat_signal,
                frame_rate_hz=frame_rate_hz,
                duration_ms=duration_ms,
                origin_time_ms=origin_time_ms,
                bpm=bpm,
                peak_times_ms=peak_times,
                radius_ms=config.beat_event_radius_ms,
            ),
        )
        for origin_time_ms, bpm in deduped.values()
    ]
    hypotheses.sort(key=lambda value: (-value.score, -value.bpm, value.origin_time_ms))

    retained: list[_BaseHypothesis] = []
    for hypothesis in hypotheses:
        if any(
            abs(existing.bpm - hypothesis.bpm) < 0.75
            and abs(existing.origin_time_ms - hypothesis.origin_time_ms) < 10.0
            for existing in retained
        ):
            continue
        retained.append(hypothesis)
        if len(retained) >= config.maximum_base_hypotheses:
            break
    return tuple(retained)


def _observation_tempo_modes(
    observations: Sequence[LocalTempoObservation],
    *,
    config: TempoTrackConfig,
) -> tuple[float, ...]:
    if not observations:
        return ()
    best_by_bin: dict[int, tuple[float, float, int]] = {}
    aggregates: dict[int, list[LocalTempoObservation]] = {}
    for observation in observations:
        bpm = _normalize_observation_alias(
            observation.bpm,
            reference_bpm=150.0,
            config=config,
        )
        key = int(round(bpm))
        aggregates.setdefault(key, []).append(
            LocalTempoObservation(
                source=observation.source,
                center_time_ms=observation.center_time_ms,
                window_start_ms=observation.window_start_ms,
                window_end_ms=observation.window_end_ms,
                bpm=bpm,
                strength=observation.strength,
            )
        )
    for key, values in aggregates.items():
        weight = float(sum(value.strength for value in values))
        weighted_bpm = float(
            sum(value.bpm * value.strength for value in values) / max(weight, 1e-12)
        )
        # Raw audio is independent evidence, so it breaks otherwise close
        # histogram ties without overwhelming the BeatThis track.
        raw_bonus = 0.15 * sum(
            value.strength for value in values if value.source == "raw_audio"
        )
        best_by_bin[key] = (weight + raw_bonus, weighted_bpm, len(values))
    ranked = sorted(
        best_by_bin.values(),
        key=lambda value: (-value[0], -value[2], -value[1]),
    )
    modes: list[float] = []
    for _, bpm, _ in ranked:
        if any(abs(bpm - existing) < 2.0 for existing in modes):
            continue
        modes.append(bpm)
        if len(modes) >= 6:
            break
    return tuple(modes)


def _phase_origin_for_bpm(
    candidates: _global.GlobalConstantJumpCandidateSet,
    *,
    bpm: float,
    beat_signal: NDArray[np.float64],
    frame_rate_hz: float,
    duration_ms: float,
    radius_ms: float,
) -> float:
    request_cache = _PHASE_ORIGIN_REQUEST_CACHE.get()
    cache_key: tuple[object, ...] | None = None
    if request_cache is not None:
        signal_interface = beat_signal.__array_interface__
        cache_key = (
            id(candidates),
            id(beat_signal),
            int(signal_interface["data"][0]),
            tuple(beat_signal.shape),
            tuple(beat_signal.strides),
            beat_signal.dtype.str,
            float(bpm).hex(),
            float(frame_rate_hz).hex(),
            float(duration_ms).hex(),
            float(radius_ms).hex(),
        )
        cached = request_cache.get(cache_key)
        if cached is not None:
            return cached

    period_ms = 60000.0 / bpm
    if not candidates.beat_peaks:
        best_phase = 0.0
        if request_cache is not None and cache_key is not None:
            request_cache[cache_key] = best_phase
        return best_phase
    peak_times = np.asarray(
        [peak.time_ms for peak in candidates.beat_peaks], dtype=np.float64
    )
    confidences = np.asarray(
        [peak.confidence for peak in candidates.beat_peaks], dtype=np.float64
    )
    phases = np.mod(peak_times, period_ms)
    angles = 2.0 * np.pi * phases / period_ms
    vector = np.sum(confidences * np.exp(1j * angles))
    if abs(vector) > 1e-9:
        circular_phase = float(np.mod(np.angle(vector), 2.0 * np.pi) * period_ms / (2.0 * np.pi))
    else:
        circular_phase = float(phases[int(np.argmax(confidences))])
    seeds = [circular_phase]
    seeds.extend(float(value) for value in phases[np.argsort(-confidences, kind="stable")[:16]])
    best_phase = circular_phase
    best_score = -math.inf
    for seed in seeds:
        for displacement in (-10.0, -5.0, 0.0, 5.0, 10.0):
            phase = float((seed + displacement) % period_ms)
            score = _whole_track_grid_score(
                beat_signal,
                frame_rate_hz=frame_rate_hz,
                duration_ms=duration_ms,
                origin_time_ms=phase,
                bpm=bpm,
                peak_times_ms=peak_times,
                radius_ms=radius_ms,
            )
            if score > best_score or (score == best_score and phase < best_phase):
                best_score = score
                best_phase = phase
    if request_cache is not None and cache_key is not None:
        request_cache[cache_key] = best_phase
    return best_phase


def _earliest_peak_aligned_origin_for_bpm(
    candidates: _global.GlobalConstantJumpCandidateSet,
    *,
    bpm: float,
) -> float | None:
    if not candidates.beat_peaks:
        return None
    period_ms = 60000.0 / bpm
    earliest_peak = min(candidates.beat_peaks, key=lambda peak: peak.time_ms)
    return float(earliest_peak.time_ms % period_ms)


def _fallback_bpm(
    observations: Sequence[LocalTempoObservation],
    *,
    config: TempoTrackConfig,
) -> float:
    if observations:
        bpms = np.asarray([value.bpm for value in observations], dtype=np.float64)
        return float(np.median(bpms))
    return float(0.5 * (config.minimum_bpm + config.maximum_bpm))


def _whole_track_grid_score(
    signal: NDArray[np.float64],
    *,
    frame_rate_hz: float,
    duration_ms: float,
    origin_time_ms: float,
    bpm: float,
    peak_times_ms: NDArray[np.float64],
    radius_ms: float,
) -> float:
    period_ms = 60000.0 / bpm
    beat_times = np.arange(origin_time_ms, duration_ms, period_ms, dtype=np.float64)
    if not beat_times.size:
        return -math.inf
    half_times = beat_times + 0.5 * period_ms
    half_times = half_times[half_times < duration_ms]
    beat_support = _event_support(
        signal, beat_times, frame_rate_hz=frame_rate_hz, radius_ms=radius_ms
    )
    half_support = _event_support(
        signal, half_times, frame_rate_hz=frame_rate_hz, radius_ms=radius_ms
    )
    recall = _grid_peak_recall(
        peak_times_ms,
        origin_time_ms=origin_time_ms,
        period_ms=period_ms,
        tolerance_ms=max(radius_ms, 0.12 * period_ms),
    )
    return float(
        np.mean(beat_support, dtype=np.float64)
        - 0.5 * np.mean(half_support, dtype=np.float64)
        + 0.5 * recall
    )


def _grid_peak_recall(
    peak_times_ms: NDArray[np.float64],
    *,
    origin_time_ms: float,
    period_ms: float,
    tolerance_ms: float,
) -> float:
    if not peak_times_ms.size:
        return 0.0
    phases = np.mod(peak_times_ms - origin_time_ms + 0.5 * period_ms, period_ms)
    distances = np.abs(phases - 0.5 * period_ms)
    return float(np.mean(distances <= tolerance_ms))


def _event_support(
    signal: NDArray[np.float64],
    event_times_ms: NDArray[np.float64],
    *,
    frame_rate_hz: float,
    radius_ms: float,
) -> NDArray[np.float64]:
    if not event_times_ms.size:
        return np.zeros(0, dtype=np.float64)
    centers = np.rint(event_times_ms * frame_rate_hz / 1000.0).astype(np.int64)
    radius_frames = max(0, int(math.ceil(radius_ms * frame_rate_hz / 1000.0)))
    request_cache = _EVENT_SUPPORT_REQUEST_CACHE.get()
    if request_cache is not None and signal.size:
        signal_interface = signal.__array_interface__
        cache_key = (
            id(signal),
            int(signal_interface["data"][0]),
            tuple(signal.shape),
            tuple(signal.strides),
            signal.dtype.str,
            radius_frames,
        )
        support_by_center = request_cache.get(cache_key)
        if support_by_center is None:
            padded = np.pad(
                signal,
                (2 * radius_frames, 2 * radius_frames),
                constant_values=-np.inf,
            )
            windows = np.lib.stride_tricks.sliding_window_view(
                padded,
                2 * radius_frames + 1,
            )
            support_by_center = np.max(windows, axis=1)
            support_by_center[~np.isfinite(support_by_center)] = 0.0
            support_by_center = support_by_center.astype(np.float64, copy=False)
            request_cache[cache_key] = support_by_center

        result = np.zeros(centers.shape, dtype=np.float64)
        valid = (centers >= -radius_frames) & (
            centers < signal.size + radius_frames
        )
        result[valid] = support_by_center[centers[valid] + radius_frames]
        return result

    offsets = np.arange(-radius_frames, radius_frames + 1, dtype=np.int64)
    indices = centers[:, None] + offsets[None, :]
    valid = (indices >= 0) & (indices < signal.size)
    safe = np.clip(indices, 0, max(0, signal.size - 1))
    values = np.where(valid, signal[safe], -np.inf)
    result = np.max(values, axis=1)
    result[~np.isfinite(result)] = 0.0
    return result.astype(np.float64, copy=False)


def _unmerged_boundary_seeds(
    candidates: _global.GlobalConstantJumpCandidateSet,
    *,
    config: TempoTrackConfig,
) -> tuple[_BoundarySeed, ...]:
    raw: list[_BoundarySeed] = []
    for peak_index in range(len(candidates.beat_peaks)):
        boundary = _global._raw_boundary_candidate(  # noqa: SLF001
            candidates.beat_peaks,
            candidates.downbeat_peaks,
            peak_index,
            _global.GLOBAL_CONSTANT_JUMP_CONSTANTS,
        )
        if boundary is not None:
            raw.append(
                _BoundarySeed(
                    time_ms=float(boundary.time_ms),
                    rank_score=float(boundary.rank_score),
                )
            )
    ranked = sorted(raw, key=lambda value: (-value.rank_score, value.time_ms))
    retained = ranked[: config.maximum_boundary_seeds]
    return tuple(sorted(retained, key=lambda value: value.time_ms))


def _pair_seeds(
    boundaries: tuple[_BoundarySeed, ...],
    *,
    observations: tuple[LocalTempoObservation, ...],
    bases: tuple[_BaseHypothesis, ...],
    config: TempoTrackConfig,
) -> tuple[_PairSeed, ...]:
    pairs: list[_PairSeed] = []
    minimum_ms = 1000.0 * config.minimum_excursion_seconds
    maximum_ms = 1000.0 * config.maximum_excursion_seconds
    for left_index, left in enumerate(boundaries):
        for right in boundaries[left_index + 1 :]:
            duration = right.time_ms - left.time_ms
            if duration > maximum_ms:
                break
            if duration < minimum_ms:
                continue
            pairs.append(
                _PairSeed(
                    left_time_ms=left.time_ms,
                    right_time_ms=right.time_ms,
                    rank_score=left.rank_score + right.rank_score,
                    source="paired_unmerged_boundary",
                )
            )

    # If the second transition is missing from BeatThis, synthesize it from an
    # independent local-tempo estimate.  The right shoulder in the local
    # support score decides how many middle beats actually fit.
    for base in bases[:1]:
        deviating = [
            observation
            for observation in observations
            if abs(observation.bpm - base.bpm) >= config.minimum_jump_bpm
        ]
        for left in boundaries:
            nearby = sorted(
                (
                    observation
                    for observation in deviating
                    if left.time_ms - 1000.0
                    <= observation.center_time_ms
                    <= left.time_ms + maximum_ms
                ),
                key=lambda value: (
                    -value.strength,
                    abs(value.center_time_ms - left.time_ms - 0.5 * maximum_ms),
                    value.center_time_ms,
                ),
            )[:3]
            for observation in nearby:
                minimum_beats = max(
                    1,
                    int(math.ceil(config.minimum_excursion_seconds * observation.bpm / 60.0)),
                )
                maximum_beats = int(
                    math.floor(config.maximum_excursion_seconds * observation.bpm / 60.0)
                )
                for beat_count in range(minimum_beats, maximum_beats + 1):
                    right_time_ms = (
                        left.time_ms + 60000.0 * beat_count / observation.bpm
                    )
                    pairs.append(
                        _PairSeed(
                            left_time_ms=left.time_ms,
                            right_time_ms=right_time_ms,
                            rank_score=left.rank_score + observation.strength,
                            source=f"virtual_right_{observation.source}",
                            preferred_bpm=observation.bpm,
                        )
                    )

    deduped: dict[tuple[int, int, int], _PairSeed] = {}
    for pair in pairs:
        key = (
            int(round(pair.left_time_ms / 20.0)),
            int(round(pair.right_time_ms / 20.0)),
            int(round((pair.preferred_bpm or 0.0) * 2.0)),
        )
        previous = deduped.get(key)
        if previous is None or pair.rank_score > previous.rank_score:
            deduped[key] = pair
    order_key = lambda value: (  # noqa: E731
        -value.rank_score,
        value.left_time_ms,
        value.right_time_ms,
        value.source,
    )
    values = tuple(deduped.values())
    raw_virtual = sorted(
        (value for value in values if value.source == "virtual_right_raw_audio"),
        key=order_key,
    )
    beatthis_virtual = sorted(
        (value for value in values if value.source == "virtual_right_beatthis"),
        key=order_key,
    )
    paired = sorted(
        (value for value in values if value.source == "paired_unmerged_boundary"),
        key=order_key,
    )
    # Reserve proposal bandwidth for the independent raw observation.  Pure
    # rank truncation is vulnerable to hundreds of correlated boundary pairs.
    raw_cap = min(len(raw_virtual), max(16, config.maximum_pair_seeds // 4))
    beatthis_cap = min(len(beatthis_virtual), max(16, config.maximum_pair_seeds // 4))
    paired_cap = max(0, config.maximum_pair_seeds - raw_cap - beatthis_cap)
    retained = raw_virtual[:raw_cap] + beatthis_virtual[:beatthis_cap] + paired[:paired_cap]
    retained.sort(key=order_key)
    return tuple(retained[: config.maximum_pair_seeds])


def _short_chain_canonical_bpm(
    bpm: float,
    *,
    primary_bpm: float,
    config: TempoTrackConfig,
) -> float:
    return float(
        _normalize_observation_alias(
            bpm,
            reference_bpm=primary_bpm,
            config=config,
        )
    )


def _short_chain_local_mode(
    observations: tuple[LocalTempoObservation, ...],
    *,
    center_time_ms: float,
    radius_ms: float,
    primary_bpm: float,
    config: TempoTrackConfig,
) -> float | None:
    selected = tuple(
        observation
        for observation in observations
        if abs(observation.center_time_ms - center_time_ms) <= radius_ms
        and observation.strength >= config.minimum_local_strength
    )
    if not selected:
        return None
    values = np.asarray(
        [
            _short_chain_canonical_bpm(
                observation.bpm,
                primary_bpm=primary_bpm,
                config=config,
            )
            for observation in selected
        ],
        dtype=np.float64,
    )
    weights = np.asarray(
        [max(1e-3, float(observation.strength)) for observation in selected],
        dtype=np.float64,
    )
    return _weighted_median(values, weights)


def _short_chain_points(
    beatthis: tuple[LocalTempoObservation, ...],
    raw: tuple[LocalTempoObservation, ...],
    *,
    primary_bpm: float,
    config: TempoTrackConfig,
) -> tuple[_ShortChainPoint, ...]:
    points: list[_ShortChainPoint] = []
    for observation in beatthis:
        center_time_ms = float(observation.center_time_ms)
        beatthis_bpm = _short_chain_local_mode(
            beatthis,
            center_time_ms=center_time_ms,
            radius_ms=750.0,
            primary_bpm=primary_bpm,
            config=config,
        )
        raw_bpm = _short_chain_local_mode(
            raw,
            center_time_ms=center_time_ms,
            radius_ms=750.0,
            primary_bpm=primary_bpm,
            config=config,
        )
        if beatthis_bpm is None or raw_bpm is None:
            continue
        tolerance_bpm = max(
            _SHORT_CHAIN_SOURCE_AGREEMENT_BPM,
            0.025 * max(beatthis_bpm, raw_bpm),
        )
        if abs(beatthis_bpm - raw_bpm) > tolerance_bpm:
            continue
        points.append(
            _ShortChainPoint(
                time_ms=center_time_ms,
                beatthis_bpm=beatthis_bpm,
                raw_bpm=raw_bpm,
                combined_bpm=0.5 * (beatthis_bpm + raw_bpm),
            )
        )
    deduped: dict[int, _ShortChainPoint] = {}
    for point in points:
        deduped.setdefault(int(round(point.time_ms / 100.0)), point)
    return tuple(sorted(deduped.values(), key=lambda point: point.time_ms))


def _short_chain_linear_fit_score(
    times_ms: NDArray[np.float64],
    values: NDArray[np.float64],
) -> tuple[float, float]:
    times = (times_ms - float(times_ms[0])) / 1000.0
    centered_times = times - float(np.mean(times, dtype=np.float64))
    centered_values = values - float(np.mean(values, dtype=np.float64))
    denominator = float(np.dot(centered_times, centered_times))
    if denominator <= 0.0:
        return 0.0, 0.0
    slope = float(np.dot(centered_times, centered_values) / denominator)
    fitted = float(np.mean(values, dtype=np.float64)) + slope * centered_times
    residual = float(np.sum(np.square(values - fitted), dtype=np.float64))
    total = float(np.sum(np.square(centered_values), dtype=np.float64))
    r_squared = 0.0 if total <= 1e-12 else max(0.0, 1.0 - residual / total)
    return slope, r_squared


def _best_short_chain_region(
    points: tuple[_ShortChainPoint, ...],
    *,
    primary_bpm: float,
) -> _ShortChainRegion | None:
    best: _ShortChainRegion | None = None
    for left_index, left in enumerate(points):
        if abs(left.combined_bpm - primary_bpm) > max(3.0, 0.025 * primary_bpm):
            continue
        for right_index in range(left_index + 5, len(points)):
            right = points[right_index]
            duration_seconds = (right.time_ms - left.time_ms) / 1000.0
            if duration_seconds < _SHORT_CHAIN_MINIMUM_TRAJECTORY_SECONDS:
                continue
            if duration_seconds > _SHORT_CHAIN_MAXIMUM_TRAJECTORY_SECONDS:
                break
            segment = points[left_index : right_index + 1]
            if len(segment) < 8:
                continue
            beatthis_delta = float(
                np.median([point.beatthis_bpm for point in segment[-3:]])
                - np.median([point.beatthis_bpm for point in segment[:3]])
            )
            raw_delta = float(
                np.median([point.raw_bpm for point in segment[-3:]])
                - np.median([point.raw_bpm for point in segment[:3]])
            )
            if (
                abs(beatthis_delta) < _SHORT_CHAIN_MINIMUM_TOTAL_DELTA_BPM
                or abs(raw_delta) < _SHORT_CHAIN_MINIMUM_TOTAL_DELTA_BPM
                or beatthis_delta * raw_delta <= 0.0
            ):
                continue
            direction = 1 if beatthis_delta > 0.0 else -1
            times = np.asarray(
                [point.time_ms for point in segment],
                dtype=np.float64,
            )
            beatthis_values = np.asarray(
                [point.beatthis_bpm for point in segment],
                dtype=np.float64,
            )
            raw_values = np.asarray(
                [point.raw_bpm for point in segment],
                dtype=np.float64,
            )
            beatthis_slope, beatthis_r_squared = _short_chain_linear_fit_score(
                times,
                beatthis_values,
            )
            raw_slope, raw_r_squared = _short_chain_linear_fit_score(
                times,
                raw_values,
            )
            if direction * beatthis_slope <= 0.0 or direction * raw_slope <= 0.0:
                continue
            if min(beatthis_r_squared, raw_r_squared) < 0.35:
                continue
            score = min(abs(beatthis_delta), abs(raw_delta)) * math.sqrt(
                beatthis_r_squared * raw_r_squared
            )
            region = _ShortChainRegion(
                start_time_ms=float(left.time_ms),
                end_time_ms=float(right.time_ms),
                direction=direction,
                score=float(score),
            )
            if best is None or (region.score, -region.start_time_ms) > (
                best.score,
                -best.start_time_ms,
            ):
                best = region
    return best


def _short_chain_sample_level(
    *,
    center_time_ms: float,
    beatthis: tuple[LocalTempoObservation, ...],
    raw: tuple[LocalTempoObservation, ...],
    primary_bpm: float,
    config: TempoTrackConfig,
) -> float | None:
    beatthis_bpm = _short_chain_local_mode(
        beatthis,
        center_time_ms=center_time_ms,
        radius_ms=300.0,
        primary_bpm=primary_bpm,
        config=config,
    )
    raw_bpm = _short_chain_local_mode(
        raw,
        center_time_ms=center_time_ms,
        radius_ms=300.0,
        primary_bpm=primary_bpm,
        config=config,
    )
    if beatthis_bpm is None or raw_bpm is None:
        return None
    tolerance_bpm = max(
        _SHORT_CHAIN_SOURCE_AGREEMENT_BPM,
        0.025 * max(beatthis_bpm, raw_bpm),
    )
    if abs(beatthis_bpm - raw_bpm) > tolerance_bpm:
        return None
    return 0.5 * (beatthis_bpm + raw_bpm)


def _short_chain_candidate_curves(
    *,
    signal: NDArray[np.float64],
    frame_rate_hz: float,
    duration_ms: float,
    base: _BaseHypothesis,
    beatthis: tuple[LocalTempoObservation, ...],
    raw: tuple[LocalTempoObservation, ...],
    region: _ShortChainRegion,
    config: TempoTrackConfig,
) -> tuple[tuple[PhaseContinuousTimingCurve, float, bool], ...]:
    period_ms = 60000.0 / base.bpm
    minimum_time_ms = max(
        base.origin_time_ms + period_ms,
        region.start_time_ms - 2500.0,
    )
    maximum_time_ms = min(duration_ms - 1000.0, region.end_time_ms - 2500.0)
    minimum_beat = max(
        1,
        int(math.floor((minimum_time_ms - base.origin_time_ms) / period_ms)),
    )
    maximum_beat = int(
        math.ceil((maximum_time_ms - base.origin_time_ms) / period_ms)
    )
    candidates: list[tuple[PhaseContinuousTimingCurve, float, bool]] = []
    for first_beat in range(minimum_beat, maximum_beat + 1):
        first_time_ms = base.origin_time_ms + first_beat * period_ms
        if not minimum_time_ms - period_ms <= first_time_ms <= maximum_time_ms + period_ms:
            continue
        for section_count in range(
            _SHORT_CHAIN_MINIMUM_SECTIONS,
            _SHORT_CHAIN_MAXIMUM_SECTIONS + 1,
        ):
            for nominal in (False, True):
                levels = [float(round(base.bpm)) if nominal else float(base.bpm)]
                starts = [0, first_beat]
                current_time_ms = first_time_ms
                valid = True
                for level_index in range(1, section_count):
                    observed = _short_chain_sample_level(
                        center_time_ms=current_time_ms + 900.0,
                        beatthis=beatthis,
                        raw=raw,
                        primary_bpm=base.bpm,
                        config=config,
                    )
                    if observed is None:
                        valid = False
                        break
                    if nominal and level_index == section_count - 1:
                        level = float(
                            math.ceil(observed)
                            if region.direction > 0
                            else math.floor(observed)
                        )
                    else:
                        level = float(round(observed)) if nominal else float(observed)
                    if region.direction * (level - levels[-1]) < -0.5:
                        valid = False
                        break
                    if abs(level - levels[-1]) < 0.75:
                        level = levels[-1]
                    levels.append(level)
                    if level_index < section_count - 1:
                        if level == levels[-2]:
                            valid = False
                            break
                        current_time_ms += (
                            _SHORT_CHAIN_SECTION_BEATS * 60000.0 / level
                        )
                        starts.append(starts[-1] + _SHORT_CHAIN_SECTION_BEATS)
                if not valid or len(levels) != section_count:
                    continue
                if (
                    region.direction * (levels[-1] - levels[0])
                    < _SHORT_CHAIN_MINIMUM_TOTAL_DELTA_BPM
                ):
                    continue
                if any(
                    region.direction * (right - left) <= 0.0
                    for left, right in zip(levels, levels[1:])
                ):
                    continue
                last_boundary_time_ms = first_time_ms + sum(
                    _SHORT_CHAIN_SECTION_BEATS * 60000.0 / bpm
                    for bpm in levels[1:-1]
                )
                end_beat = _terminal_beat_after_boundary(
                    boundary_beat=starts[-1],
                    boundary_time_ms=last_boundary_time_ms,
                    duration_ms=duration_ms,
                    bpm=levels[-1],
                )
                ends = starts[1:] + [end_beat]
                try:
                    curve = PhaseContinuousTimingCurve(
                        origin_beat=0,
                        origin_time_ms=base.origin_time_ms,
                        sections=tuple(
                            ConstantTempoSection(start, end, bpm)
                            for start, end, bpm in zip(
                                starts,
                                ends,
                                levels,
                                strict=True,
                            )
                        ),
                    )
                except ValueError:
                    continue
                if curve.end_time_ms + 1e-6 < duration_ms:
                    continue
                collapsed = _collapsed_constant_counterfactual(
                    curve,
                    duration_ms=duration_ms,
                    base_bpm=levels[0],
                    config=config,
                )
                beatthis_delta = _generalized_support_delta(
                    signal,
                    frame_rate_hz=frame_rate_hz,
                    candidate=curve,
                    collapsed=collapsed,
                    radius_ms=config.beat_event_radius_ms,
                )
                if not math.isfinite(beatthis_delta) or beatthis_delta <= 0.0:
                    continue
                candidates.append((curve, float(beatthis_delta), nominal))
    return tuple(candidates)


def _short_window_monotone_chain_proposal(
    *,
    signal: NDArray[np.float64],
    frame_rate_hz: float,
    duration_ms: float,
    bases: tuple[_BaseHypothesis, ...],
    observations: tuple[LocalTempoObservation, ...],
    config: TempoTrackConfig,
) -> _CurveProposal | None:
    beatthis = tuple(
        observation for observation in observations if observation.source == "beatthis"
    )
    raw = tuple(
        observation for observation in observations if observation.source == "raw_audio"
    )
    if not beatthis or not raw or not bases:
        return None
    candidates: list[tuple[PhaseContinuousTimingCurve, float, bool, float]] = []
    for base in bases[:2]:
        points = _short_chain_points(
            beatthis,
            raw,
            primary_bpm=base.bpm,
            config=config,
        )
        region = _best_short_chain_region(points, primary_bpm=base.bpm)
        if region is None:
            continue
        candidates.extend(
            (
                curve,
                beatthis_delta,
                nominal,
                region.score,
            )
            for curve, beatthis_delta, nominal in _short_chain_candidate_curves(
                signal=signal,
                frame_rate_hz=frame_rate_hz,
                duration_ms=duration_ms,
                base=base,
                beatthis=beatthis,
                raw=raw,
                region=region,
                config=config,
            )
        )
    if not candidates:
        return None
    curve, beatthis_delta, nominal, region_score = max(
        candidates,
        key=lambda value: (
            value[1],
            int(value[2]),
            value[3],
            -len(value[0].sections),
            value[0].fingerprint_sha256,
        ),
    )
    return _CurveProposal(
        curve=curve,
        source=_SHORT_WINDOW_MONOTONE_CHAIN_SOURCE,
        score=float(
            beatthis_delta + 0.1 * region_score + (0.01 if nominal else 0.0)
        ),
        collapse_bpm=float(curve.sections[0].start_bpm),
    )


def _jump_proposals(
    signal: NDArray[np.float64],
    *,
    frame_rate_hz: float,
    duration_ms: float,
    shared_end_beat: int,
    bases: tuple[_BaseHypothesis, ...],
    observations: tuple[LocalTempoObservation, ...],
    pair_seeds: tuple[_PairSeed, ...],
    config: TempoTrackConfig,
    global_candidates: _global.GlobalConstantJumpCandidateSet | None = None,
    boundary_seeds: tuple[_BoundarySeed, ...] = (),
    short_observations: tuple[LocalTempoObservation, ...] = (),
) -> _JumpProposalBatch:
    specs: list[tuple[float, int, int, float, _BaseHypothesis, str, float | None]] = []
    seen: set[tuple[int, int, int, int]] = set()
    tempo_bonus_observations = tuple(
        sorted(observations, key=lambda value: value.center_time_ms)
    )
    tempo_bonus_center_times = tuple(
        value.center_time_ms for value in tempo_bonus_observations
    )
    for base_index, base in enumerate(bases[:2]):
        period_ms = 60000.0 / base.bpm
        for pair in pair_seeds:
            center_beat = int(round((pair.left_time_ms - base.origin_time_ms) / period_ms))
            for first_beat in (center_beat - 1, center_beat, center_beat + 1):
                if first_beat <= 0 or first_beat >= shared_end_beat - 1:
                    continue
                left_time_ms = base.origin_time_ms + first_beat * period_ms
                middle_duration_ms = pair.right_time_ms - left_time_ms
                if not (
                    1000.0 * config.minimum_excursion_seconds
                    <= middle_duration_ms
                    <= 1000.0 * config.maximum_excursion_seconds
                ):
                    continue
                middle_beat_counts = _sparse_middle_beat_counts(
                    pair=pair,
                    observations=observations,
                    base=base,
                    first_beat=first_beat,
                    left_time_ms=left_time_ms,
                    middle_duration_ms=middle_duration_ms,
                    shared_end_beat=shared_end_beat,
                    config=config,
                )
                for middle_beats in middle_beat_counts:
                    final_beat = first_beat + middle_beats
                    if final_beat >= shared_end_beat:
                        break
                    middle_bpm = 60000.0 * middle_beats / middle_duration_ms
                    if abs(middle_bpm - base.bpm) < config.minimum_jump_bpm:
                        continue
                    key = (
                        base_index,
                        first_beat,
                        middle_beats,
                        int(round(middle_bpm * 1000.0)),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    support_delta = _aba_support_delta(
                        signal,
                        frame_rate_hz=frame_rate_hz,
                        origin_time_ms=base.origin_time_ms,
                        base_bpm=base.bpm,
                        first_beat=first_beat,
                        middle_beats=middle_beats,
                        middle_bpm=middle_bpm,
                        radius_ms=config.beat_event_radius_ms,
                    )
                    observation_bonus = _local_tempo_bonus(
                        tempo_bonus_observations,
                        center_times_ms=tempo_bonus_center_times,
                        middle_time_ms=left_time_ms + 0.5 * middle_duration_ms,
                        bpm=middle_bpm,
                        duration_ms=middle_duration_ms,
                    )
                    preferred_bonus = 0.0
                    if pair.preferred_bpm is not None:
                        preferred_bonus = 0.20 * math.exp(
                            -abs(middle_bpm - pair.preferred_bpm) / 4.0
                        )
                    score = (
                        support_delta
                        + 0.04 * pair.rank_score
                        + observation_bonus
                        + preferred_bonus
                    )
                    paired_or_virtual_source = (
                        pair.source == "paired_unmerged_boundary"
                        or pair.source.startswith("virtual_right_")
                    )
                    aba_support_delta = (
                        float(support_delta)
                        if (
                            paired_or_virtual_source
                            and math.isfinite(support_delta)
                            and _EXP017_SHORT_ABA_MIN_SECONDS
                            <= middle_duration_ms / 1000.0
                            <= _EXP017_SHORT_ABA_MAX_SECONDS
                        )
                        else None
                    )
                    specs.append(
                        (
                            float(score),
                            first_beat,
                            middle_beats,
                            float(middle_bpm),
                            base,
                            pair.source,
                            aba_support_delta,
                        )
                    )

    specs.sort(
        key=lambda value: (
            -value[0],
            value[1],
            value[2],
            value[3],
            value[4].origin_time_ms,
        )
    )
    proposals: list[_CurveProposal] = []
    for (
        score,
        first_beat,
        middle_beats,
        middle_bpm,
        base,
        source,
        aba_support_delta,
    ) in specs:
        final_beat = first_beat + middle_beats
        middle_end_time_ms = (
            base.origin_time_ms
            + first_beat * 60000.0 / base.bpm
            + middle_beats * 60000.0 / middle_bpm
        )
        end_beat = _terminal_beat_after_boundary(
            boundary_beat=final_beat,
            boundary_time_ms=middle_end_time_ms,
            duration_ms=duration_ms,
            bpm=base.bpm,
        )
        try:
            curve = PhaseContinuousTimingCurve(
                origin_beat=0,
                origin_time_ms=base.origin_time_ms,
                sections=(
                    ConstantTempoSection(0, first_beat, base.bpm),
                    ConstantTempoSection(first_beat, final_beat, middle_bpm),
                    ConstantTempoSection(final_beat, end_beat, base.bpm),
                ),
            )
        except ValueError:
            continue
        proposals.append(
            _CurveProposal(
                curve=curve,
                source=source,
                score=float(score),
                collapse_bpm=base.bpm,
                aba_support_delta=aba_support_delta,
            )
        )

    raw_run_proposals = _raw_run_jump_proposals(
        signal,
        frame_rate_hz=frame_rate_hz,
        duration_ms=duration_ms,
        bases=bases,
        observations=observations,
        config=config,
    )
    regional_proposals = _observation_side_segment_proposals(
        signal,
        frame_rate_hz=frame_rate_hz,
        duration_ms=duration_ms,
        global_candidates=global_candidates,
        bases=bases,
        observations=observations,
        boundary_seeds=boundary_seeds,
        pair_seeds=pair_seeds,
        config=config,
    )
    retained = _retain_jump_proposals_by_family(
        regional_proposals + raw_run_proposals + tuple(proposals),
        config=config,
    )
    short_chain = _short_window_monotone_chain_proposal(
        signal=signal,
        frame_rate_hz=frame_rate_hz,
        duration_ms=duration_ms,
        bases=bases,
        observations=short_observations,
        config=config,
    )
    if short_chain is None:
        return retained
    return _JumpProposalBatch(
        proposals=retained.proposals + (short_chain,),
        pruning_reason=retained.pruning_reason,
    )


def _sparse_middle_beat_counts(
    *,
    pair: _PairSeed,
    observations: tuple[LocalTempoObservation, ...],
    base: _BaseHypothesis,
    first_beat: int,
    left_time_ms: float,
    middle_duration_ms: float,
    shared_end_beat: int,
    config: TempoTrackConfig,
) -> tuple[int, ...]:
    minimum_beats = max(
        1,
        int(math.ceil(middle_duration_ms * config.minimum_bpm / 60000.0)),
    )
    maximum_beats = int(
        math.floor(middle_duration_ms * config.maximum_bpm / 60000.0)
    )
    if maximum_beats < minimum_beats:
        return ()

    middle_time_ms = left_time_ms + 0.5 * middle_duration_ms
    regional_radius_ms = max(1000.0, 0.75 * middle_duration_ms)
    regional_observations = tuple(
        observation
        for observation in observations
        if abs(observation.center_time_ms - middle_time_ms) <= regional_radius_ms
    )
    regional_modes = _regional_observation_modes(
        regional_observations,
        config=config,
    )
    reference_bpm = pair.preferred_bpm or base.bpm

    def mode_order(mode: _RegionalTempoMode) -> tuple[float, float, float]:
        return (
            -mode.support,
            abs(math.log2(mode.bpm / reference_bpm)),
            mode.bpm,
        )

    raw_modes = sorted(
        (mode for mode in regional_modes if mode.source == "raw_audio"),
        key=mode_order,
    )
    beatthis_modes = sorted(
        (mode for mode in regional_modes if mode.source == "beatthis"),
        key=mode_order,
    )
    evidence_bpms: list[float] = []
    if pair.preferred_bpm is not None:
        evidence_bpms.append(pair.preferred_bpm)
    for rank in range(max(len(raw_modes), len(beatthis_modes))):
        if rank < len(raw_modes):
            evidence_bpms.append(raw_modes[rank].bpm)
        if rank < len(beatthis_modes):
            evidence_bpms.append(beatthis_modes[rank].bpm)

    retained: list[int] = []
    retained_set: set[int] = set()

    def append_bpm(bpm: float) -> bool:
        if not math.isfinite(bpm) or not config.minimum_bpm <= bpm <= config.maximum_bpm:
            return False
        middle_beats = int(round(middle_duration_ms * bpm / 60000.0))
        if not minimum_beats <= middle_beats <= maximum_beats:
            return False
        if first_beat + middle_beats >= shared_end_beat:
            return False
        implied_bpm = 60000.0 * middle_beats / middle_duration_ms
        if abs(implied_bpm - base.bpm) < config.minimum_jump_bpm:
            return False
        if middle_beats in retained_set:
            return False
        retained.append(middle_beats)
        retained_set.add(middle_beats)
        return True

    evidence_limit = _EXP026_SPARSE_MIDDLE_BEAT_CAP - 1
    evidence_cursor = 0
    while evidence_cursor < len(evidence_bpms) and len(retained) < evidence_limit:
        append_bpm(evidence_bpms[evidence_cursor])
        evidence_cursor += 1

    nominal_bpms = [5.0 * round(bpm / 5.0) for bpm in evidence_bpms]
    nominal_center = 5.0 * round(reference_bpm / 5.0)
    maximum_offset_steps = int(
        math.ceil((config.maximum_bpm - config.minimum_bpm) / 5.0)
    )
    for offset_step in range(maximum_offset_steps + 1):
        if offset_step == 0:
            nominal_bpms.append(nominal_center)
            continue
        nominal_bpms.extend(
            (
                nominal_center - 5.0 * offset_step,
                nominal_center + 5.0 * offset_step,
            )
        )
    for bpm in nominal_bpms:
        if len(retained) >= _EXP026_SPARSE_MIDDLE_BEAT_CAP:
            break
        append_bpm(bpm)

    while evidence_cursor < len(evidence_bpms) and len(retained) < _EXP026_SPARSE_MIDDLE_BEAT_CAP:
        append_bpm(evidence_bpms[evidence_cursor])
        evidence_cursor += 1
    return tuple(retained)


def _observation_side_segment_proposals(
    signal: NDArray[np.float64],
    *,
    frame_rate_hz: float,
    duration_ms: float,
    global_candidates: _global.GlobalConstantJumpCandidateSet | None,
    bases: tuple[_BaseHypothesis, ...],
    observations: tuple[LocalTempoObservation, ...],
    boundary_seeds: tuple[_BoundarySeed, ...],
    pair_seeds: tuple[_PairSeed, ...],
    config: TempoTrackConfig,
) -> tuple[_CurveProposal, ...]:
    if not bases or not observations:
        return ()
    if global_candidates is None:
        return ()

    proposals: list[_CurveProposal] = []
    closed_aba_proposals: list[_CurveProposal] = []
    raw_closed_aba_proposals: list[_CurveProposal] = []
    seen: set[str] = set()
    side_seeds = _observation_side_segment_seeds(
        observations,
        boundary_seeds=boundary_seeds,
        config=config,
    )
    for base in bases[:2]:
        for seed in side_seeds:
            _append_observation_side_single_boundary_proposals(
                proposals,
                seen=seen,
                signal=signal,
                frame_rate_hz=frame_rate_hz,
                duration_ms=duration_ms,
                global_candidates=global_candidates,
                base=base,
                observations=observations,
                seed=seed,
                config=config,
            )
    ranked_boundary_pairs = sorted(
        (
            (score, pair)
            for pair in pair_seeds
            if pair.source == "paired_unmerged_boundary"
            and (
                score := _regional_closed_aba_pair_score(
                    pair,
                    observations=observations,
                    config=config,
                )
            )
            is not None
        ),
        key=lambda value: (-value[0], -value[1].rank_score, value[1].left_time_ms),
    )[:14]
    raw_closed_pairs: list[tuple[float, _PairSeed]] = []
    half_window_ms = 500.0 * config.local_window_seconds
    raw_observations = tuple(
        observation for observation in observations if observation.source == "raw_audio"
    )
    for raw_run in _raw_tempo_runs(
        raw_observations,
        base_bpm=bases[0].bpm,
        config=config,
    ):
        left_target_ms = max(
            0.0,
            raw_run.expanded_start_time_ms - half_window_ms,
        )
        right_target_ms = min(
            duration_ms,
            raw_run.expanded_end_time_ms + half_window_ms,
        )
        left_brackets = _boundary_brackets(
            boundary_seeds,
            target_time_ms=left_target_ms,
        )
        right_brackets = _boundary_brackets(
            boundary_seeds,
            target_time_ms=right_target_ms,
        )
        for left_seed in left_brackets or (
            _BoundarySeed(left_target_ms, 0.0),
        ):
            for right_seed in right_brackets or (
                _BoundarySeed(right_target_ms, 0.0),
            ):
                if right_seed.time_ms <= left_seed.time_ms:
                    continue
                pair = _PairSeed(
                    left_time_ms=left_seed.time_ms,
                    right_time_ms=right_seed.time_ms,
                    rank_score=(
                        raw_run.summed_strength
                        + left_seed.rank_score
                        + right_seed.rank_score
                    ),
                    source="regional_raw_run_closed_aba",
                )
                score = _regional_closed_aba_pair_score(
                    pair,
                    observations=observations,
                    config=config,
                )
                if score is not None:
                    raw_closed_pairs.append((score, pair))
    raw_closed_pairs.sort(
        key=lambda value: (-value[0], -value[1].rank_score, value[1].left_time_ms)
    )
    plausible_pairs = tuple(ranked_boundary_pairs) + tuple(raw_closed_pairs[:8])
    for base in bases[:2]:
        for _score, pair in plausible_pairs:
            _append_observation_side_pair_proposals(
                (
                    raw_closed_aba_proposals
                    if pair.source == "regional_raw_run_closed_aba"
                    else closed_aba_proposals
                ),
                seen=seen,
                signal=signal,
                frame_rate_hz=frame_rate_hz,
                duration_ms=duration_ms,
                global_candidates=global_candidates,
                base=base,
                observations=observations,
                pair=pair,
                config=config,
            )
    retained_single = _best_unique_regional_proposals(
        tuple(proposals),
        cap=_EXP026_RECOMBINED_PROPOSAL_CAP,
    )
    retained_closed_aba = (
        ()
        if raw_closed_aba_proposals
        else _best_unique_regional_proposals(
            tuple(closed_aba_proposals),
            cap=2,
        )
    )
    retained_raw_closed_aba = _best_closed_aba_scale_proposals(
        tuple(raw_closed_aba_proposals),
        reference_bpm=bases[0].bpm,
        cap=8,
    )
    return retained_single + retained_closed_aba + retained_raw_closed_aba


def _regional_closed_aba_pair_score(
    pair: _PairSeed,
    *,
    observations: tuple[LocalTempoObservation, ...],
    config: TempoTrackConfig,
) -> float | None:
    duration_ms = pair.right_time_ms - pair.left_time_ms
    if not (
        1000.0 * config.minimum_excursion_seconds
        <= duration_ms
        <= 1000.0 * _EXP017_LONG_ABA_MAX_SECONDS
    ):
        return None
    shoulder_ms = config.local_window_seconds * 1000.0
    left_pool = _regional_tempo_pool(
        tuple(
            observation
            for observation in observations
            if pair.left_time_ms - shoulder_ms <= observation.center_time_ms
            and observation.window_end_ms <= pair.left_time_ms
        ),
        base=None,
        include_base=False,
        fallback_base=False,
        config=config,
    )
    middle_pool = _regional_tempo_pool(
        tuple(
            observation
            for observation in observations
            if observation.window_start_ms >= pair.left_time_ms
            and observation.window_end_ms <= pair.right_time_ms
        ),
        base=None,
        include_base=False,
        fallback_base=False,
        config=config,
    )
    right_pool = _regional_tempo_pool(
        tuple(
            observation
            for observation in observations
            if observation.window_start_ms >= pair.right_time_ms
            and observation.center_time_ms <= pair.right_time_ms + shoulder_ms
        ),
        base=None,
        include_base=False,
        fallback_base=False,
        config=config,
    )
    triples = _regional_closed_aba_mode_triples(
        left_pool,
        middle_pool,
        right_pool,
        max_triples=12,
        config=config,
    )
    if not triples:
        return None
    return float(triples[0][3])


def _boundary_brackets(
    boundaries: tuple[_BoundarySeed, ...],
    *,
    target_time_ms: float,
) -> tuple[_BoundarySeed, ...]:
    before = tuple(
        boundary
        for boundary in boundaries
        if boundary.time_ms <= target_time_ms
        and target_time_ms - boundary.time_ms <= _EXP026_MODE_CHANGE_BOUNDARY_RADIUS_MS
    )
    after = tuple(
        boundary
        for boundary in boundaries
        if boundary.time_ms >= target_time_ms
        and boundary.time_ms - target_time_ms <= _EXP026_MODE_CHANGE_BOUNDARY_RADIUS_MS
    )
    values = (
        (() if not before else (max(before, key=lambda value: value.time_ms),))
        + (() if not after else (min(after, key=lambda value: value.time_ms),))
    )
    return tuple(dict.fromkeys(values))


def _observation_side_segment_seeds(
    observations: tuple[LocalTempoObservation, ...],
    *,
    boundary_seeds: tuple[_BoundarySeed, ...],
    config: TempoTrackConfig,
) -> tuple[_SideSegmentSeed, ...]:
    ranked_boundaries = sorted(
        boundary_seeds,
        key=lambda value: (-value.rank_score, value.time_ms),
    )[:16]
    boundary_only = [
        _SideSegmentSeed(
            time_ms=boundary.time_ms,
            rank_score=boundary.rank_score,
            source="boundary",
        )
        for boundary in ranked_boundaries
    ]
    mode_related: list[_SideSegmentSeed] = []
    mode_regions = _observation_mode_change_regions(
        observations,
        boundary_seeds=boundary_seeds,
        config=config,
    )
    hop_ms = 1000.0 * config.local_hop_seconds
    for region_rank, (midpoint_ms, score) in enumerate(
        sorted(mode_regions, key=lambda value: (-value[1], value[0]))
    ):
        offsets = (-hop_ms, 0.0, hop_ms) if region_rank < 8 else (-hop_ms,)
        for offset_ms in offsets:
            time_ms = midpoint_ms + offset_ms
            if time_ms <= 0.0:
                continue
            mode_related.append(
                _SideSegmentSeed(
                    time_ms=time_ms,
                    rank_score=score,
                    source="mode_change_midpoint",
                )
            )

    deduped: dict[int, _SideSegmentSeed] = {}
    # Mode-change hypotheses and the global boundary ranking live on
    # different score scales.  Preserve a fixed quota for the former instead
    # of letting large boundary scores evict every local-tempo transition.
    for seed in mode_related + boundary_only:
        key = int(round(seed.time_ms / 20.0))
        previous = deduped.get(key)
        if previous is None:
            deduped[key] = seed
    return tuple(
        list(deduped.values())[:_EXP026_SIDE_SEGMENT_SEED_CAP]
    )


def _observation_mode_change_regions(
    observations: tuple[LocalTempoObservation, ...],
    *,
    boundary_seeds: tuple[_BoundarySeed, ...],
    config: TempoTrackConfig,
) -> tuple[tuple[float, float], ...]:
    del boundary_seeds
    shoulder_ms = 1000.0 * _EXP026_MODE_CHANGE_REGION_SECONDS
    selected: list[tuple[float, float]] = []
    suppression_ms = 1000.0 * config.local_window_seconds
    per_source_cap = max(1, _EXP026_MODE_CHANGE_REGION_CAP // 2)
    for source in ("beatthis", "raw_audio"):
        source_observations = tuple(
            observation for observation in observations if observation.source == source
        )
        center_times = tuple(
            sorted(
                {
                    round(observation.center_time_ms, 6)
                    for observation in source_observations
                }
            )
        )
        candidates: list[tuple[float, float]] = []
        for center_time_ms in center_times:
            left = tuple(
                observation
                for observation in source_observations
                if center_time_ms - shoulder_ms <= observation.center_time_ms
                and observation.window_end_ms <= center_time_ms
            )
            right = tuple(
                observation
                for observation in source_observations
                if observation.window_start_ms >= center_time_ms
                and observation.center_time_ms <= center_time_ms + shoulder_ms
            )
            left_pool = _regional_tempo_pool(
                left,
                base=None,
                include_base=False,
                fallback_base=False,
                config=config,
            )
            right_pool = _regional_tempo_pool(
                right,
                base=None,
                include_base=False,
                fallback_base=False,
                config=config,
            )
            pairs = _regional_mode_pairs(
                left_pool,
                right_pool,
                max_pairs=1,
                config=config,
            )
            if not pairs:
                continue
            left_mode, right_mode, _support = pairs[0]
            relative_contrast = abs(math.log(right_mode.bpm / left_mode.bpm))
            score = float(
                relative_contrast
                + 0.01 * min(left_mode.support, right_mode.support)
            )
            candidates.append((float(center_time_ms), score))

        source_selected: list[tuple[float, float]] = []
        for center_time_ms, score in sorted(
            candidates,
            key=lambda value: (-value[1], value[0]),
        ):
            if any(
                abs(center_time_ms - existing[0]) <= suppression_ms
                for existing in source_selected
            ):
                continue
            source_selected.append((center_time_ms, score))
            if len(source_selected) >= per_source_cap:
                break
        selected.extend(source_selected)

    deduped: dict[int, tuple[float, float]] = {}
    for center_time_ms, score in selected:
        key = int(round(center_time_ms / 20.0))
        previous = deduped.get(key)
        if previous is None or score > previous[1]:
            deduped[key] = (center_time_ms, score)
    return tuple(
        sorted(deduped.values(), key=lambda value: value[0])[
            :_EXP026_MODE_CHANGE_REGION_CAP
        ]
    )


def _append_observation_side_single_boundary_proposals(
    proposals: list[_CurveProposal],
    *,
    seen: set[str],
    signal: NDArray[np.float64],
    frame_rate_hz: float,
    duration_ms: float,
    global_candidates: _global.GlobalConstantJumpCandidateSet,
    base: _BaseHypothesis,
    observations: tuple[LocalTempoObservation, ...],
    seed: _SideSegmentSeed,
    config: TempoTrackConfig,
) -> None:
    shoulder_ms = 3.0 * config.local_window_seconds * 1000.0
    left_observations = tuple(
        observation
        for observation in observations
        if seed.time_ms - shoulder_ms <= observation.center_time_ms
        and observation.window_end_ms <= seed.time_ms
    )
    right_observations = tuple(
        observation
        for observation in observations
        if observation.window_start_ms >= seed.time_ms
        and observation.center_time_ms <= seed.time_ms + shoulder_ms
    )
    if not left_observations:
        left_observations = tuple(
            observation
            for observation in observations
            if seed.time_ms - shoulder_ms <= observation.center_time_ms < seed.time_ms
        )
    if not right_observations:
        right_observations = tuple(
            observation
            for observation in observations
            if seed.time_ms < observation.center_time_ms <= seed.time_ms + shoulder_ms
        )
    left_pool = _regional_tempo_pool(
        left_observations,
        base=base,
        include_base=True,
        fallback_base=False,
        config=config,
    )
    right_pool = _regional_tempo_pool(
        right_observations,
        base=base,
        include_base=False,
        fallback_base=True,
        config=config,
    )
    for left_mode, right_mode, regional_support in _regional_mode_pairs(
        left_pool,
        right_pool,
        max_pairs=8,
        config=config,
    ):
        if not _regional_left_mode_matches_base(
            left_mode,
            base=base,
            config=config,
        ):
            continue
        boundary = _regional_snapped_boundary(
            global_candidates=global_candidates,
            signal=signal,
            frame_rate_hz=frame_rate_hz,
            duration_ms=duration_ms,
            base=base,
            left_mode=left_mode,
            target_time_ms=seed.time_ms,
            config=config,
        )
        if boundary is None:
            continue
        boundary_beat, boundary_time_ms, origin_time_ms = boundary
        if abs(boundary_time_ms - seed.time_ms) > _EXP026_BOUNDARY_TOLERANCE_MS:
            continue
        end_beat = _terminal_beat_after_boundary(
            boundary_beat=boundary_beat,
            boundary_time_ms=boundary_time_ms,
            duration_ms=duration_ms,
            bpm=right_mode.bpm,
        )
        _append_observation_side_proposal(
            proposals,
            seen=seen,
            signal=signal,
            frame_rate_hz=frame_rate_hz,
            duration_ms=duration_ms,
            origin_time_ms=origin_time_ms,
            sections=(
                ConstantTempoSection(0, boundary_beat, left_mode.bpm),
                ConstantTempoSection(boundary_beat, end_beat, right_mode.bpm),
            ),
            rank_score=seed.rank_score,
            regional_support=regional_support,
            collapse_bpm=left_mode.bpm,
            config=config,
        )


def _append_observation_side_pair_proposals(
    proposals: list[_CurveProposal],
    *,
    seen: set[str],
    signal: NDArray[np.float64],
    frame_rate_hz: float,
    duration_ms: float,
    global_candidates: _global.GlobalConstantJumpCandidateSet,
    base: _BaseHypothesis,
    observations: tuple[LocalTempoObservation, ...],
    pair: _PairSeed,
    config: TempoTrackConfig,
) -> None:
    shoulder_ms = config.local_window_seconds * 1000.0
    left_observations = tuple(
        observation
        for observation in observations
        if pair.left_time_ms - shoulder_ms <= observation.center_time_ms
        and observation.window_end_ms <= pair.left_time_ms
    )
    middle_observations = tuple(
        observation
        for observation in observations
        if observation.window_start_ms >= pair.left_time_ms
        and observation.window_end_ms <= pair.right_time_ms
    )
    right_observations = tuple(
        observation
        for observation in observations
        if observation.window_start_ms >= pair.right_time_ms
        and observation.center_time_ms <= pair.right_time_ms + shoulder_ms
    )
    left_pool = _regional_tempo_pool(
        left_observations,
        base=base,
        include_base=True,
        fallback_base=False,
        config=config,
    )
    middle_pool = _regional_tempo_pool(
        middle_observations,
        base=None,
        include_base=False,
        fallback_base=False,
        config=config,
    )
    right_pool = _regional_tempo_pool(
        right_observations,
        base=base,
        include_base=False,
        fallback_base=True,
        config=config,
    )
    triples = _regional_closed_aba_mode_triples(
        left_pool,
        middle_pool,
        right_pool,
        max_triples=12,
        config=config,
    )
    expanded_triples: list[
        tuple[_RegionalTempoMode, _RegionalTempoMode, _RegionalTempoMode, float]
    ] = []
    for left_mode, middle_mode, right_mode, regional_support in triples:
        expanded_triples.append(
            (left_mode, middle_mode, right_mode, regional_support)
        )
        nominal_outer_bpm = float(round(left_mode.bpm))
        nominal_middle_bpm = float(round(middle_mode.bpm))
        if (
            _EXP026_LOCAL_ALIAS_MIN_BPM <= nominal_outer_bpm <= _EXP026_LOCAL_ALIAS_MAX_BPM
            and _EXP026_LOCAL_ALIAS_MIN_BPM
            <= nominal_middle_bpm
            <= _EXP026_LOCAL_ALIAS_MAX_BPM
            and _regional_adjacent_tempos_differ(
                nominal_outer_bpm,
                nominal_middle_bpm,
                config=config,
            )
        ):
            expanded_triples.append(
                (
                    _RegionalTempoMode(
                        source=left_mode.source,
                        bpm=nominal_outer_bpm,
                        support=left_mode.support,
                        alias_multiplier=left_mode.alias_multiplier,
                    ),
                    _RegionalTempoMode(
                        source=middle_mode.source,
                        bpm=nominal_middle_bpm,
                        support=middle_mode.support,
                        alias_multiplier=middle_mode.alias_multiplier,
                    ),
                    _RegionalTempoMode(
                        source=right_mode.source,
                        bpm=nominal_outer_bpm,
                        support=right_mode.support,
                        alias_multiplier=right_mode.alias_multiplier,
                    ),
                    regional_support,
                )
            )
    for left_mode, middle_mode, right_mode, regional_support in expanded_triples:
        boundary = _regional_snapped_boundary(
            global_candidates=global_candidates,
            signal=signal,
            frame_rate_hz=frame_rate_hz,
            duration_ms=duration_ms,
            base=base,
            left_mode=left_mode,
            target_time_ms=pair.left_time_ms,
            config=config,
        )
        if boundary is None:
            continue
        first_beat, first_time_ms, origin_time_ms = boundary
        if abs(first_time_ms - pair.left_time_ms) > _EXP026_BOUNDARY_TOLERANCE_MS:
            continue
        middle_beats = int(
            round(
                max(
                    1e-9,
                    (pair.right_time_ms - pair.left_time_ms)
                    * middle_mode.bpm
                    / 60000.0,
                )
            )
        )
        if middle_beats <= 0:
            continue
        final_beat = first_beat + middle_beats
        middle_end_time_ms = (
            first_time_ms + 60000.0 * middle_beats / middle_mode.bpm
        )
        if abs(middle_end_time_ms - pair.right_time_ms) > _EXP026_BOUNDARY_TOLERANCE_MS:
            continue
        if middle_end_time_ms <= first_time_ms or middle_end_time_ms >= duration_ms:
            continue
        end_beat = _terminal_beat_after_boundary(
            boundary_beat=final_beat,
            boundary_time_ms=middle_end_time_ms,
            duration_ms=duration_ms,
            bpm=right_mode.bpm,
        )
        _append_observation_side_proposal(
            proposals,
            seen=seen,
            signal=signal,
            frame_rate_hz=frame_rate_hz,
            duration_ms=duration_ms,
            origin_time_ms=origin_time_ms,
            sections=(
                ConstantTempoSection(0, first_beat, left_mode.bpm),
                ConstantTempoSection(first_beat, final_beat, middle_mode.bpm),
                ConstantTempoSection(final_beat, end_beat, right_mode.bpm),
            ),
            rank_score=pair.rank_score,
            regional_support=regional_support,
            collapse_bpm=left_mode.bpm,
            config=config,
        )


def _append_observation_side_proposal(
    proposals: list[_CurveProposal],
    *,
    seen: set[str],
    signal: NDArray[np.float64],
    frame_rate_hz: float,
    duration_ms: float,
    origin_time_ms: float,
    sections: tuple[ConstantTempoSection, ...],
    rank_score: float,
    regional_support: float,
    collapse_bpm: float,
    config: TempoTrackConfig,
) -> None:
    try:
        curve = PhaseContinuousTimingCurve(
            origin_beat=sections[0].start_beat,
            origin_time_ms=origin_time_ms,
            sections=sections,
        )
    except ValueError:
        return
    if curve.end_time_ms + 1e-6 < duration_ms:
        return
    if any(report.phase_discontinuity_ms != 0.0 for report in curve.seam_reports):
        return
    fingerprint = curve.fingerprint_sha256
    if fingerprint in seen:
        return
    seen.add(fingerprint)
    collapsed = _collapsed_constant_counterfactual(
        curve,
        duration_ms=duration_ms,
        base_bpm=collapse_bpm,
        config=config,
    )
    support_delta = _generalized_support_delta(
        signal,
        frame_rate_hz=frame_rate_hz,
        candidate=curve,
        collapsed=collapsed,
        radius_ms=config.beat_event_radius_ms,
    )
    proposals.append(
        _CurveProposal(
            curve=curve,
            source=_EXP026_SIDE_SEGMENT_SOURCE,
            score=float(
                support_delta
                + 0.04 * math.log1p(max(0.0, rank_score))
                + regional_support
            ),
            collapse_bpm=float(collapse_bpm),
        )
    )


def _regional_snapped_boundary(
    *,
    global_candidates: _global.GlobalConstantJumpCandidateSet,
    signal: NDArray[np.float64],
    frame_rate_hz: float,
    duration_ms: float,
    base: _BaseHypothesis,
    left_mode: _RegionalTempoMode,
    target_time_ms: float,
    config: TempoTrackConfig,
) -> tuple[int, float, float] | None:
    origin_time_ms = (
        base.origin_time_ms
        if left_mode.source == "base" or abs(left_mode.bpm - base.bpm) < 1e-9
        else _phase_origin_for_bpm(
            global_candidates,
            bpm=left_mode.bpm,
            beat_signal=signal,
            frame_rate_hz=frame_rate_hz,
            duration_ms=duration_ms,
            radius_ms=config.beat_event_radius_ms,
        )
    )
    snapped = _snapped_constant_boundary(
        origin_time_ms=origin_time_ms,
        bpm=left_mode.bpm,
        target_time_ms=target_time_ms,
        duration_ms=duration_ms,
    )
    if snapped is None:
        return None
    beat, time_ms = snapped
    return beat, time_ms, origin_time_ms


def _regional_tempo_pool(
    observations: tuple[LocalTempoObservation, ...],
    *,
    base: _BaseHypothesis | None,
    include_base: bool,
    fallback_base: bool,
    config: TempoTrackConfig,
) -> tuple[_RegionalTempoMode, ...]:
    modes: list[_RegionalTempoMode] = []
    if include_base and base is not None:
        modes.append(
            _RegionalTempoMode(
                source="base",
                bpm=base.bpm,
                support=0.0,
                alias_multiplier=1.0,
            )
        )
    modes.extend(_regional_observation_modes(observations, config=config))
    if not modes and fallback_base and base is not None:
        modes.append(
            _RegionalTempoMode(
                source="base",
                bpm=base.bpm,
                support=0.0,
                alias_multiplier=1.0,
            )
        )

    deduped: dict[int, _RegionalTempoMode] = {}
    for mode in modes:
        key = int(round(mode.bpm * 1000.0))
        previous = deduped.get(key)
        if previous is None or (
            mode.support,
            _regional_mode_source_priority(mode.source),
        ) > (
            previous.support,
            _regional_mode_source_priority(previous.source),
        ):
            deduped[key] = mode
    return tuple(
        sorted(
            deduped.values(),
            key=lambda value: (
                -value.support,
                -_regional_mode_source_priority(value.source),
                value.bpm,
            ),
        )
    )


def _regional_observation_modes(
    observations: tuple[LocalTempoObservation, ...],
    *,
    config: TempoTrackConfig,
) -> tuple[_RegionalTempoMode, ...]:
    request_cache = _REGIONAL_OBSERVATION_MODES_REQUEST_CACHE.get()
    cache_key: tuple[object, ...] | None = None
    if request_cache is not None:
        cache_key = (
            observations,
            float(config.minimum_local_strength).hex(),
        )
        cached = request_cache.get(cache_key)
        if cached is not None:
            return cached

    modes: list[_RegionalTempoMode] = []
    for source in ("beatthis", "raw_audio"):
        source_observations = tuple(
            observation
            for observation in observations
            if observation.source == source
            and observation.strength >= config.minimum_local_strength
        )
        if not source_observations:
            continue
        aggregates: dict[int, list[tuple[float, float, float]]] = {}
        for observation in source_observations:
            canonical_bpm = _regional_canonical_bpm(observation.bpm)
            if canonical_bpm is None:
                continue
            key = int(round(canonical_bpm * 2.0))
            aggregates.setdefault(key, []).append(
                (canonical_bpm, observation.bpm, observation.strength)
            )
        if not aggregates:
            continue
        ranked_values = sorted(
            aggregates.values(),
            key=lambda values: (
                -sum(strength for _, _, strength in values),
                -len(values),
                -sum(
                    canonical_bpm * strength
                    for canonical_bpm, _, strength in values
                )
                / max(sum(strength for _, _, strength in values), 1e-12),
            ),
        )[:2]
        for best_values in ranked_values:
            support = float(sum(strength for _, _, strength in best_values))
            observed_bpm = float(
                sum(
                    canonical_bpm * strength
                    for canonical_bpm, _, strength in best_values
                )
                / max(support, 1e-12)
            )
            for bpm, multiplier in _regional_octave_aliases_with_multipliers(
                observed_bpm
            ):
                modes.append(
                    _RegionalTempoMode(
                        source=source,
                        bpm=bpm,
                        support=_regional_mode_support_bonus(source, support),
                        alias_multiplier=multiplier,
                    )
                )
    result = tuple(modes)
    if request_cache is not None and cache_key is not None:
        request_cache[cache_key] = result
    return result


def _regional_canonical_bpm(bpm: float) -> float | None:
    aliases = _regional_octave_aliases(bpm)
    if not aliases:
        return None
    return min(aliases, key=lambda value: (abs(value - 150.0), -value))


def _regional_octave_aliases(bpm: float) -> tuple[float, ...]:
    return tuple(
        candidate
        for candidate, _ in _regional_octave_aliases_with_multipliers(bpm)
    )


def _regional_octave_aliases_with_multipliers(
    bpm: float,
) -> tuple[tuple[float, float], ...]:
    aliases: list[float] = []
    values: list[tuple[float, float]] = []
    for multiplier in (0.5, 1.0, 2.0):
        candidate = bpm * multiplier
        if _EXP026_LOCAL_ALIAS_MIN_BPM <= candidate <= _EXP026_LOCAL_ALIAS_MAX_BPM:
            rounded = float(candidate)
            if not any(abs(rounded - existing) < 1e-9 for existing in aliases):
                aliases.append(rounded)
                values.append((rounded, multiplier))
    return tuple(values)


def _regional_mode_pairs(
    left_pool: tuple[_RegionalTempoMode, ...],
    right_pool: tuple[_RegionalTempoMode, ...],
    *,
    max_pairs: int,
    config: TempoTrackConfig,
) -> tuple[tuple[_RegionalTempoMode, _RegionalTempoMode, float], ...]:
    pairs: list[tuple[_RegionalTempoMode, _RegionalTempoMode, float]] = []
    for left in left_pool:
        for right in right_pool:
            tempo_ratio = max(left.bpm, right.bpm) / min(left.bpm, right.bpm)
            if (
                not _regional_alias_multipliers_match(left, right)
                and tempo_ratio > 1.6
            ):
                continue
            if not _regional_adjacent_tempos_differ(left.bpm, right.bpm, config=config):
                continue
            pairs.append((left, right, left.support + right.support))
    pairs.sort(
        key=lambda value: (
            -_regional_pair_alias_consistency(value[0], value[1]),
            -value[2],
            -min(value[0].bpm, value[1].bpm),
            _regional_mode_source_sort_key(value[0]),
            _regional_mode_source_sort_key(value[1]),
            value[0].bpm,
            value[1].bpm,
        )
    )
    return tuple(pairs[:max_pairs])


def _regional_mode_triples(
    left_pool: tuple[_RegionalTempoMode, ...],
    middle_pool: tuple[_RegionalTempoMode, ...],
    right_pool: tuple[_RegionalTempoMode, ...],
    *,
    max_triples: int,
    config: TempoTrackConfig,
) -> tuple[tuple[_RegionalTempoMode, _RegionalTempoMode, _RegionalTempoMode, float], ...]:
    triples: list[
        tuple[_RegionalTempoMode, _RegionalTempoMode, _RegionalTempoMode, float]
    ] = []
    for left in left_pool:
        for middle in middle_pool:
            if not _regional_closed_aba_adjacent_tempos_differ(
                left.bpm,
                middle.bpm,
                config=config,
            ):
                continue
            if max(left.bpm, middle.bpm) / min(left.bpm, middle.bpm) > 1.5:
                continue
            for right in right_pool:
                outer_tolerance = max(1.0, 0.01 * max(left.bpm, right.bpm))
                if abs(left.bpm - right.bpm) > outer_tolerance:
                    continue
                if not _regional_closed_aba_adjacent_tempos_differ(
                    middle.bpm,
                    right.bpm,
                    config=config,
                ):
                    continue
                if max(middle.bpm, right.bpm) / min(middle.bpm, right.bpm) > 1.5:
                    continue
                triples.append(
                    (
                        left,
                        middle,
                        right,
                        left.support + middle.support + right.support,
                    )
                )
    triples.sort(
        key=lambda value: (
            -_regional_triple_alias_consistency(value[0], value[1], value[2]),
            -value[3],
            -min(value[0].bpm, value[1].bpm, value[2].bpm),
            _regional_mode_source_sort_key(value[0]),
            _regional_mode_source_sort_key(value[1]),
            _regional_mode_source_sort_key(value[2]),
            value[0].bpm,
            value[1].bpm,
            value[2].bpm,
        )
    )
    return tuple(triples[:max_triples])


def _regional_closed_aba_mode_triples(
    left_pool: tuple[_RegionalTempoMode, ...],
    middle_pool: tuple[_RegionalTempoMode, ...],
    right_pool: tuple[_RegionalTempoMode, ...],
    *,
    max_triples: int,
    config: TempoTrackConfig,
) -> tuple[
    tuple[_RegionalTempoMode, _RegionalTempoMode, _RegionalTempoMode, float], ...
]:
    all_triples = _regional_mode_triples(
        left_pool,
        middle_pool,
        right_pool,
        max_triples=512,
        config=config,
    )
    best_by_alias_family: dict[
        tuple[float, float, float],
        tuple[_RegionalTempoMode, _RegionalTempoMode, _RegionalTempoMode, float],
    ] = {}
    for triple in all_triples:
        key = (
            triple[0].alias_multiplier,
            triple[1].alias_multiplier,
            triple[2].alias_multiplier,
        )
        best_by_alias_family.setdefault(key, triple)
    return tuple(
        sorted(
            best_by_alias_family.values(),
            key=lambda value: (
                -value[3],
                -min(value[0].bpm, value[1].bpm, value[2].bpm),
                value[0].bpm,
                value[1].bpm,
                value[2].bpm,
            ),
        )[:max_triples]
    )


def _regional_closed_aba_adjacent_tempos_differ(
    left_bpm: float,
    right_bpm: float,
    *,
    config: TempoTrackConfig,
) -> bool:
    if not _regional_adjacent_tempos_differ(left_bpm, right_bpm, config=config):
        return False
    ratio = max(left_bpm, right_bpm) / min(left_bpm, right_bpm)
    # Stable tracks can expose a closed A-B-A hallucination when the local
    # estimator switches briefly to the three-against-two subdivision.  Keep
    # this guard local to the closed-ABA lane so a real persistent 3:2 tempo
    # change remains eligible in the two-section lane.
    return abs(ratio - 3.0 / 2.0) > 0.025 * (3.0 / 2.0)


def _regional_adjacent_tempos_differ(
    left_bpm: float,
    right_bpm: float,
    *,
    config: TempoTrackConfig,
) -> bool:
    # The latest Exp026 probe prioritizes stable guards for the side lane, so
    # this bounded path currently keeps the production 5 BPM floor.
    if _regional_tempos_are_octave_aliases(left_bpm, right_bpm):
        return False
    ratio = max(left_bpm, right_bpm) / min(left_bpm, right_bpm)
    # Local periodicity frequently flips between three- and four-subdivision
    # interpretations on otherwise stable tracks.  Treat that exact 4:3
    # family as a rhythmic alias, not a structural tempo change.
    if abs(ratio - 4.0 / 3.0) <= 0.025 * (4.0 / 3.0):
        return False
    return abs(left_bpm - right_bpm) >= config.minimum_jump_bpm


def _regional_left_mode_matches_base(
    mode: _RegionalTempoMode,
    *,
    base: _BaseHypothesis,
    config: TempoTrackConfig,
) -> bool:
    # A global hypothesis may describe the ending or dominant section rather
    # than the section to the left of a real boundary.  A locally observed
    # mode therefore carries its own tempo/phase hypothesis and must not be
    # rejected merely because it differs from the global constant.
    if mode.source != "base":
        return mode.support > 0.0
    tolerance_bpm = max(1.0, 0.02 * base.bpm)
    return abs(mode.bpm - base.bpm) <= tolerance_bpm


def _regional_tempos_are_octave_aliases(left_bpm: float, right_bpm: float) -> bool:
    if left_bpm <= 0.0 or right_bpm <= 0.0:
        return False
    ratio = max(left_bpm, right_bpm) / min(left_bpm, right_bpm)
    return abs(ratio - 2.0) <= 0.025


def _regional_mode_support_bonus(source: str, support: float) -> float:
    if source == "raw_audio":
        return 0.24 * support
    if source == "beatthis":
        return 0.12 * support
    return 0.0


def _regional_alias_multipliers_match(
    left: _RegionalTempoMode,
    right: _RegionalTempoMode,
) -> bool:
    return abs(left.alias_multiplier - right.alias_multiplier) < 1e-9


def _regional_pair_alias_consistency(
    left: _RegionalTempoMode,
    right: _RegionalTempoMode,
) -> int:
    if left.source == "base" or right.source == "base":
        return 1
    return int(abs(left.alias_multiplier - right.alias_multiplier) < 1e-9)


def _regional_triple_alias_consistency(
    left: _RegionalTempoMode,
    middle: _RegionalTempoMode,
    right: _RegionalTempoMode,
) -> int:
    multipliers = (
        left.alias_multiplier,
        middle.alias_multiplier,
        right.alias_multiplier,
    )
    if left.source == "base":
        return int(abs(middle.alias_multiplier - right.alias_multiplier) < 1e-9)
    return int(max(multipliers) - min(multipliers) < 1e-9)


def _regional_mode_source_priority(source: str) -> int:
    if source == "raw_audio":
        return 2
    if source == "beatthis":
        return 1
    return 0


def _regional_mode_source_sort_key(mode: _RegionalTempoMode) -> tuple[int, str]:
    return (-_regional_mode_source_priority(mode.source), mode.source)


def _best_unique_regional_proposals(
    proposals: tuple[_CurveProposal, ...],
    *,
    cap: int,
) -> tuple[_CurveProposal, ...]:
    retained: list[_CurveProposal] = []
    seen: set[str] = set()
    for proposal in sorted(proposals, key=_jump_retention_order_key):
        fingerprint = proposal.curve.fingerprint_sha256
        if fingerprint in seen:
            continue
        retained.append(proposal)
        seen.add(fingerprint)
        if len(retained) >= cap:
            break
    return tuple(retained)


def _best_closed_aba_scale_proposals(
    proposals: tuple[_CurveProposal, ...],
    *,
    reference_bpm: float,
    cap: int,
) -> tuple[_CurveProposal, ...]:
    ordered = _best_unique_regional_proposals(proposals, cap=max(cap, len(proposals)))
    retained: list[_CurveProposal] = []
    seen: set[str] = set()

    def append(proposal: _CurveProposal) -> None:
        fingerprint = proposal.curve.fingerprint_sha256
        if fingerprint not in seen and len(retained) < cap:
            retained.append(proposal)
            seen.add(fingerprint)

    scale_keys = tuple(
        sorted(
            {
                int(round(math.log2(proposal.curve.sections[0].bpm / reference_bpm)))
                for proposal in ordered
            }
        )
    )
    for require_nominal in (True, False):
        for scale_key in scale_keys:
            for proposal in ordered:
                sections = proposal.curve.sections
                proposal_scale = int(
                    round(math.log2(sections[0].bpm / reference_bpm))
                )
                nominal = (
                    len(sections) == 3
                    and all(abs(section.bpm - round(section.bpm)) < 1e-9 for section in sections)
                    and abs(sections[0].bpm - sections[2].bpm) < 1e-9
                )
                if proposal_scale == scale_key and nominal == require_nominal:
                    append(proposal)
                    break
    for proposal in ordered:
        append(proposal)
    return tuple(retained)


def _retain_jump_proposals_by_family(
    proposals: tuple[_CurveProposal, ...],
    *,
    config: TempoTrackConfig,
) -> _JumpProposalBatch:
    cap = min(
        config.maximum_jump_candidates,
        sum(quota for _, quota in _JUMP_RETENTION_FAMILY_QUOTAS),
    )
    global_order = _deduped_jump_retention_order(proposals)
    family_order: dict[_JumpRetentionFamily, tuple[_CurveProposal, ...]] = {}
    for family, _ in _JUMP_RETENTION_FAMILY_QUOTAS:
        family_proposals = tuple(
            proposal
            for proposal in global_order
            if _jump_retention_family(proposal, config=config) == family
        )
        if family == "short_aba_paired_boundary":
            family_proposals = _short_aba_pareto_support_retention_order(
                family_proposals
            )
        elif family == "long_aba":
            family_proposals = tuple(
                sorted(
                    family_proposals,
                    key=lambda proposal: (
                        0
                        if proposal.source == _EXP026_SIDE_SEGMENT_SOURCE
                        and len(proposal.curve.sections) == 3
                        else 1,
                        _jump_retention_order_key(proposal),
                    ),
                )
            )
        family_order[family] = family_proposals
    retained: list[_CurveProposal] = []
    retained_fingerprints: set[str] = set()

    def append_unique(proposal: _CurveProposal) -> bool:
        if len(retained) >= cap:
            return False
        fingerprint = proposal.curve.fingerprint_sha256
        if fingerprint in retained_fingerprints:
            return False
        retained.append(proposal)
        retained_fingerprints.add(fingerprint)
        return True

    for family, quota in _JUMP_RETENTION_FAMILY_QUOTAS:
        family_retained = 0
        for proposal in family_order[family]:
            if family_retained >= quota:
                break
            if append_unique(proposal):
                family_retained += 1

    for proposal in global_order:
        if len(retained) >= cap:
            break
        append_unique(proposal)
    pruning_reason = (
        f"maximum_jump_candidates_{config.maximum_jump_candidates}"
        if len(global_order) > len(retained)
        else None
    )
    return _JumpProposalBatch(tuple(retained), pruning_reason)


def _deduped_jump_retention_order(
    proposals: tuple[_CurveProposal, ...],
) -> tuple[_CurveProposal, ...]:
    retained: list[_CurveProposal] = []
    seen: set[str] = set()
    for proposal in sorted(proposals, key=_jump_retention_order_key):
        fingerprint = proposal.curve.fingerprint_sha256
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        retained.append(proposal)
    return tuple(retained)


def _jump_retention_order_key(
    proposal: _CurveProposal,
) -> tuple[float, str, str, tuple[float, ...]]:
    return (
        -proposal.score,
        proposal.source,
        proposal.curve.fingerprint_sha256,
        proposal.curve.boundary_times_ms,
    )


def _short_aba_support_retention_order_key(
    proposal: _CurveProposal,
) -> tuple[int, float, float, str, str, tuple[float, ...]]:
    support_delta = proposal.aba_support_delta
    has_finite_support = support_delta is not None and math.isfinite(support_delta)
    return (
        0 if has_finite_support else 1,
        -float(support_delta) if has_finite_support else 0.0,
        -proposal.score,
        proposal.source,
        proposal.curve.fingerprint_sha256,
        proposal.curve.boundary_times_ms,
    )


def _short_aba_pareto_support_retention_order(
    proposals: tuple[_CurveProposal, ...],
) -> tuple[_CurveProposal, ...]:
    support_order = tuple(
        sorted(proposals, key=_short_aba_support_retention_order_key)
    )
    first_front = _short_aba_first_pareto_front(proposals)

    retained: list[_CurveProposal] = []
    retained_fingerprints: set[str] = set()

    def append_unseen(proposal: _CurveProposal) -> None:
        fingerprint = proposal.curve.fingerprint_sha256
        if fingerprint in retained_fingerprints:
            return
        retained.append(proposal)
        retained_fingerprints.add(fingerprint)

    for proposal in sorted(first_front, key=_short_aba_support_retention_order_key):
        append_unseen(proposal)
    for proposal in support_order:
        append_unseen(proposal)
    return tuple(retained)


def _short_aba_first_pareto_front(
    proposals: tuple[_CurveProposal, ...],
) -> tuple[_CurveProposal, ...]:
    """Return the exact two-objective skyline in deterministic O(n log n)."""

    finite_order = sorted(
        (
            proposal
            for proposal in proposals
            if _has_finite_short_aba_pareto_objectives(proposal)
        ),
        key=lambda proposal: (
            -proposal.score,
            -float(proposal.aba_support_delta),
            proposal.source,
            proposal.curve.fingerprint_sha256,
            proposal.curve.boundary_times_ms,
        ),
    )
    front: list[_CurveProposal] = []
    best_support_at_higher_score = -math.inf
    group_start = 0
    while group_start < len(finite_order):
        score = finite_order[group_start].score
        group_end = group_start + 1
        while (
            group_end < len(finite_order)
            and finite_order[group_end].score == score
        ):
            group_end += 1

        group = finite_order[group_start:group_end]
        group_best_support = float(group[0].aba_support_delta)
        if group_best_support > best_support_at_higher_score:
            front.extend(
                proposal
                for proposal in group
                if float(proposal.aba_support_delta) == group_best_support
            )
        best_support_at_higher_score = max(
            best_support_at_higher_score,
            group_best_support,
        )
        group_start = group_end
    return tuple(front)


def _has_finite_short_aba_pareto_objectives(proposal: _CurveProposal) -> bool:
    support_delta = proposal.aba_support_delta
    return (
        support_delta is not None
        and math.isfinite(proposal.score)
        and math.isfinite(support_delta)
    )


def _dominates_short_aba_pareto_candidate(
    lhs: _CurveProposal,
    rhs: _CurveProposal,
) -> bool:
    lhs_support = lhs.aba_support_delta
    rhs_support = rhs.aba_support_delta
    assert lhs_support is not None
    assert rhs_support is not None
    return (
        lhs.score >= rhs.score
        and lhs_support >= rhs_support
        and (lhs.score > rhs.score or lhs_support > rhs_support)
    )


def _jump_retention_family(
    proposal: _CurveProposal,
    *,
    config: TempoTrackConfig,
) -> _JumpRetentionFamily:
    # Exp017 freezes retention taxonomy independently from proposal-generation
    # knobs so widened research configs cannot silently consume reserved slots.
    del config
    curve = proposal.curve
    if not _is_phase1_piecewise_constant_jump(curve):
        return "overflow"
    sections = curve.sections
    paired_boundary_source = (
        proposal.source == "paired_unmerged_boundary"
        or proposal.source.startswith("virtual_right_")
        or (
            proposal.source == _EXP026_SIDE_SEGMENT_SOURCE
            and len(sections) == 3
        )
    )
    if len(sections) == 3:
        middle = sections[1]
        assert isinstance(middle, ConstantTempoSection)
        if (
            paired_boundary_source
            and _EXP017_SHORT_ABA_MIN_SECONDS
            <= middle.duration_seconds
            <= _EXP017_SHORT_ABA_MAX_SECONDS
        ):
            return "short_aba_paired_boundary"
    if len(sections) == 2 and proposal.source.startswith("raw_run_persistent"):
        return "persistent"
    if len(sections) == 3:
        middle = sections[1]
        assert isinstance(middle, ConstantTempoSection)
        if (
            (paired_boundary_source or proposal.source == "raw_run_aba")
            and _EXP017_SHORT_ABA_MAX_SECONDS
            < middle.duration_seconds
            <= _EXP017_LONG_ABA_MAX_SECONDS
        ):
            return "long_aba"
    if proposal.source.startswith("raw_run_chain_") and 2 <= len(sections) - 1 <= 4:
        return "multi_step"
    return "overflow"


def _raw_run_jump_proposals(
    signal: NDArray[np.float64],
    *,
    frame_rate_hz: float,
    duration_ms: float,
    bases: tuple[_BaseHypothesis, ...],
    observations: tuple[LocalTempoObservation, ...],
    config: TempoTrackConfig,
) -> tuple[_CurveProposal, ...]:
    if not bases:
        return ()
    raw_observations = tuple(
        observation for observation in observations if observation.source == "raw_audio"
    )
    raw_runs = _raw_tempo_runs(
        raw_observations,
        base_bpm=bases[0].bpm,
        config=config,
    )
    if not raw_runs:
        return ()

    proposals: list[_CurveProposal] = []
    seen: set[str] = set()
    primary_bpm = bases[0].bpm
    proposal_bases = tuple(
        base
        for base in bases
        if _is_primary_alias_consistent(base.bpm, primary_bpm, config=config)
    )
    for base in proposal_bases:
        for run in raw_runs:
            run_bpm = _raw_run_jump_bpm(run, base_bpm=base.bpm, config=config)
            if run_bpm is None:
                continue
            _append_raw_run_persistent_proposals(
                proposals,
                seen=seen,
                signal=signal,
                frame_rate_hz=frame_rate_hz,
                duration_ms=duration_ms,
                base=base,
                raw_run=run,
                run_bpm=run_bpm,
                config=config,
            )
            _append_raw_run_aba_proposal(
                proposals,
                seen=seen,
                signal=signal,
                frame_rate_hz=frame_rate_hz,
                duration_ms=duration_ms,
                base=base,
                raw_run=run,
                run_bpm=run_bpm,
                config=config,
            )

    ordered_runs = tuple(sorted(raw_runs, key=lambda run: run.expanded_start_time_ms))
    for base in proposal_bases:
        for run_count in (2, 3):
            for start_index in range(0, len(ordered_runs) - run_count + 1):
                run_chain = ordered_runs[start_index : start_index + run_count]
                _append_raw_run_chain_proposal(
                    proposals,
                    seen=seen,
                    signal=signal,
                    frame_rate_hz=frame_rate_hz,
                    duration_ms=duration_ms,
                    base=base,
                    raw_runs=run_chain,
                    config=config,
                )

    proposals.sort(key=lambda value: (-value.score, value.curve.fingerprint_sha256))
    return tuple(proposals)


def _append_raw_run_persistent_proposals(
    proposals: list[_CurveProposal],
    *,
    seen: set[str],
    signal: NDArray[np.float64],
    frame_rate_hz: float,
    duration_ms: float,
    base: _BaseHypothesis,
    raw_run: TempoTrackRawRunDiagnostic,
    run_bpm: float,
    config: TempoTrackConfig,
) -> None:
    for anchor_name, anchor_time_ms in (
        ("start", raw_run.start_time_ms),
        ("end", raw_run.end_time_ms),
    ):
        boundary = _snapped_constant_boundary(
            origin_time_ms=base.origin_time_ms,
            bpm=base.bpm,
            target_time_ms=anchor_time_ms,
            duration_ms=duration_ms,
        )
        if boundary is not None:
            boundary_beat, boundary_time_ms = boundary
            end_beat = _terminal_beat_after_boundary(
                boundary_beat=boundary_beat,
                boundary_time_ms=boundary_time_ms,
                duration_ms=duration_ms,
                bpm=run_bpm,
            )
            _append_constant_section_proposal(
                proposals,
                seen=seen,
                signal=signal,
                frame_rate_hz=frame_rate_hz,
                duration_ms=duration_ms,
                origin_time_ms=base.origin_time_ms,
                sections=(
                    ConstantTempoSection(0, boundary_beat, base.bpm),
                    ConstantTempoSection(boundary_beat, end_beat, run_bpm),
                ),
                source=f"raw_run_persistent_a_to_b_{anchor_name}",
                source_score=raw_run.summed_strength,
                config=config,
            )

    boundary = _snapped_constant_boundary(
        origin_time_ms=base.origin_time_ms,
        bpm=run_bpm,
        target_time_ms=raw_run.end_time_ms,
        duration_ms=duration_ms,
    )
    if boundary is None:
        return
    boundary_beat, boundary_time_ms = boundary
    end_beat = _terminal_beat_after_boundary(
        boundary_beat=boundary_beat,
        boundary_time_ms=boundary_time_ms,
        duration_ms=duration_ms,
        bpm=base.bpm,
    )
    _append_constant_section_proposal(
        proposals,
        seen=seen,
        signal=signal,
        frame_rate_hz=frame_rate_hz,
        duration_ms=duration_ms,
        origin_time_ms=base.origin_time_ms,
        sections=(
            ConstantTempoSection(0, boundary_beat, run_bpm),
            ConstantTempoSection(boundary_beat, end_beat, base.bpm),
        ),
        source="raw_run_persistent_b_to_a_end",
        source_score=raw_run.summed_strength,
        config=config,
    )


def _append_raw_run_aba_proposal(
    proposals: list[_CurveProposal],
    *,
    seen: set[str],
    signal: NDArray[np.float64],
    frame_rate_hz: float,
    duration_ms: float,
    base: _BaseHypothesis,
    raw_run: TempoTrackRawRunDiagnostic,
    run_bpm: float,
    config: TempoTrackConfig,
) -> None:
    first_boundary = _snapped_constant_boundary(
        origin_time_ms=base.origin_time_ms,
        bpm=base.bpm,
        target_time_ms=raw_run.start_time_ms,
        duration_ms=duration_ms,
    )
    if first_boundary is None:
        return
    snapped_first_beat, _ = first_boundary
    period_ms = 60000.0 / base.bpm
    for first_beat in (
        snapped_first_beat - 1,
        snapped_first_beat,
        snapped_first_beat + 1,
    ):
        if first_beat <= 0:
            continue
        first_time_ms = base.origin_time_ms + first_beat * period_ms
        if first_time_ms <= base.origin_time_ms or first_time_ms >= duration_ms:
            continue
        middle_beats = int(
            round(max(1e-9, (raw_run.end_time_ms - first_time_ms) * run_bpm / 60000.0))
        )
        if middle_beats <= 0:
            continue
        final_beat = first_beat + middle_beats
        middle_duration_seconds = 60.0 * middle_beats / run_bpm
        if not (
            config.minimum_excursion_seconds
            <= middle_duration_seconds
            <= config.maximum_excursion_seconds
        ):
            continue
        middle_end_time_ms = first_time_ms + 1000.0 * middle_duration_seconds
        if middle_end_time_ms >= duration_ms:
            continue
        end_beat = _terminal_beat_after_boundary(
            boundary_beat=final_beat,
            boundary_time_ms=middle_end_time_ms,
            duration_ms=duration_ms,
            bpm=base.bpm,
        )
        _append_constant_section_proposal(
            proposals,
            seen=seen,
            signal=signal,
            frame_rate_hz=frame_rate_hz,
            duration_ms=duration_ms,
            origin_time_ms=base.origin_time_ms,
            sections=(
                ConstantTempoSection(0, first_beat, base.bpm),
                ConstantTempoSection(first_beat, final_beat, run_bpm),
                ConstantTempoSection(final_beat, end_beat, base.bpm),
            ),
            source="raw_run_aba",
            source_score=raw_run.summed_strength,
            config=config,
        )


def _append_raw_run_chain_proposal(
    proposals: list[_CurveProposal],
    *,
    seen: set[str],
    signal: NDArray[np.float64],
    frame_rate_hz: float,
    duration_ms: float,
    base: _BaseHypothesis,
    raw_runs: tuple[TempoTrackRawRunDiagnostic, ...],
    config: TempoTrackConfig,
) -> None:
    current_beat = 0
    current_time_ms = base.origin_time_ms
    current_bpm = base.bpm
    section_specs: list[tuple[int, int, float]] = []

    for raw_run in raw_runs:
        run_bpm = _raw_run_jump_bpm(raw_run, base_bpm=base.bpm, config=config)
        if run_bpm is None:
            return
        boundary_beat = current_beat + int(
            round(
                max(
                    1e-9,
                    (raw_run.start_time_ms - current_time_ms)
                    * current_bpm
                    / 60000.0,
                )
            )
        )
        if boundary_beat <= current_beat:
            return
        boundary_time_ms = current_time_ms + (
            60000.0 * (boundary_beat - current_beat) / current_bpm
        )
        if boundary_time_ms >= duration_ms:
            return
        section_specs.append((current_beat, boundary_beat, current_bpm))
        current_beat = boundary_beat
        current_time_ms = boundary_time_ms
        current_bpm = run_bpm

    end_beat = _terminal_beat_after_boundary(
        boundary_beat=current_beat,
        boundary_time_ms=current_time_ms,
        duration_ms=duration_ms,
        bpm=current_bpm,
    )
    section_specs.append((current_beat, end_beat, current_bpm))
    if len(section_specs) > 4:
        return
    try:
        sections = tuple(
            ConstantTempoSection(start_beat, end_beat, bpm)
            for start_beat, end_beat, bpm in section_specs
        )
    except ValueError:
        return
    _append_constant_section_proposal(
        proposals,
        seen=seen,
        signal=signal,
        frame_rate_hz=frame_rate_hz,
        duration_ms=duration_ms,
        origin_time_ms=base.origin_time_ms,
        sections=sections,
        source=f"raw_run_chain_{len(raw_runs) + 1}_sections",
        source_score=sum(raw_run.summed_strength for raw_run in raw_runs),
        config=config,
    )


def _append_constant_section_proposal(
    proposals: list[_CurveProposal],
    *,
    seen: set[str],
    signal: NDArray[np.float64],
    frame_rate_hz: float,
    duration_ms: float,
    origin_time_ms: float,
    sections: tuple[ConstantTempoSection, ...],
    source: str,
    source_score: float,
    config: TempoTrackConfig,
) -> None:
    if sections[-1].end_beat <= sections[0].start_beat:
        return
    try:
        curve = PhaseContinuousTimingCurve(
            origin_beat=sections[0].start_beat,
            origin_time_ms=origin_time_ms,
            sections=sections,
        )
    except ValueError:
        return
    if curve.end_time_ms + 1e-6 < duration_ms:
        return
    fingerprint = curve.fingerprint_sha256
    if fingerprint in seen:
        return
    seen.add(fingerprint)
    collapse_bpm = (
        sections[-1].bpm
        if source.startswith("raw_run_persistent_b_to_a")
        else sections[0].bpm
    )
    collapsed = _collapsed_constant_counterfactual(
        curve,
        duration_ms=duration_ms,
        base_bpm=collapse_bpm,
        config=config,
    )
    score = _generalized_support_delta(
        signal,
        frame_rate_hz=frame_rate_hz,
        candidate=curve,
        collapsed=collapsed,
        radius_ms=config.beat_event_radius_ms,
    )
    proposals.append(
        _CurveProposal(
            curve=curve,
            source=source,
            score=float(score + 0.02 * source_score),
            collapse_bpm=float(collapse_bpm),
        )
    )


def _raw_run_jump_bpm(
    raw_run: TempoTrackRawRunDiagnostic,
    *,
    base_bpm: float,
    config: TempoTrackConfig,
) -> float | None:
    bpm = float(raw_run.median_bpm)
    if not config.minimum_bpm <= bpm <= config.maximum_bpm:
        return None
    canonical = _snap_raw_observation_bpm_to_primary(
        bpm,
        primary_bpm=base_bpm,
        config=config,
    )
    if abs(canonical - base_bpm) < config.minimum_jump_bpm:
        return None
    if abs(bpm - base_bpm) < config.minimum_jump_bpm:
        return None
    return bpm


def _snapped_constant_boundary(
    *,
    origin_time_ms: float,
    bpm: float,
    target_time_ms: float,
    duration_ms: float,
) -> tuple[int, float] | None:
    period_ms = 60000.0 / bpm
    beat = int(round((target_time_ms - origin_time_ms) / period_ms))
    if beat <= 0:
        return None
    time_ms = origin_time_ms + beat * period_ms
    if time_ms <= origin_time_ms or time_ms >= duration_ms:
        return None
    return beat, time_ms


def _terminal_beat_after_boundary(
    *,
    boundary_beat: int,
    boundary_time_ms: float,
    duration_ms: float,
    bpm: float,
) -> int:
    remaining_seconds = max(0.0, (duration_ms - boundary_time_ms) / 1000.0)
    beat_offset = int(math.ceil(max(0.0, remaining_seconds * bpm / 60.0) - 1e-9))
    return boundary_beat + max(1, beat_offset)


def _aba_support_delta(
    signal: NDArray[np.float64],
    *,
    frame_rate_hz: float,
    origin_time_ms: float,
    base_bpm: float,
    first_beat: int,
    middle_beats: int,
    middle_bpm: float,
    radius_ms: float,
) -> float:
    base_period_ms = 60000.0 / base_bpm
    middle_period_ms = 60000.0 / middle_bpm
    shoulder = 8
    beat_start = max(0, first_beat - shoulder)
    beat_end = first_beat + middle_beats + shoulder
    beat_indices = np.arange(beat_start, beat_end + 1, dtype=np.float64)
    baseline_times = origin_time_ms + beat_indices * base_period_ms
    candidate_times = baseline_times.copy()
    inside = (beat_indices >= first_beat) & (
        beat_indices <= first_beat + middle_beats
    )
    candidate_times[inside] = (
        origin_time_ms
        + first_beat * base_period_ms
        + (beat_indices[inside] - first_beat) * middle_period_ms
    )
    after = beat_indices > first_beat + middle_beats
    candidate_times[after] = (
        origin_time_ms
        + first_beat * base_period_ms
        + middle_beats * middle_period_ms
        + (beat_indices[after] - first_beat - middle_beats) * base_period_ms
    )
    candidate_half = 0.5 * (candidate_times[:-1] + candidate_times[1:])
    baseline_half = baseline_times[:-1] + 0.5 * base_period_ms
    candidate_score = float(
        np.mean(
            _event_support(
                signal,
                candidate_times,
                frame_rate_hz=frame_rate_hz,
                radius_ms=radius_ms,
            )
        )
        - 0.5
        * np.mean(
            _event_support(
                signal,
                candidate_half,
                frame_rate_hz=frame_rate_hz,
                radius_ms=radius_ms,
            )
        )
    )
    baseline_score = float(
        np.mean(
            _event_support(
                signal,
                baseline_times,
                frame_rate_hz=frame_rate_hz,
                radius_ms=radius_ms,
            )
        )
        - 0.5
        * np.mean(
            _event_support(
                signal,
                baseline_half,
                frame_rate_hz=frame_rate_hz,
                radius_ms=radius_ms,
            )
        )
    )
    return candidate_score - baseline_score


def _local_tempo_bonus(
    observations: tuple[LocalTempoObservation, ...],
    *,
    center_times_ms: tuple[float, ...],
    middle_time_ms: float,
    bpm: float,
    duration_ms: float,
) -> float:
    radius_ms = max(1000.0, 0.75 * duration_ms)
    # Keep the scalar membership predicate below authoritative.  The tiny
    # padding only makes this binary-search slice conservative at a rounded
    # floating-point boundary.
    lower = bisect_left(center_times_ms, middle_time_ms - radius_ms - 1e-6)
    upper = bisect_right(center_times_ms, middle_time_ms + radius_ms + 1e-6)
    nearby = (
        observation
        for observation in observations[lower:upper]
        if abs(observation.center_time_ms - middle_time_ms)
        <= radius_ms
    )
    return max(
        (
            (0.24 if observation.source == "raw_audio" else 0.12)
            * observation.strength
            * math.exp(-abs(observation.bpm - bpm) / 5.0)
            for observation in nearby
        ),
        default=0.0,
    )


def _localized_ramp_runs(
    observations: Sequence[LocalTempoObservation],
    *,
    config: TempoTrackConfig,
    minimum_points: int = 6,
    maximum_points: int = 12,
    minimum_r_squared: float = 0.85,
    minimum_span_bpm: float = 16.0,
    maximum_runs: int = 6,
) -> tuple[_LocalizedRampRun, ...]:
    values = tuple(
        sorted(
            (value for value in observations if value.source == "beatthis"),
            key=lambda value: value.center_time_ms,
        )
    )
    proposals: list[_LocalizedRampRun] = []
    for point_count in range(minimum_points, maximum_points + 1):
        for start_index in range(0, len(values) - point_count + 1):
            window = values[start_index : start_index + point_count]
            times = np.asarray(
                [value.center_time_ms / 1000.0 for value in window],
                dtype=np.float64,
            )
            original_bpms = np.asarray(
                [value.bpm for value in window], dtype=np.float64
            )
            strengths = np.asarray(
                [max(value.strength, 1e-3) for value in window],
                dtype=np.float64,
            )
            upper = original_bpms[original_bpms >= 100.0]
            reference = float(
                np.median(upper) if upper.size >= 3 else np.median(original_bpms)
            )
            bpms = np.asarray(
                [
                    _normalize_observation_alias(
                        float(bpm), reference_bpm=reference, config=config
                    )
                    for bpm in original_bpms
                ],
                dtype=np.float64,
            )
            mean_time = float(np.average(times, weights=strengths))
            mean_bpm = float(np.average(bpms, weights=strengths))
            centered_times = times - mean_time
            denominator = float(
                np.sum(strengths * centered_times * centered_times)
            )
            if denominator <= 0.0:
                continue
            slope = float(
                np.sum(strengths * centered_times * (bpms - mean_bpm))
                / denominator
            )
            fitted = mean_bpm + slope * centered_times
            total = float(np.sum(strengths * np.square(bpms - mean_bpm)))
            if total <= 0.0:
                continue
            residual = float(np.sum(strengths * np.square(bpms - fitted)))
            r_squared = max(0.0, 1.0 - residual / total)
            span = float(fitted[-1] - fitted[0])
            if r_squared < minimum_r_squared or abs(span) < minimum_span_bpm:
                continue
            if not all(
                config.minimum_bpm <= endpoint <= config.maximum_bpm
                for endpoint in (float(fitted[0]), float(fitted[-1]))
            ):
                continue
            mean_strength = float(np.mean(strengths, dtype=np.float64))
            proposals.append(
                _LocalizedRampRun(
                    observation_start_ms=float(times[0] * 1000.0),
                    observation_end_ms=float(times[-1] * 1000.0),
                    fitted_start_bpm=float(fitted[0]),
                    fitted_end_bpm=float(fitted[-1]),
                    r_squared=r_squared,
                    mean_strength=mean_strength,
                    point_count=point_count,
                    generation_score=r_squared * abs(span) * mean_strength,
                )
            )
    proposals.sort(
        key=lambda value: (
            -value.generation_score,
            value.observation_start_ms,
            value.observation_end_ms,
            value.fitted_start_bpm,
            value.fitted_end_bpm,
        )
    )
    return tuple(proposals[:maximum_runs])


def _localized_octave_normalize(
    value: float,
    *,
    reference: float,
    config: TempoTrackConfig,
) -> float:
    alternatives = tuple(
        candidate
        for candidate in (0.5 * value, value, 2.0 * value)
        if config.minimum_bpm <= candidate <= config.maximum_bpm
    )
    if not alternatives:
        return value
    return min(alternatives, key=lambda candidate: abs(candidate - reference))


def _localized_nice_nominal(value: float) -> float:
    nearest_five = 5.0 * round(value / 5.0)
    if abs(nearest_five - value) <= 1.5:
        return nearest_five
    nearest_integer = float(round(value))
    return nearest_integer if abs(nearest_integer - value) <= 0.75 else value


def _localized_regional_mode(
    observations: Sequence[LocalTempoObservation],
    *,
    start_ms: float,
    end_ms: float,
    config: TempoTrackConfig,
) -> float | None:
    values = tuple(
        value
        for value in observations
        if value.source == "beatthis" and start_ms <= value.center_time_ms <= end_ms
    )
    if len(values) < 6:
        return None
    bpms = np.asarray([value.bpm for value in values], dtype=np.float64)
    upper = bpms[bpms >= 100.0]
    reference = float(np.median(upper) if upper.size >= 3 else np.median(bpms))
    normalized = np.asarray(
        [
            _localized_octave_normalize(
                float(value), reference=reference, config=config
            )
            for value in bpms
        ],
        dtype=np.float64,
    )
    return _localized_nice_nominal(float(np.median(normalized)))


def _infer_localized_persistent_step(
    run: _LocalizedRampRun,
    observations: Sequence[LocalTempoObservation],
    *,
    config: TempoTrackConfig,
) -> _PersistentStepSpec | None:
    left_mode = _localized_regional_mode(
        observations,
        start_ms=max(0.0, run.observation_start_ms - 100_000.0),
        end_ms=run.observation_start_ms - 55_000.0,
        config=config,
    )
    right_mode = _localized_regional_mode(
        observations,
        start_ms=max(0.0, run.observation_start_ms - 40_000.0),
        end_ms=run.observation_start_ms - 8_000.0,
        config=config,
    )
    if left_mode is None or right_mode is None or abs(right_mode - left_mode) < 16.0:
        return None

    ordered = tuple(
        sorted(
            (value for value in observations if value.source == "beatthis"),
            key=lambda value: value.center_time_ms,
        )
    )
    search_start = run.observation_start_ms - 55_000.0
    search_end = run.observation_start_ms - 8_000.0
    tolerance = max(5.0, 0.05 * right_mode)
    for index, value in enumerate(ordered):
        if not search_start <= value.center_time_ms <= search_end:
            continue
        tail = ordered[index : index + 5]
        if len(tail) < 5:
            break
        if all(
            abs(
                _localized_octave_normalize(
                    point.bpm, reference=right_mode, config=config
                )
                - right_mode
            )
            <= tolerance
            for point in tail
        ):
            return _PersistentStepSpec(
                left_bpm=left_mode,
                right_bpm=right_mode,
                transition_target_ms=value.center_time_ms
                + (config.local_window_seconds - config.local_hop_seconds)
                * 1000.0,
            )
    return None


def _localized_persistent_step_curve(
    spec: _PersistentStepSpec,
    *,
    phase_ms: float,
    duration_ms: float,
) -> PhaseContinuousTimingCurve:
    first_beats = max(
        1,
        round(
            (spec.transition_target_ms - phase_ms)
            * spec.left_bpm
            / 60_000.0
        ),
    )
    boundary_ms = phase_ms + 60_000.0 * first_beats / spec.left_bpm
    remaining_beats = max(
        1,
        math.ceil(
            max(0.0, duration_ms - boundary_ms)
            * spec.right_bpm
            / 60_000.0
            - 1e-9
        ),
    )
    return PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=phase_ms,
        sections=(
            ConstantTempoSection(0, first_beats, spec.left_bpm),
            ConstantTempoSection(
                first_beats, first_beats + remaining_beats, spec.right_bpm
            ),
        ),
    )


def _localized_constant_curve(
    bpm: float,
    *,
    phase_ms: float,
    duration_ms: float,
) -> PhaseContinuousTimingCurve:
    beats = max(
        1,
        math.ceil(max(0.0, duration_ms - phase_ms) * bpm / 60_000.0 - 1e-9),
    )
    return PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=phase_ms,
        sections=(ConstantTempoSection(0, beats, bpm),),
    )


def _score_localized_backbones(
    curves: tuple[PhaseContinuousTimingCurve, ...],
    *,
    evidence: RawAudioEvidence,
    beat_signal: NDArray[np.float64],
    frame_rate_hz: float,
    duration_ms: float,
    config: TempoTrackConfig,
) -> _LocalizedBackboneSelection | None:
    if not curves:
        return None
    curves = curves[
        : min(config.maximum_candidates, evidence.config.maximum_candidates)
    ]
    raw = score_raw_audio_evidence_independent(evidence, curves)
    raw_by_index = {
        value.candidate_index: float(value.raw_score)
        for value in raw.candidate_scores
        if value.raw_score is not None
    }
    if not raw_by_index:
        return None
    beatthis_by_index = {
        index: _physical_curve_support_score(
            beat_signal,
            frame_rate_hz=frame_rate_hz,
            curve=curve,
            duration_ms=duration_ms,
            radius_ms=config.beat_event_radius_ms,
        )
        for index, curve in enumerate(curves)
    }
    raw_rank = {
        index: rank
        for rank, index in enumerate(
            sorted(raw_by_index, key=lambda index: -raw_by_index[index]), start=1
        )
    }
    beatthis_rank = {
        index: rank
        for rank, index in enumerate(
            sorted(
                beatthis_by_index,
                key=lambda index: -beatthis_by_index[index],
            ),
            start=1,
        )
    }
    selected = min(
        raw_by_index,
        key=lambda index: (
            raw_rank[index] + beatthis_rank[index],
            beatthis_rank[index],
            raw_rank[index],
            curves[index].origin_time_ms,
        ),
    )
    selected_curve = curves[selected]
    left, right = selected_curve.sections
    if not isinstance(left, ConstantTempoSection) or not isinstance(
        right, ConstantTempoSection
    ):
        return None
    left_constant = _localized_constant_curve(
        left.bpm,
        phase_ms=selected_curve.origin_time_ms,
        duration_ms=duration_ms,
    )
    right_constant = _localized_constant_curve(
        right.bpm,
        phase_ms=selected_curve.origin_time_ms,
        duration_ms=duration_ms,
    )
    control_raw = score_raw_audio_evidence_independent(
        evidence, (left_constant, right_constant)
    )
    left_raw = control_raw.candidate_scores[0].raw_score
    right_raw = control_raw.candidate_scores[1].raw_score
    if left_raw is None or right_raw is None:
        return None
    left_beatthis = _physical_curve_support_score(
        beat_signal,
        frame_rate_hz=frame_rate_hz,
        curve=left_constant,
        duration_ms=duration_ms,
        radius_ms=config.beat_event_radius_ms,
    )
    right_beatthis = _physical_curve_support_score(
        beat_signal,
        frame_rate_hz=frame_rate_hz,
        curve=right_constant,
        duration_ms=duration_ms,
        radius_ms=config.beat_event_radius_ms,
    )
    result = _LocalizedBackboneSelection(
        curve=selected_curve,
        raw_rank=raw_rank[selected],
        beatthis_rank=beatthis_rank[selected],
        constant_left_raw_gain=raw_by_index[selected] - float(left_raw),
        constant_left_beatthis_gain=beatthis_by_index[selected] - left_beatthis,
        constant_right_raw_gain=raw_by_index[selected] - float(right_raw),
        constant_right_beatthis_gain=beatthis_by_index[selected] - right_beatthis,
    )
    return result if result.double_supported else None


def _select_localized_backbone(
    spec: _PersistentStepSpec,
    *,
    jump_proposals: tuple[_CurveProposal, ...],
    evidence: RawAudioEvidence,
    beat_signal: NDArray[np.float64],
    frame_rate_hz: float,
    duration_ms: float,
    config: TempoTrackConfig,
) -> _LocalizedBackboneSelection | None:
    tolerance_ms = max(5000.0, config.local_window_seconds * 1000.0)
    matching: list[PhaseContinuousTimingCurve] = []
    for proposal in jump_proposals:
        curve = proposal.curve
        if len(curve.sections) != 2 or not all(
            isinstance(section, ConstantTempoSection) for section in curve.sections
        ):
            continue
        left, right = curve.sections
        assert isinstance(left, ConstantTempoSection)
        assert isinstance(right, ConstantTempoSection)
        if (
            abs(left.bpm - spec.left_bpm) <= max(3.0, 0.02 * spec.left_bpm)
            and abs(right.bpm - spec.right_bpm)
            <= max(3.0, 0.02 * spec.right_bpm)
            and abs(curve.boundary_times_ms[1] - spec.transition_target_ms)
            <= tolerance_ms
        ):
            matching.append(curve)
    existing = _score_localized_backbones(
        tuple(matching),
        evidence=evidence,
        beat_signal=beat_signal,
        frame_rate_hz=frame_rate_hz,
        duration_ms=duration_ms,
        config=config,
    )
    if existing is not None:
        return existing

    period_ms = 60_000.0 / spec.left_bpm
    phase_curves = tuple(
        _localized_persistent_step_curve(
            spec, phase_ms=float(phase), duration_ms=duration_ms
        )
        for phase in np.arange(0.0, period_ms, 10.0)
    )
    return _score_localized_backbones(
        phase_curves,
        evidence=evidence,
        beat_signal=beat_signal,
        frame_rate_hz=frame_rate_hz,
        duration_ms=duration_ms,
        config=config,
    )


def _localized_ramp_boundary_interval(
    run: _LocalizedRampRun,
    boundary_seeds: Sequence[_BoundarySeed],
    *,
    config: TempoTrackConfig,
) -> tuple[float, float] | None:
    if not boundary_seeds:
        return None
    times = tuple(value.time_ms for value in boundary_seeds)
    start_target = run.observation_start_ms + (
        config.local_window_seconds * 1000.0 / 3.0
    )
    end_target = run.observation_end_ms + (
        config.local_window_seconds * 1000.0 / 2.0
    )
    start_ms = min(times, key=lambda value: abs(value - start_target))
    end_ms = min(times, key=lambda value: abs(value - end_target))
    if (
        abs(start_ms - start_target) > 1500.0
        or abs(end_ms - end_target) > 1500.0
        or end_ms <= start_ms
    ):
        return None
    return float(start_ms), float(end_ms)


def _splice_localized_ramp(
    backbone: PhaseContinuousTimingCurve,
    run: _LocalizedRampRun,
    *,
    ramp_start_target_ms: float,
    ramp_end_target_ms: float,
    duration_ms: float,
    endpoint_delta_scale: float,
) -> _LocalizedRampTriplet:
    if len(backbone.sections) != 2 or not all(
        isinstance(section, ConstantTempoSection) for section in backbone.sections
    ):
        raise ValueError("backbone must contain exactly two constant sections")
    left, right = backbone.sections
    assert isinstance(left, ConstantTempoSection)
    assert isinstance(right, ConstantTempoSection)
    ramp_start_beat = max(
        right.start_beat + 1,
        round(backbone.beat_at_time(ramp_start_target_ms)),
    )
    ramp_start_ms = backbone.time_at_beat(float(ramp_start_beat))
    mean_endpoint = 0.5 * (run.fitted_start_bpm + run.fitted_end_bpm)
    delta = endpoint_delta_scale * (
        run.fitted_end_bpm - run.fitted_start_bpm
    )
    start_bpm = mean_endpoint - 0.5 * delta
    end_bpm = mean_endpoint + 0.5 * delta
    ramp_beats = max(
        2,
        round(
            max(1.0, ramp_end_target_ms - ramp_start_ms)
            * (start_bpm + end_bpm)
            / 120_000.0
        ),
    )
    ramp_end_beat = ramp_start_beat + ramp_beats
    ramp_section = LinearTimeRampSection(
        ramp_start_beat, ramp_end_beat, start_bpm, end_bpm
    )
    ramp_end_ms = ramp_start_ms + ramp_section.duration_ms
    post_beats = max(
        1,
        math.ceil(
            max(0.0, duration_ms - ramp_end_ms) * end_bpm / 60_000.0 - 1e-9
        ),
    )
    terminal_beat = ramp_end_beat + post_beats
    ramp = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=backbone.origin_time_ms,
        sections=(
            left,
            ConstantTempoSection(right.start_beat, ramp_start_beat, right.bpm),
            ramp_section,
            ConstantTempoSection(ramp_end_beat, terminal_beat, end_bpm),
        ),
    )

    step_left_beats = max(
        1,
        min(
            ramp_beats - 1,
            round(ramp_beats * start_bpm / (start_bpm + end_bpm)),
        ),
    )
    step_beat = ramp_start_beat + step_left_beats
    step = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=backbone.origin_time_ms,
        sections=(
            left,
            ConstantTempoSection(right.start_beat, ramp_start_beat, right.bpm),
            ConstantTempoSection(ramp_start_beat, step_beat, start_bpm),
            ConstantTempoSection(step_beat, terminal_beat, end_bpm),
        ),
    )
    no_ramp_end = right.start_beat + max(
        1,
        math.ceil(
            max(0.0, duration_ms - backbone.boundary_times_ms[1])
            * right.bpm
            / 60_000.0
            - 1e-9
        ),
    )
    no_ramp = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=backbone.origin_time_ms,
        sections=(
            left,
            ConstantTempoSection(right.start_beat, no_ramp_end, right.bpm),
        ),
    )
    return _LocalizedRampTriplet(
        ramp=ramp,
        step=step,
        no_ramp=no_ramp,
        endpoint_delta_scale=endpoint_delta_scale,
    )


def _localized_observation_curve_score(
    curve: PhaseContinuousTimingCurve,
    observations: Sequence[LocalTempoObservation],
    *,
    start_ms: float,
    end_ms: float,
    config: TempoTrackConfig,
) -> float:
    errors: list[float] = []
    weights: list[float] = []
    for observation in observations:
        if not start_ms <= observation.center_time_ms <= end_ms:
            continue
        predicted = curve.bpm_at_time(observation.center_time_ms)
        aliases = tuple(
            observation.bpm * multiplier
            for multiplier in _LOCALIZED_RAMP_RATIONAL_ALIASES
            if config.minimum_bpm
            <= observation.bpm * multiplier
            <= config.maximum_bpm
        )
        if not aliases:
            continue
        errors.append(min(abs(alias - predicted) for alias in aliases))
        weights.append(max(observation.strength, 0.01))
    if not errors:
        return -math.inf
    return -float(np.average(errors, weights=weights))


def _localized_ramp_observation_gate(
    triplet: _LocalizedRampTriplet,
    *,
    raw_observations: Sequence[LocalTempoObservation],
    beat_observations: Sequence[LocalTempoObservation],
    config: TempoTrackConfig,
) -> _LocalizedRampObservationGate:
    ramp_start_ms = triplet.ramp.boundary_times_ms[2]
    ramp_end_ms = triplet.ramp.boundary_times_ms[3]
    start_ms = ramp_start_ms - 3000.0
    end_ms = ramp_end_ms + 3000.0
    curves = (triplet.ramp, triplet.step, triplet.no_ramp)
    raw_scores = tuple(
        _localized_observation_curve_score(
            curve,
            raw_observations,
            start_ms=start_ms,
            end_ms=end_ms,
            config=config,
        )
        for curve in curves
    )
    beatthis_scores = tuple(
        _localized_observation_curve_score(
            curve,
            beat_observations,
            start_ms=start_ms,
            end_ms=end_ms,
            config=config,
        )
        for curve in curves
    )
    return _LocalizedRampObservationGate(
        raw_step_gain=raw_scores[0] - raw_scores[1],
        raw_no_ramp_gain=raw_scores[0] - raw_scores[2],
        beatthis_step_gain=beatthis_scores[0] - beatthis_scores[1],
        beatthis_no_ramp_gain=beatthis_scores[0] - beatthis_scores[2],
    )


def _localized_backbone_ramp_choice(
    observations: Sequence[LocalTempoObservation],
    *,
    beat_observations: Sequence[LocalTempoObservation],
    raw_observations: Sequence[LocalTempoObservation],
    audio_evidence: RawAudioEvidence | None,
    beat_signal: NDArray[np.float64],
    frame_rate_hz: float,
    duration_ms: float,
    boundary_seeds: Sequence[_BoundarySeed],
    jump_proposals: tuple[_CurveProposal, ...],
    config: TempoTrackConfig,
) -> _LocalizedRampChoice | None:
    if audio_evidence is None or not raw_observations:
        return None
    runs = _localized_ramp_runs(beat_observations, config=config)
    if not runs:
        return None
    run = runs[0]
    spec = _infer_localized_persistent_step(run, beat_observations, config=config)
    if spec is None:
        return None
    anchored = _localized_ramp_boundary_interval(
        run, boundary_seeds, config=config
    )
    if anchored is None:
        return None
    backbone = _select_localized_backbone(
        spec,
        jump_proposals=jump_proposals,
        evidence=audio_evidence,
        beat_signal=beat_signal,
        frame_rate_hz=frame_rate_hz,
        duration_ms=duration_ms,
        config=config,
    )
    if backbone is None:
        return None
    for scale in _LOCALIZED_RAMP_ENDPOINT_SCALES:
        triplet = _splice_localized_ramp(
            backbone.curve,
            run,
            ramp_start_target_ms=anchored[0],
            ramp_end_target_ms=anchored[1],
            duration_ms=duration_ms,
            endpoint_delta_scale=scale,
        )
        gate = _localized_ramp_observation_gate(
            triplet,
            raw_observations=raw_observations,
            beat_observations=beat_observations,
            config=config,
        )
        if not gate.eligible:
            continue
        return _LocalizedRampChoice(
            proposal=_CurveProposal(
                curve=triplet.ramp,
                source=_LOCALIZED_BACKBONE_RAMP_SOURCE,
                score=run.generation_score
                + min(
                    gate.raw_step_gain,
                    gate.raw_no_ramp_gain,
                    gate.beatthis_step_gain,
                    gate.beatthis_no_ramp_gain,
                ),
                collapse_bpm=spec.right_bpm,
            ),
            backbone=backbone,
            gate=gate,
            run=run,
            endpoint_delta_scale=scale,
        )
    return None


def _nominal_backbone_mode(
    observations: Sequence[LocalTempoObservation],
) -> tuple[float | None, int]:
    bins: dict[int, list[float]] = {}
    for observation in observations:
        if 80.0 <= observation.bpm <= 300.0:
            bins.setdefault(round(observation.bpm), []).append(observation.bpm)
    if not bins:
        return None, 0
    key, values = max(bins.items(), key=lambda item: (len(item[1]), item[0]))
    return float(round(float(np.median(values)))) if values else float(key), len(values)


def _nominal_backbone_segment_observations(
    observations: Sequence[LocalTempoObservation],
    *,
    start_ms: float,
    end_ms: float,
) -> tuple[LocalTempoObservation, ...]:
    margin_ms = 250.0 if end_ms - start_ms < 10_000.0 else 1500.0
    selected = tuple(
        observation
        for observation in observations
        if start_ms + margin_ms
        <= observation.center_time_ms
        <= end_ms - margin_ms
    )
    if selected:
        return selected
    return tuple(
        observation
        for observation in observations
        if start_ms <= observation.center_time_ms <= end_ms
    )


def _nominal_backbone_bpms(
    beatthis: Sequence[LocalTempoObservation],
    raw: Sequence[LocalTempoObservation],
    *,
    boundary_times_ms: tuple[float, ...],
    duration_ms: float,
    config: TempoTrackConfig,
) -> tuple[float, ...] | None:
    edges = (0.0,) + boundary_times_ms + (duration_ms,)
    stats: list[tuple[float, float | None, int, float | None, int]] = []
    ratio_votes: dict[float, int] = {}
    for start_ms, end_ms in zip(edges, edges[1:]):
        beatthis_mode, beatthis_count = _nominal_backbone_mode(
            _nominal_backbone_segment_observations(
                beatthis, start_ms=start_ms, end_ms=end_ms
            )
        )
        raw_mode, raw_count = _nominal_backbone_mode(
            _nominal_backbone_segment_observations(
                raw, start_ms=start_ms, end_ms=end_ms
            )
        )
        duration = end_ms - start_ms
        stats.append(
            (duration, beatthis_mode, beatthis_count, raw_mode, raw_count)
        )
        if (
            duration < 10_000.0
            or beatthis_mode is None
            or raw_mode is None
            or raw_count < 3
        ):
            continue
        ratio = raw_mode / beatthis_mode
        alias = min(
            _LOCALIZED_RAMP_RATIONAL_ALIASES,
            key=lambda value: abs(value - ratio),
        )
        if abs(alias - ratio) <= 0.08 * alias:
            ratio_votes[alias] = ratio_votes.get(alias, 0) + raw_count
    canonical = (
        max(ratio_votes, key=lambda value: (ratio_votes[value], value))
        if ratio_votes
        else 1.0
    )
    bpms: list[float] = []
    for duration, beatthis_mode, _beatthis_count, raw_mode, raw_count in stats:
        projected = None if beatthis_mode is None else beatthis_mode * canonical
        while projected is not None and projected > 300.0:
            projected *= 0.5
        while projected is not None and projected < 80.0:
            projected *= 2.0
        ratio = (
            None
            if beatthis_mode is None or raw_mode is None
            else raw_mode / beatthis_mode
        )
        if (
            duration >= 10_000.0
            and raw_mode is not None
            and raw_count >= 3
            and ratio is not None
            and abs(ratio - canonical) <= 0.08 * canonical
        ):
            candidate = raw_mode
        elif projected is not None:
            candidate = projected
        elif raw_mode is not None:
            candidate = raw_mode
        else:
            return None
        nominal = float(round(candidate))
        if not config.minimum_bpm <= nominal <= max(300.0, config.maximum_bpm):
            return None
        bpms.append(nominal)
    return tuple(bpms)


def _nominal_backbone_reference(
    raw: Sequence[LocalTempoObservation],
) -> float | None:
    mode, _count = _nominal_backbone_mode(raw)
    return mode


def _nominal_backbone_has_minimum_total_delta(
    bpms: tuple[float, ...],
) -> bool:
    return len(bpms) != 2 or abs(bpms[-1] - bpms[0]) >= (
        _SHORT_CHAIN_MINIMUM_TOTAL_DELTA_BPM
    )


def _nominal_backbone_normalize(
    bpm: float,
    *,
    reference_bpm: float,
    config: TempoTrackConfig,
) -> float:
    candidates = tuple(
        (bpm * multiplier, multiplier)
        for multiplier in _LOCALIZED_RAMP_RATIONAL_ALIASES
        if config.minimum_bpm
        <= bpm * multiplier
        <= max(300.0, config.maximum_bpm)
    )
    if not candidates:
        return bpm
    return min(
        candidates,
        key=lambda value: (
            abs(value[0] - reference_bpm),
            abs(value[1] - 1.0),
            -value[0],
        ),
    )[0]


def _nominal_backbone_local_change_seeds(
    observations: Sequence[LocalTempoObservation],
    *,
    reference_bpm: float,
    config: TempoTrackConfig,
) -> tuple[float, ...]:
    output: list[tuple[float, float]] = []
    for source in ("beatthis", "raw_audio"):
        selected = tuple(
            sorted(
                (
                    observation
                    for observation in observations
                    if observation.source == source
                ),
                key=lambda observation: observation.center_time_ms,
            )
        )
        normalized = tuple(
            _nominal_backbone_normalize(
                observation.bpm,
                reference_bpm=reference_bpm,
                config=config,
            )
            for observation in selected
        )
        for index in range(4, len(selected) - 4):
            left = float(np.median(normalized[index - 4 : index]))
            right = float(np.median(normalized[index : index + 4]))
            if abs(round(right) - round(left)) < 1.0:
                continue
            score = abs(right - left)
            center_ms = selected[index].center_time_ms
            for offset_ms in (-1500.0, 0.0, 1500.0):
                if center_ms + offset_ms > 0.0:
                    output.append((center_ms + offset_ms, score))
    deduped: dict[int, tuple[float, float]] = {}
    for time_ms, score in output:
        key = round(time_ms / 100.0)
        previous = deduped.get(key)
        if previous is None or score > previous[1]:
            deduped[key] = (time_ms, score)
    return tuple(
        time_ms
        for time_ms, _score in sorted(
            deduped.values(), key=lambda value: (-value[1], value[0])
        )[:128]
    )


def _nominal_backbone_dp_splits(
    values: NDArray[np.float64],
    *,
    section_count: int,
) -> tuple[tuple[int, ...], float] | None:
    minimum = _NOMINAL_BACKBONE_MINIMUM_SEGMENT_SAMPLES
    count = len(values)
    if count < section_count * minimum:
        return None
    prefix = np.concatenate((np.zeros(1), np.cumsum(values, dtype=np.float64)))
    prefix_squared = np.concatenate(
        (np.zeros(1), np.cumsum(np.square(values), dtype=np.float64))
    )

    def cost(start: int, end: int) -> float:
        size = end - start
        total = float(prefix[end] - prefix[start])
        return float(
            prefix_squared[end]
            - prefix_squared[start]
            - total * total / size
        )

    scores = np.full((section_count + 1, count + 1), np.inf)
    back = np.full((section_count + 1, count + 1), -1, dtype=np.int64)
    for end in range(minimum, count + 1):
        scores[1, end] = cost(0, end)
    all_starts = np.arange(count + 1, dtype=np.int64)
    for sections in range(2, section_count + 1):
        for end in range(sections * minimum, count + 1):
            starts = all_starts[
                (sections - 1) * minimum : end - minimum + 1
            ]
            sizes = end - starts
            totals = prefix[end] - prefix[starts]
            segment_costs = (
                prefix_squared[end]
                - prefix_squared[starts]
                - totals * totals / sizes
            )
            candidate_scores = scores[sections - 1, starts] + segment_costs
            best_index = int(np.argmin(candidate_scores))
            best_start = int(starts[best_index])
            scores[sections, end] = candidate_scores[best_index]
            back[sections, end] = best_start
    if not math.isfinite(float(scores[section_count, count])):
        return None
    splits: list[int] = []
    end = count
    for sections in range(section_count, 1, -1):
        start = int(back[sections, end])
        splits.append(start)
        end = start
    return tuple(reversed(splits)), float(scores[section_count, count])


def _nominal_backbone_curve(
    bpms: tuple[float, ...],
    boundary_targets_ms: tuple[float, ...],
    *,
    phase_ms: float,
    duration_ms: float,
) -> PhaseContinuousTimingCurve | None:
    sections: list[ConstantTempoSection] = []
    current_beat = 0
    current_time_ms = phase_ms
    for bpm, target_ms in zip(bpms[:-1], boundary_targets_ms):
        beats = max(1, round((target_ms - current_time_ms) * bpm / 60_000.0))
        end_beat = current_beat + beats
        sections.append(ConstantTempoSection(current_beat, end_beat, bpm))
        current_beat = end_beat
        current_time_ms += beats * 60_000.0 / bpm
    terminal_beats = max(
        1,
        math.ceil(
            max(0.0, duration_ms - current_time_ms) * bpms[-1] / 60_000.0
            - 1e-9
        ),
    )
    sections.append(
        ConstantTempoSection(
            current_beat, current_beat + terminal_beats, bpms[-1]
        )
    )
    try:
        return PhaseContinuousTimingCurve(
            origin_beat=0,
            origin_time_ms=phase_ms,
            sections=tuple(sections),
        )
    except ValueError:
        return None


def _nominal_backbone_phase_curve(
    bpms: tuple[float, ...],
    boundary_targets_ms: tuple[float, ...],
    *,
    audio_evidence: RawAudioEvidence,
    beat_signal: NDArray[np.float64],
    frame_rate_hz: float,
    duration_ms: float,
    config: TempoTrackConfig,
) -> PhaseContinuousTimingCurve | None:
    period_ms = 60_000.0 / bpms[0]
    phases = np.linspace(
        0.0, period_ms, _NOMINAL_BACKBONE_PHASE_COUNT, endpoint=False
    )
    curves = tuple(
        curve
        for phase in phases
        if (
            curve := _nominal_backbone_curve(
                bpms,
                boundary_targets_ms,
                phase_ms=float(phase),
                duration_ms=duration_ms,
            )
        )
        is not None
    )
    if not curves:
        return None
    raw = score_raw_audio_evidence_independent(audio_evidence, curves)
    raw_by_index = {
        value.candidate_index: float(value.raw_score)
        for value in raw.candidate_scores
        if value.raw_score is not None
    }
    if not raw_by_index:
        return None
    beatthis_by_index = {
        index: _physical_curve_support_score(
            beat_signal,
            frame_rate_hz=frame_rate_hz,
            curve=curve,
            duration_ms=duration_ms,
            radius_ms=config.beat_event_radius_ms,
        )
        for index, curve in enumerate(curves)
    }
    raw_rank = {
        index: rank
        for rank, index in enumerate(
            sorted(raw_by_index, key=lambda index: -raw_by_index[index]), start=1
        )
    }
    beatthis_rank = {
        index: rank
        for rank, index in enumerate(
            sorted(
                beatthis_by_index,
                key=lambda index: -beatthis_by_index[index],
            ),
            start=1,
        )
    }
    selected = min(
        raw_by_index,
        key=lambda index: (
            raw_rank[index] + beatthis_rank[index],
            beatthis_rank[index],
            raw_rank[index],
            curves[index].origin_time_ms,
        ),
    )
    return curves[selected]


def _nominal_backbone_edge_gains(
    curve: PhaseContinuousTimingCurve,
    *,
    edge_index: int,
    beatthis: Sequence[LocalTempoObservation],
    raw: Sequence[LocalTempoObservation],
    config: TempoTrackConfig,
) -> tuple[float, float, float, float]:
    boundaries = curve.boundary_times_ms[1:-1]
    boundary_ms = boundaries[edge_index]
    previous_ms = curve.start_time_ms if edge_index == 0 else boundaries[edge_index - 1]
    following_ms = (
        curve.end_time_ms
        if edge_index + 1 == len(boundaries)
        else boundaries[edge_index + 1]
    )
    shoulder_ms = min(
        6000.0,
        0.45 * (boundary_ms - previous_ms),
        0.45 * (following_ms - boundary_ms),
    )
    if shoulder_ms <= 0.0:
        return (-math.inf,) * 4
    left_bpm = curve.sections[edge_index].end_bpm
    right_bpm = curve.sections[edge_index + 1].start_bpm
    left_constant = _localized_constant_curve(
        left_bpm, phase_ms=curve.origin_time_ms, duration_ms=curve.end_time_ms
    )
    right_constant = _localized_constant_curve(
        right_bpm, phase_ms=curve.origin_time_ms, duration_ms=curve.end_time_ms
    )
    start_ms = boundary_ms - shoulder_ms
    end_ms = boundary_ms + shoulder_ms
    raw_curve = _localized_observation_curve_score(
        curve, raw, start_ms=start_ms, end_ms=end_ms, config=config
    )
    raw_left = _localized_observation_curve_score(
        left_constant, raw, start_ms=start_ms, end_ms=end_ms, config=config
    )
    raw_right = _localized_observation_curve_score(
        right_constant, raw, start_ms=start_ms, end_ms=end_ms, config=config
    )
    beatthis_curve = _localized_observation_curve_score(
        curve, beatthis, start_ms=start_ms, end_ms=end_ms, config=config
    )
    beatthis_left = _localized_observation_curve_score(
        left_constant, beatthis, start_ms=start_ms, end_ms=end_ms, config=config
    )
    beatthis_right = _localized_observation_curve_score(
        right_constant, beatthis, start_ms=start_ms, end_ms=end_ms, config=config
    )
    return (
        raw_curve - raw_left,
        raw_curve - raw_right,
        beatthis_curve - beatthis_left,
        beatthis_curve - beatthis_right,
    )


def _nominal_backbone_raw_support_counts(
    curve: PhaseContinuousTimingCurve,
    *,
    raw: Sequence[LocalTempoObservation],
    config: TempoTrackConfig,
) -> tuple[int, ...]:
    counts: list[int] = []
    boundaries = curve.boundary_times_ms
    for index, section in enumerate(curve.sections):
        start_ms = boundaries[index]
        end_ms = boundaries[index + 1]
        tolerance_bpm = max(1.0, 0.01 * section.start_bpm)
        counts.append(
            sum(
                observation.window_start_ms >= start_ms
                and observation.window_end_ms <= end_ms
                and any(
                    abs(observation.bpm * multiplier - section.start_bpm)
                    <= tolerance_bpm
                    for multiplier in _LOCALIZED_RAMP_RATIONAL_ALIASES
                    if config.minimum_bpm
                    <= observation.bpm * multiplier
                    <= max(300.0, config.maximum_bpm)
                )
                for observation in raw
            )
        )
    return tuple(counts)


def _nominal_backbone_step_score(
    observations: Sequence[LocalTempoObservation],
    *,
    boundary_ms: float,
    left_bpm: float,
    right_bpm: float,
    config: TempoTrackConfig,
) -> tuple[float, float]:
    start_ms = boundary_ms - 6000.0
    end_ms = boundary_ms + 6000.0

    def error(constant: float | None) -> float:
        errors: list[float] = []
        weights: list[float] = []
        for observation in observations:
            if not start_ms <= observation.center_time_ms <= end_ms:
                continue
            predicted = (
                constant
                if constant is not None
                else left_bpm
                if observation.center_time_ms < boundary_ms
                else right_bpm
            )
            aliases = tuple(
                observation.bpm * multiplier
                for multiplier in _LOCALIZED_RAMP_RATIONAL_ALIASES
                if config.minimum_bpm
                <= observation.bpm * multiplier
                <= max(300.0, config.maximum_bpm)
            )
            if aliases:
                errors.append(min(abs(value - predicted) for value in aliases))
                weights.append(max(0.01, observation.strength))
        return float(np.average(errors, weights=weights)) if errors else math.inf

    step_error = error(None)
    return error(left_bpm) - step_error, error(right_bpm) - step_error


def _nominal_backbone_refine_targets(
    targets_ms: tuple[float, ...],
    raw_targets_ms: tuple[float, ...],
    bpms: tuple[float, ...],
    seed_times_ms: tuple[float, ...],
    *,
    beatthis: Sequence[LocalTempoObservation],
    raw: Sequence[LocalTempoObservation],
    config: TempoTrackConfig,
) -> tuple[float, ...]:
    refined: list[float] = []
    for index, (target_ms, raw_target_ms) in enumerate(
        zip(targets_ms, raw_targets_ms)
    ):
        alternatives = tuple(
            seed
            for seed in seed_times_ms
            if abs(seed - raw_target_ms) <= _NOMINAL_BACKBONE_SEED_RADIUS_MS
        ) or (target_ms,)
        ranked: list[tuple[float, float, float, float]] = []
        for seed in alternatives:
            raw_gains = _nominal_backbone_step_score(
                raw,
                boundary_ms=seed,
                left_bpm=bpms[index],
                right_bpm=bpms[index + 1],
                config=config,
            )
            beatthis_gains = _nominal_backbone_step_score(
                beatthis,
                boundary_ms=seed,
                left_bpm=bpms[index],
                right_bpm=bpms[index + 1],
                config=config,
            )
            gains = raw_gains + beatthis_gains
            ranked.append(
                (min(gains), sum(gains), -abs(seed - raw_target_ms), seed)
            )
        best = max(ranked) if ranked else (0.0, 0.0, 0.0, target_ms)
        refined.append(best[3])
    return tuple(refined)


def _nominal_backbone_post_phase_refine(
    curve: PhaseContinuousTimingCurve,
    targets_ms: tuple[float, ...],
    raw_targets_ms: tuple[float, ...],
    bpms: tuple[float, ...],
    seed_times_ms: tuple[float, ...],
    *,
    beatthis: Sequence[LocalTempoObservation],
    raw: Sequence[LocalTempoObservation],
    duration_ms: float,
    config: TempoTrackConfig,
) -> tuple[PhaseContinuousTimingCurve, tuple[float, ...]]:
    current = list(targets_ms)
    for edge_index, raw_target_ms in enumerate(raw_targets_ms):
        alternatives = tuple(
            seed
            for seed in seed_times_ms
            if abs(seed - raw_target_ms) <= _NOMINAL_BACKBONE_SEED_RADIUS_MS
        )
        ranked: list[
            tuple[bool, float, float, float, float, PhaseContinuousTimingCurve]
        ] = []
        for seed in alternatives:
            candidate_targets = tuple(
                current[:edge_index] + [seed] + current[edge_index + 1 :]
            )
            if any(
                left >= right
                for left, right in zip(candidate_targets, candidate_targets[1:])
            ):
                continue
            candidate = _nominal_backbone_curve(
                bpms,
                candidate_targets,
                phase_ms=curve.origin_time_ms,
                duration_ms=duration_ms,
            )
            if candidate is None:
                continue
            gains = _nominal_backbone_edge_gains(
                candidate,
                edge_index=edge_index,
                beatthis=beatthis,
                raw=raw,
                config=config,
            )
            ranked.append(
                (
                    min(gains) > 0.0,
                    sum(gains),
                    min(gains),
                    -abs(seed - raw_target_ms),
                    seed,
                    candidate,
                )
            )
        if ranked:
            *_rank, seed, curve = max(ranked, key=lambda value: value[:4])
            current[edge_index] = seed
    return curve, tuple(current)


def _boundary_nominal_backbone_choices(
    *,
    audio_evidence: RawAudioEvidence | None,
    beat_signal: NDArray[np.float64],
    frame_rate_hz: float,
    duration_ms: float,
    boundary_seeds: tuple[_BoundarySeed, ...],
    short_observations: tuple[LocalTempoObservation, ...],
    config: TempoTrackConfig,
) -> tuple[_NominalBackboneChoice, ...]:
    if audio_evidence is None or not short_observations or not boundary_seeds:
        return ()
    beatthis = tuple(
        value for value in short_observations if value.source == "beatthis"
    )
    raw = tuple(value for value in short_observations if value.source == "raw_audio")
    reference_bpm = _nominal_backbone_reference(raw)
    if reference_bpm is None or not beatthis or not raw:
        return ()
    short_config = replace(
        config,
        local_window_seconds=_SHORT_CHAIN_WINDOW_SECONDS,
        local_hop_seconds=_SHORT_CHAIN_HOP_SECONDS,
        maximum_bpm=max(_SHORT_CHAIN_MAXIMUM_BPM, config.maximum_bpm),
    )
    side_seeds = _observation_side_segment_seeds(
        short_observations,
        boundary_seeds=boundary_seeds,
        config=short_config,
    )
    seed_times_ms = tuple(
        sorted(
            {value.time_ms for value in boundary_seeds}
            | {value.time_ms for value in side_seeds}
            | set(
                _nominal_backbone_local_change_seeds(
                    short_observations,
                    reference_bpm=reference_bpm,
                    config=short_config,
                )
            )
        )
    )
    if not seed_times_ms:
        return ()
    ordered_raw = tuple(sorted(raw, key=lambda value: value.center_time_ms))
    normalized = np.asarray(
        [
            _nominal_backbone_normalize(
                value.bpm, reference_bpm=reference_bpm, config=short_config
            )
            for value in ordered_raw
        ],
        dtype=np.float64,
    )
    smoothed = np.asarray(
        [
            np.median(
                normalized[max(0, index - 3) : min(len(normalized), index + 4)]
            )
            for index in range(len(normalized))
        ],
        dtype=np.float64,
    )
    choices: list[_NominalBackboneChoice] = []
    for section_count in (2, 6):
        segmentation = _nominal_backbone_dp_splits(
            smoothed, section_count=section_count
        )
        if segmentation is None:
            continue
        split_indices, segmentation_cost = segmentation
        raw_targets_ms = tuple(
            0.5
            * (
                ordered_raw[index - 1].center_time_ms
                + ordered_raw[index].center_time_ms
            )
            for index in split_indices
        )
        targets_ms = tuple(
            min(seed_times_ms, key=lambda seed: abs(seed - target))
            for target in raw_targets_ms
        )
        if any(
            abs(seed - target) > 3000.0
            for seed, target in zip(targets_ms, raw_targets_ms)
        ):
            continue
        bpms = _nominal_backbone_bpms(
            beatthis,
            raw,
            boundary_times_ms=targets_ms,
            duration_ms=duration_ms,
            config=short_config,
        )
        if bpms is None or any(
            abs(left - right) < 1.0 for left, right in zip(bpms, bpms[1:])
        ):
            continue
        if not _nominal_backbone_has_minimum_total_delta(bpms):
            continue
        if section_count == 6:
            directions = tuple(
                1 if right > left else -1 for left, right in zip(bpms, bpms[1:])
            )
            if sum(
                left != right for left, right in zip(directions, directions[1:])
            ) > 2:
                continue
            targets_ms = _nominal_backbone_refine_targets(
                targets_ms,
                raw_targets_ms,
                bpms,
                seed_times_ms,
                beatthis=beatthis,
                raw=raw,
                config=short_config,
            )
            bpms = _nominal_backbone_bpms(
                beatthis,
                raw,
                boundary_times_ms=targets_ms,
                duration_ms=duration_ms,
                config=short_config,
            )
            if bpms is None:
                continue
        curve = _nominal_backbone_phase_curve(
            bpms,
            targets_ms,
            audio_evidence=audio_evidence,
            beat_signal=beat_signal,
            frame_rate_hz=frame_rate_hz,
            duration_ms=duration_ms,
            config=config,
        )
        if curve is None:
            continue
        if section_count == 6:
            curve, targets_ms = _nominal_backbone_post_phase_refine(
                curve,
                targets_ms,
                raw_targets_ms,
                bpms,
                seed_times_ms,
                beatthis=beatthis,
                raw=raw,
                duration_ms=duration_ms,
                config=short_config,
            )
        raw_support_counts = _nominal_backbone_raw_support_counts(
            curve, raw=raw, config=short_config
        )
        if any(
            count < _NOMINAL_BACKBONE_MINIMUM_RAW_SUPPORT
            for count in raw_support_counts
        ):
            continue
        edge_gains = tuple(
            _nominal_backbone_edge_gains(
                curve,
                edge_index=index,
                beatthis=beatthis,
                raw=raw,
                config=short_config,
            )
            for index in range(len(curve.sections) - 1)
        )
        flattened_gains = tuple(gain for values in edge_gains for gain in values)
        if not flattened_gains or min(flattened_gains) <= 0.0:
            continue
        nominal = all(
            abs(section.start_bpm - round(section.start_bpm)) < 1e-9
            for section in curve.sections
        )
        minimum_edge_gain = min(flattened_gains)
        total_edge_gain = sum(flattened_gains)
        choices.append(
            _NominalBackboneChoice(
                proposal=_CurveProposal(
                    curve=curve,
                    source=_BOUNDARY_NOMINAL_BACKBONE_SOURCE,
                    score=float(
                        total_edge_gain
                        + minimum_edge_gain
                        - 0.001 * segmentation_cost / max(1, len(smoothed))
                        + (0.01 if nominal else 0.0)
                    ),
                    collapse_bpm=float(curve.sections[0].start_bpm),
                ),
                minimum_edge_gain=float(minimum_edge_gain),
                total_edge_gain=float(total_edge_gain),
                raw_support_counts=raw_support_counts,
                nominal=nominal,
            )
        )
    return tuple(
        sorted(
            choices,
            key=lambda choice: (
                -choice.total_edge_gain,
                -choice.minimum_edge_gain,
                -int(choice.nominal),
                choice.proposal.curve.fingerprint_sha256,
            ),
        )[:_NOMINAL_BACKBONE_MAXIMUM_WINNERS]
    )


def _early_half_primary_prefix_step_choices(
    *,
    beat_signal: NDArray[np.float64],
    frame_rate_hz: float,
    duration_ms: float,
    global_candidates: _global.GlobalConstantJumpCandidateSet,
    primary: _BaseHypothesis,
    boundary_seeds: tuple[_BoundarySeed, ...],
    short_observations: tuple[LocalTempoObservation, ...],
    config: TempoTrackConfig,
) -> tuple[_EarlyHalfPrimaryPrefixStepChoice, ...]:
    if (
        primary.bpm < _EARLY_HALF_PREFIX_MIN_PRIMARY_BPM
        or not boundary_seeds
        or not short_observations
    ):
        return ()
    pairs: list[tuple[_BoundarySeed, _BoundarySeed]] = []
    for left_index, left in enumerate(boundary_seeds):
        if (
            left.rank_score < _EARLY_HALF_PREFIX_MIN_BOUNDARY_RANK
            or left.time_ms < _EARLY_HALF_PREFIX_MIN_TIME_MS
            or left.time_ms > _EARLY_HALF_PREFIX_MAX_TIME_MS
        ):
            continue
        for right in boundary_seeds[left_index + 1 :]:
            gap_ms = right.time_ms - left.time_ms
            if gap_ms > _EARLY_HALF_PREFIX_MAX_GAP_MS:
                break
            if (
                gap_ms < _EARLY_HALF_PREFIX_MIN_GAP_MS
                or right.rank_score < _EARLY_HALF_PREFIX_MIN_BOUNDARY_RANK
                or right.time_ms < _EARLY_HALF_PREFIX_MIN_TIME_MS
                or right.time_ms > _EARLY_HALF_PREFIX_MAX_TIME_MS
            ):
                continue
            pairs.append((left, right))
    pairs.sort(
        key=lambda value: (
            -min(value[0].rank_score, value[1].rank_score),
            -(value[0].rank_score + value[1].rank_score),
            value[0].time_ms,
            value[1].time_ms,
        )
    )
    for left, right in pairs:
        choice = _early_half_primary_prefix_step_choice_from_pair(
            beat_signal=beat_signal,
            frame_rate_hz=frame_rate_hz,
            duration_ms=duration_ms,
            global_candidates=global_candidates,
            primary=primary,
            left=left,
            right=right,
            short_observations=short_observations,
            config=config,
        )
        if choice is not None:
            return (choice,)
    return ()


def _early_half_primary_prefix_step_choice_from_pair(
    *,
    beat_signal: NDArray[np.float64],
    frame_rate_hz: float,
    duration_ms: float,
    global_candidates: _global.GlobalConstantJumpCandidateSet,
    primary: _BaseHypothesis,
    left: _BoundarySeed,
    right: _BoundarySeed,
    short_observations: tuple[LocalTempoObservation, ...],
    config: TempoTrackConfig,
) -> _EarlyHalfPrimaryPrefixStepChoice | None:
    prefix_bpm = 0.5 * primary.bpm
    tolerance_bpm = max(2.0, 0.025 * primary.bpm)
    lookback_start_ms = max(0.0, right.time_ms - _EARLY_HALF_PREFIX_LOOKBACK_MS)
    lookback_end_ms = left.time_ms - _EARLY_HALF_PREFIX_LEFT_MARGIN_MS
    if lookback_end_ms <= lookback_start_ms:
        return None
    prefix_observations = tuple(
        observation
        for observation in short_observations
        if lookback_start_ms <= observation.center_time_ms <= lookback_end_ms
        and abs(observation.bpm - prefix_bpm) <= tolerance_bpm
    )
    if any(
        sum(observation.source == source for observation in prefix_observations)
        < _EARLY_HALF_PREFIX_MIN_SOURCE_OBSERVATIONS
        for source in ("beatthis", "raw_audio")
    ):
        return None
    estimator_observations = tuple(
        value for value in prefix_observations if value.source == "beatthis"
    )
    values = np.asarray([value.bpm for value in estimator_observations], dtype=np.float64)
    weights = np.asarray(
        [max(1e-3, value.strength) for value in estimator_observations],
        dtype=np.float64,
    )
    low_bpm = _weighted_median(values, weights)
    if abs(2.0 * low_bpm - primary.bpm) > max(4.0, 0.025 * primary.bpm):
        return None
    low_origin_time_ms = _phase_origin_for_bpm(
        global_candidates,
        bpm=low_bpm,
        beat_signal=beat_signal,
        frame_rate_hz=frame_rate_hz,
        duration_ms=duration_ms,
        radius_ms=config.beat_event_radius_ms,
    )
    snapped = _snapped_constant_boundary(
        origin_time_ms=low_origin_time_ms,
        bpm=low_bpm,
        target_time_ms=right.time_ms,
        duration_ms=duration_ms,
    )
    if snapped is None:
        return None
    boundary_beat, boundary_time_ms = snapped
    if abs(boundary_time_ms - right.time_ms) > _EXP026_BOUNDARY_TOLERANCE_MS:
        return None
    end_beat = _terminal_beat_after_boundary(
        boundary_beat=boundary_beat,
        boundary_time_ms=boundary_time_ms,
        duration_ms=duration_ms,
        bpm=primary.bpm,
    )
    try:
        curve = PhaseContinuousTimingCurve(
            origin_beat=0,
            origin_time_ms=low_origin_time_ms,
            sections=(
                ConstantTempoSection(0, boundary_beat, low_bpm),
                ConstantTempoSection(boundary_beat, end_beat, primary.bpm),
            ),
        )
    except ValueError:
        return None
    if curve.end_time_ms + 1e-6 < duration_ms:
        return None
    collapsed = _collapsed_constant_counterfactual(
        curve,
        duration_ms=duration_ms,
        base_bpm=primary.bpm,
        config=config,
    )
    beatthis_delta = _generalized_support_delta(
        beat_signal,
        frame_rate_hz=frame_rate_hz,
        candidate=curve,
        collapsed=collapsed,
        radius_ms=config.beat_event_radius_ms,
    )
    if not math.isfinite(beatthis_delta):
        return None
    return _EarlyHalfPrimaryPrefixStepChoice(
        proposal=_CurveProposal(
            curve=curve,
            source=_EARLY_HALF_PRIMARY_PREFIX_STEP_SOURCE,
            score=float(
                beatthis_delta
                + 0.001 * min(left.rank_score, right.rank_score)
                + 0.0001 * len(prefix_observations)
            ),
            collapse_bpm=float(primary.bpm),
        ),
        left_boundary_time_ms=float(left.time_ms),
        right_boundary_time_ms=float(right.time_ms),
        boundary_time_ms=float(boundary_time_ms),
        left_rank_score=float(left.rank_score),
        right_rank_score=float(right.rank_score),
        beatthis_support_delta=float(beatthis_delta),
        prefix_observation_count=len(prefix_observations),
    )


def _ramp_proposals(
    observations: tuple[LocalTempoObservation, ...],
    *,
    primary: _BaseHypothesis,
    duration_ms: float,
    shared_end_beat: int,
    config: TempoTrackConfig,
) -> tuple[_CurveProposal, ...]:
    proposals: list[_CurveProposal] = []
    for source in ("raw_audio", "beatthis"):
        source_observations = tuple(
            observation for observation in observations if observation.source == source
        )
        fit = _fit_linear_tempo(source_observations, config=config)
        if fit is None:
            continue
        slope_bpm_per_second, r_squared, mean_strength = fit
        available_seconds = (duration_ms - primary.origin_time_ms) / 1000.0
        endpoint_delta = slope_bpm_per_second * available_seconds
        if abs(endpoint_delta) < 8.0 or r_squared < 0.30:
            continue
        required_average = (
            60.0 * shared_end_beat / max(available_seconds, 1e-9)
        )
        for scale in (0.85, 1.0, 1.15):
            delta = endpoint_delta * scale
            start_bpm = required_average - 0.5 * delta
            end_bpm = required_average + 0.5 * delta
            if not (
                config.minimum_bpm <= start_bpm <= config.maximum_bpm
                and config.minimum_bpm <= end_bpm <= config.maximum_bpm
            ):
                continue
            try:
                curve = PhaseContinuousTimingCurve(
                    origin_beat=0,
                    origin_time_ms=primary.origin_time_ms,
                    sections=(
                        LinearTimeRampSection(
                            start_beat=0,
                            end_beat=shared_end_beat,
                            start_bpm=float(start_bpm),
                            end_bpm=float(end_bpm),
                        ),
                    ),
                )
            except ValueError:
                continue
            proposals.append(
                _CurveProposal(
                    curve=curve,
                    source=f"linear_tempo_{source}",
                    score=float(
                        r_squared
                        + 0.25 * mean_strength
                        - 0.02 * abs(1.0 - scale)
                    ),
                    collapse_bpm=primary.bpm,
                )
            )
    proposals.sort(key=lambda value: (-value.score, value.curve.fingerprint_sha256))
    return tuple(proposals[: config.maximum_ramp_candidates])


def _fit_linear_tempo(
    observations: tuple[LocalTempoObservation, ...],
    *,
    config: TempoTrackConfig = DEFAULT_TEMPO_TRACK_CONFIG,
) -> tuple[float, float, float] | None:
    if len(observations) < 6:
        return None
    times = np.asarray(
        [observation.center_time_ms / 1000.0 for observation in observations],
        dtype=np.float64,
    )
    original_bpms = np.asarray(
        [observation.bpm for observation in observations], dtype=np.float64
    )
    weights = np.asarray(
        [max(observation.strength, 1e-3) for observation in observations],
        dtype=np.float64,
    )
    upper_family = original_bpms[original_bpms >= 100.0]
    reference_bpm = float(
        np.median(upper_family) if upper_family.size >= 3 else np.median(original_bpms)
    )
    bpms = np.asarray(
        [
            _normalize_observation_alias(
                float(bpm),
                reference_bpm=reference_bpm,
                config=config,
            )
            for bpm in original_bpms
        ],
        dtype=np.float64,
    )
    centered_times = times - float(np.average(times, weights=weights))
    centered_bpms = bpms - float(np.average(bpms, weights=weights))
    denominator = float(np.sum(weights * centered_times * centered_times))
    if denominator <= 0.0:
        return None
    slope = float(np.sum(weights * centered_times * centered_bpms) / denominator)
    intercept = float(np.average(bpms - slope * times, weights=weights))
    fitted = intercept + slope * times
    residual = float(np.sum(weights * np.square(bpms - fitted)))
    total = float(
        np.sum(weights * np.square(bpms - float(np.average(bpms, weights=weights))))
    )
    r_squared = 0.0 if total <= 0.0 else max(0.0, 1.0 - residual / total)
    return slope, r_squared, float(np.mean(weights, dtype=np.float64))


def _normalize_observation_alias(
    bpm: float,
    *,
    reference_bpm: float,
    config: TempoTrackConfig,
) -> float:
    alternatives = tuple(
        value
        for value in (0.5 * bpm, bpm, 2.0 * bpm)
        if config.minimum_bpm <= value <= config.maximum_bpm
    )
    if not alternatives:
        return bpm
    return min(alternatives, key=lambda value: (abs(value - reference_bpm), -value))


def _bounded_proposals(
    constants: tuple[_CurveProposal, ...],
    jumps: tuple[_CurveProposal, ...],
    ramps: tuple[_CurveProposal, ...],
    *,
    config: TempoTrackConfig,
) -> tuple[_CurveProposal, ...]:
    ordered = (
        tuple(sorted(constants, key=lambda value: (-value.score, value.curve.fingerprint_sha256)))
        + tuple(sorted(jumps, key=lambda value: (-value.score, value.curve.fingerprint_sha256)))
        + tuple(
            sorted(
                ramps,
                key=lambda value: (
                    (
                        0
                        if value.source == _LOCALIZED_BACKBONE_RAMP_SOURCE
                        else 1
                        if value.source == _EARLY_HALF_PRIMARY_PREFIX_STEP_SOURCE
                        else 2
                        if value.source == _BOUNDARY_NOMINAL_BACKBONE_SOURCE
                        else 3
                    ),
                    -value.score,
                    value.curve.fingerprint_sha256,
                ),
            )
        )
    )
    retained: list[_CurveProposal] = []
    seen: set[str] = set()
    for proposal in ordered:
        fingerprint = proposal.curve.fingerprint_sha256
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        retained.append(proposal)
        if len(retained) >= config.maximum_candidates:
            break
    return tuple(retained)


# Public verb synonym for callers that treat this as evidence extraction.
extract_tempo_track = generate_timing_candidates
