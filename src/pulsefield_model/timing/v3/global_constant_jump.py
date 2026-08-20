from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from pulsefield_model.timing.schema import FrameTimingPrediction
from pulsefield_model.timing.v3.schema import ConstantTimingSection, TimingV3Grid


CANDIDATE_CONTRACT_VERSION = "timing-v3-exp004-candidate-contract-v1"
PULSE_CORRELATION_VERSION = "pulse_correlation_v1"
BOUNDARY_CANDIDATE_SCORE_VERSION = "boundary_candidate_score_v1"

VARIANT_CJ0 = "CJ0"
VARIANT_CJ1 = "CJ1"
VARIANT_CJ2 = "CJ2"
VARIANT_CJ3 = "CJ3"
GLOBAL_CONSTANT_JUMP_VARIANTS = (VARIANT_CJ0, VARIANT_CJ1, VARIANT_CJ2, VARIANT_CJ3)

REASON_NO_ORIGIN_CANDIDATE = "no_origin_candidate"
REASON_NO_GLOBAL_CONSTANT_JUMP_PATH = "no_global_constant_jump_path"
REASON_EDGE_ATTEMPT_CAP_EXCEEDED = "edge_attempt_cap_exceeded"
REASON_SCHEMA_CONSTRUCTION_FAILED = "timing_v3_schema_construction_failed"


@dataclass(frozen=True)
class GlobalConstantJumpConstants:
    candidate_contract_version: str = CANDIDATE_CONTRACT_VERSION
    hard_min_bpm: float = 20.0
    hard_max_bpm: float = 1000.0
    preferred_min_bpm: float = 80.0
    preferred_max_bpm: float = 240.0
    expected_frame_rate_hz: float = 50.0
    autocorrelation_top_lag_count: int = 16
    alias_multipliers: tuple[float, ...] = (0.25, 1.0 / 3.0, 0.5, 1.0, 2.0, 3.0, 4.0)
    bpm_grid_step: float = 0.5
    fractional_bpm_parts: tuple[float, ...] = (
        1.0 / 9.0,
        1.0 / 8.0,
        2.0 / 9.0,
        1.0 / 3.0,
        3.0 / 8.0,
        4.0 / 9.0,
        5.0 / 9.0,
        5.0 / 8.0,
        2.0 / 3.0,
        7.0 / 9.0,
        7.0 / 8.0,
        8.0 / 9.0,
    )
    pulse_width_ms: float = 40.0
    peak_grid_tolerance_ms: float = 45.0
    peak_grid_tolerance_beat_fraction: float = 0.15
    boundary_support_tolerance_ms: float = 60.0
    min_section_duration_ms: float = 8000.0
    max_section_count: int = 20
    beam_width: int = 64
    max_origin_candidates: int = 16
    max_interior_boundary_candidates: int = 192
    max_tempo_candidates_retained: int = 256
    max_beat_count_candidates_per_edge: int = 16
    max_whole_track_tempo_candidates: int = 192
    max_peak_interval_tempo_candidates: int = 64
    origin_tempo_scan_count: int = 64
    origin_offset_grid_ms: float = 20.0
    origin_refine_radius_ms: int = 10
    boundary_merge_ms: float = 8000.0
    boundary_bin_ms: float = 60000.0
    default_edge_attempt_cap: int = 120000

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "candidate_contract_version": self.candidate_contract_version,
            "hard_bpm_guard": [self.hard_min_bpm, self.hard_max_bpm],
            "preferred_bpm_band": [self.preferred_min_bpm, self.preferred_max_bpm],
            "expected_frame_rate_hz": self.expected_frame_rate_hz,
            "autocorrelation_top_lag_count": self.autocorrelation_top_lag_count,
            "alias_multipliers": list(self.alias_multipliers),
            "bpm_grid_step": self.bpm_grid_step,
            "fractional_bpm_parts": list(self.fractional_bpm_parts),
            "pulse_width_ms": self.pulse_width_ms,
            "peak_grid_matching_tolerance_ms": self.peak_grid_tolerance_ms,
            "peak_grid_matching_tolerance_beat_fraction": self.peak_grid_tolerance_beat_fraction,
            "boundary_support_tolerance_ms": self.boundary_support_tolerance_ms,
            "minimum_section_duration_ms": self.min_section_duration_ms,
            "maximum_section_count": self.max_section_count,
            "beam_width": self.beam_width,
            "maximum_origin_candidates": self.max_origin_candidates,
            "maximum_interior_boundary_candidates": self.max_interior_boundary_candidates,
            "maximum_tempo_candidates_retained": self.max_tempo_candidates_retained,
            "maximum_beat_count_candidates_per_edge": self.max_beat_count_candidates_per_edge,
            "maximum_whole_track_tempo_candidates": self.max_whole_track_tempo_candidates,
            "maximum_peak_interval_tempo_candidates": self.max_peak_interval_tempo_candidates,
            "origin_tempo_scan_count": self.origin_tempo_scan_count,
            "origin_offset_grid_ms": self.origin_offset_grid_ms,
            "origin_refine_radius_ms": self.origin_refine_radius_ms,
            "boundary_merge_ms": self.boundary_merge_ms,
            "boundary_bin_ms": self.boundary_bin_ms,
            "default_edge_attempt_cap": self.default_edge_attempt_cap,
        }


GLOBAL_CONSTANT_JUMP_CONSTANTS = GlobalConstantJumpConstants()
GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON = json.dumps(
    GLOBAL_CONSTANT_JUMP_CONSTANTS.to_jsonable(),
    sort_keys=True,
    separators=(",", ":"),
)
GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON_SHA256 = hashlib.sha256(
    GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON.encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class MaterializedPeak:
    frame_index: int
    refined_frame: float
    time_ms: float
    confidence: float


@dataclass(frozen=True)
class TempoCandidate:
    bpm: float
    source: str
    score: float


@dataclass(frozen=True)
class OriginCandidate:
    anchor_id: int
    time_ms: float
    bpm: float
    score: float


@dataclass(frozen=True)
class BoundaryCandidate:
    anchor_id: int
    time_ms: float
    source_peak_index: int
    source_peak_time_ms: float
    source_peak_confidence: float
    rank_score: float
    evidence_mode: Literal["ordinary", "super"]
    left_period_ms: float
    right_period_ms: float
    ordinary_score: float | None
    super_score: float | None
    downbeat_bonus: float
    nearest_downbeat_distance_ms: float | None

    @property
    def displacement_from_source_peak_ms(self) -> float:
        return abs(self.time_ms - self.source_peak_time_ms)


@dataclass(frozen=True)
class GlobalConstantJumpCandidateDiagnostics:
    candidate_contract_version: str
    constants_json_sha256: str
    pulse_correlation_version: str
    boundary_candidate_score_version: str
    frame_count: int
    frame_rate_hz: float
    coverage_start_ms: float
    coverage_end_ms: float
    min_period_frames: int
    max_period_frames: int
    beat_peak_count: int
    downbeat_peak_count: int
    tempo_candidate_count: int
    origin_candidate_count: int
    boundary_candidate_count: int
    input_signal_sha256: str
    candidate_fingerprint: str


@dataclass(frozen=True)
class GlobalConstantJumpCandidateSet:
    beat_peaks: tuple[MaterializedPeak, ...]
    downbeat_peaks: tuple[MaterializedPeak, ...]
    tempo_candidates: tuple[TempoCandidate, ...]
    origin_candidates: tuple[OriginCandidate, ...]
    boundary_candidates: tuple[BoundaryCandidate, ...]
    diagnostics: GlobalConstantJumpCandidateDiagnostics


@dataclass(frozen=True)
class GlobalConstantJumpDiagnostics:
    variant: str
    candidate_contract_version: str
    constants_json_sha256: str
    coverage_start_ms: float
    coverage_end_ms: float
    frame_count: int
    frame_rate_hz: float
    min_period_frames: int
    max_period_frames: int
    beat_peak_count: int
    downbeat_peak_count: int
    tempo_candidate_count: int
    origin_candidate_count: int
    boundary_candidate_count: int
    section_attempt_count: int
    edge_count_cache_size: int
    section_score_cache_size: int
    beam_pruned_state_count: int
    selected_section_count: int
    selected_origin_time_ms: float | None
    selected_downbeat_phase: int | None
    objective: float | None
    duration_objective: float | None
    transition_objective: float | None
    alias_switch_count: int
    max_boundary_displacement_ms: float
    fallback_reason: str | None
    input_signal_sha256: str
    candidate_fingerprint: str
    replay_fingerprint: str
    grid_fingerprint: str | None


@dataclass(frozen=True)
class GlobalConstantJumpResult:
    variant: str
    grid: TimingV3Grid | None
    diagnostics: GlobalConstantJumpDiagnostics
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.grid is not None


@dataclass(frozen=True)
class _Anchor:
    anchor_id: int
    kind: Literal["origin", "boundary"]
    time_ms: float
    rank_score: float
    boundary: BoundaryCandidate | None = None
    origin: OriginCandidate | None = None


@dataclass(frozen=True)
class _BeatCountCandidate:
    count: int
    bpm: float
    rank: int
    tempo_distance: float
    beat_support: float


@dataclass(frozen=True)
class _SectionSpec:
    start_beat: int
    end_beat: int
    bpm: float


@dataclass(frozen=True)
class _SectionScore:
    valid: bool
    cost: float
    beat_support_cost: float
    peak_recall_precision_cost: float
    downbeat_phase_cost: float
    bpm_prior_cost: float
    beat_count_prior_cost: float
    section_duration_cost: float


@dataclass(frozen=True)
class _TransitionScore:
    cost: float
    alias_switch_cost: float
    alias_switch_increment: int
    jump_size_cost: float
    boundary_support_cost: float


@dataclass(frozen=True)
class _State:
    anchor: _Anchor
    section_count: int
    beat_at_anchor: int
    prev_bpm: float | None
    prev_alias_family: float | None
    downbeat_phase: int | None
    origin_time_ms: float
    duration_objective: float
    transition_objective: float
    objective: float
    alias_switch_count: int
    max_boundary_displacement_ms: float
    sections: tuple[_SectionSpec, ...]
    edge_tuples: tuple[tuple[int, int, int, float], ...]
    replay_key: tuple[Any, ...]


@dataclass(frozen=True)
class _StateOrderKeyParts:
    objective: float
    section_count: int
    alias_switch_count: int
    max_boundary_displacement_ms: float
    origin_time_ms: float
    edge_tuples: tuple[tuple[int, int, int, float], ...]
    edge_suffix: tuple[int, int, int, float] | None
    replay_key: tuple[Any, ...]
    replay_suffix: Any | None

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _StateOrderKeyParts):
            return NotImplemented
        if self.objective != other.objective:
            return self.objective < other.objective
        if self.section_count != other.section_count:
            return self.section_count < other.section_count
        if self.alias_switch_count != other.alias_switch_count:
            return self.alias_switch_count < other.alias_switch_count
        if self.max_boundary_displacement_ms != other.max_boundary_displacement_ms:
            return self.max_boundary_displacement_ms < other.max_boundary_displacement_ms
        if self.origin_time_ms != other.origin_time_ms:
            return self.origin_time_ms < other.origin_time_ms
        edge_comparison = _compare_tuple_with_optional_suffix(
            self.edge_tuples,
            self.edge_suffix,
            other.edge_tuples,
            other.edge_suffix,
        )
        if edge_comparison != 0:
            return edge_comparison < 0
        return (
            _compare_tuple_with_optional_suffix(
                self.replay_key,
                self.replay_suffix,
                other.replay_key,
                other.replay_suffix,
            )
            < 0
        )


@dataclass(frozen=True)
class _PendingInteriorState:
    parent: _State
    anchor: _Anchor
    section_count: int
    beat_at_anchor: int
    bpm: float
    first_start_beat: int
    duration_objective: float
    transition_objective: float
    objective: float
    alias_switch_count: int
    max_boundary_displacement_ms: float
    edge_tuple: tuple[int, int, int, float]
    replay_item: tuple[int, int, tuple[int, int, int, float]]
    order_key: _StateOrderKeyParts


class _AttemptCapExceeded(RuntimeError):
    pass


class _SearchContext:
    def __init__(
        self,
        *,
        prediction: FrameTimingPrediction,
        candidates: GlobalConstantJumpCandidateSet,
        constants: GlobalConstantJumpConstants,
        attempt_cap: int,
        shared_count_cache: dict[tuple[int, int], tuple[_BeatCountCandidate, ...]] | None = None,
    ) -> None:
        self.constants = constants
        self.attempt_cap = attempt_cap
        self.attempt_count = 0
        self.beat_signal = np.asarray(prediction.beat_prob, dtype=np.float64)
        self.downbeat_signal = np.asarray(prediction.downbeat_prob, dtype=np.float64)
        (
            self.frame_count,
            self.frame_rate_hz,
            self.coverage_start_ms,
            self.coverage_end_ms,
            _,
            _,
        ) = _prediction_geometry(self.beat_signal, prediction.frame_rate_hz, constants)
        self.frame_times_ms = np.arange(self.frame_count, dtype=np.float64) * (1000.0 / self.frame_rate_hz)
        self.beat_peak_times_ms = np.asarray([peak.time_ms for peak in candidates.beat_peaks], dtype=np.float64)
        self.beat_peak_confidences = np.asarray([peak.confidence for peak in candidates.beat_peaks], dtype=np.float64)
        self.downbeat_peak_times_ms = np.asarray([peak.time_ms for peak in candidates.downbeat_peaks], dtype=np.float64)
        self.downbeat_peak_confidences = np.asarray(
            [peak.confidence for peak in candidates.downbeat_peaks],
            dtype=np.float64,
        )
        self.tempo_bpms = tuple(candidate.bpm for candidate in candidates.tempo_candidates)
        self.ordered_tempo_bpms = tuple(
            sorted(bpm for bpm in self.tempo_bpms if math.isfinite(bpm) and bpm > 0.0)
        )
        self.count_cache = shared_count_cache if shared_count_cache is not None else {}
        self.count_cache_visited_keys: set[tuple[int, int]] = set()
        self.section_score_cache: dict[tuple[Any, ...], _SectionScore] = {}
        self.terminal_score_cache: dict[tuple[Any, ...], _SectionScore] = {}
        self.transition_score_cache: dict[tuple[Any, ...], _TransitionScore] = {}
        self.boundary_support_cost_cache: dict[tuple[float, bool], float] = {}
        self.alias_switch_cost_cache: dict[tuple[float, float], float] = {}
        self.alias_family_cache: dict[float, float] = {}
        self.terminal_bpms_cache: dict[tuple[float | None, float | None, bool], tuple[float, ...]] = {}
        self.outgoing_boundary_anchors_cache: dict[
            tuple[str, int, float],
            tuple[_Anchor, ...],
        ] = {}
        self.interior_edge_score_bundle_cache: dict[
            tuple[str, int, float, str, int | None],
            tuple[tuple[_Anchor, _BeatCountCandidate, _SectionScore], ...],
        ] = {}
        self.interior_edge_score_bundle_enabled = True
        self.cj1_monotone_rejection_enabled = True
        self.search_trace_sha256 = hashlib.sha256(b"timing-v3-exp004-search-trace-v1")
        self.beam_pruned_state_count = 0

    def check_attempt(self, cache_key: tuple[Any, ...]) -> None:
        if self.attempt_count >= self.attempt_cap:
            self.record_search_trace_event("cap_rejected", cache_key)
            raise _AttemptCapExceeded
        self.attempt_count += 1

    @property
    def edge_count_cache_size(self) -> int:
        return len(self.count_cache_visited_keys)

    def record_search_trace_event(self, event: str, key: tuple[Any, ...]) -> None:
        payload = (event, key)
        self.search_trace_sha256.update(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        )
        self.search_trace_sha256.update(b"\n")

    def search_trace_fingerprint(self) -> str:
        return self.search_trace_sha256.hexdigest()


def fit_global_constant_jump(
    prediction: FrameTimingPrediction,
    *,
    variant: str = VARIANT_CJ3,
    attempt_cap: int = GLOBAL_CONSTANT_JUMP_CONSTANTS.default_edge_attempt_cap,
    candidate_set: GlobalConstantJumpCandidateSet | None = None,
) -> GlobalConstantJumpResult:
    if variant not in GLOBAL_CONSTANT_JUMP_VARIANTS:
        raise ValueError(f"variant must be one of {GLOBAL_CONSTANT_JUMP_VARIANTS!r}, got {variant!r}")
    attempt_cap = _validated_attempt_cap(attempt_cap)

    candidates = (
        extract_global_constant_jump_candidates(prediction)
        if candidate_set is None
        else _validated_candidate_set_for_prediction(candidate_set, prediction)
    )
    return _fit_global_constant_jump_with_candidates(
        prediction,
        variant=variant,
        attempt_cap=attempt_cap,
        candidates=candidates,
        shared_count_cache=None,
    )


def _fit_global_constant_jump_with_candidates(
    prediction: FrameTimingPrediction,
    *,
    variant: str,
    attempt_cap: int,
    candidates: GlobalConstantJumpCandidateSet,
    shared_count_cache: dict[tuple[int, int], tuple[_BeatCountCandidate, ...]] | None,
) -> GlobalConstantJumpResult:
    context = _SearchContext(
        prediction=prediction,
        candidates=candidates,
        constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
        attempt_cap=attempt_cap,
        shared_count_cache=shared_count_cache,
    )
    if not candidates.origin_candidates:
        return _failure_result(variant, candidates, context, REASON_NO_ORIGIN_CANDIDATE)

    try:
        complete = (
            _assemble_constant_only(candidates, context, variant=variant)
            if variant == VARIANT_CJ0
            else _assemble_beam(candidates, context, variant=variant)
        )
    except _AttemptCapExceeded:
        return _failure_result(variant, candidates, context, REASON_EDGE_ATTEMPT_CAP_EXCEEDED)

    if complete is None:
        return _failure_result(variant, candidates, context, REASON_NO_GLOBAL_CONSTANT_JUMP_PATH)

    grid = _grid_from_state(complete, context)
    if grid is None:
        return _failure_result(variant, candidates, context, REASON_SCHEMA_CONSTRUCTION_FAILED)
    diagnostics = _success_diagnostics(variant, candidates, context, complete, grid)
    return GlobalConstantJumpResult(variant=variant, grid=grid, diagnostics=diagnostics, reason=None)


def iter_global_constant_jump_variants(
    prediction: FrameTimingPrediction,
    *,
    variants: Sequence[str] = GLOBAL_CONSTANT_JUMP_VARIANTS,
    attempt_cap: int = GLOBAL_CONSTANT_JUMP_CONSTANTS.default_edge_attempt_cap,
    candidate_set: GlobalConstantJumpCandidateSet | None = None,
) -> Iterator[tuple[str, GlobalConstantJumpResult]]:
    normalized_variants = _normalized_variants(variants)
    attempt_cap = _validated_attempt_cap(attempt_cap)
    candidates = (
        extract_global_constant_jump_candidates(prediction)
        if candidate_set is None
        else _validated_candidate_set_for_prediction(candidate_set, prediction)
    )
    shared_count_cache: dict[tuple[int, int], tuple[_BeatCountCandidate, ...]] = {}

    def _iter() -> Iterator[tuple[str, GlobalConstantJumpResult]]:
        for variant in normalized_variants:
            result = _fit_global_constant_jump_with_candidates(
                prediction,
                variant=variant,
                attempt_cap=attempt_cap,
                candidates=candidates,
                shared_count_cache=shared_count_cache,
            )
            yield variant, result

    return _iter()


def fit_global_constant_jump_variants(
    prediction: FrameTimingPrediction,
    *,
    variants: Sequence[str] = GLOBAL_CONSTANT_JUMP_VARIANTS,
    attempt_cap: int = GLOBAL_CONSTANT_JUMP_CONSTANTS.default_edge_attempt_cap,
    candidate_set: GlobalConstantJumpCandidateSet | None = None,
) -> Mapping[str, GlobalConstantJumpResult]:
    return {
        variant: result
        for variant, result in iter_global_constant_jump_variants(
            prediction,
            variants=variants,
            attempt_cap=attempt_cap,
            candidate_set=candidate_set,
        )
    }


def _validated_attempt_cap(attempt_cap: int) -> int:
    if not isinstance(attempt_cap, int) or isinstance(attempt_cap, bool) or attempt_cap <= 0:
        raise ValueError(f"attempt_cap must be positive, got {attempt_cap!r}")
    return int(attempt_cap)


def _normalized_variants(variants: Sequence[str]) -> tuple[str, ...]:
    if not variants:
        raise ValueError("variants must be non-empty")
    normalized_variants: list[str] = []
    for variant in variants:
        if variant not in GLOBAL_CONSTANT_JUMP_VARIANTS:
            raise ValueError(f"variant must be one of {GLOBAL_CONSTANT_JUMP_VARIANTS!r}, got {variant!r}")
        if variant not in normalized_variants:
            normalized_variants.append(variant)
    return tuple(normalized_variants)


def extract_global_constant_jump_candidates(
    prediction: FrameTimingPrediction,
) -> GlobalConstantJumpCandidateSet:
    constants = GLOBAL_CONSTANT_JUMP_CONSTANTS
    beat_signal = np.asarray(prediction.beat_prob, dtype=np.float64)
    downbeat_signal = np.asarray(prediction.downbeat_prob, dtype=np.float64)
    frame_count, frame_rate_hz, coverage_start_ms, coverage_end_ms, min_period_frames, max_period_frames = (
        _prediction_geometry(beat_signal, prediction.frame_rate_hz, constants)
    )

    beat_peaks = materialize_global_constant_jump_peaks(beat_signal, frame_rate_hz=frame_rate_hz)
    downbeat_peaks = materialize_global_constant_jump_peaks(downbeat_signal, frame_rate_hz=frame_rate_hz)
    tempo_candidates = _tempo_candidates(
        beat_signal,
        beat_peaks,
        frame_rate_hz=frame_rate_hz,
        min_period_frames=min_period_frames,
        max_period_frames=max_period_frames,
        constants=constants,
    )
    origin_candidates = _origin_candidates(
        beat_signal,
        tempo_candidates,
        frame_rate_hz=frame_rate_hz,
        coverage_start_ms=coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
        constants=constants,
    )
    boundary_candidates = _boundary_candidates(
        beat_peaks,
        downbeat_peaks,
        coverage_start_ms=coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
        constants=constants,
    )
    input_signal_sha256 = _input_signal_sha256(beat_signal, downbeat_signal)
    candidate_fingerprint = _candidate_fingerprint(
        tempo_candidates=tempo_candidates,
        origin_candidates=origin_candidates,
        boundary_candidates=boundary_candidates,
        beat_peaks=beat_peaks,
        downbeat_peaks=downbeat_peaks,
        input_signal_sha256=input_signal_sha256,
    )
    diagnostics = GlobalConstantJumpCandidateDiagnostics(
        candidate_contract_version=CANDIDATE_CONTRACT_VERSION,
        constants_json_sha256=GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON_SHA256,
        pulse_correlation_version=PULSE_CORRELATION_VERSION,
        boundary_candidate_score_version=BOUNDARY_CANDIDATE_SCORE_VERSION,
        frame_count=frame_count,
        frame_rate_hz=frame_rate_hz,
        coverage_start_ms=coverage_start_ms,
        coverage_end_ms=float(coverage_end_ms),
        min_period_frames=min_period_frames,
        max_period_frames=max_period_frames,
        beat_peak_count=len(beat_peaks),
        downbeat_peak_count=len(downbeat_peaks),
        tempo_candidate_count=len(tempo_candidates),
        origin_candidate_count=len(origin_candidates),
        boundary_candidate_count=len(boundary_candidates),
        input_signal_sha256=input_signal_sha256,
        candidate_fingerprint=candidate_fingerprint,
    )
    return GlobalConstantJumpCandidateSet(
        beat_peaks=beat_peaks,
        downbeat_peaks=downbeat_peaks,
        tempo_candidates=tempo_candidates,
        origin_candidates=origin_candidates,
        boundary_candidates=boundary_candidates,
        diagnostics=diagnostics,
    )


def _validated_candidate_set_for_prediction(
    candidate_set: GlobalConstantJumpCandidateSet,
    prediction: FrameTimingPrediction,
) -> GlobalConstantJumpCandidateSet:
    if not isinstance(candidate_set, GlobalConstantJumpCandidateSet):
        raise TypeError("candidate_set must be a GlobalConstantJumpCandidateSet")
    beat_signal = np.asarray(prediction.beat_prob, dtype=np.float64)
    frame_count, frame_rate_hz, coverage_start_ms, coverage_end_ms, min_period_frames, max_period_frames = (
        _prediction_geometry(
            beat_signal,
            prediction.frame_rate_hz,
            GLOBAL_CONSTANT_JUMP_CONSTANTS,
        )
    )
    diagnostics = candidate_set.diagnostics
    if diagnostics.candidate_contract_version != CANDIDATE_CONTRACT_VERSION:
        raise ValueError("candidate_set contract version does not match Exp004")
    if diagnostics.constants_json_sha256 != GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON_SHA256:
        raise ValueError("candidate_set constants hash does not match Exp004")
    if diagnostics.pulse_correlation_version != PULSE_CORRELATION_VERSION:
        raise ValueError("candidate_set pulse_correlation_version does not match Exp004")
    if diagnostics.boundary_candidate_score_version != BOUNDARY_CANDIDATE_SCORE_VERSION:
        raise ValueError("candidate_set boundary_candidate_score_version does not match Exp004")
    if diagnostics.frame_count != frame_count:
        raise ValueError("candidate_set frame_count does not match prediction")
    if diagnostics.frame_rate_hz != frame_rate_hz:
        raise ValueError("candidate_set frame_rate_hz does not match prediction")
    if diagnostics.coverage_start_ms != coverage_start_ms:
        raise ValueError("candidate_set coverage_start_ms does not match prediction")
    if diagnostics.coverage_end_ms != coverage_end_ms:
        raise ValueError("candidate_set coverage_end_ms does not match prediction")
    if diagnostics.min_period_frames != min_period_frames:
        raise ValueError("candidate_set min_period_frames does not match prediction")
    if diagnostics.max_period_frames != max_period_frames:
        raise ValueError("candidate_set max_period_frames does not match prediction")
    input_signal_sha256 = _input_signal_sha256(
        np.asarray(prediction.beat_prob, dtype=np.float64),
        np.asarray(prediction.downbeat_prob, dtype=np.float64),
    )
    if diagnostics.input_signal_sha256 != input_signal_sha256:
        raise ValueError("candidate_set input_signal_sha256 does not match prediction")
    _validate_candidate_payload(
        candidate_set,
        GLOBAL_CONSTANT_JUMP_CONSTANTS,
        coverage_start_ms=coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
    )
    recomputed_fingerprint = _candidate_fingerprint(
        tempo_candidates=candidate_set.tempo_candidates,
        origin_candidates=candidate_set.origin_candidates,
        boundary_candidates=candidate_set.boundary_candidates,
        beat_peaks=candidate_set.beat_peaks,
        downbeat_peaks=candidate_set.downbeat_peaks,
        input_signal_sha256=input_signal_sha256,
    )
    if diagnostics.candidate_fingerprint != recomputed_fingerprint:
        raise ValueError("candidate_set candidate_fingerprint does not match payload")
    return candidate_set


def _prediction_geometry(
    beat_signal: NDArray[np.float64],
    frame_rate_hz_value: float,
    constants: GlobalConstantJumpConstants,
) -> tuple[int, float, float, float, int, int]:
    frame_rate_hz = float(frame_rate_hz_value)
    frame_count = int(beat_signal.shape[0])
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if not math.isfinite(frame_rate_hz) or frame_rate_hz <= 0.0:
        raise ValueError(f"frame_rate_hz must be positive and finite, got {frame_rate_hz!r}")
    coverage_start_ms = 0.0
    coverage_end_ms = 1000.0 * frame_count / frame_rate_hz
    min_period_raw = frame_rate_hz * 60.0 / constants.hard_max_bpm
    max_period_raw = frame_rate_hz * 60.0 / constants.hard_min_bpm
    if (
        not math.isfinite(coverage_end_ms)
        or coverage_end_ms <= coverage_start_ms
        or not math.isfinite(min_period_raw)
        or not math.isfinite(max_period_raw)
    ):
        raise ValueError("derived coverage or period frame bounds are nonfinite")
    min_period_frames = int(math.ceil(min_period_raw))
    max_period_frames = int(math.floor(max_period_raw))
    if min_period_frames < 1 or max_period_frames < min_period_frames:
        raise ValueError("derived period frame bounds are invalid")
    return (
        frame_count,
        frame_rate_hz,
        coverage_start_ms,
        float(coverage_end_ms),
        min_period_frames,
        max_period_frames,
    )


def _boundary_candidate_cap_for_coverage(
    *,
    coverage_start_ms: float,
    coverage_end_ms: float,
    constants: GlobalConstantJumpConstants,
) -> int:
    duration_s = max(0.0, (coverage_end_ms - coverage_start_ms) / 1000.0)
    return min(
        constants.max_interior_boundary_candidates,
        max(16, int(math.ceil(duration_s / 4.0))),
    )


def _validate_candidate_payload(
    candidate_set: GlobalConstantJumpCandidateSet,
    constants: GlobalConstantJumpConstants,
    *,
    coverage_start_ms: float,
    coverage_end_ms: float,
) -> None:
    diagnostics = candidate_set.diagnostics
    if len(candidate_set.beat_peaks) != diagnostics.beat_peak_count:
        raise ValueError("candidate_set beat_peak_count does not match payload")
    if len(candidate_set.downbeat_peaks) != diagnostics.downbeat_peak_count:
        raise ValueError("candidate_set downbeat_peak_count does not match payload")
    if len(candidate_set.tempo_candidates) != diagnostics.tempo_candidate_count:
        raise ValueError("candidate_set tempo_candidate_count does not match payload")
    if len(candidate_set.origin_candidates) != diagnostics.origin_candidate_count:
        raise ValueError("candidate_set origin_candidate_count does not match payload")
    if len(candidate_set.boundary_candidates) != diagnostics.boundary_candidate_count:
        raise ValueError("candidate_set boundary_candidate_count does not match payload")
    if len(candidate_set.tempo_candidates) > constants.max_tempo_candidates_retained:
        raise ValueError("candidate_set tempo candidate cap exceeded")
    if len(candidate_set.origin_candidates) > constants.max_origin_candidates:
        raise ValueError("candidate_set origin candidate cap exceeded")
    boundary_candidate_cap = _boundary_candidate_cap_for_coverage(
        coverage_start_ms=coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
        constants=constants,
    )
    if len(candidate_set.boundary_candidates) > boundary_candidate_cap:
        raise ValueError("candidate_set boundary candidate cap exceeded")
    _validate_peaks(candidate_set.beat_peaks, name="beat_peaks")
    _validate_peaks(candidate_set.downbeat_peaks, name="downbeat_peaks")
    _validate_tempos(candidate_set.tempo_candidates, constants)
    _validate_origins(candidate_set.origin_candidates, constants)
    _validate_boundaries(
        candidate_set.boundary_candidates,
        candidate_set.beat_peaks,
        constants=constants,
        coverage_start_ms=coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
    )


def _validate_peaks(peaks: tuple[MaterializedPeak, ...], *, name: str) -> None:
    previous_time_ms = -math.inf
    previous_frame_index = -1
    for index, peak in enumerate(peaks):
        if not isinstance(peak.frame_index, int) or peak.frame_index < 0:
            raise ValueError(f"candidate_set {name}[{index}] frame_index is invalid")
        if peak.frame_index < previous_frame_index:
            raise ValueError(f"candidate_set {name} are not ordered by frame_index")
        if (
            not math.isfinite(peak.refined_frame)
            or not math.isfinite(peak.time_ms)
            or not math.isfinite(peak.confidence)
            or peak.confidence < 0.0
            or peak.confidence > 1.0
        ):
            raise ValueError(f"candidate_set {name}[{index}] contains invalid finite/range fields")
        if peak.time_ms <= previous_time_ms:
            raise ValueError(f"candidate_set {name} are not strictly ordered by time")
        previous_time_ms = peak.time_ms
        previous_frame_index = peak.frame_index


def _validate_tempos(
    candidates: tuple[TempoCandidate, ...],
    constants: GlobalConstantJumpConstants,
) -> None:
    seen: set[float] = set()
    for index, candidate in enumerate(candidates):
        key = round(candidate.bpm, 6)
        if key in seen:
            raise ValueError("candidate_set tempo candidates contain duplicate BPMs")
        seen.add(key)
        if (
            not _bpm_is_valid(candidate.bpm, constants)
            or not math.isfinite(candidate.score)
            or not candidate.source
        ):
            raise ValueError(f"candidate_set tempo_candidates[{index}] is invalid")


def _validate_origins(
    candidates: tuple[OriginCandidate, ...],
    constants: GlobalConstantJumpConstants,
) -> None:
    previous_key: tuple[float, float, float] | None = None
    for index, candidate in enumerate(candidates):
        if candidate.anchor_id != index:
            raise ValueError("candidate_set origin anchor IDs must be sequential")
        if (
            not math.isfinite(candidate.time_ms)
            or not _bpm_is_valid(candidate.bpm, constants)
            or not math.isfinite(candidate.score)
        ):
            raise ValueError(f"candidate_set origin_candidates[{index}] is invalid")
        key = (-candidate.score, candidate.time_ms, candidate.bpm)
        if previous_key is not None and key < previous_key:
            raise ValueError("candidate_set origin candidates are not in deterministic order")
        previous_key = key


def _validate_boundaries(
    candidates: tuple[BoundaryCandidate, ...],
    beat_peaks: tuple[MaterializedPeak, ...],
    *,
    constants: GlobalConstantJumpConstants,
    coverage_start_ms: float,
    coverage_end_ms: float,
) -> None:
    previous_time_ms = -math.inf
    for index, candidate in enumerate(candidates):
        if candidate.anchor_id != index:
            raise ValueError("candidate_set boundary anchor IDs must be sequential")
        if candidate.evidence_mode not in {"ordinary", "super"}:
            raise ValueError(f"candidate_set boundary_candidates[{index}] evidence_mode is invalid")
        finite_fields = (
            candidate.time_ms,
            candidate.source_peak_time_ms,
            candidate.source_peak_confidence,
            candidate.rank_score,
            candidate.left_period_ms,
            candidate.right_period_ms,
            candidate.downbeat_bonus,
            candidate.displacement_from_source_peak_ms,
        )
        if any(not math.isfinite(value) for value in finite_fields):
            raise ValueError(f"candidate_set boundary_candidates[{index}] contains nonfinite fields")
        if (
            not isinstance(candidate.source_peak_index, int)
            or candidate.source_peak_index < 0
            or candidate.source_peak_index >= len(beat_peaks)
            or candidate.source_peak_confidence < 0.0
            or candidate.source_peak_confidence > 1.0
            or candidate.left_period_ms <= 0.0
            or candidate.right_period_ms <= 0.0
        ):
            raise ValueError(f"candidate_set boundary_candidates[{index}] contains invalid range fields")
        if not (coverage_start_ms <= candidate.time_ms < coverage_end_ms):
            raise ValueError(f"candidate_set boundary_candidates[{index}] is outside coverage")
        source_peak = beat_peaks[candidate.source_peak_index]
        if (
            candidate.source_peak_time_ms != source_peak.time_ms
            or candidate.source_peak_confidence != source_peak.confidence
        ):
            raise ValueError("candidate_set boundary source peak fields are stale")
        if candidate.ordinary_score is not None and not math.isfinite(candidate.ordinary_score):
            raise ValueError(f"candidate_set boundary_candidates[{index}] ordinary_score is nonfinite")
        if candidate.super_score is not None and not math.isfinite(candidate.super_score):
            raise ValueError(f"candidate_set boundary_candidates[{index}] super_score is nonfinite")
        if (
            candidate.nearest_downbeat_distance_ms is not None
            and (
                not math.isfinite(candidate.nearest_downbeat_distance_ms)
                or candidate.nearest_downbeat_distance_ms < 0.0
            )
        ):
            raise ValueError(f"candidate_set boundary_candidates[{index}] downbeat distance is invalid")
        if candidate.time_ms <= previous_time_ms:
            raise ValueError("candidate_set boundary candidates are not strictly ordered by time")
        if previous_time_ms > -math.inf and candidate.time_ms - previous_time_ms <= constants.boundary_merge_ms:
            raise ValueError("candidate_set boundary candidates violate merge spacing")
        previous_time_ms = candidate.time_ms


def materialize_global_constant_jump_peaks(
    signal: Sequence[float] | NDArray[np.floating[Any]],
    *,
    frame_rate_hz: float,
) -> tuple[MaterializedPeak, ...]:
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("signal must be a 1-D vector")
    if not np.all(np.isfinite(values)):
        raise ValueError("signal must contain only finite values")
    if not math.isfinite(frame_rate_hz) or frame_rate_hz <= 0.0:
        raise ValueError(f"frame_rate_hz must be positive and finite, got {frame_rate_hz!r}")
    if values.shape[0] < 3:
        return ()

    threshold = max(float(np.mean(values) + np.std(values)), 0.35 * float(np.max(values)))
    interior = values[1:-1]
    candidate_frames = (
        np.nonzero(
            (interior >= values[:-2])
            & (interior > values[2:])
            & (interior >= threshold)
        )[0]
        + 1
    )

    peaks: list[MaterializedPeak] = []
    for frame_value in candidate_frames:
        frame = int(frame_value)
        refined_frame = float(frame)
        denominator = values[frame - 1] - 2.0 * values[frame] + values[frame + 1]
        if denominator != 0.0:
            offset = 0.5 * (values[frame - 1] - values[frame + 1]) / denominator
            if math.isfinite(offset) and -0.5 <= offset <= 0.5:
                refined_frame = float(frame + offset)
        peaks.append(
            MaterializedPeak(
                frame_index=int(frame),
                refined_frame=refined_frame,
                time_ms=1000.0 * refined_frame / frame_rate_hz,
                confidence=float(np.clip(values[frame], 0.0, 1.0)),
            )
        )
    return tuple(peaks)


def boundary_candidate_score_v1(
    beat_peaks: Sequence[MaterializedPeak],
    downbeat_peaks: Sequence[MaterializedPeak],
    peak_index: int,
) -> BoundaryCandidate | None:
    constants = GLOBAL_CONSTANT_JUMP_CONSTANTS
    peaks = tuple(beat_peaks)
    if peak_index < 0 or peak_index >= len(peaks):
        raise IndexError("peak_index is out of range")
    candidate = _raw_boundary_candidate(peaks, tuple(downbeat_peaks), peak_index, constants)
    if candidate is None:
        return None
    return _boundary_candidate_from_raw(candidate, anchor_id=0)


def pulse_correlation_v1(
    signal: Sequence[float] | NDArray[np.floating[Any]],
    frame_times_ms: Sequence[float] | NDArray[np.floating[Any]],
    *,
    tau_ms: float,
    bpm: float,
    start_ms: float,
    end_ms: float,
    pulse_width_ms: float = GLOBAL_CONSTANT_JUMP_CONSTANTS.pulse_width_ms,
) -> float:
    values = np.asarray(signal, dtype=np.float64)
    times = np.asarray(frame_times_ms, dtype=np.float64)
    if values.ndim != 1 or times.ndim != 1:
        raise ValueError("signal and frame_times_ms must be 1-D vectors")
    if values.shape != times.shape:
        raise ValueError("signal and frame_times_ms must have the same shape")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(times)):
        raise ValueError("signal and frame_times_ms must contain only finite values")
    if (
        not math.isfinite(tau_ms)
        or not math.isfinite(bpm)
        or bpm <= 0.0
        or not math.isfinite(start_ms)
        or not math.isfinite(end_ms)
        or end_ms < start_ms
        or not math.isfinite(pulse_width_ms)
        or pulse_width_ms <= 0.0
    ):
        raise ValueError("tau_ms, positive bpm, nondecreasing interval, and pulse_width_ms must be finite")
    start, end = _frame_interval(times, start_ms, end_ms)
    return _pulse_correlation_slice(
        values[start:end],
        times[start:end],
        tau_ms=tau_ms,
        bpm=bpm,
        pulse_width_ms=pulse_width_ms,
    ).correlation


def _tempo_candidates(
    beat_signal: NDArray[np.float64],
    beat_peaks: tuple[MaterializedPeak, ...],
    *,
    frame_rate_hz: float,
    min_period_frames: int,
    max_period_frames: int,
    constants: GlobalConstantJumpConstants,
) -> tuple[TempoCandidate, ...]:
    seen: set[float] = set()
    whole_track: list[TempoCandidate] = []
    for lag, score in _top_autocorrelation_lags(
        beat_signal,
        min_period_frames=min_period_frames,
        max_period_frames=max_period_frames,
        limit=constants.autocorrelation_top_lag_count,
    ):
        base_bpm = 60.0 * frame_rate_hz / lag
        for expanded_bpm in _expanded_bpm_window(base_bpm, constants):
            key = round(expanded_bpm, 6)
            if key in seen:
                continue
            seen.add(key)
            whole_track.append(TempoCandidate(bpm=float(expanded_bpm), source="autocorrelation", score=float(score)))
            if len(whole_track) >= constants.max_whole_track_tempo_candidates:
                break
        if len(whole_track) >= constants.max_whole_track_tempo_candidates:
            break

    peak_interval = _peak_interval_tempo_candidates(beat_peaks, constants)
    candidates = list(whole_track)
    for candidate in peak_interval:
        key = round(candidate.bpm, 6)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)
        if len(candidates) >= constants.max_tempo_candidates_retained:
            break
    return tuple(candidates[: constants.max_tempo_candidates_retained])


def _top_autocorrelation_lags(
    signal: NDArray[np.float64],
    *,
    min_period_frames: int,
    max_period_frames: int,
    limit: int,
) -> tuple[tuple[int, float], ...]:
    centered = signal - float(np.mean(signal))
    lag_max = min(max_period_frames, signal.shape[0] - 1)
    if lag_max < min_period_frames or not np.any(centered):
        return ()
    scored: list[tuple[int, float]] = []
    for lag in range(max(1, min_period_frames), lag_max + 1):
        left = centered[:-lag]
        right = centered[lag:]
        left_norm = float(np.linalg.norm(left))
        right_norm = float(np.linalg.norm(right))
        if left_norm == 0.0 or right_norm == 0.0:
            continue
        score = float(np.dot(left, right) / (left_norm * right_norm))
        if math.isfinite(score):
            scored.append((lag, score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return tuple(scored[:limit])


def _expanded_bpm_window(
    candidate_bpm: float,
    constants: GlobalConstantJumpConstants,
) -> tuple[float, ...]:
    bpms: list[float] = []
    for multiplier in constants.alias_multipliers:
        bpm = candidate_bpm * multiplier
        if not _bpm_is_valid(bpm, constants):
            continue
        window = max(2.0, 0.08 * bpm)
        grid_start = math.ceil((bpm - window) / constants.bpm_grid_step) * constants.bpm_grid_step
        grid_end = math.floor((bpm + window) / constants.bpm_grid_step) * constants.bpm_grid_step
        grid_value = grid_start
        while grid_value <= grid_end + 1e-9:
            if _bpm_is_valid(grid_value, constants):
                bpms.append(float(round(grid_value, 6)))
            grid_value += constants.bpm_grid_step
        integer_start = math.floor(bpm - window)
        integer_end = math.ceil(bpm + window)
        for integer_part in range(integer_start, integer_end + 1):
            if integer_part < 80:
                continue
            for fractional_part in constants.fractional_bpm_parts:
                fractional_bpm = integer_part + fractional_part
                if bpm - window <= fractional_bpm <= bpm + window and _bpm_is_valid(fractional_bpm, constants):
                    bpms.append(float(round(fractional_bpm, 6)))
    return tuple(_dedupe_floats(bpms))


def _peak_interval_tempo_candidates(
    beat_peaks: tuple[MaterializedPeak, ...],
    constants: GlobalConstantJumpConstants,
) -> tuple[TempoCandidate, ...]:
    if len(beat_peaks) < 5:
        return ()
    times = np.asarray([peak.time_ms for peak in beat_peaks], dtype=np.float64)
    confidences = np.asarray([peak.confidence for peak in beat_peaks], dtype=np.float64)
    intervals = np.diff(times)
    median_intervals_by_length: dict[int, NDArray[np.float64]] = {}
    median_confidences_by_length: dict[int, NDArray[np.float64]] = {}
    for run_length in range(4, 17):
        if run_length > intervals.shape[0]:
            continue
        median_intervals_by_length[run_length] = np.median(
            np.lib.stride_tricks.sliding_window_view(intervals, run_length),
            axis=1,
        )
        median_confidences_by_length[run_length] = np.median(
            np.lib.stride_tricks.sliding_window_view(confidences, run_length + 1),
            axis=1,
        )
    best_by_bpm: dict[float, tuple[float, int, float]] = {}
    first_seen = 0
    for start in range(intervals.shape[0]):
        for run_length in range(4, 17):
            end = start + run_length
            if end > intervals.shape[0]:
                break
            median_interval_ms = float(median_intervals_by_length[run_length][start])
            if not math.isfinite(median_interval_ms) or median_interval_ms <= 0.0:
                continue
            median_confidence = float(median_confidences_by_length[run_length][start])
            base_bpm = 60000.0 / median_interval_ms
            for multiplier in constants.alias_multipliers:
                bpm = base_bpm * multiplier
                if not _bpm_is_valid(bpm, constants):
                    continue
                key = round(bpm, 6)
                previous = best_by_bpm.get(key)
                if previous is None or (median_confidence, -first_seen) > (previous[0], -previous[1]):
                    best_by_bpm[key] = (median_confidence, first_seen, float(key))
                first_seen += 1
    ranked = sorted(best_by_bpm.values(), key=lambda item: (-item[0], item[1], item[2]))
    return tuple(
        TempoCandidate(bpm=bpm, source="peak_interval", score=score)
        for score, _, bpm in ranked[: constants.max_peak_interval_tempo_candidates]
    )


def _origin_candidates(
    beat_signal: NDArray[np.float64],
    tempo_candidates: tuple[TempoCandidate, ...],
    *,
    frame_rate_hz: float,
    coverage_start_ms: float,
    coverage_end_ms: float,
    constants: GlobalConstantJumpConstants,
) -> tuple[OriginCandidate, ...]:
    if not tempo_candidates:
        return ()
    frame_times_ms = np.arange(beat_signal.shape[0], dtype=np.float64) * (1000.0 / frame_rate_hz)
    frame_start, frame_end = _frame_interval(frame_times_ms, coverage_start_ms, coverage_end_ms)
    signal_slice = beat_signal[frame_start:frame_end]
    time_slice_ms = frame_times_ms[frame_start:frame_end]
    centered_signal = signal_slice - float(np.mean(signal_slice)) if signal_slice.size else signal_slice
    signal_norm = float(np.linalg.norm(centered_signal)) if signal_slice.size else 0.0
    candidates: list[OriginCandidate] = []
    for tempo_candidate in tempo_candidates[: constants.origin_tempo_scan_count]:
        bpm = tempo_candidate.bpm
        period_ms = 60000.0 / bpm
        best_offset_ms: float | None = None
        best_score: float | None = None
        coarse_scores: list[tuple[float, float]] = []
        offset = 0.0
        while offset < period_ms:
            score = _origin_search_correlation(
                centered_signal,
                signal_norm,
                time_slice_ms,
                tau_ms=offset,
                bpm=bpm,
                pulse_width_ms=constants.pulse_width_ms,
            )
            score_value = score.correlation
            coarse_scores.append((offset, score_value))
            if best_score is None or score_value > best_score or (
                score_value == best_score and offset < best_offset_ms
            ):
                best_score = score_value
                best_offset_ms = offset
            offset += constants.origin_offset_grid_ms
        if best_offset_ms is None or best_score is None:
            continue
        best_offset_ms, best_score = _exact_best_origin_offset_near_search_best(
            centered_signal,
            signal_norm,
            time_slice_ms,
            coarse_scores,
            search_best_score=best_score,
            bpm=bpm,
            pulse_width_ms=constants.pulse_width_ms,
        )
        refined_best_offset_ms = best_offset_ms
        refined_best_score = best_score
        refine_start = int(math.floor(best_offset_ms - constants.origin_refine_radius_ms))
        refine_end = int(math.ceil(best_offset_ms + constants.origin_refine_radius_ms))
        refine_scores: list[tuple[float, float]] = []
        for refined_offset_ms in range(refine_start, refine_end + 1):
            wrapped_offset_ms = float(refined_offset_ms % period_ms)
            score = _origin_search_correlation(
                centered_signal,
                signal_norm,
                time_slice_ms,
                tau_ms=wrapped_offset_ms,
                bpm=bpm,
                pulse_width_ms=constants.pulse_width_ms,
            )
            score_value = score.correlation
            refine_scores.append((wrapped_offset_ms, score_value))
            if score_value > refined_best_score or (
                score_value == refined_best_score and wrapped_offset_ms < refined_best_offset_ms
            ):
                refined_best_score = score_value
                refined_best_offset_ms = wrapped_offset_ms
        refined_best_offset_ms, refined_best_score = _exact_best_origin_offset_near_search_best(
            centered_signal,
            signal_norm,
            time_slice_ms,
            refine_scores,
            search_best_score=refined_best_score,
            bpm=bpm,
            pulse_width_ms=constants.pulse_width_ms,
        )
        tau_ms = refined_best_offset_ms + math.ceil(
            (coverage_start_ms - refined_best_offset_ms) / period_ms
        ) * period_ms
        if tau_ms >= coverage_end_ms:
            continue
        candidates.append(
            OriginCandidate(
                anchor_id=-1,
                time_ms=float(tau_ms),
                bpm=float(bpm),
                score=float(refined_best_score),
            )
        )

    merged: list[OriginCandidate] = []
    for candidate in sorted(candidates, key=lambda item: (-item.score, item.time_ms, item.bpm)):
        conflict_index = next(
            (index for index, existing in enumerate(merged) if abs(existing.time_ms - candidate.time_ms) <= 20.0),
            None,
        )
        if conflict_index is None:
            merged.append(candidate)
        else:
            existing = merged[conflict_index]
            if (-candidate.score, candidate.time_ms, candidate.bpm) < (-existing.score, existing.time_ms, existing.bpm):
                merged[conflict_index] = candidate

    merged.sort(key=lambda item: (-item.score, item.time_ms, item.bpm))
    retained = merged[: constants.max_origin_candidates]
    return tuple(
        OriginCandidate(anchor_id=index, time_ms=candidate.time_ms, bpm=candidate.bpm, score=candidate.score)
        for index, candidate in enumerate(retained)
    )


@dataclass(frozen=True)
class _RawBoundaryCandidate:
    time_ms: float
    source_peak_index: int
    source_peak_time_ms: float
    source_peak_confidence: float
    rank_score: float
    evidence_mode: Literal["ordinary", "super"]
    left_period_ms: float
    right_period_ms: float
    ordinary_score: float | None
    super_score: float | None
    downbeat_bonus: float
    nearest_downbeat_distance_ms: float | None


def _boundary_candidates(
    beat_peaks: tuple[MaterializedPeak, ...],
    downbeat_peaks: tuple[MaterializedPeak, ...],
    *,
    coverage_start_ms: float,
    coverage_end_ms: float,
    constants: GlobalConstantJumpConstants,
) -> tuple[BoundaryCandidate, ...]:
    peak_times = np.asarray([peak.time_ms for peak in beat_peaks], dtype=np.float64)
    peak_confidences = np.asarray([peak.confidence for peak in beat_peaks], dtype=np.float64)
    downbeat_times = np.asarray([peak.time_ms for peak in downbeat_peaks], dtype=np.float64)
    raw_candidate_list: list[_RawBoundaryCandidate] = []
    for peak_index in range(len(beat_peaks)):
        candidate = _raw_boundary_candidate_from_arrays(
            beat_peaks,
            peak_times,
            peak_confidences,
            downbeat_times,
            peak_index,
            constants,
        )
        if candidate is not None:
            raw_candidate_list.append(candidate)
    raw_candidates = tuple(raw_candidate_list)
    merged = _merge_boundary_candidates(raw_candidates, constants)
    cap = _boundary_candidate_cap_for_coverage(
        coverage_start_ms=coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
        constants=constants,
    )
    retained = _cap_boundary_candidates_by_bins(
        merged,
        cap=cap,
        coverage_start_ms=coverage_start_ms,
        constants=constants,
    )
    return tuple(
        _boundary_candidate_from_raw(candidate, anchor_id=index)
        for index, candidate in enumerate(retained)
    )


def _raw_boundary_candidate(
    beat_peaks: tuple[MaterializedPeak, ...],
    downbeat_peaks: tuple[MaterializedPeak, ...],
    peak_index: int,
    constants: GlobalConstantJumpConstants,
) -> _RawBoundaryCandidate | None:
    peak_times = np.asarray([peak.time_ms for peak in beat_peaks], dtype=np.float64)
    peak_confidences = np.asarray([peak.confidence for peak in beat_peaks], dtype=np.float64)
    downbeat_times = np.asarray([peak.time_ms for peak in downbeat_peaks], dtype=np.float64)
    return _raw_boundary_candidate_from_arrays(
        beat_peaks,
        peak_times,
        peak_confidences,
        downbeat_times,
        peak_index,
        constants,
    )


def _raw_boundary_candidate_from_arrays(
    beat_peaks: tuple[MaterializedPeak, ...],
    peak_times: NDArray[np.float64],
    peak_confidences: NDArray[np.float64],
    downbeat_times: NDArray[np.float64],
    peak_index: int,
    constants: GlobalConstantJumpConstants,
) -> _RawBoundaryCandidate | None:
    ordinary = _ordinary_boundary_evidence(peak_times, peak_index)
    super_timing = _super_boundary_evidence(peak_times, peak_confidences, peak_index)
    if ordinary is None and super_timing is None:
        return None

    if ordinary is not None and (
        super_timing is None
        or ordinary[0] > super_timing[0]
        or ordinary[0] == super_timing[0]
    ):
        evidence_mode: Literal["ordinary", "super"] = "ordinary"
        winner_score, left_period_ms, right_period_ms = ordinary
    else:
        evidence_mode = "super"
        assert super_timing is not None
        winner_score, left_period_ms, right_period_ms = super_timing

    boundary_time_ms = float(peak_times[peak_index])
    downbeat_bonus = 0.0
    nearest_downbeat_distance_ms: float | None = None
    if downbeat_times.size:
        tolerance_ms = min(0.5 * min(left_period_ms, right_period_ms), 1000.0)
        nearest_index = _nearest_time_index(downbeat_times, boundary_time_ms)
        distance_ms = abs(float(downbeat_times[nearest_index]) - boundary_time_ms)
        if distance_ms <= tolerance_ms:
            boundary_time_ms = float(downbeat_times[nearest_index])
            nearest_downbeat_distance_ms = float(distance_ms)
            downbeat_bonus = 0.5 * (1.0 - distance_ms / tolerance_ms)

    return _RawBoundaryCandidate(
        time_ms=boundary_time_ms,
        source_peak_index=int(peak_index),
        source_peak_time_ms=float(peak_times[peak_index]),
        source_peak_confidence=float(beat_peaks[peak_index].confidence),
        rank_score=float(winner_score + downbeat_bonus),
        evidence_mode=evidence_mode,
        left_period_ms=float(left_period_ms),
        right_period_ms=float(right_period_ms),
        ordinary_score=None if ordinary is None else float(ordinary[0]),
        super_score=None if super_timing is None else float(super_timing[0]),
        downbeat_bonus=float(downbeat_bonus),
        nearest_downbeat_distance_ms=nearest_downbeat_distance_ms,
    )


def _ordinary_boundary_evidence(
    peak_times: NDArray[np.float64],
    peak_index: int,
) -> tuple[float, float, float] | None:
    if peak_index < 4 or peak_index > peak_times.shape[0] - 5:
        return None
    left_intervals = np.diff(peak_times[peak_index - 4 : peak_index + 1])
    right_intervals = np.diff(peak_times[peak_index : peak_index + 5])
    left_period = float(np.median(left_intervals))
    right_period = float(np.median(right_intervals))
    left_fit_error = float(
        np.median(
            np.abs(
                peak_times[peak_index - 4 : peak_index]
                - (peak_times[peak_index] - np.asarray([4, 3, 2, 1], dtype=np.float64) * left_period)
            )
        )
    )
    right_fit_error = float(
        np.median(
            np.abs(
                peak_times[peak_index + 1 : peak_index + 5]
                - (peak_times[peak_index] + np.asarray([1, 2, 3, 4], dtype=np.float64) * right_period)
            )
        )
    )
    interval_change = abs(right_period - left_period)
    phase_change = abs(right_fit_error - left_fit_error)
    phase_residual = max(left_fit_error, right_fit_error)
    if not (
        interval_change >= 20.0
        or phase_change >= 10.0
        or phase_residual >= 20.0
    ):
        return None
    score = (
        interval_change / 20.0
        + phase_change / 10.0
        + phase_residual / max(min(left_period, right_period), 1e-6)
    )
    return float(score), float(left_period), float(right_period)


def _super_boundary_evidence(
    peak_times: NDArray[np.float64],
    peak_confidences: NDArray[np.float64],
    peak_index: int,
) -> tuple[float, float, float] | None:
    if peak_index < 3 or peak_index > peak_times.shape[0] - 4:
        return None
    left_period = float(np.median(np.diff(peak_times[peak_index - 3 : peak_index + 1])))
    right_period = float(np.median(np.diff(peak_times[peak_index : peak_index + 4])))
    left_fit_error = float(
        np.median(
            np.abs(
                peak_times[peak_index - 3 : peak_index]
                - (peak_times[peak_index] - np.asarray([3, 2, 1], dtype=np.float64) * left_period)
            )
        )
    )
    right_fit_error = float(
        np.median(
            np.abs(
                peak_times[peak_index + 1 : peak_index + 4]
                - (peak_times[peak_index] + np.asarray([1, 2, 3], dtype=np.float64) * right_period)
            )
        )
    )
    relative_change = abs(right_period - left_period) / max(min(left_period, right_period), 1e-6)
    if relative_change < 0.025:
        return None
    phase_residual = max(left_fit_error, right_fit_error)
    score = (
        relative_change / 0.025
        + abs(right_period - left_period) / 20.0
        + phase_residual / 10.0
        + 0.25 * float(peak_confidences[peak_index])
    )
    return float(score), float(left_period), float(right_period)


def _merge_boundary_candidates(
    candidates: tuple[_RawBoundaryCandidate, ...],
    constants: GlobalConstantJumpConstants,
) -> tuple[_RawBoundaryCandidate, ...]:
    selected: list[_RawBoundaryCandidate] = []
    for candidate in sorted(candidates, key=_boundary_rank_key):
        conflict_index = next(
            (
                index
                for index, existing in enumerate(selected)
                if abs(existing.time_ms - candidate.time_ms) <= constants.boundary_merge_ms
            ),
            None,
        )
        if conflict_index is None:
            selected.append(candidate)
        elif _boundary_rank_key(candidate) < _boundary_rank_key(selected[conflict_index]):
            selected[conflict_index] = candidate
    return tuple(sorted(selected, key=lambda candidate: candidate.time_ms))


def _cap_boundary_candidates_by_bins(
    candidates: tuple[_RawBoundaryCandidate, ...],
    *,
    cap: int,
    coverage_start_ms: float,
    constants: GlobalConstantJumpConstants,
) -> tuple[_RawBoundaryCandidate, ...]:
    if len(candidates) <= cap:
        return candidates
    best_by_bin: dict[int, _RawBoundaryCandidate] = {}
    for candidate in candidates:
        bin_index = int(math.floor((candidate.time_ms - coverage_start_ms) / constants.boundary_bin_ms))
        previous = best_by_bin.get(bin_index)
        if previous is None or _boundary_rank_key(candidate) < _boundary_rank_key(previous):
            best_by_bin[bin_index] = candidate
    selected = set(sorted(best_by_bin.values(), key=_boundary_rank_key)[:cap])
    ranked = sorted(candidates, key=_boundary_rank_key)
    for candidate in ranked:
        if len(selected) >= cap:
            break
        selected.add(candidate)
    return tuple(sorted(selected, key=lambda candidate: candidate.time_ms))


def _boundary_candidate_from_raw(candidate: _RawBoundaryCandidate, *, anchor_id: int) -> BoundaryCandidate:
    return BoundaryCandidate(
        anchor_id=anchor_id,
        time_ms=candidate.time_ms,
        source_peak_index=candidate.source_peak_index,
        source_peak_time_ms=candidate.source_peak_time_ms,
        source_peak_confidence=candidate.source_peak_confidence,
        rank_score=candidate.rank_score,
        evidence_mode=candidate.evidence_mode,
        left_period_ms=candidate.left_period_ms,
        right_period_ms=candidate.right_period_ms,
        ordinary_score=candidate.ordinary_score,
        super_score=candidate.super_score,
        downbeat_bonus=candidate.downbeat_bonus,
        nearest_downbeat_distance_ms=candidate.nearest_downbeat_distance_ms,
    )


def _assemble_constant_only(
    candidates: GlobalConstantJumpCandidateSet,
    context: _SearchContext,
    *,
    variant: str,
) -> _State | None:
    best_complete: _State | None = None
    for origin_anchor in _origin_anchors(candidates):
        for downbeat_phase in _origin_downbeat_phases(context, variant):
            state = _initial_state(origin_anchor, downbeat_phase=downbeat_phase)
            completed = _best_terminal_completion(
                state,
                candidates,
                context,
                variant=variant,
                constant_only=True,
            )
            if completed is not None:
                best_complete = _min_complete_state(best_complete, completed)
    return best_complete


def _assemble_beam(
    candidates: GlobalConstantJumpCandidateSet,
    context: _SearchContext,
    *,
    variant: str,
) -> _State | None:
    origin_anchors = _origin_anchors(candidates)
    boundary_anchors = _boundary_anchors(candidates)
    states_by_bucket: dict[tuple[int, int], list[tuple[_StateOrderKeyParts, _State]]] = {}
    for origin_anchor in origin_anchors:
        for downbeat_phase in _origin_downbeat_phases(context, variant):
            state = _initial_state(origin_anchor, downbeat_phase=downbeat_phase)
            _insert_state_into_beam_bucket(states_by_bucket, (origin_anchor.anchor_id, 0), state, context)

    best_complete: _State | None = None
    anchor_order = tuple(sorted(origin_anchors + boundary_anchors, key=lambda anchor: anchor.anchor_id))
    for anchor in anchor_order:
        for section_count in range(context.constants.max_section_count):
            current_states = tuple(
                state for _, state in states_by_bucket.pop((anchor.anchor_id, section_count), ())
            )
            cj1_rejected_objectives: dict[int, float] = {}
            for state in current_states:
                for completed in _terminal_completions(state, candidates, context, variant=variant):
                    best_complete = _min_complete_state(best_complete, completed)
                if state.section_count + 1 >= context.constants.max_section_count:
                    continue
                _advance_state_edges_into_beam_bucket(
                    states_by_bucket,
                    state,
                    boundary_anchors,
                    context=context,
                    variant=variant,
                    cj1_rejected_objectives=cj1_rejected_objectives,
                )
    return best_complete


def _min_complete_state(current: _State | None, candidate: _State) -> _State:
    if current is None or _complete_state_order_key(candidate) < _complete_state_order_key(current):
        return candidate
    return current


def _insert_state_into_beam_bucket(
    states_by_bucket: dict[tuple[int, int], list[tuple[_StateOrderKeyParts, _State]]],
    bucket_key: tuple[int, int],
    candidate: _State,
    context: _SearchContext,
) -> None:
    _insert_materialized_state_into_beam_bucket(
        states_by_bucket,
        bucket_key,
        order_key=_state_order_key_parts(candidate),
        candidate=candidate,
        context=context,
    )


def _insert_pending_state_into_beam_bucket(
    states_by_bucket: dict[tuple[int, int], list[tuple[_StateOrderKeyParts, _State]]],
    bucket_key: tuple[int, int],
    pending_state: _PendingInteriorState,
    context: _SearchContext,
) -> None:
    bucket = states_by_bucket.setdefault(bucket_key, [])
    insert_index = _beam_bucket_bisect_right(bucket, pending_state.order_key)
    if len(bucket) >= context.constants.beam_width and insert_index >= context.constants.beam_width:
        context.beam_pruned_state_count += 1
        return
    candidate = _materialize_pending_interior_state(pending_state, context)
    bucket.insert(insert_index, (pending_state.order_key, candidate))
    if len(bucket) > context.constants.beam_width:
        del bucket[context.constants.beam_width :]
        context.beam_pruned_state_count += 1


def _insert_materialized_state_into_beam_bucket(
    states_by_bucket: dict[tuple[int, int], list[tuple[_StateOrderKeyParts, _State]]],
    bucket_key: tuple[int, int],
    *,
    order_key: _StateOrderKeyParts,
    candidate: _State,
    context: _SearchContext,
) -> None:
    bucket = states_by_bucket.setdefault(bucket_key, [])
    insert_index = _beam_bucket_bisect_right(bucket, order_key)
    if len(bucket) >= context.constants.beam_width and insert_index >= context.constants.beam_width:
        context.beam_pruned_state_count += 1
        return
    bucket.insert(insert_index, (order_key, candidate))
    if len(bucket) > context.constants.beam_width:
        del bucket[context.constants.beam_width :]
        context.beam_pruned_state_count += 1


def _beam_bucket_bisect_right(
    bucket: list[tuple[_StateOrderKeyParts, _State]],
    candidate_key: _StateOrderKeyParts,
) -> int:
    low = 0
    high = len(bucket)
    while low < high:
        mid = (low + high) // 2
        if candidate_key < bucket[mid][0]:
            high = mid
        else:
            low = mid + 1
    return low


def _beam_bucket_bisect_right_interior(
    bucket: list[tuple[_StateOrderKeyParts, _State]],
    *,
    parent: _State,
    objective: float,
    section_count: int,
    alias_switch_count: int,
    max_boundary_displacement_ms: float,
    edge_tuple: tuple[int, int, int, float],
    replay_item: tuple[int, int, tuple[int, int, int, float]],
) -> int:
    low = 0
    high = len(bucket)
    while low < high:
        mid = (low + high) // 2
        if _interior_state_key_less_than(
            parent=parent,
            objective=objective,
            section_count=section_count,
            alias_switch_count=alias_switch_count,
            max_boundary_displacement_ms=max_boundary_displacement_ms,
            edge_tuple=edge_tuple,
            replay_item=replay_item,
            right_key=bucket[mid][0],
        ):
            high = mid
        else:
            low = mid + 1
    return low


def _interior_state_key_less_than(
    *,
    parent: _State,
    objective: float,
    section_count: int,
    alias_switch_count: int,
    max_boundary_displacement_ms: float,
    edge_tuple: tuple[int, int, int, float],
    replay_item: tuple[int, int, tuple[int, int, int, float]],
    right_key: _StateOrderKeyParts,
) -> bool:
    if objective != right_key.objective:
        return objective < right_key.objective
    if section_count != right_key.section_count:
        return section_count < right_key.section_count
    if alias_switch_count != right_key.alias_switch_count:
        return alias_switch_count < right_key.alias_switch_count
    if max_boundary_displacement_ms != right_key.max_boundary_displacement_ms:
        return max_boundary_displacement_ms < right_key.max_boundary_displacement_ms
    if parent.origin_time_ms != right_key.origin_time_ms:
        return parent.origin_time_ms < right_key.origin_time_ms
    edge_comparison = _compare_tuple_with_optional_suffix(
        parent.edge_tuples,
        edge_tuple,
        right_key.edge_tuples,
        right_key.edge_suffix,
    )
    if edge_comparison != 0:
        return edge_comparison < 0
    return (
        _compare_tuple_with_optional_suffix(
            parent.replay_key,
            replay_item,
            right_key.replay_key,
            right_key.replay_suffix,
        )
        < 0
    )


def _terminal_completions(
    state: _State,
    candidates: GlobalConstantJumpCandidateSet,
    context: _SearchContext,
    *,
    variant: str,
) -> Iterable[_State]:
    completed = _best_terminal_completion(
        state,
        candidates,
        context,
        variant=variant,
        constant_only=False,
    )
    if completed is not None:
        yield completed


def _best_terminal_completion(
    state: _State,
    candidates: GlobalConstantJumpCandidateSet,
    context: _SearchContext,
    *,
    variant: str,
    constant_only: bool,
) -> _State | None:
    best_key: tuple[Any, ...] | None = None
    best_payload: tuple[float, int, int, float, float, int, tuple[int, int, int, float], int] | None = None
    for terminal_rank, bpm in enumerate(_terminal_bpms(state, candidates, context, constant_only=constant_only)):
        payload = _terminal_completion_payload(
            state,
            bpm=bpm,
            terminal_rank=terminal_rank,
            context=context,
            variant=variant,
        )
        if payload is None:
            continue
        (
            payload_bpm,
            terminal_end_beat,
            first_start_beat,
            duration_objective,
            transition_objective,
            alias_switch_count,
            edge_tuple,
            payload_terminal_rank,
        ) = payload
        objective = duration_objective + transition_objective
        candidate_key = (
            objective,
            alias_switch_count,
            edge_tuple,
            ("T", payload_terminal_rank, edge_tuple),
        )
        if best_key is None or candidate_key < best_key:
            best_key = candidate_key
            best_payload = payload
    if best_payload is None:
        return None
    (
        bpm,
        terminal_end_beat,
        first_start_beat,
        duration_objective,
        transition_objective,
        alias_switch_count,
        edge_tuple,
        terminal_rank,
    ) = best_payload
    section = _SectionSpec(
        start_beat=first_start_beat,
        end_beat=terminal_end_beat,
        bpm=float(bpm),
    )
    edge_tuples = state.edge_tuples + (edge_tuple,)
    replay_key = state.replay_key + (("T", terminal_rank, edge_tuple),)
    objective = duration_objective + transition_objective
    return _State(
        anchor=state.anchor,
        section_count=state.section_count + 1,
        beat_at_anchor=terminal_end_beat,
        prev_bpm=float(bpm),
        prev_alias_family=_alias_family_representative_cached(bpm, context),
        downbeat_phase=state.downbeat_phase,
        origin_time_ms=state.origin_time_ms,
        duration_objective=float(duration_objective),
        transition_objective=float(transition_objective),
        objective=float(objective),
        alias_switch_count=alias_switch_count,
        max_boundary_displacement_ms=state.max_boundary_displacement_ms,
        sections=state.sections + (section,),
        edge_tuples=edge_tuples,
        replay_key=replay_key,
    )


def _terminal_completion_payload(
    state: _State,
    *,
    bpm: float,
    terminal_rank: int,
    context: _SearchContext,
    variant: str,
) -> tuple[
    float,
    int,
    int,
    float,
    float,
    int,
    tuple[int, int, int, float],
    int,
] | None:
    if state.anchor.time_ms >= context.coverage_end_ms or not _bpm_is_valid(bpm, context.constants):
        return None
    period_ms = 60000.0 / bpm
    tail_beat_count = max(1, int(math.ceil((context.coverage_end_ms - state.anchor.time_ms) / period_ms)))
    terminal_end_beat = state.beat_at_anchor + tail_beat_count
    right_frame = context.frame_count
    score = _section_score(
        state.anchor,
        right_anchor=None,
        left_beat_at_anchor=state.beat_at_anchor,
        beat_count=tail_beat_count,
        bpm=bpm,
        downbeat_phase=state.downbeat_phase,
        context=context,
        variant=variant,
    )
    if not score.valid:
        return None
    transition = _transition_score(
        state,
        right_bpm=bpm,
        context=context,
        variant=variant,
    )
    interval_start_ms = 0.0 if state.section_count == 0 else state.anchor.time_ms
    interval_end_ms = context.coverage_end_ms
    duration_objective = state.duration_objective + (interval_end_ms - interval_start_ms) * score.cost / _duration_ms(context)
    transition_objective = state.transition_objective + transition.cost / _transition_normalizer(context)
    alias_switch_count = state.alias_switch_count + transition.alias_switch_increment
    first_start_beat = (
        _first_start_beat(origin_time_ms=state.origin_time_ms, bpm=bpm)
        if state.section_count == 0
        else state.beat_at_anchor
    )
    if terminal_end_beat <= first_start_beat:
        return None
    edge_tuple = (
        _frame_index_for_time(state.anchor.time_ms, context.frame_rate_hz),
        right_frame,
        int(tail_beat_count),
        round(float(bpm), 6),
    )
    return (
        float(bpm),
        terminal_end_beat,
        first_start_beat,
        float(duration_objective),
        float(transition_objective),
        alias_switch_count,
        edge_tuple,
        terminal_rank,
    )


def _complete_terminal_state(
    state: _State,
    *,
    bpm: float,
    terminal_rank: int,
    context: _SearchContext,
    variant: str,
) -> _State | None:
    if state.anchor.time_ms >= context.coverage_end_ms or not _bpm_is_valid(bpm, context.constants):
        return None
    period_ms = 60000.0 / bpm
    tail_beat_count = max(1, int(math.ceil((context.coverage_end_ms - state.anchor.time_ms) / period_ms)))
    terminal_end_beat = state.beat_at_anchor + tail_beat_count
    right_frame = context.frame_count
    score = _section_score(
        state.anchor,
        right_anchor=None,
        left_beat_at_anchor=state.beat_at_anchor,
        beat_count=tail_beat_count,
        bpm=bpm,
        downbeat_phase=state.downbeat_phase,
        context=context,
        variant=variant,
    )
    if not score.valid:
        return None
    transition = _transition_score(
        state,
        right_bpm=bpm,
        context=context,
        variant=variant,
    )
    interval_start_ms = 0.0 if state.section_count == 0 else state.anchor.time_ms
    interval_end_ms = context.coverage_end_ms
    duration_objective = state.duration_objective + (interval_end_ms - interval_start_ms) * score.cost / _duration_ms(context)
    transition_objective = state.transition_objective + transition.cost / _transition_normalizer(context)
    alias_switch_count = state.alias_switch_count + transition.alias_switch_increment
    first_start_beat = (
        _first_start_beat(origin_time_ms=state.origin_time_ms, bpm=bpm)
        if state.section_count == 0
        else state.beat_at_anchor
    )
    section = _SectionSpec(
        start_beat=first_start_beat,
        end_beat=terminal_end_beat,
        bpm=float(bpm),
    )
    if section.end_beat <= section.start_beat:
        return None
    edge_tuple = (
        _frame_index_for_time(state.anchor.time_ms, context.frame_rate_hz),
        right_frame,
        int(tail_beat_count),
        round(float(bpm), 6),
    )
    sections = state.sections + (section,)
    edge_tuples = state.edge_tuples + (edge_tuple,)
    replay_key = state.replay_key + (("T", terminal_rank, edge_tuple),)
    objective = duration_objective + transition_objective
    return _State(
        anchor=state.anchor,
        section_count=state.section_count + 1,
        beat_at_anchor=terminal_end_beat,
        prev_bpm=float(bpm),
        prev_alias_family=_alias_family_representative_cached(bpm, context),
        downbeat_phase=state.downbeat_phase,
        origin_time_ms=state.origin_time_ms,
        duration_objective=float(duration_objective),
        transition_objective=float(transition_objective),
        objective=float(objective),
        alias_switch_count=alias_switch_count,
        max_boundary_displacement_ms=state.max_boundary_displacement_ms,
        sections=sections,
        edge_tuples=edge_tuples,
        replay_key=replay_key,
    )


def _advance_state_edges_into_beam_bucket(
    states_by_bucket: dict[tuple[int, int], list[tuple[_StateOrderKeyParts, _State]]],
    state: _State,
    boundary_anchors: tuple[_Anchor, ...],
    *,
    context: _SearchContext,
    variant: str,
    cj1_rejected_objectives: dict[int, float] | None = None,
) -> None:
    if not context.interior_edge_score_bundle_enabled:
        for right_anchor in _outgoing_boundary_anchors(state, boundary_anchors, context):
            for count_candidate in _beat_count_candidates(state.anchor, right_anchor, context):
                _advance_state_into_beam_bucket(
                    states_by_bucket,
                    state,
                    right_anchor,
                    count_candidate,
                    context=context,
                    variant=variant,
                )
        return

    bundle_key = _interior_edge_score_bundle_key(state, variant)
    cached_bundle = context.interior_edge_score_bundle_cache.get(bundle_key)
    if cached_bundle is not None:
        for entry_index, (right_anchor, count_candidate, score) in enumerate(cached_bundle):
            if (
                variant == VARIANT_CJ1
                and context.cj1_monotone_rejection_enabled
                and cj1_rejected_objectives is not None
                and entry_index in cj1_rejected_objectives
            ):
                candidate_objective = _cj1_candidate_objective(
                    state,
                    right_anchor,
                    score=score,
                    context=context,
                )
                if candidate_objective > cj1_rejected_objectives[entry_index]:
                    context.beam_pruned_state_count += 1
                    continue
            rejected_objective = _advance_state_into_beam_bucket_with_score(
                states_by_bucket,
                state,
                right_anchor,
                count_candidate,
                score=score,
                context=context,
                variant=variant,
            )
            if (
                variant == VARIANT_CJ1
                and context.cj1_monotone_rejection_enabled
                and cj1_rejected_objectives is not None
                and rejected_objective is not None
            ):
                previous = cj1_rejected_objectives.get(entry_index)
                cj1_rejected_objectives[entry_index] = (
                    rejected_objective if previous is None else min(previous, rejected_objective)
                )
        return

    bundle_entries: list[tuple[_Anchor, _BeatCountCandidate, _SectionScore]] = []
    for right_anchor in _outgoing_boundary_anchors(state, boundary_anchors, context):
        for count_candidate in _beat_count_candidates(state.anchor, right_anchor, context):
            score = _section_score(
                state.anchor,
                right_anchor=right_anchor,
                left_beat_at_anchor=state.beat_at_anchor,
                beat_count=count_candidate.count,
                bpm=count_candidate.bpm,
                downbeat_phase=state.downbeat_phase,
                context=context,
                variant=variant,
            )
            bundle_entries.append((right_anchor, count_candidate, score))
            _advance_state_into_beam_bucket_with_score(
                states_by_bucket,
                state,
                right_anchor,
                count_candidate,
                score=score,
                context=context,
                variant=variant,
            )
    context.interior_edge_score_bundle_cache[bundle_key] = tuple(bundle_entries)


def _interior_edge_score_bundle_key(
    state: _State,
    variant: str,
) -> tuple[str, int, float, str, int | None]:
    return (
        state.anchor.kind,
        state.anchor.anchor_id,
        state.anchor.time_ms,
        variant,
        _effective_downbeat_residue(
            beat_at_tau=state.beat_at_anchor,
            global_downbeat_phase=state.downbeat_phase,
            variant=variant,
        ),
    )


def _advance_state_into_beam_bucket(
    states_by_bucket: dict[tuple[int, int], list[tuple[_StateOrderKeyParts, _State]]],
    state: _State,
    right_anchor: _Anchor,
    count_candidate: _BeatCountCandidate,
    *,
    context: _SearchContext,
    variant: str,
) -> None:
    score = _section_score(
        state.anchor,
        right_anchor=right_anchor,
        left_beat_at_anchor=state.beat_at_anchor,
        beat_count=count_candidate.count,
        bpm=count_candidate.bpm,
        downbeat_phase=state.downbeat_phase,
        context=context,
        variant=variant,
    )
    if not score.valid:
        return
    _advance_state_into_beam_bucket_with_score(
        states_by_bucket,
        state,
        right_anchor,
        count_candidate,
        score=score,
        context=context,
        variant=variant,
    )


def _advance_state_into_beam_bucket_with_score(
    states_by_bucket: dict[tuple[int, int], list[tuple[_StateOrderKeyParts, _State]]],
    state: _State,
    right_anchor: _Anchor,
    count_candidate: _BeatCountCandidate,
    *,
    score: _SectionScore,
    context: _SearchContext,
    variant: str,
) -> float | None:
    if not score.valid:
        return None
    transition = _transition_score(
        state,
        right_bpm=count_candidate.bpm,
        context=context,
        variant=variant,
    )
    interval_start_ms = 0.0 if state.section_count == 0 else state.anchor.time_ms
    interval_end_ms = right_anchor.time_ms
    duration_objective = state.duration_objective + (interval_end_ms - interval_start_ms) * score.cost / _duration_ms(context)
    transition_objective = state.transition_objective + transition.cost / _transition_normalizer(context)
    alias_switch_count = state.alias_switch_count + transition.alias_switch_increment
    next_beat = state.beat_at_anchor + count_candidate.count
    first_start_beat = (
        _first_start_beat(origin_time_ms=state.origin_time_ms, bpm=count_candidate.bpm)
        if state.section_count == 0
        else state.beat_at_anchor
    )
    if next_beat <= first_start_beat:
        return None
    edge_tuple = (
        _frame_index_for_time(state.anchor.time_ms, context.frame_rate_hz),
        _frame_index_for_time(right_anchor.time_ms, context.frame_rate_hz),
        int(count_candidate.count),
        round(float(count_candidate.bpm), 6),
    )
    max_boundary_displacement_ms = state.max_boundary_displacement_ms
    if right_anchor.boundary is not None:
        max_boundary_displacement_ms = max(
            max_boundary_displacement_ms,
            right_anchor.boundary.displacement_from_source_peak_ms,
        )
    section_count = state.section_count + 1
    objective = duration_objective + transition_objective
    replay_item = (right_anchor.anchor_id, count_candidate.rank, edge_tuple)
    bucket_key = (right_anchor.anchor_id, section_count)
    bucket = states_by_bucket.setdefault(bucket_key, [])
    insert_index = _beam_bucket_bisect_right_interior(
        bucket,
        parent=state,
        objective=float(objective),
        section_count=section_count,
        alias_switch_count=alias_switch_count,
        max_boundary_displacement_ms=float(max_boundary_displacement_ms),
        edge_tuple=edge_tuple,
        replay_item=replay_item,
    )
    if len(bucket) >= context.constants.beam_width and insert_index >= context.constants.beam_width:
        context.beam_pruned_state_count += 1
        return float(objective)
    order_key = _pending_state_order_key(
        state,
        section_count=section_count,
        objective=float(objective),
        alias_switch_count=alias_switch_count,
        max_boundary_displacement_ms=float(max_boundary_displacement_ms),
        edge_tuple=edge_tuple,
        replay_item=replay_item,
    )
    candidate = _materialize_interior_state(
        parent=state,
        anchor=right_anchor,
        section_count=section_count,
        beat_at_anchor=next_beat,
        bpm=float(count_candidate.bpm),
        first_start_beat=first_start_beat,
        duration_objective=float(duration_objective),
        transition_objective=float(transition_objective),
        objective=float(objective),
        alias_switch_count=alias_switch_count,
        max_boundary_displacement_ms=float(max_boundary_displacement_ms),
        edge_tuple=edge_tuple,
        replay_item=replay_item,
        context=context,
    )
    bucket.insert(insert_index, (order_key, candidate))
    if len(bucket) > context.constants.beam_width:
        del bucket[context.constants.beam_width :]
        context.beam_pruned_state_count += 1
    return None


def _cj1_candidate_objective(
    state: _State,
    right_anchor: _Anchor,
    *,
    score: _SectionScore,
    context: _SearchContext,
) -> float:
    interval_start_ms = 0.0 if state.section_count == 0 else state.anchor.time_ms
    duration_objective = state.duration_objective + (
        (right_anchor.time_ms - interval_start_ms) * score.cost / _duration_ms(context)
    )
    transition_objective = state.transition_objective + 0.0 / _transition_normalizer(context)
    return float(duration_objective + transition_objective)


def _advance_state(
    state: _State,
    right_anchor: _Anchor,
    count_candidate: _BeatCountCandidate,
    *,
    context: _SearchContext,
    variant: str,
) -> _State | None:
    pending_state = _advance_state_payload(
        state,
        right_anchor,
        count_candidate,
        context=context,
        variant=variant,
    )
    if pending_state is None:
        return None
    return _materialize_pending_interior_state(pending_state, context)


def _advance_state_payload(
    state: _State,
    right_anchor: _Anchor,
    count_candidate: _BeatCountCandidate,
    *,
    context: _SearchContext,
    variant: str,
) -> _PendingInteriorState | None:
    score = _section_score(
        state.anchor,
        right_anchor=right_anchor,
        left_beat_at_anchor=state.beat_at_anchor,
        beat_count=count_candidate.count,
        bpm=count_candidate.bpm,
        downbeat_phase=state.downbeat_phase,
        context=context,
        variant=variant,
    )
    if not score.valid:
        return None
    transition = _transition_score(
        state,
        right_bpm=count_candidate.bpm,
        context=context,
        variant=variant,
    )
    interval_start_ms = 0.0 if state.section_count == 0 else state.anchor.time_ms
    interval_end_ms = right_anchor.time_ms
    duration_objective = state.duration_objective + (interval_end_ms - interval_start_ms) * score.cost / _duration_ms(context)
    transition_objective = state.transition_objective + transition.cost / _transition_normalizer(context)
    alias_switch_count = state.alias_switch_count + transition.alias_switch_increment
    next_beat = state.beat_at_anchor + count_candidate.count
    first_start_beat = (
        _first_start_beat(origin_time_ms=state.origin_time_ms, bpm=count_candidate.bpm)
        if state.section_count == 0
        else state.beat_at_anchor
    )
    if next_beat <= first_start_beat:
        return None
    edge_tuple = (
        _frame_index_for_time(state.anchor.time_ms, context.frame_rate_hz),
        _frame_index_for_time(right_anchor.time_ms, context.frame_rate_hz),
        int(count_candidate.count),
        round(float(count_candidate.bpm), 6),
    )
    max_boundary_displacement_ms = state.max_boundary_displacement_ms
    if right_anchor.boundary is not None:
        max_boundary_displacement_ms = max(
            max_boundary_displacement_ms,
            right_anchor.boundary.displacement_from_source_peak_ms,
        )
    objective = duration_objective + transition_objective
    replay_item = (right_anchor.anchor_id, count_candidate.rank, edge_tuple)
    order_key = _pending_state_order_key(
        state,
        section_count=state.section_count + 1,
        objective=float(objective),
        alias_switch_count=alias_switch_count,
        max_boundary_displacement_ms=float(max_boundary_displacement_ms),
        edge_tuple=edge_tuple,
        replay_item=replay_item,
    )
    return _PendingInteriorState(
        parent=state,
        anchor=right_anchor,
        section_count=state.section_count + 1,
        beat_at_anchor=next_beat,
        bpm=float(count_candidate.bpm),
        first_start_beat=first_start_beat,
        duration_objective=float(duration_objective),
        transition_objective=float(transition_objective),
        objective=float(objective),
        alias_switch_count=alias_switch_count,
        max_boundary_displacement_ms=float(max_boundary_displacement_ms),
        edge_tuple=edge_tuple,
        replay_item=replay_item,
        order_key=order_key,
    )


def _materialize_pending_interior_state(
    pending_state: _PendingInteriorState,
    context: _SearchContext,
) -> _State:
    return _materialize_interior_state(
        parent=pending_state.parent,
        anchor=pending_state.anchor,
        section_count=pending_state.section_count,
        beat_at_anchor=pending_state.beat_at_anchor,
        bpm=float(pending_state.bpm),
        first_start_beat=pending_state.first_start_beat,
        duration_objective=float(pending_state.duration_objective),
        transition_objective=float(pending_state.transition_objective),
        objective=float(pending_state.objective),
        alias_switch_count=pending_state.alias_switch_count,
        max_boundary_displacement_ms=float(pending_state.max_boundary_displacement_ms),
        edge_tuple=pending_state.edge_tuple,
        replay_item=pending_state.replay_item,
        context=context,
    )


def _materialize_interior_state(
    *,
    parent: _State,
    anchor: _Anchor,
    section_count: int,
    beat_at_anchor: int,
    bpm: float,
    first_start_beat: int,
    duration_objective: float,
    transition_objective: float,
    objective: float,
    alias_switch_count: int,
    max_boundary_displacement_ms: float,
    edge_tuple: tuple[int, int, int, float],
    replay_item: tuple[int, int, tuple[int, int, int, float]],
    context: _SearchContext,
) -> _State:
    section = _SectionSpec(
        start_beat=first_start_beat,
        end_beat=beat_at_anchor,
        bpm=float(bpm),
    )
    sections = parent.sections + (section,)
    edge_tuples = parent.edge_tuples + (edge_tuple,)
    replay_key = parent.replay_key + (replay_item,)
    return _State(
        anchor=anchor,
        section_count=section_count,
        beat_at_anchor=beat_at_anchor,
        prev_bpm=float(bpm),
        prev_alias_family=_alias_family_representative_cached(bpm, context),
        downbeat_phase=parent.downbeat_phase,
        origin_time_ms=parent.origin_time_ms,
        duration_objective=float(duration_objective),
        transition_objective=float(transition_objective),
        objective=float(objective),
        alias_switch_count=alias_switch_count,
        max_boundary_displacement_ms=float(max_boundary_displacement_ms),
        sections=sections,
        edge_tuples=edge_tuples,
        replay_key=replay_key,
    )


def _section_score(
    left_anchor: _Anchor,
    *,
    right_anchor: _Anchor | None,
    left_beat_at_anchor: int,
    beat_count: int,
    bpm: float,
    downbeat_phase: int | None,
    context: _SearchContext,
    variant: str,
) -> _SectionScore:
    cache = context.terminal_score_cache if right_anchor is None else context.section_score_cache
    cache_key = _section_score_cache_key(
        left_anchor,
        right_anchor=right_anchor,
        left_beat_at_anchor=left_beat_at_anchor,
        beat_count=beat_count,
        bpm=bpm,
        downbeat_phase=downbeat_phase,
        variant=variant,
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    context.record_search_trace_event("score_cache_miss", cache_key)
    context.check_attempt(cache_key)
    interval_end_ms = context.coverage_end_ms if right_anchor is None else right_anchor.time_ms
    partitions = _score_partitions(left_anchor, interval_end_ms)
    total_duration_ms = math.fsum(end_ms - start_ms for start_ms, end_ms in partitions)
    if total_duration_ms <= 0.0:
        score = _invalid_section_score()
        cache[cache_key] = score
        return score

    weighted_cost = 0.0
    weighted_beat_support_cost = 0.0
    weighted_peak_recall_precision_cost = 0.0
    weighted_downbeat_phase_cost = 0.0
    weighted_bpm_prior_cost = 0.0
    weighted_beat_count_prior_cost = 0.0
    weighted_section_duration_cost = 0.0
    for start_ms, end_ms in partitions:
        interval_score = _section_interval_score(
            left_anchor_time_ms=left_anchor.time_ms,
            left_beat_at_anchor=left_beat_at_anchor,
            beat_count=beat_count,
            bpm=bpm,
            downbeat_phase=downbeat_phase,
            context=context,
            variant=variant,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        if not interval_score.valid:
            cache[cache_key] = interval_score
            return interval_score
        duration_ms = end_ms - start_ms
        weighted_cost += duration_ms * interval_score.cost
        weighted_beat_support_cost += duration_ms * interval_score.beat_support_cost
        weighted_peak_recall_precision_cost += duration_ms * interval_score.peak_recall_precision_cost
        weighted_downbeat_phase_cost += duration_ms * interval_score.downbeat_phase_cost
        weighted_bpm_prior_cost += duration_ms * interval_score.bpm_prior_cost
        weighted_beat_count_prior_cost += duration_ms * interval_score.beat_count_prior_cost
        weighted_section_duration_cost += duration_ms * interval_score.section_duration_cost

    score = _SectionScore(
        valid=True,
        cost=float(weighted_cost / total_duration_ms),
        beat_support_cost=float(weighted_beat_support_cost / total_duration_ms),
        peak_recall_precision_cost=float(weighted_peak_recall_precision_cost / total_duration_ms),
        downbeat_phase_cost=float(weighted_downbeat_phase_cost / total_duration_ms),
        bpm_prior_cost=float(weighted_bpm_prior_cost / total_duration_ms),
        beat_count_prior_cost=float(weighted_beat_count_prior_cost / total_duration_ms),
        section_duration_cost=float(weighted_section_duration_cost / total_duration_ms),
    )
    cache[cache_key] = score
    return score


def _section_score_cache_key(
    left_anchor: _Anchor,
    *,
    right_anchor: _Anchor | None,
    left_beat_at_anchor: int,
    beat_count: int,
    bpm: float,
    downbeat_phase: int | None,
    variant: str,
) -> tuple[Any, ...]:
    effective_residue = _effective_downbeat_residue(
        beat_at_tau=left_beat_at_anchor,
        global_downbeat_phase=downbeat_phase,
        variant=variant,
    )
    if right_anchor is None:
        key: tuple[Any, ...] = (
            left_anchor.anchor_id,
            -1,
            int(beat_count),
            variant,
            round(float(bpm), 6),
        )
    else:
        key = (
            left_anchor.anchor_id,
            right_anchor.anchor_id,
            int(beat_count),
            variant,
        )
    if _variant_uses_downbeat(variant):
        key = key + (effective_residue,)
    return key


def _score_partitions(
    left_anchor: _Anchor,
    interval_end_ms: float,
) -> tuple[tuple[float, float], ...]:
    if left_anchor.kind != "origin":
        return ((left_anchor.time_ms, interval_end_ms),)
    partitions: list[tuple[float, float]] = []
    prefix_end_ms = min(left_anchor.time_ms, interval_end_ms)
    if prefix_end_ms > 0.0:
        partitions.append((0.0, prefix_end_ms))
    body_start_ms = max(left_anchor.time_ms, 0.0)
    if interval_end_ms > body_start_ms:
        partitions.append((body_start_ms, interval_end_ms))
    return tuple(partitions)


def _section_interval_score(
    *,
    left_anchor_time_ms: float,
    left_beat_at_anchor: int,
    beat_count: int,
    bpm: float,
    downbeat_phase: int | None,
    context: _SearchContext,
    variant: str,
    start_ms: float,
    end_ms: float,
) -> _SectionScore:
    if end_ms <= start_ms:
        return _SectionScore(
            valid=True,
            cost=0.0,
            beat_support_cost=0.0,
            peak_recall_precision_cost=0.0,
            downbeat_phase_cost=0.0,
            bpm_prior_cost=0.0,
            beat_count_prior_cost=0.0,
            section_duration_cost=0.0,
        )
    beat_corr = _pulse_correlation_interval(
        context.beat_signal,
        context.frame_times_ms,
        tau_ms=left_anchor_time_ms,
        bpm=bpm,
        start_ms=start_ms,
        end_ms=end_ms,
        pulse_width_ms=context.constants.pulse_width_ms,
    )
    if not beat_corr.valid:
        return _invalid_section_score()
    beat_support_cost = 0.5 * (1.0 - beat_corr.correlation)
    peak_recall_precision_cost = _peak_recall_precision_cost_fast(
        left_anchor_time_ms=left_anchor_time_ms,
        bpm=bpm,
        start_ms=start_ms,
        end_ms=end_ms,
        beat_peak_times_ms=context.beat_peak_times_ms,
        constants=context.constants,
    )
    downbeat_phase_cost = 0.0
    if _variant_uses_downbeat(variant) and downbeat_phase is not None:
        downbeat_corr = _downbeat_pulse_correlation_interval(
            context.downbeat_signal,
            context.frame_times_ms,
            tau_ms=left_anchor_time_ms,
            beat_at_tau=left_beat_at_anchor,
            bpm=bpm,
            global_downbeat_phase=downbeat_phase,
            start_ms=start_ms,
            end_ms=end_ms,
            pulse_width_ms=context.constants.pulse_width_ms,
        )
        if not downbeat_corr.valid:
            return _invalid_section_score()
        downbeat_phase_cost = 0.5 * (1.0 - downbeat_corr.correlation)
    bpm_prior_cost = _bpm_prior_cost(bpm, context.constants) if _variant_uses_priors(variant) else 0.0
    beat_count_prior_cost = (
        _beat_count_prior_cost(beat_count) if _variant_uses_priors(variant) else 0.0
    )
    section_duration_cost = (
        _section_duration_cost(end_ms - start_ms) if _variant_uses_priors(variant) else 0.0
    )
    cost = (
        1.00 * beat_support_cost
        + 0.25 * peak_recall_precision_cost
        + (0.20 * downbeat_phase_cost if _variant_uses_downbeat(variant) else 0.0)
        + (0.10 * bpm_prior_cost if _variant_uses_priors(variant) else 0.0)
        + (0.10 * beat_count_prior_cost if _variant_uses_priors(variant) else 0.0)
        + (0.08 * section_duration_cost if _variant_uses_priors(variant) else 0.0)
    )
    return _SectionScore(
        valid=True,
        cost=float(cost),
        beat_support_cost=float(beat_support_cost),
        peak_recall_precision_cost=float(peak_recall_precision_cost),
        downbeat_phase_cost=float(downbeat_phase_cost),
        bpm_prior_cost=float(bpm_prior_cost),
        beat_count_prior_cost=float(beat_count_prior_cost),
        section_duration_cost=float(section_duration_cost),
    )


def _transition_score(
    state: _State,
    *,
    right_bpm: float,
    context: _SearchContext,
    variant: str,
) -> _TransitionScore:
    cache_key = (state.prev_bpm, float(right_bpm), state.anchor.time_ms, variant)
    cached = context.transition_score_cache.get(cache_key)
    if cached is not None:
        return cached
    if state.prev_bpm is None or not _variant_uses_priors(variant):
        score = _TransitionScore(
            cost=0.0,
            alias_switch_cost=0.0,
            alias_switch_increment=0,
            jump_size_cost=0.0,
            boundary_support_cost=0.0,
        )
        context.transition_score_cache[cache_key] = score
        return score
    alias_switch_cost = _alias_switch_cost_cached(state.prev_bpm, right_bpm, context)
    alias_switch_increment = 1 if alias_switch_cost > 0.0 else 0
    jump_size_cost = min(1.0, abs(math.log2(right_bpm / state.prev_bpm)))
    boundary_support_cost = _boundary_support_cost_cached(
        state.anchor.time_ms,
        context=context,
        include_downbeats=_variant_uses_downbeat(variant),
    )
    cost = (
        0.18 * 1.0
        + 0.12 * alias_switch_cost
        + 0.10 * jump_size_cost
        + 0.15 * boundary_support_cost
    )
    score = _TransitionScore(
        cost=float(cost),
        alias_switch_cost=float(alias_switch_cost),
        alias_switch_increment=alias_switch_increment,
        jump_size_cost=float(jump_size_cost),
        boundary_support_cost=float(boundary_support_cost),
    )
    context.transition_score_cache[cache_key] = score
    return score


def _beat_count_candidates(
    left_anchor: _Anchor,
    right_anchor: _Anchor,
    context: _SearchContext,
) -> tuple[_BeatCountCandidate, ...]:
    key = (left_anchor.anchor_id, right_anchor.anchor_id)
    context.count_cache_visited_keys.add(key)
    cached = context.count_cache.get(key)
    if cached is not None:
        return cached
    delta_ms = right_anchor.time_ms - left_anchor.time_ms
    if delta_ms < context.constants.min_section_duration_ms:
        context.count_cache[key] = ()
        return ()
    unique_counts: set[int] = set()
    for tempo_bpm in context.tempo_bpms:
        n0 = _half_up(delta_ms * tempo_bpm / 60000.0)
        for count in (n0 - 1, n0, n0 + 1):
            if count < 1:
                continue
            bpm = 60000.0 * count / delta_ms
            if not _bpm_is_valid(bpm, context.constants):
                continue
            unique_counts.add(int(count))
    distance_ranked_counts: list[tuple[float, int, float]] = []
    for count in sorted(unique_counts):
        bpm = 60000.0 * count / delta_ms
        tempo_distance = _nearest_tempo_distance_from_ordered(bpm, context.ordered_tempo_bpms)
        distance_ranked_counts.append((float(tempo_distance), int(count), float(bpm)))
    distance_ranked_counts.sort(key=lambda item: (item[0], item[1], item[2]))
    ranked = _rank_beat_count_candidates_with_lazy_support(
        distance_ranked_counts,
        left_anchor=left_anchor,
        right_anchor=right_anchor,
        context=context,
    )
    retained = tuple(
        _BeatCountCandidate(
            count=int(count),
            bpm=float(60000.0 * count / delta_ms),
            rank=index,
            tempo_distance=float(tempo_distance),
            beat_support=float(support),
        )
        for index, (count, tempo_distance, support) in enumerate(
            ranked[: context.constants.max_beat_count_candidates_per_edge]
        )
    )
    context.count_cache[key] = retained
    return retained


def _rank_beat_count_candidates_with_lazy_support(
    distance_ranked_counts: Sequence[tuple[float, int, float]],
    *,
    left_anchor: _Anchor,
    right_anchor: _Anchor,
    context: _SearchContext,
) -> list[tuple[int, float, float]]:
    needed_distances: list[float] = []
    index = 0
    covered_count = 0
    while index < len(distance_ranked_counts):
        tempo_distance = distance_ranked_counts[index][0]
        group_count = 0
        while index < len(distance_ranked_counts) and distance_ranked_counts[index][0] == tempo_distance:
            group_count += 1
            index += 1
        needed_distances.append(tempo_distance)
        covered_count += group_count
        if covered_count >= context.constants.max_beat_count_candidates_per_edge:
            break

    scored_counts: list[tuple[int, float, float]] = []
    delta_ms = right_anchor.time_ms - left_anchor.time_ms
    for tempo_distance, count, bpm in sorted(distance_ranked_counts, key=lambda item: item[1]):
        if not any(tempo_distance == needed_distance for needed_distance in needed_distances):
            continue
        support = _beat_peak_support_for_lattice_count_ranking(
            left_anchor_time_ms=left_anchor.time_ms,
            bpm=bpm,
            start_ms=left_anchor.time_ms,
            end_ms=right_anchor.time_ms,
            beat_peak_times_ms=context.beat_peak_times_ms,
            constants=context.constants,
        )
        scored_counts.append((int(count), float(tempo_distance), float(support)))
    ranked = sorted(
        scored_counts,
        key=lambda item: (item[1], -item[2], item[0], 60000.0 * item[0] / delta_ms),
    )
    return ranked[: context.constants.max_beat_count_candidates_per_edge]


@dataclass(frozen=True)
class _PulseCorrelation:
    valid: bool
    correlation: float


def _pulse_correlation_interval(
    signal: NDArray[np.float64],
    frame_times_ms: NDArray[np.float64],
    *,
    tau_ms: float,
    bpm: float,
    start_ms: float,
    end_ms: float,
    pulse_width_ms: float,
) -> _PulseCorrelation:
    start, end = _frame_interval(frame_times_ms, start_ms, end_ms)
    return _pulse_correlation_slice(
        signal[start:end],
        frame_times_ms[start:end],
        tau_ms=tau_ms,
        bpm=bpm,
        pulse_width_ms=pulse_width_ms,
    )


def _pulse_correlation_slice(
    signal_slice: NDArray[np.float64],
    time_slice_ms: NDArray[np.float64],
    *,
    tau_ms: float,
    bpm: float,
    pulse_width_ms: float,
) -> _PulseCorrelation:
    if signal_slice.size == 0:
        return _PulseCorrelation(valid=True, correlation=0.0)
    period_ms = 60000.0 / bpm
    phase_ms = np.mod(time_slice_ms - tau_ms, period_ms)
    distance_ms = np.minimum(phase_ms, period_ms - phase_ms)
    template = np.maximum(0.0, 1.0 - distance_ms / pulse_width_ms)
    return _centered_correlation(signal_slice, template)


def _pulse_correlation_slice_with_centered_signal(
    centered_signal: NDArray[np.float64],
    signal_norm: float,
    time_slice_ms: NDArray[np.float64],
    *,
    tau_ms: float,
    bpm: float,
    pulse_width_ms: float,
) -> _PulseCorrelation:
    if centered_signal.size == 0:
        return _PulseCorrelation(valid=True, correlation=0.0)
    period_ms = 60000.0 / bpm
    phase_ms = np.mod(time_slice_ms - tau_ms, period_ms)
    distance_ms = np.minimum(phase_ms, period_ms - phase_ms)
    template = np.maximum(0.0, 1.0 - distance_ms / pulse_width_ms)
    return _centered_correlation_with_centered_signal(centered_signal, signal_norm, template)


def _origin_search_correlation(
    centered_signal: NDArray[np.float64],
    signal_norm: float,
    time_slice_ms: NDArray[np.float64],
    *,
    tau_ms: float,
    bpm: float,
    pulse_width_ms: float,
) -> _PulseCorrelation:
    if centered_signal.size == 0:
        return _PulseCorrelation(valid=True, correlation=0.0)
    period_ms = 60000.0 / bpm
    if pulse_width_ms >= 0.5 * period_ms:
        return _pulse_correlation_slice_with_centered_signal(
            centered_signal,
            signal_norm,
            time_slice_ms,
            tau_ms=tau_ms,
            bpm=bpm,
            pulse_width_ms=pulse_width_ms,
        )
    beat_progress = (time_slice_ms - tau_ms) / period_ms
    nearest_beat_index = np.floor(beat_progress + 0.5)
    distance_ms = np.abs(time_slice_ms - (tau_ms + nearest_beat_index * period_ms))
    template = np.maximum(0.0, 1.0 - distance_ms / pulse_width_ms)
    return _centered_correlation_with_centered_signal(centered_signal, signal_norm, template)


def _exact_best_origin_offset_near_search_best(
    centered_signal: NDArray[np.float64],
    signal_norm: float,
    time_slice_ms: NDArray[np.float64],
    search_scores: Sequence[tuple[float, float]],
    *,
    search_best_score: float,
    bpm: float,
    pulse_width_ms: float,
) -> tuple[float, float]:
    best_offset_ms: float | None = None
    best_score: float | None = None
    near_tolerance = 1e-10
    for offset_ms, search_score in search_scores:
        if search_score < search_best_score - near_tolerance:
            continue
        score = _pulse_correlation_slice_with_centered_signal(
            centered_signal,
            signal_norm,
            time_slice_ms,
            tau_ms=offset_ms,
            bpm=bpm,
            pulse_width_ms=pulse_width_ms,
        ).correlation
        if best_score is None or score > best_score or (
            score == best_score and offset_ms < best_offset_ms
        ):
            best_score = score
            best_offset_ms = offset_ms
    if best_offset_ms is None or best_score is None:
        raise ValueError("origin search produced no exact candidate offsets")
    return best_offset_ms, best_score


def _downbeat_pulse_correlation_interval(
    signal: NDArray[np.float64],
    frame_times_ms: NDArray[np.float64],
    *,
    tau_ms: float,
    beat_at_tau: int,
    bpm: float,
    global_downbeat_phase: int,
    start_ms: float,
    end_ms: float,
    pulse_width_ms: float,
) -> _PulseCorrelation:
    start, end = _frame_interval(frame_times_ms, start_ms, end_ms)
    signal_slice = signal[start:end]
    time_slice_ms = frame_times_ms[start:end]
    if signal_slice.size == 0:
        return _PulseCorrelation(valid=True, correlation=0.0)
    period_ms = 60000.0 / bpm
    phase_beats = _canonical_downbeat_phase_beats(
        time_slice_ms,
        tau_ms=tau_ms,
        beat_at_tau=beat_at_tau,
        period_ms=period_ms,
        global_downbeat_phase=global_downbeat_phase,
    )
    distance_beats = np.minimum(phase_beats, 4.0 - phase_beats)
    template = np.maximum(0.0, 1.0 - (distance_beats * period_ms) / pulse_width_ms)
    return _centered_correlation(signal_slice, template)


def _canonical_downbeat_phase_beats(
    time_slice_ms: NDArray[np.float64],
    *,
    tau_ms: float,
    beat_at_tau: int,
    period_ms: float,
    global_downbeat_phase: int,
) -> NDArray[np.float64]:
    beat_progress = (time_slice_ms - tau_ms) / period_ms
    whole_progress = np.floor(beat_progress)
    fractional_progress = beat_progress - whole_progress
    whole_residue = np.remainder(whole_progress, 4.0).astype(np.int64)
    effective_residue = (int(beat_at_tau) - int(global_downbeat_phase)) % 4
    phase_residue = (whole_residue + effective_residue) % 4
    phase_beats = phase_residue.astype(np.float64) + fractional_progress
    return np.where(phase_beats >= 4.0, phase_beats - 4.0, phase_beats)


def _centered_correlation(
    signal_slice: NDArray[np.float64],
    template: NDArray[np.float64],
) -> _PulseCorrelation:
    centered_signal = signal_slice - float(np.mean(signal_slice))
    signal_norm = float(np.linalg.norm(centered_signal))
    return _centered_correlation_with_centered_signal(centered_signal, signal_norm, template)


def _centered_correlation_with_centered_signal(
    centered_signal: NDArray[np.float64],
    signal_norm: float,
    template: NDArray[np.float64],
) -> _PulseCorrelation:
    centered_template = template - float(np.mean(template))
    template_norm = float(np.linalg.norm(centered_template))
    if signal_norm == 0.0 or template_norm == 0.0:
        return _PulseCorrelation(valid=False, correlation=-1.0)
    correlation = float(np.dot(centered_signal, centered_template) / (signal_norm * template_norm))
    if not math.isfinite(correlation):
        return _PulseCorrelation(valid=False, correlation=-1.0)
    return _PulseCorrelation(valid=True, correlation=float(np.clip(correlation, -1.0, 1.0)))


def _peak_recall_precision_cost(
    *,
    left_anchor_time_ms: float,
    bpm: float,
    start_ms: float,
    end_ms: float,
    beat_peak_times_ms: NDArray[np.float64],
    constants: GlobalConstantJumpConstants,
) -> float:
    period_ms = 60000.0 / bpm
    tolerance_ms = min(constants.peak_grid_tolerance_ms, constants.peak_grid_tolerance_beat_fraction * period_ms)
    grid_beats = _grid_times_in_interval(left_anchor_time_ms, period_ms, start_ms, end_ms)
    peak_start, peak_end = _time_interval(beat_peak_times_ms, start_ms, end_ms)
    peaks = beat_peak_times_ms[peak_start:peak_end]
    if grid_beats.size == 0:
        missed_rate = 0.0
    else:
        missed_rate = _unmatched_fraction(grid_beats, peaks, tolerance_ms)
    if peaks.size == 0:
        extra_rate = 0.0
    else:
        extra_rate = _unmatched_fraction(peaks, grid_beats, tolerance_ms)
    return float(0.5 * (missed_rate + extra_rate))


def _peak_recall_precision_cost_fast(
    *,
    left_anchor_time_ms: float,
    bpm: float,
    start_ms: float,
    end_ms: float,
    beat_peak_times_ms: NDArray[np.float64],
    constants: GlobalConstantJumpConstants,
) -> float:
    """Exact regular-lattice recall/precision without materializing the grid."""
    period_ms = 60000.0 / bpm
    tolerance_ms = min(constants.peak_grid_tolerance_ms, constants.peak_grid_tolerance_beat_fraction * period_ms)
    index_range = _regular_lattice_index_range(left_anchor_time_ms, period_ms, start_ms, end_ms)
    if index_range is None:
        return _peak_recall_precision_cost_fast_scalar(
            left_anchor_time_ms=left_anchor_time_ms,
            bpm=bpm,
            start_ms=start_ms,
            end_ms=end_ms,
            beat_peak_times_ms=beat_peak_times_ms,
            constants=constants,
        )
    if not _uses_normal_lattice_tolerance(period_ms, tolerance_ms):
        return _peak_recall_precision_cost_fast_scalar(
            left_anchor_time_ms=left_anchor_time_ms,
            bpm=bpm,
            start_ms=start_ms,
            end_ms=end_ms,
            beat_peak_times_ms=beat_peak_times_ms,
            constants=constants,
        )

    first_index, last_index = index_range
    grid_count = last_index - first_index
    peak_start, peak_end = _time_interval(beat_peak_times_ms, start_ms, end_ms)
    peaks = beat_peak_times_ms[peak_start:peak_end]
    matched_grid_count, matched_peak_count = _matched_regular_lattice_counts_from_peaks(
        left_anchor_time_ms=left_anchor_time_ms,
        period_ms=period_ms,
        start_ms=start_ms,
        end_ms=end_ms,
        first_index=first_index,
        last_index=last_index,
        peaks=peaks,
        tolerance_ms=tolerance_ms,
    )

    if grid_count == 0:
        missed_rate = 0.0
    else:
        missed_rate = float((grid_count - matched_grid_count) / grid_count)
    if peaks.size == 0:
        extra_rate = 0.0
    else:
        extra_rate = float((int(peaks.size) - matched_peak_count) / int(peaks.size))
    return float(0.5 * (missed_rate + extra_rate))


def _peak_recall_precision_cost_fast_scalar(
    *,
    left_anchor_time_ms: float,
    bpm: float,
    start_ms: float,
    end_ms: float,
    beat_peak_times_ms: NDArray[np.float64],
    constants: GlobalConstantJumpConstants,
) -> float:
    period_ms = 60000.0 / bpm
    tolerance_ms = min(constants.peak_grid_tolerance_ms, constants.peak_grid_tolerance_beat_fraction * period_ms)
    if not all(
        math.isfinite(value)
        for value in (left_anchor_time_ms, period_ms, tolerance_ms, start_ms, end_ms)
    ) or period_ms <= 0.0:
        return _peak_recall_precision_cost(
            left_anchor_time_ms=left_anchor_time_ms,
            bpm=bpm,
            start_ms=start_ms,
            end_ms=end_ms,
            beat_peak_times_ms=beat_peak_times_ms,
            constants=constants,
        )

    first_index = math.ceil((start_ms - left_anchor_time_ms) / period_ms)
    last_index = math.ceil((end_ms - left_anchor_time_ms) / period_ms)
    if max(abs(first_index), abs(last_index)) > 2**53:
        return _peak_recall_precision_cost(
            left_anchor_time_ms=left_anchor_time_ms,
            bpm=bpm,
            start_ms=start_ms,
            end_ms=end_ms,
            beat_peak_times_ms=beat_peak_times_ms,
            constants=constants,
        )

    while first_index < last_index:
        first_time_ms = left_anchor_time_ms + np.float64(first_index) * period_ms
        if first_time_ms >= start_ms and first_time_ms < end_ms:
            break
        if first_time_ms >= end_ms:
            first_index = last_index
            break
        first_index += 1
    while first_index < last_index:
        final_time_ms = left_anchor_time_ms + np.float64(last_index - 1) * period_ms
        if final_time_ms >= start_ms and final_time_ms < end_ms:
            break
        last_index -= 1

    grid_count = last_index - first_index
    peak_start, peak_end = _time_interval(beat_peak_times_ms, start_ms, end_ms)
    peaks = beat_peak_times_ms[peak_start:peak_end]
    matched_grid_indexes: set[int] = set()
    matched_peak_count = 0
    if grid_count > 0 and peaks.size > 0:
        for peak_time_value in peaks:
            peak_time_ms = float(peak_time_value)
            lower_index = math.ceil((peak_time_ms - tolerance_ms - left_anchor_time_ms) / period_ms) - 1
            upper_index = math.floor((peak_time_ms + tolerance_ms - left_anchor_time_ms) / period_ms) + 1
            peak_matched = False
            for candidate_index in range(
                max(first_index, lower_index),
                min(last_index - 1, upper_index) + 1,
            ):
                grid_time_ms = left_anchor_time_ms + np.float64(candidate_index) * period_ms
                if not (grid_time_ms >= start_ms and grid_time_ms < end_ms):
                    continue
                if abs(peak_time_ms - float(grid_time_ms)) <= tolerance_ms:
                    matched_grid_indexes.add(int(candidate_index))
                    peak_matched = True
            if peak_matched:
                matched_peak_count += 1

    if grid_count == 0:
        missed_rate = 0.0
    else:
        missed_rate = float((grid_count - len(matched_grid_indexes)) / grid_count)
    if peaks.size == 0:
        extra_rate = 0.0
    else:
        extra_rate = float((int(peaks.size) - matched_peak_count) / int(peaks.size))
    return float(0.5 * (missed_rate + extra_rate))


def _beat_peak_support_for_lattice(
    *,
    left_anchor_time_ms: float,
    bpm: float,
    start_ms: float,
    end_ms: float,
    beat_peak_times_ms: NDArray[np.float64],
    constants: GlobalConstantJumpConstants,
) -> float:
    period_ms = 60000.0 / bpm
    tolerance_ms = min(constants.peak_grid_tolerance_ms, constants.peak_grid_tolerance_beat_fraction * period_ms)
    grid_beats = _grid_times_in_interval(left_anchor_time_ms, period_ms, start_ms, end_ms)
    if grid_beats.size == 0:
        return 0.0
    peak_start, peak_end = _time_interval(beat_peak_times_ms, start_ms, end_ms)
    peaks = beat_peak_times_ms[peak_start:peak_end]
    return float(1.0 - _unmatched_fraction(grid_beats, peaks, tolerance_ms))


def _beat_peak_support_for_lattice_count_ranking(
    *,
    left_anchor_time_ms: float,
    bpm: float,
    start_ms: float,
    end_ms: float,
    beat_peak_times_ms: NDArray[np.float64],
    constants: GlobalConstantJumpConstants,
) -> float:
    period_ms = 60000.0 / bpm
    tolerance_ms = min(constants.peak_grid_tolerance_ms, constants.peak_grid_tolerance_beat_fraction * period_ms)
    index_range = _regular_lattice_index_range(left_anchor_time_ms, period_ms, start_ms, end_ms)
    if index_range is None:
        return _beat_peak_support_for_lattice_count_ranking_scalar(
            left_anchor_time_ms=left_anchor_time_ms,
            bpm=bpm,
            start_ms=start_ms,
            end_ms=end_ms,
            beat_peak_times_ms=beat_peak_times_ms,
            constants=constants,
        )
    if not _uses_normal_lattice_tolerance(period_ms, tolerance_ms):
        return _beat_peak_support_for_lattice_count_ranking_scalar(
            left_anchor_time_ms=left_anchor_time_ms,
            bpm=bpm,
            start_ms=start_ms,
            end_ms=end_ms,
            beat_peak_times_ms=beat_peak_times_ms,
            constants=constants,
        )

    first_index, last_index = index_range
    grid_count = last_index - first_index
    if grid_count <= 0:
        return 0.0
    peak_start, peak_end = _time_interval(beat_peak_times_ms, start_ms, end_ms)
    if peak_end <= peak_start:
        return 0.0
    matched_grid_count, _ = _matched_regular_lattice_counts_from_peaks(
        left_anchor_time_ms=left_anchor_time_ms,
        period_ms=period_ms,
        start_ms=start_ms,
        end_ms=end_ms,
        first_index=first_index,
        last_index=last_index,
        peaks=beat_peak_times_ms[peak_start:peak_end],
        tolerance_ms=tolerance_ms,
    )
    return float(1.0 - ((grid_count - matched_grid_count) / grid_count))


def _beat_peak_support_for_lattice_count_ranking_scalar(
    *,
    left_anchor_time_ms: float,
    bpm: float,
    start_ms: float,
    end_ms: float,
    beat_peak_times_ms: NDArray[np.float64],
    constants: GlobalConstantJumpConstants,
) -> float:
    period_ms = 60000.0 / bpm
    tolerance_ms = min(constants.peak_grid_tolerance_ms, constants.peak_grid_tolerance_beat_fraction * period_ms)
    first_index = math.ceil((start_ms - left_anchor_time_ms) / period_ms)
    last_index = math.ceil((end_ms - left_anchor_time_ms) / period_ms)
    if last_index <= first_index:
        return 0.0
    while first_index < last_index and left_anchor_time_ms + float(first_index) * period_ms < start_ms:
        first_index += 1
    while first_index < last_index and left_anchor_time_ms + float(last_index - 1) * period_ms >= end_ms:
        last_index -= 1
    grid_count = last_index - first_index
    if grid_count <= 0:
        return 0.0
    peak_start, peak_end = _time_interval(beat_peak_times_ms, start_ms, end_ms)
    if peak_end <= peak_start:
        return 0.0
    matched_indexes: set[int] = set()
    for peak_time_ms in beat_peak_times_ms[peak_start:peak_end]:
        nearest_floor = math.floor((float(peak_time_ms) - left_anchor_time_ms) / period_ms)
        for candidate_index in range(nearest_floor - 1, nearest_floor + 3):
            if candidate_index < first_index or candidate_index >= last_index:
                continue
            grid_time_ms = left_anchor_time_ms + float(candidate_index) * period_ms
            if abs(float(peak_time_ms) - grid_time_ms) <= tolerance_ms:
                matched_indexes.add(int(candidate_index))
    return float(1.0 - ((grid_count - len(matched_indexes)) / grid_count))


def _regular_lattice_index_range(
    tau_ms: float,
    period_ms: float,
    start_ms: float,
    end_ms: float,
) -> tuple[int, int] | None:
    if not all(math.isfinite(value) for value in (tau_ms, period_ms, start_ms, end_ms)):
        return None
    if period_ms <= 0.0 or end_ms < start_ms:
        return None
    first_index = math.ceil((start_ms - tau_ms) / period_ms)
    last_index = math.ceil((end_ms - tau_ms) / period_ms)
    if max(abs(first_index), abs(last_index)) > 2**53:
        return None
    while first_index < last_index:
        first_time_ms = tau_ms + np.float64(first_index) * period_ms
        if first_time_ms >= start_ms and first_time_ms < end_ms:
            break
        if first_time_ms >= end_ms:
            first_index = last_index
            break
        first_index += 1
    while first_index < last_index:
        final_time_ms = tau_ms + np.float64(last_index - 1) * period_ms
        if final_time_ms >= start_ms and final_time_ms < end_ms:
            break
        last_index -= 1
    return first_index, last_index


def _uses_normal_lattice_tolerance(period_ms: float, tolerance_ms: float) -> bool:
    return (
        math.isfinite(period_ms)
        and math.isfinite(tolerance_ms)
        and period_ms > 0.0
        and tolerance_ms >= 0.0
        and tolerance_ms < 0.5 * period_ms
    )


def _matched_regular_lattice_counts_from_peaks(
    *,
    left_anchor_time_ms: float,
    period_ms: float,
    start_ms: float,
    end_ms: float,
    first_index: int,
    last_index: int,
    peaks: NDArray[np.float64],
    tolerance_ms: float,
) -> tuple[int, int]:
    if last_index <= first_index or peaks.size == 0:
        return 0, 0
    nearest_floor = np.floor((peaks - left_anchor_time_ms) / period_ms).astype(np.int64)
    candidate_indexes = nearest_floor[:, np.newaxis] + np.asarray((-1, 0, 1, 2), dtype=np.int64)
    candidate_times = left_anchor_time_ms + candidate_indexes.astype(np.float64) * period_ms
    matches = (
        (candidate_indexes >= first_index)
        & (candidate_indexes < last_index)
        & (candidate_times >= start_ms)
        & (candidate_times < end_ms)
        & (np.abs(peaks[:, np.newaxis] - candidate_times) <= tolerance_ms)
    )
    if not bool(np.any(matches)):
        return 0, 0
    matched_peak_count = int(np.count_nonzero(np.any(matches, axis=1)))
    matched_grid_count = int(np.unique(candidate_indexes[matches]).size)
    return matched_grid_count, matched_peak_count


def _boundary_support_cost(
    boundary_time_ms: float,
    *,
    context: _SearchContext,
    include_downbeats: bool,
) -> float:
    support = _support_near_time(
        boundary_time_ms,
        context.beat_peak_times_ms,
        context.beat_peak_confidences,
        tolerance_ms=context.constants.boundary_support_tolerance_ms,
    )
    if include_downbeats:
        support = max(
            support,
            _support_near_time(
                boundary_time_ms,
                context.downbeat_peak_times_ms,
                context.downbeat_peak_confidences,
                tolerance_ms=context.constants.boundary_support_tolerance_ms,
            ),
        )
    return float(1.0 - support)


def _boundary_support_cost_cached(
    boundary_time_ms: float,
    *,
    context: _SearchContext,
    include_downbeats: bool,
) -> float:
    cache_key = (float(boundary_time_ms), bool(include_downbeats))
    cached = context.boundary_support_cost_cache.get(cache_key)
    if cached is not None:
        return cached
    cost = _boundary_support_cost(
        boundary_time_ms,
        context=context,
        include_downbeats=include_downbeats,
    )
    context.boundary_support_cost_cache[cache_key] = cost
    return cost


def _support_near_time(
    time_ms: float,
    times_ms: NDArray[np.float64],
    confidences: NDArray[np.float64],
    *,
    tolerance_ms: float,
) -> float:
    if times_ms.size == 0:
        return 0.0
    start = int(np.searchsorted(times_ms, time_ms - tolerance_ms, side="left"))
    end = int(np.searchsorted(times_ms, time_ms + tolerance_ms, side="right"))
    if end <= start:
        return 0.0
    distances_ms = np.abs(times_ms[start:end] - time_ms)
    supports = confidences[start:end] * (1.0 - distances_ms / tolerance_ms)
    return float(np.max(supports))


def _grid_times_in_interval(
    tau_ms: float,
    period_ms: float,
    start_ms: float,
    end_ms: float,
) -> NDArray[np.float64]:
    first_index = math.ceil((start_ms - tau_ms) / period_ms)
    last_index = math.ceil((end_ms - tau_ms) / period_ms)
    if last_index <= first_index:
        return np.asarray((), dtype=np.float64)
    indexes = np.arange(first_index, last_index, dtype=np.float64)
    times = tau_ms + indexes * period_ms
    return times[(times >= start_ms) & (times < end_ms)]


def _terminal_bpms(
    state: _State,
    candidates: GlobalConstantJumpCandidateSet,
    context: _SearchContext,
    *,
    constant_only: bool,
) -> tuple[float, ...]:
    origin_bpm = None if state.anchor.origin is None else float(state.anchor.origin.bpm)
    cache_key = (state.prev_bpm, origin_bpm, bool(constant_only))
    cached = context.terminal_bpms_cache.get(cache_key)
    if cached is not None:
        return cached
    bpms: list[float] = []
    if state.prev_bpm is not None:
        bpms.append(float(state.prev_bpm))
    elif state.anchor.origin is not None:
        bpms.append(float(state.anchor.origin.bpm))
    if not constant_only:
        bpms.extend(context.tempo_bpms)
    deduped = _dedupe_floats(bpms)
    retained = tuple(bpm for bpm in deduped if _bpm_is_valid(bpm, context.constants))[
        : context.constants.max_tempo_candidates_retained
    ]
    context.terminal_bpms_cache[cache_key] = retained
    return retained


def _origin_downbeat_phases(
    context: _SearchContext,
    variant: str,
) -> tuple[int | None, ...]:
    if not _variant_uses_downbeat(variant):
        return (None,)
    centered = context.downbeat_signal - float(np.mean(context.downbeat_signal))
    if float(np.linalg.norm(centered)) == 0.0:
        return (None,)
    return (0, 1, 2, 3)


def _origin_anchors(candidates: GlobalConstantJumpCandidateSet) -> tuple[_Anchor, ...]:
    return tuple(
        _Anchor(
            anchor_id=origin.anchor_id,
            kind="origin",
            time_ms=origin.time_ms,
            rank_score=origin.score,
            origin=origin,
        )
        for origin in candidates.origin_candidates
    )


def _boundary_anchors(candidates: GlobalConstantJumpCandidateSet) -> tuple[_Anchor, ...]:
    offset = len(candidates.origin_candidates)
    return tuple(
        _Anchor(
            anchor_id=offset + boundary.anchor_id,
            kind="boundary",
            time_ms=boundary.time_ms,
            rank_score=boundary.rank_score,
            boundary=boundary,
        )
        for boundary in sorted(
            candidates.boundary_candidates,
            key=lambda candidate: (candidate.time_ms, -candidate.rank_score, candidate.anchor_id),
        )
    )


def _outgoing_boundary_anchors(
    state: _State,
    boundary_anchors: tuple[_Anchor, ...],
    context: _SearchContext,
) -> tuple[_Anchor, ...]:
    cache_key = (state.anchor.kind, state.anchor.anchor_id, state.anchor.time_ms)
    cached = context.outgoing_boundary_anchors_cache.get(cache_key)
    if cached is not None:
        return cached
    anchors = [
        anchor
        for anchor in boundary_anchors
        if anchor.time_ms > state.anchor.time_ms
        and anchor.time_ms - state.anchor.time_ms >= context.constants.min_section_duration_ms
    ]
    anchors.sort(key=lambda anchor: (anchor.time_ms, -anchor.rank_score, anchor.anchor_id))
    retained = tuple(anchors)
    context.outgoing_boundary_anchors_cache[cache_key] = retained
    return retained


def _initial_state(origin_anchor: _Anchor, *, downbeat_phase: int | None) -> _State:
    origin_time_ms = origin_anchor.time_ms
    return _State(
        anchor=origin_anchor,
        section_count=0,
        beat_at_anchor=0,
        prev_bpm=None,
        prev_alias_family=None,
        downbeat_phase=downbeat_phase,
        origin_time_ms=origin_time_ms,
        duration_objective=0.0,
        transition_objective=0.0,
        objective=0.0,
        alias_switch_count=0,
        max_boundary_displacement_ms=0.0,
        sections=(),
        edge_tuples=(),
        replay_key=(round(origin_time_ms, 6), downbeat_phase),
    )


def _grid_from_state(state: _State, context: _SearchContext) -> TimingV3Grid | None:
    try:
        return TimingV3Grid(
            origin_beat=0,
            origin_time_ms=state.origin_time_ms,
            sections=tuple(
                ConstantTimingSection(start_beat=section.start_beat, end_beat=section.end_beat, bpm=section.bpm)
                for section in state.sections
            ),
            coverage_start_ms=context.coverage_start_ms,
            coverage_end_ms=context.coverage_end_ms,
        )
    except (TypeError, ValueError):
        return None


def _success_diagnostics(
    variant: str,
    candidates: GlobalConstantJumpCandidateSet,
    context: _SearchContext,
    state: _State,
    grid: TimingV3Grid,
) -> GlobalConstantJumpDiagnostics:
    return GlobalConstantJumpDiagnostics(
        variant=variant,
        candidate_contract_version=CANDIDATE_CONTRACT_VERSION,
        constants_json_sha256=GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON_SHA256,
        coverage_start_ms=context.coverage_start_ms,
        coverage_end_ms=context.coverage_end_ms,
        frame_count=context.frame_count,
        frame_rate_hz=context.frame_rate_hz,
        min_period_frames=candidates.diagnostics.min_period_frames,
        max_period_frames=candidates.diagnostics.max_period_frames,
        beat_peak_count=len(candidates.beat_peaks),
        downbeat_peak_count=len(candidates.downbeat_peaks),
        tempo_candidate_count=len(candidates.tempo_candidates),
        origin_candidate_count=len(candidates.origin_candidates),
        boundary_candidate_count=len(candidates.boundary_candidates),
        section_attempt_count=context.attempt_count,
        edge_count_cache_size=context.edge_count_cache_size,
        section_score_cache_size=len(context.section_score_cache) + len(context.terminal_score_cache),
        beam_pruned_state_count=context.beam_pruned_state_count,
        selected_section_count=len(grid.sections),
        selected_origin_time_ms=state.origin_time_ms,
        selected_downbeat_phase=state.downbeat_phase,
        objective=float(state.objective),
        duration_objective=float(state.duration_objective),
        transition_objective=float(state.transition_objective),
        alias_switch_count=state.alias_switch_count,
        max_boundary_displacement_ms=float(state.max_boundary_displacement_ms),
        fallback_reason=None,
        input_signal_sha256=candidates.diagnostics.input_signal_sha256,
        candidate_fingerprint=candidates.diagnostics.candidate_fingerprint,
        replay_fingerprint=_json_fingerprint(state.replay_key),
        grid_fingerprint=_json_fingerprint(grid.to_dict()),
    )


def _failure_result(
    variant: str,
    candidates: GlobalConstantJumpCandidateSet,
    context: _SearchContext,
    reason: str,
) -> GlobalConstantJumpResult:
    diagnostics = GlobalConstantJumpDiagnostics(
        variant=variant,
        candidate_contract_version=CANDIDATE_CONTRACT_VERSION,
        constants_json_sha256=GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON_SHA256,
        coverage_start_ms=context.coverage_start_ms,
        coverage_end_ms=context.coverage_end_ms,
        frame_count=context.frame_count,
        frame_rate_hz=context.frame_rate_hz,
        min_period_frames=candidates.diagnostics.min_period_frames,
        max_period_frames=candidates.diagnostics.max_period_frames,
        beat_peak_count=len(candidates.beat_peaks),
        downbeat_peak_count=len(candidates.downbeat_peaks),
        tempo_candidate_count=len(candidates.tempo_candidates),
        origin_candidate_count=len(candidates.origin_candidates),
        boundary_candidate_count=len(candidates.boundary_candidates),
        section_attempt_count=context.attempt_count,
        edge_count_cache_size=context.edge_count_cache_size,
        section_score_cache_size=len(context.section_score_cache) + len(context.terminal_score_cache),
        beam_pruned_state_count=context.beam_pruned_state_count,
        selected_section_count=0,
        selected_origin_time_ms=None,
        selected_downbeat_phase=None,
        objective=None,
        duration_objective=None,
        transition_objective=None,
        alias_switch_count=0,
        max_boundary_displacement_ms=0.0,
        fallback_reason=reason,
        input_signal_sha256=candidates.diagnostics.input_signal_sha256,
        candidate_fingerprint=candidates.diagnostics.candidate_fingerprint,
        replay_fingerprint=_json_fingerprint(
            (
                variant,
                reason,
                candidates.diagnostics.candidate_fingerprint,
                context.search_trace_fingerprint(),
            )
        ),
        grid_fingerprint=None,
    )
    return GlobalConstantJumpResult(variant=variant, grid=None, diagnostics=diagnostics, reason=reason)


def _candidate_fingerprint(
    *,
    tempo_candidates: tuple[TempoCandidate, ...],
    origin_candidates: tuple[OriginCandidate, ...],
    boundary_candidates: tuple[BoundaryCandidate, ...],
    beat_peaks: tuple[MaterializedPeak, ...],
    downbeat_peaks: tuple[MaterializedPeak, ...],
    input_signal_sha256: str,
) -> str:
    payload = {
        "contract": CANDIDATE_CONTRACT_VERSION,
        "constants_sha256": GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON_SHA256,
        "input_signal_sha256": input_signal_sha256,
        "beat_peaks": [
            (peak.frame_index, round(peak.refined_frame, 6), round(peak.time_ms, 6), round(peak.confidence, 6))
            for peak in beat_peaks
        ],
        "downbeat_peaks": [
            (peak.frame_index, round(peak.refined_frame, 6), round(peak.time_ms, 6), round(peak.confidence, 6))
            for peak in downbeat_peaks
        ],
        "tempo_candidates": [
            (round(candidate.bpm, 6), candidate.source, round(candidate.score, 6))
            for candidate in tempo_candidates
        ],
        "origin_candidates": [
            (candidate.anchor_id, round(candidate.time_ms, 6), round(candidate.bpm, 6), round(candidate.score, 6))
            for candidate in origin_candidates
        ],
        "boundary_candidates": [
            (
                candidate.anchor_id,
                round(candidate.time_ms, 6),
                candidate.source_peak_index,
                round(candidate.source_peak_time_ms, 6),
                round(candidate.source_peak_confidence, 6),
                round(candidate.displacement_from_source_peak_ms, 6),
                round(candidate.left_period_ms, 6),
                round(candidate.right_period_ms, 6),
                candidate.evidence_mode,
                round(candidate.rank_score, 6),
                None if candidate.ordinary_score is None else round(candidate.ordinary_score, 6),
                None if candidate.super_score is None else round(candidate.super_score, 6),
                round(candidate.downbeat_bonus, 6),
                (
                    None
                    if candidate.nearest_downbeat_distance_ms is None
                    else round(candidate.nearest_downbeat_distance_ms, 6)
                ),
            )
            for candidate in boundary_candidates
        ],
    }
    return _json_fingerprint(payload)


def _input_signal_sha256(
    beat_signal: NDArray[np.float64],
    downbeat_signal: NDArray[np.float64],
) -> str:
    beat_float32 = np.asarray(beat_signal, dtype="<f4")
    downbeat_float32 = np.asarray(downbeat_signal, dtype="<f4")
    digest = hashlib.sha256()
    digest.update(b"timing-v3-exp004-input-signal-v1\0beat\0")
    digest.update(np.ascontiguousarray(beat_float32).tobytes())
    digest.update(b"\0downbeat\0")
    digest.update(np.ascontiguousarray(downbeat_float32).tobytes())
    return digest.hexdigest()


def _json_fingerprint(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _invalid_section_score() -> _SectionScore:
    return _SectionScore(
        valid=False,
        cost=0.0,
        beat_support_cost=0.0,
        peak_recall_precision_cost=0.0,
        downbeat_phase_cost=0.0,
        bpm_prior_cost=0.0,
        beat_count_prior_cost=0.0,
        section_duration_cost=0.0,
    )


def _variant_uses_priors(variant: str) -> bool:
    return variant in {VARIANT_CJ0, VARIANT_CJ2, VARIANT_CJ3}


def _variant_uses_downbeat(variant: str) -> bool:
    return variant in {VARIANT_CJ0, VARIANT_CJ3}


def _effective_downbeat_residue(
    *,
    beat_at_tau: int,
    global_downbeat_phase: int | None,
    variant: str,
) -> int | None:
    if global_downbeat_phase is None or not _variant_uses_downbeat(variant):
        return None
    return (int(beat_at_tau) - int(global_downbeat_phase)) % 4


def _bpm_is_valid(bpm: float, constants: GlobalConstantJumpConstants) -> bool:
    return math.isfinite(bpm) and constants.hard_min_bpm <= bpm <= constants.hard_max_bpm


def _bpm_prior_cost(bpm: float, constants: GlobalConstantJumpConstants) -> float:
    clamped = min(max(bpm, constants.preferred_min_bpm), constants.preferred_max_bpm)
    return float(abs(math.log2(bpm / clamped)))


def _beat_count_prior_cost(beat_count: int) -> float:
    clamped = min(max(beat_count, 16), 384)
    return float(min(1.0, abs(math.log2(beat_count / clamped))))


def _section_duration_cost(duration_ms: float) -> float:
    clamped = min(max(duration_ms, 8000.0), 180000.0)
    return float(min(1.0, abs(math.log2(duration_ms / clamped))))


def _alias_switch_cost(left_bpm: float, right_bpm: float, constants: GlobalConstantJumpConstants) -> float:
    if _relative_close(left_bpm, right_bpm, tolerance=0.005):
        return 0.0
    if any(_relative_close(left_bpm * multiplier, right_bpm, tolerance=0.005) for multiplier in constants.alias_multipliers):
        return 0.5
    return 1.0


def _alias_switch_cost_cached(left_bpm: float, right_bpm: float, context: _SearchContext) -> float:
    cache_key = (float(left_bpm), float(right_bpm))
    cached = context.alias_switch_cost_cache.get(cache_key)
    if cached is not None:
        return cached
    cost = _alias_switch_cost(left_bpm, right_bpm, context.constants)
    context.alias_switch_cost_cache[cache_key] = cost
    return cost


def _alias_family_representative_v1(bpm: float, constants: GlobalConstantJumpConstants) -> float:
    preferred = sorted(
        candidate
        for multiplier in constants.alias_multipliers
        if _bpm_is_valid((candidate := bpm * multiplier), constants)
        and constants.preferred_min_bpm <= candidate < 160.0
    )
    if preferred:
        return float(round(preferred[0], 6))
    valid = sorted(
        candidate
        for multiplier in constants.alias_multipliers
        if _bpm_is_valid((candidate := bpm * multiplier), constants)
    )
    return float(round(valid[0] if valid else bpm, 6))


def _alias_family_representative_cached(bpm: float, context: _SearchContext) -> float:
    cache_key = float(bpm)
    cached = context.alias_family_cache.get(cache_key)
    if cached is not None:
        return cached
    representative = _alias_family_representative_v1(bpm, context.constants)
    context.alias_family_cache[cache_key] = representative
    return representative


def _relative_close(left: float, right: float, *, tolerance: float) -> bool:
    return abs(left - right) / max(abs(right), 1e-12) <= tolerance


def _nearest_tempo_distance(bpm: float, tempo_bpms: tuple[float, ...]) -> float:
    ordered = tuple(sorted(tempo_bpm for tempo_bpm in tempo_bpms if math.isfinite(tempo_bpm) and tempo_bpm > 0.0))
    return _nearest_tempo_distance_from_ordered(bpm, ordered)


def _nearest_tempo_distance_from_ordered(bpm: float, ordered_tempo_bpms: tuple[float, ...]) -> float:
    if not ordered_tempo_bpms:
        return 1.0
    insertion_index = bisect_left(ordered_tempo_bpms, bpm)
    best_distance = math.inf
    if insertion_index < len(ordered_tempo_bpms):
        best_distance = abs(math.log2(bpm / ordered_tempo_bpms[insertion_index]))
    if insertion_index > 0:
        left_distance = abs(math.log2(bpm / ordered_tempo_bpms[insertion_index - 1]))
        if left_distance < best_distance:
            best_distance = left_distance
    return float(best_distance)


def _dedupe_floats(values: Iterable[float]) -> tuple[float, ...]:
    seen: set[float] = set()
    deduped: list[float] = []
    for value in values:
        if not math.isfinite(value):
            continue
        key = round(float(value), 6)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(float(key))
    return tuple(deduped)


def _half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _first_start_beat(*, origin_time_ms: float, bpm: float) -> int:
    period_ms = 60000.0 / bpm
    return int(math.floor((0.0 - origin_time_ms) / period_ms))


def _duration_ms(context: _SearchContext) -> float:
    return context.coverage_end_ms - context.coverage_start_ms


def _transition_normalizer(context: _SearchContext) -> float:
    return max(1.0, _duration_ms(context) / 60000.0)


def _frame_interval(
    frame_times_ms: NDArray[np.float64],
    start_ms: float,
    end_ms: float,
) -> tuple[int, int]:
    return (
        int(np.searchsorted(frame_times_ms, start_ms, side="left")),
        int(np.searchsorted(frame_times_ms, end_ms, side="left")),
    )


def _time_interval(
    times_ms: NDArray[np.float64],
    start_ms: float,
    end_ms: float,
) -> tuple[int, int]:
    return (
        int(np.searchsorted(times_ms, start_ms, side="left")),
        int(np.searchsorted(times_ms, end_ms, side="left")),
    )


def _has_time_within(times_ms: NDArray[np.float64], time_ms: float, tolerance_ms: float) -> bool:
    if times_ms.size == 0:
        return False
    index = int(np.searchsorted(times_ms, time_ms))
    if index < times_ms.size and abs(float(times_ms[index]) - time_ms) <= tolerance_ms:
        return True
    if index > 0 and abs(float(times_ms[index - 1]) - time_ms) <= tolerance_ms:
        return True
    return False


def _unmatched_fraction(
    query_times_ms: NDArray[np.float64],
    reference_times_ms: NDArray[np.float64],
    tolerance_ms: float,
) -> float:
    if query_times_ms.size == 0:
        return 0.0
    if reference_times_ms.size == 0:
        return 1.0
    indexes = np.searchsorted(reference_times_ms, query_times_ms)
    distances = np.full(query_times_ms.shape, np.inf, dtype=np.float64)
    after_mask = indexes < reference_times_ms.size
    if np.any(after_mask):
        distances[after_mask] = np.minimum(
            distances[after_mask],
            np.abs(reference_times_ms[indexes[after_mask]] - query_times_ms[after_mask]),
        )
    before_mask = indexes > 0
    if np.any(before_mask):
        before_indexes = indexes[before_mask] - 1
        distances[before_mask] = np.minimum(
            distances[before_mask],
            np.abs(reference_times_ms[before_indexes] - query_times_ms[before_mask]),
        )
    return float(np.count_nonzero(distances > tolerance_ms) / query_times_ms.size)


def _nearest_time_index(times_ms: NDArray[np.float64], time_ms: float) -> int:
    index = int(np.searchsorted(times_ms, time_ms))
    if index <= 0:
        return 0
    if index >= times_ms.size:
        return int(times_ms.size - 1)
    before = index - 1
    after = index
    if abs(float(times_ms[before]) - time_ms) <= abs(float(times_ms[after]) - time_ms):
        return before
    return after


def _boundary_rank_key(candidate: _RawBoundaryCandidate) -> tuple[float, int, float, int]:
    mode_rank = 0 if candidate.evidence_mode == "ordinary" else 1
    return (-candidate.rank_score, mode_rank, candidate.time_ms, candidate.source_peak_index)


def _state_order_key_parts(state: _State) -> _StateOrderKeyParts:
    return _StateOrderKeyParts(
        objective=state.objective,
        section_count=state.section_count,
        alias_switch_count=state.alias_switch_count,
        max_boundary_displacement_ms=state.max_boundary_displacement_ms,
        origin_time_ms=state.origin_time_ms,
        edge_tuples=state.edge_tuples,
        edge_suffix=None,
        replay_key=state.replay_key,
        replay_suffix=None,
    )


def _pending_state_order_key(
    parent: _State,
    *,
    section_count: int,
    objective: float,
    alias_switch_count: int,
    max_boundary_displacement_ms: float,
    edge_tuple: tuple[int, int, int, float],
    replay_item: Any,
) -> _StateOrderKeyParts:
    return _StateOrderKeyParts(
        objective=objective,
        section_count=section_count,
        alias_switch_count=alias_switch_count,
        max_boundary_displacement_ms=max_boundary_displacement_ms,
        origin_time_ms=parent.origin_time_ms,
        edge_tuples=parent.edge_tuples,
        edge_suffix=edge_tuple,
        replay_key=parent.replay_key,
        replay_suffix=replay_item,
    )


def _compare_tuple_with_optional_suffix(
    left_prefix: tuple[Any, ...],
    left_suffix: Any | None,
    right_prefix: tuple[Any, ...],
    right_suffix: Any | None,
) -> int:
    left_length = len(left_prefix) + (0 if left_suffix is None else 1)
    right_length = len(right_prefix) + (0 if right_suffix is None else 1)
    shared_length = min(left_length, right_length)
    for index in range(shared_length):
        left_item = left_prefix[index] if index < len(left_prefix) else left_suffix
        right_item = right_prefix[index] if index < len(right_prefix) else right_suffix
        item_comparison = _compare_lexicographic_items((left_item,), (right_item,))
        if item_comparison != 0:
            return item_comparison
    if left_length == right_length:
        return 0
    return -1 if left_length < right_length else 1


def _compare_lexicographic_items(left_items: tuple[Any, ...], right_items: tuple[Any, ...]) -> int:
    shared_length = min(len(left_items), len(right_items))
    for index in range(shared_length):
        left_item = left_items[index]
        right_item = right_items[index]
        if left_item == right_item:
            continue
        if left_item < right_item:
            return -1
        if right_item < left_item:
            return 1
        return 0
    if len(left_items) == len(right_items):
        return 0
    return -1 if len(left_items) < len(right_items) else 1


def _state_order_key(state: _State) -> tuple[Any, ...]:
    return (
        state.objective,
        state.section_count,
        state.alias_switch_count,
        state.max_boundary_displacement_ms,
        state.origin_time_ms,
        state.edge_tuples,
        state.replay_key,
    )


def _complete_state_order_key(state: _State) -> tuple[Any, ...]:
    return _state_order_key(state)


def _frame_index_for_time(time_ms: float, frame_rate_hz: float) -> int:
    return _half_up(time_ms * frame_rate_hz / 1000.0)


__all__ = [
    "BOUNDARY_CANDIDATE_SCORE_VERSION",
    "CANDIDATE_CONTRACT_VERSION",
    "GLOBAL_CONSTANT_JUMP_CONSTANTS",
    "GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON",
    "GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON_SHA256",
    "GLOBAL_CONSTANT_JUMP_VARIANTS",
    "GlobalConstantJumpCandidateDiagnostics",
    "GlobalConstantJumpCandidateSet",
    "GlobalConstantJumpConstants",
    "GlobalConstantJumpDiagnostics",
    "GlobalConstantJumpResult",
    "MaterializedPeak",
    "OriginCandidate",
    "PULSE_CORRELATION_VERSION",
    "REASON_EDGE_ATTEMPT_CAP_EXCEEDED",
    "REASON_NO_GLOBAL_CONSTANT_JUMP_PATH",
    "REASON_NO_ORIGIN_CANDIDATE",
    "REASON_SCHEMA_CONSTRUCTION_FAILED",
    "TempoCandidate",
    "BoundaryCandidate",
    "VARIANT_CJ0",
    "VARIANT_CJ1",
    "VARIANT_CJ2",
    "VARIANT_CJ3",
    "boundary_candidate_score_v1",
    "extract_global_constant_jump_candidates",
    "fit_global_constant_jump",
    "fit_global_constant_jump_variants",
    "iter_global_constant_jump_variants",
    "materialize_global_constant_jump_peaks",
    "pulse_correlation_v1",
]
