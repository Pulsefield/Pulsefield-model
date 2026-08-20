from __future__ import annotations

import hashlib
import json
import math
from array import array
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from enum import Enum
from typing import Any, Iterable, Sequence

import numpy as np

from pulsefield_model.timing.schema import FrameTimingPrediction
from pulsefield_model.timing.v3 import global_constant_jump as exp004
from pulsefield_model.timing.v3.global_constant_jump import (
    GLOBAL_CONSTANT_JUMP_CONSTANTS,
    BoundaryCandidate,
    GlobalConstantJumpCandidateSet,
    TempoCandidate,
)
from pulsefield_model.timing.v3.schema import ConstantTimingSection, TimingV3Grid


LOCAL_FRONTIER_CONTRACT_VERSION = "timing-v3-exp005-local-frontier-v1"
OVERLAP_DISAGREEMENT_VERSION = "timing_v3_overlap_disagreement_v1"
_BOUNDARY_PAIR_TRANSITION_CONTRACT_VERSION_V1 = (
    "timing-v3-exp006-boundary-pair-transition-v1"
)
BOUNDARY_PAIR_TRANSITION_CONTRACT_VERSION = (
    "timing-v3-exp006-boundary-pair-transition-v3"
)
BOUNDARY_PAIR_BOUNDED_CONTRACT_VERSION = (
    "timing-v3-exp007-boundary-pair-bounded-v1"
)
LOCAL_FRONTIER_OVERLAP_METRIC_VERSION = (
    "timing-v3-exp007-local-frontier-overlap-v1"
)
LOOKAHEAD_OVERLAP_RECORD_CONTRACT_VERSION = (
    "timing-v3-exp007-lookahead-overlap-record-v1"
)
BOUNDARY_PAIR_ALIAS_MULTIPLIERS = (0.25, 1.0 / 3.0, 0.5, 1.0, 2.0, 3.0, 4.0)
EXP006_PAIR_SPARSITY_FLOOR = 0.25
MAX_OVERLAP_RECORDS_PER_AUDIO = 16 * (192 - 1)
MAX_OVERLAP_TRACE_BEATS = 501
MAX_OVERLAP_TRACE_REAL_BOUNDARIES = 19
MAX_OVERLAP_RESIDUAL_PAIRS = MAX_OVERLAP_RECORDS_PER_AUDIO * MAX_OVERLAP_TRACE_BEATS

REASON_NO_ORIGIN_CANDIDATE = "no_origin_candidate"
REASON_NO_LOCAL_FRONTIER_PATH = "no_local_frontier_path"
REASON_RESOURCE_CAP_EXCEEDED = "local_frontier_resource_cap_exceeded"
REASON_SCHEMA_CONSTRUCTION_FAILED = "timing_v3_schema_construction_failed"
REASON_DIAGNOSTICS_INTEGRITY_FAILURE = "diagnostics_integrity_failure"

UNAVAILABLE_EMPTY_COMMON_TIME_DOMAIN = "empty_common_time_domain"
UNAVAILABLE_LINEAGE_NOT_RETAINED_AT_NEXT_CUT = "lineage_not_retained_at_next_cut"
UNAVAILABLE_FEWER_THAN_8_COMPARABLE_BEATS = "fewer_than_8_comparable_beats"


class LocalFrontierScheduleArm(str, Enum):
    S30 = "S30"
    S60 = "S60"
    S90 = "S90"
    S64 = "S64"


class LocalFrontierObjectiveVariant(str, Enum):
    """Frozen transition-objective identities available to the prototype."""

    EXP005_CONSTANT_CHANGE = "exp005_constant_change"
    EXP006_PAIR_CONDITIONED_CHANGE_FLOOR_1_4 = (
        "pair_conditioned_change_floor_1_4"
    )


@dataclass(frozen=True)
class LocalFrontierConfig:
    schedule_arm: LocalFrontierScheduleArm = LocalFrontierScheduleArm.S30
    exported_frontier_width: int = 16
    local_beam_width: int = 64
    max_boundary_candidates_per_block: int = 32
    max_tempo_candidates_per_block: int = 64
    max_blocks: int = 192
    max_sections: int = 20
    max_section_score_misses_per_block: int = 30_000
    max_section_score_misses_per_audio: int = 500_000

    def __post_init__(self) -> None:
        try:
            arm = LocalFrontierScheduleArm(self.schedule_arm)
        except ValueError as exc:
            raise ValueError(f"unsupported local-frontier schedule arm: {self.schedule_arm!r}") from exc
        object.__setattr__(self, "schedule_arm", arm)
        frozen_positive = {
            "exported_frontier_width": 16,
            "local_beam_width": 64,
            "max_boundary_candidates_per_block": 32,
            "max_tempo_candidates_per_block": 64,
            "max_blocks": 192,
            "max_sections": 20,
            "max_section_score_misses_per_block": 30_000,
            "max_section_score_misses_per_audio": 500_000,
        }
        for name, expected in frozen_positive.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value != expected:
                raise ValueError(f"{name} is frozen at {expected}, got {value!r}")


@dataclass(frozen=True)
class BoundaryPairTransitionComponents:
    """A cacheable, path-independent decomposition of one transition edge."""

    objective_variant: str
    boundary_anchor_id: int
    source_candidate_time_ms: float
    left_period_ms: float
    right_period_ms: float
    left_bpm: float
    right_bpm: float
    qhat_left_bpm: float
    qhat_right_bpm: float
    alias_min_left_unclipped: float
    alias_min_right_unclipped: float
    alias_distance_left: float
    alias_distance_right: float
    pair_cost: float
    sparsity_floor: float | None
    change_cost: float
    alias_switch_cost: float
    jump_size_cost: float
    boundary_support_cost: float
    raw_transition_cost: float
    transition_normalizer: float
    normalized_increment: float


@dataclass(frozen=True)
class BoundaryPairTransitionLedgerEntry:
    """A transition charge bound to one exact predecessor/successor replay."""

    predecessor_replay_key: tuple[Any, ...]
    successor_replay_key: tuple[Any, ...]
    boundary_beat: int
    boundary_time_ms: float
    components: BoundaryPairTransitionComponents


@dataclass(frozen=True)
class BoundaryPairTransitionCacheRecord:
    """One sorted unique transition-component cache entry."""

    component_cache_key: tuple[Any, ...]
    components: BoundaryPairTransitionComponents


@dataclass(frozen=True)
class BoundaryPairScoredEdgeRecord:
    """One actual scored transition edge in natural generation order."""

    edge_order: int
    block_index: int
    stage: str
    predecessor_replay_key: tuple[Any, ...]
    successor_replay_key: tuple[Any, ...]
    boundary_anchor_id: int
    source_candidate_time_ms: float
    boundary_beat: int
    boundary_time_ms: float
    snapped_beat: int
    snapped_time_ms: float
    left_period_ms: float
    right_period_ms: float
    left_bpm: float
    right_bpm: float
    objective_variant: str
    component_cache_key: tuple[Any, ...]
    components: BoundaryPairTransitionComponents
    normalized_increment: float
    retained_terminal_path: bool = False
    retained_provisional_path: bool = False
    selected_traceback_path: bool = False


@dataclass(frozen=True)
class ProvisionalTransitionLedgerRecord:
    """One lookahead path ledger; deliberately not collapsed by anchor ID."""

    block_index: int
    committed_replay_key: tuple[Any, ...]
    ranked_lookahead_replay_key: tuple[Any, ...]
    transition_entries: tuple[BoundaryPairTransitionLedgerEntry, ...]


@dataclass(frozen=True)
class _ProvisionalTransitionOccurrenceRecord:
    """Internal occurrence ledger parallel to ProvisionalTransitionLedgerRecord."""

    block_index: int
    committed_replay_key: tuple[Any, ...]
    ranked_lookahead_replay_key: tuple[Any, ...]
    edge_orders: tuple[int, ...]


@dataclass(frozen=True)
class BoundaryPairTerminalOccurrenceRecord:
    """Public Exp006 terminal-path occurrence references."""

    selection_rank: int
    selected: bool
    replay_key: tuple[Any, ...]
    edge_orders: tuple[int, ...]


@dataclass(frozen=True)
class BoundaryPairProvisionalOccurrenceRecord:
    """Public Exp006 provisional-path occurrence references."""

    record_index: int
    block_index: int
    committed_replay_key: tuple[Any, ...]
    ranked_lookahead_replay_key: tuple[Any, ...]
    edge_orders: tuple[int, ...]


@dataclass(frozen=True)
class TerminalObjectiveLedger:
    """Complete terminal decomposition for an independently replayable path."""

    selection_rank: int
    selected: bool
    replay_key: tuple[Any, ...]
    section_signature: tuple[tuple[int, int, str], ...]
    bpm_sequence: tuple[float, ...]
    candidate_period_pairs_by_anchor: tuple[tuple[int, float, float], ...]
    duration_objective_numerator: float
    duration_normalizer: float
    normalized_duration_objective: float
    transition_entries: tuple[BoundaryPairTransitionLedgerEntry, ...]
    recorded_transition_objective: float
    tail_prior_numerator: float
    normalized_tail_prior: float
    reconstructed_terminal_objective: float
    recorded_terminal_objective: float


@dataclass(frozen=True)
class BoundaryPairObjectiveDiagnostics:
    contract_version: str
    objective_variant: str
    sparsity_floor: float | None
    alias_multipliers: tuple[float, ...]
    candidate_fingerprint: str
    transition_cache_size: int
    terminal_path_ledgers: tuple[TerminalObjectiveLedger, ...]
    provisional_path_ledgers: tuple[ProvisionalTransitionLedgerRecord, ...]
    selected_terminal_objective: float | None
    runner_up_terminal_objective: float | None
    selected_runner_up_margin: float | None
    deterministic_fingerprint: str


@dataclass(frozen=True)
class BoundaryPairBlockResourceRecord:
    """Exp006-only resource counters captured around one block."""

    block_index: int
    core_start_ms: float
    core_end_ms: float
    lookahead_end_ms: float
    incoming_path_count: int
    raw_committed_path_count: int
    dominant_committed_path_count: int
    lookahead_call_count: int
    lookahead_successor_count: int
    pre_export_state_count: int
    exported_state_count: int
    block_score_miss_count_before: int
    block_score_miss_count_after: int
    row_score_miss_count_before: int
    row_score_miss_count_after: int
    transition_component_cache_count_before: int
    transition_component_cache_count_after: int
    scored_edge_count_before: int
    scored_edge_count_after: int
    dominance_pruned_state_count: int
    width_pruned_state_count: int
    exported_frontier_width_cap: int
    local_beam_width_cap: int
    max_boundary_candidates_per_block_cap: int
    max_tempo_candidates_per_block_cap: int
    max_blocks_cap: int
    max_sections_cap: int
    max_section_score_misses_per_block_cap: int
    max_section_score_misses_per_audio_cap: int


@dataclass(frozen=True)
class BoundaryPairClassCoverageRecord:
    """Exp006-only class-key coverage at the four export stages."""

    block_index: int
    cut_time_ms: float
    input_state_count: int
    input_unique_class_keys: tuple[tuple[float, int | None], ...]
    post_future_equivalence_state_count: int
    post_future_equivalence_unique_class_keys: tuple[tuple[float, int | None], ...]
    reserved_state_count: int
    reserved_unique_class_keys: tuple[tuple[float, int | None], ...]
    final_state_count: int
    final_unique_class_keys: tuple[tuple[float, int | None], ...]


@dataclass(frozen=True)
class BoundaryPairObjectiveDiagnosticsV2(BoundaryPairObjectiveDiagnostics):
    """Exp006-only measurement repair schema."""

    transition_component_cache_entries: tuple[BoundaryPairTransitionCacheRecord, ...]
    actual_scored_edges: tuple[BoundaryPairScoredEdgeRecord, ...]
    block_resource_records: tuple[BoundaryPairBlockResourceRecord, ...]
    class_coverage_records: tuple[BoundaryPairClassCoverageRecord, ...]
    terminal_path_occurrence_records: tuple[
        BoundaryPairTerminalOccurrenceRecord, ...
    ]
    provisional_path_occurrence_records: tuple[
        BoundaryPairProvisionalOccurrenceRecord, ...
    ]
    selected_traceback_edge_orders: tuple[int, ...]


@dataclass(frozen=True)
class TerminalObjectiveReplay:
    transition_objective: float
    terminal_objective: float


@dataclass(frozen=True)
class LookaheadOverlapRecord:
    previous_block_index: int
    next_block_index: int
    previous_export_ordinal: int
    lineage_sha256: str
    comparison_start_ms: float
    comparison_end_ms: float
    provisional_trace_sha256: str
    recomputed_trace_sha256: str | None
    comparison_domain_sha256: str
    comparable_beat_count: int
    residual_vector_sha256: str | None
    p90_ms: float | None
    p90_beats: float | None
    unavailable_reason: str | None


@dataclass(frozen=True)
class LocalFrontierOverlapDiagnostics:
    metric_version: str
    record_contract_version: str
    record_count: int
    available_record_count: int
    unavailable_record_count: int
    comparable_beat_count: int
    p90_ms: float | None
    p90_beats: float | None
    residual_vector_sha256: str | None
    records_sha256: str
    records: tuple[LookaheadOverlapRecord, ...]


@dataclass(frozen=True)
class BoundaryPairBoundedDiagnostics:
    contract_version: str
    objective_variant: str
    candidate_fingerprint: str
    transition_cache_size: int
    actual_scored_edge_count: int
    selected_terminal_objective: float | None
    runner_up_terminal_objective: float | None
    selected_runner_up_margin: float | None
    block_resource_records: tuple[BoundaryPairBlockResourceRecord, ...]
    class_coverage_records: tuple[BoundaryPairClassCoverageRecord, ...]
    overlap: LocalFrontierOverlapDiagnostics
    deterministic_fingerprint: str


@dataclass(frozen=True)
class LocalFrontierState:
    cut_time_ms: float
    next_beat_index: int
    next_beat_time_ms: float
    current_section_start_beat: int
    current_section_start_time_ms: float
    serialized_first_start_beat: int
    current_bpm: float
    previous_bpm: float | None
    alias_family: float
    global_downbeat_phase: int | None
    committed_duration_objective_numerator: float
    committed_transition_objective: float
    rank_objective: float
    committed_objective: float
    real_section_count: int
    alias_switch_count: int
    max_boundary_displacement_ms: float
    open_section_state: tuple[Any, ...]
    prefix_sections_or_backpointer: tuple[Any, ...]
    deterministic_replay_key: tuple[Any, ...]

    def __post_init__(self) -> None:
        finite_fields = (
            "cut_time_ms",
            "next_beat_time_ms",
            "current_section_start_time_ms",
            "current_bpm",
            "alias_family",
            "committed_duration_objective_numerator",
            "committed_transition_objective",
            "rank_objective",
            "committed_objective",
            "max_boundary_displacement_ms",
        )
        for name in finite_fields:
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 20.0 <= self.current_bpm <= 1000.0:
            raise ValueError("current_bpm must be inside the frozen 20-1000 BPM guard")
        if self.previous_bpm is not None and (
            not math.isfinite(self.previous_bpm) or not 20.0 <= self.previous_bpm <= 1000.0
        ):
            raise ValueError("previous_bpm must be None or inside the frozen BPM guard")
        if self.global_downbeat_phase not in (None, 0, 1, 2, 3):
            raise ValueError("global_downbeat_phase must be None or an integer residue in 0..3")
        for name in (
            "next_beat_index",
            "current_section_start_beat",
            "serialized_first_start_beat",
            "real_section_count",
            "alias_switch_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.real_section_count <= 0 or self.real_section_count > 20:
            raise ValueError("real_section_count must be in 1..20")
        if self.alias_switch_count < 0 or self.max_boundary_displacement_ms < 0.0:
            raise ValueError("switch and displacement diagnostics must be non-negative")
        if self.next_beat_time_ms < self.cut_time_ms:
            raise ValueError("next_beat_time_ms must be on or after cut_time_ms")

    @property
    def future_equivalence_key(self) -> tuple[Any, ...]:
        return (
            float.hex(self.cut_time_ms),
            self.next_beat_index,
            float.hex(self.next_beat_time_ms),
            self.current_section_start_beat,
            float.hex(self.current_section_start_time_ms),
            float.hex(self.current_bpm),
            None if self.previous_bpm is None else float.hex(self.previous_bpm),
            self.alias_family,
            self.global_downbeat_phase,
            self.real_section_count,
            self.alias_switch_count,
            float.hex(self.max_boundary_displacement_ms),
            _json_fingerprint(self.open_section_state),
        )

    @property
    def frontier_class_key(self) -> tuple[float, int | None]:
        return (self.alias_family, self.global_downbeat_phase)


@dataclass(frozen=True)
class ClosureCountCandidate:
    count: int
    boundary_beat: int
    boundary_time_ms: float
    displacement_ms: float


@dataclass(frozen=True)
class BlockSchedule:
    core_duration_ms: float
    lookahead_duration_ms: float
    next_cut_time_ms: float
    window_end_ms: float


@dataclass(frozen=True)
class FrontierExport:
    states: tuple[LocalFrontierState, ...]
    dominance_pruned_state_count: int
    width_pruned_state_count: int
    class_coverage: BoundaryPairClassCoverageRecord | None = None


@dataclass(frozen=True)
class LocalFrontierBlockDiagnostics:
    block_index: int
    cut_time_ms: float
    core_start_ms: float
    core_end_ms: float
    lookahead_end_ms: float
    frontier_state_count: int
    bucket_width_max: int
    boundary_candidate_count: int
    tempo_candidate_count: int
    dominance_pruned_state_count: int
    width_pruned_state_count: int
    lookahead_trace_fingerprint: str
    exported_frontier_fingerprint: str


@dataclass(frozen=True)
class BoundaryOwnershipRecord:
    boundary_anchor_id: int
    source_candidate_time_ms: float
    boundary_time_ms: float
    owner_block_index: int
    provisional_block_indexes: tuple[int, ...]
    committed_by_previous_block: bool
    retained_frontier_charge_path_count: int
    selected_traceback_objective_charge_count: int
    objective_charge_count: int


@dataclass(frozen=True)
class LookaheadRecomputeRecord:
    boundary_anchor_id: int
    source_candidate_time_ms: float
    boundary_time_ms: float
    previous_block_index: int
    next_block_index: int
    provisional_trace_fingerprint: str
    recomputed_trace_fingerprint: str
    retained_frontier_charge_path_count: int
    selected_traceback_objective_charge_count: int
    objective_charge_count: int


@dataclass(frozen=True)
class LocalFrontierDiagnostics:
    contract_version: str
    schedule_arm: str
    coverage_start_ms: float
    coverage_end_ms: float
    frame_count: int
    frame_rate_hz: float
    input_signal_sha256: str
    candidate_fingerprint: str
    origin_candidate_count: int
    boundary_candidate_count: int
    tempo_candidate_count: int
    bootstrap_downbeat_phases: tuple[int | None, ...]
    bootstrap_state_count: int
    bootstrap_replay_keys: tuple[tuple[Any, ...], ...]
    block_count: int
    frontier_widths: tuple[int, ...]
    local_bucket_width_max: int
    block_diagnostics: tuple[LocalFrontierBlockDiagnostics, ...]
    boundary_ownership_records: tuple[BoundaryOwnershipRecord, ...]
    lookahead_recompute_records: tuple[LookaheadRecomputeRecord, ...]
    selected_origin_time_ms: float | None
    selected_serialized_first_start_beat: int | None
    first_section_logical_beat_count: int | None
    first_section_serialized_beat_count: int | None
    selected_section_count: int
    final_global_rescore_count: int
    replay_fingerprint: str
    grid_fingerprint: str | None
    deterministic_projection_sha256: str
    fallback_reason: str | None
    fallback_stage: str | None


@dataclass(frozen=True)
class LocalFrontierResult:
    grid: TimingV3Grid | None
    diagnostics: LocalFrontierDiagnostics
    reason: str | None = None
    objective_diagnostics: BoundaryPairObjectiveDiagnostics | None = None

    @property
    def ok(self) -> bool:
        return self.grid is not None


@dataclass(frozen=True)
class LocalFrontierBoundedResult:
    fit_result: LocalFrontierResult
    diagnostics: BoundaryPairBoundedDiagnostics

    @property
    def ok(self) -> bool:
        return self.fit_result.ok

    @property
    def grid(self) -> TimingV3Grid | None:
        return self.fit_result.grid

    @property
    def reason(self) -> str | None:
        return self.fit_result.reason


class _LocalFrontierDiagnosticsMode(str, Enum):
    NONE = "none"
    FULL = "full"
    BOUNDED = "bounded"


@dataclass(frozen=True)
class LocalFrontierFailureClassification:
    reason: str
    stage: str


class LocalFrontierDiagnosticsIntegrityFailure(RuntimeError):
    reason = REASON_DIAGNOSTICS_INTEGRITY_FAILURE
    stage = "diagnostics"

    def __init__(self, reason: str | None = None) -> None:
        if reason is not None and reason != self.reason:
            super().__init__(str(reason))
        else:
            super().__init__(self.reason)

    @property
    def failure_classification(self) -> LocalFrontierFailureClassification:
        return LocalFrontierFailureClassification(
            reason=self.reason,
            stage=self.stage,
        )


def classify_local_frontier_exception(
    exc: BaseException,
) -> LocalFrontierFailureClassification | None:
    if isinstance(exc, LocalFrontierDiagnosticsIntegrityFailure):
        return exc.failure_classification
    return None


@dataclass(frozen=True)
class _OverlapTrace:
    beats: tuple[tuple[int, str, str], ...]
    boundaries: tuple[tuple[int, str, int, str, str, str], ...]
    sha256: str


@dataclass(frozen=True)
class _OverlapLineage:
    record_contract_version: str
    prior_block_index: int
    prior_export_ordinal: int
    future_equivalence_sha256: str
    committed_replay_sha256: str
    lineage_sha256: str
    previous_core_end_ms: float
    previous_lookahead_end_ms: float
    provisional_trace: _OverlapTrace


@dataclass(frozen=True)
class _LookaheadChoice:
    lookahead_cost: float
    provisional_boundaries: tuple["_SelectedBoundary", ...]
    ranked_path: "_Path"


@dataclass(frozen=True)
class _ClosedSection:
    start_beat: int
    end_beat: int
    bpm: float


@dataclass(frozen=True)
class _SelectedBoundary:
    anchor_id: int
    source_time_ms: float
    boundary_beat: int
    boundary_time_ms: float


@dataclass(frozen=True)
class _Path:
    origin_time_ms: float
    serialized_first_start_beat: int
    open_start_beat: int
    open_start_time_ms: float
    current_bpm: float
    previous_bpm: float | None
    global_downbeat_phase: int | None
    closed_sections: tuple[_ClosedSection, ...]
    duration_objective_numerator: float
    transition_objective: float
    real_section_count: int
    alias_switch_count: int
    max_boundary_displacement_ms: float
    replay_key: tuple[Any, ...]
    selected_boundaries: tuple[_SelectedBoundary, ...]
    last_boundary_anchor_id: int
    transition_ledger: tuple[BoundaryPairTransitionLedgerEntry, ...] = ()
    transition_edge_orders: tuple[int, ...] = ()
    overlap_lineage: _OverlapLineage | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    @property
    def committed_objective(self) -> float:
        raise AssertionError("a duration normalizer is required")


@dataclass(frozen=True)
class _WorkingPath:
    path: _Path
    cursor_time_ms: float


class _ResourceCapExceeded(RuntimeError):
    pass


class _DiagnosticsIntegrityFailure(LocalFrontierDiagnosticsIntegrityFailure):
    pass


class _CanonicalJsonListHasher:
    def __init__(self) -> None:
        self._sha256 = hashlib.sha256()
        self._sha256.update(b"[")
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def append(self, item: Any) -> None:
        if self._count:
            self._sha256.update(b",")
        self._sha256.update(_canonical_json_bytes(item))
        self._count += 1

    def hexdigest(self) -> str:
        digest = self._sha256.copy()
        digest.update(b"]")
        return digest.hexdigest()


class _PackedOverlapResiduals:
    """Bounded aggregate residual storage for Exp007 audio-level overlap."""

    def __init__(self) -> None:
        self.residual_ms = array("d")
        self.residual_beats = array("d")
        self._payload_sha256 = _CanonicalJsonListHasher()

    @property
    def count(self) -> int:
        return len(self.residual_ms)

    def extend_tagged(
        self,
        *,
        prior_block_index: int,
        prior_export_ordinal: int,
        residuals: Sequence[tuple[int, float, float]],
    ) -> None:
        _ensure_overlap_residual_capacity(self.count, len(residuals))
        for beat, residual_ms, residual_beats in residuals:
            if not math.isfinite(residual_ms) or not math.isfinite(residual_beats):
                raise _DiagnosticsIntegrityFailure(
                    REASON_DIAGNOSTICS_INTEGRITY_FAILURE
                )
            self.residual_ms.append(float(residual_ms))
            self.residual_beats.append(float(residual_beats))
            self._payload_sha256.append(
                (
                    int(prior_block_index),
                    int(prior_export_ordinal),
                    int(beat),
                    float.hex(float(residual_ms)),
                    float.hex(float(residual_beats)),
                )
            )

    def residual_vector_sha256(self) -> str:
        return self._payload_sha256.hexdigest()

    def release(self) -> None:
        self.residual_ms = array("d")
        self.residual_beats = array("d")


class _LocalScoreContext:
    def __init__(
        self,
        prediction: FrameTimingPrediction,
        candidates: GlobalConstantJumpCandidateSet,
        config: LocalFrontierConfig,
        objective_variant: LocalFrontierObjectiveVariant,
        diagnostics_mode: _LocalFrontierDiagnosticsMode,
    ) -> None:
        self.config = config
        self.objective_variant = objective_variant
        self.diagnostics_mode = _LocalFrontierDiagnosticsMode(diagnostics_mode)
        self.record_objective_ledgers = (
            self.diagnostics_mode is _LocalFrontierDiagnosticsMode.FULL
        )
        self.record_measurement_v2 = (
            self.record_objective_ledgers
            and objective_variant
            is LocalFrontierObjectiveVariant.EXP006_PAIR_CONDITIONED_CHANGE_FLOOR_1_4
        )
        self.record_bounded_diagnostics = (
            self.diagnostics_mode is _LocalFrontierDiagnosticsMode.BOUNDED
            and objective_variant
            is LocalFrontierObjectiveVariant.EXP006_PAIR_CONDITIONED_CHANGE_FLOOR_1_4
        )
        self.record_resource_summaries = (
            (self.record_measurement_v2 or self.record_bounded_diagnostics)
            and objective_variant
            is LocalFrontierObjectiveVariant.EXP006_PAIR_CONDITIONED_CHANGE_FLOOR_1_4
        )
        self.beat_signal = np.asarray(prediction.beat_prob, dtype=np.float64)
        self.downbeat_signal = np.asarray(prediction.downbeat_prob, dtype=np.float64)
        self.frame_rate_hz = float(prediction.frame_rate_hz)
        self.frame_count = int(self.beat_signal.size)
        self.coverage_start_ms = float(candidates.diagnostics.coverage_start_ms)
        self.coverage_end_ms = float(candidates.diagnostics.coverage_end_ms)
        self.duration_ms = self.coverage_end_ms - self.coverage_start_ms
        self.transition_normalizer = max(1.0, self.duration_ms / 60000.0)
        self.frame_times_ms = (
            np.arange(self.frame_count, dtype=np.float64) * (1000.0 / self.frame_rate_hz)
        )
        self.beat_peak_times_ms = np.asarray(
            [peak.time_ms for peak in candidates.beat_peaks], dtype=np.float64
        )
        self.beat_peak_confidences = np.asarray(
            [peak.confidence for peak in candidates.beat_peaks], dtype=np.float64
        )
        self.downbeat_peak_times_ms = np.asarray(
            [peak.time_ms for peak in candidates.downbeat_peaks], dtype=np.float64
        )
        self.downbeat_peak_confidences = np.asarray(
            [peak.confidence for peak in candidates.downbeat_peaks], dtype=np.float64
        )
        self.local_score_cache: dict[tuple[Any, ...], float] = {}
        self.transition_component_cache: dict[
            tuple[Any, ...], BoundaryPairTransitionComponents
        ] = {}
        self.provisional_transition_ledgers: list[
            ProvisionalTransitionLedgerRecord
        ] | None = [] if self.record_objective_ledgers else None
        self.provisional_transition_edge_order_records: list[
            _ProvisionalTransitionOccurrenceRecord
        ] | None = [] if self.record_measurement_v2 else None
        self.actual_scored_edges: list[BoundaryPairScoredEdgeRecord] | None = (
            [] if self.record_measurement_v2 else None
        )
        self.actual_scored_edge_count = 0
        self.block_resource_records: list[BoundaryPairBlockResourceRecord] = []
        self.class_coverage_records: list[BoundaryPairClassCoverageRecord] = []
        self.pending_overlap_lineages: tuple[_OverlapLineage, ...] = ()
        self.overlap_records: list[LookaheadOverlapRecord] = []
        self.overlap_residuals = _PackedOverlapResiduals()
        self.bounded_diagnostics: BoundaryPairBoundedDiagnostics | None = None
        self.current_block_index: int | None = None
        self.current_scoring_stage: str | None = None
        self.candidate_period_pairs_by_anchor = tuple(
            (
                boundary.anchor_id,
                float(boundary.left_period_ms),
                float(boundary.right_period_ms),
            )
            for boundary in candidates.boundary_candidates
        )
        self.row_score_miss_keys: set[tuple[Any, ...]] = set()
        self.block_score_miss_keys: set[tuple[Any, ...]] = set()

    def start_block(self, block_index: int | None = None) -> None:
        self.block_score_miss_keys.clear()
        self.current_block_index = block_index
        self.current_scoring_stage = None

    def enter_scoring_stage(self, *, block_index: int, stage: str) -> None:
        if stage not in ("core", "lookahead"):
            raise ValueError("scoring stage must be core or lookahead")
        self.current_block_index = int(block_index)
        self.current_scoring_stage = stage

    def leave_scoring_stage(self) -> None:
        self.current_scoring_stage = None

    def local_cost(
        self,
        *,
        section_start_time_ms: float,
        section_start_beat: int,
        bpm: float,
        global_downbeat_phase: int | None,
        start_ms: float,
        end_ms: float,
    ) -> float:
        if end_ms <= start_ms:
            return 0.0
        key = (
            float.hex(float(section_start_time_ms)),
            int(section_start_beat),
            float.hex(float(bpm)),
            global_downbeat_phase,
            float.hex(float(start_ms)),
            float.hex(float(end_ms)),
        )
        cached = self.local_score_cache.get(key)
        if cached is not None:
            return cached
        if key not in self.block_score_miss_keys:
            if len(self.block_score_miss_keys) >= self.config.max_section_score_misses_per_block:
                raise _ResourceCapExceeded
            self.block_score_miss_keys.add(key)
        if key not in self.row_score_miss_keys:
            if len(self.row_score_miss_keys) >= self.config.max_section_score_misses_per_audio:
                raise _ResourceCapExceeded
            self.row_score_miss_keys.add(key)

        beat = exp004._pulse_correlation_interval(  # noqa: SLF001
            self.beat_signal,
            self.frame_times_ms,
            tau_ms=section_start_time_ms,
            bpm=bpm,
            start_ms=start_ms,
            end_ms=end_ms,
            pulse_width_ms=GLOBAL_CONSTANT_JUMP_CONSTANTS.pulse_width_ms,
        )
        if not beat.valid:
            cost = math.inf
        else:
            beat_support = 0.5 * (1.0 - beat.correlation)
            peak_cost = exp004._peak_recall_precision_cost_fast(  # noqa: SLF001
                left_anchor_time_ms=section_start_time_ms,
                bpm=bpm,
                start_ms=start_ms,
                end_ms=end_ms,
                beat_peak_times_ms=self.beat_peak_times_ms,
                constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
            )
            downbeat_cost = 0.0
            if global_downbeat_phase is not None:
                downbeat = exp004._downbeat_pulse_correlation_interval(  # noqa: SLF001
                    self.downbeat_signal,
                    self.frame_times_ms,
                    tau_ms=section_start_time_ms,
                    beat_at_tau=section_start_beat,
                    bpm=bpm,
                    global_downbeat_phase=global_downbeat_phase,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    pulse_width_ms=GLOBAL_CONSTANT_JUMP_CONSTANTS.pulse_width_ms,
                )
                if not downbeat.valid:
                    cost = math.inf
                    self.local_score_cache[key] = cost
                    return cost
                downbeat_cost = 0.5 * (1.0 - downbeat.correlation)
            bpm_prior = exp004._bpm_prior_cost(  # noqa: SLF001
                bpm, GLOBAL_CONSTANT_JUMP_CONSTANTS
            )
            cost = (
                beat_support
                + 0.25 * peak_cost
                + 0.20 * downbeat_cost
                + 0.10 * bpm_prior
            )
        self.local_score_cache[key] = float(cost)
        return float(cost)

    def boundary_support_cost(self, boundary_time_ms: float) -> float:
        support = exp004._support_near_time(  # noqa: SLF001
            boundary_time_ms,
            self.beat_peak_times_ms,
            self.beat_peak_confidences,
            tolerance_ms=GLOBAL_CONSTANT_JUMP_CONSTANTS.boundary_support_tolerance_ms,
        )
        if self.downbeat_peak_times_ms.size:
            support = max(
                support,
                exp004._support_near_time(  # noqa: SLF001
                    boundary_time_ms,
                    self.downbeat_peak_times_ms,
                    self.downbeat_peak_confidences,
                    tolerance_ms=GLOBAL_CONSTANT_JUMP_CONSTANTS.boundary_support_tolerance_ms,
                ),
            )
        return float(1.0 - support)

    def transition_components(
        self,
        *,
        boundary: BoundaryCandidate,
        left_bpm: float,
        right_bpm: float,
    ) -> BoundaryPairTransitionComponents:
        # Variant and exact period identity are part of the cache contract.
        # In particular, an Exp005 component or a component for a different
        # period pair can never leak into an Exp006 edge.
        key = self.transition_component_cache_key(
            boundary=boundary,
            left_bpm=left_bpm,
            right_bpm=right_bpm,
        )
        cached = self.transition_component_cache.get(key)
        if cached is not None:
            return cached
        component = boundary_pair_transition_components(
            boundary_anchor_id=boundary.anchor_id,
            source_candidate_time_ms=boundary.time_ms,
            left_period_ms=boundary.left_period_ms,
            right_period_ms=boundary.right_period_ms,
            left_bpm=left_bpm,
            right_bpm=right_bpm,
            boundary_support_cost=self.boundary_support_cost(boundary.time_ms),
            transition_normalizer=self.transition_normalizer,
            objective_variant=self.objective_variant,
        )
        self.transition_component_cache[key] = component
        return component

    def transition_component_cache_key(
        self,
        *,
        boundary: BoundaryCandidate,
        left_bpm: float,
        right_bpm: float,
    ) -> tuple[Any, ...]:
        return (
            self.objective_variant.value,
            boundary.anchor_id,
            float.hex(boundary.time_ms),
            float.hex(boundary.left_period_ms),
            float.hex(boundary.right_period_ms),
            float.hex(left_bpm),
            float.hex(right_bpm),
        )

    def record_actual_scored_edge(
        self,
        *,
        predecessor_replay_key: tuple[Any, ...],
        successor_replay_key: tuple[Any, ...],
        boundary: BoundaryCandidate,
        closure: ClosureCountCandidate,
        right_bpm: float,
        components: BoundaryPairTransitionComponents,
    ) -> int | None:
        if not self.record_resource_summaries:
            return None
        if self.current_block_index is None or self.current_scoring_stage is None:
            raise AssertionError("Exp006 measurement requires an active block stage")
        edge_order = self.actual_scored_edge_count
        self.actual_scored_edge_count += 1
        if not self.record_measurement_v2:
            return None
        if self.actual_scored_edges is None:
            raise AssertionError("full Exp006 edge recording was not initialized")
        cache_key = self.transition_component_cache_key(
            boundary=boundary,
            left_bpm=components.left_bpm,
            right_bpm=right_bpm,
        )
        self.actual_scored_edges.append(
            BoundaryPairScoredEdgeRecord(
                edge_order=edge_order,
                block_index=self.current_block_index,
                stage=self.current_scoring_stage,
                predecessor_replay_key=predecessor_replay_key,
                successor_replay_key=successor_replay_key,
                boundary_anchor_id=boundary.anchor_id,
                source_candidate_time_ms=float(boundary.time_ms),
                boundary_beat=closure.boundary_beat,
                boundary_time_ms=float(closure.boundary_time_ms),
                snapped_beat=closure.boundary_beat,
                snapped_time_ms=float(closure.boundary_time_ms),
                left_period_ms=float(boundary.left_period_ms),
                right_period_ms=float(boundary.right_period_ms),
                left_bpm=float(components.left_bpm),
                right_bpm=float(right_bpm),
                objective_variant=self.objective_variant.value,
                component_cache_key=cache_key,
                components=components,
                normalized_increment=components.normalized_increment,
            )
        )
        return edge_order


def alias_aware_boundary_period_distance(q_bpm: float, qhat_bpm: float) -> float:
    """Return the frozen Exp006 clipped alias-aware log2 distance."""
    _, clipped = _alias_distance_parts(q_bpm, qhat_bpm)
    return clipped


def boundary_pair_transition_components(
    *,
    boundary_anchor_id: int,
    source_candidate_time_ms: float,
    left_period_ms: float,
    right_period_ms: float,
    left_bpm: float,
    right_bpm: float,
    boundary_support_cost: float,
    transition_normalizer: float,
    objective_variant: LocalFrontierObjectiveVariant,
) -> BoundaryPairTransitionComponents:
    """Score one edge under a frozen Exp005 or Exp006 transition variant."""
    try:
        variant = LocalFrontierObjectiveVariant(objective_variant)
    except ValueError as exc:
        raise ValueError(f"unsupported local-frontier objective: {objective_variant!r}") from exc
    floor = (
        EXP006_PAIR_SPARSITY_FLOOR
        if variant
        is LocalFrontierObjectiveVariant.EXP006_PAIR_CONDITIONED_CHANGE_FLOOR_1_4
        else None
    )
    return _boundary_pair_components_with_floor(
        objective_variant=variant.value,
        boundary_anchor_id=boundary_anchor_id,
        source_candidate_time_ms=source_candidate_time_ms,
        left_period_ms=left_period_ms,
        right_period_ms=right_period_ms,
        left_bpm=left_bpm,
        right_bpm=right_bpm,
        boundary_support_cost=boundary_support_cost,
        transition_normalizer=transition_normalizer,
        sparsity_floor=floor,
    )


def replay_terminal_objective(
    ledger: TerminalObjectiveLedger,
    *,
    sparsity_floor: float | None = EXP006_PAIR_SPARSITY_FLOOR,
    boundary_period_pairs_by_anchor: dict[int, tuple[float, float]] | None = None,
) -> TerminalObjectiveReplay:
    """Replay a retained path with only its change potential replaced.

    ``sparsity_floor=None`` is the pinned Exp005 constant-one comparator.
    A mapping can replace exact boundary period pairs for the pre-registered
    pair-scrambled counterfactual without changing paths or candidates.
    """
    if sparsity_floor is not None and (
        not math.isfinite(sparsity_floor) or not 0.0 <= sparsity_floor <= 1.0
    ):
        raise ValueError("sparsity_floor must be None or finite in [0,1]")
    period_pairs = boundary_period_pairs_by_anchor or {}
    transition_objective = 0.0
    for entry in ledger.transition_entries:
        original = entry.components
        left_period_ms, right_period_ms = period_pairs.get(
            original.boundary_anchor_id,
            (original.left_period_ms, original.right_period_ms),
        )
        replayed = _boundary_pair_components_with_floor(
            objective_variant=(
                LocalFrontierObjectiveVariant.EXP005_CONSTANT_CHANGE.value
                if sparsity_floor is None
                else f"counterfactual_rho_{float.hex(sparsity_floor)}"
            ),
            boundary_anchor_id=original.boundary_anchor_id,
            source_candidate_time_ms=original.source_candidate_time_ms,
            left_period_ms=left_period_ms,
            right_period_ms=right_period_ms,
            left_bpm=original.left_bpm,
            right_bpm=original.right_bpm,
            boundary_support_cost=original.boundary_support_cost,
            transition_normalizer=original.transition_normalizer,
            sparsity_floor=sparsity_floor,
        )
        transition_objective += replayed.normalized_increment
    terminal = (
        ledger.normalized_duration_objective
        + transition_objective
        + ledger.normalized_tail_prior
    )
    return TerminalObjectiveReplay(
        transition_objective=float(transition_objective),
        terminal_objective=float(terminal),
    )


def replay_terminal_objective_with_scrambled_pairs(
    ledger: TerminalObjectiveLedger,
    *,
    boundary_candidates: Sequence[BoundaryCandidate] | None = None,
    sparsity_floor: float = EXP006_PAIR_SPARSITY_FLOOR,
) -> TerminalObjectiveReplay:
    """Replay a path under the frozen global two-candidate pair swap.

    The mapping is built from the immutable candidate set, not from the path's
    selected entries.  Thus a shortcut using only anchor 0 still receives the
    period pair originally owned by anchor 1, as required by the Exp006
    counterfactual.
    """
    mapping = (
        scrambled_boundary_period_pairs(boundary_candidates)
        if boundary_candidates is not None
        else _scrambled_boundary_period_pair_records(
            ledger.candidate_period_pairs_by_anchor
        )
    )
    return replay_terminal_objective(
        ledger,
        sparsity_floor=sparsity_floor,
        boundary_period_pairs_by_anchor=mapping,
    )


def scrambled_boundary_period_pairs(
    boundary_candidates: Sequence[BoundaryCandidate],
) -> dict[int, tuple[float, float]]:
    """Swap the exact period pairs of the two frozen kill-fixture candidates."""
    records = tuple(
        (
            candidate.anchor_id,
            candidate.left_period_ms,
            candidate.right_period_ms,
        )
        for candidate in boundary_candidates
    )
    return _scrambled_boundary_period_pair_records(records)


def _scrambled_boundary_period_pair_records(
    records: Sequence[tuple[int, float, float]],
) -> dict[int, tuple[float, float]]:
    if len(records) != 2:
        raise ValueError("the frozen pair-scrambled control requires exactly two candidates")
    left, right = sorted(records, key=lambda item: item[0])
    if left[0] == right[0]:
        raise ValueError("boundary candidate anchor IDs must be unique")
    for _, left_period_ms, right_period_ms in (left, right):
        if (
            not math.isfinite(left_period_ms)
            or left_period_ms <= 0.0
            or not math.isfinite(right_period_ms)
            or right_period_ms <= 0.0
        ):
            raise ValueError("boundary periods must be positive and finite")
    return {
        left[0]: (right[1], right[2]),
        right[0]: (left[1], left[2]),
    }


def _alias_distance_parts(q_bpm: float, qhat_bpm: float) -> tuple[float, float]:
    if (
        not math.isfinite(q_bpm)
        or q_bpm <= 0.0
        or not math.isfinite(qhat_bpm)
        or qhat_bpm <= 0.0
    ):
        raise ValueError("alias distance BPMs must be positive and finite")
    distances = tuple(
        abs(math.log2(q_bpm / (multiplier * qhat_bpm)))
        for multiplier in BOUNDARY_PAIR_ALIAS_MULTIPLIERS
    )
    minimum = min(distances)
    return float(minimum), float(min(1.0, minimum))


def _pair_cost_from_periods(
    *,
    left_period_ms: float,
    right_period_ms: float,
    left_bpm: float,
    right_bpm: float,
) -> tuple[float, float, float, float, float, float, float]:
    if (
        not math.isfinite(left_period_ms)
        or left_period_ms <= 0.0
        or not math.isfinite(right_period_ms)
        or right_period_ms <= 0.0
    ):
        raise ValueError("boundary periods must be positive and finite")
    # Keep the binary64 evaluation order identical to the frozen card.
    qhat_left = 60000.0 / left_period_ms
    qhat_right = 60000.0 / right_period_ms
    left_min, left_distance = _alias_distance_parts(left_bpm, qhat_left)
    right_min, right_distance = _alias_distance_parts(right_bpm, qhat_right)
    pair_cost = min(1.0, left_distance + right_distance)
    return (
        float(qhat_left),
        float(qhat_right),
        left_min,
        right_min,
        left_distance,
        right_distance,
        float(pair_cost),
    )


def _boundary_pair_components_with_floor(
    *,
    objective_variant: str,
    boundary_anchor_id: int,
    source_candidate_time_ms: float,
    left_period_ms: float,
    right_period_ms: float,
    left_bpm: float,
    right_bpm: float,
    boundary_support_cost: float,
    transition_normalizer: float,
    sparsity_floor: float | None,
) -> BoundaryPairTransitionComponents:
    finite_positive = {
        "left_period_ms": left_period_ms,
        "right_period_ms": right_period_ms,
        "left_bpm": left_bpm,
        "right_bpm": right_bpm,
        "transition_normalizer": transition_normalizer,
    }
    for name, value in finite_positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(source_candidate_time_ms):
        raise ValueError("source_candidate_time_ms must be finite")
    if not math.isfinite(boundary_support_cost):
        raise ValueError("boundary_support_cost must be finite")
    if sparsity_floor is not None and (
        not math.isfinite(sparsity_floor) or not 0.0 <= sparsity_floor <= 1.0
    ):
        raise ValueError("sparsity_floor must be None or finite in [0,1]")

    (
        qhat_left,
        qhat_right,
        left_min,
        right_min,
        left_distance,
        right_distance,
        pair_cost,
    ) = _pair_cost_from_periods(
        left_period_ms=left_period_ms,
        right_period_ms=right_period_ms,
        left_bpm=left_bpm,
        right_bpm=right_bpm,
    )
    if sparsity_floor is None:
        change_cost = 1.0
    else:
        change_cost = sparsity_floor + (1.0 - sparsity_floor) * pair_cost
    alias_switch = exp004._alias_switch_cost(  # noqa: SLF001
        left_bpm,
        right_bpm,
        GLOBAL_CONSTANT_JUMP_CONSTANTS,
    )
    jump_size = min(1.0, abs(math.log2(right_bpm / left_bpm)))
    if sparsity_floor is None:
        # Preserve the exact Exp005 addition path (including its first 0.18
        # literal) rather than relying on an algebraically equal refactor.
        raw_transition = (
            0.18
            + 0.12 * alias_switch
            + 0.10 * jump_size
            + 0.15 * boundary_support_cost
        )
    else:
        raw_transition = (
            0.18 * change_cost
            + 0.12 * alias_switch
            + 0.10 * jump_size
            + 0.15 * boundary_support_cost
        )
    normalized = raw_transition / transition_normalizer
    return BoundaryPairTransitionComponents(
        objective_variant=objective_variant,
        boundary_anchor_id=boundary_anchor_id,
        source_candidate_time_ms=float(source_candidate_time_ms),
        left_period_ms=float(left_period_ms),
        right_period_ms=float(right_period_ms),
        left_bpm=float(left_bpm),
        right_bpm=float(right_bpm),
        qhat_left_bpm=float(qhat_left),
        qhat_right_bpm=float(qhat_right),
        alias_min_left_unclipped=left_min,
        alias_min_right_unclipped=right_min,
        alias_distance_left=left_distance,
        alias_distance_right=right_distance,
        pair_cost=float(pair_cost),
        sparsity_floor=sparsity_floor,
        change_cost=float(change_cost),
        alias_switch_cost=float(alias_switch),
        jump_size_cost=float(jump_size),
        boundary_support_cost=float(boundary_support_cost),
        raw_transition_cost=float(raw_transition),
        transition_normalizer=float(transition_normalizer),
        normalized_increment=float(normalized),
    )


def lattice_time_ms(beat: int, *, origin_time_ms: float, bpm: float) -> float:
    _validate_lattice_inputs(origin_time_ms=origin_time_ms, bpm=bpm)
    if isinstance(beat, bool) or not isinstance(beat, int):
        raise ValueError("beat must be an integer")
    value = float(origin_time_ms + beat * (60000.0 / bpm))
    if not math.isfinite(value):
        raise ValueError("derived lattice time must be finite")
    return value


def next_beat_on_or_after(time_ms: float, *, origin_time_ms: float, bpm: float) -> int:
    _validate_lattice_inputs(origin_time_ms=origin_time_ms, bpm=bpm)
    if not math.isfinite(time_ms):
        raise ValueError("time_ms must be finite")
    period_ms = 60000.0 / bpm
    estimate = (time_ms - origin_time_ms) / period_ms
    if not math.isfinite(estimate):
        raise ValueError("derived beat estimate must be finite")
    beat = math.floor(estimate)
    while lattice_time_ms(beat, origin_time_ms=origin_time_ms, bpm=bpm) < time_ms:
        beat += 1
    while lattice_time_ms(beat - 1, origin_time_ms=origin_time_ms, bpm=bpm) >= time_ms:
        beat -= 1
    return int(beat)


def previous_beat_on_or_before(time_ms: float, *, origin_time_ms: float, bpm: float) -> int:
    _validate_lattice_inputs(origin_time_ms=origin_time_ms, bpm=bpm)
    if not math.isfinite(time_ms):
        raise ValueError("time_ms must be finite")
    period_ms = 60000.0 / bpm
    estimate = (time_ms - origin_time_ms) / period_ms
    if not math.isfinite(estimate):
        raise ValueError("derived beat estimate must be finite")
    beat = math.floor(estimate)
    while lattice_time_ms(beat, origin_time_ms=origin_time_ms, bpm=bpm) > time_ms:
        beat -= 1
    while lattice_time_ms(beat + 1, origin_time_ms=origin_time_ms, bpm=bpm) <= time_ms:
        beat += 1
    return int(beat)


def closure_count_candidates(
    state: LocalFrontierState,
    *,
    boundary_time_ms: float,
) -> tuple[ClosureCountCandidate, ...]:
    if not math.isfinite(boundary_time_ms):
        raise ValueError("boundary_time_ms must be finite")
    x = (
        (boundary_time_ms - state.current_section_start_time_ms)
        * state.current_bpm
        / 60000.0
    )
    if not math.isfinite(x):
        raise ValueError("derived closure count must be finite")
    nominal = math.floor(x + 0.5)
    ordered_counts: list[int] = []
    for count in (nominal, nominal - 1, nominal + 1):
        if count > 0 and count not in ordered_counts:
            ordered_counts.append(int(count))
    return tuple(
        ClosureCountCandidate(
            count=count,
            boundary_beat=state.current_section_start_beat + count,
            boundary_time_ms=(
                state.current_section_start_time_ms
                + count * (60000.0 / state.current_bpm)
            ),
            displacement_ms=abs(
                state.current_section_start_time_ms
                + count * (60000.0 / state.current_bpm)
                - boundary_time_ms
            ),
        )
        for count in ordered_counts
    )


def block_schedule_for_cut(
    cut_time_ms: float,
    *,
    coverage_end_ms: float,
    schedule: LocalFrontierScheduleArm,
    q_ref_bpm: float | None = None,
) -> BlockSchedule:
    if not math.isfinite(cut_time_ms) or not math.isfinite(coverage_end_ms):
        raise ValueError("cut and coverage times must be finite")
    if coverage_end_ms <= cut_time_ms:
        raise ValueError("coverage_end_ms must be after cut_time_ms")
    arm = LocalFrontierScheduleArm(schedule)
    if arm is LocalFrontierScheduleArm.S30:
        core_ms, lookahead_ms = 30_000.0, 10_000.0
    elif arm is LocalFrontierScheduleArm.S60:
        core_ms, lookahead_ms = 60_000.0, 20_000.0
    elif arm is LocalFrontierScheduleArm.S90:
        core_ms, lookahead_ms = 90_000.0, 30_000.0
    elif q_ref_bpm is None:
        core_ms, lookahead_ms = 90_000.0, 30_000.0
    else:
        if not math.isfinite(q_ref_bpm) or not 20.0 <= q_ref_bpm <= 1000.0:
            raise ValueError("q_ref_bpm must satisfy the frozen BPM guard")
        core_ms = min(90_000.0, max(30_000.0, 64.0 * 60000.0 / q_ref_bpm))
        lookahead_ms = min(30_000.0, max(10_000.0, 16.0 * 60000.0 / q_ref_bpm))
    next_cut = min(coverage_end_ms, cut_time_ms + core_ms)
    window_end = min(coverage_end_ms, next_cut + lookahead_ms)
    if next_cut <= cut_time_ms:
        raise ValueError("schedule produced a non-advancing cut")
    return BlockSchedule(
        core_duration_ms=float(next_cut - cut_time_ms),
        lookahead_duration_ms=float(window_end - next_cut),
        next_cut_time_ms=float(next_cut),
        window_end_ms=float(window_end),
    )


def restrict_timing_prediction(prediction: FrameTimingPrediction) -> FrameTimingPrediction:
    beat_prob = np.asarray(prediction.beat_prob)
    downbeat_prob = np.asarray(prediction.downbeat_prob)
    beat_prob.setflags(write=False)
    downbeat_prob.setflags(write=False)
    restricted = FrameTimingPrediction(
        provider="cached-beatthis-restricted",
        checkpoint_path=None,
        source_path=None,
        beat_prob=beat_prob,
        downbeat_prob=downbeat_prob,
        frame_rate_hz=prediction.frame_rate_hz,
    )
    restricted.beat_prob.setflags(write=False)
    restricted.downbeat_prob.setflags(write=False)
    validate_restricted_prediction(restricted)
    return restricted


def validate_restricted_prediction(prediction: FrameTimingPrediction) -> None:
    if prediction.source_path is not None:
        raise ValueError("restricted prediction source_path must be None")
    if prediction.checkpoint_path is not None:
        raise ValueError("restricted prediction checkpoint_path must be None")
    if prediction.beat_prob.flags.writeable or prediction.downbeat_prob.flags.writeable:
        raise ValueError("restricted prediction arrays must be read-only")
    if prediction.beat_prob.shape != prediction.downbeat_prob.shape or prediction.beat_prob.ndim != 1:
        raise ValueError("restricted prediction arrays must be matching 1-D vectors")
    if prediction.frame_count <= 0:
        raise ValueError("restricted prediction must contain frames")


def build_tempo_shortlist(
    frontier: Sequence[LocalFrontierState],
    boundary_candidates: Sequence[BoundaryCandidate],
    global_tempos: Sequence[TempoCandidate],
) -> tuple[float, ...]:
    result: list[float] = []
    seen: set[float] = set()

    def append(value: float) -> bool:
        value = float(value)
        if not math.isfinite(value) or not 20.0 <= value <= 1000.0:
            return False
        key = round(value, 6)
        if key in seen:
            return False
        seen.add(key)
        result.append(value)
        return True

    incoming_added = 0
    for state in sorted(frontier, key=_state_order_key):
        if append(state.current_bpm):
            incoming_added += 1
            if incoming_added == 16:
                break

    boundary_added = 0
    for boundary in boundary_candidates:
        for period_ms in (boundary.right_period_ms, boundary.left_period_ms):
            if math.isfinite(period_ms) and period_ms > 0.0 and append(60000.0 / period_ms):
                boundary_added += 1
                if boundary_added == 16:
                    break
        if boundary_added == 16:
            break

    global_added = 0
    for candidate in global_tempos:
        if append(candidate.bpm):
            global_added += 1
            if global_added == 16:
                break

    base_bpms = tuple(result)
    for bpm in base_bpms:
        for multiplier in GLOBAL_CONSTANT_JUMP_CONSTANTS.alias_multipliers:
            append(round(bpm * multiplier, 6))
            if len(result) == 64:
                return tuple(result)
    return tuple(result[:64])


def export_frontier(
    states: Sequence[LocalFrontierState],
    *,
    cut_time_ms: float,
    max_states: int = 16,
    capture_class_coverage: bool = False,
    block_index: int | None = None,
) -> FrontierExport:
    if not math.isfinite(cut_time_ms):
        raise ValueError("cut_time_ms must be finite")
    if max_states != 16:
        raise ValueError("the exported frontier width is frozen at 16")
    best_by_equivalence: dict[tuple[Any, ...], LocalFrontierState] = {}
    dominance_pruned = 0
    for state in states:
        if state.cut_time_ms != cut_time_ms:
            raise ValueError("all exported states must share the exact cut_time_ms")
        key = state.future_equivalence_key
        prior = best_by_equivalence.get(key)
        if prior is None:
            best_by_equivalence[key] = state
        elif _state_order_key(state) < _state_order_key(prior):
            best_by_equivalence[key] = state
            dominance_pruned += 1
        else:
            dominance_pruned += 1

    ordered = sorted(best_by_equivalence.values(), key=_state_order_key)
    best_by_class: dict[tuple[float, int | None], LocalFrontierState] = {}
    for state in ordered:
        best_by_class.setdefault(state.frontier_class_key, state)
    representatives = sorted(best_by_class.values(), key=_state_order_key)[:max_states]
    selected_keys = {state.future_equivalence_key for state in representatives}
    selected = list(representatives)
    for state in ordered:
        if len(selected) == max_states:
            break
        if state.future_equivalence_key not in selected_keys:
            selected.append(state)
            selected_keys.add(state.future_equivalence_key)
    selected.sort(key=_state_order_key)
    width_pruned = max(0, len(ordered) - len(selected))
    class_coverage = (
        BoundaryPairClassCoverageRecord(
            block_index=-1 if block_index is None else int(block_index),
            cut_time_ms=float(cut_time_ms),
            input_state_count=len(states),
            input_unique_class_keys=_sorted_unique_frontier_class_keys(states),
            post_future_equivalence_state_count=len(best_by_equivalence),
            post_future_equivalence_unique_class_keys=(
                _sorted_unique_frontier_class_keys(best_by_equivalence.values())
            ),
            reserved_state_count=len(representatives),
            reserved_unique_class_keys=_sorted_unique_frontier_class_keys(
                representatives
            ),
            final_state_count=len(selected),
            final_unique_class_keys=_sorted_unique_frontier_class_keys(selected),
        )
        if capture_class_coverage
        else None
    )
    return FrontierExport(
        states=tuple(selected),
        dominance_pruned_state_count=dominance_pruned,
        width_pruned_state_count=width_pruned,
        class_coverage=class_coverage,
    )


def _sorted_unique_frontier_class_keys(
    states: Iterable[LocalFrontierState],
) -> tuple[tuple[float, int | None], ...]:
    return tuple(
        sorted(
            {
                (float(state.alias_family), state.global_downbeat_phase)
                for state in states
            },
            key=lambda item: (
                float(item[0]),
                -1 if item[1] is None else int(item[1]),
            ),
        )
    )


def fit_local_frontier_constant_jump(
    prediction: FrameTimingPrediction,
    *,
    config: LocalFrontierConfig = LocalFrontierConfig(),
    candidate_set: GlobalConstantJumpCandidateSet | None = None,
) -> LocalFrontierResult:
    """Fit the pre-registered Exp005 bounded fixed-lag constant/jump grid."""
    return _fit_local_frontier_objective(
        prediction,
        config=config,
        candidate_set=candidate_set,
        objective_variant=LocalFrontierObjectiveVariant.EXP005_CONSTANT_CHANGE,
        record_objective_ledgers=False,
    )


def fit_local_frontier_boundary_pair_transition(
    prediction: FrameTimingPrediction,
    *,
    config: LocalFrontierConfig = LocalFrontierConfig(),
    candidate_set: GlobalConstantJumpCandidateSet | None = None,
) -> LocalFrontierResult:
    """Fit the frozen Exp006 pair-conditioned transition objective."""
    return _fit_local_frontier_objective(
        prediction,
        config=config,
        candidate_set=candidate_set,
        objective_variant=(
            LocalFrontierObjectiveVariant.EXP006_PAIR_CONDITIONED_CHANGE_FLOOR_1_4
        ),
        record_objective_ledgers=True,
    )


def fit_local_frontier_boundary_pair_transition_bounded(
    prediction: FrameTimingPrediction,
    *,
    config: LocalFrontierConfig = LocalFrontierConfig(),
    candidate_set: GlobalConstantJumpCandidateSet | None = None,
) -> LocalFrontierBoundedResult:
    """Fit Exp006 with bounded diagnostics that avoid full transition ledgers."""
    fit_result, diagnostics = _fit_local_frontier_objective_run(
        prediction,
        config=config,
        candidate_set=candidate_set,
        objective_variant=(
            LocalFrontierObjectiveVariant.EXP006_PAIR_CONDITIONED_CHANGE_FLOOR_1_4
        ),
        diagnostics_mode=_LocalFrontierDiagnosticsMode.BOUNDED,
    )
    if diagnostics is None:
        raise AssertionError("bounded diagnostics were not produced")
    return LocalFrontierBoundedResult(
        fit_result=fit_result,
        diagnostics=diagnostics,
    )


def fit_local_frontier_objective_variant(
    prediction: FrameTimingPrediction,
    *,
    objective_variant: LocalFrontierObjectiveVariant,
    config: LocalFrontierConfig = LocalFrontierConfig(),
    candidate_set: GlobalConstantJumpCandidateSet | None = None,
) -> LocalFrontierResult:
    """Run an explicit frozen objective and expose its reconstruction ledger."""
    try:
        variant = LocalFrontierObjectiveVariant(objective_variant)
    except ValueError as exc:
        raise ValueError(f"unsupported local-frontier objective: {objective_variant!r}") from exc
    return _fit_local_frontier_objective(
        prediction,
        config=config,
        candidate_set=candidate_set,
        objective_variant=variant,
        record_objective_ledgers=True,
    )


def _fit_local_frontier_objective(
    prediction: FrameTimingPrediction,
    *,
    config: LocalFrontierConfig,
    candidate_set: GlobalConstantJumpCandidateSet | None,
    objective_variant: LocalFrontierObjectiveVariant,
    record_objective_ledgers: bool,
) -> LocalFrontierResult:
    result, _ = _fit_local_frontier_objective_run(
        prediction,
        config=config,
        candidate_set=candidate_set,
        objective_variant=objective_variant,
        diagnostics_mode=(
            _LocalFrontierDiagnosticsMode.FULL
            if record_objective_ledgers
            else _LocalFrontierDiagnosticsMode.NONE
        ),
    )
    return result


def _fit_local_frontier_objective_run(
    prediction: FrameTimingPrediction,
    *,
    config: LocalFrontierConfig,
    candidate_set: GlobalConstantJumpCandidateSet | None,
    objective_variant: LocalFrontierObjectiveVariant,
    diagnostics_mode: _LocalFrontierDiagnosticsMode,
) -> tuple[LocalFrontierResult, BoundaryPairBoundedDiagnostics | None]:
    if not isinstance(config, LocalFrontierConfig):
        raise TypeError("config must be a LocalFrontierConfig")
    mode = _LocalFrontierDiagnosticsMode(diagnostics_mode)
    # The inference entry point accepts an unrestricted loader object, but the
    # core must never inspect its provenance or metadata.  Construct the
    # array-only view unconditionally; in particular, do not branch on
    # source_path/checkpoint_path before crossing this boundary.
    restricted = restrict_timing_prediction(prediction)
    validate_restricted_prediction(restricted)
    candidates = (
        exp004.extract_global_constant_jump_candidates(restricted)
        if candidate_set is None
        else exp004._validated_candidate_set_for_prediction(candidate_set, restricted)  # noqa: SLF001
    )
    context = _LocalScoreContext(
        restricted,
        candidates,
        config,
        objective_variant,
        mode,
    )
    phases: tuple[int | None, ...] = (
        (None,)
        if not _has_centered_downbeat_signal(context.downbeat_signal)
        else (0, 1, 2, 3)
    )
    bootstrap_replay_keys = tuple(
        (origin_index, float.hex(origin.time_ms), float.hex(origin.bpm), phase)
        for origin_index, origin in enumerate(candidates.origin_candidates)
        for phase in phases
    )
    if not candidates.origin_candidates:
        result = _failure_result(
            config=config,
            candidates=candidates,
            context=context,
            phases=phases,
            bootstrap_replay_keys=bootstrap_replay_keys,
            reason=REASON_NO_ORIGIN_CANDIDATE,
            stage="bootstrap",
        )
        return result, context.bounded_diagnostics
    try:
        result = _fit_with_candidates(
            config=config,
            candidates=candidates,
            context=context,
            phases=phases,
            bootstrap_replay_keys=bootstrap_replay_keys,
        )
        return result, context.bounded_diagnostics
    except _ResourceCapExceeded:
        result = _failure_result(
            config=config,
            candidates=candidates,
            context=context,
            phases=phases,
            bootstrap_replay_keys=bootstrap_replay_keys,
            reason=REASON_RESOURCE_CAP_EXCEEDED,
            stage="local_graph",
        )
        return result, context.bounded_diagnostics


def _fit_with_candidates(
    *,
    config: LocalFrontierConfig,
    candidates: GlobalConstantJumpCandidateSet,
    context: _LocalScoreContext,
    phases: tuple[int | None, ...],
    bootstrap_replay_keys: tuple[tuple[Any, ...], ...],
) -> LocalFrontierResult:
    paths = _bootstrap_paths(candidates, context=context, phases=phases)
    if not paths:
        return _failure_result(
            config=config,
            candidates=candidates,
            context=context,
            phases=phases,
            bootstrap_replay_keys=bootstrap_replay_keys,
            reason=REASON_NO_LOCAL_FRONTIER_PATH,
            stage="bootstrap",
        )

    cut = context.coverage_start_ms
    block_diagnostics: list[LocalFrontierBlockDiagnostics] = []
    ownership_by_anchor: dict[int, BoundaryOwnershipRecord] = {}
    recompute_records: list[LookaheadRecomputeRecord] = []
    prior_provisional: dict[int, tuple[int, str]] = {}
    frontier_widths: list[int] = []
    local_bucket_width_max = min(len(paths), config.local_beam_width)
    while cut < context.coverage_end_ms:
        if len(block_diagnostics) >= config.max_blocks:
            raise _ResourceCapExceeded
        block_index = len(block_diagnostics)
        context.start_block(block_index)
        incoming_path_count = len(paths)
        block_score_miss_count_before = len(context.block_score_miss_keys)
        row_score_miss_count_before = len(context.row_score_miss_keys)
        transition_cache_count_before = len(context.transition_component_cache)
        scored_edge_count_before = context.actual_scored_edge_count
        q_ref = paths[0].current_bpm if config.schedule_arm is LocalFrontierScheduleArm.S64 and cut > 0.0 else None
        schedule = block_schedule_for_cut(
            cut,
            coverage_end_ms=context.coverage_end_ms,
            schedule=config.schedule_arm,
            q_ref_bpm=q_ref,
        )
        core_end = schedule.next_cut_time_ms
        window_end = schedule.window_end_ms
        # Candidate time is evidence, not the mathematical boundary.  Include
        # the maximum frozen snap halo here, then let _advance_paths_interval
        # assign a proposal solely by its derived t_jump.
        snap_halo_ms = GLOBAL_CONSTANT_JUMP_CONSTANTS.boundary_support_tolerance_ms
        local_boundaries = tuple(
            boundary
            for boundary in candidates.boundary_candidates
            if cut - snap_halo_ms <= boundary.time_ms <= window_end + snap_halo_ms
        )[: config.max_boundary_candidates_per_block]
        incoming_states = tuple(
            _state_from_path(path, cut_time_ms=cut, context=context) for path in paths
        )
        tempos = build_tempo_shortlist(incoming_states, local_boundaries, candidates.tempo_candidates)

        context.enter_scoring_stage(block_index=block_index, stage="core")
        try:
            committed_candidates = list(
                _advance_paths_interval(
                    paths,
                    start_ms=cut,
                    end_ms=core_end,
                    boundaries=local_boundaries,
                    tempos=tempos,
                    context=context,
                )
            )
        finally:
            context.leave_scoring_stage()
        raw_committed_path_count = len(committed_candidates)
        lookahead_by_replay: dict[tuple[Any, ...], _LookaheadChoice] = {}
        committed_candidates = list(_dominant_paths_by_future(committed_candidates, context))
        dominant_committed_path_count = len(committed_candidates)
        lookahead_call_count = 0
        lookahead_successor_count = 0
        for committed in committed_candidates:
            lookahead_call_count += 1
            context.enter_scoring_stage(block_index=block_index, stage="lookahead")
            try:
                lookahead_paths = _advance_paths_interval(
                    (committed,),
                    start_ms=core_end,
                    end_ms=window_end,
                    boundaries=local_boundaries,
                    tempos=tempos,
                    context=context,
                )
            finally:
                context.leave_scoring_stage()
            lookahead_successor_count += len(lookahead_paths)
            if lookahead_paths:
                ranked = min(
                    lookahead_paths,
                    key=lambda candidate: _path_order_key(candidate, context),
                )
                lookahead_cost = _path_objective(ranked, context)
                provisional = ranked.selected_boundaries[len(committed.selected_boundaries) :]
            else:
                lookahead_cost = _path_objective(committed, context)
                provisional = ()
                ranked = committed
            if context.record_objective_ledgers:
                if context.provisional_transition_ledgers is None:
                    raise AssertionError("full transition ledgers were not initialized")
                context.provisional_transition_ledgers.append(
                    ProvisionalTransitionLedgerRecord(
                        block_index=block_index,
                        committed_replay_key=committed.replay_key,
                        ranked_lookahead_replay_key=ranked.replay_key,
                        transition_entries=ranked.transition_ledger[
                            len(committed.transition_ledger) :
                        ],
                    )
                )
            if context.record_measurement_v2:
                if context.provisional_transition_edge_order_records is None:
                    raise AssertionError("full occurrence ledgers were not initialized")
                context.provisional_transition_edge_order_records.append(
                    _ProvisionalTransitionOccurrenceRecord(
                        block_index=block_index,
                        committed_replay_key=committed.replay_key,
                        ranked_lookahead_replay_key=ranked.replay_key,
                        edge_orders=ranked.transition_edge_orders[
                            len(committed.transition_edge_orders) :
                        ],
                    )
                )
            ranking_key = committed.replay_key
            prior = lookahead_by_replay.get(ranking_key)
            if prior is None or (lookahead_cost, tuple(_boundary_replay_item(item) for item in provisional)) < (
                prior.lookahead_cost,
                tuple(_boundary_replay_item(item) for item in prior.provisional_boundaries),
            ):
                lookahead_by_replay[ranking_key] = _LookaheadChoice(
                    lookahead_cost=lookahead_cost,
                    provisional_boundaries=provisional,
                    ranked_path=ranked,
                )

        if not committed_candidates:
            return _failure_result(
                config=config,
                candidates=candidates,
                context=context,
                phases=phases,
                bootstrap_replay_keys=bootstrap_replay_keys,
                reason=REASON_NO_LOCAL_FRONTIER_PATH,
                stage="local_graph",
                block_diagnostics=tuple(block_diagnostics),
            )

        ranked_paths = []
        for path in committed_candidates:
            lookahead_choice = lookahead_by_replay[path.replay_key]
            ranked_paths.append(
                (
                    replace(path, replay_key=path.replay_key),
                    lookahead_choice.lookahead_cost,
                    lookahead_choice.provisional_boundaries,
                    lookahead_choice.ranked_path,
                )
            )
        frontier_states = tuple(
            _state_from_path(path, cut_time_ms=core_end, context=context, rank_objective=rank)
            for path, rank, _, _ in ranked_paths
        )
        pre_export_state_count = len(frontier_states)
        exported = export_frontier(
            frontier_states,
            cut_time_ms=core_end,
            capture_class_coverage=context.record_resource_summaries,
            block_index=block_index,
        )
        if context.record_resource_summaries and exported.class_coverage is not None:
            context.class_coverage_records.append(exported.class_coverage)
        path_by_replay = {
            path.replay_key: (path, provisional, ranked)
            for path, _, provisional, ranked in ranked_paths
        }
        paths = tuple(
            path_by_replay[state.deterministic_replay_key][0] for state in exported.states
        )
        if context.record_bounded_diagnostics:
            _record_pending_overlap_at_export(
                context=context,
                exported_paths=paths,
                next_block_index=block_index,
                next_core_end_ms=core_end,
            )
            paths = _paths_with_new_overlap_lineage(
                context=context,
                exported_states=exported.states,
                path_by_replay=path_by_replay,
                block_index=block_index,
                core_end_ms=core_end,
                lookahead_end_ms=window_end,
            )
        provisional_by_selected = {
            boundary.anchor_id: boundary
            for state in exported.states
            for boundary in path_by_replay[state.deterministic_replay_key][1]
        }
        lookahead_trace = _json_fingerprint(
            tuple(
                _boundary_replay_item(boundary)
                for boundary in sorted(provisional_by_selected.values(), key=_boundary_replay_item)
            )
        )
        frontier_fp = _json_fingerprint(tuple(state.future_equivalence_key for state in exported.states))
        if context.record_resource_summaries:
            context.block_resource_records.append(
                BoundaryPairBlockResourceRecord(
                    block_index=block_index,
                    core_start_ms=float(cut),
                    core_end_ms=float(core_end),
                    lookahead_end_ms=float(window_end),
                    incoming_path_count=incoming_path_count,
                    raw_committed_path_count=raw_committed_path_count,
                    dominant_committed_path_count=dominant_committed_path_count,
                    lookahead_call_count=lookahead_call_count,
                    lookahead_successor_count=lookahead_successor_count,
                    pre_export_state_count=pre_export_state_count,
                    exported_state_count=len(exported.states),
                    block_score_miss_count_before=block_score_miss_count_before,
                    block_score_miss_count_after=len(context.block_score_miss_keys),
                    row_score_miss_count_before=row_score_miss_count_before,
                    row_score_miss_count_after=len(context.row_score_miss_keys),
                    transition_component_cache_count_before=(
                        transition_cache_count_before
                    ),
                    transition_component_cache_count_after=(
                        len(context.transition_component_cache)
                    ),
                    scored_edge_count_before=scored_edge_count_before,
                    scored_edge_count_after=context.actual_scored_edge_count,
                    dominance_pruned_state_count=(
                        exported.dominance_pruned_state_count
                    ),
                    width_pruned_state_count=exported.width_pruned_state_count,
                    exported_frontier_width_cap=config.exported_frontier_width,
                    local_beam_width_cap=config.local_beam_width,
                    max_boundary_candidates_per_block_cap=(
                        config.max_boundary_candidates_per_block
                    ),
                    max_tempo_candidates_per_block_cap=(
                        config.max_tempo_candidates_per_block
                    ),
                    max_blocks_cap=config.max_blocks,
                    max_sections_cap=config.max_sections,
                    max_section_score_misses_per_block_cap=(
                        config.max_section_score_misses_per_block
                    ),
                    max_section_score_misses_per_audio_cap=(
                        config.max_section_score_misses_per_audio
                    ),
                )
            )
        block_diagnostics.append(
            LocalFrontierBlockDiagnostics(
                block_index=block_index,
                cut_time_ms=cut,
                core_start_ms=cut,
                core_end_ms=core_end,
                lookahead_end_ms=window_end,
                frontier_state_count=len(exported.states),
                bucket_width_max=min(len(committed_candidates), config.local_beam_width),
                boundary_candidate_count=len(local_boundaries),
                tempo_candidate_count=len(tempos),
                dominance_pruned_state_count=exported.dominance_pruned_state_count,
                width_pruned_state_count=exported.width_pruned_state_count,
                lookahead_trace_fingerprint=lookahead_trace,
                exported_frontier_fingerprint=frontier_fp,
            )
        )
        frontier_widths.append(len(exported.states))
        local_bucket_width_max = max(
            local_bucket_width_max, min(len(committed_candidates), config.local_beam_width)
        )

        for boundary in local_boundaries:
            retained_selected = tuple(
                selected
                for path in paths
                for selected in path.selected_boundaries
                if selected.anchor_id == boundary.anchor_id
            )
            retained_charge_path_count = sum(
                any(selected.anchor_id == boundary.anchor_id for selected in path.selected_boundaries)
                for path in paths
            )
            representative = (
                min(retained_selected, key=_boundary_replay_item)
                if retained_selected
                else None
            )
            authoritative_time_ms = (
                representative.boundary_time_ms
                if representative is not None
                else boundary.time_ms
            )
            owner = _owner_block_index(authoritative_time_ms, block_diagnostics)
            provisional_indexes = tuple(
                sorted(
                    {
                        prior[0]
                        for anchor_id, prior in prior_provisional.items()
                        if anchor_id == boundary.anchor_id
                    }
                    | ({block_index} if boundary.anchor_id in provisional_by_selected else set())
                )
            )
            ownership_by_anchor[boundary.anchor_id] = BoundaryOwnershipRecord(
                boundary_anchor_id=boundary.anchor_id,
                source_candidate_time_ms=boundary.time_ms,
                boundary_time_ms=authoritative_time_ms,
                owner_block_index=owner,
                provisional_block_indexes=provisional_indexes,
                committed_by_previous_block=authoritative_time_ms < cut,
                retained_frontier_charge_path_count=retained_charge_path_count,
                # This is populated from the final traceback after the last
                # frontier has been selected.  A retained alternative is not
                # yet a selected objective charge.
                selected_traceback_objective_charge_count=0,
                objective_charge_count=0,
            )
            prior = prior_provisional.get(boundary.anchor_id)
            if prior is not None and owner == block_index:
                recompute_records.append(
                    LookaheadRecomputeRecord(
                        boundary_anchor_id=boundary.anchor_id,
                        source_candidate_time_ms=boundary.time_ms,
                        boundary_time_ms=authoritative_time_ms,
                        previous_block_index=prior[0],
                        next_block_index=block_index,
                        provisional_trace_fingerprint=prior[1],
                        recomputed_trace_fingerprint=lookahead_trace,
                        retained_frontier_charge_path_count=retained_charge_path_count,
                        selected_traceback_objective_charge_count=0,
                        objective_charge_count=0,
                    )
                )
        prior_provisional = {
            anchor_id: (block_index, lookahead_trace) for anchor_id in provisional_by_selected
        }
        cut = core_end

    best = min(paths, key=lambda path: _terminal_path_order_key(path, context))
    finalized_ownership, finalized_recompute = _finalize_boundary_charge_diagnostics(
        ownership_by_anchor=ownership_by_anchor,
        recompute_records=recompute_records,
        final_frontier=paths,
        selected_path=best,
        blocks=block_diagnostics,
    )
    grid = _grid_from_path(best, context)
    if grid is None:
        return _failure_result(
            config=config,
            candidates=candidates,
            context=context,
            phases=phases,
            bootstrap_replay_keys=bootstrap_replay_keys,
            reason=REASON_SCHEMA_CONSTRUCTION_FAILED,
            stage="schema",
            block_diagnostics=tuple(block_diagnostics),
        )
    return _success_result(
        grid=grid,
        path=best,
        final_frontier=paths,
        config=config,
        candidates=candidates,
        context=context,
        phases=phases,
        bootstrap_replay_keys=bootstrap_replay_keys,
        block_diagnostics=tuple(block_diagnostics),
        frontier_widths=tuple(frontier_widths),
        local_bucket_width_max=local_bucket_width_max,
        ownership_records=finalized_ownership,
        recompute_records=finalized_recompute,
    )


def _state_order_key(state: LocalFrontierState) -> tuple[Any, ...]:
    return (
        state.rank_objective,
        state.committed_objective,
        state.real_section_count,
        state.alias_switch_count,
        state.max_boundary_displacement_ms,
        state.deterministic_replay_key,
    )


def _bootstrap_paths(
    candidates: GlobalConstantJumpCandidateSet,
    *,
    context: _LocalScoreContext,
    phases: tuple[int | None, ...],
) -> tuple[_Path, ...]:
    raw: list[_Path] = []
    for origin_index, origin in enumerate(candidates.origin_candidates):
        if not 20.0 <= origin.bpm <= 1000.0:
            continue
        serialized_start = previous_beat_on_or_before(
            context.coverage_start_ms,
            origin_time_ms=origin.time_ms,
            bpm=origin.bpm,
        )
        for phase in phases:
            raw.append(
                _Path(
                    origin_time_ms=float(origin.time_ms),
                    serialized_first_start_beat=serialized_start,
                    open_start_beat=0,
                    open_start_time_ms=float(origin.time_ms),
                    current_bpm=float(origin.bpm),
                    previous_bpm=None,
                    global_downbeat_phase=phase,
                    closed_sections=(),
                    duration_objective_numerator=0.0,
                    transition_objective=0.0,
                    real_section_count=1,
                    alias_switch_count=0,
                    max_boundary_displacement_ms=0.0,
                    replay_key=(
                        origin_index,
                        float.hex(origin.time_ms),
                        float.hex(origin.bpm),
                        phase,
                    ),
                    selected_boundaries=(),
                    last_boundary_anchor_id=-1,
                )
            )
    ordered = sorted(raw, key=lambda path: _path_order_key(path, context))
    return tuple(ordered[:64])


def _advance_paths_interval(
    incoming: Sequence[_Path],
    *,
    start_ms: float,
    end_ms: float,
    boundaries: Sequence[BoundaryCandidate],
    tempos: Sequence[float],
    context: _LocalScoreContext,
) -> tuple[_Path, ...]:
    if end_ms <= start_ms:
        return tuple(incoming)
    active = tuple(_WorkingPath(path, start_ms) for path in incoming)
    snap_halo_ms = GLOBAL_CONSTANT_JUMP_CONSTANTS.boundary_support_tolerance_ms
    interval_boundaries = tuple(
        sorted(
            (
                boundary
                for boundary in boundaries
                if start_ms - snap_halo_ms <= boundary.time_ms <= end_ms + snap_halo_ms
            ),
            key=lambda boundary: (boundary.time_ms, boundary.anchor_id),
        )
    )
    for boundary in interval_boundaries:
        expanded: list[_WorkingPath] = []
        for working in active:
            path = working.path
            if boundary.anchor_id <= path.last_boundary_anchor_id:
                expanded.append(working)
                continue
            # A rejected proposal is not a mathematical section boundary.  Its
            # persistence branch therefore keeps both the state and scoring
            # cursor unchanged; only an actually selected closure partitions
            # the local evidence interval.
            expanded.append(working)
            for closure in _feasible_closures(path, boundary=boundary, context=context):
                # The snapped integer-lattice closure, not the observed
                # proposal timestamp, owns the half-open interval.
                if not start_ms <= closure.boundary_time_ms < end_ms:
                    continue
                if closure.boundary_time_ms < working.cursor_time_ms:
                    continue
                accrued = _accrue_local_evidence(
                    path,
                    start_ms=working.cursor_time_ms,
                    end_ms=closure.boundary_time_ms,
                    context=context,
                )
                if accrued is None:
                    continue
                expanded.extend(
                    _WorkingPath(successor, closure.boundary_time_ms)
                    for successor in _jump_successors_for_closure(
                    accrued,
                    boundary=boundary,
                    closure=closure,
                    tempos=tempos,
                    context=context,
                )
                )
        active = _prune_working_paths(
            expanded,
            right_boundary_candidate_id=boundary.anchor_id,
            context=context,
        )
    scored: list[_Path] = []
    for working in active:
        accrued = _accrue_local_evidence(
            working.path,
            start_ms=working.cursor_time_ms,
            end_ms=end_ms,
            context=context,
        )
        if accrued is not None:
            scored.append(accrued)
    return _prune_cut_paths(scored, context=context)


def _feasible_closures(
    path: _Path,
    *,
    boundary: BoundaryCandidate,
    context: _LocalScoreContext,
) -> tuple[ClosureCountCandidate, ...]:
    state = _state_from_path(
        path,
        cut_time_ms=max(context.coverage_start_ms, min(boundary.time_ms, context.coverage_end_ms)),
        context=context,
    )
    feasible: list[ClosureCountCandidate] = []
    for closure in closure_count_candidates(state, boundary_time_ms=boundary.time_ms):
        duration_ms = closure.boundary_time_ms - path.open_start_time_ms
        tolerance_ms = min(
            GLOBAL_CONSTANT_JUMP_CONSTANTS.boundary_support_tolerance_ms,
            GLOBAL_CONSTANT_JUMP_CONSTANTS.peak_grid_tolerance_beat_fraction
            * (60000.0 / path.current_bpm),
        )
        if (
            duration_ms < GLOBAL_CONSTANT_JUMP_CONSTANTS.min_section_duration_ms
            or closure.displacement_ms > tolerance_ms
        ):
            continue
        feasible.append(closure)
    return tuple(feasible)


def _jump_successors_for_closure(
    path: _Path,
    *,
    boundary: BoundaryCandidate,
    closure: ClosureCountCandidate,
    tempos: Sequence[float],
    context: _LocalScoreContext,
) -> tuple[_Path, ...]:
    duration_ms = closure.boundary_time_ms - path.open_start_time_ms
    successors: list[_Path] = []
    for right_bpm in tempos:
        if right_bpm == path.current_bpm:
            continue
        if path.real_section_count >= 20:
            continue
        section = _ClosedSection(
                start_beat=(
                    path.serialized_first_start_beat
                    if not path.closed_sections
                    else path.open_start_beat
                ),
                end_beat=closure.boundary_beat,
                bpm=path.current_bpm,
            )
        logical_count = closure.boundary_beat - path.open_start_beat
        visible_start = max(path.open_start_time_ms, context.coverage_start_ms)
        visible_end = min(closure.boundary_time_ms, context.coverage_end_ms)
        visible_duration = max(0.0, visible_end - visible_start)
        close_prior = visible_duration * (
                0.10 * exp004._beat_count_prior_cost(logical_count)  # noqa: SLF001
                + 0.08 * exp004._section_duration_cost(duration_ms)  # noqa: SLF001
            )
        components = context.transition_components(
            boundary=boundary,
            left_bpm=path.current_bpm,
            right_bpm=right_bpm,
        )
        alias_cost = components.alias_switch_cost
        selected = _SelectedBoundary(
                anchor_id=boundary.anchor_id,
                source_time_ms=boundary.time_ms,
                boundary_beat=closure.boundary_beat,
                boundary_time_ms=closure.boundary_time_ms,
            )
        successor_replay_key = path.replay_key + (
            (
                boundary.anchor_id,
                closure.count,
                float.hex(closure.boundary_time_ms),
                float.hex(right_bpm),
            ),
        )
        ledger_entry = (
            BoundaryPairTransitionLedgerEntry(
                predecessor_replay_key=path.replay_key,
                successor_replay_key=successor_replay_key,
                boundary_beat=closure.boundary_beat,
                boundary_time_ms=closure.boundary_time_ms,
                components=components,
            )
            if context.record_objective_ledgers
            else None
        )
        edge_order = context.record_actual_scored_edge(
            predecessor_replay_key=path.replay_key,
            successor_replay_key=successor_replay_key,
            boundary=boundary,
            closure=closure,
            right_bpm=right_bpm,
            components=components,
        )
        transition_edge_orders = (
            path.transition_edge_orders + (edge_order,)
            if edge_order is not None
            else path.transition_edge_orders
        )
        successors.append(
                _Path(
                    origin_time_ms=path.origin_time_ms,
                    serialized_first_start_beat=path.serialized_first_start_beat,
                    open_start_beat=closure.boundary_beat,
                    open_start_time_ms=closure.boundary_time_ms,
                    current_bpm=float(right_bpm),
                    previous_bpm=path.current_bpm,
                    global_downbeat_phase=path.global_downbeat_phase,
                    closed_sections=path.closed_sections + (section,),
                    duration_objective_numerator=(
                        path.duration_objective_numerator + close_prior
                    ),
                    transition_objective=(
                        path.transition_objective
                        + components.normalized_increment
                    ),
                    real_section_count=path.real_section_count + 1,
                    alias_switch_count=path.alias_switch_count + (1 if alias_cost > 0.0 else 0),
                    max_boundary_displacement_ms=max(
                        path.max_boundary_displacement_ms,
                        closure.displacement_ms,
                    ),
                    replay_key=successor_replay_key,
                    selected_boundaries=path.selected_boundaries + (selected,),
                    last_boundary_anchor_id=boundary.anchor_id,
                    transition_ledger=(
                        path.transition_ledger + (ledger_entry,)
                        if ledger_entry is not None
                        else ()
                    ),
                    transition_edge_orders=transition_edge_orders,
                    overlap_lineage=path.overlap_lineage,
                )
            )
    return tuple(successors)


def _accrue_local_evidence(
    path: _Path,
    *,
    start_ms: float,
    end_ms: float,
    context: _LocalScoreContext,
) -> _Path | None:
    local_start = max(start_ms, context.coverage_start_ms)
    local_end = min(end_ms, context.coverage_end_ms)
    if local_end <= local_start:
        return path
    cost = context.local_cost(
        section_start_time_ms=path.open_start_time_ms,
        section_start_beat=path.open_start_beat,
        bpm=path.current_bpm,
        global_downbeat_phase=path.global_downbeat_phase,
        start_ms=local_start,
        end_ms=local_end,
    )
    if not math.isfinite(cost):
        return None
    return replace(
        path,
        duration_objective_numerator=(
            path.duration_objective_numerator + (local_end - local_start) * cost
        ),
    )


def _prune_working_paths(
    paths: Iterable[_WorkingPath],
    *,
    right_boundary_candidate_id: int,
    context: _LocalScoreContext,
) -> tuple[_WorkingPath, ...]:
    if isinstance(right_boundary_candidate_id, bool) or not isinstance(
        right_boundary_candidate_id, int
    ):
        raise ValueError("right_boundary_candidate_id must be an integer")
    buckets: dict[tuple[int, int], dict[tuple[Any, ...], _WorkingPath]] = {}
    for working in paths:
        bucket_key = (
            right_boundary_candidate_id,
            working.path.real_section_count,
        )
        best = buckets.setdefault(bucket_key, {})
        key = (_path_future_key(working.path), float.hex(working.cursor_time_ms))
        prior = best.get(key)
        if prior is None or _path_order_key(working.path, context) < _path_order_key(prior.path, context):
            best[key] = working
    retained: list[_WorkingPath] = []
    for bucket_key in sorted(buckets):
        retained.extend(
            sorted(
                buckets[bucket_key].values(),
                key=lambda item: _path_order_key(item.path, context),
            )[:64]
        )
    return tuple(retained)


def _prune_cut_paths(
    paths: Iterable[_Path], *, context: _LocalScoreContext
) -> tuple[_Path, ...]:
    buckets: dict[int, dict[tuple[Any, ...], _Path]] = {}
    for path in paths:
        best = buckets.setdefault(path.real_section_count, {})
        key = _path_future_key(path)
        prior = best.get(key)
        if prior is None or _path_order_key(path, context) < _path_order_key(prior, context):
            best[key] = path
    retained: list[_Path] = []
    for section_count in sorted(buckets):
        retained.extend(
            sorted(
                buckets[section_count].values(),
                key=lambda path: _path_order_key(path, context),
            )[:64]
        )
    return tuple(retained)


def _state_from_path(
    path: _Path,
    *,
    cut_time_ms: float,
    context: _LocalScoreContext,
    rank_objective: float | None = None,
) -> LocalFrontierState:
    next_index = next_beat_on_or_after(
        cut_time_ms,
        origin_time_ms=path.open_start_time_ms
        - path.open_start_beat * (60000.0 / path.current_bpm),
        bpm=path.current_bpm,
    )
    next_time = path.open_start_time_ms + (
        next_index - path.open_start_beat
    ) * (60000.0 / path.current_bpm)
    committed = _path_objective(path, context)
    return LocalFrontierState(
        cut_time_ms=float(cut_time_ms),
        next_beat_index=next_index,
        next_beat_time_ms=float(next_time),
        current_section_start_beat=path.open_start_beat,
        current_section_start_time_ms=path.open_start_time_ms,
        serialized_first_start_beat=path.serialized_first_start_beat,
        current_bpm=path.current_bpm,
        previous_bpm=path.previous_bpm,
        alias_family=exp004._alias_family_representative_v1(  # noqa: SLF001
            path.current_bpm, GLOBAL_CONSTANT_JUMP_CONSTANTS
        ),
        global_downbeat_phase=path.global_downbeat_phase,
        committed_duration_objective_numerator=path.duration_objective_numerator,
        committed_transition_objective=path.transition_objective,
        rank_objective=committed if rank_objective is None else rank_objective,
        committed_objective=committed,
        real_section_count=path.real_section_count,
        alias_switch_count=path.alias_switch_count,
        max_boundary_displacement_ms=path.max_boundary_displacement_ms,
        open_section_state=(
            path.open_start_beat,
            float.hex(path.open_start_time_ms),
            float.hex(path.current_bpm),
            float.hex(float(cut_time_ms)),
            "closure_prior_unpaid",
        ),
        prefix_sections_or_backpointer=tuple(
            (section.start_beat, section.end_beat, float.hex(section.bpm))
            for section in path.closed_sections
        ),
        deterministic_replay_key=path.replay_key,
    )


def _path_objective(path: _Path, context: _LocalScoreContext) -> float:
    return float(path.duration_objective_numerator / context.duration_ms + path.transition_objective)


def _path_order_key(path: _Path, context: _LocalScoreContext) -> tuple[Any, ...]:
    return (
        _path_objective(path, context),
        path.real_section_count,
        path.alias_switch_count,
        path.max_boundary_displacement_ms,
        path.origin_time_ms,
        path.replay_key,
    )


def _terminal_path_order_key(path: _Path, context: _LocalScoreContext) -> tuple[Any, ...]:
    tail_prior = _tail_prior_numerator(path, context)
    return (
        _path_objective(path, context) + tail_prior / context.duration_ms,
        path.real_section_count,
        path.alias_switch_count,
        path.max_boundary_displacement_ms,
        path.replay_key,
    )


def _tail_prior_numerator(path: _Path, context: _LocalScoreContext) -> float:
    tail_end = next_beat_on_or_after(
        context.coverage_end_ms,
        origin_time_ms=path.open_start_time_ms
        - path.open_start_beat * (60000.0 / path.current_bpm),
        bpm=path.current_bpm,
    )
    if tail_end <= path.open_start_beat:
        tail_end = path.open_start_beat + 1
    logical_count = tail_end - path.open_start_beat
    full_duration = logical_count * 60000.0 / path.current_bpm
    visible_duration = max(
        0.0,
        context.coverage_end_ms - max(context.coverage_start_ms, path.open_start_time_ms),
    )
    return float(visible_duration * (
        0.10 * exp004._beat_count_prior_cost(logical_count)  # noqa: SLF001
        + 0.08 * exp004._section_duration_cost(full_duration)  # noqa: SLF001
    ))


def _prune_paths(
    paths: Iterable[_Path],
    *,
    context: _LocalScoreContext,
    limit: int,
) -> tuple[_Path, ...]:
    best: dict[tuple[Any, ...], _Path] = {}
    for path in paths:
        key = _path_future_key(path)
        prior = best.get(key)
        if prior is None or _path_order_key(path, context) < _path_order_key(prior, context):
            best[key] = path
    return tuple(sorted(best.values(), key=lambda path: _path_order_key(path, context))[:limit])


def _dominant_paths_by_future(
    paths: Iterable[_Path], context: _LocalScoreContext
) -> tuple[_Path, ...]:
    return _prune_cut_paths(paths, context=context)


def _path_future_key(path: _Path) -> tuple[Any, ...]:
    return (
        path.open_start_beat,
        float.hex(path.open_start_time_ms),
        float.hex(path.current_bpm),
        None if path.previous_bpm is None else float.hex(path.previous_bpm),
        path.global_downbeat_phase,
        path.real_section_count,
        path.alias_switch_count,
        float.hex(path.max_boundary_displacement_ms),
    )


def _grid_from_path(path: _Path, context: _LocalScoreContext) -> TimingV3Grid | None:
    origin_for_open_lattice = path.open_start_time_ms - (
        path.open_start_beat * 60000.0 / path.current_bpm
    )
    tail_end = next_beat_on_or_after(
        context.coverage_end_ms,
        origin_time_ms=origin_for_open_lattice,
        bpm=path.current_bpm,
    )
    if tail_end <= path.open_start_beat:
        tail_end = path.open_start_beat + 1
    sections = list(path.closed_sections)
    sections.append(
        _ClosedSection(
            start_beat=(
                path.serialized_first_start_beat
                if not sections
                else path.open_start_beat
            ),
            end_beat=tail_end,
            bpm=path.current_bpm,
        )
    )
    merged: list[_ClosedSection] = []
    for section in sections:
        if merged and merged[-1].bpm == section.bpm and merged[-1].end_beat == section.start_beat:
            prior = merged[-1]
            merged[-1] = _ClosedSection(prior.start_beat, section.end_beat, prior.bpm)
        else:
            merged.append(section)
    try:
        return TimingV3Grid(
            origin_beat=0,
            origin_time_ms=path.origin_time_ms,
            sections=tuple(
                ConstantTimingSection(
                    start_beat=section.start_beat,
                    end_beat=section.end_beat,
                    bpm=section.bpm,
                )
                for section in merged
            ),
            coverage_start_ms=context.coverage_start_ms,
            coverage_end_ms=context.coverage_end_ms,
        )
    except (TypeError, ValueError):
        return None


def _paths_with_new_overlap_lineage(
    *,
    context: _LocalScoreContext,
    exported_states: Sequence[LocalFrontierState],
    path_by_replay: dict[
        tuple[Any, ...],
        tuple[_Path, tuple[_SelectedBoundary, ...], _Path],
    ],
    block_index: int,
    core_end_ms: float,
    lookahead_end_ms: float,
) -> tuple[_Path, ...]:
    paths: list[_Path] = []
    lineages: list[_OverlapLineage] = []
    for ordinal, state in enumerate(exported_states):
        path, _, ranked_lookahead = path_by_replay[state.deterministic_replay_key]
        provisional_trace = _overlap_trace_for_path(
            ranked_lookahead,
            context=context,
            start_ms=core_end_ms,
            end_ms=lookahead_end_ms,
        )
        lineage = _make_overlap_lineage(
            state=state,
            path=path,
            block_index=block_index,
            export_ordinal=ordinal,
            core_end_ms=core_end_ms,
            lookahead_end_ms=lookahead_end_ms,
            provisional_trace=provisional_trace,
        )
        lineages.append(lineage)
        paths.append(replace(path, overlap_lineage=lineage))
    context.pending_overlap_lineages = tuple(lineages)
    return tuple(paths)


def _make_overlap_lineage(
    *,
    state: LocalFrontierState,
    path: _Path,
    block_index: int,
    export_ordinal: int,
    core_end_ms: float,
    lookahead_end_ms: float,
    provisional_trace: _OverlapTrace,
) -> _OverlapLineage:
    future_equivalence_sha256 = _json_fingerprint(state.future_equivalence_key)
    committed_replay_sha256 = _json_fingerprint(path.replay_key)
    token = (
        LOOKAHEAD_OVERLAP_RECORD_CONTRACT_VERSION,
        int(block_index),
        int(export_ordinal),
        future_equivalence_sha256,
        committed_replay_sha256,
    )
    return _OverlapLineage(
        record_contract_version=LOOKAHEAD_OVERLAP_RECORD_CONTRACT_VERSION,
        prior_block_index=int(block_index),
        prior_export_ordinal=int(export_ordinal),
        future_equivalence_sha256=future_equivalence_sha256,
        committed_replay_sha256=committed_replay_sha256,
        lineage_sha256=_json_fingerprint(token),
        previous_core_end_ms=float(core_end_ms),
        previous_lookahead_end_ms=float(lookahead_end_ms),
        provisional_trace=provisional_trace,
    )


def _record_pending_overlap_at_export(
    *,
    context: _LocalScoreContext,
    exported_paths: Sequence[_Path],
    next_block_index: int,
    next_core_end_ms: float,
) -> None:
    if not context.pending_overlap_lineages:
        return
    _ensure_overlap_record_capacity(
        len(context.overlap_records),
        len(context.pending_overlap_lineages),
    )
    pending_by_sha = {
        lineage.lineage_sha256: lineage
        for lineage in context.pending_overlap_lineages
    }
    first_descendant_by_lineage: dict[str, _Path] = {}
    for path in exported_paths:
        lineage = path.overlap_lineage
        if lineage is None:
            continue
        if lineage.lineage_sha256 not in pending_by_sha:
            raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
        first_descendant_by_lineage.setdefault(lineage.lineage_sha256, path)

    for lineage in context.pending_overlap_lineages:
        descendant = first_descendant_by_lineage.get(lineage.lineage_sha256)
        record, residuals = _lookahead_overlap_record(
            lineage=lineage,
            descendant=descendant,
            context=context,
            next_block_index=next_block_index,
            next_core_end_ms=next_core_end_ms,
            residual_capacity_remaining=(
                MAX_OVERLAP_RESIDUAL_PAIRS - context.overlap_residuals.count
            ),
        )
        if record.unavailable_reason is None:
            context.overlap_residuals.extend_tagged(
                prior_block_index=lineage.prior_block_index,
                prior_export_ordinal=lineage.prior_export_ordinal,
                residuals=residuals,
            )
        context.overlap_records.append(record)


def _ensure_overlap_record_capacity(existing_count: int, append_count: int) -> None:
    if existing_count < 0 or append_count < 0:
        raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
    if existing_count + append_count > MAX_OVERLAP_RECORDS_PER_AUDIO:
        raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)


def _ensure_overlap_residual_capacity(existing_count: int, append_count: int) -> None:
    if existing_count < 0 or append_count < 0:
        raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
    if existing_count + append_count > MAX_OVERLAP_RESIDUAL_PAIRS:
        raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)


def _lookahead_overlap_record(
    *,
    lineage: _OverlapLineage,
    descendant: _Path | None,
    context: _LocalScoreContext,
    next_block_index: int,
    next_core_end_ms: float,
    residual_capacity_remaining: int = MAX_OVERLAP_RESIDUAL_PAIRS,
) -> tuple[LookaheadOverlapRecord, tuple[tuple[int, float, float], ...]]:
    comparison_start_ms = float(lineage.previous_core_end_ms)
    comparison_end_ms = float(min(lineage.previous_lookahead_end_ms, next_core_end_ms))
    if not math.isfinite(comparison_start_ms) or not math.isfinite(comparison_end_ms):
        raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
    common = dict(
        previous_block_index=lineage.prior_block_index,
        next_block_index=int(next_block_index),
        previous_export_ordinal=lineage.prior_export_ordinal,
        lineage_sha256=lineage.lineage_sha256,
        comparison_start_ms=comparison_start_ms,
        comparison_end_ms=comparison_end_ms,
        provisional_trace_sha256=lineage.provisional_trace.sha256,
    )
    empty_domain_sha256 = _comparison_domain_sha256(
        comparison_start_ms=comparison_start_ms,
        comparison_end_ms=comparison_end_ms,
        common_beats=(),
    )
    if comparison_end_ms <= comparison_start_ms:
        return (
            LookaheadOverlapRecord(
                **common,
                comparison_domain_sha256=empty_domain_sha256,
                recomputed_trace_sha256=None,
                comparable_beat_count=0,
                residual_vector_sha256=None,
                p90_ms=None,
                p90_beats=None,
                unavailable_reason=UNAVAILABLE_EMPTY_COMMON_TIME_DOMAIN,
            ),
            (),
        )
    if descendant is None:
        return (
            LookaheadOverlapRecord(
                **common,
                comparison_domain_sha256=empty_domain_sha256,
                recomputed_trace_sha256=None,
                comparable_beat_count=0,
                residual_vector_sha256=None,
                p90_ms=None,
                p90_beats=None,
                unavailable_reason=UNAVAILABLE_LINEAGE_NOT_RETAINED_AT_NEXT_CUT,
            ),
            (),
        )

    recomputed_trace = _overlap_trace_for_path(
        descendant,
        context=context,
        start_ms=comparison_start_ms,
        end_ms=comparison_end_ms,
    )
    residuals = _overlap_residuals(
        provisional=lineage.provisional_trace,
        recomputed=recomputed_trace,
        comparison_start_ms=comparison_start_ms,
        comparison_end_ms=comparison_end_ms,
        residual_capacity_remaining=residual_capacity_remaining,
    )
    domain_sha256 = _comparison_domain_sha256(
        comparison_start_ms=comparison_start_ms,
        comparison_end_ms=comparison_end_ms,
        common_beats=tuple(beat for beat, _, _ in residuals),
    )
    residual_payload = tuple(
        (beat, float.hex(residual_ms), float.hex(residual_beats))
        for beat, residual_ms, residual_beats in residuals
    )
    residual_sha256 = _json_fingerprint(residual_payload)
    if len(residuals) < 8:
        return (
            LookaheadOverlapRecord(
                **common,
                comparison_domain_sha256=domain_sha256,
                recomputed_trace_sha256=recomputed_trace.sha256,
                comparable_beat_count=len(residuals),
                residual_vector_sha256=residual_sha256,
                p90_ms=None,
                p90_beats=None,
                unavailable_reason=UNAVAILABLE_FEWER_THAN_8_COMPARABLE_BEATS,
            ),
            (),
        )

    p90_ms = _p90_linear(tuple(residual_ms for _, residual_ms, _ in residuals))
    p90_beats = _p90_linear(tuple(residual_beats for _, _, residual_beats in residuals))
    return (
        LookaheadOverlapRecord(
            **common,
            comparison_domain_sha256=domain_sha256,
            recomputed_trace_sha256=recomputed_trace.sha256,
            comparable_beat_count=len(residuals),
            residual_vector_sha256=residual_sha256,
            p90_ms=p90_ms,
            p90_beats=p90_beats,
            unavailable_reason=None,
        ),
        residuals,
    )


def _comparison_domain_sha256(
    *,
    comparison_start_ms: float,
    comparison_end_ms: float,
    common_beats: Sequence[int],
) -> str:
    if not math.isfinite(comparison_start_ms) or not math.isfinite(comparison_end_ms):
        raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
    exact_beats: list[int] = []
    previous: int | None = None
    for beat in common_beats:
        if isinstance(beat, bool) or not isinstance(beat, int):
            raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
        if previous is not None and beat <= previous:
            raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
        exact_beats.append(beat)
        previous = beat
    return _json_fingerprint(
        [
            float.hex(float(comparison_start_ms)),
            float.hex(float(comparison_end_ms)),
            exact_beats,
        ]
    )


def _overlap_trace_for_path(
    path: _Path,
    *,
    context: _LocalScoreContext,
    start_ms: float,
    end_ms: float,
) -> _OverlapTrace:
    if not math.isfinite(start_ms) or not math.isfinite(end_ms):
        raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
    beats: list[tuple[int, str, str]] = []
    seen_beats: set[int] = set()
    if end_ms > start_ms:
        for section_start_beat, section_end_beat, section_start_time_ms, bpm in _trace_sections(
            path,
            end_ms=end_ms,
        ):
            if section_end_beat <= section_start_beat:
                continue
            if not math.isfinite(section_start_time_ms) or not math.isfinite(bpm) or bpm <= 0.0:
                raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
            origin_time_ms = section_start_time_ms - section_start_beat * (
                60000.0 / bpm
            )
            beat = max(
                section_start_beat,
                next_beat_on_or_after(start_ms, origin_time_ms=origin_time_ms, bpm=bpm),
            )
            while beat < section_end_beat:
                time_ms = lattice_time_ms(beat, origin_time_ms=origin_time_ms, bpm=bpm)
                if time_ms >= end_ms:
                    break
                if time_ms >= start_ms:
                    if beat in seen_beats:
                        raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
                    if len(beats) >= MAX_OVERLAP_TRACE_BEATS:
                        raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
                    seen_beats.add(beat)
                    beats.append((beat, float.hex(time_ms), float.hex(float(bpm))))
                beat += 1

    boundary_items = []
    seen_boundaries: set[tuple[int, str, int, str, str, str]] = set()
    for index, boundary in sorted(
        enumerate(path.selected_boundaries),
        key=lambda item: (
            item[1].boundary_time_ms,
            item[1].anchor_id,
            item[1].boundary_beat,
        ),
    ):
        if start_ms <= boundary.boundary_time_ms < end_ms:
            item = _boundary_trace_item(path, boundary, index)
            if item in seen_boundaries:
                raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
            if len(boundary_items) >= MAX_OVERLAP_TRACE_REAL_BOUNDARIES:
                raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
            seen_boundaries.add(item)
            boundary_items.append(item)

    ordered_beats = tuple(sorted(beats, key=lambda item: item[0]))
    payload = {
        "beats": ordered_beats,
        "boundaries": tuple(boundary_items),
    }
    return _OverlapTrace(
        beats=ordered_beats,
        boundaries=tuple(boundary_items),
        sha256=_json_fingerprint(payload),
    )


def _trace_sections(
    path: _Path,
    *,
    end_ms: float,
) -> tuple[tuple[int, int, float, float], ...]:
    if len(path.closed_sections) != len(path.selected_boundaries):
        raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
    sections: list[tuple[int, int, float, float]] = []
    for index, section in enumerate(path.closed_sections):
        if section.end_beat <= section.start_beat:
            raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
        if index == 0:
            section_start_time_ms = path.origin_time_ms + section.start_beat * (
                60000.0 / section.bpm
            )
        else:
            prior_boundary = path.selected_boundaries[index - 1]
            if prior_boundary.boundary_beat != section.start_beat:
                raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
            section_start_time_ms = prior_boundary.boundary_time_ms
        closing_boundary = path.selected_boundaries[index]
        if closing_boundary.boundary_beat != section.end_beat:
            raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
        sections.append(
            (
                section.start_beat,
                section.end_beat,
                float(section_start_time_ms),
                float(section.bpm),
            )
        )

    open_period_ms = 60000.0 / path.current_bpm
    open_origin_time_ms = path.open_start_time_ms - path.open_start_beat * open_period_ms
    open_end_beat = next_beat_on_or_after(
        end_ms,
        origin_time_ms=open_origin_time_ms,
        bpm=path.current_bpm,
    )
    open_end_beat = max(path.open_start_beat, open_end_beat)
    sections.append(
        (
            path.open_start_beat,
            open_end_beat,
            path.open_start_time_ms,
            path.current_bpm,
        )
    )
    return tuple(sections)


def _boundary_trace_item(
    path: _Path,
    boundary: _SelectedBoundary,
    selected_boundary_index: int,
) -> tuple[int, str, int, str, str, str]:
    if selected_boundary_index < 0 or selected_boundary_index >= len(path.closed_sections):
        raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
    closed = path.closed_sections[selected_boundary_index]
    if (
        closed.end_beat != boundary.boundary_beat
        or selected_boundary_index >= len(path.selected_boundaries)
        or path.selected_boundaries[selected_boundary_index] != boundary
    ):
        raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
    if selected_boundary_index + 1 < len(path.closed_sections):
        right_bpm = path.closed_sections[selected_boundary_index + 1].bpm
    else:
        if path.open_start_beat != boundary.boundary_beat:
            raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
        right_bpm = path.current_bpm
    if (
        not math.isfinite(boundary.source_time_ms)
        or not math.isfinite(boundary.boundary_time_ms)
        or not math.isfinite(closed.bpm)
        or not math.isfinite(right_bpm)
        or closed.bpm <= 0.0
        or right_bpm <= 0.0
    ):
        raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
    return (
        int(boundary.anchor_id),
        float.hex(float(boundary.source_time_ms)),
        int(boundary.boundary_beat),
        float.hex(float(boundary.boundary_time_ms)),
        float.hex(float(closed.bpm)),
        float.hex(float(right_bpm)),
    )


def _overlap_residuals(
    *,
    provisional: _OverlapTrace,
    recomputed: _OverlapTrace,
    comparison_start_ms: float,
    comparison_end_ms: float,
    residual_capacity_remaining: int = MAX_OVERLAP_RESIDUAL_PAIRS,
) -> tuple[tuple[int, float, float], ...]:
    if (
        isinstance(residual_capacity_remaining, bool)
        or not isinstance(residual_capacity_remaining, int)
        or residual_capacity_remaining < 0
    ):
        raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
    provisional_by_beat = _trace_beat_map(provisional)
    recomputed_by_beat = _trace_beat_map(recomputed)
    residuals: list[tuple[int, float, float]] = []
    for beat in sorted(set(provisional_by_beat) & set(recomputed_by_beat)):
        provisional_time_ms, provisional_bpm = provisional_by_beat[beat]
        recomputed_time_ms, recomputed_bpm = recomputed_by_beat[beat]
        if not (
            comparison_start_ms <= provisional_time_ms < comparison_end_ms
            and comparison_start_ms <= recomputed_time_ms < comparison_end_ms
        ):
            continue
        provisional_period_ms = 60000.0 / provisional_bpm
        recomputed_period_ms = 60000.0 / recomputed_bpm
        residual_ms = abs(provisional_time_ms - recomputed_time_ms)
        residual_beats = residual_ms / min(provisional_period_ms, recomputed_period_ms)
        if not math.isfinite(residual_ms) or not math.isfinite(residual_beats):
            raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
        if len(residuals) >= residual_capacity_remaining:
            raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
        residuals.append((beat, float(residual_ms), float(residual_beats)))
    return tuple(residuals)


def _trace_beat_map(trace: _OverlapTrace) -> dict[int, tuple[float, float]]:
    by_beat: dict[int, tuple[float, float]] = {}
    for beat, time_hex, bpm_hex in trace.beats:
        if beat in by_beat:
            raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
        time_ms = float.fromhex(time_hex)
        bpm = float.fromhex(bpm_hex)
        if not math.isfinite(time_ms) or not math.isfinite(bpm) or bpm <= 0.0:
            raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
        by_beat[int(beat)] = (time_ms, bpm)
    return by_beat


def _p90_linear(values: Sequence[float]) -> float:
    if not values:
        raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 for value in ordered):
        raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
    h = (len(ordered) - 1) * 0.9
    lo = math.floor(h)
    hi = math.ceil(h)
    return float(ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo]))


def _terminal_objective_ledger(
    path: _Path,
    *,
    context: _LocalScoreContext,
    selection_rank: int,
    selected: bool,
) -> TerminalObjectiveLedger:
    grid = _grid_from_path(path, context)
    if grid is None:
        section_signature: tuple[tuple[int, int, str], ...] = ()
        bpm_sequence: tuple[float, ...] = ()
    else:
        section_signature = tuple(
            (section.start_beat, section.end_beat, float.hex(section.bpm))
            for section in grid.sections
        )
        bpm_sequence = tuple(section.bpm for section in grid.sections)
    normalized_duration = path.duration_objective_numerator / context.duration_ms
    tail_prior = _tail_prior_numerator(path, context)
    normalized_tail = tail_prior / context.duration_ms
    reconstructed_transition = sum(
        entry.components.normalized_increment for entry in path.transition_ledger
    )
    reconstructed_terminal = normalized_duration + reconstructed_transition + normalized_tail
    recorded_terminal = _terminal_path_order_key(path, context)[0]
    return TerminalObjectiveLedger(
        selection_rank=selection_rank,
        selected=selected,
        replay_key=path.replay_key,
        section_signature=section_signature,
        bpm_sequence=bpm_sequence,
        candidate_period_pairs_by_anchor=context.candidate_period_pairs_by_anchor,
        duration_objective_numerator=path.duration_objective_numerator,
        duration_normalizer=context.duration_ms,
        normalized_duration_objective=float(normalized_duration),
        transition_entries=path.transition_ledger,
        recorded_transition_objective=path.transition_objective,
        tail_prior_numerator=tail_prior,
        normalized_tail_prior=float(normalized_tail),
        reconstructed_terminal_objective=float(reconstructed_terminal),
        recorded_terminal_objective=float(recorded_terminal),
    )


def _transition_component_cache_records(
    context: _LocalScoreContext,
) -> tuple[BoundaryPairTransitionCacheRecord, ...]:
    return tuple(
        BoundaryPairTransitionCacheRecord(
            component_cache_key=key,
            components=context.transition_component_cache[key],
        )
        for key in sorted(context.transition_component_cache)
    )


def _transition_edge_identity(
    entry: BoundaryPairTransitionLedgerEntry,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    return (entry.predecessor_replay_key, entry.successor_replay_key)


def _actual_scored_edges_with_membership(
    context: _LocalScoreContext,
    *,
    final_frontier: Sequence[_Path],
    selected_path: _Path | None,
) -> tuple[BoundaryPairScoredEdgeRecord, ...]:
    if (
        context.actual_scored_edges is None
        or context.provisional_transition_edge_order_records is None
    ):
        raise AssertionError("full Exp006 edge records were not initialized")
    retained_terminal_edge_orders = {
        edge_order
        for path in final_frontier
        for edge_order in path.transition_edge_orders
    }
    retained_provisional_edge_orders = {
        edge_order
        for record in context.provisional_transition_edge_order_records
        for edge_order in record.edge_orders
    }
    selected_edge_orders = (
        set()
        if selected_path is None
        else set(selected_path.transition_edge_orders)
    )
    return tuple(
        replace(
            edge,
            retained_terminal_path=edge.edge_order in retained_terminal_edge_orders,
            retained_provisional_path=edge.edge_order in retained_provisional_edge_orders,
            selected_traceback_path=edge.edge_order in selected_edge_orders,
        )
        for edge in context.actual_scored_edges
    )


def _terminal_occurrence_records(
    final_frontier: Sequence[_Path],
    *,
    selected_path: _Path | None,
) -> tuple[BoundaryPairTerminalOccurrenceRecord, ...]:
    return tuple(
        BoundaryPairTerminalOccurrenceRecord(
            selection_rank=index,
            selected=(
                selected_path is not None and path.replay_key == selected_path.replay_key
            ),
            replay_key=path.replay_key,
            edge_orders=tuple(path.transition_edge_orders),
        )
        for index, path in enumerate(final_frontier)
    )


def _provisional_occurrence_records(
    context: _LocalScoreContext,
) -> tuple[BoundaryPairProvisionalOccurrenceRecord, ...]:
    if context.provisional_transition_edge_order_records is None:
        raise AssertionError("full Exp006 occurrence records were not initialized")
    return tuple(
        BoundaryPairProvisionalOccurrenceRecord(
            record_index=index,
            block_index=record.block_index,
            committed_replay_key=record.committed_replay_key,
            ranked_lookahead_replay_key=record.ranked_lookahead_replay_key,
            edge_orders=tuple(record.edge_orders),
        )
        for index, record in enumerate(context.provisional_transition_edge_order_records)
    )


def _build_objective_diagnostics(
    *,
    context: _LocalScoreContext,
    candidates: GlobalConstantJumpCandidateSet,
    final_frontier: Sequence[_Path],
    selected_path: _Path | None,
) -> BoundaryPairObjectiveDiagnostics:
    if context.provisional_transition_ledgers is None:
        raise AssertionError("full objective ledgers were not initialized")
    ordered = sorted(final_frontier, key=lambda item: _terminal_path_order_key(item, context))
    ledgers = tuple(
        _terminal_objective_ledger(
            path,
            context=context,
            selection_rank=index,
            selected=(selected_path is not None and path.replay_key == selected_path.replay_key),
        )
        for index, path in enumerate(ordered)
    )
    selected_objective = ledgers[0].recorded_terminal_objective if ledgers else None
    runner_up = ledgers[1].recorded_terminal_objective if len(ledgers) > 1 else None
    margin = None if selected_objective is None or runner_up is None else runner_up - selected_objective
    contract_version = (
        BOUNDARY_PAIR_TRANSITION_CONTRACT_VERSION
        if context.record_measurement_v2
        else _BOUNDARY_PAIR_TRANSITION_CONTRACT_VERSION_V1
    )
    transition_component_cache_entries = (
        _transition_component_cache_records(context)
        if context.record_measurement_v2
        else ()
    )
    actual_scored_edges = (
        _actual_scored_edges_with_membership(
            context,
            final_frontier=ordered,
            selected_path=selected_path,
        )
        if context.record_measurement_v2
        else ()
    )
    terminal_path_occurrence_records = (
        _terminal_occurrence_records(ordered, selected_path=selected_path)
        if context.record_measurement_v2
        else ()
    )
    provisional_path_occurrence_records = (
        _provisional_occurrence_records(context)
        if context.record_measurement_v2
        else ()
    )
    selected_traceback_edge_orders = (
        tuple() if selected_path is None else tuple(selected_path.transition_edge_orders)
    )
    payload = {
        "contract": contract_version,
        "variant": context.objective_variant.value,
        "candidate": candidates.diagnostics.candidate_fingerprint,
        "cache": tuple(
            asdict(context.transition_component_cache[key])
            for key in sorted(context.transition_component_cache)
        ),
        "terminal": tuple(asdict(ledger) for ledger in ledgers),
        "provisional": tuple(
            asdict(record) for record in context.provisional_transition_ledgers
        ),
    }
    if context.record_measurement_v2:
        payload.update(
            {
                "transition_component_cache_entries": tuple(
                    asdict(record) for record in transition_component_cache_entries
                ),
                "actual_scored_edges": tuple(
                    asdict(record) for record in actual_scored_edges
                ),
                "block_resource_records": tuple(
                    asdict(record) for record in context.block_resource_records
                ),
                "class_coverage_records": tuple(
                    asdict(record) for record in context.class_coverage_records
                ),
                "terminal_path_occurrence_records": tuple(
                    asdict(record) for record in terminal_path_occurrence_records
                ),
                "provisional_path_occurrence_records": tuple(
                    asdict(record) for record in provisional_path_occurrence_records
                ),
                "selected_traceback_edge_orders": selected_traceback_edge_orders,
            }
        )
    common = dict(
        contract_version=contract_version,
        objective_variant=context.objective_variant.value,
        sparsity_floor=(
            EXP006_PAIR_SPARSITY_FLOOR
            if context.objective_variant
            is LocalFrontierObjectiveVariant.EXP006_PAIR_CONDITIONED_CHANGE_FLOOR_1_4
            else None
        ),
        alias_multipliers=BOUNDARY_PAIR_ALIAS_MULTIPLIERS,
        candidate_fingerprint=candidates.diagnostics.candidate_fingerprint,
        transition_cache_size=len(context.transition_component_cache),
        terminal_path_ledgers=ledgers,
        provisional_path_ledgers=tuple(context.provisional_transition_ledgers),
        selected_terminal_objective=selected_objective,
        runner_up_terminal_objective=runner_up,
        selected_runner_up_margin=margin,
        deterministic_fingerprint=_json_fingerprint(payload),
    )
    if context.record_measurement_v2:
        return BoundaryPairObjectiveDiagnosticsV2(
            **common,
            transition_component_cache_entries=transition_component_cache_entries,
            actual_scored_edges=actual_scored_edges,
            block_resource_records=tuple(context.block_resource_records),
            class_coverage_records=tuple(context.class_coverage_records),
            terminal_path_occurrence_records=terminal_path_occurrence_records,
            provisional_path_occurrence_records=provisional_path_occurrence_records,
            selected_traceback_edge_orders=selected_traceback_edge_orders,
        )
    return BoundaryPairObjectiveDiagnostics(**common)


def _terminal_objective_scalars(
    *,
    context: _LocalScoreContext,
    final_frontier: Sequence[_Path],
) -> tuple[float | None, float | None, float | None]:
    ordered = sorted(final_frontier, key=lambda item: _terminal_path_order_key(item, context))
    selected_objective = (
        _terminal_path_order_key(ordered[0], context)[0] if ordered else None
    )
    runner_up = (
        _terminal_path_order_key(ordered[1], context)[0]
        if len(ordered) > 1
        else None
    )
    margin = (
        None
        if selected_objective is None or runner_up is None
        else runner_up - selected_objective
    )
    return selected_objective, runner_up, margin


def _build_bounded_diagnostics(
    *,
    context: _LocalScoreContext,
    candidates: GlobalConstantJumpCandidateSet,
    final_frontier: Sequence[_Path],
) -> BoundaryPairBoundedDiagnostics:
    if len(context.block_resource_records) > context.config.max_blocks:
        raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
    if len(context.class_coverage_records) > context.config.max_blocks:
        raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
    selected_objective, runner_up, margin = _terminal_objective_scalars(
        context=context,
        final_frontier=final_frontier,
    )
    overlap = _build_overlap_diagnostics(context)
    common = dict(
        contract_version=BOUNDARY_PAIR_BOUNDED_CONTRACT_VERSION,
        objective_variant=context.objective_variant.value,
        candidate_fingerprint=candidates.diagnostics.candidate_fingerprint,
        transition_cache_size=len(context.transition_component_cache),
        actual_scored_edge_count=context.actual_scored_edge_count,
        selected_terminal_objective=selected_objective,
        runner_up_terminal_objective=runner_up,
        selected_runner_up_margin=margin,
        block_resource_records=tuple(context.block_resource_records),
        class_coverage_records=tuple(context.class_coverage_records),
        overlap=overlap,
    )
    payload = {
        "contract_version": common["contract_version"],
        "objective_variant": common["objective_variant"],
        "candidate_fingerprint": common["candidate_fingerprint"],
        "transition_cache_size": common["transition_cache_size"],
        "actual_scored_edge_count": common["actual_scored_edge_count"],
        "selected_terminal_objective": common["selected_terminal_objective"],
        "runner_up_terminal_objective": common["runner_up_terminal_objective"],
        "selected_runner_up_margin": common["selected_runner_up_margin"],
        "block_resource_records": tuple(
            asdict(record) for record in common["block_resource_records"]
        ),
        "class_coverage_records": tuple(
            asdict(record) for record in common["class_coverage_records"]
        ),
        "overlap": asdict(overlap),
    }
    return BoundaryPairBoundedDiagnostics(
        **common,
        deterministic_fingerprint=_json_fingerprint(payload),
    )


def _build_overlap_diagnostics(
    context: _LocalScoreContext,
) -> LocalFrontierOverlapDiagnostics:
    records = tuple(context.overlap_records)
    if len(records) > MAX_OVERLAP_RECORDS_PER_AUDIO:
        raise _DiagnosticsIntegrityFailure(REASON_DIAGNOSTICS_INTEGRITY_FAILURE)
    available_records = tuple(record for record in records if record.unavailable_reason is None)
    unavailable_count = len(records) - len(available_records)
    comparable_beat_count = sum(record.comparable_beat_count for record in available_records)
    if available_records:
        p90_ms = _p90_linear(context.overlap_residuals.residual_ms)
        p90_beats = _p90_linear(context.overlap_residuals.residual_beats)
        residual_vector_sha256 = context.overlap_residuals.residual_vector_sha256()
    else:
        p90_ms = None
        p90_beats = None
        residual_vector_sha256 = None
    records_sha256 = _json_fingerprint(tuple(asdict(record) for record in records))
    diagnostics = LocalFrontierOverlapDiagnostics(
        metric_version=LOCAL_FRONTIER_OVERLAP_METRIC_VERSION,
        record_contract_version=LOOKAHEAD_OVERLAP_RECORD_CONTRACT_VERSION,
        record_count=len(records),
        available_record_count=len(available_records),
        unavailable_record_count=unavailable_count,
        comparable_beat_count=comparable_beat_count,
        p90_ms=p90_ms,
        p90_beats=p90_beats,
        residual_vector_sha256=residual_vector_sha256,
        records_sha256=records_sha256,
        records=records,
    )
    context.overlap_residuals.release()
    return diagnostics


def _success_result(
    *,
    grid: TimingV3Grid,
    path: _Path,
    final_frontier: Sequence[_Path],
    config: LocalFrontierConfig,
    candidates: GlobalConstantJumpCandidateSet,
    context: _LocalScoreContext,
    phases: tuple[int | None, ...],
    bootstrap_replay_keys: tuple[tuple[Any, ...], ...],
    block_diagnostics: tuple[LocalFrontierBlockDiagnostics, ...],
    frontier_widths: tuple[int, ...],
    local_bucket_width_max: int,
    ownership_records: tuple[BoundaryOwnershipRecord, ...],
    recompute_records: tuple[LookaheadRecomputeRecord, ...],
) -> LocalFrontierResult:
    grid_fp = _json_fingerprint(grid.to_dict())
    replay_fp = _json_fingerprint(path.replay_key)
    logical_count = grid.sections[0].end_beat - grid.origin_beat
    serialized_count = grid.sections[0].end_beat - grid.sections[0].start_beat
    diagnostics = _base_diagnostics(
        config=config,
        candidates=candidates,
        context=context,
        phases=phases,
        bootstrap_replay_keys=bootstrap_replay_keys,
        block_diagnostics=block_diagnostics,
        frontier_widths=frontier_widths,
        local_bucket_width_max=local_bucket_width_max,
        ownership_records=ownership_records,
        recompute_records=recompute_records,
        selected_origin_time_ms=path.origin_time_ms,
        selected_serialized_first_start_beat=path.serialized_first_start_beat,
        first_section_logical_beat_count=logical_count,
        first_section_serialized_beat_count=serialized_count,
        selected_section_count=len(grid.sections),
        replay_fingerprint=replay_fp,
        grid_fingerprint=grid_fp,
        fallback_reason=None,
        fallback_stage=None,
    )
    objective_diagnostics = (
        _build_objective_diagnostics(
            context=context,
            candidates=candidates,
            final_frontier=final_frontier,
            selected_path=path,
        )
        if context.record_objective_ledgers
        else None
    )
    if context.record_bounded_diagnostics:
        context.bounded_diagnostics = _build_bounded_diagnostics(
            context=context,
            candidates=candidates,
            final_frontier=final_frontier,
        )
    return LocalFrontierResult(
        grid=grid,
        diagnostics=diagnostics,
        reason=None,
        objective_diagnostics=objective_diagnostics,
    )


def _failure_result(
    *,
    config: LocalFrontierConfig,
    candidates: GlobalConstantJumpCandidateSet,
    context: _LocalScoreContext,
    phases: tuple[int | None, ...],
    bootstrap_replay_keys: tuple[tuple[Any, ...], ...],
    reason: str,
    stage: str,
    block_diagnostics: tuple[LocalFrontierBlockDiagnostics, ...] = (),
) -> LocalFrontierResult:
    diagnostics = _base_diagnostics(
        config=config,
        candidates=candidates,
        context=context,
        phases=phases,
        bootstrap_replay_keys=bootstrap_replay_keys,
        block_diagnostics=block_diagnostics,
        frontier_widths=tuple(block.frontier_state_count for block in block_diagnostics),
        local_bucket_width_max=max((block.bucket_width_max for block in block_diagnostics), default=0),
        ownership_records=(),
        recompute_records=(),
        selected_origin_time_ms=None,
        selected_serialized_first_start_beat=None,
        first_section_logical_beat_count=None,
        first_section_serialized_beat_count=None,
        selected_section_count=0,
        replay_fingerprint=_json_fingerprint(("fallback", reason, stage)),
        grid_fingerprint=None,
        fallback_reason=reason,
        fallback_stage=stage,
    )
    objective_diagnostics = None
    if context.record_objective_ledgers and not context.record_measurement_v2:
        objective_diagnostics = _build_objective_diagnostics(
            context=context,
            candidates=candidates,
            final_frontier=(),
            selected_path=None,
        )
    if context.record_bounded_diagnostics:
        context.bounded_diagnostics = _build_bounded_diagnostics(
            context=context,
            candidates=candidates,
            final_frontier=(),
        )
    return LocalFrontierResult(
        grid=None,
        diagnostics=diagnostics,
        reason=reason,
        objective_diagnostics=objective_diagnostics,
    )


def _base_diagnostics(
    *,
    config: LocalFrontierConfig,
    candidates: GlobalConstantJumpCandidateSet,
    context: _LocalScoreContext,
    phases: tuple[int | None, ...],
    bootstrap_replay_keys: tuple[tuple[Any, ...], ...],
    block_diagnostics: tuple[LocalFrontierBlockDiagnostics, ...],
    frontier_widths: tuple[int, ...],
    local_bucket_width_max: int,
    ownership_records: tuple[BoundaryOwnershipRecord, ...],
    recompute_records: tuple[LookaheadRecomputeRecord, ...],
    selected_origin_time_ms: float | None,
    selected_serialized_first_start_beat: int | None,
    first_section_logical_beat_count: int | None,
    first_section_serialized_beat_count: int | None,
    selected_section_count: int,
    replay_fingerprint: str,
    grid_fingerprint: str | None,
    fallback_reason: str | None,
    fallback_stage: str | None,
) -> LocalFrontierDiagnostics:
    deterministic_payload = {
        "contract": LOCAL_FRONTIER_CONTRACT_VERSION,
        "schedule": config.schedule_arm.value,
        "candidate": candidates.diagnostics.candidate_fingerprint,
        "blocks": tuple(asdict(block) for block in block_diagnostics),
        "replay": replay_fingerprint,
        "grid": grid_fingerprint,
        "fallback": (fallback_reason, fallback_stage),
    }
    return LocalFrontierDiagnostics(
        contract_version=LOCAL_FRONTIER_CONTRACT_VERSION,
        schedule_arm=config.schedule_arm.value,
        coverage_start_ms=context.coverage_start_ms,
        coverage_end_ms=context.coverage_end_ms,
        frame_count=context.frame_count,
        frame_rate_hz=context.frame_rate_hz,
        input_signal_sha256=candidates.diagnostics.input_signal_sha256,
        candidate_fingerprint=candidates.diagnostics.candidate_fingerprint,
        origin_candidate_count=len(candidates.origin_candidates),
        boundary_candidate_count=len(candidates.boundary_candidates),
        tempo_candidate_count=len(candidates.tempo_candidates),
        bootstrap_downbeat_phases=phases,
        bootstrap_state_count=len(bootstrap_replay_keys),
        bootstrap_replay_keys=bootstrap_replay_keys,
        block_count=len(block_diagnostics),
        frontier_widths=frontier_widths,
        local_bucket_width_max=local_bucket_width_max,
        block_diagnostics=block_diagnostics,
        boundary_ownership_records=ownership_records,
        lookahead_recompute_records=recompute_records,
        selected_origin_time_ms=selected_origin_time_ms,
        selected_serialized_first_start_beat=selected_serialized_first_start_beat,
        first_section_logical_beat_count=first_section_logical_beat_count,
        first_section_serialized_beat_count=first_section_serialized_beat_count,
        selected_section_count=selected_section_count,
        final_global_rescore_count=0,
        replay_fingerprint=replay_fingerprint,
        grid_fingerprint=grid_fingerprint,
        deterministic_projection_sha256=_json_fingerprint(deterministic_payload),
        fallback_reason=fallback_reason,
        fallback_stage=fallback_stage,
    )


def _owner_block_index(
    boundary_time_ms: float, blocks: Sequence[LocalFrontierBlockDiagnostics]
) -> int:
    for block in blocks:
        if block.core_start_ms <= boundary_time_ms < block.core_end_ms:
            return block.block_index
    if blocks and boundary_time_ms == blocks[-1].core_end_ms:
        return blocks[-1].block_index + 1
    return -1


def _finalize_boundary_charge_diagnostics(
    *,
    ownership_by_anchor: dict[int, BoundaryOwnershipRecord],
    recompute_records: Sequence[LookaheadRecomputeRecord],
    final_frontier: Sequence[_Path],
    selected_path: _Path,
    blocks: Sequence[LocalFrontierBlockDiagnostics],
) -> tuple[tuple[BoundaryOwnershipRecord, ...], tuple[LookaheadRecomputeRecord, ...]]:
    """Bind diagnostic charges to the retained frontier and final traceback.

    Search alternatives legitimately carry their own already-paid closure
    terms.  They are not, however, charges in the selected result.  This final
    pass makes the two denominators explicit and prevents an alternative path
    from making ``objective_charge_count`` look selected.
    """

    retained_by_anchor: dict[int, list[_SelectedBoundary]] = {}
    retained_path_count: dict[int, int] = {}
    for path in final_frontier:
        anchors_in_path: set[int] = set()
        for selected in path.selected_boundaries:
            retained_by_anchor.setdefault(selected.anchor_id, []).append(selected)
            anchors_in_path.add(selected.anchor_id)
        for anchor_id in anchors_in_path:
            retained_path_count[anchor_id] = retained_path_count.get(anchor_id, 0) + 1

    selected_by_anchor = {
        selected.anchor_id: selected for selected in selected_path.selected_boundaries
    }
    finalized_ownership: list[BoundaryOwnershipRecord] = []
    finalized_by_anchor: dict[int, BoundaryOwnershipRecord] = {}
    for anchor_id in sorted(ownership_by_anchor):
        record = ownership_by_anchor[anchor_id]
        selected = selected_by_anchor.get(anchor_id)
        retained = retained_by_anchor.get(anchor_id, ())
        representative = selected or (
            min(retained, key=_boundary_replay_item) if retained else None
        )
        authoritative_time_ms = (
            representative.boundary_time_ms
            if representative is not None
            else record.source_candidate_time_ms
        )
        selected_charge_count = 1 if selected is not None else 0
        finalized = replace(
            record,
            boundary_time_ms=authoritative_time_ms,
            owner_block_index=_owner_block_index(authoritative_time_ms, blocks),
            retained_frontier_charge_path_count=retained_path_count.get(anchor_id, 0),
            selected_traceback_objective_charge_count=selected_charge_count,
            objective_charge_count=selected_charge_count,
        )
        finalized_ownership.append(finalized)
        finalized_by_anchor[anchor_id] = finalized

    finalized_recompute: list[LookaheadRecomputeRecord] = []
    for record in recompute_records:
        ownership = finalized_by_anchor.get(record.boundary_anchor_id)
        if ownership is None:
            finalized_recompute.append(record)
            continue
        finalized_recompute.append(
            replace(
                record,
                boundary_time_ms=ownership.boundary_time_ms,
                retained_frontier_charge_path_count=(
                    ownership.retained_frontier_charge_path_count
                ),
                selected_traceback_objective_charge_count=(
                    ownership.selected_traceback_objective_charge_count
                ),
                objective_charge_count=ownership.objective_charge_count,
            )
        )
    return tuple(finalized_ownership), tuple(finalized_recompute)


def _boundary_replay_item(boundary: _SelectedBoundary) -> tuple[Any, ...]:
    return (
        boundary.anchor_id,
        float.hex(boundary.source_time_ms),
        boundary.boundary_beat,
        float.hex(boundary.boundary_time_ms),
    )


def _has_centered_downbeat_signal(signal: np.ndarray) -> bool:
    if signal.size == 0:
        return False
    centered = signal - float(np.mean(signal))
    return bool(np.linalg.norm(centered) > 0.0)


def _validate_lattice_inputs(*, origin_time_ms: float, bpm: float) -> None:
    if not math.isfinite(origin_time_ms) or not math.isfinite(bpm) or not 20.0 <= bpm <= 1000.0:
        raise ValueError("origin_time_ms must be finite and bpm must satisfy the frozen guard")


def _json_fingerprint(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    if is_dataclass(payload):
        payload = asdict(payload)
    elif isinstance(payload, tuple):
        payload = tuple(asdict(item) if is_dataclass(item) else item for item in payload)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
