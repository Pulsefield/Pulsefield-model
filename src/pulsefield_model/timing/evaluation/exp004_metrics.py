from __future__ import annotations

import bisect
import hashlib
import importlib
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pulsefield_model.timing.canonicalization import (
    TIMING_CANONICALIZATION_BPM_80_160,
    TIMING_CANONICALIZATION_NONE,
    canonical_bpm_80_160,
)
from pulsefield_model.timing.schema import FittedTimingGrid, TimingSegment
from pulsefield_model.timing.v3.schema import TimingV3Grid


PHASE_SAMPLING_V1 = "phase_sampling_v1"
EXP004_METRICS_SCHEMA = "pulsefield_model.timing_v3_exp004_metrics_v1"
WEAK_BOUNDARY_LOG2_RATIO_THRESHOLD = math.log2(1.005)
WEAK_BOUNDARY_MAX_TOLERANCE_MS = 750.0
PHASE_SAMPLING_HOP_MS = 20.0
WEAK_BPM_ALIAS_MULTIPLIERS = (0.25, 1.0 / 3.0, 0.5, 1.0, 2.0, 3.0, 4.0)

GUARD_PASS = "pass"
GUARD_AMBIGUOUS = "ambiguous"
GUARD_KILL = "kill"
GUARD_NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class TimingBoundary:
    time_ms: float
    left_period_ms: float
    right_period_ms: float
    left_bpm: float
    right_bpm: float
    left_index: int
    right_index: int

    def __post_init__(self) -> None:
        for name in (
            "time_ms",
            "left_period_ms",
            "right_period_ms",
            "left_bpm",
            "right_bpm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.left_period_ms <= 0.0 or self.right_period_ms <= 0.0:
            raise ValueError("boundary periods must be positive")
        if self.left_bpm <= 0.0 or self.right_bpm <= 0.0:
            raise ValueError("boundary BPMs must be positive")
        if self.left_index < 0 or self.right_index <= self.left_index:
            raise ValueError("boundary indices must be ordered")

    @property
    def tempo_log2_ratio(self) -> float:
        return float(math.log2(self.right_bpm / self.left_bpm))


@dataclass(frozen=True)
class WeakRedlineBoundaryExtraction:
    valid_comparator: bool
    boundaries: tuple[TimingBoundary, ...]
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        boundaries = tuple(self.boundaries)
        if self.valid_comparator and self.rejection_reason is not None:
            raise ValueError("valid comparator cannot have a rejection reason")
        if not self.valid_comparator and self.rejection_reason is None:
            raise ValueError("invalid comparator must have a rejection reason")
        object.__setattr__(self, "boundaries", boundaries)

    @property
    def weak_redline_boundary_count(self) -> int:
        return len(self.boundaries)


@dataclass(frozen=True)
class WeakBoundaryMatch:
    predicted: TimingBoundary
    weak_redline: TimingBoundary
    signed_error_ms: float
    abs_error_ms: float
    tolerance_ms: float

    def __post_init__(self) -> None:
        for name in ("signed_error_ms", "abs_error_ms", "tolerance_ms"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.abs_error_ms < 0.0 or self.tolerance_ms < 0.0:
            raise ValueError("match errors and tolerance must be non-negative")


@dataclass(frozen=True)
class WeakBoundaryMatchSummary:
    matches: tuple[WeakBoundaryMatch, ...]
    unmatched_predicted_boundaries: tuple[TimingBoundary, ...]
    unmatched_weak_redline_boundaries: tuple[TimingBoundary, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "matches", tuple(self.matches))
        object.__setattr__(
            self,
            "unmatched_predicted_boundaries",
            tuple(self.unmatched_predicted_boundaries),
        )
        object.__setattr__(
            self,
            "unmatched_weak_redline_boundaries",
            tuple(self.unmatched_weak_redline_boundaries),
        )

    @property
    def matched_count(self) -> int:
        return len(self.matches)

    @property
    def predicted_boundary_count(self) -> int:
        return self.matched_count + len(self.unmatched_predicted_boundaries)

    @property
    def weak_redline_boundary_count(self) -> int:
        return self.matched_count + len(self.unmatched_weak_redline_boundaries)

    @property
    def unmatched_predicted_boundary_count(self) -> int:
        return len(self.unmatched_predicted_boundaries)

    @property
    def unmatched_weak_redline_boundary_count(self) -> int:
        return len(self.unmatched_weak_redline_boundaries)

    @property
    def matched_signed_errors_ms(self) -> tuple[float, ...]:
        return tuple(match.signed_error_ms for match in self.matches)

    @property
    def matched_abs_errors_ms(self) -> tuple[float, ...]:
        return tuple(match.abs_error_ms for match in self.matches)

    @property
    def weak_boundary_match_rate(self) -> float | None:
        if self.weak_redline_boundary_count == 0:
            return None
        return float(self.matched_count / self.weak_redline_boundary_count)

    @property
    def predicted_boundary_match_rate(self) -> float | None:
        if self.predicted_boundary_count == 0:
            return None
        return float(self.matched_count / self.predicted_boundary_count)


@dataclass(frozen=True)
class WeakBoundaryDifficultyMetrics:
    audio_key: str
    difficulty_key: str
    valid_comparator: bool
    match_summary: WeakBoundaryMatchSummary | None
    rejection_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.audio_key:
            raise ValueError("audio_key must be non-empty")
        if not self.difficulty_key:
            raise ValueError("difficulty_key must be non-empty")
        if self.valid_comparator and self.match_summary is None:
            raise ValueError("valid comparator requires a match summary")
        if self.valid_comparator and self.rejection_reason is not None:
            raise ValueError("valid comparator cannot have a rejection reason")
        if not self.valid_comparator and self.rejection_reason is None:
            raise ValueError("invalid comparator requires a rejection reason")

    @property
    def weak_boundary_matched_count(self) -> int:
        return 0 if self.match_summary is None else self.match_summary.matched_count

    @property
    def weak_boundary_predicted_count(self) -> int:
        return 0 if self.match_summary is None else self.match_summary.predicted_boundary_count

    @property
    def weak_boundary_redline_count(self) -> int:
        return 0 if self.match_summary is None else self.match_summary.weak_redline_boundary_count

    @property
    def weak_boundary_unmatched_predicted_count(self) -> int:
        return 0 if self.match_summary is None else self.match_summary.unmatched_predicted_boundary_count

    @property
    def weak_boundary_unmatched_redline_count(self) -> int:
        return 0 if self.match_summary is None else self.match_summary.unmatched_weak_redline_boundary_count

    @property
    def weak_boundary_mean_signed_error_ms(self) -> float | None:
        if self.match_summary is None or not self.match_summary.matches:
            return None
        return float(statistics.fmean(self.match_summary.matched_signed_errors_ms))

    @property
    def weak_boundary_mean_abs_error_ms(self) -> float | None:
        if self.match_summary is None or not self.match_summary.matches:
            return None
        return float(statistics.fmean(self.match_summary.matched_abs_errors_ms))

    @property
    def weak_boundary_match_rate(self) -> float | None:
        return None if self.match_summary is None else self.match_summary.weak_boundary_match_rate

    @property
    def predicted_boundary_match_rate(self) -> float | None:
        return None if self.match_summary is None else self.match_summary.predicted_boundary_match_rate


@dataclass(frozen=True)
class WeakConsensusBoundary:
    predicted_boundary: TimingBoundary
    matched_valid_difficulty_count: int
    required_valid_difficulty_count: int
    weak_consensus_supported: bool


@dataclass(frozen=True)
class WeakBoundaryAudioMetrics:
    audio_key: str
    valid_difficulty_count: int
    invalid_difficulty_count: int
    weak_boundary_matched_count: int
    weak_boundary_predicted_count: int
    weak_boundary_redline_count: int
    weak_boundary_unmatched_predicted_count: int
    weak_boundary_unmatched_redline_count: int
    weak_boundary_matched_count_per_difficulty_mean: float
    weak_boundary_predicted_count_per_difficulty_mean: float
    weak_boundary_redline_count_per_difficulty_mean: float
    weak_boundary_unmatched_predicted_count_per_difficulty_mean: float
    weak_boundary_unmatched_redline_count_per_difficulty_mean: float
    weak_boundary_mean_signed_error_median_ms: float | None
    weak_boundary_mean_abs_error_median_ms: float | None
    weak_boundary_match_rate_median: float | None
    predicted_boundary_match_rate_median: float | None
    weak_consensus_boundaries: tuple[WeakConsensusBoundary, ...]

    @property
    def weak_consensus_supported_boundary_count(self) -> int:
        return sum(1 for item in self.weak_consensus_boundaries if item.weak_consensus_supported)


@dataclass(frozen=True)
class CanonicalBpmBinding:
    canonicalization: str
    function_module: str
    function_qualname: str
    source_path: str
    source_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActiveSectionSignatureSpan:
    """A clipped half-open active-section span compared with exact float equality.

    Exp004 intentionally does not apply tolerance here: all values must be
    finite, and two signatures are equal only when their clipped interval
    boundaries and canonical active period/BPM values are exactly equal.
    """

    start_time_ms: float
    end_time_ms: float
    beat_length_ms: float
    bpm: float

    def __post_init__(self) -> None:
        for name in ("start_time_ms", "end_time_ms", "beat_length_ms", "bpm"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.end_time_ms <= self.start_time_ms:
            raise ValueError("end_time_ms must be greater than start_time_ms")
        if self.beat_length_ms <= 0.0 or self.bpm <= 0.0:
            raise ValueError("active section period and BPM must be positive")


@dataclass(frozen=True)
class PhaseSamplingV1Comparison:
    schema: str
    sampling_version: str
    canonicalization: str
    coverage_start_ms: float
    coverage_end_ms: float
    sample_hop_ms: float
    sample_count: int
    phase_mean_beats: float
    phase_p50_beats: float
    phase_p90_beats: float
    phase_max_beats: float
    phase_mean_ms: float
    phase_p50_ms: float
    phase_p90_ms: float
    phase_max_ms: float
    local_bpm_mae: float
    local_bpm_p90_abs_error: float
    local_bpm_alias_mae: float
    local_bpm_alias_p90_abs_error: float
    initial_signed_phase_error_beats: float
    initial_signed_phase_error_ms: float
    endpoint_relative_drift_beats: float
    endpoint_relative_drift_ms: float
    max_abs_prefix_relative_drift_beats: float
    max_abs_prefix_relative_drift_ms: float
    drift_slope_beats_per_minute: float
    drift_slope_ms_per_minute: float
    p90_abs_30s_relative_drift_ms: float
    p90_abs_60s_relative_drift_ms: float
    predicted_active_section_signature: tuple[ActiveSectionSignatureSpan, ...]
    reference_active_section_signature: tuple[ActiveSectionSignatureSpan, ...]
    active_section_signature_equal: bool
    active_section_disagreement_sample_count: int
    active_section_disagreement_fraction: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "predicted_active_section_signature",
            tuple(self.predicted_active_section_signature),
        )
        object.__setattr__(
            self,
            "reference_active_section_signature",
            tuple(self.reference_active_section_signature),
        )
        if self.schema != EXP004_METRICS_SCHEMA:
            raise ValueError(f"schema must be {EXP004_METRICS_SCHEMA!r}")
        if self.sampling_version != PHASE_SAMPLING_V1:
            raise ValueError(f"sampling_version must be {PHASE_SAMPLING_V1!r}")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if self.active_section_signature_equal != (
            self.predicted_active_section_signature == self.reference_active_section_signature
        ):
            raise ValueError("active_section_signature_equal is inconsistent with signatures")
        for name, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class Exp004DenominatorRow:
    audio_key: str
    cache_valid: bool
    projection_evaluable: bool
    comparison_eligible: bool
    pure_cj3_grid_produced: bool
    pure_cj3_phase_matched: bool
    current_v2_phase_matched: bool
    selected_safety_phase_scored: bool
    selected_used_fallback: bool

    def __post_init__(self) -> None:
        if not self.audio_key:
            raise ValueError("audio_key must be non-empty")
        for name in (
            "cache_valid",
            "projection_evaluable",
            "comparison_eligible",
            "pure_cj3_grid_produced",
            "pure_cj3_phase_matched",
            "current_v2_phase_matched",
            "selected_safety_phase_scored",
            "selected_used_fallback",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        if self.projection_evaluable and not self.cache_valid:
            raise ValueError("projection_evaluable requires cache_valid")
        if self.comparison_eligible and not self.projection_evaluable:
            raise ValueError("comparison_eligible requires projection_evaluable")
        if self.pure_cj3_grid_produced and not self.projection_evaluable:
            raise ValueError("pure_cj3_grid_produced requires projection_evaluable")
        if self.current_v2_phase_matched and not self.comparison_eligible:
            raise ValueError("current_v2_phase_matched requires comparison_eligible")
        if self.pure_cj3_phase_matched and not self.comparison_eligible:
            raise ValueError("pure_cj3_phase_matched requires comparison_eligible")
        if self.pure_cj3_phase_matched and not self.pure_cj3_grid_produced:
            raise ValueError("pure_cj3_phase_matched requires pure_cj3_grid_produced")
        if self.pure_cj3_phase_matched and not self.current_v2_phase_matched:
            raise ValueError("pure_cj3_phase_matched requires current_v2_phase_matched")
        if self.selected_safety_phase_scored and not self.comparison_eligible:
            raise ValueError("selected_safety_phase_scored requires comparison_eligible")
        if self.selected_used_fallback and not self.projection_evaluable:
            raise ValueError("selected_used_fallback requires projection_evaluable")
        if self.selected_used_fallback and self.pure_cj3_grid_produced:
            raise ValueError("selected_used_fallback excludes pure_cj3_grid_produced")
        if self.selected_used_fallback and self.pure_cj3_phase_matched:
            raise ValueError("selected_used_fallback excludes pure_cj3_phase_matched")
        if (
            self.selected_used_fallback
            and self.comparison_eligible
            and not self.selected_safety_phase_scored
        ):
            raise ValueError(
                "comparison-eligible selected_used_fallback requires selected_safety_phase_scored"
            )
        if (
            self.selected_used_fallback
            and self.comparison_eligible
            and not self.current_v2_phase_matched
        ):
            raise ValueError(
                "comparison-eligible selected_used_fallback requires current_v2_phase_matched"
            )
        if self.selected_safety_phase_scored and not self.selected_used_fallback and not self.pure_cj3_phase_matched:
            raise ValueError(
                "selected_safety_phase_scored without fallback requires pure_cj3_phase_matched"
            )


@dataclass(frozen=True)
class Exp004DenominatorSummary:
    stage_audio_count: int
    cache_valid_count: int
    projection_evaluable_count: int
    comparison_eligible_count: int
    pure_CJ3_phase_count: int
    pure_CJ3_phase_coverage: float | None
    selected_safety_phase_count: int
    selected_fallback_count: int
    selected_fallback_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Exp004GuardClassification:
    mean_phase_ratio: str
    p90_phase_ratio: str
    pure_CJ3_phase_coverage: str
    selected_fallback_rate: str
    overall: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _GridPoint:
    offset_ms: float
    beat_length_ms: float
    meter: int = 4


@dataclass(frozen=True)
class _SectionedGrid:
    offsets_ms: tuple[float, ...]
    periods_ms: tuple[float, ...]
    cumulative_start_beats: tuple[float, ...]

    def __post_init__(self) -> None:
        if not (
            len(self.offsets_ms)
            == len(self.periods_ms)
            == len(self.cumulative_start_beats)
        ):
            raise ValueError("sectioned grid arrays must have equal lengths")
        if not self.offsets_ms:
            raise ValueError("sectioned grid must be non-empty")

    def active_index(self, time_ms: float) -> int:
        index = bisect.bisect_right(self.offsets_ms, time_ms) - 1
        return min(max(index, 0), len(self.offsets_ms) - 1)

    def active_index_left_limit(self, time_ms: float) -> int:
        index = bisect.bisect_left(self.offsets_ms, time_ms) - 1
        return min(max(index, 0), len(self.offsets_ms) - 1)

    def active_period_ms(self, time_ms: float) -> float:
        return self.periods_ms[self.active_index(time_ms)]

    def active_period_ms_left_limit(self, time_ms: float) -> float:
        return self.periods_ms[self.active_index_left_limit(time_ms)]

    def active_bpm(self, time_ms: float) -> float:
        return 60000.0 / self.active_period_ms(time_ms)

    def beat_at_time(self, time_ms: float) -> float:
        index = self.active_index(time_ms)
        return (
            self.cumulative_start_beats[index]
            + (time_ms - self.offsets_ms[index]) / self.periods_ms[index]
        )

    def beat_at_time_left_limit(self, time_ms: float) -> float:
        index = self.active_index_left_limit(time_ms)
        return (
            self.cumulative_start_beats[index]
            + (time_ms - self.offsets_ms[index]) / self.periods_ms[index]
        )


def extract_weak_redline_boundaries(
    redline_grid: FittedTimingGrid | Sequence[Any],
    *,
    coverage_start_ms: float,
    coverage_end_ms: float,
) -> WeakRedlineBoundaryExtraction:
    points, rejection_reason = _validated_timing_points(redline_grid)
    if rejection_reason is not None:
        return WeakRedlineBoundaryExtraction(
            valid_comparator=False,
            boundaries=(),
            rejection_reason=rejection_reason,
        )
    return WeakRedlineBoundaryExtraction(
        valid_comparator=True,
        boundaries=extract_tempo_change_boundaries(
            points,
            coverage_start_ms=coverage_start_ms,
            coverage_end_ms=coverage_end_ms,
        ),
    )


def extract_tempo_change_boundaries(
    timing_grid: TimingV3Grid | FittedTimingGrid | Sequence[Any],
    *,
    coverage_start_ms: float,
    coverage_end_ms: float,
    log2_ratio_threshold: float = WEAK_BOUNDARY_LOG2_RATIO_THRESHOLD,
) -> tuple[TimingBoundary, ...]:
    coverage_start_ms = _finite_float(coverage_start_ms, "coverage_start_ms")
    coverage_end_ms = _finite_float(coverage_end_ms, "coverage_end_ms")
    if coverage_end_ms <= coverage_start_ms:
        raise ValueError("coverage_end_ms must be greater than coverage_start_ms")
    if log2_ratio_threshold < 0.0 or not math.isfinite(log2_ratio_threshold):
        raise ValueError("log2_ratio_threshold must be finite and non-negative")

    points, rejection_reason = _validated_timing_points(timing_grid)
    if rejection_reason is not None:
        raise ValueError(rejection_reason)

    boundaries: list[TimingBoundary] = []
    for right_index in range(1, len(points)):
        left = points[right_index - 1]
        right = points[right_index]
        time_ms = right.offset_ms
        if time_ms < coverage_start_ms or time_ms >= coverage_end_ms:
            continue
        left_bpm = 60000.0 / left.beat_length_ms
        right_bpm = 60000.0 / right.beat_length_ms
        if abs(math.log2(right_bpm / left_bpm)) < log2_ratio_threshold:
            continue
        boundaries.append(
            TimingBoundary(
                time_ms=time_ms,
                left_period_ms=left.beat_length_ms,
                right_period_ms=right.beat_length_ms,
                left_bpm=left_bpm,
                right_bpm=right_bpm,
                left_index=right_index - 1,
                right_index=right_index,
            )
        )
    return tuple(boundaries)


def match_weak_boundaries(
    predicted_boundaries: Sequence[TimingBoundary],
    weak_redline_boundaries: Sequence[TimingBoundary],
) -> WeakBoundaryMatchSummary:
    predicted = tuple(predicted_boundaries)
    weak = tuple(weak_redline_boundaries)
    candidate_pairs: list[tuple[float, float, float, int, int, float]] = []
    for predicted_index, predicted_boundary in enumerate(predicted):
        for weak_index, weak_boundary in enumerate(weak):
            signed_error_ms = predicted_boundary.time_ms - weak_boundary.time_ms
            abs_error_ms = abs(signed_error_ms)
            tolerance_ms = weak_boundary_match_tolerance_ms(
                predicted_boundary,
                weak_boundary,
            )
            if abs_error_ms <= tolerance_ms:
                candidate_pairs.append(
                    (
                        abs_error_ms,
                        predicted_boundary.time_ms,
                        weak_boundary.time_ms,
                        predicted_index,
                        weak_index,
                        tolerance_ms,
                    )
                )

    matched_predicted: set[int] = set()
    matched_weak: set[int] = set()
    matches: list[WeakBoundaryMatch] = []
    for abs_error_ms, _, _, predicted_index, weak_index, tolerance_ms in sorted(candidate_pairs):
        if predicted_index in matched_predicted or weak_index in matched_weak:
            continue
        predicted_boundary = predicted[predicted_index]
        weak_boundary = weak[weak_index]
        matched_predicted.add(predicted_index)
        matched_weak.add(weak_index)
        matches.append(
            WeakBoundaryMatch(
                predicted=predicted_boundary,
                weak_redline=weak_boundary,
                signed_error_ms=predicted_boundary.time_ms - weak_boundary.time_ms,
                abs_error_ms=abs_error_ms,
                tolerance_ms=tolerance_ms,
            )
        )

    return WeakBoundaryMatchSummary(
        matches=tuple(matches),
        unmatched_predicted_boundaries=tuple(
            boundary for index, boundary in enumerate(predicted) if index not in matched_predicted
        ),
        unmatched_weak_redline_boundaries=tuple(
            boundary for index, boundary in enumerate(weak) if index not in matched_weak
        ),
    )


def weak_boundary_match_tolerance_ms(
    predicted_boundary: TimingBoundary,
    weak_redline_boundary: TimingBoundary,
) -> float:
    return float(
        min(
            WEAK_BOUNDARY_MAX_TOLERANCE_MS,
            0.5
            * min(
                predicted_boundary.left_period_ms,
                predicted_boundary.right_period_ms,
                weak_redline_boundary.left_period_ms,
                weak_redline_boundary.right_period_ms,
            ),
        )
    )


def weak_boundary_difficulty_metrics(
    *,
    audio_key: str,
    difficulty_key: str,
    predicted_boundaries: Sequence[TimingBoundary],
    weak_redline_extraction: WeakRedlineBoundaryExtraction,
) -> WeakBoundaryDifficultyMetrics:
    if not weak_redline_extraction.valid_comparator:
        return WeakBoundaryDifficultyMetrics(
            audio_key=audio_key,
            difficulty_key=difficulty_key,
            valid_comparator=False,
            match_summary=None,
            rejection_reason=weak_redline_extraction.rejection_reason,
        )
    return WeakBoundaryDifficultyMetrics(
        audio_key=audio_key,
        difficulty_key=difficulty_key,
        valid_comparator=True,
        match_summary=match_weak_boundaries(
            predicted_boundaries,
            weak_redline_extraction.boundaries,
        ),
    )


def aggregate_weak_boundary_audio_metrics(
    difficulty_metrics: Sequence[WeakBoundaryDifficultyMetrics],
) -> tuple[WeakBoundaryAudioMetrics, ...]:
    by_audio: dict[str, list[WeakBoundaryDifficultyMetrics]] = defaultdict(list)
    for metrics in difficulty_metrics:
        by_audio[metrics.audio_key].append(metrics)

    results: list[WeakBoundaryAudioMetrics] = []
    for audio_key in sorted(by_audio):
        rows = sorted(by_audio[audio_key], key=lambda row: row.difficulty_key)
        valid_rows = [row for row in rows if row.valid_comparator]
        invalid_count = len(rows) - len(valid_rows)
        valid_count = len(valid_rows)

        matched_count = sum(row.weak_boundary_matched_count for row in valid_rows)
        predicted_count = sum(row.weak_boundary_predicted_count for row in valid_rows)
        redline_count = sum(row.weak_boundary_redline_count for row in valid_rows)
        unmatched_predicted_count = sum(
            row.weak_boundary_unmatched_predicted_count for row in valid_rows
        )
        unmatched_redline_count = sum(
            row.weak_boundary_unmatched_redline_count for row in valid_rows
        )
        consensus = _weak_consensus_boundaries(valid_rows)

        results.append(
            WeakBoundaryAudioMetrics(
                audio_key=audio_key,
                valid_difficulty_count=valid_count,
                invalid_difficulty_count=invalid_count,
                weak_boundary_matched_count=matched_count,
                weak_boundary_predicted_count=predicted_count,
                weak_boundary_redline_count=redline_count,
                weak_boundary_unmatched_predicted_count=unmatched_predicted_count,
                weak_boundary_unmatched_redline_count=unmatched_redline_count,
                weak_boundary_matched_count_per_difficulty_mean=_mean_count(
                    matched_count,
                    valid_count,
                ),
                weak_boundary_predicted_count_per_difficulty_mean=_mean_count(
                    predicted_count,
                    valid_count,
                ),
                weak_boundary_redline_count_per_difficulty_mean=_mean_count(
                    redline_count,
                    valid_count,
                ),
                weak_boundary_unmatched_predicted_count_per_difficulty_mean=_mean_count(
                    unmatched_predicted_count,
                    valid_count,
                ),
                weak_boundary_unmatched_redline_count_per_difficulty_mean=_mean_count(
                    unmatched_redline_count,
                    valid_count,
                ),
                weak_boundary_mean_signed_error_median_ms=_median_optional(
                    row.weak_boundary_mean_signed_error_ms for row in valid_rows
                ),
                weak_boundary_mean_abs_error_median_ms=_median_optional(
                    row.weak_boundary_mean_abs_error_ms for row in valid_rows
                ),
                weak_boundary_match_rate_median=_median_optional(
                    row.weak_boundary_match_rate for row in valid_rows
                ),
                predicted_boundary_match_rate_median=_median_optional(
                    row.predicted_boundary_match_rate for row in valid_rows
                ),
                weak_consensus_boundaries=consensus,
            )
        )
    return tuple(results)


def alias_aware_bpm_error(predicted_bpm: float, weak_reference_bpm: float) -> float:
    predicted_bpm = _positive_finite_float(predicted_bpm, "predicted_bpm")
    weak_reference_bpm = _positive_finite_float(weak_reference_bpm, "weak_reference_bpm")
    candidates = [
        abs(predicted_bpm * multiplier - weak_reference_bpm)
        for multiplier in WEAK_BPM_ALIAS_MULTIPLIERS
        if 20.0 <= predicted_bpm * multiplier <= 1000.0
    ]
    if not candidates:
        raise ValueError("predicted BPM has no valid alias in [20, 1000]")
    return float(min(candidates))


def canonical_bpm_80_160_for_exp004(bpm: float) -> float:
    return canonical_bpm_80_160(bpm)


def canonical_bpm_binding_for_exp004() -> CanonicalBpmBinding:
    module_name = canonical_bpm_80_160.__module__
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise ValueError(f"module {module_name!r} has no source file")
    path = Path(module_file).expanduser().resolve(strict=False)
    return CanonicalBpmBinding(
        canonicalization=TIMING_CANONICALIZATION_BPM_80_160,
        function_module=module_name,
        function_qualname=canonical_bpm_80_160.__qualname__,
        source_path=path.as_posix(),
        source_sha256=_file_sha256(path),
    )


def phase_sample_times_v1(
    *,
    coverage_start_ms: float,
    coverage_end_ms: float,
    sample_hop_ms: float = PHASE_SAMPLING_HOP_MS,
) -> tuple[float, ...]:
    coverage_start_ms = _finite_float(coverage_start_ms, "coverage_start_ms")
    coverage_end_ms = _finite_float(coverage_end_ms, "coverage_end_ms")
    sample_hop_ms = _positive_finite_float(sample_hop_ms, "sample_hop_ms")
    if coverage_end_ms <= coverage_start_ms:
        raise ValueError("coverage_end_ms must be greater than coverage_start_ms")

    first_k = math.ceil(coverage_start_ms / sample_hop_ms)
    times: list[float] = []
    k = first_k
    while True:
        time_ms = float(sample_hop_ms * k)
        if time_ms >= coverage_end_ms:
            break
        if time_ms >= coverage_start_ms:
            times.append(time_ms)
        k += 1
    return tuple(times)


def active_section_signature_v1(
    grid: TimingV3Grid | FittedTimingGrid,
    *,
    coverage_start_ms: float,
    coverage_end_ms: float,
    canonicalization: str = TIMING_CANONICALIZATION_NONE,
) -> tuple[ActiveSectionSignatureSpan, ...]:
    """Return Exp004's deterministic clipped active-section signature.

    The signature is clipped to the half-open coverage interval and records
    exact finite float values after optional canonical BPM-80-160 conversion.
    Equality is plain tuple/dataclass equality; no tolerance, rounding, or
    ordinal section-index comparison is applied.
    """

    coverage_start_ms = _finite_float(coverage_start_ms, "coverage_start_ms")
    coverage_end_ms = _finite_float(coverage_end_ms, "coverage_end_ms")
    if coverage_end_ms <= coverage_start_ms:
        raise ValueError("coverage_end_ms must be greater than coverage_start_ms")
    _validate_grid_coverage(grid, coverage_start_ms, coverage_end_ms, "grid")

    sectioned = _sectioned_grid(grid, canonicalization=canonicalization)
    interval_boundaries = [coverage_start_ms]
    interval_boundaries.extend(
        offset
        for offset in sectioned.offsets_ms
        if coverage_start_ms < offset < coverage_end_ms
    )
    interval_boundaries.append(coverage_end_ms)

    spans: list[ActiveSectionSignatureSpan] = []
    for start_time_ms, end_time_ms in zip(interval_boundaries, interval_boundaries[1:]):
        period_ms = sectioned.active_period_ms(start_time_ms)
        spans.append(
            ActiveSectionSignatureSpan(
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                beat_length_ms=period_ms,
                bpm=60000.0 / period_ms,
            )
        )
    return tuple(spans)


def compare_phase_sampling_v1(
    predicted_grid: TimingV3Grid | FittedTimingGrid,
    reference_grid: TimingV3Grid | FittedTimingGrid,
    *,
    coverage_start_ms: float,
    coverage_end_ms: float,
    canonicalization: str = TIMING_CANONICALIZATION_NONE,
) -> PhaseSamplingV1Comparison:
    coverage_start_ms = _finite_float(coverage_start_ms, "coverage_start_ms")
    coverage_end_ms = _finite_float(coverage_end_ms, "coverage_end_ms")
    if canonicalization not in (
        TIMING_CANONICALIZATION_NONE,
        TIMING_CANONICALIZATION_BPM_80_160,
    ):
        raise ValueError(
            "canonicalization must be "
            f"{TIMING_CANONICALIZATION_NONE!r} or {TIMING_CANONICALIZATION_BPM_80_160!r}"
        )

    _validate_grid_coverage(predicted_grid, coverage_start_ms, coverage_end_ms, "predicted_grid")
    _validate_grid_coverage(reference_grid, coverage_start_ms, coverage_end_ms, "reference_grid")
    sample_times_ms = phase_sample_times_v1(
        coverage_start_ms=coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
    )
    if not sample_times_ms:
        raise ValueError("phase_sampling_v1 sample partition is empty")

    predicted = _sectioned_grid(predicted_grid, canonicalization=canonicalization)
    reference = _sectioned_grid(reference_grid, canonicalization=canonicalization)
    predicted_signature = active_section_signature_v1(
        predicted_grid,
        coverage_start_ms=coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
        canonicalization=canonicalization,
    )
    reference_signature = active_section_signature_v1(
        reference_grid,
        coverage_start_ms=coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
        canonicalization=canonicalization,
    )
    times = np.asarray(sample_times_ms, dtype=np.float64)

    predicted_beats = np.asarray([predicted.beat_at_time(time_ms) for time_ms in times], dtype=np.float64)
    reference_beats = np.asarray([reference.beat_at_time(time_ms) for time_ms in times], dtype=np.float64)
    reference_periods_ms = np.asarray([reference.active_period_ms(time_ms) for time_ms in times], dtype=np.float64)

    signed_phase_errors = ((predicted_beats - reference_beats + 0.5) % 1.0) - 0.5
    phase_errors_beats = np.abs(signed_phase_errors)
    phase_errors_ms = phase_errors_beats * reference_periods_ms

    beat_errors = predicted_beats - reference_beats
    sample_relative_drift_beats = beat_errors - beat_errors[0]
    sample_relative_drift_ms = sample_relative_drift_beats * reference_periods_ms
    endpoint_beat_error = (
        predicted.beat_at_time_left_limit(coverage_end_ms)
        - reference.beat_at_time_left_limit(coverage_end_ms)
    )
    endpoint_relative_drift_beats = float(endpoint_beat_error - beat_errors[0])
    endpoint_relative_drift_ms = float(
        endpoint_relative_drift_beats
        * reference.active_period_ms_left_limit(coverage_end_ms)
    )
    drift_relative_beats = np.append(
        sample_relative_drift_beats,
        endpoint_relative_drift_beats,
    )
    drift_relative_ms = np.append(
        sample_relative_drift_ms,
        endpoint_relative_drift_ms,
    )
    reference_bpms = 60000.0 / reference_periods_ms
    predicted_bpms = np.asarray([predicted.active_bpm(time_ms) for time_ms in times], dtype=np.float64)
    bpm_abs_errors = np.abs(predicted_bpms - reference_bpms)
    alias_bpm_abs_errors = np.asarray(
        [
            alias_aware_bpm_error(predicted_bpm, reference_bpm)
            for predicted_bpm, reference_bpm in zip(predicted_bpms, reference_bpms)
        ],
        dtype=np.float64,
    )

    active_section_disagreement_count = _active_section_signature_disagreement_sample_count(
        predicted_signature,
        reference_signature,
        sample_times_ms,
    )
    duration_minutes = (times - times[0]) / 60000.0

    return PhaseSamplingV1Comparison(
        schema=EXP004_METRICS_SCHEMA,
        sampling_version=PHASE_SAMPLING_V1,
        canonicalization=canonicalization,
        coverage_start_ms=coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
        sample_hop_ms=PHASE_SAMPLING_HOP_MS,
        sample_count=len(sample_times_ms),
        phase_mean_beats=_np_mean(phase_errors_beats),
        phase_p50_beats=_np_percentile(phase_errors_beats, 50.0),
        phase_p90_beats=_np_percentile(phase_errors_beats, 90.0),
        phase_max_beats=_np_max(phase_errors_beats),
        phase_mean_ms=_np_mean(phase_errors_ms),
        phase_p50_ms=_np_percentile(phase_errors_ms, 50.0),
        phase_p90_ms=_np_percentile(phase_errors_ms, 90.0),
        phase_max_ms=_np_max(phase_errors_ms),
        local_bpm_mae=_np_mean(bpm_abs_errors),
        local_bpm_p90_abs_error=_np_percentile(bpm_abs_errors, 90.0),
        local_bpm_alias_mae=_np_mean(alias_bpm_abs_errors),
        local_bpm_alias_p90_abs_error=_np_percentile(alias_bpm_abs_errors, 90.0),
        initial_signed_phase_error_beats=float(signed_phase_errors[0]),
        initial_signed_phase_error_ms=float(signed_phase_errors[0] * reference_periods_ms[0]),
        endpoint_relative_drift_beats=endpoint_relative_drift_beats,
        endpoint_relative_drift_ms=endpoint_relative_drift_ms,
        max_abs_prefix_relative_drift_beats=_np_max(np.abs(drift_relative_beats)),
        max_abs_prefix_relative_drift_ms=_np_max(np.abs(drift_relative_ms)),
        drift_slope_beats_per_minute=_linear_slope(duration_minutes, sample_relative_drift_beats),
        drift_slope_ms_per_minute=_linear_slope(duration_minutes, sample_relative_drift_ms),
        p90_abs_30s_relative_drift_ms=_window_p90_abs_relative_drift_ms(
            beat_errors,
            reference_periods_ms,
            window_seconds=30.0,
        ),
        p90_abs_60s_relative_drift_ms=_window_p90_abs_relative_drift_ms(
            beat_errors,
            reference_periods_ms,
            window_seconds=60.0,
        ),
        predicted_active_section_signature=predicted_signature,
        reference_active_section_signature=reference_signature,
        active_section_signature_equal=predicted_signature == reference_signature,
        active_section_disagreement_sample_count=active_section_disagreement_count,
        active_section_disagreement_fraction=float(
            active_section_disagreement_count / len(sample_times_ms)
        ),
    )


def classify_exp004_denominators(
    rows: Sequence[Exp004DenominatorRow],
) -> Exp004DenominatorSummary:
    rows = tuple(rows)
    audio_keys = [row.audio_key for row in rows]
    if len(audio_keys) != len(set(audio_keys)):
        raise ValueError("Exp004 denominator rows must be unique by audio_key")

    stage_audio_count = len(rows)
    cache_valid_count = sum(row.cache_valid for row in rows)
    projection_evaluable_count = sum(row.projection_evaluable for row in rows)
    comparison_eligible_count = sum(row.comparison_eligible for row in rows)
    pure_cj3_phase_count = sum(
        row.comparison_eligible
        and row.pure_cj3_grid_produced
        and row.pure_cj3_phase_matched
        and row.current_v2_phase_matched
        for row in rows
    )
    selected_safety_phase_count = sum(
        row.comparison_eligible and row.selected_safety_phase_scored for row in rows
    )
    selected_fallback_count = sum(row.selected_used_fallback for row in rows)
    return Exp004DenominatorSummary(
        stage_audio_count=stage_audio_count,
        cache_valid_count=cache_valid_count,
        projection_evaluable_count=projection_evaluable_count,
        comparison_eligible_count=comparison_eligible_count,
        pure_CJ3_phase_count=pure_cj3_phase_count,
        pure_CJ3_phase_coverage=_rate(pure_cj3_phase_count, comparison_eligible_count),
        selected_safety_phase_count=selected_safety_phase_count,
        selected_fallback_count=selected_fallback_count,
        selected_fallback_rate=_rate(selected_fallback_count, projection_evaluable_count),
    )


def classify_exp004_primary_guards(
    *,
    denominators: Exp004DenominatorSummary,
    mean_phase_ratio: float | None,
    p90_phase_ratio: float | None,
) -> Exp004GuardClassification:
    mean_status = classify_max_guard(mean_phase_ratio, pass_max=1.05, ambiguous_max=1.10)
    p90_status = classify_max_guard(p90_phase_ratio, pass_max=1.10, ambiguous_max=1.15)
    coverage_status = classify_min_guard(
        denominators.pure_CJ3_phase_coverage,
        pass_min=0.95,
        ambiguous_min=0.90,
    )
    fallback_status = classify_max_guard(
        denominators.selected_fallback_rate,
        pass_max=0.05,
        ambiguous_max=0.10,
    )
    return Exp004GuardClassification(
        mean_phase_ratio=mean_status,
        p90_phase_ratio=p90_status,
        pure_CJ3_phase_coverage=coverage_status,
        selected_fallback_rate=fallback_status,
        overall=_overall_guard_status(
            (mean_status, p90_status, coverage_status, fallback_status)
        ),
    )


def classify_max_guard(
    value: float | None,
    *,
    pass_max: float,
    ambiguous_max: float,
) -> str:
    if value is None:
        return GUARD_NOT_APPLICABLE
    value = _finite_float(value, "value")
    if value <= pass_max:
        return GUARD_PASS
    if value <= ambiguous_max:
        return GUARD_AMBIGUOUS
    return GUARD_KILL


def classify_min_guard(
    value: float | None,
    *,
    pass_min: float,
    ambiguous_min: float,
) -> str:
    if value is None:
        return GUARD_NOT_APPLICABLE
    value = _finite_float(value, "value")
    if value >= pass_min:
        return GUARD_PASS
    if value >= ambiguous_min:
        return GUARD_AMBIGUOUS
    return GUARD_KILL


def _active_section_signature_disagreement_sample_count(
    predicted_signature: Sequence[ActiveSectionSignatureSpan],
    reference_signature: Sequence[ActiveSectionSignatureSpan],
    sample_times_ms: Sequence[float],
) -> int:
    return sum(
        _signature_span_at_time(predicted_signature, time_ms)
        != _signature_span_at_time(reference_signature, time_ms)
        for time_ms in sample_times_ms
    )


def _signature_span_at_time(
    signature: Sequence[ActiveSectionSignatureSpan],
    time_ms: float,
) -> ActiveSectionSignatureSpan:
    if not signature:
        raise ValueError("active section signature must be non-empty")
    time_ms = _finite_float(time_ms, "time_ms")
    starts = tuple(span.start_time_ms for span in signature)
    index = bisect.bisect_right(starts, time_ms) - 1
    index = min(max(index, 0), len(signature) - 1)
    span = signature[index]
    if not span.start_time_ms <= time_ms < span.end_time_ms:
        raise ValueError("time_ms is outside the active section signature")
    return span


def _validated_timing_points(
    timing_grid: TimingV3Grid | FittedTimingGrid | Sequence[Any],
) -> tuple[tuple[_GridPoint, ...], str | None]:
    try:
        points = _coerce_timing_points(timing_grid)
    except (TypeError, ValueError) as exc:
        return (), str(exc)
    if not points:
        return (), "no_valid_red_point"
    for point in points:
        if not math.isfinite(point.offset_ms) or not math.isfinite(point.beat_length_ms):
            return (), "nonfinite_fields"
        if point.beat_length_ms <= 0.0:
            return (), "nonpositive_beat_length"
    offsets = tuple(point.offset_ms for point in points)
    if any(right <= left for left, right in zip(offsets, offsets[1:])):
        return (), "nonincreasing_offsets"
    return points, None


def _coerce_timing_points(
    timing_grid: TimingV3Grid | FittedTimingGrid | Sequence[Any],
) -> tuple[_GridPoint, ...]:
    if isinstance(timing_grid, TimingV3Grid):
        return tuple(
            _GridPoint(
                offset_ms=timing_grid.section_start_time_ms(index),
                beat_length_ms=section.beat_length_ms,
                meter=4,
            )
            for index, section in enumerate(timing_grid.sections)
        )
    if isinstance(timing_grid, FittedTimingGrid):
        values: Sequence[Any] = timing_grid.segments
    elif isinstance(timing_grid, SequenceABC) and not isinstance(timing_grid, (str, bytes)):
        values = timing_grid
    else:
        raise TypeError("timing_grid must be TimingV3Grid, FittedTimingGrid, or a sequence")

    points = tuple(
        _GridPoint(
            offset_ms=_attribute_float(value, "offset_ms"),
            beat_length_ms=_attribute_float(value, "beat_length_ms"),
            meter=int(getattr(value, "meter", 4)),
        )
        for value in values
    )
    return points


def _sectioned_grid(
    grid: TimingV3Grid | FittedTimingGrid,
    *,
    canonicalization: str,
) -> _SectionedGrid:
    points, rejection_reason = _validated_timing_points(grid)
    if rejection_reason is not None:
        raise ValueError(rejection_reason)

    offsets = tuple(point.offset_ms for point in points)
    periods = tuple(
        _canonical_period_ms(point.beat_length_ms, canonicalization=canonicalization)
        for point in points
    )
    if isinstance(grid, TimingV3Grid) and canonicalization == TIMING_CANONICALIZATION_NONE:
        cumulative_starts = tuple(float(section.start_beat) for section in grid.sections)
    else:
        cumulative_values = [0.0]
        for index in range(1, len(offsets)):
            cumulative_values.append(
                cumulative_values[-1] + (offsets[index] - offsets[index - 1]) / periods[index - 1]
            )
        cumulative_starts = tuple(cumulative_values)
    return _SectionedGrid(
        offsets_ms=offsets,
        periods_ms=periods,
        cumulative_start_beats=cumulative_starts,
    )


def _validate_grid_coverage(
    grid: TimingV3Grid | FittedTimingGrid,
    coverage_start_ms: float,
    coverage_end_ms: float,
    name: str,
) -> None:
    if coverage_end_ms <= coverage_start_ms:
        raise ValueError("coverage_end_ms must be greater than coverage_start_ms")
    if isinstance(grid, TimingV3Grid):
        if coverage_start_ms < grid.coverage_start_ms or coverage_end_ms > grid.coverage_end_ms:
            raise ValueError(f"{name} coverage is outside the TimingV3Grid coverage")


def _canonical_period_ms(beat_length_ms: float, *, canonicalization: str) -> float:
    if canonicalization == TIMING_CANONICALIZATION_NONE:
        return beat_length_ms
    if canonicalization == TIMING_CANONICALIZATION_BPM_80_160:
        return 60000.0 / canonical_bpm_80_160(60000.0 / beat_length_ms)
    raise ValueError(f"unsupported canonicalization: {canonicalization!r}")


def _weak_consensus_boundaries(
    valid_rows: Sequence[WeakBoundaryDifficultyMetrics],
) -> tuple[WeakConsensusBoundary, ...]:
    if not valid_rows:
        return ()
    valid_count = len(valid_rows)
    required = 2 if valid_count >= 3 else math.ceil(valid_count / 2)
    boundary_by_key: dict[tuple[float, float, float], TimingBoundary] = {}
    matched_counts: dict[tuple[float, float, float], int] = defaultdict(int)
    for row in valid_rows:
        if row.match_summary is None:
            continue
        for boundary in (
            *(match.predicted for match in row.match_summary.matches),
            *row.match_summary.unmatched_predicted_boundaries,
        ):
            boundary_by_key[_boundary_key(boundary)] = boundary
        for match in row.match_summary.matches:
            matched_counts[_boundary_key(match.predicted)] += 1

    return tuple(
        WeakConsensusBoundary(
            predicted_boundary=boundary_by_key[key],
            matched_valid_difficulty_count=matched_counts.get(key, 0),
            required_valid_difficulty_count=required,
            weak_consensus_supported=matched_counts.get(key, 0) >= required,
        )
        for key in sorted(boundary_by_key)
    )


def _boundary_key(boundary: TimingBoundary) -> tuple[float, float, float]:
    return (boundary.time_ms, boundary.left_period_ms, boundary.right_period_ms)


def _mean_count(count: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else float(count / denominator)


def _median_optional(values: Sequence[float | None] | Any) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return float(statistics.median(finite))


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _overall_guard_status(statuses: Sequence[str]) -> str:
    if GUARD_KILL in statuses:
        return GUARD_KILL
    if GUARD_AMBIGUOUS in statuses or GUARD_NOT_APPLICABLE in statuses:
        return GUARD_AMBIGUOUS
    return GUARD_PASS


def _window_p90_abs_relative_drift_ms(
    beat_errors: np.ndarray,
    reference_periods_ms: np.ndarray,
    *,
    window_seconds: float,
) -> float:
    window_samples = max(1, int(round(window_seconds * 1000.0 / PHASE_SAMPLING_HOP_MS)))
    if beat_errors.size <= window_samples:
        return 0.0
    drift_beats = beat_errors[window_samples:] - beat_errors[:-window_samples]
    drift_ms = drift_beats * reference_periods_ms[window_samples:]
    return _np_percentile(np.abs(drift_ms), 90.0)


def _linear_slope(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 2 or float(xs[-1] - xs[0]) <= 0.0:
        return 0.0
    centered_xs = xs - float(np.mean(xs))
    denominator = float(np.dot(centered_xs, centered_xs))
    if denominator <= 0.0:
        return 0.0
    centered_ys = ys - float(np.mean(ys))
    return float(np.dot(centered_xs, centered_ys) / denominator)


def _np_mean(values: np.ndarray) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _np_percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _np_max(values: np.ndarray) -> float:
    return float(np.max(np.asarray(values, dtype=np.float64)))


def _attribute_float(value: Any, name: str) -> float:
    if isinstance(value, MappingABC):
        raw = value[name]
    else:
        raw = getattr(value, name)
    return float(raw)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_finite_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value
