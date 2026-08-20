from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from pulsefield_model.osu_core.hitobjects import (
    ManiaHitObject,
    ManiaHitObjectKind,
    parse_mania_hit_objects,
)
from pulsefield_model.osu_core.timing import (
    RedTimingPoint,
    require_red_timing_points,
    validate_red_timing_point,
)
from pulsefield_model.timing.schema import FittedTimingGrid, TimingSegment


REDLINE_EVIDENCE_MISSING = "missing"
REDLINE_EVIDENCE_STABLE = "stable"
REDLINE_EVIDENCE_JUMP_CANDIDATE = "jump_candidate"
REDLINE_EVIDENCE_DENSE = "dense"
REDLINE_EVIDENCE_RAMP_CANDIDATE = "ramp_candidate"
REDLINE_EVIDENCE_AMBIGUOUS = "ambiguous"

OBJECT_EVENT_START = "start"
OBJECT_EVENT_HOLD_END = "hold_end"

DEFAULT_OBJECT_GRID_SUBDIVISIONS = (1, 2, 3, 4, 6, 8, 12, 16)
DEFAULT_TEMPO_ALIAS_MULTIPLIERS = (0.25, 1.0 / 3.0, 0.5, 1.0, 2.0, 3.0, 4.0)


@dataclass(frozen=True)
class RedlineEvidenceConfig:
    bpm_abs_tolerance: float = 0.05
    bpm_rel_tolerance: float = 0.001
    dense_redline_count: int = 17
    dense_max_change_gap_ms: float = 1000.0
    dense_redlines_per_minute: float = 16.0
    ramp_min_points: int = 3
    ramp_min_total_bpm_delta: float = 3.0
    ramp_min_linear_r2: float = 0.85


@dataclass(frozen=True)
class RedlineSpanEvidence:
    index: int
    start_time_ms: float
    end_time_ms: float | None
    duration_ms: float | None
    beat_length_ms: float
    bpm: float
    meter: int


@dataclass(frozen=True)
class RedlineChangeEvidence:
    previous_index: int
    index: int
    time_ms: float
    gap_ms: float
    previous_bpm: float
    bpm: float
    bpm_delta: float
    bpm_ratio: float
    previous_grid_phase_residual_ms: float


@dataclass(frozen=True)
class RedlineSummary:
    redline_count: int
    bpms: tuple[float, ...]
    beat_lengths_ms: tuple[float, ...]
    unique_bpms: tuple[float, ...]
    spans: tuple[RedlineSpanEvidence, ...]
    changes: tuple[RedlineChangeEvidence, ...]
    change_gaps_ms: tuple[float, ...]
    significant_change_count: int
    map_duration_ms: float | None
    stable_evidence: bool
    jump_evidence: bool
    dense_evidence: bool
    ramp_candidate_evidence: bool
    ramp_direction: str | None
    ramp_linear_r2: float | None
    evidence_class: str
    ambiguous: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ObjectGridEvidenceConfig:
    subdivisions: Sequence[int] = DEFAULT_OBJECT_GRID_SUBDIVISIONS
    alias_multipliers: Sequence[float] = DEFAULT_TEMPO_ALIAS_MULTIPLIERS
    inlier_threshold_ms: float = 8.0
    hold_end_weight: float = 0.5
    min_weighted_events_for_support: float = 1.0
    support_min_weighted_inlier_rate: float = 0.8
    alias_ambiguity_inlier_rate_delta: float = 0.02
    alias_ambiguity_p90_delta_ms: float = 2.0
    min_start_events_for_support: int = 8
    min_start_time_span_ms_for_support: float = 4000.0


@dataclass(frozen=True)
class ObjectResidualStats:
    event_count: int
    total_weight: float
    inlier_count: int
    inlier_weight: float
    inlier_rate: float
    weighted_inlier_rate: float
    mean_abs_residual_beats: float
    p50_abs_residual_beats: float
    p90_abs_residual_beats: float
    max_abs_residual_beats: float
    mean_abs_residual_ms: float
    p50_abs_residual_ms: float
    p90_abs_residual_ms: float
    max_abs_residual_ms: float


@dataclass(frozen=True)
class ObjectSubdivisionEvidence:
    alias_multiplier: float
    subdivision: int
    start_stats: ObjectResidualStats
    hold_end_stats: ObjectResidualStats
    combined_stats: ObjectResidualStats


@dataclass(frozen=True)
class ObjectGridEvidence:
    start_event_count: int
    hold_end_event_count: int
    total_event_count: int
    start_time_span_ms: float
    hold_end_weight: float
    subdivisions: tuple[int, ...]
    alias_multipliers: tuple[float, ...]
    inlier_threshold_ms: float
    best_alias_multiplier: float | None
    best_subdivision: int | None
    best_evidence: ObjectSubdivisionEvidence | None
    subdivision_evidence: tuple[ObjectSubdivisionEvidence, ...]
    grid_supported: bool
    alias_resolved: bool
    supported: bool
    ambiguous: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class BeatmapTimingEvidence:
    beatmap_path: str
    redlines: RedlineSummary
    object_grid: ObjectGridEvidence


@dataclass(frozen=True)
class _GridPoint:
    offset_ms: float
    beat_length_ms: float
    meter: int

    @property
    def bpm(self) -> float:
        return 60000.0 / self.beat_length_ms


@dataclass(frozen=True)
class _ObjectGridEvent:
    kind: str
    time_ms: float
    lane: int
    weight: float


@dataclass(frozen=True)
class _ResidualSample:
    kind: str
    weight: float
    residual_beats: float
    abs_residual_beats: float
    residual_ms: float
    abs_residual_ms: float
    inlier: bool


@dataclass(frozen=True)
class _ObjectGridWorkset:
    event_times_ms: NDArray[np.float64]
    event_weights: NDArray[np.float64]
    start_mask: NDArray[np.bool_]
    hold_end_mask: NDArray[np.bool_]
    active_grid_offsets_ms: NDArray[np.float64]
    active_grid_beat_lengths_ms: NDArray[np.float64]
    start_event_count: int
    hold_end_event_count: int
    start_time_span_ms: float


def summarize_beatmap_timing_evidence(
    beatmap_path: str | Path,
    *,
    candidate_grid: FittedTimingGrid | Sequence[RedTimingPoint | TimingSegment] | None = None,
    expected_key_count: int | None = 4,
    redline_config: RedlineEvidenceConfig | None = None,
    object_grid_config: ObjectGridEvidenceConfig | None = None,
) -> BeatmapTimingEvidence:
    """Parse one `.osu` file and return weak timing evidence, not truth labels."""

    beatmap_path = Path(beatmap_path)
    redlines = require_red_timing_points(beatmap_path)
    hitobjects = parse_mania_hit_objects(beatmap_path, expected_key_count=expected_key_count)
    map_end_time_ms = _map_end_time_ms(hitobjects)
    object_grid_points = redlines if candidate_grid is None else candidate_grid
    return BeatmapTimingEvidence(
        beatmap_path=beatmap_path.as_posix(),
        redlines=summarize_redline_evidence(
            redlines,
            map_end_time_ms=map_end_time_ms,
            config=redline_config,
        ),
        object_grid=summarize_object_grid_evidence(
            hitobjects,
            object_grid_points,
            config=object_grid_config,
        ),
    )


def summarize_redline_evidence(
    timing_points: Sequence[RedTimingPoint],
    *,
    map_end_time_ms: float | None = None,
    config: RedlineEvidenceConfig | None = None,
) -> RedlineSummary:
    config = RedlineEvidenceConfig() if config is None else config
    _validate_redline_config(config)

    sorted_points = tuple(sorted(timing_points, key=lambda point: point.offset_ms))
    for point in sorted_points:
        validate_red_timing_point(point)

    if not sorted_points:
        return RedlineSummary(
            redline_count=0,
            bpms=(),
            beat_lengths_ms=(),
            unique_bpms=(),
            spans=(),
            changes=(),
            change_gaps_ms=(),
            significant_change_count=0,
            map_duration_ms=map_end_time_ms,
            stable_evidence=False,
            jump_evidence=False,
            dense_evidence=False,
            ramp_candidate_evidence=False,
            ramp_direction=None,
            ramp_linear_r2=None,
            evidence_class=REDLINE_EVIDENCE_MISSING,
            ambiguous=True,
            reasons=("no_red_timing_points",),
        )

    spans = _redline_spans(sorted_points, map_end_time_ms=map_end_time_ms)
    changes = _redline_changes(sorted_points)
    significant_changes = tuple(
        change
        for change in changes
        if _bpms_differ(change.previous_bpm, change.bpm, config=config)
    )
    trend_points = _significant_bpm_trend_points(sorted_points, config=config)
    ramp_direction, ramp_linear_r2 = _ramp_candidate_shape(trend_points, config=config)
    ramp_candidate = ramp_direction is not None

    map_duration = _summary_duration_ms(sorted_points, map_end_time_ms)
    redlines_per_minute = (
        len(sorted_points) / (map_duration / 60000.0)
        if map_duration is not None and map_duration > 0.0
        else 0.0
    )
    dense_by_count = len(sorted_points) >= config.dense_redline_count
    dense_by_gap = len(sorted_points) >= 3 and any(
        0.0 <= change.gap_ms <= config.dense_max_change_gap_ms
        for change in changes
    )
    dense_by_rate = len(sorted_points) >= 3 and redlines_per_minute > config.dense_redlines_per_minute
    dense_evidence = dense_by_count or dense_by_gap or dense_by_rate

    stable_evidence = len(significant_changes) == 0
    jump_evidence = bool(significant_changes) and not ramp_candidate and not dense_evidence

    evidence_class = REDLINE_EVIDENCE_AMBIGUOUS
    if stable_evidence and not dense_evidence:
        evidence_class = REDLINE_EVIDENCE_STABLE
    elif ramp_candidate:
        evidence_class = REDLINE_EVIDENCE_RAMP_CANDIDATE
    elif dense_evidence:
        evidence_class = REDLINE_EVIDENCE_DENSE
    elif jump_evidence:
        evidence_class = REDLINE_EVIDENCE_JUMP_CANDIDATE

    reasons = _redline_reasons(
        sorted_points,
        significant_change_count=len(significant_changes),
        stable_evidence=stable_evidence,
        jump_evidence=jump_evidence,
        dense_by_count=dense_by_count,
        dense_by_gap=dense_by_gap,
        dense_by_rate=dense_by_rate,
        ramp_direction=ramp_direction,
        ramp_linear_r2=ramp_linear_r2,
        redlines_per_minute=redlines_per_minute,
        config=config,
    )
    ambiguous = evidence_class == REDLINE_EVIDENCE_AMBIGUOUS or dense_evidence

    return RedlineSummary(
        redline_count=len(sorted_points),
        bpms=tuple(_bpm(point.beat_length_ms) for point in sorted_points),
        beat_lengths_ms=tuple(point.beat_length_ms for point in sorted_points),
        unique_bpms=_unique_bpms(sorted_points, config=config),
        spans=spans,
        changes=changes,
        change_gaps_ms=tuple(change.gap_ms for change in changes),
        significant_change_count=len(significant_changes),
        map_duration_ms=map_duration,
        stable_evidence=stable_evidence,
        jump_evidence=jump_evidence,
        dense_evidence=dense_evidence,
        ramp_candidate_evidence=ramp_candidate,
        ramp_direction=ramp_direction,
        ramp_linear_r2=ramp_linear_r2,
        evidence_class=evidence_class,
        ambiguous=ambiguous,
        reasons=reasons,
    )


def summarize_object_grid_evidence(
    hitobjects: Sequence[ManiaHitObject],
    candidate_grid: FittedTimingGrid | Sequence[RedTimingPoint | TimingSegment],
    *,
    config: ObjectGridEvidenceConfig | None = None,
) -> ObjectGridEvidence:
    config = ObjectGridEvidenceConfig() if config is None else config
    subdivisions = _validate_subdivisions(config.subdivisions)
    alias_multipliers = _validate_alias_multipliers(config.alias_multipliers)
    _validate_object_grid_config(config)

    events = _hitobject_grid_events(hitobjects, hold_end_weight=config.hold_end_weight)
    grid_points = _normalize_candidate_grid(candidate_grid)
    workset = _object_grid_workset(events, grid_points)
    subdivision_evidence: list[ObjectSubdivisionEvidence] = []
    for alias_multiplier in alias_multipliers:
        alias_beat_lengths_ms = workset.active_grid_beat_lengths_ms / alias_multiplier
        beat_positions = (workset.event_times_ms - workset.active_grid_offsets_ms) / alias_beat_lengths_ms
        for subdivision in subdivisions:
            scaled = beat_positions * subdivision
            residual_beats = (scaled - np.floor(scaled + 0.5)) / subdivision
            abs_residual_beats = np.abs(residual_beats)
            abs_residual_ms = np.abs(residual_beats * alias_beat_lengths_ms)
            inliers = abs_residual_ms <= config.inlier_threshold_ms
            subdivision_evidence.append(
                _object_subdivision_evidence_from_arrays(
                    alias_multiplier=alias_multiplier,
                    subdivision=subdivision,
                    workset=workset,
                    abs_residual_beats=abs_residual_beats,
                    abs_residual_ms=abs_residual_ms,
                    inliers=inliers,
                ),
            )

    return _object_grid_evidence_from_subdivision_evidence(
        tuple(subdivision_evidence),
        workset=workset,
        config=config,
        subdivisions=subdivisions,
        alias_multipliers=alias_multipliers,
    )


def _summarize_object_grid_evidence_reference(
    hitobjects: Sequence[ManiaHitObject],
    candidate_grid: FittedTimingGrid | Sequence[RedTimingPoint | TimingSegment],
    *,
    config: ObjectGridEvidenceConfig | None = None,
) -> ObjectGridEvidence:
    config = ObjectGridEvidenceConfig() if config is None else config
    subdivisions = _validate_subdivisions(config.subdivisions)
    alias_multipliers = _validate_alias_multipliers(config.alias_multipliers)
    _validate_object_grid_config(config)

    events = _hitobject_grid_events(hitobjects, hold_end_weight=config.hold_end_weight)
    grid_points = _normalize_candidate_grid(candidate_grid)
    workset = _object_grid_workset(events, grid_points)
    subdivision_evidence: list[ObjectSubdivisionEvidence] = []
    for alias_multiplier in alias_multipliers:
        for subdivision in subdivisions:
            samples = tuple(
                _residual_sample(
                    event,
                    grid_points,
                    alias_multiplier=alias_multiplier,
                    subdivision=subdivision,
                    inlier_threshold_ms=config.inlier_threshold_ms,
                )
                for event in events
            )
            subdivision_evidence.append(
                ObjectSubdivisionEvidence(
                    alias_multiplier=alias_multiplier,
                    subdivision=subdivision,
                    start_stats=_residual_stats(
                        tuple(sample for sample in samples if sample.kind == OBJECT_EVENT_START),
                    ),
                    hold_end_stats=_residual_stats(
                        tuple(sample for sample in samples if sample.kind == OBJECT_EVENT_HOLD_END),
                    ),
                    combined_stats=_residual_stats(samples),
                ),
            )

    return _object_grid_evidence_from_subdivision_evidence(
        tuple(subdivision_evidence),
        workset=workset,
        config=config,
        subdivisions=subdivisions,
        alias_multipliers=alias_multipliers,
    )


def _object_grid_evidence_from_subdivision_evidence(
    evidence_tuple: tuple[ObjectSubdivisionEvidence, ...],
    *,
    workset: _ObjectGridWorkset,
    config: ObjectGridEvidenceConfig,
    subdivisions: tuple[int, ...],
    alias_multipliers: tuple[float, ...],
) -> ObjectGridEvidence:
    best_evidence = max(evidence_tuple, key=_object_evidence_score_key) if evidence_tuple else None
    grid_supported = _grid_supported_by_starts(
        best_evidence,
        start_event_count=workset.start_event_count,
        start_time_span_ms=workset.start_time_span_ms,
        config=config,
    )
    alias_ambiguous = _has_alias_ambiguity(
        evidence_tuple,
        best_evidence,
        inlier_rate_delta=config.alias_ambiguity_inlier_rate_delta,
        p90_delta_ms=config.alias_ambiguity_p90_delta_ms,
    )
    alias_resolved = not alias_ambiguous
    supported = grid_supported and alias_resolved

    reasons = _object_grid_reasons(
        start_event_count=workset.start_event_count,
        hold_end_event_count=workset.hold_end_event_count,
        start_time_span_ms=workset.start_time_span_ms,
        grid_supported=grid_supported,
        alias_resolved=alias_resolved,
        supported=supported,
        best_evidence=best_evidence,
        config=config,
    )

    return ObjectGridEvidence(
        start_event_count=workset.start_event_count,
        hold_end_event_count=workset.hold_end_event_count,
        total_event_count=int(workset.event_times_ms.shape[0]),
        start_time_span_ms=workset.start_time_span_ms,
        hold_end_weight=config.hold_end_weight,
        subdivisions=subdivisions,
        alias_multipliers=alias_multipliers,
        inlier_threshold_ms=config.inlier_threshold_ms,
        best_alias_multiplier=None if best_evidence is None else best_evidence.alias_multiplier,
        best_subdivision=None if best_evidence is None else best_evidence.subdivision,
        best_evidence=best_evidence,
        subdivision_evidence=evidence_tuple,
        grid_supported=grid_supported,
        alias_resolved=alias_resolved,
        supported=supported,
        ambiguous=not supported,
        reasons=reasons,
    )


def _validate_redline_config(config: RedlineEvidenceConfig) -> None:
    if config.bpm_abs_tolerance < 0.0:
        raise ValueError("bpm_abs_tolerance must be non-negative")
    if config.bpm_rel_tolerance < 0.0:
        raise ValueError("bpm_rel_tolerance must be non-negative")
    if config.dense_redline_count <= 0:
        raise ValueError("dense_redline_count must be positive")
    if config.dense_max_change_gap_ms < 0.0:
        raise ValueError("dense_max_change_gap_ms must be non-negative")
    if config.dense_redlines_per_minute < 0.0:
        raise ValueError("dense_redlines_per_minute must be non-negative")
    if config.ramp_min_points <= 1:
        raise ValueError("ramp_min_points must be greater than 1")
    if config.ramp_min_total_bpm_delta < 0.0:
        raise ValueError("ramp_min_total_bpm_delta must be non-negative")
    if not 0.0 <= config.ramp_min_linear_r2 <= 1.0:
        raise ValueError("ramp_min_linear_r2 must be in [0, 1]")


def _validate_object_grid_config(config: ObjectGridEvidenceConfig) -> None:
    if config.inlier_threshold_ms < 0.0:
        raise ValueError("inlier_threshold_ms must be non-negative")
    if config.hold_end_weight < 0.0:
        raise ValueError("hold_end_weight must be non-negative")
    if config.min_weighted_events_for_support < 0.0:
        raise ValueError("min_weighted_events_for_support must be non-negative")
    if not 0.0 <= config.support_min_weighted_inlier_rate <= 1.0:
        raise ValueError("support_min_weighted_inlier_rate must be in [0, 1]")
    if config.alias_ambiguity_inlier_rate_delta < 0.0:
        raise ValueError("alias_ambiguity_inlier_rate_delta must be non-negative")
    if config.alias_ambiguity_p90_delta_ms < 0.0:
        raise ValueError("alias_ambiguity_p90_delta_ms must be non-negative")
    if config.min_start_events_for_support < 0:
        raise ValueError("min_start_events_for_support must be non-negative")
    if config.min_start_time_span_ms_for_support < 0.0:
        raise ValueError("min_start_time_span_ms_for_support must be non-negative")


def _redline_spans(
    sorted_points: Sequence[RedTimingPoint],
    *,
    map_end_time_ms: float | None,
) -> tuple[RedlineSpanEvidence, ...]:
    spans: list[RedlineSpanEvidence] = []
    for index, point in enumerate(sorted_points):
        next_offset = sorted_points[index + 1].offset_ms if index + 1 < len(sorted_points) else None
        end_time_ms = next_offset
        if end_time_ms is None and map_end_time_ms is not None and map_end_time_ms > point.offset_ms:
            end_time_ms = float(map_end_time_ms)
        duration_ms = None if end_time_ms is None else max(0.0, float(end_time_ms - point.offset_ms))
        spans.append(
            RedlineSpanEvidence(
                index=index,
                start_time_ms=point.offset_ms,
                end_time_ms=end_time_ms,
                duration_ms=duration_ms,
                beat_length_ms=point.beat_length_ms,
                bpm=_bpm(point.beat_length_ms),
                meter=point.meter,
            ),
        )
    return tuple(spans)


def _redline_changes(sorted_points: Sequence[RedTimingPoint]) -> tuple[RedlineChangeEvidence, ...]:
    changes: list[RedlineChangeEvidence] = []
    for index in range(1, len(sorted_points)):
        previous = sorted_points[index - 1]
        current = sorted_points[index]
        previous_bpm = _bpm(previous.beat_length_ms)
        current_bpm = _bpm(current.beat_length_ms)
        residual_beats = _signed_residual_to_subdivision(
            (current.offset_ms - previous.offset_ms) / previous.beat_length_ms,
            subdivision=1,
        )
        changes.append(
            RedlineChangeEvidence(
                previous_index=index - 1,
                index=index,
                time_ms=current.offset_ms,
                gap_ms=current.offset_ms - previous.offset_ms,
                previous_bpm=previous_bpm,
                bpm=current_bpm,
                bpm_delta=current_bpm - previous_bpm,
                bpm_ratio=current_bpm / previous_bpm,
                previous_grid_phase_residual_ms=abs(residual_beats) * previous.beat_length_ms,
            ),
        )
    return tuple(changes)


def _redline_reasons(
    sorted_points: Sequence[RedTimingPoint],
    *,
    significant_change_count: int,
    stable_evidence: bool,
    jump_evidence: bool,
    dense_by_count: bool,
    dense_by_gap: bool,
    dense_by_rate: bool,
    ramp_direction: str | None,
    ramp_linear_r2: float | None,
    redlines_per_minute: float,
    config: RedlineEvidenceConfig,
) -> tuple[str, ...]:
    reasons: list[str] = [f"redline_count={len(sorted_points)}"]
    if stable_evidence:
        reasons.append("no_significant_bpm_change")
    else:
        reasons.append(f"significant_bpm_changes={significant_change_count}")
    if jump_evidence:
        reasons.append("sparse_significant_bpm_change")
    if dense_by_count:
        reasons.append(f"dense_redline_count>={config.dense_redline_count}")
    if dense_by_gap:
        reasons.append(f"change_gap<={config.dense_max_change_gap_ms:g}ms")
    if dense_by_rate:
        reasons.append(f"redline_rate={redlines_per_minute:.3f}/min>{config.dense_redlines_per_minute:g}/min")
    if ramp_direction is not None:
        reasons.append(f"monotonic_{ramp_direction}_bpm")
        if ramp_linear_r2 is not None:
            reasons.append(f"ramp_linear_r2={ramp_linear_r2:.6f}")
    if not stable_evidence and not jump_evidence and ramp_direction is None and not (
        dense_by_count or dense_by_gap or dense_by_rate
    ):
        reasons.append("conflicting_or_weak_redline_shape")
    return tuple(reasons)


def _summary_duration_ms(sorted_points: Sequence[RedTimingPoint], map_end_time_ms: float | None) -> float | None:
    if map_end_time_ms is not None:
        return max(0.0, float(map_end_time_ms - sorted_points[0].offset_ms))
    if len(sorted_points) <= 1:
        return None
    return max(0.0, float(sorted_points[-1].offset_ms - sorted_points[0].offset_ms))


def _significant_bpm_trend_points(
    sorted_points: Sequence[RedTimingPoint],
    *,
    config: RedlineEvidenceConfig,
) -> tuple[RedTimingPoint, ...]:
    if not sorted_points:
        return ()
    trend_points = [sorted_points[0]]
    for point in sorted_points[1:]:
        if _bpms_differ(_bpm(trend_points[-1].beat_length_ms), _bpm(point.beat_length_ms), config=config):
            trend_points.append(point)
    return tuple(trend_points)


def _ramp_candidate_shape(
    trend_points: Sequence[RedTimingPoint],
    *,
    config: RedlineEvidenceConfig,
) -> tuple[str | None, float | None]:
    if len(trend_points) < config.ramp_min_points:
        return None, None
    bpms = tuple(_bpm(point.beat_length_ms) for point in trend_points)
    offsets = tuple(point.offset_ms for point in trend_points)
    deltas = tuple(bpms[index] - bpms[index - 1] for index in range(1, len(bpms)))
    tolerance = max(config.bpm_abs_tolerance, config.bpm_rel_tolerance * max(abs(bpm) for bpm in bpms))
    increasing = all(delta > tolerance for delta in deltas)
    decreasing = all(delta < -tolerance for delta in deltas)
    total_delta = abs(bpms[-1] - bpms[0])
    if not (increasing or decreasing) or total_delta < config.ramp_min_total_bpm_delta:
        return None, None
    r2 = _linear_r2(offsets, bpms)
    if r2 < config.ramp_min_linear_r2:
        return None, r2
    return ("increasing" if increasing else "decreasing"), r2


def _linear_r2(xs: Sequence[float], ys: Sequence[float]) -> float:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    centered_x = tuple(x - mean_x for x in xs)
    centered_y = tuple(y - mean_y for y in ys)
    ss_x = sum(value * value for value in centered_x)
    ss_total = sum(value * value for value in centered_y)
    if ss_x == 0.0:
        return 0.0
    if ss_total == 0.0:
        return 1.0
    slope = sum(x * y for x, y in zip(centered_x, centered_y)) / ss_x
    residual = sum((y - slope * x) ** 2 for x, y in zip(centered_x, centered_y))
    return max(0.0, min(1.0, 1.0 - residual / ss_total))


def _unique_bpms(
    sorted_points: Sequence[RedTimingPoint],
    *,
    config: RedlineEvidenceConfig,
) -> tuple[float, ...]:
    unique: list[float] = []
    for point in sorted_points:
        bpm = _bpm(point.beat_length_ms)
        if not any(not _bpms_differ(existing, bpm, config=config) for existing in unique):
            unique.append(bpm)
    return tuple(unique)


def _bpms_differ(first_bpm: float, second_bpm: float, *, config: RedlineEvidenceConfig) -> bool:
    tolerance = max(config.bpm_abs_tolerance, config.bpm_rel_tolerance * max(abs(first_bpm), abs(second_bpm)))
    return abs(first_bpm - second_bpm) > tolerance


def _bpm(beat_length_ms: float) -> float:
    return 60000.0 / float(beat_length_ms)


def _map_end_time_ms(hitobjects: Sequence[ManiaHitObject]) -> float | None:
    if not hitobjects:
        return None
    return max(hitobject.end_time_ms for hitobject in hitobjects)


def _hitobject_grid_events(
    hitobjects: Sequence[ManiaHitObject],
    *,
    hold_end_weight: float,
) -> tuple[_ObjectGridEvent, ...]:
    events: list[_ObjectGridEvent] = []
    for hitobject in hitobjects:
        events.append(
            _ObjectGridEvent(
                kind=OBJECT_EVENT_START,
                time_ms=hitobject.start_time_ms,
                lane=hitobject.lane,
                weight=1.0,
            ),
        )
        if hitobject.kind == ManiaHitObjectKind.HOLD:
            events.append(
                _ObjectGridEvent(
                    kind=OBJECT_EVENT_HOLD_END,
                    time_ms=hitobject.end_time_ms,
                    lane=hitobject.lane,
                    weight=hold_end_weight,
                ),
            )
    return tuple(events)


def _event_time_span_ms(events: Sequence[_ObjectGridEvent]) -> float:
    if len(events) <= 1:
        return 0.0
    times = tuple(event.time_ms for event in events)
    return max(times) - min(times)


def _object_grid_workset(
    events: Sequence[_ObjectGridEvent],
    grid_points: Sequence[_GridPoint],
) -> _ObjectGridWorkset:
    event_times_ms = np.asarray([event.time_ms for event in events], dtype=np.float64)
    event_weights = np.asarray([event.weight for event in events], dtype=np.float64)
    start_mask = np.asarray([event.kind == OBJECT_EVENT_START for event in events], dtype=np.bool_)
    hold_end_mask = np.asarray([event.kind == OBJECT_EVENT_HOLD_END for event in events], dtype=np.bool_)

    grid_offsets_ms = np.asarray([point.offset_ms for point in grid_points], dtype=np.float64)
    grid_beat_lengths_ms = np.asarray([point.beat_length_ms for point in grid_points], dtype=np.float64)
    if event_times_ms.size:
        active_grid_indexes = np.searchsorted(grid_offsets_ms, event_times_ms, side="right") - 1
        active_grid_indexes = np.maximum(active_grid_indexes, 0)
        active_grid_offsets_ms = grid_offsets_ms[active_grid_indexes]
        active_grid_beat_lengths_ms = grid_beat_lengths_ms[active_grid_indexes]
    else:
        active_grid_offsets_ms = np.asarray([], dtype=np.float64)
        active_grid_beat_lengths_ms = np.asarray([], dtype=np.float64)

    start_event_count = int(np.count_nonzero(start_mask))
    hold_end_event_count = int(np.count_nonzero(hold_end_mask))
    start_time_span_ms = 0.0
    if start_event_count > 1:
        start_times_ms = event_times_ms[start_mask]
        start_time_span_ms = float(np.max(start_times_ms) - np.min(start_times_ms))

    return _ObjectGridWorkset(
        event_times_ms=event_times_ms,
        event_weights=event_weights,
        start_mask=start_mask,
        hold_end_mask=hold_end_mask,
        active_grid_offsets_ms=active_grid_offsets_ms,
        active_grid_beat_lengths_ms=active_grid_beat_lengths_ms,
        start_event_count=start_event_count,
        hold_end_event_count=hold_end_event_count,
        start_time_span_ms=start_time_span_ms,
    )


def _normalize_candidate_grid(
    candidate_grid: FittedTimingGrid | Sequence[RedTimingPoint | TimingSegment],
) -> tuple[_GridPoint, ...]:
    if isinstance(candidate_grid, FittedTimingGrid):
        raw_points: Sequence[RedTimingPoint | TimingSegment] = candidate_grid.segments
    else:
        raw_points = candidate_grid

    points: list[_GridPoint] = []
    for point in raw_points:
        if isinstance(point, RedTimingPoint):
            validate_red_timing_point(point)
            points.append(
                _GridPoint(
                    offset_ms=point.offset_ms,
                    beat_length_ms=point.beat_length_ms,
                    meter=point.meter,
                ),
            )
            continue
        if isinstance(point, TimingSegment):
            points.append(
                _GridPoint(
                    offset_ms=point.offset_ms,
                    beat_length_ms=point.beat_length_ms,
                    meter=point.meter,
                ),
            )
            continue
        raise TypeError(f"unsupported candidate grid point: {point!r}")

    if not points:
        raise ValueError("candidate_grid must contain at least one timing point")
    return tuple(sorted(points, key=lambda point: point.offset_ms))


def _residual_sample(
    event: _ObjectGridEvent,
    grid_points: Sequence[_GridPoint],
    *,
    alias_multiplier: float,
    subdivision: int,
    inlier_threshold_ms: float,
) -> _ResidualSample:
    grid_point = _grid_point_at(grid_points, event.time_ms)
    beat_length_ms = grid_point.beat_length_ms / alias_multiplier
    beat_position = (event.time_ms - grid_point.offset_ms) / beat_length_ms
    residual_beats = _signed_residual_to_subdivision(beat_position, subdivision=subdivision)
    residual_ms = residual_beats * beat_length_ms
    abs_residual_ms = abs(residual_ms)
    return _ResidualSample(
        kind=event.kind,
        weight=event.weight,
        residual_beats=residual_beats,
        abs_residual_beats=abs(residual_beats),
        residual_ms=residual_ms,
        abs_residual_ms=abs_residual_ms,
        inlier=abs_residual_ms <= inlier_threshold_ms,
    )


def _grid_point_at(grid_points: Sequence[_GridPoint], time_ms: float) -> _GridPoint:
    offsets = [point.offset_ms for point in grid_points]
    index = bisect_right(offsets, time_ms) - 1
    if index < 0:
        return grid_points[0]
    return grid_points[index]


def _signed_residual_to_subdivision(beat_position: float, *, subdivision: int) -> float:
    scaled = beat_position * subdivision
    nearest = math.floor(scaled + 0.5)
    return (scaled - nearest) / subdivision


def _object_subdivision_evidence_from_arrays(
    *,
    alias_multiplier: float,
    subdivision: int,
    workset: _ObjectGridWorkset,
    abs_residual_beats: NDArray[np.float64],
    abs_residual_ms: NDArray[np.float64],
    inliers: NDArray[np.bool_],
) -> ObjectSubdivisionEvidence:
    return ObjectSubdivisionEvidence(
        alias_multiplier=alias_multiplier,
        subdivision=subdivision,
        start_stats=_residual_stats_from_arrays(
            abs_residual_beats,
            abs_residual_ms,
            workset.event_weights,
            inliers,
            mask=workset.start_mask,
        ),
        hold_end_stats=_residual_stats_from_arrays(
            abs_residual_beats,
            abs_residual_ms,
            workset.event_weights,
            inliers,
            mask=workset.hold_end_mask,
        ),
        combined_stats=_residual_stats_from_arrays(
            abs_residual_beats,
            abs_residual_ms,
            workset.event_weights,
            inliers,
        ),
    )


def _residual_stats_from_arrays(
    abs_residual_beats: NDArray[np.float64],
    abs_residual_ms: NDArray[np.float64],
    weights: NDArray[np.float64],
    inliers: NDArray[np.bool_],
    *,
    mask: NDArray[np.bool_] | None = None,
) -> ObjectResidualStats:
    if mask is not None:
        abs_residual_beats = abs_residual_beats[mask]
        abs_residual_ms = abs_residual_ms[mask]
        weights = weights[mask]
        inliers = inliers[mask]

    event_count = int(abs_residual_ms.shape[0])
    if event_count == 0:
        return _empty_residual_stats()

    total_weight = float(np.sum(weights, dtype=np.float64))
    inlier_count = int(np.count_nonzero(inliers))
    inlier_weight = float(np.sum(weights[inliers], dtype=np.float64))
    return ObjectResidualStats(
        event_count=event_count,
        total_weight=total_weight,
        inlier_count=inlier_count,
        inlier_weight=inlier_weight,
        inlier_rate=inlier_count / event_count,
        weighted_inlier_rate=0.0 if total_weight == 0.0 else inlier_weight / total_weight,
        mean_abs_residual_beats=_weighted_mean_array(abs_residual_beats, weights),
        p50_abs_residual_beats=_percentile_array(abs_residual_beats, 50.0),
        p90_abs_residual_beats=_percentile_array(abs_residual_beats, 90.0),
        max_abs_residual_beats=float(np.max(abs_residual_beats)),
        mean_abs_residual_ms=_weighted_mean_array(abs_residual_ms, weights),
        p50_abs_residual_ms=_percentile_array(abs_residual_ms, 50.0),
        p90_abs_residual_ms=_percentile_array(abs_residual_ms, 90.0),
        max_abs_residual_ms=float(np.max(abs_residual_ms)),
    )


def _residual_stats(samples: Sequence[_ResidualSample]) -> ObjectResidualStats:
    if not samples:
        return _empty_residual_stats()
    total_weight = sum(sample.weight for sample in samples)
    inlier_count = sum(1 for sample in samples if sample.inlier)
    inlier_weight = sum(sample.weight for sample in samples if sample.inlier)
    abs_residual_beats = tuple(sample.abs_residual_beats for sample in samples)
    abs_residual_ms = tuple(sample.abs_residual_ms for sample in samples)
    return ObjectResidualStats(
        event_count=len(samples),
        total_weight=total_weight,
        inlier_count=inlier_count,
        inlier_weight=inlier_weight,
        inlier_rate=inlier_count / len(samples),
        weighted_inlier_rate=0.0 if total_weight == 0.0 else inlier_weight / total_weight,
        mean_abs_residual_beats=_weighted_mean(
            tuple(sample.abs_residual_beats for sample in samples),
            tuple(sample.weight for sample in samples),
        ),
        p50_abs_residual_beats=_percentile(abs_residual_beats, 50.0),
        p90_abs_residual_beats=_percentile(abs_residual_beats, 90.0),
        max_abs_residual_beats=max(abs_residual_beats),
        mean_abs_residual_ms=_weighted_mean(
            tuple(sample.abs_residual_ms for sample in samples),
            tuple(sample.weight for sample in samples),
        ),
        p50_abs_residual_ms=_percentile(abs_residual_ms, 50.0),
        p90_abs_residual_ms=_percentile(abs_residual_ms, 90.0),
        max_abs_residual_ms=max(abs_residual_ms),
    )


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    total_weight = sum(weights)
    if total_weight == 0.0:
        return 0.0
    return sum(value * weight for value, weight in zip(values, weights)) / total_weight


def _weighted_mean_array(values: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    total_weight = float(np.sum(weights, dtype=np.float64))
    if total_weight == 0.0:
        return 0.0
    return float(np.sum(values * weights, dtype=np.float64) / total_weight)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * percentile / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[int(index)]
    fraction = index - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _percentile_array(values: NDArray[np.float64], percentile: float) -> float:
    if values.size == 0:
        return 0.0
    sorted_values = np.sort(values)
    if sorted_values.size == 1:
        return float(sorted_values[0])
    index = (sorted_values.size - 1) * percentile / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(sorted_values[int(index)])
    fraction = index - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def _empty_residual_stats() -> ObjectResidualStats:
    return ObjectResidualStats(
        event_count=0,
        total_weight=0.0,
        inlier_count=0,
        inlier_weight=0.0,
        inlier_rate=0.0,
        weighted_inlier_rate=0.0,
        mean_abs_residual_beats=0.0,
        p50_abs_residual_beats=0.0,
        p90_abs_residual_beats=0.0,
        max_abs_residual_beats=0.0,
        mean_abs_residual_ms=0.0,
        p50_abs_residual_ms=0.0,
        p90_abs_residual_ms=0.0,
        max_abs_residual_ms=0.0,
    )


def _object_evidence_score_key(evidence: ObjectSubdivisionEvidence) -> tuple[float, float, float, float, int]:
    stats = evidence.combined_stats
    return (
        stats.weighted_inlier_rate,
        -stats.mean_abs_residual_ms,
        -stats.p90_abs_residual_ms,
        -abs(math.log2(evidence.alias_multiplier)),
        -evidence.subdivision,
    )


def _has_alias_ambiguity(
    subdivision_evidence: Sequence[ObjectSubdivisionEvidence],
    best_evidence: ObjectSubdivisionEvidence | None,
    *,
    inlier_rate_delta: float,
    p90_delta_ms: float,
) -> bool:
    if best_evidence is None:
        return False
    best_stats = best_evidence.combined_stats
    aliases: set[float] = {best_evidence.alias_multiplier}
    for evidence in subdivision_evidence:
        if math.isclose(evidence.alias_multiplier, best_evidence.alias_multiplier, rel_tol=0.0, abs_tol=1e-12):
            continue
        stats = evidence.combined_stats
        if (
            stats.weighted_inlier_rate >= best_stats.weighted_inlier_rate - inlier_rate_delta
            and stats.p90_abs_residual_ms <= best_stats.p90_abs_residual_ms + p90_delta_ms
        ):
            aliases.add(evidence.alias_multiplier)
    return len(aliases) > 1


def _grid_supported_by_starts(
    best_evidence: ObjectSubdivisionEvidence | None,
    *,
    start_event_count: int,
    start_time_span_ms: float,
    config: ObjectGridEvidenceConfig,
) -> bool:
    if best_evidence is None:
        return False
    start_stats = best_evidence.start_stats
    return (
        start_event_count >= config.min_start_events_for_support
        and start_time_span_ms >= config.min_start_time_span_ms_for_support
        and start_stats.total_weight >= config.min_weighted_events_for_support
        and start_stats.weighted_inlier_rate >= config.support_min_weighted_inlier_rate
    )


def _object_grid_reasons(
    *,
    start_event_count: int,
    hold_end_event_count: int,
    start_time_span_ms: float,
    grid_supported: bool,
    alias_resolved: bool,
    supported: bool,
    best_evidence: ObjectSubdivisionEvidence | None,
    config: ObjectGridEvidenceConfig,
) -> tuple[str, ...]:
    reasons: list[str] = [
        f"start_events={start_event_count}",
        f"hold_end_events={hold_end_event_count}",
        f"start_time_span_ms={start_time_span_ms:.6f}",
    ]
    if start_event_count == 0:
        reasons.append("no_start_events")
    if start_event_count < config.min_start_events_for_support:
        reasons.append(f"start_events<{config.min_start_events_for_support}")
    if start_time_span_ms < config.min_start_time_span_ms_for_support:
        reasons.append(f"start_time_span_ms<{config.min_start_time_span_ms_for_support:g}")
    if hold_end_event_count == 0:
        reasons.append("no_hold_end_events")
    if best_evidence is None:
        reasons.append("no_subdivision_evidence")
    else:
        stats = best_evidence.combined_stats
        start_stats = best_evidence.start_stats
        reasons.append(f"best_alias_multiplier={best_evidence.alias_multiplier:g}")
        reasons.append(f"best_subdivision={best_evidence.subdivision}")
        reasons.append(f"start_weighted_inlier_rate={start_stats.weighted_inlier_rate:.6f}")
        reasons.append(f"weighted_inlier_rate={stats.weighted_inlier_rate:.6f}")
        reasons.append(f"p90_abs_residual_ms={stats.p90_abs_residual_ms:.6f}")
    if grid_supported:
        reasons.append("grid_supported_by_starts")
    else:
        reasons.append("weak_or_missing_start_grid_support")
        reasons.append("weak_or_missing_object_grid_support")
    if alias_resolved:
        reasons.append("alias_resolved")
    else:
        reasons.append("alias_ambiguous")
    if supported:
        reasons.append("supported=grid_supported_and_alias_resolved")
    return tuple(reasons)


def _validate_subdivisions(subdivisions: Sequence[int]) -> tuple[int, ...]:
    unique: set[int] = set()
    for subdivision in subdivisions:
        subdivision = int(subdivision)
        if subdivision <= 0:
            raise ValueError(f"subdivisions must be positive: {subdivision}")
        unique.add(subdivision)
    if not unique:
        raise ValueError("subdivisions must contain at least one value")
    return tuple(sorted(unique))


def _validate_alias_multipliers(alias_multipliers: Sequence[float]) -> tuple[float, ...]:
    unique: set[float] = set()
    for alias_multiplier in alias_multipliers:
        alias_multiplier = float(alias_multiplier)
        if not math.isfinite(alias_multiplier) or alias_multiplier <= 0.0:
            raise ValueError(f"alias multipliers must be positive and finite: {alias_multiplier!r}")
        unique.add(alias_multiplier)
    if not unique:
        raise ValueError("alias_multipliers must contain at least one value")
    return tuple(sorted(unique))
