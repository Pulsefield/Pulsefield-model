from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from pulsefield_model.features.audio import load_audio_file
from pulsefield_model.features.mel import stage2_log_mel_cache_path
from pulsefield_model.features.mel_base import DEFAULT_MEL_CACHE_CONFIG, compute_log_mel_10ms
from pulsefield_model.timing.evaluation.curve_metrics import (
    WeakOracleClass,
    evaluate_curve_against_weak_oracle,
)
from pulsefield_model.timing.providers.beatthis_cache import (
    beatthis_frame_prediction_cache_path,
    load_beatthis_frame_prediction_cache,
)
from pulsefield_model.timing.providers.oracle import oracle_timing_grid_from_beatmap
from pulsefield_model.timing.schema import FittedTimingGrid, TimingSegment
from pulsefield_model.timing.v3.analytic_curve import PhaseContinuousTimingCurve
from pulsefield_model.timing.v3.audio_evidence import extract_raw_audio_evidence
from pulsefield_model.timing.v3.tempo_track import (
    TempoTrackResult,
    generate_timing_candidates,
    tempo_track_result_to_dict,
)


EXP013_PILOT_RESULT_SCHEMA = "pulsefield_model.timing_v3_exp013_pilot_result_v1"
EXP013_PILOT_SUMMARY_SCHEMA = "pulsefield_model.timing_v3_exp013_pilot_summary_v1"
EXP013_FROZEN_INFERENCE_SCHEMA = (
    "pulsefield_model.timing_v3_exp013_frozen_inference_v1"
)
EXP013_INFERENCE_FAMILY = "timing_v3_exp013_raw_run_ordinal_selector_v1"
EXP013_WEAK_ORACLE_POLICY = "post_frozen_inference_representative_redline_v1"

EXP013_ALLOWED_CONFIDENCES = frozenset(("high", "medium"))
EXP013_ALLOWED_STRATA = frozenset(("stable", "jump_candidate"))


@dataclass(frozen=True)
class _PilotRunMetadata:
    result_schema: str
    summary_schema: str
    frozen_inference_schema: str
    inference_family: str
    weak_oracle_policy: str


EXP013_RUN_METADATA = _PilotRunMetadata(
    result_schema=EXP013_PILOT_RESULT_SCHEMA,
    summary_schema=EXP013_PILOT_SUMMARY_SCHEMA,
    frozen_inference_schema=EXP013_FROZEN_INFERENCE_SCHEMA,
    inference_family=EXP013_INFERENCE_FAMILY,
    weak_oracle_policy=EXP013_WEAK_ORACLE_POLICY,
)


class BeatThisCacheLoader(Protocol):
    def __call__(self, cache_audio_key: str) -> object | None: ...


class BeatThisCachePathResolver(Protocol):
    def __call__(self, cache_audio_key: str) -> Path: ...


class MelLoader(Protocol):
    def __call__(
        self,
        audio_path: Path,
        *,
        repo_root: Path,
    ) -> tuple[np.ndarray[Any, np.dtype[np.float32]], str, Path | None]: ...


class RawEvidenceExtractor(Protocol):
    def __call__(self, log_mel_10ms: object, *, audio_duration_seconds: float) -> object: ...


class Exp013CandidateGenerator(Protocol):
    def __call__(self, prediction: object, *, audio_evidence: object) -> TempoTrackResult: ...


class WeakOracleLoader(Protocol):
    def __call__(self, beatmap_path: Path) -> FittedTimingGrid: ...


class CurveMetricEvaluator(Protocol):
    def __call__(
        self,
        predicted: PhaseContinuousTimingCurve,
        weak_oracle: FittedTimingGrid,
        *,
        weak_oracle_class: WeakOracleClass,
        coverage_start_ms: float,
        coverage_end_ms: float,
    ) -> object: ...


FrozenPayloadObserver = Callable[[Mapping[str, Any], str], None]
Clock = Callable[[], float]


@dataclass(frozen=True)
class _PilotEntry:
    output_row_index: int
    source_row_index: int
    cache_audio_key: str
    audio_path: Path
    duration_seconds: float
    stratum: str
    confidence: str
    row: Mapping[str, Any]


@dataclass(frozen=True)
class _JsonlSnapshot:
    path: Path
    sha256: str
    rows: tuple[Mapping[str, Any], ...]


class _LazyJsonlSnapshot:
    def __init__(self, path: Path | None) -> None:
        self._path = path
        self._snapshot: _JsonlSnapshot | None = None

    @property
    def path(self) -> Path | None:
        return self._path

    def snapshot(self) -> _JsonlSnapshot | None:
        if self._path is None:
            return None
        if self._snapshot is None:
            self._snapshot = _read_jsonl_snapshot(self._path)
        return self._snapshot

    def sha256_if_available(self) -> str | None:
        snapshot = self.snapshot()
        return None if snapshot is None else snapshot.sha256

    def row_by_audio_key(self, cache_audio_key: str) -> Mapping[str, Any] | None:
        snapshot = self.snapshot()
        if snapshot is None:
            return None
        for row in snapshot.rows:
            audio_key = row.get("audio_key") or row.get("cache_audio_key")
            if audio_key == cache_audio_key:
                return row
        return None


def run_exp013_pilot(
    *,
    pilot_jsonl_path: str | Path,
    baseline_v2_jsonl_path: str | Path | None = None,
    output_jsonl_path: str | Path,
    summary_json_path: str | Path,
    repo_root: str | Path | None = None,
    limit: int | None = None,
    explicit_cache_audio_keys: Sequence[str] | None = None,
    beatthis_cache_loader: BeatThisCacheLoader = load_beatthis_frame_prediction_cache,
    beatthis_cache_path_resolver: BeatThisCachePathResolver = (
        beatthis_frame_prediction_cache_path
    ),
    mel_loader: MelLoader | None = None,
    raw_evidence_extractor: RawEvidenceExtractor = extract_raw_audio_evidence,
    candidate_generator: Exp013CandidateGenerator = generate_timing_candidates,
    weak_oracle_loader: WeakOracleLoader = oracle_timing_grid_from_beatmap,
    metric_evaluator: CurveMetricEvaluator = evaluate_curve_against_weak_oracle,
    frozen_payload_observer: FrozenPayloadObserver | None = None,
    clock: Clock = time.perf_counter,
    run_metadata: _PilotRunMetadata = EXP013_RUN_METADATA,
) -> dict[str, Any]:
    """Run the frozen Exp013 exposed pilot adapter.

    The inference phase consumes only identity, duration, audio path, BeatThis
    cache, and deterministic raw-audio evidence.  The representative weak
    oracle is not inspected until after the selected product status and stable
    frozen-inference SHA-256 have been computed.
    """

    pilot_path = Path(pilot_jsonl_path)
    baseline_path = None if baseline_v2_jsonl_path is None else Path(baseline_v2_jsonl_path)
    output_path = Path(output_jsonl_path)
    summary_path = Path(summary_json_path)
    root = Path(repo_root or Path.cwd()).expanduser().resolve(strict=False)
    _require_distinct_input_output_paths(
        pilot_path=pilot_path,
        baseline_path=baseline_path,
        output_path=output_path,
        summary_path=summary_path,
    )
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if mel_loader is None:
        mel_loader = _load_log_mel_10ms

    explicit_keys = None
    if explicit_cache_audio_keys is not None:
        explicit_keys = frozenset(explicit_cache_audio_keys)

    pilot_snapshot = _read_jsonl_snapshot(pilot_path)
    rows = pilot_snapshot.rows
    baseline_snapshot = _LazyJsonlSnapshot(baseline_path)
    selected_rows = [
        (source_row_index, row)
        for source_row_index, row in enumerate(rows)
        if _row_matches_exp013_filter(
            row,
            explicit_cache_audio_keys=explicit_keys,
        )
    ]
    if limit is not None:
        selected_rows = selected_rows[:limit]

    results: list[dict[str, Any]] = []
    started = clock()
    for output_index, (source_row_index, row) in enumerate(selected_rows):
        row_started = clock()
        try:
            effective_entry = _entry_from_row(
                row,
                source_row_index=source_row_index,
                output_row_index=output_index,
            )
        except Exception as exc:
            results.append(
                _input_schema_failure_row(
                    row,
                    output_row_index=output_index,
                    source_row_index=source_row_index,
                    error=f"{type(exc).__name__}: {exc}",
                    row_started=row_started,
                    clock=clock,
                    run_metadata=run_metadata,
                )
            )
            continue
        try:
            results.append(
                _run_exp013_entry(
                    effective_entry,
                    repo_root=root,
                    baseline_v2_snapshot=baseline_snapshot,
                    beatthis_cache_loader=beatthis_cache_loader,
                    beatthis_cache_path_resolver=beatthis_cache_path_resolver,
                    mel_loader=mel_loader,
                    raw_evidence_extractor=raw_evidence_extractor,
                    candidate_generator=candidate_generator,
                    weak_oracle_loader=weak_oracle_loader,
                    metric_evaluator=metric_evaluator,
                    frozen_payload_observer=frozen_payload_observer,
                    clock=clock,
                    run_metadata=run_metadata,
                )
            )
        except Exception as exc:
            results.append(
                _hard_failure_row(
                    effective_entry,
                    failure_stage="inference",
                    error=f"{type(exc).__name__}: {exc}",
                    row_started=row_started,
                    clock=clock,
                    run_metadata=run_metadata,
                )
            )

    _write_jsonl_atomic(output_path, results)
    output_jsonl_sha256 = _file_sha256(output_path)
    summary = _summary_payload(
        results,
        pilot_path=pilot_path,
        pilot_sha256=pilot_snapshot.sha256,
        baseline_path=baseline_path,
        baseline_sha256=baseline_snapshot.sha256_if_available(),
        output_path=output_path,
        output_sha256=output_jsonl_sha256,
        summary_path=summary_path,
        total_seconds=clock() - started,
        run_metadata=run_metadata,
    )
    _write_json_atomic(summary_path, summary)
    return summary


def _require_distinct_input_output_paths(
    *,
    pilot_path: Path,
    baseline_path: Path | None,
    output_path: Path,
    summary_path: Path,
) -> None:
    named_paths = [
        ("pilot_jsonl_path", pilot_path.resolve(strict=False)),
        ("output_jsonl_path", output_path.resolve(strict=False)),
        ("summary_json_path", summary_path.resolve(strict=False)),
    ]
    if baseline_path is not None:
        named_paths.append(
            ("baseline_v2_jsonl_path", baseline_path.resolve(strict=False))
        )
    for left_index, (left_name, left_path) in enumerate(named_paths):
        for right_name, right_path in named_paths[left_index + 1 :]:
            if left_path == right_path:
                raise ValueError(f"{left_name} and {right_name} must differ")


def _validate_production_selection(tempo_track: TempoTrackResult) -> None:
    selection = tempo_track.production_selection
    if selection is None:
        raise ValueError("TempoTrackResult.production_selection is required")

    candidate_count = len(tempo_track.candidates)
    for index in selection.eligible_candidate_indices:
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("eligible candidate indices must be integers")
        if index < 0 or index >= candidate_count:
            raise ValueError("eligible candidate index out of range")

    if selection.status == "v2_fallback":
        if selection.lane != "fallback":
            raise ValueError("v2_fallback selection must use fallback lane")
        if not selection.fallback_reason:
            raise ValueError("v2_fallback selection must include fallback_reason")
        if selection.selected_candidate_index is not None:
            raise ValueError("v2_fallback selection cannot carry selected candidate")
        if selection.selected_fingerprint_sha256 is not None:
            raise ValueError("v2_fallback selection cannot carry selected fingerprint")
        if selection.eligible_candidate_indices:
            raise ValueError("v2_fallback selection cannot carry eligible candidates")
        return

    if selection.status != "v3_accepted":
        raise ValueError(f"unsupported production selection status {selection.status!r}")
    if selection.lane == "fallback":
        raise ValueError("v3_accepted selection cannot use fallback lane")
    if selection.fallback_reason is not None:
        raise ValueError("v3_accepted selection cannot include fallback_reason")
    selected_index = selection.selected_candidate_index
    if not isinstance(selected_index, int) or isinstance(selected_index, bool):
        raise ValueError("v3_accepted selection must include selected candidate index")
    if selected_index < 0 or selected_index >= candidate_count:
        raise ValueError("v3_accepted selected candidate index out of range")
    if selected_index not in selection.eligible_candidate_indices:
        raise ValueError("v3_accepted selected candidate must be production eligible")
    selected = tempo_track.candidates[selected_index]
    if selection.selected_fingerprint_sha256 != selected.fingerprint_sha256:
        raise ValueError("v3_accepted selected fingerprint mismatch")


def _run_exp013_entry(
    entry: _PilotEntry,
    *,
    repo_root: Path,
    baseline_v2_snapshot: _LazyJsonlSnapshot,
    beatthis_cache_loader: BeatThisCacheLoader,
    beatthis_cache_path_resolver: BeatThisCachePathResolver,
    mel_loader: MelLoader,
    raw_evidence_extractor: RawEvidenceExtractor,
    candidate_generator: Exp013CandidateGenerator,
    weak_oracle_loader: WeakOracleLoader,
    metric_evaluator: CurveMetricEvaluator,
    frozen_payload_observer: FrozenPayloadObserver | None,
    clock: Clock,
    run_metadata: _PilotRunMetadata,
) -> dict[str, Any]:
    row_started = clock()
    inference_started = clock()

    prediction = beatthis_cache_loader(entry.cache_audio_key)
    if prediction is None:
        raise ValueError("BeatThis shift-0 cache is unavailable")

    mel, mel_source, mel_path = mel_loader(entry.audio_path, repo_root=repo_root)
    evidence = raw_evidence_extractor(
        mel,
        audio_duration_seconds=entry.duration_seconds,
    )
    tempo_track = candidate_generator(prediction, audio_evidence=evidence)
    if not isinstance(tempo_track, TempoTrackResult):
        raise TypeError("candidate_generator must return a TempoTrackResult")
    selection = tempo_track.production_selection
    if selection is None:
        raise ValueError("TempoTrackResult.production_selection is required")
    _validate_production_selection(tempo_track)

    frozen_payload = _frozen_inference_payload(
        entry,
        beatthis_cache_path=beatthis_cache_path_resolver(entry.cache_audio_key),
        mel_source=mel_source,
        mel_path=mel_path,
        tempo_track=tempo_track,
        run_metadata=run_metadata,
    )
    frozen_sha256 = _stable_json_sha256(frozen_payload)
    if frozen_payload_observer is not None:
        frozen_payload_observer(frozen_payload, frozen_sha256)

    selected_candidate = tempo_track.selected_candidate
    selected_curve = selected_candidate.to_dict() if selected_candidate is not None else None
    if selected_candidate is None:
        maximum_seam_ms = None
    else:
        seam_values = [
            report.phase_discontinuity_ms for report in selected_candidate.seam_reports
        ]
        maximum_seam_ms = max(seam_values) if seam_values else 0.0

    inference_seconds = clock() - inference_started
    base_result: dict[str, Any] = {
        "schema": run_metadata.result_schema,
        "row_index": entry.output_row_index,
        "source_row_index": entry.source_row_index,
        "cache_audio_key": entry.cache_audio_key,
        "cache_audio_key_sha256": _sha256_text(entry.cache_audio_key),
        "audio_path": entry.audio_path.as_posix(),
        "duration_seconds": entry.duration_seconds,
        "stratum": entry.stratum,
        "confidence": entry.confidence,
        "product_status": selection.status,
        "selection_lane": selection.lane,
        "fallback_reason": selection.fallback_reason,
        "selected_candidate_index": selection.selected_candidate_index,
        "selected_fingerprint_sha256": selection.selected_fingerprint_sha256,
        "selected_curve_class": None
        if selected_candidate is None
        else selected_candidate.curve_class,
        "selected_curve": selected_curve,
        "maximum_seam_ms": maximum_seam_ms,
        "candidate_count": len(tempo_track.candidates),
        "eligible_candidate_indices": list(selection.eligible_candidate_indices),
        "raw_run": None
        if selection.raw_run is None
        else {
            "direction": selection.raw_run.direction,
            "start_time_ms": selection.raw_run.start_time_ms,
            "end_time_ms": selection.raw_run.end_time_ms,
            "expanded_start_time_ms": selection.raw_run.expanded_start_time_ms,
            "expanded_end_time_ms": selection.raw_run.expanded_end_time_ms,
            "median_bpm": selection.raw_run.median_bpm,
            "weighted_median_delta_bpm": selection.raw_run.weighted_median_delta_bpm,
            "observation_count": selection.raw_run.observation_count,
            "summed_strength": selection.raw_run.summed_strength,
        },
        "frozen_inference_sha256": frozen_sha256,
        "frozen_inference": frozen_payload,
        "inference_runtime_seconds": inference_seconds,
        "evaluation_runtime_seconds": None,
        "row_runtime_seconds": None,
        "weak_oracle_evaluation": None,
        "failure_stage": None,
        "error": None,
    }

    if selection.status == "v2_fallback":
        baseline_status = _baseline_v2_status_post_freeze(
            baseline_v2_snapshot,
            entry.cache_audio_key,
        )
        base_result["weak_oracle_evaluation"] = {
            "available": False,
            "unavailable_reason": "product_status_v2_fallback",
            "policy": run_metadata.weak_oracle_policy,
            "frozen_inference_sha256": frozen_sha256,
            "selected_metrics": None,
            "baseline_v2": baseline_status,
        }
        base_result["row_runtime_seconds"] = clock() - row_started
        return base_result

    if selection.status != "v3_accepted":
        raise ValueError(f"unsupported production selection status {selection.status!r}")
    if selected_candidate is None:
        raise ValueError("v3_accepted selection must include a selected candidate")

    evaluation_started = clock()
    try:
        base_result["weak_oracle_evaluation"] = _evaluate_v3_selection_post_freeze(
            entry,
            selected_candidate=selected_candidate,
            baseline_v2_snapshot=baseline_v2_snapshot,
            weak_oracle_loader=weak_oracle_loader,
            metric_evaluator=metric_evaluator,
            frozen_inference_sha256=frozen_sha256,
            run_metadata=run_metadata,
        )
    except Exception as exc:
        base_result["weak_oracle_evaluation"] = {
            "available": False,
            "unavailable_reason": "weak_oracle_evaluation_error",
            "policy": run_metadata.weak_oracle_policy,
            "frozen_inference_sha256": frozen_sha256,
            "selected_metrics": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
        base_result["evaluation_runtime_seconds"] = clock() - evaluation_started
        base_result["row_runtime_seconds"] = clock() - row_started
        return base_result

    base_result["evaluation_runtime_seconds"] = clock() - evaluation_started
    base_result["row_runtime_seconds"] = clock() - row_started
    return base_result


def _evaluate_v3_selection_post_freeze(
    entry: _PilotEntry,
    *,
    selected_candidate: PhaseContinuousTimingCurve,
    baseline_v2_snapshot: _LazyJsonlSnapshot,
    weak_oracle_loader: WeakOracleLoader,
    metric_evaluator: CurveMetricEvaluator,
    frozen_inference_sha256: str,
    run_metadata: _PilotRunMetadata,
) -> dict[str, Any]:
    representative = entry.row.get("representative_redline_grid")
    if not isinstance(representative, Mapping):
        return {
            "available": False,
            "unavailable_reason": "representative_redline_grid_unavailable",
            "policy": run_metadata.weak_oracle_policy,
            "frozen_inference_sha256": frozen_inference_sha256,
            "selected_metrics": None,
        }
    beatmap_path_value = representative.get("beatmap_path")
    if not isinstance(beatmap_path_value, str) or not beatmap_path_value:
        return {
            "available": False,
            "unavailable_reason": "representative_beatmap_path_unavailable",
            "policy": run_metadata.weak_oracle_policy,
            "frozen_inference_sha256": frozen_inference_sha256,
            "selected_metrics": None,
        }

    weak_oracle_class = _weak_oracle_class(entry.stratum, representative)
    weak_oracle = weak_oracle_loader(Path(beatmap_path_value))
    coverage_start_ms = max(0.0, selected_candidate.start_time_ms)
    coverage_end_ms = min(
        entry.duration_seconds * 1000.0,
        selected_candidate.end_time_ms,
    )
    coverage_end_ms = float(np.nextafter(coverage_end_ms, -math.inf))
    if coverage_end_ms <= coverage_start_ms:
        raise ValueError("weak-oracle coverage interval is empty")

    metrics = metric_evaluator(
        selected_candidate,
        weak_oracle,
        weak_oracle_class=weak_oracle_class,
        coverage_start_ms=coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
    )
    metrics_payload = metrics.to_dict() if hasattr(metrics, "to_dict") else metrics
    baseline_payload = _baseline_v2_metrics_post_freeze(
        baseline_v2_snapshot,
        entry.cache_audio_key,
        weak_oracle=weak_oracle,
        weak_oracle_class=weak_oracle_class,
        coverage_start_ms=coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
        metric_evaluator=metric_evaluator,
    )
    return {
        "available": True,
        "policy": run_metadata.weak_oracle_policy,
        "frozen_inference_sha256": frozen_inference_sha256,
        "weak_oracle_class": weak_oracle_class,
        "weak_oracle_beatmap_path": beatmap_path_value,
        "weak_oracle_agreement_rate": representative.get("agreement_rate"),
        "selected_metrics": metrics_payload,
        "baseline_v2": baseline_payload,
    }


def _frozen_inference_payload(
    entry: _PilotEntry,
    *,
    beatthis_cache_path: Path,
    mel_source: str,
    mel_path: Path | None,
    tempo_track: TempoTrackResult,
    run_metadata: _PilotRunMetadata,
) -> dict[str, Any]:
    selection = tempo_track.production_selection
    if selection is None:
        raise ValueError("TempoTrackResult.production_selection is required")
    return {
        "schema": run_metadata.frozen_inference_schema,
        "inference_family": run_metadata.inference_family,
        "row_index": entry.output_row_index,
        "source_row_index": entry.source_row_index,
        "cache_audio_key": entry.cache_audio_key,
        "cache_audio_key_sha256": _sha256_text(entry.cache_audio_key),
        "audio_path": entry.audio_path.as_posix(),
        "duration_seconds": entry.duration_seconds,
        "beatthis_cache_path": beatthis_cache_path.as_posix(),
        "mel_source": mel_source,
        "mel_path": None if mel_path is None else mel_path.as_posix(),
        "product_status": selection.status,
        "selected_candidate_index": selection.selected_candidate_index,
        "selected_fingerprint_sha256": selection.selected_fingerprint_sha256,
        "fallback_reason": selection.fallback_reason,
        "tempo_track": tempo_track_result_to_dict(
            tempo_track,
            include_observations=False,
        ),
    }


def _select_exp013_entries(
    rows: Sequence[Mapping[str, Any]],
    *,
    explicit_cache_audio_keys: frozenset[str] | None,
) -> list[_PilotEntry]:
    entries: list[_PilotEntry] = []
    for source_row_index, row in enumerate(rows):
        if not _row_matches_exp013_filter(
            row,
            explicit_cache_audio_keys=explicit_cache_audio_keys,
        ):
            continue
        entry = _entry_from_row(
            row,
            source_row_index=source_row_index,
            output_row_index=len(entries),
        )
        entries.append(entry)
    return entries


def _row_matches_exp013_filter(
    row: Mapping[str, Any],
    *,
    explicit_cache_audio_keys: frozenset[str] | None,
) -> bool:
    confidence = _nested(row, "label", "confidence")
    ambiguous = _nested(row, "label", "ambiguous")
    stratum = _nested(row, "label", "stratum") or row.get("pilot_stratum")
    cache_audio_key = _nested(row, "source", "cache_audio_key")
    if confidence not in EXP013_ALLOWED_CONFIDENCES:
        return False
    if ambiguous is not False:
        return False
    if stratum not in EXP013_ALLOWED_STRATA:
        return False
    if explicit_cache_audio_keys is not None and cache_audio_key not in explicit_cache_audio_keys:
        return False
    return True


def _entry_from_row(
    row: Mapping[str, Any],
    *,
    source_row_index: int,
    output_row_index: int,
) -> _PilotEntry:
    cache_audio_key = _require_string(
        _nested(row, "source", "cache_audio_key"),
        "source.cache_audio_key",
    )
    audio_path_value = _require_string(row.get("resolved_audio_path"), "resolved_audio_path")
    duration_seconds = _positive_float(
        _nested(row, "source", "cache_duration_seconds"),
        "source.cache_duration_seconds",
    )
    stratum = _require_string(
        _nested(row, "label", "stratum") or row.get("pilot_stratum"),
        "label.stratum",
    )
    confidence = _require_string(
        _nested(row, "label", "confidence"),
        "label.confidence",
    )
    return _PilotEntry(
        output_row_index=output_row_index,
        source_row_index=source_row_index,
        cache_audio_key=cache_audio_key,
        audio_path=Path(audio_path_value),
        duration_seconds=duration_seconds,
        stratum=stratum,
        confidence=confidence,
        row=row,
    )


def _hard_failure_row(
    entry: _PilotEntry,
    *,
    failure_stage: str,
    error: str,
    row_started: float | None,
    clock: Clock,
    run_metadata: _PilotRunMetadata,
) -> dict[str, Any]:
    runtime = None if row_started is None else clock() - row_started
    return {
        "schema": run_metadata.result_schema,
        "row_index": entry.output_row_index,
        "source_row_index": entry.source_row_index,
        "cache_audio_key": entry.cache_audio_key,
        "cache_audio_key_sha256": _sha256_text(entry.cache_audio_key),
        "audio_path": entry.audio_path.as_posix(),
        "duration_seconds": entry.duration_seconds,
        "stratum": entry.stratum,
        "confidence": entry.confidence,
        "product_status": "hard_failure",
        "selection_lane": None,
        "fallback_reason": None,
        "selected_candidate_index": None,
        "selected_fingerprint_sha256": None,
        "selected_curve_class": None,
        "selected_curve": None,
        "maximum_seam_ms": None,
        "candidate_count": None,
        "eligible_candidate_indices": [],
        "raw_run": None,
        "frozen_inference_sha256": None,
        "frozen_inference": None,
        "inference_runtime_seconds": None,
        "evaluation_runtime_seconds": None,
        "row_runtime_seconds": runtime,
        "weak_oracle_evaluation": None,
        "failure_stage": failure_stage,
        "error": error,
    }


def _input_schema_failure_row(
    row: Mapping[str, Any],
    *,
    output_row_index: int,
    source_row_index: int,
    error: str,
    row_started: float,
    clock: Clock,
    run_metadata: _PilotRunMetadata,
) -> dict[str, Any]:
    cache_audio_key_value = _nested(row, "source", "cache_audio_key")
    cache_audio_key = (
        cache_audio_key_value
        if isinstance(cache_audio_key_value, str) and cache_audio_key_value
        else None
    )
    audio_path_value = row.get("resolved_audio_path")
    audio_path = (
        audio_path_value
        if isinstance(audio_path_value, str) and audio_path_value
        else None
    )
    duration_value = _nested(row, "source", "cache_duration_seconds")
    duration_seconds = (
        float(duration_value)
        if isinstance(duration_value, (int, float))
        and not isinstance(duration_value, bool)
        and math.isfinite(float(duration_value))
        else None
    )
    return {
        "schema": run_metadata.result_schema,
        "row_index": output_row_index,
        "source_row_index": source_row_index,
        "cache_audio_key": cache_audio_key,
        "cache_audio_key_sha256": None
        if cache_audio_key is None
        else _sha256_text(cache_audio_key),
        "audio_path": audio_path,
        "duration_seconds": duration_seconds,
        "stratum": _nested(row, "label", "stratum") or row.get("pilot_stratum"),
        "confidence": _nested(row, "label", "confidence"),
        "product_status": "hard_failure",
        "selection_lane": None,
        "fallback_reason": None,
        "selected_candidate_index": None,
        "selected_fingerprint_sha256": None,
        "selected_curve_class": None,
        "selected_curve": None,
        "maximum_seam_ms": None,
        "candidate_count": None,
        "eligible_candidate_indices": [],
        "raw_run": None,
        "frozen_inference_sha256": None,
        "frozen_inference": None,
        "inference_runtime_seconds": None,
        "evaluation_runtime_seconds": None,
        "row_runtime_seconds": clock() - row_started,
        "weak_oracle_evaluation": None,
        "failure_stage": "input_schema",
        "error": error,
    }


def _summary_payload(
    results: Sequence[Mapping[str, Any]],
    *,
    pilot_path: Path,
    pilot_sha256: str,
    baseline_path: Path | None,
    baseline_sha256: str | None,
    output_path: Path,
    output_sha256: str,
    summary_path: Path,
    total_seconds: float,
    run_metadata: _PilotRunMetadata,
) -> dict[str, Any]:
    status_counts = Counter(_string_or_unknown(row.get("product_status")) for row in results)
    stratum_counts = Counter(_string_or_unknown(row.get("stratum")) for row in results)
    status_counts_by_stratum: dict[str, dict[str, int]] = {}
    for row in results:
        stratum = _string_or_unknown(row.get("stratum"))
        status = _string_or_unknown(row.get("product_status"))
        status_counts_by_stratum.setdefault(stratum, {})
        status_counts_by_stratum[stratum][status] = (
            status_counts_by_stratum[stratum].get(status, 0) + 1
        )
    fallback_reason_counts = Counter(
        _string_or_unknown(row.get("fallback_reason"))
        for row in results
        if row.get("product_status") == "v2_fallback"
    )
    runtimes = [
        float(row["row_runtime_seconds"])
        for row in results
        if isinstance(row.get("row_runtime_seconds"), (int, float))
    ]
    return {
        "schema": run_metadata.summary_schema,
        "input_pilot_jsonl_path": pilot_path.as_posix(),
        "input_pilot_jsonl_sha256": pilot_sha256,
        "input_baseline_v2_jsonl_path": None
        if baseline_path is None
        else baseline_path.as_posix(),
        "input_baseline_v2_jsonl_sha256": baseline_sha256,
        "output_jsonl_path": output_path.as_posix(),
        "output_jsonl_sha256": output_sha256,
        "summary_json_path": summary_path.as_posix(),
        "row_count": len(results),
        "status_counts": dict(sorted(status_counts.items())),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "status_counts_by_stratum": {
            stratum: dict(sorted(counts.items()))
            for stratum, counts in sorted(status_counts_by_stratum.items())
        },
        "fallback_reason_counts": dict(sorted(fallback_reason_counts.items())),
        "v3_accepted_count": status_counts.get("v3_accepted", 0),
        "v2_fallback_count": status_counts.get("v2_fallback", 0),
        "hard_failure_count": status_counts.get("hard_failure", 0),
        "fallback_rate": (
            None
            if not results
            else status_counts.get("v2_fallback", 0) / len(results)
        ),
        "maximum_seam_ms": _numeric_max(
            row.get("maximum_seam_ms") for row in results
        ),
        "accepted_weak_metrics": _accepted_weak_metrics_summary(results),
        "total_runtime_seconds": float(total_seconds),
        "row_runtime_seconds": _runtime_summary(runtimes),
    }


def _accepted_weak_metrics_summary(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | None]]:
    metric_names = (
        "weak_oracle_phase_mean_ms",
        "weak_oracle_phase_p50_ms",
        "weak_oracle_phase_p90_ms",
        "weak_oracle_phase_max_ms",
        "weak_oracle_local_bpm_alias_mae",
        "weak_oracle_endpoint_relative_drift_ms",
        "weak_oracle_endpoint_abs_relative_drift_ms",
        "weak_oracle_max_abs_prefix_relative_drift_ms",
    )
    values_by_metric: dict[str, list[float]] = {name: [] for name in metric_names}
    for row in results:
        if row.get("product_status") != "v3_accepted":
            continue
        evaluation = row.get("weak_oracle_evaluation")
        if not isinstance(evaluation, Mapping) or not evaluation.get("available"):
            continue
        metrics = evaluation.get("selected_metrics")
        if not isinstance(metrics, Mapping):
            continue
        for name in metric_names:
            if name == "weak_oracle_endpoint_abs_relative_drift_ms":
                value = metrics.get("weak_oracle_endpoint_relative_drift_ms")
            else:
                value = metrics.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                number = float(value)
                if math.isfinite(number):
                    if name == "weak_oracle_endpoint_abs_relative_drift_ms":
                        number = abs(number)
                    values_by_metric[name].append(number)
    return {name: _numeric_summary(values) for name, values in values_by_metric.items()}


def _numeric_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p90": None, "max": None}
    ordered = np.asarray(sorted(values), dtype=np.float64)
    return {
        "mean": float(np.mean(ordered)),
        "p50": float(np.percentile(ordered, 50)),
        "p90": float(np.percentile(ordered, 90)),
        "max": float(ordered[-1]),
    }


def _numeric_max(values: Sequence[object]) -> float | None:
    numbers = [
        float(value)
        for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]
    if not numbers:
        return None
    return max(numbers)


def _baseline_v2_status_post_freeze(
    baseline_v2_snapshot: _LazyJsonlSnapshot,
    cache_audio_key: str,
) -> dict[str, Any]:
    if baseline_v2_snapshot.path is None:
        return {"available": False, "unavailable_reason": "baseline_v2_jsonl_not_supplied"}
    row = baseline_v2_snapshot.row_by_audio_key(cache_audio_key)
    if row is None:
        return {"available": False, "unavailable_reason": "baseline_v2_row_not_found"}
    return {**_baseline_v2_status_payload(row), "metrics": None}


def _baseline_v2_metrics_post_freeze(
    baseline_v2_snapshot: _LazyJsonlSnapshot,
    cache_audio_key: str,
    *,
    weak_oracle: FittedTimingGrid,
    weak_oracle_class: WeakOracleClass,
    coverage_start_ms: float,
    coverage_end_ms: float,
    metric_evaluator: CurveMetricEvaluator,
) -> dict[str, Any]:
    status = _baseline_v2_status_post_freeze(baseline_v2_snapshot, cache_audio_key)
    if not status["available"]:
        return status
    row = baseline_v2_snapshot.row_by_audio_key(cache_audio_key)
    if row is None:
        return {"available": False, "unavailable_reason": "baseline_v2_row_not_found"}
    grid = _baseline_grid(row)
    if grid is None:
        return {
            **_baseline_v2_status_payload(row),
            "metrics_available": False,
            "metrics_unavailable_reason": "baseline_v2_grid_unavailable",
            "metrics": None,
        }
    try:
        metrics = metric_evaluator(
            grid,  # type: ignore[arg-type]
            weak_oracle,
            weak_oracle_class=weak_oracle_class,
            coverage_start_ms=coverage_start_ms,
            coverage_end_ms=coverage_end_ms,
        )
    except ValueError as exc:
        return {
            **_baseline_v2_status_payload(row),
            "metrics_available": False,
            "metrics_unavailable_reason": f"{type(exc).__name__}: {exc}",
            "metrics": None,
        }
    return {
        **_baseline_v2_status_payload(row),
        "metrics_available": True,
        "metrics": metrics.to_dict() if hasattr(metrics, "to_dict") else metrics,
    }


def _baseline_v2_status_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    ok_value = row.get("ok")
    status_value = row.get("status") or _nested(row, "fit", "status")
    if isinstance(ok_value, bool):
        ok = ok_value
    elif status_value == "ok":
        ok = True
    elif status_value in ("failed", "failure", "error"):
        ok = False
    else:
        ok = None

    if ok is True:
        normalized_status = "ok"
    elif ok is False:
        normalized_status = "failed"
    else:
        normalized_status = _string_or_unknown(status_value)

    return {
        "available": True,
        "ok": ok,
        "status": normalized_status,
        "failure_stage": row.get("failure_stage"),
        "error_type": row.get("error_type"),
    }


def _baseline_grid(row: Mapping[str, Any]) -> FittedTimingGrid | None:
    fit = row.get("fit")
    if not isinstance(fit, Mapping):
        return None
    segments = fit.get("predicted_segments")
    if (
        not isinstance(segments, Sequence)
        or isinstance(segments, (str, bytes, bytearray))
        or not segments
    ):
        return None
    parsed = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            return None
        parsed.append(
            TimingSegment(
                offset_ms=float(segment["offset_ms"]),
                beat_length_ms=float(segment["beat_length_ms"]),
                meter=int(segment.get("meter", 4)),
            )
        )
    return FittedTimingGrid(tuple(parsed))


def _runtime_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p90": None, "max": None}
    ordered = np.asarray(sorted(values), dtype=np.float64)
    return {
        "p50": float(np.percentile(ordered, 50)),
        "p90": float(np.percentile(ordered, 90)),
        "max": float(ordered[-1]),
    }


def _weak_oracle_class(
    stratum: str,
    representative: Mapping[str, Any],
) -> WeakOracleClass:
    evidence_class = representative.get("evidence_class")
    if stratum in ("jump_candidate", "jump") or evidence_class == "jump_candidate":
        return "jump"
    if evidence_class == "ramp_candidate":
        return "ramp_like"
    return "constant"


def _load_log_mel_10ms(
    audio_path: Path,
    *,
    repo_root: Path,
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], str, Path | None]:
    resolved_audio = audio_path.expanduser().resolve(strict=False)
    try:
        audio_cache_key = resolved_audio.relative_to(repo_root).as_posix()
    except ValueError:
        audio_cache_key = resolved_audio.as_posix()
    cache_path = stage2_log_mel_cache_path(
        resolved_audio,
        audio_cache_key=audio_cache_key,
    )
    if cache_path.exists():
        return (
            np.load(cache_path).astype(np.float32, copy=False),
            "existing_10ms_mel_cache",
            cache_path,
        )
    waveform = load_audio_file(
        resolved_audio,
        sample_rate=DEFAULT_MEL_CACHE_CONFIG.sample_rate,
    )
    mel = compute_log_mel_10ms(
        waveform,
        sample_rate=DEFAULT_MEL_CACHE_CONFIG.sample_rate,
    )
    return mel, "decoded_in_memory", None


def _read_jsonl_snapshot(path: Path) -> _JsonlSnapshot:
    raw = path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    rows: list[Mapping[str, Any]] = []
    text = raw.decode("utf-8")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path}:{line_number} must decode to an object")
        rows.append(payload)
    return _JsonlSnapshot(path=path, sha256=sha256, rows=tuple(rows))


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    return list(_read_jsonl_snapshot(path).rows)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            for row in rows:
                handle.write(_stable_json(row))
                handle.write("\n")
        tmp_path.replace(path)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(_stable_json(payload))
            handle.write("\n")
        tmp_path.replace(path)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _stable_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _stable_json_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nested(row: Mapping[str, Any], *keys: str) -> Any:
    current: Any = row
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_float(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _string_or_unknown(value: object) -> str:
    return value if isinstance(value, str) and value else "unknown"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the narrow Timing v3 Exp013 exposed pilot adapter. "
            "No full-corpus default is provided."
        )
    )
    parser.add_argument("--pilot-jsonl", required=True, help="Exposed pilot JSONL input")
    parser.add_argument(
        "--baseline-v2-jsonl",
        default=None,
        help="Optional v2 baseline JSONL, parsed only after frozen inference per row.",
    )
    parser.add_argument("--output-jsonl", required=True, help="Atomic per-row JSONL output")
    parser.add_argument("--summary-json", required=True, help="Atomic summary JSON output")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--cache-audio-key",
        action="append",
        dest="cache_audio_keys",
        default=None,
        help="Optional explicit cache_audio_key filter; may be repeated.",
    )
    args = parser.parse_args(argv)
    summary = run_exp013_pilot(
        pilot_jsonl_path=args.pilot_jsonl,
        baseline_v2_jsonl_path=args.baseline_v2_jsonl,
        output_jsonl_path=args.output_jsonl,
        summary_json_path=args.summary_json,
        repo_root=args.repo_root,
        limit=args.limit,
        explicit_cache_audio_keys=args.cache_audio_keys,
    )
    print(_stable_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
