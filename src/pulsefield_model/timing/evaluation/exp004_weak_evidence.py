from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from pulsefield_model.timing.canonicalization import (
    TIMING_CANONICALIZATION_BPM_80_160,
    TIMING_CANONICALIZATION_NONE,
)
from pulsefield_model.timing.evaluation import exp004_runner
from pulsefield_model.timing.evaluation import exp004_protocol
from pulsefield_model.timing.evaluation.exp004_metrics import (
    EXP004_METRICS_SCHEMA,
    Exp004DenominatorRow,
    active_section_signature_v1,
    aggregate_weak_boundary_audio_metrics,
    canonical_bpm_binding_for_exp004,
    classify_exp004_denominators,
    classify_exp004_primary_guards,
    compare_phase_sampling_v1,
    extract_tempo_change_boundaries,
    extract_weak_redline_boundaries,
    weak_boundary_difficulty_metrics,
)
from pulsefield_model.timing.schema import FittedTimingGrid, TimingSegment
from pulsefield_model.timing.v3.schema import TimingV3Grid


RESULT_SCHEMA = "pulsefield_model.timing_v3_exp004_weak_result_v1"
SUMMARY_SCHEMA = "pulsefield_model.timing_v3_exp004_weak_summary_v1"
RESUME_SCHEMA = "pulsefield_model.timing_v3_exp004_weak_resume_v1"

BASELINE_RESULT_SCHEMA = "pulsefield_model.timing_v3_cache_backed_v2_baseline_result_v2"
PRIMARY_CANONICALIZATION = TIMING_CANONICALIZATION_BPM_80_160
SELECTED_METHOD = "selected_CJ3_or_current_v2_fallback"
CURRENT_V2_METHOD = "current_v2"
VARIANT_METHODS = ("CJ0", "CJ1", "CJ2", "CJ3")
METHODS = (CURRENT_V2_METHOD, *VARIANT_METHODS, SELECTED_METHOD)
UNDEFINED_RATIO_VALUE: None = None
ZERO_RATIO_EPSILON = 1e-12
SYNTHETIC_FIXTURE_STAGE = "synthetic_fixture"
FORMAL_STAGE_AUDIO_COUNTS = dict(exp004_runner.STAGE_AUDIO_COUNTS)
FORMAL_DECISION_STAGES = ("holdout100", "broad500", "full5050")
STAGE_STATUS_PASS = "pass"
STAGE_STATUS_AMBIGUOUS = "ambiguous"
STAGE_STATUS_KILL = "kill"
STAGE_STATUS_NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class _ProjectionRow:
    row_index: int
    audio_key: str
    audio_group_key: str
    payload: dict[str, Any]
    payload_sha256: str


@dataclass(frozen=True)
class _ProjectionSource:
    path: Path
    sha256: str
    summary_path: Path
    summary_sha256: str
    summary: dict[str, Any]
    stage: str
    behavior_fingerprint: str
    config_fingerprint: str
    run_fingerprint: str
    hard_guards_ok: bool
    hard_guard_violations: tuple[dict[str, Any], ...]
    formal_execution_ready: bool
    formal_execution_blockers: tuple[str, ...]
    stage_constraints: Mapping[str, Any]
    prior_baseline_jsonl_sha256: str | None


@dataclass(frozen=True)
class _StoredComparator:
    index: int
    difficulty_key: str
    beatmap_path: str | None
    payload_sha256: str
    oracle_segments_payload: Any
    oracle_grid: FittedTimingGrid | None
    valid: bool
    error_type: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class _BaselineRow:
    row_index: int
    audio_key: str
    payload: dict[str, Any]
    payload_sha256: str
    v2_grid: FittedTimingGrid | None
    comparators: tuple[_StoredComparator, ...]
    evaluation_strata: Mapping[str, Any]


@dataclass(frozen=True)
class _BaselineSource:
    path: Path
    sha256: str
    rows_by_key: Mapping[str, _BaselineRow]


def run_exp004_weak_evidence(
    *,
    projection_jsonl_path: str | Path,
    projection_summary_path: str | Path,
    baseline_jsonl_path: str | Path,
    output_jsonl_path: str | Path,
    summary_json_path: str | Path | None = None,
    retry_failures: bool = False,
    checkpoint_every: int = 1,
) -> dict[str, Any]:
    """Evaluate Exp004 projections against stored weak oracle redline segments.

    This layer consumes only projection-runner JSON plus durable baseline JSONL
    records that already contain oracle segment payloads. It never reads maps,
    labels, or oracle providers and never feeds comparator data back into core
    projection.
    """

    if isinstance(checkpoint_every, bool) or not isinstance(checkpoint_every, int) or checkpoint_every < 1:
        raise ValueError("checkpoint_every must be a positive integer")
    if not isinstance(retry_failures, bool):
        raise TypeError("retry_failures must be a bool")

    projection_path = Path(projection_jsonl_path)
    projection_summary = Path(projection_summary_path)
    baseline_path = Path(baseline_jsonl_path)
    output_path = Path(output_jsonl_path)
    summary_path = (
        Path(summary_json_path)
        if summary_json_path is not None
        else output_path.with_suffix(output_path.suffix + ".summary.json")
    )
    lock_path = output_path.with_name(f".{output_path.name}.lock")
    _reject_path_aliases(
        {
            "projection_jsonl_path": projection_path,
            "projection_summary_path": projection_summary,
            "baseline_jsonl_path": baseline_path,
            "output_jsonl_path": output_path,
            "summary_json_path": summary_path,
            "lock_path": lock_path,
        }
    )

    projection_source, projection_rows = _load_projection_inputs(
        projection_path,
        projection_summary,
    )
    baseline_source = _load_baseline_source(baseline_path)
    _require_projection_prior_baseline_matches(projection_source, baseline_source)
    evaluator_source = _evaluator_source_payload()

    with _exclusive_run_lock(lock_path, run_fingerprint=projection_source.run_fingerprint):
        existing, stale_existing_count = _load_existing_results(
            output_path,
            projection_rows=projection_rows,
            projection_jsonl_sha256=projection_source.sha256,
        )
        results: list[dict[str, Any]] = []
        resume_counts = Counter(
            reused_success=0,
            reused_failure=0,
            recomputed_stale=stale_existing_count,
            processed=0,
        )
        started = time.time()

        for projection in projection_rows:
            baseline = baseline_source.rows_by_key.get(projection.audio_key)
            resume = _resume_payload(
                projection=projection,
                projection_source=projection_source,
                baseline=baseline,
                baseline_source=baseline_source,
                evaluator_source=evaluator_source,
            )
            previous = existing.get(projection.row_index)
            if previous is not None and _nested(previous, "resume", "fingerprint") == resume["fingerprint"]:
                if retry_failures and not bool(previous.get("ok")):
                    pass
                else:
                    results.append(previous)
                    resume_counts["reused_success" if bool(previous.get("ok")) else "reused_failure"] += 1
                    if len(results) % checkpoint_every == 0:
                        _write_jsonl_atomic(output_path, results)
                    continue
            elif previous is not None:
                resume_counts["recomputed_stale"] += 1

            result = _evaluate_projection_row(
                projection,
                baseline=baseline,
                baseline_source=baseline_source,
                projection_source=projection_source,
                resume=resume,
            )
            results.append(result)
            resume_counts["processed"] += 1
            if len(results) % checkpoint_every == 0:
                _write_jsonl_atomic(output_path, results)

        if len(results) != len(projection_rows):
            raise RuntimeError("weak-evidence result ordering invariant failed")
        _write_jsonl_atomic(output_path, results)
        _require_source_hashes_unchanged(projection_source, baseline_source)
        _require_evaluator_source_unchanged(evaluator_source)
        summary = _summary_payload(
            projection_source=projection_source,
            baseline_source=baseline_source,
            evaluator_source=evaluator_source,
            output_path=output_path,
            output_sha256=_file_sha256(output_path),
            results=results,
            resume_counts=resume_counts,
            total_seconds=time.time() - started,
        )
        _write_json_atomic(summary_path, summary)
        return summary


run_exp004_weak_oracle_evidence = run_exp004_weak_evidence


def _evaluate_projection_row(
    projection: _ProjectionRow,
    *,
    baseline: _BaselineRow | None,
    baseline_source: _BaselineSource,
    projection_source: _ProjectionSource,
    resume: Mapping[str, Any],
) -> dict[str, Any]:
    row = projection.payload
    coverage_end_ms = _projection_coverage_end_ms(row)
    grids = _projection_method_grids(row)
    valid_comparators = tuple(comparator for comparator in (baseline.comparators if baseline else ()) if comparator.valid)
    invalid_comparators = tuple(comparator for comparator in (baseline.comparators if baseline else ()) if not comparator.valid)
    projection_evaluable = bool(_nested(row, "projection_flags", "projection_evaluable"))
    comparison_eligible = projection_evaluable and bool(valid_comparators)

    if baseline is not None:
        _reject_stale_v2_mismatch(projection, baseline)

    methods: dict[str, Any] = {}
    boundary_difficulty_rows = {}
    for method in METHODS:
        grid = grids.get(method)
        if grid is None:
            methods[method] = _method_unavailable_payload(row, method)
            continue
        payload, boundary_rows = _method_metrics_payload(
            method,
            grid,
            valid_comparators,
            audio_key=projection.audio_key,
            coverage_end_ms=coverage_end_ms,
        )
        methods[method] = payload
        boundary_difficulty_rows[method] = boundary_rows

    current_v2_phase_matched = comparison_eligible and bool(
        methods.get(CURRENT_V2_METHOD, {}).get("comparison_matched")
    )
    pure_cj3_phase_matched = (
        current_v2_phase_matched
        and bool(methods.get("CJ3", {}).get("comparison_matched"))
    )
    selected_safety_phase_scored = (
        current_v2_phase_matched
        and bool(methods.get(SELECTED_METHOD, {}).get("comparison_matched"))
    )
    denominator_row = Exp004DenominatorRow(
        audio_key=projection.audio_key,
        cache_valid=bool(_nested(row, "projection_flags", "cache_valid")),
        projection_evaluable=projection_evaluable,
        comparison_eligible=comparison_eligible,
        pure_cj3_grid_produced=grids.get("CJ3") is not None,
        pure_cj3_phase_matched=pure_cj3_phase_matched,
        current_v2_phase_matched=current_v2_phase_matched,
        selected_safety_phase_scored=selected_safety_phase_scored,
        selected_used_fallback=bool(_nested(row, "projection_flags", "selected_used_fallback")),
    )

    result = {
        "schema": RESULT_SCHEMA,
        "row_complete": True,
        "ok": True,
        "stage": row.get("stage"),
        "row_index": projection.row_index,
        "identity": {
            "cache_audio_key": projection.audio_key,
            "audio_group_key": projection.audio_group_key,
        },
        "resume": dict(resume),
        "projection": {
            "jsonl_sha256": projection_source.sha256,
            "row_sha256": projection.payload_sha256,
            "row": projection.payload,
        },
        "baseline": _baseline_result_payload(
            baseline,
            baseline_source=baseline_source,
            invalid_comparators=invalid_comparators,
        ),
        "coverage": {"coverage_start_ms": 0.0, "coverage_end_ms": coverage_end_ms},
        "strata": _strata_payload(
            baseline.evaluation_strata if baseline is not None else {},
            row=row,
            coverage_end_ms=coverage_end_ms,
            comparison_eligible=comparison_eligible,
            valid_comparator_count=len(valid_comparators),
            invalid_comparator_count=len(invalid_comparators),
            methods=methods,
        ),
        "denominator_row": _json_safe(asdict(denominator_row)),
        "methods": methods,
        "reasons": _row_reasons(row, baseline, valid_comparators, invalid_comparators),
    }
    result = _json_safe(result)
    result["result_fingerprint"] = _stable_json_sha256(result)
    _canonical_json(result)
    return result


def _method_metrics_payload(
    method: str,
    grid: FittedTimingGrid | TimingV3Grid,
    valid_comparators: Sequence[_StoredComparator],
    *,
    audio_key: str,
    coverage_end_ms: float,
) -> tuple[dict[str, Any], list[Any]]:
    difficulty_metrics: list[dict[str, Any]] = []
    boundary_rows = []
    predicted_boundaries = extract_tempo_change_boundaries(
        grid,
        coverage_start_ms=0.0,
        coverage_end_ms=coverage_end_ms,
    )
    grid_summary = _grid_summary_payload(grid, coverage_end_ms=coverage_end_ms)
    for comparator in valid_comparators:
        assert comparator.oracle_grid is not None
        raw = compare_phase_sampling_v1(
            grid,
            comparator.oracle_grid,
            coverage_start_ms=0.0,
            coverage_end_ms=coverage_end_ms,
            canonicalization=TIMING_CANONICALIZATION_NONE,
        )
        alias = compare_phase_sampling_v1(
            grid,
            comparator.oracle_grid,
            coverage_start_ms=0.0,
            coverage_end_ms=coverage_end_ms,
            canonicalization=PRIMARY_CANONICALIZATION,
        )
        weak_extraction = extract_weak_redline_boundaries(
            comparator.oracle_grid,
            coverage_start_ms=0.0,
            coverage_end_ms=coverage_end_ms,
        )
        boundary = weak_boundary_difficulty_metrics(
            audio_key=audio_key,
            difficulty_key=comparator.difficulty_key,
            predicted_boundaries=predicted_boundaries,
            weak_redline_extraction=weak_extraction,
        )
        boundary_rows.append(boundary)
        difficulty_metrics.append(
            {
                "difficulty_key": comparator.difficulty_key,
                "beatmap_path": comparator.beatmap_path,
                "oracle_segments_sha256": comparator.payload_sha256,
                "raw": _phase_payload(raw),
                PRIMARY_CANONICALIZATION: _phase_payload(alias),
                "weak_boundary": _weak_boundary_payload(boundary),
            }
        )
    audio_raw = _audio_metric_payload(difficulty_metrics, canonicalization=TIMING_CANONICALIZATION_NONE)
    audio_alias = _audio_metric_payload(difficulty_metrics, canonicalization=PRIMARY_CANONICALIZATION)
    boundary_audio = (
        _weak_boundary_audio_payload(aggregate_weak_boundary_audio_metrics(tuple(boundary_rows))[0])
        if boundary_rows
        else None
    )
    return (
        {
            "method": method,
            "status": "scored" if difficulty_metrics else "no_valid_comparator",
            "comparison_matched": bool(difficulty_metrics),
            "difficulty_count": len(difficulty_metrics),
            "grid": grid_summary,
            "difficulty_metrics": difficulty_metrics,
            "audio_metrics": {
                TIMING_CANONICALIZATION_NONE: audio_raw,
                PRIMARY_CANONICALIZATION: audio_alias,
            },
            "weak_boundary_audio": boundary_audio,
        },
        boundary_rows,
    )


def _method_unavailable_payload(row: Mapping[str, Any], method: str) -> dict[str, Any]:
    return {
        "method": method,
        "status": "grid_unavailable",
        "comparison_matched": False,
        "reason": _method_unavailable_reason(row, method),
        "difficulty_count": 0,
        "grid": None,
        "difficulty_metrics": [],
        "audio_metrics": {
            TIMING_CANONICALIZATION_NONE: _empty_audio_metric_payload(),
            PRIMARY_CANONICALIZATION: _empty_audio_metric_payload(),
        },
        "weak_boundary_audio": None,
    }


def _phase_payload(comparison: Any) -> dict[str, Any]:
    payload = comparison.to_dict()
    return {
        "schema": payload["schema"],
        "sampling_version": payload["sampling_version"],
        "canonicalization": payload["canonicalization"],
        "coverage_start_ms": payload["coverage_start_ms"],
        "coverage_end_ms": payload["coverage_end_ms"],
        "sample_hop_ms": payload["sample_hop_ms"],
        "sample_count": payload["sample_count"],
        "phase_mean_ms": payload["phase_mean_ms"],
        "phase_p50_ms": payload["phase_p50_ms"],
        "phase_p90_ms": payload["phase_p90_ms"],
        "phase_max_ms": payload["phase_max_ms"],
        "phase_mean_beats": payload["phase_mean_beats"],
        "phase_p50_beats": payload["phase_p50_beats"],
        "phase_p90_beats": payload["phase_p90_beats"],
        "phase_max_beats": payload["phase_max_beats"],
        "local_bpm_mae": payload["local_bpm_mae"],
        "local_bpm_p90_abs_error": payload["local_bpm_p90_abs_error"],
        "local_bpm_alias_mae": payload["local_bpm_alias_mae"],
        "local_bpm_alias_p90_abs_error": payload["local_bpm_alias_p90_abs_error"],
        "initial_signed_phase_error_beats": payload["initial_signed_phase_error_beats"],
        "initial_signed_phase_error_ms": payload["initial_signed_phase_error_ms"],
        "endpoint_relative_drift_beats": payload["endpoint_relative_drift_beats"],
        "endpoint_relative_drift_ms": payload["endpoint_relative_drift_ms"],
        "max_abs_prefix_relative_drift_beats": payload["max_abs_prefix_relative_drift_beats"],
        "max_abs_prefix_relative_drift_ms": payload["max_abs_prefix_relative_drift_ms"],
        "drift_slope_beats_per_minute": payload["drift_slope_beats_per_minute"],
        "drift_slope_ms_per_minute": payload["drift_slope_ms_per_minute"],
        "p90_abs_30s_relative_drift_ms": payload["p90_abs_30s_relative_drift_ms"],
        "p90_abs_60s_relative_drift_ms": payload["p90_abs_60s_relative_drift_ms"],
        "active_section_disagreement_sample_count": payload["active_section_disagreement_sample_count"],
        "active_section_disagreement_fraction": payload["active_section_disagreement_fraction"],
        "active_section_signature_equal": payload["active_section_signature_equal"],
    }


def _audio_metric_payload(
    difficulty_metrics: Sequence[Mapping[str, Any]],
    *,
    canonicalization: str,
) -> dict[str, Any]:
    if not difficulty_metrics:
        return _empty_audio_metric_payload()
    metric_names = (
        "phase_mean_ms",
        "phase_p50_ms",
        "phase_p90_ms",
        "phase_max_ms",
        "phase_mean_beats",
        "phase_p90_beats",
        "local_bpm_mae",
        "local_bpm_alias_mae",
        "endpoint_relative_drift_ms",
        "max_abs_prefix_relative_drift_ms",
        "drift_slope_ms_per_minute",
        "p90_abs_30s_relative_drift_ms",
        "p90_abs_60s_relative_drift_ms",
        "active_section_disagreement_fraction",
    )
    values = {
        name: [
            float(metric[canonicalization][name])
            for metric in difficulty_metrics
            if isinstance(metric.get(canonicalization), MappingABC)
        ]
        for name in metric_names
    }
    return {
        "difficulty_count": len(difficulty_metrics),
        **{name: _median(items) for name, items in values.items()},
    }


def _empty_audio_metric_payload() -> dict[str, Any]:
    return {
        "difficulty_count": 0,
        "phase_mean_ms": None,
        "phase_p50_ms": None,
        "phase_p90_ms": None,
        "phase_max_ms": None,
        "phase_mean_beats": None,
        "phase_p90_beats": None,
        "local_bpm_mae": None,
        "local_bpm_alias_mae": None,
        "endpoint_relative_drift_ms": None,
        "max_abs_prefix_relative_drift_ms": None,
        "drift_slope_ms_per_minute": None,
        "p90_abs_30s_relative_drift_ms": None,
        "p90_abs_60s_relative_drift_ms": None,
        "active_section_disagreement_fraction": None,
    }


def _weak_boundary_payload(boundary: Any) -> dict[str, Any]:
    return {
        "valid_comparator": boundary.valid_comparator,
        "matched_count": boundary.weak_boundary_matched_count,
        "predicted_boundary_count": boundary.weak_boundary_predicted_count,
        "weak_redline_boundary_count": boundary.weak_boundary_redline_count,
        "unmatched_predicted_boundary_count": boundary.weak_boundary_unmatched_predicted_count,
        "unmatched_weak_redline_boundary_count": boundary.weak_boundary_unmatched_redline_count,
        "mean_signed_error_ms": boundary.weak_boundary_mean_signed_error_ms,
        "mean_abs_error_ms": boundary.weak_boundary_mean_abs_error_ms,
        "weak_boundary_match_rate": boundary.weak_boundary_match_rate,
        "predicted_boundary_match_rate": boundary.predicted_boundary_match_rate,
        "rejection_reason": boundary.rejection_reason,
    }


def _weak_boundary_audio_payload(audio: Any) -> dict[str, Any]:
    return {
        "valid_difficulty_count": audio.valid_difficulty_count,
        "invalid_difficulty_count": audio.invalid_difficulty_count,
        "matched_count": audio.weak_boundary_matched_count,
        "predicted_boundary_count": audio.weak_boundary_predicted_count,
        "weak_redline_boundary_count": audio.weak_boundary_redline_count,
        "unmatched_predicted_boundary_count": audio.weak_boundary_unmatched_predicted_count,
        "unmatched_weak_redline_boundary_count": audio.weak_boundary_unmatched_redline_count,
        "mean_abs_error_median_ms": audio.weak_boundary_mean_abs_error_median_ms,
        "match_rate_median": audio.weak_boundary_match_rate_median,
        "weak_consensus_supported_boundary_count": audio.weak_consensus_supported_boundary_count,
        "weak_consensus_boundaries": [
            {
                "time_ms": item.predicted_boundary.time_ms,
                "matched_valid_difficulty_count": item.matched_valid_difficulty_count,
                "required_valid_difficulty_count": item.required_valid_difficulty_count,
                "weak_consensus_supported": item.weak_consensus_supported,
            }
            for item in audio.weak_consensus_boundaries
        ],
    }


def _summary_payload(
    *,
    projection_source: _ProjectionSource,
    baseline_source: _BaselineSource,
    evaluator_source: Mapping[str, Any],
    output_path: Path,
    output_sha256: str,
    results: Sequence[Mapping[str, Any]],
    resume_counts: Mapping[str, int],
    total_seconds: float,
) -> dict[str, Any]:
    denominator_rows = [
        Exp004DenominatorRow(**dict(row["denominator_row"]))
        for row in results
    ]
    denominators = classify_exp004_denominators(denominator_rows)
    primary_pairs = _primary_pairs(results)
    mean_ratio = _ratio_payload(
        _mean([pair["cj3_mean"] for pair in primary_pairs]),
        _mean([pair["v2_mean"] for pair in primary_pairs]),
    )
    p90_ratio = _ratio_payload(
        _np_p90([pair["cj3_p90"] for pair in primary_pairs]),
        _np_p90([pair["v2_p90"] for pair in primary_pairs]),
    )
    guard = classify_exp004_primary_guards(
        denominators=denominators,
        mean_phase_ratio=mean_ratio["value"] if mean_ratio["defined"] else None,
        p90_phase_ratio=p90_ratio["value"] if p90_ratio["defined"] else None,
    )
    method_summary = _method_summary(results)
    stage_gate_payload = _stage_gate_payload(
        results,
        primary_pairs=primary_pairs,
        denominators=denominators.to_dict(),
        projection_source=projection_source,
    )
    decision_payload = _decision_payload(
        stage=str(projection_source.stage),
        stage_gates=stage_gate_payload,
        projection_source=projection_source,
    )
    hard_guards = {
        "ok": projection_source.hard_guards_ok,
        "violations": list(projection_source.hard_guard_violations),
        "source": "projection_summary",
    }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "experiment": "timing_v3_experiment_004",
        "stage": projection_source.stage,
        "source": {
            "projection_jsonl": {"path": projection_source.path.as_posix(), "sha256": projection_source.sha256},
            "projection_summary": {
                "path": projection_source.summary_path.as_posix(),
                "sha256": projection_source.summary_sha256,
                "schema": projection_source.summary.get("schema"),
                "stage": projection_source.stage,
                "behavior_fingerprint": projection_source.behavior_fingerprint,
                "config_fingerprint": projection_source.config_fingerprint,
                "run_fingerprint": projection_source.run_fingerprint,
                "output_path": _nested(projection_source.summary, "output", "path"),
                "output_sha256": _nested(projection_source.summary, "output", "sha256"),
                "output_row_count": _nested(projection_source.summary, "output", "row_count"),
                "hard_guards_ok": projection_source.hard_guards_ok,
                "formal_execution_ready": projection_source.formal_execution_ready,
                "formal_execution_blockers": list(projection_source.formal_execution_blockers),
                "stage_constraints": dict(projection_source.stage_constraints),
                "prior_baseline_jsonl_sha256": projection_source.prior_baseline_jsonl_sha256,
            },
            "baseline_jsonl": {"path": baseline_source.path.as_posix(), "sha256": baseline_source.sha256},
            "evaluator": dict(evaluator_source),
        },
        "output": {"path": output_path.resolve(strict=True).as_posix(), "sha256": output_sha256, "row_count": len(results)},
        "denominators": denominators.to_dict(),
        "denominator_policy": {
            "fallback_rate": {
                "denominator": "projection_evaluable_count",
                "reason": "fallback is projection safety and remains independent of comparator availability",
            },
            "no_path_plus_candidate_extraction_failure_rate": {
                "denominator": "cache_valid_count",
                "reason": "no-path and candidate failures are projection failures after cache validity, frozen by Exp004",
            },
            "phase_ratios": {
                "denominator": "pure_CJ3_phase_count",
                "reason": "fallback-selected rows and comparator-unavailable rows are excluded from pure CJ3 phase acceptance",
            },
        },
        "reasons": _summary_reasons(results),
        "primary": {
            "canonicalization": PRIMARY_CANONICALIZATION,
            "pure_CJ3_audio_pair_count": len(primary_pairs),
            "mean_phase_ratio": mean_ratio,
            "p90_phase_ratio": p90_ratio,
            "guards": guard.to_dict(),
        },
        "methods": method_summary,
        "difficulty_aggregates": _difficulty_aggregates(results),
        "audio_aggregates": _audio_aggregates(results),
        "strata": _strata_summary(results),
        "stage_gates": stage_gate_payload,
        "hard_guards": hard_guards,
        "decision": decision_payload["decision"],
        "next_action": decision_payload["next_action"],
        "decision_reason": decision_payload["reason"],
        "protocol_binding": {
            "schema": "pulsefield_model.timing_v3_exp004_weak_protocol_binding_v1",
            "source_projection": {
                "stage": projection_source.stage,
                "behavior_fingerprint": projection_source.behavior_fingerprint,
                "config_fingerprint": projection_source.config_fingerprint,
                "run_fingerprint": projection_source.run_fingerprint,
                "projection_jsonl_sha256": projection_source.sha256,
                "projection_summary_sha256": projection_source.summary_sha256,
                "projection_summary_hard_guards_ok": projection_source.hard_guards_ok,
                "projection_summary_formal_execution_ready": projection_source.formal_execution_ready,
                "projection_summary_formal_execution_blockers": list(projection_source.formal_execution_blockers),
                "projection_summary_stage_constraints": dict(projection_source.stage_constraints),
                "projection_summary_prior_baseline_jsonl_sha256": projection_source.prior_baseline_jsonl_sha256,
            },
            "baseline": {
                "path": baseline_source.path.as_posix(),
                "sha256": baseline_source.sha256,
                "row_count": len(baseline_source.rows_by_key),
                "schema": BASELINE_RESULT_SCHEMA,
            },
            "evaluator": dict(evaluator_source),
            "output": {"path": output_path.resolve(strict=True).as_posix(), "sha256": output_sha256, "row_count": len(results)},
            "denominators": denominators.to_dict(),
            "stage_gates": stage_gate_payload,
            "decision": decision_payload["decision"],
            "next_action": decision_payload["next_action"],
            "hard_guards": hard_guards,
        },
        "resume": dict(resume_counts),
        "runtime": {"total_seconds": float(total_seconds)},
    }
    summary = _json_safe(summary)
    _canonical_json(summary)
    return summary


def _primary_pairs(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for row in results:
        denom = row.get("denominator_row")
        if not isinstance(denom, MappingABC):
            continue
        if not (
            denom.get("comparison_eligible")
            and denom.get("pure_cj3_grid_produced")
            and denom.get("pure_cj3_phase_matched")
            and denom.get("current_v2_phase_matched")
        ):
            continue
        v2 = _nested(row, "methods", CURRENT_V2_METHOD, "audio_metrics", PRIMARY_CANONICALIZATION)
        cj3 = _nested(row, "methods", "CJ3", "audio_metrics", PRIMARY_CANONICALIZATION)
        if not isinstance(v2, MappingABC) or not isinstance(cj3, MappingABC):
            continue
        values = {
            "v2_mean": _optional_float(v2.get("phase_mean_ms")),
            "cj3_mean": _optional_float(cj3.get("phase_mean_ms")),
            "v2_p90": _optional_float(v2.get("phase_p90_ms")),
            "cj3_p90": _optional_float(cj3.get("phase_p90_ms")),
            "v2_long": _optional_float(v2.get("max_abs_prefix_relative_drift_ms")),
            "cj3_long": _optional_float(cj3.get("max_abs_prefix_relative_drift_ms")),
            "v2_endpoint_drift": _abs_optional_float(v2.get("endpoint_relative_drift_ms")),
            "cj3_endpoint_drift": _abs_optional_float(cj3.get("endpoint_relative_drift_ms")),
        }
        if all(value is not None for value in values.values()):
            pair = {key: float(value) for key, value in values.items() if value is not None}
            pair.update(
                {
                    "row_index": int(row.get("row_index", -1)),
                    "audio_key": str(_nested(row, "identity", "cache_audio_key") or ""),
                    "primary_stratum": str(_nested(row, "strata", "primary_stratum") or "unknown"),
                    "stable": bool(_nested(row, "strata", "stable")),
                    "jump": bool(_nested(row, "strata", "jump")),
                    "long": bool(_nested(row, "strata", "long")),
                    "dense": bool(_nested(row, "strata", "dense")),
                    "ramp": bool(_nested(row, "strata", "ramp")),
                    "anomaly": bool(_nested(row, "strata", "anomaly")),
                }
            )
            pairs.append(pair)
    return pairs


def _method_pairs(results: Sequence[Mapping[str, Any]], method: str) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    if method not in VARIANT_METHODS:
        raise ValueError(f"unsupported method pair target: {method!r}")
    for row in results:
        denom = row.get("denominator_row")
        if not isinstance(denom, MappingABC) or not denom.get("comparison_eligible"):
            continue
        cj3 = _nested(row, "methods", "CJ3", "audio_metrics", PRIMARY_CANONICALIZATION)
        other = _nested(row, "methods", method, "audio_metrics", PRIMARY_CANONICALIZATION)
        if not isinstance(cj3, MappingABC) or not isinstance(other, MappingABC):
            continue
        if not (
            _nested(row, "methods", "CJ3", "comparison_matched")
            and _nested(row, "methods", method, "comparison_matched")
        ):
            continue
        values = {
            "other_mean": _optional_float(other.get("phase_mean_ms")),
            "cj3_mean": _optional_float(cj3.get("phase_mean_ms")),
            "other_endpoint_drift": _abs_optional_float(other.get("endpoint_relative_drift_ms")),
            "cj3_endpoint_drift": _abs_optional_float(cj3.get("endpoint_relative_drift_ms")),
        }
        if all(value is not None for value in values.values()):
            pair = {key: float(value) for key, value in values.items() if value is not None}
            pair.update(
                {
                    "row_index": int(row.get("row_index", -1)),
                    "primary_stratum": str(_nested(row, "strata", "primary_stratum") or "unknown"),
                    "jump": bool(_nested(row, "strata", "jump")),
                    "long": bool(_nested(row, "strata", "long")),
                }
            )
            pairs.append(pair)
    return pairs


def _ratio_payload(numerator: float | None, denominator: float | None) -> dict[str, Any]:
    if numerator is None or denominator is None:
        return {"defined": False, "value": UNDEFINED_RATIO_VALUE, "reason": "empty_denominator"}
    numerator = float(numerator)
    denominator = float(denominator)
    if abs(numerator) <= ZERO_RATIO_EPSILON and abs(denominator) <= ZERO_RATIO_EPSILON:
        return {"defined": True, "value": 1.0, "reason": "both_zero_convention"}
    if abs(denominator) <= ZERO_RATIO_EPSILON:
        return {"defined": False, "value": UNDEFINED_RATIO_VALUE, "reason": "zero_denominator"}
    return {"defined": True, "value": float(numerator / denominator), "reason": None}


def _stage_gate_payload(
    results: Sequence[Mapping[str, Any]],
    *,
    primary_pairs: Sequence[Mapping[str, Any]],
    denominators: Mapping[str, Any],
    projection_source: _ProjectionSource,
) -> dict[str, Any]:
    cache_valid_count = int(denominators.get("cache_valid_count") or 0)
    comparison_eligible_count = int(denominators.get("comparison_eligible_count") or 0)
    projection_evaluable_count = int(denominators.get("projection_evaluable_count") or 0)
    pure_phase_count = int(denominators.get("pure_CJ3_phase_count") or 0)
    selected_fallback_count = int(denominators.get("selected_fallback_count") or 0)
    no_path_count = sum(
        _row_cache_valid(row) and _is_no_path_or_candidate_failure(row)
        for row in results
    )
    runtime_values = [
        _optional_float(_nested(row, "projection", "row", "runtime", "total_seconds"))
        for row in results
    ]
    runtime_values = [float(value) for value in runtime_values if value is not None]
    all_mean_numerator = _mean([pair["cj3_mean"] for pair in primary_pairs])
    all_mean_denominator = _mean([pair["v2_mean"] for pair in primary_pairs])
    all_p90_numerator = _np_p90([pair["cj3_p90"] for pair in primary_pairs])
    all_p90_denominator = _np_p90([pair["v2_p90"] for pair in primary_pairs])
    stable_pairs = [pair for pair in primary_pairs if pair.get("stable")]
    jump_pairs = [pair for pair in primary_pairs if pair.get("jump")]
    long_pairs = [pair for pair in primary_pairs if pair.get("long")]
    quota_degraded = _projection_quota_degraded(projection_source.summary)
    broad_underfilled = _projection_broad_underfilled(projection_source.summary)
    jump_cj1_pairs = [pair for pair in _method_pairs(results, "CJ1") if pair.get("jump")]
    jump_cj2_pairs = [pair for pair in _method_pairs(results, "CJ2") if pair.get("jump")]
    gates = {
        "projection_hard_guards": _boolean_gate(
            value=projection_source.hard_guards_ok,
            threshold="projection summary hard_guards.ok must be true",
            kill_reason="projection_summary_hard_guard_failure",
        ),
        "phase_denominator_available": _phase_denominator_gate(
            stage=projection_source.stage,
            pure_phase_count=pure_phase_count,
            comparison_eligible_count=comparison_eligible_count,
        ),
        "all_mean_phase_ratio": _ratio_max_gate(
            numerator=all_mean_numerator,
            denominator=all_mean_denominator,
            sample_count=len(primary_pairs),
            pass_max=1.05,
            ambiguous_max=1.10,
            threshold="pass <= 1.05; ambiguous <= 1.10; kill > 1.10",
        ),
        "all_p90_phase_ratio": _ratio_max_gate(
            numerator=all_p90_numerator,
            denominator=all_p90_denominator,
            sample_count=len(primary_pairs),
            pass_max=1.10,
            ambiguous_max=1.15,
            threshold="pass <= 1.10; ambiguous <= 1.15; kill > 1.15",
        ),
        "pure_CJ3_phase_coverage": _rate_min_gate(
            numerator=pure_phase_count,
            denominator=comparison_eligible_count,
            pass_min=0.95,
            ambiguous_min=0.90,
            threshold="pass >= 0.95; ambiguous >= 0.90; kill < 0.90",
        ),
        "stable_mean_phase_ratio": _min_count_ratio_gate(
            pairs=stable_pairs,
            numerator_key="cj3_mean",
            denominator_key="v2_mean",
            minimum_count=5,
            pass_max=1.10,
            ambiguous_max=1.20,
            threshold="if n >= 5: pass <= 1.10; ambiguous <= 1.20; kill > 1.20",
        ),
        "stable_p90_phase_ratio": _min_count_ratio_gate(
            pairs=stable_pairs,
            numerator_key="cj3_p90",
            denominator_key="v2_p90",
            minimum_count=5,
            pass_max=1.10,
            ambiguous_max=1.20,
            threshold="if n >= 5: pass <= 1.10; ambiguous <= 1.20; kill > 1.20",
        ),
        "jump_mean_phase_ratio": _diagnostic_gate(
            _min_count_ratio_gate(
                pairs=jump_pairs,
                numerator_key="cj3_mean",
                denominator_key="v2_mean",
                minimum_count=15,
                pass_max=1.00,
                ambiguous_max=1.10,
                threshold="if n >= 15: pass <= 1.00; kill participates in combined jump rule",
            )
        ),
        "jump_endpoint_drift_mean_ratio": _diagnostic_gate(
            _min_count_ratio_gate(
                pairs=jump_pairs,
                numerator_key="cj3_endpoint_drift",
                denominator_key="v2_endpoint_drift",
                minimum_count=15,
                pass_max=0.90,
                ambiguous_max=1.10,
                threshold="if n >= 15: pass <= 0.90; kill participates in combined jump rule",
            )
        ),
        "jump_combined_mean_or_drift": _jump_combined_gate(jump_pairs),
        "jump_ablation_CJ3_vs_CJ1": _jump_ablation_gate(jump_cj1_pairs, method="CJ1"),
        "jump_ablation_CJ3_vs_CJ2": _jump_ablation_gate(jump_cj2_pairs, method="CJ2"),
        "long_max_prefix_drift_mean_ratio": _min_count_ratio_gate(
            pairs=long_pairs,
            numerator_key="cj3_long",
            denominator_key="v2_long",
            minimum_count=5,
            pass_max=1.15,
            ambiguous_max=1.30,
            threshold="if n >= 5: pass <= 1.15; ambiguous <= 1.30; kill > 1.30",
        ),
        "long_max_prefix_drift_p90_ratio": _min_count_p90_ratio_gate(
            pairs=long_pairs,
            numerator_key="cj3_long",
            denominator_key="v2_long",
            minimum_count=5,
            pass_max=1.15,
            ambiguous_max=1.30,
            threshold="if n >= 5: pass <= 1.15; ambiguous <= 1.30; kill > 1.30",
        ),
        "fallback_rate": _rate_max_gate(
            numerator=selected_fallback_count,
            denominator=projection_evaluable_count,
            pass_max=0.05,
            ambiguous_max=0.10,
            threshold="pass <= 0.05; ambiguous <= 0.10; kill > 0.10",
        ),
        "no_path_plus_candidate_extraction_failure_rate": _rate_max_gate(
            numerator=int(no_path_count),
            denominator=cache_valid_count,
            pass_max=0.03,
            ambiguous_max=0.05,
            threshold="pass <= 0.03; ambiguous <= 0.05; kill > 0.05",
        ),
        "runtime_p90_seconds": _value_max_gate(
            value=_np_p90(runtime_values),
            numerator=_np_p90(runtime_values),
            denominator=len(runtime_values),
            pass_max=30.0,
            ambiguous_max=60.0,
            threshold="pass <= 30 s; ambiguous <= 60 s; kill > 60 s",
        ),
        "quota_degraded_minimum_denominator": _quota_degraded_gate(
            quota_degraded=quota_degraded,
            broad_underfilled=broad_underfilled,
            jump_count=len(jump_pairs),
            long_count=len(long_pairs),
        ),
    }
    return {
        "schema": "pulsefield_model.timing_v3_exp004_weak_stage_gates_v1",
        "stage": projection_source.stage,
        "sample_counts": {
            "all": len(primary_pairs),
            "stable": len(stable_pairs),
            "jump": len(jump_pairs),
            "long": len(long_pairs),
            "jump_CJ1_ablation": len(jump_cj1_pairs),
            "jump_CJ2_ablation": len(jump_cj2_pairs),
        },
        "denominator_notes": {
            "fallback_rate": "denominator is projection_evaluable_count",
            "no_path_plus_candidate_extraction_failure_rate": "denominator is cache_valid_count",
            "phase_ratios": "fallback-selected rows and comparator-unavailable rows are excluded",
        },
        "gates": gates,
    }


def _is_no_path_or_candidate_failure(row: Mapping[str, Any]) -> bool:
    projection = _nested(row, "projection", "row")
    if not isinstance(projection, MappingABC):
        return False
    candidate_status = _nested(projection, "candidate_extraction", "status")
    candidate_reason = _nested(projection, "candidate_extraction", "reason")
    if candidate_status != "accepted" and candidate_reason == "candidate_extraction_failure":
        return True
    cj3_reason = _nested(projection, "variants", "CJ3", "reason")
    return cj3_reason in {"no_global_constant_jump_path", "no_origin_candidate"}


def _row_cache_valid(row: Mapping[str, Any]) -> bool:
    return bool(_nested(row, "denominator_row", "cache_valid"))


def _gate_payload(
    *,
    value: Any,
    numerator: Any,
    denominator: Any,
    status: str,
    threshold: str,
    reason: str | None,
    decision_gate: bool = True,
) -> dict[str, Any]:
    return {
        "value": value,
        "numerator": numerator,
        "denominator": denominator,
        "status": status,
        "threshold": threshold,
        "reason": reason,
        "decision_gate": bool(decision_gate),
    }


def _boolean_gate(*, value: bool, threshold: str, kill_reason: str) -> dict[str, Any]:
    return _gate_payload(
        value=bool(value),
        numerator=1 if value else 0,
        denominator=1,
        status=STAGE_STATUS_PASS if value else STAGE_STATUS_KILL,
        threshold=threshold,
        reason=None if value else kill_reason,
    )


def _diagnostic_gate(gate: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(gate)
    result["decision_gate"] = False
    return result


def _ratio_max_gate(
    *,
    numerator: float | None,
    denominator: float | None,
    sample_count: int,
    pass_max: float,
    ambiguous_max: float,
    threshold: str,
) -> dict[str, Any]:
    ratio = _ratio_payload(numerator, denominator)
    if not ratio["defined"]:
        return _gate_payload(
            value=ratio["value"],
            numerator=numerator,
            denominator=denominator,
            status=STAGE_STATUS_NOT_APPLICABLE if sample_count == 0 else STAGE_STATUS_AMBIGUOUS,
            threshold=threshold,
            reason=str(ratio["reason"]),
        )
    return _gate_payload(
        value=ratio["value"],
        numerator=numerator,
        denominator=denominator,
        status=_max_status(float(ratio["value"]), pass_max=pass_max, ambiguous_max=ambiguous_max),
        threshold=threshold,
        reason=ratio["reason"],
    )


def _rate_min_gate(
    *,
    numerator: int,
    denominator: int,
    pass_min: float,
    ambiguous_min: float,
    threshold: str,
) -> dict[str, Any]:
    if denominator <= 0:
        return _gate_payload(
            value=None,
            numerator=int(numerator),
            denominator=int(denominator),
            status=STAGE_STATUS_NOT_APPLICABLE,
            threshold=threshold,
            reason="empty_denominator",
        )
    value = float(numerator / denominator)
    if value >= pass_min:
        status = STAGE_STATUS_PASS
    elif value >= ambiguous_min:
        status = STAGE_STATUS_AMBIGUOUS
    else:
        status = STAGE_STATUS_KILL
    return _gate_payload(
        value=value,
        numerator=int(numerator),
        denominator=int(denominator),
        status=status,
        threshold=threshold,
        reason=None,
    )


def _rate_max_gate(
    *,
    numerator: int,
    denominator: int,
    pass_max: float,
    ambiguous_max: float,
    threshold: str,
) -> dict[str, Any]:
    if denominator <= 0:
        return _gate_payload(
            value=None,
            numerator=int(numerator),
            denominator=int(denominator),
            status=STAGE_STATUS_NOT_APPLICABLE,
            threshold=threshold,
            reason="empty_denominator",
        )
    value = float(numerator / denominator)
    return _gate_payload(
        value=value,
        numerator=int(numerator),
        denominator=int(denominator),
        status=_max_status(value, pass_max=pass_max, ambiguous_max=ambiguous_max),
        threshold=threshold,
        reason=None,
    )


def _value_max_gate(
    *,
    value: float | None,
    numerator: float | None,
    denominator: int,
    pass_max: float,
    ambiguous_max: float,
    threshold: str,
) -> dict[str, Any]:
    if value is None:
        return _gate_payload(
            value=None,
            numerator=numerator,
            denominator=int(denominator),
            status=STAGE_STATUS_NOT_APPLICABLE,
            threshold=threshold,
            reason="empty_denominator",
        )
    return _gate_payload(
        value=float(value),
        numerator=numerator,
        denominator=int(denominator),
        status=_max_status(float(value), pass_max=pass_max, ambiguous_max=ambiguous_max),
        threshold=threshold,
        reason=None,
    )


def _min_count_ratio_gate(
    *,
    pairs: Sequence[Mapping[str, Any]],
    numerator_key: str,
    denominator_key: str,
    minimum_count: int,
    pass_max: float,
    ambiguous_max: float,
    threshold: str,
) -> dict[str, Any]:
    numerator = _mean([pair[numerator_key] for pair in pairs])
    denominator = _mean([pair[denominator_key] for pair in pairs])
    return _minimum_count_gate(
        value_gate=_ratio_max_gate(
            numerator=numerator,
            denominator=denominator,
            sample_count=len(pairs),
            pass_max=pass_max,
            ambiguous_max=ambiguous_max,
            threshold=threshold,
        ),
        sample_count=len(pairs),
        minimum_count=minimum_count,
    )


def _min_count_p90_ratio_gate(
    *,
    pairs: Sequence[Mapping[str, Any]],
    numerator_key: str,
    denominator_key: str,
    minimum_count: int,
    pass_max: float,
    ambiguous_max: float,
    threshold: str,
) -> dict[str, Any]:
    numerator = _np_p90([pair[numerator_key] for pair in pairs])
    denominator = _np_p90([pair[denominator_key] for pair in pairs])
    return _minimum_count_gate(
        value_gate=_ratio_max_gate(
            numerator=numerator,
            denominator=denominator,
            sample_count=len(pairs),
            pass_max=pass_max,
            ambiguous_max=ambiguous_max,
            threshold=threshold,
        ),
        sample_count=len(pairs),
        minimum_count=minimum_count,
    )


def _minimum_count_gate(
    *,
    value_gate: Mapping[str, Any],
    sample_count: int,
    minimum_count: int,
) -> dict[str, Any]:
    gate = dict(value_gate)
    gate["sample_count"] = int(sample_count)
    gate["minimum_count"] = int(minimum_count)
    if sample_count >= minimum_count:
        return gate
    gate["status"] = STAGE_STATUS_NOT_APPLICABLE
    gate["reason"] = "empty_stratum" if sample_count == 0 else "below_minimum_non_decision_denominator"
    return gate


def _phase_denominator_gate(
    *,
    stage: str,
    pure_phase_count: int,
    comparison_eligible_count: int,
) -> dict[str, Any]:
    formal_decision_stage = stage in FORMAL_DECISION_STAGES
    missing = comparison_eligible_count <= 0 or pure_phase_count <= 0
    if not formal_decision_stage:
        return _gate_payload(
            value=not missing,
            numerator=pure_phase_count,
            denominator=comparison_eligible_count,
            status=STAGE_STATUS_NOT_APPLICABLE,
            threshold="formal holdout/broad/full require non-empty comparison and pure CJ3 phase denominators",
            reason="non_formal_stage",
            decision_gate=False,
        )
    return _gate_payload(
        value=not missing,
        numerator=pure_phase_count,
        denominator=comparison_eligible_count,
        status=STAGE_STATUS_AMBIGUOUS if missing else STAGE_STATUS_PASS,
        threshold="formal holdout/broad/full require non-empty comparison and pure CJ3 phase denominators",
        reason="formal_empty_primary_or_comparator_denominator" if missing else None,
    )


def _jump_combined_gate(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    mean_ratio = _ratio_payload(
        _mean([pair["cj3_mean"] for pair in pairs]),
        _mean([pair["v2_mean"] for pair in pairs]),
    )
    drift_ratio = _ratio_payload(
        _mean([pair["cj3_endpoint_drift"] for pair in pairs]),
        _mean([pair["v2_endpoint_drift"] for pair in pairs]),
    )
    value = {
        "mean_phase_ratio": mean_ratio["value"],
        "endpoint_drift_mean_ratio": drift_ratio["value"],
    }
    if len(pairs) == 0:
        status = STAGE_STATUS_NOT_APPLICABLE
        reason = "empty_stratum"
    elif len(pairs) < 15:
        status = STAGE_STATUS_NOT_APPLICABLE
        reason = "below_minimum_non_decision_denominator"
    elif not mean_ratio["defined"] or not drift_ratio["defined"]:
        status = STAGE_STATUS_AMBIGUOUS
        reason = "undefined_ratio"
    elif float(mean_ratio["value"]) <= 1.00 or float(drift_ratio["value"]) <= 0.90:
        status = STAGE_STATUS_PASS
        reason = None
    elif float(mean_ratio["value"]) > 1.10 and float(drift_ratio["value"]) > 1.10:
        status = STAGE_STATUS_KILL
        reason = "jump_mean_and_drift_kill_band"
    else:
        status = STAGE_STATUS_AMBIGUOUS
        reason = "jump_mean_or_drift_ambiguous_band"
    return _gate_payload(
        value=value,
        numerator={
            "cj3_mean_phase_ms": _mean([pair["cj3_mean"] for pair in pairs]),
            "cj3_endpoint_drift_ms": _mean([pair["cj3_endpoint_drift"] for pair in pairs]),
        },
        denominator={
            "current_v2_mean_phase_ms": _mean([pair["v2_mean"] for pair in pairs]),
            "current_v2_endpoint_drift_ms": _mean([pair["v2_endpoint_drift"] for pair in pairs]),
            "sample_count": len(pairs),
            "minimum_count": 15,
        },
        status=status,
        threshold="if n >= 15: pass when mean <= 1.00 or drift <= 0.90; kill when both > 1.10",
        reason=reason,
    )


def _jump_ablation_gate(pairs: Sequence[Mapping[str, Any]], *, method: str) -> dict[str, Any]:
    phase_ratio = _ratio_payload(
        _mean([pair["cj3_mean"] for pair in pairs]),
        _mean([pair["other_mean"] for pair in pairs]),
    )
    drift_ratio = _ratio_payload(
        _mean([pair["cj3_endpoint_drift"] for pair in pairs]),
        _mean([pair["other_endpoint_drift"] for pair in pairs]),
    )
    value = {
        "mean_phase_ratio": phase_ratio["value"],
        "endpoint_drift_mean_ratio": drift_ratio["value"],
    }
    if len(pairs) == 0:
        status = STAGE_STATUS_NOT_APPLICABLE
        reason = "empty_stratum"
    elif len(pairs) < 15:
        status = STAGE_STATUS_NOT_APPLICABLE
        reason = "below_minimum_non_decision_denominator"
    elif not phase_ratio["defined"] or not drift_ratio["defined"]:
        status = STAGE_STATUS_AMBIGUOUS
        reason = "undefined_ratio"
    elif (
        method == "CJ2"
        and abs(float(phase_ratio["value"]) - 1.0) < 0.01
        and abs(float(drift_ratio["value"]) - 1.0) < 0.01
    ):
        status = STAGE_STATUS_AMBIGUOUS
        reason = "CJ3_CJ2_headline_delta_under_1_percent"
    elif float(phase_ratio["value"]) < 1.0 or float(drift_ratio["value"]) < 1.0:
        status = STAGE_STATUS_PASS
        reason = None
    else:
        status = STAGE_STATUS_KILL
        reason = f"CJ3_does_not_beat_{method}_on_jump_phase_or_drift"
    return _gate_payload(
        value=value,
        numerator={
            "cj3_mean_phase_ms": _mean([pair["cj3_mean"] for pair in pairs]),
            "cj3_endpoint_drift_ms": _mean([pair["cj3_endpoint_drift"] for pair in pairs]),
        },
        denominator={
            f"{method}_mean_phase_ms": _mean([pair["other_mean"] for pair in pairs]),
            f"{method}_endpoint_drift_ms": _mean([pair["other_endpoint_drift"] for pair in pairs]),
            "sample_count": len(pairs),
            "minimum_count": 15,
        },
        status=status,
        threshold=f"if jump n >= 15: CJ3 must beat {method} on mean phase or endpoint drift",
        reason=reason,
    )


def _quota_degraded_gate(
    *,
    quota_degraded: bool,
    broad_underfilled: bool,
    jump_count: int,
    long_count: int,
) -> dict[str, Any]:
    insufficient = []
    if jump_count < 15:
        insufficient.append("jump")
    if long_count < 5:
        insufficient.append("long")
    degraded = quota_degraded or broad_underfilled
    status = STAGE_STATUS_AMBIGUOUS if degraded and insufficient else STAGE_STATUS_PASS
    return _gate_payload(
        value={"quota_degraded": bool(quota_degraded), "broad_underfilled": bool(broad_underfilled)},
        numerator={"jump_count": int(jump_count), "long_count": int(long_count)},
        denominator={"jump_minimum": 15, "long_minimum": 5},
        status=status,
        threshold="quota degradation or broad underfill plus jump < 15 or long < 5 is ambiguous",
        reason="quota_or_broad_underfilled_insufficient_denominator" if status == STAGE_STATUS_AMBIGUOUS else None,
    )


def _max_status(value: float, *, pass_max: float, ambiguous_max: float) -> str:
    if value <= pass_max:
        return STAGE_STATUS_PASS
    if value <= ambiguous_max:
        return STAGE_STATUS_AMBIGUOUS
    return STAGE_STATUS_KILL


def _projection_quota_degraded(summary: Mapping[str, Any]) -> bool:
    for path in (
        ("source", "selection_manifest", "stage_constraints", "quota_degraded"),
        ("source", "quota_degraded"),
        ("source", "selection_manifest", "quota_degraded"),
        ("selection", "quota_degraded"),
        ("denominators", "quota_degraded"),
    ):
        value = _nested(summary, *path)
        if isinstance(value, bool) and value:
            return True
    return False


def _projection_broad_underfilled(summary: Mapping[str, Any]) -> bool:
    for path in (
        ("source", "selection_manifest", "stage_constraints", "broad_underfilled"),
        ("source", "selection_manifest", "stage_constraints", "underfilled"),
    ):
        value = _nested(summary, *path)
        if isinstance(value, bool) and value:
            return True
    return False


def _decision_payload(
    *,
    stage: str,
    stage_gates: Mapping[str, Any],
    projection_source: _ProjectionSource,
) -> dict[str, str]:
    if not projection_source.hard_guards_ok:
        return {
            "decision": STAGE_STATUS_KILL,
            "next_action": "stop_projection_hard_guard_failure",
            "reason": "projection_summary_hard_guard_failure",
        }
    if stage == "repair80":
        return {
            "decision": "debug_only",
            "next_action": "proceed_to_holdout100",
            "reason": "repair80_is_regression_debug_only",
        }
    gates = _stage_gate_statuses(stage_gates)
    if any(status == STAGE_STATUS_KILL for status in gates):
        decision = STAGE_STATUS_KILL
    elif any(status == STAGE_STATUS_AMBIGUOUS for status in gates):
        decision = STAGE_STATUS_AMBIGUOUS
    else:
        decision = STAGE_STATUS_PASS
    actions = {
        ("holdout100", STAGE_STATUS_PASS): "freeze_source_config_and_materialize_broad500",
        ("holdout100", STAGE_STATUS_AMBIGUOUS): "stop_audit_and_create_new_card",
        ("holdout100", STAGE_STATUS_KILL): "kill_candidate_family",
        ("broad500", STAGE_STATUS_PASS): "freeze_source_config_and_run_full5050",
        ("broad500", STAGE_STATUS_AMBIGUOUS): "stop_no_production_switch_create_new_card",
        ("broad500", STAGE_STATUS_KILL): "kill_candidate_family",
        ("full5050", STAGE_STATUS_PASS): "accept_for_later_production_integration",
        ("full5050", STAGE_STATUS_AMBIGUOUS): "research_only_create_new_card",
        ("full5050", STAGE_STATUS_KILL): "do_not_switch_production_fitter",
        (SYNTHETIC_FIXTURE_STAGE, STAGE_STATUS_PASS): "synthetic_fixture_only",
        (SYNTHETIC_FIXTURE_STAGE, STAGE_STATUS_AMBIGUOUS): "synthetic_fixture_only",
        (SYNTHETIC_FIXTURE_STAGE, STAGE_STATUS_KILL): "synthetic_fixture_only",
    }
    return {
        "decision": decision,
        "next_action": actions.get((stage, decision), "unsupported_stage_no_action"),
        "reason": f"{stage}_{decision}",
    }


def _stage_gate_statuses(stage_gates: Mapping[str, Any]) -> list[str]:
    gates = stage_gates.get("gates")
    if not isinstance(gates, MappingABC):
        return []
    statuses = []
    for gate in gates.values():
        if isinstance(gate, MappingABC):
            if gate.get("decision_gate") is False:
                continue
            status = gate.get("status")
            if isinstance(status, str) and status != STAGE_STATUS_NOT_APPLICABLE:
                statuses.append(status)
    return statuses


def _method_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for method in METHODS:
        summary[method] = {
            "audio_count": sum(bool(_nested(row, "methods", method, "comparison_matched")) for row in results),
            "raw": _method_metric_stats(results, method, TIMING_CANONICALIZATION_NONE),
            PRIMARY_CANONICALIZATION: _method_metric_stats(results, method, PRIMARY_CANONICALIZATION),
            "weak_boundary_consensus_count": sum(
                int(_nested(row, "methods", method, "weak_boundary_audio", "weak_consensus_supported_boundary_count") or 0)
                for row in results
            ),
        }
    return summary


def _method_metric_stats(
    results: Sequence[Mapping[str, Any]],
    method: str,
    canonicalization: str,
) -> dict[str, Any]:
    metric_names = (
        "phase_mean_ms",
        "phase_p90_ms",
        "local_bpm_mae",
        "local_bpm_alias_mae",
        "endpoint_relative_drift_ms",
        "max_abs_prefix_relative_drift_ms",
        "drift_slope_ms_per_minute",
        "p90_abs_30s_relative_drift_ms",
        "p90_abs_60s_relative_drift_ms",
        "active_section_disagreement_fraction",
    )
    return {
        name: _stats(
            _optional_float(_nested(row, "methods", method, "audio_metrics", canonicalization, name))
            for row in results
        )
        for name in metric_names
    }


def _difficulty_aggregates(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = []
    for row in results:
        metrics = _nested(row, "methods", "CJ3", "difficulty_metrics")
        if isinstance(metrics, SequenceABC) and not isinstance(metrics, (str, bytes)):
            values.extend(
                _optional_float(_nested(metric, PRIMARY_CANONICALIZATION, "phase_mean_ms"))
                for metric in metrics
                if isinstance(metric, MappingABC)
            )
    return {"CJ3_alias_phase_mean_ms": _stats(value for value in values if value is not None)}


def _audio_aggregates(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "CJ3_alias_phase_mean_ms": _stats(
            _optional_float(_nested(row, "methods", "CJ3", "audio_metrics", PRIMARY_CANONICALIZATION, "phase_mean_ms"))
            for row in results
        )
    }


def _strata_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pairs = _primary_pairs(results)
    by_primary: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_primary[str(pair.get("primary_stratum") or "unknown")].append(pair)
    payload: dict[str, Any] = {
        "by_primary_stratum": {
            key: _paired_summary_for_pairs(values)
            for key, values in sorted(by_primary.items())
        }
    }
    for label in ("stable", "jump", "long", "dense", "ramp", "anomaly"):
        payload[label] = _paired_summary_for_pairs([pair for pair in pairs if pair.get(label)])
    return payload


def _paired_summary_for_pairs(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cj3_mean = _mean([pair["cj3_mean"] for pair in pairs])
    v2_mean = _mean([pair["v2_mean"] for pair in pairs])
    cj3_p90 = _np_p90([pair["cj3_p90"] for pair in pairs])
    v2_p90 = _np_p90([pair["v2_p90"] for pair in pairs])
    cj3_long = _mean([pair["cj3_long"] for pair in pairs])
    v2_long = _mean([pair["v2_long"] for pair in pairs])
    return {
        "paired_audio_count": len(pairs),
        "mean_phase_ratio": _ratio_payload(cj3_mean, v2_mean),
        "p90_phase_ratio": _ratio_payload(cj3_p90, v2_p90),
        "max_prefix_drift_mean_ratio": _ratio_payload(cj3_long, v2_long),
        "cj3_phase_mean_ms": _stats(pair["cj3_mean"] for pair in pairs),
        "current_v2_phase_mean_ms": _stats(pair["v2_mean"] for pair in pairs),
    }


def _summary_reasons(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counter = Counter(
        reason
        for row in results
        for reason in (row.get("reasons", []) if isinstance(row.get("reasons"), SequenceABC) else [])
    )
    return dict(sorted(counter.items()))


def _load_projection_inputs(path: Path, summary_path: Path) -> tuple[_ProjectionSource, list[_ProjectionRow]]:
    summary, summary_sha256 = _load_json_object_with_sha(summary_path)
    if summary.get("schema") != exp004_runner.SUMMARY_SCHEMA:
        raise ValueError("projection summary has unsupported schema")
    if summary.get("experiment") != "timing_v3_experiment_004":
        raise ValueError("projection summary experiment identifier is invalid")
    stage = _required_string(summary.get("stage"), "projection summary stage")
    expected_sha = _nested(summary, "output", "sha256")
    _require_sha256(expected_sha, "projection summary output.sha256")
    rows_payload, actual_sha = _load_jsonl_objects_with_sha(path)
    if expected_sha != actual_sha:
        raise ValueError("projection JSONL SHA does not match projection summary")
    output_path = _required_string(_nested(summary, "output", "path"), "projection summary output.path")
    if Path(output_path).expanduser().resolve(strict=False) != path.expanduser().resolve(strict=False):
        raise ValueError("projection summary output.path does not match projection JSONL path")
    expected_count = _nested(summary, "output", "row_count")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise ValueError("projection summary output.row_count must be an integer")
    if expected_count != len(rows_payload):
        raise ValueError("projection row count does not match summary")
    if stage == SYNTHETIC_FIXTURE_STAGE:
        pass
    elif stage in FORMAL_STAGE_AUDIO_COUNTS:
        expected_stage_count = FORMAL_STAGE_AUDIO_COUNTS[stage]
        if len(rows_payload) != expected_stage_count:
            raise ValueError(f"{stage} projection JSONL must contain exactly {expected_stage_count} rows")
        _require_projection_summary_count(summary, ("source", "stage_audio_count"), expected_stage_count)
        _require_projection_summary_count(summary, ("results", "result_count"), expected_stage_count)
        _require_projection_summary_count(summary, ("denominators", "stage_audio_count"), expected_stage_count)
    else:
        raise ValueError(f"unsupported projection summary stage: {stage!r}")
    provenance = summary.get("provenance")
    if not isinstance(provenance, MappingABC):
        raise ValueError("projection summary provenance is missing")
    behavior_fingerprint = _required_sha_from_mapping(provenance, "behavior_fingerprint")
    config_fingerprint = _required_sha_from_mapping(provenance, "config_fingerprint")
    run_fingerprint = _required_sha_from_mapping(provenance, "run_fingerprint")
    hard_guards = summary.get("hard_guards")
    if not isinstance(hard_guards, MappingABC) or not isinstance(hard_guards.get("ok"), bool):
        raise ValueError("projection summary hard_guards.ok is missing")
    hard_guard_violations = hard_guards.get("violations", [])
    if isinstance(hard_guard_violations, (str, bytes)) or not isinstance(hard_guard_violations, SequenceABC):
        raise ValueError("projection summary hard_guards.violations must be a sequence")
    formal_ready = _nested(summary, "integration", "formal_execution_ready")
    if not isinstance(formal_ready, bool):
        raise ValueError("projection summary integration.formal_execution_ready is missing")
    blockers = _nested(summary, "integration", "blockers")
    if blockers is None:
        blockers = []
    if isinstance(blockers, (str, bytes)) or not isinstance(blockers, SequenceABC):
        raise ValueError("projection summary integration.blockers must be a sequence")
    formal_blockers = tuple(str(item) for item in blockers)
    if stage in FORMAL_DECISION_STAGES:
        if formal_ready is not True:
            raise ValueError(f"{stage} projection summary must be formal_execution_ready")
        if formal_blockers:
            raise ValueError(f"{stage} projection summary integration blockers must be empty")
    stage_constraints = _nested(summary, "source", "selection_manifest", "stage_constraints")
    if stage_constraints is None:
        stage_constraints = {}
    if not isinstance(stage_constraints, MappingABC):
        raise ValueError("projection summary source.selection_manifest.stage_constraints must be a mapping")
    prior_baseline = _nested(summary, "source", "prior_stage", "baseline_jsonl_sha256")
    if prior_baseline is not None:
        _require_sha256(prior_baseline, "projection summary prior baseline_jsonl_sha256")
        prior_baseline = str(prior_baseline)
    elif stage in FORMAL_DECISION_STAGES:
        raise ValueError(f"{stage} projection summary prior baseline_jsonl_sha256 is required")

    rows: list[_ProjectionRow] = []
    audio_keys: set[str] = set()
    for index, payload in enumerate(rows_payload):
        if payload.get("schema") != exp004_runner.RESULT_SCHEMA:
            raise ValueError(f"projection row {index} has unsupported schema")
        result_fingerprint = payload.get("result_fingerprint")
        _require_sha256(result_fingerprint, f"projection row {index} result_fingerprint")
        fingerprint_body = dict(payload)
        fingerprint_body.pop("result_fingerprint", None)
        if str(result_fingerprint) != _stable_json_sha256(fingerprint_body):
            raise ValueError(f"projection row {index} result_fingerprint mismatch")
        if payload.get("row_complete") is not True:
            raise ValueError(f"projection row {index} is incomplete")
        if payload.get("experiment") != "timing_v3_experiment_004":
            raise ValueError(f"projection row {index} experiment identifier is invalid")
        if payload.get("stage") != stage:
            raise ValueError(f"projection row {index} stage does not match projection summary")
        if payload.get("row_index") != index:
            raise ValueError("projection rows must be ordered by row_index")
        row_provenance = payload.get("provenance")
        if not isinstance(row_provenance, MappingABC):
            raise ValueError(f"projection row {index} provenance is missing")
        if row_provenance.get("behavior_fingerprint") != behavior_fingerprint:
            raise ValueError(f"projection row {index} behavior fingerprint mismatch")
        if row_provenance.get("config_fingerprint") != config_fingerprint:
            raise ValueError(f"projection row {index} config fingerprint mismatch")
        if row_provenance.get("run_fingerprint") != run_fingerprint:
            raise ValueError(f"projection row {index} run fingerprint mismatch")
        if row_provenance.get("stage") not in {None, stage}:
            raise ValueError(f"projection row {index} provenance stage mismatch")
        identity = payload.get("identity")
        if not isinstance(identity, MappingABC):
            raise ValueError(f"projection row {index} identity is missing")
        audio_key = _required_string(identity.get("cache_audio_key"), f"projection row {index} cache_audio_key")
        group_key = _required_string(identity.get("audio_group_key"), f"projection row {index} audio_group_key")
        identity_key_sha = identity.get("cache_audio_key_sha256")
        if identity_key_sha is not None and identity_key_sha != _sha256_text(audio_key):
            raise ValueError(f"projection row {index} cache_audio_key_sha256 mismatch")
        if audio_key in audio_keys:
            raise ValueError(f"duplicate projection audio key: {audio_key!r}")
        audio_keys.add(audio_key)
        rows.append(
            _ProjectionRow(
                row_index=index,
                audio_key=audio_key,
                audio_group_key=group_key,
                payload=payload,
                payload_sha256=_stable_json_sha256(payload),
            )
        )
    if stage in FORMAL_STAGE_AUDIO_COUNTS:
        _require_formal_projection_summary_matches_rows(
            summary,
            stage=stage,
            rows=rows,
            row_payloads=rows_payload,
        )
    return (
        _ProjectionSource(
            path=path,
            sha256=actual_sha,
            summary_path=summary_path,
            summary_sha256=summary_sha256,
            summary=summary,
            stage=stage,
            behavior_fingerprint=behavior_fingerprint,
            config_fingerprint=config_fingerprint,
            run_fingerprint=run_fingerprint,
            hard_guards_ok=bool(hard_guards["ok"]),
            hard_guard_violations=tuple(
                dict(item) if isinstance(item, MappingABC) else {"violation": item}
                for item in hard_guard_violations
            ),
            formal_execution_ready=formal_ready,
            formal_execution_blockers=formal_blockers,
            stage_constraints=dict(stage_constraints),
            prior_baseline_jsonl_sha256=prior_baseline,
        ),
        rows,
    )


def _require_projection_summary_count(
    summary: Mapping[str, Any],
    path: tuple[str, ...],
    expected_count: int,
) -> None:
    value = _nested(summary, *path)
    if value != expected_count:
        dotted = ".".join(path)
        raise ValueError(f"projection summary {dotted} must equal {expected_count}")


def _require_formal_projection_summary_matches_rows(
    summary: Mapping[str, Any],
    *,
    stage: str,
    rows: Sequence[_ProjectionRow],
    row_payloads: Sequence[Mapping[str, Any]],
) -> None:
    ordered_keys = [row.audio_key for row in rows]
    key_set_hash = _key_set_sha256(set(ordered_keys))
    ordered_key_hash = exp004_protocol.ordered_cache_audio_keys_sha256(ordered_keys)
    compact_identity_rows = [
        {
            "schema": exp004_protocol.IDENTITY_ROW_SCHEMA,
            "stage": stage,
            "cache_audio_key": row.audio_key,
            "audio_group_key": row.audio_group_key,
        }
        for row in rows
    ]
    expected_identity_source = {
        "row_count": len(rows),
        "cache_audio_keys_sha256": key_set_hash,
        "ordered_cache_audio_keys_sha256": ordered_key_hash,
        "ordered_identity_rows_sha256": _stable_json_sha256(compact_identity_rows),
    }
    identity_source = _required_mapping(_nested(summary, "source", "identity_rows"), "projection summary source.identity_rows")
    for field_name, expected in expected_identity_source.items():
        if identity_source.get(field_name) != expected:
            raise ValueError(f"projection summary source.identity_rows.{field_name} does not match rows")

    expected_selection_source = {
        "row_count": len(rows),
        "cache_audio_keys_sha256": key_set_hash,
        "ordered_cache_audio_keys_sha256": ordered_key_hash,
    }
    selection_source = _required_mapping(
        _nested(summary, "source", "selection_manifest"),
        "projection summary source.selection_manifest",
    )
    for field_name, expected in expected_selection_source.items():
        if selection_source.get(field_name) != expected:
            raise ValueError(f"projection summary source.selection_manifest.{field_name} does not match rows")
    constraints = selection_source.get("stage_constraints")
    if not isinstance(constraints, MappingABC):
        raise ValueError("projection summary source.selection_manifest.stage_constraints is missing")
    if constraints.get("stage") != stage:
        raise ValueError("projection summary selection stage_constraints stage does not match summary stage")

    successful_count = sum(bool(row.payload.get("ok")) for row in rows)
    failed_count = len(rows) - successful_count
    results = _required_mapping(summary.get("results"), "projection summary results")
    expected_results = {
        "result_count": len(rows),
        "successful_count": successful_count,
        "failed_count": failed_count,
    }
    for field_name, expected in expected_results.items():
        if results.get(field_name) != expected:
            raise ValueError(f"projection summary results.{field_name} does not match rows")

    cache_valid_count = sum(bool(_nested(row.payload, "projection_flags", "cache_valid")) for row in rows)
    projection_evaluable_count = sum(
        bool(_nested(row.payload, "projection_flags", "projection_evaluable"))
        for row in rows
    )
    selected_fallback_count = sum(
        bool(_nested(row.payload, "projection_flags", "selected_used_fallback"))
        for row in rows
    )
    fallback_rate = (
        selected_fallback_count / projection_evaluable_count
        if projection_evaluable_count
        else None
    )
    fallback_reason_counts = Counter(
        str(_nested(row.payload, "selection", "fallback_reason"))
        for row in rows
        if bool(_nested(row.payload, "projection_flags", "selected_used_fallback"))
    )
    denominators = _required_mapping(summary.get("denominators"), "projection summary denominators")
    expected_denominators = {
        "cache_valid_count": cache_valid_count,
        "projection_evaluable_count": projection_evaluable_count,
        "selected_fallback_count": selected_fallback_count,
        "selected_fallback_rate": fallback_rate,
        "fallback_reason_counts": dict(sorted(fallback_reason_counts.items())),
    }
    for field_name, expected in expected_denominators.items():
        if denominators.get(field_name) != expected:
            raise ValueError(f"projection summary denominators.{field_name} does not match rows")

    expected_hard_guards = _projection_hard_guards_from_rows(row_payloads)
    hard_guards = _required_mapping(summary.get("hard_guards"), "projection summary hard_guards")
    if hard_guards.get("ok") != expected_hard_guards["ok"]:
        raise ValueError("projection summary hard_guards.ok does not match rows")
    if hard_guards.get("violations") != expected_hard_guards["violations"]:
        raise ValueError("projection summary hard_guards.violations do not match rows")

    prior_stage = summary.get("source", {}).get("prior_stage") if isinstance(summary.get("source"), MappingABC) else None
    if stage in FORMAL_DECISION_STAGES:
        prior = _required_mapping(prior_stage, "projection summary source.prior_stage")
        for field_name in (
            "stage",
            "behavior_fingerprint",
            "config_fingerprint",
            "weak_output_jsonl_sha256",
            "projection_jsonl_sha256",
            "projection_summary_sha256",
            "baseline_jsonl_sha256",
            "evaluator_sha256",
            "stage_gates_sha256",
            "protocol_binding_sha256",
        ):
            if field_name not in prior:
                raise ValueError(f"projection summary source.prior_stage.{field_name} is missing")
        if prior.get("stage") != exp004_runner.PRIOR_STAGE.get(stage):
            raise ValueError("projection summary source.prior_stage.stage is invalid")
        if prior.get("behavior_fingerprint") != _nested(summary, "provenance", "behavior_fingerprint"):
            raise ValueError("projection summary prior behavior fingerprint does not match current summary")
        if prior.get("config_fingerprint") != _nested(summary, "provenance", "config_fingerprint"):
            raise ValueError("projection summary prior config fingerprint does not match current summary")
        prior_constraints = prior.get("stage_constraints")
        if prior_constraints is not None and prior_constraints != constraints:
            raise ValueError("projection summary prior stage_constraints do not match selection stage_constraints")


def _projection_hard_guards_from_rows(row_payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows_by_reason: dict[str, list[int]] = {}
    for index, payload in enumerate(row_payloads):
        flags = _required_mapping(payload.get("projection_flags"), f"projection row {index} projection_flags")
        raw_reasons = flags.get("hard_guard_reasons")
        if raw_reasons is None:
            raw_reasons = []
        if isinstance(raw_reasons, (str, bytes)) or not isinstance(raw_reasons, SequenceABC):
            raise ValueError(f"projection row {index} hard_guard_reasons must be a sequence")
        hard_guard_violation = flags.get("hard_guard_violation")
        if hard_guard_violation is not None and bool(hard_guard_violation) != bool(raw_reasons):
            raise ValueError(f"projection row {index} hard_guard_violation does not match reasons")
        for reason in raw_reasons:
            rows_by_reason.setdefault(str(reason), []).append(index)
    violations = [
        {"reason": reason, "row_indices": row_indices}
        for reason, row_indices in sorted(rows_by_reason.items())
    ]
    return {"ok": not violations, "violations": violations}


def _load_baseline_source(path: Path) -> _BaselineSource:
    payloads, baseline_sha256 = _load_jsonl_objects_with_sha(path)
    rows_by_key: dict[str, _BaselineRow] = {}
    for index, payload in enumerate(payloads):
        if payload.get("schema") != BASELINE_RESULT_SCHEMA:
            raise ValueError(f"baseline row {index} has unsupported schema")
        if payload.get("row_index") != index:
            raise ValueError(f"baseline row {index} row_index does not match JSONL order")
        audio_key = _required_string(
            payload.get("audio_key") or _nested(payload, "identity", "cache_audio_key"),
            f"baseline row {index} audio_key",
        )
        if audio_key in rows_by_key:
            raise ValueError(f"duplicate baseline audio key: {audio_key!r}")
        v2_grid = None
        try:
            if isinstance(payload.get("fit"), MappingABC):
                v2_grid = _grid_from_segments_payload(payload["fit"].get("predicted_segments"))
            elif isinstance(_nested(payload, "current_v2", "grid"), MappingABC):
                v2_grid = _fitted_grid_from_payload(_nested(payload, "current_v2", "grid"))
        except Exception:
            v2_grid = None
        comparators = tuple(_stored_comparators(payload))
        rows_by_key[audio_key] = _BaselineRow(
            row_index=index,
            audio_key=audio_key,
            payload=payload,
            payload_sha256=_stable_json_sha256(payload),
            v2_grid=v2_grid,
            comparators=comparators,
            evaluation_strata=_baseline_strata(payload),
        )
    return _BaselineSource(path=path, sha256=baseline_sha256, rows_by_key=rows_by_key)


def _stored_comparators(payload: Mapping[str, Any]) -> Iterator[_StoredComparator]:
    comparisons = payload.get("comparisons", [])
    if isinstance(comparisons, (str, bytes)) or not isinstance(comparisons, SequenceABC):
        return
    for index, comparison in enumerate(comparisons):
        if not isinstance(comparison, MappingABC):
            yield _invalid_comparator(index, None, None, comparison, "TypeError", "comparison must be a mapping")
            continue
        beatmap_path = comparison.get("beatmap_path") if isinstance(comparison.get("beatmap_path"), str) else None
        difficulty_key = str(comparison.get("difficulty_key") or beatmap_path or f"difficulty-{index}")
        segments_payload = comparison.get("oracle_segments", [])
        payload_sha = _stable_json_sha256({"oracle_segments": _json_safe(segments_payload), "beatmap_path": beatmap_path})
        if comparison.get("ok") is not True:
            yield _StoredComparator(
                index=index,
                difficulty_key=difficulty_key,
                beatmap_path=beatmap_path,
                payload_sha256=payload_sha,
                oracle_segments_payload=_json_safe(segments_payload),
                oracle_grid=None,
                valid=False,
                error_type=str(comparison.get("error_type") or "ComparatorUnavailable"),
                error=str(comparison.get("error") or "stored comparator is not marked ok"),
            )
            continue
        try:
            grid = _grid_from_segments_payload(segments_payload)
            yield _StoredComparator(
                index=index,
                difficulty_key=difficulty_key,
                beatmap_path=beatmap_path,
                payload_sha256=payload_sha,
                oracle_segments_payload=_segments_payload(grid),
                oracle_grid=grid,
                valid=True,
            )
        except Exception as exc:
            yield _StoredComparator(
                index=index,
                difficulty_key=difficulty_key,
                beatmap_path=beatmap_path,
                payload_sha256=payload_sha,
                oracle_segments_payload=_json_safe(segments_payload),
                oracle_grid=None,
                valid=False,
                error_type=exc.__class__.__name__,
                error=str(exc),
            )


def _invalid_comparator(
    index: int,
    difficulty_key: str | None,
    beatmap_path: str | None,
    payload: Any,
    error_type: str,
    error: str,
) -> _StoredComparator:
    return _StoredComparator(
        index=index,
        difficulty_key=difficulty_key or f"difficulty-{index}",
        beatmap_path=beatmap_path,
        payload_sha256=_stable_json_sha256({"payload": _json_safe(payload)}),
        oracle_segments_payload=_json_safe(payload),
        oracle_grid=None,
        valid=False,
        error_type=error_type,
        error=error,
    )


def _projection_method_grids(row: Mapping[str, Any]) -> dict[str, FittedTimingGrid | TimingV3Grid]:
    grids: dict[str, FittedTimingGrid | TimingV3Grid] = {}
    if _nested(row, "current_v2", "status") == "accepted":
        grids[CURRENT_V2_METHOD] = _fitted_grid_from_payload(_nested(row, "current_v2", "grid"))
    variants = row.get("variants")
    if isinstance(variants, MappingABC):
        for variant in VARIANT_METHODS:
            payload = variants.get(variant)
            if isinstance(payload, MappingABC) and payload.get("status") == "accepted":
                grids[variant] = TimingV3Grid.from_dict(payload["grid"])
    selection_source = _nested(row, "selection", "source")
    if selection_source == CURRENT_V2_METHOD and CURRENT_V2_METHOD in grids:
        grids[SELECTED_METHOD] = grids[CURRENT_V2_METHOD]
    elif isinstance(selection_source, str) and selection_source in grids:
        grids[SELECTED_METHOD] = grids[selection_source]
    return grids


def _projection_coverage_end_ms(row: Mapping[str, Any]) -> float:
    value = _nested(row, "cache", "coverage_end_ms")
    if value is None:
        for grid in _projection_method_grids(row).values():
            if isinstance(grid, TimingV3Grid):
                return float(grid.coverage_end_ms)
    result = _finite_float(value, "coverage_end_ms")
    if result <= 0.0:
        raise ValueError("coverage_end_ms must be positive")
    return result


def _reject_stale_v2_mismatch(projection: _ProjectionRow, baseline: _BaselineRow) -> None:
    if _nested(projection.payload, "current_v2", "status") != "accepted":
        return
    if baseline.v2_grid is None:
        raise ValueError(f"baseline row for {projection.audio_key!r} lacks stored v2 fit")
    projection_grid = _fitted_grid_from_payload(_nested(projection.payload, "current_v2", "grid"))
    if _grid_signature(projection_grid) != _grid_signature(baseline.v2_grid):
        raise ValueError(f"stale v2 mismatch for audio key {projection.audio_key!r}")


def _fitted_grid_from_payload(payload: Any) -> FittedTimingGrid:
    if not isinstance(payload, MappingABC):
        raise ValueError("grid payload must be a mapping")
    return _grid_from_segments_payload(payload.get("segments"))


def _grid_from_segments_payload(payload: Any) -> FittedTimingGrid:
    if isinstance(payload, (str, bytes)) or not isinstance(payload, SequenceABC):
        raise ValueError("segments payload must be a sequence")
    if not payload:
        raise ValueError("segments payload must contain at least one segment")
    segments: list[TimingSegment] = []
    for index, segment in enumerate(payload):
        if not isinstance(segment, MappingABC):
            raise ValueError(f"segments[{index}] must be a mapping")
        for field in ("offset_ms", "beat_length_ms", "meter"):
            if field not in segment:
                raise ValueError(f"segments[{index}].{field} is required")
        segments.append(
            TimingSegment(
                offset_ms=_finite_float(segment["offset_ms"], f"segments[{index}].offset_ms"),
                beat_length_ms=_positive_finite_float(
                    segment["beat_length_ms"],
                    f"segments[{index}].beat_length_ms",
                ),
                meter=_positive_int(segment["meter"], f"segments[{index}].meter"),
            )
        )
    return FittedTimingGrid(tuple(segments))


def _grid_summary_payload(grid: FittedTimingGrid | TimingV3Grid, *, coverage_end_ms: float) -> dict[str, Any]:
    fitted = grid.to_fitted_timing_grid() if isinstance(grid, TimingV3Grid) else grid
    section_durations = []
    offsets = [float(segment.offset_ms) for segment in fitted.segments]
    for index, start in enumerate(offsets):
        end = offsets[index + 1] if index + 1 < len(offsets) else coverage_end_ms
        clipped_start = max(0.0, start)
        clipped_end = min(coverage_end_ms, end)
        if clipped_end > clipped_start:
            section_durations.append(float(clipped_end - clipped_start))
    signature = active_section_signature_v1(
        grid,
        coverage_start_ms=0.0,
        coverage_end_ms=coverage_end_ms,
        canonicalization=TIMING_CANONICALIZATION_NONE,
    )
    return {
        "schema": "pulsefield_model.timing_v3_exp004_grid_summary_v1",
        "section_count": len(fitted.segments),
        "jump_count": max(0, len(fitted.segments) - 1),
        "section_duration_ms": _stats(section_durations),
        "active_section_signature": [asdict(span) for span in signature],
        "segments": _segments_payload(fitted),
    }


def _baseline_result_payload(
    baseline: _BaselineRow | None,
    *,
    baseline_source: _BaselineSource,
    invalid_comparators: Sequence[_StoredComparator],
) -> dict[str, Any]:
    if baseline is None:
        return {
            "baseline_jsonl_sha256": baseline_source.sha256,
            "row_present": False,
            "row_sha256": None,
            "valid_comparator_count": 0,
            "invalid_comparator_count": 0,
            "invalid_comparators": [],
        }
    return {
        "baseline_jsonl_sha256": baseline_source.sha256,
        "row_present": True,
        "row_index": baseline.row_index,
        "row_sha256": baseline.payload_sha256,
        "valid_comparator_count": sum(comparator.valid for comparator in baseline.comparators),
        "invalid_comparator_count": len(invalid_comparators),
        "oracle_segment_payloads": [
            {
                "difficulty_key": comparator.difficulty_key,
                "beatmap_path": comparator.beatmap_path,
                "sha256": comparator.payload_sha256,
            }
            for comparator in baseline.comparators
        ],
        "invalid_comparators": [
            {
                "difficulty_key": comparator.difficulty_key,
                "beatmap_path": comparator.beatmap_path,
                "error_type": comparator.error_type,
                "error": comparator.error,
            }
            for comparator in invalid_comparators
        ],
    }


def _row_reasons(
    row: Mapping[str, Any],
    baseline: _BaselineRow | None,
    valid_comparators: Sequence[_StoredComparator],
    invalid_comparators: Sequence[_StoredComparator],
) -> list[str]:
    reasons: list[str] = []
    if baseline is None:
        reasons.append("baseline_missing")
    if baseline is not None and not valid_comparators:
        reasons.append("no_valid_stored_comparator")
    if invalid_comparators:
        reasons.append("malformed_or_unavailable_stored_comparator")
    if not bool(_nested(row, "projection_flags", "projection_evaluable")):
        reasons.append("projection_not_evaluable")
    if bool(_nested(row, "projection_flags", "selected_used_fallback")):
        reasons.append("selected_used_fallback")
    if _nested(row, "candidate_extraction", "status") != "accepted":
        reasons.append("candidate_extraction_failure")
    if _nested(row, "variants", "CJ3", "status") != "accepted":
        reasons.append("pure_CJ3_unavailable")
    return reasons


def _strata_payload(
    strata: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
    coverage_end_ms: float,
    comparison_eligible: bool,
    valid_comparator_count: int,
    invalid_comparator_count: int,
    methods: Mapping[str, Any],
) -> dict[str, Any]:
    raw_label = str(
        strata.get("pilot_stratum")
        or strata.get("label_stratum")
        or strata.get("stratum")
        or "unknown"
    )
    label = _canonical_stratum(raw_label)
    long_track = bool(strata.get("source_long_track") or strata.get("long_track"))
    grid_summary = _strata_grid_summary(methods)
    section_count = _optional_int(grid_summary.get("section_count")) if grid_summary is not None else None
    jump_count = _optional_int(grid_summary.get("jump_count")) if grid_summary is not None else None
    first_bpm = _first_grid_bpm(grid_summary) if grid_summary is not None else None
    alias_switch_count = _projection_alias_switch_count(row)
    return {
        "primary_stratum": label,
        "source_label_stratum": raw_label,
        "pilot_stratum_preferred": "pilot_stratum" in strata,
        "label_confidence": _optional_float(strata.get("label_confidence")),
        "label_ambiguous": bool(strata.get("label_ambiguous")),
        "stable": label == "stable",
        "jump": label == "jump",
        "long": long_track,
        "dense": label == "dense",
        "ramp": label in {"ramp", "ramp_audit"},
        "anomaly": label in {"anomaly", "ambiguous"} or bool(strata.get("label_ambiguous")),
        "duration_bin": _duration_bin(float(coverage_end_ms)),
        "comparator_availability": _comparator_availability(
            comparison_eligible=comparison_eligible,
            valid_comparator_count=valid_comparator_count,
            invalid_comparator_count=invalid_comparator_count,
        ),
        "beatthis_evidence_tercile": _beatthis_evidence_tercile(row),
        "predicted_section_bin": _section_count_bin(section_count),
        "predicted_jump_bin": _jump_count_bin(jump_count),
        "primary_bpm_band": _bpm_band(first_bpm),
        "alias_switch_count": alias_switch_count,
        "alias_switch_bin": _alias_switch_bin(alias_switch_count),
        "fallback_reason": str(_nested(row, "selection", "fallback_reason") or "none"),
        "source": _json_safe(dict(strata)),
    }


def _baseline_strata(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(payload.get("evaluation_strata"), MappingABC):
        return dict(payload["evaluation_strata"])
    strata: dict[str, Any] = {}
    for key in (
        "pilot_stratum",
        "label_stratum",
        "label_confidence",
        "label_ambiguous",
        "source_long_track",
        "stratum",
        "long_track",
    ):
        if key in payload:
            strata[key] = payload[key]
    return strata


def _canonical_stratum(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "jump_candidate":
        return "jump"
    if normalized == "ramp_candidate":
        return "ramp"
    if normalized in {"ramp_audit", "ramp-audit"}:
        return "ramp_audit"
    if normalized in {"stable", "jump", "dense", "ramp", "ambiguous", "anomaly", "long"}:
        return normalized
    return "unknown"


def _duration_bin(coverage_end_ms: float) -> str:
    seconds = coverage_end_ms / 1000.0
    if seconds < 60.0:
        return "<60s"
    if seconds < 180.0:
        return "60-180s"
    if seconds < 600.0:
        return "180-600s"
    return ">=600s"


def _comparator_availability(
    *,
    comparison_eligible: bool,
    valid_comparator_count: int,
    invalid_comparator_count: int,
) -> str:
    if comparison_eligible and invalid_comparator_count:
        return "valid_with_invalid_siblings"
    if comparison_eligible:
        return "valid"
    if invalid_comparator_count:
        return "invalid_only"
    return "missing"


def _beatthis_evidence_tercile(row: Mapping[str, Any]) -> str:
    for path in (
        ("candidate_extraction", "diagnostics", "beatthis_evidence_tercile"),
        ("candidate_extraction", "diagnostics", "peak_support_tercile"),
        ("candidate_extraction", "diagnostics", "beat_peak_support_tercile"),
    ):
        value = _nested(row, *path)
        if isinstance(value, str) and value:
            return value
    return "unavailable"


def _strata_grid_summary(methods: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for method in ("CJ3", SELECTED_METHOD, CURRENT_V2_METHOD):
        grid = _nested(methods, method, "grid")
        if isinstance(grid, MappingABC):
            return grid
    return None


def _first_grid_bpm(grid_summary: Mapping[str, Any]) -> float | None:
    segments = grid_summary.get("segments")
    if isinstance(segments, SequenceABC) and not isinstance(segments, (str, bytes)) and segments:
        first = segments[0]
        if isinstance(first, MappingABC):
            beat_length = _optional_float(first.get("beat_length_ms"))
            if beat_length is not None and beat_length > 0.0:
                return float(60000.0 / beat_length)
    return None


def _section_count_bin(count: int | None) -> str:
    if count is None:
        return "unavailable"
    if count <= 1:
        return "1"
    if count <= 4:
        return "2-4"
    if count <= 10:
        return "5-10"
    return ">10"


def _jump_count_bin(count: int | None) -> str:
    if count is None:
        return "unavailable"
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 4:
        return "2-4"
    return ">4"


def _bpm_band(bpm: float | None) -> str:
    if bpm is None:
        return "unavailable"
    if bpm < 80.0:
        return "<80"
    if bpm <= 160.0:
        return "80-160"
    if bpm <= 240.0:
        return "160-240"
    return ">240"


def _projection_alias_switch_count(row: Mapping[str, Any]) -> int | None:
    for path in (
        ("variants", "CJ3", "diagnostics", "alias_switch_count"),
        ("variants", "CJ3", "diagnostics", "alias_switches"),
        ("selection", "diagnostics", "alias_switch_count"),
    ):
        value = _nested(row, *path)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value >= 0:
            return value
    return None


def _alias_switch_bin(count: int | None) -> str:
    if count is None:
        return "unavailable"
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    return ">1"


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _method_unavailable_reason(row: Mapping[str, Any], method: str) -> str:
    if method == CURRENT_V2_METHOD:
        return str(_nested(row, "current_v2", "reason") or "current_v2_unavailable")
    if method == SELECTED_METHOD:
        return str(_nested(row, "selection", "fallback_reason") or "selected_grid_unavailable")
    return str(_nested(row, "variants", method, "reason") or f"{method}_unavailable")


def _resume_payload(
    *,
    projection: _ProjectionRow,
    projection_source: _ProjectionSource,
    baseline: _BaselineRow | None,
    baseline_source: _BaselineSource,
    evaluator_source: Mapping[str, Any],
) -> dict[str, Any]:
    oracle_payloads = [
        {
            "difficulty_key": comparator.difficulty_key,
            "beatmap_path": comparator.beatmap_path,
            "oracle_segments_sha256": comparator.payload_sha256,
            "valid": comparator.valid,
        }
        for comparator in (baseline.comparators if baseline is not None else ())
    ]
    components = {
        "schema": RESUME_SCHEMA,
        "projection_jsonl_sha256": projection_source.sha256,
        "projection_summary_sha256": projection_source.summary_sha256,
        "projection_behavior_fingerprint": projection_source.behavior_fingerprint,
        "projection_config_fingerprint": projection_source.config_fingerprint,
        "projection_run_fingerprint": projection_source.run_fingerprint,
        "projection_row_index": projection.row_index,
        "projection_row_sha256": projection.payload_sha256,
        "baseline_jsonl_sha256": baseline_source.sha256,
        "baseline_row_sha256": baseline.payload_sha256 if baseline is not None else None,
        "oracle_segment_payloads": oracle_payloads,
        "canonicalization": PRIMARY_CANONICALIZATION,
        "metrics_schema": EXP004_METRICS_SCHEMA,
        "evaluator_source": dict(evaluator_source),
    }
    return {
        "schema": RESUME_SCHEMA,
        "fingerprint": _stable_json_sha256(components),
        "components": _json_safe(components),
    }


def _evaluator_source_payload() -> dict[str, Any]:
    this_path = Path(__file__).resolve(strict=False)
    metrics_module = __import__(
        "pulsefield_model.timing.evaluation.exp004_metrics",
        fromlist=["__file__"],
    )
    metrics_path = Path(str(metrics_module.__file__)).resolve(strict=False)
    return {
        "schema": "pulsefield_model.timing_v3_exp004_weak_evaluator_source_v1",
        "evaluator_path": this_path.as_posix(),
        "evaluator_sha256": _file_sha256(this_path),
        "metrics_path": metrics_path.as_posix(),
        "metrics_sha256": _file_sha256(metrics_path),
        "canonical_bpm_binding": canonical_bpm_binding_for_exp004().to_dict(),
    }


def _load_existing_results(
    path: Path,
    *,
    projection_rows: Sequence[_ProjectionRow],
    projection_jsonl_sha256: str,
) -> tuple[dict[int, dict[str, Any]], int]:
    if not path.exists():
        return {}, 0
    rows = _load_jsonl_objects(path)
    by_index: dict[int, dict[str, Any]] = {}
    fingerprints: set[str] = set()
    audio_keys: set[str] = set()
    stale_count = 0
    for line_index, row in enumerate(rows):
        if row.get("schema") != RESULT_SCHEMA:
            raise ValueError(f"existing weak result row {line_index} has wrong schema")
        if row.get("row_complete") is not True:
            raise ValueError(f"existing weak result row {line_index} is incomplete")
        row_index = row.get("row_index")
        if isinstance(row_index, bool) or not isinstance(row_index, int) or not 0 <= row_index < len(projection_rows):
            raise ValueError(f"existing weak result row {line_index} has invalid row_index")
        result_fingerprint = row.get("result_fingerprint")
        _require_sha256(result_fingerprint, f"existing weak result row {line_index} result fingerprint")
        fingerprint_body = dict(row)
        fingerprint_body.pop("result_fingerprint", None)
        if str(result_fingerprint) != _stable_json_sha256(fingerprint_body):
            stale_count += 1
            continue
        identity = row.get("identity")
        if not isinstance(identity, MappingABC):
            raise ValueError(f"existing weak result row {line_index} identity is missing")
        audio_key = _required_string(identity.get("cache_audio_key"), "existing weak result cache_audio_key")
        expected_projection = projection_rows[row_index]
        if row.get("stage") != expected_projection.payload.get("stage"):
            stale_count += 1
            continue
        if audio_key != expected_projection.audio_key:
            stale_count += 1
            continue
        if identity.get("audio_group_key") != expected_projection.audio_group_key:
            stale_count += 1
            continue
        if _nested(row, "projection", "jsonl_sha256") != projection_jsonl_sha256:
            stale_count += 1
            continue
        if _nested(row, "projection", "row_sha256") != expected_projection.payload_sha256:
            stale_count += 1
            continue
        if _nested(row, "projection", "row", "row_index") != expected_projection.row_index:
            stale_count += 1
            continue
        if _nested(row, "projection", "row", "identity", "cache_audio_key") != expected_projection.audio_key:
            stale_count += 1
            continue
        resume = row.get("resume")
        if not isinstance(resume, MappingABC) or resume.get("schema") != RESUME_SCHEMA:
            raise ValueError(f"existing weak result row {line_index} resume is invalid")
        fingerprint = resume.get("fingerprint")
        _require_sha256(fingerprint, f"existing weak result row {line_index} resume fingerprint")
        if row_index in by_index:
            raise ValueError(f"duplicate existing weak result row_index: {row_index}")
        if audio_key in audio_keys:
            raise ValueError(f"duplicate existing weak result audio key: {audio_key!r}")
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate existing weak result resume fingerprint: {fingerprint}")
        by_index[row_index] = row
        audio_keys.add(audio_key)
        fingerprints.add(str(fingerprint))
    return by_index, stale_count


def _require_source_hashes_unchanged(
    projection_source: _ProjectionSource,
    baseline_source: _BaselineSource,
) -> None:
    if _file_sha256(projection_source.path) != projection_source.sha256:
        raise RuntimeError("projection JSONL changed during weak evaluation")
    if _file_sha256(projection_source.summary_path) != projection_source.summary_sha256:
        raise RuntimeError("projection summary changed during weak evaluation")
    if _file_sha256(baseline_source.path) != baseline_source.sha256:
        raise RuntimeError("baseline JSONL changed during weak evaluation")


def _require_projection_prior_baseline_matches(
    projection_source: _ProjectionSource,
    baseline_source: _BaselineSource,
) -> None:
    if projection_source.stage not in FORMAL_DECISION_STAGES:
        return
    expected = projection_source.prior_baseline_jsonl_sha256
    if expected != baseline_source.sha256:
        raise ValueError("projection summary prior baseline_jsonl_sha256 does not match baseline JSONL")


def _require_evaluator_source_unchanged(evaluator_source: Mapping[str, Any]) -> None:
    checks = (
        ("evaluator_path", "evaluator_sha256"),
        ("metrics_path", "metrics_sha256"),
    )
    for path_key, sha_key in checks:
        path = Path(_required_string(evaluator_source.get(path_key), path_key))
        expected_sha = evaluator_source.get(sha_key)
        _require_sha256(expected_sha, sha_key)
        if _file_sha256(path) != expected_sha:
            raise RuntimeError(f"{path_key} changed during weak evaluation")
    canonical_binding = evaluator_source.get("canonical_bpm_binding")
    if not isinstance(canonical_binding, MappingABC):
        raise ValueError("evaluator source canonical_bpm_binding is missing")
    canonical_path = Path(_required_string(canonical_binding.get("source_path"), "canonical source_path"))
    canonical_sha = canonical_binding.get("source_sha256")
    _require_sha256(canonical_sha, "canonical source_sha256")
    if _file_sha256(canonical_path) != canonical_sha:
        raise RuntimeError("canonical BPM source changed during weak evaluation")


def _segments_payload(grid: FittedTimingGrid) -> list[dict[str, Any]]:
    return [
        {
            "offset_ms": float(segment.offset_ms),
            "beat_length_ms": float(segment.beat_length_ms),
            "meter": int(segment.meter),
        }
        for segment in grid.segments
    ]


def _grid_signature(grid: FittedTimingGrid) -> tuple[tuple[float, float, int], ...]:
    return tuple((float(segment.offset_ms), float(segment.beat_length_ms), int(segment.meter)) for segment in grid.segments)


def _stats(values: Any) -> dict[str, Any]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "max": None}
    array = np.asarray(finite, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50.0, method="linear")),
        "p90": float(np.percentile(array, 90.0, method="linear")),
        "max": float(np.max(array)),
    }


def _mean(values: Sequence[float] | Any) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(np.asarray(finite, dtype=np.float64))) if finite else None


def _median(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    return float(np.median(np.asarray(finite, dtype=np.float64)))


def _np_p90(values: Sequence[float] | Any) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return float(np.percentile(np.asarray(finite, dtype=np.float64), 90.0, method="linear"))


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _abs_optional_float(value: Any) -> float | None:
    result = _optional_float(value)
    return None if result is None else abs(result)


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


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, MappingABC):
        raise ValueError(f"{name} must be a mapping")
    return value


def _required_sha_from_mapping(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    _require_sha256(value, key)
    return str(value)


def _require_sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase sha256 hex digest")


def _nested(payload: Any, *keys: str) -> Any:
    value = payload
    for key in keys:
        if not isinstance(value, MappingABC):
            return None
        value = value.get(key)
    return value


def _reject_path_aliases(paths: Mapping[str, Path | None]) -> None:
    resolved = {
        name: path.expanduser().resolve(strict=False)
        for name, path in paths.items()
        if path is not None
    }
    names = tuple(resolved)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1:]:
            if resolved[left_name] == resolved[right_name]:
                raise ValueError(f"{left_name} and {right_name} path aliases are not allowed")


@contextlib.contextmanager
def _exclusive_run_lock(path: Path, *, run_fingerprint: str) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Exp004 weak output is locked: {path}") from exc
    try:
        payload = _canonical_json({"pid": os.getpid(), "run_fingerprint": run_fingerprint}) + "\n"
        os.write(descriptor, payload.encode("utf-8"))
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        _fsync_parent(path.parent)
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)
        _fsync_parent(path.parent)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [_canonical_json(row) for row in rows]
    _write_bytes_atomic(path, ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"))


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_bytes_atomic(path, (_canonical_json(payload) + "\n").encode("utf-8"))


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_parent(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_json_object(path: Path) -> dict[str, Any]:
    return _load_json_object_with_sha(path)[0]


def _load_json_object_with_sha(path: Path) -> tuple[dict[str, Any], str]:
    data, sha256 = _read_stable_bytes(path)
    try:
        payload = json.loads(data.decode("utf-8"), parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return _json_safe(payload), sha256


def _load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    return _load_jsonl_objects_with_sha(path)[0]


def _load_jsonl_objects_with_sha(path: Path) -> tuple[list[dict[str, Any]], str]:
    data, sha256 = _read_stable_bytes(path)
    rows: list[dict[str, Any]] = []
    text = data.decode("utf-8")
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            raise ValueError(f"{path}:{line_number} blank JSONL lines are not allowed")
        try:
            payload = json.loads(raw, parse_constant=_reject_json_constant)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        rows.append(_json_safe(payload))
    return rows, sha256


def _read_stable_bytes(path: Path) -> tuple[bytes, str]:
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        data = handle.read()
        after = os.fstat(handle.fileno())
    if _stat_read_identity(before) != _stat_read_identity(after):
        raise RuntimeError(f"{path} changed while being read")
    return data, hashlib.sha256(data).hexdigest()


def _stat_read_identity(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(stat_result.st_dev),
        int(stat_result.st_ino),
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_ctime_ns),
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is not allowed: {value}")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, MappingABC):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite JSON value")
        return float(value)
    return value


def _stable_json_sha256(payload: Mapping[str, Any] | Any) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _key_set_sha256(values: set[str]) -> str:
    return _stable_json_sha256(sorted(values))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Timing-v3 Experiment 004 weak oracle evidence")
    parser.add_argument("--projection-jsonl", type=Path, required=True)
    parser.add_argument("--projection-summary", type=Path, required=True)
    parser.add_argument("--baseline-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--retry-failures", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = run_exp004_weak_evidence(
        projection_jsonl_path=args.projection_jsonl,
        projection_summary_path=args.projection_summary,
        baseline_jsonl_path=args.baseline_jsonl,
        output_jsonl_path=args.output_jsonl,
        summary_json_path=args.summary_json,
        checkpoint_every=args.checkpoint_every,
        retry_failures=args.retry_failures,
    )
    print(_canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
