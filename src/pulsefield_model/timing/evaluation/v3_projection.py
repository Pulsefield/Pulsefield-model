from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, TextIO

import numpy as np

from pulsefield_model.timing.canonicalization import (
    TIMING_CANONICALIZATION_BPM_80_160,
    canonicalize_timing_grid,
)
from pulsefield_model.timing.diagnostics import compare_timing_grids
from pulsefield_model.timing.evaluation.drift import (
    compare_timing_grid_drift,
    predicted_boundary_discontinuities_ms,
)
from pulsefield_model.timing.evaluation.source_projection import (
    SOURCE_PROJECTION_COMPARISON_SCHEMA,
    compare_timing_v3_projection_to_source,
)
from pulsefield_model.timing.schema import FittedTimingGrid, TimingSegment
from pulsefield_model.timing.v3.joint_projection import (
    JOINT_ADJACENT_BEAT_DISPLACEMENT_LIMIT,
    JOINT_MAX_BPM,
    JOINT_MAX_RELATIVE_BPM_ADJUSTMENT,
    JOINT_MIN_BPM,
    JOINT_SOLVER_NORMALIZED_RESIDUAL_LIMIT,
    PROJECTION_METHOD_JOINT_FIXED_COUNTS,
    PROJECTION_METHOD_JOINT_NEARBY_COUNTS,
    TimingV3JointProjectionResult,
    project_joint_phase_fixed_counts,
    project_joint_phase_nearby_counts,
)
from pulsefield_model.timing.v3.projection import (
    DEFAULT_MAX_BPM,
    DEFAULT_MAX_RELATIVE_BPM_ADJUSTMENT,
    DEFAULT_MIN_BPM,
    PROJECTION_METHOD_PRESERVE_ANCHORS,
    PROJECTION_METHOD_PRESERVE_BPM,
    TimingV3ProjectionResult,
    project_preserve_anchors,
    project_preserve_bpm,
)
from pulsefield_model.timing.v3.schema import (
    TIMING_V3_GRID_SCHEMA,
    TIMING_V3_GRID_VERSION,
    TimingV3Grid,
    roundtrip_seam_tolerance_ms,
)


BASELINE_RESULT_SCHEMA = "pulsefield_model.timing_v3_cache_backed_v2_baseline_result_v2"
RESULT_SCHEMA = "pulsefield_model.timing_v3_projection_evaluation_result_v3"
SUMMARY_SCHEMA = "pulsefield_model.timing_v3_projection_evaluation_summary_v3"
RESUME_SCHEMA = "pulsefield_model.timing_v3_projection_evaluation_resume_v3"

V2_METHOD = "v2"
FAMILY_A = "family_a"
FAMILY_B = "family_b"
FAMILY_C0 = "family_c0"
FAMILY_C1 = "family_c1"
FALLBACK_V2 = "fallback_v2"
SELECTED_CANDIDATE = "selected_candidate"
SELECTED_CANDIDATE_METHOD = "selected_family_b_or_fallback_v2"
SELECTED_FAMILY_C = "selected_family_c"
SELECTED_FAMILY_C_METHOD = "selected_family_c1_or_fallback_v2"

_GRID_V2_SEGMENTS_SCHEMA = "pulsefield_model.timing_v2_grid_segments_v1"
_PROJECTION_CONFIG_SCHEMA = "pulsefield_model.timing_v3_projection_eval_config_v3"
_BEHAVIOR_PROVENANCE_SCHEMA = "pulsefield_model.timing_v3_projection_eval_behavior_v2"

_BEHAVIOR_SOURCE_MODULES = (
    "pulsefield_model.timing.evaluation.v3_projection",
    "pulsefield_model.timing.v3.projection",
    "pulsefield_model.timing.v3.joint_projection",
    "pulsefield_model.timing.v3.schema",
    "pulsefield_model.timing.evaluation.source_projection",
    "pulsefield_model.timing.diagnostics.compare_to_oracle",
    "pulsefield_model.timing.evaluation.drift",
    "pulsefield_model.timing.canonicalization",
    "pulsefield_model.timing.rendering.dense_timing_v2",
)

_STRATIFIED_FIELDS = (
    "pilot_stratum",
    "label_stratum",
    "label_confidence",
    "label_ambiguous",
    "source_long_track",
)

_METRIC_NAMES = (
    "mean_phase_error_ms",
    "max_phase_error_ms",
    "mean_phase_error_beats",
    "max_phase_error_beats",
    "local_bpm_mae",
    "local_bpm_alias_mae",
    "beat_pulse_mae",
    "raw_abs_endpoint_relative_drift_ms",
    "alias_abs_endpoint_relative_drift_ms",
    "raw_max_abs_prefix_relative_drift_ms",
    "alias_max_abs_prefix_relative_drift_ms",
    "raw_abs_drift_slope_ms_per_minute",
    "alias_abs_drift_slope_ms_per_minute",
    "alias_p90_abs_30s_relative_drift_ms",
    "alias_p90_abs_60s_relative_drift_ms",
    "raw_max_predicted_boundary_discontinuity_ms",
    "alias_max_predicted_boundary_discontinuity_ms",
    "predicted_segment_count",
    "canonical_oracle_segment_count",
    "abs_canonical_segment_count_delta",
)

_DRIFT_GUARD_METRIC_NAMES = (
    "raw_abs_endpoint_relative_drift_ms",
    "alias_abs_endpoint_relative_drift_ms",
    "raw_max_abs_prefix_relative_drift_ms",
    "alias_max_abs_prefix_relative_drift_ms",
    "raw_abs_drift_slope_ms_per_minute",
    "alias_abs_drift_slope_ms_per_minute",
    "alias_p90_abs_30s_relative_drift_ms",
    "alias_p90_abs_60s_relative_drift_ms",
)

_SOURCE_METRIC_NAMES = (
    "wrapped_phase_mean_beats",
    "wrapped_phase_rms_beats",
    "wrapped_phase_p90_beats",
    "wrapped_phase_max_beats",
    "wrapped_phase_mean_ms",
    "wrapped_phase_rms_ms",
    "wrapped_phase_p90_ms",
    "wrapped_phase_max_ms",
    "local_bpm_mae",
    "local_bpm_rmse",
    "local_bpm_p90_abs_error",
    "local_bpm_max_abs_error",
    "local_bpm_relative_mae",
    "local_bpm_relative_p90_abs_error",
    "local_bpm_relative_max_abs_error",
    "abs_initial_signed_phase_error_beats",
    "abs_initial_signed_phase_error_ms",
    "abs_endpoint_relative_drift_beats",
    "abs_endpoint_relative_drift_ms",
    "max_abs_prefix_relative_drift_beats",
    "max_abs_prefix_relative_drift_ms",
    "abs_drift_slope_beats_per_minute",
    "abs_drift_slope_ms_per_minute",
    "active_section_disagreement_frame_count",
    "active_section_disagreement_fraction",
    "moved_paired_boundary_count",
    "unmatched_boundary_count",
)


@dataclass(frozen=True)
class TimingV3ProjectionEvalConfig:
    min_bpm: float = DEFAULT_MIN_BPM
    max_bpm: float = DEFAULT_MAX_BPM
    max_relative_bpm_adjustment: float = DEFAULT_MAX_RELATIVE_BPM_ADJUSTMENT
    coverage_start_ms: float = 0.0
    canonicalization: str = TIMING_CANONICALIZATION_BPM_80_160
    include_family_c: bool = False


@dataclass(frozen=True)
class _BaselineInputRow:
    row_index: int
    line_number: int
    raw_line: str
    row_content_sha256: str
    payload: dict[str, Any] | None
    parse_error: str | None = None
    parse_error_type: str | None = None


@dataclass(frozen=True)
class _StoredOraclePayload:
    baseline_comparison_index: int
    beatmap_path: str | None
    oracle_grid: FittedTimingGrid | None
    oracle_segments_payload: Any
    error_type: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.oracle_grid is not None


def run_timing_v3_projection_evaluation(
    *,
    baseline_jsonl_path: str | Path,
    output_jsonl_path: str | Path,
    summary_json_path: str | Path | None = None,
    limit: int | None = None,
    retry_failures: bool = False,
    progress_every: int = 25,
    checkpoint_every: int = 25,
    workers: int = 1,
    projection_config: TimingV3ProjectionEvalConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate Timing v3 projections from durable v2 baseline JSONL.

    This runner intentionally has no path back to BeatThis caches, oracle .osu
    files, or the v2 fitter. Stored baseline segments and stored oracle segment
    payloads are the only timing inputs. Family C0/C1 is opt-in so the default
    remains the Experiment 002 A/B evaluation surface.
    """

    started_at_unix = time.time()
    baseline_jsonl_path = Path(baseline_jsonl_path)
    output_jsonl_path = Path(output_jsonl_path)
    summary_json_path = (
        Path(summary_json_path)
        if summary_json_path is not None
        else output_jsonl_path.with_suffix(output_jsonl_path.suffix + ".summary.json")
    )
    _reject_equal_paths(
        baseline_jsonl_path=baseline_jsonl_path,
        output_jsonl_path=output_jsonl_path,
        summary_json_path=summary_json_path,
    )
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be non-negative, got {limit!r}")
    if progress_every < 0:
        raise ValueError(f"progress_every must be non-negative, got {progress_every!r}")
    if checkpoint_every < 0:
        raise ValueError(f"checkpoint_every must be non-negative, got {checkpoint_every!r}")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError(f"workers must be a positive int, got {workers!r}")
    if workers != 1:
        raise ValueError("Timing v3 projection evaluation currently supports workers=1 only")

    config = _normalize_projection_config(projection_config)
    behavior_provenance = _behavior_provenance()
    baseline_sha256 = _file_sha256(baseline_jsonl_path)
    rows = _read_baseline_input_rows(baseline_jsonl_path, limit=limit)
    resume_by_fingerprint = {
        resume["fingerprint"]: resume
        for row in rows
        for resume in (
            _resume_payload(
                row,
                baseline_jsonl_path=baseline_jsonl_path,
                baseline_sha256=baseline_sha256,
                config=config,
                behavior_provenance=behavior_provenance,
            ),
        )
    }
    existing_results, stale_existing_count = _matching_existing_results(
        output_jsonl_path,
        expected_fingerprints=set(resume_by_fingerprint),
    )

    results_by_fingerprint: dict[str, dict[str, Any]] = {}
    jobs: list[tuple[_BaselineInputRow, dict[str, Any]]] = []
    skipped_success_count = 0
    skipped_failure_count = 0
    for row in rows:
        resume = _resume_payload(
            row,
            baseline_jsonl_path=baseline_jsonl_path,
            baseline_sha256=baseline_sha256,
            config=config,
            behavior_provenance=behavior_provenance,
        )
        existing = existing_results.get(resume["fingerprint"])
        if existing is not None and (existing.get("ok") or not retry_failures):
            results_by_fingerprint[resume["fingerprint"]] = existing
            if existing.get("ok"):
                skipped_success_count += 1
            else:
                skipped_failure_count += 1
            continue
        jobs.append((row, resume))

    processed_count = 0
    for row, resume in jobs:
        result = _evaluate_baseline_row(row, resume=resume, config=config)
        results_by_fingerprint[resume["fingerprint"]] = result
        processed_count += 1
        if progress_every and processed_count % progress_every == 0:
            _print_progress(processed_count=processed_count, total_count=len(jobs), result=result)
        if checkpoint_every and processed_count % checkpoint_every == 0:
            _write_result_jsonl_atomic(
                output_jsonl_path,
                _ordered_results(rows, results_by_fingerprint, resume_by_fingerprint),
            )

    ordered_results = _ordered_results(rows, results_by_fingerprint, resume_by_fingerprint)
    _write_result_jsonl_atomic(output_jsonl_path, ordered_results)
    summary = _summary_payload(
        rows=rows,
        results=ordered_results,
        baseline_jsonl_path=baseline_jsonl_path,
        output_jsonl_path=output_jsonl_path,
        summary_json_path=summary_json_path,
        baseline_sha256=baseline_sha256,
        config=config,
        behavior_provenance=behavior_provenance,
        started_at_unix=started_at_unix,
        processed_count=processed_count,
        skipped_success_count=skipped_success_count,
        skipped_failure_count=skipped_failure_count,
        stale_existing_count=stale_existing_count,
        retry_failures=retry_failures,
        checkpoint_every=checkpoint_every,
        progress_every=progress_every,
        workers=workers,
        limit=limit,
    )
    _write_json_atomic(summary_json_path, summary)
    return summary


run_timing_v3_projection_eval = run_timing_v3_projection_evaluation


def _evaluate_baseline_row(
    row: _BaselineInputRow,
    *,
    resume: Mapping[str, Any],
    config: TimingV3ProjectionEvalConfig,
) -> dict[str, Any]:
    row_started_at_unix = time.time()
    if row.parse_error is not None:
        return _failure_result(
            row,
            resume=resume,
            stage="baseline_row",
            error_type=row.parse_error_type or "ValueError",
            error=row.parse_error,
            started_at_unix=row_started_at_unix,
        )
    payload = row.payload
    if payload is None:
        return _failure_result(
            row,
            resume=resume,
            stage="baseline_row",
            error_type="ValueError",
            error="baseline row must be a JSON object",
            started_at_unix=row_started_at_unix,
        )

    try:
        _require_baseline_result_payload(payload)
        _require_projection_evaluable_baseline_payload(payload)
        source_grid = _grid_from_segments_payload(_required_mapping(payload, "fit")["predicted_segments"])
        prediction = _required_mapping(payload, "prediction")
        frame_count = _positive_int(prediction.get("frame_count"), "prediction.frame_count")
        frame_rate_hz = _positive_float(prediction.get("frame_rate_hz"), "prediction.frame_rate_hz")
        coverage_end_ms = 1000.0 * frame_count / frame_rate_hz
        stored_oracles = _stored_oracle_payloads(payload)
    except Exception as exc:  # noqa: BLE001 - malformed durable rows are isolated row failures.
        return _failure_result(
            row,
            resume=resume,
            stage=_baseline_unusable_stage(payload),
            error_type=exc.__class__.__name__,
            error=str(exc),
            started_at_unix=row_started_at_unix,
        )

    valid_oracle_count = sum(1 for oracle in stored_oracles if oracle.ok)
    comparator_available = valid_oracle_count > 0
    v2 = _v2_method_payload(source_grid, stored_oracles, frame_count=frame_count, config=config)
    family_a_started = time.perf_counter()
    family_a_projection = project_preserve_bpm(
        source_grid,
        coverage_start_ms=config.coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
        min_bpm=config.min_bpm,
        max_bpm=config.max_bpm,
        max_relative_bpm_adjustment=config.max_relative_bpm_adjustment,
    )
    family_a_projection_seconds = time.perf_counter() - family_a_started
    family_b_started = time.perf_counter()
    family_b_projection = project_preserve_anchors(
        source_grid,
        coverage_start_ms=config.coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
        min_bpm=config.min_bpm,
        max_bpm=config.max_bpm,
        max_relative_bpm_adjustment=config.max_relative_bpm_adjustment,
    )
    family_b_projection_seconds = time.perf_counter() - family_b_started
    family_a = _projection_method_payload(
        source_grid,
        family_a_projection,
        stored_oracles,
        frame_count=frame_count,
        frame_rate_hz=frame_rate_hz,
        config=config,
        selected_candidate=False,
        projection_seconds=family_a_projection_seconds,
    )
    family_b = _projection_method_payload(
        source_grid,
        family_b_projection,
        stored_oracles,
        frame_count=frame_count,
        frame_rate_hz=frame_rate_hz,
        config=config,
        selected_candidate=True,
        projection_seconds=family_b_projection_seconds,
    )
    selected_candidate = _selected_candidate_payload(v2, family_b)

    family_c_payloads: dict[str, Any] = {}
    if config.include_family_c:
        family_c0_started = time.perf_counter()
        family_c0_projection = project_joint_phase_fixed_counts(
            source_grid,
            coverage_start_ms=config.coverage_start_ms,
            coverage_end_ms=coverage_end_ms,
        )
        family_c0_projection_seconds = time.perf_counter() - family_c0_started
        family_c1_started = time.perf_counter()
        family_c1_projection = project_joint_phase_nearby_counts(
            source_grid,
            coverage_start_ms=config.coverage_start_ms,
            coverage_end_ms=coverage_end_ms,
        )
        family_c1_projection_seconds = time.perf_counter() - family_c1_started
        family_c0 = _projection_method_payload(
            source_grid,
            family_c0_projection,
            stored_oracles,
            frame_count=frame_count,
            frame_rate_hz=frame_rate_hz,
            config=config,
            selected_candidate=True,
            projection_seconds=family_c0_projection_seconds,
        )
        family_c1 = _projection_method_payload(
            source_grid,
            family_c1_projection,
            stored_oracles,
            frame_count=frame_count,
            frame_rate_hz=frame_rate_hz,
            config=config,
            selected_candidate=True,
            projection_seconds=family_c1_projection_seconds,
        )
        family_c_payloads = {
            FAMILY_C0: family_c0,
            FAMILY_C1: family_c1,
            SELECTED_FAMILY_C: _selected_candidate_payload(
                v2,
                family_c1,
                candidate_family=FAMILY_C1,
                selected_method=SELECTED_FAMILY_C_METHOD,
            ),
        }

    return _json_safe(
        {
            "schema": RESULT_SCHEMA,
            "resume": resume,
            "ok": True,
            "projection_evaluable": True,
            "comparator_available": comparator_available,
            "comparison_eligible": comparator_available,
            "stored_oracle_comparison_count": len(stored_oracles),
            "valid_stored_oracle_comparison_count": valid_oracle_count,
            "baseline_ok": bool(payload.get("ok")),
            "baseline_failure_stage": payload.get("failure_stage"),
            "audio_key": _audio_key(payload),
            "row_index": row.row_index,
            "baseline_row_index": payload.get("row_index"),
            "baseline_line_number": row.line_number,
            "source_line_numbers": _json_safe(payload.get("source_line_numbers")),
            "evaluation_strata": _json_safe(payload.get("evaluation_strata", {})),
            "audio_path": payload.get("audio_path"),
            "beatmap_paths": _json_safe(payload.get("beatmap_paths", [])),
            "prediction": _json_safe(payload.get("prediction")),
            "baseline_paired_metrics": _json_safe(payload.get("paired_metrics")),
            "provenance": resume["components"],
            V2_METHOD: v2,
            FAMILY_A: family_a,
            FAMILY_B: family_b,
            SELECTED_CANDIDATE: selected_candidate,
            **family_c_payloads,
            "failure_stage": None,
            "error_type": None,
            "error": None,
            "runtime": _row_runtime_payload(row_started_at_unix),
        }
    )


def _v2_method_payload(
    source_grid: FittedTimingGrid,
    stored_oracles: Sequence[_StoredOraclePayload],
    *,
    frame_count: int,
    config: TimingV3ProjectionEvalConfig,
) -> dict[str, Any]:
    comparisons = _compare_grid_to_oracles(source_grid, stored_oracles, frame_count=frame_count, config=config)
    return {
        "ok": True,
        "method": V2_METHOD,
        "grid": _v2_grid_payload(source_grid),
        "projected_segments": _segments_payload(source_grid),
        "section_count": len(source_grid.segments),
        "boundary_seam_ms": _boundary_seam_payload(source_grid),
        "comparisons": comparisons,
        "paired_metrics": _paired_metrics_payload(comparisons),
    }


def _projection_method_payload(
    source_grid: FittedTimingGrid,
    projection: TimingV3ProjectionResult | TimingV3JointProjectionResult,
    stored_oracles: Sequence[_StoredOraclePayload],
    *,
    frame_count: int,
    frame_rate_hz: float,
    config: TimingV3ProjectionEvalConfig,
    selected_candidate: bool,
    projection_seconds: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": projection.ok,
        "method": projection.method,
        "projection": _projection_payload(projection),
        "fallback": None,
        "grid": projection.grid.to_dict() if projection.ok else None,
        "projected_segments": [],
        "section_count": 0,
        "boundary_seam_ms": None,
        "original_vs_projected_boundary_seam_ms": None,
        "serialization": None,
        "projection_seconds": float(projection_seconds),
        "bpm_adjustment": _bpm_adjustment_payload(projection),
        "source_comparison": None,
        "comparisons": [],
        "paired_metrics": _paired_metrics_payload(()),
    }
    if not projection.ok:
        if selected_candidate:
            payload["fallback"] = {"tag": FALLBACK_V2, "reason": projection.reason}
        return payload

    v3_grid = _require_v3_grid(projection.grid)
    projected_grid = v3_grid.to_fitted_timing_grid()
    comparisons = _compare_grid_to_oracles(projected_grid, stored_oracles, frame_count=frame_count, config=config)
    payload.update(
        {
            "projected_segments": _segments_payload(projected_grid),
            "section_count": len(v3_grid.sections),
            "boundary_seam_ms": _boundary_seam_payload(projected_grid),
            "original_vs_projected_boundary_seam_ms": {
                "source": _boundary_seam_payload(source_grid),
                "projected": _boundary_seam_payload(projected_grid),
            },
            "serialization": _serialization_seam_payload(v3_grid),
            "source_comparison": _source_comparison_payload(
                source_grid,
                v3_grid,
                frame_count=frame_count,
                frame_rate_hz=frame_rate_hz,
                input_start_ms=config.coverage_start_ms,
            ),
            "comparisons": comparisons,
            "paired_metrics": _paired_metrics_payload(comparisons),
        }
    )
    return payload


def _selected_candidate_payload(
    v2: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    candidate_family: str = FAMILY_B,
    selected_method: str = SELECTED_CANDIDATE_METHOD,
) -> dict[str, Any]:
    candidate_ok = bool(candidate.get("ok"))
    selected = candidate if candidate_ok else v2
    return {
        "ok": True,
        "method": selected_method,
        "tag": candidate_family if candidate_ok else FALLBACK_V2,
        "used_v2_fallback": not candidate_ok,
        "fallback": None if candidate_ok else candidate.get("fallback"),
        "source_method": candidate_family if candidate_ok else V2_METHOD,
        "grid": selected.get("grid"),
        "projected_segments": selected.get("projected_segments", []),
        "section_count": selected.get("section_count", 0),
        "boundary_seam_ms": selected.get("boundary_seam_ms"),
        "source_comparison": candidate.get("source_comparison") if candidate_ok else None,
        "comparisons": selected.get("comparisons", []),
        "paired_metrics": selected.get("paired_metrics", _paired_metrics_payload(())),
    }


def _compare_grid_to_oracles(
    predicted_grid: FittedTimingGrid,
    stored_oracles: Sequence[_StoredOraclePayload],
    *,
    frame_count: int,
    config: TimingV3ProjectionEvalConfig,
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for oracle in stored_oracles:
        if oracle.oracle_grid is None:
            comparisons.append(
                {
                    "ok": False,
                    "baseline_comparison_index": oracle.baseline_comparison_index,
                    "beatmap_path": oracle.beatmap_path,
                    "oracle_segments": oracle.oracle_segments_payload,
                    "metrics": None,
                    "drift_metrics": None,
                    "error_type": oracle.error_type or "ValueError",
                    "error": oracle.error or "stored oracle comparison is malformed",
                }
            )
            continue
        try:
            comparison = compare_timing_grids(
                predicted_grid,
                oracle.oracle_grid,
                frame_count=frame_count,
            )
            raw_drift_comparison = compare_timing_grid_drift(
                predicted_grid,
                oracle.oracle_grid,
                frame_count=frame_count,
            )
            alias_drift_comparison = compare_timing_grid_drift(
                canonicalize_timing_grid(predicted_grid, canonicalization=config.canonicalization),
                canonicalize_timing_grid(oracle.oracle_grid, canonicalization=config.canonicalization),
                frame_count=frame_count,
            )
            raw_drift_metrics = _drift_metrics_payload(raw_drift_comparison)
            alias_drift_metrics = _drift_metrics_payload(alias_drift_comparison)
            comparisons.append(
                {
                    "ok": True,
                    "baseline_comparison_index": oracle.baseline_comparison_index,
                    "beatmap_path": oracle.beatmap_path,
                    "oracle_segments": oracle.oracle_segments_payload,
                    "metrics": {
                        **_json_safe(asdict(comparison)),
                        **{f"raw_{key}": value for key, value in raw_drift_metrics.items()},
                        **{f"alias_{key}": value for key, value in alias_drift_metrics.items()},
                    },
                    "drift_metrics": {
                        "raw": raw_drift_metrics,
                        config.canonicalization: alias_drift_metrics,
                    },
                    "error_type": None,
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep one bad stored oracle local to its comparison.
            comparisons.append(
                {
                    "ok": False,
                    "baseline_comparison_index": oracle.baseline_comparison_index,
                    "beatmap_path": oracle.beatmap_path,
                    "oracle_segments": oracle.oracle_segments_payload,
                    "metrics": None,
                    "drift_metrics": None,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                }
            )
    return comparisons


def _summary_payload(
    *,
    rows: Sequence[_BaselineInputRow],
    results: Sequence[Mapping[str, Any]],
    baseline_jsonl_path: Path,
    output_jsonl_path: Path,
    summary_json_path: Path,
    baseline_sha256: str,
    config: TimingV3ProjectionEvalConfig,
    behavior_provenance: Mapping[str, Any],
    started_at_unix: float,
    processed_count: int,
    skipped_success_count: int,
    skipped_failure_count: int,
    stale_existing_count: int,
    retry_failures: bool,
    checkpoint_every: int,
    progress_every: int,
    workers: int,
    limit: int | None,
) -> dict[str, Any]:
    finished_at_unix = time.time()
    ok_results = [result for result in results if result.get("ok")]
    failed_results = [result for result in results if not result.get("ok")]
    audio_group_metrics = _aggregate_methods_audio_group_metrics(
        results,
        include_family_c=config.include_family_c,
    )
    projection_metrics = _aggregate_projection_metrics(
        results,
        include_family_c=config.include_family_c,
    )
    result_counts = _result_counts(results, include_family_c=config.include_family_c)
    return _json_safe(
        {
            "schema": SUMMARY_SCHEMA,
            "source": {
                "baseline_jsonl_path": baseline_jsonl_path.as_posix(),
                "baseline_sha256": baseline_sha256,
                "baseline_result_schema": BASELINE_RESULT_SCHEMA,
                "baseline_row_count": len(rows),
                "limit": limit,
                "output_jsonl_path": output_jsonl_path.as_posix(),
                "summary_json_path": summary_json_path.as_posix(),
            },
            "projection": _projection_config_payload(config),
            "behavior": behavior_provenance,
            "run": {
                "started_at_unix": started_at_unix,
                "finished_at_unix": finished_at_unix,
                "total_seconds": finished_at_unix - started_at_unix,
                "processed_count": processed_count,
                "skipped_success_count": skipped_success_count,
                "skipped_failure_count": skipped_failure_count,
                "stale_existing_count": stale_existing_count,
                "retry_failures": retry_failures,
                "checkpoint_every": checkpoint_every,
                "progress_every": progress_every,
                "workers": workers,
                "parallel": False,
                "resume_schema": RESUME_SCHEMA,
            },
            "results": {
                "result_count": len(results),
                "successful_audio_count": len(ok_results),
                "failed_audio_count": len(failed_results),
                **result_counts,
            },
            "failures": _failure_counts(results, include_family_c=config.include_family_c),
            "headline": _headline_payload(
                results,
                audio_group_metrics,
                projection_metrics,
                include_family_c=config.include_family_c,
            ),
            "metrics": audio_group_metrics,
            "audio_group_metrics": audio_group_metrics,
            "difficulty_comparison_metrics": _aggregate_methods_difficulty_metrics(
                results,
                include_family_c=config.include_family_c,
            ),
            "projection_metrics": projection_metrics,
            "stratified": {
                field_name: _stratified_dimension(
                    results,
                    field_name=field_name,
                    include_family_c=config.include_family_c,
                )
                for field_name in _STRATIFIED_FIELDS
            },
        }
    )


def _result_counts(
    results: Sequence[Mapping[str, Any]],
    *,
    include_family_c: bool,
) -> dict[str, int]:
    ok_results = [result for result in results if result.get("ok")]
    comparison_eligible_results = [result for result in ok_results if result.get("comparator_available")]
    family_a_failures = sum(1 for result in ok_results if not _method_payload(result, FAMILY_A).get("ok"))
    family_b_failures = sum(1 for result in ok_results if not _method_payload(result, FAMILY_B).get("ok"))
    payload = {
        "projection_evaluable_audio_count": len(ok_results),
        "comparison_eligible_audio_count": len(comparison_eligible_results),
        "comparator_unavailable_audio_count": len(ok_results) - len(comparison_eligible_results),
        "baseline_comparable_audio_count": len(comparison_eligible_results),
        "baseline_unusable_audio_count": len(results) - len(ok_results),
        "family_a_projection_failure_audio_count": family_a_failures,
        "family_b_projection_failure_audio_count": family_b_failures,
        "family_b_fallback_audio_count": sum(
            1
            for result in ok_results
            if isinstance(_method_payload(result, FAMILY_B).get("fallback"), MappingABC)
            and _method_payload(result, FAMILY_B)["fallback"].get("tag") == FALLBACK_V2
        ),
        "paired_comparison_count": sum(
            len(_ok_method_comparisons(result, FAMILY_B))
            for result in ok_results
            if _method_payload(result, FAMILY_B).get("ok")
        ),
        "selected_paired_comparison_count": sum(
            len(_ok_method_comparisons(result, SELECTED_CANDIDATE))
            for result in ok_results
        ),
    }
    if include_family_c:
        for family in (FAMILY_C0, FAMILY_C1):
            failure_count = sum(
                1 for result in ok_results if not _method_payload(result, family).get("ok")
            )
            payload[f"{family}_projection_failure_audio_count"] = failure_count
            payload[f"{family}_fallback_audio_count"] = sum(
                1
                for result in ok_results
                if isinstance(_method_payload(result, family).get("fallback"), MappingABC)
                and _method_payload(result, family)["fallback"].get("tag") == FALLBACK_V2
            )
            payload[f"{family}_paired_comparison_count"] = sum(
                len(_ok_method_comparisons(result, family))
                for result in ok_results
                if _method_payload(result, family).get("ok")
            )
        payload["selected_family_c_paired_comparison_count"] = sum(
            len(_ok_method_comparisons(result, SELECTED_FAMILY_C))
            for result in ok_results
        )
    return payload


def _failure_counts(
    results: Sequence[Mapping[str, Any]],
    *,
    include_family_c: bool,
) -> dict[str, Any]:
    stage_counts: dict[str, int] = {}
    for result in results:
        if result.get("ok"):
            continue
        stage = str(result.get("failure_stage") or "unknown")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    payload = {
        "failed_audio_count": sum(1 for result in results if not result.get("ok")),
        "stage_counts": dict(sorted(stage_counts.items())),
        "family_b_projection_failure_audio_count": sum(
            1 for result in results if result.get("ok") and not _method_payload(result, FAMILY_B).get("ok")
        ),
        "family_b_fallback_audio_count": sum(
            1
            for result in results
            if result.get("ok")
            and isinstance(_method_payload(result, FAMILY_B).get("fallback"), MappingABC)
            and _method_payload(result, FAMILY_B)["fallback"].get("tag") == FALLBACK_V2
        ),
    }
    if include_family_c:
        for family in (FAMILY_C0, FAMILY_C1):
            payload[f"{family}_projection_failure_audio_count"] = sum(
                1
                for result in results
                if result.get("ok") and not _method_payload(result, family).get("ok")
            )
            payload[f"{family}_fallback_audio_count"] = sum(
                1
                for result in results
                if result.get("ok")
                and isinstance(_method_payload(result, family).get("fallback"), MappingABC)
                and _method_payload(result, family)["fallback"].get("tag") == FALLBACK_V2
            )
    return payload


def _headline_payload(
    results: Sequence[Mapping[str, Any]],
    audio_group_metrics: Mapping[str, Any],
    projection_metrics: Mapping[str, Any],
    *,
    include_family_c: bool,
) -> dict[str, Any]:
    family_b_headline = _candidate_headline_payload(
        results,
        audio_group_metrics=audio_group_metrics,
        projection_metrics=projection_metrics,
        candidate_method=FAMILY_B,
    )
    matched_phase = family_b_headline["phase_vs_v2"]
    b_projection = _required_summary_mapping(projection_metrics, FAMILY_B)
    b_failure_count = int(b_projection["projection_failure_count"])
    payload: dict[str, Any] = {
        "denominators": {
            "input_audio_count": len(results),
            "projection_evaluable_audio_count": sum(
                1 for result in results if result.get("ok")
            ),
            "comparison_eligible_audio_count": sum(
                1
                for result in results
                if result.get("ok") and result.get("comparator_available")
            ),
        },
        "candidates": {
            FAMILY_A: _candidate_headline_payload(
                results,
                audio_group_metrics=audio_group_metrics,
                projection_metrics=projection_metrics,
                candidate_method=FAMILY_A,
            ),
            FAMILY_B: family_b_headline,
        },
        "family_b_vs_v2_mean_phase_error_ms_matched": matched_phase,
        "family_b_vs_v2_mean_phase_error_ms_paired_audio_count": matched_phase["paired_audio_count"],
        "family_b_vs_v2_mean_phase_error_ms_v2": matched_phase[V2_METHOD],
        "family_b_vs_v2_mean_phase_error_ms_family_b": matched_phase[FAMILY_B],
        "family_b_over_v2_mean_phase_error_ms_mean_ratio": matched_phase["candidate_over_baseline_mean_ratio"],
        "family_b_over_v2_mean_phase_error_ms_p90_ratio": matched_phase["candidate_over_baseline_p90_ratio"],
        "family_b_projection_failure_rate": family_b_headline["projection_failure_rate"],
        "family_b_projection_failure_audio_count": b_failure_count,
        "family_b_fallback_audio_count": int(b_projection["fallback_count"]),
        "family_b_projected_section_count": b_projection["projected_section_count"],
        "family_b_max_serialization_seam_ms": b_projection["serialization_max_boundary_delta_ms"]["max"],
        "v2_original_max_boundary_seam_ms": _method_seam_stats(results, V2_METHOD)["max"],
        "family_b_projected_max_boundary_seam_ms": _method_seam_stats(results, FAMILY_B)["max"],
        "family_b_max_abs_relative_bpm_adjustment": b_projection["max_abs_relative_bpm_adjustment"]["max"],
        "family_b_duration_weighted_abs_relative_bpm_adjustment": b_projection[
            "duration_weighted_abs_relative_bpm_adjustment"
        ],
        "family_b_raw_abs_endpoint_relative_drift_ms": _method_metric_stats(
            audio_group_metrics,
            FAMILY_B,
            "raw_abs_endpoint_relative_drift_ms",
        ),
        "family_b_alias_abs_endpoint_relative_drift_ms": _method_metric_stats(
            audio_group_metrics,
            FAMILY_B,
            "alias_abs_endpoint_relative_drift_ms",
        ),
    }
    if include_family_c:
        for family in (FAMILY_C0, FAMILY_C1):
            candidate = _candidate_headline_payload(
                results,
                audio_group_metrics=audio_group_metrics,
                projection_metrics=projection_metrics,
                candidate_method=family,
            )
            payload["candidates"][family] = candidate
            payload[f"{family}_vs_v2_mean_phase_error_ms_matched"] = candidate[
                "phase_vs_v2"
            ]
            payload[f"{family}_over_v2_mean_phase_error_ms_mean_ratio"] = candidate[
                "phase_mean_ratio"
            ]
            payload[f"{family}_over_v2_mean_phase_error_ms_p90_ratio"] = candidate[
                "phase_p90_ratio"
            ]
            payload[f"{family}_projection_failure_rate"] = candidate[
                "projection_failure_rate"
            ]
            payload[f"{family}_fallback_rate"] = candidate["fallback_rate"]
            payload[f"{family}_max_serialization_seam_ms"] = candidate[
                "max_serialization_seam_ms"
            ]
            payload[f"{family}_projected_max_boundary_seam_ms"] = candidate[
                "projected_max_boundary_seam_ms"
            ]
            payload[f"{family}_guard_audit"] = candidate["guard_audit"]
    return payload


def _candidate_headline_payload(
    results: Sequence[Mapping[str, Any]],
    *,
    audio_group_metrics: Mapping[str, Any],
    projection_metrics: Mapping[str, Any],
    candidate_method: str,
) -> dict[str, Any]:
    matched_phase = _matched_audio_metric_payload(
        results,
        baseline_method=V2_METHOD,
        candidate_method=candidate_method,
        metric_name="mean_phase_error_ms",
    )
    projection = _required_summary_mapping(projection_metrics, candidate_method)
    projection_evaluable_count = sum(1 for result in results if result.get("ok"))
    comparison_eligible_count = sum(
        1
        for result in results
        if result.get("ok") and result.get("comparator_available")
    )
    projection_failure_count = int(projection["projection_failure_count"])
    fallback_count = int(projection["fallback_count"])
    drift_matched = {
        metric_name: _matched_audio_metric_payload(
            results,
            baseline_method=V2_METHOD,
            candidate_method=candidate_method,
            metric_name=metric_name,
        )
        for metric_name in _DRIFT_GUARD_METRIC_NAMES
    }
    payload: dict[str, Any] = {
        "method": candidate_method,
        "projection_evaluable_audio_count": projection_evaluable_count,
        "comparison_eligible_audio_count": comparison_eligible_count,
        "phase_paired_audio_count": matched_phase["paired_audio_count"],
        "phase_vs_v2": matched_phase,
        "phase_mean_ratio": matched_phase["candidate_over_baseline_mean_ratio"],
        "phase_p90_ratio": matched_phase["candidate_over_baseline_p90_ratio"],
        "projection_success_audio_count": int(projection["projection_success_count"]),
        "projection_failure_audio_count": projection_failure_count,
        "projection_failure_rate": _rate(
            projection_failure_count,
            projection_evaluable_count,
        ),
        "fallback_audio_count": fallback_count,
        "fallback_rate": _rate(fallback_count, projection_evaluable_count),
        "projected_section_count": projection["projected_section_count"],
        "section_count_delta": projection["section_count_delta"],
        "max_serialization_seam_ms": projection[
            "serialization_max_boundary_delta_ms"
        ]["max"],
        "all_successful_serializations_pass": projection[
            "all_successful_serializations_pass"
        ],
        "projected_max_boundary_seam_ms": projection[
            "projected_boundary_seam_max_ms"
        ]["max"],
        "max_abs_relative_bpm_adjustment": projection[
            "max_abs_relative_bpm_adjustment"
        ]["max"],
        "duration_weighted_abs_relative_bpm_adjustment": projection[
            "duration_weighted_abs_relative_bpm_adjustment"
        ],
        "projection_seconds": projection["projection_seconds"],
        "source_comparison": projection["source_comparison"],
        "all_successful_source_comparisons_present": (
            int(projection["source_comparison"]["audio_count"])
            == int(projection["projection_success_count"])
        ),
        "drift_matched_vs_v2": drift_matched,
        "joint_diagnostics": projection.get("joint_diagnostics"),
        "oracle_audio_metrics": {
            metric_name: _method_metric_stats(
                audio_group_metrics,
                candidate_method,
                metric_name,
            )
            for metric_name in (
                "mean_phase_error_ms",
                "alias_abs_endpoint_relative_drift_ms",
                "alias_max_abs_prefix_relative_drift_ms",
                "alias_abs_drift_slope_ms_per_minute",
            )
        },
    }
    payload["guard_audit"] = _candidate_guard_audit(payload)
    return payload


def _candidate_guard_audit(candidate: Mapping[str, Any]) -> dict[str, Any]:
    drift_matched = _required_summary_mapping(candidate, "drift_matched_vs_v2")
    drift_ratios: dict[str, Any] = {}
    drift_passes: list[bool | None] = []
    for metric_name in _DRIFT_GUARD_METRIC_NAMES:
        matched = _required_summary_mapping(drift_matched, metric_name)
        mean_ratio = _float_or_none(matched.get("candidate_over_baseline_mean_ratio"))
        p90_ratio = _float_or_none(matched.get("candidate_over_baseline_p90_ratio"))
        mean_pass = None if mean_ratio is None else mean_ratio <= 1.25
        p90_pass = None if p90_ratio is None else p90_ratio <= 1.25
        drift_ratios[metric_name] = {
            "mean_ratio": mean_ratio,
            "p90_ratio": p90_ratio,
            "mean_pass": mean_pass,
            "p90_pass": p90_pass,
        }
        drift_passes.extend((mean_pass, p90_pass))

    section_delta = candidate.get("section_count_delta")
    section_delta_max = (
        _float_or_none(section_delta.get("max"))
        if isinstance(section_delta, MappingABC)
        else None
    )
    joint = candidate.get("joint_diagnostics")
    joint_mapping = joint if isinstance(joint, MappingABC) else None
    joint_solver_max = (
        _nested_float(joint_mapping, ("solver_normalized_residual", "max"))
        if joint_mapping is not None
        else None
    )
    checks: dict[str, bool | None] = {
        "phase_mean_ratio_le_1_10": _le_or_none(candidate.get("phase_mean_ratio"), 1.10),
        "phase_p90_ratio_le_1_15": _le_or_none(candidate.get("phase_p90_ratio"), 1.15),
        "fallback_rate_le_0_05": _le_or_none(candidate.get("fallback_rate"), 0.05),
        "serialized_seam_le_5_ms": _le_or_none(
            candidate.get("max_serialization_seam_ms"),
            5.0,
        ),
        "projected_boundary_seam_le_5_ms": _le_or_none(
            candidate.get("projected_max_boundary_seam_ms"),
            5.0,
        ),
        "bpm_adjustment_le_0_05": _le_or_none(
            candidate.get("max_abs_relative_bpm_adjustment"),
            JOINT_MAX_RELATIVE_BPM_ADJUSTMENT,
        ),
        "section_count_not_increased": (
            None if section_delta_max is None else section_delta_max <= 0.0
        ),
        "all_solver_residuals_pass": (
            bool(joint_mapping.get("all_successful_solver_residuals_pass"))
            and joint_solver_max is not None
            and joint_solver_max <= JOINT_SOLVER_NORMALIZED_RESIDUAL_LIMIT
            if joint_mapping is not None
            else None
        ),
        "all_searches_converged": (
            joint_mapping.get("all_successful_searches_converged")
            if joint_mapping is not None
            else None
        ),
        "all_anchor_displacements_pass": (
            joint_mapping.get("all_successful_anchor_displacements_pass")
            if joint_mapping is not None
            else None
        ),
        "all_successful_fingerprints_present": (
            joint_mapping.get("all_successful_fingerprints_present")
            if joint_mapping is not None
            else None
        ),
        "all_serializations_pass": bool(
            candidate.get("all_successful_serializations_pass")
        ),
        "all_source_comparisons_present": bool(
            candidate.get("all_successful_source_comparisons_present")
        ),
    }
    checks["all_drift_mean_and_p90_ratios_le_1_25"] = (
        None
        if not drift_passes or any(value is None for value in drift_passes)
        else all(bool(value) for value in drift_passes)
    )
    mandatory = [
        checks["phase_mean_ratio_le_1_10"],
        checks["phase_p90_ratio_le_1_15"],
        checks["fallback_rate_le_0_05"],
        checks["serialized_seam_le_5_ms"],
        checks["projected_boundary_seam_le_5_ms"],
        checks["bpm_adjustment_le_0_05"],
        checks["section_count_not_increased"],
        checks["all_serializations_pass"],
        checks["all_source_comparisons_present"],
        checks["all_drift_mean_and_p90_ratios_le_1_25"],
    ]
    if joint_mapping is not None:
        mandatory.extend(
            (
                checks["all_solver_residuals_pass"],
                checks["all_searches_converged"],
                checks["all_anchor_displacements_pass"],
                checks["all_successful_fingerprints_present"],
            )
        )
    return {
        "thresholds": {
            "phase_mean_ratio_max": 1.10,
            "phase_p90_ratio_max": 1.15,
            "fallback_rate_max": 0.05,
            "serialized_seam_ms_max": 5.0,
            "drift_mean_and_p90_ratio_max": 1.25,
            "solver_normalized_residual_max": JOINT_SOLVER_NORMALIZED_RESIDUAL_LIMIT,
            "anchor_displacement_adjacent_local_beats_max": (
                JOINT_ADJACENT_BEAT_DISPLACEMENT_LIMIT
            ),
        },
        "checks": checks,
        "drift": drift_ratios,
        "all_available_mandatory_checks_pass": (
            None
            if any(value is None for value in mandatory)
            else all(bool(value) for value in mandatory)
        ),
    }


def _aggregate_methods_audio_group_metrics(
    results: Sequence[Mapping[str, Any]],
    *,
    include_family_c: bool,
) -> dict[str, Any]:
    method_specs = _method_result_specs(include_family_c=include_family_c)
    payload: dict[str, Any] = {
        "weighting": "one successful audio group; per-audio value is the median across paired stored oracle comparisons",
        **{
            summary_method: _aggregate_method_audio_group_metrics(results, result_key)
            for summary_method, result_key in method_specs
        },
        "matched_audio": {
            "family_b_vs_v2_mean_phase_error_ms": _matched_audio_metric_payload(
                results,
                baseline_method=V2_METHOD,
                candidate_method=FAMILY_B,
                metric_name="mean_phase_error_ms",
            ),
            "family_b_vs_v2_max_phase_error_ms": _matched_audio_metric_payload(
                results,
                baseline_method=V2_METHOD,
                candidate_method=FAMILY_B,
                metric_name="max_phase_error_ms",
            ),
        },
    }
    payload["matched_vs_v2"] = {
        summary_method: {
            metric_name: _matched_audio_metric_payload(
                results,
                baseline_method=V2_METHOD,
                candidate_method=result_key,
                metric_name=metric_name,
            )
            for metric_name in _METRIC_NAMES
        }
        for summary_method, result_key in method_specs
        if result_key != V2_METHOD
    }
    return payload


def _method_result_specs(*, include_family_c: bool) -> tuple[tuple[str, str], ...]:
    base = (
        (V2_METHOD, V2_METHOD),
        (FAMILY_A, FAMILY_A),
        (FAMILY_B, FAMILY_B),
        (SELECTED_CANDIDATE_METHOD, SELECTED_CANDIDATE),
    )
    if not include_family_c:
        return base
    return base + (
        (FAMILY_C0, FAMILY_C0),
        (FAMILY_C1, FAMILY_C1),
        (SELECTED_FAMILY_C_METHOD, SELECTED_FAMILY_C),
    )


def _aggregate_method_audio_group_metrics(
    results: Sequence[Mapping[str, Any]],
    method: str,
) -> dict[str, Any]:
    return {
        metric_name: _stats(_audio_group_metric_values(results, method, metric_name))
        for metric_name in _METRIC_NAMES
    }


def _aggregate_methods_difficulty_metrics(
    results: Sequence[Mapping[str, Any]],
    *,
    include_family_c: bool,
) -> dict[str, Any]:
    return {
        "weighting": "one successful stored oracle comparison",
        **{
            summary_method: _aggregate_method_difficulty_metrics(results, result_key)
            for summary_method, result_key in _method_result_specs(
                include_family_c=include_family_c
            )
        },
    }


def _aggregate_method_difficulty_metrics(
    results: Sequence[Mapping[str, Any]],
    method: str,
) -> dict[str, Any]:
    comparisons = [
        comparison
        for result in results
        for comparison in _ok_method_comparisons(result, method)
    ]
    return {
        metric_name: _stats(
            [
                _float_or_none(metrics.get(metric_name))
                for comparison in comparisons
                if isinstance((metrics := comparison.get("metrics")), MappingABC)
            ]
        )
        for metric_name in _METRIC_NAMES
    }


def _aggregate_projection_metrics(
    results: Sequence[Mapping[str, Any]],
    *,
    include_family_c: bool,
) -> dict[str, Any]:
    return {
        FAMILY_A: _aggregate_projection_method_metrics(results, FAMILY_A),
        FAMILY_B: _aggregate_projection_method_metrics(results, FAMILY_B),
        **(
            {
                FAMILY_C0: _aggregate_projection_method_metrics(results, FAMILY_C0),
                FAMILY_C1: _aggregate_projection_method_metrics(results, FAMILY_C1),
            }
            if include_family_c
            else {}
        ),
    }


def _aggregate_projection_method_metrics(
    results: Sequence[Mapping[str, Any]],
    method: str,
) -> dict[str, Any]:
    method_payloads = [
        _method_payload(result, method)
        for result in results
        if result.get("ok") and isinstance(_method_payload(result, method), MappingABC)
    ]
    ok_payloads = [payload for payload in method_payloads if payload.get("ok")]
    bpm_adjustments = [
        adjustment
        for payload in ok_payloads
        if isinstance((adjustment := payload.get("bpm_adjustment")), MappingABC)
    ]
    failure_reason_counts: dict[str, int] = {}
    for payload in method_payloads:
        if payload.get("ok"):
            continue
        reason = _nested_value(payload, ("projection", "reason"))
        reason_key = str(reason or "unknown")
        failure_reason_counts[reason_key] = failure_reason_counts.get(reason_key, 0) + 1
    return {
        "audio_count": len(method_payloads),
        "projection_success_count": len(ok_payloads),
        "projection_failure_count": len(method_payloads) - len(ok_payloads),
        "fallback_count": sum(
            1
            for payload in method_payloads
            if isinstance(payload.get("fallback"), MappingABC)
            and payload["fallback"].get("tag") == FALLBACK_V2
        ),
        "failure_reason_counts": dict(sorted(failure_reason_counts.items())),
        "projected_section_count": _stats(
            [_float_or_none(payload.get("section_count")) for payload in ok_payloads]
        ),
        "source_section_count": _stats(
            [
                _float_or_none(_method_payload(result, V2_METHOD).get("section_count"))
                for result in results
                if result.get("ok")
            ]
        ),
        "section_count_delta": _stats(
            [
                _method_section_count_delta(result, method)
                for result in results
                if result.get("ok") and _method_payload(result, method).get("ok")
            ]
        ),
        "serialization_max_boundary_delta_ms": _stats(
            [
                _nested_float(payload, ("serialization", "max_boundary_delta_ms"))
                for payload in ok_payloads
            ]
        ),
        "serialization_ok_count": sum(
            bool(_nested_value(payload, ("serialization", "ok")))
            for payload in ok_payloads
        ),
        "all_successful_serializations_pass": all(
            bool(_nested_value(payload, ("serialization", "ok")))
            for payload in ok_payloads
        ),
        "source_boundary_seam_max_ms": _method_seam_stats(results, V2_METHOD),
        "projected_boundary_seam_max_ms": _method_seam_stats(results, method),
        "max_abs_relative_bpm_adjustment": _stats(
            [_float_or_none(adjustment.get("max_abs_relative")) for adjustment in bpm_adjustments]
        ),
        "mean_abs_relative_bpm_adjustment": _stats(
            [_float_or_none(adjustment.get("mean_abs_relative")) for adjustment in bpm_adjustments]
        ),
        "duration_weighted_abs_relative_bpm_adjustment": _duration_weighted_adjustment(
            bpm_adjustments
        ),
        "projection_seconds": _stats(
            [_float_or_none(payload.get("projection_seconds")) for payload in method_payloads]
        ),
        "source_comparison": _aggregate_source_comparison_metrics(ok_payloads),
        "joint_diagnostics": _aggregate_joint_projection_diagnostics(ok_payloads),
    }


def _aggregate_source_comparison_metrics(
    method_payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    comparisons = [
        comparison
        for payload in method_payloads
        if isinstance((comparison := payload.get("source_comparison")), MappingABC)
        and comparison.get("schema") == SOURCE_PROJECTION_COMPARISON_SCHEMA
    ]
    return {
        "audio_count": len(comparisons),
        **{
            metric_name: _stats(
                [_float_or_none(comparison.get(metric_name)) for comparison in comparisons]
            )
            for metric_name in _SOURCE_METRIC_NAMES
        },
    }


def _aggregate_joint_projection_diagnostics(
    method_payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    diagnostics = [
        diagnostic
        for payload in method_payloads
        if isinstance(
            (diagnostic := _nested_value(payload, ("projection", "diagnostics"))),
            MappingABC,
        )
        and "source_surrogate_rms_beats" in diagnostic
    ]
    if not diagnostics:
        return None

    solver_payloads = [
        solver
        for diagnostic in diagnostics
        if isinstance((solver := diagnostic.get("solver")), MappingABC)
    ]
    maximum_anchor_displacements_ms = [
        max(
            (
                abs(value)
                for value in (
                    _float_or_none(item)
                    for item in diagnostic.get("source_anchor_displacements_ms", [])
                )
                if value is not None
            ),
            default=0.0,
        )
        for diagnostic in diagnostics
    ]
    total_abs_beat_count_changes = [
        float(
            sum(
                abs(final - initial)
                for initial, final in zip(
                    _int_sequence(diagnostic.get("initial_beat_counts")),
                    _int_sequence(diagnostic.get("final_beat_counts")),
                )
            )
        )
        for diagnostic in diagnostics
    ]
    search_attempt_counts = [
        float(len(_mapping_sequence(diagnostic.get("search_attempts"))))
        for diagnostic in diagnostics
    ]
    accepted_change_counts = [
        float(
            sum(
                bool(attempt.get("accepted_change"))
                for attempt in _mapping_sequence(diagnostic.get("search_attempts"))
            )
        )
        for diagnostic in diagnostics
    ]
    boundary_diagnostics = [
        boundary
        for diagnostic in diagnostics
        for boundary in _mapping_sequence(diagnostic.get("boundary_diagnostics"))
    ]
    mathematical_fingerprints = [
        value
        for diagnostic in diagnostics
        if isinstance((value := diagnostic.get("mathematical_grid_fingerprint")), str)
        and value
    ]
    integer_fingerprints = [
        value
        for diagnostic in diagnostics
        if isinstance((value := diagnostic.get("integer_search_fingerprint")), str)
        and value
    ]
    replay_fingerprints = [
        value
        for diagnostic in diagnostics
        if isinstance((value := diagnostic.get("replay_fingerprint")), str) and value
    ]
    solver_passes = [bool(solver.get("passed")) for solver in solver_payloads]
    anchor_values = [
        _float_or_none(
            diagnostic.get("maximum_anchor_displacement_adjacent_local_beats")
        )
        for diagnostic in diagnostics
    ]
    finite_anchor_values = [value for value in anchor_values if value is not None]
    fingerprints_complete_count = sum(
        _valid_sha256(diagnostic.get("mathematical_grid_fingerprint"))
        and _valid_sha256(diagnostic.get("integer_search_fingerprint"))
        and _valid_sha256(diagnostic.get("replay_fingerprint"))
        for diagnostic in diagnostics
    )
    return {
        "audio_count": len(diagnostics),
        "objective": _stats(
            [_float_or_none(diagnostic.get("objective")) for diagnostic in diagnostics]
        ),
        "source_surrogate_rms_beats": _stats(
            [
                _float_or_none(diagnostic.get("source_surrogate_rms_beats"))
                for diagnostic in diagnostics
            ]
        ),
        "maximum_anchor_displacement_ms": _stats(maximum_anchor_displacements_ms),
        "maximum_anchor_displacement_adjacent_local_beats": _stats(anchor_values),
        "maximum_relative_bpm_adjustment": _stats(
            [
                _float_or_none(diagnostic.get("maximum_relative_bpm_adjustment"))
                for diagnostic in diagnostics
            ]
        ),
        "changed_count": _stats(
            [_float_or_none(diagnostic.get("changed_count")) for diagnostic in diagnostics]
        ),
        "changed_count_rate": _stats(
            [
                _float_or_none(diagnostic.get("changed_count_rate"))
                for diagnostic in diagnostics
            ]
        ),
        "total_abs_beat_count_change": _stats(total_abs_beat_count_changes),
        "sweeps_completed": _stats(
            [
                _float_or_none(diagnostic.get("sweeps_completed"))
                for diagnostic in diagnostics
            ]
        ),
        "search_attempt_count": _stats(search_attempt_counts),
        "accepted_search_change_count": _stats(accepted_change_counts),
        "search_converged_count": sum(
            bool(diagnostic.get("search_converged")) for diagnostic in diagnostics
        ),
        "all_successful_searches_converged": all(
            bool(diagnostic.get("search_converged")) for diagnostic in diagnostics
        ),
        "solver_normalized_residual": _stats(
            [_float_or_none(solver.get("normalized_residual")) for solver in solver_payloads]
        ),
        "solver_passed_count": sum(solver_passes),
        "solver_failed_count": len(solver_passes) - sum(solver_passes),
        "all_successful_solver_residuals_pass": (
            len(solver_payloads) == len(diagnostics) and all(solver_passes)
        ),
        "all_successful_anchor_displacements_pass": (
            len(finite_anchor_values) == len(diagnostics)
            and all(
                value <= JOINT_ADJACENT_BEAT_DISPLACEMENT_LIMIT
                for value in finite_anchor_values
            )
        ),
        "original_residual_ms": _stats(
            [
                _float_or_none(boundary.get("original_residual_ms"))
                for boundary in boundary_diagnostics
            ]
        ),
        "abs_original_residual_ms": _stats(
            [
                abs(value)
                for boundary in boundary_diagnostics
                if (value := _float_or_none(boundary.get("original_residual_ms")))
                is not None
            ]
        ),
        "fingerprints": {
            "mathematical_grid_present_count": len(mathematical_fingerprints),
            "mathematical_grid_distinct_count": len(set(mathematical_fingerprints)),
            "integer_search_present_count": len(integer_fingerprints),
            "integer_search_distinct_count": len(set(integer_fingerprints)),
            "replay_present_count": len(replay_fingerprints),
            "replay_distinct_count": len(set(replay_fingerprints)),
            "complete_triplet_count": fingerprints_complete_count,
        },
        "all_successful_fingerprints_present": fingerprints_complete_count
        == len(diagnostics),
    }


def _stratified_dimension(
    results: Sequence[Mapping[str, Any]],
    *,
    field_name: str,
    include_family_c: bool,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    values_by_key: dict[str, Any] = {}
    for result in results:
        evaluation_strata = result.get("evaluation_strata", {})
        value = (
            evaluation_strata.get(field_name)
            if isinstance(evaluation_strata, MappingABC)
            else None
        )
        key = _stratum_key(value)
        grouped.setdefault(key, []).append(result)
        values_by_key[key] = value

    entries: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: _stratum_sort_key(values_by_key[item])):
        group_results = grouped[key]
        group_metrics = _aggregate_methods_audio_group_metrics(
            group_results,
            include_family_c=include_family_c,
        )
        group_projection = _aggregate_projection_metrics(
            group_results,
            include_family_c=include_family_c,
        )
        entries.append(
            {
                "value": values_by_key[key],
                "audio_count": len(group_results),
                "ok": sum(1 for result in group_results if result.get("ok")),
                "failed": sum(1 for result in group_results if not result.get("ok")),
                "headline": _headline_payload(
                    group_results,
                    group_metrics,
                    group_projection,
                    include_family_c=include_family_c,
                ),
                "metrics": group_metrics,
                "projection_metrics": group_projection,
            }
        )
    return entries


def _audio_group_metric_values(
    results: Sequence[Mapping[str, Any]],
    method: str,
    metric_name: str,
) -> list[float | None]:
    values: list[float | None] = []
    for result in results:
        if not result.get("ok"):
            continue
        comparison_values = [
            _float_or_none(metrics.get(metric_name))
            for comparison in _ok_method_comparisons(result, method)
            if isinstance((metrics := comparison.get("metrics")), MappingABC)
        ]
        finite_values = [value for value in comparison_values if value is not None and math.isfinite(value)]
        if finite_values:
            values.append(float(np.median(np.asarray(finite_values, dtype=np.float64))))
    return values


def _matched_audio_metric_payload(
    results: Sequence[Mapping[str, Any]],
    *,
    baseline_method: str,
    candidate_method: str,
    metric_name: str,
) -> dict[str, Any]:
    pairs = _audio_group_matched_metric_pairs(
        results,
        baseline_method=baseline_method,
        candidate_method=candidate_method,
        metric_name=metric_name,
    )
    baseline_values = [baseline_value for baseline_value, _candidate_value in pairs]
    candidate_values = [candidate_value for _baseline_value, candidate_value in pairs]
    baseline_stats = _stats(baseline_values)
    candidate_stats = _stats(candidate_values)
    return {
        "metric": metric_name,
        "baseline_method": baseline_method,
        "candidate_method": candidate_method,
        "paired_audio_count": len(pairs),
        baseline_method: baseline_stats,
        candidate_method: candidate_stats,
        "candidate_over_baseline_mean_ratio": _ratio(
            _float_or_none(candidate_stats.get("mean")),
            _float_or_none(baseline_stats.get("mean")),
        ),
        "candidate_over_baseline_p90_ratio": _ratio(
            _float_or_none(candidate_stats.get("p90")),
            _float_or_none(baseline_stats.get("p90")),
        ),
    }


def _audio_group_matched_metric_pairs(
    results: Sequence[Mapping[str, Any]],
    *,
    baseline_method: str,
    candidate_method: str,
    metric_name: str,
) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for result in results:
        if not result.get("ok"):
            continue
        baseline_by_index = _comparison_metric_by_index(result, baseline_method, metric_name)
        candidate_by_index = _comparison_metric_by_index(result, candidate_method, metric_name)
        common_indexes = sorted(index for index in baseline_by_index if index in candidate_by_index)
        baseline_values = [baseline_by_index[index] for index in common_indexes]
        candidate_values = [candidate_by_index[index] for index in common_indexes]
        if baseline_values and candidate_values:
            pairs.append(
                (
                    float(np.median(np.asarray(baseline_values, dtype=np.float64))),
                    float(np.median(np.asarray(candidate_values, dtype=np.float64))),
                )
            )
    return pairs


def _comparison_metric_by_index(
    result: Mapping[str, Any],
    method: str,
    metric_name: str,
) -> dict[int, float]:
    values: dict[int, float] = {}
    for comparison in _ok_method_comparisons(result, method):
        index = comparison.get("baseline_comparison_index")
        metrics = comparison.get("metrics")
        value = _float_or_none(metrics.get(metric_name)) if isinstance(metrics, MappingABC) else None
        if isinstance(index, int) and value is not None:
            values[index] = value
    return values


def _paired_metrics_payload(comparisons: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ok_comparisons = [
        comparison
        for comparison in comparisons
        if comparison.get("ok") and isinstance(comparison.get("metrics"), MappingABC)
    ]
    return {
        "comparison_count": len(ok_comparisons),
        **{
            metric_name: _stats(
                [
                    _float_or_none(comparison["metrics"].get(metric_name))
                    for comparison in ok_comparisons
                ]
            )
            for metric_name in _METRIC_NAMES
        },
    }


def _method_payload(result: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    payload = result.get(method)
    return payload if isinstance(payload, MappingABC) else {}


def _ok_method_comparisons(result: Mapping[str, Any], method: str) -> list[Mapping[str, Any]]:
    payload = _method_payload(result, method)
    comparisons = payload.get("comparisons", [])
    if not isinstance(comparisons, SequenceABC) or isinstance(comparisons, (str, bytes)):
        return []
    return [
        comparison
        for comparison in comparisons
        if isinstance(comparison, MappingABC) and comparison.get("ok")
    ]


def _method_metric_stats(metrics: Mapping[str, Any], method: str, metric_name: str) -> Mapping[str, Any]:
    method_payload = metrics.get(method)
    if not isinstance(method_payload, MappingABC):
        return {}
    value = method_payload.get(metric_name)
    return value if isinstance(value, MappingABC) else {}


def _required_summary_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, MappingABC):
        raise ValueError(f"summary payload missing mapping {key!r}")
    return value


def _method_seam_stats(results: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    return _stats(
        [
            _nested_float(_method_payload(result, method), ("boundary_seam_ms", "max_ms"))
            for result in results
            if result.get("ok") and _method_payload(result, method).get("boundary_seam_ms") is not None
        ]
    )


def _method_section_count_delta(result: Mapping[str, Any], method: str) -> float | None:
    source_count = _float_or_none(_method_payload(result, V2_METHOD).get("section_count"))
    projected_count = _float_or_none(_method_payload(result, method).get("section_count"))
    if source_count is None or projected_count is None:
        return None
    return float(projected_count - source_count)


def _duration_weighted_adjustment(adjustments: Sequence[Mapping[str, Any]]) -> float | None:
    total_weighted = 0.0
    total_duration = 0.0
    for adjustment in adjustments:
        value = _float_or_none(adjustment.get("duration_weighted_abs_relative"))
        duration = _float_or_none(adjustment.get("total_source_interval_ms"))
        if value is None or duration is None or duration <= 0.0:
            continue
        total_weighted += value * duration
        total_duration += duration
    if total_duration <= 0.0:
        return None
    return float(total_weighted / total_duration)


def _projection_payload(
    projection: TimingV3ProjectionResult | TimingV3JointProjectionResult,
) -> dict[str, Any]:
    return {
        "method": projection.method,
        "ok": projection.ok,
        "reason": projection.reason,
        "diagnostics": _json_safe(asdict(projection.diagnostics)),
    }


def _bpm_adjustment_payload(
    projection: TimingV3ProjectionResult | TimingV3JointProjectionResult,
) -> dict[str, Any]:
    diagnostics = tuple(projection.diagnostics.boundary_diagnostics)
    adjustments = [abs(float(diagnostic.relative_bpm_adjustment)) for diagnostic in diagnostics]
    durations = [
        max(0.0, float(diagnostic.source_right_anchor_ms - diagnostic.source_left_anchor_ms))
        for diagnostic in diagnostics
    ]
    total_duration = float(sum(durations))
    return {
        "boundary_count": len(diagnostics),
        "max_abs_relative": max(adjustments, default=0.0),
        "mean_abs_relative": float(np.mean(np.asarray(adjustments, dtype=np.float64))) if adjustments else 0.0,
        "p90_abs_relative": float(np.percentile(np.asarray(adjustments, dtype=np.float64), 90.0))
        if adjustments
        else 0.0,
        "duration_weighted_abs_relative": (
            float(sum(value * duration for value, duration in zip(adjustments, durations)) / total_duration)
            if total_duration > 0.0
            else 0.0
        ),
        "total_source_interval_ms": total_duration,
    }


def _source_comparison_payload(
    source_grid: FittedTimingGrid,
    candidate_grid: TimingV3Grid,
    *,
    frame_count: int,
    frame_rate_hz: float,
    input_start_ms: float,
) -> dict[str, Any]:
    comparison = compare_timing_v3_projection_to_source(
        source_grid,
        candidate_grid,
        frame_count=frame_count,
        input_start_ms=input_start_ms,
        frame_hop_ms=1000.0 / frame_rate_hz,
    )
    payload = comparison.to_dict()
    payload.update(
        {
            "abs_initial_signed_phase_error_beats": abs(
                float(comparison.initial_signed_phase_error_beats)
            ),
            "abs_initial_signed_phase_error_ms": abs(
                float(comparison.initial_signed_phase_error_ms)
            ),
            "abs_endpoint_relative_drift_beats": abs(
                float(comparison.endpoint_relative_drift_beats)
            ),
            "abs_endpoint_relative_drift_ms": abs(
                float(comparison.endpoint_relative_drift_ms)
            ),
            "abs_drift_slope_beats_per_minute": abs(
                float(comparison.drift_slope_beats_per_minute)
            ),
            "abs_drift_slope_ms_per_minute": abs(
                float(comparison.drift_slope_ms_per_minute)
            ),
        }
    )
    return payload


def _boundary_seam_payload(grid: FittedTimingGrid) -> dict[str, Any]:
    seams = predicted_boundary_discontinuities_ms(grid)
    return {
        "boundary_count": len(seams),
        "mean_ms": _mean(seams),
        "p90_ms": _percentile(seams, 90.0),
        "max_ms": max(seams, default=0.0),
        "values_ms": list(seams),
    }


def _serialization_seam_payload(grid: TimingV3Grid) -> dict[str, Any]:
    before = tuple(float(value) for value in grid.boundary_times_ms)
    encoded = json.dumps(grid.to_dict(), allow_nan=False, sort_keys=True)
    decoded = json.loads(encoded)
    restored = TimingV3Grid.from_dict(decoded)
    after = tuple(float(value) for value in restored.boundary_times_ms)
    if len(before) != len(after):
        raise ValueError("TimingV3Grid JSON round-trip changed boundary count")
    deltas = [abs(left - right) for left, right in zip(before, after)]
    tolerances = [roundtrip_seam_tolerance_ms(value) for value in before]
    return {
        "boundary_count": len(before),
        "max_boundary_delta_ms": max(deltas, default=0.0),
        "max_allowed_delta_ms": max(tolerances, default=0.0),
        "ok": all(delta <= tolerance for delta, tolerance in zip(deltas, tolerances)),
    }


def _drift_metrics_payload(comparison: object) -> dict[str, Any]:
    payload = _json_safe(asdict(comparison))
    payload["abs_endpoint_relative_drift_ms"] = abs(float(payload["endpoint_relative_drift_ms"]))
    payload["abs_drift_slope_ms_per_minute"] = abs(float(payload["drift_slope_ms_per_minute"]))
    return payload


def _stored_oracle_payloads(payload: Mapping[str, Any]) -> tuple[_StoredOraclePayload, ...]:
    comparisons = payload.get("comparisons")
    if not isinstance(comparisons, SequenceABC) or isinstance(comparisons, (str, bytes)):
        raise ValueError("baseline field 'comparisons' must be a list")

    stored: list[_StoredOraclePayload] = []
    for index, comparison in enumerate(comparisons):
        if not isinstance(comparison, MappingABC):
            stored.append(
                _StoredOraclePayload(
                    baseline_comparison_index=index,
                    beatmap_path=None,
                    oracle_grid=None,
                    oracle_segments_payload=[],
                    error_type="ValueError",
                    error=f"baseline comparison {index} must be a mapping",
                )
            )
            continue
        if not comparison.get("ok"):
            continue
        try:
            oracle_grid = _grid_from_segments_payload(comparison.get("oracle_segments"))
            stored.append(
                _StoredOraclePayload(
                    baseline_comparison_index=index,
                    beatmap_path=_optional_string(comparison.get("beatmap_path"), f"comparisons[{index}].beatmap_path"),
                    oracle_grid=oracle_grid,
                    oracle_segments_payload=_segments_payload(oracle_grid),
                )
            )
        except Exception as exc:  # noqa: BLE001 - one malformed stored oracle must not drop valid siblings.
            stored.append(
                _StoredOraclePayload(
                    baseline_comparison_index=index,
                    beatmap_path=(
                        comparison.get("beatmap_path")
                        if isinstance(comparison.get("beatmap_path"), str)
                        else None
                    ),
                    oracle_grid=None,
                    oracle_segments_payload=_json_safe(comparison.get("oracle_segments", [])),
                    error_type=exc.__class__.__name__,
                    error=str(exc),
                )
            )
    return tuple(stored)


def _require_baseline_result_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != BASELINE_RESULT_SCHEMA:
        raise ValueError(f"baseline row schema must be {BASELINE_RESULT_SCHEMA!r}")
    _audio_key(payload)


def _require_projection_evaluable_baseline_payload(payload: Mapping[str, Any]) -> None:
    ok = payload.get("ok")
    if not isinstance(ok, bool):
        raise ValueError("baseline field 'ok' must be a boolean")
    if ok:
        return

    failure_stage = payload.get("failure_stage")
    if failure_stage != "compare":
        raise ValueError(
            "baseline row is not projection-evaluable unless ok=true or failure_stage='compare'; "
            f"got ok=false failure_stage={failure_stage!r}"
        )


def _grid_from_segments_payload(value: object) -> FittedTimingGrid:
    if not isinstance(value, SequenceABC) or isinstance(value, (str, bytes)):
        raise ValueError("segments payload must be a list")
    segments: list[TimingSegment] = []
    for index, item in enumerate(value):
        if not isinstance(item, MappingABC):
            raise ValueError(f"segments[{index}] must be a mapping")
        beat_length_ms = item.get("beat_length_ms")
        if beat_length_ms is None and item.get("bpm") is not None:
            beat_length_ms = 60000.0 / _positive_float(item.get("bpm"), f"segments[{index}].bpm")
        segments.append(
            TimingSegment(
                offset_ms=_finite_float(item.get("offset_ms"), f"segments[{index}].offset_ms"),
                beat_length_ms=_positive_float(beat_length_ms, f"segments[{index}].beat_length_ms"),
                meter=_positive_int(item.get("meter", 4), f"segments[{index}].meter"),
            )
        )
    return FittedTimingGrid(tuple(segments))


def _v2_grid_payload(grid: FittedTimingGrid) -> dict[str, Any]:
    return {
        "schema": _GRID_V2_SEGMENTS_SCHEMA,
        "segments": _segments_payload(grid),
    }


def _segments_payload(grid: FittedTimingGrid) -> list[dict[str, Any]]:
    return [
        {
            "offset_ms": float(segment.offset_ms),
            "beat_length_ms": float(segment.beat_length_ms),
            "bpm": float(segment.local_bpm),
            "meter": int(segment.meter),
        }
        for segment in grid.segments
    ]


def _read_baseline_input_rows(path: Path, *, limit: int | None) -> list[_BaselineInputRow]:
    rows: list[_BaselineInputRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload: dict[str, Any] | None
            parse_error: str | None = None
            parse_error_type: str | None = None
            try:
                decoded = json.loads(line)
                if not isinstance(decoded, dict):
                    payload = None
                    parse_error = "baseline row must be a JSON object"
                    parse_error_type = "ValueError"
                    row_content_sha256 = _raw_content_sha256(line)
                else:
                    payload = decoded
                    row_content_sha256 = _stable_json_sha256(payload)
            except json.JSONDecodeError as exc:
                payload = None
                parse_error = str(exc)
                parse_error_type = "JSONDecodeError"
                row_content_sha256 = _raw_content_sha256(line)
            rows.append(
                _BaselineInputRow(
                    row_index=len(rows),
                    line_number=line_number,
                    raw_line=line,
                    row_content_sha256=row_content_sha256,
                    payload=payload,
                    parse_error=parse_error,
                    parse_error_type=parse_error_type,
                )
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _matching_existing_results(
    path: Path,
    *,
    expected_fingerprints: set[str],
) -> tuple[dict[str, dict[str, Any]], int]:
    matching: dict[str, dict[str, Any]] = {}
    stale_count = 0
    for result in _read_result_jsonl(path):
        resume = result.get("resume")
        fingerprint = resume.get("fingerprint") if isinstance(resume, MappingABC) else None
        if (
            result.get("schema") == RESULT_SCHEMA
            and isinstance(resume, MappingABC)
            and resume.get("schema") == RESUME_SCHEMA
            and isinstance(fingerprint, str)
            and fingerprint in expected_fingerprints
        ):
            matching[fingerprint] = result
            continue
        stale_count += 1
    return matching, stale_count


def _read_result_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    results: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            results.append(payload)
    return results


def _ordered_results(
    rows: Sequence[_BaselineInputRow],
    results_by_fingerprint: Mapping[str, Mapping[str, Any]],
    resume_by_fingerprint: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    for row in rows:
        fingerprint = next(
            resume["fingerprint"]
            for resume in resume_by_fingerprint.values()
            if resume["components"]["baseline"]["row_index"] == row.row_index
        )
        result = results_by_fingerprint.get(fingerprint)
        if result is not None:
            ordered.append(dict(result))
    return ordered


def _resume_payload(
    row: _BaselineInputRow,
    *,
    baseline_jsonl_path: Path,
    baseline_sha256: str,
    config: TimingV3ProjectionEvalConfig,
    behavior_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    payload = row.payload
    baseline_components = {
        "path": baseline_jsonl_path.as_posix(),
        "sha256": baseline_sha256,
        "row_index": row.row_index,
        "line_number": row.line_number,
        "row_content_sha256": row.row_content_sha256,
        "row_schema": payload.get("schema") if isinstance(payload, MappingABC) else None,
        "audio_key": payload.get("audio_key") if isinstance(payload, MappingABC) else None,
        "baseline_resume_fingerprint": _nested_value(payload, ("resume", "fingerprint"))
        if isinstance(payload, MappingABC)
        else None,
    }
    components = {
        "result_schema": RESULT_SCHEMA,
        "baseline": baseline_components,
        "projection": _projection_config_payload(config),
        "behavior": behavior_provenance,
    }
    return {
        "schema": RESUME_SCHEMA,
        "fingerprint": _stable_json_sha256(components),
        "components": components,
    }


def _projection_config_payload(config: TimingV3ProjectionEvalConfig) -> dict[str, Any]:
    families: dict[str, Any] = {
        FAMILY_A: PROJECTION_METHOD_PRESERVE_BPM,
        FAMILY_B: PROJECTION_METHOD_PRESERVE_ANCHORS,
        SELECTED_CANDIDATE_METHOD: f"{PROJECTION_METHOD_PRESERVE_ANCHORS}_or_{FALLBACK_V2}",
    }
    if config.include_family_c:
        families.update(
            {
                FAMILY_C0: PROJECTION_METHOD_JOINT_FIXED_COUNTS,
                FAMILY_C1: PROJECTION_METHOD_JOINT_NEARBY_COUNTS,
                SELECTED_FAMILY_C_METHOD: (
                    f"{PROJECTION_METHOD_JOINT_NEARBY_COUNTS}_or_{FALLBACK_V2}"
                ),
            }
        )
    payload = {
        "schema": _PROJECTION_CONFIG_SCHEMA,
        "min_bpm": float(config.min_bpm),
        "max_bpm": float(config.max_bpm),
        "max_relative_bpm_adjustment": float(config.max_relative_bpm_adjustment),
        "coverage_start_ms": float(config.coverage_start_ms),
        "canonicalization": config.canonicalization,
        "include_family_c": config.include_family_c,
        "families": families,
        "family_c_frozen_guards": (
            {
                "min_bpm": JOINT_MIN_BPM,
                "max_bpm": JOINT_MAX_BPM,
                "max_relative_bpm_adjustment": JOINT_MAX_RELATIVE_BPM_ADJUSTMENT,
                "adjacent_beat_displacement_limit": (
                    JOINT_ADJACENT_BEAT_DISPLACEMENT_LIMIT
                ),
                "solver_normalized_residual_limit": (
                    JOINT_SOLVER_NORMALIZED_RESIDUAL_LIMIT
                ),
            }
            if config.include_family_c
            else None
        ),
        "source_projection_comparison_schema": SOURCE_PROJECTION_COMPARISON_SCHEMA,
        "timing_v3_grid_schema": TIMING_V3_GRID_SCHEMA,
        "timing_v3_grid_version": TIMING_V3_GRID_VERSION,
    }
    payload["fingerprint"] = _stable_json_sha256(payload)
    return payload


def _normalize_projection_config(
    config: TimingV3ProjectionEvalConfig | Mapping[str, Any] | None,
) -> TimingV3ProjectionEvalConfig:
    if config is None:
        config = TimingV3ProjectionEvalConfig()
    elif isinstance(config, MappingABC):
        config_values = dict(config)
        if "enable_family_c" in config_values and "include_family_c" not in config_values:
            config_values["include_family_c"] = config_values.pop("enable_family_c")
        config = TimingV3ProjectionEvalConfig(**config_values)
    if not isinstance(config, TimingV3ProjectionEvalConfig):
        raise ValueError("projection_config must be a TimingV3ProjectionEvalConfig or mapping")
    if not math.isfinite(config.min_bpm) or config.min_bpm <= 0.0:
        raise ValueError(f"min_bpm must be positive and finite, got {config.min_bpm!r}")
    if not math.isfinite(config.max_bpm) or config.max_bpm < config.min_bpm:
        raise ValueError(f"max_bpm must be finite and >= min_bpm, got {config.max_bpm!r}")
    if (
        not math.isfinite(config.max_relative_bpm_adjustment)
        or config.max_relative_bpm_adjustment < 0.0
    ):
        raise ValueError(
            "max_relative_bpm_adjustment must be non-negative and finite, "
            f"got {config.max_relative_bpm_adjustment!r}",
        )
    if not math.isfinite(config.coverage_start_ms):
        raise ValueError(f"coverage_start_ms must be finite, got {config.coverage_start_ms!r}")
    if config.canonicalization != TIMING_CANONICALIZATION_BPM_80_160:
        raise ValueError(
            f"canonicalization must be {TIMING_CANONICALIZATION_BPM_80_160!r}, "
            f"got {config.canonicalization!r}",
        )
    if not isinstance(config.include_family_c, bool):
        raise ValueError(
            f"include_family_c must be a bool, got {config.include_family_c!r}"
        )
    return config


def _reject_equal_paths(
    *,
    baseline_jsonl_path: Path,
    output_jsonl_path: Path,
    summary_json_path: Path,
) -> None:
    paths = {
        "baseline_jsonl_path": baseline_jsonl_path.expanduser().resolve(strict=False),
        "output_jsonl_path": output_jsonl_path.expanduser().resolve(strict=False),
        "summary_json_path": summary_json_path.expanduser().resolve(strict=False),
    }
    names = tuple(paths)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1:]:
            if paths[left_name] == paths[right_name]:
                raise ValueError(
                    f"{left_name} and {right_name} must be different paths: {paths[left_name].as_posix()}"
                )


def _behavior_provenance() -> dict[str, Any]:
    payload = {
        "schema": _BEHAVIOR_PROVENANCE_SCHEMA,
        "python": {
            "implementation": sys.implementation.name,
            "version": sys.version,
            "version_info": list(sys.version_info[:3]),
        },
        "numpy": {
            "version": np.__version__,
        },
        "source_modules": [
            _module_source_identity(module_name)
            for module_name in _BEHAVIOR_SOURCE_MODULES
        ],
    }
    payload["fingerprint"] = _stable_json_sha256(payload)
    return payload


def _module_source_identity(module_name: str) -> dict[str, Any]:
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise ValueError(f"module {module_name!r} has no source file")
    path = Path(module_file).expanduser().resolve(strict=False)
    return {
        "module": module_name,
        "path": path.as_posix(),
        "sha256": _file_sha256(path),
    }


def _failure_result(
    row: _BaselineInputRow,
    *,
    resume: Mapping[str, Any],
    stage: str,
    error_type: str,
    error: str,
    started_at_unix: float,
) -> dict[str, Any]:
    payload = row.payload
    projection_component = _nested_value(resume, ("components", "projection"))
    include_family_c = bool(
        projection_component.get("include_family_c")
        if isinstance(projection_component, MappingABC)
        else False
    )
    return _json_safe(
        {
            "schema": RESULT_SCHEMA,
            "resume": resume,
            "ok": False,
            "projection_evaluable": False,
            "comparator_available": False,
            "comparison_eligible": False,
            "stored_oracle_comparison_count": 0,
            "valid_stored_oracle_comparison_count": 0,
            "baseline_ok": bool(payload.get("ok")) if isinstance(payload, MappingABC) else False,
            "baseline_failure_stage": payload.get("failure_stage") if isinstance(payload, MappingABC) else None,
            "audio_key": payload.get("audio_key") if isinstance(payload, MappingABC) else None,
            "row_index": row.row_index,
            "baseline_row_index": payload.get("row_index") if isinstance(payload, MappingABC) else None,
            "baseline_line_number": row.line_number,
            "source_line_numbers": payload.get("source_line_numbers") if isinstance(payload, MappingABC) else None,
            "evaluation_strata": payload.get("evaluation_strata", {}) if isinstance(payload, MappingABC) else {},
            "audio_path": payload.get("audio_path") if isinstance(payload, MappingABC) else None,
            "beatmap_paths": payload.get("beatmap_paths", []) if isinstance(payload, MappingABC) else [],
            "prediction": payload.get("prediction") if isinstance(payload, MappingABC) else None,
            "baseline_paired_metrics": payload.get("paired_metrics") if isinstance(payload, MappingABC) else None,
            "provenance": resume["components"],
            V2_METHOD: None,
            FAMILY_A: None,
            FAMILY_B: None,
            SELECTED_CANDIDATE: None,
            **(
                {
                    FAMILY_C0: None,
                    FAMILY_C1: None,
                    SELECTED_FAMILY_C: None,
                }
                if include_family_c
                else {}
            ),
            "failure_stage": stage,
            "error_type": error_type,
            "error": error,
            "runtime": _row_runtime_payload(started_at_unix),
        }
    )


def _row_runtime_payload(started_at_unix: float) -> dict[str, float]:
    finished_at_unix = time.time()
    return {
        "started_at_unix": started_at_unix,
        "finished_at_unix": finished_at_unix,
        "evaluation_seconds": finished_at_unix - started_at_unix,
        "total_seconds": finished_at_unix - started_at_unix,
    }


def _stats(values: Sequence[float | None]) -> dict[str, float | int | None]:
    finite = np.asarray([float(value) for value in values if value is not None and math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "min": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "p50": float(np.percentile(finite, 50.0)),
        "p90": float(np.percentile(finite, 90.0)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile)) if values else 0.0


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    if abs(denominator) <= 1e-12:
        return 1.0 if abs(numerator) <= 1e-12 else None
    return float(numerator / denominator)


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _le_or_none(value: object, limit: float) -> bool | None:
    number = _float_or_none(value)
    return None if number is None else number <= limit


def _mapping_sequence(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, SequenceABC) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, MappingABC)]


def _int_sequence(value: object) -> tuple[int, ...]:
    if not isinstance(value, SequenceABC) or isinstance(value, (str, bytes)):
        return ()
    return tuple(
        int(item)
        for item in value
        if isinstance(item, int) and not isinstance(item, bool)
    )


def _valid_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _nested_float(payload: Mapping[str, Any], path: Sequence[str]) -> float | None:
    value = _nested_value(payload, path)
    return _float_or_none(value)


def _nested_value(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, MappingABC):
            return None
        current = current.get(key)
    return current


def _float_or_none(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive_float(value: object, name: str) -> float:
    number = _finite_float(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, MappingABC):
        raise ValueError(f"baseline field {key!r} must be a mapping")
    return value


def _baseline_unusable_stage(payload: Mapping[str, Any] | None) -> str:
    if isinstance(payload, MappingABC) and not bool(payload.get("ok")):
        stage = payload.get("failure_stage")
        if isinstance(stage, str) and stage in {"cache", "fit", "timeout"}:
            return stage
        return "baseline"
    return "baseline_row"


def _audio_key(payload: Mapping[str, Any]) -> str:
    value = payload.get("audio_key")
    if not isinstance(value, str) or not value:
        raise ValueError("baseline field 'audio_key' must be a non-empty string")
    return value


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string when provided")
    return value


def _require_v3_grid(value: object) -> TimingV3Grid:
    if not isinstance(value, TimingV3Grid):
        raise ValueError("projection grid must be a TimingV3Grid")
    return value


def _stratum_key(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _stratum_sort_key(value: Any) -> tuple[int, str]:
    if value is None:
        return (0, "")
    return (1, _stratum_key(value))


def _write_result_jsonl_atomic(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    with _atomic_text_output(path) as handle:
        for result in results:
            handle.write(json.dumps(_json_safe(result), allow_nan=False, sort_keys=True) + "\n")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    with _atomic_text_output(path) as handle:
        handle.write(
            json.dumps(_json_safe(payload), allow_nan=False, indent=2, sort_keys=True)
            + "\n"
        )


@contextmanager
def _atomic_text_output(path: Path) -> Iterator[TextIO]:
    """Yield a same-directory unique file and atomically replace ``path``.

    ``NamedTemporaryFile`` creates the file with exclusive-create semantics.
    Its unpredictable name cannot truncate a user-supplied baseline path that
    happens to end in ``.tmp``, unlike a deterministic ``path.name + '.tmp'``.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    tmp_path = Path(temporary.name)
    try:
        with temporary as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_content_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, MappingABC):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _print_progress(*, processed_count: int, total_count: int, result: Mapping[str, Any]) -> None:
    status = "ok" if result.get("ok") else f"failed:{result.get('failure_stage')}"
    print(
        f"[timing-v3-projection] processed={processed_count}/{total_count} "
        f"audio_key={result.get('audio_key')} {status}",
        file=sys.stderr,
    )


def _format_summary(summary: Mapping[str, Any]) -> str:
    results = summary["results"]
    headline = summary["headline"]
    text = (
        "Timing v3 projection evaluation: "
        f"{results['projection_evaluable_audio_count']}/{results['result_count']} audio projection-evaluable, "
        f"{results['comparison_eligible_audio_count']} comparison-eligible, "
        f"B fallback={headline['family_b_fallback_audio_count']}, "
        "B/v2 phase mean,p90 ratios="
        f"{_format_optional_float(headline['family_b_over_v2_mean_phase_error_ms_mean_ratio'])}/"
        f"{_format_optional_float(headline['family_b_over_v2_mean_phase_error_ms_p90_ratio'])}, "
        f"v2->B seam max={_format_optional_float(headline['v2_original_max_boundary_seam_ms'])}->"
        f"{_format_optional_float(headline['family_b_projected_max_boundary_seam_ms'])} ms"
    )
    if FAMILY_C1 in headline.get("candidates", {}):
        text += (
            ", C1 fallback="
            f"{results.get('family_c1_fallback_audio_count', 0)}, "
            "C1/v2 phase mean,p90 ratios="
            f"{_format_optional_float(headline.get('family_c1_over_v2_mean_phase_error_ms_mean_ratio'))}/"
            f"{_format_optional_float(headline.get('family_c1_over_v2_mean_phase_error_ms_p90_ratio'))}"
        )
    return text


def _format_optional_float(value: object) -> str:
    number = _float_or_none(value)
    return "n/a" if number is None else f"{number:.6g}"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Timing v3 Family A/B projection, optionally with frozen "
            "Experiment 003 Family C0/C1, from durable cache-backed v2 baseline JSONL."
        ),
    )
    parser.add_argument(
        "--baseline-jsonl",
        "--baseline-results",
        required=True,
        type=Path,
        dest="baseline_jsonl",
        help="Durable v2 baseline result JSONL.",
    )
    parser.add_argument("--output-jsonl", required=True, type=Path, help="Per-audio projection result JSONL path.")
    parser.add_argument("--summary-json", type=Path, default=None, help="Summary JSON path.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--min-bpm", type=float, default=DEFAULT_MIN_BPM)
    parser.add_argument("--max-bpm", type=float, default=DEFAULT_MAX_BPM)
    parser.add_argument("--max-relative-bpm-adjustment", type=float, default=DEFAULT_MAX_RELATIVE_BPM_ADJUSTMENT)
    parser.add_argument("--coverage-start-ms", type=float, default=0.0)
    parser.add_argument(
        "--include-family-c",
        "--enable-family-c",
        action="store_true",
        dest="include_family_c",
        help="Opt in to frozen Experiment 003 Family C0/C1 and C1-or-v2 selection.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    summary = run_timing_v3_projection_evaluation(
        baseline_jsonl_path=args.baseline_jsonl,
        output_jsonl_path=args.output_jsonl,
        summary_json_path=args.summary_json,
        limit=args.limit,
        retry_failures=args.retry_failures,
        progress_every=args.progress_every,
        checkpoint_every=args.checkpoint_every,
        workers=args.workers,
        projection_config=TimingV3ProjectionEvalConfig(
            min_bpm=args.min_bpm,
            max_bpm=args.max_bpm,
            max_relative_bpm_adjustment=args.max_relative_bpm_adjustment,
            coverage_start_ms=args.coverage_start_ms,
            include_family_c=args.include_family_c,
        ),
    )
    if args.json:
        print(json.dumps(summary, allow_nan=False, sort_keys=True))
    else:
        print(_format_summary(summary))
    return 1 if int(summary["results"]["failed_audio_count"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
