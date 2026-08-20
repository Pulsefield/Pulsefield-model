from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Mapping as MappingABC
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from pulsefield_model.timing.evaluation import exp004_protocol
from pulsefield_model.timing.grid_fitting import GridFitter, GridFitterConfig
from pulsefield_model.timing.providers.beatthis import DEFAULT_BEATTHIS_CHECKPOINT
from pulsefield_model.timing.providers.beatthis_cache import (
    DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_CONFIG,
    BeatThisFramePredictionCacheConfig,
    BeatThisFramePredictionCacheError,
    beatthis_frame_prediction_cache_path,
    load_beatthis_frame_prediction_cache,
)
from pulsefield_model.timing.schema import FittedTimingGrid, FrameTimingPrediction
from pulsefield_model.timing.v3.global_constant_jump import (
    CANDIDATE_CONTRACT_VERSION,
    GLOBAL_CONSTANT_JUMP_CONSTANTS,
    GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON_SHA256,
    GLOBAL_CONSTANT_JUMP_VARIANTS,
    VARIANT_CJ3,
    extract_global_constant_jump_candidates,
    iter_global_constant_jump_variants,
)
from pulsefield_model.timing.v3.schema import TimingV3Grid, roundtrip_seam_tolerance_ms


RESULT_SCHEMA = "pulsefield_model.timing_v3_exp004_projection_result_v1"
SUMMARY_SCHEMA = "pulsefield_model.timing_v3_exp004_projection_summary_v1"
RESUME_SCHEMA = "pulsefield_model.timing_v3_exp004_projection_resume_v1"
WEAK_SUMMARY_SCHEMA = "pulsefield_model.timing_v3_exp004_weak_summary_v1"
WEAK_STAGE_GATES_SCHEMA = "pulsefield_model.timing_v3_exp004_weak_stage_gates_v1"
WEAK_PROTOCOL_BINDING_SCHEMA = "pulsefield_model.timing_v3_exp004_weak_protocol_binding_v1"
IDENTITY_ROW_SCHEMA = exp004_protocol.IDENTITY_ROW_SCHEMA
SELECTION_MANIFEST_SCHEMA = exp004_protocol.EXECUTION_SELECTION_MANIFEST_SCHEMA
STAGE_AUDIO_COUNTS = exp004_protocol.STAGE_AUDIO_COUNTS
PRIOR_STAGE = exp004_protocol.PRIOR_STAGE

PER_AUDIO_TIMEOUT_SECONDS = 180.0
MAX_V3_SECTION_COUNT = 20
FALLBACK_POLICY = "selected_CJ3_or_current_v2_fallback_v1"
UNDEFINED_PROJECTION_ONLY = "undefined_projection_only"
_EXPERIMENT = "timing_v3_experiment_004"
INTEGRATION_BLOCKERS: tuple[str, ...] = ()
MAX_PROJECTION_WORKERS = 64
PROCESS_START_METHOD = "spawn"

_PRIOR_WEAK_DECISION: Mapping[str, tuple[str, str]] = {
    "repair80": ("debug_only", "proceed_to_holdout100"),
    "holdout100": ("pass", "freeze_source_config_and_materialize_broad500"),
    "broad500": ("pass", "freeze_source_config_and_run_full5050"),
}
_REQUIRED_WEAK_STAGE_GATES = frozenset(
    {
        "projection_hard_guards",
        "phase_denominator_available",
        "all_mean_phase_ratio",
        "all_p90_phase_ratio",
        "pure_CJ3_phase_coverage",
        "stable_mean_phase_ratio",
        "stable_p90_phase_ratio",
        "jump_mean_phase_ratio",
        "jump_endpoint_drift_mean_ratio",
        "jump_combined_mean_or_drift",
        "jump_ablation_CJ3_vs_CJ1",
        "jump_ablation_CJ3_vs_CJ2",
        "long_max_prefix_drift_mean_ratio",
        "long_max_prefix_drift_p90_ratio",
        "fallback_rate",
        "no_path_plus_candidate_extraction_failure_rate",
        "runtime_p90_seconds",
        "quota_degraded_minimum_denominator",
    }
)

_BEHAVIOR_SOURCE_RELATIVE_PATHS = (
    "docs/research/timing_v3_experiment_004_global_constant_jump.md",
    "docs/research/timing_v3_experiment_004_protocol_clarification_001.md",
    "src/pulsefield_model/timing/__init__.py",
    "src/pulsefield_model/timing/beat_materialization.py",
    "src/pulsefield_model/timing/canonicalization.py",
    "src/pulsefield_model/timing/ramp_detection.py",
    "src/pulsefield_model/timing/schema.py",
    "src/pulsefield_model/timing/v3/schema.py",
    "src/pulsefield_model/timing/providers/beatthis.py",
    "src/pulsefield_model/timing/providers/beatthis_cache.py",
    "src/pulsefield_model/timing/grid_fitting/__init__.py",
    "src/pulsefield_model/timing/grid_fitting/alias.py",
    "src/pulsefield_model/timing/grid_fitting/change_detection.py",
    "src/pulsefield_model/timing/grid_fitting/fitter.py",
    "src/pulsefield_model/timing/grid_fitting/config.py",
    "src/pulsefield_model/timing/grid_fitting/frames.py",
    "src/pulsefield_model/timing/grid_fitting/refinement.py",
    "src/pulsefield_model/timing/grid_fitting/scoring.py",
    "src/pulsefield_model/timing/grid_fitting/segment_fit.py",
    "src/pulsefield_model/timing/grid_fitting/segments.py",
    "src/pulsefield_model/timing/grid_fitting/splitting.py",
    "src/pulsefield_model/timing/grid_fitting/types.py",
    "src/pulsefield_model/timing/v3/global_constant_jump.py",
    "src/pulsefield_model/timing/evaluation/exp004_metrics.py",
    "src/pulsefield_model/timing/evaluation/exp004_protocol.py",
    "src/pulsefield_model/timing/evaluation/exp004_splits.py",
    "src/pulsefield_model/timing/evaluation/exp004_weak_evidence.py",
    "src/pulsefield_model/timing/evaluation/exp004_runner.py",
)


_IdentityRow = exp004_protocol.Exp004IdentityRow
_SelectionEntry = exp004_protocol.Exp004SelectionEntry


@dataclass(frozen=True)
class _CacheFileIdentity:
    path: str
    exists: bool
    size_bytes: int | None
    mtime_ns: int | None
    inode: int | None
    device: int | None
    sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _ProjectionWorkItem:
    identity: _IdentityRow
    entry: _SelectionEntry
    cache_identity: _CacheFileIdentity
    resume: Mapping[str, Any]


_WORKER_CACHE_CONFIG: BeatThisFramePredictionCacheConfig | None = None
_WORKER_RUN_PROVENANCE: Mapping[str, Any] | None = None


class _PerAudioTimeout(TimeoutError):
    pass


class _CacheChangedDuringLoad(RuntimeError):
    pass


def run_exp004_projection(
    *,
    stage: str,
    identity_rows_jsonl_path: str | Path,
    selection_manifest_path: str | Path,
    output_jsonl_path: str | Path,
    summary_json_path: str | Path | None = None,
    prior_stage_summary_path: str | Path | None = None,
    cache_root: str | Path = DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_CONFIG.cache_root,
    cache_version: str = DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_CONFIG.cache_version,
    checkpoint_path: str = DEFAULT_BEATTHIS_CHECKPOINT,
    float16: bool = False,
    shift_ms: float = 0.0,
    frame_rate_hz: float = DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_CONFIG.frame_rate_hz,
    cache_config: BeatThisFramePredictionCacheConfig | None = None,
    retry_failures: bool = False,
    checkpoint_every: int = 1,
    workers: int = 1,
) -> dict[str, Any]:
    """Run leakage-clean Experiment 004 projections from one declared cache per audio.

    This runner deliberately stops before any weak-comparator evaluation.  Its
    only data inputs are compact audio identities, a frozen selection manifest,
    and existing BeatThis frame-prediction caches.
    """

    if stage not in STAGE_AUDIO_COUNTS:
        raise ValueError(f"stage must be one of {tuple(STAGE_AUDIO_COUNTS)!r}, got {stage!r}")
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers < 1
        or workers > MAX_PROJECTION_WORKERS
    ):
        raise ValueError(
            f"workers must be an integer in [1, {MAX_PROJECTION_WORKERS}]"
        )
    if isinstance(checkpoint_every, bool) or not isinstance(checkpoint_every, int) or checkpoint_every < 1:
        raise ValueError("checkpoint_every must be a positive integer")
    if not isinstance(retry_failures, bool):
        raise TypeError("retry_failures must be a bool")
    if not hasattr(signal, "setitimer") or not hasattr(signal, "SIGALRM"):
        raise RuntimeError("Experiment 004 requires POSIX interval-timer support")
    if workers == 1 and threading.current_thread() is not threading.main_thread():
        raise RuntimeError("single-process Experiment 004 execution requires the main thread")

    identity_path = Path(identity_rows_jsonl_path)
    selection_path = Path(selection_manifest_path)
    output_path = Path(output_jsonl_path)
    summary_path = (
        Path(summary_json_path)
        if summary_json_path is not None
        else output_path.with_suffix(output_path.suffix + ".summary.json")
    )
    prior_path = Path(prior_stage_summary_path) if prior_stage_summary_path is not None else None
    lock_path = output_path.with_name(f".{output_path.name}.lock")
    _reject_path_aliases(
        {
            "identity_rows_jsonl_path": identity_path,
            "selection_manifest_path": selection_path,
            "output_jsonl_path": output_path,
            "summary_json_path": summary_path,
            "prior_stage_summary_path": prior_path,
            "run_lock_path": lock_path,
        }
    )

    if cache_config is not None:
        if not isinstance(cache_config, BeatThisFramePredictionCacheConfig):
            raise TypeError("cache_config must be a BeatThisFramePredictionCacheConfig")
        supplied_components = (
            Path(cache_root) != DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_CONFIG.cache_root
            or cache_version != DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_CONFIG.cache_version
            or checkpoint_path != DEFAULT_BEATTHIS_CHECKPOINT
            or float16 is not False
            or shift_ms != 0.0
            or frame_rate_hz != DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_CONFIG.frame_rate_hz
        )
        if supplied_components:
            raise ValueError("cache_config cannot be combined with component cache arguments")
        resolved_cache_config = cache_config
    else:
        resolved_cache_config = BeatThisFramePredictionCacheConfig(
            cache_root=Path(cache_root),
            cache_version=cache_version,
            checkpoint_path=checkpoint_path,
            float16=float16,
            shift_ms=shift_ms,
            frame_rate_hz=frame_rate_hz,
        )
    identities, identity_source = _load_identity_rows(identity_path, expected_stage=stage)
    manifest, entries, selection_source = _load_selection_manifest(
        selection_path,
        expected_stage=stage,
    )
    _reconcile_identities(identities, entries, expected_stage=stage)
    declared_identity_source = _nested(manifest, "source", "identity_jsonl")
    if not isinstance(declared_identity_source, MappingABC) or dict(declared_identity_source) != identity_source:
        raise ValueError(
            "execution selection manifest identity_jsonl provenance does not match "
            "the declared identity input"
        )
    _reject_declared_cache_aliases(
        identities=identities,
        cache_config=resolved_cache_config,
        paths={
            "identity_rows_jsonl_path": identity_path,
            "selection_manifest_path": selection_path,
            "output_jsonl_path": output_path,
            "summary_json_path": summary_path,
            "prior_stage_summary_path": prior_path,
            "run_lock_path": lock_path,
        },
    )

    source_identities = _behavior_source_identities()
    behavior_payload = _behavior_payload(source_identities)
    behavior_fingerprint = _stable_json_sha256(behavior_payload)
    config_payload = _config_payload(resolved_cache_config)
    config_fingerprint = _stable_json_sha256(config_payload)
    prior = _load_and_validate_prior_summary(
        prior_path,
        stage=stage,
        behavior_fingerprint=behavior_fingerprint,
        config_fingerprint=config_fingerprint,
    )
    run_provenance = _run_provenance(
        stage=stage,
        behavior_payload=behavior_payload,
        behavior_fingerprint=behavior_fingerprint,
        config_payload=config_payload,
        config_fingerprint=config_fingerprint,
        identity_source=identity_source,
        selection_source=selection_source,
        prior=prior,
        workers=workers,
    )

    with _exclusive_run_lock(lock_path, run_fingerprint=run_provenance["run_fingerprint"]):
        existing = _load_existing_results(output_path, identities=identities)
        resume_counts = Counter(
            reused_success=0,
            reused_failure=0,
            recomputed_stale=0,
            retried_failure=0,
            processed=0,
        )
        started_at = time.time()
        reused_by_index: dict[int, dict[str, Any]] = {}
        work_items: list[_ProjectionWorkItem] = []
        for identity, entry in zip(identities, entries, strict=True):
            cache_path = beatthis_frame_prediction_cache_path(identity.cache_audio_key, resolved_cache_config)
            cache_identity = _cache_file_identity(cache_path)
            resume = _resume_payload(
                identity=identity,
                entry=entry,
                cache_identity=cache_identity,
                run_provenance=run_provenance,
                identity_source=identity_source,
                selection_source=selection_source,
                prior=prior,
            )
            previous = existing.get(identity.row_index)
            if previous is not None and previous["resume"]["fingerprint"] == resume["fingerprint"]:
                if retry_failures and not bool(previous.get("ok")):
                    resume_counts["retried_failure"] += 1
                else:
                    reused_by_index[identity.row_index] = previous
                    resume_counts[
                        "reused_success" if bool(previous.get("ok")) else "reused_failure"
                    ] += 1
                    continue
            elif previous is not None:
                resume_counts["recomputed_stale"] += 1
            work_items.append(
                _ProjectionWorkItem(
                    identity=identity,
                    entry=entry,
                    cache_identity=cache_identity,
                    resume=resume,
                )
            )

        results: list[dict[str, Any]] = []
        computed = iter(
            _evaluate_projection_work_items(
                work_items,
                workers=workers,
                cache_config=resolved_cache_config,
                run_provenance=run_provenance,
            )
        )
        try:
            for identity in identities:
                result = reused_by_index.get(identity.row_index)
                if result is None:
                    try:
                        result = next(computed)
                    except StopIteration as exc:
                        raise RuntimeError("process worker result stream ended early") from exc
                    if result.get("row_index") != identity.row_index:
                        raise RuntimeError("process worker result ordering invariant failed")
                    resume_counts["processed"] += 1
                results.append(result)
                if len(results) % checkpoint_every == 0:
                    _write_jsonl_atomic(output_path, results)
            try:
                extra_result = next(computed)
            except StopIteration:
                extra_result = None
            if extra_result is not None:
                raise RuntimeError("process worker result stream produced extra rows")
        finally:
            close_computed = getattr(computed, "close", None)
            if callable(close_computed):
                close_computed()

        if len(results) != len(identities):
            raise RuntimeError("result ordering invariant failed")
        _write_jsonl_atomic(output_path, results)
        _require_sources_unchanged(source_identities)
        if _file_sha256(identity_path) != identity_source["sha256"]:
            raise RuntimeError("identity rows changed during Experiment 004 execution")
        if _file_sha256(selection_path) != selection_source["sha256"]:
            raise RuntimeError("selection manifest changed during Experiment 004 execution")
        if prior_path is not None and _file_sha256(prior_path) != prior["sha256"]:
            raise RuntimeError("prior-stage summary changed during Experiment 004 execution")
        _require_result_resume_caches_unchanged(results)

        output_sha256 = _file_sha256(output_path)
        summary = _build_summary(
            stage=stage,
            results=results,
            output_path=output_path,
            output_sha256=output_sha256,
            identity_source=identity_source,
            selection_source=selection_source,
            prior=prior,
            run_provenance=run_provenance,
            resume_counts=resume_counts,
            total_seconds=time.time() - started_at,
            workers=workers,
        )
        _write_json_atomic(summary_path, summary)
        return summary


run_exp004_cache_projection = run_exp004_projection


def _evaluate_projection_work_items(
    work_items: Sequence[_ProjectionWorkItem],
    *,
    workers: int,
    cache_config: BeatThisFramePredictionCacheConfig,
    run_provenance: Mapping[str, Any],
) -> Iterator[dict[str, Any]]:
    """Yield row results in manifest order, independent of worker completion order."""

    if not work_items:
        return
    if workers == 1:
        for work_item in work_items:
            yield _evaluate_projection_work_item_in_process(
                work_item,
                cache_config=cache_config,
                run_provenance=run_provenance,
            )
        return

    process_context = multiprocessing.get_context(PROCESS_START_METHOD)
    worker_pool = process_context.Pool(
        processes=workers,
        initializer=_initialize_projection_worker,
        initargs=(cache_config, dict(run_provenance)),
    )
    try:
        yield from worker_pool.imap(
            _evaluate_projection_work_item,
            work_items,
            chunksize=1,
        )
        worker_pool.close()
        worker_pool.join()
    except BaseException:
        worker_pool.terminate()
        worker_pool.join()
        raise


def _initialize_projection_worker(
    cache_config: BeatThisFramePredictionCacheConfig,
    run_provenance: Mapping[str, Any],
) -> None:
    global _WORKER_CACHE_CONFIG, _WORKER_RUN_PROVENANCE
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Exp004 projection worker did not initialize on its main thread")
    if not hasattr(signal, "setitimer") or not hasattr(signal, "SIGALRM"):
        raise RuntimeError("Exp004 projection worker requires POSIX interval-timer support")
    _WORKER_CACHE_CONFIG = cache_config
    _WORKER_RUN_PROVENANCE = dict(run_provenance)


def _evaluate_projection_work_item(work_item: _ProjectionWorkItem) -> dict[str, Any]:
    if _WORKER_CACHE_CONFIG is None or _WORKER_RUN_PROVENANCE is None:
        raise RuntimeError("Exp004 projection worker is not initialized")
    return _evaluate_projection_work_item_in_process(
        work_item,
        cache_config=_WORKER_CACHE_CONFIG,
        run_provenance=_WORKER_RUN_PROVENANCE,
    )


def _evaluate_projection_work_item_in_process(
    work_item: _ProjectionWorkItem,
    *,
    cache_config: BeatThisFramePredictionCacheConfig,
    run_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return _evaluate_with_timeout(
        identity=work_item.identity,
        entry=work_item.entry,
        cache_config=cache_config,
        cache_identity=work_item.cache_identity,
        resume=work_item.resume,
        run_provenance=run_provenance,
    )


def _evaluate_with_timeout(
    *,
    identity: _IdentityRow,
    entry: _SelectionEntry,
    cache_config: BeatThisFramePredictionCacheConfig,
    cache_identity: _CacheFileIdentity,
    resume: Mapping[str, Any],
    run_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    state = _new_row_state(cache_identity)
    try:
        with _per_audio_timeout(PER_AUDIO_TIMEOUT_SECONDS):
            _evaluate_row_steps(
                identity=identity,
                cache_config=cache_config,
                initial_cache_identity=cache_identity,
                state=state,
            )
            _apply_cache_execution_identity_guard(
                state=state,
                initial_cache_identity=cache_identity,
            )
    except _PerAudioTimeout:
        _apply_timeout(state)
    except Exception as exc:  # row-local isolation; the finite schema still applies
        _apply_unexpected_row_failure(state, exc)
    return _finalize_row(
        identity=identity,
        entry=entry,
        state=state,
        resume=resume,
        run_provenance=run_provenance,
        total_seconds=time.perf_counter() - started,
    )


def _new_row_state(cache_identity: _CacheFileIdentity) -> dict[str, Any]:
    return {
        "active_stage": "cache_load",
        "cache": {
            "status": "pending",
            "identity_before_load": cache_identity.to_dict(),
            "identity_after_load": None,
            "identity_after_execution": None,
            "frame_count": None,
            "frame_rate_hz": None,
            "coverage_start_ms": None,
            "coverage_end_ms": None,
            "provider": None,
            "checkpoint_path": None,
            "error_type": None,
            "error": None,
        },
        "current_v2": _not_run_payload("not_run_cache_unavailable"),
        "candidate_extraction": _not_run_payload("not_run_cache_unavailable"),
        "variants": {
            variant: _not_run_payload("not_run_cache_unavailable")
            for variant in GLOBAL_CONSTANT_JUMP_VARIANTS
        },
        "runtime": {
            "cache_load_seconds": 0.0,
            "current_v2_seconds": 0.0,
            "candidate_extraction_seconds": 0.0,
            "variant_fit_seconds": 0.0,
        },
    }


def _evaluate_row_steps(
    *,
    identity: _IdentityRow,
    cache_config: BeatThisFramePredictionCacheConfig,
    initial_cache_identity: _CacheFileIdentity,
    state: dict[str, Any],
) -> None:
    cache_started = time.perf_counter()
    prediction: FrameTimingPrediction | None = None
    cache_error: BeatThisFramePredictionCacheError | None = None
    try:
        prediction = load_beatthis_frame_prediction_cache(identity.cache_audio_key, cache_config)
    except _PerAudioTimeout:
        raise
    except BeatThisFramePredictionCacheError as exc:
        timeout_cause = _exception_cause(exc, _PerAudioTimeout)
        if timeout_cause is not None:
            raise timeout_cause
        cache_error = exc

    try:
        final_cache_identity = _cache_file_identity(Path(initial_cache_identity.path))
        state["cache"]["identity_after_load"] = final_cache_identity.to_dict()
        if initial_cache_identity != final_cache_identity:
            raise _CacheChangedDuringLoad("cache bytes or stat identity changed during load")
        if prediction is not None and not final_cache_identity.exists:
            raise _CacheChangedDuringLoad("cache loader returned a prediction without stable cache bytes")
    except _PerAudioTimeout:
        raise
    except _CacheChangedDuringLoad as exc:
        state["cache"].update(status="mutated_during_load", error_type=type(exc).__name__, error=str(exc))
        _mark_cache_unavailable(state, "cache_mutated_during_load")
        return
    finally:
        state["runtime"]["cache_load_seconds"] = time.perf_counter() - cache_started

    if cache_error is not None:
        status = "config_mismatch" if "mismatch" in str(cache_error).lower() else "invalid"
        state["cache"].update(
            status=status,
            error_type=type(cache_error).__name__,
            error=str(cache_error),
        )
        _mark_cache_unavailable(state, f"cache_{status}")
        return
    if prediction is None:
        state["cache"].update(status="missing", error_type="FileNotFoundError", error="cache file is absent")
        _mark_cache_unavailable(state, "cache_missing")
        return

    prediction.beat_prob.setflags(write=False)
    prediction.downbeat_prob.setflags(write=False)
    coverage_end_ms = 1000.0 * prediction.frame_count / prediction.frame_rate_hz
    state["cache"].update(
        status="valid",
        frame_count=prediction.frame_count,
        frame_rate_hz=float(prediction.frame_rate_hz),
        coverage_start_ms=0.0,
        coverage_end_ms=float(coverage_end_ms),
        provider=prediction.provider,
        checkpoint_path=prediction.checkpoint_path,
    )

    state["active_stage"] = "current_v2"
    v2_started = time.perf_counter()
    try:
        fit = GridFitter().fit(prediction)
        state["current_v2"] = _serialize_v2_fit(fit.grid, fit.score, fit.diagnostics)
    except _PerAudioTimeout:
        raise
    except Exception as exc:
        state["current_v2"] = _failure_payload("fit_failure", exc)
    finally:
        state["runtime"]["current_v2_seconds"] = time.perf_counter() - v2_started

    state["active_stage"] = "candidate_extraction"
    restricted_prediction = FrameTimingPrediction(
        provider=prediction.provider,
        checkpoint_path=prediction.checkpoint_path,
        source_path=None,
        beat_prob=prediction.beat_prob,
        downbeat_prob=prediction.downbeat_prob,
        frame_rate_hz=prediction.frame_rate_hz,
    )
    if restricted_prediction.source_path is not None:
        raise RuntimeError("restricted prediction unexpectedly retained source_path")
    if not np.shares_memory(restricted_prediction.beat_prob, prediction.beat_prob):
        raise RuntimeError("restricted prediction does not share the loaded beat signal")
    if not np.shares_memory(restricted_prediction.downbeat_prob, prediction.downbeat_prob):
        raise RuntimeError("restricted prediction does not share the loaded downbeat signal")

    candidate_started = time.perf_counter()
    try:
        candidate_set = extract_global_constant_jump_candidates(restricted_prediction)
        candidate_payload = _jsonable(asdict(candidate_set.diagnostics))
        state["candidate_extraction"] = {
            "status": "accepted",
            "reason": None,
            "candidate_fingerprint": candidate_set.diagnostics.candidate_fingerprint,
            "restricted_prediction_source_path": None,
            "shares_loaded_signals": True,
            "diagnostics": candidate_payload,
            "error_type": None,
            "error": None,
        }
    except _PerAudioTimeout:
        raise
    except Exception as exc:
        state["candidate_extraction"] = _failure_payload("candidate_extraction_failure", exc)
        state["variants"] = {
            variant: _not_run_payload("candidate_extraction_failure")
            for variant in GLOBAL_CONSTANT_JUMP_VARIANTS
        }
        return
    finally:
        state["runtime"]["candidate_extraction_seconds"] = time.perf_counter() - candidate_started

    variant_started = time.perf_counter()
    fitted_iterator: Iterator[tuple[str, object]] | None = None
    try:
        first_variant = GLOBAL_CONSTANT_JUMP_VARIANTS[0]
        state["active_stage"] = f"variant_fit:{first_variant}"
        try:
            fitted_iterator = iter(
                iter_global_constant_jump_variants(
                    restricted_prediction,
                    variants=GLOBAL_CONSTANT_JUMP_VARIANTS,
                    attempt_cap=GLOBAL_CONSTANT_JUMP_CONSTANTS.default_edge_attempt_cap,
                    candidate_set=candidate_set,
                )
            )
        except _PerAudioTimeout:
            raise
        except Exception as exc:
            _mark_variant_failure_and_tail_not_run(
                state,
                variant=first_variant,
                failure=_failure_payload("variant_fit_failure", exc),
                tail_reason="not_run_after_variant_fit_failure",
            )
            return
        for variant_index, variant in enumerate(GLOBAL_CONSTANT_JUMP_VARIANTS):
            state["active_stage"] = f"variant_fit:{variant}"
            try:
                yielded_variant, result = next(fitted_iterator)
                if yielded_variant != variant:
                    raise RuntimeError(
                        "shared-candidate core returned variants out of frozen order: "
                        f"expected {variant}, got {yielded_variant!r}"
                    )
                if getattr(result, "variant", None) != variant:
                    raise RuntimeError(f"core result variant mismatch for {variant}")
            except _PerAudioTimeout:
                raise
            except StopIteration as exc:
                _mark_variant_failure_and_tail_not_run(
                    state,
                    variant=variant,
                    failure=_failure_payload("variant_fit_failure", exc),
                    tail_reason="not_run_after_variant_fit_failure",
                )
                return
            except Exception as exc:
                _mark_variant_failure_and_tail_not_run(
                    state,
                    variant=variant,
                    failure=_failure_payload("variant_fit_failure", exc),
                    tail_reason="not_run_after_variant_fit_failure",
                )
                return

            state["active_stage"] = f"variant_serialize:{variant}"
            try:
                serialized_result = _serialize_variant_result(
                    result,
                    expected_coverage_end_ms=coverage_end_ms,
                    expected_candidate_fingerprint=candidate_set.diagnostics.candidate_fingerprint,
                )
                if variant_index + 1 < len(GLOBAL_CONSTANT_JUMP_VARIANTS):
                    next_variant = GLOBAL_CONSTANT_JUMP_VARIANTS[variant_index + 1]
                    next_active_stage = f"variant_fit:{next_variant}"
                else:
                    next_active_stage = "variants_complete"
                _store_completed_variant(
                    state,
                    variant=variant,
                    payload=serialized_result,
                    next_active_stage=next_active_stage,
                )
            except _PerAudioTimeout:
                raise
            except Exception as exc:
                _mark_variant_failure_and_tail_not_run(
                    state,
                    variant=variant,
                    failure=_failure_payload("schema_or_serialization_failure", exc),
                    tail_reason="not_run_after_schema_or_serialization_failure",
                )
                return
    except _PerAudioTimeout:
        raise
    finally:
        if fitted_iterator is not None:
            close_iterator = getattr(fitted_iterator, "close", None)
            if callable(close_iterator):
                close_iterator()
        state["runtime"]["variant_fit_seconds"] = time.perf_counter() - variant_started
    state["active_stage"] = "complete"


def _serialize_v2_fit(grid: FittedTimingGrid, score: float, diagnostics: object) -> dict[str, Any]:
    segments = [
        {
            "offset_ms": float(segment.offset_ms),
            "beat_length_ms": float(segment.beat_length_ms),
            "meter": int(segment.meter),
        }
        for segment in grid.segments
    ]
    payload = {
        "schema": "pulsefield_model.timing_fitted_grid_v1",
        "segments": segments,
    }
    return {
        "status": "accepted",
        "reason": None,
        "grid": payload,
        "grid_fingerprint": _stable_json_sha256(payload),
        "score": float(score),
        "diagnostics": _jsonable(asdict(diagnostics) if dataclasses.is_dataclass(diagnostics) else diagnostics),
        "error_type": None,
        "error": None,
    }


def _mark_cache_unavailable(state: dict[str, Any], reason: str) -> None:
    state["current_v2"] = _not_run_payload(reason)
    state["candidate_extraction"] = _not_run_payload(reason)
    state["variants"] = {
        variant: _not_run_payload(reason)
        for variant in GLOBAL_CONSTANT_JUMP_VARIANTS
    }


def _apply_cache_execution_identity_guard(
    *,
    state: dict[str, Any],
    initial_cache_identity: _CacheFileIdentity,
) -> None:
    if state["cache"].get("status") != "valid":
        return

    state["active_stage"] = "cache_execution_identity_check"
    try:
        current_cache_identity = _cache_file_identity(Path(initial_cache_identity.path))
    except _CacheChangedDuringLoad as exc:
        state["cache"]["identity_after_execution"] = None
        _mark_cache_mutated_during_execution(state, exc)
        return

    state["cache"]["identity_after_execution"] = current_cache_identity.to_dict()
    if current_cache_identity != initial_cache_identity:
        _mark_cache_mutated_during_execution(
            state,
            _CacheChangedDuringLoad("cache bytes or stat identity changed during row execution"),
        )


def _mark_cache_mutated_during_execution(
    state: dict[str, Any],
    exc: BaseException,
) -> None:
    state["cache"].update(
        status="mutated_during_execution",
        error_type=type(exc).__name__,
        error=str(exc),
    )
    _mark_cache_unavailable(state, "cache_mutated_during_execution")


def _serialize_variant_result(
    result: object,
    *,
    expected_coverage_end_ms: float,
    expected_candidate_fingerprint: str,
) -> dict[str, Any]:
    diagnostics = getattr(result, "diagnostics")
    diagnostics_payload = _jsonable(asdict(diagnostics))
    if diagnostics.candidate_fingerprint != expected_candidate_fingerprint:
        raise ValueError("variant candidate fingerprint does not match the shared candidate set")
    grid = getattr(result, "grid")
    reason = getattr(result, "reason")
    if grid is None:
        return {
            "status": "failed",
            "reason": str(reason or diagnostics.fallback_reason or "unknown_core_failure"),
            "grid": None,
            "grid_fingerprint": None,
            "diagnostics": diagnostics_payload,
            "error_type": None,
            "error": None,
        }
    grid_payload, grid_fingerprint, seam_max_ms = _validated_v3_grid_payload(
        grid,
        expected_coverage_end_ms=expected_coverage_end_ms,
    )
    if diagnostics.grid_fingerprint is not None and diagnostics.grid_fingerprint != grid_fingerprint:
        raise ValueError("core diagnostic grid fingerprint does not match canonical grid payload")
    return {
        "status": "accepted",
        "reason": None,
        "grid": grid_payload,
        "grid_fingerprint": grid_fingerprint,
        "serialization_max_boundary_delta_ms": seam_max_ms,
        "diagnostics": diagnostics_payload,
        "error_type": None,
        "error": None,
    }


def _validated_v3_grid_payload(
    grid: TimingV3Grid,
    *,
    expected_coverage_end_ms: float,
) -> tuple[dict[str, Any], str, float]:
    if not isinstance(grid, TimingV3Grid):
        raise TypeError("accepted core grid must be a TimingV3Grid")
    if grid.origin_beat != 0:
        raise ValueError("accepted Exp004 grid origin_beat must be zero")
    if len(grid.sections) > MAX_V3_SECTION_COUNT:
        raise ValueError(f"accepted Exp004 grid exceeds {MAX_V3_SECTION_COUNT} sections")
    if not math.isclose(grid.coverage_start_ms, 0.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("accepted Exp004 grid coverage_start_ms must be zero")
    if not math.isclose(grid.coverage_end_ms, expected_coverage_end_ms, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("accepted Exp004 grid coverage does not match cache support")

    payload = _jsonable(grid.to_dict())
    restored = TimingV3Grid.from_dict(payload)
    restored_payload = _jsonable(restored.to_dict())
    if payload != restored_payload:
        raise ValueError("TimingV3Grid JSON round trip changed the mathematical payload")
    if len(grid.boundary_times_ms) != len(restored.boundary_times_ms):
        raise ValueError("TimingV3Grid JSON round trip changed boundary count")
    seam_deltas = [
        abs(float(left) - float(right))
        for left, right in zip(grid.boundary_times_ms, restored.boundary_times_ms, strict=True)
    ]
    for original, delta in zip(grid.boundary_times_ms, seam_deltas, strict=True):
        if delta > roundtrip_seam_tolerance_ms(float(original)):
            raise ValueError("TimingV3Grid JSON round trip exceeded seam tolerance")
    _canonical_json(payload)
    return payload, _stable_json_sha256(payload), max(seam_deltas, default=0.0)


def _variant_from_active_stage(active_stage: str) -> str | None:
    for prefix in ("variant_fit:", "variant_serialize:"):
        if active_stage.startswith(prefix):
            variant = active_stage.removeprefix(prefix)
            if variant in GLOBAL_CONSTANT_JUMP_VARIANTS:
                return variant
    return None


def _store_completed_variant(
    state: dict[str, Any],
    *,
    variant: str,
    payload: Mapping[str, Any],
    next_active_stage: str,
) -> None:
    variants = dict(state["variants"])
    variants[variant] = dict(payload)
    state.update(variants=variants, active_stage=next_active_stage)


def _mark_variant_tail_not_run(
    state: dict[str, Any],
    *,
    variant: str,
    reason: str,
) -> None:
    start_index = GLOBAL_CONSTANT_JUMP_VARIANTS.index(variant)
    for pending_variant in GLOBAL_CONSTANT_JUMP_VARIANTS[start_index:]:
        state["variants"][pending_variant] = _not_run_payload(reason)


def _mark_variant_failure_and_tail_not_run(
    state: dict[str, Any],
    *,
    variant: str,
    failure: Mapping[str, Any],
    tail_reason: str,
) -> None:
    state["variants"][variant] = dict(failure)
    variant_index = GLOBAL_CONSTANT_JUMP_VARIANTS.index(variant)
    for pending_variant in GLOBAL_CONSTANT_JUMP_VARIANTS[variant_index + 1 :]:
        state["variants"][pending_variant] = _not_run_payload(tail_reason)


def _apply_timeout(state: dict[str, Any]) -> None:
    active = str(state.get("active_stage"))
    if active in {"variants_complete", "complete"}:
        return
    if active == "cache_load":
        state["cache"].update(status="timeout", error_type="TimeoutError", error="per-audio timeout")
    elif active == "cache_execution_identity_check":
        state["cache"].update(
            status="mutated_during_execution",
            error_type="TimeoutError",
            error="per-audio timeout during cache execution identity check",
        )
        _mark_cache_unavailable(state, "cache_mutated_during_execution")
        return
    elif active == "current_v2":
        state["current_v2"] = _not_run_payload("timeout")
        state["candidate_extraction"] = _not_run_payload("not_run_after_current_v2_timeout")
    elif active == "candidate_extraction":
        state["candidate_extraction"] = _not_run_payload("timeout")
    active_variant = _variant_from_active_stage(active)
    if active_variant is None:
        state["variants"] = {
            variant: _not_run_payload("timeout")
            for variant in GLOBAL_CONSTANT_JUMP_VARIANTS
        }
    else:
        _mark_variant_tail_not_run(state, variant=active_variant, reason="timeout")


def _apply_unexpected_row_failure(state: dict[str, Any], exc: Exception) -> None:
    active = str(state.get("active_stage"))
    if state["cache"].get("status") == "pending":
        state["cache"].update(status="invalid", error_type=type(exc).__name__, error=str(exc))
    elif active == "current_v2":
        state["current_v2"] = _failure_payload("internal_error", exc)
    elif active == "candidate_extraction":
        state["candidate_extraction"] = _failure_payload("internal_error", exc)
    active_variant = _variant_from_active_stage(active)
    if active_variant is None:
        state["variants"] = {
            variant: _failure_payload("internal_error", exc)
            for variant in GLOBAL_CONSTANT_JUMP_VARIANTS
        }
    else:
        _mark_variant_failure_and_tail_not_run(
            state,
            variant=active_variant,
            failure=_failure_payload("internal_error", exc),
            tail_reason="not_run_after_internal_error",
        )


def _finalize_row(
    *,
    identity: _IdentityRow,
    entry: _SelectionEntry,
    state: Mapping[str, Any],
    resume: Mapping[str, Any],
    run_provenance: Mapping[str, Any],
    total_seconds: float,
) -> dict[str, Any]:
    variants = dict(state["variants"])
    cj3 = variants[VARIANT_CJ3]
    v2 = dict(state["current_v2"])
    if cj3.get("status") == "accepted":
        selection = {
            "method": FALLBACK_POLICY,
            "source": VARIANT_CJ3,
            "used_fallback": False,
            "fallback_reason": None,
            "grid_fingerprint": cj3.get("grid_fingerprint"),
        }
    elif v2.get("status") == "accepted":
        selection = {
            "method": FALLBACK_POLICY,
            "source": "current_v2",
            "used_fallback": True,
            "fallback_reason": str(cj3.get("reason") or "CJ3_unavailable"),
            "grid_fingerprint": v2.get("grid_fingerprint"),
        }
    else:
        selection = {
            "method": FALLBACK_POLICY,
            "source": None,
            "used_fallback": False,
            "fallback_reason": str(cj3.get("reason") or "CJ3_unavailable"),
            "grid_fingerprint": None,
        }

    cache_status = str(state["cache"].get("status") or "unknown")
    cache_valid = cache_status == "valid"
    hard_guard_reasons: list[str] = []
    if not cache_valid:
        hard_guard_reasons.append(f"cache_{cache_status}")
    if v2.get("reason") == "internal_error":
        hard_guard_reasons.append("current_v2_internal_error")
    candidate_reason = str(state["candidate_extraction"].get("reason") or "")
    if candidate_reason == "internal_error":
        hard_guard_reasons.append("candidate_extraction_internal_error")
    for variant in GLOBAL_CONSTANT_JUMP_VARIANTS:
        reason = str(variants[variant].get("reason") or "")
        if reason == "schema_or_serialization_failure":
            hard_guard_reasons.append(f"{variant}_schema_or_serialization_failure")
        elif reason == "timing_v3_schema_construction_failed":
            hard_guard_reasons.append(f"{variant}_schema_construction_failed")
        elif reason in {"variant_fit_failure", "internal_error"}:
            hard_guard_reasons.append(f"{variant}_{reason}")
        elif reason == "timeout":
            hard_guard_reasons.append(f"{variant}_timeout")
    hard_guard_reasons = sorted(set(hard_guard_reasons))
    projection_evaluable = selection["source"] is not None
    runtime = dict(state["runtime"])
    runtime["total_seconds"] = float(total_seconds)
    result = {
        "schema": RESULT_SCHEMA,
        "experiment": _EXPERIMENT,
        "stage": identity.stage,
        "row_index": identity.row_index,
        "row_complete": True,
        "ok": projection_evaluable,
        "identity": {
            "cache_audio_key": identity.cache_audio_key,
            "cache_audio_key_sha256": _sha256_text(identity.cache_audio_key),
            "audio_group_key": identity.audio_group_key,
            "identity_payload_sha256": identity.payload_sha256,
            "selection_entry_sha256": entry.payload_sha256,
        },
        "provenance": run_provenance,
        "resume": dict(resume),
        "cache": state["cache"],
        "current_v2": v2,
        "candidate_extraction": state["candidate_extraction"],
        "variants": variants,
        "selection": selection,
        "projection_flags": {
            "cache_valid": cache_valid,
            "projection_evaluable": projection_evaluable,
            "pure_cj3_grid_produced": cj3.get("status") == "accepted",
            "selected_used_fallback": bool(selection["used_fallback"]),
            "hard_guard_violation": bool(hard_guard_reasons),
            "hard_guard_reasons": hard_guard_reasons,
        },
        "runtime": runtime,
    }
    result = _jsonable(result)
    result["result_fingerprint"] = _stable_json_sha256(result)
    _canonical_json(result)
    return result


def _not_run_payload(reason: str) -> dict[str, Any]:
    return {
        "status": "not_run",
        "reason": reason,
        "grid": None,
        "grid_fingerprint": None,
        "diagnostics": None,
        "error_type": None,
        "error": None,
    }


def _failure_payload(reason: str, exc: Exception) -> dict[str, Any]:
    return {
        "status": "failed",
        "reason": reason,
        "grid": None,
        "grid_fingerprint": None,
        "diagnostics": None,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _exception_cause(exc: BaseException, expected_type: type[BaseException]) -> BaseException | None:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if isinstance(current, expected_type):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


def _load_identity_rows(path: Path, *, expected_stage: str) -> tuple[list[_IdentityRow], dict[str, Any]]:
    return exp004_protocol.load_exp004_identity_rows(
        path,
        expected_stage=expected_stage,
    )


def _load_selection_manifest(
    path: Path,
    *,
    expected_stage: str,
) -> tuple[dict[str, Any], list[_SelectionEntry], dict[str, Any]]:
    return exp004_protocol.load_exp004_execution_selection_manifest(
        path,
        expected_stage=expected_stage,
    )


def _reconcile_identities(
    identities: Sequence[_IdentityRow],
    entries: Sequence[_SelectionEntry],
    *,
    expected_stage: str,
) -> None:
    exp004_protocol.reconcile_exp004_execution_inputs(
        identities,
        entries,
        expected_stage=expected_stage,
    )


def _load_and_validate_prior_summary(
    path: Path | None,
    *,
    stage: str,
    behavior_fingerprint: str,
    config_fingerprint: str,
) -> dict[str, Any] | None:
    expected_prior = PRIOR_STAGE.get(stage)
    if expected_prior is None:
        if path is not None:
            raise ValueError("repair80 must not consume a prior-stage weak summary")
        return None
    if path is None:
        raise ValueError(f"{stage} requires the {expected_prior} weak summary")
    payload, payload_sha256 = _load_json_object_with_sha256(path)
    if payload.get("schema") != WEAK_SUMMARY_SCHEMA:
        raise ValueError("prior-stage summary must be an Exp004 weak-evidence summary")
    if payload.get("experiment") != _EXPERIMENT:
        raise ValueError("prior-stage weak summary experiment is invalid")
    if payload.get("stage") != expected_prior:
        raise ValueError(f"{stage} requires a {expected_prior} prior-stage weak summary")

    hard_guards = payload.get("hard_guards")
    if not isinstance(hard_guards, MappingABC) or hard_guards.get("ok") is not True:
        raise ValueError("prior-stage weak summary did not pass projection hard guards")
    expected_decision, expected_action = _PRIOR_WEAK_DECISION[expected_prior]
    if payload.get("decision") != expected_decision or payload.get("next_action") != expected_action:
        raise ValueError(
            f"prior-stage weak summary does not authorize {stage}: expected "
            f"decision/action {(expected_decision, expected_action)!r}"
        )

    expected_count = STAGE_AUDIO_COUNTS[expected_prior]
    output = _required_mapping(payload.get("output"), "prior weak summary output")
    output_path = _required_string(output.get("path"), "prior weak summary output.path")
    output_sha256 = _required_sha256_value(
        output.get("sha256"),
        "prior weak summary output.sha256",
    )
    if output.get("row_count") != expected_count:
        raise ValueError("prior-stage weak summary output row count is incomplete")

    denominators = _required_mapping(
        payload.get("denominators"),
        "prior weak summary denominators",
    )
    if denominators.get("stage_audio_count") != expected_count:
        raise ValueError("prior-stage weak summary stage denominator is incomplete")
    for field_name in (
        "cache_valid_count",
        "projection_evaluable_count",
        "comparison_eligible_count",
        "pure_CJ3_phase_count",
        "selected_safety_phase_count",
        "selected_fallback_count",
    ):
        value = denominators.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= expected_count:
            raise ValueError(f"prior weak denominator {field_name} is invalid")

    stage_gates = _required_mapping(
        payload.get("stage_gates"),
        "prior weak summary stage_gates",
    )
    if stage_gates.get("schema") != WEAK_STAGE_GATES_SCHEMA or stage_gates.get("stage") != expected_prior:
        raise ValueError("prior weak stage-gate schema or stage is invalid")
    gates = _required_mapping(stage_gates.get("gates"), "prior weak stage_gates.gates")
    if set(gates) != set(_REQUIRED_WEAK_STAGE_GATES):
        raise ValueError("prior weak stage-gate set does not match the frozen Exp004 table")
    decision_statuses: list[str] = []
    for gate_name, gate in gates.items():
        gate_payload = _required_mapping(gate, f"prior weak gate {gate_name}")
        gate_status = gate_payload.get("status")
        if gate_status not in {"pass", "ambiguous", "kill", "not_applicable"}:
            raise ValueError(f"prior weak gate {gate_name} has an invalid status")
        for required_field in (
            "value",
            "numerator",
            "denominator",
            "threshold",
            "reason",
            "decision_gate",
        ):
            if required_field not in gate_payload:
                raise ValueError(f"prior weak gate {gate_name} is missing {required_field}")
        expected_decision_gate = gate_name not in {
            "jump_mean_phase_ratio",
            "jump_endpoint_drift_mean_ratio",
            *({"phase_denominator_available"} if expected_prior == "repair80" else set()),
        }
        if gate_payload.get("decision_gate") is not expected_decision_gate:
            raise ValueError(f"prior weak gate {gate_name} decision relevance is invalid")
        if expected_decision_gate and gate_status != "not_applicable":
            decision_statuses.append(str(gate_status))

    projection_hard_guard_gate = _required_mapping(
        gates.get("projection_hard_guards"),
        "prior weak projection_hard_guards gate",
    )
    if (
        projection_hard_guard_gate.get("status") != "pass"
        or projection_hard_guard_gate.get("value") is not True
    ):
        raise ValueError("prior weak projection hard-guard gate disagrees with hard_guards.ok")
    if expected_prior == "repair80":
        implied_decision = "debug_only"
    elif "kill" in decision_statuses:
        implied_decision = "kill"
    elif "ambiguous" in decision_statuses:
        implied_decision = "ambiguous"
    else:
        implied_decision = "pass"
    if implied_decision != expected_decision:
        raise ValueError(
            "prior weak stage gates do not imply the decision required to authorize "
            f"{stage}: {implied_decision!r}"
        )

    binding = _required_mapping(
        payload.get("protocol_binding"),
        "prior weak protocol_binding",
    )
    if binding.get("schema") != WEAK_PROTOCOL_BINDING_SCHEMA:
        raise ValueError("prior weak protocol binding schema is invalid")
    source_projection = _required_mapping(
        binding.get("source_projection"),
        "prior weak protocol source_projection",
    )
    if source_projection.get("stage") != expected_prior:
        raise ValueError("prior weak source projection stage is invalid")
    if source_projection.get("behavior_fingerprint") != behavior_fingerprint:
        raise ValueError("prior-stage behavior fingerprint does not match current execution")
    if source_projection.get("config_fingerprint") != config_fingerprint:
        raise ValueError("prior-stage config fingerprint does not match current execution")
    if source_projection.get("projection_summary_hard_guards_ok") is not True:
        raise ValueError("prior weak source projection hard guards did not pass")
    if source_projection.get("projection_summary_formal_execution_ready") is not True:
        raise ValueError("prior weak source projection was not formal-execution ready")
    projection_jsonl_sha256 = _required_sha256_value(
        source_projection.get("projection_jsonl_sha256"),
        "prior weak source projection JSONL sha256",
    )
    projection_summary_sha256 = _required_sha256_value(
        source_projection.get("projection_summary_sha256"),
        "prior weak source projection summary sha256",
    )
    _required_sha256_value(
        source_projection.get("run_fingerprint"),
        "prior weak source projection run fingerprint",
    )

    baseline = _required_mapping(binding.get("baseline"), "prior weak baseline binding")
    baseline_sha256 = _required_sha256_value(
        baseline.get("sha256"),
        "prior weak baseline sha256",
    )
    baseline_row_count = baseline.get("row_count")
    if (
        isinstance(baseline_row_count, bool)
        or not isinstance(baseline_row_count, int)
        or baseline_row_count < expected_count
    ):
        raise ValueError("prior weak baseline binding row count is invalid")
    evaluator = _required_mapping(binding.get("evaluator"), "prior weak evaluator binding")
    evaluator_sha256 = _required_sha256_value(
        evaluator.get("evaluator_sha256"),
        "prior weak evaluator sha256",
    )
    _required_sha256_value(
        evaluator.get("metrics_sha256"),
        "prior weak metrics sha256",
    )
    canonical_binding = _required_mapping(
        evaluator.get("canonical_bpm_binding"),
        "prior weak canonical BPM binding",
    )
    _required_sha256_value(
        canonical_binding.get("source_sha256"),
        "prior weak canonical BPM source sha256",
    )

    binding_output = _required_mapping(binding.get("output"), "prior weak binding output")
    if (
        binding_output.get("path") != output_path
        or binding_output.get("sha256") != output_sha256
        or binding_output.get("row_count") != expected_count
    ):
        raise ValueError("prior weak binding output does not match summary output")
    if binding.get("denominators") != denominators:
        raise ValueError("prior weak binding denominators do not match summary")
    if binding.get("stage_gates") != stage_gates:
        raise ValueError("prior weak binding stage gates do not match summary")
    if binding.get("decision") != expected_decision or binding.get("next_action") != expected_action:
        raise ValueError("prior weak binding decision/action does not match summary")
    if binding.get("hard_guards") != hard_guards:
        raise ValueError("prior weak binding hard guards do not match summary")

    source = _required_mapping(payload.get("source"), "prior weak summary source")
    source_projection_summary = _required_mapping(
        source.get("projection_summary"),
        "prior weak source projection_summary",
    )
    if (
        source_projection_summary.get("behavior_fingerprint") != behavior_fingerprint
        or source_projection_summary.get("config_fingerprint") != config_fingerprint
        or source_projection_summary.get("sha256") != projection_summary_sha256
        or source_projection_summary.get("hard_guards_ok") is not True
        or source_projection_summary.get("formal_execution_ready") is not True
    ):
        raise ValueError("prior weak source projection summary binding is inconsistent")
    source_projection_jsonl = _required_mapping(
        source.get("projection_jsonl"),
        "prior weak source projection_jsonl",
    )
    if source_projection_jsonl.get("sha256") != projection_jsonl_sha256:
        raise ValueError("prior weak source projection JSONL binding is inconsistent")
    source_baseline = _required_mapping(source.get("baseline_jsonl"), "prior weak source baseline")
    if source_baseline.get("sha256") != baseline_sha256:
        raise ValueError("prior weak source baseline binding is inconsistent")
    source_evaluator = _required_mapping(source.get("evaluator"), "prior weak source evaluator")
    if source_evaluator.get("evaluator_sha256") != evaluator_sha256 or source_evaluator != evaluator:
        raise ValueError("prior weak source evaluator binding is inconsistent")

    return {
        "path": path.resolve(strict=True).as_posix(),
        "sha256": payload_sha256,
        "stage": expected_prior,
        "behavior_fingerprint": behavior_fingerprint,
        "config_fingerprint": config_fingerprint,
        "decision": expected_decision,
        "next_action": expected_action,
        "weak_output_jsonl_sha256": output_sha256,
        "projection_jsonl_sha256": projection_jsonl_sha256,
        "projection_summary_sha256": projection_summary_sha256,
        "baseline_jsonl_sha256": baseline_sha256,
        "evaluator_sha256": evaluator_sha256,
        "stage_gates_sha256": _stable_json_sha256(stage_gates),
        "protocol_binding_sha256": _stable_json_sha256(binding),
    }


def _behavior_source_identities() -> list[dict[str, str]]:
    root = _repo_root()
    identities: list[dict[str, str]] = []
    for relative in _BEHAVIOR_SOURCE_RELATIVE_PATHS:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"required Experiment 004 source is missing: {path}")
        identities.append({"relative_path": relative, "sha256": _file_sha256(path)})
    return identities


def _behavior_payload(source_identities: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "pulsefield_model.timing_v3_exp004_behavior_provenance_v1",
        "candidate_contract_version": CANDIDATE_CONTRACT_VERSION,
        "candidate_constants_json_sha256": GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON_SHA256,
        "variants": list(GLOBAL_CONSTANT_JUMP_VARIANTS),
        "fallback_policy": FALLBACK_POLICY,
        "per_audio_timeout_seconds": PER_AUDIO_TIMEOUT_SECONDS,
        "maximum_v3_section_count": MAX_V3_SECTION_COUNT,
        "parallel_execution": {
            "maximum_workers": MAX_PROJECTION_WORKERS,
            "process_start_method": PROCESS_START_METHOD,
            "ordered_map_chunksize": 1,
        },
        "formal_execution_ready": not INTEGRATION_BLOCKERS,
        "integration_blockers": list(INTEGRATION_BLOCKERS),
        "source_files": [dict(item) for item in source_identities],
    }


def _config_payload(cache_config: BeatThisFramePredictionCacheConfig) -> dict[str, Any]:
    return {
        "schema": "pulsefield_model.timing_v3_exp004_projection_config_v1",
        "cache": {
            "cache_version": cache_config.cache_version,
            "checkpoint_path": cache_config.checkpoint_path,
            "float16": cache_config.float16,
            "shift_ms": cache_config.shift_ms,
            "frame_rate_hz": cache_config.frame_rate_hz,
            "config_fingerprint": cache_config.config_fingerprint,
        },
        "current_v2_grid_fitter_config": _jsonable(asdict(GridFitterConfig())),
        "candidate_constants": GLOBAL_CONSTANT_JUMP_CONSTANTS.to_jsonable(),
    }


def _run_provenance(
    *,
    stage: str,
    behavior_payload: Mapping[str, Any],
    behavior_fingerprint: str,
    config_payload: Mapping[str, Any],
    config_fingerprint: str,
    identity_source: Mapping[str, Any],
    selection_source: Mapping[str, Any],
    prior: Mapping[str, Any] | None,
    workers: int,
) -> dict[str, Any]:
    environment = _environment_provenance()
    resume_contract = {
        "schema": "pulsefield_model.timing_v3_exp004_projection_run_contract_v1",
        "stage": stage,
        "behavior_fingerprint": behavior_fingerprint,
        "config_fingerprint": config_fingerprint,
        "identity_source_sha256": identity_source["sha256"],
        "selection_manifest_sha256": selection_source["sha256"],
        "selection_manifest_fingerprint": selection_source["manifest_fingerprint_sha256"],
        "prior_stage_summary_sha256": prior["sha256"] if prior is not None else None,
        "execution": {
            "workers": workers,
            "process_start_method": PROCESS_START_METHOD if workers > 1 else None,
            "ordered_map_chunksize": 1 if workers > 1 else None,
        },
        "environment": environment,
    }
    payload: dict[str, Any] = {
        "schema": "pulsefield_model.timing_v3_exp004_projection_provenance_v1",
        "stage": stage,
        "behavior_fingerprint": behavior_fingerprint,
        "config_fingerprint": config_fingerprint,
        "behavior": dict(behavior_payload),
        "config": dict(config_payload),
        "identity_source": dict(identity_source),
        "selection_source": dict(selection_source),
        "prior_stage": dict(prior) if prior is not None else None,
        "git": _git_provenance(),
        "environment": environment,
        "execution": dict(resume_contract["execution"]),
        "resume_contract": resume_contract,
    }
    payload["run_fingerprint"] = _stable_json_sha256(resume_contract)
    return _jsonable(payload)


def _resume_payload(
    *,
    identity: _IdentityRow,
    entry: _SelectionEntry,
    cache_identity: _CacheFileIdentity,
    run_provenance: Mapping[str, Any],
    identity_source: Mapping[str, Any],
    selection_source: Mapping[str, Any],
    prior: Mapping[str, Any] | None,
) -> dict[str, Any]:
    components = {
        "run_fingerprint": run_provenance["run_fingerprint"],
        "behavior_fingerprint": run_provenance["behavior_fingerprint"],
        "config_fingerprint": run_provenance["config_fingerprint"],
        "identity_source_sha256": identity_source["sha256"],
        "selection_manifest_sha256": selection_source["sha256"],
        "selection_manifest_fingerprint": selection_source["manifest_fingerprint_sha256"],
        "prior_stage_summary_sha256": prior["sha256"] if prior is not None else None,
        "row_index": identity.row_index,
        "identity_payload_sha256": identity.payload_sha256,
        "selection_entry_sha256": entry.payload_sha256,
        "cache_audio_key_sha256": _sha256_text(identity.cache_audio_key),
        "cache_file": cache_identity.to_dict(),
    }
    return {
        "schema": RESUME_SCHEMA,
        "fingerprint": _stable_json_sha256(components),
        "components": components,
    }


def _load_existing_results(path: Path, *, identities: Sequence[_IdentityRow]) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = _load_jsonl_objects(path)
    by_index: dict[int, dict[str, Any]] = {}
    audio_keys: set[str] = set()
    fingerprints: set[str] = set()
    for line_index, row in enumerate(rows):
        if row.get("schema") != RESULT_SCHEMA:
            raise ValueError(f"existing result row {line_index} has wrong schema")
        result_fingerprint = row.get("result_fingerprint")
        _require_sha256(result_fingerprint, f"existing result row {line_index} result fingerprint")
        fingerprint_body = dict(row)
        fingerprint_body.pop("result_fingerprint", None)
        if result_fingerprint != _stable_json_sha256(fingerprint_body):
            raise ValueError(f"existing result row {line_index} content fingerprint mismatch")
        if row.get("row_complete") is not True:
            raise ValueError(f"existing result row {line_index} is incomplete")
        row_index = row.get("row_index")
        if isinstance(row_index, bool) or not isinstance(row_index, int) or not 0 <= row_index < len(identities):
            raise ValueError(f"existing result row {line_index} has invalid row_index")
        resume = row.get("resume")
        if not isinstance(resume, MappingABC) or resume.get("schema") != RESUME_SCHEMA:
            raise ValueError(f"existing result row {line_index} has wrong resume schema")
        fingerprint = resume.get("fingerprint")
        _require_sha256(fingerprint, f"existing result row {line_index} resume fingerprint")
        identity = row.get("identity")
        if not isinstance(identity, MappingABC):
            raise ValueError(f"existing result row {line_index} identity is missing")
        audio_key = _required_string(identity.get("cache_audio_key"), "existing result cache_audio_key")
        expected_identity = identities[row_index]
        if row.get("experiment") != _EXPERIMENT or row.get("stage") != expected_identity.stage:
            raise ValueError(f"existing result row {line_index} experiment/stage mismatch")
        if audio_key != expected_identity.cache_audio_key:
            raise ValueError(f"existing result row {line_index} cache identity mismatch")
        if identity.get("audio_group_key") != expected_identity.audio_group_key:
            raise ValueError(f"existing result row {line_index} audio-group identity mismatch")
        if row_index in by_index:
            raise ValueError(f"duplicate existing result row_index: {row_index}")
        if audio_key in audio_keys:
            raise ValueError(f"duplicate existing result cache_audio_key: {audio_key!r}")
        if fingerprint in fingerprints:
            raise ValueError(f"duplicate existing result resume fingerprint: {fingerprint}")
        by_index[row_index] = row
        audio_keys.add(audio_key)
        fingerprints.add(fingerprint)
    return by_index


def _build_summary(
    *,
    stage: str,
    results: Sequence[Mapping[str, Any]],
    output_path: Path,
    output_sha256: str,
    identity_source: Mapping[str, Any],
    selection_source: Mapping[str, Any],
    prior: Mapping[str, Any] | None,
    run_provenance: Mapping[str, Any],
    resume_counts: Mapping[str, int],
    total_seconds: float,
    workers: int,
) -> dict[str, Any]:
    cache_valid_count = sum(bool(_nested(row, "projection_flags", "cache_valid")) for row in results)
    projection_evaluable_count = sum(
        bool(_nested(row, "projection_flags", "projection_evaluable")) for row in results
    )
    fallback_count = sum(bool(_nested(row, "projection_flags", "selected_used_fallback")) for row in results)
    fallback_rate = fallback_count / projection_evaluable_count if projection_evaluable_count else None
    cache_statuses = Counter(str(_nested(row, "cache", "status") or "unknown") for row in results)
    fallback_reasons = Counter(
        str(_nested(row, "selection", "fallback_reason"))
        for row in results
        if bool(_nested(row, "projection_flags", "selected_used_fallback"))
    )
    variant_statuses = {
        variant: dict(
            sorted(
                Counter(str(_nested(row, "variants", variant, "status") or "unknown") for row in results).items()
            )
        )
        for variant in GLOBAL_CONSTANT_JUMP_VARIANTS
    }
    variant_reasons = {
        variant: dict(
            sorted(
                Counter(
                    str(_nested(row, "variants", variant, "reason") or "none")
                    for row in results
                    if _nested(row, "variants", variant, "status") != "accepted"
                ).items()
            )
        )
        for variant in GLOBAL_CONSTANT_JUMP_VARIANTS
    }
    hard_guard_rows_by_reason: dict[str, list[int]] = {}
    for row in results:
        raw_reasons = _nested(row, "projection_flags", "hard_guard_reasons")
        if not isinstance(raw_reasons, list):
            continue
        for reason in raw_reasons:
            hard_guard_rows_by_reason.setdefault(str(reason), []).append(int(row["row_index"]))
    summary = {
        "schema": SUMMARY_SCHEMA,
        "experiment": _EXPERIMENT,
        "stage": stage,
        "provenance": run_provenance,
        "source": {
            "stage_audio_count": STAGE_AUDIO_COUNTS[stage],
            "identity_rows": dict(identity_source),
            "selection_manifest": dict(selection_source),
            "prior_stage": dict(prior) if prior is not None else None,
        },
        "output": {
            "path": output_path.resolve(strict=True).as_posix(),
            "sha256": output_sha256,
            "row_count": len(results),
        },
        "results": {
            "result_count": len(results),
            "successful_count": sum(bool(row.get("ok")) for row in results),
            "failed_count": sum(not bool(row.get("ok")) for row in results),
            "cache_status_counts": dict(sorted(cache_statuses.items())),
            "variant_status_counts": variant_statuses,
            "variant_reason_counts": variant_reasons,
        },
        "denominators": {
            "stage_audio_count": STAGE_AUDIO_COUNTS[stage],
            "cache_valid_count": cache_valid_count,
            "projection_evaluable_count": projection_evaluable_count,
            "selected_fallback_count": fallback_count,
            "selected_fallback_rate": fallback_rate,
            "fallback_reason_counts": dict(sorted(fallback_reasons.items())),
        },
        "oracle_comparison": {
            "status": UNDEFINED_PROJECTION_ONLY,
            "comparison_eligible_count": None,
            "pure_CJ3_phase_count": None,
            "pure_CJ3_phase_coverage": None,
            "selected_safety_phase_count": None,
            "mean_phase_ratio": None,
            "p90_phase_ratio": None,
        },
        "resume": dict(resume_counts),
        "runtime": {
            "total_seconds": float(total_seconds),
            "workers": workers,
            "process_start_method": PROCESS_START_METHOD if workers > 1 else None,
            "ordered_map_chunksize": 1 if workers > 1 else None,
            "per_audio_timeout_seconds": PER_AUDIO_TIMEOUT_SECONDS,
        },
        "integration": {
            "formal_execution_ready": not INTEGRATION_BLOCKERS,
            "blockers": list(INTEGRATION_BLOCKERS),
        },
        "hard_guards": {
            "ok": not hard_guard_rows_by_reason,
            "violations": [
                {"reason": reason, "row_indices": row_indices}
                for reason, row_indices in sorted(hard_guard_rows_by_reason.items())
            ],
        },
    }
    summary = _jsonable(summary)
    _canonical_json(summary)
    return summary


def _cache_file_identity(path: Path) -> _CacheFileIdentity:
    try:
        before = path.stat()
    except FileNotFoundError:
        return _CacheFileIdentity(
            path=path.resolve(strict=False).as_posix(),
            exists=False,
            size_bytes=None,
            mtime_ns=None,
            inode=None,
            device=None,
            sha256=None,
        )
    try:
        digest = _file_sha256(path)
    except RuntimeError as exc:
        raise _CacheChangedDuringLoad(f"cache changed while hashing: {path}") from exc
    try:
        after = path.stat()
    except FileNotFoundError as exc:
        raise _CacheChangedDuringLoad(f"cache disappeared while hashing: {path}") from exc
    before_signature = (before.st_size, before.st_mtime_ns, before.st_ino, before.st_dev)
    after_signature = (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev)
    if before_signature != after_signature:
        raise _CacheChangedDuringLoad(f"cache changed while hashing: {path}")
    return _CacheFileIdentity(
        path=path.resolve(strict=True).as_posix(),
        exists=True,
        size_bytes=after.st_size,
        mtime_ns=after.st_mtime_ns,
        inode=after.st_ino,
        device=after.st_dev,
        sha256=digest,
    )


def _cache_file_identity_from_payload(
    payload: object,
    *,
    field_name: str,
) -> _CacheFileIdentity:
    if not isinstance(payload, MappingABC):
        raise ValueError(f"{field_name} must be a mapping")
    path = _required_string(payload.get("path"), f"{field_name}.path")
    exists = payload.get("exists")
    if not isinstance(exists, bool):
        raise ValueError(f"{field_name}.exists must be a bool")
    size_bytes = payload.get("size_bytes")
    mtime_ns = payload.get("mtime_ns")
    inode = payload.get("inode")
    device = payload.get("device")
    sha256 = payload.get("sha256")
    if exists:
        for numeric_name, numeric_value in (
            ("size_bytes", size_bytes),
            ("mtime_ns", mtime_ns),
            ("inode", inode),
            ("device", device),
        ):
            if isinstance(numeric_value, bool) or not isinstance(numeric_value, int):
                raise ValueError(f"{field_name}.{numeric_name} must be an integer")
        sha256 = _required_sha256_value(sha256, f"{field_name}.sha256")
    else:
        if any(value is not None for value in (size_bytes, mtime_ns, inode, device, sha256)):
            raise ValueError(f"{field_name} absent cache identity fields must be null")
    return _CacheFileIdentity(
        path=path,
        exists=exists,
        size_bytes=size_bytes,
        mtime_ns=mtime_ns,
        inode=inode,
        device=device,
        sha256=sha256,
    )


def _require_result_resume_caches_unchanged(results: Sequence[Mapping[str, Any]]) -> None:
    for row in results:
        row_index = row.get("row_index")
        components = _nested(row, "resume", "components")
        if not isinstance(components, MappingABC):
            raise ValueError(f"result row {row_index!r} resume components are missing")
        expected = _cache_file_identity_from_payload(
            components.get("cache_file"),
            field_name=f"result row {row_index!r} resume.components.cache_file",
        )
        try:
            current = _cache_file_identity(Path(expected.path))
        except _CacheChangedDuringLoad as exc:
            raise RuntimeError(
                "declared cache file changed during Experiment 004 execution: "
                f"row {row_index!r} {expected.path}"
            ) from exc
        if current != expected:
            raise RuntimeError(
                "declared cache file changed during Experiment 004 execution: "
                f"row {row_index!r} {expected.path}"
            )


@contextlib.contextmanager
def _per_audio_timeout(seconds: float) -> Iterator[None]:
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError("per-audio timeout must be positive and finite")
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(_signum: int, _frame: object) -> None:
        raise _PerAudioTimeout(f"Experiment 004 per-audio timeout after {seconds:g}s")

    previous_timer = (0.0, 0.0)
    signal.signal(signal.SIGALRM, _raise_timeout)
    try:
        previous_timer = signal.setitimer(signal.ITIMER_REAL, float(seconds))
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0.0 or previous_timer[1] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


@contextlib.contextmanager
def _exclusive_run_lock(path: Path, *, run_fingerprint: str) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Experiment 004 output is locked: {path}") from exc
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
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_canonical_json(row) for row in rows]
    data = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    _write_bytes_atomic(path, data)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    data = (_canonical_json(payload) + "\n").encode("utf-8")
    _write_bytes_atomic(path, data)


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
    payload, _sha256 = _load_json_object_with_sha256(path)
    return payload


def _load_json_object_with_sha256(path: Path) -> tuple[dict[str, Any], str]:
    stable = _read_stable_file_bytes(path)
    try:
        text = stable.data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not valid UTF-8 JSON") from exc
    try:
        payload = json.loads(text, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    _jsonable(payload)
    return payload, stable.sha256


def _load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"{path}:{line_number} blank JSONL lines are not allowed")
            try:
                payload = json.loads(raw, parse_constant=_reject_json_constant)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(_jsonable(payload))
    return rows


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, MappingABC):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            result[key] = _jsonable(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats are forbidden in Experiment 004 output")
        return float(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _StableFileBytes:
    path: str
    data: bytes
    sha256: str


def _read_stable_file_bytes(path: Path) -> _StableFileBytes:
    try:
        before_signature = _file_stat_signature(path)
    except FileNotFoundError:
        raise
    data = path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    try:
        after_signature = _file_stat_signature(path)
    except FileNotFoundError as exc:
        raise RuntimeError(f"file disappeared while reading: {path}") from exc
    if before_signature != after_signature:
        raise RuntimeError(f"file changed while reading: {path}")
    return _StableFileBytes(
        path=path.resolve(strict=True).as_posix(),
        data=data,
        sha256=sha256,
    )


def _file_sha256(path: Path) -> str:
    before_signature = _file_stat_signature(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    try:
        after_signature = _file_stat_signature(path)
    except FileNotFoundError as exc:
        raise RuntimeError(f"file disappeared while hashing: {path}") from exc
    if before_signature != after_signature:
        raise RuntimeError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _file_stat_signature(path: Path) -> tuple[int, int, int, int]:
    stat_result = path.stat()
    return (stat_result.st_size, stat_result.st_mtime_ns, stat_result.st_ino, stat_result.st_dev)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _key_set_sha256(values: set[str]) -> str:
    return _stable_json_sha256(sorted(values))


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _required_sha256_value(value: object, name: str) -> str:
    return _require_sha256(value, name)


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _required_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, MappingABC):
        raise ValueError(f"{name} must be a mapping")
    return value


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, MappingABC):
            return None
        current = current.get(key)
    return current


def _reject_path_aliases(paths: Mapping[str, Path | None]) -> None:
    resolved: dict[Path, str] = {}
    for name, path in paths.items():
        if path is None:
            continue
        candidate = path.expanduser().resolve(strict=False)
        other = resolved.get(candidate)
        if other is not None:
            raise ValueError(f"{name} aliases {other}: {candidate}")
        resolved[candidate] = name


def _reject_declared_cache_aliases(
    *,
    identities: Sequence[_IdentityRow],
    cache_config: BeatThisFramePredictionCacheConfig,
    paths: Mapping[str, Path | None],
) -> None:
    protected = {
        path.expanduser().resolve(strict=False): name
        for name, path in paths.items()
        if path is not None
    }
    for identity in identities:
        cache_path = beatthis_frame_prediction_cache_path(
            identity.cache_audio_key,
            cache_config,
        ).expanduser().resolve(strict=False)
        alias = protected.get(cache_path)
        if alias is not None:
            raise ValueError(
                f"declared cache path for row {identity.row_index} aliases {alias}: {cache_path}"
            )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _require_sources_unchanged(source_identities: Sequence[Mapping[str, Any]]) -> None:
    root = _repo_root()
    for identity in source_identities:
        path = root / str(identity["relative_path"])
        if _file_sha256(path) != identity["sha256"]:
            raise RuntimeError(f"Experiment 004 behavior source changed during execution: {path}")


def _git_provenance() -> dict[str, Any]:
    root = _repo_root()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--short", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("unable to capture Git provenance for Experiment 004") from exc
    return {"commit_sha": commit, "dirty_files": dirty}


def _environment_provenance() -> dict[str, Any]:
    try:
        torch_version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version = None
    return {
        "python_implementation": sys.implementation.name,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch_version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "accelerator": "cpu_numpy",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Timing-v3 Experiment 004 cache-only projections")
    parser.add_argument("--stage", choices=tuple(STAGE_AUDIO_COUNTS), required=True)
    parser.add_argument("--identity-rows-jsonl", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--prior-stage-summary", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_CONFIG.cache_root)
    parser.add_argument("--cache-version", default=DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_CONFIG.cache_version)
    parser.add_argument("--cache-checkpoint-path", default=DEFAULT_BEATTHIS_CHECKPOINT)
    parser.add_argument("--cache-float16", action="store_true")
    parser.add_argument("--cache-shift-ms", type=float, default=0.0)
    parser.add_argument(
        "--cache-frame-rate-hz",
        type=float,
        default=DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_CONFIG.frame_rate_hz,
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = run_exp004_projection(
        stage=args.stage,
        identity_rows_jsonl_path=args.identity_rows_jsonl,
        selection_manifest_path=args.selection_manifest,
        prior_stage_summary_path=args.prior_stage_summary,
        cache_root=args.cache_root,
        cache_version=args.cache_version,
        checkpoint_path=args.cache_checkpoint_path,
        float16=args.cache_float16,
        shift_ms=args.cache_shift_ms,
        frame_rate_hz=args.cache_frame_rate_hz,
        output_jsonl_path=args.output_jsonl,
        summary_json_path=args.summary_json,
        checkpoint_every=args.checkpoint_every,
        retry_failures=args.retry_failures,
        workers=args.workers,
    )
    print(_canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
