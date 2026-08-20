from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import signal
import threading
import time
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from pulsefield_model.timing.diagnostics import compare_timing_grids
from pulsefield_model.timing.canonicalization import (
    TIMING_CANONICALIZATION_BPM_80_160,
    canonicalize_timing_grid,
)
from pulsefield_model.timing.evaluation.drift import compare_timing_grid_drift
from pulsefield_model.timing.evaluation.inventory import TIMING_V3_INVENTORY_AUDIO_ROW_SCHEMA
from pulsefield_model.timing.grid_fitting import GridFitter, GridFitterConfig
from pulsefield_model.timing.grid_fitting.types import TimingFitResult
from pulsefield_model.timing.providers.beatthis import DEFAULT_BEATTHIS_CHECKPOINT
from pulsefield_model.timing.providers.beatthis_cache import (
    DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_CONFIG,
    BeatThisFramePredictionCacheConfig,
    beatthis_audio_cache_key,
    beatthis_frame_prediction_cache_path,
    load_beatthis_frame_prediction_cache,
)
from pulsefield_model.timing.providers.oracle import oracle_timing_grid_from_beatmap
from pulsefield_model.timing.schema import FittedTimingGrid, TimingSegment


RESULT_SCHEMA = "pulsefield_model.timing_v3_cache_backed_v2_baseline_result_v2"
SUMMARY_SCHEMA = "pulsefield_model.timing_v3_cache_backed_v2_baseline_summary_v2"
RESUME_SCHEMA = "pulsefield_model.timing_v3_cache_backed_v2_baseline_resume_v2"
MISSING_AUDIO_CACHE_KEY_PREFIX = "missing-audio:"


@dataclass(frozen=True)
class BaselineAudioRow:
    audio_key: str
    audio_path: Path | None
    beatmap_paths: tuple[Path, ...]
    source_line_numbers: tuple[int, ...]
    supplied_audio_key: bool
    evaluation_strata: Mapping[str, Any]


@dataclass(frozen=True)
class _BaselineEvalJob:
    row_index: int
    row: BaselineAudioRow
    resume_context: Mapping[str, Any]


def run_cache_backed_v2_baseline(
    *,
    inventory_path: str | Path,
    output_jsonl_path: str | Path,
    summary_json_path: str | Path | None = None,
    dataset_root: str | Path | None = None,
    cache_root: str | Path = DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_CONFIG.cache_root,
    cache_version: str = DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_CONFIG.cache_version,
    checkpoint_path: str = DEFAULT_BEATTHIS_CHECKPOINT,
    float16: bool = False,
    shift_ms: float = 0.0,
    limit: int | None = None,
    retry_failures: bool = False,
    timeout_seconds: float | None = None,
    progress_every: int = 25,
    checkpoint_every: int = 25,
    workers: int = 1,
    fitter: GridFitter | None = None,
) -> dict[str, Any]:
    """Run the unchanged v2 GridFitter against one existing BeatThis cache.

    The inventory is JSONL and intentionally tolerant. Each row may provide
    either an explicit ``audio_key`` matching the BeatThis cache, or enough audio
    path information to compute the cache key with ``beatthis_audio_cache_key``.
    Rows with the same audio key are merged into a single audio-level job.
    """

    started = time.time()
    inventory_path = Path(inventory_path)
    output_jsonl_path = Path(output_jsonl_path)
    summary_json_path = (
        Path(summary_json_path)
        if summary_json_path is not None
        else output_jsonl_path.with_suffix(output_jsonl_path.suffix + ".summary.json")
    )
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be non-negative, got {limit!r}")
    if progress_every < 0:
        raise ValueError(f"progress_every must be non-negative, got {progress_every!r}")
    if checkpoint_every < 0:
        raise ValueError(f"checkpoint_every must be non-negative, got {checkpoint_every!r}")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError(f"workers must be a positive int, got {workers!r}")
    if workers > 1 and fitter is not None:
        raise ValueError("workers > 1 requires fitter=None so worker processes can build default GridFitter")

    cache_config = BeatThisFramePredictionCacheConfig(
        cache_root=Path(cache_root),
        cache_version=cache_version,
        checkpoint_path=checkpoint_path,
        float16=float16,
        shift_ms=shift_ms,
    )
    rows = load_audio_inventory_jsonl(
        inventory_path,
        dataset_root=Path(dataset_root) if dataset_root is not None else None,
        limit=limit,
    )
    grid_fitter = fitter if fitter is not None else GridFitter()
    inventory_sha256 = _file_sha256(inventory_path)
    fitter_provenance = _fitter_provenance(grid_fitter)
    resume_by_key = {
        row.audio_key: _resume_payload(
            row,
            row_index=index,
            inventory_sha256=inventory_sha256,
            cache_config=cache_config,
            fitter_provenance=fitter_provenance,
        )
        for index, row in enumerate(rows)
    }
    row_keys = {row.audio_key for row in rows}
    existing_results: dict[str, dict[str, Any]] = {}
    stale_existing_count = 0
    for result in _read_result_jsonl(output_jsonl_path):
        audio_key = result.get("audio_key")
        if not isinstance(audio_key, str) or audio_key not in row_keys:
            continue
        if _result_matches_resume(result, resume_by_key[audio_key]):
            existing_results[audio_key] = result
        else:
            stale_existing_count += 1

    results_by_key: dict[str, dict[str, Any]] = dict(existing_results)
    processed_count = 0
    skipped_success_count = 0
    skipped_failure_count = 0
    jobs: list[_BaselineEvalJob] = []

    for index, row in enumerate(rows):
        resume_context = resume_by_key[row.audio_key]
        existing = results_by_key.get(row.audio_key)
        if existing is not None:
            existing_ok = bool(existing.get("ok"))
            if existing_ok or not retry_failures:
                if existing_ok:
                    skipped_success_count += 1
                else:
                    skipped_failure_count += 1
                continue

        jobs.append(_BaselineEvalJob(row_index=index, row=row, resume_context=resume_context))

    if workers == 1:
        processed_count = _run_jobs_sequential(
            jobs,
            rows=rows,
            results_by_key=results_by_key,
            output_jsonl_path=output_jsonl_path,
            cache_config=cache_config,
            fitter=grid_fitter,
            timeout_seconds=timeout_seconds,
            checkpoint_every=checkpoint_every,
            progress_every=progress_every,
        )
    else:
        processed_count = _run_jobs_parallel(
            jobs,
            rows=rows,
            results_by_key=results_by_key,
            output_jsonl_path=output_jsonl_path,
            cache_config=cache_config,
            timeout_seconds=timeout_seconds,
            checkpoint_every=checkpoint_every,
            progress_every=progress_every,
            workers=workers,
        )

    ordered_results = _ordered_results(rows, results_by_key)
    _write_result_jsonl_atomic(output_jsonl_path, ordered_results)
    summary = _summary_payload(
        rows=rows,
        results=ordered_results,
        inventory_path=inventory_path,
        output_jsonl_path=output_jsonl_path,
        summary_json_path=summary_json_path,
        cache_config=cache_config,
        grid_fitter=grid_fitter,
        started_at_unix=started,
        processed_count=processed_count,
        skipped_success_count=skipped_success_count,
        skipped_failure_count=skipped_failure_count,
        stale_existing_count=stale_existing_count,
        retry_failures=retry_failures,
        timeout_seconds=timeout_seconds,
        checkpoint_every=checkpoint_every,
        workers=workers,
        limit=limit,
        inventory_sha256=inventory_sha256,
        fitter_provenance=fitter_provenance,
    )
    _write_json_atomic(summary_json_path, summary)
    return summary


def load_audio_inventory_jsonl(
    inventory_path: str | Path,
    *,
    dataset_root: Path | None = None,
    limit: int | None = None,
) -> list[BaselineAudioRow]:
    inventory_path = Path(inventory_path)
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be non-negative, got {limit!r}")

    rows_by_key: dict[str, BaselineAudioRow] = {}
    with inventory_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{inventory_path}:{line_number} is not valid JSON") from exc
            if not isinstance(payload, MappingABC):
                raise ValueError(f"{inventory_path}:{line_number} must be a JSON object")

            row = _normalize_inventory_payload(
                payload,
                inventory_path=inventory_path,
                line_number=line_number,
                dataset_root=dataset_root,
            )
            existing = rows_by_key.get(row.audio_key)
            if existing is None:
                rows_by_key[row.audio_key] = row
                continue
            rows_by_key[row.audio_key] = BaselineAudioRow(
                audio_key=row.audio_key,
                audio_path=existing.audio_path or row.audio_path,
                beatmap_paths=_ordered_unique_paths((*existing.beatmap_paths, *row.beatmap_paths)),
                source_line_numbers=tuple(sorted((*existing.source_line_numbers, *row.source_line_numbers))),
                supplied_audio_key=existing.supplied_audio_key or row.supplied_audio_key,
                evaluation_strata=_merge_evaluation_strata(existing.evaluation_strata, row.evaluation_strata),
            )

    rows = sorted(
        rows_by_key.values(),
        key=lambda row: (row.audio_key, row.audio_path.as_posix() if row.audio_path is not None else ""),
    )
    if limit is not None:
        rows = rows[:limit]
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    summary = run_cache_backed_v2_baseline(
        inventory_path=args.inventory,
        output_jsonl_path=args.output_jsonl,
        summary_json_path=args.summary_json,
        dataset_root=args.dataset_root,
        cache_root=args.cache_root,
        cache_version=args.cache_version,
        checkpoint_path=args.checkpoint,
        float16=args.float16,
        shift_ms=args.shift_ms,
        limit=args.limit,
        retry_failures=args.retry_failures,
        timeout_seconds=args.timeout_seconds,
        progress_every=args.progress_every,
        checkpoint_every=args.checkpoint_every,
        workers=args.workers,
    )
    if args.json:
        print(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True))
    else:
        print(_format_summary(summary))
    return 1 if int(summary["failures"]["failed_audio_count"]) > 0 else 0


def _run_jobs_sequential(
    jobs: Sequence[_BaselineEvalJob],
    *,
    rows: Sequence[BaselineAudioRow],
    results_by_key: dict[str, dict[str, Any]],
    output_jsonl_path: Path,
    cache_config: BeatThisFramePredictionCacheConfig,
    fitter: GridFitter,
    timeout_seconds: float | None,
    checkpoint_every: int,
    progress_every: int,
) -> int:
    processed_count = 0
    dirty_result_count = 0
    for job in jobs:
        result = _evaluate_audio_row_in_current_process(
            job,
            cache_config=cache_config,
            fitter=fitter,
            timeout_seconds=timeout_seconds,
        )
        processed_count += 1
        dirty_result_count = _record_completed_result(
            result,
            rows=rows,
            results_by_key=results_by_key,
            output_jsonl_path=output_jsonl_path,
            dirty_result_count=dirty_result_count,
            checkpoint_every=checkpoint_every,
        )
        if progress_every > 0 and (processed_count == 1 or processed_count % progress_every == 0):
            _print_progress(processed_count=processed_count, total_count=len(rows), result=result)
    return processed_count


def _run_jobs_parallel(
    jobs: Sequence[_BaselineEvalJob],
    *,
    rows: Sequence[BaselineAudioRow],
    results_by_key: dict[str, dict[str, Any]],
    output_jsonl_path: Path,
    cache_config: BeatThisFramePredictionCacheConfig,
    timeout_seconds: float | None,
    checkpoint_every: int,
    progress_every: int,
    workers: int,
) -> int:
    if not jobs:
        return 0

    max_in_flight = max(1, workers * 2)
    processed_count = 0
    dirty_result_count = 0
    job_iter = iter(jobs)
    futures: dict[concurrent.futures.Future[dict[str, Any]], _BaselineEvalJob] = {}

    def submit_next(executor: concurrent.futures.ProcessPoolExecutor) -> bool:
        try:
            job = next(job_iter)
        except StopIteration:
            return False
        future = executor.submit(_evaluate_audio_row_worker, job, cache_config, timeout_seconds)
        futures[future] = job
        return True

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for _ in range(min(max_in_flight, len(jobs))):
                submit_next(executor)

            while futures:
                done, _ = concurrent.futures.wait(
                    futures,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    job = futures.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001 - worker process failures remain row-local.
                        result = _failure_result(
                            job.row,
                            row_index=job.row_index,
                            stage="worker",
                            error=exc,
                            cache_config=cache_config,
                            started_at_unix=time.time(),
                            timeout_seconds=timeout_seconds,
                            resume_context=job.resume_context,
                        )
                    processed_count += 1
                    dirty_result_count = _record_completed_result(
                        result,
                        rows=rows,
                        results_by_key=results_by_key,
                        output_jsonl_path=output_jsonl_path,
                        dirty_result_count=dirty_result_count,
                        checkpoint_every=checkpoint_every,
                    )
                    if progress_every > 0 and (processed_count == 1 or processed_count % progress_every == 0):
                        _print_progress(processed_count=processed_count, total_count=len(rows), result=result)
                    while len(futures) < max_in_flight and submit_next(executor):
                        pass
    except (PermissionError, NotImplementedError) as exc:
        raise _process_pool_unavailable_error(exc) from exc
    return processed_count


def _process_pool_unavailable_error(exc: BaseException) -> RuntimeError:
    return RuntimeError(
        "workers > 1 requires Python multiprocessing process-pool semaphore support. "
        "Run with workers=1 or allow OS semaphore access for this sandbox/runtime."
    )


def _record_completed_result(
    result: dict[str, Any],
    *,
    rows: Sequence[BaselineAudioRow],
    results_by_key: dict[str, dict[str, Any]],
    output_jsonl_path: Path,
    dirty_result_count: int,
    checkpoint_every: int,
) -> int:
    results_by_key[str(result["audio_key"])] = result
    dirty_result_count += 1
    if checkpoint_every > 0 and dirty_result_count >= checkpoint_every:
        _write_result_jsonl_atomic(output_jsonl_path, _ordered_results(rows, results_by_key))
        return 0
    return dirty_result_count


def _evaluate_audio_row_worker(
    job: _BaselineEvalJob,
    cache_config: BeatThisFramePredictionCacheConfig,
    timeout_seconds: float | None,
) -> dict[str, Any]:
    return _evaluate_audio_row_in_current_process(
        job,
        cache_config=cache_config,
        fitter=GridFitter(),
        timeout_seconds=timeout_seconds,
    )


def _evaluate_audio_row_in_current_process(
    job: _BaselineEvalJob,
    *,
    cache_config: BeatThisFramePredictionCacheConfig,
    fitter: GridFitter,
    timeout_seconds: float | None,
) -> dict[str, Any]:
    row_started = time.time()
    try:
        with _row_timeout(timeout_seconds):
            return _evaluate_audio_row(
                job.row,
                row_index=job.row_index,
                cache_config=cache_config,
                fitter=fitter,
                timeout_seconds=timeout_seconds,
                resume_context=job.resume_context,
            )
    except TimeoutError as exc:
        return _failure_result(
            job.row,
            row_index=job.row_index,
            stage="timeout",
            error=exc,
            cache_config=cache_config,
            started_at_unix=row_started,
            timeout_seconds=timeout_seconds,
            resume_context=job.resume_context,
        )
    except Exception as exc:  # noqa: BLE001 - eval runners must isolate row failures.
        return _failure_result(
            job.row,
            row_index=job.row_index,
            stage="unexpected",
            error=exc,
            cache_config=cache_config,
            started_at_unix=row_started,
            timeout_seconds=timeout_seconds,
            resume_context=job.resume_context,
        )


def _evaluate_audio_row(
    row: BaselineAudioRow,
    *,
    row_index: int,
    cache_config: BeatThisFramePredictionCacheConfig,
    fitter: GridFitter,
    timeout_seconds: float | None,
    resume_context: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.time()
    cache_path = beatthis_frame_prediction_cache_path(row.audio_key, cache_config)
    try:
        cache_load_started = time.time()
        prediction = load_beatthis_frame_prediction_cache(row.audio_key, cache_config)
        cache_load_seconds = time.time() - cache_load_started
        if prediction is None:
            raise FileNotFoundError(f"BeatThis cache not found: {cache_path}")
    except TimeoutError:
        raise
    except Exception as exc:  # noqa: BLE001 - cache failures are row-level eval failures.
        return _failure_result(
            row,
            row_index=row_index,
            stage="cache",
            error=exc,
            cache_config=cache_config,
            started_at_unix=started,
            timeout_seconds=timeout_seconds,
            resume_context=resume_context,
        )

    try:
        fit_started = time.time()
        fit_result = fitter.fit(prediction)
        fit_seconds = time.time() - fit_started
    except TimeoutError:
        raise
    except Exception as exc:  # noqa: BLE001 - fitter failures are row-level eval failures.
        result = _failure_result(
            row,
            row_index=row_index,
            stage="fit",
            error=exc,
            cache_config=cache_config,
            started_at_unix=started,
            timeout_seconds=timeout_seconds,
            resume_context=resume_context,
        )
        result["prediction"] = _prediction_payload(prediction)
        result["runtime"]["cache_load_seconds"] = cache_load_seconds
        return result

    compare_started = time.time()
    comparisons = _compare_to_beatmaps(fit_result.grid, row.beatmap_paths, frame_count=prediction.frame_count)
    compare_seconds = time.time() - compare_started
    ok_comparisons = [comparison for comparison in comparisons if comparison.get("ok")]
    if not ok_comparisons:
        result = _failure_result(
            row,
            row_index=row_index,
            stage="compare",
            error=ValueError("no successful .osu redline comparison"),
            cache_config=cache_config,
            started_at_unix=started,
            timeout_seconds=timeout_seconds,
            resume_context=resume_context,
        )
        result["prediction"] = _prediction_payload(prediction)
        result["fit"] = _fit_payload(fit_result)
        result["comparisons"] = comparisons
        result["runtime"].update(
            {
                "cache_load_seconds": cache_load_seconds,
                "fit_seconds": fit_seconds,
                "compare_seconds": compare_seconds,
            }
        )
        result["runtime"]["total_seconds"] = time.time() - started
        return result

    finished = time.time()
    return {
        "schema": RESULT_SCHEMA,
        "resume": resume_context,
        "ok": True,
        "audio_key": row.audio_key,
        "row_index": row_index,
        "source_line_numbers": list(row.source_line_numbers),
        "supplied_audio_key": row.supplied_audio_key,
        "evaluation_strata": _json_safe(row.evaluation_strata),
        "audio_path": _path_or_none(row.audio_path),
        "beatmap_paths": [path.as_posix() for path in row.beatmap_paths],
        "cache": _cache_payload(cache_config, row.audio_key),
        "prediction": _prediction_payload(prediction),
        "fit": _fit_payload(fit_result),
        "comparisons": comparisons,
        "paired_metrics": _paired_metrics_payload(ok_comparisons),
        "runtime": {
            "started_at_unix": started,
            "finished_at_unix": finished,
            "cache_load_seconds": cache_load_seconds,
            "fit_seconds": fit_seconds,
            "compare_seconds": compare_seconds,
            "total_seconds": finished - started,
            "timeout_seconds": timeout_seconds,
        },
        "failure_stage": None,
        "error_type": None,
        "error": None,
    }


def _compare_to_beatmaps(
    predicted_grid: FittedTimingGrid,
    beatmap_paths: Sequence[Path],
    *,
    frame_count: int,
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for beatmap_path in beatmap_paths:
        started = time.time()
        try:
            oracle_grid = oracle_timing_grid_from_beatmap(beatmap_path)
            comparison = compare_timing_grids(
                predicted_grid,
                oracle_grid,
                frame_count=frame_count,
            )
            raw_drift_comparison = compare_timing_grid_drift(
                predicted_grid,
                oracle_grid,
                frame_count=frame_count,
            )
            alias_drift_comparison = compare_timing_grid_drift(
                canonicalize_timing_grid(
                    predicted_grid,
                    canonicalization=TIMING_CANONICALIZATION_BPM_80_160,
                ),
                canonicalize_timing_grid(
                    oracle_grid,
                    canonicalization=TIMING_CANONICALIZATION_BPM_80_160,
                ),
                frame_count=frame_count,
            )
            raw_drift_metrics = _drift_metrics_payload(raw_drift_comparison)
            alias_drift_metrics = _drift_metrics_payload(alias_drift_comparison)
            flat_drift_metrics = {
                **{f"raw_{key}": value for key, value in raw_drift_metrics.items()},
                **{f"alias_{key}": value for key, value in alias_drift_metrics.items()},
            }
            comparisons.append(
                {
                    "ok": True,
                    "beatmap_path": beatmap_path.as_posix(),
                    "oracle_segments": _segments_payload(oracle_grid),
                    "metrics": {**_json_safe(asdict(comparison)), **flat_drift_metrics},
                    "drift_metrics": {
                        "raw": raw_drift_metrics,
                        "canonical_bpm_80_160": alias_drift_metrics,
                    },
                    "seconds": time.time() - started,
                    "error_type": None,
                    "error": None,
                }
            )
        except TimeoutError:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad difficulty must not stop the audio row.
            comparisons.append(
                {
                    "ok": False,
                    "beatmap_path": beatmap_path.as_posix(),
                    "oracle_segments": [],
                    "metrics": None,
                    "seconds": time.time() - started,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                }
            )
    return comparisons


def _drift_metrics_payload(comparison: object) -> dict[str, Any]:
    payload = _json_safe(asdict(comparison))
    payload["abs_endpoint_relative_drift_ms"] = abs(
        float(payload["endpoint_relative_drift_ms"])
    )
    payload["abs_drift_slope_ms_per_minute"] = abs(
        float(payload["drift_slope_ms_per_minute"])
    )
    return payload


def _failure_result(
    row: BaselineAudioRow,
    *,
    row_index: int,
    stage: str,
    error: BaseException,
    cache_config: BeatThisFramePredictionCacheConfig,
    started_at_unix: float,
    timeout_seconds: float | None = None,
    resume_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    finished = time.time()
    return {
        "schema": RESULT_SCHEMA,
        "resume": resume_context,
        "ok": False,
        "audio_key": row.audio_key,
        "row_index": row_index,
        "source_line_numbers": list(row.source_line_numbers),
        "supplied_audio_key": row.supplied_audio_key,
        "evaluation_strata": _json_safe(row.evaluation_strata),
        "audio_path": _path_or_none(row.audio_path),
        "beatmap_paths": [path.as_posix() for path in row.beatmap_paths],
        "cache": _cache_payload(cache_config, row.audio_key),
        "prediction": None,
        "fit": None,
        "comparisons": [],
        "paired_metrics": None,
        "runtime": {
            "started_at_unix": started_at_unix,
            "finished_at_unix": finished,
            "cache_load_seconds": None,
            "fit_seconds": None,
            "compare_seconds": None,
            "total_seconds": finished - started_at_unix,
            "timeout_seconds": timeout_seconds,
        },
        "failure_stage": stage,
        "error_type": error.__class__.__name__,
        "error": str(error),
    }


def _summary_payload(
    *,
    rows: Sequence[BaselineAudioRow],
    results: Sequence[Mapping[str, Any]],
    inventory_path: Path,
    output_jsonl_path: Path,
    summary_json_path: Path,
    cache_config: BeatThisFramePredictionCacheConfig,
    grid_fitter: GridFitter,
    started_at_unix: float,
    processed_count: int,
    skipped_success_count: int,
    skipped_failure_count: int,
    stale_existing_count: int,
    retry_failures: bool,
    timeout_seconds: float | None,
    checkpoint_every: int,
    workers: int,
    limit: int | None,
    inventory_sha256: str,
    fitter_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    finished = time.time()
    comparison_failures = [
        comparison
        for result in results
        for comparison in result.get("comparisons", [])
        if isinstance(comparison, MappingABC) and not comparison.get("ok")
    ]
    ok_results = [result for result in results if result.get("ok")]
    failed_results = [result for result in results if not result.get("ok")]
    paired_metrics = [
        comparison["metrics"]
        for result in results
        for comparison in result.get("comparisons", [])
        if isinstance(comparison, MappingABC)
        and comparison.get("ok")
        and isinstance(comparison.get("metrics"), MappingABC)
    ]
    failure_stage_counts: dict[str, int] = {}
    for result in failed_results:
        stage = str(result.get("failure_stage") or "unknown")
        failure_stage_counts[stage] = failure_stage_counts.get(stage, 0) + 1

    audio_result_counts = _audio_result_counts(results)
    audio_group_metrics = _aggregate_audio_group_metrics(results)
    difficulty_comparison_metrics = _aggregate_difficulty_comparison_metrics(results, paired_metrics)
    return _json_safe(
        {
            "schema": SUMMARY_SCHEMA,
            "started_at_unix": started_at_unix,
            "finished_at_unix": finished,
            "seconds": finished - started_at_unix,
            "source": {
                "inventory_path": inventory_path.as_posix(),
                "inventory_sha256": _file_sha256(inventory_path),
                "inventory_schema": TIMING_V3_INVENTORY_AUDIO_ROW_SCHEMA,
                "output_jsonl_path": output_jsonl_path.as_posix(),
                "summary_json_path": summary_json_path.as_posix(),
                "unique_audio_count": len(rows),
                "map_reference_count": sum(len(row.beatmap_paths) for row in rows),
                "limit": limit,
            },
            "cache": _cache_config_payload(cache_config),
            "fitter": _fitter_provenance(grid_fitter),
            "run": {
                "processed_count": processed_count,
                "skipped_success_count": skipped_success_count,
                "skipped_failure_count": skipped_failure_count,
                "stale_existing_count": stale_existing_count,
                "retry_failures": retry_failures,
                "timeout_seconds": timeout_seconds,
                "timeout_enforced": _timeout_supported(timeout_seconds),
                "checkpoint_every": checkpoint_every,
                "workers": workers,
                "parallel": workers > 1,
                "max_in_flight": workers * 2 if workers > 1 else 1,
                "resume_schema": RESUME_SCHEMA,
                "inventory_sha256": inventory_sha256,
                "fitter_provenance": fitter_provenance,
            },
            "results": {
                "result_count": len(results),
                "successful_audio_count": len(ok_results),
                "failed_audio_count": len(failed_results),
                **audio_result_counts,
                "paired_comparison_count": len(paired_metrics),
                "comparison_failure_count": len(comparison_failures),
            },
            "failures": {
                "failed_audio_count": len(failed_results),
                "fit_failure_audio_count": audio_result_counts["fit_failure_audio_count"],
                "comparator_unavailable_audio_count": audio_result_counts["comparator_unavailable_audio_count"],
                "comparison_failure_count": len(comparison_failures),
                "stage_counts": dict(sorted(failure_stage_counts.items())),
            },
            "metrics": audio_group_metrics,
            "audio_group_metrics": audio_group_metrics,
            "difficulty_comparison_metrics": difficulty_comparison_metrics,
            "stratified": _stratified_payload(rows, results),
            "runtime": _aggregate_runtime(results),
        }
    )


def _audio_result_counts(results: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    fit_success = sum(1 for result in results if result.get("fit") is not None)
    fit_failure = sum(1 for result in results if result.get("fit") is None)
    comparison_attempted = sum(
        1
        for result in results
        if result.get("fit") is not None and len(result.get("comparisons", [])) > 0
    )
    comparison_eligible = sum(
        1
        for result in results
        if any(
            isinstance(comparison, MappingABC) and comparison.get("ok")
            for comparison in result.get("comparisons", [])
        )
    )
    comparator_unavailable = sum(
        1
        for result in results
        if result.get("fit") is not None
        and not any(
            isinstance(comparison, MappingABC) and comparison.get("ok")
            for comparison in result.get("comparisons", [])
        )
    )
    return {
        "fit_success_audio_count": fit_success,
        "fit_failure_audio_count": fit_failure,
        "comparison_attempted_audio_count": comparison_attempted,
        "comparison_eligible_audio_count": comparison_eligible,
        "comparator_unavailable_audio_count": comparator_unavailable,
    }


def _stratified_payload(
    rows: Sequence[BaselineAudioRow],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    results_by_key = {
        str(result["audio_key"]): result
        for result in results
        if isinstance(result.get("audio_key"), str)
    }
    return {
        "pilot_stratum": _stratified_dimension(rows, results_by_key, field_name="pilot_stratum"),
        "label_stratum": _stratified_dimension(rows, results_by_key, field_name="label_stratum"),
    }


def _stratified_dimension(
    rows: Sequence[BaselineAudioRow],
    results_by_key: Mapping[str, Mapping[str, Any]],
    *,
    field_name: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[BaselineAudioRow]] = {}
    values_by_key: dict[str, Any] = {}
    for row in rows:
        value = _json_safe(row.evaluation_strata.get(field_name))
        key = _stratum_key(value)
        grouped.setdefault(key, []).append(row)
        values_by_key[key] = value

    entries: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: _stratum_sort_key(values_by_key[item])):
        group_rows = grouped[key]
        group_results = [
            result
            for row in group_rows
            if isinstance((result := results_by_key.get(row.audio_key)), MappingABC)
        ]
        ok_count = sum(1 for result in group_results if result.get("ok"))
        failed_count = sum(1 for result in group_results if not result.get("ok"))
        entries.append(
            {
                "value": values_by_key[key],
                "audio_count": len(group_rows),
                "ok": ok_count,
                "failed": failed_count,
                "missing_result_count": len(group_rows) - ok_count - failed_count,
                **_audio_result_counts(group_results),
                "metrics": _aggregate_audio_group_metrics(group_results),
            }
        )
    return entries


def _stratum_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _stratum_sort_key(value: Any) -> tuple[int, str]:
    if value is None:
        return (0, "")
    return (1, _stratum_key(value))


def _aggregate_audio_group_metrics(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "weighting": "one successful audio group; per-audio value is the median across successful .osu comparisons",
        "mean_phase_error_ms": _stats(_audio_group_metric_values(results, "mean_phase_error_ms")),
        "max_phase_error_ms": _stats(_audio_group_metric_values(results, "max_phase_error_ms")),
        "mean_phase_error_beats": _stats(_audio_group_metric_values(results, "mean_phase_error_beats")),
        "local_bpm_alias_mae": _stats(_audio_group_metric_values(results, "local_bpm_alias_mae")),
        "local_bpm_mae": _stats(_audio_group_metric_values(results, "local_bpm_mae")),
        "beat_pulse_mae": _stats(_audio_group_metric_values(results, "beat_pulse_mae")),
        "raw_abs_endpoint_relative_drift_ms": _stats(
            _audio_group_metric_values(results, "raw_abs_endpoint_relative_drift_ms")
        ),
        "alias_abs_endpoint_relative_drift_ms": _stats(
            _audio_group_metric_values(results, "alias_abs_endpoint_relative_drift_ms")
        ),
        "raw_max_abs_prefix_relative_drift_ms": _stats(
            _audio_group_metric_values(results, "raw_max_abs_prefix_relative_drift_ms")
        ),
        "alias_max_abs_prefix_relative_drift_ms": _stats(
            _audio_group_metric_values(results, "alias_max_abs_prefix_relative_drift_ms")
        ),
        "raw_abs_drift_slope_ms_per_minute": _stats(
            _audio_group_metric_values(results, "raw_abs_drift_slope_ms_per_minute")
        ),
        "alias_abs_drift_slope_ms_per_minute": _stats(
            _audio_group_metric_values(results, "alias_abs_drift_slope_ms_per_minute")
        ),
        "alias_p90_abs_30s_relative_drift_ms": _stats(
            _audio_group_metric_values(results, "alias_p90_abs_30s_relative_drift_ms")
        ),
        "alias_p90_abs_60s_relative_drift_ms": _stats(
            _audio_group_metric_values(results, "alias_p90_abs_60s_relative_drift_ms")
        ),
        "raw_max_predicted_boundary_discontinuity_ms": _stats(
            _audio_group_metric_values(results, "raw_max_predicted_boundary_discontinuity_ms")
        ),
        "alias_max_predicted_boundary_discontinuity_ms": _stats(
            _audio_group_metric_values(results, "alias_max_predicted_boundary_discontinuity_ms")
        ),
        "predicted_segment_count": _stats(
            [
                _float_or_none(fit.get("predicted_segment_count"))
                for result in results
                if result.get("ok") and isinstance((fit := result.get("fit")), MappingABC)
            ]
        ),
        "canonical_oracle_segment_count": _stats(
            _audio_group_metric_values(results, "canonical_oracle_segment_count")
        ),
        "abs_canonical_segment_count_delta": _stats(
            _audio_group_metric_values(results, "abs_canonical_segment_count_delta")
        ),
    }


def _aggregate_difficulty_comparison_metrics(
    results: Sequence[Mapping[str, Any]],
    paired_metrics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "weighting": "one successful .osu difficulty comparison",
        "mean_phase_error_ms": _stats([_float_or_none(metric.get("mean_phase_error_ms")) for metric in paired_metrics]),
        "max_phase_error_ms": _stats([_float_or_none(metric.get("max_phase_error_ms")) for metric in paired_metrics]),
        "mean_phase_error_beats": _stats(
            [_float_or_none(metric.get("mean_phase_error_beats")) for metric in paired_metrics]
        ),
        "local_bpm_alias_mae": _stats(
            [_float_or_none(metric.get("local_bpm_alias_mae")) for metric in paired_metrics]
        ),
        "local_bpm_mae": _stats([_float_or_none(metric.get("local_bpm_mae")) for metric in paired_metrics]),
        "beat_pulse_mae": _stats([_float_or_none(metric.get("beat_pulse_mae")) for metric in paired_metrics]),
        "raw_abs_endpoint_relative_drift_ms": _stats(
            [_float_or_none(metric.get("raw_abs_endpoint_relative_drift_ms")) for metric in paired_metrics]
        ),
        "alias_abs_endpoint_relative_drift_ms": _stats(
            [_float_or_none(metric.get("alias_abs_endpoint_relative_drift_ms")) for metric in paired_metrics]
        ),
        "raw_max_abs_prefix_relative_drift_ms": _stats(
            [_float_or_none(metric.get("raw_max_abs_prefix_relative_drift_ms")) for metric in paired_metrics]
        ),
        "alias_max_abs_prefix_relative_drift_ms": _stats(
            [_float_or_none(metric.get("alias_max_abs_prefix_relative_drift_ms")) for metric in paired_metrics]
        ),
        "raw_abs_drift_slope_ms_per_minute": _stats(
            [_float_or_none(metric.get("raw_abs_drift_slope_ms_per_minute")) for metric in paired_metrics]
        ),
        "alias_abs_drift_slope_ms_per_minute": _stats(
            [_float_or_none(metric.get("alias_abs_drift_slope_ms_per_minute")) for metric in paired_metrics]
        ),
        "alias_p90_abs_30s_relative_drift_ms": _stats(
            [_float_or_none(metric.get("alias_p90_abs_30s_relative_drift_ms")) for metric in paired_metrics]
        ),
        "alias_p90_abs_60s_relative_drift_ms": _stats(
            [_float_or_none(metric.get("alias_p90_abs_60s_relative_drift_ms")) for metric in paired_metrics]
        ),
        "raw_max_predicted_boundary_discontinuity_ms": _stats(
            [_float_or_none(metric.get("raw_max_predicted_boundary_discontinuity_ms")) for metric in paired_metrics]
        ),
        "alias_max_predicted_boundary_discontinuity_ms": _stats(
            [_float_or_none(metric.get("alias_max_predicted_boundary_discontinuity_ms")) for metric in paired_metrics]
        ),
        "canonical_oracle_segment_count": _stats(
            [_float_or_none(metric.get("canonical_oracle_segment_count")) for metric in paired_metrics]
        ),
        "abs_canonical_segment_count_delta": _stats(
            [_float_or_none(metric.get("abs_canonical_segment_count_delta")) for metric in paired_metrics]
        ),
    }


def _audio_group_metric_values(results: Sequence[Mapping[str, Any]], metric_name: str) -> list[float | None]:
    values: list[float | None] = []
    for result in results:
        if not result.get("ok"):
            continue
        comparison_values = [
            _float_or_none(metrics.get(metric_name))
            for comparison in result.get("comparisons", [])
            if isinstance(comparison, MappingABC)
            and comparison.get("ok")
            and isinstance((metrics := comparison.get("metrics")), MappingABC)
        ]
        finite_values = [value for value in comparison_values if value is not None and math.isfinite(value)]
        if finite_values:
            values.append(float(np.median(np.asarray(finite_values, dtype=np.float64))))
    return values


def _aggregate_runtime(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    runtimes = [
        runtime
        for result in results
        if isinstance((runtime := result.get("runtime", {})), MappingABC)
    ]
    return {
        "total_seconds": _stats([_float_or_none(runtime.get("total_seconds")) for runtime in runtimes]),
        "cache_load_seconds": _stats([_float_or_none(runtime.get("cache_load_seconds")) for runtime in runtimes]),
        "fit_seconds": _stats([_float_or_none(runtime.get("fit_seconds")) for runtime in runtimes]),
        "compare_seconds": _stats([_float_or_none(runtime.get("compare_seconds")) for runtime in runtimes]),
    }


def _paired_metrics_payload(ok_comparisons: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [
        comparison["metrics"]
        for comparison in ok_comparisons
        if isinstance(comparison.get("metrics"), MappingABC)
    ]
    return {
        "comparison_count": len(metrics),
        "mean_phase_error_ms": _stats([_float_or_none(metric.get("mean_phase_error_ms")) for metric in metrics]),
        "local_bpm_alias_mae": _stats([_float_or_none(metric.get("local_bpm_alias_mae")) for metric in metrics]),
        "canonical_oracle_segment_count": _stats(
            [_float_or_none(metric.get("canonical_oracle_segment_count")) for metric in metrics]
        ),
        "abs_canonical_segment_count_delta": _stats(
            [_float_or_none(metric.get("abs_canonical_segment_count_delta")) for metric in metrics]
        ),
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


def _normalize_inventory_payload(
    payload: Mapping[str, Any],
    *,
    inventory_path: Path,
    line_number: int,
    dataset_root: Path | None,
) -> BaselineAudioRow:
    audio_path = _resolve_audio_path(payload, inventory_path=inventory_path, dataset_root=dataset_root)
    audio_key = _inventory_audio_key(payload)
    supplied_audio_key = audio_key is not None
    if audio_key is None:
        if audio_path is None:
            raise ValueError(f"{inventory_path}:{line_number} must provide audio_key or audio_path")
        audio_key = (
            beatthis_audio_cache_key(audio_path)
            if audio_path.is_file()
            else _missing_audio_cache_key(audio_path)
        )

    beatmap_paths = _resolve_beatmap_paths(payload, inventory_path=inventory_path, dataset_root=dataset_root)
    return BaselineAudioRow(
        audio_key=audio_key,
        audio_path=audio_path,
        beatmap_paths=beatmap_paths,
        source_line_numbers=(line_number,),
        supplied_audio_key=supplied_audio_key,
        evaluation_strata=_evaluation_strata_payload(payload),
    )


def _evaluation_strata_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    label = payload.get("label")
    source = payload.get("source")
    return {
        "pilot_stratum": _json_safe(payload.get("pilot_stratum")),
        "pilot_quota_group": _json_safe(payload.get("pilot_quota_group")),
        "label_stratum": _json_safe(label.get("stratum")) if isinstance(label, MappingABC) else None,
        "label_confidence": _json_safe(label.get("confidence")) if isinstance(label, MappingABC) else None,
        "label_ambiguous": _json_safe(label.get("ambiguous")) if isinstance(label, MappingABC) else None,
        "source_long_track": _json_safe(source.get("long_track")) if isinstance(source, MappingABC) else None,
    }


def _merge_evaluation_strata(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "pilot_stratum",
        "pilot_quota_group",
        "label_stratum",
        "label_confidence",
        "label_ambiguous",
        "source_long_track",
    )
    return {
        key: left.get(key) if left.get(key) is not None else right.get(key)
        for key in keys
    }


def _inventory_audio_key(payload: Mapping[str, Any]) -> str | None:
    audio_key = _first_string(payload, ("audio_key", "beatthis_audio_key", "cache_audio_key"))
    if audio_key is not None:
        return audio_key
    cache = payload.get("cache")
    if isinstance(cache, MappingABC):
        value = cache.get("audio_cache_key")
        if isinstance(value, str) and value:
            return value
        if value is not None and not isinstance(value, str):
            raise ValueError("inventory field 'cache.audio_cache_key' must be a string when provided")
    return None


def _resolve_audio_path(
    payload: Mapping[str, Any],
    *,
    inventory_path: Path,
    dataset_root: Path | None,
) -> Path | None:
    raw_path = _first_string(payload, ("resolved_audio_path", "audio_path", "path"))
    if raw_path is None:
        return None
    return _resolve_dataset_path(
        raw_path,
        payload=payload,
        inventory_path=inventory_path,
        dataset_root=dataset_root,
    )


def _resolve_beatmap_paths(
    payload: Mapping[str, Any],
    *,
    inventory_path: Path,
    dataset_root: Path | None,
) -> tuple[Path, ...]:
    values: list[str] = []
    single = _first_string(payload, ("resolved_beatmap_path", "beatmap_path", "osu_path", "map_path"))
    if single is not None:
        values.append(single)

    for field in ("beatmap_paths", "osu_paths", "map_paths", "maps", "beatmaps"):
        raw = payload.get(field)
        if raw is None:
            continue
        if not isinstance(raw, SequenceABC) or isinstance(raw, (str, bytes)):
            raise ValueError(f"inventory field {field!r} must be a list")
        for item in raw:
            if isinstance(item, str):
                values.append(item)
            elif isinstance(item, MappingABC):
                nested = _first_string(item, ("resolved_beatmap_path", "beatmap_path", "osu_path", "path"))
                if nested is not None:
                    values.append(nested)
            else:
                raise ValueError(f"inventory field {field!r} contains unsupported item {item!r}")

    return _ordered_unique_paths(
        _resolve_dataset_path(
            value,
            payload=payload,
            inventory_path=inventory_path,
            dataset_root=dataset_root,
        )
        for value in values
    )


def _resolve_dataset_path(
    value: str,
    *,
    payload: Mapping[str, Any],
    inventory_path: Path,
    dataset_root: Path | None,
) -> Path:
    containment_root = (dataset_root if dataset_root is not None else inventory_path.parent).expanduser().resolve(
        strict=False
    )
    path = Path(value).expanduser()
    if path.is_absolute():
        resolved_path = path.resolve(strict=False)
        _require_path_within_root(resolved_path, containment_root)
        return resolved_path
    shard = _safe_relative_component(_first_string(payload, ("shard",)), field_name="shard")
    relative = _safe_relative_path(value)
    if dataset_root is not None:
        if shard is not None:
            resolved_path = (dataset_root / shard / relative).resolve(strict=False)
            _require_path_within_root(resolved_path, containment_root)
            return resolved_path
        resolved_path = (dataset_root / relative).resolve(strict=False)
        _require_path_within_root(resolved_path, containment_root)
        return resolved_path
    resolved_path = (inventory_path.parent / relative).resolve(strict=False)
    _require_path_within_root(resolved_path, containment_root)
    return resolved_path


def _require_path_within_root(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path {path.as_posix()} is outside allowed root {root.as_posix()}") from exc


def _missing_audio_cache_key(audio_path: Path) -> str:
    return f"{MISSING_AUDIO_CACHE_KEY_PREFIX}{audio_path.resolve(strict=False).as_posix()}"


def _safe_relative_path(value: str) -> Path:
    normalized = value.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or any(part == ".." for part in pure.parts):
        raise ValueError(f"path must be relative without '..' components, got {value!r}")
    return Path(*pure.parts)


def _safe_relative_component(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not value:
        raise ValueError(f"{field_name} must be non-empty when provided")
    pure = PurePosixPath(value.replace("\\", "/"))
    if pure.is_absolute() or len(pure.parts) != 1 or pure.parts[0] in {".", ".."}:
        raise ValueError(f"{field_name} must be a single relative path component, got {value!r}")
    return pure.parts[0]


def _first_string(payload: Mapping[str, Any], fields: Sequence[str]) -> str | None:
    for field in fields:
        value = payload.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"inventory field {field!r} must be a string when provided")
        if value:
            return value
    return None


def _ordered_unique_paths(paths: Sequence[Path] | Any) -> tuple[Path, ...]:
    seen: set[str] = set()
    ordered: list[Path] = []
    for path in paths:
        path = Path(path)
        key = path.as_posix()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return tuple(sorted(ordered, key=lambda path: path.as_posix()))


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
    rows: Sequence[BaselineAudioRow],
    results_by_key: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [dict(results_by_key[row.audio_key]) for row in rows if row.audio_key in results_by_key]


def _resume_payload(
    row: BaselineAudioRow,
    *,
    row_index: int,
    inventory_sha256: str,
    cache_config: BeatThisFramePredictionCacheConfig,
    fitter_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    components = {
        "result_schema": RESULT_SCHEMA,
        "inventory": {
            "sha256": inventory_sha256,
            "row_index": row_index,
            "source_line_numbers": list(row.source_line_numbers),
            "audio_key": row.audio_key,
            "audio_path": _path_or_none(row.audio_path),
            "beatmap_paths": [path.as_posix() for path in row.beatmap_paths],
            "evaluation_strata": _json_safe(row.evaluation_strata),
        },
        "cache": _cache_config_payload(cache_config),
        "cache_file": _cache_file_identity(cache_config, row.audio_key),
        "fitter": fitter_provenance,
    }
    return {
        "schema": RESUME_SCHEMA,
        "fingerprint": _stable_json_sha256(components),
        "components": components,
    }


def _result_matches_resume(result: Mapping[str, Any], expected_resume: Mapping[str, Any]) -> bool:
    resume = result.get("resume")
    return (
        result.get("schema") == RESULT_SCHEMA
        and isinstance(resume, MappingABC)
        and resume.get("schema") == RESUME_SCHEMA
        and resume.get("fingerprint") == expected_resume.get("fingerprint")
    )


def _write_result_jsonl_atomic(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(_json_safe(result), allow_nan=False, sort_keys=True) + "\n")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(
            json.dumps(_json_safe(payload), allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _cache_payload(config: BeatThisFramePredictionCacheConfig, audio_key: str) -> dict[str, Any]:
    payload = _cache_config_payload(config)
    payload["cache_path"] = beatthis_frame_prediction_cache_path(audio_key, config).as_posix()
    return payload


def _cache_config_payload(config: BeatThisFramePredictionCacheConfig) -> dict[str, Any]:
    return {
        "cache_root": config.cache_root.as_posix(),
        "cache_version": config.cache_version,
        "config_fingerprint": config.config_fingerprint,
        "checkpoint_path": config.checkpoint_path,
        "float16": config.float16,
        "shift_ms": config.shift_ms,
        "frame_rate_hz": config.frame_rate_hz,
    }


def _cache_file_identity(config: BeatThisFramePredictionCacheConfig, audio_key: str) -> dict[str, Any]:
    cache_path = beatthis_frame_prediction_cache_path(audio_key, config)
    if not cache_path.is_file():
        return {
            "path": cache_path.as_posix(),
            "exists": False,
            "size_bytes": None,
            "mtime_ns": None,
            "sha256": None,
        }
    stat = cache_path.stat()
    return {
        "path": cache_path.as_posix(),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _file_sha256(cache_path),
    }


def _prediction_payload(prediction: Any) -> dict[str, Any]:
    return {
        "provider": prediction.provider,
        "checkpoint_path": prediction.checkpoint_path,
        "source_path": prediction.source_path,
        "frame_count": prediction.frame_count,
        "frame_rate_hz": prediction.frame_rate_hz,
    }


def _fit_payload(fit_result: TimingFitResult) -> dict[str, Any]:
    return {
        "score": float(fit_result.score),
        "diagnostics": _json_safe(asdict(fit_result.diagnostics)),
        "predicted_segment_count": len(fit_result.grid.segments),
        "predicted_segments": _segments_payload(fit_result.grid),
    }


def _segments_payload(grid: FittedTimingGrid) -> list[dict[str, Any]]:
    return [
        _segment_payload(segment)
        for segment in grid.segments
    ]


def _segment_payload(segment: TimingSegment) -> dict[str, Any]:
    return {
        "offset_ms": float(segment.offset_ms),
        "beat_length_ms": float(segment.beat_length_ms),
        "bpm": float(segment.local_bpm),
        "meter": int(segment.meter),
    }


def _fitter_provenance(fitter: GridFitter) -> dict[str, Any]:
    config = getattr(fitter, "config", None)
    return {
        "class": f"{fitter.__class__.__module__}.{fitter.__class__.__qualname__}",
        "config": _json_safe(asdict(config)) if dataclasses.is_dataclass(config) else None,
        "default_grid_fitter_config": _json_safe(asdict(GridFitterConfig())),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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


def _path_or_none(path: Path | None) -> str | None:
    return path.as_posix() if path is not None else None


def _format_summary(summary: Mapping[str, Any]) -> str:
    results = summary["results"]
    metrics = summary["metrics"]
    phase = metrics["mean_phase_error_ms"]
    alias = metrics["local_bpm_alias_mae"]
    return (
        "Timing v3 cache-backed v2 baseline: "
        f"{results['successful_audio_count']}/{results['result_count']} audio ok, "
        f"{results['paired_comparison_count']} paired .osu comparisons, "
        f"phase mean/p50/p90={_stat_triplet(phase)} ms, "
        f"alias BPM MAE mean/p50/p90={_stat_triplet(alias)}, "
        f"failed_audio={results['failed_audio_count']}"
    )


def _stat_triplet(stats: Mapping[str, Any]) -> str:
    if not stats.get("count"):
        return "n/a"
    return f"{stats['mean']:.3f}/{stats['p50']:.3f}/{stats['p90']:.3f}"


def _print_progress(*, processed_count: int, total_count: int, result: Mapping[str, Any]) -> None:
    status = "ok" if result.get("ok") else f"failed:{result.get('failure_stage')}"
    print(f"[timing-v3-baseline] processed={processed_count}/{total_count} audio_key={result.get('audio_key')} {status}")


@contextmanager
def _row_timeout(timeout_seconds: float | None) -> Iterator[None]:
    if timeout_seconds is None or timeout_seconds <= 0.0 or not _timeout_supported(timeout_seconds):
        yield
        return

    old_handler = signal.getsignal(signal.SIGALRM)

    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"row exceeded timeout_seconds={timeout_seconds:g}")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def _timeout_supported(timeout_seconds: float | None) -> bool:
    return (
        timeout_seconds is not None
        and timeout_seconds > 0.0
        and hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the unchanged v2 GridFitter from an existing BeatThis frame-prediction cache.",
    )
    parser.add_argument("--inventory", required=True, type=Path, help="Audio-level JSONL inventory.")
    parser.add_argument("--output-jsonl", required=True, type=Path, help="Per-audio result JSONL path.")
    parser.add_argument("--summary-json", type=Path, default=None, help="Summary JSON path.")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_CONFIG.cache_root)
    parser.add_argument("--cache-version", default=DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_CONFIG.cache_version)
    parser.add_argument("--checkpoint", default=DEFAULT_BEATTHIS_CHECKPOINT)
    parser.add_argument("--float16", action="store_true")
    parser.add_argument("--shift-ms", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
