from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from pulsefield_model.data.beatmap_index import DEFAULT_4K_INDEX_PATH, DEFAULT_DATASET_ROOT
from pulsefield_model.timing.providers.beatthis import (
    BEATTHIS_PROVIDER_NAME,
    DEFAULT_BEATTHIS_CHECKPOINT,
    DEFAULT_BEATTHIS_DEVICE,
    BeatThisTimingProvider,
)
from pulsefield_model.timing.providers.beatthis_cache import (
    DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_CONFIG,
    BeatThisFramePredictionCacheConfig,
    BeatThisFramePredictionCacheError,
    beatthis_audio_cache_key,
    beatthis_frame_prediction_cache_path,
    load_beatthis_frame_prediction_cache,
    save_beatthis_frame_prediction_cache,
)
from pulsefield_model.timing.schema import FrameTimingPrediction


DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_VERSION = "beatthis_frame_predictions_v2"
DEFAULT_BEATTHIS_CACHE_REPORT_PATH = Path("artifacts/reports/timing/beatthis_frame_predictions_v2_build.json")
DEFAULT_LEGACY_BEATTHIS_CACHE_ROOT = Path("artifacts/cache/beatthis_frame_predictions_v1")
LEGACY_BEATTHIS_FRAME_PREDICTION_CACHE_VERSION = "beatthis_frame_predictions_v1"
DEFAULT_SPACE_HEADROOM_RATIO = 1.15
DEFAULT_SUPER_TIMING_SHIFT_MS = (0.0, 5.0, 10.0, 15.0)
REQUIRED_INDEX_COLUMNS = frozenset(("shard", "audio_path"))


def build_beatthis_frame_prediction_cache(
    *,
    index_path: str | Path = DEFAULT_4K_INDEX_PATH,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    cache_root: str | Path = DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_CONFIG.cache_root,
    cache_version: str = DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_VERSION,
    checkpoint_path: str = DEFAULT_BEATTHIS_CHECKPOINT,
    device: str = DEFAULT_BEATTHIS_DEVICE,
    float16: bool = False,
    storage_dtype: str = "float32",
    shift_ms: Sequence[float] = (0.0,),
    dry_run: bool = False,
    report_path: str | Path | None = DEFAULT_BEATTHIS_CACHE_REPORT_PATH,
    limit: int | None = None,
    overwrite: bool = False,
    migrate_legacy: bool = False,
    legacy_cache_root: str | Path = DEFAULT_LEGACY_BEATTHIS_CACHE_ROOT,
    progress_every: int = 25,
    empty_cache_every: int = 25,
    fail_fast: bool = False,
    space_headroom_ratio: float = DEFAULT_SPACE_HEADROOM_RATIO,
    ignore_space_check: bool = False,
) -> dict[str, Any]:
    started = time.time()
    index_path = Path(index_path)
    dataset_root = Path(dataset_root)
    cache_root = Path(cache_root)
    if empty_cache_every < 0:
        raise ValueError(f"empty_cache_every must be non-negative, got {empty_cache_every!r}")
    shift_ms_values = _normalize_shift_ms(shift_ms)
    base_config = BeatThisFramePredictionCacheConfig(
        cache_root=cache_root,
        cache_version=cache_version,
        checkpoint_path=checkpoint_path,
        float16=float16,
        shift_ms=0.0,
    )
    if storage_dtype != "float32":
        raise ValueError("current BeatThis cache API stores frame predictions as float32 only")
    configs_by_shift = {
        value: replace(base_config, shift_ms=value)
        for value in shift_ms_values
    }

    rows = _load_unique_audio_rows(index_path=index_path, dataset_root=dataset_root, limit=limit)
    inventory = _inventory_cache_jobs(rows=rows, configs_by_shift=configs_by_shift)
    legacy_index = (
        _build_legacy_cache_index(
            legacy_cache_root=Path(legacy_cache_root),
            checkpoint_path=checkpoint_path,
            float16=float16,
            frame_rate_hz=base_config.frame_rate_hz,
            rows=rows,
        )
        if migrate_legacy and not overwrite
        else {}
    )
    legacy_reusable_count = sum(
        1
        for job in inventory["missing_jobs"]
        if float(job["shift_ms"]) == 0.0
        and Path(job["row"]["resolved_audio_path"]).resolve(strict=False).as_posix() in legacy_index
    )
    disk = _disk_report(
        cache_root=cache_root,
        missing_jobs=inventory["missing_jobs"],
        space_headroom_ratio=space_headroom_ratio,
    )
    report: dict[str, Any] = {
        "schema": "beatthis_frame_prediction_cache_build_report_v1",
        "started_at_unix": started,
        "finished_at_unix": None,
        "seconds": None,
        "dry_run": bool(dry_run),
        "aborted": False,
        "abort_reason": None,
        "index_path": index_path.as_posix(),
        "dataset_root": dataset_root.as_posix(),
        "checkpoint_path": checkpoint_path,
        "device": device,
        "float16": bool(float16),
        "storage_dtype": storage_dtype,
        "empty_cache_every": empty_cache_every,
        "shift_ms": list(shift_ms_values),
        "cache_root": cache_root.as_posix(),
        "cache_version": cache_version,
        "source": {
            "row_count": int(inventory["source_row_count"]),
            "unique_audio_count": len(rows),
            "limited": limit is not None,
            "limit": limit,
        },
        "disk": disk,
        "cache": {
            "total_jobs": len(rows) * len(shift_ms_values),
            "existing_count": int(inventory["existing_count"]),
            "missing_count": len(inventory["missing_jobs"]),
            "initial_inference_required_count": len(inventory["missing_jobs"]) - legacy_reusable_count,
            "created_count": 0,
            "migrated_count": 0,
            "failed_count": 0,
            "skipped_missing_audio_count": int(inventory["missing_audio_count"]),
            "overwritten_count": 0,
            "corrupt_existing_recompute_count": int(inventory["corrupt_existing_count"]),
        },
        "failures": [],
        "warnings": [],
        "legacy_migration": {
            "enabled": bool(migrate_legacy),
            "legacy_cache_root": Path(legacy_cache_root).as_posix(),
            "indexed_count": len(legacy_index),
            "reusable_count": legacy_reusable_count,
            "failed_count": 0,
        },
    }

    if dry_run:
        _finish_report(report, started=started, report_path=report_path)
        return report

    if not ignore_space_check:
        if not bool(disk["space_estimate_complete"]):
            report["aborted"] = True
            report["abort_reason"] = "incomplete_disk_space_estimate"
            _finish_report(report, started=started, report_path=report_path)
            return report
        if not bool(disk["enough_space_for_known_estimate"]):
            report["aborted"] = True
            report["abort_reason"] = "insufficient_disk_space_for_known_estimate"
            _finish_report(report, started=started, report_path=report_path)
            return report

    provider: BeatThisTimingProvider | None = None
    prediction: FrameTimingPrediction | None = None
    audio: object | None = None
    processed_audio_count = 0
    for row in rows:
        processed_audio_count += 1
        if (
            empty_cache_every > 0
            and processed_audio_count > 1
            and (processed_audio_count - 1) % empty_cache_every == 0
        ):
            prediction = None
            audio = None
            _empty_torch_accelerator_cache(device=device)
        if progress_every > 0 and (processed_audio_count == 1 or processed_audio_count % progress_every == 0):
            _print_progress(
                processed_audio_count=processed_audio_count,
                total_audio_count=len(rows),
                created_count=int(report["cache"]["created_count"]),
                existing_count=int(report["cache"]["existing_count"]),
                migrated_count=int(report["cache"]["migrated_count"]),
                failed_count=int(report["cache"]["failed_count"]),
            )

        if not row["exists"]:
            _add_failure(
                report,
                row=row,
                shift_ms=None,
                error_type="missing_audio",
                message=f"audio file does not exist: {row['resolved_audio_path']}",
            )
            if fail_fast:
                break
            continue

        missing_shifts: list[float] = []
        for shift_ms_value, config in configs_by_shift.items():
            if not overwrite:
                try:
                    cached = load_beatthis_frame_prediction_cache(str(row["audio_key"]), config)
                except BeatThisFramePredictionCacheError:
                    cached = None
                if cached is not None:
                    continue

            if not overwrite:
                try:
                    if _try_migrate_legacy_cache(
                        row=row,
                        config=config,
                        legacy_index=legacy_index,
                        report=report,
                    ):
                        continue
                except Exception as exc:  # noqa: BLE001 - a bad legacy entry must not stop inference.
                    report["legacy_migration"]["failed_count"] = (
                        int(report["legacy_migration"]["failed_count"]) + 1
                    )
                    report["warnings"].append(
                        {
                            **_failure_payload(
                                row=row,
                                shift_ms=shift_ms_value,
                                error_type=exc.__class__.__name__,
                                message=f"legacy cache migration failed; recomputing: {exc}",
                            ),
                            "warning_type": "legacy_migration_failed",
                        }
                    )

            missing_shifts.append(shift_ms_value)

        if not missing_shifts:
            continue

        if provider is None:
            provider = BeatThisTimingProvider(
                checkpoint_path=checkpoint_path,
                device=device,
                float16=float16,
            )

        if missing_shifts == [0.0]:
            try:
                prediction = provider.predict_file(Path(row["resolved_audio_path"]))
                _save_prediction_for_shift(
                    prediction,
                    row=row,
                    config=configs_by_shift[0.0],
                    report=report,
                    overwrite=overwrite,
                )
            except Exception as exc:  # noqa: BLE001 - continue-failures builder boundary.
                _add_failure(
                    report,
                    row=row,
                    shift_ms=0.0,
                    error_type=exc.__class__.__name__,
                    message=str(exc),
                )
                if fail_fast:
                    break
            continue

        try:
            audio, sample_rate = provider.load_file(Path(row["resolved_audio_path"]))
        except Exception as exc:  # noqa: BLE001 - all shifts require the same decoded audio.
            for shift_ms_value in missing_shifts:
                _add_failure(
                    report,
                    row=row,
                    shift_ms=shift_ms_value,
                    error_type=exc.__class__.__name__,
                    message=str(exc),
                )
            if fail_fast:
                break
            continue

        stop_after_shift_failure = False
        for shift_ms_value in missing_shifts:
            try:
                if shift_ms_value == 0.0:
                    prediction = provider.predict_audio(
                        audio,
                        sample_rate,
                        source_path=Path(row["resolved_audio_path"]),
                    )
                else:
                    prediction = provider.predict_shifted_audio(
                        audio,
                        sample_rate,
                        shift_ms=shift_ms_value,
                        source_path=Path(row["resolved_audio_path"]),
                    )
                _save_prediction_for_shift(
                    prediction,
                    row=row,
                    config=configs_by_shift[shift_ms_value],
                    report=report,
                    overwrite=overwrite,
                )
            except Exception as exc:  # noqa: BLE001 - one shifted pass must not poison the others.
                _add_failure(
                    report,
                    row=row,
                    shift_ms=shift_ms_value,
                    error_type=exc.__class__.__name__,
                    message=str(exc),
                )
                if fail_fast:
                    stop_after_shift_failure = True
                    break
        if stop_after_shift_failure:
            break

    prediction = None
    audio = None
    provider = None
    _empty_torch_accelerator_cache(device=device)
    _finish_report(report, started=started, report_path=report_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    shift_ms = _shift_ms_from_args(args)
    report = build_beatthis_frame_prediction_cache(
        index_path=args.index_path,
        dataset_root=args.dataset_root,
        cache_root=args.cache_root,
        cache_version=args.cache_version,
        checkpoint_path=args.checkpoint,
        device=args.device,
        float16=args.float16,
        storage_dtype=args.storage_dtype,
        shift_ms=shift_ms,
        dry_run=args.dry_run,
        report_path=args.report_path,
        limit=args.limit,
        overwrite=args.overwrite,
        migrate_legacy=args.migrate_legacy,
        legacy_cache_root=args.legacy_cache_root,
        progress_every=args.progress_every,
        empty_cache_every=args.empty_cache_every,
        fail_fast=args.fail_fast,
        space_headroom_ratio=args.space_headroom_ratio,
        ignore_space_check=args.ignore_space_check,
    )
    if args.json:
        print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    else:
        print(_format_report_summary(report))
    if report["aborted"]:
        return 2
    return 1 if int(report["cache"]["failed_count"]) > 0 else 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a resumable BeatThis frame-prediction cache from an index.")
    parser.add_argument("--index-path", type=Path, default=DEFAULT_4K_INDEX_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_CONFIG.cache_root)
    parser.add_argument("--cache-version", default=DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_VERSION)
    parser.add_argument("--checkpoint", default=DEFAULT_BEATTHIS_CHECKPOINT)
    parser.add_argument("--device", default=DEFAULT_BEATTHIS_DEVICE)
    parser.add_argument("--float16", action="store_true")
    parser.add_argument("--storage-dtype", default="float32", choices=["float32"])
    parser.add_argument("--shift-ms", action="append", type=float, default=None)
    parser.add_argument(
        "--super-timing-shifts",
        action="store_true",
        help="Build the default 0/5/10/15 ms shifted BeatThis caches.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--report-path", type=Path, default=DEFAULT_BEATTHIS_CACHE_REPORT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--migrate-legacy", action="store_true")
    parser.add_argument("--legacy-cache-root", type=Path, default=DEFAULT_LEGACY_BEATTHIS_CACHE_ROOT)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--empty-cache-every",
        type=int,
        default=25,
        help="Run gc.collect() and empty CUDA/MPS allocator caches every N songs; 0 disables it.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--space-headroom-ratio", type=float, default=DEFAULT_SPACE_HEADROOM_RATIO)
    parser.add_argument("--ignore-space-check", action="store_true")
    return parser


def _shift_ms_from_args(args: argparse.Namespace) -> tuple[float, ...]:
    values: list[float] = []
    if args.super_timing_shifts:
        values.extend(DEFAULT_SUPER_TIMING_SHIFT_MS)
    if args.shift_ms is not None:
        values.extend(args.shift_ms)
    if not values:
        values.append(0.0)
    return _normalize_shift_ms(values)


def _load_unique_audio_rows(
    *,
    index_path: Path,
    dataset_root: Path,
    limit: int | None,
) -> list[dict[str, Any]]:
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be non-negative, got {limit!r}")
    df = pd.read_parquet(index_path)
    missing_columns = sorted(REQUIRED_INDEX_COLUMNS.difference(df.columns))
    if missing_columns:
        raise ValueError(f"index {index_path} is missing required columns: {missing_columns}")

    source_row_count = len(df)
    unique_df = df.drop_duplicates(["shard", "audio_path"], keep="first")
    if limit is not None:
        unique_df = unique_df.head(limit)

    rows = []
    for _, item in unique_df.iterrows():
        shard = _safe_relative_component(str(item["shard"]), field_name="shard")
        audio_rel = _safe_relative_path(str(item["audio_path"]), field_name="audio_path")
        resolved_audio_path = _resolve_dataset_audio_path(
            dataset_root=dataset_root,
            shard=shard,
            audio_rel=audio_rel,
        )
        exists = resolved_audio_path.is_file()
        audio_key = (
            beatthis_audio_cache_key(resolved_audio_path)
            if exists
            else f"missing:{resolved_audio_path.as_posix()}"
        )
        rows.append(
            {
                "shard": shard,
                "audio_path": audio_rel.as_posix(),
                "audio_key": audio_key,
                "resolved_audio_path": resolved_audio_path,
                "exists": exists,
                "source_row_count": source_row_count,
            }
        )
    for row in rows:
        row["source_row_count"] = source_row_count
    return rows


def _inventory_cache_jobs(
    *,
    rows: Sequence[Mapping[str, Any]],
    configs_by_shift: Mapping[float, BeatThisFramePredictionCacheConfig],
) -> dict[str, Any]:
    existing_count = 0
    corrupt_existing_count = 0
    missing_audio_count = 0
    missing_jobs = []
    for row in rows:
        if not bool(row["exists"]):
            missing_audio_count += len(configs_by_shift)
        for shift_ms_value, config in configs_by_shift.items():
            try:
                cached = load_beatthis_frame_prediction_cache(str(row["audio_key"]), config)
            except BeatThisFramePredictionCacheError:
                cached = None
                corrupt_existing_count += 1
            if cached is not None:
                existing_count += 1
                continue
            duration_seconds = _audio_duration_seconds(Path(row["resolved_audio_path"])) if row["exists"] else None
            missing_jobs.append(
                {
                    "row": row,
                    "shift_ms": shift_ms_value,
                    "config": config,
                    "duration_seconds": duration_seconds,
                }
            )
    source_row_count = int(rows[0]["source_row_count"]) if rows else 0
    return {
        "source_row_count": source_row_count,
        "existing_count": existing_count,
        "corrupt_existing_count": corrupt_existing_count,
        "missing_audio_count": missing_audio_count,
        "missing_jobs": missing_jobs,
    }


def _disk_report(
    *,
    cache_root: Path,
    missing_jobs: Sequence[Mapping[str, Any]],
    space_headroom_ratio: float,
) -> dict[str, Any]:
    if not math.isfinite(space_headroom_ratio) or space_headroom_ratio <= 0.0:
        raise ValueError(f"space_headroom_ratio must be positive and finite, got {space_headroom_ratio!r}")

    known_estimated_bytes = 0
    unknown_estimate_count = 0
    for job in missing_jobs:
        duration_seconds = job["duration_seconds"]
        if duration_seconds is None:
            unknown_estimate_count += 1
            continue
        config = job["config"]
        assert isinstance(config, BeatThisFramePredictionCacheConfig)
        frame_count = int(math.ceil((float(duration_seconds) + float(job["shift_ms"]) / 1000.0) * config.frame_rate_hz))
        known_estimated_bytes += _estimate_beatthis_cache_bytes(frame_count=frame_count)

    target = _nearest_existing_path(cache_root)
    usage = shutil.disk_usage(target)
    required_with_headroom = int(math.ceil(known_estimated_bytes * space_headroom_ratio))
    space_estimate_complete = unknown_estimate_count == 0
    return {
        "checked_path": target.as_posix(),
        "free_bytes": int(usage.free),
        "known_estimated_missing_bytes": int(known_estimated_bytes),
        "unknown_estimate_count": int(unknown_estimate_count),
        "space_estimate_complete": space_estimate_complete,
        "space_headroom_ratio": float(space_headroom_ratio),
        "required_free_bytes_with_headroom": int(required_with_headroom),
        "enough_space_for_known_estimate": bool(usage.free >= required_with_headroom),
    }


def _estimate_beatthis_cache_bytes(*, frame_count: int) -> int:
    if frame_count < 0:
        raise ValueError(f"frame_count must be non-negative, got {frame_count!r}")
    return int(frame_count) * 2 * 4 + 4096


def _build_legacy_cache_index(
    *,
    legacy_cache_root: Path,
    checkpoint_path: str,
    float16: bool,
    frame_rate_hz: float,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    if not legacy_cache_root.exists():
        return {}
    index: dict[str, Path] = {}
    for row in rows:
        audio_path = Path(row["resolved_audio_path"])
        if not audio_path.is_file():
            continue
        cache_path = _legacy_cache_path_for_audio(
            audio_path,
            legacy_cache_root=legacy_cache_root,
            checkpoint_path=checkpoint_path,
            float16=float16,
        )
        if not cache_path.is_file():
            continue
        try:
            prediction = _load_legacy_prediction(cache_path)
            _validate_legacy_prediction(
                prediction,
                cache_path=cache_path,
                resolved_audio_path=audio_path.resolve(strict=False).as_posix(),
                checkpoint_path=checkpoint_path,
                frame_rate_hz=frame_rate_hz,
            )
        except BeatThisFramePredictionCacheError:
            continue
        index[audio_path.resolve(strict=False).as_posix()] = cache_path
    return index


def _legacy_cache_path_for_audio(
    audio_path: Path,
    *,
    legacy_cache_root: Path,
    checkpoint_path: str,
    float16: bool,
) -> Path:
    resolved_audio_path = audio_path.resolve(strict=False)
    stat = resolved_audio_path.stat()
    model_payload = {
        "cache_version": LEGACY_BEATTHIS_FRAME_PREDICTION_CACHE_VERSION,
        "checkpoint_path": checkpoint_path,
        "float16": bool(float16),
    }
    model_id = hashlib.sha256(json.dumps(model_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    key_payload = {
        "cache_version": LEGACY_BEATTHIS_FRAME_PREDICTION_CACHE_VERSION,
        "audio_path": resolved_audio_path.as_posix(),
        "audio_size": stat.st_size,
        "audio_mtime_ns": stat.st_mtime_ns,
        "checkpoint_path": checkpoint_path,
        "float16": bool(float16),
    }
    key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()
    return legacy_cache_root / model_id / f"{key}.npz"


def _load_legacy_prediction(cache_path: Path) -> FrameTimingPrediction:
    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            expected_files = {
                "beat_prob",
                "downbeat_prob",
                "frame_rate_hz",
                "provider",
                "checkpoint_path",
                "source_path",
            }
            if set(payload.files) != expected_files:
                raise ValueError(f"expected keys {sorted(expected_files)}, got {sorted(payload.files)}")
            beat_prob = np.asarray(payload["beat_prob"], dtype=np.float32)
            downbeat_prob = np.asarray(payload["downbeat_prob"], dtype=np.float32)
            frame_rate_hz = float(np.asarray(payload["frame_rate_hz"]).item())
            provider = str(np.asarray(payload["provider"]).item())
            checkpoint_path = str(np.asarray(payload["checkpoint_path"]).item())
            source_path = str(np.asarray(payload["source_path"]).item())
    except Exception as exc:  # noqa: BLE001
        raise BeatThisFramePredictionCacheError(f"invalid BeatThis cache payload {cache_path}: {exc}") from exc
    return FrameTimingPrediction(
        provider=provider,
        checkpoint_path=None if checkpoint_path == "" else checkpoint_path,
        source_path=None if source_path == "" else source_path,
        beat_prob=beat_prob,
        downbeat_prob=downbeat_prob,
        frame_rate_hz=frame_rate_hz,
    )


def _try_migrate_legacy_cache(
    *,
    row: Mapping[str, Any],
    config: BeatThisFramePredictionCacheConfig,
    legacy_index: Mapping[str, Path],
    report: dict[str, Any],
) -> bool:
    if not legacy_index:
        return False
    if config.shift_ms != 0.0:
        return False
    resolved_audio_path = Path(row["resolved_audio_path"]).resolve(strict=False).as_posix()
    legacy_path = legacy_index.get(resolved_audio_path)
    if legacy_path is None:
        return False
    prediction = _load_legacy_prediction(legacy_path)
    _validate_legacy_prediction(
        prediction,
        cache_path=legacy_path,
        resolved_audio_path=resolved_audio_path,
        checkpoint_path=config.checkpoint_path,
        frame_rate_hz=config.frame_rate_hz,
    )
    save_beatthis_frame_prediction_cache(prediction, str(row["audio_key"]), config)
    report["cache"]["migrated_count"] = int(report["cache"]["migrated_count"]) + 1
    return True


def _validate_legacy_prediction(
    prediction: FrameTimingPrediction,
    *,
    cache_path: Path,
    resolved_audio_path: str,
    checkpoint_path: str,
    frame_rate_hz: float,
) -> None:
    if prediction.provider != BEATTHIS_PROVIDER_NAME:
        raise BeatThisFramePredictionCacheError(
            f"legacy cache {cache_path} provider mismatch: {prediction.provider!r}"
        )
    if prediction.checkpoint_path != checkpoint_path:
        raise BeatThisFramePredictionCacheError(
            f"legacy cache {cache_path} checkpoint mismatch: {prediction.checkpoint_path!r}"
        )
    if prediction.source_path is None or (
        Path(prediction.source_path).resolve(strict=False).as_posix() != resolved_audio_path
    ):
        raise BeatThisFramePredictionCacheError(
            f"legacy cache {cache_path} source mismatch: {prediction.source_path!r}"
        )
    if not math.isclose(prediction.frame_rate_hz, frame_rate_hz, rel_tol=0.0, abs_tol=1e-9):
        raise BeatThisFramePredictionCacheError(
            f"legacy cache {cache_path} frame-rate mismatch: {prediction.frame_rate_hz!r}"
        )


def _save_prediction_for_shift(
    prediction: object,
    *,
    row: Mapping[str, Any],
    config: BeatThisFramePredictionCacheConfig,
    report: dict[str, Any],
    overwrite: bool,
) -> None:
    cache_path = beatthis_frame_prediction_cache_path(str(row["audio_key"]), config)
    existed = cache_path.exists()
    save_beatthis_frame_prediction_cache(prediction, str(row["audio_key"]), config)  # type: ignore[arg-type]
    report["cache"]["created_count"] = int(report["cache"]["created_count"]) + 1
    if overwrite and existed:
        report["cache"]["overwritten_count"] = int(report["cache"]["overwritten_count"]) + 1


def _finish_report(report: dict[str, Any], *, started: float, report_path: str | Path | None) -> None:
    finished = time.time()
    report["finished_at_unix"] = finished
    report["seconds"] = finished - started
    if report_path is not None:
        _write_json_atomic(Path(report_path), report)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _add_failure(
    report: dict[str, Any],
    *,
    row: Mapping[str, Any],
    shift_ms: float | None,
    error_type: str,
    message: str,
) -> None:
    report["failures"].append(
        _failure_payload(row=row, shift_ms=shift_ms, error_type=error_type, message=message)
    )
    report["cache"]["failed_count"] = int(report["cache"]["failed_count"]) + 1


def _failure_payload(
    *,
    row: Mapping[str, Any],
    shift_ms: float | None,
    error_type: str,
    message: str,
) -> dict[str, Any]:
    return {
        "shard": str(row["shard"]),
        "audio_path": str(row["audio_path"]),
        "audio_key": str(row["audio_key"]),
        "resolved_audio_path": Path(row["resolved_audio_path"]).as_posix(),
        "shift_ms": shift_ms,
        "error_type": error_type,
        "message": message,
    }


def _normalize_shift_ms(values: Sequence[float]) -> tuple[float, ...]:
    normalized: list[float] = []
    for value in values:
        shift_ms = float(value)
        if not math.isfinite(shift_ms) or shift_ms < 0.0:
            raise ValueError(f"shift_ms must be non-negative and finite, got {value!r}")
        if not any(abs(shift_ms - existing) < 1e-9 for existing in normalized):
            normalized.append(shift_ms)
    if not normalized:
        normalized.append(0.0)
    normalized.sort()
    return tuple(normalized)


def _safe_relative_component(value: str, *, field_name: str) -> str:
    path = _safe_relative_path(value, field_name=field_name)
    if len(path.parts) != 1:
        raise ValueError(f"{field_name} must be a single relative path component, got {value!r}")
    return path.as_posix()


def _safe_relative_path(value: str, *, field_name: str) -> PurePosixPath:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute():
        raise ValueError(f"{field_name} must be relative, got {value!r}")
    if not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{field_name} must not contain empty, '.', or '..' components, got {value!r}")
    return path


def _resolve_dataset_audio_path(
    *,
    dataset_root: Path,
    shard: str,
    audio_rel: PurePosixPath,
) -> Path:
    base = dataset_root / shard
    candidate = base.joinpath(*audio_rel.parts)
    resolved_root = dataset_root.resolve(strict=False)
    resolved_candidate = candidate.resolve(strict=False)
    if not resolved_candidate.is_relative_to(resolved_root):
        raise ValueError(f"resolved audio path escapes dataset_root: {candidate}")
    return resolved_candidate


def _audio_duration_seconds(path: Path) -> float | None:
    if not path.is_file():
        return None
    try:
        from mutagen import File as mutagen_file
    except ImportError:
        return None
    try:
        audio = mutagen_file(path)
    except Exception:  # noqa: BLE001
        return None
    info = getattr(audio, "info", None)
    length = getattr(info, "length", None)
    if length is None:
        return None
    value = float(length)
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


def _nearest_existing_path(path: Path) -> Path:
    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            return current
        current = parent
    return current


def _empty_torch_accelerator_cache(*, device: str) -> None:
    gc.collect()
    normalized_device = device.lower()
    if not (normalized_device.startswith("cuda") or normalized_device.startswith("mps")):
        return
    try:
        import torch
    except ImportError:
        return
    if normalized_device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if normalized_device.startswith("mps") and getattr(torch, "mps", None) is not None:
        try:
            torch.mps.empty_cache()
        except RuntimeError:
            pass


def _print_progress(
    *,
    processed_audio_count: int,
    total_audio_count: int,
    created_count: int,
    existing_count: int,
    migrated_count: int,
    failed_count: int,
) -> None:
    print(
        "beatthis-cache "
        f"audio={processed_audio_count}/{total_audio_count} "
        f"created={created_count} existing={existing_count} migrated={migrated_count} failed={failed_count}",
        file=sys.stderr,
    )


def _format_report_summary(report: Mapping[str, Any]) -> str:
    cache = report["cache"]
    disk = report["disk"]
    return "\n".join(
        [
            f"dry_run: {report['dry_run']}",
            f"aborted: {report['aborted']} {report['abort_reason']}",
            f"unique_audio: {report['source']['unique_audio_count']}",
            f"total_jobs: {cache['total_jobs']}",
            f"existing: {cache['existing_count']}",
            f"missing: {cache['missing_count']}",
            f"initial_inference_required: {cache['initial_inference_required_count']}",
            f"created: {cache['created_count']}",
            f"migrated: {cache['migrated_count']}",
            f"failed: {cache['failed_count']}",
            f"estimated_missing_bytes: {disk['known_estimated_missing_bytes']}",
            f"free_bytes: {disk['free_bytes']}",
            f"space_estimate_complete: {disk['space_estimate_complete']}",
            f"enough_space: {disk['enough_space_for_known_estimate']}",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
