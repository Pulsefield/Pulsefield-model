from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol, Sequence

import pandas as pd

from pulsefield_model.data.beatmap_index import DEFAULT_4K_INDEX_PATH, DEFAULT_DATASET_ROOT
from pulsefield_model.timing.providers.beatthis import DEFAULT_BEATTHIS_CHECKPOINT
from pulsefield_model.timing.providers.beatthis_cache import (
    DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_CONFIG,
    BeatThisFramePredictionCacheConfig,
    BeatThisFramePredictionCacheError,
    beatthis_audio_cache_key,
    beatthis_frame_prediction_cache_path,
    load_beatthis_frame_prediction_cache,
)


TIMING_V3_INVENTORY_REPORT_SCHEMA = "pulsefield_model.timing_v3_inventory_report_v1"
TIMING_V3_INVENTORY_AUDIO_ROW_SCHEMA = "pulsefield_model.timing_v3_inventory_audio_row_v1"
REQUIRED_INDEX_COLUMNS = frozenset(("shard", "audio_path"))


class InventoryEvidenceHook(Protocol):
    """Optional seam for later redline/object evidence without importing it here."""

    def __call__(
        self,
        *,
        audio_row: Mapping[str, Any],
        map_rows: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any] | None: ...


def build_timing_v3_inventory(
    *,
    index_path: str | Path = DEFAULT_4K_INDEX_PATH,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    cache_root: str | Path = DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_CONFIG.cache_root,
    cache_version: str = DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_CONFIG.cache_version,
    checkpoint_path: str = DEFAULT_BEATTHIS_CHECKPOINT,
    float16: bool = False,
    shift_ms: float = 0.0,
    limit: int | None = None,
    report_path: str | Path,
    audio_rows_path: str | Path,
    evidence_hook: InventoryEvidenceHook | None = None,
) -> dict[str, Any]:
    """Build the canonical Timing v3 audio-group inventory from local assets.

    The inventory validates exactly one existing BeatThis frame cache per
    resolved audio path. It never instantiates a timing provider or performs
    model inference.
    """

    index_path = Path(index_path)
    dataset_root = Path(dataset_root)
    report_path = Path(report_path)
    audio_rows_path = Path(audio_rows_path)
    if report_path == audio_rows_path:
        raise ValueError("report_path and audio_rows_path must be different explicit files")
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be non-negative, got {limit!r}")

    config = BeatThisFramePredictionCacheConfig(
        cache_root=Path(cache_root),
        cache_version=cache_version,
        checkpoint_path=checkpoint_path,
        float16=float16,
        shift_ms=shift_ms,
    )
    source_rows = _load_source_rows(index_path=index_path, dataset_root=dataset_root)
    duplicate_beatmap_ids = _duplicate_beatmap_ids(source_rows)

    groups = _group_source_rows_by_resolved_audio(source_rows)
    selected_groups = groups[:limit] if limit is not None else groups

    audio_rows: list[dict[str, Any]] = []
    cache_counts: Counter[str] = Counter()
    anomaly_counts: Counter[str] = Counter()
    metadata_existing_count = 0
    metadata_missing_count = 0

    for audio_group_index, group in enumerate(selected_groups):
        audio_row = _build_audio_row(
            audio_group_index=audio_group_index,
            group=group,
            config=config,
            duplicate_beatmap_ids=duplicate_beatmap_ids,
        )
        if evidence_hook is not None:
            evidence = evidence_hook(audio_row=audio_row, map_rows=tuple(audio_row["maps"]))
            audio_row["evidence"] = _normalize_json_value(evidence) if evidence is not None else None
        else:
            audio_row["evidence"] = None

        audio_rows.append(audio_row)
        cache_counts[str(audio_row["cache"]["status"])] += 1
        for anomaly in audio_row["anomalies"]:
            anomaly_counts[str(anomaly)] += 1
        if int(audio_row["metadata_json"]["existing_count"]) > 0:
            metadata_existing_count += 1
        else:
            metadata_missing_count += 1

    _write_jsonl_atomic(audio_rows_path, audio_rows)

    report = {
        "schema": TIMING_V3_INVENTORY_REPORT_SCHEMA,
        "command": _format_command(sys.argv[1:] if sys.argv else None),
        "git_commit": _git_stdout("rev-parse", "HEAD"),
        "index": {
            "path": index_path.as_posix(),
            "sha256": _sha256(index_path),
            "row_count": len(source_rows),
            "required_columns": sorted(REQUIRED_INDEX_COLUMNS),
        },
        "dataset_root": dataset_root.as_posix(),
        "cache_config": _cache_config_payload(config),
        "source": {
            "audio_group_count": len(groups),
            "map_row_count": len(source_rows),
            "duplicate_beatmap_id_count": len(duplicate_beatmap_ids),
            "duplicate_beatmap_ids": [
                duplicate_beatmap_ids[key] for key in sorted(duplicate_beatmap_ids)
            ],
        },
        "output": {
            "limited": limit is not None,
            "limit": limit,
            "audio_group_count": len(audio_rows),
            "map_row_count": sum(int(row["map_count"]) for row in audio_rows),
            "report_path": report_path.as_posix(),
            "audio_rows_path": audio_rows_path.as_posix(),
            "audio_rows_sha256": _sha256(audio_rows_path),
        },
        "cache": {
            "valid_count": int(cache_counts["valid"]),
            "missing_count": int(cache_counts["missing"]),
            "invalid_count": int(cache_counts["invalid"]),
            "missing_audio_count": int(cache_counts["missing_audio"]),
        },
        "metadata_json": {
            "audio_group_existing_count": metadata_existing_count,
            "audio_group_missing_count": metadata_missing_count,
        },
        "anomalies": {
            "audio_group_count": sum(1 for row in audio_rows if row["anomalies"]),
            "counts": {key: int(anomaly_counts[key]) for key in sorted(anomaly_counts)},
        },
    }
    _write_json_atomic(report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    report = build_timing_v3_inventory(
        index_path=args.index_path,
        dataset_root=args.dataset_root,
        cache_root=args.cache_root,
        cache_version=args.cache_version,
        checkpoint_path=args.checkpoint,
        float16=args.float16,
        shift_ms=args.shift_ms,
        limit=args.limit,
        report_path=args.report_path,
        audio_rows_path=args.audio_rows_path,
    )
    if args.json:
        print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    else:
        print(_format_summary(report))
    cache = report["cache"]
    return 1 if any(int(cache[key]) for key in ("missing_count", "invalid_count", "missing_audio_count")) else 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Timing v3 canonical audio-group inventory.")
    parser.add_argument("--index-path", type=Path, default=DEFAULT_4K_INDEX_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_CONFIG.cache_root)
    parser.add_argument("--cache-version", default=DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_CONFIG.cache_version)
    parser.add_argument("--checkpoint", default=DEFAULT_BEATTHIS_CHECKPOINT)
    parser.add_argument("--float16", action="store_true")
    parser.add_argument("--shift-ms", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--audio-rows-path", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    return parser


def _load_source_rows(*, index_path: Path, dataset_root: Path) -> list[dict[str, Any]]:
    index_df = pd.read_parquet(index_path)
    missing_columns = sorted(REQUIRED_INDEX_COLUMNS.difference(index_df.columns))
    if missing_columns:
        raise ValueError(f"index {index_path} is missing required columns: {missing_columns}")

    rows: list[dict[str, Any]] = []
    for source_row_index, (_, raw_row) in enumerate(index_df.iterrows()):
        index_row = {str(column): _normalize_json_value(raw_row[column]) for column in index_df.columns}
        shard = _safe_relative_component(str(index_row["shard"]), field_name="shard")
        audio_rel = _safe_relative_path(str(index_row["audio_path"]), field_name="audio_path")
        resolved_audio_path = _resolve_dataset_path(
            dataset_root=dataset_root,
            shard=shard,
            relative_path=audio_rel,
            field_name="audio_path",
        )

        beatmap_rel = _optional_relative_path(index_row.get("beatmap_path"), field_name="beatmap_path")
        resolved_beatmap_path = (
            _resolve_dataset_path(
                dataset_root=dataset_root,
                shard=shard,
                relative_path=beatmap_rel,
                field_name="beatmap_path",
            )
            if beatmap_rel is not None
            else None
        )
        metadata_json_path = _metadata_json_path_for_row(
            dataset_root=dataset_root,
            shard=shard,
            index_row=index_row,
            resolved_beatmap_path=resolved_beatmap_path,
            resolved_audio_path=resolved_audio_path,
        )
        rows.append(
            {
                "source_row_index": source_row_index,
                "index_row": index_row,
                "shard": shard,
                "audio_path": audio_rel.as_posix(),
                "resolved_audio_path": resolved_audio_path.as_posix(),
                "beatmap_path": beatmap_rel.as_posix() if beatmap_rel is not None else None,
                "resolved_beatmap_path": (
                    resolved_beatmap_path.as_posix() if resolved_beatmap_path is not None else None
                ),
                "beatmap_exists": bool(resolved_beatmap_path is not None and resolved_beatmap_path.is_file()),
                "metadata_json_path": metadata_json_path.as_posix(),
                "metadata_json_exists": metadata_json_path.is_file(),
                "beatmap_id": _optional_int(index_row.get("beatmap_id")),
            }
        )
    return rows


def _group_source_rows_by_resolved_audio(rows: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["resolved_audio_path"]), []).append(row)
    return [sorted(grouped[key], key=_map_sort_key) for key in sorted(grouped)]


def _build_audio_row(
    *,
    audio_group_index: int,
    group: Sequence[Mapping[str, Any]],
    config: BeatThisFramePredictionCacheConfig,
    duplicate_beatmap_ids: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    if not group:
        raise ValueError("group must be non-empty")

    resolved_audio_path = Path(str(group[0]["resolved_audio_path"]))
    map_rows = [_map_row_payload(row) for row in group]
    metadata_paths = _metadata_json_payloads(group)
    duplicate_ids_in_group = [
        duplicate_beatmap_ids[beatmap_id]
        for beatmap_id in sorted({int(row["beatmap_id"]) for row in group if row.get("beatmap_id") is not None})
        if beatmap_id in duplicate_beatmap_ids
    ]
    cache_payload = _cache_payload(resolved_audio_path=resolved_audio_path, config=config)
    anomalies = _audio_group_anomalies(
        group=group,
        metadata_paths=metadata_paths,
        duplicate_ids_in_group=duplicate_ids_in_group,
        cache_status=str(cache_payload["status"]),
    )

    audio_stat = _file_stat_payload(resolved_audio_path) if resolved_audio_path.is_file() else None
    return {
        "schema": TIMING_V3_INVENTORY_AUDIO_ROW_SCHEMA,
        "audio_group_index": audio_group_index,
        "audio_group_key": resolved_audio_path.as_posix(),
        "resolved_audio_path": resolved_audio_path.as_posix(),
        "audio_exists": resolved_audio_path.is_file(),
        "audio_file": audio_stat,
        "audio_path_values": sorted({str(row["audio_path"]) for row in group}),
        "shards": sorted({str(row["shard"]) for row in group}),
        "map_count": len(group),
        "maps": map_rows,
        "metadata_json": {
            "paths": metadata_paths,
            "path_count": len(metadata_paths),
            "existing_count": sum(1 for item in metadata_paths if item["exists"]),
        },
        "cache": cache_payload,
        "duplicate_beatmap_ids": duplicate_ids_in_group,
        "anomalies": anomalies,
    }


def _map_row_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_row_index": int(row["source_row_index"]),
        "shard": str(row["shard"]),
        "audio_path": str(row["audio_path"]),
        "beatmap_path": row["beatmap_path"],
        "resolved_beatmap_path": row["resolved_beatmap_path"],
        "beatmap_exists": bool(row["beatmap_exists"]),
        "metadata_json_path": str(row["metadata_json_path"]),
        "metadata_json_exists": bool(row["metadata_json_exists"]),
        "beatmap_id": row["beatmap_id"],
        "index_row": row["index_row"],
    }


def _metadata_json_payloads(group: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for raw_path in sorted({str(row["metadata_json_path"]) for row in group}):
        path = Path(raw_path)
        payload = {
            "path": path.as_posix(),
            "exists": path.is_file(),
            "sha256": _sha256(path) if path.is_file() else None,
            "file": _file_stat_payload(path) if path.is_file() else None,
        }
        payloads.append(payload)
    return payloads


def _cache_payload(*, resolved_audio_path: Path, config: BeatThisFramePredictionCacheConfig) -> dict[str, Any]:
    if not resolved_audio_path.is_file():
        return {
            "status": "missing_audio",
            "cache_path": None,
            "cache_exists": False,
            "cache_file_sha256": None,
            "audio_cache_key": None,
            "audio_cache_key_sha256": None,
            "config_fingerprint": config.config_fingerprint,
            "frame_count": None,
            "frame_rate_hz": config.frame_rate_hz,
            "duration_seconds": None,
            "provider": None,
            "checkpoint_path": config.checkpoint_path,
            "source_path": None,
            "error_type": "FileNotFoundError",
            "message": f"audio file does not exist: {resolved_audio_path.as_posix()}",
        }

    audio_cache_key = beatthis_audio_cache_key(resolved_audio_path)
    cache_path = beatthis_frame_prediction_cache_path(audio_cache_key, config)
    cache_exists = cache_path.is_file()
    base = {
        "cache_path": cache_path.as_posix(),
        "cache_exists": cache_exists,
        "cache_file_sha256": _sha256(cache_path) if cache_exists else None,
        "audio_cache_key": audio_cache_key,
        "audio_cache_key_sha256": _sha256_text(audio_cache_key),
        "config_fingerprint": config.config_fingerprint,
        "frame_rate_hz": config.frame_rate_hz,
        "checkpoint_path": config.checkpoint_path,
    }
    try:
        prediction = load_beatthis_frame_prediction_cache(audio_cache_key, config)
    except BeatThisFramePredictionCacheError as exc:
        return {
            **base,
            "status": "invalid",
            "frame_count": None,
            "duration_seconds": None,
            "provider": None,
            "source_path": None,
            "error_type": exc.__class__.__name__,
            "message": str(exc),
        }
    if prediction is None:
        return {
            **base,
            "status": "missing",
            "frame_count": None,
            "duration_seconds": None,
            "provider": None,
            "source_path": None,
            "error_type": None,
            "message": "BeatThis frame prediction cache is missing",
        }

    duration_seconds = prediction.frame_count / float(prediction.frame_rate_hz)
    return {
        **base,
        "status": "valid",
        "frame_count": prediction.frame_count,
        "duration_seconds": duration_seconds,
        "provider": prediction.provider,
        "source_path": prediction.source_path,
        "error_type": None,
        "message": None,
    }


def _audio_group_anomalies(
    *,
    group: Sequence[Mapping[str, Any]],
    metadata_paths: Sequence[Mapping[str, Any]],
    duplicate_ids_in_group: Sequence[Mapping[str, Any]],
    cache_status: str,
) -> list[str]:
    anomalies: set[str] = set()
    if len({str(row["audio_path"]) for row in group}) > 1:
        anomalies.add("multiple_index_audio_paths_for_resolved_audio")
    if len({str(row["shard"]) for row in group}) > 1:
        anomalies.add("multiple_shards_for_resolved_audio")
    if any(not bool(row["beatmap_exists"]) for row in group):
        anomalies.add("missing_beatmap_file")
    if not any(bool(item["exists"]) for item in metadata_paths):
        anomalies.add("missing_metadata_json")
    if sum(1 for item in metadata_paths if item["exists"]) > 1:
        anomalies.add("multiple_metadata_json_files")
    if duplicate_ids_in_group:
        anomalies.add("duplicate_beatmap_id")
    if _has_duplicate_map_paths(group):
        anomalies.add("duplicate_beatmap_path")
    if cache_status == "missing_audio":
        anomalies.add("missing_audio")
    elif cache_status == "missing":
        anomalies.add("missing_cache")
    elif cache_status == "invalid":
        anomalies.add("invalid_cache")
    return sorted(anomalies)


def _has_duplicate_map_paths(group: Sequence[Mapping[str, Any]]) -> bool:
    values = [
        (str(row["shard"]), str(row["beatmap_path"]))
        for row in group
        if row.get("beatmap_path") is not None
    ]
    return len(values) != len(set(values))


def _duplicate_beatmap_ids(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    by_id: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        beatmap_id = row.get("beatmap_id")
        if beatmap_id is None:
            continue
        by_id.setdefault(int(beatmap_id), []).append(row)

    duplicates: dict[int, dict[str, Any]] = {}
    for beatmap_id, id_rows in by_id.items():
        if len(id_rows) <= 1:
            continue
        duplicates[beatmap_id] = {
            "beatmap_id": beatmap_id,
            "count": len(id_rows),
            "source_row_indexes": sorted(int(row["source_row_index"]) for row in id_rows),
            "beatmap_paths": sorted({str(row["beatmap_path"]) for row in id_rows}),
            "resolved_audio_paths": sorted({str(row["resolved_audio_path"]) for row in id_rows}),
        }
    return duplicates


def _metadata_json_path_for_row(
    *,
    dataset_root: Path,
    shard: str,
    index_row: Mapping[str, Any],
    resolved_beatmap_path: Path | None,
    resolved_audio_path: Path,
) -> Path:
    beatmap_set_rel = _optional_relative_path(index_row.get("beatmap_set_path"), field_name="beatmap_set_path")
    if beatmap_set_rel is not None:
        return _resolve_dataset_path(
            dataset_root=dataset_root,
            shard=shard,
            relative_path=beatmap_set_rel,
            field_name="beatmap_set_path",
        ) / "metadata.json"
    if resolved_beatmap_path is not None:
        return resolved_beatmap_path.parent / "metadata.json"
    return resolved_audio_path.parent / "metadata.json"


def _cache_config_payload(config: BeatThisFramePredictionCacheConfig) -> dict[str, Any]:
    return {
        "cache_root": config.cache_root.as_posix(),
        "cache_version": config.cache_version,
        "checkpoint_path": config.checkpoint_path,
        "float16": config.float16,
        "shift_ms": config.shift_ms,
        "frame_rate_hz": config.frame_rate_hz,
        "config_fingerprint": config.config_fingerprint,
        "cache_dir": config.cache_dir.as_posix(),
    }


def _file_stat_payload(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _safe_relative_component(value: str, *, field_name: str) -> str:
    path = _safe_relative_path(value, field_name=field_name)
    if len(path.parts) != 1:
        raise ValueError(f"{field_name} must be a single relative path component, got {value!r}")
    return path.as_posix()


def _optional_relative_path(value: object, *, field_name: str) -> PurePosixPath | None:
    if value is None:
        return None
    return _safe_relative_path(str(value), field_name=field_name)


def _safe_relative_path(value: str, *, field_name: str) -> PurePosixPath:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute():
        raise ValueError(f"{field_name} must be relative, got {value!r}")
    if not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{field_name} must not contain empty, '.', or '..' components, got {value!r}")
    return path


def _resolve_dataset_path(
    *,
    dataset_root: Path,
    shard: str,
    relative_path: PurePosixPath,
    field_name: str,
) -> Path:
    candidate = dataset_root / shard
    candidate = candidate.joinpath(*relative_path.parts)
    resolved_root = dataset_root.resolve(strict=False)
    resolved_candidate = candidate.resolve(strict=False)
    if not resolved_candidate.is_relative_to(resolved_root):
        raise ValueError(f"resolved {field_name} escapes dataset_root: {candidate}")
    return resolved_candidate


def _map_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    index_row = row["index_row"]
    assert isinstance(index_row, Mapping)
    return (
        "" if index_row.get("beatmap_set_id") is None else str(index_row.get("beatmap_set_id")),
        "" if row.get("beatmap_id") is None else f"{int(row['beatmap_id']):012d}",
        "" if row.get("beatmap_path") is None else str(row.get("beatmap_path")),
        int(row["source_row_index"]),
    )


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = float(stripped)
        except ValueError:
            return None
        if math.isfinite(parsed) and parsed.is_integer():
            return int(parsed)
    return None


def _normalize_json_value(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        return _normalize_json_value(value.tolist())
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        return _normalize_json_value(value.item())
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _git_stdout(*args: str) -> str | None:
    try:
        completed = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _format_command(argv: Sequence[str] | None) -> str:
    args = list(sys.argv[1:] if argv is None else argv)
    return " ".join(shlex.quote(part) for part in ["python", "-m", "pulsefield_model.timing.evaluation.inventory", *args])


def _format_summary(report: Mapping[str, Any]) -> str:
    source = report["source"]
    output = report["output"]
    cache = report["cache"]
    anomalies = report["anomalies"]
    assert isinstance(source, Mapping)
    assert isinstance(output, Mapping)
    assert isinstance(cache, Mapping)
    assert isinstance(anomalies, Mapping)
    return (
        "timing-v3 inventory: "
        f"{output['audio_group_count']}/{source['audio_group_count']} audio groups, "
        f"{output['map_row_count']}/{source['map_row_count']} maps, "
        f"cache valid/missing/invalid/missing-audio="
        f"{cache['valid_count']}/{cache['missing_count']}/{cache['invalid_count']}/{cache['missing_audio_count']}, "
        f"anomaly groups={anomalies['audio_group_count']}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
