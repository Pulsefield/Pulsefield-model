from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .control import mania_hit_objects_to_control_hits
from .control import red_timing_points_to_beat_length_fn
from .control_v2_artifact import (
    _empty_column,
    _git_dirty,
    _git_rev_parse,
    _load_index_with_source_index,
    _sha256_file,
    _summary_values,
)
from .control_v3 import (
    CONFIDENCE_FEATURE_NAMES,
    DEBUG_ARRAY_NAMES,
    MODEL_FEATURE_NAMES,
    VALUE_FEATURE_NAMES,
    FeatureConfigV3,
    extract_control_features,
)
from .control_v3_audit import DIAGNOSTIC_FEATURE_CONTRACT, FEATURE_CONTRACT, MODEL_FEATURE_CONTRACT
from pulsefield_model.osu_core.hitobjects import parse_mania_hit_objects
from pulsefield_model.osu_core.timing import require_red_timing_points


CONTROL_V3_SCHEMA_VERSION = 3
CONTROL_V3_FEATURE_CONTRACT_VERSION = 3

CONTROL_V3_METADATA_COLUMNS = [
    "filtered_index",
    "source_index",
    "beatmap_id",
    "beatmap_set_id",
    "difficulty",
    "time_s",
]

CONTROL_V3_DIAGNOSTIC_COLUMNS = [
    name if name not in MODEL_FEATURE_NAMES else f"{name}_debug_raw"
    for name in DEBUG_ARRAY_NAMES
]

CONTROL_V3_ARTIFACT_FAMILY = "4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3"

DEFAULT_INDEX_PATH = Path(
    f"artifacts/indexes/beatmap_index_{CONTROL_V3_ARTIFACT_FAMILY}.parquet"
)
DEFAULT_SOURCE_INDEX_PATH = Path("artifacts/indexes/beatmap_index_4k_no_timing_anomalies.parquet")
DEFAULT_DATASET_ROOT = Path("dataset")
DEFAULT_TIMESERIES_PATH = Path(
    f"artifacts/features/control_v3_timeseries_{CONTROL_V3_ARTIFACT_FAMILY}.parquet"
)
DEFAULT_SUMMARY_PATH = Path(
    f"artifacts/features/control_v3_map_summary_{CONTROL_V3_ARTIFACT_FAMILY}.parquet"
)
DEFAULT_METADATA_PATH = Path(
    f"artifacts/features/control_v3_artifact_metadata_{CONTROL_V3_ARTIFACT_FAMILY}.json"
)


def build_timeseries_frame(
    *,
    row: pd.Series,
    filtered_index: int,
    source_index: int,
    out: dict[str, Any],
) -> pd.DataFrame:
    time_s = np.asarray(out["time"], dtype=np.float32)
    frame = pd.DataFrame(
        {
            "filtered_index": np.full(len(time_s), int(filtered_index), dtype=np.int32),
            "source_index": np.full(len(time_s), int(source_index), dtype=np.int32),
            "beatmap_id": np.full(len(time_s), int(row["beatmap_id"]), dtype=np.int64),
            "beatmap_set_id": np.full(len(time_s), int(row["beatmap_set_id"]), dtype=np.int64),
            "difficulty": np.full(len(time_s), float(row["difficulty"]), dtype=np.float32),
            "time_s": time_s,
        }
    )
    for name in MODEL_FEATURE_NAMES:
        frame[name] = np.asarray(out["features"][name], dtype=np.float32)

    for source_name, column_name in zip(DEBUG_ARRAY_NAMES, CONTROL_V3_DIAGNOSTIC_COLUMNS):
        value = out["debug"].get(source_name)
        if value is None:
            frame[column_name] = _empty_column(source_name, len(time_s))
            continue
        array = np.asarray(value)
        if array.shape != time_s.shape:
            frame[column_name] = _empty_column(source_name, len(time_s))
            continue
        if source_name == "valid_control_mask":
            frame[column_name] = array.astype(bool, copy=False)
        else:
            frame[column_name] = array.astype(np.float32, copy=False)
    return frame


def summarize_map_output(
    *,
    row: pd.Series,
    filtered_index: int,
    source_index: int,
    hitobjects: int,
    timing_points: int,
    duration_s: float,
    parse_s: float,
    convert_s: float,
    feature_s: float,
    total_s: float,
    out: dict[str, Any] | None,
    error_type: str,
    error: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "filtered_index": int(filtered_index),
        "source_index": int(source_index),
        "beatmap_id": int(row["beatmap_id"]),
        "beatmap_set_id": int(row["beatmap_set_id"]),
        "difficulty": float(row["difficulty"]),
        "shard": str(row["shard"]),
        "beatmap_path": str(row["beatmap_path"]),
        "title": str(row.get("title", "")),
        "artist": str(row.get("artist", "")),
        "version": str(row.get("version", "")),
        "hitobjects": int(hitobjects),
        "timing_points": int(timing_points),
        "duration_s": float(duration_s),
        "grid_rows": int(len(out["time"]) if out is not None else 0),
        "onsets": int(len(out["debug"].get("onsets", [])) if out is not None else 0),
        "parse_s": float(parse_s),
        "convert_s": float(convert_s),
        "feature_s": float(feature_s),
        "total_s": float(total_s),
        "finite": False,
        "ranges_ok": False,
        "notes": "",
        "error_type": error_type,
        "error": error,
    }
    if out is None:
        return summary

    all_finite = True
    ranges_ok = True
    for name in MODEL_FEATURE_NAMES:
        values = _summary_values(out, name)
        if len(values) == 0:
            summary[f"{name}_min"] = 0.0
            summary[f"{name}_max"] = 0.0
            summary[f"{name}_mean"] = 0.0
            continue
        finite = values[np.isfinite(values)]
        all_finite = all_finite and len(finite) == len(values)
        if len(finite) == 0:
            summary[f"{name}_min"] = math.nan
            summary[f"{name}_max"] = math.nan
            summary[f"{name}_mean"] = math.nan
            ranges_ok = False
            continue
        summary[f"{name}_min"] = float(np.min(finite))
        summary[f"{name}_max"] = float(np.max(finite))
        summary[f"{name}_mean"] = float(np.mean(finite))
        ranges_ok = ranges_ok and bool(np.max(np.abs(finite)) < 1.0e6)

    summary["finite"] = bool(all_finite)
    summary["ranges_ok"] = bool(ranges_ok)
    return summary


def build_control_v3_artifacts(
    *,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    source_index_path: str | Path = DEFAULT_SOURCE_INDEX_PATH,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    timeseries_path: str | Path = DEFAULT_TIMESERIES_PATH,
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    limit: int | None = None,
    start: int = 0,
    batch_maps: int = 32,
    progress_every: int = 25,
    cfg: FeatureConfigV3 | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if cfg is None:
        cfg = FeatureConfigV3()
    index_path = Path(index_path)
    source_index_path = Path(source_index_path)
    dataset_root = Path(dataset_root)
    timeseries_path = Path(timeseries_path)
    summary_path = Path(summary_path)
    metadata_path = Path(metadata_path)
    timeseries_tmp = timeseries_path.with_suffix(timeseries_path.suffix + ".tmp")
    summary_tmp = summary_path.with_suffix(summary_path.suffix + ".tmp")
    metadata_tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")

    for path in (timeseries_tmp, summary_tmp, metadata_tmp):
        if path.exists():
            path.unlink()
    timeseries_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    index_df = _load_index_with_source_index(index_path, source_index_path)
    if start:
        index_df = index_df.iloc[start:].reset_index(drop=True)
    if limit is not None:
        index_df = index_df.head(limit).reset_index(drop=True)

    writer: pq.ParquetWriter | None = None
    frame_batch: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    total_rows = 0
    started_at = time.perf_counter()

    def flush_frames() -> None:
        nonlocal frame_batch, writer, total_rows
        if not frame_batch:
            return
        chunk = pd.concat(frame_batch, ignore_index=True)
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(timeseries_tmp, table.schema, compression="zstd")
        writer.write_table(table)
        total_rows += len(chunk)
        frame_batch = []

    try:
        for position, row in enumerate(index_df.itertuples(index=False), start=1):
            row_series = pd.Series(row._asdict())
            filtered_index = int(row_series["filtered_index"])
            source_index = int(row_series["source_index"])
            map_start = time.perf_counter()
            parse_s = convert_s = feature_s = 0.0
            hitobject_count = 0
            timing_point_count = 0
            duration_s = 0.0
            out: dict[str, Any] | None = None
            error_type = ""
            error = ""
            try:
                parse_start = time.perf_counter()
                beatmap_path = dataset_root / str(row_series["shard"]) / str(row_series["beatmap_path"])
                hitobjects = parse_mania_hit_objects(beatmap_path, expected_key_count=4)
                timing_points = require_red_timing_points(beatmap_path)
                parse_s = time.perf_counter() - parse_start

                convert_start = time.perf_counter()
                hits = mania_hit_objects_to_control_hits(hitobjects)
                beat_length_at = red_timing_points_to_beat_length_fn(timing_points)
                duration_s = max((hit.end if hit.end is not None else hit.start for hit in hits), default=0.0)
                hitobject_count = len(hitobjects)
                timing_point_count = len(timing_points)
                convert_s = time.perf_counter() - convert_start

                feature_start = time.perf_counter()
                out = extract_control_features(
                    hits,
                    beat_length_at=beat_length_at,
                    cfg=cfg,
                    start_time=0.0,
                    end_time=duration_s,
                    return_debug=True,
                )
                feature_s = time.perf_counter() - feature_start
                frame_batch.append(
                    build_timeseries_frame(
                        row=row_series,
                        filtered_index=filtered_index,
                        source_index=source_index,
                        out=out,
                    )
                )
                if len(frame_batch) >= batch_maps:
                    flush_frames()
            except Exception as exc:  # pragma: no cover - exercised by real corpus runs
                error_type = type(exc).__name__
                error = str(exc)

            total_s = time.perf_counter() - map_start
            summary_rows.append(
                summarize_map_output(
                    row=row_series,
                    filtered_index=filtered_index,
                    source_index=source_index,
                    hitobjects=hitobject_count,
                    timing_points=timing_point_count,
                    duration_s=duration_s,
                    parse_s=parse_s,
                    convert_s=convert_s,
                    feature_s=feature_s,
                    total_s=total_s,
                    out=out,
                    error_type=error_type,
                    error=error,
                )
            )
            if progress_every and position % progress_every == 0:
                elapsed = time.perf_counter() - started_at
                print(
                    "progress "
                    f"maps={position}/{len(index_df)} "
                    f"timeseries_rows={total_rows + sum(len(frame) for frame in frame_batch)} "
                    f"elapsed_s={elapsed:.1f} "
                    f"last_filtered_index={filtered_index} "
                    f"last_feature_s={feature_s:.3f} "
                    f"errors={sum(1 for item in summary_rows if item['error_type'])}",
                    flush=True,
                )
        flush_frames()
    finally:
        if writer is not None:
            writer.close()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_parquet(summary_tmp, index=False, compression="snappy")
    metadata = {
        "schema_version": CONTROL_V3_SCHEMA_VERSION,
        "feature_contract_version": CONTROL_V3_FEATURE_CONTRACT_VERSION,
        "artifact_name": f"control_v3_{CONTROL_V3_ARTIFACT_FAMILY}",
        "index_path": index_path.as_posix(),
        "index_sha256": _sha256_file(index_path),
        "source_index_path": source_index_path.as_posix(),
        "source_index_sha256": _sha256_file(source_index_path) if source_index_path.exists() else "",
        "dataset_root": dataset_root.as_posix(),
        "timeseries_path": timeseries_path.as_posix(),
        "summary_path": summary_path.as_posix(),
        "feature_names": MODEL_FEATURE_NAMES,
        "model_feature_names": MODEL_FEATURE_NAMES,
        "value_feature_names": VALUE_FEATURE_NAMES,
        "confidence_feature_names": CONFIDENCE_FEATURE_NAMES,
        "diagnostic_columns": CONTROL_V3_DIAGNOSTIC_COLUMNS,
        "feature_contract": FEATURE_CONTRACT,
        "model_feature_contract": MODEL_FEATURE_CONTRACT,
        "diagnostic_feature_contract": DIAGNOSTIC_FEATURE_CONTRACT,
        "config": vars(cfg),
        "map_count": int(len(index_df)),
        "timeseries_rows": int(total_rows),
        "error_count": int(summary_df["error_type"].astype(bool).sum()) if not summary_df.empty else 0,
        "code_commit": _git_rev_parse("HEAD"),
        "code_dirty": _git_dirty(),
        "command": " ".join(sys.argv),
        "elapsed_s": float(time.perf_counter() - started_at),
    }
    metadata_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    os.replace(timeseries_tmp, timeseries_path)
    os.replace(summary_tmp, summary_path)
    os.replace(metadata_tmp, metadata_path)
    return summary_df, metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Stage 1 Control Feature V3 parquet artifacts.")
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH.as_posix())
    parser.add_argument("--source-index-path", default=DEFAULT_SOURCE_INDEX_PATH.as_posix())
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT.as_posix())
    parser.add_argument("--timeseries-path", default=DEFAULT_TIMESERIES_PATH.as_posix())
    parser.add_argument("--summary-path", default=DEFAULT_SUMMARY_PATH.as_posix())
    parser.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH.as_posix())
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch-maps", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args(argv)

    summary_df, metadata = build_control_v3_artifacts(
        index_path=args.index_path,
        source_index_path=args.source_index_path,
        dataset_root=args.dataset_root,
        timeseries_path=args.timeseries_path,
        summary_path=args.summary_path,
        metadata_path=args.metadata_path,
        limit=args.limit,
        start=args.start,
        batch_maps=args.batch_maps,
        progress_every=args.progress_every,
    )
    print(f"map_count {len(summary_df)}")
    print(f"timeseries_rows {metadata['timeseries_rows']}")
    print(f"error_count {metadata['error_count']}")
    print(f"timeseries_path {args.timeseries_path}")
    print(f"summary_path {args.summary_path}")
    print(f"metadata_path {args.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
