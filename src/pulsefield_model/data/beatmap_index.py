from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, TypeAlias, cast, runtime_checkable

import numpy as np
import pandas as pd

from pulsefield_model.osu_core.difficulty import calculate_mania_difficulties
from pulsefield_model.osu_core.metadata import parse_osu_metadata
from pulsefield_model.osu_core.timing import InvalidRedTimingError, MissingRedTimingError, require_red_timing_points
from pulsefield_model.timing.rendering.dense_timing_v2 import DENSE_TIMING_V2_CHANNELS, DENSE_TIMING_V2_VERSION
from pulsefield_model.timing.schema import FittedTimingGrid, TimingSegment


INDEX_4K_FILENAME = "beatmap_index_4k.parquet"
INDEX_4K_NO_TIMING_ANOMALIES_FILENAME = "beatmap_index_4k_no_timing_anomalies.parquet"
INDEX_4K_NO_TIMING_ANOMALIES_2TO6_FILENAME = "beatmap_index_4k_no_timing_anomalies_2to6.parquet"
INDEX_4K_NO_TIMING_ANOMALIES_2TO6_DENSE_LOCAL_BPM_NORM_UNIQUE_LE3_FILENAME = (
    "beatmap_index_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.parquet"
)
DEFAULT_DATASET_ROOT = Path("dataset")
DEFAULT_SHARD = "0"
DEFAULT_INDEX_ROOT = Path("artifacts/indexes")
DEFAULT_REPORT_ROOT = Path("artifacts/reports/indexes")
DEFAULT_4K_INDEX_PATH = DEFAULT_INDEX_ROOT / INDEX_4K_FILENAME
DEFAULT_4K_NO_TIMING_ANOMALY_INDEX_PATH = DEFAULT_INDEX_ROOT / INDEX_4K_NO_TIMING_ANOMALIES_FILENAME
DEFAULT_4K_NO_TIMING_ANOMALY_2TO6_INDEX_PATH = DEFAULT_INDEX_ROOT / INDEX_4K_NO_TIMING_ANOMALIES_2TO6_FILENAME
DEFAULT_DENSE_TIMING_V2_LOCAL_BPM_NORM_UNIQUE_INDEX_PATH = (
    DEFAULT_INDEX_ROOT / INDEX_4K_NO_TIMING_ANOMALIES_2TO6_DENSE_LOCAL_BPM_NORM_UNIQUE_LE3_FILENAME
)
DEFAULT_DENSE_TIMING_V2_LOCAL_BPM_NORM_UNIQUE_REPORT_PATH = (
    DEFAULT_REPORT_ROOT
    / INDEX_4K_NO_TIMING_ANOMALIES_2TO6_DENSE_LOCAL_BPM_NORM_UNIQUE_LE3_FILENAME.replace(".parquet", ".json")
)
DEFAULT_MAX_LOCAL_BPM_NORM_UNIQUE_PER_BEATMAPSET = 3
DEFAULT_BPM_ROUND_DECIMALS = 6
SR_SPEEDS = (0.5, 0.75, 1.0, 1.25, 1.5)
NULLABLE_INT_COLUMNS = (
    "audio_lead_in",
    "preview_time",
    "mode",
    "beatmap_id",
    "key_count",
)
FLOAT_COLUMNS = (
    "hp_drain_rate",
    "circle_size",
    "overall_difficulty",
    "difficulty",
)
DIFFICULTY_MIN = 2.0
DIFFICULTY_MAX = 6.0
_LOCAL_BPM_CHANNEL = DENSE_TIMING_V2_CHANNELS.index("local_bpm")
_REQUIRED_DENSE_INDEX_COLUMNS = frozenset(("shard", "beatmap_set_id", "beatmap_path"))
NormalizedValue: TypeAlias = None | bool | int | float | str | bytes | list["NormalizedValue"]


@runtime_checkable
class _SupportsToList(Protocol):
    def tolist(self) -> object: ...


@runtime_checkable
class _SupportsItem(Protocol):
    def item(self) -> object: ...


@dataclass(frozen=True)
class BeatmapIndexRecord:
    shard: str
    beatmap_set_id: int | str
    beatmap_set_path: str
    beatmap_path: str
    beatmap_filename: str
    audio_path: str
    audio_filename: str
    audio_lead_in: int | None
    preview_time: int | None
    mode: int | None
    title: str | None
    artist: str | None
    creator: str | None
    version: str | None
    beatmap_id: int | None
    hp_drain_rate: float | None
    circle_size: float | None
    overall_difficulty: float | None
    key_count: int | None
    difficulty: float
    sr_difficulties: list[float]


@dataclass(frozen=True)
class TimingCleanIndexReport:
    source_index_path: Path
    output_path: Path
    dataset_root: Path
    source_map_count: int
    clean_map_count: int
    missing_red_timing_map_count: int
    invalid_red_timing_map_count: int
    invalid_red_timing_point_count: int
    nonfinite_red_timing_point_count: int
    nonpositive_red_timing_point_count: int
    implausible_red_timing_point_count: int


@dataclass(frozen=True)
class DifficultyFilteredIndexReport:
    source_index_path: Path
    output_path: Path
    source_map_count: int
    retained_map_count: int
    min_difficulty: float
    max_difficulty: float


@dataclass(frozen=True)
class DenseTimingV2LocalBpmNormUniqueIndexReport:
    source_index_path: Path
    output_path: Path
    dataset_root: Path
    source_map_count: int
    retained_map_count: int
    dropped_map_count: int
    source_beatmapset_count: int
    retained_beatmapset_count: int
    dropped_beatmapset_count: int
    max_local_bpm_norm_unique_per_beatmapset: int
    max_observed_local_bpm_norm_unique_per_beatmapset: int
    bpm_round_decimals: int
    global_min_local_bpm: float | None
    global_max_local_bpm: float | None
    dropped_examples: list[dict[str, Any]]


def build_4k_index(
    shard_path: str | Path,
    output_path: str | Path = DEFAULT_4K_INDEX_PATH,
) -> Path:
    return build_filtered_index(shard_path=shard_path, output_path=output_path, key_count=4)


def build_filtered_index(
    *,
    shard_path: str | Path,
    output_path: str | Path,
    key_count: int | None = None,
) -> Path:
    shard_path = Path(shard_path)
    output_path = Path(output_path)
    rows: list[dict[str, object]] = []

    for beatmap_set_path in sorted(path for path in shard_path.iterdir() if path.is_dir()):
        for beatmap_path in sorted(beatmap_set_path.glob("*.osu")):
            metadata = parse_osu_metadata(beatmap_path)
            if metadata.mode != 3:
                continue
            if key_count is not None and metadata.key_count != key_count:
                continue

            record = _build_record(shard_path, beatmap_set_path, beatmap_path, metadata=metadata)
            if record is None:
                continue
            rows.append(asdict(record))

    index_df = pd.DataFrame.from_records(
        rows,
        columns=[
            "shard",
            "beatmap_set_id",
            "beatmap_set_path",
            "beatmap_path",
            "beatmap_filename",
            "audio_path",
            "audio_filename",
            "audio_lead_in",
            "preview_time",
            "mode",
            "title",
            "artist",
            "creator",
            "version",
            "beatmap_id",
            "hp_drain_rate",
            "circle_size",
            "overall_difficulty",
            "key_count",
            "difficulty",
            "sr_difficulties",
        ],
    )
    index_df = _cast_index_dtypes(index_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    index_df.to_parquet(tmp_output_path, index=False)
    tmp_output_path.replace(output_path)
    return output_path


def build_4k_no_timing_anomaly_index(
    *,
    source_index_path: str | Path = DEFAULT_4K_INDEX_PATH,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    output_path: str | Path = DEFAULT_4K_NO_TIMING_ANOMALY_INDEX_PATH,
) -> TimingCleanIndexReport:
    source_index_path = Path(source_index_path)
    dataset_root = Path(dataset_root)
    output_path = Path(output_path)
    source_df = load_index(source_index_path)

    keep_mask: list[bool] = []
    missing_red_timing_map_count = 0
    invalid_red_timing_map_count = 0
    invalid_red_timing_point_count = 0
    nonfinite_red_timing_point_count = 0
    nonpositive_red_timing_point_count = 0
    implausible_red_timing_point_count = 0

    for row in source_df.itertuples(index=False):
        beatmap_path = dataset_root / str(row.shard) / str(row.beatmap_path)
        try:
            require_red_timing_points(beatmap_path)
        except InvalidRedTimingError as exc:
            keep_mask.append(False)
            invalid_red_timing_map_count += 1
            invalid_red_timing_point_count += exc.counts.total
            nonfinite_red_timing_point_count += exc.counts.nonfinite
            nonpositive_red_timing_point_count += exc.counts.nonpositive
            implausible_red_timing_point_count += exc.counts.implausible
        except MissingRedTimingError:
            keep_mask.append(False)
            missing_red_timing_map_count += 1
        else:
            keep_mask.append(True)

    clean_df = _cast_index_dtypes(source_df.loc[keep_mask].reset_index(drop=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    clean_df.to_parquet(tmp_output_path, index=False)
    tmp_output_path.replace(output_path)

    return TimingCleanIndexReport(
        source_index_path=source_index_path,
        output_path=output_path,
        dataset_root=dataset_root,
        source_map_count=len(source_df),
        clean_map_count=len(clean_df),
        missing_red_timing_map_count=missing_red_timing_map_count,
        invalid_red_timing_map_count=invalid_red_timing_map_count,
        invalid_red_timing_point_count=invalid_red_timing_point_count,
        nonfinite_red_timing_point_count=nonfinite_red_timing_point_count,
        nonpositive_red_timing_point_count=nonpositive_red_timing_point_count,
        implausible_red_timing_point_count=implausible_red_timing_point_count,
    )


def build_difficulty_filtered_index(
    *,
    source_index_path: str | Path = DEFAULT_4K_NO_TIMING_ANOMALY_INDEX_PATH,
    output_path: str | Path = DEFAULT_4K_NO_TIMING_ANOMALY_2TO6_INDEX_PATH,
    min_difficulty: float = DIFFICULTY_MIN,
    max_difficulty: float = DIFFICULTY_MAX,
) -> DifficultyFilteredIndexReport:
    if not math.isfinite(min_difficulty) or not math.isfinite(max_difficulty):
        raise ValueError("difficulty bounds must be finite")
    if min_difficulty > max_difficulty:
        raise ValueError("min_difficulty must be <= max_difficulty")

    source_index_path = Path(source_index_path)
    output_path = Path(output_path)
    source_df = load_index(source_index_path)
    difficulty = pd.to_numeric(source_df["difficulty"], errors="coerce")
    output_df = _cast_index_dtypes(
        source_df[(difficulty >= min_difficulty) & (difficulty <= max_difficulty)].reset_index(drop=True),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    output_df.to_parquet(tmp_output_path, index=False)
    tmp_output_path.replace(output_path)
    return DifficultyFilteredIndexReport(
        source_index_path=source_index_path,
        output_path=output_path,
        source_map_count=len(source_df),
        retained_map_count=len(output_df),
        min_difficulty=float(min_difficulty),
        max_difficulty=float(max_difficulty),
    )


def build_dense_timing_v2_local_bpm_norm_unique_index(
    *,
    source_index_path: str | Path = DEFAULT_4K_NO_TIMING_ANOMALY_2TO6_INDEX_PATH,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    output_path: str | Path = DEFAULT_DENSE_TIMING_V2_LOCAL_BPM_NORM_UNIQUE_INDEX_PATH,
    report_path: str | Path | None = DEFAULT_DENSE_TIMING_V2_LOCAL_BPM_NORM_UNIQUE_REPORT_PATH,
    max_local_bpm_norm_unique_per_beatmapset: int = DEFAULT_MAX_LOCAL_BPM_NORM_UNIQUE_PER_BEATMAPSET,
    bpm_round_decimals: int = DEFAULT_BPM_ROUND_DECIMALS,
    dropped_example_limit: int = 20,
    progress_every: int = 0,
    command: str | None = None,
) -> DenseTimingV2LocalBpmNormUniqueIndexReport:
    if max_local_bpm_norm_unique_per_beatmapset < 1:
        raise ValueError(
            "max_local_bpm_norm_unique_per_beatmapset must be positive, "
            f"got {max_local_bpm_norm_unique_per_beatmapset!r}",
        )
    if bpm_round_decimals < 0:
        raise ValueError(f"bpm_round_decimals must be non-negative, got {bpm_round_decimals!r}")
    if dropped_example_limit < 0:
        raise ValueError(f"dropped_example_limit must be non-negative, got {dropped_example_limit!r}")

    started_at = time.perf_counter()
    source_index_path = Path(source_index_path)
    dataset_root = Path(dataset_root)
    output_path = Path(output_path)
    report_path = None if report_path is None else Path(report_path)

    source_df = pd.read_parquet(source_index_path)
    _require_dense_index_columns(source_df, source_index_path)

    group_rows: dict[tuple[str, str], list[int]] = {}
    group_values: dict[tuple[str, str], set[float]] = {}
    group_metadata: dict[tuple[str, str], dict[str, Any]] = {}
    local_bpm_min: float | None = None
    local_bpm_max: float | None = None

    for row_index, row in enumerate(source_df.itertuples(index=False), start=1):
        group_key = _beatmapset_group_key(row)
        local_bpms = dense_timing_v2_local_bpms_for_beatmap(dataset_root, row)
        if local_bpms.size:
            row_min = float(np.min(local_bpms))
            row_max = float(np.max(local_bpms))
            local_bpm_min = row_min if local_bpm_min is None else min(local_bpm_min, row_min)
            local_bpm_max = row_max if local_bpm_max is None else max(local_bpm_max, row_max)

        group_rows.setdefault(group_key, []).append(row_index - 1)
        group_values.setdefault(group_key, set()).update(_local_bpm_norm_unique_values(local_bpms, bpm_round_decimals))
        group_metadata.setdefault(group_key, _group_metadata(row))

        if progress_every > 0 and row_index % progress_every == 0:
            print(f"processed {row_index}/{len(source_df)} maps", file=sys.stderr)

    dropped_groups = {
        key
        for key, values in group_values.items()
        if len(values) > max_local_bpm_norm_unique_per_beatmapset
    }
    keep_mask = [_beatmapset_group_key(row) not in dropped_groups for row in source_df.itertuples(index=False)]
    output_df = _cast_index_dtypes(source_df.loc[keep_mask].reset_index(drop=True))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    output_df.to_parquet(tmp_output_path, index=False)
    tmp_output_path.replace(output_path)

    report = DenseTimingV2LocalBpmNormUniqueIndexReport(
        source_index_path=source_index_path,
        output_path=output_path,
        dataset_root=dataset_root,
        source_map_count=len(source_df),
        retained_map_count=len(output_df),
        dropped_map_count=len(source_df) - len(output_df),
        source_beatmapset_count=len(group_rows),
        retained_beatmapset_count=len(group_rows) - len(dropped_groups),
        dropped_beatmapset_count=len(dropped_groups),
        max_local_bpm_norm_unique_per_beatmapset=max_local_bpm_norm_unique_per_beatmapset,
        max_observed_local_bpm_norm_unique_per_beatmapset=max(
            (len(values) for values in group_values.values()),
            default=0,
        ),
        bpm_round_decimals=bpm_round_decimals,
        global_min_local_bpm=local_bpm_min,
        global_max_local_bpm=local_bpm_max,
        dropped_examples=_dropped_examples(
            dropped_groups,
            group_rows=group_rows,
            group_values=group_values,
            group_metadata=group_metadata,
            limit=dropped_example_limit,
        ),
    )

    if report_path is not None:
        _write_dense_index_report(
            report_path,
            report,
            source_index_sha256=_sha256(source_index_path),
            output_sha256=_sha256(output_path),
            elapsed_s=time.perf_counter() - started_at,
            command=command,
        )
    return report


def dense_timing_v2_local_bpms_for_beatmap(dataset_root: str | Path, row: Mapping[str, object] | object) -> np.ndarray:
    beatmap_path = Path(dataset_root) / str(_row_value(row, "shard")) / str(_row_value(row, "beatmap_path"))
    red_timing_points = require_red_timing_points(beatmap_path)
    grid = _timing_grid_from_red_timing_points(red_timing_points)
    return np.asarray([segment.local_bpm for segment in grid.segments], dtype=np.float64)


def load_index(index_path: str | Path) -> pd.DataFrame:
    return _cast_index_dtypes(pd.read_parquet(index_path))


def _build_record(
    shard_path: Path,
    beatmap_set_path: Path,
    beatmap_path: Path,
    *,
    metadata: Any,
) -> BeatmapIndexRecord | None:
    if not metadata.audio_filename:
        return None
    if metadata.mode != 3:
        return None

    audio_path = beatmap_set_path / metadata.audio_filename
    if not audio_path.is_file():
        return None

    beatmap_set_id: int | str
    if metadata.beatmap_set_id is not None:
        beatmap_set_id = metadata.beatmap_set_id
    else:
        try:
            beatmap_set_id = int(beatmap_set_path.name)
        except ValueError:
            beatmap_set_id = beatmap_set_path.name

    sr_difficulties = [_round_2f(value) for value in calculate_mania_difficulties(beatmap_path, audio_path, SR_SPEEDS)]
    difficulty = sr_difficulties[SR_SPEEDS.index(1.0)]

    return BeatmapIndexRecord(
        shard=shard_path.name,
        beatmap_set_id=beatmap_set_id,
        beatmap_set_path=beatmap_set_path.relative_to(shard_path).as_posix(),
        beatmap_path=beatmap_path.relative_to(shard_path).as_posix(),
        beatmap_filename=beatmap_path.name,
        audio_path=audio_path.relative_to(shard_path).as_posix(),
        audio_filename=metadata.audio_filename,
        audio_lead_in=metadata.audio_lead_in,
        preview_time=metadata.preview_time,
        mode=metadata.mode,
        title=metadata.title,
        artist=metadata.artist,
        creator=metadata.creator,
        version=metadata.version,
        beatmap_id=metadata.beatmap_id,
        hp_drain_rate=metadata.hp_drain_rate,
        circle_size=metadata.circle_size,
        overall_difficulty=metadata.overall_difficulty,
        key_count=metadata.key_count,
        difficulty=difficulty,
        sr_difficulties=sr_difficulties,
    )


def _cast_index_dtypes(index_df: pd.DataFrame) -> pd.DataFrame:
    for column in NULLABLE_INT_COLUMNS:
        if column in index_df.columns:
            index_df[column] = pd.to_numeric(index_df[column], errors="coerce").astype("Int64")

    for column in FLOAT_COLUMNS:
        if column in index_df.columns:
            index_df[column] = pd.to_numeric(index_df[column], errors="coerce").astype("Float64")

    return index_df


def _round_2f(value: float) -> float:
    return float(f"{value:.2f}")


def _normalize_scalar(value: object) -> NormalizedValue:
    if isinstance(value, list):
        return [_normalize_scalar(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_scalar(item) for item in value]
    if isinstance(value, _SupportsToList) and not isinstance(value, (str, bytes)):
        return _normalize_scalar(value.tolist())
    if pd.isna(value):
        return None
    if isinstance(value, _SupportsItem) and not isinstance(value, (str, bytes)):
        return _normalize_scalar(value.item())
    return cast(NormalizedValue, value)


def _require_dense_index_columns(index_df: pd.DataFrame, index_path: Path) -> None:
    missing_columns = sorted(_REQUIRED_DENSE_INDEX_COLUMNS.difference(index_df.columns))
    if missing_columns:
        raise ValueError(f"{index_path} is missing required column(s): {missing_columns}")


def _timing_grid_from_red_timing_points(red_timing_points: Sequence[object]) -> FittedTimingGrid:
    segments_by_offset: dict[float, TimingSegment] = {}
    for point in red_timing_points:
        offset_ms = float(point.offset_ms)
        segments_by_offset[offset_ms] = TimingSegment(
            offset_ms=offset_ms,
            beat_length_ms=float(point.beat_length_ms),
            meter=int(getattr(point, "meter", 4)),
        )
    return FittedTimingGrid(tuple(segments_by_offset[offset] for offset in sorted(segments_by_offset)))


def _beatmapset_group_key(row: Mapping[str, object] | object) -> tuple[str, str]:
    return str(_row_value(row, "shard")), str(_row_value(row, "beatmap_set_id"))


def _row_value(row: Mapping[str, object] | object, name: str) -> object:
    if isinstance(row, Mapping):
        return row[name]
    return getattr(row, name)


def _has_row_value(row: Mapping[str, object] | object, name: str) -> bool:
    if isinstance(row, Mapping):
        return name in row
    return hasattr(row, name)


def _group_metadata(row: Mapping[str, object] | object) -> dict[str, Any]:
    return {
        "shard": str(_row_value(row, "shard")),
        "beatmap_set_id": _json_scalar(_row_value(row, "beatmap_set_id")),
        "beatmap_set_path": str(_row_value(row, "beatmap_set_path")) if _has_row_value(row, "beatmap_set_path") else None,
        "title": str(_row_value(row, "title")) if _has_row_value(row, "title") else None,
        "artist": str(_row_value(row, "artist")) if _has_row_value(row, "artist") else None,
    }


def _local_bpm_norm_unique_values(local_bpms: np.ndarray, decimals: int) -> set[float]:
    return {round(float(value), decimals) for value in local_bpms}


def _dropped_examples(
    dropped_groups: set[tuple[str, str]],
    *,
    group_rows: Mapping[tuple[str, str], Sequence[int]],
    group_values: Mapping[tuple[str, str], set[float]],
    group_metadata: Mapping[tuple[str, str], Mapping[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    ranked_groups = sorted(
        dropped_groups,
        key=lambda key: (len(group_values[key]), len(group_rows[key]), key[0], key[1]),
        reverse=True,
    )
    for key in ranked_groups[:limit]:
        values = sorted(group_values[key])
        examples.append(
            {
                **dict(group_metadata[key]),
                "map_count": len(group_rows[key]),
                "local_bpm_norm_unique_count": len(values),
                "local_bpm_norm_values_head": values[:12],
            }
        )
    return examples


def _write_dense_index_report(
    report_path: Path,
    report: DenseTimingV2LocalBpmNormUniqueIndexReport,
    *,
    source_index_sha256: str,
    output_sha256: str,
    elapsed_s: float,
    command: str | None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "source_index_path": report.source_index_path.as_posix(),
        "source_index_sha256": source_index_sha256,
        "output_path": report.output_path.as_posix(),
        "output_sha256": output_sha256,
        "dataset_root": report.dataset_root.as_posix(),
        "dense_timing_version": DENSE_TIMING_V2_VERSION,
        "dense_timing_channels": list(DENSE_TIMING_V2_CHANNELS),
        "local_bpm_channel_index": _LOCAL_BPM_CHANNEL,
        "local_bpm_norm_definition": (
            "dense_timing_v2 segment local_bpm rounded to bpm_round_decimals for stable unique counting"
        ),
        "drop_policy": (
            "drop_entire_shard_beatmap_set_id_group_when_dense_timing_v2_local_bpm_norm_unique_count_exceeds_threshold"
        ),
        "max_local_bpm_norm_unique_per_beatmapset": report.max_local_bpm_norm_unique_per_beatmapset,
        "max_observed_local_bpm_norm_unique_per_beatmapset": (
            report.max_observed_local_bpm_norm_unique_per_beatmapset
        ),
        "bpm_round_decimals": report.bpm_round_decimals,
        "source_map_count": report.source_map_count,
        "retained_map_count": report.retained_map_count,
        "dropped_map_count": report.dropped_map_count,
        "source_beatmapset_count": report.source_beatmapset_count,
        "retained_beatmapset_count": report.retained_beatmapset_count,
        "dropped_beatmapset_count": report.dropped_beatmapset_count,
        "global_min_local_bpm": report.global_min_local_bpm,
        "global_max_local_bpm": report.global_max_local_bpm,
        "dropped_examples": report.dropped_examples,
        "elapsed_s": elapsed_s,
        "code_commit": _git_stdout("rev-parse", "HEAD"),
        "code_dirty": bool(_git_stdout("status", "--porcelain")),
        "command": command,
    }
    tmp_report_path = report_path.with_suffix(report_path.suffix + ".tmp")
    tmp_report_path.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_report_path.replace(report_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_stdout(*args: str) -> str | None:
    try:
        completed = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _format_command(argv: Sequence[str] | None, subcommand: str) -> str:
    args = list(sys.argv[1:] if argv is None else argv)
    return " ".join(
        shlex.quote(part)
        for part in ["python", "-m", "pulsefield_model.data.beatmap_index", subcommand, *args]
    )


def _json_scalar(value: object) -> object:
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return value.item()
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Pulsefield beatmap indexes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_4k_parser = subparsers.add_parser("build-4k")
    build_4k_parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    build_4k_parser.add_argument("--shard", default=DEFAULT_SHARD)
    build_4k_parser.add_argument("--output-path", type=Path, default=DEFAULT_4K_INDEX_PATH)

    clean_parser = subparsers.add_parser("drop-timing-anomalies")
    clean_parser.add_argument("--source-index-path", type=Path, default=DEFAULT_4K_INDEX_PATH)
    clean_parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    clean_parser.add_argument("--output-path", type=Path, default=DEFAULT_4K_NO_TIMING_ANOMALY_INDEX_PATH)

    difficulty_parser = subparsers.add_parser("filter-difficulty")
    difficulty_parser.add_argument("--source-index-path", type=Path, default=DEFAULT_4K_NO_TIMING_ANOMALY_INDEX_PATH)
    difficulty_parser.add_argument("--output-path", type=Path, default=DEFAULT_4K_NO_TIMING_ANOMALY_2TO6_INDEX_PATH)
    difficulty_parser.add_argument("--min-difficulty", type=float, default=DIFFICULTY_MIN)
    difficulty_parser.add_argument("--max-difficulty", type=float, default=DIFFICULTY_MAX)

    dense_parser = subparsers.add_parser("filter-local-bpm-unique")
    dense_parser.add_argument("--source-index-path", type=Path, default=DEFAULT_4K_NO_TIMING_ANOMALY_2TO6_INDEX_PATH)
    dense_parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    dense_parser.add_argument("--output-path", type=Path, default=DEFAULT_DENSE_TIMING_V2_LOCAL_BPM_NORM_UNIQUE_INDEX_PATH)
    dense_parser.add_argument("--report-path", type=Path, default=DEFAULT_DENSE_TIMING_V2_LOCAL_BPM_NORM_UNIQUE_REPORT_PATH)
    dense_parser.add_argument(
        "--max-local-bpm-norm-unique-per-beatmapset",
        "--max-localbpmnorm-unique-per-beatmapset",
        dest="max_local_bpm_norm_unique_per_beatmapset",
        type=int,
        default=DEFAULT_MAX_LOCAL_BPM_NORM_UNIQUE_PER_BEATMAPSET,
    )
    dense_parser.add_argument("--bpm-round-decimals", type=int, default=DEFAULT_BPM_ROUND_DECIMALS)
    dense_parser.add_argument("--dropped-example-limit", type=int, default=20)
    dense_parser.add_argument("--progress-every", type=int, default=0)

    args = parser.parse_args(argv)
    if args.command == "build-4k":
        output_path = build_4k_index(args.dataset_root / args.shard, args.output_path)
        print(f"index_path {output_path}")
        return 0
    if args.command == "drop-timing-anomalies":
        report = build_4k_no_timing_anomaly_index(
            source_index_path=args.source_index_path,
            dataset_root=args.dataset_root,
            output_path=args.output_path,
        )
        print(f"clean_map_count {report.clean_map_count}")
        print(f"index_path {report.output_path}")
        return 0
    if args.command == "filter-difficulty":
        report = build_difficulty_filtered_index(
            source_index_path=args.source_index_path,
            output_path=args.output_path,
            min_difficulty=args.min_difficulty,
            max_difficulty=args.max_difficulty,
        )
        print(f"retained_map_count {report.retained_map_count}")
        print(f"index_path {report.output_path}")
        return 0
    if args.command == "filter-local-bpm-unique":
        report = build_dense_timing_v2_local_bpm_norm_unique_index(
            source_index_path=args.source_index_path,
            dataset_root=args.dataset_root,
            output_path=args.output_path,
            report_path=args.report_path,
            max_local_bpm_norm_unique_per_beatmapset=args.max_local_bpm_norm_unique_per_beatmapset,
            bpm_round_decimals=args.bpm_round_decimals,
            dropped_example_limit=args.dropped_example_limit,
            progress_every=args.progress_every,
            command=_format_command(argv, "filter-local-bpm-unique"),
        )
        print(
            "retained "
            f"{report.retained_map_count}/{report.source_map_count} maps; "
            f"dropped {report.dropped_beatmapset_count} beatmapsets",
        )
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
