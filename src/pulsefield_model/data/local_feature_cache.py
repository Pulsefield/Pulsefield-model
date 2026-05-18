from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Sequence

import numpy as np
import pandas as pd

from pulsefield_model.data.control_windows import DEFAULT_DATASET_ROOT
from pulsefield_model.data.control_windows import DEFAULT_MAP_INDEX_PATH
from pulsefield_model.data.control_windows import _audio_frame_count_20ms
from pulsefield_model.data.control_windows import _resolve_index_path
from pulsefield_model.data.control_windows import _resolve_shard_root
from pulsefield_model.features.mel import DEFAULT_STAGE2_MEL_CONFIG
from pulsefield_model.features.mel import Stage2MelConfig
from pulsefield_model.features.mel import load_full_song_packed_mel_20ms
from pulsefield_model.features.mel import stage2_log_mel_cache_path
from pulsefield_model.timing.providers.oracle import DEFAULT_ORACLE_DENSE_TIMING_CACHE_CONFIG
from pulsefield_model.timing.providers.oracle import DEFAULT_ORACLE_TIMING_CONFIG
from pulsefield_model.timing.providers.oracle import OracleDenseTimingCacheConfig
from pulsefield_model.timing.providers.oracle import OracleTimingConfig
from pulsefield_model.timing.providers.oracle import load_or_create_oracle_dense_timing_v2_cache
from pulsefield_model.timing.providers.oracle import oracle_dense_timing_v2_cache_path


_REQUIRED_INDEX_COLUMNS = frozenset(("shard", "audio_path", "beatmap_path"))


@dataclass(frozen=True)
class LocalFeatureCacheBuildReport:
    index_path: Path
    dataset_root: Path
    row_count: int
    unique_audio_count: int
    unique_beatmap_count: int
    mel_enabled: bool
    oracle_timing_enabled: bool
    mel_existing: int
    mel_created: int
    oracle_timing_existing: int
    oracle_timing_created: int
    elapsed_s: float


def build_local_mel_and_oracle_timing_cache(
    *,
    index_path: str | Path = DEFAULT_MAP_INDEX_PATH,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    limit: int | None = None,
    progress_every: int = 100,
    build_mel: bool = True,
    build_oracle_timing: bool = True,
    mel_config: Stage2MelConfig = DEFAULT_STAGE2_MEL_CONFIG,
    oracle_timing_config: OracleTimingConfig = DEFAULT_ORACLE_TIMING_CONFIG,
    oracle_cache_config: OracleDenseTimingCacheConfig = DEFAULT_ORACLE_DENSE_TIMING_CACHE_CONFIG,
) -> LocalFeatureCacheBuildReport:
    started_at = time.perf_counter()
    index_path = Path(index_path)
    dataset_root = Path(dataset_root)
    index_df = pd.read_parquet(index_path)
    _require_columns(index_df, index_path)
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be positive when provided, got {limit!r}")
        index_df = index_df.head(limit)

    audio_frame_counts: dict[tuple[str, str], int] = {}
    mel_existing = 0
    mel_created = 0
    for audio_number, row in enumerate(_unique_audio_rows(index_df), start=1):
        audio_path = _audio_path_for_row(dataset_root, row)
        cache_path = stage2_log_mel_cache_path(audio_path, config=mel_config)
        if build_mel and cache_path.exists():
            frame_count = _cached_mel_20ms_frame_count(cache_path)
            mel_existing += 1
        elif build_mel:
            frame_count = int(load_full_song_packed_mel_20ms(audio_path, config=mel_config).shape[0])
            mel_created += 1
        elif hasattr(row, "frame_count"):
            frame_count = _positive_frame_count(row.frame_count)
        elif cache_path.exists():
            frame_count = _cached_mel_20ms_frame_count(cache_path)
        else:
            frame_count = _audio_frame_count_20ms(audio_path)
        audio_frame_counts[(str(row.shard), str(row.audio_path))] = frame_count

        if build_mel and _should_print_progress(audio_number, progress_every):
            print(
                "local_feature_cache_progress "
                f"kind=mel audio={audio_number} existing={mel_existing} created={mel_created}",
                flush=True,
            )
        elif not build_mel and build_oracle_timing and _should_print_progress(audio_number, progress_every):
            print(
                "local_feature_cache_progress "
                f"kind=frame_count audio={audio_number}",
                flush=True,
            )

    oracle_timing_existing = 0
    oracle_timing_created = 0
    seen_timing_keys: set[tuple[str, str, int]] = set()
    if build_oracle_timing:
        row_iterator = index_df.itertuples(index=False)
    else:
        row_iterator = ()
    for row_number, row in enumerate(row_iterator, start=1):
        frame_count = _frame_count_for_row(row, audio_frame_counts)
        timing_key = (str(row.shard), str(row.beatmap_path), frame_count)
        if timing_key in seen_timing_keys:
            continue
        seen_timing_keys.add(timing_key)

        beatmap_path = _beatmap_path_for_row(dataset_root, row)
        cache_path = oracle_dense_timing_v2_cache_path(
            beatmap_path,
            frame_count=frame_count,
            timing_config=oracle_timing_config,
            cache_config=oracle_cache_config,
        )
        if cache_path.exists():
            oracle_timing_existing += 1
        else:
            load_or_create_oracle_dense_timing_v2_cache(
                beatmap_path,
                frame_count=frame_count,
                timing_config=oracle_timing_config,
                cache_config=oracle_cache_config,
            )
            oracle_timing_created += 1

        processed_timing = oracle_timing_existing + oracle_timing_created
        if _should_print_progress(processed_timing, progress_every):
            print(
                "local_feature_cache_progress "
                f"kind=oracle_timing beatmaps={processed_timing} "
                f"existing={oracle_timing_existing} created={oracle_timing_created}",
                flush=True,
            )

    return LocalFeatureCacheBuildReport(
        index_path=index_path,
        dataset_root=dataset_root,
        row_count=len(index_df),
        unique_audio_count=len(audio_frame_counts),
        unique_beatmap_count=len(seen_timing_keys),
        mel_enabled=build_mel,
        oracle_timing_enabled=build_oracle_timing,
        mel_existing=mel_existing,
        mel_created=mel_created,
        oracle_timing_existing=oracle_timing_existing,
        oracle_timing_created=oracle_timing_created,
        elapsed_s=time.perf_counter() - started_at,
    )


def _unique_audio_rows(index_df: pd.DataFrame) -> list[Any]:
    rows: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for row in index_df.itertuples(index=False):
        key = (str(row.shard), str(row.audio_path))
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def _frame_count_for_row(row: Any, audio_frame_counts: dict[tuple[str, str], int]) -> int:
    if hasattr(row, "frame_count"):
        return _positive_frame_count(row.frame_count)

    audio_key = (str(row.shard), str(row.audio_path))
    try:
        return audio_frame_counts[audio_key]
    except KeyError as exc:
        raise ValueError(f"missing frame count for audio {audio_key}") from exc


def _positive_frame_count(value: object) -> int:
    frame_count = int(value)
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count!r}")
    return frame_count


def _cached_mel_20ms_frame_count(cache_path: Path) -> int:
    mel = np.load(cache_path, mmap_mode="r")
    try:
        if mel.ndim != 2 or mel.shape[1] != 80:
            raise ValueError(f"mel cache must have shape [frames,80], got {mel.shape}: {cache_path}")
        return (int(mel.shape[0]) + 1) // 2
    finally:
        del mel


def _beatmap_path_for_row(dataset_root: Path, row: Any) -> Path:
    shard_root = _resolve_shard_root(dataset_root, row.shard)
    return _resolve_index_path(dataset_root, shard_root, row.beatmap_path, field="beatmap_path")


def _audio_path_for_row(dataset_root: Path, row: Any) -> Path:
    shard_root = _resolve_shard_root(dataset_root, row.shard)
    return _resolve_index_path(dataset_root, shard_root, row.audio_path, field="audio_path")


def _require_columns(index_df: pd.DataFrame, index_path: Path) -> None:
    missing = sorted(_REQUIRED_INDEX_COLUMNS.difference(index_df.columns))
    if missing:
        raise ValueError(f"{index_path} is missing required column(s): {missing}")


def _should_print_progress(count: int, progress_every: int) -> bool:
    return progress_every > 0 and (count == 1 or count % progress_every == 0)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build local mel and oracle dense timing caches from an index.")
    parser.add_argument("--index-path", type=Path, default=DEFAULT_MAP_INDEX_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--mel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--oracle-timing", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    report = build_local_mel_and_oracle_timing_cache(
        index_path=args.index_path,
        dataset_root=args.dataset_root,
        limit=args.limit,
        progress_every=args.progress_every,
        build_mel=bool(args.mel),
        build_oracle_timing=bool(args.oracle_timing),
    )
    printable = {
        key: value.as_posix() if isinstance(value, Path) else value
        for key, value in asdict(report).items()
    }
    print(f"local_feature_cache_done {printable}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
