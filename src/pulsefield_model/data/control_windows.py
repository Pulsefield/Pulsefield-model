from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pulsefield_model.features.control_v3_targets import CONFIDENCE_FEATURE_NAMES
from pulsefield_model.features.control_v3_targets import DEFAULT_TIMESERIES_PATH as DEFAULT_CONTROL_V3_TIMESERIES_PATH
from pulsefield_model.features.control_v3_targets import MODEL_FEATURE_NAMES


DEFAULT_MAP_INDEX_PATH = Path(
    "artifacts/indexes/"
    "beatmap_index_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.parquet"
)
DEFAULT_CONTROL_WINDOW_INDEX_PATH = Path(
    "artifacts/indexes/"
    "stage2_control_windows_4k_2to6_dense_local_bpm_norm_unique_le3.parquet"
)
DEFAULT_CONTROL_WINDOW_INDEX_REPORT_PATH = Path(
    "artifacts/reports/indexes/"
    "stage2_control_windows_4k_2to6_dense_local_bpm_norm_unique_le3.json"
)
DEFAULT_CONTROL_V3_SUMMARY_PATH = Path(
    "artifacts/features/control_v3_map_summary_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.parquet"
)
DEFAULT_DATASET_ROOT = Path("dataset")
DEFAULT_INDEX_PATH = DEFAULT_CONTROL_WINDOW_INDEX_PATH
DEFAULT_MAX_CACHED_MAPS = 16

DIFFICULTY_MIN = 2.0
DIFFICULTY_MAX = 6.0
FRAME_HOP_MS = 20
TARGET_WINDOW_LENGTH_FRAMES = 100
TARGET_WINDOW_STRIDE_FRAMES = TARGET_WINDOW_LENGTH_FRAMES
CONTEXT_LENGTH_FRAMES = 600
TARGET_OFFSET_IN_CONTEXT = 250
PACKED_MEL_CHANNELS = 160
DENSE_TIMING_V2_CHANNELS = 4
CONTROL_V3_TARGET_CHANNELS = 20

_REQUIRED_INDEX_COLUMNS = frozenset(("shard", "audio_path", "beatmap_path", "difficulty"))
_WINDOW_INDEX_COLUMNS = frozenset(("frame_count", "target_start_frame"))
_CONTROL_V3_SUMMARY_REQUIRED_COLUMNS = frozenset(
    ("filtered_index", "source_index", "beatmap_path", "error_type", "finite", "ranges_ok"),
)

FullMelLoader = Callable[[Path], Any]
DenseTimingV2Loader = Callable[[Path, int], Any]
AudioFrameCountLoader = Callable[[Path], int]


@dataclass(frozen=True)
class ControlWindowRecord:
    beatmap_path: Path
    audio_path: Path
    difficulty: float
    frame_count: int
    target_start_frame: int
    beatmap_id: int | None = None
    filtered_index: int | None = None
    source_index: int | None = None

    @property
    def target_start_ms(self) -> int:
        return self.target_start_frame * FRAME_HOP_MS


ControlV3TargetLoader = Callable[[ControlWindowRecord], Any]
LnChangeNEffTargetLoader = Callable[[ControlWindowRecord], Any]


@dataclass(frozen=True)
class ControlWindowIndexBuildReport:
    source_index_path: Path
    control_v3_summary_path: Path
    output_path: Path
    dataset_root: Path
    source_map_count: int
    retained_map_count: int
    unique_audio_count: int
    window_count: int
    min_frame_count: int | None
    max_frame_count: int | None
    target_window_length_frames: int
    target_window_stride_frames: int
    elapsed_s: float


class ControlWindowDataset(Dataset):
    """Stage 2 control-window dataset over filtered mania beatmap index rows."""

    def __init__(
        self,
        index_path: str | Path = DEFAULT_INDEX_PATH,
        *,
        dataset_root: str | Path = DEFAULT_DATASET_ROOT,
        mel_loader: FullMelLoader | None = None,
        timing_loader: DenseTimingV2Loader | None = None,
        target_loader: ControlV3TargetLoader | None = None,
        ln_change_n_eff_target_loader: LnChangeNEffTargetLoader | None = None,
        allow_missing_ln_change_n_eff_target: bool = False,
        control_v3_timeseries_path: str | Path = DEFAULT_CONTROL_V3_TIMESERIES_PATH,
        max_cached_maps: int = DEFAULT_MAX_CACHED_MAPS,
        progress: bool = False,
    ) -> None:
        self.index_path = Path(index_path)
        self.dataset_root = Path(dataset_root)
        self.control_v3_timeseries_path = Path(control_v3_timeseries_path)
        self.mel_loader = _default_load_full_song_packed_mel if mel_loader is None else mel_loader
        self.timing_loader = _default_load_oracle_dense_timing_v2 if timing_loader is None else timing_loader
        self.target_loader = self._default_load_control_v3_target_window if target_loader is None else target_loader
        if ln_change_n_eff_target_loader is None:
            if target_loader is None:
                self.ln_change_n_eff_target_loader = self._default_load_ln_change_n_eff_target_window
            elif allow_missing_ln_change_n_eff_target:
                self.ln_change_n_eff_target_loader = _full_support_ln_change_n_eff_target
            else:
                raise ValueError(
                    "custom target_loader requires ln_change_n_eff_target_loader, "
                    "or set allow_missing_ln_change_n_eff_target=True for explicit full-support test data"
                )
        else:
            self.ln_change_n_eff_target_loader = ln_change_n_eff_target_loader
        self.max_cached_maps = _validate_cache_size(max_cached_maps)
        self._full_mel_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._dense_timing_cache: OrderedDict[tuple[str, int], torch.Tensor] = OrderedDict()
        self._control_v3_rows_cache: OrderedDict[tuple[str, int], pd.DataFrame] = OrderedDict()

        index_df = pd.read_parquet(self.index_path)
        _require_index_columns(index_df, self.index_path)
        self.source_row_count = len(index_df)
        self.source_map_count = _map_count(index_df)
        index_df = filter_supported_difficulty_range(index_df)
        self.filtered_row_count = len(index_df)
        self.filtered_map_count = _map_count(index_df)
        self.records = self._build_records(index_df, progress=progress)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]

        full_mel = self._load_full_mel(record.audio_path, expected_frame_count=record.frame_count)
        full_dense_timing_v2 = self._load_dense_timing_v2(record.beatmap_path, frame_count=record.frame_count)
        control_v3_target = _validate_control_v3_target(
            self.target_loader(record),
            source=record.beatmap_path,
        )

        return {
            "full_mel": full_mel,
            "full_dense_timing_v2": full_dense_timing_v2,
            "control_v3_target": control_v3_target,
            "ln_change_n_eff_target": _validate_ln_change_n_eff_target(
                self.ln_change_n_eff_target_loader(record),
                source=record.beatmap_path,
            ),
            "target_valid_mask": target_valid_mask(
                target_start_frame=record.target_start_frame,
                frame_count=record.frame_count,
            ),
            "difficulty": torch.tensor(record.difficulty, dtype=torch.float32),
            "normalized_difficulty": torch.tensor(normalize_difficulty(record.difficulty), dtype=torch.float32),
            "target_start_frame": torch.tensor(record.target_start_frame, dtype=torch.long),
            "target_start_ms": torch.tensor(record.target_start_ms, dtype=torch.long),
            "frame_count": torch.tensor(record.frame_count, dtype=torch.long),
            "beatmap_path": record.beatmap_path.as_posix(),
            "audio_path": record.audio_path.as_posix(),
            "beatmap_id": _optional_int_tensor(record.beatmap_id),
            "filtered_index": _optional_int_tensor(record.filtered_index),
            "source_index": _optional_int_tensor(record.source_index),
        }

    def _build_records(self, index_df: pd.DataFrame, *, progress: bool) -> list[ControlWindowRecord]:
        if _WINDOW_INDEX_COLUMNS.issubset(index_df.columns):
            return self._build_records_from_window_index(index_df, progress=progress)
        return self._build_records_from_map_index(index_df, progress=progress)

    def _build_records_from_window_index(self, index_df: pd.DataFrame, *, progress: bool) -> list[ControlWindowRecord]:
        records: list[ControlWindowRecord] = []
        path_cache: dict[tuple[str, str, str], tuple[Path, Path]] = {}
        for row_number, row in enumerate(index_df.itertuples(index=False), start=1):
            difficulty = _validate_difficulty(row.difficulty)
            path_key = (str(row.shard), str(row.beatmap_path), str(row.audio_path))
            paths = path_cache.get(path_key)
            if paths is None:
                shard_root = _resolve_shard_root(self.dataset_root, row.shard)
                paths = (
                    _resolve_index_path(self.dataset_root, shard_root, row.beatmap_path, field="beatmap_path"),
                    _resolve_index_path(self.dataset_root, shard_root, row.audio_path, field="audio_path"),
                )
                path_cache[path_key] = paths
            beatmap_path, audio_path = paths
            frame_count = _validate_positive_int(row.frame_count, "frame_count")
            target_start_frame = _validate_nonnegative_int(row.target_start_frame, "target_start_frame")
            if target_start_frame >= frame_count:
                raise ValueError(
                    f"target_start_frame must be less than frame_count: {target_start_frame} >= {frame_count}"
                )
            if hasattr(row, "target_start_ms"):
                target_start_ms = _validate_nonnegative_int(row.target_start_ms, "target_start_ms")
                expected_ms = target_start_frame * FRAME_HOP_MS
                if target_start_ms != expected_ms:
                    raise ValueError(f"target_start_ms mismatch: {target_start_ms} != {expected_ms}")

            records.append(
                ControlWindowRecord(
                    beatmap_path=beatmap_path,
                    audio_path=audio_path,
                    difficulty=difficulty,
                    frame_count=frame_count,
                    target_start_frame=target_start_frame,
                    beatmap_id=_optional_row_int(row, "beatmap_id"),
                    filtered_index=_optional_row_int(row, "filtered_index"),
                    source_index=_optional_row_int(row, "source_index"),
                )
            )

            if progress and (row_number == 1 or row_number % 10000 == 0):
                print(
                    f"control_window_dataset_progress scanned_rows={row_number} "
                    f"records={len(records)}",
                    flush=True,
                )
        return records

    def _build_records_from_map_index(self, index_df: pd.DataFrame, *, progress: bool) -> list[ControlWindowRecord]:
        records: list[ControlWindowRecord] = []
        for row_number, row in enumerate(index_df.itertuples(index=False), start=1):
            difficulty = _validate_difficulty(row.difficulty)
            shard_root = _resolve_shard_root(self.dataset_root, row.shard)
            beatmap_path = _resolve_index_path(self.dataset_root, shard_root, row.beatmap_path, field="beatmap_path")
            audio_path = _resolve_index_path(self.dataset_root, shard_root, row.audio_path, field="audio_path")
            full_mel = self._load_full_mel(audio_path)
            frame_count = int(full_mel.shape[0])
            beatmap_id = _optional_row_int(row, "beatmap_id")
            filtered_index = _optional_row_int(row, "filtered_index")
            source_index = _optional_row_int(row, "source_index")

            for target_start_frame in iter_target_start_frames(frame_count):
                records.append(
                    ControlWindowRecord(
                        beatmap_path=beatmap_path,
                        audio_path=audio_path,
                        difficulty=difficulty,
                        frame_count=frame_count,
                        target_start_frame=target_start_frame,
                        beatmap_id=beatmap_id,
                        filtered_index=filtered_index,
                        source_index=source_index,
                    )
                )

            if progress and (row_number == 1 or row_number % 100 == 0):
                print(
                    f"control_window_dataset_progress scanned_maps={row_number} "
                    f"records={len(records)}",
                    flush=True,
                )
        return records

    def _load_full_mel(self, audio_path: Path, *, expected_frame_count: int | None = None) -> torch.Tensor:
        key = audio_path.as_posix()
        cached = _lru_get(self._full_mel_cache, key)
        if cached is None:
            cached = _validate_full_mel(self.mel_loader(audio_path), source=audio_path)
            _lru_put(self._full_mel_cache, key, cached, max_items=self.max_cached_maps)
        if expected_frame_count is not None and int(cached.shape[0]) != expected_frame_count:
            raise ValueError(
                f"full_mel for {audio_path} must have frame_count={expected_frame_count}, got shape {tuple(cached.shape)}"
            )
        return cached

    def _load_dense_timing_v2(self, beatmap_path: Path, *, frame_count: int) -> torch.Tensor:
        key = (beatmap_path.as_posix(), frame_count)
        cached = _lru_get(self._dense_timing_cache, key)
        if cached is None:
            cached = _validate_dense_timing_v2(
                self.timing_loader(beatmap_path, frame_count),
                expected_frame_count=frame_count,
                source=beatmap_path,
            )
            _lru_put(self._dense_timing_cache, key, cached, max_items=self.max_cached_maps)
        return cached

    def _default_load_control_v3_target_window(self, record: ControlWindowRecord) -> Any:
        from pulsefield_model.features.control_v3_targets import slice_control_v3_target_window

        window_start_s = record.target_start_frame * FRAME_HOP_MS / 1000.0
        rows = self._load_control_v3_rows(_target_selector(record))
        return slice_control_v3_target_window(rows, window_start_s)

    def _default_load_ln_change_n_eff_target_window(self, record: ControlWindowRecord) -> Any:
        from pulsefield_model.features.control_v3_targets import slice_ln_change_n_eff_target_window

        window_start_s = record.target_start_frame * FRAME_HOP_MS / 1000.0
        rows = self._load_control_v3_rows(_target_selector(record))
        return slice_ln_change_n_eff_target_window(rows, window_start_s)

    def _load_control_v3_rows(self, selector: tuple[str, int]) -> pd.DataFrame:
        cached = _lru_get(self._control_v3_rows_cache, selector)
        if cached is None:
            cached = _default_control_v3_rows(self.control_v3_timeseries_path, selector)
            _lru_put(self._control_v3_rows_cache, selector, cached, max_items=self.max_cached_maps)
        return cached


def collate_control_windows(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("collate_control_windows requires at least one sample")

    batch_size = len(samples)
    frame_counts = [int(sample["frame_count"].item()) for sample in samples]
    max_frame_count = max(frame_counts)

    full_mel = torch.zeros((batch_size, max_frame_count, PACKED_MEL_CHANNELS), dtype=torch.float32)
    full_dense_timing_v2 = torch.zeros((batch_size, max_frame_count, DENSE_TIMING_V2_CHANNELS), dtype=torch.float32)
    padding_mask = torch.ones((batch_size, max_frame_count), dtype=torch.bool)

    for batch_index, sample in enumerate(samples):
        sample_full_mel = _validate_full_mel(
            sample["full_mel"],
            expected_frame_count=frame_counts[batch_index],
            source=sample["audio_path"],
        )
        sample_full_dense_timing_v2 = _validate_dense_timing_v2(
            sample["full_dense_timing_v2"],
            expected_frame_count=frame_counts[batch_index],
            source=sample["beatmap_path"],
        )

        frame_count = frame_counts[batch_index]
        full_mel[batch_index, :frame_count] = sample_full_mel
        full_dense_timing_v2[batch_index, :frame_count] = sample_full_dense_timing_v2
        padding_mask[batch_index, :frame_count] = False

    return {
        "full_mel": full_mel,
        "full_dense_timing_v2": full_dense_timing_v2,
        "padding_mask": padding_mask,
        "frame_count": torch.tensor(frame_counts, dtype=torch.long),
        "control_v3_target": torch.stack(
            [_validate_control_v3_target(sample["control_v3_target"], source=sample["beatmap_path"]) for sample in samples]
        ),
        "ln_change_n_eff_target": torch.stack(
            [
                _validate_ln_change_n_eff_target(sample["ln_change_n_eff_target"], source=sample["beatmap_path"])
                for sample in samples
            ]
        ),
        "target_valid_mask": torch.stack(
            [_validate_target_valid_mask(sample["target_valid_mask"], source=sample["beatmap_path"]) for sample in samples]
        ),
        "difficulty": torch.stack([sample["difficulty"] for sample in samples]).reshape(batch_size),
        "normalized_difficulty": torch.stack([sample["normalized_difficulty"] for sample in samples]).reshape(batch_size),
        "target_start_frame": torch.stack([sample["target_start_frame"] for sample in samples]).reshape(batch_size),
        "target_start_ms": torch.stack([sample["target_start_ms"] for sample in samples]).reshape(batch_size),
        "metadata": [
            {
                "beatmap_path": sample["beatmap_path"],
                "audio_path": sample["audio_path"],
                "difficulty": float(sample["difficulty"].item()),
                "target_start_frame": int(sample["target_start_frame"].item()),
                "target_start_ms": int(sample["target_start_ms"].item()),
                "beatmap_id": _metadata_optional_int(sample, "beatmap_id"),
                "filtered_index": _metadata_optional_int(sample, "filtered_index"),
                "source_index": _metadata_optional_int(sample, "source_index"),
            }
            for sample in samples
        ],
    }


def collate_control_context_windows(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("collate_control_context_windows requires at least one sample")

    batch_size = len(samples)
    context_mel = torch.zeros((batch_size, CONTEXT_LENGTH_FRAMES, PACKED_MEL_CHANNELS), dtype=torch.float32)
    context_dense_timing_v2 = torch.zeros(
        (batch_size, CONTEXT_LENGTH_FRAMES, DENSE_TIMING_V2_CHANNELS),
        dtype=torch.float32,
    )
    context_padding_mask = torch.ones((batch_size, CONTEXT_LENGTH_FRAMES), dtype=torch.bool)
    frame_counts: list[int] = []
    target_start_frames: list[int] = []
    target_start_ms_values: list[int] = []
    for sample in samples:
        source = sample.get("beatmap_path", "<sample>")
        frame_count = _sample_positive_int(sample, "frame_count", source=source)
        target_start_frame = _sample_nonnegative_int(sample, "target_start_frame", source=source)
        if target_start_frame >= frame_count:
            raise ValueError(
                f"target_start_frame for {source} must be less than frame_count: "
                f"{target_start_frame} >= {frame_count}"
            )
        target_start_ms = _sample_nonnegative_int(sample, "target_start_ms", source=source)
        expected_ms = target_start_frame * FRAME_HOP_MS
        if target_start_ms != expected_ms:
            raise ValueError(f"target_start_ms for {source} must equal {expected_ms}, got {target_start_ms}")
        frame_counts.append(frame_count)
        target_start_frames.append(target_start_frame)
        target_start_ms_values.append(target_start_ms)

    for batch_index, sample in enumerate(samples):
        frame_count = frame_counts[batch_index]
        target_start_frame = target_start_frames[batch_index]
        sample_full_mel = _context_source_feature_matrix(
            sample["full_mel"],
            name="full_mel",
            expected_channels=PACKED_MEL_CHANNELS,
            expected_frame_count=frame_count,
            source=sample["audio_path"],
        )
        sample_full_dense_timing_v2 = _context_source_feature_matrix(
            sample["full_dense_timing_v2"],
            name="full_dense_timing_v2",
            expected_channels=DENSE_TIMING_V2_CHANNELS,
            expected_frame_count=frame_count,
            source=sample["beatmap_path"],
        )

        context_start_frame = target_start_frame - TARGET_OFFSET_IN_CONTEXT
        source_start = max(context_start_frame, 0)
        source_end = min(context_start_frame + CONTEXT_LENGTH_FRAMES, frame_count)
        if source_start >= source_end:
            continue
        destination_start = source_start - context_start_frame
        destination_end = destination_start + (source_end - source_start)
        context_mel[batch_index, destination_start:destination_end] = sample_full_mel[source_start:source_end]
        context_dense_timing_v2[batch_index, destination_start:destination_end] = sample_full_dense_timing_v2[
            source_start:source_end
        ]
        if not torch.isfinite(context_mel[batch_index, destination_start:destination_end]).all():
            raise ValueError(f"context_mel for {sample['audio_path']} must contain only finite values")
        if not torch.isfinite(context_dense_timing_v2[batch_index, destination_start:destination_end]).all():
            raise ValueError(
                f"context_dense_timing_v2 for {sample['beatmap_path']} must contain only finite values"
            )
        context_padding_mask[batch_index, destination_start:destination_end] = False

    target_offsets = torch.arange(TARGET_WINDOW_LENGTH_FRAMES, dtype=torch.long)
    target_valid_masks = torch.stack(
        [
            target_start_frame + target_offsets < frame_count
            for target_start_frame, frame_count in zip(target_start_frames, frame_counts, strict=True)
        ]
    )

    return {
        "context_mel": context_mel,
        "context_dense_timing_v2": context_dense_timing_v2,
        "context_padding_mask": context_padding_mask,
        "control_v3_target": torch.stack(
            [_validate_control_v3_target(sample["control_v3_target"], source=sample["beatmap_path"]) for sample in samples]
        ),
        "ln_change_n_eff_target": torch.stack(
            [
                _validate_ln_change_n_eff_target(sample["ln_change_n_eff_target"], source=sample["beatmap_path"])
                for sample in samples
            ]
        ),
        "target_valid_mask": target_valid_masks,
        "difficulty": torch.stack([sample["difficulty"] for sample in samples]).reshape(batch_size),
        "normalized_difficulty": torch.stack([sample["normalized_difficulty"] for sample in samples]).reshape(batch_size),
        "target_start_frame": torch.tensor(target_start_frames, dtype=torch.long),
        "target_start_ms": torch.tensor(target_start_ms_values, dtype=torch.long),
        "frame_count": torch.tensor(frame_counts, dtype=torch.long),
        "context_start_frame": torch.tensor(
            [target_start_frame - TARGET_OFFSET_IN_CONTEXT for target_start_frame in target_start_frames],
            dtype=torch.long,
        ),
        "target_offset_in_context": torch.full((batch_size,), TARGET_OFFSET_IN_CONTEXT, dtype=torch.long),
        "metadata": [
            {
                "beatmap_path": sample["beatmap_path"],
                "audio_path": sample["audio_path"],
                "difficulty": float(sample["difficulty"].item()),
                "target_start_frame": int(sample["target_start_frame"].item()),
                "target_start_ms": target_start_ms_values[index],
                "beatmap_id": _metadata_optional_int(sample, "beatmap_id"),
                "filtered_index": _metadata_optional_int(sample, "filtered_index"),
                "source_index": _metadata_optional_int(sample, "source_index"),
            }
            for index, sample in enumerate(samples)
        ],
    }


def build_control_window_index(
    *,
    source_index_path: str | Path = DEFAULT_MAP_INDEX_PATH,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    control_v3_summary_path: str | Path = DEFAULT_CONTROL_V3_SUMMARY_PATH,
    output_path: str | Path = DEFAULT_CONTROL_WINDOW_INDEX_PATH,
    report_path: str | Path | None = DEFAULT_CONTROL_WINDOW_INDEX_REPORT_PATH,
    frame_count_loader: AudioFrameCountLoader | None = None,
    progress_every: int = 0,
    command: str | None = None,
) -> ControlWindowIndexBuildReport:
    started_at = time.perf_counter()
    source_index_path = Path(source_index_path)
    dataset_root = Path(dataset_root)
    control_v3_summary_path = Path(control_v3_summary_path)
    output_path = Path(output_path)
    report_path = None if report_path is None else Path(report_path)
    frame_count_loader = _audio_frame_count_20ms if frame_count_loader is None else frame_count_loader

    source_df = pd.read_parquet(source_index_path)
    _require_index_columns(source_df, source_index_path)
    source_map_count = _map_count(source_df)
    source_df = filter_supported_difficulty_range(source_df)

    summary_df = pd.read_parquet(control_v3_summary_path)
    _require_control_v3_summary_columns(summary_df, control_v3_summary_path)
    summary_df = _validated_control_v3_summary(summary_df, control_v3_summary_path)
    index_df = _attach_control_v3_summary_indexes(source_df, summary_df)

    audio_frame_counts: dict[str, int] = {}
    window_rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(index_df.itertuples(index=False), start=1):
        shard_root = _resolve_shard_root(dataset_root, row.shard)
        audio_path = _resolve_index_path(dataset_root, shard_root, row.audio_path, field="audio_path")
        audio_key = audio_path.as_posix()
        frame_count = audio_frame_counts.get(audio_key)
        if frame_count is None:
            frame_count = _validate_positive_int(frame_count_loader(audio_path), "frame_count")
            audio_frame_counts[audio_key] = frame_count

        row_data = row._asdict()
        row_data["filtered_index"] = _validate_nonnegative_int(row_data["filtered_index"], "filtered_index")
        row_data["source_index"] = _validate_nonnegative_int(row_data["source_index"], "source_index")
        row_data["frame_count"] = frame_count
        for target_start_frame in iter_target_start_frames(frame_count):
            out_row = dict(row_data)
            out_row["target_start_frame"] = int(target_start_frame)
            out_row["target_start_ms"] = int(target_start_frame * FRAME_HOP_MS)
            window_rows.append(out_row)

        if progress_every > 0 and (row_number == 1 or row_number % progress_every == 0):
            print(
                f"control_window_index_progress maps={row_number}/{len(index_df)} "
                f"unique_audio={len(audio_frame_counts)} windows={len(window_rows)}",
                flush=True,
            )

    output_df = pd.DataFrame.from_records(window_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_output_path.exists():
        tmp_output_path.unlink()
    output_df.to_parquet(tmp_output_path, index=False, compression="zstd")
    tmp_output_path.replace(output_path)

    frame_counts = list(audio_frame_counts.values())
    report = ControlWindowIndexBuildReport(
        source_index_path=source_index_path,
        control_v3_summary_path=control_v3_summary_path,
        output_path=output_path,
        dataset_root=dataset_root,
        source_map_count=source_map_count,
        retained_map_count=_map_count(index_df),
        unique_audio_count=len(audio_frame_counts),
        window_count=len(output_df),
        min_frame_count=min(frame_counts) if frame_counts else None,
        max_frame_count=max(frame_counts) if frame_counts else None,
        target_window_length_frames=TARGET_WINDOW_LENGTH_FRAMES,
        target_window_stride_frames=TARGET_WINDOW_STRIDE_FRAMES,
        elapsed_s=time.perf_counter() - started_at,
    )
    if report_path is not None:
        _write_control_window_index_report(report, report_path, command=command)
    return report


def filter_supported_difficulty_range(index_df: pd.DataFrame) -> pd.DataFrame:
    difficulty = pd.to_numeric(index_df["difficulty"], errors="coerce")
    return index_df[(difficulty >= DIFFICULTY_MIN) & (difficulty <= DIFFICULTY_MAX)].reset_index(drop=True)


def iter_target_start_frames(frame_count: int) -> range:
    if frame_count < 0:
        raise ValueError(f"frame_count must be non-negative, got {frame_count!r}")
    return range(0, frame_count, TARGET_WINDOW_STRIDE_FRAMES)


def target_valid_mask(*, target_start_frame: int, frame_count: int) -> torch.Tensor:
    target_start_frame = _validate_nonnegative_int(target_start_frame, "target_start_frame")
    frame_count = _validate_positive_int(frame_count, "frame_count")
    frame_offsets = torch.arange(TARGET_WINDOW_LENGTH_FRAMES, dtype=torch.long)
    return target_start_frame + frame_offsets < frame_count


def normalize_difficulty(difficulty: float) -> float:
    difficulty = _validate_difficulty(difficulty)
    normalized = 2.0 * (difficulty - DIFFICULTY_MIN) / (DIFFICULTY_MAX - DIFFICULTY_MIN) - 1.0
    if not -1.0 <= normalized <= 1.0:
        raise ValueError(f"normalized difficulty outside [-1, 1]: {normalized!r}")
    return float(normalized)


def _audio_frame_count_20ms(audio_path: Path) -> int:
    from pulsefield_model.features.audio import load_audio_file
    from pulsefield_model.features.mel import DEFAULT_STAGE2_MEL_CONFIG

    config = DEFAULT_STAGE2_MEL_CONFIG
    waveform = load_audio_file(
        audio_path,
        sample_rate=config.sample_rate,
        speed=config.speed,
        normalize=config.normalize,
    )
    mel_10ms_frame_count = int(math.ceil(len(waveform) / config.mel_cache_config.hop_length))
    return (mel_10ms_frame_count + 1) // 2


def _attach_control_v3_summary_indexes(source_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    index_df = source_df.copy()
    if "filtered_index" in index_df.columns and "source_index" in index_df.columns:
        return index_df

    summary_columns = ["beatmap_path", "filtered_index", "source_index"]
    merged = index_df.merge(
        summary_df[summary_columns],
        on="beatmap_path",
        how="left",
        validate="one_to_one",
    )
    missing = merged["filtered_index"].isna() | merged["source_index"].isna()
    if bool(missing.any()):
        examples = merged.loc[missing, "beatmap_path"].astype(str).head(5).tolist()
        raise ValueError(f"control_v3 summary is missing {int(missing.sum())} source index row(s), examples={examples}")
    return merged


def _validated_control_v3_summary(summary_df: pd.DataFrame, summary_path: Path) -> pd.DataFrame:
    duplicate_paths = summary_df["beatmap_path"].duplicated(keep=False)
    if bool(duplicate_paths.any()):
        examples = summary_df.loc[duplicate_paths, "beatmap_path"].astype(str).head(5).tolist()
        raise ValueError(f"{summary_path} contains duplicate beatmap_path rows, examples={examples}")

    if bool(summary_df["error_type"].astype(str).ne("").any()):
        failed = summary_df[summary_df["error_type"].astype(str).ne("")]
        examples = failed["beatmap_path"].astype(str).head(5).tolist()
        raise ValueError(f"{summary_path} contains {len(failed)} control_v3 error row(s), examples={examples}")
    if "finite" in summary_df.columns and not bool(summary_df["finite"].fillna(False).all()):
        raise ValueError(f"{summary_path} contains non-finite control_v3 summary row(s)")
    if "ranges_ok" in summary_df.columns and not bool(summary_df["ranges_ok"].fillna(False).all()):
        raise ValueError(f"{summary_path} contains out-of-range control_v3 summary row(s)")
    return summary_df


def _require_control_v3_summary_columns(summary_df: pd.DataFrame, summary_path: Path) -> None:
    missing_columns = sorted(_CONTROL_V3_SUMMARY_REQUIRED_COLUMNS.difference(summary_df.columns))
    if missing_columns:
        raise ValueError(f"{summary_path} is missing required column(s): {missing_columns}")


def _write_control_window_index_report(
    report: ControlWindowIndexBuildReport,
    report_path: Path,
    *,
    command: str | None,
) -> None:
    payload = {
        **{
            key: (value.as_posix() if isinstance(value, Path) else value)
            for key, value in asdict(report).items()
        },
        "code_commit": _git_rev_parse("HEAD"),
        "command": command,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_report_path = report_path.with_suffix(report_path.suffix + ".tmp")
    tmp_report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_report_path.replace(report_path)


def _git_rev_parse(ref: str) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", ref], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _validate_difficulty(difficulty: object) -> float:
    value = float(difficulty)
    if not math.isfinite(value):
        raise ValueError(f"difficulty must be finite, got {difficulty!r}")
    if value < DIFFICULTY_MIN or value > DIFFICULTY_MAX:
        raise ValueError(f"difficulty outside supported 2.0..6.0 range: {value}")
    return value


def _require_index_columns(index_df: pd.DataFrame, index_path: Path) -> None:
    missing_columns = sorted(_REQUIRED_INDEX_COLUMNS.difference(index_df.columns))
    if missing_columns:
        raise ValueError(f"{index_path} is missing required column(s): {missing_columns}")


def _map_count(index_df: pd.DataFrame) -> int:
    if "beatmap_path" in index_df.columns:
        return int(index_df["beatmap_path"].astype(str).nunique())
    return len(index_df)


def _resolve_shard_root(dataset_root: Path, shard: object) -> Path:
    text = str(shard)
    if not text:
        raise ValueError("shard must be a non-empty relative path component")
    shard_path = Path(text)
    if shard_path.is_absolute():
        raise ValueError(f"shard must be relative to dataset_root, got absolute path: {text!r}")
    if len(shard_path.parts) != 1 or any(part in {"", ".", ".."} for part in shard_path.parts):
        raise ValueError(f"shard must be a single relative path component: {text!r}")

    root = dataset_root.resolve(strict=False)
    resolved = (dataset_root / shard_path).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"shard must stay under dataset_root {root}: {text!r}") from exc
    return dataset_root / shard_path


def _resolve_index_path(dataset_root: Path, shard_root: Path, value: object, *, field: str) -> Path:
    text = str(value)
    if not text:
        raise ValueError(f"{field} must be a non-empty relative path")

    relative_path = Path(text)
    if relative_path.is_absolute():
        raise ValueError(f"{field} must be relative to the shard root, got absolute path: {text!r}")
    if any(part == ".." for part in relative_path.parts):
        raise ValueError(f"{field} must not contain parent traversal: {text!r}")

    root = dataset_root.resolve(strict=False)
    resolved = (shard_root / relative_path).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} must stay under dataset_root {root}: {text!r}") from exc
    return shard_root / relative_path


def _optional_row_int(row: object, field: str) -> int | None:
    if not hasattr(row, field):
        return None
    value = getattr(row, field)
    if value is None or pd.isna(value):
        return None
    return _validate_nonnegative_int(value, field)


def _validate_positive_int(value: object, field: str) -> int:
    integer = _validate_nonnegative_int(value, field)
    if integer <= 0:
        raise ValueError(f"{field} must be positive, got {integer}")
    return integer


def _validate_nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{field} must be an integer, got bool")
    if isinstance(value, (int, np.integer)):
        integer = int(value)
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{field} must be an integer, got {value!r}")
        integer = int(value)
    else:
        raise TypeError(f"{field} must be an integer, got {type(value).__name__}")
    if integer < 0:
        raise ValueError(f"{field} must be non-negative, got {integer}")
    return integer


def _optional_int_tensor(value: int | None) -> torch.Tensor:
    if value is None:
        return torch.tensor(-1, dtype=torch.long)
    return torch.tensor(value, dtype=torch.long)


def _metadata_optional_int(sample: dict[str, Any], field: str) -> int | None:
    value = sample.get(field)
    if not isinstance(value, torch.Tensor):
        return None
    integer = int(value.item())
    if integer < 0:
        return None
    return integer


def _sample_positive_int(sample: Mapping[str, Any], field: str, *, source: object) -> int:
    integer = _sample_integer(sample, field, source=source)
    if integer <= 0:
        raise ValueError(f"{field} for {source} must be positive, got {integer}")
    return integer


def _sample_nonnegative_int(sample: Mapping[str, Any], field: str, *, source: object) -> int:
    integer = _sample_integer(sample, field, source=source)
    if integer < 0:
        raise ValueError(f"{field} for {source} must be non-negative, got {integer}")
    return integer


def _sample_integer(sample: Mapping[str, Any], field: str, *, source: object) -> int:
    value = sample[field]
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.numel() != 1:
            raise ValueError(f"{field} for {source} must be scalar, got shape {tuple(tensor.shape)}")
        if tensor.dtype == torch.bool:
            raise ValueError(f"{field} for {source} must be an integer, got bool")
        if tensor.dtype.is_complex:
            raise ValueError(f"{field} for {source} must be an integer, got {tensor.dtype}")
        scalar = tensor.item()
    elif isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.size != 1:
            raise ValueError(f"{field} for {source} must be scalar, got shape {array.shape}")
        if array.dtype == np.bool_:
            raise ValueError(f"{field} for {source} must be an integer, got bool")
        if np.issubdtype(array.dtype, np.complexfloating):
            raise ValueError(f"{field} for {source} must be an integer, got {array.dtype}")
        scalar = array.item()
    else:
        scalar = value
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(f"{field} for {source} must be an integer, got bool")
    if isinstance(scalar, (int, np.integer)):
        return int(scalar)
    if isinstance(scalar, (float, np.floating)):
        scalar_float = float(scalar)
        if not math.isfinite(scalar_float) or not scalar_float.is_integer():
            raise ValueError(f"{field} for {source} must be an integer, got {scalar!r}")
        return int(scalar_float)
    raise TypeError(f"{field} for {source} must be an integer, got {type(scalar).__name__}")


def _validate_cache_size(max_cached_maps: int) -> int:
    if not isinstance(max_cached_maps, int) or isinstance(max_cached_maps, bool):
        raise TypeError(f"max_cached_maps must be an integer, got {type(max_cached_maps).__name__}")
    if max_cached_maps < 0:
        raise ValueError(f"max_cached_maps must be non-negative, got {max_cached_maps}")
    return max_cached_maps


def _lru_get(cache: OrderedDict[Any, Any], key: Any) -> Any | None:
    try:
        value = cache.pop(key)
    except KeyError:
        return None
    cache[key] = value
    return value


def _lru_put(cache: OrderedDict[Any, Any], key: Any, value: Any, *, max_items: int) -> None:
    if max_items == 0:
        return
    cache[key] = value
    while len(cache) > max_items:
        cache.popitem(last=False)


def _validate_full_mel(value: Any, *, source: object, expected_frame_count: int | None = None) -> torch.Tensor:
    return _validate_feature_matrix(
        value,
        name="full_mel",
        source=source,
        expected_channels=PACKED_MEL_CHANNELS,
        expected_frame_count=expected_frame_count,
    )


def _validate_dense_timing_v2(value: Any, *, source: object, expected_frame_count: int) -> torch.Tensor:
    return _validate_feature_matrix(
        value,
        name="full_dense_timing_v2",
        source=source,
        expected_channels=DENSE_TIMING_V2_CHANNELS,
        expected_frame_count=expected_frame_count,
    )


def _validate_control_v3_target(value: Any, *, source: object) -> torch.Tensor:
    tensor = _as_float32_tensor(value, name="control_v3_target", source=source)
    expected_shape = (TARGET_WINDOW_LENGTH_FRAMES, CONTROL_V3_TARGET_CHANNELS)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"control_v3_target for {source} must have shape {expected_shape}, got {tuple(tensor.shape)}")
    confidence_indexes = [MODEL_FEATURE_NAMES.index(name) for name in CONFIDENCE_FEATURE_NAMES]
    confidence = tensor[:, confidence_indexes]
    if torch.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError(f"control_v3_target confidence channels for {source} must be in [0, 1]")
    return tensor.contiguous()


def _validate_target_valid_mask(value: Any, *, source: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.dtype != torch.bool:
            raise ValueError(f"target_valid_mask for {source} must be bool, got {tensor.dtype}")
    else:
        array = np.asarray(value)
        if array.dtype != np.bool_:
            raise ValueError(f"target_valid_mask for {source} must be bool, got {array.dtype}")
        tensor = torch.from_numpy(np.ascontiguousarray(array))

    expected_shape = (TARGET_WINDOW_LENGTH_FRAMES,)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"target_valid_mask for {source} must have shape {expected_shape}, got {tuple(tensor.shape)}")
    return tensor.contiguous()


def _validate_ln_change_n_eff_target(value: Any, *, source: object) -> torch.Tensor:
    tensor = _as_float32_tensor(value, name="ln_change_n_eff_target", source=source)
    expected_shape = (TARGET_WINDOW_LENGTH_FRAMES,)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"ln_change_n_eff_target for {source} must have shape {expected_shape}, got {tuple(tensor.shape)}")
    if torch.any(tensor < 0.0):
        raise ValueError(f"ln_change_n_eff_target for {source} must be non-negative")
    return tensor.contiguous()


def _validate_feature_matrix(
    value: Any,
    *,
    name: str,
    source: object,
    expected_channels: int,
    expected_frame_count: int | None,
) -> torch.Tensor:
    tensor = _as_float32_tensor(value, name=name, source=source)
    if tensor.ndim != 2:
        raise ValueError(f"{name} for {source} must be rank 2, got shape {tuple(tensor.shape)}")
    if int(tensor.shape[1]) != expected_channels:
        raise ValueError(
            f"{name} for {source} must have {expected_channels} channels, got shape {tuple(tensor.shape)}"
        )
    if expected_frame_count is not None and int(tensor.shape[0]) != expected_frame_count:
        raise ValueError(
            f"{name} for {source} must have frame_count={expected_frame_count}, got shape {tuple(tensor.shape)}"
        )
    return tensor.contiguous()


def _context_source_feature_matrix(
    value: Any,
    *,
    name: str,
    source: object,
    expected_channels: int,
    expected_frame_count: int,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} for {source} must be float32, got {tensor.dtype}")
    else:
        array = np.asarray(value)
        if array.dtype != np.float32:
            raise ValueError(f"{name} for {source} must be float32, got {array.dtype}")
        tensor = torch.as_tensor(array)
    if tensor.ndim != 2:
        raise ValueError(f"{name} for {source} must be rank 2, got shape {tuple(tensor.shape)}")
    expected_shape = (expected_frame_count, expected_channels)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"{name} for {source} must have shape {expected_shape}, got {tuple(tensor.shape)}")
    return tensor


def _as_float32_tensor(value: Any, *, name: str, source: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} for {source} must be float32, got {tensor.dtype}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} for {source} must contain only finite values")
        return tensor

    array = np.asarray(value)
    if array.dtype != np.float32:
        raise ValueError(f"{name} for {source} must be float32, got {array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} for {source} must contain only finite values")
    return torch.from_numpy(np.ascontiguousarray(array))


def _default_load_full_song_packed_mel(audio_path: Path) -> Any:
    from pulsefield_model.features.mel import load_full_song_packed_mel_20ms

    return load_full_song_packed_mel_20ms(audio_path)


def _default_load_oracle_dense_timing_v2(beatmap_path: Path, frame_count: int) -> Any:
    from pulsefield_model.timing.providers.oracle import render_oracle_dense_timing_v2

    return render_oracle_dense_timing_v2(beatmap_path, frame_count=frame_count)


def _full_support_ln_change_n_eff_target(_record: ControlWindowRecord) -> torch.Tensor:
    return torch.full((TARGET_WINDOW_LENGTH_FRAMES,), 3.0, dtype=torch.float32)


def _default_control_v3_rows(timeseries_path: str | Path, selector: tuple[str, int]) -> pd.DataFrame:
    from pulsefield_model.features.control_v3_targets import load_control_v3_timeseries_rows

    field, value = selector
    kwargs = {field: value}
    rows = load_control_v3_timeseries_rows(timeseries_path, include_ln_change_n_eff=True, **kwargs)
    if rows is None:
        raise FileNotFoundError(f"control_v3 timeseries rows not found for {field}={value}")
    return rows


def _target_selector(record: ControlWindowRecord) -> tuple[str, int]:
    if record.filtered_index is not None:
        return "filtered_index", record.filtered_index
    if record.source_index is not None:
        return "source_index", record.source_index
    if record.beatmap_id is not None:
        return "beatmap_id", record.beatmap_id
    raise ValueError(
        "default control_v3 target loading requires beatmap_id, filtered_index, or source_index in the index row"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Stage 2 control-window training index.")
    parser.add_argument("--source-index-path", type=Path, default=DEFAULT_MAP_INDEX_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--control-v3-summary-path",
        type=Path,
        default=DEFAULT_CONTROL_V3_SUMMARY_PATH,
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_CONTROL_WINDOW_INDEX_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_CONTROL_WINDOW_INDEX_REPORT_PATH)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args(argv)

    report = build_control_window_index(
        source_index_path=args.source_index_path,
        dataset_root=args.dataset_root,
        control_v3_summary_path=args.control_v3_summary_path,
        output_path=args.output_path,
        report_path=args.report_path,
        progress_every=args.progress_every,
        command=_format_command(argv),
    )
    print(
        "built control window index "
        f"maps={report.retained_map_count} "
        f"unique_audio={report.unique_audio_count} "
        f"windows={report.window_count} "
        f"output={report.output_path}",
    )
    return 0


def _format_command(argv: Sequence[str] | None) -> str:
    if argv is None:
        return " ".join(shlex.quote(arg) for arg in sys.argv)
    return " ".join([shlex.quote(sys.argv[0]), *(shlex.quote(arg) for arg in argv)])


if __name__ == "__main__":
    raise SystemExit(main())
