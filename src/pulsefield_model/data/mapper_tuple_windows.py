from __future__ import annotations

import hashlib
import json
import math
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import Dataset

from pulsefield_model.osu_core.hitobjects import parse_mania_hit_objects
from pulsefield_model.models.control.context import prepare_control_context_batch
from pulsefield_model.data.control_windows import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_INDEX_PATH,
    FRAME_HOP_MS,
    TARGET_WINDOW_LENGTH_FRAMES,
    ControlWindowDataset,
    ControlWindowRecord,
    normalize_difficulty,
)
from pulsefield_model.features.control_v3_targets import MODEL_FEATURE_NAMES, VALUE_FEATURE_NAMES
from pulsefield_model.models.mapper.shared.replay import CLOSED_OPEN_START_MS, ln_carry_state_tensors
from pulsefield_model.models.mapper.shared.tokenizer import (
    MAPPER_DENSITY_FRAMES,
    MAPPER_WRITE_MS,
    TokenizedMapperWindow,
    UnsupportedMapperActionError,
    encode_mapper_window,
    hitobjects_to_mapper_timepoints,
)
from pulsefield_model.models.mapper.shared.vocab import MapperTupleVocab


MAPPER_WRITE_FRAMES = MAPPER_WRITE_MS // FRAME_HOP_MS
MAPPER_CONTEXT_FRAMES = MAPPER_WRITE_FRAMES
DENSITY_LEVEL_TARGET_INDEX = MODEL_FEATURE_NAMES.index("density_level")
DENSITY_CONFIDENCE_TARGET_INDEX = MODEL_FEATURE_NAMES.index("density_confidence")
CONTROL_TEACHER_CACHE_SCHEMA_VERSION = 1
MAPPER_TUPLE_RECORD_CACHE_SCHEMA_VERSION = 1
MAPPER_TUPLE_TOKENIZER_CACHE_VERSION = 1
DEFAULT_MAX_CACHED_TIMEPOINT_MAPS = 256


@dataclass(frozen=True)
class MapperTupleWindowRecord:
    control_record_index: int
    control_record: ControlWindowRecord
    target_seq_len: int

    @property
    def write_start_frame(self) -> int:
        return self.control_record.target_start_frame

    @property
    def write_start_ms(self) -> int:
        return self.control_record.target_start_ms

    @property
    def write_end_ms(self) -> int:
        return self.write_start_ms + MAPPER_WRITE_MS


@dataclass(frozen=True)
class MapperTupleWindowFilterReport:
    num_total_windows: int
    num_mapper_eligible_windows: int
    num_dropped_short_windows: int
    num_dropped_cross_window_ln_windows: int
    num_dropped_unsupported_action_windows: int
    drop_rate: float
    short_drop_rate: float
    cross_window_ln_drop_rate: float
    unsupported_action_drop_rate: float
    drop_rate_by_difficulty: dict[str, float]
    drop_rate_by_song: dict[str, float]


class MapperTupleWindowDataset(Dataset):
    """Mapper tuple 8s window dataset derived from the existing Stage 2 control dataset."""

    def __init__(
        self,
        index_path: str | Path = DEFAULT_INDEX_PATH,
        *,
        dataset_root: str | Path = DEFAULT_DATASET_ROOT,
        control_dataset: ControlWindowDataset | None = None,
        vocab: MapperTupleVocab | None = None,
        mapper_stride_frames: int = MAPPER_WRITE_FRAMES,
        mapper_record_cache_path: str | Path | None = None,
        control_teacher_cache_dir: str | Path | None = None,
        require_control_teacher_cache: bool = False,
        include_full_song_context: bool = False,
        max_cached_timepoint_maps: int = DEFAULT_MAX_CACHED_TIMEPOINT_MAPS,
        progress: bool = False,
        **control_dataset_kwargs: Any,
    ) -> None:
        if mapper_stride_frames <= 0:
            raise ValueError(f"mapper_stride_frames must be positive: {mapper_stride_frames}")
        self.control_dataset = control_dataset or ControlWindowDataset(
            index_path=index_path,
            dataset_root=dataset_root,
            progress=progress,
            **control_dataset_kwargs,
        )
        self.vocab = MapperTupleVocab() if vocab is None else vocab
        self.mapper_stride_frames = int(mapper_stride_frames)
        self.mapper_record_cache_path = None if mapper_record_cache_path is None else Path(mapper_record_cache_path)
        self.control_teacher_cache_dir = None if control_teacher_cache_dir is None else Path(control_teacher_cache_dir)
        self.require_control_teacher_cache = bool(require_control_teacher_cache)
        self.include_full_song_context = bool(include_full_song_context)
        self.max_cached_timepoint_maps = _validate_timepoint_cache_size(max_cached_timepoint_maps)
        self._timepoints_by_beatmap: OrderedDict[str, tuple] = OrderedDict()
        cached_records = self._load_cached_records(progress=progress)
        if cached_records is None:
            self.records, self.filter_report = self._build_records(progress=progress)
            self._save_cached_records(progress=progress)
        else:
            self.records, self.filter_report = cached_records
        self.target_token_lengths = [record.target_seq_len for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def _load_cached_records(
        self,
        *,
        progress: bool,
    ) -> tuple[list[MapperTupleWindowRecord], MapperTupleWindowFilterReport] | None:
        cache_path = self.mapper_record_cache_path
        if cache_path is None:
            return None
        metadata_path = mapper_tuple_record_cache_metadata_path(cache_path)
        if not cache_path.exists() or not metadata_path.exists():
            return None

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(metadata, Mapping):
            return None
        if metadata.get("validity") != self._record_cache_validity_metadata():
            return None

        import pandas as pd

        frame = pd.read_parquet(cache_path)
        expected_columns = ["control_record_index", "target_seq_len"]
        if list(frame.columns) != expected_columns:
            raise ValueError(f"mapper tuple record cache must contain columns {expected_columns}: {cache_path}")
        filter_report = _mapper_tuple_filter_report_from_metadata(metadata.get("filter_report"))
        if len(frame) != filter_report.num_mapper_eligible_windows:
            raise ValueError(
                "mapper tuple record cache row count does not match filter_report "
                f"({len(frame)} != {filter_report.num_mapper_eligible_windows}): {cache_path}"
            )
        source_records = self.control_dataset.records
        records: list[MapperTupleWindowRecord] = []
        for row_number, row in enumerate(frame.itertuples(index=False), start=1):
            control_record_index = _cache_positive_index(row.control_record_index, len(source_records), row_number)
            target_seq_len = _cache_positive_int(row.target_seq_len, "target_seq_len", row_number)
            records.append(
                MapperTupleWindowRecord(
                    control_record_index=control_record_index,
                    control_record=source_records[control_record_index],
                    target_seq_len=target_seq_len,
                )
            )
        if progress:
            print(
                "mapper_tuple_window_dataset_cache status=hit "
                f"path={cache_path.as_posix()} records={len(records)}",
                flush=True,
            )
        return records, filter_report

    def _save_cached_records(self, *, progress: bool) -> None:
        cache_path = self.mapper_record_cache_path
        if cache_path is None:
            return

        import pandas as pd

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = mapper_tuple_record_cache_metadata_path(cache_path)
        tmp_cache_path = cache_path.with_name(f"{cache_path.name}.tmp")
        tmp_metadata_path = metadata_path.with_name(f"{metadata_path.name}.tmp")
        frame = pd.DataFrame(
            {
                "control_record_index": [record.control_record_index for record in self.records],
                "target_seq_len": [record.target_seq_len for record in self.records],
            }
        )
        metadata = {
            "validity": self._record_cache_validity_metadata(),
            "filter_report": asdict(self.filter_report),
        }
        frame.to_parquet(tmp_cache_path, index=False)
        tmp_metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp_cache_path.replace(cache_path)
        tmp_metadata_path.replace(metadata_path)
        if progress:
            print(
                "mapper_tuple_window_dataset_cache status=written "
                f"path={cache_path.as_posix()} records={len(self.records)}",
                flush=True,
            )

    def _record_cache_validity_metadata(self) -> dict[str, Any]:
        index_path = getattr(self.control_dataset, "index_path", None)
        dataset_root = getattr(self.control_dataset, "dataset_root", None)
        # The record cache assumes beatmap .osu contents are immutable for a given index.
        # If mapper data is regenerated from edited .osu files, rebuild the source index
        # or delete this cache so tokenization eligibility and lengths are recomputed.
        metadata: dict[str, Any] = {
            "schema_version": MAPPER_TUPLE_RECORD_CACHE_SCHEMA_VERSION,
            "tokenizer_cache_version": MAPPER_TUPLE_TOKENIZER_CACHE_VERSION,
            "mapper_stride_frames": int(self.mapper_stride_frames),
            "mapper_write_ms": int(MAPPER_WRITE_MS),
            "frame_hop_ms": int(FRAME_HOP_MS),
            "source_window_count": len(self.control_dataset.records),
            "control_records_sha1": mapper_tuple_control_records_sha1(self.control_dataset.records),
        }
        if index_path is not None:
            path = Path(index_path)
            metadata["source_index_path"] = path.as_posix()
            if path.exists():
                metadata["source_index_sha1"] = _file_sha1(path)
        if dataset_root is not None:
            metadata["dataset_root"] = Path(dataset_root).as_posix()
        return metadata

    def __getitem__(self, index: int) -> dict[str, Any]:
        mapper_record = self.records[index]
        record = mapper_record.control_record
        tokenized = self._tokenize_record(record)
        cache_path = self.control_teacher_cache_path(record)
        cache_entry = None
        if cache_path is not None and cache_path.exists():
            cache_entry = load_control_teacher_cache_entry(cache_path, record=record)
        elif self.require_control_teacher_cache and cache_path is not None:
            raise FileNotFoundError(f"missing mapper tuple control teacher cache entry: {cache_path}")
        elif self.require_control_teacher_cache:
            raise ValueError("require_control_teacher_cache=True requires control_teacher_cache_dir")

        metadata = {
            "beatmap_path": record.beatmap_path.as_posix(),
            "audio_path": record.audio_path.as_posix(),
            "difficulty": record.difficulty,
            "source_frame_count": record.frame_count,
            "inference_frame_count": mapper_tuple_padded_frame_count(record),
            "target_start_frame": record.target_start_frame,
            "target_start_ms": record.target_start_ms,
            "control_record_index": mapper_record.control_record_index,
        }
        if cache_path is not None:
            metadata["control_teacher_cache_key"] = control_teacher_cache_key(record)
            metadata["control_teacher_cache_path"] = cache_path.as_posix()
            metadata["control_teacher_cache_hit"] = cache_entry is not None

        sample: dict[str, Any] = {
            "difficulty": torch.tensor([record.difficulty], dtype=torch.float32),
            "normalized_difficulty": torch.tensor([normalize_difficulty(record.difficulty)], dtype=torch.float32),
            "decoder_input_tokens": tokenized.decoder_input_tensor(),
            "target_fragment_tokens": tokenized.target_fragment_tensor(),
            "target_fragment_states": {
                "current_ms": tokenized.target_fragment_current_ms,
                "open_mask": tokenized.target_fragment_open_mask,
                "open_start_ms": tokenized.target_fragment_open_start_ms,
                "open_age_ms": tokenized.target_fragment_open_age_ms,
            },
            "ln_carry_in": ln_carry_state_tensors(tokenized.ln_carry_in),
            "ln_carry_out": ln_carry_state_tensors(tokenized.ln_carry_out),
            "close_labels": tokenized.close_labels,
            "close_label_mask": tokenized.close_label_mask,
            "write_start_ms": torch.tensor(tokenized.write_start_ms, dtype=torch.long),
            "write_end_ms": torch.tensor(tokenized.write_end_ms, dtype=torch.long),
            "is_full_chart_start": torch.tensor(tokenized.is_full_chart_start, dtype=torch.bool),
            "is_full_chart_end": torch.tensor(tokenized.is_full_chart_end, dtype=torch.bool),
            "metadata": metadata,
        }
        density_target_8s, density_confidence_8s = extract_mapper_density_8s(
            self._load_control_v3_target_8s(record),
        )
        if cache_entry is not None:
            sample["control_memory_8s"] = cache_entry["control_memory_8s"]
            sample["density_teacher_8s"] = cache_entry["density_teacher_8s"]
            sample["density_target_8s"] = density_target_8s
            sample["density_confidence_8s"] = density_confidence_8s
            if self.include_full_song_context:
                sample.update(self._load_full_song_context_fields(mapper_record, record))
            return sample

        sample.update(self._load_full_song_context_fields(mapper_record, record))
        sample["density_target_8s"] = density_target_8s
        sample["density_confidence_8s"] = density_confidence_8s
        return sample

    def _load_full_song_context_fields(
        self,
        mapper_record: MapperTupleWindowRecord,
        record: ControlWindowRecord,
    ) -> dict[str, Any]:
        base_sample = self.control_dataset[mapper_record.control_record_index]
        source_frame_count = int(base_sample["frame_count"].item())
        write_start_frame = int(base_sample["target_start_frame"].item())
        write_end_frame = write_start_frame + MAPPER_WRITE_FRAMES
        inference_frame_count = max(source_frame_count, write_end_frame)
        full_mel = pad_mapper_tuple_feature_frames(
            base_sample["full_mel"],
            inference_frame_count=inference_frame_count,
            expected_source_frame_count=source_frame_count,
            expected_channels=160,
            name="full_mel",
        )
        full_dense_timing_v2 = pad_mapper_tuple_feature_frames(
            base_sample["full_dense_timing_v2"],
            inference_frame_count=inference_frame_count,
            expected_source_frame_count=source_frame_count,
            expected_channels=4,
            name="full_dense_timing_v2",
        )
        context_frame_indexes = torch.arange(MAPPER_CONTEXT_FRAMES, dtype=torch.long) + write_start_frame
        context_padding_mask = context_frame_indexes >= source_frame_count
        return {
            "full_mel": full_mel,
            "full_dense_timing_v2": full_dense_timing_v2,
            "frame_count": torch.tensor(inference_frame_count, dtype=torch.long),
            "source_frame_count": torch.tensor(source_frame_count, dtype=torch.long),
            "target_start_frame": torch.as_tensor(base_sample["target_start_frame"], dtype=torch.long),
            "control_slice_start_frames": torch.tensor(
                [
                    record.target_start_frame + offset
                    for offset in range(0, MAPPER_WRITE_FRAMES, TARGET_WINDOW_LENGTH_FRAMES)
                ],
                dtype=torch.long,
            ),
            "mel_context": full_mel[write_start_frame:write_end_frame].contiguous(),
            "timing_context": full_dense_timing_v2[write_start_frame:write_end_frame].contiguous(),
            "context_padding_mask": context_padding_mask,
            "difficulty": base_sample["difficulty"].reshape(1),
            "normalized_difficulty": base_sample["normalized_difficulty"].reshape(1),
        }

    def control_teacher_cache_path(self, record: ControlWindowRecord) -> Path | None:
        if self.control_teacher_cache_dir is None:
            return None
        return control_teacher_cache_path(self.control_teacher_cache_dir, record)

    def _build_records(
        self,
        *,
        progress: bool,
    ) -> tuple[list[MapperTupleWindowRecord], MapperTupleWindowFilterReport]:
        records: list[MapperTupleWindowRecord] = []
        total_windows = 0
        dropped_short = 0
        dropped_cross_window = 0
        dropped_unsupported_action = 0
        valid_length_windows = 0
        dropped_by_difficulty: dict[str, int] = {}
        valid_by_difficulty: dict[str, int] = {}
        valid_by_song: dict[str, int] = {}
        dropped_by_song: dict[str, int] = {}
        started_at = time.monotonic()
        source_window_count = len(self.control_dataset.records)
        if progress:
            print(
                "mapper_tuple_window_dataset_progress status=start "
                f"source_windows={source_window_count} mapper_stride_frames={self.mapper_stride_frames}",
                flush=True,
            )

        def maybe_print_progress(source_index: int) -> None:
            if not progress or (total_windows != 1 and total_windows % 1000 != 0):
                return
            elapsed_s = time.monotonic() - started_at
            print(
                "mapper_tuple_window_dataset_progress "
                f"scanned_source_windows={source_index + 1}/{source_window_count} "
                f"candidate_windows={total_windows} eligible_windows={len(records)} "
                f"dropped_short={dropped_short} dropped_cross_window_ln={dropped_cross_window} "
                f"dropped_unsupported_action={dropped_unsupported_action} "
                f"parsed_maps={len(self._timepoints_by_beatmap)} elapsed_s={elapsed_s:.1f}",
                flush=True,
            )

        for index, record in enumerate(self.control_dataset.records):
            if not is_mapper_tuple_window_start_allowed(record, mapper_stride_frames=self.mapper_stride_frames):
                continue
            total_windows += 1
            difficulty_key = _difficulty_report_key(record.difficulty)
            song_key = record.beatmap_path.as_posix()
            valid_length_windows += 1
            valid_by_difficulty[difficulty_key] = valid_by_difficulty.get(difficulty_key, 0) + 1
            valid_by_song[song_key] = valid_by_song.get(song_key, 0) + 1
            try:
                tokenized = self._tokenize_record(record)
            except UnsupportedMapperActionError:
                dropped_unsupported_action += 1
                _increment_drop(dropped_by_difficulty, difficulty_key)
                _increment_drop(dropped_by_song, song_key)
                maybe_print_progress(index)
                continue
            records.append(
                MapperTupleWindowRecord(
                    control_record_index=index,
                    control_record=record,
                    target_seq_len=tokenized.seq_len,
                )
            )
            maybe_print_progress(index)

        dropped = dropped_short + dropped_cross_window + dropped_unsupported_action
        if progress:
            elapsed_s = time.monotonic() - started_at
            print(
                "mapper_tuple_window_dataset_progress status=done "
                f"candidate_windows={total_windows} eligible_windows={len(records)} "
                f"dropped={dropped} parsed_maps={len(self._timepoints_by_beatmap)} "
                f"elapsed_s={elapsed_s:.1f}",
                flush=True,
            )
        report = MapperTupleWindowFilterReport(
            num_total_windows=total_windows,
            num_mapper_eligible_windows=len(records),
            num_dropped_short_windows=dropped_short,
            num_dropped_cross_window_ln_windows=dropped_cross_window,
            num_dropped_unsupported_action_windows=dropped_unsupported_action,
            drop_rate=float(dropped / total_windows) if total_windows else 0.0,
            short_drop_rate=float(dropped_short / total_windows) if total_windows else 0.0,
            cross_window_ln_drop_rate=float(dropped_cross_window / valid_length_windows) if valid_length_windows else 0.0,
            unsupported_action_drop_rate=float(dropped_unsupported_action / valid_length_windows) if valid_length_windows else 0.0,
            drop_rate_by_difficulty=_drop_rates(valid_by_difficulty, dropped_by_difficulty),
            drop_rate_by_song=_drop_rates(valid_by_song, dropped_by_song),
        )
        return records, report

    def _tokenize_record(self, record: ControlWindowRecord) -> TokenizedMapperWindow:
        write_start_ms = record.target_start_ms
        write_end_ms = write_start_ms + MAPPER_WRITE_MS
        chart_end_ms = max(int(record.frame_count) * FRAME_HOP_MS, write_end_ms)
        timepoints = self._load_timepoints(record.beatmap_path)
        return encode_mapper_window(
            timepoints,
            vocab=self.vocab,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            chart_start_ms=0,
            chart_end_ms=chart_end_ms,
        )

    def _load_timepoints(self, beatmap_path: Path) -> tuple:
        key = beatmap_path.as_posix()
        cached = self._timepoints_by_beatmap.get(key)
        if cached is not None:
            self._timepoints_by_beatmap.move_to_end(key)
            return cached
        cached = tuple(hitobjects_to_mapper_timepoints(parse_mania_hit_objects(beatmap_path, expected_key_count=4)))
        if self.max_cached_timepoint_maps > 0:
            self._timepoints_by_beatmap[key] = cached
            while len(self._timepoints_by_beatmap) > self.max_cached_timepoint_maps:
                self._timepoints_by_beatmap.popitem(last=False)
        return cached

    def _load_control_v3_target_8s(self, record: ControlWindowRecord) -> torch.Tensor:
        slices = []
        for offset_frames in range(0, MAPPER_WRITE_FRAMES, TARGET_WINDOW_LENGTH_FRAMES):
            slice_record = replace(record, target_start_frame=record.target_start_frame + offset_frames)
            target = self.control_dataset.target_loader(slice_record)
            slice_tensor = torch.as_tensor(target, dtype=torch.float32)
            expected_shape = (TARGET_WINDOW_LENGTH_FRAMES, len(MODEL_FEATURE_NAMES))
            if tuple(slice_tensor.shape) != expected_shape:
                raise ValueError(f"control_v3 target slice must have shape {expected_shape}, got {tuple(slice_tensor.shape)}")
            if not torch.isfinite(slice_tensor).all():
                raise ValueError("control_v3 target slice must contain only finite values")
            slices.append(slice_tensor)
        return torch.cat(slices, dim=0).contiguous()


def mapper_tuple_record_cache_metadata_path(cache_path: str | Path) -> Path:
    return Path(cache_path).with_suffix(".json")


def mapper_tuple_control_records_sha1(records: Sequence[ControlWindowRecord]) -> str:
    digest = hashlib.sha1()
    for index, record in enumerate(records):
        if not isinstance(record, ControlWindowRecord):
            raise TypeError(f"control dataset record {index} must be a ControlWindowRecord")
        digest.update(
            "\t".join(
                (
                    str(index),
                    record.beatmap_path.as_posix(),
                    record.audio_path.as_posix(),
                    f"{float(record.difficulty):.8f}",
                    str(int(record.frame_count)),
                    str(int(record.target_start_frame)),
                    "" if record.beatmap_id is None else str(int(record.beatmap_id)),
                    "" if record.filtered_index is None else str(int(record.filtered_index)),
                    "" if record.source_index is None else str(int(record.source_index)),
                )
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_timepoint_cache_size(max_cached_timepoint_maps: int) -> int:
    if not isinstance(max_cached_timepoint_maps, int) or isinstance(max_cached_timepoint_maps, bool):
        raise TypeError(
            "max_cached_timepoint_maps must be an integer, "
            f"got {type(max_cached_timepoint_maps).__name__}"
        )
    if max_cached_timepoint_maps < 0:
        raise ValueError(f"max_cached_timepoint_maps must be non-negative, got {max_cached_timepoint_maps}")
    return max_cached_timepoint_maps


def _mapper_tuple_filter_report_from_metadata(payload: object) -> MapperTupleWindowFilterReport:
    if not isinstance(payload, Mapping):
        raise ValueError("mapper tuple record cache metadata missing filter_report")
    return MapperTupleWindowFilterReport(
        num_total_windows=_metadata_int(payload, "num_total_windows"),
        num_mapper_eligible_windows=_metadata_int(payload, "num_mapper_eligible_windows"),
        num_dropped_short_windows=_metadata_int(payload, "num_dropped_short_windows"),
        num_dropped_cross_window_ln_windows=_metadata_int(payload, "num_dropped_cross_window_ln_windows"),
        num_dropped_unsupported_action_windows=_metadata_int(payload, "num_dropped_unsupported_action_windows"),
        drop_rate=_metadata_float(payload, "drop_rate"),
        short_drop_rate=_metadata_float(payload, "short_drop_rate"),
        cross_window_ln_drop_rate=_metadata_float(payload, "cross_window_ln_drop_rate"),
        unsupported_action_drop_rate=_metadata_float(payload, "unsupported_action_drop_rate"),
        drop_rate_by_difficulty=_metadata_float_mapping(payload, "drop_rate_by_difficulty"),
        drop_rate_by_song=_metadata_float_mapping(payload, "drop_rate_by_song"),
    )


def _metadata_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool):
        raise ValueError(f"mapper tuple record cache filter_report.{key} must be an integer")
    try:
        integer = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"mapper tuple record cache filter_report.{key} must be an integer") from exc
    if integer < 0:
        raise ValueError(f"mapper tuple record cache filter_report.{key} must be non-negative")
    return integer


def _metadata_float(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool):
        raise ValueError(f"mapper tuple record cache filter_report.{key} must be a number")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"mapper tuple record cache filter_report.{key} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"mapper tuple record cache filter_report.{key} must be finite")
    return number


def _metadata_float_mapping(payload: Mapping[str, object], key: str) -> dict[str, float]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"mapper tuple record cache filter_report.{key} must be a mapping")
    return {str(item_key): _metadata_float(value, str(item_key)) for item_key in value}


def _cache_positive_index(value: object, record_count: int, row_number: int) -> int:
    index = _cache_nonnegative_int(value, "control_record_index", row_number)
    if index >= record_count:
        raise ValueError(
            f"mapper tuple record cache row {row_number} control_record_index is out of range: "
            f"{index} >= {record_count}"
        )
    return index


def _cache_positive_int(value: object, name: str, row_number: int) -> int:
    integer = _cache_nonnegative_int(value, name, row_number)
    if integer <= 0:
        raise ValueError(f"mapper tuple record cache row {row_number} {name} must be positive")
    return integer


def _cache_nonnegative_int(value: object, name: str, row_number: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"mapper tuple record cache row {row_number} {name} must be an integer")
    try:
        integer = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"mapper tuple record cache row {row_number} {name} must be an integer") from exc
    if integer < 0:
        raise ValueError(f"mapper tuple record cache row {row_number} {name} must be non-negative")
    return integer


def extract_mapper_density_8s(control_v3_target_8s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.as_tensor(control_v3_target_8s, dtype=torch.float32)
    expected_shape = (MAPPER_DENSITY_FRAMES, len(MODEL_FEATURE_NAMES))
    if tuple(target.shape) != expected_shape:
        raise ValueError(f"control_v3_target_8s must have shape {expected_shape}, got {tuple(target.shape)}")
    if not torch.isfinite(target).all():
        raise ValueError("control_v3_target_8s must contain only finite values")
    density_target = target[:, DENSITY_LEVEL_TARGET_INDEX : DENSITY_LEVEL_TARGET_INDEX + 1].contiguous()
    density_confidence = target[:, DENSITY_CONFIDENCE_TARGET_INDEX : DENSITY_CONFIDENCE_TARGET_INDEX + 1].contiguous()
    if not torch.isfinite(density_target).all() or not torch.isfinite(density_confidence).all():
        raise ValueError("density_target_8s and density_confidence_8s must contain only finite values")
    if torch.any((density_confidence < 0.0) | (density_confidence > 1.0)):
        raise ValueError("density_confidence_8s must be in [0, 1]")
    return density_target, density_confidence


def is_mapper_tuple_window_start_allowed(record: ControlWindowRecord, *, mapper_stride_frames: int = MAPPER_WRITE_FRAMES) -> bool:
    if mapper_stride_frames <= 0:
        raise ValueError(f"mapper_stride_frames must be positive: {mapper_stride_frames}")
    target_start_frame = int(record.target_start_frame)
    if target_start_frame % int(mapper_stride_frames) == 0:
        return True
    terminal_start_frame = max(0, int(record.frame_count) - MAPPER_WRITE_FRAMES)
    return target_start_frame == terminal_start_frame


def mapper_tuple_padded_frame_count(record: ControlWindowRecord) -> int:
    return max(int(record.frame_count), int(record.target_start_frame) + MAPPER_WRITE_FRAMES)


def validate_mapper_tuple_feature_frames(
    value: Any,
    *,
    expected_frame_count: int,
    expected_channels: int,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    expected_shape = (int(expected_frame_count), int(expected_channels))
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values")
    return tensor.contiguous()


def pad_mapper_tuple_feature_frames(
    value: Any,
    *,
    inference_frame_count: int,
    expected_source_frame_count: int,
    expected_channels: int,
    name: str,
) -> torch.Tensor:
    tensor = validate_mapper_tuple_feature_frames(
        value,
        expected_frame_count=int(expected_source_frame_count),
        expected_channels=int(expected_channels),
        name=name,
    )
    inference_frame_count = int(inference_frame_count)
    if inference_frame_count < int(expected_source_frame_count):
        raise ValueError(
            f"inference_frame_count must cover source frames: "
            f"{inference_frame_count} < {expected_source_frame_count}"
        )
    if inference_frame_count == int(expected_source_frame_count):
        return tensor.contiguous()
    padded = tensor.new_zeros((inference_frame_count, int(expected_channels)))
    padded[: int(expected_source_frame_count)] = tensor
    return padded.contiguous()


def control_teacher_cache_key(record: ControlWindowRecord) -> str:
    identity = "\n".join(
        (
            record.beatmap_path.as_posix(),
            record.audio_path.as_posix(),
            f"difficulty={float(record.difficulty):.8f}",
            f"frame_count={int(record.frame_count)}",
            f"target_start_frame={int(record.target_start_frame)}",
        )
    )
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()


def control_teacher_cache_path(cache_dir: str | Path, record: ControlWindowRecord) -> Path:
    key = control_teacher_cache_key(record)
    return Path(cache_dir) / key[:2] / f"{key}.pt"


def load_control_teacher_cache_entry(path: str | Path, *, record: ControlWindowRecord | None = None) -> dict[str, torch.Tensor]:
    cache_path = Path(path)
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"control teacher cache entry must contain a mapping: {cache_path}")
    schema_version = payload.get("schema_version")
    if int(schema_version) != CONTROL_TEACHER_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported control teacher cache schema {schema_version}; "
            f"expected {CONTROL_TEACHER_CACHE_SCHEMA_VERSION}"
        )
    if record is not None:
        expected_key = control_teacher_cache_key(record)
        if payload.get("cache_key") != expected_key:
            raise ValueError(f"control teacher cache key mismatch for {cache_path}")
        if int(payload.get("write_start_ms", -1)) != int(record.target_start_ms):
            raise ValueError(f"control teacher cache write_start_ms mismatch for {cache_path}")
        if int(payload.get("write_end_ms", -1)) != int(record.target_start_ms + MAPPER_WRITE_MS):
            raise ValueError(f"control teacher cache write_end_ms mismatch for {cache_path}")

    control_memory = payload.get("control_memory_8s")
    density_teacher = payload.get("density_teacher_8s")
    if not isinstance(control_memory, torch.Tensor):
        raise ValueError(f"control teacher cache missing control_memory_8s tensor: {cache_path}")
    if not isinstance(density_teacher, torch.Tensor):
        raise ValueError(f"control teacher cache missing density_teacher_8s tensor: {cache_path}")
    control_memory = control_memory.to(dtype=torch.float32).contiguous()
    density_teacher = density_teacher.to(dtype=torch.float32).contiguous()
    if control_memory.ndim != 2 or int(control_memory.shape[0]) != MAPPER_DENSITY_FRAMES:
        raise ValueError(
            f"control_memory_8s cache tensor must have shape [{MAPPER_DENSITY_FRAMES},D], "
            f"got {tuple(control_memory.shape)}"
        )
    if int(control_memory.shape[1]) <= 0:
        raise ValueError("control_memory_8s cache tensor must have a positive hidden dimension")
    if tuple(density_teacher.shape) != (MAPPER_DENSITY_FRAMES, 1):
        raise ValueError(
            f"density_teacher_8s cache tensor must have shape [{MAPPER_DENSITY_FRAMES},1], "
            f"got {tuple(density_teacher.shape)}"
        )
    if not torch.isfinite(control_memory).all() or not torch.isfinite(density_teacher).all():
        raise ValueError(f"control teacher cache contains non-finite values: {cache_path}")
    return {
        "control_memory_8s": control_memory,
        "density_teacher_8s": density_teacher,
    }


def save_control_teacher_cache_entry(
    path: str | Path,
    *,
    record: ControlWindowRecord,
    control_memory_8s: torch.Tensor,
    density_teacher_8s: torch.Tensor,
) -> None:
    cache_path = Path(path)
    control_memory = control_memory_8s.detach().to(device="cpu", dtype=torch.float32).clone(
        memory_format=torch.contiguous_format,
    )
    density_teacher = density_teacher_8s.detach().to(device="cpu", dtype=torch.float32).clone(
        memory_format=torch.contiguous_format,
    )
    if control_memory.ndim != 2 or int(control_memory.shape[0]) != MAPPER_DENSITY_FRAMES:
        raise ValueError(
            f"control_memory_8s must have shape [{MAPPER_DENSITY_FRAMES},D], got {tuple(control_memory.shape)}"
        )
    if tuple(density_teacher.shape) != (MAPPER_DENSITY_FRAMES, 1):
        raise ValueError(
            f"density_teacher_8s must have shape [{MAPPER_DENSITY_FRAMES},1], got {tuple(density_teacher.shape)}"
        )
    if not torch.isfinite(control_memory).all() or not torch.isfinite(density_teacher).all():
        raise ValueError("control teacher cache tensors must contain only finite values")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CONTROL_TEACHER_CACHE_SCHEMA_VERSION,
        "cache_key": control_teacher_cache_key(record),
        "write_start_ms": int(record.target_start_ms),
        "write_end_ms": int(record.target_start_ms + MAPPER_WRITE_MS),
        "control_dim": int(control_memory.shape[1]),
        "control_memory_8s": control_memory,
        "density_teacher_8s": density_teacher,
    }
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(cache_path)


def control_teacher_slice_batch(mapper_batch: dict[str, Any], slice_index: int) -> dict[str, Any]:
    if not 0 <= int(slice_index) < 4:
        raise ValueError(f"slice_index must be in 0..3, got {slice_index}")
    control_slice_start_frames = mapper_batch.get("control_slice_start_frames")
    if not isinstance(control_slice_start_frames, torch.Tensor) or control_slice_start_frames.ndim != 2:
        raise ValueError("mapper_batch must contain control_slice_start_frames with shape [B,4]")
    if int(control_slice_start_frames.shape[1]) != 4:
        raise ValueError("control_slice_start_frames must have four aligned 2s starts")
    normalized_difficulty = mapper_batch.get("normalized_difficulty")
    if not isinstance(normalized_difficulty, torch.Tensor):
        raise ValueError("mapper_batch must contain normalized_difficulty")
    normalized_difficulty = normalized_difficulty.reshape(normalized_difficulty.shape[0])
    control_batch = {
        "full_mel": mapper_batch["full_mel"],
        "full_dense_timing_v2": mapper_batch["full_dense_timing_v2"],
        "padding_mask": mapper_batch["padding_mask"],
        "frame_count": mapper_batch["frame_count"],
        "target_start_frame": control_slice_start_frames[:, int(slice_index)],
        "normalized_difficulty": normalized_difficulty,
    }
    return prepare_control_context_batch(control_batch)


def control_teacher_stacked_slices_batch(mapper_batch: dict[str, Any]) -> dict[str, Any]:
    control_slice_start_frames = mapper_batch.get("control_slice_start_frames")
    if not isinstance(control_slice_start_frames, torch.Tensor) or control_slice_start_frames.ndim != 2:
        raise ValueError("mapper_batch must contain control_slice_start_frames with shape [B,4]")
    if int(control_slice_start_frames.shape[1]) != 4:
        raise ValueError("control_slice_start_frames must have four aligned 2s starts")
    normalized_difficulty = mapper_batch.get("normalized_difficulty")
    if not isinstance(normalized_difficulty, torch.Tensor):
        raise ValueError("mapper_batch must contain normalized_difficulty")
    batch_size = int(control_slice_start_frames.shape[0])
    normalized_difficulty = normalized_difficulty.reshape(batch_size).repeat_interleave(4)
    control_batch = {
        "full_mel": mapper_batch["full_mel"].repeat_interleave(4, dim=0),
        "full_dense_timing_v2": mapper_batch["full_dense_timing_v2"].repeat_interleave(4, dim=0),
        "padding_mask": mapper_batch["padding_mask"].repeat_interleave(4, dim=0),
        "frame_count": mapper_batch["frame_count"].reshape(batch_size).repeat_interleave(4),
        "target_start_frame": control_slice_start_frames.reshape(batch_size * 4),
        "normalized_difficulty": normalized_difficulty,
    }
    return prepare_control_context_batch(control_batch)


def concatenate_density_teacher_8s(control_outputs: Sequence[Any]) -> torch.Tensor:
    if len(control_outputs) != 4:
        raise ValueError(f"expected four 2s control outputs, got {len(control_outputs)}")
    values = []
    density_index = VALUE_FEATURE_NAMES.index("density_level")
    for index, output in enumerate(control_outputs):
        value_pred = getattr(output, "value_pred", None)
        if not isinstance(value_pred, torch.Tensor) or value_pred.ndim != 3 or int(value_pred.shape[1]) != 100:
            raise ValueError(f"control output {index} value_pred must have shape [B,100,C]")
        if int(value_pred.shape[2]) == 1:
            values.append(value_pred)
        elif int(value_pred.shape[2]) == len(VALUE_FEATURE_NAMES):
            values.append(value_pred[:, :, density_index : density_index + 1])
        else:
            raise ValueError(
                f"control output {index} value_pred channel count must be 1 or {len(VALUE_FEATURE_NAMES)}, "
                f"got {value_pred.shape[2]}",
            )
    return torch.cat(values, dim=1).contiguous()


def collate_mapper_tuple_windows(samples: Sequence[dict[str, Any]], *, pad_id: int = 0) -> dict[str, Any]:
    if not samples:
        raise ValueError("collate_mapper_tuple_windows requires at least one sample")
    batch_size = len(samples)
    max_seq_len = max(int(sample["target_fragment_tokens"].shape[0]) for sample in samples)
    decoder_input_tokens = torch.full((batch_size, max_seq_len), int(pad_id), dtype=torch.long)
    target_fragment_tokens = torch.full((batch_size, max_seq_len), int(pad_id), dtype=torch.long)
    target_fragment_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)
    target_fragment_current_ms = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    target_fragment_open_mask = torch.zeros((batch_size, max_seq_len, 4), dtype=torch.bool)
    target_fragment_open_start_ms = torch.full(
        (batch_size, max_seq_len, 4),
        int(CLOSED_OPEN_START_MS),
        dtype=torch.long,
    )
    target_fragment_open_age_ms = torch.zeros((batch_size, max_seq_len, 4), dtype=torch.long)
    close_labels = torch.zeros((batch_size, max_seq_len, 4), dtype=torch.bool)
    close_label_mask = torch.zeros((batch_size, max_seq_len, 4), dtype=torch.bool)

    for batch_index, sample in enumerate(samples):
        length = int(sample["target_fragment_tokens"].shape[0])
        fragment_states = sample["target_fragment_states"]
        decoder_input_tokens[batch_index, :length] = sample["decoder_input_tokens"].to(dtype=torch.long)
        target_fragment_tokens[batch_index, :length] = sample["target_fragment_tokens"].to(dtype=torch.long)
        target_fragment_mask[batch_index, :length] = True
        target_fragment_current_ms[batch_index, :length] = fragment_states["current_ms"].to(dtype=torch.long)
        target_fragment_open_mask[batch_index, :length] = fragment_states["open_mask"].to(dtype=torch.bool)
        target_fragment_open_start_ms[batch_index, :length] = fragment_states["open_start_ms"].to(dtype=torch.long)
        target_fragment_open_age_ms[batch_index, :length] = fragment_states["open_age_ms"].to(dtype=torch.long)
        close_labels[batch_index, :length] = sample["close_labels"].to(dtype=torch.bool)
        close_label_mask[batch_index, :length] = sample["close_label_mask"].to(dtype=torch.bool)

    batch = {
        "difficulty": torch.stack([sample["difficulty"].to(dtype=torch.float32) for sample in samples]),
        "normalized_difficulty": torch.stack(
            [sample.get("normalized_difficulty", sample["difficulty"]).to(dtype=torch.float32) for sample in samples],
        ),
        "decoder_input_tokens": decoder_input_tokens,
        "target_fragment_tokens": target_fragment_tokens,
        "target_fragment_mask": target_fragment_mask,
        "target_fragment_states": {
            "current_ms": target_fragment_current_ms,
            "open_mask": target_fragment_open_mask,
            "open_start_ms": target_fragment_open_start_ms,
            "open_age_ms": target_fragment_open_age_ms,
        },
        "ln_carry_in": _stack_carry_batch(samples, "ln_carry_in"),
        "ln_carry_out": _stack_carry_batch(samples, "ln_carry_out"),
        "close_labels": close_labels,
        "close_label_mask": close_label_mask,
        "write_start_ms": torch.stack([sample["write_start_ms"].to(dtype=torch.long) for sample in samples]).reshape(
            batch_size,
        ),
        "write_end_ms": torch.stack([sample["write_end_ms"].to(dtype=torch.long) for sample in samples]).reshape(
            batch_size,
        ),
        "is_full_chart_start": torch.stack(
            [sample["is_full_chart_start"].to(dtype=torch.bool) for sample in samples],
        ).reshape(batch_size),
        "is_full_chart_end": torch.stack(
            [sample["is_full_chart_end"].to(dtype=torch.bool) for sample in samples],
        ).reshape(batch_size),
        "metadata": [sample.get("metadata", {}) for sample in samples],
    }
    if all("chart_end_ms" in sample for sample in samples):
        batch["chart_end_ms"] = torch.stack(
            [sample["chart_end_ms"].to(dtype=torch.long) for sample in samples],
        ).reshape(batch_size)

    has_control_teacher_cache = [
        "control_memory_8s" in sample or "density_teacher_8s" in sample
        for sample in samples
    ]
    if any(has_control_teacher_cache):
        if not all("control_memory_8s" in sample and "density_teacher_8s" in sample for sample in samples):
            raise ValueError(
                "partial mapper tuple control teacher cache batch is not supported; "
                "precompute all entries or disable the cache"
            )
        batch["control_memory_8s"] = torch.stack(
            [sample["control_memory_8s"].to(dtype=torch.float32) for sample in samples]
        )
        batch["density_teacher_8s"] = torch.stack(
            [sample["density_teacher_8s"].to(dtype=torch.float32) for sample in samples]
        )

    if all("density_target_8s" in sample and "density_confidence_8s" in sample for sample in samples):
        batch["density_target_8s"] = torch.stack(
            [sample["density_target_8s"].to(dtype=torch.float32) for sample in samples]
        )
        batch["density_confidence_8s"] = torch.stack(
            [sample["density_confidence_8s"].to(dtype=torch.float32) for sample in samples],
        )

    has_control_inputs = all(
        "full_mel" in sample and "full_dense_timing_v2" in sample and "frame_count" in sample
        for sample in samples
    )
    if has_control_inputs:
        if all("mel_context" in sample and "timing_context" in sample and "context_padding_mask" in sample for sample in samples):
            batch["mel_context"] = torch.stack([sample["mel_context"].to(dtype=torch.float32) for sample in samples])
            batch["timing_context"] = torch.stack([sample["timing_context"].to(dtype=torch.float32) for sample in samples])
            batch["context_padding_mask"] = torch.stack(
                [sample["context_padding_mask"].to(dtype=torch.bool) for sample in samples]
            )
        frame_counts = [int(sample["frame_count"].item()) for sample in samples]
        source_frame_counts = [int(sample["source_frame_count"].item()) for sample in samples]
        max_frame_count = max(frame_counts)
        full_mel = torch.zeros((batch_size, max_frame_count, 160), dtype=torch.float32)
        full_dense_timing_v2 = torch.zeros((batch_size, max_frame_count, 4), dtype=torch.float32)
        padding_mask = torch.ones((batch_size, max_frame_count), dtype=torch.bool)
        for batch_index, sample in enumerate(samples):
            frame_count = frame_counts[batch_index]
            source_frame_count = source_frame_counts[batch_index]
            if source_frame_count > frame_count:
                raise ValueError(
                    f"source_frame_count sample {batch_index} cannot exceed frame_count: "
                    f"{source_frame_count} > {frame_count}"
                )
            sample_full_mel = sample["full_mel"].to(dtype=torch.float32)
            sample_full_dense_timing_v2 = sample["full_dense_timing_v2"].to(dtype=torch.float32)
            if tuple(sample_full_mel.shape) != (frame_count, 160):
                raise ValueError(f"full_mel sample {batch_index} must have shape {(frame_count, 160)}")
            if tuple(sample_full_dense_timing_v2.shape) != (frame_count, 4):
                raise ValueError(f"full_dense_timing_v2 sample {batch_index} must have shape {(frame_count, 4)}")
            full_mel[batch_index, :frame_count] = sample_full_mel
            full_dense_timing_v2[batch_index, :frame_count] = sample_full_dense_timing_v2
            padding_mask[batch_index, :source_frame_count] = False
        batch["full_mel"] = full_mel
        batch["full_dense_timing_v2"] = full_dense_timing_v2
        batch["padding_mask"] = padding_mask
        batch["frame_count"] = torch.tensor(frame_counts, dtype=torch.long)
        batch["source_frame_count"] = torch.tensor(source_frame_counts, dtype=torch.long)
        if all("target_start_frame" in sample for sample in samples):
            batch["target_start_frame"] = torch.stack(
                [torch.as_tensor(sample["target_start_frame"], dtype=torch.long) for sample in samples],
            ).reshape(batch_size)
        if all("control_slice_start_frames" in sample for sample in samples):
            batch["control_slice_start_frames"] = torch.stack(
                [sample["control_slice_start_frames"].to(dtype=torch.long) for sample in samples],
            )
    return batch


def _stack_carry_batch(samples: Sequence[dict[str, Any]], key: str) -> dict[str, torch.Tensor]:
    return {
        "current_ms": torch.stack([sample[key]["current_ms"].to(dtype=torch.long) for sample in samples]).reshape(len(samples)),
        "open_mask": torch.stack([sample[key]["open_mask"].to(dtype=torch.bool) for sample in samples]),
        "open_start_ms": torch.stack([sample[key]["open_start_ms"].to(dtype=torch.long) for sample in samples]),
        "open_age_ms": torch.stack([sample[key]["open_age_ms"].to(dtype=torch.long) for sample in samples]),
    }


def _difficulty_report_key(difficulty: float) -> str:
    return f"{float(difficulty):.2f}"


def _increment_drop(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _drop_rates(total: dict[str, int], dropped: dict[str, int]) -> dict[str, float]:
    return {
        key: float(dropped.get(key, 0) / count) if count else 0.0
        for key, count in sorted(total.items())
    }
