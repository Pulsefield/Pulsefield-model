from __future__ import annotations

import argparse
import gc
import math
import pickle
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from pulsefield_model.data.control_windows import (
    ControlWindowDataset,
    ControlWindowRecord,
    DEFAULT_MAX_CACHED_MAPS,
    DENSE_TIMING_V2_CHANNELS,
    PACKED_MEL_CHANNELS,
    TARGET_OFFSET_IN_CONTEXT,
    TARGET_WINDOW_LENGTH_FRAMES,
    normalize_difficulty,
)
from pulsefield_model.data.mapper_tuple_windows import (
    MAPPER_WRITE_FRAMES,
    MapperTupleWindowDataset,
    collate_mapper_tuple_windows,
    control_teacher_cache_path,
    is_mapper_tuple_window_start_allowed,
    mapper_tuple_padded_frame_count,
    pad_mapper_tuple_feature_frames,
    save_control_teacher_cache_entry,
)
from pulsefield_model.models.control import ControlDemoGlobalEncoder, ControlDemoGlobalEncoderConfig
from pulsefield_model.models.mapper.shared import MapperTupleConfig, TupleMapperBase, MapperTupleModelOutput, MapperTupleVocab
from pulsefield_model.models.mapper.shared.loss import adapter_bias_regularization, density_auxiliary_loss, token_cross_entropy
from pulsefield_model.training.common import (
    CHECKPOINT_SCHEMA_VERSION,
    DEFAULT_FINAL_TRAIN_EVAL_SIZE,
    ControlTrainingResult,
    _atomic_torch_save,
    _advance_training_iterator,
    _capture_rng_state,
    _copy_file_atomically,
    _infinite_loader,
    _json_metrics,
    _json_safe,
    _metric_is_count,
    _move_batch_tensors,
    _move_optimizer_state_to_device,
    _restore_rng_state,
    _safe_float_div,
    _set_deterministic_seed,
    _validate_training_args,
    _write_report,
    limit_final_train_eval_dataset,
    select_torch_device,
    split_train_eval_dataset,
)
from pulsefield_model.training.control_demo_global import initialize_global_control_demo_from_control_checkpoint


DEFAULT_RUNS_ROOT = Path("artifacts/runs/stage2_mapper_tuple")
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_ROOT / "phase_b_teacher_forced"
RUN_CONFIG_KEYS = {
    "dataset_root",
    "index_path",
    "eval_index_path",
    "control_v3_timeseries_path",
    "output_dir",
    "max_steps",
    "eval_every",
    "save_every",
    "log_every",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "seed",
    "device",
    "run_name",
    "init_from_control_checkpoint",
    "init_from_mapper_checkpoint",
    "resume_from",
    "eval_fraction",
    "eval_size",
    "final_train_eval_size",
    "num_workers",
    "max_cached_maps",
    "dataset_progress",
    "mapper_record_cache_path",
    "length_bucketed_batches",
    "length_bucket_size_multiplier",
    "control_teacher_cache_dir",
    "precompute_control_teacher_cache",
    "precompute_control_teacher_cache_only",
    "control_teacher_precompute_batch_size",
    "require_control_teacher_cache",
    "control_teacher_cache_overwrite",
    "mps_cleanup_every",
    "model",
    "control_model",
    "loss",
}
MAPPER_RESUME_TRAINING_RUNTIME_KEYS = frozenset(("mps_cleanup_every",))
MAPPER_RESUME_DATASET_RUNTIME_KEYS = frozenset(("max_cached_maps", "num_workers", "dataset_progress"))
MODEL_CONFIG_KEYS = {field.name for field in fields(MapperTupleConfig)}
CONTROL_MODEL_CONFIG_KEYS = {field.name for field in fields(ControlDemoGlobalEncoderConfig)}
LOSS_CONFIG_KEYS = {
    "lambda_ln_close",
    "lambda_adapter_reg",
    "lambda_density",
    "lambda_density_teacher",
    "close_pos_weight_max",
    "density_calibration_scale",
    "density_calibration_bias",
}
MAPPER_BATCH_TENSOR_KEYS = frozenset(
    (
        "decoder_input_tokens",
        "target_fragment_tokens",
        "target_fragment_mask",
        "target_fragment_states",
        "ln_carry_in",
        "ln_carry_out",
        "close_labels",
        "close_label_mask",
        "density_target_8s",
        "density_confidence_8s",
        "density_teacher_8s",
        "control_memory_8s",
        "control_memory_padding_mask_8s",
        "write_start_ms",
        "write_end_ms",
        "chart_end_ms",
        "is_full_chart_start",
        "is_full_chart_end",
        "difficulty",
        "normalized_difficulty",
        "full_mel",
        "full_dense_timing_v2",
        "padding_mask",
        "frame_count",
        "source_frame_count",
        "target_start_frame",
        "control_slice_start_frames",
    )
)


@dataclass(frozen=True)
class MapperTuplePhaseBLossConfig:
    lambda_ln_close: float = 0.05
    lambda_adapter_reg: float = 1e-5
    lambda_density: float = 0.0
    lambda_density_teacher: float = 0.0
    close_pos_weight_max: float = 20.0
    density_calibration_scale: float = 1.0
    density_calibration_bias: float = 0.0

    def __post_init__(self) -> None:
        if self.lambda_density < 0.0:
            raise ValueError("lambda_density must be non-negative")
        if self.lambda_density_teacher < 0.0:
            raise ValueError("lambda_density_teacher must be non-negative")
        if self.lambda_density_teacher != 0.0:
            raise ValueError("density teacher loss is not implemented for mapper tuple training")
        if self.lambda_ln_close < 0.0:
            raise ValueError("lambda_ln_close must be non-negative")
        if self.lambda_adapter_reg < 0.0:
            raise ValueError("lambda_adapter_reg must be non-negative")
        if self.close_pos_weight_max < 1.0:
            raise ValueError("close_pos_weight_max must be at least 1")
        if self.density_calibration_scale < 0.0:
            raise ValueError("density_calibration_scale must be non-negative")
        for name, value in (
            ("density_calibration_scale", self.density_calibration_scale),
            ("density_calibration_bias", self.density_calibration_bias),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class MapperTupleLossOutput:
    total_loss: torch.Tensor
    metrics: dict[str, float]
    metric_numerators: dict[str, float]
    metric_denominators: dict[str, float]


@dataclass(frozen=True)
class MapperTupleControlTeacherCachePrecomputeResult:
    cache_dir: Path
    total_entries: int
    computed_entries: int
    skipped_entries: int
    elapsed_s: float

    def to_report(self) -> dict[str, Any]:
        return {
            "cache_dir": self.cache_dir.as_posix(),
            "total_entries": self.total_entries,
            "computed_entries": self.computed_entries,
            "skipped_entries": self.skipped_entries,
            "elapsed_s": self.elapsed_s,
        }


@dataclass(frozen=True)
class MapperTupleControlTeacherCachePrecomputeRunResult:
    reports: list[dict[str, Any]]
    source_control_dataset: ControlWindowDataset
    eval_control_dataset: ControlWindowDataset | None


def load_run_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML run config: {path}") from exc
    if loaded is None:
        return {"model": {}, "control_model": {}, "loss": {}}
    if not isinstance(loaded, dict):
        raise ValueError(f"run config must be a mapping: {path}")

    config = _normalize_config_mapping(loaded, source_name="run config")
    unknown = sorted(set(config) - RUN_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown run config keys: {unknown}")
    config["model"] = _normalized_section(config.get("model", {}), allowed=MODEL_CONFIG_KEYS, name="model config")
    config["control_model"] = _normalized_section(
        config.get("control_model", {}),
        allowed=CONTROL_MODEL_CONFIG_KEYS,
        name="control model config",
    )
    config["loss"] = _normalized_section(config.get("loss", {}), allowed=LOSS_CONFIG_KEYS, name="loss config")
    return config


def run_mapper_tuple_phase_b_training(
    *,
    dataset_root: Path = Path("dataset"),
    index_path: Path | None = None,
    eval_index_path: Path | None = None,
    control_v3_timeseries_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_steps: int = 5000,
    eval_every: int = 100,
    save_every: int | None = None,
    log_every: int | None = None,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    seed: int = 1337,
    device_name: str = "auto",
    run_name: str = "mapper_tuple_phase_b_teacher_forced",
    init_from_control_checkpoint: Path | None = None,
    init_from_mapper_checkpoint: Path | None = None,
    resume_from: Path | None = None,
    eval_fraction: float = 0.1,
    eval_size: int | None = None,
    final_train_eval_size: int | None = DEFAULT_FINAL_TRAIN_EVAL_SIZE,
    num_workers: int = 0,
    max_cached_maps: int | None = None,
    dataset_progress: bool | None = None,
    mapper_record_cache_path: Path | None = None,
    length_bucketed_batches: bool = False,
    length_bucket_size_multiplier: int = 32,
    control_teacher_cache_dir: Path | None = None,
    precompute_control_teacher_cache: bool = False,
    control_teacher_precompute_batch_size: int | None = None,
    require_control_teacher_cache: bool = False,
    control_teacher_cache_overwrite: bool = False,
    mps_cleanup_every: int | None = None,
    model_config_overrides: Mapping[str, Any] | None = None,
    control_model_config_overrides: Mapping[str, Any] | None = None,
    loss_config_overrides: Mapping[str, Any] | None = None,
) -> ControlTrainingResult:
    _set_deterministic_seed(seed)
    control_model_config = ControlDemoGlobalEncoderConfig(**dict(control_model_config_overrides or {}))
    model_values = dict(model_config_overrides or {})
    model_values.setdefault("control_dim", control_model_config.d_model)
    model_config = MapperTupleConfig(**model_values)
    if model_config.control_dim != control_model_config.d_model:
        raise ValueError("mapper control_dim must match frozen control_model d_model")
    loss_config = MapperTuplePhaseBLossConfig(**dict(loss_config_overrides or {}))

    dataset_kwargs: dict[str, Any] = {"dataset_root": dataset_root}
    if index_path is not None:
        dataset_kwargs["index_path"] = index_path
    if control_v3_timeseries_path is not None:
        dataset_kwargs["control_v3_timeseries_path"] = control_v3_timeseries_path
    effective_max_cached_maps = DEFAULT_MAX_CACHED_MAPS if max_cached_maps is None else max_cached_maps
    dataset_kwargs["max_cached_maps"] = effective_max_cached_maps
    effective_dataset_progress = bool(precompute_control_teacher_cache) if dataset_progress is None else bool(dataset_progress)
    dataset_kwargs["progress"] = effective_dataset_progress

    cache_precompute_reports: list[dict[str, Any]] = []
    source_control_dataset: ControlWindowDataset | None = None
    eval_control_dataset: ControlWindowDataset | None = None
    if precompute_control_teacher_cache:
        precompute_run = precompute_mapper_tuple_phase_b_control_teacher_cache(
            dataset_root=dataset_root,
            index_path=index_path,
            eval_index_path=eval_index_path,
            control_v3_timeseries_path=control_v3_timeseries_path,
            batch_size=batch_size,
            seed=seed,
            device_name=device_name,
            init_from_control_checkpoint=init_from_control_checkpoint,
            num_workers=num_workers,
            max_cached_maps=max_cached_maps,
            dataset_progress=effective_dataset_progress,
            control_teacher_cache_dir=control_teacher_cache_dir,
            control_teacher_precompute_batch_size=control_teacher_precompute_batch_size,
            control_teacher_cache_overwrite=control_teacher_cache_overwrite,
            control_model_config=control_model_config,
        )
        cache_precompute_reports = precompute_run.reports
        source_control_dataset = precompute_run.source_control_dataset
        eval_control_dataset = precompute_run.eval_control_dataset

    mapper_dataset_kwargs: dict[str, Any]
    if source_control_dataset is None:
        mapper_dataset_kwargs = dict(dataset_kwargs)
    else:
        mapper_dataset_kwargs = {"control_dataset": source_control_dataset, "progress": effective_dataset_progress}
    if control_teacher_cache_dir is not None:
        mapper_dataset_kwargs["control_teacher_cache_dir"] = control_teacher_cache_dir
        mapper_dataset_kwargs["require_control_teacher_cache"] = bool(require_control_teacher_cache)
    if mapper_record_cache_path is not None:
        mapper_dataset_kwargs["mapper_record_cache_path"] = mapper_record_cache_path
    train_source = MapperTupleWindowDataset(**mapper_dataset_kwargs)
    if len(train_source) == 0:
        raise ValueError("MapperTupleWindowDataset produced no training windows")

    if eval_index_path is not None:
        if eval_control_dataset is None:
            eval_kwargs = dict(dataset_kwargs)
            eval_kwargs["index_path"] = eval_index_path
        else:
            eval_kwargs = {"control_dataset": eval_control_dataset, "progress": effective_dataset_progress}
        if control_teacher_cache_dir is not None:
            eval_kwargs["control_teacher_cache_dir"] = control_teacher_cache_dir
            eval_kwargs["require_control_teacher_cache"] = bool(require_control_teacher_cache)
        eval_dataset: Dataset[Any] = MapperTupleWindowDataset(**eval_kwargs)
        train_dataset: Dataset[Any] = train_source
    else:
        train_dataset, eval_dataset = split_train_eval_dataset(
            train_source,
            eval_fraction=eval_fraction,
            eval_size=eval_size,
            seed=seed,
        )
    if len(train_dataset) == 0:
        raise ValueError("training split is empty")
    if len(eval_dataset) == 0:
        eval_dataset = train_dataset

    loader = _make_mapper_tuple_phase_b_train_loader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        length_bucketed_batches=length_bucketed_batches,
        length_bucket_size_multiplier=length_bucket_size_multiplier,
    )
    train_eval_dataset = limit_final_train_eval_dataset(
        train_dataset,
        final_train_eval_size=final_train_eval_size,
        seed=seed,
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_mapper_tuple_windows,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_mapper_tuple_windows,
    )
    return _run_training(
        loader=loader,
        train_eval_loader=train_eval_loader,
        eval_loader=eval_loader,
        output_dir=output_dir,
        model_config=model_config,
        control_model_config=control_model_config,
        loss_config=loss_config,
        max_steps=max_steps,
        eval_every=eval_every,
        save_every=save_every,
        log_every=log_every,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        device_name=device_name,
        run_name=run_name,
        dataset_report={
            "train_window_count": len(train_dataset),
            "eval_window_count": len(eval_dataset),
            "source_window_count": len(train_source),
            "eval_index_path": eval_index_path.as_posix() if eval_index_path is not None else None,
            "eval_fraction": eval_fraction,
            "eval_size": eval_size,
            "final_train_eval_size": final_train_eval_size,
            "final_train_eval_window_count": len(train_eval_dataset),
            "filter_report": asdict(train_source.filter_report),
            "max_cached_maps": int(getattr(train_source.control_dataset, "max_cached_maps", effective_max_cached_maps)),
            "dataset_progress": bool(effective_dataset_progress),
            "mapper_record_cache_path": (
                mapper_record_cache_path.as_posix() if mapper_record_cache_path is not None else None
            ),
            "num_workers": num_workers,
            "length_bucketed_batches": bool(length_bucketed_batches),
            "length_bucket_size_multiplier": int(length_bucket_size_multiplier),
            "control_teacher_cache_dir": (
                control_teacher_cache_dir.as_posix() if control_teacher_cache_dir is not None else None
            ),
            "precompute_control_teacher_cache": bool(precompute_control_teacher_cache),
            "control_teacher_precompute_batch_size": control_teacher_precompute_batch_size,
            "require_control_teacher_cache": bool(require_control_teacher_cache),
            "control_teacher_cache_overwrite": bool(control_teacher_cache_overwrite),
            "control_teacher_cache_precompute": cache_precompute_reports,
        },
        init_from_control_checkpoint=init_from_control_checkpoint,
        init_from_mapper_checkpoint=init_from_mapper_checkpoint,
        resume_from=resume_from,
        mps_cleanup_every=mps_cleanup_every,
    )


def _make_mapper_tuple_phase_b_train_loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    length_bucketed_batches: bool,
    length_bucket_size_multiplier: int,
) -> DataLoader:
    if not length_bucketed_batches:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=num_workers,
            collate_fn=collate_mapper_tuple_windows,
        )

    return DataLoader(
        dataset,
        batch_sampler=_MapperTupleTokenLengthBucketBatchSampler(
            dataset,
            batch_size=batch_size,
            bucket_size_multiplier=length_bucket_size_multiplier,
            shuffle=True,
            seed=seed,
        ),
        num_workers=num_workers,
        collate_fn=collate_mapper_tuple_windows,
    )


class _MapperTupleTokenLengthBucketBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: Dataset[Any],
        *,
        batch_size: int,
        bucket_size_multiplier: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if bucket_size_multiplier <= 0:
            raise ValueError(f"length_bucket_size_multiplier must be positive, got {bucket_size_multiplier}")
        self.target_token_lengths = [
            _mapper_tuple_target_token_length(dataset, index)
            for index in range(len(dataset))
        ]
        self.batch_size = int(batch_size)
        self.bucket_size = int(batch_size) * int(bucket_size_multiplier)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = list(range(len(self.target_token_lengths)))
        if self.shuffle:
            indices = torch.randperm(len(indices), generator=generator).tolist()
        batches = list(self._batches_for(indices))
        if self.shuffle and len(batches) > 1:
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[index] for index in order]
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        return math.ceil(len(self.target_token_lengths) / self.batch_size)

    def _batches_for(self, indices: Sequence[int]):
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=lambda index: (self.target_token_lengths[index], index))
            for batch_start in range(0, len(bucket), self.batch_size):
                yield bucket[batch_start : batch_start + self.batch_size]


def _mapper_tuple_target_token_length(dataset: Dataset[Any], index: int) -> int:
    if isinstance(dataset, Subset):
        return _mapper_tuple_target_token_length(dataset.dataset, int(dataset.indices[index]))

    target_token_lengths = getattr(dataset, "target_token_lengths", None)
    if target_token_lengths is not None:
        return _positive_int(target_token_lengths[index], "target_token_length")

    records = getattr(dataset, "records", None)
    tokenizer = getattr(dataset, "_tokenize_record", None)
    if records is None or not callable(tokenizer):
        raise ValueError(
            "length-aware mapper tuple batching requires a MapperTupleWindowDataset "
            "or a dataset exposing target_token_lengths"
        )
    mapper_record = records[index]
    target_seq_len = getattr(mapper_record, "target_seq_len", None)
    if target_seq_len is not None:
        return _positive_int(target_seq_len, "target_token_length")
    control_record = getattr(mapper_record, "control_record", mapper_record)
    tokenized = tokenizer(control_record)
    seq_len = getattr(tokenized, "seq_len", None)
    if seq_len is None:
        target_fragment_tensor = getattr(tokenized, "target_fragment_tensor", None)
        if not callable(target_fragment_tensor):
            raise ValueError("tokenized mapper window does not expose seq_len or target_fragment_tensor")
        seq_len = int(target_fragment_tensor().shape[0])
    return _positive_int(seq_len, "target_token_length")


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} tensor must contain exactly one value")
        value = int(value.item())
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


class _ControlTeacherPrecomputeDataset(Dataset[Any]):
    def __init__(
        self,
        control_dataset: Dataset[Any],
        indexed_records: Sequence[tuple[int, ControlWindowRecord]],
    ) -> None:
        self.control_dataset = control_dataset
        self.indexed_records = [
            (int(control_record_index), record)
            for control_record_index, record in indexed_records
        ]

    def __len__(self) -> int:
        return len(self.indexed_records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        control_record_index, record = self.indexed_records[index]
        control_dataset = self.control_dataset
        inference_frame_count = mapper_tuple_padded_frame_count(record)
        full_mel_loader = getattr(control_dataset, "_load_full_mel", None)
        dense_timing_loader = getattr(control_dataset, "_load_dense_timing_v2", None)
        if callable(full_mel_loader) and callable(dense_timing_loader):
            full_mel = full_mel_loader(record.audio_path, expected_frame_count=record.frame_count)
            full_dense_timing_v2 = dense_timing_loader(record.beatmap_path, frame_count=record.frame_count)
        else:
            base_sample = control_dataset[control_record_index]
            full_mel = base_sample["full_mel"]
            full_dense_timing_v2 = base_sample["full_dense_timing_v2"]
        return {
            "full_mel": pad_mapper_tuple_feature_frames(
                full_mel,
                inference_frame_count=inference_frame_count,
                expected_source_frame_count=record.frame_count,
                expected_channels=PACKED_MEL_CHANNELS,
                name="full_mel",
            ),
            "full_dense_timing_v2": pad_mapper_tuple_feature_frames(
                full_dense_timing_v2,
                inference_frame_count=inference_frame_count,
                expected_source_frame_count=record.frame_count,
                expected_channels=DENSE_TIMING_V2_CHANNELS,
                name="full_dense_timing_v2",
            ),
            "frame_count": torch.tensor(inference_frame_count, dtype=torch.long),
            "source_frame_count": torch.tensor(record.frame_count, dtype=torch.long),
            "control_slice_start_frames": torch.tensor(
                [
                    record.target_start_frame + offset
                    for offset in range(0, MAPPER_WRITE_FRAMES, TARGET_WINDOW_LENGTH_FRAMES)
                ],
                dtype=torch.long,
            ),
            "difficulty": torch.tensor(record.difficulty, dtype=torch.float32),
            "normalized_difficulty": torch.tensor(normalize_difficulty(record.difficulty), dtype=torch.float32),
            "metadata": {
                "beatmap_path": record.beatmap_path.as_posix(),
                "audio_path": record.audio_path.as_posix(),
                "difficulty": record.difficulty,
                "source_frame_count": record.frame_count,
                "inference_frame_count": inference_frame_count,
                "target_start_frame": record.target_start_frame,
                "target_start_ms": record.target_start_ms,
                "control_record_index": control_record_index,
            },
        }


def _collate_mapper_tuple_control_teacher_precompute(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("_collate_mapper_tuple_control_teacher_precompute requires at least one sample")
    batch_size = len(samples)
    frame_counts = [int(sample["frame_count"].item()) for sample in samples]
    source_frame_counts = [int(sample["source_frame_count"].item()) for sample in samples]
    max_frame_count = max(frame_counts)
    full_mel = torch.zeros((batch_size, max_frame_count, PACKED_MEL_CHANNELS), dtype=torch.float32)
    full_dense_timing_v2 = torch.zeros(
        (batch_size, max_frame_count, DENSE_TIMING_V2_CHANNELS),
        dtype=torch.float32,
    )
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
        if tuple(sample_full_mel.shape) != (frame_count, PACKED_MEL_CHANNELS):
            raise ValueError(
                f"full_mel sample {batch_index} must have shape {(frame_count, PACKED_MEL_CHANNELS)}"
            )
        if tuple(sample_full_dense_timing_v2.shape) != (frame_count, DENSE_TIMING_V2_CHANNELS):
            raise ValueError(
                "full_dense_timing_v2 sample "
                f"{batch_index} must have shape {(frame_count, DENSE_TIMING_V2_CHANNELS)}"
            )
        full_mel[batch_index, :frame_count] = sample_full_mel
        full_dense_timing_v2[batch_index, :frame_count] = sample_full_dense_timing_v2
        padding_mask[batch_index, :source_frame_count] = False
    return {
        "full_mel": full_mel,
        "full_dense_timing_v2": full_dense_timing_v2,
        "padding_mask": padding_mask,
        "frame_count": torch.tensor(frame_counts, dtype=torch.long),
        "source_frame_count": torch.tensor(source_frame_counts, dtype=torch.long),
        "control_slice_start_frames": torch.stack(
            [sample["control_slice_start_frames"].to(dtype=torch.long) for sample in samples],
        ),
        "difficulty": torch.stack([sample["difficulty"].to(dtype=torch.float32) for sample in samples]).reshape(
            batch_size,
        ),
        "normalized_difficulty": torch.stack(
            [sample["normalized_difficulty"].to(dtype=torch.float32) for sample in samples],
        ).reshape(batch_size),
        "metadata": [sample.get("metadata", {}) for sample in samples],
    }


def _release_torch_device_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        return
    if device.type != "mps" or not hasattr(torch, "mps"):
        return
    synchronize = getattr(torch.mps, "synchronize", None)
    if synchronize is not None:
        synchronize()
    empty_cache = getattr(torch.mps, "empty_cache", None)
    if empty_cache is not None:
        empty_cache()


def _cleanup_mps_training_memory(device: torch.device) -> None:
    if device.type != "mps" or not hasattr(torch, "mps"):
        return
    gc.collect()
    _release_torch_device_cache(device)


def _compute_control_teacher_8s_for_precompute(
    control_encoder: ControlDemoGlobalEncoder,
    batch: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    from pulsefield_model.data.mapper_tuple_windows import control_teacher_slice_batch
    from pulsefield_model.features.control_v3_targets import VALUE_FEATURE_NAMES

    control_slice_start_frames = batch.get("control_slice_start_frames")
    if not isinstance(control_slice_start_frames, torch.Tensor) or control_slice_start_frames.ndim != 2:
        raise ValueError("control teacher precompute requires control_slice_start_frames with shape [B,4]")
    if int(control_slice_start_frames.shape[1]) != 4:
        raise ValueError("control_slice_start_frames must have four aligned 2s starts")
    batch_size = int(control_slice_start_frames.shape[0])

    control_encoder.to(device)
    control_encoder.eval()
    control_memory_slices: list[torch.Tensor] = []
    density_teacher_slices: list[torch.Tensor] = []
    density_index = VALUE_FEATURE_NAMES.index("density_level")
    for slice_index in range(4):
        control_batch = control_teacher_slice_batch(dict(batch), slice_index)
        with torch.no_grad():
            output = control_encoder(
                context_mel=control_batch["context_mel"],
                context_dense_timing_v2=control_batch["context_dense_timing_v2"],
                normalized_difficulty=control_batch["normalized_difficulty"].reshape(batch_size),
                context_padding_mask=control_batch["context_padding_mask"],
                full_mel=control_batch.get("full_mel"),
                full_dense_timing_v2=control_batch.get("full_dense_timing_v2"),
                padding_mask=control_batch.get("padding_mask"),
                frame_count=control_batch.get("frame_count"),
                target_start_frame=control_batch.get("target_start_frame"),
            )
            memory = getattr(output, "control_memory", None)
            if not isinstance(memory, torch.Tensor) or memory.ndim != 3:
                raise ValueError(f"control output {slice_index} control_memory must have shape [B,T,D]")
            if int(memory.shape[0]) != batch_size:
                raise ValueError(f"control output {slice_index} batch must be {batch_size}, got {memory.shape[0]}")
            target_start = TARGET_OFFSET_IN_CONTEXT
            target_end = target_start + TARGET_WINDOW_LENGTH_FRAMES
            if int(memory.shape[1]) < target_end:
                raise ValueError(
                    f"control output {slice_index} memory is too short for target slice: "
                    f"{memory.shape[1]} < {target_end}"
                )
            control_memory_slices.append(
                memory[:, target_start:target_end].detach().to(device="cpu", dtype=torch.float32).contiguous()
            )

            value_pred = getattr(output, "value_pred", None)
            if (
                not isinstance(value_pred, torch.Tensor)
                or value_pred.ndim != 3
                or int(value_pred.shape[0]) != batch_size
                or int(value_pred.shape[1]) != TARGET_WINDOW_LENGTH_FRAMES
            ):
                raise ValueError(f"control output {slice_index} value_pred must have shape [B,100,C]")
            if int(value_pred.shape[2]) == 1:
                density = value_pred
            elif int(value_pred.shape[2]) == len(VALUE_FEATURE_NAMES):
                density = value_pred[:, :, density_index : density_index + 1]
            else:
                raise ValueError(
                    f"control output {slice_index} value_pred channel count must be 1 or {len(VALUE_FEATURE_NAMES)}, "
                    f"got {value_pred.shape[2]}"
                )
            density_teacher_slices.append(density.detach().to(device="cpu", dtype=torch.float32).contiguous())
        del control_batch, output, memory, value_pred, density
        _release_torch_device_cache(device)

    return (
        torch.cat(control_memory_slices, dim=1).contiguous(),
        torch.cat(density_teacher_slices, dim=1).contiguous(),
    )


def precompute_mapper_tuple_phase_b_control_teacher_cache(
    *,
    dataset_root: Path = Path("dataset"),
    index_path: Path | None = None,
    eval_index_path: Path | None = None,
    control_v3_timeseries_path: Path | None = None,
    batch_size: int = 4,
    seed: int = 1337,
    device_name: str = "auto",
    init_from_control_checkpoint: Path | None = None,
    num_workers: int = 0,
    max_cached_maps: int | None = None,
    dataset_progress: bool | None = None,
    control_teacher_cache_dir: Path | None = None,
    control_teacher_precompute_batch_size: int | None = None,
    control_teacher_cache_overwrite: bool = False,
    control_model_config: ControlDemoGlobalEncoderConfig | None = None,
    control_model_config_overrides: Mapping[str, Any] | None = None,
) -> MapperTupleControlTeacherCachePrecomputeRunResult:
    _set_deterministic_seed(seed)
    if control_teacher_cache_dir is None:
        raise ValueError("control teacher cache precompute requires control_teacher_cache_dir")
    effective_precompute_batch_size = (
        batch_size if control_teacher_precompute_batch_size is None else int(control_teacher_precompute_batch_size)
    )
    if effective_precompute_batch_size <= 0:
        raise ValueError("control_teacher_precompute_batch_size must be positive")

    dataset_kwargs: dict[str, Any] = {"dataset_root": dataset_root}
    if index_path is not None:
        dataset_kwargs["index_path"] = index_path
    if control_v3_timeseries_path is not None:
        dataset_kwargs["control_v3_timeseries_path"] = control_v3_timeseries_path
    effective_max_cached_maps = DEFAULT_MAX_CACHED_MAPS if max_cached_maps is None else max_cached_maps
    dataset_kwargs["max_cached_maps"] = effective_max_cached_maps
    dataset_kwargs["progress"] = True if dataset_progress is None else bool(dataset_progress)

    resolved_control_model_config = (
        ControlDemoGlobalEncoderConfig(**dict(control_model_config_overrides or {}))
        if control_model_config is None
        else control_model_config
    )
    precompute_device = select_torch_device(device_name)
    precompute_encoder = ControlDemoGlobalEncoder(resolved_control_model_config).to(precompute_device)
    try:
        if init_from_control_checkpoint is not None:
            initialize_global_control_demo_from_control_checkpoint(
                precompute_encoder,
                init_from_control_checkpoint,
            )
        precompute_encoder.eval()
        for parameter in precompute_encoder.parameters():
            parameter.requires_grad_(False)
        source_control_dataset = ControlWindowDataset(**dataset_kwargs)
        source_result = precompute_phase_b_control_teacher_cache_from_control_dataset(
            source_control_dataset,
            cache_dir=control_teacher_cache_dir,
            control_encoder=precompute_encoder,
            batch_size=effective_precompute_batch_size,
            device=precompute_device,
            num_workers=num_workers,
            overwrite=control_teacher_cache_overwrite,
        )
        reports = [{"split": "source", **source_result.to_report()}]
        eval_control_dataset: ControlWindowDataset | None = None
        if eval_index_path is not None:
            eval_dataset_kwargs = dict(dataset_kwargs)
            eval_dataset_kwargs["index_path"] = eval_index_path
            eval_control_dataset = ControlWindowDataset(**eval_dataset_kwargs)
            eval_result = precompute_phase_b_control_teacher_cache_from_control_dataset(
                eval_control_dataset,
                cache_dir=control_teacher_cache_dir,
                control_encoder=precompute_encoder,
                batch_size=effective_precompute_batch_size,
                device=precompute_device,
                num_workers=num_workers,
                overwrite=control_teacher_cache_overwrite,
            )
            reports.append({"split": "eval", **eval_result.to_report()})
    finally:
        del precompute_encoder
        _release_torch_device_cache(precompute_device)

    return MapperTupleControlTeacherCachePrecomputeRunResult(
        reports=reports,
        source_control_dataset=source_control_dataset,
        eval_control_dataset=eval_control_dataset,
    )


def _mapper_tuple_raw_control_indexed_records(
    control_dataset: Dataset[Any],
    *,
    mapper_stride_frames: int = MAPPER_WRITE_FRAMES,
) -> list[tuple[int, ControlWindowRecord]]:
    records = getattr(control_dataset, "records", None)
    if not isinstance(records, Sequence):
        raise TypeError("control teacher cache precompute requires a dataset with records")
    indexed_records: list[tuple[int, ControlWindowRecord]] = []
    skipped_stride = 0
    for index, record in enumerate(records):
        if not isinstance(record, ControlWindowRecord):
            raise TypeError(f"control dataset record {index} must be a ControlWindowRecord")
        if not is_mapper_tuple_window_start_allowed(record, mapper_stride_frames=mapper_stride_frames):
            skipped_stride += 1
            continue
        indexed_records.append((index, record))
    print(
        "mapper_tuple_control_teacher_cache_precompute raw_control_select "
        f"source_windows={len(records)} selected_windows={len(indexed_records)} "
        f"skipped_stride={skipped_stride}",
        flush=True,
    )
    return indexed_records


def _precompute_phase_b_control_teacher_cache_for_indexed_records(
    *,
    control_dataset: Dataset[Any],
    indexed_records: Sequence[tuple[int, ControlWindowRecord]],
    cache_dir: Path,
    control_encoder: ControlDemoGlobalEncoder,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    overwrite: bool,
    source_label: str,
) -> MapperTupleControlTeacherCachePrecomputeResult:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        control_teacher_cache_path(cache_dir, record)
        for _, record in indexed_records
    ]
    missing_indices = [
        index
        for index, path in enumerate(paths)
        if overwrite or not path.exists()
    ]
    skipped_entries = len(paths) - len(missing_indices)
    start_time = time.monotonic()
    print(
        f"mapper_tuple_control_teacher_cache_precompute start source={source_label} "
        f"total={len(paths)} missing={len(missing_indices)} cache_dir={cache_dir.as_posix()}",
        flush=True,
    )
    if not missing_indices:
        elapsed_s = time.monotonic() - start_time
        print(
            f"mapper_tuple_control_teacher_cache_precompute done source={source_label} "
            f"computed=0 skipped={skipped_entries} elapsed_s={elapsed_s:.1f}",
            flush=True,
        )
        return MapperTupleControlTeacherCachePrecomputeResult(
            cache_dir=cache_dir,
            total_entries=len(paths),
            computed_entries=0,
            skipped_entries=skipped_entries,
            elapsed_s=elapsed_s,
        )

    computed_entries = 0
    loader = DataLoader(
        _ControlTeacherPrecomputeDataset(
            control_dataset,
            [indexed_records[index] for index in missing_indices],
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_mapper_tuple_control_teacher_precompute,
    )
    control_encoder.to(device)
    control_encoder.eval()
    offset = 0
    for raw_batch in loader:
        batch: dict[str, Any] | None = None
        control_memory_8s: torch.Tensor | None = None
        density_teacher_8s: torch.Tensor | None = None
        try:
            current_batch_size = int(raw_batch["control_slice_start_frames"].shape[0])
            batch_indices = missing_indices[offset : offset + current_batch_size]
            offset += current_batch_size
            batch = _move_batch_tensors(raw_batch, device, keys=MAPPER_BATCH_TENSOR_KEYS)
            control_memory_8s, density_teacher_8s = _compute_control_teacher_8s_for_precompute(
                control_encoder,
                batch,
                device=device,
            )
            for batch_index, record_index in enumerate(batch_indices):
                record = indexed_records[record_index][1]
                save_control_teacher_cache_entry(
                    paths[record_index],
                    record=record,
                    control_memory_8s=control_memory_8s[batch_index],
                    density_teacher_8s=density_teacher_8s[batch_index],
                )
                computed_entries += 1
            if computed_entries == len(batch_indices) or computed_entries % max(batch_size * 25, 1) == 0:
                print(
                    f"mapper_tuple_control_teacher_cache_precompute progress source={source_label} "
                    f"computed={computed_entries}/{len(missing_indices)}",
                    flush=True,
                )
        finally:
            del raw_batch
            if batch is not None:
                del batch
            if control_memory_8s is not None:
                del control_memory_8s
            if density_teacher_8s is not None:
                del density_teacher_8s
            _release_torch_device_cache(device)

    elapsed_s = time.monotonic() - start_time
    print(
        f"mapper_tuple_control_teacher_cache_precompute done source={source_label} "
        f"computed={computed_entries} skipped={skipped_entries} elapsed_s={elapsed_s:.1f}",
        flush=True,
    )
    return MapperTupleControlTeacherCachePrecomputeResult(
        cache_dir=cache_dir,
        total_entries=len(paths),
        computed_entries=computed_entries,
        skipped_entries=skipped_entries,
        elapsed_s=elapsed_s,
    )


def precompute_phase_b_control_teacher_cache_from_control_dataset(
    control_dataset: Dataset[Any],
    *,
    cache_dir: Path,
    control_encoder: ControlDemoGlobalEncoder,
    batch_size: int = 4,
    device: torch.device,
    num_workers: int = 0,
    overwrite: bool = False,
) -> MapperTupleControlTeacherCachePrecomputeResult:
    indexed_records = _mapper_tuple_raw_control_indexed_records(control_dataset)
    return _precompute_phase_b_control_teacher_cache_for_indexed_records(
        control_dataset=control_dataset,
        indexed_records=indexed_records,
        cache_dir=cache_dir,
        control_encoder=control_encoder,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        overwrite=overwrite,
        source_label="raw_control",
    )


def precompute_phase_b_control_teacher_cache(
    dataset: MapperTupleWindowDataset,
    *,
    cache_dir: Path,
    control_encoder: ControlDemoGlobalEncoder,
    batch_size: int = 4,
    device: torch.device,
    num_workers: int = 0,
    overwrite: bool = False,
) -> MapperTupleControlTeacherCachePrecomputeResult:
    if not isinstance(dataset, MapperTupleWindowDataset):
        raise TypeError("precompute_phase_b_control_teacher_cache requires a MapperTupleWindowDataset")
    result = _precompute_phase_b_control_teacher_cache_for_indexed_records(
        control_dataset=dataset.control_dataset,
        indexed_records=[
            (mapper_record.control_record_index, mapper_record.control_record)
            for mapper_record in dataset.records
        ],
        cache_dir=cache_dir,
        control_encoder=control_encoder,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        overwrite=overwrite,
        source_label="mapper_filtered",
    )
    dataset.control_teacher_cache_dir = Path(cache_dir)
    return result


def compute_phase_b_loss(
    model_output: MapperTupleModelOutput,
    *,
    target_fragment_tokens: torch.Tensor,
    target_fragment_mask: torch.Tensor,
    target_fragment_states: Mapping[str, torch.Tensor],
    close_labels: torch.Tensor,
    close_label_mask: torch.Tensor,
    density_target_8s: torch.Tensor | None = None,
    density_confidence_8s: torch.Tensor | None = None,
    write_start_ms: torch.Tensor,
    vocab: MapperTupleVocab,
    loss_config: MapperTuplePhaseBLossConfig = MapperTuplePhaseBLossConfig(),
) -> MapperTupleLossOutput:
    loss_target = target_fragment_tokens.to(dtype=torch.long, device=model_output.logits_final.device)
    if tuple(loss_target.shape) != tuple(model_output.logits_final.shape[:2]):
        raise ValueError("target_fragment_tokens must align with teacher-forced decoder output")
    if tuple(target_fragment_mask.shape) != tuple(target_fragment_tokens.shape):
        raise ValueError("target_fragment_mask must match target_fragment_tokens")
    target_mask = target_fragment_mask.to(device=model_output.logits_final.device, dtype=torch.bool)
    input_mask = target_mask
    token_loss = token_cross_entropy(
        model_output.logits_final,
        loss_target,
        pad_id=vocab.pad_id,
        target_mask=target_mask,
    )
    close_loss, close_metrics = _ln_close_loss(
        close_logits=model_output.close_logits,
        labels=close_labels.to(device=model_output.close_logits.device),
        mask=(
            close_label_mask.to(device=model_output.close_logits.device, dtype=torch.bool)
            & target_mask.to(device=model_output.close_logits.device, dtype=torch.bool).unsqueeze(-1)
        ),
        max_pos_weight=loss_config.close_pos_weight_max,
    )
    adapter_reg = adapter_bias_regularization(
        model_output.state_prior_bias,
        model_output.ln_close_bias,
        model_output.time_shift_bias,
        mask=input_mask,
    )
    current_ms = target_fragment_states.get("current_ms") if isinstance(target_fragment_states, Mapping) else None
    if density_target_8s is None or density_confidence_8s is None:
        if loss_config.lambda_density > 0.0:
            raise ValueError("density_target_8s and density_confidence_8s are required when lambda_density > 0")
        density_loss = token_loss.new_zeros(())
        density_weight = 0.0
    else:
        if not isinstance(current_ms, torch.Tensor):
            raise ValueError("target_fragment_states.current_ms must be supplied for density loss")
        density_loss = density_auxiliary_loss(
            logits_final=model_output.logits_final,
            current_ms=current_ms.to(device=model_output.logits_final.device),
            write_start_ms=write_start_ms.to(device=model_output.logits_final.device),
            target=density_target_8s.to(device=model_output.logits_final.device),
            confidence=density_confidence_8s.to(device=model_output.logits_final.device),
            vocab=vocab,
            target_mask=target_mask,
            calibration_scale=loss_config.density_calibration_scale,
            calibration_bias=loss_config.density_calibration_bias,
        )
        density_weight = float(density_confidence_8s.detach().to(dtype=torch.float32).clamp_min(0.0).sum().cpu())
    total_loss = (
        token_loss
        + float(loss_config.lambda_ln_close) * close_loss
        + float(loss_config.lambda_adapter_reg) * adapter_reg
        + float(loss_config.lambda_density) * density_loss
    )
    metrics = {
        "loss/total": float(total_loss.detach().cpu()),
        "loss/token": float(token_loss.detach().cpu()),
        "loss/ln_close": float(close_loss.detach().cpu()),
        "loss/adapter_reg": float(adapter_reg.detach().cpu()),
        "loss/density": float(density_loss.detach().cpu()),
        "target/token_count": float(target_mask.sum().detach().cpu()),
        "density/frame_count": density_weight,
        **close_metrics,
    }
    return MapperTupleLossOutput(
        total_loss=total_loss,
        metrics=metrics,
        metric_numerators={
            "loss/token": float(token_loss.detach().cpu()) * max(metrics["target/token_count"], 1.0),
            "loss/ln_close": float(close_loss.detach().cpu()) * max(metrics["ln_close/open_lane_count"], 1.0),
            "loss/density": float(density_loss.detach().cpu()) * max(density_weight, 1.0),
        },
        metric_denominators={
            "loss/token": max(metrics["target/token_count"], 1.0),
            "loss/ln_close": max(metrics["ln_close/open_lane_count"], 1.0),
            "loss/density": max(density_weight, 1.0),
        },
    )


def _move_mapper_batch_tensors(raw_batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    batch = _move_batch_tensors(raw_batch, device, keys=MAPPER_BATCH_TENSOR_KEYS)
    for key in ("target_fragment_states", "ln_carry_in", "ln_carry_out"):
        value = batch.get(key)
        if isinstance(value, Mapping):
            batch[key] = {
                nested_key: nested_value.to(device) if isinstance(nested_value, torch.Tensor) else nested_value
                for nested_key, nested_value in value.items()
            }
    return batch


def _reject_old_mapper_contract(batch: Mapping[str, Any]) -> None:
    old_keys = {
        "target_tokens",
        "target_token_mask",
        "teacher_current_ms",
        "teacher_open_mask",
        "teacher_open_age_ms",
    }
    present = sorted(key for key in old_keys if key in batch)
    if present:
        raise ValueError(
            "old target_tokens/teacher_* mapper contract is not supported by mapper tuple training; "
            f"received {present}"
        )


def _loss_for_raw_batch(
    model: TupleMapperBase,
    raw_batch: Mapping[str, Any],
    *,
    device: torch.device,
    loss_config: MapperTuplePhaseBLossConfig | None = None,
) -> MapperTupleLossOutput:
    _reject_old_mapper_contract(raw_batch)
    batch = _move_mapper_batch_tensors(raw_batch, device)
    if isinstance(batch.get("control_memory_padding_mask_8s"), torch.Tensor):
        raise ValueError("control_memory_padding_mask_8s is not supported in Phase B")
    model_output = model(
        decoder_input_tokens=batch["decoder_input_tokens"],
        target_fragment_tokens=batch["target_fragment_tokens"],
        target_fragment_mask=batch["target_fragment_mask"],
        target_fragment_states=batch["target_fragment_states"],
        ln_carry_in=batch["ln_carry_in"],
        ln_carry_out=batch["ln_carry_out"],
        write_start_ms=batch["write_start_ms"],
        write_end_ms=batch["write_end_ms"],
        is_full_chart_start=batch["is_full_chart_start"],
        is_full_chart_end=batch["is_full_chart_end"],
        difficulty=batch.get("difficulty"),
        normalized_difficulty=batch.get("normalized_difficulty"),
        control_memory_8s=batch.get("control_memory_8s"),
        density_teacher_8s=batch.get("density_teacher_8s"),
        full_mel=batch.get("full_mel"),
        full_dense_timing_v2=batch.get("full_dense_timing_v2"),
        padding_mask=batch.get("padding_mask"),
        frame_count=batch.get("frame_count"),
        source_frame_count=batch.get("source_frame_count"),
        target_start_frame=batch.get("target_start_frame"),
        control_slice_start_frames=batch.get("control_slice_start_frames"),
    )
    return compute_phase_b_loss(
        model_output,
        target_fragment_tokens=batch["target_fragment_tokens"],
        target_fragment_mask=batch["target_fragment_mask"],
        target_fragment_states=batch["target_fragment_states"],
        close_labels=batch["close_labels"],
        close_label_mask=batch["close_label_mask"],
        density_target_8s=batch.get("density_target_8s"),
        density_confidence_8s=batch.get("density_confidence_8s"),
        write_start_ms=batch["write_start_ms"],
        vocab=model.vocab,
        loss_config=MapperTuplePhaseBLossConfig() if loss_config is None else loss_config,
    )


@torch.inference_mode()
def metrics_for_loader(
    model: TupleMapperBase,
    loader: DataLoader,
    *,
    device: torch.device,
    loss_config: MapperTuplePhaseBLossConfig,
) -> dict[str, float]:
    model.eval()
    count_totals: dict[str, float] = {}
    mean_numerators: dict[str, float] = {}
    mean_denominators: dict[str, float] = {}
    fallback_totals: dict[str, float] = {}
    fallback_weights: dict[str, float] = {}
    for raw_batch in loader:
        loss_output = _loss_for_raw_batch(model, raw_batch, device=device, loss_config=loss_config)
        for key, value in loss_output.metrics.items():
            if _metric_is_count(key):
                count_totals[key] = count_totals.get(key, 0.0) + float(value)
        for key, numerator in loss_output.metric_numerators.items():
            mean_numerators[key] = mean_numerators.get(key, 0.0) + float(numerator)
        for key, denominator in loss_output.metric_denominators.items():
            mean_denominators[key] = mean_denominators.get(key, 0.0) + float(denominator)
        unresolved = set(loss_output.metrics) - set(loss_output.metric_numerators) - set(count_totals)
        weight = max(float(loss_output.metrics.get("target/token_count", 0.0)), 1.0)
        for key in unresolved:
            fallback_totals[key] = fallback_totals.get(key, 0.0) + float(loss_output.metrics[key]) * weight
            fallback_weights[key] = fallback_weights.get(key, 0.0) + weight
        del loss_output
    if not (count_totals or mean_numerators or fallback_totals):
        return {"loss/total": math.nan, "loss/token": math.nan, "loss/density": 0.0}
    metrics = dict(count_totals)
    for key, numerator in mean_numerators.items():
        metrics[key] = _safe_float_div(numerator, mean_denominators.get(key, 0.0))
    for key, total in fallback_totals.items():
        metrics[key] = _safe_float_div(total, fallback_weights[key])
    metrics["loss/total"] = (
        metrics.get("loss/token", 0.0)
        + loss_config.lambda_ln_close * metrics.get("loss/ln_close", 0.0)
        + loss_config.lambda_adapter_reg * metrics.get("loss/adapter_reg", 0.0)
        + loss_config.lambda_density * metrics.get("loss/density", 0.0)
    )
    return metrics


def _run_training(
    *,
    loader: DataLoader,
    train_eval_loader: DataLoader,
    eval_loader: DataLoader,
    output_dir: Path,
    model_config: MapperTupleConfig,
    control_model_config: ControlDemoGlobalEncoderConfig | None,
    loss_config: MapperTuplePhaseBLossConfig,
    max_steps: int,
    eval_every: int,
    save_every: int | None,
    log_every: int | None,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device_name: str,
    run_name: str,
    dataset_report: Mapping[str, Any],
    init_from_control_checkpoint: Path | None,
    init_from_mapper_checkpoint: Path | None,
    resume_from: Path | None = None,
    model_factory: Callable[[MapperTupleConfig, ControlDemoGlobalEncoder | None], TupleMapperBase] | None = None,
    mapper_checkpoint_initializer: Callable[..., Mapping[str, Any]] | None = None,
    progress_label: str = "mapper_tuple_phase_b",
    skip_first_eval_pass: bool = False,
    mps_cleanup_every: int | None = None,
) -> ControlTrainingResult:
    _validate_training_args(
        max_steps=max_steps,
        eval_every=eval_every,
        save_every=save_every,
        log_every=log_every,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    if mps_cleanup_every is not None:
        if isinstance(mps_cleanup_every, bool):
            raise ValueError("mps_cleanup_every must be an integer step interval")
        mps_cleanup_every = int(mps_cleanup_every)
        if mps_cleanup_every < 0:
            raise ValueError("mps_cleanup_every must be non-negative")
        if mps_cleanup_every == 0:
            mps_cleanup_every = None
    if resume_from is not None and init_from_mapper_checkpoint is not None:
        raise ValueError("resume_from and init_from_mapper_checkpoint cannot both be set")
    _set_deterministic_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_torch_device(device_name)
    save_every = eval_every if save_every is None else save_every
    checkpoint_path = output_dir / "checkpoint.pt"
    report_path = output_dir / "report.json"

    control_encoder: ControlDemoGlobalEncoder | None = None
    initialization_report: dict[str, Any] | None = None
    if control_model_config is not None:
        control_encoder = ControlDemoGlobalEncoder(control_model_config)
        if init_from_control_checkpoint is not None and init_from_mapper_checkpoint is None and resume_from is None:
            initialization_report = initialize_global_control_demo_from_control_checkpoint(
                control_encoder,
                init_from_control_checkpoint,
            )
    if model_factory is None:
        model = TupleMapperBase(model_config, control_encoder=control_encoder)
    else:
        model = model_factory(model_config, control_encoder)
    if init_from_mapper_checkpoint is not None:
        initializer = initialize_mapper_tuple_from_mapper_checkpoint
        if mapper_checkpoint_initializer is not None:
            initializer = mapper_checkpoint_initializer
        initialization_report = initializer(
            model,
            init_from_mapper_checkpoint,
            expected_model_config=model_config,
            expected_control_model_config=control_model_config,
        )
    model = model.to(device)
    optimizer = _build_mapper_tuple_optimizer(model, learning_rate=learning_rate, weight_decay=weight_decay)
    resume_config = _mapper_resume_training_config(
        seed=seed,
        run_name=run_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        eval_every=eval_every,
        save_every=save_every,
        skip_first_eval_pass=skip_first_eval_pass,
        loss_config=loss_config,
        dataset_report=dataset_report,
        mps_cleanup_every=mps_cleanup_every,
    )
    iterator = _infinite_loader(loader)
    history: list[dict[str, Any]] = []
    completed_step = 0
    last_train_metrics: dict[str, float] = {}
    final_train_metrics: dict[str, float] = {}
    final_eval_metrics: dict[str, float] = {}

    if resume_from is not None:
        checkpoint = _load_mapper_resume_checkpoint(
            resume_from,
            expected_model_config=model_config,
            expected_control_model_config=control_model_config,
            expected_loss_config=loss_config,
            expected_training_config=resume_config,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _move_optimizer_state_to_device(optimizer, device)
        training_state = checkpoint["training_state"]
        completed_step = int(training_state["step"])
        if completed_step > max_steps:
            raise ValueError(f"resume checkpoint step {completed_step} exceeds requested max_steps {max_steps}")
        history = [dict(entry) for entry in checkpoint["history"]]
        last_train_metrics = dict(training_state.get("last_train_metrics", {}))
        final_train_metrics = dict(training_state.get("final_train_metrics", {}))
        final_eval_metrics = dict(training_state.get("final_eval_metrics", {}))
        raw_initialization = checkpoint.get("initialization")
        initialization_report = dict(raw_initialization) if isinstance(raw_initialization, Mapping) else None
        _restore_rng_state(training_state["rng_state"])
        iterator = _advance_training_iterator(iterator, completed_step)
        print(f"{progress_label}_resume checkpoint={resume_from} step={completed_step}/{max_steps}", flush=True)

    log_start_time = time.monotonic()
    log_start_step = completed_step
    for step in range(completed_step + 1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_output = _loss_for_raw_batch(model, next(iterator), device=device, loss_config=loss_config)
        loss_output.total_loss.backward()
        torch.nn.utils.clip_grad_norm_((parameter for parameter in model.parameters() if parameter.requires_grad), 1.0)
        optimizer.step()
        last_train_metrics = dict(loss_output.metrics)
        del loss_output
        completed_step = step
        if mps_cleanup_every is not None and step % mps_cleanup_every == 0:
            _cleanup_mps_training_memory(device)

        should_eval = step == 1 or step % eval_every == 0 or step == max_steps
        if skip_first_eval_pass and step == 1 and step != max_steps:
            should_eval = False
        should_save = step == 1 or step % save_every == 0 or step == max_steps
        should_log = log_every is not None and (step == 1 or step % log_every == 0 or step == max_steps)
        if should_log or should_eval or should_save:
            elapsed_s = time.monotonic() - log_start_time
            completed_since_start = max(step - log_start_step, 1)
            steps_per_s = completed_since_start / max(elapsed_s, 1e-9)
            print(
                f"{progress_label}_progress step={step}/{max_steps} "
                f"loss={last_train_metrics['loss/total']:.6f} "
                f"elapsed_s={elapsed_s:.1f} steps_per_s={steps_per_s:.3f}",
                flush=True,
            )
        if should_eval:
            final_eval_metrics = metrics_for_loader(model, eval_loader, device=device, loss_config=loss_config)
            history_entry: dict[str, Any] = {
                "step": step,
                "train": _json_metrics(last_train_metrics),
                "eval": _json_metrics(final_eval_metrics),
            }
            if step == max_steps:
                final_train_metrics = metrics_for_loader(model, train_eval_loader, device=device, loss_config=loss_config)
                history_entry["train_eval"] = _json_metrics(final_train_metrics)
            history.append(history_entry)
            print(
                f"{progress_label}_eval step={step}/{max_steps} "
                f"loss={final_eval_metrics.get('loss/total', float('nan')):.6f}",
                flush=True,
            )
            _cleanup_mps_training_memory(device)
        if should_save:
            _write_checkpoint_and_report(
                output_dir=output_dir,
                checkpoint_path=checkpoint_path,
                report_path=report_path,
                model=model,
                optimizer=optimizer,
                model_config=model_config,
                control_model_config=control_model_config,
                loss_config=loss_config,
                seed=seed,
                run_name=run_name,
                max_steps=max_steps,
                completed_steps=completed_step,
                eval_every=eval_every,
                save_every=save_every,
                log_every=log_every,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                device=device,
                dataset_report=dataset_report,
                history=history,
                last_train_metrics=last_train_metrics,
                final_train_metrics=final_train_metrics,
                final_eval_metrics=final_eval_metrics,
                initialization_report=initialization_report,
                skip_first_eval_pass=skip_first_eval_pass,
                mps_cleanup_every=mps_cleanup_every,
                resume_from=resume_from,
            )
            _cleanup_mps_training_memory(device)

    result_metrics = final_eval_metrics or last_train_metrics
    return ControlTrainingResult(
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        final_loss=float(result_metrics.get("loss/total", float("nan"))),
        final_value_loss=float(result_metrics.get("loss/token", float("nan"))),
        final_confidence_loss=0.0,
        completed_steps=completed_step,
    )


def _build_mapper_tuple_optimizer(
    model: TupleMapperBase,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise ValueError("mapper model has no trainable parameters")
    return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)


def initialize_mapper_tuple_from_mapper_checkpoint(
    model: TupleMapperBase,
    checkpoint_path: Path,
    *,
    expected_model_config: MapperTupleConfig,
    expected_control_model_config: ControlDemoGlobalEncoderConfig | None,
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError(
            "mapper checkpoint could not be loaded safely with weights_only=True; "
            "use a checkpoint written by the mapper trainer"
        ) from exc
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"mapper checkpoint must contain a mapping: {checkpoint_path}")
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("mapper checkpoint schema version mismatch")
    if checkpoint.get("model_config") != asdict(expected_model_config):
        raise ValueError("mapper checkpoint model_config does not match the requested run")
    expected_control_config = (
        None if expected_control_model_config is None else asdict(expected_control_model_config)
    )
    if checkpoint.get("control_model_config") != expected_control_config:
        raise ValueError("mapper checkpoint control_model_config does not match the requested run")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("mapper checkpoint missing model_state_dict")
    non_tensor_keys = [str(key) for key, value in state.items() if not isinstance(value, torch.Tensor)]
    if non_tensor_keys:
        raise ValueError(f"mapper checkpoint model_state_dict contains non-tensor values: {non_tensor_keys}")

    load_result = model.load_state_dict(state, strict=True)
    loaded_keys = len(state)
    training_state = checkpoint.get("training_state")
    checkpoint_step = None
    if isinstance(training_state, Mapping) and isinstance(training_state.get("step"), int):
        checkpoint_step = int(training_state["step"])
    report = {
        "kind": "mapper_tuple_model_state",
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_step": checkpoint_step,
        "loaded_keys": loaded_keys,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "optimizer_state_loaded": False,
    }
    del checkpoint, state
    return report


def _mapper_resume_training_config(
    *,
    seed: int,
    run_name: str,
    learning_rate: float,
    weight_decay: float,
    eval_every: int,
    save_every: int,
    skip_first_eval_pass: bool,
    loss_config: MapperTuplePhaseBLossConfig,
    dataset_report: Mapping[str, Any],
    mps_cleanup_every: int | None,
) -> dict[str, Any]:
    return {
        "phase": "B",
        "seed": seed,
        "run_name": run_name,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "eval_every": eval_every,
        "skip_first_eval_pass": bool(skip_first_eval_pass),
        "save_every": save_every,
        "mps_cleanup_every": mps_cleanup_every,
        "density_enabled": bool(loss_config.lambda_density > 0.0),
        "dataset": _json_safe(_strict_mapper_resume_dataset_report(dataset_report)),
    }


def _load_mapper_resume_checkpoint(
    resume_from: Path,
    *,
    expected_model_config: MapperTupleConfig,
    expected_control_model_config: ControlDemoGlobalEncoderConfig | None,
    expected_loss_config: MapperTuplePhaseBLossConfig,
    expected_training_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError(
            "mapper resume checkpoint could not be loaded safely with weights_only=True; "
            "use a checkpoint written by the mapper trainer"
        ) from exc
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"mapper resume checkpoint must contain a mapping: {resume_from}")
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("mapper resume checkpoint schema version mismatch")
    if checkpoint.get("model_config") != asdict(expected_model_config):
        raise ValueError("mapper resume checkpoint model_config does not match the requested run")
    expected_control_config = None if expected_control_model_config is None else asdict(expected_control_model_config)
    if checkpoint.get("control_model_config") != expected_control_config:
        raise ValueError("mapper resume checkpoint control_model_config does not match the requested run")
    if checkpoint.get("loss_config") != asdict(expected_loss_config):
        raise ValueError("mapper resume checkpoint loss_config does not match the requested run")
    if _normalized_mapper_resume_training_config(
        checkpoint.get("training_config")
    ) != _normalized_mapper_resume_training_config(expected_training_config):
        raise ValueError("mapper resume checkpoint training_config does not match the requested run")
    if not isinstance(checkpoint.get("model_state_dict"), Mapping):
        raise ValueError("mapper resume checkpoint missing model_state_dict")
    if "optimizer_state_dict" not in checkpoint:
        raise ValueError("mapper resume checkpoint missing optimizer_state_dict")
    if not isinstance(checkpoint.get("training_state"), Mapping):
        raise ValueError("mapper resume checkpoint missing training_state")
    if not isinstance(checkpoint.get("history"), list):
        raise ValueError("mapper resume checkpoint history must be a list")
    training_state = checkpoint["training_state"]
    if not isinstance(training_state.get("step"), int) or training_state["step"] < 0:
        raise ValueError("mapper resume checkpoint training_state.step must be a non-negative integer")
    if "rng_state" not in training_state:
        raise ValueError("mapper resume checkpoint missing training_state.rng_state")
    return checkpoint


def _normalized_mapper_resume_training_config(config: object) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    normalized = {
        key: value
        for key, value in config.items()
        if key not in MAPPER_RESUME_TRAINING_RUNTIME_KEYS
    }
    dataset = normalized.get("dataset")
    if isinstance(dataset, Mapping):
        normalized["dataset"] = _strict_mapper_resume_dataset_report(dataset)
    return normalized


def _strict_mapper_resume_dataset_report(dataset_report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in dataset_report.items()
        if key not in MAPPER_RESUME_DATASET_RUNTIME_KEYS
    }


def _ln_close_loss(
    *,
    close_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    max_pos_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if tuple(labels.shape) != tuple(close_logits.shape) or tuple(mask.shape) != tuple(close_logits.shape):
        raise ValueError("close labels and mask must align with close_logits")
    mask = mask.to(dtype=torch.bool)
    open_count = int(mask.sum().detach().cpu().item())
    if open_count == 0:
        zero = close_logits.sum() * 0.0
        return zero, {
            "ln_close/open_lane_count": 0.0,
            "ln_close/positive_count": 0.0,
            "ln_close/pos_weight": 1.0,
        }
    labels_f = labels.to(dtype=close_logits.dtype)
    positive_count = float((labels_f * mask.to(dtype=labels_f.dtype)).sum().detach().cpu().item())
    negative_count = max(float(open_count) - positive_count, 0.0)
    pos_weight = 1.0 if positive_count <= 0.0 else min(max(negative_count / positive_count, 1.0), max_pos_weight)
    loss = F.binary_cross_entropy_with_logits(
        close_logits,
        labels_f,
        pos_weight=close_logits.new_tensor(pos_weight),
        reduction="none",
    )
    loss = (loss * mask.to(dtype=loss.dtype)).sum() / max(float(open_count), 1.0)
    return loss, {
        "ln_close/open_lane_count": float(open_count),
        "ln_close/positive_count": positive_count,
        "ln_close/pos_weight": float(pos_weight),
    }


def _write_checkpoint_and_report(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    report_path: Path,
    model: TupleMapperBase,
    optimizer: torch.optim.Optimizer,
    model_config: MapperTupleConfig,
    control_model_config: ControlDemoGlobalEncoderConfig | None,
    loss_config: MapperTuplePhaseBLossConfig,
    seed: int,
    run_name: str,
    max_steps: int,
    completed_steps: int,
    eval_every: int,
    save_every: int,
    log_every: int | None,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    dataset_report: Mapping[str, Any],
    history: list[dict[str, Any]],
    last_train_metrics: Mapping[str, float],
    final_train_metrics: Mapping[str, float],
    final_eval_metrics: Mapping[str, float],
    initialization_report: Mapping[str, Any] | None,
    skip_first_eval_pass: bool,
    mps_cleanup_every: int | None,
    resume_from: Path | None,
) -> None:
    training_config = {
        "phase": "B",
        "seed": seed,
        "run_name": run_name,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "eval_every": eval_every,
        "skip_first_eval_pass": bool(skip_first_eval_pass),
        "save_every": save_every,
        "mps_cleanup_every": mps_cleanup_every,
        "density_enabled": bool(loss_config.lambda_density > 0.0),
        "dataset": _json_safe(dataset_report),
    }
    checkpoint_payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "control_model_config": None if control_model_config is None else asdict(control_model_config),
        "loss_config": asdict(loss_config),
        "training_config": training_config,
        "seed": seed,
        "run_name": run_name,
        "history": history,
        "initialization": None if initialization_report is None else dict(initialization_report),
        "training_state": {
            "step": completed_steps,
            "max_steps": max_steps,
            "is_complete": completed_steps >= max_steps,
            "eval_every": eval_every,
            "skip_first_eval_pass": bool(skip_first_eval_pass),
            "save_every": save_every,
            "log_every": log_every,
            "mps_cleanup_every": mps_cleanup_every,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "device": str(device),
            "resume_from": resume_from.as_posix() if resume_from is not None else None,
            "last_train_metrics": _json_metrics(last_train_metrics),
            "final_train_metrics": _json_metrics(final_train_metrics),
            "final_eval_metrics": _json_metrics(final_eval_metrics),
            "rng_state": _capture_rng_state(),
        },
    }
    archive_path = output_dir / "checkpoints" / f"checkpoint_step_{completed_steps:06d}.pt"
    _atomic_torch_save(checkpoint_payload, archive_path)
    _copy_file_atomically(archive_path, checkpoint_path)
    report_payload = {
        "run_name": run_name,
        "phase": "B",
        "seed": seed,
        "max_steps": max_steps,
        "completed_steps": completed_steps,
        "is_complete": completed_steps >= max_steps,
        "eval_every": eval_every,
        "skip_first_eval_pass": bool(skip_first_eval_pass),
        "save_every": save_every,
        "log_every": log_every,
        "mps_cleanup_every": mps_cleanup_every,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "resume_from": resume_from.as_posix() if resume_from is not None else None,
        "model_config": asdict(model_config),
        "control_model_config": None if control_model_config is None else asdict(control_model_config),
        "loss_config": asdict(loss_config),
        "training_config": training_config,
        "device": str(device),
        "parameter_count": model.parameter_count(),
        "dataset": dict(dataset_report),
        "initialization": None if initialization_report is None else dict(initialization_report),
        "history": history,
        "last_train_metrics": _json_metrics(last_train_metrics),
        "final_train_metrics": _json_metrics(final_train_metrics),
        "final_eval_metrics": _json_metrics(final_eval_metrics),
    }
    _write_report(report_path, report_payload)


def _open_start_from_age(*, current_ms: torch.Tensor, open_mask: torch.Tensor, open_age_ms: torch.Tensor) -> torch.Tensor:
    current = current_ms.reshape(*current_ms.shape, 1).expand_as(open_age_ms)
    return torch.where(open_mask.to(dtype=torch.bool), current - open_age_ms.to(dtype=torch.long), torch.full_like(open_age_ms, -1))


def _normalized_section(source: object, *, allowed: set[str], name: str) -> dict[str, Any]:
    if source is None:
        return {}
    if not isinstance(source, dict):
        raise ValueError(f"{name} must be a mapping")
    normalized = _normalize_config_mapping(source, source_name=name)
    unknown = sorted(set(normalized) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} keys: {unknown}")
    return normalized


def _normalize_config_mapping(source: dict[Any, Any], *, source_name: str) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_key, value in source.items():
        if not isinstance(raw_key, str):
            raise ValueError(f"{source_name} keys must be strings")
        key = raw_key.replace("-", "_")
        if key in normalized:
            raise ValueError(f"{source_name} contains duplicate key after normalization: {key}")
        normalized[key] = value
    return normalized


def main(argv: Sequence[str] | None = None) -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None, help="YAML run config; CLI flags override config values")
    config_args, _ = config_parser.parse_known_args(argv)
    config_defaults = load_run_config(config_args.config) if config_args.config is not None else {
        "model": {},
        "control_model": {},
        "loss": {},
    }
    model_defaults = config_defaults["model"]
    control_model_defaults = config_defaults["control_model"]
    loss_defaults = config_defaults["loss"]

    parser = argparse.ArgumentParser(description="Train the Stage 2 mapper tuple Phase B teacher-forced skeleton.")
    parser.add_argument("--config", default=config_args.config)
    parser.add_argument("--dataset-root", default=config_defaults.get("dataset_root", "dataset"))
    parser.add_argument("--index-path", default=config_defaults.get("index_path"))
    parser.add_argument("--eval-index-path", default=config_defaults.get("eval_index_path"))
    parser.add_argument("--control-v3-timeseries-path", default=config_defaults.get("control_v3_timeseries_path"))
    parser.add_argument("--output-dir", default=config_defaults.get("output_dir", DEFAULT_OUTPUT_DIR.as_posix()))
    parser.add_argument("--max-steps", type=int, default=config_defaults.get("max_steps", 5000))
    parser.add_argument("--eval-every", type=int, default=config_defaults.get("eval_every", 100))
    parser.add_argument("--save-every", type=int, default=config_defaults.get("save_every"))
    parser.add_argument("--log-every", type=int, default=config_defaults.get("log_every"))
    parser.add_argument("--mps-cleanup-every", type=int, default=config_defaults.get("mps_cleanup_every"))
    parser.add_argument("--batch-size", type=int, default=config_defaults.get("batch_size", 4))
    parser.add_argument("--learning-rate", type=float, default=config_defaults.get("learning_rate", 2e-4))
    parser.add_argument("--weight-decay", type=float, default=config_defaults.get("weight_decay", 0.01))
    parser.add_argument("--seed", type=int, default=config_defaults.get("seed", 1337))
    parser.add_argument("--device", default=config_defaults.get("device", "auto"), choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--run-name", default=config_defaults.get("run_name", "mapper_tuple_phase_b_teacher_forced"))
    parser.add_argument("--init-from-control-checkpoint", default=config_defaults.get("init_from_control_checkpoint"))
    parser.add_argument("--init-from-mapper-checkpoint", default=config_defaults.get("init_from_mapper_checkpoint"))
    parser.add_argument("--resume-from", default=config_defaults.get("resume_from"))
    parser.add_argument("--eval-fraction", type=float, default=config_defaults.get("eval_fraction", 0.1))
    parser.add_argument("--eval-size", type=int, default=config_defaults.get("eval_size"))
    parser.add_argument(
        "--final-train-eval-size",
        type=int,
        default=config_defaults.get("final_train_eval_size", DEFAULT_FINAL_TRAIN_EVAL_SIZE),
    )
    parser.add_argument("--num-workers", type=int, default=config_defaults.get("num_workers", 0))
    parser.add_argument("--max-cached-maps", type=int, default=config_defaults.get("max_cached_maps"))
    parser.add_argument("--mapper-record-cache-path", default=config_defaults.get("mapper_record_cache_path"))
    parser.add_argument(
        "--dataset-progress",
        action=argparse.BooleanOptionalAction,
        default=config_defaults.get("dataset_progress"),
    )
    parser.add_argument(
        "--length-bucketed-batches",
        action=argparse.BooleanOptionalAction,
        default=bool(config_defaults.get("length_bucketed_batches", False)),
    )
    parser.add_argument(
        "--length-bucket-size-multiplier",
        type=int,
        default=config_defaults.get("length_bucket_size_multiplier", 32),
    )
    parser.add_argument("--control-teacher-cache-dir", default=config_defaults.get("control_teacher_cache_dir"))
    parser.add_argument(
        "--precompute-control-teacher-cache",
        action="store_true",
        default=bool(config_defaults.get("precompute_control_teacher_cache", False)),
    )
    parser.add_argument(
        "--precompute-control-teacher-cache-only",
        action="store_true",
        default=bool(config_defaults.get("precompute_control_teacher_cache_only", False)),
    )
    parser.add_argument(
        "--control-teacher-precompute-batch-size",
        type=int,
        default=config_defaults.get("control_teacher_precompute_batch_size"),
    )
    parser.add_argument(
        "--require-control-teacher-cache",
        action="store_true",
        default=bool(config_defaults.get("require_control_teacher_cache", False)),
    )
    parser.add_argument(
        "--control-teacher-cache-overwrite",
        action="store_true",
        default=bool(config_defaults.get("control_teacher_cache_overwrite", False)),
    )
    args = parser.parse_args(argv)

    init_from = Path(args.init_from_control_checkpoint) if args.init_from_control_checkpoint is not None else None
    init_from_mapper = (
        Path(args.init_from_mapper_checkpoint)
        if args.init_from_mapper_checkpoint is not None
        else None
    )
    resume_from = Path(args.resume_from) if args.resume_from is not None else None
    if args.precompute_control_teacher_cache_only:
        result = precompute_mapper_tuple_phase_b_control_teacher_cache(
            dataset_root=Path(args.dataset_root),
            index_path=Path(args.index_path) if args.index_path is not None else None,
            eval_index_path=Path(args.eval_index_path) if args.eval_index_path is not None else None,
            control_v3_timeseries_path=(
                Path(args.control_v3_timeseries_path)
                if args.control_v3_timeseries_path is not None
                else None
            ),
            batch_size=args.batch_size,
            seed=args.seed,
            device_name=args.device,
            init_from_control_checkpoint=init_from,
            num_workers=args.num_workers,
            max_cached_maps=args.max_cached_maps,
            dataset_progress=args.dataset_progress,
            control_teacher_cache_dir=(
                Path(args.control_teacher_cache_dir)
                if args.control_teacher_cache_dir is not None
                else None
            ),
            control_teacher_precompute_batch_size=args.control_teacher_precompute_batch_size,
            control_teacher_cache_overwrite=args.control_teacher_cache_overwrite,
            control_model_config_overrides=control_model_defaults,
        )
        for report in result.reports:
            print(
                "control_teacher_cache_report "
                f"split={report['split']} total={report['total_entries']} "
                f"computed={report['computed_entries']} skipped={report['skipped_entries']} "
                f"elapsed_s={float(report['elapsed_s']):.1f}",
            )
        return

    result = run_mapper_tuple_phase_b_training(
        dataset_root=Path(args.dataset_root),
        index_path=Path(args.index_path) if args.index_path is not None else None,
        eval_index_path=Path(args.eval_index_path) if args.eval_index_path is not None else None,
        control_v3_timeseries_path=(
            Path(args.control_v3_timeseries_path)
            if args.control_v3_timeseries_path is not None
            else None
        ),
        output_dir=Path(args.output_dir),
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        save_every=args.save_every,
        log_every=args.log_every,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device_name=args.device,
        run_name=args.run_name,
        init_from_control_checkpoint=init_from,
        init_from_mapper_checkpoint=init_from_mapper,
        resume_from=resume_from,
        eval_fraction=args.eval_fraction,
        eval_size=args.eval_size,
        final_train_eval_size=args.final_train_eval_size,
        num_workers=args.num_workers,
        max_cached_maps=args.max_cached_maps,
        dataset_progress=args.dataset_progress,
        mapper_record_cache_path=(
            Path(args.mapper_record_cache_path)
            if args.mapper_record_cache_path is not None
            else None
        ),
        length_bucketed_batches=args.length_bucketed_batches,
        length_bucket_size_multiplier=args.length_bucket_size_multiplier,
        control_teacher_cache_dir=(
            Path(args.control_teacher_cache_dir)
            if args.control_teacher_cache_dir is not None
            else None
        ),
        precompute_control_teacher_cache=args.precompute_control_teacher_cache,
        control_teacher_precompute_batch_size=args.control_teacher_precompute_batch_size,
        require_control_teacher_cache=args.require_control_teacher_cache,
        control_teacher_cache_overwrite=args.control_teacher_cache_overwrite,
        mps_cleanup_every=args.mps_cleanup_every,
        model_config_overrides=model_defaults,
        control_model_config_overrides=control_model_defaults,
        loss_config_overrides=loss_defaults,
    )
    print(f"report_path {result.report_path}")
    print(f"checkpoint_path {result.checkpoint_path}")
    print(f"final_loss {result.final_loss:.6f}")
    print(f"completed_steps {result.completed_steps}")


if __name__ == "__main__":
    main()
