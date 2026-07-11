from __future__ import annotations

import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from pulsefield_model.data.control_windows import DEFAULT_MAX_CACHED_MAPS
from pulsefield_model.data.mapper_tuple_windows import MapperTupleWindowDataset, collate_mapper_tuple_windows
from pulsefield_model.models.control import ControlDemoGlobalEncoderConfig
from pulsefield_model.models.mapper.v2 import MapperV2Config, MapperV2Model
from pulsefield_model.training.common import (
    CHECKPOINT_SCHEMA_VERSION,
    DEFAULT_FINAL_TRAIN_EVAL_SIZE,
    ControlTrainingResult,
    _set_deterministic_seed,
    limit_final_train_eval_dataset,
    split_train_eval_dataset,
)
from pulsefield_model.training.mapper_common import (
    RUN_CONFIG_KEYS as MAPPER_TUPLE_RUN_CONFIG_KEYS,
    MapperTuplePhaseBLossConfig,
    _make_mapper_tuple_phase_b_train_loader,
    _run_training,
    precompute_mapper_tuple_phase_b_control_teacher_cache,
)


DEFAULT_RUNS_ROOT = Path("artifacts/runs/stage2_mapper_v2")
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_ROOT / "phase_b_global_teacher_forced"
RUN_CONFIG_KEYS = set(MAPPER_TUPLE_RUN_CONFIG_KEYS) | {"include_full_song_context", "skip_first_eval_pass"}


def run_mapper_v2_phase_b_training(
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
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    seed: int = 1337,
    device_name: str = "auto",
    run_name: str = "mapper_v2_phase_b_global_teacher_forced",
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
    include_full_song_context: bool = True,
    skip_first_eval_pass: bool = True,
    mps_cleanup_every: int | None = None,
    model_config_overrides: Mapping[str, Any] | None = None,
    control_model_config_overrides: Mapping[str, Any] | None = None,
    loss_config_overrides: Mapping[str, Any] | None = None,
) -> ControlTrainingResult:
    _set_deterministic_seed(seed)
    control_model_config = ControlDemoGlobalEncoderConfig(**dict(control_model_config_overrides or {}))
    model_values = dict(model_config_overrides or {})
    model_values.setdefault("control_dim", control_model_config.d_model)
    model_config = MapperV2Config(**model_values)
    if model_config.control_dim != control_model_config.d_model:
        raise ValueError("mapper control_dim must match frozen control_model d_model")
    if model_config.use_global_context and not include_full_song_context:
        raise ValueError("Mapper V2 global context requires include_full_song_context=True")
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
    source_control_dataset = None
    eval_control_dataset = None
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
    mapper_dataset_kwargs["include_full_song_context"] = bool(include_full_song_context)
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
        eval_kwargs["include_full_song_context"] = bool(include_full_song_context)
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
            "include_full_song_context": bool(include_full_song_context),
            "skip_first_eval_pass": bool(skip_first_eval_pass),
        },
        init_from_control_checkpoint=init_from_control_checkpoint,
        init_from_mapper_checkpoint=init_from_mapper_checkpoint,
        resume_from=resume_from,
        model_factory=_mapper_v2_model_factory,
        mapper_checkpoint_initializer=initialize_mapper_v2_from_mapper_checkpoint,
        progress_label="mapper_v2_phase_b",
        skip_first_eval_pass=skip_first_eval_pass,
        mps_cleanup_every=mps_cleanup_every,
    )


def initialize_mapper_v2_from_mapper_checkpoint(
    model: MapperV2Model,
    checkpoint_path: Path,
    *,
    expected_model_config: MapperV2Config,
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
        raise ValueError("mapper checkpoint model_config does not match the requested mapper v2 run")
    expected_control_config = None if expected_control_model_config is None else asdict(expected_control_model_config)
    if checkpoint.get("control_model_config") != expected_control_config:
        raise ValueError("mapper checkpoint control_model_config does not match the requested mapper v2 run")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("mapper checkpoint missing model_state_dict")
    non_tensor_keys = [str(key) for key, value in state.items() if not isinstance(value, torch.Tensor)]
    if non_tensor_keys:
        raise ValueError(f"mapper checkpoint model_state_dict contains non-tensor values: {non_tensor_keys}")

    load_result = model.load_state_dict(state, strict=True)
    training_state = checkpoint.get("training_state")
    checkpoint_step = None
    if isinstance(training_state, Mapping) and isinstance(training_state.get("step"), int):
        checkpoint_step = int(training_state["step"])
    report = {
        "kind": "mapper_v2_model_state",
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_step": checkpoint_step,
        "loaded_keys": len(state),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "optimizer_state_loaded": False,
    }
    del checkpoint, state
    return report


def _mapper_v2_model_factory(model_config: MapperV2Config, control_encoder: torch.nn.Module | None) -> MapperV2Model:
    if not isinstance(model_config, MapperV2Config):
        raise TypeError("mapper v2 training requires MapperV2Config")
    return MapperV2Model(model_config, control_encoder=control_encoder)


def main(argv: Sequence[str] | None = None) -> None:
    from pulsefield_model.training.mapper_training_hydra import run_mapper_preset_cli

    run_mapper_preset_cli(
        argv,
        mapper_preset="v2_tuple_d384_l4_phase_b",
        v2_runner=run_mapper_v2_phase_b_training,
        precompute_runner=precompute_mapper_tuple_phase_b_control_teacher_cache,
    )


if __name__ == "__main__":
    main()
