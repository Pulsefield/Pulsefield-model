from __future__ import annotations

import argparse
import pickle
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
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
    CONTROL_MODEL_CONFIG_KEYS,
    LOSS_CONFIG_KEYS,
    RUN_CONFIG_KEYS as MAPPER_TUPLE_RUN_CONFIG_KEYS,
    MapperTuplePhaseBLossConfig,
    _make_mapper_tuple_phase_b_train_loader,
    _normalize_config_mapping,
    _normalized_section,
    _run_training,
    precompute_mapper_tuple_phase_b_control_teacher_cache,
)


DEFAULT_RUNS_ROOT = Path("artifacts/runs/stage2_mapper_v2")
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_ROOT / "phase_b_global_teacher_forced"
RUN_CONFIG_KEYS = set(MAPPER_TUPLE_RUN_CONFIG_KEYS) | {"include_full_song_context", "skip_first_eval_pass"}
MODEL_CONFIG_KEYS = {field.name for field in fields(MapperV2Config)}


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

    parser = argparse.ArgumentParser(description="Train the Stage 2 mapper v2 Phase B global teacher-forced model.")
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
    parser.add_argument("--batch-size", type=int, default=config_defaults.get("batch_size", 2))
    parser.add_argument("--learning-rate", type=float, default=config_defaults.get("learning_rate", 2e-4))
    parser.add_argument("--weight-decay", type=float, default=config_defaults.get("weight_decay", 0.01))
    parser.add_argument("--seed", type=int, default=config_defaults.get("seed", 1337))
    parser.add_argument("--device", default=config_defaults.get("device", "auto"), choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--run-name", default=config_defaults.get("run_name", "mapper_v2_phase_b_global_teacher_forced"))
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
    parser.add_argument(
        "--include-full-song-context",
        action=argparse.BooleanOptionalAction,
        default=bool(config_defaults.get("include_full_song_context", True)),
    )
    parser.add_argument(
        "--skip-first-eval-pass",
        action=argparse.BooleanOptionalAction,
        default=bool(config_defaults.get("skip_first_eval_pass", True)),
    )
    args = parser.parse_args(argv)

    init_from = Path(args.init_from_control_checkpoint) if args.init_from_control_checkpoint is not None else None
    init_from_mapper = Path(args.init_from_mapper_checkpoint) if args.init_from_mapper_checkpoint is not None else None
    resume_from = Path(args.resume_from) if args.resume_from is not None else None
    if args.precompute_control_teacher_cache_only:
        result = precompute_mapper_tuple_phase_b_control_teacher_cache(
            dataset_root=Path(args.dataset_root),
            index_path=Path(args.index_path) if args.index_path is not None else None,
            eval_index_path=Path(args.eval_index_path) if args.eval_index_path is not None else None,
            control_v3_timeseries_path=(
                Path(args.control_v3_timeseries_path) if args.control_v3_timeseries_path is not None else None
            ),
            batch_size=args.batch_size,
            seed=args.seed,
            device_name=args.device,
            init_from_control_checkpoint=init_from,
            num_workers=args.num_workers,
            max_cached_maps=args.max_cached_maps,
            dataset_progress=args.dataset_progress,
            control_teacher_cache_dir=(
                Path(args.control_teacher_cache_dir) if args.control_teacher_cache_dir is not None else None
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

    result = run_mapper_v2_phase_b_training(
        dataset_root=Path(args.dataset_root),
        index_path=Path(args.index_path) if args.index_path is not None else None,
        eval_index_path=Path(args.eval_index_path) if args.eval_index_path is not None else None,
        control_v3_timeseries_path=(
            Path(args.control_v3_timeseries_path) if args.control_v3_timeseries_path is not None else None
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
            Path(args.mapper_record_cache_path) if args.mapper_record_cache_path is not None else None
        ),
        length_bucketed_batches=args.length_bucketed_batches,
        length_bucket_size_multiplier=args.length_bucket_size_multiplier,
        control_teacher_cache_dir=(
            Path(args.control_teacher_cache_dir) if args.control_teacher_cache_dir is not None else None
        ),
        precompute_control_teacher_cache=args.precompute_control_teacher_cache,
        control_teacher_precompute_batch_size=args.control_teacher_precompute_batch_size,
        require_control_teacher_cache=args.require_control_teacher_cache,
        control_teacher_cache_overwrite=args.control_teacher_cache_overwrite,
        include_full_song_context=args.include_full_song_context,
        skip_first_eval_pass=args.skip_first_eval_pass,
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
