from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from torch.utils.data import DataLoader

from pulsefield_model.data.control_windows import DEFAULT_MAX_CACHED_MAPS
from pulsefield_model.data.mapper_sparse_windows_v2_1 import MapperV21WindowDataset, collate_mapper_v2_1_windows
from pulsefield_model.models.control import ControlDemoGlobalEncoder, ControlDemoGlobalEncoderConfig
from pulsefield_model.models.mapper.v2_1 import (
    MapperV21Config,
    MapperV21LossConfig,
    MapperV21Model,
    MapperV21ModelLoss,
)
from pulsefield_model.training.common import (
    DEFAULT_FINAL_TRAIN_EVAL_SIZE,
    ControlTrainingResult,
    ResumableRandomBatchSampler,
    _json_safe,
    _metric_is_count,
    _set_deterministic_seed,
    limit_final_train_eval_dataset,
    split_train_eval_dataset,
)
from pulsefield_model.training.mapper_common import (
    _build_mapper_tuple_optimizer,
    _cleanup_mps_training_memory,
    _move_mapper_batch_tensors,
    precompute_mapper_tuple_phase_b_control_teacher_cache,
)
from pulsefield_model.training.mapper_runner import (
    MapperTrainingSpec,
    default_mapper_metric_finalizer,
    load_mapper_training_resume_checkpoint,
    mapper_metrics_for_loader,
    resume_resumable_loader_cursor_or_advance,
    run_mapper_training,
    write_mapper_checkpoint_and_report,
)


DEFAULT_RUNS_ROOT = Path("artifacts/runs/stage2_mapper_v2_1")
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_ROOT / "phase_b_sparse_global"
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
    "resume_from",
    "eval_fraction",
    "eval_size",
    "final_train_eval_size",
    "num_workers",
    "max_cached_maps",
    "dataset_progress",
    "mapper_record_cache_path",
    "control_teacher_cache_dir",
    "require_control_teacher_cache",
    "precompute_control_teacher_cache",
    "precompute_control_teacher_cache_only",
    "control_teacher_precompute_batch_size",
    "control_teacher_cache_overwrite",
    "include_full_song_context",
    "skip_first_eval_pass",
    "mps_cleanup_every",
    "model",
    "control_model",
    "loss",
}
MODEL_CONFIG_KEYS = {field.name for field in fields(MapperV21Config)}
CONTROL_MODEL_CONFIG_KEYS = {field.name for field in fields(ControlDemoGlobalEncoderConfig)}
LOSS_CONFIG_KEYS = {field.name for field in fields(MapperV21LossConfig)}
MAPPER_V2_1_RESUME_TRAINING_RUNTIME_KEYS = frozenset(("mps_cleanup_every",))
MAPPER_V2_1_RESUME_DATASET_RUNTIME_KEYS = frozenset(("max_cached_maps", "num_workers", "dataset_progress"))


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
    config = dict(loaded)
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


def run_mapper_v2_1_phase_b_training(
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
    run_name: str = "mapper_v2_1_phase_b_sparse_global",
    init_from_control_checkpoint: Path | None = None,
    resume_from: Path | None = None,
    eval_fraction: float = 0.1,
    eval_size: int | None = None,
    final_train_eval_size: int | None = DEFAULT_FINAL_TRAIN_EVAL_SIZE,
    num_workers: int = 0,
    max_cached_maps: int | None = None,
    dataset_progress: bool = False,
    mapper_record_cache_path: Path | None = None,
    control_teacher_cache_dir: Path | None = None,
    require_control_teacher_cache: bool = False,
    precompute_control_teacher_cache: bool = False,
    control_teacher_precompute_batch_size: int | None = None,
    control_teacher_cache_overwrite: bool = False,
    include_full_song_context: bool = True,
    skip_first_eval_pass: bool = False,
    mps_cleanup_every: int | None = None,
    model_config_overrides: Mapping[str, Any] | None = None,
    control_model_config_overrides: Mapping[str, Any] | None = None,
    loss_config_overrides: Mapping[str, Any] | None = None,
) -> ControlTrainingResult:
    _set_deterministic_seed(seed)
    control_model_config = ControlDemoGlobalEncoderConfig(**dict(control_model_config_overrides or {}))
    model_values = dict(model_config_overrides or {})
    model_values.setdefault("control_dim", control_model_config.d_model)
    model_config = MapperV21Config(**model_values)
    if model_config.control_dim != control_model_config.d_model:
        raise ValueError("mapper control_dim must match frozen control_model d_model")
    if model_config.use_global_context and not include_full_song_context:
        raise ValueError("Mapper V2.1 global context requires include_full_song_context=True")
    loss_config = MapperV21LossConfig(**dict(loss_config_overrides or {}))

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
            dataset_progress=dataset_progress,
            control_teacher_cache_dir=control_teacher_cache_dir,
            control_teacher_precompute_batch_size=control_teacher_precompute_batch_size,
            control_teacher_cache_overwrite=control_teacher_cache_overwrite,
            control_model_config=control_model_config,
        )
        cache_precompute_reports = precompute_run.reports
        source_control_dataset = precompute_run.source_control_dataset
        eval_control_dataset = precompute_run.eval_control_dataset

    if source_control_dataset is None:
        dataset_kwargs: dict[str, Any] = {"dataset_root": dataset_root}
        if index_path is not None:
            dataset_kwargs["index_path"] = index_path
        if control_v3_timeseries_path is not None:
            dataset_kwargs["control_v3_timeseries_path"] = control_v3_timeseries_path
    else:
        dataset_kwargs = {"control_dataset": source_control_dataset}
    dataset_kwargs.update(
        {
            "include_full_song_context": bool(include_full_song_context),
            "max_cached_maps": DEFAULT_MAX_CACHED_MAPS if max_cached_maps is None else max_cached_maps,
            "progress": bool(dataset_progress),
        }
    )
    if mapper_record_cache_path is not None:
        dataset_kwargs["mapper_record_cache_path"] = mapper_record_cache_path
    if control_teacher_cache_dir is not None:
        dataset_kwargs["control_teacher_cache_dir"] = control_teacher_cache_dir
        dataset_kwargs["require_control_teacher_cache"] = bool(require_control_teacher_cache)
    train_source = MapperV21WindowDataset(**dataset_kwargs)
    if len(train_source) == 0:
        raise ValueError("MapperV21WindowDataset produced no training windows")

    if eval_index_path is not None:
        if eval_control_dataset is None:
            eval_kwargs = dict(dataset_kwargs)
            eval_kwargs["index_path"] = eval_index_path
        else:
            eval_kwargs = {
                "control_dataset": eval_control_dataset,
                "include_full_song_context": bool(include_full_song_context),
                "max_cached_maps": DEFAULT_MAX_CACHED_MAPS if max_cached_maps is None else max_cached_maps,
                "progress": bool(dataset_progress),
            }
            if mapper_record_cache_path is not None:
                eval_kwargs["mapper_record_cache_path"] = mapper_record_cache_path
            if control_teacher_cache_dir is not None:
                eval_kwargs["control_teacher_cache_dir"] = control_teacher_cache_dir
                eval_kwargs["require_control_teacher_cache"] = bool(require_control_teacher_cache)
        eval_dataset = MapperV21WindowDataset(**eval_kwargs)
        train_dataset = train_source
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

    loader = DataLoader(
        train_dataset,
        batch_sampler=ResumableRandomBatchSampler(
            train_dataset,
            batch_size=batch_size,
            seed=seed,
        ),
        num_workers=num_workers,
        collate_fn=collate_mapper_v2_1_windows,
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
        collate_fn=collate_mapper_v2_1_windows,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_mapper_v2_1_windows,
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
            "mapper_record_cache_path": (
                mapper_record_cache_path.as_posix() if mapper_record_cache_path is not None else None
            ),
            "num_workers": num_workers,
            "control_teacher_cache_dir": (
                control_teacher_cache_dir.as_posix() if control_teacher_cache_dir is not None else None
            ),
            "require_control_teacher_cache": bool(require_control_teacher_cache),
            "precompute_control_teacher_cache": bool(precompute_control_teacher_cache),
            "control_teacher_precompute_batch_size": control_teacher_precompute_batch_size,
            "control_teacher_cache_overwrite": bool(control_teacher_cache_overwrite),
            "control_teacher_cache_precompute": cache_precompute_reports,
            "include_full_song_context": bool(include_full_song_context),
            "mapper_token_contract": "v2.1_sparse_lane_actions",
        },
        init_from_control_checkpoint=init_from_control_checkpoint,
        resume_from=resume_from,
        progress_label="mapper_v2_1_phase_b",
        skip_first_eval_pass=skip_first_eval_pass,
        mps_cleanup_every=mps_cleanup_every,
    )


def _run_training(
    *,
    loader: DataLoader,
    train_eval_loader: DataLoader,
    eval_loader: DataLoader,
    output_dir: Path,
    model_config: MapperV21Config,
    control_model_config: ControlDemoGlobalEncoderConfig,
    loss_config: MapperV21LossConfig,
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
    resume_from: Path | None,
    progress_label: str,
    skip_first_eval_pass: bool = False,
    mps_cleanup_every: int | None = None,
) -> ControlTrainingResult:
    return run_mapper_training(
        loader=loader,
        train_eval_loader=train_eval_loader,
        eval_loader=eval_loader,
        output_dir=output_dir,
        spec=_mapper_v2_1_training_spec(
            model_config=model_config,
            control_model_config=control_model_config,
            loss_config=loss_config,
            progress_label=progress_label,
        ),
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
        dataset_report=dataset_report,
        init_from_control_checkpoint=init_from_control_checkpoint,
        resume_from=resume_from,
        skip_first_eval_pass=skip_first_eval_pass,
        mps_cleanup_every=mps_cleanup_every,
    )


def _loss_for_raw_batch(
    model: MapperV21Model,
    loss_fn: MapperV21ModelLoss,
    raw_batch: Mapping[str, Any],
    *,
    device: torch.device,
):
    batch = _move_mapper_batch_tensors(raw_batch, device)
    output = model(batch)
    return loss_fn(output, batch)


def _mapper_v2_1_training_spec(
    *,
    model_config: MapperV21Config,
    control_model_config: ControlDemoGlobalEncoderConfig | None,
    loss_config: MapperV21LossConfig,
    progress_label: str,
) -> MapperTrainingSpec:
    return MapperTrainingSpec(
        model_config=model_config,
        control_model_config=control_model_config,
        loss_config=loss_config,
        model_factory=_mapper_v2_1_model_factory,
        loss_factory=_mapper_v2_1_loss_factory,
        batch_loss_adapter=_mapper_v2_1_batch_loss_adapter,
        optimizer_factory=lambda model, learning_rate, weight_decay: _build_mapper_tuple_optimizer(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        ),
        training_config_factory=lambda context: _mapper_v2_1_training_config(
            seed=context.seed,
            run_name=context.run_name,
            learning_rate=context.learning_rate,
            weight_decay=context.weight_decay,
            eval_every=context.eval_every,
            save_every=context.save_every,
            skip_first_eval_pass=context.skip_first_eval_pass,
            dataset_report=context.dataset_report,
            mps_cleanup_every=context.mps_cleanup_every,
        ),
        resume_checkpoint_loader=lambda path, context: _load_mapper_v2_1_resume_checkpoint(
            path,
            expected_model_config=context.expected_model_config,
            expected_control_model_config=context.expected_control_model_config,
            expected_loss_config=context.expected_loss_config,
            expected_training_config=context.expected_training_config,
        ),
        metric_count_predicate=lambda key: _metric_is_count(key) or key.endswith("_count"),
        metric_fallback_weight_key="token/valid_count",
        metric_empty={"loss/total": float("nan"), "loss/token": float("nan"), "loss/density": 0.0},
        metric_finalizer=default_mapper_metric_finalizer,
        progress_label=progress_label,
        metric_divider=lambda numerator, denominator: numerator / max(denominator, 1e-12),
        resume_loader_cursor=resume_resumable_loader_cursor_or_advance,
        cleanup_device_memory=_cleanup_mps_training_memory,
    )


def _mapper_v2_1_model_factory(
    model_config: Any,
    control_encoder: ControlDemoGlobalEncoder | None,
) -> MapperV21Model:
    if not isinstance(model_config, MapperV21Config):
        raise TypeError("mapper v2.1 training requires MapperV21Config")
    return MapperV21Model(model_config, control_encoder=control_encoder)


def _mapper_v2_1_loss_factory(model: torch.nn.Module, loss_config: Any) -> MapperV21ModelLoss:
    if not isinstance(model, MapperV21Model):
        raise TypeError("mapper v2.1 training requires MapperV21Model")
    if not isinstance(loss_config, MapperV21LossConfig):
        raise TypeError("mapper v2.1 training requires MapperV21LossConfig")
    return MapperV21ModelLoss(loss_config, vocab=model.vocab)


def _mapper_v2_1_batch_loss_adapter(
    model: torch.nn.Module,
    loss_fn: Any,
    raw_batch: Mapping[str, Any],
    device: torch.device,
):
    if not isinstance(model, MapperV21Model):
        raise TypeError("mapper v2.1 training requires MapperV21Model")
    if not isinstance(loss_fn, MapperV21ModelLoss):
        raise TypeError("mapper v2.1 training requires MapperV21ModelLoss")
    return _loss_for_raw_batch(model, loss_fn, raw_batch, device=device)


@torch.inference_mode()
def metrics_for_loader(
    model: MapperV21Model,
    loss_fn: MapperV21ModelLoss,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    spec = _mapper_v2_1_training_spec(
        model_config=getattr(model, "config", MapperV21Config()),
        control_model_config=None,
        loss_config=loss_fn.config,
        progress_label="mapper_v2_1_phase_b",
    )
    return mapper_metrics_for_loader(model, loss_fn, loader, device=device, spec=spec)


def _mapper_v2_1_resume_training_config(
    *,
    seed: int,
    run_name: str,
    learning_rate: float,
    weight_decay: float,
    eval_every: int,
    save_every: int,
    skip_first_eval_pass: bool,
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
        "mapper_token_contract": "v2.1_sparse_lane_actions",
        "dataset": _json_safe(_strict_mapper_v2_1_resume_dataset_report(dataset_report)),
    }


def _mapper_v2_1_training_config(
    *,
    seed: int,
    run_name: str,
    learning_rate: float,
    weight_decay: float,
    eval_every: int,
    save_every: int,
    skip_first_eval_pass: bool,
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
        "mapper_token_contract": "v2.1_sparse_lane_actions",
        "dataset": _json_safe(dataset_report),
    }


def _load_mapper_v2_1_resume_checkpoint(
    resume_from: Path,
    *,
    expected_model_config: MapperV21Config,
    expected_control_model_config: ControlDemoGlobalEncoderConfig | None,
    expected_loss_config: MapperV21LossConfig,
    expected_training_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    return load_mapper_training_resume_checkpoint(
        resume_from,
        expected_model_config=expected_model_config,
        expected_control_model_config=expected_control_model_config,
        expected_loss_config=expected_loss_config,
        expected_training_config=expected_training_config,
        normalize_training_config=_normalized_mapper_v2_1_resume_training_config,
        checkpoint_label="mapper v2.1 resume",
    )


def _normalized_mapper_v2_1_resume_training_config(config: object) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    normalized = {
        key: value
        for key, value in config.items()
        if key not in MAPPER_V2_1_RESUME_TRAINING_RUNTIME_KEYS
    }
    dataset = normalized.get("dataset")
    if isinstance(dataset, Mapping):
        normalized["dataset"] = _strict_mapper_v2_1_resume_dataset_report(dataset)
    return normalized


def _strict_mapper_v2_1_resume_dataset_report(dataset_report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in dataset_report.items()
        if key not in MAPPER_V2_1_RESUME_DATASET_RUNTIME_KEYS
    }


def _write_checkpoint_and_report(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    report_path: Path,
    model: MapperV21Model,
    optimizer: torch.optim.Optimizer,
    model_config: MapperV21Config,
    control_model_config: ControlDemoGlobalEncoderConfig,
    loss_config: MapperV21LossConfig,
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
    training_config = _mapper_v2_1_training_config(
        seed=seed,
        run_name=run_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        eval_every=eval_every,
        save_every=save_every,
        skip_first_eval_pass=skip_first_eval_pass,
        dataset_report=dataset_report,
        mps_cleanup_every=mps_cleanup_every,
    )
    write_mapper_checkpoint_and_report(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        report_path=report_path,
        model=model,
        optimizer=optimizer,
        spec=_mapper_v2_1_training_spec(
            model_config=model_config,
            control_model_config=control_model_config,
            loss_config=loss_config,
            progress_label="mapper_v2_1_phase_b",
        ),
        training_config=training_config,
        seed=seed,
        run_name=run_name,
        max_steps=max_steps,
        completed_steps=completed_steps,
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


def _normalized_section(source: object, *, allowed: set[str], name: str) -> dict[str, Any]:
    if source is None:
        return {}
    if not isinstance(source, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = sorted(set(source) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} keys: {unknown}")
    return dict(source)


def main(argv: Sequence[str] | None = None) -> None:
    from pulsefield_model.training.mapper_training_hydra import run_mapper_preset_cli

    run_mapper_preset_cli(
        argv,
        mapper_preset="v2_1_sparse_d384_l4_phase_b",
        v21_runner=run_mapper_v2_1_phase_b_training,
        precompute_runner=precompute_mapper_tuple_phase_b_control_teacher_cache,
    )


if __name__ == "__main__":
    main()
