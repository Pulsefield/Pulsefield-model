from __future__ import annotations

import argparse
import math
import time
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
    CHECKPOINT_SCHEMA_VERSION,
    DEFAULT_FINAL_TRAIN_EVAL_SIZE,
    ControlTrainingResult,
    _atomic_torch_save,
    _capture_rng_state,
    _infinite_loader,
    _json_metrics,
    _json_safe,
    _metric_is_count,
    _set_deterministic_seed,
    _validate_training_args,
    _write_report,
    limit_final_train_eval_dataset,
    select_torch_device,
    split_train_eval_dataset,
)
from pulsefield_model.training.control_demo_global import initialize_global_control_demo_from_control_checkpoint
from pulsefield_model.training.mapper_common import (
    _build_mapper_tuple_optimizer,
    _cleanup_mps_training_memory,
    _move_mapper_batch_tensors,
    precompute_mapper_tuple_phase_b_control_teacher_cache,
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

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
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
    progress_label: str,
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
    _set_deterministic_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_torch_device(device_name)
    save_every = eval_every if save_every is None else save_every
    checkpoint_path = output_dir / "checkpoint.pt"
    report_path = output_dir / "report.json"

    initialization_report: dict[str, Any] | None = None
    control_encoder = ControlDemoGlobalEncoder(control_model_config)
    if init_from_control_checkpoint is not None:
        initialization_report = initialize_global_control_demo_from_control_checkpoint(
            control_encoder,
            init_from_control_checkpoint,
        )
    model = MapperV21Model(model_config, control_encoder=control_encoder).to(device)
    loss_fn = MapperV21ModelLoss(loss_config, vocab=model.vocab).to(device)
    optimizer = _build_mapper_tuple_optimizer(model, learning_rate=learning_rate, weight_decay=weight_decay)
    iterator = _infinite_loader(loader)
    history: list[dict[str, Any]] = []
    last_train_metrics: dict[str, float] = {}
    final_train_metrics: dict[str, float] = {}
    final_eval_metrics: dict[str, float] = {}

    log_start_time = time.monotonic()
    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_output = _loss_for_raw_batch(model, loss_fn, next(iterator), device=device)
        loss_output.total_loss.backward()
        torch.nn.utils.clip_grad_norm_((parameter for parameter in model.parameters() if parameter.requires_grad), 1.0)
        optimizer.step()
        last_train_metrics = dict(loss_output.metrics)
        del loss_output
        if mps_cleanup_every is not None and step % mps_cleanup_every == 0:
            _cleanup_mps_training_memory(device)

        should_eval = step == 1 or step % eval_every == 0 or step == max_steps
        if skip_first_eval_pass and step == 1 and step != max_steps:
            should_eval = False
        should_save = step == 1 or step % save_every == 0 or step == max_steps
        should_log = log_every is not None and (step == 1 or step % log_every == 0 or step == max_steps)
        if should_log or should_eval or should_save:
            elapsed_s = time.monotonic() - log_start_time
            steps_per_s = step / max(elapsed_s, 1e-9)
            print(
                f"{progress_label}_progress step={step}/{max_steps} "
                f"loss={last_train_metrics['loss/total']:.6f} "
                f"elapsed_s={elapsed_s:.1f} steps_per_s={steps_per_s:.3f}",
                flush=True,
            )
        if should_eval:
            final_eval_metrics = metrics_for_loader(model, loss_fn, eval_loader, device=device)
            history_entry: dict[str, Any] = {
                "step": step,
                "train": _json_metrics(last_train_metrics),
                "eval": _json_metrics(final_eval_metrics),
            }
            if step == max_steps:
                final_train_metrics = metrics_for_loader(model, loss_fn, train_eval_loader, device=device)
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
                completed_steps=step,
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
            )
            _cleanup_mps_training_memory(device)

    result_metrics = final_eval_metrics or last_train_metrics
    return ControlTrainingResult(
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        final_loss=float(result_metrics.get("loss/total", float("nan"))),
        final_value_loss=float(result_metrics.get("loss/token", float("nan"))),
        final_confidence_loss=0.0,
        completed_steps=max_steps,
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


@torch.inference_mode()
def metrics_for_loader(
    model: MapperV21Model,
    loss_fn: MapperV21ModelLoss,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    count_totals: dict[str, float] = {}
    mean_numerators: dict[str, float] = {}
    mean_denominators: dict[str, float] = {}
    fallback_totals: dict[str, float] = {}
    fallback_weights: dict[str, float] = {}
    for raw_batch in loader:
        loss_output = _loss_for_raw_batch(model, loss_fn, raw_batch, device=device)
        for key, value in loss_output.metrics.items():
            if _metric_is_count(key) or key.endswith("_count"):
                count_totals[key] = count_totals.get(key, 0.0) + float(value)
        for key, numerator in loss_output.metric_numerators.items():
            mean_numerators[key] = mean_numerators.get(key, 0.0) + float(numerator)
        for key, denominator in loss_output.metric_denominators.items():
            mean_denominators[key] = mean_denominators.get(key, 0.0) + float(denominator)
        unresolved = set(loss_output.metrics) - set(loss_output.metric_numerators) - set(count_totals)
        weight = max(float(loss_output.metrics.get("token/valid_count", 0.0)), 1.0)
        for key in unresolved:
            fallback_totals[key] = fallback_totals.get(key, 0.0) + float(loss_output.metrics[key]) * weight
            fallback_weights[key] = fallback_weights.get(key, 0.0) + weight
        del loss_output
    if not (count_totals or mean_numerators or fallback_totals):
        return {"loss/total": math.nan, "loss/token": math.nan, "loss/density": 0.0}
    metrics = dict(count_totals)
    for key, numerator in mean_numerators.items():
        metrics[key] = numerator / max(mean_denominators.get(key, 0.0), 1e-12)
    for key, total in fallback_totals.items():
        metrics[key] = total / max(fallback_weights[key], 1e-12)
    metrics["loss/total"] = (
        metrics.get("loss/token", 0.0)
        + loss_fn.config.lambda_ln_close * metrics.get("loss/ln_close", 0.0)
        + loss_fn.config.lambda_density * metrics.get("loss/density", 0.0)
        + loss_fn.config.lambda_adapter_reg * metrics.get("loss/adapter_reg", 0.0)
    )
    return metrics


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
        "mapper_token_contract": "v2.1_sparse_lane_actions",
        "dataset": _json_safe(dataset_report),
    }
    checkpoint_payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "control_model_config": asdict(control_model_config),
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
            "save_every": save_every,
            "log_every": log_every,
            "skip_first_eval_pass": bool(skip_first_eval_pass),
            "mps_cleanup_every": mps_cleanup_every,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "device": str(device),
            "last_train_metrics": _json_metrics(last_train_metrics),
            "final_train_metrics": _json_metrics(final_train_metrics),
            "final_eval_metrics": _json_metrics(final_eval_metrics),
            "rng_state": _capture_rng_state(),
        },
    }
    archive_path = output_dir / "checkpoints" / f"checkpoint_step_{completed_steps:06d}.pt"
    _atomic_torch_save(checkpoint_payload, archive_path)
    _atomic_torch_save(checkpoint_payload, checkpoint_path)
    report_payload = {
        "run_name": run_name,
        "phase": "B",
        "seed": seed,
        "max_steps": max_steps,
        "completed_steps": completed_steps,
        "is_complete": completed_steps >= max_steps,
        "eval_every": eval_every,
        "save_every": save_every,
        "log_every": log_every,
        "skip_first_eval_pass": bool(skip_first_eval_pass),
        "mps_cleanup_every": mps_cleanup_every,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "model_config": asdict(model_config),
        "control_model_config": asdict(control_model_config),
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
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args(argv)
    config_defaults = load_run_config(config_args.config) if config_args.config is not None else {
        "model": {},
        "control_model": {},
        "loss": {},
    }
    model_defaults = config_defaults["model"]
    control_model_defaults = config_defaults["control_model"]
    loss_defaults = config_defaults["loss"]

    parser = argparse.ArgumentParser(description="Train the Stage 2 mapper v2.1 sparse Phase B model.")
    parser.add_argument("--config", default=config_args.config)
    parser.add_argument("--dataset-root", default=config_defaults.get("dataset_root", "dataset"))
    parser.add_argument("--index-path", default=config_defaults.get("index_path"))
    parser.add_argument("--eval-index-path", default=config_defaults.get("eval_index_path"))
    parser.add_argument("--control-v3-timeseries-path", default=config_defaults.get("control_v3_timeseries_path"))
    parser.add_argument("--output-dir", default=config_defaults.get("output_dir"))
    parser.add_argument("--max-steps", type=int, default=config_defaults.get("max_steps"))
    parser.add_argument("--eval-every", type=int, default=config_defaults.get("eval_every"))
    parser.add_argument("--save-every", type=int, default=config_defaults.get("save_every"))
    parser.add_argument("--log-every", type=int, default=config_defaults.get("log_every"))
    parser.add_argument("--batch-size", type=int, default=config_defaults.get("batch_size", 2))
    parser.add_argument("--learning-rate", type=float, default=config_defaults.get("learning_rate", 2e-4))
    parser.add_argument("--weight-decay", type=float, default=config_defaults.get("weight_decay", 0.01))
    parser.add_argument("--seed", type=int, default=config_defaults.get("seed", 1337))
    parser.add_argument("--device", default=config_defaults.get("device", "auto"), choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--run-name", default=config_defaults.get("run_name", "mapper_v2_1_phase_b_sparse_global"))
    parser.add_argument("--init-from-control-checkpoint", default=config_defaults.get("init_from_control_checkpoint"))
    parser.add_argument("--eval-fraction", type=float, default=config_defaults.get("eval_fraction", 0.1))
    parser.add_argument("--eval-size", type=int, default=config_defaults.get("eval_size"))
    parser.add_argument(
        "--final-train-eval-size",
        type=int,
        default=config_defaults.get("final_train_eval_size", DEFAULT_FINAL_TRAIN_EVAL_SIZE),
    )
    parser.add_argument("--num-workers", type=int, default=config_defaults.get("num_workers", 0))
    parser.add_argument("--max-cached-maps", type=int, default=config_defaults.get("max_cached_maps"))
    parser.add_argument(
        "--dataset-progress",
        action=argparse.BooleanOptionalAction,
        default=bool(config_defaults.get("dataset_progress", False)),
    )
    parser.add_argument("--mapper-record-cache-path", default=config_defaults.get("mapper_record_cache_path"))
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
        "--control-teacher-cache-overwrite",
        action="store_true",
        default=bool(config_defaults.get("control_teacher_cache_overwrite", False)),
    )
    parser.add_argument(
        "--require-control-teacher-cache",
        action="store_true",
        default=bool(config_defaults.get("require_control_teacher_cache", False)),
    )
    parser.add_argument(
        "--include-full-song-context",
        action=argparse.BooleanOptionalAction,
        default=bool(config_defaults.get("include_full_song_context", True)),
    )
    parser.add_argument(
        "--skip-first-eval-pass",
        action=argparse.BooleanOptionalAction,
        default=bool(config_defaults.get("skip_first_eval_pass", False)),
    )
    parser.add_argument("--mps-cleanup-every", type=int, default=config_defaults.get("mps_cleanup_every"))
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR.as_posix()
    if args.max_steps is None:
        args.max_steps = 5000
    if args.eval_every is None:
        args.eval_every = 100

    init_from = Path(args.init_from_control_checkpoint) if args.init_from_control_checkpoint is not None else None
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

    result = run_mapper_v2_1_phase_b_training(
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
        eval_fraction=args.eval_fraction,
        eval_size=args.eval_size,
        final_train_eval_size=args.final_train_eval_size,
        num_workers=args.num_workers,
        max_cached_maps=args.max_cached_maps,
        dataset_progress=args.dataset_progress,
        mapper_record_cache_path=Path(args.mapper_record_cache_path) if args.mapper_record_cache_path is not None else None,
        control_teacher_cache_dir=Path(args.control_teacher_cache_dir) if args.control_teacher_cache_dir is not None else None,
        require_control_teacher_cache=args.require_control_teacher_cache,
        precompute_control_teacher_cache=args.precompute_control_teacher_cache,
        control_teacher_precompute_batch_size=args.control_teacher_precompute_batch_size,
        control_teacher_cache_overwrite=args.control_teacher_cache_overwrite,
        include_full_song_context=args.include_full_song_context,
        skip_first_eval_pass=args.skip_first_eval_pass,
        mps_cleanup_every=args.mps_cleanup_every,
        model_config_overrides=model_defaults,
        control_model_config_overrides=control_model_defaults,
        loss_config_overrides=loss_defaults,
    )
    print(
        "mapper_v2_1_training_done "
        f"steps={result.completed_steps} final_loss={result.final_loss:.6f} "
        f"report={result.report_path.as_posix()} checkpoint={result.checkpoint_path.as_posix()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
