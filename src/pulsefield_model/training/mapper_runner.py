from __future__ import annotations

import pickle
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, MutableMapping

import torch
from torch import nn
from torch.utils.data import DataLoader

from pulsefield_model.models.control import ControlDemoGlobalEncoder
from pulsefield_model.training.common import (
    CHECKPOINT_SCHEMA_VERSION,
    ControlTrainingResult,
    _advance_training_iterator,
    _atomic_torch_save,
    _capture_rng_state,
    _copy_file_atomically,
    _infinite_loader,
    _json_metrics,
    _move_optimizer_state_to_device,
    _restore_rng_state,
    _safe_float_div,
    _set_deterministic_seed,
    _validate_training_args,
    _write_report,
    select_torch_device,
    set_resumable_loader_batch_cursor,
)
from pulsefield_model.training.control_demo_global import initialize_global_control_demo_from_control_checkpoint


BatchLossAdapter = Callable[[nn.Module, Any, Mapping[str, Any], torch.device], Any]
MetricCountPredicate = Callable[[str], bool]
MetricDivider = Callable[[float, float], float]
MetricFinalizer = Callable[[dict[str, float], Any], dict[str, float]]
ModelFactory = Callable[[Any, ControlDemoGlobalEncoder | None], nn.Module]
LossFactory = Callable[[nn.Module, Any], Any]
OptimizerFactory = Callable[[nn.Module, float, float], torch.optim.Optimizer]
TrainingConfigFactory = Callable[["MapperTrainingConfigContext"], Mapping[str, Any]]
ResumeCheckpointLoader = Callable[[Path, "MapperTrainingResumeContext"], Mapping[str, Any]]
LoaderCursorHook = Callable[[DataLoader, Iterator[Any], int], Iterator[Any]]
MapperCheckpointInitializer = Callable[..., Mapping[str, Any]]
ControlCheckpointInitializer = Callable[[ControlDemoGlobalEncoder, Path], Mapping[str, Any]]
PayloadHook = Callable[[MutableMapping[str, Any], Mapping[str, Any]], None]
DeviceCleanup = Callable[[torch.device], None]


def resume_resumable_loader_cursor_or_advance(
    loader: DataLoader,
    iterator: Iterator[Any],
    completed_step: int,
) -> Iterator[Any]:
    if set_resumable_loader_batch_cursor(loader, completed_step):
        return iterator
    return _advance_training_iterator(iterator, completed_step)


def _advance_loader_cursor(
    loader: DataLoader,
    iterator: Iterator[Any],
    completed_step: int,
) -> Iterator[Any]:
    del loader
    return _advance_training_iterator(iterator, completed_step)


@dataclass(frozen=True)
class MapperTrainingConfigContext:
    seed: int
    run_name: str
    learning_rate: float
    weight_decay: float
    eval_every: int
    save_every: int
    skip_first_eval_pass: bool
    dataset_report: Mapping[str, Any]
    mps_cleanup_every: int | None


@dataclass(frozen=True)
class MapperTrainingResumeContext:
    expected_model_config: Any
    expected_control_model_config: Any | None
    expected_loss_config: Any
    expected_training_config: Mapping[str, Any]


@dataclass(frozen=True)
class MapperTrainingSpec:
    model_config: Any
    control_model_config: Any | None
    loss_config: Any
    model_factory: ModelFactory
    loss_factory: LossFactory
    batch_loss_adapter: BatchLossAdapter
    optimizer_factory: OptimizerFactory
    training_config_factory: TrainingConfigFactory
    resume_checkpoint_loader: ResumeCheckpointLoader
    metric_count_predicate: MetricCountPredicate
    metric_fallback_weight_key: str
    metric_empty: Mapping[str, float]
    metric_finalizer: MetricFinalizer
    progress_label: str
    metric_divider: MetricDivider = _safe_float_div
    phase: str = "B"
    mapper_checkpoint_initializer: MapperCheckpointInitializer | None = None
    control_checkpoint_initializer: ControlCheckpointInitializer = initialize_global_control_demo_from_control_checkpoint
    resume_loader_cursor: LoaderCursorHook = _advance_loader_cursor
    checkpoint_payload_hook: PayloadHook | None = None
    report_payload_hook: PayloadHook | None = None
    cleanup_device_memory: DeviceCleanup | None = None


def run_mapper_training(
    *,
    loader: DataLoader,
    train_eval_loader: DataLoader,
    eval_loader: DataLoader,
    output_dir: Path,
    spec: MapperTrainingSpec,
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
    init_from_control_checkpoint: Path | None = None,
    init_from_mapper_checkpoint: Path | None = None,
    resume_from: Path | None = None,
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
    mps_cleanup_every = _normalized_mps_cleanup_every(mps_cleanup_every)
    if resume_from is not None and init_from_mapper_checkpoint is not None:
        raise ValueError("resume_from and init_from_mapper_checkpoint cannot both be set")

    _set_deterministic_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_torch_device(device_name)
    save_every = eval_every if save_every is None else save_every
    checkpoint_path = output_dir / "checkpoint.pt"
    report_path = output_dir / "report.json"

    model, loss_adapter, initialization_report = _build_mapper_training_components(
        spec=spec,
        device=device,
        init_from_control_checkpoint=init_from_control_checkpoint,
        init_from_mapper_checkpoint=init_from_mapper_checkpoint,
        resume_from=resume_from,
    )
    optimizer = spec.optimizer_factory(model, learning_rate, weight_decay)
    training_config = dict(
        spec.training_config_factory(
            MapperTrainingConfigContext(
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
        )
    )

    iterator = _infinite_loader(loader)
    history: list[dict[str, Any]] = []
    completed_step = 0
    last_train_metrics: dict[str, float] = {}
    final_train_metrics: dict[str, float] = {}
    final_eval_metrics: dict[str, float] = {}

    if resume_from is not None:
        checkpoint = spec.resume_checkpoint_loader(
            resume_from,
            MapperTrainingResumeContext(
                expected_model_config=spec.model_config,
                expected_control_model_config=spec.control_model_config,
                expected_loss_config=spec.loss_config,
                expected_training_config=training_config,
            ),
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
        iterator = spec.resume_loader_cursor(loader, iterator, completed_step)
        print(f"{spec.progress_label}_resume checkpoint={resume_from} step={completed_step}/{max_steps}", flush=True)

    log_start_time = time.monotonic()
    log_start_step = completed_step
    for step in range(completed_step + 1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_output = spec.batch_loss_adapter(model, loss_adapter, next(iterator), device)
        loss_output.total_loss.backward()
        torch.nn.utils.clip_grad_norm_((parameter for parameter in model.parameters() if parameter.requires_grad), 1.0)
        optimizer.step()
        last_train_metrics = dict(loss_output.metrics)
        del loss_output
        completed_step = step
        _cleanup_if_requested(spec, device=device, step=step, mps_cleanup_every=mps_cleanup_every)

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
                f"{spec.progress_label}_progress step={step}/{max_steps} "
                f"loss={last_train_metrics['loss/total']:.6f} "
                f"elapsed_s={elapsed_s:.1f} steps_per_s={steps_per_s:.3f}",
                flush=True,
            )
        if should_eval:
            final_eval_metrics = mapper_metrics_for_loader(
                model,
                loss_adapter,
                eval_loader,
                device=device,
                spec=spec,
            )
            history_entry: dict[str, Any] = {
                "step": step,
                "train": _json_metrics(last_train_metrics),
                "eval": _json_metrics(final_eval_metrics),
            }
            if step == max_steps:
                final_train_metrics = mapper_metrics_for_loader(
                    model,
                    loss_adapter,
                    train_eval_loader,
                    device=device,
                    spec=spec,
                )
                history_entry["train_eval"] = _json_metrics(final_train_metrics)
            history.append(history_entry)
            print(
                f"{spec.progress_label}_eval step={step}/{max_steps} "
                f"loss={final_eval_metrics.get('loss/total', float('nan')):.6f}",
                flush=True,
            )
            _cleanup_device(spec, device)
        if should_save:
            write_mapper_checkpoint_and_report(
                output_dir=output_dir,
                checkpoint_path=checkpoint_path,
                report_path=report_path,
                model=model,
                optimizer=optimizer,
                spec=spec,
                training_config=training_config,
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
            _cleanup_device(spec, device)

    result_metrics = final_eval_metrics or last_train_metrics
    return ControlTrainingResult(
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        final_loss=float(result_metrics.get("loss/total", float("nan"))),
        final_value_loss=float(result_metrics.get("loss/token", float("nan"))),
        final_confidence_loss=0.0,
        completed_steps=completed_step,
    )


@torch.inference_mode()
def mapper_metrics_for_loader(
    model: nn.Module,
    loss_adapter: Any,
    loader: DataLoader,
    *,
    device: torch.device,
    spec: MapperTrainingSpec,
) -> dict[str, float]:
    model.eval()
    count_totals: dict[str, float] = {}
    mean_numerators: dict[str, float] = {}
    mean_denominators: dict[str, float] = {}
    fallback_totals: dict[str, float] = {}
    fallback_weights: dict[str, float] = {}
    for raw_batch in loader:
        loss_output = spec.batch_loss_adapter(model, loss_adapter, raw_batch, device)
        for key, value in loss_output.metrics.items():
            if spec.metric_count_predicate(key):
                count_totals[key] = count_totals.get(key, 0.0) + float(value)
        for key, numerator in loss_output.metric_numerators.items():
            mean_numerators[key] = mean_numerators.get(key, 0.0) + float(numerator)
        for key, denominator in loss_output.metric_denominators.items():
            mean_denominators[key] = mean_denominators.get(key, 0.0) + float(denominator)
        unresolved = set(loss_output.metrics) - set(loss_output.metric_numerators) - set(count_totals)
        weight = max(float(loss_output.metrics.get(spec.metric_fallback_weight_key, 0.0)), 1.0)
        for key in unresolved:
            fallback_totals[key] = fallback_totals.get(key, 0.0) + float(loss_output.metrics[key]) * weight
            fallback_weights[key] = fallback_weights.get(key, 0.0) + weight
        del loss_output
    if not (count_totals or mean_numerators or fallback_totals):
        return dict(spec.metric_empty)
    metrics = dict(count_totals)
    for key, numerator in mean_numerators.items():
        metrics[key] = spec.metric_divider(numerator, mean_denominators.get(key, 0.0))
    for key, total in fallback_totals.items():
        metrics[key] = spec.metric_divider(total, fallback_weights[key])
    return spec.metric_finalizer(metrics, loss_adapter)


def default_mapper_metric_finalizer(metrics: dict[str, float], loss_adapter: Any) -> dict[str, float]:
    config = getattr(loss_adapter, "config", loss_adapter)
    metrics["loss/total"] = (
        metrics.get("loss/token", 0.0)
        + float(getattr(config, "lambda_ln_close", 0.0)) * metrics.get("loss/ln_close", 0.0)
        + float(getattr(config, "lambda_adapter_reg", 0.0)) * metrics.get("loss/adapter_reg", 0.0)
        + float(getattr(config, "lambda_density", 0.0)) * metrics.get("loss/density", 0.0)
    )
    return metrics


def write_mapper_checkpoint_and_report(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    report_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    spec: MapperTrainingSpec,
    training_config: Mapping[str, Any],
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
    checkpoint_payload: dict[str, Any] = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": _config_payload(spec.model_config),
        "control_model_config": _config_payload(spec.control_model_config),
        "loss_config": _config_payload(spec.loss_config),
        "training_config": dict(training_config),
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
    context = _payload_context(
        spec=spec,
        model=model,
        optimizer=optimizer,
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
    if spec.checkpoint_payload_hook is not None:
        spec.checkpoint_payload_hook(checkpoint_payload, context)
    archive_path = output_dir / "checkpoints" / f"checkpoint_step_{completed_steps:06d}.pt"
    _atomic_torch_save(checkpoint_payload, archive_path)
    _copy_file_atomically(archive_path, checkpoint_path)

    report_payload: dict[str, Any] = {
        "run_name": run_name,
        "phase": spec.phase,
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
        "model_config": _config_payload(spec.model_config),
        "control_model_config": _config_payload(spec.control_model_config),
        "loss_config": _config_payload(spec.loss_config),
        "training_config": dict(training_config),
        "device": str(device),
        "parameter_count": _parameter_count(model),
        "dataset": dict(dataset_report),
        "initialization": None if initialization_report is None else dict(initialization_report),
        "history": history,
        "last_train_metrics": _json_metrics(last_train_metrics),
        "final_train_metrics": _json_metrics(final_train_metrics),
        "final_eval_metrics": _json_metrics(final_eval_metrics),
    }
    if spec.report_payload_hook is not None:
        spec.report_payload_hook(report_payload, context)
    _write_report(report_path, report_payload)


def load_mapper_training_resume_checkpoint(
    resume_from: Path,
    *,
    expected_model_config: Any,
    expected_control_model_config: Any | None,
    expected_loss_config: Any,
    expected_training_config: Mapping[str, Any],
    normalize_training_config: Callable[[object], dict[str, Any]],
    checkpoint_label: str,
) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError(
            f"{checkpoint_label} checkpoint could not be loaded safely with weights_only=True; "
            f"use a checkpoint written by the {checkpoint_label.replace(' resume', '')} trainer"
        ) from exc
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{checkpoint_label} checkpoint must contain a mapping: {resume_from}")
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"{checkpoint_label} checkpoint schema version mismatch")
    if checkpoint.get("model_config") != _config_payload(expected_model_config):
        raise ValueError(f"{checkpoint_label} checkpoint model_config does not match the requested run")
    if checkpoint.get("control_model_config") != _config_payload(expected_control_model_config):
        raise ValueError(f"{checkpoint_label} checkpoint control_model_config does not match the requested run")
    if checkpoint.get("loss_config") != _config_payload(expected_loss_config):
        raise ValueError(f"{checkpoint_label} checkpoint loss_config does not match the requested run")
    if normalize_training_config(checkpoint.get("training_config")) != normalize_training_config(expected_training_config):
        raise ValueError(f"{checkpoint_label} checkpoint training_config does not match the requested run")
    if not isinstance(checkpoint.get("model_state_dict"), Mapping):
        raise ValueError(f"{checkpoint_label} checkpoint missing model_state_dict")
    if "optimizer_state_dict" not in checkpoint:
        raise ValueError(f"{checkpoint_label} checkpoint missing optimizer_state_dict")
    if not isinstance(checkpoint.get("training_state"), Mapping):
        raise ValueError(f"{checkpoint_label} checkpoint missing training_state")
    if not isinstance(checkpoint.get("history"), list):
        raise ValueError(f"{checkpoint_label} checkpoint history must be a list")
    training_state = checkpoint["training_state"]
    if not isinstance(training_state.get("step"), int) or training_state["step"] < 0:
        raise ValueError(f"{checkpoint_label} checkpoint training_state.step must be a non-negative integer")
    if "rng_state" not in training_state:
        raise ValueError(f"{checkpoint_label} checkpoint missing training_state.rng_state")
    return checkpoint


def _build_mapper_training_components(
    *,
    spec: MapperTrainingSpec,
    device: torch.device,
    init_from_control_checkpoint: Path | None,
    init_from_mapper_checkpoint: Path | None,
    resume_from: Path | None,
) -> tuple[nn.Module, Any, dict[str, Any] | None]:
    control_encoder: ControlDemoGlobalEncoder | None = None
    initialization_report: dict[str, Any] | None = None
    if spec.control_model_config is not None:
        control_encoder = ControlDemoGlobalEncoder(spec.control_model_config)
        if init_from_control_checkpoint is not None and init_from_mapper_checkpoint is None and resume_from is None:
            initialization_report = dict(spec.control_checkpoint_initializer(control_encoder, init_from_control_checkpoint))

    model = spec.model_factory(spec.model_config, control_encoder)
    if init_from_mapper_checkpoint is not None:
        if spec.mapper_checkpoint_initializer is None:
            raise ValueError("init_from_mapper_checkpoint is not supported by this mapper trainer")
        initialization_report = dict(
            spec.mapper_checkpoint_initializer(
                model,
                init_from_mapper_checkpoint,
                expected_model_config=spec.model_config,
                expected_control_model_config=spec.control_model_config,
            )
        )
    model = model.to(device)
    loss_adapter = spec.loss_factory(model, spec.loss_config)
    if isinstance(loss_adapter, nn.Module):
        loss_adapter = loss_adapter.to(device)
    return model, loss_adapter, initialization_report


def _normalized_mps_cleanup_every(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("mps_cleanup_every must be an integer step interval")
    normalized = int(value)
    if normalized < 0:
        raise ValueError("mps_cleanup_every must be non-negative")
    if normalized == 0:
        return None
    return normalized


def _cleanup_if_requested(
    spec: MapperTrainingSpec,
    *,
    device: torch.device,
    step: int,
    mps_cleanup_every: int | None,
) -> None:
    if mps_cleanup_every is not None and step % mps_cleanup_every == 0:
        _cleanup_device(spec, device)


def _cleanup_device(spec: MapperTrainingSpec, device: torch.device) -> None:
    if spec.cleanup_device_memory is not None:
        spec.cleanup_device_memory(device)


def _config_payload(config: Any) -> Any:
    if config is None:
        return None
    if is_dataclass(config) and not isinstance(config, type):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    return config


def _parameter_count(model: nn.Module) -> int:
    parameter_count = getattr(model, "parameter_count", None)
    if callable(parameter_count):
        return int(parameter_count())
    return sum(parameter.numel() for parameter in model.parameters())


def _payload_context(**kwargs: Any) -> dict[str, Any]:
    return dict(kwargs)
