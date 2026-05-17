from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import shutil
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

from pulsefield_model.data.control_windows import (
    ControlWindowDataset,
    DEFAULT_MAX_CACHED_MAPS,
    collate_control_context_windows,
)
from pulsefield_model.features.control_v3_targets import CONFIDENCE_FEATURE_NAMES, MODEL_FEATURE_NAMES, VALUE_FEATURE_NAMES
from pulsefield_model.models.control import (
    ControlEncoder,
    ControlEncoderConfig,
    ControlLossConfig,
    ControlModelLoss,
    prepare_control_context_batch,
)


CHECKPOINT_SCHEMA_VERSION = 1
DEFAULT_RUNS_ROOT = Path("artifacts/runs/stage2_control")
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_ROOT / "control_encoder"
DEFAULT_FINAL_TRAIN_EVAL_SIZE = 1024
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
    "resume_from",
    "eval_fraction",
    "eval_size",
    "final_train_eval_size",
    "num_workers",
    "max_cached_maps",
    "model",
    "loss",
}
MODEL_CONFIG_KEYS = {field.name for field in fields(ControlEncoderConfig)}
LOSS_CONFIG_KEYS = {field.name for field in fields(ControlLossConfig)}
LOSS_BATCH_TENSOR_KEYS = frozenset(
    (
        "context_mel",
        "context_dense_timing_v2",
        "normalized_difficulty",
        "context_padding_mask",
        "control_v3_target",
        "target_valid_mask",
        "ln_change_n_eff_target",
    )
)
RESUME_DATASET_RUNTIME_KEYS = frozenset(("max_cached_maps", "num_workers"))


@dataclass(frozen=True)
class ControlTrainingResult:
    report_path: Path
    checkpoint_path: Path
    final_loss: float
    final_value_loss: float
    final_confidence_loss: float
    completed_steps: int


def select_torch_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("requested cuda device is not available")
        return torch.device("cuda")
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("requested mps device is not available")
        return torch.device("mps")
    raise ValueError(f"unknown device: {device_name}")


def load_run_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML run config: {path}") from exc
    if loaded is None:
        return {"model": {}, "loss": {}}
    if not isinstance(loaded, dict):
        raise ValueError(f"run config must be a mapping: {path}")

    config = _normalize_config_mapping(loaded, source_name="run config")
    unknown = sorted(set(config) - RUN_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"unknown run config keys: {unknown}")
    config["model"] = _normalized_section(config.get("model", {}), allowed=MODEL_CONFIG_KEYS, name="model config")
    config["loss"] = _normalized_section(config.get("loss", {}), allowed=LOSS_CONFIG_KEYS, name="loss config")
    return config


def run_control_training(
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
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    seed: int = 1337,
    device_name: str = "auto",
    run_name: str = "control_encoder",
    resume_from: Path | None = None,
    eval_fraction: float = 0.1,
    eval_size: int | None = None,
    final_train_eval_size: int | None = DEFAULT_FINAL_TRAIN_EVAL_SIZE,
    num_workers: int = 0,
    max_cached_maps: int | None = None,
    model_config_overrides: Mapping[str, Any] | None = None,
    loss_config_overrides: Mapping[str, Any] | None = None,
) -> ControlTrainingResult:
    _set_deterministic_seed(seed)
    dataset_kwargs: dict[str, Any] = {
        "dataset_root": dataset_root,
    }
    if index_path is not None:
        dataset_kwargs["index_path"] = index_path
    if control_v3_timeseries_path is not None:
        dataset_kwargs["control_v3_timeseries_path"] = control_v3_timeseries_path
    effective_max_cached_maps = DEFAULT_MAX_CACHED_MAPS if max_cached_maps is None else max_cached_maps
    dataset_kwargs["max_cached_maps"] = effective_max_cached_maps
    train_source = ControlWindowDataset(**dataset_kwargs)
    if len(train_source) == 0:
        raise ValueError("ControlWindowDataset produced no training windows")

    if eval_index_path is not None:
        eval_kwargs = dict(dataset_kwargs)
        eval_kwargs["index_path"] = eval_index_path
        eval_dataset: Dataset[Any] = ControlWindowDataset(**eval_kwargs)
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

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        collate_fn=collate_control_context_windows,
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
        collate_fn=collate_control_context_windows,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_control_context_windows,
    )
    return _run_training(
        loader=loader,
        train_eval_loader=train_eval_loader,
        eval_loader=eval_loader,
        output_dir=output_dir,
        model_config=ControlEncoderConfig(**dict(model_config_overrides or {})),
        loss_config=ControlLossConfig(**dict(loss_config_overrides or {})),
        max_steps=max_steps,
        eval_every=eval_every,
        save_every=save_every,
        log_every=log_every,
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
            "max_cached_maps": int(getattr(train_source, "max_cached_maps", effective_max_cached_maps)),
            "num_workers": num_workers,
        },
        resume_from=resume_from,
    )


def split_train_eval_dataset(
    dataset: Dataset[Any],
    *,
    eval_fraction: float,
    eval_size: int | None,
    seed: int,
) -> tuple[Dataset[Any], Dataset[Any]]:
    _require_finite_number(float(eval_fraction), "eval_fraction")
    if not 0.0 <= eval_fraction < 1.0:
        raise ValueError(f"eval_fraction must be within [0, 1), got {eval_fraction}")
    count = len(dataset)
    if count <= 1:
        return dataset, Subset(dataset, [])
    if eval_size is None:
        resolved_eval_size = int(round(count * eval_fraction))
        if eval_fraction > 0.0:
            resolved_eval_size = max(1, resolved_eval_size)
    else:
        resolved_eval_size = int(eval_size)
    if resolved_eval_size < 0:
        raise ValueError(f"eval_size must be non-negative, got {eval_size}")
    resolved_eval_size = min(resolved_eval_size, count - 1)
    grouped_indices = _map_identity_index_groups(dataset, count)
    if grouped_indices is None:
        train_indices, eval_indices = _split_indices_by_window(count, resolved_eval_size, seed)
    else:
        train_indices, eval_indices = _split_indices_by_map_group(
            grouped_indices,
            count=count,
            eval_size=resolved_eval_size,
            seed=seed,
        )
    return Subset(dataset, train_indices), Subset(dataset, eval_indices)


def limit_final_train_eval_dataset(
    dataset: Dataset[Any],
    *,
    final_train_eval_size: int | None,
    seed: int,
) -> Dataset[Any]:
    if final_train_eval_size is None:
        return dataset
    resolved_size = int(final_train_eval_size)
    if resolved_size < 0:
        raise ValueError(f"final_train_eval_size must be non-negative, got {final_train_eval_size}")
    count = len(dataset)
    if resolved_size >= count:
        return dataset
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    return Subset(dataset, sorted(indices[:resolved_size]))


def _split_indices_by_window(count: int, eval_size: int, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    eval_indices = sorted(indices[:eval_size])
    train_indices = sorted(indices[eval_size:])
    return train_indices, eval_indices


def _split_indices_by_map_group(
    grouped_indices: list[list[int]],
    *,
    count: int,
    eval_size: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    if eval_size == 0 or len(grouped_indices) <= 1:
        return list(range(count)), []

    shuffled_groups = [list(group) for group in grouped_indices]
    random.Random(seed).shuffle(shuffled_groups)
    eval_indices: list[int] = []
    for group in shuffled_groups[:-1]:
        if len(eval_indices) >= eval_size:
            break
        eval_indices.extend(group)

    eval_index_set = set(eval_indices)
    train_indices = [index for index in range(count) if index not in eval_index_set]
    if not train_indices:
        return list(range(count)), []
    return sorted(train_indices), sorted(eval_indices)


def _map_identity_index_groups(dataset: Dataset[Any], count: int) -> list[list[int]] | None:
    groups: dict[tuple[tuple[str, object], ...], list[int]] = {}
    saw_map_identity = False
    for index in range(count):
        identity = _map_identity_for_dataset_index(dataset, index)
        if identity is None:
            identity = (("__dataset_index__", index),)
        else:
            saw_map_identity = True
        groups.setdefault(identity, []).append(index)
    if not saw_map_identity:
        return None
    return list(groups.values())


def _map_identity_for_dataset_index(dataset: Dataset[Any], index: int) -> tuple[tuple[str, object], ...] | None:
    if isinstance(dataset, Subset):
        return _map_identity_for_dataset_index(dataset.dataset, int(dataset.indices[index]))

    records = getattr(dataset, "records", None)
    if records is not None:
        try:
            identity = _map_identity_from_metadata(records[index])
        except (IndexError, KeyError, TypeError):
            identity = None
        if identity is not None:
            return identity

    if isinstance(dataset, (list, tuple)):
        return _map_identity_from_metadata(dataset[index])
    return None


def _map_identity_from_metadata(metadata: object) -> tuple[tuple[str, object], ...] | None:
    if isinstance(metadata, Mapping) and isinstance(metadata.get("metadata"), Mapping):
        nested_identity = _map_identity_from_metadata(metadata["metadata"])
        if nested_identity is not None:
            return nested_identity

    for field in ("beatmap_path", "map_path", "beatmap_id", "map_id", "filtered_index", "source_index"):
        value = _metadata_field_value(metadata, field)
        normalized = _normalize_map_identity_value(value, numeric=field not in {"beatmap_path", "map_path"})
        if normalized is not None:
            return ((field, normalized),)
    return None


def _metadata_field_value(metadata: object, field: str) -> object:
    if isinstance(metadata, Mapping):
        return metadata.get(field)
    return getattr(metadata, field, None)


def _normalize_map_identity_value(value: object, *, numeric: bool) -> object | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.item()
    if numeric:
        if isinstance(value, (bool, np.bool_)):
            return None
        if isinstance(value, (int, np.integer)):
            integer = int(value)
        elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
            integer = int(value)
        else:
            return None
        return integer if integer >= 0 else None
    text = value.as_posix() if isinstance(value, Path) else str(value)
    return text if text else None


def _run_training(
    *,
    loader: DataLoader,
    train_eval_loader: DataLoader,
    eval_loader: DataLoader,
    output_dir: Path,
    model_config: ControlEncoderConfig,
    loss_config: ControlLossConfig,
    max_steps: int,
    eval_every: int,
    save_every: int | None,
    log_every: int | None,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device_name: str,
    run_name: str,
    dataset_report: Mapping[str, Any],
    resume_from: Path | None,
) -> ControlTrainingResult:
    _validate_training_args(
        max_steps=max_steps,
        eval_every=eval_every,
        save_every=save_every,
        log_every=log_every,
        batch_size=getattr(loader, "batch_size", 1) or 1,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    _set_deterministic_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_torch_device(device_name)
    save_every = eval_every if save_every is None else save_every
    checkpoint_path = output_dir / "checkpoint.pt"
    report_path = output_dir / "report.json"

    model = ControlEncoder(model_config).to(device)
    loss_fn = ControlModelLoss(loss_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    resume_config = _resume_training_config(
        seed=seed,
        run_name=run_name,
        batch_size=getattr(loader, "batch_size", 1) or 1,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        eval_every=eval_every,
        save_every=save_every,
        dataset_report=dataset_report,
    )
    iterator = _infinite_loader(loader)
    history: list[dict[str, Any]] = []
    completed_step = 0
    last_train_metrics: dict[str, float] = {}
    final_train_metrics: dict[str, float] = {}
    final_eval_metrics: dict[str, float] = {}

    if resume_from is not None:
        checkpoint = _load_resume_checkpoint(
            resume_from,
            expected_model_config=model_config,
            expected_loss_config=loss_config,
            expected_training_config=resume_config,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _move_optimizer_state_to_device(optimizer, device)
        training_state = checkpoint["training_state"]
        completed_step = int(training_state["step"])
        history = [dict(entry) for entry in checkpoint["history"]]
        last_train_metrics = dict(training_state.get("last_train_metrics", {}))
        final_train_metrics = dict(training_state.get("final_train_metrics", {}))
        final_eval_metrics = dict(training_state.get("final_eval_metrics", {}))
        _restore_rng_state(training_state["rng_state"])
        iterator = _advance_training_iterator(iterator, completed_step)
        print(f"resume_progress checkpoint={resume_from} step={completed_step}/{max_steps}", flush=True)

    log_start_time = time.monotonic()
    log_start_step = completed_step
    for step in range(completed_step + 1, max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_output = _loss_for_raw_batch(model, loss_fn, next(iterator), device=device)
        loss_output.total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_train_metrics = dict(loss_output.metrics)
        last_train_metrics["loss/total"] = float(loss_output.total_loss.detach().cpu())
        completed_step = step

        should_eval = step == 1 or step % eval_every == 0 or step == max_steps
        should_save = step == 1 or step % save_every == 0 or step == max_steps
        should_log = log_every is not None and (step == 1 or step % log_every == 0 or step == max_steps)
        if should_log or should_eval or should_save:
            elapsed_s = time.monotonic() - log_start_time
            completed_since_start = max(step - log_start_step, 1)
            steps_per_s = completed_since_start / max(elapsed_s, 1e-9)
            print(
                f"train_progress step={step}/{max_steps} "
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
                f"eval_progress step={step}/{max_steps} "
                f"loss={final_eval_metrics.get('loss/total', float('nan')):.6f}",
                flush=True,
            )
        if should_save:
            _write_checkpoint_and_report(
                output_dir=output_dir,
                checkpoint_path=checkpoint_path,
                report_path=report_path,
                model=model,
                optimizer=optimizer,
                model_config=model_config,
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
                parameter_count=model.parameter_count(),
                dataset_report=dataset_report,
                history=history,
                last_train_metrics=last_train_metrics,
                final_train_metrics=final_train_metrics,
                final_eval_metrics=final_eval_metrics,
                resume_config=resume_config,
                resume_from=resume_from,
            )
            print(
                f"checkpoint_progress step={step}/{max_steps} latest_path={checkpoint_path}",
                flush=True,
            )

    if not checkpoint_path.is_file():
        _write_checkpoint_and_report(
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            report_path=report_path,
            model=model,
            optimizer=optimizer,
            model_config=model_config,
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
            parameter_count=model.parameter_count(),
            dataset_report=dataset_report,
            history=history,
            last_train_metrics=last_train_metrics,
            final_train_metrics=final_train_metrics,
            final_eval_metrics=final_eval_metrics,
            resume_config=resume_config,
            resume_from=resume_from,
        )
    result_metrics = final_eval_metrics or last_train_metrics
    return ControlTrainingResult(
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        final_loss=float(result_metrics.get("loss/total", float("nan"))),
        final_value_loss=float(result_metrics.get("loss/value", float("nan"))),
        final_confidence_loss=float(result_metrics.get("loss/confidence", float("nan"))),
        completed_steps=completed_step,
    )


def _infinite_loader(loader: DataLoader):
    while True:
        yield from loader


@torch.no_grad()
def metrics_for_loader(
    model: ControlEncoder,
    loss_fn: ControlModelLoss,
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
            if _metric_is_count(key):
                count_totals[key] = count_totals.get(key, 0.0) + float(value)
        for key, numerator in loss_output.metric_numerators.items():
            mean_numerators[key] = mean_numerators.get(key, 0.0) + float(numerator)
        for key, denominator in loss_output.metric_denominators.items():
            mean_denominators[key] = mean_denominators.get(key, 0.0) + float(denominator)

        unresolved = set(loss_output.metrics) - set(loss_output.metric_numerators) - set(count_totals) - {"loss/total"}
        valid_frame_weight = max(float(loss_output.metrics.get("target/valid_frame_count", 0.0)), 1.0)
        for key in unresolved:
            fallback_totals[key] = fallback_totals.get(key, 0.0) + float(loss_output.metrics[key]) * valid_frame_weight
            fallback_weights[key] = fallback_weights.get(key, 0.0) + valid_frame_weight

    if not (count_totals or mean_numerators or fallback_totals):
        return {"loss/total": float("nan"), "loss/value": float("nan"), "loss/confidence": float("nan")}
    metrics = dict(count_totals)
    for key, numerator in mean_numerators.items():
        metrics[key] = _safe_float_div(numerator, mean_denominators.get(key, 0.0))
    for key, total in fallback_totals.items():
        metrics[key] = _safe_float_div(total, fallback_weights[key])
    if "loss/value" in metrics and "loss/confidence" in metrics:
        metrics["loss/total"] = metrics["loss/value"] + loss_fn.config.confidence_loss_weight * metrics["loss/confidence"]
    return metrics


def _loss_for_raw_batch(
    model: ControlEncoder,
    loss_fn: ControlModelLoss,
    raw_batch: Mapping[str, Any],
    *,
    device: torch.device,
):
    batch = dict(raw_batch) if "context_mel" in raw_batch else prepare_control_context_batch(raw_batch)
    batch = _move_batch_tensors(batch, device, keys=LOSS_BATCH_TENSOR_KEYS)
    output = model(
        context_mel=batch["context_mel"],
        context_dense_timing_v2=batch["context_dense_timing_v2"],
        normalized_difficulty=batch["normalized_difficulty"],
        context_padding_mask=batch["context_padding_mask"],
    )
    return loss_fn(
        output,
        control_v3_target=batch["control_v3_target"],
        target_valid_mask=batch["target_valid_mask"],
        ln_change_n_eff_target=batch.get("ln_change_n_eff_target"),
    )


def _write_checkpoint_and_report(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    report_path: Path,
    model: ControlEncoder,
    optimizer: torch.optim.Optimizer,
    model_config: ControlEncoderConfig,
    loss_config: ControlLossConfig,
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
    parameter_count: int,
    dataset_report: Mapping[str, Any],
    history: list[dict[str, Any]],
    last_train_metrics: Mapping[str, float],
    final_train_metrics: Mapping[str, float],
    final_eval_metrics: Mapping[str, float],
    resume_config: Mapping[str, Any],
    resume_from: Path | None,
) -> None:
    checkpoint_payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "loss_config": asdict(loss_config),
        "training_config": dict(resume_config),
        "seed": seed,
        "run_name": run_name,
        "history": history,
        "training_state": {
            "step": completed_steps,
            "max_steps": max_steps,
            "is_complete": completed_steps >= max_steps,
            "eval_every": eval_every,
            "save_every": save_every,
            "log_every": log_every,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "device": str(device),
            "last_train_metrics": _json_metrics(last_train_metrics),
            "final_train_metrics": _json_metrics(final_train_metrics),
            "final_eval_metrics": _json_metrics(final_eval_metrics),
            "resume_from": resume_from.as_posix() if resume_from is not None else None,
            "rng_state": _capture_rng_state(),
        },
    }
    archive_path = output_dir / "checkpoints" / f"checkpoint_step_{completed_steps:06d}.pt"
    _atomic_torch_save(checkpoint_payload, archive_path)
    _copy_file_atomically(archive_path, checkpoint_path)
    report_payload = {
        "run_name": run_name,
        "seed": seed,
        "max_steps": max_steps,
        "completed_steps": completed_steps,
        "is_complete": completed_steps >= max_steps,
        "eval_every": eval_every,
        "save_every": save_every,
        "log_every": log_every,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "model_config": asdict(model_config),
        "loss_config": asdict(loss_config),
        "training_config": dict(resume_config),
        "device": str(device),
        "parameter_count": parameter_count,
        "dataset": dict(dataset_report),
        "feature_names": {
            "value": list(VALUE_FEATURE_NAMES),
            "confidence": list(CONFIDENCE_FEATURE_NAMES),
            "model": list(MODEL_FEATURE_NAMES),
        },
        "history": history,
        "last_train_metrics": _json_metrics(last_train_metrics),
        "final_train_metrics": _json_metrics(final_train_metrics),
        "final_eval_metrics": _json_metrics(final_eval_metrics),
        "resume_from": resume_from.as_posix() if resume_from is not None else None,
    }
    _write_report(report_path, report_payload)


def _resume_training_config(
    *,
    seed: int,
    run_name: str,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    eval_every: int,
    save_every: int,
    dataset_report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "seed": seed,
        "run_name": run_name,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "eval_every": eval_every,
        "save_every": save_every,
        "dataset": _json_safe(_strict_resume_dataset_report(dataset_report)),
    }


def _load_resume_checkpoint(
    resume_from: Path,
    *,
    expected_model_config: ControlEncoderConfig,
    expected_loss_config: ControlLossConfig,
    expected_training_config: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(resume_from, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError(
            "resume checkpoint could not be loaded safely with weights_only=True; "
            "use a checkpoint written by this trainer"
        ) from exc
    if not isinstance(checkpoint, dict):
        raise ValueError(f"resume checkpoint must contain a mapping: {resume_from}")
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("resume checkpoint schema version mismatch")
    if checkpoint.get("model_config") != asdict(expected_model_config):
        raise ValueError("resume checkpoint model_config does not match the requested run")
    if checkpoint.get("loss_config") != asdict(expected_loss_config):
        raise ValueError("resume checkpoint loss_config does not match the requested run")
    if _normalized_resume_training_config(checkpoint.get("training_config")) != _normalized_resume_training_config(
        expected_training_config
    ):
        raise ValueError("resume checkpoint training_config does not match the requested run")
    if "model_state_dict" not in checkpoint:
        raise ValueError("resume checkpoint missing model_state_dict")
    if "optimizer_state_dict" not in checkpoint:
        raise ValueError("resume checkpoint missing optimizer_state_dict")
    if not isinstance(checkpoint.get("training_state"), Mapping):
        raise ValueError("resume checkpoint missing training_state")
    if not isinstance(checkpoint.get("history"), list):
        raise ValueError("resume checkpoint history must be a list")
    state = checkpoint["training_state"]
    if not isinstance(state.get("step"), int) or state["step"] < 0:
        raise ValueError("resume checkpoint training_state.step must be a non-negative integer")
    if "rng_state" not in state:
        raise ValueError("resume checkpoint missing training_state.rng_state")
    return checkpoint


def _normalized_resume_training_config(config: object) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    normalized = dict(config)
    dataset = normalized.get("dataset")
    if isinstance(dataset, Mapping):
        normalized["dataset"] = _strict_resume_dataset_report(dataset)
    return normalized


def _strict_resume_dataset_report(dataset_report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in dataset_report.items()
        if key not in RESUME_DATASET_RUNTIME_KEYS
    }


def _validate_training_args(
    *,
    max_steps: int,
    eval_every: int,
    save_every: int | None,
    log_every: int | None,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
) -> None:
    _require_finite_number(float(learning_rate), "learning_rate")
    _require_finite_number(float(weight_decay), "weight_decay")
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if eval_every <= 0:
        raise ValueError(f"eval_every must be positive, got {eval_every}")
    if save_every is not None and save_every <= 0:
        raise ValueError(f"save_every must be positive, got {save_every}")
    if log_every is not None and log_every <= 0:
        raise ValueError(f"log_every must be positive, got {log_every}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if weight_decay < 0.0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")


def _set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def _capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    state: dict[str, Any] = {
        "python_random": {
            "version": int(python_state[0]),
            "state": [int(value) for value in python_state[1]],
            "gauss": None if python_state[2] is None else float(python_state[2]),
        },
        "numpy_random": {
            "bit_generator": str(numpy_state[0]),
            "state": [int(value) for value in numpy_state[1].tolist()],
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if hasattr(torch, "mps") and torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        try:
            state["mps"] = torch.mps.get_rng_state()
        except RuntimeError:
            pass
    return state


def _restore_rng_state(raw_state: object) -> None:
    if not isinstance(raw_state, Mapping):
        raise ValueError("resume checkpoint training_state.rng_state must be a mapping")
    if raw_state.get("python_random") is not None:
        random.setstate(_python_rng_state_from_checkpoint(raw_state["python_random"]))
    if raw_state.get("numpy_random") is not None:
        np.random.set_state(_numpy_rng_state_from_checkpoint(raw_state["numpy_random"]))
    torch_state = raw_state.get("torch")
    if torch_state is not None:
        if not isinstance(torch_state, torch.Tensor):
            raise ValueError("resume checkpoint torch RNG state must be a tensor")
        torch.set_rng_state(torch_state.cpu())
    if raw_state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(raw_state["cuda"])
    mps_state = raw_state.get("mps")
    if mps_state is not None and hasattr(torch, "mps") and torch.backends.mps.is_available() and hasattr(torch.mps, "set_rng_state"):
        if not isinstance(mps_state, torch.Tensor):
            raise ValueError("resume checkpoint mps RNG state must be a tensor")
        torch.mps.set_rng_state(mps_state.cpu())


def _python_rng_state_from_checkpoint(raw_state: object) -> tuple[int, tuple[int, ...], float | None]:
    if isinstance(raw_state, Mapping):
        version = raw_state.get("version")
        state = raw_state.get("state")
        gauss = raw_state.get("gauss")
    elif isinstance(raw_state, (list, tuple)) and len(raw_state) == 3:
        version, state, gauss = raw_state
    else:
        raise ValueError("resume checkpoint python RNG state must be a mapping")

    if not isinstance(state, (list, tuple)):
        raise ValueError("resume checkpoint python RNG state keys must be a sequence")
    return (int(version), tuple(int(value) for value in state), None if gauss is None else float(gauss))


def _numpy_rng_state_from_checkpoint(raw_state: object) -> tuple[str, np.ndarray, int, int, float]:
    if isinstance(raw_state, Mapping):
        bit_generator = raw_state.get("bit_generator")
        keys = raw_state.get("state")
        pos = raw_state.get("pos")
        has_gauss = raw_state.get("has_gauss")
        cached_gaussian = raw_state.get("cached_gaussian")
    elif isinstance(raw_state, (list, tuple)) and len(raw_state) == 5:
        bit_generator, keys, pos, has_gauss, cached_gaussian = raw_state
    else:
        raise ValueError("resume checkpoint numpy RNG state must be a mapping")

    if not isinstance(bit_generator, str):
        raise ValueError("resume checkpoint numpy RNG bit_generator must be a string")
    if isinstance(keys, torch.Tensor):
        key_array = keys.detach().cpu().numpy().astype(np.uint32, copy=False)
    elif isinstance(keys, np.ndarray):
        key_array = keys.astype(np.uint32, copy=False)
    elif isinstance(keys, (list, tuple)):
        key_array = np.asarray(keys, dtype=np.uint32)
    else:
        raise ValueError("resume checkpoint numpy RNG state keys must be a sequence")
    return (bit_generator, key_array, int(pos), int(has_gauss), float(cached_gaussian))


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _advance_training_iterator(iterator: Any, completed_step: int) -> Any:
    for _ in range(completed_step):
        next(iterator)
    return iterator


def _move_batch_tensors(
    batch: Mapping[str, Any],
    device: torch.device,
    *,
    keys: frozenset[str] | None = None,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if keys is not None and key not in keys:
            continue
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    torch.save(dict(payload), tmp_path)
    tmp_path.replace(path)


def _copy_file_atomically(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination_path.with_name(f".{destination_path.name}.tmp")
    shutil.copy2(source_path, tmp_path)
    tmp_path.replace(destination_path)


def _write_report(report_path: Path, payload: Mapping[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = report_path.with_name(f".{report_path.name}.tmp")
    tmp_path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(report_path)


def _json_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in metrics.items()}


def _metric_is_count(key: str) -> bool:
    return key.endswith("_count")


def _safe_float_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _require_finite_number(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


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
    config_defaults = load_run_config(config_args.config) if config_args.config is not None else {"model": {}, "loss": {}}
    model_defaults = config_defaults["model"]
    loss_defaults = config_defaults["loss"]

    parser = argparse.ArgumentParser(description="Train the Stage 2 control encoder.", parents=[config_parser])
    parser.add_argument("--dataset-root", default=config_defaults.get("dataset_root", "dataset"))
    parser.add_argument("--index-path", default=config_defaults.get("index_path"))
    parser.add_argument("--eval-index-path", default=config_defaults.get("eval_index_path"))
    parser.add_argument("--control-v3-timeseries-path", default=config_defaults.get("control_v3_timeseries_path"))
    parser.add_argument("--output-dir", default=config_defaults.get("output_dir", DEFAULT_OUTPUT_DIR.as_posix()))
    parser.add_argument("--max-steps", type=int, default=config_defaults.get("max_steps", 5000))
    parser.add_argument("--eval-every", type=int, default=config_defaults.get("eval_every", 100))
    parser.add_argument("--save-every", type=int, default=config_defaults.get("save_every"))
    parser.add_argument("--log-every", type=int, default=config_defaults.get("log_every"))
    parser.add_argument("--batch-size", type=int, default=config_defaults.get("batch_size", 8))
    parser.add_argument("--learning-rate", type=float, default=config_defaults.get("learning_rate", 3e-4))
    parser.add_argument("--weight-decay", type=float, default=config_defaults.get("weight_decay", 0.01))
    parser.add_argument("--seed", type=int, default=config_defaults.get("seed", 1337))
    parser.add_argument("--device", default=config_defaults.get("device", "auto"), choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--run-name", default=config_defaults.get("run_name", "control_encoder"))
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
    parser.add_argument("--d-model", type=int, default=model_defaults.get("d_model"))
    parser.add_argument("--heads", type=int, default=model_defaults.get("heads"))
    parser.add_argument("--layers", type=int, default=model_defaults.get("layers"))
    parser.add_argument("--ffn-dim", type=int, default=model_defaults.get("ffn_dim"))
    parser.add_argument("--dropout", type=float, default=model_defaults.get("dropout"))
    parser.add_argument("--conv-blocks", type=int, default=model_defaults.get("conv_blocks"))
    parser.add_argument("--conv-kernel-size", type=int, default=model_defaults.get("conv_kernel_size"))
    parser.add_argument("--confidence-loss-weight", type=float, default=loss_defaults.get("confidence_loss_weight"))
    parser.add_argument("--sparse-boost", type=float, default=loss_defaults.get("sparse_boost"))
    args = parser.parse_args(argv)

    model_overrides = dict(model_defaults)
    for key in ("d_model", "heads", "layers", "ffn_dim", "dropout", "conv_blocks", "conv_kernel_size"):
        value = getattr(args, key)
        if value is not None:
            model_overrides[key] = value
    loss_overrides = dict(loss_defaults)
    if args.confidence_loss_weight is not None:
        loss_overrides["confidence_loss_weight"] = args.confidence_loss_weight
    if args.sparse_boost is not None:
        loss_overrides["sparse_boost"] = args.sparse_boost

    result = run_control_training(
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
        resume_from=Path(args.resume_from) if args.resume_from is not None else None,
        eval_fraction=args.eval_fraction,
        eval_size=args.eval_size,
        final_train_eval_size=args.final_train_eval_size,
        num_workers=args.num_workers,
        max_cached_maps=args.max_cached_maps,
        model_config_overrides=model_overrides,
        loss_config_overrides=loss_overrides,
    )
    print(f"report_path {result.report_path}")
    print(f"checkpoint_path {result.checkpoint_path}")
    print(f"final_loss {result.final_loss:.6f}")
    print(f"completed_steps {result.completed_steps}")


if __name__ == "__main__":
    main()
