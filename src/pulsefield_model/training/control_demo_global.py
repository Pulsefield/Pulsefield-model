from __future__ import annotations

import argparse
import math
import pickle
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from pulsefield_model.data.control_demo_global_windows import (
    CONTROL_DEMO_CONFIDENCE_FEATURE_NAMES,
    CONTROL_DEMO_TARGET_FEATURE_NAMES,
    CONTROL_DEMO_VALUE_FEATURE_NAMES,
    collate_control_demo_global_windows,
    extract_control_demo_target,
)
from pulsefield_model.data.control_windows import ControlWindowDataset, DEFAULT_MAX_CACHED_MAPS
from pulsefield_model.features.control_v3_targets import VALUE_FEATURE_NAMES
from pulsefield_model.models.control import (
    ControlDemoLossConfig,
    ControlDemoModelLoss,
    prepare_control_context_batch,
)
from pulsefield_model.models.control import (
    ControlDemoGlobalEncoder,
    ControlDemoGlobalEncoderConfig,
)
from pulsefield_model.training.common import (
    CHECKPOINT_SCHEMA_VERSION,
    DEFAULT_FINAL_TRAIN_EVAL_SIZE,
    ControlTrainingResult,
    _advance_training_iterator,
    _atomic_torch_save,
    _capture_rng_state,
    _copy_file_atomically,
    _infinite_loader,
    _json_metrics,
    _json_safe,
    _load_resume_checkpoint,
    _metric_is_count,
    _move_batch_tensors,
    _move_optimizer_state_to_device,
    _normalize_config_mapping,
    _normalized_section,
    _restore_rng_state,
    _safe_float_div,
    _set_deterministic_seed,
    _validate_training_args,
    _write_report,
    limit_final_train_eval_dataset,
    select_torch_device,
    split_train_eval_dataset,
)


DEFAULT_RUNS_ROOT = Path("artifacts/runs/stage2_control_demo")
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_ROOT / "control_demo_global_encoder"
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
    "global_attention_budget",
    "learning_rate",
    "weight_decay",
    "seed",
    "device",
    "run_name",
    "resume_from",
    "init_from_control_checkpoint",
    "eval_fraction",
    "eval_size",
    "final_train_eval_size",
    "num_workers",
    "max_cached_maps",
    "model",
    "loss",
}
MODEL_CONFIG_KEYS = {field.name for field in fields(ControlDemoGlobalEncoderConfig)}
LOSS_CONFIG_KEYS = {field.name for field in fields(ControlDemoLossConfig)}
LOSS_BATCH_TENSOR_KEYS = frozenset(
    (
        "full_mel",
        "full_dense_timing_v2",
        "padding_mask",
        "frame_count",
        "target_start_frame",
        "context_mel",
        "context_dense_timing_v2",
        "normalized_difficulty",
        "context_padding_mask",
        "control_demo_target",
        "target_valid_mask",
    )
)
GLOBAL_PARAMETER_PREFIXES = (
    "global_encoder",
    "global_condition_projection",
    "global_fusions",
)


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


def run_control_demo_global_training(
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
    global_attention_budget: int | None = None,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    seed: int = 1337,
    device_name: str = "auto",
    run_name: str = "control_demo_global_encoder",
    resume_from: Path | None = None,
    init_from_control_checkpoint: Path | None = None,
    eval_fraction: float = 0.1,
    eval_size: int | None = None,
    final_train_eval_size: int | None = DEFAULT_FINAL_TRAIN_EVAL_SIZE,
    num_workers: int = 0,
    max_cached_maps: int | None = None,
    model_config_overrides: Mapping[str, Any] | None = None,
    loss_config_overrides: Mapping[str, Any] | None = None,
) -> ControlTrainingResult:
    _set_deterministic_seed(seed)
    model_config = ControlDemoGlobalEncoderConfig(**dict(model_config_overrides or {}))
    global_attention_budget = _validate_optional_positive_int(
        global_attention_budget,
        "global_attention_budget",
    )
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

    loader = _make_global_control_demo_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=seed,
        global_attention_budget=global_attention_budget,
        global_stride=model_config.global_stride,
    )
    train_eval_dataset = limit_final_train_eval_dataset(
        train_dataset,
        final_train_eval_size=final_train_eval_size,
        seed=seed,
    )
    train_eval_loader = _make_global_control_demo_loader(
        train_eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
        global_attention_budget=global_attention_budget,
        global_stride=model_config.global_stride,
    )
    eval_loader = _make_global_control_demo_loader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed,
        global_attention_budget=global_attention_budget,
        global_stride=model_config.global_stride,
    )
    return _run_training(
        loader=loader,
        train_eval_loader=train_eval_loader,
        eval_loader=eval_loader,
        output_dir=output_dir,
        model_config=model_config,
        loss_config=ControlDemoLossConfig(**dict(loss_config_overrides or {})),
        max_steps=max_steps,
        eval_every=eval_every,
        save_every=save_every,
        log_every=log_every,
        batch_size=batch_size,
        global_attention_budget=global_attention_budget,
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
        init_from_control_checkpoint=init_from_control_checkpoint,
    )


def _make_global_control_demo_loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    global_attention_budget: int | None,
    global_stride: int,
) -> DataLoader:
    if global_attention_budget is None:
        generator = None
        if shuffle:
            generator = torch.Generator()
            generator.manual_seed(seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
            num_workers=num_workers,
            collate_fn=collate_control_demo_global_windows,
        )

    return DataLoader(
        dataset,
        batch_sampler=_GlobalAttentionBudgetBatchSampler(
            dataset,
            batch_size=batch_size,
            global_attention_budget=global_attention_budget,
            global_stride=global_stride,
            shuffle=shuffle,
            seed=seed,
        ),
        num_workers=num_workers,
        collate_fn=collate_control_demo_global_windows,
    )


class _GlobalAttentionBudgetBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: Dataset[Any],
        *,
        batch_size: int,
        global_attention_budget: int,
        global_stride: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if global_stride <= 0:
            raise ValueError(f"global_stride must be positive, got {global_stride}")
        self.frame_counts = [_dataset_frame_count(dataset, index) for index in range(len(dataset))]
        self.batch_size = batch_size
        self.global_attention_budget = _positive_int(global_attention_budget, "global_attention_budget")
        self.global_stride = global_stride
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        indices = list(range(len(self.frame_counts)))
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(indices), generator=generator).tolist()
        self.epoch += 1
        yield from self._batches_for(indices)

    def __len__(self) -> int:
        return len(list(self._batches_for(list(range(len(self.frame_counts))))))

    def _batches_for(self, indices: Sequence[int]):
        batch: list[int] = []
        max_pooled_tokens = 0
        for index in indices:
            pooled_tokens = math.ceil(self.frame_counts[index] / self.global_stride)
            candidate_size = len(batch) + 1
            candidate_max = max(max_pooled_tokens, pooled_tokens)
            candidate_attention = candidate_size * candidate_max * candidate_max
            if batch and (
                len(batch) >= self.batch_size or candidate_attention > self.global_attention_budget
            ):
                yield batch
                batch = []
                max_pooled_tokens = 0

            batch.append(index)
            max_pooled_tokens = max(max_pooled_tokens, pooled_tokens)

        if batch:
            yield batch


def _dataset_frame_count(dataset: Dataset[Any], index: int) -> int:
    if isinstance(dataset, Subset):
        return _dataset_frame_count(dataset.dataset, int(dataset.indices[index]))

    records = getattr(dataset, "records", None)
    if records is not None:
        return _positive_int(getattr(records[index], "frame_count"), "frame_count")

    sample = dataset[index]
    if not isinstance(sample, Mapping) or "frame_count" not in sample:
        raise ValueError("dataset samples must expose frame_count for global attention budgeting")
    return _positive_int(sample["frame_count"], "frame_count")


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


def _validate_optional_positive_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


def _run_training(
    *,
    loader: DataLoader,
    train_eval_loader: DataLoader,
    eval_loader: DataLoader,
    output_dir: Path,
    model_config: ControlDemoGlobalEncoderConfig,
    loss_config: ControlDemoLossConfig,
    max_steps: int,
    eval_every: int,
    save_every: int | None,
    log_every: int | None,
    batch_size: int,
    global_attention_budget: int | None,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device_name: str,
    run_name: str,
    dataset_report: Mapping[str, Any],
    resume_from: Path | None,
    init_from_control_checkpoint: Path | None,
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
    _set_deterministic_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_torch_device(device_name)
    save_every = eval_every if save_every is None else save_every
    checkpoint_path = output_dir / "checkpoint.pt"
    report_path = output_dir / "report.json"

    model = ControlDemoGlobalEncoder(model_config).to(device)
    initialization_report: dict[str, Any] | None = None
    if resume_from is None and init_from_control_checkpoint is not None:
        initialization_report = initialize_global_control_demo_from_control_checkpoint(
            model,
            init_from_control_checkpoint,
        )
    loss_fn = ControlDemoModelLoss(loss_config)
    optimizer = _build_control_demo_global_optimizer(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    resume_config = _resume_training_config(
        seed=seed,
        run_name=run_name,
        batch_size=batch_size,
        global_attention_budget=global_attention_budget,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        eval_every=eval_every,
        save_every=save_every,
        dataset_report=dataset_report,
        init_from_control_checkpoint=init_from_control_checkpoint,
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
        total_loss = loss_output.total_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_train_metrics = dict(loss_output.metrics)
        last_train_metrics["loss/total"] = float(total_loss.detach().cpu())
        del loss_output, total_loss
        _release_device_cache(device)
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
                initialization_report=initialization_report,
            )
            print(
                f"checkpoint_progress step={step}/{max_steps} latest_path={checkpoint_path}",
                flush=True,
            )

    result_metrics = final_eval_metrics or last_train_metrics
    return ControlTrainingResult(
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        final_loss=float(result_metrics.get("loss/total", float("nan"))),
        final_value_loss=float(result_metrics.get("loss/value", float("nan"))),
        final_confidence_loss=float(result_metrics.get("loss/confidence", 0.0)),
        completed_steps=completed_step,
    )


@torch.no_grad()
def metrics_for_loader(
    model: ControlDemoGlobalEncoder,
    loss_fn: ControlDemoModelLoss,
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
        del loss_output
        _release_device_cache(device)

    if not (count_totals or mean_numerators or fallback_totals):
        return {"loss/total": float("nan"), "loss/value": float("nan"), "loss/confidence": 0.0}
    metrics = dict(count_totals)
    for key, numerator in mean_numerators.items():
        metrics[key] = _safe_float_div(numerator, mean_denominators.get(key, 0.0))
    for key, total in fallback_totals.items():
        metrics[key] = _safe_float_div(total, fallback_weights[key])
    if "loss/value" in metrics:
        metrics["loss/total"] = metrics["loss/value"]
    return metrics


def _release_device_cache(device: torch.device) -> None:
    if device.type != "mps" or not hasattr(torch, "mps"):
        return
    synchronize = getattr(torch.mps, "synchronize", None)
    if synchronize is not None:
        synchronize()
    empty_cache = getattr(torch.mps, "empty_cache", None)
    if empty_cache is not None:
        empty_cache()


def _loss_for_raw_batch(
    model: ControlDemoGlobalEncoder,
    loss_fn: ControlDemoModelLoss,
    raw_batch: Mapping[str, Any],
    *,
    device: torch.device,
):
    batch = dict(raw_batch) if "context_mel" in raw_batch else prepare_control_context_batch(raw_batch)
    if "control_demo_target" not in batch:
        if "control_v3_target" not in batch:
            raise ValueError("batch must contain control_demo_target or control_v3_target")
        batch["control_demo_target"] = extract_control_demo_target(batch["control_v3_target"])
    batch = _move_batch_tensors(batch, device, keys=LOSS_BATCH_TENSOR_KEYS)
    if model.config.use_global_memory:
        output = model(
            context_mel=batch["context_mel"],
            context_dense_timing_v2=batch["context_dense_timing_v2"],
            normalized_difficulty=batch["normalized_difficulty"],
            context_padding_mask=batch["context_padding_mask"],
            full_mel=batch["full_mel"],
            full_dense_timing_v2=batch["full_dense_timing_v2"],
            padding_mask=batch["padding_mask"],
            frame_count=batch["frame_count"],
            target_start_frame=batch["target_start_frame"],
        )
    else:
        output = model(
            context_mel=batch["context_mel"],
            context_dense_timing_v2=batch["context_dense_timing_v2"],
            normalized_difficulty=batch["normalized_difficulty"],
            context_padding_mask=batch["context_padding_mask"],
        )
    return loss_fn(
        output,
        control_demo_target=batch["control_demo_target"],
        target_valid_mask=batch["target_valid_mask"],
    )


def _build_control_demo_global_optimizer(
    model: ControlDemoGlobalEncoder,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    local_params: list[torch.nn.Parameter] = []
    value_head_params: list[torch.nn.Parameter] = []
    global_params: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("value_head"):
            value_head_params.append(parameter)
        elif name.startswith(GLOBAL_PARAMETER_PREFIXES):
            global_params.append(parameter)
        else:
            local_params.append(parameter)

    param_groups: list[dict[str, Any]] = []
    if local_params:
        param_groups.append({"name": "local_pretrained", "params": local_params, "lr": learning_rate * 0.25})
    if value_head_params:
        param_groups.append({"name": "value_head", "params": value_head_params, "lr": learning_rate * 0.50})
    if global_params:
        param_groups.append({"name": "global_path", "params": global_params, "lr": learning_rate})
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def initialize_global_control_demo_from_control_checkpoint(
    model: ControlDemoGlobalEncoder,
    checkpoint_path: Path,
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError(
            "control checkpoint could not be loaded safely with weights_only=True; "
            "use a checkpoint written by the control trainer"
        ) from exc
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"control checkpoint must contain a mapping: {checkpoint_path}")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("control checkpoint missing model_state_dict")

    model_state = model.state_dict()
    compatible_state: dict[str, torch.Tensor] = {}
    loaded_keys: list[str] = []
    skipped_keys: list[str] = []
    unexpected_keys: list[str] = []
    density_head_weight_copied = False
    density_head_bias_copied = False
    for key, value in state.items():
        if key in {"value_head.weight", "value_head.bias", "confidence_head.weight", "confidence_head.bias"}:
            continue
        if not isinstance(value, torch.Tensor):
            skipped_keys.append(str(key))
            continue
        if key not in model_state:
            unexpected_keys.append(str(key))
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            skipped_keys.append(str(key))
            continue
        compatible_state[str(key)] = value.to(dtype=model_state[key].dtype)
        loaded_keys.append(str(key))

    density_index = VALUE_FEATURE_NAMES.index("density_level")
    value_weight = state.get("value_head.weight")
    value_bias = state.get("value_head.bias")
    if isinstance(value_weight, torch.Tensor) and tuple(value_weight[density_index : density_index + 1].shape) == tuple(
        model_state["value_head.weight"].shape
    ):
        compatible_state["value_head.weight"] = value_weight[density_index : density_index + 1].to(
            dtype=model_state["value_head.weight"].dtype
        )
        loaded_keys.append("value_head.weight[density_level]")
        density_head_weight_copied = True
    if isinstance(value_bias, torch.Tensor) and tuple(value_bias[density_index : density_index + 1].shape) == tuple(
        model_state["value_head.bias"].shape
    ):
        compatible_state["value_head.bias"] = value_bias[density_index : density_index + 1].to(
            dtype=model_state["value_head.bias"].dtype
        )
        loaded_keys.append("value_head.bias[density_level]")
        density_head_bias_copied = True

    load_result = model.load_state_dict(compatible_state, strict=False)
    missing_global_keys = [
        key
        for key in load_result.missing_keys
        if key.startswith(GLOBAL_PARAMETER_PREFIXES)
    ]
    unexpected_load_keys = [str(key) for key in load_result.unexpected_keys]
    if unexpected_load_keys:
        unexpected_keys.extend(unexpected_load_keys)

    report = {
        "source_checkpoint": checkpoint_path.as_posix(),
        "loaded_key_count": len(loaded_keys),
        "missing_global_key_count": len(missing_global_keys),
        "unexpected_key_count": len(unexpected_keys),
        "skipped_key_count": len(skipped_keys),
        "density_head_copied": density_head_weight_copied or density_head_bias_copied,
        "density_head_weight_copied": density_head_weight_copied,
        "density_head_bias_copied": density_head_bias_copied,
        "loaded_keys": loaded_keys,
        "missing_global_keys": missing_global_keys[:64],
        "unexpected_keys": unexpected_keys[:64],
        "skipped_keys": skipped_keys[:32],
    }
    print(
        "global_control_demo_init "
        f"checkpoint={checkpoint_path} "
        f"loaded_keys={report['loaded_key_count']} "
        f"missing_global_keys={report['missing_global_key_count']} "
        f"unexpected_keys={report['unexpected_key_count']} "
        f"density_head_copied={str(report['density_head_copied']).lower()}",
        flush=True,
    )
    return report


def _write_checkpoint_and_report(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    report_path: Path,
    model: ControlDemoGlobalEncoder,
    optimizer: torch.optim.Optimizer,
    model_config: ControlDemoGlobalEncoderConfig,
    loss_config: ControlDemoLossConfig,
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
    initialization_report: Mapping[str, Any] | None,
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
        "initialization": None if initialization_report is None else dict(initialization_report),
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
            "value": list(CONTROL_DEMO_VALUE_FEATURE_NAMES),
            "confidence": list(CONTROL_DEMO_CONFIDENCE_FEATURE_NAMES),
            "target": list(CONTROL_DEMO_TARGET_FEATURE_NAMES),
        },
        "initialization": None if initialization_report is None else dict(initialization_report),
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
    global_attention_budget: int | None,
    learning_rate: float,
    weight_decay: float,
    eval_every: int,
    save_every: int,
    dataset_report: Mapping[str, Any],
    init_from_control_checkpoint: Path | None,
) -> dict[str, Any]:
    config = {
        "seed": seed,
        "run_name": run_name,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "eval_every": eval_every,
        "save_every": save_every,
        "dataset": _json_safe(_strict_resume_dataset_report(dataset_report)),
        "init_from_control_checkpoint": (
            init_from_control_checkpoint.as_posix() if init_from_control_checkpoint is not None else None
        ),
    }
    if global_attention_budget is not None:
        config["global_attention_budget"] = global_attention_budget
    return config


def _strict_resume_dataset_report(dataset_report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in dataset_report.items()
        if key not in {"max_cached_maps", "num_workers"}
    }


def main(argv: Sequence[str] | None = None) -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None, help="YAML run config; CLI flags override config values")
    config_args, _ = config_parser.parse_known_args(argv)
    config_defaults = load_run_config(config_args.config) if config_args.config is not None else {"model": {}, "loss": {}}
    model_defaults = config_defaults["model"]
    loss_defaults = config_defaults["loss"]

    parser = argparse.ArgumentParser(
        description="Train the Stage 2 density-only global control demo encoder.",
        parents=[config_parser],
    )
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
    parser.add_argument("--global-attention-budget", type=int, default=config_defaults.get("global_attention_budget"))
    parser.add_argument("--learning-rate", type=float, default=config_defaults.get("learning_rate", 3e-4))
    parser.add_argument("--weight-decay", type=float, default=config_defaults.get("weight_decay", 0.01))
    parser.add_argument("--seed", type=int, default=config_defaults.get("seed", 1337))
    parser.add_argument("--device", default=config_defaults.get("device", "auto"), choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--run-name", default=config_defaults.get("run_name", "control_demo_global_encoder"))
    parser.add_argument("--resume-from", default=config_defaults.get("resume_from"))
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
    parser.add_argument("--d-model", type=int, default=model_defaults.get("d_model"))
    parser.add_argument("--heads", type=int, default=model_defaults.get("heads"))
    parser.add_argument("--layers", type=int, default=model_defaults.get("layers"))
    parser.add_argument("--ffn-dim", type=int, default=model_defaults.get("ffn_dim"))
    parser.add_argument("--dropout", type=float, default=model_defaults.get("dropout"))
    parser.add_argument("--conv-blocks", type=int, default=model_defaults.get("conv_blocks"))
    parser.add_argument("--conv-kernel-size", type=int, default=model_defaults.get("conv_kernel_size"))
    parser.add_argument(
        "--use-global-memory",
        action=argparse.BooleanOptionalAction,
        default=model_defaults.get("use_global_memory"),
    )
    parser.add_argument("--global-stride", type=int, default=model_defaults.get("global_stride"))
    parser.add_argument("--global-layers", type=int, default=model_defaults.get("global_layers"))
    parser.add_argument("--global-ffn-dim", type=int, default=model_defaults.get("global_ffn_dim"))
    parser.add_argument("--global-conv-blocks", type=int, default=model_defaults.get("global_conv_blocks"))
    parser.add_argument("--global-fusion-start-layer", type=int, default=model_defaults.get("global_fusion_start_layer"))
    parser.add_argument("--global-gate-init", type=float, default=model_defaults.get("global_gate_init"))
    parser.add_argument("--density-loss-weight", type=float, default=loss_defaults.get("density_loss_weight"))
    parser.add_argument("--smooth-l1-delta", type=float, default=loss_defaults.get("smooth_l1_delta"))
    args = parser.parse_args(argv)

    model_overrides = dict(model_defaults)
    for key in (
        "d_model",
        "heads",
        "layers",
        "ffn_dim",
        "dropout",
        "conv_blocks",
        "conv_kernel_size",
        "use_global_memory",
        "global_stride",
        "global_layers",
        "global_ffn_dim",
        "global_conv_blocks",
        "global_fusion_start_layer",
        "global_gate_init",
    ):
        value = getattr(args, key)
        if value is not None:
            model_overrides[key] = value
    loss_overrides = dict(loss_defaults)
    if args.density_loss_weight is not None:
        loss_overrides["density_loss_weight"] = args.density_loss_weight
    if args.smooth_l1_delta is not None:
        loss_overrides["smooth_l1_delta"] = args.smooth_l1_delta

    init_from = Path(args.init_from_control_checkpoint) if args.init_from_control_checkpoint is not None else None
    result = run_control_demo_global_training(
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
        global_attention_budget=args.global_attention_budget,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device_name=args.device,
        run_name=args.run_name,
        resume_from=Path(args.resume_from) if args.resume_from is not None else None,
        init_from_control_checkpoint=init_from,
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
