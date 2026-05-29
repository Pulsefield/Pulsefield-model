import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pulsefield_model.models.control import ControlDemoGlobalEncoderConfig
from pulsefield_model.models.mapper.v2_1 import MapperV21Config, MapperV21LossConfig
from pulsefield_model.training.common import ResumableRandomBatchSampler, _infinite_loader
from pulsefield_model.training import mapper_v2_1 as mapper_v2_1_training
from pulsefield_model.training.mapper_runner import (
    MapperTrainingConfigContext,
    MapperTrainingResumeContext,
    MapperTrainingSpec,
    default_mapper_metric_finalizer,
    run_mapper_training,
)


@dataclass(frozen=True)
class _TinyModelConfig:
    width: int = 1


@dataclass(frozen=True)
class _TinyLossConfig:
    lambda_ln_close: float = 0.5
    lambda_adapter_reg: float = 0.25
    lambda_density: float = 0.125


@dataclass(frozen=True)
class _TinyLossOutput:
    total_loss: torch.Tensor
    metrics: dict[str, float]
    metric_numerators: dict[str, float]
    metric_denominators: dict[str, float]


class MapperTrainingRunnerTests(unittest.TestCase):
    def test_runner_wires_spec_and_preserves_checkpoint_report_payload(self) -> None:
        events: list[str] = []
        model_config = _TinyModelConfig(width=3)
        loss_config = _TinyLossConfig()
        dataset_report = {
            "train_window_count": 2,
            "eval_window_count": 1,
            "num_workers": 7,
            "dataset_progress": True,
        }

        def model_factory(config: _TinyModelConfig, control_encoder: object | None) -> _TinyModel:
            self.assertEqual(config, model_config)
            self.assertIsNone(control_encoder)
            events.append("model_factory")
            return _TinyModel()

        def loss_factory(model: nn.Module, config: _TinyLossConfig) -> _TinyLossConfig:
            self.assertIsInstance(model, _TinyModel)
            self.assertEqual(config, loss_config)
            events.append("loss_factory")
            return config

        def training_config_factory(context: MapperTrainingConfigContext) -> dict[str, Any]:
            self.assertEqual(context.dataset_report, dataset_report)
            events.append("training_config_factory")
            return {
                "phase": "B",
                "seed": context.seed,
                "run_name": context.run_name,
                "learning_rate": context.learning_rate,
                "weight_decay": context.weight_decay,
                "eval_every": context.eval_every,
                "save_every": context.save_every,
                "skip_first_eval_pass": context.skip_first_eval_pass,
                "mps_cleanup_every": context.mps_cleanup_every,
                "dataset": dict(context.dataset_report),
                "custom_training_field": "kept",
            }

        def resume_loader(path: Path, context: MapperTrainingResumeContext) -> Mapping[str, Any]:
            raise AssertionError("resume loader should not be called")

        def checkpoint_hook(payload: dict[str, Any], context: Mapping[str, Any]) -> None:
            events.append("checkpoint_hook")
            payload["extra_checkpoint_field"] = context["run_name"]

        def report_hook(payload: dict[str, Any], context: Mapping[str, Any]) -> None:
            events.append("report_hook")
            payload["extra_report_field"] = context["run_name"]

        spec = MapperTrainingSpec(
            model_config=model_config,
            control_model_config=None,
            loss_config=loss_config,
            model_factory=model_factory,
            loss_factory=loss_factory,
            batch_loss_adapter=_tiny_batch_loss,
            optimizer_factory=lambda model, learning_rate, weight_decay: torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            ),
            training_config_factory=training_config_factory,
            resume_checkpoint_loader=resume_loader,
            metric_count_predicate=lambda key: key.endswith("_count"),
            metric_fallback_weight_key="target/token_count",
            metric_empty={"loss/total": float("nan"), "loss/token": float("nan"), "loss/density": 0.0},
            metric_finalizer=default_mapper_metric_finalizer,
            progress_label="tiny_mapper",
            checkpoint_payload_hook=checkpoint_hook,
            report_payload_hook=report_hook,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            loader = DataLoader(_TinyDataset(2), batch_size=1, shuffle=False)
            eval_loader = DataLoader(_TinyDataset(1), batch_size=1, shuffle=False)
            result = run_mapper_training(
                loader=loader,
                train_eval_loader=eval_loader,
                eval_loader=eval_loader,
                output_dir=Path(temp_dir),
                spec=spec,
                max_steps=1,
                eval_every=1,
                save_every=None,
                log_every=None,
                batch_size=1,
                learning_rate=0.1,
                weight_decay=0.0,
                seed=123,
                device_name="cpu",
                run_name="tiny_run",
                dataset_report=dataset_report,
            )

            checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
            report = json.loads(result.report_path.read_text(encoding="utf-8"))

        self.assertEqual(result.completed_steps, 1)
        self.assertEqual(checkpoint["model_config"], {"width": 3})
        self.assertEqual(checkpoint["loss_config"], {
            "lambda_ln_close": 0.5,
            "lambda_adapter_reg": 0.25,
            "lambda_density": 0.125,
        })
        self.assertEqual(checkpoint["training_config"]["dataset"], dataset_report)
        self.assertEqual(checkpoint["extra_checkpoint_field"], "tiny_run")
        self.assertEqual(report["training_config"]["dataset"], dataset_report)
        self.assertEqual(report["dataset"], dataset_report)
        self.assertEqual(report["extra_report_field"], "tiny_run")
        self.assertEqual(report["parameter_count"], 1)
        self.assertIn("train_eval", report["history"][-1])
        self.assertEqual(
            events,
            ["model_factory", "loss_factory", "training_config_factory", "checkpoint_hook", "report_hook"],
        )

    def test_v2_1_resume_normalization_ignores_runtime_only_fields(self) -> None:
        full = mapper_v2_1_training._mapper_v2_1_training_config(
            seed=1,
            run_name="resume",
            learning_rate=0.01,
            weight_decay=0.02,
            eval_every=10,
            save_every=20,
            skip_first_eval_pass=True,
            dataset_report={
                "train_window_count": 5,
                "eval_window_count": 2,
                "num_workers": 4,
                "max_cached_maps": 8,
                "dataset_progress": True,
            },
            mps_cleanup_every=25,
        )
        legacy_expected = mapper_v2_1_training._mapper_v2_1_resume_training_config(
            seed=1,
            run_name="resume",
            learning_rate=0.01,
            weight_decay=0.02,
            eval_every=10,
            save_every=20,
            skip_first_eval_pass=True,
            dataset_report={
                "train_window_count": 5,
                "eval_window_count": 2,
                "num_workers": 0,
                "max_cached_maps": 1,
                "dataset_progress": False,
            },
            mps_cleanup_every=None,
        )

        self.assertEqual(
            mapper_v2_1_training._normalized_mapper_v2_1_resume_training_config(full),
            mapper_v2_1_training._normalized_mapper_v2_1_resume_training_config(legacy_expected),
        )
        self.assertEqual(full["mapper_token_contract"], "v2.1_sparse_lane_actions")

    def test_v2_1_spec_uses_resumable_sampler_cursor_without_loading_skipped_batches(self) -> None:
        dataset_size = 7
        batch_size = 3
        seed = 923
        completed_batches = 5
        next_batch_count = 4

        baseline_generator = torch.Generator().manual_seed(seed)
        baseline_loader = DataLoader(
            _IndexDataset(dataset_size),
            batch_size=batch_size,
            shuffle=True,
            generator=baseline_generator,
            num_workers=0,
        )
        baseline_iterator = _infinite_loader(baseline_loader)
        for _ in range(completed_batches):
            next(baseline_iterator)
        expected = _take_index_batches(baseline_iterator, next_batch_count)

        resume_dataset = _IndexDataset(dataset_size)
        resume_loader = DataLoader(
            resume_dataset,
            batch_sampler=ResumableRandomBatchSampler(
                resume_dataset,
                batch_size=batch_size,
                seed=seed,
            ),
            num_workers=0,
        )
        spec = mapper_v2_1_training._mapper_v2_1_training_spec(
            model_config=MapperV21Config(),
            control_model_config=ControlDemoGlobalEncoderConfig(),
            loss_config=MapperV21LossConfig(),
            progress_label="mapper_v2_1_phase_b",
        )

        resumed_iterator = spec.resume_loader_cursor(
            resume_loader,
            _infinite_loader(resume_loader),
            completed_batches,
        )
        actual = _take_index_batches(resumed_iterator, next_batch_count)

        self.assertEqual(actual, expected)
        self.assertEqual(len(resume_dataset.loaded_indexes), sum(len(batch) for batch in actual))


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class _TinyDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = int(size)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"x": torch.tensor(float(index + 1), dtype=torch.float32)}


class _IndexDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = int(size)
        self.loaded_indexes: list[int] = []

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> torch.Tensor:
        self.loaded_indexes.append(int(index))
        return torch.tensor(int(index), dtype=torch.long)


def _tiny_batch_loss(
    model: nn.Module,
    loss_config: _TinyLossConfig,
    raw_batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> _TinyLossOutput:
    tiny_model = model
    if not isinstance(tiny_model, _TinyModel):
        raise TypeError("expected _TinyModel")
    x = raw_batch["x"].to(device=device, dtype=torch.float32)
    prediction = tiny_model.weight * x
    token_loss = (prediction.mean() - 1.0).pow(2)
    ln_close_loss = token_loss.detach().new_tensor(0.2) + token_loss * 0.0
    adapter_reg_loss = token_loss.detach().new_tensor(0.3) + token_loss * 0.0
    density_loss = token_loss.detach().new_tensor(0.4) + token_loss * 0.0
    total_loss = (
        token_loss
        + loss_config.lambda_ln_close * ln_close_loss
        + loss_config.lambda_adapter_reg * adapter_reg_loss
        + loss_config.lambda_density * density_loss
    )
    token_count = float(x.numel())
    metrics = {
        "loss/total": float(total_loss.detach().cpu()),
        "loss/token": float(token_loss.detach().cpu()),
        "loss/ln_close": float(ln_close_loss.detach().cpu()),
        "loss/adapter_reg": float(adapter_reg_loss.detach().cpu()),
        "loss/density": float(density_loss.detach().cpu()),
        "target/token_count": token_count,
        "sample_count": token_count,
    }
    return _TinyLossOutput(
        total_loss=total_loss,
        metrics=metrics,
        metric_numerators={
            "loss/token": metrics["loss/token"] * token_count,
            "loss/ln_close": metrics["loss/ln_close"] * token_count,
            "loss/adapter_reg": metrics["loss/adapter_reg"] * token_count,
            "loss/density": metrics["loss/density"] * token_count,
        },
        metric_denominators={
            "loss/token": token_count,
            "loss/ln_close": token_count,
            "loss/adapter_reg": token_count,
            "loss/density": token_count,
        },
    )


def _take_index_batches(iterator: object, count: int) -> list[list[int]]:
    batches: list[list[int]] = []
    for _ in range(count):
        batch = next(iterator)  # type: ignore[arg-type]
        batches.append([int(value) for value in batch.tolist()])
    return batches


if __name__ == "__main__":
    unittest.main()
