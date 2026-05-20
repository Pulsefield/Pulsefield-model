import tempfile
import unittest

import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from pulsefield_model.models.control import ControlDemoGlobalEncoderConfig
from pulsefield_model.models.mapper.v2_1 import MapperV21Config, MapperV21LossConfig
from pulsefield_model.training import mapper_v2_1 as mapper_v2_1_training
from pulsefield_model.training.common import ResumableRandomBatchSampler, _infinite_loader
from pulsefield_model.training.mapper_v2_1 import load_run_config

_STALE_ROOT = "train" + "/"


class MapperV21PhaseBTrainingTests(unittest.TestCase):
    def test_phase_b_sparse_global_config_loads_v2_1_fields(self) -> None:
        config = load_run_config(
            "configs/training/stage2_mapper_v2_1_phase_b_sparse_global_mps.yaml",
        )

        self.assertTrue(config["include_full_song_context"])
        self.assertTrue(config["skip_first_eval_pass"])
        self.assertEqual(config["mps_cleanup_every"], 20)
        self.assertEqual(config["batch_size"], 2)
        self.assertEqual(config["model"]["max_seq_len"], 1024)
        self.assertTrue(config["model"]["use_global_context"])
        self.assertEqual(config["loss"]["lambda_density"], 0.05)
        self.assertFalse(config["precompute_control_teacher_cache"])
        self.assertFalse(config["precompute_control_teacher_cache_only"])
        self.assertEqual(config["control_teacher_precompute_batch_size"], 12)
        self.assertFalse(config["control_teacher_cache_overwrite"])
        self.assertIn("stage2_control_windows", config["index_path"])
        self.assertIn("stage2_mapper_v2_1/window_records", config["mapper_record_cache_path"])
        self.assertIn("stage2_mapper_v2_1/control_teacher", config["control_teacher_cache_dir"])
        _assert_artifacts_paths(
            self,
            config,
            (
                "index_path",
                "control_v3_timeseries_path",
                "output_dir",
                "init_from_control_checkpoint",
                "mapper_record_cache_path",
                "control_teacher_cache_dir",
            ),
        )

        MapperV21Config(**config["model"])
        ControlDemoGlobalEncoderConfig(**config["control_model"])
        MapperV21LossConfig(**config["loss"])

    def test_run_config_accepts_resume_from(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "mapper_v2_1_child.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "resume_from": "artifacts/runs/stage2_mapper_v2_1/example/checkpoint.pt",
                        "model": {},
                        "control_model": {},
                        "loss": {},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            config = load_run_config(config_path)

        self.assertEqual(config["resume_from"], "artifacts/runs/stage2_mapper_v2_1/example/checkpoint.pt")

    def test_main_forwards_v2_1_training_options(self) -> None:
        train_result = SimpleNamespace(
            report_path=Path("report.json"),
            checkpoint_path=Path("checkpoint.pt"),
            final_loss=0.0,
            completed_steps=0,
        )
        with patch.object(
            mapper_v2_1_training,
            "run_mapper_v2_1_phase_b_training",
            return_value=train_result,
            autospec=True,
        ) as train:
            mapper_v2_1_training.main(
                [
                    "--config",
                    "configs/training/stage2_mapper_v2_1_phase_b_sparse_global_mps.yaml",
                    "--max-steps",
                    "1",
                    "--resume-from",
                    "artifacts/runs/stage2_mapper_v2_1/example/checkpoint.pt",
                ],
            )

        train.assert_called_once()
        kwargs = train.call_args.kwargs
        self.assertTrue(kwargs["include_full_song_context"])
        self.assertTrue(kwargs["skip_first_eval_pass"])
        self.assertFalse(kwargs["precompute_control_teacher_cache"])
        self.assertEqual(kwargs["mps_cleanup_every"], 20)
        self.assertTrue(kwargs["require_control_teacher_cache"])
        self.assertEqual(kwargs["batch_size"], 2)
        self.assertEqual(kwargs["resume_from"], Path("artifacts/runs/stage2_mapper_v2_1/example/checkpoint.pt"))
        self.assertEqual(kwargs["model_config_overrides"]["max_seq_len"], 1024)
        self.assertEqual(kwargs["loss_config_overrides"]["lambda_density"], 0.05)

    def test_cache_only_cli_runs_shared_control_teacher_precompute(self) -> None:
        precompute_result = SimpleNamespace(
            reports=[
                {
                    "split": "source",
                    "total_entries": 1,
                    "computed_entries": 1,
                    "skipped_entries": 0,
                    "elapsed_s": 0.0,
                }
            ],
        )
        with patch.object(
            mapper_v2_1_training,
            "precompute_mapper_tuple_phase_b_control_teacher_cache",
            return_value=precompute_result,
            autospec=True,
        ) as precompute:
            with patch.object(mapper_v2_1_training, "run_mapper_v2_1_phase_b_training", autospec=True) as train:
                mapper_v2_1_training.main(
                    [
                        "--config",
                        "configs/training/stage2_mapper_v2_1_phase_b_sparse_global_mps.yaml",
                        "--precompute-control-teacher-cache-only",
                    ],
                )

        precompute.assert_called_once()
        train.assert_not_called()
        kwargs = precompute.call_args.kwargs
        self.assertEqual(kwargs["control_teacher_cache_dir"], Path("artifacts/cache/stage2_mapper_v2_1/control_teacher_d384_l3_stride16_step002000"))
        self.assertEqual(kwargs["control_teacher_precompute_batch_size"], 12)
        self.assertFalse(kwargs["control_teacher_cache_overwrite"])
        self.assertIn("max_seq_len", load_run_config("configs/training/stage2_mapper_v2_1_phase_b_sparse_global_mps.yaml")["model"])

    def test_resumable_random_sampler_jumps_without_loading_skipped_batches(self) -> None:
        dataset_size = 7
        batch_size = 3
        seed = 923
        completed_batches = 5
        next_batch_count = 6

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
                completed_batches=completed_batches,
            ),
            num_workers=0,
        )

        actual = _take_index_batches(_infinite_loader(resume_loader), next_batch_count)

        self.assertEqual(actual, expected)
        self.assertEqual(len(resume_dataset.loaded_indexes), sum(len(batch) for batch in actual))


def _assert_artifacts_paths(test: unittest.TestCase, config: dict[str, object], keys: tuple[str, ...]) -> None:
    for key in keys:
        value = str(config[key])
        test.assertTrue(value.startswith("artifacts/"), msg=f"{key}={value}")
        test.assertNotIn(_STALE_ROOT, value, msg=f"{key}={value}")


class _IndexDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = int(size)
        self.loaded_indexes: list[int] = []

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> torch.Tensor:
        self.loaded_indexes.append(int(index))
        return torch.tensor(int(index), dtype=torch.long)


def _take_index_batches(iterator: object, count: int) -> list[list[int]]:
    batches: list[list[int]] = []
    for _ in range(count):
        batch = next(iterator)  # type: ignore[arg-type]
        batches.append([int(value) for value in batch.tolist()])
    return batches


if __name__ == "__main__":
    unittest.main()
