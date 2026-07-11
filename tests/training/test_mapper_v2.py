import tempfile
import unittest

import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

from pulsefield_model.models.mapper.v2 import MapperV2Config, MapperV2Model
from pulsefield_model.training import mapper_v2 as mapper_v2_training
from pulsefield_model.training.hydra_config import (
    compose_training_experiment_config,
    training_experiment_config_to_legacy_dict,
)
from pulsefield_model.training.mapper_v2 import initialize_mapper_v2_from_mapper_checkpoint

_STALE_ROOT = "train" + "/"


def _load_mapper_v2_preset(preset: str) -> dict[str, Any]:
    config = compose_training_experiment_config(overrides=[f"training/mapper={preset}"])
    return training_experiment_config_to_legacy_dict(config)


class MapperV2PhaseBTrainingTests(unittest.TestCase):
    def test_phase_b_global_preset_loads_v2_fields(self) -> None:
        config = _load_mapper_v2_preset("v2_tuple_d384_l4_phase_b")

        self.assertTrue(config["include_full_song_context"])
        self.assertTrue(config["skip_first_eval_pass"])
        self.assertTrue(config["precompute_control_teacher_cache"])
        self.assertTrue(config["model"]["use_global_context"])
        self.assertEqual(config["model"]["global_stride"], 16)
        self.assertEqual(config["model"]["global_layers"], 1)
        self.assertEqual(config["batch_size"], 2)
        self.assertIn("stage2_control_windows", config["index_path"])
        self.assertIn("stage2_mapper_v2/window_records", config["mapper_record_cache_path"])
        self.assertIn("stage2_mapper_v2/control_teacher", config["control_teacher_cache_dir"])
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
        MapperV2Config(**config["model"])

    def test_phase_b_large_global_preset_loads_v2_fields(self) -> None:
        config = _load_mapper_v2_preset("v2_tuple_d768_l8_phase_b")

        self.assertTrue(config["include_full_song_context"])
        self.assertTrue(config["skip_first_eval_pass"])
        self.assertTrue(config["precompute_control_teacher_cache"])
        self.assertEqual(config["model"]["d_model"], 768)
        self.assertEqual(config["model"]["heads"], 12)
        self.assertEqual(config["model"]["layers"], 8)
        self.assertEqual(config["model"]["ffn_dim"], 3072)
        self.assertEqual(config["mps_cleanup_every"], 20)
        self.assertTrue(config["resume_from"].endswith("checkpoint.pt"))
        self.assertIn("stage2_control_windows", config["index_path"])
        self.assertIn("stage2_mapper_v2/window_records", config["mapper_record_cache_path"])
        self.assertIn("stage2_mapper_v2/control_teacher", config["control_teacher_cache_dir"])
        self.assertIn("d768_l8", config["mapper_record_cache_path"])
        _assert_artifacts_paths(
            self,
            config,
            (
                "index_path",
                "control_v3_timeseries_path",
                "output_dir",
                "resume_from",
                "init_from_control_checkpoint",
                "mapper_record_cache_path",
                "control_teacher_cache_dir",
            ),
        )
        MapperV2Config(**config["model"])

    def test_main_forwards_v2_training_options(self) -> None:
        train_result = SimpleNamespace(
            report_path=Path("report.json"),
            checkpoint_path=Path("checkpoint.pt"),
            final_loss=0.0,
            completed_steps=0,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                mapper_v2_training,
                "run_mapper_v2_phase_b_training",
                return_value=train_result,
                autospec=True,
            ) as train:
                mapper_v2_training.main(
                    [
                        "run.max_steps=1",
                        f"output.output_dir={(Path(temp_dir) / 'run').as_posix()}",
                    ]
                )

        train.assert_called_once()
        kwargs = train.call_args.kwargs
        self.assertTrue(kwargs["include_full_song_context"])
        self.assertTrue(kwargs["skip_first_eval_pass"])
        self.assertTrue(kwargs["precompute_control_teacher_cache"])
        self.assertEqual(kwargs["batch_size"], 2)
        self.assertTrue(kwargs["model_config_overrides"]["use_global_context"])
        self.assertEqual(kwargs["model_config_overrides"]["global_gate_init"], -2.94)

    def test_main_rejects_mapper_group_override(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            mapper_v2_training.main(["training/mapper=v2_1_sparse_d384_l4_phase_b", "--dry-run"])
        self.assertEqual(raised.exception.code, 2)

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
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(
                mapper_v2_training,
                "precompute_mapper_tuple_phase_b_control_teacher_cache",
                return_value=precompute_result,
                autospec=True,
            ) as precompute:
                with patch.object(mapper_v2_training, "run_mapper_v2_phase_b_training", autospec=True) as train:
                    mapper_v2_training.main(
                        [
                            "data.precompute_control_teacher_cache_only=true",
                            f"output.output_dir={(Path(temp_dir) / 'run').as_posix()}",
                            f"data.control_teacher_cache_dir={Path(temp_dir) / 'cache'}",
                        ]
                    )

        precompute.assert_called_once()
        train.assert_not_called()
        self.assertIn("control_model_config_overrides", precompute.call_args.kwargs)

    def test_initialize_mapper_v2_checkpoint_loads_state_only(self) -> None:
        config = MapperV2Config(
            control_dim=16,
            d_model=16,
            heads=4,
            layers=1,
            ffn_dim=32,
            dropout=0.0,
            max_seq_len=16,
            state_hidden_dim=16,
            ln_close_hidden_dim=16,
            global_stride=16,
            global_layers=1,
            global_ffn_dim=32,
            global_conv_blocks=0,
        )
        source = MapperV2Model(config)
        with torch.no_grad():
            for index, parameter in enumerate(source.parameters()):
                parameter.fill_(0.01 * (index + 1))
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "mapper_v2.pt"
            torch.save(
                {
                    "checkpoint_schema_version": mapper_v2_training.CHECKPOINT_SCHEMA_VERSION,
                    "model_state_dict": source.state_dict(),
                    "optimizer_state_dict": {"state": {"ignored": torch.ones(1)}},
                    "model_config": config.__dict__,
                    "control_model_config": None,
                    "training_state": {"step": 17},
                },
                checkpoint_path,
            )
            target = MapperV2Model(config)

            report = initialize_mapper_v2_from_mapper_checkpoint(
                target,
                checkpoint_path,
                expected_model_config=config,
                expected_control_model_config=None,
            )

        self.assertEqual(report["kind"], "mapper_v2_model_state")
        self.assertEqual(report["checkpoint_step"], 17)
        self.assertFalse(report["optimizer_state_loaded"])
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(target.state_dict()[key], value), key)

def _assert_artifacts_paths(test: unittest.TestCase, config: dict[str, object], keys: tuple[str, ...]) -> None:
    for key in keys:
        value = str(config[key])
        test.assertTrue(value.startswith("artifacts/"), msg=f"{key}={value}")
        test.assertNotIn(_STALE_ROOT, value, msg=f"{key}={value}")


if __name__ == "__main__":
    unittest.main()
