import tempfile
import unittest

import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from pulsefield_model.data.control_demo_global_windows import collate_control_demo_global_windows
from pulsefield_model.features.control_v3_targets import CONFIDENCE_FEATURE_NAMES, MODEL_FEATURE_NAMES, VALUE_FEATURE_NAMES
from pulsefield_model.models.control.encoder import ControlEncoder, ControlEncoderConfig
from pulsefield_model.models.control import ControlDemoModelLoss
from pulsefield_model.models.control import ControlDemoGlobalEncoder, ControlDemoGlobalEncoderConfig
from pulsefield_model.training.control_demo_global import (
    _GlobalAttentionBudgetBatchSampler,
    _build_control_demo_global_optimizer,
    _loss_for_raw_batch,
    initialize_global_control_demo_from_control_checkpoint,
    load_run_config,
)

_STALE_ROOT = "train" + "/"


class ControlDemoGlobalTrainingTests(unittest.TestCase):
    def test_global_config_uses_repo_artifacts_paths(self) -> None:
        config = load_run_config("configs/training/stage2_control_demo_global_mps.yaml")

        for key in (
            "index_path",
            "control_v3_timeseries_path",
            "output_dir",
            "init_from_control_checkpoint",
        ):
            value = str(config[key])
            self.assertTrue(value.startswith("artifacts/"), msg=f"{key}={value}")
            self.assertNotIn(_STALE_ROOT, value, msg=f"{key}={value}")

    def test_loss_for_raw_batch_prepares_context_and_uses_full_song_inputs(self) -> None:
        model = ControlDemoGlobalEncoder(_small_config())
        loss_fn = ControlDemoModelLoss()
        raw_batch = collate_control_demo_global_windows(
            [_sample(target_start_frame=300, frame_count=760, density_level=0.25)]
        )

        loss_output = _loss_for_raw_batch(model, loss_fn, raw_batch, device=torch.device("cpu"))

        self.assertEqual(loss_output.total_loss.shape, ())
        self.assertTrue(torch.isfinite(loss_output.total_loss))
        self.assertIn("loss/value", loss_output.metrics)
        self.assertGreater(loss_output.metrics["target/valid_frame_count"], 0)

    def test_optimizer_uses_separate_local_head_and_global_learning_rates(self) -> None:
        model = ControlDemoGlobalEncoder(_small_config())

        optimizer = _build_control_demo_global_optimizer(model, learning_rate=0.004, weight_decay=0.01)

        groups = {group["name"]: group for group in optimizer.param_groups}
        self.assertEqual(set(groups), {"local_pretrained", "value_head", "global_path"})
        self.assertAlmostEqual(groups["local_pretrained"]["lr"], 0.001)
        self.assertAlmostEqual(groups["value_head"]["lr"], 0.002)
        self.assertAlmostEqual(groups["global_path"]["lr"], 0.004)

    def test_checkpoint_init_loads_local_path_and_leaves_global_path_random(self) -> None:
        torch.manual_seed(23)
        full_model = ControlEncoder(
            ControlEncoderConfig(d_model=32, heads=4, layers=3, ffn_dim=64, dropout=0.0, conv_blocks=1)
        )
        density_index = VALUE_FEATURE_NAMES.index("density_level")
        with torch.no_grad():
            full_model.value_head.weight[density_index].fill_(0.125)
            full_model.value_head.bias[density_index].fill_(0.75)

        global_model = ControlDemoGlobalEncoder(_small_config())
        assert global_model.global_encoder is not None
        global_projection_before = global_model.global_encoder.projection.weight.detach().clone()
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            torch.save({"model_state_dict": full_model.state_dict()}, checkpoint_path)

            report = initialize_global_control_demo_from_control_checkpoint(global_model, checkpoint_path)

        self.assertGreater(report["loaded_key_count"], 0)
        self.assertGreater(report["missing_global_key_count"], 0)
        self.assertTrue(report["density_head_copied"])
        self.assertTrue(torch.equal(global_model.position, full_model.position))
        self.assertTrue(torch.equal(global_model.value_head.weight[0], full_model.value_head.weight[density_index]))
        self.assertEqual(float(global_model.value_head.bias[0].item()), 0.75)
        assert global_model.global_encoder is not None
        self.assertTrue(torch.equal(global_model.global_encoder.projection.weight, global_projection_before))

    def test_global_attention_budget_sampler_splits_long_song_batches(self) -> None:
        dataset = _FrameCountDataset([100, 100, 300, 100, 100])

        sampler = _GlobalAttentionBudgetBatchSampler(
            dataset,
            batch_size=4,
            global_attention_budget=400,
            global_stride=10,
            shuffle=False,
            seed=1337,
        )

        self.assertEqual(list(iter(sampler)), [[0, 1], [2], [3, 4]])


def _small_config() -> ControlDemoGlobalEncoderConfig:
    return ControlDemoGlobalEncoderConfig(
        d_model=32,
        heads=4,
        layers=3,
        ffn_dim=64,
        dropout=0.0,
        conv_blocks=1,
        use_global_memory=True,
        global_stride=16,
        global_layers=1,
        global_ffn_dim=64,
        global_conv_blocks=1,
        global_fusion_start_layer=1,
    )


class _FrameCountDataset(Dataset):
    def __init__(self, frame_counts: list[int]) -> None:
        self.records = [SimpleNamespace(frame_count=frame_count) for frame_count in frame_counts]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"frame_count": torch.tensor(self.records[index].frame_count)}


def _sample(*, target_start_frame: int, frame_count: int, density_level: float) -> dict[str, object]:
    target = torch.zeros(100, len(MODEL_FEATURE_NAMES), dtype=torch.float32)
    target[:, MODEL_FEATURE_NAMES.index("density_level")] = density_level
    target[:, MODEL_FEATURE_NAMES.index("density_confidence")] = 1.0
    for name in CONFIDENCE_FEATURE_NAMES:
        if name != "density_confidence":
            target[:, MODEL_FEATURE_NAMES.index(name)] = 1.0
    return {
        "full_mel": torch.randn(frame_count, 160, dtype=torch.float32) * 0.05,
        "full_dense_timing_v2": torch.randn(frame_count, 4, dtype=torch.float32) * 0.05,
        "control_v3_target": target,
        "ln_change_n_eff_target": torch.full((100,), 3.0, dtype=torch.float32),
        "target_valid_mask": target_start_frame + torch.arange(100) < frame_count,
        "difficulty": torch.tensor(3.0, dtype=torch.float32),
        "normalized_difficulty": torch.tensor(0.0, dtype=torch.float32),
        "target_start_frame": torch.tensor(target_start_frame, dtype=torch.long),
        "target_start_ms": torch.tensor(target_start_frame * 20, dtype=torch.long),
        "frame_count": torch.tensor(frame_count, dtype=torch.long),
        "beatmap_path": "demo.osu",
        "audio_path": "demo.mp3",
    }


if __name__ == "__main__":
    unittest.main()
