import unittest

import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")

import torch

from pulsefield_model.data.control_demo_global_windows import (
    CONTROL_DEMO_TARGET_FEATURE_NAMES,
    collate_control_demo_global_windows,
)
from pulsefield_model.features.control_v3_targets import CONFIDENCE_FEATURE_NAMES, MODEL_FEATURE_NAMES


class ControlDemoGlobalWindowTests(unittest.TestCase):
    def test_collate_preserves_full_song_tensors_and_extracts_density_target(self) -> None:
        batch = collate_control_demo_global_windows(
            [
                _sample(target_start_frame=0, frame_count=260, density_offset=0.0),
                _sample(target_start_frame=100, frame_count=180, density_offset=1.0),
            ]
        )

        self.assertNotIn("control_v3_target", batch)
        self.assertNotIn("ln_change_n_eff_target", batch)
        for key in (
            "full_mel",
            "full_dense_timing_v2",
            "padding_mask",
            "frame_count",
            "target_start_frame",
            "normalized_difficulty",
            "control_demo_target",
        ):
            self.assertIn(key, batch)

        self.assertEqual(batch["full_mel"].shape, (2, 260, 160))
        self.assertEqual(batch["full_dense_timing_v2"].shape, (2, 260, 4))
        self.assertEqual(batch["padding_mask"].shape, (2, 260))
        self.assertFalse(batch["padding_mask"][0].any())
        self.assertFalse(batch["padding_mask"][1, :180].any())
        self.assertTrue(batch["padding_mask"][1, 180:].all())
        self.assertEqual(batch["frame_count"].tolist(), [260, 180])
        self.assertEqual(batch["target_start_frame"].tolist(), [0, 100])
        self.assertEqual(batch["control_demo_target"].shape, (2, 100, 2))
        self.assertEqual(CONTROL_DEMO_TARGET_FEATURE_NAMES, ("density_level", "density_confidence"))
        self.assertTrue(torch.equal(batch["control_demo_target"][0, :, 0], torch.linspace(0.0, 1.0, 100)))
        self.assertTrue(torch.equal(batch["control_demo_target"][1, :, 0], torch.linspace(1.0, 2.0, 100)))
        self.assertTrue(torch.equal(batch["control_demo_target"][:, :, 1], torch.full((2, 100), 0.25)))


def _sample(*, target_start_frame: int, frame_count: int, density_offset: float) -> dict[str, object]:
    target = torch.zeros(100, len(MODEL_FEATURE_NAMES), dtype=torch.float32)
    target[:, MODEL_FEATURE_NAMES.index("density_level")] = torch.linspace(
        density_offset,
        density_offset + 1.0,
        100,
    )
    target[:, MODEL_FEATURE_NAMES.index("density_confidence")] = 0.25
    for name in CONFIDENCE_FEATURE_NAMES:
        if name != "density_confidence":
            target[:, MODEL_FEATURE_NAMES.index(name)] = 1.0
    return {
        "full_mel": torch.zeros(frame_count, 160, dtype=torch.float32),
        "full_dense_timing_v2": torch.zeros(frame_count, 4, dtype=torch.float32),
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
