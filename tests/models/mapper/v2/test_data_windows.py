import tempfile
import unittest

import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")
from pathlib import Path

import torch

from pulsefield_model.data.control_windows import ControlWindowRecord, normalize_difficulty
from pulsefield_model.data.mapper_tuple_windows import (
    MapperTupleWindowDataset,
    collate_mapper_tuple_windows,
    control_teacher_cache_path,
    save_control_teacher_cache_entry,
)
from pulsefield_model.features.control_v3_targets import MODEL_FEATURE_NAMES


class MapperV2DataWindowTests(unittest.TestCase):
    def test_cached_teacher_sample_can_include_full_song_context(self) -> None:
        record = ControlWindowRecord(
            beatmap_path=Path("tail.osu"),
            audio_path=Path("tail.mp3"),
            difficulty=4.0,
            frame_count=450,
            target_start_frame=400,
        )
        control_memory = torch.zeros(400, 6, dtype=torch.float32)
        density_teacher = torch.zeros(400, 1, dtype=torch.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = control_teacher_cache_path(temp_dir, record)
            save_control_teacher_cache_entry(
                cache_path,
                record=record,
                control_memory_8s=control_memory,
                density_teacher_8s=density_teacher,
            )
            dataset = _MapperV2ContextDataset([record], cache_dir=Path(temp_dir))

            sample = dataset[0]
            batch = collate_mapper_tuple_windows([sample])

        self.assertIn("control_memory_8s", sample)
        self.assertIn("density_teacher_8s", sample)
        self.assertIn("full_mel", sample)
        self.assertEqual(tuple(sample["full_mel"].shape), (800, 160))
        self.assertEqual(tuple(sample["full_dense_timing_v2"].shape), (800, 4))
        self.assertEqual(int(sample["frame_count"].item()), 800)
        self.assertEqual(int(sample["source_frame_count"].item()), 450)
        self.assertEqual(int(sample["target_start_frame"].item()), 400)
        self.assertFalse(batch["padding_mask"][0, :450].any().item())
        self.assertTrue(batch["padding_mask"][0, 450:].all().item())
        self.assertEqual(batch["target_start_frame"].tolist(), [400])
        self.assertEqual(tuple(batch["control_memory_8s"].shape), (1, 400, 6))
        self.assertEqual(tuple(batch["full_mel"].shape), (1, 800, 160))


class _FullContextControlDataset:
    def __init__(self, records: list[ControlWindowRecord]) -> None:
        self.records = records

    def __getitem__(self, index: int):
        record = self.records[index]
        return {
            "full_mel": torch.ones(record.frame_count, 160, dtype=torch.float32),
            "full_dense_timing_v2": torch.ones(record.frame_count, 4, dtype=torch.float32),
            "frame_count": torch.tensor(record.frame_count, dtype=torch.long),
            "target_start_frame": torch.tensor(record.target_start_frame, dtype=torch.long),
            "difficulty": torch.tensor(record.difficulty, dtype=torch.float32),
            "normalized_difficulty": torch.tensor(normalize_difficulty(record.difficulty), dtype=torch.float32),
        }

    def target_loader(self, record: ControlWindowRecord) -> torch.Tensor:
        target = torch.zeros(100, len(MODEL_FEATURE_NAMES), dtype=torch.float32)
        target[:, MODEL_FEATURE_NAMES.index("density_confidence")] = 1.0
        return target


class _MapperV2ContextDataset(MapperTupleWindowDataset):
    def __init__(self, records: list[ControlWindowRecord], *, cache_dir: Path) -> None:
        super().__init__(
            control_dataset=_FullContextControlDataset(records),
            control_teacher_cache_dir=cache_dir,
            include_full_song_context=True,
        )

    def _load_timepoints(self, beatmap_path: Path) -> tuple:
        return ()


if __name__ == "__main__":
    unittest.main()
