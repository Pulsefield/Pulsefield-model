import unittest

import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")
from pathlib import Path

from torch.utils.data import Dataset

from pulsefield_model.data.control_windows import ControlWindowRecord
from pulsefield_model.data.mapper_tuple_windows import MapperTupleWindowRecord
from pulsefield_model.training.common import split_train_eval_dataset


class TrainEvalSplitTests(unittest.TestCase):
    def test_mapper_wrapped_records_split_whole_maps(self) -> None:
        records: list[MapperTupleWindowRecord] = []
        for map_index, beatmap_name in enumerate(("low.osu", "mid.osu", "high.osu")):
            for window_index in range(2):
                control_record = _control_record(
                    beatmap_name,
                    difficulty=2.5 + map_index,
                    target_start_frame=window_index * 400,
                )
                records.append(
                    MapperTupleWindowRecord(
                        control_record_index=len(records),
                        control_record=control_record,
                        target_seq_len=8,
                    )
                )
        dataset = _RecordDataset(records)

        train_dataset, eval_dataset = split_train_eval_dataset(
            dataset,
            eval_fraction=0.5,
            eval_size=3,
            seed=1337,
        )

        train_paths = _mapper_beatmap_paths(dataset, train_dataset.indices)
        eval_paths = _mapper_beatmap_paths(dataset, eval_dataset.indices)
        self.assertTrue(eval_paths)
        self.assertFalse(train_paths & eval_paths)

    def test_map_split_stratifies_eval_by_difficulty_bucket(self) -> None:
        records: list[ControlWindowRecord] = []
        for map_index in range(5):
            records.extend(_map_windows(f"low_{map_index}.osu", difficulty=2.25))
        for map_index in range(5):
            records.extend(_map_windows(f"high_{map_index}.osu", difficulty=5.25))
        dataset = _RecordDataset(records)

        _train_dataset, eval_dataset = split_train_eval_dataset(
            dataset,
            eval_fraction=0.2,
            eval_size=4,
            seed=1337,
        )

        eval_difficulties = [dataset.records[index].difficulty for index in eval_dataset.indices]
        self.assertEqual(sum(difficulty < 4.0 for difficulty in eval_difficulties), 2)
        self.assertEqual(sum(difficulty >= 4.0 for difficulty in eval_difficulties), 2)


class _RecordDataset(Dataset):
    def __init__(self, records: list[object]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> object:
        return self.records[index]


def _map_windows(beatmap_path: str, *, difficulty: float) -> list[ControlWindowRecord]:
    return [
        _control_record(beatmap_path, difficulty=difficulty, target_start_frame=0),
        _control_record(beatmap_path, difficulty=difficulty, target_start_frame=400),
    ]


def _control_record(beatmap_path: str, *, difficulty: float, target_start_frame: int) -> ControlWindowRecord:
    return ControlWindowRecord(
        beatmap_path=Path(beatmap_path),
        audio_path=Path(f"{beatmap_path}.mp3"),
        difficulty=difficulty,
        frame_count=1000,
        target_start_frame=target_start_frame,
    )


def _mapper_beatmap_paths(dataset: _RecordDataset, indices: list[int]) -> set[Path]:
    return {dataset.records[index].control_record.beatmap_path for index in indices}


if __name__ == "__main__":
    unittest.main()
