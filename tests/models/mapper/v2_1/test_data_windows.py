import unittest

import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")
from pathlib import Path

import torch

from pulsefield_model.data.control_windows import ControlWindowRecord, normalize_difficulty
from pulsefield_model.data.mapper_sparse_windows_v2_1 import MapperV21WindowDataset, collate_mapper_v2_1_windows
from pulsefield_model.features.control_v3_targets import MODEL_FEATURE_NAMES
from pulsefield_model.models.mapper.v2_1.replay import NO_EMITTED_LANE_INDEX, ln_carry_state_tensors
from pulsefield_model.models.mapper.v2_1.tokenizer import MapperTimepoint, encode_mapper_window
from pulsefield_model.models.mapper.v2_1.vocab import LaneAction, MapperV21Vocab


def _actions(*actions: LaneAction) -> tuple[LaneAction, ...]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(LaneAction.NONE)
    return tuple(padded)


class MapperV21DataWindowTests(unittest.TestCase):
    def test_dataset_and_collate_preserve_sparse_lane_state(self) -> None:
        vocab = MapperV21Vocab()
        dataset = _MapperV21DatasetWithFullInputs(
            [_record("sparse.osu", difficulty=4.0)],
            timepoints_by_path={
                "sparse.osu": (
                    MapperTimepoint(1000, _actions(LaneAction.TAP, LaneAction.NONE, LaneAction.TAP)),
                ),
            },
        )
        sparse_sample = dataset[0]
        empty_sample = _sample(
            encode_mapper_window([], vocab=vocab, write_start_ms=8000, write_end_ms=16000),
        )

        batch = collate_mapper_v2_1_windows([sparse_sample, empty_sample], pad_id=vocab.pad_id)

        token_names = [vocab.token_name(token_id) for token_id in sparse_sample["target_fragment_tokens"][:3].tolist()]
        self.assertEqual(token_names, ["TS_1000", "LANE_1_TAP", "LANE_3_TAP"])
        self.assertEqual(int(sparse_sample["chart_end_ms"].item()), 1000)
        self.assertTrue(sparse_sample["is_full_chart_end"].item())
        self.assertEqual(vocab.token_name(int(sparse_sample["target_fragment_tokens"][-1].item())), "EOS")
        self.assertTrue(sparse_sample["target_fragment_states"]["emitted_lane_mask"][2, 0].item())
        self.assertEqual(int(sparse_sample["target_fragment_states"]["last_lane_index"][2].item()), 0)

        states = batch["target_fragment_states"]
        self.assertEqual(states["emitted_lane_mask"].shape, (2, sparse_sample["target_fragment_tokens"].shape[0], 4))
        self.assertTrue(states["emitted_lane_mask"][0, 2, 0].item())
        self.assertEqual(int(states["last_lane_index"][0, 2].item()), 0)
        self.assertFalse(states["emitted_lane_mask"][1, -1].any().item())
        self.assertEqual(int(states["last_lane_index"][1, -1].item()), NO_EMITTED_LANE_INDEX)
        self.assertFalse(batch["target_fragment_mask"][1, -1].item())
        self.assertEqual(tuple(batch["density_target_8s"].shape), (2, 400, 1))


def _sample(tokenized) -> dict:
    return {
        "mel_context": torch.zeros(400, 160, dtype=torch.float32),
        "timing_context": torch.zeros(400, 4, dtype=torch.float32),
        "context_padding_mask": torch.zeros(400, dtype=torch.bool),
        "difficulty": torch.zeros(1, dtype=torch.float32),
        "decoder_input_tokens": tokenized.decoder_input_tensor(),
        "target_fragment_tokens": tokenized.target_fragment_tensor(),
        "target_fragment_states": {
            "current_ms": tokenized.target_fragment_current_ms,
            "open_mask": tokenized.target_fragment_open_mask,
            "open_start_ms": tokenized.target_fragment_open_start_ms,
            "open_age_ms": tokenized.target_fragment_open_age_ms,
            "emitted_lane_mask": tokenized.target_fragment_emitted_lane_mask,
            "last_lane_index": tokenized.target_fragment_last_lane_index,
        },
        "ln_carry_in": ln_carry_state_tensors(tokenized.ln_carry_in),
        "ln_carry_out": ln_carry_state_tensors(tokenized.ln_carry_out),
        "close_labels": tokenized.close_labels,
        "close_label_mask": tokenized.close_label_mask,
        "density_target_8s": torch.zeros(400, 1, dtype=torch.float32),
        "density_confidence_8s": torch.ones(400, 1, dtype=torch.float32),
        "write_start_ms": torch.tensor(tokenized.write_start_ms, dtype=torch.long),
        "write_end_ms": torch.tensor(tokenized.write_end_ms, dtype=torch.long),
        "chart_end_ms": torch.tensor(tokenized.chart_end_ms, dtype=torch.long),
        "is_full_chart_start": torch.tensor(tokenized.is_full_chart_start, dtype=torch.bool),
        "is_full_chart_end": torch.tensor(tokenized.is_full_chart_end, dtype=torch.bool),
    }


def _record(
    beatmap_path: str,
    *,
    difficulty: float,
    frame_count: int = 400,
    target_start_frame: int = 0,
) -> ControlWindowRecord:
    return ControlWindowRecord(
        beatmap_path=Path(beatmap_path),
        audio_path=Path(f"{beatmap_path}.mp3"),
        difficulty=difficulty,
        frame_count=frame_count,
        target_start_frame=target_start_frame,
    )


class _FullInputControlDataset:
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
        target[:, MODEL_FEATURE_NAMES.index("density_level")] = 0.5
        target[:, MODEL_FEATURE_NAMES.index("density_confidence")] = 1.0
        return target


class _MapperV21DatasetWithFullInputs(MapperV21WindowDataset):
    def __init__(
        self,
        records: list[ControlWindowRecord],
        *,
        timepoints_by_path: dict[str, tuple[MapperTimepoint, ...]],
    ) -> None:
        self.timepoints_by_path = timepoints_by_path
        super().__init__(control_dataset=_FullInputControlDataset(records))

    def _load_timepoints(self, beatmap_path: Path) -> tuple:
        return self.timepoints_by_path.get(beatmap_path.as_posix(), ())


if __name__ == "__main__":
    unittest.main()
