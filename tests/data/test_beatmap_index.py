import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path

import pandas as pd

from pulsefield_model.data.beatmap_index import (
    build_4k_index,
    build_4k_no_timing_anomaly_index,
    build_dense_timing_v2_local_bpm_norm_unique_index,
    build_difficulty_filtered_index,
)


class BeatmapIndexBuilderTests(unittest.TestCase):
    def test_builds_filtered_training_indexes_from_dataset_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_root = root / "dataset"
            beatmap_set = dataset_root / "0" / "1001"
            beatmap_set.mkdir(parents=True)
            _write_wav(beatmap_set / "audio.wav")
            _write_osu(beatmap_set / "good.osu", beatmap_id=10001, include_timing=True)
            _write_osu(beatmap_set / "missing_timing.osu", beatmap_id=10002, include_timing=False)

            raw_index = root / "artifacts/indexes/beatmap_index_4k.parquet"
            clean_index = root / "artifacts/indexes/beatmap_index_4k_no_timing_anomalies.parquet"
            difficulty_index = root / "artifacts/indexes/beatmap_index_4k_no_timing_anomalies_2to6.parquet"
            local_bpm_index = (
                root
                / "artifacts/indexes/beatmap_index_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.parquet"
            )
            local_bpm_report = root / "artifacts/reports/indexes/local_bpm.json"

            build_4k_index(dataset_root / "0", raw_index)
            timing_report = build_4k_no_timing_anomaly_index(
                source_index_path=raw_index,
                dataset_root=dataset_root,
                output_path=clean_index,
            )
            difficulty_report = build_difficulty_filtered_index(
                source_index_path=clean_index,
                output_path=difficulty_index,
            )
            local_bpm_report_data = build_dense_timing_v2_local_bpm_norm_unique_index(
                source_index_path=difficulty_index,
                dataset_root=dataset_root,
                output_path=local_bpm_index,
                report_path=local_bpm_report,
            )

            raw_df = pd.read_parquet(raw_index)
            clean_df = pd.read_parquet(clean_index)
            difficulty_df = pd.read_parquet(difficulty_index)
            local_bpm_df = pd.read_parquet(local_bpm_index)
            local_bpm_report_exists = local_bpm_report.exists()

        self.assertEqual(len(raw_df), 2)
        self.assertEqual(timing_report.clean_map_count, 1)
        self.assertEqual(timing_report.missing_red_timing_map_count, 1)
        self.assertEqual(difficulty_report.retained_map_count, 1)
        self.assertEqual(local_bpm_report_data.retained_map_count, 1)
        self.assertTrue(local_bpm_report_exists)
        self.assertEqual(clean_df["beatmap_path"].tolist(), ["1001/good.osu"])
        self.assertEqual(difficulty_df["beatmap_path"].tolist(), ["1001/good.osu"])
        self.assertEqual(local_bpm_df["beatmap_path"].tolist(), ["1001/good.osu"])
        self.assertGreater(float(local_bpm_df["difficulty"].iloc[0]), 2.0)
        self.assertLess(float(local_bpm_df["difficulty"].iloc[0]), 6.0)


def _write_wav(path: Path, *, seconds: int = 10, sample_rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        for index in range(seconds * sample_rate):
            sample = int(12000 * math.sin(2.0 * math.pi * 440.0 * index / sample_rate))
            handle.writeframes(struct.pack("<h", sample))


def _write_osu(path: Path, *, beatmap_id: int, include_timing: bool) -> None:
    lines = [
        "osu file format v14",
        "",
        "[General]",
        "AudioFilename: audio.wav",
        "Mode: 3",
        "",
        "[Metadata]",
        "Title: Fixture",
        "Artist: Test",
        "Creator: Tests",
        f"Version: {path.stem}",
        f"BeatmapID: {beatmap_id}",
        "BeatmapSetID: 1001",
        "",
        "[Difficulty]",
        "HPDrainRate:5",
        "CircleSize:4",
        "OverallDifficulty:5",
        "",
        "[TimingPoints]",
    ]
    if include_timing:
        lines.append("0,500,4,2,1,50,1,0")
    lines.extend(["", "[HitObjects]"])
    for timestamp in range(0, 8000, 200):
        for x in (64, 192, 320, 448):
            lines.append(f"{x},192,{timestamp},1,0,0:0:0:0:")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
