import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pulsefield_model.timing.evaluation.inventory import (
    TIMING_V3_INVENTORY_AUDIO_ROW_SCHEMA,
    TIMING_V3_INVENTORY_REPORT_SCHEMA,
    build_timing_v3_inventory,
    main,
)
from pulsefield_model.timing.providers.beatthis import BEATTHIS_FRAME_RATE_HZ, BEATTHIS_PROVIDER_NAME
from pulsefield_model.timing.providers.beatthis_cache import (
    BeatThisFramePredictionCacheConfig,
    beatthis_audio_cache_key,
    beatthis_frame_prediction_cache_path,
    save_beatthis_frame_prediction_cache,
)
from pulsefield_model.timing.schema import FrameTimingPrediction


class TimingV3InventoryTests(unittest.TestCase):
    def test_groups_by_resolved_audio_and_records_cache_maps_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            beatmapset = dataset_root / "0" / "1001"
            audio_path = beatmapset / "audio.wav"
            _write_file(audio_path, b"audio")
            _write_file(beatmapset / "a.osu", b"osu a")
            _write_file(beatmapset / "b.osu", b"osu b")
            metadata_path = beatmapset / "metadata.json"
            metadata_path.write_text('{"schema_version":1}\n', encoding="utf-8")
            index_path = _write_index(
                root,
                [
                    _index_row("0", "1001/audio.wav", "1001/b.osu", beatmap_id=102),
                    _index_row("0", "1001/audio.wav", "1001/a.osu", beatmap_id=101),
                ],
            )
            cache_root = root / "cache"
            config = BeatThisFramePredictionCacheConfig(cache_root=cache_root, checkpoint_path="checkpoint-a")
            save_beatthis_frame_prediction_cache(
                _prediction(checkpoint_path="checkpoint-a", source_path=audio_path),
                beatthis_audio_cache_key(audio_path),
                config,
            )
            report_path = root / "report.json"
            audio_rows_path = root / "audio_rows.jsonl"

            first = build_timing_v3_inventory(
                index_path=index_path,
                dataset_root=dataset_root,
                cache_root=cache_root,
                checkpoint_path="checkpoint-a",
                report_path=report_path,
                audio_rows_path=audio_rows_path,
            )
            first_report_bytes = report_path.read_bytes()
            first_rows_bytes = audio_rows_path.read_bytes()
            second = build_timing_v3_inventory(
                index_path=index_path,
                dataset_root=dataset_root,
                cache_root=cache_root,
                checkpoint_path="checkpoint-a",
                report_path=report_path,
                audio_rows_path=audio_rows_path,
            )

            self.assertEqual(first["schema"], TIMING_V3_INVENTORY_REPORT_SCHEMA)
            self.assertEqual(first["source"]["map_row_count"], 2)
            self.assertEqual(first["source"]["audio_group_count"], 1)
            self.assertEqual(first["cache"]["valid_count"], 1)
            self.assertEqual(first["cache"]["missing_count"], 0)
            self.assertEqual(first, second)
            self.assertEqual(report_path.read_bytes(), first_report_bytes)
            self.assertEqual(audio_rows_path.read_bytes(), first_rows_bytes)
            self.assertFalse(report_path.with_name(report_path.name + ".tmp").exists())
            self.assertFalse(audio_rows_path.with_name(audio_rows_path.name + ".tmp").exists())

            rows = _read_jsonl(audio_rows_path)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["schema"], TIMING_V3_INVENTORY_AUDIO_ROW_SCHEMA)
            self.assertEqual(row["resolved_audio_path"], audio_path.resolve().as_posix())
            self.assertEqual(row["map_count"], 2)
            self.assertEqual([item["beatmap_path"] for item in row["maps"]], ["1001/a.osu", "1001/b.osu"])
            self.assertEqual(row["metadata_json"]["existing_count"], 1)
            self.assertEqual(row["metadata_json"]["paths"][0]["path"], metadata_path.resolve().as_posix())
            self.assertIsNotNone(row["metadata_json"]["paths"][0]["sha256"])
            self.assertEqual(row["cache"]["status"], "valid")
            self.assertEqual(row["cache"]["frame_count"], 4)
            self.assertAlmostEqual(row["cache"]["duration_seconds"], 4 / BEATTHIS_FRAME_RATE_HZ)
            self.assertTrue(Path(row["cache"]["cache_path"]).is_file())
            self.assertEqual(row["anomalies"], [])

    def test_limit_is_applied_after_audio_grouping_in_stable_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            first_audio = dataset_root / "0" / "1001" / "audio.wav"
            second_audio = dataset_root / "0" / "1002" / "audio.wav"
            for path in (first_audio, second_audio):
                _write_file(path, b"audio")
                _write_file(path.with_name("map.osu"), b"osu")
                path.with_name("metadata.json").write_text("{}\n", encoding="utf-8")
            index_path = _write_index(
                root,
                [
                    _index_row("0", "1002/audio.wav", "1002/map.osu", beatmap_id=202),
                    _index_row("0", "1001/audio.wav", "1001/map.osu", beatmap_id=101),
                ],
            )
            cache_root = root / "cache"
            for audio_path in (first_audio, second_audio):
                config = BeatThisFramePredictionCacheConfig(cache_root=cache_root, checkpoint_path="checkpoint-a")
                save_beatthis_frame_prediction_cache(
                    _prediction(checkpoint_path="checkpoint-a", source_path=audio_path),
                    beatthis_audio_cache_key(audio_path),
                    config,
                )

            report = build_timing_v3_inventory(
                index_path=index_path,
                dataset_root=dataset_root,
                cache_root=cache_root,
                checkpoint_path="checkpoint-a",
                limit=1,
                report_path=root / "report.json",
                audio_rows_path=root / "rows.jsonl",
            )

            rows = _read_jsonl(root / "rows.jsonl")
            self.assertEqual(report["source"]["audio_group_count"], 2)
            self.assertEqual(report["output"]["audio_group_count"], 1)
            self.assertEqual(rows[0]["resolved_audio_path"], first_audio.resolve().as_posix())

    def test_reports_duplicate_ids_and_resolved_audio_grouping_anomalies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            beatmapset = dataset_root / "0" / "1001"
            audio_path = beatmapset / "audio.wav"
            alias_path = beatmapset / "alias.wav"
            _write_file(audio_path, b"audio")
            alias_path.symlink_to(audio_path.name)
            _write_file(beatmapset / "a.osu", b"osu a")
            _write_file(beatmapset / "b.osu", b"osu b")
            beatmapset.joinpath("metadata.json").write_text("{}\n", encoding="utf-8")
            index_path = _write_index(
                root,
                [
                    _index_row("0", "1001/audio.wav", "1001/a.osu", beatmap_id=101),
                    _index_row("0", "1001/alias.wav", "1001/b.osu", beatmap_id=101),
                ],
            )
            cache_root = root / "cache"
            config = BeatThisFramePredictionCacheConfig(cache_root=cache_root, checkpoint_path="checkpoint-a")
            save_beatthis_frame_prediction_cache(
                _prediction(checkpoint_path="checkpoint-a", source_path=audio_path),
                beatthis_audio_cache_key(audio_path),
                config,
            )

            report = build_timing_v3_inventory(
                index_path=index_path,
                dataset_root=dataset_root,
                cache_root=cache_root,
                checkpoint_path="checkpoint-a",
                report_path=root / "report.json",
                audio_rows_path=root / "rows.jsonl",
            )

            row = _read_jsonl(root / "rows.jsonl")[0]
            self.assertEqual(report["source"]["duplicate_beatmap_id_count"], 1)
            self.assertIn("duplicate_beatmap_id", row["anomalies"])
            self.assertIn("multiple_index_audio_paths_for_resolved_audio", row["anomalies"])
            self.assertEqual(row["duplicate_beatmap_ids"][0]["beatmap_id"], 101)
            self.assertEqual(row["audio_path_values"], ["1001/alias.wav", "1001/audio.wav"])

    def test_reports_missing_and_invalid_cache_without_recomputing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            missing_cache_audio = dataset_root / "0" / "1001" / "audio.wav"
            invalid_cache_audio = dataset_root / "0" / "1002" / "audio.wav"
            for audio_path in (missing_cache_audio, invalid_cache_audio):
                _write_file(audio_path, b"audio")
                _write_file(audio_path.with_name("map.osu"), b"osu")
                audio_path.with_name("metadata.json").write_text("{}\n", encoding="utf-8")
            index_path = _write_index(
                root,
                [
                    _index_row("0", "1001/audio.wav", "1001/map.osu", beatmap_id=101),
                    _index_row("0", "1002/audio.wav", "1002/map.osu", beatmap_id=202),
                ],
            )
            cache_root = root / "cache"
            config = BeatThisFramePredictionCacheConfig(cache_root=cache_root, checkpoint_path="checkpoint-a")
            invalid_cache_path = beatthis_frame_prediction_cache_path(beatthis_audio_cache_key(invalid_cache_audio), config)
            invalid_cache_path.parent.mkdir(parents=True, exist_ok=True)
            invalid_cache_path.write_bytes(b"not an npz")

            report = build_timing_v3_inventory(
                index_path=index_path,
                dataset_root=dataset_root,
                cache_root=cache_root,
                checkpoint_path="checkpoint-a",
                report_path=root / "report.json",
                audio_rows_path=root / "rows.jsonl",
            )

            rows = _read_jsonl(root / "rows.jsonl")
            self.assertEqual(report["cache"]["valid_count"], 0)
            self.assertEqual(report["cache"]["missing_count"], 1)
            self.assertEqual(report["cache"]["invalid_count"], 1)
            self.assertEqual([row["cache"]["status"] for row in rows], ["missing", "invalid"])
            self.assertIn("missing_cache", rows[0]["anomalies"])
            self.assertIn("invalid_cache", rows[1]["anomalies"])

    def test_cli_requires_explicit_outputs_and_returns_nonzero_for_missing_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            audio_path = dataset_root / "0" / "1001" / "audio.wav"
            _write_file(audio_path, b"audio")
            _write_file(audio_path.with_name("map.osu"), b"osu")
            audio_path.with_name("metadata.json").write_text("{}\n", encoding="utf-8")
            index_path = _write_index(root, [_index_row("0", "1001/audio.wav", "1001/map.osu", beatmap_id=101)])

            with self.assertRaises(SystemExit):
                main(["--index-path", str(index_path), "--dataset-root", str(dataset_root)])

            exit_code = main(
                [
                    "--index-path",
                    str(index_path),
                    "--dataset-root",
                    str(dataset_root),
                    "--cache-root",
                    str(root / "cache"),
                    "--checkpoint",
                    "checkpoint-a",
                    "--report-path",
                    str(root / "report.json"),
                    "--audio-rows-path",
                    str(root / "rows.jsonl"),
                    "--json",
                ]
            )

            self.assertEqual(exit_code, 1)
            self.assertTrue((root / "report.json").is_file())
            self.assertTrue((root / "rows.jsonl").is_file())


def _prediction(*, checkpoint_path: str, source_path: Path) -> FrameTimingPrediction:
    return FrameTimingPrediction(
        provider=BEATTHIS_PROVIDER_NAME,
        checkpoint_path=checkpoint_path,
        source_path=source_path,
        frame_rate_hz=BEATTHIS_FRAME_RATE_HZ,
        beat_prob=np.asarray([0.0, 0.25, 0.75, 1.0], dtype=np.float32),
        downbeat_prob=np.asarray([1.0, 0.5, 0.25, 0.0], dtype=np.float32),
    )


def _index_row(shard: str, audio_path: str, beatmap_path: str, *, beatmap_id: int) -> dict[str, object]:
    return {
        "shard": shard,
        "beatmap_set_id": int(beatmap_path.split("/", 1)[0]),
        "beatmap_set_path": beatmap_path.split("/", 1)[0],
        "beatmap_path": beatmap_path,
        "beatmap_filename": Path(beatmap_path).name,
        "audio_path": audio_path,
        "audio_filename": Path(audio_path).name,
        "beatmap_id": beatmap_id,
    }


def _write_index(root: Path, rows: list[dict[str, object]]) -> Path:
    path = root / "index.parquet"
    pd.DataFrame.from_records(rows).to_parquet(path, index=False)
    return path


def _write_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    unittest.main()
