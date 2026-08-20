import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from pulsefield_model.timing import build_beatthis_cache
from pulsefield_model.timing.providers.beatthis import BEATTHIS_FRAME_RATE_HZ, BEATTHIS_PROVIDER_NAME
from pulsefield_model.timing.providers.beatthis_cache import (
    BeatThisFramePredictionCacheConfig,
    beatthis_audio_cache_key,
    beatthis_frame_prediction_cache_path,
    load_beatthis_frame_prediction_cache,
)
from pulsefield_model.timing.schema import FrameTimingPrediction


class _FailingProvider:
    def __init__(self, **_: object) -> None:
        raise AssertionError("dry-run must not instantiate BeatThis provider")


class _FakeProvider:
    instances: list["_FakeProvider"] = []

    def __init__(self, *, checkpoint_path: str, device: str, float16: bool) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.float16 = float16
        self.calls: list[tuple[str, str]] = []
        self.__class__.instances.append(self)

    def predict_file(self, audio_path: Path) -> FrameTimingPrediction:
        self.calls.append(("file", Path(audio_path).name))
        return _prediction(checkpoint_path=self.checkpoint_path, source_path=audio_path)


class _FakeShiftProvider(_FakeProvider):
    def load_file(self, audio_path: Path) -> tuple[np.ndarray, int]:
        self.calls.append(("load", Path(audio_path).name))
        return np.asarray([0.0, 0.5, -0.5], dtype=np.float32), 1000

    def predict_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        source_path: Path | None = None,
    ) -> FrameTimingPrediction:
        self.calls.append(("base", str(sample_rate)))
        return _prediction(checkpoint_path=self.checkpoint_path, source_path=source_path)

    def predict_shifted_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        shift_ms: float,
        source_path: Path | None = None,
    ) -> FrameTimingPrediction:
        self.calls.append(("shift", f"{shift_ms:g}"))
        return _prediction(checkpoint_path=self.checkpoint_path, source_path=source_path)


class _PartiallyFailingShiftProvider(_FakeShiftProvider):
    def predict_shifted_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        shift_ms: float,
        source_path: Path | None = None,
    ) -> FrameTimingPrediction:
        self.calls.append(("shift", f"{shift_ms:g}"))
        if shift_ms == 5.0:
            raise RuntimeError("shift failed")
        return _prediction(checkpoint_path=self.checkpoint_path, source_path=source_path)


class BuildBeatThisCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeProvider.instances.clear()
        _FakeShiftProvider.instances.clear()

    def test_dry_run_dedupes_estimates_space_and_does_not_instantiate_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            audio_path = dataset_root / "0" / "1001" / "audio.wav"
            _write_wav(audio_path)
            index_path = _write_index(
                root,
                [
                    {"shard": "0", "audio_path": "1001/audio.wav", "beatmap_path": "1001/a.osu"},
                    {"shard": "0", "audio_path": "1001/audio.wav", "beatmap_path": "1001/b.osu"},
                ],
            )
            report_path = root / "report.json"

            with mock.patch.object(build_beatthis_cache, "BeatThisTimingProvider", _FailingProvider):
                report = build_beatthis_cache.build_beatthis_frame_prediction_cache(
                    index_path=index_path,
                    dataset_root=dataset_root,
                    cache_root=root / "cache",
                    checkpoint_path="checkpoint-a",
                    dry_run=True,
                    report_path=report_path,
                )

            self.assertTrue(report_path.exists())
            self.assertFalse(report_path.with_name(report_path.name + ".tmp").exists())
            self.assertEqual(report["source"]["row_count"], 2)
            self.assertEqual(report["source"]["unique_audio_count"], 1)
            self.assertEqual(report["cache"]["total_jobs"], 1)
            self.assertEqual(report["cache"]["existing_count"], 0)
            self.assertEqual(report["cache"]["missing_count"], 1)
            self.assertGreater(report["disk"]["known_estimated_missing_bytes"], 0)
            self.assertTrue(report["disk"]["enough_space_for_known_estimate"])

    def test_build_writes_cache_and_resume_hits_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            audio_path = dataset_root / "0" / "1001" / "audio.wav"
            _write_wav(audio_path)
            index_path = _write_index(
                root,
                [{"shard": "0", "audio_path": "1001/audio.wav", "beatmap_path": "1001/a.osu"}],
            )
            cache_root = root / "cache"
            config = BeatThisFramePredictionCacheConfig(
                cache_root=cache_root,
                checkpoint_path="checkpoint-a",
            )
            audio_key = beatthis_audio_cache_key(audio_path)

            with mock.patch.object(build_beatthis_cache, "BeatThisTimingProvider", _FakeProvider):
                first = build_beatthis_cache.build_beatthis_frame_prediction_cache(
                    index_path=index_path,
                    dataset_root=dataset_root,
                    cache_root=cache_root,
                    checkpoint_path="checkpoint-a",
                    report_path=root / "first.json",
                    progress_every=0,
                )
                _FakeProvider.instances.clear()
                second = build_beatthis_cache.build_beatthis_frame_prediction_cache(
                    index_path=index_path,
                    dataset_root=dataset_root,
                    cache_root=cache_root,
                    checkpoint_path="checkpoint-a",
                    report_path=root / "second.json",
                    progress_every=0,
                )

            cache_path = beatthis_frame_prediction_cache_path(audio_key, config)
            loaded = load_beatthis_frame_prediction_cache(audio_key, config)

            self.assertTrue(cache_path.exists())
            self.assertIsNotNone(loaded)
            self.assertEqual(first["cache"]["created_count"], 1)
            self.assertEqual(first["cache"]["failed_count"], 0)
            self.assertEqual(second["cache"]["existing_count"], 1)
            self.assertEqual(second["cache"]["created_count"], 0)
            self.assertEqual(_FakeProvider.instances, [])

    def test_migrates_matching_legacy_v1_cache_by_rewriting_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            audio_path = dataset_root / "0" / "1001" / "audio.wav"
            _write_wav(audio_path)
            index_path = _write_index(
                root,
                [{"shard": "0", "audio_path": "1001/audio.wav", "beatmap_path": "1001/a.osu"}],
            )
            cache_root = root / "cache"
            legacy_root = cache_root / "beatthis_frame_predictions_v1"
            prediction = _prediction(checkpoint_path="checkpoint-a", source_path=audio_path.resolve())
            legacy_path = build_beatthis_cache._legacy_cache_path_for_audio(
                audio_path,
                legacy_cache_root=legacy_root,
                checkpoint_path="checkpoint-a",
                float16=False,
            )
            _write_legacy_cache(legacy_path, prediction)

            with mock.patch.object(build_beatthis_cache, "BeatThisTimingProvider", _FailingProvider):
                report = build_beatthis_cache.build_beatthis_frame_prediction_cache(
                    index_path=index_path,
                    dataset_root=dataset_root,
                    cache_root=cache_root,
                    checkpoint_path="checkpoint-a",
                    migrate_legacy=True,
                    legacy_cache_root=legacy_root,
                    report_path=root / "report.json",
                    progress_every=0,
                )

            config = BeatThisFramePredictionCacheConfig(
                cache_root=cache_root,
                checkpoint_path="checkpoint-a",
            )
            loaded = load_beatthis_frame_prediction_cache(beatthis_audio_cache_key(audio_path), config)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            np.testing.assert_array_equal(loaded.beat_prob, prediction.beat_prob)
            np.testing.assert_array_equal(loaded.downbeat_prob, prediction.downbeat_prob)
            self.assertEqual(report["legacy_migration"]["indexed_count"], 1)
            self.assertEqual(report["legacy_migration"]["failed_count"], 0)
            self.assertEqual(report["cache"]["migrated_count"], 1)
            self.assertEqual(report["cache"]["created_count"], 0)
            self.assertEqual(report["cache"]["failed_count"], 0)

    def test_corrupt_v2_cache_is_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            audio_path = dataset_root / "0" / "1001" / "audio.wav"
            _write_wav(audio_path)
            index_path = _write_index(
                root,
                [{"shard": "0", "audio_path": "1001/audio.wav", "beatmap_path": "1001/a.osu"}],
            )
            cache_root = root / "cache"
            config = BeatThisFramePredictionCacheConfig(
                cache_root=cache_root,
                checkpoint_path="checkpoint-a",
            )
            audio_key = beatthis_audio_cache_key(audio_path)
            cache_path = beatthis_frame_prediction_cache_path(audio_key, config)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(b"not an npz")

            with mock.patch.object(build_beatthis_cache, "BeatThisTimingProvider", _FakeProvider):
                report = build_beatthis_cache.build_beatthis_frame_prediction_cache(
                    index_path=index_path,
                    dataset_root=dataset_root,
                    cache_root=cache_root,
                    checkpoint_path="checkpoint-a",
                    report_path=root / "report.json",
                    progress_every=0,
                )

            self.assertIsNotNone(load_beatthis_frame_prediction_cache(audio_key, config))
            self.assertEqual(report["cache"]["existing_count"], 0)
            self.assertEqual(report["cache"]["missing_count"], 1)
            self.assertEqual(report["cache"]["corrupt_existing_recompute_count"], 1)
            self.assertEqual(report["cache"]["created_count"], 1)
            self.assertEqual(report["cache"]["failed_count"], 0)

    def test_shifted_cache_uses_one_provider_and_one_audio_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            audio_path = dataset_root / "0" / "1001" / "audio.wav"
            _write_wav(audio_path)
            index_path = _write_index(
                root,
                [{"shard": "0", "audio_path": "1001/audio.wav", "beatmap_path": "1001/a.osu"}],
            )

            with mock.patch.object(build_beatthis_cache, "BeatThisTimingProvider", _FakeShiftProvider):
                report = build_beatthis_cache.build_beatthis_frame_prediction_cache(
                    index_path=index_path,
                    dataset_root=dataset_root,
                    cache_root=root / "cache",
                    checkpoint_path="checkpoint-a",
                    shift_ms=(0.0, 5.0),
                    report_path=root / "report.json",
                    progress_every=0,
                )

            self.assertEqual(report["cache"]["created_count"], 2)
            self.assertEqual(len(_FakeShiftProvider.instances), 1)
            self.assertEqual(
                _FakeShiftProvider.instances[0].calls,
                [("load", "audio.wav"), ("base", "1000"), ("shift", "5")],
            )

    def test_partial_shift_failure_is_reported_only_for_failed_shift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            audio_path = dataset_root / "0" / "1001" / "audio.wav"
            _write_wav(audio_path)
            index_path = _write_index(
                root,
                [{"shard": "0", "audio_path": "1001/audio.wav", "beatmap_path": "1001/a.osu"}],
            )

            with mock.patch.object(
                build_beatthis_cache,
                "BeatThisTimingProvider",
                _PartiallyFailingShiftProvider,
            ):
                report = build_beatthis_cache.build_beatthis_frame_prediction_cache(
                    index_path=index_path,
                    dataset_root=dataset_root,
                    cache_root=root / "cache",
                    checkpoint_path="checkpoint-a",
                    shift_ms=(0.0, 5.0, 10.0),
                    report_path=root / "report.json",
                    progress_every=0,
                )

            self.assertEqual(report["cache"]["created_count"], 2)
            self.assertEqual(report["cache"]["failed_count"], 1)
            self.assertEqual([failure["shift_ms"] for failure in report["failures"]], [5.0])

    def test_unknown_audio_duration_blocks_full_run_space_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            audio_path = dataset_root / "0" / "1001" / "not-audio.bin"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_path.write_bytes(b"not audio")
            index_path = _write_index(
                root,
                [{"shard": "0", "audio_path": "1001/not-audio.bin", "beatmap_path": "1001/a.osu"}],
            )

            with mock.patch.object(build_beatthis_cache, "BeatThisTimingProvider", _FailingProvider):
                report = build_beatthis_cache.build_beatthis_frame_prediction_cache(
                    index_path=index_path,
                    dataset_root=dataset_root,
                    cache_root=root / "cache",
                    checkpoint_path="checkpoint-a",
                    report_path=root / "report.json",
                    progress_every=0,
                )

            self.assertTrue(report["aborted"])
            self.assertEqual(report["abort_reason"], "incomplete_disk_space_estimate")
            self.assertFalse(report["disk"]["space_estimate_complete"])
            self.assertEqual(report["disk"]["unknown_estimate_count"], 1)

    def test_unsafe_audio_paths_are_rejected_before_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            index_path = _write_index(
                root,
                [{"shard": "0", "audio_path": "../evil.wav", "beatmap_path": "bad.osu"}],
            )

            with self.assertRaisesRegex(ValueError, "audio_path"):
                build_beatthis_cache.build_beatthis_frame_prediction_cache(
                    index_path=index_path,
                    dataset_root=root / "dataset",
                    cache_root=root / "cache",
                    dry_run=True,
                    report_path=None,
                )


def _prediction(*, checkpoint_path: str, source_path: str | Path | None) -> FrameTimingPrediction:
    return FrameTimingPrediction(
        provider=BEATTHIS_PROVIDER_NAME,
        checkpoint_path=checkpoint_path,
        source_path=source_path,
        frame_rate_hz=BEATTHIS_FRAME_RATE_HZ,
        beat_prob=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        downbeat_prob=np.asarray([1.0, 0.25, 0.0], dtype=np.float32),
    )


def _write_index(root: Path, rows: list[dict[str, object]]) -> Path:
    path = root / "index.parquet"
    pd.DataFrame.from_records(rows).to_parquet(path, index=False)
    return path


def _write_wav(path: Path, *, seconds: int = 1, sample_rate: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        for index in range(seconds * sample_rate):
            sample = int(12000 * math.sin(2.0 * math.pi * 440.0 * index / sample_rate))
            handle.writeframes(struct.pack("<h", sample))


def _write_legacy_cache(path: Path, prediction: FrameTimingPrediction) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            beat_prob=prediction.beat_prob,
            downbeat_prob=prediction.downbeat_prob,
            frame_rate_hz=np.asarray(prediction.frame_rate_hz, dtype=np.float64),
            provider=np.asarray(prediction.provider),
            checkpoint_path=np.asarray(prediction.checkpoint_path or ""),
            source_path=np.asarray(prediction.source_path or ""),
        )


if __name__ == "__main__":
    unittest.main()
