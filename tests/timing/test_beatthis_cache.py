import hashlib
import json
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from pulsefield_model.timing.providers.beatthis import BEATTHIS_FRAME_RATE_HZ, BEATTHIS_PROVIDER_NAME
from pulsefield_model.timing.providers.beatthis_cache import (
    BeatThisFramePredictionCacheConfig,
    BeatThisFramePredictionCacheError,
    beatthis_audio_cache_key,
    beatthis_frame_prediction_cache_path,
    load_beatthis_frame_prediction_cache,
    save_beatthis_frame_prediction_cache,
)
from pulsefield_model.timing.schema import FrameTimingPrediction


def _prediction(
    *,
    checkpoint_path: str = "checkpoint-a",
    source_path: str | Path | None = "song.mp3",
    frame_rate_hz: float = BEATTHIS_FRAME_RATE_HZ,
    beat_prob: np.ndarray | None = None,
    downbeat_prob: np.ndarray | None = None,
) -> FrameTimingPrediction:
    return FrameTimingPrediction(
        provider=BEATTHIS_PROVIDER_NAME,
        checkpoint_path=checkpoint_path,
        source_path=source_path,
        frame_rate_hz=frame_rate_hz,
        beat_prob=np.asarray([0.0, 0.5, 1.0], dtype=np.float32) if beat_prob is None else beat_prob,
        downbeat_prob=np.asarray([1.0, 0.25, 0.0], dtype=np.float32) if downbeat_prob is None else downbeat_prob,
    )


class BeatThisFramePredictionCacheTests(unittest.TestCase):
    def test_audio_cache_key_changes_when_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "song.mp3"
            audio_path.write_bytes(b"first")

            first = beatthis_audio_cache_key(audio_path)
            self.assertEqual(beatthis_audio_cache_key(audio_path), first)

            audio_path.write_bytes(b"second-version")
            self.assertNotEqual(beatthis_audio_cache_key(audio_path), first)

    def test_cache_path_hashes_audio_key_and_fingerprints_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = BeatThisFramePredictionCacheConfig(
                cache_root=Path(tmp_dir),
                cache_version="beatthis-cache-test",
                checkpoint_path="checkpoint-a",
                float16=False,
                shift_ms=0.0,
                frame_rate_hz=50.0,
            )
            audio_key = "/music/artist/song.mp3"

            path = beatthis_frame_prediction_cache_path(audio_key, config)

            self.assertEqual(path.parent, config.cache_dir)
            self.assertEqual(path.name, f"{hashlib.sha256(audio_key.encode('utf-8')).hexdigest()}.npz")
            self.assertNotIn("song.mp3", path.as_posix())

            variants = [
                replace(config, cache_version="beatthis-cache-test-v2"),
                replace(config, checkpoint_path="checkpoint-b"),
                replace(config, float16=True),
                replace(config, shift_ms=20.0),
                replace(config, frame_rate_hz=100.0),
            ]
            for variant in variants:
                self.assertNotEqual(variant.cache_dir, config.cache_dir)

    def test_save_and_load_roundtrip_uses_compressed_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = BeatThisFramePredictionCacheConfig(
                cache_root=Path(tmp_dir),
                checkpoint_path="checkpoint-a",
                float16=True,
                shift_ms=10.0,
            )
            prediction = _prediction(checkpoint_path="checkpoint-a", source_path=Path("audio/song.mp3"))

            cache_path = save_beatthis_frame_prediction_cache(prediction, "audio-key", config)
            loaded = load_beatthis_frame_prediction_cache("audio-key", config)

            self.assertEqual(cache_path.suffix, ".npz")
            with zipfile.ZipFile(cache_path) as archive:
                self.assertTrue(archive.infolist())
                self.assertTrue(all(item.compress_type == zipfile.ZIP_DEFLATED for item in archive.infolist()))

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.provider, BEATTHIS_PROVIDER_NAME)
            self.assertEqual(loaded.checkpoint_path, "checkpoint-a")
            self.assertEqual(loaded.source_path, "audio/song.mp3")
            self.assertEqual(loaded.frame_rate_hz, BEATTHIS_FRAME_RATE_HZ)
            np.testing.assert_array_equal(loaded.beat_prob, prediction.beat_prob)
            np.testing.assert_array_equal(loaded.downbeat_prob, prediction.downbeat_prob)

    def test_load_missing_cache_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = BeatThisFramePredictionCacheConfig(cache_root=Path(tmp_dir))

            self.assertIsNone(load_beatthis_frame_prediction_cache("missing-audio-key", config))

    def test_corrupt_cache_raises_dedicated_error_and_can_be_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = BeatThisFramePredictionCacheConfig(cache_root=Path(tmp_dir), checkpoint_path="checkpoint-a")
            cache_path = beatthis_frame_prediction_cache_path("audio-key", config)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(b"not an npz")

            with self.assertRaises(BeatThisFramePredictionCacheError):
                load_beatthis_frame_prediction_cache("audio-key", config)

            prediction = _prediction(checkpoint_path="checkpoint-a")
            saved_path = save_beatthis_frame_prediction_cache(prediction, "audio-key", config)
            loaded = load_beatthis_frame_prediction_cache("audio-key", config)

            self.assertEqual(saved_path, cache_path)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            np.testing.assert_array_equal(loaded.beat_prob, prediction.beat_prob)

    def test_load_rejects_metadata_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = BeatThisFramePredictionCacheConfig(
                cache_root=Path(tmp_dir),
                checkpoint_path="checkpoint-a",
                float16=True,
            )
            cache_path = save_beatthis_frame_prediction_cache(
                _prediction(checkpoint_path="checkpoint-a"),
                "audio-key",
                config,
            )

            metadata, beat_prob, downbeat_prob = _read_npz(cache_path)
            metadata["float16"] = False
            _write_npz(cache_path, metadata=metadata, beat_prob=beat_prob, downbeat_prob=downbeat_prob)

            with self.assertRaisesRegex(BeatThisFramePredictionCacheError, "float16 mismatch"):
                load_beatthis_frame_prediction_cache("audio-key", config)

    def test_load_rejects_invalid_probability_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = BeatThisFramePredictionCacheConfig(cache_root=Path(tmp_dir), checkpoint_path="checkpoint-a")
            cache_path = save_beatthis_frame_prediction_cache(
                _prediction(checkpoint_path="checkpoint-a"),
                "audio-key",
                config,
            )

            metadata, _, downbeat_prob = _read_npz(cache_path)
            bad_beat_prob = np.asarray([0.0, 2.0, 0.5], dtype=np.float32)
            _write_npz(cache_path, metadata=metadata, beat_prob=bad_beat_prob, downbeat_prob=downbeat_prob)

            with self.assertRaisesRegex(BeatThisFramePredictionCacheError, "probabilities in \\[0, 1\\]"):
                load_beatthis_frame_prediction_cache("audio-key", config)

    def test_save_rejects_prediction_that_does_not_match_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = BeatThisFramePredictionCacheConfig(cache_root=Path(tmp_dir), checkpoint_path="checkpoint-a")
            prediction = _prediction(checkpoint_path="checkpoint-b")

            with self.assertRaisesRegex(ValueError, "checkpoint_path"):
                save_beatthis_frame_prediction_cache(prediction, "audio-key", config)


def _read_npz(cache_path: Path) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as payload:
        metadata = json.loads(np.asarray(payload["metadata_json"]).item())
        beat_prob = np.asarray(payload["beat_prob"]).copy()
        downbeat_prob = np.asarray(payload["downbeat_prob"]).copy()
    return metadata, beat_prob, downbeat_prob


def _write_npz(
    cache_path: Path,
    *,
    metadata: dict[str, object],
    beat_prob: np.ndarray,
    downbeat_prob: np.ndarray,
) -> None:
    with cache_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
            beat_prob=beat_prob,
            downbeat_prob=downbeat_prob,
        )
