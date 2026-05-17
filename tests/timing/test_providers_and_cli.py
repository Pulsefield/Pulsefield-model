import contextlib
import io
import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from pulsefield_model.timing.providers import beatthis
from pulsefield_model.timing.providers.beatthis import (
    BEATTHIS_FRAME_RATE_HZ,
    DEFAULT_BEATTHIS_CHECKPOINT,
    DEFAULT_BEATTHIS_DEVICE,
    BeatThisTimingProvider,
)
from pulsefield_model.timing.providers.oracle import oracle_timing_grid_from_beatmap, render_oracle_dense_timing_v2
from pulsefield_model.timing.schema import FrameTimingPrediction


class _FakeAudio2Frames:
    instances: list["_FakeAudio2Frames"] = []

    def __init__(self, checkpoint_path: str, device: str, float16: bool) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.float16 = float16
        self.calls: list[tuple[np.ndarray, int]] = []
        self.__class__.instances.append(self)

    def __call__(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append((audio, sample_rate))
        return (
            np.asarray([-2.0, 0.0, 2.0], dtype=np.float32),
            np.asarray([2.0, 0.0, -2.0], dtype=np.float32),
        )


def _fake_load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    return np.asarray([0.25, -0.25, 0.0], dtype=np.float32), 44100


def _synthetic_prediction() -> FrameTimingPrediction:
    frame_count = 1000
    frame_rate_hz = 50.0
    frame_times_ms = np.arange(frame_count, dtype=np.float64) / frame_rate_hz * 1000.0
    beat_length_ms = 500.0
    offset_ms = 120.0
    phase = ((frame_times_ms - offset_ms) / beat_length_ms) % 1.0
    distance_ms = np.minimum(phase, 1.0 - phase) * beat_length_ms
    beat_prob = np.maximum(0.0, 1.0 - distance_ms / 40.0).astype(np.float32)
    return FrameTimingPrediction(
        provider="fake-beat-this",
        checkpoint_path="fake-checkpoint",
        source_path="song.mp3",
        beat_prob=beat_prob,
        downbeat_prob=np.zeros(frame_count, dtype=np.float32),
        frame_rate_hz=frame_rate_hz,
    )


class _FakeBeatThisTimingProvider:
    def __init__(self, *, checkpoint_path: str, device: str, float16: bool) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.float16 = float16

    def predict_file(self, audio_path: Path) -> FrameTimingPrediction:
        self.audio_path = audio_path
        return _synthetic_prediction()


class TimingProviderCliTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeAudio2Frames.instances.clear()

    def test_beatthis_predict_audio_uses_defaults_and_probability_vectors(self) -> None:
        audio = np.asarray([0.0, 0.5, -0.5], dtype=np.float32)
        provider = BeatThisTimingProvider()

        with mock.patch.object(
            beatthis,
            "_load_beat_this_api",
            return_value=beatthis.BeatThisAPI(_FakeAudio2Frames, _fake_load_audio),
        ):
            prediction = provider.predict_audio(audio, sample_rate=22050)

        frame_model = _FakeAudio2Frames.instances[0]
        self.assertEqual(frame_model.checkpoint_path, DEFAULT_BEATTHIS_CHECKPOINT)
        self.assertEqual(frame_model.device, DEFAULT_BEATTHIS_DEVICE)
        self.assertFalse(frame_model.float16)
        np.testing.assert_array_equal(frame_model.calls[0][0], audio)
        self.assertEqual(frame_model.calls[0][1], 22050)

        self.assertEqual(prediction.provider, "beat-this")
        self.assertEqual(prediction.checkpoint_path, DEFAULT_BEATTHIS_CHECKPOINT)
        self.assertEqual(prediction.frame_rate_hz, BEATTHIS_FRAME_RATE_HZ)
        np.testing.assert_allclose(prediction.beat_prob, np.asarray([0.11920292, 0.5, 0.88079708]), rtol=1e-6)
        np.testing.assert_allclose(prediction.downbeat_prob, np.asarray([0.88079708, 0.5, 0.11920292]), rtol=1e-6)

    def test_oracle_provider_renders_dense_timing_from_osu_red_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            osu_path = Path(tmp_dir) / "map.osu"
            _write_timing_osu(
                osu_path,
                [
                    "0,500,4,2,0,80,1,0",
                    "1000,250,4,2,0,80,1,0",
                ],
            )

            grid = oracle_timing_grid_from_beatmap(osu_path)
            track = render_oracle_dense_timing_v2(osu_path, frame_count=4)

        self.assertEqual([segment.offset_ms for segment in grid.segments], [0.0, 1000.0])
        self.assertEqual(track.shape, (4, 4))
        self.assertEqual(track.dtype, np.dtype("float32"))

    def test_fit_audio_main_can_emit_json(self) -> None:
        from pulsefield_model.timing import fit_audio

        stdout = io.StringIO()
        with mock.patch.object(fit_audio, "BeatThisTimingProvider", _FakeBeatThisTimingProvider):
            with contextlib.redirect_stdout(stdout):
                exit_code = fit_audio.main(["song.mp3", "--json"])

        self.assertEqual(exit_code, 0)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["source_path"], "song.mp3")
        self.assertEqual(report["provider"], "fake-beat-this")
        self.assertEqual(report["segments"][0]["offset_ms"], 120.0)
        self.assertEqual(report["segments"][0]["beat_length_ms"], 500.0)
        self.assertEqual(report["segments"][0]["bpm"], 120.0)


def _write_timing_osu(path: Path, timing_lines: list[str]) -> None:
    path.write_text(
        textwrap.dedent(
            f"""\
            osu file format v14

            [TimingPoints]
            {chr(10).join(timing_lines)}
            """
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
