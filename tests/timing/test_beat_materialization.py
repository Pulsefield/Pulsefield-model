import contextlib
import io
import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from pulsefield_model.timing.beat_materialization import (
    BeatMaterializationConfig,
    RampBeatGridHint,
    fit_ramp_beat_grid,
    materialize_beats,
    ramp_beat_grid_report,
)
from pulsefield_model.timing.schema import FrameTimingPrediction


def _ramp_beat_times(
    *,
    start_ms: float = 1000.0,
    end_ms: float = 15000.0,
    start_bpm: float = 120.0,
    end_bpm: float = 260.0,
) -> list[float]:
    times_ms: list[float] = []
    cursor_ms = start_ms
    while cursor_ms <= end_ms:
        progress = (cursor_ms - start_ms) / (end_ms - start_ms)
        bpm = start_bpm + progress * (end_bpm - start_bpm)
        times_ms.append(cursor_ms)
        cursor_ms += 60000.0 / bpm
    return times_ms


def _stable_beat_times(
    *,
    start_ms: float = 1000.0,
    end_ms: float = 15000.0,
    bpm: float = 180.0,
) -> list[float]:
    beat_length_ms = 60000.0 / bpm
    return list(np.arange(start_ms, end_ms + beat_length_ms / 2.0, beat_length_ms))


def _prediction_from_beats(
    beat_times_ms: list[float],
    *,
    frame_rate_hz: float = 100.0,
    source_path: str = "song.mp3",
) -> FrameTimingPrediction:
    frame_count = int((max(beat_times_ms) + 2000.0) / 1000.0 * frame_rate_hz)
    frame_times_ms = np.arange(frame_count, dtype=np.float64) / frame_rate_hz * 1000.0
    beat_prob = np.full(frame_count, 0.01, dtype=np.float64)
    for beat_time_ms in beat_times_ms:
        distance_ms = np.abs(frame_times_ms - beat_time_ms)
        beat_prob = np.maximum(beat_prob, np.maximum(0.0, 1.0 - distance_ms / 35.0))
    return FrameTimingPrediction(
        provider="synthetic",
        checkpoint_path="synthetic",
        source_path=source_path,
        beat_prob=beat_prob.astype(np.float32),
        downbeat_prob=np.zeros(frame_count, dtype=np.float32),
        frame_rate_hz=frame_rate_hz,
    )


class _FakeRampBeatProvider:
    def __init__(self, *, checkpoint_path: str, device: str, float16: bool) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.float16 = float16

    def predict_file(self, audio_path: Path) -> FrameTimingPrediction:
        return _prediction_from_beats(_ramp_beat_times(), source_path=audio_path.as_posix())


class BeatMaterializationTests(unittest.TestCase):
    def test_materializes_synthetic_ramp_beats_before_fitting_grid(self) -> None:
        beat_times_ms = _ramp_beat_times()
        prediction = _prediction_from_beats(beat_times_ms)

        materialized = materialize_beats(prediction)
        result = fit_ramp_beat_grid(prediction, allow_no_hint=True)

        self.assertEqual(materialized.reasons, ("materialized_beats",))
        self.assertIsNotNone(materialized.sequence)
        assert materialized.sequence is not None
        self.assertGreaterEqual(materialized.sequence.beat_count, len(beat_times_ms) - 2)

        self.assertTrue(result.accepted, result.reasons)
        self.assertEqual(result.reasons, ("materialized_ramp_beat_grid",))
        self.assertIsNotNone(result.candidate)
        assert result.candidate is not None
        self.assertGreater(result.candidate.duration_s, 10.0)
        self.assertGreaterEqual(result.candidate.point_count, 8)
        self.assertLess(abs(result.candidate.start_ms - beat_times_ms[0]), 750.0)
        self.assertLess(abs(result.candidate.end_ms - beat_times_ms[-2]), 1500.0)
        self.assertGreater(result.candidate.end_bpm, result.candidate.start_bpm)

        segment_bpms = [segment.local_bpm for segment in result.candidate.grid.segments]
        self.assertTrue(all(next_bpm > bpm for bpm, next_bpm in zip(segment_bpms, segment_bpms[1:])))

    def test_rejects_stable_tempo_after_materializing_beats(self) -> None:
        prediction = _prediction_from_beats(_stable_beat_times())

        result = fit_ramp_beat_grid(prediction, allow_no_hint=True)

        self.assertFalse(result.accepted)
        self.assertEqual(result.reasons, ("no_ramp_candidates",))
        self.assertIsNotNone(result.materialization.sequence)
        assert result.materialization.sequence is not None
        self.assertGreater(result.materialization.sequence.beat_count, 20)

    def test_report_serializes_candidate_segments(self) -> None:
        prediction = _prediction_from_beats(_ramp_beat_times())

        report = ramp_beat_grid_report(fit_ramp_beat_grid(prediction, allow_no_hint=True))

        self.assertTrue(report["accepted"])
        self.assertIsInstance(report["sequence"], dict)
        self.assertIsInstance(report["candidate"], dict)
        candidate = report["candidate"]
        assert isinstance(candidate, dict)
        self.assertGreaterEqual(len(candidate["segments"]), 8)
        json.dumps(report, allow_nan=False)

    def test_trusted_hint_scores_rendered_ramp_against_probabilities(self) -> None:
        beat_times_ms = _ramp_beat_times()
        prediction = _prediction_from_beats(beat_times_ms)
        hint = RampBeatGridHint(
            start_ms=1000.0,
            end_ms=15000.0,
            start_bpm=120.0,
            end_bpm=260.0,
        )

        result = fit_ramp_beat_grid(prediction, hint=hint)

        self.assertTrue(result.accepted, result.reasons)
        self.assertEqual(result.reasons, ("trusted_hint_ramp_beat_grid",))
        self.assertIsNotNone(result.candidate)
        assert result.candidate is not None
        self.assertEqual(result.candidate.hypothesis_kind, "trusted_hint")
        self.assertGreater(result.candidate.probability_score, 0.4)
        self.assertLess(abs(result.candidate.start_ms - hint.start_ms), 750.0)
        self.assertLess(abs(result.candidate.end_ms - hint.end_ms), 750.0)
        self.assertTrue(result.candidate.ramp_detection.is_ramp)

    def test_trusted_hint_uses_detector_duration_floor(self) -> None:
        beat_times_ms = _ramp_beat_times(
            start_ms=1000.0,
            end_ms=5500.0,
            start_bpm=200.0,
            end_bpm=300.0,
        )
        prediction = _prediction_from_beats(beat_times_ms)
        hint = RampBeatGridHint(
            start_ms=1000.0,
            end_ms=5500.0,
            start_bpm=200.0,
            end_bpm=300.0,
        )

        result = fit_ramp_beat_grid(prediction, hint=hint)

        self.assertTrue(result.accepted, result.reasons)
        self.assertEqual(result.reasons, ("trusted_hint_ramp_beat_grid",))
        self.assertIsNotNone(result.candidate)
        assert result.candidate is not None
        self.assertGreaterEqual(result.candidate.duration_s, 4.0)
        self.assertLess(result.candidate.duration_s, 8.0)
        self.assertTrue(result.candidate.ramp_detection.is_ramp)

    def test_no_hint_fitting_is_disabled_by_default(self) -> None:
        prediction = _prediction_from_beats(_ramp_beat_times())

        result = fit_ramp_beat_grid(prediction)

        self.assertFalse(result.accepted)
        self.assertEqual(result.reasons, ("no_hint_fitting_disabled",))
        self.assertEqual(result.candidate_count, 0)
        self.assertIsNone(result.materialization.sequence)

    def test_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            BeatMaterializationConfig(min_bpm=500.0, max_bpm=80.0)
        with self.assertRaises(ValueError):
            BeatMaterializationConfig(min_peak_prob=1.5)
        with self.assertRaises(ValueError):
            BeatMaterializationConfig(local_max_radius_frames=0)
        with self.assertRaises(ValueError):
            BeatMaterializationConfig(min_hint_ramp_duration_s=0.0)

    def test_fit_audio_flag_rejects_no_hint_by_default(self) -> None:
        from pulsefield_model.timing import fit_audio

        stdout = io.StringIO()
        with mock.patch.object(fit_audio, "BeatThisTimingProvider", _FakeRampBeatProvider):
            with contextlib.redirect_stdout(stdout):
                exit_code = fit_audio.main(["song.mp3", "--json", "--ramp-beat-grid"])

        self.assertEqual(exit_code, 0)
        report = json.loads(stdout.getvalue())
        self.assertIn("ramp_beat_grid", report)
        self.assertFalse(report["ramp_beat_grid"]["accepted"])
        self.assertEqual(report["ramp_beat_grid"]["reasons"], ["no_hint_fitting_disabled"])

    def test_fit_audio_allow_no_hint_flag_includes_ramp_beat_grid_report(self) -> None:
        from pulsefield_model.timing import fit_audio

        stdout = io.StringIO()
        with mock.patch.object(fit_audio, "BeatThisTimingProvider", _FakeRampBeatProvider):
            with contextlib.redirect_stdout(stdout):
                exit_code = fit_audio.main(
                    ["song.mp3", "--json", "--ramp-beat-grid", "--ramp-beat-grid-allow-no-hint"]
                )

        self.assertEqual(exit_code, 0)
        report = json.loads(stdout.getvalue())
        self.assertIn("ramp_beat_grid", report)
        self.assertTrue(report["ramp_beat_grid"]["accepted"])

    def test_fit_audio_flag_accepts_trusted_ramp_hint(self) -> None:
        from pulsefield_model.timing import fit_audio

        stdout = io.StringIO()
        with mock.patch.object(fit_audio, "BeatThisTimingProvider", _FakeRampBeatProvider):
            with contextlib.redirect_stdout(stdout):
                exit_code = fit_audio.main(
                    [
                        "song.mp3",
                        "--json",
                        "--ramp-beat-grid",
                        "--ramp-hint-start-ms",
                        "1000",
                        "--ramp-hint-end-ms",
                        "15000",
                        "--ramp-hint-start-bpm",
                        "120",
                        "--ramp-hint-end-bpm",
                        "260",
                    ]
                )

        self.assertEqual(exit_code, 0)
        report = json.loads(stdout.getvalue())
        candidate = report["ramp_beat_grid"]["candidate"]
        self.assertTrue(report["ramp_beat_grid"]["accepted"])
        self.assertEqual(report["ramp_beat_grid"]["reasons"], ["trusted_hint_ramp_beat_grid"])
        self.assertEqual(candidate["hypothesis_kind"], "trusted_hint")
        self.assertGreater(candidate["probability_score"], 0.4)

    def test_fit_audio_rejects_partial_trusted_ramp_hint(self) -> None:
        from pulsefield_model.timing import fit_audio

        with self.assertRaises(ValueError):
            fit_audio.main(
                [
                    "song.mp3",
                    "--json",
                    "--ramp-beat-grid",
                    "--ramp-hint-start-ms",
                    "1000",
                ]
            )

    def test_fit_audio_default_report_excludes_ramp_beat_grid(self) -> None:
        from pulsefield_model.timing import fit_audio

        stdout = io.StringIO()
        with mock.patch.object(fit_audio, "BeatThisTimingProvider", _FakeRampBeatProvider):
            with contextlib.redirect_stdout(stdout):
                exit_code = fit_audio.main(["song.mp3", "--json"])

        self.assertEqual(exit_code, 0)
        report = json.loads(stdout.getvalue())
        self.assertNotIn("ramp_beat_grid", report)


if __name__ == "__main__":
    unittest.main()
