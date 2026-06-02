import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pulsefield_model.events.canonical import LaneAction
from pulsefield_model.timing.mock_osu_export import (
    build_mock_beat_grid_timepoints,
    create_timing_mock_beatmap,
    main,
    timing_grid_from_report,
)
from pulsefield_model.timing.schema import FittedTimingGrid, TimingSegment


def _timing_report() -> dict[str, object]:
    return {
        "source_path": "song.mp3",
        "provider": "unit-test",
        "checkpoint_path": "fake",
        "device": "cpu",
        "canonicalization": "bpm_80_160",
        "frame_count": 110,
        "frame_rate_hz": 50.0,
        "fit_seconds": 0.001,
        "score": 1.0,
        "diagnostics": {},
        "segments": [
            {
                "offset_ms": 120.0,
                "beat_length_ms": 500.0,
                "bpm": 120.0,
                "meter": 4,
            },
        ],
    }


class TimingMockOsuExportTests(unittest.TestCase):
    def test_build_mock_beat_grid_alternates_0011_1100_on_beats(self) -> None:
        grid = FittedTimingGrid(
            (
                TimingSegment(offset_ms=120.0, beat_length_ms=500.0),
            )
        )

        result = build_mock_beat_grid_timepoints(grid, start_ms=0, end_ms=1700)

        self.assertEqual([timepoint.time_ms for timepoint in result.timepoints], [120, 620, 1120, 1620])
        self.assertEqual(
            result.timepoints[0].lane_actions,
            (LaneAction.NONE, LaneAction.NONE, LaneAction.TAP, LaneAction.TAP),
        )
        self.assertEqual(
            result.timepoints[1].lane_actions,
            (LaneAction.TAP, LaneAction.TAP, LaneAction.NONE, LaneAction.NONE),
        )
        self.assertEqual(result.max_rounding_error_ms, 0.0)

    def test_build_mock_beat_grid_rounds_half_milliseconds_up(self) -> None:
        grid = FittedTimingGrid(
            (
                TimingSegment(offset_ms=120.5, beat_length_ms=500.0),
            )
        )

        result = build_mock_beat_grid_timepoints(grid, start_ms=0, end_ms=700)

        self.assertEqual([timepoint.time_ms for timepoint in result.timepoints], [121, 621])
        self.assertEqual(result.max_rounding_error_ms, 0.5)

    def test_create_timing_mock_beatmap_writes_osu_and_report_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            audio_path = root / "phase test.mp3"
            audio_path.write_bytes(b"")
            output_dir = root / "mock-output"

            with mock.patch(
                "pulsefield_model.timing.mock_osu_export.fit_audio_file",
                return_value=_timing_report(),
            ):
                result = create_timing_mock_beatmap(audio_path, output_dir=output_dir)

            osu_path = Path(str(result["osu_mock_beatmap_path"]))
            report_path = Path(str(result["timing_report_path"]))
            osu_text = osu_path.read_text(encoding="utf-8")
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["audio_path"], audio_path.resolve(strict=False).as_posix())
        self.assertEqual(result["osumockbeatmappath"], result["osu_mock_beatmap_path"])
        self.assertTrue(osu_path.name.endswith("_timing_mock.osu"))
        self.assertIn("AudioFilename:../phase test.mp3", osu_text)
        self.assertIn("[TimingPoints]\n120,500,4,2,0,100,1,0", osu_text)
        self.assertIn("320,192,120,1,0,0:0:0:0:", osu_text)
        self.assertIn("448,192,120,1,0,0:0:0:0:", osu_text)
        self.assertIn("64,192,620,1,0,0:0:0:0:", osu_text)
        self.assertEqual(report["osu_mock_beatmap_path"], result["osu_mock_beatmap_path"])
        self.assertEqual(report["beat_count"], 5)
        self.assertEqual(report["hitobject_count"], 10)

    def test_timing_grid_from_report_reconstructs_segments(self) -> None:
        grid = timing_grid_from_report(_timing_report())

        self.assertEqual(len(grid.segments), 1)
        self.assertEqual(grid.segments[0].offset_ms, 120.0)
        self.assertEqual(grid.segments[0].beat_length_ms, 500.0)

    def test_main_prints_path_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            audio_path = root / "song.mp3"
            audio_path.write_bytes(b"")
            stdout = io.StringIO()

            with mock.patch(
                "pulsefield_model.timing.mock_osu_export.fit_audio_file",
                return_value=_timing_report(),
            ):
                with contextlib.redirect_stdout(stdout):
                    exit_code = main([audio_path.as_posix(), "--output-dir", (root / "out").as_posix()])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["audio_path"], audio_path.resolve(strict=False).as_posix())
        self.assertTrue(payload["osu_mock_beatmap_path"].endswith("song_timing_mock.osu"))


if __name__ == "__main__":
    unittest.main()
