import tempfile
import unittest
from pathlib import Path

from pulsefield_model.osu_core.beat_representation import (
    BeatEventKind,
    beatmap_to_beat_representation,
    parse_mania_beat_events,
)
from pulsefield_model.timing.canonicalization import TIMING_CANONICALIZATION_NONE


def _write_osu(
    path: Path,
    *,
    timing_lines: list[str],
    hitobject_lines: list[str],
    circle_size: int = 4,
) -> None:
    path.write_text(
        "\n".join(
            [
                "osu file format v14",
                "",
                "[General]",
                "Mode: 3",
                "",
                "[Difficulty]",
                f"CircleSize:{circle_size}",
                "",
                "[TimingPoints]",
                *timing_lines,
                "",
                "[HitObjects]",
                *hitobject_lines,
            ],
        ),
        encoding="utf-8",
    )


class BeatRepresentationTests(unittest.TestCase):
    def test_default_snap_uses_canonical_bpm_80_160(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            osu_path = Path(tmp_dir) / "chart.osu"
            _write_osu(
                osu_path,
                timing_lines=["0,315.789,4,2,0,80,1,0"],
                hitobject_lines=["64,192,316,1,0,0:0:0:0:"],
            )

            events = parse_mania_beat_events(osu_path)

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.kind, BeatEventKind.TAP)
        self.assertEqual(event.beat_offset_numerator, 24)
        self.assertEqual(event.beat_offset_denominator, 48)
        self.assertAlmostEqual(event.redline_beat_length_ms, 315.789)
        self.assertAlmostEqual(event.redline_bpm, 60000.0 / 315.789)
        self.assertAlmostEqual(event.beat_length_ms, 631.578)
        self.assertAlmostEqual(event.bpm, 60000.0 / 631.578)
        self.assertAlmostEqual(event.beat_offset, 0.5)
        self.assertAlmostEqual(event.snapped_time_ms, 315.789)
        self.assertAlmostEqual(event.snap_error_ms, 0.211)

    def test_timing_canonicalization_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            osu_path = Path(tmp_dir) / "chart.osu"
            _write_osu(
                osu_path,
                timing_lines=["0,315.789,4,2,0,80,1,0"],
                hitobject_lines=["64,192,316,1,0,0:0:0:0:"],
            )

            events = parse_mania_beat_events(osu_path, timing_canonicalization=TIMING_CANONICALIZATION_NONE)

        self.assertEqual(events[0].beat_offset_numerator, 48)
        self.assertAlmostEqual(events[0].beat_offset, 1.0)
        self.assertAlmostEqual(events[0].beat_length_ms, 315.789)

    def test_hold_start_and_end_use_last_redline_at_each_event_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            osu_path = Path(tmp_dir) / "chart.osu"
            _write_osu(
                osu_path,
                timing_lines=[
                    "0,500,4,2,0,80,1,0",
                    "2000,250,4,2,0,80,1,0",
                ],
                hitobject_lines=["192,192,1500,128,0,2250:0:0:0:0:"],
            )

            events = parse_mania_beat_events(osu_path)

        self.assertEqual([event.kind for event in events], [BeatEventKind.HOLD_START, BeatEventKind.HOLD_END])
        self.assertEqual(events[0].lane, 1)
        self.assertEqual(events[0].redline_offset_ms, 0.0)
        self.assertEqual(events[0].beat_length_ms, 500.0)
        self.assertEqual(events[0].beat_offset, 3.0)
        self.assertEqual(events[1].redline_offset_ms, 2000.0)
        self.assertEqual(events[1].redline_beat_length_ms, 250.0)
        self.assertEqual(events[1].beat_length_ms, 500.0)
        self.assertEqual(events[1].beat_offset, 0.5)

    def test_fractional_snap_denominator_preserves_subdivision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            osu_path = Path(tmp_dir) / "chart.osu"
            _write_osu(
                osu_path,
                timing_lines=["0,500,4,2,0,80,1,0"],
                hitobject_lines=["64,192,1125,1,0,0:0:0:0:"],
            )

            events = parse_mania_beat_events(osu_path, snap_denominator=4)

        self.assertEqual(events[0].beat_offset_numerator, 9)
        self.assertEqual(events[0].beat_offset_denominator, 4)
        self.assertEqual(events[0].beat_offset, 2.25)
        self.assertEqual(events[0].snap_error_ms, 0.0)

    def test_exact_half_snap_uses_half_up_rounding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            osu_path = Path(tmp_dir) / "chart.osu"
            _write_osu(
                osu_path,
                timing_lines=["0,500,4,2,0,80,1,0"],
                hitobject_lines=["64,192,125,1,0,0:0:0:0:"],
            )

            events = parse_mania_beat_events(osu_path, snap_denominator=2)

        self.assertEqual(events[0].beat_offset_numerator, 1)
        self.assertEqual(events[0].beat_offset_denominator, 2)
        self.assertEqual(events[0].beat_offset, 0.5)
        self.assertEqual(events[0].snapped_time_ms, 250.0)
        self.assertEqual(events[0].snap_error_ms, -125.0)

    def test_optional_diagnostics_report_subdivision_relative_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            osu_path = Path(tmp_dir) / "chart.osu"
            _write_osu(
                osu_path,
                timing_lines=["0,500,4,2,0,80,1,0"],
                hitobject_lines=["64,192,1126,1,0,0:0:0:0:"],
            )

            summary = beatmap_to_beat_representation(
                osu_path,
                snap_denominator=4,
                include_diagnostics=True,
                diagnostic_subdivisions=(1, 2, 4),
            )

        event = summary["events"][0]
        diagnostics = event["diagnostics"]
        self.assertEqual(event["beat_offset_fraction"], "9/4")
        self.assertEqual(diagnostics["subdivision"], 4)
        self.assertAlmostEqual(diagnostics["raw_beat_offset"], 2.252)
        self.assertAlmostEqual(diagnostics["snap_error_beats"], 0.002)
        self.assertAlmostEqual(diagnostics["relative_error_to_subdivision"], 0.008)
        self.assertAlmostEqual(summary["max_relative_error_to_subdivision"], 0.008)

    def test_beatmap_summary_can_limit_printed_events_without_losing_counts_or_adding_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            osu_path = Path(tmp_dir) / "chart.osu"
            _write_osu(
                osu_path,
                timing_lines=["0,500,4,2,0,80,1,0"],
                hitobject_lines=[
                    "64,192,500,1,0,0:0:0:0:",
                    "192,192,1000,1,0,0:0:0:0:",
                ],
            )

            summary = beatmap_to_beat_representation(osu_path, limit_events=1)

        self.assertEqual(summary["hitobject_count"], 2)
        self.assertEqual(summary["event_count"], 2)
        self.assertEqual(summary["truncated_event_count"], 1)
        self.assertEqual(len(summary["events"]), 1)
        self.assertEqual(summary["events"][0]["beat_offset_fraction"], "1")
        self.assertNotIn("diagnostics", summary["events"][0])
        self.assertEqual(summary["timing_canonicalization"], "bpm-80-160")


if __name__ == "__main__":
    unittest.main()
