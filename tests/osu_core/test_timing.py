import tempfile
import textwrap
import unittest
from pathlib import Path

from pulsefield_model.osu_core.timing import (
    InvalidRedTimingError,
    MissingRedTimingError,
    RedTimingPoint,
    parse_red_timing_points,
    red_timing_point_at,
    require_red_timing_points,
)


def _write_osu(path: Path, timing_lines: list[str]) -> None:
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


class OsuTimingTests(unittest.TestCase):
    def test_parse_red_timing_points_ignores_green_lines_and_sorts_by_offset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            osu_path = Path(tmp_dir) / "chart.osu"
            _write_osu(
                osu_path,
                [
                    "1000,500,4,2,0,80,1,0",
                    "1250,-100,4,2,0,80,0,0",
                    "-250,600,3,2,0,70,1,0",
                    "2000,333.333,7,2,0,80,1,8",
                ],
            )

            self.assertEqual(
                parse_red_timing_points(osu_path),
                [
                    RedTimingPoint(offset_ms=-250.0, beat_length_ms=600.0, meter=3),
                    RedTimingPoint(offset_ms=1000.0, beat_length_ms=500.0, meter=4),
                    RedTimingPoint(offset_ms=2000.0, beat_length_ms=333.333, meter=7),
                ],
            )

    def test_require_rejects_missing_and_invalid_red_timing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_path = Path(tmp_dir) / "missing.osu"
            _write_osu(missing_path, ["0,-100,4,2,0,80,0,0"])
            with self.assertRaisesRegex(MissingRedTimingError, "no red timing point"):
                require_red_timing_points(missing_path)

            invalid_path = Path(tmp_dir) / "invalid.osu"
            _write_osu(invalid_path, ["0,0,4,2,0,80,1,0"])
            with self.assertRaisesRegex(InvalidRedTimingError, "nonpositive=1"):
                require_red_timing_points(invalid_path)

    def test_red_timing_point_at_extrapolates_with_first_and_last_red_point(self) -> None:
        timing_points = [
            RedTimingPoint(offset_ms=1000.0, beat_length_ms=500.0, meter=4),
            RedTimingPoint(offset_ms=3000.0, beat_length_ms=250.0, meter=3),
        ]

        self.assertEqual(red_timing_point_at(timing_points, -2000.0), timing_points[0])
        self.assertEqual(red_timing_point_at(timing_points, 2999.0), timing_points[0])
        self.assertEqual(red_timing_point_at(timing_points, 3000.0), timing_points[1])
        self.assertEqual(red_timing_point_at(timing_points, 6000.0), timing_points[1])

    def test_timing_lines_can_omit_optional_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            osu_path = Path(tmp_dir) / "chart.osu"
            _write_osu(osu_path, ["0,500"])

            self.assertEqual(
                parse_red_timing_points(osu_path),
                [RedTimingPoint(offset_ms=0.0, beat_length_ms=500.0, meter=4)],
            )


if __name__ == "__main__":
    unittest.main()
