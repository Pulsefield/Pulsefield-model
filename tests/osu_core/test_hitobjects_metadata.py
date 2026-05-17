import tempfile
import unittest
from pathlib import Path

from pulsefield_model.osu_core.hitobjects import ManiaHitObject, ManiaHitObjectKind, parse_mania_hit_objects
from pulsefield_model.osu_core.metadata import parse_osu_metadata


def _write_osu(path: Path, hitobject_lines: list[str]) -> None:
    path.write_text(
        "\n".join(
            [
                "osu file format v14",
                "",
                "[General]",
                "AudioFilename: song.mp3",
                "AudioLeadIn: 250",
                "PreviewTime: 12345",
                "Mode: 3",
                "",
                "[Metadata]",
                "Title: Test Title",
                "Artist: Test Artist",
                "Creator: Mapper",
                "Version: Hard",
                "BeatmapID: 42",
                "BeatmapSetID: 24",
                "",
                "[Difficulty]",
                "HPDrainRate: 7",
                "CircleSize:4",
                "OverallDifficulty: 8.5",
                "",
                "[HitObjects]",
                *hitobject_lines,
            ],
        ),
        encoding="utf-8",
    )


class OsuHitobjectsMetadataTests(unittest.TestCase):
    def test_parse_mania_hit_objects_preserves_zero_length_hold_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            osu_path = Path(tmp_dir) / "chart.osu"
            _write_osu(
                osu_path,
                [
                    "64,192,1000,1,0,0:0:0:0:",
                    "192,192,2000,128,0,2000:0:0:0:0:",
                ],
            )

            self.assertEqual(
                parse_mania_hit_objects(osu_path),
                [
                    ManiaHitObject(1000.0, 1000.0, 0, ManiaHitObjectKind.TAP),
                    ManiaHitObject(2000.0, 2000.0, 1, ManiaHitObjectKind.HOLD),
                ],
            )

    def test_parse_osu_metadata_derives_key_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            osu_path = Path(tmp_dir) / "chart.osu"
            _write_osu(osu_path, ["64,192,1000,1,0,0:0:0:0:"])

            metadata = parse_osu_metadata(osu_path)

        self.assertEqual(metadata.audio_filename, "song.mp3")
        self.assertEqual(metadata.audio_lead_in, 250)
        self.assertEqual(metadata.preview_time, 12345)
        self.assertEqual(metadata.mode, 3)
        self.assertEqual(metadata.title, "Test Title")
        self.assertEqual(metadata.artist, "Test Artist")
        self.assertEqual(metadata.creator, "Mapper")
        self.assertEqual(metadata.version, "Hard")
        self.assertEqual(metadata.beatmap_id, 42)
        self.assertEqual(metadata.beatmap_set_id, 24)
        self.assertEqual(metadata.hp_drain_rate, 7.0)
        self.assertEqual(metadata.circle_size, 4.0)
        self.assertEqual(metadata.overall_difficulty, 8.5)
        self.assertEqual(metadata.key_count, 4)


if __name__ == "__main__":
    unittest.main()
