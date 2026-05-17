import unittest

from pulsefield_model.events.canonical import CanonicalTimepoint
from pulsefield_model.events.canonical import LaneAction as CanonicalLaneAction
from pulsefield_model.inference.osu_export import (
    OsuExportMetadata,
    decode_mapper_tokens_to_timepoints,
    format_osu_export,
)
from pulsefield_model.models.mapper.shared.vocab import LaneAction as MapperLaneAction
from pulsefield_model.models.mapper.shared.vocab import MapperTupleVocab


def _mapper_actions(*actions: MapperLaneAction) -> tuple[MapperLaneAction, ...]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(MapperLaneAction.NONE)
    return tuple(padded)


def _canonical_actions(*actions: CanonicalLaneAction) -> tuple[CanonicalLaneAction, ...]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(CanonicalLaneAction.NONE)
    return tuple(padded)


class OsuExportTests(unittest.TestCase):
    def test_decode_mapper_tokens_to_canonical_timepoints(self) -> None:
        vocab = MapperTupleVocab()
        tokens = [
            vocab.bos_id,
            vocab.pad_id,
            vocab.time_shift_token_id(1000),
            vocab.encode_event(_mapper_actions(MapperLaneAction.TAP)),
            vocab.time_shift_token_id(500),
            vocab.encode_event(
                _mapper_actions(MapperLaneAction.HOLD_START, MapperLaneAction.TAP),
            ),
            vocab.time_shift_token_id(300),
            vocab.encode_event(_mapper_actions(MapperLaneAction.HOLD_END)),
            vocab.time_shift_token_id(4000),
            vocab.time_shift_token_id(2000),
            vocab.time_shift_token_id(200),
            vocab.eos_id,
        ]

        timepoints = decode_mapper_tokens_to_timepoints(tokens, vocab)

        self.assertEqual(
            timepoints,
            [
                CanonicalTimepoint(
                    time_ms=1000,
                    lane_actions=_canonical_actions(CanonicalLaneAction.TAP),
                ),
                CanonicalTimepoint(
                    time_ms=1500,
                    lane_actions=_canonical_actions(
                        CanonicalLaneAction.HOLD_START,
                        CanonicalLaneAction.TAP,
                    ),
                ),
                CanonicalTimepoint(
                    time_ms=1800,
                    lane_actions=_canonical_actions(CanonicalLaneAction.HOLD_END),
                ),
            ],
        )

    def test_format_osu_export_includes_sections_and_hitobjects(self) -> None:
        timepoints = [
            CanonicalTimepoint(
                time_ms=1000,
                lane_actions=_canonical_actions(CanonicalLaneAction.TAP),
            ),
            CanonicalTimepoint(
                time_ms=1500,
                lane_actions=_canonical_actions(
                    CanonicalLaneAction.HOLD_START,
                    CanonicalLaneAction.TAP,
                ),
            ),
            CanonicalTimepoint(
                time_ms=1800,
                lane_actions=_canonical_actions(CanonicalLaneAction.HOLD_END),
            ),
        ]

        osu_text = format_osu_export(
            timepoints,
            metadata=OsuExportMetadata(
                audio_filename="preview.ogg",
                title="Token Decode",
                artist="Mapper",
                creator="s3d-i",
                version="8s generated",
            ),
        )

        self.assertTrue(osu_text.startswith("osu file format v14\n"))
        self.assertIn("[General]\nAudioFilename:preview.ogg", osu_text)
        self.assertIn("Mode:3", osu_text)
        self.assertIn("[Metadata]\nTitle:Token Decode", osu_text)
        self.assertIn("Creator:s3d-i\nVersion:8s generated", osu_text)
        self.assertIn(
            "[Difficulty]\nHPDrainRate:5\nCircleSize:4\nOverallDifficulty:5",
            osu_text,
        )
        self.assertIn("[TimingPoints]\n0,500,4,2,0,100,1,0", osu_text)
        self.assertIn("[HitObjects]\n64,192,1000,1,0,0:0:0:0:", osu_text)
        self.assertIn("64,192,1500,128,0,1800:0:0:0:0:", osu_text)
        self.assertIn("192,192,1500,1,0,0:0:0:0:", osu_text)

    def test_invalid_mapper_sequence_fails_during_decode(self) -> None:
        vocab = MapperTupleVocab()
        tokens = [
            vocab.encode_event(_mapper_actions(MapperLaneAction.HOLD_END)),
        ]

        with self.assertRaisesRegex(ValueError, "HOLD_END is illegal on closed lane 0"):
            decode_mapper_tokens_to_timepoints(tokens, vocab)


if __name__ == "__main__":
    unittest.main()
