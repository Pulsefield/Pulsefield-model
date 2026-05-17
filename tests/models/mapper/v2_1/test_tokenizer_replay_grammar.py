import unittest

import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")

import torch

from pulsefield_model.models.mapper.v2_1.grammar import build_grammar_mask
from pulsefield_model.models.mapper.v2_1.replay import (
    ln_carry_state_from_open_starts,
    ln_carry_state_tensors,
    replay_terminal_state,
    replay_tokens,
)
from pulsefield_model.models.mapper.v2_1.tokenizer import MapperTimepoint, encode_mapper_window
from pulsefield_model.models.mapper.v2_1.vocab import LaneAction, MapperV21Vocab


def _actions(*actions: LaneAction) -> tuple[LaneAction, ...]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(LaneAction.NONE)
    return tuple(padded)


class MapperV21TokenizerReplayGrammarTests(unittest.TestCase):
    def test_sparse_same_time_tokens_replay_and_grammar_state(self) -> None:
        vocab = MapperV21Vocab()
        tokenized = encode_mapper_window(
            [
                MapperTimepoint(
                    1000,
                    (
                        LaneAction.TAP,
                        LaneAction.NONE,
                        LaneAction.HOLD_START,
                        LaneAction.TAP,
                    ),
                ),
                MapperTimepoint(1300, _actions(LaneAction.NONE, LaneAction.NONE, LaneAction.HOLD_END)),
            ],
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=8000,
        )

        self.assertEqual(
            [vocab.token_name(token_id) for token_id in tokenized.target_fragment_ids[:6]],
            [
                "TS_1000",
                "LANE_1_TAP",
                "LANE_3_HOLD_START",
                "LANE_4_TAP",
                "TS_300",
                "LANE_3_HOLD_END",
            ],
        )

        replay_states = replay_tokens(
            tokenized.target_fragment_ids,
            vocab=vocab,
            write_start_ms=tokenized.write_start_ms,
            write_end_ms=tokenized.write_end_ms,
            ln_carry_in=tokenized.ln_carry_in,
            ln_carry_out=tokenized.ln_carry_out,
        )
        self.assertTrue(replay_states[2].emitted_lane_mask[0])
        self.assertEqual(replay_states[2].last_lane_index, 0)

        terminal = replay_terminal_state(
            tokenized.target_fragment_ids,
            vocab=vocab,
            write_start_ms=tokenized.write_start_ms,
            write_end_ms=tokenized.write_end_ms,
            ln_carry_in=tokenized.ln_carry_in,
            ln_carry_out=tokenized.ln_carry_out,
        )
        self.assertEqual(terminal.current_ms, 8000)
        self.assertFalse(any(terminal.open_mask))

        grammar_mask = build_grammar_mask(
            current_ms=tokenized.target_fragment_current_ms.unsqueeze(0),
            open_mask=tokenized.target_fragment_open_mask.unsqueeze(0),
            open_start_ms=tokenized.target_fragment_open_start_ms.unsqueeze(0),
            open_age_ms=tokenized.target_fragment_open_age_ms.unsqueeze(0),
            emitted_lane_mask=tokenized.target_fragment_emitted_lane_mask.unsqueeze(0),
            last_lane_index=tokenized.target_fragment_last_lane_index.unsqueeze(0),
            write_start_ms=torch.tensor([tokenized.write_start_ms]),
            write_end_ms=torch.tensor([tokenized.write_end_ms]),
            ln_carry_in=ln_carry_state_tensors(tokenized.ln_carry_in),
            ln_carry_out=ln_carry_state_tensors(tokenized.ln_carry_out),
            is_full_chart_start=torch.tensor([tokenized.is_full_chart_start]),
            is_full_chart_end=torch.tensor([tokenized.is_full_chart_end]),
            vocab=vocab,
        )
        lane_1_tap = vocab.lane_action_token_id(0, LaneAction.TAP)
        lane_3_hold_start = vocab.lane_action_token_id(2, LaneAction.HOLD_START)
        self.assertTrue(torch.isneginf(grammar_mask[0, 2, lane_1_tap]))
        self.assertEqual(float(grammar_mask[0, 2, lane_3_hold_start].item()), 0.0)

    def test_cross_window_hold_is_represented_by_boundary_carry_state(self) -> None:
        vocab = MapperV21Vocab()

        tokenized = encode_mapper_window(
            [
                MapperTimepoint(500, _actions(LaneAction.HOLD_START)),
                MapperTimepoint(9500, _actions(LaneAction.HOLD_END)),
            ],
            vocab=vocab,
            write_start_ms=1000,
            write_end_ms=9000,
        )

        self.assertEqual(
            tokenized.ln_carry_in,
            ln_carry_state_from_open_starts(1000, (500, None, None, None)),
        )
        self.assertTrue(tokenized.ln_carry_out.open_mask[0])
        self.assertEqual(tokenized.ln_carry_out.open_start_ms[0], 500)
        self.assertEqual(
            [vocab.token_name(token_id) for token_id in tokenized.target_fragment_ids],
            ["TS_4000", "TS_4000"],
        )

    def test_terminal_padded_window_places_eos_at_chart_end(self) -> None:
        vocab = MapperV21Vocab()

        tokenized = encode_mapper_window(
            [
                MapperTimepoint(8500, _actions(LaneAction.TAP)),
                MapperTimepoint(9000, _actions(LaneAction.NONE, LaneAction.TAP)),
            ],
            vocab=vocab,
            write_start_ms=8000,
            write_end_ms=16000,
            chart_end_ms=9000,
        )

        self.assertTrue(tokenized.is_full_chart_end)
        self.assertEqual(tokenized.write_end_ms, 16000)
        self.assertEqual(tokenized.chart_end_ms, 9000)
        self.assertEqual(
            [vocab.token_name(token_id) for token_id in tokenized.target_fragment_ids],
            ["TS_500", "LANE_1_TAP", "TS_500", "LANE_2_TAP", "EOS"],
        )
        self.assertEqual(int(tokenized.target_fragment_current_ms[-1].item()), 9000)

        terminal = replay_terminal_state(
            tokenized.target_fragment_ids,
            vocab=vocab,
            write_start_ms=tokenized.write_start_ms,
            write_end_ms=tokenized.write_end_ms,
            chart_end_ms=tokenized.chart_end_ms,
            ln_carry_in=tokenized.ln_carry_in,
            ln_carry_out=tokenized.ln_carry_out,
            is_full_chart_end=tokenized.is_full_chart_end,
        )
        self.assertEqual(terminal.current_ms, 9000)

        grammar_mask = build_grammar_mask(
            current_ms=tokenized.target_fragment_current_ms.unsqueeze(0),
            open_mask=tokenized.target_fragment_open_mask.unsqueeze(0),
            open_start_ms=tokenized.target_fragment_open_start_ms.unsqueeze(0),
            open_age_ms=tokenized.target_fragment_open_age_ms.unsqueeze(0),
            emitted_lane_mask=tokenized.target_fragment_emitted_lane_mask.unsqueeze(0),
            last_lane_index=tokenized.target_fragment_last_lane_index.unsqueeze(0),
            write_start_ms=torch.tensor([tokenized.write_start_ms]),
            write_end_ms=torch.tensor([tokenized.write_end_ms]),
            chart_end_ms=torch.tensor([tokenized.chart_end_ms]),
            ln_carry_in=ln_carry_state_tensors(tokenized.ln_carry_in),
            ln_carry_out=ln_carry_state_tensors(tokenized.ln_carry_out),
            is_full_chart_start=torch.tensor([tokenized.is_full_chart_start]),
            is_full_chart_end=torch.tensor([tokenized.is_full_chart_end]),
            vocab=vocab,
        )
        self.assertEqual(float(grammar_mask[0, -1, vocab.eos_id].item()), 0.0)


if __name__ == "__main__":
    unittest.main()
