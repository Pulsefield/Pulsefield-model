from __future__ import annotations

import torch

from pulsefield_model.inference.mapper_v2_1_rollout import (
    MapperV21FullRollout,
    grammar_constrained_window_generation_v2_1,
    rollout_to_timepoints_v2_1,
    zero_control_batch_provider_v2_1,
)
from pulsefield_model.models.mapper.v2_1 import MapperV21Vocab, empty_ln_carry_state
from pulsefield_model.models.mapper.v2_1.model import MapperV21Config, MapperV21Model


def test_sparse_window_generation_exports_grouped_timepoints() -> None:
    vocab = MapperV21Vocab()
    tokens = [
        vocab.time_shift_token_id(100),
        vocab.lane_action_token_id(0, "TAP"),
        vocab.lane_action_token_id(2, "TAP"),
        vocab.time_shift_token_id(4000),
        vocab.time_shift_token_id(3000),
        vocab.time_shift_token_id(900),
        vocab.eos_id,
    ]

    def logits_fn(step):
        logits = torch.full((vocab.size,), -1000.0)
        token_id = tokens[step.token_index]
        assert bool(step.valid_token_mask[token_id].item())
        logits[token_id] = 1000.0
        return logits

    window = grammar_constrained_window_generation_v2_1(
        vocab=vocab,
        write_start_ms=0,
        write_end_ms=8_000,
        chart_end_ms=8_000,
        ln_carry_in=empty_ln_carry_state(0),
        ln_carry_out=empty_ln_carry_state(8_000),
        logits_fn=logits_fn,
        is_full_chart_start=True,
        is_full_chart_end=True,
        max_tokens=16,
    )

    assert window.completed
    assert window.tokens == tokens
    rollout = MapperV21FullRollout(
        chart_end_ms=8_000,
        windows=[window],
        tokens=list(window.tokens),
        completed=True,
        dead_end=False,
        max_tokens_exceeded=False,
    )
    timepoints = rollout_to_timepoints_v2_1(rollout, vocab)
    assert len(timepoints) == 1
    assert timepoints[0].time_ms == 100
    assert [action.value for action in timepoints[0].lane_actions] == ["TAP", "NONE", "TAP", "NONE"]


def test_zero_control_batch_provider_matches_mapper_v2_1_shapes() -> None:
    model = MapperV21Model(
        MapperV21Config(
            control_dim=8,
            d_model=16,
            heads=4,
            layers=1,
            ffn_dim=32,
            max_seq_len=32,
            use_global_context=True,
            global_conv_blocks=0,
        ),
        vocab=MapperV21Vocab(),
    )

    batch = zero_control_batch_provider_v2_1(model=model, device="cpu")(0, 8_000)

    assert tuple(batch["projected_control_memory_8s"].shape) == (1, 400, 16)
    assert tuple(batch["density_teacher_8s"].shape) == (1, 400, 1)
    assert tuple(batch["global_memory"].shape) == (1, 4, 16)
