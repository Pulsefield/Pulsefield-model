from __future__ import annotations

import torch

from pulsefield_model.inference.mapper_v2_1_rollout import (
    MapperV21FullRollout,
    _apply_time_shift_length_penalty_v2_1,
    _time_shift_length_penalty_tensors_v2_1,
    grammar_constrained_window_generation_v2_1,
    mapper_v2_1_logits_fn,
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


def test_time_shift_delta_penalty_scales_by_shift_seconds() -> None:
    vocab = MapperV21Vocab()
    base_logits = torch.zeros(vocab.size, dtype=torch.float32)
    flat_only = _time_shift_length_penalty_tensors_v2_1(
        vocab,
        alpha=0.5,
        delta_alpha=0.0,
        device=torch.device("cpu"),
    )
    scaled = _time_shift_length_penalty_tensors_v2_1(
        vocab,
        alpha=0.5,
        delta_alpha=2.0,
        device=torch.device("cpu"),
    )

    flat_adjusted = _apply_time_shift_length_penalty_v2_1(base_logits, time_shift_penalty=flat_only)
    scaled_adjusted = _apply_time_shift_length_penalty_v2_1(base_logits, time_shift_penalty=scaled)

    ts_10 = vocab.time_shift_token_id(10)
    ts_4000 = vocab.time_shift_token_id(4000)
    lane_token = vocab.lane_action_token_id(0, "TAP")
    assert flat_adjusted[ts_10].item() == -0.5
    assert flat_adjusted[ts_4000].item() == -0.5
    assert torch.isclose(scaled_adjusted[ts_10], torch.tensor(-0.52))
    assert scaled_adjusted[ts_4000].item() == -8.5
    assert scaled_adjusted[lane_token].item() == 0.0


def test_logits_observer_cannot_mutate_generation_decision() -> None:
    vocab = MapperV21Vocab()
    chosen_ts = vocab.time_shift_token_id(100)
    alternate_ts = vocab.time_shift_token_id(10)
    expected_tokens = [chosen_ts, vocab.eos_id]

    def logits_fn(step):
        logits = torch.full((vocab.size,), -1000.0)
        logits[expected_tokens[step.token_index]] = 1000.0
        return logits

    def mutating_observer(step, logits):
        if step.token_index != 0:
            return
        logits[chosen_ts] = -1000.0
        logits[alternate_ts] = 2000.0
        step.valid_token_mask[chosen_ts] = False

    window = grammar_constrained_window_generation_v2_1(
        vocab=vocab,
        write_start_ms=0,
        write_end_ms=100,
        chart_end_ms=100,
        ln_carry_in=empty_ln_carry_state(0),
        ln_carry_out=empty_ln_carry_state(100),
        logits_fn=logits_fn,
        logits_observer=mutating_observer,
        is_full_chart_start=True,
        is_full_chart_end=True,
        max_tokens=4,
    )

    assert window.tokens == expected_tokens


def test_autoregressive_logits_skip_internal_grammar_mask_matches_greedy_tokens() -> None:
    torch.manual_seed(20260520)
    vocab = MapperV21Vocab()
    model = MapperV21Model(
        MapperV21Config(
            control_dim=8,
            d_model=16,
            heads=4,
            layers=1,
            ffn_dim=32,
            dropout=0.0,
            max_seq_len=32,
            use_global_context=False,
            global_conv_blocks=0,
        ),
        vocab=vocab,
    )
    model.eval()
    provider = zero_control_batch_provider_v2_1(model=model, device="cpu")

    def generate(*, apply_grammar_mask: bool):
        chart_end_ms = 500
        carry_in = empty_ln_carry_state(0)
        carry_out = empty_ln_carry_state(chart_end_ms)
        logits_fn = mapper_v2_1_logits_fn(
            model=model,
            vocab=vocab,
            device=torch.device("cpu"),
            normalized_difficulty=0.0,
            control_batch=dict(provider(0, 8_000)),
            ln_carry_in=carry_in,
            ln_carry_out=carry_out,
            write_start_ms=0,
            write_end_ms=8_000,
            chart_end_ms=chart_end_ms,
            is_full_chart_start=True,
            is_full_chart_end=True,
            time_shift_length_penalty_alpha=0.0,
            apply_grammar_mask=apply_grammar_mask,
        )
        return grammar_constrained_window_generation_v2_1(
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=8_000,
            chart_end_ms=chart_end_ms,
            ln_carry_in=carry_in,
            ln_carry_out=carry_out,
            logits_fn=logits_fn,
            is_full_chart_start=True,
            is_full_chart_end=True,
            max_tokens=16,
            temperature=0.0,
            top_p=None,
        )

    with_internal_mask = generate(apply_grammar_mask=True)
    without_internal_mask = generate(apply_grammar_mask=False)

    assert without_internal_mask.tokens
    assert without_internal_mask.tokens == with_internal_mask.tokens
    assert without_internal_mask.completed == with_internal_mask.completed
    assert without_internal_mask.dead_end == with_internal_mask.dead_end
    assert without_internal_mask.max_tokens_exceeded == with_internal_mask.max_tokens_exceeded
