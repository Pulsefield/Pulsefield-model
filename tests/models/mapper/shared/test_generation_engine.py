from __future__ import annotations

from dataclasses import dataclass

import torch

from pulsefield_model.models.mapper.shared.generation import grammar_constrained_window_generation
from pulsefield_model.models.mapper.shared.generation_engine import (
    IncrementalPrefixDecoder,
    apply_top_p,
    apply_time_shift_penalty,
    default_generation_logits,
    select_next_token,
    time_shift_penalty_tensors,
)
from pulsefield_model.models.mapper.shared.replay import empty_ln_carry_state
from pulsefield_model.models.mapper.shared.vocab import MapperTupleVocab


TUPLE_LEGACY_DEFAULT_DETERMINISTIC_FIXTURE = {
    "tokens": [13, 7, 2],
    "states_before": [
        {
            "current_ms": 0,
            "open_mask": (False, False, False, False),
            "open_start_ms": (None, None, None, None),
            "open_age_ms": (0, 0, 0, 0),
        },
        {
            "current_ms": 200,
            "open_mask": (False, False, False, False),
            "open_start_ms": (None, None, None, None),
            "open_age_ms": (0, 0, 0, 0),
        },
        {
            "current_ms": 250,
            "open_mask": (False, False, False, False),
            "open_start_ms": (None, None, None, None),
            "open_age_ms": (0, 0, 0, 0),
        },
    ],
    "states_after": [
        {
            "current_ms": 200,
            "open_mask": (False, False, False, False),
            "open_start_ms": (None, None, None, None),
            "open_age_ms": (0, 0, 0, 0),
        },
        {
            "current_ms": 250,
            "open_mask": (False, False, False, False),
            "open_start_ms": (None, None, None, None),
            "open_age_ms": (0, 0, 0, 0),
        },
        {
            "current_ms": 250,
            "open_mask": (False, False, False, False),
            "open_start_ms": (None, None, None, None),
            "open_age_ms": (0, 0, 0, 0),
        },
    ],
    "terminal_state": {
        "current_ms": 250,
        "open_mask": (False, False, False, False),
        "open_start_ms": (None, None, None, None),
        "open_age_ms": (0, 0, 0, 0),
    },
    "completed": True,
    "dead_end": False,
    "max_tokens_exceeded": False,
}


def _tuple_window_parity_payload(window) -> dict[str, object]:
    payload = window.to_dict()
    return {
        "tokens": payload["tokens"],
        "states_before": payload["states_before"],
        "states_after": payload["states_after"],
        "terminal_state": payload["terminal_state"],
        "completed": payload["completed"],
        "dead_end": payload["dead_end"],
        "max_tokens_exceeded": payload["max_tokens_exceeded"],
    }


def test_select_next_token_applies_valid_mask_before_greedy_selection() -> None:
    logits = torch.tensor([0.0, 10.0, 9.0], dtype=torch.float32)
    valid_mask = torch.tensor([True, False, True])

    token_id = select_next_token(logits, valid_mask=valid_mask, temperature=0.0)

    assert token_id == 2


def test_select_next_token_top_p_keeps_at_least_highest_probability_token() -> None:
    generator = torch.Generator().manual_seed(123)
    logits = torch.tensor([10.0, 9.0, 0.0], dtype=torch.float32)
    valid_mask = torch.ones(3, dtype=torch.bool)

    token_id = select_next_token(
        logits,
        valid_mask=valid_mask,
        temperature=1.0,
        top_p=0.5,
        generator=generator,
    )

    assert token_id == 0


def test_apply_top_p_drops_token_that_crosses_threshold() -> None:
    probs = torch.tensor([0.40, 0.35, 0.25], dtype=torch.float32)

    filtered = apply_top_p(probs, p=0.5)

    assert torch.allclose(filtered, torch.tensor([1.0, 0.0, 0.0]))


def test_apply_top_p_preserves_at_least_one_token_below_top_probability() -> None:
    probs = torch.tensor([0.20, 0.30, 0.50], dtype=torch.float32)

    filtered = apply_top_p(probs, p=0.1)

    assert torch.allclose(filtered, torch.tensor([0.0, 0.0, 1.0]))


def test_select_next_token_top_k_filters_sampling_candidates() -> None:
    generator = torch.Generator().manual_seed(123)
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
    valid_mask = torch.ones(4, dtype=torch.bool)

    token_id = select_next_token(
        logits,
        valid_mask=valid_mask,
        temperature=1.0,
        top_k=1,
        generator=generator,
    )

    assert token_id == 3


def test_default_generation_logits_prefers_longest_valid_time_shift() -> None:
    vocab = MapperTupleVocab()
    mask = torch.zeros(vocab.size, dtype=torch.bool)
    ts_50 = vocab.time_shift_token_id(50)
    ts_200 = vocab.time_shift_token_id(200)
    event_id = vocab.event_token_ids[0]
    mask[ts_50] = True
    mask[ts_200] = True
    mask[event_id] = True

    logits = default_generation_logits(
        mask,
        vocab_size=vocab.size,
        time_shift_token_ids=vocab.time_shift_token_ids,
        time_shift_value=vocab.time_shift_value,
    )

    assert int(torch.argmax(logits).item()) == ts_200
    assert torch.isneginf(logits[vocab.pad_id])


def test_time_shift_penalty_supports_flat_and_delta_scaled_terms() -> None:
    vocab = MapperTupleVocab()
    logits = torch.zeros(vocab.size, dtype=torch.float32)
    ts_50 = vocab.time_shift_token_id(50)
    ts_1000 = vocab.time_shift_token_id(1000)
    event_id = vocab.event_token_ids[0]

    penalty = time_shift_penalty_tensors(
        vocab,
        alpha=0.5,
        delta_alpha=2.0,
        device=torch.device("cpu"),
    )
    adjusted = apply_time_shift_penalty(logits, time_shift_penalty=penalty)

    assert torch.isclose(adjusted[ts_50], torch.tensor(-0.6))
    assert adjusted[ts_1000].item() == -2.5
    assert adjusted[event_id].item() == 0.0


def test_incremental_prefix_decoder_only_decodes_new_suffix_tokens() -> None:
    calls: list[tuple[int, int, int]] = []
    state_counter = 0

    @dataclass(frozen=True)
    class Output:
        decode_state: int
        logits_final: torch.Tensor

    def create_empty_decode_state(*, batch_size: int, device: torch.device) -> int:
        nonlocal state_counter
        assert batch_size == 1
        assert device == torch.device("cpu")
        state_counter += 1
        return state_counter

    decoder = IncrementalPrefixDecoder(
        create_empty_decode_state=create_empty_decode_state,
        batch_size=1,
        device=torch.device("cpu"),
    )

    def decode_one(decode_state: int, position: int) -> Output:
        token_id = active_prefix[position]
        calls.append((decode_state, position, token_id))
        logits = torch.zeros((1, 4), dtype=torch.float32)
        logits[0, token_id] = 1.0
        return Output(decode_state=decode_state, logits_final=logits)

    active_prefix = (1,)
    first = decoder.decode(active_prefix, decode_one=decode_one)
    active_prefix = (1, 2)
    second = decoder.decode(active_prefix, decode_one=decode_one)
    second_again = decoder.decode(active_prefix, decode_one=decode_one)
    active_prefix = (3,)
    reset_logits = decoder.decode(active_prefix, decode_one=decode_one)

    assert int(torch.argmax(first).item()) == 1
    assert int(torch.argmax(second).item()) == 2
    assert int(torch.argmax(second_again).item()) == 2
    assert int(torch.argmax(reset_logits).item()) == 3
    assert calls == [(1, 0, 1), (1, 1, 2), (2, 0, 3)]


def test_tuple_window_generation_matches_legacy_default_deterministic_fixture() -> None:
    vocab = MapperTupleVocab()

    window = grammar_constrained_window_generation(
        vocab=vocab,
        write_start_ms=0,
        write_end_ms=250,
        ln_carry_in=empty_ln_carry_state(0),
        ln_carry_out=empty_ln_carry_state(250),
        is_full_chart_start=True,
        is_full_chart_end=True,
        max_tokens=4,
    )

    assert [vocab.token_name(token_id) for token_id in window.tokens] == ["TS_200", "TS_50", "EOS"]
    assert _tuple_window_parity_payload(window) == TUPLE_LEGACY_DEFAULT_DETERMINISTIC_FIXTURE
