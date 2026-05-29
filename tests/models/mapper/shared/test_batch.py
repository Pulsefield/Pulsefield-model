import pytest
import torch

from pulsefield_model.models.mapper.shared import MapperBatch, MapperFragmentState, MapperTokenContract
from pulsefield_model.models.mapper.shared.vocab import MapperTupleVocab
from pulsefield_model.models.mapper.v2_1.vocab import MapperV21Vocab


def test_tuple_contract_parses_minimal_batch_and_normalizes_tensors() -> None:
    vocab = MapperTupleVocab()
    contract = MapperTokenContract(name="tuple", vocab=vocab)
    raw = _minimal_batch(vocab, sparse=False)

    batch = MapperBatch.from_mapping(raw, contract=contract)

    assert batch.batch_size == 2
    assert batch.seq_len == 3
    assert batch.contract.pad_id == vocab.pad_id
    assert batch.contract.bos_id == vocab.bos_id
    assert batch.contract.eos_id == vocab.eos_id
    assert batch.contract.vocab_size == vocab.size
    assert batch.decoder_input_tokens.shape == (2, 3)
    assert batch.decoder_input_tokens.dtype == torch.long
    assert batch.target_fragment_tokens.dtype == torch.long
    assert batch.target_fragment_mask.dtype == torch.bool
    assert batch.target_fragment_states.current_ms.dtype == torch.long
    assert batch.target_fragment_states.open_mask.dtype == torch.bool
    assert batch.target_fragment_states.emitted_lane_mask is None
    assert batch.target_fragment_states.last_lane_index is None
    assert torch.equal(batch.input_padding_mask, ~raw["target_fragment_mask"].to(dtype=torch.bool))
    assert torch.equal(batch.positions(), torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long))
    assert torch.equal(batch.target_end_ms, batch.write_end_ms)
    assert set(batch.target_fragment_states.as_mapping()) == {
        "current_ms",
        "open_mask",
        "open_start_ms",
        "open_age_ms",
    }
    assert set(batch.ln_carry_in.as_mapping()) == {
        "current_ms",
        "open_mask",
        "open_start_ms",
        "open_age_ms",
    }


def test_v21_contract_requires_and_parses_sparse_state() -> None:
    vocab = MapperV21Vocab()
    contract = MapperTokenContract(
        name="v2_1",
        vocab=vocab,
        requires_sparse_lane_state=True,
        uses_chart_end_for_terminal_windows=True,
    )
    raw = _minimal_batch(vocab, sparse=True)
    raw["write_end_ms"] = torch.tensor([8000, 9000], dtype=torch.long)
    raw["chart_end_ms"] = torch.tensor([1230, 4560], dtype=torch.long)
    raw["is_full_chart_end"] = torch.tensor([True, False], dtype=torch.bool)

    batch = MapperBatch.from_mapping(raw, contract=contract)

    assert batch.target_fragment_states.emitted_lane_mask is not None
    assert batch.target_fragment_states.last_lane_index is not None
    assert batch.target_fragment_states.emitted_lane_mask.dtype == torch.bool
    assert batch.target_fragment_states.last_lane_index.dtype == torch.long
    assert torch.equal(batch.target_end_ms, torch.tensor([1230, 9000], dtype=torch.long))
    assert "emitted_lane_mask" in batch.target_fragment_states.as_mapping()
    assert "last_lane_index" in batch.target_fragment_states.as_mapping()


@pytest.mark.parametrize("field", ["emitted_lane_mask", "last_lane_index"])
def test_missing_sparse_fields_raise_useful_error(field: str) -> None:
    vocab = MapperV21Vocab()
    contract = MapperTokenContract(name="v2_1", vocab=vocab, requires_sparse_lane_state=True)
    raw = _minimal_batch(vocab, sparse=True)
    raw["target_fragment_states"] = dict(raw["target_fragment_states"])
    del raw["target_fragment_states"][field]

    with pytest.raises(ValueError, match=field):
        MapperBatch.from_mapping(raw, contract=contract)


def test_tuple_padded_state_sanitization_matches_existing_behavior() -> None:
    state = MapperFragmentState.from_mapping(
        _fragment_state_values(sparse=False),
        batch_shape=(2, 3),
        device=torch.device("cpu"),
        requires_sparse_lane_state=False,
    )
    valid = torch.tensor([[True, False, True], [False, True, False]], dtype=torch.bool)
    target_end_ms = torch.tensor([8000, 9000], dtype=torch.long)

    sanitized = state.sanitized(target_end_ms, valid)

    padded = ~valid
    assert torch.equal(
        sanitized.current_ms,
        torch.where(padded, target_end_ms.reshape(-1, 1), state.current_ms),
    )
    assert torch.equal(
        sanitized.open_mask,
        torch.where(padded.unsqueeze(-1), torch.zeros_like(state.open_mask), state.open_mask),
    )
    assert torch.equal(
        sanitized.open_start_ms,
        torch.where(padded.unsqueeze(-1), torch.full_like(state.open_start_ms, -1), state.open_start_ms),
    )
    assert torch.equal(
        sanitized.open_age_ms,
        torch.where(padded.unsqueeze(-1), torch.zeros_like(state.open_age_ms), state.open_age_ms),
    )
    assert sanitized.emitted_lane_mask is None
    assert sanitized.last_lane_index is None


def test_v21_padded_state_sanitization_clears_sparse_state() -> None:
    state = MapperFragmentState.from_mapping(
        _fragment_state_values(sparse=True),
        batch_shape=(2, 3),
        device=torch.device("cpu"),
        requires_sparse_lane_state=True,
    )
    valid = torch.tensor([[True, False, True], [False, True, False]], dtype=torch.bool)
    target_end_ms = torch.tensor([8000, 9000], dtype=torch.long)

    sanitized = state.sanitized(target_end_ms, valid)

    assert sanitized.emitted_lane_mask is not None
    assert sanitized.last_lane_index is not None
    padded = ~valid
    assert torch.equal(
        sanitized.emitted_lane_mask,
        torch.where(padded.unsqueeze(-1), torch.zeros_like(state.emitted_lane_mask), state.emitted_lane_mask),
    )
    assert torch.equal(
        sanitized.last_lane_index,
        torch.where(padded, torch.full_like(state.last_lane_index, -1), state.last_lane_index),
    )


def test_old_mapper_keys_are_rejected() -> None:
    vocab = MapperTupleVocab()
    raw = _minimal_batch(vocab, sparse=False)
    raw["target_tokens"] = torch.zeros((2, 3), dtype=torch.long)

    with pytest.raises(ValueError, match="old target_tokens/teacher_"):
        MapperBatch.from_mapping(raw, contract=MapperTokenContract(name="tuple", vocab=vocab))


def test_invalid_fragment_state_shape_fails_before_model_execution() -> None:
    vocab = MapperTupleVocab()
    raw = _minimal_batch(vocab, sparse=False)
    raw["target_fragment_states"] = dict(raw["target_fragment_states"])
    raw["target_fragment_states"]["open_mask"] = torch.zeros((2, 3, 3), dtype=torch.bool)

    with pytest.raises(ValueError, match=r"target_fragment_states\.open_mask"):
        MapperBatch.from_mapping(raw, contract=MapperTokenContract(name="tuple", vocab=vocab))


def test_invalid_carry_shape_fails_before_model_execution() -> None:
    vocab = MapperTupleVocab()
    raw = _minimal_batch(vocab, sparse=False)
    raw["ln_carry_in"] = dict(raw["ln_carry_in"])
    raw["ln_carry_in"]["open_mask"] = torch.zeros((2, 3), dtype=torch.bool)

    with pytest.raises(ValueError, match=r"ln_carry_in\.open_mask"):
        MapperBatch.from_mapping(raw, contract=MapperTokenContract(name="tuple", vocab=vocab))


def _minimal_batch(vocab: object, *, sparse: bool) -> dict[str, object]:
    decoder = torch.tensor(
        [
            [vocab.bos_id, vocab.time_shift_token_id(10), vocab.pad_id],
            [vocab.bos_id, vocab.time_shift_token_id(20), vocab.eos_id],
        ],
        dtype=torch.int32,
    )
    target = torch.tensor(
        [
            [vocab.time_shift_token_id(10), vocab.eos_id, vocab.pad_id],
            [vocab.time_shift_token_id(20), vocab.time_shift_token_id(10), vocab.eos_id],
        ],
        dtype=torch.int16,
    )
    state = _fragment_state_values(sparse=sparse)
    return {
        "decoder_input_tokens": decoder,
        "target_fragment_tokens": target,
        "target_fragment_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.int8),
        "target_fragment_states": state,
        "ln_carry_in": _carry_batch([0, 1000]),
        "ln_carry_out": _carry_batch([8000, 9000]),
        "write_start_ms": torch.tensor([0, 1000], dtype=torch.int32),
        "write_end_ms": torch.tensor([8000, 9000], dtype=torch.int32),
        "is_full_chart_start": torch.tensor([True, False], dtype=torch.bool),
        "is_full_chart_end": torch.tensor([False, True], dtype=torch.bool),
    }


def _fragment_state_values(*, sparse: bool) -> dict[str, torch.Tensor]:
    current_ms = torch.tensor([[0, 10, 20], [1000, 1010, 1020]], dtype=torch.int32)
    open_mask = torch.tensor(
        [
            [[False, True, False, False], [True, False, False, False], [False, False, True, False]],
            [[False, False, False, True], [True, True, False, False], [False, False, False, False]],
        ],
        dtype=torch.bool,
    )
    open_start_ms = torch.tensor(
        [
            [[-1, -10, -1, -1], [0, -1, -1, -1], [-1, -1, 10, -1]],
            [[-1, -1, -1, 990], [1000, 990, -1, -1], [-1, -1, -1, -1]],
        ],
        dtype=torch.int16,
    )
    open_age_ms = torch.tensor(
        [
            [[0, 10, 0, 0], [10, 0, 0, 0], [0, 0, 10, 0]],
            [[0, 0, 0, 10], [10, 20, 0, 0], [0, 0, 0, 0]],
        ],
        dtype=torch.int16,
    )
    state = {
        "current_ms": current_ms,
        "open_mask": open_mask,
        "open_start_ms": open_start_ms,
        "open_age_ms": open_age_ms,
    }
    if sparse:
        state["emitted_lane_mask"] = torch.tensor(
            [
                [[False, False, False, False], [True, False, False, False], [False, True, False, False]],
                [[False, False, True, False], [True, True, False, False], [False, False, False, True]],
            ],
            dtype=torch.int8,
        )
        state["last_lane_index"] = torch.tensor([[-1, 0, 1], [2, 1, 3]], dtype=torch.int16)
    return state


def _carry_batch(current_ms: list[int]) -> dict[str, torch.Tensor]:
    batch_size = len(current_ms)
    return {
        "current_ms": torch.tensor(current_ms, dtype=torch.int32),
        "open_mask": torch.zeros((batch_size, 4), dtype=torch.bool),
        "open_start_ms": torch.full((batch_size, 4), -1, dtype=torch.int16),
        "open_age_ms": torch.zeros((batch_size, 4), dtype=torch.int16),
    }
