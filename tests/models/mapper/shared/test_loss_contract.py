from types import SimpleNamespace

import torch

from pulsefield_model.models.mapper.shared import loss as shared_loss
from pulsefield_model.models.mapper.shared.batch import MapperTokenContract
from pulsefield_model.models.mapper.shared.loss import (
    MapperLossTokenSpec,
    MapperTupleLossConfig,
    MapperTupleModelLoss,
    adapter_bias_regularization,
)
from pulsefield_model.models.mapper.shared.tokenizer import MAPPER_DENSITY_FRAMES as TUPLE_DENSITY_FRAMES
from pulsefield_model.models.mapper.shared.vocab import MapperTupleVocab
from pulsefield_model.models.mapper.v2_1.loss import MapperV21LossConfig, MapperV21LossOutput, MapperV21ModelLoss
from pulsefield_model.models.mapper.v2_1.tokenizer import MAPPER_DENSITY_FRAMES as V21_DENSITY_FRAMES
from pulsefield_model.models.mapper.v2_1.vocab import LaneAction, MapperV21Vocab


def test_tuple_loss_accepts_token_only_batch_without_fragment_mask() -> None:
    vocab = MapperTupleVocab()
    target = torch.tensor([[vocab.time_shift_token_id(10), vocab.eos_id, vocab.pad_id]], dtype=torch.long)
    output = SimpleNamespace(logits_final=torch.zeros((1, 3, vocab.size), dtype=torch.float32))
    loss_fn = MapperTupleModelLoss(
        MapperTupleLossConfig(lambda_ln_close=0.0, lambda_adapter_reg=0.0, lambda_density=0.0),
        vocab=vocab,
    )

    loss = loss_fn(output, {"target_fragment_tokens": target})

    assert torch.isfinite(loss.total_loss)
    assert torch.equal(loss.total_loss, loss.token_loss)
    assert loss.metrics["token/valid_count"] == 2
    assert float(loss.ln_close_loss.item()) == 0.0
    assert float(loss.adapter_reg_loss.item()) == 0.0
    assert float(loss.density_loss.item()) == 0.0


def test_v21_loss_accepts_token_only_batch_without_sparse_state() -> None:
    vocab = MapperV21Vocab()
    target = torch.tensor([[vocab.time_shift_token_id(10), vocab.eos_id, vocab.pad_id]], dtype=torch.long)
    output = SimpleNamespace(logits_final=torch.zeros((1, 3, vocab.size), dtype=torch.float32))
    loss_fn = MapperV21ModelLoss(
        MapperV21LossConfig(lambda_ln_close=0.0, lambda_adapter_reg=0.0, lambda_density=0.0),
        vocab=vocab,
    )

    loss = loss_fn(output, {"target_fragment_tokens": target})

    assert torch.isfinite(loss.total_loss)
    assert torch.equal(loss.total_loss, loss.token_loss)
    assert loss.metrics["token/valid_count"] == 2
    assert float(loss.ln_close_loss.item()) == 0.0
    assert float(loss.adapter_reg_loss.item()) == 0.0
    assert float(loss.density_loss.item()) == 0.0


def test_v21_loss_wrapper_uses_shared_loss_implementation(monkeypatch) -> None:
    vocab = MapperV21Vocab()
    target = torch.tensor([[vocab.time_shift_token_id(10), vocab.eos_id]], dtype=torch.long)
    output = SimpleNamespace(logits_final=torch.zeros((1, 2, vocab.size), dtype=torch.float32))
    calls = {"token_ce": 0}
    original = shared_loss.token_cross_entropy

    def spy_token_cross_entropy(*args, **kwargs):
        calls["token_ce"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(shared_loss, "token_cross_entropy", spy_token_cross_entropy)
    loss_fn = MapperV21ModelLoss(
        MapperV21LossConfig(lambda_ln_close=0.0, lambda_adapter_reg=0.0, lambda_density=0.0),
        vocab=vocab,
    )

    loss = loss_fn(output, {"target_fragment_tokens": target})

    assert isinstance(loss_fn, MapperTupleModelLoss)
    assert isinstance(loss, MapperV21LossOutput)
    assert calls["token_ce"] == 1


def test_v21_loss_module_exports_only_v21_loss_types() -> None:
    from pulsefield_model.models.mapper.v2_1 import loss as v21_loss

    assert set(v21_loss.__all__) == {"MapperV21LossConfig", "MapperV21LossOutput", "MapperV21ModelLoss"}
    assert not hasattr(v21_loss, "adapter_bias_regularization")
    assert not hasattr(v21_loss, "adapter_reg")
    assert not hasattr(v21_loss, "adapter_regularization")
    assert not hasattr(v21_loss, "density_auxiliary_loss")
    assert not hasattr(v21_loss, "token_ce")
    assert not hasattr(v21_loss, "token_cross_entropy")


def test_loss_token_spec_from_contract_uses_tuple_contract_not_event_token_ids_attribute() -> None:
    base_vocab = MapperTupleVocab()
    vocab = _TupleVocabWithMisleadingEventIds(base_vocab)

    spec = MapperLossTokenSpec.from_contract(MapperTokenContract(name="tuple", vocab=vocab))

    assert spec.onset_token_ids == base_vocab.event_token_ids
    assert spec.onset_weights == tuple(
        float(base_vocab.event_onset_weight(token_id)) for token_id in base_vocab.event_token_ids
    )


def test_loss_token_spec_from_contract_uses_sparse_contract_not_lane_action_ids_attribute() -> None:
    base_vocab = MapperV21Vocab()
    vocab = _V21VocabWithMisleadingLaneActionIds(base_vocab)

    spec = MapperLossTokenSpec.from_contract(
        MapperTokenContract(name="v2.1", vocab=vocab, requires_sparse_lane_state=True)
    )

    assert spec.onset_token_ids == base_vocab.lane_action_token_ids
    assert spec.onset_weights == tuple(
        float(base_vocab.event_onset_weight(token_id)) for token_id in base_vocab.lane_action_token_ids
    )


def test_adapter_bias_regularization_uses_all_trailing_dimensions_in_masked_denominator() -> None:
    bias = torch.ones((1, 2, 3, 4), dtype=torch.float32)
    mask = torch.tensor([[True, False]])

    loss = adapter_bias_regularization(bias, mask=mask)

    assert torch.allclose(loss, torch.tensor(1.0))


def test_tuple_density_loss_requires_only_density_state_and_window_fields() -> None:
    vocab = MapperTupleVocab()
    target = torch.tensor([[vocab.encode_event([LaneAction.TAP, LaneAction.NONE, LaneAction.NONE, LaneAction.NONE])]])
    output = SimpleNamespace(logits_final=torch.zeros((1, 1, vocab.size), dtype=torch.float32))
    loss_fn = MapperTupleModelLoss(
        MapperTupleLossConfig(lambda_ln_close=0.0, lambda_adapter_reg=0.0, lambda_density=0.1),
        vocab=vocab,
    )

    loss = loss_fn(
        output,
        {
            "target_fragment_tokens": target,
            "target_fragment_states": {"current_ms": torch.tensor([[0]], dtype=torch.long)},
            "write_start_ms": torch.tensor([0], dtype=torch.long),
            "density_target_8s": torch.zeros((1, TUPLE_DENSITY_FRAMES, 1), dtype=torch.float32),
            "density_confidence_8s": torch.ones((1, TUPLE_DENSITY_FRAMES, 1), dtype=torch.float32),
        },
    )

    assert torch.isfinite(loss.total_loss)
    assert torch.isfinite(loss.density_loss)
    assert loss.metric_denominators["loss/density"] == float(TUPLE_DENSITY_FRAMES)


def test_v21_density_loss_does_not_require_full_sparse_mapper_batch() -> None:
    vocab = MapperV21Vocab()
    target = torch.tensor([[vocab.lane_action_token_id(0, LaneAction.TAP)]], dtype=torch.long)
    output = SimpleNamespace(logits_final=torch.zeros((1, 1, vocab.size), dtype=torch.float32))
    loss_fn = MapperV21ModelLoss(
        MapperV21LossConfig(lambda_ln_close=0.0, lambda_adapter_reg=0.0, lambda_density=0.1),
        vocab=vocab,
    )

    loss = loss_fn(
        output,
        {
            "target_fragment_tokens": target,
            "target_fragment_states": {"current_ms": torch.tensor([[0]], dtype=torch.long)},
            "write_start_ms": torch.tensor([0], dtype=torch.long),
            "density_target_8s": torch.zeros((1, V21_DENSITY_FRAMES, 1), dtype=torch.float32),
            "density_confidence_8s": torch.ones((1, V21_DENSITY_FRAMES, 1), dtype=torch.float32),
        },
    )

    assert torch.isfinite(loss.total_loss)
    assert torch.isfinite(loss.density_loss)
    assert loss.metric_denominators["loss/density"] == float(V21_DENSITY_FRAMES)


class _TupleVocabWithMisleadingEventIds:
    def __init__(self, base_vocab: MapperTupleVocab) -> None:
        self._base_vocab = base_vocab
        self.event_token_ids = (base_vocab.eos_id,)

    @property
    def size(self) -> int:
        return self._base_vocab.size

    @property
    def pad_id(self) -> int:
        return self._base_vocab.pad_id

    @property
    def bos_id(self) -> int:
        return self._base_vocab.bos_id

    @property
    def eos_id(self) -> int:
        return self._base_vocab.eos_id

    def decode_event(self, token_id: int) -> tuple[LaneAction, ...]:
        return self._base_vocab.decode_event(token_id)

    def event_onset_weight(self, token_id: int) -> int:
        return self._base_vocab.event_onset_weight(token_id)


class _V21VocabWithMisleadingLaneActionIds:
    def __init__(self, base_vocab: MapperV21Vocab) -> None:
        self._base_vocab = base_vocab
        self.lane_action_token_ids = (base_vocab.eos_id,)

    @property
    def size(self) -> int:
        return self._base_vocab.size

    @property
    def pad_id(self) -> int:
        return self._base_vocab.pad_id

    @property
    def bos_id(self) -> int:
        return self._base_vocab.bos_id

    @property
    def eos_id(self) -> int:
        return self._base_vocab.eos_id

    def decode_lane_action(self, token_id: int) -> tuple[int, LaneAction]:
        return self._base_vocab.decode_lane_action(token_id)

    def event_onset_weight(self, token_id: int) -> int:
        return self._base_vocab.event_onset_weight(token_id)
