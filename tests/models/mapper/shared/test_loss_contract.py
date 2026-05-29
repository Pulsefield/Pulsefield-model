from types import SimpleNamespace

import torch

from pulsefield_model.models.mapper.shared.loss import MapperTupleLossConfig, MapperTupleModelLoss
from pulsefield_model.models.mapper.shared.tokenizer import MAPPER_DENSITY_FRAMES as TUPLE_DENSITY_FRAMES
from pulsefield_model.models.mapper.shared.vocab import MapperTupleVocab
from pulsefield_model.models.mapper.v2_1.loss import MapperV21LossConfig, MapperV21ModelLoss
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
