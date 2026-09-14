from dataclasses import replace

import pytest
import torch

from pulsefield_model.research.scoped_style_modeling.dataset import ContractError
from pulsefield_model.research.source_action_modeling.model import ModelConfig, initialize_model
from pulsefield_model.research.source_action_modeling.tensors import ROW_CLASSES, collate, row_token
from .conftest import example

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_finite_normalized_joint_probabilities_loss_and_gradients(device):
    batch = collate([example(size=1), example(), example(long=True, size=16)]).to(device)
    model = initialize_model().to(device)
    output = model(batch)
    torch.testing.assert_close(output.log_probs.exp().sum(-1), torch.ones_like(output.row_nll))
    assert torch.isfinite(output.log_probs.exp()).all()
    expected = output.row_nll.sum(1) / batch.queries.steps.sum(1)
    torch.testing.assert_close(output.block_nll, expected)
    torch.testing.assert_close(output.loss, expected.mean())
    assert not torch.isclose(output.loss, output.row_nll.sum() / batch.queries.steps.sum())
    output.loss.backward()
    for component in (model.encoder, model.decoder):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in component.parameters())
    # A full joint table has nonzero within-row log odds interactions.
    scores = model.decoder.score(torch.randn(1, 2, 64, device=device), torch.zeros(1, 2, 32, device=device),
                                 torch.zeros(1, 2, 2, dtype=torch.long, device=device), torch.tensor([True], device=device))
    ids = [row_token(a) for a in ((1, 0, 1, 0), (2, 0, 1, 0), (1, 0, 2, 0), (2, 0, 2, 0))]
    assert (scores[0, ids[0]] + scores[0, ids[3]] - scores[0, ids[1]] - scores[0, ids[2]]).abs() > 1e-7


@pytest.mark.parametrize("device", DEVICES)
def test_mirror_encoder_queries_legal_masks_distribution_and_prefix_states(device):
    a, b = collate([example()]).to(device), collate([example(mirror=True)]).to(device)
    model = initialize_model().to(device).eval()
    with torch.no_grad():
        torch.testing.assert_close(model.encoder(a.observation).flip(2), model.encoder(b.observation), atol=3e-6, rtol=3e-5)
        x, y = model(a), model(b)
    permutation = torch.arange(ROW_CLASSES, device=device).reshape(36, 36).T.flatten()
    torch.testing.assert_close(x.log_probs, y.log_probs[:, :, permutation], atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(x.loss, y.loss)
    torch.testing.assert_close(x.final_states.flip(1), y.final_states, atol=3e-6, rtol=3e-5)
    assert torch.equal(x.final_occupancy.flip(1), y.final_occupancy)
    assert torch.equal(model.decoder.legal_rows(a.queries.entering_occupancy),
                       model.decoder.legal_rows(b.queries.entering_occupancy)[:, permutation])


@pytest.mark.parametrize("device", DEVICES)
def test_padding_preserves_valid_outputs_per_block_loss_and_decoder_state(device):
    short = collate([example()]).to(device)
    padded = collate([example(), example(long=True, size=16)]).to(device)
    model = initialize_model().to(device).eval()
    with torch.no_grad():
        hs, hp = model.encoder(short.observation), model.encoder(padded.observation)
        torch.testing.assert_close(hs[0], hp[0, :hs.shape[1]], atol=3e-6, rtol=3e-5)
        assert hp[0, hs.shape[1]:].count_nonzero() == 0
        a, b = model(short), model(padded)
    torch.testing.assert_close(a.log_probs[0], b.log_probs[0, :4], atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(a.block_nll[0], b.block_nll[0])
    torch.testing.assert_close(a.final_states[0], b.final_states[0], atol=3e-6, rtol=3e-5)
    assert b.row_nll[0, 4:].count_nonzero() == 0
    assert torch.equal(a.final_occupancy[0], b.final_occupancy[0])


def test_teacher_forcing_changes_only_future_decoder_outputs():
    a, b = collate([example()]), collate([example(changed=True)])
    model = initialize_model().eval()
    captured = []
    hook = model.encoder.register_forward_pre_hook(lambda module, args: captured.append(args))
    x, y = model(a), model(b)
    hook.remove()
    assert all(len(args) == 1 and args[0].__class__ is a.observation.__class__ for args in captured)
    assert torch.equal(model.encoder(a.observation), model.encoder(b.observation))
    assert torch.equal(x.log_probs[:, 0], y.log_probs[:, 0])
    assert not torch.equal(x.log_probs[:, 1:], y.log_probs[:, 1:])
    assert not torch.equal(x.final_states, y.final_states)


def test_prefix_legality_retains_close_plus_tap_and_close_plus_head():
    model = initialize_model()
    occupied = torch.tensor([[[1, 0], [0, 0]]])
    legal = model.decoder.legal_rows(occupied)[0]
    for action in (4, 5, 6):
        assert legal[row_token((action, 0, 0, 0))]
    for action in (1, 2):
        assert not legal[row_token((action, 0, 0, 0))]
    assert not legal[0]
    unknown = model.decoder.legal_rows(torch.full((1, 2, 2), -1))[0]
    assert unknown.sum() == ROW_CLASSES - 1
    batch = collate([example()])
    invalid = replace(batch, targets=torch.zeros_like(batch.targets))
    with pytest.raises(ContractError, match="legality"):
        model(invalid)


def test_invalid_config_fails_and_initialization_preserves_rng():
    state = torch.random.get_rng_state().clone()
    initialize_model()
    assert torch.equal(state, torch.random.get_rng_state())
    with pytest.raises(ContractError):
        ModelConfig(hand_hidden=3, attention_heads=4)
