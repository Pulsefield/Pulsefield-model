from dataclasses import replace
import math

import pytest
import torch

from ensomi_model.research.scoped_style_modeling.dataset import ContractError
from ensomi_model.research.source_action_modeling.model import ModelConfig, initialize_model
from ensomi_model.research.source_action_modeling.tensors import ROW_CLASSES, collate, row_token
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
    torch.testing.assert_close(output.sequence_nll, output.row_nll.sum(1))
    torch.testing.assert_close(output.mean_row_nll, expected)
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
    permutation = torch.arange(ROW_CLASSES, device=device).reshape(16, 16).T.flatten()
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
    torch.testing.assert_close(a.sequence_nll[0], b.sequence_nll[0])
    torch.testing.assert_close(a.mean_row_nll[0], b.mean_row_nll[0])
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


def test_prefix_legality_uses_v3_four_action_transitions():
    model = initialize_model()
    occupied = torch.tensor([[[1, 0], [0, 0]]])
    legal = model.decoder.legal_rows(occupied)[0]
    assert legal[row_token((3, 0, 0, 0))]
    for action in (1, 2):
        assert not legal[row_token((action, 0, 0, 0))]
    assert not legal[0]
    assert legal.sum() == 2 * 3 ** 3 - 1
    for action in (4, 5, 6):
        with pytest.raises(ContractError, match="Invalid joint"):
            row_token((action, 0, 0, 0))
    assert model.decoder.legal_rows(torch.zeros(1, 2, 2, dtype=torch.long)).sum() == 3 ** 4 - 1
    assert model.decoder.legal_rows(torch.ones(1, 2, 2, dtype=torch.long)).sum() == 2 ** 4 - 1
    unknown = model.decoder.legal_rows(torch.full((1, 2, 2), -1))[0]
    assert unknown.sum() == ROW_CLASSES - 1
    batch = collate([example()])
    invalid = replace(batch, targets=torch.zeros_like(batch.targets))
    with pytest.raises(ContractError, match="legality"):
        model(invalid)


def test_v3_row_tokens_round_trip_and_sampled_actions_replay_legally():
    from itertools import product
    from ensomi_model.research.source_action_modeling.actions import LANE_ACTIONS
    from ensomi_model.research.source_action_modeling.observation import advance_occupancy
    decoder = initialize_model().decoder
    assert LANE_ACTIONS == (0, 1, 2, 3) and ROW_CLASSES == 256
    for actions in product(LANE_ACTIONS, repeat=4):
        assert decoder.actions[row_token(actions)].flatten().tolist() == [actions[i] for i in (0, 1, 3, 2)]
    state = (False,) * 4
    occupancy = torch.zeros(1, 2, 2, dtype=torch.long)
    context, memory = torch.randn(1, 2, 64), torch.zeros(1, 2, 32)
    for _ in range(50):
        scores = decoder.score(context, memory, occupancy, torch.tensor([True]))
        chosen = torch.multinomial(scores.exp(), 1).squeeze(1)
        hand_actions = decoder.actions[chosen[0]].flatten().tolist()
        actions = tuple(hand_actions[i] for i in (0, 1, 3, 2))
        assert any(actions) and set(actions) <= set(LANE_ACTIONS)
        state = advance_occupancy(state, actions)
        memory, occupancy = decoder.advance(context, memory, occupancy, chosen, torch.tensor([True]))
        assert occupancy.flatten().tolist() == [int(state[i]) for i in (0, 1, 3, 2)]


def test_invalid_config_fails_and_initialization_preserves_rng():
    state = torch.random.get_rng_state().clone()
    initialize_model()
    assert torch.equal(state, torch.random.get_rng_state())
    with pytest.raises(ContractError):
        ModelConfig(hand_hidden=3, attention_heads=4)


@pytest.mark.parametrize("device", DEVICES)
def test_context_can_reverse_joint_hand_log_odds_with_all_four_choices_legal(device):
    decoder = initialize_model().decoder.to(device)
    # One active coordinate realizes the two four-combination distributions
    # in the objective contract. Shared Gram scoring cannot realize the second.
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.zero_()
        decoder.action.weight[4, 0] = 1  # tap on the outer lane
        decoder.action.weight[8, 0] = -1  # head on the outer lane
        decoder.pair_action.weight[0, 0] = 1
        decoder.context[0].weight[0, 0] = 1
        decoder.context[-1].weight[0, 0] = 1
        query_values = torch.nn.functional.gelu(torch.tensor([1., -1.], device=device))
        magnitude = math.sqrt(8) * math.log(3)
        slope = 2 * magnitude / (query_values[0] - query_values[1])
        decoder.interaction.weight[0, 0] = slope
        decoder.interaction.bias[0] = magnitude - slope * query_values[0]
    context = torch.zeros(2, 2, 64, device=device)
    context[:, :, 0] = torch.tensor([1., -1.], device=device)[:, None]
    scores = decoder.score(context, torch.zeros(2, 2, 32, device=device),
                           torch.zeros(2, 2, 2, dtype=torch.long, device=device),
                           torch.ones(2, dtype=torch.bool, device=device)).reshape(2, 16, 16)
    odds = scores[:, 4, 4] + scores[:, 8, 8] - scores[:, 4, 8] - scores[:, 8, 4]
    table = scores[:, [4, 4, 8, 8], [4, 8, 4, 8]].softmax(-1)
    torch.testing.assert_close(table, torch.tensor([[.45, .05, .05, .45], [.05, .45, .45, .05]], device=device),
                               atol=2e-7, rtol=2e-6)
    expected = torch.tensor([math.log(81), -math.log(81)], device=device)
    torch.testing.assert_close(odds, expected, atol=3e-6, rtol=3e-5)
    assert odds[0] > 0 and odds[1] < 0
    odds.sum().backward()
    assert decoder.interaction.weight.grad.abs().sum() > 0


def test_one_shared_fair_bit_has_constant_sequence_cost_and_diluted_row_cost(monkeypatch):
    model = initialize_model()
    batch = collate([example(size=1), example(long=True, size=16)])
    chosen, alternative = row_token((1, 0, 0, 0)), row_token((0, 1, 0, 0))
    batch = replace(batch, targets=torch.where(batch.queries.steps, chosen, 0))
    step = 0
    def shared_bit(context, states, occupancy, active):
        nonlocal step
        values = context.new_full((context.shape[0], ROW_CLASSES), -torch.inf)
        values[:, chosen] = -math.log(2) if step == 0 else 0
        if step == 0:
            values[:, alternative] = -math.log(2)
        values[~active] = -torch.inf
        values[~active, 0] = 0
        step += 1
        return values
    monkeypatch.setattr(model.decoder, "score", shared_bit)
    output = model(batch)
    torch.testing.assert_close(output.sequence_nll, torch.full((2,), math.log(2)))
    torch.testing.assert_close(output.mean_row_nll, torch.tensor([math.log(2), math.log(2) / 16]))
    torch.testing.assert_close(output.loss, output.mean_row_nll.mean())
