from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from pulsefield_model.research.bounded_typed_continuation.consequence import (
    LANE_DIM, TIMING_DIM, RowConsequence, consequence_features,
)
from pulsefield_model.research.bounded_typed_continuation.contract import Arm, ROW_ACTIONS, Schedule, Timing
from pulsefield_model.research.bounded_typed_continuation.data import SourceInterval, batch_likelihood, prepare_batch
from pulsefield_model.research.bounded_typed_continuation.features import RELATIVE_LANES, TIME_DIM, time_features
from pulsefield_model.research.bounded_typed_continuation.model import BoundedModel, ModelConfig
from pulsefield_model.research.oracle_time_continuation.schema import CompleteRow
from pulsefield_model.research.scoped_style_modeling.dataset import ContractError
from .test_data import chart, mixed_chart


def test_consequence_clocks_match_independent_replay_for_every_legal_row():
    source = mixed_chart()
    states = [source.state(Arm.R1, i) for i in range(source.seed_rows, len(source.rows))]
    local, timing = consequence_features(states, 'frontier')
    assert local.shape == (len(states), 4, 4, LANE_DIM) and timing.shape == (len(states), TIMING_DIM)
    for index, state in enumerate(states):
        future = [t for t, role in zip(state.timing.times_ms[state.index + 1:], state.timing.onsets[state.index + 1:]) if role]
        next_h = future[0] if future else None
        next_r = state.timing.times_ms[state.index + 1] if state.index + 1 < len(source.rows) else None
        for actions in ROW_ACTIONS:
            if not state.row_possible(actions):
                continue
            after, _ = state.advance(actions)
            current = state.replay.clocks_at(state.time_ms)
            clocks = after.replay.clocks_at(next_h) if next_h is not None else None
            for lane, action in enumerate(actions):
                end = after.known_ends[lane]
                earliest = state.timing.times_ms[end] if end is not None else next_r
                values = [
                    current.lane_attack_ms[lane] if action in (1, 2) else None,
                    current.lane_release_ms[lane] if action in (1, 2) else None,
                    current.ln_age_ms[lane] if action == 3 else None,
                    None if clocks is None else clocks.lane_attack_ms[lane],
                    None if clocks is None else clocks.lane_release_ms[lane],
                    None if clocks is None else clocks.ln_age_ms[lane],
                    next_h - earliest if after.replay.occupancy[lane] and next_h is not None and earliest is not None else None,
                ]
                np.testing.assert_array_equal(local[index, lane, action, :4], np.eye(4)[action])
                assert local[index, lane, action, 4] == after.replay.occupancy[lane]
                np.testing.assert_array_equal(local[index, lane, action, 5:], time_features(values).reshape(-1))
        expected_timing = np.r_[time_features(None if next_r is None else next_r - state.time_ms),
                                time_features(None if next_h is None else next_h - state.time_ms),
                                float(next_r is not None and state.timing.onsets[state.index + 1])]
        np.testing.assert_array_equal(timing[index], expected_timing)


def test_intervening_release_slack_true_end_and_large_time_translation():
    values = (0., 79., 82., 100.)
    role = (True, False, True, False)
    state = Schedule(Arm.R1, Timing(values, role))
    shifted = Schedule(Arm.R1, Timing(tuple(t + 2 ** 40 for t in values), role))
    a, b = consequence_features([state], 'frontier'), consequence_features([shifted], 'frontier')
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    # A new LN has only 3ms after its earliest possible release before the next H.
    clocks = a[0][0, 0, 2, 5:].reshape(7, TIME_DIM)
    np.testing.assert_array_equal(clocks[-1], time_features(3.))
    np.testing.assert_array_equal(clocks[3], time_features(82.))
    next_state, _ = state.advance((2, 2, 2, 2))
    next_state, _ = next_state.advance((3, 3, 3, 3))
    head, _ = consequence_features([next_state], 'frontier')
    np.testing.assert_array_equal(head[0, 0, 1, 5 + TIME_DIM:5 + 2 * TIME_DIM], time_features(3.))
    terminal, _ = next_state.advance((1, 0, 0, 0))
    at_end, end_timing = consequence_features([terminal], 'frontier')
    assert not np.any(end_timing)
    assert not np.any(at_end[..., 5 + 3 * TIME_DIM:])
    done, _ = terminal.advance((0, 0, 0, 0))
    with pytest.raises(ContractError, match='active R1'):
        consequence_features([done], 'frontier')


def test_only_seed_obligations_enter_earliest_release_and_source_future_is_hidden():
    timing = Timing((0., 100., 200., 300., 400.), (True, False, True, False, True))
    seed = [CompleteRow(0., (2, 0, 0, 0))]
    states = [Schedule.from_seed(Arm.R1, timing, seed, {0: end}) for end in (1, 3)]
    local, _ = consequence_features(states, 'frontier')
    np.testing.assert_array_equal(local[0, 0, 3, 5 + 6 * TIME_DIM:], time_features(None))
    np.testing.assert_array_equal(local[1, 0, 0, 5 + 6 * TIME_DIM:], time_features(-100.))
    a = chart([(1, 0, 0, 0), (2, 2, 0, 0), (0, 0, 1, 0), (3, 0, 0, 0), (0, 3, 0, 0), (1, 0, 0, 0)])
    b = chart([(1, 0, 0, 0), (2, 2, 0, 0), (0, 0, 1, 0), (0, 3, 0, 0), (3, 0, 0, 0), (1, 0, 0, 0)])
    for i in (1, 2, 3):
        for first, second in zip(consequence_features([a.state(Arm.R1, i)], 'frontier'),
                                 consequence_features([b.state(Arm.R1, i)], 'frontier')):
            np.testing.assert_array_equal(first, second)
    control, control_timing = consequence_features(states, 'actions')
    assert not np.any(control[..., 4:]) and not np.any(control_timing)
    np.testing.assert_array_equal(control[0], control[1])


def dense_energy(module, hands, local, timing):
    count = len(hands)
    shared = [hands[:, :, None].expand(-1, -1, 256, -1), timing[:, None, None].expand(-1, 2, 256, -1)]
    for position in range(4):
        shared.append(torch.stack([torch.stack([local[:, lanes[position], row[lanes[position]]] for row in ROW_ACTIONS], 1)
                                   for lanes in RELATIVE_LANES], 1))
    features = torch.cat(shared, -1)
    weight = torch.cat([module.context.weight, module.timing.weight] + [p.weight for p in module.lanes], -1)
    hidden = F.linear(features.reshape(count * 2 * 256, -1), weight, module.timing.bias)
    return module.output(F.gelu(hidden)).reshape(count, 2, 256).mean(1)


@pytest.mark.parametrize('device', ['cpu', 'mps'])
@pytest.mark.parametrize('mode', ['actions', 'frontier', 'frontier2'])
def test_factorized_energy_matches_dense_values_input_and_all_parameter_gradients(device, mode):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    dtype = torch.float64 if device == 'cpu' else torch.float32
    torch.manual_seed(271)
    module = RowConsequence(8, mode).to(device=device, dtype=dtype)
    torch.nn.init.normal_(module.output.weight, std=.05)
    dense = deepcopy(module)
    source = mixed_chart()
    states = [source.state(Arm.R1, i) for i in (1, 3, 5, 8)]
    raw, future = consequence_features(states, mode)
    local = torch.tensor(raw, device=device, dtype=dtype)
    timing = torch.tensor(future, device=device, dtype=dtype)
    hands = torch.randn(4, 2, 8, device=device, dtype=dtype, requires_grad=True)
    copied = hands.detach().clone().requires_grad_()
    actual, expected = module.score(hands, local, timing), dense_energy(dense, copied, local, timing)
    tolerance = 2e-10 if device == 'cpu' else 4e-5
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    weighting = torch.randn_like(actual)
    (actual * weighting).sum().backward()
    (expected * weighting).sum().backward()
    torch.testing.assert_close(hands.grad, copied.grad, atol=tolerance, rtol=tolerance)
    for (name, a), (other, b) in zip(module.named_parameters(), dense.named_parameters()):
        assert name == other and a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad, b.grad, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('mode', ['actions', 'frontier', 'frontier2'])
def test_nonzero_residual_keeps_mirror_equivariance_and_exact_support(mode):
    torch.manual_seed(17)
    model = BoundedModel(ModelConfig(Arm.R1, hidden=12, levels=2, coupling_rank=3, row_consequence=mode)).double()
    torch.nn.init.normal_(model.row_consequence.output.weight, std=.1)
    timing = Timing((0., 20., 23., 100.), (True, False, True, False))
    state, _ = Schedule(Arm.R1, timing).advance((2, 2, 1, 0))
    mirror, _ = Schedule(Arm.R1, timing).advance((0, 1, 2, 2))
    hands = torch.randn(1, 2, 12, dtype=torch.float64)
    outputs = model.decision_log_probs(torch.cat((hands, hands.flip(1))), [state, mirror])
    reverse = [ROW_ACTIONS.index(row[::-1]) for row in ROW_ACTIONS]
    torch.testing.assert_close(outputs[0], outputs[1, reverse], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(outputs.exp().sum(-1), torch.ones(2, dtype=torch.float64))
    assert torch.isfinite(outputs[0]).tolist() == list(state.row_support())
    assert torch.isfinite(outputs[1]).tolist() == list(mirror.row_support())


def test_residual_size_and_r1_only_configuration():
    assert sum(p.numel() for p in RowConsequence(128, 'frontier').parameters()) == 26912
    for arm in (Arm.R0, Arm.O1):
        with pytest.raises(ContractError, match='R1-only'):
            ModelConfig(arm, row_consequence='frontier')
    with pytest.raises(ContractError, match='Row consequence'):
        ModelConfig(Arm.R1, row_consequence='unknown')


def test_zero_output_first_learns_then_delivers_gradient_to_consequence_encoder():
    torch.manual_seed(21)
    model = BoundedModel(ModelConfig(Arm.R1, hidden=8, levels=2, coupling_rank=2, row_consequence='frontier'))
    source = mixed_chart()
    batch = prepare_batch([SourceInterval(source, 0, len(source.onsets))], Arm.R1, model.temporal.config.receptive_tokens)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    batch_likelihood(model, batch)[0].backward()
    assert model.row_consequence.output.weight.grad.norm() > 1e-8
    assert model.row_consequence.context.weight.grad.norm() == 0
    assert all(layer.weight.grad.norm() == 0 for layer in model.row_consequence.lanes)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    batch_likelihood(model, batch)[0].backward()
    assert model.row_consequence.context.weight.grad.norm() > 1e-8
    assert all(layer.weight.grad.norm() > 1e-8 for layer in model.row_consequence.lanes)
