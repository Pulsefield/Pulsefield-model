from copy import deepcopy
from itertools import product

import numpy as np
import pytest
import torch

from ensomi_model.research.bounded_typed_continuation import model as model_module
from ensomi_model.research.bounded_typed_continuation import generation as generation_module
from ensomi_model.research.bounded_typed_continuation.contract import Arm, ROW_ACTIONS, Schedule, Timing
from ensomi_model.research.bounded_typed_continuation.data import SourceInterval, batch_likelihood, prepare_batch
from ensomi_model.research.bounded_typed_continuation.generation import Rollout
from ensomi_model.research.bounded_typed_continuation.model import BoundedModel, ModelConfig
from ensomi_model.research.bounded_typed_continuation.support import row_supports
from ensomi_model.research.oracle_time_continuation.replay import ExactReplayState, commit
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.scoped_style_modeling.dataset import ContractError
from .test_data import mixed_chart
from .test_generation import setup


def scalar_supports(states):
    return np.array([state.row_support() for state in states], dtype=np.bool_)


def test_exhaustive_small_occupancy_known_end_roles_and_true_terminal_match_scalar_oracle():
    times = (0., 100., 200., 300.)
    replays = {occupied: commit(ExactReplayState(), CompleteRow(0., tuple(2 if opened else 1 for opened in occupied)))
               for occupied in product((False, True), repeat=4)}
    checked = 0
    for roles in product((False, True), repeat=3):
        timing = Timing(times, (True,) + roles)
        for index in (1, 2, 3):
            states = []
            # -2 is closed, -1 is an unknown rowwise end; all remaining values
            # are distinct known obligations, including due/current and late.
            for lanes in product((-2, -1, *range(index, 4)), repeat=4):
                occupied = tuple(lane != -2 for lane in lanes)
                ends = tuple(lane if lane >= 0 else None for lane in lanes)
                states.append(Schedule(Arm.R1, timing, index, replays[occupied], ends))
            np.testing.assert_array_equal(row_supports(states), scalar_supports(states))
            checked += len(states)
    untyped = [Schedule(Arm.R0, Timing(times), index, replay) for index in (1,2,3) for replay in replays.values()]
    np.testing.assert_array_equal(row_supports(untyped), scalar_supports(untyped))
    assert checked == 7696


def test_generated_prefixes_skips_finished_schedules_and_empty_batch():
    random = np.random.default_rng(1321)
    states = []
    for arm in (Arm.R0, Arm.R1):
        for _ in range(8):
            timing = Timing(tuple(map(float, np.cumsum(random.integers(1, 100000, size=37)))),
                            None if arm == Arm.R0 else tuple(bool(x) for x in random.integers(0,2,size=37)))
            state = Schedule(arm, timing)
            while not state.finished:
                states.append(state)
                legal = np.flatnonzero(state.row_support())
                state, _ = state.advance(ROW_ACTIONS[int(random.choice(legal))])
            states.append(state)
    actual = row_supports(states)
    np.testing.assert_array_equal(actual, scalar_supports(states))
    assert not np.any(actual[[s.finished for s in states]])
    assert row_supports([]).shape == (0, 256) and row_supports([]).dtype == np.bool_
    with pytest.raises(ContractError, match='R0/R1-only'):
        row_supports([Schedule(Arm.O1, Timing((0.,), (True,)))])


@pytest.mark.parametrize('arm,consequence', [(Arm.R0,'none'),(Arm.R1,'none'),(Arm.R1,'actions'),(Arm.R1,'frontier')])
@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_model_likelihood_and_all_gradients_preserve_scalar_support(arm, consequence, device, monkeypatch):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    torch.manual_seed(41)
    model = BoundedModel(ModelConfig(arm, hidden=8, levels=2, coupling_rank=2, row_consequence=consequence)).to(device)
    if model.row_consequence is not None:
        torch.nn.init.normal_(model.row_consequence.output.weight, std=.05)
    reference = deepcopy(model)
    source = mixed_chart()
    batch = prepare_batch([SourceInterval(source,0,2),SourceInterval(source,2,2)], arm, model.temporal.config.receptive_tokens)
    actual, factors = batch_likelihood(model, batch)
    actual.backward()
    monkeypatch.setattr(model_module, 'row_supports', scalar_supports)
    expected, other = batch_likelihood(reference, batch)
    expected.backward()
    tolerance = 0 if device == 'cpu' else 2e-6
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(factors, other, atol=tolerance, rtol=tolerance)
    for (name,a),(label,b) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == label
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('consequence', ['none','actions','frontier'])
@pytest.mark.parametrize('device', ['cpu','mps'])
def test_native_draws_probabilities_rng_and_raw_recovery_preserve_scalar_path(consequence, device, monkeypatch):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    model, timing, _, fast = setup(Arm.R1, device, consequence=consequence)
    rng = torch.Generator().manual_seed(791)
    initial = fast.snapshot(rng)
    draws = []
    for _ in range(10):
        draws.append(fast.step(rng))
    fast, rng = Rollout.restore(model, timing, fast.snapshot(rng))
    while not fast.state.finished:
        draws.append(fast.step(rng))
    monkeypatch.setattr(model_module, 'row_supports', scalar_supports)
    monkeypatch.setattr(generation_module, 'row_supports', scalar_supports)
    slow, other_rng = Rollout.restore(model, timing, initial)
    for expected in draws:
        actual = slow.step(other_rng)
        assert actual == expected
    assert slow.state == fast.state and torch.equal(other_rng.get_state(), rng.get_state())
