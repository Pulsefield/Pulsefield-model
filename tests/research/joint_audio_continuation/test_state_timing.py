from itertools import product

import numpy as np
import pytest
import torch

from ensomi_model.research.bounded_typed_continuation.contract import Arm, ROW_ACTIONS, Schedule, Timing
from ensomi_model.research.bounded_typed_continuation.features import TimingView, query_features
from ensomi_model.research.joint_audio_continuation.state import BASE_QUERY_DIM, exact_features, legal_rows
from ensomi_model.research.joint_audio_continuation.timing import hazard_nll, sample_hazards
from ensomi_model.research.oracle_time_continuation.features import TIME_DIM
from ensomi_model.research.oracle_time_continuation.replay import ExactReplayState, commit
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.scoped_style_modeling.dataset import ContractError


@pytest.mark.parametrize('held', list(product((False, True), repeat=4)))
@pytest.mark.parametrize('terminal', (False, True))
def test_support_agrees_with_atomic_commit(held, terminal):
    replay = ExactReplayState()
    if any(held):
        replay = commit(replay, CompleteRow(0, tuple(2 if value else 0 for value in held)))
    support = legal_rows([replay], [terminal])
    assert support.dtype == np.bool_ and support.shape == (1, 256)
    assert not support[0, 0]
    for index, actions in enumerate(ROW_ACTIONS[1:], 1):
        row = CompleteRow(1, actions)
        if support[0, index]:
            after = commit(replay, row, is_terminal=terminal)
            assert not terminal or not any(after.occupancy)
        else:
            with pytest.raises(ContractError):
                commit(replay, row, is_terminal=terminal)


def test_simultaneous_heads_releases_and_four_held_lanes_have_support():
    partial = commit(ExactReplayState(), CompleteRow(0, (2, 0, 2, 0)))
    support = legal_rows([partial], [False])[0]
    assert support[ROW_ACTIONS.index((3, 2, 0, 1))]
    assert support[ROW_ACTIONS.index((3, 0, 0, 0))]
    full = commit(ExactReplayState(), CompleteRow(0, (2, 2, 2, 2)))
    ordinary, terminal = legal_rows([full, full], [False, True])
    assert ordinary.sum() == 15
    assert terminal.sum() == 1 and terminal[ROW_ACTIONS.index((3, 3, 3, 3))]
    assert ordinary[ROW_ACTIONS.index((3, 0, 0, 0))]
    assert not terminal[ROW_ACTIONS.index((3, 0, 0, 0))]


def test_exact_features_equal_r1_prefix_without_future_endpoints():
    timing = Timing((0., 10., 30., 1_000_000., 2_000_000.), (True, True, True, True, False))
    schedule = Schedule(Arm.R1, timing)
    view = TimingView(timing)
    states = [schedule]
    for actions in ((2, 1, 0, 0), (0, 0, 2, 1), (3, 2, 0, 1)):
        schedule, _ = schedule.advance(actions)
        states.append(schedule)
    actual = exact_features([state.replay for state in states], [state.time_ms for state in states])
    np.testing.assert_array_equal(actual, query_features(states, view)[..., :BASE_QUERY_DIM])
    assert actual.dtype == np.float32
    assert not actual[..., 18 * TIME_DIM:22 * TIME_DIM].any()


def test_exact_feature_reads_preserve_long_holds_and_mirror_hands():
    rows = (CompleteRow(0., (2, 1, 0, 0)), CompleteRow(100., (0, 0, 1, 2)))
    replay = mirrored = ExactReplayState()
    for row in rows:
        replay = commit(replay, row)
        mirrored = commit(mirrored, CompleteRow(row.time_ms, row.actions[::-1]))
    features = exact_features([replay, replay], [100., 1_000_000.])
    assert replay.occupancy == (True, False, False, True)
    np.testing.assert_array_equal(features[:, ::-1], exact_features([mirrored, mirrored], [100., 1_000_000.]))
    assert not np.array_equal(features[0], features[1])
    with pytest.raises(ContractError, match='precedes'):
        exact_features([replay], [99.])
    with pytest.raises(ContractError, match='one query'):
        exact_features([replay], [])
    assert exact_features([], []).shape == (0, 2, BASE_QUERY_DIM)
    assert legal_rows([], []).shape == (0, 256)


def test_analytic_event_mass_and_survival_sum_to_one():
    probabilities = torch.tensor([.2, .3, .4], dtype=torch.float64)
    logits = torch.logit(probabilities).repeat(4, 1).requires_grad_()
    loss = hazard_nll(logits, torch.ones_like(logits, dtype=torch.bool),
                      torch.tensor([0, 1, 2, -1]), torch.zeros_like(logits, dtype=torch.bool))
    expected = torch.tensor([.2, .8 * .3, .8 * .7 * .4, .8 * .7 * .6], dtype=torch.float64)
    torch.testing.assert_close(loss.exp().reciprocal(), expected)
    torch.testing.assert_close(loss.exp().reciprocal().sum(), torch.tensor(1., dtype=torch.float64))
    loss.sum().backward()
    expected_grad = torch.tensor([[-.8, 0., 0.], [.2, -.7, 0.], [.2, .3, -.6], [.2, .3, .4]],
                                 dtype=torch.float64)
    torch.testing.assert_close(logits.grad, expected_grad)


def test_censored_chunk_loss_equals_survival_plus_later_event():
    logits = torch.tensor([[-2., -.5, .2, -1.]], dtype=torch.float64, requires_grad=True)
    valid = torch.ones_like(logits, dtype=torch.bool)
    forced = torch.zeros_like(valid)
    full = hazard_nll(logits, valid, torch.tensor([3]), forced)
    left = hazard_nll(logits[:, :2], valid[:, :2], torch.tensor([-1]), forced[:, :2])
    right = hazard_nll(logits[:, 2:], valid[:, 2:], torch.tensor([1]), forced[:, 2:])
    torch.testing.assert_close(full, left + right)
    assert torch.autograd.gradcheck(lambda x: hazard_nll(x, valid, torch.tensor([3]), forced), (logits,))


def test_forced_terminal_is_exact_and_ignored_values_have_zero_gradient():
    logits = torch.tensor([[0., float('nan'), float('nan'), float('nan')],
                           [0., float('nan'), float('nan'), float('nan')]], requires_grad=True)
    valid = torch.tensor([[True, False, True, False]]).repeat(2, 1)
    forced = torch.tensor([[False, True, True, False]]).repeat(2, 1)
    loss = hazard_nll(logits, valid, torch.tensor([2, -1]), forced)
    torch.testing.assert_close(loss[0], torch.log(torch.tensor(2.)))
    assert torch.isposinf(loss[1])
    loss[0].backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[.5, 0., 0., 0.], [0., 0., 0., 0.]]))
    later = hazard_nll(torch.zeros(1, 3), torch.ones(1, 3, dtype=torch.bool), torch.tensor([2]),
                       torch.tensor([[False, True, False]]))
    assert torch.isposinf(later).all()


def test_empty_censored_interval_and_invalid_event_target():
    logits = torch.zeros(2, 0, requires_grad=True)
    loss = hazard_nll(logits, torch.zeros(2, 0, dtype=torch.bool), torch.tensor([-1, -1]),
                      torch.zeros(2, 0, dtype=torch.bool))
    loss.sum().backward()
    assert torch.equal(loss, torch.zeros(2))
    with pytest.raises(ContractError, match='valid clock'):
        hazard_nll(torch.zeros(1, 2), torch.tensor([[True, False]]), torch.tensor([1]),
                   torch.zeros(1, 2, dtype=torch.bool))


@pytest.mark.parametrize('seed', range(16))
def test_sampling_draw_is_preserved_across_chunks(seed):
    logits = torch.linspace(-8., -2., 97, dtype=torch.float64)
    valid = torch.ones(97, dtype=torch.bool)
    valid[::9] = False
    forced = torch.zeros(97, dtype=torch.bool)
    complete_generator = torch.Generator().manual_seed(seed)
    chunk_generator = torch.Generator().manual_seed(seed)
    expected = sample_hazards(logits, valid, forced, complete_generator)
    budget = None
    actual = None
    for start, stop in zip((0, 7, 23, 65), (7, 23, 65, 97)):
        index, budget = sample_hazards(logits[start:stop], valid[start:stop], forced[start:stop],
                                       chunk_generator, budget)
        if index is not None:
            actual = start + index
            break
    assert (actual, budget) == expected
    assert torch.equal(complete_generator.get_state(), chunk_generator.get_state())


def test_sampling_forced_terminal_and_survived_crop_have_different_meanings():
    logits = torch.tensor([-float('inf'), float('nan'), -float('inf')])
    valid = torch.tensor([True, False, True])
    generator = torch.Generator().manual_seed(12)
    assert sample_hazards(logits, valid, torch.zeros(3, dtype=torch.bool), generator, 2.) == (None, 2.)
    assert sample_hazards(logits, valid, torch.tensor([False, True, True]), generator, 2.) == (2, 0.)
