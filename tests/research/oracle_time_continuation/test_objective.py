from itertools import product

import pytest
import torch

from pulsefield_model.research.oracle_time_continuation.model import row_index
from pulsefield_model.research.oracle_time_continuation.objective import ObjectiveConfig, sequence_cost
from pulsefield_model.research.oracle_time_continuation.replay import ExactReplayState, commit, legal_rows
from pulsefield_model.research.oracle_time_continuation.schema import CompleteRow
from pulsefield_model.research.scoped_style_modeling.dataset import ContractError


def structural_values(actions, before):
    press = tuple(action in (1, 2) for action in actions)
    after = tuple(True if action == 2 else False if action == 3 else held for action, held in zip(actions, before))
    return sum(press), ((press[0], press[1]), (press[3], press[2])), tuple(zip(before, after))


def all_occupancy_inputs():
    supports, occupancies, truths = [], [], []
    for occupied in product((False, True), repeat=4):
        opening = tuple(2 if held else 0 for held in occupied) if any(occupied) else (1, 0, 0, 0)
        history = commit(ExactReplayState(), CompleteRow(0, opening))
        for terminal in (False, True):
            support = legal_rows(history, is_terminal=terminal)
            supports.append(support)
            occupancies.append(occupied)
            truths.append(support[len(support) // 2])
    legal = torch.zeros(32, 1, 256, dtype=torch.bool)
    for i, support in enumerate(supports):
        legal[i, 0, [row_index(row) for row in support]] = True
    return (supports, occupancies, truths, legal, torch.ones(32, 1, dtype=torch.bool),
            torch.tensor([[row_index(row)] for row in truths]), torch.tensor(occupancies)[:, None])


def test_three_exact_marginals_match_independent_enumeration_for_every_occupancy_and_terminal():
    torch.manual_seed(81)
    supports, occupied, truth, legal, valid, targets, occupancy = all_occupancy_inputs()
    logits = torch.randn(32, 1, 256, requires_grad=True)
    lp = logits.masked_fill(~legal, -torch.inf).log_softmax(-1)
    config = ObjectiveConfig(lambda_struct=.7)
    cost = sequence_cost(lp, legal, valid, targets, occupancy, effective_batch_size=37, config=config)
    expected = []
    for index, (support, before, target) in enumerate(zip(supports, occupied, truth)):
        target_values = structural_values(target, before)
        expected.append(torch.stack([-torch.logsumexp(torch.stack([
            lp[index, 0, row_index(row)] for row in support if structural_values(row, before)[group] == target_values[group]
        ]), 0) for group in range(3)]))
    reference = torch.stack(expected)[:, None]
    torch.testing.assert_close(cost.marginal_nll, reference)
    nll = -lp.gather(-1, targets[:, :, None]).squeeze(-1)
    torch.testing.assert_close(cost.loss, (nll.sum() + .7 * reference.sum() / 3) / (37 * 128))
    cost.loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.norm() > 0
    assert not logits.grad[~legal].any()
    assert (cost.weighted_logit_gradient_squared > 0).all()


@pytest.mark.parametrize("group", (0, 1, 2))
def test_reported_marginal_logit_gradient_matches_autograd(group):
    torch.manual_seed(39)
    _, _, _, legal, valid, targets, occupancy = all_occupancy_inputs()
    logits = torch.randn(32, 1, 256, requires_grad=True)
    lp = logits.masked_fill(~legal, -torch.inf).log_softmax(-1)
    cost = sequence_cost(lp, legal, valid, targets, occupancy,
                         effective_batch_size=40, config=ObjectiveConfig(lambda_struct=.4))
    (.4 * cost.marginal_nll[:, :, group].sum() / (3 * 40 * 128)).backward()
    torch.testing.assert_close(logits.grad.square().sum(), cost.weighted_logit_gradient_squared[group])


def test_ragged_padding_and_short_tails_share_the_complete_update_denominator():
    legal = torch.zeros(2, 4, 256, dtype=torch.bool)
    # Two legal TAP rows, both with closed occupancy; deliberately simple reference probabilities.
    legal[:, :, [1, 4]] = True
    logits = torch.randn(2, 4, 256, requires_grad=True)
    lp = logits.masked_fill(~legal, -torch.inf).log_softmax(-1)
    valid = torch.tensor([[True, True, True, False], [True, False, False, False]])
    targets = torch.ones(2, 4, dtype=torch.long)
    targets[~valid] = -100
    occupancy = torch.zeros(2, 4, 4, dtype=torch.bool)
    cost = sequence_cost(lp.masked_fill(~valid[:, :, None], torch.nan), legal, valid, targets, occupancy,
                         effective_batch_size=7)
    torch.testing.assert_close(cost.loss, -lp[:, :, 1][valid].sum() / (7 * 128))
    assert not cost.nll[~valid].any() and not cost.marginal_nll[~valid].any()
    assert not cost.weighted_logit_gradient_squared.any()
    cost.loss.backward()
    assert not logits.grad[~valid].any()


def test_illegal_truth_and_nonfinite_probabilities_are_not_repaired():
    _, _, _, legal, valid, targets, occupancy = all_occupancy_inputs()
    lp = torch.zeros_like(legal, dtype=torch.float).masked_fill(~legal, -torch.inf).log_softmax(-1)
    invalid = targets.clone()
    invalid[0, 0] = 3
    with pytest.raises(ContractError, match="illegal target"):
        sequence_cost(lp, legal, valid, invalid, occupancy, effective_batch_size=32)
    broken = lp.clone()
    broken[0, 0, targets[0, 0]] = torch.nan
    with pytest.raises(ContractError, match="normalized"):
        sequence_cost(broken, legal, valid, targets, occupancy, effective_batch_size=32)
    with pytest.raises(ContractError, match="effective_batch_size"):
        sequence_cost(lp, legal, valid, targets, occupancy, effective_batch_size=31)
    for settings in ({"lambda_struct": -1}, {"lambda_struct": float("nan")}, {"normalization_rows": 0}):
        with pytest.raises(ContractError):
            ObjectiveConfig(**settings)
