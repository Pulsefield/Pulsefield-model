"""Complete sequence code length and exact marginals of the same joint rows."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math

import torch
from torch import Tensor

from ..scoped_style_modeling.dataset import ContractError

MARGINAL_NAMES = ("press_count", "hand_press_configuration", "lane_state_transition")


@dataclass(frozen=True)
class ObjectiveConfig:
    normalization_rows: float = 128.
    lambda_struct: float = 0.

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ContractError(f"{name} must be finite and nonnegative")
        if self.normalization_rows == 0:
            raise ContractError("normalization_rows must be positive")


@dataclass(frozen=True)
class SequenceCost:
    loss: Tensor
    nll: Tensor
    marginal_nll: Tensor
    weighted_logit_gradient_squared: Tensor


def sequence_cost(log_probs: Tensor, legal: Tensor, valid: Tensor, targets: Tensor, occupancy: Tensor,
                  *, effective_batch_size: int, config: ObjectiveConfig = ObjectiveConfig()) -> SequenceCost:
    """Score ragged [B,Q,256] rows using one fixed effective-batch denominator.

    Targets are serialized row indices; occupancy is pre-row [B,Q,4]. Padding
    is removed before validation and logsumexp, so it has neither cost nor a
    gradient. Illegal truth, empty groups and nonfinite legal probabilities
    raise ContractError; no epsilon or target repair is applied.

    All three marginals have fixed weight 1/3. The detached diagnostic is the
    squared L2 norm of each weighted marginal's derivative with respect to
    pre-softmax row logits, summed over real rows. It is not a parameter norm.
    """
    shape = log_probs.shape[:-1]
    if (log_probs.ndim != 3 or log_probs.shape[-1] != 256 or legal.shape != log_probs.shape or
            valid.shape != shape or targets.shape != shape or occupancy.shape != (*shape, 4) or
            legal.dtype != torch.bool or valid.dtype != torch.bool or occupancy.dtype != torch.bool or
            targets.dtype != torch.long):
        raise ContractError("Sequence cost requires ragged joint rows, integer targets and boolean masks/occupancy")
    if type(effective_batch_size) is not int or effective_batch_size < shape[0]:
        raise ContractError("effective_batch_size must cover every window in the microbatch")
    if not bool(valid.any()):
        raise ContractError("Sequence cost requires at least one supervised row")
    lp, support, target, before = log_probs[valid], legal[valid], targets[valid], occupancy[valid]
    if bool(((target <= 0) | (target >= 256)).any()) or not bool(support.gather(1, target[:, None]).all()):
        raise ContractError("Cannot score an illegal target row")
    if (not bool(torch.isfinite(lp[support]).all()) or not bool(torch.isneginf(lp[~support]).all()) or
            not torch.allclose(lp.logsumexp(-1), torch.zeros_like(lp[:, 0]), atol=2e-5, rtol=0)):
        raise ContractError("Joint log probabilities must be normalized, finite on support and -inf elsewhere")
    table = torch.tensor(list(product(range(4), repeat=4)), device=lp.device)
    press = (table == 1) | (table == 2)
    count = press.sum(-1)
    # Preserve both ordered outer/inner hand pairs: ((lane 0,1), (lane 3,2)).
    hand_code = (press[:, (0, 1, 3, 2)] * table.new_tensor((8, 4, 2, 1))).sum(-1)
    after = (table[None] == 2) | (before[:, None] & (table[None] != 3))
    truth_after = after[torch.arange(len(target), device=lp.device), target]
    groups = torch.stack((count[None] == count[target, None], hand_code[None] == hand_code[target, None],
                          (after == truth_after[:, None]).all(-1)), dim=1) & support[:, None]
    marginal_logp = lp[:, None].masked_fill(~groups, -torch.inf).logsumexp(-1)
    if not bool(torch.isfinite(marginal_logp).all()):
        raise ContractError("A true structural marginal has an empty or nonfinite target group")
    nll = -lp.gather(1, target[:, None]).squeeze(1)
    marginal = -marginal_logp
    denominator = effective_batch_size * config.normalization_rows
    loss = nll.sum() / denominator
    if config.lambda_struct:
        loss = loss + config.lambda_struct * marginal.sum() / (3 * denominator)
    with torch.no_grad():
        conditional = (lp[:, None] - marginal_logp[:, :, None]).masked_fill(~groups, -torch.inf).exp()
        derivative = (lp.exp()[:, None] - conditional) * (config.lambda_struct / (3 * denominator))
        gradient_squared = derivative.square().sum((0, 2))
    full_nll = log_probs.new_zeros(shape).masked_scatter(valid, nll)
    full_marginal = log_probs.new_zeros((*shape, 3)).masked_scatter(valid[:, :, None].expand(-1, -1, 3), marginal)
    return SequenceCost(loss, full_nll, full_marginal, gradient_squared)
