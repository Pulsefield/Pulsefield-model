"""Native-clock event hazards with exact survival and resumable sampling."""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from ..scoped_style_modeling.dataset import ContractError


def _check_hazards(logits: Tensor, valid: Tensor, forced: Tensor, ndim: int) -> None:
    if (logits.ndim != ndim or not logits.is_floating_point() or
            valid.shape != logits.shape or forced.shape != logits.shape or
            valid.dtype != torch.bool or forced.dtype != torch.bool or
            valid.device != logits.device or forced.device != logits.device):
        raise ContractError('Hazards need floating logits and matching boolean masks on one device')


def hazard_nll(logits: Tensor, valid: Tensor, event_index: Tensor, forced: Tensor) -> Tensor:
    """Return one event/survival negative log likelihood per batch member.

    Inputs logits/valid/forced have shape [batch, clocks]; event_index is long
    [batch] on the same device. A target of -1 observes no event in the valid
    interval. Other targets index a valid event clock. Invalid clocks are
    ignored, including their logits and forced flags. Forced valid clocks have
    hazard exactly one: their event cost is zero and surviving them has infinite
    cost. Target events after an earlier forced clock also have infinite cost.
    Masking occurs before nonlinearities so ignored nonfinite values cannot
    contaminate gradients. Malformed tensor contracts raise ContractError.
    """
    _check_hazards(logits, valid, forced, 2)
    batch, clocks = logits.shape
    if (event_index.shape != (batch,) or event_index.dtype != torch.long or
            event_index.device != logits.device):
        raise ContractError('Event indices must be one long value per example on the logits device')
    if bool(((event_index < -1) | (event_index >= clocks)).any()):
        raise ContractError('Event index must be -1 or a clock inside the interval')
    positions = torch.arange(clocks, device=logits.device)[None]
    event = positions == event_index[:, None]
    if bool((event & ~valid).any()):
        raise ContractError('An observed event must occur at a valid clock')
    survival = valid & ((event_index[:, None] < 0) | (positions < event_index[:, None]))
    learned = valid & ~forced
    active_logits = torch.where(learned & (survival | event), logits, 0.)
    costs = torch.where(survival & learned, F.softplus(active_logits), 0.)
    costs += torch.where(event & learned, F.softplus(-active_logits), 0.)
    loss = costs.sum(dim=-1)
    impossible = (survival & forced).any(dim=-1)
    return torch.where(impossible, torch.full_like(loss, math.inf), loss)


def sample_hazards(
    logits: Tensor, valid: Tensor, forced: Tensor, generator: torch.Generator,
    residual_budget: float | None = None,
) -> tuple[int | None, float]:
    """Sample a [clocks] hazard sequence using one CPU exponential threshold.

    Return (local event index, 0.) on an event, or (None, remaining threshold)
    after surviving the interval. Passing that threshold to the next chunk
    consumes no additional randomness and preserves the draw when the hazard
    values are unchanged. A valid forced clock always fires if reached. Invalid
    clocks are ignored. Logits/masks may reside on any one device; sampling
    computes float64 hazard masses on CPU without gradients. A residual must be
    finite and nonnegative. Malformed inputs raise ContractError.
    """
    _check_hazards(logits, valid, forced, 1)
    if generator.device.type != 'cpu':
        raise ContractError('Hazard sampling requires a CPU random generator')
    if residual_budget is None:
        budget = torch.empty((), dtype=torch.float64).exponential_(generator=generator).item()
    else:
        budget = float(residual_budget)
        if not math.isfinite(budget) or budget < 0.:
            raise ContractError('Residual hazard budget must be finite and nonnegative')
    valid_values = valid.detach().cpu().tolist()
    forced_values = forced.detach().cpu().tolist()
    values = logits.detach().cpu().to(dtype=torch.float64)
    masses = F.softplus(values).tolist()
    for index, (mass, present, certain) in enumerate(zip(masses, valid_values, forced_values)):
        if not present:
            continue
        if certain:
            return index, 0.
        if math.isnan(mass):
            raise ContractError('A valid learned hazard cannot be NaN')
        if budget <= mass:
            return index, 0.
        budget -= mass
    return None, budget
