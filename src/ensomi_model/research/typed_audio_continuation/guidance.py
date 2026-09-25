"""Difficulty guidance with explicit ownership of the amplified factors."""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .program import HEADS


def _guided(conditional, marginal, strength):
    valid = torch.isfinite(conditional) & torch.isfinite(marginal)
    c = torch.where(valid, conditional, 0.)
    u = torch.where(valid, marginal, 0.)
    return (c+(strength-1)*(c-u)).masked_fill(~valid, -torch.inf).log_softmax(-1)


def guide_head_clock(conditional, marginal, strength):
    """Guide P(head); retain conditional release/no-event odds given no head."""
    c_no = torch.logsumexp(conditional[..., (0, 2)], -1)
    u_no = torch.logsumexp(marginal[..., (0, 2)], -1)
    usable = (torch.isfinite(conditional[..., 1]) & torch.isfinite(c_no)
              & torch.isfinite(marginal[..., 1]) & torch.isfinite(u_no))
    c = torch.where(usable, conditional[..., 1]-c_no, 0.)
    u = torch.where(usable, marginal[..., 1]-u_no, 0.)
    logit = c+(strength-1)*(c-u)
    log_no = F.logsigmoid(-logit)
    guided = torch.stack((log_no+conditional[..., 0]-c_no, F.logsigmoid(logit),
                          log_no+conditional[..., 2]-c_no), -1)
    return torch.where(usable[..., None], guided, conditional)


def guide_head_count(conditional, marginal, strength):
    """Guide head-count mass; retain the LN/release law within each count."""
    heads = torch.as_tensor(HEADS, device=conditional.device)
    member = heads[None] == torch.arange(5, device=heads.device)[:, None]
    def masses(values):
        return values[..., None, :].masked_fill(~member, -torch.inf).logsumexp(-1)
    c, u = masses(conditional), masses(marginal)
    mass = _guided(c, u, strength)
    delta = torch.where(torch.isfinite(c), mass-c, 0.)
    return conditional+delta[..., heads]


@dataclass(frozen=True)
class StarGuidance:
    strength: float = 2.
    guide_releases: bool = False

    @torch.inference_mode()
    def scores(self, model, factor, *args, **kwargs):
        """Sampling-only contrast with stars omitted; other conditions retained.

        By default the head marginal, head-count marginal and complete-row
        geometry are guided. Releases still read the actual star condition but
        their conditional contrast is not amplified. ``guide_releases=True``
        selects the earlier full categorical guidance for comparison.
        """
        function = getattr(model, factor+'_log_probs')
        conditional = function(*args, **kwargs)
        position = 6 if factor == 'row' else 3
        known = 2+len(model.style_names)
        if self.strength == 1 or not bool(args[position][..., known].any()):
            return conditional
        control = args[position].clone()
        control[..., 0] = 0.
        control[..., known] = 0.
        altered = list(args)
        altered[position] = control
        marginal = function(*altered, **kwargs)
        if self.guide_releases or factor == 'row':
            return _guided(conditional, marginal, self.strength)
        if factor == 'clock':
            return guide_head_clock(conditional, marginal, self.strength)
        return guide_head_count(conditional, marginal, self.strength)
