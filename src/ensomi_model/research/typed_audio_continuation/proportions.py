"""A semantic LN-count base that cannot be ignored by a history shortcut.

The request sets the natural parameter of a conditional binomial family. At
fixed history/support, its expected LN count increases with that parameter;
head-count/release-mask group masses remain those of the learned mark law.
"""
import math

import torch

from .program import MARKS

GROUPS = tuple(sorted({(tap+ln, mask) for tap, ln, mask in MARKS}))
GROUP_INDEX = tuple(GROUPS.index((tap+ln, mask)) for tap, ln, mask in MARKS)


def tilt_ln_count(raw, support, fraction, reference_fraction):
    """Shift scoped LN odds without capping local learned preferences.

    The head-count/release-mask marginal is unchanged. Unlike the bounded
    binomial base, sufficiently strong local evidence can concentrate on any
    feasible LN count. This is a conditional prior, not an exact scope quota.
    """
    group = torch.tensor(GROUP_INDEX, device=raw.device)
    membership = group[None] == torch.arange(len(GROUPS), device=raw.device)[:, None]
    allowed = support[:, None] & membership[None]
    active = allowed.any(-1)

    def group_sum(values):
        masked = values[:, None].masked_fill(~allowed, -torch.inf)
        # Empty groups have no selected mark; a finite dummy keeps their
        # unused logsumexp derivative from contaminating valid groups.
        return torch.logsumexp(torch.where(active[..., None], masked, 0.), -1)

    original = raw.masked_fill(~support, -torch.inf).log_softmax(-1)
    mass = group_sum(original)
    rho = fraction.clamp(.0001, .9999)
    shift = torch.logit(rho) - math.log(reference_fraction/(1-reference_fraction))
    ln = raw.new_tensor([m[1] for m in MARKS])
    scores = raw + shift[:, None]*ln
    conditional = scores - group_sum(scores)[:, group]
    return (mass[:, group]+conditional).masked_fill(~support, -torch.inf)


def condition_ln_count(raw, support, fraction, *, residual_bound=1.):
    group = torch.tensor(GROUP_INDEX, device=raw.device)
    membership = group[None] == torch.arange(len(GROUPS), device=raw.device)[:, None]
    allowed = support[:, None] & membership[None]
    old = raw.masked_fill(~support, -torch.inf).log_softmax(-1)
    group_mass = old.exp() @ membership.to(raw.dtype).T
    mean = torch.where(allowed, raw[:, None], 0.).sum(-1) / allowed.sum(-1).clamp_min(1)
    centered = raw - mean[:, group]
    residual = residual_bound * centered.tanh()
    ln = raw.new_tensor([m[1] for m in MARKS])
    tap = raw.new_tensor([m[0] for m in MARKS])
    coefficient = raw.new_tensor([math.lgamma(t+l+1)-math.lgamma(t+1)-math.lgamma(l+1) for t,l,_ in MARKS])
    # These are proportions, not hard all-TAP/all-LN feasibility commands.
    rho = fraction.clamp(.0001, .9999)[:, None]
    score = residual + coefficient + ln*rho.log() + tap*torch.log1p(-rho)
    unnormalized = score.exp()*support
    group_z = unnormalized @ membership.to(raw.dtype).T
    probability = group_mass[:, group] * unnormalized / group_z[:, group].clamp_min(1e-30)
    return probability.clamp_min(1e-30).log().masked_fill(~support, -torch.inf)
