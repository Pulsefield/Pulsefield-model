"""Exact feasibility conditioning for a future linked-endpoint decoder.

This module is not used by the current endpoint-fitting experiment. The input
heads have conditionally independent endpoint distributions. Conditioning their
joint distribution on at least one early release preserves that distribution,
unlike forcing the last sampled head to close early without adjusting preceding
sampling probabilities.
"""
import torch


def requires_early_release(old_ends, newly_held, next_onset):
    """A release exactly at an onset cannot also attack in that same lane."""
    if next_onset is None:
        return False
    return all((end is not None and end >= next_onset) or lane in newly_held
               for lane,end in enumerate(old_ends))


def conditional_branch(log_probs, early, position, satisfied):
    """Next-head log probabilities under the joint at-least-one-early event.

    early marks endpoints strictly before the next required onset. satisfied
    means a lane is already available, or an earlier sampled head closes early.
    No source endpoint label is an argument.
    """
    log_probs=log_probs.double()
    if satisfied:
        return log_probs[position]
    # Sum disjoint first-early events in log space. Subtracting an all-late
    # probability from one loses rare but valid early mass through cancellation.
    remaining=log_probs[position+1:]
    if len(remaining):
        early_mass=remaining[:,early].logsumexp(-1)
        late_mass=remaining[:,~early].logsumexp(-1)
        preceding_late=torch.cat((late_mass.new_zeros(1),late_mass.cumsum(0)[:-1]))
        log_other_succeeds=(preceding_late+early_mass).logsumexp(0)
    else:
        log_other_succeeds=log_probs.new_tensor(-torch.inf)
    adjusted=log_probs[position]+torch.where(early,0.,log_other_succeeds)
    normalizer=adjusted.logsumexp(-1)
    if not bool(torch.isfinite(normalizer)):
        raise ValueError('Chosen head row has no feasible endpoint assignment')
    return adjusted-normalizer
