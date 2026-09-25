"""Bounded LN-amount feedback over effective request episodes.

Only skeleton head counts enter this state. It is an optional inference policy,
not a hard quota, a player frontier or a change to the fitted likelihood.
"""
from dataclasses import dataclass
import math

from .controls import ControlSpan


def ln_episodes(controls):
    """Resolve LN request ownership without splitting on other attributes.

    Returning to an earlier request starts a fresh accounting episode. A later
    request's boundary does not reset counts before that boundary, and a fully
    shadowed request does not interrupt the effective owner's episode.
    """
    spans = [s for s in controls.spans if s.ln_fraction is not None]
    edges = sorted({t for s in spans for t in (s.start_ms, s.end_ms)})
    result, previous_owner = [], None
    for a, b in zip(edges, edges[1:]):
        owner = next((i for i in range(len(spans)-1, -1, -1)
                      if spans[i].start_ms <= a < spans[i].end_ms), None)
        if owner is not None:
            start = result.pop().start_ms if owner == previous_owner else a
            result.append(ControlSpan(start, b, ln_fraction=spans[owner].ln_fraction))
        previous_owner = owner
    return tuple(result)


@dataclass(frozen=True)
class Allocation:
    start_ms: int | None = None
    fraction: float | None = None
    heads: int = 0
    ln_heads: int = 0

    def in_scope(self, span):
        if span is None:
            return Allocation()
        if self.start_ms == span.start_ms and self.fraction == span.ln_fraction:
            return self
        return Allocation(span.start_ms, span.ln_fraction)

    def advance(self, tap, ln):
        return Allocation(self.start_ms, self.fraction, self.heads+tap+ln, self.ln_heads+ln)


@dataclass(frozen=True)
class LnFeedback:
    strength: float = 1.
    pseudocount_heads: float = 32.
    maximum_log_odds: float = 1.

    def log_odds_shift(self, allocation):
        """Shrink early observations toward the request, then apply negative feedback.

        The finite log-odds bound leaves learned local preferences unrestricted.
        No remaining-time term forces a compensating burst at scope expiry.
        """
        if allocation.fraction is None or allocation.heads == 0:
            return 0.
        rho = min(.9999, max(.0001, allocation.fraction))
        realized = (allocation.ln_heads+self.pseudocount_heads*rho)/(allocation.heads+self.pseudocount_heads)
        logit = lambda p: math.log(p/(1-p))
        delta = self.strength*(logit(rho)-logit(realized))
        return min(self.maximum_log_odds, max(-self.maximum_log_odds, delta))
