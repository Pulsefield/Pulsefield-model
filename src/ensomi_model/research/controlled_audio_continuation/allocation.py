"""Finite LN-amount feedback within R1's chosen head/release-count families."""
from dataclasses import dataclass, replace

import torch

from ..planned_audio_continuation.counts import ROW_COUNTS


_COUNTS = torch.tensor(ROW_COUNTS)
_GROUP = 5*_COUNTS[:, 0]+_COUNTS[:, 2]
_MEMBERS = torch.arange(25)[:, None] == _GROUP


@dataclass(frozen=True)
class LnAmountState:
    start_ms: int | None = None
    fraction: float | None = None
    offset: float = 0.

    def in_scope(self, span):
        if span is None:
            return LnAmountState()
        if self.start_ms == span.start_ms and self.fraction == span.ln_fraction:
            return self
        return LnAmountState(span.start_ms, span.ln_fraction)


@dataclass(frozen=True)
class LnAmountFeedback:
    specification: str = 'r1-projected-ln-amount-v1'
    gain: float = 1/8
    maximum_log_odds: float = 2.

    def advance(self, state, tap, ln):
        """Remember a bounded correction, with no accumulated debt beyond it.

        The next decision receives the update from the actual committed heads.
        Releases, elapsed time and scope expiry create no compensating quota.
        """
        if state.fraction is None or tap+ln == 0:
            return state
        offset = state.offset+self.gain*(state.fraction*(tap+ln)-ln)
        return replace(state, offset=max(-self.maximum_log_odds, min(self.maximum_log_odds, offset)))

    def scores(self, log_probs, state):
        """Tilt LN allocation after other preferences, preserving P(heads, releases).

        Layout odds within each (heads, new LNs, releases) family also remain
        unchanged. Every previously supported row retains finite probability.
        """
        if state.fraction is None or state.offset == 0:
            return log_probs
        members = _MEMBERS.to(log_probs.device)
        group = _GROUP.to(log_probs.device)
        longs = _COUNTS[:, 1].to(log_probs)
        active = (members & torch.isfinite(log_probs)[None]).any(-1)
        def normalizer(values):
            masked = values[None].masked_fill(~members, -torch.inf)
            # Empty groups have no effect on the distribution. Keep their
            # unused normalizers finite so backward never evaluates 0 * NaN.
            return torch.where(active[:, None], masked, 0.).logsumexp(-1)
        before = normalizer(log_probs)
        tilted = log_probs+state.offset*longs
        after = normalizer(tilted)
        delta = after-before
        return tilted-delta[group]
