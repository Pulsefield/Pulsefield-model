"""Optional joint HH/RH conditioning of an already normalized row law.

The release-to-head predicate is an experimental preference. Neither policy
changes physical row support, H times, LN duration support or learned weights.
"""
import numpy as np
import torch

from ..bounded_typed_continuation.contract import ROW_ACTIONS
from ..scoped_style_modeling.dataset import ContractError

_ACTIONS = np.asarray(ROW_ACTIONS)
_HEADS = np.isin(_ACTIONS, (1, 2))


class NoRowContinuation(Exception):
    """A sampled event has no allowed row; the unpublished proposal can retry."""
    def __init__(self, decision):
        super().__init__('row_constraint_empty')
        self.decision = decision


def allowed_rows(state, now, preview, mode):
    """Return current-clean candidates, optionally with a feasible short future.

    The caller intersects physical support. HH excludes gaps strictly below
    20 ms; the experimental RH predicate excludes gaps at most 20 ms. Preview
    mode checks every supplied H through now + 20, including that boundary.
    A complete preview, coverage through the horizon, or five future heads is
    required: five heads within this interval already make it impossible.

    No future release can make a currently held column attackable in this
    interval under RH. Future Hs are less than 20 ms apart, so they must use
    distinct columns. Their eligible-column sets are nested as clocks advance;
    a one-TAP-per-H continuation exists iff the jth set has at least j columns.
    This tests existence over that horizon, not its probability or later liveness.
    """
    if mode not in ('current', 'preview'):
        raise ContractError('Row constraints require current or preview mode')
    attacks = np.asarray([t if t is not None else -np.inf for t in state.last_lane_attack_ms])
    releases = np.asarray([t if t is not None else -np.inf for t in state.last_lane_release_ms])
    active_release = releases > attacks
    recent = (now-attacks < 20) | (active_release & (now-releases <= 20))
    allowed = ~(_HEADS & recent).any(-1)
    if mode == 'current':
        return allowed
    if any(t <= now for t in preview.times_ms):
        raise ContractError('Constraint preview must contain strictly future H times')
    future = [t for t in preview.times_ms if t <= now+20]
    if (not preview.complete and len(future) < 5 and
            (not preview.times_ms or preview.times_ms[-1] < now+20)):
        raise ContractError('Preview constraint requires the full 20-ms H horizon or five future heads')
    if not future:
        return allowed
    if len(future) > 4:
        return np.zeros(len(_ACTIONS), dtype=np.bool_)
    post_attacks = np.where(_HEADS, now, attacks)
    post_releases = np.where(_ACTIONS == 3, now, releases)
    occupied = (np.asarray(state.occupancy) | (_ACTIONS == 2)) & (_ACTIONS != 3)
    for required, time_ms in enumerate(future, 1):
        available = (~occupied & (time_ms-post_attacks >= 20) &
                     ((post_releases <= post_attacks) | (time_ms-post_releases > 20)))
        allowed &= available.sum(-1) >= required
    return allowed


def condition_rows(log_probs, state, now, preview, mode):
    """Mask the complete base law after count/layout composition.

    Returns conditional log probabilities and an optional changed-support
    record. Normalization prevents underflow when only very unlikely rows
    survive. Keeping all positive-probability rows returns the original tensor
    unchanged. Empty model support raises NoRowContinuation before row sampling.
    """
    allowed = torch.as_tensor(allowed_rows(state, now, preview, mode), device=log_probs.device)
    finite = torch.isfinite(log_probs)
    if not bool(finite.any()):
        raise ContractError('Row constraint requires a finite base row distribution')
    if not bool((finite & ~allowed).any()):
        return log_probs, None
    retained = log_probs.masked_fill(~allowed, -torch.inf)
    total = log_probs.logsumexp(-1)
    if not bool(torch.isfinite(total)):
        raise ContractError('Row constraint requires a finite base row distribution')
    normalizer = retained.logsumexp(-1)
    log_mass = normalizer-total
    mass = float(log_mass.exp())
    record = dict(mode=mode, time_ms=now, retained_probability=mass,
                  log_retained_probability=float(log_mass) if bool(torch.isfinite(log_mass)) else None,
                  excluded_rows=int((finite & ~allowed).sum()), allowed_rows=int((finite & allowed).sum()),
                  last_attacks=list(state.last_lane_attack_ms), last_releases=list(state.last_lane_release_ms),
                  ln_starts=list(state.open_ln_start_ms), future_heads=list(preview.times_ms),
                  preview_complete=preview.complete)
    if not bool((finite & allowed).any()):
        raise NoRowContinuation(record)
    return retained-normalizer, record
