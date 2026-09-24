"""Optimistic row responses for the confirmed strict same-lane attack criterion.

This inference policy changes selection from a learned proposal. Its costs are
not likelihoods or physical legality, and hypothetical releases are not forecasts.
"""
import numpy as np
import torch

from ..bounded_typed_continuation.contract import ROW_ACTIONS

SHORT_ATTACK_MS = 20
_ACTIONS = np.asarray(ROW_ACTIONS)
_COUNTS = np.stack((np.isin(_ACTIONS, (1, 2)).sum(-1), (_ACTIONS == 2).sum(-1),
                    (_ACTIONS == 3).sum(-1)), -1)


def short_attack_pairs(rows):
    """Successive TAP/LN-head pairs; exactly 20 ms and releases do not qualify."""
    previous = [None] * 4
    result = []
    for row in rows:
        for lane, action in enumerate(row.actions):
            if action not in (1, 2):
                continue
            prior = previous[lane]
            if prior is not None and row.time_ms - prior[0] < SHORT_ATTACK_MS:
                result.append(dict(lane=lane, previous_ms=prior[0], time_ms=row.time_ms,
                                   gap_ms=row.time_ms - prior[0], actions=[prior[1], action]))
            previous[lane] = row.time_ms, action
    return result


def short_attack_costs(state, now, preview, *, candidates=_ACTIONS):
    """Minimum short-attack count for each candidate and its previewed H events.

    Inputs are a valid committed prefix, a later native-ms clock, and strictly
    future head times. Callers apply physical/role support separately. Future
    materialization uses one TAP per H; held lanes may close at now + 1. Thus a
    held lane cannot attack at now + 1, but can at any later H. This is an
    optimistic lower bound over those H events, not a guarantee that the actual
    release policy will free those lanes. A finite preview can omit later Hs.

    All queried Hs are strictly before now + 20. Reusing any lane within that
    horizon incurs another short pair, so 16 lane-use masks suffice. Previous
    committed attack times determine the cost of each lane's first future use.
    No actual future action or LN endpoint enters this computation.
    """
    actions = np.asarray(candidates)
    heads = np.isin(actions, (1, 2))
    previous = np.array([t if t is not None else -np.inf for t in state.last_lane_attack_ms])
    immediate = (heads & (now - previous < SHORT_ATTACK_MS)).sum(-1)
    future = [t for t in preview.times_ms if t < now + SHORT_ATTACK_MS]
    if not future:
        return immediate
    post_attack = np.where(heads, now, previous)
    occupied = np.where(actions == 3, False, np.asarray(state.occupancy) | (actions == 2))
    # Infinity keeps a physically impossible relaxed continuation distinct from
    # an unavoidable positive count. Native row support prevents that case.
    cost = np.full((len(actions), 16), np.inf)
    cost[:, 0] = immediate
    masks = np.arange(16)
    rows = np.arange(len(actions))[:, None]
    for time_ms in future:
        updated = np.full_like(cost, np.inf)
        for lane in range(4):
            repeat = ((masks & (1 << lane)) != 0)[None] | (
                time_ms - post_attack[:, lane, None] < SHORT_ATTACK_MS)
            possible = ~occupied[:, lane] | (time_ms > now + 1)
            values = np.where(possible[:, None], cost + repeat, np.inf)
            np.minimum.at(updated, (rows, (masks | (1 << lane))[None]), values)
        cost = updated
    return cost.min(-1)


def select_response_row(log_probs, proposal, legal, costs, rng):
    """Keep a minimum-cost proposal, otherwise draw a minimally changed row.

    Count changes are prioritized lexicographically: heads, LN starts, releases,
    then lane-action Hamming distance. Draw within that family using the original
    log probabilities and an independent CPU correction generator. No random
    number is consumed when the proposal already minimizes response cost.
    """
    family = legal & (costs == costs[legal].min())
    if family[proposal]:
        return proposal
    changes = np.concatenate((np.abs(_COUNTS - _COUNTS[proposal]),
                              (_ACTIONS != _ACTIONS[proposal]).sum(-1)[:, None]), -1)
    for values in changes.T:
        family &= values == values[family].min()
    indices = torch.from_numpy(np.flatnonzero(family))
    probabilities = log_probs[indices].softmax(-1)
    chosen = int(torch.multinomial(probabilities, 1, generator=rng))
    return int(indices[chosen])
