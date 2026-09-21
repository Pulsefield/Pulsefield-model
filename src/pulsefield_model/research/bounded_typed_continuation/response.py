"""Optimistic short-head counts over a row and the next two required onsets.

Future actions may use one TAP per H and earliest legal unknown-LN releases.
More future heads or new holds cannot improve this minimum. Original seed
endpoints remain fixed. This machine preference is not a calibrated gameplay
cost or an inference constraint, and consumes no source suffix actions.
"""
import numpy as np

from ..scoped_style_modeling.dataset import ContractError
from .contract import Arm, ROW_ACTIONS
from .support import row_supports

ACTIONS = np.asarray(ROW_ACTIONS)
HEADS = np.isin(ACTIONS, (1, 2))
HEAD_COUNTS = HEADS.sum(-1)
LN_COUNTS = (ACTIONS == 2).sum(-1)


def response_costs(state, threshold=30.):
    """Return exact optimistic costs and support for all 256 current rows."""
    if state.arm != Arm.R1 or state.finished or not np.isfinite(threshold) or threshold <= 0:
        raise ContractError('Response costs require an active R1 state and positive finite threshold')
    legal = np.asarray(row_supports([state])[0], bool)
    now = state.time_ms
    attacks = np.asarray([float('-inf') if t is None else t for t in state.replay.last_lane_attack_ms])
    releases = np.asarray([float('-inf') if t is None else t for t in state.replay.last_lane_release_ms])
    occupied = np.asarray(state.replay.occupancy)
    immediate = (HEADS & ((now - attacks < threshold) | (now - releases < threshold))[None]).sum(-1)
    first = state._next_onset(state.index)
    if first is None:
        return np.where(legal, immediate, 99), legal
    second = state._next_onset(first)
    post_occupied = np.where(ACTIONS == 3, False, occupied[None] | (ACTIONS == 2))
    post_attack = np.where(HEADS, now, attacks[None])
    post_release = np.where(ACTIONS == 3, now, releases[None])
    ends = np.asarray([state.index + 1 if end is None else end for end in state.known_ends])
    end_times = np.asarray(state.timing.times_ms)[ends]

    def available(index):
        released = post_occupied & (ends[None] < index)
        free = ~post_occupied | released
        last_release = np.where(released, end_times[None], post_release)
        t = state.timing.times_ms[index]
        recovered = free & (t - post_attack >= threshold) & (t - last_release >= threshold)
        return free, recovered

    free1, safe1 = available(first)
    assert free1[legal].any(-1).all()
    values = np.where(free1, ~safe1, 99).astype(np.int64)
    if second is not None:
        _, safe2 = available(second)
        short_spacing = state.timing.times_ms[second] - state.timing.times_ms[first] < threshold
        # The sole TAP at H1 resets only its own attack clock. Unknown holds
        # close at their earliest candidate, which may be H1 itself.
        remaining = safe2.sum(-1)[:, None] - (safe2 & short_spacing)
        values = values + (remaining == 0)
    total = immediate + values.min(-1)
    return np.where(legal, total, 99), legal


def response_preference(state, action, threshold=30.):
    """Prefer a lower cost with the least head/LN-count and lane-action change.

    The denominator family includes the sampled composition and the preferred
    compositions. A learner must improve relative probability within this family;
    assigning probability to an unrelated composition cannot satisfy the loss.
    """
    costs, legal = response_costs(state, threshold)
    actual = np.flatnonzero((ACTIONS == action).all(-1)).item()
    if not legal[actual]:
        raise ContractError('Response preference requires a legal sampled action')
    lower = legal & (costs < costs[actual])
    if not lower.any():
        return None
    head_change = np.abs(HEAD_COUNTS - HEAD_COUNTS[actual])
    ln_change = np.abs(LN_COUNTS - LN_COUNTS[actual])
    lower &= head_change == head_change[lower].min()
    lower &= ln_change == ln_change[lower].min()
    family = legal & (HEAD_COUNTS == HEAD_COUNTS[actual]) & (LN_COUNTS == LN_COUNTS[actual])
    for heads, lns in set(zip(HEAD_COUNTS[lower].tolist(), LN_COUNTS[lower].tolist())):
        family |= legal & (HEAD_COUNTS == heads) & (LN_COUNTS == lns)
    lower &= costs == costs[lower].min()
    distance = (ACTIONS != ACTIONS[actual]).sum(-1)
    good = lower & (distance == distance[lower].min())
    assert good.any() and not good[actual] and family[actual] and (good <= family).all()
    return good, family, dict(actual_cost=int(costs[actual]), preferred_cost=int(costs[good].min()),
        head_count_delta=int(head_change[good].min()), ln_count_delta=int(ln_change[good].min()),
        good_actions=int(good.sum()), family_actions=int(family.sum()))
