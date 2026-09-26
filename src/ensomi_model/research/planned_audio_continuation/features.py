"""Explicit timing/LN inputs and candidate transitions on the native-ms clock.

Skeleton feature builders accept timestamps, roles and LN projections, never
row-content embeddings or generic replay state. Row consequence/support builders
receive complete replay and may send a release window to the sampler. A release
opportunity is not a tail plan.
"""
from dataclasses import dataclass

import numpy as np

from ..bounded_typed_continuation.consequence import LANE_DIM, TIMING_DIM
from ..bounded_typed_continuation.contract import ROW_ACTIONS
from ..bounded_typed_continuation.features import RELATIVE_LANES, TIME_DIM, time_features
from ..joint_audio_continuation.state import legal_rows
from ..scoped_style_modeling.dataset import ContractError

RELEASE_CLOCK_DIM = 8 * TIME_DIM + 4
ACTIONS = np.asarray(ROW_ACTIONS)
HAS_HEAD = np.isin(ACTIONS, (1, 2)).any(-1)


@dataclass(frozen=True)
class LNProjection:
    starts_ms: tuple
    observed_through_ms: int

    def __post_init__(self):
        object.__setattr__(self, 'starts_ms', tuple(self.starts_ms))
        if (len(self.starts_ms) != 4 or type(self.observed_through_ms) is not int or
                self.observed_through_ms < -1 or any(t is not None and
                (not np.isfinite(t) or not 0 <= t <= self.observed_through_ms) for t in self.starts_ms)):
            raise ContractError('LN projection requires four committed starts and an observed clock')


def ln_start_times(starts):
    """Pack committed native-ms start tuples; -1 denotes an unoccupied slot."""
    return np.asarray([[-1 if t is None else int(t) for t in row] for row in starts],
                      dtype=np.int64).reshape(-1, 4)


@dataclass(frozen=True)
class HeadPreview:
    times_ms: tuple
    complete: bool

    def __post_init__(self):
        object.__setattr__(self, 'times_ms', tuple(self.times_ms))
        if (type(self.complete) is not bool or any(type(t) is not int or t < 0 for t in self.times_ms) or
                any(a >= b for a, b in zip(self.times_ms, self.times_ms[1:]))):
            raise ContractError('Head preview requires strictly increasing native times and an explicit end flag')


def preview_after(head_times, now, count):
    """A fixed count is not EOS; fewer remaining heads imply a complete plan."""
    first = int(np.searchsorted(head_times, now, side='right'))
    times = tuple(int(t) for t in head_times[first:first + count])
    return HeadPreview(times, len(times) < count)


def skeleton_tokens(times, previous_times, roles=None):
    gaps = time_features(np.asarray(times, np.float64) - np.asarray(previous_times, np.float64))
    if roles is not None:
        roles = np.asarray(roles, np.bool_)
        gaps = np.concatenate((gaps, np.stack((roles, ~roles), -1)), -1)
    return np.repeat(gaps[:, None], 2, axis=1).astype(np.float32)


def head_clocks(previous_heads, times):
    now = np.asarray(times, np.float64)
    previous = np.asarray(previous_heads, np.float64)
    return time_features(np.stack((now - previous, now), -1)).reshape(len(now), 2 * TIME_DIM)


def preview_features(previews, times, roles, duration_ms, count):
    result = np.zeros((len(times), (count + 1) * TIME_DIM + 2), np.float32)
    for i, (preview, now, role) in enumerate(zip(previews, times, roles)):
        if len(preview.times_ms) > count or any(t <= now or t > duration_ms for t in preview.times_ms):
            raise ContractError('Row preview must contain only future heads inside the true audio clock')
        gaps = [t - now for t in preview.times_ms] + [None] * (count - len(preview.times_ms))
        result[i, :-2] = time_features([*gaps, duration_ms - now]).reshape(-1)
        result[i, -2:] = [role, preview.complete]
    return result


def release_clocks(states, previous_skeleton_times, times, previews, duration_ms):
    """Mirror-paired LN clocks plus skeleton clocks; no tap/chord facts enter."""
    result = np.zeros((len(times), 2, RELEASE_CLOCK_DIM), np.float32)
    for i, (state, previous, now, preview) in enumerate(zip(states, previous_skeleton_times, times, previews)):
        if not isinstance(state, LNProjection) or now < state.observed_through_ms:
            raise ContractError('Release queries require a past LN projection')
        future = list(preview.times_ms[:2]) + [None] * max(0, 2 - len(preview.times_ms))
        for hand, lanes in enumerate(RELATIVE_LANES):
            ages = [None if state.starts_ms[lane] is None else now - state.starts_ms[lane] for lane in lanes]
            clocks = [*ages, None if previous is None else now - previous,
                      *(None if t is None else t - now for t in future), duration_ms - now]
            result[i, hand, :8 * TIME_DIM] = time_features(clocks).reshape(-1)
            result[i, hand, -4:] = [state.starts_ms[lane] is not None for lane in lanes]
    return result


def release_masks(states, native_times, previews, duration_ms, *, minimum_action_gap_ms=0, windows=None):
    """Mark physical support and certain last hazards, never a crop-edge closure.

    A full-hold wait's earlier hazards determine whether its final event mass
    follows the raw deadline-atom law or the conditionally normalized law.
    Optional windows are earliest/latest bounds supplied by R1 feasibility;
    they contain neither row embeddings nor future release identities.
    """
    valid = np.zeros_like(native_times, dtype=np.bool_)
    forced = np.zeros_like(valid)
    for i, (state, preview) in enumerate(zip(states, previews)):
        held = sum(t is not None for t in state.starts_ms)
        if not held:
            continue
        next_h = preview.times_ms[0] if preview.times_ms else None
        if next_h is None and not preview.complete:
            raise ContractError('Release scheduling needs the next head or known plan completion')
        valid[i] = (native_times[i] > state.observed_through_ms) & (native_times[i] <= duration_ms)
        if next_h is not None:
            valid[i] &= native_times[i] < next_h
        if minimum_action_gap_ms:
            from .spacing import release_limits
            earliest, deadline = (release_limits(state, preview, duration_ms, minimum_action_gap_ms)
                                  if windows is None else windows[i])
            valid[i] &= (native_times[i] >= earliest) & (native_times[i] <= deadline)
            if (next_h is None or deadline < next_h) and earliest > deadline:
                raise ContractError('Action spacing has no eligible release before its deadline')
        else:
            deadline = next_h - 1 if held == 4 and next_h is not None else duration_ms
            if held == 4 and next_h is not None and deadline <= state.observed_through_ms:
                raise ContractError('All-held state has no release clock before the planned head')
        forced[i] = valid[i] & (native_times[i] == deadline)
    return valid, forced


def row_support(replays, times, roles, previews, duration_ms):
    support = legal_rows(replays, [t == duration_ms for t in times])
    for i, (state, now, head, preview) in enumerate(zip(replays, times, roles, previews)):
        support[i] &= HAS_HEAD if head else ~HAS_HEAD
        if preview.times_ms and preview.times_ms[0] == now + 1:
            occupied = np.asarray(state.occupancy)
            post = np.where(ACTIONS == 3, False, occupied[None] | (ACTIONS == 2))
            support[i] &= ~post.all(-1)
        if not support[i].any():
            raise ContractError('Planned role has no legal row under committed LN state')
    return support


def consequences(replays, times, previews, duration_ms):
    """R1 frontier2 coordinates with earliest native opportunity at now + 1.

    The next candidate is a possible native event clock, not a forecast of the
    next realized release. No seed/source endpoints enter this representation.
    Future clocks passively advance the immediate candidate state.
    """
    count = len(times)
    local = np.zeros((count, 4, 4, LANE_DIM), np.float32)
    local[..., :4] = np.eye(4, dtype=np.float32)
    timing = np.zeros((count, TIMING_DIM + TIME_DIM), np.float32)
    if not count:
        return local, timing
    now = np.asarray(times, np.float64)[:, None, None]
    starts = np.asarray([s.open_ln_start_ms for s in replays], np.float64)[..., None]
    attacks = np.asarray([s.last_lane_attack_ms for s in replays], np.float64)[..., None]
    releases = np.asarray([s.last_lane_release_ms for s in replays], np.float64)[..., None]
    next_h = np.asarray([p.times_ms[0] if p.times_ms else np.nan for p in previews])[:, None, None]
    next_r = np.where(now < duration_ms, now + 1, np.nan)
    action = np.arange(4)[None, None]
    heads = (action == 1) | (action == 2)
    post_attack = np.where(heads, now, attacks)
    post_release = np.where(action == 3, now, releases)
    post_start = np.where(action == 3, np.nan, np.where(action == 2, now, starts))
    occupied = ~np.isnan(post_start)
    clocks = np.stack((
        np.where(heads, now - attacks, np.nan),
        np.where(heads, now - releases, np.nan),
        np.where(action == 3, now - starts, np.nan),
        next_h - post_attack, next_h - post_release, next_h - post_start,
        np.where(occupied, next_h - next_r, np.nan),
    ), -1)
    local[..., 4] = occupied
    local[..., 5:] = time_features(clocks).reshape(count, 4, 4, 7 * TIME_DIM)
    timing[:, :TIME_DIM] = time_features((next_r - now)[:, 0, 0])
    timing[:, TIME_DIM:2 * TIME_DIM] = time_features((next_h - now)[:, 0, 0])
    timing[:, TIMING_DIM - 1] = (next_h == next_r)[:, 0, 0]
    second = np.asarray([p.times_ms[1] if len(p.times_ms) > 1 else np.nan for p in previews])
    timing[:, TIMING_DIM:] = time_features(second - now[:, 0, 0])
    return local, timing
