"""Parameter-independent features from permitted facts and external timing.

All timestamp subtraction happens in float64 on CPU before float32 conversion.
Raw content contains one physical row, its preceding gap and newly committed
object plans. Exact historical clocks are read separately at the current query.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..oracle_time_continuation.features import TIME_DIM, TIME_SCALES_MS
from ..oracle_time_continuation.schema import CompleteRow
from ..scoped_style_modeling.dataset import ContractError
from .contract import Schedule, Timing

RELATIVE_LANES = ((0, 1, 3, 2), (3, 2, 0, 1))
LOOKAHEAD = 16
COUNT_SPANS_MS = (250., 1000., 4000., 16000., 64000.)
GAP_THRESHOLDS_MS = (2000., 8000., 32000.)
CONTENT_DIM = 16 + 5 * TIME_DIM
TIMING_DIM = (2 * LOOKAHEAD + 2 * len(GAP_THRESHOLDS_MS)) * TIME_DIM + LOOKAHEAD + 2 * len(COUNT_SPANS_MS) + 6
QUERY_DIM = 22 * TIME_DIM + 23 + TIMING_DIM
CANDIDATE_DIM = 4 * TIME_DIM + 5
FACTOR_DIM = 12 + 4 * TIME_DIM + 2


def time_features(values):
    """Encode signed millisecond differences; NaN/None denotes unavailable."""
    values = np.asarray(values, dtype=np.float64)
    available = ~np.isnan(values)
    delta = np.where(available, values, 0.)
    seconds = delta / 1000.
    ratio = delta[..., None] / np.asarray(TIME_SCALES_MS)
    envelope = 1. / np.sqrt(1. + ratio * ratio)
    basis = np.concatenate((
        (seconds / (1. + np.abs(seconds)))[..., None], np.arcsinh(seconds)[..., None],
        ratio * envelope, envelope,
    ), axis=-1)
    return np.concatenate((basis * available[..., None], available[..., None]), axis=-1).astype(np.float32)


def content_features(rows: Sequence[CompleteRow], previous_times, new_end_times):
    """Encode committed rows with only their available, already chosen plans.

    The caller supplies one previous physical timestamp and four optional new
    LN endpoints per row. R0 and suffix-born R1 objects must pass missing ends.
    Use transition_features for validated online transitions.
    """
    count = len(rows)
    if not count:
        return np.empty((0, 2, CONTENT_DIM), dtype=np.float32)
    previous = np.asarray(previous_times, dtype=np.float64)
    ends = np.asarray(new_end_times, dtype=np.float64)
    if previous.shape != (count,) or ends.shape != (count, 4):
        raise ContractError('Content needs one preceding timestamp and four optional new plans per row')
    times = np.asarray([row.time_ms for row in rows], dtype=np.float64)
    actions = np.asarray([row.actions for row in rows])
    if (np.any(previous >= times) or
            np.any(~np.isnan(ends) & ((actions != 2) | (ends <= times[:, None])))):
        raise ContractError('Content plans belong only to newly committed strictly future LN ends')
    gap = time_features(times - previous)
    endpoint = time_features(ends - times[:, None])
    encoded = []
    for lanes in RELATIVE_LANES:
        onehot = np.eye(4, dtype=np.float32)[actions[:, lanes]].reshape(count, 16)
        encoded.append(np.concatenate((onehot, gap, endpoint[:, lanes].reshape(count, -1)), -1))
    return np.stack(encoded, axis=1)


def transition_features(before: Schedule, after: Schedule, row: CompleteRow | None):
    """Return one owned raw token, or None when the candidate had no event."""
    if after.index != before.index + 1 or before.timing is not after.timing or before.arm != after.arm:
        raise ContractError('Content transition must advance exactly one candidate in the same task')
    if row is None:
        if after.replay is not before.replay:
            raise ContractError('No-event cannot alter physical replay')
        return None
    if after.replay.last_row != row or after.replay.row_count != before.replay.row_count + 1:
        raise ContractError('Content row must be the newly committed physical row')
    ends = [before.timing.times_ms[end] if action == 2 and end is not None else None
            for action, end in zip(row.actions, after.known_ends)]
    previous = None if before.replay.last_row is None else before.replay.last_row.time_ms
    return content_features([row], [previous], [ends])[0]


class TimingView:
    """Compact timing-only indexes; dense query features are made on demand.

    This owner never accepts actions, source LN labels or a training-window end.
    The arrays are read-only so checkpoint recomputation sees the same inputs.
    """
    def __init__(self, timing: Timing):
        self.timing = timing
        self.times = np.asarray(timing.times_ms, dtype=np.float64)
        self.roles = None if timing.onsets is None else np.asarray(timing.onsets, dtype=np.bool_)
        self.role_prefix = None if self.roles is None else np.r_[0, np.cumsum(self.roles)]
        self.gaps = np.diff(self.times)
        self.gap_starts = tuple(np.flatnonzero(self.gaps >= threshold) for threshold in GAP_THRESHOLDS_MS)
        for array in (self.times, self.roles, self.role_prefix, self.gaps, *self.gap_starts):
            if array is not None:
                array.setflags(write=False)

    def _indices(self, indices):
        indices = np.asarray(indices)
        if indices.dtype.kind not in 'iu' or indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(self.times)):
            raise ContractError('Timing query indices must belong to the supplied candidates')
        return indices.astype(np.int64, copy=False)

    def queries(self, indices):
        indices = self._indices(indices)
        times = self.times[indices]
        future = indices[:, None] + np.arange(1, LOOKAHEAD + 1)
        available = future < len(self.times)
        safe = np.minimum(future, len(self.times) - 1)
        offsets = np.where(available, self.times[safe] - times[:, None], np.nan)
        gaps = np.where(available, self.times[safe] - self.times[np.maximum(safe - 1, 0)], np.nan)
        clock_parts = [offsets, gaps]
        for starts in self.gap_starts:
            position = np.searchsorted(starts, indices)
            if len(starts):
                valid = position < len(starts)
                chosen = starts[np.minimum(position, len(starts) - 1)]
                clock_parts.append(np.stack((np.where(valid, self.times[chosen] - times, np.nan),
                                             np.where(valid, self.gaps[chosen], np.nan)), -1))
            else:
                clock_parts.append(np.full((len(indices), 2), np.nan))
        end = np.searchsorted(self.times, times[:, None] + np.asarray(COUNT_SPANS_MS), side='right')
        count_r = np.log1p(end - indices[:, None] - 1) / 8.
        count_h = (np.zeros_like(count_r) if self.roles is None else
                   np.log1p(self.role_prefix[end] - self.role_prefix[indices[:, None] + 1]) / 8.)
        roles = np.zeros_like(offsets) if self.roles is None else self.roles[safe] * available
        extras = np.stack((np.full(len(indices), self.roles is not None),
                           np.zeros(len(indices)) if self.roles is None else self.roles[indices],
                           indices == len(self.times) - 1,
                           np.log1p(len(self.times) - indices - 1) / 8.,
                           np.zeros(len(indices)) if self.roles is None else
                           np.log1p(self.role_prefix[-1] - self.role_prefix[indices + 1]) / 8.,
                           indices / max(1, len(self.times) - 1)), -1)
        clocks = time_features(np.concatenate(clock_parts, -1)).reshape(len(indices), -1)
        return np.concatenate((clocks, roles, count_r, count_h, extras), -1).astype(np.float32)

    def candidates(self, index: int, start: int, stop: int):
        """Encode a full-support interval in bounded chunks, with no rank cap."""
        if not 0 <= index < start < stop <= len(self.times):
            raise ContractError('Endpoint block must be a nonempty strictly future candidate interval')
        selected = np.arange(start, stop)
        now, times = self.times[index], self.times[selected]
        after = np.where(selected + 1 < len(self.times), self.times[np.minimum(selected + 1, len(self.times) - 1)] - times, np.nan)
        clocks = np.stack((times - now, times - self.times[selected - 1], after, self.times[-1] - times), -1)
        extras = np.stack((np.log1p(selected - index) / 8.,
                           (selected - index) / (len(self.times) - 1 - index),
                           selected == len(self.times) - 1,
                           np.zeros(len(selected)) if self.roles is None else self.roles[selected],
                           np.full(len(selected), self.roles is not None)), -1)
        return np.concatenate((time_features(clocks).reshape(len(selected), -1), extras), -1).astype(np.float32)


def query_features(states: Sequence[Schedule], view: TimingView):
    """Read complete exact clocks/plans alongside the arm's permitted timing."""
    if not states or any(state.timing is not view.timing or state.finished for state in states):
        raise ContractError('Query states must share one timing view and have a pending candidate')
    time_rows, fact_rows = [], []
    for state in states:
        replay = state.replay
        clocks = replay.clocks_at(state.time_ms)
        ends = [None if end is None else view.times[end] - state.time_ms for end in state.known_ends]
        for hand, lanes in enumerate(RELATIVE_LANES):
            times = [value for lane in lanes for value in
                     (clocks.ln_age_ms[lane], clocks.lane_attack_ms[lane], clocks.lane_release_ms[lane])]
            times += [value for side in (hand, 1 - hand) for value in
                      (clocks.hand_attack_ms[side], clocks.hand_release_ms[side])]
            times += [clocks.previous_row_ms, clocks.since_first_row_ms]
            times += [ends[lane] for lane in lanes]
            time_rows.append(times)
            last = [0.] * 16 if replay.last_row is None else np.eye(4)[np.asarray(replay.last_row.actions)[list(lanes)]].ravel().tolist()
            fact_rows.append([*last, *(replay.occupancy[lane] for lane in lanes),
                              np.log1p(replay.row_count) / 10., np.log1p(replay.note_count) / 10., not replay.row_count])
    clocks = time_features(time_rows).reshape(len(states), 2, 22 * TIME_DIM)
    facts = np.asarray(fact_rows, dtype=np.float32).reshape(len(states), 2, 23)
    external = view.queries([state.index for state in states])[:, None].repeat(2, axis=1)
    return np.concatenate((clocks, facts, external), -1)


def factor_features(state: Schedule, heads, assigned, lane):
    """Use only the chosen head group and previous within-row endpoint factors."""
    state.endpoint_bounds(heads, assigned, lane)
    hand = 0 if lane < 2 else 1
    lanes = RELATIVE_LANES[hand]
    actions = np.eye(3, dtype=np.float32)[np.asarray(heads)[list(lanes)]].ravel()
    ends = [None if c not in assigned else state.timing.times_ms[assigned[c]] - state.time_ms for c in lanes]
    local_lane = np.eye(2, dtype=np.float32)[lanes.index(lane)]
    return np.concatenate((actions, time_features(ends).ravel(), local_lane))
