"""Native-time score events separated from column assignment.

Four abstract resources carry LN identities and recovery deadlines. Identities
are allocated in ascending free-ID order; simultaneous source starts bind in
column order. These labels are canonical, not permutation-invariant semantics.
"""
from dataclasses import dataclass

import numpy as np

from ..bounded_typed_continuation.contract import ROW_ACTIONS
from ..bounded_typed_continuation.features import TIME_DIM, time_features

MARKS = tuple((tap, ln, mask) for tap in range(5) for ln in range(5-tap)
              for mask in range(16) if tap + ln or mask)
MARK_INDEX = {m: i for i, m in enumerate(MARKS)}
MARK_ARRAY = np.asarray(MARKS)
RELEASE_BITS = ((MARK_ARRAY[:, 2, None] >> np.arange(4)) & 1).astype(bool)
HEADS = MARK_ARRAY[:, 0] + MARK_ARRAY[:, 1]
ACTIONS = np.asarray(ROW_ACTIONS)
TOKEN_DIM = TIME_DIM + 6
CLOCK_DIM = 10 * TIME_DIM + 4


@dataclass(frozen=True)
class Recovery:
    hh: int = 37
    rh: int = 25
    hr: int = 21


@dataclass(frozen=True)
class Resources:
    starts: tuple = (None, None, None, None)
    free_at: tuple = (-1000000,)*4
    previous: int | None = None
    last_head: int | None = None

    def support(self, now, duration, recovery):
        occupied = np.array([s is not None for s in self.starts])
        releasable = np.array([s is not None and now-s >= recovery.hr for s in self.starts])
        eligible = sum(t <= now for t in self.free_at)
        allowed = (HEADS <= eligible) & ~(RELEASE_BITS & ~releasable).any(-1)
        if now + recovery.hr > duration:
            allowed &= MARK_ARRAY[:, 1] == 0
        if now == duration:
            allowed &= (RELEASE_BITS == occupied).all(-1)
        return allowed

    def type_support(self, times, duration, recovery):
        times = np.asarray(times)
        free = np.array(self.free_at, dtype=np.int64)
        available = (times[..., None] >= free).any(-1) if len(free) else np.zeros(times.shape, bool)
        starts = np.array([s for s in self.starts if s is not None], dtype=np.int64)
        release = ((times[..., None] - starts) >= recovery.hr).any(-1) if len(starts) else np.zeros(times.shape, bool)
        none = ~((times == duration) & bool(len(starts)))
        return np.stack((none, available, release), -1)

    def advance(self, now, mark, recovery):
        tap, ln, mask = MARKS[mark]
        starts = list(self.starts)
        free = sorted(max(t, now) for t in self.free_at)
        # Releases at this event cannot supply an immediate new attack.
        free = free[tap+ln:]
        for i in range(4):
            if mask & (1 << i):
                free.append(max(starts[i] + recovery.hh, now + recovery.rh))
                starts[i] = None
        fresh = [i for i, s in enumerate(starts) if s is None][:ln]
        for i in fresh:
            starts[i] = now
        free.extend([now + recovery.hh]*tap)
        return Resources(tuple(starts), tuple(sorted(free)), now,
                         now if tap+ln else self.last_head), tuple(fresh)

    def clocks(self, times, duration, *, remaining_availability=False, head_phase=False):
        times = np.asarray(times, dtype=np.float64)
        free = list(sorted(self.free_at)) + [None]*(4-len(self.free_at))
        values = np.stack((*(times-t if t is not None else np.full_like(times, np.nan) for t in self.starts),
                           *((np.maximum(t-times, 0) if remaining_availability else t-times)
                             if t is not None else np.full_like(times, np.nan) for t in free),
                           times-(self.last_head if head_phase else self.previous)
                           if (self.last_head if head_phase else self.previous) is not None else np.full_like(times, np.nan),
                           duration-times), -1)
        clocks = time_features(values).reshape(*times.shape, 10*TIME_DIM)
        held = np.broadcast_to([s is not None for s in self.starts], (*times.shape, 4))
        return np.concatenate((clocks, held), -1).astype(np.float32)


def tokens(times, marks, previous):
    counts = MARK_ARRAY[np.asarray(marks, int), :2] / 4
    bits = RELEASE_BITS[np.asarray(marks, int)]
    raw = np.concatenate((time_features(np.asarray(times)-np.asarray(previous, float)), counts, bits), -1)
    return np.repeat(raw[:, None, :], 2, axis=1).astype(np.float32)


def head_tokens(times, marks, previous):
    """Only head gaps/counts enter musical head memory; tails have their own path."""
    raw = tokens(times, marks, previous)
    raw[..., -4:] = 0
    return raw


def preview(events, now, count, bindings):
    """Current and future typed events; no future materialized columns."""
    result = np.zeros((count, TOKEN_DIM), np.float32)
    for i, (t, m) in enumerate(events[:count]):
        result[i] = np.concatenate((time_features(t-now), MARK_ARRAY[m, :2]/4, RELEASE_BITS[m]))
    binding = np.eye(5, dtype=np.float32)[[4 if lane is None else lane for lane in bindings]]
    return np.concatenate((result.reshape(-1), binding.reshape(-1)))


def row_support(replay, bindings, now, mark, duration, recovery):
    """Realize exactly this mark with its LN identities and physical recovery."""
    tap, ln, mask = MARKS[mark]
    releases = np.zeros(4, bool)
    for i, lane in enumerate(bindings):
        if mask & (1 << i):
            releases[lane] = True
    held = np.asarray(replay.occupancy)
    head = np.isin(ACTIONS, (1, 2))
    valid = ((ACTIONS == 1).sum(-1) == tap) & ((ACTIONS == 2).sum(-1) == ln)
    valid &= ((ACTIONS == 3) == releases).all(-1)
    valid &= ~(head & held).any(-1)
    for lane in range(4):
        attack, release, start = replay.last_lane_attack_ms[lane], replay.last_lane_release_ms[lane], replay.open_ln_start_ms[lane]
        if ((attack is not None and now-attack < recovery.hh) or
                (release is not None and now-release < recovery.rh)):
            valid &= ~head[:, lane]
        if start is not None and now-start < recovery.hr:
            valid &= ACTIONS[:, lane] != 3
    if now + recovery.hr > duration:
        valid &= ~(ACTIONS == 2).any(-1)
    return valid


def bind_row(bindings, mark, actions, fresh):
    result = list(bindings)
    mask = MARKS[mark][2]
    for i in range(4):
        if mask & (1 << i):
            result[i] = None
    for i, lane in zip(fresh, np.flatnonzero(np.asarray(actions) == 2)):
        result[i] = int(lane)
    return tuple(result)


@dataclass
class Program:
    events: list
    states: list
    bindings: list


def extract(rows, duration, recovery=Recovery()):
    state, binding = Resources(), (None,)*4
    events, states, bindings = [], [], []
    for row in rows:
        now, actions = int(row['time']), row['actions']
        mask = sum(1 << i for i, lane in enumerate(binding) if lane is not None and actions[lane] == 3)
        mark = MARK_INDEX[(int((actions == 1).sum()), int((actions == 2).sum()), mask)]
        if not state.support(now, duration, recovery)[mark]:
            raise ValueError(f'Source program exceeds resource support at {now}')
        states.append(state); bindings.append(binding); events.append((now, mark))
        state, fresh = state.advance(now, mark, recovery)
        binding = bind_row(binding, mark, actions, fresh)
    states.append(state); bindings.append(binding)
    return Program(events, states, bindings)
