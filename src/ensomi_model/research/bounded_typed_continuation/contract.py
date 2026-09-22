"""External timing, exact obligations and support for the three task arms.

Candidate position and materialized-row position are separate. Skipping a
candidate changes neither replay clocks nor learned action history. Known
endpoints are previous object decisions or explicit seed information; unknown
endpoints of open row-model LNs are represented separately from closed lanes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from itertools import product
from typing import Mapping, Sequence

from ..oracle_time_continuation.replay import ExactReplayState, commit
from ..oracle_time_continuation.schema import CompleteRow, checked_time
from ..scoped_style_modeling.dataset import ContractError

ROW_ACTIONS = tuple(product(range(4), repeat=4))
HEAD_ACTIONS = tuple(product(range(3), repeat=4))
NO_ENDS = (None, None, None, None)


class Arm(str, Enum):
    R0 = 'r0'
    R1 = 'r1'
    O1 = 'o1'


@dataclass(frozen=True)
class Timing:
    times_ms: tuple[float, ...]
    onsets: tuple[bool, ...] | None = None

    def __post_init__(self):
        if not isinstance(self.times_ms, tuple) or not self.times_ms:
            raise ContractError('Timing needs a nonempty immutable candidate sequence')
        times = tuple(checked_time(t) for t in self.times_ms)
        if any(a >= b for a, b in zip(times, times[1:])):
            raise ContractError('Candidate times must strictly increase')
        object.__setattr__(self, 'times_ms', times)
        if self.onsets is not None and (not isinstance(self.onsets, tuple) or
                len(self.onsets) != len(times) or any(type(x) is not bool for x in self.onsets)):
            raise ContractError('Onset roles must be immutable booleans matching candidate times')


def checked_actions(actions, maximum=3):
    if (not isinstance(actions, tuple) or len(actions) != 4 or
            any(type(a) is not int or not 0 <= a <= maximum for a in actions)):
        raise ContractError('An action decision requires four integer lane actions')
    return actions


@dataclass(frozen=True)
class Schedule:
    arm: Arm
    timing: Timing
    index: int = 0
    replay: ExactReplayState = ExactReplayState()
    known_ends: tuple[int | None, ...] = NO_ENDS

    def __post_init__(self):
        if not isinstance(self.arm, Arm):
            raise ContractError('Use an explicit R0, R1 or O1 task arm')
        if (self.arm == Arm.R0) != (self.timing.onsets is None):
            raise ContractError('Only R0 has untyped timing; R1/O1 require onset roles')
        if type(self.index) is not int or not 0 <= self.index <= len(self.timing.times_ms):
            raise ContractError('Candidate cursor is outside the supplied timing')
        if self.replay.row_count > self.index:
            raise ContractError('Physical replay cannot contain more rows than consumed candidates')
        if not isinstance(self.known_ends, tuple) or len(self.known_ends) != 4:
            raise ContractError('Known endpoint state requires four immutable lane entries')
        for opened, end in zip(self.replay.occupancy, self.known_ends):
            if end is not None:
                if (not opened or self.arm == Arm.R0 or type(end) is not int or
                        not self.index <= end < len(self.timing.times_ms)):
                    raise ContractError('A known endpoint must be a pending open-lane obligation')
            elif opened and self.arm == Arm.O1:
                raise ContractError('Every open O1 object must have a known endpoint')
        if self.index and self.replay.last_row is not None:
            if self.replay.last_row.time_ms > self.timing.times_ms[self.index - 1]:
                raise ContractError('Replay extends beyond the candidate cursor')
        if self.finished and any(self.replay.occupancy):
            raise ContractError('A completed candidate schedule cannot retain a hold')

    @classmethod
    def from_seed(cls, arm: Arm, timing: Timing, rows: Sequence[CompleteRow],
                  crossing_ends: Mapping[int, int] | None = None):
        """Validate a physical seed; typed arms also receive its open-object ends.

        The caller owns the minimum seed size. This primitive also supports
        short synthetic seeds for exact state and context-boundary checks.
        """
        if len(rows) > len(timing.times_ms):
            raise ContractError('Seed exceeds the candidate sequence')
        replay = ExactReplayState()
        for i, row in enumerate(rows):
            if row.time_ms != timing.times_ms[i]:
                raise ContractError('Seed must cover the initial candidate prefix exactly')
            if arm != Arm.R0:
                if timing.onsets is None or bool(any(a in (1, 2) for a in row.actions)) != timing.onsets[i]:
                    raise ContractError('Seed attacks disagree with supplied onset roles')
            replay = commit(replay, row, is_terminal=i == len(timing.times_ms) - 1)
        crossing_ends = {} if crossing_ends is None else dict(crossing_ends)
        needed = {c for c, opened in enumerate(replay.occupancy) if opened} if arm != Arm.R0 else set()
        if set(crossing_ends) != needed or any(type(c) is not int for c in crossing_ends):
            raise ContractError('Typed seed needs exactly its crossing LN endpoints; R0 receives none')
        state = cls(arm, timing, len(rows), replay, tuple(crossing_ends.get(c) for c in range(4)))
        if not state.finished and not state._future_room(replay.occupancy, state.known_ends, state.index - 1):
            raise ContractError('Seed commitments block a required future onset')
        return state

    @property
    def finished(self):
        return self.index == len(self.timing.times_ms)

    @property
    def time_ms(self):
        if self.finished:
            raise ContractError('Completed timing has no current candidate')
        return self.timing.times_ms[self.index]

    def _next_onset(self, after):
        roles = self.timing.onsets
        return None if roles is None else next((i for i in range(after + 1, len(roles)) if roles[i]), None)

    def _future_room(self, occupied, ends, after):
        following = self._next_onset(after)
        if following is None or not all(occupied):
            return True
        # A rowwise LN can still release at an intervening candidate. A known
        # object cannot be released early to make room. Equal-time release is
        # insufficient because same-lane close/restart is excluded.
        return any((end is not None and end < following) or
                   (end is None and after + 1 < following) for end in ends)

    def row_possible(self, actions):
        if self.finished:
            return False
        checked_actions(actions)
        if self.arm == Arm.O1:
            raise ContractError('O1 predicts heads and endpoints, not stochastic release rows')
        if self.arm == Arm.R0 and not any(actions):
            return False
        if self.arm == Arm.R1 and bool(any(a in (1, 2) for a in actions)) != self.timing.onsets[self.index]:
            return False
        occupied, ends = list(self.replay.occupancy), list(self.known_ends)
        terminal = self.index == len(self.timing.times_ms) - 1
        for c, (opened, end, action) in enumerate(zip(occupied, ends, actions)):
            if opened:
                if action not in (0, 3) or (terminal and action != 3):
                    return False
                if end is not None and action != (3 if end == self.index else 0):
                    return False
                if action == 3:
                    occupied[c], ends[c] = False, None
            else:
                if action == 3 or (terminal and action == 2):
                    return False
                if action == 2:
                    occupied[c] = True
        return self._future_room(occupied, ends, self.index)

    def row_support(self):
        return tuple(self.row_possible(row) for row in ROW_ACTIONS)

    def _object_completion_possible(self, heads, assigned):
        occupied, ends = list(self.replay.occupancy), list(self.known_ends)
        for c in range(4):
            if ends[c] == self.index:
                occupied[c], ends[c] = False, None
            if heads[c] == 2:
                occupied[c], ends[c] = True, assigned.get(c, self.index + 1)
        return self._future_room(occupied, ends, self.index)

    def head_possible(self, heads):
        if self.finished:
            return False
        checked_actions(heads, maximum=2)
        if self.arm != Arm.O1:
            raise ContractError('Only O1 uses the complete-object head vocabulary')
        if not self.timing.onsets[self.index] or not any(heads):
            return False
        if any(opened and head for opened, head in zip(self.replay.occupancy, heads)):
            return False
        if self.index == len(self.timing.times_ms) - 1 and 2 in heads:
            return False
        return self._object_completion_possible(heads, {})

    def head_support(self):
        return tuple(self.head_possible(heads) for heads in HEAD_ACTIONS)

    def endpoint_bounds(self, heads, assigned: Mapping[int, int], lane: int):
        """Return the half-open interval of feasible R indices for one factor.

        Remaining current-row LN factors may take any later candidate. Assigning
        each remaining factor its earliest possible end tests existence of a
        completion; it does not invent a committed endpoint or a sampled label.
        """
        if not self.head_possible(heads):
            raise ContractError('Endpoint prediction requires a feasible chosen head group')
        starts = {c for c, head in enumerate(heads) if head == 2}
        if (type(lane) is not int or lane not in starts or lane in assigned or
                not set(assigned).issubset(starts)):
            raise ContractError('Endpoint factors must follow distinct chosen LN lanes')
        for c, end in assigned.items():
            if type(c) is not int or type(end) is not int or not self.index < end < len(self.timing.times_ms):
                raise ContractError('Prior endpoint factors must be strictly future candidates')
        # Earlier ends cannot make future-onset availability worse. If even the
        # last candidate permits a completion, every future candidate does.
        # Otherwise this factor must release strictly before the next onset.
        length = len(self.timing.times_ms)
        stop = length if self._object_completion_possible(heads, {**assigned, lane: length - 1}) else self._next_onset(self.index)
        if stop is None or stop <= self.index + 1:
            raise ContractError('Earlier endpoint factors leave no feasible completion')
        return self.index + 1, stop

    def endpoint_support(self, heads, assigned: Mapping[int, int], lane: int):
        """Retain every feasible future candidate, including arbitrarily late ends."""
        start, stop = self.endpoint_bounds(heads, assigned, lane)
        return tuple(start <= i < stop for i in range(len(self.timing.times_ms)))

    def forced_row(self):
        if self.arm != Arm.O1 or self.finished or self.timing.onsets[self.index]:
            raise ContractError('Only an O1 non-onset candidate has a forced release row')
        actions = tuple(3 if end == self.index else 0 for end in self.known_ends)
        return actions if any(actions) else None

    def advance(self, actions, endpoints: Mapping[int, int] | None = None):
        """Commit one decision transaction; an absent event has no physical row."""
        endpoints = {} if endpoints is None else dict(endpoints)
        actions = (0, 0, 0, 0) if actions is None else checked_actions(actions)
        if self.finished:
            raise ContractError('Cannot advance completed timing')
        if self.arm == Arm.O1:
            heads = tuple(a if a in (1, 2) else 0 for a in actions)
            if self.timing.onsets[self.index]:
                if not self.head_possible(heads):
                    raise ContractError('Illegal O1 onset group')
            elif any(heads):
                raise ContractError('A non-onset candidate cannot introduce heads')
            expected = tuple(3 if end == self.index else head for end, head in zip(self.known_ends, heads))
            if actions != expected:
                raise ContractError('O1 releases must execute exactly the previous plans')
            starts = {c for c, action in enumerate(heads) if action == 2}
            if set(endpoints) != starts or any(type(c) is not int for c in endpoints):
                raise ContractError('Each new O1 LN needs its own endpoint')
            if any(type(end) is not int or not self.index < end < len(self.timing.times_ms) for end in endpoints.values()):
                raise ContractError('LN endpoints must be strictly future R candidates')
            if not self._object_completion_possible(heads, endpoints):
                raise ContractError('Object endpoints block a required future onset')
        else:
            if endpoints or not self.row_possible(actions):
                raise ContractError('Illegal row decision or endpoint supplied to a row actor')
        ends = list(self.known_ends)
        for c, action in enumerate(actions):
            if action == 3:
                ends[c] = None
            elif action == 2:
                ends[c] = endpoints.get(c)
        row = CompleteRow(self.time_ms, actions) if any(actions) else None
        replay = self.replay if row is None else commit(self.replay, row, is_terminal=self.index == len(self.timing.times_ms) - 1)
        following = replace(self, index=self.index + 1, replay=replay, known_ends=tuple(ends))
        return following, row
