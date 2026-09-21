"""Immutable exact replay using only complete, already committed rows.

No source object, future endpoint, window boundary or learned cache belongs
here. Elapsed clocks are evaluated at the query time without advancing rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from ..scoped_style_modeling.dataset import ContractError
from ..scoped_style_modeling.replay import HAND_COLUMNS
from ..source_action_modeling.actions import ATTACK_ACTIONS, EMPTY, LN_CLOSE, LN_START, TAP
from .schema import Actions, CompleteRow, checked_time

LaneTimes = tuple[float | None, float | None, float | None, float | None]
NO_LANE_TIMES: LaneTimes = (None, None, None, None)


@dataclass(frozen=True)
class ReplayClocks:
    """Elapsed milliseconds; None means no such committed predecessor exists."""

    previous_row_ms: float | None
    since_first_row_ms: float | None
    ln_age_ms: LaneTimes
    lane_attack_ms: LaneTimes
    lane_release_ms: LaneTimes
    hand_attack_ms: tuple[float | None, float | None]
    hand_release_ms: tuple[float | None, float | None]


@dataclass(frozen=True)
class ExactReplayState:
    """Constant-size facts from the chart's true beginning through last_row.

    Open LN start times remain exact across arbitrarily long gaps. They carry
    no close time or duration. Occupancy is derived from those starts. None in
    an action clock means that action has never occurred, not unknown history.
    """

    row_count: int = 0
    note_count: int = 0
    first_time_ms: float | None = None
    last_row: CompleteRow | None = None
    open_ln_start_ms: LaneTimes = NO_LANE_TIMES
    last_lane_attack_ms: LaneTimes = NO_LANE_TIMES
    last_lane_release_ms: LaneTimes = NO_LANE_TIMES
    is_complete: bool = False

    def __post_init__(self) -> None:
        if any(type(n) is not int or n < 0 for n in (self.row_count, self.note_count)):
            raise ContractError("Replay counts must be nonnegative integers")
        if type(self.is_complete) is not bool:
            raise ContractError("Replay completion must be a boolean")
        if self.row_count == 0:
            if self.last_row is not None or self.first_time_ms is not None or self.note_count or self.is_complete:
                raise ContractError("BOS replay has no rows, notes, clocks or terminal commit")
        elif self.last_row is None or self.first_time_ms is None or self.note_count == 0:
            raise ContractError("Nonempty replay requires its first time, last row and note count")
        elif checked_time(self.first_time_ms) > self.last_row.time_ms:
            raise ContractError("Replay first time must not follow its last row")
        for name in ("open_ln_start_ms", "last_lane_attack_ms", "last_lane_release_ms"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(values) != 4:
                raise ContractError("Replay clocks require four immutable lane values")
            for value in values:
                if value is not None:
                    checked_time(value)
                    if self.last_row is None or not self.first_time_ms <= value <= self.last_row.time_ms:
                        raise ContractError("Replay clocks must belong to the committed prefix")
        if self.is_complete and any(self.occupancy):
            raise ContractError("Completed replay cannot retain open long notes")

    @property
    def occupancy(self) -> tuple[bool, bool, bool, bool]:
        return tuple(start is not None for start in self.open_ln_start_ms)

    def clocks_at(self, time_ms: float) -> ReplayClocks:
        """Read pre/post-row or later clocks without mutation; reject time reversal."""
        time_ms = checked_time(time_ms)
        if self.last_row is not None and time_ms < self.last_row.time_ms:
            raise ContractError("Clock query precedes the committed prefix")

        def ages(values):
            return tuple(None if value is None else time_ms - value for value in values)

        def hand_ages(values):
            latest = [max((values[lane] for lane in lanes if values[lane] is not None), default=None)
                      for lanes in HAND_COLUMNS]
            return ages(latest)

        return ReplayClocks(
            None if self.last_row is None else time_ms - self.last_row.time_ms,
            None if self.first_time_ms is None else time_ms - self.first_time_ms,
            ages(self.open_ln_start_ms), ages(self.last_lane_attack_ms), ages(self.last_lane_release_ms),
            hand_ages(self.last_lane_attack_ms), hand_ages(self.last_lane_release_ms),
        )


def _lane_support(state: ExactReplayState, is_terminal: bool) -> tuple[tuple[int, ...], ...]:
    if type(is_terminal) is not bool:
        raise ContractError("The true skeleton terminal flag must be a boolean")
    if state.is_complete:
        raise ContractError("Cannot query a completed chart")
    if is_terminal:
        return tuple((LN_CLOSE,) if held else (EMPTY, TAP) for held in state.occupancy)
    return tuple((EMPTY, LN_CLOSE) if held else (EMPTY, TAP, LN_START) for held in state.occupancy)


def legal_rows(state: ExactReplayState, *, is_terminal: bool = False) -> tuple[Actions, ...]:
    """Enumerate complete legal rows; true song end closes every occupied lane.

    A closed terminal lane may TAP or EMPTY. A nonterminal EMPTY retains its
    current occupancy. There is always a legal nonempty choice before completion.
    """
    return tuple(row for row in product(*_lane_support(state, is_terminal)) if any(row))


def commit(state: ExactReplayState, row: CompleteRow, *, is_terminal: bool = False) -> ExactReplayState:
    """Validate before atomically returning post-row state; never modify state.

    Duplicate/decreasing times, illegal lane actions, and terminal closure
    violations raise ContractError. All lane transitions read the same pre-state.
    """
    if state.last_row is not None and row.time_ms <= state.last_row.time_ms:
        raise ContractError("Commit time must strictly follow the previous complete row")
    if any(action not in choices for action, choices in zip(row.actions, _lane_support(state, is_terminal))):
        raise ContractError("Row violates prefix occupancy or true terminal closure")
    starts, attacks, releases = map(list, (
        state.open_ln_start_ms, state.last_lane_attack_ms, state.last_lane_release_ms,
    ))
    for lane, action in enumerate(row.actions):
        if action in ATTACK_ACTIONS:
            attacks[lane] = row.time_ms
        if action == LN_START:
            starts[lane] = row.time_ms
        elif action == LN_CLOSE:
            starts[lane] = None
            releases[lane] = row.time_ms
    return ExactReplayState(
        row_count=state.row_count + 1,
        note_count=state.note_count + sum(action in ATTACK_ACTIONS for action in row.actions),
        first_time_ms=row.time_ms if state.first_time_ms is None else state.first_time_ms,
        last_row=row, open_ln_start_ms=tuple(starts),
        last_lane_attack_ms=tuple(attacks), last_lane_release_ms=tuple(releases),
        is_complete=is_terminal,
    )
