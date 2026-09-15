"""Scheduler/replay boundary shared by teacher forcing and future sampling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..scoped_style_modeling.dataset import ContractError
from .replay import ExactReplayState, ReplayClocks, commit, legal_rows
from .schema import Actions, CompleteRow, TimeSkeleton, checked_time


@dataclass(frozen=True)
class PredictionInput:
    """The model's entire exact input: current time, terminal condition, history.

    Target rows, full skeleton, source identity and execution position stay in
    their owners. Reading this object cannot consume a row or advance a cache.
    """

    time_ms: float
    is_terminal: bool
    history: ExactReplayState

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_ms", checked_time(self.time_ms))
        if type(self.is_terminal) is not bool:
            raise ContractError("The true skeleton terminal flag must be a boolean")
        if self.history.is_complete or (self.history.last_row is not None and
                                        self.time_ms <= self.history.last_row.time_ms):
            raise ContractError("Prediction time must follow an unfinished committed prefix")

    @property
    def clocks(self) -> ReplayClocks:
        return self.history.clocks_at(self.time_ms)

    @property
    def legal_actions(self) -> tuple[Actions, ...]:
        return legal_rows(self.history, is_terminal=self.is_terminal)


@dataclass(frozen=True)
class ContinuationState:
    """Execution state; pass query(), rather than this object, to a predictor."""

    skeleton: TimeSkeleton
    replay: ExactReplayState = ExactReplayState()

    def __post_init__(self) -> None:
        position = self.next_index
        if position > len(self.skeleton.times_ms):
            raise ContractError("Replay position exceeds the skeleton")
        if position and (
            self.replay.first_time_ms != self.skeleton.times_ms[0]
            or self.replay.last_row.time_ms != self.skeleton.times_ms[position - 1]
            or self.replay.is_complete != self.finished
        ):
            raise ContractError("Replay boundary differs from its skeleton position or terminal condition")

    @property
    def next_index(self) -> int:
        return self.replay.row_count

    @property
    def finished(self) -> bool:
        return self.next_index == len(self.skeleton.times_ms)

    def query(self) -> PredictionInput:
        """Return immutable pre-row input, rejecting an exhausted skeleton."""
        if self.finished:
            raise ContractError("The skeleton is exhausted; there is no next row")
        return PredictionInput(self.skeleton.times_ms[self.next_index],
                               self.next_index == len(self.skeleton.times_ms) - 1, self.replay)

    def commit(self, row: CompleteRow) -> ContinuationState:
        """Commit one true or sampled row at exactly the next skeleton time."""
        query = self.query()
        if row.time_ms != query.time_ms:
            raise ContractError("Commit must match the next skeleton time")
        return ContinuationState(self.skeleton, commit(self.replay, row, is_terminal=query.is_terminal))


def prefill(skeleton: TimeSkeleton, prefix: Iterable[CompleteRow]) -> ContinuationState:
    """Replay from the true beginning, including release rows; never reset at a window.

    The prefix must cover consecutive skeleton positions starting at zero. A
    resource or sample horizon has no terminal semantics. An empty prefix is BOS.
    """
    state = ContinuationState(skeleton)
    for row in prefix:
        state = state.commit(row)
    return state
