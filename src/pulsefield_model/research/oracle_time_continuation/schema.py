"""Complete V3 rows and the scheduler-owned attack/release time skeleton."""
from __future__ import annotations

from dataclasses import dataclass
import math

from ..scoped_style_modeling.dataset import ContractError
from ..source_action_modeling.actions import EMPTY, LANE_ACTIONS

Actions = tuple[int, int, int, int]


def checked_time(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ContractError("Row time must be a finite, nonnegative number of milliseconds")
    return float(value)


@dataclass(frozen=True)
class CompleteRow:
    """One materialized simultaneous row; unknown and padding are not rows."""

    time_ms: float
    actions: Actions

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_ms", checked_time(self.time_ms))
        if not isinstance(self.actions, (tuple, list)) or len(self.actions) != 4 or any(
            type(action) is not int or action not in LANE_ACTIONS for action in self.actions
        ):
            raise ContractError("A complete row requires four known V3 lane actions")
        if all(action == EMPTY for action in self.actions):
            raise ContractError("A materialized row must be nonempty, including release-only rows")
        object.__setattr__(self, "actions", tuple(self.actions))


@dataclass(frozen=True)
class TimeSkeleton:
    """Scheduler data; only the current time and true terminal flag reach a query.

    Source admission merges simultaneous events before constructing this type.
    Direct callers must supply strictly increasing times, without padding.
    """

    times_ms: tuple[float, ...]

    def __post_init__(self) -> None:
        times = tuple(checked_time(t) for t in self.times_ms)
        if any(a >= b for a, b in zip(times, times[1:])):
            raise ContractError("Skeleton times must be strictly increasing and deduplicated")
        object.__setattr__(self, "times_ms", times)
