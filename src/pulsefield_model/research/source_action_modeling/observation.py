"""Separate supplied observations from immutable complete replay and targets."""
from __future__ import annotations

from dataclasses import dataclass

from ..scoped_style_modeling.dataset import ContractError, Interval
from ..scoped_style_modeling.replay import PreparedChart

# Bit fields preserve a close simultaneous with either kind of new head.
LANE_ACTIONS = (0, 1, 2, 4, 5, 6)
BLOCK_SIZES = (4, 16, 64)
INPUT_CONTRACT = "source-action-partial-v1"


@dataclass(frozen=True)
class EventBlock:
    start: int
    size: int

    def __post_init__(self):
        if type(self.start) is not int or type(self.size) is not int or self.start < 0 or self.size < 1:
            raise ContractError("A target block requires a nonnegative event index and positive size")


@dataclass(frozen=True)
class ObservedRow:
    time_ms: float
    phase: str
    markers: tuple[str, ...]
    # None is an unknown four-lane action row; (0, 0, 0, 0) is observed silence.
    actions: tuple[int, ...] | None


@dataclass(frozen=True)
class PartialObservation:
    scope: Interval
    context: Interval
    rows: tuple[ObservedRow, ...]
    target_indices: tuple[int, ...]
    # State immediately before all actions at context.start, with no LN identity.
    entering_occupancy: tuple[bool | None, ...]

    def __post_init__(self):
        if len(self.entering_occupancy) != 4 or any(
            v is not None and type(v) is not bool for v in self.entering_occupancy
        ):
            raise ContractError("Entering occupancy requires four bool-or-unknown values")
        source = [i for i, row in enumerate(self.rows) if row.phase == "source"]
        if not self.target_indices or not set(self.target_indices) <= set(source):
            raise ContractError("Targets must be nonempty real source-event rows")
        start = source.index(self.target_indices[0])
        if tuple(source[start:start + len(self.target_indices)]) != self.target_indices:
            raise ContractError("Targets must form a contiguous source-event block")
        for i, row in enumerate(self.rows):
            if row.phase not in ("source", "boundary"):
                raise ContractError("Unknown observation phase")
            if (row.actions is None) != (i in self.target_indices):
                raise ContractError("Only target rows have unknown actions")
            if row.actions is not None and (len(row.actions) != 4 or any(a not in LANE_ACTIONS for a in row.actions)):
                raise ContractError("Invalid four-lane source actions")
            if row.phase == "boundary" and row.actions != (0, 0, 0, 0):
                raise ContractError("Synthetic boundaries carry no source actions")


@dataclass(frozen=True)
class BlockExample:
    observation: PartialObservation
    targets: tuple[tuple[int, ...], ...]
    # Analysis only; collate() never puts this target-derived count in inputs.
    attack_group_span: int


def declared_entering_occupancy(chart: PreparedChart) -> tuple[bool, ...]:
    """Extract the explicitly supplied context-entry condition, without endpoints."""
    t = chart.inputs.context.start_ms
    return tuple(any(n.column == lane and n.kind == "long" and n.start_ms < t <= n.end_ms
                     for n in chart.visible_objects) for lane in range(4))


def observe(chart: PreparedChart, block: EventBlock, *,
            entering_occupancy: tuple[bool | None, ...] = (None,) * 4) -> BlockExample:
    """Hide complete event rows before deriving any features or relations.

    The caller declares context-entry occupation explicitly. Omission leaves it
    unknown. Complete replay descriptors, identities and post-block occupation
    are never copied. The complete chart remains available only to its owner.
    """
    source = [i for i, row in enumerate(chart.inputs.rows) if row.phase == "source"]
    if block.start + block.size > len(source):
        raise ContractError("Target block exceeds the supplied source-event skeleton")
    indices = tuple(source[block.start:block.start + block.size])
    targets, rows = [], []
    for i, row in enumerate(chart.inputs.rows):
        actions = tuple(int(l.tap) | (int(l.ln_start) << 1) | (int(l.ln_close) << 2) for l in row.lanes)
        if i in indices:
            targets.append(actions)
        rows.append(ObservedRow(row.time_ms, row.phase, row.markers, None if i in indices else actions))
    observation = PartialObservation(chart.inputs.scope, chart.inputs.context, tuple(rows), indices, entering_occupancy)
    return BlockExample(observation, tuple(targets), sum(any(a & 3 for a in row) for row in targets))


def advance_occupancy(state: tuple[bool | None, ...], actions: tuple[int, ...] | None) -> tuple[bool | None, ...]:
    """Propagate only visible facts; hidden rows invalidate every lane's state."""
    if actions is None:
        return (None,) * 4
    result = []
    for occupied, action in zip(state, actions):
        close, attack = bool(action & 4), bool(action & 3)
        if action not in LANE_ACTIONS or (close and occupied is False) or (attack and not close and occupied is True):
            raise ContractError("Source action contradicts declared or observed lane occupation")
        result.append(bool(action & 2) if close or attack else occupied)
    return tuple(result)


def visible_states(observation: PartialObservation):
    """Return before/after occupation plus the decoder's prefix-only entry state."""
    state = observation.entering_occupancy
    before, after = [], []
    for row in observation.rows:
        before.append(state)
        if row.phase == "source":
            state = advance_occupancy(state, row.actions)
        after.append(state)
    return tuple(before), tuple(after), before[observation.target_indices[0]]
