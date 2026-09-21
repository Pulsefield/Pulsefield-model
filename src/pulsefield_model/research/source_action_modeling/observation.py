"""Separate supplied observations from immutable complete replay and targets."""
from __future__ import annotations

from dataclasses import dataclass, replace

from ..scoped_style_modeling.dataset import ContractError, Interval
from ..scoped_style_modeling.replay import PreparedChart
from .actions import ATTACK_ACTIONS, EMPTY, LANE_ACTIONS, LN_CLOSE, LN_START, TAP, source_actions

BLOCK_SIZES = (4, 16, 64)
INPUT_CONTRACT = "source-action-visibility-v3-four-actions"


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
class FarSummary:
    """Unordered non-target counts, by lane, on one side of a near neighborhood.

    Each lane stores taps, heads, closes, occupied/known before, occupied/known
    after, and source-row count. Occupation is derived after target erasure.
    """
    side: str
    lanes: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        if self.side not in ("before", "after") or len(self.lanes) != 4 or any(
            len(lane) != 8 or any(type(v) is not int or v < 0 for v in lane) for lane in self.lanes
        ):
            raise ContractError("Far summaries require a side and four lanes of eight nonnegative counts")


@dataclass(frozen=True)
class PartialObservation:
    scope: Interval
    context: Interval
    rows: tuple[ObservedRow, ...]
    target_indices: tuple[int, ...]
    # State immediately before all actions at context.start, with no LN identity.
    entering_occupancy: tuple[bool | None, ...]
    summaries: tuple[FarSummary, ...] = ()

    def __post_init__(self):
        if len(self.entering_occupancy) != 4 or any(
            v is not None and type(v) is not bool for v in self.entering_occupancy
        ):
            raise ContractError("Entering occupancy requires four bool-or-unknown values")
        source = [i for i, row in enumerate(self.rows) if row.phase == "source"]
        if not self.rows or not set(self.target_indices) <= set(source):
            raise ContractError("Targets must be real source-event rows in a nonempty timeline")
        if self.target_indices:
            start = source.index(self.target_indices[0])
            if tuple(source[start:start + len(self.target_indices)]) != self.target_indices:
                raise ContractError("Targets must form a contiguous source-event block")
        if len({s.side for s in self.summaries}) != len(self.summaries):
            raise ContractError("Duplicate far-summary side")
        for i, row in enumerate(self.rows):
            if row.phase not in ("source", "boundary"):
                raise ContractError("Unknown observation phase")
            if i in self.target_indices and row.actions is not None:
                raise ContractError("Prediction targets must have unknown actions")
            if row.actions is not None and (len(row.actions) != 4 or any(
                type(a) is not int or a not in LANE_ACTIONS for a in row.actions
            )):
                raise ContractError("Invalid four-lane source actions")
            if row.phase == "source" and row.actions == (EMPTY,) * 4:
                raise ContractError("Real source events require a nonempty action row")
            if row.phase == "boundary" and row.actions != (0, 0, 0, 0):
                raise ContractError("Synthetic boundaries carry no source actions")


@dataclass(frozen=True)
class BlockExample:
    observation: PartialObservation
    targets: tuple[tuple[int, ...], ...]
    # Analysis only; collate() never puts this target-derived count in inputs.
    attack_group_span: int
    # A paired view may explicitly supply the common near-prefix condition.
    query_entering_occupancy: tuple[bool | None, ...] | None = None


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
        actions = source_actions(row)
        if i in indices:
            targets.append(actions)
        rows.append(ObservedRow(row.time_ms, row.phase, row.markers, None if i in indices else actions))
    observation = PartialObservation(chart.inputs.scope, chart.inputs.context, tuple(rows), indices, entering_occupancy)
    return BlockExample(observation, tuple(targets), sum(any(a in ATTACK_ACTIONS for a in row) for row in targets))


def observe_complete(chart: PreparedChart, *,
                     entering_occupancy: tuple[bool | None, ...] = (None,) * 4) -> PartialObservation:
    """Expose the original complete scope/context without creating prediction queries."""
    rows = tuple(ObservedRow(r.time_ms, r.phase, r.markers, source_actions(r))
                 for r in chart.inputs.rows)
    return PartialObservation(chart.inputs.scope, chart.inputs.context, rows, (), entering_occupancy)


@dataclass(frozen=True)
class ViewPolicy:
    """A fixed timeline-row radius; the target block itself is always unavailable."""
    near_radius: int = 8
    summary_policy: str = "far-lane-action-and-known-occupation-counts-v1"

    def __post_init__(self):
        if type(self.near_radius) is not int or self.near_radius < 0:
            raise ContractError("Near radius must be a nonnegative timeline-row count")
        if self.summary_policy != "far-lane-action-and-known-occupation-counts-v1":
            raise ContractError("Unsupported coarse-summary policy")


def paired_views(example: BlockExample, policy: ViewPolicy = ViewPolicy()) -> dict[str, BlockExample]:
    """Pair near/detailed/coarse inputs, targets, skeleton and prefix-only legality.

    Input must have only its target block erased. Detailed and coarse receive
    identical unordered far summaries. Near receives neither far actions nor
    summaries. No complete-chart occupation or target action enters a summary.
    """
    obs = example.observation
    if not obs.target_indices or obs.summaries or any(
        row.actions is None and i not in obs.target_indices for i, row in enumerate(obs.rows)
    ):
        raise ContractError("Paired views require an unsummarized observation with only targets hidden")
    lo, hi = obs.target_indices[0] - policy.near_radius, obs.target_indices[-1] + policy.near_radius
    far = {side: [i for i, r in enumerate(obs.rows) if r.phase == "source" and predicate(i)]
           for side, predicate in (("before", lambda i: i < lo), ("after", lambda i: i > hi))}
    hidden = set(far["before"] + far["after"])
    near = replace(obs, rows=tuple(replace(r, actions=None) if i in hidden else r for i, r in enumerate(obs.rows)))
    before, after, _ = visible_states(obs)
    summaries = []
    for side, indices in far.items():
        counts = []
        for lane in range(4):
            actions = [sum(obs.rows[i].actions[lane] == action for i in indices) for action in (TAP, LN_START, LN_CLOSE)]
            counts.append((*actions, sum(before[i][lane] is True for i in indices),
                           sum(before[i][lane] is not None for i in indices),
                           sum(after[i][lane] is True for i in indices),
                           sum(after[i][lane] is not None for i in indices), len(indices)))
        summaries.append(FarSummary(side, tuple(counts)))
    entry = visible_states(near)[2]
    return {name: replace(example, observation=view, query_entering_occupancy=entry) for name, view in (
        ("near", near), ("detailed", replace(obs, summaries=tuple(summaries))),
        ("coarse", replace(near, summaries=tuple(summaries))))}


def advance_occupancy(state: tuple[bool | None, ...], actions: tuple[int, ...] | None) -> tuple[bool | None, ...]:
    """Propagate only visible facts; hidden rows invalidate every lane's state."""
    if actions is None:
        return (None,) * 4
    if len(state) != 4 or len(actions) != 4:
        raise ContractError("Occupation and actions require four lanes")
    result = []
    for occupied, action in zip(state, actions):
        close, attack = action == LN_CLOSE, action in ATTACK_ACTIONS
        if type(action) is not int or action not in LANE_ACTIONS or (close and occupied is False) or (attack and occupied is True):
            raise ContractError("Source action contradicts declared or observed lane occupation")
        result.append(action == LN_START if close or attack else occupied)
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
    entry = before[observation.target_indices[0]] if observation.target_indices else observation.entering_occupancy
    return tuple(before), tuple(after), entry
