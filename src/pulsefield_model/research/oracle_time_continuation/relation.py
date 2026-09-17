"""Bounded, candidate-independent relation frontiers with unique complete-row nodes."""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from ..scoped_style_modeling.replay import HAND_COLUMNS
from ..source_action_modeling.actions import ATTACK_ACTIONS, LN_CLOSE, LN_START
from .attention import MemoryAttention
from .config import BackboneConfig
from .features import TIME_DIM, action_features, clock_features, relative_lanes
from .schema import CompleteRow

EMPTY_INDICES = ((), (), (), ())
NO_HEADS = (None, None, None, None)


@dataclass(frozen=True)
class RelationNode:
    row_id: int
    row: CompleteRow
    payload: Tensor
    lane_predecessors: tuple[int | None, ...]
    hand_predecessors: tuple[int | None, ...]
    closed_heads: tuple[int | None, ...]
    lane_intervals_ms: tuple[float | None, ...]
    hand_intervals_ms: tuple[float | None, ...]
    closed_durations_ms: tuple[float | None, ...]

    def detached(self) -> RelationNode:
        return replace(self, payload=self.payload.detach().clone())


@dataclass(frozen=True)
class RelationState:
    attacks: tuple[tuple[int, ...], ...] = EMPTY_INDICES
    releases: tuple[tuple[int, ...], ...] = EMPTY_INDICES
    active_heads: tuple[int | None, ...] = NO_HEADS
    nodes: tuple[RelationNode, ...] = ()

    @property
    def visible_ids(self) -> tuple[int, ...]:
        return tuple(node.row_id for node in self.nodes)

    def detached(self) -> RelationState:
        return replace(self, nodes=tuple(node.detached() for node in self.nodes))


class RelationEncoder(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        # Node age, four lane predecessor intervals, two hand intervals,
        # four completed LN durations, plus role/rank/pin/chord facts.
        self.attention = MemoryAttention(config.hidden, config.heads, 11 * TIME_DIM + 39)
        self.bos = nn.Parameter(torch.zeros(config.hidden))

    def commit(self, state: RelationState, row: CompleteRow, row_id: int, payload: Tensor) -> RelationState:
        nodes = {node.row_id: node for node in state.nodes}
        lane_previous = tuple(ids[-1] if ids else None for ids in state.attacks)
        hand_previous = tuple(max((lane_previous[lane] for lane in lanes if lane_previous[lane] is not None),
                                  default=None) for lanes in HAND_COLUMNS)
        closed_heads = tuple(state.active_heads[lane] if action == LN_CLOSE else None
                             for lane, action in enumerate(row.actions))

        def elapsed(ids):
            return tuple(None if index is None else row.time_ms - nodes[index].row.time_ms for index in ids)

        node = RelationNode(row_id, row, payload.clone(), lane_previous, hand_previous, closed_heads,
                            elapsed(lane_previous), elapsed(hand_previous), elapsed(closed_heads))
        nodes[row_id] = node
        attacks, releases, heads = [], [], list(state.active_heads)
        for lane, action in enumerate(row.actions):
            attacks.append((state.attacks[lane] + ((row_id,) if action in ATTACK_ACTIONS else ()))
                           [-self.config.attacks_per_lane:])
            releases.append((state.releases[lane] + ((row_id,) if action == LN_CLOSE else ()))
                            [-self.config.releases_per_lane:])
            if action == LN_START:
                heads[lane] = row_id
            elif action == LN_CLOSE:
                heads[lane] = None
        retained = {index for ids in (*attacks, *releases) for index in ids}
        retained.update(index for index in heads if index is not None)
        return RelationState(tuple(attacks), tuple(releases), tuple(heads),
                             tuple(nodes[index] for index in sorted(retained)))

    def edge_features(self, state: RelationState, time_ms: float, row_id: int, like: Tensor) -> Tensor:
        ranks = [{index: rank + 1 for rank, index in enumerate(reversed(ids))}
                 for ids in (*state.attacks, *state.releases)]
        hand_latest = [max((state.attacks[lane][-1] for lane in lanes if state.attacks[lane]), default=None)
                       for lanes in HAND_COLUMNS]
        edges = []
        for node in state.nodes:
            actions = action_features(node.row.actions, like)
            hands = []
            for hand in range(2):
                lanes = relative_lanes(hand)
                times = [time_ms - node.row.time_ms]
                times += [node.lane_intervals_ms[lane] for lane in lanes]
                times += [node.hand_intervals_ms[side] for side in (hand, 1 - hand)]
                times += [node.closed_durations_ms[lane] for lane in lanes]
                tags = [value for lane in lanes for value in
                        (ranks[lane].get(node.row_id, 0) / self.config.attacks_per_lane,
                         ranks[4 + lane].get(node.row_id, 0) / self.config.releases_per_lane,
                         state.active_heads[lane] == node.row_id,
                         node.closed_heads[lane] is not None,
                         node.lane_predecessors[lane] is not None)]
                tags += [hand_latest[side] == node.row_id for side in (hand, 1 - hand)]
                tags += [(row_id - node.row_id) / 32]
                hands.append(torch.cat((clock_features(times, like).flatten(), actions[hand],
                                        like.new_tensor(tags))))
            edges.append(torch.stack(hands))
        return torch.stack(edges)

    def forward(self, frontier: Tensor, state: RelationState, time_ms: float, row_id: int) -> Tensor:
        if state.nodes:
            payload = torch.stack([node.payload for node in state.nodes])
            edges = self.edge_features(state, time_ms, row_id, frontier)
        else:
            payload = self.bos.expand(1, 2, -1)
            edges = frontier.new_zeros(1, 2, 11 * TIME_DIM + 39)
        visible = torch.ones(1, len(payload), dtype=torch.bool, device=frontier.device)
        return self.attention.read(frontier[None], self.attention.project(payload), edges[None], visible)[0]
