"""Mask-aware facts and sparse relations derived exclusively from observations."""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from ..scoped_style_modeling.dataset import ContractError
from ..scoped_style_modeling.replay import HAND_COLUMNS, hand_role, time_feature
from .observation import BlockExample, LANE_ACTIONS, PartialObservation, visible_states

LANE_DIM, ROW_DIM, EDGE_DIM = 12, 11, 4
RELATIONS = ("self", "simultaneous", "event", "attack_1", "attack_2", "recurrence", "ln_identity")
RELATION_DIM = len(RELATIONS) + 10
ROW_CLASSES = len(LANE_ACTIONS) ** 4


def row_token(actions: tuple[int, ...]) -> int:
    if len(actions) != 4 or any(a not in LANE_ACTIONS for a in actions):
        raise ContractError("Invalid joint action row")
    a = [LANE_ACTIONS.index(v) for v in actions]
    return (a[0] * 6 + a[1]) * 36 + a[3] * 6 + a[2]


def action_table() -> Tensor:
    """All joint rows in [class, hand, outer/inner] coordinates; mirror swaps hands."""
    return torch.tensor([[[a, b], [c, d]] for a in LANE_ACTIONS for b in LANE_ACTIONS
                         for c in LANE_ACTIONS for d in LANE_ACTIONS])


def _known(value):
    return [int(bool(value)), value is not None]


def _attack_intervals(observation, reverse=False):
    latest = [None] * 4
    result = [None] * len(observation.rows)
    indices = range(len(observation.rows) - 1, -1, -1) if reverse else range(len(observation.rows))
    for i in indices:
        row = observation.rows[i]
        if row.actions is None:
            latest = [None] * 4
            result[i] = [(0.0, False)] * 4
            continue
        result[i] = [time_feature(None if t is None else abs(row.time_ms - t)) for t in latest]
        if row.phase == "source":
            for lane, action in enumerate(row.actions):
                if action & 3:
                    latest[lane] = row.time_ms
    return result


def observation_features(observation: PartialObservation):
    before, after, entry = visible_states(observation)
    previous, following = _attack_intervals(observation), _attack_intervals(observation, reverse=True)
    lanes, rows = [], []
    for i, row in enumerate(observation.rows):
        lane_values = []
        for lane in range(4):
            action = 0 if row.actions is None else row.actions[lane]
            lane_values.append([*[bool(action & bit) for bit in (1, 2, 4)], row.actions is not None,
                                *_known(before[i][lane]), *_known(after[i][lane]),
                                *previous[i][lane], *following[i][lane]])
        lanes.append([[lane_values[lane] for lane in columns] for columns in HAND_COLUMNS])
        times = (0 if i == 0 else row.time_ms - observation.rows[i - 1].time_ms,
                 row.time_ms - observation.scope.start_ms, row.time_ms - observation.context.start_ms,
                 observation.scope.end_ms - row.time_ms, observation.context.end_ms - row.time_ms)
        rows.append([*[time_feature(t)[0] for t in times], row.phase == "source",
                     row.phase == "source" and observation.scope.contains(row.time_ms),
                     *[name in row.markers for name in ("context_start", "section_start", "section_end", "context_end")]])
    occupation = [[-1 if entry[lane] is None else int(entry[lane]) for lane in columns] for columns in HAND_COLUMNS]
    return torch.tensor(lanes, dtype=torch.float32), torch.tensor(rows, dtype=torch.float32), torch.tensor(occupation)


def observation_relations(observation: PartialObservation):
    """Known succession and LN identity never bridge an unknown source row.

    Event adjacency and simultaneous hand nodes use only the supplied skeleton.
    An entering hold has no observable identity endpoint. LN edges require both
    visible endpoints and an uninterrupted visible interval between them.
    """
    pairs = {}

    def add(q, n, kind, role=-1):
        pairs.setdefault((q, n), set()).add((kind, role, role))

    def connect(a, b, kind):
        for h in range(2):
            for k in range(2):
                add(2 * a + h, 2 * b + k, kind)
                add(2 * b + k, 2 * a + h, kind)

    attacks, lane_previous, holds = [], [None] * 4, [None] * 4
    last_source = None
    for i, row in enumerate(observation.rows):
        for h in range(2):
            add(2 * i + h, 2 * i + h, "self")
            add(2 * i + h, 2 * i + 1 - h, "simultaneous")
        if i:
            connect(i - 1, i, "event")
        if row.phase != "source":
            continue
        if last_source is not None:
            connect(last_source, i, "event")
        last_source = i
        if row.actions is None:
            attacks, lane_previous, holds = [], [None] * 4, [None] * 4
            continue
        if any(a & 3 for a in row.actions):
            for offset in (1, 2):
                if len(attacks) >= offset:
                    connect(attacks[-offset], i, f"attack_{offset}")
            attacks.append(i)
        for lane, action in enumerate(row.actions):
            h, role = hand_role(lane)
            if action & 4:
                if holds[lane] is not None:
                    for a, b in ((holds[lane], i), (i, holds[lane])):
                        add(2 * a + h, 2 * b + h, "ln_identity", role)
                holds[lane] = None
            if action & 3:
                if lane_previous[lane] is not None:
                    for a, b in ((lane_previous[lane], i), (i, lane_previous[lane])):
                        add(2 * a + h, 2 * b + h, "recurrence", role)
                lane_previous[lane] = i
            if action & 2:
                holds[lane] = i
    return tuple((q, n, tuple(sorted(kinds))) for (q, n), kinds in sorted(pairs.items()))


@dataclass(frozen=True)
class ObservationTensors:
    lanes: Tensor
    rows: Tensor
    lengths: Tensor
    edge_index: Tensor
    edge_features: Tensor
    relation_edges: Tensor
    relation_features: Tensor

    def to(self, device):
        return ObservationTensors(**{f.name: getattr(self, f.name) if f.name == "lengths" else
                                     getattr(self, f.name).to(device) for f in fields(self)})


@dataclass(frozen=True)
class BlockQueries:
    indices: Tensor
    steps: Tensor
    entering_occupancy: Tensor

    def to(self, device):
        return BlockQueries(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})


@dataclass(frozen=True)
class BlockBatch:
    observation: ObservationTensors
    queries: BlockQueries
    targets: Tensor

    def to(self, device):
        return BlockBatch(self.observation.to(device), self.queries.to(device), self.targets.to(device))


def collate(examples: list[BlockExample]) -> BlockBatch:
    """Keep encoder tensors, decoder conditions and target tokens separate."""
    if not examples:
        raise ContractError("Cannot collate an empty block batch")
    lengths = torch.tensor([len(e.observation.rows) for e in examples])
    width = int(lengths.max())
    values = [observation_features(e.observation) for e in examples]
    edges, descriptors, relation_edges, relations, indices, targets = [], [], [], [], [], []
    for b, example in enumerate(examples):
        obs = example.observation
        if len(example.targets) != len(obs.target_indices):
            raise ContractError("Target length differs from observation block")
        for q, n, kinds in observation_relations(obs):
            edges.append([b * width * 2 + q, b * width * 2 + n])
            a, z = obs.rows[q // 2], obs.rows[n // 2]
            descriptors.append([time_feature(z.time_ms - a.time_ms)[0], q % 2 == n % 2,
                                a.phase == "source", z.phase == "source"])
            for kind, qr, nr in kinds:
                relation_edges.append(len(edges) - 1)
                relations.append([*[kind == k for k in RELATIONS], *[qr == r for r in (-1, 0, 1, 2, 3)],
                                  *[nr == r for r in (-1, 0, 1, 2, 3)]])
        indices.append(torch.tensor(obs.target_indices))
        targets.append(torch.tensor([row_token(row) for row in example.targets]))
    def pad(items):
        return pad_sequence(items, batch_first=True)
    steps = torch.arange(max(len(t) for t in targets))[None] < torch.tensor([len(t) for t in targets])[:, None]
    observations = ObservationTensors(pad([v[0] for v in values]), pad([v[1] for v in values]), lengths,
                                     torch.tensor(edges).T, torch.tensor(descriptors, dtype=torch.float32),
                                     torch.tensor(relation_edges), torch.tensor(relations, dtype=torch.float32))
    return BlockBatch(observations, BlockQueries(pad(indices), steps, torch.stack([v[2] for v in values])), pad(targets))
