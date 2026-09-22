"""Tensorize replay facts separately from concept queries and supervision.

Time features use signed log1p seconds. Source identities are used only while
replaying exact selector history; they never enter learned feature tensors.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from .dataset import ContractError
from .relations import Edge
from .replay import HAND_COLUMNS, PreparedChart, selected_objects, time_feature

RELATION_KINDS = ("self", "simultaneous", "event_succession", "attack_succession_1",
                  "attack_succession_2", "same_lane_recurrence", "ln_identity", "occupied_role")
LANE_DIM = 25
ROW_DIM = 11
EDGE_DIM = 58
RELATION_DIM = 28
HISTORY_DIM = 6


def elapsed(ms: float) -> float:
    return time_feature(ms)[0]


def bits(mask: int, width: int = 4) -> list[int]:
    return [(mask >> i) & 1 for i in range(width)]


def onehot(value, choices) -> list[int]:
    if value not in choices:
        raise ContractError(f"Unknown feature category {value!r}")
    return [int(value == x) for x in choices]


@dataclass(frozen=True)
class ChartTensors:
    lanes: Tensor
    rows: Tensor
    lengths: Tensor
    edge_index: Tensor
    edge_features: Tensor
    relation_edges: Tensor
    relation_features: Tensor
    section_indices: Tensor
    section_lengths: Tensor
    section_features: Tensor
    section_events: Tensor
    duration: Tensor

    def to(self, device) -> ChartTensors:
        # Packed sequence lengths are intentionally kept on the CPU.
        return ChartTensors(**{f.name: getattr(self, f.name) if f.name.endswith("lengths")
                              else getattr(self, f.name).to(device) for f in fields(self)})


@dataclass(frozen=True)
class SelectorTensors:
    indices: Tensor
    valid_masks: Tensor
    steps: Tensor
    times: Tensor
    history: Tensor
    targets: Tensor
    available: Tensor

    def to(self, device) -> SelectorTensors:
        return SelectorTensors(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})


@dataclass(frozen=True)
class Example:
    chart: PreparedChart
    edges: tuple[Edge, ...]
    concept: int
    assessment: int
    masks: tuple[int, ...] | None


@dataclass(frozen=True)
class Batch:
    chart: ChartTensors
    concepts: Tensor
    assessments: Tensor
    selector: SelectorTensors

    def to(self, device) -> Batch:
        return Batch(self.chart.to(device), self.concepts.to(device),
                     self.assessments.to(device), self.selector.to(device))


def exact_history(chart: PreparedChart, masks: tuple[int, ...]) -> list:
    """Return facts before each decision in [hand, outer/inner, feature] order.

    Boundary selection observes an entering object without inventing an attack.
    A context-only predecessor is unavailable, not an observed unselected head.
    Releases remove occupation after their decision, including forced skips.
    """
    selected_objects(chart, masks)
    observed = {}
    latest = [None] * 4
    previous = [None] * 4
    result = []
    for decision, mask in zip(chart.decisions, masks):
        row = chart.inputs.rows[decision.encoder_index]
        lanes = []
        for lane in range(4):
            active = next((n for n in chart.visible_objects if n.column == lane and
                           n.kind == "long" and n.start_ms < row.time_ms <= n.end_ms), None)
            lanes.append([*time_feature(None if latest[lane] is None else row.time_ms-latest[lane]),
                          int(bool(previous[lane])), previous[lane] is not None,
                          int(observed.get(active, False)), active in observed])
        result.append([[lanes[j] for j in columns] for columns in HAND_COLUMNS])
        for lane, note in enumerate(decision.candidates):
            if note is not None:
                chosen = bool(mask & (1 << lane))
                observed[note] = chosen
                previous[lane] = chosen
                if row.phase == "source":
                    if chosen:
                        latest[lane] = note.start_ms
    return result


def _chart_features(chart):
    lanes, rows, section = [], [], []
    for row in chart.inputs.rows:
        lane_values = []
        for lane in row.lanes:
            values = [lane.tap, lane.ln_start, lane.ln_close, lane.occupied_before, lane.occupied_after]
            for key in ("previous_attack_ms", "next_attack_ms", "before_age_ms", "before_remaining_ms",
                        "before_duration_ms", "after_age_ms", "after_remaining_ms", "after_duration_ms"):
                values.extend(time_feature(getattr(lane, key)))
            values.extend([lane.before_head_visible, lane.before_close_visible,
                           lane.after_head_visible, lane.after_close_visible])
            lane_values.append(values)
        lanes.append([[lane_values[j] for j in columns] for columns in HAND_COLUMNS])
        rows.append([elapsed(row.elapsed_ms), elapsed(row.scope_relative_ms), elapsed(row.context_relative_ms),
                     elapsed(chart.inputs.scope.end_ms-row.time_ms), elapsed(chart.inputs.context.end_ms-row.time_ms),
                     row.phase == "source", row.in_scope,
                     *[name in row.markers for name in ("context_start", "section_start", "section_end", "context_end")]])
    for index, dt in zip(chart.inputs.section_indices, chart.inputs.readout_elapsed_ms):
        row = chart.inputs.rows[index]
        section.append([elapsed(dt), elapsed(row.scope_relative_ms), row.phase == "source"])
    return torch.tensor(lanes, dtype=torch.float32), torch.tensor(rows, dtype=torch.float32), torch.tensor(section, dtype=torch.float32)


def _edge_features(edge):
    values = [elapsed(edge.elapsed_ms), edge.query_phase == "source", edge.neighbor_phase == "source"]
    for key in ("query_attack_group", "neighbor_attack_group", "role_intersection",
                "query_occupied_before", "query_occupied_after", "neighbor_occupied_before", "neighbor_occupied_after"):
        values.extend(bits(getattr(edge, key)))
    for actions in (edge.query_actions, edge.neighbor_actions):
        for action in actions:
            values.extend(bits(action, 3))
    # Same/other hand is a relative coordinate, independent of absolute side.
    values.extend([edge.query % 2 == edge.neighbor % 2,
                   edge.query_phase == "boundary", edge.neighbor_phase == "boundary"])
    return values


def _relation_features(r):
    return [*onehot(r.kind, RELATION_KINDS), *onehot(r.query_role, (-1, 0, 1, 2, 3)),
            *onehot(r.neighbor_role, (-1, 0, 1, 2, 3)),
            elapsed(max(0, r.intervening_attack_rows) * 1000), r.intervening_attack_rows >= 0,
            r.occupied_before, r.occupied_after, elapsed(r.duration_ms), elapsed(r.head_delta_ms),
            elapsed(r.close_delta_ms), *onehot(r.endpoint, ("", "head", "close"))]


def collate(examples: list[Example]) -> Batch:
    """Validate all masks before availability masking and construct a padded batch."""
    if not examples:
        raise ContractError("Cannot collate an empty batch")
    if any(e.concept not in range(5) or e.assessment not in range(3) for e in examples):
        raise ContractError("Expected a resolved assessment and frozen concept")
    lengths = torch.tensor([len(e.chart.inputs.rows) for e in examples])
    width = int(lengths.max())
    chart_values = [_chart_features(e.chart) for e in examples]
    edge_indices, edge_values, relation_edges, relation_values = [], [], [], []
    section_indices, events, decision_indices, valid, times, histories, targets = [], [], [], [], [], [], []
    for b, example in enumerate(examples):
        chart = example.chart
        for edge in example.edges:
            edge_indices.append([b*width*2+edge.query, b*width*2+edge.neighbor])
            edge_values.append(_edge_features(edge))
            for relation in edge.relations:
                relation_edges.append(len(edge_values)-1)
                relation_values.append(_relation_features(relation))
        section_indices.append(torch.tensor(chart.inputs.section_indices))
        events.append(torch.tensor([chart.inputs.rows[i].in_scope for i in chart.inputs.section_indices]))
        masks = example.masks if example.masks is not None else (0,) * len(chart.decisions)
        selected_objects(chart, masks)
        decision_indices.append(torch.tensor([d.encoder_index for d in chart.decisions]))
        valid.append(torch.tensor([[m in d.valid_masks for m in range(16)] for d in chart.decisions]))
        decision_times = [chart.inputs.rows[d.encoder_index].time_ms for d in chart.decisions]
        times.append(torch.tensor([0.0]+[elapsed(t-s) for s, t in zip(decision_times, decision_times[1:])]))
        histories.append(torch.tensor(exact_history(chart, masks), dtype=torch.float32))
        targets.append(torch.tensor(masks))
    def pad(values, **kwargs):
        return pad_sequence(values, batch_first=True, **kwargs)
    decision_lengths = torch.tensor([len(e.chart.decisions) for e in examples])
    steps = torch.arange(int(decision_lengths.max()))[None] < decision_lengths[:, None]
    valid_masks = pad(valid)
    valid_masks[:, :, 0] = True  # Padding has a harmless forced skip and no state update.
    chart = ChartTensors(pad([v[0] for v in chart_values]), pad([v[1] for v in chart_values]), lengths,
                         torch.tensor(edge_indices).T, torch.tensor(edge_values, dtype=torch.float32),
                         torch.tensor(relation_edges), torch.tensor(relation_values, dtype=torch.float32),
                         pad(section_indices), torch.tensor([len(v) for v in section_indices]),
                         pad([v[2] for v in chart_values]), pad(events),
                         torch.tensor([[elapsed(e.chart.inputs.scope.end_ms-e.chart.inputs.scope.start_ms)] for e in examples]))
    selector = SelectorTensors(pad(decision_indices), valid_masks, steps, pad(times), pad(histories), pad(targets),
                               torch.tensor([e.masks is not None for e in examples]))
    return Batch(chart, torch.tensor([e.concept for e in examples]), torch.tensor([e.assessment for e in examples]), selector)
