"""Retained source/local/relation/context states and shared position-dependent access.

Only intrinsic row actions, availability and supplied timing enter local states.
Propagated occupation, attack intervals and far summaries enter global context.
See docs/research/source_action_stage2.md for support and comparison semantics.
"""
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from ..scoped_style_modeling.config import ModelConfig as AttentionConfig
from ..scoped_style_modeling.dataset import ContractError
from ..scoped_style_modeling.model import RelationAttention, packed_gru
from .model import ModelConfig, ReferenceEncoder
from .observation import PartialObservation
from .tensors import (BlockQueries, ObservationTensors, ROW_DIM, LANE_DIM, SUMMARY_DIM,
                      EDGE_DIM, RELATION_DIM, observation_relations)

LEVELS = ("U", "L1", "L2", "L3", "R", "H")
COMPOSITION_LEVELS = ("U", "S1", "S2", "S3", "S4", "H")
LOCAL_RADII = {"U": 0, "L1": 1, "L2": 3, "L3": 7}
ARCHITECTURE = "full-row/local-1-2-4/relation/bigru-64-v1"
READER_POLICY = "shared-hand/row-level/time-relative-v1"


def valid_rows(observation: ObservationTensors) -> Tensor:
    return torch.arange(observation.rows.shape[1], device=observation.rows.device)[None] < observation.lengths.to(
        observation.rows.device)[:, None]


def level_descriptors(levels: tuple[str, ...], dim: int, reference: Tensor) -> Tensor:
    """Fixed sinusoidal identities; no level has its own trainable projection."""
    identities = {name: i for names in (LEVELS, COMPOSITION_LEVELS) for i, name in enumerate(names)}
    positions = reference.new_tensor([identities[level] for level in levels])[:, None]
    frequency = torch.exp(torch.arange(0, dim, 2, device=reference.device, dtype=reference.dtype) * (-math.log(10000) / dim))
    return torch.stack(((positions * frequency).sin(), (positions * frequency).cos()), -1).flatten(1)[:, :dim]


@dataclass(frozen=True)
class RepresentationBank:
    """Aligned [batch,row,hand,64] states; metadata contains no target action facts."""
    levels: tuple[str, ...]
    states: tuple[Tensor, ...]
    observation: ObservationTensors

    def select(self, access: str) -> tuple[tuple[str, ...], Tensor]:
        if access not in ("all", "H"):
            raise ContractError("Representation access must be all or H")
        levels = self.levels if access == "all" else ("H",)
        return levels, torch.stack([self.states[self.levels.index(level)] for level in levels], 2)

    @property
    def contextual(self) -> Tensor:
        return self.states[self.levels.index("H")]

    def detach(self):
        return RepresentationBank(self.levels, tuple(s.detach() for s in self.states), self.observation)


class LocalBlock(nn.Module):
    def __init__(self, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(64)
        self.convolution = nn.Conv1d(64, 128, 3, dilation=dilation, padding=dilation)
        self.output = nn.Linear(64, 64)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        b, t = x.shape[:2]
        # Mask after affine normalization too: its bias must not fill padding.
        values = (self.norm(x) * valid[:, :, None, None]).permute(0, 2, 3, 1).reshape(b * 2, 64, t)
        values, gates = self.convolution(values).chunk(2, 1)
        values = (values.tanh() * gates.sigmoid()).reshape(b, 2, 64, t).permute(0, 3, 1, 2)
        return (x + self.dropout(self.output(values))) * valid[:, :, None, None]


class ComposedEncoder(nn.Module):
    output_dim = 64

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.hand_hidden != 32:
            raise ContractError("The fixed composed architecture requires 32 GRU units per direction")
        self.source = nn.Sequential(nn.Linear(4 * 4 + ROW_DIM, 64), nn.GELU())
        self.local = nn.ModuleList(LocalBlock(d, config.dropout) for d in (1, 2, 4))
        attention = AttentionConfig(hand_hidden=32, attention_heads=config.attention_heads,
                                    feedforward_dim=config.feedforward_dim, dropout=config.dropout)
        self.relations = RelationAttention(attention, edge_dim=EDGE_DIM, relation_dim=RELATION_DIM)
        self.hand = nn.GRU(64 + 2 * LANE_DIM + ROW_DIM + 2 * SUMMARY_DIM, 32,
                           batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(config.dropout)

    def local_states(self, observation: ObservationTensors) -> list[Tensor]:
        valid = valid_rows(observation)
        intrinsic = observation.lanes[..., :4].flatten(-2)
        rows = observation.rows[:, :, None].expand(-1, -1, 2, -1)
        u = self.source(torch.cat((intrinsic, intrinsic.flip(2), rows), -1)) * valid[:, :, None, None]
        states = [u]
        for block in self.local:
            states.append(block(states[-1], valid))
        return states

    def forward(self, observation: ObservationTensors) -> RepresentationBank:
        valid = valid_rows(observation)
        states = self.local_states(observation)
        r = self.relations(states[-1], observation) * valid[:, :, None, None]
        h = self.contextual_state(r, observation)
        return RepresentationBank(LEVELS, (*states, r, h), observation)

    def contextual_state(self, r: Tensor, observation: ObservationTensors) -> Tensor:
        valid = valid_rows(observation)
        rows = observation.rows[:, :, None].expand(-1, -1, 2, -1)
        summary = torch.cat((observation.summaries, observation.summaries.flip(1)), -1)
        facts = torch.cat((observation.lanes.flatten(-2), rows, summary[:, None].expand(-1, rows.shape[1], -1, -1)), -1)
        b, t = r.shape[:2]
        inputs = self.dropout(torch.cat((r, facts), -1).permute(0, 2, 1, 3).reshape(b * 2, t, -1))
        h, _ = packed_gru(self.hand, inputs, observation.lengths.repeat_interleave(2))
        h = h.reshape(b, 2, t, 64).transpose(1, 2) * valid[:, :, None, None]
        return h


class ReferenceBankEncoder(nn.Module):
    """Contextual reference family with the same far-summary information budget."""
    output_dim = 64

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.hand_hidden != 32:
            raise ContractError("The fixed reference bank requires width 64")
        self.reference = ReferenceEncoder(config)
        self.summary = nn.Linear(2 * SUMMARY_DIM, 2 * config.lane_dim + ROW_DIM, bias=False)

    def forward(self, observation: ObservationTensors) -> RepresentationBank:
        summaries = self.summary(torch.cat((observation.summaries, observation.summaries.flip(1)), -1))
        h = self.reference(observation, extra_features=summaries[:, None])
        return RepresentationBank(("H",), (h,), observation)


class ActionReader(nn.Module):
    """One shared-hand attention block; queries never include a decoded action prefix."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.attention_heads
        self.query = nn.Linear(64 + ROW_DIM, 64)
        self.key = nn.Linear(64, 64)
        self.value = nn.Linear(64, 64)
        self.relative_time = nn.Linear(3, self.heads, bias=False)
        self.output = nn.Linear(64, 64)
        self.norm = nn.LayerNorm(64)

    def forward(self, bank: RepresentationBank, queries: BlockQueries, *, access: str) -> Tensor:
        levels, values = bank.select(access)
        obs = bank.observation
        b, t, count, hands, dim = values.shape
        batch = torch.arange(b, device=values.device)[:, None]
        context = bank.contextual[batch, queries.indices]
        rows = obs.rows[batch, queries.indices, None].expand(-1, -1, 2, -1)
        q = self.query(torch.cat((context, rows), -1))
        # Flatten nodes before projection: MPS LinearBackward on a five-axis
        # bank can request channels_last_3d for a gradient of the wrong rank.
        nodes = values.reshape(-1, dim)
        keys = self.key(nodes).reshape_as(values) + level_descriptors(levels, dim, values)[None, None, :, None]
        vals = self.value(nodes).reshape_as(values)
        def split(x):
            return x.reshape(b, -1, hands, self.heads, dim // self.heads).permute(0, 2, 3, 1, 4)
        scores = torch.matmul(split(q), split(keys).transpose(-2, -1)) / math.sqrt(dim // self.heads)
        dt = (obs.times_ms[:, None] - obs.times_ms[batch, queries.indices, None]) / 1000
        relative = torch.stack((dt.sign() * dt.abs().log1p(), dt.abs().log1p(), (dt == 0).to(dt.dtype)), -1)
        bias = self.relative_time(relative).permute(0, 3, 1, 2).repeat_interleave(count, -1)
        scores = scores + bias[:, None]
        mask = valid_rows(obs).repeat_interleave(count, 1)
        weights = scores.masked_fill(~mask[:, None, None, None], -torch.inf).softmax(-1)
        read = torch.matmul(weights, split(vals)).permute(0, 3, 1, 2, 4).flatten(-2)
        return self.norm(context + self.output(read)) * queries.steps[:, :, None, None]


def support_report(observation: PartialObservation, *, complete_chart=None) -> dict:
    """Report exact row supports, durations and optional realized attack spans.

    The optional aligned complete chart is analysis-only and never enters a bank
    or reader. Relation support is the union of L3 supports at incident endpoints;
    supplied timing/boundaries remain dependencies even for a one-row U state.
    """
    rows = observation.rows
    supports = {level: [set(range(max(0, i - radius), min(len(rows), i + radius + 1))) for i in range(len(rows))]
                for level, radius in LOCAL_RADII.items()}
    relation = [set() for _ in rows]
    for q, n, _ in observation_relations(observation):
        relation[q // 2].update(supports["L3"][n // 2])
    supports.update(R=relation, H=[set(range(len(rows))) for _ in rows])
    attacks = None
    if complete_chart is not None:
        source_rows = complete_chart.inputs.rows
        if (complete_chart.inputs.scope, complete_chart.inputs.context) != (observation.scope, observation.context) or (
            [(r.time_ms, r.phase, r.markers) for r in source_rows] != [(r.time_ms, r.phase, r.markers) for r in rows]
        ):
            raise ContractError("Support reporting requires the exact aligned complete chart")
        if any(r.actions is not None and r.actions != tuple(int(l.tap) | (int(l.ln_start) << 1) | (int(l.ln_close) << 2)
                                                            for l in source.lanes)
               for r, source in zip(rows, source_rows)):
            raise ContractError("Support reporting chart contradicts visible source actions")
        attacks = {i for i, r in enumerate(source_rows) if r.phase == "source" and any(l.tap or l.ln_start for l in r.lanes)}
    return {level: [{"row_indices": sorted(indices), "timeline_rows": len(indices),
                     "duration_ms": rows[max(indices)].time_ms - rows[min(indices)].time_ms,
                     "source_events": sum(rows[i].phase == "source" for i in indices),
                     "attack_group_span": None if attacks is None else len(indices & attacks)}
                    for indices in items] for level, items in supports.items()}
