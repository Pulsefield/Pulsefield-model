"""Online causal backbone and mirror-equivariant complete-row distribution."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math

import torch
from torch import Tensor, nn

from ..scoped_style_modeling.dataset import ContractError
from .config import BackboneConfig
from .engine import PredictionInput
from .features import ClockReadout, HistoryEncoder, SkeletonTimeEncoder, TIME_FEATURE_SCHEMA
from .local import LocalEncoder, LocalState
from .relation import RelationEncoder, RelationState
from .replay import ExactReplayState
from .schema import Actions, CompleteRow
from .temporal import TemporalEncoder, TemporalState

CACHE_SCHEMA = "oracle-causal/layer-input-mean/post-content-archive-v1/skeleton-time-v1/" + TIME_FEATURE_SCHEMA
WEIGHTS_FORMAT = "oracle-time-continuation/weights-v2-bounded-time"


def row_index(actions: Actions) -> int:
    """Encode a materialized row in serialized lane order; reject unknown/padding."""
    actions = CompleteRow(0, actions).actions
    return 64 * actions[0] + 16 * actions[1] + 4 * actions[2] + actions[3]


@dataclass(frozen=True)
class PreRowEncoding:
    """Final query outputs [rows,2,D]; content outputs cannot be scored by the head."""

    hidden: Tensor
    inputs: tuple[PredictionInput, ...]


@dataclass(frozen=True)
class JointRowDistribution:
    """256 serialized-lane rows with illegal and empty rows assigned -inf."""

    log_probs: Tensor
    legal: Tensor

    def score(self, actions: Actions) -> Tensor:
        index = row_index(actions)
        if not bool(self.legal[index]):
            raise ContractError("Cannot score an illegal complete row")
        return self.log_probs[index]


class JointHead(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.rank = config.coupling_rank
        self.unary = nn.Linear(config.temporal_hidden, 16)
        self.actions = nn.Embedding(16, self.rank)
        self.interaction = nn.Linear(config.temporal_hidden, self.rank ** 2)
        table = torch.tensor(list(product(range(4), repeat=4)), dtype=torch.long)
        self.register_buffer("rows", table)
        self.register_buffer("left", 4 * table[:, 0] + table[:, 1])
        self.register_buffer("right", 4 * table[:, 3] + table[:, 2])
        self.clock_readout = (ClockReadout(config.clock_readout_hidden, config.time_lookahead_rows)
                              if config.clock_readout_hidden else None)

    def forward(self, query: PreRowEncoding) -> tuple[Tensor, Tensor]:
        if not isinstance(query, PreRowEncoding):
            raise ContractError("Joint head requires pre-row query outputs")
        hidden = query.hidden
        unary = self.unary(hidden)
        if self.clock_readout is not None:
            unary = unary + self.clock_readout(query.inputs)
        matrices = self.interaction(hidden).reshape(-1, 2, self.rank, self.rank)
        coupling = (matrices[:, 0] + matrices[:, 1].transpose(-1, -2)) / 2
        pair = (self.actions.weight @ coupling @ self.actions.weight.T) / math.sqrt(self.rank)
        scores = (unary[:, 0, :, None] + unary[:, 1, None, :] + pair)[:, self.left, self.right]
        occupied = torch.tensor([item.history.occupancy for item in query.inputs], dtype=torch.bool,
                                device=hidden.device)[:, None]
        terminal = torch.tensor([item.is_terminal for item in query.inputs], dtype=torch.bool,
                                device=hidden.device)[:, None, None]
        rows = self.rows[None]
        ordinary = (rows == 0) | (occupied & (rows == 3)) | (~occupied & ((rows == 1) | (rows == 2)))
        final = (occupied & (rows == 3)) | (~occupied & ((rows == 0) | (rows == 1)))
        legal = torch.where(terminal, final, ordinary).all(-1) & self.rows.any(-1)[None]
        if not bool(torch.isfinite(scores.masked_select(legal)).all()):
            raise ContractError("Legal joint-row scores must be finite")
        return scores.masked_fill(~legal, -torch.inf).log_softmax(-1), legal


class CausalBackbone(nn.Module):
    """Shared hand operators; local/relation writes use only committed actions.

    Use ContinuationEngine for source scheduling, atomic commit, content-only
    prefill and dense teacher forcing. No optimizer or sampling policy is owned
    here. Parameter/device changes invalidate all existing learned states.
    """

    def __init__(self, config: BackboneConfig = BackboneConfig()):
        super().__init__()
        self.config = config
        self.facts = HistoryEncoder(config.hidden)
        self.local = LocalEncoder(config.hidden)
        self.relation = RelationEncoder(config)
        self.temporal = TemporalEncoder(config)
        self.head = JointHead(config)
        # Construct after the shared backbone to preserve its initialization.
        self.timing = (SkeletonTimeEncoder(config.hidden, config.time_lookahead_rows)
                       if config.time_lookahead_rows else None)

    def cache_signature(self) -> tuple:
        # Process-local guards also detect optimizer/load_state_dict in-place
        # writes. Durable checkpoint identity is a runtime responsibility.
        return (CACHE_SCHEMA, self.config,
                tuple((id(p), p._version, p.device, p.dtype)
                      for p in (*self.parameters(), *self.buffers())))

    def query_input(self, query: PredictionInput, local: LocalState, relation: RelationState) -> Tensor:
        raw = self.facts(query.history, query.time_ms, local.pace, query.clocks.previous_row_ms, query.is_terminal)
        if self.timing is not None:
            raw = raw + self.timing((query.future_offsets_ms,))
        frontier = self.local.read(raw, local)
        return self.relation(frontier, relation, query.time_ms, query.history.row_count + 1)

    def content_input(self, query: PredictionInput, row: CompleteRow, post: ExactReplayState,
                      local: LocalState, relation: RelationState) -> tuple[Tensor, LocalState, RelationState]:
        gap = query.clocks.previous_row_ms
        pace = local.pace.commit(gap)
        raw = self.facts(post, row.time_ms, pace, gap, query.is_terminal)
        if self.timing is not None:
            raw = raw + self.timing((query.future_offsets_ms,))
        local = self.local.commit(local, row, post.row_count, raw, gap)
        frontier = self.local.read(raw, local)
        relation = self.relation.commit(relation, row, post.row_count, frontier)
        content = self.relation(frontier, relation, row.time_ms, post.row_count)
        return content, local, relation

    def forward(self, query: PredictionInput, local: LocalState, relation: RelationState,
                temporal: TemporalState) -> JointRowDistribution:
        frontier = self.query_input(query, local, relation)
        hidden = self.temporal.query(frontier, temporal, query.time_ms)
        log_probs, legal = self.head(PreRowEncoding(hidden[None], (query,)))
        return JointRowDistribution(log_probs[0], legal[0])
