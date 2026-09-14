"""Shared-hand contextual reference encoder and a joint autoregressive decoder."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import torch
from torch import Tensor, nn

from ..scoped_style_modeling.config import ModelConfig as AttentionConfig
from ..scoped_style_modeling.dataset import ContractError
from ..scoped_style_modeling.model import RelationAttention, mlp, packed_gru
from .tensors import (BlockBatch, BlockQueries, ObservationTensors, LANE_DIM, ROW_DIM,
                      EDGE_DIM, RELATION_DIM, ROW_CLASSES, action_table)


@dataclass(frozen=True)
class ModelConfig:
    lane_dim: int = 16
    hand_hidden: int = 32
    attention_heads: int = 4
    feedforward_dim: int = 128
    decoder_hidden: int = 32
    action_dim: int = 8
    interaction_dim: int = 8
    dropout: float = 0.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if name != "dropout" and (type(value) is not int or value < 1):
                raise ContractError(f"{name} must be a positive integer")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1 or 2 * self.hand_hidden % self.attention_heads:
            raise ContractError("Invalid dropout or attention dimensions")


class ActionEncoder(Protocol):
    """Candidate interface: mask-aware inputs -> [batch, row, hand, output_dim]."""
    output_dim: int

    def __call__(self, observation: ObservationTensors) -> Tensor: ...


class ReferenceEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.output_dim = 2 * config.hand_hidden
        self.lane = nn.Sequential(nn.Linear(LANE_DIM, config.lane_dim), nn.GELU())
        self.hand = nn.GRU(2 * config.lane_dim + ROW_DIM, config.hand_hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(config.dropout)
        attention = AttentionConfig(hand_hidden=config.hand_hidden, attention_heads=config.attention_heads,
                                    feedforward_dim=config.feedforward_dim, dropout=config.dropout)
        self.relations = RelationAttention(attention, edge_dim=EDGE_DIM, relation_dim=RELATION_DIM)

    def forward(self, observation: ObservationTensors) -> Tensor:
        lanes = self.lane(observation.lanes).flatten(-2)
        b, t = lanes.shape[:2]
        values = torch.cat((lanes, observation.rows[:, :, None].expand(-1, -1, 2, -1)), -1)
        values = self.dropout(values.permute(0, 2, 1, 3).reshape(b * 2, t, -1))
        states, _ = packed_gru(self.hand, values, observation.lengths.repeat_interleave(2))
        states = self.relations(states.reshape(b, 2, t, -1).transpose(1, 2), observation)
        valid = torch.arange(t, device=states.device)[None] < observation.lengths.to(states.device)[:, None]
        return states * valid[:, :, None, None]


@dataclass(frozen=True)
class Prediction:
    log_probs: Tensor
    row_nll: Tensor
    block_nll: Tensor
    loss: Tensor
    final_states: Tensor
    final_occupancy: Tensor


class JointDecoder(nn.Module):
    """One categorical distribution over 6^4 exact joint source-action rows.

    Each hand has 36 action pairs. Learned bilateral interactions couple the
    hands, and a shared recurrent update reads only preceding chosen rows.
    All operations commute with exchanging the two hands in evaluation mode.
    """
    def __init__(self, encoder_dim: int, config: ModelConfig):
        super().__init__()
        h, a = config.decoder_hidden, config.action_dim
        self.hidden = h
        self.action = nn.Embedding(36, a)
        self.context = mlp(2 * encoder_dim + 2 * h + 8, h, h)
        self.unary = mlp(h + a, h, 1)
        self.interaction = nn.Linear(h + a, config.interaction_dim)
        self.memory = nn.GRUCell(encoder_dim + h + 2 * a, h)
        self.register_buffer("actions", action_table())
        self.register_buffer("hand_tokens", torch.tensor([[i // 36, i % 36] for i in range(ROW_CLASSES)]))

    def legal_rows(self, occupancy: Tensor) -> Tensor:
        """Use only declared occupation and the decoded prefix; no suffix constraints."""
        actions = self.actions[None]
        occupied = occupancy[:, None]
        invalid = (((actions & 4) != 0) & (occupied == 0)) | (
            ((actions & 3) != 0) & ((actions & 4) == 0) & (occupied == 1))
        legal = ~invalid.flatten(2).any(-1)
        # Every supplied real event has an action, including release-only events.
        return legal & (self.actions.flatten(1).any(-1)[None])

    def score(self, context: Tensor, states: Tensor, occupancy: Tensor, active: Tensor) -> Tensor:
        facts = torch.stack(((occupancy == 1).to(context.dtype), (occupancy >= 0).to(context.dtype)), -1).flatten(-2)
        query = self.context(torch.cat((context, context.flip(1), states, states.flip(1), facts, facts.flip(1)), -1))
        embeddings = self.action.weight[None, None].expand(query.shape[0], 2, -1, -1)
        values = torch.cat((query[:, :, None].expand(-1, -1, 36, -1), embeddings), -1)
        unary = self.unary(values).squeeze(-1)
        pair = self.interaction(values)
        interactions = torch.matmul(pair[:, 0], pair[:, 1].transpose(1, 2)) / math.sqrt(pair.shape[-1])
        logits = (unary[:, 0, :, None] + unary[:, 1, None, :] + interactions).flatten(1)
        padding = torch.arange(ROW_CLASSES, device=logits.device)[None] == 0
        legal = torch.where(active[:, None], self.legal_rows(occupancy), padding)
        return logits.masked_fill(~legal, -torch.inf).log_softmax(-1)

    def advance(self, context: Tensor, states: Tensor, occupancy: Tensor, chosen: Tensor, active: Tensor):
        actions = self.actions[chosen]
        embeddings = self.action(self.hand_tokens[chosen])
        inputs = torch.cat((context, states.flip(1), embeddings, embeddings.flip(1)), -1)
        updated = self.memory(inputs.flatten(0, 1), states.flatten(0, 1)).reshape_as(states)
        new_occupancy = torch.where(actions != 0, ((actions & 2) != 0).long(), occupancy)
        return (torch.where(active[:, None, None], updated, states),
                torch.where(active[:, None, None], new_occupancy, occupancy))

    def forward(self, encoded: Tensor, queries: BlockQueries, targets: Tensor) -> Prediction:
        if targets.shape != queries.indices.shape or targets.dtype != torch.long or (
            (targets < 0) | (targets >= ROW_CLASSES)
        ).any():
            raise ContractError("Targets require one valid joint action token per query")
        states = encoded.new_zeros(encoded.shape[0], 2, self.hidden)
        occupancy = queries.entering_occupancy
        batch = torch.arange(encoded.shape[0], device=encoded.device)
        probabilities = []
        for step in range(queries.indices.shape[1]):
            context = encoded[batch, queries.indices[:, step]]
            active = queries.steps[:, step]
            # Current targets are read only after the current distribution exists.
            probabilities.append(self.score(context, states, occupancy, active))
            states, occupancy = self.advance(context, states, occupancy, targets[:, step], active)
        log_probs = torch.stack(probabilities, 1)
        row_nll = -log_probs.gather(-1, targets[:, :, None]).squeeze(-1)
        row_nll = row_nll.masked_fill(~queries.steps, 0)
        if not torch.isfinite(row_nll).all():
            raise ContractError("Nonfinite likelihood or target contradicts prefix-only legality")
        denominator = queries.steps.sum(1)
        if (denominator == 0).any():
            raise ContractError("Every block must contain at least one target row")
        block_nll = row_nll.sum(1) / denominator
        return Prediction(log_probs, row_nll, block_nll, block_nll.mean(), states, occupancy)


class SourceActionPredictor(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig(), *, encoder: nn.Module | None = None):
        super().__init__()
        self.config = config
        self.encoder = ReferenceEncoder(config) if encoder is None else encoder
        self.decoder = JointDecoder(self.encoder.output_dim, config)

    def forward(self, batch: BlockBatch) -> Prediction:
        return self.decoder(self.encoder(batch.observation), batch.queries, batch.targets)


def initialize_model(config: ModelConfig = ModelConfig(), seed: int = 17) -> SourceActionPredictor:
    """Initialize from scratch without consuming the caller's CPU RNG stream."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return SourceActionPredictor(config)
