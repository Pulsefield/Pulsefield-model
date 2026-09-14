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

DECODER_POLICY = "joint-row/context-bilinear-hand-transpose-v1"
LOSS_POLICY = "equal-block/mean-row-nll-v1"


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

    def forward(self, observation: ObservationTensors, *, extra_features: Tensor | None = None) -> Tensor:
        lanes = self.lane(observation.lanes).flatten(-2)
        b, t = lanes.shape[:2]
        values = torch.cat((lanes, observation.rows[:, :, None].expand(-1, -1, 2, -1)), -1)
        if extra_features is not None:
            values = values + extra_features
        values = self.dropout(values.permute(0, 2, 1, 3).reshape(b * 2, t, -1))
        states, _ = packed_gru(self.hand, values, observation.lengths.repeat_interleave(2))
        states = self.relations(states.reshape(b, 2, t, -1).transpose(1, 2), observation)
        valid = torch.arange(t, device=states.device)[None] < observation.lengths.to(states.device)[:, None]
        return states * valid[:, :, None, None]


@dataclass(frozen=True)
class Prediction:
    """Teacher-forced costs in nats; loss is the equal-block mean of mean_row_nll."""
    log_probs: Tensor
    row_nll: Tensor
    sequence_nll: Tensor
    mean_row_nll: Tensor
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
        self.pair_action = nn.Linear(a, config.interaction_dim, bias=False)
        self.interaction = nn.Linear(h, config.interaction_dim ** 2)
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
        pair = self.pair_action(self.action.weight)
        matrices = self.interaction(query).reshape(-1, 2, pair.shape[-1], pair.shape[-1])
        # Hand exchange transposes B. Its eigenvalues and context dependence
        # are unconstrained, so pair odds can favor agreement or alternation.
        matrix = (matrices[:, 0] + matrices[:, 1].transpose(-1, -2)) / 2
        interactions = (pair @ matrix @ pair.T) / math.sqrt(pair.shape[-1])
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
        """Consume one already-read context per query; score/advance own the prefix."""
        if targets.shape != queries.indices.shape or targets.dtype != torch.long or (
            (targets < 0) | (targets >= ROW_CLASSES)
        ).any():
            raise ContractError("Targets require one valid joint action token per query")
        states = encoded.new_zeros(encoded.shape[0], 2, self.hidden)
        occupancy = queries.entering_occupancy
        if encoded.shape[:2] != queries.indices.shape:
            raise ContractError("Decoder contexts must align with prediction queries")
        probabilities = []
        for step in range(queries.indices.shape[1]):
            context = encoded[:, step]
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
        sequence_nll = row_nll.sum(1)
        mean_row_nll = sequence_nll / denominator
        return Prediction(log_probs, row_nll, sequence_nll, mean_row_nll, mean_row_nll.mean(), states, occupancy)


class SourceActionPredictor(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig(), *, encoder: nn.Module | None = None):
        super().__init__()
        self.config = config
        self.encoder = ReferenceEncoder(config) if encoder is None else encoder
        self.decoder = JointDecoder(self.encoder.output_dim, config)

    @property
    def policy_identity(self):
        return {"architecture": "stage1-position-gather-v1", "decoder": DECODER_POLICY, "loss": LOSS_POLICY}

    def forward(self, batch: BlockBatch) -> Prediction:
        encoded = self.encoder(batch.observation)
        indices = torch.arange(encoded.shape[0], device=encoded.device)[:, None]
        return self.decoder(encoded[indices, batch.queries.indices], batch.queries, batch.targets)


def initialize_model(config: ModelConfig = ModelConfig(), seed: int = 17) -> SourceActionPredictor:
    """Initialize from scratch without consuming the caller's CPU RNG stream."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return SourceActionPredictor(config)


CONFIGURATIONS = ("reference_h", "composed_h", "composed_all")


class BankPredictor(nn.Module):
    """Fixed architecture/access arms sharing the action-reader/decoder interface."""
    def __init__(self, configuration: str, config: ModelConfig = ModelConfig()):
        super().__init__()
        from .representation import ActionReader, ComposedEncoder, ReferenceBankEncoder
        if configuration not in CONFIGURATIONS:
            raise ContractError(f"Unknown bank configuration: {configuration}")
        self.config, self.configuration = config, configuration
        self.access = "all" if configuration == "composed_all" else "H"
        self.encoder = ReferenceBankEncoder(config) if configuration == "reference_h" else ComposedEncoder(config)
        self.reader = ActionReader(config)
        self.decoder = JointDecoder(64, config)

    @property
    def policy_identity(self):
        from .representation import ARCHITECTURE, READER_POLICY
        return {"architecture": ARCHITECTURE if self.configuration != "reference_h" else "reference-bank-summary-v1",
                "reader": READER_POLICY, "access": self.access, "configuration": self.configuration,
                "decoder": DECODER_POLICY, "loss": LOSS_POLICY}

    def forward(self, batch: BlockBatch) -> Prediction:
        bank = self.encoder(batch.observation)
        return self.decoder(self.reader(bank, batch.queries, access=self.access), batch.queries, batch.targets)


def initialize_comparison(config: ModelConfig = ModelConfig(), seed: int = 17) -> dict[str, BankPredictor]:
    """Explicitly match reader, decoder and relation weights; clone the composed arms.

    The caller's CPU RNG stream is unchanged. The two composed arms differ only
    in access policy; they start with identical parameters and buffers.
    """
    from copy import deepcopy
    base = initialize_model(config, seed)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed + 10000)
        reference = BankPredictor("reference_h", config)
        torch.random.default_generator.manual_seed(seed + 20000)
        composed = BankPredictor("composed_h", config)
    reference.encoder.reference.load_state_dict(base.encoder.state_dict())
    composed.encoder.relations.load_state_dict(base.encoder.relations.state_dict())
    for model in (reference, composed):
        model.decoder.load_state_dict(base.decoder.state_dict())
    composed.reader.load_state_dict(reference.reader.state_dict())
    full = deepcopy(composed)
    full.configuration, full.access = "composed_all", "all"
    return dict(zip(CONFIGURATIONS, (reference, composed, full)))
