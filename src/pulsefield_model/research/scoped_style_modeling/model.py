"""Shared chart encoder, selector-free assessment, and teacher-forced evidence.

The encoder is the only shared trainable component. Branches read its output
without mutation; no target or selector state is accepted by assessment().
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .dataset import ContractError
from .config import ModelConfig
from .replay import hand_mask
from .tensors import Batch, ChartTensors, SelectorTensors, LANE_DIM, ROW_DIM, EDGE_DIM, RELATION_DIM, HISTORY_DIM


def mlp(inputs, hidden, outputs, dropout=0.0):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, outputs))


def packed_gru(gru: nn.GRU, values: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
    packed = pack_padded_sequence(values, lengths, batch_first=True, enforce_sorted=False)
    output, terminal = gru(packed)
    output, _ = pad_packed_sequence(output, batch_first=True, total_length=values.shape[1])
    # h_n contains each direction's terminal; output[:, -1] does not.
    return output, terminal.transpose(0, 1).flatten(1)


class RelationAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = 2*config.hand_hidden
        self.heads = config.attention_heads
        self.qkv = nn.Linear(dim, 3*dim)
        self.edge = nn.Linear(EDGE_DIM, dim)
        self.relation = mlp(RELATION_DIM, dim, dim)
        self.bias = nn.Linear(dim, self.heads)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = mlp(dim, config.feedforward_dim, dim, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, chart: ChartTensors, extra_descriptor: Tensor | None = None) -> Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        n, dim = x.shape
        query, neighbor = chart.edge_index
        q, k, v = self.qkv(x).reshape(n, 3, self.heads, dim//self.heads).unbind(1)
        descriptor = self.edge(chart.edge_features)
        descriptor = descriptor.index_add(0, chart.relation_edges, self.relation(chart.relation_features))
        if extra_descriptor is not None:
            descriptor = descriptor + extra_descriptor
        scores = (q[query]*k[neighbor]).sum(-1)/math.sqrt(dim//self.heads) + self.bias(descriptor)
        indices = query[:, None].expand(-1, self.heads)
        maxima = scores.new_full((n, self.heads), -torch.inf)
        maxima.scatter_reduce_(0, indices, scores.detach(), reduce="amax", include_self=True)
        weights = (scores-maxima[query]).exp()
        denominator = scores.new_zeros(n, self.heads).index_add(0, query, weights)
        weights = weights/denominator[query]
        values = v[neighbor] + self.value(descriptor).reshape(-1, self.heads, dim//self.heads)
        message = x.new_zeros(n, self.heads, dim//self.heads).index_add(0, query, weights[:, :, None]*values)
        x = self.norm1(x+self.dropout(self.output(message.flatten(1))))
        x = self.norm2(x+self.dropout(self.ff(x)))
        return x.reshape(shape)


class ChartEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.lane = nn.Sequential(nn.Linear(LANE_DIM, config.lane_dim), nn.GELU())
        self.hand = nn.GRU(2*config.lane_dim+ROW_DIM, config.hand_hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(config.dropout)
        self.relations = RelationAttention(config)

    def forward(self, chart: ChartTensors) -> Tensor:
        lanes = self.lane(chart.lanes).flatten(-2)
        b, t = lanes.shape[:2]
        values = torch.cat((lanes, chart.rows[:, :, None].expand(-1, -1, 2, -1)), -1)
        values = self.dropout(values.permute(0, 2, 1, 3).reshape(b*2, t, -1))
        h, _ = packed_gru(self.hand, values, chart.lengths.repeat_interleave(2))
        h = h.reshape(b, 2, t, -1).transpose(1, 2)
        h = self.relations(h, chart)
        valid = torch.arange(t, device=h.device)[None] < chart.lengths.to(h.device)[:, None]
        return h * valid[:, :, None, None]


class AssessmentHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.concept = nn.Embedding(5, config.concept_dim)
        self.pair = mlp(4*config.hand_hidden, config.row_dim, config.row_dim, config.dropout)
        self.section = nn.GRU(config.row_dim+config.concept_dim+3, config.section_hidden,
                              batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(config.dropout)
        self.head = mlp(4*config.section_hidden+2, config.head_hidden, 3, config.dropout)

    def summarize(self, h: Tensor, chart: ChartTensors, concepts: Tensor) -> Tensor:
        batch = torch.arange(h.shape[0], device=h.device)[:, None]
        section = h[batch, chart.section_indices]
        row = (self.pair(section.flatten(-2)) + self.pair(section.flip(-2).flatten(-2)))/2
        query = self.concept(concepts)[:, None].expand(-1, row.shape[1], -1)
        values = self.dropout(torch.cat((row, query, chart.section_features), -1))
        output, terminal = packed_gru(self.section, values, chart.section_lengths)
        count = chart.section_events.sum(1, keepdim=True)
        mean = (output*chart.section_events[:, :, None]).sum(1)/count.clamp_min(1)
        return torch.cat((terminal, mean, chart.duration, (count == 0).to(h.dtype)), -1)

    def forward(self, h: Tensor, chart: ChartTensors, concepts: Tensor) -> Tensor:
        return self.head(self.summarize(h, chart, concepts))


@dataclass(frozen=True)
class EvidenceOutput:
    log_probs: Tensor
    sequence_nll: Tensor
    normalized_nll: Tensor
    eligible: Tensor
    final_states: Tensor


class EvidenceSelector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        c, q = 2*config.hand_hidden, config.selector_hidden
        self.hidden = q
        self.concept = nn.Embedding(5, config.concept_dim)
        self.assessment = nn.Embedding(3, config.assessment_dim)
        self.mask = nn.Embedding(4, config.mask_dim)
        condition = config.concept_dim+config.assessment_dim
        self.context = mlp(2*c+2*q+4*HISTORY_DIM+condition+1, config.head_hidden, q)
        self.unary = mlp(q+config.mask_dim, config.head_hidden, 1)
        self.interaction = mlp(2*q+2*config.mask_dim, config.head_hidden, 1)
        self.memory = nn.GRUCell(c+q+2*config.mask_dim+condition+1, q)
        self.register_buffer("hand_masks", torch.tensor([[hand_mask(m, h) for h in range(2)] for m in range(16)]))

    def advance(self, c, prior, chosen, condition, dt):
        """Update both hands from the same pre-decision states."""
        masks = self.mask(self.hand_masks[chosen])
        inputs = torch.cat((c, prior.flip(1), masks, masks.flip(1),
                            condition[:, None].expand(-1, 2, -1), dt[:, None, None].expand(-1, 2, 1)), -1)
        return self.memory(inputs.flatten(0, 1), prior.flatten(0, 1)).reshape_as(prior)

    def forward(self, h: Tensor, concepts: Tensor, assessments: Tensor, data: SelectorTensors) -> EvidenceOutput:
        condition = torch.cat((self.concept(concepts), self.assessment(assessments)), -1)
        states = h.new_zeros(h.shape[0], 2, self.hidden)
        batch = torch.arange(h.shape[0], device=h.device)
        all_masks = self.mask(self.hand_masks)
        probabilities = []
        for d in range(data.indices.shape[1]):
            c = h[batch, data.indices[:, d]]
            history = data.history[:, d].flatten(-2)
            values = torch.cat((c, c.flip(1), states, states.flip(1), history, history.flip(1),
                                condition[:, None].expand(-1, 2, -1),
                                data.times[:, d, None, None].expand(-1, 2, 1)), -1)
            v = self.context(values)[:, None].expand(-1, 16, -1, -1)
            masks = all_masks[None].expand(h.shape[0], -1, -1, -1)
            unary = self.unary(torch.cat((v, masks), -1)).squeeze(-1).sum(-1)
            left, right = v.unbind(2)
            ml, mr = masks.unbind(2)
            interaction = (self.interaction(torch.cat((left, right, ml, mr), -1)) +
                           self.interaction(torch.cat((right, left, mr, ml), -1))).squeeze(-1)/2
            logits = (unary+interaction).masked_fill(~data.valid_masks[:, d], -torch.inf)
            probabilities.append(F.log_softmax(logits, -1))
            updated = self.advance(c, states, data.targets[:, d], condition, data.times[:, d])
            states = torch.where(data.steps[:, d, None, None], updated, states)
        log_probs = torch.stack(probabilities, 1)
        nontrivial = (data.valid_masks.sum(-1) > 1) & data.steps
        nll = -log_probs.gather(-1, data.targets[:, :, None]).squeeze(-1)
        sequence = (nll*nontrivial).sum(1)
        eligible = data.available & nontrivial.any(1)
        normalized = torch.where(eligible, sequence/nontrivial.sum(1).clamp_min(1), 0.0)
        return EvidenceOutput(log_probs, torch.where(eligible, sequence, 0.0), normalized, eligible, states)


class ScopedStyleModel(nn.Module):
    def __init__(self, config: ModelConfig, *, auxiliary: bool = True):
        super().__init__()
        config.validate()
        self.encoder = ChartEncoder(config)
        self.assessor = AssessmentHead(config)
        self.selector = EvidenceSelector(config) if auxiliary else None

    def assessment(self, chart: ChartTensors, concepts: Tensor) -> Tensor:
        return self.assessor(self.encoder(chart), chart, concepts)


def initialize_model(config: ModelConfig, seed: int, *, auxiliary: bool) -> ScopedStyleModel:
    """Keep shared initialization and caller RNG independent of selector creation."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return ScopedStyleModel(config, auxiliary=auxiliary)


@dataclass(frozen=True)
class Losses:
    total: Tensor
    assessment: Tensor
    evidence: Tensor
    logits: Tensor
    auxiliary: EvidenceOutput | None


def losses(model: ScopedStyleModel, batch: Batch, beta: float) -> Losses:
    if not math.isfinite(beta) or beta < 0:
        raise ContractError("beta must be finite and nonnegative")
    h = model.encoder(batch.chart)
    logits = model.assessor(h, batch.chart, batch.concepts)
    primary = F.cross_entropy(logits, batch.assessments, reduction="none")
    evidence = torch.zeros_like(primary)
    auxiliary = None
    if beta:
        if model.selector is None:
            raise ContractError("Positive beta requires an auxiliary selector")
        auxiliary = model.selector(h, batch.concepts, batch.assessments, batch.selector)
        evidence = auxiliary.normalized_nll
    # beta=0 never evaluates the selector or consumes its random stream.
    return Losses((primary+beta*evidence).mean(), primary.mean(), evidence.mean(), logits, auxiliary)
