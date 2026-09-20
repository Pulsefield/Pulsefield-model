"""Matched joint action heads and full-support autoregressive LN pointers.

The object likelihood marginalizes two mirror-reversed factor orders. Each
factor uses the exact same completion support at training and sampling time.
Candidate activation storage is bounded by recomputing packed score blocks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..scoped_style_modeling.dataset import ContractError
from .consequence import RowConsequence
from .long_memory import LandmarkMemory, MemoryQuery
from .contract import Arm, HEAD_ACTIONS, ROW_ACTIONS, Schedule
from .features import (AVAILABILITY_DIM, CANDIDATE_DIM, CONTENT_DIM, FACTOR_DIM, QUERY_DIM,
                       EndpointAvailability, TimingView, endpoint_availability, factor_features)
from .support import row_supports
from .routing import HeadRouting, ReleaseRouting
from .temporal import FiniteTemporal, TemporalConfig, pointwise


@dataclass(frozen=True)
class ModelConfig:
    arm: Arm
    hidden: int = 128
    levels: int = 8
    expansion: int = 4
    coupling_rank: int = 16
    endpoint_availability: str = 'none'
    row_consequence: str = 'none'
    seed_context: str = 'none'
    long_memory: str = 'none'
    memory_hidden: int = 256
    memory_stride: int = 64
    head_routing: str = 'none'
    routing_hidden: int = 512
    release_routing: str = 'none'
    release_hidden: int = 512

    def __post_init__(self):
        if not isinstance(self.arm, Arm):
            raise ContractError('Model configuration requires an explicit task arm')
        TemporalConfig(CONTENT_DIM, self.hidden, self.levels, self.expansion)
        if type(self.coupling_rank) is not int or self.coupling_rank <= 0:
            raise ContractError('Joint action coupling rank must be a positive integer')
        if (self.endpoint_availability not in ('none', 'zero', 'commitment') or
                self.arm != Arm.O1 and self.endpoint_availability != 'none'):
            raise ContractError('Endpoint availability must be none, zero or commitment, and is O1-only')
        if (self.row_consequence not in ('none', 'actions', 'frontier') or
                self.arm != Arm.R1 and self.row_consequence != 'none'):
            raise ContractError('Row consequence must be none, actions or frontier, and is R1-only')
        if (self.seed_context not in ('none', 'zero', 'observed') or
                self.arm != Arm.R1 and self.seed_context != 'none'):
            raise ContractError('Seed context must be none, zero or observed, and is R1-only')
        if self.long_memory not in ('none', 'landmarks') or self.arm != Arm.R1 and self.long_memory != 'none':
            raise ContractError('Long memory must be none or landmarks, and is R1-only')
        if self.head_routing not in ('none', 'residual') or self.arm != Arm.R1 and self.head_routing != 'none':
            raise ContractError('Head routing must be none or residual, and is R1-only')
        if type(self.routing_hidden) is not int or self.routing_hidden <= 0:
            raise ContractError('Head-routing width must be a positive integer')
        if self.release_routing not in ('none', 'residual') or self.arm != Arm.R1 and self.release_routing != 'none':
            raise ContractError('Release routing must be none or residual, and is R1-only')
        if type(self.release_hidden) is not int or self.release_hidden <= 0:
            raise ContractError('Release-routing width must be a positive integer')
        if any(type(n) is not int or n <= 0 for n in (self.memory_hidden, self.memory_stride)):
            raise ContractError('Long-memory width and onset stride must be positive integers')


class JointHead(nn.Module):
    """Shared hand unaries plus a state-dependent mirror-equivariant coupling."""
    def __init__(self, hidden, vocabulary, rank):
        super().__init__()
        self.rank = rank
        self.unary = nn.Linear(hidden, vocabulary ** 2)
        self.matrix = nn.Linear(hidden, rank ** 2)
        self.actions = nn.Embedding(vocabulary ** 2, rank)
        choices = HEAD_ACTIONS if vocabulary == 3 else ROW_ACTIONS
        self.register_buffer('left', torch.tensor([a[0] * vocabulary + a[1] for a in choices]), persistent=False)
        self.register_buffer('right', torch.tensor([a[3] * vocabulary + a[2] for a in choices]), persistent=False)

    def forward(self, hands):
        unaries = self.unary(hands)
        matrices = self.matrix(hands).reshape(-1, 2, self.rank, self.rank)
        coupling = (matrices[:, 0] + matrices[:, 1].transpose(-1, -2)) / 2.
        pair = (self.actions.weight @ coupling @ self.actions.weight.T) / math.sqrt(self.rank)
        scores = unaries[:, 0, :, None] + unaries[:, 1, None, :] + pair
        return scores[:, self.left, self.right]


@dataclass(frozen=True)
class EndpointFactor:
    view: TimingView
    index: int
    start: int
    stop: int
    target: int | None = None
    availability: EndpointAvailability | None = None

    def __post_init__(self):
        if (any(type(value) is not int for value in (self.index, self.start, self.stop)) or
                not 0 <= self.index < self.start < self.stop <= len(self.view.times)):
            raise ContractError('Endpoint factor needs nonempty strictly future support')
        if self.target is not None and (type(self.target) is not int or not self.start <= self.target < self.stop):
            raise ContractError('Endpoint target is outside the factor completion support')


class EndpointPointer(nn.Module):
    def __init__(self, hidden, availability='none'):
        super().__init__()
        if availability not in ('none', 'zero', 'commitment'):
            raise ContractError('Unknown endpoint availability mode')
        self.availability = availability
        self.context = nn.Sequential(nn.Linear(hidden + FACTOR_DIM, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.candidate = nn.Sequential(nn.Linear(CANDIDATE_DIM, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.bias = nn.Linear(CANDIDATE_DIM, 1)
        self.availability_residual = None
        if availability != 'none':
            self.availability_residual = nn.Sequential(
                nn.Linear(hidden + CANDIDATE_DIM + AVAILABILITY_DIM, max(4, hidden // 2)),
                nn.GELU(), nn.Linear(max(4, hidden // 2), 1))
            nn.init.zeros_(self.availability_residual[-1].weight)
            nn.init.zeros_(self.availability_residual[-1].bias)

    def features(self, factor, start, stop):
        base = factor.view.candidates(factor.index, start, stop)
        if self.availability == 'none':
            return base
        if self.availability == 'commitment':
            if factor.availability is None:
                raise ContractError('Commitment endpoint scoring requires its known partial plan')
            extra = factor.view.availability(factor.index, start, stop, factor.availability)
        else:
            extra = np.zeros((stop - start, AVAILABILITY_DIM), dtype=np.float32)
        return np.concatenate((base, extra), -1)

    def score(self, context, features):
        base = features[..., :CANDIDATE_DIM].contiguous()
        score = (context * self.candidate(base)).sum(-1) / math.sqrt(context.shape[-1]) + self.bias(base).squeeze(-1)
        if self.availability_residual is not None:
            expanded = context.expand(*features.shape[:-1], context.shape[-1])
            score = score + self.availability_residual(torch.cat((expanded, features), -1)).squeeze(-1)
        return score

    def log_prob(self, context: Tensor, factors: Sequence[EndpointFactor], *,
                 candidate_budget=8192, recompute=True):
        """Exact normalized log probabilities for ragged, full-support factors.

        Each packed block contains at most candidate_budget pairs. With
        recompute=True its candidate features/activations are recreated during
        backward, rather than retaining all future candidates for every LN.
        The small reduction graph and factor contexts remain live. No optimizer
        update or timing mutation is allowed between forward and backward.
        """
        if type(candidate_budget) is not int or candidate_budget <= 0:
            raise ContractError('Candidate budget must be a positive integer')
        if context.ndim != 2 or len(context) != len(factors) or any(f.target is None for f in factors):
            raise ContractError('Pointer likelihood requires one context and observed target per factor')
        if not factors:
            return context.new_empty((0,))
        factors = tuple(factors)
        targets = np.concatenate([self.features(f, f.target, f.target + 1) for f in factors])
        selected = self.score(context, context.new_tensor(targets))
        normalizers = [None] * len(factors)

        def consume(parts):
            def block(values, owned_parts=tuple(parts)):
                features = np.concatenate([self.features(factors[i], start, stop)
                                           for i, start, stop in owned_parts])
                owner_ids = np.repeat([i for i, _, _ in owned_parts], [stop - start for _, start, stop in owned_parts])
                owners = torch.as_tensor(owner_ids, dtype=torch.long, device=values.device)
                scores = self.score(values.index_select(0, owners), values.new_tensor(features))
                offset, sums = 0, []
                for _, start, stop in owned_parts:
                    count = stop - start
                    sums.append(torch.logsumexp(scores[offset:offset + count], 0))
                    offset += count
                return torch.stack(sums)

            sums = checkpoint(block, context, use_reentrant=False) if recompute and torch.is_grad_enabled() else block(context)
            for j, (i, _, _) in enumerate(parts):
                previous = normalizers[i]
                normalizers[i] = sums[j] if previous is None else torch.logaddexp(previous, sums[j])

        parts, used = [], 0
        for i, factor in enumerate(factors):
            start = factor.start
            while start < factor.stop:
                stop = min(factor.stop, start + candidate_budget - used)
                parts.append((i, start, stop))
                used += stop - start
                start = stop
                if used == candidate_budget:
                    consume(parts)
                    parts, used = [], 0
        if parts:
            consume(parts)
        return selected - torch.stack(normalizers)

    @torch.no_grad()
    def sample(self, context: Tensor, factor: EndpointFactor, *, generator, candidate_budget=8192):
        """Exact categorical draw using a streaming Gumbel maximum on CPU RNG."""
        if type(candidate_budget) is not int or candidate_budget <= 0 or context.ndim != 1:
            raise ContractError('Pointer sampling needs one context and a positive candidate budget')
        best, chosen = -math.inf, None
        for start in range(factor.start, factor.stop, candidate_budget):
            stop = min(factor.stop, start + candidate_budget)
            features = context.new_tensor(self.features(factor, start, stop))
            logits = self.score(context, features).cpu().double()
            uniform = torch.rand(len(logits), generator=generator, dtype=torch.float64).clamp_min(torch.finfo(torch.float64).tiny)
            scores = logits - (-uniform.log()).log()
            value, offset = scores.max(0)
            if float(value) > best:
                best, chosen = float(value), start + int(offset)
        return chosen


class BoundedModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.temporal = FiniteTemporal(TemporalConfig(CONTENT_DIM, config.hidden, config.levels, config.expansion))
        self.exact = nn.Sequential(nn.Linear(QUERY_DIM, config.hidden), nn.GELU(), nn.Linear(config.hidden, config.hidden))
        self.fuse = nn.Sequential(nn.LayerNorm(2 * config.hidden), nn.Linear(2 * config.hidden, config.hidden),
                                  nn.GELU(), nn.Linear(config.hidden, config.hidden))
        self.joint = JointHead(config.hidden, 3 if config.arm == Arm.O1 else 4, config.coupling_rank)
        self.pointer = EndpointPointer(config.hidden, config.endpoint_availability) if config.arm == Arm.O1 else None
        self.row_consequence = (RowConsequence(config.hidden, config.row_consequence)
                                if config.row_consequence != 'none' else None)
        self.seed_residual = None
        if config.seed_context != 'none':
            self.seed_residual = nn.Sequential(nn.Linear(2 * config.hidden, config.hidden), nn.GELU(),
                                               nn.Linear(config.hidden, config.hidden, bias=False))
            nn.init.zeros_(self.seed_residual[-1].weight)
        self.long_memory = (LandmarkMemory(CONTENT_DIM, config.memory_hidden, config.hidden, config.memory_stride)
                            if config.long_memory != 'none' else None)
        self.route_residual = (HeadRouting(config.hidden, config.routing_hidden)
                               if config.head_routing != 'none' else None)
        self.release_residual = (ReleaseRouting(config.hidden, config.release_hidden)
                                 if config.release_routing != 'none' else None)

    @property
    def choices(self):
        return HEAD_ACTIONS if self.config.arm == Arm.O1 else ROW_ACTIONS

    def encode_seed(self, raw: Tensor, valid: Tensor):
        """Pool only the externally supplied seed, under current shared weights.

        Every seed token contributes, even when the seed exceeds the local
        receptive field. Padding is a suffix; suffix-label features are forbidden
        by the caller's seed projection. This learned value is not durable state.
        """
        if (self.config.seed_context != 'observed' or raw.ndim != 4 or
                raw.shape[2:] != (2, CONTENT_DIM) or valid.shape != raw.shape[:2] or
                valid.dtype != torch.bool or valid.device != raw.device or raw.shape[1] == 0 or
                not bool(valid[:, 0].all()) or bool((valid[:, 1:] & ~valid[:, :-1]).any())):
            raise ContractError('Observed seed encoding requires a nonempty prefix-padded seed per sample')
        encoded = self.temporal(raw, valid)
        return (encoded * valid[..., None, None]).sum(1) / valid.sum(1)[:, None, None]

    def readout(self, before: Tensor, exact: Tensor, seed: Tensor | None = None,
                memory: MemoryQuery | None = None):
        if before.shape[:-1] != exact.shape[:-1] or before.shape[-2:] != (2, self.config.hidden) or exact.shape[-1] != QUERY_DIM:
            raise ContractError('Readout requires aligned pre-decision history and exact hand features')
        hands = pointwise(self.fuse, torch.cat((before, pointwise(self.exact, exact)), -1))
        if self.config.seed_context == 'observed':
            if seed is None or seed.shape != hands.shape or seed.device != hands.device or seed.dtype != hands.dtype:
                raise ContractError('Observed readout requires aligned original seed hand vectors')
        elif seed is not None:
            raise ContractError('Unconditioned and zero controls do not consume observed seed vectors')
        if self.seed_residual is not None:
            context = seed if self.config.seed_context == 'observed' else torch.zeros_like(hands)
            hands = hands + pointwise(self.seed_residual, torch.cat((hands, context), -1))
        if self.long_memory is not None:
            if memory is None:
                raise ContractError('Landmark readout requires its complete causal memory query')
            hands = hands + self.long_memory.read(hands, memory)
        elif memory is not None:
            raise ContractError('A model without long memory cannot consume landmark queries')
        return hands

    def decision_log_probs(self, hands: Tensor, states: Sequence[Schedule]):
        if hands.shape != (len(states), 2, self.config.hidden) or not states or any(s.arm != self.config.arm for s in states):
            raise ContractError('Decision queries must match the model task and encoded hands')
        supports = [s.head_support() for s in states] if self.config.arm == Arm.O1 else row_supports(states)
        mask = torch.tensor(supports, dtype=torch.bool, device=hands.device)
        if not bool(mask.any(-1).all()):
            raise ContractError('Decision query has no feasible action; O1 non-onsets execute deterministically')
        scores = self.joint(hands)
        if self.row_consequence is not None:
            scores = scores + self.row_consequence(hands, states)
        if self.route_residual is not None:
            onsets = torch.tensor([s.timing.onsets[s.index] for s in states], device=hands.device)
            scores = scores + self.route_residual(hands, onsets)
        if self.release_residual is not None:
            occupied = torch.tensor([any(s.replay.occupancy) for s in states], device=hands.device)
            scores = scores + self.release_residual(hands, occupied)
        return scores.masked_fill(~mask, -torch.inf).log_softmax(-1)

    def endpoint_log_probs(self, hands: Tensor, states: Sequence[Schedule], heads, endpoints,
                           views: Sequence[TimingView], *, candidate_budget=8192, recompute=True, ledger=None):
        """Joint endpoint likelihood per onset, marginalizing both factor orders.

        Chosen head groups and previous within-row true endpoints are legal
        teacher-forced inputs here. They must not enter the head/type readout.
        """
        if self.pointer is None or hands.shape != (len(states), 2, self.config.hidden) or not len(states):
            raise ContractError('Endpoint likelihood requires encoded O1 onset queries')
        if not len(states) == len(heads) == len(endpoints) == len(views):
            raise ContractError('Endpoint labels and timing views must align with onset queries')
        contexts, raw, factors, owners = [], [], [], []
        for i, (state, group, targets, view) in enumerate(zip(states, heads, endpoints, views)):
            if state.arm != Arm.O1 or view.timing is not state.timing or not state.head_possible(group):
                raise ContractError('Endpoint likelihood requires a feasible O1 head group and matching timing')
            if set(targets) != {c for c, action in enumerate(group) if action == 2}:
                raise ContractError('Endpoint targets must name exactly the new LN lanes')
            for order_id, order in enumerate(((0, 1, 2, 3), (3, 2, 1, 0))):
                assigned = {}
                for lane in order:
                    if group[lane] != 2:
                        continue
                    start, stop = state.endpoint_bounds(group, assigned, lane)
                    contexts.append(hands[i, 0 if lane < 2 else 1])
                    raw.append(factor_features(state, group, assigned, lane))
                    availability = (endpoint_availability(state, group, assigned, lane)
                                    if self.config.endpoint_availability == 'commitment' else None)
                    factors.append(EndpointFactor(view, state.index, start, stop, targets[lane], availability))
                    owners.append(2 * i + order_id)
                    assigned[lane] = targets[lane]
        if ledger is not None:
            ledger.update(endpoint_decisions=len(factors) // 2, scored_order_factors=len(factors),
                          candidate_pairs=sum(f.stop - f.start for f in factors))
        if not factors:
            return hands.sum((1, 2)) * 0.
        context = self.pointer.context(torch.cat((torch.stack(contexts), hands.new_tensor(np.stack(raw))), -1))
        values = self.pointer.log_prob(context, factors, candidate_budget=candidate_budget, recompute=recompute)
        owner_tensor = torch.tensor(owners, dtype=torch.long, device=hands.device)
        sums = hands.new_zeros(2 * len(states)).index_add(0, owner_tensor, values).reshape(len(states), 2)
        return torch.logsumexp(sums, -1) - math.log(2.)

    @torch.no_grad()
    def sample_endpoints(self, hands, state, heads, view, *, generator, candidate_budget=8192):
        if self.pointer is None or view.timing is not state.timing or not state.head_possible(heads):
            raise ContractError('Object sampling requires a feasible chosen O1 head group')
        order = (0, 1, 2, 3) if int(torch.randint(2, (), generator=generator)) == 0 else (3, 2, 1, 0)
        assigned = {}
        for lane in order:
            if heads[lane] != 2:
                continue
            start, stop = state.endpoint_bounds(heads, assigned, lane)
            raw = hands.new_tensor(factor_features(state, heads, assigned, lane))
            context = self.pointer.context(torch.cat((hands[0 if lane < 2 else 1], raw)))
            availability = (endpoint_availability(state, heads, assigned, lane)
                            if self.config.endpoint_availability == 'commitment' else None)
            assigned[lane] = self.pointer.sample(context, EndpointFactor(view, state.index, start, stop, availability=availability),
                                                 generator=generator, candidate_budget=candidate_budget)
        return assigned
