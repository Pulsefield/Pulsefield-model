"""Causal full-history encoding with retrievable completed-passage landmarks.

A landmark is written after each fixed count of head rows. Queries can read
only earlier physical rows; the current target never contributes its own memory.
Native caches are parameter-dependent and must be rebuilt from raw history.
"""
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from ..scoped_style_modeling.dataset import ContractError


@dataclass(frozen=True)
class MemoryBank:
    keys: tuple[Tensor, ...]
    values: tuple[Tensor, ...]
    positions: tuple[Tensor, ...]

    def query(self, batches, rows):
        return MemoryQuery(self, batches, rows)


@dataclass(frozen=True)
class MemoryQuery:
    bank: MemoryBank
    batches: Tensor
    rows: Tensor


@dataclass(frozen=True)
class MemoryCache:
    hidden: Tensor
    bank: MemoryBank
    rows: int = 0
    onsets: int = 0


class LandmarkMemory(nn.Module):
    def __init__(self, input_dim, hidden, output_dim, stride):
        super().__init__()
        self.input_dim, self.hidden, self.output_dim, self.stride = input_dim, hidden, output_dim, stride
        self.encoder = nn.GRU(input_dim, hidden, batch_first=True)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, output_dim)
        self.query = nn.Linear(output_dim, hidden)
        self.output = nn.Linear(output_dim, output_dim, bias=False)
        nn.init.zeros_(self.output.weight)

    def encode(self, raw, valid, onsets):
        if (raw.ndim != 4 or raw.shape[2:] != (2, self.input_dim) or raw.shape[1] == 0 or
                valid.shape != raw.shape[:2] or onsets.shape != valid.shape or
                valid.dtype != torch.bool or onsets.dtype != torch.bool or
                raw.device != valid.device or raw.device != onsets.device or
                not bool(valid[:, 0].all()) or bool((valid[:, 1:] & ~valid[:, :-1]).any()) or
                bool((onsets & ~valid).any())):
            raise ContractError('Long memory requires full BOS prefixes with aligned head flags and suffix padding')
        batch, length = raw.shape[:2]
        streams = (raw * valid[..., None, None]).permute(0, 2, 1, 3).reshape(batch * 2, length, self.input_dim)
        encoded, _ = self.encoder(streams)
        encoded = encoded.reshape(batch, 2, length, self.hidden).permute(0, 2, 1, 3)
        landmarks = onsets & (onsets.long().cumsum(1).remainder(self.stride) == 0)
        keys, values, positions = [], [], []
        for i in range(batch):
            at = torch.nonzero(landmarks[i], as_tuple=False).flatten()
            states = encoded[i].index_select(0, at)
            keys.append(self.key(states))
            values.append(self.value(states))
            positions.append(at)
        return MemoryBank(tuple(keys), tuple(values), tuple(positions))

    def read(self, hands, request: MemoryQuery):
        if (not isinstance(request, MemoryQuery) or hands.ndim != 3 or hands.shape[1:] != (2, self.output_dim) or
                request.batches.shape != (len(hands),) or request.rows.shape != (len(hands),) or
                request.batches.dtype != torch.long or request.rows.dtype != torch.long or
                request.batches.device != hands.device or request.rows.device != hands.device or
                bool((request.rows < 0).any()) or
                bool(((request.batches < 0) | (request.batches >= len(request.bank.keys))).any())):
            raise ContractError('Long-memory queries require aligned batch owners and physical prefix lengths')
        context = torch.zeros_like(hands)
        for batch in request.batches.unique().tolist():
            selected = torch.nonzero(request.batches == batch, as_tuple=False).flatten()
            keys, values, positions = (getattr(request.bank, name)[batch] for name in ('keys', 'values', 'positions'))
            if not len(positions):
                continue
            # A training bank may contain later landmarks. Exclude them before
            # normalization, including a landmark written by the current label.
            visible = positions[None] < request.rows.index_select(0, selected)[:, None]
            active = visible.any(1)
            selected, visible = selected[active], visible[active]
            if not len(selected):
                continue
            q = self.query(hands.index_select(0, selected))
            scores = torch.einsum('nhd,shd->nhs', q, keys) / math.sqrt(self.hidden)
            weights = scores.masked_fill(~visible[:, None], -torch.inf).softmax(-1)
            attended = torch.einsum('nhs,shd->nhd', weights, values)
            context = context.index_copy(0, selected, attended)
        return self.output(context)

    def empty_cache(self):
        like = self.output.weight
        bank = MemoryBank((like.new_empty(0, 2, self.hidden),),
                          (like.new_empty(0, 2, self.output_dim),),
                          (torch.empty(0, dtype=torch.long, device=like.device),))
        return MemoryCache(like.new_zeros(1, 2, self.hidden), bank)

    @torch.no_grad()
    def append(self, cache, raw, has_head):
        if raw.shape != (2, self.input_dim) or type(has_head) is not bool:
            raise ContractError('A long-memory append needs one physical row and its head-row role')
        encoded, hidden = self.encoder(raw[:, None], cache.hidden)
        onsets = cache.onsets + int(has_head)
        bank = cache.bank
        if has_head and onsets % self.stride == 0:
            state = encoded[:, 0][None]
            bank = MemoryBank((torch.cat((bank.keys[0], self.key(state)), 0),),
                              (torch.cat((bank.values[0], self.value(state)), 0),),
                              (torch.cat((bank.positions[0], bank.positions[0].new_tensor([cache.rows]))),))
        return MemoryCache(hidden, bank, cache.rows + 1, onsets)

    def cached_query(self, cache):
        positions = cache.bank.positions[0]
        return cache.bank.query(positions.new_zeros(1), positions.new_tensor([cache.rows]))
