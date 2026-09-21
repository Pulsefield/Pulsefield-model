"""A finite causal content encoder with equivalent dense and cached execution.

One kernel-three convolution per dilation gives an exact token receptive field.
The feature owner may need one earlier timestamp for the oldest token's gap;
current exact clocks and object obligations are separate predictor inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..scoped_style_modeling.dataset import ContractError


@dataclass(frozen=True)
class TemporalConfig:
    input_dim: int
    hidden: int = 128
    levels: int = 8
    expansion: int = 4

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in vars(self).values()):
            raise ContractError('Finite temporal dimensions must be positive integers')
        if self.levels > 10:
            raise ContractError('Finite temporal levels are limited to ten')

    @property
    def dilations(self):
        return tuple(2 ** i for i in range(self.levels))

    @property
    def receptive_tokens(self):
        return 1 + 2 * sum(self.dilations)


def pointwise(module, values):
    projected = module(values.reshape(-1, values.shape[-1]))
    return projected.reshape(*values.shape[:-1], projected.shape[-1])


class CausalBlock(nn.Module):
    def __init__(self, width, dilation, expansion):
        super().__init__()
        self.dilation = dilation
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Conv1d(width, 2 * width, 3, dilation=dilation)
        self.mix = nn.Linear(width, width)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, expansion * width), nn.GELU(),
                                nn.Linear(expansion * width, width))

    def combine(self, values, gates):
        signal, gate = gates.chunk(2, dim=-1)
        values = values + pointwise(self.mix, signal.tanh() * gate.sigmoid())
        return values + pointwise(self.ff, self.ff_norm(values))

    def forward(self, values, valid):
        normalized = self.norm(values) * valid[..., None]
        convolved = self.conv(F.pad(normalized.transpose(1, 2), (2 * self.dilation, 0)))
        return self.combine(values, convolved.transpose(1, 2).contiguous()) * valid[..., None]

    def step(self, values, buffer):
        normalized = self.norm(values)
        taps = torch.stack((buffer[-2 * self.dilation], buffer[-self.dilation], normalized), dim=-1)
        # Selected taps already have the dilated spacing. The same three kernel
        # weights can therefore use dilation one for a single online position.
        gates = F.conv1d(taps, self.conv.weight, self.conv.bias)[:, :, 0]
        result = self.combine(values, gates)
        owned = torch.cat((buffer[1:], normalized[None]), dim=0).clone()
        return result, owned


@dataclass(frozen=True)
class TemporalCache:
    signature: tuple
    buffers: tuple[Tensor, ...]
    rows: int = 0
    last: Tensor | None = None
    truncated_start: bool = False


class FiniteTemporal(nn.Module):
    """Shared hand-coordinate content processing; queries never commit content.

    Dense input is [batch, physical_rows, two_hands, input_dim]. Padding may be
    outside each sample's contiguous materialized sequence, never an internal
    stand-in for a skipped candidate. Cache use is inference-only and bound to
    one parameter identity/version. Durable recovery should re-encode raw rows.
    """
    def __init__(self, config: TemporalConfig):
        super().__init__()
        self.config = config
        self.input = nn.Linear(config.input_dim, config.hidden)
        self.blocks = nn.ModuleList(CausalBlock(config.hidden, d, config.expansion) for d in config.dilations)
        self.boundary = nn.Parameter(torch.zeros(2, config.hidden))  # BOS, TRUNCATED

    def signature(self):
        return tuple((id(p), p._version, str(p.device), p.dtype) for p in self.parameters())

    def forward(self, raw: Tensor, valid: Tensor | None = None):
        if raw.ndim != 4 or raw.shape[2:] != (2, self.config.input_dim) or raw.shape[1] == 0:
            raise ContractError('Dense history requires nonempty [B,T,2,input_dim] content')
        if valid is None:
            valid = torch.ones(raw.shape[:2], dtype=torch.bool, device=raw.device)
        if valid.shape != raw.shape[:2] or valid.dtype != torch.bool or valid.device != raw.device:
            raise ContractError('Padding mask must match the materialized history batch')
        beginnings = valid & ~F.pad(valid[:, :-1], (1, 0), value=False)
        if bool((beginnings.sum(-1) > 1).any()):
            raise ContractError('Skipped candidates cannot occupy internal history positions')
        batch, rows = raw.shape[:2]
        mask = valid[:, None].expand(-1, 2, -1).reshape(2 * batch, rows)
        values = pointwise(self.input, raw).permute(0, 2, 1, 3).reshape(2 * batch, rows, self.config.hidden)
        values = values * mask[..., None]
        for block in self.blocks:
            values = block(values, mask)
        return values.reshape(batch, 2, rows, self.config.hidden).permute(0, 2, 1, 3).contiguous()

    def before(self, content: Tensor, valid: Tensor, *, truncated_start: Tensor):
        """Align pre-row history; BOS and truncated boundaries remain distinct."""
        if (content.ndim != 4 or content.shape[2:] != (2, self.config.hidden) or
                valid.shape != content.shape[:2] or valid.dtype != torch.bool or
                truncated_start.shape != (content.shape[0],) or truncated_start.dtype != torch.bool):
            raise ContractError('Pre-row alignment requires content, validity and one boundary flag per sample')
        previous = F.pad(content[:, :-1], (0, 0, 0, 0, 1, 0))
        begins = valid & ~F.pad(valid[:, :-1], (1, 0), value=False)
        boundary = self.boundary[truncated_start.long()][:, None, None].expand(-1, content.shape[1], 2, -1)
        return torch.where(begins[..., None, None], boundary, previous) * valid[..., None, None]

    def empty_cache(self, *, truncated_start=False):
        if type(truncated_start) is not bool:
            raise ContractError('History boundary status must be boolean')
        like = self.input.weight
        return TemporalCache(self.signature(), tuple(like.new_zeros(2 * d, 2, self.config.hidden)
                                                     for d in self.config.dilations), truncated_start=truncated_start)

    def _check(self, cache):
        if cache.signature != self.signature():
            raise ContractError('Learned cache belongs to a different model or parameter version')

    def read(self, cache):
        self._check(cache)
        return (self.boundary[int(cache.truncated_start)].expand(2, -1) if cache.last is None else cache.last)

    @torch.no_grad()
    def append(self, cache, raw):
        self._check(cache)
        if raw.shape != (2, self.config.input_dim):
            raise ContractError('Online content requires [2,input_dim] for one physical row')
        value = self.input(raw)
        buffers = []
        for block, buffer in zip(self.blocks, cache.buffers):
            value, owned = block.step(value, buffer)
            buffers.append(owned)
        return TemporalCache(cache.signature, tuple(buffers), cache.rows + 1, value.clone(), cache.truncated_start)
