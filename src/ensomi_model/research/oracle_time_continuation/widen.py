"""Double temporal width while retaining the trained FP32 row distribution.

Feature duplication preserves LayerNorm and the feed-forward function. Within
each attention head, duplicated Q/K channels each scale by 2**(-1/4), compensating
for SDPA's wider square-root denominator. Split outgoing weights sum to their
original values; zero-sum noise breaks training symmetry without changing the
initial function beyond floating-point tolerance. Replay all learned caches and
start a fresh optimizer after this transformation.
"""
from dataclasses import replace
import math

import torch

from ..scoped_style_modeling.dataset import ContractError
from .model import CausalBackbone


@torch.no_grad()
def widen_temporal(model: CausalBackbone, *, split_noise: float = .01, seed: int = 17) -> CausalBackbone:
    """Return an independent CPU model; input parameters and RNG remain unchanged.

    Only temporal expansion changes. The source must have an explicit temporal
    input projection and fixed bias-MLP width. Noise is relative to the source
    matrix RMS and lies in the duplicated-input null space.
    """
    cfg = model.config
    if cfg.temporal_expansion < 2 or cfg.temporal_bias_hidden is None:
        raise ContractError('Widening requires temporal_expansion>=2 and an explicit temporal_bias_hidden')
    if (isinstance(split_noise, bool) or not math.isfinite(split_noise) or split_noise < 0 or
            type(seed) is not int or not 0 <= seed < 2**63):
        raise ContractError('Widening requires finite nonnegative noise and an integer seed in [0,2**63)')
    if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
        raise ContractError('Temporal widening requires CPU FP32 source weights')
    generator = torch.Generator().manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        result = CausalBackbone(replace(cfg, temporal_expansion=2 * cfg.temporal_expansion))
    old, new = model.state_dict(), result.state_dict()
    width = cfg.temporal_hidden
    features = torch.arange(width).repeat(2)
    head_channels = torch.arange(width).reshape(cfg.heads, -1).repeat(1, 2).flatten()
    ff_channels = torch.arange(4 * width).repeat(2)

    def split_columns(value, indices):
        expanded = value[:, indices] / 2
        if split_noise:
            # Two occurrences of each source column receive opposite noise.
            order = torch.argsort(indices, stable=True).reshape(-1, 2)
            noise = torch.randn(value.shape, generator=generator) * (split_noise * value.square().mean().sqrt())
            expanded[:, order[:, 0]] += noise
            expanded[:, order[:, 1]] -= noise
        return expanded

    for name, target in new.items():
        value = old[name]
        if name == 'temporal.input_projection.weight' or name == 'temporal.input_projection.bias':
            mapped = value[features]
        elif name == 'temporal.bos':
            mapped = value[:, features]
        elif name.startswith('temporal.layers.'):
            suffix = name.split('.', 3)[3]
            if suffix.startswith('bias.'):
                mapped = value
            elif suffix in ('norm.weight', 'norm.bias', 'ff_norm.weight', 'ff_norm.bias', 'output.bias', 'ff.2.bias'):
                mapped = value[features]
            elif suffix in ('query.weight', 'key.weight', 'value.weight'):
                mapped = split_columns(value[head_channels], features)
                if suffix != 'value.weight':
                    mapped = mapped * 2**(-.25)
            elif suffix == 'output.weight':
                mapped = split_columns(value[features], head_channels)
            elif suffix == 'ff.0.weight':
                mapped = split_columns(value[ff_channels], features)
            elif suffix == 'ff.0.bias':
                mapped = value[ff_channels]
            elif suffix == 'ff.2.weight':
                mapped = split_columns(value[features], ff_channels)
            else:
                raise ContractError(f'Unsupported temporal parameter during widening: {name}')
        elif name in ('head.unary.weight', 'head.interaction.weight'):
            mapped = split_columns(value, features)
        else:
            mapped = value
        if mapped.shape != target.shape:
            raise ContractError(f'Widening parameter shape mismatch: {name}')
        target.copy_(mapped)
    result.train(model.training)
    return result
