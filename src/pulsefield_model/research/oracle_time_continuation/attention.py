"""Shared pre-normalized attention with explicit visibility and no time statistics."""
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MemoryAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, edge_dim: int, *, bias_hidden: int | None = None):
        super().__init__()
        self.heads = heads
        self.width = hidden // heads
        self.norm = nn.LayerNorm(hidden)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        bias_hidden = hidden if bias_hidden is None else bias_hidden
        self.bias = nn.Sequential(nn.Linear(edge_dim, bias_hidden), nn.GELU(), nn.Linear(bias_hidden, heads))
        self.output = nn.Linear(hidden, hidden)
        self.ff_norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden))

    def split(self, value: Tensor) -> Tensor:
        # [events, hands, channels] -> [hands, heads, events, head_channels]
        return value.reshape(value.shape[0], 2, self.heads, self.width).permute(1, 2, 0, 3)

    def project(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        normalized = self.norm(inputs)
        return self.split(self.key(normalized)), self.split(self.value(normalized))

    def read(self, queries: Tensor, kv: tuple[Tensor, Tensor], edges: Tensor, visible: Tensor) -> Tensor:
        """Read [Q,2,D] queries; edges are [Q,K,2,E], visibility is [Q,K].

        Every real query must have at least one key (BOS for empty history).
        Padding is excluded by the caller, never represented as a row/key.
        """
        bias = self.bias(edges).permute(2, 3, 0, 1)
        mask = bias.masked_fill(~visible[None, None], -torch.inf)
        values = F.scaled_dot_product_attention(self.split(self.query(self.norm(queries))), *kv,
                                               attn_mask=mask, dropout_p=0., is_causal=False)
        values = values.permute(2, 0, 1, 3).reshape_as(queries)
        output = queries + self.output(values)
        return output + self.ff(self.ff_norm(output))
