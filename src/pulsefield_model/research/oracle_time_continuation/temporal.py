"""Two-stream temporal attention with layer-input carry and atomic mean archives.

Step execution is the reference. Chunk execution builds the bounded union of
fine/coarse tokens, then evaluates each temporal layer in parallel over rows.
Both use the pre-commit bank for content construction; archiving becomes visible
only to subsequent queries. Query outputs never enter persistent memory.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
from torch import Tensor, nn

from .attention import MemoryAttention
from .config import BackboneConfig
from .features import TIME_DIM, clock_features

TEMPORAL_EDGE_DIM = 3 * TIME_DIM + 6


@dataclass(frozen=True)
class TokenPosition:
    first_id: int
    last_id: int
    start_ms: float
    end_ms: float
    count: int
    is_coarse: bool = False
    birth: int = 0
    eviction: int | None = None

    @property
    def key(self) -> tuple[int, int, bool]:
        return self.first_id, self.last_id, self.is_coarse

    @classmethod
    def row(cls, row_id: int, time_ms: float) -> TokenPosition:
        return cls(row_id, row_id, time_ms, time_ms, 1, birth=row_id)


@dataclass(frozen=True)
class TemporalToken:
    position: TokenPosition
    inputs: tuple[Tensor, ...]  # unnormalized layer inputs, each [2,D]
    projected: tuple[tuple[Tensor, Tensor], ...] | None = None  # each [2,H,D/H]

    def detached(self) -> TemporalToken:
        return replace(self, inputs=tuple(value.detach().clone() for value in self.inputs),
                       projected=None if self.projected is None else
                       tuple(tuple(value.detach().clone() for value in kv) for kv in self.projected))


@dataclass(frozen=True)
class TemporalState:
    row_count: int = 0
    recent: tuple[TemporalToken, ...] = ()
    coarse: tuple[TemporalToken, ...] = ()

    @property
    def tokens(self) -> tuple[TemporalToken, ...]:
        return self.coarse + self.recent

    def detached(self) -> TemporalState:
        return TemporalState(self.row_count, tuple(token.detached() for token in self.recent),
                             tuple(token.detached() for token in self.coarse))


@dataclass(frozen=True)
class TemporalTrace:
    """CPU metadata for auditing visibility; no learned tensor references."""

    candidates: tuple[TokenPosition, ...]
    query_ids: tuple[tuple[tuple[int, int, bool], ...], ...]
    content_ids: tuple[tuple[tuple[int, int, bool], ...], ...]


def archive_count(n: int, config: BackboneConfig) -> int:
    return max(0, n - config.recent) // config.coarse_group


def visible_at(position: TokenPosition, n: int, config: BackboneConfig, *, content: bool) -> bool:
    if position.count == 0:
        return n == 0 and not content
    if position.is_coarse:
        return position.birth <= n and n < position.eviction
    lower = config.coarse_group * archive_count(n, config)
    return lower < position.last_id <= n + int(content)


def coarse_position(block: int, fine: dict[int, TokenPosition], config: BackboneConfig) -> TokenPosition:
    first, last = (block - 1) * config.coarse_group + 1, block * config.coarse_group
    return TokenPosition(first, last, fine[first].start_ms, fine[last].end_ms, config.coarse_group, True,
                         config.recent + last, config.recent + (block + config.coarse_capacity) * config.coarse_group)


def temporal_edges(queries: tuple[TokenPosition, ...], candidates: tuple[TokenPosition, ...],
                   like: Tensor, config: BackboneConfig) -> Tensor:
    times, tags = [], []
    for query in queries:
        for token in candidates:
            bos = token.count == 0
            times.extend([None, None, None] if bos else
                         [query.end_ms - token.end_ms, query.end_ms - token.start_ms,
                          token.end_ms - token.start_ms])
            tags.append([math.asinh(query.last_id - token.first_id), math.asinh(query.last_id - token.last_id),
                         token.count / config.coarse_group, token.is_coarse, bos, token.first_id > 1])
    clocks = clock_features(times, like).reshape(len(queries), len(candidates), 3 * TIME_DIM)
    metadata = like.new_tensor(tags).reshape(len(queries), len(candidates), 6)
    return torch.cat((clocks, metadata), -1)[:, :, None].expand(-1, -1, 2, -1)


class TemporalEncoder(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(MemoryAttention(config.hidden, config.heads, TEMPORAL_EDGE_DIM)
                                    for _ in range(config.temporal_layers))
        self.bos = nn.Parameter(torch.zeros(config.temporal_layers, config.hidden))

    def _project(self, layer: int, inputs: list[Tensor],
                 cached: list[tuple[Tensor, Tensor] | None]) -> tuple[Tensor, Tensor]:
        pending = [index for index, kv in enumerate(cached) if kv is None]
        if len(pending) == len(inputs):
            return self.layers[layer].project(torch.stack(inputs))
        values = list(cached)
        if pending:
            projected = self.layers[layer].project(torch.stack([inputs[index] for index in pending]))
            for offset, index in enumerate(pending):
                values[index] = tuple(value[:, :, offset] for value in projected)
        return tuple(torch.stack([kv[side] for kv in values], dim=2) for side in range(2))

    def query(self, query: Tensor, state: TemporalState, time_ms: float) -> Tensor:
        """Pure pre-row read. Training projects raw history with current parameters."""
        position = (TokenPosition.row(state.row_count + 1, time_ms),)
        tokens = state.tokens
        positions = tuple(token.position for token in tokens) or (TokenPosition(0, 0, 0., 0., 0),)
        edges = temporal_edges(position, positions, query, self.config)
        mask = torch.ones(1, len(positions), dtype=torch.bool, device=query.device)
        for layer, attention in enumerate(self.layers):
            inputs = [token.inputs[layer] for token in tokens] or [self.bos[layer].expand(2, -1)]
            cached = [None if token.projected is None else token.projected[layer] for token in tokens] or [None]
            query = attention.read(query[None], self._project(layer, inputs, cached), edges, mask)[0]
        return query

    def commit(self, content: Tensor, state: TemporalState, time_ms: float, *, inference: bool) -> TemporalState:
        """Construct every layer against the old bank plus self, then archive/evict."""
        position = TokenPosition.row(state.row_count + 1, time_ms)
        tokens = state.tokens
        positions = tuple(token.position for token in tokens) + (position,)
        edges = temporal_edges((position,), positions, content, self.config)
        mask = torch.ones(1, len(positions), dtype=torch.bool, device=content.device)
        inputs, projections = [], []
        for layer, attention in enumerate(self.layers):
            inputs.append(content.clone())
            bank = [token.inputs[layer] for token in tokens] + [content]
            cached = [None if token.projected is None else token.projected[layer] for token in tokens] + [None]
            kv = self._project(layer, bank, cached)
            if inference:
                projections.append(tuple(value[:, :, -1].clone() for value in kv))
            content = attention.read(content[None], kv, edges, mask)[0]
        token = TemporalToken(position, tuple(inputs), tuple(projections) if inference else None)
        recent, coarse = state.recent + (token,), state.coarse
        if len(recent) == self.config.recent + self.config.coarse_group:
            group, recent = recent[:self.config.coarse_group], recent[self.config.coarse_group:]
            position = coarse_position(archive_count(token.position.last_id, self.config),
                                       {item.position.last_id: item.position for item in group}, self.config)
            means = tuple(torch.stack([item.inputs[layer] for item in group]).mean(0)
                          for layer in range(len(self.layers)))
            projected = tuple(tuple(value[:, :, 0].clone() for value in attention.project(mean[None]))
                              for attention, mean in zip(self.layers, means)) if inference else None
            coarse = (coarse + (TemporalToken(position, means, projected),))[-self.config.coarse_capacity:]
        return TemporalState(state.row_count + 1, recent, coarse)

    def chunk(self, queries: Tensor, contents: Tensor, state: TemporalState, times_ms: tuple[float, ...],
              *, inference: bool) -> tuple[Tensor, TemporalState, TemporalTrace]:
        """Evaluate a bounded dense chunk with per-call birth, eviction and self masks.

        Inputs are [Q,2,D]. The caller builds local/relation inputs in causal
        order. Carry retains owned row tensors; detach at an explicit TBPTT
        boundary, not here, so later losses can still train current writers.
        """
        rows = tuple(TokenPosition.row(state.row_count + index + 1, time) for index, time in enumerate(times_ms))
        n_final = state.row_count + len(rows)
        fine = tuple(token.position for token in state.recent) + rows
        fine_by_id = {position.last_id: position for position in fine}
        born = tuple(coarse_position(block, fine_by_id, self.config)
                     for block in range(archive_count(state.row_count, self.config) + 1,
                                        archive_count(n_final, self.config) + 1))
        coarse = tuple(token.position for token in state.coarse) + born
        bos = (TokenPosition(0, 0, 0., 0., 0),) if state.row_count == 0 else ()
        candidates = coarse + fine + bos
        query_visible = [[visible_at(token, row.last_id - 1, self.config, content=False) for token in candidates]
                         for row in rows]
        content_visible = [[visible_at(token, row.last_id - 1, self.config, content=True) for token in candidates]
                           for row in rows]
        query_mask = torch.tensor(query_visible, device=queries.device, dtype=torch.bool)
        content_mask = torch.tensor(content_visible, device=queries.device, dtype=torch.bool)
        edges = temporal_edges(rows, candidates, queries, self.config)
        carried = {token.position.key: token for token in state.tokens}
        layer_inputs = {position.key: [] for position in coarse + fine}
        projected_inputs = {position.key: [] for position in coarse + fine}
        for layer, attention in enumerate(self.layers):
            raw = {token.position.key: token.inputs[layer] for token in state.tokens}
            raw.update((row.key, contents[index].clone()) for index, row in enumerate(rows))
            for position in born:
                raw[position.key] = torch.stack([raw[fine_by_id[index].key]
                                                 for index in range(position.first_id, position.last_id + 1)]).mean(0)
            if bos:
                raw[bos[0].key] = self.bos[layer].expand(2, -1)
            cached = [carried[token.key].projected[layer]
                      if token.key in carried and carried[token.key].projected is not None else None
                      for token in candidates]
            kv = self._project(layer, [raw[token.key] for token in candidates], cached)
            for index, token in enumerate(candidates):
                if token.count:
                    layer_inputs[token.key].append(raw[token.key])
                    if inference:
                        projected_inputs[token.key].append(tuple(value[:, :, index].clone() for value in kv))
            queries = attention.read(queries, kv, edges, query_mask)
            contents = attention.read(contents, kv, edges, content_mask)

        def materialize(position):
            return TemporalToken(position, tuple(layer_inputs[position.key]),
                                 tuple(projected_inputs[position.key]) if inference else None)

        final = TemporalState(n_final,
                              tuple(materialize(p) for p in fine if visible_at(p, n_final, self.config, content=False)),
                              tuple(materialize(p) for p in coarse if visible_at(p, n_final, self.config, content=False)))
        trace = TemporalTrace(candidates,
                              tuple(tuple(p.key for p, visible in zip(candidates, mask) if visible)
                                    for mask in query_visible),
                              tuple(tuple(p.key for p, visible in zip(candidates, mask) if visible)
                                    for mask in content_visible))
        return queries, final, trace
