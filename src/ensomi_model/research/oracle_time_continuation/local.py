"""Three time-conditioned causal convolutions over committed complete rows."""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from .features import HistoryStatus, PaceState, TIME_DIM, action_features, clock_features, status_features
from .schema import CompleteRow

DILATIONS = (1, 2, 4)


@dataclass(frozen=True)
class LocalEntry:
    row_id: int
    row: CompleteRow
    value: Tensor  # [hand, channel], owned by this entry
    support: tuple[tuple[int, float], ...]

    @property
    def count(self) -> int:
        return len(self.support)

    @property
    def span_ms(self) -> float:
        return self.support[-1][1] - self.support[0][1]

    @property
    def status(self) -> HistoryStatus:
        return HistoryStatus.TRUNCATED if self.support[0][0] > 1 else HistoryStatus.PRESENT

    def detached(self) -> LocalEntry:
        return replace(self, value=self.value.detach().clone())


@dataclass(frozen=True)
class LocalState:
    # Each layer retains only its last 2*d inputs; outputs are not predecessors
    # at the same layer. The latest outputs expose the three separate scales.
    buffers: tuple[tuple[LocalEntry, ...], ...] = ((), (), ())
    latest: tuple[LocalEntry, ...] = ()
    pace: PaceState = PaceState()

    def detached(self) -> LocalState:
        return LocalState(tuple(tuple(entry.detached() for entry in buf) for buf in self.buffers),
                          tuple(entry.detached() for entry in self.latest), self.pace)


class CausalLocalLayer(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.maps = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(3))
        self.gates = nn.ModuleList(nn.Sequential(nn.Linear(TIME_DIM + 32, hidden), nn.GELU(),
                                                nn.Linear(hidden, hidden)) for _ in range(3))
        self.metadata = nn.Linear(TIME_DIM + 5, hidden)
        self.update = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden))

    def forward(self, current: LocalEntry, predecessors: tuple[LocalEntry, ...], dilation: int) -> LocalEntry:
        inputs = predecessors + (current,)
        total = torch.zeros_like(current.value)
        support = {}
        for slot, offset in enumerate((0, dilation, 2 * dilation)):
            if offset >= len(inputs):
                continue
            previous = inputs[-1 - offset]
            edge = torch.cat((clock_features([current.row.time_ms - previous.row.time_ms], current.value)
                              .expand(2, -1), action_features(current.row.actions, current.value),
                              action_features(previous.row.actions, current.value)), -1)
            total = total + (1 + self.gates[slot](edge).tanh()) * self.maps[slot](self.norm(previous.value))
            support.update(previous.support)
        owned_support = tuple(sorted(support.items()))
        status = HistoryStatus.TRUNCATED if owned_support[0][0] > 1 else HistoryStatus.PRESENT
        metadata = torch.cat((clock_features([owned_support[-1][1] - owned_support[0][1]], total)[0],
                              total.new_tensor([len(owned_support)]), status_features(status, total)))
        value = current.value + self.update(total + self.metadata(metadata))
        return LocalEntry(current.row_id, current.row, value, owned_support)


class LocalEncoder(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.layers = nn.ModuleList(CausalLocalLayer(hidden) for _ in DILATIONS)
        self.boundary = nn.Parameter(torch.zeros(hidden))
        self.fusion = nn.Sequential(nn.Linear(4 * hidden + 3 * (TIME_DIM + 5), hidden), nn.GELU(),
                                    nn.Linear(hidden, hidden))

    def commit(self, state: LocalState, row: CompleteRow, row_id: int, raw: Tensor,
               gap_ms: float | None) -> LocalState:
        current = LocalEntry(row_id, row, raw.clone(), ((row_id, row.time_ms),))
        buffers, latest = [], []
        for layer, dilation, previous in zip(self.layers, DILATIONS, state.buffers):
            buffers.append((previous + (current,))[-2 * dilation:])
            current = layer(current, previous, dilation)
            latest.append(current)
        return LocalState(tuple(buffers), tuple(latest), state.pace.commit(gap_ms))

    def read(self, raw_facts: Tensor, state: LocalState) -> Tensor:
        values, metadata = [], []
        for level in range(3):
            if state.latest:
                entry = state.latest[level]
                values.append(entry.value)
                time, count, status = entry.span_ms, entry.count, entry.status
            else:
                values.append(self.boundary.expand(2, -1))
                time, count, status = None, 0, HistoryStatus.BOS
            metadata.append(torch.cat((clock_features([time], raw_facts)[0], raw_facts.new_tensor([count]),
                                       status_features(status, raw_facts))).expand(2, -1))
        return self.fusion(torch.cat((raw_facts, *values, *metadata), -1))
