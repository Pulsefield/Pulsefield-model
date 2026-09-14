"""Controlled row, time and local-composition interventions over the retained bank.

U contains current-row facts and supplied time only. Optional propagated state
conditions enter local blocks separately and invalidate unconditional locality.
The contextual mixer, relation graph and both reader contracts stay shared.
"""
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..scoped_style_modeling.dataset import ContractError
from .model import ModelConfig
from .representation import ComposedEncoder, LocalBlock, valid_rows
from .tensors import ObservationTensors, ROW_DIM
from .time_basis import EventGeometry, event_geometry, shift_events, time_basis_dim

LOCAL_POLICY = "row-packet/source-event/conditioned-residual-v1"
RAW_DIM = 16 + ROW_DIM


@dataclass(frozen=True)
class LocalRepresentationConfig:
    """Independent interventions; defaults reproduce the timeline-convolution bank.

    `legacy` retains the original row coordinates. `scalar` adds physical event
    gaps and ratios; `smooth` expands the same information into smooth bases.
    `state_condition` reads visible-prefix occupation, with prefix support.
    """
    time_basis: str = "legacy"
    row_interaction: bool = False
    local_operator: str = "timeline"
    kernel_condition: str = "time_action"
    source_skip: bool = False
    state_condition: bool = False

    def __post_init__(self):
        if self.time_basis not in ("legacy", "scalar", "smooth"):
            raise ContractError("Row time basis must be legacy, scalar or smooth")
        if self.local_operator not in ("timeline", "event", "time"):
            raise ContractError("Local operator must be timeline, event or time")
        if self.kernel_condition not in ("time_action", "time", "action", "constant"):
            raise ContractError("Kernel condition must be time_action, time, action or constant")
        if self.local_operator != "time" and self.kernel_condition != "time_action":
            raise ContractError("Kernel conditioning controls require the time local operator")
        if any(type(v) is not bool for v in (self.row_interaction, self.source_skip, self.state_condition)):
            raise ContractError("Representation switches require boolean values")
        if self.local_operator == "time" and self.time_basis == "legacy":
            raise ContractError("Time-conditioned local operators require a scalar or smooth time basis")


@dataclass(frozen=True)
class SourcePacket:
    """Timeline-aligned facts, time coordinates and separately sourced state.

    row_facts has own/other action bits and availability. state_before contains
    own/other occupation value and availability from the visible prefix. It has
    no endpoint, duration, previous-attack or next-attack descriptor.
    """
    row_facts: Tensor
    row_metadata: Tensor
    time: Tensor | None
    state_before: Tensor

    @property
    def raw(self) -> Tensor:
        return torch.cat((self.row_facts, self.row_metadata), -1)


def source_packet(observation: ObservationTensors, geometry: EventGeometry,
                  config: LocalRepresentationConfig) -> SourcePacket:
    valid = valid_rows(observation)
    intrinsic = observation.lanes[..., :4].flatten(-2)
    facts = torch.cat((intrinsic, intrinsic.flip(2)), -1)
    metadata = observation.rows.clone()
    if config.local_operator != "timeline":
        # The preceding source gap cannot change when a synthetic row is inserted.
        gaps = geometry.restore((geometry.gaps_ms[..., :1] / 1000).log1p()).squeeze(-1)
        metadata[..., 0] = gaps
    rows = metadata[:, :, None].expand(-1, -1, 2, -1)
    before = observation.lanes[..., 4:6].flatten(-2)
    state = torch.cat((before, before.flip(2)), -1)
    timing = None if config.time_basis == "legacy" else geometry.restore(geometry.row_time(config.time_basis))
    mask = valid[:, :, None, None]
    return SourcePacket(facts * mask, rows * mask, timing, state * mask)


class ConditionedLocalBlock(LocalBlock):
    """Offset-specific value maps modulated by pair time and visible endpoint facts.

    No normalization statistic spans rows. Raw source/state drives bypass the
    content LayerNorm. Gates can change sign; the neighborhood remains bounded.
    """
    def __init__(self, dilation: int, dropout: float, config: LocalRepresentationConfig):
        super().__init__(dilation, dropout)
        self.dilation, self.config = dilation, config
        self.condition = None
        if config.local_operator == "time":
            # Both full endpoint rows preserve lane/hand roles and unknowns.
            self.condition = nn.Sequential(nn.Linear(time_basis_dim(config.time_basis) + 4 + 32, 32),
                                           nn.GELU(), nn.Linear(32, 128))
        self.raw = nn.Linear(RAW_DIM, 128, bias=False) if config.source_skip else None
        self.state = nn.Linear(8, 128, bias=False) if config.state_condition else None

    def forward(self, x: Tensor, valid: Tensor, *, packet: SourcePacket,
                geometry: EventGeometry) -> Tensor:
        mask = valid[:, :, None, None]
        normalized = self.norm(x) * mask
        drive = self.convolution.bias[None, None, None].expand(*x.shape[:-1], 128)
        for tap, offset in enumerate((-self.dilation, 0, self.dilation)):
            neighbor_valid = shift_events(valid, offset) & valid
            neighbor = shift_events(normalized, offset)
            value = F.linear(neighbor, self.convolution.weight[:, :, tap])
            if self.condition is not None:
                timing = geometry.pair_time(offset, self.config.time_basis)[:, :, None].expand(-1, -1, 2, -1)
                endpoints = torch.cat((packet.row_facts, shift_events(packet.row_facts, offset)), -1)
                if self.config.kernel_condition in ("action", "constant"):
                    timing = torch.zeros_like(timing)
                if self.config.kernel_condition in ("time", "constant"):
                    endpoints = torch.zeros_like(endpoints)
                modulation = 1 + 2 * self.condition(torch.cat((timing, endpoints), -1)).tanh()
                value = value * modulation
            drive = drive + value * neighbor_valid[:, :, None, None]
        if self.raw is not None:
            drive = drive + self.raw(packet.raw)
        if self.state is not None:
            drive = drive + self.state(packet.state_before)
        values, gates = drive.chunk(2, -1)
        return (x + self.dropout(self.output(values.tanh() * gates.sigmoid()))) * mask


class TimeLocalEncoder(ComposedEncoder):
    """Retain U/L1/L2/L3 while changing only the configured early computation."""
    def __init__(self, config: ModelConfig, representation: LocalRepresentationConfig):
        super().__init__(config)
        self.representation = representation
        self.source_time = None if representation.time_basis == "legacy" else nn.Linear(
            2 * time_basis_dim(representation.time_basis) + 4, 64, bias=False)
        self.row_interaction = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64)) if (
            representation.row_interaction) else None
        if representation.local_operator == "time" or representation.source_skip or representation.state_condition:
            self.local = nn.ModuleList(ConditionedLocalBlock(d, config.dropout, representation) for d in (1, 2, 4))

    def local_states(self, observation: ObservationTensors) -> list[Tensor]:
        geometry = event_geometry(observation)
        packet = source_packet(observation, geometry, self.representation)
        valid = valid_rows(observation)
        mask = valid[:, :, None, None]
        u = self.source(packet.raw)
        if self.source_time is not None:
            u = u + self.source_time(packet.time)[:, :, None]
        if self.row_interaction is not None:
            u = u + self.row_interaction(u)
        u = u * mask
        states = [u]
        compact = self.representation.local_operator != "timeline"
        local_valid = geometry.valid if compact else valid
        local_packet = SourcePacket(*(geometry.gather(v) if compact and v is not None else v
                                      for v in (packet.row_facts, packet.row_metadata, packet.time, packet.state_before)))
        values = geometry.gather(u) if compact else u
        source_mask = (observation.rows[..., 5].bool() & valid)[:, :, None, None]
        for block in self.local:
            if isinstance(block, ConditionedLocalBlock):
                values = block(values, local_valid, packet=local_packet, geometry=geometry)
            else:
                values = block(values, local_valid)
            # Boundaries remain addressable U metadata, without consuming an event step.
            states.append(torch.where(source_mask, geometry.restore(values), u) if compact else values)
        return states


def local_support_report(observation, config: LocalRepresentationConfig) -> dict:
    """Conservative early-level dependency bounds, separating action/time/state.

    Scope/context endpoints are supplied conditions for every U. Optional state
    carries the entire visible-prefix dependency envelope and context-entry
    occupation; it is never included in a claim of strict action locality.
    """
    rows = observation.rows
    source = [i for i, r in enumerate(rows) if r.phase == "source"]
    ordered = list(range(len(rows))) if config.local_operator == "timeline" else source
    neighbors = {i: {i} for i in range(len(rows))}
    time_support = {i: {i} for i in range(len(rows))}
    for k, i in enumerate(source):
        if config.local_operator != "timeline" and k:
            time_support[i].add(source[k - 1])
        if config.time_basis != "legacy":
            time_support[i].update(source[max(0, k - 1):k + 2])
    if config.local_operator == "timeline":
        for i in range(1, len(rows)):
            time_support[i].add(i - 1)
    state_support = {i: set() for i in range(len(rows))}
    result = {}

    def record(level):
        result[level] = [{"action_row_indices": sorted(neighbors[i]),
                          "time_row_indices": sorted(time_support[i]),
                          "state_condition_row_indices": sorted(state_support[i]),
                          "row_indices": sorted(neighbors[i] | state_support[i]),
                          "action_source_events": sum(rows[j].phase == "source" for j in neighbors[i]),
                          "action_duration_ms": rows[max(neighbors[i])].time_ms - rows[min(neighbors[i])].time_ms,
                          "scope_context_condition": True,
                          "entering_occupancy_condition": config.state_condition and level != "U" and i in ordered}
                         for i in range(len(rows))]
    record("U")
    for level, dilation in zip(("L1", "L2", "L3"), (1, 2, 4)):
        adjacent = {i: [ordered[k + o] for o in (-dilation, 0, dilation) if 0 <= k + o < len(ordered)]
                    for k, i in enumerate(ordered)}
        neighbors = {i: set().union(*(neighbors[j] for j in adjacent.get(i, [i]))) for i in neighbors}
        time_support = {i: set().union(*(time_support[j] for j in adjacent.get(i, [i]))) for i in time_support}
        state_support = {i: set().union(*(state_support[j] for j in adjacent.get(i, [i])),
                                        set(range(i)) if config.state_condition and i in ordered else set())
                         for i in state_support}
        record(level)
    return result
