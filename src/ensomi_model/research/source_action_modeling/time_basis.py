"""Physical and relative time coordinates derived from the supplied event skeleton.

No action type, attack rank, inferred BPM or corpus-fitted player prior enters
these coordinates. Synthetic markers are excluded from event adjacency.
"""
from dataclasses import dataclass

import torch
from torch import Tensor

from ..scoped_style_modeling.dataset import ContractError
from .tensors import ObservationTensors

TIME_SCALES_MS = (8., 16., 32., 64., 128., 256., 512., 1024., 2048., 4096.)
TIME_POLICY = "physical-seconds/asinh/smooth-8-4096ms/event-gap-ratios-v1"


def time_basis_dim(kind: str) -> int:
    if kind not in ("scalar", "smooth"):
        raise ContractError("Time basis must be scalar or smooth")
    return 2 if kind == "scalar" else 2 + 2 * len(TIME_SCALES_MS)


def time_basis(delta_ms: Tensor, kind: str) -> Tensor:
    """Retain seconds and asinh(seconds); optionally add smooth scale responses.

    At scale tau the signed channel is d / sqrt(d² + tau²), and the even
    channel is tau / sqrt(d² + tau²). Unlike periodic features, neither creates
    repeating aliases. Physical seconds remain available in the tails.
    """
    time_basis_dim(kind)
    seconds = delta_ms / 1000
    scalar = torch.stack((seconds, seconds.asinh()), -1)
    if kind == "scalar":
        return scalar
    ratio = delta_ms[..., None] / delta_ms.new_tensor(TIME_SCALES_MS)
    envelope = torch.rsqrt(1 + ratio.square())
    return torch.cat((scalar, ratio * envelope, envelope), -1)


def shift_events(values: Tensor, offset: int) -> Tensor:
    """At anchor i return source i+offset, zeroing out-of-range positions."""
    if offset == 0:
        return values
    indices = torch.arange(values.shape[1], device=values.device) + offset
    valid = (indices >= 0) & (indices < values.shape[1])
    shifted = values[:, indices.clamp(0, values.shape[1] - 1)]
    return shifted * valid.reshape(1, -1, *([1] * (values.ndim - 2)))


@dataclass(frozen=True)
class EventGeometry:
    """Compact source-event order with padding; indices retain timeline alignment."""
    indices: Tensor
    valid: Tensor
    times_ms: Tensor
    gaps_ms: Tensor
    gap_available: Tensor
    pace_ms: Tensor
    pace_available: Tensor

    def gather(self, values: Tensor) -> Tensor:
        batch = torch.arange(values.shape[0], device=values.device)[:, None]
        return values[batch, self.indices] * self.valid.reshape(*self.valid.shape, *([1] * (values.ndim - 2)))

    def restore(self, values: Tensor) -> Tensor:
        """Scatter compact events; synthetic markers and padding receive zero."""
        shape = (*self.indices.shape, *([1] * (values.ndim - 2)))
        return torch.zeros_like(values).scatter_add(
            1, self.indices.reshape(shape).expand_as(values), values * self.valid.reshape(shape))

    def row_time(self, kind: str) -> Tensor:
        basis = time_basis(self.gaps_ms, kind) * self.gap_available[..., None]
        ratios = (self.gaps_ms.clamp_min(1e-6) / self.pace_ms.clamp_min(1e-6)[..., None]).log()
        ratios = ratios * self.gap_available
        return torch.cat((basis.flatten(-2), ratios, self.gap_available.to(basis.dtype)), -1)

    def pair_time(self, offset: int, kind: str) -> Tensor:
        dt = shift_events(self.times_ms, offset) - self.times_ms
        pace = self.pace_ms.clamp_min(1e-6)
        other_pace = shift_events(self.pace_ms, offset).clamp_min(1e-6)
        other_known = shift_events(self.pace_available, offset)
        relative = torch.stack(((dt / pace).asinh() * self.pace_available,
                                (other_pace / pace).log() * (self.pace_available & other_known),
                                self.pace_available.to(dt.dtype), other_known.to(dt.dtype)), -1)
        return torch.cat((time_basis(dt, kind), relative), -1)


def event_geometry(observation: ObservationTensors) -> EventGeometry:
    """Use all supplied real events, including hidden and release-only rows.

    Local pace is the arithmetic mean of the available adjacent positive event
    gaps, so its time support extends one event on either side of the anchor.
    It is a skeleton statistic, not observed attack pace or musical beat time.
    """
    b, t = observation.rows.shape[:2]
    positions = torch.arange(t, device=observation.rows.device).expand(b, -1)
    source = observation.rows[..., 5].bool() & (positions < observation.lengths.to(positions.device)[:, None])
    indices = positions.masked_fill(~source, t).sort(1).values.clamp_max(t - 1)
    valid = positions < source.sum(1, keepdim=True)
    times = observation.times_ms[torch.arange(b, device=positions.device)[:, None], indices] * valid
    gaps = torch.stack((times - shift_events(times, -1), shift_events(times, 1) - times), -1)
    available = valid[..., None] & torch.stack((shift_events(valid, -1), shift_events(valid, 1)), -1)
    gaps = gaps * available
    positive = available & (gaps > 0)
    pace = (gaps * positive).sum(-1) / positive.sum(-1).clamp_min(1)
    return EventGeometry(indices, valid, times, gaps, available, pace, positive.any(-1))
