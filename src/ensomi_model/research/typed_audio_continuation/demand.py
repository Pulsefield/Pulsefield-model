"""Audio-conditioned nominal head demand and finite count feedback.

Demand is a mean activity prediction, not a quota or a difficulty readout.
Only skeleton head counts enter feedback; releases remain independently timed.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .controls import SHARED_SCOPE


def pool_audio(encoded, frames=None):
    """Pool the cached audio-only frame representation into 500-ms cells.

The last partial cell averages only real frames. These cells describe demand;
they do not quantize generated timestamps or reset gameplay history.
"""
    values = encoded[:frames] if frames is not None else encoded
    count = len(values)
    values = F.pad(values, (0, 0, 0, (-count) % 50)).reshape(-1, 50, values.shape[-1])
    denominator = torch.full((len(values),), 50, device=values.device, dtype=values.dtype)
    denominator[-1] = count-50*(len(values)-1)
    return values.sum(1)/denominator[:, None]


class AudioDemand(nn.Module):
    """Predict head objects/second from full-audio features and actual controls.

For fixed audio and other controls, known stars have a nonnegative effect on
log mean demand. This constrains the nominal mean, not individual arrangements.
"""
    def __init__(self, audio_width, control_width, star_known_index, hidden=96,
                 control_encoding=SHARED_SCOPE):
        super().__init__()
        self.config = dict(audio_width=audio_width, control_width=control_width,
                           star_known_index=star_known_index, hidden=hidden,
                           control_encoding=control_encoding)
        self.register_buffer('audio_mean', torch.zeros(audio_width))
        self.register_buffer('audio_std', torch.ones(audio_width))
        self.query = nn.Sequential(nn.Linear(audio_width+control_width, hidden), nn.GELU(),
                                   nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.query[-1].weight)
        with torch.no_grad():
            self.query[-1].bias.copy_(torch.tensor([math.log(8), math.log(math.expm1(.25))]))

    def forward(self, audio, control):
        """Return log nominal head rate; unknown stars use the learned base."""
        residual = control.clone()
        residual[..., 0] = 0
        values = self.query(torch.cat(((audio-self.audio_mean)/self.audio_std, residual), -1))
        stars = 2*control[..., 0]*control[..., self.config['star_known_index']]
        return values[..., 0]+F.softplus(values[..., 1])*stars


@dataclass(frozen=True)
class DemandCurve:
    edges_ms: np.ndarray
    rates: np.ndarray
    discounted: np.ndarray
    memory_ms: float

    @classmethod
    def from_rates(cls, edges_ms, rates, memory_ms):
        edges, rates = np.asarray(edges_ms, float), np.asarray(rates, float)
        discounted = np.zeros(len(edges))
        for i, delta in enumerate(np.diff(edges)):
            decay = math.exp(-delta/memory_ms)
            discounted[i+1] = decay*discounted[i]+(-math.expm1(-delta/memory_ms))*memory_ms/1000*rates[i]
        return cls(edges, rates, discounted, memory_ms)

    @classmethod
    @torch.inference_mode()
    def build(cls, model, encoded, controls, duration_ms, memory_ms):
        features = pool_audio(encoded)
        edges = np.array(sorted({0, duration_ms, *range(0, duration_ms, 500),
            *(t for span in controls.spans for t in (span.start_ms, span.end_ms) if 0 < t < duration_ms)}))
        device = next(model.parameters()).device
        index = torch.as_tensor(np.minimum(edges[:-1]//500, len(features)-1), device=features.device)
        audio = features[index].to(device)
        condition = torch.as_tensor(controls.at(edges[:-1], encoding=model.config['control_encoding']), device=device)
        rates = model(audio, condition).exp().cpu().numpy()
        return cls.from_rates(edges, rates, memory_ms)

    def mass_at(self, times):
        """Discounted desired count from audio start through each query time."""
        times = np.clip(np.asarray(times), self.edges_ms[0], self.edges_ms[-1])
        index = np.clip(np.searchsorted(self.edges_ms, times, side='right')-1, 0, len(self.rates)-1)
        elapsed = (times-self.edges_ms[index])/self.memory_ms
        return np.exp(-elapsed)*self.discounted[index]-np.expm1(-elapsed)*self.memory_ms/1000*self.rates[index]


@dataclass(frozen=True)
class DemandBalance:
    time_ms: int = 0
    count: float = 0.

    def mass_at(self, times, memory_ms):
        return self.count*np.exp(-(np.asarray(times)-self.time_ms)/memory_ms)

    def advance(self, now, heads, memory_ms):
        return DemandBalance(now, float(self.mass_at(now, memory_ms))+heads)


@dataclass(frozen=True)
class DemandFeedback:
    memory_ms: float = 4000.
    pseudocount_heads: float = 4.
    strength: float = 2.
    maximum_log_odds: float = 2.

    def shift(self, balance, curve, times):
        actual = balance.mass_at(times, self.memory_ms)
        desired = curve.mass_at(times)
        shift = self.strength*np.log((desired+self.pseudocount_heads)/(actual+self.pseudocount_heads))
        return np.clip(shift, -self.maximum_log_odds, self.maximum_log_odds)
