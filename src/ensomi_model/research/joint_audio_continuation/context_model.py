"""Full-song audio context and bounded historical modulation of event timing.

The coarse branch reads audio only and is cached before native generation.
Its 500 ms cells do not restrict the native-millisecond output support. The
optional timing base can read active hold obligations but no content history.
"""
from dataclasses import dataclass, fields
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..bounded_typed_continuation.temporal import pointwise
from ..oracle_time_continuation.features import TIME_DIM
from ..scoped_style_modeling.dataset import ContractError
from .model import JointAudioModel, JointModelConfig
from .state import BASE_QUERY_DIM


@dataclass(frozen=True)
class ContextModelConfig(JointModelConfig):
    global_audio: bool = False
    bounded_timing: bool = False
    context_width: int = 128
    context_layers: int = 2
    context_heads: int = 4
    history_bound: float = 4.
    history_decay_ms: float = 1000.

    def __post_init__(self):
        JointModelConfig(**{f.name: getattr(self, f.name) for f in fields(JointModelConfig)})
        if type(self.global_audio) is not bool or type(self.bounded_timing) is not bool:
            raise ContractError('Context and bounded timing switches must be boolean')
        if (any(type(v) is not int or v <= 0 for v in
                (self.context_width, self.context_layers, self.context_heads)) or
                self.context_width % self.context_heads or self.context_width % 2):
            raise ContractError('Context width must be even and divisible by its positive head count')
        if any(not math.isfinite(v) or v <= 0 for v in (self.history_bound, self.history_decay_ms)):
            raise ContractError('Historical modulation needs a finite positive bound and decay time')

    @property
    def conditioned_audio_width(self):
        return self.audio_width + (self.context_width if self.global_audio else 0)


class FullSongContext(nn.Module):
    """Learned 50-frame reduction followed by bidirectional audio attention.

Inputs are normalized complete-song Mel and a real-frame prefix mask. The
partial last cell rescales its real samples; padded cells cannot become keys.
Token j is anchored at 250 + 500*j ms, including a partial final cell.
"""
    def __init__(self, config):
        super().__init__()
        self.reduce = nn.Conv1d(128, 128, 50, stride=50, groups=128, bias=False)
        nn.init.constant_(self.reduce.weight, 1 / 50)
        self.project = nn.Linear(128, config.context_width)
        self.layers = nn.ModuleList(nn.TransformerEncoderLayer(
            config.context_width, config.context_heads, 4 * config.context_width,
            dropout=0., activation='gelu', batch_first=True, norm_first=True)
            for _ in range(config.context_layers))
        self.norm = nn.LayerNorm(config.context_width)

    def forward(self, mel, valid):
        if (mel.ndim != 3 or mel.shape[-1] != 128 or not mel.shape[1] or
                valid.shape != mel.shape[:2] or valid.dtype != torch.bool or
                valid.device != mel.device or not bool(valid[:, 0].all()) or
                bool((~valid[:, :-1] & valid[:, 1:]).any())):
            raise ContractError('Global audio requires complete songs with contiguous real-frame prefixes')
        padding = (-mel.shape[1]) % 50
        real = F.pad(valid.to(mel.dtype), (0, padding))
        counts = real.reshape(len(mel), -1, 50).sum(-1)
        token_valid = counts > 0
        values = self.reduce(F.pad((mel * valid[..., None]).transpose(1, 2), (0, padding))).transpose(1, 2)
        values = pointwise(self.project, values * (50 / counts.clamp_min(1))[..., None])
        time = .25 + .5 * torch.arange(values.shape[1], dtype=values.dtype, device=values.device)
        frequency = torch.exp(-math.log(10000) * torch.arange(
            0, values.shape[-1], 2, dtype=values.dtype, device=values.device) / values.shape[-1])
        angles = time[:, None] * frequency
        position = torch.stack((angles.sin(), angles.cos()), -1).flatten(-2)
        values = (values + position) * token_valid[..., None]
        for layer in self.layers:
            values = layer(values, src_key_padding_mask=~token_valid) * token_valid[..., None]
        return self.norm(values) * token_valid[..., None], token_valid.sum(-1)


def coarse_at_frames(coarse, token_counts, frames, frame_counts):
    """Interpolate coarse context at real Mel-frame centers, clamping song edges.

Frames is [B,F] in full-song coordinates, possibly virtual crop coordinates.
The 500 ms coarse knots align with the 10 ms fine frame lattice, so subsequent
native-time interpolation agrees between a crop and the cached full encoding.
"""
    frames = torch.minimum(frames.clamp_min(0), frame_counts[:, None] - 1)
    position = ((20 + 10 * frames).to(coarse.dtype) - 250) / 500
    position = torch.minimum(position.clamp_min(0), (token_counts - 1)[:, None])
    low = position.floor().long()
    high = torch.minimum(low + 1, (token_counts - 1)[:, None])
    width = coarse.shape[-1]
    left = coarse.gather(1, low[..., None].expand(-1, -1, width))
    right = coarse.gather(1, high[..., None].expand(-1, -1, width))
    return left + (position - low)[..., None] * (right - left)


class ContextAudioModel(JointAudioModel):
    """Two independent research switches over the existing complete-row model.

With both switches off this model has the same parameters and function as
JointAudioModel. Full exact state remains available to the row/history paths.
Only the bounded timing base is restricted to occupancy and active-LN ages.
"""
    def __init__(self, config: ContextModelConfig):
        super().__init__(config)
        if config.bounded_timing:
            self.hold_projection = nn.Sequential(nn.Linear(4 * TIME_DIM + 4, config.hidden), nn.GELU())
            self.timing_base = nn.Sequential(nn.Linear(config.hidden + config.audio_width, config.hidden),
                                            nn.GELU(), nn.Linear(config.hidden, 10))
            nn.init.normal_(self.timing_base[-1].weight, std=.001)
            nn.init.constant_(self.timing_base[-1].bias, math.log(.006 / .994))
            nn.init.zeros_(self.timing[-1].bias)
            hold_indices = [i for lane in range(4) for i in range(3 * lane * TIME_DIM, (3 * lane + 1) * TIME_DIM)]
            hold_indices += list(range(22 * TIME_DIM + 16, 22 * TIME_DIM + 20))
            self.register_buffer('hold_indices', torch.tensor(hold_indices), persistent=False)
        if config.global_audio:
            # The coarse branch starts identically in fused/bounded cells;
            # constructing the extra timing base must not change its draw.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed() ^ 0x4C0A)
                self.context = FullSongContext(config)
            self.context_condition = nn.Linear(config.context_width, config.hidden, bias=False)
            self.context_timing = nn.Linear(config.context_width, config.hidden, bias=False)
            nn.init.zeros_(self.context_condition.weight)
            nn.init.zeros_(self.context_timing.weight)
            if config.bounded_timing:
                self.context_base = nn.Linear(config.context_width, config.hidden, bias=False)
                nn.init.zeros_(self.context_base.weight)

    def encode_coarse(self, mel, valid=None):
        """Encode full audio once, retaining gradients during joint training."""
        if not self.config.global_audio:
            return None
        if valid is None:
            valid = torch.ones(mel.shape[:2], dtype=torch.bool, device=mel.device)
        return self.context((mel - self.audio_mean) / self.audio_std, valid)

    def encode_crop(self, mel, valid, starts, frame_counts, coarse=None):
        """Combine a halo-complete local crop with context from its entire song.

Global mode requires explicitly supplied full-song context. A crop can never
silently stand in for the complete song. Local mode rejects unused context.
"""
        local = super().encode_audio(mel, valid)
        if not self.config.global_audio:
            if coarse is not None:
                raise ContractError('Local audio cannot consume an unused coarse context')
            return local
        if coarse is None:
            raise ContractError('Global conditioning requires the full-song coarse encoding')
        positions = starts[:, None] + torch.arange(mel.shape[1], device=mel.device)[None]
        global_values = coarse_at_frames(*coarse, positions, frame_counts)
        return torch.cat((local, global_values), -1)

    def encode_audio(self, mel, valid=None):
        """Encode complete songs for cached native generation; never use a crop."""
        if valid is None:
            valid = torch.ones(mel.shape[:2], dtype=torch.bool, device=mel.device)
        return self.encode_crop(mel, valid, torch.zeros(len(mel), dtype=torch.long, device=mel.device),
                                valid.sum(-1), self.encode_coarse(mel, valid))

    def condition(self, hands, exact, audio):
        if audio.shape[-1] != self.config.conditioned_audio_width:
            raise ContractError('Audio condition width differs from the local/global model contract')
        values = super().condition(hands, exact, audio[..., :self.config.audio_width])
        if self.config.global_audio:
            values = values + self.context_condition(audio[..., self.config.audio_width:]).unsqueeze(-2)
        return values

    def _history_timing(self, audio, history, exact):
        hands = history[:, None].expand(-1, audio.shape[1], -1, -1)
        condition = self.condition(hands, exact, audio)
        values = pointwise(self.timing[0], torch.cat((condition.mean(-2), audio[..., :self.config.audio_width]), -1))
        if self.config.global_audio:
            values = values + self.context_timing(audio[..., self.config.audio_width:])
        return pointwise(self.timing[2], self.timing[1](values))

    def timing_parts(self, audio, history, exact):
        """Return audio/hold base, bounded residual and gate for diagnostics.

The hold selector contains only active age encodings and occupancy. The
previous-row clock is decoded solely for the gate, never fed to the base.
Missing previous-row time means actual BOS and gives a zero gate.
"""
        if not self.config.bounded_timing:
            raise ContractError('Timing decomposition requires bounded timing mode')
        hold = pointwise(self.hold_projection, exact.index_select(-1, self.hold_indices)).mean(-2)
        values = pointwise(self.timing_base[0], torch.cat((hold, audio[..., :self.config.audio_width]), -1))
        if self.config.global_audio:
            values = values + self.context_base(audio[..., self.config.audio_width:])
        base = pointwise(self.timing_base[2], self.timing_base[1](values))
        occupied = exact[..., 0, 22 * TIME_DIM + 16:22 * TIME_DIM + 20].bool().any(-1)
        clock = exact[..., 0, 16 * TIME_DIM:17 * TIME_DIM]
        elapsed_ms = 1000 * torch.sinh(clock[..., 1]).clamp_min(0)
        free_gate = torch.exp(-elapsed_ms / self.config.history_decay_ms) * clock[..., -1]
        gate = torch.where(occupied, torch.ones_like(free_gate), free_gate)
        residual = self.config.history_bound * gate[..., None] * self._history_timing(audio, history, exact).tanh()
        return base, residual, gate

    def timing_logits(self, audio, history, exact):
        if (audio.ndim != 3 or audio.shape[-1] != self.config.conditioned_audio_width or
                history.shape != (audio.shape[0], 2, self.config.hidden) or
                exact.shape != (*audio.shape[:2], 2, BASE_QUERY_DIM)):
            raise ContractError('Timing requires aligned audio, historical hands and exact state')
        if self.config.bounded_timing:
            base, residual, _ = self.timing_parts(audio, history, exact)
            return base + residual
        return self._history_timing(audio, history, exact)
