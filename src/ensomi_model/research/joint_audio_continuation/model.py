"""Shared local Mel conditioning for next-event time and complete action rows.

Audio uses finite symmetric context; generated content uses R1's finite causal
encoder. Timing and row queries consume the same exact historical state, without
future candidates, observed seed endpoints, or supplied onset roles.
"""
from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from ..bounded_typed_continuation.contract import ROW_ACTIONS
from ..bounded_typed_continuation.features import CONTENT_DIM
from ..bounded_typed_continuation.model import JointHead
from ..bounded_typed_continuation.routing import HeadRouting, ReleaseRouting
from ..bounded_typed_continuation.temporal import FiniteTemporal, TemporalConfig, pointwise
from ..scoped_style_modeling.dataset import ContractError
from .state import BASE_QUERY_DIM

R1_TRANSFER_MODULES = ('temporal', 'exact', 'fuse', 'joint', 'route_residual', 'release_residual')


@dataclass(frozen=True)
class JointModelConfig:
    hidden: int = 128
    audio_width: int = 96
    audio_levels: int = 6
    history_levels: int = 8
    expansion: int = 4
    coupling_rank: int = 16
    routing_hidden: int = 512
    release_hidden: int = 512

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in vars(self).values()):
            raise ContractError('Joint model dimensions must be positive integers')
        if self.audio_levels > 10 or self.history_levels > 8:
            raise ContractError('Joint audio levels are at most ten and history levels at most eight')

    @property
    def audio_dilations(self):
        return tuple(2 ** i for i in range(self.audio_levels))

    @property
    def audio_halo_frames(self):
        return 2 * sum(self.audio_dilations)


class AudioBlock(nn.Module):
    def __init__(self, width, dilation, expansion):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Conv1d(width, width, 5, padding=2 * dilation,
                              dilation=dilation, groups=width)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, expansion * width), nn.GELU(),
                                nn.Linear(expansion * width, width))

    def forward(self, values, valid):
        normalized = self.norm(values) * valid[..., None]
        values = values + self.conv(normalized.transpose(1, 2)).transpose(1, 2)
        return (values + pointwise(self.ff, self.ff_norm(values))) * valid[..., None]


class JointAudioModel(nn.Module):
    """Trainable next-time hazard and legal row distribution from shared inputs.

Timing query q denotes one absolute 10 ms bin and returns its ten native-ms
hazards. The caller evaluates exact clocks at that bin's final millisecond and
uses the same audio frame regardless of how inference partitions its queries.
Row queries use the selected event time. Neither query changes content history.
"""
    def __init__(self, config: JointModelConfig):
        super().__init__()
        self.config = config
        self.register_buffer('audio_mean', torch.zeros(128))
        self.register_buffer('audio_std', torch.ones(128))
        self.audio_input = nn.Linear(128, config.audio_width)
        self.audio_blocks = nn.ModuleList(AudioBlock(config.audio_width, d, config.expansion)
                                          for d in config.audio_dilations)
        self.temporal = FiniteTemporal(TemporalConfig(CONTENT_DIM, config.hidden,
                                                      config.history_levels, config.expansion))
        self.exact = nn.Sequential(nn.Linear(BASE_QUERY_DIM, config.hidden), nn.GELU(),
                                   nn.Linear(config.hidden, config.hidden))
        self.fuse = nn.Sequential(nn.LayerNorm(2 * config.hidden),
                                  nn.Linear(2 * config.hidden, config.hidden), nn.GELU(),
                                  nn.Linear(config.hidden, config.hidden))
        self.audio_residual = nn.Linear(config.audio_width, config.hidden, bias=False)
        self.joint = JointHead(config.hidden, 4, config.coupling_rank)
        self.route_residual = HeadRouting(config.hidden, config.routing_hidden)
        self.release_residual = ReleaseRouting(config.hidden, config.release_hidden)
        self.timing = nn.Sequential(nn.Linear(config.hidden + config.audio_width, config.hidden),
                                    nn.GELU(), nn.Linear(config.hidden, 10))
        nn.init.normal_(self.timing[-1].weight, std=.001)
        nn.init.constant_(self.timing[-1].bias, math.log(.006 / .994))
        self.register_buffer('has_head', torch.tensor([any(a in (1, 2) for a in row)
                                                       for row in ROW_ACTIONS]), persistent=False)
        self.register_buffer('nonempty', torch.tensor([any(row) for row in ROW_ACTIONS]), persistent=False)

    @property
    def audio_halo_frames(self):
        return self.config.audio_halo_frames

    @torch.no_grad()
    def set_audio_normalization(self, mean: Tensor, std: Tensor):
        """Copy finite per-bin training statistics; reject nonpositive scales."""
        if (mean.shape != (128,) or std.shape != (128,) or
                not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()) or
                not bool((std > 0).all())):
            raise ContractError('Audio normalization requires 128 finite means and positive scales')
        self.audio_mean.copy_(mean)
        self.audio_std.copy_(std)

    def encode_audio(self, mel: Tensor, valid: Tensor | None = None):
        """Encode [B,T,128] without downsampling or frequency pooling.

        Use complete audio or include audio_halo_frames on both sides of a
        requested crop. Validity marks real song frames; virtual edge padding
        stays zero through every block. Artificial training-crop edges must
        remain outside the scored region.
        """
        if mel.ndim != 3 or mel.shape[-1] != 128 or mel.shape[1] == 0:
            raise ContractError('Audio encoding requires nonempty [B,T,128] Mel frames')
        if valid is None:
            valid = torch.ones(mel.shape[:2], dtype=torch.bool, device=mel.device)
        if valid.shape != mel.shape[:2] or valid.dtype != torch.bool or valid.device != mel.device:
            raise ContractError('Audio validity must match the Mel batch and frame axes')
        values = pointwise(self.audio_input, (mel - self.audio_mean) / self.audio_std) * valid[..., None]
        for block in self.audio_blocks:
            values = block(values, valid)
        return values

    def encode_history(self, raw: Tensor, valid: Tensor, truncated: Tensor):
        """Return the last real row encoding, or the BOS/truncated boundary.

Each sample contains a contiguous valid interval and at most one receptive
field of raw physical rows. The oldest row retains its original predecessor
gap. Padding is never a synthetic empty event; zero-length prefixes are legal.
        """
        if (raw.ndim != 4 or raw.shape[2:] != (2, CONTENT_DIM) or
                raw.shape[1] > self.temporal.config.receptive_tokens or
                valid.shape != raw.shape[:2] or valid.dtype != torch.bool or
                truncated.shape != (raw.shape[0],) or truncated.dtype != torch.bool or
                valid.device != raw.device or truncated.device != raw.device):
            raise ContractError('History requires bounded raw rows, boolean validity and one boundary flag')
        boundary = self.temporal.boundary[truncated.long()][:, None].expand(-1, 2, -1)
        if raw.shape[1] == 0:
            return boundary
        encoded = self.temporal(raw, valid)
        positions = torch.arange(raw.shape[1], device=raw.device)[None].expand_as(valid)
        last = positions.masked_fill(~valid, -1).max(-1).values
        selected = encoded[torch.arange(len(raw), device=raw.device), last.clamp_min(0)]
        return torch.where((last >= 0)[:, None, None], selected, boundary)

    def condition(self, hands: Tensor, exact: Tensor, audio: Tensor):
        """Fuse aligned history, historical clocks and current audio context."""
        if (hands.shape[:-1] != exact.shape[:-1] or
                hands.shape[-2:] != (2, self.config.hidden) or exact.shape[-1] != BASE_QUERY_DIM or
                audio.shape != (*hands.shape[:-2], self.config.audio_width)):
            raise ContractError('Conditioning requires aligned hand history, exact state and audio')
        values = pointwise(self.fuse, torch.cat((hands, pointwise(self.exact, exact)), -1))
        return values + self.audio_residual(audio).unsqueeze(-2)

    def timing_logits(self, audio: Tensor, history: Tensor, exact: Tensor):
        """Return [B,Q,10] hazards from absolute-bin queries and fixed history."""
        if (audio.ndim != 3 or audio.shape[-1] != self.config.audio_width or
                history.shape != (audio.shape[0], 2, self.config.hidden) or
                exact.shape != (*audio.shape[:2], 2, BASE_QUERY_DIM)):
            raise ContractError('Timing requires [B,Q,A] audio, [B,2,H] history and aligned exact states')
        hands = history[:, None].expand(-1, audio.shape[1], -1, -1)
        conditioned = self.condition(hands, exact, audio)
        # Hand exchange leaves event time unchanged. Direct audio also trains
        # the encoder while a transferred row audio residual starts at zero.
        return pointwise(self.timing, torch.cat((conditioned.mean(-2), audio), -1))

    def row_log_probs(self, audio: Tensor, history: Tensor, exact: Tensor,
                      legal: Tensor, occupancy: Tensor):
        """Normalize all supplied legal nonempty rows; never admit the empty row."""
        batch = audio.shape[0]
        if (audio.ndim != 2 or legal.shape != (batch, 256) or legal.dtype != torch.bool or
                occupancy.shape != (batch, 4) or occupancy.dtype != torch.bool or
                legal.device != audio.device or occupancy.device != audio.device):
            raise ContractError('Row decisions require boolean legal support and four-lane occupancy')
        mask = legal & self.nonempty
        if not bool(mask.any(-1).all()):
            raise ContractError('Row decision has no feasible nonempty action')
        hands = self.condition(history, exact, audio)
        scores = self.joint(hands)
        routed = self.route_residual(hands, torch.ones(batch, dtype=torch.bool, device=hands.device))
        # R1 never trained the zero-head route relative to its nonempty routes.
        scores = scores + torch.where(self.has_head[None], routed, torch.zeros_like(routed))
        scores = scores + self.release_residual(hands, occupancy.any(-1))
        return scores.masked_fill(~mask, -torch.inf).log_softmax(-1)

    def parameter_counts(self):
        """Count all parameters by first-level module and in total."""
        counts = {name: sum(p.numel() for p in module.parameters())
                  for name, module in self.named_children()}
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts

    def encode_generation(self, mel, *, seed, code=None):
        """Encode complete audio for models without an arrangement latent."""
        if code is not None:
            raise ContractError('This checkpoint has no persistent intent code')
        return self.encode_audio(mel), {}


@torch.no_grad()
def initialize_from_r1(model: JointAudioModel, checkpoint_path, expected_sha256: str):
    """Copy compatible R1 weights and report every omitted source tensor.

The digest pins the bytes loaded with weights_only=True. Exact-state projection
keeps only historical columns; seed, landmark and future-consequence modules
are omitted. Audio enters row scores through a newly zeroed residual. This is
initialization of a different trainable model, not R1 behavior preservation.
    """
    data = Path(checkpoint_path).read_bytes()
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or
            hashlib.sha256(data).hexdigest() != expected_sha256):
        raise ContractError('R1 initialization checkpoint differs from its pinned SHA-256')
    payload = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    if (not isinstance(payload, dict) or payload.get('config', {}).get('model', {}).get('arm') != 'r1' or
            not isinstance(payload.get('model'), dict)):
        raise ContractError('Initialization requires an R1 checkpoint with model configuration')
    source, destination = payload['model'], model.state_dict()
    copied, sliced, used = [], [], set()
    updates = {}
    for name, tensor in destination.items():
        if name.split('.')[0] not in R1_TRANSFER_MODULES:
            continue
        if name not in source:
            if name.startswith(('route_residual.', 'release_residual.')):
                continue
            raise ContractError(f'R1 initialization lacks required tensor {name}')
        value = source[name]
        if not isinstance(value, Tensor) or not bool(torch.isfinite(value).all()):
            raise ContractError(f'R1 initialization has invalid tensor {name}')
        if name == 'exact.0.weight':
            if value.ndim != 2 or value.shape[1] < BASE_QUERY_DIM:
                raise ContractError('R1 exact projection lacks the complete historical prefix')
            value = value[:, :BASE_QUERY_DIM]
            sliced.append(dict(name=name, source_columns=source[name].shape[1],
                               retained_columns=BASE_QUERY_DIM))
        if value.shape != tensor.shape:
            raise ContractError(f'R1 initialization shape mismatch for {name}')
        updates[name] = value
        copied.append(name)
        used.add(name)
    destination.update(updates)
    model.load_state_dict(destination, strict=True)
    nn.init.zeros_(model.audio_residual.weight)
    omitted = sorted(set(source) - used)
    return dict(checkpoint_sha256=expected_sha256, copied=sorted(copied), sliced=sliced,
                omitted=omitted, omitted_modules=sorted({name.split('.')[0] for name in omitted}),
                parameters=model.parameter_counts(),
                copied_parameters=sum(source[name].numel() if name != 'exact.0.weight' else
                                      model.exact[0].weight.numel() for name in copied),
                omitted_source_parameters=sum(source[name].numel() for name in omitted),
                discarded_exact_parameters=source['exact.0.weight'].numel() - model.exact[0].weight.numel())
