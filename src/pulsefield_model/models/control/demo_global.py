from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from pulsefield_model.models.control.context import CONTEXT_LENGTH_FRAMES, TARGET_OFFSET_IN_CONTEXT


GLOBAL_TIME_FEATURES = 4
GLOBAL_CONDITION_FEATURES = 6
FRAME_HOP_SECONDS = 0.020
MIN_GLOBAL_STRIDE = 8
DEFAULT_GLOBAL_STRIDE = 16


@dataclass(frozen=True)
class ControlDemoGlobalEncoderConfig:
    mel_dim: int = 160
    timing_dim: int = 4

    context_frames: int = CONTEXT_LENGTH_FRAMES
    target_frames: int = 100
    target_offset: int = TARGET_OFFSET_IN_CONTEXT

    d_model: int = 384
    heads: int = 8
    layers: int = 3
    ffn_dim: int = 1536
    dropout: float = 0.1

    conv_blocks: int = 2
    conv_kernel_size: int = 5

    use_global_memory: bool = True
    global_stride: int = DEFAULT_GLOBAL_STRIDE
    global_layers: int = 2
    global_ffn_dim: int = 1536
    global_conv_blocks: int = 1
    global_fusion_start_layer: int = 1
    global_gate_init: float = -2.94


@dataclass(frozen=True)
class ControlDemoGlobalEncoderOutput:
    value_pred: torch.Tensor
    control_memory: torch.Tensor
    memory_padding_mask: torch.Tensor
    global_memory: torch.Tensor | None = None
    global_memory_padding_mask: torch.Tensor | None = None


def _mask_hidden(hidden: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
    if padding_mask is None:
        return hidden
    return hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)


def _padding_mask_from_frame_count(
    *,
    frame_count: torch.Tensor,
    max_frames: int,
    device: torch.device,
) -> torch.Tensor:
    frame_index = torch.arange(max_frames, device=device).unsqueeze(0)
    return frame_index >= frame_count.to(device=device, dtype=torch.long).unsqueeze(1)


def _validate_padding_mask_covers_frame_count_tail(
    *,
    padding_mask: torch.Tensor,
    frame_count: torch.Tensor,
) -> None:
    if frame_count.ndim != 1:
        raise ValueError("frame_count must have shape [B]")
    batch_size, max_frames = padding_mask.shape
    if frame_count.shape != (batch_size,):
        raise ValueError("frame_count must have shape [B] matching padding_mask")
    if bool(torch.any(frame_count <= 0)):
        raise ValueError("frame_count must be positive")
    if bool(torch.any(frame_count > max_frames)):
        raise ValueError("frame_count cannot exceed full-song tensor length")
    frame_index = torch.arange(max_frames, device=padding_mask.device).unsqueeze(0)
    beyond_frame_count = frame_index >= frame_count.to(device=padding_mask.device, dtype=torch.long).unsqueeze(1)
    if bool((beyond_frame_count & ~padding_mask).any()):
        raise ValueError(
            "padding_mask must cover frame_count tail: frames at or beyond frame_count "
            "must be masked"
        )


def _masked_chunk_mean(
    values: torch.Tensor,
    padding_mask: torch.Tensor | None,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 3:
        raise ValueError(f"values must have shape [B,T,C], got {tuple(values.shape)}")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if padding_mask is None:
        padding_mask = torch.zeros(values.shape[:2], dtype=torch.bool, device=values.device)
    if padding_mask.shape != values.shape[:2]:
        raise ValueError("padding_mask must have shape [B,T]")
    if padding_mask.dtype != torch.bool:
        raise ValueError("padding_mask must be bool")

    pad_frames = (-values.shape[1]) % chunk_size
    if pad_frames:
        values = F.pad(values, (0, 0, 0, pad_frames))
        padding_mask = F.pad(padding_mask, (0, pad_frames), value=True)

    batch_size, padded_frames, channels = values.shape
    chunk_count = padded_frames // chunk_size
    chunked_values = values.reshape(batch_size, chunk_count, chunk_size, channels)
    chunked_valid = (~padding_mask).reshape(batch_size, chunk_count, chunk_size)
    weights = chunked_valid.unsqueeze(-1).to(dtype=values.dtype)
    counts = weights.sum(dim=2)
    pooled = (chunked_values * weights).sum(dim=2) / counts.clamp_min(1.0)
    pooled_mask = counts.squeeze(-1) == 0
    pooled = _mask_hidden(pooled, pooled_mask)
    return pooled, pooled_mask


def _masked_mean(hidden: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
    if hidden.ndim != 3:
        raise ValueError(f"hidden must have shape [B,T,D], got {tuple(hidden.shape)}")
    if padding_mask is None:
        return hidden.mean(dim=1)
    if padding_mask.shape != hidden.shape[:2]:
        raise ValueError("padding_mask must have shape [B,T]")
    if padding_mask.dtype != torch.bool:
        raise ValueError("padding_mask must be bool")
    valid = (~padding_mask).unsqueeze(-1).to(dtype=hidden.dtype)
    return (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


def _full_song_time_features(
    *,
    padding_mask: torch.Tensor,
    frame_count: torch.Tensor,
    target_start_frame: torch.Tensor,
    target_frames: int,
) -> torch.Tensor:
    if padding_mask.ndim != 2 or padding_mask.dtype != torch.bool:
        raise ValueError("padding_mask must be bool with shape [B,T]")
    batch_size, frame_length = padding_mask.shape
    if frame_count.shape != (batch_size,) or target_start_frame.shape != (batch_size,):
        raise ValueError("frame_count and target_start_frame must have shape [B]")

    device = padding_mask.device
    frame_count_f = frame_count.to(device=device, dtype=torch.float32)
    target_start_f = target_start_frame.to(device=device, dtype=torch.float32)
    denominator = (frame_count_f - 1.0).clamp_min(1.0)
    frame_index = torch.arange(frame_length, dtype=torch.float32, device=device).unsqueeze(0)

    song_progress = frame_index / denominator.unsqueeze(1)
    target_center = target_start_f + 0.5 * float(target_frames)
    target_center_progress = (target_center / denominator).unsqueeze(1).expand_as(song_progress)
    relative_progress_to_target = song_progress - target_center_progress
    progress_to_end = 1.0 - song_progress

    features = torch.stack(
        (song_progress, target_center_progress, relative_progress_to_target, progress_to_end),
        dim=-1,
    )
    return features.masked_fill(padding_mask.unsqueeze(-1), 0.0).to(dtype=torch.float32)


def _global_condition_features(
    *,
    normalized_difficulty: torch.Tensor,
    frame_count: torch.Tensor,
    target_start_frame: torch.Tensor,
    context_frames: int,
    target_frames: int,
    target_offset: int,
) -> torch.Tensor:
    if normalized_difficulty.ndim != 1 or frame_count.ndim != 1 or target_start_frame.ndim != 1:
        raise ValueError("normalized_difficulty, frame_count, and target_start_frame must have shape [B]")
    if normalized_difficulty.shape != frame_count.shape or frame_count.shape != target_start_frame.shape:
        raise ValueError("global condition inputs must share batch shape")

    device = normalized_difficulty.device
    frame_count_f = frame_count.to(device=device, dtype=torch.float32)
    target_start_f = target_start_frame.to(device=device, dtype=torch.float32)
    denominator = (frame_count_f - 1.0).clamp_min(1.0)
    target_center = target_start_f + 0.5 * float(target_frames)
    context_start = target_start_f - float(target_offset)
    context_end = context_start + float(context_frames)
    song_seconds = frame_count_f * FRAME_HOP_SECONDS
    seconds_to_end = (frame_count_f - target_center).clamp_min(0.0) * FRAME_HOP_SECONDS

    return torch.stack(
        (
            normalized_difficulty.to(device=device, dtype=torch.float32),
            target_center / denominator,
            context_start / denominator,
            context_end / denominator,
            seconds_to_end / 300.0,
            torch.log1p(song_seconds) / 6.0,
        ),
        dim=-1,
    ).to(dtype=torch.float32)


class _TemporalConvBlock(nn.Module):
    def __init__(self, *, d_model: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = hidden
        hidden = self.norm(hidden)
        hidden = self.conv(hidden.transpose(1, 2)).transpose(1, 2)
        return residual + self.dropout(self.activation(hidden))


class _DifficultyFiLM(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * d_model),
        )
        final = self.net[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, hidden: torch.Tensor, normalized_difficulty: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.net(normalized_difficulty.to(dtype=hidden.dtype).reshape(-1, 1))
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return hidden * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class _GlobalFiLM(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * d_model),
        )
        final = self.net[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, hidden: torch.Tensor, global_condition: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.net(global_condition.to(dtype=hidden.dtype))
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return hidden * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class _GlobalSongEncoder(nn.Module):
    def __init__(self, config: ControlDemoGlobalEncoderConfig) -> None:
        super().__init__()
        self.config = config
        global_input_dim = config.mel_dim + config.timing_dim + GLOBAL_TIME_FEATURES
        self.projection = nn.Linear(global_input_dim, config.d_model)
        self.conv_stem = nn.ModuleList(
            [
                _TemporalConvBlock(
                    d_model=config.d_model,
                    kernel_size=config.conv_kernel_size,
                    dropout=config.dropout,
                )
                for _ in range(config.global_conv_blocks)
            ]
        )
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.heads,
                    dim_feedforward=config.global_ffn_dim,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.global_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        *,
        full_mel: torch.Tensor,
        full_dense_timing_v2: torch.Tensor,
        padding_mask: torch.Tensor,
        frame_count: torch.Tensor,
        target_start_frame: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_features = _full_song_time_features(
            padding_mask=padding_mask,
            frame_count=frame_count,
            target_start_frame=target_start_frame,
            target_frames=self.config.target_frames,
        )
        global_input = torch.cat(
            [full_mel, full_dense_timing_v2, time_features.to(dtype=full_mel.dtype)],
            dim=-1,
        )
        pooled, global_padding_mask = _masked_chunk_mean(
            global_input,
            padding_mask,
            chunk_size=self.config.global_stride,
        )

        hidden = self.projection(pooled)
        hidden = _mask_hidden(hidden, global_padding_mask)
        for conv_block in self.conv_stem:
            hidden = conv_block(hidden)
            hidden = _mask_hidden(hidden, global_padding_mask)
        for layer in self.encoder_layers:
            hidden = layer(hidden, src_key_padding_mask=global_padding_mask)
            hidden = _mask_hidden(hidden, global_padding_mask)
        global_memory = self.output_norm(hidden)
        global_memory = _mask_hidden(global_memory, global_padding_mask)
        global_summary = _masked_mean(global_memory, global_padding_mask)
        return global_memory, global_padding_mask, global_summary


class _GlobalFusionBlock(nn.Module):
    def __init__(self, config: ControlDemoGlobalEncoderConfig) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(config.d_model)
        self.memory_norm = nn.LayerNorm(config.d_model)
        self.cross_attn = nn.MultiheadAttention(
            config.d_model,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.gate_logit = nn.Parameter(torch.tensor(float(config.global_gate_init)))
        self.film = _GlobalFiLM(config.d_model)

    def forward(
        self,
        *,
        local_hidden: torch.Tensor,
        context_padding_mask: torch.Tensor,
        global_memory: torch.Tensor,
        global_padding_mask: torch.Tensor,
        global_condition: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query_norm(local_hidden)
        memory = self.memory_norm(global_memory)
        cross, _ = self.cross_attn(
            query=query,
            key=memory,
            value=memory,
            key_padding_mask=global_padding_mask,
            need_weights=False,
        )
        hidden = local_hidden + torch.sigmoid(self.gate_logit).to(dtype=local_hidden.dtype) * self.dropout(cross)
        hidden = self.film(hidden, global_condition)
        return _mask_hidden(hidden, context_padding_mask)


class ControlDemoGlobalEncoder(nn.Module):
    def __init__(self, config: ControlDemoGlobalEncoderConfig = ControlDemoGlobalEncoderConfig()) -> None:
        super().__init__()
        _validate_config(config)
        self.config = config
        self.input_projection = nn.Linear(config.mel_dim + config.timing_dim, config.d_model)
        self.conv_stem = nn.Sequential(
            *[
                _TemporalConvBlock(
                    d_model=config.d_model,
                    kernel_size=config.conv_kernel_size,
                    dropout=config.dropout,
                )
                for _ in range(config.conv_blocks)
            ]
        )
        self.position = nn.Parameter(torch.zeros(1, config.context_frames, config.d_model))
        self.stem_film = _DifficultyFiLM(config.d_model)
        self.encoder_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config.d_model,
                    nhead=config.heads,
                    dim_feedforward=config.ffn_dim,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.layers)
            ]
        )
        self.block_films = nn.ModuleList([_DifficultyFiLM(config.d_model) for _ in range(config.layers)])
        if config.use_global_memory:
            self.global_encoder: _GlobalSongEncoder | None = _GlobalSongEncoder(config)
            self.global_condition_projection: nn.Module | None = nn.Sequential(
                nn.Linear(config.d_model + GLOBAL_CONDITION_FEATURES, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.global_fusions = nn.ModuleList(
                [_GlobalFusionBlock(config) for _ in range(config.global_fusion_start_layer, config.layers)]
            )
        else:
            self.global_encoder = None
            self.global_condition_projection = None
            self.global_fusions = nn.ModuleList()
        self.output_norm = nn.LayerNorm(config.d_model)
        self.value_head = nn.Linear(config.d_model, 1)
        self._reset_parameters()

    def forward(
        self,
        *,
        context_mel: torch.Tensor,
        context_dense_timing_v2: torch.Tensor,
        normalized_difficulty: torch.Tensor,
        context_padding_mask: torch.Tensor | None = None,
        full_mel: torch.Tensor | None = None,
        full_dense_timing_v2: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        frame_count: torch.Tensor | None = None,
        target_start_frame: torch.Tensor | None = None,
    ) -> ControlDemoGlobalEncoderOutput:
        self._validate_inputs(
            context_mel=context_mel,
            context_dense_timing_v2=context_dense_timing_v2,
            normalized_difficulty=normalized_difficulty,
            context_padding_mask=context_padding_mask,
            full_mel=full_mel,
            full_dense_timing_v2=full_dense_timing_v2,
            padding_mask=padding_mask,
            frame_count=frame_count,
            target_start_frame=target_start_frame,
        )
        if context_padding_mask is None:
            context_padding_mask = torch.zeros(
                context_mel.shape[:2],
                dtype=torch.bool,
                device=context_mel.device,
            )

        global_memory: torch.Tensor | None = None
        global_padding_mask: torch.Tensor | None = None
        global_condition: torch.Tensor | None = None
        if self.config.use_global_memory:
            assert self.global_encoder is not None
            assert self.global_condition_projection is not None
            assert full_mel is not None
            assert full_dense_timing_v2 is not None
            assert padding_mask is not None
            assert frame_count is not None
            assert target_start_frame is not None
            global_memory, global_padding_mask, global_summary = self.global_encoder(
                full_mel=full_mel,
                full_dense_timing_v2=full_dense_timing_v2,
                padding_mask=padding_mask,
                frame_count=frame_count,
                target_start_frame=target_start_frame,
            )
            condition_features = _global_condition_features(
                normalized_difficulty=normalized_difficulty,
                frame_count=frame_count,
                target_start_frame=target_start_frame,
                context_frames=self.config.context_frames,
                target_frames=self.config.target_frames,
                target_offset=self.config.target_offset,
            ).to(device=global_summary.device, dtype=global_summary.dtype)
            global_condition = self.global_condition_projection(
                torch.cat([global_summary, condition_features], dim=-1)
            )

        model_input = torch.cat([context_mel, context_dense_timing_v2], dim=-1)
        hidden = self.input_projection(model_input)
        hidden = _mask_hidden(hidden, context_padding_mask)
        for conv_block in self.conv_stem:
            hidden = conv_block(hidden)
            hidden = _mask_hidden(hidden, context_padding_mask)
        hidden = hidden + self.position[:, : hidden.shape[1]]
        hidden = self.stem_film(hidden, normalized_difficulty)
        hidden = _mask_hidden(hidden, context_padding_mask)

        fusion_index = 0
        for layer_index, (layer, film) in enumerate(zip(self.encoder_layers, self.block_films)):
            hidden = layer(hidden, src_key_padding_mask=context_padding_mask)
            hidden = film(hidden, normalized_difficulty)
            hidden = _mask_hidden(hidden, context_padding_mask)
            if self.config.use_global_memory and layer_index >= self.config.global_fusion_start_layer:
                assert global_memory is not None
                assert global_padding_mask is not None
                assert global_condition is not None
                hidden = self.global_fusions[fusion_index](
                    local_hidden=hidden,
                    context_padding_mask=context_padding_mask,
                    global_memory=global_memory,
                    global_padding_mask=global_padding_mask,
                    global_condition=global_condition,
                )
                fusion_index += 1

        control_memory = self.output_norm(hidden)
        control_memory = _mask_hidden(control_memory, context_padding_mask)
        center_hidden = control_memory[:, self.config.target_offset : self.config.target_offset + self.config.target_frames]
        return ControlDemoGlobalEncoderOutput(
            value_pred=self.value_head(center_hidden),
            control_memory=control_memory,
            memory_padding_mask=context_padding_mask,
            global_memory=global_memory,
            global_memory_padding_mask=global_padding_mask,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _validate_inputs(
        self,
        *,
        context_mel: torch.Tensor,
        context_dense_timing_v2: torch.Tensor,
        normalized_difficulty: torch.Tensor,
        context_padding_mask: torch.Tensor | None,
        full_mel: torch.Tensor | None,
        full_dense_timing_v2: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
        frame_count: torch.Tensor | None,
        target_start_frame: torch.Tensor | None,
    ) -> None:
        if context_mel.ndim != 3 or context_mel.shape[-1] != self.config.mel_dim:
            raise ValueError(f"context_mel must have shape [B,T,{self.config.mel_dim}], got {tuple(context_mel.shape)}")
        if context_dense_timing_v2.ndim != 3 or context_dense_timing_v2.shape[-1] != self.config.timing_dim:
            raise ValueError(
                f"context_dense_timing_v2 must have shape [B,T,{self.config.timing_dim}], "
                f"got {tuple(context_dense_timing_v2.shape)}"
            )
        if context_mel.shape[:2] != context_dense_timing_v2.shape[:2]:
            raise ValueError("context_mel and context_dense_timing_v2 must share batch/frame dimensions")
        if int(context_mel.shape[1]) != self.config.context_frames:
            raise ValueError(f"context length must be {self.config.context_frames}, got {context_mel.shape[1]}")
        if normalized_difficulty.shape != (context_mel.shape[0],):
            raise ValueError("normalized_difficulty must have shape [B]")
        if context_padding_mask is not None:
            if context_padding_mask.shape != context_mel.shape[:2]:
                raise ValueError("context_padding_mask must have shape [B,T]")
            if context_padding_mask.dtype != torch.bool:
                raise ValueError("context_padding_mask must be bool")
        if not self.config.use_global_memory:
            return

        missing = [
            name
            for name, value in (
                ("full_mel", full_mel),
                ("full_dense_timing_v2", full_dense_timing_v2),
                ("padding_mask", padding_mask),
                ("frame_count", frame_count),
                ("target_start_frame", target_start_frame),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"global memory requires full-song inputs: {missing}")
        assert full_mel is not None
        assert full_dense_timing_v2 is not None
        assert padding_mask is not None
        assert frame_count is not None
        assert target_start_frame is not None
        if full_mel.ndim != 3 or full_mel.shape[-1] != self.config.mel_dim:
            raise ValueError(f"full_mel must have shape [B,T,{self.config.mel_dim}], got {tuple(full_mel.shape)}")
        if full_dense_timing_v2.ndim != 3 or full_dense_timing_v2.shape[-1] != self.config.timing_dim:
            raise ValueError(
                f"full_dense_timing_v2 must have shape [B,T,{self.config.timing_dim}], "
                f"got {tuple(full_dense_timing_v2.shape)}"
            )
        if full_mel.shape[:2] != full_dense_timing_v2.shape[:2]:
            raise ValueError("full_mel and full_dense_timing_v2 must share batch/frame dimensions")
        if full_mel.shape[0] != context_mel.shape[0]:
            raise ValueError("full-song and context tensors must share batch size")
        if padding_mask.shape != full_mel.shape[:2]:
            raise ValueError("padding_mask must have shape [B,T_full]")
        if padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask must be bool")
        if frame_count.shape != (context_mel.shape[0],):
            raise ValueError("frame_count must have shape [B]")
        if target_start_frame.shape != (context_mel.shape[0],):
            raise ValueError("target_start_frame must have shape [B]")
        for name, value in (("frame_count", frame_count), ("target_start_frame", target_start_frame)):
            if value.dtype == torch.bool or value.dtype.is_floating_point or value.dtype.is_complex:
                raise ValueError(f"{name} must be an integer tensor")
        if full_mel.device != context_mel.device or full_dense_timing_v2.device != full_mel.device:
            raise ValueError("full-song tensors must be on the same device as context tensors")
        if padding_mask.device != full_mel.device:
            raise ValueError("padding_mask must be on the same device as full_mel")
        if frame_count.device != full_mel.device or target_start_frame.device != full_mel.device:
            raise ValueError("frame metadata tensors must be on the same device as full_mel")
        _validate_padding_mask_covers_frame_count_tail(
            padding_mask=padding_mask,
            frame_count=frame_count,
        )
        if normalized_difficulty.device != context_mel.device:
            raise ValueError("normalized_difficulty must be on the same device as context tensors")
        if bool(torch.any(frame_count <= 0)):
            raise ValueError("frame_count must be positive")
        if bool(torch.any(frame_count > full_mel.shape[1])):
            raise ValueError("frame_count cannot exceed padded full-song length")
        if bool(torch.any(target_start_frame < 0)):
            raise ValueError("target_start_frame must be non-negative")
        if bool(torch.any(target_start_frame >= frame_count)):
            raise ValueError("target_start_frame must be less than frame_count")

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.position, mean=0.0, std=0.02)


def _validate_config(config: ControlDemoGlobalEncoderConfig) -> None:
    for name in (
        "mel_dim",
        "timing_dim",
        "context_frames",
        "target_frames",
        "target_offset",
        "d_model",
        "heads",
        "layers",
        "ffn_dim",
        "conv_blocks",
        "conv_kernel_size",
        "global_stride",
        "global_layers",
        "global_ffn_dim",
        "global_conv_blocks",
        "global_fusion_start_layer",
    ):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
    for name in ("dropout", "global_gate_init"):
        value = getattr(config, name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if not isinstance(config.use_global_memory, bool):
        raise ValueError("use_global_memory must be bool")
    if config.dropout < 0.0:
        raise ValueError("dropout must be non-negative")
    if config.context_frames != CONTEXT_LENGTH_FRAMES:
        raise ValueError(f"context_frames must be {CONTEXT_LENGTH_FRAMES}")
    if config.target_frames != 100:
        raise ValueError("target_frames must be 100")
    if config.target_offset != TARGET_OFFSET_IN_CONTEXT:
        raise ValueError(f"target_offset must be {TARGET_OFFSET_IN_CONTEXT}")
    if config.d_model <= 0:
        raise ValueError("d_model must be positive")
    if config.heads <= 0 or config.d_model % config.heads != 0:
        raise ValueError("heads must be positive and divide d_model")
    if config.layers <= 0:
        raise ValueError("layers must be positive")
    if config.ffn_dim <= 0:
        raise ValueError("ffn_dim must be positive")
    if config.conv_blocks < 0:
        raise ValueError("conv_blocks must be non-negative")
    if config.conv_kernel_size <= 0 or config.conv_kernel_size % 2 == 0:
        raise ValueError("conv_kernel_size must be a positive odd integer")
    if config.global_stride < MIN_GLOBAL_STRIDE:
        raise ValueError(
            "global_stride is too small for the whole-song global Transformer: "
            f"got {config.global_stride}, minimum is {MIN_GLOBAL_STRIDE}. "
            "The global branch must operate on downsampled low-rate memory tokens, "
            "not full-resolution or near-full-resolution 20ms frames."
        )
    if config.global_layers <= 0:
        raise ValueError("global_layers must be positive")
    if config.global_ffn_dim <= 0:
        raise ValueError("global_ffn_dim must be positive")
    if config.global_conv_blocks < 0:
        raise ValueError("global_conv_blocks must be non-negative")
    if config.global_fusion_start_layer < 0 or config.global_fusion_start_layer >= config.layers:
        raise ValueError("global_fusion_start_layer must select an existing local layer")
