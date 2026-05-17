from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from pulsefield_model.models.control.context import CONTEXT_LENGTH_FRAMES, TARGET_OFFSET_IN_CONTEXT
from pulsefield_model.models.control.encoder import _DifficultyFiLM, _TemporalConvBlock, _mask_hidden, _validate_config


@dataclass(frozen=True)
class ControlDemoEncoderConfig:
    mel_dim: int = 160
    timing_dim: int = 4
    context_frames: int = CONTEXT_LENGTH_FRAMES
    target_frames: int = 100
    target_offset: int = TARGET_OFFSET_IN_CONTEXT
    d_model: int = 256
    heads: int = 4
    layers: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1
    conv_blocks: int = 2
    conv_kernel_size: int = 5


@dataclass(frozen=True)
class ControlDemoEncoderOutput:
    value_pred: torch.Tensor
    control_memory: torch.Tensor
    memory_padding_mask: torch.Tensor


class ControlDemoEncoder(nn.Module):
    def __init__(self, config: ControlDemoEncoderConfig = ControlDemoEncoderConfig()) -> None:
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
    ) -> ControlDemoEncoderOutput:
        self._validate_inputs(
            context_mel=context_mel,
            context_dense_timing_v2=context_dense_timing_v2,
            normalized_difficulty=normalized_difficulty,
            context_padding_mask=context_padding_mask,
        )
        if context_padding_mask is None:
            context_padding_mask = torch.zeros(
                context_mel.shape[:2],
                dtype=torch.bool,
                device=context_mel.device,
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
        for layer, film in zip(self.encoder_layers, self.block_films):
            hidden = layer(hidden, src_key_padding_mask=context_padding_mask)
            hidden = film(hidden, normalized_difficulty)
            hidden = _mask_hidden(hidden, context_padding_mask)
        control_memory = self.output_norm(hidden)
        control_memory = _mask_hidden(control_memory, context_padding_mask)
        center_hidden = control_memory[:, self.config.target_offset : self.config.target_offset + self.config.target_frames]

        return ControlDemoEncoderOutput(
            value_pred=self.value_head(center_hidden),
            control_memory=control_memory,
            memory_padding_mask=context_padding_mask,
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

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.position, mean=0.0, std=0.02)
