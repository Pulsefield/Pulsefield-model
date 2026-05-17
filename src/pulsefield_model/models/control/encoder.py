from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from pulsefield_model.features.control_v3_targets import CONFIDENCE_FEATURE_NAMES, VALUE_FEATURE_NAMES
from pulsefield_model.models.control.context import CONTEXT_LENGTH_FRAMES, TARGET_OFFSET_IN_CONTEXT


BOUNDED_01_VALUE_FEATURES = frozenset(
    (
        "hold_occupancy",
        "chord_ratio",
        "jack_excess",
        "jack_streak_exposure",
        "hand_imbalance_abs",
        "repeat_exact",
        "repeat_shift",
        "repeat_motion",
    )
)


@dataclass(frozen=True)
class ControlEncoderConfig:
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
class ControlEncoderOutput:
    value_pred: torch.Tensor
    confidence_pred: torch.Tensor
    compound_confidence_pred: torch.Tensor
    control_memory: torch.Tensor
    memory_padding_mask: torch.Tensor


class ControlEncoder(nn.Module):
    def __init__(self, config: ControlEncoderConfig = ControlEncoderConfig()) -> None:
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
        self.value_head = nn.Linear(config.d_model, len(VALUE_FEATURE_NAMES))
        self.confidence_head = nn.Linear(config.d_model, len(CONFIDENCE_FEATURE_NAMES))
        self.register_buffer(
            "_bounded_value_indexes",
            torch.tensor(
                [index for index, name in enumerate(VALUE_FEATURE_NAMES) if name in BOUNDED_01_VALUE_FEATURES],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self._control_confidence_index = CONFIDENCE_FEATURE_NAMES.index("control_confidence")
        self._reset_parameters()

    def forward(
        self,
        *,
        context_mel: torch.Tensor,
        context_dense_timing_v2: torch.Tensor,
        normalized_difficulty: torch.Tensor,
        context_padding_mask: torch.Tensor | None = None,
    ) -> ControlEncoderOutput:
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

        value_pred = self._apply_value_ranges(self.value_head(center_hidden))
        confidence_pred = torch.sigmoid(self.confidence_head(center_hidden))
        compound_confidence_pred = confidence_pred[
            ...,
            self._control_confidence_index : self._control_confidence_index + 1,
        ]
        return ControlEncoderOutput(
            value_pred=value_pred,
            confidence_pred=confidence_pred,
            compound_confidence_pred=compound_confidence_pred,
            control_memory=control_memory,
            memory_padding_mask=context_padding_mask,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _apply_value_ranges(self, value_logits: torch.Tensor) -> torch.Tensor:
        value_pred = value_logits.clone()
        if self._bounded_value_indexes.numel() > 0:
            value_pred[..., self._bounded_value_indexes] = torch.sigmoid(
                value_pred[..., self._bounded_value_indexes],
            )
        return value_pred

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


def _mask_hidden(hidden: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    return hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)


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


def _validate_config(config: ControlEncoderConfig) -> None:
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
    ):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
    if not isinstance(config.dropout, (int, float)) or isinstance(config.dropout, bool) or not math.isfinite(float(config.dropout)):
        raise ValueError("dropout must be finite")
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
