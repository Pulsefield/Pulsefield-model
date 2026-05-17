from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from pulsefield_model.models.control.demo_global import (
    ControlDemoGlobalEncoderConfig,
    MIN_GLOBAL_STRIDE,
    _GlobalSongEncoder,
    _mask_hidden,
)
from pulsefield_model.models.mapper.shared.grammar import build_grammar_mask
from pulsefield_model.models.mapper.shared.model import (
    MapperTupleConfig,
    MapperTupleForwardOutput,
    TupleMapperBase,
    _difficulty_tensor,
    _load_carry_state,
    _reject_old_mapper_contract,
    _require_state_mapping,
    _require_state_tensor,
    _require_tensor,
    _sanitize_padded_fragment_states,
    _time_features,
    _validate_fragment_contract,
)
from pulsefield_model.models.mapper.shared.tokenizer import (
    MAPPER_DENSITY_FRAME_MS,
    MAPPER_DENSITY_FRAMES,
    MAPPER_WRITE_MS,
)
from pulsefield_model.models.mapper.shared.vocab import MapperTupleVocab


GLOBAL_POSITION_FEATURES = 4


@dataclass(frozen=True)
class MapperV2Config(MapperTupleConfig):
    use_global_context: bool = True
    global_stride: int = 16
    global_layers: int = 1
    global_ffn_dim: int | None = None
    global_conv_blocks: int = 1
    global_conv_kernel_size: int = 5
    global_gate_init: float = -2.94


@dataclass(frozen=True)
class MapperV2ForwardOutput(MapperTupleForwardOutput):
    global_memory: torch.Tensor | None = None
    global_memory_padding_mask: torch.Tensor | None = None
    global_attention_gates: torch.Tensor | None = None
    global_position_features: torch.Tensor | None = None


@dataclass(frozen=True)
class MapperV2SelfAttentionKVCache:
    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class MapperV2IncrementalDecodeState:
    self_attention_kv_cache: tuple[MapperV2SelfAttentionKVCache, ...]

    @property
    def sequence_length(self) -> int:
        if not self.self_attention_kv_cache:
            return 0
        return int(self.self_attention_kv_cache[0].key.shape[2])


@dataclass(frozen=True)
class MapperV2IncrementalDecodeOutput:
    decode_state: MapperV2IncrementalDecodeState
    decoder_input_token: torch.Tensor
    position: torch.Tensor
    base_logits: torch.Tensor
    logits_final: torch.Tensor
    decoder_hidden: torch.Tensor
    state_prior_bias: torch.Tensor
    state_prior_lane_action_bias: torch.Tensor
    ln_close_logits: torch.Tensor
    ln_close_event_bias: torch.Tensor
    ln_close_time_shift_bias: torch.Tensor
    grammar_mask: torch.Tensor
    global_attention_gates: torch.Tensor | None = None


MapperV2ModelOutput = MapperV2ForwardOutput


class MapperV2Model(TupleMapperBase):
    def __init__(
        self,
        config: MapperV2Config = MapperV2Config(),
        *,
        vocab: MapperTupleVocab | None = None,
        control_encoder: nn.Module | None = None,
    ) -> None:
        _validate_v2_config(config)
        super().__init__(config, vocab=vocab, control_encoder=control_encoder)
        self.config: MapperV2Config = config
        if config.use_global_context:
            self.global_encoder: _GlobalSongEncoder | None = _GlobalSongEncoder(_global_encoder_config(config))
            global_position_projection = nn.Linear(GLOBAL_POSITION_FEATURES, config.d_model)
            nn.init.normal_(global_position_projection.weight, mean=0.0, std=0.01)
            nn.init.zeros_(global_position_projection.bias)
            self.global_position_projection: nn.Module | None = global_position_projection
            self.global_cross_attention_layers = nn.ModuleList(
                [
                    _MapperGlobalCrossAttentionBlock(
                        d_model=config.d_model,
                        heads=config.heads,
                        dropout=config.dropout,
                        gate_init=config.global_gate_init,
                    )
                    for _ in range(config.layers)
                ],
            )
        else:
            self.global_encoder = None
            self.global_position_projection = None
            self.global_cross_attention_layers = nn.ModuleList()

    def forward(
        self,
        batch: Mapping[str, torch.Tensor] | None = None,
        *,
        control_memory_8s: torch.Tensor | None = None,
        density_teacher_8s: torch.Tensor | None = None,
        **kwargs: torch.Tensor | None,
    ) -> MapperV2ForwardOutput:
        if batch is None:
            batch = {key: value for key, value in kwargs.items() if value is not None}
        elif kwargs:
            merged = dict(batch)
            merged.update({key: value for key, value in kwargs.items() if value is not None})
            batch = merged
        if control_memory_8s is None:
            maybe_control_memory = batch.get("control_memory_8s")
            if isinstance(maybe_control_memory, torch.Tensor):
                control_memory_8s = maybe_control_memory
        projected_control_memory_8s = batch.get("projected_control_memory_8s")
        if projected_control_memory_8s is not None and not isinstance(projected_control_memory_8s, torch.Tensor):
            raise ValueError("projected_control_memory_8s must be a torch.Tensor")
        if projected_control_memory_8s is not None and control_memory_8s is not None:
            raise ValueError("projected_control_memory_8s cannot be supplied with control_memory_8s")
        if density_teacher_8s is None:
            maybe_density_teacher = batch.get("density_teacher_8s")
            if isinstance(maybe_density_teacher, torch.Tensor):
                density_teacher_8s = maybe_density_teacher
        has_control_context = control_memory_8s is not None or projected_control_memory_8s is not None
        if has_control_context != (density_teacher_8s is not None):
            raise ValueError("control_memory_8s and density_teacher_8s must be supplied together")
        if isinstance(batch.get("control_memory_padding_mask_8s"), torch.Tensor):
            raise ValueError("control_memory_padding_mask_8s is not supported in Phase B; supply full 8s control memory")

        _reject_old_mapper_contract(batch)
        decoder_input = _require_tensor(batch, "decoder_input_tokens", ndim=2).to(dtype=torch.long)
        loss_target_tokens = _require_tensor(batch, "target_fragment_tokens", ndim=2).to(
            device=decoder_input.device,
            dtype=torch.long,
        )
        target_fragment_mask = _require_tensor(batch, "target_fragment_mask", ndim=2).to(
            device=decoder_input.device,
            dtype=torch.bool,
        )
        if int(decoder_input.shape[1]) < 1:
            raise ValueError("decoder_input_tokens must contain at least one fragment position")
        if tuple(loss_target_tokens.shape) != tuple(decoder_input.shape):
            raise ValueError("target_fragment_tokens must match decoder_input_tokens shape")
        if tuple(target_fragment_mask.shape) != tuple(decoder_input.shape):
            raise ValueError("target_fragment_mask must match decoder_input_tokens shape")
        input_padding_mask = ~target_fragment_mask

        device = decoder_input.device
        states = _require_state_mapping(batch, "target_fragment_states")
        current_ms = _require_state_tensor(states, "current_ms", ndim=2).to(device=device, dtype=torch.long)
        open_mask = _require_state_tensor(states, "open_mask", ndim=3).to(device=device, dtype=torch.bool)
        open_start_ms = _require_state_tensor(states, "open_start_ms", ndim=3).to(device=device, dtype=torch.long)
        open_age_ms = _require_state_tensor(states, "open_age_ms", ndim=3).to(device=device, dtype=torch.long)
        write_start_ms = _require_tensor(batch, "write_start_ms", ndim=1).to(device=device, dtype=torch.long)
        write_end_ms = _require_tensor(batch, "write_end_ms", ndim=1).to(device=device, dtype=torch.long)
        is_full_chart_start = _require_tensor(batch, "is_full_chart_start", ndim=1).to(device=device, dtype=torch.bool)
        is_full_chart_end = _require_tensor(batch, "is_full_chart_end", ndim=1).to(device=device, dtype=torch.bool)
        ln_carry_in = _load_carry_state(batch, "ln_carry_in", device=device)
        ln_carry_out = _load_carry_state(batch, "ln_carry_out", device=device)
        if tuple(current_ms.shape) != tuple(decoder_input.shape):
            raise ValueError("target_fragment_states.current_ms must align with decoder_input_tokens")
        if tuple(open_mask.shape[:2]) != tuple(decoder_input.shape) or int(open_mask.shape[-1]) != 4:
            raise ValueError("target_fragment_states.open_mask must have shape [B,S,4]")
        if tuple(open_start_ms.shape) != tuple(open_mask.shape):
            raise ValueError("target_fragment_states.open_start_ms must align with target_fragment_states.open_mask")
        if tuple(open_age_ms.shape) != tuple(open_mask.shape):
            raise ValueError("target_fragment_states.open_age_ms must align with target_fragment_states.open_mask")
        valid_input_mask = target_fragment_mask.to(device=device, dtype=torch.bool)
        _validate_fragment_contract(
            decoder_input_tokens=decoder_input,
            target_fragment_tokens=loss_target_tokens,
            target_fragment_mask=valid_input_mask,
            current_ms=current_ms,
            open_mask=open_mask,
            open_start_ms=open_start_ms,
            open_age_ms=open_age_ms,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            is_full_chart_start=is_full_chart_start,
            is_full_chart_end=is_full_chart_end,
            ln_carry_in=ln_carry_in,
            ln_carry_out=ln_carry_out,
            bos_id=self.vocab.bos_id,
            eos_id=self.vocab.eos_id,
        )
        current_ms, open_mask, open_start_ms, open_age_ms = _sanitize_padded_fragment_states(
            current_ms=current_ms,
            open_mask=open_mask,
            open_start_ms=open_start_ms,
            open_age_ms=open_age_ms,
            write_end_ms=write_end_ms,
            valid_input_mask=valid_input_mask,
        )

        if control_memory_8s is None and projected_control_memory_8s is None:
            control_memory_8s, density_teacher_8s = self._control_teacher_8s(batch)
        density_teacher_8s = density_teacher_8s.detach().to(device=decoder_input.device, dtype=torch.float32)
        if projected_control_memory_8s is None:
            assert control_memory_8s is not None
            control_memory_8s = control_memory_8s.detach().to(device=decoder_input.device, dtype=torch.float32)
            if control_memory_8s.ndim != 3 or int(control_memory_8s.shape[1]) != MAPPER_DENSITY_FRAMES:
                raise ValueError(f"control_memory_8s must have shape [B,{MAPPER_DENSITY_FRAMES},D]")
            if int(control_memory_8s.shape[-1]) != self.config.control_dim:
                raise ValueError(
                    f"control_memory_8s last dim must match config.control_dim={self.config.control_dim}, "
                    f"got {control_memory_8s.shape[-1]}"
                )
            control_memory = self.control_projection(control_memory_8s)
        else:
            control_memory = projected_control_memory_8s.detach().to(device=decoder_input.device, dtype=torch.float32)
            if control_memory.ndim != 3 or int(control_memory.shape[1]) != MAPPER_DENSITY_FRAMES:
                raise ValueError(f"projected_control_memory_8s must have shape [B,{MAPPER_DENSITY_FRAMES},D]")
            if int(control_memory.shape[-1]) != self.config.d_model:
                raise ValueError(
                    f"projected_control_memory_8s last dim must match config.d_model={self.config.d_model}, "
                    f"got {control_memory.shape[-1]}"
                )
        if tuple(density_teacher_8s.shape) != (decoder_input.shape[0], MAPPER_DENSITY_FRAMES, 1):
            raise ValueError(f"density_teacher_8s must have shape [B,{MAPPER_DENSITY_FRAMES},1]")
        global_memory, global_memory_padding_mask, global_position_features = self._global_context_memory(
            batch=batch,
            device=device,
            batch_size=int(decoder_input.shape[0]),
            write_start_ms=write_start_ms,
        )
        global_attention_kv_cache = _precomputed_global_attention_kv_cache(
            batch=batch,
            device=device,
            batch_size=int(decoder_input.shape[0]),
            global_memory=global_memory,
            config=self.config,
        )

        decoder_hidden, base_logits = self._decode_with_global_context(
            tokens=decoder_input,
            current_ms=current_ms,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            difficulty=_difficulty_tensor(batch, device=decoder_input.device, dim=self.config.difficulty_dim),
            control_memory=control_memory,
            input_padding_mask=input_padding_mask,
            global_memory=global_memory,
            global_memory_padding_mask=global_memory_padding_mask,
            global_position_features=global_position_features,
            global_attention_kv_cache=global_attention_kv_cache,
        )
        remaining_ms = (write_end_ms.reshape(-1, 1) - current_ms).clamp_min(0)
        state_prior = self.state_prior_adapter(
            open_mask=open_mask,
            open_start_ms=open_start_ms,
            open_age_ms=open_age_ms,
            remaining_ms=remaining_ms,
            write_start_ms=write_start_ms,
        )
        ln_close = self.ln_close_adapter(
            decoder_hidden=decoder_hidden,
            control_memory_8s=control_memory,
            density_teacher_8s=density_teacher_8s,
            current_ms=current_ms,
            write_start_ms=write_start_ms,
            open_mask=open_mask,
            open_start_ms=open_start_ms,
            open_age_ms=open_age_ms,
            remaining_ms=remaining_ms,
        )
        positions = torch.arange(decoder_input.shape[1], dtype=torch.long, device=decoder_input.device).reshape(1, -1)
        grammar_mask = build_grammar_mask(
            current_ms=current_ms,
            open_mask=open_mask,
            open_start_ms=open_start_ms,
            open_age_ms=open_age_ms,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            ln_carry_in=ln_carry_in,
            ln_carry_out=ln_carry_out,
            is_full_chart_start=is_full_chart_start,
            is_full_chart_end=is_full_chart_end,
            vocab=self.vocab,
            positions=positions.expand(decoder_input.shape[0], -1),
        ).to(dtype=base_logits.dtype)
        logits_final = (
            base_logits
            + state_prior.vocab_bias
            + ln_close.event_bias
            + ln_close.time_shift_bias
            + grammar_mask
        )
        return MapperV2ForwardOutput(
            decoder_input_tokens=decoder_input,
            loss_target_tokens=loss_target_tokens,
            state_current_ms=current_ms,
            state_open_mask=open_mask,
            state_open_start_ms=open_start_ms,
            state_open_age_ms=open_age_ms,
            base_logits=base_logits,
            logits_final=logits_final,
            decoder_hidden=decoder_hidden,
            state_prior_bias=state_prior.vocab_bias,
            state_prior_lane_action_bias=state_prior.lane_action_bias,
            ln_close_logits=ln_close.close_logits,
            ln_close_event_bias=ln_close.event_bias,
            ln_close_time_shift_bias=ln_close.time_shift_bias,
            grammar_mask=grammar_mask,
            control_memory_8s=control_memory,
            density_teacher_8s=density_teacher_8s,
            global_memory=global_memory,
            global_memory_padding_mask=global_memory_padding_mask,
            global_attention_gates=self._global_attention_gates(device=device, enabled=global_memory is not None),
            global_position_features=global_position_features,
        )

    def _decode_with_global_context(
        self,
        *,
        tokens: torch.Tensor,
        current_ms: torch.Tensor,
        write_start_ms: torch.Tensor,
        write_end_ms: torch.Tensor,
        difficulty: torch.Tensor,
        control_memory: torch.Tensor,
        input_padding_mask: torch.Tensor | None,
        global_memory: torch.Tensor | None,
        global_memory_padding_mask: torch.Tensor | None,
        global_position_features: torch.Tensor | None,
        global_attention_kv_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, steps = tokens.shape
        if steps > self.config.max_seq_len:
            raise ValueError(f"decoder sequence length {steps} exceeds max_seq_len={self.config.max_seq_len}")
        token_hidden = self.token_embedding(tokens)
        position_hidden = self.position[:, :steps]
        difficulty_hidden = self.difficulty_projection(difficulty).unsqueeze(1)
        time_features = _time_features(
            current_ms=current_ms,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
        )
        hidden = token_hidden + position_hidden + difficulty_hidden + self.time_projection(time_features)
        if global_position_features is not None:
            if self.global_position_projection is None:
                raise ValueError("global_position_projection is required when global_position_features are supplied")
            global_position_input = global_position_features.to(device=tokens.device, dtype=token_hidden.dtype)
            global_position_hidden = self.global_position_projection(global_position_input).unsqueeze(1)
            hidden = hidden + global_position_hidden
        if input_padding_mask is not None:
            hidden = hidden.masked_fill(input_padding_mask.unsqueeze(-1), 0.0)
        causal_mask = torch.triu(
            torch.ones((steps, steps), dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        for layer_index, layer in enumerate(self.decoder_layers):
            hidden = layer(
                tgt=hidden,
                memory=control_memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=input_padding_mask,
                memory_key_padding_mask=None,
            )
            if input_padding_mask is not None:
                hidden = hidden.masked_fill(input_padding_mask.unsqueeze(-1), 0.0)
            if global_memory is not None:
                if global_memory_padding_mask is None:
                    raise ValueError("global_memory_padding_mask is required when global_memory is supplied")
                hidden = self.global_cross_attention_layers[layer_index](
                    hidden=hidden,
                    input_padding_mask=input_padding_mask,
                    global_memory=global_memory,
                    global_memory_padding_mask=global_memory_padding_mask,
                    global_attention_kv=None
                    if global_attention_kv_cache is None
                    else global_attention_kv_cache[layer_index],
                )
        decoder_hidden = self.output_norm(hidden)
        base_logits = self.output_head(decoder_hidden)
        return decoder_hidden, base_logits

    def global_attention_kv_cache(
        self,
        global_memory: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        if not self.config.use_global_context:
            raise ValueError("global attention K/V cache requires global context to be enabled")
        if global_memory.ndim != 3:
            raise ValueError(f"global_memory must have shape [B,G,D], got {tuple(global_memory.shape)}")
        if int(global_memory.shape[-1]) != self.config.d_model:
            raise ValueError(f"global_memory last dim must be {self.config.d_model}, got {global_memory.shape[-1]}")
        device = self.global_cross_attention_layers[0].gate_logit.device
        memory = global_memory.detach().to(device=device, dtype=torch.float32)
        return tuple(block.global_attention_kv_cache(memory) for block in self.global_cross_attention_layers)

    def control_attention_kv_cache(
        self,
        projected_control_memory_8s: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        if projected_control_memory_8s.ndim != 3:
            raise ValueError(
                f"projected_control_memory_8s must have shape [B,{MAPPER_DENSITY_FRAMES},D], "
                f"got {tuple(projected_control_memory_8s.shape)}"
            )
        if int(projected_control_memory_8s.shape[1]) != MAPPER_DENSITY_FRAMES:
            raise ValueError(
                f"projected_control_memory_8s must have {MAPPER_DENSITY_FRAMES} frames, "
                f"got {projected_control_memory_8s.shape[1]}"
            )
        if int(projected_control_memory_8s.shape[-1]) != self.config.d_model:
            raise ValueError(
                f"projected_control_memory_8s last dim must be {self.config.d_model}, "
                f"got {projected_control_memory_8s.shape[-1]}"
            )
        device = self.position.device
        memory = projected_control_memory_8s.detach().to(device=device, dtype=torch.float32)
        return tuple(_transformer_decoder_control_attention_kv_cache(layer, memory) for layer in self.decoder_layers)

    def create_empty_decode_state(
        self,
        *,
        batch_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> MapperV2IncrementalDecodeState:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.config.d_model % self.config.heads != 0:
            raise ValueError("config.d_model must be divisible by config.heads")
        resolved_device = self.position.device if device is None else torch.device(device)
        resolved_dtype = self.position.dtype if dtype is None else dtype
        head_dim = self.config.d_model // self.config.heads
        layer_caches = tuple(
            MapperV2SelfAttentionKVCache(
                key=torch.zeros(
                    (batch_size, self.config.heads, 0, head_dim),
                    dtype=resolved_dtype,
                    device=resolved_device,
                ),
                value=torch.zeros(
                    (batch_size, self.config.heads, 0, head_dim),
                    dtype=resolved_dtype,
                    device=resolved_device,
                ),
            )
            for _ in range(self.config.layers)
        )
        return MapperV2IncrementalDecodeState(self_attention_kv_cache=layer_caches)

    @torch.no_grad()
    def incremental_decode_next_token(
        self,
        *,
        decode_state: MapperV2IncrementalDecodeState,
        decoder_input_token: torch.Tensor,
        current_ms: torch.Tensor,
        open_mask: torch.Tensor,
        open_start_ms: torch.Tensor,
        open_age_ms: torch.Tensor,
        write_start_ms: torch.Tensor,
        write_end_ms: torch.Tensor,
        is_full_chart_start: torch.Tensor,
        is_full_chart_end: torch.Tensor,
        ln_carry_in: Mapping[str, Any],
        ln_carry_out: Mapping[str, Any],
        density_teacher_8s: torch.Tensor,
        control_memory_8s: torch.Tensor | None = None,
        projected_control_memory_8s: torch.Tensor | None = None,
        control_attention_kv_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        difficulty: torch.Tensor | None = None,
        normalized_difficulty: torch.Tensor | None = None,
        global_memory: torch.Tensor | None = None,
        global_memory_padding_mask: torch.Tensor | None = None,
        global_position_features: torch.Tensor | None = None,
        global_attention_kv_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        position: int | torch.Tensor | None = None,
    ) -> MapperV2IncrementalDecodeOutput:
        if self.training:
            raise ValueError("incremental decode is inference-only; call eval() before decoding")
        if control_memory_8s is not None and projected_control_memory_8s is not None:
            raise ValueError("projected_control_memory_8s cannot be supplied with control_memory_8s")

        token = decoder_input_token.to(dtype=torch.long)
        if token.ndim == 1:
            token = token.unsqueeze(1)
        if token.ndim != 2 or int(token.shape[1]) != 1:
            raise ValueError(f"decoder_input_token must have shape [B] or [B,1], got {tuple(token.shape)}")
        batch_size = int(token.shape[0])
        device = token.device
        cache_steps = _validate_incremental_decode_state(
            decode_state,
            batch_size=batch_size,
            layers=self.config.layers,
            heads=self.config.heads,
            head_dim=self.config.d_model // self.config.heads,
        )
        position_index = _decode_position_index(position, default=cache_steps)
        if cache_steps != position_index:
            raise ValueError(f"decode cache length {cache_steps} does not match next position {position_index}")
        if position_index >= self.config.max_seq_len:
            raise ValueError(f"decode position {position_index} exceeds max_seq_len={self.config.max_seq_len}")

        current_ms_step = _as_decode_step_tensor(
            current_ms,
            name="current_ms",
            batch_size=batch_size,
            device=device,
            dtype=torch.long,
        )
        open_mask_step = _as_decode_step_lane_tensor(
            open_mask,
            name="open_mask",
            batch_size=batch_size,
            device=device,
            dtype=torch.bool,
        )
        open_start_ms_step = _as_decode_step_lane_tensor(
            open_start_ms,
            name="open_start_ms",
            batch_size=batch_size,
            device=device,
            dtype=torch.long,
        )
        open_age_ms_step = _as_decode_step_lane_tensor(
            open_age_ms,
            name="open_age_ms",
            batch_size=batch_size,
            device=device,
            dtype=torch.long,
        )
        write_start_ms_step = _as_decode_batch_vector(
            write_start_ms,
            name="write_start_ms",
            batch_size=batch_size,
            device=device,
            dtype=torch.long,
        )
        write_end_ms_step = _as_decode_batch_vector(
            write_end_ms,
            name="write_end_ms",
            batch_size=batch_size,
            device=device,
            dtype=torch.long,
        )
        is_full_chart_start_step = _as_decode_batch_vector(
            is_full_chart_start,
            name="is_full_chart_start",
            batch_size=batch_size,
            device=device,
            dtype=torch.bool,
        )
        is_full_chart_end_step = _as_decode_batch_vector(
            is_full_chart_end,
            name="is_full_chart_end",
            batch_size=batch_size,
            device=device,
            dtype=torch.bool,
        )
        difficulty_source = {}
        if normalized_difficulty is not None:
            difficulty_source["normalized_difficulty"] = normalized_difficulty
        elif difficulty is not None:
            difficulty_source["difficulty"] = difficulty
        difficulty_step = _difficulty_tensor(difficulty_source, device=device, dim=self.config.difficulty_dim)
        if int(difficulty_step.shape[0]) != batch_size:
            raise ValueError(f"difficulty batch must be {batch_size}, got {difficulty_step.shape[0]}")

        density_teacher = density_teacher_8s.detach().to(device=device, dtype=torch.float32)
        if tuple(density_teacher.shape) != (batch_size, MAPPER_DENSITY_FRAMES, 1):
            raise ValueError(f"density_teacher_8s must have shape [B,{MAPPER_DENSITY_FRAMES},1]")
        if projected_control_memory_8s is None:
            if control_memory_8s is None:
                raise ValueError("control_memory_8s or projected_control_memory_8s is required")
            raw_control_memory = control_memory_8s.detach().to(device=device, dtype=torch.float32)
            if raw_control_memory.ndim != 3 or int(raw_control_memory.shape[1]) != MAPPER_DENSITY_FRAMES:
                raise ValueError(f"control_memory_8s must have shape [B,{MAPPER_DENSITY_FRAMES},D]")
            if tuple(raw_control_memory.shape[:1]) != (batch_size,):
                raise ValueError(f"control_memory_8s batch must be {batch_size}, got {raw_control_memory.shape[0]}")
            if int(raw_control_memory.shape[-1]) != self.config.control_dim:
                raise ValueError(
                    f"control_memory_8s last dim must match config.control_dim={self.config.control_dim}, "
                    f"got {raw_control_memory.shape[-1]}"
                )
            control_memory = self.control_projection(raw_control_memory)
        else:
            control_memory = projected_control_memory_8s.detach().to(device=device, dtype=torch.float32)
            if control_memory.ndim != 3 or int(control_memory.shape[1]) != MAPPER_DENSITY_FRAMES:
                raise ValueError(f"projected_control_memory_8s must have shape [B,{MAPPER_DENSITY_FRAMES},D]")
            if tuple(control_memory.shape[:1]) != (batch_size,):
                raise ValueError(f"projected_control_memory_8s batch must be {batch_size}, got {control_memory.shape[0]}")
            if int(control_memory.shape[-1]) != self.config.d_model:
                raise ValueError(
                    f"projected_control_memory_8s last dim must match config.d_model={self.config.d_model}, "
                    f"got {control_memory.shape[-1]}"
                )

        control_attention_kv = _precomputed_control_attention_kv_cache(
            control_attention_kv_cache=control_attention_kv_cache,
            device=device,
            batch_size=batch_size,
            control_memory=control_memory,
            config=self.config,
        )

        if global_position_features is not None:
            if not isinstance(global_position_features, torch.Tensor):
                raise ValueError("global_position_features must be a torch.Tensor")
            global_position_features = global_position_features.detach().to(device=device, dtype=torch.float32)
            if tuple(global_position_features.shape) != (batch_size, GLOBAL_POSITION_FEATURES):
                raise ValueError(
                    f"global_position_features must have shape [{batch_size},{GLOBAL_POSITION_FEATURES}], "
                    f"got {tuple(global_position_features.shape)}"
                )

        global_attention_kv = None
        if global_memory is not None:
            if not self.config.use_global_context:
                raise ValueError("global_memory cannot be supplied when global context is disabled")
            global_memory = global_memory.detach().to(device=device, dtype=torch.float32)
            if global_memory.ndim != 3 or tuple(global_memory.shape[:1]) != (batch_size,):
                raise ValueError(f"global_memory must have shape [B,G,D], got {tuple(global_memory.shape)}")
            if int(global_memory.shape[-1]) != self.config.d_model:
                raise ValueError(f"global_memory last dim must be {self.config.d_model}, got {global_memory.shape[-1]}")
            if global_memory_padding_mask is None:
                raise ValueError("global_memory_padding_mask is required when global_memory is supplied")
            global_memory_padding_mask = global_memory_padding_mask.detach().to(device=device, dtype=torch.bool)
            if tuple(global_memory_padding_mask.shape) != tuple(global_memory.shape[:2]):
                raise ValueError("global_memory_padding_mask must have shape [B,G]")
            if global_attention_kv_cache is not None:
                global_attention_kv = _precomputed_global_attention_kv_cache(
                    batch={"global_attention_kv_cache": global_attention_kv_cache},
                    device=device,
                    batch_size=batch_size,
                    global_memory=global_memory,
                    config=self.config,
                )
        elif global_attention_kv_cache is not None:
            raise ValueError("global_memory is required when global_attention_kv_cache is supplied")

        hidden, next_layer_caches = self._incremental_decode_hidden_next_token(
            token=token,
            current_ms=current_ms_step,
            write_start_ms=write_start_ms_step,
            write_end_ms=write_end_ms_step,
            difficulty=difficulty_step,
            control_memory=control_memory,
            decode_state=decode_state,
            position_index=position_index,
            global_memory=global_memory,
            global_memory_padding_mask=global_memory_padding_mask,
            global_position_features=global_position_features,
            global_attention_kv_cache=global_attention_kv,
            control_attention_kv_cache=control_attention_kv,
        )
        decoder_hidden = self.output_norm(hidden)
        base_logits = self.output_head(decoder_hidden)
        remaining_ms = (write_end_ms_step.reshape(-1, 1) - current_ms_step).clamp_min(0)
        state_prior = self.state_prior_adapter(
            open_mask=open_mask_step,
            open_start_ms=open_start_ms_step,
            open_age_ms=open_age_ms_step,
            remaining_ms=remaining_ms,
            write_start_ms=write_start_ms_step,
        )
        ln_close = self.ln_close_adapter(
            decoder_hidden=decoder_hidden,
            control_memory_8s=control_memory,
            density_teacher_8s=density_teacher,
            current_ms=current_ms_step,
            write_start_ms=write_start_ms_step,
            open_mask=open_mask_step,
            open_start_ms=open_start_ms_step,
            open_age_ms=open_age_ms_step,
            remaining_ms=remaining_ms,
        )
        positions = torch.full((batch_size, 1), position_index, dtype=torch.long, device=device)
        grammar_mask = build_grammar_mask(
            current_ms=current_ms_step,
            open_mask=open_mask_step,
            open_start_ms=open_start_ms_step,
            open_age_ms=open_age_ms_step,
            write_start_ms=write_start_ms_step,
            write_end_ms=write_end_ms_step,
            ln_carry_in=ln_carry_in,
            ln_carry_out=ln_carry_out,
            is_full_chart_start=is_full_chart_start_step,
            is_full_chart_end=is_full_chart_end_step,
            vocab=self.vocab,
            positions=positions,
        ).to(dtype=base_logits.dtype)
        logits_final = (
            base_logits
            + state_prior.vocab_bias
            + ln_close.event_bias
            + ln_close.time_shift_bias
            + grammar_mask
        )
        return MapperV2IncrementalDecodeOutput(
            decode_state=MapperV2IncrementalDecodeState(self_attention_kv_cache=tuple(next_layer_caches)),
            decoder_input_token=token[:, 0],
            position=positions[:, 0],
            base_logits=base_logits[:, 0],
            logits_final=logits_final[:, 0],
            decoder_hidden=decoder_hidden[:, 0],
            state_prior_bias=state_prior.vocab_bias[:, 0],
            state_prior_lane_action_bias=state_prior.lane_action_bias[:, 0],
            ln_close_logits=ln_close.close_logits[:, 0],
            ln_close_event_bias=ln_close.event_bias[:, 0],
            ln_close_time_shift_bias=ln_close.time_shift_bias[:, 0],
            grammar_mask=grammar_mask[:, 0],
            global_attention_gates=self._global_attention_gates(device=device, enabled=global_memory is not None),
        )

    def _incremental_decode_hidden_next_token(
        self,
        *,
        token: torch.Tensor,
        current_ms: torch.Tensor,
        write_start_ms: torch.Tensor,
        write_end_ms: torch.Tensor,
        difficulty: torch.Tensor,
        control_memory: torch.Tensor,
        decode_state: MapperV2IncrementalDecodeState,
        position_index: int,
        global_memory: torch.Tensor | None,
        global_memory_padding_mask: torch.Tensor | None,
        global_position_features: torch.Tensor | None,
        global_attention_kv_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None,
        control_attention_kv_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None,
    ) -> tuple[torch.Tensor, list[MapperV2SelfAttentionKVCache]]:
        token_hidden = self.token_embedding(token)
        position_hidden = self.position[:, position_index : position_index + 1]
        difficulty_hidden = self.difficulty_projection(difficulty).unsqueeze(1)
        time_features = _time_features(
            current_ms=current_ms,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
        )
        hidden = token_hidden + position_hidden + difficulty_hidden + self.time_projection(time_features)
        if global_position_features is not None:
            if self.global_position_projection is None:
                raise ValueError("global_position_projection is required when global_position_features are supplied")
            global_position_input = global_position_features.to(device=token.device, dtype=token_hidden.dtype)
            hidden = hidden + self.global_position_projection(global_position_input).unsqueeze(1)

        next_layer_caches: list[MapperV2SelfAttentionKVCache] = []
        for layer_index, layer in enumerate(self.decoder_layers):
            hidden, layer_cache = _incremental_transformer_decoder_layer_step(
                layer,
                hidden=hidden,
                memory=control_memory,
                self_attention_kv=decode_state.self_attention_kv_cache[layer_index],
                control_attention_kv=None
                if control_attention_kv_cache is None
                else control_attention_kv_cache[layer_index],
            )
            next_layer_caches.append(layer_cache)
            if global_memory is not None:
                if global_memory_padding_mask is None:
                    raise ValueError("global_memory_padding_mask is required when global_memory is supplied")
                hidden = self.global_cross_attention_layers[layer_index](
                    hidden=hidden,
                    input_padding_mask=None,
                    global_memory=global_memory,
                    global_memory_padding_mask=global_memory_padding_mask,
                    global_attention_kv=None
                    if global_attention_kv_cache is None
                    else global_attention_kv_cache[layer_index],
                )
        return hidden, next_layer_caches

    def _global_context_memory(
        self,
        *,
        batch: Mapping[str, Any],
        device: torch.device,
        batch_size: int,
        write_start_ms: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not self.config.use_global_context:
            return None, None, None
        if self.global_encoder is None:
            raise ValueError("global context is enabled but global_encoder is missing")
        cached = _precomputed_global_context_memory(
            batch=batch,
            device=device,
            batch_size=batch_size,
            config=self.config,
        )
        if cached is not None:
            return cached
        full_mel = _require_tensor(batch, "full_mel", ndim=3).to(device=device, dtype=torch.float32)
        full_dense_timing_v2 = _require_tensor(batch, "full_dense_timing_v2", ndim=3).to(device=device, dtype=torch.float32)
        padding_mask = _require_tensor(batch, "padding_mask", ndim=2).to(device=device, dtype=torch.bool)
        raw_padded_frame_count = _require_tensor(batch, "frame_count", ndim=1)
        if (
            raw_padded_frame_count.dtype == torch.bool
            or raw_padded_frame_count.dtype.is_floating_point
            or raw_padded_frame_count.dtype.is_complex
        ):
            raise ValueError("frame_count must be an integer tensor")
        raw_source_frame_count = _require_tensor(batch, "source_frame_count", ndim=1)
        if (
            raw_source_frame_count.dtype == torch.bool
            or raw_source_frame_count.dtype.is_floating_point
            or raw_source_frame_count.dtype.is_complex
        ):
            raise ValueError("source_frame_count must be an integer tensor")
        padded_frame_count = raw_padded_frame_count.to(device=device, dtype=torch.long)
        source_frame_count = raw_source_frame_count.to(device=device, dtype=torch.long)
        target_start_frame = _target_start_frame(batch=batch, write_start_ms=write_start_ms, device=device)
        _validate_global_context_inputs(
            full_mel=full_mel,
            full_dense_timing_v2=full_dense_timing_v2,
            padding_mask=padding_mask,
            padded_frame_count=padded_frame_count,
            source_frame_count=source_frame_count,
            target_start_frame=target_start_frame,
            batch_size=batch_size,
            config=self.config,
        )
        global_position_features = _global_position_features(
            source_frame_count=source_frame_count,
            target_start_frame=target_start_frame,
            device=device,
        )
        global_memory, global_memory_padding_mask, _ = self.global_encoder(
            full_mel=full_mel,
            full_dense_timing_v2=full_dense_timing_v2,
            padding_mask=padding_mask,
            frame_count=source_frame_count,
            target_start_frame=target_start_frame,
        )
        return global_memory, global_memory_padding_mask, global_position_features

    def _global_attention_gates(self, *, device: torch.device, enabled: bool) -> torch.Tensor | None:
        if not enabled:
            return None
        return torch.stack(
            [torch.sigmoid(block.gate_logit) for block in self.global_cross_attention_layers],
        ).to(device=device)


class _MapperGlobalCrossAttentionBlock(nn.Module):
    def __init__(self, *, d_model: int, heads: int, dropout: float, gate_init: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self,
        *,
        hidden: torch.Tensor,
        input_padding_mask: torch.Tensor | None,
        global_memory: torch.Tensor,
        global_memory_padding_mask: torch.Tensor,
        global_attention_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        query = self.query_norm(hidden)
        if global_attention_kv is None:
            memory = self.memory_norm(global_memory)
            cross, _ = self.cross_attn(
                query=query,
                key=memory,
                value=memory,
                key_padding_mask=global_memory_padding_mask,
                need_weights=False,
            )
        else:
            cross = self._cached_cross_attention(
                query=query,
                global_attention_kv=global_attention_kv,
                global_memory_padding_mask=global_memory_padding_mask,
            )
        hidden = hidden + torch.sigmoid(self.gate_logit).to(dtype=hidden.dtype) * self.dropout(cross)
        return _mask_hidden(hidden, input_padding_mask)

    def global_attention_kv_cache(self, global_memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        memory = self.memory_norm(global_memory.detach().to(device=self.gate_logit.device, dtype=torch.float32))
        _, key_weight, value_weight = self.cross_attn.in_proj_weight.chunk(3, dim=0)
        in_proj_bias = self.cross_attn.in_proj_bias
        if in_proj_bias is None:
            key_bias = value_bias = None
        else:
            _, key_bias, value_bias = in_proj_bias.chunk(3, dim=0)
        key = _attention_projection_to_heads(
            F.linear(memory, key_weight, key_bias),
            heads=self.cross_attn.num_heads,
        )
        value = _attention_projection_to_heads(
            F.linear(memory, value_weight, value_bias),
            heads=self.cross_attn.num_heads,
        )
        return key.detach().contiguous(), value.detach().contiguous()

    def _cached_cross_attention(
        self,
        *,
        query: torch.Tensor,
        global_attention_kv: tuple[torch.Tensor, torch.Tensor],
        global_memory_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        key, value = global_attention_kv
        batch_size, steps, d_model = query.shape
        heads = int(self.cross_attn.num_heads)
        head_dim = d_model // heads
        expected_shape = (batch_size, heads, int(global_memory_padding_mask.shape[1]), head_dim)
        if tuple(key.shape) != expected_shape:
            raise ValueError(f"global attention key cache must have shape {expected_shape}, got {tuple(key.shape)}")
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"global attention value cache must have shape {expected_shape}, got {tuple(value.shape)}")

        query_weight = self.cross_attn.in_proj_weight[:d_model]
        in_proj_bias = self.cross_attn.in_proj_bias
        query_bias = None if in_proj_bias is None else in_proj_bias[:d_model]
        projected_query = _attention_projection_to_heads(
            F.linear(query, query_weight, query_bias),
            heads=heads,
        )
        attention_mask = _global_key_padding_attention_mask(
            global_memory_padding_mask,
            dtype=projected_query.dtype,
            device=projected_query.device,
        )
        attention = F.scaled_dot_product_attention(
            projected_query,
            key.to(device=projected_query.device, dtype=projected_query.dtype),
            value.to(device=projected_query.device, dtype=projected_query.dtype),
            attn_mask=attention_mask,
            dropout_p=float(self.cross_attn.dropout) if self.training else 0.0,
            is_causal=False,
        )
        attention = attention.transpose(1, 2).contiguous().view(batch_size, steps, d_model)
        return self.cross_attn.out_proj(attention)


def _incremental_transformer_decoder_layer_step(
    layer: nn.TransformerDecoderLayer,
    *,
    hidden: torch.Tensor,
    memory: torch.Tensor,
    self_attention_kv: MapperV2SelfAttentionKVCache,
    control_attention_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, MapperV2SelfAttentionKVCache]:
    if getattr(layer.self_attn, "batch_first", False) is not True:
        raise ValueError("incremental decode requires batch_first decoder self-attention")
    if layer.norm_first:
        self_attn, updated_cache = _incremental_self_attention_step(
            layer,
            query=layer.norm1(hidden),
            self_attention_kv=self_attention_kv,
        )
        hidden = hidden + self_attn
        cross_query = layer.norm2(hidden)
        if control_attention_kv is None:
            cross_attn = layer._mha_block(cross_query, memory, None, None, False)
        else:
            cross_attn = _cached_control_cross_attention_step(
                layer,
                query=cross_query,
                control_attention_kv=control_attention_kv,
            )
        hidden = hidden + cross_attn
        hidden = hidden + layer._ff_block(layer.norm3(hidden))
        return hidden, updated_cache

    self_attn, updated_cache = _incremental_self_attention_step(
        layer,
        query=hidden,
        self_attention_kv=self_attention_kv,
    )
    hidden = layer.norm1(hidden + self_attn)
    if control_attention_kv is None:
        cross_attn = layer._mha_block(hidden, memory, None, None, False)
    else:
        cross_attn = _cached_control_cross_attention_step(
            layer,
            query=hidden,
            control_attention_kv=control_attention_kv,
        )
    hidden = layer.norm2(hidden + cross_attn)
    hidden = layer.norm3(hidden + layer._ff_block(hidden))
    return hidden, updated_cache


def _transformer_decoder_control_attention_kv_cache(
    layer: nn.TransformerDecoderLayer,
    memory: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cross_attn = layer.multihead_attn
    if getattr(cross_attn, "batch_first", False) is not True:
        raise ValueError("control attention K/V cache requires batch_first decoder cross-attention")
    if cross_attn.in_proj_weight is None:
        raise ValueError("control attention K/V cache requires packed cross-attention projection weights")
    if memory.ndim != 3:
        raise ValueError(f"control memory must have shape [B,T,D], got {tuple(memory.shape)}")
    batch_size, source_steps, d_model = memory.shape
    heads = int(cross_attn.num_heads)
    if d_model % heads != 0:
        raise ValueError("control attention embed dim must be divisible by head count")
    if int(cross_attn.embed_dim) != int(d_model):
        raise ValueError(f"control attention embed dim must be {cross_attn.embed_dim}, got {d_model}")

    _, key_weight, value_weight = cross_attn.in_proj_weight.chunk(3, dim=0)
    in_proj_bias = cross_attn.in_proj_bias
    if in_proj_bias is None:
        key_bias = value_bias = None
    else:
        _, key_bias, value_bias = in_proj_bias.chunk(3, dim=0)
    key = _attention_projection_to_heads(
        F.linear(memory, key_weight, key_bias),
        heads=heads,
    )
    value = _attention_projection_to_heads(
        F.linear(memory, value_weight, value_bias),
        heads=heads,
    )
    expected_steps = int(source_steps)
    if int(key.shape[2]) != expected_steps or int(value.shape[2]) != expected_steps:
        raise ValueError("control attention K/V cache source length mismatch")
    return key.detach().contiguous(), value.detach().contiguous()


def _cached_control_cross_attention_step(
    layer: nn.TransformerDecoderLayer,
    *,
    query: torch.Tensor,
    control_attention_kv: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    cross_attn = layer.multihead_attn
    if cross_attn.in_proj_weight is None:
        raise ValueError("cached control cross-attention requires packed projection weights")
    batch_size, steps, d_model = query.shape
    if steps != 1:
        raise ValueError(f"cached control cross-attention query must have one step, got {steps}")
    heads = int(cross_attn.num_heads)
    if d_model % heads != 0:
        raise ValueError("control cross-attention embed dim must be divisible by head count")
    head_dim = d_model // heads
    key, value = control_attention_kv
    expected_prefix = (batch_size, heads)
    if key.ndim != 4 or tuple(key.shape[:2]) != expected_prefix or int(key.shape[-1]) != head_dim:
        raise ValueError(
            f"control attention key cache must have shape [B,{heads},T,{head_dim}], got {tuple(key.shape)}"
        )
    if tuple(value.shape) != tuple(key.shape):
        raise ValueError(f"control attention value cache must match key cache shape, got {tuple(value.shape)}")

    query_weight = cross_attn.in_proj_weight[:d_model]
    in_proj_bias = cross_attn.in_proj_bias
    query_bias = None if in_proj_bias is None else in_proj_bias[:d_model]
    projected_query = _attention_projection_to_heads(
        F.linear(query, query_weight, query_bias),
        heads=heads,
    )
    attention = F.scaled_dot_product_attention(
        projected_query,
        key.to(device=projected_query.device, dtype=projected_query.dtype),
        value.to(device=projected_query.device, dtype=projected_query.dtype),
        attn_mask=None,
        dropout_p=float(cross_attn.dropout) if layer.training else 0.0,
        is_causal=False,
    )
    attention = attention.transpose(1, 2).contiguous().view(batch_size, steps, d_model)
    return layer.dropout2(cross_attn.out_proj(attention))


def _incremental_self_attention_step(
    layer: nn.TransformerDecoderLayer,
    *,
    query: torch.Tensor,
    self_attention_kv: MapperV2SelfAttentionKVCache,
) -> tuple[torch.Tensor, MapperV2SelfAttentionKVCache]:
    self_attn = layer.self_attn
    if self_attn.in_proj_weight is None:
        raise ValueError("incremental decode requires packed self-attention projection weights")
    batch_size, steps, d_model = query.shape
    if steps != 1:
        raise ValueError(f"incremental self-attention query must have one step, got {steps}")
    heads = int(self_attn.num_heads)
    if d_model % heads != 0:
        raise ValueError("self-attention embed dim must be divisible by head count")
    head_dim = d_model // heads
    previous_key = self_attention_kv.key.to(device=query.device, dtype=query.dtype)
    previous_value = self_attention_kv.value.to(device=query.device, dtype=query.dtype)
    if tuple(previous_key.shape[:2]) != (batch_size, heads) or int(previous_key.shape[-1]) != head_dim:
        raise ValueError(
            f"self-attention key cache must have shape [B,{heads},T,{head_dim}], "
            f"got {tuple(self_attention_kv.key.shape)}"
        )
    if tuple(previous_value.shape) != tuple(previous_key.shape):
        raise ValueError(
            f"self-attention value cache must match key cache shape, got {tuple(self_attention_kv.value.shape)}"
        )

    query_weight, key_weight, value_weight = self_attn.in_proj_weight.chunk(3, dim=0)
    in_proj_bias = self_attn.in_proj_bias
    if in_proj_bias is None:
        query_bias = key_bias = value_bias = None
    else:
        query_bias, key_bias, value_bias = in_proj_bias.chunk(3, dim=0)
    projected_query = _attention_projection_to_heads(
        F.linear(query, query_weight, query_bias),
        heads=heads,
    )
    current_key = _attention_projection_to_heads(
        F.linear(query, key_weight, key_bias),
        heads=heads,
    )
    current_value = _attention_projection_to_heads(
        F.linear(query, value_weight, value_bias),
        heads=heads,
    )
    key = torch.cat((previous_key, current_key), dim=2).contiguous()
    value = torch.cat((previous_value, current_value), dim=2).contiguous()
    attention = F.scaled_dot_product_attention(
        projected_query,
        key,
        value,
        attn_mask=None,
        dropout_p=float(self_attn.dropout) if layer.training else 0.0,
        is_causal=False,
    )
    attention = attention.transpose(1, 2).contiguous().view(batch_size, 1, d_model)
    output = self_attn.out_proj(attention)
    output = layer.dropout1(output)
    return output, MapperV2SelfAttentionKVCache(key=key.detach(), value=value.detach())


def _global_encoder_config(config: MapperV2Config) -> ControlDemoGlobalEncoderConfig:
    return ControlDemoGlobalEncoderConfig(
        mel_dim=config.mel_dim,
        timing_dim=config.timing_dim,
        target_frames=MAPPER_DENSITY_FRAMES,
        d_model=config.d_model,
        heads=config.heads,
        dropout=config.dropout,
        global_stride=config.global_stride,
        global_layers=config.global_layers,
        global_ffn_dim=config.ffn_dim if config.global_ffn_dim is None else int(config.global_ffn_dim),
        global_conv_blocks=config.global_conv_blocks,
        conv_kernel_size=config.global_conv_kernel_size,
    )


def _target_start_frame(
    *,
    batch: Mapping[str, Any],
    write_start_ms: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    value = batch.get("target_start_frame")
    if value is None:
        return (write_start_ms // MAPPER_DENSITY_FRAME_MS).to(device=device, dtype=torch.long)
    if not isinstance(value, torch.Tensor):
        raise ValueError("target_start_frame must be a torch.Tensor")
    if value.ndim != 1:
        raise ValueError(f"target_start_frame must have shape [B], got {tuple(value.shape)}")
    if value.dtype == torch.bool or value.dtype.is_floating_point or value.dtype.is_complex:
        raise ValueError("target_start_frame must be an integer tensor")
    return value.to(device=device, dtype=torch.long)


def _precomputed_global_context_memory(
    *,
    batch: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
    config: MapperV2Config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    global_memory = batch.get("global_memory")
    if global_memory is None:
        return None
    if not isinstance(global_memory, torch.Tensor):
        raise ValueError("global_memory must be a torch.Tensor")
    global_padding_mask = batch.get("global_memory_padding_mask")
    if not isinstance(global_padding_mask, torch.Tensor):
        raise ValueError("global_memory_padding_mask is required when global_memory is supplied")
    global_position_features = batch.get("global_position_features")
    if not isinstance(global_position_features, torch.Tensor):
        raise ValueError("global_position_features is required when global_memory is supplied")

    memory = global_memory.detach().to(device=device, dtype=torch.float32)
    padding_mask = global_padding_mask.detach().to(device=device, dtype=torch.bool)
    position_features = global_position_features.detach().to(device=device, dtype=torch.float32)
    if memory.ndim != 3:
        raise ValueError(f"global_memory must have shape [B,G,D], got {tuple(memory.shape)}")
    if int(memory.shape[0]) != int(batch_size):
        raise ValueError(f"global_memory batch must be {batch_size}, got {memory.shape[0]}")
    if int(memory.shape[-1]) != int(config.d_model):
        raise ValueError(f"global_memory last dim must be {config.d_model}, got {memory.shape[-1]}")
    if tuple(padding_mask.shape) != tuple(memory.shape[:2]):
        raise ValueError("global_memory_padding_mask must have shape [B,G]")
    if tuple(position_features.shape) != (int(batch_size), GLOBAL_POSITION_FEATURES):
        raise ValueError(
            f"global_position_features must have shape [{batch_size},{GLOBAL_POSITION_FEATURES}], "
            f"got {tuple(position_features.shape)}"
        )
    return memory, padding_mask, position_features


def _precomputed_control_attention_kv_cache(
    *,
    control_attention_kv_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None,
    device: torch.device,
    batch_size: int,
    control_memory: torch.Tensor,
    config: MapperV2Config,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...] | None:
    if control_attention_kv_cache is None:
        return None
    if not isinstance(control_attention_kv_cache, (tuple, list)):
        raise ValueError("control_attention_kv_cache must be a tuple/list of per-layer key/value tensors")
    if len(control_attention_kv_cache) != config.layers:
        raise ValueError(
            f"control_attention_kv_cache must contain {config.layers} layers, "
            f"got {len(control_attention_kv_cache)}"
        )
    if config.d_model % config.heads != 0:
        raise ValueError("config.d_model must be divisible by config.heads")

    head_dim = config.d_model // config.heads
    source_steps = int(control_memory.shape[1])
    expected_shape = (int(batch_size), int(config.heads), source_steps, head_dim)
    cache: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_index, layer_cache in enumerate(control_attention_kv_cache):
        if not isinstance(layer_cache, (tuple, list)) or len(layer_cache) != 2:
            raise ValueError(f"control_attention_kv_cache layer {layer_index} must be a key/value pair")
        key, value = layer_cache
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise ValueError(f"control_attention_kv_cache layer {layer_index} key/value must be tensors")
        key = key.detach().to(device=device, dtype=torch.float32)
        value = value.detach().to(device=device, dtype=torch.float32)
        if tuple(key.shape) != expected_shape:
            raise ValueError(
                f"control_attention_kv_cache layer {layer_index} key must have shape {expected_shape}, "
                f"got {tuple(key.shape)}"
            )
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"control_attention_kv_cache layer {layer_index} value must have shape {expected_shape}, "
                f"got {tuple(value.shape)}"
            )
        cache.append((key.contiguous(), value.contiguous()))
    return tuple(cache)


def _precomputed_global_attention_kv_cache(
    *,
    batch: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
    global_memory: torch.Tensor | None,
    config: MapperV2Config,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...] | None:
    raw_cache = batch.get("global_attention_kv_cache")
    if raw_cache is None:
        return None
    if global_memory is None:
        raise ValueError("global_memory is required when global_attention_kv_cache is supplied")
    if not isinstance(raw_cache, (tuple, list)):
        raise ValueError("global_attention_kv_cache must be a tuple/list of per-layer key/value tensors")
    if len(raw_cache) != config.layers:
        raise ValueError(f"global_attention_kv_cache must contain {config.layers} layers, got {len(raw_cache)}")
    if config.d_model % config.heads != 0:
        raise ValueError("config.d_model must be divisible by config.heads")

    head_dim = config.d_model // config.heads
    source_steps = int(global_memory.shape[1])
    expected_shape = (int(batch_size), int(config.heads), source_steps, head_dim)
    cache: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_index, layer_cache in enumerate(raw_cache):
        if not isinstance(layer_cache, (tuple, list)) or len(layer_cache) != 2:
            raise ValueError(f"global_attention_kv_cache layer {layer_index} must be a key/value pair")
        key, value = layer_cache
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise ValueError(f"global_attention_kv_cache layer {layer_index} key/value must be tensors")
        key = key.detach().to(device=device, dtype=torch.float32)
        value = value.detach().to(device=device, dtype=torch.float32)
        if tuple(key.shape) != expected_shape:
            raise ValueError(
                f"global_attention_kv_cache layer {layer_index} key must have shape {expected_shape}, "
                f"got {tuple(key.shape)}"
            )
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"global_attention_kv_cache layer {layer_index} value must have shape {expected_shape}, "
                f"got {tuple(value.shape)}"
            )
        cache.append((key.contiguous(), value.contiguous()))
    return tuple(cache)


def _decode_position_index(position: int | torch.Tensor | None, *, default: int) -> int:
    if position is None:
        return int(default)
    if isinstance(position, torch.Tensor):
        values = position.detach().reshape(-1)
        if int(values.numel()) != 1:
            raise ValueError("position tensor must contain one value")
        position_value = int(values[0].item())
    else:
        position_value = int(position)
    if position_value < 0:
        raise ValueError(f"position must be non-negative, got {position_value}")
    return position_value


def _validate_incremental_decode_state(
    state: MapperV2IncrementalDecodeState,
    *,
    batch_size: int,
    layers: int,
    heads: int,
    head_dim: int,
) -> int:
    if not isinstance(state, MapperV2IncrementalDecodeState):
        raise ValueError("decode_state must be a MapperV2IncrementalDecodeState")
    if len(state.self_attention_kv_cache) != int(layers):
        raise ValueError(
            f"decode_state must contain {layers} layer caches, got {len(state.self_attention_kv_cache)}"
        )
    sequence_length: int | None = None
    for layer_index, layer_cache in enumerate(state.self_attention_kv_cache):
        if not isinstance(layer_cache, MapperV2SelfAttentionKVCache):
            raise ValueError(f"decode_state layer {layer_index} must be a MapperV2SelfAttentionKVCache")
        key = layer_cache.key
        value = layer_cache.value
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise ValueError(f"decode_state layer {layer_index} key/value must be tensors")
        if key.ndim != 4:
            raise ValueError(f"decode_state layer {layer_index} key must have shape [B,H,T,Dh]")
        expected_prefix = (int(batch_size), int(heads))
        if tuple(key.shape[:2]) != expected_prefix or int(key.shape[-1]) != int(head_dim):
            raise ValueError(
                f"decode_state layer {layer_index} key must have shape [B,{heads},T,{head_dim}], "
                f"got {tuple(key.shape)}"
            )
        if tuple(value.shape) != tuple(key.shape):
            raise ValueError(
                f"decode_state layer {layer_index} value must match key shape, got {tuple(value.shape)}"
            )
        layer_steps = int(key.shape[2])
        if sequence_length is None:
            sequence_length = layer_steps
        elif sequence_length != layer_steps:
            raise ValueError("decode_state layer cache lengths must match")
    return 0 if sequence_length is None else int(sequence_length)


def _as_decode_step_tensor(
    value: torch.Tensor,
    *,
    name: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    tensor = value.to(device=device, dtype=dtype)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 2 or tuple(tensor.shape) != (int(batch_size), 1):
        raise ValueError(f"{name} must have shape [B] or [B,1], got {tuple(value.shape)}")
    return tensor.contiguous()


def _as_decode_step_lane_tensor(
    value: torch.Tensor,
    *,
    name: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    tensor = value.to(device=device, dtype=dtype)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 3 or tuple(tensor.shape) != (int(batch_size), 1, 4):
        raise ValueError(f"{name} must have shape [B,4] or [B,1,4], got {tuple(value.shape)}")
    return tensor.contiguous()


def _as_decode_batch_vector(
    value: torch.Tensor,
    *,
    name: str,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    tensor = value.to(device=device, dtype=dtype).reshape(-1)
    if int(tensor.numel()) == 1:
        return tensor.expand(int(batch_size)).contiguous()
    if int(tensor.numel()) != int(batch_size):
        raise ValueError(f"{name} must contain 1 or {batch_size} values, got {tensor.numel()}")
    return tensor.contiguous()


def _attention_projection_to_heads(projection: torch.Tensor, *, heads: int) -> torch.Tensor:
    if projection.ndim != 3:
        raise ValueError(f"attention projection must have shape [B,S,D], got {tuple(projection.shape)}")
    if projection.shape[-1] % int(heads) != 0:
        raise ValueError("attention projection width must be divisible by head count")
    batch_size, steps, width = projection.shape
    head_dim = int(width) // int(heads)
    return projection.view(batch_size, steps, int(heads), head_dim).transpose(1, 2).contiguous()


def _global_key_padding_attention_mask(
    padding_mask: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if padding_mask.ndim != 2:
        raise ValueError(f"global_memory_padding_mask must have shape [B,G], got {tuple(padding_mask.shape)}")
    if padding_mask.dtype != torch.bool:
        raise ValueError("global_memory_padding_mask must be bool")
    mask = padding_mask.to(device=device, dtype=torch.bool).reshape(padding_mask.shape[0], 1, 1, padding_mask.shape[1])
    return torch.zeros(mask.shape, dtype=dtype, device=device).masked_fill(mask, float("-inf"))


def _global_position_features(
    *,
    source_frame_count: torch.Tensor,
    target_start_frame: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    source_count_f = source_frame_count.to(device=device, dtype=torch.float32)
    target_start_f = target_start_frame.to(device=device, dtype=torch.float32)
    denominator = (source_count_f - 1.0).clamp_min(1.0)
    target_center = target_start_f + 0.5 * float(MAPPER_DENSITY_FRAMES)
    song_seconds = source_count_f * (float(MAPPER_DENSITY_FRAME_MS) / 1000.0)
    seconds_to_end = (source_count_f - target_center).clamp_min(0.0) * (
        float(MAPPER_DENSITY_FRAME_MS) / 1000.0
    )
    return torch.stack(
        (
            target_start_f / denominator,
            target_center / denominator,
            seconds_to_end / 300.0,
            torch.log1p(song_seconds) / 6.0,
        ),
        dim=-1,
    ).to(dtype=torch.float32)


def _validate_global_context_inputs(
    *,
    full_mel: torch.Tensor,
    full_dense_timing_v2: torch.Tensor,
    padding_mask: torch.Tensor,
    padded_frame_count: torch.Tensor,
    source_frame_count: torch.Tensor,
    target_start_frame: torch.Tensor,
    batch_size: int,
    config: MapperV2Config,
) -> None:
    if full_mel.ndim != 3 or int(full_mel.shape[-1]) != config.mel_dim:
        raise ValueError(f"full_mel must have shape [B,F,{config.mel_dim}], got {tuple(full_mel.shape)}")
    if full_dense_timing_v2.ndim != 3 or int(full_dense_timing_v2.shape[-1]) != config.timing_dim:
        raise ValueError(
            f"full_dense_timing_v2 must have shape [B,F,{config.timing_dim}], "
            f"got {tuple(full_dense_timing_v2.shape)}"
        )
    if tuple(full_mel.shape[:2]) != tuple(full_dense_timing_v2.shape[:2]):
        raise ValueError("full_mel and full_dense_timing_v2 must share batch/frame dimensions")
    if int(full_mel.shape[0]) != int(batch_size):
        raise ValueError("full-song tensors must share decoder batch size")
    if tuple(padding_mask.shape) != tuple(full_mel.shape[:2]):
        raise ValueError("padding_mask must have shape [B,F]")
    if padding_mask.dtype != torch.bool:
        raise ValueError("padding_mask must be bool")
    if tuple(padded_frame_count.shape) != (batch_size,):
        raise ValueError("frame_count must have shape [B]")
    if tuple(source_frame_count.shape) != (batch_size,):
        raise ValueError("source_frame_count must have shape [B]")
    if tuple(target_start_frame.shape) != (batch_size,):
        raise ValueError("target_start_frame must have shape [B]")
    if bool(torch.any(padded_frame_count <= 0)):
        raise ValueError("frame_count must be positive")
    if bool(torch.any(source_frame_count <= 0)):
        raise ValueError("source_frame_count must be positive")
    if bool(torch.any(source_frame_count > padded_frame_count)):
        raise ValueError("source_frame_count cannot exceed frame_count")
    if bool(torch.any(padded_frame_count > full_mel.shape[1])):
        raise ValueError("frame_count cannot exceed full-song tensor length")
    if bool(torch.any(target_start_frame < 0)):
        raise ValueError("target_start_frame must be non-negative")
    if bool(torch.any(target_start_frame >= source_frame_count)):
        raise ValueError("target_start_frame must be less than source_frame_count")
    frame_index = torch.arange(full_mel.shape[1], device=full_mel.device).unsqueeze(0)
    beyond_source_frame_count = frame_index >= source_frame_count.to(device=full_mel.device).unsqueeze(1)
    if bool((beyond_source_frame_count & ~padding_mask).any()):
        raise ValueError("padding_mask must cover source_frame_count tail")


def _validate_v2_config(config: MapperV2Config) -> None:
    if not isinstance(config.use_global_context, bool):
        raise ValueError("use_global_context must be bool")
    if not config.use_global_context:
        return
    for name in ("global_stride", "global_layers", "global_conv_blocks", "global_conv_kernel_size"):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
    if config.global_ffn_dim is not None and (
        not isinstance(config.global_ffn_dim, int) or isinstance(config.global_ffn_dim, bool)
    ):
        raise ValueError("global_ffn_dim must be an integer when set")
    if not isinstance(config.global_gate_init, (int, float)) or isinstance(config.global_gate_init, bool):
        raise ValueError("global_gate_init must be numeric")
    if not math.isfinite(float(config.global_gate_init)):
        raise ValueError("global_gate_init must be finite")
    if config.global_stride < MIN_GLOBAL_STRIDE:
        raise ValueError(f"global_stride must be at least {MIN_GLOBAL_STRIDE}")
    if config.global_layers <= 0:
        raise ValueError("global_layers must be positive")
    if config.global_ffn_dim is not None and config.global_ffn_dim <= 0:
        raise ValueError("global_ffn_dim must be positive when set")
    if config.global_conv_blocks < 0:
        raise ValueError("global_conv_blocks must be non-negative")
    if config.global_conv_kernel_size <= 0 or config.global_conv_kernel_size % 2 == 0:
        raise ValueError("global_conv_kernel_size must be a positive odd integer")
