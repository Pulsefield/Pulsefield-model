from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from pulsefield_model.models.control.context import TARGET_OFFSET_IN_CONTEXT, TARGET_WINDOW_LENGTH_FRAMES

from .adapters import LNCloseAdapter, StatePriorAdapter
from .batch import MapperBatch, MapperFragmentState, MapperTokenContract
from .grammar import build_grammar_mask
from .tokenizer import MAPPER_DENSITY_FRAME_MS, MAPPER_DENSITY_FRAMES, MAPPER_WRITE_MS
from .vocab import MapperTupleVocab


@dataclass(frozen=True)
class MapperTupleConfig:
    vocab_size: int | None = None
    mel_dim: int = 160
    timing_dim: int = 4
    difficulty_dim: int = 1
    control_dim: int = 384
    d_model: int = 384
    heads: int = 8
    layers: int = 4
    ffn_dim: int = 1536
    dropout: float = 0.1
    max_seq_len: int = 512
    density_frames: int = MAPPER_DENSITY_FRAMES
    state_hidden_dim: int | None = None
    state_prior_hidden_dim: int = 64
    ln_close_hidden_dim: int = 128
    lane_embedding_dim: int = 8
    age_embedding_dim: int = 8
    num_age_buckets: int = 32
    age_cap_ms: int = 4000
    state_prior_adapter_scale: float | None = None
    state_prior_scale_init: float = 0.03
    state_prior_max_bias: float = 1.5
    close_scale: float = 0.05
    skip_scale: float = 0.0


@dataclass(frozen=True)
class MapperTupleForwardOutput:
    decoder_input_tokens: torch.Tensor
    loss_target_tokens: torch.Tensor
    state_current_ms: torch.Tensor
    state_open_mask: torch.Tensor
    state_open_start_ms: torch.Tensor
    state_open_age_ms: torch.Tensor
    base_logits: torch.Tensor
    logits_final: torch.Tensor
    decoder_hidden: torch.Tensor
    state_prior_bias: torch.Tensor
    state_prior_lane_action_bias: torch.Tensor
    ln_close_logits: torch.Tensor
    ln_close_event_bias: torch.Tensor
    ln_close_time_shift_bias: torch.Tensor
    grammar_mask: torch.Tensor
    control_memory_8s: torch.Tensor
    density_teacher_8s: torch.Tensor

    @property
    def close_logits(self) -> torch.Tensor:
        return self.ln_close_logits

    @property
    def ln_close_bias(self) -> torch.Tensor:
        return self.ln_close_event_bias

    @property
    def time_shift_bias(self) -> torch.Tensor:
        return self.ln_close_time_shift_bias


MapperTupleModelOutput = MapperTupleForwardOutput


class TupleMapperBase(nn.Module):
    def __init__(
        self,
        config: MapperTupleConfig = MapperTupleConfig(),
        *,
        vocab: MapperTupleVocab | None = None,
        control_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        _validate_config(config)
        self.config = config
        self.vocab = MapperTupleVocab() if vocab is None else vocab
        if config.vocab_size is not None and int(config.vocab_size) != self.vocab.size:
            raise ValueError(f"config vocab_size {config.vocab_size} does not match mapper vocab size {self.vocab.size}")
        self.control_encoder = control_encoder
        if self.control_encoder is not None:
            self.control_encoder.eval()
            for parameter in self.control_encoder.parameters():
                parameter.requires_grad_(False)

        self.token_embedding = nn.Embedding(self.vocab.size, config.d_model, padding_idx=self.vocab.pad_id)
        self.position = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_model))
        self.difficulty_projection = nn.Linear(config.difficulty_dim, config.d_model)
        self.time_projection = nn.Linear(2, config.d_model)
        if config.control_dim == config.d_model:
            self.control_projection: nn.Module = nn.Identity()
        else:
            self.control_projection = nn.Linear(config.control_dim, config.d_model)
        self.decoder_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
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
        self.output_norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, self.vocab.size)
        state_prior_hidden_dim = (
            config.state_prior_hidden_dim if config.state_hidden_dim is None else int(config.state_hidden_dim)
        )
        state_prior_scale_init = (
            config.state_prior_scale_init
            if config.state_prior_adapter_scale is None
            else float(config.state_prior_adapter_scale)
        )
        self.state_prior_adapter = StatePriorAdapter(
            vocab=self.vocab,
            hidden_dim=state_prior_hidden_dim,
            lane_embedding_dim=config.lane_embedding_dim,
            age_embedding_dim=config.age_embedding_dim,
            num_age_buckets=config.num_age_buckets,
            age_cap_ms=config.age_cap_ms,
            adapter_scale_init=state_prior_scale_init,
            max_bias=config.state_prior_max_bias,
        )
        self.ln_close_adapter = LNCloseAdapter(
            vocab=self.vocab,
            d_model=config.d_model,
            hidden_dim=config.ln_close_hidden_dim,
            lane_embedding_dim=config.lane_embedding_dim,
            age_embedding_dim=config.age_embedding_dim,
            num_age_buckets=config.num_age_buckets,
            age_cap_ms=config.age_cap_ms,
            close_scale=config.close_scale,
            skip_scale=config.skip_scale,
        )
        self._reset_parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.control_encoder is not None:
            self.control_encoder.eval()
        return self

    def forward(
        self,
        batch: Mapping[str, torch.Tensor] | None = None,
        *,
        control_memory_8s: torch.Tensor | None = None,
        density_teacher_8s: torch.Tensor | None = None,
        **kwargs: torch.Tensor | None,
    ) -> MapperTupleForwardOutput:
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
        if density_teacher_8s is None:
            maybe_density_teacher = batch.get("density_teacher_8s")
            if isinstance(maybe_density_teacher, torch.Tensor):
                density_teacher_8s = maybe_density_teacher
        if (control_memory_8s is None) != (density_teacher_8s is None):
            raise ValueError("control_memory_8s and density_teacher_8s must be supplied together")
        if isinstance(batch.get("control_memory_padding_mask_8s"), torch.Tensor):
            raise ValueError("control_memory_padding_mask_8s is not supported in Phase B; supply full 8s control memory")

        mapper_batch = MapperBatch.from_mapping(
            batch,
            contract=MapperTokenContract(name="tuple", vocab=self.vocab),
        )
        decoder_input = mapper_batch.decoder_input_tokens
        loss_target_tokens = mapper_batch.target_fragment_tokens
        target_fragment_mask = mapper_batch.target_fragment_mask
        input_padding_mask = mapper_batch.input_padding_mask
        device = decoder_input.device
        fragment_states = mapper_batch.fragment_states
        current_ms = fragment_states.current_ms
        open_mask = fragment_states.open_mask
        open_start_ms = fragment_states.open_start_ms
        open_age_ms = fragment_states.open_age_ms
        write_start_ms = mapper_batch.write_start_ms
        write_end_ms = mapper_batch.write_end_ms
        is_full_chart_start = mapper_batch.is_full_chart_start
        is_full_chart_end = mapper_batch.is_full_chart_end
        ln_carry_in = mapper_batch.ln_carry_in.as_mapping()
        ln_carry_out = mapper_batch.ln_carry_out.as_mapping()
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
        sanitized_states = fragment_states.sanitized(write_end_ms, valid_input_mask)
        current_ms = sanitized_states.current_ms
        open_mask = sanitized_states.open_mask
        open_start_ms = sanitized_states.open_start_ms
        open_age_ms = sanitized_states.open_age_ms

        if control_memory_8s is None:
            control_memory_8s, density_teacher_8s = self._control_teacher_8s(batch)
        control_memory_8s = control_memory_8s.detach().to(device=decoder_input.device, dtype=torch.float32)
        density_teacher_8s = density_teacher_8s.detach().to(device=decoder_input.device, dtype=torch.float32)
        if control_memory_8s.ndim != 3 or int(control_memory_8s.shape[1]) != MAPPER_DENSITY_FRAMES:
            raise ValueError(f"control_memory_8s must have shape [B,{MAPPER_DENSITY_FRAMES},D]")
        if int(control_memory_8s.shape[-1]) != self.config.control_dim:
            raise ValueError(
                f"control_memory_8s last dim must match config.control_dim={self.config.control_dim}, "
                f"got {control_memory_8s.shape[-1]}"
            )
        if tuple(density_teacher_8s.shape) != (decoder_input.shape[0], MAPPER_DENSITY_FRAMES, 1):
            raise ValueError(f"density_teacher_8s must have shape [B,{MAPPER_DENSITY_FRAMES},1]")
        control_memory = self.control_projection(control_memory_8s)

        decoder_hidden, base_logits = self._decode(
            tokens=decoder_input,
            current_ms=current_ms,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            difficulty=_difficulty_tensor(batch, device=decoder_input.device, dim=self.config.difficulty_dim),
            control_memory=control_memory,
            input_padding_mask=input_padding_mask,
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
        return MapperTupleForwardOutput(
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
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _decode(
        self,
        *,
        tokens: torch.Tensor,
        current_ms: torch.Tensor,
        write_start_ms: torch.Tensor,
        write_end_ms: torch.Tensor,
        difficulty: torch.Tensor,
        control_memory: torch.Tensor,
        input_padding_mask: torch.Tensor | None,
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
        if input_padding_mask is not None:
            hidden = hidden.masked_fill(input_padding_mask.unsqueeze(-1), 0.0)
        causal_mask = torch.triu(
            torch.ones((steps, steps), dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        for layer in self.decoder_layers:
            hidden = layer(
                tgt=hidden,
                memory=control_memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=input_padding_mask,
                memory_key_padding_mask=None,
            )
            if input_padding_mask is not None:
                hidden = hidden.masked_fill(input_padding_mask.unsqueeze(-1), 0.0)
        decoder_hidden = self.output_norm(hidden)
        base_logits = self.output_head(decoder_hidden)
        return decoder_hidden, base_logits

    def _control_teacher_8s(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        decoder_input_tokens = _require_tensor(batch, "decoder_input_tokens", ndim=2)
        batch_size = int(decoder_input_tokens.shape[0])
        device = decoder_input_tokens.device
        if self.control_encoder is None:
            return (
                torch.zeros((batch_size, MAPPER_DENSITY_FRAMES, self.config.control_dim), dtype=torch.float32, device=device),
                torch.zeros((batch_size, MAPPER_DENSITY_FRAMES, 1), dtype=torch.float32, device=device),
            )

        self.control_encoder.eval()
        return compute_control_teacher_8s(self.control_encoder, batch)

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.position, mean=0.0, std=0.01)


def concatenate_control_memory_8s(control_outputs: Sequence[Any]) -> torch.Tensor:
    if len(control_outputs) != 4:
        raise ValueError(f"expected four 2s control outputs, got {len(control_outputs)}")
    slices = []
    for index, output in enumerate(control_outputs):
        memory = getattr(output, "control_memory", None)
        if not isinstance(memory, torch.Tensor) or memory.ndim != 3:
            raise ValueError(f"control output {index} control_memory must have shape [B,T,D]")
        start = TARGET_OFFSET_IN_CONTEXT
        end = start + TARGET_WINDOW_LENGTH_FRAMES
        if int(memory.shape[1]) < end:
            raise ValueError(f"control output {index} memory is too short for target slice: {memory.shape[1]} < {end}")
        slices.append(memory[:, start:end])
    return torch.cat(slices, dim=1).contiguous()


def compute_control_teacher_8s(
    control_encoder: nn.Module,
    batch: Mapping[str, torch.Tensor],
    *,
    stack_slices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    from pulsefield_model.data.mapper_tuple_windows import (
        concatenate_density_teacher_8s,
        control_teacher_stacked_slices_batch,
        control_teacher_slice_batch,
    )

    control_encoder.eval()
    if stack_slices:
        control_slice_start_frames = _require_tensor(batch, "control_slice_start_frames", ndim=2)
        if int(control_slice_start_frames.shape[1]) != 4:
            raise ValueError("control_slice_start_frames must have four aligned 2s starts")
        batch_size = int(control_slice_start_frames.shape[0])
        with torch.no_grad():
            control_batch = control_teacher_stacked_slices_batch(dict(batch))
            output = control_encoder(
                context_mel=control_batch["context_mel"],
                context_dense_timing_v2=control_batch["context_dense_timing_v2"],
                normalized_difficulty=control_batch["normalized_difficulty"].reshape(batch_size * 4),
                context_padding_mask=control_batch["context_padding_mask"],
                full_mel=control_batch.get("full_mel"),
                full_dense_timing_v2=control_batch.get("full_dense_timing_v2"),
                padding_mask=control_batch.get("padding_mask"),
                frame_count=control_batch.get("frame_count"),
                target_start_frame=control_batch.get("target_start_frame"),
            )
        return _stacked_control_teacher_output_8s(output, batch_size=batch_size)

    decoder_input_tokens = _require_tensor(batch, "decoder_input_tokens", ndim=2)
    batch_size = int(decoder_input_tokens.shape[0])
    outputs = []
    with torch.no_grad():
        for slice_index in range(4):
            control_batch = control_teacher_slice_batch(dict(batch), slice_index)
            outputs.append(
                control_encoder(
                    context_mel=control_batch["context_mel"],
                    context_dense_timing_v2=control_batch["context_dense_timing_v2"],
                    normalized_difficulty=control_batch["normalized_difficulty"].reshape(batch_size),
                    context_padding_mask=control_batch["context_padding_mask"],
                    full_mel=control_batch.get("full_mel"),
                    full_dense_timing_v2=control_batch.get("full_dense_timing_v2"),
                    padding_mask=control_batch.get("padding_mask"),
                    frame_count=control_batch.get("frame_count"),
                    target_start_frame=control_batch.get("target_start_frame"),
                )
            )
    return concatenate_control_memory_8s(outputs), concatenate_density_teacher_8s(outputs)


def _stacked_control_teacher_output_8s(output: Any, *, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    memory = getattr(output, "control_memory", None)
    if not isinstance(memory, torch.Tensor) or memory.ndim != 3:
        raise ValueError("stacked control output control_memory must have shape [B*4,T,D]")
    expected_batch = int(batch_size) * 4
    if int(memory.shape[0]) != expected_batch:
        raise ValueError(f"stacked control output batch must be {expected_batch}, got {memory.shape[0]}")
    start = TARGET_OFFSET_IN_CONTEXT
    end = start + TARGET_WINDOW_LENGTH_FRAMES
    if int(memory.shape[1]) < end:
        raise ValueError(f"stacked control output memory is too short for target slice: {memory.shape[1]} < {end}")
    control_memory_8s = memory[:, start:end].reshape(batch_size, 4, TARGET_WINDOW_LENGTH_FRAMES, memory.shape[-1])
    control_memory_8s = control_memory_8s.reshape(batch_size, 4 * TARGET_WINDOW_LENGTH_FRAMES, memory.shape[-1]).contiguous()

    value_pred = getattr(output, "value_pred", None)
    if not isinstance(value_pred, torch.Tensor) or value_pred.ndim != 3 or int(value_pred.shape[1]) != TARGET_WINDOW_LENGTH_FRAMES:
        raise ValueError("stacked control output value_pred must have shape [B*4,100,C]")
    if int(value_pred.shape[0]) != expected_batch:
        raise ValueError(f"stacked control output value_pred batch must be {expected_batch}, got {value_pred.shape[0]}")
    from pulsefield_model.features.control_v3_targets import VALUE_FEATURE_NAMES

    density_index = VALUE_FEATURE_NAMES.index("density_level")
    if int(value_pred.shape[2]) == 1:
        density = value_pred
    elif int(value_pred.shape[2]) == len(VALUE_FEATURE_NAMES):
        density = value_pred[:, :, density_index : density_index + 1]
    else:
        raise ValueError(
            f"stacked control output value_pred channel count must be 1 or {len(VALUE_FEATURE_NAMES)}, "
            f"got {value_pred.shape[2]}",
        )
    density_teacher_8s = density.reshape(batch_size, 4, TARGET_WINDOW_LENGTH_FRAMES, 1)
    density_teacher_8s = density_teacher_8s.reshape(batch_size, 4 * TARGET_WINDOW_LENGTH_FRAMES, 1).contiguous()
    return control_memory_8s, density_teacher_8s


def _time_features(
    *,
    current_ms: torch.Tensor,
    write_start_ms: torch.Tensor,
    write_end_ms: torch.Tensor,
) -> torch.Tensor:
    write_start = write_start_ms.reshape(-1, 1).to(device=current_ms.device, dtype=torch.float32)
    write_end = write_end_ms.reshape(-1, 1).to(device=current_ms.device, dtype=torch.float32)
    current = current_ms.to(dtype=torch.float32)
    span = (write_end - write_start).clamp_min(1.0)
    rel = ((current - write_start) / span).clamp(0.0, 1.0)
    remaining = ((write_end - current) / span).clamp(0.0, 1.0)
    return torch.stack((rel, remaining), dim=-1)


def _validate_fragment_contract(
    *,
    decoder_input_tokens: torch.Tensor,
    target_fragment_tokens: torch.Tensor,
    target_fragment_mask: torch.Tensor,
    current_ms: torch.Tensor,
    open_mask: torch.Tensor,
    open_start_ms: torch.Tensor,
    open_age_ms: torch.Tensor,
    write_start_ms: torch.Tensor,
    write_end_ms: torch.Tensor,
    is_full_chart_start: torch.Tensor,
    is_full_chart_end: torch.Tensor,
    ln_carry_in: Mapping[str, torch.Tensor],
    ln_carry_out: Mapping[str, torch.Tensor],
    bos_id: int,
    eos_id: int,
) -> None:
    batch_size = int(current_ms.shape[0])
    if tuple(write_start_ms.shape) != (current_ms.shape[0],) or tuple(write_end_ms.shape) != (current_ms.shape[0],):
        raise ValueError("write_start_ms and write_end_ms must have one value per batch item")
    if tuple(is_full_chart_start.shape) != (batch_size,) or tuple(is_full_chart_end.shape) != (batch_size,):
        raise ValueError("is_full_chart_start and is_full_chart_end must have one value per batch item")
    if tuple(decoder_input_tokens.shape) != tuple(current_ms.shape):
        raise ValueError("decoder_input_tokens must align with target_fragment_states.current_ms")
    if tuple(target_fragment_tokens.shape) != tuple(current_ms.shape):
        raise ValueError("target_fragment_tokens must align with target_fragment_states.current_ms")
    if tuple(target_fragment_mask.shape) != tuple(current_ms.shape):
        raise ValueError("target_fragment_mask must align with target_fragment_states.current_ms")
    span = write_end_ms - write_start_ms
    if bool(torch.any(span != MAPPER_WRITE_MS)):
        raise ValueError(f"mapper tuple requires an exact {MAPPER_WRITE_MS}ms write window")
    if bool(torch.any(write_start_ms % MAPPER_DENSITY_FRAME_MS != 0)) or bool(
        torch.any(write_end_ms % MAPPER_DENSITY_FRAME_MS != 0)
    ):
        raise ValueError("mapper write_start_ms and write_end_ms must align to the 20ms density grid")

    start = write_start_ms.reshape(-1, 1)
    end = write_end_ms.reshape(-1, 1)
    valid = target_fragment_mask.to(dtype=torch.bool, device=current_ms.device)
    if bool((((current_ms < start) | (current_ms > end)) & valid).any()):
        raise ValueError("target_fragment_states.current_ms must be within [write_start_ms, write_end_ms]")
    if bool(((current_ms % 10 != 0) & valid).any()):
        raise ValueError("target_fragment_states.current_ms must align to the 10ms token grid")

    at_write_end = (current_ms == end) & valid
    if bool((at_write_end & (target_fragment_tokens != int(eos_id))).any()):
        raise ValueError("target_fragment_states.current_ms == write_end_ms is valid only for EOS prediction")
    if bool((at_write_end & open_mask.any(dim=-1)).any()):
        raise ValueError("target_fragment_states.current_ms == write_end_ms requires all lanes closed")
    if bool((at_write_end & ~is_full_chart_end.reshape(-1, 1)).any()):
        raise ValueError("EOS prediction at write_end_ms requires is_full_chart_end")

    valid_target_bos = valid & (target_fragment_tokens == int(bos_id))
    if bool((valid_target_bos & ~is_full_chart_start.reshape(-1, 1)).any()):
        raise ValueError("BOS target is valid only when is_full_chart_start is true")
    valid_target_eos = valid & (target_fragment_tokens == int(eos_id))
    if bool((valid_target_eos & ~is_full_chart_end.reshape(-1, 1)).any()):
        raise ValueError("EOS target is valid only when is_full_chart_end is true")
    first_input_is_bos = decoder_input_tokens[:, 0] == int(bos_id)
    if bool((first_input_is_bos & ~is_full_chart_start).any()):
        raise ValueError("decoder_input_tokens[:, 0] may be BOS only at full-chart start")

    carry_in_current = ln_carry_in["current_ms"]
    carry_out_current = ln_carry_out["current_ms"]
    if tuple(carry_in_current.shape) != (batch_size,) or tuple(carry_out_current.shape) != (batch_size,):
        raise ValueError("ln_carry_in.current_ms and ln_carry_out.current_ms must have shape [B]")
    if bool((carry_in_current != write_start_ms).any()):
        raise ValueError("ln_carry_in.current_ms must equal write_start_ms")
    if bool((carry_out_current != write_end_ms).any()):
        raise ValueError("ln_carry_out.current_ms must equal write_end_ms")
    _validate_open_start_age_consistency(
        current_ms=current_ms,
        open_mask=open_mask,
        open_start_ms=open_start_ms,
        open_age_ms=open_age_ms,
        valid=valid,
        name="target_fragment_states",
    )
    _validate_carry_state_consistency(ln_carry_in, name="ln_carry_in")
    _validate_carry_state_consistency(ln_carry_out, name="ln_carry_out")

    first_valid = target_fragment_mask[:, 0].to(dtype=torch.bool)
    if bool(first_valid.any()):
        rows = first_valid
        if bool((current_ms[rows, 0] != ln_carry_in["current_ms"][rows]).any()):
            raise ValueError("target_fragment_states.current_ms[:, 0] must equal ln_carry_in.current_ms")
        for key, tensor in (
            ("open_mask", open_mask),
            ("open_start_ms", open_start_ms),
            ("open_age_ms", open_age_ms),
        ):
            expected = ln_carry_in[key][rows]
            actual = tensor[rows, 0]
            if bool((actual != expected).any()):
                raise ValueError(f"target_fragment_states.{key}[:, 0] must equal ln_carry_in.{key}")


def _sanitize_padded_fragment_states(
    *,
    current_ms: torch.Tensor,
    open_mask: torch.Tensor,
    open_start_ms: torch.Tensor,
    open_age_ms: torch.Tensor,
    write_end_ms: torch.Tensor,
    valid_input_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sanitized = MapperFragmentState(
        current_ms=current_ms,
        open_mask=open_mask,
        open_start_ms=open_start_ms,
        open_age_ms=open_age_ms,
    ).sanitized(write_end_ms, valid_input_mask)
    return sanitized.current_ms, sanitized.open_mask, sanitized.open_start_ms, sanitized.open_age_ms


def _reject_old_mapper_contract(batch: Mapping[str, Any]) -> None:
    old_keys = {
        "target_tokens",
        "target_token_mask",
        "teacher_current_ms",
        "teacher_open_mask",
        "teacher_open_age_ms",
    }
    present = sorted(key for key in old_keys if key in batch)
    if present:
        raise ValueError(
            "old target_tokens/teacher_* mapper contract is not supported; "
            "supply decoder_input_tokens, target_fragment_tokens, target_fragment_mask, and target_fragment_states "
            f"instead of {present}"
        )


def _require_state_mapping(batch: Mapping[str, Any], key: str) -> Mapping[str, torch.Tensor]:
    value = batch.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"batch[{key!r}] must be a mapping of state tensors")
    return value


def _require_state_tensor(state: Mapping[str, torch.Tensor], key: str, *, ndim: int) -> torch.Tensor:
    value = state.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"state[{key!r}] must be a torch.Tensor")
    if value.ndim != ndim:
        raise ValueError(f"state[{key!r}] must be rank {ndim}, got shape {tuple(value.shape)}")
    return value


def _load_carry_state(batch: Mapping[str, Any], key: str, *, device: torch.device) -> dict[str, torch.Tensor]:
    raw = _require_state_mapping(batch, key)
    current_ms = _require_state_tensor(raw, "current_ms", ndim=1).to(device=device, dtype=torch.long)
    open_mask = _require_state_tensor(raw, "open_mask", ndim=2).to(device=device, dtype=torch.bool)
    open_start_ms = _require_state_tensor(raw, "open_start_ms", ndim=2).to(device=device, dtype=torch.long)
    open_age_ms = _require_state_tensor(raw, "open_age_ms", ndim=2).to(device=device, dtype=torch.long)
    if tuple(open_mask.shape) != tuple(open_start_ms.shape) or tuple(open_mask.shape) != tuple(open_age_ms.shape):
        raise ValueError(f"{key}.open_mask, open_start_ms, and open_age_ms must have matching shapes")
    if int(open_mask.shape[-1]) != 4:
        raise ValueError(f"{key}.open_mask must have shape [B,4]")
    if int(open_mask.shape[0]) != int(current_ms.shape[0]):
        raise ValueError(f"{key} tensors must share batch size")
    return {
        "current_ms": current_ms,
        "open_mask": open_mask,
        "open_start_ms": open_start_ms,
        "open_age_ms": open_age_ms,
    }


def _validate_open_start_age_consistency(
    *,
    current_ms: torch.Tensor,
    open_mask: torch.Tensor,
    open_start_ms: torch.Tensor,
    open_age_ms: torch.Tensor,
    valid: torch.Tensor,
    name: str,
) -> None:
    if current_ms.ndim == 1:
        current = current_ms.reshape(-1, 1).expand_as(open_start_ms)
        valid_expanded = valid.reshape(-1, 1).expand_as(open_start_ms)
    else:
        current = current_ms.unsqueeze(-1).expand_as(open_start_ms)
        valid_expanded = valid.unsqueeze(-1).expand_as(open_start_ms)
    open_bool = open_mask.to(dtype=torch.bool)
    valid_bool = valid_expanded.to(dtype=torch.bool)
    closed = (~open_bool) & valid_bool
    if bool((closed & (open_start_ms >= 0)).any()):
        raise ValueError(f"{name}.open_start_ms must be negative for closed lanes")
    if bool((closed & (open_age_ms != 0)).any()):
        raise ValueError(f"{name}.open_age_ms must be zero for closed lanes")
    open_valid = open_bool & valid_bool
    if not bool(open_valid.any()):
        return
    if bool((open_valid & (open_start_ms < 0)).any()):
        raise ValueError(f"{name}.open_start_ms must be set for open lanes")
    expected_age = current - open_start_ms
    if bool((open_valid & (expected_age != open_age_ms)).any()):
        raise ValueError(f"{name}.open_age_ms must equal current_ms - open_start_ms for open lanes")
    if bool((open_valid & (open_age_ms < 0)).any()):
        raise ValueError(f"{name}.open_age_ms must be non-negative for open lanes")


def _validate_carry_state_consistency(carry: Mapping[str, torch.Tensor], *, name: str) -> None:
    current_ms = carry["current_ms"]
    open_mask = carry["open_mask"]
    open_start_ms = carry["open_start_ms"]
    open_age_ms = carry["open_age_ms"]
    valid = torch.ones_like(current_ms, dtype=torch.bool)
    _validate_open_start_age_consistency(
        current_ms=current_ms,
        open_mask=open_mask,
        open_start_ms=open_start_ms,
        open_age_ms=open_age_ms,
        valid=valid,
        name=name,
    )
    open_bool = open_mask.to(dtype=torch.bool)
    if bool((open_bool & (open_start_ms >= current_ms.reshape(-1, 1))).any()):
        raise ValueError(f"{name}.open_start_ms must be before current_ms for open carry lanes")


def _difficulty_tensor(batch: Mapping[str, torch.Tensor], *, device: torch.device, dim: int) -> torch.Tensor:
    value = batch.get("normalized_difficulty", batch.get("difficulty"))
    if not isinstance(value, torch.Tensor):
        raise ValueError("batch must contain difficulty or normalized_difficulty")
    value = value.to(device=device, dtype=torch.float32)
    if value.ndim == 1:
        value = value.reshape(-1, 1)
    if value.ndim != 2 or int(value.shape[1]) != dim:
        raise ValueError(f"difficulty must have shape [B,{dim}], got {tuple(value.shape)}")
    return value


def _require_tensor(batch: Mapping[str, torch.Tensor], key: str, *, ndim: int) -> torch.Tensor:
    value = batch.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"batch[{key!r}] must be a torch.Tensor")
    if value.ndim != ndim:
        raise ValueError(f"batch[{key!r}] must be rank {ndim}, got shape {tuple(value.shape)}")
    return value


def _optional_tensor(batch: Mapping[str, torch.Tensor], key: str) -> torch.Tensor | None:
    value = batch.get(key)
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"batch[{key!r}] must be a torch.Tensor")
    return value


def _validate_config(config: MapperTupleConfig) -> None:
    for name in (
        "mel_dim",
        "timing_dim",
        "difficulty_dim",
        "control_dim",
        "d_model",
        "heads",
        "layers",
        "ffn_dim",
        "max_seq_len",
        "density_frames",
        "state_prior_hidden_dim",
        "ln_close_hidden_dim",
        "lane_embedding_dim",
        "age_embedding_dim",
        "num_age_buckets",
        "age_cap_ms",
    ):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if config.vocab_size is not None and (
        not isinstance(config.vocab_size, int) or isinstance(config.vocab_size, bool) or config.vocab_size <= 0
    ):
        raise ValueError("vocab_size must be a positive integer when set")
    if config.state_hidden_dim is not None and (
        not isinstance(config.state_hidden_dim, int)
        or isinstance(config.state_hidden_dim, bool)
        or config.state_hidden_dim <= 0
    ):
        raise ValueError("state_hidden_dim must be a positive integer when set")
    if config.density_frames != MAPPER_DENSITY_FRAMES:
        raise ValueError(f"density_frames must be {MAPPER_DENSITY_FRAMES} for mapper tuple")
    if config.d_model % config.heads != 0:
        raise ValueError("d_model must be divisible by heads")
    for name in ("dropout", "state_prior_scale_init", "state_prior_max_bias", "close_scale", "skip_scale"):
        value = getattr(config, name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name} must be numeric")
    if config.state_prior_adapter_scale is not None and not isinstance(config.state_prior_adapter_scale, (int, float)):
        raise ValueError("state_prior_adapter_scale must be numeric when set")
    if not 0.0 <= float(config.dropout) < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if config.state_prior_max_bias <= 0.0:
        raise ValueError("state_prior_max_bias must be positive")
    if config.close_scale < 0.0 or config.skip_scale < 0.0:
        raise ValueError("close_scale and skip_scale must be non-negative")
