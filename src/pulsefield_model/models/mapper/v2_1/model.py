from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn

from pulsefield_model.models.control.demo_global import ControlDemoGlobalEncoderConfig
from pulsefield_model.models.mapper.shared.model import (
    _difficulty_tensor,
    _load_carry_state,
    _reject_old_mapper_contract,
    _require_state_mapping,
    _require_state_tensor,
    _require_tensor,
    _sanitize_padded_fragment_states,
    _validate_carry_state_consistency,
    _validate_open_start_age_consistency,
)
from pulsefield_model.models.mapper.shared.vocab import MapperTupleVocab
from pulsefield_model.models.mapper.v2.model import MapperV2Config, MapperV2ForwardOutput, MapperV2Model

from .adapters import LNCloseAdapter, StatePriorAdapter
from .grammar import build_grammar_mask
from .tokenizer import MAPPER_DENSITY_FRAME_MS, MAPPER_DENSITY_FRAMES, MAPPER_WRITE_MS
from .vocab import MapperV21Vocab


@dataclass(frozen=True)
class MapperV21Config(MapperV2Config):
    """Sparse-token Mapper V2.1 config.

    Sparse same-time chords can use more decoder tokens than V1/V2 tuple events,
    so the default sequence cap is raised while keeping the V2 global-context
    defaults otherwise intact.
    """

    max_seq_len: int = 1024


@dataclass(frozen=True)
class MapperV21ForwardOutput(MapperV2ForwardOutput):
    state_emitted_lane_mask: torch.Tensor | None = None
    state_last_lane_index: torch.Tensor | None = None


MapperV21ModelOutput = MapperV21ForwardOutput


class MapperV21Model(MapperV2Model):
    """Mapper V2 global-context decoder with the V2.1 sparse token contract."""

    def __init__(
        self,
        config: MapperV21Config = MapperV21Config(),
        *,
        vocab: MapperV21Vocab | None = None,
        control_encoder: nn.Module | None = None,
    ) -> None:
        resolved_vocab = MapperV21Vocab() if vocab is None else vocab
        if config.vocab_size is not None and int(config.vocab_size) != resolved_vocab.size:
            raise ValueError(f"config vocab_size {config.vocab_size} does not match mapper v2.1 vocab size {resolved_vocab.size}")

        # MapperV2Model initializes the shared decoder/global-context stack, but
        # its superclass constructs V1 tuple-event adapters. Use a temporary V1
        # vocab for base construction, then replace every vocab-sized/sparse
        # component below before this model can be used.
        super().__init__(
            replace(config, vocab_size=None),
            vocab=MapperTupleVocab(),
            control_encoder=control_encoder,
        )
        self.config: MapperV21Config = config
        self.vocab = resolved_vocab
        self.token_embedding = nn.Embedding(self.vocab.size, config.d_model, padding_idx=self.vocab.pad_id)
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

    def forward(
        self,
        batch: Mapping[str, torch.Tensor] | None = None,
        *,
        control_memory_8s: torch.Tensor | None = None,
        density_teacher_8s: torch.Tensor | None = None,
        **kwargs: torch.Tensor | None,
    ) -> MapperV21ForwardOutput:
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
        emitted_lane_mask = _require_state_tensor(states, "emitted_lane_mask", ndim=3).to(device=device, dtype=torch.bool)
        last_lane_index = _require_state_tensor(states, "last_lane_index", ndim=2).to(device=device, dtype=torch.long)
        write_start_ms = _require_tensor(batch, "write_start_ms", ndim=1).to(device=device, dtype=torch.long)
        write_end_ms = _require_tensor(batch, "write_end_ms", ndim=1).to(device=device, dtype=torch.long)
        raw_chart_end_ms = batch.get("chart_end_ms")
        if raw_chart_end_ms is None:
            chart_end_ms = write_end_ms
        elif not isinstance(raw_chart_end_ms, torch.Tensor):
            raise ValueError("chart_end_ms must be a torch.Tensor")
        else:
            chart_end_ms = raw_chart_end_ms.to(device=device, dtype=torch.long).reshape(-1)
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
        if tuple(emitted_lane_mask.shape) != tuple(open_mask.shape):
            raise ValueError("target_fragment_states.emitted_lane_mask must align with target_fragment_states.open_mask")
        if tuple(last_lane_index.shape) != tuple(decoder_input.shape):
            raise ValueError("target_fragment_states.last_lane_index must align with decoder_input_tokens")
        valid_input_mask = target_fragment_mask.to(device=device, dtype=torch.bool)
        target_end_ms = torch.where(is_full_chart_end, chart_end_ms, write_end_ms)
        _validate_v21_fragment_contract(
            decoder_input_tokens=decoder_input,
            target_fragment_tokens=loss_target_tokens,
            target_fragment_mask=valid_input_mask,
            current_ms=current_ms,
            open_mask=open_mask,
            open_start_ms=open_start_ms,
            open_age_ms=open_age_ms,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            chart_end_ms=chart_end_ms,
            target_end_ms=target_end_ms,
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
            write_end_ms=target_end_ms,
            valid_input_mask=valid_input_mask,
        )
        if not bool(valid_input_mask.all()):
            padded = ~valid_input_mask
            emitted_lane_mask = torch.where(padded.unsqueeze(-1), torch.zeros_like(emitted_lane_mask), emitted_lane_mask)
            last_lane_index = torch.where(padded, torch.full_like(last_lane_index, -1), last_lane_index)

        if control_memory_8s is None and projected_control_memory_8s is None:
            control_memory_8s, density_teacher_8s = self._control_teacher_8s(batch)
        assert density_teacher_8s is not None
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
            write_end_ms=target_end_ms,
            difficulty=_difficulty_tensor(batch, device=decoder_input.device, dim=self.config.difficulty_dim),
            control_memory=control_memory,
            input_padding_mask=input_padding_mask,
            global_memory=global_memory,
            global_memory_padding_mask=global_memory_padding_mask,
            global_position_features=global_position_features,
            global_attention_kv_cache=global_attention_kv_cache,
        )
        remaining_ms = (target_end_ms.reshape(-1, 1) - current_ms).clamp_min(0)
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
            emitted_lane_mask=emitted_lane_mask,
            last_lane_index=last_lane_index,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            chart_end_ms=chart_end_ms,
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
        return MapperV21ForwardOutput(
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
            state_emitted_lane_mask=emitted_lane_mask,
            state_last_lane_index=last_lane_index,
        )

    @property
    def incremental_decode_next_token(self) -> Any:
        raise AttributeError("MapperV21Model does not expose incremental decoding")


def _validate_v21_fragment_contract(
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
    chart_end_ms: torch.Tensor,
    target_end_ms: torch.Tensor,
    is_full_chart_start: torch.Tensor,
    is_full_chart_end: torch.Tensor,
    ln_carry_in: Mapping[str, torch.Tensor],
    ln_carry_out: Mapping[str, torch.Tensor],
    bos_id: int,
    eos_id: int,
) -> None:
    batch_size = int(current_ms.shape[0])
    if tuple(write_start_ms.shape) != (batch_size,) or tuple(write_end_ms.shape) != (batch_size,):
        raise ValueError("write_start_ms and write_end_ms must have one value per batch item")
    if tuple(chart_end_ms.shape) != (batch_size,) or tuple(target_end_ms.shape) != (batch_size,):
        raise ValueError("chart_end_ms and target_end_ms must have one value per batch item")
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
        raise ValueError(f"mapper v2.1 requires an exact {MAPPER_WRITE_MS}ms write window")
    if bool(torch.any(write_start_ms % MAPPER_DENSITY_FRAME_MS != 0)) or bool(
        torch.any(write_end_ms % MAPPER_DENSITY_FRAME_MS != 0)
    ):
        raise ValueError("mapper write_start_ms and write_end_ms must align to the 20ms density grid")
    if bool(torch.any(chart_end_ms % 10 != 0)):
        raise ValueError("chart_end_ms must align to the 10ms token grid")

    expected_target_end = torch.where(is_full_chart_end, chart_end_ms, write_end_ms)
    if bool((target_end_ms != expected_target_end).any()):
        raise ValueError("target_end_ms must equal chart_end_ms for terminal windows and write_end_ms otherwise")
    terminal_outside = is_full_chart_end & ((chart_end_ms < write_start_ms) | (chart_end_ms > write_end_ms))
    if bool(terminal_outside.any()):
        raise ValueError("terminal chart_end_ms must fall inside [write_start_ms, write_end_ms]")

    start = write_start_ms.reshape(-1, 1)
    target_end = target_end_ms.reshape(-1, 1)
    valid = target_fragment_mask.to(dtype=torch.bool, device=current_ms.device)
    if bool((((current_ms < start) | (current_ms > target_end)) & valid).any()):
        raise ValueError("target_fragment_states.current_ms must be within [write_start_ms, target_end_ms]")
    if bool(((current_ms % 10 != 0) & valid).any()):
        raise ValueError("target_fragment_states.current_ms must align to the 10ms token grid")

    at_target_end = (current_ms == target_end) & valid
    if bool((at_target_end & ~is_full_chart_end.reshape(-1, 1)).any()):
        raise ValueError("target_end_ms state is valid only in terminal windows")

    valid_target_bos = valid & (target_fragment_tokens == int(bos_id))
    if bool((valid_target_bos & ~is_full_chart_start.reshape(-1, 1)).any()):
        raise ValueError("BOS target is valid only when is_full_chart_start is true")
    valid_target_eos = valid & (target_fragment_tokens == int(eos_id))
    if bool((valid_target_eos & ~is_full_chart_end.reshape(-1, 1)).any()):
        raise ValueError("EOS target is valid only when is_full_chart_end is true")
    if bool((valid_target_eos & (current_ms != chart_end_ms.reshape(-1, 1))).any()):
        raise ValueError("EOS target requires current_ms == chart_end_ms")
    if bool((valid_target_eos & open_mask.any(dim=-1)).any()):
        raise ValueError("EOS target requires all lanes closed")

    first_input_is_bos = decoder_input_tokens[:, 0] == int(bos_id)
    if bool((first_input_is_bos & ~is_full_chart_start).any()):
        raise ValueError("decoder_input_tokens[:, 0] may be BOS only at full-chart start")

    carry_in_current = ln_carry_in["current_ms"]
    carry_out_current = ln_carry_out["current_ms"]
    if tuple(carry_in_current.shape) != (batch_size,) or tuple(carry_out_current.shape) != (batch_size,):
        raise ValueError("ln_carry_in.current_ms and ln_carry_out.current_ms must have shape [B]")
    if bool((carry_in_current != write_start_ms).any()):
        raise ValueError("ln_carry_in.current_ms must equal write_start_ms")
    if bool((carry_out_current != target_end_ms).any()):
        raise ValueError("ln_carry_out.current_ms must equal target_end_ms")

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


def _precomputed_global_attention_kv_cache(
    *,
    batch: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
    global_memory: torch.Tensor | None,
    config: ControlDemoGlobalEncoderConfig | MapperV2Config,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...] | None:
    raw = batch.get("global_attention_kv_cache")
    if raw is None:
        return None
    if global_memory is None:
        raise ValueError("global_attention_kv_cache requires global_memory")
    if not isinstance(raw, tuple) or len(raw) != int(config.layers):
        raise ValueError("global_attention_kv_cache must contain one key/value tuple per decoder layer")
    expected = (
        int(batch_size),
        int(config.heads),
        int(global_memory.shape[1]),
        int(config.d_model) // int(config.heads),
    )
    normalized: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer_index, item in enumerate(raw):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"global_attention_kv_cache[{layer_index}] must be a key/value tuple")
        key, value = item
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise ValueError(f"global_attention_kv_cache[{layer_index}] entries must be tensors")
        if tuple(key.shape) != expected or tuple(value.shape) != expected:
            raise ValueError(
                f"global_attention_kv_cache[{layer_index}] tensors must have shape {expected}, "
                f"got key={tuple(key.shape)} value={tuple(value.shape)}"
            )
        normalized.append((key.detach().to(device=device), value.detach().to(device=device)))
    return tuple(normalized)
