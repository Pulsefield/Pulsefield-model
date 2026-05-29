from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


AttentionKVCache = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class IncrementalSelfAttentionKVCache:
    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class IncrementalDecodeState:
    self_attention_kv_cache: tuple[IncrementalSelfAttentionKVCache, ...]

    @property
    def sequence_length(self) -> int:
        if not self.self_attention_kv_cache:
            return 0
        return int(self.self_attention_kv_cache[0].key.shape[2])


@dataclass(frozen=True)
class IncrementalDecoderStepOutput:
    hidden: torch.Tensor
    decode_state: IncrementalDecodeState


def create_empty_incremental_decode_state(
    *,
    batch_size: int,
    layers: int,
    heads: int,
    d_model: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> IncrementalDecodeState:
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if int(layers) <= 0:
        raise ValueError("layers must be positive")
    if int(heads) <= 0:
        raise ValueError("heads must be positive")
    if int(d_model) % int(heads) != 0:
        raise ValueError("d_model must be divisible by heads")
    resolved_device = torch.device(device)
    head_dim = int(d_model) // int(heads)
    return IncrementalDecodeState(
        self_attention_kv_cache=tuple(
            IncrementalSelfAttentionKVCache(
                key=torch.zeros(
                    (batch_size, int(heads), 0, head_dim),
                    dtype=dtype,
                    device=resolved_device,
                ),
                value=torch.zeros(
                    (batch_size, int(heads), 0, head_dim),
                    dtype=dtype,
                    device=resolved_device,
                ),
            )
            for _ in range(int(layers))
        ),
    )


def incremental_decoder_step(
    *,
    decoder_layers: Sequence[nn.TransformerDecoderLayer],
    hidden: torch.Tensor,
    control_memory: torch.Tensor,
    decode_state: IncrementalDecodeState,
    control_attention_kv_cache: tuple[AttentionKVCache, ...] | None = None,
    after_layer: Callable[[int, torch.Tensor], torch.Tensor] | None = None,
) -> IncrementalDecoderStepOutput:
    if hidden.ndim != 3 or int(hidden.shape[1]) != 1:
        raise ValueError(f"incremental decoder hidden must have shape [B,1,D], got {tuple(hidden.shape)}")
    if control_memory.ndim != 3:
        raise ValueError(f"control_memory must have shape [B,T,D], got {tuple(control_memory.shape)}")
    batch_size = int(hidden.shape[0])
    d_model = int(hidden.shape[-1])
    if tuple(control_memory.shape[:1]) != (batch_size,) or int(control_memory.shape[-1]) != d_model:
        raise ValueError("control_memory must match incremental hidden batch and width")
    if len(decoder_layers) == 0:
        raise ValueError("decoder_layers cannot be empty")
    first_self_attn = decoder_layers[0].self_attn
    heads = int(first_self_attn.num_heads)
    validate_incremental_decode_state(
        decode_state,
        batch_size=batch_size,
        layers=len(decoder_layers),
        heads=heads,
        head_dim=d_model // heads,
    )
    if control_attention_kv_cache is not None and len(control_attention_kv_cache) != len(decoder_layers):
        raise ValueError("control_attention_kv_cache must contain one entry per decoder layer")

    next_layer_caches: list[IncrementalSelfAttentionKVCache] = []
    step_hidden = hidden
    for layer_index, layer in enumerate(decoder_layers):
        step_hidden, layer_cache = _incremental_transformer_decoder_layer_step(
            layer,
            hidden=step_hidden,
            memory=control_memory,
            self_attention_kv=decode_state.self_attention_kv_cache[layer_index],
            control_attention_kv=None
            if control_attention_kv_cache is None
            else control_attention_kv_cache[layer_index],
        )
        next_layer_caches.append(layer_cache)
        if after_layer is not None:
            step_hidden = after_layer(layer_index, step_hidden)
    return IncrementalDecoderStepOutput(
        hidden=step_hidden,
        decode_state=IncrementalDecodeState(self_attention_kv_cache=tuple(next_layer_caches)),
    )


def transformer_decoder_control_attention_kv_cache(
    layer: nn.TransformerDecoderLayer,
    memory: torch.Tensor,
) -> AttentionKVCache:
    cross_attn = layer.multihead_attn
    if getattr(cross_attn, "batch_first", False) is not True:
        raise ValueError("control attention K/V cache requires batch_first decoder cross-attention")
    if cross_attn.in_proj_weight is None:
        raise ValueError("control attention K/V cache requires packed cross-attention projection weights")
    if memory.ndim != 3:
        raise ValueError(f"control memory must have shape [B,T,D], got {tuple(memory.shape)}")
    _, source_steps, d_model = memory.shape
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
    key = attention_projection_to_heads(
        F.linear(memory, key_weight, key_bias),
        heads=heads,
    )
    value = attention_projection_to_heads(
        F.linear(memory, value_weight, value_bias),
        heads=heads,
    )
    expected_steps = int(source_steps)
    if int(key.shape[2]) != expected_steps or int(value.shape[2]) != expected_steps:
        raise ValueError("control attention K/V cache source length mismatch")
    return key.detach().contiguous(), value.detach().contiguous()


def validate_incremental_decode_state(
    state: IncrementalDecodeState,
    *,
    batch_size: int,
    layers: int,
    heads: int,
    head_dim: int,
) -> int:
    if not isinstance(state, IncrementalDecodeState):
        raise ValueError("decode_state must be an IncrementalDecodeState")
    if len(state.self_attention_kv_cache) != int(layers):
        raise ValueError(
            f"decode_state must contain {layers} layer caches, got {len(state.self_attention_kv_cache)}"
        )
    sequence_length: int | None = None
    for layer_index, layer_cache in enumerate(state.self_attention_kv_cache):
        if not isinstance(layer_cache, IncrementalSelfAttentionKVCache):
            raise ValueError(f"decode_state layer {layer_index} must be an IncrementalSelfAttentionKVCache")
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


def decode_position_index(position: int | torch.Tensor | None, *, default: int) -> int:
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


def as_decode_step_tensor(
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


def as_decode_step_lane_tensor(
    value: torch.Tensor,
    *,
    name: str,
    batch_size: int,
    lanes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    tensor = value.to(device=device, dtype=dtype)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(1)
    expected = (int(batch_size), 1, int(lanes))
    if tensor.ndim != 3 or tuple(tensor.shape) != expected:
        raise ValueError(f"{name} must have shape [B,{lanes}] or [B,1,{lanes}], got {tuple(value.shape)}")
    return tensor.contiguous()


def as_decode_batch_vector(
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


def normalize_attention_kv_cache(
    raw_cache: tuple[AttentionKVCache, ...] | Sequence[AttentionKVCache] | None,
    *,
    device: torch.device,
    batch_size: int,
    layers: int,
    heads: int,
    d_model: int,
    source_steps: int,
    name: str,
) -> tuple[AttentionKVCache, ...] | None:
    if raw_cache is None:
        return None
    if not isinstance(raw_cache, (tuple, list)):
        raise ValueError(f"{name} must be a tuple/list of per-layer key/value tensors")
    if len(raw_cache) != int(layers):
        raise ValueError(f"{name} must contain {layers} layers, got {len(raw_cache)}")
    if int(d_model) % int(heads) != 0:
        raise ValueError("d_model must be divisible by heads")

    head_dim = int(d_model) // int(heads)
    expected_shape = (int(batch_size), int(heads), int(source_steps), head_dim)
    cache: list[AttentionKVCache] = []
    for layer_index, layer_cache in enumerate(raw_cache):
        if not isinstance(layer_cache, (tuple, list)) or len(layer_cache) != 2:
            raise ValueError(f"{name} layer {layer_index} must be a key/value pair")
        key, value = layer_cache
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} layer {layer_index} key/value must be tensors")
        key = key.detach().to(device=device, dtype=torch.float32)
        value = value.detach().to(device=device, dtype=torch.float32)
        if tuple(key.shape) != expected_shape:
            raise ValueError(
                f"{name} layer {layer_index} key must have shape {expected_shape}, "
                f"got {tuple(key.shape)}"
            )
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"{name} layer {layer_index} value must have shape {expected_shape}, "
                f"got {tuple(value.shape)}"
            )
        cache.append((key.contiguous(), value.contiguous()))
    return tuple(cache)


def attention_projection_to_heads(projection: torch.Tensor, *, heads: int) -> torch.Tensor:
    if projection.ndim != 3:
        raise ValueError(f"attention projection must have shape [B,S,D], got {tuple(projection.shape)}")
    if projection.shape[-1] % int(heads) != 0:
        raise ValueError("attention projection width must be divisible by head count")
    batch_size, steps, width = projection.shape
    head_dim = int(width) // int(heads)
    return projection.view(batch_size, steps, int(heads), head_dim).transpose(1, 2).contiguous()


def _incremental_transformer_decoder_layer_step(
    layer: nn.TransformerDecoderLayer,
    *,
    hidden: torch.Tensor,
    memory: torch.Tensor,
    self_attention_kv: IncrementalSelfAttentionKVCache,
    control_attention_kv: AttentionKVCache | None = None,
) -> tuple[torch.Tensor, IncrementalSelfAttentionKVCache]:
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


def _cached_control_cross_attention_step(
    layer: nn.TransformerDecoderLayer,
    *,
    query: torch.Tensor,
    control_attention_kv: AttentionKVCache,
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
    projected_query = attention_projection_to_heads(
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
    self_attention_kv: IncrementalSelfAttentionKVCache,
) -> tuple[torch.Tensor, IncrementalSelfAttentionKVCache]:
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
    projected_query = attention_projection_to_heads(
        F.linear(query, query_weight, query_bias),
        heads=heads,
    )
    current_key = attention_projection_to_heads(
        F.linear(query, key_weight, key_bias),
        heads=heads,
    )
    current_value = attention_projection_to_heads(
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
    return output, IncrementalSelfAttentionKVCache(key=key.detach(), value=value.detach())
