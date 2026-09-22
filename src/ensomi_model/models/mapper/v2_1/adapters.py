from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .tokenizer import MAPPER_DENSITY_FRAME_MS, MAPPER_DENSITY_FRAMES, MAPPER_WRITE_MS
from .vocab import KEY_COUNT, SPARSE_LANE_ACTIONS, LaneAction, MapperV21Vocab


LANE_ACTION_COUNT = len(SPARSE_LANE_ACTIONS)


@dataclass(frozen=True)
class StatePriorAdapterOutput:
    lane_action_bias: torch.Tensor
    vocab_bias: torch.Tensor

    def __iter__(self):
        yield self.lane_action_bias
        yield self.vocab_bias


@dataclass(frozen=True)
class LNCloseAdapterOutput:
    close_logits: torch.Tensor
    lane_action_bias: torch.Tensor
    time_shift_bias: torch.Tensor

    @property
    def lane_bias(self) -> torch.Tensor:
        return self.lane_action_bias

    @property
    def ln_close_lane_bias(self) -> torch.Tensor:
        return self.lane_action_bias

    @property
    def event_bias(self) -> torch.Tensor:
        return self.lane_action_bias

    @property
    def ln_close_bias(self) -> torch.Tensor:
        return self.lane_action_bias

    @property
    def ln_close_event_bias(self) -> torch.Tensor:
        return self.lane_action_bias

    @property
    def ln_close_time_shift_bias(self) -> torch.Tensor:
        return self.time_shift_bias

    def __iter__(self):
        yield self.close_logits
        yield self.lane_action_bias
        yield self.time_shift_bias


class StatePriorAdapter(nn.Module):
    """State-only structured prior projected directly to sparse lane tokens."""

    def __init__(
        self,
        *,
        vocab: MapperV21Vocab,
        hidden_dim: int = 64,
        lane_embedding_dim: int = 8,
        age_embedding_dim: int = 8,
        num_age_buckets: int = 32,
        age_cap_ms: int = 4000,
        remaining_cap_ms: int = MAPPER_WRITE_MS,
        adapter_scale_init: float = 0.03,
        max_bias: float = 1.5,
    ) -> None:
        super().__init__()
        _validate_positive_int(hidden_dim, "hidden_dim")
        _validate_positive_int(lane_embedding_dim, "lane_embedding_dim")
        _validate_positive_int(age_embedding_dim, "age_embedding_dim")
        _validate_positive_int(num_age_buckets, "num_age_buckets")
        _validate_positive_int(age_cap_ms, "age_cap_ms")
        _validate_positive_int(remaining_cap_ms, "remaining_cap_ms")
        if max_bias <= 0.0:
            raise ValueError("max_bias must be positive")

        self.vocab = vocab
        self.num_age_buckets = int(num_age_buckets)
        self.age_cap_ms = int(age_cap_ms)
        self.remaining_cap_ms = int(remaining_cap_ms)
        self.max_bias = float(max_bias)
        self.adapter_scale = nn.Parameter(torch.tensor(float(adapter_scale_init), dtype=torch.float32))
        self.lane_embedding = nn.Embedding(KEY_COUNT, lane_embedding_dim)
        self.age_embedding = nn.Embedding(self.num_age_buckets, age_embedding_dim)
        input_dim = 5 + lane_embedding_dim + age_embedding_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, LANE_ACTION_COUNT),
        )
        final = self.net[-1]
        assert isinstance(final, nn.Linear)
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        *,
        open_mask: torch.Tensor,
        open_age_ms: torch.Tensor,
        remaining_ms: torch.Tensor,
        open_start_ms: torch.Tensor | None = None,
        write_start_ms: torch.Tensor | int = 0,
    ) -> StatePriorAdapterOutput:
        open_mask, open_start_ms, open_age_ms, remaining_ms = _validate_lane_state_inputs(
            open_mask=open_mask,
            open_start_ms=open_start_ms,
            open_age_ms=open_age_ms,
            remaining_ms=remaining_ms,
        )
        features = self._state_features(
            open_mask=open_mask,
            open_start_ms=open_start_ms,
            open_age_ms=open_age_ms,
            remaining_ms=remaining_ms,
            write_start_ms=write_start_ms,
        )
        raw_bias = self.net(features)
        lane_action_bias = self.adapter_scale.to(dtype=raw_bias.dtype) * self.max_bias * torch.tanh(raw_bias / self.max_bias)
        return StatePriorAdapterOutput(
            lane_action_bias=lane_action_bias,
            vocab_bias=self.project_lane_action_bias(lane_action_bias),
        )

    def project_lane_action_bias(self, lane_action_bias: torch.Tensor) -> torch.Tensor:
        return project_lane_action_bias_to_lane_tokens(lane_action_bias, self.vocab)

    def _state_features(
        self,
        *,
        open_mask: torch.Tensor,
        open_start_ms: torch.Tensor,
        open_age_ms: torch.Tensor,
        remaining_ms: torch.Tensor,
        write_start_ms: torch.Tensor | int,
    ) -> torch.Tensor:
        batch_size, steps, _ = open_mask.shape
        device = open_mask.device
        open_float = open_mask.to(dtype=torch.float32).unsqueeze(-1)
        write_start = _broadcast_window_tensor(write_start_ms, batch_size=batch_size, device=device)
        start_known = (open_mask & (open_start_ms >= 0)).to(dtype=torch.float32).unsqueeze(-1)
        start_rel = (
            (open_start_ms.to(dtype=torch.float32) - write_start.reshape(batch_size, 1, 1).to(dtype=torch.float32))
            / float(self.remaining_cap_ms)
        ).clamp(-1.0, 1.0)
        start_rel = torch.where(open_mask, start_rel, torch.zeros_like(start_rel)).unsqueeze(-1)
        age_norm = (open_age_ms.to(dtype=torch.float32) / float(self.age_cap_ms)).clamp(0.0, 1.0)
        age_bucket = torch.floor(age_norm * float(self.num_age_buckets - 1)).to(dtype=torch.long)
        remaining_norm = (
            remaining_ms.to(dtype=torch.float32).reshape(batch_size, steps, 1, 1)
            / float(self.remaining_cap_ms)
        ).clamp(0.0, 1.0)
        remaining_feature = remaining_norm.expand(batch_size, steps, KEY_COUNT, 1)
        lane_ids = torch.arange(KEY_COUNT, dtype=torch.long, device=device)
        lane_feature = self.lane_embedding(lane_ids).reshape(1, 1, KEY_COUNT, -1).expand(batch_size, steps, -1, -1)
        age_feature = self.age_embedding(age_bucket)
        return torch.cat(
            (
                open_float,
                start_known,
                start_rel,
                age_norm.unsqueeze(-1),
                remaining_feature,
                lane_feature,
                age_feature,
            ),
            dim=-1,
        )


class LNCloseAdapter(nn.Module):
    """Context-aware lane-level close hazard head for sparse lane tokens."""

    def __init__(
        self,
        *,
        vocab: MapperV21Vocab,
        d_model: int,
        hidden_dim: int = 128,
        lane_embedding_dim: int = 8,
        age_embedding_dim: int = 8,
        num_age_buckets: int = 32,
        age_cap_ms: int = 4000,
        remaining_cap_ms: int = MAPPER_WRITE_MS,
        local_radius_frames: int = 2,
        close_scale: float = 0.05,
        skip_scale: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_positive_int(d_model, "d_model")
        _validate_positive_int(hidden_dim, "hidden_dim")
        _validate_positive_int(lane_embedding_dim, "lane_embedding_dim")
        _validate_positive_int(age_embedding_dim, "age_embedding_dim")
        _validate_positive_int(num_age_buckets, "num_age_buckets")
        _validate_positive_int(age_cap_ms, "age_cap_ms")
        _validate_positive_int(remaining_cap_ms, "remaining_cap_ms")
        if local_radius_frames < 0:
            raise ValueError("local_radius_frames must be non-negative")
        if close_scale < 0.0:
            raise ValueError("close_scale must be non-negative")
        if skip_scale < 0.0:
            raise ValueError("skip_scale must be non-negative")

        self.vocab = vocab
        self.d_model = int(d_model)
        self.num_age_buckets = int(num_age_buckets)
        self.age_cap_ms = int(age_cap_ms)
        self.remaining_cap_ms = int(remaining_cap_ms)
        self.local_radius_frames = int(local_radius_frames)
        self.close_scale = float(close_scale)
        self.skip_scale = float(skip_scale)
        self.lane_embedding = nn.Embedding(KEY_COUNT, lane_embedding_dim)
        self.age_embedding = nn.Embedding(self.num_age_buckets, age_embedding_dim)
        input_dim = 2 * self.d_model + 6 + lane_embedding_dim + age_embedding_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        final = self.net[-1]
        assert isinstance(final, nn.Linear)
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final.bias)
        self.register_buffer(
            "time_shift_token_ids",
            torch.tensor(vocab.time_shift_token_ids, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        *,
        decoder_hidden: torch.Tensor,
        control_memory_8s: torch.Tensor,
        density_teacher_8s: torch.Tensor,
        current_ms: torch.Tensor,
        write_start_ms: torch.Tensor | int,
        open_mask: torch.Tensor,
        open_age_ms: torch.Tensor,
        remaining_ms: torch.Tensor,
        open_start_ms: torch.Tensor | None = None,
    ) -> LNCloseAdapterOutput:
        open_mask, open_start_ms, open_age_ms, remaining_ms = _validate_lane_state_inputs(
            open_mask=open_mask,
            open_start_ms=open_start_ms,
            open_age_ms=open_age_ms,
            remaining_ms=remaining_ms,
        )
        if decoder_hidden.ndim != 3 or int(decoder_hidden.shape[-1]) != self.d_model:
            raise ValueError(f"decoder_hidden must have shape [B,T,{self.d_model}], got {tuple(decoder_hidden.shape)}")
        if tuple(decoder_hidden.shape[:2]) != tuple(open_mask.shape[:2]):
            raise ValueError("decoder_hidden and lane state tensors must share [B,T]")
        if control_memory_8s.ndim != 3 or int(control_memory_8s.shape[1]) != MAPPER_DENSITY_FRAMES:
            raise ValueError(
                f"control_memory_8s must have shape [B,{MAPPER_DENSITY_FRAMES},D], "
                f"got {tuple(control_memory_8s.shape)}"
            )
        if int(control_memory_8s.shape[-1]) != self.d_model:
            raise ValueError(f"control_memory_8s last dim must be {self.d_model}, got {control_memory_8s.shape[-1]}")
        if tuple(control_memory_8s.shape[:1]) != tuple(decoder_hidden.shape[:1]):
            raise ValueError("control_memory_8s and decoder_hidden must share batch size")
        if tuple(density_teacher_8s.shape) != (decoder_hidden.shape[0], MAPPER_DENSITY_FRAMES, 1):
            raise ValueError(
                f"density_teacher_8s must have shape [B,{MAPPER_DENSITY_FRAMES},1], "
                f"got {tuple(density_teacher_8s.shape)}"
            )
        if tuple(current_ms.shape) != tuple(decoder_hidden.shape[:2]):
            raise ValueError(f"current_ms must have shape {tuple(decoder_hidden.shape[:2])}")

        local_control, local_density = self._gather_local_control(
            control_memory_8s=control_memory_8s,
            density_teacher_8s=density_teacher_8s,
            current_ms=current_ms,
            write_start_ms=write_start_ms,
        )
        features = self._close_features(
            decoder_hidden=decoder_hidden,
            local_control=local_control,
            local_density=local_density,
            open_mask=open_mask,
            open_start_ms=open_start_ms,
            open_age_ms=open_age_ms,
            remaining_ms=remaining_ms,
            write_start_ms=write_start_ms,
        )
        close_logits = self.net(features).squeeze(-1)
        close_bias = self.close_scale * torch.tanh(close_logits)
        lane_action_bias = self._project_close_lane_action_bias(close_bias=close_bias, open_mask=open_mask)
        time_shift_bias = self._project_time_shift_bias(close_logits=close_logits, open_mask=open_mask)
        return LNCloseAdapterOutput(
            close_logits=close_logits,
            lane_action_bias=lane_action_bias,
            time_shift_bias=time_shift_bias,
        )

    def _gather_local_control(
        self,
        *,
        control_memory_8s: torch.Tensor,
        density_teacher_8s: torch.Tensor,
        current_ms: torch.Tensor,
        write_start_ms: torch.Tensor | int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, steps = current_ms.shape
        write_start = _broadcast_window_tensor(write_start_ms, batch_size=batch_size, device=current_ms.device)
        relative_ms = current_ms.to(dtype=torch.long) - write_start.reshape(batch_size, 1)
        if bool(((relative_ms < 0) | (relative_ms > MAPPER_WRITE_MS)).any()):
            raise ValueError("current_ms must be within the 8s mapper write window before local-control gather")
        raw_frame_idx = torch.div(relative_ms, MAPPER_DENSITY_FRAME_MS, rounding_mode="floor")

        offsets = torch.arange(
            -self.local_radius_frames,
            self.local_radius_frames + 1,
            dtype=torch.long,
            device=current_ms.device,
        )
        raw_neighborhood = raw_frame_idx.unsqueeze(-1) + offsets
        valid_neighborhood = (raw_neighborhood >= 0) & (raw_neighborhood < MAPPER_DENSITY_FRAMES)
        neighborhood = raw_neighborhood.clamp(0, MAPPER_DENSITY_FRAMES - 1)
        batch_index = torch.arange(batch_size, dtype=torch.long, device=current_ms.device).reshape(batch_size, 1, 1)
        weights = valid_neighborhood.to(dtype=control_memory_8s.dtype).unsqueeze(-1)
        denom = weights.sum(dim=2).clamp_min(1.0)
        local_control = (control_memory_8s[batch_index, neighborhood] * weights).sum(dim=2) / denom
        density_weights = valid_neighborhood.to(dtype=density_teacher_8s.dtype).unsqueeze(-1)
        local_density = (density_teacher_8s[batch_index, neighborhood] * density_weights).sum(dim=2) / density_weights.sum(
            dim=2,
        ).clamp_min(1.0)
        return local_control, local_density

    def _close_features(
        self,
        *,
        decoder_hidden: torch.Tensor,
        local_control: torch.Tensor,
        local_density: torch.Tensor,
        open_mask: torch.Tensor,
        open_start_ms: torch.Tensor,
        open_age_ms: torch.Tensor,
        remaining_ms: torch.Tensor,
        write_start_ms: torch.Tensor | int,
    ) -> torch.Tensor:
        batch_size, steps, _ = decoder_hidden.shape
        device = decoder_hidden.device
        open_float = open_mask.to(dtype=decoder_hidden.dtype).unsqueeze(-1)
        write_start = _broadcast_window_tensor(write_start_ms, batch_size=batch_size, device=device)
        start_known = (open_mask & (open_start_ms >= 0)).to(dtype=decoder_hidden.dtype).unsqueeze(-1)
        start_rel = (
            (
                open_start_ms.to(device=device, dtype=decoder_hidden.dtype)
                - write_start.reshape(batch_size, 1, 1).to(dtype=decoder_hidden.dtype)
            )
            / float(self.remaining_cap_ms)
        ).clamp(-1.0, 1.0)
        start_rel = torch.where(open_mask, start_rel, torch.zeros_like(start_rel)).unsqueeze(-1)
        age_norm = (open_age_ms.to(dtype=decoder_hidden.dtype) / float(self.age_cap_ms)).clamp(0.0, 1.0)
        age_bucket = torch.floor(age_norm * float(self.num_age_buckets - 1)).to(dtype=torch.long)
        remaining_norm = (
            remaining_ms.to(dtype=decoder_hidden.dtype).reshape(batch_size, steps, 1, 1)
            / float(self.remaining_cap_ms)
        ).clamp(0.0, 1.0)
        remaining_feature = remaining_norm.expand(batch_size, steps, KEY_COUNT, 1)
        lane_ids = torch.arange(KEY_COUNT, dtype=torch.long, device=device)
        lane_feature = self.lane_embedding(lane_ids).reshape(1, 1, KEY_COUNT, -1).expand(batch_size, steps, -1, -1)
        age_feature = self.age_embedding(age_bucket)
        decoder_feature = decoder_hidden.unsqueeze(2).expand(batch_size, steps, KEY_COUNT, -1)
        control_feature = local_control.unsqueeze(2).expand(batch_size, steps, KEY_COUNT, -1)
        density_feature = local_density.unsqueeze(2).expand(batch_size, steps, KEY_COUNT, -1)
        return torch.cat(
            (
                decoder_feature,
                control_feature,
                density_feature,
                open_float,
                start_known,
                start_rel,
                age_norm.unsqueeze(-1),
                remaining_feature,
                lane_feature,
                age_feature,
            ),
            dim=-1,
        )

    def _project_close_lane_action_bias(self, *, close_bias: torch.Tensor, open_mask: torch.Tensor) -> torch.Tensor:
        vocab_bias = close_bias.new_zeros((close_bias.shape[0], close_bias.shape[1], self.vocab.size))
        open_float = open_mask.to(dtype=close_bias.dtype)
        for lane in range(KEY_COUNT):
            token_id = self.vocab.lane_action_token_id(lane, LaneAction.HOLD_END)
            vocab_bias[:, :, token_id] = close_bias[:, :, lane] * open_float[:, :, lane]
        return vocab_bias

    def _project_time_shift_bias(self, *, close_logits: torch.Tensor, open_mask: torch.Tensor) -> torch.Tensor:
        vocab_bias = close_logits.new_zeros((close_logits.shape[0], close_logits.shape[1], self.vocab.size))
        if self.skip_scale == 0.0 or int(self.time_shift_token_ids.numel()) == 0:
            return vocab_bias

        open_bool = open_mask.to(dtype=torch.bool)
        hazard = torch.sigmoid(close_logits).masked_fill(~open_bool, 0.0)
        any_close_hazard = hazard.max(dim=-1).values
        any_open = open_bool.any(dim=-1)
        penalty = -float(self.skip_scale) * any_close_hazard.masked_fill(~any_open, 0.0)
        vocab_bias[:, :, self.time_shift_token_ids] = penalty.unsqueeze(-1)
        return vocab_bias


def project_lane_action_bias_to_lane_tokens(
    lane_action_bias: torch.Tensor,
    vocab: MapperV21Vocab,
) -> torch.Tensor:
    if lane_action_bias.ndim != 4 or tuple(lane_action_bias.shape[-2:]) != (KEY_COUNT, LANE_ACTION_COUNT):
        raise ValueError(
            f"lane_action_bias must have shape [B,T,{KEY_COUNT},{LANE_ACTION_COUNT}], "
            f"got {tuple(lane_action_bias.shape)}"
        )
    vocab_bias = lane_action_bias.new_zeros((lane_action_bias.shape[0], lane_action_bias.shape[1], vocab.size))
    for lane in range(KEY_COUNT):
        for action_index, action in enumerate(SPARSE_LANE_ACTIONS):
            token_id = vocab.lane_action_token_id(lane, action)
            vocab_bias[:, :, token_id] = lane_action_bias[:, :, lane, action_index]
    return vocab_bias


def project_ln_close_bias_to_tokens(
    *,
    close_logits: torch.Tensor,
    open_mask: torch.Tensor,
    vocab: MapperV21Vocab,
    close_scale: float = 0.05,
    skip_scale: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if close_logits.ndim != 3 or int(close_logits.shape[-1]) != KEY_COUNT:
        raise ValueError(f"close_logits must have shape [B,T,{KEY_COUNT}], got {tuple(close_logits.shape)}")
    if tuple(open_mask.shape) != tuple(close_logits.shape):
        raise ValueError(f"open_mask must have shape {tuple(close_logits.shape)}, got {tuple(open_mask.shape)}")
    open_float = open_mask.to(device=close_logits.device, dtype=close_logits.dtype)
    close_bias = float(close_scale) * torch.tanh(close_logits)
    ln_close_bias = close_logits.new_zeros((close_logits.shape[0], close_logits.shape[1], vocab.size))
    for lane in range(KEY_COUNT):
        token_id = vocab.lane_action_token_id(lane, LaneAction.HOLD_END)
        ln_close_bias[:, :, token_id] = close_bias[:, :, lane] * open_float[:, :, lane]

    time_shift_bias = close_logits.new_zeros((close_logits.shape[0], close_logits.shape[1], vocab.size))
    if float(skip_scale) != 0.0 and len(vocab.time_shift_token_ids) > 0:
        time_shift_token_ids = torch.tensor(vocab.time_shift_token_ids, dtype=torch.long, device=close_logits.device)
        open_bool = open_mask.to(device=close_logits.device, dtype=torch.bool)
        hazard = torch.sigmoid(close_logits).masked_fill(~open_bool, 0.0)
        any_open = open_bool.any(dim=-1)
        penalty = -float(skip_scale) * hazard.max(dim=-1).values.masked_fill(~any_open, 0.0)
        time_shift_bias[:, :, time_shift_token_ids] = penalty.unsqueeze(-1)
    return ln_close_bias, time_shift_bias


def gather_local_control(
    *,
    control_memory_8s: torch.Tensor,
    current_ms: torch.Tensor,
    write_start_ms: torch.Tensor | int,
    frame_ms: int = MAPPER_DENSITY_FRAME_MS,
) -> torch.Tensor:
    if control_memory_8s.ndim != 3:
        raise ValueError(f"control_memory_8s must have shape [B,F,D], got {tuple(control_memory_8s.shape)}")
    if current_ms.ndim != 2:
        raise ValueError(f"current_ms must have shape [B,T], got {tuple(current_ms.shape)}")
    batch_size, frame_count, dim = control_memory_8s.shape
    if int(current_ms.shape[0]) != batch_size:
        raise ValueError("control_memory_8s and current_ms must share batch size")
    write_start = _broadcast_window_tensor(write_start_ms, batch_size=batch_size, device=current_ms.device)
    relative_ms = current_ms.to(dtype=torch.long) - write_start.reshape(batch_size, 1)
    max_relative_ms = int(frame_count) * int(frame_ms)
    if bool(((relative_ms < 0) | (relative_ms > max_relative_ms)).any()):
        raise ValueError("current_ms must be within the provided control-memory frame span")
    frame_idx = torch.div(relative_ms, int(frame_ms), rounding_mode="floor").clamp(0, frame_count - 1)
    frame_idx = frame_idx.to(device=control_memory_8s.device, dtype=torch.long)
    return control_memory_8s.gather(dim=1, index=frame_idx.unsqueeze(-1).expand(-1, -1, dim))


def _validate_lane_state_inputs(
    *,
    open_mask: torch.Tensor,
    open_start_ms: torch.Tensor | None,
    open_age_ms: torch.Tensor,
    remaining_ms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if open_mask.ndim != 3 or int(open_mask.shape[-1]) != KEY_COUNT:
        raise ValueError(f"open_mask must have shape [B,T,{KEY_COUNT}], got {tuple(open_mask.shape)}")
    if open_age_ms.shape != open_mask.shape:
        raise ValueError(f"open_age_ms must match open_mask shape, got {tuple(open_age_ms.shape)}")
    if remaining_ms.ndim != 2 or tuple(remaining_ms.shape) != tuple(open_mask.shape[:2]):
        raise ValueError(f"remaining_ms must have shape {tuple(open_mask.shape[:2])}, got {tuple(remaining_ms.shape)}")
    if open_start_ms is None:
        open_start_ms = torch.full_like(open_age_ms.to(dtype=torch.long), -1)
    if open_start_ms.shape != open_mask.shape:
        raise ValueError(f"open_start_ms must match open_mask shape, got {tuple(open_start_ms.shape)}")
    return (
        open_mask.to(dtype=torch.bool),
        open_start_ms.to(device=open_mask.device, dtype=torch.long),
        open_age_ms.to(device=open_mask.device, dtype=torch.long),
        remaining_ms.to(device=open_mask.device, dtype=torch.long),
    )


def _broadcast_window_tensor(value: torch.Tensor | int, *, batch_size: int, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=torch.long).reshape(-1)
        if tensor.numel() == 1:
            return tensor.expand(batch_size)
        if tensor.numel() == batch_size:
            return tensor
        raise ValueError(f"window tensor must have 1 or {batch_size} values, got {tensor.numel()}")
    return torch.full((batch_size,), int(value), dtype=torch.long, device=device)


def _validate_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
