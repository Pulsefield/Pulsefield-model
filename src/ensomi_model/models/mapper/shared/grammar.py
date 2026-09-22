from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .replay import (
    CLOSED_OPEN_START_MS,
    LNCarryState,
    MapperReplayState,
    ReplayError,
    open_start_tensor_values_to_tuple,
    replay_state_matches_carry,
    transition_replay_state,
)
from .vocab import KEY_COUNT, LaneAction, MapperTupleVocab


def valid_token_mask(
    *,
    position: int,
    current_ms: int,
    open_mask: int | Sequence[bool] | torch.Tensor,
    open_start_ms: Sequence[int | None] | torch.Tensor,
    open_age_ms: Sequence[int] | torch.Tensor,
    write_start_ms: int,
    write_end_ms: int,
    ln_carry_in: LNCarryState,
    ln_carry_out: LNCarryState,
    is_full_chart_start: bool,
    is_full_chart_end: bool,
    vocab: MapperTupleVocab,
    min_ln_duration_ms: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    resolved_device = torch.device("cpu") if device is None else device
    mask = torch.zeros(vocab.size, dtype=torch.bool, device=resolved_device)
    state = MapperReplayState(
        position=int(position),
        current_ms=int(current_ms),
        open_mask=_normalize_open_mask(open_mask),
        open_start_ms=_normalize_open_start_ms(open_start_ms),
        open_age_ms=_normalize_open_age_ms(open_age_ms),
    )

    if int(position) < 0:
        if (
            bool(is_full_chart_start)
            and int(state.current_ms) == int(write_start_ms)
            and replay_state_matches_carry(state, ln_carry_in)
        ):
            mask[vocab.bos_id] = True
        return mask

    for token_id in range(vocab.size):
        if token_id in {vocab.pad_id, vocab.bos_id}:
            continue
        if min_ln_duration_ms is not None and vocab.is_event_token(token_id):
            remaining_ms = int(write_end_ms) - int(current_ms)
            if not _event_is_legal(
                vocab.decode_event(token_id),
                state.open_mask,
                remaining_ms=remaining_ms,
                min_ln_duration_ms=min_ln_duration_ms,
            ):
                continue
        try:
            transition_replay_state(
                state,
                token_id,
                position=int(position),
                vocab=vocab,
                write_start_ms=int(write_start_ms),
                write_end_ms=int(write_end_ms),
                ln_carry_out=ln_carry_out,
                is_full_chart_start=bool(is_full_chart_start),
                is_full_chart_end=bool(is_full_chart_end),
            )
        except (ReplayError, ValueError):
            continue
        mask[token_id] = True
    return mask


def build_grammar_mask(
    *,
    current_ms: torch.Tensor,
    open_mask: torch.Tensor,
    open_start_ms: torch.Tensor,
    open_age_ms: torch.Tensor,
    write_start_ms: torch.Tensor | int,
    write_end_ms: torch.Tensor | int,
    ln_carry_in: LNCarryState | Mapping[str, Any],
    ln_carry_out: LNCarryState | Mapping[str, Any],
    is_full_chart_start: torch.Tensor | bool,
    is_full_chart_end: torch.Tensor | bool,
    vocab: MapperTupleVocab,
    positions: torch.Tensor | None = None,
    min_ln_duration_ms: int | None = None,
    invalid_value: float = -torch.inf,
) -> torch.Tensor:
    if current_ms.ndim == 1:
        current_ms = current_ms.unsqueeze(0)
    if open_mask.ndim == 2:
        open_mask = open_mask.unsqueeze(0)
    if open_start_ms.ndim == 2:
        open_start_ms = open_start_ms.unsqueeze(0)
    if open_age_ms.ndim == 2:
        open_age_ms = open_age_ms.unsqueeze(0)
    if current_ms.ndim != 2:
        raise ValueError(f"current_ms must have shape [B,T] or [T], got {tuple(current_ms.shape)}")
    if open_mask.shape[:2] != current_ms.shape or int(open_mask.shape[-1]) != KEY_COUNT:
        raise ValueError(f"open_mask must have shape {tuple(current_ms.shape)}x{KEY_COUNT}, got {tuple(open_mask.shape)}")
    if open_start_ms.shape != open_mask.shape:
        raise ValueError(f"open_start_ms must match open_mask shape, got {tuple(open_start_ms.shape)}")
    if open_age_ms.shape != open_mask.shape:
        raise ValueError(f"open_age_ms must match open_mask shape, got {tuple(open_age_ms.shape)}")

    batch_size, steps = current_ms.shape
    device = current_ms.device
    if positions is None:
        positions = torch.arange(steps, dtype=torch.long, device=device).expand(batch_size, steps)
    elif positions.ndim == 1:
        positions = positions.unsqueeze(0).expand(batch_size, steps)
    if tuple(positions.shape) != (batch_size, steps):
        raise ValueError(f"positions must have shape {(batch_size, steps)}, got {tuple(positions.shape)}")

    write_start_values = _broadcast_window_tensor(write_start_ms, batch_size=batch_size, device=device)
    write_end_values = _broadcast_window_tensor(write_end_ms, batch_size=batch_size, device=device)
    full_start_values = _broadcast_bool_tensor(is_full_chart_start, batch_size=batch_size, device=device)
    full_end_values = _broadcast_bool_tensor(is_full_chart_end, batch_size=batch_size, device=device)
    carry_in_values = _broadcast_carry(ln_carry_in, batch_size=batch_size, device=device)
    carry_out_values = _broadcast_carry(ln_carry_out, batch_size=batch_size, device=device)

    valid = torch.zeros((batch_size, steps, vocab.size), dtype=torch.bool, device=device)
    for batch_index in range(batch_size):
        ln_in = _carry_at(carry_in_values, batch_index)
        ln_out = _carry_at(carry_out_values, batch_index)
        for step in range(steps):
            valid[batch_index, step] = valid_token_mask(
                position=int(positions[batch_index, step].item()),
                current_ms=int(current_ms[batch_index, step].item()),
                open_mask=open_mask[batch_index, step],
                open_start_ms=open_start_ms[batch_index, step],
                open_age_ms=open_age_ms[batch_index, step],
                write_start_ms=int(write_start_values[batch_index].item()),
                write_end_ms=int(write_end_values[batch_index].item()),
                ln_carry_in=ln_in,
                ln_carry_out=ln_out,
                is_full_chart_start=bool(full_start_values[batch_index].item()),
                is_full_chart_end=bool(full_end_values[batch_index].item()),
                vocab=vocab,
                min_ln_duration_ms=min_ln_duration_ms,
                device=device,
            )
    return torch.zeros_like(valid, dtype=torch.float32).masked_fill(~valid, invalid_value)


def _event_is_legal(
    lane_actions: Sequence[LaneAction],
    open_mask: Sequence[bool],
    *,
    remaining_ms: int,
    min_ln_duration_ms: int | None,
) -> bool:
    for lane, action in enumerate(lane_actions):
        is_open = bool(open_mask[lane])
        if is_open and action not in {LaneAction.NONE, LaneAction.HOLD_END}:
            return False
        if not is_open and action == LaneAction.HOLD_END:
            return False
        if (
            min_ln_duration_ms is not None
            and not is_open
            and action == LaneAction.HOLD_START
            and remaining_ms < int(min_ln_duration_ms)
        ):
            return False
    return True


def _normalize_open_mask(open_mask: int | Sequence[bool] | torch.Tensor) -> tuple[bool, bool, bool, bool]:
    if isinstance(open_mask, int):
        if not 0 <= open_mask < 2**KEY_COUNT:
            raise ValueError(f"open_mask outside 4K range: {open_mask}")
        return tuple(bool(open_mask & (1 << lane)) for lane in range(KEY_COUNT))  # type: ignore[return-value]
    if isinstance(open_mask, torch.Tensor):
        values = open_mask.detach().cpu().reshape(-1).tolist()
    else:
        values = list(open_mask)
    if len(values) != KEY_COUNT:
        raise ValueError(f"open_mask must contain {KEY_COUNT} lanes: {open_mask}")
    return tuple(bool(value) for value in values)  # type: ignore[return-value]


def _normalize_open_start_ms(
    open_start_ms: Sequence[int | None] | torch.Tensor,
) -> tuple[int | None, int | None, int | None, int | None]:
    if isinstance(open_start_ms, torch.Tensor):
        values = open_start_ms.detach().cpu().reshape(-1).tolist()
        return open_start_tensor_values_to_tuple(values)
    values = list(open_start_ms)
    if len(values) != KEY_COUNT:
        raise ValueError(f"open_start_ms must contain {KEY_COUNT} lanes: {open_start_ms}")
    return tuple(None if value is None or int(value) == CLOSED_OPEN_START_MS else int(value) for value in values)  # type: ignore[return-value]


def _normalize_open_age_ms(open_age_ms: Sequence[int] | torch.Tensor) -> tuple[int, int, int, int]:
    if isinstance(open_age_ms, torch.Tensor):
        values = open_age_ms.detach().cpu().reshape(-1).tolist()
    else:
        values = list(open_age_ms)
    if len(values) != KEY_COUNT:
        raise ValueError(f"open_age_ms must contain {KEY_COUNT} lanes: {open_age_ms}")
    return tuple(int(value) for value in values)  # type: ignore[return-value]


def _broadcast_window_tensor(value: torch.Tensor | int, *, batch_size: int, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=torch.long).reshape(-1)
        if tensor.numel() == 1:
            return tensor.expand(batch_size)
        if tensor.numel() == batch_size:
            return tensor
        raise ValueError(f"window tensor must have 1 or {batch_size} values, got {tensor.numel()}")
    return torch.full((batch_size,), int(value), dtype=torch.long, device=device)


def _broadcast_bool_tensor(value: torch.Tensor | bool, *, batch_size: int, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=torch.bool).reshape(-1)
        if tensor.numel() == 1:
            return tensor.expand(batch_size)
        if tensor.numel() == batch_size:
            return tensor
        raise ValueError(f"flag tensor must have 1 or {batch_size} values, got {tensor.numel()}")
    return torch.full((batch_size,), bool(value), dtype=torch.bool, device=device)


def _broadcast_carry(
    value: LNCarryState | Mapping[str, Any],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if isinstance(value, LNCarryState):
        return {
            "current_ms": torch.full((batch_size,), int(value.current_ms), dtype=torch.long, device=device),
            "open_mask": torch.tensor(value.open_mask, dtype=torch.bool, device=device).reshape(1, KEY_COUNT).expand(batch_size, KEY_COUNT),
            "open_start_ms": torch.tensor(
                [CLOSED_OPEN_START_MS if item is None else int(item) for item in value.open_start_ms],
                dtype=torch.long,
                device=device,
            ).reshape(1, KEY_COUNT).expand(batch_size, KEY_COUNT),
            "open_age_ms": torch.tensor(value.open_age_ms, dtype=torch.long, device=device).reshape(1, KEY_COUNT).expand(batch_size, KEY_COUNT),
        }
    required = ("current_ms", "open_mask", "open_start_ms", "open_age_ms")
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"ln carry mapping missing keys: {missing}")
    return {
        "current_ms": _broadcast_window_tensor(value["current_ms"], batch_size=batch_size, device=device),
        "open_mask": _broadcast_lane_tensor(value["open_mask"], batch_size=batch_size, dtype=torch.bool, device=device),
        "open_start_ms": _broadcast_lane_tensor(value["open_start_ms"], batch_size=batch_size, dtype=torch.long, device=device),
        "open_age_ms": _broadcast_lane_tensor(value["open_age_ms"], batch_size=batch_size, dtype=torch.long, device=device),
    }


def _broadcast_lane_tensor(
    value: torch.Tensor,
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    tensor = value.to(device=device, dtype=dtype)
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)
    if tuple(tensor.shape) == (1, KEY_COUNT):
        return tensor.expand(batch_size, KEY_COUNT)
    if tuple(tensor.shape) == (batch_size, KEY_COUNT):
        return tensor
    raise ValueError(f"lane tensor must have shape [4] or [B,4], got {tuple(tensor.shape)}")


def _carry_at(values: Mapping[str, torch.Tensor], batch_index: int) -> LNCarryState:
    open_start_values = values["open_start_ms"][batch_index].detach().cpu().tolist()
    open_start_ms = open_start_tensor_values_to_tuple(open_start_values)
    current_ms = int(values["current_ms"][batch_index].item())
    open_mask = tuple(bool(value) for value in values["open_mask"][batch_index].detach().cpu().tolist())
    open_age_ms = tuple(int(value) for value in values["open_age_ms"][batch_index].detach().cpu().tolist())
    return LNCarryState(
        current_ms=current_ms,
        open_mask=open_mask,  # type: ignore[arg-type]
        open_start_ms=open_start_ms,
        open_age_ms=open_age_ms,  # type: ignore[arg-type]
    )
