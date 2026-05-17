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
from .vocab import KEY_COUNT, LaneAction, MapperV21Vocab


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
    vocab: MapperV21Vocab,
    chart_end_ms: int | None = None,
    emitted_lane_mask: int | Sequence[bool] | torch.Tensor = 0,
    last_lane_index: int | torch.Tensor | None = None,
    min_ln_duration_ms: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    resolved_device = torch.device("cpu") if device is None else device
    emitted = _normalize_emitted_lane_mask(emitted_lane_mask)
    state = MapperReplayState(
        position=int(position),
        current_ms=int(current_ms),
        open_mask=_normalize_open_mask(open_mask),
        open_start_ms=_normalize_open_start_ms(open_start_ms),
        open_age_ms=_normalize_open_age_ms(open_age_ms),
        emitted_lane_mask=emitted,
        last_lane_index=_normalize_last_lane_index(last_lane_index, emitted),
    )
    mask = torch.zeros(vocab.size, dtype=torch.bool, device=resolved_device)
    target_end_ms = _target_end_ms(
        write_end_ms=write_end_ms,
        chart_end_ms=chart_end_ms,
        is_full_chart_end=is_full_chart_end,
    )

    if int(position) < 0:
        if (
            bool(is_full_chart_start)
            and int(state.current_ms) == int(write_start_ms)
            and replay_state_matches_carry(state, ln_carry_in)
            and not any(state.emitted_lane_mask)
        ):
            mask[vocab.bos_id] = True
        return mask

    for token_id in range(vocab.size):
        if token_id in {vocab.pad_id, vocab.bos_id}:
            continue
        if min_ln_duration_ms is not None and vocab.is_lane_action_token(token_id):
            lane, action = vocab.decode_lane_action(token_id)
            remaining_ms = int(target_end_ms) - int(current_ms)
            if not _lane_action_is_legal(
                lane=lane,
                action=action,
                open_mask=state.open_mask,
                current_ms=state.current_ms,
                remaining_ms=remaining_ms,
                ln_carry_out=ln_carry_out,
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
                chart_end_ms=chart_end_ms,
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
    vocab: MapperV21Vocab,
    chart_end_ms: torch.Tensor | int | None = None,
    emitted_lane_mask: torch.Tensor | None = None,
    last_lane_index: torch.Tensor | None = None,
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
    if emitted_lane_mask is None:
        emitted_lane_mask = torch.zeros((batch_size, steps, KEY_COUNT), dtype=torch.bool, device=device)
    elif emitted_lane_mask.ndim == 2:
        emitted_lane_mask = emitted_lane_mask.unsqueeze(0)
    emitted_lane_mask = emitted_lane_mask.to(device=device, dtype=torch.bool)
    if tuple(emitted_lane_mask.shape) != (batch_size, steps, KEY_COUNT):
        raise ValueError(f"emitted_lane_mask must have shape {(batch_size, steps, KEY_COUNT)}, got {tuple(emitted_lane_mask.shape)}")

    if last_lane_index is None:
        lane_ids = torch.arange(KEY_COUNT, dtype=torch.long, device=device).reshape(1, 1, KEY_COUNT)
        last_lane_index = torch.where(
            emitted_lane_mask,
            lane_ids.expand(batch_size, steps, KEY_COUNT),
            torch.full((batch_size, steps, KEY_COUNT), -1, dtype=torch.long, device=device),
        ).amax(dim=-1)
    elif last_lane_index.ndim == 1:
        last_lane_index = last_lane_index.unsqueeze(0).expand(batch_size, steps)
    last_lane_index = last_lane_index.to(device=device, dtype=torch.long)
    if tuple(last_lane_index.shape) != (batch_size, steps):
        raise ValueError(f"last_lane_index must have shape {(batch_size, steps)}, got {tuple(last_lane_index.shape)}")

    if positions is None:
        positions = torch.arange(steps, dtype=torch.long, device=device).expand(batch_size, steps)
    elif positions.ndim == 1:
        positions = positions.unsqueeze(0).expand(batch_size, steps)
    if tuple(positions.shape) != (batch_size, steps):
        raise ValueError(f"positions must have shape {(batch_size, steps)}, got {tuple(positions.shape)}")

    write_start_values = _broadcast_window_tensor(write_start_ms, batch_size=batch_size, device=device)
    write_end_values = _broadcast_window_tensor(write_end_ms, batch_size=batch_size, device=device)
    chart_end_values = (
        write_end_values
        if chart_end_ms is None
        else _broadcast_window_tensor(chart_end_ms, batch_size=batch_size, device=device)
    )
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
                emitted_lane_mask=emitted_lane_mask[batch_index, step],
                last_lane_index=int(last_lane_index[batch_index, step].item()),
                write_start_ms=int(write_start_values[batch_index].item()),
                write_end_ms=int(write_end_values[batch_index].item()),
                chart_end_ms=int(chart_end_values[batch_index].item()),
                ln_carry_in=ln_in,
                ln_carry_out=ln_out,
                is_full_chart_start=bool(full_start_values[batch_index].item()),
                is_full_chart_end=bool(full_end_values[batch_index].item()),
                vocab=vocab,
                min_ln_duration_ms=min_ln_duration_ms,
                device=device,
            )
    return torch.zeros_like(valid, dtype=torch.float32).masked_fill(~valid, invalid_value)


def _lane_action_is_legal(
    *,
    lane: int,
    action: LaneAction,
    open_mask: Sequence[bool],
    current_ms: int,
    remaining_ms: int,
    ln_carry_out: LNCarryState,
    min_ln_duration_ms: int | None,
) -> bool:
    is_open = bool(open_mask[lane])
    if is_open and action != LaneAction.HOLD_END:
        return False
    if not is_open and action == LaneAction.HOLD_END:
        return False
    if (
        min_ln_duration_ms is not None
        and not is_open
        and action == LaneAction.HOLD_START
        and remaining_ms < int(min_ln_duration_ms)
    ):
        carries_across_window = (
            bool(ln_carry_out.open_mask[lane])
            and ln_carry_out.open_start_ms[lane] == int(current_ms)
        )
        return carries_across_window
    return True


def _target_end_ms(
    *,
    write_end_ms: int,
    chart_end_ms: int | None,
    is_full_chart_end: bool,
) -> int:
    if bool(is_full_chart_end):
        return int(write_end_ms) if chart_end_ms is None else int(chart_end_ms)
    return int(write_end_ms)


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


def _normalize_emitted_lane_mask(mask: int | Sequence[bool] | torch.Tensor) -> tuple[bool, bool, bool, bool]:
    return _normalize_open_mask(mask)


def _normalize_last_lane_index(
    value: int | torch.Tensor | None,
    emitted_lane_mask: Sequence[bool],
) -> int:
    if value is not None:
        if isinstance(value, torch.Tensor):
            return int(value.detach().cpu().reshape(()).item())
        return int(value)
    emitted = [lane for lane, was_emitted in enumerate(emitted_lane_mask) if bool(was_emitted)]
    return max(emitted) if emitted else -1


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
            "current_ms": torch.full((batch_size,), value.current_ms, dtype=torch.long, device=device),
            "open_mask": torch.tensor(value.open_mask, dtype=torch.bool, device=device).reshape(1, KEY_COUNT).expand(batch_size, -1),
            "open_start_ms": torch.tensor(
                [CLOSED_OPEN_START_MS if item is None else int(item) for item in value.open_start_ms],
                dtype=torch.long,
                device=device,
            )
            .reshape(1, KEY_COUNT)
            .expand(batch_size, -1),
            "open_age_ms": torch.tensor(value.open_age_ms, dtype=torch.long, device=device).reshape(1, KEY_COUNT).expand(batch_size, -1),
        }
    if not isinstance(value, Mapping):
        raise ValueError("carry state must be LNCarryState or mapping")
    current_ms = _broadcast_carry_tensor(value.get("current_ms"), batch_size=batch_size, device=device, name="current_ms", lane=False)
    open_mask = _broadcast_carry_tensor(value.get("open_mask"), batch_size=batch_size, device=device, name="open_mask", lane=True).to(
        dtype=torch.bool,
    )
    open_start_ms = _broadcast_carry_tensor(value.get("open_start_ms"), batch_size=batch_size, device=device, name="open_start_ms", lane=True)
    open_age_ms = _broadcast_carry_tensor(value.get("open_age_ms"), batch_size=batch_size, device=device, name="open_age_ms", lane=True)
    return {
        "current_ms": current_ms,
        "open_mask": open_mask,
        "open_start_ms": open_start_ms,
        "open_age_ms": open_age_ms,
    }


def _broadcast_carry_tensor(
    value: Any,
    *,
    batch_size: int,
    device: torch.device,
    name: str,
    lane: bool,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"carry[{name!r}] must be a tensor")
    dtype = torch.bool if name == "open_mask" else torch.long
    tensor = value.to(device=device, dtype=dtype)
    if lane:
        if tensor.ndim == 1 and int(tensor.shape[0]) == KEY_COUNT:
            return tensor.reshape(1, KEY_COUNT).expand(batch_size, -1)
        if tensor.ndim == 2 and tuple(tensor.shape) == (batch_size, KEY_COUNT):
            return tensor
        raise ValueError(f"carry[{name!r}] must have shape [{KEY_COUNT}] or [B,{KEY_COUNT}], got {tuple(tensor.shape)}")
    tensor = tensor.reshape(-1)
    if tensor.numel() == 1:
        return tensor.expand(batch_size)
    if tensor.numel() == batch_size:
        return tensor
    raise ValueError(f"carry[{name!r}] must have 1 or {batch_size} values, got {tensor.numel()}")


def _carry_at(values: Mapping[str, torch.Tensor], index: int) -> LNCarryState:
    return LNCarryState(
        current_ms=int(values["current_ms"][index].item()),
        open_mask=tuple(bool(item) for item in values["open_mask"][index].tolist()),  # type: ignore[arg-type]
        open_start_ms=open_start_tensor_values_to_tuple(values["open_start_ms"][index].detach().cpu().tolist()),
        open_age_ms=tuple(int(item) for item in values["open_age_ms"][index].tolist()),  # type: ignore[arg-type]
    )
