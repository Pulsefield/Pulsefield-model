from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .replay import (
    LNCarryState,
    ReplayError,
    close_labels_from_tokens,
    empty_ln_carry_state,
    ln_carry_state_from_open_starts,
    replay_state_tensors,
)
from .vocab import KEY_COUNT, LaneAction, MapperTupleVocab, coerce_lane_action


MAPPER_WRITE_MS = 8000
MAPPER_DENSITY_FRAMES = 400
MAPPER_DENSITY_FRAME_MS = 20


class MapperTokenizationError(ValueError):
    pass


class UnsupportedMapperActionError(MapperTokenizationError):
    pass


@dataclass(frozen=True)
class MapperTimepoint:
    time_ms: int
    lane_actions: tuple[LaneAction, LaneAction, LaneAction, LaneAction]


@dataclass(frozen=True)
class TokenizedMapperWindow:
    target_fragment_ids: list[int]
    decoder_input_ids: list[int]
    write_start_ms: int
    write_end_ms: int
    is_full_chart_start: bool
    is_full_chart_end: bool
    ln_carry_in: LNCarryState
    ln_carry_out: LNCarryState
    target_fragment_current_ms: torch.Tensor
    target_fragment_open_mask: torch.Tensor
    target_fragment_open_start_ms: torch.Tensor
    target_fragment_open_age_ms: torch.Tensor
    close_labels: torch.Tensor
    close_label_mask: torch.Tensor

    @property
    def seq_len(self) -> int:
        return len(self.target_fragment_ids)

    def target_fragment_tensor(self) -> torch.Tensor:
        return torch.tensor(self.target_fragment_ids, dtype=torch.long)

    def decoder_input_tensor(self) -> torch.Tensor:
        return torch.tensor(self.decoder_input_ids, dtype=torch.long)


def encode_full_chart_tokens(
    timepoints: Sequence[MapperTimepoint | Any],
    *,
    vocab: MapperTupleVocab,
    chart_start_ms: int = 0,
    chart_end_ms: int | None = None,
) -> list[int]:
    chart_start_ms = int(chart_start_ms)
    grouped = _group_timepoints(timepoints)
    if chart_end_ms is None:
        chart_end_ms = max([chart_start_ms, *(timepoint.time_ms for timepoint in grouped)])
    chart_end_ms = int(chart_end_ms)
    if chart_end_ms < chart_start_ms:
        raise ValueError(f"chart_end_ms must be at or after chart_start_ms: {chart_start_ms}..{chart_end_ms}")

    token_ids = [vocab.bos_id]
    current_ms = chart_start_ms
    for timepoint in grouped:
        if timepoint.time_ms < chart_start_ms:
            raise MapperTokenizationError(f"timepoint before chart_start_ms: {timepoint}")
        if timepoint.time_ms > chart_end_ms:
            raise MapperTokenizationError(f"timepoint after chart_end_ms: {timepoint}")
        _require_10ms_grid(timepoint.time_ms)
        delta_ms = timepoint.time_ms - current_ms
        if delta_ms < 0:
            raise MapperTokenizationError(f"timepoints must be nondecreasing after grouping: {grouped}")
        token_ids.extend(vocab.time_shift_token_id(value) for value in vocab.decompose_time_shift_delta(delta_ms))
        token_ids.append(vocab.encode_event(timepoint.lane_actions))
        current_ms = timepoint.time_ms

    token_ids.extend(vocab.time_shift_token_id(value) for value in vocab.decompose_time_shift_delta(chart_end_ms - current_ms))
    token_ids.append(vocab.eos_id)
    return token_ids


def encode_mapper_window(
    timepoints: Sequence[MapperTimepoint | Any],
    *,
    vocab: MapperTupleVocab,
    write_start_ms: int,
    write_end_ms: int,
    chart_start_ms: int = 0,
    chart_end_ms: int | None = None,
) -> TokenizedMapperWindow:
    write_start_ms = int(write_start_ms)
    write_end_ms = int(write_end_ms)
    chart_start_ms = int(chart_start_ms)
    if write_end_ms <= write_start_ms:
        raise ValueError(f"write_end_ms must be after write_start_ms: {write_start_ms}..{write_end_ms}")
    if (write_end_ms - write_start_ms) % 10 != 0:
        raise ValueError("mapper write window must align to the 10ms grid")
    if write_start_ms < chart_start_ms:
        raise ValueError(f"write_start_ms must be at or after chart_start_ms: {write_start_ms} < {chart_start_ms}")
    if chart_end_ms is not None and int(chart_end_ms) < write_end_ms:
        raise ValueError(f"chart_end_ms must cover the write window: {chart_end_ms} < {write_end_ms}")

    grouped = _group_timepoints(timepoints)
    ln_carry_in = ln_carry_state_at(grouped, write_start_ms)
    ln_carry_out = ln_carry_state_at(grouped, write_end_ms)
    is_full_chart_start = write_start_ms == chart_start_ms
    is_full_chart_end = chart_end_ms is not None and write_end_ms == int(chart_end_ms)
    if is_full_chart_start and ln_carry_in != empty_ln_carry_state(write_start_ms):
        raise MapperTokenizationError("full-chart start requires empty ln_carry_in")
    if is_full_chart_end and any(ln_carry_out.open_mask):
        raise MapperTokenizationError("full-chart end requires all long notes closed")

    fragment_timepoints = _fragment_timepoints(
        grouped,
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
    )
    target_fragment_ids = _encode_fragment_tokens(
        fragment_timepoints,
        vocab=vocab,
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
    )
    if is_full_chart_end:
        target_fragment_ids.append(vocab.eos_id)

    first_decoder_input = (
        vocab.bos_id
        if is_full_chart_start
        else final_full_chart_token_before(
            grouped,
            vocab=vocab,
            chart_start_ms=chart_start_ms,
            boundary_ms=write_start_ms,
        )
    )
    decoder_input_ids = [first_decoder_input, *target_fragment_ids[:-1]]
    if len(decoder_input_ids) != len(target_fragment_ids):
        raise MapperTokenizationError("decoder input and target fragment lengths must match")

    try:
        state_tensors = replay_state_tensors(
            target_fragment_ids,
            vocab=vocab,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            ln_carry_in=ln_carry_in,
            ln_carry_out=ln_carry_out,
            is_full_chart_end=is_full_chart_end,
        )
        close_labels, close_label_mask = close_labels_from_tokens(
            target_fragment_ids,
            vocab=vocab,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            ln_carry_in=ln_carry_in,
            ln_carry_out=ln_carry_out,
            is_full_chart_end=is_full_chart_end,
        )
    except ReplayError as exc:
        raise MapperTokenizationError(str(exc)) from exc

    return TokenizedMapperWindow(
        target_fragment_ids=target_fragment_ids,
        decoder_input_ids=decoder_input_ids,
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
        is_full_chart_start=is_full_chart_start,
        is_full_chart_end=is_full_chart_end,
        ln_carry_in=ln_carry_in,
        ln_carry_out=ln_carry_out,
        target_fragment_current_ms=state_tensors["current_ms"],
        target_fragment_open_mask=state_tensors["open_mask"],
        target_fragment_open_start_ms=state_tensors["open_start_ms"],
        target_fragment_open_age_ms=state_tensors["open_age_ms"],
        close_labels=close_labels,
        close_label_mask=close_label_mask,
    )


def ln_carry_state_at(timepoints: Sequence[MapperTimepoint | Any], boundary_ms: int) -> LNCarryState:
    boundary_ms = int(boundary_ms)
    open_start_ms: list[int | None] = [None] * KEY_COUNT
    for timepoint in _group_timepoints(timepoints):
        if timepoint.time_ms >= boundary_ms:
            break
        for lane, action in enumerate(timepoint.lane_actions):
            if action == LaneAction.HOLD_START:
                if open_start_ms[lane] is not None:
                    raise MapperTokenizationError(f"HOLD_START on open lane {lane} at {timepoint.time_ms}ms")
                open_start_ms[lane] = timepoint.time_ms
            elif action == LaneAction.HOLD_END:
                if open_start_ms[lane] is None:
                    raise MapperTokenizationError(f"HOLD_END on closed lane {lane} at {timepoint.time_ms}ms")
                open_start_ms[lane] = None
            elif action == LaneAction.TAP and open_start_ms[lane] is not None:
                raise MapperTokenizationError(f"TAP on open lane {lane} at {timepoint.time_ms}ms")
    return ln_carry_state_from_open_starts(boundary_ms, open_start_ms)


def final_full_chart_token_before(
    timepoints: Sequence[MapperTimepoint | Any],
    *,
    vocab: MapperTupleVocab,
    chart_start_ms: int,
    boundary_ms: int,
) -> int:
    chart_start_ms = int(chart_start_ms)
    boundary_ms = int(boundary_ms)
    if boundary_ms < chart_start_ms:
        raise ValueError(f"boundary_ms must be at or after chart_start_ms: {boundary_ms} < {chart_start_ms}")
    if boundary_ms == chart_start_ms:
        return vocab.bos_id

    current_ms = chart_start_ms
    final_token = vocab.bos_id
    for timepoint in _group_timepoints(timepoints):
        if timepoint.time_ms < chart_start_ms:
            raise MapperTokenizationError(f"timepoint before chart_start_ms: {timepoint}")
        if timepoint.time_ms >= boundary_ms:
            break
        _require_10ms_grid(timepoint.time_ms)
        delta_ms = timepoint.time_ms - current_ms
        if delta_ms < 0:
            raise MapperTokenizationError(f"timepoints must be nondecreasing: {timepoints}")
        for value in vocab.decompose_time_shift_delta(delta_ms):
            final_token = vocab.time_shift_token_id(value)
        final_token = vocab.encode_event(timepoint.lane_actions)
        current_ms = timepoint.time_ms

    for value in vocab.decompose_time_shift_delta(boundary_ms - current_ms):
        final_token = vocab.time_shift_token_id(value)
    return final_token


def window_timepoints(
    timepoints: Sequence[MapperTimepoint | Any],
    *,
    write_start_ms: int,
    write_end_ms: int,
) -> list[MapperTimepoint]:
    return [
        timepoint
        for timepoint in _group_timepoints(timepoints)
        if int(write_start_ms) <= timepoint.time_ms < int(write_end_ms)
    ]


def hitobjects_to_mapper_timepoints(hitobjects: Sequence[Any], *, key_count: int = KEY_COUNT) -> list[MapperTimepoint]:
    if key_count != KEY_COUNT:
        raise ValueError(f"mapper tuple supports only {KEY_COUNT}K, got {key_count}")

    primitive_actions: dict[int, dict[int, list[LaneAction]]] = defaultdict(lambda: defaultdict(list))
    for hitobject in hitobjects:
        lane = int(hitobject.lane)
        if not 0 <= lane < KEY_COUNT:
            raise ValueError(f"hit object lane outside 4K range: {lane}")
        start_ms = quantize_10ms_half_up(float(hitobject.start_time_ms))
        kind = getattr(getattr(hitobject, "kind", None), "value", getattr(hitobject, "kind", None))
        if kind == "TAP":
            primitive_actions[start_ms][lane].append(LaneAction.TAP)
            continue
        if kind != "HOLD":
            raise ValueError(f"unsupported mapper tuple hit object kind: {kind}")
        end_ms = quantize_10ms_half_up(float(hitobject.end_time_ms))
        if end_ms <= start_ms:
            primitive_actions[start_ms][lane].append(LaneAction.TAP)
            continue
        primitive_actions[start_ms][lane].append(LaneAction.HOLD_START)
        primitive_actions[end_ms][lane].append(LaneAction.HOLD_END)

    timepoints: list[MapperTimepoint] = []
    for time_ms in sorted(primitive_actions):
        lane_actions = tuple(
            _merge_lane_primitive_actions(primitive_actions[time_ms].get(lane, []), time_ms=time_ms, lane=lane)
            for lane in range(KEY_COUNT)
        )
        if any(action != LaneAction.NONE for action in lane_actions):
            timepoints.append(MapperTimepoint(time_ms=time_ms, lane_actions=lane_actions))  # type: ignore[arg-type]
    return timepoints


def tokenize_hitobjects_window(
    hitobjects: Sequence[Any],
    *,
    vocab: MapperTupleVocab,
    write_start_ms: int,
    write_end_ms: int,
    chart_start_ms: int = 0,
    chart_end_ms: int | None = None,
) -> TokenizedMapperWindow:
    return encode_mapper_window(
        hitobjects_to_mapper_timepoints(hitobjects),
        vocab=vocab,
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
        chart_start_ms=chart_start_ms,
        chart_end_ms=chart_end_ms,
    )


def quantize_10ms_half_up(time_ms: float) -> int:
    if time_ms < 0:
        raise ValueError(f"cannot quantize negative time: {time_ms}")
    return int(10 * math.floor((time_ms + 5) / 10))


def _encode_fragment_tokens(
    timepoints: Sequence[MapperTimepoint],
    *,
    vocab: MapperTupleVocab,
    write_start_ms: int,
    write_end_ms: int,
) -> list[int]:
    token_ids: list[int] = []
    current_ms = int(write_start_ms)
    for timepoint in timepoints:
        if not int(write_start_ms) <= timepoint.time_ms < int(write_end_ms):
            raise MapperTokenizationError(f"timepoint outside write window: {timepoint}")
        _require_10ms_grid(timepoint.time_ms)
        delta_ms = timepoint.time_ms - current_ms
        if delta_ms < 0:
            raise MapperTokenizationError(f"timepoints must be nondecreasing after grouping: {timepoints}")
        token_ids.extend(vocab.time_shift_token_id(value) for value in vocab.decompose_time_shift_delta(delta_ms))
        token_ids.append(vocab.encode_event(timepoint.lane_actions))
        current_ms = timepoint.time_ms

    token_ids.extend(vocab.time_shift_token_id(value) for value in vocab.decompose_time_shift_delta(int(write_end_ms) - current_ms))
    return token_ids


def _fragment_timepoints(
    timepoints: Sequence[MapperTimepoint],
    *,
    write_start_ms: int,
    write_end_ms: int,
) -> list[MapperTimepoint]:
    fragment: list[MapperTimepoint] = []
    for timepoint in timepoints:
        if timepoint.time_ms < int(write_start_ms):
            continue
        if timepoint.time_ms >= int(write_end_ms):
            break
        fragment.append(timepoint)
    return fragment


def _group_timepoints(timepoints: Sequence[MapperTimepoint | Any]) -> list[MapperTimepoint]:
    grouped: dict[int, list[LaneAction]] = {}
    for raw_timepoint in sorted((_coerce_timepoint(item) for item in timepoints), key=lambda item: item.time_ms):
        actions = grouped.setdefault(raw_timepoint.time_ms, [LaneAction.NONE] * KEY_COUNT)
        for lane, action in enumerate(raw_timepoint.lane_actions):
            if action == LaneAction.NONE:
                continue
            if actions[lane] != LaneAction.NONE:
                raise UnsupportedMapperActionError(
                    f"multiple same-lane actions at {raw_timepoint.time_ms}ms lane {lane}",
                )
            actions[lane] = action
    return [
        MapperTimepoint(time_ms=time_ms, lane_actions=tuple(actions))  # type: ignore[arg-type]
        for time_ms, actions in sorted(grouped.items())
        if any(action != LaneAction.NONE for action in actions)
    ]


def _coerce_timepoint(timepoint: MapperTimepoint | Any) -> MapperTimepoint:
    if isinstance(timepoint, MapperTimepoint):
        return timepoint
    if not hasattr(timepoint, "time_ms") or not hasattr(timepoint, "lane_actions"):
        raise TypeError(f"mapper timepoint must expose time_ms and lane_actions: {timepoint!r}")
    actions = tuple(coerce_lane_action(action) for action in timepoint.lane_actions)
    if len(actions) != KEY_COUNT:
        raise ValueError(f"mapper timepoint must contain {KEY_COUNT} lane actions: {actions}")
    return MapperTimepoint(time_ms=int(timepoint.time_ms), lane_actions=actions)  # type: ignore[arg-type]


def _merge_lane_primitive_actions(actions: Sequence[LaneAction], *, time_ms: int, lane: int) -> LaneAction:
    if not actions:
        return LaneAction.NONE
    if len(actions) == 1:
        return actions[0]
    raise UnsupportedMapperActionError(
        f"mapper tuple cannot represent multiple same-lane actions at {time_ms}ms lane {lane}: {list(actions)}",
    )


def _require_10ms_grid(time_ms: int) -> None:
    if int(time_ms) % 10 != 0:
        raise MapperTokenizationError(f"timepoint must be on the 10ms grid: {time_ms}")
