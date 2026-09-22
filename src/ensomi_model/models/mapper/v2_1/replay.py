from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch

from .vocab import KEY_COUNT, LaneAction, MapperV21Vocab


CLOSED_OPEN_START_MS = -1
NO_EMITTED_LANE_INDEX = -1


class ReplayError(ValueError):
    """Raised when a sparse mapper token sequence violates replay rules."""


@dataclass(frozen=True)
class LNCarryState:
    current_ms: int
    open_mask: tuple[bool, bool, bool, bool]
    open_start_ms: tuple[int | None, int | None, int | None, int | None]
    open_age_ms: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        current_ms = int(self.current_ms)
        if current_ms % 10 != 0:
            raise ValueError(f"current_ms must be 10ms-aligned: {current_ms}")
        open_mask = _coerce_open_mask_tuple(self.open_mask)
        open_start_ms = _coerce_open_start_tuple(self.open_start_ms)
        open_age_ms = _coerce_open_age_tuple(self.open_age_ms)
        for lane, is_open in enumerate(open_mask):
            open_start = open_start_ms[lane]
            open_age = open_age_ms[lane]
            if is_open:
                if open_start is None:
                    raise ValueError(f"open lane {lane} requires open_start_ms")
                if open_start % 10 != 0:
                    raise ValueError(f"open_start_ms[{lane}] must be 10ms-aligned: {open_start}")
                if open_start > current_ms:
                    raise ValueError(f"open_start_ms[{lane}] cannot be after current_ms: {open_start} > {current_ms}")
                expected_age = current_ms - open_start
                if open_age != expected_age:
                    raise ValueError(
                        f"open_age_ms[{lane}] must equal current_ms - open_start_ms: "
                        f"{open_age} != {expected_age}"
                    )
            else:
                if open_start is not None:
                    raise ValueError(f"closed lane {lane} must use open_start_ms=None")
                if open_age != 0:
                    raise ValueError(f"closed lane {lane} must use open_age_ms=0")
        object.__setattr__(self, "current_ms", current_ms)
        object.__setattr__(self, "open_mask", open_mask)
        object.__setattr__(self, "open_start_ms", open_start_ms)
        object.__setattr__(self, "open_age_ms", open_age_ms)

    @property
    def open_mask_bits(self) -> int:
        return open_mask_tuple_to_bits(self.open_mask)

    @classmethod
    def closed(cls, current_ms: int) -> "LNCarryState":
        return empty_ln_carry_state(current_ms)

    @classmethod
    def from_open_starts(cls, current_ms: int, open_start_ms: Sequence[int | None]) -> "LNCarryState":
        return ln_carry_state_from_open_starts(current_ms, open_start_ms)

    def shifted(self, delta_ms: int) -> "LNCarryState":
        return ln_carry_state_from_open_starts(int(self.current_ms) + int(delta_ms), self.open_start_ms)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MapperReplayState:
    position: int
    current_ms: int
    open_mask: tuple[bool, bool, bool, bool]
    open_start_ms: tuple[int | None, int | None, int | None, int | None]
    open_age_ms: tuple[int, int, int, int]
    emitted_lane_mask: tuple[bool, bool, bool, bool] = (False, False, False, False)
    last_lane_index: int = NO_EMITTED_LANE_INDEX

    @property
    def open_mask_bits(self) -> int:
        return open_mask_tuple_to_bits(self.open_mask)

    @property
    def emitted_lane_mask_bits(self) -> int:
        return open_mask_tuple_to_bits(self.emitted_lane_mask)


def empty_ln_carry_state(current_ms: int) -> LNCarryState:
    return LNCarryState(
        current_ms=int(current_ms),
        open_mask=(False, False, False, False),
        open_start_ms=(None, None, None, None),
        open_age_ms=(0, 0, 0, 0),
    )


def initial_replay_state(ln_carry_in: LNCarryState) -> MapperReplayState:
    return MapperReplayState(
        position=-1,
        current_ms=int(ln_carry_in.current_ms),
        open_mask=ln_carry_in.open_mask,
        open_start_ms=ln_carry_in.open_start_ms,
        open_age_ms=ln_carry_in.open_age_ms,
    )


def replay_tokens(
    token_ids: Sequence[int],
    *,
    vocab: MapperV21Vocab,
    write_start_ms: int,
    write_end_ms: int,
    ln_carry_in: LNCarryState,
    ln_carry_out: LNCarryState,
    chart_end_ms: int | None = None,
    is_full_chart_start: bool = False,
    is_full_chart_end: bool = False,
    validate_terminal: bool = False,
) -> list[MapperReplayState]:
    _validate_replay_window(
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
        chart_end_ms=chart_end_ms,
        is_full_chart_end=is_full_chart_end,
        ln_carry_in=ln_carry_in,
        ln_carry_out=ln_carry_out,
    )
    state = initial_replay_state(ln_carry_in)
    states: list[MapperReplayState] = []
    for position, token_id in enumerate(token_ids):
        state = MapperReplayState(
            position=position,
            current_ms=state.current_ms,
            open_mask=state.open_mask,
            open_start_ms=state.open_start_ms,
            open_age_ms=state.open_age_ms,
            emitted_lane_mask=state.emitted_lane_mask,
            last_lane_index=state.last_lane_index,
        )
        states.append(state)
        state = transition_replay_state(
            state,
            int(token_id),
            position=position,
            vocab=vocab,
            write_start_ms=int(write_start_ms),
            write_end_ms=int(write_end_ms),
            chart_end_ms=chart_end_ms,
            ln_carry_out=ln_carry_out,
            is_full_chart_start=bool(is_full_chart_start),
            is_full_chart_end=bool(is_full_chart_end),
        )

    if validate_terminal and not replay_state_matches_carry(state, ln_carry_out):
        raise ReplayError(
            "token sequence terminal state does not match ln_carry_out: "
            f"terminal={format_replay_state(state)} ln_carry_out={ln_carry_out}"
        )
    return states


def replay_terminal_state(
    token_ids: Sequence[int],
    *,
    vocab: MapperV21Vocab,
    write_start_ms: int,
    write_end_ms: int,
    ln_carry_in: LNCarryState,
    ln_carry_out: LNCarryState,
    chart_end_ms: int | None = None,
    is_full_chart_start: bool = False,
    is_full_chart_end: bool = False,
) -> MapperReplayState:
    _validate_replay_window(
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
        chart_end_ms=chart_end_ms,
        is_full_chart_end=is_full_chart_end,
        ln_carry_in=ln_carry_in,
        ln_carry_out=ln_carry_out,
    )
    state = initial_replay_state(ln_carry_in)
    for position, token_id in enumerate(token_ids):
        state = transition_replay_state(
            MapperReplayState(
                position=position,
                current_ms=state.current_ms,
                open_mask=state.open_mask,
                open_start_ms=state.open_start_ms,
                open_age_ms=state.open_age_ms,
                emitted_lane_mask=state.emitted_lane_mask,
                last_lane_index=state.last_lane_index,
            ),
            int(token_id),
            position=position,
            vocab=vocab,
            write_start_ms=int(write_start_ms),
            write_end_ms=int(write_end_ms),
            chart_end_ms=chart_end_ms,
            ln_carry_out=ln_carry_out,
            is_full_chart_start=bool(is_full_chart_start),
            is_full_chart_end=bool(is_full_chart_end),
        )
    return state


def transition_replay_state(
    state: MapperReplayState,
    token_id: int,
    *,
    position: int,
    vocab: MapperV21Vocab,
    write_start_ms: int,
    write_end_ms: int,
    ln_carry_out: LNCarryState,
    chart_end_ms: int | None = None,
    is_full_chart_start: bool = False,
    is_full_chart_end: bool = False,
) -> MapperReplayState:
    token_id = int(token_id)
    target_end_ms = _target_end_ms(
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
        chart_end_ms=chart_end_ms,
        is_full_chart_end=is_full_chart_end,
    )
    if token_id == vocab.pad_id:
        raise ReplayError("PAD is not legal in replay")

    if token_id == vocab.bos_id:
        if not is_full_chart_start:
            raise ReplayError("BOS is legal only at the full-chart start")
        if position != 0 or state.current_ms != int(write_start_ms):
            raise ReplayError("BOS is legal only at full-chart token position 0")
        if any(state.open_mask) or any(start is not None for start in state.open_start_ms) or any(state.open_age_ms):
            raise ReplayError("BOS requires an empty chart-start LN state")
        if any(state.emitted_lane_mask) or state.last_lane_index != NO_EMITTED_LANE_INDEX:
            raise ReplayError("BOS requires an empty same-time lane state")
        return _copy_state_at_position(state, position=position)

    if token_id == vocab.eos_id:
        if not is_full_chart_end:
            raise ReplayError("EOS is legal only at the full-chart end")
        if state.current_ms != int(target_end_ms):
            raise ReplayError(f"EOS requires current_ms == chart_end_ms: {state.current_ms} != {target_end_ms}")
        if not replay_state_matches_carry(state, ln_carry_out):
            raise ReplayError("EOS requires current LN state to equal ln_carry_out")
        return _copy_state_at_position(state, position=position)

    if vocab.is_time_shift_token(token_id):
        delta_ms = vocab.time_shift_value(token_id)
        next_ms = state.current_ms + delta_ms
        if next_ms > int(target_end_ms):
            raise ReplayError(f"TIME_SHIFT moves past target_end_ms: {next_ms} > {target_end_ms}")
        next_age = tuple(
            int(next_ms - open_start) if is_open and open_start is not None else 0
            for is_open, open_start in zip(state.open_mask, state.open_start_ms, strict=True)
        )
        next_state = MapperReplayState(
            position=position,
            current_ms=next_ms,
            open_mask=state.open_mask,
            open_start_ms=state.open_start_ms,
            open_age_ms=next_age,  # type: ignore[arg-type]
            emitted_lane_mask=(False, False, False, False),
            last_lane_index=NO_EMITTED_LANE_INDEX,
        )
        if not is_full_chart_end and next_ms == int(target_end_ms) and not replay_state_matches_carry(next_state, ln_carry_out):
            raise ReplayError("TIME_SHIFT to write_end_ms requires resulting state to equal ln_carry_out")
        return next_state

    if vocab.is_lane_action_token(token_id):
        if state.current_ms > int(target_end_ms) or (state.current_ms == int(target_end_ms) and not is_full_chart_end):
            raise ReplayError("lane-action token is illegal after the target end")
        lane, action = vocab.decode_lane_action(token_id)
        if lane <= int(state.last_lane_index) or state.emitted_lane_mask[lane]:
            raise ReplayError(
                "same-time lane-action tokens must use strictly ascending lane order "
                "with no duplicate lane before the next TS_* token"
            )

        is_open = state.open_mask[lane]
        if is_open and action != LaneAction.HOLD_END:
            raise ReplayError(f"{action.value} is illegal on open lane {lane}")
        if not is_open and action == LaneAction.HOLD_END:
            raise ReplayError(f"HOLD_END is illegal on closed lane {lane}")

        next_open = list(state.open_mask)
        next_start = list(state.open_start_ms)
        next_age = list(state.open_age_ms)
        if action == LaneAction.HOLD_START:
            next_open[lane] = True
            next_start[lane] = state.current_ms
            next_age[lane] = 0
        elif action == LaneAction.HOLD_END:
            next_open[lane] = False
            next_start[lane] = None
            next_age[lane] = 0
        elif action == LaneAction.TAP:
            next_age[lane] = 0

        next_emitted = list(state.emitted_lane_mask)
        next_emitted[lane] = True
        return MapperReplayState(
            position=position,
            current_ms=state.current_ms,
            open_mask=tuple(next_open),  # type: ignore[arg-type]
            open_start_ms=tuple(next_start),  # type: ignore[arg-type]
            open_age_ms=tuple(int(value) for value in next_age),  # type: ignore[arg-type]
            emitted_lane_mask=tuple(next_emitted),  # type: ignore[arg-type]
            last_lane_index=lane,
        )

    raise ReplayError(f"unknown mapper v2.1 token id: {token_id}")


def replay_state_tensors(
    token_ids: Sequence[int],
    *,
    vocab: MapperV21Vocab,
    write_start_ms: int,
    write_end_ms: int,
    ln_carry_in: LNCarryState,
    ln_carry_out: LNCarryState,
    chart_end_ms: int | None = None,
    is_full_chart_start: bool = False,
    is_full_chart_end: bool = False,
) -> dict[str, torch.Tensor]:
    states = replay_tokens(
        token_ids,
        vocab=vocab,
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
        chart_end_ms=chart_end_ms,
        ln_carry_in=ln_carry_in,
        ln_carry_out=ln_carry_out,
        is_full_chart_start=is_full_chart_start,
        is_full_chart_end=is_full_chart_end,
        validate_terminal=True,
    )
    return {
        "current_ms": torch.tensor([state.current_ms for state in states], dtype=torch.long),
        "open_mask": torch.tensor([state.open_mask for state in states], dtype=torch.bool),
        "open_start_ms": torch.tensor(
            [open_start_tuple_to_tensor_values(state.open_start_ms) for state in states],
            dtype=torch.long,
        ),
        "open_age_ms": torch.tensor([state.open_age_ms for state in states], dtype=torch.long),
        "emitted_lane_mask": torch.tensor([state.emitted_lane_mask for state in states], dtype=torch.bool),
        "last_lane_index": torch.tensor([state.last_lane_index for state in states], dtype=torch.long),
    }


def close_labels_from_tokens(
    token_ids: Sequence[int],
    *,
    vocab: MapperV21Vocab,
    write_start_ms: int,
    write_end_ms: int,
    ln_carry_in: LNCarryState,
    ln_carry_out: LNCarryState,
    chart_end_ms: int | None = None,
    is_full_chart_end: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    states = replay_tokens(
        token_ids,
        vocab=vocab,
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
        chart_end_ms=chart_end_ms,
        ln_carry_in=ln_carry_in,
        ln_carry_out=ln_carry_out,
        is_full_chart_end=is_full_chart_end,
        validate_terminal=True,
    )
    labels = torch.zeros((len(token_ids), KEY_COUNT), dtype=torch.bool)
    mask = torch.zeros((len(token_ids), KEY_COUNT), dtype=torch.bool)
    for index, state in enumerate(states):
        for lane, is_open in enumerate(state.open_mask):
            mask[index, lane] = is_open
        token_id = int(token_ids[index])
        if not vocab.is_lane_action_token(token_id):
            continue
        lane, action = vocab.decode_lane_action(token_id)
        labels[index, lane] = state.open_mask[lane] and action == LaneAction.HOLD_END
    return labels, mask


def replay_state_matches_carry(state: MapperReplayState, carry: LNCarryState) -> bool:
    return (
        int(state.current_ms) == int(carry.current_ms)
        and tuple(state.open_mask) == tuple(carry.open_mask)
        and tuple(state.open_start_ms) == tuple(carry.open_start_ms)
        and tuple(int(value) for value in state.open_age_ms) == tuple(int(value) for value in carry.open_age_ms)
    )


def format_replay_state(state: MapperReplayState) -> str:
    return (
        "MapperReplayState("
        f"position={state.position}, current_ms={state.current_ms}, "
        f"open_mask={state.open_mask}, open_start_ms={state.open_start_ms}, "
        f"open_age_ms={state.open_age_ms}, emitted_lane_mask={state.emitted_lane_mask}, "
        f"last_lane_index={state.last_lane_index})"
    )


def ln_carry_state_tensors(carry: LNCarryState) -> dict[str, torch.Tensor]:
    return {
        "current_ms": torch.tensor(carry.current_ms, dtype=torch.long),
        "open_mask": torch.tensor(carry.open_mask, dtype=torch.bool),
        "open_start_ms": torch.tensor(open_start_tuple_to_tensor_values(carry.open_start_ms), dtype=torch.long),
        "open_age_ms": torch.tensor(carry.open_age_ms, dtype=torch.long),
    }


def ln_carry_state_from_open_starts(current_ms: int, open_start_ms: Sequence[int | None]) -> LNCarryState:
    starts = _coerce_open_start_tuple(open_start_ms)
    open_mask = tuple(start is not None for start in starts)
    open_age_ms = tuple(int(current_ms) - int(start) if start is not None else 0 for start in starts)
    return LNCarryState(
        current_ms=int(current_ms),
        open_mask=open_mask,  # type: ignore[arg-type]
        open_start_ms=starts,
        open_age_ms=open_age_ms,  # type: ignore[arg-type]
    )


def open_start_tuple_to_tensor_values(
    open_start_ms: Sequence[int | None],
    *,
    closed_value: int = CLOSED_OPEN_START_MS,
) -> tuple[int, int, int, int]:
    starts = _coerce_open_start_tuple(open_start_ms)
    return tuple(int(closed_value) if value is None else int(value) for value in starts)  # type: ignore[return-value]


def open_start_tensor_values_to_tuple(
    open_start_ms: Sequence[int],
    *,
    closed_value: int = CLOSED_OPEN_START_MS,
) -> tuple[int | None, int | None, int | None, int | None]:
    values = list(open_start_ms)
    if len(values) != KEY_COUNT:
        raise ValueError(f"open_start_ms must contain {KEY_COUNT} lanes: {open_start_ms}")
    return tuple(None if int(value) == int(closed_value) else int(value) for value in values)  # type: ignore[return-value]


def open_mask_tuple_to_bits(open_mask: Sequence[bool]) -> int:
    if len(open_mask) != KEY_COUNT:
        raise ValueError(f"open mask must contain {KEY_COUNT} lanes: {open_mask}")
    bits = 0
    for lane, is_open in enumerate(open_mask):
        if bool(is_open):
            bits |= 1 << lane
    return bits


def _copy_state_at_position(state: MapperReplayState, *, position: int) -> MapperReplayState:
    return MapperReplayState(
        position=position,
        current_ms=state.current_ms,
        open_mask=state.open_mask,
        open_start_ms=state.open_start_ms,
        open_age_ms=state.open_age_ms,
        emitted_lane_mask=state.emitted_lane_mask,
        last_lane_index=state.last_lane_index,
    )


def _validate_replay_window(
    *,
    write_start_ms: int,
    write_end_ms: int,
    chart_end_ms: int | None,
    is_full_chart_end: bool,
    ln_carry_in: LNCarryState,
    ln_carry_out: LNCarryState,
) -> None:
    if int(write_end_ms) <= int(write_start_ms):
        raise ValueError(f"write_end_ms must be after write_start_ms: {write_start_ms}..{write_end_ms}")
    if int(write_start_ms) % 10 != 0 or int(write_end_ms) % 10 != 0:
        raise ValueError(f"write window must be 10ms-aligned: {write_start_ms}..{write_end_ms}")
    if int(ln_carry_in.current_ms) != int(write_start_ms):
        raise ValueError(f"ln_carry_in.current_ms must equal write_start_ms: {ln_carry_in.current_ms} != {write_start_ms}")
    target_end_ms = _target_end_ms(
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
        chart_end_ms=chart_end_ms,
        is_full_chart_end=is_full_chart_end,
    )
    if int(ln_carry_out.current_ms) != int(target_end_ms):
        raise ValueError(f"ln_carry_out.current_ms must equal target_end_ms: {ln_carry_out.current_ms} != {target_end_ms}")


def _target_end_ms(
    *,
    write_start_ms: int,
    write_end_ms: int,
    chart_end_ms: int | None,
    is_full_chart_end: bool,
) -> int:
    if not bool(is_full_chart_end):
        return int(write_end_ms)
    resolved = int(write_end_ms) if chart_end_ms is None else int(chart_end_ms)
    if resolved < int(write_start_ms) or resolved > int(write_end_ms):
        raise ValueError(f"chart_end_ms must be inside the write window: {resolved} not in {write_start_ms}..{write_end_ms}")
    if resolved % 10 != 0:
        raise ValueError(f"chart_end_ms must be 10ms-aligned: {resolved}")
    return resolved


def _coerce_open_mask_tuple(open_mask: Sequence[bool]) -> tuple[bool, bool, bool, bool]:
    if len(open_mask) != KEY_COUNT:
        raise ValueError(f"open_mask must contain {KEY_COUNT} lanes: {open_mask}")
    return tuple(bool(value) for value in open_mask)  # type: ignore[return-value]


def _coerce_open_start_tuple(
    open_start_ms: Sequence[int | None],
) -> tuple[int | None, int | None, int | None, int | None]:
    if len(open_start_ms) != KEY_COUNT:
        raise ValueError(f"open_start_ms must contain {KEY_COUNT} lanes: {open_start_ms}")
    return tuple(None if value is None else int(value) for value in open_start_ms)  # type: ignore[return-value]


def _coerce_open_age_tuple(open_age_ms: Sequence[int]) -> tuple[int, int, int, int]:
    if len(open_age_ms) != KEY_COUNT:
        raise ValueError(f"open_age_ms must contain {KEY_COUNT} lanes: {open_age_ms}")
    return tuple(int(value) for value in open_age_ms)  # type: ignore[return-value]
