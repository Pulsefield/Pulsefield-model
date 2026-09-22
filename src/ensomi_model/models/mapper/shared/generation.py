from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from .generation_engine import (
    apply_top_p,
    default_generation_logits,
    run_generation_engine,
    select_next_token,
)
from .replay import LNCarryState
from .vocab import LaneAction, MapperTupleVocab


class CarryStateError(ValueError):
    pass


class MapperGenerationError(ValueError):
    pass


@dataclass(frozen=True)
class MapperGenerationStep:
    decoder_input_tokens: torch.Tensor
    generated_tokens: tuple[int, ...]
    state: LNCarryState
    valid_token_mask: torch.Tensor
    token_index: int
    write_start_ms: int
    write_end_ms: int
    ln_carry_in: LNCarryState
    ln_carry_out: LNCarryState


@dataclass(frozen=True)
class MapperGeneratedWindow:
    write_start_ms: int
    write_end_ms: int
    ln_carry_in: LNCarryState
    ln_carry_out: LNCarryState
    tokens: list[int]
    states_before: list[LNCarryState]
    states_after: list[LNCarryState]
    terminal_state: LNCarryState
    completed: bool
    dead_end: bool
    max_tokens_exceeded: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "write_start_ms": self.write_start_ms,
            "write_end_ms": self.write_end_ms,
            "ln_carry_in": self.ln_carry_in.to_dict(),
            "ln_carry_out": self.ln_carry_out.to_dict(),
            "tokens": list(self.tokens),
            "states_before": [state.to_dict() for state in self.states_before],
            "states_after": [state.to_dict() for state in self.states_after],
            "terminal_state": self.terminal_state.to_dict(),
            "completed": self.completed,
            "dead_end": self.dead_end,
            "max_tokens_exceeded": self.max_tokens_exceeded,
        }


LogitsFn = Callable[[MapperGenerationStep], torch.Tensor]


def carry_states_equal(
    left: LNCarryState,
    right: LNCarryState,
    *,
    age_tolerance_ms: int = 0,
    compare_open_start: bool = True,
) -> bool:
    if int(left.current_ms) != int(right.current_ms):
        return False
    if tuple(left.open_mask) != tuple(right.open_mask):
        return False
    if compare_open_start and tuple(left.open_start_ms) != tuple(right.open_start_ms):
        return False
    tolerance = int(age_tolerance_ms)
    for lane, is_open in enumerate(left.open_mask):
        if not is_open:
            if left.open_age_ms[lane] != right.open_age_ms[lane]:
                return False
            continue
        if abs(int(left.open_age_ms[lane]) - int(right.open_age_ms[lane])) > tolerance:
            return False
    return True


def window_is_complete(
    state: LNCarryState,
    *,
    write_end_ms: int,
    ln_carry_out: LNCarryState,
) -> bool:
    return int(state.current_ms) == int(write_end_ms) and carry_states_equal(state, ln_carry_out)


def carry_aware_valid_token_mask(
    *,
    state: LNCarryState,
    write_start_ms: int,
    write_end_ms: int,
    ln_carry_out: LNCarryState,
    vocab: MapperTupleVocab,
    is_full_chart_start: bool = False,
    is_full_chart_end: bool = False,
    token_position: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    resolved_device = torch.device("cpu") if device is None else device
    mask = torch.zeros(vocab.size, dtype=torch.bool, device=resolved_device)
    if state.current_ms < int(write_start_ms) or state.current_ms > int(write_end_ms):
        return mask

    if bool(is_full_chart_start) and int(token_position) == 0 and state.current_ms == int(write_start_ms):
        if not any(state.open_mask):
            mask[vocab.bos_id] = True

    if bool(is_full_chart_end) and window_is_complete(state, write_end_ms=write_end_ms, ln_carry_out=ln_carry_out):
        mask[vocab.eos_id] = True

    if window_is_complete(state, write_end_ms=write_end_ms, ln_carry_out=ln_carry_out):
        return mask

    for token_id in vocab.time_shift_token_ids:
        delta_ms = vocab.time_shift_value(token_id)
        next_ms = state.current_ms + delta_ms
        if next_ms > int(write_end_ms):
            continue
        next_state = state.shifted(delta_ms)
        if next_ms == int(write_end_ms) and not carry_states_equal(next_state, ln_carry_out):
            continue
        mask[token_id] = True

    if state.current_ms >= int(write_end_ms):
        return mask
    for token_id in vocab.event_token_ids:
        try:
            transition_carry_state(
                state,
                token_id,
                vocab=vocab,
                write_start_ms=write_start_ms,
                write_end_ms=write_end_ms,
                allow_bos=False,
                allow_eos=False,
            )
        except CarryStateError:
            continue
        mask[token_id] = True
    return mask


def transition_carry_state(
    state: LNCarryState,
    token_id: int,
    *,
    vocab: MapperTupleVocab,
    write_start_ms: int,
    write_end_ms: int,
    allow_bos: bool = False,
    allow_eos: bool = False,
) -> LNCarryState:
    token = int(token_id)
    if token == vocab.pad_id:
        raise CarryStateError("PAD is not legal in mapper replay")
    if token == vocab.bos_id:
        if not allow_bos:
            raise CarryStateError("BOS is not legal inside an ordinary mapper window")
        if state.current_ms != int(write_start_ms) or any(state.open_mask):
            raise CarryStateError("BOS requires the full-chart start state")
        return state
    if token == vocab.eos_id:
        if not allow_eos:
            raise CarryStateError("EOS is not legal for ordinary mapper window completion")
        if state.current_ms != int(write_end_ms):
            raise CarryStateError("EOS requires current_ms == write_end_ms")
        return state
    if vocab.is_time_shift_token(token):
        next_ms = state.current_ms + vocab.time_shift_value(token)
        if next_ms > int(write_end_ms):
            raise CarryStateError(f"TIME_SHIFT moves past write_end_ms: {next_ms} > {write_end_ms}")
        return state.shifted(vocab.time_shift_value(token))
    if vocab.is_event_token(token):
        if state.current_ms >= int(write_end_ms):
            raise CarryStateError("EVENT is illegal at or after write_end_ms")
        open_starts = list(state.open_start_ms)
        actions = vocab.decode_event(token)
        for lane, action in enumerate(actions):
            is_open = state.open_mask[lane]
            if is_open and action not in {LaneAction.NONE, LaneAction.HOLD_END}:
                raise CarryStateError(f"{action.value} is illegal on open lane {lane}")
            if not is_open and action == LaneAction.HOLD_END:
                raise CarryStateError(f"HOLD_END is illegal on closed lane {lane}")
            if action == LaneAction.TAP and is_open:
                raise CarryStateError(f"TAP is illegal on open lane {lane}")
            if action == LaneAction.HOLD_START:
                open_starts[lane] = state.current_ms
            elif action == LaneAction.HOLD_END:
                open_starts[lane] = None
        return LNCarryState.from_open_starts(state.current_ms, open_starts)
    raise CarryStateError(f"unknown mapper tuple token id: {token}")


def decoder_input_tokens_for_generation(
    *,
    vocab: MapperTupleVocab,
    left_context_tokens: Sequence[int] = (),
    generated_tokens: Sequence[int] = (),
    is_full_chart_start: bool = False,
) -> list[int]:
    context = [int(token_id) for token_id in left_context_tokens]
    if not context:
        if bool(is_full_chart_start):
            context = [vocab.bos_id]
        else:
            raise MapperGenerationError("non-initial mapper generation requires a real left_context_tokens value")
    return [*context, *(int(token_id) for token_id in generated_tokens)]


def grammar_constrained_window_generation(
    *,
    vocab: MapperTupleVocab,
    write_start_ms: int,
    write_end_ms: int,
    ln_carry_in: LNCarryState,
    ln_carry_out: LNCarryState,
    logits_fn: LogitsFn | None = None,
    left_context_tokens: Sequence[int] = (),
    is_full_chart_start: bool = False,
    is_full_chart_end: bool = False,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> MapperGeneratedWindow:
    if ln_carry_in.current_ms != int(write_start_ms):
        raise ValueError("ln_carry_in.current_ms must equal write_start_ms")
    if ln_carry_out.current_ms != int(write_end_ms):
        raise ValueError("ln_carry_out.current_ms must equal write_end_ms")

    def is_complete(state: LNCarryState) -> bool:
        return window_is_complete(state, write_end_ms=write_end_ms, ln_carry_out=ln_carry_out)

    def decoder_inputs(generated_tokens: tuple[int, ...]) -> list[int]:
        return decoder_input_tokens_for_generation(
            vocab=vocab,
            left_context_tokens=left_context_tokens,
            generated_tokens=generated_tokens,
            is_full_chart_start=is_full_chart_start,
        )

    def ordinary_mask(state: LNCarryState, generated_tokens: tuple[int, ...]) -> torch.Tensor:
        return carry_aware_valid_token_mask(
            state=state,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            ln_carry_out=ln_carry_out,
            vocab=vocab,
            is_full_chart_start=is_full_chart_start,
            is_full_chart_end=is_full_chart_end,
            token_position=len(generated_tokens),
        )

    def completion_mask(state: LNCarryState, generated_tokens: tuple[int, ...]) -> torch.Tensor | None:
        if not bool(is_full_chart_end) or (generated_tokens and generated_tokens[-1] == vocab.eos_id):
            return None
        valid_mask = ordinary_mask(state, generated_tokens)
        eos_only_mask = torch.zeros_like(valid_mask)
        if bool(valid_mask[vocab.eos_id].item()):
            eos_only_mask[vocab.eos_id] = True
        return eos_only_mask

    def make_step(
        decoder_input_tensor: torch.Tensor,
        generated_tokens: tuple[int, ...],
        state: LNCarryState,
        valid_mask: torch.Tensor,
        token_index: int,
    ) -> MapperGenerationStep:
        return MapperGenerationStep(
            decoder_input_tokens=decoder_input_tensor,
            generated_tokens=generated_tokens,
            state=state,
            valid_token_mask=valid_mask,
            token_index=token_index,
            write_start_ms=int(write_start_ms),
            write_end_ms=int(write_end_ms),
            ln_carry_in=ln_carry_in,
            ln_carry_out=ln_carry_out,
        )

    def transition(state: LNCarryState, token_id: int, _position: int) -> LNCarryState:
        return transition_carry_state(
            state,
            token_id,
            vocab=vocab,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            allow_bos=False,
            allow_eos=int(token_id) == vocab.eos_id and bool(is_full_chart_end),
        )

    result = run_generation_engine(
        initial_state=ln_carry_in,
        is_complete=is_complete,
        decoder_input_tokens=decoder_inputs,
        valid_token_mask=ordinary_mask,
        completion_token_mask=completion_mask,
        make_step=make_step,
        transition=transition,
        default_logits=lambda valid_mask: _default_generation_logits(valid_mask, vocab=vocab),
        logits_fn=logits_fn,
        ordinary_block_token_ids=(vocab.bos_id, vocab.eos_id),
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=top_p,
        top_k=top_k,
        generator=generator,
    )

    return MapperGeneratedWindow(
        write_start_ms=int(write_start_ms),
        write_end_ms=int(write_end_ms),
        ln_carry_in=ln_carry_in,
        ln_carry_out=ln_carry_out,
        tokens=result.tokens,
        states_before=result.states_before,
        states_after=result.states_after,
        terminal_state=result.terminal_state,
        completed=result.completed,
        dead_end=result.dead_end,
        max_tokens_exceeded=result.max_tokens_exceeded,
    )


def _select_token(
    logits: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    temperature: float,
    top_p: float | None,
    generator: torch.Generator | None,
) -> int:
    return select_next_token(
        logits,
        valid_mask=valid_mask,
        temperature=float(temperature),
        top_p=top_p,
        generator=generator,
    )


def _apply_top_p(probs: torch.Tensor, *, p: float) -> torch.Tensor:
    return apply_top_p(probs, p=float(p))


def _default_generation_logits(valid_mask: torch.Tensor, *, vocab: MapperTupleVocab) -> torch.Tensor:
    return default_generation_logits(
        valid_mask,
        vocab_size=vocab.size,
        time_shift_token_ids=vocab.time_shift_token_ids,
        time_shift_value=vocab.time_shift_value,
    )
