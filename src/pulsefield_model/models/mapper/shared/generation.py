from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .generation_engine import (
    apply_top_p,
    default_generation_logits,
    run_generation_engine,
    select_next_token,
)
from .replay import CLOSED_OPEN_START_MS, LNCarryState
from .vocab import KEY_COUNT, LaneAction, MapperTupleVocab, coerce_lane_action


class CarryStateError(ValueError):
    pass


class MapperGenerationError(ValueError):
    pass


@dataclass(frozen=True)
class CarryReplayTrace:
    tokens: list[int]
    states_before: list[LNCarryState]
    states_after: list[LNCarryState]
    terminal_state: LNCarryState
    completed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "tokens": list(self.tokens),
            "states_before": [state.to_dict() for state in self.states_before],
            "states_after": [state.to_dict() for state in self.states_after],
            "terminal_state": self.terminal_state.to_dict(),
            "completed": self.completed,
        }


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


@dataclass(frozen=True)
class RecoveryStateMatch:
    generated_index: int
    gold_index: int | None
    reason: str

    @property
    def matched(self) -> bool:
        return self.gold_index is not None


@dataclass(frozen=True)
class RecoveryCEReport:
    loss: torch.Tensor
    matched_count: int
    generated_state_count: int
    recovery_batch_valid_fraction: float
    mismatch_reasons: dict[str, int]
    matched_generated_indices: list[int]
    matched_gold_indices: list[int]

    def to_dict(self) -> dict[str, object]:
        return {
            "recovery_ce": float(self.loss.detach().cpu().item()),
            "matched_count": self.matched_count,
            "generated_state_count": self.generated_state_count,
            "recovery_batch_valid_fraction": self.recovery_batch_valid_fraction,
            "mismatch_reasons": dict(self.mismatch_reasons),
            "matched_generated_indices": list(self.matched_generated_indices),
            "matched_gold_indices": list(self.matched_gold_indices),
        }


LogitsFn = Callable[[MapperGenerationStep], torch.Tensor]


def reconstruct_ln_carry_states(
    timepoints: Sequence[Any],
    *,
    write_start_ms: int,
    write_end_ms: int,
) -> tuple[LNCarryState, LNCarryState]:
    """Reconstruct tuple carry-in/out from full-chart mapper timepoints."""

    start_ms = int(write_start_ms)
    end_ms = int(write_end_ms)
    if end_ms <= start_ms:
        raise ValueError(f"write_end_ms must be after write_start_ms: {start_ms}..{end_ms}")
    grouped = _group_timepoints(timepoints)
    open_starts: list[int | None] = [None] * KEY_COUNT

    for timepoint in grouped:
        if timepoint.time_ms < start_ms:
            _apply_timepoint_to_open_starts(open_starts, timepoint, include_starts=True, include_ends=True, include_taps=True)
        else:
            break

    ln_carry_in = LNCarryState.from_open_starts(start_ms, open_starts)
    validate_boundary_carry_state(ln_carry_in, boundary_name="ln_carry_in")

    for timepoint in grouped:
        if timepoint.time_ms < start_ms:
            continue
        if timepoint.time_ms == start_ms:
            _apply_timepoint_to_open_starts(open_starts, timepoint, include_starts=True, include_ends=True, include_taps=True)
            continue
        if timepoint.time_ms < end_ms:
            _apply_timepoint_to_open_starts(open_starts, timepoint, include_starts=True, include_ends=True, include_taps=True)
            continue
        break

    ln_carry_out = LNCarryState.from_open_starts(end_ms, open_starts)
    validate_boundary_carry_state(ln_carry_out, boundary_name="ln_carry_out")
    return ln_carry_in, ln_carry_out


def validate_boundary_carry_state(state: LNCarryState, *, boundary_name: str = "ln_carry") -> None:
    for lane, is_open in enumerate(state.open_mask):
        if not is_open:
            continue
        start_ms = state.open_start_ms[lane]
        if start_ms is None or int(start_ms) >= state.current_ms:
            raise CarryStateError(
                f"{boundary_name} open lane {lane} must have started before the boundary: "
                f"{start_ms} >= {state.current_ms}",
            )
        if state.open_age_ms[lane] <= 0:
            raise CarryStateError(f"{boundary_name} open lane {lane} must have positive open_age_ms")


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


def replay_fragment_tokens(
    token_ids: Sequence[int],
    *,
    vocab: MapperTupleVocab,
    write_start_ms: int,
    write_end_ms: int,
    ln_carry_in: LNCarryState,
    ln_carry_out: LNCarryState,
    allow_bos: bool = False,
    allow_eos: bool = False,
) -> CarryReplayTrace:
    state = ln_carry_in
    states_before: list[LNCarryState] = []
    states_after: list[LNCarryState] = []
    tokens = [int(token_id) for token_id in token_ids]
    for token_id in tokens:
        states_before.append(state)
        mask = carry_aware_valid_token_mask(
            state=state,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            ln_carry_out=ln_carry_out,
            vocab=vocab,
            is_full_chart_start=allow_bos,
            is_full_chart_end=allow_eos,
            token_position=len(states_before) - 1,
        )
        if 0 <= token_id < vocab.size and not bool(mask[token_id].item()):
            raise CarryStateError(f"token is invalid under carry-aware grammar: {vocab.token_name(token_id)}")
        state = transition_carry_state(
            state,
            token_id,
            vocab=vocab,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            allow_bos=allow_bos,
            allow_eos=allow_eos,
        )
        states_after.append(state)
    return CarryReplayTrace(
        tokens=tokens,
        states_before=states_before,
        states_after=states_after,
        terminal_state=state,
        completed=window_is_complete(state, write_end_ms=write_end_ms, ln_carry_out=ln_carry_out),
    )


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


def strict_match_to_gold_replay(
    *,
    generated_states: Sequence[LNCarryState],
    gold_states: Sequence[LNCarryState],
    age_tolerance_ms: int = 10,
) -> list[RecoveryStateMatch]:
    matches: list[RecoveryStateMatch] = []
    for generated_index, generated_state in enumerate(generated_states):
        current_candidates = [
            (gold_index, gold_state)
            for gold_index, gold_state in enumerate(gold_states)
            if gold_state.current_ms == generated_state.current_ms
        ]
        if not current_candidates:
            matches.append(RecoveryStateMatch(generated_index, None, "current_ms_mismatch"))
            continue
        open_mask_candidates = [
            (gold_index, gold_state)
            for gold_index, gold_state in current_candidates
            if gold_state.open_mask == generated_state.open_mask
        ]
        if not open_mask_candidates:
            matches.append(RecoveryStateMatch(generated_index, None, "open_mask_mismatch"))
            continue
        open_start_candidates = [
            (gold_index, gold_state)
            for gold_index, gold_state in open_mask_candidates
            if gold_state.open_start_ms == generated_state.open_start_ms
        ]
        if not open_start_candidates:
            matches.append(RecoveryStateMatch(generated_index, None, "open_start_mismatch"))
            continue
        for gold_index, gold_state in open_start_candidates:
            if carry_states_equal(
                generated_state,
                gold_state,
                age_tolerance_ms=age_tolerance_ms,
                compare_open_start=True,
            ):
                matches.append(RecoveryStateMatch(generated_index, gold_index, "matched"))
                break
        else:
            matches.append(RecoveryStateMatch(generated_index, None, "open_age_mismatch"))
    return matches


def short_rollout_recovery_ce(
    *,
    logits: torch.Tensor,
    generated_states: Sequence[LNCarryState],
    gold_states: Sequence[LNCarryState],
    gold_target_tokens: Sequence[int] | torch.Tensor,
    age_tolerance_ms: int = 10,
) -> RecoveryCEReport:
    matches = strict_match_to_gold_replay(
        generated_states=generated_states,
        gold_states=gold_states,
        age_tolerance_ms=age_tolerance_ms,
    )
    return recovery_ce_from_matches(
        logits=logits,
        gold_target_tokens=gold_target_tokens,
        matches=matches,
    )


def recovery_ce_from_matches(
    *,
    logits: torch.Tensor,
    gold_target_tokens: Sequence[int] | torch.Tensor,
    matches: Sequence[RecoveryStateMatch],
) -> RecoveryCEReport:
    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [generated_states,V], got {tuple(logits.shape)}")
    matched_generated = [match.generated_index for match in matches if match.gold_index is not None]
    matched_gold = [int(match.gold_index) for match in matches if match.gold_index is not None]
    reason_counts: dict[str, int] = {}
    for match in matches:
        reason_counts[match.reason] = reason_counts.get(match.reason, 0) + 1
    if matched_generated:
        target_tensor = torch.as_tensor(gold_target_tokens, dtype=torch.long, device=logits.device)
        selected_logits = logits[torch.tensor(matched_generated, dtype=torch.long, device=logits.device)]
        selected_targets = target_tensor[torch.tensor(matched_gold, dtype=torch.long, device=logits.device)]
        loss = F.cross_entropy(selected_logits, selected_targets, reduction="mean")
    else:
        loss = logits.sum() * 0.0
    generated_count = len(matches)
    return RecoveryCEReport(
        loss=loss,
        matched_count=len(matched_generated),
        generated_state_count=generated_count,
        recovery_batch_valid_fraction=float(len(matched_generated) / generated_count) if generated_count else 0.0,
        mismatch_reasons=reason_counts,
        matched_generated_indices=matched_generated,
        matched_gold_indices=matched_gold,
    )


def coerce_ln_carry_state(value: Any, *, current_ms: int | None = None) -> LNCarryState:
    if isinstance(value, LNCarryState):
        return value
    if isinstance(value, Mapping):
        payload = value
        resolved_current_ms = _coerce_single_int(payload.get("current_ms", current_ms), name="current_ms")
        return LNCarryState(
            current_ms=resolved_current_ms,
            open_mask=_normalize_open_mask(payload["open_mask"]),
            open_start_ms=_normalize_open_starts(payload["open_start_ms"]),
            open_age_ms=_normalize_open_ages(payload["open_age_ms"]),
        )
    if hasattr(value, "current_ms") and hasattr(value, "open_mask") and hasattr(value, "open_start_ms"):
        return LNCarryState(
            current_ms=_coerce_single_int(getattr(value, "current_ms"), name="current_ms"),
            open_mask=_normalize_open_mask(getattr(value, "open_mask")),
            open_start_ms=_normalize_open_starts(getattr(value, "open_start_ms")),
            open_age_ms=_normalize_open_ages(getattr(value, "open_age_ms")),
        )
    if hasattr(value, "open_mask") and current_ms is not None:
        open_mask = _normalize_open_mask(getattr(value, "open_mask"))
        open_age_ms = _normalize_open_ages(getattr(value, "open_age_ms", (0, 0, 0, 0)))
        starts: list[int | None] = []
        for is_open, age_ms in zip(open_mask, open_age_ms, strict=True):
            starts.append(int(current_ms) - int(age_ms) if is_open else None)
        return LNCarryState.from_open_starts(int(current_ms), starts)
    raise TypeError(f"cannot coerce LNCarryState from {value!r}")


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


@dataclass(frozen=True)
class _CoercedTimepoint:
    time_ms: int
    lane_actions: tuple[LaneAction, LaneAction, LaneAction, LaneAction]


def _group_timepoints(timepoints: Sequence[Any]) -> list[_CoercedTimepoint]:
    grouped: dict[int, list[LaneAction]] = {}
    for raw_timepoint in sorted((_coerce_timepoint(item) for item in timepoints), key=lambda item: item.time_ms):
        actions = grouped.setdefault(raw_timepoint.time_ms, [LaneAction.NONE] * KEY_COUNT)
        for lane, action in enumerate(raw_timepoint.lane_actions):
            if action == LaneAction.NONE:
                continue
            if actions[lane] != LaneAction.NONE:
                raise CarryStateError(
                    f"multiple same-lane actions at {raw_timepoint.time_ms}ms lane {lane}",
                )
            actions[lane] = action
    return [
        _CoercedTimepoint(time_ms=time_ms, lane_actions=tuple(actions))  # type: ignore[arg-type]
        for time_ms, actions in sorted(grouped.items())
        if any(action != LaneAction.NONE for action in actions)
    ]


def _coerce_timepoint(timepoint: Any) -> _CoercedTimepoint:
    if not hasattr(timepoint, "time_ms") or not hasattr(timepoint, "lane_actions"):
        raise TypeError(f"mapper timepoint must expose time_ms and lane_actions: {timepoint!r}")
    actions = tuple(coerce_lane_action(action) for action in getattr(timepoint, "lane_actions"))
    if len(actions) != KEY_COUNT:
        raise ValueError(f"mapper timepoint must contain {KEY_COUNT} lane actions: {actions}")
    return _CoercedTimepoint(time_ms=int(getattr(timepoint, "time_ms")), lane_actions=actions)  # type: ignore[arg-type]


def _apply_timepoint_to_open_starts(
    open_starts: list[int | None],
    timepoint: _CoercedTimepoint,
    *,
    include_starts: bool,
    include_ends: bool,
    include_taps: bool,
) -> None:
    for lane, action in enumerate(timepoint.lane_actions):
        if action == LaneAction.NONE:
            continue
        if action == LaneAction.HOLD_START:
            if not include_starts:
                continue
            if open_starts[lane] is not None:
                raise CarryStateError(f"HOLD_START on open lane {lane} at {timepoint.time_ms}ms")
            open_starts[lane] = int(timepoint.time_ms)
        elif action == LaneAction.HOLD_END:
            if not include_ends:
                continue
            if open_starts[lane] is None:
                raise CarryStateError(f"HOLD_END on closed lane {lane} at {timepoint.time_ms}ms")
            open_starts[lane] = None
        elif action == LaneAction.TAP:
            if not include_taps:
                continue
            if open_starts[lane] is not None:
                raise CarryStateError(f"TAP on open lane {lane} at {timepoint.time_ms}ms")


def _coerce_single_int(value: Any, *, name: str) -> int:
    if value is None:
        raise CarryStateError(f"{name} is required")
    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().reshape(-1)
        if int(flat.numel()) != 1:
            raise CarryStateError(f"{name} must contain exactly one value, got shape {tuple(value.shape)}")
        return int(flat.item())
    return int(value)


def _flat_sequence(value: Any, *, name: str) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    return list(value)


def _normalize_open_mask(value: Sequence[bool] | torch.Tensor) -> tuple[bool, bool, bool, bool]:
    items = tuple(bool(item) for item in _flat_sequence(value, name="open_mask"))
    if len(items) != KEY_COUNT:
        raise CarryStateError(f"open_mask must contain {KEY_COUNT} lanes: {value}")
    return items  # type: ignore[return-value]


def _normalize_open_starts(
    value: Sequence[int | None] | torch.Tensor,
) -> tuple[int | None, int | None, int | None, int | None]:
    items = tuple(
        None if item is None or int(item) == CLOSED_OPEN_START_MS else int(item)
        for item in _flat_sequence(value, name="open_start_ms")
    )
    if len(items) != KEY_COUNT:
        raise CarryStateError(f"open_start_ms must contain {KEY_COUNT} lanes: {value}")
    return items  # type: ignore[return-value]


def _normalize_open_ages(value: Sequence[int] | torch.Tensor) -> tuple[int, int, int, int]:
    items = tuple(int(item) for item in _flat_sequence(value, name="open_age_ms"))
    if len(items) != KEY_COUNT:
        raise CarryStateError(f"open_age_ms must contain {KEY_COUNT} lanes: {value}")
    if any(item < 0 for item in items):
        raise CarryStateError(f"open_age_ms must be non-negative: {value}")
    return items  # type: ignore[return-value]
