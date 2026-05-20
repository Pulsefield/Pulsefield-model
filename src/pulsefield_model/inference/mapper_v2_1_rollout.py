from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from pulsefield_model.models.mapper.v2_1.grammar import valid_token_mask
from pulsefield_model.models.mapper.v2_1.replay import (
    LNCarryState,
    MapperReplayState,
    empty_ln_carry_state,
    ln_carry_state_tensors,
    replay_state_matches_carry,
    transition_replay_state,
)
from pulsefield_model.models.mapper.v2_1.tokenizer import MAPPER_DENSITY_FRAMES, MAPPER_WRITE_MS
from pulsefield_model.models.mapper.v2_1.vocab import KEY_COUNT, MapperV21Vocab


class MapperV21GenerationError(ValueError):
    pass


@dataclass(frozen=True)
class MapperV21GenerationStep:
    decoder_input_tokens: torch.Tensor
    generated_tokens: tuple[int, ...]
    state: MapperReplayState
    valid_token_mask: torch.Tensor
    token_index: int
    write_start_ms: int
    write_end_ms: int
    chart_end_ms: int
    ln_carry_in: LNCarryState
    ln_carry_out: LNCarryState
    is_full_chart_start: bool
    is_full_chart_end: bool


@dataclass(frozen=True)
class MapperV21GeneratedWindow:
    write_start_ms: int
    write_end_ms: int
    chart_end_ms: int
    ln_carry_in: LNCarryState
    ln_carry_out: LNCarryState
    tokens: list[int]
    states_before: list[MapperReplayState]
    states_after: list[MapperReplayState]
    terminal_state: MapperReplayState
    completed: bool
    dead_end: bool
    max_tokens_exceeded: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "write_start_ms": self.write_start_ms,
            "write_end_ms": self.write_end_ms,
            "chart_end_ms": self.chart_end_ms,
            "ln_carry_in": self.ln_carry_in.to_dict(),
            "ln_carry_out": self.ln_carry_out.to_dict(),
            "tokens": list(self.tokens),
            "states_before": [_replay_state_to_dict(state) for state in self.states_before],
            "states_after": [_replay_state_to_dict(state) for state in self.states_after],
            "terminal_state": _replay_state_to_dict(self.terminal_state),
            "completed": self.completed,
            "dead_end": self.dead_end,
            "max_tokens_exceeded": self.max_tokens_exceeded,
        }


@dataclass(frozen=True)
class MapperV21FullRollout:
    chart_end_ms: int
    windows: list[MapperV21GeneratedWindow]
    tokens: list[int]
    completed: bool
    dead_end: bool
    max_tokens_exceeded: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "chart_end_ms": self.chart_end_ms,
            "tokens": list(self.tokens),
            "completed": self.completed,
            "dead_end": self.dead_end,
            "max_tokens_exceeded": self.max_tokens_exceeded,
            "windows": [window.to_dict() for window in self.windows],
        }


MapperV21LogitsFn = Callable[[MapperV21GenerationStep], torch.Tensor]
MapperV21LogitsObserver = Callable[[MapperV21GenerationStep, torch.Tensor], None]
MapperV21WindowBatchProvider = Callable[[int, int], Mapping[str, Any]]


def grammar_constrained_window_generation_v2_1(
    *,
    vocab: MapperV21Vocab,
    write_start_ms: int,
    write_end_ms: int,
    chart_end_ms: int,
    ln_carry_in: LNCarryState,
    ln_carry_out: LNCarryState,
    logits_fn: MapperV21LogitsFn | None = None,
    left_context_tokens: Sequence[int] = (),
    is_full_chart_start: bool = False,
    is_full_chart_end: bool = False,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
    logits_observer: MapperV21LogitsObserver | None = None,
) -> MapperV21GeneratedWindow:
    write_start_ms = int(write_start_ms)
    write_end_ms = int(write_end_ms)
    chart_end_ms = int(chart_end_ms)
    if int(ln_carry_in.current_ms) != write_start_ms:
        raise ValueError("ln_carry_in.current_ms must equal write_start_ms")
    target_end_ms = chart_end_ms if bool(is_full_chart_end) else write_end_ms
    if int(ln_carry_out.current_ms) != int(target_end_ms):
        raise ValueError("ln_carry_out.current_ms must equal the generation target end")
    if int(max_tokens) <= 0:
        raise ValueError("max_tokens must be positive")

    state = _initial_replay_state(ln_carry_in)
    generated: list[int] = []
    states_before: list[MapperReplayState] = []
    states_after: list[MapperReplayState] = []
    max_tokens_exceeded = False
    dead_end = False

    while True:
        if _window_complete_v2_1(state, ln_carry_out=ln_carry_out):
            if bool(is_full_chart_end) and (not generated or generated[-1] != vocab.eos_id):
                if len(generated) >= int(max_tokens):
                    max_tokens_exceeded = True
                    break
                decoder_inputs = decoder_input_tokens_for_generation_v2_1(
                    vocab=vocab,
                    left_context_tokens=left_context_tokens,
                    generated_tokens=generated,
                    is_full_chart_start=is_full_chart_start,
                )
                eos_only_mask = torch.zeros(vocab.size, dtype=torch.bool)
                eos_only_mask[vocab.eos_id] = True
                step = MapperV21GenerationStep(
                    decoder_input_tokens=torch.tensor(decoder_inputs, dtype=torch.long),
                    generated_tokens=tuple(generated),
                    state=state,
                    valid_token_mask=eos_only_mask,
                    token_index=len(generated),
                    write_start_ms=write_start_ms,
                    write_end_ms=write_end_ms,
                    chart_end_ms=chart_end_ms,
                    ln_carry_in=ln_carry_in,
                    ln_carry_out=ln_carry_out,
                    is_full_chart_start=bool(is_full_chart_start),
                    is_full_chart_end=bool(is_full_chart_end),
                )
                logits = _default_generation_logits_v2_1(eos_only_mask, vocab=vocab) if logits_fn is None else logits_fn(step)
                _observe_logits_v2_1(logits_observer, step, logits)
                token_id = _select_token_v2_1(
                    logits,
                    valid_mask=eos_only_mask.to(device=logits.device if isinstance(logits, torch.Tensor) else "cpu"),
                    temperature=float(temperature),
                    top_p=top_p,
                    generator=generator,
                )
                states_before.append(state)
                generated.append(token_id)
                state = transition_replay_state(
                    state,
                    token_id,
                    position=len(generated) - 1,
                    vocab=vocab,
                    write_start_ms=write_start_ms,
                    write_end_ms=write_end_ms,
                    chart_end_ms=chart_end_ms,
                    ln_carry_out=ln_carry_out,
                    is_full_chart_start=bool(is_full_chart_start),
                    is_full_chart_end=bool(is_full_chart_end),
                )
                states_after.append(state)
            break

        if len(generated) >= int(max_tokens):
            max_tokens_exceeded = True
            break
        decoder_inputs = decoder_input_tokens_for_generation_v2_1(
            vocab=vocab,
            left_context_tokens=left_context_tokens,
            generated_tokens=generated,
            is_full_chart_start=is_full_chart_start,
        )
        mask = valid_token_mask(
            position=len(generated),
            current_ms=state.current_ms,
            open_mask=state.open_mask,
            open_start_ms=state.open_start_ms,
            open_age_ms=state.open_age_ms,
            emitted_lane_mask=state.emitted_lane_mask,
            last_lane_index=state.last_lane_index,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            chart_end_ms=chart_end_ms,
            ln_carry_in=ln_carry_in,
            ln_carry_out=ln_carry_out,
            is_full_chart_start=bool(is_full_chart_start),
            is_full_chart_end=bool(is_full_chart_end),
            vocab=vocab,
        )
        mask[vocab.bos_id] = False
        mask[vocab.eos_id] = False
        if not bool(mask.any().item()):
            dead_end = True
            break
        step = MapperV21GenerationStep(
            decoder_input_tokens=torch.tensor(decoder_inputs, dtype=torch.long),
            generated_tokens=tuple(generated),
            state=state,
            valid_token_mask=mask,
            token_index=len(generated),
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            chart_end_ms=chart_end_ms,
            ln_carry_in=ln_carry_in,
            ln_carry_out=ln_carry_out,
            is_full_chart_start=bool(is_full_chart_start),
            is_full_chart_end=bool(is_full_chart_end),
        )
        logits = _default_generation_logits_v2_1(mask, vocab=vocab) if logits_fn is None else logits_fn(step)
        _observe_logits_v2_1(logits_observer, step, logits)
        token_id = _select_token_v2_1(
            logits,
            valid_mask=mask.to(device=logits.device if isinstance(logits, torch.Tensor) else "cpu"),
            temperature=float(temperature),
            top_p=top_p,
            generator=generator,
        )
        states_before.append(state)
        generated.append(token_id)
        state = transition_replay_state(
            state,
            token_id,
            position=len(generated) - 1,
            vocab=vocab,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            chart_end_ms=chart_end_ms,
            ln_carry_out=ln_carry_out,
            is_full_chart_start=bool(is_full_chart_start),
            is_full_chart_end=bool(is_full_chart_end),
        )
        states_after.append(state)

    completed = _window_complete_v2_1(state, ln_carry_out=ln_carry_out)
    return MapperV21GeneratedWindow(
        write_start_ms=write_start_ms,
        write_end_ms=write_end_ms,
        chart_end_ms=chart_end_ms,
        ln_carry_in=ln_carry_in,
        ln_carry_out=ln_carry_out,
        tokens=generated,
        states_before=states_before,
        states_after=states_after,
        terminal_state=state,
        completed=completed,
        dead_end=dead_end,
        max_tokens_exceeded=max_tokens_exceeded,
    )


def generate_full_song_rollout_v2_1(
    *,
    model: torch.nn.Module,
    vocab: MapperV21Vocab,
    chart_end_ms: int,
    window_batch_provider: MapperV21WindowBatchProvider,
    device: torch.device | str = "cpu",
    normalized_difficulty: float = 0.0,
    max_tokens_per_window: int | None = None,
    temperature: float = 0.0,
    top_p: float | None = None,
    time_shift_length_penalty_alpha: float = 0.0,
    time_shift_delta_penalty_alpha: float = 0.0,
    generator: torch.Generator | None = None,
    logits_observer: MapperV21LogitsObserver | None = None,
) -> MapperV21FullRollout:
    chart_end_ms = int(chart_end_ms)
    if chart_end_ms <= 0:
        raise ValueError("chart_end_ms must be positive")
    if chart_end_ms % 10 != 0:
        raise ValueError("chart_end_ms must align to the 10ms token grid")
    resolved_device = torch.device(device)
    model.to(resolved_device)
    model.eval()
    max_seq_len = int(getattr(getattr(model, "config", None), "max_seq_len", 512))
    effective_max_tokens = max_seq_len - 1 if max_tokens_per_window is None else int(max_tokens_per_window)
    effective_max_tokens = min(effective_max_tokens, max_seq_len - 1)
    if effective_max_tokens <= 0:
        raise ValueError("model max_seq_len must leave room for at least one generated token")

    windows: list[MapperV21GeneratedWindow] = []
    tokens: list[int] = []
    carry_in = empty_ln_carry_state(0)
    left_context_tokens: tuple[int, ...] = ()
    latest_start_ms = (chart_end_ms - 1) // MAPPER_WRITE_MS * MAPPER_WRITE_MS

    for write_start_ms in range(0, latest_start_ms + 1, MAPPER_WRITE_MS):
        write_end_ms = write_start_ms + MAPPER_WRITE_MS
        is_full_chart_start = write_start_ms == 0
        is_full_chart_end = write_start_ms <= chart_end_ms <= write_end_ms
        target_end_ms = chart_end_ms if is_full_chart_end else write_end_ms
        carry_out = empty_ln_carry_state(target_end_ms)
        control_batch = dict(window_batch_provider(write_start_ms, write_end_ms))
        logits_fn = mapper_v2_1_logits_fn(
            model=model,
            vocab=vocab,
            device=resolved_device,
            normalized_difficulty=float(normalized_difficulty),
            control_batch=control_batch,
            ln_carry_in=carry_in,
            ln_carry_out=carry_out,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            chart_end_ms=chart_end_ms,
            is_full_chart_start=is_full_chart_start,
            is_full_chart_end=is_full_chart_end,
            time_shift_length_penalty_alpha=float(time_shift_length_penalty_alpha),
            time_shift_delta_penalty_alpha=float(time_shift_delta_penalty_alpha),
        )
        generated = grammar_constrained_window_generation_v2_1(
            vocab=vocab,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            chart_end_ms=chart_end_ms,
            ln_carry_in=carry_in,
            ln_carry_out=carry_out,
            logits_fn=logits_fn,
            left_context_tokens=left_context_tokens,
            is_full_chart_start=is_full_chart_start,
            is_full_chart_end=is_full_chart_end,
            max_tokens=effective_max_tokens,
            temperature=float(temperature),
            top_p=top_p,
            generator=generator,
            logits_observer=logits_observer,
        )
        windows.append(generated)
        tokens.extend(generated.tokens)
        if generated.tokens:
            left_context_tokens = (int(generated.tokens[-1]),)
        if not generated.completed or generated.dead_end or generated.max_tokens_exceeded:
            break
        carry_in = _carry_from_replay_state(generated.terminal_state)

    return MapperV21FullRollout(
        chart_end_ms=chart_end_ms,
        windows=windows,
        tokens=tokens,
        completed=bool(windows) and all(window.completed for window in windows) and len(windows) == ((latest_start_ms // MAPPER_WRITE_MS) + 1),
        dead_end=any(window.dead_end for window in windows),
        max_tokens_exceeded=any(window.max_tokens_exceeded for window in windows),
    )


def mapper_v2_1_logits_fn(
    *,
    model: torch.nn.Module,
    vocab: MapperV21Vocab,
    device: torch.device,
    normalized_difficulty: float,
    control_batch: Mapping[str, Any],
    ln_carry_in: LNCarryState,
    ln_carry_out: LNCarryState,
    write_start_ms: int,
    write_end_ms: int,
    chart_end_ms: int,
    is_full_chart_start: bool,
    is_full_chart_end: bool,
    time_shift_length_penalty_alpha: float,
    time_shift_delta_penalty_alpha: float = 0.0,
    apply_grammar_mask: bool = False,
) -> MapperV21LogitsFn:
    time_shift_penalty = _time_shift_length_penalty_tensors_v2_1(
        vocab,
        alpha=float(time_shift_length_penalty_alpha),
        delta_alpha=float(time_shift_delta_penalty_alpha),
        device=device,
    )
    write_start_tensor = torch.tensor([int(write_start_ms)], dtype=torch.long, device=device)
    write_end_tensor = torch.tensor([int(write_end_ms)], dtype=torch.long, device=device)
    chart_end_tensor = torch.tensor([int(chart_end_ms)], dtype=torch.long, device=device)
    full_start_tensor = torch.tensor([bool(is_full_chart_start)], dtype=torch.bool, device=device)
    full_end_tensor = torch.tensor([bool(is_full_chart_end)], dtype=torch.bool, device=device)
    difficulty_tensor = torch.tensor([float(normalized_difficulty)], dtype=torch.float32, device=device)
    carry_in_batch = _carry_batch_v2_1(ln_carry_in, device=device)
    carry_out_batch = _carry_batch_v2_1(ln_carry_out, device=device)

    def logits_fn(step: MapperV21GenerationStep) -> torch.Tensor:
        decoder_input_tokens = step.decoder_input_tokens.to(device=device, dtype=torch.long).unsqueeze(0)
        states = _target_fragment_state_batch_v2_1(
            generated_tokens=step.generated_tokens,
            vocab=vocab,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            chart_end_ms=chart_end_ms,
            ln_carry_in=ln_carry_in,
            ln_carry_out=ln_carry_out,
            is_full_chart_start=bool(is_full_chart_start),
            is_full_chart_end=bool(is_full_chart_end),
            device=device,
        )
        target_tokens = torch.full_like(decoder_input_tokens, vocab.pad_id)
        batch: dict[str, Any] = {
            **control_batch,
            "decoder_input_tokens": decoder_input_tokens,
            "target_fragment_tokens": target_tokens,
            "target_fragment_mask": torch.ones_like(decoder_input_tokens, dtype=torch.bool),
            "target_fragment_states": states,
            "ln_carry_in": carry_in_batch,
            "ln_carry_out": carry_out_batch,
            "write_start_ms": write_start_tensor,
            "write_end_ms": write_end_tensor,
            "chart_end_ms": chart_end_tensor,
            "is_full_chart_start": full_start_tensor,
            "is_full_chart_end": full_end_tensor,
            "normalized_difficulty": difficulty_tensor,
            # Rollout applies step.valid_token_mask before selection; avoid the model's full-prefix mask here.
            "apply_grammar_mask": bool(apply_grammar_mask),
        }
        with torch.inference_mode():
            output = model(batch)
        logits = output.logits_final[0, -1].detach()
        return _apply_time_shift_length_penalty_v2_1(logits, time_shift_penalty=time_shift_penalty)

    return logits_fn


def decoder_input_tokens_for_generation_v2_1(
    *,
    vocab: MapperV21Vocab,
    left_context_tokens: Sequence[int] = (),
    generated_tokens: Sequence[int] = (),
    is_full_chart_start: bool = False,
) -> list[int]:
    context = [int(token_id) for token_id in left_context_tokens]
    if not context:
        if bool(is_full_chart_start):
            context = [vocab.bos_id]
        else:
            raise MapperV21GenerationError("non-initial v2.1 generation requires a left_context_tokens value")
    return [*context, *(int(token_id) for token_id in generated_tokens)]


def zero_control_batch_provider_v2_1(
    *,
    model: torch.nn.Module,
    device: torch.device | str,
) -> MapperV21WindowBatchProvider:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("model must expose a config for zero control batch generation")
    resolved_device = torch.device(device)

    def provider(write_start_ms: int, write_end_ms: int) -> Mapping[str, Any]:
        del write_start_ms, write_end_ms
        batch: dict[str, Any] = {
            "projected_control_memory_8s": torch.zeros(
                (1, MAPPER_DENSITY_FRAMES, int(config.d_model)),
                dtype=torch.float32,
                device=resolved_device,
            ),
            "density_teacher_8s": torch.zeros((1, MAPPER_DENSITY_FRAMES, 1), dtype=torch.float32, device=resolved_device),
        }
        if bool(getattr(config, "use_global_context", False)):
            batch.update(
                {
                    "global_memory": torch.zeros((1, 4, int(config.d_model)), dtype=torch.float32, device=resolved_device),
                    "global_memory_padding_mask": torch.zeros((1, 4), dtype=torch.bool, device=resolved_device),
                    "global_position_features": torch.zeros((1, 4), dtype=torch.float32, device=resolved_device),
                },
            )
        return batch

    return provider


def session_window_batch_provider_v2_1(
    session_runtime: Any,
    *,
    include_control_attention_kv_cache: bool = False,
) -> MapperV21WindowBatchProvider:
    def provider(write_start_ms: int, write_end_ms: int) -> Mapping[str, Any]:
        cache = session_runtime.prepare_mapper_window(
            start_ms=int(write_start_ms),
            end_ms=int(write_end_ms),
            include_control_attention_kv_cache=bool(include_control_attention_kv_cache),
        )
        return cache.as_model_batch()

    return provider


def rollout_to_timepoints_v2_1(
    rollout: MapperV21FullRollout | Sequence[MapperV21GeneratedWindow],
    vocab: MapperV21Vocab,
) -> list[Any]:
    from pulsefield_model.events.canonical import CanonicalTimepoint, LaneAction as CanonicalLaneAction

    windows = rollout.windows if isinstance(rollout, MapperV21FullRollout) else list(rollout)
    grouped: dict[int, list[CanonicalLaneAction]] = {}
    for window in windows:
        for token_id, state_before in zip(window.tokens, window.states_before, strict=True):
            token_id = int(token_id)
            if not vocab.is_lane_action_token(token_id):
                continue
            lane, action = vocab.decode_lane_action(token_id)
            actions = grouped.setdefault(int(state_before.current_ms), [CanonicalLaneAction.NONE] * KEY_COUNT)
            if actions[lane] != CanonicalLaneAction.NONE:
                raise MapperV21GenerationError(
                    f"duplicate generated lane action at {state_before.current_ms}ms lane {lane}",
                )
            actions[lane] = CanonicalLaneAction[action.name]
    return [
        CanonicalTimepoint(time_ms=time_ms, lane_actions=tuple(actions))
        for time_ms, actions in sorted(grouped.items())
        if any(action != CanonicalLaneAction.NONE for action in actions)
    ]


def _target_fragment_state_batch_v2_1(
    *,
    generated_tokens: Sequence[int],
    vocab: MapperV21Vocab,
    write_start_ms: int,
    write_end_ms: int,
    chart_end_ms: int,
    ln_carry_in: LNCarryState,
    ln_carry_out: LNCarryState,
    is_full_chart_start: bool,
    is_full_chart_end: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    state = _initial_replay_state(ln_carry_in)
    states = [state]
    for position, token_id in enumerate(generated_tokens):
        state = transition_replay_state(
            state,
            int(token_id),
            position=position,
            vocab=vocab,
            write_start_ms=int(write_start_ms),
            write_end_ms=int(write_end_ms),
            chart_end_ms=int(chart_end_ms),
            ln_carry_out=ln_carry_out,
            is_full_chart_start=bool(is_full_chart_start),
            is_full_chart_end=bool(is_full_chart_end),
        )
        states.append(state)
    return {
        "current_ms": torch.tensor([[state.current_ms for state in states]], dtype=torch.long, device=device),
        "open_mask": torch.tensor([[state.open_mask for state in states]], dtype=torch.bool, device=device),
        "open_start_ms": torch.tensor(
            [[_open_start_tensor_values_v2_1(state.open_start_ms) for state in states]],
            dtype=torch.long,
            device=device,
        ),
        "open_age_ms": torch.tensor([[state.open_age_ms for state in states]], dtype=torch.long, device=device),
        "emitted_lane_mask": torch.tensor([[state.emitted_lane_mask for state in states]], dtype=torch.bool, device=device),
        "last_lane_index": torch.tensor([[state.last_lane_index for state in states]], dtype=torch.long, device=device),
    }


def _carry_batch_v2_1(carry: LNCarryState, *, device: torch.device) -> dict[str, torch.Tensor]:
    tensors = ln_carry_state_tensors(carry)
    return {key: value.unsqueeze(0).to(device=device) for key, value in tensors.items()}


def _carry_from_replay_state(state: MapperReplayState) -> LNCarryState:
    return LNCarryState(
        current_ms=int(state.current_ms),
        open_mask=state.open_mask,
        open_start_ms=state.open_start_ms,
        open_age_ms=state.open_age_ms,
    )


def _initial_replay_state(carry: LNCarryState) -> MapperReplayState:
    return MapperReplayState(
        position=-1,
        current_ms=int(carry.current_ms),
        open_mask=carry.open_mask,
        open_start_ms=carry.open_start_ms,
        open_age_ms=carry.open_age_ms,
    )


def _window_complete_v2_1(state: MapperReplayState, *, ln_carry_out: LNCarryState) -> bool:
    return replay_state_matches_carry(state, ln_carry_out)


def _open_start_tensor_values_v2_1(values: Sequence[int | None]) -> tuple[int, int, int, int]:
    if len(values) != KEY_COUNT:
        raise ValueError(f"open_start_ms must contain {KEY_COUNT} lanes: {values}")
    return tuple(-1 if value is None else int(value) for value in values)  # type: ignore[return-value]


def _time_shift_length_penalty_tensors_v2_1(
    vocab: MapperV21Vocab,
    *,
    alpha: float,
    delta_alpha: float = 0.0,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    alpha = float(alpha)
    if alpha < 0.0:
        raise ValueError(f"time_shift_length_penalty_alpha must be non-negative, got {alpha}")
    delta_alpha = float(delta_alpha)
    if delta_alpha < 0.0:
        raise ValueError(f"time_shift_delta_penalty_alpha must be non-negative, got {delta_alpha}")
    if alpha == 0.0 and delta_alpha == 0.0:
        return None
    token_ids = [int(token_id) for token_id in vocab.time_shift_token_ids]
    if not token_ids:
        return None
    penalties = [
        alpha + delta_alpha * (float(vocab.time_shift_value(token_id)) / 1000.0)
        for token_id in token_ids
    ]
    return (
        torch.tensor(token_ids, dtype=torch.long, device=device),
        torch.tensor(penalties, dtype=torch.float32, device=device),
    )


def _apply_time_shift_length_penalty_v2_1(
    logits: torch.Tensor,
    *,
    time_shift_penalty: tuple[torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor:
    if time_shift_penalty is None:
        return logits
    token_ids, penalties = time_shift_penalty
    adjusted = logits.clone()
    adjusted[token_ids] -= penalties.to(device=adjusted.device, dtype=adjusted.dtype)
    return adjusted


def _observe_logits_v2_1(
    observer: MapperV21LogitsObserver | None,
    step: MapperV21GenerationStep,
    logits: torch.Tensor,
) -> None:
    if observer is None:
        return
    snapshot_step = replace(
        step,
        decoder_input_tokens=step.decoder_input_tokens.detach().clone(),
        valid_token_mask=step.valid_token_mask.detach().clone(),
    )
    observer(snapshot_step, logits.detach().clone())


def _select_token_v2_1(
    logits: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    temperature: float,
    top_p: float | None,
    generator: torch.Generator | None,
) -> int:
    flat_logits = torch.as_tensor(logits, dtype=torch.float32).reshape(-1)
    if int(flat_logits.numel()) != int(valid_mask.numel()):
        raise ValueError(f"logits must contain {valid_mask.numel()} values, got {flat_logits.numel()}")
    mask = valid_mask.to(device=flat_logits.device, dtype=torch.bool).reshape(-1)
    masked = flat_logits.masked_fill(~mask, -torch.inf)
    if float(temperature) <= 0.0:
        return int(torch.argmax(masked).item())
    probs = torch.softmax(masked / float(temperature), dim=-1)
    if top_p is not None:
        probs = _apply_top_p_v2_1(probs, p=float(top_p))
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def _apply_top_p_v2_1(probs: torch.Tensor, *, p: float) -> torch.Tensor:
    if not 0.0 < p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {p}")
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep_sorted = cumulative <= p
    if keep_sorted.numel() > 0:
        keep_sorted[0] = True
    keep = torch.zeros_like(probs, dtype=torch.bool)
    keep[sorted_indices[keep_sorted]] = True
    filtered = torch.where(keep, probs, torch.zeros_like(probs))
    total = filtered.sum()
    if float(total.item()) <= 0.0:
        return probs
    return filtered / total


def _default_generation_logits_v2_1(valid_mask: torch.Tensor, *, vocab: MapperV21Vocab) -> torch.Tensor:
    logits = torch.zeros(vocab.size, dtype=torch.float32)
    logits[~valid_mask.cpu()] = -torch.inf
    valid_time_shifts = [
        token_id
        for token_id in vocab.time_shift_token_ids
        if bool(valid_mask[token_id].item())
    ]
    if valid_time_shifts:
        best = max(valid_time_shifts, key=vocab.time_shift_value)
        logits[best] = 1.0
    return logits


def _replay_state_to_dict(state: MapperReplayState) -> dict[str, object]:
    return {
        "position": int(state.position),
        "current_ms": int(state.current_ms),
        "open_mask": list(state.open_mask),
        "open_start_ms": list(state.open_start_ms),
        "open_age_ms": list(state.open_age_ms),
        "emitted_lane_mask": list(state.emitted_lane_mask),
        "last_lane_index": int(state.last_lane_index),
    }


__all__ = [
    "MapperV21FullRollout",
    "MapperV21GeneratedWindow",
    "MapperV21GenerationError",
    "MapperV21GenerationStep",
    "MapperV21LogitsObserver",
    "decoder_input_tokens_for_generation_v2_1",
    "generate_full_song_rollout_v2_1",
    "grammar_constrained_window_generation_v2_1",
    "mapper_v2_1_logits_fn",
    "rollout_to_timepoints_v2_1",
    "session_window_batch_provider_v2_1",
    "zero_control_batch_provider_v2_1",
]
