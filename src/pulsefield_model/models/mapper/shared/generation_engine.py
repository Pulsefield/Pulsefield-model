from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import torch


StateT = TypeVar("StateT")
StepT = TypeVar("StepT")

TimeShiftPenalty = tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class GenerationEngineResult(Generic[StateT]):
    tokens: list[int]
    states_before: list[StateT]
    states_after: list[StateT]
    terminal_state: StateT
    completed: bool
    dead_end: bool
    max_tokens_exceeded: bool


def run_generation_engine(
    *,
    initial_state: StateT,
    is_complete: Callable[[StateT], bool],
    decoder_input_tokens: Callable[[tuple[int, ...]], Sequence[int]],
    valid_token_mask: Callable[[StateT, tuple[int, ...]], torch.Tensor],
    completion_token_mask: Callable[[StateT, tuple[int, ...]], torch.Tensor | None],
    make_step: Callable[[torch.Tensor, tuple[int, ...], StateT, torch.Tensor, int], StepT],
    transition: Callable[[StateT, int, int], StateT],
    default_logits: Callable[[torch.Tensor], torch.Tensor],
    logits_fn: Callable[[StepT], torch.Tensor] | None = None,
    logits_observer: Callable[[StepT, torch.Tensor], None] | None = None,
    ordinary_block_token_ids: Sequence[int] = (),
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> GenerationEngineResult[StateT]:
    state = initial_state
    generated: list[int] = []
    states_before: list[StateT] = []
    states_after: list[StateT] = []
    max_tokens_exceeded = False
    dead_end = False
    max_token_count = int(max_tokens)

    while True:
        generated_tuple = tuple(generated)
        if is_complete(state):
            mask = completion_token_mask(state, generated_tuple)
            if mask is None:
                break
            if len(generated) >= max_token_count:
                max_tokens_exceeded = True
                break
            if not bool(mask.any().item()):
                dead_end = True
                break
            state = _generate_one_step(
                state=state,
                generated=generated,
                states_before=states_before,
                states_after=states_after,
                decoder_input_tokens=decoder_input_tokens,
                make_step=make_step,
                transition=transition,
                default_logits=default_logits,
                logits_fn=logits_fn,
                logits_observer=logits_observer,
                valid_mask=mask,
                temperature=float(temperature),
                top_p=top_p,
                top_k=top_k,
                generator=generator,
            )
            continue

        if len(generated) >= max_token_count:
            max_tokens_exceeded = True
            break

        mask = valid_token_mask(state, generated_tuple).clone()
        for token_id in ordinary_block_token_ids:
            token = int(token_id)
            if 0 <= token < int(mask.numel()):
                mask[token] = False
        if not bool(mask.any().item()):
            dead_end = True
            break

        state = _generate_one_step(
            state=state,
            generated=generated,
            states_before=states_before,
            states_after=states_after,
            decoder_input_tokens=decoder_input_tokens,
            make_step=make_step,
            transition=transition,
            default_logits=default_logits,
            logits_fn=logits_fn,
            logits_observer=logits_observer,
            valid_mask=mask,
            temperature=float(temperature),
            top_p=top_p,
            top_k=top_k,
            generator=generator,
        )

    return GenerationEngineResult(
        tokens=generated,
        states_before=states_before,
        states_after=states_after,
        terminal_state=state,
        completed=is_complete(state),
        dead_end=dead_end,
        max_tokens_exceeded=max_tokens_exceeded,
    )


def _generate_one_step(
    *,
    state: StateT,
    generated: list[int],
    states_before: list[StateT],
    states_after: list[StateT],
    decoder_input_tokens: Callable[[tuple[int, ...]], Sequence[int]],
    make_step: Callable[[torch.Tensor, tuple[int, ...], StateT, torch.Tensor, int], StepT],
    transition: Callable[[StateT, int, int], StateT],
    default_logits: Callable[[torch.Tensor], torch.Tensor],
    logits_fn: Callable[[StepT], torch.Tensor] | None,
    logits_observer: Callable[[StepT, torch.Tensor], None] | None,
    valid_mask: torch.Tensor,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    generator: torch.Generator | None,
) -> StateT:
    token_index = len(generated)
    generated_tuple = tuple(generated)
    decoder_inputs = torch.tensor(
        list(decoder_input_tokens(generated_tuple)),
        dtype=torch.long,
    )
    step = make_step(decoder_inputs, generated_tuple, state, valid_mask, token_index)
    logits = default_logits(valid_mask) if logits_fn is None else logits_fn(step)
    if logits_observer is not None:
        logits_observer(step, logits)
    token_id = select_next_token(
        logits,
        valid_mask=valid_mask,
        temperature=float(temperature),
        top_p=top_p,
        top_k=top_k,
        generator=generator,
    )
    states_before.append(state)
    generated.append(token_id)
    next_state = transition(state, token_id, token_index)
    states_after.append(next_state)
    return next_state


def select_next_token(
    logits: torch.Tensor,
    *,
    valid_mask: torch.Tensor,
    temperature: float,
    top_p: float | None = None,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> int:
    flat_logits = torch.as_tensor(logits, dtype=torch.float32).reshape(-1)
    masked = apply_valid_mask(flat_logits, valid_mask)
    if float(temperature) <= 0.0:
        return int(torch.argmax(masked).item())

    scaled = masked / float(temperature)
    if top_k is not None:
        scaled = apply_top_k_to_logits(scaled, k=int(top_k))
    probs = torch.softmax(scaled, dim=-1)
    if top_p is not None:
        probs = apply_top_p(probs, p=float(top_p))
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def apply_valid_mask(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    flat_logits = torch.as_tensor(logits, dtype=torch.float32).reshape(-1)
    mask = valid_mask.to(device=flat_logits.device, dtype=torch.bool).reshape(-1)
    if int(flat_logits.numel()) != int(mask.numel()):
        raise ValueError(f"logits must contain {mask.numel()} values, got {flat_logits.numel()}")
    return flat_logits.masked_fill(~mask, -torch.inf)


def apply_top_k_to_logits(logits: torch.Tensor, *, k: int) -> torch.Tensor:
    if int(k) <= 0:
        raise ValueError(f"top_k must be positive, got {k}")
    if int(k) >= int(logits.numel()):
        return logits
    top_values, top_indices = torch.topk(logits, k=int(k), dim=-1)
    del top_values
    keep = torch.zeros_like(logits, dtype=torch.bool)
    keep[top_indices] = True
    return logits.masked_fill(~keep, -torch.inf)


def apply_top_p(probs: torch.Tensor, *, p: float) -> torch.Tensor:
    if not 0.0 < float(p) <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {p}")
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep_sorted = cumulative <= float(p)
    if keep_sorted.numel() > 0:
        keep_sorted[..., 0] = True
    keep = torch.zeros_like(probs, dtype=torch.bool)
    keep.scatter_(dim=-1, index=sorted_indices, src=keep_sorted)
    filtered = torch.where(keep, probs, torch.zeros_like(probs))
    total = filtered.sum()
    if float(total.item()) <= 0.0:
        return probs
    return filtered / total


def default_generation_logits(
    valid_mask: torch.Tensor,
    *,
    vocab_size: int,
    time_shift_token_ids: Sequence[int] = (),
    time_shift_value: Callable[[int], int] | None = None,
) -> torch.Tensor:
    mask = valid_mask.to(dtype=torch.bool).reshape(-1)
    if int(mask.numel()) != int(vocab_size):
        raise ValueError(f"valid_mask must contain {vocab_size} values, got {mask.numel()}")
    logits = torch.zeros(int(vocab_size), dtype=torch.float32, device=mask.device)
    logits[~mask] = -torch.inf
    valid_time_shifts = [
        int(token_id)
        for token_id in time_shift_token_ids
        if 0 <= int(token_id) < int(mask.numel()) and bool(mask[int(token_id)].item())
    ]
    if valid_time_shifts:
        value_fn = (lambda token_id: int(token_id)) if time_shift_value is None else time_shift_value
        best = max(valid_time_shifts, key=value_fn)
        logits[best] = 1.0
    return logits


def time_shift_penalty_tensors(
    vocab: Any,
    *,
    alpha: float,
    delta_alpha: float = 0.0,
    device: torch.device,
) -> TimeShiftPenalty | None:
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


def apply_time_shift_penalty(
    logits: torch.Tensor,
    *,
    time_shift_penalty: TimeShiftPenalty | None,
) -> torch.Tensor:
    if time_shift_penalty is None:
        return logits
    token_ids, penalties = time_shift_penalty
    adjusted = logits.clone()
    adjusted[token_ids.to(device=adjusted.device)] -= penalties.to(device=adjusted.device, dtype=adjusted.dtype)
    return adjusted


@dataclass
class IncrementalPrefixDecoder:
    create_empty_decode_state: Callable[..., Any]
    batch_size: int
    device: torch.device
    empty_prefix_error: str = "decoder prefix cannot be empty"
    no_logits_error: str = "incremental decode did not produce logits"
    decode_state: Any | None = None
    decoded_prefix_tokens: tuple[int, ...] = ()
    last_incremental_logits: torch.Tensor | None = None

    def reset(self) -> None:
        self.decode_state = self.create_empty_decode_state(batch_size=int(self.batch_size), device=self.device)
        self.decoded_prefix_tokens = ()
        self.last_incremental_logits = None

    def decode(
        self,
        prefix_tokens: Sequence[int],
        *,
        decode_one: Callable[[Any, int], Any],
    ) -> torch.Tensor:
        prefix_tuple = tuple(int(token_id) for token_id in prefix_tokens)
        if not prefix_tuple:
            raise RuntimeError(self.empty_prefix_error)
        if self.decode_state is None or prefix_tuple[: len(self.decoded_prefix_tokens)] != self.decoded_prefix_tokens:
            self.reset()
        if len(prefix_tuple) < len(self.decoded_prefix_tokens):
            self.reset()

        for position in range(len(self.decoded_prefix_tokens), len(prefix_tuple)):
            output = decode_one(self.decode_state, position)
            self.decode_state = output.decode_state
            self.last_incremental_logits = output.logits_final[0].detach()

        self.decoded_prefix_tokens = prefix_tuple
        if self.last_incremental_logits is None:
            raise RuntimeError(self.no_logits_error)
        return self.last_incremental_logits


__all__ = [
    "GenerationEngineResult",
    "IncrementalPrefixDecoder",
    "TimeShiftPenalty",
    "apply_time_shift_penalty",
    "apply_top_k_to_logits",
    "apply_top_p",
    "apply_valid_mask",
    "default_generation_logits",
    "run_generation_engine",
    "select_next_token",
    "time_shift_penalty_tensors",
]
