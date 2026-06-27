from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from pulsefield_model.models.mapper.shared.generation import MapperGenerationStep, transition_carry_state
from pulsefield_model.models.mapper.shared.generation_engine import (
    IncrementalPrefixDecoder,
    apply_time_shift_penalty,
    time_shift_penalty_tensors,
)
from pulsefield_model.models.mapper.shared.replay import LNCarryState, ln_carry_state_tensors
from pulsefield_model.models.mapper.shared.vocab import MapperTupleVocab


def mapper_v2_logits_fn(
    *,
    model: torch.nn.Module,
    vocab: MapperTupleVocab,
    device: torch.device,
    normalized_difficulty: float,
    audio_batch: Mapping[str, torch.Tensor],
    control_batch: Mapping[str, torch.Tensor],
    ln_carry_in: LNCarryState,
    ln_carry_out: LNCarryState,
    is_full_chart_start: bool,
    is_full_chart_end: bool,
    use_incremental_decode: bool,
    time_shift_length_penalty_alpha: float,
):
    time_shift_penalty = time_shift_length_penalty_tensors_v2_tuple(
        vocab,
        alpha=time_shift_length_penalty_alpha,
        device=device,
    )
    incremental_decode = (
        bool(use_incremental_decode)
        and hasattr(model, "create_empty_decode_state")
        and hasattr(model, "incremental_decode_next_token")
    )
    prefix_decoder = None
    if incremental_decode:
        prefix_decoder = IncrementalPrefixDecoder(
            create_empty_decode_state=model.create_empty_decode_state,
            batch_size=1,
            device=device,
            empty_prefix_error="mapper decoder prefix cannot be empty",
            no_logits_error="incremental mapper decode did not produce logits",
        )
    write_start_ms_tensor = torch.tensor([ln_carry_in.current_ms], dtype=torch.long, device=device)
    write_end_ms_tensor: torch.Tensor | None = None
    full_start_tensor = torch.tensor([bool(is_full_chart_start)], dtype=torch.bool, device=device)
    full_end_tensor = torch.tensor([bool(is_full_chart_end)], dtype=torch.bool, device=device)
    difficulty_tensor = torch.tensor([float(normalized_difficulty)], dtype=torch.float32, device=device)
    carry_in_batch = _carry_state_batch(ln_carry_in, device=device)
    carry_out_batch = _carry_state_batch(ln_carry_out, device=device)
    control_attention_kv_cache = control_batch.get("control_attention_kv_cache")

    def logits_fn(step: MapperGenerationStep) -> torch.Tensor:
        nonlocal write_end_ms_tensor

        decoder_input_tokens = step.decoder_input_tokens.to(device=device, dtype=torch.long).unsqueeze(0)
        states = _target_fragment_state_batch(
            generated_tokens=step.generated_tokens,
            vocab=vocab,
            write_start_ms=step.write_start_ms,
            write_end_ms=step.write_end_ms,
            ln_carry_in=ln_carry_in,
            ln_carry_out=ln_carry_out,
            device=device,
        )
        write_end_ms_tensor = torch.tensor([step.write_end_ms], dtype=torch.long, device=device)
        if incremental_decode:
            assert prefix_decoder is not None
            prefix_tokens = tuple(int(token) for token in step.decoder_input_tokens.reshape(-1).tolist())

            def decode_one(decode_state: Any, position: int) -> Any:
                with torch.inference_mode():
                    return model.incremental_decode_next_token(
                        decode_state=decode_state,
                        decoder_input_token=decoder_input_tokens[:, position],
                        current_ms=states["current_ms"][:, position],
                        open_mask=states["open_mask"][:, position],
                        open_start_ms=states["open_start_ms"][:, position],
                        open_age_ms=states["open_age_ms"][:, position],
                        write_start_ms=write_start_ms_tensor,
                        write_end_ms=write_end_ms_tensor,
                        is_full_chart_start=full_start_tensor,
                        is_full_chart_end=full_end_tensor,
                        ln_carry_in=carry_in_batch,
                        ln_carry_out=carry_out_batch,
                        density_teacher_8s=control_batch["density_teacher_8s"],
                        control_memory_8s=control_batch.get("control_memory_8s"),
                        projected_control_memory_8s=control_batch.get("projected_control_memory_8s"),
                        control_attention_kv_cache=control_attention_kv_cache,
                        normalized_difficulty=difficulty_tensor,
                        global_memory=control_batch.get("global_memory"),
                        global_memory_padding_mask=control_batch.get("global_memory_padding_mask"),
                        global_position_features=control_batch.get("global_position_features"),
                        global_attention_kv_cache=control_batch.get("global_attention_kv_cache"),
                        position=position,
                    )

            logits = prefix_decoder.decode(prefix_tokens, decode_one=decode_one)
            return apply_time_shift_length_penalty(
                logits,
                time_shift_penalty=time_shift_penalty,
            )

        current_ms = states["current_ms"]
        target_tokens = torch.full_like(decoder_input_tokens, vocab.pad_id)
        at_write_end = current_ms == int(step.write_end_ms)
        target_tokens = torch.where(at_write_end, torch.full_like(target_tokens, vocab.eos_id), target_tokens)
        batch: dict[str, torch.Tensor | Mapping[str, torch.Tensor]] = {
            **audio_batch,
            **control_batch,
            "decoder_input_tokens": decoder_input_tokens,
            "target_fragment_tokens": target_tokens,
            "target_fragment_mask": torch.ones_like(decoder_input_tokens, dtype=torch.bool),
            "target_fragment_states": states,
            "ln_carry_in": carry_in_batch,
            "ln_carry_out": carry_out_batch,
            "write_start_ms": write_start_ms_tensor,
            "write_end_ms": write_end_ms_tensor,
            "is_full_chart_start": full_start_tensor,
            "is_full_chart_end": full_end_tensor,
            "normalized_difficulty": difficulty_tensor,
        }
        with torch.inference_mode():
            output = model(batch)
        logits = output.logits_final[0, -1].detach()
        return apply_time_shift_length_penalty(logits, time_shift_penalty=time_shift_penalty)

    return logits_fn


def time_shift_length_penalty_tensors_v2_tuple(
    vocab: MapperTupleVocab,
    *,
    alpha: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    return time_shift_penalty_tensors(vocab, alpha=float(alpha), device=device)


def apply_time_shift_length_penalty(
    logits: torch.Tensor,
    *,
    time_shift_penalty: tuple[torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor:
    return apply_time_shift_penalty(logits, time_shift_penalty=time_shift_penalty)


def _target_fragment_state_batch(
    *,
    generated_tokens: Sequence[int],
    vocab: MapperTupleVocab,
    write_start_ms: int,
    write_end_ms: int,
    ln_carry_in: LNCarryState,
    ln_carry_out: LNCarryState,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    states = [ln_carry_in]
    state = ln_carry_in
    for token_id in generated_tokens:
        state = transition_carry_state(
            state,
            int(token_id),
            vocab=vocab,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            allow_bos=False,
            allow_eos=False,
        )
        states.append(state)
    if states[-1] != ln_carry_out and int(states[-1].current_ms) == int(write_end_ms):
        raise ValueError("generated prefix reached write_end_ms without matching ln_carry_out")
    tensors = [_carry_state_tensors_1d(state, device=device) for state in states]
    return {
        "current_ms": torch.stack([item["current_ms"] for item in tensors], dim=0).unsqueeze(0),
        "open_mask": torch.stack([item["open_mask"] for item in tensors], dim=0).unsqueeze(0),
        "open_start_ms": torch.stack([item["open_start_ms"] for item in tensors], dim=0).unsqueeze(0),
        "open_age_ms": torch.stack([item["open_age_ms"] for item in tensors], dim=0).unsqueeze(0),
    }


def _carry_state_batch(carry: LNCarryState, *, device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.unsqueeze(0) for key, value in _carry_state_tensors_1d(carry, device=device).items()}


def _carry_state_tensors_1d(carry: LNCarryState, *, device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device=device) for key, value in ln_carry_state_tensors(carry).items()}
