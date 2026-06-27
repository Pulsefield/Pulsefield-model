from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import math
import os
import random
import time
import wave
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import torch

from pulsefield_model.data.control_windows import normalize_difficulty
from pulsefield_model.events.canonical import CanonicalTimepoint, LaneAction as CanonicalLaneAction
from pulsefield_model.inference.mapper_protocol import (
    HitObjectToken,
    MapperProtocolTranslator,
    build_mapper_protocol_translator,
    infer_mapper_profile_name_from_vocab,
    resolve_mapper_profile,
)
from pulsefield_model.inference.mapper_v2_1_rollout import (
    grammar_constrained_window_generation_v2_1,
    mapper_v2_1_logits_fn,
)
from pulsefield_model.inference.model_runtime import (
    ModelRuntime,
    ModelRuntimeConfig,
    load_model_runtime,
    release_torch_cache,
)
from pulsefield_model.inference.osu_export import OsuExportMetadata, format_osu_export
from pulsefield_model.inference.session_runtime import (
    DEFAULT_MAX_CONTROL_BATCH_SIZE,
    SessionRuntime,
    SessionRuntimeConfig,
)
from pulsefield_model.models.mapper.shared.generation import (
    MapperGeneratedWindow,
    MapperGenerationStep,
    grammar_constrained_window_generation,
    transition_carry_state,
)
from pulsefield_model.models.mapper.shared.generation_engine import (
    IncrementalPrefixDecoder,
    apply_time_shift_penalty,
    time_shift_penalty_tensors,
)
from pulsefield_model.models.mapper.shared.replay import LNCarryState, empty_ln_carry_state, ln_carry_state_tensors
from pulsefield_model.models.mapper.shared.tokenizer import MAPPER_WRITE_MS
from pulsefield_model.models.mapper.shared.vocab import MapperTupleVocab
from pulsefield_model.models.mapper.v2_1.replay import (
    LNCarryState as MapperV21LNCarryState,
    empty_ln_carry_state as empty_ln_carry_state_v2_1,
)
from pulsefield_model.models.mapper.v2_1.vocab import MapperV21Vocab
from pulsefield_model.timing.canonicalization import (
    TIMING_CANONICALIZATION_BPM_80_160,
    TIMING_CANONICALIZATION_CHOICES,
    TIMING_CANONICALIZATION_NONE,
)
from pulsefield_model.timing.grid_fitting import GridFitterConfig
from pulsefield_model.timing.providers.beatthis import DEFAULT_BEATTHIS_DEVICE


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MAPPER_V2_CHECKPOINT_PATH = Path(
    "artifacts/runs/stage2_mapper_v2/"
    "stage2_mapper_v2_phase_b_global_d768_l8_b1/checkpoint.pt",
)
DEFAULT_MAPPER_V2_1_CHECKPOINT_PATH = Path(
    "artifacts/runs/stage2_mapper_v2_1/"
    "stage2_mapper_v2_1_phase_b_sparse_global_d384_l4_b2/checkpoint.pt",
)
DEFAULT_MAPPER_CHECKPOINT_PATH = DEFAULT_MAPPER_V2_1_CHECKPOINT_PATH
DEFAULT_CONTROL_CHECKPOINT_PATH = Path(
    "artifacts/runs/stage2_control_demo/"
    "stage2_control_demo_global_d384_l3_stride16_b6/checkpoints/checkpoint_step_002000.pt",
)
DEFAULT_TIME_SHIFT_LENGTH_PENALTY = 5.2
DEFAULT_INDEX_PATH = Path("artifacts/indexes/stage2_control_windows_4k_2to6_dense_local_bpm_norm_unique_le3.parquet")
DEFAULT_DATASET_ROOT = Path("dataset")
DEFAULT_OUTPUT_DIR = Path("artifacts/inference/mapper_v2_1_cached_stream_random_diff4")
T = TypeVar("T")


@dataclass(frozen=True)
class StreamWithCacheConfig:
    decoder_window_ms: int = MAPPER_WRITE_MS
    token_send_interval_s: float = 0.02
    mapper_checkpoint_path: str | Path = DEFAULT_MAPPER_CHECKPOINT_PATH
    control_checkpoint_path: str | Path = DEFAULT_CONTROL_CHECKPOINT_PATH
    mapper_profile: str = "auto"
    device: str = "auto"
    beatthis_device: str | None = DEFAULT_BEATTHIS_DEVICE
    beatthis_float16: bool = False
    eager_load_beatthis: bool = True
    canonicalization: str = TIMING_CANONICALIZATION_NONE
    default_difficulty: float = 4.0
    max_control_batch_size: int = DEFAULT_MAX_CONTROL_BATCH_SIZE
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float | None = None
    use_incremental_mapper_decode: bool = True
    time_shift_length_penalty_alpha: float = DEFAULT_TIME_SHIFT_LENGTH_PENALTY
    seed: int | None = None


@dataclass(frozen=True)
class DecoderWindow:
    start_ms: int
    end_ms: int


RuntimeLoader = Callable[[ModelRuntimeConfig], ModelRuntime]
SessionRuntimeFactory = Callable[[str, ModelRuntime, SessionRuntimeConfig], SessionRuntime]


class StreamWithCache:
    """Cached full-song mapper stream used by endpoint and offline inference."""

    def __init__(
        self,
        config: StreamWithCacheConfig,
        *,
        runtime_loader: RuntimeLoader = load_model_runtime,
        session_runtime_factory: SessionRuntimeFactory | None = None,
    ) -> None:
        self.config = config
        self.models_ready = False
        self.model_runtime: ModelRuntime | None = None
        self._runtime_loader = runtime_loader
        self._session_runtime_factory = (
            _default_session_runtime_factory if session_runtime_factory is None else session_runtime_factory
        )
        self._session_runtimes: dict[str, SessionRuntime] = {}
        self._last_context_token_by_session: dict[str, int] = {}
        self._last_carry_state_by_session: dict[str, Any] = {}
        self._protocol_translators_by_session: dict[str, MapperProtocolTranslator] = {}

    async def startup(self) -> None:
        if self.models_ready:
            return
        self.model_runtime = await asyncio.to_thread(self._load_model_runtime)
        self.models_ready = True

    async def prepare_audio(
        self,
        *,
        session_id: str,
        audio_path: Path,
        audio_length_ms: int,
        difficulty: float | None,
        route: str = "mapper",
    ) -> None:
        if route != "mapper":
            raise ValueError(f"StreamWithCache only supports mapper route, got {route!r}")
        model_runtime = self._require_model_runtime()
        normalized_difficulty = normalize_difficulty(
            self.config.default_difficulty if difficulty is None else float(difficulty),
        )
        session_runtime = self._session_runtime_factory(
            session_id,
            model_runtime,
            SessionRuntimeConfig(
                device=self.config.device,
                default_normalized_difficulty=normalized_difficulty,
                max_control_batch_size=int(self.config.max_control_batch_size),
                grid_fitter_config=_grid_fitter_config_for_canonicalization(self.config.canonicalization),
            ),
        )
        await asyncio.to_thread(
            session_runtime.prepare_audio,
            audio_path,
            audio_length_ms=audio_length_ms,
            start_ms=0,
        )
        self._session_runtimes[session_id] = session_runtime
        self._last_context_token_by_session.pop(session_id, None)
        self._last_carry_state_by_session.pop(session_id, None)
        self._protocol_translators_by_session.pop(session_id, None)

    async def iter_hitobject_tokens(
        self,
        *,
        session_id: str,
        audio_path: Path,
        audio_length_ms: int,
        window: DecoderWindow,
    ) -> AsyncIterator[HitObjectToken]:
        del audio_path
        session_runtime = self._session_runtimes.get(session_id)
        if session_runtime is None:
            raise RuntimeError(f"session audio has not been prepared: {session_id}")
        starting_window = clamp_decoder_window_to_audio(window, audio_length_ms=audio_length_ms, config=self.config)
        for decode_window in decoder_windows_until_audio_end(
            starting_window,
            audio_length_ms=audio_length_ms,
            config=self.config,
        ):
            translator = self._protocol_translator(session_id)
            generated = await asyncio.to_thread(
                self._generate_window,
                session_id,
                session_runtime,
                decode_window,
                audio_length_ms,
            )
            for token in translator.consume_window(generated):
                if int(token.ms_in_ref_audio) >= int(audio_length_ms):
                    continue
                yield token
                interval = max(0.0, float(self.config.token_send_interval_s))
                if interval:
                    await asyncio.sleep(interval)

    async def reset_session(self, session_id: str) -> None:
        session_runtime = self._session_runtimes.pop(session_id, None)
        self._last_context_token_by_session.pop(session_id, None)
        self._last_carry_state_by_session.pop(session_id, None)
        self._protocol_translators_by_session.pop(session_id, None)
        if session_runtime is not None:
            await asyncio.to_thread(session_runtime.reset_audio_cache)

    def _load_model_runtime(self) -> ModelRuntime:
        return self._runtime_loader(
            ModelRuntimeConfig(
                mapper_checkpoint_path=_resolve_repo_path(self.config.mapper_checkpoint_path),
                control_checkpoint_path=_resolve_repo_path(self.config.control_checkpoint_path),
                mapper_profile=self.config.mapper_profile,
                device=self.config.device,
                beatthis_device=self.config.beatthis_device,
                beatthis_float16=bool(self.config.beatthis_float16),
                eager_load_beatthis=bool(self.config.eager_load_beatthis),
            ),
        )

    def _generate_window(
        self,
        session_id: str,
        session_runtime: SessionRuntime,
        window: DecoderWindow,
        audio_length_ms: int,
    ) -> Any:
        if session_runtime.audio_cache is None:
            raise RuntimeError("prepare_audio must finish before mapper generation")
        write_start_ms = int(window.start_ms)
        write_end_ms = int(window.end_ms)
        if write_end_ms - write_start_ms != int(self.config.decoder_window_ms):
            raise ValueError("decoder window span does not match config.decoder_window_ms")
        if self._mapper_profile_name() == "v2_1_sparse":
            return self._generate_window_v2_1(
                session_id=session_id,
                session_runtime=session_runtime,
                window=window,
                audio_length_ms=audio_length_ms,
            )

        mapper_window_cache = session_runtime.prepare_mapper_window(
            start_ms=write_start_ms,
            end_ms=write_end_ms,
            include_control_attention_kv_cache=bool(self.config.use_incremental_mapper_decode),
        )

        vocab = self._vocab()
        carry_in = self._carry_in_for_window(session_id, write_start_ms)
        carry_out = empty_ln_carry_state(write_end_ms)
        is_full_chart_start = write_start_ms == 0 and not any(carry_in.open_mask)
        is_full_chart_end = write_end_ms >= int(audio_length_ms)
        left_context_tokens: tuple[int, ...] = ()
        if not is_full_chart_start:
            left_context_tokens = (self._left_context_token(session_id, vocab),)

        generator = _make_torch_generator(self.config.seed, device=session_runtime.device)
        logits_fn = _mapper_v2_logits_fn(
            model=session_runtime.model_runtime.mapper_model,
            vocab=vocab,
            device=session_runtime.device,
            normalized_difficulty=mapper_window_cache.normalized_difficulty,
            audio_batch={},
            control_batch=mapper_window_cache.as_model_batch(),
            ln_carry_in=carry_in,
            ln_carry_out=carry_out,
            is_full_chart_start=is_full_chart_start,
            is_full_chart_end=is_full_chart_end,
            use_incremental_decode=bool(self.config.use_incremental_mapper_decode),
            time_shift_length_penalty_alpha=float(self.config.time_shift_length_penalty_alpha),
        )
        generated = grammar_constrained_window_generation(
            vocab=vocab,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            ln_carry_in=carry_in,
            ln_carry_out=carry_out,
            logits_fn=logits_fn,
            left_context_tokens=left_context_tokens,
            is_full_chart_start=is_full_chart_start,
            is_full_chart_end=is_full_chart_end,
            max_tokens=int(self.config.max_tokens),
            temperature=float(self.config.temperature),
            top_p=self.config.top_p,
            generator=generator,
        )
        if generated.tokens:
            self._last_context_token_by_session[session_id] = int(generated.tokens[-1])
        if generated.completed:
            self._last_carry_state_by_session[session_id] = generated.terminal_state
        return generated

    def _generate_window_v2_1(
        self,
        *,
        session_id: str,
        session_runtime: SessionRuntime,
        window: DecoderWindow,
        audio_length_ms: int,
    ) -> Any:
        if session_runtime.audio_cache is None:
            raise RuntimeError("prepare_audio must finish before mapper generation")
        write_start_ms = int(window.start_ms)
        write_end_ms = int(window.end_ms)
        mapper_window_cache = session_runtime.prepare_mapper_window(
            start_ms=write_start_ms,
            end_ms=write_end_ms,
            include_control_attention_kv_cache=bool(self.config.use_incremental_mapper_decode),
        )

        vocab = self._vocab()
        if not isinstance(vocab, MapperV21Vocab):
            raise TypeError("v2_1_sparse mapper profile requires MapperV21Vocab")
        carry_in = self._carry_in_for_window(session_id, write_start_ms)
        if not isinstance(carry_in, MapperV21LNCarryState):
            raise TypeError("v2_1_sparse mapper profile requires MapperV21LNCarryState carry")
        chart_end_ms = _chart_end_ms_for_generation(audio_length_ms)
        is_full_chart_start = write_start_ms == 0 and not any(carry_in.open_mask)
        is_full_chart_end = write_end_ms >= chart_end_ms
        target_end_ms = chart_end_ms if is_full_chart_end else write_end_ms
        carry_out = empty_ln_carry_state_v2_1(target_end_ms)
        left_context_tokens: tuple[int, ...] = ()
        if not is_full_chart_start:
            left_context_tokens = (self._left_context_token(session_id, vocab),)

        generator = _make_torch_generator(self.config.seed, device=session_runtime.device)
        logits_fn = mapper_v2_1_logits_fn(
            model=session_runtime.model_runtime.mapper_model,
            vocab=vocab,
            device=session_runtime.device,
            normalized_difficulty=mapper_window_cache.normalized_difficulty,
            control_batch=mapper_window_cache.as_model_batch(),
            ln_carry_in=carry_in,
            ln_carry_out=carry_out,
            write_start_ms=write_start_ms,
            write_end_ms=write_end_ms,
            chart_end_ms=chart_end_ms,
            is_full_chart_start=is_full_chart_start,
            is_full_chart_end=is_full_chart_end,
            time_shift_length_penalty_alpha=float(self.config.time_shift_length_penalty_alpha),
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
            max_tokens=int(self.config.max_tokens),
            temperature=float(self.config.temperature),
            top_p=self.config.top_p,
            generator=generator,
        )
        if generated.tokens:
            self._last_context_token_by_session[session_id] = int(generated.tokens[-1])
        if generated.completed:
            self._last_carry_state_by_session[session_id] = _mapper_v2_1_carry_from_replay_state(
                generated.terminal_state,
            )
        return generated

    def _carry_in_for_window(self, session_id: str, write_start_ms: int) -> Any:
        previous = self._last_carry_state_by_session.get(session_id)
        if previous is not None and int(previous.current_ms) == int(write_start_ms):
            return previous
        if self._mapper_profile_name() == "v2_1_sparse":
            return empty_ln_carry_state_v2_1(write_start_ms)
        return empty_ln_carry_state(write_start_ms)

    def _left_context_token(self, session_id: str, vocab: Any) -> int:
        token = self._last_context_token_by_session.get(session_id)
        if token is not None and token != vocab.bos_id:
            return int(token)
        return int(vocab.time_shift_token_id(10))

    def _vocab(self) -> Any:
        runtime = self._require_model_runtime()
        return runtime.vocab

    def _mapper_profile_name(self) -> str:
        runtime = self.model_runtime
        if runtime is not None and getattr(runtime, "mapper_profile", None) is not None:
            return runtime.mapper_profile.name
        requested = str(getattr(self.config, "mapper_profile", "auto"))
        if requested.strip().lower() != "auto":
            return resolve_mapper_profile(requested).name
        return infer_mapper_profile_name_from_vocab(self._vocab())

    def _protocol_translator(self, session_id: str) -> MapperProtocolTranslator:
        profile_name = self._mapper_profile_name()
        translator = self._protocol_translators_by_session.get(session_id)
        if translator is not None and translator.profile_name == profile_name:
            return translator
        translator = build_mapper_protocol_translator(profile_name, source_vocab=self._vocab())
        self._protocol_translators_by_session[session_id] = translator
        return translator

    def _require_model_runtime(self) -> ModelRuntime:
        if self.model_runtime is None:
            raise RuntimeError("models are not loaded; call startup first")
        return self.model_runtime


def _default_session_runtime_factory(
    session_id: str,
    model_runtime: ModelRuntime,
    config: SessionRuntimeConfig,
) -> SessionRuntime:
    return SessionRuntime(session_id=session_id, model_runtime=model_runtime, config=config)


def _resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    cwd_path = Path.cwd() / resolved
    if cwd_path.exists():
        return cwd_path
    return REPOSITORY_ROOT / resolved


def _make_torch_generator(seed: int | None, *, device: torch.device) -> torch.Generator | None:
    if seed is None:
        return None
    generator_device = "cpu" if device.type == "mps" else device.type
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    return generator


def _chart_end_ms_for_generation(audio_length_ms: int) -> int:
    audio_length_ms = max(1, int(audio_length_ms))
    return max(10, int(math.ceil(audio_length_ms / 10.0) * 10))


def _mapper_v2_1_carry_from_replay_state(state: Any) -> MapperV21LNCarryState:
    return MapperV21LNCarryState(
        current_ms=int(state.current_ms),
        open_mask=tuple(bool(value) for value in state.open_mask),  # type: ignore[arg-type]
        open_start_ms=tuple(None if value is None else int(value) for value in state.open_start_ms),  # type: ignore[arg-type]
        open_age_ms=tuple(int(value) for value in state.open_age_ms),  # type: ignore[arg-type]
    )


def _hitobject_tokens_from_generated(
    generated: MapperGeneratedWindow,
    vocab: Any,
) -> tuple[HitObjectToken, ...]:
    profile_name = infer_mapper_profile_name_from_vocab(vocab)
    return build_mapper_protocol_translator(profile_name, source_vocab=vocab).consume_window(generated)


def _mapper_v2_logits_fn(
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
    time_shift_penalty = _time_shift_length_penalty_tensors(
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
            return _apply_time_shift_length_penalty(
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
        return _apply_time_shift_length_penalty(logits, time_shift_penalty=time_shift_penalty)

    return logits_fn


def _time_shift_length_penalty_tensors(
    vocab: MapperTupleVocab,
    *,
    alpha: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    return time_shift_penalty_tensors(vocab, alpha=float(alpha), device=device)


def _apply_time_shift_length_penalty(
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
    return {
        key: value.to(device=device)
        for key, value in ln_carry_state_tensors(carry).items()
    }


def clamp_decoder_window_to_audio(
    window: DecoderWindow,
    *,
    audio_length_ms: int,
    config: StreamWithCacheConfig,
) -> DecoderWindow:
    window_ms = int(config.decoder_window_ms)
    if window_ms <= 0:
        raise ValueError("decoder_window_ms must be positive")
    latest_start_ms = (max(1, int(audio_length_ms)) - 1) // window_ms * window_ms
    if int(window.start_ms) <= latest_start_ms:
        return window
    return DecoderWindow(start_ms=latest_start_ms, end_ms=latest_start_ms + window_ms)


def decoder_windows_until_audio_end(
    window: DecoderWindow,
    *,
    audio_length_ms: int,
    config: StreamWithCacheConfig,
) -> tuple[DecoderWindow, ...]:
    window_ms = int(config.decoder_window_ms)
    if window_ms <= 0:
        raise ValueError("decoder_window_ms must be positive")
    start_ms = int(window.start_ms)
    end_ms = int(window.end_ms)
    if end_ms - start_ms != window_ms:
        raise ValueError("decoder window span does not match config.decoder_window_ms")
    latest_start_ms = (max(1, int(audio_length_ms)) - 1) // window_ms * window_ms
    if start_ms > latest_start_ms:
        start_ms = latest_start_ms
    return tuple(
        DecoderWindow(start_ms=current_start_ms, end_ms=current_start_ms + window_ms)
        for current_start_ms in range(start_ms, latest_start_ms + 1, window_ms)
    )


def audio_length_ms_from_file(audio_path: Path) -> int | None:
    if not audio_path.exists():
        return None
    try:
        from mutagen import File as MutagenFile
    except ImportError:
        pass
    else:
        try:
            audio = MutagenFile(audio_path)
        except Exception:
            audio = None
        if audio is not None and getattr(audio, "info", None) is not None:
            length = getattr(audio.info, "length", None)
            if isinstance(length, (int, float)) and math.isfinite(float(length)) and float(length) > 0:
                return int(round(float(length) * 1000.0))
    try:
        with wave.open(str(audio_path), "rb") as audio:
            frame_count = int(audio.getnframes())
            frame_rate = int(audio.getframerate())
    except (EOFError, OSError, wave.Error):
        return None
    if frame_count <= 0 or frame_rate <= 0:
        return None
    return int(round(frame_count / frame_rate * 1000.0))


@dataclass(frozen=True)
class CandidateMap:
    row_index: int
    shard: str
    beatmap_id: int | None
    beatmap_set_id: int | None
    beatmap_path: Path
    audio_path: Path
    audio_filename: str
    title: str
    artist: str
    creator: str
    version: str
    hp_drain_rate: float
    overall_difficulty: float
    frame_count: int
    duration_s: float


def run_cached_stream_sample(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    difficulty = float(args.difficulty)
    normalize_difficulty(difficulty)

    seed = int(args.seed if args.seed is not None else time.time_ns() % (2**32))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidate_maps(
        index_path=Path(args.index_path),
        dataset_root=Path(args.dataset_root),
        min_duration_s=args.min_duration_s,
        max_duration_s=args.max_duration_s,
    )
    if len(candidates) < int(args.count):
        raise ValueError(f"not enough candidate maps after filtering: {len(candidates)} < {args.count}")
    sampled = sample_candidates(candidates, count=int(args.count), seed=seed)

    print(
        "inference_progress "
        f"status=sampled seed={seed} count={len(sampled)} pool={len(candidates)} "
        f"difficulty={difficulty:.3f} max_duration_s={args.max_duration_s}",
        flush=True,
    )
    for index, candidate in enumerate(sampled, start=1):
        print(
            "selected_map "
            f"index={index}/{len(sampled)} beatmap_id={candidate.beatmap_id} "
            f"duration_s={candidate.duration_s:.2f} audio={candidate.audio_path.as_posix()} "
            f"title={json.dumps(candidate.title, ensure_ascii=False)}",
            flush=True,
        )

    device = str(args.device)
    runtime = run_with_heartbeat(
        "runtime_load",
        lambda: load_model_runtime(
            ModelRuntimeConfig(
                mapper_checkpoint_path=Path(args.mapper_checkpoint_path),
                control_checkpoint_path=Path(args.control_checkpoint_path),
                mapper_profile=args.mapper_profile,
                device=device,
                beatthis_device=args.beatthis_device,
                beatthis_float16=bool(args.beatthis_float16),
                eager_load_beatthis=bool(args.eager_load_beatthis),
            ),
        ),
        interval_s=float(args.progress_interval_s),
    )
    config = StreamWithCacheConfig(
        mapper_checkpoint_path=Path(args.mapper_checkpoint_path),
        control_checkpoint_path=Path(args.control_checkpoint_path),
        mapper_profile=args.mapper_profile,
        device=device,
        beatthis_device=args.beatthis_device,
        beatthis_float16=bool(args.beatthis_float16),
        eager_load_beatthis=bool(args.eager_load_beatthis),
        default_difficulty=difficulty,
        max_control_batch_size=int(args.control_batch_size),
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        top_p=args.top_p,
        use_incremental_mapper_decode=bool(args.use_incremental_mapper_decode),
        time_shift_length_penalty_alpha=float(args.time_shift_length_penalty_alpha),
        seed=args.generation_seed,
        token_send_interval_s=0.0,
        canonicalization=args.canonicalization,
    )
    stream = StreamWithCache(config)
    stream.model_runtime = runtime
    stream.models_ready = True

    reports = [
        run_candidate(
            stream=stream,
            candidate=candidate,
            index=index,
            total=len(sampled),
            difficulty=difficulty,
            output_dir=output_dir,
            device=device,
            control_batch_size=int(args.control_batch_size),
            precompute_full_control=bool(args.precompute_full_control),
            progress_interval_s=float(args.progress_interval_s),
        )
        for index, candidate in enumerate(sampled, start=1)
    ]

    manifest_path = output_dir / f"manifest_seed{seed}_diff{_difficulty_slug(difficulty)}.json"
    manifest_path.write_text(
        json.dumps(
            {
                "seed": seed,
                "difficulty": difficulty,
                "mapper_checkpoint_path": Path(args.mapper_checkpoint_path).as_posix(),
                "control_checkpoint_path": Path(args.control_checkpoint_path).as_posix(),
                "mapper_profile": args.mapper_profile,
                "device": device,
                "canonicalization": args.canonicalization,
                "count": len(reports),
                "reports": reports,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"inference_progress status=done manifest={manifest_path.as_posix()}", flush=True)
    return 0


def run_candidate(
    *,
    stream: StreamWithCache,
    candidate: CandidateMap,
    index: int,
    total: int,
    difficulty: float,
    output_dir: Path,
    device: str,
    control_batch_size: int,
    precompute_full_control: bool,
    progress_interval_s: float,
) -> dict[str, Any]:
    session_id = f"offline-{index}-{candidate.beatmap_id or candidate.row_index}"
    audio_length_ms = audio_length_ms_from_file(candidate.audio_path)
    audio_length_source = "file"
    if audio_length_ms is None:
        audio_length_ms = max(1, int(round(candidate.frame_count * 20.0)))
        audio_length_source = "frame_count"

    print(
        "map_progress "
        f"index={index}/{total} status=start beatmap_id={candidate.beatmap_id} "
        f"duration_s={audio_length_ms / 1000.0:.2f} audio_length_source={audio_length_source}",
        flush=True,
    )

    assert stream.model_runtime is not None
    session_runtime = SessionRuntime(
        session_id=session_id,
        model_runtime=stream.model_runtime,
        config=SessionRuntimeConfig(
            device=device,
            default_normalized_difficulty=normalize_difficulty(difficulty),
            max_control_batch_size=control_batch_size,
            grid_fitter_config=_grid_fitter_config_for_canonicalization(stream.config.canonicalization),
        ),
    )
    stream._session_runtimes[session_id] = session_runtime
    stream._last_context_token_by_session.pop(session_id, None)
    stream._last_carry_state_by_session.pop(session_id, None)

    try:
        audio_cache = run_with_heartbeat(
            f"map{index}_prepare_audio",
            lambda: session_runtime.prepare_audio(candidate.audio_path, audio_length_ms=audio_length_ms, start_ms=0),
            interval_s=progress_interval_s,
        )
        print(
            "map_progress "
            f"index={index}/{total} status=audio_ready source_frames={audio_cache.source_frame_count} "
            f"padded_frames={audio_cache.padded_frame_count}",
            flush=True,
        )

        if precompute_full_control:
            full_control = run_with_heartbeat(
                f"map{index}_full_control",
                lambda: session_runtime.prepare_full_control(max_batch_size=control_batch_size),
                interval_s=progress_interval_s,
            )
            print(
                "map_progress "
                f"index={index}/{total} status=control_ready windows={len(full_control.start_ms_values)} "
                f"batch_size={full_control.max_batch_size}",
                flush=True,
            )

        windows = decoder_windows_until_audio_end(
            DecoderWindow(start_ms=0, end_ms=MAPPER_WRITE_MS),
            audio_length_ms=audio_length_ms,
            config=stream.config,
        )
        generated_windows = []
        window_reports = []
        started_at = time.monotonic()
        for window_index, window in enumerate(windows, start=1):
            print(
                "window_progress "
                f"map={index}/{total} window={window_index}/{len(windows)} status=start "
                f"start_ms={window.start_ms} end_ms={window.end_ms}",
                flush=True,
            )
            window_started_at = time.monotonic()
            generated = run_with_heartbeat(
                f"map{index}_window{window_index:03d}",
                lambda window=window: stream._generate_window(
                    session_id,
                    session_runtime,
                    window,
                    audio_length_ms,
                ),
                interval_s=progress_interval_s,
            )
            elapsed_s = time.monotonic() - window_started_at
            generated_windows.append(generated)
            event_count = sum(1 for token_id in generated.tokens if stream._vocab().is_event_token(token_id))
            window_report = {
                "window_index": window_index,
                "start_ms": int(window.start_ms),
                "end_ms": int(window.end_ms),
                "tokens": len(generated.tokens),
                "events": event_count,
                "completed": bool(generated.completed),
                "dead_end": bool(generated.dead_end),
                "max_tokens_exceeded": bool(generated.max_tokens_exceeded),
                "terminal_ms": int(generated.terminal_state.current_ms),
                "elapsed_s": elapsed_s,
            }
            window_reports.append(window_report)
            print(
                "window_progress "
                f"map={index}/{total} window={window_index}/{len(windows)} status=done "
                f"tokens={len(generated.tokens)} events={event_count} completed={int(generated.completed)} "
                f"dead_end={int(generated.dead_end)} max_tokens_exceeded={int(generated.max_tokens_exceeded)} "
                f"terminal_ms={generated.terminal_state.current_ms} elapsed_s={elapsed_s:.2f}",
                flush=True,
            )

        timepoints = generated_windows_to_timepoints(generated_windows, stream._vocab())
        output_path = output_dir / output_filename(candidate, index=index, difficulty=difficulty)
        metadata = OsuExportMetadata(
            audio_filename=relative_audio_filename(candidate.audio_path, output_path),
            title=candidate.title,
            artist=candidate.artist,
            creator="Mapperatorinator",
            version=f"Mapper V2 cached stream diff {difficulty:.2f} / {candidate.version}",
            difficulty=difficulty,
            hp_drain_rate=candidate.hp_drain_rate,
            overall_difficulty=candidate.overall_difficulty,
        )
        timing_grid = session_runtime.audio_cache.timing_grid if session_runtime.audio_cache is not None else None
        output_path.write_text(
            format_osu_export(timepoints=timepoints, metadata=metadata, timing_grid=timing_grid),
            encoding="utf-8",
        )

        total_elapsed_s = time.monotonic() - started_at
        completed_windows = sum(1 for item in window_reports if item["completed"])
        report = {
            "status": "ok",
            "beatmap_id": candidate.beatmap_id,
            "beatmap_set_id": candidate.beatmap_set_id,
            "title": candidate.title,
            "artist": candidate.artist,
            "source_version": candidate.version,
            "audio_path": candidate.audio_path.as_posix(),
            "output_path": output_path.as_posix(),
            "audio_length_ms": audio_length_ms,
            "audio_length_source": audio_length_source,
            "window_count": len(windows),
            "completed_windows": completed_windows,
            "timepoint_count": len(timepoints),
            "elapsed_s": total_elapsed_s,
            "window_reports": window_reports,
        }
        print(
            "map_progress "
            f"index={index}/{total} status=done output={output_path.as_posix()} "
            f"windows={completed_windows}/{len(windows)} timepoints={len(timepoints)} elapsed_s={total_elapsed_s:.2f}",
            flush=True,
        )
        return report
    except Exception as exc:
        error_path = output_dir / f"FAILED_{index:02d}_{candidate.beatmap_id or candidate.row_index}.json"
        report = {
            "status": "error",
            "beatmap_id": candidate.beatmap_id,
            "beatmap_set_id": candidate.beatmap_set_id,
            "title": candidate.title,
            "artist": candidate.artist,
            "source_version": candidate.version,
            "audio_path": candidate.audio_path.as_posix(),
            "error": str(exc),
        }
        error_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            "map_progress "
            f"index={index}/{total} status=error beatmap_id={candidate.beatmap_id} "
            f"error={json.dumps(str(exc), ensure_ascii=False)} details={error_path.as_posix()}",
            flush=True,
        )
        return report
    finally:
        stream._session_runtimes.pop(session_id, None)
        stream._last_context_token_by_session.pop(session_id, None)
        stream._last_carry_state_by_session.pop(session_id, None)
        session_runtime.reset_audio_cache()
        release_torch_cache(device)


def load_candidate_maps(
    *,
    index_path: Path,
    dataset_root: Path,
    min_duration_s: float | None,
    max_duration_s: float | None,
) -> list[CandidateMap]:
    import pandas as pd

    frame = pd.read_parquet(index_path)
    required = {
        "shard",
        "beatmap_path",
        "audio_path",
        "audio_filename",
        "title",
        "artist",
        "creator",
        "version",
        "beatmap_id",
        "beatmap_set_id",
        "hp_drain_rate",
        "overall_difficulty",
        "frame_count",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"index is missing required columns: {missing}")

    unique = frame.drop_duplicates(["beatmap_path"]).reset_index(drop=True)
    candidates: list[CandidateMap] = []
    for row_index, row in unique.iterrows():
        frame_count = _optional_int(row["frame_count"])
        if frame_count is None or frame_count <= 0:
            continue
        duration_s = float(frame_count) * 0.02
        if min_duration_s is not None and duration_s < float(min_duration_s):
            continue
        if max_duration_s is not None and duration_s > float(max_duration_s):
            continue
        shard = str(row["shard"])
        audio_path = resolve_dataset_path(dataset_root, shard, row["audio_path"])
        beatmap_path = resolve_dataset_path(dataset_root, shard, row["beatmap_path"])
        if not audio_path.is_file() or not beatmap_path.is_file():
            continue
        candidates.append(
            CandidateMap(
                row_index=int(row_index),
                shard=shard,
                beatmap_id=_optional_int(row["beatmap_id"]),
                beatmap_set_id=_optional_int(row["beatmap_set_id"]),
                beatmap_path=beatmap_path,
                audio_path=audio_path,
                audio_filename=str(row["audio_filename"]),
                title=_clean_text(row["title"], "Unknown Title"),
                artist=_clean_text(row["artist"], "Unknown Artist"),
                creator=_clean_text(row["creator"], "Unknown Creator"),
                version=_clean_text(row["version"], "Generated"),
                hp_drain_rate=_finite_float(row["hp_drain_rate"], default=5.0),
                overall_difficulty=_finite_float(row["overall_difficulty"], default=5.0),
                frame_count=frame_count,
                duration_s=duration_s,
            ),
        )
    return candidates


def sample_candidates(candidates: Sequence[CandidateMap], *, count: int, seed: int) -> list[CandidateMap]:
    return random.Random(seed).sample(list(candidates), k=count)


def generated_windows_to_timepoints(generated_windows: Sequence[Any], vocab: Any) -> list[CanonicalTimepoint]:
    timepoints: list[CanonicalTimepoint] = []
    for generated in generated_windows:
        for token_id, state_before in zip(generated.tokens, generated.states_before, strict=True):
            token_id = int(token_id)
            if not vocab.is_event_token(token_id):
                continue
            lane_actions = tuple(CanonicalLaneAction(action.value) for action in vocab.decode_event(token_id))
            timepoints.append(CanonicalTimepoint(time_ms=int(state_before.current_ms), lane_actions=lane_actions))
    return sorted(timepoints, key=lambda item: item.time_ms)


def run_with_heartbeat(label: str, fn: Callable[[], T], *, interval_s: float) -> T:
    interval_s = max(1.0, float(interval_s))
    started_at = time.monotonic()
    print(f"progress label={label} status=started", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        while True:
            try:
                result = future.result(timeout=interval_s)
            except concurrent.futures.TimeoutError:
                print(
                    f"progress label={label} status=running elapsed_s={time.monotonic() - started_at:.1f}",
                    flush=True,
                )
                continue
            elapsed_s = time.monotonic() - started_at
            print(f"progress label={label} status=done elapsed_s={elapsed_s:.1f}", flush=True)
            return result


def resolve_dataset_path(dataset_root: Path, shard: str, raw_path: object) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    shard_path = dataset_root / shard / path
    if shard_path.exists():
        return shard_path
    return dataset_root / path


def output_filename(candidate: CandidateMap, *, index: int, difficulty: float) -> str:
    identity = candidate.beatmap_id or candidate.beatmap_set_id or candidate.row_index
    return f"{index:02d}_{identity}_mapper_cached_diff{_difficulty_slug(difficulty)}.osu"


def relative_audio_filename(audio_path: Path, output_path: Path) -> str:
    return os.path.relpath(audio_path, start=output_path.parent).replace(os.sep, "/")


def main(argv: Sequence[str] | None = None) -> int:
    return run_cached_stream_sample(argv)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mapper V2 cached full-song inference on indexed maps.")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--difficulty", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mapper-checkpoint-path", type=Path, default=DEFAULT_MAPPER_CHECKPOINT_PATH)
    parser.add_argument("--control-checkpoint-path", type=Path, default=DEFAULT_CONTROL_CHECKPOINT_PATH)
    parser.add_argument("--mapper-profile", choices=("auto", "v2_tuple", "v2_1_sparse"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--beatthis-device", default=DEFAULT_BEATTHIS_DEVICE)
    parser.add_argument("--beatthis-float16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eager-load-beatthis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--canonicalization",
        nargs="?",
        const=TIMING_CANONICALIZATION_BPM_80_160,
        default=TIMING_CANONICALIZATION_NONE,
        choices=TIMING_CANONICALIZATION_CHOICES,
        help="Fold fitted timing BPMs into [80, 160); pass 'none' to leave timing unchanged.",
    )
    parser.add_argument("--min-duration-s", type=float, default=45.0)
    parser.add_argument("--max-duration-s", type=float, default=120.0)
    parser.add_argument("--control-batch-size", type=int, default=4)
    parser.add_argument("--precompute-full-control", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--generation-seed", type=int, default=None)
    parser.add_argument("--use-incremental-mapper-decode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--time-shift-length-penalty-alpha", type=float, default=DEFAULT_TIME_SHIFT_LENGTH_PENALTY)
    parser.add_argument("--progress-interval-s", type=float, default=15.0)
    args = parser.parse_args(argv)
    if int(args.count) <= 0:
        raise ValueError("--count must be positive")
    if int(args.control_batch_size) <= 0:
        raise ValueError("--control-batch-size must be positive")
    if int(args.max_tokens) <= 0:
        raise ValueError("--max-tokens must be positive")
    if args.top_p is not None and not 0.0 < float(args.top_p) <= 1.0:
        raise ValueError("--top-p must be in (0, 1]")
    if args.min_duration_s is not None and float(args.min_duration_s) <= 0:
        raise ValueError("--min-duration-s must be positive")
    if args.max_duration_s is not None and float(args.max_duration_s) <= 0:
        raise ValueError("--max-duration-s must be positive")
    return args


def _grid_fitter_config_for_canonicalization(canonicalization: str) -> GridFitterConfig:
    return GridFitterConfig(
        canonicalization=canonicalization,
        canonicalize_tempo_aliases=canonicalization == TIMING_CANONICALIZATION_NONE,
    )


def _difficulty_slug(difficulty: float) -> str:
    return f"{float(difficulty):.2f}".replace(".", "p")


def _clean_text(value: object, default: str) -> str:
    if value is None:
        return default
    text = str(value)
    if not text or text.lower() == "nan":
        return default
    return text


def _optional_int(value: object) -> int | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite_float(value: object, *, default: float) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(resolved):
        return float(default)
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())
