from __future__ import annotations

from dataclasses import replace
from typing import Any

from pulsefield_model.inference.mapper_protocol import MapperProfileName, resolve_mapper_profile
from pulsefield_model.inference.mapper_v2_1_rollout import (
    carry_from_replay_state_v2_1,
    chart_end_ms_for_generation_v2_1,
    grammar_constrained_window_generation_v2_1,
    mapper_v2_1_logits_fn,
)
from pulsefield_model.inference.model_bundles.base import RouteBackend
from pulsefield_model.inference.model_bundles.mapper_base import StreamWithCacheMapperBundle
from pulsefield_model.inference.model_runtime import ModelRuntime
from pulsefield_model.inference.session_runtime import SessionRuntime
from pulsefield_model.inference.stream_with_cache import (
    DecoderWindow,
    StreamWithCache,
    StreamWithCacheConfig,
    _make_torch_generator,
)
from pulsefield_model.models.mapper.v2_1.replay import (
    LNCarryState as MapperV21LNCarryState,
    empty_ln_carry_state as empty_ln_carry_state_v2_1,
)
from pulsefield_model.models.mapper.v2_1.vocab import MapperV21Vocab


MAPPER_V2_1_SPARSE_MODEL_ID = "mapper/v2_1_sparse"
MAPPER_V2_1_SPARSE_PROFILE: MapperProfileName = "v2_1_sparse"


class MapperV21SparseStreamWithCache(StreamWithCache):
    """Stream-with-cache mapper implementation for v2.1 sparse lane-action vocab."""

    profile_name: MapperProfileName = MAPPER_V2_1_SPARSE_PROFILE

    def __init__(self, config: StreamWithCacheConfig, **kwargs: Any) -> None:
        super().__init__(_config_for_v2_1_sparse(config), **kwargs)

    def _runtime_mapper_profile_config(self) -> str:
        return self.profile_name

    def _mapper_profile_name(self) -> MapperProfileName:
        return self.profile_name

    def _validate_loaded_runtime(self, runtime: ModelRuntime) -> None:
        if runtime.mapper_profile.name != self.profile_name:
            raise ValueError(
                "mapper runtime profile mismatch: "
                f"{runtime.mapper_profile.name!r} loaded for {self.profile_name!r} stream",
            )

    def _generate_window(
        self,
        session_id: str,
        session_runtime: SessionRuntime,
        window: DecoderWindow,
        audio_length_ms: int,
    ) -> Any:
        if int(window.end_ms) - int(window.start_ms) != int(self.config.decoder_window_ms):
            raise ValueError("decoder window span does not match config.decoder_window_ms")
        return generate_window_v2_1_sparse(self, session_id, session_runtime, window, audio_length_ms)


class MapperV21SparseBundle(StreamWithCacheMapperBundle):
    """Concrete mapper v2.1 sparse model bundle."""

    profile = resolve_mapper_profile(MAPPER_V2_1_SPARSE_PROFILE)

    def __init__(
        self,
        config: StreamWithCacheConfig,
        *,
        model_id: str = MAPPER_V2_1_SPARSE_MODEL_ID,
        backend: RouteBackend | None = None,
    ) -> None:
        super().__init__(
            _config_for_v2_1_sparse(config),
            model_id=model_id,
            backend=MapperV21SparseStreamWithCache(config) if backend is None else backend,
        )


def _config_for_v2_1_sparse(config: StreamWithCacheConfig) -> StreamWithCacheConfig:
    return replace(config, mapper_profile=MAPPER_V2_1_SPARSE_PROFILE)


def generate_window_v2_1_sparse(
    stream: StreamWithCache,
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
        include_control_attention_kv_cache=bool(stream.config.use_incremental_mapper_decode),
    )

    vocab = stream._vocab()
    if not isinstance(vocab, MapperV21Vocab):
        raise TypeError("v2_1_sparse mapper profile requires MapperV21Vocab")
    carry_in = stream._carry_in_for_window(session_id, write_start_ms)
    if not isinstance(carry_in, MapperV21LNCarryState):
        raise TypeError("v2_1_sparse mapper profile requires MapperV21LNCarryState carry")
    chart_end_ms = chart_end_ms_for_generation_v2_1(audio_length_ms)
    is_full_chart_start = write_start_ms == 0 and not any(carry_in.open_mask)
    is_full_chart_end = write_end_ms >= chart_end_ms
    target_end_ms = chart_end_ms if is_full_chart_end else write_end_ms
    carry_out = empty_ln_carry_state_v2_1(target_end_ms)
    left_context_tokens: tuple[int, ...] = ()
    if not is_full_chart_start:
        left_context_tokens = (stream._left_context_token(session_id, vocab),)

    generator = _make_torch_generator(stream.config.seed, device=session_runtime.device)
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
        time_shift_length_penalty_alpha=float(stream.config.time_shift_length_penalty_alpha),
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
        max_tokens=int(stream.config.max_tokens),
        temperature=float(stream.config.temperature),
        top_p=stream.config.top_p,
        generator=generator,
    )
    if generated.tokens:
        stream._last_context_token_by_session[session_id] = int(generated.tokens[-1])
    if generated.completed:
        stream._last_carry_state_by_session[session_id] = carry_from_replay_state_v2_1(generated.terminal_state)
    return generated
