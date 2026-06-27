from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from pulsefield_model.data.control_windows import normalize_difficulty
from pulsefield_model.inference.audio_probe import audio_length_ms_from_file
from pulsefield_model.inference.mapper_protocol import (
    HitObjectToken,
    MapperProtocolTranslator,
    MapperProfileName,
    build_mapper_protocol_translator,
    infer_mapper_profile_name_from_vocab,
    resolve_mapper_profile,
)
from pulsefield_model.inference.mapper_v2_tuple_rollout import (
    apply_time_shift_length_penalty as _apply_time_shift_length_penalty,
    mapper_v2_logits_fn as _mapper_v2_logits_fn,
    time_shift_length_penalty_tensors_v2_tuple as _time_shift_length_penalty_tensors,
)
from pulsefield_model.inference.model_runtime import (
    ModelRuntime,
    ModelRuntimeConfig,
    load_model_runtime,
    release_torch_cache,
)
from pulsefield_model.inference.session_runtime import (
    DEFAULT_MAX_CONTROL_BATCH_SIZE,
    SessionRuntime,
    SessionRuntimeConfig,
)
from pulsefield_model.inference.stream_windows import (
    DecoderWindow,
    clamp_decoder_window_to_audio,
    decoder_windows_until_audio_end,
)
from pulsefield_model.models.mapper.shared.replay import empty_ln_carry_state
from pulsefield_model.models.mapper.shared.tokenizer import MAPPER_WRITE_MS
from pulsefield_model.models.mapper.v2_1.replay import empty_ln_carry_state as empty_ln_carry_state_v2_1
from pulsefield_model.timing.canonicalization import TIMING_CANONICALIZATION_NONE
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
        model_runtime = await asyncio.to_thread(self._load_model_runtime)
        self._validate_loaded_runtime(model_runtime)
        self.model_runtime = model_runtime
        self.models_ready = True

    async def shutdown(self) -> None:
        runtime = self.model_runtime
        device = getattr(runtime, "device", self.config.device)
        for session_runtime in tuple(self._session_runtimes.values()):
            await asyncio.to_thread(session_runtime.reset_audio_cache)
        self._session_runtimes.clear()
        self._last_context_token_by_session.clear()
        self._last_carry_state_by_session.clear()
        self._protocol_translators_by_session.clear()
        self.model_runtime = None
        self.models_ready = False
        if str(device) != "auto":
            release_torch_cache(device)

    async def aclose(self) -> None:
        await self.shutdown()

    async def __aenter__(self) -> StreamWithCache:
        await self.startup()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        await self.shutdown()

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
                mapper_profile=self._runtime_mapper_profile_config(),
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
            from pulsefield_model.inference.model_bundles.mapper_v2_1_sparse import generate_window_v2_1_sparse

            return generate_window_v2_1_sparse(self, session_id, session_runtime, window, audio_length_ms)
        from pulsefield_model.inference.model_bundles.mapper_v2_tuple import generate_window_v2_tuple

        return generate_window_v2_tuple(self, session_id, session_runtime, window, audio_length_ms)

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

    def _mapper_profile_name(self) -> MapperProfileName:
        runtime = self.model_runtime
        if runtime is not None and getattr(runtime, "mapper_profile", None) is not None:
            return runtime.mapper_profile.name
        requested = str(getattr(self.config, "mapper_profile", "auto"))
        if requested.strip().lower() != "auto":
            return resolve_mapper_profile(requested).name
        return infer_mapper_profile_name_from_vocab(self._vocab())

    def _runtime_mapper_profile_config(self) -> str:
        return self.config.mapper_profile

    def _validate_loaded_runtime(self, runtime: ModelRuntime) -> None:
        del runtime

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


def run_cached_stream_sample(argv: Sequence[str] | None = None) -> int:
    from pulsefield_model.inference.cached_stream_cli import run_cached_stream_sample as run_sample

    return run_sample(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return run_cached_stream_sample(argv)


def _grid_fitter_config_for_canonicalization(canonicalization: str) -> GridFitterConfig:
    return GridFitterConfig(
        canonicalization=canonicalization,
        canonicalize_tempo_aliases=canonicalization == TIMING_CANONICALIZATION_NONE,
    )


if __name__ == "__main__":
    raise SystemExit(main())
