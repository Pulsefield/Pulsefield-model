from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from pulsefield_model.timing.schema import FrameTimingPrediction


BEATTHIS_PROVIDER_NAME = "beat-this"
DEFAULT_BEATTHIS_CHECKPOINT = "final0"
DEFAULT_BEATTHIS_DEVICE = "cpu"
BEATTHIS_FRAME_RATE_HZ = 50.0


class BeatThisDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class BeatThisAPI:
    audio2frames_cls: type
    load_audio: Callable[[str | Path], tuple[Any, int]]


class BeatThisTimingProvider:
    def __init__(
        self,
        *,
        checkpoint_path: str = DEFAULT_BEATTHIS_CHECKPOINT,
        device: str = DEFAULT_BEATTHIS_DEVICE,
        float16: bool = False,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.float16 = float16
        self._api: BeatThisAPI | None = None
        self._audio2frames: Any | None = None

    def predict_file(self, audio_path: str | Path) -> FrameTimingPrediction:
        signal, sample_rate = self.load_file(audio_path)
        return self._predict_audio(signal, sample_rate, source_path=audio_path)

    def load_file(self, audio_path: str | Path) -> tuple[Any, int]:
        try:
            return self._get_api().load_audio(audio_path)
        except Exception as primary_error:  # noqa: BLE001 - fallback boundary for third-party decoders.
            return _load_audio_with_ffmpeg(audio_path, primary_error=primary_error)

    def predict_audio(
        self,
        audio: object,
        sample_rate: int,
        *,
        source_path: str | Path | None = None,
    ) -> FrameTimingPrediction:
        return self._predict_audio(audio, sample_rate, source_path=source_path)

    def predict_shifted_audio(
        self,
        audio: object,
        sample_rate: int,
        *,
        shift_ms: float,
        source_path: str | Path | None = None,
    ) -> FrameTimingPrediction:
        shift_samples = audio_shift_samples_for_ms(shift_ms, sample_rate)
        shifted_audio = _prepend_zeros(audio, shift_samples)
        return self._predict_audio(shifted_audio, sample_rate, source_path=source_path)

    def _predict_audio(
        self,
        audio: object,
        sample_rate: int,
        *,
        source_path: str | Path | None,
    ) -> FrameTimingPrediction:
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")

        beat_logits, downbeat_logits = self._get_audio2frames()(audio, sample_rate)
        beat_prob = _logits_to_probability(_as_logit_vector(beat_logits, "beat_logits"))
        downbeat_prob = _logits_to_probability(_as_logit_vector(downbeat_logits, "downbeat_logits"))

        return FrameTimingPrediction(
            provider=BEATTHIS_PROVIDER_NAME,
            checkpoint_path=self.checkpoint_path,
            source_path=Path(source_path).as_posix() if source_path is not None else None,
            beat_prob=beat_prob,
            downbeat_prob=downbeat_prob,
            frame_rate_hz=BEATTHIS_FRAME_RATE_HZ,
        )

    def _get_api(self) -> BeatThisAPI:
        if self._api is None:
            self._api = _load_beat_this_api()
        return self._api

    def _get_audio2frames(self) -> Any:
        if self._audio2frames is None:
            self._audio2frames = self._get_api().audio2frames_cls(
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                float16=self.float16,
            )
        return self._audio2frames


def _load_beat_this_api() -> BeatThisAPI:
    try:
        from beat_this.inference import Audio2Frames, load_audio
    except ImportError as exc:
        raise BeatThisDependencyError(
            "BeatThisTimingProvider requires beat-this. Install the pulsefield-model mps or cuda optional dependencies."
        ) from exc

    return BeatThisAPI(audio2frames_cls=Audio2Frames, load_audio=load_audio)


def _load_audio_with_ffmpeg(
    audio_path: str | Path,
    *,
    primary_error: Exception,
) -> tuple[NDArray[np.float32], int]:
    path = Path(audio_path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        decoded_path = Path(handle.name)
    try:
        command = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-vn",
            "-c:a",
            "pcm_f32le",
            "-y",
            str(decoded_path),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as ffmpeg_error:
            raise RuntimeError(
                f'Could not load audio from "{path}": primary decoders failed and ffmpeg is unavailable.'
            ) from ffmpeg_error

        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            detail = stderr[-1000:] if stderr else f"ffmpeg exited with code {completed.returncode}"
            raise RuntimeError(
                f'Could not load audio from "{path}": primary decoders failed; ffmpeg fallback failed: {detail}'
            ) from primary_error

        try:
            import soundfile as sf

            audio, sample_rate = sf.read(decoded_path, dtype="float32", always_2d=False)
        except Exception as decoded_error:
            raise RuntimeError(
                f'Could not load audio from "{path}": ffmpeg output could not be read.'
            ) from decoded_error
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if audio.size == 0:
            raise RuntimeError(f'Could not load audio from "{path}": ffmpeg produced empty audio.')
        return audio, int(sample_rate)
    finally:
        decoded_path.unlink(missing_ok=True)


def audio_shift_samples_for_ms(shift_ms: float, sample_rate: int) -> int:
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")
    if not np.isfinite(shift_ms) or shift_ms < 0.0:
        raise ValueError(f"shift_ms must be non-negative and finite, got {shift_ms!r}")
    return int(np.floor(float(shift_ms) / 1000.0 * sample_rate + 0.5))


def _prepend_zeros(audio: object, sample_count: int) -> object:
    if sample_count == 0:
        return audio

    array = np.asarray(audio)
    if array.ndim not in (1, 2):
        raise ValueError(f"audio must be a 1-D or 2-D signal, got shape {array.shape}")

    pad_shape = list(array.shape)
    pad_shape[0] = sample_count
    padding = np.zeros(tuple(pad_shape), dtype=array.dtype)
    return np.concatenate((padding, array), axis=0)


def _as_logit_vector(value: object, name: str) -> NDArray[np.float32]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()

    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _logits_to_probability(logits: NDArray[np.float32]) -> NDArray[np.float32]:
    logits64 = logits.astype(np.float64, copy=False)
    probabilities = np.empty_like(logits64)
    positive = logits64 >= 0.0

    probabilities[positive] = 1.0 / (1.0 + np.exp(-logits64[positive]))
    exp_logits = np.exp(logits64[~positive])
    probabilities[~positive] = exp_logits / (1.0 + exp_logits)

    return probabilities.astype(np.float32)
