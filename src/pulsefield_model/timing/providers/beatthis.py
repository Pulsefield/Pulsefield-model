from __future__ import annotations

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
        signal, sample_rate = self._get_api().load_audio(audio_path)
        return self._predict_audio(signal, sample_rate, source_path=audio_path)

    def predict_audio(self, audio: object, sample_rate: int) -> FrameTimingPrediction:
        return self._predict_audio(audio, sample_rate, source_path=None)

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
