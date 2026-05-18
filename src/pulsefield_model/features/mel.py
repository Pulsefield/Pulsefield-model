from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from pulsefield_model.features.audio import load_audio_file
from pulsefield_model.features.mel_base import DEFAULT_MEL_CACHE_CONFIG
from pulsefield_model.features.mel_base import MelCacheConfig
from pulsefield_model.features.mel_base import compute_log_mel_10ms
from pulsefield_model.features.mel_base import log_mel_cache_path
from pulsefield_model.features.mel_base import load_or_create_log_mel_cache
from pulsefield_model.features.mel_base import pack_mel_20ms_window


PackedMel20msTrack: TypeAlias = NDArray[np.float32]


@dataclass(frozen=True)
class Stage2MelConfig:
    mel_cache_config: MelCacheConfig = DEFAULT_MEL_CACHE_CONFIG
    speed: float = 1.0
    normalize: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.mel_cache_config.sample_rate, int) or isinstance(
            self.mel_cache_config.sample_rate,
            bool,
        ):
            raise TypeError(
                f"sample_rate must be an integer, got {type(self.mel_cache_config.sample_rate).__name__}"
            )
        if self.mel_cache_config.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.mel_cache_config.sample_rate!r}")
        if self.mel_cache_config.hop_ms != 10:
            raise ValueError(
                f"Stage 2 packed mel requires 10ms mel frames, got hop_ms={self.mel_cache_config.hop_ms!r}"
            )
        if self.mel_cache_config.mel_bins != 80:
            raise ValueError(f"Stage 2 packed mel requires 80 mel bins, got {self.mel_cache_config.mel_bins!r}")
        if not math.isfinite(self.speed) or self.speed <= 0.0:
            raise ValueError(f"speed must be positive and finite, got {self.speed!r}")
        if not isinstance(self.normalize, bool):
            raise TypeError(f"normalize must be a boolean, got {type(self.normalize).__name__}")

    @property
    def sample_rate(self) -> int:
        return self.mel_cache_config.sample_rate


DEFAULT_STAGE2_MEL_CONFIG = Stage2MelConfig()


def load_full_song_packed_mel_20ms(
    audio_path: str | Path,
    *,
    audio_cache_key: str | None = None,
    config: Stage2MelConfig = DEFAULT_STAGE2_MEL_CONFIG,
) -> PackedMel20msTrack:
    audio_path = Path(audio_path)
    resolved_audio_cache_key = audio_cache_key or _default_audio_cache_key(audio_path, config)
    cache_path = log_mel_cache_path(resolved_audio_cache_key, config=config.mel_cache_config)
    if cache_path.exists():
        return pack_full_song_mel_20ms(np.load(cache_path).astype(np.float32, copy=False))

    waveform = load_audio_file(
        audio_path,
        sample_rate=config.sample_rate,
        speed=config.speed,
        normalize=config.normalize,
    )
    return full_song_packed_mel_20ms_from_waveform(
        waveform,
        sample_rate=config.sample_rate,
        audio_cache_key=resolved_audio_cache_key,
        config=config,
    )


def stage2_log_mel_cache_path(
    audio_path: str | Path,
    *,
    audio_cache_key: str | None = None,
    config: Stage2MelConfig = DEFAULT_STAGE2_MEL_CONFIG,
) -> Path:
    audio_path = Path(audio_path)
    return log_mel_cache_path(
        audio_cache_key or _default_audio_cache_key(audio_path, config),
        config=config.mel_cache_config,
    )


def full_song_packed_mel_20ms_from_waveform(
    waveform: object,
    *,
    sample_rate: int,
    audio_cache_key: str | None = None,
    config: Stage2MelConfig = DEFAULT_STAGE2_MEL_CONFIG,
) -> PackedMel20msTrack:
    _validate_sample_rate(sample_rate)
    if sample_rate != config.sample_rate:
        raise ValueError(f"expected {config.sample_rate}Hz waveform, got {sample_rate}Hz")
    audio = _as_waveform(waveform)
    if audio_cache_key is None:
        mel_10ms = compute_log_mel_10ms(audio, sample_rate=sample_rate, config=config.mel_cache_config)
    else:
        mel_10ms = load_or_create_log_mel_cache(
            audio,
            sample_rate=sample_rate,
            audio_cache_key=audio_cache_key,
            config=config.mel_cache_config,
        )
    return pack_full_song_mel_20ms(mel_10ms)


def pack_full_song_mel_20ms(mel_10ms: object) -> PackedMel20msTrack:
    mel = _as_log_mel_10ms(mel_10ms)
    frame_count = (mel.shape[0] + 1) // 2
    packed = pack_mel_20ms_window(mel, input_start_ms=0, frame_count=frame_count)
    return _as_packed_mel_20ms(packed)


def _as_waveform(waveform: object) -> NDArray[np.float32]:
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"waveform must be a 1-D mono array, got shape {audio.shape}")
    if not np.all(np.isfinite(audio)):
        raise ValueError("waveform must contain only finite values")
    return audio


def _as_log_mel_10ms(mel_10ms: object) -> NDArray[np.float32]:
    mel = np.asarray(mel_10ms, dtype=np.float32)
    if mel.ndim != 2 or mel.shape[1] != 80:
        raise ValueError(f"expected 10ms log mel shape [frames, 80], got {mel.shape}")
    if not np.all(np.isfinite(mel)):
        raise ValueError("10ms log mel must contain only finite values")
    return mel


def _as_packed_mel_20ms(packed_mel: object) -> PackedMel20msTrack:
    packed = np.asarray(packed_mel, dtype=np.float32)
    if packed.ndim != 2 or packed.shape[1] != 160:
        raise ValueError(f"expected packed 20ms mel shape [frames, 160], got {packed.shape}")
    if not np.all(np.isfinite(packed)):
        raise ValueError("packed 20ms mel must contain only finite values")
    return packed


def _validate_sample_rate(sample_rate: int) -> None:
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
        raise TypeError(f"sample_rate must be an integer, got {type(sample_rate).__name__}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")


def _default_audio_cache_key(audio_path: Path, config: Stage2MelConfig) -> str:
    if config.speed == 1.0 and config.normalize:
        return audio_path.as_posix()
    return (
        f"{audio_path.as_posix()}|"
        f"sample_rate={config.sample_rate}|"
        f"speed={config.speed:.12g}|"
        f"normalize={int(config.normalize)}"
    )
