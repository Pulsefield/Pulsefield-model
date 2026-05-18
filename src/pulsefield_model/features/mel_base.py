from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import torch
from nnAudio.features import MelSpectrogram


LogMel: TypeAlias = npt.NDArray[np.float32]
LOG_MEL_FLOOR = 1e-5
LOG_MEL_SILENCE_VALUE = np.float32(np.log(LOG_MEL_FLOOR))


@dataclass(frozen=True)
class MelCacheConfig:
    cache_root: Path = Path("artifacts/cache")
    cache_version: str = "mel_sr16000_hop10_mel80_v2"
    sample_rate: int = 16000
    mel_bins: int = 80
    hop_ms: int = 10
    n_fft: int = 400
    win_length: int = 400
    fmin: float = 20.0
    fmax: float = 8000.0

    @property
    def hop_length(self) -> int:
        return int(self.sample_rate * self.hop_ms / 1000)

    @property
    def mel_config_hash(self) -> str:
        return _mel_config_hash(self)

    @property
    def cache_dir(self) -> Path:
        return self.cache_root / self.cache_version / self.mel_config_hash


DEFAULT_MEL_CACHE_CONFIG = MelCacheConfig()


def compute_log_mel_10ms(
    waveform: npt.NDArray[np.float32],
    *,
    sample_rate: int,
    config: MelCacheConfig = DEFAULT_MEL_CACHE_CONFIG,
) -> LogMel:
    if sample_rate != config.sample_rate:
        raise ValueError(f"expected {config.sample_rate}Hz waveform, got {sample_rate}Hz")

    mono = np.asarray(waveform, dtype=np.float32).reshape(-1)
    target_frame_count = _target_10ms_frame_count(mono.shape[0], config)
    if target_frame_count == 0:
        return np.empty((0, config.mel_bins), dtype=np.float32)

    required_sample_count = _stft_sample_count_for_frame_count(target_frame_count, config)
    if mono.shape[0] < required_sample_count:
        mono = np.pad(mono, (0, required_sample_count - mono.shape[0]))

    tensor = torch.from_numpy(mono).unsqueeze(0)
    mel_layer = _mel_layer(config)
    with torch.no_grad():
        mel = mel_layer(tensor).squeeze(0).transpose(0, 1).cpu().numpy()
    log_mel = np.log(np.maximum(mel, LOG_MEL_FLOOR)).astype(np.float32, copy=False)
    return log_mel[:target_frame_count]


def load_or_create_log_mel_cache(
    waveform: npt.NDArray[np.float32],
    *,
    sample_rate: int,
    audio_cache_key: str,
    config: MelCacheConfig = DEFAULT_MEL_CACHE_CONFIG,
) -> LogMel:
    cache_path = log_mel_cache_path(audio_cache_key, config=config)
    if cache_path.exists():
        return np.load(cache_path).astype(np.float32, copy=False)

    mel = compute_log_mel_10ms(waveform, sample_rate=sample_rate, config=config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, mel)
    return mel


def log_mel_cache_path(
    audio_cache_key: str,
    *,
    config: MelCacheConfig = DEFAULT_MEL_CACHE_CONFIG,
) -> Path:
    return config.cache_dir / f"{_cache_key(audio_cache_key)}.npy"


def pack_mel_20ms_window(
    mel_10ms: npt.NDArray[np.float32],
    *,
    input_start_ms: int,
    frame_count: int = 600,
) -> npt.NDArray[np.float32]:
    if input_start_ms % 10 != 0:
        raise ValueError(f"input_start_ms must land on 10ms grid: {input_start_ms}")
    if frame_count < 0:
        raise ValueError(f"frame_count must be non-negative: {frame_count}")
    if mel_10ms.ndim != 2 or mel_10ms.shape[1] != 80:
        raise ValueError(f"expected mel shape [frames, 80], got {mel_10ms.shape}")

    packed = np.full((frame_count, 160), LOG_MEL_SILENCE_VALUE, dtype=np.float32)
    start_index = input_start_ms // 10
    for frame_index in range(frame_count):
        first_index = start_index + frame_index * 2
        second_index = first_index + 1
        if 0 <= first_index < mel_10ms.shape[0]:
            packed[frame_index, :80] = mel_10ms[first_index]
        if 0 <= second_index < mel_10ms.shape[0]:
            packed[frame_index, 80:] = mel_10ms[second_index]
    return packed


@lru_cache(maxsize=4)
def _mel_layer(config: MelCacheConfig) -> MelSpectrogram:
    return MelSpectrogram(
        sr=config.sample_rate,
        n_fft=config.n_fft,
        win_length=config.win_length,
        n_mels=config.mel_bins,
        hop_length=config.hop_length,
        window="hann",
        center=False,
        power=2.0,
        fmin=config.fmin,
        fmax=config.fmax,
        norm=1,
        trainable_mel=False,
        trainable_STFT=False,
        verbose=False,
    )


def _cache_key(audio_cache_key: str) -> str:
    return hashlib.sha256(audio_cache_key.encode("utf-8")).hexdigest()


def _target_10ms_frame_count(sample_count: int, config: MelCacheConfig) -> int:
    if sample_count <= 0:
        return 0
    return math.ceil(sample_count / config.hop_length)


def _stft_sample_count_for_frame_count(frame_count: int, config: MelCacheConfig) -> int:
    if frame_count <= 0:
        return 0
    return (frame_count - 1) * config.hop_length + config.n_fft


def _mel_config_hash(config: MelCacheConfig) -> str:
    payload = {
        "cache_version": config.cache_version,
        "sample_rate": config.sample_rate,
        "mel_bins": config.mel_bins,
        "hop_ms": config.hop_ms,
        "n_fft": config.n_fft,
        "win_length": config.win_length,
        "fmin": config.fmin,
        "fmax": config.fmax,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
