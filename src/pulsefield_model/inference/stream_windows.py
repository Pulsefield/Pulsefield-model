from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class DecoderWindow:
    start_ms: int
    end_ms: int


class DecoderWindowConfig(Protocol):
    decoder_window_ms: int


def clamp_decoder_window_to_audio(
    window: DecoderWindow,
    *,
    audio_length_ms: int,
    config: DecoderWindowConfig,
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
    config: DecoderWindowConfig,
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
