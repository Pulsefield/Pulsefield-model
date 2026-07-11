from __future__ import annotations

import math
import wave
from pathlib import Path


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
