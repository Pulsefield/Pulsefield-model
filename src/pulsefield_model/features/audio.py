from pathlib import Path
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from pydub import AudioSegment


AudioWaveform: TypeAlias = npt.NDArray[np.float32]


def load_audio_file(file: str | Path, sample_rate: int, speed: float = 1.0, normalize: bool = True) -> AudioWaveform:
    """Load audio as a mono float32 waveform, following osuT5 preprocessing."""
    file = Path(file)
    format_name = file.suffix[1:] if file.suffix else None
    with file.open("rb") as handle:
        audio = AudioSegment.from_file(handle, format=format_name)
    audio.frame_rate = int(audio.frame_rate * speed)
    audio = audio.set_frame_rate(sample_rate)
    audio = audio.set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)

    if normalize and samples.size > 0:
        peak = float(np.max(np.abs(samples)))
        if peak > 0:
            samples *= 1.0 / peak

    return samples
