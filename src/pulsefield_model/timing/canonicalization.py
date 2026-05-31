from __future__ import annotations

import math
from typing import Final

from pulsefield_model.timing.schema import FittedTimingGrid, TimingSegment


TIMING_CANONICALIZATION_NONE: Final[str] = "none"
TIMING_CANONICALIZATION_BPM_80_160: Final[str] = "bpm-80-160"
TIMING_CANONICALIZATION_CHOICES: Final[tuple[str, ...]] = (
    TIMING_CANONICALIZATION_NONE,
    TIMING_CANONICALIZATION_BPM_80_160,
)
DEFAULT_TIMING_CANONICALIZATION: Final[str] = TIMING_CANONICALIZATION_NONE
CANONICAL_BPM_MIN: Final[float] = 80.0
CANONICAL_BPM_MAX: Final[float] = 160.0


def canonicalize_timing_grid(
    grid: FittedTimingGrid,
    *,
    canonicalization: str = TIMING_CANONICALIZATION_BPM_80_160,
) -> FittedTimingGrid:
    canonicalization = require_timing_canonicalization(canonicalization)
    if canonicalization == TIMING_CANONICALIZATION_NONE:
        return grid
    if canonicalization != TIMING_CANONICALIZATION_BPM_80_160:
        raise ValueError(f"unsupported timing canonicalization: {canonicalization!r}")

    return FittedTimingGrid(
        tuple(
            TimingSegment(
                offset_ms=segment.offset_ms,
                beat_length_ms=60000.0 / canonical_bpm_80_160(segment.local_bpm),
                meter=segment.meter,
            )
            for segment in grid.segments
        )
    )


def canonical_bpm_80_160(bpm: float) -> float:
    if not math.isfinite(float(bpm)) or float(bpm) <= 0.0:
        raise ValueError(f"bpm must be positive and finite, got {bpm!r}")

    canonical_bpm = float(bpm)
    while canonical_bpm < CANONICAL_BPM_MIN:
        canonical_bpm *= 2.0
    while canonical_bpm >= CANONICAL_BPM_MAX:
        canonical_bpm *= 0.5
    return float(canonical_bpm)


def require_timing_canonicalization(canonicalization: str) -> str:
    if canonicalization not in TIMING_CANONICALIZATION_CHOICES:
        choices = ", ".join(TIMING_CANONICALIZATION_CHOICES)
        raise ValueError(f"canonicalization must be one of {choices}, got {canonicalization!r}")
    return canonicalization
