from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pulsefield_model.timing.rendering.dense_timing_v2 import DEFAULT_DENSE_TIMING_V2_CONFIG
from pulsefield_model.timing.rendering.dense_timing_v2 import DenseTimingV2Config
from pulsefield_model.timing.rendering.dense_timing_v2 import DenseTimingV2Track
from pulsefield_model.timing.rendering.dense_timing_v2 import render_dense_timing_v2
from pulsefield_model.timing.schema import FittedTimingGrid
from pulsefield_model.timing.schema import TimingSegment


ORACLE_TIMING_PROVIDER_NAME = "oracle-red-timing"


@dataclass(frozen=True)
class OracleTimingConfig:
    dense_timing_config: DenseTimingV2Config = DEFAULT_DENSE_TIMING_V2_CONFIG
    input_start_ms: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.input_start_ms):
            raise ValueError(f"input_start_ms must be finite, got {self.input_start_ms!r}")


DEFAULT_ORACLE_TIMING_CONFIG = OracleTimingConfig()


class OracleTimingProvider:
    def __init__(self, config: OracleTimingConfig = DEFAULT_ORACLE_TIMING_CONFIG) -> None:
        self.config = config

    def render_file(
        self,
        beatmap_path: str | Path,
        *,
        frame_count: int | None = None,
        audio_duration_ms: float | None = None,
        audio_duration_seconds: float | None = None,
    ) -> DenseTimingV2Track:
        return render_oracle_dense_timing_v2(
            beatmap_path,
            frame_count=frame_count,
            audio_duration_ms=audio_duration_ms,
            audio_duration_seconds=audio_duration_seconds,
            config=self.config,
        )

    def grid_for_file(self, beatmap_path: str | Path) -> FittedTimingGrid:
        return oracle_timing_grid_from_beatmap(beatmap_path)


def render_oracle_dense_timing_v2(
    beatmap_path: str | Path,
    *,
    frame_count: int | None = None,
    audio_duration_ms: float | None = None,
    audio_duration_seconds: float | None = None,
    config: OracleTimingConfig = DEFAULT_ORACLE_TIMING_CONFIG,
) -> DenseTimingV2Track:
    resolved_frame_count = _resolve_frame_count(
        frame_count=frame_count,
        audio_duration_ms=audio_duration_ms,
        audio_duration_seconds=audio_duration_seconds,
        config=config.dense_timing_config,
    )
    return render_dense_timing_v2(
        oracle_timing_grid_from_beatmap(beatmap_path),
        input_start_ms=config.input_start_ms,
        frame_count=resolved_frame_count,
        config=config.dense_timing_config,
    )


def oracle_timing_grid_from_beatmap(beatmap_path: str | Path) -> FittedTimingGrid:
    return fitted_timing_grid_from_red_points(_require_red_timing_points(beatmap_path))


def fitted_timing_grid_from_red_points(red_timing_points: Sequence[object]) -> FittedTimingGrid:
    segments_by_offset: dict[float, TimingSegment] = {}
    for point in red_timing_points:
        offset_ms = float(point.offset_ms)
        segments_by_offset[offset_ms] = TimingSegment(
            offset_ms=offset_ms,
            beat_length_ms=float(point.beat_length_ms),
            meter=int(getattr(point, "meter", 4)),
        )
    return FittedTimingGrid(tuple(segments_by_offset[offset] for offset in sorted(segments_by_offset)))


def _resolve_frame_count(
    *,
    frame_count: int | None,
    audio_duration_ms: float | None,
    audio_duration_seconds: float | None,
    config: DenseTimingV2Config,
) -> int:
    if audio_duration_ms is not None and audio_duration_seconds is not None:
        raise ValueError("provide only one of audio_duration_ms or audio_duration_seconds")
    resolved_audio_duration_ms = _optional_audio_duration_ms(
        audio_duration_ms=audio_duration_ms,
        audio_duration_seconds=audio_duration_seconds,
    )
    if frame_count is not None:
        if resolved_audio_duration_ms is not None:
            raise ValueError("provide only one of frame_count or audio duration")
        return _validate_frame_count(frame_count)
    if resolved_audio_duration_ms is None:
        raise ValueError("frame_count or audio duration is required")

    return int(math.ceil(resolved_audio_duration_ms / config.frame_hop_ms))


def _optional_audio_duration_ms(
    *,
    audio_duration_ms: float | None,
    audio_duration_seconds: float | None,
) -> float | None:
    if audio_duration_seconds is not None:
        audio_duration_ms = audio_duration_seconds * 1000.0
    if audio_duration_ms is None:
        return None
    if not math.isfinite(audio_duration_ms) or audio_duration_ms <= 0.0:
        raise ValueError(f"audio_duration_ms must be positive and finite, got {audio_duration_ms!r}")
    return audio_duration_ms


def _validate_frame_count(frame_count: int) -> int:
    if not isinstance(frame_count, int) or isinstance(frame_count, bool):
        raise TypeError(f"frame_count must be an integer, got {type(frame_count).__name__}")
    if frame_count < 0:
        raise ValueError(f"frame_count must be non-negative, got {frame_count!r}")
    return frame_count


def _require_red_timing_points(beatmap_path: str | Path) -> Sequence[object]:
    from pulsefield_model.osu_core.timing import require_red_timing_points

    return require_red_timing_points(beatmap_path)
