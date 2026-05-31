from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from pulsefield_model.timing.canonicalization import DEFAULT_TIMING_CANONICALIZATION, canonicalize_timing_grid
from pulsefield_model.timing.canonicalization import require_timing_canonicalization
from pulsefield_model.timing.rendering.dense_timing_v2 import DEFAULT_DENSE_TIMING_V2_CONFIG
from pulsefield_model.timing.rendering.dense_timing_v2 import DENSE_TIMING_V2_CHANNELS
from pulsefield_model.timing.rendering.dense_timing_v2 import DENSE_TIMING_V2_VERSION
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
    canonicalization: str = DEFAULT_TIMING_CANONICALIZATION

    def __post_init__(self) -> None:
        if not math.isfinite(self.input_start_ms):
            raise ValueError(f"input_start_ms must be finite, got {self.input_start_ms!r}")
        require_timing_canonicalization(self.canonicalization)


DEFAULT_ORACLE_TIMING_CONFIG = OracleTimingConfig()


@dataclass(frozen=True)
class OracleDenseTimingCacheConfig:
    cache_root: Path = Path("artifacts/cache")
    cache_version: str = "oracle_dense_timing_v2"


DEFAULT_ORACLE_DENSE_TIMING_CACHE_CONFIG = OracleDenseTimingCacheConfig()


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
        return oracle_timing_grid_from_beatmap(beatmap_path, canonicalization=self.config.canonicalization)


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
        oracle_timing_grid_from_beatmap(beatmap_path, canonicalization=config.canonicalization),
        input_start_ms=config.input_start_ms,
        frame_count=resolved_frame_count,
        config=config.dense_timing_config,
    )


def load_or_create_oracle_dense_timing_v2_cache(
    beatmap_path: str | Path,
    *,
    frame_count: int | None = None,
    audio_duration_ms: float | None = None,
    audio_duration_seconds: float | None = None,
    timing_config: OracleTimingConfig = DEFAULT_ORACLE_TIMING_CONFIG,
    cache_config: OracleDenseTimingCacheConfig = DEFAULT_ORACLE_DENSE_TIMING_CACHE_CONFIG,
) -> DenseTimingV2Track:
    resolved_frame_count = _resolve_frame_count(
        frame_count=frame_count,
        audio_duration_ms=audio_duration_ms,
        audio_duration_seconds=audio_duration_seconds,
        config=timing_config.dense_timing_config,
    )
    cache_path = oracle_dense_timing_v2_cache_path(
        beatmap_path,
        frame_count=resolved_frame_count,
        timing_config=timing_config,
        cache_config=cache_config,
    )
    if cache_path.exists():
        return _load_cached_dense_timing_v2(cache_path, expected_frame_count=resolved_frame_count)

    track = _validate_dense_timing_v2(
        render_oracle_dense_timing_v2(beatmap_path, frame_count=resolved_frame_count, config=timing_config),
        expected_frame_count=resolved_frame_count,
        source=beatmap_path,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, track)
    tmp_path.replace(cache_path)
    return track


def oracle_dense_timing_v2_cache_path(
    beatmap_path: str | Path,
    *,
    frame_count: int,
    timing_config: OracleTimingConfig = DEFAULT_ORACLE_TIMING_CONFIG,
    cache_config: OracleDenseTimingCacheConfig = DEFAULT_ORACLE_DENSE_TIMING_CACHE_CONFIG,
) -> Path:
    resolved_frame_count = _validate_frame_count(frame_count)
    key_payload = {
        "beatmap_path": Path(beatmap_path).as_posix(),
        "frame_count": resolved_frame_count,
    }
    key = hashlib.sha256(json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return (
        Path(cache_config.cache_root)
        / cache_config.cache_version
        / _oracle_dense_timing_config_hash(timing_config)
        / f"{key}.npy"
    )


def oracle_timing_grid_from_beatmap(
    beatmap_path: str | Path,
    *,
    canonicalization: str = DEFAULT_TIMING_CANONICALIZATION,
) -> FittedTimingGrid:
    return canonicalize_timing_grid(
        fitted_timing_grid_from_red_points(_require_red_timing_points(beatmap_path)),
        canonicalization=canonicalization,
    )


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


def _load_cached_dense_timing_v2(cache_path: Path, *, expected_frame_count: int) -> DenseTimingV2Track:
    return _validate_dense_timing_v2(
        np.load(cache_path),
        expected_frame_count=expected_frame_count,
        source=cache_path,
    )


def _validate_dense_timing_v2(value: object, *, expected_frame_count: int, source: object) -> DenseTimingV2Track:
    array = np.asarray(value, dtype=np.float32)
    expected_shape = (expected_frame_count, len(DENSE_TIMING_V2_CHANNELS))
    if array.shape != expected_shape:
        raise ValueError(f"dense timing v2 for {source} must have shape {expected_shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"dense timing v2 for {source} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float32)


def _oracle_dense_timing_config_hash(config: OracleTimingConfig) -> str:
    dense = config.dense_timing_config
    payload = {
        "provider": ORACLE_TIMING_PROVIDER_NAME,
        "version": DENSE_TIMING_V2_VERSION,
        "channels": DENSE_TIMING_V2_CHANNELS,
        "input_start_ms": config.input_start_ms,
        "canonicalization": config.canonicalization,
        "frame_hop_ms": dense.frame_hop_ms,
        "frame_center_offset_ms": dense.frame_center_offset_ms,
        "pulse_width_ms": dense.pulse_width_ms,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _require_red_timing_points(beatmap_path: str | Path) -> Sequence[object]:
    from pulsefield_model.osu_core.timing import require_red_timing_points

    return require_red_timing_points(beatmap_path)
