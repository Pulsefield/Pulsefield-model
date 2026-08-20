from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pulsefield_model.timing.providers.beatthis import (
    BEATTHIS_FRAME_RATE_HZ,
    BEATTHIS_PROVIDER_NAME,
    DEFAULT_BEATTHIS_CHECKPOINT,
)
from pulsefield_model.timing.schema import FrameTimingPrediction


BEATTHIS_FRAME_PREDICTION_CACHE_SCHEMA_VERSION = 1
BEATTHIS_FRAME_PREDICTION_CACHE_FORMAT = "pulsefield_model.beatthis_frame_prediction"
BEATTHIS_AUDIO_CACHE_KEY_VERSION = 1


@dataclass(frozen=True)
class BeatThisFramePredictionCacheConfig:
    cache_root: Path = Path("artifacts/cache")
    cache_version: str = "beatthis_frame_predictions_v2"
    checkpoint_path: str = DEFAULT_BEATTHIS_CHECKPOINT
    float16: bool = False
    shift_ms: float = 0.0
    frame_rate_hz: float = BEATTHIS_FRAME_RATE_HZ

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_root", Path(self.cache_root))

        if not isinstance(self.cache_version, str) or not self.cache_version:
            raise ValueError("cache_version must be a non-empty string")

        checkpoint_path = str(self.checkpoint_path)
        if not checkpoint_path:
            raise ValueError("checkpoint_path must be non-empty")
        object.__setattr__(self, "checkpoint_path", checkpoint_path)

        if not isinstance(self.float16, bool):
            raise TypeError(f"float16 must be a bool, got {type(self.float16).__name__}")

        shift_ms = float(self.shift_ms)
        if not math.isfinite(shift_ms) or shift_ms < 0.0:
            raise ValueError(f"shift_ms must be non-negative and finite, got {self.shift_ms!r}")
        object.__setattr__(self, "shift_ms", 0.0 if shift_ms == 0.0 else shift_ms)

        frame_rate_hz = float(self.frame_rate_hz)
        if not math.isfinite(frame_rate_hz) or frame_rate_hz <= 0.0:
            raise ValueError(f"frame_rate_hz must be positive and finite, got {self.frame_rate_hz!r}")
        object.__setattr__(self, "frame_rate_hz", frame_rate_hz)

    @property
    def config_fingerprint(self) -> str:
        return _beatthis_frame_prediction_config_fingerprint(self)

    @property
    def cache_dir(self) -> Path:
        return self.cache_root / self.cache_version / self.config_fingerprint


DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_CONFIG = BeatThisFramePredictionCacheConfig()


class BeatThisFramePredictionCacheError(ValueError):
    pass


def beatthis_audio_cache_key(audio_path: str | Path) -> str:
    """Return a cache key that is invalidated when the audio file changes."""
    resolved_path = Path(audio_path).resolve(strict=True)
    stat = resolved_path.stat()
    payload = {
        "version": BEATTHIS_AUDIO_CACHE_KEY_VERSION,
        "audio_path": resolved_path.as_posix(),
        "audio_size": stat.st_size,
        "audio_mtime_ns": stat.st_mtime_ns,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def beatthis_frame_prediction_cache_path(
    audio_cache_key: str,
    config: BeatThisFramePredictionCacheConfig = DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_CONFIG,
) -> Path:
    return config.cache_dir / f"{_audio_cache_key_hash(audio_cache_key)}.npz"


def load_beatthis_frame_prediction_cache(
    audio_cache_key: str,
    config: BeatThisFramePredictionCacheConfig = DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_CONFIG,
) -> FrameTimingPrediction | None:
    cache_path = beatthis_frame_prediction_cache_path(audio_cache_key, config)
    if not cache_path.exists():
        return None

    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            return _prediction_from_npz_payload(
                payload,
                audio_cache_key=audio_cache_key,
                config=config,
                cache_path=cache_path,
            )
    except BeatThisFramePredictionCacheError:
        raise
    except Exception as exc:
        raise BeatThisFramePredictionCacheError(f"invalid BeatThis frame prediction cache {cache_path}: {exc}") from exc


def save_beatthis_frame_prediction_cache(
    prediction: FrameTimingPrediction,
    audio_cache_key: str,
    config: BeatThisFramePredictionCacheConfig = DEFAULT_BEATTHIS_FRAME_PREDICTION_CACHE_CONFIG,
) -> Path:
    _validate_prediction_for_config(prediction, config=config)
    cache_path = beatthis_frame_prediction_cache_path(audio_cache_key, config)
    metadata = _metadata_for_prediction(
        prediction,
        audio_cache_key=audio_cache_key,
        config=config,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            np.savez_compressed(
                handle,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
                beat_prob=np.ascontiguousarray(prediction.beat_prob, dtype=np.float32),
                downbeat_prob=np.ascontiguousarray(prediction.downbeat_prob, dtype=np.float32),
            )
        tmp_path.replace(cache_path)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise
    return cache_path


def _prediction_from_npz_payload(
    payload: Any,
    *,
    audio_cache_key: str,
    config: BeatThisFramePredictionCacheConfig,
    cache_path: Path,
) -> FrameTimingPrediction:
    expected_files = {"metadata_json", "beat_prob", "downbeat_prob"}
    files = set(payload.files)
    if files != expected_files:
        raise BeatThisFramePredictionCacheError(
            f"cache {cache_path} must contain files {sorted(expected_files)}, got {sorted(files)}"
        )

    metadata = _load_metadata(payload["metadata_json"], cache_path=cache_path)
    _validate_metadata(metadata, audio_cache_key=audio_cache_key, config=config, cache_path=cache_path)

    beat_prob = _load_probability_vector(payload["beat_prob"], name="beat_prob", cache_path=cache_path)
    downbeat_prob = _load_probability_vector(payload["downbeat_prob"], name="downbeat_prob", cache_path=cache_path)

    frame_count = metadata["frame_count"]
    if beat_prob.shape[0] != frame_count or downbeat_prob.shape[0] != frame_count:
        raise BeatThisFramePredictionCacheError(
            f"cache {cache_path} frame_count metadata {frame_count} does not match array lengths "
            f"{beat_prob.shape[0]} and {downbeat_prob.shape[0]}"
        )

    try:
        return FrameTimingPrediction(
            provider=metadata["provider"],
            checkpoint_path=metadata["checkpoint_path"],
            source_path=metadata["source_path"],
            beat_prob=beat_prob,
            downbeat_prob=downbeat_prob,
            frame_rate_hz=metadata["frame_rate_hz"],
        )
    except ValueError as exc:
        raise BeatThisFramePredictionCacheError(f"invalid prediction in cache {cache_path}: {exc}") from exc


def _load_metadata(value: object, *, cache_path: Path) -> dict[str, Any]:
    array = np.asarray(value)
    if array.shape != ():
        raise BeatThisFramePredictionCacheError(f"cache {cache_path} metadata_json must be a scalar string")
    raw_json = array.item()
    if isinstance(raw_json, bytes):
        raw_json = raw_json.decode("utf-8")
    if not isinstance(raw_json, str):
        raise BeatThisFramePredictionCacheError(f"cache {cache_path} metadata_json must be a string")
    try:
        metadata = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise BeatThisFramePredictionCacheError(f"cache {cache_path} metadata_json is not valid JSON") from exc
    if not isinstance(metadata, dict):
        raise BeatThisFramePredictionCacheError(f"cache {cache_path} metadata_json must decode to an object")
    return metadata


def _validate_metadata(
    metadata: dict[str, Any],
    *,
    audio_cache_key: str,
    config: BeatThisFramePredictionCacheConfig,
    cache_path: Path,
) -> None:
    required_keys = {
        "format",
        "schema_version",
        "provider",
        "cache_version",
        "checkpoint_path",
        "float16",
        "shift_ms",
        "frame_rate_hz",
        "audio_cache_key_sha256",
        "frame_count",
        "source_path",
    }
    keys = set(metadata)
    if keys != required_keys:
        raise BeatThisFramePredictionCacheError(
            f"cache {cache_path} metadata keys must be {sorted(required_keys)}, got {sorted(keys)}"
        )

    expected = {
        "format": BEATTHIS_FRAME_PREDICTION_CACHE_FORMAT,
        "schema_version": BEATTHIS_FRAME_PREDICTION_CACHE_SCHEMA_VERSION,
        "provider": BEATTHIS_PROVIDER_NAME,
        "cache_version": config.cache_version,
        "checkpoint_path": config.checkpoint_path,
        "float16": config.float16,
        "audio_cache_key_sha256": _audio_cache_key_hash(audio_cache_key),
    }
    for key, expected_value in expected.items():
        if metadata[key] != expected_value:
            raise BeatThisFramePredictionCacheError(
                f"cache {cache_path} metadata {key} mismatch: expected {expected_value!r}, got {metadata[key]!r}"
            )

    for key, expected_value in (("shift_ms", config.shift_ms), ("frame_rate_hz", config.frame_rate_hz)):
        value = metadata[key]
        if not isinstance(value, (int, float)) or not math.isclose(
            float(value),
            expected_value,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise BeatThisFramePredictionCacheError(
                f"cache {cache_path} metadata {key} mismatch: expected {expected_value!r}, got {value!r}"
            )
        metadata[key] = float(value)

    frame_count = metadata["frame_count"]
    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count < 0:
        raise BeatThisFramePredictionCacheError(f"cache {cache_path} metadata frame_count must be a non-negative int")

    source_path = metadata["source_path"]
    if source_path is not None and not isinstance(source_path, str):
        raise BeatThisFramePredictionCacheError(f"cache {cache_path} metadata source_path must be a string or null")


def _load_probability_vector(value: object, *, name: str, cache_path: Path) -> NDArray[np.float32]:
    raw_array = np.asarray(value)
    if raw_array.dtype != np.float32:
        raise BeatThisFramePredictionCacheError(
            f"cache {cache_path} {name} must have dtype float32, got {raw_array.dtype}"
        )
    array = np.asarray(raw_array, dtype=np.float32)
    if array.ndim != 1:
        raise BeatThisFramePredictionCacheError(f"cache {cache_path} {name} must be a 1-D vector")
    if not np.all(np.isfinite(array)):
        raise BeatThisFramePredictionCacheError(f"cache {cache_path} {name} must contain only finite values")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise BeatThisFramePredictionCacheError(f"cache {cache_path} {name} must contain probabilities in [0, 1]")
    return np.ascontiguousarray(array, dtype=np.float32)


def _validate_prediction_for_config(
    prediction: FrameTimingPrediction,
    *,
    config: BeatThisFramePredictionCacheConfig,
) -> None:
    if prediction.provider != BEATTHIS_PROVIDER_NAME:
        raise ValueError(f"prediction provider must be {BEATTHIS_PROVIDER_NAME!r}, got {prediction.provider!r}")
    if prediction.checkpoint_path != config.checkpoint_path:
        raise ValueError(
            f"prediction checkpoint_path must match cache config {config.checkpoint_path!r}, "
            f"got {prediction.checkpoint_path!r}"
        )
    if not math.isclose(prediction.frame_rate_hz, config.frame_rate_hz, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"prediction frame_rate_hz must match cache config {config.frame_rate_hz!r}, "
            f"got {prediction.frame_rate_hz!r}"
        )


def _metadata_for_prediction(
    prediction: FrameTimingPrediction,
    *,
    audio_cache_key: str,
    config: BeatThisFramePredictionCacheConfig,
) -> dict[str, Any]:
    return {
        "format": BEATTHIS_FRAME_PREDICTION_CACHE_FORMAT,
        "schema_version": BEATTHIS_FRAME_PREDICTION_CACHE_SCHEMA_VERSION,
        "provider": BEATTHIS_PROVIDER_NAME,
        "cache_version": config.cache_version,
        "checkpoint_path": config.checkpoint_path,
        "float16": config.float16,
        "shift_ms": config.shift_ms,
        "frame_rate_hz": config.frame_rate_hz,
        "audio_cache_key_sha256": _audio_cache_key_hash(audio_cache_key),
        "frame_count": prediction.frame_count,
        "source_path": prediction.source_path,
    }


def _beatthis_frame_prediction_config_fingerprint(config: BeatThisFramePredictionCacheConfig) -> str:
    payload = {
        "version": config.cache_version,
        "provider": BEATTHIS_PROVIDER_NAME,
        "checkpoint_path": config.checkpoint_path,
        "float16": config.float16,
        "shift_ms": config.shift_ms,
        "frame_rate_hz": config.frame_rate_hz,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _audio_cache_key_hash(audio_cache_key: str) -> str:
    if not isinstance(audio_cache_key, str) or not audio_cache_key:
        raise ValueError("audio_cache_key must be a non-empty string")
    return hashlib.sha256(audio_cache_key.encode("utf-8")).hexdigest()
