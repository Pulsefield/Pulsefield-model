from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from hydra import main as hydra_main
from omegaconf import DictConfig

from pulsefield_model.features.audio import load_audio_file
from pulsefield_model.features.mel import music_log_mel_cache_path
from pulsefield_model.features.mel_base import (
    MUSIC_MEL_CACHE_CONFIG,
    MelCacheConfig,
    compute_log_mel_10ms,
)


DEFAULT_FULL5050_MANIFEST_PATH = "artifacts/reports/timing/timing_v3_labels_v1.jsonl"
DEFAULT_FULL5050_ROW_COUNT = 5050
_CONFIG_PATH = "../configs/hydra"


@dataclass
class MusicMelSection:
    cache_root: str = MUSIC_MEL_CACHE_CONFIG.cache_root.as_posix()
    cache_version: str = MUSIC_MEL_CACHE_CONFIG.cache_version
    sample_rate: int = MUSIC_MEL_CACHE_CONFIG.sample_rate
    mel_bins: int = MUSIC_MEL_CACHE_CONFIG.mel_bins
    hop_ms: int = MUSIC_MEL_CACHE_CONFIG.hop_ms
    n_fft: int = MUSIC_MEL_CACHE_CONFIG.n_fft
    win_length: int = MUSIC_MEL_CACHE_CONFIG.win_length
    fmin: float = MUSIC_MEL_CACHE_CONFIG.fmin
    fmax: float = MUSIC_MEL_CACHE_CONFIG.fmax


@dataclass
class MelCacheBuildConfig:
    manifest_path: str = DEFAULT_FULL5050_MANIFEST_PATH
    expected_row_count: int = DEFAULT_FULL5050_ROW_COUNT
    progress_every: int = 25
    mel: MusicMelSection = field(default_factory=MusicMelSection)


@dataclass(frozen=True)
class MelCacheBuildSummary:
    total: int
    created: int
    existing: int
    elapsed_seconds: float
    cache_dir: Path


MelBuilder = Callable[[Path, MelCacheConfig], None]


def register_mel_cache_build_config() -> None:
    from hydra.core.config_store import ConfigStore

    ConfigStore.instance().store(name="mel_cache_build_schema", node=MelCacheBuildConfig)


register_mel_cache_build_config()


def compose_mel_cache_build_config(
    *,
    overrides: Sequence[str] = (),
    config_dir: Path | None = None,
) -> Any:
    from hydra import compose, initialize_config_dir

    register_mel_cache_build_config()
    resolved_config_dir = (
        Path(__file__).resolve().parents[1] / "configs" / "hydra"
        if config_dir is None
        else Path(config_dir).resolve()
    )
    with initialize_config_dir(version_base=None, config_dir=resolved_config_dir.as_posix()):
        config = compose(config_name="mel_cache_build", overrides=list(overrides))
    return validate_mel_cache_build_config(config)


def validate_mel_cache_build_config(config: Any) -> Any:
    from omegaconf import OmegaConf

    source_config = config if OmegaConf.is_config(config) else OmegaConf.structured(config)
    container = OmegaConf.to_container(source_config, resolve=True)
    if not isinstance(container, Mapping):
        raise ValueError("Mel cache build config must be a mapping")
    unknown_top_level = sorted(
        set(container) - {"manifest_path", "expected_row_count", "progress_every", "mel"},
    )
    if unknown_top_level:
        raise ValueError(f"unknown Mel cache build config keys: {unknown_top_level}")
    mel_section = container.get("mel")
    if not isinstance(mel_section, Mapping):
        raise ValueError("mel must be a mapping")
    unknown_mel = sorted(
        set(mel_section)
        - {
            "cache_root",
            "cache_version",
            "sample_rate",
            "mel_bins",
            "hop_ms",
            "n_fft",
            "win_length",
            "fmin",
            "fmax",
        },
    )
    if unknown_mel:
        raise ValueError(f"unknown Mel frontend config keys: {unknown_mel}")
    resolved = OmegaConf.merge(OmegaConf.structured(MelCacheBuildConfig), source_config)
    if not str(resolved.manifest_path):
        raise ValueError("manifest_path must be non-empty")
    _require_positive_int(resolved.expected_row_count, "expected_row_count")
    _require_positive_int(resolved.progress_every, "progress_every")
    mel_config_from_section(resolved.mel)
    return resolved


def mel_config_from_section(section: Any) -> MelCacheConfig:
    config = MelCacheConfig(
        cache_root=Path(str(section.cache_root)),
        cache_version=str(section.cache_version),
        sample_rate=int(section.sample_rate),
        mel_bins=int(section.mel_bins),
        hop_ms=int(section.hop_ms),
        n_fft=int(section.n_fft),
        win_length=int(section.win_length),
        fmin=float(section.fmin),
        fmax=float(section.fmax),
    )
    if not config.cache_version:
        raise ValueError("mel.cache_version must be non-empty")
    for name in ("sample_rate", "mel_bins", "hop_ms", "n_fft", "win_length"):
        _require_positive_int(getattr(config, name), f"mel.{name}")
    if not np.isfinite(config.fmin) or config.fmin < 0.0:
        raise ValueError("mel.fmin must be non-negative and finite")
    if not np.isfinite(config.fmax) or config.fmax <= config.fmin:
        raise ValueError("mel.fmax must be finite and greater than mel.fmin")
    if config.fmax > config.sample_rate / 2:
        raise ValueError("mel.fmax must not exceed the Nyquist frequency")
    return config


def build_mel_cache(
    config: Any,
    *,
    mel_builder: MelBuilder | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> MelCacheBuildSummary:
    resolved = validate_mel_cache_build_config(config)
    mel_config = mel_config_from_section(resolved.mel)
    manifest_path = Path(str(resolved.manifest_path))
    audio_paths = load_manifest_audio_paths(
        manifest_path,
        expected_row_count=int(resolved.expected_row_count),
    )
    builder = _build_one_music_mel if mel_builder is None else mel_builder
    started = clock()
    created = 0
    existing = 0
    for index, audio_path in enumerate(audio_paths, start=1):
        cache_path = _music_cache_path(audio_path, mel_config)
        if cache_path.is_file():
            existing += 1
        else:
            builder(audio_path, mel_config)
            if not cache_path.is_file():
                raise RuntimeError(f"Mel builder did not create expected cache: {cache_path}")
            created += 1
        if index % int(resolved.progress_every) == 0 or index == len(audio_paths):
            print(
                f"mel_cache_progress done={index}/{len(audio_paths)} "
                f"created={created} existing={existing}",
                flush=True,
            )
    return MelCacheBuildSummary(
        total=len(audio_paths),
        created=created,
        existing=existing,
        elapsed_seconds=clock() - started,
        cache_dir=mel_config.cache_dir,
    )


def load_manifest_audio_paths(path: Path, *, expected_row_count: int) -> tuple[Path, ...]:
    rows: list[Path] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"manifest line {line_number} must not be blank")
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ValueError(f"manifest line {line_number} must be a JSON object")
            audio_path = raw.get("resolved_audio_path")
            if not isinstance(audio_path, str) or not audio_path:
                raise ValueError(
                    f"manifest line {line_number} resolved_audio_path must be a non-empty string",
                )
            resolved_audio_path = Path(audio_path).expanduser().resolve()
            if not resolved_audio_path.is_file():
                raise FileNotFoundError(
                    f"manifest line {line_number} audio file is missing: {resolved_audio_path}",
                )
            rows.append(resolved_audio_path)
    if len(rows) != expected_row_count:
        raise ValueError(f"expected {expected_row_count} manifest rows, got {len(rows)} from {path}")
    if len(set(rows)) != len(rows):
        raise ValueError("manifest resolved_audio_path values must be unique")
    return tuple(rows)


def _build_one_music_mel(audio_path: Path, config: MelCacheConfig) -> None:
    waveform = load_audio_file(audio_path, sample_rate=config.sample_rate)
    mel = compute_log_mel_10ms(
        waveform,
        sample_rate=config.sample_rate,
        config=config,
    )
    cache_path = _music_cache_path(audio_path, config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=cache_path.parent,
            prefix=f".{cache_path.stem}.",
            suffix=".npy.tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.save(handle, mel)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, cache_path)
        except FileExistsError:
            # Another worker won the race. Never replace an existing cache.
            pass
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _music_cache_path(audio_path: Path, config: MelCacheConfig) -> Path:
    return music_log_mel_cache_path(audio_path, config=config)


def _require_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def run_mel_cache_build_config(config: Any) -> MelCacheBuildSummary:
    summary = build_mel_cache(config)
    print(
        "mel_cache_done "
        f"total={summary.total} created={summary.created} existing={summary.existing} "
        f"elapsed_s={summary.elapsed_seconds:.1f} cache_dir={summary.cache_dir}",
        flush=True,
    )
    return summary


@hydra_main(version_base="1.3", config_path=_CONFIG_PATH, config_name="mel_cache_build")
def _hydra_main(config: DictConfig) -> None:
    run_mel_cache_build_config(config)


def main(argv: Sequence[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else list(argv)
    with _patched_argv([sys.argv[0], *args]):
        _hydra_main()


@contextmanager
def _patched_argv(argv: list[str]):
    original = sys.argv
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = original


if __name__ == "__main__":
    main()
