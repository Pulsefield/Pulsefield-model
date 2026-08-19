from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pulsefield_model.features.build_mel_cache import (
    MelCacheBuildConfig,
    MusicMelSection,
    build_mel_cache,
    compose_mel_cache_build_config,
)
from pulsefield_model.features.mel import music_log_mel_cache_path
from pulsefield_model.features.mel_base import MUSIC_MEL_CACHE_CONFIG, MelCacheConfig


class MelCacheBuildConfigTests(unittest.TestCase):
    def test_default_hydra_config_selects_music_frontend_and_full5050(self) -> None:
        config = compose_mel_cache_build_config()

        self.assertEqual(config.expected_row_count, 5050)
        self.assertEqual(config.mel.sample_rate, 24000)
        self.assertEqual(config.mel.mel_bins, 128)
        self.assertEqual(config.mel.win_length, 960)
        self.assertEqual(config.mel.cache_version, MUSIC_MEL_CACHE_CONFIG.cache_version)

    def test_unknown_hydra_override_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown Mel cache build config keys"):
            compose_mel_cache_build_config(overrides=("+unexpected=true",))


class MelCacheBuilderTests(unittest.TestCase):
    def test_build_is_resumable_and_does_not_use_legacy_cache_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            audio_paths = (root / "one.ogg", root / "two.mp3")
            for path in audio_paths:
                path.write_bytes(b"audio-placeholder")
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                "".join(json.dumps({"resolved_audio_path": path.as_posix()}) + "\n" for path in audio_paths),
                encoding="utf-8",
            )
            section = MusicMelSection(cache_root=(root / "cache").as_posix())
            config = MelCacheBuildConfig(
                manifest_path=manifest_path.as_posix(),
                expected_row_count=2,
                progress_every=1,
                mel=section,
            )
            mel_config = MelCacheConfig(
                cache_root=root / "cache",
                cache_version=section.cache_version,
                sample_rate=section.sample_rate,
                mel_bins=section.mel_bins,
                hop_ms=section.hop_ms,
                n_fft=section.n_fft,
                win_length=section.win_length,
                fmin=section.fmin,
                fmax=section.fmax,
            )

            def fake_builder(audio_path: Path, received: MelCacheConfig) -> None:
                self.assertEqual(received, mel_config)
                cache_path = music_log_mel_cache_path(audio_path, config=received)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, np.zeros((2, 128), dtype=np.float32))

            first = build_mel_cache(config, mel_builder=fake_builder)
            second = build_mel_cache(config, mel_builder=fake_builder)

            self.assertEqual((first.total, first.created, first.existing), (2, 2, 0))
            self.assertEqual((second.total, second.created, second.existing), (2, 0, 2))
            legacy_path = music_log_mel_cache_path(
                audio_paths[0],
                config=MelCacheConfig(cache_root=root / "cache"),
            )
            self.assertFalse(legacy_path.exists())


if __name__ == "__main__":
    unittest.main()
