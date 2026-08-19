import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

if importlib.util.find_spec("torch") is None or importlib.util.find_spec("nnAudio") is None:
    raise unittest.SkipTest("requires torch and nnAudio")

from pulsefield_model.features.mel import Stage2MelConfig
from pulsefield_model.features.mel import load_cached_music_log_mel
from pulsefield_model.features.mel import load_full_song_packed_mel_20ms
from pulsefield_model.features.mel import music_log_mel_cache_path
from pulsefield_model.features.mel import stage2_log_mel_cache_path
from pulsefield_model.features.mel_base import MUSIC_MEL_CACHE_CONFIG
from pulsefield_model.features.mel_base import MelCacheConfig


class Stage2MelCacheTests(unittest.TestCase):
    def test_full_song_mel_cache_hit_does_not_decode_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = Stage2MelConfig(mel_cache_config=MelCacheConfig(cache_root=root / "cache"))
            audio_path = root / "audio.mp3"
            audio_path.write_bytes(b"not-a-real-audio-file")
            cache_path = stage2_log_mel_cache_path(audio_path, config=config)
            cache_path.parent.mkdir(parents=True)
            np.save(cache_path, np.ones((3, 80), dtype=np.float32))

            with mock.patch(
                "pulsefield_model.features.mel.load_audio_file",
                side_effect=AssertionError("cache hit should not decode audio"),
            ):
                packed = load_full_song_packed_mel_20ms(audio_path, config=config)

        self.assertEqual(packed.shape, (2, 160))
        self.assertEqual(packed.dtype, np.dtype("float32"))


class MusicMelCacheTests(unittest.TestCase):
    def test_new_music_frontend_has_expected_geometry_and_separate_namespace(self) -> None:
        config = MUSIC_MEL_CACHE_CONFIG
        self.assertEqual(config.sample_rate, 24000)
        self.assertEqual(config.mel_bins, 128)
        self.assertEqual(config.hop_length, 240)
        self.assertEqual(config.n_fft, 960)
        self.assertEqual(config.win_length, 960)
        self.assertEqual(config.fmax, 12000.0)
        self.assertNotEqual(config.cache_dir, MelCacheConfig().cache_dir)

    def test_read_interface_is_read_only_and_supports_memory_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config = MelCacheConfig(
                **{
                    **MUSIC_MEL_CACHE_CONFIG.__dict__,
                    "cache_root": root / "cache",
                },
            )
            audio_path = root / "audio.ogg"
            audio_path.write_bytes(b"not-a-real-audio-file")

            with self.assertRaises(FileNotFoundError):
                load_cached_music_log_mel(audio_path, config=config)

            cache_path = music_log_mel_cache_path(audio_path, config=config)
            cache_path.parent.mkdir(parents=True)
            np.save(cache_path, np.ones((3, 128), dtype=np.float32))

            mel = load_cached_music_log_mel(audio_path, config=config, mmap_mode="r")

        self.assertEqual(mel.shape, (3, 128))
        self.assertEqual(mel.dtype, np.dtype("float32"))


if __name__ == "__main__":
    unittest.main()
