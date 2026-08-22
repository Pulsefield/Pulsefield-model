import importlib.util
import unittest
from dataclasses import replace

import numpy as np

from pulsefield_model.features.mir_backbone import MIRBackboneConfig
from pulsefield_model.features.mir_backbone import build_mir_backbone_from_log_mel
from pulsefield_model.features.mir_backbone import classical_plp
from pulsefield_model.features.mir_backbone import complex_fourier_tempogram
from pulsefield_model.features.mir_backbone import compute_log_mel_5ms
from pulsefield_model.features.mir_backbone import interpolate_frame_centers
from pulsefield_model.features.mir_backbone import mel_center_frequencies_hz
from pulsefield_model.features.mir_backbone import mir_probe_features
from pulsefield_model.features.mir_backbone import spectral_flux_novelty


class MIRBackboneConfigTests(unittest.TestCase):
    def test_default_clocks_and_channels_are_explicit(self) -> None:
        config = MIRBackboneConfig()

        self.assertEqual(config.mel_hop_length, 120)
        self.assertEqual(config.mel_window_length, 960)
        self.assertEqual(config.tempogram_stride, 4)
        self.assertEqual(config.tempogram_window_seconds, (8.0,))
        self.assertEqual(config.tempo_bpms.shape, (96,))
        self.assertEqual((config.tempo_min_bpm, config.tempo_max_bpm), (30.0, 600.0))

    def test_tempogram_hop_must_be_an_integer_multiple_of_mel_hop(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive multiple"):
            MIRBackboneConfig(tempogram_hop_ms=12)


class SpectralFluxTests(unittest.TestCase):
    def test_positive_flux_has_broadband_and_four_frequency_bands(self) -> None:
        config = MIRBackboneConfig(novelty_local_average_ms=0)
        frequencies = mel_center_frequencies_hz(config)
        selected_bins = [
            int(np.flatnonzero((frequencies >= low) & (frequencies < high))[0])
            for low, high in zip(
                config.novelty_band_edges_hz[:-1],
                config.novelty_band_edges_hz[1:],
            )
        ]
        mel = np.zeros((4, config.mel_bins), dtype=np.float32)
        mel[1, selected_bins] = 1.0
        mel[2, selected_bins] = 0.5
        mel[3, selected_bins] = 2.0

        novelty, valid = spectral_flux_novelty(mel, config=config)

        self.assertEqual(novelty.shape, (4, 5))
        np.testing.assert_array_equal(valid, [False, True, True, True])
        self.assertGreater(novelty[1, 0], 0.0)
        np.testing.assert_allclose(novelty[2], 0.0)
        self.assertTrue(np.all(novelty[3] > novelty[1]))

    def test_flux_validity_requires_both_differenced_frames(self) -> None:
        config = MIRBackboneConfig(novelty_local_average_ms=0)
        mel = np.arange(4, dtype=np.float32)[:, None] * np.ones((1, config.mel_bins), dtype=np.float32)

        novelty, valid = spectral_flux_novelty(
            mel,
            frame_valid=np.array([True, False, True, True]),
            config=config,
        )

        np.testing.assert_array_equal(valid, [False, False, False, True])
        np.testing.assert_allclose(novelty[:3], 0.0)
        self.assertTrue(np.all(novelty[3] > 0.0))


class TempogramAndPLPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MIRBackboneConfig(
            tempogram_window_seconds=(2.0,),
            tempo_min_bpm=60.0,
            tempo_max_bpm=240.0,
            tempo_bins=3,
        )

    def test_complex_tempogram_recovers_tempo_and_marks_centered_support(self) -> None:
        times = np.arange(0.0, 8.0 + 0.0025, 0.005, dtype=np.float64)
        novelty = (1.0 + np.cos(2.0 * np.pi * 2.0 * times)).astype(np.float32)

        centers, tempogram, valid = complex_fourier_tempogram(
            novelty,
            times,
            config=self.config,
        )

        self.assertAlmostEqual(float(centers[1] - centers[0]), 0.02, places=6)
        middle = int(np.argmin(np.abs(centers - 4.0)))
        self.assertTrue(valid[middle, 0])
        self.assertFalse(valid[0, 0])
        self.assertEqual(int(np.argmax(np.abs(tempogram[middle, 0]))), 1)
        self.assertAlmostEqual(float(self.config.tempo_bpms[1]), 120.0, places=5)
        self.assertTrue(np.iscomplexobj(tempogram))
        relative_times = times - centers[middle]
        in_window = np.abs(relative_times) <= 1.0
        hann = 0.5 + 0.5 * np.cos(np.pi * relative_times[in_window])
        expected = np.sum(
            novelty[in_window]
            * hann
            * np.exp(-2j * np.pi * 2.0 * relative_times[in_window])
        ) / np.sum(hann)
        np.testing.assert_allclose(tempogram[middle, 0, 1], expected, rtol=1e-5, atol=1e-6)

    def test_precomputed_log_mel_builds_clocked_teacher_outputs(self) -> None:
        config = MIRBackboneConfig(
            tempogram_window_seconds=(0.5,),
            tempo_min_bpm=60.0,
            tempo_max_bpm=240.0,
            tempo_bins=3,
        )
        times = np.arange(0.02, 2.02, 0.005, dtype=np.float64)
        rng = np.random.default_rng(7)
        log_mel = rng.standard_normal((times.size, config.mel_bins)).astype(np.float32)

        backbone = build_mir_backbone_from_log_mel(log_mel, times, config=config)

        self.assertEqual(backbone.novelty.shape, (times.size, 5))
        self.assertEqual(backbone.tempogram.shape, (100, 1, 3))
        self.assertEqual(backbone.tempogram_valid.shape, (100, 1))
        self.assertEqual(backbone.signed_plp.shape, (times.size, 1))
        self.assertEqual(backbone.frame_centers_s.dtype, np.dtype("float64"))
        self.assertEqual(backbone.tempogram_centers_s.dtype, np.dtype("float64"))

        probe = mir_probe_features(backbone, config=config)
        self.assertEqual(probe.acoustic.shape, (times.size, 128))
        self.assertEqual(probe.novelty.shape, (times.size, 5))
        self.assertEqual(probe.tempogram.shape, (100, 29))
        self.assertEqual(probe.pulse.shape, (times.size, 6))

        with self.assertRaisesRegex(ValueError, "exactly match"):
            mir_probe_features(backbone, config=replace(config, tempo_max_bpm=300.0))

    def test_classical_plp_overlap_add_preserves_global_phase(self) -> None:
        target_times = np.arange(0.0, 8.0 + 0.0025, 0.005, dtype=np.float64)
        tempogram_times = np.arange(1.0, 7.0 + 0.01, 0.02, dtype=np.float64)
        tempogram = np.zeros((tempogram_times.size, 1, 3), dtype=np.complex64)
        omega = 2.0 * np.pi * 2.0
        tempogram[:, 0, 1] = np.exp(1j * omega * tempogram_times)

        plp, valid = classical_plp(
            tempogram,
            tempogram_times,
            target_times,
            config=self.config,
        )

        peak = int(np.argmin(np.abs(target_times - 4.0)))
        trough = int(np.argmin(np.abs(target_times - 4.25)))
        self.assertTrue(valid[peak, 0])
        self.assertTrue(valid[trough, 0])
        self.assertGreater(plp[peak, 0], 0.9)
        self.assertAlmostEqual(float(plp[trough, 0]), 0.0, places=5)
        self.assertFalse(valid[0, 0])


class FrameInterpolationTests(unittest.TestCase):
    def test_interpolation_respects_exact_centers_brackets_and_no_extrapolation(self) -> None:
        values = np.array(
            [[0.0 + 0.0j, 10.0 + 0.0j], [2.0 + 2.0j, 20.0 + 2.0j], [4.0 + 4.0j, 30.0 + 4.0j]],
            dtype=np.complex64,
        )
        valid = np.array([[True, False], [True, True], [False, True]])

        interpolated, interpolated_valid = interpolate_frame_centers(
            values,
            [0.0, 1.0, 2.0],
            [-0.1, 0.5, 1.0, 1.5, 2.1],
            valid=valid,
        )

        np.testing.assert_array_equal(
            interpolated_valid,
            [[False, False], [True, False], [True, True], [False, True], [False, False]],
        )
        self.assertEqual(interpolated.dtype, np.dtype("complex64"))
        self.assertEqual(interpolated[1, 0], np.complex64(1.0 + 1.0j))
        self.assertEqual(interpolated[1, 1], 0.0)
        np.testing.assert_array_equal(interpolated[2], values[1])
        self.assertEqual(interpolated[3, 0], 0.0)
        self.assertEqual(interpolated[3, 1], np.complex64(25.0 + 3.0j))


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None and importlib.util.find_spec("nnAudio") is not None,
    "requires torch and nnAudio",
)
class LogMelIntegrationTests(unittest.TestCase):
    def test_log_mel_reports_frame_centers_and_padded_tail_validity(self) -> None:
        config = MIRBackboneConfig()
        waveform = np.zeros(config.sample_rate // 10, dtype=np.float32)

        log_mel, centers, valid = compute_log_mel_5ms(
            waveform,
            sample_rate=config.sample_rate,
            config=config,
        )

        self.assertEqual(log_mel.shape, (20, 128))
        self.assertAlmostEqual(float(centers[0]), 0.0, places=6)
        self.assertAlmostEqual(float(centers[1] - centers[0]), 0.005, places=6)
        self.assertEqual(int(valid.sum()), 13)
        self.assertTrue(np.all(np.isfinite(log_mel)))


if __name__ == "__main__":
    unittest.main()
