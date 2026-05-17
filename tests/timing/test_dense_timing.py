import math
import unittest

import numpy as np

from pulsefield_model.osu_core.timing import RedTimingPoint
from pulsefield_model.timing.rendering.dense_timing_v1 import (
    TIMING_TRACK_CHANNELS,
    TimingTrackConfig,
    render_timing_track_20ms_v1,
)
from pulsefield_model.timing.rendering.dense_timing_v2 import (
    DENSE_TIMING_V2_CHANNELS,
    DenseTimingV2Config,
    render_dense_timing_v2,
)
from pulsefield_model.timing.schema import FittedTimingGrid, TimingSegment


class DenseTimingRendererTests(unittest.TestCase):
    def test_v1_outputs_fixed_600_by_5_window_and_phase_unit_vectors(self) -> None:
        track = render_timing_track_20ms_v1(
            [RedTimingPoint(offset_ms=0.0, beat_length_ms=500.0)],
            input_start_ms=-2000.0,
            bpm_log_mean=math.log(120.0),
            bpm_log_std=1.0,
        )

        self.assertEqual(track.shape, (600, 5))
        self.assertEqual(
            TIMING_TRACK_CHANNELS,
            (
                "beat_pulse",
                "beat_phase_sin",
                "beat_phase_cos",
                "local_bpm_log_norm",
                "timing_confidence",
            ),
        )
        phase_norm = np.sqrt(track[:, 1] ** 2 + track[:, 2] ** 2)
        self.assertLess(float(np.max(np.abs(phase_norm - 1.0))), 1e-6)
        self.assertTrue(np.all(track[:, 4] == 1.0))

    def test_v1_clips_local_bpm_log_norm_and_rejects_bad_stats(self) -> None:
        fast_track = render_timing_track_20ms_v1(
            [RedTimingPoint(offset_ms=0.0, beat_length_ms=100.0)],
            input_start_ms=0.0,
            frame_count=1,
            bpm_log_mean=math.log(1.0),
            bpm_log_std=0.1,
            config=TimingTrackConfig(pulse_width_ms=40.0),
        )
        self.assertEqual(float(fast_track[0, 3]), 4.0)

        with self.assertRaisesRegex(ValueError, "bpm_log_std must be positive"):
            render_timing_track_20ms_v1(
                [RedTimingPoint(offset_ms=0.0, beat_length_ms=500.0)],
                input_start_ms=0.0,
                frame_count=1,
                bpm_log_mean=math.log(120.0),
                bpm_log_std=0.0,
            )

    def test_v2_outputs_four_channel_window_without_timing_confidence(self) -> None:
        track = render_dense_timing_v2(
            FittedTimingGrid(segments=(TimingSegment(offset_ms=0.0, beat_length_ms=500.0),)),
            input_start_ms=-2000.0,
        )

        self.assertEqual(track.shape, (600, 4))
        self.assertEqual(
            DENSE_TIMING_V2_CHANNELS,
            ("beat_pulse", "phase_sin", "phase_cos", "local_bpm"),
        )
        phase_norm = np.sqrt(track[:, 1] ** 2 + track[:, 2] ** 2)
        self.assertLess(float(np.max(np.abs(phase_norm - 1.0))), 1e-6)
        self.assertTrue(np.all(track[:, 3] == np.float32(120.0)))

    def test_v2_uses_frame_centers_pulses_phase_and_segment_local_bpm(self) -> None:
        track = render_dense_timing_v2(
            FittedTimingGrid(
                segments=(
                    TimingSegment(offset_ms=0.0, beat_length_ms=500.0),
                    TimingSegment(offset_ms=1000.0, beat_length_ms=250.0),
                )
            ),
            input_start_ms=960.0,
            frame_count=4,
        )

        np.testing.assert_allclose(track[:, 0], np.asarray([0.25, 0.75, 0.75, 0.25], dtype=np.float32))
        np.testing.assert_allclose(track[:, 3], np.asarray([120.0, 120.0, 240.0, 240.0], dtype=np.float32))

        expected_phase = np.asarray([0.94, 0.98, 0.04, 0.12], dtype=np.float64)
        np.testing.assert_allclose(track[:, 1], np.sin(2.0 * math.pi * expected_phase), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(track[:, 2], np.cos(2.0 * math.pi * expected_phase), rtol=1e-6, atol=1e-6)

    def test_v2_rejects_invalid_config_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "segments must be non-empty"):
            FittedTimingGrid(segments=())

        with self.assertRaisesRegex(ValueError, "pulse_width_ms must be positive"):
            render_dense_timing_v2(
                FittedTimingGrid(segments=(TimingSegment(offset_ms=0.0, beat_length_ms=500.0),)),
                input_start_ms=0.0,
                frame_count=1,
                config=DenseTimingV2Config(pulse_width_ms=0.0),
            )


if __name__ == "__main__":
    unittest.main()
