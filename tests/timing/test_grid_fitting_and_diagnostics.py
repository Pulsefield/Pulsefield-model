import unittest

import numpy as np

import pulsefield_model.timing.grid_fitting.scoring as scoring_module
from pulsefield_model.timing.diagnostics.compare_to_oracle import compare_timing_grids
from pulsefield_model.timing.grid_fitting import GridFitter, GridFitterConfig
from pulsefield_model.timing.schema import FittedTimingGrid, FrameTimingPrediction, TimingSegment


def _sample_prediction(
    *,
    frame_count: int = 1000,
    frame_rate_hz: float = 50.0,
    offset_ms: float,
    beat_length_ms: float,
    pulse_width_ms: float = 40.0,
    baseline: float = 0.05,
) -> FrameTimingPrediction:
    frame_times_ms = np.arange(frame_count, dtype=np.float64) / frame_rate_hz * 1000.0
    beat_pos = (frame_times_ms - offset_ms) / beat_length_ms
    phase = beat_pos - np.floor(beat_pos)
    distance_ms = np.minimum(phase, 1.0 - phase) * beat_length_ms
    beat_prob = np.maximum(0.0, 1.0 - distance_ms / pulse_width_ms)
    beat_prob = np.maximum(beat_prob, baseline).astype(np.float32)
    return FrameTimingPrediction(
        provider="unit-test",
        beat_prob=beat_prob,
        downbeat_prob=np.zeros_like(beat_prob),
        frame_rate_hz=frame_rate_hz,
    )


class GridFittingDiagnosticsTests(unittest.TestCase):
    def test_expands_half_bpm_candidates_with_hardcoded_fractional_parts(self) -> None:
        config = GridFitterConfig(min_bpm=220.0, max_bpm=224.0)

        candidates = scoring_module._with_fractional_bpm_candidates(
            np.asarray([222.0, 222.5], dtype=np.float64),
            config=config,
        )

        self.assertTrue(np.any(np.isclose(candidates, 222.0 + 2.0 / 9.0)))
        self.assertTrue(np.any(np.isclose(candidates, 222.0 + 1.0 / 8.0)))
        self.assertTrue(np.any(np.isclose(candidates, 222.0 + 1.0 / 3.0)))
        self.assertTrue(np.any(np.isclose(candidates, 222.0 + 2.0 / 3.0)))

    def test_grid_fitter_recovers_single_tempo(self) -> None:
        prediction = _sample_prediction(offset_ms=120.0, beat_length_ms=500.0)

        result = GridFitter(GridFitterConfig(min_bpm=100.0, max_bpm=140.0)).fit(prediction)

        segment = result.grid.segments[0]
        self.assertAlmostEqual(segment.offset_ms, 120.0, delta=1e-6)
        self.assertAlmostEqual(segment.beat_length_ms, 500.0, delta=1e-6)
        self.assertGreater(result.score, 0.95)

    def test_compare_identical_grids_has_zero_error(self) -> None:
        grid = FittedTimingGrid(segments=(TimingSegment(offset_ms=0.0, beat_length_ms=500.0),))

        comparison = compare_timing_grids(grid, grid, frame_count=50)

        self.assertEqual(comparison.beat_pulse_mae, 0.0)
        self.assertEqual(comparison.local_bpm_mae, 0.0)
        self.assertEqual(comparison.mean_phase_error_beats, 0.0)
        self.assertEqual(comparison.mean_phase_error_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
