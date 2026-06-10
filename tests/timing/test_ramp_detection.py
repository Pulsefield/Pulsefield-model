import unittest

from pulsefield_model.timing.ramp_detection import (
    TimingRampDetectionConfig,
    detect_timing_ramp,
)
from pulsefield_model.timing.schema import FittedTimingGrid, TimingSegment


def _grid_from_bpms(
    bpms: list[float],
    *,
    start_ms: float = 0.0,
    step_ms: float = 1000.0,
) -> FittedTimingGrid:
    return FittedTimingGrid(
        tuple(
            TimingSegment(offset_ms=start_ms + index * step_ms, beat_length_ms=60000.0 / bpm)
            for index, bpm in enumerate(bpms)
        )
    )


class TimingRampDetectionTests(unittest.TestCase):
    def test_detects_continuous_linear_bpm_ramp(self) -> None:
        grid = _grid_from_bpms([100.0, 112.0, 124.0, 136.0, 148.0, 160.0, 172.0, 184.0])

        detection = detect_timing_ramp(grid)

        self.assertTrue(detection.is_ramp)
        self.assertEqual(detection.candidate_status, "strict_ramp")
        self.assertEqual(detection.family, "linear_ramp")
        self.assertEqual(detection.reasons, ("long_continuous_monotonic_ramp",))
        self.assertIsNotNone(detection.best_run)
        assert detection.best_run is not None
        self.assertEqual(detection.best_run.length, 8)
        self.assertAlmostEqual(detection.best_run.bpm_span, 84.0)

    def test_rejects_short_two_tempo_change(self) -> None:
        grid = _grid_from_bpms([120.0, 180.0], step_ms=8000.0)

        detection = detect_timing_ramp(grid)

        self.assertFalse(detection.is_ramp)
        self.assertEqual(detection.candidate_status, "reject")
        self.assertIn("too_few_points", detection.reasons)
        self.assertIn("low_bpm_span", detection.reasons)

    def test_marks_low_span_smooth_shape_as_borderline_candidate(self) -> None:
        grid = _grid_from_bpms([232.0, 228.5, 228.0, 220.0, 212.0, 200.0, 195.0, 180.0, 165.0, 160.0])

        detection = detect_timing_ramp(grid)

        self.assertFalse(detection.is_ramp)
        self.assertEqual(detection.candidate_status, "borderline_low_span")
        self.assertEqual(detection.reasons, ("low_bpm_span",))
        self.assertIsNotNone(detection.best_run)
        assert detection.best_run is not None
        self.assertAlmostEqual(detection.best_run.bpm_span, 72.0)

    def test_splits_temporally_disconnected_monotonic_runs(self) -> None:
        grid = FittedTimingGrid(
            (
                TimingSegment(offset_ms=0.0, beat_length_ms=60000.0 / 100.0),
                TimingSegment(offset_ms=1000.0, beat_length_ms=60000.0 / 112.0),
                TimingSegment(offset_ms=2000.0, beat_length_ms=60000.0 / 124.0),
                TimingSegment(offset_ms=3000.0, beat_length_ms=60000.0 / 136.0),
                TimingSegment(offset_ms=35000.0, beat_length_ms=60000.0 / 148.0),
                TimingSegment(offset_ms=36000.0, beat_length_ms=60000.0 / 160.0),
                TimingSegment(offset_ms=37000.0, beat_length_ms=60000.0 / 172.0),
                TimingSegment(offset_ms=38000.0, beat_length_ms=60000.0 / 184.0),
            )
        )

        detection = detect_timing_ramp(grid)

        self.assertFalse(detection.is_ramp)
        self.assertIn("too_few_points", detection.reasons)

    def test_rejects_spiky_unsmooth_monotonic_shape(self) -> None:
        grid = _grid_from_bpms([100.0, 190.0, 191.0, 280.0, 281.0, 370.0, 371.0, 372.0])

        detection = detect_timing_ramp(
            grid,
            config=TimingRampDetectionConfig(smooth_min_linear_r2=0.95),
        )

        self.assertFalse(detection.is_ramp)
        self.assertIn("weak_ramp_smoothness", detection.reasons)


if __name__ == "__main__":
    unittest.main()
