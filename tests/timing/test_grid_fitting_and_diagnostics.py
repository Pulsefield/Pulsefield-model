import unittest

import numpy as np

import pulsefield_model.timing.grid_fitting.scoring as scoring_module
from pulsefield_model.timing.grid_fitting.alias import (
    _AliasOption,
    _alias_is_semantic_promotion,
    _alias_path_is_acceptable,
)
from pulsefield_model.timing.grid_fitting.change_detection import _detect_change_split_candidates
from pulsefield_model.timing.grid_fitting.refinement import _refine_timing_segments
from pulsefield_model.timing.diagnostics.compare_to_oracle import compare_timing_grids
from pulsefield_model.timing.canonicalization import (
    TIMING_CANONICALIZATION_BPM_80_160,
    canonical_bpm_80_160,
    canonicalize_timing_grid,
)
from pulsefield_model.timing.grid_fitting import GridFitter, GridFitterConfig
from pulsefield_model.timing.grid_fitting.types import _SegmentFit
from pulsefield_model.timing.schema import FittedTimingGrid, FrameTimingPrediction, TimingSegment


def _pulse_probabilities(
    frame_times_ms: np.ndarray,
    beat_times_ms: list[float],
    *,
    pulse_width_ms: float = 40.0,
    baseline: float = 0.05,
) -> np.ndarray:
    probabilities = np.full(frame_times_ms.shape, baseline, dtype=np.float64)
    for beat_time_ms in beat_times_ms:
        distance_ms = np.abs(frame_times_ms - beat_time_ms)
        probabilities = np.maximum(probabilities, np.maximum(0.0, 1.0 - distance_ms / pulse_width_ms))
    return probabilities.astype(np.float32)


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


def _multi_tempo_prediction(
    *,
    frame_count: int = 1000,
    frame_rate_hz: float = 50.0,
    boundary_ms: float = 8000.0,
    first_beat_length_ms: float = 500.0,
    second_beat_length_ms: float = 400.0,
    downbeat_every: int = 4,
) -> FrameTimingPrediction:
    frame_times_ms = np.arange(frame_count, dtype=np.float64) / frame_rate_hz * 1000.0
    duration_ms = float(frame_times_ms[-1])
    first_beats = list(np.arange(0.0, boundary_ms, first_beat_length_ms, dtype=np.float64))
    second_beats = list(np.arange(boundary_ms, duration_ms + second_beat_length_ms, second_beat_length_ms, dtype=np.float64))
    beat_times_ms = [*first_beats, *second_beats]
    downbeat_times_ms = [
        beat_time_ms
        for index, beat_time_ms in enumerate(beat_times_ms)
        if index % downbeat_every == 0 or np.isclose(beat_time_ms, boundary_ms)
    ]
    return FrameTimingPrediction(
        provider="unit-test",
        beat_prob=_pulse_probabilities(frame_times_ms, beat_times_ms),
        downbeat_prob=_pulse_probabilities(frame_times_ms, downbeat_times_ms, baseline=0.0),
        frame_rate_hz=frame_rate_hz,
    )


def _piecewise_beat_probabilities(
    frame_times_ms: np.ndarray,
    segments: tuple[TimingSegment, ...],
    *,
    pulse_width_ms: float = 40.0,
) -> np.ndarray:
    beat_times_ms: list[float] = []
    duration_ms = float(frame_times_ms[-1])
    for index, segment in enumerate(segments):
        end_time_ms = duration_ms
        if index + 1 < len(segments):
            end_time_ms = segments[index + 1].offset_ms
        last_beat_index = int(np.ceil((end_time_ms - segment.offset_ms) / segment.beat_length_ms))
        beat_times_ms.extend(
            float(segment.offset_ms + beat_index * segment.beat_length_ms)
            for beat_index in range(last_beat_index + 1)
        )
    return _pulse_probabilities(frame_times_ms, beat_times_ms, pulse_width_ms=pulse_width_ms).astype(np.float64)


def _fit_for_change_detection(
    *,
    frame_count: int = 1000,
    frame_rate_hz: float = 50.0,
    beat_length_ms: float = 500.0,
) -> tuple[_SegmentFit, np.ndarray]:
    frame_times_ms = np.arange(frame_count, dtype=np.float64) / frame_rate_hz * 1000.0
    return (
        _SegmentFit(
            start_frame=0,
            end_frame=frame_count,
            score=0.9,
            beat_length_ms=beat_length_ms,
            offset_ms=0.0,
            half_tempo_score=0.0,
            double_tempo_score=0.0,
            raw_bpm=60000.0 / beat_length_ms,
            raw_score=0.9,
            tempo_multiplier=1.0,
            candidate_count=1,
        ),
        frame_times_ms,
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

    def test_grid_fitter_refines_offset_to_integer_millisecond(self) -> None:
        prediction = _sample_prediction(
            frame_count=2000,
            frame_rate_hz=1000.0,
            offset_ms=123.0,
            beat_length_ms=500.0,
        )

        result = GridFitter(GridFitterConfig(min_bpm=100.0, max_bpm=140.0, bpm_step=1.0)).fit(prediction)

        self.assertEqual(result.grid.segments[0].offset_ms, 123.0)

    def test_change_detection_finds_peak_walk_bpm_boundary(self) -> None:
        fit, frame_times_ms = _fit_for_change_detection()
        first_beats = list(np.arange(0.0, 8000.0, 500.0, dtype=np.float64))
        second_beats = list(np.arange(8000.0, 16000.0, 400.0, dtype=np.float64))
        signal = _pulse_probabilities(frame_times_ms, [*first_beats, *second_beats]).astype(np.float64)
        config = GridFitterConfig(min_segment_duration_ms=4000.0)

        candidates = _detect_change_split_candidates(
            signal,
            frame_times_ms=frame_times_ms,
            downbeat_signal=np.zeros_like(signal),
            fit=fit,
            config=config,
        )

        self.assertTrue(candidates)
        self.assertAlmostEqual(frame_times_ms[candidates[0].frame], 8000.0, delta=40.0)

    def test_change_detection_prefers_downbeat_boundary(self) -> None:
        fit, frame_times_ms = _fit_for_change_detection()
        first_beats = list(np.arange(0.0, 8000.0, 500.0, dtype=np.float64))
        second_beats = list(np.arange(7800.0, 16000.0, 400.0, dtype=np.float64))
        signal = _pulse_probabilities(frame_times_ms, [*first_beats, *second_beats]).astype(np.float64)
        downbeat_signal = _pulse_probabilities(frame_times_ms, [8000.0], baseline=0.0).astype(np.float64)
        config = GridFitterConfig(min_segment_duration_ms=4000.0, split_downbeat_signal_weight=0.5)

        candidates = _detect_change_split_candidates(
            signal,
            frame_times_ms=frame_times_ms,
            downbeat_signal=downbeat_signal,
            fit=fit,
            config=config,
        )

        self.assertTrue(candidates)
        self.assertAlmostEqual(frame_times_ms[candidates[0].frame], 8000.0, delta=40.0)

    def test_grid_fitter_recovers_two_tempo_boundary(self) -> None:
        prediction = _multi_tempo_prediction()

        result = GridFitter(
            GridFitterConfig(
                min_bpm=100.0,
                max_bpm=170.0,
                max_segments=2,
                min_segment_duration_ms=6000.0,
            )
        ).fit(prediction)

        self.assertEqual(len(result.grid.segments), 2)
        self.assertAlmostEqual(result.grid.segments[0].beat_length_ms, 500.0, delta=1e-6)
        self.assertAlmostEqual(result.grid.segments[1].beat_length_ms, 400.0, delta=1e-6)
        self.assertAlmostEqual(result.grid.segments[1].offset_ms, 8000.0, delta=40.0)

    def test_refinement_collapses_repeated_dominant_tempo_with_outlier_islands(self) -> None:
        frame_times_ms = np.arange(0.0, 130000.0, 20.0, dtype=np.float64)
        segments = (
            TimingSegment(offset_ms=725.421, beat_length_ms=60000.0 / 170.111111),
            TimingSegment(offset_ms=37785.123, beat_length_ms=60000.0 / 113.222222),
            TimingSegment(offset_ms=55971.906, beat_length_ms=60000.0 / 170.222222),
            TimingSegment(offset_ms=65308.962, beat_length_ms=60000.0 / 169.875),
            TimingSegment(offset_ms=93208.011, beat_length_ms=60000.0 / 122.333333),
            TimingSegment(offset_ms=103090.522, beat_length_ms=60000.0 / 244.777778),
            TimingSegment(offset_ms=117252.156, beat_length_ms=60000.0 / 125.222222),
        )
        signal = _piecewise_beat_probabilities(
            frame_times_ms,
            (TimingSegment(offset_ms=725.0, beat_length_ms=60000.0 / 170.0),),
        )

        refined = _refine_timing_segments(
            segments,
            frame_times_ms,
            beat_signal=signal,
            config=GridFitterConfig(),
        )

        self.assertEqual(len(refined), 1)
        self.assertAlmostEqual(refined[0].local_bpm, 170.0, delta=1e-6)
        self.assertEqual(refined[0].offset_ms, round(refined[0].offset_ms))

    def test_refinement_does_not_collapse_dominant_tempo_without_signal_guard(self) -> None:
        frame_times_ms = np.arange(0.0, 130000.0, 20.0, dtype=np.float64)
        segments = (
            TimingSegment(offset_ms=725.421, beat_length_ms=60000.0 / 170.111111),
            TimingSegment(offset_ms=37785.123, beat_length_ms=60000.0 / 113.222222),
            TimingSegment(offset_ms=55971.906, beat_length_ms=60000.0 / 170.222222),
            TimingSegment(offset_ms=65308.962, beat_length_ms=60000.0 / 169.875),
            TimingSegment(offset_ms=93208.011, beat_length_ms=60000.0 / 122.333333),
            TimingSegment(offset_ms=103090.522, beat_length_ms=60000.0 / 244.777778),
            TimingSegment(offset_ms=117252.156, beat_length_ms=60000.0 / 125.222222),
        )

        refined = _refine_timing_segments(segments, frame_times_ms, config=GridFitterConfig())

        self.assertGreater(len(refined), 1)

    def test_refinement_signal_guard_keeps_repeated_real_tempo_changes(self) -> None:
        frame_times_ms = np.arange(0.0, 56000.0, 20.0, dtype=np.float64)
        segments = (
            TimingSegment(offset_ms=0.0, beat_length_ms=400.0),
            TimingSegment(offset_ms=12000.0, beat_length_ms=600.0),
            TimingSegment(offset_ms=16200.0, beat_length_ms=400.0),
            TimingSegment(offset_ms=28200.0, beat_length_ms=600.0),
            TimingSegment(offset_ms=32400.0, beat_length_ms=400.0),
            TimingSegment(offset_ms=44400.0, beat_length_ms=600.0),
        )
        signal = _piecewise_beat_probabilities(frame_times_ms, segments)

        refined = _refine_timing_segments(
            segments,
            frame_times_ms,
            beat_signal=signal,
            config=GridFitterConfig(),
        )

        self.assertEqual(len(refined), len(segments))
        self.assertEqual([segment.beat_length_ms for segment in refined], [400.0, 600.0, 400.0, 600.0, 400.0, 600.0])

    def test_refinement_keeps_simple_two_tempo_structure(self) -> None:
        frame_times_ms = np.arange(0.0, 120000.0, 20.0, dtype=np.float64)
        segments = (
            TimingSegment(offset_ms=0.4, beat_length_ms=500.0),
            TimingSegment(offset_ms=60000.6, beat_length_ms=400.0),
        )

        refined = _refine_timing_segments(segments, frame_times_ms, config=GridFitterConfig())

        self.assertEqual(len(refined), 2)
        self.assertEqual([segment.offset_ms for segment in refined], [0.0, 60001.0])
        self.assertEqual([segment.beat_length_ms for segment in refined], [500.0, 400.0])

    def test_refinement_keeps_near_bpm_anchor_churn(self) -> None:
        frame_times_ms = np.arange(0.0, 90000.0, 20.0, dtype=np.float64)
        segments = (
            TimingSegment(offset_ms=0.1, beat_length_ms=60000.0 / 180.222),
            TimingSegment(offset_ms=12000.2, beat_length_ms=60000.0 / 181.778),
            TimingSegment(offset_ms=26000.3, beat_length_ms=60000.0 / 181.778),
            TimingSegment(offset_ms=43000.4, beat_length_ms=60000.0 / 180.222),
            TimingSegment(offset_ms=58000.5, beat_length_ms=60000.0 / 181.778),
            TimingSegment(offset_ms=73000.6, beat_length_ms=60000.0 / 181.778),
        )

        refined = _refine_timing_segments(segments, frame_times_ms, config=GridFitterConfig())

        self.assertEqual(len(refined), len(segments))
        self.assertEqual([segment.offset_ms for segment in refined], [0.0, 12000.0, 26000.0, 43000.0, 58000.0, 73001.0])

    def test_grid_fitter_does_not_move_short_speedup_boundary_early(self) -> None:
        prediction = _multi_tempo_prediction(frame_count=260, boundary_ms=2000.0)

        result = GridFitter(
            GridFitterConfig(
                min_bpm=100.0,
                max_bpm=170.0,
                max_segments=2,
                min_segment_duration_ms=1000.0,
                initial_batch_split_candidate_count=0,
            )
        ).fit(prediction)

        self.assertEqual(len(result.grid.segments), 2)
        self.assertAlmostEqual(result.grid.segments[0].beat_length_ms, 500.0, delta=1e-6)
        self.assertAlmostEqual(result.grid.segments[1].beat_length_ms, 400.0, delta=1e-6)
        self.assertAlmostEqual(result.grid.segments[1].offset_ms, 2000.0, delta=40.0)

    def test_grid_fitter_keeps_steady_tempo_with_downbeats_single_segment(self) -> None:
        frame_rate_hz = 50.0
        frame_times_ms = np.arange(1000, dtype=np.float64) / frame_rate_hz * 1000.0
        beat_times_ms = list(np.arange(0.0, float(frame_times_ms[-1]) + 500.0, 500.0, dtype=np.float64))
        downbeat_times_ms = beat_times_ms[::4]
        prediction = FrameTimingPrediction(
            provider="unit-test",
            beat_prob=_pulse_probabilities(frame_times_ms, beat_times_ms),
            downbeat_prob=_pulse_probabilities(frame_times_ms, downbeat_times_ms, baseline=0.0),
            frame_rate_hz=frame_rate_hz,
        )

        result = GridFitter(
            GridFitterConfig(
                min_bpm=100.0,
                max_bpm=140.0,
                max_segments=4,
                min_segment_duration_ms=4000.0,
            )
        ).fit(prediction)

        self.assertEqual(len(result.grid.segments), 1)
        self.assertAlmostEqual(result.grid.segments[0].beat_length_ms, 500.0, delta=1e-6)

    def test_canonical_bpm_80_160_uses_half_open_octave(self) -> None:
        self.assertEqual(
            [canonical_bpm_80_160(bpm) for bpm in (60.0, 80.0, 120.0, 160.0, 200.0, 240.0)],
            [120.0, 80.0, 120.0, 80.0, 100.0, 120.0],
        )

    def test_canonicalize_timing_grid_keeps_offsets_and_meters(self) -> None:
        grid = FittedTimingGrid(
            segments=(
                TimingSegment(offset_ms=0.0, beat_length_ms=1000.0, meter=3),
                TimingSegment(offset_ms=1000.0, beat_length_ms=250.0, meter=4),
            )
        )

        canonical = canonicalize_timing_grid(grid, canonicalization=TIMING_CANONICALIZATION_BPM_80_160)

        self.assertEqual([segment.offset_ms for segment in canonical.segments], [0.0, 1000.0])
        self.assertEqual([segment.meter for segment in canonical.segments], [3, 4])
        self.assertEqual([segment.local_bpm for segment in canonical.segments], [120.0, 120.0])

    def test_grid_fitter_canonicalization_skips_alias_dealiasing(self) -> None:
        prediction = _sample_prediction(offset_ms=120.0, beat_length_ms=250.0)

        result = GridFitter(
            GridFitterConfig(
                min_bpm=200.0,
                max_bpm=260.0,
                canonicalization=TIMING_CANONICALIZATION_BPM_80_160,
            )
        ).fit(prediction)

        segment = result.grid.segments[0]
        self.assertAlmostEqual(segment.local_bpm, 120.0, delta=1e-6)
        self.assertEqual(result.diagnostics.alias_candidate_count, 0)

    def test_grid_fitter_keeps_confident_low_octave_alias(self) -> None:
        prediction = _sample_prediction(offset_ms=120.0, beat_length_ms=60000.0 / 70.0)

        result = GridFitter(
            GridFitterConfig(
                min_bpm=60.0,
                max_bpm=220.0,
                max_segments=1,
            )
        ).fit(prediction)

        segment = result.grid.segments[0]
        self.assertAlmostEqual(segment.local_bpm, 70.0, delta=1e-6)
        self.assertAlmostEqual(segment.offset_ms, 120.0, delta=1e-6)
        self.assertEqual(result.diagnostics.tempo_multiplier_distribution, {"1": 1})

    def test_semantic_alias_promotion_is_confidence_gated(self) -> None:
        config = GridFitterConfig()

        self.assertTrue(
            _alias_is_semantic_promotion(
                70.0,
                140.0,
                score=0.567,
                current_score=0.771,
                config=config,
            )
        )
        self.assertFalse(
            _alias_is_semantic_promotion(
                72.0,
                144.0,
                score=0.576,
                current_score=0.797,
                config=config,
            )
        )
        self.assertTrue(
            _alias_is_semantic_promotion(
                90.8,
                181.6,
                score=0.410,
                current_score=0.637,
                config=config,
            )
        )
        self.assertFalse(
            _alias_is_semantic_promotion(
                88.1,
                176.2,
                score=0.516,
                current_score=0.800,
                config=config,
            )
        )
        self.assertFalse(
            _alias_is_semantic_promotion(
                90.0,
                180.0,
                score=0.01,
                current_score=0.50,
                config=config,
            )
        )

    def test_alias_path_rejects_weak_segment_collapse(self) -> None:
        config = GridFitterConfig()
        frame_times_ms = np.arange(1000, dtype=np.float64) * 20.0
        original = (
            _SegmentFit(
                start_frame=0,
                end_frame=500,
                score=0.9,
                beat_length_ms=600.0,
                offset_ms=0.0,
                half_tempo_score=0.0,
                double_tempo_score=0.0,
                raw_bpm=100.0,
                raw_score=0.9,
                tempo_multiplier=1.0,
                candidate_count=1,
            ),
            _SegmentFit(
                start_frame=500,
                end_frame=1000,
                score=0.9,
                beat_length_ms=300.0,
                offset_ms=0.0,
                half_tempo_score=0.0,
                double_tempo_score=0.0,
                raw_bpm=200.0,
                raw_score=0.9,
                tempo_multiplier=1.0,
                candidate_count=1,
            ),
        )
        weak_proposed = (
            _AliasOption(
                fit=_SegmentFit(
                    start_frame=0,
                    end_frame=500,
                    score=0.5,
                    beat_length_ms=300.0,
                    offset_ms=0.0,
                    half_tempo_score=0.0,
                    double_tempo_score=0.0,
                    raw_bpm=100.0,
                    raw_score=0.9,
                    tempo_multiplier=2.0,
                    candidate_count=1,
                ),
                local_score=0.5,
            ),
            _AliasOption(fit=original[1], local_score=0.9),
        )
        strong_proposed = (
            _AliasOption(
                fit=_SegmentFit(
                    start_frame=0,
                    end_frame=500,
                    score=0.8,
                    beat_length_ms=300.0,
                    offset_ms=0.0,
                    half_tempo_score=0.0,
                    double_tempo_score=0.0,
                    raw_bpm=100.0,
                    raw_score=0.9,
                    tempo_multiplier=2.0,
                    candidate_count=1,
                ),
                local_score=0.8,
            ),
            _AliasOption(fit=original[1], local_score=0.9),
        )

        self.assertFalse(
            _alias_path_is_acceptable(
                original,
                weak_proposed,
                frame_times_ms=frame_times_ms,
                config=config,
            )
        )
        self.assertTrue(
            _alias_path_is_acceptable(
                original,
                strong_proposed,
                frame_times_ms=frame_times_ms,
                config=config,
            )
        )

    def test_alias_path_rejects_low_bpm_promotion_in_complex_path(self) -> None:
        config = GridFitterConfig()
        frame_times_ms = np.arange(1000, dtype=np.float64) * 20.0
        original = tuple(
            _SegmentFit(
                start_frame=index * 100,
                end_frame=(index + 1) * 100,
                score=0.8,
                beat_length_ms=800.0 if index == 0 else 400.0,
                offset_ms=float(index * 8000),
                half_tempo_score=0.0,
                double_tempo_score=0.0,
                raw_bpm=75.0 if index == 0 else 150.0,
                raw_score=0.8,
                tempo_multiplier=1.0,
                candidate_count=1,
            )
            for index in range(config.alias_semantic_promotion_low_bpm_max_segments + 1)
        )
        proposed = (
            _AliasOption(
                fit=_SegmentFit(
                    start_frame=0,
                    end_frame=100,
                    score=0.75,
                    beat_length_ms=400.0,
                    offset_ms=0.0,
                    half_tempo_score=0.0,
                    double_tempo_score=0.0,
                    raw_bpm=75.0,
                    raw_score=0.8,
                    tempo_multiplier=2.0,
                    candidate_count=1,
                ),
                local_score=0.75,
            ),
            *(_AliasOption(fit=fit, local_score=fit.score) for fit in original[1:]),
        )

        self.assertFalse(
            _alias_path_is_acceptable(
                original,
                proposed,
                frame_times_ms=frame_times_ms,
                config=config,
            )
        )

    def test_grid_fitter_keeps_low_preferred_band_tempo(self) -> None:
        prediction = _sample_prediction(offset_ms=120.0, beat_length_ms=60000.0 / 85.0)

        result = GridFitter(
            GridFitterConfig(
                min_bpm=60.0,
                max_bpm=220.0,
                max_segments=1,
            )
        ).fit(prediction)

        segment = result.grid.segments[0]
        self.assertAlmostEqual(segment.local_bpm, 85.0, delta=0.5)
        self.assertEqual(result.diagnostics.tempo_multiplier_distribution, {"1": 1})

    def test_compare_identical_grids_has_zero_error(self) -> None:
        grid = FittedTimingGrid(segments=(TimingSegment(offset_ms=0.0, beat_length_ms=500.0),))

        comparison = compare_timing_grids(grid, grid, frame_count=50)

        self.assertEqual(comparison.beat_pulse_mae, 0.0)
        self.assertEqual(comparison.local_bpm_mae, 0.0)
        self.assertEqual(comparison.mean_phase_error_beats, 0.0)
        self.assertEqual(comparison.mean_phase_error_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
