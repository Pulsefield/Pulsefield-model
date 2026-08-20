from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pulsefield_model.osu_core.hitobjects import ManiaHitObject, ManiaHitObjectKind
from pulsefield_model.osu_core.timing import RedTimingPoint
from pulsefield_model.timing.evaluation.evidence import (
    OBJECT_EVENT_HOLD_END,
    OBJECT_EVENT_START,
    REDLINE_EVIDENCE_JUMP_CANDIDATE,
    REDLINE_EVIDENCE_RAMP_CANDIDATE,
    REDLINE_EVIDENCE_STABLE,
    ObjectGridEvidenceConfig,
    ObjectGridEvidence,
    ObjectResidualStats,
    RedlineEvidenceConfig,
    _summarize_object_grid_evidence_reference,
    summarize_beatmap_timing_evidence,
    summarize_object_grid_evidence,
)


def _write_osu(
    path: Path,
    *,
    timing_lines: list[str],
    hitobject_lines: list[str],
) -> None:
    path.write_text(
        "\n".join(
            [
                "osu file format v14",
                "",
                "[General]",
                "Mode: 3",
                "",
                "[Difficulty]",
                "CircleSize:4",
                "",
                "[TimingPoints]",
                *timing_lines,
                "",
                "[HitObjects]",
                *hitobject_lines,
            ],
        ),
        encoding="utf-8",
    )


class TimingV3EvidenceTests(unittest.TestCase):
    def test_constant_redline_and_on_grid_objects_return_stable_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            osu_path = Path(tmp_dir) / "constant.osu"
            _write_osu(
                osu_path,
                timing_lines=["0,500,4,2,0,80,1,0"],
                hitobject_lines=[
                    "64,192,0,1,0,0:0:0:0:",
                    "192,192,625,1,0,0:0:0:0:",
                    "320,192,1000,128,0,1250:0:0:0:0:",
                ],
            )

            evidence = summarize_beatmap_timing_evidence(
                osu_path,
                object_grid_config=ObjectGridEvidenceConfig(
                    subdivisions=(1, 2, 4),
                    alias_multipliers=(1.0,),
                    inlier_threshold_ms=1.0,
                    min_start_events_for_support=1,
                    min_start_time_span_ms_for_support=0.0,
                ),
            )

        self.assertEqual(evidence.redlines.evidence_class, REDLINE_EVIDENCE_STABLE)
        self.assertFalse(evidence.redlines.ambiguous)
        self.assertEqual(evidence.redlines.redline_count, 1)
        self.assertEqual(evidence.redlines.unique_bpms, (120.0,))
        self.assertIn("no_significant_bpm_change", evidence.redlines.reasons)
        self.assertEqual(evidence.object_grid.start_event_count, 3)
        self.assertEqual(evidence.object_grid.hold_end_event_count, 1)
        self.assertAlmostEqual(evidence.object_grid.best_evidence.combined_stats.total_weight, 3.5)
        self.assertEqual(evidence.object_grid.best_subdivision, 4)
        self.assertEqual(evidence.object_grid.best_alias_multiplier, 1.0)
        self.assertTrue(evidence.object_grid.supported)
        self.assertFalse(evidence.object_grid.ambiguous)

    def test_sparse_bpm_change_returns_jump_candidate_without_oracle_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            osu_path = Path(tmp_dir) / "jump.osu"
            _write_osu(
                osu_path,
                timing_lines=[
                    "0,500,4,2,0,80,1,0",
                    "2000,400,4,2,0,80,1,0",
                ],
                hitobject_lines=[
                    "64,192,0,1,0,0:0:0:0:",
                    "64,192,500,1,0,0:0:0:0:",
                    "64,192,1500,1,0,0:0:0:0:",
                    "64,192,2000,1,0,0:0:0:0:",
                    "64,192,2400,1,0,0:0:0:0:",
                ],
            )

            evidence = summarize_beatmap_timing_evidence(osu_path)

        self.assertEqual(evidence.redlines.evidence_class, REDLINE_EVIDENCE_JUMP_CANDIDATE)
        self.assertTrue(evidence.redlines.jump_evidence)
        self.assertFalse(evidence.redlines.ambiguous)
        self.assertEqual(evidence.redlines.change_gaps_ms, (2000.0,))
        self.assertEqual(evidence.redlines.significant_change_count, 1)
        self.assertAlmostEqual(evidence.redlines.changes[0].previous_grid_phase_residual_ms, 0.0)
        self.assertFalse(hasattr(evidence.redlines, "oracle_label"))
        self.assertIn("sparse_significant_bpm_change", evidence.redlines.reasons)

    def test_monotonic_redline_sequence_returns_ramp_candidate_raw_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            osu_path = Path(tmp_dir) / "ramp.osu"
            _write_osu(
                osu_path,
                timing_lines=[
                    "0,500,4,2,0,80,1,0",
                    "4000,480,4,2,0,80,1,0",
                    "8000,460,4,2,0,80,1,0",
                    "12000,440,4,2,0,80,1,0",
                ],
                hitobject_lines=[
                    "64,192,0,1,0,0:0:0:0:",
                    "64,192,4000,1,0,0:0:0:0:",
                    "64,192,8000,1,0,0:0:0:0:",
                    "64,192,12000,1,0,0:0:0:0:",
                ],
            )

            evidence = summarize_beatmap_timing_evidence(
                osu_path,
                redline_config=RedlineEvidenceConfig(dense_redlines_per_minute=1000.0),
            )

        self.assertEqual(evidence.redlines.evidence_class, REDLINE_EVIDENCE_RAMP_CANDIDATE)
        self.assertTrue(evidence.redlines.ramp_candidate_evidence)
        self.assertEqual(evidence.redlines.ramp_direction, "increasing")
        self.assertGreater(evidence.redlines.ramp_linear_r2, 0.98)
        self.assertIn("monotonic_increasing_bpm", evidence.redlines.reasons)
        self.assertGreater(len(evidence.redlines.unique_bpms), 2)

    def test_inherited_timing_points_are_ignored_by_redline_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            osu_path = Path(tmp_dir) / "inherited.osu"
            _write_osu(
                osu_path,
                timing_lines=[
                    "0,500,4,2,0,80,1,0",
                    "1000,-50,4,2,0,80,0,0",
                    "2000,-80,4,2,0,80,0,0",
                ],
                hitobject_lines=[
                    "64,192,0,1,0,0:0:0:0:",
                    "64,192,1000,1,0,0:0:0:0:",
                    "64,192,2000,1,0,0:0:0:0:",
                ],
            )

            evidence = summarize_beatmap_timing_evidence(osu_path)

        self.assertEqual(evidence.redlines.evidence_class, REDLINE_EVIDENCE_STABLE)
        self.assertEqual(evidence.redlines.redline_count, 1)
        self.assertEqual(evidence.redlines.bpms, (120.0,))

    def test_tap_starts_and_hold_ends_are_reported_separately_with_hold_weight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            osu_path = Path(tmp_dir) / "holds.osu"
            _write_osu(
                osu_path,
                timing_lines=["0,500,4,2,0,80,1,0"],
                hitobject_lines=[
                    "64,192,500,1,0,0:0:0:0:",
                    "192,192,1000,128,0,1750:0:0:0:0:",
                ],
            )

            evidence = summarize_beatmap_timing_evidence(
                osu_path,
                object_grid_config=ObjectGridEvidenceConfig(
                    subdivisions=(1, 2, 4),
                    alias_multipliers=(1.0,),
                    hold_end_weight=0.25,
                    inlier_threshold_ms=1.0,
                ),
            ).object_grid

        self.assertEqual(evidence.start_event_count, 2)
        self.assertEqual(evidence.hold_end_event_count, 1)
        self.assertAlmostEqual(evidence.best_evidence.start_stats.total_weight, 2.0)
        self.assertAlmostEqual(evidence.best_evidence.hold_end_stats.total_weight, 0.25)
        self.assertAlmostEqual(evidence.best_evidence.combined_stats.total_weight, 2.25)
        self.assertEqual(evidence.best_evidence.start_stats.event_count, 2)
        self.assertEqual(evidence.best_evidence.hold_end_stats.event_count, 1)
        self.assertEqual(evidence.best_evidence.start_stats.inlier_count, 2)
        self.assertEqual(evidence.best_evidence.hold_end_stats.inlier_count, 1)

        kinds = {OBJECT_EVENT_START, OBJECT_EVENT_HOLD_END}
        self.assertEqual(kinds, {"start", "hold_end"})

    def test_off_grid_object_start_lowers_inlier_rate_and_keeps_ambiguous_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            osu_path = Path(tmp_dir) / "off_grid.osu"
            _write_osu(
                osu_path,
                timing_lines=["0,500,4,2,0,80,1,0"],
                hitobject_lines=[
                    "64,192,0,1,0,0:0:0:0:",
                    "192,192,111,1,0,0:0:0:0:",
                ],
            )

            evidence = summarize_beatmap_timing_evidence(
                osu_path,
                object_grid_config=ObjectGridEvidenceConfig(
                    subdivisions=(4,),
                    alias_multipliers=(1.0,),
                    inlier_threshold_ms=3.0,
                    support_min_weighted_inlier_rate=0.8,
                ),
            ).object_grid

        self.assertEqual(evidence.best_subdivision, 4)
        self.assertEqual(evidence.best_evidence.combined_stats.inlier_count, 1)
        self.assertAlmostEqual(evidence.best_evidence.combined_stats.weighted_inlier_rate, 0.5)
        self.assertFalse(evidence.supported)
        self.assertTrue(evidence.ambiguous)
        self.assertIn("weak_or_missing_object_grid_support", evidence.reasons)

    def test_one_tap_does_not_satisfy_default_object_grid_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            osu_path = Path(tmp_dir) / "one_tap.osu"
            _write_osu(
                osu_path,
                timing_lines=["0,500,4,2,0,80,1,0"],
                hitobject_lines=["64,192,0,1,0,0:0:0:0:"],
            )

            evidence = summarize_beatmap_timing_evidence(
                osu_path,
                object_grid_config=ObjectGridEvidenceConfig(
                    subdivisions=(1,),
                    alias_multipliers=(1.0,),
                ),
            ).object_grid

        self.assertEqual(evidence.start_event_count, 1)
        self.assertFalse(evidence.grid_supported)
        self.assertTrue(evidence.alias_resolved)
        self.assertFalse(evidence.supported)
        self.assertTrue(evidence.ambiguous)
        self.assertIn("start_events<8", evidence.reasons)

    def test_short_start_span_does_not_satisfy_default_object_grid_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            osu_path = Path(tmp_dir) / "short_span.osu"
            _write_osu(
                osu_path,
                timing_lines=["0,500,4,2,0,80,1,0"],
                hitobject_lines=[
                    "64,192,0,1,0,0:0:0:0:",
                    "64,192,500,1,0,0:0:0:0:",
                    "64,192,1000,1,0,0:0:0:0:",
                    "64,192,1500,1,0,0:0:0:0:",
                    "64,192,2000,1,0,0:0:0:0:",
                    "64,192,2500,1,0,0:0:0:0:",
                    "64,192,3000,1,0,0:0:0:0:",
                    "64,192,3500,1,0,0:0:0:0:",
                ],
            )

            evidence = summarize_beatmap_timing_evidence(
                osu_path,
                object_grid_config=ObjectGridEvidenceConfig(
                    subdivisions=(1,),
                    alias_multipliers=(1.0,),
                ),
            ).object_grid

        self.assertEqual(evidence.start_event_count, 8)
        self.assertEqual(evidence.start_time_span_ms, 3500.0)
        self.assertFalse(evidence.grid_supported)
        self.assertFalse(evidence.supported)
        self.assertIn("start_time_span_ms<4000", evidence.reasons)

    def test_enough_on_grid_starts_satisfy_grid_support_when_alias_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            osu_path = Path(tmp_dir) / "enough_on_grid.osu"
            _write_osu(
                osu_path,
                timing_lines=["0,500,4,2,0,80,1,0"],
                hitobject_lines=[
                    "64,192,0,1,0,0:0:0:0:",
                    "64,192,500,1,0,0:0:0:0:",
                    "64,192,1000,1,0,0:0:0:0:",
                    "64,192,1500,1,0,0:0:0:0:",
                    "64,192,2000,1,0,0:0:0:0:",
                    "64,192,2500,1,0,0:0:0:0:",
                    "64,192,3000,1,0,0:0:0:0:",
                    "64,192,3500,1,0,0:0:0:0:",
                    "64,192,4000,1,0,0:0:0:0:",
                ],
            )

            evidence = summarize_beatmap_timing_evidence(
                osu_path,
                object_grid_config=ObjectGridEvidenceConfig(
                    subdivisions=(1,),
                    alias_multipliers=(1.0,),
                ),
            ).object_grid

        self.assertEqual(evidence.start_event_count, 9)
        self.assertEqual(evidence.start_time_span_ms, 4000.0)
        self.assertTrue(evidence.grid_supported)
        self.assertTrue(evidence.alias_resolved)
        self.assertTrue(evidence.supported)
        self.assertFalse(evidence.ambiguous)
        self.assertIn("grid_supported_by_starts", evidence.reasons)
        self.assertIn("alias_resolved", evidence.reasons)

    def test_alias_ambiguity_prevents_supported_even_when_grid_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            osu_path = Path(tmp_dir) / "alias_ambiguous.osu"
            _write_osu(
                osu_path,
                timing_lines=["0,500,4,2,0,80,1,0"],
                hitobject_lines=[
                    "64,192,0,1,0,0:0:0:0:",
                    "64,192,500,1,0,0:0:0:0:",
                    "64,192,1000,1,0,0:0:0:0:",
                    "64,192,1500,1,0,0:0:0:0:",
                    "64,192,2000,1,0,0:0:0:0:",
                    "64,192,2500,1,0,0:0:0:0:",
                    "64,192,3000,1,0,0:0:0:0:",
                    "64,192,3500,1,0,0:0:0:0:",
                    "64,192,4000,1,0,0:0:0:0:",
                ],
            )

            evidence = summarize_beatmap_timing_evidence(
                osu_path,
                object_grid_config=ObjectGridEvidenceConfig(
                    subdivisions=(1,),
                    alias_multipliers=(1.0, 2.0),
                ),
            ).object_grid

        self.assertTrue(evidence.grid_supported)
        self.assertFalse(evidence.alias_resolved)
        self.assertFalse(evidence.supported)
        self.assertTrue(evidence.ambiguous)
        self.assertEqual(evidence.best_alias_multiplier, 1.0)
        self.assertIn("alias_ambiguous", evidence.reasons)

    def test_vectorized_object_grid_matches_reference_on_multisegment_alias_subdivision_case(self) -> None:
        hitobjects = [
            ManiaHitObject(0.0, 250.0, 0, ManiaHitObjectKind.HOLD),
            ManiaHitObject(500.0, 500.0, 1, ManiaHitObjectKind.TAP),
            ManiaHitObject(875.0, 1125.0, 2, ManiaHitObjectKind.HOLD),
            ManiaHitObject(1500.0, 1500.0, 3, ManiaHitObjectKind.TAP),
            ManiaHitObject(2100.0, 2300.0, 0, ManiaHitObjectKind.HOLD),
            ManiaHitObject(2500.0, 2500.0, 1, ManiaHitObjectKind.TAP),
            ManiaHitObject(3100.0, 3300.0, 2, ManiaHitObjectKind.HOLD),
            ManiaHitObject(3700.0, 3700.0, 3, ManiaHitObjectKind.TAP),
            ManiaHitObject(4300.0, 4500.0, 0, ManiaHitObjectKind.HOLD),
        ]
        grid = [
            RedTimingPoint(offset_ms=0.0, beat_length_ms=500.0),
            RedTimingPoint(offset_ms=2000.0, beat_length_ms=400.0),
            RedTimingPoint(offset_ms=3600.0, beat_length_ms=600.0),
        ]
        config = ObjectGridEvidenceConfig(
            subdivisions=(1, 2, 3, 4, 8),
            alias_multipliers=(0.5, 1.0, 2.0),
            hold_end_weight=0.25,
            inlier_threshold_ms=7.5,
            min_start_time_span_ms_for_support=0.0,
        )

        vectorized = summarize_object_grid_evidence(hitobjects, grid, config=config)
        reference = _summarize_object_grid_evidence_reference(hitobjects, grid, config=config)

        _assert_object_grid_evidence_close(self, vectorized, reference)

    def test_large_object_grid_workload_exercises_vectorized_path_without_timing_assertion(self) -> None:
        hitobjects = [
            ManiaHitObject(float(index * 125), float(index * 125 + (250 if index % 5 == 0 else 0)), index % 4,
                           ManiaHitObjectKind.HOLD if index % 5 == 0 else ManiaHitObjectKind.TAP)
            for index in range(2000)
        ]
        grid = [
            RedTimingPoint(offset_ms=0.0, beat_length_ms=500.0),
            RedTimingPoint(offset_ms=60000.0, beat_length_ms=480.0),
            RedTimingPoint(offset_ms=120000.0, beat_length_ms=400.0),
        ]
        config = ObjectGridEvidenceConfig()

        # This is a scale coverage check rather than a wall-clock benchmark:
        # performance is validated by the full label run, while this test keeps
        # CI deterministic and non-fragile.
        evidence = summarize_object_grid_evidence(hitobjects, grid, config=config)

        self.assertEqual(evidence.start_event_count, 2000)
        self.assertEqual(evidence.hold_end_event_count, 400)
        self.assertEqual(
            len(evidence.subdivision_evidence),
            len(evidence.subdivisions) * len(evidence.alias_multipliers),
        )
        self.assertGreaterEqual(evidence.start_time_span_ms, 4000.0)
        self.assertIsNotNone(evidence.best_evidence)

def _assert_object_grid_evidence_close(
    test_case: unittest.TestCase,
    actual: ObjectGridEvidence,
    expected: ObjectGridEvidence,
) -> None:
    test_case.assertEqual(actual.start_event_count, expected.start_event_count)
    test_case.assertEqual(actual.hold_end_event_count, expected.hold_end_event_count)
    test_case.assertEqual(actual.total_event_count, expected.total_event_count)
    test_case.assertAlmostEqual(actual.start_time_span_ms, expected.start_time_span_ms)
    test_case.assertEqual(actual.subdivisions, expected.subdivisions)
    test_case.assertEqual(actual.alias_multipliers, expected.alias_multipliers)
    test_case.assertEqual(actual.best_alias_multiplier, expected.best_alias_multiplier)
    test_case.assertEqual(actual.best_subdivision, expected.best_subdivision)
    test_case.assertEqual(actual.grid_supported, expected.grid_supported)
    test_case.assertEqual(actual.alias_resolved, expected.alias_resolved)
    test_case.assertEqual(actual.supported, expected.supported)
    test_case.assertEqual(actual.ambiguous, expected.ambiguous)
    test_case.assertEqual(actual.reasons, expected.reasons)
    test_case.assertEqual(len(actual.subdivision_evidence), len(expected.subdivision_evidence))
    for actual_subdivision, expected_subdivision in zip(actual.subdivision_evidence, expected.subdivision_evidence):
        test_case.assertEqual(actual_subdivision.alias_multiplier, expected_subdivision.alias_multiplier)
        test_case.assertEqual(actual_subdivision.subdivision, expected_subdivision.subdivision)
        _assert_residual_stats_close(test_case, actual_subdivision.start_stats, expected_subdivision.start_stats)
        _assert_residual_stats_close(test_case, actual_subdivision.hold_end_stats, expected_subdivision.hold_end_stats)
        _assert_residual_stats_close(test_case, actual_subdivision.combined_stats, expected_subdivision.combined_stats)


def _assert_residual_stats_close(
    test_case: unittest.TestCase,
    actual: ObjectResidualStats,
    expected: ObjectResidualStats,
) -> None:
    test_case.assertEqual(actual.event_count, expected.event_count)
    test_case.assertAlmostEqual(actual.total_weight, expected.total_weight)
    test_case.assertEqual(actual.inlier_count, expected.inlier_count)
    test_case.assertAlmostEqual(actual.inlier_weight, expected.inlier_weight)
    test_case.assertAlmostEqual(actual.inlier_rate, expected.inlier_rate)
    test_case.assertAlmostEqual(actual.weighted_inlier_rate, expected.weighted_inlier_rate)
    test_case.assertAlmostEqual(actual.mean_abs_residual_beats, expected.mean_abs_residual_beats)
    test_case.assertAlmostEqual(actual.p50_abs_residual_beats, expected.p50_abs_residual_beats)
    test_case.assertAlmostEqual(actual.p90_abs_residual_beats, expected.p90_abs_residual_beats)
    test_case.assertAlmostEqual(actual.max_abs_residual_beats, expected.max_abs_residual_beats)
    test_case.assertAlmostEqual(actual.mean_abs_residual_ms, expected.mean_abs_residual_ms)
    test_case.assertAlmostEqual(actual.p50_abs_residual_ms, expected.p50_abs_residual_ms)
    test_case.assertAlmostEqual(actual.p90_abs_residual_ms, expected.p90_abs_residual_ms)
    test_case.assertAlmostEqual(actual.max_abs_residual_ms, expected.max_abs_residual_ms)


if __name__ == "__main__":
    unittest.main()
