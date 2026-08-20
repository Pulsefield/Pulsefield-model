from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import pytest

import pulsefield_model.timing.evaluation.v3_projection as v3_projection_module
from pulsefield_model.timing.evaluation.v3_projection import (
    BASELINE_RESULT_SCHEMA,
    FALLBACK_V2,
    FAMILY_A,
    FAMILY_B,
    FAMILY_C0,
    FAMILY_C1,
    RESULT_SCHEMA,
    RESUME_SCHEMA,
    SELECTED_CANDIDATE_METHOD,
    SELECTED_FAMILY_C,
    SELECTED_FAMILY_C_METHOD,
    SUMMARY_SCHEMA,
    V2_METHOD,
    main,
    run_timing_v3_projection_evaluation,
)


class TimingV3ProjectionEvaluationTests(unittest.TestCase):
    def test_projects_durable_baseline_rows_and_aggregates_by_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            first_v2 = _segments((0.0, 500.0), (2100.0, 400.0))
            first_b_oracle = _segments((0.0, 525.0), (2100.0, 400.0))
            second_v2 = _segments((0.0, 500.0))
            baseline = _write_jsonl(
                root / "baseline.jsonl",
                [
                    _baseline_row(
                        "audio-a",
                        predicted_segments=first_v2,
                        oracle_segment_groups=[first_b_oracle],
                        pilot_stratum="jump",
                    ),
                    _baseline_row(
                        "audio-b",
                        predicted_segments=second_v2,
                        oracle_segment_groups=[second_v2, second_v2],
                        pilot_stratum="stable",
                    ),
                ],
            )

            summary = run_timing_v3_projection_evaluation(
                baseline_jsonl_path=baseline,
                output_jsonl_path=root / "projection.jsonl",
                summary_json_path=root / "summary.json",
                progress_every=0,
            )

            results = _read_jsonl(root / "projection.jsonl")
            self.assertEqual(len(results), 2)
            first = results[0]
            self.assertEqual(first["schema"], RESULT_SCHEMA)
            self.assertEqual(first["resume"]["schema"], RESUME_SCHEMA)
            self.assertTrue(first["ok"])
            self.assertIn("evaluation_seconds", first["runtime"])
            self.assertEqual(first[V2_METHOD]["paired_metrics"]["comparison_count"], 1)
            self.assertEqual(first[FAMILY_A]["paired_metrics"]["comparison_count"], 1)
            self.assertEqual(first[FAMILY_B]["paired_metrics"]["comparison_count"], 1)
            self.assertTrue(first[FAMILY_B]["ok"])
            self.assertIsNone(first[FAMILY_B]["fallback"])
            self.assertNotIn(FAMILY_C0, first)
            self.assertNotIn(FAMILY_C1, first)
            self.assertNotIn(SELECTED_FAMILY_C, first)
            for family in (FAMILY_A, FAMILY_B):
                self.assertEqual(
                    first[family]["source_comparison"]["schema"],
                    "pulsefield_model.timing_v3_source_projection_comparison_v1",
                )
            self.assertEqual(first[FAMILY_B]["section_count"], first[V2_METHOD]["section_count"])
            self.assertGreater(first[V2_METHOD]["boundary_seam_ms"]["max_ms"], 90.0)
            self.assertLess(first[FAMILY_B]["boundary_seam_ms"]["max_ms"], 1e-6)
            self.assertLess(first[FAMILY_B]["serialization"]["max_boundary_delta_ms"], 1e-6)
            self.assertAlmostEqual(
                first[FAMILY_B]["bpm_adjustment"]["max_abs_relative"],
                100.0 / 2100.0,
                delta=1e-12,
            )

            self.assertEqual(summary["results"]["successful_audio_count"], 2)
            self.assertEqual(summary["schema"], SUMMARY_SCHEMA)
            self.assertFalse(summary["projection"]["include_family_c"])
            self.assertNotIn(FAMILY_C0, summary["projection"]["families"])
            self.assertEqual(summary["results"]["projection_evaluable_audio_count"], 2)
            self.assertEqual(summary["results"]["comparison_eligible_audio_count"], 2)
            self.assertEqual(summary["results"]["comparator_unavailable_audio_count"], 0)
            self.assertEqual(summary["results"]["paired_comparison_count"], 3)
            self.assertEqual(summary["metrics"][V2_METHOD]["mean_phase_error_ms"]["count"], 2)
            self.assertEqual(summary["difficulty_comparison_metrics"][V2_METHOD]["mean_phase_error_ms"]["count"], 3)
            self.assertEqual(summary["projection_metrics"][FAMILY_B]["projection_failure_count"], 0)
            self.assertEqual(summary["headline"]["family_b_projection_failure_rate"], 0.0)
            self.assertLess(summary["headline"]["family_b_projected_max_boundary_seam_ms"], 1e-6)
            self.assertGreater(summary["headline"]["v2_original_max_boundary_seam_ms"], 90.0)
            self.assertLess(summary["headline"]["family_b_over_v2_mean_phase_error_ms_mean_ratio"], 1.0)
            self.assertIn("started_at_unix", summary["run"])
            self.assertIn("finished_at_unix", summary["run"])
            self.assertGreaterEqual(summary["run"]["total_seconds"], 0.0)
            self.assertEqual(
                set(summary["stratified"]),
                {"pilot_stratum", "label_stratum", "label_confidence", "label_ambiguous", "source_long_track"},
            )
            self.assertEqual(_stratum_entry(summary, "jump")["audio_count"], 1)
            self.assertEqual(_stratum_entry(summary, "stable")["audio_count"], 1)
            self.assertEqual(_stratum_entry(summary, "jump", dimension="label_stratum")["audio_count"], 1)
            self.assertEqual(_stratum_entry(summary, "synthetic", dimension="label_confidence")["audio_count"], 2)
            self.assertEqual(_stratum_entry(summary, False, dimension="label_ambiguous")["audio_count"], 2)
            self.assertEqual(_stratum_entry(summary, False, dimension="source_long_track")["audio_count"], 2)

    def test_family_b_failure_emits_tagged_v2_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            distorted_v2 = _segments((0.0, 500.0), (526.5, 400.0))
            baseline = _write_jsonl(
                root / "baseline.jsonl",
                [
                    _baseline_row(
                        "audio-fallback",
                        predicted_segments=distorted_v2,
                        oracle_segment_groups=[distorted_v2],
                        pilot_stratum="jump",
                    )
                ],
            )

            summary = run_timing_v3_projection_evaluation(
                baseline_jsonl_path=baseline,
                output_jsonl_path=root / "projection.jsonl",
                summary_json_path=root / "summary.json",
                progress_every=0,
            )

            result = _read_jsonl(root / "projection.jsonl")[0]
            self.assertTrue(result["ok"])
            self.assertTrue(result[V2_METHOD]["ok"])
            self.assertFalse(result[FAMILY_B]["ok"])
            self.assertEqual(result[FAMILY_B]["fallback"]["tag"], FALLBACK_V2)
            self.assertEqual(result[FAMILY_B]["grid"], None)
            self.assertEqual(result[FAMILY_B]["comparisons"], [])
            self.assertEqual(result[FAMILY_B]["paired_metrics"]["comparison_count"], 0)
            self.assertEqual(summary["results"]["family_b_projection_failure_audio_count"], 1)
            self.assertEqual(summary["results"]["family_b_fallback_audio_count"], 1)
            self.assertEqual(summary["headline"]["family_b_projection_failure_rate"], 1.0)

    def test_compare_stage_baseline_failure_with_fit_is_projection_evaluable_without_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            row = _baseline_row(
                "compare-unavailable",
                predicted_segments=_segments((0.0, 500.0), (2000.0, 400.0)),
                oracle_segment_groups=[],
                ok=False,
                failure_stage="compare",
            )
            row["comparisons"] = [
                {
                    "ok": False,
                    "beatmap_path": "missing.osu",
                    "oracle_segments": [],
                    "metrics": None,
                    "drift_metrics": None,
                    "error_type": "MissingRedTimingError",
                    "error": "no red timing point",
                }
            ]
            baseline = _write_jsonl(root / "baseline.jsonl", [row])

            summary = run_timing_v3_projection_evaluation(
                baseline_jsonl_path=baseline,
                output_jsonl_path=root / "projection.jsonl",
                summary_json_path=root / "summary.json",
                progress_every=0,
            )

            result = _read_jsonl(root / "projection.jsonl")[0]
            self.assertTrue(result["ok"])
            self.assertFalse(result["comparator_available"])
            self.assertTrue(result[V2_METHOD]["ok"])
            self.assertTrue(result[FAMILY_B]["ok"])
            self.assertEqual(result[V2_METHOD]["paired_metrics"]["comparison_count"], 0)
            self.assertEqual(result[FAMILY_B]["paired_metrics"]["comparison_count"], 0)
            self.assertEqual(summary["results"]["projection_evaluable_audio_count"], 1)
            self.assertEqual(summary["results"]["comparison_eligible_audio_count"], 0)
            self.assertEqual(summary["results"]["comparator_unavailable_audio_count"], 1)
            self.assertEqual(summary["results"]["baseline_unusable_audio_count"], 0)
            self.assertEqual(summary["metrics"][FAMILY_B]["mean_phase_error_ms"]["count"], 0)
            self.assertEqual(summary["headline"]["family_b_projection_failure_rate"], 0.0)

    def test_compare_stage_baseline_failure_with_b_distortion_counts_fallback_not_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            distorted_v2 = _segments((0.0, 500.0), (526.5, 400.0))
            baseline = _write_jsonl(
                root / "baseline.jsonl",
                [
                    _baseline_row(
                        "compare-fallback",
                        predicted_segments=distorted_v2,
                        oracle_segment_groups=[],
                        ok=False,
                        failure_stage="compare",
                    )
                ],
            )

            summary = run_timing_v3_projection_evaluation(
                baseline_jsonl_path=baseline,
                output_jsonl_path=root / "projection.jsonl",
                summary_json_path=root / "summary.json",
                progress_every=0,
            )

            result = _read_jsonl(root / "projection.jsonl")[0]
            self.assertTrue(result["ok"])
            self.assertFalse(result["comparator_available"])
            self.assertFalse(result[FAMILY_B]["ok"])
            self.assertEqual(result[FAMILY_B]["fallback"]["tag"], FALLBACK_V2)
            self.assertEqual(summary["results"]["projection_evaluable_audio_count"], 1)
            self.assertEqual(summary["results"]["comparison_eligible_audio_count"], 0)
            self.assertEqual(summary["results"]["comparator_unavailable_audio_count"], 1)
            self.assertEqual(summary["results"]["family_b_projection_failure_audio_count"], 1)
            self.assertEqual(summary["results"]["family_b_fallback_audio_count"], 1)
            self.assertEqual(summary["metrics"][FAMILY_B]["mean_phase_error_ms"]["count"], 0)
            self.assertEqual(summary["headline"]["family_b_projection_failure_rate"], 1.0)

    def test_selected_candidate_with_v2_fallback_is_distinct_from_pure_family_b_when_comparator_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            distorted_v2 = _segments((0.0, 500.0), (526.5, 400.0))
            baseline = _write_jsonl(
                root / "baseline.jsonl",
                [
                    _baseline_row(
                        "fallback-with-comparator",
                        predicted_segments=distorted_v2,
                        oracle_segment_groups=[distorted_v2],
                    )
                ],
            )

            summary = run_timing_v3_projection_evaluation(
                baseline_jsonl_path=baseline,
                output_jsonl_path=root / "projection.jsonl",
                summary_json_path=root / "summary.json",
                progress_every=0,
            )

            result = _read_jsonl(root / "projection.jsonl")[0]
            self.assertTrue(result["comparator_available"])
            self.assertFalse(result[FAMILY_B]["ok"])
            self.assertEqual(result[FAMILY_B]["paired_metrics"]["comparison_count"], 0)
            self.assertEqual(result["selected_candidate"]["tag"], FALLBACK_V2)
            self.assertEqual(result["selected_candidate"]["method"], "selected_family_b_or_fallback_v2")
            self.assertTrue(result["selected_candidate"]["used_v2_fallback"])
            self.assertEqual(result["selected_candidate"]["paired_metrics"]["comparison_count"], 1)
            self.assertEqual(summary["difficulty_comparison_metrics"][FAMILY_B]["mean_phase_error_ms"]["count"], 0)
            self.assertEqual(
                summary["difficulty_comparison_metrics"]["selected_family_b_or_fallback_v2"]["mean_phase_error_ms"][
                    "count"
                ],
                1,
            )

    def test_opt_in_family_c_reports_c0_c1_selected_source_and_joint_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = _segments((0.0, 500.0), (2100.0, 400.0))
            baseline = _write_jsonl(
                root / "baseline.jsonl",
                [
                    _baseline_row(
                        "family-c-success",
                        predicted_segments=source,
                        oracle_segment_groups=[source],
                        pilot_stratum="jump",
                    )
                ],
            )

            summary = run_timing_v3_projection_evaluation(
                baseline_jsonl_path=baseline,
                output_jsonl_path=root / "projection.jsonl",
                summary_json_path=root / "summary.json",
                progress_every=0,
                projection_config={"include_family_c": True},
            )

            result = _read_jsonl(root / "projection.jsonl")[0]
            self.assertTrue(summary["projection"]["include_family_c"])
            self.assertEqual(
                summary["projection"]["families"][FAMILY_C0],
                "joint_phase_fixed_counts",
            )
            self.assertEqual(
                summary["projection"]["families"][FAMILY_C1],
                "joint_phase_nearby_counts",
            )
            self.assertEqual(
                result["resume"]["components"]["projection"]["fingerprint"],
                summary["projection"]["fingerprint"],
            )
            self.assertTrue(result[FAMILY_C0]["ok"])
            self.assertTrue(result[FAMILY_C1]["ok"])
            self.assertEqual(result[FAMILY_C0]["fallback"], None)
            self.assertEqual(result[FAMILY_C1]["fallback"], None)
            self.assertEqual(result[SELECTED_FAMILY_C]["method"], SELECTED_FAMILY_C_METHOD)
            self.assertEqual(result[SELECTED_FAMILY_C]["tag"], FAMILY_C1)
            self.assertFalse(result[SELECTED_FAMILY_C]["used_v2_fallback"])
            for family in (FAMILY_A, FAMILY_B, FAMILY_C0, FAMILY_C1):
                source_metrics = result[family]["source_comparison"]
                self.assertEqual(
                    source_metrics["schema"],
                    "pulsefield_model.timing_v3_source_projection_comparison_v1",
                )
                self.assertGreaterEqual(source_metrics["wrapped_phase_rms_beats"], 0.0)
                self.assertGreaterEqual(source_metrics["abs_endpoint_relative_drift_ms"], 0.0)

            c1_projection = summary["projection_metrics"][FAMILY_C1]
            self.assertEqual(c1_projection["projection_success_count"], 1)
            self.assertEqual(c1_projection["source_comparison"]["audio_count"], 1)
            self.assertEqual(
                c1_projection["source_comparison"]["wrapped_phase_rms_beats"]["count"],
                1,
            )
            joint = c1_projection["joint_diagnostics"]
            self.assertTrue(joint["all_successful_solver_residuals_pass"])
            self.assertTrue(joint["all_successful_searches_converged"])
            self.assertTrue(joint["all_successful_anchor_displacements_pass"])
            self.assertTrue(joint["all_successful_fingerprints_present"])
            self.assertEqual(joint["fingerprints"]["complete_triplet_count"], 1)
            self.assertEqual(joint["changed_count_rate"]["count"], 1)
            self.assertEqual(joint["abs_original_residual_ms"]["count"], 1)
            self.assertEqual(c1_projection["projection_seconds"]["count"], 1)

            matched = summary["metrics"]["matched_vs_v2"][FAMILY_C1]
            self.assertEqual(matched["mean_phase_error_ms"]["paired_audio_count"], 1)
            self.assertEqual(
                matched["alias_abs_endpoint_relative_drift_ms"]["paired_audio_count"],
                1,
            )
            c1_headline = summary["headline"]["candidates"][FAMILY_C1]
            self.assertEqual(c1_headline["projection_evaluable_audio_count"], 1)
            self.assertEqual(c1_headline["comparison_eligible_audio_count"], 1)
            self.assertEqual(c1_headline["phase_paired_audio_count"], 1)
            self.assertIn("guard_audit", c1_headline)
            self.assertIn(FAMILY_C1, _stratum_entry(summary, "jump")["headline"]["candidates"])

    def test_family_c_failure_emits_tagged_fallback_and_selected_copies_v2(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            infeasible = _segments((0.0, 500.0), (2300.0, 400.0))
            baseline = _write_jsonl(
                root / "baseline.jsonl",
                [
                    _baseline_row(
                        "family-c-fallback",
                        predicted_segments=infeasible,
                        oracle_segment_groups=[infeasible],
                    )
                ],
            )

            summary = run_timing_v3_projection_evaluation(
                baseline_jsonl_path=baseline,
                output_jsonl_path=root / "projection.jsonl",
                summary_json_path=root / "summary.json",
                progress_every=0,
                projection_config={"include_family_c": True},
            )

            result = _read_jsonl(root / "projection.jsonl")[0]
            for family in (FAMILY_C0, FAMILY_C1):
                self.assertFalse(result[family]["ok"])
                self.assertEqual(result[family]["fallback"]["tag"], FALLBACK_V2)
                self.assertEqual(result[family]["comparisons"], [])
                self.assertIsNone(result[family]["source_comparison"])
            selected = result[SELECTED_FAMILY_C]
            self.assertEqual(selected["tag"], FALLBACK_V2)
            self.assertTrue(selected["used_v2_fallback"])
            self.assertEqual(selected["source_method"], V2_METHOD)
            self.assertEqual(selected["paired_metrics"]["comparison_count"], 1)
            self.assertEqual(summary["results"]["family_c1_projection_failure_audio_count"], 1)
            self.assertEqual(summary["results"]["family_c1_fallback_audio_count"], 1)
            self.assertEqual(summary["headline"]["family_c1_fallback_rate"], 1.0)
            self.assertEqual(
                summary["difficulty_comparison_metrics"][SELECTED_FAMILY_C_METHOD][
                    "mean_phase_error_ms"
                ]["count"],
                1,
            )

    def test_family_c_projection_and_comparator_denominators_remain_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            feasible = _segments((0.0, 500.0), (2100.0, 400.0))
            infeasible = _segments((0.0, 500.0), (2300.0, 400.0))
            baseline = _write_jsonl(
                root / "baseline.jsonl",
                [
                    _baseline_row(
                        "comparable-one",
                        predicted_segments=feasible,
                        oracle_segment_groups=[feasible],
                    ),
                    _baseline_row(
                        "comparable-two",
                        predicted_segments=feasible,
                        oracle_segment_groups=[feasible],
                    ),
                    _baseline_row(
                        "projection-only-fallback",
                        predicted_segments=infeasible,
                        oracle_segment_groups=[],
                        ok=False,
                        failure_stage="compare",
                    ),
                ],
            )

            summary = run_timing_v3_projection_evaluation(
                baseline_jsonl_path=baseline,
                output_jsonl_path=root / "projection.jsonl",
                summary_json_path=root / "summary.json",
                progress_every=0,
                projection_config={"include_family_c": True},
            )

            self.assertEqual(summary["results"]["projection_evaluable_audio_count"], 3)
            self.assertEqual(summary["results"]["comparison_eligible_audio_count"], 2)
            self.assertEqual(summary["results"]["comparator_unavailable_audio_count"], 1)
            c1 = summary["headline"]["candidates"][FAMILY_C1]
            self.assertEqual(c1["projection_evaluable_audio_count"], 3)
            self.assertEqual(c1["comparison_eligible_audio_count"], 2)
            self.assertEqual(c1["phase_paired_audio_count"], 2)
            self.assertEqual(c1["projection_failure_audio_count"], 1)
            self.assertEqual(c1["fallback_audio_count"], 1)
            self.assertEqual(c1["fallback_rate"], pytest.approx(1.0 / 3.0))
            self.assertEqual(
                summary["metrics"]["matched_vs_v2"][FAMILY_C1]["mean_phase_error_ms"][
                    "paired_audio_count"
                ],
                2,
            )

    def test_family_c_config_flag_invalidates_default_ab_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = _segments((0.0, 500.0), (2100.0, 400.0))
            baseline = _write_jsonl(
                root / "baseline.jsonl",
                [
                    _baseline_row(
                        "resume-family-c",
                        predicted_segments=source,
                        oracle_segment_groups=[source],
                    )
                ],
            )
            output = root / "projection.jsonl"

            first = run_timing_v3_projection_evaluation(
                baseline_jsonl_path=baseline,
                output_jsonl_path=output,
                summary_json_path=root / "summary.json",
                progress_every=0,
            )
            first_result = _read_jsonl(output)[0]
            second = run_timing_v3_projection_evaluation(
                baseline_jsonl_path=baseline,
                output_jsonl_path=output,
                summary_json_path=root / "summary.json",
                progress_every=0,
                projection_config={"include_family_c": True},
            )
            second_result = _read_jsonl(output)[0]
            third = run_timing_v3_projection_evaluation(
                baseline_jsonl_path=baseline,
                output_jsonl_path=output,
                summary_json_path=root / "summary.json",
                progress_every=0,
                projection_config={"include_family_c": True},
            )

            self.assertEqual(first["run"]["processed_count"], 1)
            self.assertNotIn(FAMILY_C0, first_result)
            self.assertEqual(second["run"]["stale_existing_count"], 1)
            self.assertEqual(second["run"]["processed_count"], 1)
            self.assertIn(FAMILY_C0, second_result)
            self.assertNotEqual(
                first_result["resume"]["fingerprint"],
                second_result["resume"]["fingerprint"],
            )
            self.assertEqual(third["run"]["processed_count"], 0)
            self.assertEqual(third["run"]["skipped_success_count"], 1)

    def test_resume_reuses_matching_output_and_invalidates_changed_baseline_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            baseline = root / "baseline.jsonl"
            row = _baseline_row(
                "audio-a",
                predicted_segments=_segments((0.0, 500.0), (2100.0, 400.0)),
                oracle_segment_groups=[_segments((0.0, 525.0), (2100.0, 400.0))],
            )
            _write_jsonl(baseline, [row])

            first = run_timing_v3_projection_evaluation(
                baseline_jsonl_path=baseline,
                output_jsonl_path=root / "projection.jsonl",
                summary_json_path=root / "summary.json",
                progress_every=0,
                checkpoint_every=1,
            )
            first_jsonl = (root / "projection.jsonl").read_text(encoding="utf-8")
            second = run_timing_v3_projection_evaluation(
                baseline_jsonl_path=baseline,
                output_jsonl_path=root / "projection.jsonl",
                summary_json_path=root / "summary.json",
                progress_every=0,
                checkpoint_every=1,
            )
            second_jsonl = (root / "projection.jsonl").read_text(encoding="utf-8")
            row["prediction"]["frame_count"] = 251
            _write_jsonl(baseline, [row])
            third = run_timing_v3_projection_evaluation(
                baseline_jsonl_path=baseline,
                output_jsonl_path=root / "projection.jsonl",
                summary_json_path=root / "summary.json",
                progress_every=0,
                checkpoint_every=1,
            )
            changed = _read_jsonl(root / "projection.jsonl")[0]

            self.assertEqual(first["run"]["processed_count"], 1)
            self.assertEqual(second["run"]["processed_count"], 0)
            self.assertEqual(second["run"]["skipped_success_count"], 1)
            self.assertEqual(first_jsonl, second_jsonl)
            self.assertEqual(third["run"]["stale_existing_count"], 1)
            self.assertEqual(third["run"]["processed_count"], 1)
            self.assertEqual(changed["prediction"]["frame_count"], 251)
            self.assertNotEqual(
                json.loads(first_jsonl)["resume"]["fingerprint"],
                changed["resume"]["fingerprint"],
            )

    def test_resume_invalidates_when_behavior_source_hash_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            baseline = _write_jsonl(
                root / "baseline.jsonl",
                [
                    _baseline_row(
                        "audio-a",
                        predicted_segments=_segments((0.0, 500.0)),
                        oracle_segment_groups=[_segments((0.0, 500.0))],
                    )
                ],
            )
            behavior_a = _behavior("a" * 64)
            behavior_b = _behavior("b" * 64)

            with mock.patch.object(v3_projection_module, "_behavior_provenance", return_value=behavior_a):
                first = run_timing_v3_projection_evaluation(
                    baseline_jsonl_path=baseline,
                    output_jsonl_path=root / "projection.jsonl",
                    summary_json_path=root / "summary.json",
                    progress_every=0,
                )
                second = run_timing_v3_projection_evaluation(
                    baseline_jsonl_path=baseline,
                    output_jsonl_path=root / "projection.jsonl",
                    summary_json_path=root / "summary.json",
                    progress_every=0,
                )
            first_result = _read_jsonl(root / "projection.jsonl")[0]

            with mock.patch.object(v3_projection_module, "_behavior_provenance", return_value=behavior_b):
                third = run_timing_v3_projection_evaluation(
                    baseline_jsonl_path=baseline,
                    output_jsonl_path=root / "projection.jsonl",
                    summary_json_path=root / "summary.json",
                    progress_every=0,
                )
            third_result = _read_jsonl(root / "projection.jsonl")[0]

            self.assertEqual(first["run"]["processed_count"], 1)
            self.assertEqual(second["run"]["processed_count"], 0)
            self.assertEqual(second["run"]["skipped_success_count"], 1)
            self.assertEqual(third["run"]["stale_existing_count"], 1)
            self.assertEqual(third["run"]["processed_count"], 1)
            self.assertNotEqual(first_result["resume"]["fingerprint"], third_result["resume"]["fingerprint"])

    def test_behavior_provenance_records_source_hashes_and_runtime_versions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            baseline = _write_jsonl(
                root / "baseline.jsonl",
                [
                    _baseline_row(
                        "audio-a",
                        predicted_segments=_segments((0.0, 500.0)),
                        oracle_segment_groups=[_segments((0.0, 500.0))],
                    )
                ],
            )

            summary = run_timing_v3_projection_evaluation(
                baseline_jsonl_path=baseline,
                output_jsonl_path=root / "projection.jsonl",
                summary_json_path=root / "summary.json",
                progress_every=0,
            )

            modules = {entry["module"]: entry for entry in summary["behavior"]["source_modules"]}
            expected_modules = {
                "pulsefield_model.timing.evaluation.v3_projection",
                "pulsefield_model.timing.v3.projection",
                "pulsefield_model.timing.v3.joint_projection",
                "pulsefield_model.timing.v3.schema",
                "pulsefield_model.timing.evaluation.source_projection",
                "pulsefield_model.timing.diagnostics.compare_to_oracle",
                "pulsefield_model.timing.evaluation.drift",
                "pulsefield_model.timing.canonicalization",
                "pulsefield_model.timing.rendering.dense_timing_v2",
            }
            self.assertEqual(set(modules), expected_modules)
            self.assertTrue(all(len(entry["sha256"]) == 64 for entry in modules.values()))
            self.assertIn("version_info", summary["behavior"]["python"])
            self.assertIn("version", summary["behavior"]["numpy"])
            result = _read_jsonl(root / "projection.jsonl")[0]
            self.assertEqual(result["provenance"]["behavior"]["fingerprint"], summary["behavior"]["fingerprint"])

    def test_invalid_baseline_rows_are_isolated_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            invalid_success = {
                "schema": BASELINE_RESULT_SCHEMA,
                "ok": True,
                "audio_key": "invalid-audio",
                "prediction": {"frame_count": 250, "frame_rate_hz": 50.0},
                "fit": {"predicted_segments": []},
                "comparisons": [],
            }
            baseline_failure = {
                "schema": BASELINE_RESULT_SCHEMA,
                "ok": False,
                "audio_key": "cache-failed",
                "failure_stage": "cache",
                "error_type": "FileNotFoundError",
                "error": "cache missing",
            }
            baseline = root / "baseline.jsonl"
            baseline.write_text(
                json.dumps(invalid_success, sort_keys=True) + "\n"
                + json.dumps(baseline_failure, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            summary = run_timing_v3_projection_evaluation(
                baseline_jsonl_path=baseline,
                output_jsonl_path=root / "projection.jsonl",
                summary_json_path=root / "summary.json",
                progress_every=0,
            )

            results = _read_jsonl(root / "projection.jsonl")
            self.assertEqual([result["ok"] for result in results], [False, False])
            self.assertEqual(results[0]["failure_stage"], "baseline_row")
            self.assertEqual(results[1]["failure_stage"], "cache")
            self.assertEqual(summary["failures"]["stage_counts"], {"baseline_row": 1, "cache": 1})
            self.assertEqual(summary["headline"]["family_b_projection_failure_rate"], None)

    def test_cache_fit_and_timeout_baseline_failures_are_not_projection_evaluable_even_with_stray_fit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            rows = [
                _baseline_row(
                    f"{stage}-failed",
                    predicted_segments=_segments((0.0, 500.0)),
                    oracle_segment_groups=[_segments((0.0, 500.0))],
                    ok=False,
                    failure_stage=stage,
                )
                for stage in ("cache", "fit", "timeout")
            ]
            baseline = _write_jsonl(root / "baseline.jsonl", rows)

            summary = run_timing_v3_projection_evaluation(
                baseline_jsonl_path=baseline,
                output_jsonl_path=root / "projection.jsonl",
                summary_json_path=root / "summary.json",
                progress_every=0,
            )

            results = _read_jsonl(root / "projection.jsonl")
            self.assertEqual([result["ok"] for result in results], [False, False, False])
            self.assertEqual([result["failure_stage"] for result in results], ["cache", "fit", "timeout"])
            self.assertEqual(summary["results"]["projection_evaluable_audio_count"], 0)
            self.assertEqual(summary["results"]["baseline_unusable_audio_count"], 3)

    def test_malformed_stored_oracle_is_recorded_while_valid_sibling_remains_paired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            row = _baseline_row(
                "mixed-oracles",
                predicted_segments=_segments((0.0, 500.0)),
                oracle_segment_groups=[_segments((0.0, 500.0))],
            )
            row["comparisons"].insert(
                0,
                {
                    "ok": True,
                    "beatmap_path": "malformed.osu",
                    "oracle_segments": [{"offset_ms": 0.0, "beat_length_ms": -1.0, "meter": 4}],
                    "metrics": {},
                    "drift_metrics": {},
                    "error_type": None,
                    "error": None,
                },
            )
            baseline = _write_jsonl(root / "baseline.jsonl", [row])

            summary = run_timing_v3_projection_evaluation(
                baseline_jsonl_path=baseline,
                output_jsonl_path=root / "projection.jsonl",
                summary_json_path=root / "summary.json",
                progress_every=0,
            )

            result = _read_jsonl(root / "projection.jsonl")[0]
            self.assertTrue(result["ok"])
            self.assertTrue(result["comparator_available"])
            self.assertEqual(result["stored_oracle_comparison_count"], 2)
            self.assertEqual(result["valid_stored_oracle_comparison_count"], 1)
            self.assertEqual(len(result[V2_METHOD]["comparisons"]), 2)
            self.assertFalse(result[V2_METHOD]["comparisons"][0]["ok"])
            self.assertTrue(result[V2_METHOD]["comparisons"][1]["ok"])
            self.assertEqual(result[V2_METHOD]["paired_metrics"]["comparison_count"], 1)
            self.assertEqual(summary["results"]["comparison_eligible_audio_count"], 1)
            self.assertEqual(summary["results"]["paired_comparison_count"], 1)

    def test_path_collisions_are_rejected_before_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            baseline = _write_jsonl(
                root / "baseline.jsonl",
                [
                    _baseline_row(
                        "audio-a",
                        predicted_segments=_segments((0.0, 500.0)),
                        oracle_segment_groups=[_segments((0.0, 500.0))],
                    )
                ],
            )
            original = baseline.read_text(encoding="utf-8")

            cases = [
                {"output_jsonl_path": baseline, "summary_json_path": root / "summary.json"},
                {"output_jsonl_path": root / "projection.jsonl", "summary_json_path": baseline},
                {"output_jsonl_path": root / "same.json", "summary_json_path": root / "same.json"},
            ]
            for kwargs in cases:
                with self.subTest(kwargs=kwargs):
                    with pytest.raises(ValueError, match="must be different paths"):
                        run_timing_v3_projection_evaluation(
                            baseline_jsonl_path=baseline,
                            progress_every=0,
                            **kwargs,
                        )
                    self.assertEqual(baseline.read_text(encoding="utf-8"), original)

    def test_output_atomic_temp_name_cannot_collide_with_baseline_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output = root / "projection.jsonl"
            baseline = _write_jsonl(
                root / "projection.jsonl.tmp",
                [
                    _baseline_row(
                        "output-temp-collision",
                        predicted_segments=_segments((0.0, 500.0)),
                        oracle_segment_groups=[_segments((0.0, 500.0))],
                    )
                ],
            )
            original = baseline.read_bytes()

            summary = run_timing_v3_projection_evaluation(
                baseline_jsonl_path=baseline,
                output_jsonl_path=output,
                summary_json_path=root / "summary.json",
                progress_every=0,
            )

            self.assertEqual(baseline.read_bytes(), original)
            self.assertEqual(len(_read_jsonl(output)), 1)
            self.assertEqual(summary["results"]["successful_audio_count"], 1)
            self.assertEqual(list(root.glob(".projection.jsonl.*.tmp")), [])

    def test_summary_atomic_temp_name_cannot_collide_with_baseline_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            summary_path = root / "summary.json"
            baseline = _write_jsonl(
                root / "summary.json.tmp",
                [
                    _baseline_row(
                        "summary-temp-collision",
                        predicted_segments=_segments((0.0, 500.0)),
                        oracle_segment_groups=[_segments((0.0, 500.0))],
                    )
                ],
            )
            original = baseline.read_bytes()

            summary = run_timing_v3_projection_evaluation(
                baseline_jsonl_path=baseline,
                output_jsonl_path=root / "projection.jsonl",
                summary_json_path=summary_path,
                progress_every=0,
            )

            self.assertEqual(baseline.read_bytes(), original)
            self.assertEqual(json.loads(summary_path.read_text(encoding="utf-8")), summary)
            self.assertEqual(list(root.glob(".summary.json.*.tmp")), [])

    def test_atomic_temp_is_cleaned_when_serialization_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output = root / "summary.json"

            with (
                mock.patch.object(
                    v3_projection_module.json,
                    "dumps",
                    side_effect=TypeError("synthetic serialization failure"),
                ),
                pytest.raises(TypeError, match="synthetic serialization failure"),
            ):
                v3_projection_module._write_json_atomic(output, {"ok": True})

            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".summary.json.*.tmp")), [])

    def test_headline_family_b_ratios_use_matched_audio_aggregates_not_individual_ratios(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            off_lattice = _segments((0.0, 500.0), (2100.0, 400.0))
            distorted = _segments((0.0, 500.0), (526.5, 400.0))
            baseline = _write_jsonl(
                root / "baseline.jsonl",
                [
                    _baseline_row("ratio-one", predicted_segments=off_lattice, oracle_segment_groups=[off_lattice]),
                    _baseline_row("ratio-two", predicted_segments=off_lattice, oracle_segment_groups=[off_lattice]),
                    _baseline_row("ratio-fallback", predicted_segments=distorted, oracle_segment_groups=[distorted]),
                ],
            )

            with mock.patch.object(v3_projection_module, "_compare_grid_to_oracles", side_effect=_fake_ratio_compare):
                summary = run_timing_v3_projection_evaluation(
                    baseline_jsonl_path=baseline,
                    output_jsonl_path=root / "projection.jsonl",
                    summary_json_path=root / "summary.json",
                    progress_every=0,
                )

            matched = summary["headline"]["family_b_vs_v2_mean_phase_error_ms_matched"]
            self.assertEqual(matched["paired_audio_count"], 2)
            self.assertEqual(matched[V2_METHOD]["count"], 2)
            self.assertEqual(matched[FAMILY_B]["count"], 2)
            self.assertEqual(matched[V2_METHOD]["mean"], pytest.approx(20.0))
            self.assertEqual(matched[FAMILY_B]["mean"], pytest.approx(40.0))
            self.assertEqual(summary["headline"]["family_b_over_v2_mean_phase_error_ms_mean_ratio"], pytest.approx(2.0))
            self.assertEqual(summary["headline"]["family_b_over_v2_mean_phase_error_ms_p90_ratio"], pytest.approx(2.0))
            self.assertEqual(summary["results"]["family_b_fallback_audio_count"], 1)

    def test_cli_returns_zero_for_b_fallbacks_and_nonzero_for_evaluator_row_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            fallback_baseline = _write_jsonl(
                root / "fallback-baseline.jsonl",
                [
                    _baseline_row(
                        "fallback",
                        predicted_segments=_segments((0.0, 500.0), (526.5, 400.0)),
                        oracle_segment_groups=[_segments((0.0, 500.0), (526.5, 400.0))],
                    )
                ],
            )
            failed_baseline = _write_jsonl(
                root / "failed-baseline.jsonl",
                [
                    {
                        "schema": BASELINE_RESULT_SCHEMA,
                        "ok": False,
                        "audio_key": "cache-failed",
                        "failure_stage": "cache",
                        "error_type": "FileNotFoundError",
                        "error": "cache missing",
                    }
                ],
            )

            fallback_status = main(
                [
                    "--baseline-jsonl",
                    fallback_baseline.as_posix(),
                    "--output-jsonl",
                    (root / "fallback.jsonl").as_posix(),
                    "--summary-json",
                    (root / "fallback-summary.json").as_posix(),
                    "--progress-every",
                    "0",
                    "--json",
                ]
            )
            failed_status = main(
                [
                    "--baseline-jsonl",
                    failed_baseline.as_posix(),
                    "--output-jsonl",
                    (root / "failed.jsonl").as_posix(),
                    "--summary-json",
                    (root / "failed-summary.json").as_posix(),
                    "--progress-every",
                    "0",
                    "--json",
                ]
            )

            self.assertEqual(fallback_status, 0)
            self.assertEqual(failed_status, 1)

    def test_cli_family_c_flag_enables_c0_c1_without_switching_fitter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = _segments((0.0, 500.0), (2100.0, 400.0))
            baseline = _write_jsonl(
                root / "baseline.jsonl",
                [
                    _baseline_row(
                        "cli-family-c",
                        predicted_segments=source,
                        oracle_segment_groups=[source],
                    )
                ],
            )

            with redirect_stdout(io.StringIO()):
                status = main(
                    [
                        "--baseline-jsonl",
                        baseline.as_posix(),
                        "--output-jsonl",
                        (root / "projection.jsonl").as_posix(),
                        "--summary-json",
                        (root / "summary.json").as_posix(),
                        "--include-family-c",
                        "--progress-every",
                        "0",
                    ]
                )

            result = _read_jsonl(root / "projection.jsonl")[0]
            self.assertEqual(status, 0)
            self.assertIn(FAMILY_C0, result)
            self.assertIn(FAMILY_C1, result)
            self.assertIn(SELECTED_FAMILY_C, result)
            self.assertTrue(result[FAMILY_C1]["ok"])

    def test_cli_json_stdout_stays_parseable_when_pending_row_prints_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            baseline = _write_jsonl(
                root / "baseline.jsonl",
                [
                    _baseline_row(
                        "json-progress",
                        predicted_segments=_segments((0.0, 500.0)),
                        oracle_segment_groups=[_segments((0.0, 500.0))],
                    )
                ],
            )
            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = main(
                    [
                        "--baseline-jsonl",
                        baseline.as_posix(),
                        "--output-jsonl",
                        (root / "projection.jsonl").as_posix(),
                        "--summary-json",
                        (root / "summary.json").as_posix(),
                        "--progress-every",
                        "1",
                        "--json",
                    ]
                )

            decoded = json.loads(stdout.getvalue())
            self.assertEqual(status, 0)
            self.assertEqual(decoded["run"]["processed_count"], 1)
            self.assertNotIn("[timing-v3-projection]", stdout.getvalue())
            self.assertIn("[timing-v3-projection] processed=1/1", stderr.getvalue())

    def test_cli_summary_wording_distinguishes_projection_evaluable_from_comparison_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            baseline = _write_jsonl(
                root / "baseline.jsonl",
                [
                    _baseline_row(
                        "compare-unavailable",
                        predicted_segments=_segments((0.0, 500.0)),
                        oracle_segment_groups=[],
                        ok=False,
                        failure_stage="compare",
                    )
                ],
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                status = main(
                    [
                        "--baseline-jsonl",
                        baseline.as_posix(),
                        "--output-jsonl",
                        (root / "projection.jsonl").as_posix(),
                        "--summary-json",
                        (root / "summary.json").as_posix(),
                        "--progress-every",
                        "0",
                    ]
                )

            self.assertEqual(status, 0)
            text = stdout.getvalue()
            self.assertIn("1/1 audio projection-evaluable", text)
            self.assertIn("0 comparison-eligible", text)
            self.assertNotIn("audio comparable", text)

    def test_runner_does_not_call_cache_fitter_or_oracle_loader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            baseline = _write_jsonl(
                root / "baseline.jsonl",
                [
                    _baseline_row(
                        "audio-a",
                        predicted_segments=_segments((0.0, 500.0)),
                        oracle_segment_groups=[_segments((0.0, 500.0))],
                    )
                ],
            )

            with (
                mock.patch(
                    "pulsefield_model.timing.providers.beatthis_cache.load_beatthis_frame_prediction_cache",
                    side_effect=AssertionError("cache should not be read"),
                ),
                mock.patch(
                    "pulsefield_model.timing.grid_fitting.GridFitter",
                    side_effect=AssertionError("fitter should not be constructed"),
                ),
                mock.patch(
                    "pulsefield_model.timing.providers.oracle.oracle_timing_grid_from_beatmap",
                    side_effect=AssertionError("oracle files should not be read"),
                ),
            ):
                summary = run_timing_v3_projection_evaluation(
                    baseline_jsonl_path=baseline,
                    output_jsonl_path=root / "projection.jsonl",
                    summary_json_path=root / "summary.json",
                    progress_every=0,
                )

            self.assertEqual(summary["results"]["successful_audio_count"], 1)

    def test_workers_above_one_are_rejected_until_process_support_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            baseline = _write_jsonl(
                root / "baseline.jsonl",
                [
                    _baseline_row(
                        "audio-a",
                        predicted_segments=_segments((0.0, 500.0)),
                        oracle_segment_groups=[_segments((0.0, 500.0))],
                    )
                ],
            )

            with pytest.raises(ValueError, match="workers=1"):
                run_timing_v3_projection_evaluation(
                    baseline_jsonl_path=baseline,
                    output_jsonl_path=root / "projection.jsonl",
                    workers=2,
                )


def _baseline_row(
    audio_key: str,
    *,
    predicted_segments: list[dict[str, object]],
    oracle_segment_groups: list[list[dict[str, object]]],
    pilot_stratum: str = "stable",
    ok: bool = True,
    failure_stage: str | None = None,
) -> dict[str, object]:
    return {
        "schema": BASELINE_RESULT_SCHEMA,
        "resume": {
            "schema": "pulsefield_model.timing_v3_cache_backed_v2_baseline_resume_v2",
            "fingerprint": f"baseline-{audio_key}",
        },
        "ok": ok,
        "audio_key": audio_key,
        "row_index": 0,
        "source_line_numbers": [1],
        "evaluation_strata": {
            "pilot_stratum": pilot_stratum,
            "pilot_quota_group": f"{pilot_stratum}-quota",
            "label_stratum": pilot_stratum,
            "label_confidence": "synthetic",
            "label_ambiguous": False,
            "source_long_track": False,
        },
        "audio_path": f"{audio_key}.wav",
        "beatmap_paths": [f"{audio_key}-{index}.osu" for index in range(len(oracle_segment_groups))],
        "prediction": {
            "provider": "stored",
            "checkpoint_path": "stored-checkpoint",
            "source_path": f"{audio_key}.wav",
            "frame_count": 250,
            "frame_rate_hz": 50.0,
        },
        "fit": {
            "score": 0.9,
            "diagnostics": {},
            "predicted_segment_count": len(predicted_segments),
            "predicted_segments": predicted_segments,
        },
        "comparisons": [
            {
                "ok": True,
                "beatmap_path": f"{audio_key}-{index}.osu",
                "oracle_segments": oracle_segments,
                "metrics": {},
                "drift_metrics": {},
                "error_type": None,
                "error": None,
            }
            for index, oracle_segments in enumerate(oracle_segment_groups)
        ],
        "paired_metrics": {"comparison_count": len(oracle_segment_groups)} if ok else None,
        "failure_stage": failure_stage,
        "error_type": "ValueError" if failure_stage else None,
        "error": "baseline comparison failed" if failure_stage else None,
    }


def _segments(*segments: tuple[float, float]) -> list[dict[str, object]]:
    return [
        {
            "offset_ms": offset_ms,
            "beat_length_ms": beat_length_ms,
            "bpm": 60000.0 / beat_length_ms,
            "meter": 4,
        }
        for offset_ms, beat_length_ms in segments
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _behavior(source_sha256: str) -> dict[str, object]:
    return {
        "schema": "test-behavior",
        "fingerprint": source_sha256,
        "python": {"implementation": "cpython", "version": "test", "version_info": [3, 10, 0]},
        "numpy": {"version": "test"},
        "source_modules": [
            {
                "module": "pulsefield_model.timing.evaluation.v3_projection",
                "path": "src/pulsefield_model/timing/evaluation/v3_projection.py",
                "sha256": source_sha256,
            }
        ],
    }


def _fake_ratio_compare(predicted_grid: object, stored_oracles: object, **_kwargs: object) -> list[dict[str, object]]:
    first_segment = predicted_grid.segments[0]  # type: ignore[attr-defined]
    first_period_ms = round(float(first_segment.beat_length_ms), 6)
    rows: list[dict[str, object]] = []
    for oracle in stored_oracles:  # type: ignore[union-attr]
        beatmap_path = str(oracle.beatmap_path)
        if "ratio-one" in beatmap_path:
            value = 20.0 if first_period_ms == 525.0 else 10.0
        elif "ratio-two" in beatmap_path:
            value = 60.0 if first_period_ms == 525.0 else 30.0
        else:
            value = 100.0
        rows.append(
            {
                "ok": True,
                "baseline_comparison_index": oracle.baseline_comparison_index,
                "beatmap_path": oracle.beatmap_path,
                "oracle_segments": oracle.oracle_segments_payload,
                "metrics": {
                    "mean_phase_error_ms": value,
                    "max_phase_error_ms": value,
                },
                "drift_metrics": {},
                "error_type": None,
                "error": None,
            }
        )
    return rows


def _stratum_entry(summary: dict[str, object], value: object, *, dimension: str = "pilot_stratum") -> dict[str, object]:
    stratified = summary["stratified"]
    if not isinstance(stratified, dict):
        raise AssertionError("stratified summary must be a dict")
    entries = stratified[dimension]
    if not isinstance(entries, list):
        raise AssertionError(f"{dimension} summary must be a list")
    for entry in entries:
        if isinstance(entry, dict) and entry["value"] == value:
            return entry
    raise AssertionError(f"missing {dimension} stratum {value!r}")


if __name__ == "__main__":
    unittest.main()
