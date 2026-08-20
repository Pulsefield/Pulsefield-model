from __future__ import annotations

import concurrent.futures
import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import pulsefield_model.timing.evaluation.baseline as baseline_module
from pulsefield_model.timing.evaluation.baseline import run_cache_backed_v2_baseline
from pulsefield_model.timing.evaluation.inventory import TIMING_V3_INVENTORY_AUDIO_ROW_SCHEMA
from pulsefield_model.timing.grid_fitting import TimingFitDiagnostics, TimingFitResult
from pulsefield_model.timing.providers.beatthis import BEATTHIS_PROVIDER_NAME
from pulsefield_model.timing.providers.beatthis_cache import (
    BeatThisFramePredictionCacheConfig,
    beatthis_audio_cache_key,
    beatthis_frame_prediction_cache_path,
    save_beatthis_frame_prediction_cache,
)
from pulsefield_model.timing.schema import FittedTimingGrid, FrameTimingPrediction, TimingSegment


class _StaticFitter:
    def __init__(self, grid: FittedTimingGrid) -> None:
        self.grid = grid
        self.calls: list[int] = []
        self.config = None

    def fit(self, prediction: FrameTimingPrediction) -> TimingFitResult:
        self.calls.append(prediction.frame_count)
        return TimingFitResult(
            grid=self.grid,
            score=0.9,
            diagnostics=TimingFitDiagnostics(
                fit_score=0.9,
                selected_period_frames=25.0,
                selected_offset_frames=0.0,
                selected_bpm=120.0,
                candidate_count=1,
                half_tempo_score=0.1,
                double_tempo_score=0.2,
                raw_selected_bpm=120.0,
                raw_score=0.9,
                tempo_multiplier=1.0,
                segment_alias_switch_count=0,
                tempo_multiplier_distribution={"1.0": 1},
            ),
        )


class _TimeoutFitter:
    config = None

    def fit(self, prediction: FrameTimingPrediction) -> TimingFitResult:
        raise TimeoutError("synthetic fit timeout")


class TimingV3BaselineTests(unittest.TestCase):
    def test_runs_from_existing_cache_and_merges_maps_by_audio_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            audio_path = root / "song.wav"
            audio_path.write_bytes(b"fake-audio")
            map_a = _write_osu(root / "a.osu", ["0,500,4,2,0,80,1,0"])
            map_b = _write_osu(root / "b.osu", ["0,500,4,2,0,80,1,0"])
            audio_key = beatthis_audio_cache_key(audio_path)
            inventory = _write_jsonl(
                root / "inventory.jsonl",
                [
                    {
                        "audio_key": audio_key,
                        "audio_path": audio_path.as_posix(),
                        "beatmap_path": map_b.as_posix(),
                    },
                    {
                        "audio_key": audio_key,
                        "audio_path": audio_path.as_posix(),
                        "beatmap_path": map_a.as_posix(),
                    },
                ],
            )
            cache_config = _save_prediction_cache(root, audio_key, audio_path)
            fitter = _StaticFitter(FittedTimingGrid((TimingSegment(offset_ms=0.0, beat_length_ms=500.0),)))

            summary = run_cache_backed_v2_baseline(
                inventory_path=inventory,
                output_jsonl_path=root / "results.jsonl",
                summary_json_path=root / "summary.json",
                cache_root=cache_config.cache_root,
                cache_version=cache_config.cache_version,
                checkpoint_path=cache_config.checkpoint_path,
                progress_every=0,
                fitter=fitter,  # type: ignore[arg-type]
            )

            results = _read_jsonl(root / "results.jsonl")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["schema"], "pulsefield_model.timing_v3_cache_backed_v2_baseline_result_v2")
            self.assertTrue(results[0]["ok"])
            self.assertEqual(results[0]["audio_key"], audio_key)
            self.assertEqual(
                results[0]["beatmap_paths"],
                sorted([map_a.resolve(strict=False).as_posix(), map_b.resolve(strict=False).as_posix()]),
            )
            self.assertEqual(len(results[0]["comparisons"]), 2)
            self.assertEqual(fitter.calls, [100])
            self.assertEqual(summary["source"]["unique_audio_count"], 1)
            self.assertEqual(summary["source"]["inventory_schema"], TIMING_V3_INVENTORY_AUDIO_ROW_SCHEMA)
            self.assertEqual(summary["results"]["paired_comparison_count"], 2)
            self.assertEqual(summary["metrics"]["mean_phase_error_ms"]["mean"], 0.0)
            self.assertEqual(summary["metrics"]["local_bpm_alias_mae"]["mean"], 0.0)
            self.assertEqual(summary["audio_group_metrics"]["mean_phase_error_ms"]["count"], 1)
            self.assertEqual(summary["difficulty_comparison_metrics"]["mean_phase_error_ms"]["count"], 2)
            self.assertNotIn("predicted_segment_count", summary["difficulty_comparison_metrics"])
            self.assertFalse((root / "results.jsonl.tmp").exists())
            self.assertFalse((root / "summary.json.tmp").exists())

    def test_headline_metrics_weight_each_audio_group_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            first_audio = root / "first.wav"
            second_audio = root / "second.wav"
            first_audio.write_bytes(b"first-audio")
            second_audio.write_bytes(b"second-audio")
            exact_a = _write_osu(root / "exact-a.osu", ["0,500,4,2,0,80,1,0"])
            exact_b = _write_osu(root / "exact-b.osu", ["0,500,4,2,0,80,1,0"])
            shifted = _write_osu(root / "shifted.osu", ["250,500,4,2,0,80,1,0"])
            first_key = beatthis_audio_cache_key(first_audio)
            second_key = beatthis_audio_cache_key(second_audio)
            inventory = _write_jsonl(
                root / "inventory.jsonl",
                [
                    {
                        "cache": {"audio_cache_key": first_key},
                        "resolved_audio_path": first_audio.as_posix(),
                        "maps": [
                            {"resolved_beatmap_path": exact_a.as_posix()},
                            {"resolved_beatmap_path": exact_b.as_posix()},
                        ],
                    },
                    {
                        "cache": {"audio_cache_key": second_key},
                        "resolved_audio_path": second_audio.as_posix(),
                        "maps": [{"resolved_beatmap_path": shifted.as_posix()}],
                    },
                ],
            )
            cache_config = _save_prediction_cache(root, first_key, first_audio)
            _save_prediction_cache(root, second_key, second_audio)
            fitter = _StaticFitter(FittedTimingGrid((TimingSegment(offset_ms=0.0, beat_length_ms=500.0),)))

            summary = run_cache_backed_v2_baseline(
                inventory_path=inventory,
                output_jsonl_path=root / "results.jsonl",
                summary_json_path=root / "summary.json",
                cache_root=cache_config.cache_root,
                cache_version=cache_config.cache_version,
                checkpoint_path=cache_config.checkpoint_path,
                progress_every=0,
                fitter=fitter,  # type: ignore[arg-type]
            )

            self.assertEqual(summary["results"]["successful_audio_count"], 2)
            self.assertEqual(summary["results"]["paired_comparison_count"], 3)
            self.assertAlmostEqual(summary["audio_group_metrics"]["mean_phase_error_ms"]["mean"], 125.0, delta=1e-4)
            self.assertAlmostEqual(
                summary["difficulty_comparison_metrics"]["mean_phase_error_ms"]["mean"],
                250.0 / 3.0,
                delta=1e-4,
            )
            self.assertEqual(summary["metrics"], summary["audio_group_metrics"])

    def test_evaluation_strata_passthrough_and_stratified_fit_vs_compare_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            ok_audio = root / "ok.wav"
            cache_fail_audio = root / "cache-fail.wav"
            compare_fail_audio = root / "compare-fail.wav"
            for path in (ok_audio, cache_fail_audio, compare_fail_audio):
                path.write_bytes(path.name.encode("utf-8"))
            ok_map = _write_osu(root / "ok.osu", ["0,500,4,2,0,80,1,0"])
            missing_map = root / "missing.osu"
            ok_key = beatthis_audio_cache_key(ok_audio)
            cache_fail_key = beatthis_audio_cache_key(cache_fail_audio)
            compare_fail_key = beatthis_audio_cache_key(compare_fail_audio)
            cache_config = _save_prediction_cache(root, ok_key, ok_audio)
            _save_prediction_cache(root, compare_fail_key, compare_fail_audio)
            inventory = _write_jsonl(
                root / "inventory.jsonl",
                [
                    {
                        "audio_key": ok_key,
                        "audio_path": ok_audio.as_posix(),
                        "beatmap_path": ok_map.as_posix(),
                        "pilot_stratum": "stable",
                        "pilot_quota_group": "stable-main",
                        "label": {"stratum": "stable", "confidence": "high", "ambiguous": False},
                        "source": {"long_track": False},
                    },
                    {
                        "audio_key": cache_fail_key,
                        "audio_path": cache_fail_audio.as_posix(),
                        "beatmap_path": ok_map.as_posix(),
                        "pilot_stratum": "jump",
                        "pilot_quota_group": "jump-main",
                        "label": {"stratum": "jump", "confidence": "low", "ambiguous": True},
                        "source": {"long_track": True},
                    },
                    {
                        "audio_key": compare_fail_key,
                        "audio_path": compare_fail_audio.as_posix(),
                        "beatmap_path": missing_map.as_posix(),
                        "pilot_stratum": "stable",
                        "pilot_quota_group": "stable-main",
                        "label": {"stratum": "ambiguous", "confidence": "medium", "ambiguous": False},
                        "source": {"long_track": False},
                    },
                ],
            )
            fitter = _StaticFitter(FittedTimingGrid((TimingSegment(offset_ms=0.0, beat_length_ms=500.0),)))

            summary = run_cache_backed_v2_baseline(
                inventory_path=inventory,
                output_jsonl_path=root / "results.jsonl",
                summary_json_path=root / "summary.json",
                cache_root=cache_config.cache_root,
                cache_version=cache_config.cache_version,
                checkpoint_path=cache_config.checkpoint_path,
                progress_every=0,
                fitter=fitter,  # type: ignore[arg-type]
            )

            results = _read_jsonl(root / "results.jsonl")
            ok_result = next(result for result in results if result["audio_key"] == ok_key)
            self.assertEqual(
                ok_result["evaluation_strata"],
                {
                    "pilot_stratum": "stable",
                    "pilot_quota_group": "stable-main",
                    "label_stratum": "stable",
                    "label_confidence": "high",
                    "label_ambiguous": False,
                    "source_long_track": False,
                },
            )
            self.assertEqual(ok_result["resume"]["components"]["inventory"]["evaluation_strata"], ok_result["evaluation_strata"])
            self.assertEqual(summary["schema"], "pulsefield_model.timing_v3_cache_backed_v2_baseline_summary_v2")
            self.assertEqual(summary["results"]["successful_audio_count"], 1)
            self.assertEqual(summary["results"]["failed_audio_count"], 2)
            self.assertEqual(summary["results"]["fit_success_audio_count"], 2)
            self.assertEqual(summary["results"]["fit_failure_audio_count"], 1)
            self.assertEqual(summary["results"]["comparison_attempted_audio_count"], 2)
            self.assertEqual(summary["results"]["comparison_eligible_audio_count"], 1)
            self.assertEqual(summary["results"]["comparator_unavailable_audio_count"], 1)
            self.assertEqual(summary["failures"]["fit_failure_audio_count"], 1)
            self.assertEqual(summary["failures"]["comparator_unavailable_audio_count"], 1)

            stable = _stratum_entry(summary, "pilot_stratum", "stable")
            self.assertEqual(stable["audio_count"], 2)
            self.assertEqual(stable["ok"], 1)
            self.assertEqual(stable["failed"], 1)
            self.assertEqual(stable["fit_success_audio_count"], 2)
            self.assertEqual(stable["fit_failure_audio_count"], 0)
            self.assertEqual(stable["comparison_attempted_audio_count"], 2)
            self.assertEqual(stable["comparison_eligible_audio_count"], 1)
            self.assertEqual(stable["comparator_unavailable_audio_count"], 1)
            self.assertEqual(stable["metrics"]["mean_phase_error_ms"]["count"], 1)

            ambiguous = _stratum_entry(summary, "label_stratum", "ambiguous")
            self.assertEqual(ambiguous["audio_count"], 1)
            self.assertEqual(ambiguous["ok"], 0)
            self.assertEqual(ambiguous["failed"], 1)
            self.assertEqual(ambiguous["fit_success_audio_count"], 1)
            self.assertEqual(ambiguous["fit_failure_audio_count"], 0)
            self.assertEqual(ambiguous["comparison_attempted_audio_count"], 1)
            self.assertEqual(ambiguous["comparison_eligible_audio_count"], 0)
            self.assertEqual(ambiguous["comparator_unavailable_audio_count"], 1)
            self.assertEqual(ambiguous["metrics"]["mean_phase_error_ms"]["count"], 0)

            jump = _stratum_entry(summary, "label_stratum", "jump")
            self.assertEqual(jump["audio_count"], 1)
            self.assertEqual(jump["ok"], 0)
            self.assertEqual(jump["failed"], 1)
            self.assertEqual(jump["fit_success_audio_count"], 0)
            self.assertEqual(jump["fit_failure_audio_count"], 1)
            self.assertEqual(jump["comparison_attempted_audio_count"], 0)
            self.assertEqual(jump["comparison_eligible_audio_count"], 0)
            self.assertEqual(jump["comparator_unavailable_audio_count"], 0)

    def test_resume_skips_failed_rows_unless_retry_failures_is_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            audio_path = root / "song.wav"
            audio_path.write_bytes(b"fake-audio")
            map_path = _write_osu(root / "a.osu", ["0,500,4,2,0,80,1,0"])
            audio_key = beatthis_audio_cache_key(audio_path)
            inventory = _write_jsonl(
                root / "inventory.jsonl",
                [
                    {
                        "audio_key": audio_key,
                        "audio_path": audio_path.as_posix(),
                        "beatmap_path": map_path.as_posix(),
                    }
                ],
            )
            cache_config = BeatThisFramePredictionCacheConfig(
                cache_root=root / "cache",
                cache_version="baseline-test-cache",
                checkpoint_path="checkpoint-a",
            )
            fitter = _StaticFitter(FittedTimingGrid((TimingSegment(offset_ms=0.0, beat_length_ms=500.0),)))

            first = run_cache_backed_v2_baseline(
                inventory_path=inventory,
                output_jsonl_path=root / "results.jsonl",
                summary_json_path=root / "summary.json",
                cache_root=cache_config.cache_root,
                cache_version=cache_config.cache_version,
                checkpoint_path=cache_config.checkpoint_path,
                progress_every=0,
                fitter=fitter,  # type: ignore[arg-type]
            )
            second = run_cache_backed_v2_baseline(
                inventory_path=inventory,
                output_jsonl_path=root / "results.jsonl",
                summary_json_path=root / "summary.json",
                cache_root=cache_config.cache_root,
                cache_version=cache_config.cache_version,
                checkpoint_path=cache_config.checkpoint_path,
                progress_every=0,
                fitter=fitter,  # type: ignore[arg-type]
            )
            third = run_cache_backed_v2_baseline(
                inventory_path=inventory,
                output_jsonl_path=root / "results.jsonl",
                summary_json_path=root / "summary.json",
                cache_root=cache_config.cache_root,
                cache_version=cache_config.cache_version,
                checkpoint_path=cache_config.checkpoint_path,
                retry_failures=True,
                progress_every=0,
                fitter=fitter,  # type: ignore[arg-type]
            )
            _save_prediction_cache(root, audio_key, audio_path)
            fourth = run_cache_backed_v2_baseline(
                inventory_path=inventory,
                output_jsonl_path=root / "results.jsonl",
                summary_json_path=root / "summary.json",
                cache_root=cache_config.cache_root,
                cache_version=cache_config.cache_version,
                checkpoint_path=cache_config.checkpoint_path,
                progress_every=0,
                fitter=fitter,  # type: ignore[arg-type]
            )

            self.assertEqual(first["failures"]["stage_counts"], {"cache": 1})
            self.assertEqual(second["run"]["skipped_failure_count"], 1)
            self.assertEqual(second["failures"]["stage_counts"], {"cache": 1})
            self.assertEqual(third["run"]["processed_count"], 1)
            self.assertEqual(third["failures"]["stage_counts"], {"cache": 1})
            self.assertEqual(fourth["run"]["stale_existing_count"], 1)
            self.assertEqual(fourth["results"]["successful_audio_count"], 1)
            self.assertEqual(fitter.calls, [100])
            self.assertTrue(_read_jsonl(root / "results.jsonl")[0]["ok"])

    def test_resume_does_not_reuse_result_when_cache_config_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            audio_path = root / "song.wav"
            audio_path.write_bytes(b"fake-audio")
            map_path = _write_osu(root / "a.osu", ["0,500,4,2,0,80,1,0"])
            audio_key = beatthis_audio_cache_key(audio_path)
            inventory = _write_jsonl(
                root / "inventory.jsonl",
                [{"audio_key": audio_key, "audio_path": audio_path.as_posix(), "beatmap_path": map_path.as_posix()}],
            )
            cache_config = _save_prediction_cache(root, audio_key, audio_path, checkpoint_path="checkpoint-a")
            fitter = _StaticFitter(FittedTimingGrid((TimingSegment(offset_ms=0.0, beat_length_ms=500.0),)))

            first = run_cache_backed_v2_baseline(
                inventory_path=inventory,
                output_jsonl_path=root / "results.jsonl",
                summary_json_path=root / "summary.json",
                cache_root=cache_config.cache_root,
                cache_version=cache_config.cache_version,
                checkpoint_path="checkpoint-a",
                progress_every=0,
                fitter=fitter,  # type: ignore[arg-type]
            )
            second = run_cache_backed_v2_baseline(
                inventory_path=inventory,
                output_jsonl_path=root / "results.jsonl",
                summary_json_path=root / "summary.json",
                cache_root=cache_config.cache_root,
                cache_version=cache_config.cache_version,
                checkpoint_path="checkpoint-b",
                progress_every=0,
                fitter=fitter,  # type: ignore[arg-type]
            )

            self.assertEqual(first["results"]["successful_audio_count"], 1)
            self.assertEqual(second["run"]["stale_existing_count"], 1)
            self.assertEqual(second["failures"]["stage_counts"], {"cache": 1})
            self.assertEqual(fitter.calls, [100])

    def test_resume_does_not_reuse_result_when_inventory_maps_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            audio_path = root / "song.wav"
            audio_path.write_bytes(b"fake-audio")
            exact = _write_osu(root / "exact.osu", ["0,500,4,2,0,80,1,0"])
            shifted = _write_osu(root / "shifted.osu", ["250,500,4,2,0,80,1,0"])
            audio_key = beatthis_audio_cache_key(audio_path)
            inventory = root / "inventory.jsonl"
            _write_jsonl(
                inventory,
                [{"audio_key": audio_key, "audio_path": audio_path.as_posix(), "beatmap_path": exact.as_posix()}],
            )
            cache_config = _save_prediction_cache(root, audio_key, audio_path)
            fitter = _StaticFitter(FittedTimingGrid((TimingSegment(offset_ms=0.0, beat_length_ms=500.0),)))

            run_cache_backed_v2_baseline(
                inventory_path=inventory,
                output_jsonl_path=root / "results.jsonl",
                summary_json_path=root / "summary.json",
                cache_root=cache_config.cache_root,
                cache_version=cache_config.cache_version,
                checkpoint_path=cache_config.checkpoint_path,
                progress_every=0,
                fitter=fitter,  # type: ignore[arg-type]
            )
            _write_jsonl(
                inventory,
                [{"audio_key": audio_key, "audio_path": audio_path.as_posix(), "beatmap_path": shifted.as_posix()}],
            )
            second = run_cache_backed_v2_baseline(
                inventory_path=inventory,
                output_jsonl_path=root / "results.jsonl",
                summary_json_path=root / "summary.json",
                cache_root=cache_config.cache_root,
                cache_version=cache_config.cache_version,
                checkpoint_path=cache_config.checkpoint_path,
                progress_every=0,
                fitter=fitter,  # type: ignore[arg-type]
            )

            result = _read_jsonl(root / "results.jsonl")[0]
            self.assertEqual(second["run"]["stale_existing_count"], 1)
            self.assertEqual(result["beatmap_paths"], [shifted.resolve(strict=False).as_posix()])
            self.assertAlmostEqual(result["paired_metrics"]["mean_phase_error_ms"]["mean"], 250.0, delta=1e-4)
            self.assertEqual(fitter.calls, [100, 100])

    def test_resume_does_not_reuse_result_when_cache_file_is_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            audio_path = root / "song.wav"
            audio_path.write_bytes(b"fake-audio")
            map_path = _write_osu(root / "a.osu", ["0,500,4,2,0,80,1,0"])
            audio_key = beatthis_audio_cache_key(audio_path)
            inventory = _write_jsonl(
                root / "inventory.jsonl",
                [{"audio_key": audio_key, "audio_path": audio_path.as_posix(), "beatmap_path": map_path.as_posix()}],
            )
            cache_config = _save_prediction_cache(root, audio_key, audio_path, frame_count=100)
            fitter = _StaticFitter(FittedTimingGrid((TimingSegment(offset_ms=0.0, beat_length_ms=500.0),)))

            first = run_cache_backed_v2_baseline(
                inventory_path=inventory,
                output_jsonl_path=root / "results.jsonl",
                summary_json_path=root / "summary.json",
                cache_root=cache_config.cache_root,
                cache_version=cache_config.cache_version,
                checkpoint_path=cache_config.checkpoint_path,
                progress_every=0,
                fitter=fitter,  # type: ignore[arg-type]
            )
            _save_prediction_cache(root, audio_key, audio_path, frame_count=120)
            second = run_cache_backed_v2_baseline(
                inventory_path=inventory,
                output_jsonl_path=root / "results.jsonl",
                summary_json_path=root / "summary.json",
                cache_root=cache_config.cache_root,
                cache_version=cache_config.cache_version,
                checkpoint_path=cache_config.checkpoint_path,
                progress_every=0,
                fitter=fitter,  # type: ignore[arg-type]
            )

            result = _read_jsonl(root / "results.jsonl")[0]
            self.assertEqual(first["results"]["successful_audio_count"], 1)
            self.assertEqual(second["run"]["stale_existing_count"], 1)
            self.assertEqual(fitter.calls, [100, 120])
            self.assertEqual(result["prediction"]["frame_count"], 120)
            self.assertEqual(
                result["resume"]["components"]["cache_file"]["size_bytes"],
                beatthis_frame_prediction_cache_path(audio_key, cache_config).stat().st_size,
            )

    def test_missing_audio_canonical_inventory_row_becomes_cache_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            beatmap_dir = dataset_root / "0" / "1001"
            missing_audio = beatmap_dir / "missing.wav"
            map_path = _write_osu(beatmap_dir / "map.osu", ["0,500,4,2,0,80,1,0"])
            inventory = _write_jsonl(
                root / "inventory.jsonl",
                [
                    {
                        "schema": TIMING_V3_INVENTORY_AUDIO_ROW_SCHEMA,
                        "resolved_audio_path": missing_audio.as_posix(),
                        "audio_exists": False,
                        "cache": {"audio_cache_key": None},
                        "maps": [{"resolved_beatmap_path": map_path.as_posix()}],
                    }
                ],
            )
            fitter = _StaticFitter(FittedTimingGrid((TimingSegment(offset_ms=0.0, beat_length_ms=500.0),)))

            summary = run_cache_backed_v2_baseline(
                inventory_path=inventory,
                output_jsonl_path=root / "results.jsonl",
                summary_json_path=root / "summary.json",
                dataset_root=dataset_root,
                cache_root=root / "cache",
                cache_version="baseline-test-cache",
                checkpoint_path="checkpoint-a",
                progress_every=0,
                fitter=fitter,  # type: ignore[arg-type]
            )

            result = _read_jsonl(root / "results.jsonl")[0]
            self.assertEqual(summary["failures"]["stage_counts"], {"cache": 1})
            self.assertTrue(result["audio_key"].startswith("missing-audio:"))
            self.assertEqual(result["failure_stage"], "cache")
            self.assertEqual(fitter.calls, [])

    def test_timeout_error_from_fit_is_owned_by_outer_timeout_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            audio_path = root / "song.wav"
            audio_path.write_bytes(b"fake-audio")
            map_path = _write_osu(root / "a.osu", ["0,500,4,2,0,80,1,0"])
            audio_key = beatthis_audio_cache_key(audio_path)
            inventory = _write_jsonl(
                root / "inventory.jsonl",
                [{"audio_key": audio_key, "audio_path": audio_path.as_posix(), "beatmap_path": map_path.as_posix()}],
            )
            cache_config = _save_prediction_cache(root, audio_key, audio_path)

            summary = run_cache_backed_v2_baseline(
                inventory_path=inventory,
                output_jsonl_path=root / "results.jsonl",
                summary_json_path=root / "summary.json",
                cache_root=cache_config.cache_root,
                cache_version=cache_config.cache_version,
                checkpoint_path=cache_config.checkpoint_path,
                timeout_seconds=10.0,
                progress_every=0,
                fitter=_TimeoutFitter(),  # type: ignore[arg-type]
            )

            result = _read_jsonl(root / "results.jsonl")[0]
            self.assertEqual(summary["failures"]["stage_counts"], {"timeout": 1})
            self.assertEqual(result["failure_stage"], "timeout")
            self.assertEqual(result["error_type"], "TimeoutError")

    def test_rejects_paths_outside_dataset_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            inside_audio = dataset_root / "0" / "1001" / "audio.wav"
            inside_audio.parent.mkdir(parents=True, exist_ok=True)
            inside_audio.write_bytes(b"fake-audio")
            outside_audio = root / "outside.wav"
            outside_audio.write_bytes(b"fake-audio")
            map_path = _write_osu(dataset_root / "0" / "1001" / "map.osu", ["0,500,4,2,0,80,1,0"])
            outside_map = _write_osu(root / "outside.osu", ["0,500,4,2,0,80,1,0"])
            cases = [
                (
                    "absolute audio",
                    {
                        "audio_key": "outside-audio-key",
                        "resolved_audio_path": outside_audio.as_posix(),
                        "maps": [{"resolved_beatmap_path": map_path.as_posix()}],
                    },
                    "outside allowed root",
                ),
                (
                    "absolute beatmap",
                    {
                        "audio_key": "outside-map-key",
                        "resolved_audio_path": inside_audio.as_posix(),
                        "maps": [{"resolved_beatmap_path": outside_map.as_posix()}],
                    },
                    "outside allowed root",
                ),
                (
                    "relative escape",
                    {
                        "audio_key": "relative-escape-key",
                        "audio_path": "../outside.wav",
                        "beatmap_path": "0/1001/map.osu",
                    },
                    "without '..'",
                ),
            ]

            for name, row, error_pattern in cases:
                with self.subTest(name=name):
                    inventory = _write_jsonl(root / f"{name.replace(' ', '-')}.jsonl", [row])
                    with self.assertRaisesRegex(ValueError, error_pattern):
                        run_cache_backed_v2_baseline(
                            inventory_path=inventory,
                            output_jsonl_path=root / "results.jsonl",
                            summary_json_path=root / "summary.json",
                            dataset_root=dataset_root,
                            cache_root=root / "cache",
                            cache_version="baseline-test-cache",
                            checkpoint_path="checkpoint-a",
                            progress_every=0,
                            fitter=_StaticFitter(
                                FittedTimingGrid((TimingSegment(offset_ms=0.0, beat_length_ms=500.0),))
                            ),  # type: ignore[arg-type]
                        )

    def test_checkpoint_every_batches_jsonl_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            rows: list[dict[str, object]] = []
            cache_config: BeatThisFramePredictionCacheConfig | None = None
            for index in range(3):
                audio_path = root / f"song-{index}.wav"
                audio_path.write_bytes(f"fake-audio-{index}".encode("utf-8"))
                map_path = _write_osu(root / f"map-{index}.osu", ["0,500,4,2,0,80,1,0"])
                audio_key = beatthis_audio_cache_key(audio_path)
                cache_config = _save_prediction_cache(root, audio_key, audio_path)
                rows.append(
                    {
                        "audio_key": audio_key,
                        "audio_path": audio_path.as_posix(),
                        "beatmap_path": map_path.as_posix(),
                    }
                )
            inventory = _write_jsonl(root / "inventory.jsonl", rows)
            fitter = _StaticFitter(FittedTimingGrid((TimingSegment(offset_ms=0.0, beat_length_ms=500.0),)))
            assert cache_config is not None
            real_write = baseline_module._write_result_jsonl_atomic
            write_calls = 0

            def counting_write(path: Path, results: object) -> None:
                nonlocal write_calls
                write_calls += 1
                real_write(path, results)  # type: ignore[arg-type]

            with mock.patch.object(baseline_module, "_write_result_jsonl_atomic", side_effect=counting_write):
                run_cache_backed_v2_baseline(
                    inventory_path=inventory,
                    output_jsonl_path=root / "results.jsonl",
                    summary_json_path=root / "summary.json",
                    cache_root=cache_config.cache_root,
                    cache_version=cache_config.cache_version,
                    checkpoint_path=cache_config.checkpoint_path,
                    checkpoint_every=2,
                    progress_every=0,
                    fitter=fitter,  # type: ignore[arg-type]
                )

            self.assertEqual(write_calls, 2)
            self.assertEqual(len(_read_jsonl(root / "results.jsonl")), 3)

    def test_workers_two_matches_workers_one_for_default_fitter(self) -> None:
        self._skip_if_process_pool_unavailable()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            inventory, cache_config = _write_real_fitter_inventory(root, count=2)

            single = run_cache_backed_v2_baseline(
                inventory_path=inventory,
                output_jsonl_path=root / "single.jsonl",
                summary_json_path=root / "single-summary.json",
                cache_root=cache_config.cache_root,
                cache_version=cache_config.cache_version,
                checkpoint_path=cache_config.checkpoint_path,
                workers=1,
                progress_every=0,
            )
            parallel = run_cache_backed_v2_baseline(
                inventory_path=inventory,
                output_jsonl_path=root / "parallel.jsonl",
                summary_json_path=root / "parallel-summary.json",
                cache_root=cache_config.cache_root,
                cache_version=cache_config.cache_version,
                checkpoint_path=cache_config.checkpoint_path,
                workers=2,
                progress_every=0,
            )

            single_results = _read_jsonl(root / "single.jsonl")
            parallel_results = _read_jsonl(root / "parallel.jsonl")
            self.assertEqual([row["audio_key"] for row in single_results], [row["audio_key"] for row in parallel_results])
            self.assertEqual(
                [_stable_result_payload(row) for row in single_results],
                [_stable_result_payload(row) for row in parallel_results],
            )
            self.assertEqual(single["results"], parallel["results"])
            self.assertEqual(single["audio_group_metrics"], parallel["audio_group_metrics"])
            self.assertEqual(parallel["run"]["workers"], 2)
            self.assertTrue(parallel["run"]["parallel"])

    def test_workers_two_resume_skips_matching_existing_successes(self) -> None:
        self._skip_if_process_pool_unavailable()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            inventory, cache_config = _write_real_fitter_inventory(root, count=2)

            first = run_cache_backed_v2_baseline(
                inventory_path=inventory,
                output_jsonl_path=root / "results.jsonl",
                summary_json_path=root / "summary.json",
                cache_root=cache_config.cache_root,
                cache_version=cache_config.cache_version,
                checkpoint_path=cache_config.checkpoint_path,
                workers=2,
                progress_every=0,
            )
            second = run_cache_backed_v2_baseline(
                inventory_path=inventory,
                output_jsonl_path=root / "results.jsonl",
                summary_json_path=root / "summary.json",
                cache_root=cache_config.cache_root,
                cache_version=cache_config.cache_version,
                checkpoint_path=cache_config.checkpoint_path,
                workers=2,
                progress_every=0,
            )

            self.assertEqual(first["results"]["successful_audio_count"], 2)
            self.assertEqual(second["run"]["processed_count"], 0)
            self.assertEqual(second["run"]["skipped_success_count"], 2)
            self.assertEqual(second["run"]["stale_existing_count"], 0)

    def test_workers_gt_one_rejects_custom_fitter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            inventory, cache_config = _write_real_fitter_inventory(root, count=1)

            with self.assertRaisesRegex(ValueError, "workers > 1 requires fitter=None"):
                run_cache_backed_v2_baseline(
                    inventory_path=inventory,
                    output_jsonl_path=root / "results.jsonl",
                    summary_json_path=root / "summary.json",
                    cache_root=cache_config.cache_root,
                    cache_version=cache_config.cache_version,
                    checkpoint_path=cache_config.checkpoint_path,
                    workers=2,
                    progress_every=0,
                    fitter=_StaticFitter(FittedTimingGrid((TimingSegment(offset_ms=0.0, beat_length_ms=500.0),))),  # type: ignore[arg-type]
                )

    def test_workers_two_pool_permission_error_is_clear_runtime_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            inventory, cache_config = _write_real_fitter_inventory(root, count=1)

            with mock.patch.object(
                baseline_module.concurrent.futures,
                "ProcessPoolExecutor",
                side_effect=PermissionError("sem denied"),
            ):
                with self.assertRaisesRegex(RuntimeError, "workers=1 or allow OS semaphore access"):
                    run_cache_backed_v2_baseline(
                        inventory_path=inventory,
                        output_jsonl_path=root / "results.jsonl",
                        summary_json_path=root / "summary.json",
                        cache_root=cache_config.cache_root,
                        cache_version=cache_config.cache_version,
                        checkpoint_path=cache_config.checkpoint_path,
                        workers=2,
                        progress_every=0,
                    )

    def _skip_if_process_pool_unavailable(self) -> None:
        if not _process_pool_supported():
            self.skipTest("ProcessPoolExecutor is unavailable in this sandbox; workers=2 requires OS semaphore access")


def _save_prediction_cache(
    root: Path,
    audio_key: str,
    audio_path: Path,
    *,
    checkpoint_path: str = "checkpoint-a",
    frame_count: int = 100,
) -> BeatThisFramePredictionCacheConfig:
    config = BeatThisFramePredictionCacheConfig(
        cache_root=root / "cache",
        cache_version="baseline-test-cache",
        checkpoint_path=checkpoint_path,
    )
    save_beatthis_frame_prediction_cache(
        FrameTimingPrediction(
            provider=BEATTHIS_PROVIDER_NAME,
            checkpoint_path=checkpoint_path,
            source_path=audio_path,
            beat_prob=np.ones(frame_count, dtype=np.float32),
            downbeat_prob=np.zeros(frame_count, dtype=np.float32),
            frame_rate_hz=config.frame_rate_hz,
        ),
        audio_key,
        config,
    )
    return config


def _process_pool_probe_value() -> int:
    return 1


def _process_pool_supported() -> bool:
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_process_pool_probe_value)
            return future.result(timeout=10.0) == 1
    except (PermissionError, NotImplementedError, RuntimeError):
        return False


def _write_real_fitter_inventory(root: Path, *, count: int) -> tuple[Path, BeatThisFramePredictionCacheConfig]:
    rows: list[dict[str, object]] = []
    cache_config: BeatThisFramePredictionCacheConfig | None = None
    for index in range(count):
        audio_path = root / f"real-{index}.wav"
        audio_path.write_bytes(f"fake-audio-{index}".encode("utf-8"))
        map_path = _write_osu(root / f"real-{index}.osu", ["0,500,4,2,0,80,1,0"])
        audio_key = beatthis_audio_cache_key(audio_path)
        cache_config = _save_pulse_prediction_cache(root, audio_key, audio_path)
        rows.append(
            {
                "audio_key": audio_key,
                "audio_path": audio_path.as_posix(),
                "beatmap_path": map_path.as_posix(),
            }
        )
    assert cache_config is not None
    return _write_jsonl(root / "real-inventory.jsonl", rows), cache_config


def _save_pulse_prediction_cache(
    root: Path,
    audio_key: str,
    audio_path: Path,
) -> BeatThisFramePredictionCacheConfig:
    config = BeatThisFramePredictionCacheConfig(
        cache_root=root / "cache",
        cache_version="baseline-test-cache",
        checkpoint_path="checkpoint-a",
    )
    frame_count = 600
    frame_rate_hz = config.frame_rate_hz
    frame_times_ms = np.arange(frame_count, dtype=np.float64) / frame_rate_hz * 1000.0
    phase = (frame_times_ms / 500.0) % 1.0
    distance_ms = np.minimum(phase, 1.0 - phase) * 500.0
    beat_prob = np.maximum(0.05, 1.0 - distance_ms / 40.0).astype(np.float32)
    downbeat_prob = np.zeros(frame_count, dtype=np.float32)
    save_beatthis_frame_prediction_cache(
        FrameTimingPrediction(
            provider=BEATTHIS_PROVIDER_NAME,
            checkpoint_path="checkpoint-a",
            source_path=audio_path,
            beat_prob=beat_prob,
            downbeat_prob=downbeat_prob,
            frame_rate_hz=frame_rate_hz,
        ),
        audio_key,
        config,
    )
    return config


def _stable_result_payload(result: dict[str, object]) -> dict[str, object]:
    return {
        "ok": result["ok"],
        "audio_key": result["audio_key"],
        "audio_path": result["audio_path"],
        "beatmap_paths": result["beatmap_paths"],
        "prediction": result["prediction"],
        "fit": result["fit"],
        "paired_metrics": result["paired_metrics"],
        "comparisons": [
            {
                "ok": comparison["ok"],
                "beatmap_path": comparison["beatmap_path"],
                "oracle_segments": comparison["oracle_segments"],
                "metrics": comparison["metrics"],
                "drift_metrics": comparison.get("drift_metrics"),
                "error_type": comparison["error_type"],
                "error": comparison["error"],
            }
            for comparison in result["comparisons"]  # type: ignore[index]
        ],
        "failure_stage": result["failure_stage"],
        "error_type": result["error_type"],
        "error": result["error"],
    }


def _write_osu(path: Path, timing_lines: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        textwrap.dedent(
            f"""\
            osu file format v14

            [TimingPoints]
            {chr(10).join(timing_lines)}
            """
        ),
        encoding="utf-8",
    )
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _stratum_entry(summary: dict[str, object], dimension: str, value: object) -> dict[str, object]:
    stratified = summary["stratified"]
    if not isinstance(stratified, dict):
        raise AssertionError(f"summary stratified payload is not a dict: {stratified!r}")
    entries = stratified[dimension]
    if not isinstance(entries, list):
        raise AssertionError(f"summary stratum dimension is not a list: {entries!r}")
    for entry in entries:
        if isinstance(entry, dict) and entry["value"] == value:
            return entry
    raise AssertionError(f"missing {dimension} stratum {value!r}: {entries!r}")


if __name__ == "__main__":
    unittest.main()
