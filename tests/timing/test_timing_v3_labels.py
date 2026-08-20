from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from pulsefield_model.timing.evaluation import labels as labels_module
from pulsefield_model.timing.evaluation.inventory import TIMING_V3_INVENTORY_AUDIO_ROW_SCHEMA
from pulsefield_model.timing.evaluation.labels import (
    LABEL_AMBIGUOUS,
    LABEL_DENSE,
    LABEL_RAMP_CANDIDATE,
    LABEL_STABLE,
    TIMING_V3_LABEL_AUDIO_ROW_SCHEMA,
    TIMING_V3_LABEL_PILOT_SCHEMA,
    TIMING_V3_LABEL_REPORT_SCHEMA,
    build_timing_v3_labels,
    main,
    select_timing_v3_pilot,
)


class TimingV3LabelsTests(unittest.TestCase):
    def test_builds_compact_stable_audio_label_with_cross_map_and_metadata_agreement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            beatmapset = root / "dataset" / "0" / "1001"
            audio_path = beatmapset / "audio.wav"
            audio_path.parent.mkdir(parents=True)
            audio_path.write_bytes(b"audio")
            first_map = beatmapset / "a.osu"
            second_map = beatmapset / "b.osu"
            _write_osu(
                first_map,
                timing_lines=["0,500,4,2,0,80,1,0"],
                hitobject_lines=[
                    _tap_line(index * 500.0)
                    for index in range(9)
                ],
            )
            _write_osu(
                second_map,
                timing_lines=["0,500,4,2,0,80,1,0"],
                hitobject_lines=[
                    _tap_line(250.0 + index * 500.0)
                    for index in range(9)
                ],
            )
            metadata_path = beatmapset / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "fetched_at": "2026-08-05T00:00:00+00:00",
                        "api_endpoint": "https://osu.ppy.sh/api/v2/beatmapsets/1001",
                        "id": 1001,
                        "bpm": 240,
                        "beatmaps": [
                            {"id": 11, "beatmapset_id": 1001, "bpm": 240},
                            {"id": 12, "beatmapset_id": 1001, "bpm": 240},
                        ],
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            inventory_path = root / "inventory.jsonl"
            _write_jsonl(
                inventory_path,
                [
                    _inventory_row(
                        audio_path=audio_path,
                        maps=[
                            _map_row(first_map, beatmap_id=11),
                            _map_row(second_map, beatmap_id=12),
                        ],
                        metadata_path=metadata_path,
                        duration_seconds=75.0,
                    )
                ],
            )

            report = build_timing_v3_labels(
                inventory_path=inventory_path,
                output_jsonl_path=root / "labels.jsonl",
                report_path=root / "report.json",
                pilot_output_path=root / "pilot.json",
                pilot_quotas={"stable": 1, "jump": 0, "tempo_change": 0, "long": 0, "anomaly": 0},
            )

            rows = _read_jsonl(root / "labels.jsonl")
            self.assertEqual(report["schema"], TIMING_V3_LABEL_REPORT_SCHEMA)
            self.assertEqual(rows[0]["schema"], TIMING_V3_LABEL_AUDIO_ROW_SCHEMA)
            self.assertEqual(rows[0]["label"]["stratum"], LABEL_STABLE)
            self.assertEqual(rows[0]["label"]["confidence"], "high")
            self.assertFalse(rows[0]["label"]["ambiguous"])
            self.assertEqual(rows[0]["representative_redline_grid"]["signature_audio_group_count"], 2)
            self.assertEqual(rows[0]["evidence_counts"]["cross_map_scored_count"], 1)
            self.assertEqual(rows[0]["evidence_counts"]["object_grid_supported_count"], 2)
            self.assertEqual(rows[0]["evidence_counts"]["cross_map_grid_supported_count"], 1)
            self.assertEqual(rows[0]["evidence_counts"]["cross_map_supported_count"], 0)
            self.assertIn("cross_map_alias_resolved=0/1", rows[0]["label"]["reasons"])
            self.assertEqual(rows[0]["metadata_bpm_evidence"]["alias_agreement"]["status"], "all_agree_alias_aware")
            self.assertFalse(rows[0]["metadata_bpm_evidence"]["used_for_label_inference"])
            self.assertNotIn("subdivision_evidence", json.dumps(rows[0]["maps"][0]["object_grid"]))
            self.assertEqual(report["strata"]["audio_counts"][LABEL_STABLE], 1)
            self.assertEqual(report["evidence"]["metadata_bpm_alias_agreement_count"], 3)

            pilot = json.loads((root / "pilot.json").read_text(encoding="utf-8"))
            self.assertEqual(pilot["schema"], TIMING_V3_LABEL_PILOT_SCHEMA)
            self.assertEqual(pilot["selected_counts"]["stable"], 1)
            self.assertEqual(pilot["selection"]["target_audio_count"], 1)
            self.assertEqual(pilot["selected_counts_by_quota_group"]["stable"], 1)

    def test_ramp_candidate_remains_ambiguous_audit_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            beatmapset = root / "dataset" / "0" / "1002"
            audio_path = beatmapset / "audio.wav"
            audio_path.parent.mkdir(parents=True)
            audio_path.write_bytes(b"audio")
            ramp_map = beatmapset / "ramp.osu"
            _write_osu(
                ramp_map,
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
            inventory_path = root / "inventory.jsonl"
            _write_jsonl(
                inventory_path,
                [
                    _inventory_row(
                        audio_path=audio_path,
                        maps=[_map_row(ramp_map, beatmap_id=21)],
                        metadata_path=None,
                        duration_seconds=20.0,
                    )
                ],
            )

            report = build_timing_v3_labels(
                inventory_path=inventory_path,
                output_jsonl_path=root / "labels.jsonl",
                report_path=root / "report.json",
            )

            row = _read_jsonl(root / "labels.jsonl")[0]
            self.assertEqual(row["label"]["stratum"], LABEL_RAMP_CANDIDATE)
            self.assertEqual(row["label"]["confidence"], "low")
            self.assertTrue(row["label"]["ambiguous"])
            self.assertTrue(row["label"]["audit_candidate"])
            self.assertIn("ramp_candidates_require_manual_confirmation", row["label"]["reasons"])
            self.assertEqual(report["strata"]["audio_counts"][LABEL_RAMP_CANDIDATE], 1)
            self.assertEqual(report["strata"]["ambiguous_audio_count"], 1)

    def test_missing_map_is_evidence_error_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            beatmapset = root / "dataset" / "0" / "1003"
            audio_path = beatmapset / "audio.wav"
            audio_path.parent.mkdir(parents=True)
            audio_path.write_bytes(b"audio")
            missing_map = beatmapset / "missing.osu"
            inventory_path = root / "inventory.jsonl"
            _write_jsonl(
                inventory_path,
                [
                    _inventory_row(
                        audio_path=audio_path,
                        maps=[_map_row(missing_map, beatmap_id=31, exists=False)],
                        metadata_path=None,
                        duration_seconds=45.0,
                        anomalies=["missing_beatmap_file"],
                    )
                ],
            )

            report = build_timing_v3_labels(
                inventory_path=inventory_path,
                output_jsonl_path=root / "labels.jsonl",
                report_path=root / "report.json",
            )

            row = _read_jsonl(root / "labels.jsonl")[0]
            self.assertEqual(row["label"]["stratum"], LABEL_AMBIGUOUS)
            self.assertTrue(row["label"]["ambiguous"])
            self.assertEqual(row["evidence_counts"]["map_error_count"], 1)
            self.assertEqual(row["maps"][0]["error_type"], "FileNotFoundError")
            self.assertEqual(report["evidence"]["evidence_error_audio_count"], 1)

    def test_pilot_selection_is_deterministic_audio_grouped_and_priority_ordered(self) -> None:
        rows = [
            _label_row("a", LABEL_STABLE, duration_seconds=100.0),
            _label_row("b", LABEL_RAMP_CANDIDATE, duration_seconds=700.0),
            _label_row("c", LABEL_AMBIGUOUS, duration_seconds=700.0),
            _label_row("d", LABEL_AMBIGUOUS, duration_seconds=100.0, anomalies=["missing_cache"]),
            _label_row("e", LABEL_STABLE, duration_seconds=100.0),
        ]

        quotas = {"stable": 1, "jump": 1, "tempo_change": 1, "long": 1, "anomaly": 1}
        first = select_timing_v3_pilot(rows, quotas=quotas, seed="fixed", long_threshold_seconds=600.0)
        second = select_timing_v3_pilot(rows, quotas=quotas, seed="fixed", long_threshold_seconds=600.0)

        self.assertEqual(first, second)
        self.assertEqual(first["selection"]["quotas"], quotas)
        self.assertEqual(first["selection"]["target_audio_count"], 5)
        self.assertEqual(first["selected_audio_count"], 4)
        self.assertEqual(first["selected_counts"]["stable"], 1)
        self.assertEqual(first["selected_counts"]["ramp"], 1)
        self.assertEqual(first["selected_counts"]["long"], 1)
        self.assertEqual(first["selected_counts"]["anomaly"], 1)
        self.assertEqual(first["selected_counts_by_quota_group"]["tempo_change"], 1)
        self.assertEqual(first["underfilled_counts_by_quota_group"]["jump"], 1)
        self.assertEqual(first["available_counts"]["ramp"], 1)
        self.assertEqual(first["available_counts"]["dense"], 0)
        selected_audio = [
            item["audio_group_key"]
            for stratum_rows in first["selected"].values()
            for item in stratum_rows
        ]
        self.assertEqual(len(selected_audio), len(set(selected_audio)))

    def test_default_pilot_quota_targets_eighty_audio_groups_with_combined_ramp_dense_bucket(self) -> None:
        pilot = select_timing_v3_pilot([], seed="fixed")

        self.assertEqual(
            pilot["selection"]["quotas"],
            {"stable": 20, "jump": 20, "tempo_change": 20, "long": 10, "anomaly": 10},
        )
        self.assertEqual(pilot["selection"]["target_audio_count"], 80)
        tempo_group = _quota_group(pilot, "tempo_change")
        self.assertEqual(tempo_group["output_strata"], ["ramp", "dense"])
        self.assertEqual(tempo_group["quota"], 20)
        self.assertEqual(tempo_group["available_count"], 0)
        self.assertEqual(tempo_group["selected_count"], 0)
        self.assertEqual(tempo_group["underfilled_count"], 20)
        self.assertEqual(tempo_group["strategy"], "ramp_reserve_then_dense_then_ramp_backfill")
        self.assertEqual(tempo_group["ramp_reserved_quota"], 0)

    def test_tempo_change_pilot_reserves_ramp_then_fills_remaining_with_dense(self) -> None:
        rows = [
            *[
                _label_row(f"ramp-{index:02d}", LABEL_RAMP_CANDIDATE, duration_seconds=100.0)
                for index in range(12)
            ],
            *[
                _label_row(f"dense-{index:02d}", LABEL_DENSE, duration_seconds=100.0)
                for index in range(30)
            ],
        ]

        pilot = select_timing_v3_pilot(
            rows,
            quotas={"stable": 0, "jump": 0, "tempo_change": 20, "long": 0, "anomaly": 0},
            seed="fixed",
        )

        tempo_group = _quota_group(pilot, "tempo_change")
        self.assertEqual(pilot["selected_audio_count"], 20)
        self.assertEqual(pilot["selected_counts"]["ramp"], 10)
        self.assertEqual(pilot["selected_counts"]["dense"], 10)
        self.assertEqual(pilot["selected_counts_by_quota_group"]["tempo_change"], 20)
        self.assertEqual(tempo_group["ramp_reserved_quota"], 10)
        self.assertEqual(
            tempo_group["selected_counts_by_output_stratum"],
            {"ramp": 10, "dense": 10},
        )
        self.assertEqual(tempo_group["underfilled_count"], 0)

    def test_tempo_change_pilot_backfills_when_ramp_or_dense_is_underfilled(self) -> None:
        ramp_short_rows = [
            *[
                _label_row(f"ramp-short-{index:02d}", LABEL_RAMP_CANDIDATE, duration_seconds=100.0)
                for index in range(3)
            ],
            *[
                _label_row(f"dense-fill-{index:02d}", LABEL_DENSE, duration_seconds=100.0)
                for index in range(30)
            ],
        ]
        ramp_short = select_timing_v3_pilot(
            ramp_short_rows,
            quotas={"stable": 0, "jump": 0, "tempo_change": 20, "long": 0, "anomaly": 0},
            seed="fixed",
        )
        self.assertEqual(ramp_short["selected_counts"]["ramp"], 3)
        self.assertEqual(ramp_short["selected_counts"]["dense"], 17)
        self.assertEqual(_quota_group(ramp_short, "tempo_change")["underfilled_count"], 0)

        dense_short_rows = [
            *[
                _label_row(f"ramp-fill-{index:02d}", LABEL_RAMP_CANDIDATE, duration_seconds=100.0)
                for index in range(30)
            ],
            *[
                _label_row(f"dense-short-{index:02d}", LABEL_DENSE, duration_seconds=100.0)
                for index in range(4)
            ],
        ]
        dense_short = select_timing_v3_pilot(
            dense_short_rows,
            quotas={"stable": 0, "jump": 0, "tempo_change": 20, "long": 0, "anomaly": 0},
            seed="fixed",
        )
        self.assertEqual(dense_short["selected_counts"]["ramp"], 16)
        self.assertEqual(dense_short["selected_counts"]["dense"], 4)
        self.assertEqual(_quota_group(dense_short, "tempo_change")["underfilled_count"], 0)

        both_short = select_timing_v3_pilot(
            [
                _label_row("ramp-only", LABEL_RAMP_CANDIDATE, duration_seconds=100.0),
                _label_row("dense-only", LABEL_DENSE, duration_seconds=100.0),
            ],
            quotas={"stable": 0, "jump": 0, "tempo_change": 20, "long": 0, "anomaly": 0},
            seed="fixed",
        )
        self.assertEqual(both_short["selected_counts_by_quota_group"]["tempo_change"], 2)
        self.assertEqual(both_short["underfilled_counts_by_quota_group"]["tempo_change"], 18)

    def test_cli_writes_explicit_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            beatmapset = root / "dataset" / "0" / "1004"
            audio_path = beatmapset / "audio.wav"
            audio_path.parent.mkdir(parents=True)
            audio_path.write_bytes(b"audio")
            osu_path = beatmapset / "map.osu"
            _write_osu(
                osu_path,
                timing_lines=["0,500,4,2,0,80,1,0"],
                hitobject_lines=["64,192,0,1,0,0:0:0:0:"],
            )
            inventory_path = root / "inventory.jsonl"
            _write_jsonl(
                inventory_path,
                [
                    _inventory_row(
                        audio_path=audio_path,
                        maps=[_map_row(osu_path, beatmap_id=41)],
                        metadata_path=None,
                        duration_seconds=30.0,
                    )
                ],
            )

            exit_code = main(
                [
                    "--inventory",
                    str(inventory_path),
                    "--output-jsonl",
                    str(root / "labels.jsonl"),
                    "--report-json",
                    str(root / "report.json"),
                    "--pilot-output-json",
                    str(root / "pilot.json"),
                    "--pilot-stable-quota",
                    "1",
                    "--pilot-jump-quota",
                    "0",
                    "--pilot-tempo-change-quota",
                    "0",
                    "--pilot-long-quota",
                    "0",
                    "--pilot-anomaly-quota",
                    "0",
                    "--json",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((root / "labels.jsonl").is_file())
            self.assertTrue((root / "report.json").is_file())
            self.assertTrue((root / "pilot.json").is_file())

    def test_dataset_root_containment_allows_inside_absolute_paths_and_rejects_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_root = root / "dataset"
            beatmapset = dataset_root / "0" / "1005"
            audio_path = beatmapset / "audio.wav"
            audio_path.parent.mkdir(parents=True)
            audio_path.write_bytes(b"audio")
            osu_path = beatmapset / "map.osu"
            _write_osu(
                osu_path,
                timing_lines=["0,500,4,2,0,80,1,0"],
                hitobject_lines=[_tap_line(index * 500.0) for index in range(9)],
            )
            metadata_path = beatmapset / "metadata.json"
            metadata_path.write_text('{"bpm":120,"beatmaps":[{"id":51,"bpm":120}]}\n', encoding="utf-8")
            inside_inventory = root / "inside.jsonl"
            _write_jsonl(
                inside_inventory,
                [
                    _inventory_row(
                        audio_path=audio_path,
                        maps=[_map_row(osu_path, beatmap_id=51)],
                        metadata_path=metadata_path,
                        duration_seconds=30.0,
                    )
                ],
            )

            build_timing_v3_labels(
                inventory_path=inside_inventory,
                output_jsonl_path=root / "inside_labels.jsonl",
                report_path=root / "inside_report.json",
                dataset_root=dataset_root,
            )

            outside_map = root / "outside.osu"
            _write_osu(
                outside_map,
                timing_lines=["0,500,4,2,0,80,1,0"],
                hitobject_lines=[_tap_line(index * 500.0) for index in range(9)],
            )
            absolute_escape_inventory = root / "absolute_escape.jsonl"
            _write_jsonl(
                absolute_escape_inventory,
                [
                    _inventory_row(
                        audio_path=audio_path,
                        maps=[_map_row(outside_map, beatmap_id=52)],
                        metadata_path=metadata_path,
                        duration_seconds=30.0,
                    )
                ],
            )
            with self.assertRaisesRegex(ValueError, "escapes dataset_root"):
                build_timing_v3_labels(
                    inventory_path=absolute_escape_inventory,
                    output_jsonl_path=root / "absolute_escape_labels.jsonl",
                    report_path=root / "absolute_escape_report.json",
                    dataset_root=dataset_root,
                )

            relative_escape_inventory = root / "relative_escape.jsonl"
            row = _inventory_row(
                audio_path=audio_path,
                maps=[_map_row(osu_path, beatmap_id=53)],
                metadata_path=metadata_path,
                duration_seconds=30.0,
            )
            row["maps"][0].pop("resolved_beatmap_path")
            row["maps"][0]["beatmap_path"] = "../outside.osu"
            _write_jsonl(relative_escape_inventory, [row])
            with self.assertRaisesRegex(ValueError, "path must be relative"):
                build_timing_v3_labels(
                    inventory_path=relative_escape_inventory,
                    output_jsonl_path=root / "relative_escape_labels.jsonl",
                    report_path=root / "relative_escape_report.json",
                    dataset_root=dataset_root,
                )

            metadata_escape_inventory = root / "metadata_escape.jsonl"
            outside_metadata = root / "metadata.json"
            outside_metadata.write_text('{"bpm":120}\n', encoding="utf-8")
            _write_jsonl(
                metadata_escape_inventory,
                [
                    _inventory_row(
                        audio_path=audio_path,
                        maps=[_map_row(osu_path, beatmap_id=54)],
                        metadata_path=outside_metadata,
                        duration_seconds=30.0,
                    )
                ],
            )
            with self.assertRaisesRegex(ValueError, "escapes dataset_root"):
                build_timing_v3_labels(
                    inventory_path=metadata_escape_inventory,
                    output_jsonl_path=root / "metadata_escape_labels.jsonl",
                    report_path=root / "metadata_escape_report.json",
                    dataset_root=dataset_root,
                )

    def test_resume_reuses_matching_fingerprint_and_keeps_final_audio_key_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            first = _stable_audio_case(root, beatmapset_id=1006, beatmap_id=61)
            second = _stable_audio_case(root, beatmapset_id=1007, beatmap_id=71)
            inventory_path = root / "inventory.jsonl"
            _write_jsonl(
                inventory_path,
                [
                    second["inventory_row"],
                    first["inventory_row"],
                ],
            )
            output_path = root / "labels.jsonl"

            first_report = build_timing_v3_labels(
                inventory_path=inventory_path,
                output_jsonl_path=output_path,
                report_path=root / "first_report.json",
                limit=1,
                progress_every=0,
                checkpoint_every=0,
            )
            self.assertEqual(first_report["run"]["processed_count"], 1)
            self.assertEqual(len(_read_jsonl(output_path)), 1)

            second_report = build_timing_v3_labels(
                inventory_path=inventory_path,
                output_jsonl_path=output_path,
                report_path=root / "second_report.json",
                progress_every=0,
                checkpoint_every=0,
            )

            rows = _read_jsonl(output_path)
            self.assertEqual(second_report["run"]["resumed_count"], 1)
            self.assertEqual(second_report["run"]["processed_count"], 1)
            self.assertEqual(second_report["run"]["stale_count"], 0)
            self.assertEqual([row["audio_group_key"] for row in rows], sorted(row["audio_group_key"] for row in rows))
            self.assertEqual(len(rows), 2)
            self.assertIn("started_at_unix", second_report["run"])
            self.assertIn("finished_at_unix", second_report["run"])
            self.assertGreaterEqual(second_report["run"]["total_seconds"], 0.0)
            self.assertGreaterEqual(second_report["run"]["finished_at_unix"], second_report["run"]["started_at_unix"])

    def test_stale_fingerprint_context_is_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            case = _stable_audio_case(root, beatmapset_id=1008, beatmap_id=81)
            inventory_path = root / "inventory.jsonl"
            _write_jsonl(inventory_path, [case["inventory_row"]])
            output_path = root / "labels.jsonl"

            build_timing_v3_labels(
                inventory_path=inventory_path,
                output_jsonl_path=output_path,
                report_path=root / "first_report.json",
                expected_key_count=4,
                progress_every=0,
                checkpoint_every=0,
            )
            first_fingerprint = _read_jsonl(output_path)[0]["fingerprint"]["sha256"]

            report = build_timing_v3_labels(
                inventory_path=inventory_path,
                output_jsonl_path=output_path,
                report_path=root / "second_report.json",
                expected_key_count=None,
                progress_every=0,
                checkpoint_every=0,
            )
            second_fingerprint = _read_jsonl(output_path)[0]["fingerprint"]["sha256"]

            self.assertEqual(report["run"]["resumed_count"], 0)
            self.assertEqual(report["run"]["stale_count"], 1)
            self.assertEqual(report["run"]["processed_count"], 1)
            self.assertNotEqual(first_fingerprint, second_fingerprint)

    def test_checkpoint_writes_partial_jsonl_with_bounded_frequency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            first = _stable_audio_case(root, beatmapset_id=1009, beatmap_id=91)
            second = _stable_audio_case(root, beatmapset_id=1010, beatmap_id=101)
            inventory_path = root / "inventory.jsonl"
            _write_jsonl(inventory_path, [first["inventory_row"], second["inventory_row"]])
            output_path = root / "labels.jsonl"
            writes: list[int] = []
            original_write = labels_module._write_jsonl_atomic

            def _record_write(path: Path, rows: list[dict[str, object]]) -> None:
                if Path(path) == output_path:
                    writes.append(len(rows))
                original_write(path, rows)

            with mock.patch.object(labels_module, "_write_jsonl_atomic", side_effect=_record_write):
                report = build_timing_v3_labels(
                    inventory_path=inventory_path,
                    output_jsonl_path=output_path,
                    report_path=root / "report.json",
                    progress_every=0,
                    checkpoint_every=1,
                )

            self.assertEqual(report["run"]["checkpoint_every"], 1)
            self.assertEqual(report["run"]["checkpoint_write_count"], 2)
            self.assertEqual(writes, [1, 2, 2])

    def test_progress_reports_processed_total_and_last_key_to_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            first = _stable_audio_case(root, beatmapset_id=1011, beatmap_id=111)
            second = _stable_audio_case(root, beatmapset_id=1012, beatmap_id=121)
            inventory_path = root / "inventory.jsonl"
            _write_jsonl(inventory_path, [first["inventory_row"], second["inventory_row"]])
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                build_timing_v3_labels(
                    inventory_path=inventory_path,
                    output_jsonl_path=root / "labels.jsonl",
                    report_path=root / "report.json",
                    progress_every=1,
                    checkpoint_every=0,
                )

            progress = stderr.getvalue()
            self.assertIn("processed=1/2", progress)
            self.assertIn("processed=2/2", progress)
            self.assertIn("last_key=", progress)

    def test_pilot_rows_output_jsonl_writes_full_selected_label_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            first = _stable_audio_case(root, beatmapset_id=1013, beatmap_id=131)
            second = _stable_audio_case(root, beatmapset_id=1014, beatmap_id=141)
            inventory_path = root / "inventory.jsonl"
            _write_jsonl(inventory_path, [second["inventory_row"], first["inventory_row"]])
            pilot_rows_path = root / "pilot_rows.jsonl"

            report = build_timing_v3_labels(
                inventory_path=inventory_path,
                output_jsonl_path=root / "labels.jsonl",
                report_path=root / "report.json",
                pilot_rows_output_path=pilot_rows_path,
                pilot_quotas={"stable": 2, "jump": 0, "tempo_change": 0, "long": 0, "anomaly": 0},
                progress_every=0,
                checkpoint_every=0,
            )

            pilot_rows = _read_jsonl(pilot_rows_path)
            self.assertEqual(report["output"]["pilot_rows_count"], 2)
            self.assertEqual(report["output"]["pilot_rows_output_path"], pilot_rows_path.as_posix())
            self.assertIsNotNone(report["output"]["pilot_rows_output_sha256"])
            self.assertEqual([row["audio_group_key"] for row in pilot_rows], sorted(row["audio_group_key"] for row in pilot_rows))
            self.assertEqual({row["pilot_stratum"] for row in pilot_rows}, {"stable"})
            self.assertEqual({row["pilot_quota_group"] for row in pilot_rows}, {"stable"})
            self.assertIn("maps", pilot_rows[0])
            self.assertIn("resolved_audio_path", pilot_rows[0])
            self.assertIn("fingerprint", pilot_rows[0])

    def test_pilot_rows_output_jsonl_preserves_tempo_change_strata_annotations(self) -> None:
        rows = [
            _label_row("ramp-a", LABEL_RAMP_CANDIDATE, duration_seconds=100.0),
            _label_row("dense-a", LABEL_DENSE, duration_seconds=100.0),
        ]
        pilot = select_timing_v3_pilot(
            rows,
            quotas={"stable": 0, "jump": 0, "tempo_change": 2, "long": 0, "anomaly": 0},
            seed="fixed",
        )

        pilot_rows = labels_module._pilot_full_label_rows(rows, pilot)

        annotations = {
            row["audio_group_key"]: (row["pilot_stratum"], row["pilot_quota_group"])
            for row in pilot_rows
        }
        self.assertEqual(annotations["ramp-a"], ("ramp", "tempo_change"))
        self.assertEqual(annotations["dense-a"], ("dense", "tempo_change"))
        self.assertNotIn("pilot_stratum", rows[0])


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


def _tap_line(time_ms: float) -> str:
    return f"64,192,{time_ms:g},1,0,0:0:0:0:"


def _stable_audio_case(root: Path, *, beatmapset_id: int, beatmap_id: int) -> dict[str, object]:
    beatmapset = root / "dataset" / "0" / str(beatmapset_id)
    audio_path = beatmapset / "audio.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"audio")
    osu_path = beatmapset / "map.osu"
    _write_osu(
        osu_path,
        timing_lines=["0,500,4,2,0,80,1,0"],
        hitobject_lines=[_tap_line(index * 500.0) for index in range(9)],
    )
    return {
        "audio_path": audio_path,
        "osu_path": osu_path,
        "inventory_row": _inventory_row(
            audio_path=audio_path,
            maps=[_map_row(osu_path, beatmap_id=beatmap_id)],
            metadata_path=None,
            duration_seconds=30.0,
        ),
    }


def _inventory_row(
    *,
    audio_path: Path,
    maps: list[dict[str, object]],
    metadata_path: Path | None,
    duration_seconds: float,
    anomalies: list[str] | None = None,
) -> dict[str, object]:
    return {
        "schema": TIMING_V3_INVENTORY_AUDIO_ROW_SCHEMA,
        "audio_group_index": 0,
        "audio_group_key": audio_path.resolve().as_posix(),
        "resolved_audio_path": audio_path.resolve().as_posix(),
        "audio_exists": True,
        "map_count": len(maps),
        "maps": maps,
        "metadata_json": {
            "paths": (
                []
                if metadata_path is None
                else [
                    {
                        "path": metadata_path.resolve().as_posix(),
                        "exists": metadata_path.is_file(),
                        "sha256": "synthetic",
                    }
                ]
            ),
            "path_count": 0 if metadata_path is None else 1,
            "existing_count": 0 if metadata_path is None else int(metadata_path.is_file()),
        },
        "cache": {
            "status": "valid",
            "audio_cache_key": f"cache::{audio_path.name}",
            "config_fingerprint": "synthetic",
            "duration_seconds": duration_seconds,
            "frame_count": int(duration_seconds * 50),
        },
        "anomalies": anomalies or [],
    }


def _map_row(path: Path, *, beatmap_id: int, exists: bool = True) -> dict[str, object]:
    return {
        "source_row_index": beatmap_id,
        "shard": "0",
        "audio_path": "audio.wav",
        "beatmap_path": path.name,
        "resolved_beatmap_path": path.resolve().as_posix(),
        "beatmap_exists": exists,
        "metadata_json_path": (path.parent / "metadata.json").resolve().as_posix(),
        "metadata_json_exists": (path.parent / "metadata.json").is_file(),
        "beatmap_id": beatmap_id,
        "index_row": {},
    }


def _label_row(
    audio_key: str,
    stratum: str,
    *,
    duration_seconds: float,
    anomalies: list[str] | None = None,
) -> dict[str, object]:
    return {
        "audio_group_key": audio_key,
        "resolved_audio_path": f"/tmp/{audio_key}.wav",
        "map_count": 1,
        "source": {"cache_duration_seconds": duration_seconds},
        "label": {
            "stratum": stratum,
            "confidence": "low",
            "ambiguous": stratum != LABEL_STABLE,
            "audit_candidate": stratum != LABEL_STABLE,
            "reasons": [],
        },
        "evidence_counts": {"map_error_count": 0 if not anomalies else 1},
        "inventory_anomalies": anomalies or [],
    }


def _quota_group(pilot: dict[str, object], name: str) -> dict[str, object]:
    for group in pilot["quota_groups"]:
        if group["name"] == name:
            return group
    raise AssertionError(f"missing quota group {name!r}")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    unittest.main()
