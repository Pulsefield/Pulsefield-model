from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from pulsefield_model.timing.evaluation import exp004_protocol
from pulsefield_model.timing.evaluation.exp004_splits import (
    EXP004_BROAD_SEED,
    EXP004_DEFICIT_PRIORITY,
    EXP004_HOLDOUT_DEFICIT_SEED,
    EXP004_HOLDOUT_AUDIO_COUNT,
    EXP004_HOLDOUT_QUOTAS,
    EXP004_HOLDOUT_SEED,
    EXP004_MINIMUM_REQUIRED_EXCLUSION_KEYS,
    EXP004_PRIORITY,
    TIMING_V3_EXP004_BROAD_MANIFEST_SCHEMA,
    TIMING_V3_EXP004_EXPOSURE_MANIFEST_SCHEMA,
    TIMING_V3_EXP004_HOLDOUT_MANIFEST_SCHEMA,
    build_exp004_broad500,
    build_exp004_execution_inputs_from_selected_keys,
    build_exp004_execution_inputs_from_split_manifest,
    build_exp004_full5050_execution_inputs,
    build_exp004_exposure_manifest,
    build_exp004_holdout100,
    load_exp004_exposure_manifest,
    load_exp004_manifest,
    main,
    materialize_label_subset,
    select_exp004_broad500,
    select_exp004_holdout100,
    validate_exp004_exposure_manifest,
    validate_exp004_manifest,
)
from pulsefield_model.timing.evaluation.labels import (
    LABEL_AMBIGUOUS,
    LABEL_DENSE,
    LABEL_JUMP_CANDIDATE,
    LABEL_RAMP_CANDIDATE,
    LABEL_STABLE,
    TIMING_V3_LABEL_AUDIO_ROW_SCHEMA,
)


class TimingV3Experiment004SplitTests(unittest.TestCase):
    def test_exposure_manifest_schema_fails_closed(self) -> None:
        rows, pilot_keys, protocol_keys, exp003_keys = _corpus()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            exposure, exposure_path, _labels_path = _build_exposure(
                root,
                rows,
                pilot_keys=pilot_keys,
                protocol_keys=protocol_keys,
                exp003_keys=exp003_keys,
            )

            keys = validate_exp004_exposure_manifest(exposure)
            self.assertEqual(len(keys), EXP004_MINIMUM_REQUIRED_EXCLUSION_KEYS)
            self.assertEqual(exposure["schema_id"], TIMING_V3_EXP004_EXPOSURE_MANIFEST_SCHEMA)
            self.assertEqual(
                TIMING_V3_EXP004_EXPOSURE_MANIFEST_SCHEMA,
                "pulsefield_model.timing_v3_exp004_oracle_exposure_exclusion_manifest_v1",
            )
            self.assertEqual(
                [entry["cache_audio_key"] for entry in exposure["entries"]],
                sorted(keys),
            )
            self.assertEqual(len(exposure["entries_sha256"]), 64)
            self.assertEqual(load_exp004_exposure_manifest(exposure_path), exposure)

            with self.assertRaises(FileNotFoundError):
                load_exp004_exposure_manifest(root / "missing-exposure.json")

            missing = json.loads(json.dumps(exposure))
            missing.pop("entries")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                validate_exp004_exposure_manifest(missing)

            stale_source = json.loads(json.dumps(exposure))
            stale_source["sources"]["pilot80"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "exposure_scan_source_sha256.*stale"):
                validate_exp004_exposure_manifest(stale_source)

            stale_entries = json.loads(json.dumps(exposure))
            stale_entries["entries_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "entries_sha256"):
                validate_exp004_exposure_manifest(stale_entries)

            duplicate = json.loads(json.dumps(exposure))
            duplicate["entries"].append(dict(duplicate["entries"][0]))
            duplicate["entry_count"] += 1
            with self.assertRaisesRegex(ValueError, "duplicate exposure cache_audio_key"):
                validate_exp004_exposure_manifest(duplicate)

            with_metric = json.loads(json.dumps(exposure))
            with_metric["entries"][0]["rmse_ms"] = 1.25
            with self.assertRaisesRegex(ValueError, "metric-valued exposure field"):
                validate_exp004_exposure_manifest(with_metric)

            by_key = {str(row["source"]["cache_audio_key"]): row for row in rows}
            overlap_protocol_path = _write_jsonl(
                root / "protocol_overlap.jsonl",
                [by_key[key] for key in sorted(pilot_keys)[:3]],
            )
            with self.assertRaisesRegex(ValueError, "overlap by cache audio key"):
                build_exp004_exposure_manifest(
                    pilot_manifest_path=root / "pilot.jsonl",
                    protocol_manifest_path=overlap_protocol_path,
                    exp003_holdout_manifest_path=root / "exp003_holdout.json",
                    labels_jsonl_path=root / "labels.jsonl",
                    generated_from_commit="0421de8a",
                    generated_at_utc="2026-08-11T00:00:00Z",
                    manifest_output_path=root / "overlap_exposure.json",
                )

            protocol_rows = [by_key[key] for key in sorted(protocol_keys)]
            mismatched_cache_rows = json.loads(json.dumps(protocol_rows))
            mismatched_cache_rows[0]["cache_audio_key"] = "cache::wrong-top-level"
            mismatched_cache_path = _write_jsonl(
                root / "protocol_mismatched_cache.jsonl",
                mismatched_cache_rows,
            )
            with self.assertRaisesRegex(ValueError, "cache identity fields disagree"):
                build_exp004_exposure_manifest(
                    pilot_manifest_path=root / "pilot.jsonl",
                    protocol_manifest_path=mismatched_cache_path,
                    exp003_holdout_manifest_path=root / "exp003_holdout.json",
                    labels_jsonl_path=root / "labels.jsonl",
                    generated_from_commit="0421de8a",
                    generated_at_utc="2026-08-11T00:00:00Z",
                    manifest_output_path=root / "mismatched_cache_exposure.json",
                )

            mismatched_audio_group_rows = [
                {"cache_audio_key": key}
                for key in sorted(protocol_keys)
            ]
            mismatched_audio_group_rows[0]["audio_group_key"] = "audio/pilot-000"
            mismatched_audio_group_path = _write_jsonl(
                root / "protocol_mismatched_audio_group.jsonl",
                mismatched_audio_group_rows,
            )
            with self.assertRaisesRegex(ValueError, "cache identity fields disagree"):
                build_exp004_exposure_manifest(
                    pilot_manifest_path=root / "pilot.jsonl",
                    protocol_manifest_path=mismatched_audio_group_path,
                    exp003_holdout_manifest_path=root / "exp003_holdout.json",
                    labels_jsonl_path=root / "labels.jsonl",
                    generated_from_commit="0421de8a",
                    generated_at_utc="2026-08-11T00:00:00Z",
                    manifest_output_path=root / "mismatched_audio_group_exposure.json",
                )

    def test_holdout_is_exposure_disjoint_deterministic_and_hash_replayed(self) -> None:
        rows, pilot_keys, protocol_keys, exp003_keys = _corpus()
        with tempfile.TemporaryDirectory() as tmp_dir:
            exposure, _exposure_path, _labels_path = _build_exposure(
                Path(tmp_dir),
                rows,
                pilot_keys=pilot_keys,
                protocol_keys=protocol_keys,
                exp003_keys=exp003_keys,
            )

        first = select_exp004_holdout100(rows, exposure_manifest=exposure)
        second = select_exp004_holdout100(list(reversed(rows)), exposure_manifest=exposure)

        self.assertEqual(first, second)
        self.assertEqual(first["schema"], TIMING_V3_EXP004_HOLDOUT_MANIFEST_SCHEMA)
        self.assertEqual(first["seed"], EXP004_HOLDOUT_SEED)
        self.assertEqual(first["deficit_seed"], EXP004_HOLDOUT_DEFICIT_SEED)
        self.assertEqual(EXP004_HOLDOUT_SEED, "timing-v3-exp004-holdout100-v1")
        self.assertEqual(
            EXP004_HOLDOUT_DEFICIT_SEED,
            "timing-v3-exp004-holdout100-deficit-v1",
        )
        self.assertEqual(first["selection"]["priority_order"], list(EXP004_PRIORITY))
        self.assertEqual(first["selection"]["deficit_fill"]["priority_order"], list(EXP004_DEFICIT_PRIORITY))
        self.assertEqual(first["selection"]["quotas"], EXP004_HOLDOUT_QUOTAS)
        self.assertEqual(first["selection"]["degraded_underfilled"], {})
        self.assertEqual(first["selected_audio_count"], 100)

        exposure_keys = validate_exp004_exposure_manifest(exposure)
        selected_keys = [entry["cache_audio_key"] for entry in first["selected"]]
        self.assertEqual(len(selected_keys), len(set(selected_keys)))
        self.assertTrue(exposure_keys.isdisjoint(selected_keys))
        self.assertEqual(first["selected_counts"]["deficit_fill"], 0)
        for quota, expected_count in EXP004_HOLDOUT_QUOTAS.items():
            self.assertEqual(first["selected_counts"][quota], expected_count)
        for entry in first["selected"]:
            expected = hashlib.sha256(
                f"{EXP004_HOLDOUT_SEED}\0{entry['cache_audio_key']}".encode("utf-8")
            ).hexdigest()
            self.assertEqual(entry["selection_hash_sha256"], expected)
        validate_exp004_manifest(first)

        stale_exposure_source = json.loads(json.dumps(first))
        stale_exposure_source["source"]["exposure_manifest"]["entries_sha256"] = "0" * 64
        _refingerprint(stale_exposure_source)
        with self.assertRaisesRegex(ValueError, "source.exposure_manifest.entries_sha256"):
            validate_exp004_manifest(stale_exposure_source)

        stale_exposure_summary = json.loads(json.dumps(first))
        stale_exposure_summary["exposure"]["cache_audio_keys_sha256"] = "0" * 64
        _refingerprint(stale_exposure_summary)
        with self.assertRaisesRegex(ValueError, "cache-audio-key hash"):
            validate_exp004_manifest(stale_exposure_summary)

    def test_rare_quota_degradation_uses_preregistered_deficit_fill(self) -> None:
        rows, pilot_keys, protocol_keys, exp003_keys = _corpus(
            ramp_unexposed_count=4,
            extra_jump_count=1,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            exposure, _exposure_path, _labels_path = _build_exposure(
                Path(tmp_dir),
                rows,
                pilot_keys=pilot_keys,
                protocol_keys=protocol_keys,
                exp003_keys=exp003_keys,
            )

        holdout = select_exp004_holdout100(rows, exposure_manifest=exposure)

        self.assertEqual(
            holdout["selection"]["degraded_underfilled"],
            {"ramp_audit": {"requested": 5, "available": 4, "deficit": 1}},
        )
        self.assertEqual(holdout["selection"]["deficit_fill"]["fill_count"], 1)
        self.assertEqual(holdout["selected_counts"]["ramp_audit"], 4)
        self.assertEqual(holdout["selected_counts"]["deficit_fill"], 1)
        deficit_entries = [
            entry
            for entry in holdout["selected"]
            if entry["selection_substage"] == "deficit_fill"
        ]
        self.assertEqual(len(deficit_entries), 1)
        self.assertEqual(deficit_entries[0]["quota_assignment"], "jump")
        expected_hash = hashlib.sha256(
            f"{EXP004_HOLDOUT_DEFICIT_SEED}\0{deficit_entries[0]['cache_audio_key']}".encode("utf-8")
        ).hexdigest()
        self.assertEqual(deficit_entries[0]["selection_hash_sha256"], expected_hash)
        validate_exp004_manifest(holdout)

    def test_builders_and_cli_create_exposure_and_holdout_identity_artifacts(self) -> None:
        rows, pilot_keys, protocol_keys, exp003_keys = _corpus()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            labels_path = _write_jsonl(root / "labels.jsonl", rows)
            pilot_path, protocol_path, exp003_path = _write_required_sources(
                root,
                rows,
                pilot_keys=pilot_keys,
                protocol_keys=protocol_keys,
                exp003_keys=exp003_keys,
            )
            exposure_path = root / "exposure.json"
            holdout_path = root / "holdout.json"
            holdout_rows_path = root / "holdout_rows.jsonl"

            exposure_exit = main(
                [
                    "exposure",
                    "--pilot-manifest",
                    str(pilot_path),
                    "--protocol-manifest",
                    str(protocol_path),
                    "--exp003-holdout-manifest",
                    str(exp003_path),
                    "--labels-jsonl",
                    str(labels_path),
                    "--generated-from-commit",
                    "0421de8a",
                    "--generated-at-utc",
                    "2026-08-11T00:00:00Z",
                    "--manifest-output",
                    str(exposure_path),
                ]
            )
            self.assertEqual(exposure_exit, 0)

            exact_replay = build_exp004_exposure_manifest(
                pilot_manifest_path=pilot_path,
                protocol_manifest_path=protocol_path,
                exp003_holdout_manifest_path=exp003_path,
                labels_jsonl_path=labels_path,
                generated_from_commit="0421de8a",
                generated_at_utc="2026-08-11T00:00:00Z",
                manifest_output_path=exposure_path,
            )
            self.assertEqual(exact_replay, load_exp004_exposure_manifest(exposure_path))
            with self.assertRaisesRegex(ValueError, "immutable output already exists"):
                build_exp004_exposure_manifest(
                    pilot_manifest_path=pilot_path,
                    protocol_manifest_path=protocol_path,
                    exp003_holdout_manifest_path=exp003_path,
                    labels_jsonl_path=labels_path,
                    generated_from_commit="0421de8a",
                    generated_at_utc="2026-08-11T00:00:01Z",
                    manifest_output_path=exposure_path,
                )

            stale_tmp_path = holdout_path.with_name(holdout_path.name + ".tmp")
            stale_tmp_path.write_text("occupied", encoding="utf-8")
            holdout_exit = main(
                [
                    "holdout100",
                    "--labels-jsonl",
                    str(labels_path),
                    "--exposure-manifest",
                    str(exposure_path),
                    "--manifest-output",
                    str(holdout_path),
                    "--label-rows-output",
                    str(holdout_rows_path),
                ]
            )
            self.assertEqual(holdout_exit, 0)
            self.assertEqual(stale_tmp_path.read_text(encoding="utf-8"), "occupied")
            exact_holdout_replay = build_exp004_holdout100(
                labels_jsonl_path=labels_path,
                exposure_manifest_path=exposure_path,
                manifest_output_path=holdout_path,
                label_rows_output_path=holdout_rows_path,
            )
            self.assertEqual(exact_holdout_replay, load_exp004_manifest(holdout_path))

            exposure = load_exp004_exposure_manifest(exposure_path)
            holdout = load_exp004_manifest(holdout_path)
            materialized = _read_jsonl(holdout_rows_path)
            self.assertEqual(len(materialized), 100)
            self.assertEqual(
                [row["cache_audio_key"] for row in materialized],
                [entry["cache_audio_key"] for entry in holdout["selected"]],
            )
            self.assertEqual(holdout["source"]["labels"]["sha256"], _sha256(labels_path))
            self.assertEqual(
                holdout["source"]["exposure_manifest"]["sha256"],
                _sha256(exposure_path),
            )
            self.assertEqual(
                holdout["source"]["exposure_manifest"]["exposure_scan_source_sha256"],
                exposure["exposure_scan_source_sha256"],
            )
            self.assertEqual(
                holdout["source"]["exposure_manifest"]["entries_sha256"],
                exposure["entries_sha256"],
            )
            self.assertEqual(
                holdout["source"]["exposure_manifest"]["cache_audio_keys_sha256"],
                holdout["exposure"]["cache_audio_keys_sha256"],
            )
            self.assertEqual(len(materialize_label_subset(rows, holdout)), 100)

            with self.assertRaisesRegex(ValueError, "paths must be distinct"):
                build_exp004_holdout100(
                    labels_jsonl_path=labels_path,
                    exposure_manifest_path=exposure_path,
                    manifest_output_path=exposure_path,
                )

            broad_path = root / "broad_cli.json"
            broad_rows_path = root / "broad_cli_rows.jsonl"
            broad_exit = main(
                [
                    "broad500",
                    "--labels-jsonl",
                    str(labels_path),
                    "--exposure-manifest",
                    str(exposure_path),
                    "--holdout-manifest",
                    str(holdout_path),
                    "--manifest-output",
                    str(broad_path),
                    "--label-rows-output",
                    str(broad_rows_path),
                ]
            )
            self.assertEqual(broad_exit, 0)
            broad = load_exp004_manifest(broad_path)
            self.assertEqual(broad["selected_audio_count"], 500)
            self.assertEqual(len(_read_jsonl(broad_rows_path)), 500)
            self.assertEqual(
                broad["source"]["holdout_manifest"]["manifest_fingerprint_sha256"],
                holdout["manifest_fingerprint_sha256"],
            )
            with self.assertRaisesRegex(ValueError, "paths must be distinct"):
                build_exp004_broad500(
                    labels_jsonl_path=labels_path,
                    exposure_manifest_path=exposure_path,
                    holdout_manifest_path=holdout_path,
                    manifest_output_path=holdout_path,
                )

    def test_broad_builder_records_degraded_underfilled_stage(self) -> None:
        rows, pilot_keys, protocol_keys, exp003_keys = _corpus(broad_extra_count=7)
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            exposure, exposure_path, labels_path = _build_exposure(
                root,
                rows,
                pilot_keys=pilot_keys,
                protocol_keys=protocol_keys,
                exp003_keys=exp003_keys,
            )
            holdout_path = root / "holdout.json"
            holdout = build_exp004_holdout100(
                labels_jsonl_path=labels_path,
                exposure_manifest_path=exposure_path,
                manifest_output_path=holdout_path,
            )
            broad_path = root / "broad_degraded.json"
            broad = build_exp004_broad500(
                labels_jsonl_path=labels_path,
                exposure_manifest_path=exposure_path,
                holdout_manifest_path=holdout_path,
                manifest_output_path=broad_path,
            )

            self.assertEqual(broad["selected_audio_count"], 107)
            self.assertEqual(broad["selected"][:100], holdout["selected"])
            self.assertEqual(
                broad["selection"]["degraded_underfilled"],
                {"broad500_added": {"requested": 400, "available": 7, "deficit": 393}},
            )
            self.assertEqual(broad["selected_counts"], {"holdout100": 100, "broad500_added": 7})
            self.assertEqual(load_exp004_manifest(broad_path), broad)
            exposure_keys = validate_exp004_exposure_manifest(exposure)
            selected_keys = {entry["cache_audio_key"] for entry in broad["selected"]}
            self.assertTrue(exposure_keys.isdisjoint(selected_keys))

    def test_broad500_replays_holdout_and_stores_source_hashes(self) -> None:
        rows, pilot_keys, protocol_keys, exp003_keys = _corpus()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            exposure, exposure_path, labels_path = _build_exposure(
                root,
                rows,
                pilot_keys=pilot_keys,
                protocol_keys=protocol_keys,
                exp003_keys=exp003_keys,
            )
            holdout_path = root / "holdout.json"
            holdout = build_exp004_holdout100(
                labels_jsonl_path=labels_path,
                exposure_manifest_path=exposure_path,
                manifest_output_path=holdout_path,
            )
            broad_path = root / "broad.json"
            broad_rows_path = root / "broad_rows.jsonl"
            broad = build_exp004_broad500(
                labels_jsonl_path=labels_path,
                exposure_manifest_path=exposure_path,
                holdout_manifest_path=holdout_path,
                manifest_output_path=broad_path,
                label_rows_output_path=broad_rows_path,
            )
            replay = select_exp004_broad500(
                list(reversed(rows)),
                exposure_manifest=exposure,
                holdout_manifest=holdout,
                source=broad["source"],
            )

            self.assertEqual(broad, replay)
            self.assertEqual(load_exp004_manifest(broad_path), broad)
            self.assertEqual(broad["schema"], TIMING_V3_EXP004_BROAD_MANIFEST_SCHEMA)
            self.assertEqual(EXP004_BROAD_SEED, "timing-v3-exp004-broad500-v1")
            self.assertEqual(broad["selected_audio_count"], 500)
            self.assertEqual(len(_read_jsonl(broad_rows_path)), 500)
            self.assertEqual(broad["selected"][:100], holdout["selected"])
            added = broad["selected"][100:]
            self.assertEqual(len(added), 400)
            self.assertTrue(all(entry["stage"] == "broad500_added" for entry in added))
            self.assertEqual(
                [entry["selection_hash_sha256"] for entry in added],
                sorted(entry["selection_hash_sha256"] for entry in added),
            )
            expected_hashes = [
                hashlib.sha256(
                    f"{EXP004_BROAD_SEED}\0{entry['cache_audio_key']}".encode("utf-8")
                ).hexdigest()
                for entry in added
            ]
            self.assertEqual([entry["selection_hash_sha256"] for entry in added], expected_hashes)
            exposure_keys = validate_exp004_exposure_manifest(exposure)
            selected_keys = {entry["cache_audio_key"] for entry in broad["selected"]}
            holdout_keys = {entry["cache_audio_key"] for entry in holdout["selected"]}
            self.assertTrue(exposure_keys.isdisjoint(selected_keys))
            self.assertTrue(holdout_keys.isdisjoint(entry["cache_audio_key"] for entry in added))
            self.assertEqual(broad["source"]["holdout_manifest"]["sha256"], _sha256(holdout_path))
            self.assertEqual(
                broad["source"]["holdout_manifest"]["selected_ordered_sha256"],
                broad["holdout"]["selected_ordered_sha256"],
            )
            validate_exp004_manifest(broad)

            swapped_prefix = json.loads(json.dumps(broad))
            swapped_prefix["selected"][0], swapped_prefix["selected"][1] = (
                swapped_prefix["selected"][1],
                swapped_prefix["selected"][0],
            )
            _refingerprint(swapped_prefix)
            swapped_path = root / "swapped_broad.json"
            swapped_path.write_text(
                json.dumps(swapped_prefix, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "ordered holdout prefix"):
                load_exp004_manifest(swapped_path)

            drifted_source = json.loads(json.dumps(broad["source"]))
            drifted_source["labels"]["path"] = "/copied/labels.jsonl"
            with self.assertRaisesRegex(ValueError, "provenance does not exactly match"):
                select_exp004_broad500(
                    rows,
                    exposure_manifest=exposure,
                    holdout_manifest=holdout,
                    source=drifted_source,
                )

    def test_broad_rejects_internally_valid_nonlowest_holdout_replay(self) -> None:
        rows, pilot_keys, protocol_keys, exp003_keys = _corpus()
        with tempfile.TemporaryDirectory() as tmp_dir:
            exposure, _exposure_path, _labels_path = _build_exposure(
                Path(tmp_dir),
                rows,
                pilot_keys=pilot_keys,
                protocol_keys=protocol_keys,
                exp003_keys=exp003_keys,
            )
        holdout = select_exp004_holdout100(rows, exposure_manifest=exposure)
        tampered = json.loads(json.dumps(holdout))
        exposure_keys = validate_exp004_exposure_manifest(exposure)
        selected_keys = {entry["cache_audio_key"] for entry in tampered["selected"]}
        replacement_row = next(
            row
            for row in rows
            if row["label"]["stratum"] == LABEL_STABLE
            and not row["source"]["long_track"]
            and row["source"]["cache_audio_key"] not in selected_keys
            and row["source"]["cache_audio_key"] not in exposure_keys
        )
        stable_start = sum(
            EXP004_HOLDOUT_QUOTAS[quota]
            for quota in EXP004_PRIORITY
            if quota != "stable"
        )
        stable_entries = tampered["selected"][stable_start:]
        stable_entries[-1] = _holdout_entry(replacement_row, quota="stable")
        stable_entries.sort(
            key=lambda entry: (
                entry["selection_hash_sha256"],
                entry["cache_audio_key"],
                entry["audio_group_key"],
            )
        )
        for rank, entry in enumerate(stable_entries, start=1):
            entry["selection_rank"] = rank
        tampered["selected"][stable_start:] = stable_entries
        _refingerprint(tampered)
        validate_exp004_manifest(tampered)

        with self.assertRaisesRegex(ValueError, "not the deterministic Exp004 holdout replay"):
            select_exp004_broad500(
                rows,
                exposure_manifest=exposure,
                holdout_manifest=tampered,
            )

    def test_manifest_validation_rejects_refingerprinted_metric_or_oracle_fields(self) -> None:
        rows, pilot_keys, protocol_keys, exp003_keys = _corpus()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            exposure, exposure_path, labels_path = _build_exposure(
                root,
                rows,
                pilot_keys=pilot_keys,
                protocol_keys=protocol_keys,
                exp003_keys=exp003_keys,
            )
            holdout_path = root / "holdout.json"
            holdout = build_exp004_holdout100(
                labels_jsonl_path=labels_path,
                exposure_manifest_path=exposure_path,
                manifest_output_path=holdout_path,
            )
            broad_path = root / "broad.json"
            broad = build_exp004_broad500(
                labels_jsonl_path=labels_path,
                exposure_manifest_path=exposure_path,
                holdout_manifest_path=holdout_path,
                manifest_output_path=broad_path,
            )

        for base_manifest in (holdout, broad):
            top_level = json.loads(json.dumps(base_manifest))
            top_level["oracle_metric"] = 1.0
            _refingerprint(top_level)
            with self.assertRaisesRegex(ValueError, "metric/oracle-valued manifest field"):
                validate_exp004_manifest(top_level)

            nested = json.loads(json.dumps(base_manifest))
            nested["selection"]["metrics"] = {"rmse_ms": 1.0}
            _refingerprint(nested)
            with self.assertRaisesRegex(ValueError, "metric/oracle-valued manifest field"):
                validate_exp004_manifest(nested)

            selected = json.loads(json.dumps(base_manifest))
            selected["selected"][0]["scores"] = [1.0]
            _refingerprint(selected)
            with self.assertRaisesRegex(ValueError, "metric/oracle-valued manifest field"):
                validate_exp004_manifest(selected)

            extra = json.loads(json.dumps(base_manifest))
            extra["selected"][0]["unsupported"] = True
            _refingerprint(extra)
            with self.assertRaisesRegex(ValueError, "selected\\[0\\] fields"):
                validate_exp004_manifest(extra)

    def test_protocol_execution_input_wrappers_check_split_and_label_identities(self) -> None:
        rows, pilot_keys, protocol_keys, exp003_keys = _corpus()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _exposure, exposure_path, labels_path = _build_exposure(
                root,
                rows,
                pilot_keys=pilot_keys,
                protocol_keys=protocol_keys,
                exp003_keys=exp003_keys,
            )
            holdout_path = root / "holdout.json"
            holdout = build_exp004_holdout100(
                labels_jsonl_path=labels_path,
                exposure_manifest_path=exposure_path,
                manifest_output_path=holdout_path,
            )
            holdout_execution = build_exp004_execution_inputs_from_split_manifest(
                labels_jsonl_path=labels_path,
                exposure_manifest_path=exposure_path,
                split_manifest_path=holdout_path,
                identity_rows_jsonl_path=root / "holdout_identity.jsonl",
                execution_selection_manifest_path=root / "holdout_execution.json",
            )
            holdout_identities, _identity_source = exp004_protocol.load_exp004_identity_rows(
                root / "holdout_identity.jsonl",
                expected_stage="holdout100",
            )
            _loaded_holdout_execution, holdout_entries, _selection_source = (
                exp004_protocol.load_exp004_execution_selection_manifest(
                    root / "holdout_execution.json",
                    expected_stage="holdout100",
                )
            )
            exp004_protocol.reconcile_exp004_execution_inputs(
                holdout_identities,
                holdout_entries,
                expected_stage="holdout100",
            )
            self.assertEqual(
                [entry["cache_audio_key"] for entry in holdout_execution["selected"]],
                [entry["cache_audio_key"] for entry in holdout["selected"]],
            )
            self.assertEqual(
                holdout_execution["source"]["stage_constraints"],
                {
                    "schema": exp004_protocol.STAGE_CONSTRAINT_SCHEMA,
                    "stage": "holdout100",
                    "quota_degraded": False,
                    "degraded_quotas": [],
                    "broad_underfilled": False,
                },
            )

            broad_path = root / "broad.json"
            broad = build_exp004_broad500(
                labels_jsonl_path=labels_path,
                exposure_manifest_path=exposure_path,
                holdout_manifest_path=holdout_path,
                manifest_output_path=broad_path,
            )
            build_exp004_execution_inputs_from_split_manifest(
                labels_jsonl_path=labels_path,
                exposure_manifest_path=exposure_path,
                split_manifest_path=broad_path,
                holdout_manifest_path=holdout_path,
                identity_rows_jsonl_path=root / "broad_identity.jsonl",
                execution_selection_manifest_path=root / "broad_execution.json",
            )
            broad_identities, _ = exp004_protocol.load_exp004_identity_rows(
                root / "broad_identity.jsonl",
                expected_stage="broad500",
            )
            _broad_execution, broad_entries, _ = (
                exp004_protocol.load_exp004_execution_selection_manifest(
                    root / "broad_execution.json",
                    expected_stage="broad500",
                )
            )
            exp004_protocol.reconcile_exp004_execution_inputs(
                broad_identities,
                broad_entries,
                expected_stage="broad500",
            )
            self.assertEqual(len(broad["selected"]), 500)
            broad_execution = _read_json(root / "broad_execution.json")
            self.assertEqual(
                broad_execution["source"]["stage_constraints"],
                {
                    "schema": exp004_protocol.STAGE_CONSTRAINT_SCHEMA,
                    "stage": "broad500",
                    "quota_degraded": False,
                    "degraded_quotas": [],
                    "broad_underfilled": False,
                },
            )

            selected_repair80_keys = [
                str(row["source"]["cache_audio_key"])
                for row in rows[:80]
            ]
            build_exp004_execution_inputs_from_selected_keys(
                labels_jsonl_path=labels_path,
                selected_cache_audio_keys=selected_repair80_keys,
                stage="repair80",
                identity_rows_jsonl_path=root / "repair80_identity.jsonl",
                execution_selection_manifest_path=root / "repair80_execution.json",
            )
            repair80_identities, _ = exp004_protocol.load_exp004_identity_rows(
                root / "repair80_identity.jsonl",
                expected_stage="repair80",
            )
            _repair80_execution, repair80_entries, _ = (
                exp004_protocol.load_exp004_execution_selection_manifest(
                    root / "repair80_execution.json",
                    expected_stage="repair80",
                )
            )
            exp004_protocol.reconcile_exp004_execution_inputs(
                repair80_identities,
                repair80_entries,
                expected_stage="repair80",
            )

            full_rows = [_label_row(f"full-{index:04d}", LABEL_STABLE) for index in range(5050)]
            full_labels_path = _write_jsonl(root / "full_labels.jsonl", full_rows)
            build_exp004_full5050_execution_inputs(
                labels_jsonl_path=full_labels_path,
                identity_rows_jsonl_path=root / "full5050_identity.jsonl",
                execution_selection_manifest_path=root / "full5050_execution.json",
            )
            full_identities, _ = exp004_protocol.load_exp004_identity_rows(
                root / "full5050_identity.jsonl",
                expected_stage="full5050",
            )
            _full_execution, full_entries, _ = (
                exp004_protocol.load_exp004_execution_selection_manifest(
                    root / "full5050_execution.json",
                    expected_stage="full5050",
                )
            )
            exp004_protocol.reconcile_exp004_execution_inputs(
                full_identities,
                full_entries,
                expected_stage="full5050",
            )
            self.assertEqual(full_identities[0].cache_audio_key, "cache::full-0000")
            self.assertEqual(full_identities[-1].cache_audio_key, "cache::full-5049")

            stale_rows = json.loads(json.dumps(rows))
            stale_key = holdout["selected"][0]["cache_audio_key"]
            for row in stale_rows:
                if row["source"]["cache_audio_key"] == stale_key:
                    row["resolved_audio_path"] = "/dataset/stale.mp3"
                    break
            stale_labels_path = _write_jsonl(root / "stale_labels.jsonl", stale_rows)
            with self.assertRaisesRegex(ValueError, "source.labels.path"):
                build_exp004_execution_inputs_from_split_manifest(
                    labels_jsonl_path=stale_labels_path,
                    exposure_manifest_path=exposure_path,
                    split_manifest_path=holdout_path,
                    identity_rows_jsonl_path=root / "stale_identity.jsonl",
                    execution_selection_manifest_path=root / "stale_execution.json",
                )

    def test_execution_input_wrapper_replays_and_rejects_forged_or_changed_sources(self) -> None:
        rows, pilot_keys, protocol_keys, exp003_keys = _corpus()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            exposure, exposure_path, labels_path = _build_exposure(
                root,
                rows,
                pilot_keys=pilot_keys,
                protocol_keys=protocol_keys,
                exp003_keys=exp003_keys,
            )
            holdout_path = root / "holdout.json"
            holdout = build_exp004_holdout100(
                labels_jsonl_path=labels_path,
                exposure_manifest_path=exposure_path,
                manifest_output_path=holdout_path,
            )
            broad_path = root / "broad.json"
            broad = build_exp004_broad500(
                labels_jsonl_path=labels_path,
                exposure_manifest_path=exposure_path,
                holdout_manifest_path=holdout_path,
                manifest_output_path=broad_path,
            )

            forged_holdout = _forged_nonlowest_holdout(rows, exposure, holdout)
            validate_exp004_manifest(forged_holdout)
            forged_holdout_path = root / "forged_holdout.json"
            _write_json(forged_holdout_path, forged_holdout)
            with self.assertRaisesRegex(ValueError, "deterministic Exp004 replay"):
                build_exp004_execution_inputs_from_split_manifest(
                    labels_jsonl_path=labels_path,
                    exposure_manifest_path=exposure_path,
                    split_manifest_path=forged_holdout_path,
                    identity_rows_jsonl_path=root / "forged_holdout_identity.jsonl",
                    execution_selection_manifest_path=root / "forged_holdout_execution.json",
                )

            forged_broad = _forged_nonlowest_broad(rows, exposure, holdout, broad)
            validate_exp004_manifest(forged_broad)
            forged_broad_path = root / "forged_broad.json"
            _write_json(forged_broad_path, forged_broad)
            with self.assertRaisesRegex(ValueError, "deterministic Exp004 replay"):
                build_exp004_execution_inputs_from_split_manifest(
                    labels_jsonl_path=labels_path,
                    exposure_manifest_path=exposure_path,
                    split_manifest_path=forged_broad_path,
                    holdout_manifest_path=holdout_path,
                    identity_rows_jsonl_path=root / "forged_broad_identity.jsonl",
                    execution_selection_manifest_path=root / "forged_broad_execution.json",
                )

            changed_rows = json.loads(json.dumps(rows))
            selected_keys = {entry["cache_audio_key"] for entry in holdout["selected"]}
            for row in changed_rows:
                if row["source"]["cache_audio_key"] not in selected_keys:
                    row["resolved_audio_path"] = "/dataset/unselected-changed.mp3"
                    break
            _write_jsonl(labels_path, changed_rows)
            with self.assertRaisesRegex(ValueError, "source.labels.sha256"):
                build_exp004_execution_inputs_from_split_manifest(
                    labels_jsonl_path=labels_path,
                    exposure_manifest_path=exposure_path,
                    split_manifest_path=holdout_path,
                    identity_rows_jsonl_path=root / "changed_labels_identity.jsonl",
                    execution_selection_manifest_path=root / "changed_labels_execution.json",
                )
            _write_jsonl(labels_path, rows)

            changed_exposure = json.loads(json.dumps(exposure))
            changed_exposure["generated_at_utc"] = "2026-08-11T00:00:01Z"
            _write_json(exposure_path, changed_exposure)
            with self.assertRaisesRegex(ValueError, "source.exposure_manifest.sha256"):
                build_exp004_execution_inputs_from_split_manifest(
                    labels_jsonl_path=labels_path,
                    exposure_manifest_path=exposure_path,
                    split_manifest_path=holdout_path,
                    identity_rows_jsonl_path=root / "changed_exposure_identity.jsonl",
                    execution_selection_manifest_path=root / "changed_exposure_execution.json",
                )
            _write_json(exposure_path, exposure)

            changed_holdout = _forged_nonlowest_holdout(rows, exposure, holdout)
            _write_json(holdout_path, changed_holdout)
            with self.assertRaisesRegex(ValueError, "source.holdout_manifest.sha256"):
                build_exp004_execution_inputs_from_split_manifest(
                    labels_jsonl_path=labels_path,
                    exposure_manifest_path=exposure_path,
                    split_manifest_path=broad_path,
                    holdout_manifest_path=holdout_path,
                    identity_rows_jsonl_path=root / "changed_holdout_identity.jsonl",
                    execution_selection_manifest_path=root / "changed_holdout_execution.json",
                )

    def test_execution_input_stage_constraints_record_degraded_holdout_quota(self) -> None:
        rows, pilot_keys, protocol_keys, exp003_keys = _corpus(
            ramp_unexposed_count=4,
            extra_jump_count=1,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _exposure, exposure_path, labels_path = _build_exposure(
                root,
                rows,
                pilot_keys=pilot_keys,
                protocol_keys=protocol_keys,
                exp003_keys=exp003_keys,
            )
            holdout_path = root / "holdout.json"
            build_exp004_holdout100(
                labels_jsonl_path=labels_path,
                exposure_manifest_path=exposure_path,
                manifest_output_path=holdout_path,
            )
            execution = build_exp004_execution_inputs_from_split_manifest(
                labels_jsonl_path=labels_path,
                exposure_manifest_path=exposure_path,
                split_manifest_path=holdout_path,
                identity_rows_jsonl_path=root / "degraded_holdout_identity.jsonl",
                execution_selection_manifest_path=root / "degraded_holdout_execution.json",
            )

            self.assertEqual(
                execution["source"]["stage_constraints"],
                {
                    "schema": exp004_protocol.STAGE_CONSTRAINT_SCHEMA,
                    "stage": "holdout100",
                    "quota_degraded": True,
                    "degraded_quotas": ["ramp_audit"],
                    "broad_underfilled": False,
                },
            )

    def test_explicit_selected_key_execution_inputs_are_repair80_only(self) -> None:
        rows, pilot_keys, protocol_keys, exp003_keys = _corpus()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _exposure, _exposure_path, labels_path = _build_exposure(
                root,
                rows,
                pilot_keys=pilot_keys,
                protocol_keys=protocol_keys,
                exp003_keys=exp003_keys,
            )
            for stage in ("holdout100", "broad500", "full5050"):
                with self.subTest(stage=stage):
                    with self.assertRaisesRegex(ValueError, "only supported for repair80"):
                        build_exp004_execution_inputs_from_selected_keys(
                            labels_jsonl_path=labels_path,
                            selected_cache_audio_keys=[],
                            stage=stage,
                            identity_rows_jsonl_path=root / f"{stage}_identity.jsonl",
                            execution_selection_manifest_path=root / f"{stage}_execution.json",
                        )

    def test_selected_key_and_full5050_builders_reject_label_file_toctou(self) -> None:
        rows, pilot_keys, protocol_keys, exp003_keys = _corpus()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _exposure, _exposure_path, labels_path = _build_exposure(
                root,
                rows,
                pilot_keys=pilot_keys,
                protocol_keys=protocol_keys,
                exp003_keys=exp003_keys,
            )
            selected_repair80_keys = [
                str(row["source"]["cache_audio_key"])
                for row in rows[:80]
            ]
            original_build = exp004_protocol.build_exp004_execution_inputs

            def mutate_after_selected_build(*args: object, **kwargs: object) -> dict[str, object]:
                result = original_build(*args, **kwargs)
                changed_rows = json.loads(json.dumps(rows))
                changed_rows[0]["resolved_audio_path"] = "/dataset/toctou-selected.mp3"
                _write_jsonl(labels_path, changed_rows)
                return result

            with mock.patch.object(
                exp004_protocol,
                "build_exp004_execution_inputs",
                side_effect=mutate_after_selected_build,
            ):
                with self.assertRaisesRegex(RuntimeError, "labels changed"):
                    build_exp004_execution_inputs_from_selected_keys(
                        labels_jsonl_path=labels_path,
                        selected_cache_audio_keys=selected_repair80_keys,
                        stage="repair80",
                        identity_rows_jsonl_path=root / "toctou_repair80_identity.jsonl",
                        execution_selection_manifest_path=root / "toctou_repair80_execution.json",
                    )

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            full_rows = [_label_row(f"full-{index:04d}", LABEL_STABLE) for index in range(5050)]
            full_labels_path = _write_jsonl(root / "full_labels.jsonl", full_rows)
            original_build = exp004_protocol.build_exp004_execution_inputs

            def mutate_after_full_build(*args: object, **kwargs: object) -> dict[str, object]:
                result = original_build(*args, **kwargs)
                changed_rows = json.loads(json.dumps(full_rows))
                changed_rows[-1]["resolved_audio_path"] = "/dataset/toctou-full5050.mp3"
                _write_jsonl(full_labels_path, changed_rows)
                return result

            with mock.patch.object(
                exp004_protocol,
                "build_exp004_execution_inputs",
                side_effect=mutate_after_full_build,
            ):
                with self.assertRaisesRegex(RuntimeError, "labels changed"):
                    build_exp004_full5050_execution_inputs(
                        labels_jsonl_path=full_labels_path,
                        identity_rows_jsonl_path=root / "toctou_full5050_identity.jsonl",
                        execution_selection_manifest_path=root / "toctou_full5050_execution.json",
                    )

    def test_broad_underfilled_split_validates_but_execution_input_fails_closed(self) -> None:
        rows, pilot_keys, protocol_keys, exp003_keys = _corpus(broad_extra_count=7)
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            _exposure, exposure_path, labels_path = _build_exposure(
                root,
                rows,
                pilot_keys=pilot_keys,
                protocol_keys=protocol_keys,
                exp003_keys=exp003_keys,
            )
            holdout_path = root / "holdout.json"
            build_exp004_holdout100(
                labels_jsonl_path=labels_path,
                exposure_manifest_path=exposure_path,
                manifest_output_path=holdout_path,
            )
            broad_path = root / "broad_underfilled.json"
            broad = build_exp004_broad500(
                labels_jsonl_path=labels_path,
                exposure_manifest_path=exposure_path,
                holdout_manifest_path=holdout_path,
                manifest_output_path=broad_path,
            )
            validate_exp004_manifest(broad)
            self.assertEqual(broad["selected_audio_count"], 107)

            # Underfilled broad manifests may be audited as split manifests, but
            # formal execution must fail closed until a new card defines a
            # smaller stage.  Do not mint an ambiguous broad500 run.
            with self.assertRaisesRegex(ValueError, "exactly 500"):
                build_exp004_execution_inputs_from_split_manifest(
                    labels_jsonl_path=labels_path,
                    exposure_manifest_path=exposure_path,
                    split_manifest_path=broad_path,
                    holdout_manifest_path=holdout_path,
                    identity_rows_jsonl_path=root / "underfilled_broad_identity.jsonl",
                    execution_selection_manifest_path=root / "underfilled_broad_execution.json",
                )


def _corpus(
    *,
    ramp_unexposed_count: int = 5,
    extra_jump_count: int = 0,
    broad_extra_count: int = 410,
) -> tuple[list[dict[str, object]], set[str], set[str], set[str]]:
    rows: list[dict[str, object]] = []
    for index in range(80):
        rows.append(_label_row(f"pilot-{index:03d}", LABEL_STABLE))
    pilot_keys = {str(row["source"]["cache_audio_key"]) for row in rows}

    protocol_rows: list[dict[str, object]] = []
    for index in range(3):
        row = _label_row(f"protocol-{index:03d}", LABEL_RAMP_CANDIDATE)
        protocol_rows.append(row)
        rows.append(row)
    protocol_keys = {str(row["source"]["cache_audio_key"]) for row in protocol_rows}

    exp003_rows: list[dict[str, object]] = []
    for index in range(100):
        row = _label_row(f"exp003-holdout-{index:03d}", LABEL_STABLE)
        exp003_rows.append(row)
        rows.append(row)
    exp003_keys = {str(row["source"]["cache_audio_key"]) for row in exp003_rows}

    for index in range(ramp_unexposed_count):
        rows.append(_label_row(f"ramp-{index:03d}", LABEL_RAMP_CANDIDATE))
    for index in range(10):
        rows.append(_label_row(f"anomaly-{index:03d}", LABEL_AMBIGUOUS, long_track=index == 0))
    for index in range(10):
        stratum = (LABEL_DENSE, LABEL_JUMP_CANDIDATE, LABEL_STABLE)[index % 3]
        rows.append(_label_row(f"long-{index:03d}", stratum, long_track=True))
    for index in range(10):
        rows.append(_label_row(f"dense-{index:03d}", LABEL_DENSE))
    for index in range(25 + extra_jump_count):
        rows.append(_label_row(f"jump-{index:03d}", LABEL_JUMP_CANDIDATE))
    for index in range(40):
        rows.append(_label_row(f"stable-{index:03d}", LABEL_STABLE))
    for index in range(broad_extra_count):
        rows.append(_label_row(f"extra-{index:03d}", LABEL_STABLE))
    return rows, pilot_keys, protocol_keys, exp003_keys


def _label_row(
    name: str,
    stratum: str,
    *,
    long_track: bool = False,
) -> dict[str, object]:
    return {
        "schema": TIMING_V3_LABEL_AUDIO_ROW_SCHEMA,
        "audio_group_key": f"audio/{name}",
        "resolved_audio_path": f"/dataset/{name}.mp3",
        "map_count": 1,
        "maps": [{"resolved_beatmap_path": f"/dataset/{name}.osu"}],
        "source": {
            "cache_audio_key": f"cache::{name}",
            "cache_duration_seconds": 700.0 if long_track else 120.0,
            "long_track": long_track,
        },
        "label": {
            "stratum": stratum,
            "confidence": "medium",
            "ambiguous": stratum in {LABEL_AMBIGUOUS, LABEL_DENSE, LABEL_RAMP_CANDIDATE},
        },
    }


def _build_exposure(
    root: Path,
    rows: list[dict[str, object]],
    *,
    pilot_keys: set[str],
    protocol_keys: set[str],
    exp003_keys: set[str],
) -> tuple[dict[str, object], Path, Path]:
    labels_path = _write_jsonl(root / "labels.jsonl", rows)
    pilot_path, protocol_path, exp003_path = _write_required_sources(
        root,
        rows,
        pilot_keys=pilot_keys,
        protocol_keys=protocol_keys,
        exp003_keys=exp003_keys,
    )
    exposure_path = root / "exposure.json"
    exposure = build_exp004_exposure_manifest(
        pilot_manifest_path=pilot_path,
        protocol_manifest_path=protocol_path,
        exp003_holdout_manifest_path=exp003_path,
        labels_jsonl_path=labels_path,
        generated_from_commit="0421de8a",
        generated_at_utc="2026-08-11T00:00:00Z",
        manifest_output_path=exposure_path,
    )
    return exposure, exposure_path, labels_path


def _write_required_sources(
    root: Path,
    rows: list[dict[str, object]],
    *,
    pilot_keys: set[str],
    protocol_keys: set[str],
    exp003_keys: set[str],
) -> tuple[Path, Path, Path]:
    by_key = {str(row["source"]["cache_audio_key"]): row for row in rows}
    pilot_path = _write_jsonl(root / "pilot.jsonl", [by_key[key] for key in sorted(pilot_keys)])
    protocol_path = _write_jsonl(
        root / "protocol.jsonl",
        [by_key[key] for key in sorted(protocol_keys)],
    )
    exp003_path = root / "exp003_holdout.json"
    exp003_path.write_text(
        json.dumps(
            {
                "selected": [
                    {"cache_audio_key": key}
                    for key in sorted(exp003_keys)
                ]
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return pilot_path, protocol_path, exp003_path


def _holdout_entry(row: dict[str, object], *, quota: str) -> dict[str, object]:
    cache_audio_key = str(row["source"]["cache_audio_key"])
    return {
        "stage": "holdout100",
        "quota_assignment": quota,
        "selection_substage": "quota",
        "selection_rank": 1,
        "selection_hash_sha256": hashlib.sha256(
            f"{EXP004_HOLDOUT_SEED}\0{cache_audio_key}".encode("utf-8")
        ).hexdigest(),
        "cache_audio_key": cache_audio_key,
        "audio_group_key": row["audio_group_key"],
        "resolved_audio_path": row["resolved_audio_path"],
        "label_stratum": row["label"]["stratum"],
        "source_long_track": row["source"]["long_track"],
    }


def _broad_added_entry(row: dict[str, object]) -> dict[str, object]:
    cache_audio_key = str(row["source"]["cache_audio_key"])
    return {
        "stage": "broad500_added",
        "quota_assignment": None,
        "selection_substage": "broad500_added",
        "selection_rank": 1,
        "selection_hash_sha256": hashlib.sha256(
            f"{EXP004_BROAD_SEED}\0{cache_audio_key}".encode("utf-8")
        ).hexdigest(),
        "cache_audio_key": cache_audio_key,
        "audio_group_key": row["audio_group_key"],
        "resolved_audio_path": row["resolved_audio_path"],
        "label_stratum": row["label"]["stratum"],
        "source_long_track": row["source"]["long_track"],
    }


def _forged_nonlowest_holdout(
    rows: list[dict[str, object]],
    exposure: dict[str, object],
    holdout: dict[str, object],
) -> dict[str, object]:
    tampered = json.loads(json.dumps(holdout))
    exposure_keys = validate_exp004_exposure_manifest(exposure)
    selected_keys = {entry["cache_audio_key"] for entry in tampered["selected"]}
    replacement_row = next(
        row
        for row in rows
        if row["label"]["stratum"] == LABEL_STABLE
        and not row["source"]["long_track"]
        and row["source"]["cache_audio_key"] not in selected_keys
        and row["source"]["cache_audio_key"] not in exposure_keys
    )
    stable_start = sum(
        EXP004_HOLDOUT_QUOTAS[quota]
        for quota in EXP004_PRIORITY
        if quota != "stable"
    )
    stable_entries = tampered["selected"][stable_start:]
    stable_entries[-1] = _holdout_entry(replacement_row, quota="stable")
    stable_entries.sort(
        key=lambda entry: (
            entry["selection_hash_sha256"],
            entry["cache_audio_key"],
            entry["audio_group_key"],
        )
    )
    for rank, entry in enumerate(stable_entries, start=1):
        entry["selection_rank"] = rank
    tampered["selected"][stable_start:] = stable_entries
    _refingerprint(tampered)
    return tampered


def _forged_nonlowest_broad(
    rows: list[dict[str, object]],
    exposure: dict[str, object],
    holdout: dict[str, object],
    broad: dict[str, object],
) -> dict[str, object]:
    tampered = json.loads(json.dumps(broad))
    exposure_keys = validate_exp004_exposure_manifest(exposure)
    holdout_keys = {entry["cache_audio_key"] for entry in holdout["selected"]}
    broad_keys = {entry["cache_audio_key"] for entry in broad["selected"]}
    replacement_row = next(
        row
        for row in rows
        if row["source"]["cache_audio_key"] not in exposure_keys
        and row["source"]["cache_audio_key"] not in holdout_keys
        and row["source"]["cache_audio_key"] not in broad_keys
    )
    added_entries = tampered["selected"][EXP004_HOLDOUT_AUDIO_COUNT:]
    added_entries[-1] = _broad_added_entry(replacement_row)
    added_entries.sort(
        key=lambda entry: (
            entry["selection_hash_sha256"],
            entry["cache_audio_key"],
            entry["audio_group_key"],
        )
    )
    for rank, entry in enumerate(added_entries, start=1):
        entry["selection_rank"] = rank
    tampered["selected"][EXP004_HOLDOUT_AUDIO_COUNT:] = added_entries
    _refingerprint(tampered)
    return tampered


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(row, allow_nan=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _key_set_sha256(keys: set[str]) -> str:
    payload = json.dumps(
        sorted(keys),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _refingerprint(manifest: dict[str, object]) -> None:
    manifest["selected_cache_audio_keys_sha256"] = _key_set_sha256(
        {entry["cache_audio_key"] for entry in manifest["selected"]}
    )
    manifest.pop("manifest_fingerprint_sha256", None)
    payload = json.dumps(
        manifest,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    manifest["manifest_fingerprint_sha256"] = hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    unittest.main()
