from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pulsefield_model.timing.evaluation.labels import (
    LABEL_AMBIGUOUS,
    LABEL_DENSE,
    LABEL_JUMP_CANDIDATE,
    LABEL_RAMP_CANDIDATE,
    LABEL_STABLE,
    TIMING_V3_LABEL_AUDIO_ROW_SCHEMA,
)
from pulsefield_model.timing.evaluation.splits import (
    EXP003_BROAD_SEED,
    EXP003_HOLDOUT_QUOTAS,
    EXP003_HOLDOUT_SEED,
    EXP003_PRIORITY,
    TIMING_V3_EXP003_BROAD_MANIFEST_SCHEMA,
    TIMING_V3_EXP003_HOLDOUT_MANIFEST_SCHEMA,
    build_exp003_broad500,
    build_exp003_holdout100,
    load_exclusion_cache_audio_keys,
    load_exp003_manifest,
    main,
    materialize_baseline_subset,
    materialize_label_subset,
    select_exp003_broad500,
    select_exp003_holdout100,
    validate_exp003_manifest,
)


class TimingV3Experiment003SplitTests(unittest.TestCase):
    def test_holdout_is_frozen_exclusive_audio_disjoint_and_replay_deterministic(self) -> None:
        rows, pilot_keys, protocol_keys = _corpus()

        first = select_exp003_holdout100(
            rows,
            pilot_excluded_cache_audio_keys=pilot_keys,
            protocol_excluded_cache_audio_keys=protocol_keys,
        )
        second = select_exp003_holdout100(
            list(reversed(rows)),
            pilot_excluded_cache_audio_keys=sorted(pilot_keys, reverse=True),
            protocol_excluded_cache_audio_keys=sorted(protocol_keys, reverse=True),
        )

        self.assertEqual(first, second)
        self.assertEqual(first["schema"], TIMING_V3_EXP003_HOLDOUT_MANIFEST_SCHEMA)
        self.assertEqual(
            TIMING_V3_EXP003_HOLDOUT_MANIFEST_SCHEMA,
            "pulsefield_model.timing_v3_exp003_holdout100_manifest_v2",
        )
        self.assertEqual(first["seed"], EXP003_HOLDOUT_SEED)
        self.assertEqual(EXP003_HOLDOUT_SEED, "timing-v3-exp003-holdout100-v2")
        self.assertEqual(first["selection"]["priority_order"], list(EXP003_PRIORITY))
        self.assertEqual(first["selection"]["quotas"], EXP003_HOLDOUT_QUOTAS)
        self.assertEqual(
            EXP003_HOLDOUT_QUOTAS,
            {
                "ramp_audit": 4,
                "anomaly": 10,
                "long": 11,
                "dense": 10,
                "jump": 25,
                "stable": 40,
            },
        )
        self.assertEqual(first["selected_audio_count"], 100)
        self.assertEqual(first["selected_counts"], EXP003_HOLDOUT_QUOTAS)
        selected = first["selected"]
        selected_keys = [entry["cache_audio_key"] for entry in selected]
        self.assertEqual(len(selected_keys), len(set(selected_keys)))
        self.assertTrue(pilot_keys.isdisjoint(selected_keys))
        self.assertTrue(protocol_keys.isdisjoint(selected_keys))
        self.assertEqual(
            [entry["quota_assignment"] for entry in selected[:4]],
            ["ramp_audit"] * 4,
        )
        # The long ramp/ambiguous rows prove higher-priority classes win.
        by_audio_group = {entry["audio_group_key"]: entry for entry in selected}
        self.assertEqual(by_audio_group["audio/ramp-003"]["quota_assignment"], "ramp_audit")
        self.assertEqual(by_audio_group["audio/anomaly-000"]["quota_assignment"], "anomaly")
        for entry in selected:
            expected = hashlib.sha256(
                f"{EXP003_HOLDOUT_SEED}\0{entry['cache_audio_key']}".encode("utf-8")
            ).hexdigest()
            self.assertEqual(entry["selection_hash_sha256"], expected)
        validate_exp003_manifest(first)

    def test_holdout_stops_on_underfilled_quota_without_backfill(self) -> None:
        rows, pilot_keys, protocol_keys = _corpus()
        rows = [row for row in rows if row["audio_group_key"] != "audio/ramp-006"]

        with self.assertRaisesRegex(ValueError, "ramp_audit.*underfilled"):
            select_exp003_holdout100(
                rows,
                pilot_excluded_cache_audio_keys=pilot_keys,
                protocol_excluded_cache_audio_keys=protocol_keys,
            )

    def test_label_and_exclusion_duplicates_fail_closed(self) -> None:
        rows, pilot_keys, protocol_keys = _corpus()
        duplicate_cache = dict(rows[-1])
        duplicate_cache["audio_group_key"] = "audio/different"
        with self.assertRaisesRegex(ValueError, "duplicate label cache_audio_key"):
            select_exp003_holdout100(
                [*rows, duplicate_cache],
                pilot_excluded_cache_audio_keys=pilot_keys,
                protocol_excluded_cache_audio_keys=protocol_keys,
            )

        with self.assertRaisesRegex(ValueError, "duplicate cache audio key"):
            select_exp003_holdout100(
                rows,
                pilot_excluded_cache_audio_keys=[next(iter(pilot_keys))] * 2,
                protocol_excluded_cache_audio_keys=protocol_keys,
            )
        with self.assertRaisesRegex(ValueError, "exactly 80 frozen pilot exclusions"):
            select_exp003_holdout100(
                rows,
                pilot_excluded_cache_audio_keys=[],
                protocol_excluded_cache_audio_keys=protocol_keys,
            )

    def test_protocol_exclusion_omission_duplicate_overlap_and_missing_fail_closed(self) -> None:
        rows, pilot_keys, protocol_keys = _corpus()
        ordered_protocol = sorted(protocol_keys)
        with self.assertRaisesRegex(ValueError, "exactly 3 protocol exclusions"):
            select_exp003_holdout100(
                rows,
                pilot_excluded_cache_audio_keys=pilot_keys,
                protocol_excluded_cache_audio_keys=ordered_protocol[:2],
            )
        with self.assertRaisesRegex(ValueError, "duplicate cache audio key"):
            select_exp003_holdout100(
                rows,
                pilot_excluded_cache_audio_keys=pilot_keys,
                protocol_excluded_cache_audio_keys=[ordered_protocol[0]] * 3,
            )
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            select_exp003_holdout100(
                rows,
                pilot_excluded_cache_audio_keys=pilot_keys,
                protocol_excluded_cache_audio_keys=sorted(pilot_keys)[:3],
            )
        with self.assertRaisesRegex(ValueError, "absent from label_rows"):
            select_exp003_holdout100(
                rows,
                pilot_excluded_cache_audio_keys=pilot_keys,
                protocol_excluded_cache_audio_keys={
                    "cache::missing-a",
                    "cache::missing-b",
                    "cache::missing-c",
                },
            )
        with self.assertRaises(SystemExit) as raised:
            main(
                [
                    "holdout100",
                    "--labels-jsonl",
                    "labels.jsonl",
                    "--pilot-exclusion",
                    "pilot.jsonl",
                    "--manifest-output",
                    "holdout.json",
                ]
            )
        self.assertEqual(raised.exception.code, 2)

    def test_broad500_is_holdout_plus_lowest_400_new_hashes(self) -> None:
        rows, pilot_keys, protocol_keys = _corpus()
        holdout = select_exp003_holdout100(
            rows,
            pilot_excluded_cache_audio_keys=pilot_keys,
            protocol_excluded_cache_audio_keys=protocol_keys,
        )

        broad = select_exp003_broad500(
            list(reversed(rows)),
            pilot_excluded_cache_audio_keys=pilot_keys,
            protocol_excluded_cache_audio_keys=protocol_keys,
            holdout_manifest=holdout,
        )
        replay = select_exp003_broad500(
            rows,
            pilot_excluded_cache_audio_keys=sorted(pilot_keys),
            protocol_excluded_cache_audio_keys=sorted(protocol_keys),
            holdout_manifest=holdout,
        )

        self.assertEqual(broad, replay)
        self.assertEqual(broad["schema"], TIMING_V3_EXP003_BROAD_MANIFEST_SCHEMA)
        self.assertEqual(
            TIMING_V3_EXP003_BROAD_MANIFEST_SCHEMA,
            "pulsefield_model.timing_v3_exp003_broad500_manifest_v2",
        )
        self.assertEqual(EXP003_BROAD_SEED, "timing-v3-exp003-broad500-v2")
        self.assertEqual(broad["selected_audio_count"], 500)
        self.assertEqual(broad["selected"][:100], holdout["selected"])
        added = broad["selected"][100:]
        self.assertEqual(len(added), 400)
        self.assertTrue(all(entry["stage"] == "broad500_added" for entry in added))
        self.assertTrue(all(entry["quota_assignment"] is None for entry in added))
        self.assertEqual([entry["selection_rank"] for entry in added], list(range(1, 401)))
        self.assertEqual(
            [entry["selection_hash_sha256"] for entry in added],
            sorted(entry["selection_hash_sha256"] for entry in added),
        )
        expected_hashes = [
            hashlib.sha256(
                f"{EXP003_BROAD_SEED}\0{entry['cache_audio_key']}".encode("utf-8")
            ).hexdigest()
            for entry in added
        ]
        self.assertEqual(
            [entry["selection_hash_sha256"] for entry in added],
            expected_hashes,
        )
        all_keys = [entry["cache_audio_key"] for entry in broad["selected"]]
        self.assertEqual(len(all_keys), len(set(all_keys)))
        self.assertTrue(pilot_keys.isdisjoint(all_keys))
        self.assertTrue(protocol_keys.isdisjoint(all_keys))
        validate_exp003_manifest(broad)

        drifted_source = json.loads(json.dumps(broad["source"]))
        drifted_source["labels"]["path"] = "/copied/labels.jsonl"
        with self.assertRaisesRegex(ValueError, "provenance does not exactly match"):
            select_exp003_broad500(
                rows,
                pilot_excluded_cache_audio_keys=pilot_keys,
                protocol_excluded_cache_audio_keys=protocol_keys,
                holdout_manifest=holdout,
                source=drifted_source,
            )

    def test_broad_rejects_internally_valid_nonlowest_holdout_replay(self) -> None:
        rows, pilot_keys, protocol_keys = _corpus()
        holdout = select_exp003_holdout100(
            rows,
            pilot_excluded_cache_audio_keys=pilot_keys,
            protocol_excluded_cache_audio_keys=protocol_keys,
        )
        tampered = json.loads(json.dumps(holdout))
        stable_start = sum(
            EXP003_HOLDOUT_QUOTAS[quota]
            for quota in EXP003_PRIORITY
            if quota != "stable"
        )
        selected_keys = {entry["cache_audio_key"] for entry in tampered["selected"]}
        replacement_row = next(
            row
            for row in rows
            if row["label"]["stratum"] == LABEL_STABLE
            and not row["source"]["long_track"]
            and row["source"]["cache_audio_key"] not in selected_keys
            and row["source"]["cache_audio_key"] not in pilot_keys
            and row["source"]["cache_audio_key"] not in protocol_keys
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
        validate_exp003_manifest(tampered)

        with self.assertRaisesRegex(ValueError, "not the deterministic protocol-v2"):
            select_exp003_broad500(
                rows,
                pilot_excluded_cache_audio_keys=pilot_keys,
                protocol_excluded_cache_audio_keys=protocol_keys,
                holdout_manifest=tampered,
            )

    def test_manifest_tampering_and_pilot_holdout_overlap_fail_closed(self) -> None:
        rows, pilot_keys, protocol_keys = _corpus()
        holdout = select_exp003_holdout100(
            rows,
            pilot_excluded_cache_audio_keys=pilot_keys,
            protocol_excluded_cache_audio_keys=protocol_keys,
        )
        tampered = json.loads(json.dumps(holdout))
        tampered["selected"][0]["selection_rank"] = 99
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            validate_exp003_manifest(tampered)

        missing_source = json.loads(json.dumps(holdout))
        missing_source["source"] = {}
        _refingerprint(missing_source)
        with self.assertRaisesRegex(ValueError, "source.labels"):
            validate_exp003_manifest(missing_source)

        tampered_protocol = json.loads(json.dumps(holdout))
        tampered_protocol["protocol_exclusion"]["cache_audio_keys_sha256"] = "0" * 64
        _refingerprint(tampered_protocol)
        with self.assertRaisesRegex(ValueError, "protocol_exclusion.*does not replay"):
            validate_exp003_manifest(tampered_protocol)

        tampered_protocol_source = json.loads(json.dumps(holdout))
        tampered_protocol_source["source"]["protocol_exclusion"]["sha256"] = "0" * 64
        _refingerprint(tampered_protocol_source)
        with self.assertRaisesRegex(ValueError, "source.protocol_exclusion.*does not replay"):
            validate_exp003_manifest(tampered_protocol_source)

        overlap_keys = set(pilot_keys)
        overlap_keys.remove(next(iter(overlap_keys)))
        overlap_keys.add(holdout["selected"][0]["cache_audio_key"])
        with self.assertRaisesRegex(ValueError, "do not match the frozen holdout exclusions"):
            select_exp003_broad500(
                rows,
                pilot_excluded_cache_audio_keys=overlap_keys,
                protocol_excluded_cache_audio_keys=protocol_keys,
                holdout_manifest=holdout,
            )

        changed_protocol = set(protocol_keys)
        changed_protocol.remove(next(iter(changed_protocol)))
        changed_protocol.add("cache::stable-000")
        with self.assertRaisesRegex(ValueError, "protocol exclusions do not match"):
            select_exp003_broad500(
                rows,
                pilot_excluded_cache_audio_keys=pilot_keys,
                protocol_excluded_cache_audio_keys=changed_protocol,
                holdout_manifest=holdout,
            )

    def test_builders_store_source_hashes_and_materialize_full_label_rows(self) -> None:
        rows, _pilot_keys, _protocol_keys = _corpus()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            labels_path = _write_jsonl(root / "labels.jsonl", rows)
            pilot_path = _write_jsonl(root / "pilot.jsonl", rows[:80])
            protocol_path = _write_jsonl(root / "protocol.jsonl", rows[80:83])
            holdout_path = root / "holdout.json"
            holdout_rows_path = root / "holdout_rows.jsonl"

            with self.assertRaisesRegex(ValueError, "paths must be distinct"):
                build_exp003_holdout100(
                    labels_jsonl_path=labels_path,
                    pilot_exclusion_path=pilot_path,
                    protocol_exclusion_path=pilot_path,
                    manifest_output_path=holdout_path,
                )

            holdout = build_exp003_holdout100(
                labels_jsonl_path=labels_path,
                pilot_exclusion_path=pilot_path,
                protocol_exclusion_path=protocol_path,
                manifest_output_path=holdout_path,
                label_rows_output_path=holdout_rows_path,
            )

            self.assertEqual(load_exp003_manifest(holdout_path), holdout)
            self.assertEqual(holdout["source"]["labels"]["sha256"], _sha256(labels_path))
            self.assertEqual(
                holdout["source"]["pilot_exclusion"]["sha256"],
                _sha256(pilot_path),
            )
            self.assertEqual(
                holdout["source"]["protocol_exclusion"]["sha256"],
                _sha256(protocol_path),
            )
            self.assertEqual(len(holdout["exclusion"]["cache_audio_keys_sha256"]), 64)
            self.assertEqual(
                len(holdout["protocol_exclusion"]["cache_audio_keys_sha256"]),
                64,
            )
            materialized = _read_jsonl(holdout_rows_path)
            self.assertEqual(len(materialized), 100)
            self.assertEqual(
                [row["cache_audio_key"] for row in materialized],
                [entry["cache_audio_key"] for entry in holdout["selected"]],
            )
            self.assertTrue(all("maps" in row for row in materialized))
            self.assertTrue(all("experiment_split" in row for row in materialized))

            broad_path = root / "broad.json"
            broad_rows_path = root / "broad_rows.jsonl"
            broad = build_exp003_broad500(
                labels_jsonl_path=labels_path,
                pilot_exclusion_path=pilot_path,
                protocol_exclusion_path=protocol_path,
                holdout_manifest_path=holdout_path,
                manifest_output_path=broad_path,
                label_rows_output_path=broad_rows_path,
            )
            self.assertEqual(broad["selected_audio_count"], 500)
            self.assertEqual(len(_read_jsonl(broad_rows_path)), 500)
            self.assertEqual(
                broad["source"]["holdout_manifest"]["sha256"],
                _sha256(holdout_path),
            )

    def test_pilot_json_manifest_can_resolve_audio_groups_to_cache_keys(self) -> None:
        rows, pilot_keys, _protocol_keys = _corpus()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pilot_manifest_path = root / "pilot.json"
            pilot_manifest_path.write_text(
                json.dumps(
                    {
                        "selected": {
                            "stable": [
                                {"audio_group_key": row["audio_group_key"]}
                                for row in rows[:80]
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )

            resolved = load_exclusion_cache_audio_keys(
                pilot_manifest_path,
                label_rows=rows,
                expected_count=80,
            )

            self.assertEqual(resolved, frozenset(pilot_keys))

    def test_materialized_label_subset_rejects_identity_mismatch(self) -> None:
        rows, pilot_keys, protocol_keys = _corpus()
        manifest = select_exp003_holdout100(
            rows,
            pilot_excluded_cache_audio_keys=pilot_keys,
            protocol_excluded_cache_audio_keys=protocol_keys,
        )
        self.assertEqual(len(materialize_label_subset(rows, manifest)), 100)
        missing = [
            row
            for row in rows
            if row["source"]["cache_audio_key"]
            != manifest["selected"][0]["cache_audio_key"]
        ]
        with self.assertRaisesRegex(ValueError, "absent from label inventory"):
            materialize_label_subset(missing, manifest)

    def test_broad_and_materialization_reject_stale_label_source_fields(self) -> None:
        rows, pilot_keys, protocol_keys = _corpus()
        manifest = select_exp003_holdout100(
            rows,
            pilot_excluded_cache_audio_keys=pilot_keys,
            protocol_excluded_cache_audio_keys=protocol_keys,
        )
        selected_key = manifest["selected"][0]["cache_audio_key"]
        stale_rows = json.loads(json.dumps(rows))
        stale_row = next(
            row for row in stale_rows if row["source"]["cache_audio_key"] == selected_key
        )
        stale_row["label"]["stratum"] = LABEL_STABLE
        stale_row["source"]["long_track"] = False

        with self.assertRaisesRegex(ValueError, "label stratum is stale"):
            materialize_label_subset(stale_rows, manifest)
        with self.assertRaisesRegex(ValueError, "label stratum is stale"):
            select_exp003_broad500(
                stale_rows,
                pilot_excluded_cache_audio_keys=pilot_keys,
                protocol_excluded_cache_audio_keys=protocol_keys,
                holdout_manifest=manifest,
            )

    def test_baseline_subset_preserves_source_row_bytes_in_manifest_order(self) -> None:
        rows, pilot_keys, protocol_keys = _corpus()
        manifest = select_exp003_holdout100(
            rows,
            pilot_excluded_cache_audio_keys=pilot_keys,
            protocol_excluded_cache_audio_keys=protocol_keys,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            source_lines = [
                '{"z": %d, "audio_key": %s}'
                % (index, json.dumps(entry["cache_audio_key"]))
                for index, entry in enumerate(reversed(manifest["selected"]))
            ]
            baseline_path = root / "full.jsonl"
            baseline_path.write_text("\n".join(source_lines) + "\n", encoding="utf-8")
            output_path = root / "subset.jsonl"

            report = materialize_baseline_subset(
                baseline_jsonl_path=baseline_path,
                manifest_path=manifest_path,
                output_jsonl_path=output_path,
            )

            output_lines = output_path.read_text(encoding="utf-8").splitlines()
            by_key = {json.loads(line)["audio_key"]: line for line in source_lines}
            self.assertEqual(
                output_lines,
                [by_key[entry["cache_audio_key"]] for entry in manifest["selected"]],
            )
            self.assertEqual(report["output"]["row_count"], 100)
            self.assertEqual(report["output"]["sha256"], _sha256(output_path))

    def test_baseline_subset_rejects_missing_and_duplicate_keys(self) -> None:
        rows, pilot_keys, protocol_keys = _corpus()
        manifest = select_exp003_holdout100(
            rows,
            pilot_excluded_cache_audio_keys=pilot_keys,
            protocol_excluded_cache_audio_keys=protocol_keys,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            first_key = manifest["selected"][0]["cache_audio_key"]
            duplicate_path = root / "duplicate.jsonl"
            duplicate_path.write_text(
                json.dumps({"audio_key": first_key}) + "\n" + json.dumps({"audio_key": first_key}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate baseline audio_key"):
                materialize_baseline_subset(
                    baseline_jsonl_path=duplicate_path,
                    manifest_path=manifest_path,
                    output_jsonl_path=root / "duplicate-output.jsonl",
                )

            missing_path = root / "missing.jsonl"
            missing_path.write_text(json.dumps({"audio_key": first_key}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "baseline is missing"):
                materialize_baseline_subset(
                    baseline_jsonl_path=missing_path,
                    manifest_path=manifest_path,
                    output_jsonl_path=root / "missing-output.jsonl",
                )

    def test_baseline_subset_cli_option(self) -> None:
        rows, pilot_keys, protocol_keys = _corpus()
        manifest = select_exp003_holdout100(
            rows,
            pilot_excluded_cache_audio_keys=pilot_keys,
            protocol_excluded_cache_audio_keys=protocol_keys,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            baseline_path = _write_jsonl(
                root / "full.jsonl",
                [{"audio_key": entry["cache_audio_key"]} for entry in reversed(manifest["selected"])],
            )
            output_path = root / "subset.jsonl"

            exit_code = main(
                [
                    "baseline-subset",
                    "--baseline-jsonl",
                    str(baseline_path),
                    "--manifest",
                    str(manifest_path),
                    "--output-jsonl",
                    str(output_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(_read_jsonl(output_path)), 100)


def _corpus() -> tuple[list[dict[str, object]], set[str], set[str]]:
    rows: list[dict[str, object]] = []
    for index in range(80):
        rows.append(_label_row(f"pilot-{index:03d}", LABEL_STABLE))
    pilot_keys = {str(row["source"]["cache_audio_key"]) for row in rows}

    # The first three ramp rows model the protocol-breached v1 rows. The four
    # remaining ramps are the complete unexposed v2 ramp-audit population.
    for index in range(7):
        rows.append(
            _label_row(
                f"ramp-{index:03d}",
                LABEL_RAMP_CANDIDATE,
                long_track=index == 2,
            )
        )
    protocol_keys = {
        str(row["source"]["cache_audio_key"])
        for row in rows[80:83]
    }
    for index in range(10):
        rows.append(
            _label_row(
                f"anomaly-{index:03d}",
                LABEL_AMBIGUOUS,
                long_track=index == 0,
            )
        )
    for index in range(11):
        # Dense/jump/stable labels are all eligible for the earlier long bucket.
        stratum = (LABEL_DENSE, LABEL_JUMP_CANDIDATE, LABEL_STABLE)[index % 3]
        rows.append(_label_row(f"long-{index:03d}", stratum, long_track=True))
    for index in range(10):
        rows.append(_label_row(f"dense-{index:03d}", LABEL_DENSE))
    for index in range(25):
        rows.append(_label_row(f"jump-{index:03d}", LABEL_JUMP_CANDIDATE))
    for index in range(40):
        rows.append(_label_row(f"stable-{index:03d}", LABEL_STABLE))
    # The added pool leaves at least 400 rows after pilot and holdout exclusion.
    for index in range(430):
        rows.append(_label_row(f"extra-{index:03d}", LABEL_STABLE))
    return rows, pilot_keys, protocol_keys


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


def _holdout_entry(row: dict[str, object], *, quota: str) -> dict[str, object]:
    cache_audio_key = str(row["source"]["cache_audio_key"])
    return {
        "stage": "holdout100",
        "quota_assignment": quota,
        "selection_rank": 1,
        "selection_hash_sha256": hashlib.sha256(
            f"{EXP003_HOLDOUT_SEED}\0{cache_audio_key}".encode("utf-8")
        ).hexdigest(),
        "cache_audio_key": cache_audio_key,
        "audio_group_key": row["audio_group_key"],
        "resolved_audio_path": row["resolved_audio_path"],
        "label_stratum": row["label"]["stratum"],
        "source_long_track": row["source"]["long_track"],
    }


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refingerprint(manifest: dict[str, object]) -> None:
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
