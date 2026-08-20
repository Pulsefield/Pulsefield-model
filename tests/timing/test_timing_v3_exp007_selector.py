from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

import pytest

from pulsefield_model.timing.evaluation import exp007_protocol as protocol
from pulsefield_model.timing.evaluation import exp007_selector as selector


def test_selector_manifest_replays_frozen_hashes_and_long_overlap_priority() -> None:
    identity_rows, label_rows, identity_source, label_source = _fixture(
        [
            ("overlap-dense", "dense", True),
            ("overlap-jump", "jump_candidate", True),
            ("overlap-stable", "stable", True),
            ("long-pure", "ambiguous", True),
            ("dense-a", "dense", False),
            ("dense-b", "dense", False),
            ("dense-c", "dense", False),
            ("dense-d", "dense", False),
            ("jump-a", "jump_candidate", False),
            ("jump-b", "jump_candidate", False),
            ("jump-c", "jump_candidate", False),
            ("jump-d", "jump_candidate", False),
        ]
    )

    manifest = selector.make_selector_manifest(
        repair80_identity_rows=identity_rows,
        label_rows=label_rows,
        source_repair80_identity=identity_source,
        source_labels=label_source,
    )

    assert selector.validate_selector_manifest(
        manifest,
        repair80_identity_rows=identity_rows,
        label_rows=label_rows,
        source_repair80_identity=identity_source,
        source_labels=label_source,
    ) == manifest
    assert manifest["schema"] == selector.SELECTOR_MANIFEST_SCHEMA
    assert manifest["seed"] == protocol.EXP007_SELECTOR_SEED
    assert manifest["selected_count"] == 16
    assert manifest["manifest_fingerprint_sha256"] == protocol.payload_hash(
        manifest,
        "manifest_fingerprint_sha256",
    )

    selected_keys = [entry["cache_audio_key"] for entry in manifest["selected"]]
    assert manifest["selected_cache_audio_keys_sha256"] == protocol.canonical_json_sha256(
        sorted(selected_keys)
    )
    assert manifest["selected_ordered_cache_audio_keys_sha256"] == (
        protocol.canonical_json_sha256(selected_keys)
    )
    assert manifest["selected_ordered_entries_sha256"] == protocol.canonical_json_sha256(
        manifest["selected"]
    )

    by_key = {row["cache_audio_key"]: row for row in identity_rows}
    overlap_keys = {"overlap-dense", "overlap-jump", "overlap-stable", "long-pure"}
    assert {selector.class_of(by_key[key]) for key in overlap_keys} == {"long"}
    overlap_entries = [
        entry for entry in manifest["selected"] if entry["cache_audio_key"] in overlap_keys
    ]
    assert len(overlap_entries) == 4
    assert {entry["bucket"] for entry in overlap_entries} == {"long"}
    assert {entry["selection_substage"] for entry in overlap_entries} == {"long_quota"}
    assert all(
        entry["source_row_index"] == by_key[entry["cache_audio_key"]]["row_index"]
        for entry in manifest["selected"]
    )
    assert all(
        entry["source_row_index"] != by_key[entry["cache_audio_key"]]["source_row_index"]
        for entry in manifest["selected"]
    )


def test_selector_deficit_donors_are_fixed_order_and_ranked_per_substage() -> None:
    identity_rows, label_rows, identity_source, label_source = _fixture(
        [
            ("long-a", "stable", True),
            ("long-b", "ambiguous", True),
            ("dense-a", "dense", False),
            ("jump-a", "jump_candidate", False),
            ("jump-b", "jump_candidate", False),
            ("jump-c", "jump_candidate", False),
        ]
    )

    manifest = selector.make_selector_manifest(
        repair80_identity_rows=identity_rows,
        label_rows=label_rows,
        source_repair80_identity=identity_source,
        source_labels=label_source,
    )

    counts = {row["bucket"]: row for row in manifest["bucket_counts"]}
    assert counts == {
        "long": {"bucket": "long", "requested": 4, "available": 2, "selected": 2, "deficit": 2},
        "dense": {"bucket": "dense", "requested": 4, "available": 1, "selected": 0, "deficit": 4},
        "jump": {"bucket": "jump", "requested": 4, "available": 3, "selected": 0, "deficit": 4},
        "stable": {"bucket": "stable", "requested": 4, "available": 74, "selected": 4, "deficit": 0},
    }
    assert manifest["deficit_count"] == 10
    assert Counter(entry["selection_substage"] for entry in manifest["selected"]) == {
        "long_quota": 2,
        "long_deficit_from_dense": 1,
        "long_deficit_from_jump": 1,
        "dense_deficit_from_jump": 2,
        "dense_deficit_from_stable": 2,
        "jump_deficit_from_stable": 4,
        "stable_quota": 4,
    }
    for substage in {entry["selection_substage"] for entry in manifest["selected"]}:
        ranks = [
            entry["selection_rank"]
            for entry in manifest["selected"]
            if entry["selection_substage"] == substage
        ]
        assert ranks == list(range(len(ranks)))


def test_selector_manifest_validation_rejects_rehashed_replay_tamper() -> None:
    identity_rows, label_rows, identity_source, label_source = _fixture(
        [
            ("long-a", "stable", True),
            ("long-b", "stable", True),
            ("long-c", "stable", True),
            ("long-d", "stable", True),
            ("dense-a", "dense", False),
            ("dense-b", "dense", False),
            ("dense-c", "dense", False),
            ("dense-d", "dense", False),
            ("jump-a", "jump_candidate", False),
            ("jump-b", "jump_candidate", False),
            ("jump-c", "jump_candidate", False),
            ("jump-d", "jump_candidate", False),
        ]
    )
    manifest = selector.make_selector_manifest(
        repair80_identity_rows=identity_rows,
        label_rows=label_rows,
        source_repair80_identity=identity_source,
        source_labels=label_source,
    )

    tampered = deepcopy(manifest)
    tampered["selected"][0]["bucket"] = "deficit_fill"
    tampered["selected_ordered_entries_sha256"] = protocol.canonical_json_sha256(
        tampered["selected"]
    )
    tampered["deficit_count"] += 1
    tampered["manifest_fingerprint_sha256"] = protocol.payload_hash(
        tampered,
        "manifest_fingerprint_sha256",
    )

    with pytest.raises(ValueError, match="replay mismatch"):
        selector.validate_selector_manifest(
            tampered,
            repair80_identity_rows=identity_rows,
            label_rows=label_rows,
            source_repair80_identity=identity_source,
            source_labels=label_source,
        )


def test_selector_rejects_source_ref_and_label_provenance_tamper() -> None:
    identity_rows, label_rows, identity_source, label_source = _fixture(
        [("long-a", "stable", True)]
    )

    bad_identity_source = dict(identity_source, ordered_rows_sha256="0" * 64)
    with pytest.raises(ValueError, match="ordered_rows_sha256 mismatch"):
        selector.make_selector_manifest(
            repair80_identity_rows=identity_rows,
            label_rows=label_rows,
            source_repair80_identity=bad_identity_source,
            source_labels=label_source,
        )

    bad_label_rows = deepcopy(label_rows)
    bad_label_rows[0]["audit_note"] = "same identity fields but different source row"
    bad_label_source = _source_ref(selector.SOURCE_LABELS_ARTIFACT_SCHEMA, bad_label_rows)
    with pytest.raises(ValueError, match="label_source_sha256 mismatch"):
        selector.make_selector_manifest(
            repair80_identity_rows=identity_rows,
            label_rows=bad_label_rows,
            source_repair80_identity=identity_source,
            source_labels=bad_label_source,
        )


def test_selector_distinguishes_artifact_bytes_sha_from_ordered_rows_sha() -> None:
    identity_rows, label_rows, _, _ = _fixture([("long-a", "stable", True)])
    identity_artifact_sha = protocol.canonical_json_sha256(
        {"encoding": "jsonl", "rows": identity_rows}
    )
    label_artifact_sha = protocol.canonical_json_sha256(
        {"encoding": "jsonl", "rows": label_rows}
    )
    identity_source = selector.make_source_ref_for_rows(
        artifact_schema=selector.SOURCE_REPAIR80_IDENTITY_ARTIFACT_SCHEMA,
        rows=identity_rows,
        artifact_sha256=identity_artifact_sha,
    )
    label_source = selector.make_source_ref_for_rows(
        artifact_schema=selector.SOURCE_LABELS_ARTIFACT_SCHEMA,
        rows=label_rows,
        artifact_sha256=label_artifact_sha,
    )

    assert identity_source["sha256"] != identity_source["ordered_rows_sha256"]
    assert label_source["sha256"] != label_source["ordered_rows_sha256"]
    manifest = selector.make_selector_manifest(
        repair80_identity_rows=identity_rows,
        label_rows=label_rows,
        source_repair80_identity=identity_source,
        source_labels=label_source,
    )
    assert manifest["source_repair80_identity"] == identity_source
    assert manifest["source_labels"] == label_source


def test_repair80_identity_label_sources_require_authoritative_artifact_bytes() -> None:
    identity_rows, label_rows, identity_source, label_source = _fixture(
        [("long-a", "stable", True)]
    )
    identity_bytes = _source_artifact_bytes(
        selector.SOURCE_REPAIR80_IDENTITY_ARTIFACT_SCHEMA,
        identity_rows,
    )
    label_bytes = _source_artifact_bytes(
        selector.SOURCE_LABELS_ARTIFACT_SCHEMA,
        label_rows,
    )

    validated = selector.validate_repair80_identity_label_sources(
        repair80_identity_source_artifact=identity_bytes,
        label_source_artifact=label_bytes,
        repair80_identity_rows=identity_rows,
        label_rows=label_rows,
        source_repair80_identity=identity_source,
        source_labels=label_source,
    )
    assert len(validated) == 80

    with pytest.raises(ValueError, match="immutable canonical bytes"):
        selector.validate_repair80_identity_label_sources(
            repair80_identity_source_artifact={
                "schema": selector.SOURCE_REPAIR80_IDENTITY_ARTIFACT_SCHEMA,
                "rows": identity_rows,
            },
            label_source_artifact=label_bytes,
            repair80_identity_rows=identity_rows,
            label_rows=label_rows,
            source_repair80_identity=identity_source,
            source_labels=label_source,
        )

    with pytest.raises(ValueError, match="immutable canonical bytes"):
        selector.validate_repair80_identity_label_sources(
            repair80_identity_source_artifact=bytearray(identity_bytes),
            label_source_artifact=label_bytes,
            repair80_identity_rows=identity_rows,
            label_rows=label_rows,
            source_repair80_identity=identity_source,
            source_labels=label_source,
        )

    bad_source = dict(identity_source, sha256="0" * 64)
    with pytest.raises(ValueError, match="source_repair80_identity.sha256 mismatch"):
        selector.validate_repair80_identity_label_sources(
            repair80_identity_source_artifact=identity_bytes,
            label_source_artifact=label_bytes,
            repair80_identity_rows=identity_rows,
            label_rows=label_rows,
            source_repair80_identity=bad_source,
            source_labels=label_source,
        )


def test_selector_manifest_authoritative_validation_rejects_self_attested_sources() -> None:
    identity_rows, label_rows, identity_source, label_source = _fixture(
        [("long-a", "stable", True)]
    )
    identity_bytes = _source_artifact_bytes(
        selector.SOURCE_REPAIR80_IDENTITY_ARTIFACT_SCHEMA,
        identity_rows,
    )
    label_bytes = _source_artifact_bytes(
        selector.SOURCE_LABELS_ARTIFACT_SCHEMA,
        label_rows,
    )
    manifest = selector.make_selector_manifest(
        repair80_identity_rows=identity_rows,
        label_rows=label_rows,
        source_repair80_identity=identity_source,
        source_labels=label_source,
    )
    assert (
        selector.validate_selector_manifest_authoritatively(
            manifest,
            repair80_identity_source_artifact_bytes=identity_bytes,
            label_source_artifact_bytes=label_bytes,
        )
        == manifest
    )

    bad_source = dict(identity_source, sha256="0" * 64)
    self_attested = selector.make_selector_manifest(
        repair80_identity_rows=identity_rows,
        label_rows=label_rows,
        source_repair80_identity=bad_source,
        source_labels=label_source,
    )
    with pytest.raises(ValueError, match="source_repair80_identity.sha256 mismatch"):
        selector.validate_selector_manifest_authoritatively(
            self_attested,
            repair80_identity_source_artifact_bytes=identity_bytes,
            label_source_artifact_bytes=label_bytes,
        )

    same_rows_wrapper = {
        "schema": selector.SOURCE_REPAIR80_IDENTITY_ARTIFACT_SCHEMA,
        "rows": identity_rows,
        "wrapper": "alternate",
    }
    wrapper_source = dict(
        identity_source,
        sha256=protocol.canonical_json_sha256(same_rows_wrapper),
    )
    wrapper_manifest = selector.make_selector_manifest(
        repair80_identity_rows=identity_rows,
        label_rows=label_rows,
        source_repair80_identity=wrapper_source,
        source_labels=label_source,
    )
    with pytest.raises(ValueError, match="incomplete or unsupported"):
        selector.validate_selector_manifest_authoritatively(
            wrapper_manifest,
            repair80_identity_source_artifact_bytes=protocol.canonical_json_bytes(
                same_rows_wrapper
            ),
            label_source_artifact_bytes=label_bytes,
        )

    with pytest.raises(ValueError, match="immutable canonical bytes"):
        selector.validate_selector_manifest_authoritatively(
            manifest,
            repair80_identity_source_artifact_bytes=bytearray(identity_bytes),
            label_source_artifact_bytes=label_bytes,
        )


def test_selector_recursively_rejects_forbidden_source_only_fields() -> None:
    identity_rows, label_rows, identity_source, _ = _fixture([("long-a", "stable", True)])
    bad_label_rows = deepcopy(label_rows)
    bad_label_rows[0]["nested"] = {"metrics": {"phase_error_ms": 12.0}}
    bad_label_source = _source_ref(selector.SOURCE_LABELS_ARTIFACT_SCHEMA, bad_label_rows)

    with pytest.raises(ValueError, match="forbidden selector field"):
        selector.make_selector_manifest(
            repair80_identity_rows=identity_rows,
            label_rows=bad_label_rows,
            source_repair80_identity=identity_source,
            source_labels=bad_label_source,
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "osu_path",
        "red_line",
        "redLine",
        "hit_objects",
        "beatmap_osu",
        "objectPlacement",
        "objectgrid",
        "boundaryscore",
    ],
)
def test_selector_rejects_forbidden_source_alias_fields_nested(
    field_name: str,
) -> None:
    identity_rows, label_rows, identity_source, _ = _fixture([("long-a", "stable", True)])
    bad_label_rows = deepcopy(label_rows)
    bad_label_rows[0]["nested"] = {"audit": [{field_name: "source-only leakage"}]}
    bad_label_source = _source_ref(selector.SOURCE_LABELS_ARTIFACT_SCHEMA, bad_label_rows)

    with pytest.raises(ValueError, match="forbidden selector field"):
        selector.make_selector_manifest(
            repair80_identity_rows=identity_rows,
            label_rows=bad_label_rows,
            source_repair80_identity=identity_source,
            source_labels=bad_label_source,
        )


def test_selector_allows_legitimate_identity_label_schema_field_names() -> None:
    identity_rows, label_rows, _, _ = _fixture([("long-a", "stable", True)])
    label_rows = deepcopy(label_rows)
    label_rows[0]["legitimate_projection_fields"] = {
        "cache_audio_key": "ignored-cache-key",
        "audio_group_key": "ignored-group-key",
        "label_stratum": "stable",
        "source_long_track": True,
        "duration_ms": 120_000,
        "label_source_sha256": "a" * 64,
        "source": {"cache_audio_key": "ignored-cache-key", "long_track": True},
        "label": {"stratum": "stable"},
    }
    label_zero = label_rows[0]
    identity_rows = list(identity_rows)
    identity_rows[0] = protocol.make_identity(
        stage=protocol.EXP007_REPAIR_STAGE,
        row_index=0,
        source_row_index=10_000,
        cache_audio_key=label_zero["cache_audio_key"],
        audio_group_key=label_zero["audio_group_key"],
        label_stratum=label_zero["label_stratum"],
        source_long_track=label_zero["source_long_track"],
        duration_ms=label_zero["duration_ms"],
        label_source_sha256=protocol.canonical_json_sha256(label_zero),
    )
    identity_source = _source_ref(
        selector.SOURCE_REPAIR80_IDENTITY_ARTIFACT_SCHEMA,
        identity_rows,
    )
    label_source = _source_ref(selector.SOURCE_LABELS_ARTIFACT_SCHEMA, label_rows)

    manifest = selector.make_selector_manifest(
        repair80_identity_rows=identity_rows,
        label_rows=label_rows,
        source_repair80_identity=identity_source,
        source_labels=label_source,
    )
    assert manifest["selected_count"] == selector.SELECTOR_STAGE_ROW_COUNT


def _fixture(
    specs: list[tuple[str, str, bool]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    filled_specs = list(specs)
    filler_index = 0
    while len(filled_specs) < selector.REPAIR80_SOURCE_ROW_COUNT:
        key = f"stable-fill-{filler_index:03d}"
        if all(existing_key != key for existing_key, _, _ in filled_specs):
            filled_specs.append((key, "stable", False))
        filler_index += 1

    label_rows = [
        _label_row(
            key=key,
            group=f"group-{index % 7}",
            label=label,
            long_track=long_track,
            duration_ms=120_000 + index,
        )
        for index, (key, label, long_track) in enumerate(filled_specs)
    ]
    identity_rows = [
        protocol.make_identity(
            stage=protocol.EXP007_REPAIR_STAGE,
            row_index=index,
            source_row_index=10_000 + index,
            cache_audio_key=label_row["cache_audio_key"],
            audio_group_key=label_row["audio_group_key"],
            label_stratum=label_row["label_stratum"],
            source_long_track=label_row["source_long_track"],
            duration_ms=label_row["duration_ms"],
            label_source_sha256=protocol.canonical_json_sha256(label_row),
        )
        for index, label_row in enumerate(label_rows)
    ]
    return (
        identity_rows,
        label_rows,
        _source_ref(selector.SOURCE_REPAIR80_IDENTITY_ARTIFACT_SCHEMA, identity_rows),
        _source_ref(selector.SOURCE_LABELS_ARTIFACT_SCHEMA, label_rows),
    )


def _label_row(
    *,
    key: str,
    group: str,
    label: str,
    long_track: bool,
    duration_ms: int,
) -> dict[str, Any]:
    return {
        "cache_audio_key": key,
        "audio_group_key": group,
        "label_stratum": label,
        "source_long_track": long_track,
        "duration_ms": duration_ms,
        "source": {"cache_audio_key": key, "long_track": long_track},
        "label": {"stratum": label},
    }


def _source_ref(artifact_schema: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return selector.make_source_ref_for_rows(artifact_schema=artifact_schema, rows=rows)


def _source_artifact_bytes(artifact_schema: str, rows: list[dict[str, Any]]) -> bytes:
    return protocol.canonical_json_bytes(
        {
            "schema": artifact_schema,
            "rows": [dict(row) for row in rows],
        }
    )
