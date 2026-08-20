from __future__ import annotations

import ast
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from pulsefield_model.timing.evaluation import exp004_protocol as protocol


def test_protocol_module_imports_only_stdlib_and_no_evaluation_transitives() -> None:
    source_path = Path(protocol.__file__ or "")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.add(node.module or "")

    forbidden_tokens = {
        "pulsefield_model",
        "labels",
        "evidence",
        "hitobjects",
        "object",
        "redline",
        "requests",
        "socket",
        "urllib",
    }
    assert not any(
        token in module
        for module in imported_modules
        for token in forbidden_tokens
    )
    for module in imported_modules:
        root = module.split(".", 1)[0]
        assert root == "__future__" or root in sys.stdlib_module_names


def test_builder_loaders_and_reconcile_create_runner_compatible_inputs(tmp_path: Path) -> None:
    identities = _identities(protocol.STAGE_AUDIO_COUNTS["repair80"])
    upstream = _upstream(tmp_path, identities)

    manifest = protocol.build_exp004_execution_inputs(
        stage="repair80",
        ordered_identities=identities,
        upstream_source=upstream,
        identity_rows_jsonl_path=tmp_path / "identity.jsonl",
        execution_selection_manifest_path=tmp_path / "selection.json",
    )

    loaded_identities, identity_source = protocol.load_exp004_identity_rows(
        tmp_path / "identity.jsonl",
        expected_stage="repair80",
    )
    loaded_manifest, entries, selection_source = (
        protocol.load_exp004_execution_selection_manifest(
            tmp_path / "selection.json",
            expected_stage="repair80",
        )
    )
    protocol.reconcile_exp004_execution_inputs(
        loaded_identities,
        entries,
        expected_stage="repair80",
    )

    assert manifest == loaded_manifest
    assert manifest["schema"] == protocol.EXECUTION_SELECTION_MANIFEST_SCHEMA
    assert manifest["stage_audio_count"] == 80
    assert protocol.exp004_prior_stage("repair80") is None
    assert protocol.exp004_prior_stage("holdout100") == "repair80"
    assert identity_source["row_count"] == 80
    assert selection_source["row_count"] == 80
    expected_constraints = {
        "schema": protocol.STAGE_CONSTRAINT_SCHEMA,
        "stage": "repair80",
        "quota_degraded": False,
        "degraded_quotas": [],
        "broad_underfilled": False,
    }
    assert manifest["source"]["stage_constraints"] == expected_constraints
    assert selection_source["stage_constraints"] == expected_constraints

    raw_identity_rows = _read_jsonl(tmp_path / "identity.jsonl")
    assert set(raw_identity_rows[0]) == protocol.IDENTITY_ROW_FIELDS
    assert "resolved_audio_path" not in raw_identity_rows[0]
    assert manifest["selected"][0] == {
        "cache_audio_key": identities[0]["cache_audio_key"],
        "audio_group_key": identities[0]["audio_group_key"],
    }
    json.dumps(manifest, allow_nan=False)


def test_tamper_reorder_duplicate_count_and_provenance_fail_closed(tmp_path: Path) -> None:
    identities = _identities(protocol.STAGE_AUDIO_COUNTS["repair80"])
    upstream = _upstream(tmp_path, identities)
    protocol.build_exp004_execution_inputs(
        stage="repair80",
        ordered_identities=identities,
        upstream_source=upstream,
        identity_rows_jsonl_path=tmp_path / "identity.jsonl",
        execution_selection_manifest_path=tmp_path / "selection.json",
    )

    rows = _read_jsonl(tmp_path / "identity.jsonl")
    swapped_rows = [rows[1], rows[0], *rows[2:]]
    _write_jsonl(tmp_path / "identity_swapped.jsonl", swapped_rows)
    loaded_identities, _ = protocol.load_exp004_identity_rows(
        tmp_path / "identity_swapped.jsonl",
        expected_stage="repair80",
    )
    _manifest, entries, _ = protocol.load_exp004_execution_selection_manifest(
        tmp_path / "selection.json",
        expected_stage="repair80",
    )
    with pytest.raises(ValueError, match="cache key mismatch"):
        protocol.reconcile_exp004_execution_inputs(
            loaded_identities,
            entries,
            expected_stage="repair80",
        )

    duplicate_rows = list(rows)
    duplicate_rows[1] = dict(duplicate_rows[1])
    duplicate_rows[1]["cache_audio_key"] = duplicate_rows[0]["cache_audio_key"]
    _write_jsonl(tmp_path / "identity_duplicate.jsonl", duplicate_rows)
    with pytest.raises(ValueError, match="duplicate identity cache_audio_key"):
        protocol.load_exp004_identity_rows(
            tmp_path / "identity_duplicate.jsonl",
            expected_stage="repair80",
        )

    _write_jsonl(tmp_path / "identity_short.jsonl", rows[:-1])
    with pytest.raises(ValueError, match="exactly 80"):
        protocol.load_exp004_identity_rows(
            tmp_path / "identity_short.jsonl",
            expected_stage="repair80",
        )

    tampered = _read_json(tmp_path / "selection.json")
    tampered["source"]["upstream"]["ordered_cache_audio_keys_sha256"] = "0" * 64
    _refingerprint(tampered)
    _write_json(tmp_path / "selection_bad_provenance.json", tampered)
    with pytest.raises(ValueError, match="upstream ordered key hash"):
        protocol.load_exp004_execution_selection_manifest(
            tmp_path / "selection_bad_provenance.json",
            expected_stage="repair80",
        )

    nonfinite = _read_json(tmp_path / "selection.json")
    nonfinite["selected_audio_count"] = float("nan")
    (tmp_path / "selection_nan.json").write_text(
        json.dumps(nonfinite, allow_nan=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite"):
        protocol.load_exp004_execution_selection_manifest(
            tmp_path / "selection_nan.json",
            expected_stage="repair80",
        )

    tampered_constraints = _read_json(tmp_path / "selection.json")
    tampered_constraints["source"]["stage_constraints"]["quota_degraded"] = True
    _refingerprint(tampered_constraints)
    _write_json(tmp_path / "selection_bad_constraints.json", tampered_constraints)
    with pytest.raises(ValueError, match="quota_degraded"):
        protocol.load_exp004_execution_selection_manifest(
            tmp_path / "selection_bad_constraints.json",
            expected_stage="repair80",
        )


def test_selection_source_provenance_carries_degraded_stage_constraints(
    tmp_path: Path,
) -> None:
    identities = _identities(protocol.STAGE_AUDIO_COUNTS["holdout100"])
    upstream = _upstream(tmp_path, identities)
    constraints = {
        "schema": protocol.STAGE_CONSTRAINT_SCHEMA,
        "stage": "holdout100",
        "quota_degraded": True,
        "degraded_quotas": ["ramp_audit"],
        "broad_underfilled": False,
    }

    manifest = protocol.build_exp004_execution_inputs(
        stage="holdout100",
        ordered_identities=identities,
        upstream_source=upstream,
        stage_constraints=constraints,
        identity_rows_jsonl_path=tmp_path / "holdout_identity.jsonl",
        execution_selection_manifest_path=tmp_path / "holdout_selection.json",
    )
    _loaded_manifest, _entries, selection_source = (
        protocol.load_exp004_execution_selection_manifest(
            tmp_path / "holdout_selection.json",
            expected_stage="holdout100",
        )
    )

    assert manifest["source"]["stage_constraints"] == constraints
    assert selection_source["stage_constraints"] == constraints


def test_loader_source_sha_is_bound_to_parsed_identity_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities = _identities(protocol.STAGE_AUDIO_COUNTS["repair80"])
    upstream = _upstream(tmp_path, identities)
    protocol.build_exp004_execution_inputs(
        stage="repair80",
        ordered_identities=identities,
        upstream_source=upstream,
        identity_rows_jsonl_path=tmp_path / "identity.jsonl",
        execution_selection_manifest_path=tmp_path / "selection.json",
    )
    identity_path = tmp_path / "identity.jsonl"
    parsed_bytes = identity_path.read_bytes()
    parsed_sha256 = hashlib.sha256(parsed_bytes).hexdigest()
    current_reader = protocol._read_file_bytes_bound

    def mutating_reader(path: Path) -> tuple[bytes, str]:
        data, sha256 = current_reader(path)
        if path == identity_path:
            rows = _read_jsonl(identity_path)
            rows[0], rows[1] = rows[1], rows[0]
            _write_jsonl(identity_path, rows)
        return data, sha256

    monkeypatch.setattr(protocol, "_read_file_bytes_bound", mutating_reader)

    loaded_identities, identity_source = protocol.load_exp004_identity_rows(
        identity_path,
        expected_stage="repair80",
    )

    assert loaded_identities[0].cache_audio_key == identities[0]["cache_audio_key"]
    assert identity_source["sha256"] == parsed_sha256
    assert protocol.file_sha256(identity_path) != identity_source["sha256"]


def test_loader_source_sha_is_bound_to_parsed_selection_manifest_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities = _identities(protocol.STAGE_AUDIO_COUNTS["repair80"])
    upstream = _upstream(tmp_path, identities)
    protocol.build_exp004_execution_inputs(
        stage="repair80",
        ordered_identities=identities,
        upstream_source=upstream,
        identity_rows_jsonl_path=tmp_path / "identity.jsonl",
        execution_selection_manifest_path=tmp_path / "selection.json",
    )
    selection_path = tmp_path / "selection.json"
    parsed_bytes = selection_path.read_bytes()
    parsed_manifest = json.loads(parsed_bytes.decode("utf-8"))
    parsed_sha256 = hashlib.sha256(parsed_bytes).hexdigest()
    current_reader = protocol._read_file_bytes_bound

    def mutating_reader(path: Path) -> tuple[bytes, str]:
        data, sha256 = current_reader(path)
        if path == selection_path:
            selection_path.write_text('{"mutated":true}\n', encoding="utf-8")
        return data, sha256

    monkeypatch.setattr(protocol, "_read_file_bytes_bound", mutating_reader)

    loaded_manifest, entries, selection_source = (
        protocol.load_exp004_execution_selection_manifest(
            selection_path,
            expected_stage="repair80",
        )
    )

    assert loaded_manifest == parsed_manifest
    assert entries[0].cache_audio_key == identities[0]["cache_audio_key"]
    assert selection_source["sha256"] == parsed_sha256
    assert protocol.file_sha256(selection_path) != selection_source["sha256"]


def test_metric_oracle_fields_path_aliases_and_overwrite_different_bytes_rejected(
    tmp_path: Path,
) -> None:
    identities = _identities(protocol.STAGE_AUDIO_COUNTS["repair80"])
    upstream = _upstream(tmp_path, identities)
    poisoned = [dict(item) for item in identities]
    poisoned[0]["oracle_metric"] = 0.25
    with pytest.raises(ValueError, match="metric/oracle-valued field"):
        protocol.build_exp004_execution_inputs(
            stage="repair80",
            ordered_identities=poisoned,
            upstream_source=upstream,
            identity_rows_jsonl_path=tmp_path / "poisoned_identity.jsonl",
            execution_selection_manifest_path=tmp_path / "poisoned_selection.json",
        )

    with pytest.raises(ValueError, match="aliases"):
        protocol.build_exp004_execution_inputs(
            stage="repair80",
            ordered_identities=identities,
            upstream_source=upstream,
            identity_rows_jsonl_path=tmp_path / "same.json",
            execution_selection_manifest_path=tmp_path / "same.json",
        )

    protocol.build_exp004_execution_inputs(
        stage="repair80",
        ordered_identities=identities,
        upstream_source=upstream,
        identity_rows_jsonl_path=tmp_path / "identity.jsonl",
        execution_selection_manifest_path=tmp_path / "selection.json",
    )
    changed = list(reversed(identities))
    changed_upstream = _upstream(tmp_path, changed, name="source_changed.json")
    with pytest.raises(ValueError, match="immutable output already exists"):
        protocol.build_exp004_execution_inputs(
            stage="repair80",
            ordered_identities=changed,
            upstream_source=changed_upstream,
            identity_rows_jsonl_path=tmp_path / "identity.jsonl",
            execution_selection_manifest_path=tmp_path / "selection.json",
        )


def test_upstream_source_sha_and_build_time_bytes_must_match(tmp_path: Path) -> None:
    identities = _identities(protocol.STAGE_AUDIO_COUNTS["repair80"])
    source_path = tmp_path / "source.json"
    _write_json(
        source_path,
        {
            "schema": "synthetic_exp004_source_v1",
            "selected_cache_audio_keys": [identity["cache_audio_key"] for identity in identities],
        },
    )
    ordered_hash = protocol.ordered_cache_audio_keys_sha256(
        [identity["cache_audio_key"] for identity in identities]
    )
    with pytest.raises(ValueError, match="actual file bytes"):
        protocol.build_exp004_upstream_source(
            source_schema="synthetic_exp004_source_v1",
            source_path=source_path,
            source_sha256="0" * 64,
            source_fingerprint_sha256=protocol.file_sha256(source_path),
            row_count=len(identities),
            ordered_cache_audio_keys_sha256=ordered_hash,
        )

    upstream = protocol.build_exp004_upstream_source(
        source_schema="synthetic_exp004_source_v1",
        source_path=source_path,
        source_fingerprint_sha256=protocol.file_sha256(source_path),
        row_count=len(identities),
        ordered_cache_audio_keys_sha256=ordered_hash,
    )
    _write_json(source_path, {"schema": "synthetic_exp004_source_v1", "mutated": True})
    with pytest.raises(ValueError, match="upstream source bytes"):
        protocol.build_exp004_execution_inputs(
            stage="repair80",
            ordered_identities=identities,
            upstream_source=upstream,
            identity_rows_jsonl_path=tmp_path / "identity.jsonl",
            execution_selection_manifest_path=tmp_path / "selection.json",
        )


def test_immutable_builder_accepts_exact_concurrent_replay(tmp_path: Path) -> None:
    identities = _identities(protocol.STAGE_AUDIO_COUNTS["repair80"])
    upstream = _upstream(tmp_path, identities)
    identity_path = tmp_path / "identity.jsonl"
    selection_path = tmp_path / "selection.json"

    def build_once() -> str:
        manifest = protocol.build_exp004_execution_inputs(
            stage="repair80",
            ordered_identities=identities,
            upstream_source=upstream,
            identity_rows_jsonl_path=identity_path,
            execution_selection_manifest_path=selection_path,
        )
        return manifest["manifest_fingerprint_sha256"]

    with ThreadPoolExecutor(max_workers=4) as executor:
        fingerprints = list(executor.map(lambda _index: build_once(), range(4)))

    assert len(set(fingerprints)) == 1
    loaded_identities, _ = protocol.load_exp004_identity_rows(
        identity_path,
        expected_stage="repair80",
    )
    _manifest, entries, _ = protocol.load_exp004_execution_selection_manifest(
        selection_path,
        expected_stage="repair80",
    )
    protocol.reconcile_exp004_execution_inputs(
        loaded_identities,
        entries,
        expected_stage="repair80",
    )


def _identities(count: int) -> list[dict[str, str]]:
    return [
        {
            "cache_audio_key": f"cache::{index:04d}",
            "audio_group_key": f"audio/{index:04d}",
            "resolved_audio_path": f"/dataset/audio/{index:04d}.mp3",
        }
        for index in range(count)
    ]


def _upstream(
    root: Path,
    identities: list[dict[str, str]],
    *,
    name: str = "source.json",
) -> dict[str, Any]:
    ordered_keys = [identity["cache_audio_key"] for identity in identities]
    payload = {
        "schema": "synthetic_exp004_source_v1",
        "selected_cache_audio_keys": ordered_keys,
    }
    source_path = root / name
    _write_json(source_path, payload)
    return protocol.build_exp004_upstream_source(
        source_schema=payload["schema"],
        source_path=source_path,
        source_fingerprint_sha256=protocol.stable_json_sha256(payload),
        row_count=len(identities),
        ordered_cache_audio_keys_sha256=protocol.ordered_cache_audio_keys_sha256(
            ordered_keys
        ),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _refingerprint(manifest: dict[str, Any]) -> None:
    manifest.pop("manifest_fingerprint_sha256", None)
    manifest["manifest_fingerprint_sha256"] = protocol.stable_json_sha256(manifest)
