from __future__ import annotations

import ast
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from pulsefield_model.timing.evaluation import exp007_protocol as protocol
from pulsefield_model.timing.v3 import global_constant_jump as candidate_source


SHA0 = "0" * 64
SHA1 = "1" * 64
SHA2 = "2" * 64
SHA3 = "3" * 64
SHA4 = "4" * 64
SHA5 = "5" * 64
SHA6 = "6" * 64
SHA7 = "7" * 64
SHA8 = "8" * 64
SHA9 = "9" * 64
SHAA = "a" * 64
SHAB = "b" * 64
SHAC = "c" * 64
SHAD = "d" * 64


def test_protocol_imports_only_stdlib() -> None:
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
        "artifacts",
        "cache",
        "hitobject",
        "labels",
        "metrics",
        "osu",
        "redline",
        "requests",
        "socket",
        "urllib",
        "weak",
    }
    assert not any(
        token in module
        for module in imported_modules
        for token in forbidden_tokens
    )
    for module in imported_modules:
        root = module.split(".", 1)[0]
        assert root == "__future__" or root in sys.stdlib_module_names


def test_strict_json_and_canonical_hashes_fail_closed() -> None:
    first = {"b": 2, "a": [1, True, None]}
    second = {"a": [1, True, None], "b": 2}
    assert protocol.canonical_json_bytes(first) == protocol.canonical_json_bytes(second)
    assert protocol.canonical_json_sha256(first) == protocol.canonical_json_sha256(second)

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        protocol.load_json_strict(b'{"x":1,"x":2}')
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        protocol.load_json_strict(json.dumps({"x": float("nan")}, allow_nan=True))
    with pytest.raises(ValueError, match="non-finite floats"):
        protocol.canonical_json_bytes({"x": float("inf")})

    payload = protocol.with_payload_hash({"schema": "x", "value": 1}, "payload_sha256")
    protocol.validate_payload_hash(payload, "payload_sha256", context="Payload")
    tampered = dict(payload, value=2)
    with pytest.raises(ValueError, match="payload_sha256 mismatch"):
        protocol.validate_payload_hash(tampered, "payload_sha256", context="Payload")


def test_schema_descriptors_are_recursive_source_owned_and_mutation_sensitive() -> None:
    row_descriptor = protocol.schema_descriptor_payload(protocol.ROW_RESULT_SCHEMA)
    assert row_descriptor["schema"] == protocol.ROW_RESULT_SCHEMA
    assert "MethodResult" in row_descriptor["nested_descriptors"]
    assert "branches" in row_descriptor["nested_descriptors"]["MethodResult"]
    assert "DenominatorFlags" in row_descriptor["nested_descriptors"]

    base_sha = protocol.schema_descriptor_sha256(protocol.ROW_RESULT_SCHEMA)
    mutated = _clone(row_descriptor)
    mutated["nested_descriptors"]["MethodResult"]["fields"].append("observed_runtime")
    assert protocol.canonical_json_sha256(mutated) != base_sha

    assert protocol.schema_descriptor_sha256(protocol.SELECTOR_MANIFEST_SCHEMA)
    assert protocol.schema_descriptor_sha256(protocol.CANDIDATE_GLOBAL_MANIFEST_SCHEMA)
    assert protocol.schema_descriptor_sha256(protocol.WEAK_ROW_SCHEMA)
    assert protocol.schema_descriptor_sha256(protocol.REPAIR80_SUMMARY_SCHEMA)
    with pytest.raises(ValueError, match="unknown Exp007 schema descriptor"):
        protocol.schema_descriptor_sha256("not.source.owned")


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
def test_selector_forbidden_field_guard_normalizes_alias_spellings(
    field_name: str,
) -> None:
    with pytest.raises(ValueError, match="forbidden selector field"):
        protocol.reject_forbidden_selector_fields(
            {"metadata": [{field_name: "source-only leakage"}]},
            context="label_rows[0]",
        )


def test_selector_forbidden_field_guard_allows_identity_label_schema_keys() -> None:
    protocol.reject_forbidden_selector_fields(
        {
            "schema": protocol.IDENTITY_SCHEMA,
            "stage": protocol.EXP007_REPAIR_STAGE,
            "row_index": 0,
            "source_row_index": 100,
            "cache_audio_key": "audio-000",
            "audio_group_key": "group-000",
            "label_stratum": "stable",
            "source_long_track": False,
            "duration_ms": 120_000,
            "label_source_sha256": SHA1,
            "identity_payload_sha256": SHA2,
            "label": {"stratum": "stable"},
            "source": {"cache_audio_key": "audio-000", "long_track": False},
        },
        context="identity_rows[0]",
    )


def test_identity_and_run_config_branch_contracts() -> None:
    identity = _identity(stage=protocol.EXP007_SCHEDULE_STAGE, row_index=0)
    assert identity["identity_payload_sha256"] == protocol.payload_hash(
        identity,
        "identity_payload_sha256",
    )

    tampered_identity = dict(identity, cache_audio_key="other")
    with pytest.raises(ValueError, match="identity_payload_sha256 mismatch"):
        protocol.validate_identity(tampered_identity)

    schedule_config = protocol.make_run_config(
        stage=protocol.EXP007_SCHEDULE_STAGE,
        schedule_arm="S30",
        selector_manifest_sha256=SHA1,
        input_manifest_sha256=SHA1,
        source_closure_fingerprint_sha256=SHA2,
        cache_config_sha256=SHA3,
        grid_fitter_config_sha256=SHA4,
        weak_config_sha256=SHA5,
    )
    assert schedule_config["schedule_weak_veto_outcome_sha256"] is None

    with pytest.raises(ValueError, match="input manifest must equal selector"):
        protocol.make_run_config(
            stage=protocol.EXP007_SCHEDULE_STAGE,
            schedule_arm="S30",
            selector_manifest_sha256=SHA1,
            input_manifest_sha256=SHA2,
            source_closure_fingerprint_sha256=SHA2,
            cache_config_sha256=SHA3,
            grid_fitter_config_sha256=SHA4,
            weak_config_sha256=SHA5,
        )
    with pytest.raises(ValueError, match="weak outcome SHA must be null"):
        protocol.make_run_config(
            stage=protocol.EXP007_SCHEDULE_STAGE,
            schedule_arm="S30",
            selector_manifest_sha256=SHA1,
            input_manifest_sha256=SHA1,
            schedule_weak_veto_outcome_sha256=SHA9,
            source_closure_fingerprint_sha256=SHA2,
            cache_config_sha256=SHA3,
            grid_fitter_config_sha256=SHA4,
            weak_config_sha256=SHA5,
        )

    repair_config = protocol.make_run_config(
        stage=protocol.EXP007_REPAIR_STAGE,
        schedule_arm="S64",
        selector_manifest_sha256=SHA1,
        input_manifest_sha256=SHA2,
        schedule_weak_veto_outcome_sha256=SHA9,
        source_closure_fingerprint_sha256=SHA3,
        cache_config_sha256=SHA4,
        grid_fitter_config_sha256=SHA5,
        weak_config_sha256=SHA6,
    )
    assert repair_config["input_manifest_sha256"] == SHA2

    with pytest.raises(ValueError, match="repair binding SHA"):
        protocol.make_run_config(
            stage=protocol.EXP007_REPAIR_STAGE,
            schedule_arm="S64",
            selector_manifest_sha256=SHA1,
            input_manifest_sha256=SHA1,
            schedule_weak_veto_outcome_sha256=SHA9,
            source_closure_fingerprint_sha256=SHA3,
            cache_config_sha256=SHA4,
            grid_fitter_config_sha256=SHA5,
            weak_config_sha256=SHA6,
        )


def test_row_result_recursion_hashes_and_byte_cap_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    row = _row_result(0)
    validated = protocol.validate_row_result(row)
    assert validated["cache_identity"]["schema"] == protocol.CACHE_IDENTITY_SCHEMA
    assert validated["methods"]["candidate"]["grid"]["kind"] == "timing_v3"

    placeholder = _clone(row)
    placeholder["cache_identity"] = {"synthetic": True}
    with pytest.raises(ValueError, match="CacheIdentity fields"):
        protocol.validate_row_result(placeholder)

    bad_grid = _clone(row)
    bad_grid["methods"]["candidate"]["grid"]["payload"]["sections"][0]["bpm"] = 10.0
    with pytest.raises(ValueError, match="bpm"):
        protocol.validate_row_result(bad_grid)

    bad_flags = _clone(row)
    bad_flags["denominator_flags"]["pure_exp006_phase_matched"] = True
    with pytest.raises(ValueError, match="pure phase implication"):
        protocol.validate_row_result(bad_flags)

    bad_hash = _clone(row)
    bad_hash["row_payload_sha256"] = SHA0
    with pytest.raises(ValueError, match="row_payload_sha256 mismatch"):
        protocol.validate_row_result(bad_hash)

    monkeypatch.setattr(
        protocol,
        "EXP007_ROW_JSON_BYTE_CAP",
        len(protocol.canonical_json_bytes(row)),
    )
    with pytest.raises(ValueError, match="RowResult at or above byte cap"):
        protocol.validate_row_result(row)


def test_arm_failure_record_reference_rules_and_hashes() -> None:
    completed = [
        protocol.make_completed_row_ref(
            row_index=0,
            cache_audio_key="audio-000",
            identity_payload_sha256=SHA1,
            row_payload_sha256=SHA2,
            candidate_reference_entry_payload_sha256=SHA3,
        ),
        protocol.make_completed_row_ref(
            row_index=1,
            cache_audio_key="audio-001",
            identity_payload_sha256=SHA4,
            row_payload_sha256=SHA5,
            candidate_reference_entry_payload_sha256=SHA6,
        ),
    ]
    pending = [
        protocol.make_pending_identity_ref(
            row_index=index,
            cache_audio_key=f"audio-{index:03d}",
            identity_payload_sha256=_digest(f"identity-{index}"),
        )
        for index in range(2, 16)
    ]
    failure = protocol.make_arm_failure_record(
        stage=protocol.EXP007_SCHEDULE_STAGE,
        schedule_arm="S30",
        run_config_fingerprint_sha256=SHA7,
        source_closure_fingerprint_sha256=SHA8,
        input_manifest_sha256=SHA9,
        failure_kind="row_timeout",
        failure_stage="diagnostics",
        completed_prefix_rows=completed,
        completed_reference_entry_payload_sha256s=[SHA3, SHA6],
        pending_identities=pending,
        causing_row_index=2,
        causing_cache_audio_key="audio-002",
        causing_worker_slot=0,
        causing_worker_generation_nonce=SHAA,
        causing_worker_pid=12345,
        causing_dispatch_token=SHAB,
        causing_worker_rss_bytes=1024,
    )
    protocol.validate_arm_failure_record(failure)

    nondeterministic = dict(failure, causing_worker_pid=99999)
    nondeterministic["full_payload_sha256"] = protocol.payload_hash(
        nondeterministic,
        "full_payload_sha256",
    )
    assert protocol.validate_arm_failure_record(nondeterministic)[
        "failure_deterministic_fingerprint_sha256"
    ] == failure["failure_deterministic_fingerprint_sha256"]

    tampered = dict(failure, prefix_candidate_fallback_count=1)
    tampered["full_payload_sha256"] = protocol.payload_hash(tampered, "full_payload_sha256")
    with pytest.raises(ValueError, match="deterministic fingerprint mismatch"):
        protocol.validate_arm_failure_record(tampered)

    with pytest.raises(ValueError, match="reference-arm completed reference count"):
        protocol.make_arm_failure_record(
            stage=protocol.EXP007_SCHEDULE_STAGE,
            schedule_arm="S30",
            run_config_fingerprint_sha256=SHA7,
            source_closure_fingerprint_sha256=SHA8,
            input_manifest_sha256=SHA9,
            failure_kind="row_timeout",
            failure_stage="diagnostics",
            completed_prefix_rows=completed,
            completed_reference_entry_payload_sha256s=[SHA3],
            pending_identities=pending,
        )

    with pytest.raises(ValueError, match="completed prefix must be contiguous"):
        protocol.make_arm_failure_record(
            stage=protocol.EXP007_SCHEDULE_STAGE,
            schedule_arm="S30",
            run_config_fingerprint_sha256=SHA7,
            source_closure_fingerprint_sha256=SHA8,
            input_manifest_sha256=SHA9,
            failure_kind="row_timeout",
            failure_stage="diagnostics",
            completed_prefix_rows=[completed[0], dict(completed[1], row_index=2)],
            completed_reference_entry_payload_sha256s=[SHA3, SHA6],
            pending_identities=[
                protocol.make_pending_identity_ref(
                    row_index=1,
                    cache_audio_key="audio-001",
                    identity_payload_sha256=SHA4,
                ),
                *pending[1:],
            ],
        )

    parallel_pending_failure = protocol.make_arm_failure_record(
        stage=protocol.EXP007_SCHEDULE_STAGE,
        schedule_arm="S30",
        run_config_fingerprint_sha256=SHA7,
        source_closure_fingerprint_sha256=SHA8,
        input_manifest_sha256=SHA9,
        failure_kind="row_timeout",
        failure_stage="diagnostics",
        completed_prefix_rows=completed,
        completed_reference_entry_payload_sha256s=[SHA3, SHA6],
        pending_identities=pending,
        causing_row_index=3,
        causing_cache_audio_key="audio-003",
        causing_worker_slot=0,
        causing_worker_generation_nonce=SHAA,
        causing_worker_pid=12345,
        causing_dispatch_token=SHAB,
    )
    assert parallel_pending_failure["causing_row_index"] == 3

    committed_duplicate_failure = protocol.make_arm_failure_record(
        stage=protocol.EXP007_SCHEDULE_STAGE,
        schedule_arm="S30",
        run_config_fingerprint_sha256=SHA7,
        source_closure_fingerprint_sha256=SHA8,
        input_manifest_sha256=SHA9,
        failure_kind="broken_stream",
        failure_stage="pool_stream",
        completed_prefix_rows=completed,
        completed_reference_entry_payload_sha256s=[SHA3, SHA6],
        pending_identities=pending,
        causing_row_index=1,
        causing_cache_audio_key="audio-001",
        causing_worker_slot=0,
        causing_worker_generation_nonce=SHAA,
        causing_worker_pid=12345,
        causing_dispatch_token=SHAB,
    )
    protocol.validate_arm_failure_record(committed_duplicate_failure)

    with pytest.raises(ValueError, match="row_timeout causing row must be pending"):
        protocol.make_arm_failure_record(
            stage=protocol.EXP007_SCHEDULE_STAGE,
            schedule_arm="S30",
            run_config_fingerprint_sha256=SHA7,
            source_closure_fingerprint_sha256=SHA8,
            input_manifest_sha256=SHA9,
            failure_kind="row_timeout",
            failure_stage="diagnostics",
            completed_prefix_rows=completed,
            completed_reference_entry_payload_sha256s=[SHA3, SHA6],
            pending_identities=pending,
            causing_row_index=1,
            causing_cache_audio_key="audio-001",
            causing_worker_slot=0,
            causing_worker_generation_nonce=SHAA,
            causing_worker_pid=12345,
            causing_dispatch_token=SHAB,
        )

    with pytest.raises(ValueError, match="causing row must be in completed or pending"):
        protocol.make_arm_failure_record(
            stage=protocol.EXP007_SCHEDULE_STAGE,
            schedule_arm="S30",
            run_config_fingerprint_sha256=SHA7,
            source_closure_fingerprint_sha256=SHA8,
            input_manifest_sha256=SHA9,
            failure_kind="row_timeout",
            failure_stage="diagnostics",
            completed_prefix_rows=completed,
            completed_reference_entry_payload_sha256s=[SHA3, SHA6],
            pending_identities=pending,
            causing_row_index=3,
            causing_cache_audio_key="audio-999",
            causing_worker_slot=0,
            causing_worker_generation_nonce=SHAA,
            causing_worker_pid=12345,
            causing_dispatch_token=SHAB,
        )

    later_completed = [
        dict(row, candidate_reference_entry_payload_sha256=None) for row in completed
    ]
    with pytest.raises(ValueError, match="later schedule arms cannot carry references"):
        protocol.make_arm_failure_record(
            stage=protocol.EXP007_SCHEDULE_STAGE,
            schedule_arm="S60",
            run_config_fingerprint_sha256=SHA7,
            source_closure_fingerprint_sha256=SHA8,
            input_manifest_sha256=SHA9,
            failure_kind="row_timeout",
            failure_stage="diagnostics",
            completed_prefix_rows=later_completed,
            completed_reference_entry_payload_sha256s=[SHA3],
            pending_identities=pending,
        )

    with pytest.raises(ValueError, match="requires diagnostics stage"):
        protocol.make_arm_failure_record(
            stage=protocol.EXP007_SCHEDULE_STAGE,
            schedule_arm="S30",
            run_config_fingerprint_sha256=SHA7,
            source_closure_fingerprint_sha256=SHA8,
            input_manifest_sha256=SHA9,
            failure_kind="diagnostics_integrity_failure",
            failure_stage="candidate",
            completed_prefix_rows=completed,
            completed_reference_entry_payload_sha256s=[SHA3, SHA6],
            pending_identities=pending,
        )


def test_outcomes_config_selection_four_arm_and_repair_binding() -> None:
    success = protocol.make_arm_stage_success(
        stage=protocol.EXP007_SCHEDULE_STAGE,
        schedule_arm="S30",
        row_payloads_sha256=SHA1,
        candidate_reference_manifest_sha256=SHA2,
        stage_summary_sha256=SHA3,
    )
    assert success["status"] == "success"

    outcome_map = {"S30": SHA1, "S60": SHA2, "S90": SHA3, "S64": SHA4}
    config = protocol.make_config_selection(
        arm_outcome_sha256_by_execution_order=outcome_map,
        candidate_global_manifest_sha256=SHA5,
        source_closure_fingerprint_sha256=SHA6,
        selector_manifest_sha256=SHA7,
        overlap_common=protocol.make_audio_set_binding(
            [f"group-{index:03d}" for index in range(16)]
        ),
        section_common=protocol.make_audio_set_binding(
            [f"audio-{index:03d}" for index in range(8)]
        ),
        source_decision="positive",
        arm_order_values=[
            _arm_order("S30", e0=True, e1=True),
            _arm_order("S60", e0=True, e1=False),
            _arm_order("S90", e0=True, e1=False),
            _arm_order("S64", e0=True, e1=True),
        ],
        selected_schedule_arm="S64",
        selected_run_config_fingerprint_sha256=SHA8,
    )
    assert config["source_winner_selected_before_weak"] is True

    bad_reasons = dict(config)
    bad_values = [dict(value) for value in config["arm_order_values"]]
    bad_values[0]["elimination_reasons"] = [
        "runtime_p90_guard",
        "runtime_nonfinite",
    ]
    bad_reasons["arm_order_values"] = bad_values
    bad_reasons["selection_fingerprint_sha256"] = protocol.payload_hash(
        bad_reasons,
        "selection_fingerprint_sha256",
    )
    with pytest.raises(ValueError, match="elimination_reasons order"):
        protocol.validate_config_selection(bad_reasons)

    bad_weak_flag = dict(config, source_winner_selected_before_weak=False)
    bad_weak_flag["selection_fingerprint_sha256"] = protocol.payload_hash(
        bad_weak_flag,
        "selection_fingerprint_sha256",
    )
    with pytest.raises(ValueError, match="must select before weak"):
        protocol.validate_config_selection(bad_weak_flag)

    bad_winner = dict(config, selected_schedule_arm="S30")
    bad_winner["selection_fingerprint_sha256"] = protocol.payload_hash(
        bad_winner,
        "selection_fingerprint_sha256",
    )
    with pytest.raises(ValueError, match="selected arm is not source winner"):
        protocol.validate_config_selection(bad_winner)

    bad_decision = dict(
        config,
        source_decision="negative",
        selected_schedule_arm=None,
        selected_run_config_fingerprint_sha256=None,
        source_winner_selected_before_weak=False,
    )
    bad_decision["selection_fingerprint_sha256"] = protocol.payload_hash(
        bad_decision,
        "selection_fingerprint_sha256",
    )
    with pytest.raises(ValueError, match="source_decision mismatch"):
        protocol.validate_config_selection(bad_decision)

    bad_rss_type = _clone(config)
    bad_rss_type["arm_order_values"][0]["max_worker_rss"] = 1024.5
    bad_rss_type["selection_fingerprint_sha256"] = protocol.payload_hash(
        bad_rss_type,
        "selection_fingerprint_sha256",
    )
    with pytest.raises(ValueError, match="non-negative integer"):
        protocol.validate_config_selection(bad_rss_type)

    summary = protocol.make_four_arm_stage_summary(
        status="success",
        arm_outcome_sha256_by_execution_order=outcome_map,
        candidate_global_manifest_sha256=SHA5,
        source_selection_status="positive",
        config_selection_sha256=protocol.object_complete_sha256(config),
    )
    assert summary["status"] == "success"

    binding = protocol.make_repair80_input_binding(
        identity_source=protocol.make_source_ref(
            artifact_schema=protocol.SOURCE_REPAIR80_IDENTITY_ARTIFACT_SCHEMA,
            sha256=SHA1,
            row_count=80,
            ordered_rows_sha256=SHA2,
        ),
        label_source=protocol.make_source_ref(
            artifact_schema=protocol.SOURCE_LABELS_ARTIFACT_SCHEMA,
            sha256=SHA3,
            row_count=80,
            ordered_rows_sha256=SHA4,
        ),
        four_arm_stage_summary_sha256=protocol.object_complete_sha256(summary),
        candidate_global_manifest_sha256=SHA5,
        source_selection_sha256=protocol.object_complete_sha256(config),
        schedule_weak_veto_outcome_sha256=SHA9,
    )
    assert binding["row_count"] == 80

    with pytest.raises(ValueError, match="sources must have row_count 80"):
        protocol.make_repair80_input_binding(
            identity_source=protocol.make_source_ref(
                artifact_schema=protocol.SOURCE_REPAIR80_IDENTITY_ARTIFACT_SCHEMA,
                sha256=SHA1,
                row_count=79,
                ordered_rows_sha256=SHA2,
            ),
            label_source=protocol.make_source_ref(
                artifact_schema=protocol.SOURCE_LABELS_ARTIFACT_SCHEMA,
                sha256=SHA3,
                row_count=80,
                ordered_rows_sha256=SHA4,
            ),
            four_arm_stage_summary_sha256=protocol.object_complete_sha256(summary),
            candidate_global_manifest_sha256=SHA5,
            source_selection_sha256=protocol.object_complete_sha256(config),
            schedule_weak_veto_outcome_sha256=SHA9,
        )


def test_repair80_identity_sources_for_execution_require_exact_artifacts() -> None:
    (
        identity_rows,
        label_rows,
        identity_artifact,
        label_artifact,
        identity_source,
        label_source,
    ) = _repair80_identity_label_sources()
    identity_bytes = protocol.canonical_json_bytes(identity_artifact)
    label_bytes = protocol.canonical_json_bytes(label_artifact)

    identities = protocol.validate_repair80_identity_sources_for_execution(
        repair80_identity_source_artifact=identity_bytes,
        repair80_label_source_artifact=label_bytes,
        repair80_identity_rows=identity_rows,
        repair80_label_rows=label_rows,
        identity_source=identity_source,
        label_source=label_source,
    )
    assert len(identities) == 80
    assert identities[0]["cache_audio_key"] == "audio-000"

    with pytest.raises(ValueError, match="immutable canonical bytes"):
        protocol.validate_repair80_identity_sources_for_execution(
            repair80_identity_source_artifact=identity_artifact,
            repair80_label_source_artifact=label_bytes,
            repair80_identity_rows=identity_rows,
            repair80_label_rows=label_rows,
            identity_source=identity_source,
            label_source=label_source,
        )

    with pytest.raises(ValueError, match="immutable canonical bytes"):
        protocol.validate_repair80_identity_sources_for_execution(
            repair80_identity_source_artifact=bytearray(identity_bytes),
            repair80_label_source_artifact=label_bytes,
            repair80_identity_rows=identity_rows,
            repair80_label_rows=label_rows,
            identity_source=identity_source,
            label_source=label_source,
        )

    bad_identity_source = dict(identity_source, sha256=SHA0)
    with pytest.raises(ValueError, match="source_repair80_identity.sha256 mismatch"):
        protocol.validate_repair80_identity_sources_for_execution(
            repair80_identity_source_artifact=identity_bytes,
            repair80_label_source_artifact=label_bytes,
            repair80_identity_rows=identity_rows,
            repair80_label_rows=label_rows,
            identity_source=bad_identity_source,
            label_source=label_source,
        )

    bad_order_source = dict(identity_source, ordered_rows_sha256=SHA0)
    with pytest.raises(
        ValueError,
        match="source_repair80_identity.ordered_rows_sha256 mismatch",
    ):
        protocol.validate_repair80_identity_sources_for_execution(
            repair80_identity_source_artifact=identity_bytes,
            repair80_label_source_artifact=label_bytes,
            repair80_identity_rows=identity_rows,
            repair80_label_rows=label_rows,
            identity_source=bad_order_source,
            label_source=label_source,
        )

    short_artifact, short_source = _source_artifact_and_ref(
        protocol.SOURCE_REPAIR80_IDENTITY_ARTIFACT_SCHEMA,
        identity_rows[:-1],
    )
    with pytest.raises(ValueError, match="row_count"):
        protocol.validate_repair80_identity_sources_for_execution(
            repair80_identity_source_artifact=protocol.canonical_json_bytes(
                short_artifact
            ),
            repair80_label_source_artifact=label_bytes,
            repair80_identity_rows=identity_rows[:-1],
            repair80_label_rows=label_rows,
            identity_source=short_source,
            label_source=label_source,
        )

    reordered_rows = list(identity_rows)
    reordered_rows[0], reordered_rows[1] = reordered_rows[1], reordered_rows[0]
    reordered_artifact, reordered_source = _source_artifact_and_ref(
        protocol.SOURCE_REPAIR80_IDENTITY_ARTIFACT_SCHEMA,
        reordered_rows,
    )
    with pytest.raises(ValueError, match="ordered by row_index"):
        protocol.validate_repair80_identity_sources_for_execution(
            repair80_identity_source_artifact=protocol.canonical_json_bytes(
                reordered_artifact
            ),
            repair80_label_source_artifact=label_bytes,
            repair80_identity_rows=reordered_rows,
            repair80_label_rows=label_rows,
            identity_source=reordered_source,
            label_source=label_source,
        )

    with pytest.raises(ValueError, match="do not match source artifact rows"):
        protocol.validate_repair80_identity_sources_for_execution(
            repair80_identity_source_artifact=protocol.canonical_json_bytes(
                reordered_artifact
            ),
            repair80_label_source_artifact=label_bytes,
            repair80_identity_rows=identity_rows,
            repair80_label_rows=label_rows,
            identity_source=reordered_source,
            label_source=label_source,
        )

    duplicated_rows = list(identity_rows)
    label_zero = label_rows[0]
    duplicated_rows[1] = protocol.make_identity(
        stage=protocol.EXP007_REPAIR_STAGE,
        row_index=1,
        source_row_index=10_001,
        cache_audio_key=label_zero["cache_audio_key"],
        audio_group_key=label_zero["audio_group_key"],
        label_stratum=label_zero["label_stratum"],
        source_long_track=label_zero["source_long_track"],
        duration_ms=label_zero["duration_ms"],
        label_source_sha256=protocol.canonical_json_sha256(label_zero),
    )
    duplicated_artifact, duplicated_source = _source_artifact_and_ref(
        protocol.SOURCE_REPAIR80_IDENTITY_ARTIFACT_SCHEMA,
        duplicated_rows,
    )
    with pytest.raises(ValueError, match="duplicate repair80 cache_audio_key"):
        protocol.validate_repair80_identity_sources_for_execution(
            repair80_identity_source_artifact=protocol.canonical_json_bytes(
                duplicated_artifact
            ),
            repair80_label_source_artifact=label_bytes,
            repair80_identity_rows=duplicated_rows,
            repair80_label_rows=label_rows,
            identity_source=duplicated_source,
            label_source=label_source,
        )

    swapped_labels = _clone(label_rows)
    swapped_labels[0]["label_stratum"], swapped_labels[20]["label_stratum"] = (
        swapped_labels[20]["label_stratum"],
        swapped_labels[0]["label_stratum"],
    )
    swapped_labels[0]["label"]["stratum"] = swapped_labels[0]["label_stratum"]
    swapped_labels[20]["label"]["stratum"] = swapped_labels[20]["label_stratum"]
    swapped_label_artifact, swapped_label_source = _source_artifact_and_ref(
        protocol.SOURCE_LABELS_ARTIFACT_SCHEMA,
        swapped_labels,
    )
    with pytest.raises(ValueError, match="label_stratum mismatch"):
        protocol.validate_repair80_identity_sources_for_execution(
            repair80_identity_source_artifact=identity_bytes,
            repair80_label_source_artifact=protocol.canonical_json_bytes(
                swapped_label_artifact
            ),
            repair80_identity_rows=identity_rows,
            repair80_label_rows=swapped_labels,
            identity_source=identity_source,
            label_source=swapped_label_source,
        )

    tampered_class_rows = list(identity_rows)
    tampered_class_rows[0] = protocol.make_identity(
        stage=protocol.EXP007_REPAIR_STAGE,
        row_index=0,
        source_row_index=10_000,
        cache_audio_key=label_zero["cache_audio_key"],
        audio_group_key=label_zero["audio_group_key"],
        label_stratum="dense",
        source_long_track=label_zero["source_long_track"],
        duration_ms=label_zero["duration_ms"],
        label_source_sha256=protocol.canonical_json_sha256(label_zero),
    )
    tampered_class_artifact, tampered_class_source = _source_artifact_and_ref(
        protocol.SOURCE_REPAIR80_IDENTITY_ARTIFACT_SCHEMA,
        tampered_class_rows,
    )
    with pytest.raises(ValueError, match="label_stratum mismatch"):
        protocol.validate_repair80_identity_sources_for_execution(
            repair80_identity_source_artifact=protocol.canonical_json_bytes(
                tampered_class_artifact
            ),
            repair80_label_source_artifact=label_bytes,
            repair80_identity_rows=tampered_class_rows,
            repair80_label_rows=label_rows,
            identity_source=tampered_class_source,
            label_source=label_source,
        )

    forbidden_labels = _clone(label_rows)
    forbidden_labels[0]["metrics"] = {"phase_error_ms": 1.0}
    forbidden_label_artifact, forbidden_label_source = _source_artifact_and_ref(
        protocol.SOURCE_LABELS_ARTIFACT_SCHEMA,
        forbidden_labels,
    )
    with pytest.raises(ValueError, match="forbidden selector field"):
        protocol.validate_repair80_identity_sources_for_execution(
            repair80_identity_source_artifact=identity_bytes,
            repair80_label_source_artifact=protocol.canonical_json_bytes(
                forbidden_label_artifact
            ),
            repair80_identity_rows=identity_rows,
            repair80_label_rows=forbidden_labels,
            identity_source=identity_source,
            label_source=forbidden_label_source,
        )


def test_repair_run_config_execution_requires_committed_dependency_chain(
    tmp_path: Path,
) -> None:
    deps = _repair_execution_dependencies(tmp_path)
    protocol.validate_run_config_for_execution(
        deps["repair_config"],
        **_repair_execution_kwargs(deps, tmp_path),
    )

    with pytest.raises(ValueError, match="source_closure"):
        protocol.validate_run_config_for_execution(deps["repair_config"])

    missing_reference_kwargs = _repair_execution_kwargs(deps, tmp_path)
    missing_reference_kwargs.pop("candidate_reference_manifest")
    with pytest.raises(ValueError, match="candidate_reference_manifest"):
        protocol.validate_run_config_for_execution(
            deps["repair_config"],
            **missing_reference_kwargs,
        )

    missing_identity_artifact_kwargs = _repair_execution_kwargs(deps, tmp_path)
    missing_identity_artifact_kwargs.pop("repair80_identity_source_artifact")
    with pytest.raises(ValueError, match="repair80_identity_source_artifact"):
        protocol.validate_run_config_for_execution(
            deps["repair_config"],
            **missing_identity_artifact_kwargs,
        )

    missing_identity_rows_kwargs = _repair_execution_kwargs(deps, tmp_path)
    missing_identity_rows_kwargs.pop("repair80_identity_rows")
    with pytest.raises(ValueError, match="repair80_identity_rows"):
        protocol.validate_run_config_for_execution(
            deps["repair_config"],
            **missing_identity_rows_kwargs,
        )

    random_input = _clone(deps["repair_config"])
    random_input["input_manifest_sha256"] = SHA0
    random_input["run_config_fingerprint_sha256"] = protocol.payload_hash(
        random_input,
        "run_config_fingerprint_sha256",
    )
    protocol.validate_run_config(random_input)
    with pytest.raises(ValueError, match="input manifest does not match binding"):
        protocol.validate_run_config_for_execution(
            random_input,
            **_repair_execution_kwargs(deps, tmp_path),
        )

    stale_weak = _clone(deps["weak_outcome"])
    stale_weak["summary"]["decision"] = "ambiguous"
    stale_weak["summary"]["action"] = "stop_ambiguous"
    stale_weak["summary"]["summary_fingerprint_sha256"] = protocol.payload_hash(
        stale_weak["summary"],
        "summary_fingerprint_sha256",
    )
    stale_weak["summary_payload_sha256"] = protocol.object_complete_sha256(
        stale_weak["summary"]
    )
    stale_weak["outcome_fingerprint_sha256"] = protocol.payload_hash(
        stale_weak,
        "outcome_fingerprint_sha256",
    )
    with pytest.raises(ValueError, match="weak outcome SHA mismatch"):
        protocol.validate_run_config_for_execution(
            deps["repair_config"],
            **{
                **_repair_execution_kwargs(deps, tmp_path),
                "schedule_weak_veto_outcome": stale_weak,
            },
        )

    stale_source_summary = _clone(deps["source_arm_summaries"])
    stale_source_summary["S64"]["gates"]["worker_rss_bytes"] = protocol.make_stats_value(
        [2048.0] * 4
    )
    stale_source_summary["S64"]["rss_summary"]["worker_lifetime_bytes"] = [
        2048,
        2048,
        2048,
        2048,
    ]
    stale_source_summary["S64"]["rss_summary"]["arm_max_worker_bytes"] = 2048
    stale_source_summary["S64"]["summary_fingerprint_sha256"] = protocol.payload_hash(
        stale_source_summary["S64"],
        "summary_fingerprint_sha256",
    )
    with pytest.raises(ValueError, match="source summary SHA mismatch"):
        protocol.validate_run_config_for_execution(
            deps["repair_config"],
            **{
                **_repair_execution_kwargs(deps, tmp_path),
                "source_arm_stage_summaries_by_execution_order": stale_source_summary,
            },
        )


def test_repair_execution_binds_full_rows_to_committed_source_refs(
    tmp_path: Path,
) -> None:
    deps = _repair_execution_dependencies(tmp_path)

    rows_with_run_config_tamper = _clone(deps["arm_rows_by_arm"])
    rows_with_run_config_tamper["S60"][0] = _row_result(
        0,
        arm="S60",
        selector_manifest_sha256=SHA7,
        input_manifest_sha256=SHA7,
        run_config_fingerprint_sha256=SHA0,
        source_closure_fingerprint_sha256=deps["source_closure_fingerprint_sha256"],
        candidate_payload=deps["candidate_payloads_by_arm"]["S60"][0],
    )
    summaries_with_run_config_tamper = _clone(deps["source_arm_summaries"])
    summaries_with_run_config_tamper["S60"] = _source_arm_stage_summary(
        arm="S60",
        rows=rows_with_run_config_tamper["S60"],
        run_config_fingerprint_sha256=deps["source_arm_summaries"]["S60"][
            "run_config_fingerprint_sha256"
        ],
        candidate_reference_manifest_sha256=deps["source_arm_summaries"]["S60"][
            "candidate_reference_manifest_sha256"
        ],
        candidate_reference_entry_payload_sha256s=[
            ref["candidate_reference_entry_payload_sha256"]
            for ref in deps["source_arm_summaries"]["S60"]["row_refs"]
        ],
    )
    candidate_global = _clone(deps["candidate_global_manifest"])
    candidate_global["entries"][0]["arm_row_payload_sha256"]["S60"] = (
        rows_with_run_config_tamper["S60"][0]["row_payload_sha256"]
    )
    candidate_global["ordered_entries_sha256"] = protocol.canonical_json_sha256(
        candidate_global["entries"]
    )
    candidate_global["manifest_fingerprint_sha256"] = protocol.payload_hash(
        candidate_global,
        "manifest_fingerprint_sha256",
    )
    run_config_tamper = _rebind_schedule_dependencies(
        deps,
        arm_rows_by_arm=rows_with_run_config_tamper,
        source_arm_summaries=summaries_with_run_config_tamper,
        candidate_global_manifest=candidate_global,
    )
    with pytest.raises(ValueError, match="row run config mismatch"):
        protocol.validate_run_config_for_execution(
            run_config_tamper["repair_config"],
            **_repair_execution_kwargs(run_config_tamper, tmp_path),
        )

    detached_refs = _clone(deps["source_arm_summaries"])
    detached_refs["S60"]["row_refs"][0]["row_payload_sha256"] = SHA0
    detached_refs["S60"]["row_payloads_sha256"] = protocol.canonical_json_sha256(
        detached_refs["S60"]["row_refs"]
    )
    detached_refs["S60"]["summary_fingerprint_sha256"] = protocol.payload_hash(
        detached_refs["S60"],
        "summary_fingerprint_sha256",
    )
    detached_deps = _rebind_schedule_dependencies(
        deps,
        source_arm_summaries=detached_refs,
    )
    with pytest.raises(ValueError, match="source row refs mismatch"):
        protocol.validate_run_config_for_execution(
            detached_deps["repair_config"],
            **_repair_execution_kwargs(detached_deps, tmp_path),
        )


def test_repair_execution_requires_authoritative_candidate_global_result(
    tmp_path: Path,
) -> None:
    deps = _repair_execution_dependencies(tmp_path)

    def digest_token_validator(
        payload: Mapping[str, Any],
        **_: Any,
    ) -> Mapping[str, Any]:
        return {
            "accepted": True,
            "candidate_global_manifest_sha256": protocol.object_complete_sha256(payload),
        }

    with pytest.raises(ValueError, match="authoritative candidate global"):
        protocol.validate_run_config_for_execution(
            deps["repair_config"],
            **_repair_execution_kwargs(
                deps,
                tmp_path,
                authoritative_candidate_global_validator=digest_token_validator,
            ),
        )

    divergent = _clone(deps["candidate_global_manifest"])
    divergent["entries"][0]["candidate_payload_sha256"] = SHA0
    divergent["ordered_entries_sha256"] = protocol.canonical_json_sha256(
        divergent["entries"]
    )
    divergent["manifest_fingerprint_sha256"] = protocol.payload_hash(
        divergent,
        "manifest_fingerprint_sha256",
    )

    def divergent_validator(
        _: Mapping[str, Any],
        **__: Any,
    ) -> Mapping[str, Any]:
        return divergent

    with pytest.raises(ValueError, match="authoritative candidate global"):
        protocol.validate_run_config_for_execution(
            deps["repair_config"],
            **_repair_execution_kwargs(
                deps,
                tmp_path,
                authoritative_candidate_global_validator=divergent_validator,
            ),
        )


def test_repair_execution_rejects_stale_candidate_global_context(
    tmp_path: Path,
) -> None:
    deps = _repair_execution_dependencies(tmp_path)

    stale_reference_kwargs = _repair_execution_kwargs(deps, tmp_path)
    stale_reference = _clone(deps["candidate_reference_manifest"])
    stale_reference["entries"] = list(reversed(stale_reference["entries"]))
    stale_reference["ordered_entries_sha256"] = protocol.canonical_json_sha256(
        stale_reference["entries"]
    )
    stale_reference["manifest_fingerprint_sha256"] = protocol.payload_hash(
        stale_reference,
        "manifest_fingerprint_sha256",
    )
    stale_reference_kwargs["candidate_reference_manifest"] = stale_reference
    with pytest.raises(ValueError, match="canonical manifest bundle path"):
        protocol.validate_run_config_for_execution(
            deps["repair_config"],
            **stale_reference_kwargs,
        )

    stale_row_kwargs = _repair_execution_kwargs(deps, tmp_path)
    stale_rows = _clone(deps["arm_rows_by_arm"])
    stale_rows["S60"][0] = _row_result(
        0,
        arm="S60",
        selector_manifest_sha256=SHA7,
        input_manifest_sha256=SHA7,
        run_config_fingerprint_sha256=SHA0,
        source_closure_fingerprint_sha256=deps["source_closure_fingerprint_sha256"],
        candidate_payload=deps["candidate_payloads_by_arm"]["S60"][0],
    )
    stale_row_kwargs["arm_rows_by_arm"] = stale_rows
    with pytest.raises(ValueError, match="row run config mismatch"):
        protocol.validate_run_config_for_execution(
            deps["repair_config"],
            **stale_row_kwargs,
        )


def test_repair_execution_rejects_fabricated_source_selection_and_weak_arm(
    tmp_path: Path,
) -> None:
    deps = _repair_execution_dependencies(tmp_path)

    fabricated_selection = _clone(deps["config_selection"])
    fabricated_selection["overlap_common"] = protocol.make_audio_set_binding(
        [f"fabricated-{index:03d}" for index in range(16)]
    )
    fabricated_selection["selection_fingerprint_sha256"] = protocol.payload_hash(
        fabricated_selection,
        "selection_fingerprint_sha256",
    )
    fabricated_deps = _rebind_selection(deps, fabricated_selection)
    with pytest.raises(ValueError, match="row rederive mismatch"):
        protocol.validate_run_config_for_execution(
            fabricated_deps["repair_config"],
            **_repair_execution_kwargs(fabricated_deps, tmp_path),
        )

    wrong_arm_weak = _clone(deps["weak_outcome"])
    wrong_arm_weak["summary"]["schedule_arm"] = "S30"
    wrong_arm_weak["summary"]["summary_fingerprint_sha256"] = protocol.payload_hash(
        wrong_arm_weak["summary"],
        "summary_fingerprint_sha256",
    )
    wrong_arm_weak["summary_payload_sha256"] = protocol.object_complete_sha256(
        wrong_arm_weak["summary"]
    )
    wrong_arm_weak["outcome_fingerprint_sha256"] = protocol.payload_hash(
        wrong_arm_weak,
        "outcome_fingerprint_sha256",
    )
    wrong_arm_deps = _rebind_weak_outcome(deps, wrong_arm_weak)
    with pytest.raises(ValueError, match="weak summary arm mismatch"):
        protocol.validate_run_config_for_execution(
            wrong_arm_deps["repair_config"],
            **_repair_execution_kwargs(wrong_arm_deps, tmp_path),
        )


def test_repair80_gates_exact_card_schema_rejects_lossy_adapter_shape() -> None:
    denominators = _repair80_denominators()
    gates = _repair80_gates()

    assert protocol.REPAIR80_GATES_FIELDS == frozenset(gates)
    protocol.validate_repair80_gates(gates, denominators=denominators)

    lossy = _clone(gates)
    lossy.pop("selected_product_fallback_rate")
    lossy["hard_guards_passed"] = True
    with pytest.raises(ValueError, match="Repair80Gates fields"):
        protocol.validate_repair80_gates(lossy, denominators=denominators)


def test_source_closure_and_source_arm_summary_validators_are_exact() -> None:
    closure = _source_closure()
    validated = protocol.validate_source_closure(closure)
    assert validated["source_closure_fingerprint_sha256"] == protocol.canonical_json_sha256(
        validated["behavior"]
    )

    audit_only = _clone(closure)
    audit_only["audit"]["generated_at_utc"] = "2026-08-13T00:00:00Z"
    audit_only["full_payload_sha256"] = protocol.payload_hash(
        audit_only,
        "full_payload_sha256",
    )
    assert protocol.validate_source_closure(audit_only)[
        "source_closure_fingerprint_sha256"
    ] == closure["source_closure_fingerprint_sha256"]

    behavior_mutation = _clone(closure)
    behavior_mutation["behavior"]["relative_source_files"][0]["sha256"] = SHA9
    behavior_mutation["behavior"]["relative_source_files_sha256"] = protocol.canonical_json_sha256(
        behavior_mutation["behavior"]["relative_source_files"]
    )
    behavior_mutation["full_payload_sha256"] = protocol.payload_hash(
        behavior_mutation,
        "full_payload_sha256",
    )
    with pytest.raises(ValueError, match="source_closure_fingerprint_sha256 mismatch"):
        protocol.validate_source_closure(behavior_mutation)

    summary = _source_arm_stage_summary()
    protocol.validate_source_arm_stage_summary(summary)
    bad_summary = _clone(summary)
    bad_summary["gates"]["candidate_fallback_rate"] = protocol.make_rate_value(2, 16)
    bad_summary["summary_fingerprint_sha256"] = protocol.payload_hash(
        bad_summary,
        "summary_fingerprint_sha256",
    )
    with pytest.raises(ValueError, match="candidate fallback numerator mismatch"):
        protocol.validate_source_arm_stage_summary(bad_summary)

    runtime_mismatch = _clone(summary)
    runtime_mismatch["runtime_summary"]["row_seconds"] = protocol.make_stats_value(
        [2.0] * 16
    )
    runtime_mismatch["summary_fingerprint_sha256"] = protocol.payload_hash(
        runtime_mismatch,
        "summary_fingerprint_sha256",
    )
    with pytest.raises(ValueError, match="runtime summary mismatch"):
        protocol.validate_source_arm_stage_summary(runtime_mismatch)

    rss_mismatch = _clone(summary)
    rss_mismatch["gates"]["worker_rss_bytes"] = protocol.make_stats_value(
        [2048.0] * 4
    )
    rss_mismatch["summary_fingerprint_sha256"] = protocol.payload_hash(
        rss_mismatch,
        "summary_fingerprint_sha256",
    )
    with pytest.raises(ValueError, match="worker RSS summary mismatch"):
        protocol.validate_source_arm_stage_summary(rss_mismatch)


def test_authoritative_source_closure_rebuilds_repo_bytes_and_import_graph(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    closure = protocol.make_source_closure(
        repo_root,
        generated_at_utc="2026-08-12T00:00:00Z",
    )
    validated = protocol.validate_source_closure_for_repo(closure, repo_root)
    behavior = validated["behavior"]
    assert behavior["entry_modules"] == list(protocol.SOURCE_CLOSURE_ENTRY_MODULES)
    assert any(
        item["relative_path"]
        == "docs/research/timing_v3_experiment_007_real_cache_schedule_repair.md"
        for item in behavior["required_non_import_files"]
    )
    assert any(
        item["module_name"] == "pulsefield_model.timing.evaluation.exp007_protocol"
        for item in behavior["module_identities"]
    )
    assert any(
        edge["imported_module"] == "pulsefield_model.timing.evaluation.exp007_metrics"
        and edge["resolved_relative_path"]
        == "src/pulsefield_model/timing/evaluation/exp007_metrics.py"
        for edge in behavior["import_edges"]
    )

    fake_sha = _clone(closure)
    fake_sha["behavior"]["relative_source_files"][0]["sha256"] = SHA0
    fake_sha["behavior"]["relative_source_files_sha256"] = protocol.canonical_json_sha256(
        fake_sha["behavior"]["relative_source_files"]
    )
    fake_sha["behavior"]["module_identities_sha256"] = protocol.canonical_json_sha256(
        fake_sha["behavior"]["module_identities"]
    )
    fake_sha["source_closure_fingerprint_sha256"] = protocol.canonical_json_sha256(
        fake_sha["behavior"]
    )
    fake_sha["full_payload_sha256"] = protocol.payload_hash(
        fake_sha,
        "full_payload_sha256",
    )
    with pytest.raises(ValueError, match="repo source bytes"):
        protocol.validate_source_closure_for_repo(fake_sha, repo_root)

    missing_module = _clone(closure)
    missing_module["behavior"]["module_identities"] = missing_module["behavior"][
        "module_identities"
    ][1:]
    missing_module["behavior"]["module_identities_sha256"] = protocol.canonical_json_sha256(
        missing_module["behavior"]["module_identities"]
    )
    missing_module["source_closure_fingerprint_sha256"] = protocol.canonical_json_sha256(
        missing_module["behavior"]
    )
    missing_module["full_payload_sha256"] = protocol.payload_hash(
        missing_module,
        "full_payload_sha256",
    )
    with pytest.raises(ValueError, match="repo source bytes"):
        protocol.validate_source_closure_for_repo(missing_module, repo_root)
    missing_edge = _clone(closure)
    assert missing_edge["behavior"]["import_edges"]
    missing_edge["behavior"]["import_edges"] = missing_edge["behavior"]["import_edges"][1:]
    missing_edge["behavior"]["import_graph_sha256"] = protocol.canonical_json_sha256(
        missing_edge["behavior"]["import_edges"]
    )
    missing_edge["source_closure_fingerprint_sha256"] = protocol.canonical_json_sha256(
        missing_edge["behavior"]
    )
    missing_edge["full_payload_sha256"] = protocol.payload_hash(
        missing_edge,
        "full_payload_sha256",
    )
    with pytest.raises(ValueError, match="repo source bytes"):
        protocol.validate_source_closure_for_repo(missing_edge, repo_root)

    audit_only = _clone(closure)
    audit_only["audit"]["generated_at_utc"] = "2026-08-13T00:00:00Z"
    audit_only["full_payload_sha256"] = protocol.payload_hash(
        audit_only,
        "full_payload_sha256",
    )
    assert protocol.validate_source_closure(audit_only)[
        "source_closure_fingerprint_sha256"
    ] == closure["source_closure_fingerprint_sha256"]


def test_source_closure_default_timestamp_supports_project_python() -> None:
    repo_root = _repo_root()
    closure = protocol.make_source_closure(repo_root)

    generated_at_utc = closure["audit"]["generated_at_utc"]
    assert generated_at_utc.endswith("Z")
    assert "+00:00" not in generated_at_utc
    protocol.validate_source_closure_for_repo(closure, repo_root)


def test_source_closure_includes_implicit_parent_package_inits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path
    providers_dir = repo_root / "src" / "pulsefield_model" / "timing" / "providers"
    providers_dir.mkdir(parents=True)
    card = repo_root / "docs" / "research" / "timing_v3_experiment_007_real_cache_schedule_repair.md"
    card.parent.mkdir(parents=True)
    card.write_text("frozen card\n", encoding="utf-8")
    (repo_root / "src" / "pulsefield_model" / "__init__.py").write_text(
        '"""root package."""\n',
        encoding="utf-8",
    )
    (repo_root / "src" / "pulsefield_model" / "timing" / "__init__.py").write_text(
        "from pulsefield_model.timing import support\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "pulsefield_model" / "timing" / "support.py").write_text(
        "SUPPORT = 1\n",
        encoding="utf-8",
    )
    (providers_dir / "__init__.py").write_text(
        "from pulsefield_model.timing.providers import helper\n",
        encoding="utf-8",
    )
    (providers_dir / "helper.py").write_text("HELPER = 1\n", encoding="utf-8")
    (providers_dir / "beatthis_cache.py").write_text(
        "def optional_loader():\n"
        "    import unbound_function_body_dependency\n"
        "    return unbound_function_body_dependency\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        protocol,
        "SOURCE_CLOSURE_ENTRY_MODULES",
        ("pulsefield_model.timing.providers.beatthis_cache",),
    )

    closure = protocol.make_source_closure(
        repo_root,
        generated_at_utc="2026-08-12T00:00:00Z",
    )
    behavior = closure["behavior"]
    relative_paths = {row["relative_path"] for row in behavior["relative_source_files"]}
    module_names = {row["module_name"] for row in behavior["module_identities"]}
    external_imports = {
        row["imported_module"]
        for row in behavior["import_edges"]
        if row["resolved_relative_path"] is None
    }

    assert {
        "src/pulsefield_model/__init__.py",
        "src/pulsefield_model/timing/__init__.py",
        "src/pulsefield_model/timing/providers/__init__.py",
    }.issubset(relative_paths)
    assert {
        "pulsefield_model",
        "pulsefield_model.timing",
        "pulsefield_model.timing.providers",
        "pulsefield_model.timing.support",
        "pulsefield_model.timing.providers.helper",
    }.issubset(module_names)
    assert "unbound_function_body_dependency" not in external_imports

    (providers_dir / "__init__.py").write_text(
        "from pulsefield_model.timing.providers import helper\nMUTATED = True\n",
        encoding="utf-8",
    )
    rebuilt = protocol.make_source_closure(
        repo_root,
        generated_at_utc="2026-08-12T00:00:00Z",
    )
    assert (
        rebuilt["source_closure_fingerprint_sha256"]
        != closure["source_closure_fingerprint_sha256"]
    )
    with pytest.raises(ValueError, match="repo source bytes"):
        protocol.validate_source_closure_for_repo(closure, repo_root)


def test_source_closure_rejects_unbound_import_time_external_dependency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path
    package_dir = repo_root / "src" / "pulsefield_model" / "timing" / "evaluation"
    package_dir.mkdir(parents=True)
    card = repo_root / "docs" / "research" / "timing_v3_experiment_007_real_cache_schedule_repair.md"
    card.parent.mkdir(parents=True)
    card.write_text("frozen card\n", encoding="utf-8")
    (repo_root / "src" / "pulsefield_model" / "__init__.py").write_text(
        '"""root package."""\n',
        encoding="utf-8",
    )
    (repo_root / "src" / "pulsefield_model" / "timing" / "__init__.py").write_text(
        "",
        encoding="utf-8",
    )
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "entry.py").write_text(
        "class ImportAtDefinitionTime:\n"
        "    import unbound_import_time_dependency_for_exp007\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        protocol,
        "SOURCE_CLOSURE_ENTRY_MODULES",
        ("pulsefield_model.timing.evaluation.entry",),
    )

    with pytest.raises(ValueError, match="not behavior-bound"):
        protocol.make_source_closure(
            repo_root,
            generated_at_utc="2026-08-12T00:00:00Z",
        )


def test_source_closure_external_imports_are_stdlib_or_numpy() -> None:
    closure = protocol.make_source_closure(
        _repo_root(),
        generated_at_utc="2026-08-12T00:00:00Z",
    )
    unresolved = {
        row["imported_module"]
        for row in closure["behavior"]["import_edges"]
        if row["resolved_relative_path"] is None
    }

    assert "beat_this.inference" not in unresolved
    assert "soundfile" not in unresolved
    for imported_module in unresolved:
        top_level = imported_module.split(".", 1)[0]
        assert top_level == "numpy" or top_level in sys.stdlib_module_names


def test_source_closure_dynamic_import_targets_are_bound() -> None:
    repo_root = _repo_root()
    closure = protocol.make_source_closure(
        repo_root,
        generated_at_utc="2026-08-12T00:00:00Z",
    )
    module_names = {
        row["module_name"]
        for row in closure["behavior"]["module_identities"]
    }

    targets: list[tuple[str, str]] = []
    for row in closure["behavior"]["relative_source_files"]:
        path = repo_root / row["relative_path"]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "importlib"
            ):
                continue
            assert node.args, f"{row['relative_path']} import_module call has no target"
            target = node.args[0]
            assert isinstance(
                target,
                ast.Constant,
            ) and isinstance(
                target.value,
                str,
            ), f"{row['relative_path']} import_module target must be a string literal"
            targets.append((row["relative_path"], target.value))

    assert targets
    for relative_path, target in targets:
        if target == "numpy":
            continue
        assert target in module_names, (
            f"{relative_path} dynamically imports {target}, "
            "but SourceClosure does not bind it as an owned module"
        )


def test_run_config_execution_requires_authoritative_source_closure(
    tmp_path: Path,
) -> None:
    repo_root = _repo_root()
    closure = protocol.make_source_closure(
        repo_root,
        generated_at_utc="2026-08-12T00:00:00Z",
    )
    schedule_config = protocol.make_run_config(
        stage=protocol.EXP007_SCHEDULE_STAGE,
        schedule_arm="S30",
        selector_manifest_sha256=SHA7,
        input_manifest_sha256=SHA7,
        source_closure_fingerprint_sha256=closure["source_closure_fingerprint_sha256"],
        cache_config_sha256=SHA3,
        grid_fitter_config_sha256=SHA4,
        weak_config_sha256=SHA5,
    )
    assert protocol.validate_run_config_for_execution(
        schedule_config,
        source_closure=closure,
        source_repo_root=repo_root,
    ) == schedule_config
    with pytest.raises(ValueError, match="source_closure"):
        protocol.validate_run_config_for_execution(schedule_config)

    fake_closure = _clone(closure)
    fake_closure["behavior"]["relative_source_files"][0]["sha256"] = SHA0
    fake_closure["behavior"]["relative_source_files_sha256"] = (
        protocol.canonical_json_sha256(fake_closure["behavior"]["relative_source_files"])
    )
    fake_closure["source_closure_fingerprint_sha256"] = protocol.canonical_json_sha256(
        fake_closure["behavior"]
    )
    fake_closure["full_payload_sha256"] = protocol.payload_hash(
        fake_closure,
        "full_payload_sha256",
    )
    stale_config = protocol.make_run_config(
        stage=protocol.EXP007_SCHEDULE_STAGE,
        schedule_arm="S30",
        selector_manifest_sha256=SHA7,
        input_manifest_sha256=SHA7,
        source_closure_fingerprint_sha256=fake_closure[
            "source_closure_fingerprint_sha256"
        ],
        cache_config_sha256=SHA3,
        grid_fitter_config_sha256=SHA4,
        weak_config_sha256=SHA5,
    )
    with pytest.raises(ValueError, match="repo source bytes"):
        protocol.validate_run_config_for_execution(
            stale_config,
            source_closure=fake_closure,
            source_repo_root=repo_root,
        )


def _identity(*, stage: str, row_index: int) -> dict[str, Any]:
    return protocol.make_identity(
        stage=stage,
        row_index=row_index,
        source_row_index=row_index + 10,
        cache_audio_key=f"audio-{row_index:03d}",
        audio_group_key=f"group-{row_index:03d}",
        label_stratum="stable",
        source_long_track=False,
        duration_ms=120_000,
        label_source_sha256=_digest(f"label-{row_index}"),
    )


def _arm_order(
    arm: str,
    *,
    e0: bool = True,
    e1: bool = False,
) -> dict[str, Any]:
    p90_overlap = 0.0 if e1 else 91.0 if e0 else None
    section_violation_count = 0 if e0 else None
    p90_section_excess = 0.0 if e0 else None
    p90_runtime = 0.0
    order_tuple = [
        0,
        0,
        p90_overlap,
        section_violation_count,
        p90_section_excess,
        protocol.EXP007_TIE_RANK[arm],
    ]
    return {
        "schedule_arm": arm,
        "e0_eligible": e0,
        "e1_eligible": e1,
        "elimination_reasons": [] if e1 else ["overlap_e1_guard"],
        "candidate_fallback_count": 0,
        "no_origin_or_path_count": 0,
        "p90_overlap_ms": p90_overlap,
        "section_inflation_violation_count": section_violation_count,
        "p90_section_excess": p90_section_excess,
        "p90_runtime": p90_runtime,
        "max_worker_rss": 1024,
        "tie_rank": protocol.EXP007_TIE_RANK[arm],
        "order_tuple_sha256": protocol.canonical_json_sha256(order_tuple) if e1 else None,
    }


def _row_result(
    index: int,
    *,
    arm: str = "S30",
    selector_manifest_sha256: str = SHA8,
    input_manifest_sha256: str = SHA8,
    run_config_fingerprint_sha256: str = SHA7,
    candidate_payload_sha256: str | None = None,
    source_closure_fingerprint_sha256: str = SHA6,
    candidate_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if candidate_payload is not None:
        artifacts = importlib.import_module(
            "pulsefield_model.timing.evaluation.exp007_artifacts"
        )
        serialized = artifacts.serialize_candidate_payload(candidate_payload)
        payload_schema = serialized.schema
        payload_bytes = serialized.byte_count
        payload_field_set = serialized.field_set_sha256
        payload_sha = serialized.payload_sha256
        candidate_fingerprint = serialized.candidate_fingerprint
    else:
        payload_schema = protocol.CANDIDATE_PAYLOAD_SCHEMA
        payload_bytes = 128
        payload_field_set = protocol.candidate_payload_field_set_sha256()
        payload_sha = (
            _digest(f"candidate-payload-{index}")
            if candidate_payload_sha256 is None
            else candidate_payload_sha256
        )
        candidate_fingerprint = _digest(f"candidate-fingerprint-{index}")
    row = protocol.minimal_row_result(
        stage=protocol.EXP007_SCHEDULE_STAGE,
        schedule_arm=arm,
        row_index=index,
        cache_audio_key=f"audio-{index:03d}",
        audio_group_key=f"group-{index:03d}",
        identity_payload_sha256=_digest(f"identity-{index}"),
        source_closure_fingerprint_sha256=source_closure_fingerprint_sha256,
        run_config_fingerprint_sha256=run_config_fingerprint_sha256,
        selector_manifest_sha256=selector_manifest_sha256,
        input_manifest_sha256=input_manifest_sha256,
        candidate_payload_schema=payload_schema,
        candidate_payload_byte_count=payload_bytes,
        candidate_payload_field_set_sha256=payload_field_set,
        candidate_payload_sha256=payload_sha,
        candidate_fingerprint=candidate_fingerprint,
    )
    if candidate_payload is not None:
        return _row_with_restricted_input_signal(
            row,
            candidate_payload["diagnostics"]["input_signal_sha256"],
        )
    return row


def _row_with_restricted_input_signal(
    row: Mapping[str, Any],
    input_signal_sha256: str,
) -> dict[str, Any]:
    updated_prediction = dict(
        row["restricted_prediction"],
        input_signal_sha256=input_signal_sha256,
    )
    return protocol.make_row_result(
        stage=row["stage"],
        schedule_arm=row["schedule_arm"],
        row_index=row["row_index"],
        cache_audio_key=row["cache_audio_key"],
        audio_group_key=row["audio_group_key"],
        identity_payload_sha256=row["identity_payload_sha256"],
        cache_identity=row["cache_identity"],
        source_closure_fingerprint_sha256=row["source_closure_fingerprint_sha256"],
        run_config_fingerprint_sha256=row["run_config_fingerprint_sha256"],
        selector_manifest_sha256=row["selector_manifest_sha256"],
        input_manifest_sha256=row["input_manifest_sha256"],
        resume=row["resume"],
        restricted_prediction=updated_prediction,
        candidate_payload_schema=row["candidate_payload_schema"],
        candidate_payload_byte_count=row["candidate_payload_byte_count"],
        candidate_payload_field_set_sha256=row["candidate_payload_field_set_sha256"],
        candidate_payload_sha256=row["candidate_payload_sha256"],
        candidate_fingerprint=row["candidate_fingerprint"],
        methods=row["methods"],
        denominator_flags=row["denominator_flags"],
        diagnostics_summary=row["diagnostics_summary"],
        runtime=row["runtime"],
        rss=row["rss"],
        hard_guards=row["hard_guards"],
    )


def _repair80_identity_label_sources() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    label_rows = []
    for index in range(80):
        key = f"audio-{index:03d}"
        if index < 20:
            label = "stable"
            long_track = False
        elif index < 40:
            label = "dense"
            long_track = False
        elif index < 60:
            label = "jump_candidate"
            long_track = False
        else:
            label = "ambiguous"
            long_track = True
        label_rows.append(
            {
                "cache_audio_key": key,
                "audio_group_key": f"group-{index:03d}",
                "label_stratum": label,
                "source_long_track": long_track,
                "duration_ms": 120_000 + index,
                "source": {"cache_audio_key": key, "long_track": long_track},
                "label": {"stratum": label},
            }
        )
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
    identity_artifact, identity_source = _source_artifact_and_ref(
        protocol.SOURCE_REPAIR80_IDENTITY_ARTIFACT_SCHEMA,
        identity_rows,
    )
    label_artifact, label_source = _source_artifact_and_ref(
        protocol.SOURCE_LABELS_ARTIFACT_SCHEMA,
        label_rows,
    )
    return (
        identity_rows,
        label_rows,
        identity_artifact,
        label_artifact,
        identity_source,
        label_source,
    )


def _source_artifact_and_ref(
    artifact_schema: str,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    row_dicts = [dict(row) for row in rows]
    artifact = {"schema": artifact_schema, "rows": row_dicts}
    source = protocol.make_source_ref(
        artifact_schema=artifact_schema,
        sha256=protocol.canonical_json_sha256(artifact),
        row_count=len(row_dicts),
        ordered_rows_sha256=protocol.canonical_json_sha256(row_dicts),
    )
    return artifact, source


def _repair_execution_dependencies(artifact_root: Path) -> dict[str, Any]:
    repo_root = _repo_root()
    source_closure = protocol.make_source_closure(
        repo_root,
        generated_at_utc="2026-08-12T00:00:00Z",
    )
    source_sha = source_closure["source_closure_fingerprint_sha256"]
    artifacts = importlib.import_module(
        "pulsefield_model.timing.evaluation.exp007_artifacts"
    )
    candidate_payloads = [_candidate_payload(index) for index in range(16)]
    run_configs_by_arm = {
        arm: protocol.make_run_config(
            stage=protocol.EXP007_SCHEDULE_STAGE,
            schedule_arm=arm,
            selector_manifest_sha256=SHA7,
            input_manifest_sha256=SHA7,
            source_closure_fingerprint_sha256=source_sha,
            cache_config_sha256=SHA3,
            grid_fitter_config_sha256=SHA4,
            weak_config_sha256=SHA5,
        )
        for arm in protocol.EXP007_EXECUTION_ORDER
    }
    arm_rows_by_arm = {
        arm: [
            _row_result(
                index,
                arm=arm,
                selector_manifest_sha256=SHA7,
                input_manifest_sha256=SHA7,
                run_config_fingerprint_sha256=run_configs_by_arm[arm][
                    "run_config_fingerprint_sha256"
                ],
                source_closure_fingerprint_sha256=source_sha,
                candidate_payload=candidate_payloads[index],
            )
            for index in range(16)
        ]
        for arm in protocol.EXP007_EXECUTION_ORDER
    }
    for index, row in enumerate(arm_rows_by_arm["S30"]):
        bundle = artifacts.make_candidate_reference_row_bundle(
            stage=protocol.EXP007_SCHEDULE_STAGE,
            schedule_arm="S30",
            row=row,
            candidate_payload=candidate_payloads[index],
            input_signal_sha256=candidate_payloads[index]["diagnostics"][
                "input_signal_sha256"
            ],
        )
        artifacts.publish_candidate_reference_row_bundle(
            root=artifact_root,
            bundle=bundle,
        )
    candidate_reference_manifest = artifacts.build_candidate_reference_manifest(
        root=artifact_root,
        stage=protocol.EXP007_SCHEDULE_STAGE,
        input_manifest_sha256=SHA7,
        source_closure_fingerprint_sha256=source_sha,
        reference_arm="S30",
    )
    candidate_reference_sha = protocol.object_complete_sha256(
        candidate_reference_manifest
    )
    candidate_payloads_by_arm = {
        arm: list(candidate_payloads)
        for arm in protocol.EXP007_EXECUTION_ORDER
    }
    candidate_global = artifacts.build_candidate_global_manifest(
        root=artifact_root,
        reference_manifest=candidate_reference_manifest,
        selector_manifest_sha256=SHA7,
        source_closure_fingerprint_sha256=source_sha,
        run_configs_by_arm=run_configs_by_arm,
        arm_rows_by_arm=arm_rows_by_arm,
        candidate_payloads_by_arm=candidate_payloads_by_arm,
    )
    candidate_global_sha = protocol.object_complete_sha256(candidate_global)
    reference_entry_sha256s = [
        entry["candidate_reference_entry_payload_sha256"]
        for entry in candidate_global["entries"]
    ]
    source_summaries = {
        arm: _source_arm_stage_summary(
            arm=arm,
            rows=arm_rows_by_arm[arm],
            candidate_reference_manifest_sha256=candidate_reference_sha,
            candidate_reference_entry_payload_sha256s=reference_entry_sha256s,
        )
        for arm in protocol.EXP007_EXECUTION_ORDER
    }
    arm_outcomes = {
        arm: protocol.make_arm_stage_success(
            stage=protocol.EXP007_SCHEDULE_STAGE,
            schedule_arm=arm,
            row_payloads_sha256=source_summaries[arm]["row_payloads_sha256"],
            candidate_reference_manifest_sha256=candidate_reference_sha,
            stage_summary_sha256=protocol.object_complete_sha256(source_summaries[arm]),
        )
        for arm in protocol.EXP007_EXECUTION_ORDER
    }
    outcome_map = {
        arm: protocol.object_complete_sha256(outcome)
        for arm, outcome in arm_outcomes.items()
    }
    config = protocol.make_config_selection(
        arm_outcome_sha256_by_execution_order=outcome_map,
        candidate_global_manifest_sha256=candidate_global_sha,
        source_closure_fingerprint_sha256=source_sha,
        selector_manifest_sha256=SHA7,
        overlap_common=protocol.make_audio_set_binding(
            [f"group-{index:03d}" for index in range(16)]
        ),
        section_common=protocol.make_audio_set_binding(
            [f"group-{index:03d}" for index in range(16)]
        ),
        source_decision="positive",
        arm_order_values=[
            _arm_order("S30", e0=True, e1=True),
            _arm_order("S60", e0=True, e1=True),
            _arm_order("S90", e0=True, e1=True),
            _arm_order("S64", e0=True, e1=True),
        ],
        selected_schedule_arm="S64",
        selected_run_config_fingerprint_sha256=run_configs_by_arm["S64"][
            "run_config_fingerprint_sha256"
        ],
    )
    config_sha = protocol.object_complete_sha256(config)
    four_arm_summary = protocol.make_four_arm_stage_summary(
        status="success",
        arm_outcome_sha256_by_execution_order=outcome_map,
        candidate_global_manifest_sha256=candidate_global_sha,
        source_selection_status="positive",
        config_selection_sha256=config_sha,
    )
    four_arm_sha = protocol.object_complete_sha256(four_arm_summary)
    weak_refs = [
        {
            "row_index": index,
            "cache_audio_key": f"audio-{index:03d}",
            "prediction_row_sha256": arm_rows_by_arm["S64"][index][
                "row_payload_sha256"
            ],
            "weak_row_payload_sha256": _digest(f"weak-row-{index}"),
        }
        for index in range(16)
    ]
    weak_summary = protocol.make_schedule_weak_veto_summary(
        schedule_arm="S64",
        four_arm_stage_summary_sha256=four_arm_sha,
        candidate_global_manifest_sha256=candidate_global_sha,
        source_closure_fingerprint_sha256=source_sha,
        source_selection_sha256=config_sha,
        weak_row_refs=weak_refs,
        denominators=_schedule_weak_denominators(),
        gates=_schedule_weak_gates(),
        decision="pass",
    )
    weak_outcome = protocol.make_schedule_weak_success_outcome(weak_summary)
    weak_sha = protocol.object_complete_sha256(weak_outcome)
    (
        repair80_identity_rows,
        repair80_label_rows,
        repair80_identity_artifact,
        repair80_label_artifact,
        repair80_identity_source,
        repair80_label_source,
    ) = _repair80_identity_label_sources()
    binding = protocol.make_repair80_input_binding(
        identity_source=repair80_identity_source,
        label_source=repair80_label_source,
        four_arm_stage_summary_sha256=four_arm_sha,
        candidate_global_manifest_sha256=candidate_global_sha,
        source_selection_sha256=config_sha,
        schedule_weak_veto_outcome_sha256=weak_sha,
    )
    repair_config = protocol.make_run_config(
        stage=protocol.EXP007_REPAIR_STAGE,
        schedule_arm="S64",
        selector_manifest_sha256=SHA7,
        input_manifest_sha256=binding["binding_fingerprint_sha256"],
        schedule_weak_veto_outcome_sha256=weak_sha,
        source_closure_fingerprint_sha256=source_sha,
        cache_config_sha256=SHA3,
        grid_fitter_config_sha256=SHA4,
        weak_config_sha256=SHA5,
    )
    return {
        "arm_outcomes": arm_outcomes,
        "arm_rows_by_arm": arm_rows_by_arm,
        "run_configs_by_arm": run_configs_by_arm,
        "candidate_global_manifest": candidate_global,
        "candidate_payloads_by_arm": candidate_payloads_by_arm,
        "candidate_reference_manifest": candidate_reference_manifest,
        "config_selection": config,
        "four_arm_summary": four_arm_summary,
        "artifact_root": artifact_root,
        "source_closure_fingerprint_sha256": source_sha,
        "source_closure": source_closure,
        "source_repo_root": repo_root,
        "source_arm_summaries": source_summaries,
        "repair80_identity_rows": repair80_identity_rows,
        "repair80_label_rows": repair80_label_rows,
        "repair80_identity_source_artifact": repair80_identity_artifact,
        "repair80_label_source_artifact": repair80_label_artifact,
        "repair80_identity_source_artifact_bytes": protocol.canonical_json_bytes(
            repair80_identity_artifact
        ),
        "repair80_label_source_artifact_bytes": protocol.canonical_json_bytes(
            repair80_label_artifact
        ),
        "weak_outcome": weak_outcome,
        "binding": binding,
        "repair_config": repair_config,
    }


def _candidate_reference_manifest() -> dict[str, Any]:
    return {
        "schema": protocol.CANDIDATE_REFERENCE_MANIFEST_SCHEMA,
        "marker": "candidate-reference",
    }


def _candidate_global_manifest(
    *,
    candidate_reference_manifest_sha256: str,
    arm_rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    source_closure_fingerprint_sha256: str = SHA6,
) -> dict[str, Any]:
    entries = []
    for index in range(16):
        row = protocol.validate_row_result(arm_rows_by_arm["S30"][index])
        entries.append(
            {
                "row_index": index,
                "cache_audio_key": row["cache_audio_key"],
                "audio_group_key": row["audio_group_key"],
                "input_signal_sha256": row["restricted_prediction"][
                    "input_signal_sha256"
                ],
                "candidate_payload_schema": row["candidate_payload_schema"],
                "candidate_payload_field_set_sha256": row[
                    "candidate_payload_field_set_sha256"
                ],
                "candidate_payload_byte_count": row["candidate_payload_byte_count"],
                "candidate_payload_sha256": row["candidate_payload_sha256"],
                "candidate_fingerprint": row["candidate_fingerprint"],
                "candidate_reference_entry_payload_sha256": _digest(f"reference-{index}"),
                "arm_row_payload_sha256": {
                    arm: protocol.validate_row_result(arm_rows_by_arm[arm][index])[
                        "row_payload_sha256"
                    ]
                    for arm in protocol.EXP007_EXECUTION_ORDER
                },
            }
        )
    payload = {
        "schema": protocol.CANDIDATE_GLOBAL_MANIFEST_SCHEMA,
        "experiment_id": protocol.EXP007_EXPERIMENT_ID,
        "stage": protocol.EXP007_SCHEDULE_STAGE,
        "schema_descriptor_sha256": protocol.schema_descriptor_sha256(
            protocol.CANDIDATE_GLOBAL_MANIFEST_SCHEMA
        ),
        "selector_manifest_sha256": SHA7,
        "source_closure_fingerprint_sha256": source_closure_fingerprint_sha256,
        "candidate_reference_manifest_sha256": candidate_reference_manifest_sha256,
        "row_count": 16,
        "entries": entries,
        "ordered_entries_sha256": protocol.canonical_json_sha256(entries),
    }
    return protocol.validate_candidate_global_manifest_non_authoritative_shape(
        protocol.with_payload_hash(payload, "manifest_fingerprint_sha256")
    )


def _repair_execution_kwargs(
    deps: Mapping[str, Any],
    artifact_root: Path,
    *,
    authoritative_candidate_global_validator: Any | None = None,
) -> dict[str, Any]:
    result = {
        "source_closure": deps["source_closure"],
        "source_repo_root": deps["source_repo_root"],
        "repair80_input_binding": deps["binding"],
        "repair80_identity_source_artifact": deps[
            "repair80_identity_source_artifact_bytes"
        ],
        "repair80_label_source_artifact": deps[
            "repair80_label_source_artifact_bytes"
        ],
        "repair80_identity_rows": deps["repair80_identity_rows"],
        "repair80_label_rows": deps["repair80_label_rows"],
        "schedule_weak_veto_outcome": deps["weak_outcome"],
        "four_arm_stage_summary": deps["four_arm_summary"],
        "config_selection": deps["config_selection"],
        "candidate_global_manifest": deps["candidate_global_manifest"],
        "candidate_reference_manifest": deps["candidate_reference_manifest"],
        "artifact_root": artifact_root,
        "run_configs_by_arm": deps["run_configs_by_arm"],
        "arm_rows_by_arm": deps["arm_rows_by_arm"],
        "candidate_payloads_by_arm": deps["candidate_payloads_by_arm"],
        "arm_stage_outcomes_by_execution_order": deps["arm_outcomes"],
        "source_arm_stage_summaries_by_execution_order": deps["source_arm_summaries"],
    }
    if authoritative_candidate_global_validator is not None:
        result["authoritative_candidate_global_validator"] = (
            authoritative_candidate_global_validator
        )
    return result


def _rebind_selection(
    deps: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(deps)
    selection_sha = protocol.object_complete_sha256(selection)
    four_arm_summary = protocol.make_four_arm_stage_summary(
        status="success",
        arm_outcome_sha256_by_execution_order={
            arm: protocol.object_complete_sha256(deps["arm_outcomes"][arm])
            for arm in protocol.EXP007_EXECUTION_ORDER
        },
        candidate_global_manifest_sha256=protocol.object_complete_sha256(
            deps["candidate_global_manifest"]
        ),
        source_selection_status="positive",
        config_selection_sha256=selection_sha,
    )
    weak_summary = protocol.make_schedule_weak_veto_summary(
        schedule_arm=deps["weak_outcome"]["summary"]["schedule_arm"],
        four_arm_stage_summary_sha256=protocol.object_complete_sha256(four_arm_summary),
        candidate_global_manifest_sha256=protocol.object_complete_sha256(
            deps["candidate_global_manifest"]
        ),
        source_closure_fingerprint_sha256=deps["weak_outcome"]["summary"][
            "source_closure_fingerprint_sha256"
        ],
        source_selection_sha256=selection_sha,
        weak_row_refs=deps["weak_outcome"]["summary"]["weak_row_refs"],
        denominators=deps["weak_outcome"]["summary"]["denominators"],
        gates=deps["weak_outcome"]["summary"]["gates"],
        decision=deps["weak_outcome"]["summary"]["decision"],
    )
    result["config_selection"] = dict(selection)
    result["four_arm_summary"] = four_arm_summary
    result["weak_outcome"] = protocol.make_schedule_weak_success_outcome(weak_summary)
    return _rebind_weak_outcome(result, result["weak_outcome"])


def _rebind_weak_outcome(
    deps: Mapping[str, Any],
    weak_outcome: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(deps)
    weak_sha = protocol.object_complete_sha256(weak_outcome)
    binding = protocol.make_repair80_input_binding(
        identity_source=deps["binding"]["identity_source"],
        label_source=deps["binding"]["label_source"],
        four_arm_stage_summary_sha256=protocol.object_complete_sha256(
            deps["four_arm_summary"]
        ),
        candidate_global_manifest_sha256=protocol.object_complete_sha256(
            deps["candidate_global_manifest"]
        ),
        source_selection_sha256=protocol.object_complete_sha256(
            deps["config_selection"]
        ),
        schedule_weak_veto_outcome_sha256=weak_sha,
    )
    result["weak_outcome"] = dict(weak_outcome)
    result["binding"] = binding
    result["repair_config"] = protocol.make_run_config(
        stage=protocol.EXP007_REPAIR_STAGE,
        schedule_arm=deps["repair_config"]["schedule_arm"],
        selector_manifest_sha256=deps["repair_config"]["selector_manifest_sha256"],
        input_manifest_sha256=binding["binding_fingerprint_sha256"],
        schedule_weak_veto_outcome_sha256=weak_sha,
        source_closure_fingerprint_sha256=deps["repair_config"][
            "source_closure_fingerprint_sha256"
        ],
        cache_config_sha256=deps["repair_config"]["cache_config_sha256"],
        grid_fitter_config_sha256=deps["repair_config"][
            "grid_fitter_config_sha256"
        ],
        weak_config_sha256=deps["repair_config"]["weak_config_sha256"],
    )
    return result


def _rebind_schedule_dependencies(
    deps: Mapping[str, Any],
    *,
    arm_rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    source_arm_summaries: Mapping[str, Mapping[str, Any]] | None = None,
    candidate_global_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(deps)
    result["arm_rows_by_arm"] = (
        _clone(arm_rows_by_arm)
        if arm_rows_by_arm is not None
        else _clone(deps["arm_rows_by_arm"])
    )
    result["source_arm_summaries"] = (
        _clone(source_arm_summaries)
        if source_arm_summaries is not None
        else _clone(deps["source_arm_summaries"])
    )
    result["run_configs_by_arm"] = _clone(deps["run_configs_by_arm"])
    result["candidate_global_manifest"] = (
        _clone(candidate_global_manifest)
        if candidate_global_manifest is not None
        else _clone(deps["candidate_global_manifest"])
    )
    result["candidate_payloads_by_arm"] = _clone(deps["candidate_payloads_by_arm"])
    result["arm_outcomes"] = {
        arm: protocol.make_arm_stage_success(
            stage=protocol.EXP007_SCHEDULE_STAGE,
            schedule_arm=arm,
            row_payloads_sha256=result["source_arm_summaries"][arm][
                "row_payloads_sha256"
            ],
            candidate_reference_manifest_sha256=result["source_arm_summaries"][arm][
                "candidate_reference_manifest_sha256"
            ],
            stage_summary_sha256=protocol.object_complete_sha256(
                result["source_arm_summaries"][arm]
            ),
        )
        for arm in protocol.EXP007_EXECUTION_ORDER
    }
    outcome_map = {
        arm: protocol.object_complete_sha256(outcome)
        for arm, outcome in result["arm_outcomes"].items()
    }
    selection = deps["config_selection"]
    selected_arm = selection["selected_schedule_arm"]
    result["config_selection"] = protocol.make_config_selection(
        arm_outcome_sha256_by_execution_order=outcome_map,
        candidate_global_manifest_sha256=protocol.object_complete_sha256(
            result["candidate_global_manifest"]
        ),
        source_closure_fingerprint_sha256=selection[
            "source_closure_fingerprint_sha256"
        ],
        selector_manifest_sha256=selection["selector_manifest_sha256"],
        overlap_common=selection["overlap_common"],
        section_common=selection["section_common"],
        source_decision=selection["source_decision"],
        arm_order_values=selection["arm_order_values"],
        selected_schedule_arm=selected_arm,
        selected_run_config_fingerprint_sha256=(
            None
            if selected_arm is None
            else result["source_arm_summaries"][selected_arm][
                "run_config_fingerprint_sha256"
            ]
        ),
    )
    selection_sha = protocol.object_complete_sha256(result["config_selection"])
    result["four_arm_summary"] = protocol.make_four_arm_stage_summary(
        status="success",
        arm_outcome_sha256_by_execution_order=outcome_map,
        candidate_global_manifest_sha256=protocol.object_complete_sha256(
            result["candidate_global_manifest"]
        ),
        source_selection_status="positive",
        config_selection_sha256=selection_sha,
    )
    weak_summary = deps["weak_outcome"]["summary"]
    weak_refs = [
        {
            **ref,
            "prediction_row_sha256": protocol.validate_row_result(
                result["arm_rows_by_arm"][selected_arm][index]
            )["row_payload_sha256"],
        }
        for index, ref in enumerate(weak_summary["weak_row_refs"])
    ]
    result["weak_outcome"] = protocol.make_schedule_weak_success_outcome(
        protocol.make_schedule_weak_veto_summary(
            schedule_arm=selected_arm,
            four_arm_stage_summary_sha256=protocol.object_complete_sha256(
                result["four_arm_summary"]
            ),
            candidate_global_manifest_sha256=protocol.object_complete_sha256(
                result["candidate_global_manifest"]
            ),
            source_closure_fingerprint_sha256=weak_summary[
                "source_closure_fingerprint_sha256"
            ],
            source_selection_sha256=selection_sha,
            weak_row_refs=weak_refs,
            denominators=weak_summary["denominators"],
            gates=weak_summary["gates"],
            decision=weak_summary["decision"],
        )
    )
    return _rebind_weak_outcome(result, result["weak_outcome"])


def _authoritative_candidate_global_validator(
    payload: Mapping[str, Any],
    *,
    reference_manifest: Mapping[str, Any],
    root: Path,
    run_configs_by_arm: Mapping[str, Mapping[str, Any]],
    arm_rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_payloads_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    assert root.exists()
    result = protocol.validate_candidate_global_manifest_non_authoritative_shape(payload)
    if (
        protocol.object_complete_sha256(reference_manifest)
        != result["candidate_reference_manifest_sha256"]
    ):
        raise ValueError("CandidateGlobalManifest reference SHA mismatch")
    for entry in result["entries"]:
        index = entry["row_index"]
        for arm in protocol.EXP007_EXECUTION_ORDER:
            row = protocol.validate_row_result(arm_rows_by_arm[arm][index])
            config = protocol.validate_run_config(run_configs_by_arm[arm])
            if (
                row["run_config_fingerprint_sha256"]
                != config["run_config_fingerprint_sha256"]
            ):
                raise ValueError("CandidateGlobalManifest row run config mismatch")
            if row["row_payload_sha256"] != entry["arm_row_payload_sha256"][arm]:
                raise ValueError("CandidateGlobalManifest arm row SHA mismatch")
            candidate_payload = candidate_payloads_by_arm[arm][index]
            if (
                candidate_payload["candidate_payload_sha256"]
                != entry["candidate_payload_sha256"]
            ):
                raise ValueError("CandidateGlobalManifest candidate payload SHA mismatch")
    return result


def _schedule_weak_denominators() -> dict[str, Any]:
    keys = [f"audio-{index:03d}" for index in range(16)]
    empty = protocol.make_audio_set_binding([])
    full = protocol.make_audio_set_binding(keys)
    return {
        "stage_audio_count": 16,
        "stage_audio": full,
        "comparator_available_audio": full,
        "comparator_unavailable_audio": empty,
        "comparator_conflicting_audio": empty,
        "current_v2_phase_matched": full,
        "pure_exp006_phase_matched": full,
        "selected_safety_phase_matched": full,
        "phase_common": full,
        "alias_drift_common": full,
        "weak_change_boundary_audio": full,
    }


def _schedule_weak_gates() -> dict[str, Any]:
    return {
        "pure_mean_phase_ratio": protocol.make_ratio_value(1, 1),
        "pure_p90_phase_ratio": protocol.make_ratio_value(1, 1),
        "pure_phase_coverage": protocol.make_coverage_value(16, 16),
        "current_v2_phase_mean_ms": 1.0,
        "pure_exp006_phase_mean_ms": 1.0,
        "current_v2_phase_p90_ms": 1.0,
        "pure_exp006_phase_p90_ms": 1.0,
        "current_v2_alias_drift_mean_ms": 1.0,
        "pure_exp006_alias_drift_mean_ms": 1.0,
        "current_v2_alias_drift_p90_ms": 1.0,
        "pure_exp006_alias_drift_p90_ms": 1.0,
        "alias_max_prefix_drift_mean_ratio": protocol.make_ratio_value(1, 1),
        "alias_max_prefix_drift_p90_ratio": protocol.make_ratio_value(1, 1),
        "current_v2_boundary_f1_mean": 0.8,
        "pure_exp006_boundary_f1_mean": 0.8,
        "selected_boundary_f1_mean": 0.8,
        "pure_minus_v2_boundary_f1_delta": 0.0,
    }


def _repair80_denominators() -> dict[str, Any]:
    keys = [f"audio-{index:03d}" for index in range(80)]
    first = lambda count: keys[:count]
    empty = protocol.make_audio_set_binding([])
    full = protocol.make_audio_set_binding(keys)
    return protocol.validate_repair80_denominators(
        {
            "stage_audio_count": 80,
            "stage_audio": full,
            "cache_valid_audio": full,
            "projection_evaluable_audio": full,
            "candidate_accepted_audio": full,
            "candidate_fallback_audio": empty,
            "selected_product_fallback_audio": empty,
            "baseline_accepted_audio": full,
            "product_grid_available_audio": full,
            "no_origin_or_path_audio": empty,
            "resource_cap_fallback_audio": empty,
            "overlap_available_audio": protocol.make_audio_set_binding(first(20)),
            "current_v2_phase_matched": protocol.make_audio_set_binding(first(40)),
            "pure_exp006_phase_matched": protocol.make_audio_set_binding(first(40)),
            "selected_safety_phase_matched": protocol.make_audio_set_binding(first(40)),
            "phase_common": protocol.make_audio_set_binding(first(40)),
            "stable_pure_paired": protocol.make_audio_set_binding(first(5)),
            "jump_pure_paired": protocol.make_audio_set_binding(first(15)),
            "long_pure_paired": protocol.make_audio_set_binding(first(5)),
            "jump_alias_drift_common": protocol.make_audio_set_binding(first(15)),
            "long_alias_drift_common": protocol.make_audio_set_binding(first(5)),
            "repair_boundary_common": protocol.make_audio_set_binding(first(15)),
        }
    )


def _repair80_gates() -> dict[str, Any]:
    return {
        "candidate_fallback_rate": protocol.make_rate_value(0, 80),
        "selected_product_fallback_rate": protocol.make_rate_value(0, 80),
        "no_origin_or_path_rate": protocol.make_rate_value(0, 80),
        "runtime_seconds": protocol.make_stats_value([1.0] * 80),
        "worker_rss_bytes": protocol.make_stats_value([1024.0] * 4),
        "overlap_ms": protocol.make_stats_value([1.0] * 20),
        "stable_section_excess": protocol.make_stats_value([0.0] * 5),
        "pure_mean_phase_ratio": protocol.make_ratio_value(1, 1),
        "pure_p90_phase_ratio": protocol.make_ratio_value(1, 1),
        "pure_phase_coverage": protocol.make_coverage_value(40, 40),
        "current_v2_phase_mean_ms": 1.0,
        "pure_exp006_phase_mean_ms": 1.0,
        "current_v2_phase_p90_ms": 1.0,
        "pure_exp006_phase_p90_ms": 1.0,
        "stable_phase_mean_ratio": protocol.make_ratio_value(1, 1),
        "stable_phase_p90_ratio": protocol.make_ratio_value(1, 1),
        "jump_phase_mean_ratio": protocol.make_ratio_value(1, 1),
        "current_v2_jump_alias_drift_mean_ms": 1.0,
        "pure_exp006_jump_alias_drift_mean_ms": 1.0,
        "jump_alias_drift_mean_ratio": protocol.make_ratio_value(1, 1),
        "current_v2_long_alias_drift_mean_ms": 1.0,
        "pure_exp006_long_alias_drift_mean_ms": 1.0,
        "current_v2_long_alias_drift_p90_ms": 1.0,
        "pure_exp006_long_alias_drift_p90_ms": 1.0,
        "long_alias_drift_mean_ratio": protocol.make_ratio_value(1, 1),
        "long_alias_drift_p90_ratio": protocol.make_ratio_value(1, 1),
        "current_v2_boundary_f1_mean": 0.8,
        "pure_exp006_boundary_f1_mean": 0.8,
        "selected_boundary_f1_mean": 0.8,
        "pure_minus_v2_boundary_f1_delta": 0.0,
        "every_row_under_180_seconds": True,
        "seam_zero": True,
        "section_cap_valid": True,
        "replay_schema_source_cache_integrity": True,
    }


def _source_closure() -> dict[str, Any]:
    relative_files = [
        {
            "relative_path": "src/pulsefield_model/timing/evaluation/exp007_protocol.py",
            "sha256": SHA1,
        }
    ]
    required_files = [
        {
            "relative_path": "docs/research/timing_v3_experiment_007_real_cache_schedule_repair.md",
            "sha256": SHA2,
        }
    ]
    modules = [
        {
            "module_name": "pulsefield_model.timing.evaluation.exp007_protocol",
            "relative_path": "src/pulsefield_model/timing/evaluation/exp007_protocol.py",
            "sha256": SHA1,
        }
    ]
    behavior = {
        "entry_modules": list(protocol.SOURCE_CLOSURE_ENTRY_MODULES),
        "required_non_import_files": required_files,
        "relative_source_files": relative_files,
        "relative_source_files_sha256": protocol.canonical_json_sha256(relative_files),
        "import_edges": [],
        "import_graph_sha256": protocol.canonical_json_sha256([]),
        "module_identities": modules,
        "module_identities_sha256": protocol.canonical_json_sha256(modules),
        "python_behavior_version": "CPython 3.10.0",
        "numpy_behavior_version": "1.0.0",
        "canonical_json_contract_sha256": _digest("canonical-json"),
    }
    payload = {
        "schema": protocol.SOURCE_CLOSURE_SCHEMA,
        "experiment_id": protocol.EXP007_EXPERIMENT_ID,
        "schema_descriptor_sha256": protocol.schema_descriptor_sha256(
            protocol.SOURCE_CLOSURE_SCHEMA
        ),
        "behavior": behavior,
        "audit": {
            "generated_at_utc": "2026-08-12T00:00:00Z",
            "absolute_root_path": "/repo",
            "git_commit": "a" * 40,
            "dirty_files": [],
            "platform": "darwin",
            "python_full_version": "3.10.0",
            "numpy_full_version": "1.0.0",
        },
        "source_closure_fingerprint_sha256": protocol.canonical_json_sha256(behavior),
    }
    return protocol.validate_source_closure(
        protocol.with_payload_hash(payload, "full_payload_sha256")
    )


def _candidate_payload(index: int) -> dict[str, Any]:
    input_sha = _digest(f"input-{index}")
    frame_rate_hz = candidate_source.GLOBAL_CONSTANT_JUMP_CONSTANTS.expected_frame_rate_hz
    frame_count = 4000 + index
    (
        _,
        frame_rate_hz,
        coverage_start_ms,
        coverage_end_ms,
        min_period_frames,
        max_period_frames,
    ) = candidate_source._prediction_geometry(  # noqa: SLF001
        _FrameCountSignal(frame_count),
        frame_rate_hz,
        candidate_source.GLOBAL_CONSTANT_JUMP_CONSTANTS,
    )
    peak_time_ms = float(index * 500)
    payload = {
        "schema": protocol.CANDIDATE_PAYLOAD_SCHEMA,
        "beat_peaks": [
            {
                "frame_index": index,
                "refined_frame": float(index) + 0.25,
                "time_ms": peak_time_ms,
                "confidence": 0.9,
            }
        ],
        "downbeat_peaks": [
            {
                "frame_index": index + 1,
                "refined_frame": float(index) + 1.25,
                "time_ms": peak_time_ms + 250.0,
                "confidence": 0.8,
            }
        ],
        "tempo_candidates": [
            {"bpm": 120.0, "source": "autocorrelation", "score": 1.0}
        ],
        "origin_candidates": [
            {"anchor_id": 0, "time_ms": peak_time_ms, "bpm": 120.0, "score": 0.5}
        ],
        "boundary_candidates": [
            {
                "anchor_id": 0,
                "time_ms": peak_time_ms,
                "source_peak_index": 0,
                "source_peak_time_ms": peak_time_ms,
                "source_peak_confidence": 0.9,
                "rank_score": 1.0,
                "evidence_mode": "ordinary",
                "left_period_ms": 500.0,
                "right_period_ms": 500.0,
                "ordinary_score": 1.0,
                "super_score": None,
                "downbeat_bonus": 0.0,
                "nearest_downbeat_distance_ms": 0.0,
            }
        ],
        "diagnostics": {
            "candidate_contract_version": candidate_source.CANDIDATE_CONTRACT_VERSION,
            "constants_json_sha256": (
                candidate_source.GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON_SHA256
            ),
            "pulse_correlation_version": candidate_source.PULSE_CORRELATION_VERSION,
            "boundary_candidate_score_version": (
                candidate_source.BOUNDARY_CANDIDATE_SCORE_VERSION
            ),
            "frame_count": frame_count,
            "frame_rate_hz": frame_rate_hz,
            "coverage_start_ms": coverage_start_ms,
            "coverage_end_ms": coverage_end_ms,
            "min_period_frames": min_period_frames,
            "max_period_frames": max_period_frames,
            "beat_peak_count": 1,
            "downbeat_peak_count": 1,
            "tempo_candidate_count": 1,
            "origin_candidate_count": 1,
            "boundary_candidate_count": 1,
            "input_signal_sha256": input_sha,
            "candidate_fingerprint": SHA1,
        },
    }
    payload["diagnostics"]["candidate_fingerprint"] = _candidate_fingerprint(payload)
    return payload


def _candidate_fingerprint(payload: Mapping[str, Any]) -> str:
    return candidate_source._candidate_fingerprint(  # noqa: SLF001
        tempo_candidates=tuple(
            candidate_source.TempoCandidate(**candidate)
            for candidate in payload["tempo_candidates"]
        ),
        origin_candidates=tuple(
            candidate_source.OriginCandidate(**candidate)
            for candidate in payload["origin_candidates"]
        ),
        boundary_candidates=tuple(
            candidate_source.BoundaryCandidate(**candidate)
            for candidate in payload["boundary_candidates"]
        ),
        beat_peaks=tuple(
            candidate_source.MaterializedPeak(**peak)
            for peak in payload["beat_peaks"]
        ),
        downbeat_peaks=tuple(
            candidate_source.MaterializedPeak(**peak)
            for peak in payload["downbeat_peaks"]
        ),
        input_signal_sha256=payload["diagnostics"]["input_signal_sha256"],
    )


class _FrameCountSignal:
    def __init__(self, frame_count: int) -> None:
        self.shape = (frame_count,)


def _source_arm_stage_summary(
    *,
    arm: str = "S30",
    rows: Sequence[Mapping[str, Any]] | None = None,
    run_config_fingerprint_sha256: str | None = None,
    candidate_reference_manifest_sha256: str = SHA9,
    candidate_reference_entry_payload_sha256s: Sequence[str] | None = None,
) -> dict[str, Any]:
    validated_rows = (
        [_row_result(index, arm=arm) for index in range(16)]
        if rows is None
        else [protocol.validate_row_result(row) for row in rows]
    )
    refs = [
        protocol.make_completed_row_ref(
            row_index=row["row_index"],
            cache_audio_key=row["cache_audio_key"],
            identity_payload_sha256=row["identity_payload_sha256"],
            row_payload_sha256=row["row_payload_sha256"],
            candidate_reference_entry_payload_sha256=(
                _digest(f"reference-{index}")
                if candidate_reference_entry_payload_sha256s is None
                else candidate_reference_entry_payload_sha256s[index]
            ),
        )
        for index, row in enumerate(validated_rows)
    ]
    keys = [row["audio_group_key"] for row in validated_rows]
    full = protocol.make_audio_set_binding(keys)
    empty = protocol.make_audio_set_binding([])
    denominators = {
        "stage_audio_count": 16,
        "stage_audio": full,
        "cache_valid_audio": full,
        "projection_evaluable_audio": full,
        "candidate_accepted_audio": full,
        "candidate_fallback_audio": empty,
        "selected_product_fallback_audio": empty,
        "baseline_accepted_audio": full,
        "product_grid_available_audio": full,
        "no_origin_or_path_audio": empty,
        "resource_cap_fallback_audio": empty,
        "overlap_available_audio": full,
    }
    gates = {
        "candidate_fallback_rate": protocol.make_rate_value(0, 16),
        "selected_product_fallback_rate": protocol.make_rate_value(0, 16),
        "no_origin_or_path_rate": protocol.make_rate_value(0, 16),
        "runtime_seconds": protocol.make_stats_value(
            [row["runtime"]["audio_arm_seconds"] for row in validated_rows]
        ),
        "worker_rss_bytes": protocol.make_stats_value([1024.0] * 4),
        "candidate_seam_ms": protocol.make_stats_value([0.0] * 16),
        "candidate_section_count": protocol.make_stats_value([1.0] * 16),
        "row_json_bytes": protocol.make_stats_value([1000.0] * 16),
        "every_row_under_180_seconds": True,
        "seam_zero": True,
        "section_cap_valid": True,
        "row_byte_cap_valid": True,
        "replay_schema_source_cache_candidate_v2_consistent": True,
    }
    payload = {
        "schema": protocol.SOURCE_ARM_STAGE_SUMMARY_SCHEMA,
        "experiment_id": protocol.EXP007_EXPERIMENT_ID,
        "stage": protocol.EXP007_SCHEDULE_STAGE,
        "schema_descriptor_sha256": protocol.schema_descriptor_sha256(
            protocol.SOURCE_ARM_STAGE_SUMMARY_SCHEMA
        ),
        "schedule_arm": arm,
        "run_config_fingerprint_sha256": (
            validated_rows[0]["run_config_fingerprint_sha256"]
            if run_config_fingerprint_sha256 is None
            else run_config_fingerprint_sha256
        ),
        "source_closure_fingerprint_sha256": validated_rows[0][
            "source_closure_fingerprint_sha256"
        ],
        "selector_manifest_sha256": validated_rows[0]["selector_manifest_sha256"],
        "candidate_reference_manifest_sha256": candidate_reference_manifest_sha256,
        "row_count": 16,
        "row_refs": refs,
        "row_payloads_sha256": protocol.canonical_json_sha256(refs),
        "denominators": denominators,
        "gates": gates,
        "runtime_summary": {
            "row_seconds": protocol.make_stats_value(
                [row["runtime"]["audio_arm_seconds"] for row in validated_rows]
            ),
            "aggregate_wall_seconds": 16.0,
        },
        "rss_summary": {
            "worker_count": 4,
            "worker_lifetime_bytes": [1024, 1024, 1024, 1024],
            "arm_max_worker_bytes": 1024,
        },
    }
    return protocol.validate_source_arm_stage_summary(
        protocol.with_payload_hash(payload, "summary_fingerprint_sha256")
    )


def _clone(value: Any) -> Any:
    return protocol.load_json_strict(protocol.canonical_json_bytes(value))


def _digest(value: str) -> str:
    return protocol.canonical_json_sha256({"value": value})


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "pulsefield_model").is_dir():
            return parent
    raise AssertionError("repo root not found")
