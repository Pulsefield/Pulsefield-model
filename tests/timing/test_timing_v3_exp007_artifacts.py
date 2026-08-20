from __future__ import annotations

import copy
import hashlib
import threading
from pathlib import Path
from typing import Any

import pytest

from pulsefield_model.timing.evaluation import exp007_artifacts as artifacts
from pulsefield_model.timing.evaluation import exp007_protocol as protocol
from pulsefield_model.timing.v3 import global_constant_jump as candidate_source


SHA1 = "1" * 64
SHA2 = "2" * 64
SHA3 = "3" * 64
SHA4 = "4" * 64
SHA5 = "5" * 64


def test_reference_bundle_manifest_round_trip_refs_only_and_atomic_immutable(
    tmp_path: Path,
) -> None:
    _, candidates = _publish_s30_reference_bundles(tmp_path)

    manifest = artifacts.build_candidate_reference_manifest(
        root=tmp_path,
        stage=protocol.EXP007_SCHEDULE_STAGE,
        input_manifest_sha256=SHA1,
        source_closure_fingerprint_sha256=SHA2,
        reference_arm="S30",
    )
    manifest_rel = artifacts.publish_candidate_reference_manifest(
        root=tmp_path,
        manifest=manifest,
    )
    loaded = artifacts.read_candidate_reference_manifest(tmp_path, manifest_rel)

    assert loaded == manifest
    assert loaded["row_count"] == 16
    assert [entry["row_index"] for entry in loaded["entries"]] == list(range(16))
    assert all("candidate_payload" not in entry for entry in loaded["entries"])
    assert all("row" not in entry for entry in loaded["entries"])
    loaded_bundle = artifacts.read_candidate_reference_row_bundle(
        tmp_path,
        loaded["entries"][0]["bundle_relative_path"],
    )
    assert loaded["entries"][0]["entry_payload_sha256"] == loaded_bundle["entry"][
        "entry_payload_sha256"
    ]
    assert loaded_bundle["entry"]["candidate_payload"] == candidates[0]

    artifacts.publish_candidate_reference_manifest(root=tmp_path, manifest=manifest)
    divergent = dict(manifest, source_closure_fingerprint_sha256=SHA3)
    divergent["manifest_fingerprint_sha256"] = protocol.payload_hash(
        divergent,
        "manifest_fingerprint_sha256",
    )
    with pytest.raises(artifacts.Exp007ArtifactError, match="destination differs"):
        artifacts.write_json_atomic(
            tmp_path / manifest_rel,
            divergent,
            byte_cap=protocol.EXP007_CANDIDATE_REFERENCE_MANIFEST_BYTE_CAP,
            context="CandidateReferenceManifest",
        )


def test_reference_manifest_reopens_bundles_and_rejects_stale_swapped_and_aliases(
    tmp_path: Path,
) -> None:
    _publish_s30_reference_bundles(tmp_path)
    manifest = artifacts.build_candidate_reference_manifest(
        root=tmp_path,
        stage=protocol.EXP007_SCHEDULE_STAGE,
        input_manifest_sha256=SHA1,
        source_closure_fingerprint_sha256=SHA2,
        reference_arm="S30",
    )

    stale = _refingerprinted_manifest(manifest)
    stale["entries"][0] = dict(stale["entries"][0], candidate_payload_sha256=SHA5)
    stale = _refingerprinted_manifest(stale)
    with pytest.raises(artifacts.Exp007ArtifactError, match="does not match bundle"):
        artifacts.validate_candidate_reference_manifest(stale, root=tmp_path)

    swapped = _refingerprinted_manifest(manifest)
    swapped["entries"][0], swapped["entries"][1] = (
        swapped["entries"][1],
        swapped["entries"][0],
    )
    swapped = _refingerprinted_manifest(swapped)
    with pytest.raises(artifacts.Exp007ArtifactError, match="canonical"):
        artifacts.validate_candidate_reference_manifest(swapped, root=tmp_path)

    aliased = _refingerprinted_manifest(manifest)
    aliased["entries"][0] = dict(
        aliased["entries"][0],
        bundle_relative_path="candidate_reference/schedule16/S30/../S30/row-00000.bundle.json",
    )
    aliased = _refingerprinted_manifest(aliased)
    with pytest.raises(artifacts.Exp007ArtifactError, match="alias component"):
        artifacts.validate_candidate_reference_manifest(aliased, root=tmp_path)

    symlink_manifest = _refingerprinted_manifest(manifest)
    link_rel = "candidate_reference/schedule16/S30/link.bundle.json"
    link_path = tmp_path / link_rel
    link_path.symlink_to(tmp_path / symlink_manifest["entries"][0]["bundle_relative_path"])
    bundle = artifacts.read_candidate_reference_row_bundle(
        tmp_path,
        symlink_manifest["entries"][0]["bundle_relative_path"],
    )
    with pytest.raises(artifacts.Exp007ArtifactError, match="canonical"):
        artifacts.make_candidate_reference_ref(
            bundle=bundle,
            bundle_relative_path=link_rel,
        )


def test_candidate_artifact_publish_paths_are_canonical(tmp_path: Path) -> None:
    context = _candidate_global_context(tmp_path)
    bundle = artifacts.read_candidate_reference_row_bundle(
        tmp_path,
        context["reference"]["entries"][0]["bundle_relative_path"],
    )
    with pytest.raises(artifacts.Exp007ArtifactError, match="canonical"):
        artifacts.publish_candidate_reference_row_bundle(
            root=tmp_path,
            bundle=bundle,
            relative_path="candidate_reference/schedule16/S30/alias-row.bundle.json",
        )
    with pytest.raises(artifacts.Exp007ArtifactError, match="canonical"):
        artifacts.publish_candidate_reference_manifest(
            root=tmp_path,
            manifest=context["reference"],
            relative_path="candidate_reference/schedule16/S30/alias-manifest.json",
        )
    global_manifest = artifacts.build_candidate_global_manifest(
        root=tmp_path,
        reference_manifest=context["reference"],
        selector_manifest_sha256=SHA1,
        source_closure_fingerprint_sha256=SHA2,
        run_configs_by_arm=context["run_configs_by_arm"],
        arm_rows_by_arm=context["rows_by_arm"],
        candidate_payloads_by_arm=context["payloads_by_arm"],
    )
    with pytest.raises(artifacts.Exp007ArtifactError, match="canonical"):
        artifacts.publish_candidate_global_manifest(
            root=tmp_path,
            manifest=global_manifest,
            reference_manifest=context["reference"],
            run_configs_by_arm=context["run_configs_by_arm"],
            arm_rows_by_arm=context["rows_by_arm"],
            candidate_payloads_by_arm=context["payloads_by_arm"],
            relative_path="candidate_reference/schedule16/global-alias.json",
        )
    with pytest.raises(artifacts.Exp007ArtifactError, match="canonical"):
        artifacts.publish_row_result(
            root=tmp_path,
            row=context["rows_by_arm"]["S60"][0],
            relative_path="rows/schedule16/S60/alias-row.json",
        )


def test_reference_arm_prefix_requires_complete_bundles_and_contiguous_order(
    tmp_path: Path,
) -> None:
    row0, candidate0 = _row_and_candidate(0, arm="S30")
    row1, candidate1 = _row_and_candidate(1, arm="S30")
    artifacts.publish_candidate_reference_row_bundle(
        root=tmp_path,
        bundle=artifacts.make_candidate_reference_row_bundle(
            stage=protocol.EXP007_SCHEDULE_STAGE,
            schedule_arm="S30",
            row=row0,
            candidate_payload=candidate0,
            input_signal_sha256=_input_sha(0),
        ),
    )
    artifacts.publish_candidate_reference_row_bundle(
        root=tmp_path,
        bundle=artifacts.make_candidate_reference_row_bundle(
            stage=protocol.EXP007_SCHEDULE_STAGE,
            schedule_arm="S30",
            row=row1,
            candidate_payload=candidate1,
            input_signal_sha256=_input_sha(1),
        ),
    )

    prefix = artifacts.validate_reference_arm_bundle_prefix(
        root=tmp_path,
        stage=protocol.EXP007_SCHEDULE_STAGE,
        schedule_arm="S30",
        input_manifest_sha256=SHA1,
        source_closure_fingerprint_sha256=SHA2,
    )
    assert [bundle["row_index"] for bundle in prefix] == [0, 1]

    row_only_root = tmp_path / "row-only"
    artifacts.publish_row_result(root=row_only_root, row=row0)
    with pytest.raises(artifacts.Exp007ArtifactError, match="row-only prefix"):
        artifacts.validate_reference_arm_bundle_prefix(
            root=row_only_root,
            stage=protocol.EXP007_SCHEDULE_STAGE,
            schedule_arm="S30",
            input_manifest_sha256=SHA1,
            source_closure_fingerprint_sha256=SHA2,
        )

    gapped_root = tmp_path / "gapped"
    artifacts.publish_candidate_reference_row_bundle(
        root=gapped_root,
        bundle=artifacts.make_candidate_reference_row_bundle(
            stage=protocol.EXP007_SCHEDULE_STAGE,
            schedule_arm="S30",
            row=row0,
            candidate_payload=candidate0,
            input_signal_sha256=_input_sha(0),
        ),
    )
    row2, candidate2 = _row_and_candidate(2, arm="S30")
    artifacts.publish_candidate_reference_row_bundle(
        root=gapped_root,
        bundle=artifacts.make_candidate_reference_row_bundle(
            stage=protocol.EXP007_SCHEDULE_STAGE,
            schedule_arm="S30",
            row=row2,
            candidate_payload=candidate2,
            input_signal_sha256=_input_sha(2),
        ),
    )
    with pytest.raises(artifacts.Exp007ArtifactError, match="gapped"):
        artifacts.validate_reference_arm_bundle_prefix(
            root=gapped_root,
            stage=protocol.EXP007_SCHEDULE_STAGE,
            schedule_arm="S30",
            input_manifest_sha256=SHA1,
            source_closure_fingerprint_sha256=SHA2,
        )


def test_later_arm_prefix_requires_s30_manifest_and_direct_candidate_byte_match(
    tmp_path: Path,
) -> None:
    _, candidates = _publish_s30_reference_bundles(tmp_path)
    manifest = artifacts.build_candidate_reference_manifest(
        root=tmp_path,
        stage=protocol.EXP007_SCHEDULE_STAGE,
        input_manifest_sha256=SHA1,
        source_closure_fingerprint_sha256=SHA2,
        reference_arm="S30",
    )
    artifacts.publish_candidate_reference_manifest(root=tmp_path, manifest=manifest)
    s60_rows = [_row_and_candidate(index, arm="S60")[0] for index in range(2)]
    for row in s60_rows:
        artifacts.publish_row_result(root=tmp_path, row=row)

    prefix = artifacts.validate_later_arm_row_prefix(
        root=tmp_path,
        schedule_arm="S60",
        candidate_payloads_by_row_index=candidates,
    )
    assert [row["row_index"] for row in prefix] == [0, 1]

    mutated = list(candidates)
    mutated[1] = _candidate_payload(1, variant="different")
    with pytest.raises(artifacts.Exp007ArtifactError, match="candidate bytes mismatch"):
        artifacts.validate_later_arm_row_prefix(
            root=tmp_path,
            schedule_arm="S60",
            candidate_payloads_by_row_index=mutated,
        )

    missing_reference_root = tmp_path / "missing-reference"
    artifacts.publish_row_result(
        root=missing_reference_root,
        row=_row_and_candidate(0, arm="S90")[0],
    )
    with pytest.raises(FileNotFoundError):
        artifacts.validate_later_arm_row_prefix(
            root=missing_reference_root,
            schedule_arm="S90",
            candidate_payloads_by_row_index=candidates,
        )


def test_candidate_payload_uses_source_constants_and_fingerprint_contract() -> None:
    payload = _candidate_payload(0)
    serialized = artifacts.serialize_candidate_payload(payload)

    assert payload["diagnostics"]["candidate_contract_version"] == (
        candidate_source.CANDIDATE_CONTRACT_VERSION
    )
    assert payload["diagnostics"]["constants_json_sha256"] == (
        candidate_source.GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON_SHA256
    )
    assert serialized.candidate_fingerprint == _candidate_fingerprint(payload)
    assert serialized.field_set_sha256 == protocol.candidate_payload_field_set_sha256()


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda payload: payload["diagnostics"].__setitem__(
                "candidate_contract_version",
                "stale-v0",
            ),
            "candidate_contract_version",
        ),
        (
            lambda payload: payload["diagnostics"].__setitem__(
                "constants_json_sha256",
                SHA5,
            ),
            "constants_json_sha256",
        ),
        (
            lambda payload: payload["diagnostics"].__setitem__(
                "candidate_fingerprint",
                SHA5,
            ),
            "candidate_fingerprint",
        ),
        (
            lambda payload: payload["boundary_candidates"][0].__setitem__(
                "source_peak_time_ms",
                999.0,
            ),
            "source peak fields are stale",
        ),
        (
            lambda payload: payload["origin_candidates"][0].__setitem__(
                "anchor_id",
                7,
            ),
            "origin anchor IDs",
        ),
        (
            lambda payload: payload["boundary_candidates"][0].__setitem__(
                "left_period_ms",
                0.0,
            ),
            "left_period_ms",
        ),
        (
            lambda payload: _append_out_of_order_peak(payload),
            "beat_peaks",
        ),
        (
            lambda payload: _exceed_tempo_candidate_cap(payload),
            "tempo candidate cap",
        ),
    ],
)
def test_candidate_payload_rejects_stale_and_nondeterministic_payloads(
    mutator: Any,
    message: str,
) -> None:
    payload = _candidate_payload(1)
    mutator(payload)

    with pytest.raises(artifacts.Exp007ArtifactError, match=message):
        artifacts.serialize_candidate_payload(payload)


def test_candidate_reference_entry_binds_input_signal_sha_to_payload() -> None:
    row, candidate = _row_and_candidate(0, arm="S30")

    with pytest.raises(artifacts.Exp007ArtifactError, match="input signal SHA"):
        artifacts.make_candidate_reference_entry(
            row=row,
            candidate_payload=candidate,
            input_signal_sha256=SHA5,
        )


def test_candidate_reference_entry_binds_row_restricted_prediction_to_payload() -> None:
    row, candidate = _row_and_candidate(0, arm="S30")
    mismatched_row = _row_with_restricted_input_signal(row, SHA5)

    with pytest.raises(artifacts.Exp007ArtifactError, match="row candidate input signal"):
        artifacts.make_candidate_reference_entry(
            row=mismatched_row,
            candidate_payload=candidate,
            input_signal_sha256=candidate["diagnostics"]["input_signal_sha256"],
        )


def test_candidate_reference_bundle_revalidates_persisted_row_candidate_input_signal() -> None:
    row, candidate = _row_and_candidate(0, arm="S30")
    bundle = artifacts.make_candidate_reference_row_bundle(
        stage=protocol.EXP007_SCHEDULE_STAGE,
        schedule_arm="S30",
        row=row,
        candidate_payload=candidate,
        input_signal_sha256=candidate["diagnostics"]["input_signal_sha256"],
    )
    tampered = copy.deepcopy(bundle)
    tampered["row"] = _row_with_restricted_input_signal(row, SHA5)
    tampered["entry"] = _refingerprinted_candidate_reference_entry(
        dict(
            tampered["entry"],
            bound_row_payload_sha256=tampered["row"]["row_payload_sha256"],
        )
    )
    tampered = _refingerprinted_candidate_reference_bundle(tampered)

    with pytest.raises(artifacts.Exp007ArtifactError, match="row candidate input signal"):
        artifacts.validate_candidate_reference_row_bundle(tampered)


def test_candidate_global_manifest_binds_s30_reference_and_all_arm_rows(
    tmp_path: Path,
) -> None:
    context = _candidate_global_context(tmp_path)
    reference = context["reference"]
    run_configs_by_arm = context["run_configs_by_arm"]
    rows_by_arm = context["rows_by_arm"]
    payloads_by_arm = context["payloads_by_arm"]

    global_manifest = artifacts.build_candidate_global_manifest(
        root=tmp_path,
        reference_manifest=reference,
        selector_manifest_sha256=SHA1,
        source_closure_fingerprint_sha256=SHA2,
        run_configs_by_arm=run_configs_by_arm,
        arm_rows_by_arm=rows_by_arm,
        candidate_payloads_by_arm=payloads_by_arm,
    )

    assert global_manifest["row_count"] == 16
    assert set(global_manifest["entries"][0]["arm_row_payload_sha256"]) == set(
        protocol.EXP007_EXECUTION_ORDER
    )
    artifacts.validate_candidate_global_manifest(
        global_manifest,
        reference_manifest=reference,
        root=tmp_path,
        run_configs_by_arm=run_configs_by_arm,
        arm_rows_by_arm=rows_by_arm,
        candidate_payloads_by_arm=payloads_by_arm,
    )
    with pytest.raises(TypeError):
        artifacts.validate_candidate_global_manifest(global_manifest)  # type: ignore[call-arg]

    manifest_rel = artifacts.publish_candidate_global_manifest(
        root=tmp_path,
        manifest=global_manifest,
        reference_manifest=reference,
        run_configs_by_arm=run_configs_by_arm,
        arm_rows_by_arm=rows_by_arm,
        candidate_payloads_by_arm=payloads_by_arm,
    )
    assert artifacts.read_candidate_global_manifest(
        tmp_path,
        manifest_rel,
        reference_manifest=reference,
        run_configs_by_arm=run_configs_by_arm,
        arm_rows_by_arm=rows_by_arm,
        candidate_payloads_by_arm=payloads_by_arm,
    ) == global_manifest

    payloads_by_arm["S90"] = list(context["candidates"])
    payloads_by_arm["S90"][0] = _candidate_payload(0, variant="different")
    with pytest.raises(artifacts.Exp007ArtifactError, match="candidate bytes mismatch"):
        artifacts.build_candidate_global_manifest(
            root=tmp_path,
            reference_manifest=reference,
            selector_manifest_sha256=SHA1,
            source_closure_fingerprint_sha256=SHA2,
            run_configs_by_arm=run_configs_by_arm,
            arm_rows_by_arm=rows_by_arm,
            candidate_payloads_by_arm=payloads_by_arm,
        )

    missing_arm_rows = dict(rows_by_arm)
    missing_arm_rows.pop("S64")
    with pytest.raises(artifacts.Exp007ArtifactError, match="all schedule arms"):
        artifacts.build_candidate_global_manifest(
            root=tmp_path,
            reference_manifest=reference,
            selector_manifest_sha256=SHA1,
            source_closure_fingerprint_sha256=SHA2,
            run_configs_by_arm=run_configs_by_arm,
            arm_rows_by_arm=missing_arm_rows,
            candidate_payloads_by_arm={
                arm: list(context["candidates"])
                for arm in protocol.EXP007_EXECUTION_ORDER
            },
        )

    tampered_row_sha = _refingerprinted_manifest(global_manifest)
    tampered_row_sha["entries"][0]["arm_row_payload_sha256"]["S60"] = SHA5
    tampered_row_sha = _refingerprinted_manifest(tampered_row_sha)
    with pytest.raises(artifacts.Exp007ArtifactError, match="arm row SHA mismatch"):
        artifacts.publish_candidate_global_manifest(
            root=tmp_path,
            manifest=tampered_row_sha,
            reference_manifest=reference,
            run_configs_by_arm=run_configs_by_arm,
            arm_rows_by_arm=rows_by_arm,
            candidate_payloads_by_arm={
                arm: list(context["candidates"])
                for arm in protocol.EXP007_EXECUTION_ORDER
            },
        )


def test_candidate_global_manifest_rejects_cross_arm_source_mismatch_matrix(
    tmp_path: Path,
) -> None:
    for case, expected in (
        ("cache_sha", "cache_identity"),
        ("restricted_beat", "restricted_prediction"),
        ("current_v2_reason", "current_v2"),
        ("run_config_projection", "run config projection"),
        ("row_run_config", "row run config"),
    ):
        root = tmp_path / case
        context = _candidate_global_context(root)
        run_configs_by_arm = _clone(context["run_configs_by_arm"])
        rows_by_arm = _clone(context["rows_by_arm"])
        payloads_by_arm = _clone(context["payloads_by_arm"])
        if case == "cache_sha":
            rows_by_arm["S60"][0] = _row_with_cache_sha(
                rows_by_arm["S60"][0],
                SHA5,
            )
        elif case == "restricted_beat":
            rows_by_arm["S60"][0] = _row_with_restricted_prediction_field(
                rows_by_arm["S60"][0],
                "beat_bytes_sha256",
                SHA5,
            )
        elif case == "current_v2_reason":
            rows_by_arm["S60"][0] = _row_with_baseline_unavailable(
                rows_by_arm["S60"][0],
            )
        elif case == "run_config_projection":
            run_configs_by_arm["S60"] = _schedule_run_config(
                "S60",
                cache_config_sha256=SHA5,
            )
        else:
            rows_by_arm["S60"][0] = _row_with_run_config_sha(
                rows_by_arm["S60"][0],
                SHA5,
            )
        with pytest.raises(artifacts.Exp007ArtifactError, match=expected):
            artifacts.build_candidate_global_manifest(
                root=root,
                reference_manifest=context["reference"],
                selector_manifest_sha256=SHA1,
                source_closure_fingerprint_sha256=SHA2,
                run_configs_by_arm=run_configs_by_arm,
                arm_rows_by_arm=rows_by_arm,
                candidate_payloads_by_arm=payloads_by_arm,
            )


def test_repair80_reference_manifest_uses_selected_arm_and_eighty_rows(
    tmp_path: Path,
) -> None:
    for index in range(80):
        row, candidate = _row_and_candidate(
            index,
            stage=protocol.EXP007_REPAIR_STAGE,
            arm="S64",
            selector_sha=SHA1,
            input_sha=SHA3,
        )
        artifacts.publish_candidate_reference_row_bundle(
            root=tmp_path,
            bundle=artifacts.make_candidate_reference_row_bundle(
                stage=protocol.EXP007_REPAIR_STAGE,
                schedule_arm="S64",
                row=row,
                candidate_payload=candidate,
                input_signal_sha256=_input_sha(index),
            ),
        )

    manifest = artifacts.build_candidate_reference_manifest(
        root=tmp_path,
        stage=protocol.EXP007_REPAIR_STAGE,
        input_manifest_sha256=SHA3,
        source_closure_fingerprint_sha256=SHA2,
        reference_arm="S64",
    )

    assert manifest["row_count"] == 80
    assert manifest["reference_arm"] == "S64"
    assert len(manifest["entries"]) == 80


def test_exclusive_byte_caps_for_payload_bundle_reference_and_global_manifest(
    tmp_path: Path,
) -> None:
    context = _candidate_global_context(tmp_path)
    candidates = context["candidates"]
    reference = context["reference"]
    bundle = artifacts.read_candidate_reference_row_bundle(
        tmp_path,
        reference["entries"][0]["bundle_relative_path"],
    )
    global_manifest = artifacts.build_candidate_global_manifest(
        root=tmp_path,
        reference_manifest=reference,
        selector_manifest_sha256=SHA1,
        source_closure_fingerprint_sha256=SHA2,
        run_configs_by_arm=context["run_configs_by_arm"],
        arm_rows_by_arm=context["rows_by_arm"],
        candidate_payloads_by_arm=context["payloads_by_arm"],
    )

    _assert_exclusive_cap(candidates[0], "CandidatePayload")
    _assert_exclusive_cap(bundle, "CandidateReferenceRowBundle")
    _assert_exclusive_cap(reference, "CandidateReferenceManifest")
    _assert_exclusive_cap(global_manifest, "CandidateGlobalManifest")


def test_read_json_artifact_rejects_at_and_above_cap(tmp_path: Path) -> None:
    payload = _candidate_payload(0)
    path = tmp_path / "payload.json"
    raw = protocol.canonical_json_bytes(payload)
    path.write_bytes(raw)

    assert artifacts.read_json_artifact(
        path,
        byte_cap=len(raw) + 1,
        context="CandidatePayload",
    ) == payload
    for byte_cap in (len(raw), len(raw) - 1):
        with pytest.raises(artifacts.Exp007ArtifactError, match="byte cap"):
            artifacts.read_json_artifact(
                path,
                byte_cap=byte_cap,
                context="CandidatePayload",
            )


def test_read_json_artifact_anchored_rejects_huge_file_without_full_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative_path = "candidate_reference/schedule16/S30/huge.json"
    destination = artifacts.contained_write_path(tmp_path, relative_path)
    with destination.open("wb") as handle:
        handle.truncate(32 * 1024 * 1024)
    requested_reads: list[int] = []
    original_read = artifacts.os.read

    def record_read(file_descriptor: int, byte_count: int) -> bytes:
        requested_reads.append(byte_count)
        return original_read(file_descriptor, byte_count)

    monkeypatch.setattr(artifacts.os, "read", record_read)

    with pytest.raises(artifacts.Exp007ArtifactError, match="byte cap"):
        artifacts.read_json_artifact_anchored(
            tmp_path,
            relative_path,
            byte_cap=1024,
            context="AnchoredHuge",
        )

    assert requested_reads
    assert sum(requested_reads) <= 1024


def test_read_row_result_rejects_parent_swap_during_anchored_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row, _candidate = _row_and_candidate(0, arm="S30")
    relative_path = artifacts.publish_row_result(root=tmp_path, row=row)
    destination = tmp_path / relative_path
    moved_parent = tmp_path / "moved-row-parent"

    def swap_parent(event: str, path: Path) -> None:
        if event != "before_anchored_read":
            return
        assert path == destination
        path.parent.rename(moved_parent)
        path.parent.mkdir(parents=True)

    monkeypatch.setattr(artifacts, "_DIRFD_TEST_HOOK", swap_parent)

    with pytest.raises(artifacts.Exp007ArtifactError, match="destination parent changed"):
        artifacts.read_row_result(tmp_path, relative_path)

    assert not destination.exists()
    assert (moved_parent / destination.name).exists()


def test_contained_write_path_rejects_symlink_before_outside_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(artifacts.Exp007ArtifactError, match="symlink"):
        artifacts.contained_write_path(root, "escape/created/file.json")

    assert not (outside / "created").exists()


def test_contained_write_path_rejects_broken_final_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    parent = root / "candidate_reference"
    parent.mkdir(parents=True)
    broken = parent / "manifest.json"
    broken.symlink_to(root / "missing-target.json")

    with pytest.raises(artifacts.Exp007ArtifactError, match="destination symlink"):
        artifacts.contained_write_path(root, "candidate_reference/manifest.json")

    assert broken.is_symlink()
    assert not broken.exists()


def test_write_json_atomic_rechecks_destination_before_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact.json"
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    original_fsync = artifacts.os.fsync
    installed = False

    def install_symlink_once(file_descriptor: int) -> None:
        nonlocal installed
        if not installed:
            installed = True
            destination.symlink_to(target)
        original_fsync(file_descriptor)

    monkeypatch.setattr(artifacts.os, "fsync", install_symlink_once)

    with pytest.raises(artifacts.Exp007ArtifactError, match="symlink"):
        artifacts.write_json_atomic(
            destination,
            {"ok": True},
            byte_cap=1024,
            context="AtomicTest",
        )

    assert destination.is_symlink()
    assert not list(tmp_path.glob(".artifact.json.*.tmp"))


def test_write_json_atomic_existing_exact_match_fsyncs_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact.json"
    payload = {"ok": True}
    destination.write_bytes(protocol.canonical_json_bytes(payload))
    fsynced: list[tuple[int, int]] = []

    def record_fsync(file_descriptor: int) -> None:
        metadata = artifacts.os.fstat(file_descriptor)
        fsynced.append((metadata.st_dev, metadata.st_ino))

    monkeypatch.setattr(artifacts, "_fsync_directory_fd", record_fsync)

    artifacts.write_json_atomic(
        destination,
        payload,
        byte_cap=1024,
        context="AtomicTest",
    )

    parent_metadata = tmp_path.stat()
    assert fsynced == [(parent_metadata.st_dev, parent_metadata.st_ino)]


def test_write_json_atomic_existing_huge_destination_reads_only_compare_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact.json"
    payload = {"ok": True}
    canonical = protocol.canonical_json_bytes(payload)
    with destination.open("wb") as handle:
        handle.truncate(32 * 1024 * 1024)
    requested_reads: list[int] = []
    original_read = artifacts.os.read

    def record_read(file_descriptor: int, byte_count: int) -> bytes:
        requested_reads.append(byte_count)
        return original_read(file_descriptor, byte_count)

    monkeypatch.setattr(artifacts.os, "read", record_read)

    with pytest.raises(artifacts.Exp007ArtifactError, match="destination differs"):
        artifacts.write_json_atomic(
            destination,
            payload,
            byte_cap=1024,
            context="AtomicTest",
        )

    assert requested_reads
    assert sum(requested_reads) <= len(canonical) + 1


def test_write_json_atomic_rejects_parent_swap_before_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    relative_path = "candidate_reference/manifest.json"
    destination = artifacts.contained_write_path(root, relative_path)
    replacement_parent = tmp_path / "replacement-parent"
    replacement_parent.mkdir()
    moved_parent = tmp_path / "moved-parent"

    def swap_parent(event: str, path: Path) -> None:
        if event != "before_atomic_link":
            return
        assert path == destination
        destination.parent.rename(moved_parent)
        replacement_parent.rename(destination.parent)

    monkeypatch.setattr(artifacts, "_DIRFD_TEST_HOOK", swap_parent)

    with pytest.raises(artifacts.Exp007ArtifactError, match="destination parent changed"):
        artifacts.write_json_atomic(
            destination,
            {"ok": True},
            byte_cap=1024,
            context="AtomicTest",
            root=root,
            relative_path=relative_path,
        )

    assert not destination.exists()
    assert not list(moved_parent.glob(".manifest.json.*.tmp"))


def test_write_json_atomic_two_divergent_writers_never_overwrite_existing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact.json"
    payloads = [{"writer": "a"}, {"writer": "b"}]
    barrier = threading.Barrier(2)
    outcomes: list[tuple[str, str]] = []
    lock = threading.Lock()

    def wait_before_link(event: str, _path: Path) -> None:
        if event == "before_atomic_link":
            barrier.wait(timeout=5.0)

    def publish(payload: dict[str, Any]) -> None:
        try:
            artifacts.write_json_atomic(
                destination,
                payload,
                byte_cap=1024,
                context="AtomicRace",
            )
        except artifacts.Exp007ArtifactError as exc:
            outcome = ("error", str(exc))
        except Exception as exc:  # pragma: no cover - assertion below reports the detail.
            outcome = ("unexpected", repr(exc))
        else:
            outcome = ("ok", payload["writer"])
        with lock:
            outcomes.append(outcome)

    monkeypatch.setattr(artifacts, "_DIRFD_TEST_HOOK", wait_before_link)
    threads = [threading.Thread(target=publish, args=(payload,)) for payload in payloads]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert all(not thread.is_alive() for thread in threads)
    assert len(outcomes) == 2
    assert [status for status, _ in outcomes].count("ok") == 1
    assert any(
        status == "error" and "destination differs" in detail
        for status, detail in outcomes
    )
    assert artifacts.read_json_artifact(
        destination,
        byte_cap=1024,
        context="AtomicRace",
    ) in payloads
    assert not list(tmp_path.glob(".artifact.json.*.tmp"))


def test_empty_exposure_delta_publish_read_resume_and_no_real_guard(
    tmp_path: Path,
) -> None:
    lock = _acquire_run_lock(tmp_path)
    delta = artifacts.build_exposure_delta(
        generated_at_utc="2026-08-12T00:00:00Z",
        prior_exposure_manifest_sha256=SHA1,
        source_closure_fingerprint_sha256=SHA2,
        delta_reason="exp007-no-real-data",
        entries=[],
    )

    rel = artifacts.publish_exposure_delta(
        root=tmp_path,
        delta=delta,
        run_lock=lock,
        require_empty=True,
    )

    assert artifacts.read_exposure_delta(tmp_path, rel, require_empty=True) == delta
    assert artifacts.resume_exposure_delta(tmp_path, rel, require_empty=True) == delta
    artifacts.release_run_lock(lock, owner_run_id="owner-1")


def test_exposure_delta_records_all_observed_payload_kinds_and_hashes() -> None:
    entries = [
        _exposure_entry(index, kind)
        for index, kind in reversed(list(enumerate(sorted(artifacts.OBSERVED_PAYLOAD_KINDS))))
    ]

    delta = artifacts.build_exposure_delta(
        generated_at_utc="2026-08-12T00:00:00Z",
        prior_exposure_manifest_sha256=SHA1,
        source_closure_fingerprint_sha256=SHA2,
        delta_reason="synthetic-observed-keys",
        entries=entries,
    )

    assert delta["entry_count"] == len(artifacts.OBSERVED_PAYLOAD_KINDS)
    assert [entry["cache_audio_key"] for entry in delta["entries"]] == sorted(
        entry["cache_audio_key"] for entry in entries
    )
    assert {entry["observed_payload_kind"] for entry in delta["entries"]} == set(
        artifacts.OBSERVED_PAYLOAD_KINDS
    )
    assert delta["cache_audio_keys_sha256"] == protocol.canonical_json_sha256(
        [entry["cache_audio_key"] for entry in delta["entries"]]
    )
    assert delta["entries_sha256"] == protocol.canonical_json_sha256(delta["entries"])


def test_exposure_delta_rejects_malformed_unsorted_duplicate_and_nonempty_guard(
    tmp_path: Path,
) -> None:
    entry0 = _exposure_entry(0, "identity")
    entry1 = _exposure_entry(1, "runtime")
    nonempty = artifacts.build_exposure_delta(
        generated_at_utc="2026-08-12T00:00:00Z",
        prior_exposure_manifest_sha256=SHA1,
        source_closure_fingerprint_sha256=SHA2,
        delta_reason="synthetic-observed-key",
        entries=[entry0],
    )

    with pytest.raises(artifacts.Exp007ArtifactError, match="must be empty"):
        artifacts.validate_exp007_empty_exposure_delta(nonempty)

    malformed_entry = dict(entry0, unexpected=True)
    with pytest.raises(ValueError, match="extra=\\['unexpected'\\]"):
        artifacts.validate_exposure_entry(malformed_entry)

    with pytest.raises(artifacts.Exp007ArtifactError, match="observed_payload_kind"):
        artifacts.validate_exposure_entry(dict(entry0, observed_payload_kind="candidate"))

    with pytest.raises(artifacts.Exp007ArtifactError, match="duplicate cache_audio_key"):
        artifacts.build_exposure_delta(
            generated_at_utc="2026-08-12T00:00:00Z",
            prior_exposure_manifest_sha256=SHA1,
            source_closure_fingerprint_sha256=SHA2,
            delta_reason="duplicate",
            entries=[entry0, dict(entry1, cache_audio_key=entry0["cache_audio_key"])],
        )

    unsorted = artifacts.build_exposure_delta(
        generated_at_utc="2026-08-12T00:00:00Z",
        prior_exposure_manifest_sha256=SHA1,
        source_closure_fingerprint_sha256=SHA2,
        delta_reason="sorted",
        entries=[entry0, entry1],
    )
    unsorted["entries"] = list(reversed(unsorted["entries"]))
    unsorted = _refingerprinted_exposure_delta(unsorted)
    with pytest.raises(artifacts.Exp007ArtifactError, match="sorted"):
        artifacts.validate_exposure_delta(unsorted)

    lock = _acquire_run_lock(tmp_path)
    empty = artifacts.build_exposure_delta(
        generated_at_utc="2026-08-12T00:00:00Z",
        prior_exposure_manifest_sha256=SHA1,
        source_closure_fingerprint_sha256=SHA2,
        delta_reason="empty",
        entries=[],
    )
    artifacts.publish_exposure_delta(root=tmp_path, delta=empty, run_lock=lock)
    divergent = artifacts.build_exposure_delta(
        generated_at_utc="2026-08-12T00:00:00Z",
        prior_exposure_manifest_sha256=SHA1,
        source_closure_fingerprint_sha256=SHA2,
        delta_reason="divergent-empty",
        entries=[],
    )
    with pytest.raises(artifacts.Exp007ArtifactError, match="destination differs"):
        artifacts.publish_exposure_delta(root=tmp_path, delta=divergent, run_lock=lock)
    artifacts.release_run_lock(lock, owner_run_id="owner-1")


def test_exposure_delta_publish_requires_live_held_lock(
    tmp_path: Path,
) -> None:
    raw_payload = artifacts.build_run_lock_payload(
        root=tmp_path,
        stage=protocol.EXP007_SCHEDULE_STAGE,
        schedule_arm="S30",
        selector_manifest_sha256=SHA1,
        source_closure_fingerprint_sha256=SHA2,
        run_config_fingerprint_sha256=SHA4,
        owner_run_id="owner-1",
        acquired_at_utc="2026-08-12T00:00:00Z",
    )
    delta = artifacts.build_exposure_delta(
        generated_at_utc="2026-08-12T00:00:00Z",
        prior_exposure_manifest_sha256=SHA1,
        source_closure_fingerprint_sha256=SHA2,
        delta_reason="exp007-no-real-data",
        entries=[],
    )

    with pytest.raises(artifacts.Exp007ArtifactError, match="held Exp007RunLock"):
        artifacts.publish_exposure_delta(
            root=tmp_path,
            delta=delta,
            run_lock=raw_payload,  # type: ignore[arg-type]
        )

    lock = _acquire_run_lock(tmp_path)
    observed = artifacts.resume_run_lock(tmp_path)
    assert observed is not None
    assert observed.acquired is False
    with pytest.raises(artifacts.Exp007ArtifactError, match="acquired run lock"):
        artifacts.publish_exposure_delta(root=tmp_path, delta=delta, run_lock=observed)

    artifacts.release_run_lock(lock, owner_run_id="owner-1")
    with pytest.raises(artifacts.Exp007ArtifactError, match="missing"):
        artifacts.publish_exposure_delta(root=tmp_path, delta=delta, run_lock=lock)


def test_exposure_delta_publish_rejects_divergent_lock_source_stage_and_nonempty(
    tmp_path: Path,
) -> None:
    lock = _acquire_run_lock(tmp_path)
    empty_wrong_source = artifacts.build_exposure_delta(
        generated_at_utc="2026-08-12T00:00:00Z",
        prior_exposure_manifest_sha256=SHA1,
        source_closure_fingerprint_sha256=SHA3,
        delta_reason="wrong-source",
        entries=[],
    )
    with pytest.raises(artifacts.Exp007ArtifactError, match="source closure"):
        artifacts.publish_exposure_delta(
            root=tmp_path,
            delta=empty_wrong_source,
            run_lock=lock,
        )

    repair_stage_entry = artifacts.build_exposure_entry(
        cache_audio_key="stage-mismatch-key",
        audio_group_key="stage-mismatch-group",
        exposure_stage=protocol.EXP007_REPAIR_STAGE,
        exposure_reason="wrong-stage",
        first_exposed_at_or_run_id="synthetic-run",
        observed_payload_kind="identity",
        source_manifest_sha256=SHA3,
    )
    wrong_stage = artifacts.build_exposure_delta(
        generated_at_utc="2026-08-12T00:00:00Z",
        prior_exposure_manifest_sha256=SHA1,
        source_closure_fingerprint_sha256=SHA2,
        delta_reason="wrong-stage",
        entries=[repair_stage_entry],
    )
    with pytest.raises(artifacts.Exp007ArtifactError, match="exposure_stage"):
        artifacts.publish_exposure_delta(root=tmp_path, delta=wrong_stage, run_lock=lock)

    schedule_entry = artifacts.build_exposure_entry(
        cache_audio_key="schedule-key",
        audio_group_key="schedule-group",
        exposure_stage=protocol.EXP007_SCHEDULE_STAGE,
        exposure_reason="nonempty",
        first_exposed_at_or_run_id="synthetic-run",
        observed_payload_kind="identity",
        source_manifest_sha256=SHA3,
    )
    nonempty = artifacts.build_exposure_delta(
        generated_at_utc="2026-08-12T00:00:00Z",
        prior_exposure_manifest_sha256=SHA1,
        source_closure_fingerprint_sha256=SHA2,
        delta_reason="nonempty",
        entries=[schedule_entry],
    )
    with pytest.raises(artifacts.Exp007ArtifactError, match="must be empty"):
        artifacts.publish_exposure_delta(root=tmp_path, delta=nonempty, run_lock=lock)

    divergent_payload = artifacts.build_run_lock_payload(
        root=tmp_path,
        stage=protocol.EXP007_SCHEDULE_STAGE,
        schedule_arm="S30",
        selector_manifest_sha256=SHA1,
        source_closure_fingerprint_sha256=SHA3,
        run_config_fingerprint_sha256=SHA4,
        owner_run_id="owner-1",
        acquired_at_utc="2026-08-12T00:00:00Z",
    )
    (tmp_path / lock.relative_path).write_bytes(
        protocol.canonical_json_bytes(divergent_payload)
    )
    with pytest.raises(artifacts.Exp007ArtifactError, match="payload changed"):
        artifacts.publish_exposure_delta(
            root=tmp_path,
            delta=empty_wrong_source,
            run_lock=lock,
        )


def test_run_lock_is_exclusive_owner_released_and_symlink_safe(tmp_path: Path) -> None:
    lock = _acquire_run_lock(tmp_path, owner_run_id="owner-1")

    with pytest.raises(artifacts.Exp007ArtifactError, match="identical payload"):
        _acquire_run_lock(tmp_path, owner_run_id="owner-1")

    with pytest.raises(artifacts.Exp007ArtifactError, match="divergent payload"):
        _acquire_run_lock(tmp_path, owner_run_id="owner-2")

    with pytest.raises(artifacts.Exp007ArtifactError, match="owner mismatch"):
        artifacts.release_run_lock(lock, owner_run_id="wrong-owner")
    assert artifacts.resume_run_lock(tmp_path) is not None

    with pytest.raises(artifacts.Exp007ArtifactError, match="explicit owner"):
        artifacts.release_run_lock(lock)

    artifacts.release_run_lock(lock, owner_run_id="owner-1")
    assert artifacts.resume_run_lock(tmp_path) is None
    with pytest.raises(artifacts.Exp007ArtifactError, match="missing"):
        artifacts.release_run_lock(lock, owner_run_id="owner-1")

    root = tmp_path / "symlink-root"
    rel = artifacts.run_lock_relative_path(protocol.EXP007_SCHEDULE_STAGE, "S30")
    destination = artifacts.contained_write_path(root, rel)
    target = root / "target.json"
    target.write_text("{}", encoding="utf-8")
    destination.symlink_to(target)
    with pytest.raises(artifacts.Exp007ArtifactError, match="symlink"):
        _acquire_run_lock(root)


def test_run_lock_rejects_noncanonical_relative_path_before_write(tmp_path: Path) -> None:
    root = tmp_path / "root"

    with pytest.raises(artifacts.Exp007ArtifactError, match="canonical"):
        _acquire_run_lock(
            root,
            relative_path="locks/timing_v3_experiment_007/schedule16/noncanonical.lock.json",
        )

    assert not root.exists()


def test_run_lock_acquire_rejects_parent_swap_before_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    relative_path = artifacts.run_lock_relative_path(protocol.EXP007_SCHEDULE_STAGE, "S30")
    destination_holder: list[Path] = []
    moved_parent = tmp_path / "moved-lock-parent"

    def swap_parent(event: str, path: Path) -> None:
        if event != "before_lock_create":
            return
        destination_holder.append(path)
        path.parent.rename(moved_parent)
        path.parent.mkdir(parents=True)

    monkeypatch.setattr(artifacts, "_DIRFD_TEST_HOOK", swap_parent)

    with pytest.raises(artifacts.Exp007ArtifactError, match="destination parent changed"):
        _acquire_run_lock(root)

    assert destination_holder
    assert not (root / relative_path).exists()
    assert not list(moved_parent.glob("*.lock.json"))


def test_run_lock_release_rejects_parent_swap_before_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    lock = _acquire_run_lock(root)
    destination = root / lock.relative_path
    moved_parent = tmp_path / "released-lock-parent"

    def swap_parent(event: str, path: Path) -> None:
        if event != "before_lock_unlink":
            return
        assert path == destination
        path.parent.rename(moved_parent)
        path.parent.mkdir(parents=True)

    monkeypatch.setattr(artifacts, "_DIRFD_TEST_HOOK", swap_parent)

    with pytest.raises(artifacts.Exp007ArtifactError, match="destination parent changed"):
        artifacts.release_run_lock(lock, owner_run_id="owner-1")

    assert not destination.exists()
    assert (moved_parent / destination.name).exists()


def test_run_lock_resume_keeps_stale_lock_and_root_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    root_a = tmp_path / "root-a"
    root_b = tmp_path / "root-b"
    lock = _acquire_run_lock(root_a, acquired_at_utc="2000-01-01T00:00:00Z")

    resumed = artifacts.resume_run_lock(root_a)

    assert resumed is not None
    assert resumed.payload == lock.payload
    assert (root_a / lock.relative_path).exists()

    copied_path = artifacts.contained_write_path(root_b, lock.relative_path)
    copied_path.write_bytes((root_a / lock.relative_path).read_bytes())
    with pytest.raises(artifacts.Exp007ArtifactError, match="output root fingerprint"):
        artifacts.read_run_lock(root_b, lock.relative_path)

    resumed_for_release = artifacts.resume_run_lock(root_a)
    assert resumed_for_release is not None
    with pytest.raises(artifacts.Exp007ArtifactError, match="acquired run lock"):
        artifacts.release_run_lock(resumed_for_release, owner_run_id="owner-1")

    artifacts.release_run_lock(lock, owner_run_id="owner-1")


def test_run_lock_release_rejects_divergent_current_payload(tmp_path: Path) -> None:
    lock = _acquire_run_lock(tmp_path)
    divergent_payload = artifacts.build_run_lock_payload(
        root=tmp_path,
        stage=protocol.EXP007_SCHEDULE_STAGE,
        schedule_arm="S30",
        selector_manifest_sha256=SHA1,
        source_closure_fingerprint_sha256=SHA3,
        run_config_fingerprint_sha256=SHA4,
        owner_run_id="owner-1",
        acquired_at_utc="2026-08-12T00:00:00Z",
    )
    (tmp_path / lock.relative_path).write_bytes(
        protocol.canonical_json_bytes(divergent_payload)
    )

    with pytest.raises(artifacts.Exp007ArtifactError, match="payload changed"):
        artifacts.release_run_lock(lock, owner_run_id="owner-1")


def _publish_s30_reference_bundles(
    root: Path,
    *,
    run_configs_by_arm: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    run_config_sha = (
        SHA4
        if run_configs_by_arm is None
        else run_configs_by_arm["S30"]["run_config_fingerprint_sha256"]
    )
    for index in range(16):
        row, candidate = _row_and_candidate(
            index,
            arm="S30",
            run_config_fingerprint_sha256=run_config_sha,
        )
        artifacts.publish_candidate_reference_row_bundle(
            root=root,
            bundle=artifacts.make_candidate_reference_row_bundle(
                stage=protocol.EXP007_SCHEDULE_STAGE,
                schedule_arm="S30",
                row=row,
                candidate_payload=candidate,
                input_signal_sha256=_input_sha(index),
            ),
        )
        rows.append(row)
        candidates.append(candidate)
    return rows, candidates


def _candidate_global_context(root: Path) -> dict[str, Any]:
    run_configs_by_arm = _schedule_run_configs_by_arm()
    _, candidates = _publish_s30_reference_bundles(
        root,
        run_configs_by_arm=run_configs_by_arm,
    )
    reference = artifacts.build_candidate_reference_manifest(
        root=root,
        stage=protocol.EXP007_SCHEDULE_STAGE,
        input_manifest_sha256=SHA1,
        source_closure_fingerprint_sha256=SHA2,
        reference_arm="S30",
    )
    rows_by_arm = {
        arm: [
            _row_and_candidate(
                index,
                arm=arm,
                run_config_fingerprint_sha256=run_configs_by_arm[arm][
                    "run_config_fingerprint_sha256"
                ],
            )[0]
            for index in range(16)
        ]
        for arm in protocol.EXP007_EXECUTION_ORDER
    }
    payloads_by_arm = {
        arm: list(candidates)
        for arm in protocol.EXP007_EXECUTION_ORDER
    }
    return {
        "candidates": candidates,
        "reference": reference,
        "run_configs_by_arm": run_configs_by_arm,
        "rows_by_arm": rows_by_arm,
        "payloads_by_arm": payloads_by_arm,
    }


def _clone(value: Any) -> Any:
    return protocol.load_json_strict(protocol.canonical_json_bytes(value))


def _schedule_run_configs_by_arm() -> dict[str, dict[str, Any]]:
    return {arm: _schedule_run_config(arm) for arm in protocol.EXP007_EXECUTION_ORDER}


def _schedule_run_config(
    arm: str,
    *,
    cache_config_sha256: str = SHA3,
    grid_fitter_config_sha256: str = SHA4,
    weak_config_sha256: str = SHA5,
) -> dict[str, Any]:
    return protocol.make_run_config(
        stage=protocol.EXP007_SCHEDULE_STAGE,
        schedule_arm=arm,
        selector_manifest_sha256=SHA1,
        input_manifest_sha256=SHA1,
        source_closure_fingerprint_sha256=SHA2,
        cache_config_sha256=cache_config_sha256,
        grid_fitter_config_sha256=grid_fitter_config_sha256,
        weak_config_sha256=weak_config_sha256,
    )


def _row_and_candidate(
    index: int,
    *,
    arm: str,
    stage: str = protocol.EXP007_SCHEDULE_STAGE,
    selector_sha: str = SHA1,
    input_sha: str = SHA1,
    run_config_fingerprint_sha256: str = SHA4,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = _candidate_payload(index)
    serialized = artifacts.serialize_candidate_payload(candidate)
    row = protocol.minimal_row_result(
        stage=stage,
        schedule_arm=arm,
        row_index=index,
        cache_audio_key=f"audio-{index:03d}",
        audio_group_key=f"group-{index:03d}",
        identity_payload_sha256=_sha(f"identity-{stage}-{index}"),
        source_closure_fingerprint_sha256=SHA2,
        run_config_fingerprint_sha256=run_config_fingerprint_sha256,
        selector_manifest_sha256=selector_sha,
        input_manifest_sha256=input_sha,
        candidate_payload_schema=serialized.schema,
        candidate_payload_byte_count=serialized.byte_count,
        candidate_payload_field_set_sha256=serialized.field_set_sha256,
        candidate_payload_sha256=serialized.payload_sha256,
        candidate_fingerprint=serialized.candidate_fingerprint,
    )
    return (
        _row_with_restricted_input_signal(
            row,
            candidate["diagnostics"]["input_signal_sha256"],
        ),
        candidate,
    )


def _row_with_restricted_input_signal(
    row: dict[str, Any],
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


def _row_with_cache_sha(row: Mapping[str, Any], cache_sha256: str) -> dict[str, Any]:
    cache_identity = dict(row["cache_identity"], sha256=cache_sha256)
    resume = dict(row["resume"], validated_cache_sha256=cache_sha256)
    return _rebuild_row(row, cache_identity=cache_identity, resume=resume)


def _row_with_restricted_prediction_field(
    row: Mapping[str, Any],
    field_name: str,
    value: Any,
) -> dict[str, Any]:
    return _rebuild_row(
        row,
        restricted_prediction=dict(row["restricted_prediction"], **{field_name: value}),
    )


def _row_with_run_config_sha(
    row: Mapping[str, Any],
    run_config_fingerprint_sha256: str,
) -> dict[str, Any]:
    resume = dict(
        row["resume"],
        validated_config_sha256=run_config_fingerprint_sha256,
    )
    return _rebuild_row(
        row,
        run_config_fingerprint_sha256=run_config_fingerprint_sha256,
        resume=resume,
    )


def _row_with_baseline_unavailable(row: Mapping[str, Any]) -> dict[str, Any]:
    methods = copy.deepcopy(row["methods"])
    methods["baseline"] = protocol.make_method_result(
        method_kind="baseline",
        status="unavailable",
        reason="prediction_too_short",
    )
    denominator_flags = dict(
        row["denominator_flags"],
        baseline_accepted=False,
        current_v2_phase_matched=False,
    )
    return _rebuild_row(
        row,
        methods=methods,
        denominator_flags=denominator_flags,
    )


def _rebuild_row(row: Mapping[str, Any], **updates: Any) -> dict[str, Any]:
    values = {
        "stage": row["stage"],
        "schedule_arm": row["schedule_arm"],
        "row_index": row["row_index"],
        "cache_audio_key": row["cache_audio_key"],
        "audio_group_key": row["audio_group_key"],
        "identity_payload_sha256": row["identity_payload_sha256"],
        "cache_identity": row["cache_identity"],
        "source_closure_fingerprint_sha256": row[
            "source_closure_fingerprint_sha256"
        ],
        "run_config_fingerprint_sha256": row["run_config_fingerprint_sha256"],
        "selector_manifest_sha256": row["selector_manifest_sha256"],
        "input_manifest_sha256": row["input_manifest_sha256"],
        "resume": row["resume"],
        "restricted_prediction": row["restricted_prediction"],
        "candidate_payload_schema": row["candidate_payload_schema"],
        "candidate_payload_byte_count": row["candidate_payload_byte_count"],
        "candidate_payload_field_set_sha256": row[
            "candidate_payload_field_set_sha256"
        ],
        "candidate_payload_sha256": row["candidate_payload_sha256"],
        "candidate_fingerprint": row["candidate_fingerprint"],
        "methods": row["methods"],
        "denominator_flags": row["denominator_flags"],
        "diagnostics_summary": row["diagnostics_summary"],
        "runtime": row["runtime"],
        "rss": row["rss"],
        "hard_guards": row["hard_guards"],
    }
    values.update(updates)
    return protocol.make_row_result(**values)


def _candidate_payload(index: int, *, variant: str = "default") -> dict[str, Any]:
    input_sha = _input_sha(index)
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
    variant_offset = 0.0 if variant == "default" else 0.001
    peak_time_ms = float(index * 500)
    payload = {
        "schema": artifacts.CANDIDATE_PAYLOAD_SCHEMA,
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
            {
                "bpm": 120.0,
                "source": "autocorrelation",
                "score": 1.0 + variant_offset,
            }
        ],
        "origin_candidates": [
            {
                "anchor_id": 0,
                "time_ms": peak_time_ms,
                "bpm": 120.0,
                "score": 0.5,
            }
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


def _append_out_of_order_peak(payload: dict[str, Any]) -> None:
    payload["beat_peaks"].append(
        {
            "frame_index": payload["beat_peaks"][0]["frame_index"] + 1,
            "refined_frame": payload["beat_peaks"][0]["refined_frame"] + 1.0,
            "time_ms": payload["beat_peaks"][0]["time_ms"] - 1.0,
            "confidence": 0.7,
        }
    )
    payload["diagnostics"]["beat_peak_count"] = len(payload["beat_peaks"])
    payload["diagnostics"]["candidate_fingerprint"] = _candidate_fingerprint(payload)


def _exceed_tempo_candidate_cap(payload: dict[str, Any]) -> None:
    cap = candidate_source.GLOBAL_CONSTANT_JUMP_CONSTANTS.max_tempo_candidates_retained
    payload["tempo_candidates"] = [
        {"bpm": 20.0 + float(offset), "source": "autocorrelation", "score": 1.0}
        for offset in range(cap + 1)
    ]
    payload["diagnostics"]["tempo_candidate_count"] = len(payload["tempo_candidates"])
    payload["diagnostics"]["candidate_fingerprint"] = _candidate_fingerprint(payload)


class _FrameCountSignal:
    def __init__(self, frame_count: int) -> None:
        self.shape = (frame_count,)


def _candidate_fingerprint(payload: dict[str, Any]) -> str:
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


def _refingerprinted_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(manifest)
    result.pop("manifest_fingerprint_sha256", None)
    result["ordered_entries_sha256"] = protocol.canonical_json_sha256(result["entries"])
    result["manifest_fingerprint_sha256"] = protocol.payload_hash(
        result,
        "manifest_fingerprint_sha256",
    )
    return result


def _refingerprinted_candidate_reference_entry(entry: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(entry)
    result.pop("entry_payload_sha256", None)
    result["entry_payload_sha256"] = protocol.payload_hash(
        result,
        "entry_payload_sha256",
    )
    return result


def _refingerprinted_candidate_reference_bundle(
    bundle: dict[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(bundle)
    result.pop("bundle_fingerprint_sha256", None)
    result["bundle_fingerprint_sha256"] = protocol.payload_hash(
        result,
        "bundle_fingerprint_sha256",
    )
    return result


def _assert_exclusive_cap(payload: dict[str, Any], context: str) -> None:
    byte_count = len(protocol.canonical_json_bytes(payload))
    assert artifacts.canonical_json_bytes_under_cap(
        payload,
        byte_cap=byte_count + 1,
        context=context,
    )
    with pytest.raises(artifacts.Exp007ArtifactError, match="byte cap"):
        artifacts.canonical_json_bytes_under_cap(
            payload,
            byte_cap=byte_count,
            context=context,
        )
    with pytest.raises(artifacts.Exp007ArtifactError, match="byte cap"):
        artifacts.canonical_json_bytes_under_cap(
            payload,
            byte_cap=byte_count - 1,
            context=context,
        )


def _acquire_run_lock(
    root: Path,
    *,
    owner_run_id: str = "owner-1",
    acquired_at_utc: str = "2026-08-12T00:00:00Z",
    relative_path: str | None = None,
) -> artifacts.Exp007RunLock:
    return artifacts.acquire_run_lock(
        root=root,
        stage=protocol.EXP007_SCHEDULE_STAGE,
        schedule_arm="S30",
        selector_manifest_sha256=SHA1,
        source_closure_fingerprint_sha256=SHA2,
        run_config_fingerprint_sha256=SHA4,
        owner_run_id=owner_run_id,
        acquired_at_utc=acquired_at_utc,
        relative_path=relative_path,
    )


def _exposure_entry(index: int, kind: str) -> dict[str, Any]:
    return artifacts.build_exposure_entry(
        cache_audio_key=f"exposure-audio-{index:03d}",
        audio_group_key=f"exposure-group-{index:03d}",
        exposure_stage=protocol.EXP007_SCHEDULE_STAGE,
        exposure_reason=f"observed-{kind}",
        first_exposed_at_or_run_id="synthetic-run",
        observed_payload_kind=kind,
        source_manifest_sha256=SHA3,
    )


def _refingerprinted_exposure_delta(delta: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(delta)
    result["cache_audio_keys_sha256"] = protocol.canonical_json_sha256(
        [entry["cache_audio_key"] for entry in result["entries"]]
    )
    result["entries_sha256"] = protocol.canonical_json_sha256(result["entries"])
    result["manifest_fingerprint_sha256"] = protocol.payload_hash(
        result,
        "manifest_fingerprint_sha256",
    )
    return result


def _input_sha(index: int) -> str:
    return _sha(f"input-{index}")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
