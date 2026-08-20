from __future__ import annotations

import hashlib
import importlib
import multiprocessing as mp
import pickle
import signal
import time
from dataclasses import dataclass, replace
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any, Callable

import pytest

from pulsefield_model.timing.evaluation import exp007_artifacts as artifacts
from pulsefield_model.timing.evaluation.exp007_metrics import (
    SourceMetricRow,
    evaluate_source_arm,
    rate_value,
    ratio_value,
    select_source_schedule,
    stats_value,
)
from pulsefield_model.timing.evaluation import exp007_protocol as protocol
from pulsefield_model.timing.evaluation import exp007_runner as exp007_runner_module
from pulsefield_model.timing.evaluation.exp007_runner import (
    EXP007_EXECUTION_ORDER,
    EXP007_REPAIR_STAGE,
    Exp007ArmSupervisor,
    Exp007RunnerConfig,
    Exp007RunnerError,
    Exp007SyntheticIdentity,
    Exp007SyntheticWorkerResult,
    Exp007WorkerFailure,
    Exp007WorkerTimeout,
    RepairSummaryValidationContext,
    RunnerStageTelemetry,
    RowFinishedEvent,
    RowStartedEvent,
    StageSummaryPublication,
    WorkerHello,
    _accept_initial_handshake,
    _handle_control_eof,
    _process_control_payload,
    _terminate_pool_bounded,
    canonical_sha256,
    normalize_ru_maxrss_bytes,
    run_synthetic_exp007_arm,
    run_synthetic_exp007_schedule,
)
from pulsefield_model.timing.evaluation.exp007_weak_evidence import (
    BoundaryEvidence,
    BoundarySummary,
    ComparatorAvailability,
    DriftSummary,
    ObjectGridSummary,
    PhaseSummary,
    PredictionRowRef,
    Repair80MetricRow,
    WeakMetricRow,
    evaluate_repair80,
    evaluate_schedule_weak_veto,
    make_repair80_summary,
    make_schedule_weak_success_outcome,
    make_schedule_weak_veto_summary,
    make_weak_row,
)
from pulsefield_model.timing.v3 import global_constant_jump as candidate_source


def test_supervisor_hello_ack_slot_map_and_ordered_prefix_commit(
    tmp_path: Path,
) -> None:
    identities = _identities(16)
    config, row_callback, stage_summary_factory = _schedule_artifact_config_and_callback(
        tmp_path,
        identities,
        use_spawn_pool=False,
    )
    supervisor = Exp007ArmSupervisor(
        identities=identities,
        config=config,
        initial_pid_order=(101, 102, 103, 104),
    )

    ack0 = supervisor.accept_hello(WorkerHello(pid=101, generation_nonce=_sha("nonce-0")))
    ack1 = supervisor.accept_hello(WorkerHello(pid=102, generation_nonce=_sha("nonce-1")))
    supervisor.accept_hello(WorkerHello(pid=103, generation_nonce=_sha("nonce-2")))
    supervisor.accept_hello(WorkerHello(pid=104, generation_nonce=_sha("nonce-3")))
    supervisor.assert_handshake_complete()

    assert ack0.slot == 0
    assert ack1.slot == 1

    start1 = supervisor.start_row(
        RowStartedEvent(
            row=identities[1],
            slot=1,
            generation_nonce=ack1.generation_nonce,
            pid=ack1.pid,
            worker_ready_ns=10,
        ),
        parent_start_ns=100,
    )
    row1 = row_callback(identities[1])
    supervisor.finish_row(
        RowFinishedEvent(
            row=identities[1],
            slot=1,
            generation_nonce=start1.generation_nonce,
            pid=start1.pid,
            token=start1.token,
            worker_elapsed_ns=20,
            envelope_sha256=canonical_sha256(row1),
        ),
        received_ns=130,
    )
    supervisor.receive_envelope(row1, received_ns=131)
    assert supervisor.completed_rows == ()

    start0 = supervisor.start_row(
        RowStartedEvent(
            row=identities[0],
            slot=0,
            generation_nonce=ack0.generation_nonce,
            pid=ack0.pid,
            worker_ready_ns=11,
        ),
        parent_start_ns=140,
    )
    row0 = row_callback(identities[0])
    supervisor.receive_envelope(row0, received_ns=141)
    supervisor.finish_row(
        RowFinishedEvent(
            row=identities[0],
            slot=0,
            generation_nonce=start0.generation_nonce,
            pid=start0.pid,
            token=start0.token,
            worker_elapsed_ns=30,
            envelope_sha256=canonical_sha256(row0),
        ),
        received_ns=142,
    )

    assert [row.row_index for row in supervisor.completed_rows] == [0, 1]
    for index in range(2, 16):
        _supervisor_complete_row(
            supervisor,
            identities[index],
            slot=index % 4,
            row_callback=row_callback,
        )
    stage_publication = stage_summary_factory(
        supervisor.config,
        supervisor.completed_rows,
        (0, 0, 0, 0),
        supervisor.protocol,
        RunnerStageTelemetry((0, 0, 0, 0), 0.0),
    )
    stage_summary = stage_publication.payload
    assert stage_summary["denominators"]["stage_audio"] == protocol.make_audio_set_binding(
        [identity.audio_group_key for identity in identities]
    )
    assert stage_summary["denominators"]["stage_audio"] != protocol.make_audio_set_binding(
        [identity.cache_audio_key for identity in identities]
    )
    assert supervisor.complete_success(
        stage_publication=stage_publication,
        telemetry=RunnerStageTelemetry((0, 0, 0, 0), 0.0),
    ).ok


def test_supervisor_rejects_unknown_hello_duplicate_envelope_and_sha_mismatch() -> None:
    identities = _identities(16)
    supervisor = _ready_supervisor(identities)
    with pytest.raises(Exp007RunnerError, match="unknown worker"):
        supervisor.accept_hello(WorkerHello(pid=999, generation_nonce=_sha("unknown")))

    supervisor = _ready_supervisor(identities)
    ack = supervisor.start_row(
        RowStartedEvent(
            row=identities[0],
            slot=0,
            generation_nonce=_sha("nonce-0"),
            pid=101,
            worker_ready_ns=1,
        ),
        parent_start_ns=10,
    )
    row = _row(identities[0])
    supervisor.receive_envelope(row, received_ns=20)
    with pytest.raises(Exp007RunnerError, match="duplicate envelope"):
        supervisor.receive_envelope(row, received_ns=21)

    supervisor = _ready_supervisor(identities)
    ack = supervisor.start_row(
        RowStartedEvent(
            row=identities[0],
            slot=0,
            generation_nonce=_sha("nonce-0"),
            pid=101,
            worker_ready_ns=1,
        ),
        parent_start_ns=10,
    )
    supervisor.finish_row(
        RowFinishedEvent(
            row=identities[0],
            slot=0,
            generation_nonce=ack.generation_nonce,
            pid=ack.pid,
            token=ack.token,
            worker_elapsed_ns=1,
            envelope_sha256=_sha("wrong"),
        ),
        received_ns=11,
    )
    with pytest.raises(Exp007RunnerError, match="SHA mismatch"):
        supervisor.receive_envelope(row, received_ns=12)


def test_deadline_equality_and_join_guard_are_hard_failures() -> None:
    identities = _identities(16)
    config = Exp007RunnerConfig(
        use_spawn_pool=False,
        per_audio_arm_timeout_seconds=1.0,
        finish_result_delivery_seconds=0.1,
    )
    supervisor = _ready_supervisor(identities, config=config)
    ack = supervisor.start_row(
        RowStartedEvent(
            row=identities[0],
            slot=0,
            generation_nonce=_sha("nonce-0"),
            pid=101,
            worker_ready_ns=1,
        ),
        parent_start_ns=1_000,
    )

    with pytest.raises(Exp007RunnerError, match="deadline"):
        supervisor.check_deadlines(now_ns=ack.deadline_ns)

    supervisor = _ready_supervisor(identities, config=config)
    ack = supervisor.start_row(
        RowStartedEvent(
            row=identities[0],
            slot=0,
            generation_nonce=_sha("nonce-0"),
            pid=101,
            worker_ready_ns=1,
        ),
        parent_start_ns=1_000,
    )
    row = _row(identities[0])
    supervisor.receive_envelope(row, received_ns=2_000)
    with pytest.raises(Exp007RunnerError, match="join guard"):
        supervisor.check_deadlines(now_ns=2_000 + 100_000_000)


def test_pending_suffix_row_timeout_preserves_concurrent_causality() -> None:
    identities = _identities(16)
    config = Exp007RunnerConfig(
        use_spawn_pool=False,
        per_audio_arm_timeout_seconds=1.0,
    )
    supervisor = _ready_supervisor(identities, config=config)
    starts = []
    for row_index, parent_start_ns in ((0, 1_000), (1, 1_000), (2, 1_000), (3, 10)):
        starts.append(
            supervisor.start_row(
                RowStartedEvent(
                    row=identities[row_index],
                    slot=row_index,
                    generation_nonce=_sha(f"nonce-{row_index}"),
                    pid=101 + row_index,
                    worker_ready_ns=parent_start_ns,
                ),
                parent_start_ns=parent_start_ns,
            )
        )

    with pytest.raises(Exp007RunnerError, match="row deadline"):
        supervisor.check_deadlines(
            now_ns=starts[3].deadline_ns,
            worker_slot_lifetime_bytes=[11, 22, 33, 44],
        )

    assert supervisor.failure_record is not None
    assert supervisor.failure_record["failure_kind"] == "row_timeout"
    assert supervisor.failure_record["causing_row_index"] == 3
    assert supervisor.failure_record["causing_worker_slot"] == 3
    assert supervisor.failure_record["completed_prefix_count"] == 0
    assert [row["row_index"] for row in supervisor.failure_record["pending_identities"]] == list(range(16))


def test_duplicate_finish_for_committed_row_records_hard_failure() -> None:
    identities = _identities(16)
    supervisor = _ready_supervisor(identities)
    ack = supervisor.start_row(
        RowStartedEvent(
            row=identities[0],
            slot=0,
            generation_nonce=_sha("nonce-0"),
            pid=101,
            worker_ready_ns=1,
        ),
        parent_start_ns=10,
    )
    row = _row(identities[0])
    finish = RowFinishedEvent(
        row=identities[0],
        slot=0,
        generation_nonce=ack.generation_nonce,
        pid=ack.pid,
        token=ack.token,
        worker_elapsed_ns=10,
        envelope_sha256=canonical_sha256(row),
    )
    supervisor.finish_row(finish, received_ns=20)
    supervisor.receive_envelope(row, received_ns=21)

    with pytest.raises(Exp007RunnerError, match="invalid row finish"):
        supervisor.finish_row(finish, received_ns=22)

    assert supervisor.failure_record is not None
    assert supervisor.failure_record["failure_kind"] == "broken_stream"
    assert supervisor.failure_record["causing_row_index"] == 0
    assert supervisor.failure_record["completed_prefix_count"] == 1


def test_inline_runner_success_and_failure_prefix_records(tmp_path: Path) -> None:
    identities = _identities(16)
    config, row_callback, stage_summary_factory = _schedule_artifact_config_and_callback(
        tmp_path,
        identities,
        use_spawn_pool=False,
    )

    success = run_synthetic_exp007_arm(
        identities,
        row_callback,
        config=config,
        stage_summary_factory=stage_summary_factory,
    )
    assert success.ok
    assert [row.row_index for row in success.rows] == list(range(16))
    assert success.outcome["status"] == "success"

    failed = run_synthetic_exp007_arm(
        identities,
        _failing_row_callback,
        config=Exp007RunnerConfig(use_spawn_pool=False),
    )
    assert not failed.ok
    assert failed.failure_record is not None
    assert failed.failure_record["failure_kind"] == "row_hard_failure"
    assert failed.failure_record["failure_stage"] == "local_frontier"
    assert failed.failure_record["completed_prefix_count"] == 1
    assert failed.failure_record["pending_identity_count"] == 15
    assert [row["row_index"] for row in failed.failure_record["pending_identities"]] == list(range(1, 16))


def test_schedule_artifact_fixture_binds_nondefault_config_fingerprints(
    tmp_path: Path,
) -> None:
    identities = _identities(16)
    config, row_callback, stage_summary_factory = _schedule_artifact_config_and_callback(
        tmp_path,
        identities,
        use_spawn_pool=False,
        input_manifest_sha256=_sha("custom-selector"),
        source_closure_fingerprint_sha256=_sha("custom-source"),
        run_config_fingerprint_sha256=_sha("custom-run-config"),
    )

    outcome = run_synthetic_exp007_arm(
        identities,
        row_callback,
        config=config,
        stage_summary_factory=stage_summary_factory,
    )

    assert outcome.ok
    for result in outcome.rows:
        row = protocol.validate_row_result(result.payload["row_result"])
        assert row["selector_manifest_sha256"] == config.input_manifest_sha256
        assert row["input_manifest_sha256"] == config.input_manifest_sha256
        assert row["source_closure_fingerprint_sha256"] == config.source_closure_fingerprint_sha256
        assert row["run_config_fingerprint_sha256"] == config.run_config_fingerprint_sha256


@pytest.mark.parametrize(
    ("failure_kind", "failure_stage"),
    [
        ("source_mismatch", "row_source_check"),
        ("cache_mismatch", "cache_load"),
        ("restricted_input_mismatch", "restricted_prediction"),
        ("candidate_mismatch", "candidate"),
        ("current_v2_mismatch", "current_v2"),
    ],
)
def test_inline_typed_worker_failure_preserves_allowed_kind_and_stage(
    failure_kind: str,
    failure_stage: str,
) -> None:
    def callback(_identity: Exp007SyntheticIdentity) -> Exp007SyntheticWorkerResult:
        raise Exp007WorkerFailure(
            "typed worker failure",
            failure_kind=failure_kind,
            failure_stage=failure_stage,
        )

    outcome = run_synthetic_exp007_arm(
        _identities(16),
        callback,
        config=Exp007RunnerConfig(use_spawn_pool=False),
    )

    assert not outcome.ok
    assert outcome.failure_record is not None
    assert outcome.failure_record["failure_kind"] == failure_kind
    assert outcome.failure_record["failure_stage"] == failure_stage


def test_local_frontier_classifier_maps_diagnostics_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("pulsefield_model.timing.v3.local_frontier")

    class Classification:
        reason = "diagnostics_integrity_failure"
        stage = "diagnostics"

    monkeypatch.setattr(
        module,
        "classify_local_frontier_exception",
        lambda _exc: Classification(),
    )

    def callback(_identity: Exp007SyntheticIdentity) -> Exp007SyntheticWorkerResult:
        raise RuntimeError("diagnostics failure")

    outcome = run_synthetic_exp007_arm(
        _identities(16),
        callback,
        config=Exp007RunnerConfig(use_spawn_pool=False),
    )

    assert not outcome.ok
    assert outcome.failure_record is not None
    assert outcome.failure_record["failure_kind"] == "diagnostics_integrity_failure"
    assert outcome.failure_record["failure_stage"] == "diagnostics"


def test_spawn_runner_executes_synthetic_rows_when_enabled(tmp_path: Path) -> None:
    identities = _identities(16)
    config, row_callback, stage_summary_factory = _schedule_artifact_config_and_callback(
        tmp_path,
        identities,
    )

    outcome = run_synthetic_exp007_arm(
        identities,
        row_callback,
        config=config,
        stage_summary_factory=stage_summary_factory,
    )

    assert outcome.ok
    assert [row.row_index for row in outcome.rows] == list(range(16))
    assert len(outcome.worker_slot_lifetime_bytes) == 4
    assert all(value is not None for value in outcome.worker_slot_lifetime_bytes)


def test_spawn_worker_initializer_config_has_no_parent_authority_fields(
    tmp_path: Path,
) -> None:
    identities = _identities(16)
    manifest = _synthetic_candidate_reference_manifest(identities)
    manifest["padding"] = [f"{index}:{'x' * 1024}" for index in range(1024)]
    config = Exp007RunnerConfig()
    publication = StageSummaryPublication(
        {"candidate_reference_manifest_sha256": protocol.object_complete_sha256(manifest)},
        candidate_reference_manifest=manifest,
        candidate_reference_artifact_root=tmp_path,
    )
    scrubbed = exp007_runner_module._worker_initializer_config(config)  # noqa: SLF001

    assert not hasattr(config, "candidate_reference_manifest_sha256")
    assert not hasattr(config, "candidate_reference_manifest")
    assert not hasattr(config, "candidate_reference_artifact_root")
    assert scrubbed is config
    assert scrubbed.run_config_fingerprint_sha256 == config.run_config_fingerprint_sha256
    assert len(pickle.dumps(scrubbed)) == len(pickle.dumps(config))
    assert len(pickle.dumps(publication)) > len(pickle.dumps(config)) * 10


def test_reference_manifest_complete_sha_uses_artifacts_validated_object(
    tmp_path: Path,
) -> None:
    identities = _identities(16)
    manifest, row_payload_sha_by_index = _publish_artifacts_reference_manifest(
        tmp_path,
        identities,
    )
    manifest_sha = protocol.object_complete_sha256(manifest)
    entry_list_sha = protocol.canonical_json_sha256(
        [entry["entry_payload_sha256"] for entry in manifest["entries"]]
    )
    assert entry_list_sha != manifest_sha

    def callback(identity: Exp007SyntheticIdentity) -> Exp007SyntheticWorkerResult:
        return _row(
            identity,
            row_payload_sha256=row_payload_sha_by_index[identity.row_index],
            candidate_reference_entry_payload_sha256=manifest["entries"][
                identity.row_index
            ]["entry_payload_sha256"],
        )

    s30 = run_synthetic_exp007_arm(
        identities,
        callback,
        config=_schedule_config(identities, use_spawn_pool=False),
        stage_summary_factory=_schedule_stage_summary_factory(tmp_path),
    )

    assert s30.ok
    assert s30.outcome["candidate_reference_manifest_sha256"] == manifest_sha

    s60 = run_synthetic_exp007_arm(
        identities,
        lambda identity: _row(identity, arm="S60"),
        config=Exp007RunnerConfig(
            schedule_arm="S60",
            use_spawn_pool=False,
        ),
        stage_summary_factory=_schedule_stage_summary_factory(tmp_path),
    )

    assert s60.ok
    assert s60.outcome["candidate_reference_manifest_sha256"] == manifest_sha


def test_reference_arm_entry_list_hash_publication_is_schema_failure(
    tmp_path: Path,
) -> None:
    identities = _identities(16)
    config, row_callback, good_factory = _schedule_artifact_config_and_callback(
        tmp_path,
        identities,
        use_spawn_pool=False,
    )
    def entry_list_hash_factory(*args: object) -> StageSummaryPublication:
        publication = good_factory(*args)  # type: ignore[arg-type]
        assert publication.candidate_reference_manifest is not None
        entry_list_sha = protocol.canonical_json_sha256(
            [
                entry["entry_payload_sha256"]
                for entry in publication.candidate_reference_manifest["entries"]
            ]
        )
        return StageSummaryPublication(
            _replace_summary_manifest_sha(publication.payload, entry_list_sha),
            candidate_reference_manifest=publication.candidate_reference_manifest,
            candidate_reference_artifact_root=publication.candidate_reference_artifact_root,
            candidate_reference_manifest_relative_path=(
                publication.candidate_reference_manifest_relative_path
            ),
        )

    outcome = run_synthetic_exp007_arm(
        identities,
        row_callback,
        config=config,
        stage_summary_factory=entry_list_hash_factory,
    )

    assert not outcome.ok
    assert outcome.failure_record is not None
    assert outcome.failure_record["failure_kind"] == "schema_failure"
    assert outcome.failure_record["failure_stage"] == "arm_summary"
    assert outcome.failure_record["completed_prefix_count"] == 16


def test_success_rejects_self_consistent_but_unvalidated_reference_manifest(
    tmp_path: Path,
) -> None:
    identities = _identities(16)
    manifest = _synthetic_candidate_reference_manifest(identities)
    manifest_sha = protocol.object_complete_sha256(manifest)
    config, row_callback, good_factory = _schedule_artifact_config_and_callback(
        tmp_path,
        identities,
        use_spawn_pool=False,
    )

    def forged_factory(*args: object) -> StageSummaryPublication:
        publication = good_factory(*args)  # type: ignore[arg-type]
        return StageSummaryPublication(
            _replace_summary_manifest_sha(publication.payload, manifest_sha),
            candidate_reference_manifest=manifest,
            candidate_reference_artifact_root=tmp_path,
            candidate_reference_manifest_relative_path=(
                publication.candidate_reference_manifest_relative_path
            ),
        )

    outcome = run_synthetic_exp007_arm(
        identities,
        row_callback,
        config=config,
        stage_summary_factory=forged_factory,
    )

    assert not outcome.ok
    assert outcome.failure_record is not None
    assert outcome.failure_record["failure_kind"] == "schema_failure"
    assert outcome.failure_record["failure_stage"] == "arm_summary"
    assert outcome.failure_record["completed_prefix_count"] == 16


def test_success_reopens_manifest_root_and_rejects_missing_or_tampered_bundle(
    tmp_path: Path,
) -> None:
    for case in ("wrong_root", "missing_bundle", "tampered_bundle", "tampered_manifest"):
        root = tmp_path / case / "root"
        root.mkdir(parents=True)
        identities = _identities(16)
        config, row_callback, good_factory = _schedule_artifact_config_and_callback(
            root,
            identities,
            use_spawn_pool=False,
        )

        def stage_factory(*args: object) -> StageSummaryPublication:
            publication = good_factory(*args)  # type: ignore[arg-type]
            assert publication.candidate_reference_manifest is not None
            if case == "wrong_root":
                wrong_root = tmp_path / case / "wrong"
                wrong_root.mkdir()
                return StageSummaryPublication(
                    publication.payload,
                    candidate_reference_manifest=publication.candidate_reference_manifest,
                    candidate_reference_artifact_root=wrong_root,
                    candidate_reference_manifest_relative_path=(
                        publication.candidate_reference_manifest_relative_path
                    ),
                )
            bundle_path = root / publication.candidate_reference_manifest["entries"][0][
                "bundle_relative_path"
            ]
            if case == "missing_bundle":
                bundle_path.unlink()
            elif case == "tampered_bundle":
                bundle_path.write_text('{"schema":"tampered"}', encoding="utf-8")
            else:
                assert publication.candidate_reference_manifest_relative_path is not None
                manifest_path = root / publication.candidate_reference_manifest_relative_path
                manifest_path.write_text('{"schema":"tampered"}', encoding="utf-8")
            return publication

        outcome = run_synthetic_exp007_arm(
            identities,
            row_callback,
            config=config,
            stage_summary_factory=stage_factory,
        )

        assert not outcome.ok, case
        assert outcome.failure_record is not None
        assert outcome.failure_record["failure_kind"] == "schema_failure"
        assert outcome.failure_record["failure_stage"] == "arm_summary"
        assert outcome.failure_record["completed_prefix_count"] == 16


@pytest.mark.parametrize("use_spawn_pool", [False, True])
def test_bad_stage_summary_maps_to_schema_failure_after_all_rows(
    use_spawn_pool: bool,
) -> None:
    identities = _identities(16)

    outcome = run_synthetic_exp007_arm(
        identities,
        _row,
        config=_schedule_config(identities, use_spawn_pool=use_spawn_pool),
        stage_summary_factory=_bad_stage_summary_factory,
    )

    assert not outcome.ok
    assert outcome.failure_record is not None
    assert outcome.failure_record["failure_kind"] == "schema_failure"
    assert outcome.failure_record["failure_stage"] == "arm_summary"
    assert outcome.failure_record["completed_prefix_count"] == 16


@pytest.mark.parametrize("use_spawn_pool", [False, True])
def test_summary_publication_failure_preserves_contract_kind_after_all_rows(
    use_spawn_pool: bool,
) -> None:
    identities = _identities(16)

    outcome = run_synthetic_exp007_arm(
        identities,
        _row,
        config=_schedule_config(identities, use_spawn_pool=use_spawn_pool),
        stage_summary_factory=_summary_publication_failure_factory,
    )

    assert not outcome.ok
    assert outcome.failure_record is not None
    assert outcome.failure_record["failure_kind"] == "summary_publication_failure"
    assert outcome.failure_record["failure_stage"] == "arm_summary"
    assert outcome.failure_record["completed_prefix_count"] == 16


def test_repair_bad_stage_summary_maps_to_repair_summary_schema_failure() -> None:
    identities = _identities(80)

    outcome = run_synthetic_exp007_arm(
        identities,
        _row,
        config=Exp007RunnerConfig(
            stage=EXP007_REPAIR_STAGE,
            use_spawn_pool=False,
        ),
        stage_summary_factory=_bad_stage_summary_factory,
    )

    assert not outcome.ok
    assert outcome.failure_record is not None
    assert outcome.failure_record["failure_kind"] == "schema_failure"
    assert outcome.failure_record["failure_stage"] == "repair_summary"
    assert outcome.failure_record["completed_prefix_count"] == 80


def test_repair_success_uses_authoritative_summary_context(tmp_path: Path) -> None:
    identities, config, row_callback, publication_factory = _repair_artifact_config_and_callback(
        tmp_path
    )

    outcome = run_synthetic_exp007_arm(
        identities,
        row_callback,
        config=config,
        stage_summary_factory=publication_factory,
    )

    assert outcome.ok
    assert outcome.outcome["status"] == "success"
    assert outcome.outcome["candidate_reference_manifest_sha256"] != "4" * 64


def test_repair_success_missing_authoritative_context_hard_fails(
    tmp_path: Path,
) -> None:
    identities, config, row_callback, publication_factory = _repair_artifact_config_and_callback(
        tmp_path
    )

    def payload_only_factory(*args: object) -> StageSummaryPublication:
        publication = publication_factory(*args)  # type: ignore[arg-type]
        return StageSummaryPublication(
            dict(publication.payload),
            candidate_reference_manifest=publication.candidate_reference_manifest,
            candidate_reference_artifact_root=publication.candidate_reference_artifact_root,
            candidate_reference_manifest_relative_path=(
                publication.candidate_reference_manifest_relative_path
            ),
        )

    outcome = run_synthetic_exp007_arm(
        identities,
        row_callback,
        config=config,
        stage_summary_factory=payload_only_factory,
    )

    assert not outcome.ok
    assert outcome.failure_record is not None
    assert outcome.failure_record["failure_kind"] == "schema_failure"
    assert outcome.failure_record["failure_stage"] == "repair_summary"
    assert outcome.failure_record["completed_prefix_count"] == 80


def test_repair_context_prediction_rows_must_match_committed_payload(
    tmp_path: Path,
) -> None:
    identities, config, row_callback, publication_factory = _repair_artifact_config_and_callback(
        tmp_path
    )

    def swapped_prediction_rows_factory(*args: object) -> StageSummaryPublication:
        publication = publication_factory(*args)  # type: ignore[arg-type]
        context = publication.repair_summary_validation_context
        assert context is not None
        prediction_rows = [dict(row) for row in context.prediction_rows]
        prediction_rows[0] = dict(prediction_rows[0], row_payload_sha256=_sha("wrong-row"))
        bad_context = RepairSummaryValidationContext(
            repair_metric_rows=context.repair_metric_rows,
            repair80_input_binding=context.repair80_input_binding,
            repair80_identity_source_artifact=context.repair80_identity_source_artifact,
            repair80_label_source_artifact=context.repair80_label_source_artifact,
            repair80_identity_rows=context.repair80_identity_rows,
            repair80_label_rows=context.repair80_label_rows,
            schedule_weak_veto_outcome=context.schedule_weak_veto_outcome,
            candidate_reference_manifest=context.candidate_reference_manifest,
            schedule_candidate_reference_manifest=(
                context.schedule_candidate_reference_manifest
            ),
            artifact_root=context.artifact_root,
            four_arm_stage_summary=context.four_arm_stage_summary,
            config_selection=context.config_selection,
            candidate_global_manifest=context.candidate_global_manifest,
            run_configs_by_arm=context.run_configs_by_arm,
            arm_rows_by_arm=context.arm_rows_by_arm,
            candidate_payloads_by_arm=context.candidate_payloads_by_arm,
            arm_stage_outcomes_by_execution_order=(
                context.arm_stage_outcomes_by_execution_order
            ),
            source_arm_stage_summaries_by_execution_order=(
                context.source_arm_stage_summaries_by_execution_order
            ),
            repair_run_config=context.repair_run_config,
            source_closure=context.source_closure,
            source_repo_root=context.source_repo_root,
            prediction_rows=prediction_rows,
            weak_rows=context.weak_rows,
        )
        return StageSummaryPublication(
            publication.payload,
            bad_context,
            candidate_reference_manifest=publication.candidate_reference_manifest,
            candidate_reference_artifact_root=publication.candidate_reference_artifact_root,
            candidate_reference_manifest_relative_path=(
                publication.candidate_reference_manifest_relative_path
            ),
        )

    outcome = run_synthetic_exp007_arm(
        identities,
        row_callback,
        config=config,
        stage_summary_factory=swapped_prediction_rows_factory,
    )

    assert not outcome.ok
    assert outcome.failure_record is not None
    assert outcome.failure_record["failure_kind"] == "schema_failure"
    assert outcome.failure_record["failure_stage"] == "repair_summary"
    assert outcome.failure_record["completed_prefix_count"] == 80


def test_repair_terminal_preflight_rejects_tampered_upstream_authority(
    tmp_path: Path,
) -> None:
    cases = {
        "run_config": lambda context: replace(
            context,
            repair_run_config=dict(context.repair_run_config, schedule_arm="S30"),
        ),
        "source_closure": lambda context: replace(
            context,
            source_closure=dict(
                context.source_closure,
                source_closure_fingerprint_sha256=_sha("wrong-source"),
            ),
        ),
        "identity_source_artifact": lambda context: replace(
            context,
            repair80_identity_source_artifact=b'{"schema":"tampered","rows":[]}',
        ),
        "four_arm_summary": lambda context: replace(
            context,
            four_arm_stage_summary=dict(context.four_arm_stage_summary, status="hard_failure"),
        ),
        "config_selection": lambda context: replace(
            context,
            config_selection=dict(context.config_selection, source_decision="negative"),
        ),
        "candidate_global_manifest": lambda context: replace(
            context,
            candidate_global_manifest=dict(
                context.candidate_global_manifest,
                row_count=15,
            ),
        ),
        "schedule_reference_manifest": lambda context: replace(
            context,
            schedule_candidate_reference_manifest=context.candidate_reference_manifest,
        ),
        "arm_rows": lambda context: replace(
            context,
            arm_rows_by_arm={
                **context.arm_rows_by_arm,
                "S64": tuple(context.arm_rows_by_arm["S64"][:-1]),
            },
        ),
        "candidate_payloads": lambda context: replace(
            context,
            candidate_payloads_by_arm={
                **context.candidate_payloads_by_arm,
                "S30": tuple(context.candidate_payloads_by_arm["S30"][:-1]),
            },
        ),
        "arm_outcomes": lambda context: replace(
            context,
            arm_stage_outcomes_by_execution_order={
                **context.arm_stage_outcomes_by_execution_order,
                "S64": context.arm_stage_outcomes_by_execution_order["S30"],
            },
        ),
        "source_summaries": lambda context: replace(
            context,
            source_arm_stage_summaries_by_execution_order={
                **context.source_arm_stage_summaries_by_execution_order,
                "S64": context.source_arm_stage_summaries_by_execution_order["S30"],
            },
        ),
        "weak_outcome": lambda context: replace(
            context,
            schedule_weak_veto_outcome=dict(
                context.schedule_weak_veto_outcome,
                status="hard_failure",
            ),
        ),
    }
    for case, mutate in cases.items():
        root = tmp_path / case
        root.mkdir()
        identities, config, row_callback, publication_factory = (
            _repair_artifact_config_and_callback(root)
        )

        def tampered_factory(*args: object) -> StageSummaryPublication:
            publication = publication_factory(*args)  # type: ignore[arg-type]
            context = publication.repair_summary_validation_context
            assert context is not None
            return _repair_publication_with_context(
                publication,
                mutate(context),
            )

        outcome = run_synthetic_exp007_arm(
            identities,
            row_callback,
            config=config,
            stage_summary_factory=tampered_factory,
        )

        assert not outcome.ok, case
        assert outcome.failure_record is not None
        assert outcome.failure_record["failure_kind"] == "schema_failure"
        assert outcome.failure_record["failure_stage"] == "repair_summary"
        assert outcome.failure_record["completed_prefix_count"] == 80


def test_repair_success_reopens_published_manifest_and_rejects_divergence(
    tmp_path: Path,
) -> None:
    for case in ("missing_manifest", "divergent_mapping", "tampered_file"):
        root = tmp_path / case
        root.mkdir()
        identities, config, row_callback, publication_factory = (
            _repair_artifact_config_and_callback(root)
        )

        def manifest_factory(*args: object) -> StageSummaryPublication:
            publication = publication_factory(*args)  # type: ignore[arg-type]
            assert publication.candidate_reference_manifest_relative_path is not None
            manifest_path = root / publication.candidate_reference_manifest_relative_path
            if case == "missing_manifest":
                manifest_path.unlink()
                return publication
            if case == "tampered_file":
                manifest_path.write_text('{"schema":"tampered"}', encoding="utf-8")
                return publication
            context = publication.repair_summary_validation_context
            assert context is not None
            return StageSummaryPublication(
                publication.payload,
                context,
                candidate_reference_manifest=context.schedule_candidate_reference_manifest,
                candidate_reference_artifact_root=publication.candidate_reference_artifact_root,
                candidate_reference_manifest_relative_path=(
                    publication.candidate_reference_manifest_relative_path
                ),
            )

        outcome = run_synthetic_exp007_arm(
            identities,
            row_callback,
            config=config,
            stage_summary_factory=manifest_factory,
        )

        assert not outcome.ok, case
        assert outcome.failure_record is not None
        assert outcome.failure_record["failure_kind"] == "schema_failure"
        assert outcome.failure_record["failure_stage"] == "repair_summary"
        assert outcome.failure_record["completed_prefix_count"] == 80


def test_repair_shape_valid_summary_tamper_hard_fails_authoritative_context(
    tmp_path: Path,
) -> None:
    identities, config, row_callback, publication_factory = _repair_artifact_config_and_callback(
        tmp_path
    )

    def tampered_factory(*args: object) -> StageSummaryPublication:
        publication = publication_factory(*args)  # type: ignore[arg-type]
        return StageSummaryPublication(
            _tamper_repair_summary_runtime(publication.payload),
            publication.repair_summary_validation_context,
            candidate_reference_manifest=publication.candidate_reference_manifest,
            candidate_reference_artifact_root=publication.candidate_reference_artifact_root,
            candidate_reference_manifest_relative_path=(
                publication.candidate_reference_manifest_relative_path
            ),
        )

    outcome = run_synthetic_exp007_arm(
        identities,
        row_callback,
        config=config,
        stage_summary_factory=tampered_factory,
    )

    assert not outcome.ok
    assert outcome.failure_record is not None
    assert outcome.failure_record["failure_kind"] == "schema_failure"
    assert outcome.failure_record["failure_stage"] == "repair_summary"
    assert outcome.failure_record["completed_prefix_count"] == 80


def test_repair_protocol_shape_bypass_still_hard_fails_authoritative_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities, config, row_callback, publication_factory = _repair_artifact_config_and_callback(
        tmp_path
    )
    monkeypatch.setattr(
        protocol,
        "validate_repair80_summary",
        lambda payload: dict(payload),
    )

    def tampered_factory(*args: object) -> StageSummaryPublication:
        publication = publication_factory(*args)  # type: ignore[arg-type]
        return StageSummaryPublication(
            _tamper_repair_summary_runtime(publication.payload),
            publication.repair_summary_validation_context,
            candidate_reference_manifest=publication.candidate_reference_manifest,
            candidate_reference_artifact_root=publication.candidate_reference_artifact_root,
            candidate_reference_manifest_relative_path=(
                publication.candidate_reference_manifest_relative_path
            ),
        )

    outcome = run_synthetic_exp007_arm(
        identities,
        row_callback,
        config=config,
        stage_summary_factory=tampered_factory,
    )

    assert not outcome.ok
    assert outcome.failure_record is not None
    assert outcome.failure_record["failure_kind"] == "schema_failure"
    assert outcome.failure_record["failure_stage"] == "repair_summary"
    assert outcome.failure_record["completed_prefix_count"] == 80


def test_repair_summary_uses_runner_owned_worker_rss(
    tmp_path: Path,
) -> None:
    identities, config, row_callback, publication_factory = _repair_artifact_config_and_callback(
        tmp_path,
        summary_worker_rss=(1, 1, 1, 1),
    )

    outcome = run_synthetic_exp007_arm(
        identities,
        row_callback,
        config=config,
        stage_summary_factory=publication_factory,
    )

    assert not outcome.ok
    assert outcome.failure_record is not None
    assert outcome.failure_record["failure_kind"] == "schema_failure"
    assert outcome.failure_record["failure_stage"] == "repair_summary"
    assert outcome.failure_record["completed_prefix_count"] == 80


def test_repair_summary_uses_runner_owned_wall_seconds(
    tmp_path: Path,
) -> None:
    identities, config, row_callback, publication_factory = _repair_artifact_config_and_callback(
        tmp_path,
        summary_wall_seconds=999.0,
    )

    outcome = run_synthetic_exp007_arm(
        identities,
        row_callback,
        config=config,
        stage_summary_factory=publication_factory,
    )

    assert not outcome.ok
    assert outcome.failure_record is not None
    assert outcome.failure_record["failure_kind"] == "schema_failure"
    assert outcome.failure_record["failure_stage"] == "repair_summary"
    assert outcome.failure_record["completed_prefix_count"] == 80


def test_schedule_runs_fixed_order_and_not_run_after_prior_failure(tmp_path: Path) -> None:
    identities = _identities(16)
    config, s30_row_callback, stage_summary_factory = _schedule_artifact_config_and_callback(
        tmp_path,
        identities,
        use_spawn_pool=False,
    )
    seen: list[str] = []

    def callback(arm: str, identity: Exp007SyntheticIdentity) -> Exp007SyntheticWorkerResult:
        if identity.row_index == 0:
            seen.append(arm)
        if arm == "S60":
            raise RuntimeError("synthetic arm failure")
        if arm == "S30":
            return s30_row_callback(identity)
        return _row(identity, arm=arm)

    outcomes = run_synthetic_exp007_schedule(
        identities,
        callback,
        config_template=config,
        stage_summary_factory=stage_summary_factory,
    )

    assert seen == ["S30", "S60"]
    assert tuple(outcomes) == EXP007_EXECUTION_ORDER
    assert outcomes["S30"].ok
    assert not outcomes["S60"].ok
    assert outcomes["S90"].status == "not_run"
    assert outcomes["S90"].outcome["status"] == "not_run_due_prior_hard_failure"
    assert outcomes["S90"].outcome["causing_outcome_sha256"] != "0" * 64
    assert outcomes["S64"].outcome["status"] == "not_run_due_prior_hard_failure"


def test_protocol_adapter_does_not_swallow_builder_type_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identities = _identities(16)
    config, row_callback, stage_summary_factory = _schedule_artifact_config_and_callback(
        tmp_path,
        identities,
        use_spawn_pool=False,
    )
    outcome = run_synthetic_exp007_arm(
        identities,
        row_callback,
        config=config,
        stage_summary_factory=stage_summary_factory,
    )
    assert outcome.ok

    from pulsefield_model.timing.evaluation.exp007_runner import Exp007ProtocolAdapter

    adapter = Exp007ProtocolAdapter()
    with pytest.raises(Exp007RunnerError, match="requires a complete validated stage summary"):
        adapter.arm_success(config=Exp007RunnerConfig(use_spawn_pool=False), rows=outcome.rows)

    publication = stage_summary_factory(
        config,
        outcome.rows,
        outcome.worker_slot_lifetime_bytes,
        adapter,
        RunnerStageTelemetry((0, 0, 0, 0), 0.0),
    )
    monkeypatch.setattr(
        adapter._module,  # noqa: SLF001
        "make_arm_stage_success",
        lambda **_kwargs: (_ for _ in ()).throw(TypeError("strict builder mismatch")),
    )
    with pytest.raises(TypeError, match="strict builder mismatch"):
        adapter.arm_success(
            config=config,
            rows=outcome.rows,
            stage_publication=publication,
            telemetry=RunnerStageTelemetry((0, 0, 0, 0), 0.0),
        )


def test_bounded_hello_handshake_times_out_without_accepting_forever() -> None:
    identities = _identities(16)
    config = Exp007RunnerConfig(
        use_spawn_pool=False,
        per_audio_arm_timeout_seconds=0.01,
        parent_poll_max_seconds=0.001,
    )
    listener = Listener(("127.0.0.1", 0), family="AF_INET")
    supervisor = _ready_supervisor_without_hellos(identities, config=config)
    try:
        with pytest.raises(Exp007RunnerError, match="HELLO handshake deadline"):
            _accept_initial_handshake(
                listener=listener,
                pool=_FakePool((101, 102, 103, 104)),
                supervisor=supervisor,
                config=config,
                worker_slot_lifetime_bytes=[None, None, None, None],
            )
    finally:
        listener.close()


def test_bounded_hello_handshake_detects_pre_hello_worker_death() -> None:
    identities = _identities(16)
    config = Exp007RunnerConfig(
        use_spawn_pool=False,
        per_audio_arm_timeout_seconds=0.05,
        parent_poll_max_seconds=0.001,
    )
    listener = Listener(("127.0.0.1", 0), family="AF_INET")
    supervisor = _ready_supervisor_without_hellos(identities, config=config)
    try:
        with pytest.raises(Exp007RunnerError, match="worker death"):
            _accept_initial_handshake(
                listener=listener,
                pool=_FakePool((101, 102, 103)),
                supervisor=supervisor,
                config=config,
                worker_slot_lifetime_bytes=[None, None, None, None],
            )
    finally:
        listener.close()
    assert supervisor.failure_record["failure_kind"] == "worker_death"
    assert supervisor.failure_record["failure_stage"] == "pool_start"


def test_spawn_worker_sigalrm_maps_to_row_timeout() -> None:
    outcome = run_synthetic_exp007_arm(
        _identities(16),
        _sleeping_row,
        config=Exp007RunnerConfig(
            per_audio_arm_timeout_seconds=1.0,
            parent_poll_max_seconds=0.01,
        ),
    )

    assert not outcome.ok
    assert outcome.failure_record is not None
    assert outcome.failure_record["failure_kind"] == "row_timeout", outcome.failure_record
    assert outcome.failure_record["causing_row_index"] is not None


def test_same_slot_finish_disarms_before_ordered_envelope_commit() -> None:
    identities = _identities(16)
    supervisor = _ready_supervisor(identities)
    start0 = supervisor.start_row(
        RowStartedEvent(
            row=identities[0],
            slot=0,
            generation_nonce=_sha("nonce-0"),
            pid=101,
            worker_ready_ns=1,
        ),
        parent_start_ns=10,
    )
    row0 = _row(identities[0])
    supervisor.finish_row(
        RowFinishedEvent(
            row=identities[0],
            slot=0,
            generation_nonce=start0.generation_nonce,
            pid=start0.pid,
            token=start0.token,
            worker_elapsed_ns=10,
            envelope_sha256=canonical_sha256(row0),
        ),
        received_ns=20,
    )

    start4 = supervisor.start_row(
        RowStartedEvent(
            row=identities[4],
            slot=0,
            generation_nonce=_sha("nonce-0"),
            pid=101,
            worker_ready_ns=21,
        ),
        parent_start_ns=21,
    )

    assert start4.row.row_index == 4
    supervisor.receive_envelope(row0, received_ns=22)
    assert [row.row_index for row in supervisor.completed_rows] == [0]


def test_schedule_deadline_creates_hard_failure_and_not_run_later_arms() -> None:
    seen: list[str] = []

    def callback(arm: str, identity: Exp007SyntheticIdentity) -> Exp007SyntheticWorkerResult:
        if identity.row_index == 0:
            seen.append(arm)
            time.sleep(0.01)
        return _row(identity, arm=arm)

    outcomes = run_synthetic_exp007_schedule(
        _identities(16),
        callback,
        config_template=Exp007RunnerConfig(
            use_spawn_pool=False,
            schedule_four_arm_stop_seconds=0.001,
        ),
    )

    assert seen == ["S30"]
    assert outcomes["S30"].failure_record["failure_kind"] == "schedule_deadline"
    assert outcomes["S60"].status == "not_run"
    assert outcomes["S60"].outcome["reason"] == "schedule_deadline_already_crossed"


def test_schedule_deadline_crossing_mid_arm_uses_common_absolute_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(exp007_runner_module.time, "monotonic_ns", clock.monotonic_ns)
    monkeypatch.setattr(exp007_runner_module.time, "monotonic", clock.monotonic)
    seen: list[str] = []

    def callback(arm: str, identity: Exp007SyntheticIdentity) -> Exp007SyntheticWorkerResult:
        if identity.row_index == 0:
            seen.append(arm)
            clock.advance_ns(2_000_000)
        return _row(identity, arm=arm)

    outcomes = run_synthetic_exp007_schedule(
        _identities(16),
        callback,
        config_template=Exp007RunnerConfig(
            use_spawn_pool=False,
            per_audio_arm_timeout_seconds=60.0,
            schedule_four_arm_stop_seconds=0.001,
        ),
    )

    assert seen == ["S30"]
    assert outcomes["S30"].failure_record["failure_kind"] == "schedule_deadline"
    assert outcomes["S30"].failure_record["failure_stage"] == "schedule_deadline"
    assert outcomes["S30"].failure_record["completed_prefix_count"] == 0
    assert outcomes["S60"].status == "not_run"
    assert outcomes["S60"].outcome["reason"] == "schedule_deadline_already_crossed"


def test_schedule_deadline_after_four_successes_publishes_hard_final_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clock = _PostArmDeadlineClock()
    monkeypatch.setattr(exp007_runner_module.time, "monotonic_ns", clock.monotonic_ns)
    monkeypatch.setattr(exp007_runner_module.time, "monotonic", clock.monotonic)
    identities = _identities(16)
    config, s30_row_callback, base_factory = _schedule_artifact_config_and_callback(
        tmp_path,
        identities,
        use_spawn_pool=False,
        schedule_four_arm_stop_seconds=1.0,
    )

    def callback(arm: str, identity: Exp007SyntheticIdentity) -> Exp007SyntheticWorkerResult:
        if arm == "S30":
            return s30_row_callback(identity)
        return _row(identity, arm=arm)

    def stage_factory(*args: object) -> StageSummaryPublication:
        publication = base_factory(*args)  # type: ignore[arg-type]
        stage_config = args[0]
        assert isinstance(stage_config, Exp007RunnerConfig)
        if stage_config.schedule_arm == "S64":
            clock.cross_after_next_deadline_read()
        return publication

    outcomes = run_synthetic_exp007_schedule(
        identities,
        callback,
        config_template=config,
        stage_summary_factory=stage_factory,
    )

    assert tuple(outcomes) == EXP007_EXECUTION_ORDER
    assert all(outcomes[arm].ok for arm in EXP007_EXECUTION_ORDER)
    summary = outcomes.four_arm_stage_summary
    assert summary is not None
    assert summary["status"] == "hard_failure"
    assert summary["candidate_global_manifest_sha256"] is None
    assert summary["config_selection_sha256"] is None
    assert summary["source_selection_status"] == "not_run"
    assert summary["arm_outcome_sha256_by_execution_order"] == {
        arm: protocol.object_complete_sha256(outcomes[arm].outcome)
        for arm in EXP007_EXECUTION_ORDER
    }
    details = summary["failure_details"]
    assert details["failure_kind"] == "schedule_deadline"
    assert details["completed_success_arm_count"] == 4
    assert details["first_failure_arm"] is None
    assert details["causing_outcome_sha256"] is None


def test_repair_arm_deadline_crossing_mid_row_uses_repair_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(exp007_runner_module.time, "monotonic_ns", clock.monotonic_ns)
    monkeypatch.setattr(exp007_runner_module.time, "monotonic", clock.monotonic)

    def callback(identity: Exp007SyntheticIdentity) -> Exp007SyntheticWorkerResult:
        if identity.row_index == 0:
            clock.advance_ns(2_000_000)
        return _row(identity)

    outcome = run_synthetic_exp007_arm(
        _identities(80),
        callback,
        config=Exp007RunnerConfig(
            stage=EXP007_REPAIR_STAGE,
            use_spawn_pool=False,
            per_audio_arm_timeout_seconds=60.0,
            repair_stop_seconds=0.001,
        ),
    )

    assert not outcome.ok
    assert outcome.failure_record is not None
    assert outcome.failure_record["failure_kind"] == "arm_deadline"
    assert outcome.failure_record["failure_stage"] == "pool_stream"
    assert outcome.failure_record["completed_prefix_count"] == 0


def test_repair_deadline_after_final_row_maps_to_repair_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _PostArmDeadlineClock()
    monkeypatch.setattr(exp007_runner_module.time, "monotonic_ns", clock.monotonic_ns)
    monkeypatch.setattr(exp007_runner_module.time, "monotonic", clock.monotonic)

    def callback(identity: Exp007SyntheticIdentity) -> Exp007SyntheticWorkerResult:
        row = _row(identity)
        if identity.row_index == 79:
            clock.cross_after_next_deadline_read()
        return row

    outcome = run_synthetic_exp007_arm(
        _identities(80),
        callback,
        config=Exp007RunnerConfig(
            stage=EXP007_REPAIR_STAGE,
            use_spawn_pool=False,
            repair_stop_seconds=1.0,
        ),
    )

    assert not outcome.ok
    assert outcome.failure_record is not None
    assert outcome.failure_record["failure_kind"] == "arm_deadline"
    assert outcome.failure_record["failure_stage"] == "repair_summary"
    assert outcome.failure_record["completed_prefix_count"] == 80
    assert outcome.failure_record["pending_identity_count"] == 0


def test_repair_summary_factory_timeout_maps_to_repair_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _FakeClock()
    monkeypatch.setattr(exp007_runner_module.time, "monotonic_ns", clock.monotonic_ns)
    monkeypatch.setattr(exp007_runner_module.time, "monotonic", clock.monotonic)
    identities = _identities(80)

    def slow_summary_factory(*_args: object) -> dict[str, object]:
        time.sleep(0.05)
        return {"schema": "too-late"}

    outcome = run_synthetic_exp007_arm(
        identities,
        lambda identity: _row(identity),
        config=Exp007RunnerConfig(
            stage=EXP007_REPAIR_STAGE,
            use_spawn_pool=False,
            repair_stop_seconds=0.01,
        ),
        stage_summary_factory=slow_summary_factory,
    )

    assert not outcome.ok
    assert outcome.failure_record is not None
    assert outcome.failure_record["failure_kind"] == "arm_deadline"
    assert outcome.failure_record["failure_stage"] == "repair_summary"
    assert outcome.failure_record["completed_prefix_count"] == 80


def test_schedule_summary_factory_timeout_stops_later_arms(
    tmp_path: Path,
) -> None:
    identities = _identities(16)
    config, row_callback, _stage_summary_factory = _schedule_artifact_config_and_callback(
        tmp_path,
        identities,
        use_spawn_pool=False,
        schedule_four_arm_stop_seconds=0.2,
    )
    seen: list[str] = []

    def callback(arm: str, identity: Exp007SyntheticIdentity) -> Exp007SyntheticWorkerResult:
        if identity.row_index == 0:
            seen.append(arm)
        if arm == "S30":
            return row_callback(identity)
        return _row(identity, arm=arm)

    def slow_summary_factory(*_args: object) -> dict[str, object]:
        time.sleep(0.3)
        return {"schema": "too-late"}

    outcomes = run_synthetic_exp007_schedule(
        identities,
        callback,
        config_template=config,
        stage_summary_factory=slow_summary_factory,
    )

    assert seen == ["S30"]
    assert outcomes["S30"].failure_record is not None
    assert outcomes["S30"].failure_record["failure_kind"] == "schedule_deadline"
    assert outcomes["S30"].failure_record["failure_stage"] == "arm_summary"
    assert outcomes["S60"].status == "not_run"


def test_summary_deadline_alarm_restores_prior_timer() -> None:
    if not hasattr(signal, "setitimer") or not hasattr(signal, "SIGALRM"):
        pytest.skip("POSIX interval timers unavailable")
    identities = _identities(80)
    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.setitimer(signal.ITIMER_REAL, 1.0)
    try:
        outcome = run_synthetic_exp007_arm(
            identities,
            lambda identity: _row(identity),
            config=Exp007RunnerConfig(
                stage=EXP007_REPAIR_STAGE,
                use_spawn_pool=False,
                repair_stop_seconds=0.01,
            ),
            stage_summary_factory=lambda *_args: (time.sleep(0.05) or {}),
        )
        delay, interval = signal.getitimer(signal.ITIMER_REAL)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)

    assert not outcome.ok
    assert signal.getsignal(signal.SIGALRM) == previous_handler
    assert delay > 0.0
    assert interval == 0.0


def test_control_worker_failure_preserves_diagnostics_failure_kind_and_stage() -> None:
    identities = _identities(16)
    supervisor = _ready_supervisor(identities)
    ack = supervisor.start_row(
        RowStartedEvent(
            row=identities[0],
            slot=0,
            generation_nonce=_sha("nonce-0"),
            pid=101,
            worker_ready_ns=1,
        ),
        parent_start_ns=10,
    )
    parent_conn, child_conn = mp.Pipe()
    child_conn.send(
        {
            "type": "row_failed",
            "row": identities[0].pending_ref(),
            "slot": 0,
            "generation_nonce": ack.generation_nonce,
            "pid": ack.pid,
            "token": ack.token,
            "worker_elapsed_ns": 20,
            "worker_rss_bytes": 123,
            "failure_kind": "diagnostics_integrity_failure",
            "failure_stage": "diagnostics",
            "diagnostics_classification": {
                "reason": "diagnostics_integrity_failure",
                "stage": "diagnostics",
            },
        }
    )
    try:
        with pytest.raises(Exp007RunnerError, match="diagnostics_integrity_failure"):
            _process_control_payload(
                parent_conn,
                supervisor=supervisor,
                worker_slot_lifetime_bytes=[None, None, None, None],
                received_ns=30,
            )
    finally:
        parent_conn.close()
        child_conn.close()

    assert supervisor.failure_record is not None
    assert supervisor.failure_record["failure_kind"] == "diagnostics_integrity_failure"
    assert supervisor.failure_record["failure_stage"] == "diagnostics"


def test_control_eof_prefers_typed_result_exception_before_broken_stream() -> None:
    identities = _identities(16)
    supervisor = _ready_supervisor(identities)
    supervisor.start_row(
        RowStartedEvent(
            row=identities[0],
            slot=0,
            generation_nonce=_sha("nonce-0"),
            pid=101,
            worker_ready_ns=1,
        ),
        parent_start_ns=10,
    )
    result_iter = _TimeoutResultIterator()

    with pytest.raises(Exp007RunnerError, match="result iterator raised"):
        _handle_control_eof(
            result_iter,
            supervisor=supervisor,
            identities=identities,
            expected_envelope_index=0,
            worker_slot_lifetime_bytes=[None, None, None, None],
            cause=EOFError(),
        )

    assert result_iter.timeout > 0.0
    assert supervisor.failure_record is not None
    assert supervisor.failure_record["failure_kind"] == "row_timeout"
    assert supervisor.failure_record["failure_stage"] == "pool_stream"
    assert supervisor.failure_record["causing_row_index"] == 0


def test_bounded_teardown_raises_when_survivor_remains_alive() -> None:
    pool = _FakePool((101, 102, 103, 104), alive_after_kill=True)

    with pytest.raises(Exp007RunnerError, match="bounded pool teardown failed"):
        _terminate_pool_bounded(
            pool,
            Exp007RunnerConfig(
                use_spawn_pool=False,
                worker_terminate_grace_seconds=0.001,
                worker_kill_grace_seconds=0.001,
            ),
        )


def test_rss_normalization_is_platform_specific() -> None:
    assert normalize_ru_maxrss_bytes(123, system="Darwin") == (123, "macos_bytes")
    assert normalize_ru_maxrss_bytes(123, system="Linux") == (123 * 1024, "linux_kib_times_1024")
    with pytest.raises(RuntimeError):
        normalize_ru_maxrss_bytes(1, system="Plan9")


def _ready_supervisor(
    identities: tuple[Exp007SyntheticIdentity, ...],
    *,
    config: Exp007RunnerConfig | None = None,
) -> Exp007ArmSupervisor:
    supervisor = Exp007ArmSupervisor(
        identities=identities,
        config=config or Exp007RunnerConfig(use_spawn_pool=False),
        initial_pid_order=(101, 102, 103, 104),
    )
    for slot, pid in enumerate((101, 102, 103, 104)):
        supervisor.accept_hello(WorkerHello(pid=pid, generation_nonce=_sha(f"nonce-{slot}")))
    supervisor.assert_handshake_complete()
    return supervisor


def _ready_supervisor_without_hellos(
    identities: tuple[Exp007SyntheticIdentity, ...],
    *,
    config: Exp007RunnerConfig | None = None,
) -> Exp007ArmSupervisor:
    return Exp007ArmSupervisor(
        identities=identities,
        config=config or Exp007RunnerConfig(use_spawn_pool=False),
        initial_pid_order=(101, 102, 103, 104),
    )


def _supervisor_complete_row(
    supervisor: Exp007ArmSupervisor,
    identity: Exp007SyntheticIdentity,
    *,
    slot: int,
    row_callback: Callable[
        [Exp007SyntheticIdentity],
        Exp007SyntheticWorkerResult,
    ] | None = None,
) -> None:
    pid = 101 + slot
    generation_nonce = _sha(f"nonce-{slot}")
    ack = supervisor.start_row(
        RowStartedEvent(
            row=identity,
            slot=slot,
            generation_nonce=generation_nonce,
            pid=pid,
            worker_ready_ns=1,
        ),
        parent_start_ns=1000 + identity.row_index,
    )
    row = _row(identity) if row_callback is None else row_callback(identity)
    supervisor.finish_row(
        RowFinishedEvent(
            row=identity,
            slot=slot,
            generation_nonce=ack.generation_nonce,
            pid=ack.pid,
            token=ack.token,
            worker_elapsed_ns=10,
            envelope_sha256=canonical_sha256(row),
        ),
        received_ns=2000 + identity.row_index,
    )
    supervisor.receive_envelope(row, received_ns=3000 + identity.row_index)


def _identities(count: int) -> tuple[Exp007SyntheticIdentity, ...]:
    return tuple(
        Exp007SyntheticIdentity(
            row_index=index,
            cache_audio_key=f"audio-{index:03d}",
            audio_group_key=f"group-{index:03d}",
            identity_payload_sha256=_sha(f"identity-{index}"),
        )
        for index in range(count)
    )


def _row(
    identity: Exp007SyntheticIdentity,
    *,
    arm: str = "S30",
    row_payload_sha256: str | None = None,
    candidate_reference_entry_payload_sha256: str | None = None,
    row_result: Mapping[str, Any] | None = None,
) -> Exp007SyntheticWorkerResult:
    row_result_payload = (
        _synthetic_schedule_row_result(identity, arm=arm)
        if row_result is None
        else protocol.validate_row_result(row_result)
    )
    if row_payload_sha256 is None:
        row_payload_sha256 = row_result_payload["row_payload_sha256"]
    elif row_payload_sha256 == row_result_payload["row_payload_sha256"]:
        pass
    elif row_result is not None:
        raise ValueError("row_result payload SHA does not match synthetic wrapper")
    else:
        row_result_payload = None
    candidate_reference_sha = (
        candidate_reference_entry_payload_sha256
        if candidate_reference_entry_payload_sha256 is not None
        else (_sha(f"candidate:{identity.row_index}") if arm == "S30" else None)
    )
    return Exp007SyntheticWorkerResult(
        row_index=identity.row_index,
        cache_audio_key=identity.cache_audio_key,
        identity_payload_sha256=identity.identity_payload_sha256,
        row_payload_sha256=row_payload_sha256,
        candidate_reference_entry_payload_sha256=candidate_reference_sha,
        audio_group_key=identity.audio_group_key,
        payload={
            "candidate_status": "accepted",
            "arm": arm,
            **({} if row_result_payload is None else {"row_result": row_result_payload}),
        },
    )


def _failing_row_callback(identity: Exp007SyntheticIdentity) -> Exp007SyntheticWorkerResult:
    if identity.row_index == 1:
        raise RuntimeError("synthetic failure")
    return _row(identity)


def _synthetic_schedule_row_result(
    identity: Exp007SyntheticIdentity,
    *,
    arm: str,
    input_manifest_sha256: str = "2" * 64,
    source_closure_fingerprint_sha256: str = "1" * 64,
    run_config_fingerprint_sha256: str = "0" * 64,
) -> dict[str, Any]:
    candidate = _candidate_payload(
        identity.row_index,
        input_signal_sha256=_input_sha(identity.row_index),
    )
    serialized = artifacts.serialize_candidate_payload(candidate)
    return protocol.minimal_row_result(
        stage=protocol.EXP007_SCHEDULE_STAGE,
        schedule_arm=arm,
        row_index=identity.row_index,
        cache_audio_key=identity.cache_audio_key,
        audio_group_key=identity.audio_group_key or identity.cache_audio_key,
        identity_payload_sha256=identity.identity_payload_sha256,
        source_closure_fingerprint_sha256=source_closure_fingerprint_sha256,
        run_config_fingerprint_sha256=run_config_fingerprint_sha256,
        selector_manifest_sha256=input_manifest_sha256,
        input_manifest_sha256=input_manifest_sha256,
        candidate_payload_schema=serialized.schema,
        candidate_payload_byte_count=serialized.byte_count,
        candidate_payload_field_set_sha256=serialized.field_set_sha256,
        candidate_payload_sha256=serialized.payload_sha256,
        candidate_fingerprint=serialized.candidate_fingerprint,
    )


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bad_stage_summary_factory(
    _config: Exp007RunnerConfig,
    _rows: tuple[Exp007SyntheticWorkerResult, ...],
    _worker_slot_lifetime_bytes: tuple[int | None, ...],
    _adapter: object,
    _telemetry: RunnerStageTelemetry,
) -> dict[str, object]:
    return {"schema": "not_exp007"}


def _summary_publication_failure_factory(
    _config: Exp007RunnerConfig,
    _rows: tuple[Exp007SyntheticWorkerResult, ...],
    _worker_slot_lifetime_bytes: tuple[int | None, ...],
    _adapter: object,
    _telemetry: RunnerStageTelemetry,
) -> dict[str, object]:
    raise Exp007WorkerFailure(
        "summary publish failed",
        failure_kind="summary_publication_failure",
        failure_stage="arm_summary",
    )


def _publish_artifacts_reference_manifest(
    root: Path,
    identities: tuple[Exp007SyntheticIdentity, ...],
) -> tuple[dict[str, Any], dict[int, str]]:
    row_payload_sha_by_index: dict[int, str] = {}
    for identity in identities:
        row, _entry_sha = _publish_candidate_reference_row_bundle(
            root,
            identity,
            stage=protocol.EXP007_SCHEDULE_STAGE,
            arm="S30",
            input_manifest_sha256="2" * 64,
            source_closure_fingerprint_sha256="1" * 64,
            run_config_fingerprint_sha256="0" * 64,
        )
        row_payload_sha_by_index[identity.row_index] = row["row_payload_sha256"]
    manifest = artifacts.build_candidate_reference_manifest(
        root=root,
        stage=protocol.EXP007_SCHEDULE_STAGE,
        input_manifest_sha256="2" * 64,
        source_closure_fingerprint_sha256="1" * 64,
        reference_arm="S30",
    )
    _write_candidate_reference_manifest(root, manifest)
    return manifest, row_payload_sha_by_index


def _write_candidate_reference_manifest(
    root: Path,
    manifest: dict[str, Any],
) -> str:
    relative_path = artifacts.candidate_reference_manifest_relative_path(
        manifest["stage"],
        manifest["reference_arm"],
    )
    destination = root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(protocol.canonical_json_bytes(manifest))
    return relative_path


def _publish_candidate_reference_row_bundle(
    root: Path,
    identity: Exp007SyntheticIdentity,
    *,
    stage: str,
    arm: str,
    input_manifest_sha256: str,
    source_closure_fingerprint_sha256: str,
    run_config_fingerprint_sha256: str,
) -> tuple[dict[str, Any], str]:
    candidate = _candidate_payload(
        identity.row_index,
        input_signal_sha256=_input_sha(identity.row_index),
    )
    serialized = artifacts.serialize_candidate_payload(candidate)
    row = protocol.minimal_row_result(
        stage=stage,
        schedule_arm=arm,
        row_index=identity.row_index,
        cache_audio_key=identity.cache_audio_key,
        audio_group_key=identity.audio_group_key or identity.cache_audio_key,
        identity_payload_sha256=identity.identity_payload_sha256,
        source_closure_fingerprint_sha256=source_closure_fingerprint_sha256,
        run_config_fingerprint_sha256=run_config_fingerprint_sha256,
        selector_manifest_sha256=(
            _sha("repair-selector")
            if stage == protocol.EXP007_REPAIR_STAGE
            else input_manifest_sha256
        ),
        input_manifest_sha256=input_manifest_sha256,
        candidate_payload_schema=serialized.schema,
        candidate_payload_byte_count=serialized.byte_count,
        candidate_payload_field_set_sha256=serialized.field_set_sha256,
        candidate_payload_sha256=serialized.payload_sha256,
        candidate_fingerprint=serialized.candidate_fingerprint,
    )
    bundle = artifacts.make_candidate_reference_row_bundle(
        stage=stage,
        schedule_arm=arm,
        row=row,
        candidate_payload=candidate,
        input_signal_sha256=candidate["diagnostics"]["input_signal_sha256"],
    )
    relative_path = artifacts.reference_bundle_relative_path(
        stage,
        arm,
        identity.row_index,
    )
    destination = root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(protocol.canonical_json_bytes(bundle))
    return row, bundle["entry"]["entry_payload_sha256"]


def _schedule_artifact_config_and_callback(
    root: Path,
    identities: tuple[Exp007SyntheticIdentity, ...],
    *,
    arm: str = "S30",
    **kwargs: object,
) -> tuple[
    Exp007RunnerConfig,
    Callable[[Exp007SyntheticIdentity], Exp007SyntheticWorkerResult],
    Callable[..., StageSummaryPublication],
]:
    config = _schedule_config(
        identities,
        arm=arm,
        **kwargs,
    )
    return (
        config,
        _ScheduleArtifactRowCallback(root=root, config=config),
        _schedule_stage_summary_factory(root),
    )


@dataclass(frozen=True)
class _ScheduleArtifactRowCallback:
    root: Path
    config: Exp007RunnerConfig

    def __call__(
        self,
        identity: Exp007SyntheticIdentity,
    ) -> Exp007SyntheticWorkerResult:
        if self.config.schedule_arm == "S30":
            row, entry_sha = _publish_candidate_reference_row_bundle(
                self.root,
                identity,
                stage=protocol.EXP007_SCHEDULE_STAGE,
                arm="S30",
                input_manifest_sha256=self.config.input_manifest_sha256,
                source_closure_fingerprint_sha256=(
                    self.config.source_closure_fingerprint_sha256
                ),
                run_config_fingerprint_sha256=(
                    self.config.run_config_fingerprint_sha256
                ),
            )
            return _row(
                identity,
                row_payload_sha256=row["row_payload_sha256"],
                candidate_reference_entry_payload_sha256=entry_sha,
                row_result=row,
            )
        row = _synthetic_schedule_row_result(
            identity,
            arm=self.config.schedule_arm,
            input_manifest_sha256=self.config.input_manifest_sha256,
            source_closure_fingerprint_sha256=(
                self.config.source_closure_fingerprint_sha256
            ),
            run_config_fingerprint_sha256=(
                self.config.run_config_fingerprint_sha256
            ),
        )
        return _row(identity, arm=self.config.schedule_arm, row_result=row)


def _repair_artifact_config_and_callback(
    root: Path,
    *,
    summary_worker_rss: tuple[int, int, int, int] | None = None,
    summary_wall_seconds: float | None = None,
) -> tuple[
    tuple[Exp007SyntheticIdentity, ...],
    Exp007RunnerConfig,
    Callable[[Exp007SyntheticIdentity], Exp007SyntheticWorkerResult],
    Callable[..., StageSummaryPublication],
]:
    deps = _repair_schedule_chain_dependencies(root)
    weak_outcome = deps["weak_outcome"]
    binding = deps["binding"]
    (
        repair_rows,
        prediction_rows,
        weak_rows,
        manifest,
    ) = _repair_authority_inputs(
        root,
        binding=binding,
        deps=deps,
        repair80_identity_rows=deps["repair80_identity_rows"],
        run_config_fingerprint_sha256=deps["repair_config"][
            "run_config_fingerprint_sha256"
        ],
        selector_manifest_sha256=deps["repair_config"]["selector_manifest_sha256"],
    )
    identities = tuple(
        Exp007SyntheticIdentity(
            row_index=row["row_index"],
            cache_audio_key=row["cache_audio_key"],
            audio_group_key=row["audio_group_key"],
            identity_payload_sha256=row["identity_payload_sha256"],
        )
        for row in prediction_rows
    )

    def callback(identity: Exp007SyntheticIdentity) -> Exp007SyntheticWorkerResult:
        row = prediction_rows[identity.row_index]
        entry = manifest["entries"][identity.row_index]
        result = _row(
            identity,
            arm="S64",
            row_payload_sha256=row["row_payload_sha256"],
            candidate_reference_entry_payload_sha256=entry["entry_payload_sha256"],
        )
        return replace(
            result,
            payload={**result.payload, "row_result": row},
        )

    def publication_factory(
        _config: Exp007RunnerConfig,
        rows: tuple[Exp007SyntheticWorkerResult, ...],
        _worker_slot_lifetime_bytes: tuple[int | None, ...],
        _adapter: object,
        telemetry: RunnerStageTelemetry,
    ) -> StageSummaryPublication:
        committed_prediction_rows = [
            dict(row.payload["row_result"])
            for row in rows
        ]
        worker_rss = (
            telemetry.worker_lifetime_rss_bytes
            if summary_worker_rss is None
            else summary_worker_rss
        )
        evaluation = evaluate_repair80(
            repair_rows,
            selected_schedule_arm="S64",
            worker_lifetime_rss_bytes=worker_rss,
        )
        wall_seconds = (
            telemetry.aggregate_wall_seconds
            if summary_wall_seconds is None
            else summary_wall_seconds
        )
        summary = make_repair80_summary(
            evaluation,
            schedule_arm="S64",
            repair80_input_binding=binding,
            schedule_weak_veto_outcome=weak_outcome,
            run_config_fingerprint_sha256=deps["repair_config"][
                "run_config_fingerprint_sha256"
            ],
            candidate_reference_manifest=manifest,
            artifact_root=root,
            row_refs=_completed_row_refs_from_manifest(
                committed_prediction_rows,
                manifest,
            ),
            weak_row_refs=_weak_row_refs_from_rows(weak_rows),
            aggregate_wall_seconds=wall_seconds,
            worker_lifetime_rss_bytes=worker_rss,
        )
        context = RepairSummaryValidationContext(
            repair_run_config=deps["repair_config"],
            source_closure=deps["source_closure"],
            source_repo_root=deps["source_repo_root"],
            repair_metric_rows=repair_rows,
            repair80_input_binding=binding,
            repair80_identity_source_artifact=deps[
                "repair80_identity_source_artifact_bytes"
            ],
            repair80_label_source_artifact=deps[
                "repair80_label_source_artifact_bytes"
            ],
            repair80_identity_rows=deps["repair80_identity_rows"],
            repair80_label_rows=deps["repair80_label_rows"],
            schedule_weak_veto_outcome=weak_outcome,
            four_arm_stage_summary=deps["four_arm_summary"],
            config_selection=deps["config_selection"],
            candidate_global_manifest=deps["candidate_global_manifest"],
            schedule_candidate_reference_manifest=deps[
                "candidate_reference_manifest"
            ],
            candidate_reference_manifest=manifest,
            artifact_root=root,
            run_configs_by_arm=deps["run_configs_by_arm"],
            arm_rows_by_arm=deps["arm_rows_by_arm"],
            candidate_payloads_by_arm=deps["candidate_payloads_by_arm"],
            arm_stage_outcomes_by_execution_order=deps["arm_outcomes"],
            source_arm_stage_summaries_by_execution_order=deps[
                "source_arm_summaries"
            ],
            prediction_rows=committed_prediction_rows,
            weak_rows=weak_rows,
        )
        return StageSummaryPublication(
            summary,
            context,
            candidate_reference_manifest=manifest,
            candidate_reference_artifact_root=root,
            candidate_reference_manifest_relative_path=(
                artifacts.candidate_reference_manifest_relative_path(
                    manifest["stage"],
                    manifest["reference_arm"],
                )
            ),
        )

    config = Exp007RunnerConfig(
        stage=EXP007_REPAIR_STAGE,
        schedule_arm="S64",
        use_spawn_pool=False,
        run_config_fingerprint_sha256=deps["repair_config"][
            "run_config_fingerprint_sha256"
        ],
        source_closure_fingerprint_sha256=deps["source_closure_fingerprint_sha256"],
        input_manifest_sha256=binding["binding_fingerprint_sha256"],
    )
    return identities, config, callback, publication_factory


def _repair_schedule_chain_dependencies(root: Path) -> dict[str, Any]:
    source_repo_root, source_closure = _repo_source_closure()
    source_sha = source_closure["source_closure_fingerprint_sha256"]
    selector_sha = _sha("selector-manifest")
    run_configs_by_arm = {
        arm: protocol.make_run_config(
            stage=protocol.EXP007_SCHEDULE_STAGE,
            schedule_arm=arm,
            selector_manifest_sha256=selector_sha,
            input_manifest_sha256=selector_sha,
            source_closure_fingerprint_sha256=source_sha,
            cache_config_sha256=_sha("cache-config"),
            grid_fitter_config_sha256=_sha("grid-fitter-config"),
            weak_config_sha256=_sha("weak-config"),
        )
        for arm in EXP007_EXECUTION_ORDER
    }
    candidate_payloads = [
        _candidate_payload(index, input_signal_sha256=_input_sha(index))
        for index in range(16)
    ]
    arm_rows_by_arm = {
        arm: [
            _schedule_row_result(
                index,
                arm=arm,
                selector_manifest_sha256=selector_sha,
                source_closure_fingerprint_sha256=source_sha,
                run_config_fingerprint_sha256=run_configs_by_arm[arm][
                    "run_config_fingerprint_sha256"
                ],
                candidate_payload=candidate_payloads[index],
            )
            for index in range(16)
        ]
        for arm in EXP007_EXECUTION_ORDER
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
        artifacts.publish_candidate_reference_row_bundle(root=root, bundle=bundle)
    candidate_reference_manifest = artifacts.build_candidate_reference_manifest(
        root=root,
        stage=protocol.EXP007_SCHEDULE_STAGE,
        input_manifest_sha256=selector_sha,
        source_closure_fingerprint_sha256=source_sha,
        reference_arm="S30",
    )
    artifacts.publish_candidate_reference_manifest(
        root=root,
        manifest=candidate_reference_manifest,
    )
    candidate_reference_sha = protocol.object_complete_sha256(
        candidate_reference_manifest
    )
    candidate_payloads_by_arm = {
        arm: list(candidate_payloads)
        for arm in EXP007_EXECUTION_ORDER
    }
    candidate_global_manifest = artifacts.build_candidate_global_manifest(
        root=root,
        reference_manifest=candidate_reference_manifest,
        selector_manifest_sha256=selector_sha,
        source_closure_fingerprint_sha256=source_sha,
        run_configs_by_arm=run_configs_by_arm,
        arm_rows_by_arm=arm_rows_by_arm,
        candidate_payloads_by_arm=candidate_payloads_by_arm,
    )
    candidate_global_sha = protocol.object_complete_sha256(candidate_global_manifest)
    reference_entry_sha256s = [
        entry["candidate_reference_entry_payload_sha256"]
        for entry in candidate_global_manifest["entries"]
    ]
    source_evaluations = {
        arm: evaluate_source_arm(
            [
                _source_metric_row_from_row_result(row)
                for row in arm_rows_by_arm[arm]
            ],
            schedule_arm=arm,  # type: ignore[arg-type]
            worker_lifetime_rss_bytes=(1024, 1024, 1024, 1024),
            aggregate_wall_seconds=16.0,
        )
        for arm in EXP007_EXECUTION_ORDER
    }
    source_arm_summaries = {
        arm: _source_arm_stage_summary_from_evaluation(
            rows=arm_rows_by_arm[arm],
            evaluation=source_evaluations[arm],
            candidate_reference_manifest_sha256=candidate_reference_sha,
            candidate_reference_entry_payload_sha256s=reference_entry_sha256s,
        )
        for arm in EXP007_EXECUTION_ORDER
    }
    arm_outcomes = {
        arm: protocol.make_arm_stage_success(
            stage=protocol.EXP007_SCHEDULE_STAGE,
            schedule_arm=arm,
            row_payloads_sha256=source_arm_summaries[arm]["row_payloads_sha256"],
            candidate_reference_manifest_sha256=candidate_reference_sha,
            stage_summary_sha256=protocol.object_complete_sha256(
                source_arm_summaries[arm]
            ),
        )
        for arm in EXP007_EXECUTION_ORDER
    }
    outcome_map = {
        arm: protocol.object_complete_sha256(outcome)
        for arm, outcome in arm_outcomes.items()
    }
    selected = select_source_schedule(source_evaluations)  # type: ignore[arg-type]
    assert selected.selected_schedule_arm == "S64"
    config_selection = protocol.make_config_selection(
        arm_outcome_sha256_by_execution_order=outcome_map,
        candidate_global_manifest_sha256=candidate_global_sha,
        source_closure_fingerprint_sha256=source_sha,
        selector_manifest_sha256=selector_sha,
        overlap_common=_metrics_audio_binding_dict(selected.overlap_common),
        section_common=_metrics_audio_binding_dict(selected.section_common),
        source_decision=selected.source_decision,
        arm_order_values=[
            _metrics_arm_order_value_dict(value)
            for value in selected.arm_order_values
        ],
        selected_schedule_arm=selected.selected_schedule_arm,
        selected_run_config_fingerprint_sha256=source_arm_summaries[
            selected.selected_schedule_arm
        ]["run_config_fingerprint_sha256"],
    )
    config_selection_sha = protocol.object_complete_sha256(config_selection)
    four_arm_summary = protocol.make_four_arm_stage_summary(
        status="success",
        arm_outcome_sha256_by_execution_order=outcome_map,
        candidate_global_manifest_sha256=candidate_global_sha,
        source_selection_status="positive",
        config_selection_sha256=config_selection_sha,
    )
    four_arm_sha = protocol.object_complete_sha256(four_arm_summary)
    schedule_weak_rows = [
        replace(
            _weak_metric(index, arm="S64"),
            prediction_row_sha256=arm_rows_by_arm["S64"][index][
                "row_payload_sha256"
            ],
        )
        for index in range(16)
    ]
    weak_evaluation = evaluate_schedule_weak_veto(
        schedule_weak_rows,
        selected_rows=_selected_prediction_refs(schedule_weak_rows),
        selected_schedule_arm="S64",
    )
    weak_summary = make_schedule_weak_veto_summary(
        weak_evaluation,
        schedule_arm="S64",
        four_arm_stage_summary_sha256=four_arm_sha,
        candidate_global_manifest_sha256=candidate_global_sha,
        source_closure_fingerprint_sha256=source_sha,
        source_selection_sha256=config_selection_sha,
    )
    weak_outcome = make_schedule_weak_success_outcome(weak_summary)
    weak_outcome_sha = protocol.object_complete_sha256(weak_outcome)
    source_context = _repair_identity_label_sources()
    binding = protocol.make_repair80_input_binding(
        identity_source=source_context["identity_source"],
        label_source=source_context["label_source"],
        four_arm_stage_summary_sha256=four_arm_sha,
        candidate_global_manifest_sha256=candidate_global_sha,
        source_selection_sha256=config_selection_sha,
        schedule_weak_veto_outcome_sha256=weak_outcome_sha,
    )
    repair_config = protocol.make_run_config(
        stage=protocol.EXP007_REPAIR_STAGE,
        schedule_arm="S64",
        selector_manifest_sha256=selector_sha,
        input_manifest_sha256=binding["binding_fingerprint_sha256"],
        schedule_weak_veto_outcome_sha256=weak_outcome_sha,
        source_closure_fingerprint_sha256=source_sha,
        cache_config_sha256=_sha("cache-config"),
        grid_fitter_config_sha256=_sha("grid-fitter-config"),
        weak_config_sha256=_sha("weak-config"),
    )
    return {
        **source_context,
        "arm_outcomes": arm_outcomes,
        "arm_rows_by_arm": arm_rows_by_arm,
        "run_configs_by_arm": run_configs_by_arm,
        "candidate_global_manifest": candidate_global_manifest,
        "candidate_payloads_by_arm": candidate_payloads_by_arm,
        "candidate_reference_manifest": candidate_reference_manifest,
        "config_selection": config_selection,
        "four_arm_summary": four_arm_summary,
        "four_arm_stage_summary_sha256": four_arm_sha,
        "candidate_global_manifest_sha256": candidate_global_sha,
        "source_selection_sha256": config_selection_sha,
        "source_closure": source_closure,
        "source_repo_root": source_repo_root,
        "source_closure_fingerprint_sha256": source_sha,
        "source_arm_summaries": source_arm_summaries,
        "weak_outcome": weak_outcome,
        "binding": binding,
        "repair_config": repair_config,
    }


_SOURCE_CLOSURE_CACHE: tuple[Path, dict[str, Any]] | None = None


def _repo_root() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / "src" / "pulsefield_model").is_dir():
        raise AssertionError("repo root not found")
    return repo_root


def _clone(value: Any) -> Any:
    return protocol.load_json_strict(protocol.canonical_json_bytes(value))


def _repo_source_closure() -> tuple[Path, dict[str, Any]]:
    global _SOURCE_CLOSURE_CACHE
    if _SOURCE_CLOSURE_CACHE is None:
        repo_root = _repo_root()
        _SOURCE_CLOSURE_CACHE = (
            repo_root,
            protocol.make_source_closure(
                repo_root,
                generated_at_utc="2026-08-13T00:00:00Z",
            ),
        )
    repo_root, closure = _SOURCE_CLOSURE_CACHE
    return repo_root, _clone(closure)


def _schedule_row_result(
    index: int,
    *,
    arm: str,
    selector_manifest_sha256: str,
    source_closure_fingerprint_sha256: str,
    run_config_fingerprint_sha256: str,
    candidate_payload: dict[str, Any],
) -> dict[str, Any]:
    serialized = artifacts.serialize_candidate_payload(candidate_payload)
    return protocol.minimal_row_result(
        stage=protocol.EXP007_SCHEDULE_STAGE,
        schedule_arm=arm,
        row_index=index,
        cache_audio_key=f"audio-{index:03d}",
        audio_group_key=f"group-{index:03d}",
        identity_payload_sha256=_sha(f"schedule-identity-{index}"),
        source_closure_fingerprint_sha256=source_closure_fingerprint_sha256,
        run_config_fingerprint_sha256=run_config_fingerprint_sha256,
        selector_manifest_sha256=selector_manifest_sha256,
        input_manifest_sha256=selector_manifest_sha256,
        candidate_payload_schema=serialized.schema,
        candidate_payload_byte_count=serialized.byte_count,
        candidate_payload_field_set_sha256=serialized.field_set_sha256,
        candidate_payload_sha256=serialized.payload_sha256,
        candidate_fingerprint=serialized.candidate_fingerprint,
    )


def _source_metric_row_from_row_result(row: Mapping[str, Any]) -> SourceMetricRow:
    validated = protocol.validate_row_result(row)
    methods = validated["methods"]
    candidate = methods["candidate"]
    baseline = methods["baseline"]
    selected = methods["selected"]
    flags = validated["denominator_flags"]
    diagnostics = validated["diagnostics_summary"]
    candidate_accepted = candidate["status"] == "accepted"
    baseline_accepted = baseline["status"] == "accepted"
    return SourceMetricRow(
        schedule_arm=validated["schedule_arm"],  # type: ignore[arg-type]
        row_index=validated["row_index"],
        cache_audio_key=validated["cache_audio_key"],
        audio_group_key=validated["audio_group_key"],
        cache_valid=flags["cache_valid"],
        projection_evaluable=flags["projection_evaluable"],
        candidate_status=candidate["status"],  # type: ignore[arg-type]
        candidate_fallback_reason=(
            None if candidate_accepted else candidate["reason"]
        ),
        baseline_status=baseline["status"],  # type: ignore[arg-type]
        selected_status=selected["status"],  # type: ignore[arg-type]
        candidate_section_count=(
            candidate["grid_summary"]["section_count"]
            if candidate_accepted
            else None
        ),
        current_v2_segment_count=(
            len(baseline["grid"]["payload"]["segments"])
            if baseline_accepted
            else None
        ),
        current_v2_projection_sha256=(
            baseline["deterministic_projection_sha256"]
            if baseline_accepted
            else None
        ),
        candidate_seam_ms=(
            candidate["grid_summary"]["maximum_seam_discontinuity_ms"]
            if candidate_accepted
            else None
        ),
        overlap_p90_ms=(
            diagnostics["overlap"]["p90_ms"] if flags["overlap_available"] else None
        ),
        audio_arm_seconds=validated["runtime"]["audio_arm_seconds"],
        row_json_bytes=len(protocol.canonical_json_bytes(validated)),
        replay_schema_source_cache_candidate_v2_consistent=all(
            value
            for name, value in validated["hard_guards"].items()
            if name != "timed_out"
        ),
    )


def _source_arm_stage_summary_from_evaluation(
    *,
    rows: Sequence[Mapping[str, Any]],
    evaluation: Any,
    candidate_reference_manifest_sha256: str,
    candidate_reference_entry_payload_sha256s: Sequence[str],
) -> dict[str, Any]:
    validated_rows = [protocol.validate_row_result(row) for row in rows]
    refs = [
        protocol.make_completed_row_ref(
            row_index=row["row_index"],
            cache_audio_key=row["cache_audio_key"],
            identity_payload_sha256=row["identity_payload_sha256"],
            row_payload_sha256=row["row_payload_sha256"],
            candidate_reference_entry_payload_sha256=(
                candidate_reference_entry_payload_sha256s[index]
            ),
        )
        for index, row in enumerate(validated_rows)
    ]
    denominators = evaluation.denominators
    gates = evaluation.gates
    runtime_values = [
        row["runtime"]["audio_arm_seconds"]
        for row in validated_rows
    ]
    worker_lifetime = list(evaluation.rss_summary.worker_lifetime_bytes)
    payload = {
        "schema": protocol.SOURCE_ARM_STAGE_SUMMARY_SCHEMA,
        "experiment_id": protocol.EXP007_EXPERIMENT_ID,
        "stage": protocol.EXP007_SCHEDULE_STAGE,
        "schema_descriptor_sha256": protocol.schema_descriptor_sha256(
            protocol.SOURCE_ARM_STAGE_SUMMARY_SCHEMA
        ),
        "schedule_arm": evaluation.schedule_arm,
        "run_config_fingerprint_sha256": validated_rows[0][
            "run_config_fingerprint_sha256"
        ],
        "source_closure_fingerprint_sha256": validated_rows[0][
            "source_closure_fingerprint_sha256"
        ],
        "selector_manifest_sha256": validated_rows[0]["selector_manifest_sha256"],
        "candidate_reference_manifest_sha256": candidate_reference_manifest_sha256,
        "row_count": 16,
        "row_refs": refs,
        "row_payloads_sha256": protocol.canonical_json_sha256(refs),
        "denominators": {
            "stage_audio_count": denominators.stage_audio_count,
            "stage_audio": denominators.stage_audio.to_dict(),
            "cache_valid_audio": denominators.cache_valid_audio.to_dict(),
            "projection_evaluable_audio": (
                denominators.projection_evaluable_audio.to_dict()
            ),
            "candidate_accepted_audio": (
                denominators.candidate_accepted_audio.to_dict()
            ),
            "candidate_fallback_audio": (
                denominators.candidate_fallback_audio.to_dict()
            ),
            "selected_product_fallback_audio": (
                denominators.selected_product_fallback_audio.to_dict()
            ),
            "baseline_accepted_audio": (
                denominators.baseline_accepted_audio.to_dict()
            ),
            "product_grid_available_audio": (
                denominators.product_grid_available_audio.to_dict()
            ),
            "no_origin_or_path_audio": (
                denominators.no_origin_or_path_audio.to_dict()
            ),
            "resource_cap_fallback_audio": (
                denominators.resource_cap_fallback_audio.to_dict()
            ),
            "overlap_available_audio": (
                denominators.overlap_available_audio.to_dict()
            ),
        },
        "gates": {
            "candidate_fallback_rate": protocol.make_rate_value(
                gates.candidate_fallback_rate.numerator,
                gates.candidate_fallback_rate.denominator,
            ),
            "selected_product_fallback_rate": (
                protocol.make_rate_value(
                    gates.selected_product_fallback_rate.numerator,
                    gates.selected_product_fallback_rate.denominator,
                )
            ),
            "no_origin_or_path_rate": protocol.make_rate_value(
                gates.no_origin_or_path_rate.numerator,
                gates.no_origin_or_path_rate.denominator,
            ),
            "runtime_seconds": gates.runtime_seconds.to_dict(),
            "worker_rss_bytes": gates.worker_rss_bytes.to_dict(),
            "candidate_seam_ms": gates.candidate_seam_ms.to_dict(),
            "candidate_section_count": gates.candidate_section_count.to_dict(),
            "row_json_bytes": gates.row_json_bytes.to_dict(),
            "every_row_under_180_seconds": gates.every_row_under_180_seconds,
            "seam_zero": gates.seam_zero,
            "section_cap_valid": gates.section_cap_valid,
            "row_byte_cap_valid": gates.row_byte_cap_valid,
            "replay_schema_source_cache_candidate_v2_consistent": (
                gates.replay_schema_source_cache_candidate_v2_consistent
            ),
        },
        "runtime_summary": {
            "row_seconds": protocol.make_stats_value(runtime_values),
            "aggregate_wall_seconds": (
                evaluation.runtime_summary.aggregate_wall_seconds
            ),
        },
        "rss_summary": {
            "worker_count": evaluation.rss_summary.worker_count,
            "worker_lifetime_bytes": worker_lifetime,
            "arm_max_worker_bytes": evaluation.rss_summary.arm_max_worker_bytes,
        },
    }
    return protocol.validate_source_arm_stage_summary(
        protocol.with_payload_hash(payload, "summary_fingerprint_sha256")
    )


def _metrics_audio_binding_dict(value: Any) -> dict[str, Any]:
    return {
        "count": value.count,
        "sorted_cache_audio_keys_sha256": value.sorted_cache_audio_keys_sha256,
    }


def _metrics_arm_order_value_dict(value: Any) -> dict[str, Any]:
    return {
        "schedule_arm": value.schedule_arm,
        "e0_eligible": value.e0_eligible,
        "e1_eligible": value.e1_eligible,
        "elimination_reasons": list(value.elimination_reasons),
        "candidate_fallback_count": value.candidate_fallback_count,
        "no_origin_or_path_count": value.no_origin_or_path_count,
        "p90_overlap_ms": value.p90_overlap_ms,
        "section_inflation_violation_count": (
            value.section_inflation_violation_count
        ),
        "p90_section_excess": value.p90_section_excess,
        "p90_runtime": value.p90_runtime,
        "max_worker_rss": value.max_worker_rss,
        "tie_rank": value.tie_rank,
        "order_tuple_sha256": value.order_tuple_sha256,
    }


def _repair_identity_label_sources() -> dict[str, Any]:
    label_rows = []
    for index in range(80):
        if index < 5:
            label = "stable"
            is_long = True
        elif index < 20:
            label = "jump_candidate"
            is_long = False
        else:
            label = "dense"
            is_long = False
        label_row = {
            "cache_audio_key": f"audio-{index:03d}",
            "audio_group_key": f"group-{index:03d}",
            "label_stratum": label,
            "source_long_track": is_long,
            "duration_ms": 120_000 + index,
            "source": {
                "cache_audio_key": f"audio-{index:03d}",
                "long_track": is_long,
            },
            "label": {"stratum": label},
        }
        label_rows.append(label_row)
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
    return {
        "repair80_identity_rows": identity_rows,
        "repair80_label_rows": label_rows,
        "repair80_identity_source_artifact": protocol.canonical_json_bytes(
            identity_artifact
        ),
        "repair80_label_source_artifact": protocol.canonical_json_bytes(
            label_artifact
        ),
        "repair80_identity_source_artifact_bytes": protocol.canonical_json_bytes(
            identity_artifact
        ),
        "repair80_label_source_artifact_bytes": protocol.canonical_json_bytes(
            label_artifact
        ),
        "identity_source": identity_source,
        "label_source": label_source,
    }


def _source_artifact_and_ref(
    artifact_schema: str,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = {
        "schema": artifact_schema,
        "rows": [dict(row) for row in rows],
    }
    source = protocol.make_source_ref(
        artifact_schema=artifact_schema,
        sha256=protocol.canonical_json_sha256(artifact),
        row_count=len(rows),
        ordered_rows_sha256=protocol.canonical_json_sha256(artifact["rows"]),
    )
    return artifact, source


def _schedule_weak_success_outcome() -> tuple[dict[str, Any], dict[str, str]]:
    rows = [_weak_metric(index) for index in range(16)]
    deps = _weak_deps()
    evaluation = evaluate_schedule_weak_veto(
        rows,
        selected_rows=_selected_prediction_refs(rows),
        selected_schedule_arm="S64",
    )
    summary = make_schedule_weak_veto_summary(
        evaluation,
        schedule_arm="S64",
        **deps,
    )
    return make_schedule_weak_success_outcome(summary), deps


def _repair_authority_inputs(
    root: Path,
    *,
    binding: dict[str, Any],
    deps: dict[str, Any],
    repair80_identity_rows: list[dict[str, Any]],
    run_config_fingerprint_sha256: str,
    selector_manifest_sha256: str,
) -> tuple[
    list[Repair80MetricRow],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    repair_rows: list[Repair80MetricRow] = []
    prediction_rows: list[dict[str, Any]] = []
    weak_rows: list[dict[str, Any]] = []
    for index in range(80):
        if index < 5:
            label = "stable"
            is_long = True
        elif index < 20:
            label = "jump_candidate"
            is_long = False
        else:
            label = "dense"
            is_long = False
        identity_row = repair80_identity_rows[index]
        row, candidate = _repair_prediction_row_and_candidate(
            index,
            input_sha=binding["binding_fingerprint_sha256"],
            source_sha=deps["source_closure_fingerprint_sha256"],
            identity_row=identity_row,
            run_config_fingerprint_sha256=run_config_fingerprint_sha256,
            selector_manifest_sha256=selector_manifest_sha256,
        )
        bundle = artifacts.make_candidate_reference_row_bundle(
            stage=protocol.EXP007_REPAIR_STAGE,
            schedule_arm="S64",
            row=row,
            candidate_payload=candidate,
            input_signal_sha256=candidate["diagnostics"]["input_signal_sha256"],
        )
        relative_path = artifacts.reference_bundle_relative_path(
            protocol.EXP007_REPAIR_STAGE,
            "S64",
            index,
        )
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(protocol.canonical_json_bytes(bundle))
        weak_metric = replace(
            _weak_metric(
                index,
                stage="repair80",
                current_f1=1.0,
                pure_f1=1.0,
                current_drift=10.0,
                pure_drift=8.0,
            ),
            prediction_row_sha256=row["row_payload_sha256"],
        )
        weak_payload = _persisted_weak_row(weak_metric, deps, stage="repair80")
        weak_metric = replace(
            weak_metric,
            weak_row_payload_sha256=weak_payload["weak_row_payload_sha256"],
        )
        repair_rows.append(
            Repair80MetricRow(
                weak=weak_metric,
                label_stratum=label,  # type: ignore[arg-type]
                source_long_track=is_long,
                cache_valid=True,
                projection_evaluable=True,
                fallback_reason=None,
                audio_arm_seconds=0.0,
                overlap_p90_ms=0.0,
                candidate_section_count=1,
                current_v2_segment_count=1,
                seam_ms=0.0,
            )
        )
        prediction_rows.append(row)
        weak_rows.append(weak_payload)
    manifest = artifacts.build_candidate_reference_manifest(
        root=root,
        stage=protocol.EXP007_REPAIR_STAGE,
        input_manifest_sha256=binding["binding_fingerprint_sha256"],
        source_closure_fingerprint_sha256=deps["source_closure_fingerprint_sha256"],
        reference_arm="S64",
    )
    _write_candidate_reference_manifest(root, manifest)
    return repair_rows, prediction_rows, weak_rows, manifest


def _repair_prediction_row_and_candidate(
    index: int,
    *,
    input_sha: str,
    source_sha: str,
    identity_row: dict[str, Any],
    run_config_fingerprint_sha256: str,
    selector_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = _candidate_payload(
        index,
        input_signal_sha256=_input_sha(index),
    )
    serialized = artifacts.serialize_candidate_payload(candidate)
    row = protocol.minimal_row_result(
        stage=protocol.EXP007_REPAIR_STAGE,
        schedule_arm="S64",
        row_index=index,
        cache_audio_key=identity_row["cache_audio_key"],
        audio_group_key=identity_row["audio_group_key"],
        identity_payload_sha256=identity_row["identity_payload_sha256"],
        source_closure_fingerprint_sha256=source_sha,
        run_config_fingerprint_sha256=run_config_fingerprint_sha256,
        selector_manifest_sha256=selector_manifest_sha256,
        input_manifest_sha256=input_sha,
        candidate_payload_schema=serialized.schema,
        candidate_payload_byte_count=serialized.byte_count,
        candidate_payload_field_set_sha256=serialized.field_set_sha256,
        candidate_payload_sha256=serialized.payload_sha256,
        candidate_fingerprint=serialized.candidate_fingerprint,
    )
    return row, candidate


def _weak_metric(
    index: int,
    *,
    stage: str = "schedule16",
    arm: str = "S64",
    current_phase: float = 10.0,
    pure_phase: float = 10.0,
    current_drift: float = 10.0,
    pure_drift: float = 10.0,
    current_f1: float = 0.8,
    pure_f1: float = 0.8,
) -> WeakMetricRow:
    phase_current = stats_value([current_phase])
    drift_current = stats_value([current_drift])
    phase_pure = stats_value([pure_phase])
    drift_pure = stats_value([pure_drift])
    return WeakMetricRow(
        stage=stage,  # type: ignore[arg-type]
        row_index=index,
        cache_audio_key=f"audio-{index:03d}",
        audio_group_key=f"group-{index:03d}",
        prediction_row_sha256=_sha(f"prediction-{stage}-{index}"),
        weak_row_payload_sha256=_sha(f"weak-{stage}-{index}"),
        schedule_arm=arm,
        comparator_state="available",
        candidate_status="accepted",
        baseline_status="accepted",
        selected_status="accepted",
        product_grid_available=True,
        current_v2_phase_matched=True,
        pure_exp006_phase_matched=True,
        selected_safety_phase_matched=True,
        current_v2_phase_ms=phase_current,
        pure_exp006_phase_ms=phase_pure,
        product_phase_ms=phase_pure,
        current_v2_alias_max_prefix_ms=drift_current,
        pure_exp006_alias_max_prefix_ms=drift_pure,
        product_alias_max_prefix_ms=drift_pure,
        current_v2_boundary=_boundary_evidence(current_f1),
        pure_exp006_boundary=_boundary_evidence(pure_f1),
        selected_boundary=_boundary_evidence(pure_f1),
    )


def _boundary_evidence(f1: float = 0.8) -> BoundaryEvidence:
    return BoundaryEvidence(True, 1, True, ratio_value(f1, 1.0))


def _selected_prediction_refs(rows: list[WeakMetricRow]) -> list[PredictionRowRef]:
    return [
        PredictionRowRef(
            row.row_index,
            row.cache_audio_key,
            row.prediction_row_sha256,
            row.schedule_arm,
        )
        for row in rows
    ]


def _weak_deps() -> dict[str, str]:
    return {
        "four_arm_stage_summary_sha256": _sha("four-arm"),
        "candidate_global_manifest_sha256": _sha("global"),
        "source_closure_fingerprint_sha256": _sha("source"),
        "source_selection_sha256": _sha("selection"),
    }


def _persisted_weak_row(
    row: WeakMetricRow,
    deps: dict[str, str],
    *,
    stage: str | None = None,
) -> dict[str, Any]:
    return make_weak_row(
        stage=stage or row.stage,
        schedule_arm=row.schedule_arm,
        row_index=row.row_index,
        cache_audio_key=row.cache_audio_key,
        audio_group_key=row.audio_group_key,
        prediction_row_sha256=row.prediction_row_sha256,
        four_arm_stage_summary_sha256=deps["four_arm_stage_summary_sha256"],
        candidate_global_manifest_sha256=deps["candidate_global_manifest_sha256"],
        source_selection_sha256=deps["source_selection_sha256"],
        comparator_availability=ComparatorAvailability(
            "available",
            valid_difficulty_count=1,
            invalid_difficulty_count=0,
            reason=None,
            comparator_payloads_sha256=_sha("comparator"),
        ),
        current_v2_phase_matched=True,
        pure_exp006_phase_matched=True,
        selected_safety_phase_matched=True,
        phase_metrics_summary=PhaseSummary(
            current_v2_ms=row.current_v2_phase_ms,
            pure_exp006_ms=row.pure_exp006_phase_ms,
            product_ms=row.product_phase_ms,
        ),
        drift_metrics_summary=DriftSummary(
            current_v2_alias_max_prefix_ms=row.current_v2_alias_max_prefix_ms,
            pure_exp006_alias_max_prefix_ms=row.pure_exp006_alias_max_prefix_ms,
            product_alias_max_prefix_ms=row.product_alias_max_prefix_ms,
        ),
        current_v2_boundary_summary=_boundary_summary(),
        pure_exp006_boundary_summary=_boundary_summary(),
        selected_boundary_summary=_boundary_summary(),
        object_grid_summary=_object_grid_summary(),
    )


def _boundary_summary() -> BoundarySummary:
    return BoundarySummary(
        eligible=True,
        valid_difficulty_count=1,
        tp=1,
        fp=0,
        fn=0,
        f1=ratio_value(2, 2),
        matched_error_ms=stats_value([0.0]),
        weak_consensus_supported_count=1,
    )


def _object_grid_summary() -> ObjectGridSummary:
    return ObjectGridSummary(
        eligible=True,
        object_count=2,
        start_residual_ms=stats_value([1.0, 2.0]),
        end_residual_ms=stats_value([1.0, 2.0]),
        inlier_count=2,
        inlier_rate=rate_value(2, 2),
    )


def _completed_row_refs_from_manifest(
    prediction_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        protocol.make_completed_row_ref(
            row_index=row["row_index"],
            cache_audio_key=row["cache_audio_key"],
            identity_payload_sha256=row["identity_payload_sha256"],
            row_payload_sha256=row["row_payload_sha256"],
            candidate_reference_entry_payload_sha256=entry["entry_payload_sha256"],
        )
        for row, entry in zip(prediction_rows, manifest["entries"], strict=True)
    ]


def _weak_row_refs_from_rows(weak_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        protocol.validate_weak_row_ref(
            {
                "row_index": row["row_index"],
                "cache_audio_key": row["cache_audio_key"],
                "prediction_row_sha256": row["prediction_row_sha256"],
                "weak_row_payload_sha256": row["weak_row_payload_sha256"],
            }
        )
        for row in weak_rows
    ]


def _repair_publication_with_context(
    publication: StageSummaryPublication,
    context: RepairSummaryValidationContext,
) -> StageSummaryPublication:
    return StageSummaryPublication(
        publication.payload,
        context,
        candidate_reference_manifest=publication.candidate_reference_manifest,
        candidate_reference_artifact_root=publication.candidate_reference_artifact_root,
        candidate_reference_manifest_relative_path=(
            publication.candidate_reference_manifest_relative_path
        ),
    )


def _replace_summary_manifest_sha(
    summary: dict[str, Any] | object,
    manifest_sha256: str,
) -> dict[str, Any]:
    result = dict(summary)  # type: ignore[arg-type]
    result["candidate_reference_manifest_sha256"] = manifest_sha256
    result["summary_fingerprint_sha256"] = protocol.payload_hash(
        result,
        "summary_fingerprint_sha256",
    )
    return result


def _tamper_repair_summary_runtime(summary: dict[str, Any]) -> dict[str, Any]:
    tampered = dict(summary)
    tampered["gates"] = dict(summary["gates"])
    tampered["gates"]["runtime_seconds"] = dict(
        tampered["gates"]["runtime_seconds"],
        p90=20.0,
        maximum=20.0,
    )
    tampered["summary_fingerprint_sha256"] = protocol.payload_hash(
        tampered,
        "summary_fingerprint_sha256",
    )
    return tampered


def _candidate_payload(
    index: int,
    *,
    input_signal_sha256: str | None = None,
) -> dict[str, Any]:
    input_sha = input_signal_sha256 or _sha(f"input-signal-{index}")
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
    payload: dict[str, Any] = {
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
            "candidate_fingerprint": "0" * 64,
        },
    }
    payload["diagnostics"]["candidate_fingerprint"] = _candidate_fingerprint(payload)
    return payload


def _input_sha(index: int) -> str:
    return protocol.canonical_json_sha256(
        {"row_index": index, "signal": "synthetic"}
    )


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


class _FrameCountSignal:
    def __init__(self, frame_count: int) -> None:
        self.shape = (frame_count,)


def _synthetic_candidate_reference_manifest(
    identities: tuple[Exp007SyntheticIdentity, ...],
) -> dict[str, object]:
    return {
        "schema": "pulsefield_model.timing_v3_exp007_runner_test_manifest_v1",
        "reference_arm": "S30",
        "row_count": len(identities),
        "entries": [
            {
                "row_index": identity.row_index,
                "entry_payload_sha256": _sha(f"candidate:{identity.row_index}"),
            }
            for identity in identities
        ],
    }


def _candidate_reference_manifest_sha(
    identities: tuple[Exp007SyntheticIdentity, ...],
) -> str:
    return protocol.object_complete_sha256(
        _synthetic_candidate_reference_manifest(identities)
    )


def _legacy_entry_list_manifest_sha(
    identities: tuple[Exp007SyntheticIdentity, ...],
) -> str:
    return protocol.canonical_json_sha256(
        [_sha(f"candidate:{identity.row_index}") for identity in identities]
    )


def _schedule_config(
    identities: tuple[Exp007SyntheticIdentity, ...],
    *,
    arm: str = "S30",
    **kwargs: object,
) -> Exp007RunnerConfig:
    del identities
    return Exp007RunnerConfig(
        schedule_arm=arm,
        **kwargs,
    )


def _schedule_stage_summary_factory(
    root: Path,
) -> Callable[..., StageSummaryPublication]:
    def factory(
        config: Exp007RunnerConfig,
        rows: tuple[Exp007SyntheticWorkerResult, ...],
        worker_slot_lifetime_bytes: tuple[int | None, ...],
        adapter: object,
        telemetry: RunnerStageTelemetry,
    ) -> StageSummaryPublication:
        return _stage_summary_factory(
            config,
            rows,
            worker_slot_lifetime_bytes,
            adapter,
            telemetry,
            artifact_root=root,
        )

    return factory


def _stage_summary_factory(
    config: Exp007RunnerConfig,
    rows: tuple[Exp007SyntheticWorkerResult, ...],
    worker_slot_lifetime_bytes: tuple[int | None, ...],
    _adapter: object,
    telemetry: RunnerStageTelemetry,
    *,
    artifact_root: Path,
) -> StageSummaryPublication:
    if config.stage != "schedule16":
        raise Exp007RunnerError("test fixture only builds schedule16 summaries")
    manifest = artifacts.build_candidate_reference_manifest(
        root=artifact_root,
        stage=protocol.EXP007_SCHEDULE_STAGE,
        input_manifest_sha256=config.input_manifest_sha256,
        source_closure_fingerprint_sha256=config.source_closure_fingerprint_sha256,
        reference_arm="S30",
    )
    manifest_relative_path = _write_candidate_reference_manifest(artifact_root, manifest)
    manifest_sha = protocol.object_complete_sha256(manifest)
    exact_rows = [
        protocol.validate_row_result(row.payload["row_result"])
        for row in rows
    ]
    worker_rss = tuple(
        0 if value is None else int(value)
        for value in worker_slot_lifetime_bytes
    )
    evaluation = evaluate_source_arm(
        [_source_metric_row_from_row_result(row) for row in exact_rows],
        schedule_arm=config.schedule_arm,  # type: ignore[arg-type]
        worker_lifetime_rss_bytes=worker_rss,
        aggregate_wall_seconds=telemetry.aggregate_wall_seconds,
    )
    summary = _source_arm_stage_summary_from_evaluation(
        rows=exact_rows,
        evaluation=evaluation,
        candidate_reference_manifest_sha256=manifest_sha,
        candidate_reference_entry_payload_sha256s=[
            row.candidate_reference_entry_payload_sha256
            for row in rows
        ],
    )
    return StageSummaryPublication(
        summary,
        candidate_reference_manifest=manifest,
        candidate_reference_artifact_root=artifact_root,
        candidate_reference_manifest_relative_path=manifest_relative_path,
    )


class _FakeClock:
    def __init__(self) -> None:
        self._ns = 0

    def monotonic_ns(self) -> int:
        return self._ns

    def monotonic(self) -> float:
        return self._ns / 1_000_000_000

    def advance_ns(self, value: int) -> None:
        self._ns += value


class _PostArmDeadlineClock(_FakeClock):
    def __init__(self) -> None:
        super().__init__()
        self._cross_after_reads: int | None = None

    def cross_after_next_deadline_read(self) -> None:
        self._cross_after_reads = 1

    def monotonic_ns(self) -> int:
        if self._cross_after_reads is None:
            return self._ns
        if self._cross_after_reads > 0:
            self._cross_after_reads -= 1
            return self._ns
        return self._ns + 2_000_000_000


def _sleeping_row(identity: Exp007SyntheticIdentity) -> Exp007SyntheticWorkerResult:
    time.sleep(2.0)
    return _row(identity)


class _TimeoutResultIterator:
    def __init__(self) -> None:
        self.timeout = 0.0

    def next(self, *, timeout: float) -> object:
        self.timeout = timeout
        raise Exp007WorkerTimeout("synthetic result timeout")


class _FakeProcess:
    def __init__(self, pid: int, *, alive_after_kill: bool = False) -> None:
        self.pid = pid
        self.exitcode = None
        self._alive_after_kill = alive_after_kill
        self._killed = False

    def join(self, _timeout: float | None = None) -> None:
        return None

    def kill(self) -> None:
        self._killed = True

    def is_alive(self) -> bool:
        return self._alive_after_kill or not self._killed


class _FakePool:
    def __init__(self, pids: tuple[int, ...], *, alive_after_kill: bool = False) -> None:
        self._pool = tuple(_FakeProcess(pid, alive_after_kill=alive_after_kill) for pid in pids)

    def terminate(self) -> None:
        return None
