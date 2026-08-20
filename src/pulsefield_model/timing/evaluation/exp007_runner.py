from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
import platform
import resource
import signal
import socket
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from multiprocessing.connection import Client, Listener, wait
from typing import Any, Callable, Literal, Mapping, Sequence, get_args


EXP007_EXPERIMENT_ID = "timing_v3_experiment_007"
EXP007_PROTOCOL_STAGE = "protocol_freeze"
EXP007_SCHEDULE_STAGE = "schedule16"
EXP007_REPAIR_STAGE = "repair80"
EXP007_SCHEDULE_ARMS = ("S30", "S60", "S90", "S64")
EXP007_EXECUTION_ORDER = ("S30", "S60", "S90", "S64")
EXP007_WORKER_COUNT = 4
EXP007_WORKER_START_METHOD = "spawn"
EXP007_IMAP_CHUNKSIZE = 1
EXP007_MAXTASKSPERCHILD = None
EXP007_PER_AUDIO_ARM_TIMEOUT_S = 180.0
EXP007_SCHEDULE_FOUR_ARM_STOP_S = 1200.0
EXP007_REPAIR_STOP_S = 1800.0
EXP007_WORKER_RSS_CAP_BYTES = 4_294_967_296
EXP007_PARENT_POLL_MAX_SECONDS = 0.25
EXP007_FINISH_RESULT_DELIVERY_S = 5.0
EXP007_WORKER_TERMINATE_GRACE_S = 5.0
EXP007_WORKER_KILL_GRACE_S = 5.0

FailureKind = Literal[
    "row_timeout",
    "row_hard_failure",
    "worker_death",
    "pool_replacement",
    "broken_stream",
    "missing_envelope",
    "duplicate_envelope",
    "arm_deadline",
    "schedule_deadline",
    "identity_mismatch",
    "source_mismatch",
    "cache_mismatch",
    "config_mismatch",
    "restricted_input_mismatch",
    "candidate_mismatch",
    "current_v2_mismatch",
    "schema_failure",
    "rss_failure",
    "diagnostics_integrity_failure",
    "artifact_resource_cap",
    "atomic_publication_failure",
    "summary_publication_failure",
]

FailureStage = Literal[
    "preflight",
    "pool_start",
    "row_source_check",
    "cache_load",
    "restricted_prediction",
    "candidate",
    "current_v2",
    "local_frontier",
    "diagnostics",
    "row_serialization",
    "row_publication",
    "pool_stream",
    "pool_join",
    "arm_summary",
    "repair_summary",
    "schedule_deadline",
]

EXP007_FAILURE_KIND_SET = frozenset(get_args(FailureKind))
EXP007_FAILURE_STAGE_SET = frozenset(get_args(FailureStage))
StageSummaryFactory = Callable[
    [
        "Exp007RunnerConfig",
        Sequence["Exp007SyntheticWorkerResult"],
        Sequence[int | None],
        "Exp007ProtocolAdapter",
        "RunnerStageTelemetry",
    ],
    Mapping[str, Any] | "StageSummaryPublication",
]


class Exp007RunnerError(RuntimeError):
    """Raised when the synthetic Exp007 runner cannot publish a valid outcome."""


class Exp007WorkerTimeout(TimeoutError):
    """Raised inside a worker by the secondary SIGALRM guard."""


class _Exp007StageSummaryTimeout(TimeoutError):
    """Raised in the parent when summary publication crosses the stage deadline."""


class _Exp007ControlEOF(Exp007RunnerError):
    """Raised when a worker control pipe closes before failure attribution is final."""


@dataclass(frozen=True)
class RunnerStageTelemetry:
    worker_lifetime_rss_bytes: tuple[int, int, int, int]
    aggregate_wall_seconds: float

    def __post_init__(self) -> None:
        if len(self.worker_lifetime_rss_bytes) != EXP007_WORKER_COUNT:
            raise ValueError("worker_lifetime_rss_bytes must contain four values")
        for value in self.worker_lifetime_rss_bytes:
            _require_nonnegative_int(value, "worker_lifetime_rss_bytes[]")
        if (
            isinstance(self.aggregate_wall_seconds, bool)
            or not isinstance(self.aggregate_wall_seconds, (int, float))
            or not math.isfinite(float(self.aggregate_wall_seconds))
            or float(self.aggregate_wall_seconds) < 0.0
        ):
            raise ValueError("aggregate_wall_seconds must be finite and nonnegative")


@dataclass(frozen=True)
class RepairSummaryValidationContext:
    repair_run_config: Mapping[str, Any]
    source_closure: Mapping[str, Any]
    source_repo_root: str | os.PathLike[str]
    repair_metric_rows: Sequence[Any]
    repair80_input_binding: Mapping[str, Any]
    repair80_identity_source_artifact: bytes
    repair80_label_source_artifact: bytes
    repair80_identity_rows: Sequence[Mapping[str, Any]]
    repair80_label_rows: Sequence[Mapping[str, Any]]
    schedule_weak_veto_outcome: Mapping[str, Any]
    four_arm_stage_summary: Mapping[str, Any]
    config_selection: Mapping[str, Any]
    candidate_global_manifest: Mapping[str, Any]
    schedule_candidate_reference_manifest: Mapping[str, Any]
    candidate_reference_manifest: Mapping[str, Any]
    artifact_root: str | os.PathLike[str]
    run_configs_by_arm: Mapping[str, Mapping[str, Any]]
    arm_rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]]
    candidate_payloads_by_arm: Mapping[str, Sequence[Mapping[str, Any]]]
    arm_stage_outcomes_by_execution_order: Mapping[str, Mapping[str, Any]]
    source_arm_stage_summaries_by_execution_order: Mapping[str, Mapping[str, Any]]
    prediction_rows: Sequence[Mapping[str, Any]]
    weak_rows: Sequence[Mapping[str, Any]]

    def __post_init__(self) -> None:
        for name in (
            "repair_run_config",
            "source_closure",
            "repair80_input_binding",
            "schedule_weak_veto_outcome",
            "four_arm_stage_summary",
            "config_selection",
            "candidate_global_manifest",
            "schedule_candidate_reference_manifest",
            "candidate_reference_manifest",
            "run_configs_by_arm",
            "arm_rows_by_arm",
            "candidate_payloads_by_arm",
            "arm_stage_outcomes_by_execution_order",
            "source_arm_stage_summaries_by_execution_order",
        ):
            if not isinstance(getattr(self, name), Mapping):
                raise ValueError(f"{name} must be a mapping")
        _require_nonempty_sequence(self.repair_metric_rows, "repair_metric_rows")
        _require_nonempty_sequence(
            self.repair80_identity_rows,
            "repair80_identity_rows",
        )
        _require_nonempty_sequence(self.repair80_label_rows, "repair80_label_rows")
        _require_nonempty_sequence(self.prediction_rows, "prediction_rows")
        _require_nonempty_sequence(self.weak_rows, "weak_rows")
        if not isinstance(self.repair80_identity_source_artifact, bytes):
            raise ValueError("repair80_identity_source_artifact must be bytes")
        if not isinstance(self.repair80_label_source_artifact, bytes):
            raise ValueError("repair80_label_source_artifact must be bytes")
        if not isinstance(self.artifact_root, (str, os.PathLike)):
            raise ValueError("artifact_root must be a path")
        if not isinstance(self.source_repo_root, (str, os.PathLike)):
            raise ValueError("source_repo_root must be a path")


@dataclass(frozen=True)
class StageSummaryPublication:
    payload: Mapping[str, Any]
    repair_summary_validation_context: RepairSummaryValidationContext | None = None
    candidate_reference_manifest: Mapping[str, Any] | None = None
    candidate_reference_artifact_root: str | os.PathLike[str] | None = None
    candidate_reference_manifest_relative_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.payload, Mapping):
            raise ValueError("StageSummaryPublication payload must be a mapping")
        if (
            self.repair_summary_validation_context is not None
            and not isinstance(
                self.repair_summary_validation_context,
                RepairSummaryValidationContext,
            )
        ):
            raise ValueError(
                "repair_summary_validation_context must be RepairSummaryValidationContext"
            )
        if self.candidate_reference_manifest is not None and not isinstance(
            self.candidate_reference_manifest,
            Mapping,
        ):
            raise ValueError("candidate_reference_manifest must be a mapping")
        if self.candidate_reference_artifact_root is not None and not isinstance(
            self.candidate_reference_artifact_root,
            (str, os.PathLike),
        ):
            raise ValueError("candidate_reference_artifact_root must be a path")
        if (
            self.candidate_reference_manifest_relative_path is not None
            and not isinstance(self.candidate_reference_manifest_relative_path, str)
        ):
            raise ValueError("candidate_reference_manifest_relative_path must be a string")


@dataclass(frozen=True)
class Exp007WorkerFailureReport:
    failure_kind: FailureKind
    failure_stage: FailureStage
    diagnostics_classification: Mapping[str, str] | None = None
    exception_type: str | None = None

    def __post_init__(self) -> None:
        _require_failure_kind(self.failure_kind, "failure_kind")
        _require_failure_stage(self.failure_stage, "failure_stage")
        if self.failure_kind == "diagnostics_integrity_failure" and self.failure_stage != "diagnostics":
            raise ValueError("diagnostics_integrity_failure requires diagnostics stage")
        if self.failure_kind == "artifact_resource_cap" and self.failure_stage == "diagnostics":
            raise ValueError("artifact_resource_cap cannot represent diagnostics caps")
        if self.diagnostics_classification is not None:
            diagnostics = dict(self.diagnostics_classification)
            if set(diagnostics) != {"reason", "stage"}:
                raise ValueError("diagnostics_classification must contain reason and stage")
            _require_nonempty_string(diagnostics["reason"], "diagnostics_classification.reason")
            _require_nonempty_string(diagnostics["stage"], "diagnostics_classification.stage")
        if self.exception_type is not None:
            _require_nonempty_string(self.exception_type, "exception_type")


class Exp007WorkerFailure(RuntimeError):
    """Worker-raised hard failure with protocol failure classification."""

    def __init__(
        self,
        message: str,
        *,
        failure_kind: FailureKind,
        failure_stage: FailureStage,
        diagnostics_classification: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        report = Exp007WorkerFailureReport(
            failure_kind=failure_kind,
            failure_stage=failure_stage,
            diagnostics_classification=diagnostics_classification,
            exception_type=type(self).__name__,
        )
        self.failure_kind = report.failure_kind
        self.failure_stage = report.failure_stage
        self.diagnostics_classification = report.diagnostics_classification

    def __reduce__(self) -> tuple[Any, tuple[str, str, str, Mapping[str, str] | None]]:
        return (
            _make_exp007_worker_failure,
            (
                str(self),
                self.failure_kind,
                self.failure_stage,
                self.diagnostics_classification,
            ),
        )


def _make_exp007_worker_failure(
    message: str,
    failure_kind: str,
    failure_stage: str,
    diagnostics_classification: Mapping[str, str] | None,
) -> Exp007WorkerFailure:
    return Exp007WorkerFailure(
        message,
        failure_kind=_require_failure_kind(failure_kind, "failure_kind"),
        failure_stage=_require_failure_stage(failure_stage, "failure_stage"),
        diagnostics_classification=diagnostics_classification,
    )


def _worker_failure_report_from_exception(exc: BaseException) -> Exp007WorkerFailureReport:
    if isinstance(exc, Exp007WorkerTimeout):
        return Exp007WorkerFailureReport(
            failure_kind="row_timeout",
            failure_stage="pool_stream",
            exception_type=type(exc).__name__,
        )
    if isinstance(exc, Exp007WorkerFailure):
        return Exp007WorkerFailureReport(
            failure_kind=exc.failure_kind,
            failure_stage=exc.failure_stage,
            diagnostics_classification=exc.diagnostics_classification,
            exception_type=type(exc).__name__,
        )
    classification = _classify_local_frontier_exception(exc)
    if classification is not None:
        reason = getattr(classification, "reason", None)
        stage = getattr(classification, "stage", None)
        if reason in EXP007_FAILURE_KIND_SET and stage in EXP007_FAILURE_STAGE_SET:
            diagnostics_classification = {
                "reason": str(reason),
                "stage": str(stage),
            }
            return Exp007WorkerFailureReport(
                failure_kind=_require_failure_kind(reason, "failure_kind"),
                failure_stage=_require_failure_stage(stage, "failure_stage"),
                diagnostics_classification=diagnostics_classification,
                exception_type=type(exc).__name__,
            )
    return Exp007WorkerFailureReport(
        failure_kind="row_hard_failure",
        failure_stage="local_frontier",
        exception_type=type(exc).__name__,
    )


def _classify_local_frontier_exception(exc: BaseException) -> Any | None:
    module = importlib.import_module("pulsefield_model.timing.v3.local_frontier")
    return module.classify_local_frontier_exception(exc)


def _worker_failure_report_from_payload(payload: Mapping[str, Any]) -> Exp007WorkerFailureReport:
    diagnostics = payload.get("diagnostics_classification")
    if diagnostics is not None:
        if not isinstance(diagnostics, Mapping):
            raise ValueError("diagnostics_classification must be a mapping")
        diagnostics = {str(key): str(value) for key, value in diagnostics.items()}
    exception_type = payload.get("exception_type")
    return Exp007WorkerFailureReport(
        failure_kind=_require_failure_kind(
            payload.get("failure_kind", "row_hard_failure"),
            "failure_kind",
        ),
        failure_stage=_require_failure_stage(
            payload.get("failure_stage", "local_frontier"),
            "failure_stage",
        ),
        diagnostics_classification=diagnostics,
        exception_type=None if exception_type is None else str(exception_type),
    )


def _worker_failure_payload(
    *,
    identity: Exp007SyntheticIdentity,
    row_ack: RowStartAck,
    elapsed_ns: int,
    rss_bytes: int | None,
    report: Exp007WorkerFailureReport,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "row_failed",
        "row": asdict(identity),
        "slot": row_ack.slot,
        "generation_nonce": row_ack.generation_nonce,
        "pid": row_ack.pid,
        "token": row_ack.token,
        "worker_elapsed_ns": elapsed_ns,
        "worker_rss_bytes": rss_bytes,
        "failure_kind": report.failure_kind,
        "failure_stage": report.failure_stage,
    }
    if report.exception_type is not None:
        payload["exception_type"] = report.exception_type
    if report.diagnostics_classification is not None:
        payload["diagnostics_classification"] = dict(report.diagnostics_classification)
    return payload


@dataclass(frozen=True)
class Exp007SyntheticIdentity:
    row_index: int
    cache_audio_key: str
    identity_payload_sha256: str
    audio_group_key: str | None = None

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.row_index, "row_index")
        _require_nonempty_string(self.cache_audio_key, "cache_audio_key")
        _require_sha256(self.identity_payload_sha256, "identity_payload_sha256")
        if self.audio_group_key is not None:
            _require_nonempty_string(self.audio_group_key, "audio_group_key")

    def pending_ref(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "cache_audio_key": self.cache_audio_key,
            "identity_payload_sha256": self.identity_payload_sha256,
        }


@dataclass(frozen=True)
class Exp007SyntheticWorkerResult:
    row_index: int
    cache_audio_key: str
    identity_payload_sha256: str
    row_payload_sha256: str
    candidate_reference_entry_payload_sha256: str | None = None
    audio_group_key: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    worker_rss_bytes: int | None = None

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.row_index, "row_index")
        _require_nonempty_string(self.cache_audio_key, "cache_audio_key")
        _require_sha256(self.identity_payload_sha256, "identity_payload_sha256")
        _require_sha256(self.row_payload_sha256, "row_payload_sha256")
        if self.candidate_reference_entry_payload_sha256 is not None:
            _require_sha256(
                self.candidate_reference_entry_payload_sha256,
                "candidate_reference_entry_payload_sha256",
            )
        if self.audio_group_key is not None:
            _require_nonempty_string(self.audio_group_key, "audio_group_key")
        if self.worker_rss_bytes is not None:
            _require_nonnegative_int(self.worker_rss_bytes, "worker_rss_bytes")

    def completed_ref(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "cache_audio_key": self.cache_audio_key,
            "identity_payload_sha256": self.identity_payload_sha256,
            "row_payload_sha256": self.row_payload_sha256,
            "candidate_reference_entry_payload_sha256": self.candidate_reference_entry_payload_sha256,
        }


@dataclass(frozen=True)
class Exp007RunnerConfig:
    stage: Literal["schedule16", "repair80"] = EXP007_SCHEDULE_STAGE
    schedule_arm: str = "S30"
    worker_count: int = EXP007_WORKER_COUNT
    start_method: str = EXP007_WORKER_START_METHOD
    imap_chunksize: int = EXP007_IMAP_CHUNKSIZE
    maxtasksperchild: int | None = EXP007_MAXTASKSPERCHILD
    per_audio_arm_timeout_seconds: float = EXP007_PER_AUDIO_ARM_TIMEOUT_S
    schedule_four_arm_stop_seconds: float = EXP007_SCHEDULE_FOUR_ARM_STOP_S
    repair_stop_seconds: float = EXP007_REPAIR_STOP_S
    parent_poll_max_seconds: float = EXP007_PARENT_POLL_MAX_SECONDS
    finish_result_delivery_seconds: float = EXP007_FINISH_RESULT_DELIVERY_S
    worker_terminate_grace_seconds: float = EXP007_WORKER_TERMINATE_GRACE_S
    worker_kill_grace_seconds: float = EXP007_WORKER_KILL_GRACE_S
    worker_rss_cap_bytes: int = EXP007_WORKER_RSS_CAP_BYTES
    run_config_fingerprint_sha256: str = "0" * 64
    source_closure_fingerprint_sha256: str = "1" * 64
    input_manifest_sha256: str = "2" * 64
    schema_descriptor_sha256: str = "3" * 64
    use_spawn_pool: bool = True

    def __post_init__(self) -> None:
        if self.stage not in {EXP007_SCHEDULE_STAGE, EXP007_REPAIR_STAGE}:
            raise ValueError("stage must be schedule16 or repair80")
        if self.schedule_arm not in EXP007_SCHEDULE_ARMS:
            raise ValueError(f"schedule_arm must be one of {EXP007_SCHEDULE_ARMS!r}")
        if self.worker_count != EXP007_WORKER_COUNT:
            raise ValueError("Exp007 runner requires exactly four workers")
        if self.start_method != EXP007_WORKER_START_METHOD:
            raise ValueError("Exp007 runner requires spawn start method")
        if self.imap_chunksize != EXP007_IMAP_CHUNKSIZE:
            raise ValueError("Exp007 runner requires imap chunksize 1")
        if self.maxtasksperchild is not None:
            raise ValueError("Exp007 runner requires persistent workers")
        _require_positive_float(self.per_audio_arm_timeout_seconds, "per_audio_arm_timeout_seconds")
        _require_positive_float(self.schedule_four_arm_stop_seconds, "schedule_four_arm_stop_seconds")
        _require_positive_float(self.repair_stop_seconds, "repair_stop_seconds")
        _require_positive_float(self.parent_poll_max_seconds, "parent_poll_max_seconds")
        _require_positive_float(self.finish_result_delivery_seconds, "finish_result_delivery_seconds")
        _require_positive_float(self.worker_terminate_grace_seconds, "worker_terminate_grace_seconds")
        _require_positive_float(self.worker_kill_grace_seconds, "worker_kill_grace_seconds")
        _require_nonnegative_int(self.worker_rss_cap_bytes, "worker_rss_cap_bytes")
        _require_sha256(self.run_config_fingerprint_sha256, "run_config_fingerprint_sha256")
        _require_sha256(self.source_closure_fingerprint_sha256, "source_closure_fingerprint_sha256")
        _require_sha256(self.input_manifest_sha256, "input_manifest_sha256")
        _require_sha256(self.schema_descriptor_sha256, "schema_descriptor_sha256")

    @property
    def expected_row_count(self) -> int:
        return 16 if self.stage == EXP007_SCHEDULE_STAGE else 80


@dataclass(frozen=True)
class WorkerHello:
    pid: int
    generation_nonce: str

    def __post_init__(self) -> None:
        _require_positive_int(self.pid, "pid")
        _require_sha256(self.generation_nonce, "generation_nonce")


@dataclass(frozen=True)
class WorkerHelloAck:
    pid: int
    generation_nonce: str
    slot: int

    def __post_init__(self) -> None:
        _require_positive_int(self.pid, "pid")
        _require_sha256(self.generation_nonce, "generation_nonce")
        _require_slot(self.slot)


@dataclass(frozen=True)
class RowStartedEvent:
    row: Exp007SyntheticIdentity
    slot: int
    generation_nonce: str
    pid: int
    worker_ready_ns: int

    def __post_init__(self) -> None:
        _require_slot(self.slot)
        _require_sha256(self.generation_nonce, "generation_nonce")
        _require_positive_int(self.pid, "pid")
        _require_nonnegative_int(self.worker_ready_ns, "worker_ready_ns")


@dataclass(frozen=True)
class RowStartAck:
    row: Exp007SyntheticIdentity
    slot: int
    generation_nonce: str
    pid: int
    token: str
    parent_start_ns: int
    deadline_ns: int

    def __post_init__(self) -> None:
        _require_slot(self.slot)
        _require_sha256(self.generation_nonce, "generation_nonce")
        _require_positive_int(self.pid, "pid")
        _require_sha256(self.token, "token")
        _require_nonnegative_int(self.parent_start_ns, "parent_start_ns")
        _require_nonnegative_int(self.deadline_ns, "deadline_ns")
        if self.deadline_ns <= self.parent_start_ns:
            raise ValueError("deadline_ns must be after parent_start_ns")


@dataclass(frozen=True)
class RowFinishedEvent:
    row: Exp007SyntheticIdentity
    slot: int
    generation_nonce: str
    pid: int
    token: str
    worker_elapsed_ns: int
    envelope_sha256: str

    def __post_init__(self) -> None:
        _require_slot(self.slot)
        _require_sha256(self.generation_nonce, "generation_nonce")
        _require_positive_int(self.pid, "pid")
        _require_sha256(self.token, "token")
        _require_nonnegative_int(self.worker_elapsed_ns, "worker_elapsed_ns")
        _require_sha256(self.envelope_sha256, "envelope_sha256")


@dataclass(frozen=True)
class _ActiveRow:
    row: Exp007SyntheticIdentity
    slot: int
    generation_nonce: str
    pid: int
    token: str
    parent_start_ns: int
    deadline_ns: int
    finish_event: RowFinishedEvent | None = None
    envelope: Exp007SyntheticWorkerResult | None = None
    join_guard_start_ns: int | None = None


@dataclass(frozen=True)
class Exp007RunnerOutcome:
    status: Literal["success", "hard_failure", "not_run"]
    rows: tuple[Exp007SyntheticWorkerResult, ...]
    outcome: Mapping[str, Any]
    failure_record: Mapping[str, Any] | None = None
    worker_slot_lifetime_bytes: tuple[int | None, ...] = (None, None, None, None)

    @property
    def ok(self) -> bool:
        return self.status == "success"


class Exp007ScheduleOutcomes(dict[str, Exp007RunnerOutcome]):
    """Schedule-arm outcomes plus the optional final FourArm commit marker."""

    def __init__(
        self,
        *args: Any,
        four_arm_stage_summary: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.four_arm_stage_summary = four_arm_stage_summary


class Exp007ProtocolAdapter:
    """Strict adapter over the frozen Exp007 protocol builders."""

    def __init__(self) -> None:
        try:
            self._module = importlib.import_module("pulsefield_model.timing.evaluation.exp007_protocol")
        except ModuleNotFoundError as exc:
            raise Exp007RunnerError("Exp007 protocol module is required") from exc

    def canonical_json_bytes(self, payload: Any) -> bytes:
        return self._module.canonical_json_bytes(payload)

    def canonical_sha256(self, payload: Any) -> str:
        return self._module.canonical_json_sha256(payload)

    def completed_row_ref(self, result: Exp007SyntheticWorkerResult) -> dict[str, Any]:
        return self._module.make_completed_row_ref(
            row_index=result.row_index,
            cache_audio_key=result.cache_audio_key,
            identity_payload_sha256=result.identity_payload_sha256,
            row_payload_sha256=result.row_payload_sha256,
            candidate_reference_entry_payload_sha256=(
                result.candidate_reference_entry_payload_sha256
            ),
        )

    def pending_identity_ref(self, identity: Exp007SyntheticIdentity) -> dict[str, Any]:
        return self._module.make_pending_identity_ref(
            row_index=identity.row_index,
            cache_audio_key=identity.cache_audio_key,
            identity_payload_sha256=identity.identity_payload_sha256,
        )

    def arm_failure_record(
        self,
        *,
        config: Exp007RunnerConfig,
        identities: Sequence[Exp007SyntheticIdentity],
        completed_rows: Sequence[Exp007SyntheticWorkerResult],
        failure_kind: str,
        failure_stage: str,
        causing_identity: Exp007SyntheticIdentity | None,
        causing_worker_slot: int | None,
        causing_worker_generation_nonce: str | None,
        causing_worker_pid: int | None,
        causing_dispatch_token: str | None,
        causing_worker_rss_bytes: int | None,
        worker_slot_lifetime_bytes: Sequence[int | None],
    ) -> dict[str, Any]:
        completed_refs = [self.completed_row_ref(row) for row in completed_rows]
        completed_indexes = {row.row_index for row in completed_rows}
        pending_refs = [
            self.pending_identity_ref(identity)
            for identity in identities
            if identity.row_index not in completed_indexes
        ]
        is_reference_arm = config.stage == EXP007_REPAIR_STAGE or config.schedule_arm == "S30"
        reference_entry_sha256s = []
        if is_reference_arm:
            for row in completed_rows:
                if row.candidate_reference_entry_payload_sha256 is None:
                    raise Exp007RunnerError("reference arm completed row is missing candidate reference SHA")
                reference_entry_sha256s.append(row.candidate_reference_entry_payload_sha256)
        elif any(row.candidate_reference_entry_payload_sha256 is not None for row in completed_rows):
            raise Exp007RunnerError("non-reference schedule arm completed row carried candidate reference SHA")
        snapshot = _worker_rss_snapshot(worker_slot_lifetime_bytes)
        return self._module.make_arm_failure_record(
            stage=config.stage,
            schedule_arm=config.schedule_arm,
            run_config_fingerprint_sha256=config.run_config_fingerprint_sha256,
            source_closure_fingerprint_sha256=config.source_closure_fingerprint_sha256,
            input_manifest_sha256=config.input_manifest_sha256,
            failure_kind=failure_kind,
            failure_stage=failure_stage,
            completed_prefix_rows=completed_refs,
            pending_identities=pending_refs,
            completed_reference_entry_payload_sha256s=reference_entry_sha256s,
            causing_row_index=None if causing_identity is None else causing_identity.row_index,
            causing_cache_audio_key=None if causing_identity is None else causing_identity.cache_audio_key,
            causing_worker_slot=causing_worker_slot,
            causing_worker_generation_nonce=causing_worker_generation_nonce,
            causing_worker_pid=causing_worker_pid,
            causing_dispatch_token=causing_dispatch_token,
            causing_worker_rss_bytes=causing_worker_rss_bytes,
            prefix_candidate_fallback_count=sum(
                1 for row in completed_rows if row.payload.get("candidate_status") == "tagged_fallback"
            ),
            prefix_no_origin_or_path_count=sum(
                1
                for row in completed_rows
                if row.payload.get("candidate_reason") in {"no_origin_candidate", "no_local_frontier_path"}
            ),
            prefix_resource_cap_fallback_count=sum(
                1
                for row in completed_rows
                if row.payload.get("candidate_reason") == "local_frontier_resource_cap_exceeded"
            ),
            worker_rss_snapshot=snapshot,
        )

    def arm_success(
        self,
        *,
        config: Exp007RunnerConfig,
        rows: Sequence[Exp007SyntheticWorkerResult],
        stage_publication: StageSummaryPublication | None = None,
        telemetry: RunnerStageTelemetry | None = None,
        stage_summary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if stage_publication is None and stage_summary is None:
            raise Exp007RunnerError("arm success requires a complete validated stage summary")
        if stage_publication is None:
            assert stage_summary is not None
            stage_publication = StageSummaryPublication(stage_summary)
        if telemetry is None:
            raise Exp007RunnerError("arm success requires runner-owned stage telemetry")
        if len(rows) != config.expected_row_count:
            raise Exp007RunnerError("arm success row count does not match stage")
        row_refs = [self.completed_row_ref(row) for row in rows]
        candidate_reference_manifest = self._validated_candidate_reference_manifest(
            config,
            stage_publication,
        )
        candidate_reference_manifest_sha256 = self._candidate_reference_manifest_sha256(
            config=config,
            rows=rows,
            manifest=candidate_reference_manifest,
        )
        validated_summary = self.validate_stage_summary(
            config=config,
            payload=stage_publication.payload,
            repair_context=stage_publication.repair_summary_validation_context,
            telemetry=telemetry,
            rows=rows,
            candidate_reference_manifest=candidate_reference_manifest,
        )
        if validated_summary["schedule_arm"] != config.schedule_arm:
            raise Exp007RunnerError("stage summary arm does not match run config")
        if validated_summary["row_refs"] != row_refs:
            raise Exp007RunnerError("stage summary row refs do not match completed rows")
        if validated_summary["row_payloads_sha256"] != self.canonical_sha256(row_refs):
            raise Exp007RunnerError("stage summary row payload hash does not match rows")
        if validated_summary["candidate_reference_manifest_sha256"] != candidate_reference_manifest_sha256:
            raise Exp007RunnerError("stage summary candidate reference manifest hash mismatch")
        if validated_summary["run_config_fingerprint_sha256"] != config.run_config_fingerprint_sha256:
            raise Exp007RunnerError("stage summary run config fingerprint mismatch")
        if validated_summary["source_closure_fingerprint_sha256"] != config.source_closure_fingerprint_sha256:
            raise Exp007RunnerError("stage summary source closure fingerprint mismatch")
        if config.stage == EXP007_REPAIR_STAGE:
            if validated_summary["repair_input_binding_sha256"] != config.input_manifest_sha256:
                raise Exp007RunnerError("repair summary input binding hash mismatch")
        payload = self._module.make_arm_stage_success(
            stage=config.stage,
            schedule_arm=config.schedule_arm,
            row_payloads_sha256=self.canonical_sha256(row_refs),
            candidate_reference_manifest_sha256=candidate_reference_manifest_sha256,
            stage_summary_sha256=self._module.object_complete_sha256(validated_summary),
        )
        return self._module.validate_arm_stage_outcome(payload)

    def validate_stage_summary(
        self,
        *,
        config: Exp007RunnerConfig,
        payload: Mapping[str, Any],
        repair_context: RepairSummaryValidationContext | None = None,
        telemetry: RunnerStageTelemetry | None = None,
        rows: Sequence[Exp007SyntheticWorkerResult] = (),
        candidate_reference_manifest: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if config.stage == EXP007_SCHEDULE_STAGE:
            if repair_context is not None:
                raise Exp007RunnerError("schedule success cannot carry repair summary context")
            exact_rows = _exact_schedule_row_results_from_worker_results(
                rows,
                protocol_module=self._module,
            )
            return self._module.validate_source_arm_stage_summary_authoritatively(
                payload,
                rows=exact_rows,
            )
        if repair_context is None:
            raise Exp007RunnerError(
                "repair success requires RepairSummaryValidationContext"
            )
        if telemetry is None:
            raise Exp007RunnerError("repair success requires runner-owned stage telemetry")
        _validate_repair_context_against_committed_rows(
            context=repair_context,
            rows=rows,
        )
        if candidate_reference_manifest is None:
            raise Exp007RunnerError(
                "repair success requires reopened candidate reference manifest"
            )
        self._validate_repair_terminal_input_chain(
            config=config,
            context=repair_context,
            candidate_reference_manifest=candidate_reference_manifest,
        )
        weak_module = importlib.import_module(
            "pulsefield_model.timing.evaluation.exp007_weak_evidence"
        )
        return weak_module.validate_repair80_summary_authoritatively(
            payload,
            repair_metric_rows=repair_context.repair_metric_rows,
            repair80_input_binding=repair_context.repair80_input_binding,
            repair80_identity_source_artifact=(
                repair_context.repair80_identity_source_artifact
            ),
            repair80_label_source_artifact=(
                repair_context.repair80_label_source_artifact
            ),
            repair80_identity_rows=repair_context.repair80_identity_rows,
            repair80_label_rows=repair_context.repair80_label_rows,
            schedule_weak_veto_outcome=repair_context.schedule_weak_veto_outcome,
            candidate_reference_manifest=candidate_reference_manifest,
            artifact_root=repair_context.artifact_root,
            prediction_rows=repair_context.prediction_rows,
            weak_rows=repair_context.weak_rows,
            aggregate_wall_seconds=telemetry.aggregate_wall_seconds,
            worker_lifetime_rss_bytes=telemetry.worker_lifetime_rss_bytes,
        )

    def arm_hard_failure(
        self,
        *,
        config: Exp007RunnerConfig,
        failure_record: Mapping[str, Any],
    ) -> dict[str, Any]:
        payload = self._module.make_arm_stage_hard_failure(failure_record)
        return self._module.validate_arm_stage_outcome(payload)

    def not_run_arm(
        self,
        *,
        identities: Sequence[Exp007SyntheticIdentity],
        schedule_arm: str,
        reason: str,
        causing_arm: str,
        causing_outcome_sha256: str,
    ) -> dict[str, Any]:
        payload = self._module.make_not_run_arm_record(
            schedule_arm=schedule_arm,
            reason=reason,
            causing_arm=causing_arm,
            causing_outcome_sha256=causing_outcome_sha256,
            pending_identities=[self.pending_identity_ref(identity) for identity in identities],
        )
        return self._module.validate_arm_stage_outcome(payload)

    def validate_arm_stage_outcome(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._module.validate_arm_stage_outcome(payload)

    def _candidate_reference_manifest_sha256(
        self,
        *,
        config: Exp007RunnerConfig,
        rows: Sequence[Exp007SyntheticWorkerResult],
        manifest: Mapping[str, Any],
    ) -> str:
        refs = [
            row.candidate_reference_entry_payload_sha256
            for row in rows
            if row.candidate_reference_entry_payload_sha256 is not None
        ]
        is_reference_arm = config.stage == EXP007_REPAIR_STAGE or config.schedule_arm == "S30"
        if is_reference_arm and len(refs) != len(rows):
            raise Exp007RunnerError("reference arm success is missing candidate reference SHAs")
        if not is_reference_arm and refs:
            raise Exp007RunnerError("non-reference schedule arm success carried candidate references")
        if is_reference_arm:
            entries = manifest.get("entries")
            if not isinstance(entries, Sequence) or len(entries) != len(rows):
                raise Exp007RunnerError("candidate reference manifest entries mismatch")
            manifest_refs = []
            for expected_index, entry in enumerate(entries):
                if not isinstance(entry, Mapping):
                    raise Exp007RunnerError("candidate reference manifest entry is invalid")
                if entry.get("row_index") != expected_index:
                    raise Exp007RunnerError("candidate reference manifest row order mismatch")
                entry_sha = entry.get("entry_payload_sha256")
                _require_sha256(entry_sha, "candidate_reference_manifest.entries[].entry_payload_sha256")
                manifest_refs.append(entry_sha)
            if refs != manifest_refs:
                raise Exp007RunnerError("candidate reference manifest entry SHA mismatch")
        return self._module.object_complete_sha256(manifest)

    def _validated_candidate_reference_manifest(
        self,
        config: Exp007RunnerConfig,
        publication: StageSummaryPublication,
    ) -> Mapping[str, Any]:
        manifest = publication.candidate_reference_manifest
        artifact_root = publication.candidate_reference_artifact_root
        relative_path = publication.candidate_reference_manifest_relative_path
        if manifest is None:
            raise Exp007RunnerError(
                "successful arm publication requires complete candidate reference manifest"
            )
        if artifact_root is None:
            raise Exp007RunnerError(
                "successful arm publication requires candidate reference artifact root"
            )
        if relative_path is None:
            raise Exp007RunnerError(
                "successful arm publication requires candidate reference manifest path"
            )
        artifacts = importlib.import_module(
            "pulsefield_model.timing.evaluation.exp007_artifacts"
        )
        validated = artifacts.read_candidate_reference_manifest(
            artifact_root,
            relative_path,
        )
        if validated != artifacts.validate_candidate_reference_manifest(
            manifest,
            root=artifact_root,
        ):
            raise Exp007RunnerError("candidate reference manifest snapshot mismatch")
        actual_sha = self._module.object_complete_sha256(validated)
        summary_sha = publication.payload.get("candidate_reference_manifest_sha256")
        if actual_sha != summary_sha:
            raise Exp007RunnerError(
                "stage summary candidate reference manifest SHA mismatch"
            )
        if validated["stage"] != config.stage:
            raise Exp007RunnerError("candidate reference manifest stage mismatch")
        if validated["input_manifest_sha256"] != config.input_manifest_sha256:
            raise Exp007RunnerError("candidate reference manifest input mismatch")
        if (
            validated["source_closure_fingerprint_sha256"]
            != config.source_closure_fingerprint_sha256
        ):
            raise Exp007RunnerError("candidate reference manifest source mismatch")
        if config.stage == EXP007_SCHEDULE_STAGE:
            if validated["reference_arm"] != "S30":
                raise Exp007RunnerError(
                    "schedule candidate reference manifest arm mismatch"
                )
        elif validated["reference_arm"] != config.schedule_arm:
            raise Exp007RunnerError("repair candidate reference manifest arm mismatch")
        return validated

    def _validate_repair_terminal_input_chain(
        self,
        *,
        config: Exp007RunnerConfig,
        context: RepairSummaryValidationContext,
        candidate_reference_manifest: Mapping[str, Any],
    ) -> None:
        if context.candidate_reference_manifest != candidate_reference_manifest:
            raise Exp007RunnerError(
                "repair context candidate reference manifest differs from reopened artifact"
            )
        validated_config = self._module.validate_run_config_for_execution(
            context.repair_run_config,
            source_closure=context.source_closure,
            source_repo_root=context.source_repo_root,
            repair80_input_binding=context.repair80_input_binding,
            repair80_identity_source_artifact=(
                context.repair80_identity_source_artifact
            ),
            repair80_label_source_artifact=context.repair80_label_source_artifact,
            repair80_identity_rows=context.repair80_identity_rows,
            repair80_label_rows=context.repair80_label_rows,
            schedule_weak_veto_outcome=context.schedule_weak_veto_outcome,
            four_arm_stage_summary=context.four_arm_stage_summary,
            config_selection=context.config_selection,
            candidate_global_manifest=context.candidate_global_manifest,
            candidate_reference_manifest=context.schedule_candidate_reference_manifest,
            artifact_root=context.artifact_root,
            run_configs_by_arm=context.run_configs_by_arm,
            arm_rows_by_arm=context.arm_rows_by_arm,
            candidate_payloads_by_arm=context.candidate_payloads_by_arm,
            arm_stage_outcomes_by_execution_order=(
                context.arm_stage_outcomes_by_execution_order
            ),
            source_arm_stage_summaries_by_execution_order=(
                context.source_arm_stage_summaries_by_execution_order
            ),
        )
        expected = {
            "stage": config.stage,
            "schedule_arm": config.schedule_arm,
            "run_config_fingerprint_sha256": config.run_config_fingerprint_sha256,
            "source_closure_fingerprint_sha256": (
                config.source_closure_fingerprint_sha256
            ),
            "input_manifest_sha256": config.input_manifest_sha256,
        }
        for field_name, expected_value in expected.items():
            if validated_config[field_name] != expected_value:
                raise Exp007RunnerError(
                    f"repair RunConfig {field_name} does not match runner config"
                )


def _exact_schedule_row_results_from_worker_results(
    rows: Sequence[Exp007SyntheticWorkerResult],
    *,
    protocol_module: Any,
) -> tuple[Mapping[str, Any], ...]:
    exact_rows: list[Mapping[str, Any]] = []
    for row in rows:
        payload_row = row.payload.get("row_result")
        if not isinstance(payload_row, Mapping):
            raise Exp007RunnerError(
                "schedule success requires exact RowResult payloads"
            )
        try:
            validated = protocol_module.validate_row_result(payload_row)
        except ValueError as exc:
            raise Exp007RunnerError(
                "schedule success RowResult payload is invalid"
            ) from exc
        if validated["stage"] != EXP007_SCHEDULE_STAGE:
            raise Exp007RunnerError("schedule success RowResult stage mismatch")
        if (
            validated["row_index"] != row.row_index
            or validated["cache_audio_key"] != row.cache_audio_key
            or validated["identity_payload_sha256"] != row.identity_payload_sha256
            or validated["row_payload_sha256"] != row.row_payload_sha256
        ):
            raise Exp007RunnerError("schedule success RowResult wrapper mismatch")
        if (
            row.audio_group_key is not None
            and validated["audio_group_key"] != row.audio_group_key
        ):
            raise Exp007RunnerError("schedule success RowResult audio group mismatch")
        exact_rows.append(validated)
    return tuple(exact_rows)


class Exp007ArmSupervisor:
    """Pure parent-side watchdog state machine for Exp007 worker events."""

    def __init__(
        self,
        *,
        identities: Sequence[Exp007SyntheticIdentity],
        config: Exp007RunnerConfig,
        initial_pid_order: Sequence[int],
        stage_deadline_ns: int | None = None,
        protocol: Exp007ProtocolAdapter | None = None,
    ) -> None:
        if len(initial_pid_order) != config.worker_count:
            raise ValueError("initial_pid_order must contain exactly four PIDs")
        if len(set(initial_pid_order)) != len(initial_pid_order):
            raise ValueError("initial_pid_order contains duplicate PIDs")
        if stage_deadline_ns is not None:
            _require_nonnegative_int(stage_deadline_ns, "stage_deadline_ns")
        _validate_identity_order(identities, expected_count=None)
        self.identities = tuple(identities)
        self.config = config
        self.stage_deadline_ns = stage_deadline_ns
        self.protocol = protocol or Exp007ProtocolAdapter()
        self._pid_slot = {int(pid): slot for slot, pid in enumerate(initial_pid_order)}
        self._hello_by_slot: dict[int, WorkerHello] = {}
        self._active_by_slot: dict[int, _ActiveRow] = {}
        self._finished_buffer_by_row: dict[int, tuple[RowFinishedEvent, int]] = {}
        self._envelope_buffer_by_row: dict[int, tuple[Exp007SyntheticWorkerResult, int]] = {}
        self._join_guard_start_by_row: dict[int, int] = {}
        self._committed: list[Exp007SyntheticWorkerResult] = []
        self._failure: dict[str, Any] | None = None
        self._failure_record: dict[str, Any] | None = None

    @property
    def completed_rows(self) -> tuple[Exp007SyntheticWorkerResult, ...]:
        return tuple(self._committed)

    @property
    def next_expected_row_index(self) -> int:
        return len(self._committed)

    @property
    def failed(self) -> bool:
        return self._failure is not None

    @property
    def failure(self) -> Mapping[str, Any] | None:
        return self._failure

    @property
    def failure_record(self) -> Mapping[str, Any] | None:
        return self._failure_record

    def accept_hello(self, hello: WorkerHello) -> WorkerHelloAck:
        self._ensure_not_failed()
        slot = self._pid_slot.get(hello.pid)
        if slot is None:
            self._fail(
                failure_kind="broken_stream",
                failure_stage="pool_start",
                causing_identity=None,
                slot=None,
                generation_nonce=hello.generation_nonce,
                pid=hello.pid,
                token=None,
                worker_rss_bytes=None,
            )
            raise Exp007RunnerError("unknown worker HELLO PID")
        if slot in self._hello_by_slot:
            self._fail(
                failure_kind="broken_stream",
                failure_stage="pool_start",
                causing_identity=None,
                slot=slot,
                generation_nonce=hello.generation_nonce,
                pid=hello.pid,
                token=None,
                worker_rss_bytes=None,
            )
            raise Exp007RunnerError("duplicate worker HELLO")
        self._hello_by_slot[slot] = hello
        return WorkerHelloAck(pid=hello.pid, generation_nonce=hello.generation_nonce, slot=slot)

    def assert_handshake_complete(self) -> None:
        self._ensure_not_failed()
        if len(self._hello_by_slot) != self.config.worker_count:
            self._fail(
                failure_kind="broken_stream",
                failure_stage="pool_start",
                causing_identity=None,
                slot=None,
                generation_nonce=None,
                pid=None,
                token=None,
                worker_rss_bytes=None,
            )
            raise Exp007RunnerError("not all workers completed HELLO")

    def start_row(self, event: RowStartedEvent, *, parent_start_ns: int) -> RowStartAck:
        self._ensure_not_failed()
        expected_hello = self._hello_by_slot.get(event.slot)
        if (
            expected_hello is None
            or expected_hello.pid != event.pid
            or expected_hello.generation_nonce != event.generation_nonce
            or event.slot in self._active_by_slot
        ):
            self._fail_event("broken_stream", "pool_stream", event)
            raise Exp007RunnerError("invalid row start event")
        if event.row.row_index < 0 or event.row.row_index >= len(self.identities):
            self._fail_event("identity_mismatch", "pool_stream", event)
            raise Exp007RunnerError("row index outside manifest")
        if self.identities[event.row.row_index] != event.row:
            self._fail_event("identity_mismatch", "pool_stream", event)
            raise Exp007RunnerError("row identity mismatch")
        if (
            event.row.row_index < self.next_expected_row_index
            or event.row.row_index in self._finished_buffer_by_row
            or event.row.row_index in self._envelope_buffer_by_row
        ):
            self._fail_event("duplicate_envelope", "pool_stream", event)
            raise Exp007RunnerError("row already completed or pending commit")
        token = _dispatch_token(event, parent_start_ns=parent_start_ns)
        timeout_ns = int(self.config.per_audio_arm_timeout_seconds * 1_000_000_000)
        ack = RowStartAck(
            row=event.row,
            slot=event.slot,
            generation_nonce=event.generation_nonce,
            pid=event.pid,
            token=token,
            parent_start_ns=parent_start_ns,
            deadline_ns=parent_start_ns + timeout_ns,
        )
        self._active_by_slot[event.slot] = _ActiveRow(
            row=event.row,
            slot=event.slot,
            generation_nonce=event.generation_nonce,
            pid=event.pid,
            token=token,
            parent_start_ns=parent_start_ns,
            deadline_ns=ack.deadline_ns,
        )
        return ack

    def finish_row(self, event: RowFinishedEvent, *, received_ns: int) -> None:
        self._ensure_not_failed()
        active = self._active_by_slot.get(event.slot)
        if active is None or not _finish_matches_active(event, active):
            self._fail_event("broken_stream", "pool_stream", event)
            raise Exp007RunnerError("invalid row finish event")
        if event.row.row_index in self._finished_buffer_by_row:
            self._fail_event("broken_stream", "pool_stream", event)
            raise Exp007RunnerError("duplicate row finish event")
        timeout_ns = int(self.config.per_audio_arm_timeout_seconds * 1_000_000_000)
        if event.worker_elapsed_ns >= timeout_ns:
            self._fail_event("row_timeout", "pool_stream", event)
            raise Exp007RunnerError("row finished at or over timeout")
        self._active_by_slot.pop(event.slot, None)
        self._finished_buffer_by_row[event.row.row_index] = (event, received_ns)
        self._start_join_guard_if_next(event.row.row_index, received_ns)
        self._commit_available_prefix(received_ns)

    def receive_envelope(self, envelope: Exp007SyntheticWorkerResult, *, received_ns: int) -> None:
        self._ensure_not_failed()
        if envelope.row_index < 0 or envelope.row_index >= len(self.identities):
            self._fail(
                failure_kind="identity_mismatch",
                failure_stage="pool_stream",
                causing_identity=None,
                slot=None,
                generation_nonce=None,
                pid=None,
                token=None,
                worker_rss_bytes=envelope.worker_rss_bytes,
            )
            raise Exp007RunnerError("envelope row index outside manifest")
        if envelope.row_index in self._envelope_buffer_by_row or envelope.row_index < self.next_expected_row_index:
            self._fail_row_index(
                failure_kind="duplicate_envelope",
                failure_stage="pool_stream",
                row_index=envelope.row_index,
                worker_rss_bytes=envelope.worker_rss_bytes,
            )
            raise Exp007RunnerError("duplicate envelope")
        identity = self.identities[envelope.row_index]
        if (
            envelope.cache_audio_key != identity.cache_audio_key
            or envelope.identity_payload_sha256 != identity.identity_payload_sha256
            or (
                envelope.audio_group_key is not None
                and envelope.audio_group_key != identity.audio_group_key
            )
        ):
            self._fail_row_index(
                failure_kind="identity_mismatch",
                failure_stage="pool_stream",
                row_index=envelope.row_index,
                worker_rss_bytes=envelope.worker_rss_bytes,
            )
            raise Exp007RunnerError("envelope identity mismatch")
        self._envelope_buffer_by_row[envelope.row_index] = (envelope, received_ns)
        self._start_join_guard_if_next(envelope.row_index, received_ns)
        self._commit_available_prefix(received_ns)

    def check_deadlines(
        self,
        *,
        now_ns: int,
        worker_slot_lifetime_bytes: Sequence[int | None] = (None, None, None, None),
    ) -> None:
        self._ensure_not_failed()
        due: list[tuple[int, int, str, Any]] = []
        if self.stage_deadline_ns is not None and now_ns >= self.stage_deadline_ns:
            due.append((self.stage_deadline_ns, 0, "stage_deadline", None))
        for active in list(self._active_by_slot.values()):
            if active.finish_event is None and now_ns >= active.deadline_ns:
                due.append((active.deadline_ns, 1, "row_timeout", active))
        guard_ns = int(self.config.finish_result_delivery_seconds * 1_000_000_000)
        row_index = self.next_expected_row_index
        guard_start = self._join_guard_start_by_row.get(row_index)
        if guard_start is not None and now_ns - guard_start >= guard_ns:
            due.append((guard_start + guard_ns, 2, "join_guard", row_index))
        if not due:
            return
        _, _, deadline_type, payload = min(due, key=lambda item: (item[0], item[1]))
        if deadline_type == "stage_deadline":
            failure_kind, failure_stage = _stage_deadline_failure(self.config)
            if (
                self.config.stage == EXP007_REPAIR_STAGE
                and len(self._committed) == len(self.identities)
            ):
                failure_stage = "repair_summary"
            self._fail(
                failure_kind=failure_kind,
                failure_stage=failure_stage,
                causing_identity=None,
                slot=None,
                generation_nonce=None,
                pid=None,
                token=None,
                worker_rss_bytes=None,
                worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
            )
            raise Exp007RunnerError(f"{failure_kind} crossed")
        if deadline_type == "row_timeout":
            active = payload
            self._fail(
                failure_kind="row_timeout",
                failure_stage="pool_stream",
                causing_identity=active.row,
                slot=active.slot,
                generation_nonce=active.generation_nonce,
                pid=active.pid,
                token=active.token,
                worker_rss_bytes=None,
                worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
            )
            raise Exp007RunnerError("row deadline crossed")
        if deadline_type == "join_guard":
            self._fail_row_index(
                failure_kind="broken_stream",
                failure_stage="pool_stream",
                row_index=int(payload),
                worker_rss_bytes=None,
                worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
            )
            raise Exp007RunnerError("finish/envelope join guard crossed")

    def complete_success(
        self,
        *,
        stage_publication: StageSummaryPublication,
        telemetry: RunnerStageTelemetry,
        worker_slot_lifetime_bytes: Sequence[int | None] = (None, None, None, None),
    ) -> Exp007RunnerOutcome:
        self._ensure_not_failed()
        if len(self._committed) != len(self.identities):
            raise Exp007RunnerError("cannot complete success before all rows commit")
        success = self.protocol.arm_success(
            config=self.config,
            rows=self._committed,
            stage_publication=stage_publication,
            telemetry=telemetry,
        )
        return Exp007RunnerOutcome(
            status="success",
            rows=tuple(self._committed),
            outcome=success,
            failure_record=None,
            worker_slot_lifetime_bytes=tuple(worker_slot_lifetime_bytes),
        )

    def hard_failure_outcome(
        self,
        *,
        worker_slot_lifetime_bytes: Sequence[int | None] = (None, None, None, None),
    ) -> Exp007RunnerOutcome:
        if self._failure_record is None:
            self._fail(
                failure_kind="broken_stream",
                failure_stage="pool_stream",
                causing_identity=None,
                slot=None,
                generation_nonce=None,
                pid=None,
                token=None,
                worker_rss_bytes=None,
                worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
            )
        assert self._failure_record is not None
        outcome = self.protocol.arm_hard_failure(config=self.config, failure_record=self._failure_record)
        return Exp007RunnerOutcome(
            status="hard_failure",
            rows=tuple(self._committed),
            outcome=outcome,
            failure_record=self._failure_record,
            worker_slot_lifetime_bytes=tuple(worker_slot_lifetime_bytes),
        )

    def fail_worker_death(
        self,
        *,
        pid: int,
        failure_stage: str = "pool_stream",
        worker_slot_lifetime_bytes: Sequence[int | None] = (None, None, None, None),
    ) -> Exp007RunnerOutcome:
        slot = self._pid_slot.get(pid)
        active = self._active_by_slot.get(slot) if slot is not None else None
        self._fail(
            failure_kind="worker_death",
            failure_stage=failure_stage,
            causing_identity=None if active is None else active.row,
            slot=slot,
            generation_nonce=None if active is None else active.generation_nonce,
            pid=pid,
            token=None if active is None else active.token,
            worker_rss_bytes=None,
            worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
        )
        return self.hard_failure_outcome(worker_slot_lifetime_bytes=worker_slot_lifetime_bytes)

    def fail_pool_replacement(
        self,
        *,
        pid: int,
        failure_stage: str = "pool_stream",
        worker_slot_lifetime_bytes: Sequence[int | None] = (None, None, None, None),
    ) -> Exp007RunnerOutcome:
        self._fail(
            failure_kind="pool_replacement",
            failure_stage=failure_stage,
            causing_identity=None,
            slot=None,
            generation_nonce=None,
            pid=None if pid <= 0 else pid,
            token=None,
            worker_rss_bytes=None,
            worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
        )
        return self.hard_failure_outcome(worker_slot_lifetime_bytes=worker_slot_lifetime_bytes)

    def _commit_available_prefix(self, now_ns: int) -> None:
        while True:
            row_index = self.next_expected_row_index
            finished = self._finished_buffer_by_row.get(row_index)
            envelope = self._envelope_buffer_by_row.get(row_index)
            if finished is None or envelope is None:
                return
            finish, _finish_received_ns = finished
            row_result, _envelope_received_ns = envelope
            if finish.envelope_sha256 != canonical_sha256(row_result):
                self._fail_event("broken_stream", "pool_stream", finish)
                raise Exp007RunnerError("finish/envelope SHA mismatch")
            self._committed.append(row_result)
            del self._finished_buffer_by_row[row_index]
            del self._envelope_buffer_by_row[row_index]
            self._join_guard_start_by_row.pop(row_index, None)
            next_row = self.next_expected_row_index
            if next_row in self._finished_buffer_by_row or next_row in self._envelope_buffer_by_row:
                self._start_join_guard_if_next(next_row, now_ns)

    def _start_join_guard_if_next(self, row_index: int, now_ns: int) -> None:
        if row_index == self.next_expected_row_index and row_index not in self._join_guard_start_by_row:
            has_finish = row_index in self._finished_buffer_by_row
            has_envelope = row_index in self._envelope_buffer_by_row
            if has_finish != has_envelope:
                self._join_guard_start_by_row[row_index] = now_ns

    def _fail_event(self, failure_kind: str, failure_stage: str, event: Any) -> None:
        identity = getattr(event, "row", None)
        token = getattr(event, "token", None)
        self._fail(
            failure_kind=failure_kind,
            failure_stage=failure_stage,
            causing_identity=(
                identity
                if isinstance(identity, Exp007SyntheticIdentity) and token is not None
                else None
            ),
            slot=getattr(event, "slot", None),
            generation_nonce=getattr(event, "generation_nonce", None),
            pid=getattr(event, "pid", None),
            token=token,
            worker_rss_bytes=None,
        )

    def _fail_row_index(
        self,
        *,
        failure_kind: str,
        failure_stage: str,
        row_index: int,
        worker_rss_bytes: int | None,
        worker_slot_lifetime_bytes: Sequence[int | None] = (None, None, None, None),
    ) -> None:
        active = next(
            (row for row in self._active_by_slot.values() if row.row.row_index == row_index),
            None,
        )
        if active is not None:
            self._fail(
                failure_kind=failure_kind,
                failure_stage=failure_stage,
                causing_identity=active.row,
                slot=active.slot,
                generation_nonce=active.generation_nonce,
                pid=active.pid,
                token=active.token,
                worker_rss_bytes=worker_rss_bytes,
                worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
            )
            return
        finished = self._finished_buffer_by_row.get(row_index)
        if finished is not None:
            finish = finished[0]
            self._fail(
                failure_kind=failure_kind,
                failure_stage=failure_stage,
                causing_identity=finish.row,
                slot=finish.slot,
                generation_nonce=finish.generation_nonce,
                pid=finish.pid,
                token=finish.token,
                worker_rss_bytes=worker_rss_bytes,
                worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
            )
            return
        self._fail(
            failure_kind=failure_kind,
            failure_stage=failure_stage,
            causing_identity=None,
            slot=None,
            generation_nonce=None,
            pid=None,
            token=None,
            worker_rss_bytes=None,
            worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
        )

    def _fail(
        self,
        *,
        failure_kind: str,
        failure_stage: str,
        causing_identity: Exp007SyntheticIdentity | None,
        slot: int | None,
        generation_nonce: str | None,
        pid: int | None,
        token: str | None,
        worker_rss_bytes: int | None,
        worker_slot_lifetime_bytes: Sequence[int | None] = (None, None, None, None),
    ) -> None:
        if self._failure is not None:
            return
        record_identity = causing_identity
        record_slot = slot if record_identity is not None else None
        record_generation_nonce = generation_nonce if record_identity is not None else None
        record_pid = pid if record_identity is not None else None
        record_token = token if record_identity is not None else None
        record_rss = worker_rss_bytes if record_identity is not None else None
        failure_record = self.protocol.arm_failure_record(
            config=self.config,
            identities=self.identities,
            completed_rows=self._committed,
            failure_kind=failure_kind,
            failure_stage=failure_stage,
            causing_identity=record_identity,
            causing_worker_slot=record_slot,
            causing_worker_generation_nonce=record_generation_nonce,
            causing_worker_pid=record_pid,
            causing_dispatch_token=record_token,
            causing_worker_rss_bytes=record_rss,
            worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
        )
        self._failure = {
            "failure_kind": failure_kind,
            "failure_stage": failure_stage,
            "causing_row_index": None if record_identity is None else record_identity.row_index,
        }
        self._failure_record = failure_record

    def _ensure_not_failed(self) -> None:
        if self._failure is not None:
            raise Exp007RunnerError(f"supervisor already failed: {self._failure!r}")


def run_synthetic_exp007_arm(
    identities: Sequence[Exp007SyntheticIdentity],
    row_callback: Callable[[Exp007SyntheticIdentity], Exp007SyntheticWorkerResult],
    *,
    config: Exp007RunnerConfig | None = None,
    protocol: Exp007ProtocolAdapter | None = None,
    stage_deadline_monotonic_ns: int | None = None,
    stage_summary_factory: StageSummaryFactory | None = None,
) -> Exp007RunnerOutcome:
    """Run one synthetic arm through the Exp007 parent/worker control protocol."""

    stage_start_ns = time.monotonic_ns()
    resolved_config = config or Exp007RunnerConfig()
    protocol = protocol or Exp007ProtocolAdapter()
    stage_deadline_ns = _resolve_stage_deadline_ns(
        resolved_config,
        stage_deadline_monotonic_ns=stage_deadline_monotonic_ns,
    )
    _validate_identity_order(identities, expected_count=resolved_config.expected_row_count)
    if not identities:
        raise ValueError("identities must be non-empty")
    if not resolved_config.use_spawn_pool:
        return _run_synthetic_exp007_arm_inline(
            identities,
            row_callback,
            config=resolved_config,
            protocol=protocol,
            stage_deadline_ns=stage_deadline_ns,
            stage_summary_factory=stage_summary_factory,
            stage_start_ns=stage_start_ns,
        )
    _preflight_platform()

    ctx = mp.get_context(resolved_config.start_method)
    listener = Listener(("127.0.0.1", 0), family="AF_INET")
    listener_address = listener.address
    worker_slot_lifetime_bytes: list[int | None] = [None] * resolved_config.worker_count
    pool: mp.pool.Pool | None = None
    control_by_slot: dict[int, Any] = {}
    supervisor: Exp007ArmSupervisor | None = None
    result_iter: Any = None

    try:
        pool = ctx.Pool(
            processes=resolved_config.worker_count,
            initializer=_synthetic_worker_initializer,
            initargs=(listener_address, _worker_initializer_config(resolved_config)),
            maxtasksperchild=resolved_config.maxtasksperchild,
        )
        initial_processes = tuple(pool._pool)  # noqa: SLF001 - source-closed by the Exp007 card.
        initial_pid_order = tuple(process.pid for process in initial_processes)
        if any(pid is None for pid in initial_pid_order):
            raise Exp007RunnerError("pool process PID was unavailable")
        supervisor = Exp007ArmSupervisor(
            identities=identities,
            config=resolved_config,
            initial_pid_order=tuple(int(pid) for pid in initial_pid_order),
            stage_deadline_ns=stage_deadline_ns,
            protocol=protocol,
        )

        control_by_slot = _accept_initial_handshake(
            listener=listener,
            pool=pool,
            supervisor=supervisor,
            config=resolved_config,
            worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
        )

        tasks = tuple((identity, row_callback) for identity in identities)
        result_iter = pool.imap(_synthetic_worker_run_row, tasks, chunksize=resolved_config.imap_chunksize)
        expected_envelope_index = 0
        finished = False
        while not finished:
            try:
                _drain_control_connections(
                    control_by_slot.values(),
                    supervisor=supervisor,
                    worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
                )
                timeout = _supervisor_wait_timeout(supervisor, resolved_config)
                readable = wait(tuple(control_by_slot.values()), timeout=timeout)
                now_ns = time.monotonic_ns()
                for conn in readable:
                    _process_control_payload(
                        conn,
                        supervisor=supervisor,
                        worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
                        received_ns=now_ns,
                    )
                _drain_control_connections(
                    control_by_slot.values(),
                    supervisor=supervisor,
                    worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
                )
                _detect_worker_death(
                    pool,
                    supervisor,
                    worker_slot_lifetime_bytes,
                    before_failure=lambda: _drain_control_connections_before_worker_death(
                        control_by_slot.values(),
                        supervisor=supervisor,
                        worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
                    ),
                )

                expected_envelope_index = _drain_result_envelopes(
                    result_iter,
                    supervisor=supervisor,
                    identities=identities,
                    expected_envelope_index=expected_envelope_index,
                    worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
                )

                _drain_control_connections(
                    control_by_slot.values(),
                    supervisor=supervisor,
                    worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
                )
                decision_ns = time.monotonic_ns()
                if _has_due_deadline(supervisor, now_ns=decision_ns):
                    _drain_control_connections(
                        control_by_slot.values(),
                        supervisor=supervisor,
                        worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
                    )
                    expected_envelope_index = _drain_result_envelopes(
                        result_iter,
                        supervisor=supervisor,
                        identities=identities,
                        expected_envelope_index=expected_envelope_index,
                        worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
                    )
                    decision_ns = time.monotonic_ns()
                supervisor.check_deadlines(
                    now_ns=decision_ns,
                    worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
                )
                finished = len(supervisor.completed_rows) == len(identities)
            except _Exp007ControlEOF as exc:
                expected_envelope_index = _handle_control_eof(
                    result_iter,
                    supervisor=supervisor,
                    identities=identities,
                    expected_envelope_index=expected_envelope_index,
                    worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
                    cause=exc,
                )

        supervisor.check_deadlines(
            now_ns=time.monotonic_ns(),
            worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
        )
        _validate_rss_success(worker_slot_lifetime_bytes, resolved_config)
        pool.close()
        pool.join()
        return _complete_success_or_summary_failure(
            supervisor=supervisor,
            config=resolved_config,
            worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
            protocol=protocol,
            stage_summary_factory=stage_summary_factory,
            stage_start_ns=stage_start_ns,
        )
    except Exception:
        if supervisor is not None and not supervisor.failed:
            supervisor._fail(  # noqa: SLF001
                failure_kind="broken_stream",
                failure_stage="pool_stream",
                causing_identity=None,
                slot=None,
                generation_nonce=None,
                pid=None,
                token=None,
                worker_rss_bytes=None,
                worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
            )
        if pool is not None:
            _close_connections(tuple(control_by_slot.values()))
            control_by_slot.clear()
            _terminate_pool_bounded(pool, resolved_config)
        if supervisor is None:
            raise
        return supervisor.hard_failure_outcome(worker_slot_lifetime_bytes=worker_slot_lifetime_bytes)
    finally:
        _close_connections(tuple(control_by_slot.values()))
        listener.close()


def run_synthetic_exp007_schedule(
    identities: Sequence[Exp007SyntheticIdentity],
    row_callback: Callable[[str, Exp007SyntheticIdentity], Exp007SyntheticWorkerResult],
    *,
    config_template: Exp007RunnerConfig | None = None,
    protocol: Exp007ProtocolAdapter | None = None,
    stage_summary_factory: StageSummaryFactory | None = None,
) -> Exp007ScheduleOutcomes:
    """Run the four Exp007 schedule arms in frozen order, stopping after failure."""

    template = config_template or Exp007RunnerConfig(stage=EXP007_SCHEDULE_STAGE)
    _validate_identity_order(identities, expected_count=16)
    outcomes = Exp007ScheduleOutcomes()
    prior_failure: tuple[str, str, str] | None = None
    deadline_ns = time.monotonic_ns() + int(template.schedule_four_arm_stop_seconds * 1_000_000_000)
    protocol = protocol or Exp007ProtocolAdapter()
    for arm in EXP007_EXECUTION_ORDER:
        if prior_failure is not None:
            outcomes[arm] = _not_run_due_prior_failure(
                identities,
                config=dataclasses.replace(template, schedule_arm=arm),
                causing_arm=prior_failure[0],
                causing_outcome_sha256=prior_failure[1],
                reason=prior_failure[2],
                protocol=protocol,
            )
            continue
        if time.monotonic_ns() >= deadline_ns:
            config = dataclasses.replace(template, stage=EXP007_SCHEDULE_STAGE, schedule_arm=arm)
            outcome = _schedule_deadline_outcome(
                identities,
                completed_rows=(),
                config=config,
                protocol=protocol,
            )
            outcomes[arm] = outcome
            prior_failure = (
                arm,
                protocol.canonical_sha256(outcome.outcome),
                "schedule_deadline_already_crossed",
            )
            continue
        config = dataclasses.replace(template, stage=EXP007_SCHEDULE_STAGE, schedule_arm=arm)
        outcome = run_synthetic_exp007_arm(
            identities,
            functools.partial(_call_schedule_row_callback, row_callback, arm),
            config=config,
            protocol=protocol,
            stage_deadline_monotonic_ns=deadline_ns,
            stage_summary_factory=stage_summary_factory,
        )
        if arm != EXP007_EXECUTION_ORDER[-1] and outcome.ok and time.monotonic_ns() >= deadline_ns:
            outcome = _schedule_deadline_outcome(
                identities,
                completed_rows=outcome.rows,
                config=config,
                protocol=protocol,
            )
        outcomes[arm] = outcome
        if not outcome.ok:
            reason = (
                "schedule_deadline_already_crossed"
                if outcome.failure_record is not None
                and outcome.failure_record.get("failure_kind") == "schedule_deadline"
                else "prior_arm_hard_failure"
            )
            prior_failure = (arm, protocol.canonical_sha256(outcome.outcome), reason)
    if (
        prior_failure is None
        and len(outcomes) == len(EXP007_EXECUTION_ORDER)
        and all(outcomes[arm].ok for arm in EXP007_EXECUTION_ORDER)
        and time.monotonic_ns() >= deadline_ns
    ):
        outcomes.four_arm_stage_summary = _post_arm_schedule_deadline_four_arm_summary(
            outcomes,
            protocol=protocol,
        )
    return outcomes


def _call_schedule_row_callback(
    row_callback: Callable[[str, Exp007SyntheticIdentity], Exp007SyntheticWorkerResult],
    arm: str,
    identity: Exp007SyntheticIdentity,
) -> Exp007SyntheticWorkerResult:
    return row_callback(arm, identity)


def _accept_initial_handshake(
    *,
    listener: Listener,
    pool: Any,
    supervisor: Exp007ArmSupervisor,
    config: Exp007RunnerConfig,
    worker_slot_lifetime_bytes: Sequence[int | None],
) -> dict[int, Any]:
    control_by_slot: dict[int, Any] = {}
    handshake_deadline_ns = time.monotonic_ns() + int(config.per_audio_arm_timeout_seconds * 1_000_000_000)
    listener_socket = _listener_socket(listener)
    previous_timeout = listener_socket.gettimeout()
    try:
        while len(control_by_slot) < config.worker_count:
            now_ns = time.monotonic_ns()
            supervisor.check_deadlines(
                now_ns=now_ns,
                worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
            )
            _detect_worker_death(
                pool,
                supervisor,
                worker_slot_lifetime_bytes,
                failure_stage="pool_start",
            )
            remaining = _seconds_until_deadline_ns(
                min(_active_deadline_values(handshake_deadline_ns, supervisor.stage_deadline_ns)),
                now_ns=now_ns,
            )
            if remaining <= 0.0:
                supervisor._fail(  # noqa: SLF001
                    failure_kind="broken_stream",
                    failure_stage="pool_start",
                    causing_identity=None,
                    slot=None,
                    generation_nonce=None,
                    pid=None,
                    token=None,
                    worker_rss_bytes=None,
                    worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
                )
                raise Exp007RunnerError("worker HELLO handshake deadline crossed")
            listener_socket.settimeout(min(config.parent_poll_max_seconds, remaining))
            try:
                conn = listener.accept()
            except socket.timeout:
                continue
            try:
                ready = wait(
                    (conn,),
                    timeout=_seconds_until_deadline_ns(
                        min(_active_deadline_values(handshake_deadline_ns, supervisor.stage_deadline_ns))
                    ),
                )
                if not ready:
                    supervisor.check_deadlines(
                        now_ns=time.monotonic_ns(),
                        worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
                    )
                    raise Exp007RunnerError("worker HELLO payload deadline crossed")
                payload = conn.recv()
                if not isinstance(payload, Mapping):
                    raise Exp007RunnerError("worker HELLO payload was not a mapping")
                hello = WorkerHello(**payload)
                ack = supervisor.accept_hello(hello)
                conn.send(asdict(ack))
                ready = wait(
                    (conn,),
                    timeout=_seconds_until_deadline_ns(
                        min(_active_deadline_values(handshake_deadline_ns, supervisor.stage_deadline_ns))
                    ),
                )
                if not ready:
                    supervisor.check_deadlines(
                        now_ns=time.monotonic_ns(),
                        worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
                    )
                    raise Exp007RunnerError("worker HELLO ACK confirmation deadline crossed")
                confirmation = conn.recv()
                if not isinstance(confirmation, Mapping) or confirmation.get("slot") != ack.slot:
                    raise Exp007RunnerError("worker HELLO ACK confirmation is invalid")
                if ack.slot in control_by_slot:
                    raise Exp007RunnerError("duplicate worker control slot")
                control_by_slot[ack.slot] = conn
            except Exception:
                try:
                    conn.close()
                except OSError:
                    pass
                if not supervisor.failed:
                    supervisor._fail(  # noqa: SLF001
                        failure_kind="broken_stream",
                        failure_stage="pool_start",
                        causing_identity=None,
                        slot=None,
                        generation_nonce=None,
                        pid=None,
                        token=None,
                        worker_rss_bytes=None,
                        worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
                    )
                raise
        supervisor.assert_handshake_complete()
        _reject_immediate_extra_hello(listener, supervisor=supervisor)
        return control_by_slot
    finally:
        listener_socket.settimeout(previous_timeout)


def _listener_socket(listener: Listener) -> socket.socket:
    try:
        return listener._listener._socket  # noqa: SLF001
    except AttributeError as exc:
        raise Exp007RunnerError("unsupported Listener implementation") from exc


def _reject_immediate_extra_hello(
    listener: Listener,
    *,
    supervisor: Exp007ArmSupervisor,
) -> None:
    listener_socket = _listener_socket(listener)
    previous_timeout = listener_socket.gettimeout()
    listener_socket.settimeout(0.0)
    try:
        try:
            conn = listener.accept()
        except (socket.timeout, BlockingIOError):
            return
        try:
            supervisor._fail(  # noqa: SLF001
                failure_kind="broken_stream",
                failure_stage="pool_start",
                causing_identity=None,
                slot=None,
                generation_nonce=None,
                pid=None,
                token=None,
                worker_rss_bytes=None,
            )
            raise Exp007RunnerError("extra worker HELLO connection")
        finally:
            conn.close()
    finally:
        listener_socket.settimeout(previous_timeout)


def _process_control_payload(
    conn: Any,
    *,
    supervisor: Exp007ArmSupervisor,
    worker_slot_lifetime_bytes: list[int | None],
    received_ns: int,
) -> None:
    try:
        payload = conn.recv()
    except (EOFError, OSError) as exc:
        raise _Exp007ControlEOF("worker control connection closed") from exc
    _handle_control_payload(
        conn,
        payload,
        supervisor=supervisor,
        worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
        received_ns=received_ns,
    )


def _handle_control_payload(
    conn: Any,
    payload: Any,
    *,
    supervisor: Exp007ArmSupervisor,
    worker_slot_lifetime_bytes: list[int | None],
    received_ns: int,
) -> None:
    if not isinstance(payload, Mapping):
        supervisor._fail(  # noqa: SLF001
            failure_kind="broken_stream",
            failure_stage="pool_stream",
            causing_identity=None,
            slot=None,
            generation_nonce=None,
            pid=None,
            token=None,
            worker_rss_bytes=None,
            worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
        )
        raise Exp007RunnerError("worker control payload was not a mapping")
    event_type = payload.get("type")
    if event_type == "row_started":
        event = RowStartedEvent(
            row=Exp007SyntheticIdentity(**payload["row"]),
            slot=payload["slot"],
            generation_nonce=payload["generation_nonce"],
            pid=payload["pid"],
            worker_ready_ns=payload["worker_ready_ns"],
        )
        ack = supervisor.start_row(event, parent_start_ns=received_ns)
        conn.send({"type": "row_start_ack", "ack": asdict(ack)})
        return
    if event_type == "row_finished":
        event = RowFinishedEvent(
            row=Exp007SyntheticIdentity(**payload["row"]),
            slot=payload["slot"],
            generation_nonce=payload["generation_nonce"],
            pid=payload["pid"],
            token=payload["token"],
            worker_elapsed_ns=payload["worker_elapsed_ns"],
            envelope_sha256=payload["envelope_sha256"],
        )
        worker_slot_lifetime_bytes[event.slot] = max(
            worker_slot_lifetime_bytes[event.slot] or 0,
            int(payload.get("worker_rss_bytes") or 0),
        )
        supervisor.finish_row(event, received_ns=received_ns)
        return
    if event_type == "row_failed":
        identity = Exp007SyntheticIdentity(**payload["row"])
        slot = int(payload["slot"])
        worker_slot_lifetime_bytes[slot] = max(
            worker_slot_lifetime_bytes[slot] or 0,
            int(payload.get("worker_rss_bytes") or 0),
        )
        try:
            report = _worker_failure_report_from_payload(payload)
        except ValueError as exc:
            supervisor._fail(  # noqa: SLF001
                failure_kind="schema_failure",
                failure_stage="pool_stream",
                causing_identity=identity,
                slot=slot,
                generation_nonce=str(payload["generation_nonce"]),
                pid=int(payload["pid"]),
                token=str(payload["token"]),
                worker_rss_bytes=payload.get("worker_rss_bytes"),
                worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
            )
            raise Exp007RunnerError("worker failure report was invalid") from exc
        supervisor._fail(  # noqa: SLF001
            failure_kind=report.failure_kind,
            failure_stage=report.failure_stage,
            causing_identity=identity,
            slot=slot,
            generation_nonce=str(payload["generation_nonce"]),
            pid=int(payload["pid"]),
            token=str(payload["token"]),
            worker_rss_bytes=payload.get("worker_rss_bytes"),
            worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
        )
        raise Exp007RunnerError(f"worker reported {report.failure_kind}")
    supervisor._fail(  # noqa: SLF001
        failure_kind="broken_stream",
        failure_stage="pool_stream",
        causing_identity=None,
        slot=None,
        generation_nonce=None,
        pid=None,
        token=None,
        worker_rss_bytes=None,
        worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
    )
    raise Exp007RunnerError(f"unknown control event type {event_type!r}")


def _drain_control_connections(
    connections: Sequence[Any],
    *,
    supervisor: Exp007ArmSupervisor,
    worker_slot_lifetime_bytes: list[int | None],
) -> None:
    while True:
        readable = wait(tuple(connections), timeout=0.0)
        if not readable:
            return
        now_ns = time.monotonic_ns()
        for conn in readable:
            _process_control_payload(
                conn,
                supervisor=supervisor,
                worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
                received_ns=now_ns,
            )


def _drain_control_connections_before_worker_death(
    connections: Sequence[Any],
    *,
    supervisor: Exp007ArmSupervisor,
    worker_slot_lifetime_bytes: list[int | None],
) -> None:
    if not supervisor._active_by_slot:  # noqa: SLF001
        return
    deadline_ns = time.monotonic_ns() + int(
        supervisor.config.parent_poll_max_seconds * 1_000_000_000
    )
    while True:
        now_ns = time.monotonic_ns()
        remaining = max(0.0, (deadline_ns - now_ns) / 1_000_000_000)
        readable = wait(tuple(connections), timeout=remaining)
        if not readable:
            return
        received_ns = time.monotonic_ns()
        for conn in readable:
            try:
                payload = conn.recv()
            except (EOFError, OSError):
                continue
            _handle_control_payload(
                conn,
                payload,
                supervisor=supervisor,
                worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
                received_ns=received_ns,
            )
        if time.monotonic_ns() >= deadline_ns:
            return


def _handle_control_eof(
    result_iter: Any,
    *,
    supervisor: Exp007ArmSupervisor,
    identities: Sequence[Exp007SyntheticIdentity],
    expected_envelope_index: int,
    worker_slot_lifetime_bytes: list[int | None],
    cause: BaseException,
) -> int:
    if result_iter is not None and expected_envelope_index < len(identities):
        envelope_payload = _next_result_payload(
            result_iter,
            supervisor=supervisor,
            worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
            timeout=_control_eof_result_grace_seconds(supervisor),
            row_index=expected_envelope_index,
        )
        if envelope_payload is not None and expected_envelope_index < len(identities):
            envelope = Exp007SyntheticWorkerResult(**envelope_payload)
            if envelope.worker_rss_bytes is not None:
                active_slot = _slot_for_identity(supervisor, envelope.row_index)
                if active_slot is not None:
                    worker_slot_lifetime_bytes[active_slot] = max(
                        worker_slot_lifetime_bytes[active_slot] or 0,
                        envelope.worker_rss_bytes,
                    )
            supervisor.receive_envelope(envelope, received_ns=time.monotonic_ns())
            expected_envelope_index += 1
    if not supervisor.failed:
        try:
            supervisor.check_deadlines(
                now_ns=time.monotonic_ns(),
                worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
            )
        except Exp007RunnerError:
            raise
    if not supervisor.failed:
        supervisor._fail(  # noqa: SLF001
            failure_kind="broken_stream",
            failure_stage="pool_stream",
            causing_identity=None,
            slot=None,
            generation_nonce=None,
            pid=None,
            token=None,
            worker_rss_bytes=None,
            worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
        )
    raise Exp007RunnerError("worker control connection closed") from cause


def _drain_result_envelopes(
    result_iter: Any,
    *,
    supervisor: Exp007ArmSupervisor,
    identities: Sequence[Exp007SyntheticIdentity],
    expected_envelope_index: int,
    worker_slot_lifetime_bytes: list[int | None],
) -> int:
    while expected_envelope_index < len(identities):
        envelope_payload = _next_result_payload(
            result_iter,
            supervisor=supervisor,
            worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
            timeout=0.0,
            row_index=expected_envelope_index,
        )
        if envelope_payload is None:
            break
        envelope = Exp007SyntheticWorkerResult(**envelope_payload)
        if envelope.worker_rss_bytes is not None:
            active_slot = _slot_for_identity(supervisor, envelope.row_index)
            if active_slot is not None:
                worker_slot_lifetime_bytes[active_slot] = max(
                    worker_slot_lifetime_bytes[active_slot] or 0,
                    envelope.worker_rss_bytes,
                )
        supervisor.receive_envelope(envelope, received_ns=time.monotonic_ns())
        expected_envelope_index += 1
    return expected_envelope_index


def _next_result_payload(
    result_iter: Any,
    *,
    supervisor: Exp007ArmSupervisor,
    worker_slot_lifetime_bytes: Sequence[int | None],
    timeout: float,
    row_index: int | None = None,
) -> Mapping[str, Any] | None:
    try:
        return result_iter.next(timeout=timeout)
    except mp.TimeoutError:
        return None
    except StopIteration:
        supervisor._fail(  # noqa: SLF001
            failure_kind="missing_envelope",
            failure_stage="pool_stream",
            causing_identity=None,
            slot=None,
            generation_nonce=None,
            pid=None,
            token=None,
            worker_rss_bytes=None,
            worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
        )
        raise Exp007RunnerError("worker result iterator ended before all envelopes")
    except Exception as exc:
        report = _worker_failure_report_from_exception(exc)
        if not supervisor.failed:
            if row_index is None:
                supervisor._fail(  # noqa: SLF001
                    failure_kind=report.failure_kind,
                    failure_stage=report.failure_stage,
                    causing_identity=None,
                    slot=None,
                    generation_nonce=None,
                    pid=None,
                    token=None,
                    worker_rss_bytes=None,
                    worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
                )
            else:
                supervisor._fail_row_index(  # noqa: SLF001
                    failure_kind=report.failure_kind,
                    failure_stage=report.failure_stage,
                    row_index=row_index,
                    worker_rss_bytes=None,
                    worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
                )
        raise Exp007RunnerError("worker result iterator raised") from exc


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        _to_jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def normalize_ru_maxrss_bytes(value: int, *, system: str | None = None) -> tuple[int, str]:
    _require_nonnegative_int(value, "ru_maxrss")
    system_name = platform.system() if system is None else system
    if system_name == "Darwin":
        return int(value), "macos_bytes"
    if system_name == "Linux":
        return int(value) * 1024, "linux_kib_times_1024"
    raise RuntimeError(f"unsupported ru_maxrss platform {system_name!r}")


def _run_synthetic_exp007_arm_inline(
    identities: Sequence[Exp007SyntheticIdentity],
    row_callback: Callable[[Exp007SyntheticIdentity], Exp007SyntheticWorkerResult],
    *,
    config: Exp007RunnerConfig,
    protocol: Exp007ProtocolAdapter,
    stage_deadline_ns: int | None,
    stage_summary_factory: StageSummaryFactory | None,
    stage_start_ns: int,
) -> Exp007RunnerOutcome:
    pid_order = tuple(range(10_000, 10_000 + config.worker_count))
    supervisor = Exp007ArmSupervisor(
        identities=identities,
        config=config,
        initial_pid_order=pid_order,
        stage_deadline_ns=stage_deadline_ns,
        protocol=protocol,
    )
    for slot, pid in enumerate(pid_order):
        supervisor.accept_hello(WorkerHello(pid=pid, generation_nonce=_test_nonce(slot)))
    supervisor.assert_handshake_complete()
    worker_slot_lifetime_bytes: list[int | None] = [0] * config.worker_count
    for identity in identities:
        try:
            supervisor.check_deadlines(
                now_ns=time.monotonic_ns(),
                worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
            )
        except Exp007RunnerError:
            return supervisor.hard_failure_outcome(worker_slot_lifetime_bytes=worker_slot_lifetime_bytes)
        slot = identity.row_index % config.worker_count
        pid = pid_order[slot]
        hello = supervisor._hello_by_slot[slot]  # noqa: SLF001
        start_ns = time.monotonic_ns()
        ack = supervisor.start_row(
            RowStartedEvent(
                row=identity,
                slot=slot,
                generation_nonce=hello.generation_nonce,
                pid=pid,
                worker_ready_ns=start_ns,
            ),
            parent_start_ns=start_ns,
        )
        try:
            row = row_callback(identity)
        except Exception as exc:
            report = _worker_failure_report_from_exception(exc)
            supervisor._fail(  # noqa: SLF001
                failure_kind=report.failure_kind,
                failure_stage=report.failure_stage,
                causing_identity=identity,
                slot=slot,
                generation_nonce=ack.generation_nonce,
                pid=pid,
                token=ack.token,
                worker_rss_bytes=None,
                worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
            )
            return supervisor.hard_failure_outcome(worker_slot_lifetime_bytes=worker_slot_lifetime_bytes)
        envelope_sha = canonical_sha256(row)
        now_ns = time.monotonic_ns()
        try:
            supervisor.check_deadlines(
                now_ns=now_ns,
                worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
            )
        except Exp007RunnerError:
            return supervisor.hard_failure_outcome(worker_slot_lifetime_bytes=worker_slot_lifetime_bytes)
        supervisor.finish_row(
            RowFinishedEvent(
                row=identity,
                slot=slot,
                generation_nonce=ack.generation_nonce,
                pid=pid,
                token=ack.token,
                worker_elapsed_ns=max(0, now_ns - start_ns),
                envelope_sha256=envelope_sha,
            ),
            received_ns=now_ns,
        )
        supervisor.receive_envelope(row, received_ns=now_ns)
    try:
        supervisor.check_deadlines(
            now_ns=time.monotonic_ns(),
            worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
        )
    except Exp007RunnerError:
        return supervisor.hard_failure_outcome(worker_slot_lifetime_bytes=worker_slot_lifetime_bytes)
    return _complete_success_or_summary_failure(
        supervisor=supervisor,
        config=config,
        worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
        protocol=protocol,
        stage_summary_factory=stage_summary_factory,
        stage_start_ns=stage_start_ns,
    )


def _schedule_deadline_outcome(
    identities: Sequence[Exp007SyntheticIdentity],
    *,
    completed_rows: Sequence[Exp007SyntheticWorkerResult],
    config: Exp007RunnerConfig,
    protocol: Exp007ProtocolAdapter,
) -> Exp007RunnerOutcome:
    supervisor = Exp007ArmSupervisor(
        identities=identities,
        config=config,
        initial_pid_order=(10_000, 10_001, 10_002, 10_003),
        protocol=protocol,
    )
    supervisor._committed.extend(completed_rows)  # noqa: SLF001
    supervisor._fail(  # noqa: SLF001
        failure_kind="schedule_deadline",
        failure_stage="schedule_deadline",
        causing_identity=None,
        slot=None,
        generation_nonce=None,
        pid=None,
        token=None,
        worker_rss_bytes=None,
        worker_slot_lifetime_bytes=(None, None, None, None),
    )
    return supervisor.hard_failure_outcome()


def _post_arm_schedule_deadline_four_arm_summary(
    outcomes: Mapping[str, Exp007RunnerOutcome],
    *,
    protocol: Exp007ProtocolAdapter,
) -> dict[str, Any]:
    outcome_sha_by_arm = {
        arm: protocol.canonical_sha256(outcomes[arm].outcome)
        for arm in EXP007_EXECUTION_ORDER
    }
    failure_details = _four_arm_failure_details(
        failure_kind="schedule_deadline",
        completed_success_arm_count=len(EXP007_EXECUTION_ORDER),
        first_failure_arm=None,
        causing_outcome_sha256=None,
    )
    return protocol._module.make_four_arm_stage_summary(  # noqa: SLF001
        status="hard_failure",
        arm_outcome_sha256_by_execution_order=outcome_sha_by_arm,
        candidate_global_manifest_sha256=None,
        source_selection_status="not_run",
        config_selection_sha256=None,
        failure_details=failure_details,
    )


def _four_arm_failure_details(
    *,
    failure_kind: str,
    completed_success_arm_count: int,
    first_failure_arm: str | None,
    causing_outcome_sha256: str | None,
    mismatch_cache_audio_key: str | None = None,
    mismatch_field: str | None = None,
) -> dict[str, Any]:
    base = {
        "failure_kind": failure_kind,
        "first_failure_arm": first_failure_arm,
        "causing_outcome_sha256": causing_outcome_sha256,
        "mismatch_cache_audio_key": mismatch_cache_audio_key,
        "mismatch_field": mismatch_field,
        "completed_success_arm_count": completed_success_arm_count,
    }
    deterministic = hashlib.sha256(canonical_json_bytes(base)).hexdigest()
    with_deterministic = {
        **base,
        "deterministic_failure_sha256": deterministic,
    }
    return {
        **with_deterministic,
        "full_failure_sha256": hashlib.sha256(
            canonical_json_bytes(with_deterministic)
        ).hexdigest(),
    }


def _not_run_due_prior_failure(
    identities: Sequence[Exp007SyntheticIdentity],
    *,
    config: Exp007RunnerConfig,
    causing_arm: str,
    causing_outcome_sha256: str,
    reason: str,
    protocol: Exp007ProtocolAdapter,
) -> Exp007RunnerOutcome:
    payload = protocol.not_run_arm(
        identities=identities,
        schedule_arm=config.schedule_arm,
        reason=reason,
        causing_arm=causing_arm,
        causing_outcome_sha256=causing_outcome_sha256,
    )
    return Exp007RunnerOutcome(status="not_run", rows=(), outcome=payload, failure_record=None)


def _resolve_stage_deadline_ns(
    config: Exp007RunnerConfig,
    *,
    stage_deadline_monotonic_ns: int | None,
) -> int | None:
    if stage_deadline_monotonic_ns is not None:
        _require_nonnegative_int(stage_deadline_monotonic_ns, "stage_deadline_monotonic_ns")
        return int(stage_deadline_monotonic_ns)
    if config.stage == EXP007_REPAIR_STAGE:
        return time.monotonic_ns() + int(config.repair_stop_seconds * 1_000_000_000)
    return None


def _runner_stage_telemetry(
    *,
    worker_slot_lifetime_bytes: Sequence[int | None],
    stage_start_ns: int,
) -> RunnerStageTelemetry:
    if len(worker_slot_lifetime_bytes) != EXP007_WORKER_COUNT:
        raise Exp007RunnerError("worker RSS telemetry must contain four values")
    if any(value is None for value in worker_slot_lifetime_bytes):
        raise Exp007RunnerError("successful arm requires complete worker RSS telemetry")
    worker_rss = tuple(int(value) for value in worker_slot_lifetime_bytes)
    elapsed = max(0.0, (time.monotonic_ns() - stage_start_ns) / 1_000_000_000)
    return RunnerStageTelemetry(
        worker_lifetime_rss_bytes=worker_rss,
        aggregate_wall_seconds=elapsed,
    )


def _synthetic_stage_summary_publication(
    *,
    config: Exp007RunnerConfig,
    rows: Sequence[Exp007SyntheticWorkerResult],
    worker_slot_lifetime_bytes: Sequence[int | None],
    protocol: Exp007ProtocolAdapter,
    stage_summary_factory: StageSummaryFactory | None,
    telemetry: RunnerStageTelemetry,
) -> StageSummaryPublication:
    if stage_summary_factory is None:
        raise Exp007RunnerError("successful arm requires stage_summary_factory")
    publication = stage_summary_factory(
        config,
        tuple(rows),
        tuple(worker_slot_lifetime_bytes),
        protocol,
        telemetry,
    )
    if isinstance(publication, StageSummaryPublication):
        return publication
    if isinstance(publication, Mapping):
        return StageSummaryPublication(publication)
    raise Exp007RunnerError(
        "stage_summary_factory must return a mapping or StageSummaryPublication"
    )


def _complete_success_or_summary_failure(
    *,
    supervisor: Exp007ArmSupervisor,
    config: Exp007RunnerConfig,
    worker_slot_lifetime_bytes: Sequence[int | None],
    protocol: Exp007ProtocolAdapter,
    stage_summary_factory: StageSummaryFactory | None,
    stage_start_ns: int,
) -> Exp007RunnerOutcome:
    try:
        with _summary_deadline_alarm(supervisor):
            telemetry = _runner_stage_telemetry(
                worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
                stage_start_ns=stage_start_ns,
            )
            stage_publication = _synthetic_stage_summary_publication(
                config=config,
                rows=supervisor.completed_rows,
                worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
                protocol=protocol,
                stage_summary_factory=stage_summary_factory,
                telemetry=telemetry,
            )
            supervisor.check_deadlines(
                now_ns=time.monotonic_ns(),
                worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
            )
            return supervisor.complete_success(
                stage_publication=stage_publication,
                telemetry=telemetry,
                worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
            )
    except Exception as exc:
        if not supervisor.failed:
            failure_kind, failure_stage = _summary_failure_from_exception(config, exc)
            supervisor._fail(  # noqa: SLF001
                failure_kind=failure_kind,
                failure_stage=failure_stage,
                causing_identity=None,
                slot=None,
                generation_nonce=None,
                pid=None,
                token=None,
                worker_rss_bytes=None,
                worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
            )
        return supervisor.hard_failure_outcome(
            worker_slot_lifetime_bytes=worker_slot_lifetime_bytes
        )


def _summary_failure_from_exception(
    config: Exp007RunnerConfig,
    exc: BaseException,
) -> tuple[FailureKind, FailureStage]:
    if isinstance(exc, _Exp007StageSummaryTimeout):
        failure_kind, _deadline_stage = _stage_deadline_failure(config)
        return failure_kind, _summary_failure_stage(config)
    failure_stage = _summary_failure_stage(config)
    if isinstance(exc, Exp007WorkerFailure):
        report = _worker_failure_report_from_exception(exc)
        if report.failure_stage == failure_stage and report.failure_kind in {
            "atomic_publication_failure",
            "summary_publication_failure",
            "schema_failure",
        }:
            return report.failure_kind, report.failure_stage
    return "schema_failure", failure_stage


def _summary_failure_stage(config: Exp007RunnerConfig) -> FailureStage:
    if config.stage == EXP007_REPAIR_STAGE:
        return "repair_summary"
    return "arm_summary"


@contextmanager
def _summary_deadline_alarm(
    supervisor: Exp007ArmSupervisor,
) -> Any:
    deadline_ns = supervisor.stage_deadline_ns
    if deadline_ns is None or not hasattr(signal, "setitimer") or not hasattr(
        signal,
        "SIGALRM",
    ):
        yield
        return
    remaining_seconds = max(0.0, (deadline_ns - time.monotonic_ns()) / 1_000_000_000)
    if remaining_seconds <= 0.0:
        raise _Exp007StageSummaryTimeout("stage deadline crossed before summary")

    def _handler(_signum: int, _frame: Any) -> None:
        raise _Exp007StageSummaryTimeout("stage deadline crossed during summary")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, max(remaining_seconds, 0.001))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        delay, interval = previous_timer
        if delay > 0.0 or interval > 0.0:
            signal.setitimer(signal.ITIMER_REAL, delay, interval)


def _stage_deadline_failure(config: Exp007RunnerConfig) -> tuple[FailureKind, FailureStage]:
    if config.stage == EXP007_SCHEDULE_STAGE:
        return "schedule_deadline", "schedule_deadline"
    return "arm_deadline", "pool_stream"


def _validate_repair_context_against_committed_rows(
    *,
    context: RepairSummaryValidationContext,
    rows: Sequence[Exp007SyntheticWorkerResult],
) -> None:
    if len(context.prediction_rows) != len(rows):
        raise Exp007RunnerError(
            "repair summary context prediction rows do not match completed row count"
        )
    for result, prediction_row in zip(rows, context.prediction_rows, strict=True):
        if not isinstance(prediction_row, Mapping):
            raise Exp007RunnerError("repair summary context prediction row is invalid")
        committed_payload = result.payload.get("row_result")
        if not isinstance(committed_payload, Mapping):
            raise Exp007RunnerError(
                "repair completed row payload is missing row_result authority bytes"
            )
        if canonical_json_bytes(committed_payload) != canonical_json_bytes(prediction_row):
            raise Exp007RunnerError(
                "repair summary context prediction row differs from committed row payload"
            )
        if prediction_row.get("row_index") != result.row_index:
            raise Exp007RunnerError("repair prediction row index mismatch")
        if prediction_row.get("cache_audio_key") != result.cache_audio_key:
            raise Exp007RunnerError("repair prediction row cache key mismatch")
        if prediction_row.get("identity_payload_sha256") != result.identity_payload_sha256:
            raise Exp007RunnerError("repair prediction row identity SHA mismatch")
        if prediction_row.get("row_payload_sha256") != result.row_payload_sha256:
            raise Exp007RunnerError("repair prediction row payload SHA mismatch")


def _active_deadline_values(*values: int | None) -> tuple[int, ...]:
    return tuple(int(value) for value in values if value is not None)


def _seconds_until_deadline_ns(deadline_ns: int, *, now_ns: int | None = None) -> float:
    now = time.monotonic_ns() if now_ns is None else now_ns
    return max(0.0, (deadline_ns - now) / 1_000_000_000)


def _has_due_deadline(supervisor: Exp007ArmSupervisor, *, now_ns: int) -> bool:
    if supervisor.stage_deadline_ns is not None and now_ns >= supervisor.stage_deadline_ns:
        return True
    if any(now_ns >= active.deadline_ns for active in supervisor._active_by_slot.values()):  # noqa: SLF001
        return True
    guard_ns = int(supervisor.config.finish_result_delivery_seconds * 1_000_000_000)
    return any(
        now_ns - guard_start >= guard_ns
        for guard_start in supervisor._join_guard_start_by_row.values()  # noqa: SLF001
    )


def _worker_rss_snapshot(values: Sequence[int | None]) -> dict[str, Any]:
    normalized = [None if value is None else int(value) for value in values]
    observed = None if all(value is None for value in normalized) else max(value or 0 for value in normalized)
    return {
        "worker_slot_lifetime_bytes": normalized,
        "observed_worker_max_bytes": observed,
    }


_WORKER_CONTROL_CONNECTION: Any | None = None
_WORKER_HELLO_ACK: WorkerHelloAck | None = None
_WORKER_CONFIG: Exp007RunnerConfig | None = None


def _worker_initializer_config(config: Exp007RunnerConfig) -> Exp007RunnerConfig:
    return config


def _synthetic_worker_initializer(listener_address: Any, config: Exp007RunnerConfig) -> None:
    global _WORKER_CONTROL_CONNECTION, _WORKER_HELLO_ACK, _WORKER_CONFIG
    _WORKER_CONFIG = config
    conn = Client(listener_address, family="AF_INET")
    pid = os.getpid()
    nonce = hashlib.sha256(f"{pid}:{time.time_ns()}:{os.urandom(16).hex()}".encode("ascii")).hexdigest()
    conn.send(asdict(WorkerHello(pid=pid, generation_nonce=nonce)))
    ack_payload = conn.recv()
    ack = WorkerHelloAck(**ack_payload)
    if ack.pid != pid or ack.generation_nonce != nonce:
        raise Exp007RunnerError("worker received invalid HELLO ACK")
    conn.send({"slot": ack.slot})
    _WORKER_CONTROL_CONNECTION = conn
    _WORKER_HELLO_ACK = ack


def _synthetic_worker_run_row(
    item: tuple[Exp007SyntheticIdentity, Callable[[Exp007SyntheticIdentity], Exp007SyntheticWorkerResult]],
) -> dict[str, Any]:
    identity, callback = item
    if _WORKER_CONTROL_CONNECTION is None or _WORKER_HELLO_ACK is None or _WORKER_CONFIG is None:
        raise Exp007RunnerError("worker was not initialized")
    ack = _WORKER_HELLO_ACK
    conn = _WORKER_CONTROL_CONNECTION
    start = RowStartedEvent(
        row=identity,
        slot=ack.slot,
        generation_nonce=ack.generation_nonce,
        pid=ack.pid,
        worker_ready_ns=time.monotonic_ns(),
    )
    conn.send(
        {
            "type": "row_started",
            "row": asdict(identity),
            "slot": start.slot,
            "generation_nonce": start.generation_nonce,
            "pid": start.pid,
            "worker_ready_ns": start.worker_ready_ns,
        }
    )
    ack_message = conn.recv()
    row_ack = RowStartAck(**ack_message["ack"])
    started_ns = time.monotonic_ns()
    alarm_state = _install_worker_alarm(_WORKER_CONFIG.per_audio_arm_timeout_seconds)
    try:
        result = callback(identity)
        rss_bytes, _ = normalize_ru_maxrss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        result = dataclasses.replace(result, worker_rss_bytes=rss_bytes)
        envelope_sha256 = canonical_sha256(result)
        elapsed_ns = time.monotonic_ns() - started_ns
        finish = RowFinishedEvent(
            row=identity,
            slot=row_ack.slot,
            generation_nonce=row_ack.generation_nonce,
            pid=row_ack.pid,
            token=row_ack.token,
            worker_elapsed_ns=elapsed_ns,
            envelope_sha256=envelope_sha256,
        )
        conn.send(
            {
                "type": "row_finished",
                "row": asdict(identity),
                "slot": finish.slot,
                "generation_nonce": finish.generation_nonce,
                "pid": finish.pid,
                "token": finish.token,
                "worker_elapsed_ns": finish.worker_elapsed_ns,
                "envelope_sha256": finish.envelope_sha256,
                "worker_rss_bytes": rss_bytes,
            }
        )
        return asdict(result)
    except Exp007WorkerTimeout as exc:
        elapsed_ns = time.monotonic_ns() - started_ns
        try:
            rss_bytes, _ = normalize_ru_maxrss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        except Exception:
            rss_bytes = None
        conn.send(
            _worker_failure_payload(
                identity=identity,
                row_ack=row_ack,
                elapsed_ns=elapsed_ns,
                rss_bytes=rss_bytes,
                report=_worker_failure_report_from_exception(exc),
            )
        )
        raise
    except Exception as exc:
        elapsed_ns = time.monotonic_ns() - started_ns
        try:
            rss_bytes, _ = normalize_ru_maxrss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        except Exception:
            rss_bytes = None
        conn.send(
            _worker_failure_payload(
                identity=identity,
                row_ack=row_ack,
                elapsed_ns=elapsed_ns,
                rss_bytes=rss_bytes,
                report=_worker_failure_report_from_exception(exc),
            )
        )
        raise
    finally:
        _cancel_worker_alarm(alarm_state)


def _install_worker_alarm(timeout_seconds: float) -> tuple[Any, tuple[float, float]] | None:
    if not hasattr(signal, "setitimer") or not hasattr(signal, "SIGALRM"):
        return None

    def _handler(_signum: int, _frame: Any) -> None:
        raise Exp007WorkerTimeout("Exp007 worker row timeout")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, max(timeout_seconds, 0.001))
    return previous_handler, previous_timer


def _cancel_worker_alarm(state: tuple[Any, tuple[float, float]] | None) -> None:
    if hasattr(signal, "setitimer") and hasattr(signal, "SIGALRM"):
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        if state is None:
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
            return
        previous_handler, previous_timer = state
        signal.signal(signal.SIGALRM, previous_handler)
        delay, interval = previous_timer
        if delay > 0.0 or interval > 0.0:
            signal.setitimer(signal.ITIMER_REAL, delay, interval)


def _detect_worker_death(
    pool: Any,
    supervisor: Exp007ArmSupervisor,
    worker_slot_lifetime_bytes: Sequence[int | None],
    *,
    failure_stage: str = "pool_stream",
    before_failure: Callable[[], None] | None = None,
) -> None:
    initial_pids = set(supervisor._pid_slot)  # noqa: SLF001
    current = tuple(pool._pool)  # noqa: SLF001
    current_pids = {process.pid for process in current if process.pid is not None}
    if current_pids != initial_pids:
        missing = next(iter(initial_pids - current_pids), None)
        extra = next(iter(current_pids - initial_pids), None)
        if before_failure is not None:
            before_failure()
        if missing is not None:
            supervisor.fail_worker_death(
                pid=int(missing),
                failure_stage=failure_stage,
                worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
            )
        else:
            supervisor.fail_pool_replacement(
                pid=int(extra or -1),
                failure_stage=failure_stage,
                worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
            )
        raise Exp007RunnerError("worker death or pool replacement detected")
    for process in current:
        if process.exitcode is not None:
            if before_failure is not None:
                before_failure()
            supervisor.fail_worker_death(
                pid=int(process.pid),
                failure_stage=failure_stage,
                worker_slot_lifetime_bytes=worker_slot_lifetime_bytes,
            )
            raise Exp007RunnerError("worker exited unexpectedly")


def _terminate_pool_bounded(pool: Any, config: Exp007RunnerConfig) -> None:
    processes = tuple(getattr(pool, "_pool", ()))
    errors: list[str] = []
    try:
        pool.terminate()
    except Exception as exc:
        errors.append(f"terminate:{type(exc).__name__}")
    deadline = time.monotonic() + config.worker_terminate_grace_seconds
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.join(remaining)
        except Exception as exc:
            errors.append(f"terminate_join:{type(exc).__name__}")
    survivors = [process for process in processes if process.is_alive()]
    for process in survivors:
        try:
            process.kill()
        except Exception as exc:
            errors.append(f"kill:{type(exc).__name__}")
    deadline = time.monotonic() + config.worker_kill_grace_seconds
    for process in survivors:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.join(remaining)
        except Exception as exc:
            errors.append(f"kill_join:{type(exc).__name__}")
    survivors = [process for process in processes if process.is_alive()]
    if survivors or errors:
        raise Exp007RunnerError(
            "bounded pool teardown failed"
            f" survivors={len(survivors)} errors={','.join(errors) if errors else 'none'}"
        )


def _close_connections(connections: Sequence[Any]) -> None:
    for conn in connections:
        try:
            conn.close()
        except OSError:
            pass


def _supervisor_wait_timeout(supervisor: Exp007ArmSupervisor, config: Exp007RunnerConfig) -> float:
    now_ns = time.monotonic_ns()
    deadlines = [active.deadline_ns for active in supervisor._active_by_slot.values()]  # noqa: SLF001
    if supervisor.stage_deadline_ns is not None:
        deadlines.append(supervisor.stage_deadline_ns)
    guard_starts = list(supervisor._join_guard_start_by_row.values())  # noqa: SLF001
    for guard_start in guard_starts:
        deadlines.append(guard_start + int(config.finish_result_delivery_seconds * 1_000_000_000))
    if not deadlines:
        return config.parent_poll_max_seconds
    nearest = max(0.0, (min(deadlines) - now_ns) / 1_000_000_000)
    return min(config.parent_poll_max_seconds, nearest)


def _control_eof_result_grace_seconds(supervisor: Exp007ArmSupervisor) -> float:
    config = supervisor.config
    return min(
        config.finish_result_delivery_seconds,
        max(config.parent_poll_max_seconds, 0.25),
    )


def _slot_for_identity(supervisor: Exp007ArmSupervisor, row_index: int) -> int | None:
    for slot, active in supervisor._active_by_slot.items():  # noqa: SLF001
        if active.row.row_index == row_index:
            return slot
    return None


def _validate_rss_success(values: Sequence[int | None], config: Exp007RunnerConfig) -> None:
    if len(values) != config.worker_count:
        raise Exp007RunnerError("RSS vector must contain four slots")
    if any(value is None for value in values):
        raise Exp007RunnerError("successful arm is missing worker RSS telemetry")
    if any(int(value) > config.worker_rss_cap_bytes for value in values if value is not None):
        raise Exp007RunnerError("worker RSS cap exceeded")


def _preflight_platform() -> None:
    normalize_ru_maxrss_bytes(0)
    if not hasattr(signal, "setitimer") or not hasattr(signal, "SIGALRM"):
        raise RuntimeError("Exp007 synthetic worker guard requires POSIX interval timers")


def _dispatch_token(event: RowStartedEvent, *, parent_start_ns: int) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "row": event.row.pending_ref(),
                "slot": event.slot,
                "generation_nonce": event.generation_nonce,
                "pid": event.pid,
                "parent_start_ns": parent_start_ns,
            }
        )
    ).hexdigest()


def _finish_matches_active(event: RowFinishedEvent, active: _ActiveRow) -> bool:
    return (
        event.row == active.row
        and event.slot == active.slot
        and event.generation_nonce == active.generation_nonce
        and event.pid == active.pid
        and event.token == active.token
    )


def _validate_identity_order(
    identities: Sequence[Exp007SyntheticIdentity],
    *,
    expected_count: int | None,
) -> None:
    seen_indexes: set[int] = set()
    seen_keys: set[str] = set()
    for expected_index, identity in enumerate(identities):
        if not isinstance(identity, Exp007SyntheticIdentity):
            raise TypeError("identities must contain Exp007SyntheticIdentity")
        if identity.row_index != expected_index:
            raise ValueError("identity row_index values must be contiguous from zero")
        if identity.row_index in seen_indexes or identity.cache_audio_key in seen_keys:
            raise ValueError("identity rows must have unique row indexes and cache keys")
        seen_indexes.add(identity.row_index)
        seen_keys.add(identity.cache_audio_key)
    if expected_count is not None and len(identities) != expected_count:
        raise ValueError(f"expected {expected_count} identities")


def _to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def _test_nonce(slot: int) -> str:
    return hashlib.sha256(f"exp007-test-nonce:{slot}".encode("ascii")).hexdigest()


def _require_slot(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < EXP007_WORKER_COUNT:
        raise ValueError("slot must be an integer in [0, 3]")


def _require_nonnegative_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _require_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_positive_float(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")


def _require_failure_kind(value: object, name: str) -> FailureKind:
    if not isinstance(value, str) or value not in EXP007_FAILURE_KIND_SET:
        raise ValueError(f"{name} must be an Exp007 failure kind")
    return value  # type: ignore[return-value]


def _require_failure_stage(value: object, name: str) -> FailureStage:
    if not isinstance(value, str) or value not in EXP007_FAILURE_STAGE_SET:
        raise ValueError(f"{name} must be an Exp007 failure stage")
    return value  # type: ignore[return-value]


def _require_nonempty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")


def _require_nonempty_sequence(value: object, name: str) -> None:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) == 0
    ):
        raise ValueError(f"{name} must be a nonempty sequence")


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lower-case SHA-256 hex digest")


__all__ = [
    "EXP007_EXECUTION_ORDER",
    "EXP007_EXPERIMENT_ID",
    "EXP007_REPAIR_STAGE",
    "EXP007_SCHEDULE_ARMS",
    "EXP007_SCHEDULE_STAGE",
    "EXP007_WORKER_COUNT",
    "Exp007ArmSupervisor",
    "Exp007ProtocolAdapter",
    "Exp007RunnerConfig",
    "Exp007RunnerError",
    "Exp007RunnerOutcome",
    "Exp007ScheduleOutcomes",
    "Exp007SyntheticIdentity",
    "Exp007SyntheticWorkerResult",
    "Exp007WorkerFailure",
    "Exp007WorkerFailureReport",
    "RepairSummaryValidationContext",
    "RunnerStageTelemetry",
    "RowFinishedEvent",
    "RowStartAck",
    "RowStartedEvent",
    "StageSummaryPublication",
    "WorkerHello",
    "WorkerHelloAck",
    "canonical_json_bytes",
    "canonical_sha256",
    "normalize_ru_maxrss_bytes",
    "run_synthetic_exp007_arm",
    "run_synthetic_exp007_schedule",
]
