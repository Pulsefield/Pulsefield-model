from __future__ import annotations

import hashlib
import os
import stat
import uuid
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from pulsefield_model.timing.evaluation import exp007_protocol as protocol
from pulsefield_model.timing.v3 import global_constant_jump as candidate_source


CANDIDATE_PAYLOAD_SCHEMA = "pulsefield_model.timing_v3_exp007_candidate_payload_v1"
CANDIDATE_REFERENCE_ROW_BUNDLE_SCHEMA = (
    "pulsefield_model.timing_v3_exp007_candidate_reference_row_bundle_v1"
)
CANDIDATE_REFERENCE_MANIFEST_SCHEMA = (
    "pulsefield_model.timing_v3_exp007_candidate_reference_manifest_v1"
)
CANDIDATE_GLOBAL_MANIFEST_SCHEMA = (
    "pulsefield_model.timing_v3_exp007_candidate_global_manifest_v1"
)
EXPOSURE_DELTA_SCHEMA = "pulsefield_model.timing_v3_exp007_exposure_delta_v1"
RUN_LOCK_SCHEMA = "pulsefield_model.timing_v3_exp007_run_lock_v1"
EXPOSURE_DELTA_BYTE_CAP = 1_048_576
RUN_LOCK_BYTE_CAP = 16_384
_DIRFD_TEST_HOOK: Any = None
CANDIDATE_PAYLOAD_FIELDS = frozenset(
    {
        "schema",
        "beat_peaks",
        "downbeat_peaks",
        "tempo_candidates",
        "origin_candidates",
        "boundary_candidates",
        "diagnostics",
    }
)
MATERIALIZED_PEAK_FIELDS = frozenset(
    {"frame_index", "refined_frame", "time_ms", "confidence"}
)
TEMPO_CANDIDATE_FIELDS = frozenset({"bpm", "source", "score"})
ORIGIN_CANDIDATE_FIELDS = frozenset({"anchor_id", "time_ms", "bpm", "score"})
BOUNDARY_CANDIDATE_FIELDS = frozenset(
    {
        "anchor_id",
        "time_ms",
        "source_peak_index",
        "source_peak_time_ms",
        "source_peak_confidence",
        "rank_score",
        "evidence_mode",
        "left_period_ms",
        "right_period_ms",
        "ordinary_score",
        "super_score",
        "downbeat_bonus",
        "nearest_downbeat_distance_ms",
    }
)
CANDIDATE_DIAGNOSTICS_FIELDS = frozenset(
    {
        "candidate_contract_version",
        "constants_json_sha256",
        "pulse_correlation_version",
        "boundary_candidate_score_version",
        "frame_count",
        "frame_rate_hz",
        "coverage_start_ms",
        "coverage_end_ms",
        "min_period_frames",
        "max_period_frames",
        "beat_peak_count",
        "downbeat_peak_count",
        "tempo_candidate_count",
        "origin_candidate_count",
        "boundary_candidate_count",
        "input_signal_sha256",
        "candidate_fingerprint",
    }
)
CANDIDATE_PAYLOAD_DESCRIPTOR: Mapping[str, Any] = {
    "schema": CANDIDATE_PAYLOAD_SCHEMA,
    "fields": sorted(CANDIDATE_PAYLOAD_FIELDS),
    "materialized_peak_fields": sorted(MATERIALIZED_PEAK_FIELDS),
    "tempo_candidate_fields": sorted(TEMPO_CANDIDATE_FIELDS),
    "origin_candidate_fields": sorted(ORIGIN_CANDIDATE_FIELDS),
    "boundary_candidate_fields": sorted(BOUNDARY_CANDIDATE_FIELDS),
    "candidate_diagnostics_fields": sorted(CANDIDATE_DIAGNOSTICS_FIELDS),
}
CANDIDATE_PAYLOAD_FIELD_SET_SHA256 = protocol.candidate_payload_field_set_sha256()

CANDIDATE_REFERENCE_ENTRY_FIELDS = frozenset(
    {
        "row_index",
        "cache_audio_key",
        "audio_group_key",
        "input_signal_sha256",
        "candidate_payload_schema",
        "candidate_payload_field_set_sha256",
        "candidate_payload_byte_count",
        "candidate_payload_sha256",
        "candidate_fingerprint",
        "candidate_payload",
        "bound_row_payload_sha256",
        "entry_payload_sha256",
    }
)
CANDIDATE_REFERENCE_REF_FIELDS = frozenset(
    {
        "row_index",
        "cache_audio_key",
        "audio_group_key",
        "input_signal_sha256",
        "entry_payload_sha256",
        "candidate_payload_byte_count",
        "candidate_payload_sha256",
        "candidate_fingerprint",
        "bundle_relative_path",
        "bundle_fingerprint_sha256",
    }
)
CANDIDATE_REFERENCE_ROW_BUNDLE_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "stage",
        "schema_descriptor_sha256",
        "schedule_arm",
        "row_index",
        "entry",
        "row",
        "bundle_fingerprint_sha256",
    }
)
CANDIDATE_REFERENCE_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "stage",
        "schema_descriptor_sha256",
        "input_manifest_sha256",
        "source_closure_fingerprint_sha256",
        "reference_arm",
        "row_count",
        "entries",
        "ordered_entries_sha256",
        "manifest_fingerprint_sha256",
    }
)
ARM_ROW_SHA_MAP_FIELDS = frozenset(protocol.EXP007_EXECUTION_ORDER)
CANDIDATE_GLOBAL_ENTRY_FIELDS = frozenset(
    {
        "row_index",
        "cache_audio_key",
        "audio_group_key",
        "input_signal_sha256",
        "candidate_payload_schema",
        "candidate_payload_field_set_sha256",
        "candidate_payload_byte_count",
        "candidate_payload_sha256",
        "candidate_fingerprint",
        "candidate_reference_entry_payload_sha256",
        "arm_row_payload_sha256",
    }
)
CANDIDATE_GLOBAL_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "stage",
        "schema_descriptor_sha256",
        "selector_manifest_sha256",
        "source_closure_fingerprint_sha256",
        "candidate_reference_manifest_sha256",
        "row_count",
        "entries",
        "ordered_entries_sha256",
        "manifest_fingerprint_sha256",
    }
)
EXPOSURE_STAGES = frozenset(
    {protocol.EXP007_SCHEDULE_STAGE, protocol.EXP007_REPAIR_STAGE, "accidental_batch"}
)
OBSERVED_PAYLOAD_KINDS = frozenset(
    {
        "identity",
        "cache",
        "prediction",
        "grid",
        "metric",
        "diagnostic",
        "runtime",
        "failure",
        "trace",
        "osu",
        "rendering",
        "batch_aggregate",
    }
)
EXPOSURE_ENTRY_FIELDS = frozenset(
    {
        "cache_audio_key",
        "audio_group_key",
        "exposure_stage",
        "exposure_reason",
        "first_exposed_at_or_run_id",
        "observed_payload_kind",
        "source_manifest_sha256",
    }
)
EXPOSURE_DELTA_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "schema_descriptor_sha256",
        "generated_at_utc",
        "prior_exposure_manifest_sha256",
        "source_closure_fingerprint_sha256",
        "delta_reason",
        "entry_count",
        "cache_audio_keys_sha256",
        "entries_sha256",
        "entries",
        "manifest_fingerprint_sha256",
    }
)
RUN_LOCK_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "stage",
        "schema_descriptor_sha256",
        "schedule_arm",
        "selector_manifest_sha256",
        "source_closure_fingerprint_sha256",
        "run_config_fingerprint_sha256",
        "output_root_fingerprint_sha256",
        "owner_run_id",
        "acquired_at_utc",
        "lock_payload_sha256",
    }
)


class Exp007ArtifactError(ValueError):
    """Raised when an Exp007 artifact cannot be validated or published."""


@dataclass(frozen=True)
class Exp007RunLock:
    root: Path
    relative_path: str
    payload: dict[str, Any]
    acquired: bool


@dataclass(frozen=True)
class CandidatePayloadBytes:
    payload: dict[str, Any]
    canonical_bytes: bytes
    schema: str
    field_set_sha256: str
    byte_count: int
    payload_sha256: str
    candidate_fingerprint: str


@dataclass(frozen=True)
class _AnchoredDestination:
    path: Path
    parent: Path
    name: str
    parent_fd: int
    parent_metadata: os.stat_result
    root_resolved: Path | None


def serialize_candidate_payload(
    candidate_payload: Mapping[str, Any],
    *,
    byte_cap: int = protocol.EXP007_CANDIDATE_PAYLOAD_BYTE_CAP,
) -> CandidatePayloadBytes:
    payload = validate_candidate_payload(candidate_payload)
    canonical_bytes = canonical_json_bytes_under_cap(
        payload,
        byte_cap=byte_cap,
        context="CandidatePayload",
    )
    payload_sha256 = hashlib.sha256(canonical_bytes).hexdigest()
    candidate_fingerprint = payload["diagnostics"]["candidate_fingerprint"]
    return CandidatePayloadBytes(
        payload=payload,
        canonical_bytes=canonical_bytes,
        schema=payload["schema"],
        field_set_sha256=CANDIDATE_PAYLOAD_FIELD_SET_SHA256,
        byte_count=len(canonical_bytes),
        payload_sha256=payload_sha256,
        candidate_fingerprint=candidate_fingerprint,
    )


def validate_candidate_payload(candidate_payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(candidate_payload, "CandidatePayload")
    protocol.validate_exact_fields(
        candidate_payload,
        CANDIDATE_PAYLOAD_FIELDS,
        "CandidatePayload",
    )
    schema = protocol.require_nonempty_string(
        candidate_payload.get("schema"),
        "CandidatePayload.schema",
    )
    if schema != CANDIDATE_PAYLOAD_SCHEMA:
        raise Exp007ArtifactError("CandidatePayload schema is invalid")
    beat_peaks = [
        _validate_materialized_peak(item, context="CandidatePayload.beat_peaks[]")
        for item in _require_sequence(candidate_payload.get("beat_peaks"), "beat_peaks")
    ]
    downbeat_peaks = [
        _validate_materialized_peak(item, context="CandidatePayload.downbeat_peaks[]")
        for item in _require_sequence(
            candidate_payload.get("downbeat_peaks"),
            "downbeat_peaks",
        )
    ]
    tempo_candidates = [
        _validate_tempo_candidate(item)
        for item in _require_sequence(
            candidate_payload.get("tempo_candidates"),
            "tempo_candidates",
        )
    ]
    origin_candidates = [
        _validate_origin_candidate(item)
        for item in _require_sequence(
            candidate_payload.get("origin_candidates"),
            "origin_candidates",
        )
    ]
    boundary_candidates = [
        _validate_boundary_candidate(item)
        for item in _require_sequence(
            candidate_payload.get("boundary_candidates"),
            "boundary_candidates",
        )
    ]
    diagnostics = _validate_candidate_diagnostics(
        candidate_payload.get("diagnostics"),
        beat_peak_count=len(beat_peaks),
        downbeat_peak_count=len(downbeat_peaks),
        tempo_candidate_count=len(tempo_candidates),
        origin_candidate_count=len(origin_candidates),
        boundary_candidate_count=len(boundary_candidates),
    )
    payload = {
        "schema": CANDIDATE_PAYLOAD_SCHEMA,
        "beat_peaks": beat_peaks,
        "downbeat_peaks": downbeat_peaks,
        "tempo_candidates": tempo_candidates,
        "origin_candidates": origin_candidates,
        "boundary_candidates": boundary_candidates,
        "diagnostics": diagnostics,
    }
    _validate_source_candidate_contract(payload)
    return payload


def make_candidate_reference_entry(
    *,
    row: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
    input_signal_sha256: str,
) -> dict[str, Any]:
    row_payload = protocol.validate_row_result(row)
    candidate = serialize_candidate_payload(candidate_payload)
    _require_row_candidate_binding(row_payload, candidate)
    payload = {
        "row_index": row_payload["row_index"],
        "cache_audio_key": row_payload["cache_audio_key"],
        "audio_group_key": row_payload["audio_group_key"],
        "input_signal_sha256": input_signal_sha256,
        "candidate_payload_schema": candidate.schema,
        "candidate_payload_field_set_sha256": candidate.field_set_sha256,
        "candidate_payload_byte_count": candidate.byte_count,
        "candidate_payload_sha256": candidate.payload_sha256,
        "candidate_fingerprint": row_payload["candidate_fingerprint"],
        "candidate_payload": candidate.payload,
        "bound_row_payload_sha256": row_payload["row_payload_sha256"],
    }
    return validate_candidate_reference_entry(
        protocol.with_payload_hash(payload, "entry_payload_sha256")
    )


def validate_candidate_reference_entry(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "CandidateReferenceEntry")
    protocol.validate_exact_fields(
        payload,
        CANDIDATE_REFERENCE_ENTRY_FIELDS,
        "CandidateReferenceEntry",
    )
    candidate = serialize_candidate_payload(payload.get("candidate_payload"))
    result = {
        "row_index": protocol.require_nonnegative_int(
            payload.get("row_index"),
            "CandidateReferenceEntry.row_index",
        ),
        "cache_audio_key": protocol.require_nonempty_string(
            payload.get("cache_audio_key"),
            "CandidateReferenceEntry.cache_audio_key",
        ),
        "audio_group_key": protocol.require_nonempty_string(
            payload.get("audio_group_key"),
            "CandidateReferenceEntry.audio_group_key",
        ),
        "input_signal_sha256": protocol.require_sha256(
            payload.get("input_signal_sha256"),
            "CandidateReferenceEntry.input_signal_sha256",
        ),
        "candidate_payload_schema": protocol.require_nonempty_string(
            payload.get("candidate_payload_schema"),
            "CandidateReferenceEntry.candidate_payload_schema",
        ),
        "candidate_payload_field_set_sha256": protocol.require_sha256(
            payload.get("candidate_payload_field_set_sha256"),
            "CandidateReferenceEntry.candidate_payload_field_set_sha256",
        ),
        "candidate_payload_byte_count": protocol.require_nonnegative_int(
            payload.get("candidate_payload_byte_count"),
            "CandidateReferenceEntry.candidate_payload_byte_count",
        ),
        "candidate_payload_sha256": protocol.require_sha256(
            payload.get("candidate_payload_sha256"),
            "CandidateReferenceEntry.candidate_payload_sha256",
        ),
        "candidate_fingerprint": protocol.require_sha256(
            payload.get("candidate_fingerprint"),
            "CandidateReferenceEntry.candidate_fingerprint",
        ),
        "candidate_payload": candidate.payload,
        "bound_row_payload_sha256": protocol.require_sha256(
            payload.get("bound_row_payload_sha256"),
            "CandidateReferenceEntry.bound_row_payload_sha256",
        ),
        "entry_payload_sha256": protocol.require_sha256(
            payload.get("entry_payload_sha256"),
            "CandidateReferenceEntry.entry_payload_sha256",
        ),
    }
    _require_entry_candidate_binding(result, candidate)
    protocol.validate_payload_hash(
        result,
        "entry_payload_sha256",
        context="CandidateReferenceEntry",
    )
    return result


def make_candidate_reference_row_bundle(
    *,
    stage: str,
    schedule_arm: str,
    row: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
    input_signal_sha256: str,
) -> dict[str, Any]:
    stage_value = _require_stage(stage, "CandidateReferenceRowBundle.stage")
    arm = _require_schedule_arm(
        schedule_arm,
        "CandidateReferenceRowBundle.schedule_arm",
    )
    _require_reference_arm(stage_value, arm)
    row_payload = protocol.validate_row_result(row)
    if row_payload["stage"] != stage_value or row_payload["schedule_arm"] != arm:
        raise Exp007ArtifactError("CandidateReferenceRowBundle row stage/arm mismatch")
    entry = make_candidate_reference_entry(
        row=row_payload,
        candidate_payload=candidate_payload,
        input_signal_sha256=input_signal_sha256,
    )
    payload = {
        "schema": CANDIDATE_REFERENCE_ROW_BUNDLE_SCHEMA,
        "experiment_id": protocol.EXP007_EXPERIMENT_ID,
        "stage": stage_value,
        "schema_descriptor_sha256": protocol.schema_descriptor_sha256(
            CANDIDATE_REFERENCE_ROW_BUNDLE_SCHEMA
        ),
        "schedule_arm": arm,
        "row_index": row_payload["row_index"],
        "entry": entry,
        "row": row_payload,
    }
    return validate_candidate_reference_row_bundle(
        protocol.with_payload_hash(payload, "bundle_fingerprint_sha256")
    )


def validate_candidate_reference_row_bundle(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    _require_mapping(payload, "CandidateReferenceRowBundle")
    protocol.validate_exact_fields(
        payload,
        CANDIDATE_REFERENCE_ROW_BUNDLE_FIELDS,
        "CandidateReferenceRowBundle",
    )
    if payload.get("schema") != CANDIDATE_REFERENCE_ROW_BUNDLE_SCHEMA:
        raise Exp007ArtifactError("CandidateReferenceRowBundle schema is invalid")
    if payload.get("experiment_id") != protocol.EXP007_EXPERIMENT_ID:
        raise Exp007ArtifactError("CandidateReferenceRowBundle experiment_id is invalid")
    stage = _require_stage(payload.get("stage"), "CandidateReferenceRowBundle.stage")
    arm = _require_schedule_arm(
        payload.get("schedule_arm"),
        "CandidateReferenceRowBundle.schedule_arm",
    )
    _require_reference_arm(stage, arm)
    _require_descriptor(
        payload,
        CANDIDATE_REFERENCE_ROW_BUNDLE_SCHEMA,
        "CandidateReferenceRowBundle",
    )
    row_index = protocol.require_nonnegative_int(
        payload.get("row_index"),
        "CandidateReferenceRowBundle.row_index",
    )
    entry = validate_candidate_reference_entry(payload.get("entry"))
    row = protocol.validate_row_result(payload.get("row"))
    if row["stage"] != stage or row["schedule_arm"] != arm:
        raise Exp007ArtifactError("CandidateReferenceRowBundle row stage/arm mismatch")
    if row["row_index"] != row_index or entry["row_index"] != row_index:
        raise Exp007ArtifactError("CandidateReferenceRowBundle row index mismatch")
    _require_row_candidate_binding(
        row,
        serialize_candidate_payload(entry["candidate_payload"]),
    )
    _require_entry_row_binding(entry, row)
    result = {
        "schema": CANDIDATE_REFERENCE_ROW_BUNDLE_SCHEMA,
        "experiment_id": protocol.EXP007_EXPERIMENT_ID,
        "stage": stage,
        "schema_descriptor_sha256": payload["schema_descriptor_sha256"],
        "schedule_arm": arm,
        "row_index": row_index,
        "entry": entry,
        "row": row,
        "bundle_fingerprint_sha256": protocol.require_sha256(
            payload.get("bundle_fingerprint_sha256"),
            "CandidateReferenceRowBundle.bundle_fingerprint_sha256",
        ),
    }
    protocol.validate_payload_hash(
        result,
        "bundle_fingerprint_sha256",
        context="CandidateReferenceRowBundle",
    )
    canonical_json_bytes_under_cap(
        result,
        byte_cap=protocol.EXP007_CANDIDATE_BUNDLE_BYTE_CAP,
        context="CandidateReferenceRowBundle",
    )
    return result


def make_candidate_reference_ref(
    *,
    bundle: Mapping[str, Any],
    bundle_relative_path: str,
) -> dict[str, Any]:
    bundle_payload = validate_candidate_reference_row_bundle(bundle)
    relative_path = validate_bundle_relative_path(bundle_relative_path)
    _require_canonical_reference_bundle_path(
        stage=bundle_payload["stage"],
        schedule_arm=bundle_payload["schedule_arm"],
        row_index=bundle_payload["row_index"],
        relative_path=relative_path,
        context="CandidateReferenceRef.bundle_relative_path",
    )
    return validate_candidate_reference_ref(
        _candidate_reference_ref_from_bundle(bundle_payload, relative_path)
    )


def validate_candidate_reference_ref(
    payload: Mapping[str, Any],
    *,
    root: str | Path | None = None,
    stage: str | None = None,
    reference_arm: str | None = None,
    expected_row_index: int | None = None,
) -> dict[str, Any]:
    ref = _validate_candidate_reference_ref_shape(payload)
    if stage is not None or reference_arm is not None or expected_row_index is not None:
        if stage is None or reference_arm is None or expected_row_index is None:
            raise Exp007ArtifactError(
                "CandidateReferenceRef canonical path validation requires stage, arm, and row index"
            )
        _require_canonical_reference_bundle_path(
            stage=stage,
            schedule_arm=reference_arm,
            row_index=expected_row_index,
            relative_path=ref["bundle_relative_path"],
            context="CandidateReferenceRef.bundle_relative_path",
        )
    if root is None:
        return ref
    bundle = read_candidate_reference_row_bundle(root, ref["bundle_relative_path"])
    expected = _candidate_reference_ref_from_bundle(
        bundle,
        ref["bundle_relative_path"],
    )
    if ref != expected:
        raise Exp007ArtifactError("CandidateReferenceRef does not match bundle")
    return ref


def build_candidate_reference_manifest(
    *,
    root: str | Path,
    stage: str,
    input_manifest_sha256: str,
    source_closure_fingerprint_sha256: str,
    reference_arm: str,
    bundle_relative_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    stage_value = _require_stage(stage, "CandidateReferenceManifest.stage")
    arm = _require_schedule_arm(reference_arm, "CandidateReferenceManifest.reference_arm")
    _require_reference_arm(stage_value, arm)
    expected_count = protocol.expected_row_count_for_stage(stage_value)
    if bundle_relative_paths is None:
        bundle_relative_paths = [
            reference_bundle_relative_path(stage_value, arm, index)
            for index in range(expected_count)
        ]
    if len(bundle_relative_paths) != expected_count:
        raise Exp007ArtifactError("CandidateReferenceManifest row_count is incomplete")
    entries = []
    for expected_index, relative_path in enumerate(bundle_relative_paths):
        relative_path = validate_bundle_relative_path(relative_path)
        _require_canonical_reference_bundle_path(
            stage=stage_value,
            schedule_arm=arm,
            row_index=expected_index,
            relative_path=relative_path,
            context="CandidateReferenceManifest.bundle_relative_paths[]",
        )
        bundle = read_candidate_reference_row_bundle(root, relative_path)
        if bundle["row_index"] != expected_index:
            raise Exp007ArtifactError("CandidateReferenceManifest row order mismatch")
        row = bundle["row"]
        if row["input_manifest_sha256"] != input_manifest_sha256:
            raise Exp007ArtifactError("CandidateReferenceManifest input SHA mismatch")
        if row["source_closure_fingerprint_sha256"] != source_closure_fingerprint_sha256:
            raise Exp007ArtifactError("CandidateReferenceManifest source SHA mismatch")
        entries.append(
            make_candidate_reference_ref(
                bundle=bundle,
                bundle_relative_path=relative_path,
            )
        )
    payload = {
        "schema": CANDIDATE_REFERENCE_MANIFEST_SCHEMA,
        "experiment_id": protocol.EXP007_EXPERIMENT_ID,
        "stage": stage_value,
        "schema_descriptor_sha256": protocol.schema_descriptor_sha256(
            CANDIDATE_REFERENCE_MANIFEST_SCHEMA
        ),
        "input_manifest_sha256": input_manifest_sha256,
        "source_closure_fingerprint_sha256": source_closure_fingerprint_sha256,
        "reference_arm": arm,
        "row_count": expected_count,
        "entries": entries,
        "ordered_entries_sha256": protocol.canonical_json_sha256(entries),
    }
    return validate_candidate_reference_manifest(
        protocol.with_payload_hash(payload, "manifest_fingerprint_sha256"),
        root=root,
    )


def validate_candidate_reference_manifest(
    payload: Mapping[str, Any],
    *,
    root: str | Path,
) -> dict[str, Any]:
    _require_mapping(payload, "CandidateReferenceManifest")
    protocol.validate_exact_fields(
        payload,
        CANDIDATE_REFERENCE_MANIFEST_FIELDS,
        "CandidateReferenceManifest",
    )
    if payload.get("schema") != CANDIDATE_REFERENCE_MANIFEST_SCHEMA:
        raise Exp007ArtifactError("CandidateReferenceManifest schema is invalid")
    if payload.get("experiment_id") != protocol.EXP007_EXPERIMENT_ID:
        raise Exp007ArtifactError("CandidateReferenceManifest experiment_id is invalid")
    stage = _require_stage(payload.get("stage"), "CandidateReferenceManifest.stage")
    _require_descriptor(
        payload,
        CANDIDATE_REFERENCE_MANIFEST_SCHEMA,
        "CandidateReferenceManifest",
    )
    reference_arm = _require_schedule_arm(
        payload.get("reference_arm"),
        "CandidateReferenceManifest.reference_arm",
    )
    _require_reference_arm(stage, reference_arm)
    expected_count = protocol.expected_row_count_for_stage(stage)
    if payload.get("row_count") != expected_count:
        raise Exp007ArtifactError("CandidateReferenceManifest row_count is invalid")
    input_sha = protocol.require_sha256(
        payload.get("input_manifest_sha256"),
        "CandidateReferenceManifest.input_manifest_sha256",
    )
    source_sha = protocol.require_sha256(
        payload.get("source_closure_fingerprint_sha256"),
        "CandidateReferenceManifest.source_closure_fingerprint_sha256",
    )
    entries_payload = _require_sequence(payload.get("entries"), "entries")
    if len(entries_payload) != expected_count:
        raise Exp007ArtifactError("CandidateReferenceManifest entries length mismatch")
    entries: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_resolved_targets: set[Path] = set()
    seen_cache_audio_keys: set[str] = set()
    for expected_index, item in enumerate(entries_payload):
        ref = validate_candidate_reference_ref(
            item,
            root=root,
            stage=stage,
            reference_arm=reference_arm,
            expected_row_index=expected_index,
        )
        if ref["row_index"] != expected_index:
            raise Exp007ArtifactError("CandidateReferenceManifest row order mismatch")
        if ref["bundle_relative_path"] in seen_paths:
            raise Exp007ArtifactError("CandidateReferenceManifest duplicate path")
        seen_paths.add(ref["bundle_relative_path"])
        resolved = resolve_existing_bundle_path(root, ref["bundle_relative_path"])
        if resolved in seen_resolved_targets:
            raise Exp007ArtifactError("CandidateReferenceManifest duplicate target")
        seen_resolved_targets.add(resolved)
        if ref["cache_audio_key"] in seen_cache_audio_keys:
            raise Exp007ArtifactError("CandidateReferenceManifest duplicate cache key")
        seen_cache_audio_keys.add(ref["cache_audio_key"])
        bundle = read_candidate_reference_row_bundle(root, ref["bundle_relative_path"])
        row = bundle["row"]
        if row["stage"] != stage or row["schedule_arm"] != reference_arm:
            raise Exp007ArtifactError("CandidateReferenceManifest bundle stage/arm mismatch")
        if row["input_manifest_sha256"] != input_sha:
            raise Exp007ArtifactError("CandidateReferenceManifest input SHA mismatch")
        if row["source_closure_fingerprint_sha256"] != source_sha:
            raise Exp007ArtifactError("CandidateReferenceManifest source SHA mismatch")
        entries.append(ref)
    if payload.get("ordered_entries_sha256") != protocol.canonical_json_sha256(entries):
        raise Exp007ArtifactError("CandidateReferenceManifest ordered entries hash mismatch")
    result = {
        "schema": CANDIDATE_REFERENCE_MANIFEST_SCHEMA,
        "experiment_id": protocol.EXP007_EXPERIMENT_ID,
        "stage": stage,
        "schema_descriptor_sha256": payload["schema_descriptor_sha256"],
        "input_manifest_sha256": input_sha,
        "source_closure_fingerprint_sha256": source_sha,
        "reference_arm": reference_arm,
        "row_count": expected_count,
        "entries": entries,
        "ordered_entries_sha256": payload["ordered_entries_sha256"],
        "manifest_fingerprint_sha256": protocol.require_sha256(
            payload.get("manifest_fingerprint_sha256"),
            "CandidateReferenceManifest.manifest_fingerprint_sha256",
        ),
    }
    protocol.validate_payload_hash(
        result,
        "manifest_fingerprint_sha256",
        context="CandidateReferenceManifest",
    )
    canonical_json_bytes_under_cap(
        result,
        byte_cap=protocol.EXP007_CANDIDATE_REFERENCE_MANIFEST_BYTE_CAP,
        context="CandidateReferenceManifest",
    )
    return result


def build_candidate_global_manifest(
    *,
    root: str | Path,
    reference_manifest: Mapping[str, Any],
    selector_manifest_sha256: str,
    source_closure_fingerprint_sha256: str,
    run_configs_by_arm: Mapping[str, Mapping[str, Any]],
    arm_rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_payloads_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    reference = validate_candidate_reference_manifest(reference_manifest, root=root)
    if reference["stage"] != protocol.EXP007_SCHEDULE_STAGE:
        raise Exp007ArtifactError("CandidateGlobalManifest requires schedule reference")
    if reference["reference_arm"] != "S30":
        raise Exp007ArtifactError("CandidateGlobalManifest requires S30 reference")
    if reference["input_manifest_sha256"] != selector_manifest_sha256:
        raise Exp007ArtifactError("CandidateGlobalManifest selector/reference mismatch")
    if reference["source_closure_fingerprint_sha256"] != source_closure_fingerprint_sha256:
        raise Exp007ArtifactError("CandidateGlobalManifest source/reference mismatch")
    run_configs = _validated_schedule_run_configs_by_arm(
        run_configs_by_arm,
        selector_manifest_sha256=selector_manifest_sha256,
        source_closure_fingerprint_sha256=source_closure_fingerprint_sha256,
    )
    _require_exact_arm_sequences(arm_rows_by_arm, "arm_rows_by_arm")
    _require_exact_arm_sequences(candidate_payloads_by_arm, "candidate_payloads_by_arm")
    entries = []
    for index, ref in enumerate(reference["entries"]):
        bundle = read_candidate_reference_row_bundle(root, ref["bundle_relative_path"])
        reference_projection = _schedule_row_cross_arm_projection(
            row=bundle["row"],
            candidate_payload=bundle["entry"]["candidate_payload"],
        )
        arm_row_sha256: dict[str, str] = {}
        for arm in protocol.EXP007_EXECUTION_ORDER:
            rows = _arm_sequence(arm_rows_by_arm, arm, expected_count=16)
            payloads = _arm_sequence(candidate_payloads_by_arm, arm, expected_count=16)
            row = protocol.validate_row_result(rows[index])
            candidate_payload = payloads[index]
            _validate_schedule_arm_row_against_reference(
                row=row,
                arm=arm,
                selector_manifest_sha256=selector_manifest_sha256,
                source_closure_fingerprint_sha256=source_closure_fingerprint_sha256,
                reference_bundle=bundle,
                candidate_payload=candidate_payload,
            )
            _require_schedule_row_run_config_binding(
                row=row,
                arm=arm,
                run_config=run_configs[arm],
            )
            _require_schedule_row_cross_arm_match(
                row=row,
                candidate_payload=candidate_payload,
                reference_projection=reference_projection,
                arm=arm,
            )
            if arm == "S30" and row["row_payload_sha256"] != bundle["row"]["row_payload_sha256"]:
                raise Exp007ArtifactError("schedule S30 row payload differs from reference bundle")
            arm_row_sha256[arm] = row["row_payload_sha256"]
        entries.append(
            _candidate_global_entry_from_reference_bundle(bundle, arm_row_sha256)
        )
    payload = {
        "schema": CANDIDATE_GLOBAL_MANIFEST_SCHEMA,
        "experiment_id": protocol.EXP007_EXPERIMENT_ID,
        "stage": protocol.EXP007_SCHEDULE_STAGE,
        "schema_descriptor_sha256": protocol.schema_descriptor_sha256(
            CANDIDATE_GLOBAL_MANIFEST_SCHEMA
        ),
        "selector_manifest_sha256": selector_manifest_sha256,
        "source_closure_fingerprint_sha256": source_closure_fingerprint_sha256,
        "candidate_reference_manifest_sha256": protocol.object_complete_sha256(reference),
        "row_count": 16,
        "entries": entries,
        "ordered_entries_sha256": protocol.canonical_json_sha256(entries),
    }
    return validate_candidate_global_manifest(
        protocol.with_payload_hash(payload, "manifest_fingerprint_sha256"),
        reference_manifest=reference,
        root=root,
        run_configs_by_arm=run_configs,
        arm_rows_by_arm=arm_rows_by_arm,
        candidate_payloads_by_arm=candidate_payloads_by_arm,
    )


def validate_candidate_global_manifest(
    payload: Mapping[str, Any],
    *,
    reference_manifest: Mapping[str, Any],
    root: str | Path,
    run_configs_by_arm: Mapping[str, Mapping[str, Any]],
    arm_rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_payloads_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    result = _validate_candidate_global_manifest_shape(payload)
    selector_sha = result["selector_manifest_sha256"]
    source_sha = result["source_closure_fingerprint_sha256"]
    reference_sha = result["candidate_reference_manifest_sha256"]
    entries = result["entries"]
    run_configs = _validated_schedule_run_configs_by_arm(
        run_configs_by_arm,
        selector_manifest_sha256=selector_sha,
        source_closure_fingerprint_sha256=source_sha,
    )
    _require_exact_arm_sequences(arm_rows_by_arm, "arm_rows_by_arm")
    _require_exact_arm_sequences(candidate_payloads_by_arm, "candidate_payloads_by_arm")
    reference = validate_candidate_reference_manifest(reference_manifest, root=root)
    if protocol.object_complete_sha256(reference) != reference_sha:
        raise Exp007ArtifactError("CandidateGlobalManifest reference SHA mismatch")
    if reference["stage"] != protocol.EXP007_SCHEDULE_STAGE:
        raise Exp007ArtifactError("CandidateGlobalManifest reference stage mismatch")
    if reference["reference_arm"] != "S30":
        raise Exp007ArtifactError("CandidateGlobalManifest reference arm mismatch")
    if reference["input_manifest_sha256"] != selector_sha:
        raise Exp007ArtifactError("CandidateGlobalManifest selector/reference mismatch")
    if reference["source_closure_fingerprint_sha256"] != source_sha:
        raise Exp007ArtifactError("CandidateGlobalManifest source/reference mismatch")
    for entry, ref in zip(entries, reference["entries"], strict=True):
        reference_bundle = read_candidate_reference_row_bundle(
            root,
            ref["bundle_relative_path"],
        )
        _require_global_entry_matches_reference_bundle(entry, reference_bundle)
        reference_projection = _schedule_row_cross_arm_projection(
            row=reference_bundle["row"],
            candidate_payload=reference_bundle["entry"]["candidate_payload"],
        )
        for arm in protocol.EXP007_EXECUTION_ORDER:
            rows = _arm_sequence(arm_rows_by_arm, arm, expected_count=16)
            payloads = _arm_sequence(
                candidate_payloads_by_arm,
                arm,
                expected_count=16,
            )
            row = protocol.validate_row_result(rows[entry["row_index"]])
            _validate_schedule_arm_row_against_reference(
                row=row,
                arm=arm,
                selector_manifest_sha256=selector_sha,
                source_closure_fingerprint_sha256=source_sha,
                reference_bundle=reference_bundle,
                candidate_payload=payloads[entry["row_index"]],
            )
            _require_schedule_row_run_config_binding(
                row=row,
                arm=arm,
                run_config=run_configs[arm],
            )
            _require_schedule_row_cross_arm_match(
                row=row,
                candidate_payload=payloads[entry["row_index"]],
                reference_projection=reference_projection,
                arm=arm,
            )
            if (
                arm == "S30"
                and row["row_payload_sha256"] != reference_bundle["row"]["row_payload_sha256"]
            ):
                raise Exp007ArtifactError("schedule S30 row payload differs from reference bundle")
            if entry["arm_row_payload_sha256"][arm] != row["row_payload_sha256"]:
                raise Exp007ArtifactError(
                    "CandidateGlobalManifest arm row SHA mismatch"
                )
    return result


def _validate_candidate_global_manifest_shape(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    _require_mapping(payload, "CandidateGlobalManifest")
    protocol.validate_exact_fields(
        payload,
        CANDIDATE_GLOBAL_MANIFEST_FIELDS,
        "CandidateGlobalManifest",
    )
    if payload.get("schema") != CANDIDATE_GLOBAL_MANIFEST_SCHEMA:
        raise Exp007ArtifactError("CandidateGlobalManifest schema is invalid")
    if payload.get("experiment_id") != protocol.EXP007_EXPERIMENT_ID:
        raise Exp007ArtifactError("CandidateGlobalManifest experiment_id is invalid")
    if payload.get("stage") != protocol.EXP007_SCHEDULE_STAGE:
        raise Exp007ArtifactError("CandidateGlobalManifest stage is invalid")
    _require_descriptor(
        payload,
        CANDIDATE_GLOBAL_MANIFEST_SCHEMA,
        "CandidateGlobalManifest",
    )
    selector_sha = protocol.require_sha256(
        payload.get("selector_manifest_sha256"),
        "CandidateGlobalManifest.selector_manifest_sha256",
    )
    source_sha = protocol.require_sha256(
        payload.get("source_closure_fingerprint_sha256"),
        "CandidateGlobalManifest.source_closure_fingerprint_sha256",
    )
    reference_sha = protocol.require_sha256(
        payload.get("candidate_reference_manifest_sha256"),
        "CandidateGlobalManifest.candidate_reference_manifest_sha256",
    )
    if payload.get("row_count") != 16:
        raise Exp007ArtifactError("CandidateGlobalManifest row_count is invalid")
    entries_payload = _require_sequence(payload.get("entries"), "entries")
    if len(entries_payload) != 16:
        raise Exp007ArtifactError("CandidateGlobalManifest entries length mismatch")
    entries = [
        _validate_candidate_global_entry(item, expected_row_index=index)
        for index, item in enumerate(entries_payload)
    ]
    if payload.get("ordered_entries_sha256") != protocol.canonical_json_sha256(entries):
        raise Exp007ArtifactError("CandidateGlobalManifest ordered entries hash mismatch")
    result = {
        "schema": CANDIDATE_GLOBAL_MANIFEST_SCHEMA,
        "experiment_id": protocol.EXP007_EXPERIMENT_ID,
        "stage": protocol.EXP007_SCHEDULE_STAGE,
        "schema_descriptor_sha256": payload["schema_descriptor_sha256"],
        "selector_manifest_sha256": selector_sha,
        "source_closure_fingerprint_sha256": source_sha,
        "candidate_reference_manifest_sha256": reference_sha,
        "row_count": 16,
        "entries": entries,
        "ordered_entries_sha256": payload["ordered_entries_sha256"],
        "manifest_fingerprint_sha256": protocol.require_sha256(
            payload.get("manifest_fingerprint_sha256"),
            "CandidateGlobalManifest.manifest_fingerprint_sha256",
        ),
    }
    protocol.validate_payload_hash(
        result,
        "manifest_fingerprint_sha256",
        context="CandidateGlobalManifest",
    )
    canonical_json_bytes_under_cap(
        result,
        byte_cap=protocol.EXP007_CANDIDATE_GLOBAL_MANIFEST_BYTE_CAP,
        context="CandidateGlobalManifest",
    )
    return result


def build_exposure_entry(
    *,
    cache_audio_key: str,
    audio_group_key: str,
    exposure_stage: str,
    exposure_reason: str,
    first_exposed_at_or_run_id: str,
    observed_payload_kind: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    return validate_exposure_entry(
        {
            "cache_audio_key": cache_audio_key,
            "audio_group_key": audio_group_key,
            "exposure_stage": exposure_stage,
            "exposure_reason": exposure_reason,
            "first_exposed_at_or_run_id": first_exposed_at_or_run_id,
            "observed_payload_kind": observed_payload_kind,
            "source_manifest_sha256": source_manifest_sha256,
        }
    )


def make_exposure_entry(**kwargs: Any) -> dict[str, Any]:
    return build_exposure_entry(**kwargs)


def validate_exposure_entry(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "ExposureEntry")
    protocol.validate_exact_fields(payload, EXPOSURE_ENTRY_FIELDS, "ExposureEntry")
    exposure_stage = _require_exposure_stage(
        payload.get("exposure_stage"),
        "ExposureEntry.exposure_stage",
    )
    observed_payload_kind = _require_observed_payload_kind(
        payload.get("observed_payload_kind"),
        "ExposureEntry.observed_payload_kind",
    )
    return {
        "cache_audio_key": protocol.require_nonempty_string(
            payload.get("cache_audio_key"),
            "ExposureEntry.cache_audio_key",
        ),
        "audio_group_key": protocol.require_nonempty_string(
            payload.get("audio_group_key"),
            "ExposureEntry.audio_group_key",
        ),
        "exposure_stage": exposure_stage,
        "exposure_reason": protocol.require_nonempty_string(
            payload.get("exposure_reason"),
            "ExposureEntry.exposure_reason",
        ),
        "first_exposed_at_or_run_id": protocol.require_nonempty_string(
            payload.get("first_exposed_at_or_run_id"),
            "ExposureEntry.first_exposed_at_or_run_id",
        ),
        "observed_payload_kind": observed_payload_kind,
        "source_manifest_sha256": protocol.require_sha256(
            payload.get("source_manifest_sha256"),
            "ExposureEntry.source_manifest_sha256",
        ),
    }


def build_exposure_delta(
    *,
    generated_at_utc: str,
    prior_exposure_manifest_sha256: str,
    source_closure_fingerprint_sha256: str,
    delta_reason: str,
    entries: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    validated_entries = sorted(
        (validate_exposure_entry(entry) for entry in entries),
        key=lambda entry: entry["cache_audio_key"],
    )
    keys = [entry["cache_audio_key"] for entry in validated_entries]
    _reject_duplicate_cache_audio_keys(keys, "ExposureDelta.entries")
    payload = {
        "schema": EXPOSURE_DELTA_SCHEMA,
        "experiment_id": protocol.EXP007_EXPERIMENT_ID,
        "schema_descriptor_sha256": artifact_schema_descriptor_sha256(
            EXPOSURE_DELTA_SCHEMA
        ),
        "generated_at_utc": _require_utc_marker(
            generated_at_utc,
            "ExposureDelta.generated_at_utc",
        ),
        "prior_exposure_manifest_sha256": protocol.require_sha256(
            prior_exposure_manifest_sha256,
            "ExposureDelta.prior_exposure_manifest_sha256",
        ),
        "source_closure_fingerprint_sha256": protocol.require_sha256(
            source_closure_fingerprint_sha256,
            "ExposureDelta.source_closure_fingerprint_sha256",
        ),
        "delta_reason": protocol.require_nonempty_string(
            delta_reason,
            "ExposureDelta.delta_reason",
        ),
        "entry_count": len(validated_entries),
        "cache_audio_keys_sha256": protocol.canonical_json_sha256(keys),
        "entries_sha256": protocol.canonical_json_sha256(validated_entries),
        "entries": validated_entries,
    }
    return validate_exposure_delta(
        protocol.with_payload_hash(payload, "manifest_fingerprint_sha256")
    )


def validate_exposure_delta(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "ExposureDelta")
    protocol.validate_exact_fields(payload, EXPOSURE_DELTA_FIELDS, "ExposureDelta")
    if payload.get("schema") != EXPOSURE_DELTA_SCHEMA:
        raise Exp007ArtifactError("ExposureDelta schema is invalid")
    if payload.get("experiment_id") != protocol.EXP007_EXPERIMENT_ID:
        raise Exp007ArtifactError("ExposureDelta experiment_id is invalid")
    _require_artifact_descriptor(payload, EXPOSURE_DELTA_SCHEMA, "ExposureDelta")
    entries_payload = _require_sequence(payload.get("entries"), "ExposureDelta.entries")
    entries = [validate_exposure_entry(entry) for entry in entries_payload]
    sorted_entries = sorted(entries, key=lambda entry: entry["cache_audio_key"])
    if entries != sorted_entries:
        raise Exp007ArtifactError("ExposureDelta entries must be sorted by cache_audio_key")
    keys = [entry["cache_audio_key"] for entry in entries]
    _reject_duplicate_cache_audio_keys(keys, "ExposureDelta.entries")
    if payload.get("entry_count") != len(entries):
        raise Exp007ArtifactError("ExposureDelta entry_count mismatch")
    if payload.get("cache_audio_keys_sha256") != protocol.canonical_json_sha256(keys):
        raise Exp007ArtifactError("ExposureDelta cache_audio_keys_sha256 mismatch")
    if payload.get("entries_sha256") != protocol.canonical_json_sha256(entries):
        raise Exp007ArtifactError("ExposureDelta entries_sha256 mismatch")
    result = {
        "schema": EXPOSURE_DELTA_SCHEMA,
        "experiment_id": protocol.EXP007_EXPERIMENT_ID,
        "schema_descriptor_sha256": payload["schema_descriptor_sha256"],
        "generated_at_utc": _require_utc_marker(
            payload.get("generated_at_utc"),
            "ExposureDelta.generated_at_utc",
        ),
        "prior_exposure_manifest_sha256": protocol.require_sha256(
            payload.get("prior_exposure_manifest_sha256"),
            "ExposureDelta.prior_exposure_manifest_sha256",
        ),
        "source_closure_fingerprint_sha256": protocol.require_sha256(
            payload.get("source_closure_fingerprint_sha256"),
            "ExposureDelta.source_closure_fingerprint_sha256",
        ),
        "delta_reason": protocol.require_nonempty_string(
            payload.get("delta_reason"),
            "ExposureDelta.delta_reason",
        ),
        "entry_count": len(entries),
        "cache_audio_keys_sha256": payload["cache_audio_keys_sha256"],
        "entries_sha256": payload["entries_sha256"],
        "entries": entries,
        "manifest_fingerprint_sha256": protocol.require_sha256(
            payload.get("manifest_fingerprint_sha256"),
            "ExposureDelta.manifest_fingerprint_sha256",
        ),
    }
    protocol.validate_payload_hash(
        result,
        "manifest_fingerprint_sha256",
        context="ExposureDelta",
    )
    canonical_json_bytes_under_cap(
        result,
        byte_cap=EXPOSURE_DELTA_BYTE_CAP,
        context="ExposureDelta",
    )
    return result


def validate_exp007_empty_exposure_delta(payload: Mapping[str, Any]) -> dict[str, Any]:
    delta = validate_exposure_delta(payload)
    if delta["entry_count"] != 0:
        raise Exp007ArtifactError("Exp007 no-real-data exposure delta must be empty")
    return delta


def publish_exposure_delta(
    *,
    root: str | Path,
    delta: Mapping[str, Any],
    run_lock: Exp007RunLock,
    relative_path: str | None = None,
    require_empty: bool = True,
) -> str:
    if require_empty is not True:
        raise Exp007ArtifactError("Exp007 exposure delta publication is empty-only")
    lock_payload = _require_live_held_run_lock_for_root(run_lock, root=root)
    delta_payload = validate_exposure_delta(delta)
    _require_exposure_delta_bound_to_run_lock(delta_payload, lock_payload)
    delta_payload = validate_exp007_empty_exposure_delta(delta_payload)
    if relative_path is None:
        relative_path = exposure_delta_relative_path()
    destination = contained_write_path(root, relative_path)
    write_json_atomic(
        destination,
        delta_payload,
        byte_cap=EXPOSURE_DELTA_BYTE_CAP,
        context="ExposureDelta",
        root=root,
        relative_path=relative_path,
    )
    return validate_bundle_relative_path(relative_path)


def read_exposure_delta(
    root: str | Path,
    relative_path: str | None = None,
    *,
    require_empty: bool = False,
) -> dict[str, Any]:
    if relative_path is None:
        relative_path = exposure_delta_relative_path()
    payload = read_json_artifact_anchored(
        root,
        relative_path,
        byte_cap=EXPOSURE_DELTA_BYTE_CAP,
        context="ExposureDelta",
    )
    if require_empty:
        return validate_exp007_empty_exposure_delta(payload)
    return validate_exposure_delta(payload)


def resume_exposure_delta(
    root: str | Path,
    relative_path: str | None = None,
    *,
    require_empty: bool = False,
) -> dict[str, Any] | None:
    if relative_path is None:
        relative_path = exposure_delta_relative_path()
    if not _contained_existing_path_or_none(root, relative_path):
        return None
    return read_exposure_delta(root, relative_path, require_empty=require_empty)


def build_run_lock_payload(
    *,
    root: str | Path,
    stage: str,
    schedule_arm: str,
    selector_manifest_sha256: str,
    source_closure_fingerprint_sha256: str,
    run_config_fingerprint_sha256: str,
    owner_run_id: str,
    acquired_at_utc: str,
) -> dict[str, Any]:
    payload = {
        "schema": RUN_LOCK_SCHEMA,
        "experiment_id": protocol.EXP007_EXPERIMENT_ID,
        "stage": _require_stage(stage, "RunLock.stage"),
        "schema_descriptor_sha256": artifact_schema_descriptor_sha256(RUN_LOCK_SCHEMA),
        "schedule_arm": _require_schedule_arm(schedule_arm, "RunLock.schedule_arm"),
        "selector_manifest_sha256": protocol.require_sha256(
            selector_manifest_sha256,
            "RunLock.selector_manifest_sha256",
        ),
        "source_closure_fingerprint_sha256": protocol.require_sha256(
            source_closure_fingerprint_sha256,
            "RunLock.source_closure_fingerprint_sha256",
        ),
        "run_config_fingerprint_sha256": protocol.require_sha256(
            run_config_fingerprint_sha256,
            "RunLock.run_config_fingerprint_sha256",
        ),
        "output_root_fingerprint_sha256": output_root_fingerprint_sha256(root),
        "owner_run_id": protocol.require_nonempty_string(
            owner_run_id,
            "RunLock.owner_run_id",
        ),
        "acquired_at_utc": _require_utc_marker(
            acquired_at_utc,
            "RunLock.acquired_at_utc",
        ),
    }
    return validate_run_lock_payload(
        protocol.with_payload_hash(payload, "lock_payload_sha256"),
        root=root,
    )


def validate_run_lock_payload(
    payload: Mapping[str, Any],
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    _require_mapping(payload, "RunLock")
    protocol.validate_exact_fields(payload, RUN_LOCK_FIELDS, "RunLock")
    if payload.get("schema") != RUN_LOCK_SCHEMA:
        raise Exp007ArtifactError("RunLock schema is invalid")
    if payload.get("experiment_id") != protocol.EXP007_EXPERIMENT_ID:
        raise Exp007ArtifactError("RunLock experiment_id is invalid")
    _require_artifact_descriptor(payload, RUN_LOCK_SCHEMA, "RunLock")
    result = {
        "schema": RUN_LOCK_SCHEMA,
        "experiment_id": protocol.EXP007_EXPERIMENT_ID,
        "stage": _require_stage(payload.get("stage"), "RunLock.stage"),
        "schema_descriptor_sha256": payload["schema_descriptor_sha256"],
        "schedule_arm": _require_schedule_arm(
            payload.get("schedule_arm"),
            "RunLock.schedule_arm",
        ),
        "selector_manifest_sha256": protocol.require_sha256(
            payload.get("selector_manifest_sha256"),
            "RunLock.selector_manifest_sha256",
        ),
        "source_closure_fingerprint_sha256": protocol.require_sha256(
            payload.get("source_closure_fingerprint_sha256"),
            "RunLock.source_closure_fingerprint_sha256",
        ),
        "run_config_fingerprint_sha256": protocol.require_sha256(
            payload.get("run_config_fingerprint_sha256"),
            "RunLock.run_config_fingerprint_sha256",
        ),
        "output_root_fingerprint_sha256": protocol.require_sha256(
            payload.get("output_root_fingerprint_sha256"),
            "RunLock.output_root_fingerprint_sha256",
        ),
        "owner_run_id": protocol.require_nonempty_string(
            payload.get("owner_run_id"),
            "RunLock.owner_run_id",
        ),
        "acquired_at_utc": _require_utc_marker(
            payload.get("acquired_at_utc"),
            "RunLock.acquired_at_utc",
        ),
        "lock_payload_sha256": protocol.require_sha256(
            payload.get("lock_payload_sha256"),
            "RunLock.lock_payload_sha256",
        ),
    }
    if root is not None:
        expected_root = output_root_fingerprint_sha256(root)
        if result["output_root_fingerprint_sha256"] != expected_root:
            raise Exp007ArtifactError("RunLock output root fingerprint mismatch")
    protocol.validate_payload_hash(
        result,
        "lock_payload_sha256",
        context="RunLock",
    )
    canonical_json_bytes_under_cap(
        result,
        byte_cap=RUN_LOCK_BYTE_CAP,
        context="RunLock",
    )
    return result


def acquire_run_lock(
    *,
    root: str | Path,
    stage: str,
    schedule_arm: str,
    selector_manifest_sha256: str,
    source_closure_fingerprint_sha256: str,
    run_config_fingerprint_sha256: str,
    owner_run_id: str,
    acquired_at_utc: str,
    relative_path: str | None = None,
) -> Exp007RunLock:
    expected_relative_path = run_lock_relative_path(stage, schedule_arm)
    if relative_path is None:
        relative_path = expected_relative_path
    elif validate_bundle_relative_path(relative_path) != expected_relative_path:
        raise Exp007ArtifactError("run lock relative path must be canonical")
    destination = contained_write_path(root, relative_path)
    payload = build_run_lock_payload(
        root=root,
        stage=stage,
        schedule_arm=schedule_arm,
        selector_manifest_sha256=selector_manifest_sha256,
        source_closure_fingerprint_sha256=source_closure_fingerprint_sha256,
        run_config_fingerprint_sha256=run_config_fingerprint_sha256,
        owner_run_id=owner_run_id,
        acquired_at_utc=acquired_at_utc,
    )
    canonical = canonical_json_bytes_under_cap(
        payload,
        byte_cap=RUN_LOCK_BYTE_CAP,
        context="RunLock",
    )
    anchor = _open_anchored_destination_parent(
        destination,
        root_resolved=Path(root).resolve(strict=True),
        context="RunLock",
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    file_descriptor: int | None = None
    try:
        _run_dirfd_test_hook("before_lock_create", destination)
        _require_anchor_parent_current(anchor)
        file_descriptor = os.open(
            anchor.name,
            flags,
            0o600,
            dir_fd=anchor.parent_fd,
        )
        with os.fdopen(file_descriptor, "wb", closefd=True) as handle:
            file_descriptor = None
            handle.write(canonical)
            handle.flush()
            os.fsync(handle.fileno())
        _require_anchor_parent_current(anchor)
        _fsync_directory_fd(anchor.parent_fd)
    except FileExistsError as exc:
        _raise_existing_run_lock_error(anchor, canonical)
        raise Exp007ArtifactError("run lock already exists") from exc
    except OSError as exc:
        raise Exp007ArtifactError("run lock acquisition failed") from exc
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        os.close(anchor.parent_fd)
    return Exp007RunLock(
        root=Path(root),
        relative_path=validate_bundle_relative_path(relative_path),
        payload=payload,
        acquired=True,
    )


def read_run_lock(
    root: str | Path,
    relative_path: str | None = None,
) -> Exp007RunLock:
    if relative_path is None:
        relative_path = _default_existing_run_lock_relative_path(root)
    payload = read_json_artifact_anchored(
        root,
        relative_path,
        byte_cap=RUN_LOCK_BYTE_CAP,
        context="RunLock",
    )
    return Exp007RunLock(
        root=Path(root),
        relative_path=validate_bundle_relative_path(relative_path),
        payload=validate_run_lock_payload(payload, root=root),
        acquired=False,
    )


def resume_run_lock(
    root: str | Path,
    relative_path: str | None = None,
) -> Exp007RunLock | None:
    if relative_path is None:
        relative_path = _default_existing_run_lock_relative_path(root, missing_ok=True)
        if relative_path is None:
            return None
    if not _contained_existing_path_or_none(root, relative_path):
        return None
    return read_run_lock(root, relative_path)


def release_run_lock(
    lock: Exp007RunLock,
    *,
    owner_run_id: str | None = None,
) -> None:
    if owner_run_id is None:
        raise Exp007ArtifactError("explicit owner token is required to release run lock")
    owner = protocol.require_nonempty_string(owner_run_id, "RunLock.owner_run_id")
    current_payload, path, opened_metadata = _require_live_held_run_lock_snapshot(
        lock,
        root=lock.root,
    )
    if current_payload["owner_run_id"] != owner:
        raise Exp007ArtifactError("run lock owner mismatch")
    anchor = _open_anchored_destination_parent(
        path,
        root_resolved=Path(lock.root).resolve(strict=True),
        context="RunLock",
    )
    try:
        current_metadata = _stat_existing_entry_dirfd(
            anchor,
            missing_ok=False,
            context="RunLock",
            symlink_message="run lock destination symlink is forbidden",
        )
        if not _same_file_identity(opened_metadata, current_metadata):
            raise Exp007ArtifactError("run lock changed before release")
        _run_dirfd_test_hook("before_lock_unlink", path)
        _require_anchor_parent_current(anchor)
        try:
            os.unlink(anchor.name, dir_fd=anchor.parent_fd)
        except FileNotFoundError as exc:
            raise Exp007ArtifactError("run lock is missing") from exc
        _fsync_directory_fd(anchor.parent_fd)
    finally:
        os.close(anchor.parent_fd)


def output_root_fingerprint_sha256(root: str | Path) -> str:
    root_path = Path(root)
    if _is_symlink_path(root_path):
        raise Exp007ArtifactError("artifact root symlink is forbidden")
    root_path.mkdir(parents=True, exist_ok=True)
    if _is_symlink_path(root_path):
        raise Exp007ArtifactError("artifact root symlink is forbidden")
    return protocol.canonical_json_sha256(
        {
            "experiment_id": protocol.EXP007_EXPERIMENT_ID,
            "output_root_resolved": str(root_path.resolve(strict=True)),
        }
    )


def exposure_delta_relative_path() -> str:
    return f"exposure/{protocol.EXP007_EXPERIMENT_ID}/exposure_delta.json"


def run_lock_relative_path(stage: str, schedule_arm: str) -> str:
    stage_value = _require_stage(stage, "RunLock.stage")
    arm = _require_schedule_arm(schedule_arm, "RunLock.schedule_arm")
    return f"locks/{protocol.EXP007_EXPERIMENT_ID}/{stage_value}/{arm}.lock.json"


def publish_candidate_reference_row_bundle(
    *,
    root: str | Path,
    bundle: Mapping[str, Any],
    relative_path: str | None = None,
) -> str:
    bundle_payload = validate_candidate_reference_row_bundle(bundle)
    expected_relative_path = reference_bundle_relative_path(
        bundle_payload["stage"],
        bundle_payload["schedule_arm"],
        bundle_payload["row_index"],
    )
    if relative_path is None:
        relative_path = expected_relative_path
    elif validate_bundle_relative_path(relative_path) != expected_relative_path:
        raise Exp007ArtifactError(
            "CandidateReferenceRowBundle relative path must be canonical"
        )
    destination = contained_write_path(root, relative_path)
    write_json_atomic(
        destination,
        bundle_payload,
        byte_cap=protocol.EXP007_CANDIDATE_BUNDLE_BYTE_CAP,
        context="CandidateReferenceRowBundle",
        root=root,
        relative_path=relative_path,
    )
    return validate_bundle_relative_path(relative_path)


def publish_candidate_reference_manifest(
    *,
    root: str | Path,
    manifest: Mapping[str, Any],
    relative_path: str | None = None,
) -> str:
    manifest_payload = validate_candidate_reference_manifest(manifest, root=root)
    expected_relative_path = candidate_reference_manifest_relative_path(
        manifest_payload["stage"],
        manifest_payload["reference_arm"],
    )
    if relative_path is None:
        relative_path = expected_relative_path
    elif validate_bundle_relative_path(relative_path) != expected_relative_path:
        raise Exp007ArtifactError(
            "CandidateReferenceManifest relative path must be canonical"
        )
    destination = contained_write_path(root, relative_path)
    write_json_atomic(
        destination,
        manifest_payload,
        byte_cap=protocol.EXP007_CANDIDATE_REFERENCE_MANIFEST_BYTE_CAP,
        context="CandidateReferenceManifest",
        root=root,
        relative_path=relative_path,
    )
    return validate_bundle_relative_path(relative_path)


def publish_candidate_global_manifest(
    *,
    root: str | Path,
    manifest: Mapping[str, Any],
    reference_manifest: Mapping[str, Any],
    run_configs_by_arm: Mapping[str, Mapping[str, Any]],
    arm_rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_payloads_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    relative_path: str | None = None,
) -> str:
    manifest_payload = validate_candidate_global_manifest(
        manifest,
        reference_manifest=reference_manifest,
        root=root,
        run_configs_by_arm=run_configs_by_arm,
        arm_rows_by_arm=arm_rows_by_arm,
        candidate_payloads_by_arm=candidate_payloads_by_arm,
    )
    expected_relative_path = candidate_global_manifest_relative_path()
    if relative_path is None:
        relative_path = expected_relative_path
    elif validate_bundle_relative_path(relative_path) != expected_relative_path:
        raise Exp007ArtifactError(
            "CandidateGlobalManifest relative path must be canonical"
        )
    destination = contained_write_path(root, relative_path)
    write_json_atomic(
        destination,
        manifest_payload,
        byte_cap=protocol.EXP007_CANDIDATE_GLOBAL_MANIFEST_BYTE_CAP,
        context="CandidateGlobalManifest",
        root=root,
        relative_path=relative_path,
    )
    return validate_bundle_relative_path(relative_path)


def publish_row_result(
    *,
    root: str | Path,
    row: Mapping[str, Any],
    relative_path: str | None = None,
) -> str:
    row_payload = protocol.validate_row_result(row)
    expected_relative_path = row_result_relative_path(
        row_payload["stage"],
        row_payload["schedule_arm"],
        row_payload["row_index"],
    )
    if relative_path is None:
        relative_path = expected_relative_path
    elif validate_bundle_relative_path(relative_path) != expected_relative_path:
        raise Exp007ArtifactError("RowResult relative path must be canonical")
    destination = contained_write_path(root, relative_path)
    write_json_atomic(
        destination,
        row_payload,
        byte_cap=protocol.EXP007_ROW_JSON_BYTE_CAP,
        context="RowResult",
        root=root,
        relative_path=relative_path,
    )
    return validate_bundle_relative_path(relative_path)


def read_candidate_reference_row_bundle(
    root: str | Path,
    relative_path: str,
) -> dict[str, Any]:
    payload = read_json_artifact_anchored(
        root,
        relative_path,
        byte_cap=protocol.EXP007_CANDIDATE_BUNDLE_BYTE_CAP,
        context="CandidateReferenceRowBundle",
    )
    bundle = validate_candidate_reference_row_bundle(payload)
    _require_canonical_reference_bundle_path(
        stage=bundle["stage"],
        schedule_arm=bundle["schedule_arm"],
        row_index=bundle["row_index"],
        relative_path=relative_path,
        context="CandidateReferenceRowBundle read path",
    )
    return bundle


def read_candidate_reference_manifest(
    root: str | Path,
    relative_path: str,
) -> dict[str, Any]:
    payload = read_json_artifact_anchored(
        root,
        relative_path,
        byte_cap=protocol.EXP007_CANDIDATE_REFERENCE_MANIFEST_BYTE_CAP,
        context="CandidateReferenceManifest",
    )
    manifest = validate_candidate_reference_manifest(payload, root=root)
    expected_relative_path = candidate_reference_manifest_relative_path(
        manifest["stage"],
        manifest["reference_arm"],
    )
    if validate_bundle_relative_path(relative_path) != expected_relative_path:
        raise Exp007ArtifactError(
            "CandidateReferenceManifest read path must be canonical"
        )
    return manifest


def read_candidate_global_manifest(
    root: str | Path,
    relative_path: str,
    *,
    reference_manifest: Mapping[str, Any],
    run_configs_by_arm: Mapping[str, Mapping[str, Any]],
    arm_rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    candidate_payloads_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    if validate_bundle_relative_path(relative_path) != candidate_global_manifest_relative_path():
        raise Exp007ArtifactError("CandidateGlobalManifest read path must be canonical")
    payload = read_json_artifact_anchored(
        root,
        relative_path,
        byte_cap=protocol.EXP007_CANDIDATE_GLOBAL_MANIFEST_BYTE_CAP,
        context="CandidateGlobalManifest",
    )
    return validate_candidate_global_manifest(
        payload,
        reference_manifest=reference_manifest,
        root=root,
        run_configs_by_arm=run_configs_by_arm,
        arm_rows_by_arm=arm_rows_by_arm,
        candidate_payloads_by_arm=candidate_payloads_by_arm,
    )


def read_row_result(root: str | Path, relative_path: str) -> dict[str, Any]:
    payload = read_json_artifact_anchored(
        root,
        relative_path,
        byte_cap=protocol.EXP007_ROW_JSON_BYTE_CAP,
        context="RowResult",
    )
    row = protocol.validate_row_result(payload)
    expected_relative_path = row_result_relative_path(
        row["stage"],
        row["schedule_arm"],
        row["row_index"],
    )
    if validate_bundle_relative_path(relative_path) != expected_relative_path:
        raise Exp007ArtifactError("RowResult read path must be canonical")
    return row


def validate_reference_arm_bundle_prefix(
    *,
    root: str | Path,
    stage: str,
    schedule_arm: str,
    input_manifest_sha256: str,
    source_closure_fingerprint_sha256: str,
) -> tuple[dict[str, Any], ...]:
    stage_value = _require_stage(stage, "stage")
    arm = _require_schedule_arm(schedule_arm, "schedule_arm")
    _require_reference_arm(stage_value, arm)
    input_sha = protocol.require_sha256(input_manifest_sha256, "input_manifest_sha256")
    source_sha = protocol.require_sha256(
        source_closure_fingerprint_sha256,
        "source_closure_fingerprint_sha256",
    )
    expected_count = protocol.expected_row_count_for_stage(stage_value)
    prefix: list[dict[str, Any]] = []
    missing_seen = False
    for index in range(expected_count):
        bundle_rel = reference_bundle_relative_path(stage_value, arm, index)
        row_rel = row_result_relative_path(stage_value, arm, index)
        bundle_exists = contained_path_exists(root, bundle_rel)
        row_exists = contained_path_exists(root, row_rel)
        if row_exists:
            raise Exp007ArtifactError("reference-arm row-only prefix is invalid")
        if not bundle_exists:
            missing_seen = True
            continue
        if missing_seen:
            raise Exp007ArtifactError("reference-arm bundle prefix is gapped")
        bundle = read_candidate_reference_row_bundle(root, bundle_rel)
        row = bundle["row"]
        if row["row_index"] != index:
            raise Exp007ArtifactError("reference-arm bundle row order mismatch")
        if row["input_manifest_sha256"] != input_sha:
            raise Exp007ArtifactError("reference-arm bundle input SHA mismatch")
        if row["source_closure_fingerprint_sha256"] != source_sha:
            raise Exp007ArtifactError("reference-arm bundle source SHA mismatch")
        prefix.append(bundle)
    return tuple(prefix)


def validate_later_arm_row_prefix(
    *,
    root: str | Path,
    schedule_arm: str,
    candidate_payloads_by_row_index: Mapping[int, Mapping[str, Any]]
    | Sequence[Mapping[str, Any]],
    s30_reference_manifest_relative_path: str | None = None,
) -> tuple[dict[str, Any], ...]:
    arm = _require_schedule_arm(schedule_arm, "schedule_arm")
    if arm == "S30":
        raise Exp007ArtifactError("later-arm resume cannot use S30")
    if s30_reference_manifest_relative_path is None:
        s30_reference_manifest_relative_path = candidate_reference_manifest_relative_path(
            protocol.EXP007_SCHEDULE_STAGE,
            "S30",
        )
    reference = read_candidate_reference_manifest(
        root,
        s30_reference_manifest_relative_path,
    )
    if reference["stage"] != protocol.EXP007_SCHEDULE_STAGE:
        raise Exp007ArtifactError("later-arm prefix requires schedule reference")
    prefix: list[dict[str, Any]] = []
    missing_seen = False
    for index, ref in enumerate(reference["entries"]):
        row_rel = row_result_relative_path(protocol.EXP007_SCHEDULE_STAGE, arm, index)
        row_exists = contained_path_exists(root, row_rel)
        if not row_exists:
            missing_seen = True
            continue
        if missing_seen:
            raise Exp007ArtifactError("later-arm row prefix is gapped")
        row = read_row_result(root, row_rel)
        candidate_payload = _candidate_payload_for_index(
            candidate_payloads_by_row_index,
            index,
        )
        bundle = read_candidate_reference_row_bundle(root, ref["bundle_relative_path"])
        _validate_schedule_arm_row_against_reference(
            row=row,
            arm=arm,
            selector_manifest_sha256=reference["input_manifest_sha256"],
            source_closure_fingerprint_sha256=reference[
                "source_closure_fingerprint_sha256"
            ],
            reference_bundle=bundle,
            candidate_payload=candidate_payload,
        )
        prefix.append(row)
    return tuple(prefix)


def reference_bundle_relative_path(stage: str, schedule_arm: str, row_index: int) -> str:
    stage_value = _require_stage(stage, "stage")
    arm = _require_schedule_arm(schedule_arm, "schedule_arm")
    index = protocol.require_nonnegative_int(row_index, "row_index")
    return f"candidate_reference/{stage_value}/{arm}/row-{index:05d}.bundle.json"


def candidate_reference_manifest_relative_path(stage: str, reference_arm: str) -> str:
    stage_value = _require_stage(stage, "stage")
    arm = _require_schedule_arm(reference_arm, "reference_arm")
    return f"candidate_reference/{stage_value}/{arm}/manifest.json"


def candidate_global_manifest_relative_path() -> str:
    return "candidate_reference/schedule16/candidate_global_manifest.json"


def row_result_relative_path(stage: str, schedule_arm: str, row_index: int) -> str:
    stage_value = _require_stage(stage, "stage")
    arm = _require_schedule_arm(schedule_arm, "schedule_arm")
    index = protocol.require_nonnegative_int(row_index, "row_index")
    return f"rows/{stage_value}/{arm}/row-{index:05d}.json"


def validate_bundle_relative_path(relative_path: str | Path) -> str:
    if not isinstance(relative_path, (str, Path)):
        raise Exp007ArtifactError("relative path must be a string")
    text = relative_path.as_posix() if isinstance(relative_path, Path) else str(relative_path)
    if "\\" in text:
        raise Exp007ArtifactError("relative path contains unsupported separators")
    parsed = PurePosixPath(text)
    if parsed.is_absolute() or text.startswith("/"):
        raise Exp007ArtifactError("relative path must not be absolute")
    parts = parsed.parts
    if not parts:
        raise Exp007ArtifactError("relative path must not be empty")
    if any(part in {"", ".", ".."} for part in parts):
        raise Exp007ArtifactError("relative path contains an alias component")
    if parsed.as_posix() != text:
        raise Exp007ArtifactError("relative path must be normalized")
    return text


def contained_path_exists(root: str | Path, relative_path: str) -> bool:
    path = contained_read_candidate_path(root, relative_path, must_exist=False)
    return path.exists()


def resolve_existing_bundle_path(root: str | Path, relative_path: str) -> Path:
    return contained_read_candidate_path(root, relative_path, must_exist=True)


def contained_read_candidate_path(
    root: str | Path,
    relative_path: str,
    *,
    must_exist: bool,
) -> Path:
    root_path = Path(root)
    root_resolved = root_path.resolve(strict=True)
    relative = validate_bundle_relative_path(relative_path)
    path = root_path.joinpath(*PurePosixPath(relative).parts)
    _reject_symlink_components(root_path, PurePosixPath(relative).parts)
    if must_exist:
        resolved = path.resolve(strict=True)
        _require_contained(resolved, root_resolved)
    return path


def contained_write_path(root: str | Path, relative_path: str) -> Path:
    relative = validate_bundle_relative_path(relative_path)
    parts = PurePosixPath(relative).parts
    root_path = Path(root)
    if _is_symlink_path(root_path):
        raise Exp007ArtifactError("artifact root symlink is forbidden")
    root_path.mkdir(parents=True, exist_ok=True)
    if _is_symlink_path(root_path):
        raise Exp007ArtifactError("artifact root symlink is forbidden")
    root_resolved = root_path.resolve(strict=True)
    path = root_path.joinpath(*parts)
    current = root_path
    for part in parts[:-1]:
        current = current / part
        _ensure_directory_component(current, root_resolved=root_resolved)
    _require_contained(path.parent.resolve(strict=True), root_resolved)
    _reject_existing_symlink(path, "destination symlink is forbidden")
    return path


def canonical_json_bytes_under_cap(
    payload: Any,
    *,
    byte_cap: int,
    context: str,
) -> bytes:
    cap = protocol.require_positive_int(byte_cap, f"{context}.byte_cap")
    canonical = protocol.canonical_json_bytes(payload)
    if len(canonical) >= cap:
        raise Exp007ArtifactError(f"{context} at or above byte cap")
    return canonical


def read_json_artifact(path: str | Path, *, byte_cap: int, context: str) -> Any:
    artifact_path = Path(path)
    raw = artifact_path.read_bytes()
    if len(raw) >= byte_cap:
        raise Exp007ArtifactError(f"{context} at or above byte cap")
    payload = protocol.load_json_strict(raw)
    if protocol.canonical_json_bytes(payload) != raw:
        raise Exp007ArtifactError(f"{context} is not canonical JSON")
    return payload


def read_json_artifact_anchored(
    root: str | Path,
    relative_path: str,
    *,
    byte_cap: int,
    context: str,
) -> Any:
    root_path = Path(root)
    path = contained_read_candidate_path(root_path, relative_path, must_exist=False)
    try:
        anchor = _open_anchored_destination_parent(
            path,
            root_resolved=root_path.resolve(strict=True),
            context=context,
        )
    except Exp007ArtifactError as exc:
        if str(exc) == "destination parent is missing":
            raise FileNotFoundError(f"No such file: {context} is missing") from exc
        raise
    try:
        _run_dirfd_test_hook("before_anchored_read", path)
        try:
            raw, _ = _read_existing_regular_file_bytes_snapshot_dirfd(
                anchor,
                byte_cap=byte_cap,
                context=context,
            )
        except Exp007ArtifactError as exc:
            if str(exc) == f"{context} is missing":
                raise FileNotFoundError(f"No such file: {context} is missing") from exc
            raise
        _run_dirfd_test_hook("before_anchored_read_parent_check", path)
        _require_anchor_parent_current(anchor)
    finally:
        os.close(anchor.parent_fd)
    payload = protocol.load_json_strict(raw)
    if protocol.canonical_json_bytes(payload) != raw:
        raise Exp007ArtifactError(f"{context} is not canonical JSON")
    return payload


def write_json_atomic(
    path: str | Path,
    payload: Any,
    *,
    byte_cap: int,
    context: str,
    root: str | Path | None = None,
    relative_path: str | None = None,
) -> None:
    destination = Path(path)
    canonical = canonical_json_bytes_under_cap(
        payload,
        byte_cap=byte_cap,
        context=context,
    )
    root_resolved: Path | None = None
    if (root is None) != (relative_path is None):
        raise Exp007ArtifactError("atomic write root and relative path must be paired")
    if root is not None and relative_path is not None:
        expected_destination = contained_write_path(root, relative_path)
        if expected_destination != destination:
            raise Exp007ArtifactError("atomic write destination mismatch")
        root_resolved = Path(root).resolve(strict=True)
    anchor = _open_anchored_destination_parent(
        destination,
        root_resolved=root_resolved,
        context=context,
    )
    temp_name = f".{destination.name}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    file_descriptor: int | None = None
    try:
        if _existing_destination_matches_dirfd(anchor, canonical):
            _fsync_directory_fd(anchor.parent_fd)
            return
        _run_dirfd_test_hook("before_temp_create", destination)
        _require_anchor_parent_current(anchor)
        file_descriptor = os.open(temp_name, flags, 0o600, dir_fd=anchor.parent_fd)
        with os.fdopen(file_descriptor, "wb", closefd=True) as handle:
            file_descriptor = None
            handle.write(canonical)
            handle.flush()
            os.fsync(handle.fileno())
        _run_dirfd_test_hook("before_existing_recheck", destination)
        _require_anchor_parent_current(anchor)
        if _existing_destination_matches_dirfd(anchor, canonical):
            _fsync_directory_fd(anchor.parent_fd)
            return
        _run_dirfd_test_hook("before_atomic_link", destination)
        _require_anchor_parent_current(anchor)
        try:
            os.link(
                temp_name,
                anchor.name,
                src_dir_fd=anchor.parent_fd,
                dst_dir_fd=anchor.parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            if _existing_destination_matches_dirfd(anchor, canonical):
                _fsync_directory_fd(anchor.parent_fd)
                return
            raise
        _require_anchor_parent_current(anchor)
        _fsync_directory_fd(anchor.parent_fd)
    except OSError as exc:
        raise Exp007ArtifactError("atomic publication failure") from exc
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        try:
            os.unlink(temp_name, dir_fd=anchor.parent_fd)
        except FileNotFoundError:
            pass
        except OSError:
            pass
        os.close(anchor.parent_fd)


def _candidate_reference_ref_from_bundle(
    bundle: Mapping[str, Any],
    bundle_relative_path: str,
) -> dict[str, Any]:
    entry = validate_candidate_reference_row_bundle(bundle)["entry"]
    return {
        "row_index": entry["row_index"],
        "cache_audio_key": entry["cache_audio_key"],
        "audio_group_key": entry["audio_group_key"],
        "input_signal_sha256": entry["input_signal_sha256"],
        "entry_payload_sha256": entry["entry_payload_sha256"],
        "candidate_payload_byte_count": entry["candidate_payload_byte_count"],
        "candidate_payload_sha256": entry["candidate_payload_sha256"],
        "candidate_fingerprint": entry["candidate_fingerprint"],
        "bundle_relative_path": bundle_relative_path,
        "bundle_fingerprint_sha256": bundle["bundle_fingerprint_sha256"],
    }


def _validate_materialized_peak(payload: Any, *, context: str) -> dict[str, Any]:
    _require_mapping(payload, context)
    protocol.validate_exact_fields(payload, MATERIALIZED_PEAK_FIELDS, context)
    confidence = protocol.require_finite_number(payload.get("confidence"), f"{context}.confidence")
    if confidence < 0.0 or confidence > 1.0:
        raise Exp007ArtifactError(f"{context}.confidence must be in [0, 1]")
    return {
        "frame_index": protocol.require_nonnegative_int(
            payload.get("frame_index"),
            f"{context}.frame_index",
        ),
        "refined_frame": protocol.require_finite_number(
            payload.get("refined_frame"),
            f"{context}.refined_frame",
        ),
        "time_ms": protocol.require_finite_number(
            payload.get("time_ms"),
            f"{context}.time_ms",
        ),
        "confidence": confidence,
    }


def _validate_tempo_candidate(payload: Any) -> dict[str, Any]:
    _require_mapping(payload, "TempoCandidate")
    protocol.validate_exact_fields(payload, TEMPO_CANDIDATE_FIELDS, "TempoCandidate")
    source = payload.get("source")
    if source not in {"autocorrelation", "peak_interval"}:
        raise Exp007ArtifactError("TempoCandidate.source is invalid")
    return {
        "bpm": _require_bpm(payload.get("bpm"), "TempoCandidate.bpm"),
        "source": source,
        "score": protocol.require_finite_number(
            payload.get("score"),
            "TempoCandidate.score",
        ),
    }


def _validate_origin_candidate(payload: Any) -> dict[str, Any]:
    _require_mapping(payload, "OriginCandidate")
    protocol.validate_exact_fields(payload, ORIGIN_CANDIDATE_FIELDS, "OriginCandidate")
    return {
        "anchor_id": protocol.require_nonnegative_int(
            payload.get("anchor_id"),
            "OriginCandidate.anchor_id",
        ),
        "time_ms": protocol.require_finite_number(
            payload.get("time_ms"),
            "OriginCandidate.time_ms",
        ),
        "bpm": _require_bpm(payload.get("bpm"), "OriginCandidate.bpm"),
        "score": protocol.require_finite_number(
            payload.get("score"),
            "OriginCandidate.score",
        ),
    }


def _validate_boundary_candidate(payload: Any) -> dict[str, Any]:
    _require_mapping(payload, "BoundaryCandidate")
    protocol.validate_exact_fields(
        payload,
        BOUNDARY_CANDIDATE_FIELDS,
        "BoundaryCandidate",
    )
    evidence_mode = payload.get("evidence_mode")
    if evidence_mode not in {"ordinary", "super"}:
        raise Exp007ArtifactError("BoundaryCandidate.evidence_mode is invalid")
    source_peak_confidence = protocol.require_finite_number(
        payload.get("source_peak_confidence"),
        "BoundaryCandidate.source_peak_confidence",
    )
    if source_peak_confidence < 0.0 or source_peak_confidence > 1.0:
        raise Exp007ArtifactError("BoundaryCandidate.source_peak_confidence must be in [0, 1]")
    return {
        "anchor_id": protocol.require_nonnegative_int(
            payload.get("anchor_id"),
            "BoundaryCandidate.anchor_id",
        ),
        "time_ms": protocol.require_finite_number(
            payload.get("time_ms"),
            "BoundaryCandidate.time_ms",
        ),
        "source_peak_index": protocol.require_nonnegative_int(
            payload.get("source_peak_index"),
            "BoundaryCandidate.source_peak_index",
        ),
        "source_peak_time_ms": protocol.require_finite_number(
            payload.get("source_peak_time_ms"),
            "BoundaryCandidate.source_peak_time_ms",
        ),
        "source_peak_confidence": source_peak_confidence,
        "rank_score": protocol.require_finite_number(
            payload.get("rank_score"),
            "BoundaryCandidate.rank_score",
        ),
        "evidence_mode": evidence_mode,
        "left_period_ms": _require_positive_finite(
            payload.get("left_period_ms"),
            "BoundaryCandidate.left_period_ms",
        ),
        "right_period_ms": _require_positive_finite(
            payload.get("right_period_ms"),
            "BoundaryCandidate.right_period_ms",
        ),
        "ordinary_score": _optional_finite(
            payload.get("ordinary_score"),
            "BoundaryCandidate.ordinary_score",
        ),
        "super_score": _optional_finite(
            payload.get("super_score"),
            "BoundaryCandidate.super_score",
        ),
        "downbeat_bonus": protocol.require_finite_number(
            payload.get("downbeat_bonus"),
            "BoundaryCandidate.downbeat_bonus",
        ),
        "nearest_downbeat_distance_ms": _optional_finite_nonnegative(
            payload.get("nearest_downbeat_distance_ms"),
            "BoundaryCandidate.nearest_downbeat_distance_ms",
        ),
    }


def _validate_candidate_diagnostics(
    payload: Any,
    *,
    beat_peak_count: int,
    downbeat_peak_count: int,
    tempo_candidate_count: int,
    origin_candidate_count: int,
    boundary_candidate_count: int,
) -> dict[str, Any]:
    _require_mapping(payload, "CandidateDiagnostics")
    protocol.validate_exact_fields(
        payload,
        CANDIDATE_DIAGNOSTICS_FIELDS,
        "CandidateDiagnostics",
    )
    frame_rate = protocol.require_finite_number(
        payload.get("frame_rate_hz"),
        "CandidateDiagnostics.frame_rate_hz",
    )
    if frame_rate <= 0.0:
        raise Exp007ArtifactError("CandidateDiagnostics.frame_rate_hz must be positive")
    coverage_start = protocol.require_finite_number(
        payload.get("coverage_start_ms"),
        "CandidateDiagnostics.coverage_start_ms",
    )
    coverage_end = protocol.require_finite_number(
        payload.get("coverage_end_ms"),
        "CandidateDiagnostics.coverage_end_ms",
    )
    if coverage_end <= coverage_start:
        raise Exp007ArtifactError("CandidateDiagnostics coverage must increase")
    result = {
        "candidate_contract_version": protocol.require_nonempty_string(
            payload.get("candidate_contract_version"),
            "CandidateDiagnostics.candidate_contract_version",
        ),
        "constants_json_sha256": protocol.require_sha256(
            payload.get("constants_json_sha256"),
            "CandidateDiagnostics.constants_json_sha256",
        ),
        "pulse_correlation_version": protocol.require_nonempty_string(
            payload.get("pulse_correlation_version"),
            "CandidateDiagnostics.pulse_correlation_version",
        ),
        "boundary_candidate_score_version": protocol.require_nonempty_string(
            payload.get("boundary_candidate_score_version"),
            "CandidateDiagnostics.boundary_candidate_score_version",
        ),
        "frame_count": protocol.require_positive_int(
            payload.get("frame_count"),
            "CandidateDiagnostics.frame_count",
        ),
        "frame_rate_hz": frame_rate,
        "coverage_start_ms": coverage_start,
        "coverage_end_ms": coverage_end,
        "min_period_frames": protocol.require_positive_int(
            payload.get("min_period_frames"),
            "CandidateDiagnostics.min_period_frames",
        ),
        "max_period_frames": protocol.require_positive_int(
            payload.get("max_period_frames"),
            "CandidateDiagnostics.max_period_frames",
        ),
        "beat_peak_count": protocol.require_nonnegative_int(
            payload.get("beat_peak_count"),
            "CandidateDiagnostics.beat_peak_count",
        ),
        "downbeat_peak_count": protocol.require_nonnegative_int(
            payload.get("downbeat_peak_count"),
            "CandidateDiagnostics.downbeat_peak_count",
        ),
        "tempo_candidate_count": protocol.require_nonnegative_int(
            payload.get("tempo_candidate_count"),
            "CandidateDiagnostics.tempo_candidate_count",
        ),
        "origin_candidate_count": protocol.require_nonnegative_int(
            payload.get("origin_candidate_count"),
            "CandidateDiagnostics.origin_candidate_count",
        ),
        "boundary_candidate_count": protocol.require_nonnegative_int(
            payload.get("boundary_candidate_count"),
            "CandidateDiagnostics.boundary_candidate_count",
        ),
        "input_signal_sha256": protocol.require_sha256(
            payload.get("input_signal_sha256"),
            "CandidateDiagnostics.input_signal_sha256",
        ),
        "candidate_fingerprint": protocol.require_sha256(
            payload.get("candidate_fingerprint"),
            "CandidateDiagnostics.candidate_fingerprint",
        ),
    }
    expected_counts = {
        "beat_peak_count": beat_peak_count,
        "downbeat_peak_count": downbeat_peak_count,
        "tempo_candidate_count": tempo_candidate_count,
        "origin_candidate_count": origin_candidate_count,
        "boundary_candidate_count": boundary_candidate_count,
    }
    for field_name, expected in expected_counts.items():
        if result[field_name] != expected:
            raise Exp007ArtifactError("CandidateDiagnostics count mismatch")
    if result["max_period_frames"] < result["min_period_frames"]:
        raise Exp007ArtifactError("CandidateDiagnostics period bounds mismatch")
    expected_versions = {
        "candidate_contract_version": candidate_source.CANDIDATE_CONTRACT_VERSION,
        "constants_json_sha256": candidate_source.GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON_SHA256,
        "pulse_correlation_version": candidate_source.PULSE_CORRELATION_VERSION,
        "boundary_candidate_score_version": (
            candidate_source.BOUNDARY_CANDIDATE_SCORE_VERSION
        ),
    }
    for field_name, expected in expected_versions.items():
        if result[field_name] != expected:
            raise Exp007ArtifactError(f"CandidateDiagnostics {field_name} mismatch")
    return result


def _validate_source_candidate_contract(payload: Mapping[str, Any]) -> None:
    candidate_set = _source_candidate_set_from_payload(payload)
    diagnostics = candidate_set.diagnostics
    constants = candidate_source.GLOBAL_CONSTANT_JUMP_CONSTANTS
    try:
        geometry = candidate_source._prediction_geometry(  # noqa: SLF001
            _FrameCountSignal(diagnostics.frame_count),
            diagnostics.frame_rate_hz,
            constants,
        )
        expected_geometry = {
            "frame_count": geometry[0],
            "frame_rate_hz": geometry[1],
            "coverage_start_ms": geometry[2],
            "coverage_end_ms": geometry[3],
            "min_period_frames": geometry[4],
            "max_period_frames": geometry[5],
        }
        for field_name, expected in expected_geometry.items():
            if getattr(diagnostics, field_name) != expected:
                raise Exp007ArtifactError(
                    f"CandidateDiagnostics {field_name} mismatch"
                )
        candidate_source._validate_candidate_payload(  # noqa: SLF001
            candidate_set,
            constants,
            coverage_start_ms=diagnostics.coverage_start_ms,
            coverage_end_ms=diagnostics.coverage_end_ms,
        )
        recomputed = candidate_source._candidate_fingerprint(  # noqa: SLF001
            tempo_candidates=candidate_set.tempo_candidates,
            origin_candidates=candidate_set.origin_candidates,
            boundary_candidates=candidate_set.boundary_candidates,
            beat_peaks=candidate_set.beat_peaks,
            downbeat_peaks=candidate_set.downbeat_peaks,
            input_signal_sha256=diagnostics.input_signal_sha256,
        )
    except Exp007ArtifactError:
        raise
    except (TypeError, ValueError) as exc:
        raise Exp007ArtifactError(f"CandidatePayload source contract rejected: {exc}") from exc
    if diagnostics.candidate_fingerprint != recomputed:
        raise Exp007ArtifactError(
            "CandidateDiagnostics candidate_fingerprint mismatch"
        )


class _FrameCountSignal:
    def __init__(self, frame_count: int) -> None:
        self.shape = (frame_count,)


def _source_candidate_set_from_payload(
    payload: Mapping[str, Any],
) -> candidate_source.GlobalConstantJumpCandidateSet:
    diagnostics = payload["diagnostics"]
    return candidate_source.GlobalConstantJumpCandidateSet(
        beat_peaks=tuple(
            candidate_source.MaterializedPeak(**peak)
            for peak in payload["beat_peaks"]
        ),
        downbeat_peaks=tuple(
            candidate_source.MaterializedPeak(**peak)
            for peak in payload["downbeat_peaks"]
        ),
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
        diagnostics=candidate_source.GlobalConstantJumpCandidateDiagnostics(
            **diagnostics
        ),
    )


def _validate_candidate_reference_ref_shape(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require_mapping(payload, "CandidateReferenceRef")
    protocol.validate_exact_fields(
        payload,
        CANDIDATE_REFERENCE_REF_FIELDS,
        "CandidateReferenceRef",
    )
    return {
        "row_index": protocol.require_nonnegative_int(
            payload.get("row_index"),
            "CandidateReferenceRef.row_index",
        ),
        "cache_audio_key": protocol.require_nonempty_string(
            payload.get("cache_audio_key"),
            "CandidateReferenceRef.cache_audio_key",
        ),
        "audio_group_key": protocol.require_nonempty_string(
            payload.get("audio_group_key"),
            "CandidateReferenceRef.audio_group_key",
        ),
        "input_signal_sha256": protocol.require_sha256(
            payload.get("input_signal_sha256"),
            "CandidateReferenceRef.input_signal_sha256",
        ),
        "entry_payload_sha256": protocol.require_sha256(
            payload.get("entry_payload_sha256"),
            "CandidateReferenceRef.entry_payload_sha256",
        ),
        "candidate_payload_byte_count": protocol.require_nonnegative_int(
            payload.get("candidate_payload_byte_count"),
            "CandidateReferenceRef.candidate_payload_byte_count",
        ),
        "candidate_payload_sha256": protocol.require_sha256(
            payload.get("candidate_payload_sha256"),
            "CandidateReferenceRef.candidate_payload_sha256",
        ),
        "candidate_fingerprint": protocol.require_sha256(
            payload.get("candidate_fingerprint"),
            "CandidateReferenceRef.candidate_fingerprint",
        ),
        "bundle_relative_path": validate_bundle_relative_path(
            payload.get("bundle_relative_path")
        ),
        "bundle_fingerprint_sha256": protocol.require_sha256(
            payload.get("bundle_fingerprint_sha256"),
            "CandidateReferenceRef.bundle_fingerprint_sha256",
        ),
    }


def _require_canonical_reference_bundle_path(
    *,
    stage: str,
    schedule_arm: str,
    row_index: int,
    relative_path: str,
    context: str,
) -> None:
    expected = reference_bundle_relative_path(stage, schedule_arm, row_index)
    if validate_bundle_relative_path(relative_path) != expected:
        raise Exp007ArtifactError(
            f"{context} must be canonical manifest bundle path"
        )


def _validated_schedule_run_configs_by_arm(
    payload: Mapping[str, Mapping[str, Any]],
    *,
    selector_manifest_sha256: str,
    source_closure_fingerprint_sha256: str,
) -> dict[str, dict[str, Any]]:
    _require_mapping(payload, "run_configs_by_arm")
    actual = set(payload)
    expected = set(protocol.EXP007_EXECUTION_ORDER)
    if actual != expected:
        raise Exp007ArtifactError(
            "run_configs_by_arm must contain exactly all schedule arms"
        )
    selector_sha = protocol.require_sha256(
        selector_manifest_sha256,
        "CandidateGlobalManifest.selector_manifest_sha256",
    )
    source_sha = protocol.require_sha256(
        source_closure_fingerprint_sha256,
        "CandidateGlobalManifest.source_closure_fingerprint_sha256",
    )
    configs: dict[str, dict[str, Any]] = {}
    for arm in protocol.EXP007_EXECUTION_ORDER:
        try:
            config = protocol.validate_run_config(payload[arm])
        except ValueError as exc:
            raise Exp007ArtifactError(f"schedule RunConfig {arm} is invalid") from exc
        if config["stage"] != protocol.EXP007_SCHEDULE_STAGE:
            raise Exp007ArtifactError("schedule RunConfig stage mismatch")
        if config["schedule_arm"] != arm:
            raise Exp007ArtifactError("schedule RunConfig arm mismatch")
        if config["selector_manifest_sha256"] != selector_sha:
            raise Exp007ArtifactError("schedule RunConfig selector SHA mismatch")
        if config["input_manifest_sha256"] != selector_sha:
            raise Exp007ArtifactError("schedule RunConfig input SHA mismatch")
        if config["source_closure_fingerprint_sha256"] != source_sha:
            raise Exp007ArtifactError("schedule RunConfig source SHA mismatch")
        configs[arm] = config
    reference_projection = _schedule_run_config_non_arm_projection(configs["S30"])
    for arm in protocol.EXP007_EXECUTION_ORDER:
        projection = _schedule_run_config_non_arm_projection(configs[arm])
        if projection != reference_projection:
            raise Exp007ArtifactError(
                "CandidateGlobalManifest run config projection mismatch"
            )
    return configs


def _schedule_run_config_non_arm_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    local_frontier = dict(config["local_frontier_config"])
    local_frontier.pop("schedule_arm", None)
    return {
        "schema": config["schema"],
        "experiment_id": config["experiment_id"],
        "stage": config["stage"],
        "schema_descriptor_sha256": config["schema_descriptor_sha256"],
        "method_ids": config["method_ids"],
        "candidate_policy": config["candidate_policy"],
        "pool_policy": config["pool_policy"],
        "limits": config["limits"],
        "selector_manifest_sha256": config["selector_manifest_sha256"],
        "input_manifest_sha256": config["input_manifest_sha256"],
        "schedule_weak_veto_outcome_sha256": (
            config["schedule_weak_veto_outcome_sha256"]
        ),
        "source_closure_fingerprint_sha256": (
            config["source_closure_fingerprint_sha256"]
        ),
        "cache_config_sha256": config["cache_config_sha256"],
        "grid_fitter_config_sha256": config["grid_fitter_config_sha256"],
        "local_frontier_config": local_frontier,
        "weak_config_sha256": config["weak_config_sha256"],
    }


def _require_schedule_row_run_config_binding(
    *,
    row: Mapping[str, Any],
    arm: str,
    run_config: Mapping[str, Any],
) -> None:
    row_payload = protocol.validate_row_result(row)
    config = protocol.validate_run_config(run_config)
    if row_payload["schedule_arm"] != arm or config["schedule_arm"] != arm:
        raise Exp007ArtifactError("schedule row/config arm mismatch")
    if (
        row_payload["run_config_fingerprint_sha256"]
        != config["run_config_fingerprint_sha256"]
    ):
        raise Exp007ArtifactError("schedule row run config mismatch")


def _schedule_row_cross_arm_projection(
    *,
    row: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
) -> dict[str, Any]:
    row_payload = protocol.validate_row_result(row)
    candidate = serialize_candidate_payload(candidate_payload)
    _require_row_candidate_binding(row_payload, candidate)
    cache = row_payload["cache_identity"]
    return {
        "row_index": row_payload["row_index"],
        "cache_audio_key": row_payload["cache_audio_key"],
        "audio_group_key": row_payload["audio_group_key"],
        "identity_payload_sha256": row_payload["identity_payload_sha256"],
        "cache_identity": {
            "relative_cache_path": cache["relative_cache_path"],
            "exists": cache["exists"],
            "size_bytes": cache["size_bytes"],
            "sha256": cache["sha256"],
            "cache_config_sha256": cache["cache_config_sha256"],
            "audio_cache_key_sha256": cache["audio_cache_key_sha256"],
        },
        "source_closure_fingerprint_sha256": row_payload[
            "source_closure_fingerprint_sha256"
        ],
        "selector_manifest_sha256": row_payload["selector_manifest_sha256"],
        "input_manifest_sha256": row_payload["input_manifest_sha256"],
        "restricted_prediction": row_payload["restricted_prediction"],
        "candidate_payload_schema": candidate.schema,
        "candidate_payload_field_set_sha256": candidate.field_set_sha256,
        "candidate_payload_byte_count": candidate.byte_count,
        "candidate_payload_sha256": candidate.payload_sha256,
        "candidate_fingerprint": candidate.candidate_fingerprint,
        "current_v2": row_payload["methods"]["baseline"],
    }


def _require_schedule_row_cross_arm_match(
    *,
    row: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
    reference_projection: Mapping[str, Any],
    arm: str,
) -> None:
    projection = _schedule_row_cross_arm_projection(
        row=row,
        candidate_payload=candidate_payload,
    )
    if projection == reference_projection:
        return
    for field_name in reference_projection:
        if projection[field_name] != reference_projection[field_name]:
            raise Exp007ArtifactError(
                f"CandidateGlobalManifest cross-arm {field_name} mismatch"
            )
    raise Exp007ArtifactError(f"CandidateGlobalManifest cross-arm mismatch for {arm}")


def _candidate_global_entry_from_reference_bundle(
    reference_bundle: Mapping[str, Any],
    arm_row_payload_sha256: Mapping[str, str],
) -> dict[str, Any]:
    entry = validate_candidate_reference_row_bundle(reference_bundle)["entry"]
    return _validate_candidate_global_entry(
        {
            "row_index": entry["row_index"],
            "cache_audio_key": entry["cache_audio_key"],
            "audio_group_key": entry["audio_group_key"],
            "input_signal_sha256": entry["input_signal_sha256"],
            "candidate_payload_schema": entry["candidate_payload_schema"],
            "candidate_payload_field_set_sha256": entry[
                "candidate_payload_field_set_sha256"
            ],
            "candidate_payload_byte_count": entry["candidate_payload_byte_count"],
            "candidate_payload_sha256": entry["candidate_payload_sha256"],
            "candidate_fingerprint": entry["candidate_fingerprint"],
            "candidate_reference_entry_payload_sha256": entry["entry_payload_sha256"],
            "arm_row_payload_sha256": dict(arm_row_payload_sha256),
        },
        expected_row_index=entry["row_index"],
    )


def _validate_candidate_global_entry(
    payload: Mapping[str, Any],
    *,
    expected_row_index: int,
) -> dict[str, Any]:
    _require_mapping(payload, "CandidateGlobalEntry")
    protocol.validate_exact_fields(
        payload,
        CANDIDATE_GLOBAL_ENTRY_FIELDS,
        "CandidateGlobalEntry",
    )
    row_index = protocol.require_nonnegative_int(
        payload.get("row_index"),
        "CandidateGlobalEntry.row_index",
    )
    if row_index != expected_row_index:
        raise Exp007ArtifactError("CandidateGlobalEntry row index mismatch")
    schema = protocol.require_nonempty_string(
        payload.get("candidate_payload_schema"),
        "CandidateGlobalEntry.candidate_payload_schema",
    )
    if schema != CANDIDATE_PAYLOAD_SCHEMA:
        raise Exp007ArtifactError("CandidateGlobalEntry candidate payload schema is invalid")
    arm_map = _validate_arm_row_sha_map(payload.get("arm_row_payload_sha256"))
    return {
        "row_index": row_index,
        "cache_audio_key": protocol.require_nonempty_string(
            payload.get("cache_audio_key"),
            "CandidateGlobalEntry.cache_audio_key",
        ),
        "audio_group_key": protocol.require_nonempty_string(
            payload.get("audio_group_key"),
            "CandidateGlobalEntry.audio_group_key",
        ),
        "input_signal_sha256": protocol.require_sha256(
            payload.get("input_signal_sha256"),
            "CandidateGlobalEntry.input_signal_sha256",
        ),
        "candidate_payload_schema": schema,
        "candidate_payload_field_set_sha256": protocol.require_sha256(
            payload.get("candidate_payload_field_set_sha256"),
            "CandidateGlobalEntry.candidate_payload_field_set_sha256",
        ),
        "candidate_payload_byte_count": protocol.require_nonnegative_int(
            payload.get("candidate_payload_byte_count"),
            "CandidateGlobalEntry.candidate_payload_byte_count",
        ),
        "candidate_payload_sha256": protocol.require_sha256(
            payload.get("candidate_payload_sha256"),
            "CandidateGlobalEntry.candidate_payload_sha256",
        ),
        "candidate_fingerprint": protocol.require_sha256(
            payload.get("candidate_fingerprint"),
            "CandidateGlobalEntry.candidate_fingerprint",
        ),
        "candidate_reference_entry_payload_sha256": protocol.require_sha256(
            payload.get("candidate_reference_entry_payload_sha256"),
            "CandidateGlobalEntry.candidate_reference_entry_payload_sha256",
        ),
        "arm_row_payload_sha256": arm_map,
    }


def _validate_arm_row_sha_map(payload: Any) -> dict[str, str]:
    _require_mapping(payload, "ArmRowShaMap")
    protocol.validate_exact_fields(payload, ARM_ROW_SHA_MAP_FIELDS, "ArmRowShaMap")
    return {
        arm: protocol.require_sha256(payload.get(arm), f"ArmRowShaMap.{arm}")
        for arm in protocol.EXP007_EXECUTION_ORDER
    }


def _require_global_entry_matches_reference_bundle(
    entry: Mapping[str, Any],
    reference_bundle: Mapping[str, Any],
) -> None:
    reference_entry = validate_candidate_reference_row_bundle(reference_bundle)["entry"]
    comparable_fields = (
        "row_index",
        "cache_audio_key",
        "audio_group_key",
        "input_signal_sha256",
        "candidate_payload_schema",
        "candidate_payload_field_set_sha256",
        "candidate_payload_byte_count",
        "candidate_payload_sha256",
        "candidate_fingerprint",
    )
    for field_name in comparable_fields:
        if entry[field_name] != reference_entry[field_name]:
            raise Exp007ArtifactError("CandidateGlobalEntry reference mismatch")
    if (
        entry["candidate_reference_entry_payload_sha256"]
        != reference_entry["entry_payload_sha256"]
    ):
        raise Exp007ArtifactError("CandidateGlobalEntry reference entry SHA mismatch")


def _require_row_candidate_binding(
    row: Mapping[str, Any],
    candidate: CandidatePayloadBytes,
) -> None:
    if (
        row["restricted_prediction"]["input_signal_sha256"]
        != candidate.payload["diagnostics"]["input_signal_sha256"]
    ):
        raise Exp007ArtifactError("row candidate input signal SHA mismatch")
    if row["candidate_payload_schema"] != candidate.schema:
        raise Exp007ArtifactError("row candidate payload schema mismatch")
    if row["candidate_payload_field_set_sha256"] != candidate.field_set_sha256:
        raise Exp007ArtifactError("row candidate field-set SHA mismatch")
    if row["candidate_payload_byte_count"] != candidate.byte_count:
        raise Exp007ArtifactError("row candidate payload byte count mismatch")
    if row["candidate_payload_sha256"] != candidate.payload_sha256:
        raise Exp007ArtifactError("row candidate payload SHA mismatch")
    if row["candidate_fingerprint"] != candidate.candidate_fingerprint:
        raise Exp007ArtifactError("row candidate fingerprint mismatch")


def _require_entry_candidate_binding(
    entry: Mapping[str, Any],
    candidate: CandidatePayloadBytes,
) -> None:
    if (
        entry["input_signal_sha256"]
        != candidate.payload["diagnostics"]["input_signal_sha256"]
    ):
        raise Exp007ArtifactError("entry candidate input signal SHA mismatch")
    if entry["candidate_payload_schema"] != candidate.schema:
        raise Exp007ArtifactError("entry candidate payload schema mismatch")
    if entry["candidate_payload_field_set_sha256"] != candidate.field_set_sha256:
        raise Exp007ArtifactError("entry candidate field-set SHA mismatch")
    if entry["candidate_payload_byte_count"] != candidate.byte_count:
        raise Exp007ArtifactError("entry candidate payload byte count mismatch")
    if entry["candidate_payload_sha256"] != candidate.payload_sha256:
        raise Exp007ArtifactError("entry candidate payload SHA mismatch")
    if entry["candidate_fingerprint"] != candidate.candidate_fingerprint:
        raise Exp007ArtifactError("entry candidate fingerprint mismatch")


def _require_entry_row_binding(
    entry: Mapping[str, Any],
    row: Mapping[str, Any],
) -> None:
    for field_name in ("row_index", "cache_audio_key", "audio_group_key"):
        if entry[field_name] != row[field_name]:
            raise Exp007ArtifactError("CandidateReferenceRowBundle identity mismatch")
    if entry["bound_row_payload_sha256"] != row["row_payload_sha256"]:
        raise Exp007ArtifactError("CandidateReferenceRowBundle bound row mismatch")
    for field_name in (
        "candidate_payload_schema",
        "candidate_payload_field_set_sha256",
        "candidate_payload_byte_count",
        "candidate_payload_sha256",
        "candidate_fingerprint",
    ):
        if entry[field_name] != row[field_name]:
            raise Exp007ArtifactError("CandidateReferenceRowBundle candidate mismatch")


def _validate_schedule_arm_row_against_reference(
    *,
    row: Mapping[str, Any],
    arm: str,
    selector_manifest_sha256: str,
    source_closure_fingerprint_sha256: str,
    reference_bundle: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
) -> None:
    row_payload = protocol.validate_row_result(row)
    if row_payload["stage"] != protocol.EXP007_SCHEDULE_STAGE:
        raise Exp007ArtifactError("schedule row stage mismatch")
    if row_payload["schedule_arm"] != arm:
        raise Exp007ArtifactError("schedule row arm mismatch")
    if row_payload["selector_manifest_sha256"] != selector_manifest_sha256:
        raise Exp007ArtifactError("schedule row selector SHA mismatch")
    if row_payload["input_manifest_sha256"] != selector_manifest_sha256:
        raise Exp007ArtifactError("schedule row input SHA mismatch")
    if row_payload["source_closure_fingerprint_sha256"] != source_closure_fingerprint_sha256:
        raise Exp007ArtifactError("schedule row source SHA mismatch")
    reference = validate_candidate_reference_row_bundle(reference_bundle)
    ref_entry = reference["entry"]
    if row_payload["row_index"] != ref_entry["row_index"]:
        raise Exp007ArtifactError("schedule row index mismatch")
    if row_payload["cache_audio_key"] != ref_entry["cache_audio_key"]:
        raise Exp007ArtifactError("schedule row cache key mismatch")
    if row_payload["audio_group_key"] != ref_entry["audio_group_key"]:
        raise Exp007ArtifactError("schedule row audio group mismatch")
    candidate = serialize_candidate_payload(candidate_payload)
    reference_candidate_bytes = protocol.canonical_json_bytes(ref_entry["candidate_payload"])
    if candidate.canonical_bytes != reference_candidate_bytes:
        raise Exp007ArtifactError("schedule row candidate bytes mismatch")
    _require_row_candidate_binding(row_payload, candidate)
    for field_name in (
        "candidate_payload_schema",
        "candidate_payload_field_set_sha256",
        "candidate_payload_byte_count",
        "candidate_payload_sha256",
        "candidate_fingerprint",
    ):
        if row_payload[field_name] != ref_entry[field_name]:
            raise Exp007ArtifactError("schedule row candidate reference mismatch")


def _arm_sequence(
    payload: Mapping[str, Sequence[Any]],
    arm: str,
    *,
    expected_count: int,
) -> Sequence[Any]:
    if arm not in payload:
        raise Exp007ArtifactError(f"missing arm sequence {arm}")
    sequence = payload[arm]
    if isinstance(sequence, (str, bytes)) or not isinstance(sequence, SequenceABC):
        raise Exp007ArtifactError(f"arm sequence {arm} must be a sequence")
    if len(sequence) != expected_count:
        raise Exp007ArtifactError(f"arm sequence {arm} has wrong length")
    return sequence


def _require_exact_arm_sequences(
    payload: Mapping[str, Sequence[Any]],
    context: str,
) -> None:
    _require_mapping(payload, context)
    actual = set(payload)
    expected = set(protocol.EXP007_EXECUTION_ORDER)
    if actual != expected:
        raise Exp007ArtifactError(f"{context} must contain exactly all schedule arms")


def _candidate_payload_for_index(
    payloads: Mapping[int, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    index: int,
) -> Mapping[str, Any]:
    if isinstance(payloads, MappingABC):
        if index not in payloads:
            raise Exp007ArtifactError("missing candidate payload for row prefix")
        return payloads[index]
    if isinstance(payloads, (str, bytes)) or not isinstance(payloads, SequenceABC):
        raise Exp007ArtifactError("candidate payload prefix must be a mapping or sequence")
    if index >= len(payloads):
        raise Exp007ArtifactError("missing candidate payload for row prefix")
    return payloads[index]


def _require_bpm(value: Any, field_name: str) -> float:
    result = protocol.require_finite_number(value, field_name)
    if result < 20.0 or result > 1000.0:
        raise Exp007ArtifactError(f"{field_name} must be in [20, 1000]")
    return result


def _optional_finite(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return protocol.require_finite_number(value, field_name)


def _optional_finite_nonnegative(value: Any, field_name: str) -> float | None:
    result = _optional_finite(value, field_name)
    if result is not None and result < 0.0:
        raise Exp007ArtifactError(f"{field_name} must be nonnegative")
    return result


def _require_positive_finite(value: Any, field_name: str) -> float:
    result = protocol.require_finite_number(value, field_name)
    if result <= 0.0:
        raise Exp007ArtifactError(f"{field_name} must be positive")
    return result


def _require_reference_arm(stage: str, arm: str) -> None:
    if stage == protocol.EXP007_SCHEDULE_STAGE and arm != "S30":
        raise Exp007ArtifactError("schedule candidate reference arm must be S30")


def _require_stage(value: Any, field_name: str) -> str:
    if value not in {protocol.EXP007_SCHEDULE_STAGE, protocol.EXP007_REPAIR_STAGE}:
        raise Exp007ArtifactError(f"{field_name} must be schedule16 or repair80")
    return str(value)


def _require_schedule_arm(value: Any, field_name: str) -> str:
    if value not in set(protocol.EXP007_SCHEDULE_ARMS):
        raise Exp007ArtifactError(
            f"{field_name} must be one of {protocol.EXP007_SCHEDULE_ARMS!r}"
        )
    return str(value)


def _require_exposure_stage(value: Any, field_name: str) -> str:
    if value not in EXPOSURE_STAGES:
        raise Exp007ArtifactError(f"{field_name} is invalid")
    return str(value)


def _require_observed_payload_kind(value: Any, field_name: str) -> str:
    if value not in OBSERVED_PAYLOAD_KINDS:
        raise Exp007ArtifactError(f"{field_name} is invalid")
    return str(value)


def _require_utc_marker(value: Any, field_name: str) -> str:
    text = protocol.require_nonempty_string(value, field_name)
    if not text.endswith("Z"):
        raise Exp007ArtifactError(f"{field_name} must be UTC and end with Z")
    return text


def _require_descriptor(payload: Mapping[str, Any], schema_id: str, context: str) -> None:
    actual = protocol.require_sha256(
        payload.get("schema_descriptor_sha256"),
        f"{context}.schema_descriptor_sha256",
    )
    expected = protocol.schema_descriptor_sha256(schema_id)
    if actual != expected:
        raise Exp007ArtifactError(f"{context}.schema_descriptor_sha256 mismatch")


def _require_artifact_descriptor(
    payload: Mapping[str, Any],
    schema_id: str,
    context: str,
) -> None:
    actual = protocol.require_sha256(
        payload.get("schema_descriptor_sha256"),
        f"{context}.schema_descriptor_sha256",
    )
    expected = artifact_schema_descriptor_sha256(schema_id)
    if actual != expected:
        raise Exp007ArtifactError(f"{context}.schema_descriptor_sha256 mismatch")


def artifact_schema_descriptor_sha256(schema_id: str) -> str:
    return protocol.canonical_json_sha256(artifact_schema_descriptor_payload(schema_id))


def artifact_schema_descriptor_payload(schema_id: str) -> dict[str, Any]:
    descriptors = _artifact_schema_descriptors()
    if schema_id not in descriptors:
        raise Exp007ArtifactError(f"unknown Exp007 artifact schema descriptor: {schema_id}")
    return descriptors[schema_id]


def _artifact_schema_descriptors() -> dict[str, Any]:
    common = {
        "descriptor_contract_version": "timing-v3-exp007-artifact-lifecycle-v1",
        "experiment_id": protocol.EXP007_EXPERIMENT_ID,
        "canonical_json": {
            "sort_keys": True,
            "separators": [",", ":"],
            "ensure_ascii": True,
            "allow_nan": False,
            "duplicate_keys": "reject",
        },
        "scalar_rules": {
            "sha256": "lowercase [0-9a-f]{64}",
            "int": "reject bool",
            "string": "nonempty unless enum/null branch says otherwise",
        },
    }
    return {
        EXPOSURE_DELTA_SCHEMA: {
            **common,
            "schema": EXPOSURE_DELTA_SCHEMA,
            "type": "exact_object",
            "fields": sorted(EXPOSURE_DELTA_FIELDS),
            "nested_descriptors": {
                "ExposureEntry": {
                    "type": "exact_object",
                    "fields": sorted(EXPOSURE_ENTRY_FIELDS),
                }
            },
            "enums": {
                "exposure_stage": sorted(EXPOSURE_STAGES),
                "observed_payload_kind": sorted(OBSERVED_PAYLOAD_KINDS),
            },
            "hash_preimages": {
                "cache_audio_keys_sha256": (
                    "canonical JSON of sorted unique cache_audio_key list"
                ),
                "entries_sha256": (
                    "canonical JSON of complete sorted ExposureEntry list"
                ),
                "manifest_fingerprint_sha256": (
                    "canonical ExposureDelta with this hash field omitted"
                ),
            },
        },
        RUN_LOCK_SCHEMA: {
            **common,
            "schema": RUN_LOCK_SCHEMA,
            "type": "exact_object",
            "fields": sorted(RUN_LOCK_FIELDS),
            "enums": {
                "stage": sorted({protocol.EXP007_SCHEDULE_STAGE, protocol.EXP007_REPAIR_STAGE}),
                "schedule_arm": list(protocol.EXP007_SCHEDULE_ARMS),
            },
            "hash_preimages": {
                "output_root_fingerprint_sha256": (
                    "canonical experiment/output-root binding"
                ),
                "lock_payload_sha256": (
                    "canonical RunLock with this hash field omitted"
                ),
            },
        },
    }


def _reject_duplicate_cache_audio_keys(keys: Sequence[str], context: str) -> None:
    if len(set(keys)) != len(keys):
        raise Exp007ArtifactError(f"{context} duplicate cache_audio_key")


def _require_run_lock_for_root(
    run_lock: Exp007RunLock | Mapping[str, Any],
    *,
    root: str | Path,
) -> dict[str, Any]:
    payload = run_lock.payload if isinstance(run_lock, Exp007RunLock) else run_lock
    return validate_run_lock_payload(payload, root=root)


def _require_live_held_run_lock_for_root(
    run_lock: Exp007RunLock,
    *,
    root: str | Path,
) -> dict[str, Any]:
    payload, _, _ = _require_live_held_run_lock_snapshot(run_lock, root=root)
    return payload


def _require_live_held_run_lock_snapshot(
    run_lock: Exp007RunLock,
    *,
    root: str | Path,
) -> tuple[dict[str, Any], Path, os.stat_result]:
    if not isinstance(run_lock, Exp007RunLock):
        raise Exp007ArtifactError("exposure publication requires held Exp007RunLock")
    if not run_lock.acquired:
        raise Exp007ArtifactError("run lock handle is not an acquired run lock")
    root_path = Path(root)
    lock_root = Path(run_lock.root)
    try:
        root_resolved = root_path.resolve(strict=True)
        lock_root_resolved = lock_root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise Exp007ArtifactError("run lock artifact root is missing") from exc
    if lock_root_resolved != root_resolved:
        raise Exp007ArtifactError("run lock root mismatch")
    relative_path = validate_bundle_relative_path(run_lock.relative_path)
    payload = validate_run_lock_payload(run_lock.payload, root=root_path)
    expected_relative_path = run_lock_relative_path(
        payload["stage"],
        payload["schedule_arm"],
    )
    if relative_path != expected_relative_path:
        raise Exp007ArtifactError("run lock relative path mismatch")
    expected_bytes = canonical_json_bytes_under_cap(
        payload,
        byte_cap=RUN_LOCK_BYTE_CAP,
        context="RunLock",
    )
    path = contained_read_candidate_path(root_path, relative_path, must_exist=False)
    if not path.exists():
        raise Exp007ArtifactError("run lock is missing")
    current_bytes, opened_metadata = _read_existing_regular_file_bytes_snapshot(
        path,
        byte_cap=RUN_LOCK_BYTE_CAP,
        context="RunLock",
        root_resolved=root_resolved,
    )
    if current_bytes != expected_bytes:
        raise Exp007ArtifactError("run lock payload changed")
    current_payload = _validate_run_lock_bytes(current_bytes, root=root_path)
    if current_payload != payload:
        raise Exp007ArtifactError("run lock payload changed")
    return payload, path, opened_metadata


def _require_exposure_delta_bound_to_run_lock(
    delta_payload: Mapping[str, Any],
    run_lock_payload: Mapping[str, Any],
) -> None:
    if (
        delta_payload["source_closure_fingerprint_sha256"]
        != run_lock_payload["source_closure_fingerprint_sha256"]
    ):
        raise Exp007ArtifactError("ExposureDelta source closure mismatch with RunLock")
    lock_stage = run_lock_payload["stage"]
    for entry in delta_payload["entries"]:
        if entry["exposure_stage"] != lock_stage:
            raise Exp007ArtifactError("ExposureDelta exposure_stage mismatch with RunLock")


def _read_existing_regular_file_bytes_snapshot(
    path: Path,
    *,
    byte_cap: int,
    context: str,
    root_resolved: Path | None,
) -> tuple[bytes, os.stat_result]:
    anchor = _open_anchored_destination_parent(
        path,
        root_resolved=root_resolved,
        context=context,
    )
    try:
        return _read_existing_regular_file_bytes_snapshot_dirfd(
            anchor,
            byte_cap=byte_cap,
            context=context,
        )
    finally:
        os.close(anchor.parent_fd)


def _validate_run_lock_bytes(raw: bytes, *, root: str | Path) -> dict[str, Any]:
    if len(raw) >= RUN_LOCK_BYTE_CAP:
        raise Exp007ArtifactError("RunLock at or above byte cap")
    payload = protocol.load_json_strict(raw)
    if protocol.canonical_json_bytes(payload) != raw:
        raise Exp007ArtifactError("RunLock is not canonical JSON")
    return validate_run_lock_payload(payload, root=root)


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _contained_existing_path_or_none(root: str | Path, relative_path: str) -> Path | None:
    root_path = Path(root)
    if not root_path.exists():
        return None
    try:
        path = contained_read_candidate_path(root_path, relative_path, must_exist=False)
    except (Exp007ArtifactError, FileNotFoundError):
        return None
    return path if path.exists() else None


def _default_existing_run_lock_relative_path(
    root: str | Path,
    *,
    missing_ok: bool = False,
) -> str | None:
    lock_root = Path(root) / "locks" / protocol.EXP007_EXPERIMENT_ID
    if not lock_root.exists():
        if missing_ok:
            return None
        raise FileNotFoundError("run lock directory is missing")
    matches = sorted(lock_root.glob("*/*.lock.json"))
    if not matches:
        if missing_ok:
            return None
        raise FileNotFoundError("run lock is missing")
    if len(matches) != 1:
        raise Exp007ArtifactError("multiple run locks exist")
    return validate_bundle_relative_path(matches[0].relative_to(root).as_posix())


def _raise_existing_run_lock_error(
    anchor: _AnchoredDestination,
    canonical: bytes,
) -> None:
    try:
        if _existing_destination_matches_dirfd(
            anchor,
            canonical,
            symlink_message="run lock destination symlink is forbidden",
        ):
            raise Exp007ArtifactError("run lock already held by identical payload")
    except Exp007ArtifactError as exc:
        if str(exc) == "existing immutable destination differs":
            raise Exp007ArtifactError("run lock already held by divergent payload") from exc
        raise
    raise Exp007ArtifactError("run lock already exists but cannot be read")


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, MappingABC):
        raise Exp007ArtifactError(f"{context} must be a mapping")
    return value


def _require_sequence(value: Any, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, SequenceABC):
        raise Exp007ArtifactError(f"{context} must be a sequence")
    return value


def _reject_symlink_components(root: Path, parts: Sequence[str]) -> None:
    current = root
    for part in parts:
        current = current / part
        _reject_existing_symlink(current, "symlink path component is forbidden")


def _ensure_directory_component(path: Path, *, root_resolved: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        try:
            path.mkdir()
        except FileExistsError:
            pass
        metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise Exp007ArtifactError("symlink path component is forbidden")
    if not stat.S_ISDIR(metadata.st_mode):
        raise Exp007ArtifactError("path component is not a directory")
    _require_contained(path.resolve(strict=True), root_resolved)


def _require_atomic_destination_safe(
    destination: Path,
    *,
    root_resolved: Path | None,
) -> None:
    _reject_existing_symlink(
        destination.parent,
        "destination parent symlink is forbidden",
    )
    try:
        parent_metadata = destination.parent.lstat()
    except FileNotFoundError as exc:
        raise Exp007ArtifactError("destination parent is missing") from exc
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise Exp007ArtifactError("destination parent is not a directory")
    if root_resolved is not None:
        _require_contained(destination.parent.resolve(strict=True), root_resolved)
    _reject_existing_symlink(destination, "destination symlink is forbidden")


def _run_dirfd_test_hook(event: str, destination: Path) -> None:
    hook = _DIRFD_TEST_HOOK
    if hook is not None:
        hook(event, destination)


def _require_dirfd_support(context: str) -> None:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise Exp007ArtifactError(f"{context} requires nofollow directory fd support")
    required_dirfd = (
        (os.open, "os.open"),
        (os.stat, "os.stat"),
        (os.unlink, "os.unlink"),
        (os.link, "os.link"),
    )
    for function, name in required_dirfd:
        if function not in os.supports_dir_fd:
            raise Exp007ArtifactError(f"{context} requires {name} dir_fd support")
    if os.stat not in os.supports_follow_symlinks:
        raise Exp007ArtifactError(f"{context} requires nofollow stat support")
    if os.link not in os.supports_follow_symlinks:
        raise Exp007ArtifactError(f"{context} requires nofollow link support")


def _open_anchored_destination_parent(
    destination: Path,
    *,
    root_resolved: Path | None,
    context: str,
) -> _AnchoredDestination:
    _require_dirfd_support(context)
    _require_atomic_destination_safe(destination, root_resolved=root_resolved)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        parent_fd = os.open(destination.parent, flags)
    except FileNotFoundError as exc:
        raise Exp007ArtifactError("destination parent is missing") from exc
    except OSError as exc:
        raise Exp007ArtifactError("destination parent cannot be opened safely") from exc
    try:
        parent_metadata = os.fstat(parent_fd)
        if not stat.S_ISDIR(parent_metadata.st_mode):
            raise Exp007ArtifactError("destination parent is not a directory")
        anchor = _AnchoredDestination(
            path=destination,
            parent=destination.parent,
            name=destination.name,
            parent_fd=parent_fd,
            parent_metadata=parent_metadata,
            root_resolved=root_resolved,
        )
        _require_anchor_parent_current(anchor)
        _stat_existing_entry_dirfd(
            anchor,
            missing_ok=True,
            context=context,
            symlink_message="destination symlink is forbidden",
        )
        return anchor
    except Exception:
        os.close(parent_fd)
        raise


def _require_anchor_parent_current(anchor: _AnchoredDestination) -> None:
    try:
        current_metadata = anchor.parent.lstat()
    except FileNotFoundError as exc:
        raise Exp007ArtifactError("destination parent is missing") from exc
    if stat.S_ISLNK(current_metadata.st_mode):
        raise Exp007ArtifactError("destination parent symlink is forbidden")
    if not stat.S_ISDIR(current_metadata.st_mode):
        raise Exp007ArtifactError("destination parent is not a directory")
    if not _same_file_identity(anchor.parent_metadata, current_metadata):
        raise Exp007ArtifactError("destination parent changed")
    if anchor.root_resolved is not None:
        _require_contained(anchor.parent.resolve(strict=True), anchor.root_resolved)


def _stat_existing_entry_dirfd(
    anchor: _AnchoredDestination,
    *,
    missing_ok: bool,
    context: str,
    symlink_message: str,
) -> os.stat_result | None:
    try:
        metadata = os.stat(
            anchor.name,
            dir_fd=anchor.parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError as exc:
        if missing_ok:
            return None
        raise Exp007ArtifactError(f"{context} is missing") from exc
    except OSError as exc:
        raise Exp007ArtifactError(f"{context} cannot be inspected safely") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise Exp007ArtifactError(symlink_message)
    return metadata


def _read_regular_file_prefix_dirfd(
    anchor: _AnchoredDestination,
    *,
    max_bytes: int,
    context: str,
    metadata: os.stat_result | None = None,
) -> tuple[bytes, os.stat_result]:
    limit = protocol.require_positive_int(max_bytes, f"{context}.max_bytes")
    initial_metadata = metadata
    if initial_metadata is None:
        initial_metadata = _stat_existing_entry_dirfd(
            anchor,
            missing_ok=False,
            context=context,
            symlink_message=f"{context} is a symlink",
        )
    if initial_metadata is None:
        raise Exp007ArtifactError(f"{context} is missing")
    if not stat.S_ISREG(initial_metadata.st_mode):
        raise Exp007ArtifactError(f"{context} is not a regular file")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    try:
        descriptor = os.open(anchor.name, flags, dir_fd=anchor.parent_fd)
    except FileNotFoundError as exc:
        raise Exp007ArtifactError(f"{context} is missing") from exc
    except OSError as exc:
        raise Exp007ArtifactError(f"{context} cannot be opened safely") from exc
    try:
        opened_metadata = os.fstat(descriptor)
        if not stat.S_ISREG(opened_metadata.st_mode):
            raise Exp007ArtifactError(f"{context} is not a regular file")
        if not _same_file_identity(opened_metadata, initial_metadata):
            raise Exp007ArtifactError(f"{context} changed before read")
        chunks: list[bytes] = []
        total = 0
        while total < limit:
            chunk = os.read(descriptor, min(64 * 1024, limit - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        current_metadata = _stat_existing_entry_dirfd(
            anchor,
            missing_ok=False,
            context=context,
            symlink_message=f"{context} is a symlink",
        )
        if current_metadata is None or not _same_file_identity(
            opened_metadata,
            current_metadata,
        ):
            raise Exp007ArtifactError(f"{context} changed during read")
        return b"".join(chunks), opened_metadata
    finally:
        os.close(descriptor)


def _read_existing_regular_file_bytes_snapshot_dirfd(
    anchor: _AnchoredDestination,
    *,
    byte_cap: int,
    context: str,
) -> tuple[bytes, os.stat_result]:
    cap = protocol.require_positive_int(byte_cap, f"{context}.byte_cap")
    metadata = _stat_existing_entry_dirfd(
        anchor,
        missing_ok=False,
        context=context,
        symlink_message=f"{context} is a symlink",
    )
    raw, opened_metadata = _read_regular_file_prefix_dirfd(
        anchor,
        max_bytes=cap,
        context=context,
        metadata=metadata,
    )
    if len(raw) >= cap:
        raise Exp007ArtifactError(f"{context} at or above byte cap")
    return raw, opened_metadata


def _existing_destination_matches_dirfd(
    anchor: _AnchoredDestination,
    canonical: bytes,
    *,
    symlink_message: str = "existing immutable destination is a symlink",
) -> bool:
    metadata = _stat_existing_entry_dirfd(
        anchor,
        missing_ok=True,
        context="existing immutable destination",
        symlink_message=symlink_message,
    )
    if metadata is None:
        return False
    existing, _ = _read_regular_file_prefix_dirfd(
        anchor,
        max_bytes=len(canonical) + 1,
        context="existing immutable destination",
        metadata=metadata,
    )
    if existing == canonical:
        return True
    raise Exp007ArtifactError("existing immutable destination differs")


def _existing_destination_matches(destination: Path, canonical: bytes) -> bool:
    anchor = _open_anchored_destination_parent(
        destination,
        root_resolved=None,
        context="existing immutable destination",
    )
    try:
        return _existing_destination_matches_dirfd(anchor, canonical)
    finally:
        os.close(anchor.parent_fd)


def _reject_existing_symlink(path: Path, message: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode):
        raise Exp007ArtifactError(message)


def _is_symlink_path(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISLNK(metadata.st_mode)


def _require_contained(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise Exp007ArtifactError("path escapes artifact root") from exc


def _fsync_directory(path: Path) -> None:
    _require_dirfd_support("directory fsync")
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        _fsync_directory_fd(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory_fd(descriptor: int) -> None:
    os.fsync(descriptor)
