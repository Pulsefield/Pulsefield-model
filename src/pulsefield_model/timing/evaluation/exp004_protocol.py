from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import tempfile
import uuid
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPERIMENT_ID = "timing_v3_experiment_004"

IDENTITY_ROW_SCHEMA = "pulsefield_model.timing_v3_exp004_projection_identity_v1"
EXECUTION_SELECTION_MANIFEST_SCHEMA = (
    "pulsefield_model.timing_v3_exp004_execution_selection_v1"
)
EXECUTION_INPUT_SOURCE_SCHEMA = (
    "pulsefield_model.timing_v3_exp004_execution_input_source_v1"
)
UPSTREAM_SOURCE_SCHEMA = "pulsefield_model.timing_v3_exp004_upstream_source_v1"
CONFIRMED_IDENTITY_SCHEMA = "pulsefield_model.timing_v3_exp004_confirmed_identity_v1"

EXP004_HOLDOUT_MANIFEST_SCHEMA = "pulsefield_model.timing_v3_exp004_holdout100_manifest_v1"
EXP004_BROAD_MANIFEST_SCHEMA = "pulsefield_model.timing_v3_exp004_broad500_manifest_v1"

STAGE_AUDIO_COUNTS: Mapping[str, int] = {
    "repair80": 80,
    "holdout100": 100,
    "broad500": 500,
    "full5050": 5050,
}
PRIOR_STAGE: Mapping[str, str] = {
    "holdout100": "repair80",
    "broad500": "holdout100",
    "full5050": "broad500",
}

IDENTITY_ROW_FIELDS = frozenset(
    {"schema", "stage", "cache_audio_key", "audio_group_key"}
)
CONFIRMED_IDENTITY_FIELDS = frozenset(
    {"cache_audio_key", "audio_group_key", "resolved_audio_path"}
)
SELECTION_ENTRY_FIELDS = frozenset({"cache_audio_key", "audio_group_key"})
UPSTREAM_SOURCE_FIELDS = frozenset(
    {
        "schema",
        "source_schema",
        "path",
        "sha256",
        "fingerprint_sha256",
        "row_count",
        "ordered_cache_audio_keys_sha256",
    }
)
IDENTITY_JSONL_SOURCE_FIELDS = frozenset(
    {
        "schema",
        "path",
        "sha256",
        "row_count",
        "cache_audio_keys_sha256",
        "ordered_cache_audio_keys_sha256",
        "ordered_identity_rows_sha256",
        "hash_encoding",
    }
)
EXECUTION_INPUT_SOURCE_FIELDS = frozenset(
    {
        "schema",
        "upstream",
        "confirmed_identity_schema",
        "confirmed_identity_count",
        "ordered_confirmed_identities_sha256",
        "cache_audio_keys_sha256",
        "ordered_cache_audio_keys_sha256",
        "stage_constraints",
        "identity_jsonl",
        "hash_encoding",
    }
)
STAGE_CONSTRAINT_SCHEMA = "pulsefield_model.timing_v3_exp004_stage_constraints_v1"
STAGE_CONSTRAINT_FIELDS = frozenset(
    {"schema", "stage", "quota_degraded", "degraded_quotas", "broad_underfilled"}
)
EXECUTION_SELECTION_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "experiment",
        "stage",
        "stage_audio_count",
        "selected_audio_count",
        "selected_cache_audio_keys_sha256",
        "selected_ordered_cache_audio_keys_sha256",
        "selected_ordered_entries_sha256",
        "source",
        "selected",
        "manifest_fingerprint_sha256",
    }
)

HASH_ENCODING_CANONICAL_JSON = "sha256_canonical_json_utf8"
HASH_ENCODING_SORTED_STRING_ARRAY = "sha256_canonical_json_sorted_string_array_utf8"

_FORBIDDEN_FIELD_EXACT = {
    "absolute_error_ms",
    "boundary_error_ms",
    "candidate_relative_metric",
    "candidate_relative_metrics",
    "comparator_row",
    "drift_ms",
    "error_ms",
    "mae_ms",
    "metric",
    "metrics",
    "oracle_metric",
    "oracle_metrics",
    "oracle_row",
    "phase_error_ms",
    "rmse_ms",
    "score",
    "scores",
}
_FORBIDDEN_FIELD_SUBSTRINGS = (
    "candidate_relative",
    "comparator",
    "evidence",
    "hitobject",
    "object_grid",
    "oracle",
    "phase_error",
    "redline",
)


@dataclass(frozen=True)
class Exp004IdentityRow:
    row_index: int
    cache_audio_key: str
    audio_group_key: str
    stage: str
    payload_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Exp004SelectionEntry:
    row_index: int
    cache_audio_key: str
    audio_group_key: str
    payload_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Exp004ConfirmedIdentity:
    row_index: int
    cache_audio_key: str
    audio_group_key: str
    resolved_audio_path: str
    payload_sha256: str

    def compact_identity_row(self, *, stage: str) -> dict[str, Any]:
        return {
            "schema": IDENTITY_ROW_SCHEMA,
            "stage": stage,
            "cache_audio_key": self.cache_audio_key,
            "audio_group_key": self.audio_group_key,
        }

    def compact_selection_entry(self) -> dict[str, Any]:
        return {
            "cache_audio_key": self.cache_audio_key,
            "audio_group_key": self.audio_group_key,
        }

    def confirmed_payload(self) -> dict[str, Any]:
        return {
            "schema": CONFIRMED_IDENTITY_SCHEMA,
            "row_index": self.row_index,
            "cache_audio_key": self.cache_audio_key,
            "audio_group_key": self.audio_group_key,
            "resolved_audio_path": self.resolved_audio_path,
        }


def build_exp004_upstream_source(
    *,
    source_schema: str,
    source_path: str | Path,
    source_fingerprint_sha256: str,
    row_count: int,
    ordered_cache_audio_keys_sha256: str,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """Return strict upstream provenance for compact Exp004 execution inputs."""

    path = Path(source_path)
    actual_sha256 = _file_sha256(path)
    if source_sha256 is not None and source_sha256 != actual_sha256:
        raise ValueError("upstream.source_sha256 does not match actual file bytes")
    source = {
        "schema": UPSTREAM_SOURCE_SCHEMA,
        "source_schema": _required_nonempty_string(
            source_schema,
            field_name="upstream.source_schema",
        ),
        "path": path.expanduser().resolve(strict=True).as_posix(),
        "sha256": _require_sha256(actual_sha256, "upstream.sha256"),
        "fingerprint_sha256": _require_sha256(
            source_fingerprint_sha256,
            "upstream.fingerprint_sha256",
        ),
        "row_count": _required_nonnegative_int(row_count, "upstream.row_count"),
        "ordered_cache_audio_keys_sha256": _require_sha256(
            ordered_cache_audio_keys_sha256,
            "upstream.ordered_cache_audio_keys_sha256",
        ),
    }
    validate_exp004_upstream_source(source)
    return source


def build_exp004_execution_inputs(
    *,
    stage: str,
    ordered_identities: Sequence[Mapping[str, Any]],
    upstream_source: Mapping[str, Any],
    identity_rows_jsonl_path: str | Path,
    execution_selection_manifest_path: str | Path,
    stage_constraints: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build immutable compact runner inputs from already-confirmed identities.

    ``ordered_identities`` must contain only cache key, audio-group key, and the
    resolved audio path used by the source wrapper to prove current identity.
    The emitted artifacts deliberately omit audio paths and any oracle fields.
    """

    expected_count = _expected_stage_count(stage)
    identities = _confirmed_identities(ordered_identities, expected_stage=stage)
    if len(identities) != expected_count:
        raise ValueError(
            f"{stage} execution inputs must contain exactly {expected_count} identities"
        )

    identity_path = Path(identity_rows_jsonl_path)
    selection_path = Path(execution_selection_manifest_path)
    _reject_path_aliases(
        {
            "identity_rows_jsonl_path": identity_path,
            "execution_selection_manifest_path": selection_path,
            "upstream_source.path": _optional_path(upstream_source.get("path")),
        }
    )

    compact_identity_rows = [
        identity.compact_identity_row(stage=stage) for identity in identities
    ]
    compact_selection_entries = [
        identity.compact_selection_entry() for identity in identities
    ]
    ordered_cache_keys = [identity.cache_audio_key for identity in identities]
    cache_key_set = set(ordered_cache_keys)
    ordered_cache_hash = _ordered_key_list_sha256(ordered_cache_keys)
    key_set_hash = _key_set_sha256(cache_key_set)
    compact_entries_hash = _stable_json_sha256(compact_selection_entries)
    confirmed_identity_hash = _ordered_confirmed_identities_sha256(identities)

    upstream = validate_exp004_upstream_source(upstream_source)
    _require_path_sha256(Path(upstream["path"]), upstream["sha256"], "upstream source")
    if upstream["row_count"] != expected_count:
        raise ValueError(
            f"upstream.row_count must be {expected_count} for {stage}, "
            f"got {upstream['row_count']}"
        )
    if upstream["ordered_cache_audio_keys_sha256"] != ordered_cache_hash:
        raise ValueError("upstream ordered cache-audio-key hash does not match identities")
    constraints = validate_exp004_stage_constraints(
        _default_stage_constraints(stage) if stage_constraints is None else stage_constraints,
        expected_stage=stage,
    )

    _write_jsonl_immutable_atomic(identity_path, compact_identity_rows)
    identity_source = _identity_jsonl_source(identity_path, compact_identity_rows)

    body: dict[str, Any] = {
        "schema": EXECUTION_SELECTION_MANIFEST_SCHEMA,
        "experiment": EXPERIMENT_ID,
        "stage": stage,
        "stage_audio_count": expected_count,
        "selected_audio_count": len(compact_selection_entries),
        "selected_cache_audio_keys_sha256": key_set_hash,
        "selected_ordered_cache_audio_keys_sha256": ordered_cache_hash,
        "selected_ordered_entries_sha256": compact_entries_hash,
        "source": {
            "schema": EXECUTION_INPUT_SOURCE_SCHEMA,
            "upstream": upstream,
            "confirmed_identity_schema": CONFIRMED_IDENTITY_SCHEMA,
            "confirmed_identity_count": len(identities),
            "ordered_confirmed_identities_sha256": confirmed_identity_hash,
            "cache_audio_keys_sha256": key_set_hash,
            "ordered_cache_audio_keys_sha256": ordered_cache_hash,
            "stage_constraints": constraints,
            "identity_jsonl": identity_source,
            "hash_encoding": HASH_ENCODING_CANONICAL_JSON,
        },
        "selected": compact_selection_entries,
    }
    manifest = _with_fingerprint(body)
    validate_exp004_execution_selection_manifest(manifest, expected_stage=stage)
    _write_json_immutable_atomic(selection_path, manifest)
    _require_path_sha256(Path(upstream["path"]), upstream["sha256"], "upstream source")
    return manifest


def load_exp004_identity_rows(
    path: str | Path,
    *,
    expected_stage: str,
) -> tuple[list[Exp004IdentityRow], dict[str, Any]]:
    identity_path = Path(path)
    rows, source_sha256 = _load_jsonl_objects_bound(identity_path)
    identities = validate_exp004_identity_rows(rows, expected_stage=expected_stage)
    return identities, _identity_rows_source(
        identity_path,
        rows,
        identities,
        source_sha256=source_sha256,
    )


def validate_exp004_identity_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_stage: str,
) -> list[Exp004IdentityRow]:
    expected_count = _expected_stage_count(expected_stage)
    if isinstance(rows, (str, bytes)) or not isinstance(rows, SequenceABC):
        raise ValueError("identity rows must be a sequence of mappings")
    if len(rows) != expected_count:
        raise ValueError(
            f"{expected_stage} identity JSONL must contain exactly {expected_count} rows"
        )

    cache_keys: set[str] = set()
    group_keys: set[str] = set()
    identities: list[Exp004IdentityRow] = []
    for index, payload in enumerate(rows):
        if not isinstance(payload, MappingABC):
            raise ValueError(f"identity row {index} must be a mapping")
        _reject_metric_or_oracle_fields(payload, path=f"identity row {index}")
        _require_exact_fields(payload, IDENTITY_ROW_FIELDS, f"identity row {index}")
        if payload.get("schema") != IDENTITY_ROW_SCHEMA:
            raise ValueError(f"identity row {index} has unsupported schema")
        if payload.get("stage") != expected_stage:
            raise ValueError(f"identity row {index} stage does not match {expected_stage}")
        cache_key = _required_nonempty_string(
            payload.get("cache_audio_key"),
            field_name=f"identity row {index} cache_audio_key",
        )
        group_key = _required_nonempty_string(
            payload.get("audio_group_key"),
            field_name=f"identity row {index} audio_group_key",
        )
        if cache_key in cache_keys:
            raise ValueError(f"duplicate identity cache_audio_key: {cache_key!r}")
        if group_key in group_keys:
            raise ValueError(f"duplicate identity audio_group_key: {group_key!r}")
        cache_keys.add(cache_key)
        group_keys.add(group_key)
        identities.append(
            Exp004IdentityRow(
                row_index=index,
                cache_audio_key=cache_key,
                audio_group_key=group_key,
                stage=expected_stage,
                payload_sha256=_stable_json_sha256(payload),
            )
        )
    return identities


def load_exp004_execution_selection_manifest(
    path: str | Path,
    *,
    expected_stage: str,
) -> tuple[dict[str, Any], list[Exp004SelectionEntry], dict[str, Any]]:
    manifest_path = Path(path)
    manifest, source_sha256 = _load_json_object_bound(manifest_path)
    entries = validate_exp004_execution_selection_manifest(
        manifest,
        expected_stage=expected_stage,
    )
    return manifest, entries, _selection_manifest_source(
        manifest_path,
        manifest,
        entries,
        source_sha256=source_sha256,
    )


def validate_exp004_execution_selection_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_stage: str,
) -> list[Exp004SelectionEntry]:
    expected_count = _expected_stage_count(expected_stage)
    if not isinstance(manifest, MappingABC):
        raise ValueError("execution selection manifest must be a mapping")
    _reject_metric_or_oracle_fields(manifest, path="execution selection manifest")
    _require_exact_fields(
        manifest,
        EXECUTION_SELECTION_MANIFEST_FIELDS,
        "execution selection manifest",
    )
    if manifest.get("schema") != EXECUTION_SELECTION_MANIFEST_SCHEMA:
        raise ValueError("execution selection manifest schema is invalid")
    if manifest.get("experiment") != EXPERIMENT_ID:
        raise ValueError("execution selection manifest experiment is invalid")
    if manifest.get("stage") != expected_stage:
        raise ValueError("execution selection manifest stage does not match requested stage")
    if manifest.get("stage_audio_count") != expected_count:
        raise ValueError("execution selection manifest stage_audio_count is invalid")

    fingerprint = _require_sha256(
        manifest.get("manifest_fingerprint_sha256"),
        "manifest_fingerprint_sha256",
    )
    fingerprint_body = dict(manifest)
    fingerprint_body.pop("manifest_fingerprint_sha256", None)
    actual_fingerprint = _stable_json_sha256(fingerprint_body)
    if fingerprint != actual_fingerprint:
        raise ValueError("execution selection manifest fingerprint mismatch")

    selected = manifest.get("selected")
    if isinstance(selected, (str, bytes)) or not isinstance(selected, SequenceABC):
        raise ValueError("execution selection manifest selected must be a sequence")
    if manifest.get("selected_audio_count") != expected_count:
        raise ValueError("execution selection manifest selected_audio_count is invalid")
    if len(selected) != expected_count:
        raise ValueError(
            f"{expected_stage} execution selection must contain exactly {expected_count} rows"
        )

    entries: list[Exp004SelectionEntry] = []
    cache_keys: set[str] = set()
    group_keys: set[str] = set()
    ordered_cache_keys: list[str] = []
    compact_entries: list[dict[str, str]] = []
    for index, raw_entry in enumerate(selected):
        if not isinstance(raw_entry, MappingABC):
            raise ValueError(f"execution selection entry {index} must be a mapping")
        _require_exact_fields(
            raw_entry,
            SELECTION_ENTRY_FIELDS,
            f"execution selection entry {index}",
        )
        cache_key = _required_nonempty_string(
            raw_entry.get("cache_audio_key"),
            field_name=f"execution selection entry {index} cache_audio_key",
        )
        group_key = _required_nonempty_string(
            raw_entry.get("audio_group_key"),
            field_name=f"execution selection entry {index} audio_group_key",
        )
        if cache_key in cache_keys:
            raise ValueError(f"duplicate selection cache_audio_key: {cache_key!r}")
        if group_key in group_keys:
            raise ValueError(f"duplicate selection audio_group_key: {group_key!r}")
        cache_keys.add(cache_key)
        group_keys.add(group_key)
        ordered_cache_keys.append(cache_key)
        entry = {"cache_audio_key": cache_key, "audio_group_key": group_key}
        compact_entries.append(entry)
        entries.append(
            Exp004SelectionEntry(
                row_index=index,
                cache_audio_key=cache_key,
                audio_group_key=group_key,
                payload_sha256=_stable_json_sha256(entry),
            )
        )

    key_set_hash = _key_set_sha256(cache_keys)
    ordered_key_hash = _ordered_key_list_sha256(ordered_cache_keys)
    entries_hash = _stable_json_sha256(compact_entries)
    if manifest.get("selected_cache_audio_keys_sha256") != key_set_hash:
        raise ValueError("execution selection cache-audio-key set hash mismatch")
    if manifest.get("selected_ordered_cache_audio_keys_sha256") != ordered_key_hash:
        raise ValueError("execution selection ordered cache-audio-key hash mismatch")
    if manifest.get("selected_ordered_entries_sha256") != entries_hash:
        raise ValueError("execution selection ordered entry hash mismatch")

    _validate_execution_source(
        manifest.get("source"),
        expected_stage=expected_stage,
        expected_count=expected_count,
        cache_audio_keys_sha256=key_set_hash,
        ordered_cache_audio_keys_sha256=ordered_key_hash,
    )
    return entries


def reconcile_exp004_execution_inputs(
    identities: Sequence[Exp004IdentityRow],
    entries: Sequence[Exp004SelectionEntry],
    *,
    expected_stage: str,
) -> None:
    expected_count = _expected_stage_count(expected_stage)
    if len(identities) != expected_count or len(entries) != expected_count:
        raise ValueError("identity and execution selection counts do not reconcile")
    for identity, entry in zip(identities, entries, strict=True):
        if identity.row_index != entry.row_index:
            raise ValueError("identity and execution selection ordering does not reconcile")
        if identity.cache_audio_key != entry.cache_audio_key:
            raise ValueError(
                f"identity/selection cache key mismatch at row {identity.row_index}"
            )
        if identity.audio_group_key != entry.audio_group_key:
            raise ValueError(
                f"identity/selection audio group mismatch at row {identity.row_index}"
            )


def validate_exp004_upstream_source(source: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(source, MappingABC):
        raise ValueError("upstream source must be a mapping")
    _reject_metric_or_oracle_fields(source, path="upstream source")
    _require_exact_fields(source, UPSTREAM_SOURCE_FIELDS, "upstream source")
    if source.get("schema") != UPSTREAM_SOURCE_SCHEMA:
        raise ValueError("upstream source schema is invalid")
    resolved = {
        "schema": UPSTREAM_SOURCE_SCHEMA,
        "source_schema": _required_nonempty_string(
            source.get("source_schema"),
            field_name="upstream.source_schema",
        ),
        "path": _required_nonempty_string(source.get("path"), field_name="upstream.path"),
        "sha256": _require_sha256(source.get("sha256"), "upstream.sha256"),
        "fingerprint_sha256": _require_sha256(
            source.get("fingerprint_sha256"),
            "upstream.fingerprint_sha256",
        ),
        "row_count": _required_nonnegative_int(source.get("row_count"), "upstream.row_count"),
        "ordered_cache_audio_keys_sha256": _require_sha256(
            source.get("ordered_cache_audio_keys_sha256"),
            "upstream.ordered_cache_audio_keys_sha256",
        ),
    }
    return resolved


def validate_exp004_stage_constraints(
    constraints: Mapping[str, Any],
    *,
    expected_stage: str,
) -> dict[str, Any]:
    if not isinstance(constraints, MappingABC):
        raise ValueError("stage_constraints must be a mapping")
    _require_exact_fields(constraints, STAGE_CONSTRAINT_FIELDS, "stage_constraints")
    if constraints.get("schema") != STAGE_CONSTRAINT_SCHEMA:
        raise ValueError("stage_constraints schema is invalid")
    if constraints.get("stage") != expected_stage:
        raise ValueError("stage_constraints stage does not match execution stage")
    quota_degraded = constraints.get("quota_degraded")
    if not isinstance(quota_degraded, bool):
        raise ValueError("stage_constraints.quota_degraded must be a boolean")
    broad_underfilled = constraints.get("broad_underfilled")
    if not isinstance(broad_underfilled, bool):
        raise ValueError("stage_constraints.broad_underfilled must be a boolean")
    raw_quotas = constraints.get("degraded_quotas")
    if isinstance(raw_quotas, (str, bytes)) or not isinstance(raw_quotas, SequenceABC):
        raise ValueError("stage_constraints.degraded_quotas must be a string list")
    degraded_quotas: list[str] = []
    seen: set[str] = set()
    for index, raw_quota in enumerate(raw_quotas):
        quota = _required_nonempty_string(
            raw_quota,
            field_name=f"stage_constraints.degraded_quotas[{index}]",
        )
        if quota in seen:
            raise ValueError(f"duplicate degraded quota: {quota!r}")
        seen.add(quota)
        degraded_quotas.append(quota)
    if quota_degraded != bool(degraded_quotas):
        raise ValueError("stage_constraints quota_degraded does not match degraded_quotas")
    if expected_stage not in {"broad500"} and broad_underfilled:
        raise ValueError("stage_constraints broad_underfilled is only valid for broad500")
    return {
        "schema": STAGE_CONSTRAINT_SCHEMA,
        "stage": expected_stage,
        "quota_degraded": quota_degraded,
        "degraded_quotas": degraded_quotas,
        "broad_underfilled": broad_underfilled,
    }


def exp004_stage_count(stage: str) -> int:
    return _expected_stage_count(stage)


def exp004_prior_stage(stage: str) -> str | None:
    _expected_stage_count(stage)
    return PRIOR_STAGE.get(stage)


def stable_json_sha256(value: Any) -> str:
    return _stable_json_sha256(value)


def ordered_cache_audio_keys_sha256(keys: Sequence[str]) -> str:
    return _ordered_key_list_sha256(keys)


def key_set_sha256(keys: set[str]) -> str:
    return _key_set_sha256(keys)


def file_sha256(path: str | Path) -> str:
    return _file_sha256(Path(path))


def default_stage_constraints(stage: str) -> dict[str, Any]:
    return _default_stage_constraints(stage)


def _confirmed_identities(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_stage: str,
) -> list[Exp004ConfirmedIdentity]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, SequenceABC):
        raise ValueError("ordered_identities must be a sequence of mappings")
    cache_keys: set[str] = set()
    group_keys: set[str] = set()
    identities: list[Exp004ConfirmedIdentity] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, MappingABC):
            raise ValueError(f"ordered_identities[{index}] must be a mapping")
        _reject_metric_or_oracle_fields(raw, path=f"ordered_identities[{index}]")
        _require_exact_fields(
            raw,
            CONFIRMED_IDENTITY_FIELDS,
            f"ordered_identities[{index}]",
        )
        cache_key = _required_nonempty_string(
            raw.get("cache_audio_key"),
            field_name=f"ordered_identities[{index}].cache_audio_key",
        )
        group_key = _required_nonempty_string(
            raw.get("audio_group_key"),
            field_name=f"ordered_identities[{index}].audio_group_key",
        )
        audio_path = _required_nonempty_string(
            raw.get("resolved_audio_path"),
            field_name=f"ordered_identities[{index}].resolved_audio_path",
        )
        if cache_key in cache_keys:
            raise ValueError(f"duplicate confirmed cache_audio_key: {cache_key!r}")
        if group_key in group_keys:
            raise ValueError(f"duplicate confirmed audio_group_key: {group_key!r}")
        cache_keys.add(cache_key)
        group_keys.add(group_key)
        payload = {
            "schema": CONFIRMED_IDENTITY_SCHEMA,
            "stage": expected_stage,
            "row_index": index,
            "cache_audio_key": cache_key,
            "audio_group_key": group_key,
            "resolved_audio_path": audio_path,
        }
        identities.append(
            Exp004ConfirmedIdentity(
                row_index=index,
                cache_audio_key=cache_key,
                audio_group_key=group_key,
                resolved_audio_path=audio_path,
                payload_sha256=_stable_json_sha256(payload),
            )
        )
    return identities


def _identity_jsonl_source(
    path: Path,
    compact_identity_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    identities = validate_exp004_identity_rows(
        compact_identity_rows,
        expected_stage=str(compact_identity_rows[0]["stage"]) if compact_identity_rows else "",
    )
    return _identity_rows_source(path, compact_identity_rows, identities)


def _identity_rows_source(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    identities: Sequence[Exp004IdentityRow],
    *,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    ordered_keys = [identity.cache_audio_key for identity in identities]
    return {
        "schema": IDENTITY_ROW_SCHEMA,
        "path": path.expanduser().resolve(strict=False).as_posix(),
        "sha256": _file_sha256(path) if source_sha256 is None else source_sha256,
        "row_count": len(identities),
        "cache_audio_keys_sha256": _key_set_sha256(set(ordered_keys)),
        "ordered_cache_audio_keys_sha256": _ordered_key_list_sha256(ordered_keys),
        "ordered_identity_rows_sha256": _stable_json_sha256([dict(row) for row in rows]),
        "hash_encoding": HASH_ENCODING_CANONICAL_JSON,
    }


def _selection_manifest_source(
    path: Path,
    manifest: Mapping[str, Any],
    entries: Sequence[Exp004SelectionEntry],
    *,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    ordered_keys = [entry.cache_audio_key for entry in entries]
    stage = _required_nonempty_string(manifest.get("stage"), field_name="selection manifest stage")
    source = manifest.get("source")
    if not isinstance(source, MappingABC):
        raise ValueError("selection manifest source must be a mapping")
    stage_constraints = validate_exp004_stage_constraints(
        source.get("stage_constraints"),
        expected_stage=stage,
    )
    return {
        "path": path.expanduser().resolve(strict=False).as_posix(),
        "sha256": _file_sha256(path) if source_sha256 is None else source_sha256,
        "schema": manifest["schema"],
        "manifest_fingerprint_sha256": manifest["manifest_fingerprint_sha256"],
        "row_count": len(entries),
        "cache_audio_keys_sha256": _key_set_sha256(set(ordered_keys)),
        "ordered_cache_audio_keys_sha256": _ordered_key_list_sha256(ordered_keys),
        "stage_constraints": stage_constraints,
    }


def _validate_execution_source(
    source: Any,
    *,
    expected_stage: str,
    expected_count: int,
    cache_audio_keys_sha256: str,
    ordered_cache_audio_keys_sha256: str,
) -> None:
    if not isinstance(source, MappingABC):
        raise ValueError("execution selection source must be a mapping")
    _require_exact_fields(source, EXECUTION_INPUT_SOURCE_FIELDS, "execution selection source")
    if source.get("schema") != EXECUTION_INPUT_SOURCE_SCHEMA:
        raise ValueError("execution selection source schema is invalid")
    if source.get("confirmed_identity_schema") != CONFIRMED_IDENTITY_SCHEMA:
        raise ValueError("execution selection confirmed identity schema is invalid")
    if source.get("confirmed_identity_count") != expected_count:
        raise ValueError("execution selection confirmed identity count is invalid")
    _require_sha256(
        source.get("ordered_confirmed_identities_sha256"),
        "source.ordered_confirmed_identities_sha256",
    )
    if source.get("cache_audio_keys_sha256") != cache_audio_keys_sha256:
        raise ValueError("execution selection source cache-audio-key hash mismatch")
    if source.get("ordered_cache_audio_keys_sha256") != ordered_cache_audio_keys_sha256:
        raise ValueError("execution selection source ordered cache-audio-key hash mismatch")
    if source.get("hash_encoding") != HASH_ENCODING_CANONICAL_JSON:
        raise ValueError("execution selection source hash_encoding is invalid")
    validate_exp004_stage_constraints(source.get("stage_constraints"), expected_stage=expected_stage)

    upstream = validate_exp004_upstream_source(source.get("upstream"))
    if upstream["row_count"] != expected_count:
        raise ValueError("execution selection upstream row count is invalid")
    if upstream["ordered_cache_audio_keys_sha256"] != ordered_cache_audio_keys_sha256:
        raise ValueError("execution selection upstream ordered key hash mismatch")

    identity_jsonl = source.get("identity_jsonl")
    if not isinstance(identity_jsonl, MappingABC):
        raise ValueError("execution selection identity_jsonl source must be a mapping")
    _require_exact_fields(
        identity_jsonl,
        IDENTITY_JSONL_SOURCE_FIELDS,
        "execution selection identity_jsonl source",
    )
    if identity_jsonl.get("schema") != IDENTITY_ROW_SCHEMA:
        raise ValueError("execution selection identity_jsonl schema is invalid")
    if identity_jsonl.get("row_count") != expected_count:
        raise ValueError("execution selection identity_jsonl row_count is invalid")
    for field_name in (
        "sha256",
        "cache_audio_keys_sha256",
        "ordered_cache_audio_keys_sha256",
        "ordered_identity_rows_sha256",
    ):
        _require_sha256(identity_jsonl.get(field_name), f"identity_jsonl.{field_name}")
    _required_nonempty_string(identity_jsonl.get("path"), field_name="identity_jsonl.path")
    if identity_jsonl.get("cache_audio_keys_sha256") != cache_audio_keys_sha256:
        raise ValueError("execution selection identity_jsonl cache key hash mismatch")
    if identity_jsonl.get("ordered_cache_audio_keys_sha256") != ordered_cache_audio_keys_sha256:
        raise ValueError("execution selection identity_jsonl ordered key hash mismatch")
    if identity_jsonl.get("hash_encoding") != HASH_ENCODING_CANONICAL_JSON:
        raise ValueError("execution selection identity_jsonl hash_encoding is invalid")


def _ordered_confirmed_identities_sha256(
    identities: Sequence[Exp004ConfirmedIdentity],
) -> str:
    return _stable_json_sha256(
        [identity.confirmed_payload() for identity in identities]
    )


def _expected_stage_count(stage: str) -> int:
    if stage not in STAGE_AUDIO_COUNTS:
        raise ValueError(f"stage must be one of {tuple(STAGE_AUDIO_COUNTS)!r}, got {stage!r}")
    return STAGE_AUDIO_COUNTS[stage]


def _default_stage_constraints(stage: str) -> dict[str, Any]:
    _expected_stage_count(stage)
    return {
        "schema": STAGE_CONSTRAINT_SCHEMA,
        "stage": stage,
        "quota_degraded": False,
        "degraded_quotas": [],
        "broad_underfilled": False,
    }


def _with_fingerprint(body: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(body)
    payload["manifest_fingerprint_sha256"] = _stable_json_sha256(body)
    return payload


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return _jsonable(payload)


def _load_json_object_bound(path: Path) -> tuple[dict[str, Any], str]:
    data, sha256 = _read_file_bytes_bound(path)
    try:
        payload = json.loads(data.decode("utf-8"), parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return _jsonable(payload), sha256


def _load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"{path}:{line_number} blank JSONL lines are not allowed")
            try:
                payload = json.loads(raw, parse_constant=_reject_json_constant)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(_jsonable(payload))
    return rows


def _load_jsonl_objects_bound(path: Path) -> tuple[list[dict[str, Any]], str]:
    data, sha256 = _read_file_bytes_bound(path)
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(data.decode("utf-8").splitlines(), start=1):
        if not raw.strip():
            raise ValueError(f"{path}:{line_number} blank JSONL lines are not allowed")
        try:
            payload = json.loads(raw, parse_constant=_reject_json_constant)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        rows.append(_jsonable(payload))
    return rows, sha256


def _read_file_bytes_bound(path: Path) -> tuple[bytes, str]:
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        data = handle.read()
        after = os.fstat(handle.fileno())
    if (
        before.st_size,
        before.st_mtime_ns,
        before.st_ino,
        before.st_dev,
    ) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
        after.st_dev,
    ):
        raise RuntimeError(f"file changed while reading: {path}")
    return data, hashlib.sha256(data).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, MappingABC):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            result[key] = _jsonable(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats are forbidden in Experiment 004 JSON")
        return float(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_json_immutable_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_bytes_immutable_atomic(path, (_canonical_json(payload) + "\n").encode("utf-8"))


def _write_jsonl_immutable_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    lines = [_canonical_json(row) for row in rows]
    data = ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")
    _write_bytes_immutable_atomic(path, data)


def _write_bytes_immutable_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == data:
            return
        raise ValueError(f"immutable output already exists with different bytes: {path}")

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"immutable output already exists with different bytes: {path}")
        else:
            _fsync_directory(path.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
            _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_sha256(path: Path) -> str:
    try:
        before = path.stat()
    except FileNotFoundError:
        raise
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (
        before.st_size,
        before.st_mtime_ns,
        before.st_ino,
        before.st_dev,
    ) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
        after.st_dev,
    ):
        raise RuntimeError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _require_path_sha256(path: Path, expected_sha256: str, field_name: str) -> None:
    actual = _file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"{field_name} bytes do not match declared SHA-256")


def _key_set_sha256(keys: set[str]) -> str:
    return _stable_json_sha256(sorted(keys))


def _ordered_key_list_sha256(keys: Sequence[str]) -> str:
    return _stable_json_sha256(list(keys))


def _required_nonempty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _required_nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _require_sha256(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _require_exact_fields(
    value: Mapping[str, Any],
    expected_fields: frozenset[str],
    field_name: str,
) -> None:
    actual = set(value)
    if actual != expected_fields:
        missing = sorted(expected_fields.difference(actual))
        extra = sorted(actual.difference(expected_fields))
        raise ValueError(
            f"{field_name} fields are incomplete or unsupported: "
            f"missing={missing!r} extra={extra!r}"
        )


def _reject_metric_or_oracle_fields(value: Any, *, path: str) -> None:
    if isinstance(value, MappingABC):
        for raw_key, child in value.items():
            key = str(raw_key)
            lowered = key.lower()
            if lowered in _FORBIDDEN_FIELD_EXACT or any(
                marker in lowered for marker in _FORBIDDEN_FIELD_SUBSTRINGS
            ):
                raise ValueError(f"metric/oracle-valued field is forbidden: {path}.{key}")
            _reject_metric_or_oracle_fields(child, path=f"{path}.{key}")
    elif isinstance(value, SequenceABC) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _reject_metric_or_oracle_fields(child, path=f"{path}[{index}]")


def _reject_path_aliases(paths: Mapping[str, Path | None]) -> None:
    resolved: dict[Path, str] = {}
    for name, path in paths.items():
        if path is None:
            continue
        candidate = path.expanduser().resolve(strict=False)
        other = resolved.get(candidate)
        if other is not None:
            raise ValueError(f"{name} aliases {other}: {candidate}")
        resolved[candidate] = name


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("upstream_source.path must be a non-empty string")
    return Path(value)
