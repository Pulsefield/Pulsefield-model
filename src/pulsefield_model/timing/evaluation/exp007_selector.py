from __future__ import annotations

import hashlib
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from typing import Any, Mapping, Sequence

from pulsefield_model.timing.evaluation import exp007_protocol as protocol


SELECTOR_MANIFEST_SCHEMA = (
    "pulsefield_model.timing_v3_exp007_schedule16_selector_manifest_v1"
)
SOURCE_REPAIR80_IDENTITY_ARTIFACT_SCHEMA = (
    "pulsefield_model.timing_v3_exp007_repair80_identity_rows_v1"
)
SOURCE_LABELS_ARTIFACT_SCHEMA = (
    "pulsefield_model.timing_v3_exp007_source_label_rows_v1"
)
SOURCE_ARTIFACT_FIELDS = frozenset({"schema", "rows"})

SELECTOR_CLASSES = ("long", "dense", "jump", "stable")
SELECTOR_BUCKETS = SELECTOR_CLASSES + ("deficit_fill",)
SELECTOR_QUOTA = 4
SELECTOR_STAGE_ROW_COUNT = 16
REPAIR80_SOURCE_ROW_COUNT = 80
SELECTION_SUBSTAGES = frozenset(
    {
        "long_quota",
        "dense_quota",
        "jump_quota",
        "stable_quota",
        "long_deficit_from_dense",
        "long_deficit_from_jump",
        "long_deficit_from_stable",
        "dense_deficit_from_jump",
        "dense_deficit_from_stable",
        "jump_deficit_from_stable",
        "deficit_remaining",
    }
)

SELECTOR_ENTRY_FIELDS = frozenset(
    {
        "row_index",
        "source_row_index",
        "cache_audio_key",
        "audio_group_key",
        "bucket",
        "selection_substage",
        "selection_rank",
        "selection_hash_sha256",
        "label_stratum",
        "source_long_track",
        "duration_ms",
        "label_source_sha256",
        "identity_payload_sha256",
    }
)
BUCKET_COUNT_FIELDS = frozenset(
    {"bucket", "requested", "available", "selected", "deficit"}
)
SELECTOR_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "experiment_id",
        "stage",
        "schema_descriptor_sha256",
        "seed",
        "source_repair80_identity",
        "source_labels",
        "source_repair80_row_count",
        "selected_count",
        "bucket_counts",
        "deficit_count",
        "selected_cache_audio_keys_sha256",
        "selected_ordered_cache_audio_keys_sha256",
        "selected_ordered_entries_sha256",
        "selected",
        "manifest_fingerprint_sha256",
    }
)


def make_selector_manifest(
    *,
    repair80_identity_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    source_repair80_identity: Mapping[str, Any],
    source_labels: Mapping[str, Any],
) -> dict[str, Any]:
    identities = _validate_sources_and_join(
        repair80_identity_rows=repair80_identity_rows,
        label_rows=label_rows,
        source_repair80_identity=source_repair80_identity,
        source_labels=source_labels,
    )
    selected, bucket_counts = replay_selector_entries(identities)
    selected_keys = [entry["cache_audio_key"] for entry in selected]
    payload = {
        "schema": SELECTOR_MANIFEST_SCHEMA,
        "experiment_id": protocol.EXP007_EXPERIMENT_ID,
        "stage": protocol.EXP007_SCHEDULE_STAGE,
        "schema_descriptor_sha256": protocol.schema_descriptor_sha256(
            SELECTOR_MANIFEST_SCHEMA
        ),
        "seed": protocol.EXP007_SELECTOR_SEED,
        "source_repair80_identity": protocol.validate_source_ref(
            source_repair80_identity
        ),
        "source_labels": protocol.validate_source_ref(source_labels),
        "source_repair80_row_count": REPAIR80_SOURCE_ROW_COUNT,
        "selected_count": SELECTOR_STAGE_ROW_COUNT,
        "bucket_counts": bucket_counts,
        "deficit_count": sum(
            1 for entry in selected if entry["bucket"] == "deficit_fill"
        ),
        "selected_cache_audio_keys_sha256": protocol.canonical_json_sha256(
            sorted(selected_keys)
        ),
        "selected_ordered_cache_audio_keys_sha256": protocol.canonical_json_sha256(
            selected_keys
        ),
        "selected_ordered_entries_sha256": protocol.canonical_json_sha256(selected),
        "selected": selected,
    }
    return validate_selector_manifest(
        protocol.with_payload_hash(payload, "manifest_fingerprint_sha256"),
        repair80_identity_rows=repair80_identity_rows,
        label_rows=label_rows,
        source_repair80_identity=source_repair80_identity,
        source_labels=source_labels,
    )


def validate_repair80_identity_label_sources(
    *,
    repair80_identity_source_artifact: bytes,
    label_source_artifact: bytes,
    source_repair80_identity: Mapping[str, Any],
    source_labels: Mapping[str, Any],
    repair80_identity_rows: Sequence[Mapping[str, Any]] | None = None,
    label_rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], ...]:
    identity_source = protocol.validate_source_ref(source_repair80_identity)
    label_source = protocol.validate_source_ref(source_labels)
    identity_artifact_rows = _validate_source_artifact_rows(
        source=identity_source,
        artifact=repair80_identity_source_artifact,
        expected_schema=SOURCE_REPAIR80_IDENTITY_ARTIFACT_SCHEMA,
        context="source_repair80_identity",
    )
    label_artifact_rows = _validate_source_artifact_rows(
        source=label_source,
        artifact=label_source_artifact,
        expected_schema=SOURCE_LABELS_ARTIFACT_SCHEMA,
        context="source_labels",
    )
    if (
        repair80_identity_rows is not None
        and [dict(row) for row in repair80_identity_rows] != identity_artifact_rows
    ):
        raise ValueError("repair80_identity_rows do not match source artifact rows")
    if (
        label_rows is not None
        and [dict(row) for row in label_rows] != label_artifact_rows
    ):
        raise ValueError("label_rows do not match source artifact rows")
    return tuple(
        _validate_sources_and_join(
            repair80_identity_rows=identity_artifact_rows,
            label_rows=label_artifact_rows,
            source_repair80_identity=identity_source,
            source_labels=label_source,
        )
    )


def validate_selector_manifest(
    payload: Mapping[str, Any],
    *,
    repair80_identity_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    source_repair80_identity: Mapping[str, Any],
    source_labels: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_selector_manifest_shape(payload)
    protocol.validate_payload_hash(
        payload,
        "manifest_fingerprint_sha256",
        context="SelectorManifest",
    )

    expected = _build_selector_manifest_unchecked(
        repair80_identity_rows=repair80_identity_rows,
        label_rows=label_rows,
        source_repair80_identity=source_repair80_identity,
        source_labels=source_labels,
    )
    if dict(payload) != expected:
        raise ValueError("SelectorManifest replay mismatch")
    return dict(payload)


def validate_selector_manifest_authoritatively(
    payload: Mapping[str, Any],
    *,
    repair80_identity_source_artifact_bytes: bytes,
    label_source_artifact_bytes: bytes,
) -> dict[str, Any]:
    _validate_selector_manifest_shape(payload)
    protocol.validate_payload_hash(
        payload,
        "manifest_fingerprint_sha256",
        context="SelectorManifest",
    )
    identity_source = protocol.validate_source_ref(payload.get("source_repair80_identity"))
    label_source = protocol.validate_source_ref(payload.get("source_labels"))
    identity_rows = _validate_source_artifact_rows(
        source=identity_source,
        artifact=repair80_identity_source_artifact_bytes,
        expected_schema=SOURCE_REPAIR80_IDENTITY_ARTIFACT_SCHEMA,
        context="source_repair80_identity",
    )
    label_rows = _validate_source_artifact_rows(
        source=label_source,
        artifact=label_source_artifact_bytes,
        expected_schema=SOURCE_LABELS_ARTIFACT_SCHEMA,
        context="source_labels",
    )
    return validate_selector_manifest(
        payload,
        repair80_identity_rows=identity_rows,
        label_rows=label_rows,
        source_repair80_identity=identity_source,
        source_labels=label_source,
    )


def replay_selector_entries(
    repair80_identity_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [protocol.validate_identity(row) for row in repair80_identity_rows]
    _validate_identity_keys(rows)
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    substage_counts: dict[str, int] = {}
    own_selected_by_class: dict[str, int] = {bucket: 0 for bucket in SELECTOR_CLASSES}

    def take(rows_to_take: Sequence[Mapping[str, Any]], *, bucket: str, substage: str) -> None:
        if bucket not in SELECTOR_BUCKETS:
            raise ValueError("selector bucket is invalid")
        if substage not in SELECTION_SUBSTAGES:
            raise ValueError("selection_substage is invalid")
        for row in rows_to_take:
            key = row["cache_audio_key"]
            if key in used:
                raise ValueError("selector replay attempted to reuse a row")
            rank = substage_counts.get(substage, 0)
            substage_counts[substage] = rank + 1
            selected.append(
                make_selector_entry(
                    row_index=len(selected),
                    source_row_index=row["row_index"],
                    identity=row,
                    bucket=bucket,
                    selection_substage=substage,
                    selection_rank=rank,
                )
            )
            used.add(key)

    for index, quota in enumerate(SELECTOR_CLASSES):
        own = _ranked_rows(
            row
            for row in rows
            if row["cache_audio_key"] not in used and class_of(row) == quota
        )
        chosen = own[:SELECTOR_QUOTA]
        take(chosen, bucket=quota, substage=f"{quota}_quota")
        own_selected_by_class[quota] = len(chosen)
        deficit = SELECTOR_QUOTA - min(SELECTOR_QUOTA, len(own))
        for donor in SELECTOR_CLASSES[index + 1 :]:
            if deficit == 0:
                break
            donor_rows = _ranked_rows(
                row
                for row in rows
                if row["cache_audio_key"] not in used and class_of(row) == donor
            )
            donor_chosen = donor_rows[:deficit]
            take(
                donor_chosen,
                bucket="deficit_fill",
                substage=f"{quota}_deficit_from_{donor}",
            )
            deficit -= len(donor_chosen)

    if len(selected) < SELECTOR_STAGE_ROW_COUNT:
        remaining = _ranked_rows(
            row for row in rows if row["cache_audio_key"] not in used
        )
        take(
            remaining[: SELECTOR_STAGE_ROW_COUNT - len(selected)],
            bucket="deficit_fill",
            substage="deficit_remaining",
        )
    if len(selected) != SELECTOR_STAGE_ROW_COUNT:
        raise ValueError("selector replay did not produce exactly 16 rows")

    available_by_class = {
        bucket: sum(1 for row in rows if class_of(row) == bucket)
        for bucket in SELECTOR_CLASSES
    }
    bucket_counts = [
        validate_bucket_count(
            {
                "bucket": bucket,
                "requested": SELECTOR_QUOTA,
                "available": available_by_class[bucket],
                "selected": own_selected_by_class[bucket],
                "deficit": SELECTOR_QUOTA - own_selected_by_class[bucket],
            }
        )
        for bucket in SELECTOR_CLASSES
    ]
    return selected, bucket_counts


def make_selector_entry(
    *,
    row_index: int,
    source_row_index: int,
    identity: Mapping[str, Any],
    bucket: str,
    selection_substage: str,
    selection_rank: int,
) -> dict[str, Any]:
    row = protocol.validate_identity(identity)
    entry = {
        "row_index": row_index,
        "source_row_index": source_row_index,
        "cache_audio_key": row["cache_audio_key"],
        "audio_group_key": row["audio_group_key"],
        "bucket": bucket,
        "selection_substage": selection_substage,
        "selection_rank": selection_rank,
        "selection_hash_sha256": selection_hash(row["cache_audio_key"]),
        "label_stratum": row["label_stratum"],
        "source_long_track": row["source_long_track"],
        "duration_ms": row["duration_ms"],
        "label_source_sha256": row["label_source_sha256"],
        "identity_payload_sha256": row["identity_payload_sha256"],
    }
    return validate_selector_entry(entry)


def validate_selector_entry(payload: Mapping[str, Any]) -> dict[str, Any]:
    protocol.validate_exact_fields(payload, SELECTOR_ENTRY_FIELDS, "SelectorEntry")
    bucket = payload.get("bucket")
    if bucket not in SELECTOR_BUCKETS:
        raise ValueError("SelectorEntry.bucket is invalid")
    substage = payload.get("selection_substage")
    if substage not in SELECTION_SUBSTAGES:
        raise ValueError("SelectorEntry.selection_substage is invalid")
    label = payload.get("label_stratum")
    if label not in protocol.LABEL_STRATA:
        raise ValueError("SelectorEntry.label_stratum is invalid")
    source_long_track = payload.get("source_long_track")
    if not isinstance(source_long_track, bool):
        raise ValueError("SelectorEntry.source_long_track must be a bool")
    duration_ms = protocol.require_finite_number(
        payload.get("duration_ms"),
        "SelectorEntry.duration_ms",
    )
    if duration_ms <= 0:
        raise ValueError("SelectorEntry.duration_ms must be positive")
    result = {
        "row_index": protocol.require_nonnegative_int(
            payload.get("row_index"),
            "SelectorEntry.row_index",
        ),
        "source_row_index": protocol.require_nonnegative_int(
            payload.get("source_row_index"),
            "SelectorEntry.source_row_index",
        ),
        "cache_audio_key": protocol.require_nonempty_string(
            payload.get("cache_audio_key"),
            "SelectorEntry.cache_audio_key",
        ),
        "audio_group_key": protocol.require_nonempty_string(
            payload.get("audio_group_key"),
            "SelectorEntry.audio_group_key",
        ),
        "bucket": bucket,
        "selection_substage": substage,
        "selection_rank": protocol.require_nonnegative_int(
            payload.get("selection_rank"),
            "SelectorEntry.selection_rank",
        ),
        "selection_hash_sha256": protocol.require_sha256(
            payload.get("selection_hash_sha256"),
            "SelectorEntry.selection_hash_sha256",
        ),
        "label_stratum": label,
        "source_long_track": source_long_track,
        "duration_ms": payload.get("duration_ms"),
        "label_source_sha256": protocol.require_sha256(
            payload.get("label_source_sha256"),
            "SelectorEntry.label_source_sha256",
        ),
        "identity_payload_sha256": protocol.require_sha256(
            payload.get("identity_payload_sha256"),
            "SelectorEntry.identity_payload_sha256",
        ),
    }
    if result["selection_hash_sha256"] != selection_hash(result["cache_audio_key"]):
        raise ValueError("SelectorEntry.selection_hash_sha256 mismatch")
    return result


def validate_bucket_count(payload: Mapping[str, Any]) -> dict[str, Any]:
    protocol.validate_exact_fields(payload, BUCKET_COUNT_FIELDS, "BucketCount")
    bucket = payload.get("bucket")
    if bucket not in SELECTOR_CLASSES:
        raise ValueError("BucketCount.bucket is invalid")
    requested = protocol.require_nonnegative_int(
        payload.get("requested"),
        "BucketCount.requested",
    )
    if requested != SELECTOR_QUOTA:
        raise ValueError("BucketCount.requested is invalid")
    available = protocol.require_nonnegative_int(
        payload.get("available"),
        "BucketCount.available",
    )
    selected = protocol.require_nonnegative_int(
        payload.get("selected"),
        "BucketCount.selected",
    )
    deficit = protocol.require_nonnegative_int(
        payload.get("deficit"),
        "BucketCount.deficit",
    )
    if selected > requested:
        raise ValueError("BucketCount.selected exceeds requested")
    if deficit != requested - selected:
        raise ValueError("BucketCount.deficit mismatch")
    return {
        "bucket": bucket,
        "requested": requested,
        "available": available,
        "selected": selected,
        "deficit": deficit,
    }


def make_source_ref_for_rows(
    *,
    artifact_schema: str,
    rows: Sequence[Mapping[str, Any]],
    artifact_sha256: str | None = None,
) -> dict[str, Any]:
    rows_sha256 = source_rows_sha256(rows)
    source_artifact_sha256 = protocol.canonical_json_sha256(
        {
            "schema": artifact_schema,
            "rows": [dict(row) for row in rows],
        }
    )
    return protocol.make_source_ref(
        artifact_schema=artifact_schema,
        sha256=source_artifact_sha256 if artifact_sha256 is None else artifact_sha256,
        row_count=len(rows),
        ordered_rows_sha256=rows_sha256,
    )


def source_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return protocol.canonical_json_sha256([dict(row) for row in rows])


def selection_hash(cache_audio_key: str) -> str:
    key = protocol.require_nonempty_string(cache_audio_key, "cache_audio_key")
    preimage = f"{protocol.EXP007_SELECTOR_SEED}\0{key}".encode("utf-8")
    return hashlib.sha256(preimage).hexdigest()


def class_of(row: Mapping[str, Any]) -> str:
    label = row.get("label_stratum")
    source_long_track = row.get("source_long_track")
    if source_long_track is True:
        return "long"
    if label == "dense":
        return "dense"
    if label == "jump_candidate":
        return "jump"
    if label == "stable":
        return "stable"
    return "remaining"


def _build_selector_manifest_unchecked(
    *,
    repair80_identity_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    source_repair80_identity: Mapping[str, Any],
    source_labels: Mapping[str, Any],
) -> dict[str, Any]:
    identities = _validate_sources_and_join(
        repair80_identity_rows=repair80_identity_rows,
        label_rows=label_rows,
        source_repair80_identity=source_repair80_identity,
        source_labels=source_labels,
    )
    selected, bucket_counts = replay_selector_entries(identities)
    selected_keys = [entry["cache_audio_key"] for entry in selected]
    payload = {
        "schema": SELECTOR_MANIFEST_SCHEMA,
        "experiment_id": protocol.EXP007_EXPERIMENT_ID,
        "stage": protocol.EXP007_SCHEDULE_STAGE,
        "schema_descriptor_sha256": protocol.schema_descriptor_sha256(
            SELECTOR_MANIFEST_SCHEMA
        ),
        "seed": protocol.EXP007_SELECTOR_SEED,
        "source_repair80_identity": protocol.validate_source_ref(
            source_repair80_identity
        ),
        "source_labels": protocol.validate_source_ref(source_labels),
        "source_repair80_row_count": REPAIR80_SOURCE_ROW_COUNT,
        "selected_count": SELECTOR_STAGE_ROW_COUNT,
        "bucket_counts": bucket_counts,
        "deficit_count": sum(
            1 for entry in selected if entry["bucket"] == "deficit_fill"
        ),
        "selected_cache_audio_keys_sha256": protocol.canonical_json_sha256(
            sorted(selected_keys)
        ),
        "selected_ordered_cache_audio_keys_sha256": protocol.canonical_json_sha256(
            selected_keys
        ),
        "selected_ordered_entries_sha256": protocol.canonical_json_sha256(selected),
        "selected": selected,
    }
    return protocol.with_payload_hash(payload, "manifest_fingerprint_sha256")


def _validate_source_artifact_rows(
    *,
    source: Mapping[str, Any],
    artifact: bytes,
    expected_schema: str,
    context: str,
) -> list[dict[str, Any]]:
    source_ref = protocol.validate_source_ref(source)
    if source_ref["artifact_schema"] != expected_schema:
        raise ValueError(f"{context}.artifact_schema mismatch")

    if not isinstance(artifact, bytes):
        raise ValueError(f"{context} artifact must be immutable canonical bytes")
    artifact_bytes = artifact
    payload = protocol.load_json_strict(artifact_bytes)
    canonical_bytes = protocol.canonical_json_bytes(payload)
    if artifact_bytes != canonical_bytes:
        raise ValueError(f"{context} must use canonical JSON bytes")

    if not isinstance(payload, MappingABC):
        raise ValueError(f"{context} artifact must be a mapping")
    protocol.validate_exact_fields(payload, SOURCE_ARTIFACT_FIELDS, f"{context} artifact")
    if payload.get("schema") != expected_schema:
        raise ValueError(f"{context}.schema mismatch")
    rows = payload.get("rows")
    if isinstance(rows, (str, bytes)) or not isinstance(rows, SequenceABC):
        raise ValueError(f"{context}.rows must be a sequence")
    row_dicts: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, MappingABC):
            raise ValueError(f"{context}.rows[{index}] must be a mapping")
        row_dicts.append(dict(row))

    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    if source_ref["sha256"] != artifact_sha256:
        raise ValueError(f"{context}.sha256 mismatch")
    if source_ref["row_count"] != len(row_dicts):
        raise ValueError(f"{context}.row_count mismatch")
    rows_sha256 = source_rows_sha256(row_dicts)
    if source_ref["ordered_rows_sha256"] != rows_sha256:
        raise ValueError(f"{context}.ordered_rows_sha256 mismatch")
    return row_dicts


def _validate_sources_and_join(
    *,
    repair80_identity_rows: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    source_repair80_identity: Mapping[str, Any],
    source_labels: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(repair80_identity_rows, (str, bytes)) or not isinstance(
        repair80_identity_rows,
        SequenceABC,
    ):
        raise ValueError("repair80_identity_rows must be a sequence")
    if isinstance(label_rows, (str, bytes)) or not isinstance(label_rows, SequenceABC):
        raise ValueError("label_rows must be a sequence")
    identity_source = protocol.validate_source_ref(source_repair80_identity)
    label_source = protocol.validate_source_ref(source_labels)
    if identity_source["artifact_schema"] != SOURCE_REPAIR80_IDENTITY_ARTIFACT_SCHEMA:
        raise ValueError("source_repair80_identity.artifact_schema mismatch")
    if label_source["artifact_schema"] != SOURCE_LABELS_ARTIFACT_SCHEMA:
        raise ValueError("source_labels.artifact_schema mismatch")
    if identity_source["row_count"] != REPAIR80_SOURCE_ROW_COUNT:
        raise ValueError("source_repair80_identity.row_count must be 80")
    if label_source["row_count"] != REPAIR80_SOURCE_ROW_COUNT:
        raise ValueError("source_labels.row_count must be 80")
    if len(repair80_identity_rows) != REPAIR80_SOURCE_ROW_COUNT:
        raise ValueError("repair80_identity_rows must contain exactly 80 rows")
    if len(label_rows) != REPAIR80_SOURCE_ROW_COUNT:
        raise ValueError("label_rows must contain exactly 80 rows")
    _validate_source_ref_rows(
        source=identity_source,
        rows=repair80_identity_rows,
        context="source_repair80_identity",
    )
    _validate_source_ref_rows(
        source=label_source,
        rows=label_rows,
        context="source_labels",
    )

    identities = []
    for index, row in enumerate(repair80_identity_rows):
        protocol.reject_forbidden_selector_fields(row, context=f"identity_rows[{index}]")
        identity = protocol.validate_identity(row)
        if identity["stage"] != protocol.EXP007_REPAIR_STAGE:
            raise ValueError("repair80 identity rows must have stage=repair80")
        if identity["row_index"] != index:
            raise ValueError("repair80 identity rows must be ordered by row_index")
        identities.append(identity)
    _validate_identity_keys(identities)

    labels_by_key: dict[str, _LabelProjection] = {}
    for index, row in enumerate(label_rows):
        protocol.reject_forbidden_selector_fields(row, context=f"label_rows[{index}]")
        label = _project_label_row(row, index=index)
        if label.cache_audio_key in labels_by_key:
            raise ValueError(f"duplicate label cache_audio_key: {label.cache_audio_key!r}")
        labels_by_key[label.cache_audio_key] = label

    identity_keys = {identity["cache_audio_key"] for identity in identities}
    if set(labels_by_key) != identity_keys:
        raise ValueError("identity/label cache_audio_key sets mismatch")

    for identity in identities:
        label = labels_by_key[identity["cache_audio_key"]]
        if identity["audio_group_key"] != label.audio_group_key:
            raise ValueError("identity/label audio_group_key mismatch")
        if identity["label_stratum"] != label.label_stratum:
            raise ValueError("identity/label label_stratum mismatch")
        if identity["source_long_track"] is not label.source_long_track:
            raise ValueError("identity/label source_long_track mismatch")
        if label.duration_ms is not None and identity["duration_ms"] != label.duration_ms:
            raise ValueError("identity/label duration_ms mismatch")
        if identity["label_source_sha256"] != label.row_sha256:
            raise ValueError("identity/label label_source_sha256 mismatch")
    return identities


def _validate_source_ref_rows(
    *,
    source: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    context: str,
) -> None:
    rows_sha256 = source_rows_sha256(rows)
    if source["row_count"] != len(rows):
        raise ValueError(f"{context}.row_count mismatch")
    if source["ordered_rows_sha256"] != rows_sha256:
        raise ValueError(f"{context}.ordered_rows_sha256 mismatch")


def _validate_identity_keys(rows: Sequence[Mapping[str, Any]]) -> None:
    seen_cache_keys: set[str] = set()
    seen_stage_rows: set[tuple[str, int]] = set()
    for row in rows:
        key = row["cache_audio_key"]
        stage_row = (row["stage"], row["row_index"])
        if key in seen_cache_keys:
            raise ValueError(f"duplicate repair80 cache_audio_key: {key!r}")
        if stage_row in seen_stage_rows:
            raise ValueError(f"duplicate identity stage/row_index: {stage_row!r}")
        seen_cache_keys.add(key)
        seen_stage_rows.add(stage_row)


class _LabelProjection:
    def __init__(
        self,
        *,
        cache_audio_key: str,
        audio_group_key: str,
        label_stratum: str,
        source_long_track: bool,
        duration_ms: int | float | None,
        row_sha256: str,
    ) -> None:
        self.cache_audio_key = cache_audio_key
        self.audio_group_key = audio_group_key
        self.label_stratum = label_stratum
        self.source_long_track = source_long_track
        self.duration_ms = duration_ms
        self.row_sha256 = row_sha256


def _project_label_row(row: Mapping[str, Any], *, index: int) -> _LabelProjection:
    if not isinstance(row, MappingABC):
        raise ValueError(f"label_rows[{index}] must be a mapping")
    cache_audio_key = _coalesced_nonempty_string(
        row,
        (("cache_audio_key",), ("source", "cache_audio_key")),
        f"label_rows[{index}].cache_audio_key",
    )
    audio_group_key = _coalesced_nonempty_string(
        row,
        (("audio_group_key",),),
        f"label_rows[{index}].audio_group_key",
    )
    label_stratum = _coalesced_nonempty_string(
        row,
        (("label_stratum",), ("label", "stratum")),
        f"label_rows[{index}].label_stratum",
    )
    if label_stratum not in protocol.LABEL_STRATA:
        raise ValueError(f"label_rows[{index}].label_stratum is invalid")
    source_long_track = _coalesced_bool(
        row,
        (("source_long_track",), ("source", "long_track")),
        f"label_rows[{index}].source_long_track",
    )
    duration_ms = _optional_duration_ms(row, index=index)
    return _LabelProjection(
        cache_audio_key=cache_audio_key,
        audio_group_key=audio_group_key,
        label_stratum=label_stratum,
        source_long_track=source_long_track,
        duration_ms=duration_ms,
        row_sha256=protocol.canonical_json_sha256(dict(row)),
    )


def _coalesced_nonempty_string(
    row: Mapping[str, Any],
    paths: Sequence[tuple[str, ...]],
    field_name: str,
) -> str:
    values = _present_values(row, paths)
    if not values:
        raise ValueError(f"{field_name} is required")
    first = protocol.require_nonempty_string(values[0], field_name)
    if any(value != first for value in values[1:]):
        raise ValueError(f"{field_name} has conflicting values")
    return first


def _coalesced_bool(
    row: Mapping[str, Any],
    paths: Sequence[tuple[str, ...]],
    field_name: str,
) -> bool:
    values = _present_values(row, paths)
    if not values:
        raise ValueError(f"{field_name} is required")
    first = values[0]
    if not isinstance(first, bool):
        raise ValueError(f"{field_name} must be a bool")
    if any(value is not first for value in values[1:]):
        raise ValueError(f"{field_name} has conflicting values")
    return first


def _optional_duration_ms(row: Mapping[str, Any], *, index: int) -> int | float | None:
    values = _present_values(row, (("duration_ms",), ("map_duration_ms",)))
    if not values:
        return None
    first = values[0]
    number = protocol.require_finite_number(first, f"label_rows[{index}].duration_ms")
    if number <= 0:
        raise ValueError(f"label_rows[{index}].duration_ms must be positive")
    if any(value != first for value in values[1:]):
        raise ValueError(f"label_rows[{index}].duration_ms has conflicting values")
    return first


def _present_values(
    row: Mapping[str, Any],
    paths: Sequence[tuple[str, ...]],
) -> list[Any]:
    values: list[Any] = []
    for path in paths:
        current: Any = row
        missing = False
        for key in path:
            if not isinstance(current, MappingABC) or key not in current:
                missing = True
                break
            current = current[key]
        if not missing:
            values.append(current)
    return values


def _ranked_rows(rows: Sequence[Mapping[str, Any]] | Any) -> list[Mapping[str, Any]]:
    return sorted(
        list(rows),
        key=lambda row: (selection_hash(row["cache_audio_key"]), row["cache_audio_key"]),
    )


def _validate_selector_manifest_shape(payload: Mapping[str, Any]) -> None:
    protocol.validate_exact_fields(
        payload,
        SELECTOR_MANIFEST_FIELDS,
        "SelectorManifest",
    )
    if payload.get("schema") != SELECTOR_MANIFEST_SCHEMA:
        raise ValueError("SelectorManifest schema is invalid")
    if payload.get("experiment_id") != protocol.EXP007_EXPERIMENT_ID:
        raise ValueError("SelectorManifest experiment_id is invalid")
    if payload.get("stage") != protocol.EXP007_SCHEDULE_STAGE:
        raise ValueError("SelectorManifest stage is invalid")
    if payload.get("schema_descriptor_sha256") != protocol.schema_descriptor_sha256(
        SELECTOR_MANIFEST_SCHEMA
    ):
        raise ValueError("SelectorManifest.schema_descriptor_sha256 mismatch")
    if payload.get("seed") != protocol.EXP007_SELECTOR_SEED:
        raise ValueError("SelectorManifest seed is invalid")
    identity_source = protocol.validate_source_ref(payload.get("source_repair80_identity"))
    label_source = protocol.validate_source_ref(payload.get("source_labels"))
    if (
        payload.get("source_repair80_row_count") != REPAIR80_SOURCE_ROW_COUNT
        or identity_source["row_count"] != REPAIR80_SOURCE_ROW_COUNT
        or label_source["row_count"] != REPAIR80_SOURCE_ROW_COUNT
    ):
        raise ValueError("SelectorManifest source row count is invalid")
    if payload.get("selected_count") != SELECTOR_STAGE_ROW_COUNT:
        raise ValueError("SelectorManifest selected_count is invalid")
    bucket_counts = payload.get("bucket_counts")
    if not isinstance(bucket_counts, SequenceABC) or isinstance(bucket_counts, (str, bytes)):
        raise ValueError("SelectorManifest.bucket_counts must be a sequence")
    if len(bucket_counts) != len(SELECTOR_CLASSES):
        raise ValueError("SelectorManifest must have four BucketCount rows")
    for index, bucket_count in enumerate(bucket_counts):
        validated = validate_bucket_count(bucket_count)
        if validated["bucket"] != SELECTOR_CLASSES[index]:
            raise ValueError("SelectorManifest bucket_counts order is invalid")
    selected = payload.get("selected")
    if not isinstance(selected, SequenceABC) or isinstance(selected, (str, bytes)):
        raise ValueError("SelectorManifest.selected must be a sequence")
    if len(selected) != SELECTOR_STAGE_ROW_COUNT:
        raise ValueError("SelectorManifest.selected must contain 16 rows")
    entries = [validate_selector_entry(entry) for entry in selected]
    for index, entry in enumerate(entries):
        if entry["row_index"] != index:
            raise ValueError("SelectorEntry row_index order mismatch")
    keys = [entry["cache_audio_key"] for entry in entries]
    if len(set(keys)) != len(keys):
        raise ValueError("SelectorManifest selected cache_audio_key values must be unique")
    if payload.get("deficit_count") != sum(
        1 for entry in entries if entry["bucket"] == "deficit_fill"
    ):
        raise ValueError("SelectorManifest deficit_count mismatch")
    expected_hashes = {
        "selected_cache_audio_keys_sha256": protocol.canonical_json_sha256(
            sorted(keys)
        ),
        "selected_ordered_cache_audio_keys_sha256": protocol.canonical_json_sha256(
            keys
        ),
        "selected_ordered_entries_sha256": protocol.canonical_json_sha256(entries),
    }
    for field_name, expected in expected_hashes.items():
        actual = protocol.require_sha256(
            payload.get(field_name),
            f"SelectorManifest.{field_name}",
        )
        if actual != expected:
            raise ValueError(f"SelectorManifest.{field_name} mismatch")
