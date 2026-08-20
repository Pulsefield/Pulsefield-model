from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from pulsefield_model.timing.evaluation import exp004_protocol
from pulsefield_model.timing.evaluation.labels import (
    LABEL_AMBIGUOUS,
    LABEL_DENSE,
    LABEL_JUMP_CANDIDATE,
    LABEL_RAMP_CANDIDATE,
    LABEL_STABLE,
    TIMING_V3_LABEL_AUDIO_ROW_SCHEMA,
)


TIMING_V3_EXP004_EXPOSURE_MANIFEST_SCHEMA = (
    "pulsefield_model.timing_v3_exp004_oracle_exposure_exclusion_manifest_v1"
)
TIMING_V3_EXP004_HOLDOUT_MANIFEST_SCHEMA = (
    "pulsefield_model.timing_v3_exp004_holdout100_manifest_v1"
)
TIMING_V3_EXP004_BROAD_MANIFEST_SCHEMA = (
    "pulsefield_model.timing_v3_exp004_broad500_manifest_v1"
)
TIMING_V3_EXP004_SPLIT_ANNOTATION_SCHEMA = (
    "pulsefield_model.timing_v3_exp004_split_annotation_v1"
)
TIMING_V3_EXP004_EXPLICIT_SELECTED_KEYS_SCHEMA = (
    "pulsefield_model.timing_v3_exp004_explicit_selected_cache_keys_v1"
)

EXP004_HOLDOUT_SEED = "timing-v3-exp004-holdout100-v1"
EXP004_HOLDOUT_DEFICIT_SEED = "timing-v3-exp004-holdout100-deficit-v1"
EXP004_BROAD_SEED = "timing-v3-exp004-broad500-v1"
EXP004_PRIORITY = ("ramp_audit", "anomaly", "long", "dense", "jump", "stable")
EXP004_DEFICIT_PRIORITY = ("jump", "dense", "long", "anomaly", "ramp_audit", "stable")
EXP004_HOLDOUT_QUOTAS = {
    "ramp_audit": 5,
    "anomaly": 10,
    "long": 10,
    "dense": 10,
    "jump": 25,
    "stable": 40,
}
EXP004_REQUIRED_EXPOSURE_SOURCE_COUNTS = {
    "pilot80": 80,
    "protocol3": 3,
    "exp003_holdout100": 100,
}
EXP004_MINIMUM_REQUIRED_EXCLUSION_KEYS = 183
EXP004_HOLDOUT_AUDIO_COUNT = 100
EXP004_BROAD_ADDED_AUDIO_COUNT = 400
EXP004_BROAD_AUDIO_COUNT = 500

_EXPOSURE_MANIFEST_KEYS = {
    "schema_id",
    "generated_from_commit",
    "exposure_scan_source_sha256",
    "generated_at_utc",
    "entries_sha256",
    "entries_hash_encoding",
    "minimum_required_exclusion_keys",
    "required_sources",
    "sources",
    "entry_count",
    "entries",
}
_EXPOSURE_ENTRY_KEYS = {
    "cache_audio_key",
    "exposure_reason",
    "exposure_source",
    "first_exposed_at_or_run_id",
}
_SOURCE_KEYS = {
    "path",
    "sha256",
    "row_count",
    "cache_audio_key_count",
    "cache_audio_keys_sha256",
    "hash_encoding",
}
_METRIC_FIELD_EXACT_DENYLIST = {
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
_METRIC_FIELD_SUBSTRINGS = (
    "candidate_relative",
    "comparator_metric",
    "metric",
    "oracle",
    "oracle_metric",
    "phase_error",
)
_HOLDOUT_MANIFEST_KEYS = {
    "schema",
    "experiment",
    "stage",
    "seed",
    "deficit_seed",
    "selection",
    "source",
    "exposure",
    "selected_audio_count",
    "selected_counts",
    "selected_cache_audio_keys_sha256",
    "selected",
    "manifest_fingerprint_sha256",
}
_BROAD_MANIFEST_KEYS = {
    "schema",
    "experiment",
    "stage",
    "seed",
    "selection",
    "source",
    "exposure",
    "holdout",
    "selected_audio_count",
    "selected_counts",
    "selected_cache_audio_keys_sha256",
    "selected",
    "manifest_fingerprint_sha256",
}
_SPLIT_SELECTED_ENTRY_KEYS = {
    "stage",
    "quota_assignment",
    "selection_substage",
    "selection_rank",
    "selection_hash_sha256",
    "cache_audio_key",
    "audio_group_key",
    "resolved_audio_path",
    "label_stratum",
    "source_long_track",
}


@dataclass(frozen=True)
class _LabelRecord:
    row: Mapping[str, Any]
    audio_group_key: str
    cache_audio_key: str
    label_stratum: str
    long_track: bool


def load_label_rows_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load Exp004 label inventory rows without touching metric artifacts."""

    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(payload)
    _label_records(rows)
    return rows


def build_exp004_exposure_manifest(
    *,
    pilot_manifest_path: str | Path,
    protocol_manifest_path: str | Path,
    exp003_holdout_manifest_path: str | Path,
    manifest_output_path: str | Path,
    labels_jsonl_path: str | Path | None = None,
    generated_from_commit: str,
    generated_at_utc: str | None = None,
    additional_exposure_entries: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build the required fail-closed oracle-exposure exclusion manifest."""

    pilot_path = Path(pilot_manifest_path)
    protocol_path = Path(protocol_manifest_path)
    exp003_path = Path(exp003_holdout_manifest_path)
    manifest_path = Path(manifest_output_path)
    labels_path = Path(labels_jsonl_path) if labels_jsonl_path is not None else None
    _require_distinct_paths([pilot_path, protocol_path, exp003_path, manifest_path])

    label_rows = load_label_rows_jsonl(labels_path) if labels_path is not None else None
    source_specs = (
        (
            "pilot80",
            pilot_path,
            "timing_v3_exp004_required_pilot80_repair",
            "timing_v3_exp001_002_pilot80",
        ),
        (
            "protocol3",
            protocol_path,
            "timing_v3_exp004_required_exp003_protocol_exposure",
            "timing_v3_exp003_protocol3",
        ),
        (
            "exp003_holdout100",
            exp003_path,
            "timing_v3_exp004_required_exp003_holdout100_oracle_rows",
            "timing_v3_exp003_holdout100",
        ),
    )

    entries: list[dict[str, Any]] = []
    sources: dict[str, dict[str, Any]] = {}
    source_key_sets: dict[str, set[str]] = {}
    for source_name, source_path, reason, run_id in source_specs:
        keys = _load_cache_audio_keys(source_path, label_rows=label_rows)
        expected_count = EXP004_REQUIRED_EXPOSURE_SOURCE_COUNTS[source_name]
        if len(keys) != expected_count:
            raise ValueError(
                f"{source_name} must expose exactly {expected_count} cache audio keys, "
                f"got {len(keys)}"
            )
        for other_name, other_keys in source_key_sets.items():
            overlap = sorted(keys.intersection(other_keys))
            if overlap:
                raise ValueError(
                    f"required exposure sources {source_name!r} and {other_name!r} "
                    f"overlap by cache audio key: {overlap[:3]!r}"
                )
        source_key_sets[source_name] = set(keys)
        sources[source_name] = _file_source_provenance(source_path, keys=keys)
        entries.extend(
            {
                "cache_audio_key": cache_audio_key,
                "exposure_reason": reason,
                "exposure_source": source_name,
                "first_exposed_at_or_run_id": run_id,
            }
            for cache_audio_key in sorted(keys)
        )

    for index, raw_entry in enumerate(additional_exposure_entries):
        if not isinstance(raw_entry, MappingABC):
            raise ValueError(f"additional_exposure_entries[{index}] must be a mapping")
        entries.append(dict(raw_entry))

    normalized_entries = _normalize_exposure_entries(entries)
    generated_at = (
        generated_at_utc
        or datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    body: dict[str, Any] = {
        "schema_id": TIMING_V3_EXP004_EXPOSURE_MANIFEST_SCHEMA,
        "generated_from_commit": _required_nonempty_string(
            generated_from_commit,
            field_name="generated_from_commit",
        ),
        "exposure_scan_source_sha256": _json_sha256(sources),
        "generated_at_utc": _required_nonempty_string(
            generated_at,
            field_name="generated_at_utc",
        ),
        "entries_sha256": _key_set_sha256(
            {entry["cache_audio_key"] for entry in normalized_entries}
        ),
        "entries_hash_encoding": "sha256_canonical_json_sorted_string_array_utf8",
        "minimum_required_exclusion_keys": EXP004_MINIMUM_REQUIRED_EXCLUSION_KEYS,
        "required_sources": dict(EXP004_REQUIRED_EXPOSURE_SOURCE_COUNTS),
        "sources": sources,
        "entry_count": len(normalized_entries),
        "entries": normalized_entries,
    }
    validate_exp004_exposure_manifest(body)
    _write_json_atomic(manifest_path, body)
    return body


def select_exp004_holdout100(
    label_rows: Sequence[Mapping[str, Any]],
    *,
    exposure_manifest: Mapping[str, Any],
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select Exp004 holdout100 from labels using only exposure identities."""

    records = _label_records(label_rows)
    exposure_keys = validate_exp004_exposure_manifest(exposure_manifest)
    by_cache_key = {record.cache_audio_key: record for record in records}
    missing_exposures = sorted(exposure_keys.difference(by_cache_key))
    if missing_exposures:
        raise ValueError(
            "exposure manifest contains cache audio keys absent from label_rows: "
            f"{missing_exposures[:3]!r}"
        )

    candidates: dict[str, list[_LabelRecord]] = {quota: [] for quota in EXP004_PRIORITY}
    for record in records:
        if record.cache_audio_key in exposure_keys:
            continue
        candidates[_quota_assignment(record)].append(record)

    resolved_quotas = _validate_quotas(EXP004_HOLDOUT_QUOTAS)
    selected_records: set[str] = set()
    selected: list[dict[str, Any]] = []
    available_counts: dict[str, int] = {}
    degraded: dict[str, dict[str, int]] = {}

    for quota in EXP004_PRIORITY:
        ordered = _ordered_records(candidates[quota], seed=EXP004_HOLDOUT_SEED)
        available_counts[quota] = len(ordered)
        requested = resolved_quotas[quota]
        quota_selected = ordered[: min(requested, len(ordered))]
        if len(quota_selected) < requested:
            degraded[quota] = {
                "requested": requested,
                "available": len(quota_selected),
                "deficit": requested - len(quota_selected),
            }
        for selection_rank, record in enumerate(quota_selected, start=1):
            selected_records.add(record.cache_audio_key)
            selected.append(
                _selection_entry(
                    record,
                    stage="holdout100",
                    quota_assignment=quota,
                    selection_substage="quota",
                    selection_rank=selection_rank,
                    selection_hash_sha256=_selection_hash(
                        EXP004_HOLDOUT_SEED,
                        record.cache_audio_key,
                    ),
                )
            )

    deficit = EXP004_HOLDOUT_AUDIO_COUNT - len(selected)
    deficit_fill_count = 0
    if deficit > 0:
        deficit_candidates = [
            record
            for record in records
            if record.cache_audio_key not in exposure_keys
            and record.cache_audio_key not in selected_records
        ]
        deficit_selected: list[_LabelRecord] = []
        for quota in EXP004_DEFICIT_PRIORITY:
            remaining = deficit - len(deficit_selected)
            if remaining <= 0:
                break
            ordered = _ordered_records(
                [
                    record
                    for record in deficit_candidates
                    if _quota_assignment(record) == quota
                ],
                seed=EXP004_HOLDOUT_DEFICIT_SEED,
            )
            deficit_selected.extend(ordered[:remaining])
        if len(deficit_selected) < deficit:
            raise ValueError(
                "Experiment 004 holdout100 is underfilled after exposure exclusions: "
                f"target {EXP004_HOLDOUT_AUDIO_COUNT}, available {len(selected) + len(deficit_selected)}"
            )
        for selection_rank, record in enumerate(deficit_selected, start=1):
            selected_records.add(record.cache_audio_key)
            selected.append(
                _selection_entry(
                    record,
                    stage="holdout100",
                    quota_assignment=_quota_assignment(record),
                    selection_substage="deficit_fill",
                    selection_rank=selection_rank,
                    selection_hash_sha256=_selection_hash(
                        EXP004_HOLDOUT_DEFICIT_SEED,
                        record.cache_audio_key,
                    ),
                )
            )
        deficit_fill_count = len(deficit_selected)

    if len(selected) != EXP004_HOLDOUT_AUDIO_COUNT:
        raise ValueError(
            f"Experiment 004 holdout100 must contain {EXP004_HOLDOUT_AUDIO_COUNT} audio groups"
        )
    resolved_source = _resolved_holdout_source(
        source,
        records=records,
        exposure_manifest=exposure_manifest,
    )
    exposure_summary = _exposure_summary(exposure_manifest, exposure_keys)
    body: dict[str, Any] = {
        "schema": TIMING_V3_EXP004_HOLDOUT_MANIFEST_SCHEMA,
        "experiment": "timing_v3_experiment_004",
        "stage": "holdout100",
        "seed": EXP004_HOLDOUT_SEED,
        "deficit_seed": EXP004_HOLDOUT_DEFICIT_SEED,
        "selection": {
            "method": "exclusive_priority_sha256_with_preregistered_deficit_fill",
            "priority_order": list(EXP004_PRIORITY),
            "quotas": dict(resolved_quotas),
            "target_audio_count": EXP004_HOLDOUT_AUDIO_COUNT,
            "available_counts_after_exposure_exclusion": available_counts,
            "degraded_underfilled": degraded,
            "deficit_fill": {
                "seed": EXP004_HOLDOUT_DEFICIT_SEED,
                "priority_order": list(EXP004_DEFICIT_PRIORITY),
                "fill_count": deficit_fill_count,
            },
        },
        "source": resolved_source,
        "exposure": exposure_summary,
        "selected_audio_count": len(selected),
        "selected_counts": _selected_counts(selected),
        "selected_cache_audio_keys_sha256": _key_set_sha256(
            {entry["cache_audio_key"] for entry in selected}
        ),
        "selected": selected,
    }
    manifest = _with_manifest_fingerprint(body)
    validate_exp004_manifest(manifest)
    return manifest


def select_exp004_broad500(
    label_rows: Sequence[Mapping[str, Any]],
    *,
    exposure_manifest: Mapping[str, Any],
    holdout_manifest: Mapping[str, Any],
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay a frozen holdout and select the Exp004 broad500 identities."""

    records = _label_records(label_rows)
    exposure_keys = validate_exp004_exposure_manifest(exposure_manifest)
    validate_exp004_manifest(holdout_manifest)
    if holdout_manifest.get("schema") != TIMING_V3_EXP004_HOLDOUT_MANIFEST_SCHEMA:
        raise ValueError("holdout_manifest must use the Experiment 004 holdout100 schema")

    holdout_entries = _manifest_entries(holdout_manifest)
    holdout_keys = {str(entry["cache_audio_key"]) for entry in holdout_entries}
    overlap = sorted(holdout_keys.intersection(exposure_keys))
    if overlap:
        raise ValueError(
            "holdout_manifest overlaps the frozen exposure exclusion manifest: "
            f"{overlap[:3]!r}"
        )

    by_cache_key = {record.cache_audio_key: record for record in records}
    missing_holdout = sorted(holdout_keys.difference(by_cache_key))
    if missing_holdout:
        raise ValueError(
            "holdout_manifest contains cache audio keys absent from label_rows: "
            f"{missing_holdout[:3]!r}"
        )
    for entry in holdout_entries:
        _validate_entry_against_record(entry, by_cache_key[str(entry["cache_audio_key"])])

    resolved_source = _resolved_broad_source(
        source,
        records=records,
        exposure_manifest=exposure_manifest,
        holdout_manifest=holdout_manifest,
    )
    holdout_source = holdout_manifest["source"]
    for source_name in ("labels", "exposure_manifest"):
        if resolved_source[source_name] != holdout_source[source_name]:
            raise ValueError(
                f"broad {source_name} provenance does not exactly match the frozen holdout"
            )

    replayed_holdout = select_exp004_holdout100(
        label_rows,
        exposure_manifest=exposure_manifest,
        source=holdout_source,
    )
    if holdout_manifest != replayed_holdout:
        raise ValueError("holdout_manifest is not the deterministic Exp004 holdout replay")

    added_candidates = [
        record
        for record in records
        if record.cache_audio_key not in exposure_keys
        and record.cache_audio_key not in holdout_keys
    ]
    ordered_added = _ordered_records(added_candidates, seed=EXP004_BROAD_SEED)
    selected = [dict(entry) for entry in holdout_entries]
    added_count = min(EXP004_BROAD_ADDED_AUDIO_COUNT, len(ordered_added))
    for selection_rank, record in enumerate(ordered_added[:added_count], start=1):
        selected.append(
            _selection_entry(
                record,
                stage="broad500_added",
                quota_assignment=None,
                selection_substage="broad500_added",
                selection_rank=selection_rank,
                selection_hash_sha256=_selection_hash(
                    EXP004_BROAD_SEED,
                    record.cache_audio_key,
                ),
            )
        )

    degraded = {}
    if added_count < EXP004_BROAD_ADDED_AUDIO_COUNT:
        degraded = {
            "broad500_added": {
                "requested": EXP004_BROAD_ADDED_AUDIO_COUNT,
                "available": added_count,
                "deficit": EXP004_BROAD_ADDED_AUDIO_COUNT - added_count,
            }
        }

    body: dict[str, Any] = {
        "schema": TIMING_V3_EXP004_BROAD_MANIFEST_SCHEMA,
        "experiment": "timing_v3_experiment_004",
        "stage": "broad500",
        "seed": EXP004_BROAD_SEED,
        "selection": {
            "method": "holdout100_then_lowest_sha256_seed_nul_cache_audio_key",
            "holdout_audio_count": len(holdout_entries),
            "added_audio_count": added_count,
            "target_audio_count": EXP004_BROAD_AUDIO_COUNT,
            "added_candidate_count_after_exclusions": len(ordered_added),
            "degraded_underfilled": degraded,
        },
        "source": resolved_source,
        "exposure": _exposure_summary(exposure_manifest, exposure_keys),
        "holdout": {
            "schema": holdout_manifest["schema"],
            "manifest_fingerprint_sha256": holdout_manifest["manifest_fingerprint_sha256"],
            "cache_audio_keys_sha256": _key_set_sha256(holdout_keys),
            "ordered_cache_audio_keys_sha256": _ordered_key_list_sha256(
                [str(entry["cache_audio_key"]) for entry in holdout_entries]
            ),
            "selected_ordered_sha256": _selected_ordered_sha256(holdout_entries),
        },
        "selected_audio_count": len(selected),
        "selected_counts": {
            "holdout100": len(holdout_entries),
            "broad500_added": added_count,
        },
        "selected_cache_audio_keys_sha256": _key_set_sha256(
            {entry["cache_audio_key"] for entry in selected}
        ),
        "selected": selected,
    }
    manifest = _with_manifest_fingerprint(body)
    validate_exp004_manifest(manifest)
    return manifest


def build_exp004_holdout100(
    *,
    labels_jsonl_path: str | Path,
    exposure_manifest_path: str | Path,
    manifest_output_path: str | Path,
    label_rows_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build and write the Exp004 holdout identity manifest from frozen inputs."""

    labels_path = Path(labels_jsonl_path)
    exposure_path = Path(exposure_manifest_path)
    manifest_path = Path(manifest_output_path)
    rows_path = Path(label_rows_output_path) if label_rows_output_path is not None else None
    _require_distinct_paths([labels_path, exposure_path, manifest_path, rows_path])

    label_rows = load_label_rows_jsonl(labels_path)
    exposure_manifest = load_exp004_exposure_manifest(exposure_path)
    manifest = select_exp004_holdout100(
        label_rows,
        exposure_manifest=exposure_manifest,
        source={
            "labels": _file_provenance(labels_path, row_count=len(label_rows)),
            "exposure_manifest": _exposure_file_provenance(
                exposure_path,
                exposure_manifest=exposure_manifest,
            ),
        },
    )
    _write_json_atomic(manifest_path, manifest)
    if rows_path is not None:
        _write_jsonl_atomic(rows_path, materialize_label_subset(label_rows, manifest))
    return manifest


def build_exp004_broad500(
    *,
    labels_jsonl_path: str | Path,
    exposure_manifest_path: str | Path,
    holdout_manifest_path: str | Path,
    manifest_output_path: str | Path,
    label_rows_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build and write the Exp004 broad500 identity manifest from frozen inputs."""

    labels_path = Path(labels_jsonl_path)
    exposure_path = Path(exposure_manifest_path)
    holdout_path = Path(holdout_manifest_path)
    manifest_path = Path(manifest_output_path)
    rows_path = Path(label_rows_output_path) if label_rows_output_path is not None else None
    _require_distinct_paths([labels_path, exposure_path, holdout_path, manifest_path, rows_path])

    label_rows = load_label_rows_jsonl(labels_path)
    exposure_manifest = load_exp004_exposure_manifest(exposure_path)
    holdout_manifest = load_exp004_manifest(holdout_path)
    manifest = select_exp004_broad500(
        label_rows,
        exposure_manifest=exposure_manifest,
        holdout_manifest=holdout_manifest,
        source={
            "labels": _file_provenance(labels_path, row_count=len(label_rows)),
            "exposure_manifest": _exposure_file_provenance(
                exposure_path,
                exposure_manifest=exposure_manifest,
            ),
            "holdout_manifest": _holdout_file_provenance(
                holdout_path,
                holdout_manifest=holdout_manifest,
            ),
        },
    )
    _write_json_atomic(manifest_path, manifest)
    if rows_path is not None:
        _write_jsonl_atomic(rows_path, materialize_label_subset(label_rows, manifest))
    return manifest


def build_exp004_execution_inputs_from_split_manifest(
    *,
    labels_jsonl_path: str | Path,
    exposure_manifest_path: str | Path,
    split_manifest_path: str | Path,
    holdout_manifest_path: str | Path | None = None,
    identity_rows_jsonl_path: str | Path,
    execution_selection_manifest_path: str | Path,
) -> dict[str, Any]:
    """Build compact Exp004 execution inputs from a validated holdout/broad split.

    This wrapper may read current labels and split manifests because it is an
    evaluation-side builder.  It passes only confirmed cache/group/audio-path
    identities to the import-safe protocol module.
    """

    labels_path = Path(labels_jsonl_path)
    exposure_path = Path(exposure_manifest_path)
    manifest_path = Path(split_manifest_path)
    holdout_path = Path(holdout_manifest_path) if holdout_manifest_path is not None else None
    identity_path = Path(identity_rows_jsonl_path)
    selection_path = Path(execution_selection_manifest_path)
    _require_distinct_paths(
        [labels_path, exposure_path, manifest_path, holdout_path, identity_path, selection_path]
    )

    label_rows, labels_sha256 = _load_label_rows_jsonl_bound(labels_path)
    exposure_manifest, exposure_sha256 = _load_exp004_exposure_manifest_bound(exposure_path)
    manifest, split_sha256 = _load_exp004_manifest_bound(manifest_path)
    bound_sources = [
        (labels_path, labels_sha256, "labels"),
        (exposure_path, exposure_sha256, "exposure_manifest"),
        (manifest_path, split_sha256, "split_manifest"),
    ]
    schema = manifest["schema"]
    if schema == TIMING_V3_EXP004_HOLDOUT_MANIFEST_SCHEMA:
        stage = "holdout100"
        if holdout_path is not None:
            raise ValueError("holdout100 execution input build must not receive holdout_manifest_path")
        _require_source_file_matches(
            manifest["source"]["labels"],
            labels_path,
            labels_sha256,
            field_name="source.labels",
        )
        _require_source_file_matches(
            manifest["source"]["exposure_manifest"],
            exposure_path,
            exposure_sha256,
            field_name="source.exposure_manifest",
        )
        replayed_manifest = select_exp004_holdout100(
            label_rows,
            exposure_manifest=exposure_manifest,
            source={
                "labels": _file_provenance_from_hash(
                    labels_path,
                    row_count=len(label_rows),
                    sha256=labels_sha256,
                ),
                "exposure_manifest": _exposure_file_provenance_from_hash(
                    exposure_path,
                    exposure_manifest=exposure_manifest,
                    sha256=exposure_sha256,
                ),
            },
        )
    elif schema == TIMING_V3_EXP004_BROAD_MANIFEST_SCHEMA:
        stage = "broad500"
        if holdout_path is None:
            raise ValueError("broad500 execution input build requires holdout_manifest_path")
        holdout_manifest, holdout_sha256 = _load_exp004_manifest_bound(holdout_path)
        bound_sources.append((holdout_path, holdout_sha256, "holdout_manifest"))
        _require_source_file_matches(
            manifest["source"]["labels"],
            labels_path,
            labels_sha256,
            field_name="source.labels",
        )
        _require_source_file_matches(
            manifest["source"]["exposure_manifest"],
            exposure_path,
            exposure_sha256,
            field_name="source.exposure_manifest",
        )
        _require_source_file_matches(
            manifest["source"]["holdout_manifest"],
            holdout_path,
            holdout_sha256,
            field_name="source.holdout_manifest",
        )
        replayed_manifest = select_exp004_broad500(
            label_rows,
            exposure_manifest=exposure_manifest,
            holdout_manifest=holdout_manifest,
            source={
                "labels": _file_provenance_from_hash(
                    labels_path,
                    row_count=len(label_rows),
                    sha256=labels_sha256,
                ),
                "exposure_manifest": _exposure_file_provenance_from_hash(
                    exposure_path,
                    exposure_manifest=exposure_manifest,
                    sha256=exposure_sha256,
                ),
                "holdout_manifest": _holdout_file_provenance_from_hash(
                    holdout_path,
                    holdout_manifest=holdout_manifest,
                    sha256=holdout_sha256,
                ),
            },
        )
    else:
        raise ValueError(f"unsupported Exp004 execution split manifest schema: {schema!r}")
    if replayed_manifest != manifest:
        raise ValueError("split manifest is not the deterministic Exp004 replay from current sources")

    entries = _manifest_entries(manifest)
    identities = _confirmed_identities_for_entries(label_rows, entries)
    ordered_key_hash = exp004_protocol.ordered_cache_audio_keys_sha256(
        [identity["cache_audio_key"] for identity in identities]
    )
    upstream_source = exp004_protocol.build_exp004_upstream_source(
        source_schema=schema,
        source_path=manifest_path,
        source_fingerprint_sha256=manifest["manifest_fingerprint_sha256"],
        row_count=len(identities),
        ordered_cache_audio_keys_sha256=ordered_key_hash,
    )
    _require_bound_sources_unchanged(bound_sources)
    result = exp004_protocol.build_exp004_execution_inputs(
        stage=stage,
        ordered_identities=identities,
        upstream_source=upstream_source,
        stage_constraints=_stage_constraints_from_replayed_manifest(
            stage=stage,
            split_manifest=manifest,
            holdout_manifest=holdout_manifest if stage == "broad500" else None,
        ),
        identity_rows_jsonl_path=identity_path,
        execution_selection_manifest_path=selection_path,
    )
    _require_bound_sources_unchanged(bound_sources)
    return result


def build_exp004_execution_inputs_from_selected_keys(
    *,
    labels_jsonl_path: str | Path,
    selected_cache_audio_keys: Sequence[str],
    stage: str,
    identity_rows_jsonl_path: str | Path,
    execution_selection_manifest_path: str | Path,
    source_schema: str = TIMING_V3_EXP004_EXPLICIT_SELECTED_KEYS_SCHEMA,
) -> dict[str, Any]:
    """Build compact execution inputs from an explicit selected-key order."""

    if stage != "repair80":
        raise ValueError("explicit selected-key execution inputs are only supported for repair80")
    labels_path = Path(labels_jsonl_path)
    identity_path = Path(identity_rows_jsonl_path)
    selection_path = Path(execution_selection_manifest_path)
    _require_distinct_paths([labels_path, identity_path, selection_path])

    label_rows, labels_sha256 = _load_label_rows_jsonl_bound(labels_path)
    bound_sources = [(labels_path, labels_sha256, "labels")]
    records = _label_records(label_rows)
    selected_keys = _ordered_selected_keys(selected_cache_audio_keys)
    expected_count = exp004_protocol.exp004_stage_count(stage)
    if len(selected_keys) != expected_count:
        raise ValueError(
            f"{stage} selected-key order must contain exactly {expected_count} keys"
        )
    identities = _confirmed_identities_for_cache_key_order(records, selected_keys)
    ordered_key_hash = exp004_protocol.ordered_cache_audio_keys_sha256(selected_keys)
    upstream_fingerprint = exp004_protocol.stable_json_sha256(
        {
            "schema": source_schema,
            "stage": stage,
            "labels_sha256": labels_sha256,
            "selected_cache_audio_keys": selected_keys,
        }
    )
    upstream_source = exp004_protocol.build_exp004_upstream_source(
        source_schema=source_schema,
        source_path=labels_path,
        source_sha256=labels_sha256,
        source_fingerprint_sha256=upstream_fingerprint,
        row_count=len(identities),
        ordered_cache_audio_keys_sha256=ordered_key_hash,
    )
    _require_bound_sources_unchanged(bound_sources)
    result = exp004_protocol.build_exp004_execution_inputs(
        stage=stage,
        ordered_identities=identities,
        upstream_source=upstream_source,
        identity_rows_jsonl_path=identity_path,
        execution_selection_manifest_path=selection_path,
    )
    _require_bound_sources_unchanged(bound_sources)
    return result


def build_exp004_full5050_execution_inputs(
    *,
    labels_jsonl_path: str | Path,
    identity_rows_jsonl_path: str | Path,
    execution_selection_manifest_path: str | Path,
) -> dict[str, Any]:
    """Build compact full5050 execution inputs in current label-inventory order."""

    labels_path = Path(labels_jsonl_path)
    identity_path = Path(identity_rows_jsonl_path)
    selection_path = Path(execution_selection_manifest_path)
    _require_distinct_paths([labels_path, identity_path, selection_path])

    label_rows, labels_sha256 = _load_label_rows_jsonl_bound(labels_path)
    bound_sources = [(labels_path, labels_sha256, "labels")]
    records = _label_records(label_rows)
    expected_count = exp004_protocol.exp004_stage_count("full5050")
    if len(records) != expected_count:
        raise ValueError(
            f"full5050 label inventory must contain exactly {expected_count} rows"
        )
    identities = [_confirmed_identity_from_record(record) for record in records]
    ordered_keys = [identity["cache_audio_key"] for identity in identities]
    ordered_key_hash = exp004_protocol.ordered_cache_audio_keys_sha256(ordered_keys)
    upstream_source = exp004_protocol.build_exp004_upstream_source(
        source_schema=TIMING_V3_LABEL_AUDIO_ROW_SCHEMA,
        source_path=labels_path,
        source_sha256=labels_sha256,
        source_fingerprint_sha256=labels_sha256,
        row_count=len(identities),
        ordered_cache_audio_keys_sha256=ordered_key_hash,
    )
    _require_bound_sources_unchanged(bound_sources)
    result = exp004_protocol.build_exp004_execution_inputs(
        stage="full5050",
        ordered_identities=identities,
        upstream_source=upstream_source,
        identity_rows_jsonl_path=identity_path,
        execution_selection_manifest_path=selection_path,
    )
    _require_bound_sources_unchanged(bound_sources)
    return result


def load_exp004_exposure_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    validate_exp004_exposure_manifest(payload)
    return payload


def load_exp004_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    validate_exp004_manifest(payload)
    return payload


def _load_label_rows_jsonl_bound(path: Path) -> tuple[list[dict[str, Any]], str]:
    data, sha256 = _read_file_bytes_with_sha256(path)
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(data.decode("utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        rows.append(payload)
    _label_records(rows)
    return rows, sha256


def _load_exp004_exposure_manifest_bound(path: Path) -> tuple[dict[str, Any], str]:
    payload, sha256 = _load_json_object_bound(path)
    validate_exp004_exposure_manifest(payload)
    return payload, sha256


def _load_exp004_manifest_bound(path: Path) -> tuple[dict[str, Any], str]:
    payload, sha256 = _load_json_object_bound(path)
    validate_exp004_manifest(payload)
    return payload, sha256


def _load_json_object_bound(path: Path) -> tuple[dict[str, Any], str]:
    data, sha256 = _read_file_bytes_with_sha256(path)
    try:
        payload = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload, sha256


def _read_file_bytes_with_sha256(path: Path) -> tuple[bytes, str]:
    before = path.stat()
    data = path.read_bytes()
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
        raise RuntimeError(f"source changed while reading: {path}")
    return data, hashlib.sha256(data).hexdigest()


def validate_exp004_exposure_manifest(manifest: Mapping[str, Any]) -> frozenset[str]:
    """Return the excluded cache-audio keys or fail closed on schema drift."""

    if not isinstance(manifest, MappingABC):
        raise ValueError("exposure manifest must be a mapping")
    _reject_metric_valued_fields(manifest)
    if set(manifest) != _EXPOSURE_MANIFEST_KEYS:
        missing = sorted(_EXPOSURE_MANIFEST_KEYS.difference(manifest))
        extra = sorted(set(manifest).difference(_EXPOSURE_MANIFEST_KEYS))
        raise ValueError(
            "exposure manifest fields are incomplete or unsupported: "
            f"missing={missing!r} extra={extra!r}"
        )
    if manifest.get("schema_id") != TIMING_V3_EXP004_EXPOSURE_MANIFEST_SCHEMA:
        raise ValueError("exposure manifest schema_id is invalid")
    for field_name in ("generated_from_commit", "generated_at_utc"):
        _required_nonempty_string(manifest.get(field_name), field_name=field_name)
    if manifest.get("minimum_required_exclusion_keys") != EXP004_MINIMUM_REQUIRED_EXCLUSION_KEYS:
        raise ValueError(
            "exposure manifest minimum_required_exclusion_keys must be "
            f"{EXP004_MINIMUM_REQUIRED_EXCLUSION_KEYS}"
        )
    if manifest.get("entries_hash_encoding") != "sha256_canonical_json_sorted_string_array_utf8":
        raise ValueError("exposure manifest entries hash encoding is invalid")

    required_sources = manifest.get("required_sources")
    if required_sources != EXP004_REQUIRED_EXPOSURE_SOURCE_COUNTS:
        raise ValueError("exposure manifest required_sources do not match Exp004")
    sources = manifest.get("sources")
    if not isinstance(sources, MappingABC) or set(sources) != set(EXP004_REQUIRED_EXPOSURE_SOURCE_COUNTS):
        raise ValueError("exposure manifest sources must cover the required source manifests")
    for source_name in EXP004_REQUIRED_EXPOSURE_SOURCE_COUNTS:
        _validate_exposure_source(
            sources.get(source_name),
            field_name=f"sources.{source_name}",
            expected_count=EXP004_REQUIRED_EXPOSURE_SOURCE_COUNTS[source_name],
        )
    if manifest.get("exposure_scan_source_sha256") != _json_sha256(sources):
        raise ValueError("exposure_scan_source_sha256 is stale or does not replay")

    entries = _exposure_entries(manifest)
    normalized_entries = _normalize_exposure_entries(entries)
    if entries != normalized_entries:
        raise ValueError("exposure manifest entries must be sorted by cache_audio_key")
    keys = {entry["cache_audio_key"] for entry in entries}
    entry_count = manifest.get("entry_count")
    if isinstance(entry_count, bool) or not isinstance(entry_count, int):
        raise ValueError("exposure manifest entry_count must be an integer")
    if entry_count != len(entries):
        raise ValueError(
            f"exposure manifest entry_count={entry_count} does not match {len(entries)} entries"
        )
    if len(keys) < EXP004_MINIMUM_REQUIRED_EXCLUSION_KEYS:
        raise ValueError(
            "exposure manifest must contain at least "
            f"{EXP004_MINIMUM_REQUIRED_EXCLUSION_KEYS} unique cache audio keys"
        )
    if manifest.get("entries_sha256") != _key_set_sha256(keys):
        raise ValueError("exposure manifest entries_sha256 does not replay sorted keys")

    source_union: set[str] = set()
    source_sets: dict[str, set[str]] = {}
    for source_name, source in sources.items():
        raw_keys = source.get("cache_audio_keys")
        if isinstance(raw_keys, SequenceABC) and not isinstance(raw_keys, (str, bytes)):
            source_sets[str(source_name)] = set(str(key) for key in raw_keys)
            source_union.update(source_sets[str(source_name)])
    for source_name, source_keys in source_sets.items():
        source_entries = {
            entry["cache_audio_key"]
            for entry in entries
            if entry["exposure_source"] == source_name
        }
        if source_entries != source_keys:
            raise ValueError(
                f"exposure entries for {source_name!r} do not match source key provenance"
            )
    for left_name, left_keys in source_sets.items():
        for right_name, right_keys in source_sets.items():
            if left_name >= right_name:
                continue
            overlap = sorted(left_keys.intersection(right_keys))
            if overlap:
                raise ValueError(
                    f"exposure source key sets overlap for {left_name!r}/{right_name!r}: "
                    f"{overlap[:3]!r}"
                )
    if not source_union.issubset(keys):
        raise ValueError("required exposure source keys are missing from entries")
    return frozenset(keys)


def validate_exp004_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed on malformed, duplicated, overlapping, or edited split manifests."""

    if not isinstance(manifest, MappingABC):
        raise ValueError("manifest must be a mapping")
    _reject_metric_or_oracle_manifest_fields(manifest)
    schema = manifest.get("schema")
    if schema not in {
        TIMING_V3_EXP004_HOLDOUT_MANIFEST_SCHEMA,
        TIMING_V3_EXP004_BROAD_MANIFEST_SCHEMA,
    }:
        raise ValueError(f"unsupported Experiment 004 manifest schema: {schema!r}")
    expected_fields = (
        _HOLDOUT_MANIFEST_KEYS
        if schema == TIMING_V3_EXP004_HOLDOUT_MANIFEST_SCHEMA
        else _BROAD_MANIFEST_KEYS
    )
    _require_exact_mapping_fields(manifest, expected_fields, "manifest")
    if manifest.get("experiment") != "timing_v3_experiment_004":
        raise ValueError("manifest experiment identifier is invalid")
    expected_fingerprint = manifest.get("manifest_fingerprint_sha256")
    if not _is_sha256(expected_fingerprint):
        raise ValueError("manifest_fingerprint_sha256 must be a lowercase SHA-256 digest")
    fingerprint_body = dict(manifest)
    fingerprint_body.pop("manifest_fingerprint_sha256", None)
    actual_fingerprint = _json_sha256(fingerprint_body)
    if expected_fingerprint != actual_fingerprint:
        raise ValueError(
            "manifest fingerprint mismatch: "
            f"expected {expected_fingerprint}, computed {actual_fingerprint}"
        )

    entries = _manifest_entries(manifest)
    expected_count = manifest.get("selected_audio_count")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise ValueError("selected_audio_count must be an integer")
    if expected_count != len(entries):
        raise ValueError(
            f"selected_audio_count={expected_count} does not match {len(entries)} selected rows"
        )

    _validate_split_entries(entries)
    selected_keys = {str(entry["cache_audio_key"]) for entry in entries}
    if manifest.get("selected_cache_audio_keys_sha256") != _key_set_sha256(selected_keys):
        raise ValueError("selected cache-audio-key hash does not replay")

    source = manifest.get("source")
    if not isinstance(source, MappingABC):
        raise ValueError("manifest source provenance must be a mapping")
    _validate_source_entry(source.get("labels"), field_name="source.labels", expected_row_count=None)
    exposure_summary = manifest.get("exposure")
    exposure_keys = _validate_exposure_summary(exposure_summary)
    _validate_exposure_manifest_source_entry(
        source.get("exposure_manifest"),
        exposure_summary=exposure_summary,
    )
    selected_exposure_overlap = sorted(selected_keys.intersection(exposure_keys))
    if selected_exposure_overlap:
        raise ValueError(
            "manifest selected rows overlap exposure exclusions: "
            f"{selected_exposure_overlap[:3]!r}"
        )

    if schema == TIMING_V3_EXP004_HOLDOUT_MANIFEST_SCHEMA:
        _validate_holdout_manifest_details(manifest, entries)
    else:
        _validate_broad_manifest_details(manifest, entries, source=source)


def materialize_label_subset(
    label_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return full label rows in Exp004 manifest order with split metadata."""

    records = _label_records(label_rows)
    validate_exp004_manifest(manifest)
    by_cache_key = {record.cache_audio_key: record for record in records}
    rows: list[dict[str, Any]] = []
    for entry in _manifest_entries(manifest):
        cache_audio_key = str(entry["cache_audio_key"])
        record = by_cache_key.get(cache_audio_key)
        if record is None:
            raise ValueError(
                f"manifest cache audio key is absent from label inventory: {cache_audio_key!r}"
            )
        _validate_entry_against_record(entry, record)
        payload = dict(record.row)
        payload["cache_audio_key"] = cache_audio_key
        payload["experiment_split"] = {
            "schema": TIMING_V3_EXP004_SPLIT_ANNOTATION_SCHEMA,
            "experiment": "timing_v3_experiment_004",
            "manifest_schema": manifest["schema"],
            "manifest_fingerprint_sha256": manifest["manifest_fingerprint_sha256"],
            "manifest_stage": manifest["stage"],
            "selection_stage": entry["stage"],
            "quota_assignment": entry.get("quota_assignment"),
            "selection_substage": entry["selection_substage"],
            "selection_rank": entry["selection_rank"],
            "selection_hash_sha256": entry["selection_hash_sha256"],
        }
        rows.append(payload)
    return rows


def _confirmed_identities_for_entries(
    label_rows: Sequence[Mapping[str, Any]],
    entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    records = _label_records(label_rows)
    by_cache_key = {record.cache_audio_key: record for record in records}
    identities: list[dict[str, str]] = []
    for index, entry in enumerate(entries):
        cache_audio_key = _required_nonempty_string(
            entry.get("cache_audio_key"),
            field_name=f"manifest selected[{index}].cache_audio_key",
        )
        record = by_cache_key.get(cache_audio_key)
        if record is None:
            raise ValueError(
                f"manifest cache audio key is absent from current labels: {cache_audio_key!r}"
            )
        _validate_entry_against_record(entry, record)
        identities.append(_confirmed_identity_from_record(record))
    return identities


def _require_source_file_matches(
    source: Any,
    path: Path,
    sha256: str,
    *,
    field_name: str,
) -> None:
    if not isinstance(source, MappingABC):
        raise ValueError(f"{field_name} must be a provenance mapping")
    if source.get("path") != path.as_posix():
        raise ValueError(f"{field_name}.path does not match the exact requested path")
    if source.get("sha256") != sha256:
        raise ValueError(f"{field_name}.sha256 does not match current file bytes")


def _require_bound_sources_unchanged(
    sources: Sequence[tuple[Path, str, str]],
) -> None:
    for path, expected_sha256, field_name in sources:
        actual_sha256 = _file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"{field_name} changed during execution input build: {path}")


def _stage_constraints_from_replayed_manifest(
    *,
    stage: str,
    split_manifest: Mapping[str, Any],
    holdout_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if stage == "holdout100":
        degraded = split_manifest["selection"]["degraded_underfilled"]
        degraded_quotas = [quota for quota in EXP004_PRIORITY if quota in degraded]
        return {
            "schema": exp004_protocol.STAGE_CONSTRAINT_SCHEMA,
            "stage": stage,
            "quota_degraded": bool(degraded_quotas),
            "degraded_quotas": degraded_quotas,
            "broad_underfilled": False,
        }
    if stage == "broad500":
        if holdout_manifest is None:
            raise ValueError("broad500 stage constraints require the holdout manifest")
        holdout_degraded = holdout_manifest["selection"]["degraded_underfilled"]
        degraded_quotas = [quota for quota in EXP004_PRIORITY if quota in holdout_degraded]
        broad_degraded = split_manifest["selection"]["degraded_underfilled"]
        return {
            "schema": exp004_protocol.STAGE_CONSTRAINT_SCHEMA,
            "stage": stage,
            "quota_degraded": bool(degraded_quotas),
            "degraded_quotas": degraded_quotas,
            "broad_underfilled": bool(broad_degraded),
        }
    return exp004_protocol.default_stage_constraints(stage)


def _confirmed_identities_for_cache_key_order(
    records: Sequence[_LabelRecord],
    selected_cache_audio_keys: Sequence[str],
) -> list[dict[str, str]]:
    by_cache_key = {record.cache_audio_key: record for record in records}
    identities: list[dict[str, str]] = []
    for index, cache_audio_key in enumerate(selected_cache_audio_keys):
        record = by_cache_key.get(cache_audio_key)
        if record is None:
            raise ValueError(
                f"selected cache audio key at index {index} is absent from current labels: "
                f"{cache_audio_key!r}"
            )
        identities.append(_confirmed_identity_from_record(record))
    return identities


def _confirmed_identity_from_record(record: _LabelRecord) -> dict[str, str]:
    return {
        "cache_audio_key": record.cache_audio_key,
        "audio_group_key": record.audio_group_key,
        "resolved_audio_path": _required_nonempty_string(
            record.row.get("resolved_audio_path"),
            field_name=f"label row {record.cache_audio_key!r}.resolved_audio_path",
        ),
    }


def _ordered_selected_keys(values: Sequence[str]) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, SequenceABC):
        raise ValueError("selected_cache_audio_keys must be a sequence of strings")
    keys: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        key = _required_nonempty_string(
            value,
            field_name=f"selected_cache_audio_keys[{index}]",
        )
        if key in seen:
            raise ValueError(f"duplicate selected cache_audio_key: {key!r}")
        seen.add(key)
        keys.append(key)
    return keys


def _label_records(label_rows: Sequence[Mapping[str, Any]]) -> list[_LabelRecord]:
    if isinstance(label_rows, (str, bytes)) or not isinstance(label_rows, SequenceABC):
        raise ValueError("label_rows must be a sequence of mappings")
    records: list[_LabelRecord] = []
    audio_group_keys: set[str] = set()
    cache_audio_keys: set[str] = set()
    for index, row in enumerate(label_rows):
        if not isinstance(row, MappingABC):
            raise ValueError(f"label_rows[{index}] must be a mapping")
        schema = row.get("schema")
        if schema != TIMING_V3_LABEL_AUDIO_ROW_SCHEMA:
            raise ValueError(
                f"label_rows[{index}] must use schema "
                f"{TIMING_V3_LABEL_AUDIO_ROW_SCHEMA!r}, got {schema!r}"
            )
        audio_group_key = _required_nonempty_string(
            row.get("audio_group_key"),
            field_name=f"label_rows[{index}].audio_group_key",
        )
        source = row.get("source")
        if not isinstance(source, MappingABC):
            raise ValueError(f"label_rows[{index}].source must be a mapping")
        cache_audio_key = _required_nonempty_string(
            source.get("cache_audio_key"),
            field_name=f"label_rows[{index}].source.cache_audio_key",
        )
        long_track = source.get("long_track")
        if not isinstance(long_track, bool):
            raise ValueError(f"label_rows[{index}].source.long_track must be a boolean")
        label = row.get("label")
        if not isinstance(label, MappingABC):
            raise ValueError(f"label_rows[{index}].label must be a mapping")
        label_stratum = _required_nonempty_string(
            label.get("stratum"),
            field_name=f"label_rows[{index}].label.stratum",
        )
        if label_stratum not in {
            LABEL_STABLE,
            LABEL_JUMP_CANDIDATE,
            LABEL_DENSE,
            LABEL_RAMP_CANDIDATE,
            LABEL_AMBIGUOUS,
        }:
            raise ValueError(
                f"label_rows[{index}].label.stratum is unsupported: {label_stratum!r}"
            )
        if audio_group_key in audio_group_keys:
            raise ValueError(f"duplicate label audio_group_key: {audio_group_key!r}")
        if cache_audio_key in cache_audio_keys:
            raise ValueError(f"duplicate label cache_audio_key: {cache_audio_key!r}")
        audio_group_keys.add(audio_group_key)
        cache_audio_keys.add(cache_audio_key)
        records.append(
            _LabelRecord(
                row=row,
                audio_group_key=audio_group_key,
                cache_audio_key=cache_audio_key,
                label_stratum=label_stratum,
                long_track=long_track,
            )
        )
    return records


def _quota_assignment(record: _LabelRecord) -> str:
    if record.label_stratum == LABEL_RAMP_CANDIDATE:
        return "ramp_audit"
    if record.label_stratum == LABEL_AMBIGUOUS:
        return "anomaly"
    if record.long_track:
        return "long"
    if record.label_stratum == LABEL_DENSE:
        return "dense"
    if record.label_stratum == LABEL_JUMP_CANDIDATE:
        return "jump"
    if record.label_stratum == LABEL_STABLE:
        return "stable"
    raise AssertionError(f"unreachable label stratum {record.label_stratum!r}")


def _entry_quota_assignment(entry: Mapping[str, Any]) -> str:
    label_stratum = entry.get("label_stratum")
    long_track = entry.get("source_long_track")
    if not isinstance(long_track, bool):
        raise ValueError("manifest source_long_track must be a boolean")
    if label_stratum == LABEL_RAMP_CANDIDATE:
        return "ramp_audit"
    if label_stratum == LABEL_AMBIGUOUS:
        return "anomaly"
    if long_track:
        return "long"
    if label_stratum == LABEL_DENSE:
        return "dense"
    if label_stratum == LABEL_JUMP_CANDIDATE:
        return "jump"
    if label_stratum == LABEL_STABLE:
        return "stable"
    raise ValueError(f"manifest label_stratum is unsupported: {label_stratum!r}")


def _ordered_records(records: Sequence[_LabelRecord], *, seed: str) -> list[_LabelRecord]:
    return sorted(
        records,
        key=lambda record: (
            _selection_hash(seed, record.cache_audio_key),
            record.cache_audio_key,
            record.audio_group_key,
        ),
    )


def _selection_entry(
    record: _LabelRecord,
    *,
    stage: str,
    quota_assignment: str | None,
    selection_substage: str,
    selection_rank: int,
    selection_hash_sha256: str,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "quota_assignment": quota_assignment,
        "selection_substage": selection_substage,
        "selection_rank": selection_rank,
        "selection_hash_sha256": selection_hash_sha256,
        "cache_audio_key": record.cache_audio_key,
        "audio_group_key": record.audio_group_key,
        "resolved_audio_path": record.row.get("resolved_audio_path"),
        "label_stratum": record.label_stratum,
        "source_long_track": record.long_track,
    }


def _validate_entry_against_record(entry: Mapping[str, Any], record: _LabelRecord) -> None:
    if entry.get("cache_audio_key") != record.cache_audio_key:
        raise ValueError(
            f"manifest cache audio identity mismatch for {record.cache_audio_key!r}"
        )
    if entry.get("audio_group_key") != record.audio_group_key:
        raise ValueError(
            f"manifest audio group identity mismatch for cache key {record.cache_audio_key!r}"
        )
    if entry.get("resolved_audio_path") != record.row.get("resolved_audio_path"):
        raise ValueError(
            f"manifest resolved audio path is stale for cache key {record.cache_audio_key!r}"
        )
    if entry.get("label_stratum") != record.label_stratum:
        raise ValueError(
            f"manifest label stratum is stale for cache key {record.cache_audio_key!r}"
        )
    if entry.get("source_long_track") is not record.long_track:
        raise ValueError(
            f"manifest long-track flag is stale for cache key {record.cache_audio_key!r}"
        )
    stage = entry.get("stage")
    expected_quota = _quota_assignment(record) if stage == "holdout100" else None
    if stage == "holdout100" and entry.get("quota_assignment") != expected_quota:
        raise ValueError(
            f"manifest quota assignment is stale for cache key {record.cache_audio_key!r}"
        )
    seed = {
        ("holdout100", "quota"): EXP004_HOLDOUT_SEED,
        ("holdout100", "deficit_fill"): EXP004_HOLDOUT_DEFICIT_SEED,
        ("broad500_added", "broad500_added"): EXP004_BROAD_SEED,
    }.get((str(stage), str(entry.get("selection_substage"))))
    if seed is None:
        raise ValueError("manifest selection substage is invalid")
    if entry.get("selection_hash_sha256") != _selection_hash(seed, record.cache_audio_key):
        raise ValueError(
            f"manifest selection hash is stale for cache key {record.cache_audio_key!r}"
        )


def _validate_split_entries(entries: Sequence[Mapping[str, Any]]) -> None:
    audio_group_keys: set[str] = set()
    cache_audio_keys: set[str] = set()
    stage_ranks: dict[tuple[str, str], set[int]] = {}
    for index, entry in enumerate(entries):
        _require_exact_mapping_fields(
            entry,
            _SPLIT_SELECTED_ENTRY_KEYS,
            f"selected[{index}]",
        )
        audio_group_key = _required_nonempty_string(
            entry.get("audio_group_key"),
            field_name=f"selected[{index}].audio_group_key",
        )
        cache_audio_key = _required_nonempty_string(
            entry.get("cache_audio_key"),
            field_name=f"selected[{index}].cache_audio_key",
        )
        if audio_group_key in audio_group_keys:
            raise ValueError(f"duplicate manifest audio_group_key: {audio_group_key!r}")
        if cache_audio_key in cache_audio_keys:
            raise ValueError(f"duplicate manifest cache_audio_key: {cache_audio_key!r}")
        audio_group_keys.add(audio_group_key)
        cache_audio_keys.add(cache_audio_key)
        stage = _required_nonempty_string(
            entry.get("stage"),
            field_name=f"selected[{index}].stage",
        )
        substage = _required_nonempty_string(
            entry.get("selection_substage"),
            field_name=f"selected[{index}].selection_substage",
        )
        quota = entry.get("quota_assignment")
        if quota is not None and quota not in EXP004_PRIORITY:
            raise ValueError(f"selected[{index}].quota_assignment is invalid: {quota!r}")
        rank = entry.get("selection_rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError(f"selected[{index}].selection_rank must be a positive integer")
        rank_key = (stage, substage if substage != "quota" else f"quota:{quota}")
        ranks = stage_ranks.setdefault(rank_key, set())
        if rank in ranks:
            raise ValueError(
                f"duplicate selection_rank={rank} for stage/substage {rank_key!r}"
            )
        ranks.add(rank)
        seed = {
            ("holdout100", "quota"): EXP004_HOLDOUT_SEED,
            ("holdout100", "deficit_fill"): EXP004_HOLDOUT_DEFICIT_SEED,
            ("broad500_added", "broad500_added"): EXP004_BROAD_SEED,
        }.get((stage, substage))
        if seed is None:
            raise ValueError(f"selected[{index}] has invalid stage/substage")
        if entry.get("selection_hash_sha256") != _selection_hash(seed, cache_audio_key):
            raise ValueError(f"selected[{index}] selection hash does not replay")


def _validate_holdout_manifest_details(
    manifest: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
) -> None:
    if manifest.get("stage") != "holdout100":
        raise ValueError("holdout manifest stage must be 'holdout100'")
    if manifest.get("seed") != EXP004_HOLDOUT_SEED:
        raise ValueError("holdout manifest seed does not match the frozen seed")
    if manifest.get("deficit_seed") != EXP004_HOLDOUT_DEFICIT_SEED:
        raise ValueError("holdout deficit seed does not match the frozen seed")
    if len(entries) != EXP004_HOLDOUT_AUDIO_COUNT:
        raise ValueError(
            f"holdout manifest must contain {EXP004_HOLDOUT_AUDIO_COUNT} audio groups"
        )
    selection = manifest.get("selection")
    if not isinstance(selection, MappingABC):
        raise ValueError("holdout selection must be a mapping")
    if selection.get("method") != "exclusive_priority_sha256_with_preregistered_deficit_fill":
        raise ValueError("holdout selection method does not match the frozen method")
    if selection.get("priority_order") != list(EXP004_PRIORITY):
        raise ValueError("holdout priority order does not match the frozen order")
    if selection.get("quotas") != EXP004_HOLDOUT_QUOTAS:
        raise ValueError("holdout quotas do not match the frozen quotas")
    if selection.get("target_audio_count") != EXP004_HOLDOUT_AUDIO_COUNT:
        raise ValueError("holdout target_audio_count must be 100")
    available_counts = selection.get("available_counts_after_exposure_exclusion")
    if not isinstance(available_counts, MappingABC) or set(available_counts) != set(
        EXP004_PRIORITY
    ):
        raise ValueError("holdout available counts must cover every frozen quota")
    degraded = selection.get("degraded_underfilled")
    if not isinstance(degraded, MappingABC):
        raise ValueError("holdout degraded_underfilled must be a mapping")
    for quota in EXP004_PRIORITY:
        available = available_counts.get(quota)
        if isinstance(available, bool) or not isinstance(available, int) or available < 0:
            raise ValueError(f"holdout available count for {quota!r} must be an integer")
        requested = EXP004_HOLDOUT_QUOTAS[quota]
        if available < requested:
            expected = {
                "requested": requested,
                "available": available,
                "deficit": requested - available,
            }
            if degraded.get(quota) != expected:
                raise ValueError(f"holdout degraded quota {quota!r} is not recorded")
        elif quota in degraded:
            raise ValueError(f"holdout quota {quota!r} is marked degraded despite availability")
    deficit_fill = selection.get("deficit_fill")
    if not isinstance(deficit_fill, MappingABC):
        raise ValueError("holdout deficit_fill must be a mapping")
    if deficit_fill.get("seed") != EXP004_HOLDOUT_DEFICIT_SEED:
        raise ValueError("holdout deficit fill seed is invalid")
    if deficit_fill.get("priority_order") != list(EXP004_DEFICIT_PRIORITY):
        raise ValueError("holdout deficit fill priority order is invalid")
    fill_count = deficit_fill.get("fill_count")
    if isinstance(fill_count, bool) or not isinstance(fill_count, int) or fill_count < 0:
        raise ValueError("holdout deficit fill count must be a non-negative integer")
    if fill_count != sum(item["deficit"] for item in degraded.values()):
        raise ValueError("holdout deficit fill count does not match degraded deficits")
    _validate_holdout_entry_order(entries, degraded=degraded, deficit_fill_count=fill_count)
    if manifest.get("selected_counts") != _selected_counts(entries):
        raise ValueError("holdout selected_counts do not match selected entries")


def _validate_holdout_entry_order(
    entries: Sequence[Mapping[str, Any]],
    *,
    degraded: Mapping[str, Mapping[str, int]],
    deficit_fill_count: int,
) -> None:
    cursor = 0
    for quota in EXP004_PRIORITY:
        requested = EXP004_HOLDOUT_QUOTAS[quota]
        quota_count = requested - int(degraded.get(quota, {}).get("deficit", 0))
        quota_entries = list(entries[cursor : cursor + quota_count])
        cursor += quota_count
        if any(entry.get("stage") != "holdout100" for entry in quota_entries):
            raise ValueError("holdout entries must all use stage='holdout100'")
        if any(entry.get("selection_substage") != "quota" for entry in quota_entries):
            raise ValueError("holdout quota entries must use selection_substage='quota'")
        if any(entry.get("quota_assignment") != quota for entry in quota_entries):
            raise ValueError("holdout entries do not follow the frozen exclusive quota order")
        if [entry.get("selection_rank") for entry in quota_entries] != list(
            range(1, quota_count + 1)
        ):
            raise ValueError(
                f"holdout quota {quota!r} selection ranks must be contiguous from 1"
            )
        _validate_ordered_entries(quota_entries, quota=quota, seed=EXP004_HOLDOUT_SEED)

    deficit_entries = list(entries[cursor:])
    if len(deficit_entries) != deficit_fill_count:
        raise ValueError("holdout deficit fill entry count does not match recorded deficit")
    if deficit_entries:
        if any(entry.get("selection_substage") != "deficit_fill" for entry in deficit_entries):
            raise ValueError("holdout deficit fill entries must use selection_substage='deficit_fill'")
        if [entry.get("selection_rank") for entry in deficit_entries] != list(
            range(1, len(deficit_entries) + 1)
        ):
            raise ValueError("holdout deficit fill ranks must be contiguous from 1")
        cursor = 0
        for quota in EXP004_DEFICIT_PRIORITY:
            quota_entries = [
                entry for entry in deficit_entries if entry.get("quota_assignment") == quota
            ]
            contiguous = deficit_entries[cursor : cursor + len(quota_entries)]
            if list(contiguous) != quota_entries:
                raise ValueError("holdout deficit fill entries do not follow class priority")
            cursor += len(quota_entries)
            _validate_ordered_entries(
                quota_entries,
                quota=quota,
                seed=EXP004_HOLDOUT_DEFICIT_SEED,
            )
        if cursor != len(deficit_entries):
            raise ValueError("holdout deficit fill contains unsupported quota assignments")


def _validate_ordered_entries(
    entries: Sequence[Mapping[str, Any]],
    *,
    quota: str,
    seed: str,
) -> None:
    expected_order = sorted(
        entries,
        key=lambda entry: (
            _selection_hash(seed, str(entry["cache_audio_key"])),
            str(entry["cache_audio_key"]),
            str(entry["audio_group_key"]),
        ),
    )
    if list(entries) != expected_order:
        raise ValueError(f"holdout quota {quota!r} entries are not in frozen SHA-256 rank order")
    for entry in entries:
        if _entry_quota_assignment(entry) != quota:
            raise ValueError(f"holdout entry does not satisfy quota assignment {quota!r}")


def _validate_broad_manifest_details(
    manifest: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    *,
    source: Mapping[str, Any],
) -> None:
    if manifest.get("stage") != "broad500":
        raise ValueError("broad manifest stage must be 'broad500'")
    if manifest.get("seed") != EXP004_BROAD_SEED:
        raise ValueError("broad manifest seed does not match the frozen seed")
    selection = manifest.get("selection")
    if not isinstance(selection, MappingABC):
        raise ValueError("broad selection must be a mapping")
    if selection.get("method") != "holdout100_then_lowest_sha256_seed_nul_cache_audio_key":
        raise ValueError("broad selection method does not match the frozen method")
    if selection.get("holdout_audio_count") != EXP004_HOLDOUT_AUDIO_COUNT:
        raise ValueError("broad selection holdout_audio_count must be 100")
    added_count = selection.get("added_audio_count")
    if isinstance(added_count, bool) or not isinstance(added_count, int) or added_count < 0:
        raise ValueError("broad added_audio_count must be a non-negative integer")
    if selection.get("target_audio_count") != EXP004_BROAD_AUDIO_COUNT:
        raise ValueError("broad selection target_audio_count must be 500")
    candidate_count = selection.get("added_candidate_count_after_exclusions")
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int):
        raise ValueError("broad added candidate count must be an integer")
    if added_count != min(candidate_count, EXP004_BROAD_ADDED_AUDIO_COUNT):
        raise ValueError("broad added count does not match candidate availability")
    degraded = selection.get("degraded_underfilled")
    if not isinstance(degraded, MappingABC):
        raise ValueError("broad degraded_underfilled must be a mapping")
    if added_count < EXP004_BROAD_ADDED_AUDIO_COUNT:
        expected = {
            "requested": EXP004_BROAD_ADDED_AUDIO_COUNT,
            "available": added_count,
            "deficit": EXP004_BROAD_ADDED_AUDIO_COUNT - added_count,
        }
        if degraded.get("broad500_added") != expected:
            raise ValueError("broad underfilled added stage is not recorded")
    elif degraded:
        raise ValueError("broad degraded_underfilled must be empty when broad500 is filled")

    holdout_entries = entries[:EXP004_HOLDOUT_AUDIO_COUNT]
    added_entries = entries[EXP004_HOLDOUT_AUDIO_COUNT:]
    if len(holdout_entries) != EXP004_HOLDOUT_AUDIO_COUNT:
        raise ValueError("broad manifest must start with 100 holdout entries")
    if any(entry.get("stage") != "holdout100" for entry in holdout_entries):
        raise ValueError("the first 100 broad entries must be the frozen holdout")
    if any(entry.get("stage") != "broad500_added" for entry in added_entries):
        raise ValueError("the final broad entries must be broad500_added")
    if len(added_entries) != added_count:
        raise ValueError("broad selected entries do not match added_audio_count")
    if any(entry.get("quota_assignment") is not None for entry in added_entries):
        raise ValueError("broad added entries must not have quota assignments")
    if any(entry.get("selection_substage") != "broad500_added" for entry in added_entries):
        raise ValueError("broad added entries must use selection_substage='broad500_added'")
    if [entry.get("selection_rank") for entry in added_entries] != list(
        range(1, added_count + 1)
    ):
        raise ValueError("broad added selection ranks must be contiguous from 1")
    expected_added_order = sorted(
        added_entries,
        key=lambda entry: (
            str(entry["selection_hash_sha256"]),
            str(entry["cache_audio_key"]),
            str(entry["audio_group_key"]),
        ),
    )
    if list(added_entries) != expected_added_order:
        raise ValueError("broad added entries are not in frozen SHA-256 rank order")
    if manifest.get("selected_counts") != {
        "holdout100": EXP004_HOLDOUT_AUDIO_COUNT,
        "broad500_added": added_count,
    }:
        raise ValueError("broad selected_counts do not match the frozen holdout+added split")
    holdout = manifest.get("holdout")
    if not isinstance(holdout, MappingABC):
        raise ValueError("broad holdout provenance must be a mapping")
    if holdout.get("schema") != TIMING_V3_EXP004_HOLDOUT_MANIFEST_SCHEMA:
        raise ValueError("broad holdout provenance schema is invalid")
    if not _is_sha256(holdout.get("manifest_fingerprint_sha256")):
        raise ValueError("broad holdout manifest fingerprint is invalid")
    if holdout.get("cache_audio_keys_sha256") != _key_set_sha256(
        {str(entry["cache_audio_key"]) for entry in holdout_entries}
    ):
        raise ValueError("broad holdout cache-key hash does not match its first 100 rows")
    if holdout.get("ordered_cache_audio_keys_sha256") != _ordered_key_list_sha256(
        [str(entry["cache_audio_key"]) for entry in holdout_entries]
    ):
        raise ValueError("broad ordered holdout prefix cache-key hash does not replay")
    if holdout.get("selected_ordered_sha256") != _selected_ordered_sha256(holdout_entries):
        raise ValueError("broad ordered holdout prefix identity hash does not replay")
    _validate_holdout_manifest_source_entry(
        source.get("holdout_manifest"),
        holdout_summary=holdout,
    )


def _selected_counts(entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {quota: 0 for quota in EXP004_PRIORITY}
    counts["deficit_fill"] = 0
    for entry in entries:
        if entry.get("selection_substage") == "deficit_fill":
            counts["deficit_fill"] += 1
        elif entry.get("stage") == "holdout100":
            quota = str(entry.get("quota_assignment"))
            if quota in EXP004_PRIORITY:
                counts[quota] += 1
    return counts


def _resolved_holdout_source(
    source: Mapping[str, Any] | None,
    *,
    records: Sequence[_LabelRecord],
    exposure_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = (
        {
            "labels": _in_memory_source_provenance(
                [record.row for record in sorted(records, key=lambda item: item.cache_audio_key)]
            ),
            "exposure_manifest": _exposure_manifest_provenance(exposure_manifest),
        }
        if source is None
        else dict(source)
    )
    _validate_source_entry(
        resolved.get("labels"),
        field_name="source.labels",
        expected_row_count=len(records),
    )
    _validate_source_entry(
        resolved.get("exposure_manifest"),
        field_name="source.exposure_manifest",
        expected_row_count=None,
    )
    return resolved


def _resolved_broad_source(
    source: Mapping[str, Any] | None,
    *,
    records: Sequence[_LabelRecord],
    exposure_manifest: Mapping[str, Any],
    holdout_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = (
        {
            "labels": _in_memory_source_provenance(
                [record.row for record in sorted(records, key=lambda item: item.cache_audio_key)]
            ),
            "exposure_manifest": _exposure_manifest_provenance(exposure_manifest),
            "holdout_manifest": _holdout_manifest_provenance(holdout_manifest),
        }
        if source is None
        else dict(source)
    )
    _validate_source_entry(
        resolved.get("labels"),
        field_name="source.labels",
        expected_row_count=len(records),
    )
    _validate_source_entry(
        resolved.get("exposure_manifest"),
        field_name="source.exposure_manifest",
        expected_row_count=None,
    )
    _validate_source_entry(
        resolved.get("holdout_manifest"),
        field_name="source.holdout_manifest",
        expected_row_count=EXP004_HOLDOUT_AUDIO_COUNT,
    )
    return resolved


def _exposure_summary(
    exposure_manifest: Mapping[str, Any],
    exposure_keys: frozenset[str],
) -> dict[str, Any]:
    return {
        "schema_id": exposure_manifest["schema_id"],
        "exposure_scan_source_sha256": exposure_manifest["exposure_scan_source_sha256"],
        "entries_sha256": exposure_manifest["entries_sha256"],
        "entry_count": exposure_manifest["entry_count"],
        "cache_audio_key_count": len(exposure_keys),
        "cache_audio_keys": sorted(exposure_keys),
        "cache_audio_keys_sha256": _key_set_sha256(set(exposure_keys)),
        "hash_encoding": "sha256_canonical_json_sorted_string_array_utf8",
    }


def _validate_exposure_summary(value: Any) -> set[str]:
    if not isinstance(value, MappingABC):
        raise ValueError("manifest exposure summary must be a mapping")
    if value.get("schema_id") != TIMING_V3_EXP004_EXPOSURE_MANIFEST_SCHEMA:
        raise ValueError("manifest exposure schema_id is invalid")
    if not _is_sha256(value.get("exposure_scan_source_sha256")):
        raise ValueError("manifest exposure source hash is invalid")
    if not _is_sha256(value.get("entries_sha256")):
        raise ValueError("manifest exposure entries hash is invalid")
    if value.get("hash_encoding") != "sha256_canonical_json_sorted_string_array_utf8":
        raise ValueError("manifest exposure hash encoding is invalid")
    raw_keys = value.get("cache_audio_keys")
    if isinstance(raw_keys, (str, bytes)) or not isinstance(raw_keys, SequenceABC):
        raise ValueError("manifest exposure cache_audio_keys must be a sorted string list")
    keys = _validated_key_set(list(raw_keys), field_name="manifest exposure cache_audio_keys")
    if list(raw_keys) != sorted(keys):
        raise ValueError("manifest exposure cache_audio_keys must be sorted")
    if value.get("cache_audio_key_count") != len(keys):
        raise ValueError("manifest exposure cache_audio_key_count does not match")
    if value.get("entry_count") != len(keys):
        raise ValueError("manifest exposure entry_count does not match")
    if value.get("cache_audio_keys_sha256") != _key_set_sha256(keys):
        raise ValueError("manifest exposure cache-audio-key hash does not replay")
    if value.get("entries_sha256") != value.get("cache_audio_keys_sha256"):
        raise ValueError("manifest exposure entries hash does not match key-set hash")
    return keys


def _exposure_manifest_provenance(exposure_manifest: Mapping[str, Any]) -> dict[str, Any]:
    exposure_keys = validate_exp004_exposure_manifest(exposure_manifest)
    cache_audio_keys_sha256 = _key_set_sha256(set(exposure_keys))
    return {
        "path": None,
        "sha256": _json_sha256(exposure_manifest),
        "row_count": len(exposure_keys),
        "schema_id": exposure_manifest["schema_id"],
        "exposure_scan_source_sha256": exposure_manifest["exposure_scan_source_sha256"],
        "entries_sha256": exposure_manifest["entries_sha256"],
        "cache_audio_keys_sha256": cache_audio_keys_sha256,
        "hash_encoding": "sha256_canonical_json_utf8",
    }


def _exposure_file_provenance(
    path: Path,
    *,
    exposure_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    exposure_keys = validate_exp004_exposure_manifest(exposure_manifest)
    cache_audio_keys_sha256 = _key_set_sha256(set(exposure_keys))
    return {
        "path": path.as_posix(),
        "sha256": _file_sha256(path),
        "row_count": len(exposure_keys),
        "schema_id": exposure_manifest["schema_id"],
        "exposure_scan_source_sha256": exposure_manifest["exposure_scan_source_sha256"],
        "entries_sha256": exposure_manifest["entries_sha256"],
        "cache_audio_keys_sha256": cache_audio_keys_sha256,
    }


def _exposure_file_provenance_from_hash(
    path: Path,
    *,
    exposure_manifest: Mapping[str, Any],
    sha256: str,
) -> dict[str, Any]:
    exposure_keys = validate_exp004_exposure_manifest(exposure_manifest)
    cache_audio_keys_sha256 = _key_set_sha256(set(exposure_keys))
    return {
        "path": path.as_posix(),
        "sha256": _require_sha256(sha256, field_name=f"{path}.sha256"),
        "row_count": len(exposure_keys),
        "schema_id": exposure_manifest["schema_id"],
        "exposure_scan_source_sha256": exposure_manifest["exposure_scan_source_sha256"],
        "entries_sha256": exposure_manifest["entries_sha256"],
        "cache_audio_keys_sha256": cache_audio_keys_sha256,
    }


def _holdout_manifest_provenance(holdout_manifest: Mapping[str, Any]) -> dict[str, Any]:
    validate_exp004_manifest(holdout_manifest)
    entries = _manifest_entries(holdout_manifest)
    cache_audio_keys = [str(entry["cache_audio_key"]) for entry in entries]
    return {
        "path": None,
        "sha256": _json_sha256(holdout_manifest),
        "row_count": len(entries),
        "schema": holdout_manifest["schema"],
        "manifest_fingerprint_sha256": holdout_manifest["manifest_fingerprint_sha256"],
        "cache_audio_keys_sha256": _key_set_sha256(set(cache_audio_keys)),
        "ordered_cache_audio_keys_sha256": _ordered_key_list_sha256(cache_audio_keys),
        "selected_ordered_sha256": _selected_ordered_sha256(entries),
        "hash_encoding": "sha256_canonical_json_utf8",
    }


def _holdout_file_provenance(
    path: Path,
    *,
    holdout_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    validate_exp004_manifest(holdout_manifest)
    entries = _manifest_entries(holdout_manifest)
    cache_audio_keys = [str(entry["cache_audio_key"]) for entry in entries]
    return {
        "path": path.as_posix(),
        "sha256": _file_sha256(path),
        "row_count": len(entries),
        "schema": holdout_manifest["schema"],
        "manifest_fingerprint_sha256": holdout_manifest["manifest_fingerprint_sha256"],
        "cache_audio_keys_sha256": _key_set_sha256(set(cache_audio_keys)),
        "ordered_cache_audio_keys_sha256": _ordered_key_list_sha256(cache_audio_keys),
        "selected_ordered_sha256": _selected_ordered_sha256(entries),
    }


def _holdout_file_provenance_from_hash(
    path: Path,
    *,
    holdout_manifest: Mapping[str, Any],
    sha256: str,
) -> dict[str, Any]:
    validate_exp004_manifest(holdout_manifest)
    entries = _manifest_entries(holdout_manifest)
    cache_audio_keys = [str(entry["cache_audio_key"]) for entry in entries]
    return {
        "path": path.as_posix(),
        "sha256": _require_sha256(sha256, field_name=f"{path}.sha256"),
        "row_count": len(entries),
        "schema": holdout_manifest["schema"],
        "manifest_fingerprint_sha256": holdout_manifest["manifest_fingerprint_sha256"],
        "cache_audio_keys_sha256": _key_set_sha256(set(cache_audio_keys)),
        "ordered_cache_audio_keys_sha256": _ordered_key_list_sha256(cache_audio_keys),
        "selected_ordered_sha256": _selected_ordered_sha256(entries),
    }


def _in_memory_source_provenance(
    value: Any,
    *,
    row_count: int | None = None,
) -> dict[str, Any]:
    resolved_row_count = len(value) if row_count is None else row_count
    return {
        "path": None,
        "sha256": _json_sha256(value),
        "row_count": resolved_row_count,
        "hash_encoding": "sha256_canonical_json_utf8",
    }


def _file_provenance(path: Path, *, row_count: int) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "sha256": _file_sha256(path),
        "row_count": row_count,
    }


def _file_provenance_from_hash(
    path: Path,
    *,
    row_count: int,
    sha256: str,
) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "sha256": _require_sha256(sha256, field_name=f"{path}.sha256"),
        "row_count": row_count,
    }


def _validate_source_entry(
    value: Any,
    *,
    field_name: str,
    expected_row_count: int | None,
) -> None:
    if not isinstance(value, MappingABC):
        raise ValueError(f"{field_name} must be a provenance mapping")
    path = value.get("path")
    if path is not None and (not isinstance(path, str) or not path):
        raise ValueError(f"{field_name}.path must be null or a non-empty string")
    if not _is_sha256(value.get("sha256")):
        raise ValueError(f"{field_name}.sha256 must be a lowercase SHA-256 digest")
    row_count = value.get("row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
        raise ValueError(f"{field_name}.row_count must be a non-negative integer")
    if expected_row_count is not None and row_count != expected_row_count:
        raise ValueError(
            f"{field_name}.row_count must be {expected_row_count}, got {row_count}"
        )
    if expected_row_count is None and row_count < EXP004_MINIMUM_REQUIRED_EXCLUSION_KEYS:
        raise ValueError(
            f"{field_name}.row_count must cover at least "
            f"{EXP004_MINIMUM_REQUIRED_EXCLUSION_KEYS} rows"
        )
    if path is None and value.get("hash_encoding") != "sha256_canonical_json_utf8":
        raise ValueError(
            f"{field_name}.hash_encoding must describe the in-memory canonical JSON hash"
        )


def _validate_exposure_manifest_source_entry(
    value: Any,
    *,
    exposure_summary: Any,
) -> None:
    if not isinstance(exposure_summary, MappingABC):
        raise ValueError("manifest exposure summary must be a mapping")
    expected_count = exposure_summary.get("cache_audio_key_count")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise ValueError("manifest exposure cache_audio_key_count must be an integer")
    _validate_source_entry(
        value,
        field_name="source.exposure_manifest",
        expected_row_count=expected_count,
    )
    assert isinstance(value, MappingABC)
    checks = {
        "schema_id": exposure_summary.get("schema_id"),
        "exposure_scan_source_sha256": exposure_summary.get("exposure_scan_source_sha256"),
        "entries_sha256": exposure_summary.get("entries_sha256"),
        "cache_audio_keys_sha256": exposure_summary.get("cache_audio_keys_sha256"),
    }
    for field_name, expected in checks.items():
        if value.get(field_name) != expected:
            raise ValueError(
                f"source.exposure_manifest.{field_name} does not match exposure summary"
            )


def _validate_holdout_manifest_source_entry(
    value: Any,
    *,
    holdout_summary: Mapping[str, Any],
) -> None:
    _validate_source_entry(
        value,
        field_name="source.holdout_manifest",
        expected_row_count=EXP004_HOLDOUT_AUDIO_COUNT,
    )
    assert isinstance(value, MappingABC)
    checks = {
        "schema": holdout_summary.get("schema"),
        "manifest_fingerprint_sha256": holdout_summary.get("manifest_fingerprint_sha256"),
        "cache_audio_keys_sha256": holdout_summary.get("cache_audio_keys_sha256"),
        "ordered_cache_audio_keys_sha256": holdout_summary.get("ordered_cache_audio_keys_sha256"),
        "selected_ordered_sha256": holdout_summary.get("selected_ordered_sha256"),
    }
    for field_name, expected in checks.items():
        if value.get(field_name) != expected:
            raise ValueError(
                f"source.holdout_manifest.{field_name} does not match broad holdout provenance"
            )


def _validate_exposure_source(
    value: Any,
    *,
    field_name: str,
    expected_count: int,
) -> None:
    if not isinstance(value, MappingABC):
        raise ValueError(f"{field_name} must be a provenance mapping")
    if set(value) != _SOURCE_KEYS.union({"cache_audio_keys"}):
        missing = sorted(_SOURCE_KEYS.union({"cache_audio_keys"}).difference(value))
        extra = sorted(set(value).difference(_SOURCE_KEYS.union({"cache_audio_keys"})))
        raise ValueError(
            f"{field_name} fields are incomplete or unsupported: "
            f"missing={missing!r} extra={extra!r}"
        )
    _validate_source_entry(value, field_name=field_name, expected_row_count=expected_count)
    raw_keys = value.get("cache_audio_keys")
    if isinstance(raw_keys, (str, bytes)) or not isinstance(raw_keys, SequenceABC):
        raise ValueError(f"{field_name}.cache_audio_keys must be a sorted string list")
    keys = _validated_key_set(list(raw_keys), field_name=f"{field_name}.cache_audio_keys")
    if list(raw_keys) != sorted(keys):
        raise ValueError(f"{field_name}.cache_audio_keys must be sorted")
    if value.get("cache_audio_key_count") != expected_count:
        raise ValueError(f"{field_name}.cache_audio_key_count must be {expected_count}")
    if len(keys) != expected_count:
        raise ValueError(f"{field_name}.cache_audio_keys must contain {expected_count} keys")
    if value.get("cache_audio_keys_sha256") != _key_set_sha256(keys):
        raise ValueError(f"{field_name}.cache_audio_keys_sha256 does not replay")
    if value.get("hash_encoding") != "sha256_canonical_json_sorted_string_array_utf8":
        raise ValueError(f"{field_name}.hash_encoding is invalid")


def _file_source_provenance(path: Path, *, keys: set[str]) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "sha256": _file_sha256(path),
        "row_count": len(keys),
        "cache_audio_key_count": len(keys),
        "cache_audio_keys": sorted(keys),
        "cache_audio_keys_sha256": _key_set_sha256(keys),
        "hash_encoding": "sha256_canonical_json_sorted_string_array_utf8",
    }


def _normalize_exposure_entries(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, MappingABC):
            raise ValueError(f"exposure entries[{index}] must be a mapping")
        if set(entry) != _EXPOSURE_ENTRY_KEYS:
            missing = sorted(_EXPOSURE_ENTRY_KEYS.difference(entry))
            extra = sorted(set(entry).difference(_EXPOSURE_ENTRY_KEYS))
            raise ValueError(
                f"exposure entries[{index}] fields are incomplete or unsupported: "
                f"missing={missing!r} extra={extra!r}"
            )
        cache_audio_key = _required_nonempty_string(
            entry.get("cache_audio_key"),
            field_name=f"exposure entries[{index}].cache_audio_key",
        )
        if cache_audio_key in seen:
            raise ValueError(f"duplicate exposure cache_audio_key: {cache_audio_key!r}")
        seen.add(cache_audio_key)
        normalized.append(
            {
                "cache_audio_key": cache_audio_key,
                "exposure_reason": _required_nonempty_string(
                    entry.get("exposure_reason"),
                    field_name=f"exposure entries[{index}].exposure_reason",
                ),
                "exposure_source": _required_nonempty_string(
                    entry.get("exposure_source"),
                    field_name=f"exposure entries[{index}].exposure_source",
                ),
                "first_exposed_at_or_run_id": _required_nonempty_string(
                    entry.get("first_exposed_at_or_run_id"),
                    field_name=f"exposure entries[{index}].first_exposed_at_or_run_id",
                ),
            }
        )
    return sorted(normalized, key=lambda item: item["cache_audio_key"])


def _exposure_entries(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_entries = manifest.get("entries")
    if isinstance(raw_entries, (str, bytes)) or not isinstance(raw_entries, SequenceABC):
        raise ValueError("exposure manifest entries must be a list")
    entries: list[Mapping[str, Any]] = []
    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, MappingABC):
            raise ValueError(f"exposure manifest entries[{index}] must be a mapping")
        entries.append(entry)
    _normalize_exposure_entries(entries)
    return entries


def _load_cache_audio_keys(
    path: Path,
    *,
    label_rows: Sequence[Mapping[str, Any]] | None,
) -> set[str]:
    items = _load_selected_items(path)
    records = _label_records(label_rows) if label_rows is not None else []
    by_audio_group_key = {record.audio_group_key: record for record in records}
    keys: list[str] = []
    for index, item in enumerate(items):
        keys.append(
            _resolve_item_cache_audio_key(
                path,
                index=index,
                item=item,
                by_audio_group_key=by_audio_group_key,
                labels_supplied=label_rows is not None,
            )
        )
    return _validated_key_set(keys, field_name=f"cache audio keys from {path}")


def _load_selected_items(path: Path) -> list[Mapping[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        items: list[Mapping[str, Any]] = []
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(item, MappingABC):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            items.append(item)
        return items

    if isinstance(document, list):
        raw_items = document
    elif isinstance(document, MappingABC):
        selected = document.get("selected")
        if isinstance(selected, MappingABC):
            raw_items = [
                item
                for group in selected.values()
                if isinstance(group, SequenceABC) and not isinstance(group, (str, bytes))
                for item in group
            ]
        elif isinstance(selected, SequenceABC) and not isinstance(selected, (str, bytes)):
            raw_items = list(selected)
        elif "entries" in document and isinstance(document["entries"], SequenceABC):
            raw_items = list(document["entries"])
        else:
            raw_items = [document]
    else:
        raise ValueError(f"{path} must contain JSON objects")

    items: list[Mapping[str, Any]] = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, MappingABC):
            raise ValueError(f"{path}: selected item {index} must be a mapping")
        items.append(item)
    return items


def _resolve_item_cache_audio_key(
    path: Path,
    *,
    index: int,
    item: Mapping[str, Any],
    by_audio_group_key: Mapping[str, _LabelRecord],
    labels_supplied: bool,
) -> str:
    identity_values: list[tuple[str, str]] = []
    for field_name in ("cache_audio_key", "audio_key"):
        value = _identity_field_value(
            item,
            field_name,
            field_name=f"{path}: selected item {index}.{field_name}",
        )
        if value is not None:
            identity_values.append((field_name, value))

    source = item.get("source")
    if source is not None:
        if not isinstance(source, MappingABC):
            raise ValueError(f"{path}: selected item {index}.source must be a mapping")
        value = _identity_field_value(
            source,
            "cache_audio_key",
            field_name=f"{path}: selected item {index}.source.cache_audio_key",
        )
        if value is not None:
            identity_values.append(("source.cache_audio_key", value))

    audio_group_key = _identity_field_value(
        item,
        "audio_group_key",
        field_name=f"{path}: selected item {index}.audio_group_key",
    )
    if audio_group_key is not None and labels_supplied:
        record = by_audio_group_key.get(audio_group_key)
        if record is None:
            raise ValueError(
                f"{path}: audio_group_key is absent from labels: {audio_group_key!r}"
            )
        identity_values.append(("labels.audio_group_key", record.cache_audio_key))

    if not identity_values:
        if audio_group_key is not None and not labels_supplied:
            raise ValueError(
                f"{path}: selected item {index} uses audio_group_key but no labels_jsonl "
                "was provided for cache-key resolution"
            )
        raise ValueError(
            f"{path}: selected item {index} must provide cache_audio_key, "
            "audio_key, source.cache_audio_key, or audio_group_key"
        )

    distinct_values = {value for _name, value in identity_values}
    if len(distinct_values) != 1:
        labels = ", ".join(f"{name}={value!r}" for name, value in identity_values)
        raise ValueError(
            f"{path}: selected item {index} cache identity fields disagree: {labels}"
        )
    return identity_values[0][1]


def _identity_field_value(
    item: Mapping[str, Any],
    key: str,
    *,
    field_name: str,
) -> str | None:
    if key not in item:
        return None
    value = item.get(key)
    if value is None:
        return None
    return _required_nonempty_string(value, field_name=field_name)


def _validate_quotas(quotas: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(quotas, MappingABC):
        raise ValueError("quotas must be a mapping")
    if set(quotas) != set(EXP004_PRIORITY):
        raise ValueError(
            f"quotas must have exactly these keys: {list(EXP004_PRIORITY)!r}"
        )
    resolved: dict[str, int] = {}
    for quota in EXP004_PRIORITY:
        value = quotas[quota]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"quota {quota!r} must be a non-negative integer")
        resolved[quota] = value
    return resolved


def _manifest_entries(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    selected = manifest.get("selected")
    if isinstance(selected, (str, bytes)) or not isinstance(selected, SequenceABC):
        raise ValueError("manifest selected must be a list")
    entries: list[Mapping[str, Any]] = []
    for index, entry in enumerate(selected):
        if not isinstance(entry, MappingABC):
            raise ValueError(f"manifest selected[{index}] must be a mapping")
        entries.append(entry)
    return entries


def _validated_key_set(
    values: Sequence[str] | set[str] | frozenset[str],
    *,
    field_name: str,
) -> set[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (SequenceABC, set, frozenset)):
        raise ValueError(f"{field_name} must be a sequence or set of strings")
    keys: set[str] = set()
    for index, value in enumerate(values):
        key = _required_nonempty_string(value, field_name=f"{field_name}[{index}]")
        if key in keys:
            raise ValueError(f"{field_name} contains duplicate cache audio key {key!r}")
        keys.add(key)
    return keys


def _reject_metric_valued_fields(value: Any, *, path: str = "exposure_manifest") -> None:
    if isinstance(value, MappingABC):
        for raw_key, child in value.items():
            key = str(raw_key)
            lowered = key.lower()
            if lowered in _METRIC_FIELD_EXACT_DENYLIST or any(
                marker in lowered for marker in _METRIC_FIELD_SUBSTRINGS
            ):
                raise ValueError(f"metric-valued exposure field is forbidden: {path}.{key}")
            _reject_metric_valued_fields(child, path=f"{path}.{key}")
    elif isinstance(value, SequenceABC) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _reject_metric_valued_fields(child, path=f"{path}[{index}]")


def _reject_metric_or_oracle_manifest_fields(value: Any, *, path: str = "manifest") -> None:
    if isinstance(value, MappingABC):
        for raw_key, child in value.items():
            key = str(raw_key)
            lowered = key.lower()
            if lowered in _METRIC_FIELD_EXACT_DENYLIST or any(
                marker in lowered for marker in _METRIC_FIELD_SUBSTRINGS
            ):
                raise ValueError(f"metric/oracle-valued manifest field is forbidden: {path}.{key}")
            _reject_metric_or_oracle_manifest_fields(child, path=f"{path}.{key}")
    elif isinstance(value, SequenceABC) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _reject_metric_or_oracle_manifest_fields(child, path=f"{path}[{index}]")


def _require_exact_mapping_fields(
    value: Mapping[str, Any],
    expected_fields: set[str],
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


def _with_manifest_fingerprint(body: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(body)
    manifest["manifest_fingerprint_sha256"] = _json_sha256(body)
    return manifest


def _key_set_sha256(keys: set[str]) -> str:
    return _json_sha256(sorted(keys))


def _ordered_key_list_sha256(keys: Sequence[str]) -> str:
    return _json_sha256(list(keys))


def _selected_ordered_sha256(entries: Sequence[Mapping[str, Any]]) -> str:
    return _json_sha256([dict(entry) for entry in entries])


def _selection_hash(seed: str, cache_audio_key: str) -> str:
    return hashlib.sha256(f"{seed}\0{cache_audio_key}".encode("utf-8")).hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    before = path.stat()
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


def _required_nonempty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _optional_nonempty_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, *, field_name: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    assert isinstance(value, str)
    return value


def _require_distinct_paths(paths: Sequence[Path | None]) -> None:
    resolved: set[Path] = set()
    for path in paths:
        if path is None:
            continue
        candidate = path.expanduser().resolve(strict=False)
        if candidate in resolved:
            raise ValueError(f"input and output paths must be distinct; duplicate {path}")
        resolved.add(candidate)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    data = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_bytes_immutable_atomic(path, data)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [json.dumps(row, allow_nan=False, sort_keys=True) for row in rows]
    data = ("".join(f"{line}\n" for line in lines)).encode("utf-8")
    _write_bytes_immutable_atomic(path, data)


def _write_bytes_immutable_atomic(path: Path, data: bytes) -> None:
    """Write immutable artifacts, allowing only exact byte-identical replay.

    Existing outputs are accepted only when their bytes already match ``data``.
    Fields such as ``generated_at_utc`` participate in that byte comparison, so a
    rerun with a different timestamp is not treated as identical.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == data:
            return
        raise ValueError(f"immutable output already exists with different bytes: {path}")

    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp_path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp_path, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(
                    f"immutable output already exists with different bytes: {path}"
                )
        else:
            _fsync_directory(path.parent)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
            _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            os.fsync(fd)
        except OSError:
            return
    finally:
        os.close(fd)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build frozen Timing v3 Experiment 004 exposure and holdout splits."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    exposure = subparsers.add_parser(
        "exposure",
        help="Build the oracle-exposure exclusion manifest.",
    )
    exposure.add_argument("--pilot-manifest", type=Path, required=True)
    exposure.add_argument("--protocol-manifest", type=Path, required=True)
    exposure.add_argument("--exp003-holdout-manifest", type=Path, required=True)
    exposure.add_argument("--labels-jsonl", type=Path, default=None)
    exposure.add_argument("--generated-from-commit", required=True)
    exposure.add_argument("--generated-at-utc", default=None)
    exposure.add_argument("--manifest-output", type=Path, required=True)

    holdout = subparsers.add_parser(
        "holdout100",
        help="Build the frozen Exp004 100-audio holdout identity manifest.",
    )
    holdout.add_argument("--labels-jsonl", type=Path, required=True)
    holdout.add_argument("--exposure-manifest", type=Path, required=True)
    holdout.add_argument("--manifest-output", type=Path, required=True)
    holdout.add_argument("--label-rows-output", type=Path, default=None)

    broad = subparsers.add_parser(
        "broad500",
        help="Build the Exp004 broad500 identity manifest from frozen holdout100.",
    )
    broad.add_argument("--labels-jsonl", type=Path, required=True)
    broad.add_argument("--exposure-manifest", type=Path, required=True)
    broad.add_argument("--holdout-manifest", type=Path, required=True)
    broad.add_argument("--manifest-output", type=Path, required=True)
    broad.add_argument("--label-rows-output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.command == "exposure":
        result = build_exp004_exposure_manifest(
            pilot_manifest_path=args.pilot_manifest,
            protocol_manifest_path=args.protocol_manifest,
            exp003_holdout_manifest_path=args.exp003_holdout_manifest,
            labels_jsonl_path=args.labels_jsonl,
            generated_from_commit=args.generated_from_commit,
            generated_at_utc=args.generated_at_utc,
            manifest_output_path=args.manifest_output,
        )
    elif args.command == "holdout100":
        result = build_exp004_holdout100(
            labels_jsonl_path=args.labels_jsonl,
            exposure_manifest_path=args.exposure_manifest,
            manifest_output_path=args.manifest_output,
            label_rows_output_path=args.label_rows_output,
        )
    elif args.command == "broad500":
        result = build_exp004_broad500(
            labels_jsonl_path=args.labels_jsonl,
            exposure_manifest_path=args.exposure_manifest,
            holdout_manifest_path=args.holdout_manifest,
            manifest_output_path=args.manifest_output,
            label_rows_output_path=args.label_rows_output,
        )
    else:  # pragma: no cover - argparse owns the finite command set.
        raise AssertionError(f"unhandled command {args.command!r}")
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
