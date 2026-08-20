from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pulsefield_model.timing.evaluation.labels import (
    LABEL_AMBIGUOUS,
    LABEL_DENSE,
    LABEL_JUMP_CANDIDATE,
    LABEL_RAMP_CANDIDATE,
    LABEL_STABLE,
    TIMING_V3_LABEL_AUDIO_ROW_SCHEMA,
)


TIMING_V3_EXP003_HOLDOUT_MANIFEST_SCHEMA = (
    "pulsefield_model.timing_v3_exp003_holdout100_manifest_v2"
)
TIMING_V3_EXP003_BROAD_MANIFEST_SCHEMA = (
    "pulsefield_model.timing_v3_exp003_broad500_manifest_v2"
)
TIMING_V3_EXP003_SPLIT_ANNOTATION_SCHEMA = (
    "pulsefield_model.timing_v3_exp003_split_annotation_v2"
)

EXP003_HOLDOUT_SEED = "timing-v3-exp003-holdout100-v2"
EXP003_BROAD_SEED = "timing-v3-exp003-broad500-v2"
EXP003_PRIORITY = ("ramp_audit", "anomaly", "long", "dense", "jump", "stable")
# Protocol v2 excludes the three exposed v1 ramp rows. Only four unexposed
# ramp candidates remain, so the retired ramp slot is preregistered for long.
EXP003_HOLDOUT_QUOTAS = {
    "ramp_audit": 4,
    "anomaly": 10,
    "long": 11,
    "dense": 10,
    "jump": 25,
    "stable": 40,
}
EXP003_PILOT_AUDIO_COUNT = 80
EXP003_PROTOCOL_EXCLUSION_AUDIO_COUNT = 3
EXP003_HOLDOUT_AUDIO_COUNT = 100
EXP003_BROAD_ADDED_AUDIO_COUNT = 400
EXP003_BROAD_AUDIO_COUNT = 500


@dataclass(frozen=True)
class _LabelRecord:
    row: Mapping[str, Any]
    audio_group_key: str
    cache_audio_key: str
    label_stratum: str
    long_track: bool


def load_label_rows_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a label inventory and reject duplicate audio identities."""

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


def select_exp003_holdout100(
    label_rows: Sequence[Mapping[str, Any]],
    *,
    pilot_excluded_cache_audio_keys: Sequence[str] | set[str] | frozenset[str],
    protocol_excluded_cache_audio_keys: Sequence[str] | set[str] | frozenset[str],
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Select the Experiment 003 audio-disjoint holdout.

    Rows are assigned to exactly one quota in the preregistered priority order,
    then ranked within that quota by ``sha256(seed + NUL + cache_audio_key)``.
    An unavailable quota is an error; no cross-quota backfill is permitted.
    """

    records = _label_records(label_rows)
    pilot_exclusions = _validated_key_set(
        pilot_excluded_cache_audio_keys,
        field_name="pilot_excluded_cache_audio_keys",
    )
    protocol_exclusions = _validated_key_set(
        protocol_excluded_cache_audio_keys,
        field_name="protocol_excluded_cache_audio_keys",
    )
    _validate_frozen_exclusion_sets(
        pilot_exclusions=pilot_exclusions,
        protocol_exclusions=protocol_exclusions,
    )
    all_exclusions = pilot_exclusions.union(protocol_exclusions)
    record_cache_keys = {record.cache_audio_key for record in records}
    missing_exclusions = sorted(all_exclusions.difference(record_cache_keys))
    if missing_exclusions:
        raise ValueError(
            "frozen pilot/protocol exclusions are absent from label_rows: "
            f"{missing_exclusions[:3]!r}"
        )
    resolved_source = _resolved_holdout_source(
        source,
        records=records,
        pilot_exclusions=pilot_exclusions,
        protocol_exclusions=protocol_exclusions,
    )
    resolved_quotas = _validate_quotas(EXP003_HOLDOUT_QUOTAS)
    candidates: dict[str, list[_LabelRecord]] = {quota: [] for quota in EXP003_PRIORITY}
    for record in records:
        if record.cache_audio_key in all_exclusions:
            continue
        assignment = _quota_assignment(record)
        candidates[assignment].append(record)

    selected: list[dict[str, Any]] = []
    available_counts: dict[str, int] = {}
    for quota in EXP003_PRIORITY:
        ordered = sorted(
            candidates[quota],
            key=lambda record: (
                _selection_hash(EXP003_HOLDOUT_SEED, record.cache_audio_key),
                record.cache_audio_key,
                record.audio_group_key,
            ),
        )
        available_counts[quota] = len(ordered)
        requested = resolved_quotas[quota]
        if len(ordered) < requested:
            raise ValueError(
                f"Experiment 003 holdout quota {quota!r} is underfilled: "
                f"required {requested}, available {len(ordered)} after exclusions; "
                "cross-quota backfill is forbidden"
            )
        for selection_rank, record in enumerate(ordered[:requested], start=1):
            selected.append(
                _selection_entry(
                    record,
                    stage="holdout100",
                    quota_assignment=quota,
                    selection_rank=selection_rank,
                    selection_hash_sha256=_selection_hash(
                        EXP003_HOLDOUT_SEED,
                        record.cache_audio_key,
                    ),
                )
            )

    body: dict[str, Any] = {
        "schema": TIMING_V3_EXP003_HOLDOUT_MANIFEST_SCHEMA,
        "experiment": "timing_v3_experiment_003",
        "stage": "holdout100",
        "seed": EXP003_HOLDOUT_SEED,
        "selection": {
            "method": "exclusive_priority_then_sha256_seed_nul_cache_audio_key",
            "priority_order": list(EXP003_PRIORITY),
            "quotas": dict(resolved_quotas),
            "target_audio_count": sum(resolved_quotas.values()),
            "available_counts_after_exclusion": available_counts,
        },
        "source": resolved_source,
        "exclusion": _key_set_provenance(pilot_exclusions),
        "protocol_exclusion": _key_set_provenance(protocol_exclusions),
        "selected_audio_count": len(selected),
        "selected_counts": {
            quota: sum(1 for entry in selected if entry["quota_assignment"] == quota)
            for quota in EXP003_PRIORITY
        },
        "selected": selected,
    }
    manifest = _with_manifest_fingerprint(body)
    validate_exp003_manifest(manifest)
    return manifest


def select_exp003_broad500(
    label_rows: Sequence[Mapping[str, Any]],
    *,
    pilot_excluded_cache_audio_keys: Sequence[str] | set[str] | frozenset[str],
    protocol_excluded_cache_audio_keys: Sequence[str] | set[str] | frozenset[str],
    holdout_manifest: Mapping[str, Any],
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct the 500 stage as the frozen holdout plus 400 new rows."""

    records = _label_records(label_rows)
    pilot_exclusions = _validated_key_set(
        pilot_excluded_cache_audio_keys,
        field_name="pilot_excluded_cache_audio_keys",
    )
    protocol_exclusions = _validated_key_set(
        protocol_excluded_cache_audio_keys,
        field_name="protocol_excluded_cache_audio_keys",
    )
    _validate_frozen_exclusion_sets(
        pilot_exclusions=pilot_exclusions,
        protocol_exclusions=protocol_exclusions,
    )
    all_exclusions = pilot_exclusions.union(protocol_exclusions)
    record_cache_keys = {record.cache_audio_key for record in records}
    missing_exclusions = sorted(all_exclusions.difference(record_cache_keys))
    if missing_exclusions:
        raise ValueError(
            "frozen pilot/protocol exclusions are absent from label_rows: "
            f"{missing_exclusions[:3]!r}"
        )
    validate_exp003_manifest(holdout_manifest)
    if holdout_manifest.get("schema") != TIMING_V3_EXP003_HOLDOUT_MANIFEST_SCHEMA:
        raise ValueError("holdout_manifest must use the Experiment 003 holdout100 schema")

    holdout_entries = _manifest_entries(holdout_manifest)
    holdout_exclusion = holdout_manifest["exclusion"]
    if set(holdout_exclusion["cache_audio_keys"]) != pilot_exclusions:
        raise ValueError("broad pilot exclusions do not match the frozen holdout exclusions")
    holdout_protocol_exclusion = holdout_manifest["protocol_exclusion"]
    if set(holdout_protocol_exclusion["cache_audio_keys"]) != protocol_exclusions:
        raise ValueError(
            "broad protocol exclusions do not match the frozen holdout exclusions"
        )
    holdout_keys = {str(entry["cache_audio_key"]) for entry in holdout_entries}
    overlap = sorted(all_exclusions.intersection(holdout_keys))
    if overlap:
        raise ValueError(
            "holdout_manifest overlaps frozen pilot/protocol exclusions by cache audio key: "
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
        record = by_cache_key[str(entry["cache_audio_key"])]
        _validate_entry_against_record(entry, record)

    resolved_source = _resolved_broad_source(
        source,
        records=records,
        pilot_exclusions=pilot_exclusions,
        protocol_exclusions=protocol_exclusions,
        holdout_manifest=holdout_manifest,
    )
    holdout_source = holdout_manifest["source"]
    for source_name in ("labels", "pilot_exclusion", "protocol_exclusion"):
        if resolved_source[source_name] != holdout_source[source_name]:
            raise ValueError(
                f"broad {source_name} provenance does not exactly match the frozen holdout"
            )

    replayed_holdout = select_exp003_holdout100(
        label_rows,
        pilot_excluded_cache_audio_keys=pilot_exclusions,
        protocol_excluded_cache_audio_keys=protocol_exclusions,
        source=holdout_source,
    )
    if holdout_manifest != replayed_holdout:
        raise ValueError(
            "holdout_manifest is not the deterministic protocol-v2 holdout replay"
        )

    added_candidates = [
        record
        for record in records
        if record.cache_audio_key not in all_exclusions
        and record.cache_audio_key not in holdout_keys
    ]
    ordered_added = sorted(
        added_candidates,
        key=lambda record: (
            _selection_hash(EXP003_BROAD_SEED, record.cache_audio_key),
            record.cache_audio_key,
            record.audio_group_key,
        ),
    )
    if len(ordered_added) < EXP003_BROAD_ADDED_AUDIO_COUNT:
        raise ValueError(
            "Experiment 003 broad stage is underfilled: "
            f"required {EXP003_BROAD_ADDED_AUDIO_COUNT} new audio groups, "
            f"available {len(ordered_added)}"
        )

    selected = [dict(entry) for entry in holdout_entries]
    for selection_rank, record in enumerate(
        ordered_added[:EXP003_BROAD_ADDED_AUDIO_COUNT],
        start=1,
    ):
        selected.append(
            _selection_entry(
                record,
                stage="broad500_added",
                quota_assignment=None,
                selection_rank=selection_rank,
                selection_hash_sha256=_selection_hash(
                    EXP003_BROAD_SEED,
                    record.cache_audio_key,
                ),
            )
        )

    body: dict[str, Any] = {
        "schema": TIMING_V3_EXP003_BROAD_MANIFEST_SCHEMA,
        "experiment": "timing_v3_experiment_003",
        "stage": "broad500",
        "seed": EXP003_BROAD_SEED,
        "selection": {
            "method": "holdout100_then_lowest_sha256_seed_nul_cache_audio_key",
            "holdout_audio_count": len(holdout_entries),
            "added_audio_count": EXP003_BROAD_ADDED_AUDIO_COUNT,
            "target_audio_count": len(holdout_entries) + EXP003_BROAD_ADDED_AUDIO_COUNT,
            "added_candidate_count_after_exclusions": len(ordered_added),
        },
        "source": resolved_source,
        "exclusion": _key_set_provenance(pilot_exclusions),
        "protocol_exclusion": _key_set_provenance(protocol_exclusions),
        "holdout": {
            "schema": holdout_manifest["schema"],
            "manifest_fingerprint_sha256": holdout_manifest["manifest_fingerprint_sha256"],
            "cache_audio_keys_sha256": _key_set_sha256(holdout_keys),
        },
        "selected_audio_count": len(selected),
        "selected_counts": {
            "holdout100": len(holdout_entries),
            "broad500_added": EXP003_BROAD_ADDED_AUDIO_COUNT,
        },
        "selected": selected,
    }
    manifest = _with_manifest_fingerprint(body)
    validate_exp003_manifest(manifest)
    return manifest


def validate_exp003_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed on malformed, duplicated, overlapping, or edited manifests."""

    if not isinstance(manifest, MappingABC):
        raise ValueError("manifest must be a mapping")
    schema = manifest.get("schema")
    if schema not in {
        TIMING_V3_EXP003_HOLDOUT_MANIFEST_SCHEMA,
        TIMING_V3_EXP003_BROAD_MANIFEST_SCHEMA,
    }:
        raise ValueError(f"unsupported Experiment 003 manifest schema: {schema!r}")
    if manifest.get("experiment") != "timing_v3_experiment_003":
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

    audio_group_keys: set[str] = set()
    cache_audio_keys: set[str] = set()
    stage_ranks: dict[tuple[str, str | None], set[int]] = {}
    for index, entry in enumerate(entries):
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
        quota = entry.get("quota_assignment")
        if quota is not None and quota not in EXP003_PRIORITY:
            raise ValueError(f"selected[{index}].quota_assignment is invalid: {quota!r}")
        rank = entry.get("selection_rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError(f"selected[{index}].selection_rank must be a positive integer")
        rank_key = (stage, str(quota) if quota is not None else None)
        ranks = stage_ranks.setdefault(rank_key, set())
        if rank in ranks:
            raise ValueError(
                f"duplicate selection_rank={rank} for stage/quota {rank_key!r}"
            )
        ranks.add(rank)

        selection_hash = entry.get("selection_hash_sha256")
        expected_selection_hash = _selection_hash(
            EXP003_HOLDOUT_SEED if stage == "holdout100" else EXP003_BROAD_SEED,
            cache_audio_key,
        )
        if selection_hash != expected_selection_hash:
            raise ValueError(
                f"selected[{index}] selection hash does not replay for cache audio key"
            )

    pilot_exclusion_keys = _validate_exclusion_provenance(
        manifest.get("exclusion"),
        field_name="exclusion",
        expected_count=EXP003_PILOT_AUDIO_COUNT,
    )
    protocol_exclusion_keys = _validate_exclusion_provenance(
        manifest.get("protocol_exclusion"),
        field_name="protocol_exclusion",
        expected_count=EXP003_PROTOCOL_EXCLUSION_AUDIO_COUNT,
    )
    exclusion_overlap = sorted(pilot_exclusion_keys.intersection(protocol_exclusion_keys))
    if exclusion_overlap:
        raise ValueError(
            "manifest pilot and protocol exclusions overlap: "
            f"{exclusion_overlap[:3]!r}"
        )
    all_exclusion_keys = pilot_exclusion_keys.union(protocol_exclusion_keys)
    selected_exclusion_overlap = sorted(cache_audio_keys.intersection(all_exclusion_keys))
    if selected_exclusion_overlap:
        raise ValueError(
            "manifest selected rows overlap frozen pilot/protocol exclusions: "
            f"{selected_exclusion_overlap[:3]!r}"
        )

    source = manifest.get("source")
    if not isinstance(source, MappingABC):
        raise ValueError("manifest source provenance must be a mapping")
    _validate_source_entry(
        source.get("labels"),
        field_name="source.labels",
        expected_row_count=None,
    )
    _validate_source_entry(
        source.get("pilot_exclusion"),
        field_name="source.pilot_exclusion",
        expected_row_count=EXP003_PILOT_AUDIO_COUNT,
    )
    _validate_source_entry(
        source.get("protocol_exclusion"),
        field_name="source.protocol_exclusion",
        expected_row_count=EXP003_PROTOCOL_EXCLUSION_AUDIO_COUNT,
    )
    _validate_in_memory_exclusion_source_hash(
        source.get("pilot_exclusion"),
        exclusion_keys=pilot_exclusion_keys,
        field_name="source.pilot_exclusion",
    )
    _validate_in_memory_exclusion_source_hash(
        source.get("protocol_exclusion"),
        exclusion_keys=protocol_exclusion_keys,
        field_name="source.protocol_exclusion",
    )

    if schema == TIMING_V3_EXP003_HOLDOUT_MANIFEST_SCHEMA:
        _validate_holdout_manifest_details(manifest, entries)
    else:
        _validate_source_entry(
            source.get("holdout_manifest"),
            field_name="source.holdout_manifest",
            expected_row_count=EXP003_HOLDOUT_AUDIO_COUNT,
        )
        _validate_broad_manifest_details(manifest, entries)


def build_exp003_holdout100(
    *,
    labels_jsonl_path: str | Path,
    pilot_exclusion_path: str | Path,
    protocol_exclusion_path: str | Path,
    manifest_output_path: str | Path,
    label_rows_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build and atomically write the frozen 100-audio holdout artifacts."""

    labels_path = Path(labels_jsonl_path)
    pilot_path = Path(pilot_exclusion_path)
    protocol_path = Path(protocol_exclusion_path)
    manifest_path = Path(manifest_output_path)
    rows_path = Path(label_rows_output_path) if label_rows_output_path is not None else None
    _require_distinct_paths(
        [labels_path, pilot_path, protocol_path, manifest_path, rows_path]
    )

    label_rows = load_label_rows_jsonl(labels_path)
    records = _label_records(label_rows)
    exclusion_keys = load_exclusion_cache_audio_keys(
        pilot_path,
        label_rows=label_rows,
        expected_count=EXP003_PILOT_AUDIO_COUNT,
    )
    protocol_exclusion_keys = load_exclusion_cache_audio_keys(
        protocol_path,
        label_rows=label_rows,
        expected_count=EXP003_PROTOCOL_EXCLUSION_AUDIO_COUNT,
    )
    manifest = select_exp003_holdout100(
        label_rows,
        pilot_excluded_cache_audio_keys=exclusion_keys,
        protocol_excluded_cache_audio_keys=protocol_exclusion_keys,
        source={
            "labels": _file_provenance(labels_path, row_count=len(records)),
            "pilot_exclusion": _file_provenance(
                pilot_path,
                row_count=len(exclusion_keys),
            ),
            "protocol_exclusion": _file_provenance(
                protocol_path,
                row_count=len(protocol_exclusion_keys),
            ),
        },
    )
    _write_json_atomic(manifest_path, manifest)
    if rows_path is not None:
        _write_jsonl_atomic(rows_path, materialize_label_subset(label_rows, manifest))
    return manifest


def build_exp003_broad500(
    *,
    labels_jsonl_path: str | Path,
    pilot_exclusion_path: str | Path,
    protocol_exclusion_path: str | Path,
    holdout_manifest_path: str | Path,
    manifest_output_path: str | Path,
    label_rows_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build and atomically write the frozen 500-audio stage artifacts."""

    labels_path = Path(labels_jsonl_path)
    pilot_path = Path(pilot_exclusion_path)
    protocol_path = Path(protocol_exclusion_path)
    holdout_path = Path(holdout_manifest_path)
    manifest_path = Path(manifest_output_path)
    rows_path = Path(label_rows_output_path) if label_rows_output_path is not None else None
    _require_distinct_paths(
        [
            labels_path,
            pilot_path,
            protocol_path,
            holdout_path,
            manifest_path,
            rows_path,
        ]
    )

    label_rows = load_label_rows_jsonl(labels_path)
    exclusion_keys = load_exclusion_cache_audio_keys(
        pilot_path,
        label_rows=label_rows,
        expected_count=EXP003_PILOT_AUDIO_COUNT,
    )
    protocol_exclusion_keys = load_exclusion_cache_audio_keys(
        protocol_path,
        label_rows=label_rows,
        expected_count=EXP003_PROTOCOL_EXCLUSION_AUDIO_COUNT,
    )
    holdout_manifest = load_exp003_manifest(holdout_path)
    manifest = select_exp003_broad500(
        label_rows,
        pilot_excluded_cache_audio_keys=exclusion_keys,
        protocol_excluded_cache_audio_keys=protocol_exclusion_keys,
        holdout_manifest=holdout_manifest,
        source={
            "labels": _file_provenance(labels_path, row_count=len(label_rows)),
            "pilot_exclusion": _file_provenance(
                pilot_path,
                row_count=len(exclusion_keys),
            ),
            "protocol_exclusion": _file_provenance(
                protocol_path,
                row_count=len(protocol_exclusion_keys),
            ),
            "holdout_manifest": _file_provenance(
                holdout_path,
                row_count=len(_manifest_entries(holdout_manifest)),
            ),
        },
    )
    _write_json_atomic(manifest_path, manifest)
    if rows_path is not None:
        _write_jsonl_atomic(rows_path, materialize_label_subset(label_rows, manifest))
    return manifest


def load_exp003_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    validate_exp003_manifest(payload)
    return payload


def load_exclusion_cache_audio_keys(
    path: str | Path,
    *,
    label_rows: Sequence[Mapping[str, Any]],
    expected_count: int | None = None,
) -> frozenset[str]:
    """Load exclusions from full-row JSONL or a selected-row manifest."""

    path = Path(path)
    items = _load_selected_items(path)
    records = _label_records(label_rows)
    by_audio_group_key = {record.audio_group_key: record for record in records}
    by_cache_audio_key = {record.cache_audio_key: record for record in records}
    resolved: list[str] = []
    seen_audio_group_keys: set[str] = set()
    for index, item in enumerate(items):
        cache_key = _optional_cache_audio_key(item)
        audio_group_key = _optional_nonempty_string(item.get("audio_group_key"))
        if cache_key is None and audio_group_key is None:
            raise ValueError(
                f"{path}: selected item {index} must provide cache_audio_key or audio_group_key"
            )
        if audio_group_key is not None:
            if audio_group_key in seen_audio_group_keys:
                raise ValueError(f"{path}: duplicate excluded audio_group_key {audio_group_key!r}")
            seen_audio_group_keys.add(audio_group_key)
            record = by_audio_group_key.get(audio_group_key)
            if record is None:
                raise ValueError(
                    f"{path}: excluded audio_group_key is absent from label inventory: "
                    f"{audio_group_key!r}"
                )
            if cache_key is None:
                cache_key = record.cache_audio_key
            elif cache_key != record.cache_audio_key:
                raise ValueError(
                    f"{path}: excluded cache_audio_key does not match label inventory for "
                    f"{audio_group_key!r}"
                )
        assert cache_key is not None
        if cache_key not in by_cache_audio_key:
            raise ValueError(
                f"{path}: excluded cache_audio_key is absent from label inventory: "
                f"{cache_key!r}"
            )
        resolved.append(cache_key)

    keys = _validated_key_set(resolved, field_name=f"exclusions from {path}")
    if expected_count is not None and len(keys) != expected_count:
        raise ValueError(
            f"{path}: expected {expected_count} unique excluded cache audio keys, got {len(keys)}"
        )
    return frozenset(keys)


def materialize_label_subset(
    label_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return full label rows in manifest order with deterministic split metadata."""

    records = _label_records(label_rows)
    validate_exp003_manifest(manifest)
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
        # Baseline loading prefers this explicit key over re-statting the audio.
        payload["cache_audio_key"] = cache_audio_key
        payload["experiment_split"] = {
            "schema": TIMING_V3_EXP003_SPLIT_ANNOTATION_SCHEMA,
            "experiment": "timing_v3_experiment_003",
            "manifest_schema": manifest["schema"],
            "manifest_fingerprint_sha256": manifest["manifest_fingerprint_sha256"],
            "manifest_stage": manifest["stage"],
            "selection_stage": entry["stage"],
            "quota_assignment": entry.get("quota_assignment"),
            "selection_rank": entry["selection_rank"],
            "selection_hash_sha256": entry["selection_hash_sha256"],
        }
        rows.append(payload)
    return rows


def materialize_baseline_subset(
    *,
    baseline_jsonl_path: str | Path,
    manifest_path: str | Path,
    output_jsonl_path: str | Path,
) -> dict[str, Any]:
    """Copy baseline rows byte-for-byte into manifest order by cache audio key."""

    baseline_path = Path(baseline_jsonl_path)
    manifest_file = Path(manifest_path)
    output_path = Path(output_jsonl_path)
    _require_distinct_paths([baseline_path, manifest_file, output_path])
    manifest = load_exp003_manifest(manifest_file)

    raw_by_cache_key: dict[str, str] = {}
    with baseline_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{baseline_path}:{line_number} is not valid JSON") from exc
            if not isinstance(payload, MappingABC):
                raise ValueError(f"{baseline_path}:{line_number} must be a JSON object")
            cache_audio_key = _required_nonempty_string(
                payload.get("audio_key"),
                field_name=f"{baseline_path}:{line_number}.audio_key",
            )
            if cache_audio_key in raw_by_cache_key:
                raise ValueError(
                    f"{baseline_path}: duplicate baseline audio_key {cache_audio_key!r}"
                )
            raw_by_cache_key[cache_audio_key] = line

    ordered_lines: list[str] = []
    for entry in _manifest_entries(manifest):
        cache_audio_key = str(entry["cache_audio_key"])
        raw_line = raw_by_cache_key.get(cache_audio_key)
        if raw_line is None:
            raise ValueError(
                f"baseline is missing manifest cache audio key {cache_audio_key!r}"
            )
        ordered_lines.append(raw_line)
    _write_raw_jsonl_atomic(output_path, ordered_lines)
    return {
        "schema": "pulsefield_model.timing_v3_exp003_baseline_subset_report_v2",
        "baseline": _file_provenance(baseline_path, row_count=len(raw_by_cache_key)),
        "manifest": {
            **_file_provenance(manifest_file, row_count=len(ordered_lines)),
            "schema": manifest["schema"],
            "manifest_fingerprint_sha256": manifest["manifest_fingerprint_sha256"],
        },
        "output": _file_provenance(output_path, row_count=len(ordered_lines)),
    }


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
    # This explicit order is the preregistered exclusivity rule.
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


def _validate_quotas(quotas: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(quotas, MappingABC):
        raise ValueError("quotas must be a mapping")
    if set(quotas) != set(EXP003_PRIORITY):
        raise ValueError(
            f"quotas must have exactly these keys: {list(EXP003_PRIORITY)!r}"
        )
    resolved: dict[str, int] = {}
    for quota in EXP003_PRIORITY:
        value = quotas[quota]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"quota {quota!r} must be a non-negative integer")
        resolved[quota] = value
    return resolved


def _selection_entry(
    record: _LabelRecord,
    *,
    stage: str,
    quota_assignment: str | None,
    selection_rank: int,
    selection_hash_sha256: str,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "quota_assignment": quota_assignment,
        "selection_rank": selection_rank,
        "selection_hash_sha256": selection_hash_sha256,
        "cache_audio_key": record.cache_audio_key,
        "audio_group_key": record.audio_group_key,
        "resolved_audio_path": record.row.get("resolved_audio_path"),
        "label_stratum": record.label_stratum,
        "source_long_track": record.long_track,
    }


def _validate_entry_against_record(
    entry: Mapping[str, Any],
    record: _LabelRecord,
) -> None:
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
    if entry.get("quota_assignment") != expected_quota:
        raise ValueError(
            f"manifest quota assignment is stale for cache key {record.cache_audio_key!r}"
        )
    seed = EXP003_HOLDOUT_SEED if stage == "holdout100" else EXP003_BROAD_SEED
    if entry.get("selection_hash_sha256") != _selection_hash(seed, record.cache_audio_key):
        raise ValueError(
            f"manifest selection hash is stale for cache key {record.cache_audio_key!r}"
        )


def _resolved_holdout_source(
    source: Mapping[str, Any] | None,
    *,
    records: Sequence[_LabelRecord],
    pilot_exclusions: set[str],
    protocol_exclusions: set[str],
) -> dict[str, Any]:
    resolved = (
        {
            "labels": _in_memory_source_provenance(
                [record.row for record in sorted(records, key=lambda item: item.cache_audio_key)]
            ),
            "pilot_exclusion": _in_memory_source_provenance(sorted(pilot_exclusions)),
            "protocol_exclusion": _in_memory_source_provenance(
                sorted(protocol_exclusions)
            ),
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
        resolved.get("pilot_exclusion"),
        field_name="source.pilot_exclusion",
        expected_row_count=len(pilot_exclusions),
    )
    _validate_source_entry(
        resolved.get("protocol_exclusion"),
        field_name="source.protocol_exclusion",
        expected_row_count=len(protocol_exclusions),
    )
    return resolved


def _resolved_broad_source(
    source: Mapping[str, Any] | None,
    *,
    records: Sequence[_LabelRecord],
    pilot_exclusions: set[str],
    protocol_exclusions: set[str],
    holdout_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = (
        {
            "labels": _in_memory_source_provenance(
                [record.row for record in sorted(records, key=lambda item: item.cache_audio_key)]
            ),
            "pilot_exclusion": _in_memory_source_provenance(sorted(pilot_exclusions)),
            "protocol_exclusion": _in_memory_source_provenance(
                sorted(protocol_exclusions)
            ),
            "holdout_manifest": _in_memory_source_provenance(
                holdout_manifest,
                row_count=EXP003_HOLDOUT_AUDIO_COUNT,
            ),
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
        resolved.get("pilot_exclusion"),
        field_name="source.pilot_exclusion",
        expected_row_count=len(pilot_exclusions),
    )
    _validate_source_entry(
        resolved.get("protocol_exclusion"),
        field_name="source.protocol_exclusion",
        expected_row_count=len(protocol_exclusions),
    )
    _validate_source_entry(
        resolved.get("holdout_manifest"),
        field_name="source.holdout_manifest",
        expected_row_count=EXP003_HOLDOUT_AUDIO_COUNT,
    )
    return resolved


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
    minimum_source_rows = (
        EXP003_PILOT_AUDIO_COUNT
        + EXP003_PROTOCOL_EXCLUSION_AUDIO_COUNT
        + EXP003_HOLDOUT_AUDIO_COUNT
    )
    if expected_row_count is None and row_count < minimum_source_rows:
        raise ValueError(
            f"{field_name}.row_count must cover at least {minimum_source_rows} rows"
        )
    if path is None and value.get("hash_encoding") != "sha256_canonical_json_utf8":
        raise ValueError(
            f"{field_name}.hash_encoding must describe the in-memory canonical JSON hash"
        )


def _validate_in_memory_exclusion_source_hash(
    value: Any,
    *,
    exclusion_keys: set[str],
    field_name: str,
) -> None:
    if not isinstance(value, MappingABC) or value.get("path") is not None:
        return
    if value.get("sha256") != _key_set_sha256(exclusion_keys):
        raise ValueError(
            f"{field_name}.sha256 does not replay the persisted exclusion set"
        )


def _with_manifest_fingerprint(body: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(body)
    manifest["manifest_fingerprint_sha256"] = _json_sha256(body)
    return manifest


def _validate_holdout_manifest_details(
    manifest: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
) -> None:
    if manifest.get("stage") != "holdout100":
        raise ValueError("holdout manifest stage must be 'holdout100'")
    if manifest.get("seed") != EXP003_HOLDOUT_SEED:
        raise ValueError("holdout manifest seed does not match the frozen seed")
    selection = manifest.get("selection")
    if not isinstance(selection, MappingABC):
        raise ValueError("holdout selection must be a mapping")
    if selection.get("method") != "exclusive_priority_then_sha256_seed_nul_cache_audio_key":
        raise ValueError("holdout selection method does not match the frozen method")
    if selection.get("priority_order") != list(EXP003_PRIORITY):
        raise ValueError("holdout priority order does not match the frozen order")
    if selection.get("quotas") != EXP003_HOLDOUT_QUOTAS:
        raise ValueError("holdout quotas do not match the frozen quotas")
    if selection.get("target_audio_count") != EXP003_HOLDOUT_AUDIO_COUNT:
        raise ValueError("holdout target_audio_count must be 100")
    available_counts = selection.get("available_counts_after_exclusion")
    if not isinstance(available_counts, MappingABC) or set(available_counts) != set(
        EXP003_PRIORITY
    ):
        raise ValueError("holdout available counts must cover every frozen quota")
    for quota in EXP003_PRIORITY:
        available = available_counts.get(quota)
        if isinstance(available, bool) or not isinstance(available, int):
            raise ValueError(f"holdout available count for {quota!r} must be an integer")
        if available < EXP003_HOLDOUT_QUOTAS[quota]:
            raise ValueError(f"holdout available count for {quota!r} is underfilled")
    if len(entries) != EXP003_HOLDOUT_AUDIO_COUNT:
        raise ValueError(
            f"holdout manifest must contain {EXP003_HOLDOUT_AUDIO_COUNT} audio groups"
        )
    counts = manifest.get("selected_counts")
    expected_counts = {
        quota: sum(1 for entry in entries if entry.get("quota_assignment") == quota)
        for quota in EXP003_PRIORITY
    }
    if counts != expected_counts or counts != EXP003_HOLDOUT_QUOTAS:
        raise ValueError("holdout selected_counts do not match frozen quotas")
    _validate_holdout_entry_order(entries)


def _validate_broad_manifest_details(
    manifest: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
) -> None:
    if manifest.get("stage") != "broad500":
        raise ValueError("broad manifest stage must be 'broad500'")
    if manifest.get("seed") != EXP003_BROAD_SEED:
        raise ValueError("broad manifest seed does not match the frozen seed")
    selection = manifest.get("selection")
    if not isinstance(selection, MappingABC):
        raise ValueError("broad selection must be a mapping")
    if selection.get("method") != "holdout100_then_lowest_sha256_seed_nul_cache_audio_key":
        raise ValueError("broad selection method does not match the frozen method")
    if selection.get("holdout_audio_count") != EXP003_HOLDOUT_AUDIO_COUNT:
        raise ValueError("broad selection holdout_audio_count must be 100")
    if selection.get("added_audio_count") != EXP003_BROAD_ADDED_AUDIO_COUNT:
        raise ValueError("broad selection added_audio_count must be 400")
    if selection.get("target_audio_count") != EXP003_BROAD_AUDIO_COUNT:
        raise ValueError("broad selection target_audio_count must be 500")
    candidate_count = selection.get("added_candidate_count_after_exclusions")
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int):
        raise ValueError("broad added candidate count must be an integer")
    if candidate_count < EXP003_BROAD_ADDED_AUDIO_COUNT:
        raise ValueError("broad added candidate count is underfilled")
    if len(entries) != EXP003_BROAD_AUDIO_COUNT:
        raise ValueError(
            f"broad manifest must contain {EXP003_BROAD_AUDIO_COUNT} audio groups"
        )
    holdout_entries = entries[:EXP003_HOLDOUT_AUDIO_COUNT]
    added_entries = entries[EXP003_HOLDOUT_AUDIO_COUNT:]
    if any(entry.get("stage") != "holdout100" for entry in holdout_entries):
        raise ValueError("the first 100 broad entries must be the frozen holdout")
    if any(entry.get("stage") != "broad500_added" for entry in added_entries):
        raise ValueError("the final 400 broad entries must be broad500_added")
    if len(added_entries) != EXP003_BROAD_ADDED_AUDIO_COUNT:
        raise ValueError("broad manifest must contain exactly 400 added audio groups")
    _validate_holdout_entry_order(holdout_entries)
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
    if [entry.get("selection_rank") for entry in added_entries] != list(
        range(1, EXP003_BROAD_ADDED_AUDIO_COUNT + 1)
    ):
        raise ValueError("broad added selection ranks must be contiguous from 1 to 400")
    if any(entry.get("quota_assignment") is not None for entry in added_entries):
        raise ValueError("broad added entries must not have quota assignments")
    counts = manifest.get("selected_counts")
    if counts != {
        "holdout100": EXP003_HOLDOUT_AUDIO_COUNT,
        "broad500_added": EXP003_BROAD_ADDED_AUDIO_COUNT,
    }:
        raise ValueError("broad selected_counts do not match the frozen 100+400 split")
    holdout = manifest.get("holdout")
    if not isinstance(holdout, MappingABC):
        raise ValueError("broad holdout provenance must be a mapping")
    if holdout.get("schema") != TIMING_V3_EXP003_HOLDOUT_MANIFEST_SCHEMA:
        raise ValueError("broad holdout provenance schema is invalid")
    if not _is_sha256(holdout.get("manifest_fingerprint_sha256")):
        raise ValueError("broad holdout manifest fingerprint is invalid")
    if holdout.get("cache_audio_keys_sha256") != _key_set_sha256(
        {str(entry["cache_audio_key"]) for entry in holdout_entries}
    ):
        raise ValueError("broad holdout cache-key hash does not match its first 100 rows")


def _validate_holdout_entry_order(entries: Sequence[Mapping[str, Any]]) -> None:
    cursor = 0
    for quota in EXP003_PRIORITY:
        quota_count = EXP003_HOLDOUT_QUOTAS[quota]
        quota_entries = list(entries[cursor : cursor + quota_count])
        cursor += quota_count
        if len(quota_entries) != quota_count:
            raise ValueError(f"holdout quota {quota!r} has the wrong number of entries")
        if any(entry.get("stage") != "holdout100" for entry in quota_entries):
            raise ValueError("holdout entries must all use stage='holdout100'")
        if any(entry.get("quota_assignment") != quota for entry in quota_entries):
            raise ValueError("holdout entries do not follow the frozen exclusive quota order")
        if [entry.get("selection_rank") for entry in quota_entries] != list(
            range(1, quota_count + 1)
        ):
            raise ValueError(
                f"holdout quota {quota!r} selection ranks must be contiguous from 1"
            )
        expected_order = sorted(
            quota_entries,
            key=lambda entry: (
                str(entry["selection_hash_sha256"]),
                str(entry["cache_audio_key"]),
                str(entry["audio_group_key"]),
            ),
        )
        if quota_entries != expected_order:
            raise ValueError(
                f"holdout quota {quota!r} entries are not in frozen SHA-256 rank order"
            )
        for entry in quota_entries:
            if _entry_quota_assignment(entry) != quota:
                raise ValueError(
                    f"holdout entry does not satisfy its quota assignment {quota!r}"
                )
    if cursor != len(entries):
        raise ValueError("holdout contains entries outside the frozen quotas")


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
        else:
            raw_items = [document]
    else:
        raise ValueError(f"{path} must contain JSON objects")
    items = []
    for index, item in enumerate(raw_items):
        if not isinstance(item, MappingABC):
            raise ValueError(f"{path}: selected item {index} must be a mapping")
        items.append(item)
    return items


def _optional_cache_audio_key(item: Mapping[str, Any]) -> str | None:
    for field_name in ("cache_audio_key", "audio_key"):
        value = _optional_nonempty_string(item.get(field_name))
        if value is not None:
            return value
    source = item.get("source")
    if isinstance(source, MappingABC):
        return _optional_nonempty_string(source.get("cache_audio_key"))
    return None


def _validated_key_set(values: Sequence[str] | set[str] | frozenset[str], *, field_name: str) -> set[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (SequenceABC, set, frozenset)):
        raise ValueError(f"{field_name} must be a sequence or set of strings")
    keys: set[str] = set()
    for index, value in enumerate(values):
        key = _required_nonempty_string(value, field_name=f"{field_name}[{index}]")
        if key in keys:
            raise ValueError(f"{field_name} contains duplicate cache audio key {key!r}")
        keys.add(key)
    return keys


def _validate_frozen_exclusion_sets(
    *,
    pilot_exclusions: set[str],
    protocol_exclusions: set[str],
) -> None:
    if len(pilot_exclusions) != EXP003_PILOT_AUDIO_COUNT:
        raise ValueError(
            f"Experiment 003 requires exactly {EXP003_PILOT_AUDIO_COUNT} frozen "
            f"pilot exclusions, got {len(pilot_exclusions)}"
        )
    if len(protocol_exclusions) != EXP003_PROTOCOL_EXCLUSION_AUDIO_COUNT:
        raise ValueError(
            "Experiment 003 protocol v2 requires exactly "
            f"{EXP003_PROTOCOL_EXCLUSION_AUDIO_COUNT} protocol exclusions, "
            f"got {len(protocol_exclusions)}"
        )
    overlap = sorted(pilot_exclusions.intersection(protocol_exclusions))
    if overlap:
        raise ValueError(
            "frozen pilot and protocol exclusions must be disjoint: "
            f"{overlap[:3]!r}"
        )


def _validate_exclusion_provenance(
    value: Any,
    *,
    field_name: str,
    expected_count: int,
) -> set[str]:
    if not isinstance(value, MappingABC):
        raise ValueError(f"manifest {field_name} provenance must be a mapping")
    if value.get("cache_audio_key_count") != expected_count:
        raise ValueError(
            f"{field_name}.cache_audio_key_count must be {expected_count}"
        )
    if not _is_sha256(value.get("cache_audio_keys_sha256")):
        raise ValueError(
            f"{field_name}.cache_audio_keys_sha256 must be a lowercase SHA-256 digest"
        )
    if value.get("hash_encoding") != "sha256_canonical_json_sorted_string_array_utf8":
        raise ValueError(f"{field_name} hash encoding is invalid")
    raw_keys = value.get("cache_audio_keys")
    if isinstance(raw_keys, (str, bytes)) or not isinstance(raw_keys, SequenceABC):
        raise ValueError(f"{field_name}.cache_audio_keys must be a sorted string list")
    keys = _validated_key_set(
        list(raw_keys),
        field_name=f"{field_name}.cache_audio_keys",
    )
    if list(raw_keys) != sorted(keys):
        raise ValueError(f"{field_name}.cache_audio_keys must be in sorted order")
    if len(keys) != expected_count:
        raise ValueError(
            f"{field_name}.cache_audio_keys must contain exactly {expected_count} keys"
        )
    if value.get("cache_audio_keys_sha256") != _key_set_sha256(keys):
        raise ValueError(f"{field_name} cache-audio-key hash does not replay")
    return keys


def _key_set_provenance(keys: set[str]) -> dict[str, Any]:
    return {
        "cache_audio_key_count": len(keys),
        "cache_audio_keys": sorted(keys),
        "cache_audio_keys_sha256": _key_set_sha256(keys),
        "hash_encoding": "sha256_canonical_json_sorted_string_array_utf8",
    }


def _key_set_sha256(keys: set[str]) -> str:
    return _json_sha256(sorted(keys))


def _selection_hash(seed: str, cache_audio_key: str) -> str:
    return hashlib.sha256(f"{seed}\0{cache_audio_key}".encode("utf-8")).hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_provenance(path: Path, *, row_count: int) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "sha256": _file_sha256(path),
        "row_count": row_count,
    }


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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [json.dumps(row, allow_nan=False, sort_keys=True) for row in rows]
    _write_raw_jsonl_atomic(path, lines)


def _write_raw_jsonl_atomic(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
            for line in lines:
                handle.write(line)
                handle.write("\n")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build frozen Timing v3 Experiment 003 splits.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    holdout = subparsers.add_parser("holdout100", help="Build the frozen new 100-audio holdout.")
    _add_common_split_arguments(holdout)

    broad = subparsers.add_parser("broad500", help="Build holdout100 plus 400 new audio groups.")
    _add_common_split_arguments(broad)
    broad.add_argument("--holdout-manifest", type=Path, required=True)

    subset = subparsers.add_parser(
        "baseline-subset",
        help="Materialize full-baseline rows in a frozen manifest's order.",
    )
    subset.add_argument("--baseline-jsonl", type=Path, required=True)
    subset.add_argument("--manifest", type=Path, required=True)
    subset.add_argument("--output-jsonl", type=Path, required=True)
    return parser


def _add_common_split_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--labels-jsonl", type=Path, required=True)
    parser.add_argument("--pilot-exclusion", type=Path, required=True)
    parser.add_argument(
        "--protocol-exclusion",
        type=Path,
        required=True,
        help="Independent JSON/JSONL source containing exactly the three exposed v1 rows.",
    )
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--label-rows-output", type=Path, default=None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.command == "holdout100":
        result = build_exp003_holdout100(
            labels_jsonl_path=args.labels_jsonl,
            pilot_exclusion_path=args.pilot_exclusion,
            protocol_exclusion_path=args.protocol_exclusion,
            manifest_output_path=args.manifest_output,
            label_rows_output_path=args.label_rows_output,
        )
    elif args.command == "broad500":
        result = build_exp003_broad500(
            labels_jsonl_path=args.labels_jsonl,
            pilot_exclusion_path=args.pilot_exclusion,
            protocol_exclusion_path=args.protocol_exclusion,
            holdout_manifest_path=args.holdout_manifest,
            manifest_output_path=args.manifest_output,
            label_rows_output_path=args.label_rows_output,
        )
    elif args.command == "baseline-subset":
        result = materialize_baseline_subset(
            baseline_jsonl_path=args.baseline_jsonl,
            manifest_path=args.manifest,
            output_jsonl_path=args.output_jsonl,
        )
    else:  # pragma: no cover - argparse owns the finite command set.
        raise AssertionError(f"unhandled command {args.command!r}")
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
