from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from pulsefield_model.osu_core.hitobjects import parse_mania_hit_objects
from pulsefield_model.osu_core.timing import RedTimingPoint, require_red_timing_points
from pulsefield_model.timing.evaluation.evidence import (
    DEFAULT_TEMPO_ALIAS_MULTIPLIERS,
    REDLINE_EVIDENCE_AMBIGUOUS,
    REDLINE_EVIDENCE_DENSE,
    REDLINE_EVIDENCE_JUMP_CANDIDATE,
    REDLINE_EVIDENCE_MISSING,
    REDLINE_EVIDENCE_RAMP_CANDIDATE,
    REDLINE_EVIDENCE_STABLE,
    BeatmapTimingEvidence,
    ObjectGridEvidenceConfig,
    ObjectGridEvidence,
    RedlineEvidenceConfig,
    RedlineSummary,
    summarize_beatmap_timing_evidence,
    summarize_object_grid_evidence,
)
from pulsefield_model.timing.evaluation.inventory import TIMING_V3_INVENTORY_AUDIO_ROW_SCHEMA


TIMING_V3_LABEL_AUDIO_ROW_SCHEMA = "pulsefield_model.timing_v3_label_audio_row_v1"
TIMING_V3_LABEL_REPORT_SCHEMA = "pulsefield_model.timing_v3_label_report_v1"
TIMING_V3_LABEL_PILOT_SCHEMA = "pulsefield_model.timing_v3_label_pilot_v1"
TIMING_V3_LABEL_ROW_FINGERPRINT_SCHEMA = "pulsefield_model.timing_v3_label_row_fingerprint_v1"
TIMING_V3_LABEL_RUN_CONTEXT_SCHEMA = "pulsefield_model.timing_v3_label_run_context_v1"

LABEL_STABLE = REDLINE_EVIDENCE_STABLE
LABEL_JUMP_CANDIDATE = REDLINE_EVIDENCE_JUMP_CANDIDATE
LABEL_DENSE = REDLINE_EVIDENCE_DENSE
LABEL_RAMP_CANDIDATE = REDLINE_EVIDENCE_RAMP_CANDIDATE
LABEL_AMBIGUOUS = REDLINE_EVIDENCE_AMBIGUOUS

LABEL_STRATA = (
    LABEL_STABLE,
    LABEL_JUMP_CANDIDATE,
    LABEL_DENSE,
    LABEL_RAMP_CANDIDATE,
    LABEL_AMBIGUOUS,
)
REDLINE_CLASSES = (
    REDLINE_EVIDENCE_MISSING,
    REDLINE_EVIDENCE_STABLE,
    REDLINE_EVIDENCE_JUMP_CANDIDATE,
    REDLINE_EVIDENCE_DENSE,
    REDLINE_EVIDENCE_RAMP_CANDIDATE,
    REDLINE_EVIDENCE_AMBIGUOUS,
)
PILOT_STRATA = ("stable", "jump", "ramp", "dense", "long", "anomaly")
PILOT_QUOTA_GROUPS = (
    ("stable", ("stable",)),
    ("jump", ("jump",)),
    ("tempo_change", ("ramp", "dense")),
    ("long", ("long",)),
    ("anomaly", ("anomaly",)),
)
DEFAULT_PILOT_QUOTAS = {
    "stable": 20,
    "jump": 20,
    "tempo_change": 20,
    "long": 10,
    "anomaly": 10,
}
DEFAULT_LONG_TRACK_THRESHOLD_SECONDS = 600.0
DEFAULT_METADATA_BPM_ABS_TOLERANCE = 0.25
DEFAULT_METADATA_BPM_REL_TOLERANCE = 0.002
DEFAULT_PROGRESS_EVERY = 25
DEFAULT_CHECKPOINT_EVERY = 100


@dataclass(frozen=True)
class _MapEvidenceRecord:
    map_row: Mapping[str, Any]
    beatmap_path: Path | None
    ok: bool
    redline_points: tuple[RedTimingPoint, ...]
    timing_evidence: BeatmapTimingEvidence | None
    redline_signature: str | None
    error_type: str | None
    error: str | None


def build_timing_v3_labels(
    *,
    inventory_path: str | Path,
    output_jsonl_path: str | Path,
    report_path: str | Path,
    dataset_root: str | Path | None = None,
    limit: int | None = None,
    expected_key_count: int | None = 4,
    pilot_output_path: str | Path | None = None,
    pilot_rows_output_path: str | Path | None = None,
    pilot_quotas: Mapping[str, int] | None = None,
    pilot_per_stratum: int | None = None,
    pilot_seed: str = "timing-v3-001",
    long_threshold_seconds: float = DEFAULT_LONG_TRACK_THRESHOLD_SECONDS,
    progress_every: int = DEFAULT_PROGRESS_EVERY,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
) -> dict[str, Any]:
    """Build compact Timing v3 weak labels from canonical audio inventory JSONL.

    The output is evaluation evidence only. It parses local `.osu` red timing
    and object placement plus local `metadata.json` scalar BPM. It does not load
    BeatThis activations, run model inference, or call the v2/v3 fitter.
    """

    started_at_unix = time.time()
    inventory_path = Path(inventory_path)
    output_jsonl_path = Path(output_jsonl_path)
    report_path = Path(report_path)
    dataset_root_path = Path(dataset_root) if dataset_root is not None else None
    if output_jsonl_path == report_path:
        raise ValueError("output_jsonl_path and report_path must be different explicit files")
    if pilot_output_path is not None:
        pilot_output_path = Path(pilot_output_path)
        if pilot_output_path in {output_jsonl_path, report_path}:
            raise ValueError("pilot_output_path must be distinct from label output files")
    if pilot_rows_output_path is not None:
        pilot_rows_output_path = Path(pilot_rows_output_path)
        if pilot_rows_output_path in {output_jsonl_path, report_path, pilot_output_path}:
            raise ValueError("pilot_rows_output_path must be distinct from label output files")
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be non-negative, got {limit!r}")
    resolved_pilot_quotas = _resolve_pilot_quotas(pilot_quotas, pilot_per_stratum=pilot_per_stratum)
    if long_threshold_seconds < 0.0:
        raise ValueError(f"long_threshold_seconds must be non-negative, got {long_threshold_seconds!r}")
    if progress_every < 0:
        raise ValueError(f"progress_every must be non-negative, got {progress_every!r}")
    if checkpoint_every < 0:
        raise ValueError(f"checkpoint_every must be non-negative, got {checkpoint_every!r}")

    inventory_sha256 = _file_sha256(inventory_path)
    run_context = _run_context_payload(
        inventory_sha256=inventory_sha256,
        expected_key_count=expected_key_count,
        long_threshold_seconds=long_threshold_seconds,
    )

    inventory_rows = load_timing_v3_label_inventory(
        inventory_path,
        dataset_root=dataset_root_path,
        limit=limit,
    )
    existing_rows = _read_existing_label_rows(output_jsonl_path)
    results_by_key: dict[str, dict[str, Any]] = {}
    processed_count = 0
    resumed_count = 0
    stale_count = 0
    checkpoint_write_count = 0

    for row_index, audio_row in enumerate(inventory_rows, start=1):
        audio_key = str(audio_row["audio_group_key"])
        fingerprint = _row_fingerprint(audio_row, run_context=run_context)
        existing = existing_rows.get(audio_key)
        if existing is not None and _row_fingerprint_matches(existing, fingerprint):
            results_by_key[audio_key] = existing
            resumed_count += 1
        else:
            if existing is not None:
                stale_count += 1
            label_row = _label_audio_row(
                audio_row,
                inventory_path=inventory_path,
                dataset_root=dataset_root_path,
                expected_key_count=expected_key_count,
                long_threshold_seconds=long_threshold_seconds,
                fingerprint=fingerprint,
            )
            results_by_key[audio_key] = label_row
            processed_count += 1
            if checkpoint_every > 0 and processed_count % checkpoint_every == 0:
                _write_jsonl_atomic(output_jsonl_path, _ordered_label_rows(inventory_rows, results_by_key))
                checkpoint_write_count += 1

        if progress_every > 0 and (row_index == 1 or row_index % progress_every == 0 or row_index == len(inventory_rows)):
            _print_progress(
                current=row_index,
                total=len(inventory_rows),
                last_key=audio_key,
                processed_count=processed_count,
                resumed_count=resumed_count,
                stale_count=stale_count,
            )

    label_rows = _ordered_label_rows(inventory_rows, results_by_key)
    _write_jsonl_atomic(output_jsonl_path, label_rows)

    pilot = None
    if pilot_output_path is not None or pilot_rows_output_path is not None:
        pilot = select_timing_v3_pilot(
            label_rows,
            quotas=resolved_pilot_quotas,
            seed=pilot_seed,
            long_threshold_seconds=long_threshold_seconds,
        )
    if pilot_output_path is not None:
        _write_json_atomic(pilot_output_path, pilot)

    pilot_row_count = 0
    if pilot_rows_output_path is not None:
        pilot_rows = _pilot_full_label_rows(label_rows, pilot)
        pilot_row_count = len(pilot_rows)
        _write_jsonl_atomic(pilot_rows_output_path, pilot_rows)

    finished_at_unix = time.time()
    report = _report_payload(
        inventory_path=inventory_path,
        output_jsonl_path=output_jsonl_path,
        report_path=report_path,
        pilot_output_path=pilot_output_path,
        pilot_rows_output_path=pilot_rows_output_path,
        rows=label_rows,
        limit=limit,
        expected_key_count=expected_key_count,
        long_threshold_seconds=long_threshold_seconds,
        pilot=pilot,
        run={
            "run_context": run_context,
            "processed_count": processed_count,
            "resumed_count": resumed_count,
            "stale_count": stale_count,
            "checkpoint_every": checkpoint_every,
            "progress_every": progress_every,
            "checkpoint_write_count": checkpoint_write_count,
            "final_write_count": 1,
            "pilot_rows_count": pilot_row_count,
            "started_at_unix": started_at_unix,
            "finished_at_unix": finished_at_unix,
            "total_seconds": finished_at_unix - started_at_unix,
        },
    )
    _write_json_atomic(report_path, report)
    return report


def load_timing_v3_label_inventory(
    inventory_path: str | Path,
    *,
    dataset_root: Path | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load canonical inventory JSONL in a deterministic audio-group order."""

    inventory_path = Path(inventory_path)
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be non-negative, got {limit!r}")

    rows_by_audio_key: dict[str, dict[str, Any]] = {}
    with inventory_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{inventory_path}:{line_number} is not valid JSON") from exc
            if not isinstance(payload, MappingABC):
                raise ValueError(f"{inventory_path}:{line_number} must be a JSON object")

            row = _normalize_inventory_row(
                payload,
                inventory_path=inventory_path,
                line_number=line_number,
                dataset_root=dataset_root,
            )
            audio_key = str(row["audio_group_key"])
            existing = rows_by_audio_key.get(audio_key)
            if existing is None:
                rows_by_audio_key[audio_key] = row
            else:
                rows_by_audio_key[audio_key] = _merge_duplicate_inventory_rows(existing, row)

    rows = sorted(
        rows_by_audio_key.values(),
        key=lambda row: (str(row["audio_group_key"]), str(row.get("resolved_audio_path") or "")),
    )
    if limit is not None:
        rows = rows[:limit]
    return rows


def select_timing_v3_pilot(
    label_rows: Sequence[Mapping[str, Any]],
    *,
    quotas: Mapping[str, int] | None = None,
    per_stratum: int | None = None,
    seed: str = "timing-v3-001",
    long_threshold_seconds: float = DEFAULT_LONG_TRACK_THRESHOLD_SECONDS,
) -> dict[str, Any]:
    """Select an audio-grouped pilot manifest using stable hash ordering."""

    resolved_quotas = _resolve_pilot_quotas(quotas, pilot_per_stratum=per_stratum)
    if long_threshold_seconds < 0.0:
        raise ValueError(f"long_threshold_seconds must be non-negative, got {long_threshold_seconds!r}")

    assigned: set[str] = set()
    selected: dict[str, list[dict[str, Any]]] = {stratum: [] for stratum in PILOT_STRATA}
    candidates = {stratum: _pilot_candidates(label_rows, stratum, long_threshold_seconds) for stratum in PILOT_STRATA}

    quota_group_payloads: list[dict[str, Any]] = []
    for group_name, group_strata in PILOT_QUOTA_GROUPS:
        quota = resolved_quotas[group_name]
        selected_count_before = sum(len(selected[stratum]) for stratum in group_strata)
        if group_name == "tempo_change":
            ordered = _ordered_tempo_change_pilot_candidates(candidates, seed=seed)
            ramp_available_count = len(
                {
                    _row_audio_key(row)
                    for row in ordered["ramp"]
                    if _row_audio_key(row) not in assigned
                }
            )
            ramp_reserved_quota = min(10, quota, ramp_available_count)
            _select_pilot_rows(
                ordered["ramp"],
                output_stratum="ramp",
                quota=ramp_reserved_quota,
                selected=selected,
                assigned=assigned,
            )
            remaining_quota = quota - (sum(len(selected[stratum]) for stratum in group_strata) - selected_count_before)
            _select_pilot_rows(
                ordered["dense"],
                output_stratum="dense",
                quota=remaining_quota,
                selected=selected,
                assigned=assigned,
            )
            remaining_quota = quota - (sum(len(selected[stratum]) for stratum in group_strata) - selected_count_before)
            _select_pilot_rows(
                ordered["ramp"],
                output_stratum="ramp",
                quota=remaining_quota,
                selected=selected,
                assigned=assigned,
            )
        else:
            ordered = _ordered_pilot_group_candidates(
                candidates,
                group_name=group_name,
                group_strata=group_strata,
                seed=seed,
            )
            _select_pilot_rows(
                ordered,
                output_stratum=group_name,
                quota=quota,
                selected=selected,
                assigned=assigned,
            )
        selected_count = sum(len(selected[stratum]) for stratum in group_strata) - selected_count_before
        quota_group_payloads.append(
            _quota_group_payload(
                group_name=group_name,
                group_strata=group_strata,
                quota=quota,
                selected=selected,
                selected_count_before=selected_count_before,
                selected_count=selected_count,
                candidates=candidates,
            )
        )

    return {
        "schema": TIMING_V3_LABEL_PILOT_SCHEMA,
        "seed": seed,
        "selection": {
            "method": "audio_grouped_stable_sha256",
            "priority_order": [group_name for group_name, _group_strata in PILOT_QUOTA_GROUPS],
            "target_audio_count": sum(resolved_quotas.values()),
            "quotas": dict(resolved_quotas),
            "long_threshold_seconds": long_threshold_seconds,
        },
        "source": {
            "audio_group_count": len({_row_audio_key(row) for row in label_rows}),
        },
        "selected_audio_count": len(assigned),
        "selected": selected,
        "available_counts": {
            stratum: len({_row_audio_key(row) for row in rows})
            for stratum, rows in candidates.items()
        },
        "quota_groups": quota_group_payloads,
        "available_counts_by_quota_group": {
            item["name"]: item["available_count"] for item in quota_group_payloads
        },
        "selected_counts_by_quota_group": {
            item["name"]: item["selected_count"] for item in quota_group_payloads
        },
        "underfilled_counts_by_quota_group": {
            item["name"]: item["underfilled_count"] for item in quota_group_payloads
        },
        "selected_counts": {stratum: len(rows) for stratum, rows in selected.items()},
        "underfilled_counts": {stratum: None for stratum in PILOT_STRATA},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    report = build_timing_v3_labels(
        inventory_path=args.inventory,
        output_jsonl_path=args.output_jsonl,
        report_path=args.report_json,
        dataset_root=args.dataset_root,
        limit=args.limit,
        expected_key_count=args.expected_key_count,
        pilot_output_path=args.pilot_output_json,
        pilot_rows_output_path=args.pilot_rows_output_jsonl,
        pilot_quotas=(
            None
            if args.pilot_per_stratum is not None
            else {
                "stable": args.pilot_stable_quota,
                "jump": args.pilot_jump_quota,
                "tempo_change": args.pilot_tempo_change_quota,
                "long": args.pilot_long_quota,
                "anomaly": args.pilot_anomaly_quota,
            }
        ),
        pilot_per_stratum=args.pilot_per_stratum,
        pilot_seed=args.pilot_seed,
        long_threshold_seconds=args.long_threshold_seconds,
        progress_every=args.progress_every,
        checkpoint_every=args.checkpoint_every,
    )
    if args.json:
        print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    else:
        print(_format_summary(report))
    return 0


def _read_existing_label_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    rows: dict[str, dict[str, Any]] = {}
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
            audio_key = payload.get("audio_group_key")
            if isinstance(audio_key, str) and audio_key:
                rows[audio_key] = payload
    return rows


def _ordered_label_rows(
    inventory_rows: Sequence[Mapping[str, Any]],
    results_by_key: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    inventory_keys = {str(row["audio_group_key"]) for row in inventory_rows}
    return [
        dict(results_by_key[audio_key])
        for audio_key in sorted(inventory_keys)
        if audio_key in results_by_key
    ]


def _run_context_payload(
    *,
    inventory_sha256: str,
    expected_key_count: int | None,
    long_threshold_seconds: float,
) -> dict[str, Any]:
    return {
        "schema": TIMING_V3_LABEL_RUN_CONTEXT_SCHEMA,
        "label_schema": TIMING_V3_LABEL_AUDIO_ROW_SCHEMA,
        "inventory_schema": TIMING_V3_INVENTORY_AUDIO_ROW_SCHEMA,
        "inventory_sha256": inventory_sha256,
        "expected_key_count": expected_key_count,
        "threshold_defaults": {
            "redline_evidence": _json_safe(asdict(RedlineEvidenceConfig())),
            "object_grid_evidence": _json_safe(asdict(ObjectGridEvidenceConfig())),
            "metadata_bpm_abs_tolerance": DEFAULT_METADATA_BPM_ABS_TOLERANCE,
            "metadata_bpm_rel_tolerance": DEFAULT_METADATA_BPM_REL_TOLERANCE,
            "long_threshold_seconds": long_threshold_seconds,
            "tempo_alias_multipliers": list(DEFAULT_TEMPO_ALIAS_MULTIPLIERS),
            "pilot_quotas": dict(DEFAULT_PILOT_QUOTAS),
        },
        "label_policy": {
            "stable_jump_phase_corroboration": "object_grid.grid_supported",
            "alias_family_confirmation": "object_grid.alias_resolved",
            "ramp_candidate_ambiguous_until_manual_confirmation": True,
            "dense_redlines_ambiguous_until_audit": True,
        },
    }


def _row_fingerprint(audio_row: Mapping[str, Any], *, run_context: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema": TIMING_V3_LABEL_ROW_FINGERPRINT_SCHEMA,
        "context": run_context,
        "row_identity": _row_identity(audio_row),
        "row_content_sha256": _json_sha256(audio_row),
    }
    return {**body, "sha256": _json_sha256(body)}


def _row_identity(audio_row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "audio_group_key": str(audio_row["audio_group_key"]),
        "audio_group_index": audio_row.get("audio_group_index"),
        "resolved_audio_path": audio_row.get("resolved_audio_path"),
        "source_line_numbers": list(audio_row.get("source_line_numbers", [])),
        "map_count": int(audio_row.get("map_count", len(_inventory_maps(audio_row)))),
    }


def _row_fingerprint_matches(row: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    fingerprint = row.get("fingerprint")
    return (
        isinstance(fingerprint, MappingABC)
        and fingerprint.get("schema") == TIMING_V3_LABEL_ROW_FINGERPRINT_SCHEMA
        and fingerprint.get("sha256") == expected.get("sha256")
        and fingerprint.get("row_content_sha256") == expected.get("row_content_sha256")
        and fingerprint.get("context") == expected.get("context")
    )


def _pilot_full_label_rows(
    label_rows: Sequence[Mapping[str, Any]],
    pilot: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if pilot is None:
        return []
    selected = pilot.get("selected")
    if not isinstance(selected, MappingABC):
        return []
    annotations: dict[str, dict[str, str]] = {}
    for quota_group, rows in selected.items():
        if not isinstance(rows, SequenceABC) or isinstance(rows, (str, bytes)):
            continue
        for item in rows:
            if not isinstance(item, MappingABC) or not isinstance(item.get("audio_group_key"), str):
                continue
            pilot_stratum = item.get("pilot_stratum")
            annotations[str(item["audio_group_key"])] = {
                "pilot_stratum": str(pilot_stratum or quota_group),
                "pilot_quota_group": _pilot_quota_group_for_stratum(str(pilot_stratum or quota_group)),
            }

    rows: list[dict[str, Any]] = []
    for row in sorted(label_rows, key=lambda item: _row_audio_key(item)):
        annotation = annotations.get(_row_audio_key(row))
        if annotation is None:
            continue
        payload = dict(row)
        payload.update(annotation)
        rows.append(payload)
    return rows


def _pilot_quota_group_for_stratum(pilot_stratum: str) -> str:
    if pilot_stratum in {"ramp", "dense"}:
        return "tempo_change"
    return pilot_stratum


def _print_progress(
    *,
    current: int,
    total: int,
    last_key: str,
    processed_count: int,
    resumed_count: int,
    stale_count: int,
) -> None:
    print(
        "[timing-v3-labels] "
        f"processed={current}/{total} "
        f"computed={processed_count} "
        f"resumed={resumed_count} "
        f"stale={stale_count} "
        f"last_key={last_key}",
        file=sys.stderr,
        flush=True,
    )


def _label_audio_row(
    audio_row: Mapping[str, Any],
    *,
    inventory_path: Path,
    dataset_root: Path | None,
    expected_key_count: int | None,
    long_threshold_seconds: float,
    fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    map_records = [
        _parse_map_evidence(
            map_row,
            inventory_path=inventory_path,
            dataset_root=dataset_root,
            expected_key_count=expected_key_count,
        )
        for map_row in _inventory_maps(audio_row)
    ]
    valid_records = [record for record in map_records if record.ok and record.timing_evidence is not None]
    representative = _choose_representative_record(valid_records)
    cross_map_evidence = _cross_map_object_evidence(
        map_records,
        representative=representative,
        expected_key_count=expected_key_count,
    )
    representative_payload = _representative_payload(representative, valid_records)
    metadata_evidence = _metadata_bpm_evidence(
        audio_row,
        representative=representative,
        inventory_path=inventory_path,
        dataset_root=dataset_root,
    )
    evidence_counts = _audio_evidence_counts(
        audio_row=audio_row,
        map_records=map_records,
        cross_map_evidence=cross_map_evidence,
        metadata_evidence=metadata_evidence,
    )
    label = _audio_label(
        audio_row=audio_row,
        map_records=map_records,
        representative=representative,
        cross_map_evidence=cross_map_evidence,
        evidence_counts=evidence_counts,
    )
    duration_seconds = _duration_seconds(audio_row)

    return {
        "schema": TIMING_V3_LABEL_AUDIO_ROW_SCHEMA,
        "fingerprint": dict(fingerprint),
        "audio_group_index": audio_row.get("audio_group_index"),
        "audio_group_key": str(audio_row["audio_group_key"]),
        "resolved_audio_path": audio_row.get("resolved_audio_path"),
        "map_count": len(map_records),
        "source": {
            "inventory_schema": audio_row.get("schema"),
            "inventory_line_numbers": audio_row.get("source_line_numbers", []),
            "cache_status": _nested_get(audio_row, ("cache", "status")),
            "cache_audio_key": _nested_get(audio_row, ("cache", "audio_cache_key")),
            "cache_config_fingerprint": _nested_get(audio_row, ("cache", "config_fingerprint")),
            "cache_duration_seconds": duration_seconds,
            "long_threshold_seconds": long_threshold_seconds,
            "long_track": duration_seconds is not None and duration_seconds >= long_threshold_seconds,
        },
        "inventory_anomalies": sorted(str(item) for item in audio_row.get("anomalies", [])),
        "label": label,
        "evidence_counts": evidence_counts,
        "representative_redline_grid": representative_payload,
        "maps": [
            _map_record_payload(record, cross_map_evidence.get(_record_key(record)))
            for record in map_records
        ],
        "metadata_bpm_evidence": metadata_evidence,
    }


def _parse_map_evidence(
    map_row: Mapping[str, Any],
    *,
    inventory_path: Path,
    dataset_root: Path | None,
    expected_key_count: int | None,
) -> _MapEvidenceRecord:
    beatmap_path = _resolve_beatmap_path(
        map_row,
        inventory_path=inventory_path,
        dataset_root=dataset_root,
    )
    try:
        if beatmap_path is None:
            raise FileNotFoundError("inventory map row does not provide a beatmap path")
        if not beatmap_path.is_file():
            raise FileNotFoundError(f"beatmap file does not exist: {beatmap_path.as_posix()}")
        redline_points = tuple(require_red_timing_points(beatmap_path))
        timing_evidence = summarize_beatmap_timing_evidence(
            beatmap_path,
            expected_key_count=expected_key_count,
        )
        return _MapEvidenceRecord(
            map_row=map_row,
            beatmap_path=beatmap_path,
            ok=True,
            redline_points=redline_points,
            timing_evidence=timing_evidence,
            redline_signature=_redline_signature(redline_points),
            error_type=None,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001 - labels must isolate bad maps as row evidence errors.
        return _MapEvidenceRecord(
            map_row=map_row,
            beatmap_path=beatmap_path,
            ok=False,
            redline_points=(),
            timing_evidence=None,
            redline_signature=None,
            error_type=exc.__class__.__name__,
            error=str(exc),
        )


def _choose_representative_record(records: Sequence[_MapEvidenceRecord]) -> _MapEvidenceRecord | None:
    if not records:
        return None
    signature_counts = Counter(record.redline_signature for record in records if record.redline_signature is not None)
    return sorted(
        records,
        key=lambda record: (
            -signature_counts[str(record.redline_signature)],
            -_redline_class_priority(record.timing_evidence.redlines.evidence_class if record.timing_evidence else None),
            -int(bool(record.timing_evidence and record.timing_evidence.object_grid.supported)),
            int(bool(record.timing_evidence and record.timing_evidence.object_grid.ambiguous)),
            _record_sort_path(record),
        ),
    )[0]


def _cross_map_object_evidence(
    records: Sequence[_MapEvidenceRecord],
    *,
    representative: _MapEvidenceRecord | None,
    expected_key_count: int | None,
) -> dict[str, dict[str, Any]]:
    if representative is None:
        return {}

    evidence: dict[str, dict[str, Any]] = {}
    for record in records:
        key = _record_key(record)
        if record is representative:
            evidence[key] = {
                "ok": True,
                "evidence_kind": "representative_redline_grid_self",
                "source": "representative map; not counted as cross-map support",
                "representative_beatmap_path": _path_or_none(representative.beatmap_path),
                "object_grid": None,
                "error_type": None,
                "error": None,
            }
            continue
        try:
            if record.beatmap_path is None:
                raise FileNotFoundError("inventory map row does not provide a beatmap path")
            if not record.beatmap_path.is_file():
                raise FileNotFoundError(f"beatmap file does not exist: {record.beatmap_path.as_posix()}")
            hitobjects = parse_mania_hit_objects(record.beatmap_path, expected_key_count=expected_key_count)
            object_grid = summarize_object_grid_evidence(hitobjects, representative.redline_points)
            evidence[key] = {
                "ok": True,
                "evidence_kind": "correlated_weak_cross_difficulty_object_grid",
                "source": (
                    "objects from this difficulty scored against the deterministic representative "
                    "redline grid; correlated weak evidence, not timing truth"
                ),
                "representative_beatmap_path": _path_or_none(representative.beatmap_path),
                "object_grid": _object_grid_payload(object_grid),
                "error_type": None,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - cross-map evidence is optional and weak.
            evidence[key] = {
                "ok": False,
                "evidence_kind": "correlated_weak_cross_difficulty_object_grid",
                "source": (
                    "objects from this difficulty scored against the deterministic representative "
                    "redline grid; correlated weak evidence, not timing truth"
                ),
                "representative_beatmap_path": _path_or_none(representative.beatmap_path),
                "object_grid": None,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
            }
    return evidence


def _audio_label(
    *,
    audio_row: Mapping[str, Any],
    map_records: Sequence[_MapEvidenceRecord],
    representative: _MapEvidenceRecord | None,
    cross_map_evidence: Mapping[str, Mapping[str, Any]],
    evidence_counts: Mapping[str, Any],
) -> dict[str, Any]:
    valid_records = [record for record in map_records if record.ok and record.timing_evidence is not None]
    class_counts = Counter(
        record.timing_evidence.redlines.evidence_class
        for record in valid_records
        if record.timing_evidence is not None
    )
    error_count = int(evidence_counts["map_error_count"])
    anomaly_count = len(audio_row.get("anomalies", []))
    representative_signature_count = (
        sum(1 for record in valid_records if record.redline_signature == representative.redline_signature)
        if representative is not None
        else 0
    )
    consensus_complete = bool(valid_records) and representative_signature_count == len(valid_records)
    own_grid_supported = any(
        bool(record.timing_evidence and record.timing_evidence.object_grid.grid_supported)
        for record in valid_records
    )
    own_alias_resolved = any(
        bool(record.timing_evidence and record.timing_evidence.object_grid.alias_resolved)
        for record in valid_records
    )
    cross_grid_supported_count = int(evidence_counts["cross_map_grid_supported_count"])
    cross_alias_resolved_count = int(evidence_counts["cross_map_alias_resolved_count"])
    cross_scored_count = int(evidence_counts["cross_map_scored_count"])

    reasons = [
        f"valid_maps={len(valid_records)}/{len(map_records)}",
        _class_count_reason(class_counts),
    ]
    if representative is not None:
        reasons.append(f"representative_redline_grid_agreement={representative_signature_count}/{len(valid_records)}")
    if own_grid_supported:
        reasons.append("own_difficulty_object_grid_phase_supported")
    if own_alias_resolved:
        reasons.append("own_difficulty_object_grid_alias_resolved")
    if cross_scored_count:
        reasons.append(f"cross_map_grid_phase_support={cross_grid_supported_count}/{cross_scored_count}")
        reasons.append(f"cross_map_alias_resolved={cross_alias_resolved_count}/{cross_scored_count}")
    if anomaly_count:
        reasons.append(f"inventory_anomalies={anomaly_count}")
    if error_count:
        reasons.append(f"evidence_errors={error_count}")

    if not valid_records:
        return _label_payload(
            stratum=LABEL_AMBIGUOUS,
            confidence="none",
            ambiguous=True,
            audit_candidate=True,
            reasons=(*reasons, "no_valid_redline_evidence"),
        )

    if class_counts[REDLINE_EVIDENCE_RAMP_CANDIDATE] > 0:
        return _label_payload(
            stratum=LABEL_RAMP_CANDIDATE,
            confidence="low",
            ambiguous=True,
            audit_candidate=True,
            reasons=(
                *reasons,
                "ramp_candidates_require_manual_confirmation",
                "metadata_bpm_not_used_for_label_inference",
            ),
        )

    if class_counts[REDLINE_EVIDENCE_DENSE] > 0:
        return _label_payload(
            stratum=LABEL_DENSE,
            confidence="low",
            ambiguous=True,
            audit_candidate=True,
            reasons=(*reasons, "dense_redlines_require_audit"),
        )

    nonstable_classes = sum(
        class_counts[item]
        for item in (REDLINE_EVIDENCE_JUMP_CANDIDATE, REDLINE_EVIDENCE_AMBIGUOUS, REDLINE_EVIDENCE_MISSING)
    )
    if class_counts[REDLINE_EVIDENCE_STABLE] == len(valid_records) and nonstable_classes == 0:
        confidence, ambiguous, audit_candidate = _confidence_from_support(
            valid_count=len(valid_records),
            consensus_complete=consensus_complete,
            own_grid_supported=own_grid_supported,
            cross_grid_supported_count=cross_grid_supported_count,
            cross_scored_count=cross_scored_count,
            error_count=error_count,
            anomaly_count=anomaly_count,
        )
        return _label_payload(
            stratum=LABEL_STABLE,
            confidence=confidence,
            ambiguous=ambiguous,
            audit_candidate=audit_candidate,
            reasons=tuple(reasons),
        )

    nonjump_classes = sum(
        class_counts[item]
        for item in (REDLINE_EVIDENCE_STABLE, REDLINE_EVIDENCE_AMBIGUOUS, REDLINE_EVIDENCE_MISSING)
    )
    if class_counts[REDLINE_EVIDENCE_JUMP_CANDIDATE] == len(valid_records) and nonjump_classes == 0:
        confidence, ambiguous, audit_candidate = _confidence_from_support(
            valid_count=len(valid_records),
            consensus_complete=consensus_complete,
            own_grid_supported=own_grid_supported,
            cross_grid_supported_count=cross_grid_supported_count,
            cross_scored_count=cross_scored_count,
            error_count=error_count,
            anomaly_count=anomaly_count,
        )
        return _label_payload(
            stratum=LABEL_JUMP_CANDIDATE,
            confidence=confidence,
            ambiguous=ambiguous,
            audit_candidate=audit_candidate,
            reasons=tuple(reasons),
        )

    return _label_payload(
        stratum=LABEL_AMBIGUOUS,
        confidence="low",
        ambiguous=True,
        audit_candidate=True,
        reasons=(*reasons, "conflicting_or_weak_audio_level_evidence"),
    )


def _confidence_from_support(
    *,
    valid_count: int,
    consensus_complete: bool,
    own_grid_supported: bool,
    cross_grid_supported_count: int,
    cross_scored_count: int,
    error_count: int,
    anomaly_count: int,
) -> tuple[str, bool, bool]:
    if not own_grid_supported:
        return "low", True, True
    if error_count or anomaly_count or not consensus_complete:
        return "low", True, True
    if valid_count >= 2:
        if cross_scored_count > 0 and cross_grid_supported_count == cross_scored_count:
            return "high", False, False
        return "low", True, True
    return "medium", False, False


def _label_payload(
    *,
    stratum: str,
    confidence: str,
    ambiguous: bool,
    audit_candidate: bool,
    reasons: Sequence[str],
) -> dict[str, Any]:
    return {
        "stratum": stratum,
        "confidence": confidence,
        "ambiguous": bool(ambiguous),
        "audit_candidate": bool(audit_candidate),
        "reasons": list(reasons),
    }


def _audio_evidence_counts(
    *,
    audio_row: Mapping[str, Any],
    map_records: Sequence[_MapEvidenceRecord],
    cross_map_evidence: Mapping[str, Mapping[str, Any]],
    metadata_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    valid_records = [record for record in map_records if record.ok and record.timing_evidence is not None]
    class_counts = Counter(
        record.timing_evidence.redlines.evidence_class
        for record in valid_records
        if record.timing_evidence is not None
    )
    cross_scored = [
        payload
        for payload in cross_map_evidence.values()
        if payload.get("evidence_kind") == "correlated_weak_cross_difficulty_object_grid"
    ]
    cross_grid_supported = [
        payload
        for payload in cross_scored
        if payload.get("ok")
        and isinstance((object_grid := payload.get("object_grid")), MappingABC)
        and bool(object_grid.get("grid_supported"))
    ]
    cross_alias_resolved = [
        payload
        for payload in cross_scored
        if payload.get("ok")
        and isinstance((object_grid := payload.get("object_grid")), MappingABC)
        and bool(object_grid.get("alias_resolved"))
    ]
    cross_supported = [
        payload
        for payload in cross_scored
        if payload.get("ok")
        and isinstance((object_grid := payload.get("object_grid")), MappingABC)
        and bool(object_grid.get("supported"))
    ]
    return {
        "maps_total": len(map_records),
        "maps_ok": len(valid_records),
        "map_error_count": len(map_records) - len(valid_records),
        "redline_classes": {key: int(class_counts[key]) for key in REDLINE_CLASSES},
        "object_grid_supported_count": sum(
            1
            for record in valid_records
            if record.timing_evidence is not None and record.timing_evidence.object_grid.grid_supported
        ),
        "object_alias_resolved_count": sum(
            1
            for record in valid_records
            if record.timing_evidence is not None and record.timing_evidence.object_grid.alias_resolved
        ),
        "object_supported_count": sum(
            1
            for record in valid_records
            if record.timing_evidence is not None and record.timing_evidence.object_grid.supported
        ),
        "object_ambiguous_count": sum(
            1
            for record in valid_records
            if record.timing_evidence is not None and record.timing_evidence.object_grid.ambiguous
        ),
        "cross_map_scored_count": len(cross_scored),
        "cross_map_grid_supported_count": len(cross_grid_supported),
        "cross_map_alias_resolved_count": len(cross_alias_resolved),
        "cross_map_supported_count": len(cross_supported),
        "cross_map_error_count": sum(1 for payload in cross_scored if not payload.get("ok")),
        "metadata_bpm_source_count": int(metadata_evidence["source_count"]),
        "metadata_bpm_alias_agreement_count": int(metadata_evidence["alias_agreement"]["agreeing_source_count"]),
        "inventory_anomaly_count": len(audio_row.get("anomalies", [])),
    }


def _metadata_bpm_evidence(
    audio_row: Mapping[str, Any],
    *,
    representative: _MapEvidenceRecord | None,
    inventory_path: Path,
    dataset_root: Path | None,
) -> dict[str, Any]:
    metadata_paths = _metadata_paths(audio_row)
    inventory_beatmap_ids = {
        int(value)
        for map_row in _inventory_maps(audio_row)
        if (value := _int_or_none(map_row.get("beatmap_id"))) is not None
    }
    sources: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for item in metadata_paths:
        path = _resolve_metadata_path(str(item["path"]), inventory_path=inventory_path, dataset_root=dataset_root)
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, MappingABC):
                raise ValueError("metadata JSON root must be an object")
            sources.extend(_metadata_bpm_sources(payload, path, inventory_beatmap_ids))
        except Exception as exc:  # noqa: BLE001 - metadata is weak evidence.
            errors.append(
                {
                    "metadata_json_path": path.as_posix(),
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                }
            )

    reference_bpms = (
        list(representative.timing_evidence.redlines.unique_bpms)
        if representative is not None and representative.timing_evidence is not None
        else []
    )
    comparisons = [
        _metadata_alias_comparison(source, reference_bpms)
        for source in sources
    ]
    agreeing = [comparison for comparison in comparisons if comparison["agrees"]]
    agreement_status = _metadata_agreement_status(
        source_count=len(sources),
        reference_count=len(reference_bpms),
        agreeing_count=len(agreeing),
    )
    return {
        "source": "local_osu_metadata_json_scalar_bpm",
        "provider": "osu_api_v2_snapshot",
        "trust_note": "map-derived scalar BPM cross-check; never used as timing inference input",
        "used_for_label_inference": False,
        "metadata_json_paths": metadata_paths,
        "source_count": len(sources),
        "sources": sources,
        "alias_agreement": {
            "status": agreement_status,
            "reference": "deterministic_representative_redline_unique_bpms",
            "reference_bpms": reference_bpms,
            "alias_multipliers": list(DEFAULT_TEMPO_ALIAS_MULTIPLIERS),
            "bpm_abs_tolerance": DEFAULT_METADATA_BPM_ABS_TOLERANCE,
            "bpm_rel_tolerance": DEFAULT_METADATA_BPM_REL_TOLERANCE,
            "comparison_count": len(comparisons),
            "agreeing_source_count": len(agreeing),
            "comparisons": comparisons,
        },
        "errors": errors,
    }


def _metadata_bpm_sources(
    payload: Mapping[str, Any],
    path: Path,
    inventory_beatmap_ids: set[int],
) -> list[dict[str, Any]]:
    beatmapset_id = _int_or_none(payload.get("id"))
    api_endpoint = payload.get("api_endpoint") if isinstance(payload.get("api_endpoint"), str) else None
    fetched_at = payload.get("fetched_at") if isinstance(payload.get("fetched_at"), str) else None
    sources: list[dict[str, Any]] = []

    beatmapset_bpm = _positive_float_or_none(payload.get("bpm"))
    if beatmapset_bpm is not None:
        sources.append(
            {
                "source_role": "beatmapset",
                "metadata_json_path": path.as_posix(),
                "api_endpoint": api_endpoint,
                "fetched_at": fetched_at,
                "beatmapset_id": beatmapset_id,
                "beatmap_id": None,
                "matched_inventory_map": True,
                "bpm": beatmapset_bpm,
                "map_derived": True,
            }
        )

    raw_beatmaps = payload.get("beatmaps")
    if isinstance(raw_beatmaps, SequenceABC) and not isinstance(raw_beatmaps, (str, bytes)):
        for beatmap in raw_beatmaps:
            if not isinstance(beatmap, MappingABC):
                continue
            beatmap_id = _int_or_none(beatmap.get("id"))
            if inventory_beatmap_ids and beatmap_id not in inventory_beatmap_ids:
                continue
            bpm = _positive_float_or_none(beatmap.get("bpm"))
            if bpm is None:
                continue
            sources.append(
                {
                    "source_role": "beatmap",
                    "metadata_json_path": path.as_posix(),
                    "api_endpoint": api_endpoint,
                    "fetched_at": fetched_at,
                    "beatmapset_id": _int_or_none(beatmap.get("beatmapset_id")) or beatmapset_id,
                    "beatmap_id": beatmap_id,
                    "matched_inventory_map": beatmap_id in inventory_beatmap_ids if inventory_beatmap_ids else None,
                    "bpm": bpm,
                    "map_derived": True,
                }
            )
    return sources


def _metadata_alias_comparison(source: Mapping[str, Any], reference_bpms: Sequence[float]) -> dict[str, Any]:
    source_bpm = float(source["bpm"])
    best: dict[str, Any] | None = None
    for reference_bpm in reference_bpms:
        for multiplier in DEFAULT_TEMPO_ALIAS_MULTIPLIERS:
            aliased_bpm = source_bpm * multiplier
            delta = abs(aliased_bpm - reference_bpm)
            tolerance = max(
                DEFAULT_METADATA_BPM_ABS_TOLERANCE,
                DEFAULT_METADATA_BPM_REL_TOLERANCE * max(abs(aliased_bpm), abs(reference_bpm)),
            )
            candidate = {
                "reference_bpm": float(reference_bpm),
                "alias_multiplier": float(multiplier),
                "aliased_source_bpm": float(aliased_bpm),
                "abs_delta_bpm": float(delta),
                "tolerance_bpm": float(tolerance),
                "agrees": delta <= tolerance,
            }
            if best is None or (delta, abs(math.log2(multiplier)), reference_bpm) < (
                best["abs_delta_bpm"],
                abs(math.log2(best["alias_multiplier"])),
                best["reference_bpm"],
            ):
                best = candidate
    if best is None:
        best = {
            "reference_bpm": None,
            "alias_multiplier": None,
            "aliased_source_bpm": None,
            "abs_delta_bpm": None,
            "tolerance_bpm": None,
            "agrees": False,
        }

    return {
        "source_role": source.get("source_role"),
        "beatmapset_id": source.get("beatmapset_id"),
        "beatmap_id": source.get("beatmap_id"),
        "source_bpm": source_bpm,
        **best,
    }


def _metadata_agreement_status(
    *,
    source_count: int,
    reference_count: int,
    agreeing_count: int,
) -> str:
    if reference_count == 0:
        return "missing_reference_redline_bpm"
    if source_count == 0:
        return "missing_metadata_bpm"
    if agreeing_count == source_count:
        return "all_agree_alias_aware"
    if agreeing_count > 0:
        return "partial_agreement_alias_aware"
    return "no_alias_aware_agreement"


def _map_record_payload(
    record: _MapEvidenceRecord,
    cross_map_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "source_row_index": record.map_row.get("source_row_index"),
        "beatmap_id": record.map_row.get("beatmap_id"),
        "beatmap_path": _path_or_none(record.beatmap_path),
        "inventory_beatmap_path": record.map_row.get("beatmap_path"),
        "ok": record.ok,
        "redline_signature": record.redline_signature,
        "redlines": None,
        "object_grid": None,
        "cross_map_object_grid": cross_map_evidence,
        "error_type": record.error_type,
        "error": record.error,
    }
    if record.timing_evidence is not None:
        payload["redlines"] = _redline_payload(record.timing_evidence.redlines)
        payload["object_grid"] = _object_grid_payload(record.timing_evidence.object_grid)
    return payload


def _redline_payload(summary: RedlineSummary) -> dict[str, Any]:
    change_bpm_deltas = [change.bpm_delta for change in summary.changes]
    change_phase_residuals = [change.previous_grid_phase_residual_ms for change in summary.changes]
    return {
        "evidence_class": summary.evidence_class,
        "ambiguous": summary.ambiguous,
        "redline_count": summary.redline_count,
        "unique_bpms": list(summary.unique_bpms),
        "bpm_min": min(summary.bpms) if summary.bpms else None,
        "bpm_max": max(summary.bpms) if summary.bpms else None,
        "significant_change_count": summary.significant_change_count,
        "map_duration_ms": summary.map_duration_ms,
        "stable_evidence": summary.stable_evidence,
        "jump_evidence": summary.jump_evidence,
        "dense_evidence": summary.dense_evidence,
        "ramp_candidate_evidence": summary.ramp_candidate_evidence,
        "ramp_direction": summary.ramp_direction,
        "ramp_linear_r2": summary.ramp_linear_r2,
        "change_gap_ms": _number_stats(summary.change_gaps_ms),
        "change_bpm_delta": _number_stats(change_bpm_deltas),
        "change_phase_residual_ms": _number_stats(change_phase_residuals),
        "reasons": list(summary.reasons),
    }


def _object_grid_payload(evidence: ObjectGridEvidence) -> dict[str, Any]:
    best = evidence.best_evidence
    return {
        "start_event_count": evidence.start_event_count,
        "hold_end_event_count": evidence.hold_end_event_count,
        "total_event_count": evidence.total_event_count,
        "start_time_span_ms": evidence.start_time_span_ms,
        "hold_end_weight": evidence.hold_end_weight,
        "inlier_threshold_ms": evidence.inlier_threshold_ms,
        "subdivision_candidate_count": len(evidence.subdivisions) * len(evidence.alias_multipliers),
        "best_alias_multiplier": evidence.best_alias_multiplier,
        "best_subdivision": evidence.best_subdivision,
        "grid_supported": evidence.grid_supported,
        "alias_resolved": evidence.alias_resolved,
        "supported": evidence.supported,
        "ambiguous": evidence.ambiguous,
        "best": (
            None
            if best is None
            else {
                "alias_multiplier": best.alias_multiplier,
                "subdivision": best.subdivision,
                "start_stats": _json_safe(asdict(best.start_stats)),
                "hold_end_stats": _json_safe(asdict(best.hold_end_stats)),
                "combined_stats": _json_safe(asdict(best.combined_stats)),
            }
        ),
        "reasons": list(evidence.reasons),
    }


def _representative_payload(
    representative: _MapEvidenceRecord | None,
    valid_records: Sequence[_MapEvidenceRecord],
) -> dict[str, Any] | None:
    if representative is None or representative.timing_evidence is None:
        return None
    signature_count = sum(1 for record in valid_records if record.redline_signature == representative.redline_signature)
    redlines = representative.timing_evidence.redlines
    return {
        "beatmap_path": _path_or_none(representative.beatmap_path),
        "beatmap_id": representative.map_row.get("beatmap_id"),
        "redline_signature": representative.redline_signature,
        "signature_audio_group_count": signature_count,
        "valid_map_count": len(valid_records),
        "agreement_rate": 0.0 if not valid_records else signature_count / len(valid_records),
        "evidence_class": redlines.evidence_class,
        "redline_count": redlines.redline_count,
        "unique_bpms": list(redlines.unique_bpms),
        "points_head": _redline_points_payload(representative.redline_points[:5]),
        "points_tail": _redline_points_payload(representative.redline_points[-5:]),
    }


def _redline_points_payload(points: Sequence[RedTimingPoint]) -> list[dict[str, Any]]:
    return [
        {
            "offset_ms": float(point.offset_ms),
            "beat_length_ms": float(point.beat_length_ms),
            "bpm": 60000.0 / float(point.beat_length_ms),
            "meter": int(point.meter),
        }
        for point in points
    ]


def _report_payload(
    *,
    inventory_path: Path,
    output_jsonl_path: Path,
    report_path: Path,
    pilot_output_path: Path | None,
    pilot_rows_output_path: Path | None,
    rows: Sequence[Mapping[str, Any]],
    limit: int | None,
    expected_key_count: int | None,
    long_threshold_seconds: float,
    pilot: Mapping[str, Any] | None,
    run: Mapping[str, Any],
) -> dict[str, Any]:
    stratum_counts = Counter(str(row["label"]["stratum"]) for row in rows)
    confidence_counts = Counter(str(row["label"]["confidence"]) for row in rows)
    redline_counts: Counter[str] = Counter()
    label_reason_counts: Counter[str] = Counter()
    redline_reason_counts: Counter[str] = Counter()
    object_reason_counts: Counter[str] = Counter()
    evidence_error_types: Counter[str] = Counter()

    for row in rows:
        for reason in row["label"]["reasons"]:
            label_reason_counts[str(reason)] += 1
        for map_payload in row["maps"]:
            if map_payload.get("error_type"):
                evidence_error_types[str(map_payload["error_type"])] += 1
            redlines = map_payload.get("redlines")
            if isinstance(redlines, MappingABC):
                redline_counts[str(redlines["evidence_class"])] += 1
                for reason in redlines.get("reasons", []):
                    redline_reason_counts[str(reason)] += 1
            object_grid = map_payload.get("object_grid")
            if isinstance(object_grid, MappingABC):
                for reason in object_grid.get("reasons", []):
                    object_reason_counts[str(reason)] += 1

    return {
        "schema": TIMING_V3_LABEL_REPORT_SCHEMA,
        "command": _format_command(sys.argv[1:] if sys.argv else None),
        "source": {
            "inventory_path": inventory_path.as_posix(),
            "inventory_sha256": _file_sha256(inventory_path),
            "inventory_schema": TIMING_V3_INVENTORY_AUDIO_ROW_SCHEMA,
            "limit": limit,
            "expected_key_count": expected_key_count,
        },
        "output": {
            "label_jsonl_path": output_jsonl_path.as_posix(),
            "label_jsonl_sha256": _file_sha256(output_jsonl_path),
            "report_path": report_path.as_posix(),
            "pilot_output_path": _path_or_none(pilot_output_path),
            "pilot_output_sha256": _file_sha256(pilot_output_path) if pilot_output_path is not None else None,
            "pilot_rows_output_path": _path_or_none(pilot_rows_output_path),
            "pilot_rows_output_sha256": (
                _file_sha256(pilot_rows_output_path) if pilot_rows_output_path is not None else None
            ),
            "pilot_rows_count": int(run.get("pilot_rows_count", 0)),
        },
        "run": {
            "started_at_unix": float(run["started_at_unix"]),
            "finished_at_unix": float(run["finished_at_unix"]),
            "total_seconds": float(run["total_seconds"]),
            "processed_count": int(run["processed_count"]),
            "resumed_count": int(run["resumed_count"]),
            "stale_count": int(run["stale_count"]),
            "checkpoint_every": int(run["checkpoint_every"]),
            "progress_every": int(run["progress_every"]),
            "checkpoint_write_count": int(run["checkpoint_write_count"]),
            "final_write_count": int(run["final_write_count"]),
            "run_context": run["run_context"],
        },
        "audio": {
            "audio_group_count": len(rows),
            "map_count": sum(int(row["map_count"]) for row in rows),
            "long_threshold_seconds": long_threshold_seconds,
            "long_audio_count": sum(1 for row in rows if row["source"]["long_track"]),
        },
        "strata": {
            "audio_counts": {key: int(stratum_counts[key]) for key in LABEL_STRATA},
            "confidence_counts": dict(sorted((key, int(value)) for key, value in confidence_counts.items())),
            "ambiguous_audio_count": sum(1 for row in rows if bool(row["label"]["ambiguous"])),
            "audit_candidate_audio_count": sum(1 for row in rows if bool(row["label"]["audit_candidate"])),
        },
        "evidence": {
            "map_redline_class_counts": {key: int(redline_counts[key]) for key in REDLINE_CLASSES},
            "map_error_count": sum(int(row["evidence_counts"]["map_error_count"]) for row in rows),
            "evidence_error_audio_count": sum(
                1 for row in rows if int(row["evidence_counts"]["map_error_count"]) > 0
            ),
            "evidence_error_types": dict(sorted((key, int(value)) for key, value in evidence_error_types.items())),
            "object_grid_supported_map_count": sum(
                int(row["evidence_counts"]["object_grid_supported_count"]) for row in rows
            ),
            "object_alias_resolved_map_count": sum(
                int(row["evidence_counts"]["object_alias_resolved_count"]) for row in rows
            ),
            "object_supported_map_count": sum(
                int(row["evidence_counts"]["object_supported_count"]) for row in rows
            ),
            "cross_map_scored_count": sum(int(row["evidence_counts"]["cross_map_scored_count"]) for row in rows),
            "cross_map_grid_supported_count": sum(
                int(row["evidence_counts"]["cross_map_grid_supported_count"]) for row in rows
            ),
            "cross_map_alias_resolved_count": sum(
                int(row["evidence_counts"]["cross_map_alias_resolved_count"]) for row in rows
            ),
            "cross_map_supported_count": sum(
                int(row["evidence_counts"]["cross_map_supported_count"]) for row in rows
            ),
            "metadata_bpm_source_count": sum(
                int(row["evidence_counts"]["metadata_bpm_source_count"]) for row in rows
            ),
            "metadata_bpm_alias_agreement_count": sum(
                int(row["evidence_counts"]["metadata_bpm_alias_agreement_count"]) for row in rows
            ),
            "raw_reason_counts": {
                "label": dict(sorted((key, int(value)) for key, value in label_reason_counts.items())),
                "redline": dict(sorted((key, int(value)) for key, value in redline_reason_counts.items())),
                "object_grid": dict(sorted((key, int(value)) for key, value in object_reason_counts.items())),
            },
        },
        "pilot": None if pilot is None else {
            "schema": pilot.get("schema"),
            "selected_audio_count": pilot.get("selected_audio_count"),
            "selected_counts": pilot.get("selected_counts"),
            "available_counts": pilot.get("available_counts"),
            "quota_groups": pilot.get("quota_groups"),
            "selected_counts_by_quota_group": pilot.get("selected_counts_by_quota_group"),
            "underfilled_counts": pilot.get("underfilled_counts"),
            "underfilled_counts_by_quota_group": pilot.get("underfilled_counts_by_quota_group"),
        },
    }


def _normalize_inventory_row(
    payload: Mapping[str, Any],
    *,
    inventory_path: Path,
    line_number: int,
    dataset_root: Path | None,
) -> dict[str, Any]:
    maps = payload.get("maps")
    if not isinstance(maps, SequenceABC) or isinstance(maps, (str, bytes)):
        raise ValueError(f"{inventory_path}:{line_number} must provide a maps list")

    normalized = _json_safe(dict(payload))
    audio_key = _first_string(payload, ("audio_group_key", "resolved_audio_path", "audio_path"))
    if audio_key is None:
        raise ValueError(f"{inventory_path}:{line_number} must provide audio_group_key or resolved_audio_path")
    resolved_audio_path = _resolve_optional_path(
        _first_string(payload, ("resolved_audio_path", "audio_path")),
        inventory_path=inventory_path,
        dataset_root=dataset_root,
        shard=_first_string(payload, ("shard",)),
    )
    if resolved_audio_path is not None:
        normalized["resolved_audio_path"] = resolved_audio_path.as_posix()
        if "audio_group_key" not in normalized or not normalized["audio_group_key"]:
            audio_key = resolved_audio_path.as_posix()
    normalized["audio_group_key"] = str(audio_key)
    normalized["source_line_numbers"] = [line_number]
    normalized["maps"] = [
        _normalize_map_row(
            item,
            inventory_path=inventory_path,
            dataset_root=dataset_root,
        )
        for item in maps
        if isinstance(item, MappingABC)
    ]
    normalized["map_count"] = len(normalized["maps"])
    normalized.setdefault("anomalies", [])
    return normalized


def _normalize_map_row(
    map_row: Mapping[str, Any],
    *,
    inventory_path: Path,
    dataset_root: Path | None,
) -> dict[str, Any]:
    normalized = _json_safe(dict(map_row))
    beatmap_path = _resolve_beatmap_path(
        normalized,
        inventory_path=inventory_path,
        dataset_root=dataset_root,
    )
    if beatmap_path is not None:
        normalized["resolved_beatmap_path"] = beatmap_path.as_posix()
    return normalized


def _merge_duplicate_inventory_rows(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    merged = _json_safe(dict(first))
    merged["source_line_numbers"] = sorted(
        {int(value) for value in (*first.get("source_line_numbers", []), *second.get("source_line_numbers", []))}
    )
    merged["maps"] = _merge_map_rows(_inventory_maps(first), _inventory_maps(second))
    merged["map_count"] = len(merged["maps"])
    merged["anomalies"] = sorted(
        {
            *(str(item) for item in first.get("anomalies", [])),
            *(str(item) for item in second.get("anomalies", [])),
            "duplicate_inventory_audio_group_key",
        }
    )
    merged["metadata_json"] = _merge_metadata_json(first.get("metadata_json"), second.get("metadata_json"))
    return merged


def _merge_map_rows(first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in (*first, *second):
        key = (str(row.get("resolved_beatmap_path") or row.get("beatmap_path") or ""), str(row.get("beatmap_id")))
        rows.setdefault(key, _json_safe(dict(row)))
    return [rows[key] for key in sorted(rows)]


def _merge_metadata_json(first: Any, second: Any) -> dict[str, Any]:
    paths: dict[str, dict[str, Any]] = {}
    for raw in (first, second):
        if not isinstance(raw, MappingABC):
            continue
        for item in raw.get("paths", []):
            if isinstance(item, MappingABC) and isinstance(item.get("path"), str):
                paths[str(item["path"])] = _json_safe(dict(item))
    return {
        "paths": [paths[key] for key in sorted(paths)],
        "path_count": len(paths),
        "existing_count": sum(1 for item in paths.values() if item.get("exists")),
    }


def _inventory_maps(audio_row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    maps = audio_row.get("maps", [])
    if not isinstance(maps, SequenceABC) or isinstance(maps, (str, bytes)):
        return []
    return [item for item in maps if isinstance(item, MappingABC)]


def _metadata_paths(audio_row: Mapping[str, Any]) -> list[dict[str, Any]]:
    metadata = audio_row.get("metadata_json")
    if not isinstance(metadata, MappingABC):
        return []
    paths = metadata.get("paths", [])
    if not isinstance(paths, SequenceABC) or isinstance(paths, (str, bytes)):
        return []
    payloads: list[dict[str, Any]] = []
    for item in paths:
        if not isinstance(item, MappingABC) or not isinstance(item.get("path"), str):
            continue
        payloads.append(
            {
                "path": str(item["path"]),
                "exists": bool(item.get("exists")),
                "sha256": item.get("sha256") if isinstance(item.get("sha256"), str) else None,
            }
        )
    return sorted(payloads, key=lambda item: item["path"])


def _resolve_beatmap_path(
    map_row: Mapping[str, Any],
    *,
    inventory_path: Path,
    dataset_root: Path | None,
) -> Path | None:
    raw_path = _first_string(map_row, ("resolved_beatmap_path", "beatmap_path", "osu_path", "path"))
    if raw_path is None:
        return None
    return _resolve_optional_path(
        raw_path,
        inventory_path=inventory_path,
        dataset_root=dataset_root,
        shard=_first_string(map_row, ("shard",)),
    )


def _resolve_metadata_path(
    raw_path: str,
    *,
    inventory_path: Path,
    dataset_root: Path | None,
) -> Path:
    path = _resolve_optional_path(
        raw_path,
        inventory_path=inventory_path,
        dataset_root=dataset_root,
    )
    if path is None:
        raise ValueError("metadata path must be provided")
    return path


def _resolve_optional_path(
    raw_path: str | None,
    *,
    inventory_path: Path,
    dataset_root: Path | None,
    shard: str | None = None,
) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path).expanduser()
    dataset_root_resolved = dataset_root.resolve() if dataset_root is not None else None
    if path.is_absolute():
        resolved_path = path.resolve()
        if dataset_root_resolved is not None:
            _require_path_within_root(resolved_path, dataset_root_resolved)
        return resolved_path
    relative = _safe_relative_path(raw_path)
    if dataset_root is not None:
        safe_shard = _safe_relative_component(shard, field_name="shard") if shard is not None else None
        if safe_shard is not None:
            resolved_path = (dataset_root_resolved / safe_shard / relative).resolve()
        else:
            resolved_path = (dataset_root_resolved / relative).resolve()
        _require_path_within_root(resolved_path, dataset_root_resolved)
        return resolved_path
    return (inventory_path.parent / relative).resolve()


def _require_path_within_root(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"resolved inventory path escapes dataset_root: path={path.as_posix()}, "
            f"dataset_root={root.as_posix()}"
        ) from exc


def _safe_relative_path(value: str) -> Path:
    pure = PurePosixPath(value.replace("\\", "/"))
    if pure.is_absolute() or any(part == ".." for part in pure.parts):
        raise ValueError(f"path must be relative without '..' components, got {value!r}")
    return Path(*pure.parts)


def _safe_relative_component(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    pure = PurePosixPath(value.replace("\\", "/"))
    if pure.is_absolute() or len(pure.parts) != 1 or pure.parts[0] in {"", ".", ".."}:
        raise ValueError(f"{field_name} must be a single relative path component, got {value!r}")
    return pure.parts[0]


def _first_string(payload: Mapping[str, Any], fields: Sequence[str]) -> str | None:
    for field in fields:
        value = payload.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"inventory field {field!r} must be a string when provided")
        if value:
            return value
    return None


def _redline_signature(points: Sequence[RedTimingPoint]) -> str:
    payload = [
        [format(float(point.offset_ms), ".12g"), format(float(point.beat_length_ms), ".12g"), int(point.meter)]
        for point in points
    ]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()


def _redline_class_priority(evidence_class: str | None) -> int:
    return {
        REDLINE_EVIDENCE_STABLE: 50,
        REDLINE_EVIDENCE_JUMP_CANDIDATE: 40,
        REDLINE_EVIDENCE_RAMP_CANDIDATE: 30,
        REDLINE_EVIDENCE_DENSE: 20,
        REDLINE_EVIDENCE_AMBIGUOUS: 10,
        REDLINE_EVIDENCE_MISSING: 0,
    }.get(str(evidence_class), 0)


def _class_count_reason(class_counts: Counter[str]) -> str:
    values = [f"{key}={int(class_counts[key])}" for key in REDLINE_CLASSES if class_counts[key]]
    return "redline_classes=" + (",".join(values) if values else "none")


def _resolve_pilot_quotas(
    quotas: Mapping[str, int] | None,
    *,
    pilot_per_stratum: int | None,
) -> dict[str, int]:
    if pilot_per_stratum is not None:
        if pilot_per_stratum < 0:
            raise ValueError(f"pilot_per_stratum must be non-negative, got {pilot_per_stratum!r}")
        return {key: int(pilot_per_stratum) for key in DEFAULT_PILOT_QUOTAS}

    resolved = dict(DEFAULT_PILOT_QUOTAS)
    if quotas is not None:
        for key, value in quotas.items():
            if key not in resolved:
                raise ValueError(f"unknown pilot quota key {key!r}; expected {sorted(resolved)}")
            if int(value) < 0:
                raise ValueError(f"pilot quota {key!r} must be non-negative, got {value!r}")
            resolved[key] = int(value)
    return resolved


def _ordered_pilot_group_candidates(
    candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    group_name: str,
    group_strata: Sequence[str],
    seed: str,
) -> list[Mapping[str, Any]]:
    rows_by_audio_key: dict[str, Mapping[str, Any]] = {}
    for stratum in group_strata:
        for row in candidates[stratum]:
            rows_by_audio_key.setdefault(_row_audio_key(row), row)
    return sorted(
        rows_by_audio_key.values(),
        key=lambda row: (
            _stable_hash(seed, group_name, _row_audio_key(row)),
            _pilot_output_stratum(row, group_name=group_name),
            _row_audio_key(row),
        ),
    )


def _ordered_tempo_change_pilot_candidates(
    candidates: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    seed: str,
) -> dict[str, list[Mapping[str, Any]]]:
    return {
        "ramp": _ordered_pilot_stratum_candidates(candidates["ramp"], seed=seed, stratum="tempo_change:ramp"),
        "dense": _ordered_pilot_stratum_candidates(candidates["dense"], seed=seed, stratum="tempo_change:dense"),
    }


def _ordered_pilot_stratum_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: str,
    stratum: str,
) -> list[Mapping[str, Any]]:
    rows_by_audio_key: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        rows_by_audio_key.setdefault(_row_audio_key(row), row)
    return sorted(
        rows_by_audio_key.values(),
        key=lambda row: (
            _stable_hash(seed, stratum, _row_audio_key(row)),
            _row_audio_key(row),
        ),
    )


def _select_pilot_rows(
    ordered: Sequence[Mapping[str, Any]],
    *,
    output_stratum: str,
    quota: int,
    selected: dict[str, list[dict[str, Any]]],
    assigned: set[str],
) -> None:
    if quota <= 0:
        return
    added = 0
    for row in ordered:
        audio_key = _row_audio_key(row)
        if audio_key in assigned:
            continue
        selected[output_stratum].append(_pilot_row_payload(row, stratum=output_stratum))
        assigned.add(audio_key)
        added += 1
        if added >= quota:
            break


def _quota_group_payload(
    *,
    group_name: str,
    group_strata: Sequence[str],
    quota: int,
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_count_before: int,
    selected_count: int,
    candidates: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    available_counts_by_output_stratum = {
        stratum: len({_row_audio_key(row) for row in candidates[stratum]})
        for stratum in group_strata
    }
    payload = {
        "name": group_name,
        "output_strata": list(group_strata),
        "quota": quota,
        "available_count": len(
            {
                _row_audio_key(row)
                for stratum in group_strata
                for row in candidates[stratum]
            }
        ),
        "available_counts_by_output_stratum": available_counts_by_output_stratum,
        "selected_count": selected_count,
        "selected_counts_by_output_stratum": {
            stratum: len(selected[stratum])
            for stratum in group_strata
        },
        "underfilled_count": max(0, quota - selected_count),
    }
    if group_name == "tempo_change":
        ramp_available_count = available_counts_by_output_stratum["ramp"]
        ramp_reserved_quota = min(10, quota, ramp_available_count)
        payload["strategy"] = "ramp_reserve_then_dense_then_ramp_backfill"
        payload["ramp_reserved_quota"] = ramp_reserved_quota
        payload["dense_preferred_after_ramp_reserve"] = True
        payload["reserve_rule"] = "min(10, quota, available_ramp)"
    else:
        payload["strategy"] = "single_stratum_stable_hash"
        payload["ramp_reserved_quota"] = None
    if selected_count_before:
        payload["selected_count_before_group"] = selected_count_before
    return payload


def _pilot_output_stratum(row: Mapping[str, Any], *, group_name: str) -> str:
    if group_name == "tempo_change":
        label_stratum = _nested_get(row, ("label", "stratum"))
        return "ramp" if label_stratum == LABEL_RAMP_CANDIDATE else "dense"
    if group_name in {"stable", "jump", "long", "anomaly"}:
        return group_name
    raise ValueError(f"unknown pilot quota group: {group_name!r}")


def _pilot_candidates(
    label_rows: Sequence[Mapping[str, Any]],
    stratum: str,
    long_threshold_seconds: float,
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for row in label_rows:
        label = row.get("label", {})
        if not isinstance(label, MappingABC):
            continue
        label_stratum = label.get("stratum")
        if stratum == "stable" and label_stratum == LABEL_STABLE:
            rows.append(row)
        elif stratum == "jump" and label_stratum == LABEL_JUMP_CANDIDATE:
            rows.append(row)
        elif stratum == "ramp" and label_stratum == LABEL_RAMP_CANDIDATE:
            rows.append(row)
        elif stratum == "dense" and label_stratum == LABEL_DENSE:
            rows.append(row)
        elif stratum == "long" and _row_duration_seconds(row) is not None and _row_duration_seconds(row) >= long_threshold_seconds:
            rows.append(row)
        elif stratum == "anomaly" and _row_is_anomaly(row):
            rows.append(row)
    return rows


def _pilot_row_payload(row: Mapping[str, Any], *, stratum: str) -> dict[str, Any]:
    return {
        "pilot_stratum": stratum,
        "audio_group_key": _row_audio_key(row),
        "resolved_audio_path": row.get("resolved_audio_path"),
        "label_stratum": _nested_get(row, ("label", "stratum")),
        "confidence": _nested_get(row, ("label", "confidence")),
        "ambiguous": _nested_get(row, ("label", "ambiguous")),
        "audit_candidate": _nested_get(row, ("label", "audit_candidate")),
        "duration_seconds": _row_duration_seconds(row),
        "map_count": row.get("map_count"),
        "evidence_error_count": _nested_get(row, ("evidence_counts", "map_error_count")),
        "inventory_anomalies": row.get("inventory_anomalies", []),
    }


def _row_audio_key(row: Mapping[str, Any]) -> str:
    return str(row.get("audio_group_key") or row.get("resolved_audio_path") or row.get("audio_key") or "")


def _row_duration_seconds(row: Mapping[str, Any]) -> float | None:
    return _float_or_none(_nested_get(row, ("source", "cache_duration_seconds")))


def _row_is_anomaly(row: Mapping[str, Any]) -> bool:
    anomalies = row.get("inventory_anomalies", [])
    evidence_error_count = _nested_get(row, ("evidence_counts", "map_error_count"))
    return bool(anomalies) or (_int_or_none(evidence_error_count) or 0) > 0


def _duration_seconds(audio_row: Mapping[str, Any]) -> float | None:
    return _float_or_none(_nested_get(audio_row, ("cache", "duration_seconds")))


def _record_key(record: _MapEvidenceRecord) -> str:
    if record.beatmap_path is not None:
        return record.beatmap_path.as_posix()
    return str(record.map_row.get("resolved_beatmap_path") or record.map_row.get("beatmap_path") or id(record))


def _record_sort_path(record: _MapEvidenceRecord) -> str:
    return _record_key(record)


def _path_or_none(path: Path | None) -> str | None:
    return path.as_posix() if path is not None else None


def _nested_get(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, MappingABC):
            return None
        current = current.get(key)
    return current


def _positive_float_or_none(value: Any) -> float | None:
    number = _float_or_none(value)
    if number is None or number <= 0.0:
        return None
    return number


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number


def _number_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "min": None, "max": None}
    return {
        "count": len(finite),
        "mean": sum(finite) / len(finite),
        "p50": _percentile(finite, 50.0),
        "p90": _percentile(finite, 90.0),
        "min": finite[0],
        "max": finite[-1],
    }


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * percentile / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[int(index)]
    fraction = index - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _stable_hash(seed: str, stratum: str, audio_key: str) -> str:
    return hashlib.sha256(f"{seed}\0{stratum}\0{audio_key}".encode("utf-8")).hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(_json_safe(value), allow_nan=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, MappingABC):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(_json_safe(row), allow_nan=False, sort_keys=True) + "\n")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.write_text(
            json.dumps(_json_safe(payload), allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_command(argv: Sequence[str] | None) -> str:
    if argv is None:
        return ""
    return " ".join(_shell_quote(arg) for arg in argv)


def _shell_quote(value: object) -> str:
    import shlex

    return shlex.quote(str(value))


def _format_summary(report: Mapping[str, Any]) -> str:
    audio = report["audio"]
    strata = report["strata"]["audio_counts"]
    evidence = report["evidence"]
    return (
        "Timing v3 labels: "
        f"{audio['audio_group_count']} audio groups, "
        f"stable={strata[LABEL_STABLE]}, "
        f"jump={strata[LABEL_JUMP_CANDIDATE]}, "
        f"ramp={strata[LABEL_RAMP_CANDIDATE]}, "
        f"dense={strata[LABEL_DENSE]}, "
        f"ambiguous={strata[LABEL_AMBIGUOUS]}, "
        f"map_errors={evidence['map_error_count']}"
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build compact audio-grouped Timing v3 weak labels from inventory JSONL.",
    )
    parser.add_argument("--inventory", required=True, type=Path, help="Canonical Timing v3 audio inventory JSONL.")
    parser.add_argument("--output-jsonl", required=True, type=Path, help="Enriched per-audio label JSONL.")
    parser.add_argument("--report-json", required=True, type=Path, help="Aggregate label report JSON.")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--expected-key-count", type=int, default=4)
    parser.add_argument("--pilot-output-json", type=Path, default=None)
    parser.add_argument("--pilot-rows-output-jsonl", type=Path, default=None)
    parser.add_argument("--pilot-stable-quota", type=int, default=DEFAULT_PILOT_QUOTAS["stable"])
    parser.add_argument("--pilot-jump-quota", type=int, default=DEFAULT_PILOT_QUOTAS["jump"])
    parser.add_argument("--pilot-tempo-change-quota", type=int, default=DEFAULT_PILOT_QUOTAS["tempo_change"])
    parser.add_argument("--pilot-long-quota", type=int, default=DEFAULT_PILOT_QUOTAS["long"])
    parser.add_argument("--pilot-anomaly-quota", type=int, default=DEFAULT_PILOT_QUOTAS["anomaly"])
    parser.add_argument("--pilot-per-stratum", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--pilot-seed", default="timing-v3-001")
    parser.add_argument("--long-threshold-seconds", type=float, default=DEFAULT_LONG_TRACK_THRESHOLD_SECONDS)
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--json", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
