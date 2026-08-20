from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping as MappingABC
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BaselineArtifactAudit:
    baseline_jsonl_path: str
    baseline_sha256: str
    summary_json_path: str
    inventory_jsonl_path: str | None
    row_count: int
    unique_audio_key_count: int
    unique_resume_fingerprint_count: int
    successful_audio_count: int
    fit_success_audio_count: int
    comparison_eligible_audio_count: int
    comparator_unavailable_audio_count: int
    paired_comparison_count: int
    comparison_failure_count: int
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def require_ok(self) -> BaselineArtifactAudit:
        if self.errors:
            raise ValueError("baseline artifact audit failed: " + "; ".join(self.errors))
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "pulsefield_model.timing_v3_baseline_artifact_audit_v1",
            "ok": self.ok,
            **asdict(self),
        }


def audit_timing_v3_baseline_artifacts(
    baseline_jsonl_path: str | Path,
    summary_json_path: str | Path,
    *,
    inventory_jsonl_path: str | Path | None = None,
) -> BaselineArtifactAudit:
    """Reconcile a durable baseline summary with its actual JSONL rows.

    This is intentionally independent of the cache loader and fitter. It
    catches stale summaries and, in particular, prevents fitted-audio counts
    from being reused as `.osu` comparison denominators.
    """

    baseline_path = Path(baseline_jsonl_path)
    summary_path = Path(summary_json_path)
    inventory_path = Path(inventory_jsonl_path) if inventory_jsonl_path is not None else None
    rows = _load_jsonl_objects(baseline_path)
    summary = _load_json_object(summary_path)

    audio_keys = [row.get("audio_key") for row in rows]
    resume_fingerprints = [
        resume.get("fingerprint")
        for row in rows
        if isinstance((resume := row.get("resume")), MappingABC)
    ]
    comparisons = [
        comparison
        for row in rows
        for comparison in _comparisons(row)
    ]
    successful_audio_count = sum(bool(row.get("ok")) for row in rows)
    failed_audio_count = len(rows) - successful_audio_count
    fit_success_audio_count = sum(row.get("fit") is not None for row in rows)
    fit_failure_audio_count = len(rows) - fit_success_audio_count
    comparison_attempted_audio_count = sum(
        row.get("fit") is not None and len(_comparisons(row)) > 0
        for row in rows
    )
    comparison_eligible_audio_count = sum(
        any(bool(comparison.get("ok")) for comparison in _comparisons(row))
        for row in rows
    )
    comparator_unavailable_audio_count = sum(
        row.get("fit") is not None
        and not any(bool(comparison.get("ok")) for comparison in _comparisons(row))
        for row in rows
    )
    paired_comparison_count = sum(bool(comparison.get("ok")) for comparison in comparisons)
    comparison_failure_count = len(comparisons) - paired_comparison_count
    failure_stage_counts: dict[str, int] = {}
    for row in rows:
        if row.get("ok"):
            continue
        stage = str(row.get("failure_stage") or "unknown")
        failure_stage_counts[stage] = failure_stage_counts.get(stage, 0) + 1

    actual_results = {
        "result_count": len(rows),
        "successful_audio_count": successful_audio_count,
        "failed_audio_count": failed_audio_count,
        "fit_success_audio_count": fit_success_audio_count,
        "fit_failure_audio_count": fit_failure_audio_count,
        "comparison_attempted_audio_count": comparison_attempted_audio_count,
        "comparison_eligible_audio_count": comparison_eligible_audio_count,
        "comparator_unavailable_audio_count": comparator_unavailable_audio_count,
        "paired_comparison_count": paired_comparison_count,
        "comparison_failure_count": comparison_failure_count,
    }
    actual_failures: dict[str, Any] = {
        "failed_audio_count": failed_audio_count,
        "fit_failure_audio_count": fit_failure_audio_count,
        "comparator_unavailable_audio_count": comparator_unavailable_audio_count,
        "comparison_failure_count": comparison_failure_count,
        "stage_counts": dict(sorted(failure_stage_counts.items())),
    }

    errors: list[str] = []
    _compare_summary_mapping(summary, "results", actual_results, errors)
    _compare_summary_mapping(summary, "failures", actual_failures, errors)
    source = summary.get("source")
    if not isinstance(source, MappingABC):
        errors.append("summary.source must be a mapping")
    else:
        _compare_value(source, "unique_audio_count", len(rows), "summary.source", errors)

    valid_audio_keys = [value for value in audio_keys if isinstance(value, str) and value]
    if len(valid_audio_keys) != len(rows):
        errors.append("every baseline row must have a non-empty string audio_key")
    if len(set(valid_audio_keys)) != len(valid_audio_keys):
        errors.append("baseline audio_key values must be unique")
    valid_resume_fingerprints = [
        value for value in resume_fingerprints if isinstance(value, str) and value
    ]
    if len(valid_resume_fingerprints) != len(rows):
        errors.append("every baseline row must have a non-empty resume fingerprint")
    if len(set(valid_resume_fingerprints)) != len(valid_resume_fingerprints):
        errors.append("baseline resume fingerprints must be unique")

    if inventory_path is not None:
        inventory_rows = _load_jsonl_objects(inventory_path)
        inventory_sha256 = _file_sha256(inventory_path)
        map_reference_count = sum(
            len(maps) if isinstance((maps := row.get("maps")), list) else 0
            for row in inventory_rows
        )
        if not isinstance(source, MappingABC):
            pass
        else:
            _compare_value(
                source,
                "inventory_sha256",
                inventory_sha256,
                "summary.source",
                errors,
            )
            _compare_value(
                source,
                "unique_audio_count",
                len(inventory_rows),
                "summary.source",
                errors,
            )
            _compare_value(
                source,
                "map_reference_count",
                map_reference_count,
                "summary.source",
                errors,
            )

    return BaselineArtifactAudit(
        baseline_jsonl_path=baseline_path.as_posix(),
        baseline_sha256=_file_sha256(baseline_path),
        summary_json_path=summary_path.as_posix(),
        inventory_jsonl_path=inventory_path.as_posix() if inventory_path is not None else None,
        row_count=len(rows),
        unique_audio_key_count=len(set(valid_audio_keys)),
        unique_resume_fingerprint_count=len(set(valid_resume_fingerprints)),
        successful_audio_count=successful_audio_count,
        fit_success_audio_count=fit_success_audio_count,
        comparison_eligible_audio_count=comparison_eligible_audio_count,
        comparator_unavailable_audio_count=comparator_unavailable_audio_count,
        paired_comparison_count=paired_comparison_count,
        comparison_failure_count=comparison_failure_count,
        errors=tuple(errors),
    )


def _comparisons(row: MappingABC[str, Any]) -> tuple[MappingABC[str, Any], ...]:
    payload = row.get("comparisons")
    if not isinstance(payload, list):
        return ()
    return tuple(item for item in payload if isinstance(item, MappingABC))


def _compare_summary_mapping(
    summary: MappingABC[str, Any],
    field_name: str,
    actual: MappingABC[str, Any],
    errors: list[str],
) -> None:
    expected = summary.get(field_name)
    if not isinstance(expected, MappingABC):
        errors.append(f"summary.{field_name} must be a mapping")
        return
    for key, value in actual.items():
        _compare_value(expected, key, value, f"summary.{field_name}", errors)


def _compare_value(
    payload: MappingABC[str, Any],
    key: str,
    actual: Any,
    location: str,
    errors: list[str],
) -> None:
    expected = payload.get(key)
    if expected != actual:
        errors.append(f"{location}.{key}={expected!r}, recomputed={actual!r}")


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(payload)
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
