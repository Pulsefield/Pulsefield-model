from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


CHECKPOINT_DIVERGENCE_REPORT_SCHEMA = "timing_checkpoint_divergence_report_v1"
CHECKPOINT_DIVERGENCE_EXPERIMENT_SCHEMA = "timing_checkpoint_divergence_experiment_v1"


def selection_summary(
    selected: Sequence[Mapping[str, object]],
    *,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Summarize an evaluation selection without treating strata as complete."""
    counts_by_slice: dict[str, int] = {}
    for item in selected:
        selection_slice = str(item.get("selection_slice", "unassigned"))
        counts_by_slice[selection_slice] = counts_by_slice.get(selection_slice, 0) + 1

    summary: dict[str, object] = {
        "selected": len(selected),
        "counts_by_slice": dict(sorted(counts_by_slice.items())),
    }
    if metadata is not None:
        summary.update(copy.deepcopy(dict(metadata)))
    return summary


def initial_payload(
    *,
    args: object,
    selected: Sequence[Mapping[str, object]],
    checkpoints: Sequence[str],
    selection_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Create the immutable experiment header and empty resumable results."""
    normalized_checkpoints = _normalized_checkpoints(checkpoints)
    return {
        "schema": CHECKPOINT_DIVERGENCE_EXPERIMENT_SCHEMA,
        "experiment": {
            "inputs": {
                "index": _path_text(getattr(args, "index")),
                "dataset_root": _path_text(getattr(args, "dataset_root")),
            },
            "selection": {
                "mixed_maps": int(getattr(args, "mixed_maps")),
                "multi_bpm_maps": int(getattr(args, "multi_bpm_maps")),
                "max_duration_seconds": float(getattr(args, "max_duration_seconds")),
                "sample_seed": str(getattr(args, "sample_seed")),
            },
            "runtime": {
                "device": str(getattr(args, "device")),
                "float16": bool(getattr(args, "float16")),
            },
            "thresholds": {
                "divergence_phase_threshold_ms": float(
                    getattr(args, "divergence_phase_threshold_ms")
                ),
                "high_error_phase_threshold_ms": float(
                    getattr(args, "high_error_phase_threshold_ms")
                ),
            },
            "checkpoints": list(normalized_checkpoints),
        },
        "selection": [copy.deepcopy(dict(item)) for item in selected],
        "selection_summary": selection_summary(selected, metadata=selection_metadata),
        "results": {checkpoint: [] for checkpoint in normalized_checkpoints},
    }


def merge_existing_results(
    payload: Mapping[str, object],
    existing_payload: Mapping[str, object],
) -> dict[str, object]:
    """Resume only successful comparable rows from an identical experiment.

    Failed rows are deliberately omitted so a resumed run retries them.
    """
    mismatches = _resume_mismatches(payload, existing_payload)
    if mismatches:
        raise ValueError(f"existing checkpoint evaluation has mismatched fields: {', '.join(mismatches)}")

    merged = copy.deepcopy(dict(payload))
    current_results = _require_result_mapping(merged)
    existing_results = _require_result_mapping(existing_payload)
    for checkpoint in current_results:
        reusable: list[dict[str, object]] = []
        raw_rows = existing_results.get(checkpoint, [])
        if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
            raise ValueError(f"results.{checkpoint} must be a sequence")
        for raw_row in raw_rows:
            if not isinstance(raw_row, Mapping):
                raise ValueError(f"results.{checkpoint} rows must be objects")
            if raw_row.get("ok") is True:
                reusable.append(copy.deepcopy(dict(raw_row)))
        current_results[checkpoint] = reusable
    return merged


def compute_divergence_report(
    *,
    selected: Sequence[Mapping[str, object]],
    results_by_checkpoint: Mapping[str, Sequence[Mapping[str, object]]],
    checkpoints: Sequence[str],
    divergence_phase_threshold_ms: float = 8.0,
) -> dict[str, object]:
    """Compare checkpoints only when every requested checkpoint succeeded."""
    if divergence_phase_threshold_ms < 0.0:
        raise ValueError("divergence_phase_threshold_ms must be non-negative")
    normalized_checkpoints = _normalized_checkpoints(checkpoints)
    indexed_results = {
        checkpoint: _results_by_selection_key(results_by_checkpoint.get(checkpoint, ()))
        for checkpoint in normalized_checkpoints
    }

    complete_rows: list[dict[str, object]] = []
    failed_or_missing_keys: list[str] = []
    attempted_all_checkpoint_count = 0
    for item in selected:
        selection_key = _selection_key(item)
        rows = [indexed_results[checkpoint].get(selection_key) for checkpoint in normalized_checkpoints]
        if all(row is not None for row in rows):
            attempted_all_checkpoint_count += 1
        if any(row is None or row.get("ok") is not True for row in rows):
            failed_or_missing_keys.append(selection_key)
            continue

        assert all(row is not None for row in rows)
        phase_errors_ms = {
            checkpoint: _mean_phase_error_ms(row)
            for checkpoint, row in zip(normalized_checkpoints, rows)
        }
        phase_spread_ms = max(phase_errors_ms.values()) - min(phase_errors_ms.values())
        complete_rows.append(
            {
                "selection_key": selection_key,
                "mean_phase_error_ms_by_checkpoint": phase_errors_ms,
                "phase_spread_ms": phase_spread_ms,
                "material_phase_divergence": phase_spread_ms >= divergence_phase_threshold_ms,
            }
        )

    material_count = sum(bool(row["material_phase_divergence"]) for row in complete_rows)
    complete_count = len(complete_rows)
    return {
        "schema": CHECKPOINT_DIVERGENCE_REPORT_SCHEMA,
        "selected_map_count": len(selected),
        "attempted_all_checkpoint_count": attempted_all_checkpoint_count,
        "complete_map_count": complete_count,
        "failed_or_missing_map_count": len(failed_or_missing_keys),
        "material_phase_divergence_count": material_count,
        "material_phase_divergence_rate": material_count / complete_count if complete_count else None,
        "divergence_phase_threshold_ms": float(divergence_phase_threshold_ms),
        "failed_or_missing_selection_keys": failed_or_missing_keys,
        "complete_rows": complete_rows,
    }


def _resume_mismatches(
    payload: Mapping[str, object],
    existing_payload: Mapping[str, object],
) -> list[str]:
    comparable_fields = ("schema", "experiment", "selection")
    mismatches: list[str] = []
    for field in comparable_fields:
        _collect_mismatches(
            payload.get(field),
            existing_payload.get(field),
            path=field,
            mismatches=mismatches,
        )
    return mismatches


def _collect_mismatches(
    expected: object,
    actual: object,
    *,
    path: str,
    mismatches: list[str],
) -> None:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        keys = sorted(set(expected) | set(actual), key=str)
        for key in keys:
            _collect_mismatches(
                expected.get(key),
                actual.get(key),
                path=f"{path}.{key}",
                mismatches=mismatches,
            )
        return
    if expected != actual:
        mismatches.append(path)


def _require_result_mapping(payload: Mapping[str, object]) -> dict[str, object]:
    results = payload.get("results")
    if not isinstance(results, dict):
        raise ValueError("checkpoint evaluation results must be an object")
    return results


def _normalized_checkpoints(checkpoints: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(checkpoint) for checkpoint in checkpoints)
    if not normalized or any(not checkpoint for checkpoint in normalized):
        raise ValueError("checkpoints must contain non-empty names")
    if len(set(normalized)) != len(normalized):
        raise ValueError("checkpoints must be unique")
    return normalized


def _results_by_selection_key(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    indexed: dict[str, Mapping[str, object]] = {}
    for row in rows:
        key = _selection_key(row)
        if key in indexed:
            raise ValueError(f"duplicate checkpoint result selection_key: {key}")
        indexed[key] = row
    return indexed


def _selection_key(item: Mapping[str, object]) -> str:
    value = item.get("selection_key")
    if not isinstance(value, str) or not value:
        raise ValueError("selection rows must contain a non-empty selection_key")
    return value


def _mean_phase_error_ms(row: Mapping[str, object]) -> float:
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("successful checkpoint result is missing metrics")
    value = metrics.get("mean_phase_error_ms")
    if not isinstance(value, (int, float)):
        raise ValueError("successful checkpoint result is missing mean_phase_error_ms")
    return float(value)


def _path_text(value: object) -> str:
    return Path(value).as_posix() if isinstance(value, (str, Path)) else str(value)
