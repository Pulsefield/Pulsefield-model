from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from pulsefield_model.timing.canonicalization import TIMING_CANONICALIZATION_NONE
from pulsefield_model.timing.providers.oracle import oracle_timing_grid_from_beatmap
from pulsefield_model.timing.ramp_detection import detect_timing_ramp


DEFAULT_POSITIVES = Path(
    "artifacts/evals/bpm_ramp_candidate_mining/real_ramp_beatmapset_audit_unique_bpm_gt5.parquet"
)
DEFAULT_CANDIDATES = Path(
    "artifacts/evals/bpm_ramp_candidate_mining/candidate_index_unique_bpm_gt5.parquet"
)
DEFAULT_NEGATIVES = Path(
    "artifacts/indexes/beatmap_index_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.parquet"
)
DEFAULT_DATASET_ROOT = Path("dataset")
DEFAULT_OUTPUT = Path("artifacts/evals/bpm_ramp_timing_detection/results.json")
DEFAULT_CSV = Path("artifacts/evals/bpm_ramp_timing_detection/results.csv")
DEFAULT_RESULT_LOG = Path("artifacts/evals/bpm_ramp_timing_detection/result_log.md")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate cheap BPM-ramp recognition over timing grids.")
    parser.add_argument("--positives", type=Path, default=DEFAULT_POSITIVES)
    parser.add_argument("--candidate-index", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--negatives", type=Path, default=DEFAULT_NEGATIVES)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--negative-count", type=int, default=500)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--result-log", type=Path, default=DEFAULT_RESULT_LOG)
    args = parser.parse_args(argv)

    if args.negative_count <= 0:
        raise ValueError(f"negative-count must be positive, got {args.negative_count!r}")

    started_at = time.perf_counter()
    positive_rows = _positive_rows(args.positives)
    negative_rows = _negative_rows(
        args.negatives,
        positives=positive_rows,
        candidate_index=args.candidate_index,
        dataset_root=args.dataset_root,
        count=args.negative_count,
    )

    results = [
        *[
            _evaluate_row(
                row,
                label="true_ramp_pass",
                expected_ramp=True,
                beatmap_path=Path(row["resolved_beatmap_path"]),
            )
            for row in positive_rows
        ],
        *[
            _evaluate_row(
                row,
                label="non_ramp_guard",
                expected_ramp=False,
                beatmap_path=Path(row["resolved_beatmap_path"]),
            )
            for row in negative_rows
        ],
    ]
    payload = _payload(args, results, started_at=started_at)
    _write_json(args.output, payload)
    _write_csv(args.csv, results)
    _write_text(args.result_log, _format_result_log(payload))
    print(json.dumps(payload["summary"], allow_nan=False, indent=2, sort_keys=True))
    return 0


def _positive_rows(path: Path) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    rows = df[df["audit_status"].eq("pass")].copy()
    rows = rows.sort_values(["beatmap_set_id", "resolved_beatmap_path"]).reset_index(drop=True)
    return rows.to_dict(orient="records")


def _negative_rows(
    path: Path,
    *,
    positives: Sequence[dict[str, Any]],
    candidate_index: Path,
    dataset_root: Path,
    count: int,
) -> list[dict[str, Any]]:
    negative_df = pd.read_parquet(path)
    candidate_df = pd.read_parquet(candidate_index)
    excluded_sets = {
        *candidate_df["beatmap_set_id"].astype(str).tolist(),
        *(str(row["beatmap_set_id"]) for row in positives),
    }
    selected = negative_df[~negative_df["beatmap_set_id"].astype(str).isin(excluded_sets)].copy()
    selected = selected.sort_values(["shard", "audio_path", "difficulty", "beatmap_id"], na_position="last")
    selected = selected.drop_duplicates(["shard", "audio_path"], keep="first").head(count).copy()
    if selected.shape[0] < count:
        raise RuntimeError(f"negative pool only produced {selected.shape[0]} row(s), expected {count}")
    selected["resolved_beatmap_path"] = selected.apply(
        lambda row: (dataset_root / str(row["shard"]) / str(row["beatmap_path"])).as_posix(),
        axis=1,
    )
    return selected.to_dict(orient="records")


def _evaluate_row(
    row: dict[str, Any],
    *,
    label: str,
    expected_ramp: bool,
    beatmap_path: Path,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    try:
        parse_started_at = time.perf_counter()
        grid = oracle_timing_grid_from_beatmap(beatmap_path, canonicalization=TIMING_CANONICALIZATION_NONE)
        parse_seconds = time.perf_counter() - parse_started_at
        detect_started_at = time.perf_counter()
        detection = detect_timing_ramp(grid)
        detect_seconds = time.perf_counter() - detect_started_at
        return {
            **_identity(row),
            "label": label,
            "expected_ramp": expected_ramp,
            "ok": True,
            "error": None,
            "segment_count": len(grid.segments),
            "is_ramp": bool(detection.is_ramp),
            "correct": bool(detection.is_ramp) == expected_ramp,
            "parse_seconds": parse_seconds,
            "detect_seconds": detect_seconds,
            "total_seconds": time.perf_counter() - started_at,
            "detection": asdict(detection),
        }
    except Exception as exc:
        return {
            **_identity(row),
            "label": label,
            "expected_ramp": expected_ramp,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "segment_count": None,
            "is_ramp": False,
            "correct": False,
            "parse_seconds": 0.0,
            "detect_seconds": 0.0,
            "total_seconds": time.perf_counter() - started_at,
            "detection": None,
        }


def _identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "shard": _optional(row.get("shard")),
        "beatmap_set_id": _optional(row.get("beatmap_set_id")),
        "beatmap_id": _optional(row.get("beatmap_id")),
        "artist": _optional(row.get("artist")),
        "title": _optional(row.get("title")),
        "creator": _optional(row.get("creator")),
        "version": _optional(row.get("version")),
        "audit_status": _optional(row.get("audit_status")),
        "ramp_family": _optional(row.get("ramp_family")),
        "beatmap_path": _optional(row.get("beatmap_path")),
        "resolved_beatmap_path": _optional(row.get("resolved_beatmap_path")),
    }


def _payload(args: argparse.Namespace, results: Sequence[dict[str, Any]], *, started_at: float) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": {
            "name": "bpm_ramp_timing_detection",
            "positives": args.positives.as_posix(),
            "candidate_index": args.candidate_index.as_posix(),
            "negatives": args.negatives.as_posix(),
            "dataset_root": args.dataset_root.as_posix(),
            "negative_count": args.negative_count,
            "target_recall": 0.85,
            "target_false_positives": 0,
            "target_mean_seconds": 0.2,
        },
        "elapsed_seconds": time.perf_counter() - started_at,
        "summary": _summary(results),
        "results": list(results),
    }


def _summary(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in results if row["ok"]]
    failed = [row for row in results if not row["ok"]]
    positives = [row for row in ok if row["expected_ramp"]]
    negatives = [row for row in ok if not row["expected_ramp"]]
    true_positive_count = sum(1 for row in positives if row["is_ramp"])
    false_negative_count = len(positives) - true_positive_count
    false_positive_rows = [row for row in negatives if row["is_ramp"]]
    true_negative_count = len(negatives) - len(false_positive_rows)
    recall = _safe_ratio(true_positive_count, len(positives))
    false_positive_rate = _safe_ratio(len(false_positive_rows), len(negatives))
    timing = _timing_summary(ok)
    positive_signal = (
        recall is not None
        and recall >= 0.85
        and len(false_positive_rows) == 0
        and timing["total_seconds"]["mean"] < 0.2
    )
    return {
        "attempted": len(results),
        "ok": len(ok),
        "failed": len(failed),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "true_positive_count": true_positive_count,
        "false_negative_count": false_negative_count,
        "true_negative_count": true_negative_count,
        "false_positive_count": len(false_positive_rows),
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "timing": timing,
        "positive_signal": positive_signal,
        "false_negative_examples": _brief_cases([row for row in positives if not row["is_ramp"]], limit=20),
        "false_positive_examples": _brief_cases(false_positive_rows, limit=20),
        "failure_examples": _brief_cases(failed, limit=10),
        "recognized_families": _value_counts(
            row["detection"]["family"]
            for row in ok
            if row["is_ramp"] and row.get("detection")
        ),
    }


def _timing_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "parse_seconds": _distribution(row["parse_seconds"] for row in rows),
        "detect_seconds": _distribution(row["detect_seconds"] for row in rows),
        "total_seconds": _distribution(row["total_seconds"] for row in rows),
    }


def _distribution(values: Sequence[float] | Any) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _brief_cases(rows: Sequence[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    cases = []
    for row in rows[:limit]:
        detection = row.get("detection") or {}
        best_run = detection.get("best_run") or {}
        cases.append(
            {
                "beatmap_set_id": row.get("beatmap_set_id"),
                "beatmap_id": row.get("beatmap_id"),
                "artist": row.get("artist"),
                "title": row.get("title"),
                "version": row.get("version"),
                "label": row.get("label"),
                "audit_status": row.get("audit_status"),
                "family": detection.get("family"),
                "reasons": detection.get("reasons"),
                "best_run": {
                    "length": best_run.get("length"),
                    "bpm_span": best_run.get("bpm_span"),
                    "duration_s": best_run.get("duration_s"),
                    "linear_r2": best_run.get("linear_r2"),
                    "p90_gap_s": best_run.get("p90_gap_s"),
                    "jumpiness": best_run.get("jumpiness"),
                },
                "resolved_beatmap_path": row.get("resolved_beatmap_path"),
                "error": row.get("error"),
            }
        )
    return cases


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _write_csv(path: Path, results: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "label",
        "expected_ramp",
        "ok",
        "correct",
        "is_ramp",
        "beatmap_set_id",
        "beatmap_id",
        "artist",
        "title",
        "creator",
        "version",
        "audit_status",
        "ramp_family",
        "segment_count",
        "detected_family",
        "score",
        "reasons_json",
        "best_run_length",
        "best_run_bpm_span",
        "best_run_duration_s",
        "best_run_linear_r2",
        "best_run_p90_gap_s",
        "best_run_jumpiness",
        "parse_seconds",
        "detect_seconds",
        "total_seconds",
        "resolved_beatmap_path",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in results:
            writer.writerow(_csv_row(row, fieldnames))
    tmp_path.replace(path)


def _csv_row(row: dict[str, Any], fieldnames: Sequence[str]) -> dict[str, Any]:
    detection = row.get("detection") or {}
    best_run = detection.get("best_run") or {}
    payload = {
        **row,
        "detected_family": detection.get("family"),
        "score": detection.get("score"),
        "reasons_json": json.dumps(detection.get("reasons") or []),
        "best_run_length": best_run.get("length"),
        "best_run_bpm_span": best_run.get("bpm_span"),
        "best_run_duration_s": best_run.get("duration_s"),
        "best_run_linear_r2": best_run.get("linear_r2"),
        "best_run_p90_gap_s": best_run.get("p90_gap_s"),
        "best_run_jumpiness": best_run.get("jumpiness"),
    }
    return {field: payload.get(field, "") for field in fieldnames}


def _format_result_log(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    timing = summary["timing"]["total_seconds"]
    return "\n".join(
        [
            "# BPM Ramp Timing Detection Result",
            "",
            "## Status",
            "",
            f"- Attempted rows: {summary['attempted']}",
            f"- Successful rows: {summary['ok']}",
            f"- Failed rows: {summary['failed']}",
            f"- Positives: {summary['positive_count']}",
            f"- Negatives: {summary['negative_count']}",
            f"- Recall: {_percent(summary['recall'])} ({summary['true_positive_count']}/{summary['positive_count']})",
            f"- False positives: {summary['false_positive_count']}/{summary['negative_count']}",
            f"- False-positive rate: {_percent(summary['false_positive_rate'])}",
            f"- Mean total seconds/map: {timing['mean']:.6f}",
            f"- P95 total seconds/map: {timing['p95']:.6f}",
            f"- Max total seconds/map: {timing['max']:.6f}",
            f"- Positive signal observed: `{summary['positive_signal']}`",
            "",
            "## Interpretation",
            "",
            "- The eval measures cheap structural ramp recognition over timing grids parsed from `.osu` red timing.",
            "- It does not run BeatThis audio inference and does not prove audio-ground-truth novelty.",
            "- Borderline ramp-audit rows are intentionally excluded from the primary recall metric.",
            "",
            "## False Negatives",
            "",
            json.dumps(summary["false_negative_examples"], indent=2, sort_keys=True),
            "",
            "## False Positives",
            "",
            json.dumps(summary["false_positive_examples"], indent=2, sort_keys=True),
            "",
        ]
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100.0:.1f}%"


def _value_counts(values: Sequence[str] | Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _optional(value: object) -> object:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
