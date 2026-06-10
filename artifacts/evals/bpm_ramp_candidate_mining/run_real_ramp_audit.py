from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_bpm_ramp_candidate_mining import MONOTONIC_EPS_BPM, _linear_r2, _redline_arrays


DEFAULT_INPUT_PATH = Path(
    "artifacts/evals/bpm_ramp_candidate_mining/audio_representative_ramp_features_unique_bpm_gt5.parquet"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/evals/bpm_ramp_candidate_mining")
DEFAULT_UNIQUE_BPM_THRESHOLD = 5
MAX_CONTINUOUS_GAP_S = 20.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit unique-BPM>5 candidates for redline shapes that look like real BPM ramps.",
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--unique-bpm-threshold", type=int, default=DEFAULT_UNIQUE_BPM_THRESHOLD)
    parser.add_argument("--top-n", type=int, default=80)
    args = parser.parse_args()

    summary = audit_real_ramp_candidates(
        input_path=args.input_path,
        output_dir=args.output_dir,
        unique_bpm_threshold=args.unique_bpm_threshold,
        top_n=args.top_n,
    )
    print(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True))
    return 0


def audit_real_ramp_candidates(
    *,
    input_path: Path,
    output_dir: Path,
    unique_bpm_threshold: int,
    top_n: int,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_df = pd.read_parquet(input_path).reset_index(drop=True)
    required_columns = {
        "shard",
        "beatmap_set_id",
        "title",
        "artist",
        "version",
        "audio_path",
        "beatmap_path",
        "resolved_beatmap_path",
        "group_unique_bpm_count",
        "red_point_count",
    }
    missing = sorted(required_columns.difference(audio_df.columns))
    if missing:
        raise ValueError(f"{input_path} is missing required column(s): {missing}")

    records: list[dict[str, Any]] = []
    for row in audio_df.to_dict(orient="records"):
        redline = _redline_arrays(Path(row["resolved_beatmap_path"]))
        runs = _monotonic_runs(redline["offsets_ms"], redline["bpms"])
        best = max(runs, key=_run_rank_key) if runs else _empty_run_payload()
        classification = _classify_run(best)
        records.append(
            {
                **_metadata(row),
                "audit_status": classification["status"],
                "audit_status_rank": classification["status_rank"],
                "audit_reasons": classification["reasons"],
                "ramp_family": _ramp_family(best),
                "real_ramp_score": _real_ramp_score(best),
                "best_run_start_index": best["start_index"],
                "best_run_end_index": best["end_index"],
                "best_run_length": best["length"],
                "best_run_start_ms": best["start_ms"],
                "best_run_end_ms": best["end_ms"],
                "best_run_duration_s": best["duration_s"],
                "best_run_start_bpm": best["start_bpm"],
                "best_run_end_bpm": best["end_bpm"],
                "best_run_bpm_span": best["bpm_span"],
                "best_run_direction": best["direction"],
                "best_run_linear_r2": best["linear_r2"],
                "best_run_point_density_per_s": best["point_density_per_s"],
                "best_run_median_gap_s": best["median_gap_s"],
                "best_run_p90_gap_s": best["p90_gap_s"],
                "best_run_max_gap_s": best["max_gap_s"],
                "best_run_gap_cv": best["gap_cv"],
                "best_run_median_abs_delta_bpm": best["median_abs_delta_bpm"],
                "best_run_p90_abs_delta_bpm": best["p90_abs_delta_bpm"],
                "best_run_max_abs_delta_bpm": best["max_abs_delta_bpm"],
                "best_run_jumpiness": best["jumpiness"],
                "best_run_big_jump_count": best["big_jump_count"],
                "best_run_big_jump_fraction": best["big_jump_fraction"],
                "best_run_point_coverage": best["point_coverage"],
                "best_run_span_coverage": best["span_coverage"],
                "best_run_offsets_ms": _round_list(best["offsets_ms"], digits=3),
                "best_run_bpms": _round_list(best["bpms"], digits=6),
            }
        )

    audit_df = pd.DataFrame(records)
    audit_df = audit_df.sort_values(
        [
            "audit_status_rank",
            "real_ramp_score",
            "best_run_bpm_span",
            "best_run_length",
            "group_unique_bpm_count",
        ],
        ascending=[True, False, False, False, False],
    ).drop(columns=["audit_status_rank"])
    beatmapset_df = _aggregate_beatmapsets(audit_df)

    audio_path = output_dir / f"real_ramp_audio_audit_unique_bpm_gt{unique_bpm_threshold}.parquet"
    audio_csv_path = output_dir / f"real_ramp_audio_audit_unique_bpm_gt{unique_bpm_threshold}.csv"
    beatmapset_path = output_dir / f"real_ramp_beatmapset_audit_unique_bpm_gt{unique_bpm_threshold}.parquet"
    beatmapset_csv_path = output_dir / f"real_ramp_beatmapset_audit_unique_bpm_gt{unique_bpm_threshold}.csv"
    top_csv_path = output_dir / f"real_ramp_top_{top_n}_unique_bpm_gt{unique_bpm_threshold}.csv"
    top_compact_csv_path = output_dir / f"real_ramp_top_{top_n}_compact_unique_bpm_gt{unique_bpm_threshold}.csv"
    summary_path = output_dir / f"real_ramp_summary_unique_bpm_gt{unique_bpm_threshold}.json"

    audit_df.to_parquet(audio_path, index=False)
    audit_df.to_csv(audio_csv_path, index=False)
    beatmapset_df.to_parquet(beatmapset_path, index=False)
    beatmapset_df.to_csv(beatmapset_csv_path, index=False)
    beatmapset_df.head(top_n).to_csv(top_csv_path, index=False)
    _compact_top_view(beatmapset_df.head(top_n)).to_csv(top_compact_csv_path, index=False)

    status_counts = beatmapset_df["audit_status"].value_counts().sort_index().to_dict()
    family_counts = beatmapset_df["ramp_family"].value_counts().sort_index().to_dict()
    runengon_hits = beatmapset_df[
        beatmapset_df["beatmap_set_id"].astype(str).eq("971561")
        | beatmapset_df["title"].astype(str).str.contains("Runengon", case=False, na=False)
    ]
    runengon_rank = int(runengon_hits.index[0]) + 1 if not runengon_hits.empty else None

    summary = {
        "schema_version": 1,
        "method": (
            "second-pass audit over audio representatives from unique-red-BPM>threshold candidates; "
            "score best monotonic redline run by length, span, duration, temporal continuity, and smooth/stepped shape"
        ),
        "input_path": input_path.as_posix(),
        "unique_bpm_threshold": unique_bpm_threshold,
        "audio_candidate_count": int(audit_df.shape[0]),
        "beatmapset_candidate_count": int(beatmapset_df.shape[0]),
        "beatmapset_status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "beatmapset_family_counts": {str(key): int(value) for key, value in family_counts.items()},
        "runengon_rank": runengon_rank,
        "runengon_status": None if runengon_hits.empty else str(runengon_hits.iloc[0]["audit_status"]),
        "outputs": {
            "audio_audit": audio_path.as_posix(),
            "audio_audit_csv": audio_csv_path.as_posix(),
            "beatmapset_audit": beatmapset_path.as_posix(),
            "beatmapset_audit_csv": beatmapset_csv_path.as_posix(),
            "top_csv": top_csv_path.as_posix(),
            "top_compact_csv": top_compact_csv_path.as_posix(),
            "summary": summary_path.as_posix(),
        },
        "elapsed_s": time.perf_counter() - started_at,
    }
    summary_path.write_text(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "shard": row["shard"],
        "beatmap_set_id": row["beatmap_set_id"],
        "title": row.get("title"),
        "artist": row.get("artist"),
        "creator": row.get("creator"),
        "version": row.get("version"),
        "beatmap_id": row.get("beatmap_id"),
        "difficulty": row.get("difficulty"),
        "group_unique_bpm_count": row.get("group_unique_bpm_count"),
        "unique_bpm_count": row.get("unique_bpm_count"),
        "red_point_count": row.get("red_point_count"),
        "audio_path": row.get("audio_path"),
        "beatmap_path": row.get("beatmap_path"),
        "resolved_beatmap_path": row.get("resolved_beatmap_path"),
        "previous_ramp_score": row.get("ramp_score"),
    }


def _monotonic_runs(offsets_ms: np.ndarray, bpms: np.ndarray) -> list[dict[str, Any]]:
    if bpms.size == 0:
        return []
    if bpms.size == 1:
        return [_run_payload(offsets_ms, bpms, 0, 0)]

    runs: list[dict[str, Any]] = []
    for direction in (1.0, -1.0):
        start = 0
        for index in range(1, bpms.size):
            diff = float(bpms[index] - bpms[index - 1])
            if abs(diff) <= MONOTONIC_EPS_BPM or diff * direction > 0.0:
                continue
            runs.extend(_split_run_by_gap(offsets_ms, bpms, start, index - 1))
            start = index
        runs.extend(_split_run_by_gap(offsets_ms, bpms, start, bpms.size - 1))
    return runs


def _split_run_by_gap(offsets_ms: np.ndarray, bpms: np.ndarray, start: int, end: int) -> list[dict[str, Any]]:
    if end < start:
        return []
    split_starts = [start]
    for index in range(start + 1, end + 1):
        gap_s = float(offsets_ms[index] - offsets_ms[index - 1]) / 1000.0
        if gap_s > MAX_CONTINUOUS_GAP_S:
            split_starts.append(index)

    runs: list[dict[str, Any]] = []
    for run_start, next_start in zip(split_starts, split_starts[1:] + [end + 1]):
        run_end = next_start - 1
        if run_end >= run_start:
            runs.append(_run_payload(offsets_ms, bpms, run_start, run_end))
    return runs


def _run_payload(offsets_ms: np.ndarray, bpms: np.ndarray, start: int, end: int) -> dict[str, Any]:
    run_offsets = offsets_ms[start : end + 1]
    run_bpms = bpms[start : end + 1]
    gaps_s = np.diff(run_offsets) / 1000.0
    deltas = np.diff(run_bpms)
    abs_deltas = np.abs(deltas[np.abs(deltas) > MONOTONIC_EPS_BPM])

    duration_s = max(0.0, float(run_offsets[-1] - run_offsets[0]) / 1000.0)
    start_bpm = float(run_bpms[0])
    end_bpm = float(run_bpms[-1])
    bpm_delta = end_bpm - start_bpm
    span = abs(bpm_delta)
    direction = "up" if bpm_delta > MONOTONIC_EPS_BPM else "down" if bpm_delta < -MONOTONIC_EPS_BPM else "flat"
    red_point_count = int(bpms.size)
    whole_span = float(np.max(bpms) - np.min(bpms)) if bpms.size else 0.0
    big_jump_threshold = max(25.0, span * 0.20)
    big_jump_count = int(np.sum(abs_deltas > big_jump_threshold)) if abs_deltas.size else 0

    return {
        "start_index": int(start),
        "end_index": int(end),
        "length": int(run_bpms.size),
        "start_ms": float(run_offsets[0]),
        "end_ms": float(run_offsets[-1]),
        "duration_s": duration_s,
        "start_bpm": start_bpm,
        "end_bpm": end_bpm,
        "bpm_span": float(span),
        "direction": direction,
        "linear_r2": _linear_r2(run_offsets / 1000.0, run_bpms),
        "point_density_per_s": float((run_bpms.size - 1) / duration_s) if duration_s > 0.0 else 0.0,
        "median_gap_s": _percentile_or_zero(gaps_s, 50.0),
        "p90_gap_s": _percentile_or_zero(gaps_s, 90.0),
        "max_gap_s": float(np.max(gaps_s)) if gaps_s.size else 0.0,
        "gap_cv": _coefficient_of_variation(gaps_s),
        "median_abs_delta_bpm": _percentile_or_zero(abs_deltas, 50.0),
        "p90_abs_delta_bpm": _percentile_or_zero(abs_deltas, 90.0),
        "max_abs_delta_bpm": float(np.max(abs_deltas)) if abs_deltas.size else 0.0,
        "jumpiness": float(np.max(abs_deltas) / span) if abs_deltas.size and span > 0.0 else 0.0,
        "big_jump_count": big_jump_count,
        "big_jump_fraction": float(big_jump_count / abs_deltas.size) if abs_deltas.size else 0.0,
        "point_coverage": float(run_bpms.size / red_point_count) if red_point_count else 0.0,
        "span_coverage": float(span / whole_span) if whole_span > 0.0 else 0.0,
        "offsets_ms": run_offsets,
        "bpms": run_bpms,
    }


def _run_rank_key(run: dict[str, Any]) -> tuple[float, float, int, float]:
    return (
        _real_ramp_score(run),
        float(run["bpm_span"]),
        int(run["length"]),
        float(run["duration_s"]),
    )


def _real_ramp_score(run: dict[str, Any]) -> float:
    span = float(run["bpm_span"])
    length = int(run["length"])
    duration_s = float(run["duration_s"])
    if span <= 0.0 or length <= 1 or duration_s <= 0.0:
        return 0.0

    duration_factor = min(1.0, duration_s / 8.0)
    r2_factor = 0.40 + 0.60 * float(run["linear_r2"])
    p90_gap_s = float(run["p90_gap_s"])
    continuity_factor = 1.0 if p90_gap_s <= 15.0 else max(0.25, 15.0 / p90_gap_s)
    jumpiness = float(run["jumpiness"])
    jump_factor = 1.0 if jumpiness <= 0.25 else max(0.20, 1.0 - (jumpiness - 0.25) / 0.75)
    coverage_factor = math.sqrt(max(0.0, min(1.0, float(run["span_coverage"]))))
    return float(span * math.log1p(length) * duration_factor * r2_factor * continuity_factor * jump_factor * coverage_factor)


def _classify_run(run: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    length = int(run["length"])
    span = float(run["bpm_span"])
    duration_s = float(run["duration_s"])
    p90_gap_s = float(run["p90_gap_s"])
    density = float(run["point_density_per_s"])
    r2 = float(run["linear_r2"])
    jumpiness = float(run["jumpiness"])
    point_coverage = float(run["point_coverage"])

    if length < 6:
        reasons.append("too_few_points")
    if span < 60.0:
        reasons.append("low_bpm_span")
    if duration_s < 3.0:
        reasons.append("too_short_duration")
    if p90_gap_s > 30.0 and density < 0.05:
        reasons.append("temporally_disconnected")
    if jumpiness > 0.65 and r2 < 0.35:
        reasons.append("spike_like_jumps")
    if point_coverage < 0.05 and length < 10:
        reasons.append("tiny_run_inside_noisy_timing")

    smooth_enough = r2 >= 0.45 or jumpiness <= 0.25
    pass_shape = (
        length >= 8
        and span >= 80.0
        and duration_s >= 4.0
        and (p90_gap_s <= 20.0 or density >= 0.08)
        and smooth_enough
        and "spike_like_jumps" not in reasons
    )
    borderline_shape = (
        length >= 6
        and span >= 60.0
        and duration_s >= 3.0
        and "temporally_disconnected" not in reasons
        and "spike_like_jumps" not in reasons
    )

    if pass_shape:
        status = "pass"
        status_rank = 0
        if not reasons:
            reasons.append("long_continuous_monotonic_ramp")
    elif borderline_shape:
        status = "borderline"
        status_rank = 1
        if not reasons:
            reasons.append("ramp_like_but_weak_shape_or_spacing")
    else:
        status = "reject"
        status_rank = 2
        if not reasons:
            reasons.append("weak_ramp_shape")

    return {"status": status, "status_rank": status_rank, "reasons": reasons}


def _ramp_family(run: dict[str, Any]) -> str:
    length = int(run["length"])
    span = float(run["bpm_span"])
    r2 = float(run["linear_r2"])
    jumpiness = float(run["jumpiness"])
    if length < 6 or span < 60.0:
        return "short_or_low_span"
    if r2 >= 0.85:
        return "linear_ramp"
    if r2 >= 0.45:
        return "stepped_ramp"
    if jumpiness <= 0.25:
        return "curved_or_exponential_ramp"
    return "spiky_monotonic_run"


def _aggregate_beatmapsets(audit_df: pd.DataFrame) -> pd.DataFrame:
    if audit_df.empty:
        return audit_df.copy()

    with_rank = audit_df.copy()
    status_rank = {"pass": 0, "borderline": 1, "reject": 2}
    with_rank["audit_status_rank"] = with_rank["audit_status"].map(status_rank).fillna(3).astype(int)
    ranked = with_rank.sort_values(
        [
            "audit_status_rank",
            "real_ramp_score",
            "best_run_bpm_span",
            "best_run_length",
            "group_unique_bpm_count",
        ],
        ascending=[True, False, False, False, False],
    )
    best = ranked.drop_duplicates(["shard", "beatmap_set_id"], keep="first").copy()
    aggregates = (
        with_rank.groupby(["shard", "beatmap_set_id"], dropna=False)
        .agg(
            audio_representative_count=("audio_path", "nunique"),
            beatmapset_max_real_ramp_score=("real_ramp_score", "max"),
            beatmapset_max_best_run_span=("best_run_bpm_span", "max"),
            beatmapset_max_best_run_length=("best_run_length", "max"),
            beatmapset_max_best_run_duration_s=("best_run_duration_s", "max"),
        )
        .reset_index()
    )
    merged = best.merge(aggregates, on=["shard", "beatmap_set_id"], how="left", suffixes=("", "_aggregate"))
    return merged.sort_values(
        [
            "audit_status_rank",
            "beatmapset_max_real_ramp_score",
            "beatmapset_max_best_run_span",
            "beatmapset_max_best_run_length",
            "group_unique_bpm_count",
        ],
        ascending=[True, False, False, False, False],
    ).drop(columns=["audit_status_rank"]).reset_index(drop=True)


def _compact_top_view(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "beatmap_set_id",
        "title",
        "artist",
        "version",
        "audit_status",
        "ramp_family",
        "real_ramp_score",
        "best_run_length",
        "best_run_start_bpm",
        "best_run_end_bpm",
        "best_run_bpm_span",
        "best_run_duration_s",
        "best_run_linear_r2",
        "best_run_p90_gap_s",
        "best_run_max_gap_s",
        "best_run_jumpiness",
        "group_unique_bpm_count",
        "resolved_beatmap_path",
        "audit_reasons",
    ]
    return df[[column for column in columns if column in df.columns]].copy()


def _empty_run_payload() -> dict[str, Any]:
    return {
        "start_index": -1,
        "end_index": -1,
        "length": 0,
        "start_ms": math.nan,
        "end_ms": math.nan,
        "duration_s": 0.0,
        "start_bpm": math.nan,
        "end_bpm": math.nan,
        "bpm_span": 0.0,
        "direction": "none",
        "linear_r2": 0.0,
        "point_density_per_s": 0.0,
        "median_gap_s": 0.0,
        "p90_gap_s": 0.0,
        "max_gap_s": 0.0,
        "gap_cv": 0.0,
        "median_abs_delta_bpm": 0.0,
        "p90_abs_delta_bpm": 0.0,
        "max_abs_delta_bpm": 0.0,
        "jumpiness": 0.0,
        "big_jump_count": 0,
        "big_jump_fraction": 0.0,
        "point_coverage": 0.0,
        "span_coverage": 0.0,
        "offsets_ms": np.asarray([], dtype=np.float64),
        "bpms": np.asarray([], dtype=np.float64),
    }


def _percentile_or_zero(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else 0.0


def _coefficient_of_variation(values: np.ndarray) -> float:
    if not values.size:
        return 0.0
    mean = float(np.mean(values))
    return float(np.std(values) / mean) if mean > 0.0 else 0.0


def _round_list(values: np.ndarray, *, digits: int) -> list[float]:
    return [round(float(value), digits) for value in values]


if __name__ == "__main__":
    raise SystemExit(main())
