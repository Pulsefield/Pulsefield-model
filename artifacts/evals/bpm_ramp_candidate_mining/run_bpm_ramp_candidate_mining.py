from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from pulsefield_model.osu_core.timing import InvalidRedTimingError, MissingRedTimingError, require_red_timing_points


DEFAULT_SOURCE_INDEX_PATH = Path("artifacts/indexes/beatmap_index_4k_no_timing_anomalies_2to6.parquet")
DEFAULT_DATASET_ROOT = Path("dataset")
DEFAULT_OUTPUT_DIR = Path("artifacts/evals/bpm_ramp_candidate_mining")
DEFAULT_UNIQUE_BPM_THRESHOLD = 5
DEFAULT_BPM_ROUND_DECIMALS = 6
MONOTONIC_EPS_BPM = 1e-6


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Mine likely BPM-ramp candidates from the inverse of a per-beatmapset "
            "unique-red-BPM threshold filter."
        ),
    )
    parser.add_argument("--source-index-path", type=Path, default=DEFAULT_SOURCE_INDEX_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--unique-bpm-threshold", type=int, default=DEFAULT_UNIQUE_BPM_THRESHOLD)
    parser.add_argument("--bpm-round-decimals", type=int, default=DEFAULT_BPM_ROUND_DECIMALS)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    result = mine_bpm_ramp_candidates(
        source_index_path=args.source_index_path,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        unique_bpm_threshold=args.unique_bpm_threshold,
        bpm_round_decimals=args.bpm_round_decimals,
        progress_every=args.progress_every,
    )
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


def mine_bpm_ramp_candidates(
    *,
    source_index_path: Path,
    dataset_root: Path,
    output_dir: Path,
    unique_bpm_threshold: int,
    bpm_round_decimals: int,
    progress_every: int,
) -> dict[str, Any]:
    if unique_bpm_threshold < 1:
        raise ValueError(f"unique_bpm_threshold must be positive, got {unique_bpm_threshold!r}")
    if bpm_round_decimals < 0:
        raise ValueError(f"bpm_round_decimals must be non-negative, got {bpm_round_decimals!r}")

    started_at = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_df = pd.read_parquet(source_index_path).reset_index(drop=True)
    required_columns = {"shard", "beatmap_set_id", "beatmap_path", "audio_path"}
    missing_columns = sorted(required_columns.difference(source_df.columns))
    if missing_columns:
        raise ValueError(f"{source_index_path} is missing required column(s): {missing_columns}")

    row_records: list[dict[str, Any]] = []
    group_unique_bpms: dict[tuple[str, str], set[float]] = {}
    group_source_counts: dict[tuple[str, str], int] = {}
    parse_error_counts: dict[str, int] = {}

    for row_index, row in enumerate(source_df.itertuples(index=False), start=1):
        row_dict = row._asdict()
        group_key = _group_key(row_dict)
        group_source_counts[group_key] = group_source_counts.get(group_key, 0) + 1
        beatmap_path = _resolve_dataset_path(dataset_root, row_dict["shard"], row_dict["beatmap_path"])
        try:
            redline = _redline_arrays(beatmap_path)
            unique_bpms = _rounded_unique(redline["bpms"], decimals=bpm_round_decimals)
            parse_status = "ok"
            parse_error = None
        except (MissingRedTimingError, InvalidRedTimingError, OSError, ValueError) as exc:
            redline = _empty_redline()
            unique_bpms = []
            parse_status = type(exc).__name__
            parse_error = str(exc)
            parse_error_counts[parse_status] = parse_error_counts.get(parse_status, 0) + 1

        group_unique_bpms.setdefault(group_key, set()).update(unique_bpms)
        row_records.append(
            {
                "source_row_index": row_index - 1,
                "shard": str(row_dict["shard"]),
                "beatmap_set_id": _json_scalar(row_dict["beatmap_set_id"]),
                "beatmap_path": str(row_dict["beatmap_path"]),
                "audio_path": str(row_dict["audio_path"]),
                "title": _optional_string(row_dict.get("title")),
                "artist": _optional_string(row_dict.get("artist")),
                "creator": _optional_string(row_dict.get("creator")),
                "version": _optional_string(row_dict.get("version")),
                "beatmap_id": _json_scalar(row_dict.get("beatmap_id")),
                "difficulty": _optional_float(row_dict.get("difficulty")),
                "parse_status": parse_status,
                "parse_error": parse_error,
                "red_point_count": int(redline["offsets_ms"].size),
                "row_unique_bpm_count": len(unique_bpms),
                "row_unique_bpms_head": unique_bpms[:16],
            }
        )
        if progress_every > 0 and row_index % progress_every == 0:
            print(f"parsed {row_index}/{len(source_df)} beatmaps", file=sys.stderr, flush=True)

    row_df = pd.DataFrame(row_records)
    group_count_records = [
        {
            "shard": shard,
            "beatmap_set_id": _parse_numeric_string(beatmap_set_id),
            "group_source_map_count": group_source_counts[(shard, beatmap_set_id)],
            "group_unique_bpm_count": len(values),
            "group_unique_bpms_head": sorted(values)[:24],
        }
        for shard, beatmap_set_id in sorted(group_unique_bpms)
        for values in [group_unique_bpms[(shard, beatmap_set_id)]]
    ]
    group_count_df = pd.DataFrame(group_count_records)

    candidate_group_df = group_count_df[group_count_df["group_unique_bpm_count"] > unique_bpm_threshold].copy()
    source_with_counts_df = source_df.merge(
        candidate_group_df[["shard", "beatmap_set_id", "group_unique_bpm_count", "group_unique_bpms_head"]],
        on=["shard", "beatmap_set_id"],
        how="inner",
    )
    candidate_index_path = output_dir / f"candidate_index_unique_bpm_gt{unique_bpm_threshold}.parquet"
    source_with_counts_df.to_parquet(candidate_index_path, index=False)

    candidate_row_df = row_df.merge(
        candidate_group_df[["shard", "beatmap_set_id", "group_source_map_count", "group_unique_bpm_count"]],
        on=["shard", "beatmap_set_id"],
        how="inner",
    )
    audio_rep_df = _select_audio_representatives(candidate_row_df)
    audio_feature_df = _feature_audio_representatives(
        audio_rep_df,
        dataset_root=dataset_root,
        bpm_round_decimals=bpm_round_decimals,
    )
    beatmapset_feature_df = _aggregate_beatmapsets(audio_feature_df)

    audio_features_path = output_dir / f"audio_representative_ramp_features_unique_bpm_gt{unique_bpm_threshold}.parquet"
    beatmapset_features_path = output_dir / f"beatmapset_ramp_features_unique_bpm_gt{unique_bpm_threshold}.parquet"
    top_csv_path = output_dir / f"top_50_beatmapset_ramp_candidates_unique_bpm_gt{unique_bpm_threshold}.csv"
    audio_csv_path = output_dir / f"audio_representative_ramp_features_unique_bpm_gt{unique_bpm_threshold}.csv"
    beatmapset_csv_path = output_dir / f"beatmapset_ramp_features_unique_bpm_gt{unique_bpm_threshold}.csv"
    summary_path = output_dir / f"summary_unique_bpm_gt{unique_bpm_threshold}.json"

    audio_feature_df.to_parquet(audio_features_path, index=False)
    beatmapset_feature_df.to_parquet(beatmapset_features_path, index=False)
    audio_feature_df.to_csv(audio_csv_path, index=False)
    beatmapset_feature_df.to_csv(beatmapset_csv_path, index=False)
    beatmapset_feature_df.head(50).to_csv(top_csv_path, index=False)

    runengon_hits = beatmapset_feature_df[
        beatmapset_feature_df["beatmap_set_id"].astype(str).eq("971561")
        | beatmapset_feature_df["title"].astype(str).str.contains("Runengon", case=False, na=False)
    ]
    runengon_rank = None
    if not runengon_hits.empty:
        runengon_rank = int(runengon_hits.index[0]) + 1

    summary = {
        "schema_version": 1,
        "method": (
            "source index minus hypothetical per-beatmapset unique-red-BPM <= threshold index; "
            "one representative beatmap per audio; ramp features aggregated per beatmapset"
        ),
        "source_index_path": source_index_path.as_posix(),
        "dataset_root": dataset_root.as_posix(),
        "unique_bpm_threshold": unique_bpm_threshold,
        "bpm_round_decimals": bpm_round_decimals,
        "source_map_count": int(len(source_df)),
        "source_beatmapset_count": int(group_count_df.shape[0]),
        "candidate_map_count": int(source_with_counts_df.shape[0]),
        "candidate_beatmapset_count": int(candidate_group_df.shape[0]),
        "audio_representative_count": int(audio_feature_df.shape[0]),
        "beatmapset_feature_count": int(beatmapset_feature_df.shape[0]),
        "parse_error_counts": parse_error_counts,
        "runengon_rank": runengon_rank,
        "outputs": {
            "candidate_index": candidate_index_path.as_posix(),
            "audio_representative_features": audio_features_path.as_posix(),
            "audio_representative_features_csv": audio_csv_path.as_posix(),
            "beatmapset_features": beatmapset_features_path.as_posix(),
            "beatmapset_features_csv": beatmapset_csv_path.as_posix(),
            "top_50_csv": top_csv_path.as_posix(),
            "summary": summary_path.as_posix(),
        },
        "elapsed_s": time.perf_counter() - started_at,
    }
    summary_path.write_text(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _select_audio_representatives(candidate_row_df: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        "shard",
        "audio_path",
        "row_unique_bpm_count",
        "red_point_count",
        "difficulty",
        "beatmap_id",
        "beatmap_path",
    ]
    ascending = [True, True, False, False, False, True, True]
    sorted_df = candidate_row_df.sort_values(sort_columns, ascending=ascending, na_position="last")
    return sorted_df.drop_duplicates(["shard", "audio_path"], keep="first").reset_index(drop=True)


def _feature_audio_representatives(
    audio_rep_df: pd.DataFrame,
    *,
    dataset_root: Path,
    bpm_round_decimals: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in audio_rep_df.to_dict(orient="records"):
        beatmap_path = _resolve_dataset_path(dataset_root, row["shard"], row["beatmap_path"])
        redline = _redline_arrays(beatmap_path)
        features = _ramp_features(redline["offsets_ms"], redline["bpms"])
        unique_bpms = _rounded_unique(redline["bpms"], decimals=bpm_round_decimals)
        records.append(
            {
                **{key: row.get(key) for key in _AUDIO_REP_METADATA_COLUMNS},
                "resolved_beatmap_path": beatmap_path.as_posix(),
                "red_offsets_ms_head": _round_list(redline["offsets_ms"][:24], digits=3),
                "red_bpms_head": _round_list(redline["bpms"][:24], digits=6),
                "unique_bpms_head": unique_bpms[:24],
                "unique_bpm_count": len(unique_bpms),
                **features,
            }
        )
    feature_df = pd.DataFrame(records)
    return feature_df.sort_values(
        ["ramp_score", "max_monotonic_bpm_span", "max_monotonic_run_length", "unique_bpm_count"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


_AUDIO_REP_METADATA_COLUMNS = (
    "shard",
    "beatmap_set_id",
    "group_source_map_count",
    "group_unique_bpm_count",
    "audio_path",
    "beatmap_path",
    "title",
    "artist",
    "creator",
    "version",
    "beatmap_id",
    "difficulty",
)


def _aggregate_beatmapsets(audio_feature_df: pd.DataFrame) -> pd.DataFrame:
    if audio_feature_df.empty:
        return audio_feature_df.copy()

    ranked = audio_feature_df.sort_values(
        ["ramp_score", "max_monotonic_bpm_span", "max_monotonic_run_length", "unique_bpm_count"],
        ascending=[False, False, False, False],
    )
    best = ranked.drop_duplicates(["shard", "beatmap_set_id"], keep="first").copy()
    aggregates = (
        audio_feature_df.groupby(["shard", "beatmap_set_id"], dropna=False)
        .agg(
            audio_representative_count=("audio_path", "nunique"),
            beatmapset_max_unique_bpm_count=("unique_bpm_count", "max"),
            beatmapset_max_red_point_count=("red_point_count", "max"),
            beatmapset_max_bpm_span=("bpm_span", "max"),
            beatmapset_max_monotonic_run_length=("max_monotonic_run_length", "max"),
            beatmapset_max_monotonic_bpm_span=("max_monotonic_bpm_span", "max"),
            beatmapset_max_ramp_score=("ramp_score", "max"),
        )
        .reset_index()
    )
    merged = best.merge(aggregates, on=["shard", "beatmap_set_id"], how="left", suffixes=("", "_aggregate"))
    return merged.sort_values(
        [
            "beatmapset_max_ramp_score",
            "beatmapset_max_monotonic_bpm_span",
            "beatmapset_max_monotonic_run_length",
            "beatmapset_max_unique_bpm_count",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def _ramp_features(offsets_ms: np.ndarray, bpms: np.ndarray) -> dict[str, Any]:
    red_point_count = int(bpms.size)
    if red_point_count == 0:
        return _empty_ramp_features()

    bpm_min = float(np.min(bpms))
    bpm_max = float(np.max(bpms))
    bpm_span = bpm_max - bpm_min
    diffs = np.diff(bpms)
    nonzero_diffs = diffs[np.abs(diffs) > MONOTONIC_EPS_BPM]
    positive_count = int(np.sum(nonzero_diffs > 0.0))
    negative_count = int(np.sum(nonzero_diffs < 0.0))
    nonzero_diff_count = int(nonzero_diffs.size)
    same_direction_fraction = (
        float(max(positive_count, negative_count) / nonzero_diff_count) if nonzero_diff_count else 0.0
    )
    direction_change_count = _direction_change_count(nonzero_diffs)
    best_run = _best_monotonic_run(offsets_ms, bpms)
    ramp_score = _ramp_score(best_run)

    return {
        "red_point_count": red_point_count,
        "bpm_min": bpm_min,
        "bpm_max": bpm_max,
        "bpm_span": float(bpm_span),
        "nonzero_bpm_diff_count": nonzero_diff_count,
        "positive_bpm_diff_count": positive_count,
        "negative_bpm_diff_count": negative_count,
        "same_direction_fraction": same_direction_fraction,
        "direction_change_count": direction_change_count,
        "max_monotonic_run_length": best_run["length"],
        "max_monotonic_run_start_ms": best_run["start_ms"],
        "max_monotonic_run_end_ms": best_run["end_ms"],
        "max_monotonic_run_duration_s": best_run["duration_s"],
        "max_monotonic_start_bpm": best_run["start_bpm"],
        "max_monotonic_end_bpm": best_run["end_bpm"],
        "max_monotonic_bpm_span": best_run["bpm_span"],
        "max_monotonic_abs_slope_bpm_per_s": best_run["abs_slope_bpm_per_s"],
        "max_monotonic_direction": best_run["direction"],
        "max_monotonic_linear_r2": best_run["linear_r2"],
        "ramp_score": ramp_score,
    }


def _best_monotonic_run(offsets_ms: np.ndarray, bpms: np.ndarray) -> dict[str, Any]:
    if bpms.size == 0:
        return _empty_monotonic_run()
    if bpms.size == 1:
        return _monotonic_run_payload(offsets_ms, bpms, 0, 0)

    best = _monotonic_run_payload(offsets_ms, bpms, 0, 0)
    for direction in (1.0, -1.0):
        start = 0
        for index in range(1, bpms.size):
            diff = float(bpms[index] - bpms[index - 1])
            if abs(diff) <= MONOTONIC_EPS_BPM or diff * direction > 0.0:
                continue
            best = _choose_better_run(best, _monotonic_run_payload(offsets_ms, bpms, start, index - 1))
            start = index - 1
        best = _choose_better_run(best, _monotonic_run_payload(offsets_ms, bpms, start, bpms.size - 1))
    return best


def _choose_better_run(current: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    current_key = (
        current["bpm_span"],
        current["length"],
        current["duration_s"],
        current["linear_r2"],
    )
    candidate_key = (
        candidate["bpm_span"],
        candidate["length"],
        candidate["duration_s"],
        candidate["linear_r2"],
    )
    return candidate if candidate_key > current_key else current


def _monotonic_run_payload(offsets_ms: np.ndarray, bpms: np.ndarray, start: int, end: int) -> dict[str, Any]:
    run_offsets = offsets_ms[start : end + 1]
    run_bpms = bpms[start : end + 1]
    start_ms = float(run_offsets[0])
    end_ms = float(run_offsets[-1])
    start_bpm = float(run_bpms[0])
    end_bpm = float(run_bpms[-1])
    bpm_delta = end_bpm - start_bpm
    bpm_span = abs(bpm_delta)
    duration_s = max(0.0, (end_ms - start_ms) / 1000.0)
    direction = "up" if bpm_delta > MONOTONIC_EPS_BPM else "down" if bpm_delta < -MONOTONIC_EPS_BPM else "flat"
    abs_slope = float(bpm_span / duration_s) if duration_s > 0.0 else 0.0
    return {
        "length": int(run_bpms.size),
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_s": duration_s,
        "start_bpm": start_bpm,
        "end_bpm": end_bpm,
        "bpm_span": float(bpm_span),
        "abs_slope_bpm_per_s": abs_slope,
        "direction": direction,
        "linear_r2": _linear_r2(run_offsets / 1000.0, run_bpms),
    }


def _linear_r2(times_s: np.ndarray, bpms: np.ndarray) -> float:
    if times_s.size < 2:
        return 0.0
    if float(np.max(times_s) - np.min(times_s)) <= 0.0:
        return 0.0
    if float(np.max(bpms) - np.min(bpms)) <= MONOTONIC_EPS_BPM:
        return 0.0
    slope, intercept = np.polyfit(times_s, bpms, 1)
    predicted = slope * times_s + intercept
    residual = float(np.sum((bpms - predicted) ** 2))
    total = float(np.sum((bpms - float(np.mean(bpms))) ** 2))
    if total <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - residual / total)))


def _ramp_score(run: dict[str, Any]) -> float:
    span = float(run["bpm_span"])
    length = int(run["length"])
    duration_s = float(run["duration_s"])
    r2 = float(run["linear_r2"])
    duration_factor = min(1.0, duration_s / 5.0)
    return float(span * math.log1p(length) * duration_factor * max(0.05, r2))


def _direction_change_count(nonzero_diffs: np.ndarray) -> int:
    if nonzero_diffs.size < 2:
        return 0
    signs = np.sign(nonzero_diffs)
    return int(np.sum(signs[1:] != signs[:-1]))


def _redline_arrays(beatmap_path: Path) -> dict[str, np.ndarray]:
    # Match oracle provider semantics: duplicate offsets collapse to the last redline.
    by_offset: dict[float, float] = {}
    for point in require_red_timing_points(beatmap_path):
        by_offset[float(point.offset_ms)] = 60000.0 / float(point.beat_length_ms)
    offsets = np.asarray(sorted(by_offset), dtype=np.float64)
    bpms = np.asarray([by_offset[offset] for offset in offsets], dtype=np.float64)
    return {"offsets_ms": offsets, "bpms": bpms}


def _rounded_unique(values: np.ndarray, *, decimals: int) -> list[float]:
    return sorted({round(float(value), decimals) for value in values})


def _resolve_dataset_path(dataset_root: Path, shard: object, relative_path: object) -> Path:
    return dataset_root / str(shard) / str(relative_path)


def _group_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["shard"]), str(row["beatmap_set_id"])


def _empty_redline() -> dict[str, np.ndarray]:
    return {"offsets_ms": np.asarray([], dtype=np.float64), "bpms": np.asarray([], dtype=np.float64)}


def _empty_ramp_features() -> dict[str, Any]:
    empty_run = _empty_monotonic_run()
    return {
        "red_point_count": 0,
        "bpm_min": math.nan,
        "bpm_max": math.nan,
        "bpm_span": 0.0,
        "nonzero_bpm_diff_count": 0,
        "positive_bpm_diff_count": 0,
        "negative_bpm_diff_count": 0,
        "same_direction_fraction": 0.0,
        "direction_change_count": 0,
        "max_monotonic_run_length": empty_run["length"],
        "max_monotonic_run_start_ms": empty_run["start_ms"],
        "max_monotonic_run_end_ms": empty_run["end_ms"],
        "max_monotonic_run_duration_s": empty_run["duration_s"],
        "max_monotonic_start_bpm": empty_run["start_bpm"],
        "max_monotonic_end_bpm": empty_run["end_bpm"],
        "max_monotonic_bpm_span": empty_run["bpm_span"],
        "max_monotonic_abs_slope_bpm_per_s": empty_run["abs_slope_bpm_per_s"],
        "max_monotonic_direction": empty_run["direction"],
        "max_monotonic_linear_r2": empty_run["linear_r2"],
        "ramp_score": 0.0,
    }


def _empty_monotonic_run() -> dict[str, Any]:
    return {
        "length": 0,
        "start_ms": math.nan,
        "end_ms": math.nan,
        "duration_s": 0.0,
        "start_bpm": math.nan,
        "end_bpm": math.nan,
        "bpm_span": 0.0,
        "abs_slope_bpm_per_s": 0.0,
        "direction": "none",
        "linear_r2": 0.0,
    }


def _round_list(values: Iterable[float], *, digits: int) -> list[float]:
    return [round(float(value), digits) for value in values]


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_string(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _json_scalar(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return value.item()
    return value


def _parse_numeric_string(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":
    raise SystemExit(main())
