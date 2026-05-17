from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from pulsefield_model.timing.grid_fitting import GridFitter, GridFitterConfig
from pulsefield_model.timing.grid_fitting.alias import (
    _alias_bpm_abs_error,
    _alias_bpm_mae,
    _distribution_or_default,
    _segment_alias_switch_count,
)
from pulsefield_model.timing.rendering.dense_timing_v2 import (
    DEFAULT_DENSE_TIMING_V2_CONFIG,
    DenseTimingV2Config,
    active_timing_arrays,
    dense_timing_v2_frame_times,
    render_dense_timing_v2,
)
from pulsefield_model.timing.schema import FittedTimingGrid, TimingSegment


_BEAT_PULSE_CHANNEL = 0
_PHASE_SIN_CHANNEL = 1
_PHASE_COS_CHANNEL = 2
_LOCAL_BPM_CHANNEL = 3

_SUMMARY_METRICS = (
    "fit_score",
    "fit_seconds",
    "beat_pulse_mae",
    "local_bpm_mae",
    "local_bpm_alias_mae",
    "first_bpm_abs_error",
    "first_bpm_alias_error",
    "mean_phase_error_beats",
    "max_phase_error_beats",
    "mean_phase_error_ms",
    "max_phase_error_ms",
)


@dataclass(frozen=True)
class TimingGridComparison:
    frame_count: int
    beat_pulse_mae: float
    local_bpm_mae: float
    local_bpm_alias_mae: float
    mean_phase_error_beats: float
    max_phase_error_beats: float
    mean_phase_error_ms: float
    max_phase_error_ms: float


def compare_timing_grids(
    predicted_grid: FittedTimingGrid,
    oracle_grid: FittedTimingGrid,
    *,
    frame_count: int,
    input_start_ms: float = 0.0,
    config: DenseTimingV2Config = DEFAULT_DENSE_TIMING_V2_CONFIG,
) -> TimingGridComparison:
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count!r}")

    predicted_track = render_dense_timing_v2(
        predicted_grid,
        input_start_ms=input_start_ms,
        frame_count=frame_count,
        config=config,
    )
    oracle_track = render_dense_timing_v2(
        oracle_grid,
        input_start_ms=input_start_ms,
        frame_count=frame_count,
        config=config,
    )

    frame_times_ms = dense_timing_v2_frame_times(
        input_start_ms,
        frame_count=frame_count,
        config=config,
    )
    _, oracle_beat_lengths_ms = active_timing_arrays(oracle_grid, frame_times_ms)

    phase_error_beats = _phase_error_beats(predicted_track, oracle_track)
    phase_error_ms = phase_error_beats * oracle_beat_lengths_ms

    return TimingGridComparison(
        frame_count=frame_count,
        beat_pulse_mae=_mean_absolute_channel_delta(
            predicted_track,
            oracle_track,
            _BEAT_PULSE_CHANNEL,
        ),
        local_bpm_mae=_mean_absolute_channel_delta(
            predicted_track,
            oracle_track,
            _LOCAL_BPM_CHANNEL,
        ),
        local_bpm_alias_mae=_alias_bpm_mae(
            predicted_track[:, _LOCAL_BPM_CHANNEL].astype(np.float64),
            oracle_track[:, _LOCAL_BPM_CHANNEL].astype(np.float64),
        ),
        mean_phase_error_beats=float(np.mean(phase_error_beats)),
        max_phase_error_beats=float(np.max(phase_error_beats)),
        mean_phase_error_ms=float(np.mean(phase_error_ms)),
        max_phase_error_ms=float(np.max(phase_error_ms)),
    )


def oracle_grid_from_red_timing_points(red_timing_points: Sequence[object]) -> FittedTimingGrid:
    return FittedTimingGrid(
        segments=tuple(
            TimingSegment(
                offset_ms=float(point.offset_ms),
                beat_length_ms=float(point.beat_length_ms),
                meter=int(getattr(point, "meter", 4)),
            )
            for point in red_timing_points
        )
    )


def run_beatthis_oracle_comparison(
    *,
    index_path: Path,
    dataset_root: Path,
    sample_size: int,
    seed: int,
    device: str,
    double_tempo_score_ratio_threshold: float | None = None,
    max_average_fit_seconds: float | None = 1.0,
) -> dict[str, object]:
    import pandas as pd

    from pulsefield_model.osu_core.timing import require_red_timing_points
    from pulsefield_model.timing.providers.beatthis import BeatThisTimingProvider

    if sample_size <= 0:
        raise ValueError(f"sample_size must be positive, got {sample_size!r}")

    index_df = pd.read_parquet(index_path)
    sample_df = index_df.sample(n=min(sample_size, len(index_df)), random_state=seed)
    provider = BeatThisTimingProvider(device=device)
    fitter = GridFitter(_grid_fitter_config(double_tempo_score_ratio_threshold))

    rows: list[dict[str, object]] = []
    for _, row in sample_df.iterrows():
        total_start_seconds = time.perf_counter()
        beatmap_path, audio_path = _sample_paths(dataset_root, row)
        oracle_grid = oracle_grid_from_red_timing_points(require_red_timing_points(beatmap_path))
        prediction_start_seconds = time.perf_counter()
        prediction = provider.predict_file(audio_path)
        prediction_seconds = time.perf_counter() - prediction_start_seconds
        fit_start_seconds = time.perf_counter()
        fit_result = fitter.fit(prediction)
        fit_seconds = time.perf_counter() - fit_start_seconds
        comparison = compare_timing_grids(
            fit_result.grid,
            oracle_grid,
            frame_count=prediction.frame_count,
        )
        total_seconds = time.perf_counter() - total_start_seconds
        rows.append(
            _comparison_row(
                beatmap_path=beatmap_path,
                audio_path=audio_path,
                frame_count=prediction.frame_count,
                frame_rate_hz=prediction.frame_rate_hz,
                prediction_seconds=prediction_seconds,
                candidate_count=fit_result.diagnostics.candidate_count,
                alias_candidate_count=fit_result.diagnostics.alias_candidate_count,
                fit_score=fit_result.score,
                fit_seconds=fit_seconds,
                total_seconds=total_seconds,
                predicted_segments=fit_result.grid.segments,
                oracle_segments=oracle_grid.segments,
                raw_selected_bpm=fit_result.diagnostics.raw_selected_bpm,
                raw_score=fit_result.diagnostics.raw_score,
                half_tempo_score=fit_result.diagnostics.half_tempo_score,
                double_tempo_score=fit_result.diagnostics.double_tempo_score,
                tempo_multiplier=fit_result.diagnostics.tempo_multiplier,
                tempo_multiplier_distribution=fit_result.diagnostics.tempo_multiplier_distribution,
                segment_alias_switch_count=fit_result.diagnostics.segment_alias_switch_count,
                comparison=comparison,
            )
        )

    summary = _summarize_rows(rows)
    _enforce_average_fit_seconds(summary, max_average_fit_seconds=max_average_fit_seconds)
    return {
        "sample_size": len(rows),
        "max_average_fit_seconds": max_average_fit_seconds,
        "rows": rows,
        "summary": summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare BeatThis-fitted dense timing v2 against osu red timing.")
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--double-tempo-score-ratio-threshold", type=float, default=None)
    parser.add_argument(
        "--max-average-fit-seconds",
        type=float,
        default=1.0,
        help="Fail if average GridFitter time exceeds this value; use 0 to disable.",
    )
    args = parser.parse_args(argv)

    report = run_beatthis_oracle_comparison(
        index_path=args.index_path,
        dataset_root=args.dataset_root,
        sample_size=args.sample_size,
        seed=args.seed,
        device=args.device,
        double_tempo_score_ratio_threshold=args.double_tempo_score_ratio_threshold,
        max_average_fit_seconds=_max_average_fit_seconds_from_args(args),
    )
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0


def _phase_error_beats(
    predicted_track: np.ndarray,
    oracle_track: np.ndarray,
) -> np.ndarray:
    angle_delta = np.arctan2(
        predicted_track[:, _PHASE_SIN_CHANNEL] * oracle_track[:, _PHASE_COS_CHANNEL]
        - predicted_track[:, _PHASE_COS_CHANNEL] * oracle_track[:, _PHASE_SIN_CHANNEL],
        predicted_track[:, _PHASE_COS_CHANNEL] * oracle_track[:, _PHASE_COS_CHANNEL]
        + predicted_track[:, _PHASE_SIN_CHANNEL] * oracle_track[:, _PHASE_SIN_CHANNEL],
    )
    return np.abs(angle_delta) / (2.0 * np.pi)


def _mean_absolute_channel_delta(
    predicted_track: np.ndarray,
    oracle_track: np.ndarray,
    channel_index: int,
) -> float:
    return float(np.mean(np.abs(predicted_track[:, channel_index] - oracle_track[:, channel_index])))


def _grid_fitter_config(double_tempo_score_ratio_threshold: float | None) -> GridFitterConfig:
    if double_tempo_score_ratio_threshold is None:
        return GridFitterConfig()
    return GridFitterConfig(double_tempo_score_ratio_threshold=double_tempo_score_ratio_threshold)


def _sample_paths(dataset_root: Path, row: Mapping[str, object]) -> tuple[Path, Path]:
    shard_path = dataset_root / str(row["shard"])
    return shard_path / str(row["beatmap_path"]), shard_path / str(row["audio_path"])


def _comparison_row(
    *,
    beatmap_path: Path,
    audio_path: Path,
    frame_count: int,
    frame_rate_hz: float,
    prediction_seconds: float,
    candidate_count: int,
    alias_candidate_count: int,
    fit_score: float,
    fit_seconds: float,
    total_seconds: float,
    predicted_segments: Sequence[TimingSegment],
    oracle_segments: Sequence[TimingSegment],
    raw_selected_bpm: float,
    raw_score: float,
    half_tempo_score: float,
    double_tempo_score: float,
    tempo_multiplier: float,
    comparison: TimingGridComparison,
    tempo_multiplier_distribution: Mapping[str, int] | None = None,
    segment_alias_switch_count: int | None = None,
) -> dict[str, object]:
    predicted_first_segment = predicted_segments[0]
    oracle_first_segment = oracle_segments[0]
    predicted_segment_count = len(predicted_segments)
    oracle_segment_count = len(oracle_segments)
    first_bpm_abs_error = abs(predicted_first_segment.local_bpm - oracle_first_segment.local_bpm)
    first_bpm_alias_error = _alias_bpm_abs_error(
        predicted_first_segment.local_bpm,
        oracle_first_segment.local_bpm,
    )
    resolved_segment_alias_switch_count = (
        _segment_alias_switch_count(predicted_segments, config=GridFitterConfig())
        if segment_alias_switch_count is None
        else int(segment_alias_switch_count)
    )
    return {
        "beatmap_path": beatmap_path.as_posix(),
        "audio_path": audio_path.as_posix(),
        "frame_count": frame_count,
        "audio_duration_seconds": float(frame_count) / frame_rate_hz,
        "prediction_seconds": float(prediction_seconds),
        "candidate_count": int(candidate_count),
        "alias_candidate_count": int(alias_candidate_count),
        "fit_score": fit_score,
        "fit_seconds": fit_seconds,
        "total_seconds": float(total_seconds),
        "predicted_bpm": predicted_first_segment.local_bpm,
        "predicted_offset_ms": predicted_first_segment.offset_ms,
        "predicted_first_bpm": predicted_first_segment.local_bpm,
        "predicted_first_offset_ms": predicted_first_segment.offset_ms,
        "oracle_first_bpm": oracle_first_segment.local_bpm,
        "oracle_first_offset_ms": oracle_first_segment.offset_ms,
        "first_bpm_abs_error": float(first_bpm_abs_error),
        "first_bpm_alias_error": float(first_bpm_alias_error),
        "predicted_segment_count": predicted_segment_count,
        "oracle_segment_count": oracle_segment_count,
        "segment_count_delta": predicted_segment_count - oracle_segment_count,
        "segment_alias_switch_count": resolved_segment_alias_switch_count,
        "predicted_segments": _segments_payload(predicted_segments),
        "oracle_segments": _segments_payload(oracle_segments),
        "raw_selected_bpm": raw_selected_bpm,
        "raw_score": raw_score,
        "half_tempo_score": _finite_float_or_none(half_tempo_score),
        "double_tempo_score": _finite_float_or_none(double_tempo_score),
        "tempo_multiplier": tempo_multiplier,
        "tempo_multiplier_distribution": _distribution_or_default(
            tempo_multiplier_distribution,
            tempo_multiplier=tempo_multiplier,
            segment_count=predicted_segment_count,
        ),
        **asdict(comparison),
    }


def _max_average_fit_seconds_from_args(args: argparse.Namespace) -> float | None:
    if args.max_average_fit_seconds <= 0.0:
        return None
    return args.max_average_fit_seconds


def _summarize_rows(rows: Sequence[dict[str, object]]) -> dict[str, float]:
    if not rows:
        return {}

    summary: dict[str, float] = {}
    for name in _SUMMARY_METRICS:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        summary[f"{name}_mean"] = float(np.mean(values))
        summary[f"{name}_max"] = float(np.max(values))
    return summary


def _enforce_average_fit_seconds(
    summary: dict[str, float],
    *,
    max_average_fit_seconds: float | None,
) -> None:
    if max_average_fit_seconds is None or not summary:
        return
    if max_average_fit_seconds <= 0.0:
        raise ValueError(f"max_average_fit_seconds must be positive or None, got {max_average_fit_seconds!r}")
    average_fit_seconds = summary.get("fit_seconds_mean")
    if average_fit_seconds is not None and average_fit_seconds > max_average_fit_seconds:
        raise RuntimeError(
            "average fitter time exceeded hard limit: "
            f"{average_fit_seconds:.3f}s > {max_average_fit_seconds:.3f}s"
        )


def _finite_float_or_none(value: float) -> float | None:
    if not np.isfinite(value):
        return None
    return float(value)


def _segments_payload(segments: Sequence[TimingSegment]) -> list[dict[str, float | int]]:
    return [
        {
            "offset_ms": segment.offset_ms,
            "beat_length_ms": segment.beat_length_ms,
            "bpm": segment.local_bpm,
            "meter": segment.meter,
        }
        for segment in segments
    ]


if __name__ == "__main__":
    raise SystemExit(main())
