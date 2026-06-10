from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from pulsefield_model.timing.canonicalization import (
    TIMING_CANONICALIZATION_BPM_80_160,
    TIMING_CANONICALIZATION_CHOICES,
    TIMING_CANONICALIZATION_NONE,
)
from pulsefield_model.timing.beat_materialization import (
    RampBeatGridHint,
    fit_ramp_beat_grid,
    ramp_beat_grid_report,
)
from pulsefield_model.timing.grid_fitting import GridFitter, GridFitterConfig, TimingFitResult
from pulsefield_model.timing.providers.beatthis import (
    DEFAULT_BEATTHIS_CHECKPOINT,
    DEFAULT_BEATTHIS_DEVICE,
    BeatThisTimingProvider,
    audio_shift_samples_for_ms,
)
from pulsefield_model.timing.ramp_detection import detect_timing_ramp
from pulsefield_model.timing.schema import FrameTimingPrediction


DEFAULT_SUPER_TIMING_SHIFT_MS = (0.0, 5.0, 10.0, 15.0)
SUPER_TIMING_ALIGNMENT = "segment offsets subtract shift_ms to align shifted runs back to the original audio"


def fit_audio_file(
    audio_path: Path,
    *,
    checkpoint_path: str = DEFAULT_BEATTHIS_CHECKPOINT,
    device: str = DEFAULT_BEATTHIS_DEVICE,
    float16: bool = False,
    fitter_config: GridFitterConfig = GridFitterConfig(),
    super_timing_shift_ms: Sequence[float] | None = None,
    ramp_beat_grid: bool = False,
    ramp_beat_grid_hint: RampBeatGridHint | None = None,
    ramp_beat_grid_allow_no_hint: bool = False,
) -> dict[str, object]:
    provider = BeatThisTimingProvider(
        checkpoint_path=checkpoint_path,
        device=device,
        float16=float16,
    )
    fitter = GridFitter(fitter_config)

    if super_timing_shift_ms is None:
        prediction = provider.predict_file(audio_path)
        fit_result, fit_seconds = _fit_prediction(fitter, prediction)
        report = _timing_report(
            prediction,
            fit_result,
            fit_seconds=fit_seconds,
            device=device,
            canonicalization=fitter_config.canonicalization,
        )
        _maybe_add_ramp_beat_grid(
            report,
            prediction,
            enabled=ramp_beat_grid,
            hint=ramp_beat_grid_hint,
            allow_no_hint=ramp_beat_grid_allow_no_hint,
        )
        return report

    shift_ms_values = _normalize_super_timing_shift_ms(super_timing_shift_ms)
    audio, sample_rate = provider.load_file(audio_path)
    prediction = provider.predict_audio(audio, sample_rate, source_path=audio_path)
    fit_result, fit_seconds = _fit_prediction(fitter, prediction)
    report = _timing_report(
        prediction,
        fit_result,
        fit_seconds=fit_seconds,
        device=device,
        canonicalization=fitter_config.canonicalization,
    )
    _maybe_add_ramp_beat_grid(
        report,
        prediction,
        enabled=ramp_beat_grid,
        hint=ramp_beat_grid_hint,
        allow_no_hint=ramp_beat_grid_allow_no_hint,
    )
    report["super_timing"] = _super_timing_report(
        provider,
        fitter,
        audio,
        sample_rate,
        audio_path=audio_path,
        default_run_report=report,
        shift_ms_values=shift_ms_values,
        device=device,
        canonicalization=fitter_config.canonicalization,
    )
    return report


def _fit_prediction(
    fitter: GridFitter,
    prediction: FrameTimingPrediction,
) -> tuple[TimingFitResult, float]:
    fit_start_seconds = time.perf_counter()
    fit_result = fitter.fit(prediction)
    fit_seconds = time.perf_counter() - fit_start_seconds
    return fit_result, fit_seconds


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    report = fit_audio_file(
        args.audio_path,
        checkpoint_path=args.checkpoint,
        device=args.device,
        float16=args.float16,
        fitter_config=_fitter_config_from_args(args),
        super_timing_shift_ms=_super_timing_shift_ms_from_args(args),
        ramp_beat_grid=args.ramp_beat_grid,
        ramp_beat_grid_hint=_ramp_beat_grid_hint_from_args(args),
        ramp_beat_grid_allow_no_hint=args.ramp_beat_grid_allow_no_hint,
    )
    if args.emit_json:
        print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    else:
        print(format_timing_report(report))
    return 0


def format_timing_report(report: dict[str, object]) -> str:
    lines = _report_header_lines(report)
    lines.append("segments:")
    lines.extend(
        _format_segment_line(index, segment)
        for index, segment in enumerate(_report_segments(report), start=1)
    )
    super_timing = report.get("super_timing")
    if isinstance(super_timing, dict):
        lines.extend(_format_super_timing_lines(super_timing))
    ramp_beat_grid = report.get("ramp_beat_grid")
    if isinstance(ramp_beat_grid, dict):
        lines.extend(_format_ramp_beat_grid_lines(ramp_beat_grid))
    return "\n".join(lines)


def _report_header_lines(report: dict[str, object]) -> list[str]:
    return [
        f"source: {report['source_path']}",
        f"provider: {report['provider']}",
        f"checkpoint: {report['checkpoint_path']}",
        f"device: {report['device']}",
        f"canonicalization: {report['canonicalization']}",
        f"frame_count: {report['frame_count']}",
        f"fit_seconds: {float(report['fit_seconds']):.3f}",
        f"score: {float(report['score']):.6f}",
    ]


def _report_segments(report: dict[str, object]) -> list[dict[str, object]]:
    segments = report["segments"]
    if not isinstance(segments, list):
        raise TypeError("report['segments'] must be a list")
    return segments


def _format_segment_line(index: int, segment: dict[str, object]) -> str:
    return (
        "  "
        f"{index}. "
        f"offset_ms={float(segment['offset_ms']):.3f} "
        f"beat_length_ms={float(segment['beat_length_ms']):.3f} "
        f"bpm={float(segment['bpm']):.3f} "
        f"meter={int(segment['meter'])}"
    )


def _format_super_timing_lines(super_timing: dict[str, object]) -> list[str]:
    runs = super_timing.get("runs")
    if not isinstance(runs, list):
        raise TypeError("report['super_timing']['runs'] must be a list")

    lines = ["super_timing:"]
    lines.append(f"  alignment: {super_timing['alignment']}")
    for run in runs:
        if not isinstance(run, dict):
            raise TypeError("super_timing runs must be dictionaries")
        lines.append(
            "  "
            f"shift_ms={float(run['shift_ms']):.3f} "
            f"pad_samples={int(run['pad_samples'])} "
            f"frame_count={int(run['frame_count'])} "
            f"score={float(run['score']):.6f}"
        )
        lines.extend(
            "  " + _format_segment_line(index, segment).strip()
            for index, segment in enumerate(_run_segments(run), start=1)
        )
    return lines


def _run_segments(run: dict[str, object]) -> list[dict[str, object]]:
    segments = run["segments"]
    if not isinstance(segments, list):
        raise TypeError("super_timing run['segments'] must be a list")
    return segments


def _build_arg_parser() -> argparse.ArgumentParser:
    default_config = GridFitterConfig()
    parser = argparse.ArgumentParser(description="Fit timing segments from an audio file with BeatThis.")
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--checkpoint", default=DEFAULT_BEATTHIS_CHECKPOINT)
    parser.add_argument("--device", default=DEFAULT_BEATTHIS_DEVICE)
    parser.add_argument("--float16", action="store_true")
    parser.add_argument("--json", action="store_true", dest="emit_json")
    parser.add_argument(
        "--super-timing-shifts",
        action="store_true",
        help="Run extra BeatThis passes at 0, 5, 10, and 15 ms and include aligned BPM segments.",
    )
    parser.add_argument(
        "--super-timing-shift-ms",
        action="append",
        type=float,
        default=None,
        help="Add a custom non-negative shifted BeatThis pass in milliseconds. Repeat to run multiple shifts.",
    )
    parser.add_argument(
        "--ramp-beat-grid",
        action="store_true",
        help="Run an opt-in BPM-ramp grid pass and include its auxiliary output.",
    )
    parser.add_argument(
        "--ramp-beat-grid-allow-no-hint",
        action="store_true",
        help="Allow exploratory no-hint ramp beat-grid mining. By default, ramp beat-grid output requires detector hints.",
    )
    parser.add_argument("--ramp-hint-start-ms", type=float, default=None)
    parser.add_argument("--ramp-hint-end-ms", type=float, default=None)
    parser.add_argument("--ramp-hint-start-bpm", type=float, default=None)
    parser.add_argument("--ramp-hint-end-bpm", type=float, default=None)
    parser.add_argument("--min-bpm", type=float, default=default_config.min_bpm)
    parser.add_argument("--max-bpm", type=float, default=default_config.max_bpm)
    parser.add_argument("--max-segments", type=int, default=default_config.max_segments)
    parser.add_argument("--double-tempo-score-ratio-threshold", type=float, default=None)
    parser.add_argument(
        "--canonicalization",
        nargs="?",
        const=TIMING_CANONICALIZATION_BPM_80_160,
        default=default_config.canonicalization,
        choices=TIMING_CANONICALIZATION_CHOICES,
        help="Fold fitted BPMs into [80, 160); pass 'none' to leave timing unchanged.",
    )
    return parser


def _super_timing_shift_ms_from_args(args: argparse.Namespace) -> Sequence[float] | None:
    if args.super_timing_shift_ms is not None:
        return args.super_timing_shift_ms
    if args.super_timing_shifts:
        return DEFAULT_SUPER_TIMING_SHIFT_MS
    return None


def _fitter_config_from_args(args: argparse.Namespace) -> GridFitterConfig:
    default_config = GridFitterConfig()
    double_tempo_threshold = (
        default_config.double_tempo_score_ratio_threshold
        if args.double_tempo_score_ratio_threshold is None
        else args.double_tempo_score_ratio_threshold
    )
    return GridFitterConfig(
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm,
        max_segments=args.max_segments,
        double_tempo_score_ratio_threshold=double_tempo_threshold,
        canonicalization=args.canonicalization,
        canonicalize_tempo_aliases=args.canonicalization == TIMING_CANONICALIZATION_NONE,
    )


def _ramp_beat_grid_hint_from_args(args: argparse.Namespace) -> RampBeatGridHint | None:
    values = (
        args.ramp_hint_start_ms,
        args.ramp_hint_end_ms,
        args.ramp_hint_start_bpm,
        args.ramp_hint_end_bpm,
    )
    if all(value is None for value in values):
        return None
    if not all(value is not None for value in values):
        raise ValueError(
            "ramp hint requires --ramp-hint-start-ms, --ramp-hint-end-ms, "
            "--ramp-hint-start-bpm, and --ramp-hint-end-bpm"
        )
    return RampBeatGridHint(
        start_ms=float(args.ramp_hint_start_ms),
        end_ms=float(args.ramp_hint_end_ms),
        start_bpm=float(args.ramp_hint_start_bpm),
        end_bpm=float(args.ramp_hint_end_bpm),
    )


def _timing_report(
    prediction: FrameTimingPrediction,
    fit_result: TimingFitResult,
    *,
    fit_seconds: float,
    device: str,
    canonicalization: str,
) -> dict[str, object]:
    ramp_detection = fit_result.diagnostics.ramp_detection
    if ramp_detection is None:
        ramp_detection = detect_timing_ramp(fit_result.grid)
    return {
        "source_path": prediction.source_path,
        "provider": prediction.provider,
        "checkpoint_path": prediction.checkpoint_path,
        "device": device,
        "canonicalization": canonicalization,
        "frame_count": prediction.frame_count,
        "frame_rate_hz": prediction.frame_rate_hz,
        "fit_seconds": float(fit_seconds),
        "score": fit_result.score,
        "diagnostics": {
            "selected_bpm": fit_result.diagnostics.selected_bpm,
            "raw_selected_bpm": fit_result.diagnostics.raw_selected_bpm,
            "tempo_multiplier": fit_result.diagnostics.tempo_multiplier,
            "candidate_count": fit_result.diagnostics.candidate_count,
            "alias_candidate_count": fit_result.diagnostics.alias_candidate_count,
        },
        "ramp": asdict(ramp_detection),
        "segments": [
            {
                "offset_ms": segment.offset_ms,
                "beat_length_ms": segment.beat_length_ms,
                "bpm": segment.local_bpm,
                "meter": segment.meter,
            }
            for segment in fit_result.grid.segments
        ],
    }


def _maybe_add_ramp_beat_grid(
    report: dict[str, object],
    prediction: FrameTimingPrediction,
    *,
    enabled: bool,
    hint: RampBeatGridHint | None,
    allow_no_hint: bool,
) -> None:
    if not enabled:
        return
    report["ramp_beat_grid"] = ramp_beat_grid_report(
        fit_ramp_beat_grid(prediction, hint=hint, allow_no_hint=allow_no_hint)
    )


def _super_timing_report(
    provider: BeatThisTimingProvider,
    fitter: GridFitter,
    audio: object,
    sample_rate: int,
    *,
    audio_path: Path,
    default_run_report: dict[str, object],
    shift_ms_values: Sequence[float],
    device: str,
    canonicalization: str,
) -> dict[str, object]:
    runs = []
    for shift_ms in shift_ms_values:
        if shift_ms == 0.0:
            run_report = default_run_report
        else:
            prediction = provider.predict_shifted_audio(
                audio,
                sample_rate,
                shift_ms=shift_ms,
                source_path=audio_path,
            )
            fit_result, fit_seconds = _fit_prediction(fitter, prediction)
            run_report = _timing_report(
                prediction,
                fit_result,
                fit_seconds=fit_seconds,
                device=device,
                canonicalization=canonicalization,
            )
        runs.append(
            _super_timing_run_report(
                run_report,
                shift_ms=shift_ms,
                pad_samples=audio_shift_samples_for_ms(shift_ms, sample_rate),
            )
        )

    return {
        "shift_ms": list(shift_ms_values),
        "alignment": SUPER_TIMING_ALIGNMENT,
        "runs": runs,
    }


def _format_ramp_beat_grid_lines(ramp_beat_grid: dict[str, object]) -> list[str]:
    lines = ["ramp_beat_grid:"]
    lines.append(
        "  "
        f"accepted={bool(ramp_beat_grid['accepted'])} "
        f"seconds={float(ramp_beat_grid['seconds']):.3f} "
        f"peaks={int(ramp_beat_grid['peak_count'])} "
        f"candidates={int(ramp_beat_grid['candidate_count'])} "
        f"reasons={','.join(str(reason) for reason in ramp_beat_grid['reasons'])}"
    )
    sequence = ramp_beat_grid.get("sequence")
    if isinstance(sequence, dict):
        lines.append(
            "  "
            f"beats={int(sequence['beat_count'])} "
            f"sequence_ms={float(sequence['start_ms']):.3f}-{float(sequence['end_ms']):.3f} "
            f"median_bpm={float(sequence['median_bpm']):.3f} "
            f"mean_peak_prob={float(sequence['mean_peak_prob']):.3f}"
        )
    candidate = ramp_beat_grid.get("candidate")
    if isinstance(candidate, dict):
        lines.append(
            "  "
            f"window_ms={float(candidate['start_ms']):.3f}-{float(candidate['end_ms']):.3f} "
            f"points={int(candidate['point_count'])} "
            f"bpm={float(candidate['start_bpm']):.3f}->{float(candidate['end_bpm']):.3f} "
            f"r2={float(candidate['linear_r2']):.3f} "
            f"prob_score={float(candidate['probability_score']):.3f}"
        )
        segments = candidate.get("segments")
        if isinstance(segments, list):
            lines.append("  segments:")
            lines.extend(
                "  " + _format_segment_line(index, segment).strip()
                for index, segment in enumerate(segments, start=1)
                if isinstance(segment, dict)
            )
    return lines


def _super_timing_run_report(
    report: dict[str, object],
    *,
    shift_ms: float,
    pad_samples: int,
) -> dict[str, object]:
    raw_segments = [dict(segment) for segment in _report_segments(report)]
    return {
        "shift_ms": float(shift_ms),
        "pad_samples": int(pad_samples),
        "frame_count": report["frame_count"],
        "frame_rate_hz": report["frame_rate_hz"],
        "fit_seconds": report["fit_seconds"],
        "score": report["score"],
        "diagnostics": report["diagnostics"],
        "raw_segments": raw_segments,
        "segments": [_align_segment(segment, shift_ms=shift_ms) for segment in raw_segments],
    }


def _align_segment(segment: dict[str, object], *, shift_ms: float) -> dict[str, object]:
    aligned = dict(segment)
    aligned["offset_ms"] = float(aligned["offset_ms"]) - shift_ms
    return aligned


def _normalize_super_timing_shift_ms(values: Sequence[float]) -> tuple[float, ...]:
    normalized: list[float] = [0.0]
    for value in values:
        shift_ms = float(value)
        if not np.isfinite(shift_ms) or shift_ms < 0.0:
            raise ValueError(f"super timing shift_ms must be non-negative and finite, got {value!r}")
        if not any(abs(shift_ms - existing) < 1e-9 for existing in normalized):
            normalized.append(shift_ms)
    return tuple(normalized)


if __name__ == "__main__":
    raise SystemExit(main())
