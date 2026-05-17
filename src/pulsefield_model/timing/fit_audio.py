from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from pulsefield_model.timing.grid_fitting import GridFitter, GridFitterConfig, TimingFitResult
from pulsefield_model.timing.providers.beatthis import (
    DEFAULT_BEATTHIS_CHECKPOINT,
    DEFAULT_BEATTHIS_DEVICE,
    BeatThisTimingProvider,
)
from pulsefield_model.timing.schema import FrameTimingPrediction


def fit_audio_file(
    audio_path: Path,
    *,
    checkpoint_path: str = DEFAULT_BEATTHIS_CHECKPOINT,
    device: str = DEFAULT_BEATTHIS_DEVICE,
    float16: bool = False,
    fitter_config: GridFitterConfig = GridFitterConfig(),
) -> dict[str, object]:
    provider = BeatThisTimingProvider(
        checkpoint_path=checkpoint_path,
        device=device,
        float16=float16,
    )
    prediction = provider.predict_file(audio_path)
    fitter = GridFitter(fitter_config)

    fit_start_seconds = time.perf_counter()
    fit_result = fitter.fit(prediction)
    fit_seconds = time.perf_counter() - fit_start_seconds
    return _timing_report(prediction, fit_result, fit_seconds=fit_seconds, device=device)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    report = fit_audio_file(
        args.audio_path,
        checkpoint_path=args.checkpoint,
        device=args.device,
        float16=args.float16,
        fitter_config=_fitter_config_from_args(args),
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
    return "\n".join(lines)


def _report_header_lines(report: dict[str, object]) -> list[str]:
    return [
        f"source: {report['source_path']}",
        f"provider: {report['provider']}",
        f"checkpoint: {report['checkpoint_path']}",
        f"device: {report['device']}",
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


def _build_arg_parser() -> argparse.ArgumentParser:
    default_config = GridFitterConfig()
    parser = argparse.ArgumentParser(description="Fit timing segments from an audio file with BeatThis.")
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--checkpoint", default=DEFAULT_BEATTHIS_CHECKPOINT)
    parser.add_argument("--device", default=DEFAULT_BEATTHIS_DEVICE)
    parser.add_argument("--float16", action="store_true")
    parser.add_argument("--json", action="store_true", dest="emit_json")
    parser.add_argument("--min-bpm", type=float, default=default_config.min_bpm)
    parser.add_argument("--max-bpm", type=float, default=default_config.max_bpm)
    parser.add_argument("--max-segments", type=int, default=default_config.max_segments)
    parser.add_argument("--double-tempo-score-ratio-threshold", type=float, default=None)
    return parser


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
    )


def _timing_report(
    prediction: FrameTimingPrediction,
    fit_result: TimingFitResult,
    *,
    fit_seconds: float,
    device: str,
) -> dict[str, object]:
    return {
        "source_path": prediction.source_path,
        "provider": prediction.provider,
        "checkpoint_path": prediction.checkpoint_path,
        "device": device,
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


if __name__ == "__main__":
    raise SystemExit(main())
