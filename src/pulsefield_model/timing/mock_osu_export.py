from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pulsefield_model.events.canonical import CanonicalTimepoint
from pulsefield_model.events.canonical import LaneAction as CanonicalLaneAction
from pulsefield_model.inference.osu_export import OsuExportMetadata, format_osu_export
from pulsefield_model.timing.canonicalization import (
    TIMING_CANONICALIZATION_CHOICES,
    TIMING_CANONICALIZATION_NONE,
)
from pulsefield_model.timing.fit_audio import fit_audio_file
from pulsefield_model.timing.grid_fitting import GridFitterConfig
from pulsefield_model.timing.providers.beatthis import (
    DEFAULT_BEATTHIS_CHECKPOINT,
    DEFAULT_BEATTHIS_DEVICE,
)
from pulsefield_model.timing.schema import FittedTimingGrid, TimingSegment


DEFAULT_OUTPUT_DIR = Path("artifacts/timing_mock_beatmaps")
MOCK_PATTERN_LABEL = "[0011] [1100]"
_MOCK_PATTERNS = (
    (
        CanonicalLaneAction.NONE,
        CanonicalLaneAction.NONE,
        CanonicalLaneAction.TAP,
        CanonicalLaneAction.TAP,
    ),
    (
        CanonicalLaneAction.TAP,
        CanonicalLaneAction.TAP,
        CanonicalLaneAction.NONE,
        CanonicalLaneAction.NONE,
    ),
)


@dataclass(frozen=True)
class MockBeatGridBuildResult:
    timepoints: tuple[CanonicalTimepoint, ...]
    max_rounding_error_ms: float


def create_timing_mock_beatmap(
    audio_path: str | Path,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    checkpoint_path: str = DEFAULT_BEATTHIS_CHECKPOINT,
    device: str = DEFAULT_BEATTHIS_DEVICE,
    float16: bool = False,
    fitter_config: GridFitterConfig = GridFitterConfig(),
    start_ms: int = 0,
    end_ms: int | None = None,
    max_beats: int | None = None,
    title: str | None = None,
    artist: str = "Unknown Artist",
    creator: str = "Pulsefield Timing Mock",
    version: str = "Timing grid mock [0011] [1100]",
) -> dict[str, object]:
    resolved_audio_path = Path(audio_path).expanduser().resolve(strict=False)
    resolved_output_dir = Path(output_dir).expanduser().resolve(strict=False)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    timing_report = fit_audio_file(
        resolved_audio_path,
        checkpoint_path=checkpoint_path,
        device=device,
        float16=float16,
        fitter_config=fitter_config,
    )
    timing_grid = timing_grid_from_report(timing_report)
    resolved_end_ms = _resolve_end_ms(timing_report, end_ms=end_ms)
    grid_build = build_mock_beat_grid_timepoints(
        timing_grid,
        start_ms=start_ms,
        end_ms=resolved_end_ms,
        max_beats=max_beats,
    )

    output_path = _mock_output_path(resolved_output_dir, resolved_audio_path)
    report_path = output_path.with_suffix(".json")
    output_path.write_text(
        format_osu_export(
            grid_build.timepoints,
            metadata=_mock_metadata(
                audio_path=resolved_audio_path,
                output_path=output_path,
                title=title,
                artist=artist,
                creator=creator,
                version=version,
            ),
            timing_grid=timing_grid,
        ),
        encoding="utf-8",
    )

    result = _result_payload(
        audio_path=resolved_audio_path,
        output_path=output_path,
        report_path=report_path,
        start_ms=start_ms,
        end_ms=resolved_end_ms,
        grid_build=grid_build,
        timing_report=timing_report,
    )
    _write_json_report(report_path, result=result, timing_report=timing_report)
    return result


def timing_grid_from_report(report: dict[str, object]) -> FittedTimingGrid:
    segments = report.get("segments")
    if not isinstance(segments, list):
        raise TypeError("timing report must contain a segments list")

    timing_segments: list[TimingSegment] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise TypeError(f"timing report segment {index} must be a dictionary")
        timing_segments.append(_timing_segment_from_report(segment))
    return FittedTimingGrid(tuple(timing_segments))


def build_mock_beat_grid_timepoints(
    timing_grid: FittedTimingGrid,
    *,
    start_ms: int = 0,
    end_ms: int,
    max_beats: int | None = None,
) -> MockBeatGridBuildResult:
    if start_ms < 0:
        raise ValueError(f"start_ms must be non-negative, got {start_ms!r}")
    if end_ms <= start_ms:
        raise ValueError(f"end_ms must be after start_ms, got {start_ms}..{end_ms}")
    if max_beats is not None and max_beats <= 0:
        raise ValueError(f"max_beats must be positive when provided, got {max_beats!r}")

    timepoints: list[CanonicalTimepoint] = []
    seen_times: set[int] = set()
    max_rounding_error_ms = 0.0
    for segment_index, segment in enumerate(timing_grid.segments):
        segment_end_ms = float(end_ms)
        if segment_index + 1 < len(timing_grid.segments):
            segment_end_ms = min(segment_end_ms, timing_grid.segments[segment_index + 1].offset_ms)
        if segment_end_ms <= start_ms:
            continue

        beat_index = max(
            0,
            int(math.ceil((float(start_ms) - segment.offset_ms) / segment.beat_length_ms - 1e-9)),
        )
        while True:
            beat_time_ms = segment.offset_ms + beat_index * segment.beat_length_ms
            if beat_time_ms >= segment_end_ms - 1e-9 or beat_time_ms >= end_ms:
                break
            if beat_time_ms >= start_ms - 1e-9:
                rounded_time_ms = _round_ms_half_up(beat_time_ms)
                if rounded_time_ms not in seen_times:
                    seen_times.add(rounded_time_ms)
                    max_rounding_error_ms = max(max_rounding_error_ms, abs(rounded_time_ms - beat_time_ms))
                    pattern = _MOCK_PATTERNS[len(timepoints) % len(_MOCK_PATTERNS)]
                    timepoints.append(
                        CanonicalTimepoint(
                            time_ms=rounded_time_ms,
                            lane_actions=pattern,
                        )
                    )
                    if max_beats is not None and len(timepoints) >= max_beats:
                        return MockBeatGridBuildResult(
                            timepoints=tuple(timepoints),
                            max_rounding_error_ms=float(max_rounding_error_ms),
                        )
            beat_index += 1

    return MockBeatGridBuildResult(
        timepoints=tuple(sorted(timepoints, key=lambda item: item.time_ms)),
        max_rounding_error_ms=float(max_rounding_error_ms),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    result = create_timing_mock_beatmap(
        args.audio_path,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        device=args.device,
        float16=args.float16,
        fitter_config=_fitter_config_from_args(args),
        start_ms=args.start_ms,
        end_ms=args.end_ms,
        max_beats=args.max_beats,
        title=args.title,
        artist=args.artist,
        creator=args.creator,
        version=args.version,
    )
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    default_config = GridFitterConfig()
    parser = argparse.ArgumentParser(
        description="Fit audio timing and write a mock osu!mania map with [0011] [1100] taps on the beat grid.",
    )
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_BEATTHIS_CHECKPOINT)
    parser.add_argument("--device", default=DEFAULT_BEATTHIS_DEVICE)
    parser.add_argument("--float16", action="store_true")
    parser.add_argument("--start-ms", type=int, default=0)
    parser.add_argument("--end-ms", type=int, default=None)
    parser.add_argument("--max-beats", type=int, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--artist", default="Unknown Artist")
    parser.add_argument("--creator", default="Pulsefield Timing Mock")
    parser.add_argument("--version", default="Timing grid mock [0011] [1100]")
    parser.add_argument("--min-bpm", type=float, default=default_config.min_bpm)
    parser.add_argument("--max-bpm", type=float, default=default_config.max_bpm)
    parser.add_argument("--max-segments", type=int, default=default_config.max_segments)
    parser.add_argument("--double-tempo-score-ratio-threshold", type=float, default=None)
    parser.add_argument(
        "--canonicalization",
        nargs="?",
        const=default_config.canonicalization,
        default=default_config.canonicalization,
        choices=TIMING_CANONICALIZATION_CHOICES,
        help="Fold fitted BPMs into the default canonical band; pass 'none' to leave timing unchanged.",
    )
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
        canonicalization=args.canonicalization,
        canonicalize_tempo_aliases=args.canonicalization == TIMING_CANONICALIZATION_NONE,
    )


def _resolve_end_ms(report: dict[str, object], *, end_ms: int | None) -> int:
    if end_ms is not None:
        if end_ms <= 0:
            raise ValueError(f"end_ms must be positive when provided, got {end_ms!r}")
        return int(end_ms)
    frame_count = int(report["frame_count"])
    frame_rate_hz = float(report["frame_rate_hz"])
    if frame_count <= 0 or frame_rate_hz <= 0.0:
        raise ValueError(f"invalid report duration fields: frame_count={frame_count}, frame_rate_hz={frame_rate_hz}")
    return int(math.ceil(frame_count / frame_rate_hz * 1000.0))


def _timing_segment_from_report(segment: dict[str, object]) -> TimingSegment:
    return TimingSegment(
        offset_ms=float(segment["offset_ms"]),
        beat_length_ms=float(segment["beat_length_ms"]),
        meter=int(segment.get("meter", 4)),
    )


def _mock_output_path(output_dir: Path, audio_path: Path) -> Path:
    return output_dir / f"{_safe_filename_stem(audio_path.stem)}_timing_mock.osu"


def _mock_metadata(
    *,
    audio_path: Path,
    output_path: Path,
    title: str | None,
    artist: str,
    creator: str,
    version: str,
) -> OsuExportMetadata:
    return OsuExportMetadata(
        audio_filename=_relative_audio_filename(audio_path, output_path),
        title=title or audio_path.stem or "Timing Mock",
        artist=artist,
        creator=creator,
        version=version,
    )


def _result_payload(
    *,
    audio_path: Path,
    output_path: Path,
    report_path: Path,
    start_ms: int,
    end_ms: int,
    grid_build: MockBeatGridBuildResult,
    timing_report: dict[str, object],
) -> dict[str, object]:
    return {
        "status": "ok",
        "audio_path": audio_path.as_posix(),
        "osu_mock_beatmap_path": output_path.as_posix(),
        "osumockbeatmappath": output_path.as_posix(),
        "timing_report_path": report_path.as_posix(),
        "pattern": MOCK_PATTERN_LABEL,
        "start_ms": int(start_ms),
        "end_ms": int(end_ms),
        "beat_count": len(grid_build.timepoints),
        "hitobject_count": _hitobject_count(grid_build.timepoints),
        "max_grid_rounding_error_ms": grid_build.max_rounding_error_ms,
        "timing_segments": timing_report["segments"],
    }


def _write_json_report(
    report_path: Path,
    *,
    result: dict[str, object],
    timing_report: dict[str, object],
) -> None:
    report_path.write_text(
        json.dumps(
            {
                **result,
                "timing_report": timing_report,
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _hitobject_count(timepoints: Sequence[CanonicalTimepoint]) -> int:
    return sum(
        1
        for timepoint in timepoints
        for action in timepoint.lane_actions
        if action != CanonicalLaneAction.NONE
    )


def _relative_audio_filename(audio_path: Path, output_path: Path) -> str:
    return os.path.relpath(audio_path, start=output_path.parent).replace(os.sep, "/")


def _round_ms_half_up(value: float) -> int:
    if value < 0.0:
        raise ValueError(f"cannot round negative beat time: {value}")
    return int(math.floor(value + 0.5))


def _safe_filename_stem(stem: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return safe or "audio"


if __name__ == "__main__":
    raise SystemExit(main())
