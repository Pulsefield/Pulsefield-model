from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_right
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any, Final, Sequence

from pulsefield_model.osu_core.hitobjects import ManiaHitObject, ManiaHitObjectKind, parse_mania_hit_objects
from pulsefield_model.osu_core.timing import (
    MissingRedTimingError,
    RedTimingPoint,
    require_red_timing_points,
    validate_red_timing_point,
)
from pulsefield_model.timing.canonicalization import (
    TIMING_CANONICALIZATION_BPM_80_160,
    TIMING_CANONICALIZATION_CHOICES,
    TIMING_CANONICALIZATION_NONE,
    canonical_bpm_80_160,
    require_timing_canonicalization,
)


DEFAULT_SNAP_DENOMINATOR: Final[int] = 48
DEFAULT_BEAT_REPRESENTATION_TIMING_CANONICALIZATION: Final[str] = TIMING_CANONICALIZATION_BPM_80_160
DEFAULT_SNAP_DIAGNOSTIC_SUBDIVISIONS: Final[tuple[int, ...]] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 48)


class BeatEventKind(str, Enum):
    TAP = "tap"
    HOLD_START = "hold_start"
    HOLD_END = "hold_end"


@dataclass(frozen=True)
class SnapDiagnostics:
    raw_beat_offset: float
    snap_error_beats: float
    abs_snap_error_beats: float
    subdivision: int
    subdivision_step_beats: float
    relative_error_to_subdivision: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_beat_offset": self.raw_beat_offset,
            "snap_error_beats": self.snap_error_beats,
            "abs_snap_error_beats": self.abs_snap_error_beats,
            "subdivision": self.subdivision,
            "subdivision_step_beats": self.subdivision_step_beats,
            "relative_error_to_subdivision": self.relative_error_to_subdivision,
        }


@dataclass(frozen=True)
class BeatEvent:
    kind: BeatEventKind
    lane: int
    time_ms: float
    redline_offset_ms: float
    redline_beat_length_ms: float
    redline_bpm: float
    beat_length_ms: float
    bpm: float
    beat_offset_numerator: int
    beat_offset_denominator: int
    beat_offset: float
    snapped_time_ms: float
    snap_error_ms: float
    diagnostics: SnapDiagnostics | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "kind": self.kind.value,
            "lane": self.lane,
            "time_ms": self.time_ms,
            "redline_offset_ms": self.redline_offset_ms,
            "redline_beat_length_ms": self.redline_beat_length_ms,
            "redline_bpm": self.redline_bpm,
            "beat_length_ms": self.beat_length_ms,
            "bpm": self.bpm,
            "beat_offset_numerator": self.beat_offset_numerator,
            "beat_offset_denominator": self.beat_offset_denominator,
            "beat_offset": self.beat_offset,
            "beat_offset_fraction": str(Fraction(self.beat_offset_numerator, self.beat_offset_denominator)),
            "snapped_time_ms": self.snapped_time_ms,
            "snap_error_ms": self.snap_error_ms,
        }
        if self.diagnostics is not None:
            data["diagnostics"] = self.diagnostics.to_dict()
        return data


@dataclass(frozen=True)
class _RawBeatEvent:
    kind: BeatEventKind
    lane: int
    time_ms: float
    source_index: int


@dataclass(frozen=True)
class _BeatRepresentationTimingPoint:
    offset_ms: float
    redline_beat_length_ms: float
    beat_length_ms: float
    meter: int

    @property
    def redline_bpm(self) -> float:
        return 60000.0 / self.redline_beat_length_ms

    @property
    def bpm(self) -> float:
        return 60000.0 / self.beat_length_ms


_KIND_ORDER = {
    BeatEventKind.HOLD_START: 0,
    BeatEventKind.TAP: 1,
    BeatEventKind.HOLD_END: 2,
}


def parse_mania_beat_events(
    beatmap_path: str | Path,
    *,
    snap_denominator: int = DEFAULT_SNAP_DENOMINATOR,
    expected_key_count: int | None = 4,
    timing_canonicalization: str = DEFAULT_BEAT_REPRESENTATION_TIMING_CANONICALIZATION,
    include_diagnostics: bool = False,
    diagnostic_subdivisions: Sequence[int] = DEFAULT_SNAP_DIAGNOSTIC_SUBDIVISIONS,
) -> list[BeatEvent]:
    timing_points = require_red_timing_points(beatmap_path)
    hitobjects = parse_mania_hit_objects(beatmap_path, expected_key_count=expected_key_count)
    return hitobjects_to_beat_events(
        hitobjects,
        timing_points,
        snap_denominator=snap_denominator,
        timing_canonicalization=timing_canonicalization,
        include_diagnostics=include_diagnostics,
        diagnostic_subdivisions=diagnostic_subdivisions,
    )


def hitobjects_to_beat_events(
    hitobjects: Sequence[ManiaHitObject],
    timing_points: Sequence[RedTimingPoint],
    *,
    snap_denominator: int = DEFAULT_SNAP_DENOMINATOR,
    timing_canonicalization: str = DEFAULT_BEAT_REPRESENTATION_TIMING_CANONICALIZATION,
    include_diagnostics: bool = False,
    diagnostic_subdivisions: Sequence[int] = DEFAULT_SNAP_DIAGNOSTIC_SUBDIVISIONS,
) -> list[BeatEvent]:
    snap_denominator = _validate_snap_denominator(snap_denominator)
    subdivisions = _validate_diagnostic_subdivisions(diagnostic_subdivisions)
    sorted_timing_points = _beat_representation_timing_points(
        timing_points,
        canonicalization=timing_canonicalization,
    )
    raw_events = _hitobjects_to_raw_events(hitobjects)
    raw_events.sort(key=lambda event: (event.time_ms, _KIND_ORDER[event.kind], event.lane, event.source_index))
    return [
        _snap_raw_event_to_beat_event(
            event,
            sorted_timing_points,
            snap_denominator=snap_denominator,
            include_diagnostics=include_diagnostics,
            diagnostic_subdivisions=subdivisions,
        )
        for event in raw_events
    ]


def beatmap_to_beat_representation(
    beatmap_path: str | Path,
    *,
    snap_denominator: int = DEFAULT_SNAP_DENOMINATOR,
    expected_key_count: int | None = 4,
    limit_events: int | None = None,
    timing_canonicalization: str = DEFAULT_BEAT_REPRESENTATION_TIMING_CANONICALIZATION,
    include_diagnostics: bool = False,
    diagnostic_subdivisions: Sequence[int] = DEFAULT_SNAP_DIAGNOSTIC_SUBDIVISIONS,
) -> dict[str, Any]:
    beatmap_path = Path(beatmap_path)
    timing_points = require_red_timing_points(beatmap_path)
    hitobjects = parse_mania_hit_objects(beatmap_path, expected_key_count=expected_key_count)
    events = hitobjects_to_beat_events(
        hitobjects,
        timing_points,
        snap_denominator=snap_denominator,
        timing_canonicalization=timing_canonicalization,
        include_diagnostics=include_diagnostics,
        diagnostic_subdivisions=diagnostic_subdivisions,
    )
    selected_events = _limit_events(events, limit_events)
    max_abs_snap_error_ms = max((abs(event.snap_error_ms) for event in events), default=0.0)
    summary: dict[str, Any] = {
        "beatmap": beatmap_path.as_posix(),
        "snap_denominator": int(snap_denominator),
        "timing_canonicalization": timing_canonicalization,
        "red_timing_count": len(timing_points),
        "hitobject_count": len(hitobjects),
        "event_count": len(events),
        "max_abs_snap_error_ms": max_abs_snap_error_ms,
        "truncated_event_count": len(events) - len(selected_events),
        "events": [event.to_dict() for event in selected_events],
    }
    if include_diagnostics:
        summary["snap_diagnostic_subdivisions"] = list(_validate_diagnostic_subdivisions(diagnostic_subdivisions))
        summary["max_relative_error_to_subdivision"] = max(
            (
                event.diagnostics.relative_error_to_subdivision
                for event in events
                if event.diagnostics is not None
            ),
            default=0.0,
        )
    return summary


def _hitobjects_to_raw_events(hitobjects: Sequence[ManiaHitObject]) -> list[_RawBeatEvent]:
    events: list[_RawBeatEvent] = []
    for index, hitobject in enumerate(hitobjects):
        if hitobject.kind == ManiaHitObjectKind.TAP:
            events.append(
                _RawBeatEvent(
                    kind=BeatEventKind.TAP,
                    lane=hitobject.lane,
                    time_ms=hitobject.start_time_ms,
                    source_index=index,
                ),
            )
            continue

        if hitobject.kind == ManiaHitObjectKind.HOLD:
            events.append(
                _RawBeatEvent(
                    kind=BeatEventKind.HOLD_START,
                    lane=hitobject.lane,
                    time_ms=hitobject.start_time_ms,
                    source_index=index,
                ),
            )
            events.append(
                _RawBeatEvent(
                    kind=BeatEventKind.HOLD_END,
                    lane=hitobject.lane,
                    time_ms=hitobject.end_time_ms,
                    source_index=index,
                ),
            )
            continue

        raise ValueError(f"unsupported hitobject kind: {hitobject.kind}")
    return events


def _snap_raw_event_to_beat_event(
    event: _RawBeatEvent,
    timing_points: Sequence[_BeatRepresentationTimingPoint],
    *,
    snap_denominator: int,
    include_diagnostics: bool,
    diagnostic_subdivisions: Sequence[int],
) -> BeatEvent:
    if not math.isfinite(event.time_ms):
        raise ValueError(f"event times must be finite: {event}")

    redline = _red_timing_point_at_sorted(timing_points, event.time_ms)
    raw_beat_offset = (event.time_ms - redline.offset_ms) / redline.beat_length_ms
    beat_offset_numerator = _round_half_up(raw_beat_offset * snap_denominator)
    beat_offset = beat_offset_numerator / snap_denominator
    snapped_time_ms = redline.offset_ms + beat_offset * redline.beat_length_ms
    snap_error_ms = event.time_ms - snapped_time_ms

    return BeatEvent(
        kind=event.kind,
        lane=event.lane,
        time_ms=event.time_ms,
        redline_offset_ms=redline.offset_ms,
        redline_beat_length_ms=redline.redline_beat_length_ms,
        redline_bpm=redline.redline_bpm,
        beat_length_ms=redline.beat_length_ms,
        bpm=redline.bpm,
        beat_offset_numerator=beat_offset_numerator,
        beat_offset_denominator=snap_denominator,
        beat_offset=beat_offset,
        snapped_time_ms=snapped_time_ms,
        snap_error_ms=snap_error_ms,
        diagnostics=(
            _snap_diagnostics(
                raw_beat_offset=raw_beat_offset,
                beat_offset=beat_offset,
                beat_offset_numerator=beat_offset_numerator,
                snap_denominator=snap_denominator,
                subdivisions=diagnostic_subdivisions,
            )
            if include_diagnostics
            else None
        ),
    )


def _red_timing_point_at_sorted(
    timing_points: Sequence[_BeatRepresentationTimingPoint],
    time_ms: float,
) -> _BeatRepresentationTimingPoint:
    offsets = [point.offset_ms for point in timing_points]
    index = bisect_right(offsets, time_ms) - 1
    if index < 0:
        return timing_points[0]
    return timing_points[index]


def _beat_representation_timing_points(
    timing_points: Sequence[RedTimingPoint],
    *,
    canonicalization: str,
) -> list[_BeatRepresentationTimingPoint]:
    if not timing_points:
        raise MissingRedTimingError("cannot convert hitobjects with no red timing points")
    canonicalization = require_timing_canonicalization(canonicalization)
    sorted_points = sorted(timing_points, key=lambda point: point.offset_ms)
    beat_points: list[_BeatRepresentationTimingPoint] = []
    for point in sorted_points:
        validate_red_timing_point(point)
        beat_points.append(
            _BeatRepresentationTimingPoint(
                offset_ms=point.offset_ms,
                redline_beat_length_ms=point.beat_length_ms,
                beat_length_ms=_canonicalized_beat_length_ms(point.beat_length_ms, canonicalization=canonicalization),
                meter=point.meter,
            ),
        )
    return beat_points


def _canonicalized_beat_length_ms(beat_length_ms: float, *, canonicalization: str) -> float:
    if canonicalization == TIMING_CANONICALIZATION_NONE:
        return float(beat_length_ms)
    if canonicalization == TIMING_CANONICALIZATION_BPM_80_160:
        return 60000.0 / canonical_bpm_80_160(60000.0 / float(beat_length_ms))
    raise ValueError(f"unsupported timing canonicalization: {canonicalization!r}")


def _round_half_up(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError(f"cannot round non-finite value: {value}")
    return int(math.floor(value + 0.5))


def _snap_diagnostics(
    *,
    raw_beat_offset: float,
    beat_offset: float,
    beat_offset_numerator: int,
    snap_denominator: int,
    subdivisions: Sequence[int],
) -> SnapDiagnostics:
    subdivision = _snapped_subdivision(
        beat_offset_numerator=beat_offset_numerator,
        snap_denominator=snap_denominator,
        subdivisions=subdivisions,
    )
    snap_error_beats = raw_beat_offset - beat_offset
    return SnapDiagnostics(
        raw_beat_offset=raw_beat_offset,
        snap_error_beats=snap_error_beats,
        abs_snap_error_beats=abs(snap_error_beats),
        subdivision=subdivision,
        subdivision_step_beats=1.0 / subdivision,
        relative_error_to_subdivision=abs(snap_error_beats) * subdivision,
    )


def _snapped_subdivision(
    *,
    beat_offset_numerator: int,
    snap_denominator: int,
    subdivisions: Sequence[int],
) -> int:
    candidates = _validate_diagnostic_subdivisions((*subdivisions, snap_denominator))
    for subdivision in candidates:
        if (beat_offset_numerator * subdivision) % snap_denominator == 0:
            return subdivision
    return snap_denominator


def _validate_snap_denominator(snap_denominator: int) -> int:
    snap_denominator = int(snap_denominator)
    if snap_denominator <= 0:
        raise ValueError(f"snap_denominator must be positive: {snap_denominator}")
    return snap_denominator


def _validate_diagnostic_subdivisions(subdivisions: Sequence[int]) -> tuple[int, ...]:
    if not subdivisions:
        raise ValueError("diagnostic_subdivisions must contain at least one subdivision")
    unique: set[int] = set()
    for subdivision in subdivisions:
        subdivision = int(subdivision)
        if subdivision <= 0:
            raise ValueError(f"diagnostic subdivisions must be positive: {subdivision}")
        unique.add(subdivision)
    return tuple(sorted(unique))


def _parse_diagnostic_subdivisions(value: str) -> tuple[int, ...]:
    try:
        subdivisions = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"diagnostic subdivisions must be comma-separated integers: {value}") from exc
    try:
        return _validate_diagnostic_subdivisions(subdivisions)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _limit_events(events: Sequence[BeatEvent], limit_events: int | None) -> list[BeatEvent]:
    if limit_events is None:
        return list(events)
    limit_events = int(limit_events)
    if limit_events < 0:
        raise ValueError(f"limit_events must be non-negative: {limit_events}")
    return list(events[:limit_events])


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert osu!mania .osu hitobjects to redline-relative snapped beat events.",
    )
    parser.add_argument("beatmaps", nargs="+", help=".osu beatmap path(s) to convert.")
    parser.add_argument(
        "--snap-denominator",
        type=int,
        default=DEFAULT_SNAP_DENOMINATOR,
        help="Number of snap divisions per beat. Default: %(default)s.",
    )
    parser.add_argument(
        "--timing-canonicalization",
        choices=TIMING_CANONICALIZATION_CHOICES,
        default=DEFAULT_BEAT_REPRESENTATION_TIMING_CANONICALIZATION,
        help="Normalize redline beat lengths before snapping. Default: %(default)s.",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Include per-event snap diagnostics in the JSON output.",
    )
    parser.add_argument(
        "--diagnostic-subdivisions",
        type=_parse_diagnostic_subdivisions,
        default=DEFAULT_SNAP_DIAGNOSTIC_SUBDIVISIONS,
        help="Comma-separated subdivision denominators for diagnostics. Default: %(default)s.",
    )
    parser.add_argument(
        "--expected-key-count",
        type=int,
        default=4,
        help="Expected osu!mania key count. Ignored when --any-key-count is set. Default: %(default)s.",
    )
    parser.add_argument(
        "--any-key-count",
        action="store_true",
        help="Accept any osu!mania key count instead of requiring --expected-key-count.",
    )
    parser.add_argument(
        "--limit-events",
        type=int,
        default=None,
        help="Limit printed events per beatmap while keeping full summary counts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    expected_key_count = None if args.any_key_count else args.expected_key_count
    summaries = [
        beatmap_to_beat_representation(
            beatmap,
            snap_denominator=args.snap_denominator,
            expected_key_count=expected_key_count,
            limit_events=args.limit_events,
            timing_canonicalization=args.timing_canonicalization,
            include_diagnostics=args.diagnostics,
            diagnostic_subdivisions=args.diagnostic_subdivisions,
        )
        for beatmap in args.beatmaps
    ]
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
