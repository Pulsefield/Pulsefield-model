from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence


MIN_VALID_RED_BPM = 20.0
MAX_VALID_RED_BPM = 1000.0
MIN_VALID_RED_BEAT_LENGTH_MS = 60000.0 / MAX_VALID_RED_BPM
MAX_VALID_RED_BEAT_LENGTH_MS = 60000.0 / MIN_VALID_RED_BPM


@dataclass(frozen=True)
class RedTimingPoint:
    offset_ms: float
    beat_length_ms: float
    meter: int = 4


@dataclass(frozen=True)
class RedTimingInvalidCounts:
    nonfinite: int = 0
    nonpositive: int = 0
    implausible: int = 0

    @property
    def total(self) -> int:
        return self.nonfinite + self.nonpositive + self.implausible

    def add_reason(self, reason: str) -> "RedTimingInvalidCounts":
        if reason not in _INVALID_COUNT_FIELDS:
            raise ValueError(f"unknown invalid red timing reason: {reason}")
        return replace(self, **{reason: getattr(self, reason) + 1})


_INVALID_COUNT_FIELDS = frozenset(("nonfinite", "nonpositive", "implausible"))


class MissingRedTimingError(ValueError):
    pass


class InvalidRedTimingError(MissingRedTimingError):
    def __init__(self, beatmap_path: str | Path, counts: RedTimingInvalidCounts) -> None:
        self.counts = counts
        super().__init__(
            f"{beatmap_path} has invalid red timing point(s): "
            f"nonfinite={counts.nonfinite}, "
            f"nonpositive={counts.nonpositive}, "
            f"implausible={counts.implausible}; "
            "valid red timing requires finite offsets and "
            f"{MIN_VALID_RED_BEAT_LENGTH_MS:g} <= beat_length_ms <= "
            f"{MAX_VALID_RED_BEAT_LENGTH_MS:g} "
            f"({MIN_VALID_RED_BPM:g} <= bpm <= {MAX_VALID_RED_BPM:g})",
        )


def parse_red_timing_points(beatmap_path: str | Path) -> list[RedTimingPoint]:
    beatmap_path = Path(beatmap_path)
    section: str | None = None
    timing_points: list[RedTimingPoint] = []
    invalid_counts = RedTimingInvalidCounts()

    with beatmap_path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue

            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                continue

            if section != "TimingPoints":
                continue

            line = line.split("//", 1)[0].strip()
            if not line:
                continue

            try:
                timing_points.append(_parse_timing_line(beatmap_path, line))
            except _NonRedTimingPoint:
                continue
            except _InvalidRedTimingPoint as exc:
                invalid_counts = invalid_counts.add_reason(exc.reason)

    if invalid_counts.total:
        raise InvalidRedTimingError(beatmap_path, invalid_counts)

    return sorted(timing_points, key=lambda point: point.offset_ms)


def require_red_timing_points(beatmap_path: str | Path) -> list[RedTimingPoint]:
    timing_points = parse_red_timing_points(beatmap_path)
    if not timing_points:
        raise MissingRedTimingError(f"{beatmap_path} has no red timing point")
    return timing_points


def red_timing_point_at(timing_points: Sequence[RedTimingPoint], time_ms: float) -> RedTimingPoint:
    if not timing_points:
        raise MissingRedTimingError("cannot look up timing with no red timing points")

    sorted_points = sorted(timing_points, key=lambda point: point.offset_ms)
    for point in sorted_points:
        validate_red_timing_point(point)
    offsets = [point.offset_ms for point in sorted_points]
    index = bisect_right(offsets, time_ms) - 1
    if index < 0:
        return sorted_points[0]
    return sorted_points[index]


def validate_red_timing_point(point: RedTimingPoint) -> None:
    if not math.isfinite(point.offset_ms) or not math.isfinite(point.beat_length_ms):
        raise ValueError(f"red timing points must be finite: {point}")
    if point.beat_length_ms <= 0:
        raise ValueError(f"red timing points must have positive beat lengths: {point}")
    if not is_plausible_red_beat_length_ms(point.beat_length_ms):
        raise ValueError(
            f"implausible red timing beat length: {point}; "
            f"expected {MIN_VALID_RED_BEAT_LENGTH_MS:g} <= beat_length_ms <= "
            f"{MAX_VALID_RED_BEAT_LENGTH_MS:g}",
        )


def is_plausible_red_beat_length_ms(beat_length_ms: float) -> bool:
    return MIN_VALID_RED_BEAT_LENGTH_MS <= beat_length_ms <= MAX_VALID_RED_BEAT_LENGTH_MS


def _parse_timing_line(beatmap_path: Path, line: str) -> RedTimingPoint:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 2:
        raise ValueError(f"Malformed timing point in {beatmap_path}: {line}")

    try:
        offset_ms = float(parts[0])
        beat_length_ms = float(parts[1])
        meter = _parse_optional_int(parts, 2, default=4)
        uninherited = _parse_optional_int(parts, 6, default=1)
    except ValueError as exc:
        raise ValueError(f"Malformed timing point in {beatmap_path}: {line}") from exc

    if uninherited == 0:
        raise _NonRedTimingPoint
    if not math.isfinite(offset_ms) or not math.isfinite(beat_length_ms):
        raise _InvalidRedTimingPoint("nonfinite")
    if beat_length_ms <= 0:
        raise _InvalidRedTimingPoint("nonpositive")
    if not is_plausible_red_beat_length_ms(beat_length_ms):
        raise _InvalidRedTimingPoint("implausible")
    if meter <= 0:
        raise ValueError(f"Malformed red timing point in {beatmap_path}: meter must be positive: {line}")

    return RedTimingPoint(offset_ms=offset_ms, beat_length_ms=beat_length_ms, meter=meter)


def _parse_optional_int(parts: Sequence[str], index: int, *, default: int) -> int:
    if index >= len(parts) or parts[index] == "":
        return default
    return int(float(parts[index]))


class _NonRedTimingPoint(Exception):
    pass


class _InvalidRedTimingPoint(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)
