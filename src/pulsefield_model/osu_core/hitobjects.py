from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ManiaHitObjectKind(str, Enum):
    TAP = "TAP"
    HOLD = "HOLD"


@dataclass(frozen=True)
class ManiaHitObject:
    start_time_ms: float
    end_time_ms: float
    lane: int
    kind: ManiaHitObjectKind


def parse_mania_hit_objects(beatmap_path: str | Path, *, expected_key_count: int | None = 4) -> list[ManiaHitObject]:
    beatmap_path = Path(beatmap_path)
    section: str | None = None
    mode: int | None = None
    circle_size: float | None = None
    hitobjects: list[ManiaHitObject] = []

    with beatmap_path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue

            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                continue

            if section in {"General", "Difficulty"}:
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if section == "General" and key == "Mode":
                    mode = _parse_int(value, beatmap_path, line)
                elif section == "Difficulty" and key == "CircleSize":
                    circle_size = _parse_float(value, beatmap_path, line)
                continue

            if section != "HitObjects":
                continue

            if mode != 3:
                continue
            if circle_size is None:
                raise ValueError(f"{beatmap_path} is missing CircleSize before [HitObjects].")

            key_count = _key_count_from_circle_size(circle_size)
            if expected_key_count is not None and key_count != expected_key_count:
                raise ValueError(f"{beatmap_path} is not a {expected_key_count}K osu!mania map.")

            hitobjects.append(_parse_hitobject_line(beatmap_path, line, key_count))

    if mode is None:
        raise ValueError(f"{beatmap_path} is missing Mode.")
    if mode != 3:
        raise ValueError(f"{beatmap_path} is not an osu!mania map (Mode={mode}).")
    if circle_size is None:
        raise ValueError(f"{beatmap_path} is missing CircleSize.")

    return hitobjects


def _parse_hitobject_line(beatmap_path: Path, line: str, key_count: int) -> ManiaHitObject:
    parts = line.split(",")
    if len(parts) < 5:
        raise ValueError(f"Malformed hit object in {beatmap_path}: {line}")

    try:
        x = int(parts[0])
        start_time_ms = float(parts[2])
        object_type = int(parts[3])
    except ValueError as exc:
        raise ValueError(f"Malformed hit object in {beatmap_path}: {line}") from exc

    lane = _clamp(int(math.floor(x * key_count / 512.0)), 0, key_count - 1)
    is_hold = (object_type & 128) != 0
    is_tap = (object_type & 1) != 0

    if is_hold:
        if len(parts) < 6:
            raise ValueError(f"Malformed mania hold note in {beatmap_path}: {line}")
        end_time_text = parts[5].split(":", 1)[0]
        try:
            end_time_ms = float(end_time_text)
        except ValueError as exc:
            raise ValueError(f"Malformed mania hold end time in {beatmap_path}: {line}") from exc
        return ManiaHitObject(
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            lane=lane,
            kind=ManiaHitObjectKind.HOLD,
        )

    if is_tap:
        return ManiaHitObject(
            start_time_ms=start_time_ms,
            end_time_ms=start_time_ms,
            lane=lane,
            kind=ManiaHitObjectKind.TAP,
        )

    raise ValueError(f"Unsupported osu!mania hit object type in {beatmap_path}: type={object_type}, line={line}")


def _key_count_from_circle_size(circle_size: float) -> int:
    rounded = round(circle_size)
    if abs(circle_size - rounded) > 1e-6:
        raise ValueError(f"CircleSize must be an integer for osu!mania key count: {circle_size}")
    return max(1, int(rounded))


def _parse_int(value: str, beatmap_path: Path, line: str) -> int:
    try:
        return int(float(value))
    except ValueError as exc:
        raise ValueError(f"Malformed integer field in {beatmap_path}: {line}") from exc


def _parse_float(value: str, beatmap_path: Path, line: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Malformed float field in {beatmap_path}: {line}") from exc


def _clamp(value: int, lower: int, upper: int) -> int:
    return lower if value < lower else upper if value > upper else value
