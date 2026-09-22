from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OsuBeatmapMetadata:
    audio_filename: str | None
    audio_lead_in: int | None
    preview_time: int | None
    mode: int | None
    title: str | None
    artist: str | None
    creator: str | None
    version: str | None
    beatmap_id: int | None
    beatmap_set_id: int | None
    hp_drain_rate: float | None
    circle_size: float | None
    overall_difficulty: float | None
    key_count: int | None


def _to_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _derive_key_count(mode: int | None, circle_size: float | None) -> int | None:
    if mode != 3 or circle_size is None:
        return None
    rounded = round(circle_size)
    if abs(circle_size - rounded) > 1e-6:
        return None
    return int(rounded)


def parse_osu_metadata(beatmap_path: str | Path) -> OsuBeatmapMetadata:
    beatmap_path = Path(beatmap_path)
    values: dict[tuple[str, str], str] = {}
    section: str | None = None

    with beatmap_path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue

            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                continue

            if section not in {"General", "Metadata", "Difficulty"}:
                continue

            if ":" not in line:
                continue

            key, value = line.split(":", 1)
            values[(section, key)] = value.strip()

    mode = _to_int(values.get(("General", "Mode")))
    circle_size = _to_float(values.get(("Difficulty", "CircleSize")))

    return OsuBeatmapMetadata(
        audio_filename=values.get(("General", "AudioFilename")),
        audio_lead_in=_to_int(values.get(("General", "AudioLeadIn"))),
        preview_time=_to_int(values.get(("General", "PreviewTime"))),
        mode=mode,
        title=values.get(("Metadata", "Title")),
        artist=values.get(("Metadata", "Artist")),
        creator=values.get(("Metadata", "Creator")),
        version=values.get(("Metadata", "Version")),
        beatmap_id=_to_int(values.get(("Metadata", "BeatmapID"))),
        beatmap_set_id=_to_int(values.get(("Metadata", "BeatmapSetID"))),
        hp_drain_rate=_to_float(values.get(("Difficulty", "HPDrainRate"))),
        circle_size=circle_size,
        overall_difficulty=_to_float(values.get(("Difficulty", "OverallDifficulty"))),
        key_count=_derive_key_count(mode, circle_size),
    )
