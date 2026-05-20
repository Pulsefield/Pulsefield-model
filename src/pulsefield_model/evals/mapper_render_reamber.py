from __future__ import annotations

import importlib
import importlib.util
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pulsefield_model.osu_core.hitobjects import ManiaHitObject, parse_mania_hit_objects


SPAN_SECONDS = 30.0
SPAN_NAMES = (
    "first_30s",
    "middle_30s",
    "last_30s",
    "longest_empty_span",
    "most_repetitive_span",
)

_REAMBER_INSTALL_HINT = (
    "Reamber rendering requires the optional dependency 'reamber'. "
    "Install this repo with the render extra, for example: "
    "`uv sync --extra render --group dev`."
)


@dataclass(frozen=True)
class RenderSpan:
    name: str
    start_ms: float
    end_ms: float

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


def reamber_available() -> bool:
    return importlib.util.find_spec("reamber") is not None


def select_named_spans(
    hitobjects: Sequence[ManiaHitObject],
    *,
    span_seconds: float = SPAN_SECONDS,
) -> dict[str, RenderSpan]:
    if span_seconds <= 0:
        raise ValueError(f"span_seconds must be positive: {span_seconds}")

    span_ms = float(span_seconds) * 1000.0
    chart_end_ms = _chart_end_ms(hitobjects)

    spans = {
        "first_30s": RenderSpan("first_30s", 0.0, span_ms),
        "middle_30s": _span_around_center("middle_30s", chart_end_ms / 2.0, span_ms, chart_end_ms),
        "last_30s": _fixed_duration_span(
            "last_30s",
            max(0.0, chart_end_ms - span_ms),
            span_ms,
            chart_end_ms,
        ),
        "longest_empty_span": _longest_empty_span(hitobjects, span_ms, chart_end_ms),
        "most_repetitive_span": _most_repetitive_span(hitobjects, span_ms, chart_end_ms),
    }

    return {name: spans[name] for name in SPAN_NAMES}


def plan_named_spans(
    beatmap_path: str | Path,
    *,
    span_seconds: float = SPAN_SECONDS,
    expected_key_count: int | None = 4,
) -> dict[str, RenderSpan]:
    hitobjects = parse_mania_hit_objects(beatmap_path, expected_key_count=expected_key_count)
    if not hitobjects:
        raise ValueError(f"cannot plan Reamber spans for an empty osu!mania map: {beatmap_path}")
    return select_named_spans(hitobjects, span_seconds=span_seconds)


def render_named_spans(
    beatmap_path: str | Path,
    output_dir: str | Path,
    *,
    span_seconds: float = SPAN_SECONDS,
    expected_key_count: int | None = 4,
    duration_per_px: float = 5.0,
    padding: int = 40,
    fold_max_height: int | None = 2000,
    overwrite: bool = True,
) -> dict[str, Path]:
    api = _load_reamber_api()
    beatmap_path = Path(beatmap_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spans = plan_named_spans(
        beatmap_path,
        span_seconds=span_seconds,
        expected_key_count=expected_key_count,
    )
    rendered = _render_full_map(
        api,
        beatmap_path,
        duration_per_px=duration_per_px,
        padding=padding,
    )

    image_paths: dict[str, Path] = {}
    for span in spans.values():
        image_path = output_dir / f"{beatmap_path.stem}__{span.name}.png"
        if image_path.exists() and not overwrite:
            raise FileExistsError(f"render target already exists: {image_path}")

        image = _crop_span_image(rendered.image, rendered.playfield, span)
        if fold_max_height is not None:
            image = _fold_image(image, max_height=fold_max_height)
        image.save(image_path)
        image_paths[span.name] = image_path

    return image_paths


render_reamber_spans = render_named_spans


def _load_reamber_api() -> SimpleNamespace:
    try:
        playfield_module = importlib.import_module("reamber.algorithms.playField")
        parts_module = importlib.import_module("reamber.algorithms.playField.parts")
        osu_map_module = importlib.import_module("reamber.osu.OsuMap")
    except ImportError as exc:
        raise RuntimeError(_REAMBER_INSTALL_HINT) from exc

    return SimpleNamespace(
        OsuMap=osu_map_module.OsuMap,
        PlayField=playfield_module.PlayField,
        PFDrawBpm=parts_module.PFDrawBpm,
        PFDrawBeatLines=parts_module.PFDrawBeatLines,
        PFDrawColumnLines=parts_module.PFDrawColumnLines,
        PFDrawNotes=parts_module.PFDrawNotes,
        PFDrawOffsets=parts_module.PFDrawOffsets,
    )


def _render_full_map(
    api: SimpleNamespace,
    beatmap_path: Path,
    *,
    duration_per_px: float,
    padding: int,
) -> SimpleNamespace:
    if duration_per_px <= 0:
        raise ValueError(f"duration_per_px must be positive: {duration_per_px}")

    osu_map = api.OsuMap.read_file(beatmap_path)
    playfield = (
        api.PlayField(m=osu_map, duration_per_px=duration_per_px, padding=padding)
        + api.PFDrawBpm()
        + api.PFDrawBeatLines()
        + api.PFDrawColumnLines()
        + api.PFDrawNotes()
        + api.PFDrawOffsets()
    )
    return SimpleNamespace(playfield=playfield, image=playfield.export())


def _crop_span_image(image: Any, playfield: Any, span: RenderSpan) -> Any:
    from PIL import Image

    if span.end_ms <= span.start_ms:
        raise ValueError(f"span end must be after start: {span}")

    target_height = max(1, int(math.ceil(span.duration_ms / float(playfield.duration_per_px))))
    target = Image.new(
        mode=image.mode,
        size=(image.width, target_height),
        color=getattr(playfield, "background_color", "#000000"),
    )

    source_top = int(math.floor(_time_to_y(playfield, span.end_ms)))
    source_bottom = int(math.ceil(_time_to_y(playfield, span.start_ms)))
    if source_bottom <= source_top:
        source_bottom = source_top + 1

    clipped_top = max(0, source_top)
    clipped_bottom = min(image.height, source_bottom)
    if clipped_bottom <= clipped_top:
        return target

    crop = image.crop((0, clipped_top, image.width, clipped_bottom))
    paste_y = clipped_top - source_top
    target.paste(crop, (0, paste_y))
    return target


def _fold_image(
    image: Any,
    *,
    max_height: int,
    stage_line_width: int = 3,
    stage_line_color: str = "#525252",
) -> Any:
    from PIL import Image

    if max_height <= 0:
        raise ValueError(f"fold_max_height must be positive or None: {max_height}")
    if image.height <= max_height:
        return image

    columns = int(image.height / max_height + 1)
    new_width = columns * image.width + (columns - 1) * stage_line_width
    new_image = Image.new(image.mode, (new_width, max_height), color=stage_line_color)

    for column in range(columns):
        top = image.height - max_height * (column + 1)
        bottom = image.height - max_height * column
        chunk = image.crop((0, max(0, top), image.width, max(0, bottom)))
        paste_y = max(0, -top)
        new_image.paste(chunk, (column * (image.width + stage_line_width), paste_y))

    return new_image


def _time_to_y(playfield: Any, time_ms: float) -> float:
    return (
        float(playfield.canvas_h)
        - ((float(time_ms) - float(playfield.start)) / float(playfield.duration_per_px))
        - float(playfield.hit_height)
    )


def _chart_end_ms(hitobjects: Sequence[ManiaHitObject]) -> float:
    if not hitobjects:
        return 0.0
    return max(float(hitobject.end_time_ms) for hitobject in hitobjects)


def _fixed_duration_span(
    name: str,
    start_ms: float,
    span_ms: float,
    chart_end_ms: float,
) -> RenderSpan:
    max_start = max(0.0, chart_end_ms - span_ms)
    start = min(max(0.0, start_ms), max_start)
    return RenderSpan(name, start, start + span_ms)


def _span_around_center(
    name: str,
    center_ms: float,
    span_ms: float,
    chart_end_ms: float,
) -> RenderSpan:
    return _fixed_duration_span(name, center_ms - span_ms / 2.0, span_ms, chart_end_ms)


def _longest_empty_span(
    hitobjects: Sequence[ManiaHitObject],
    span_ms: float,
    chart_end_ms: float,
) -> RenderSpan:
    if not hitobjects:
        return RenderSpan("longest_empty_span", 0.0, span_ms)

    intervals = sorted((float(obj.start_time_ms), float(obj.end_time_ms)) for obj in hitobjects)
    merged: list[tuple[float, float]] = []
    for start_ms, end_ms in intervals:
        if not merged or start_ms > merged[-1][1]:
            merged.append((start_ms, end_ms))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_ms))

    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start_ms, end_ms in merged:
        if start_ms >= cursor:
            gaps.append((cursor, start_ms))
        cursor = max(cursor, end_ms)
    if chart_end_ms >= cursor:
        gaps.append((cursor, chart_end_ms))

    gap_start, gap_end = max(gaps, key=lambda gap: (gap[1] - gap[0], -gap[0]))
    return _span_around_center(
        "longest_empty_span",
        (gap_start + gap_end) / 2.0,
        span_ms,
        chart_end_ms,
    )


def _most_repetitive_span(
    hitobjects: Sequence[ManiaHitObject],
    span_ms: float,
    chart_end_ms: float,
) -> RenderSpan:
    if not hitobjects:
        return RenderSpan("most_repetitive_span", 0.0, span_ms)

    candidate_starts = _candidate_window_starts(hitobjects, span_ms, chart_end_ms)
    best_start = 0.0
    best_score: tuple[int, int, float] | None = None
    for start_ms in candidate_starts:
        end_ms = start_ms + span_ms
        objects = [
            obj for obj in hitobjects if start_ms <= float(obj.start_time_ms) < end_ms
        ]
        signatures = Counter(_repetition_signature(obj) for obj in objects)
        repeated = sum(count - 1 for count in signatures.values() if count > 1)
        score = (repeated, len(objects), -start_ms)
        if best_score is None or score > best_score:
            best_score = score
            best_start = start_ms

    return _fixed_duration_span("most_repetitive_span", best_start, span_ms, chart_end_ms)


def _candidate_window_starts(
    hitobjects: Sequence[ManiaHitObject],
    span_ms: float,
    chart_end_ms: float,
) -> list[float]:
    max_start = max(0.0, chart_end_ms - span_ms)
    starts = {0.0, max_start}
    for obj in hitobjects:
        obj_start = float(obj.start_time_ms)
        starts.add(obj_start)
        starts.add(obj_start - span_ms / 2.0)
        starts.add(obj_start - span_ms)
    return sorted(min(max(0.0, start), max_start) for start in starts)


def _repetition_signature(hitobject: ManiaHitObject) -> tuple[int, str, int]:
    duration_ms = max(0.0, float(hitobject.end_time_ms) - float(hitobject.start_time_ms))
    return (
        int(hitobject.lane),
        str(hitobject.kind.value),
        int(round(duration_ms / 10.0)),
    )


__all__ = [
    "RenderSpan",
    "SPAN_NAMES",
    "SPAN_SECONDS",
    "plan_named_spans",
    "reamber_available",
    "render_named_spans",
    "render_reamber_spans",
    "select_named_spans",
]
