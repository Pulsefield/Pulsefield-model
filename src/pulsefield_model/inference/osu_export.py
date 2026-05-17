from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from pulsefield_model.events.canonical import (
    CanonicalTimepoint,
    LaneAction as CanonicalLaneAction,
)
from pulsefield_model.osu_core.export import format_hitobjects
from pulsefield_model.models.mapper.shared.generation import transition_carry_state
from pulsefield_model.models.mapper.shared.replay import LNCarryState
from pulsefield_model.models.mapper.shared.vocab import (
    KEY_COUNT,
    LaneAction as MapperLaneAction,
    MapperTupleVocab,
)
from pulsefield_model.timing.schema import FittedTimingGrid, TimingSegment


@dataclass(frozen=True)
class OsuExportMetadata:
    audio_filename: str = "audio.mp3"
    title: str = "Pulsefield Mapper Stream"
    artist: str = "Unknown Artist"
    creator: str = "Mapperatorinator"
    version: str = "Mapper v1 8s inference"
    difficulty: float | None = None
    hp_drain_rate: float = 5.0
    overall_difficulty: float = 5.0
    timing_offset_ms: float = 0.0
    beat_length_ms: float = 500.0


def decode_mapper_tokens_to_timepoints(
    tokens: Sequence[int],
    vocab: MapperTupleVocab,
    *,
    start_ms: int = 0,
    end_ms: int = 8000,
) -> list[CanonicalTimepoint]:
    start_ms = int(start_ms)
    end_ms = int(end_ms)
    if end_ms <= start_ms:
        raise ValueError(f"end_ms must be after start_ms: {start_ms}..{end_ms}")
    if start_ms % 10 != 0 or end_ms % 10 != 0:
        raise ValueError(f"mapper decode window must be 10ms-aligned: {start_ms}..{end_ms}")

    state = LNCarryState.closed(start_ms)
    timepoints: list[CanonicalTimepoint] = []
    special_token_ids = {vocab.pad_id, vocab.bos_id, vocab.eos_id}

    for raw_token_id in tokens:
        token_id = int(raw_token_id)
        if token_id in special_token_ids:
            continue

        if vocab.is_event_token(token_id):
            lane_actions = tuple(
                _to_canonical_lane_action(action) for action in vocab.decode_event(token_id)
            )
            state_after = transition_carry_state(
                state,
                token_id,
                vocab=vocab,
                write_start_ms=start_ms,
                write_end_ms=end_ms,
            )
            timepoints.append(
                CanonicalTimepoint(time_ms=state.current_ms, lane_actions=lane_actions),
            )
            state = state_after
            continue

        state = transition_carry_state(
            state,
            token_id,
            vocab=vocab,
            write_start_ms=start_ms,
            write_end_ms=end_ms,
        )

    return timepoints


def iter_osu_lines(
    timepoints: Sequence[CanonicalTimepoint],
    *,
    metadata: OsuExportMetadata | None = None,
    timing_grid: FittedTimingGrid | None = None,
) -> Iterator[str]:
    metadata = OsuExportMetadata() if metadata is None else metadata
    key_count = KEY_COUNT

    yield "osu file format v14"
    yield ""
    yield "[General]"
    yield f"AudioFilename:{metadata.audio_filename}"
    yield "AudioLeadIn:0"
    yield "PreviewTime:-1"
    yield "Countdown:0"
    yield "SampleSet:Soft"
    yield "StackLeniency:0.7"
    yield "Mode:3"
    yield "LetterboxInBreaks:0"
    yield "SpecialStyle:0"
    yield "WidescreenStoryboard:0"
    yield ""
    yield "[Metadata]"
    yield f"Title:{metadata.title}"
    yield f"TitleUnicode:{metadata.title}"
    yield f"Artist:{metadata.artist}"
    yield f"ArtistUnicode:{metadata.artist}"
    yield f"Creator:{metadata.creator}"
    yield f"Version:{metadata.version}"
    yield "BeatmapID:0"
    yield "BeatmapSetID:-1"
    yield ""
    yield "[Difficulty]"
    yield f"HPDrainRate:{_format_osu_number(metadata.hp_drain_rate)}"
    yield f"CircleSize:{key_count}"
    yield f"OverallDifficulty:{_format_osu_number(metadata.overall_difficulty)}"
    yield "ApproachRate:5"
    yield "SliderMultiplier:1"
    yield "SliderTickRate:1"
    yield ""
    yield "[TimingPoints]"
    for segment in _timing_segments(metadata, timing_grid):
        yield _format_timing_point(segment)
    yield ""
    yield "[HitObjects]"
    yield from format_hitobjects(timepoints, key_count=key_count)


def format_osu_export(
    timepoints: Sequence[CanonicalTimepoint],
    *,
    metadata: OsuExportMetadata | None = None,
    timing_grid: FittedTimingGrid | None = None,
    generated_tokens: Sequence[int] | None = None,
    vocab: MapperTupleVocab | None = None,
) -> str:
    del generated_tokens, vocab
    return "\n".join(iter_osu_lines(timepoints, metadata=metadata, timing_grid=timing_grid)) + "\n"


def _to_canonical_lane_action(action: MapperLaneAction) -> CanonicalLaneAction:
    try:
        return CanonicalLaneAction[action.name]
    except KeyError as exc:
        raise ValueError(f"unsupported mapper lane action for osu export: {action}") from exc


def _format_osu_number(value: float) -> str:
    return f"{float(value):g}"


def _timing_segments(
    metadata: OsuExportMetadata,
    timing_grid: FittedTimingGrid | None,
) -> Sequence[TimingSegment]:
    if timing_grid is not None:
        return timing_grid.segments
    return (
        TimingSegment(
            offset_ms=metadata.timing_offset_ms,
            beat_length_ms=metadata.beat_length_ms,
            meter=4,
        ),
    )


def _format_timing_point(segment: TimingSegment) -> str:
    return (
        f"{_format_osu_number(segment.offset_ms)},"
        f"{_format_osu_number(segment.beat_length_ms)},"
        f"{int(segment.meter)},2,0,100,1,0"
    )
