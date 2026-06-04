from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from pulsefield_model.osu_core.hitobjects import ManiaHitObject, ManiaHitObjectKind


class LaneAction(str, Enum):
    NONE = "NONE"
    TAP = "TAP"
    HOLD_START = "HOLD_START"
    HOLD_END = "HOLD_END"


@dataclass(frozen=True)
class CanonicalTimepoint:
    time_ms: int
    lane_actions: tuple[LaneAction, ...]


@dataclass(frozen=True)
class CanonicalEventBuildResult:
    timepoints: list[CanonicalTimepoint]
    zero_length_hold_normalized_count: int


class NegativeHitObjectTimeError(ValueError):
    pass


def quantize_10ms_half_up(time_ms: float) -> int:
    if time_ms < 0:
        raise NegativeHitObjectTimeError(f"cannot quantize negative time: {time_ms}")
    return int(10 * math.floor((time_ms + 5) / 10))


def ceil_10ms(time_ms: float) -> int:
    if time_ms < 0:
        raise NegativeHitObjectTimeError(f"cannot ceil negative time: {time_ms}")
    return int(10 * math.ceil(time_ms / 10))


def build_canonical_quantized_events(
    hitobjects: Sequence[ManiaHitObject],
    *,
    key_count: int = 4,
) -> CanonicalEventBuildResult:
    primitive_actions: dict[int, dict[int, list[LaneAction]]] = defaultdict(lambda: defaultdict(list))
    zero_length_hold_normalized_count = 0

    for hitobject in hitobjects:
        _validate_hitobject(hitobject, key_count)
        q_start = quantize_10ms_half_up(hitobject.start_time_ms)

        if hitobject.kind == ManiaHitObjectKind.TAP:
            primitive_actions[q_start][hitobject.lane].append(LaneAction.TAP)
            continue

        q_end = quantize_10ms_half_up(hitobject.end_time_ms)
        if q_end <= q_start:
            primitive_actions[q_start][hitobject.lane].append(LaneAction.TAP)
            zero_length_hold_normalized_count += 1
            continue

        primitive_actions[q_start][hitobject.lane].append(LaneAction.HOLD_START)
        primitive_actions[q_end][hitobject.lane].append(LaneAction.HOLD_END)

    timepoints: list[CanonicalTimepoint] = []
    for time_ms in sorted(primitive_actions):
        lane_actions = tuple(
            _merge_lane_actions(primitive_actions[time_ms].get(lane, []), time_ms=time_ms, lane=lane)
            for lane in range(key_count)
        )
        if any(action != LaneAction.NONE for action in lane_actions):
            timepoints.append(CanonicalTimepoint(time_ms=time_ms, lane_actions=lane_actions))

    return CanonicalEventBuildResult(
        timepoints=timepoints,
        zero_length_hold_normalized_count=zero_length_hold_normalized_count,
    )


def _validate_hitobject(hitobject: ManiaHitObject, key_count: int) -> None:
    if hitobject.start_time_ms < 0 or hitobject.end_time_ms < 0:
        raise NegativeHitObjectTimeError(f"hit object has negative time: {hitobject}")
    if not 0 <= hitobject.lane < key_count:
        raise ValueError(f"hit object lane {hitobject.lane} outside 0..{key_count - 1}: {hitobject}")


def _merge_lane_actions(actions: Sequence[LaneAction], *, time_ms: int, lane: int) -> LaneAction:
    if not actions:
        return LaneAction.NONE
    if len(actions) == 1:
        return actions[0]

    raise ValueError(
        f"multiple same-lane actions at {time_ms}ms lane {lane}: {list(actions)}",
    )


__all__ = [
    "CanonicalEventBuildResult",
    "CanonicalTimepoint",
    "LaneAction",
    "NegativeHitObjectTimeError",
    "build_canonical_quantized_events",
    "ceil_10ms",
    "quantize_10ms_half_up",
]
