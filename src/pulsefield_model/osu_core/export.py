from __future__ import annotations

from typing import Sequence

from pulsefield_model.events.canonical import CanonicalTimepoint, LaneAction


def format_hitobjects(timepoints: Sequence[CanonicalTimepoint], *, key_count: int = 4) -> list[str]:
    open_holds: dict[int, int] = {}
    lines: list[str] = []
    for timepoint in sorted(timepoints, key=lambda item: item.time_ms):
        if len(timepoint.lane_actions) != key_count:
            raise ValueError(f"timepoint must contain {key_count} lane actions: {timepoint}")
        for lane, action in enumerate(timepoint.lane_actions):
            if action == LaneAction.NONE:
                continue
            if action == LaneAction.TAP:
                if lane in open_holds:
                    raise ValueError(f"TAP while hold is open at {timepoint.time_ms}ms lane {lane}")
                lines.append(_format_tap(lane, timepoint.time_ms, key_count=key_count))
            elif action == LaneAction.HOLD_START:
                if lane in open_holds:
                    raise ValueError(f"HOLD_START while hold is open at {timepoint.time_ms}ms lane {lane}")
                open_holds[lane] = timepoint.time_ms
            elif action == LaneAction.HOLD_END:
                if lane not in open_holds:
                    raise ValueError(f"HOLD_END without open hold at {timepoint.time_ms}ms lane {lane}")
                lines.append(
                    _format_hold(
                        lane,
                        open_holds.pop(lane),
                        timepoint.time_ms,
                        key_count=key_count,
                    ),
                )
            else:
                raise ValueError(f"unsupported osu!mania export action: {action}")

    if open_holds:
        lanes = ", ".join(str(lane) for lane in sorted(open_holds))
        raise ValueError(f"unclosed hold lane(s): {lanes}")
    return sorted(lines, key=_hitobject_sort_key)


def _format_tap(lane: int, time_ms: int, *, key_count: int) -> str:
    return f"{_lane_x(lane, key_count)},192,{time_ms},1,0,0:0:0:0:"


def _format_hold(lane: int, start_time_ms: int, end_time_ms: int, *, key_count: int) -> str:
    if end_time_ms <= start_time_ms:
        raise ValueError(f"hold end must be after start: {start_time_ms} -> {end_time_ms}")
    return f"{_lane_x(lane, key_count)},192,{start_time_ms},128,0,{end_time_ms}:0:0:0:0:"


def _lane_x(lane: int, key_count: int) -> int:
    if not 0 <= lane < key_count:
        raise ValueError(f"lane outside 0..{key_count - 1}: {lane}")
    return int((lane + 0.5) * 512 / key_count)


def _hitobject_sort_key(line: str) -> tuple[int, int]:
    parts = line.split(",", 3)
    return int(parts[2]), int(parts[0])


__all__ = ["format_hitobjects"]
