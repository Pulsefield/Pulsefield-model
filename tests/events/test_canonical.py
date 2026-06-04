from __future__ import annotations

import pytest

from pulsefield_model.events.canonical import LaneAction, build_canonical_quantized_events
from pulsefield_model.osu_core.hitobjects import ManiaHitObject, ManiaHitObjectKind


def test_lane_actions_do_not_include_compound_actions() -> None:
    assert tuple(action.value for action in LaneAction) == (
        "NONE",
        "TAP",
        "HOLD_START",
        "HOLD_END",
    )


def test_same_lane_hold_end_and_start_is_rejected() -> None:
    hitobjects = [
        ManiaHitObject(
            start_time_ms=0,
            end_time_ms=100,
            lane=0,
            kind=ManiaHitObjectKind.HOLD,
        ),
        ManiaHitObject(
            start_time_ms=100,
            end_time_ms=200,
            lane=0,
            kind=ManiaHitObjectKind.HOLD,
        ),
    ]

    with pytest.raises(ValueError, match="multiple same-lane actions at 100ms lane 0"):
        build_canonical_quantized_events(hitobjects)


def test_same_lane_hold_end_and_tap_is_rejected() -> None:
    hitobjects = [
        ManiaHitObject(
            start_time_ms=0,
            end_time_ms=100,
            lane=0,
            kind=ManiaHitObjectKind.HOLD,
        ),
        ManiaHitObject(
            start_time_ms=100,
            end_time_ms=100,
            lane=0,
            kind=ManiaHitObjectKind.TAP,
        ),
    ]

    with pytest.raises(ValueError, match="multiple same-lane actions at 100ms lane 0"):
        build_canonical_quantized_events(hitobjects)
