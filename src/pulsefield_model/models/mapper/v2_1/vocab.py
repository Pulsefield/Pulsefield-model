from __future__ import annotations

from collections.abc import Iterable

from pulsefield_model.models.mapper.shared.vocab import (
    DEFAULT_TIME_SHIFT_VALUES_MS,
    KEY_COUNT,
    LaneAction,
    coerce_lane_action,
)


SPARSE_LANE_ACTIONS = (
    LaneAction.TAP,
    LaneAction.HOLD_START,
    LaneAction.HOLD_END,
)
CANONICAL_LANE_ORDER = tuple(range(KEY_COUNT))


class MapperV21Vocab:
    """Mapper v2.1 sparse lane-action vocabulary.

    This follows osuT5's mania shape: present notes are explicit lane/action
    events, and per-lane NONE is represented by not emitting a lane token.
    """

    def __init__(self, time_shift_values_ms: Iterable[int] = DEFAULT_TIME_SHIFT_VALUES_MS) -> None:
        self.time_shift_values_ms = _validate_time_shift_values(tuple(time_shift_values_ms))

        token_names = ["PAD", "BOS", "EOS"]
        token_names.extend(f"TS_{value}" for value in self.time_shift_values_ms)

        self._lane_action_by_id: dict[int, tuple[int, LaneAction]] = {}
        self._lane_action_id_by_lane_action: dict[tuple[int, LaneAction], int] = {}
        lane_action_start = len(token_names)
        for lane_index in range(KEY_COUNT):
            for action in SPARSE_LANE_ACTIONS:
                token_names.append(f"LANE_{lane_index + 1}_{action.value}")
                token_id = len(token_names) - 1
                self._lane_action_by_id[token_id] = (lane_index, action)
                self._lane_action_id_by_lane_action[(lane_index, action)] = token_id

        self.id_to_token = tuple(token_names)
        self.token_to_id = {name: token_id for token_id, name in enumerate(self.id_to_token)}

        self.pad_id = self.token_to_id["PAD"]
        self.bos_id = self.token_to_id["BOS"]
        self.eos_id = self.token_to_id["EOS"]
        self.time_shift_token_ids = tuple(
            self.token_to_id[f"TS_{value}"] for value in self.time_shift_values_ms
        )
        self.lane_action_token_ids = tuple(range(lane_action_start, len(self.id_to_token)))
        self._time_shift_value_by_id = dict(zip(self.time_shift_token_ids, self.time_shift_values_ms, strict=True))
        self._time_shift_id_by_value = dict(zip(self.time_shift_values_ms, self.time_shift_token_ids, strict=True))
        self._lane_action_token_ids_by_lane = tuple(
            tuple(self._lane_action_id_by_lane_action[(lane_index, action)] for action in SPARSE_LANE_ACTIONS)
            for lane_index in range(KEY_COUNT)
        )

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    def token_name(self, token_id: int) -> str:
        self._require_token_id(token_id)
        return self.id_to_token[int(token_id)]

    def is_time_shift_token(self, token_id: int) -> bool:
        return int(token_id) in self._time_shift_value_by_id

    def time_shift_token_id(self, delta_ms: int) -> int:
        try:
            return self._time_shift_id_by_value[int(delta_ms)]
        except KeyError as exc:
            raise ValueError(f"TS value outside mapper v2.1 vocabulary: {delta_ms}") from exc

    def time_shift_value(self, token_id: int) -> int:
        try:
            return self._time_shift_value_by_id[int(token_id)]
        except KeyError as exc:
            raise ValueError(f"not a mapper v2.1 time-shift token: {token_id}") from exc

    def decompose_time_shift_delta(self, delta_ms: int) -> list[int]:
        delta_ms = int(delta_ms)
        if delta_ms < 0:
            raise ValueError(f"time-shift delta must be non-negative: {delta_ms}")
        if delta_ms % 10 != 0:
            raise ValueError(f"time-shift delta must be on the 10ms grid: {delta_ms}")
        if delta_ms == 0:
            return []

        remaining = delta_ms
        pieces: list[int] = []
        for value in sorted(self.time_shift_values_ms, reverse=True):
            while remaining >= value:
                pieces.append(value)
                remaining -= value
        if remaining != 0:
            raise ValueError(f"time-shift delta has no canonical mapper v2.1 encoding: {delta_ms}")
        return pieces

    def is_lane_action_token(self, token_id: int) -> bool:
        return int(token_id) in self._lane_action_by_id

    def lane_action_token_id(self, lane_index: int, action: LaneAction | str) -> int:
        lane = _validate_lane_index(lane_index)
        lane_action = coerce_lane_action(action)
        if lane_action == LaneAction.NONE:
            raise ValueError("mapper v2.1 has no LANE_X_NONE token; omit the lane action instead")
        try:
            return self._lane_action_id_by_lane_action[(lane, lane_action)]
        except KeyError as exc:
            raise ValueError(f"unsupported mapper v2.1 lane action: {lane_action}") from exc

    def decode_lane_action(self, token_id: int) -> tuple[int, LaneAction]:
        try:
            return self._lane_action_by_id[int(token_id)]
        except KeyError as exc:
            raise ValueError(f"not a mapper v2.1 lane-action token: {token_id}") from exc

    def canonical_lane_action_token_ids(self, lane_actions: Iterable[LaneAction | str]) -> tuple[int, ...]:
        actions = tuple(coerce_lane_action(action) for action in lane_actions)
        if len(actions) != KEY_COUNT:
            raise ValueError(f"lane_actions must contain exactly {KEY_COUNT} lanes: {actions}")
        return tuple(
            self.lane_action_token_id(lane_index, actions[lane_index])
            for lane_index in CANONICAL_LANE_ORDER
            if actions[lane_index] != LaneAction.NONE
        )

    def validate_canonical_lane_action_run(self, token_ids: Iterable[int]) -> tuple[int, ...]:
        tokens = tuple(int(token_id) for token_id in token_ids)
        previous_lane = -1
        for token_id in tokens:
            lane_index, _ = self.decode_lane_action(token_id)
            if lane_index <= previous_lane:
                raise ValueError(
                    "same-time lane-action tokens must use strictly ascending lane order "
                    "with no duplicate lane before the next TS_* token"
                )
            previous_lane = lane_index
        return tokens

    def lane_action_token_ids_for_lane(self, lane_index: int) -> tuple[int, ...]:
        return self._lane_action_token_ids_by_lane[_validate_lane_index(lane_index)]

    def event_onset_weight(self, token_id: int) -> int:
        if not self.is_lane_action_token(token_id):
            return 0
        _, action = self.decode_lane_action(token_id)
        return int(action in {LaneAction.TAP, LaneAction.HOLD_START})

    def _require_token_id(self, token_id: int) -> None:
        if not 0 <= int(token_id) < self.size:
            raise ValueError(f"token id outside mapper v2.1 vocabulary: {token_id}")


def _validate_lane_index(lane_index: int) -> int:
    lane = int(lane_index)
    if not 0 <= lane < KEY_COUNT:
        raise ValueError(f"lane_index must be in [0,{KEY_COUNT - 1}], got {lane_index}")
    return lane


def _validate_time_shift_values(values: tuple[int, ...]) -> tuple[int, ...]:
    if not values:
        raise ValueError("mapper v2.1 time-shift vocabulary cannot be empty")
    normalized = tuple(int(value) for value in values)
    if tuple(sorted(normalized)) != normalized:
        raise ValueError(f"time-shift values must be sorted ascending: {values}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"time-shift values must be unique: {values}")
    for value in normalized:
        if value <= 0 or value % 10 != 0:
            raise ValueError(f"time-shift values must be positive 10ms-grid values: {value}")
    return normalized
