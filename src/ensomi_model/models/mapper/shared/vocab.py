from __future__ import annotations

from enum import Enum
from typing import Iterable, Sequence


KEY_COUNT = 4


class LaneAction(str, Enum):
    NONE = "NONE"
    TAP = "TAP"
    HOLD_START = "HOLD_START"
    HOLD_END = "HOLD_END"


FROZEN_EVENT_ACTIONS = (
    LaneAction.NONE,
    LaneAction.TAP,
    LaneAction.HOLD_START,
    LaneAction.HOLD_END,
)
ACTION_TO_CODE = {action: index for index, action in enumerate(FROZEN_EVENT_ACTIONS)}
CODE_TO_ACTION = {index: action for action, index in ACTION_TO_CODE.items()}

DEFAULT_TIME_SHIFT_VALUES_MS = (
    *range(10, 110, 10),
    *range(200, 1001, 100),
    2000,
    3000,
    4000,
)


class MapperTupleVocab:
    """Frozen mapper-event vocabulary from DESIGN.md.

    The mapper vocabulary intentionally contains only special, time-shift, and
    4-lane EVENT tokens. Difficulty and open-LN state are model inputs, not
    token prefixes.
    """

    def __init__(self, time_shift_values_ms: Iterable[int] = DEFAULT_TIME_SHIFT_VALUES_MS) -> None:
        self.time_shift_values_ms = _validate_time_shift_values(tuple(time_shift_values_ms))

        token_names = ["PAD", "BOS", "EOS"]
        token_names.extend(f"TS_{value}" for value in self.time_shift_values_ms)

        event_start = len(token_names)
        self._event_actions_by_id: dict[int, tuple[LaneAction, ...]] = {}
        self._event_id_by_actions: dict[tuple[LaneAction, ...], int] = {}
        for offset, code in enumerate(range(1, 4**KEY_COUNT)):
            actions = self._decode_event_code(code)
            token_names.append(f"EV_{''.join(str(ACTION_TO_CODE[action]) for action in actions)}")
            token_id = event_start + offset
            self._event_actions_by_id[token_id] = actions
            self._event_id_by_actions[actions] = token_id

        self.id_to_token = tuple(token_names)
        self.token_to_id = {name: token_id for token_id, name in enumerate(self.id_to_token)}

        self.pad_id = self.token_to_id["PAD"]
        self.bos_id = self.token_to_id["BOS"]
        self.eos_id = self.token_to_id["EOS"]
        self.time_shift_token_ids = tuple(
            self.token_to_id[f"TS_{value}"] for value in self.time_shift_values_ms
        )
        self.event_token_ids = tuple(range(event_start, len(self.id_to_token)))
        self._time_shift_value_by_id = dict(zip(self.time_shift_token_ids, self.time_shift_values_ms, strict=True))
        self._time_shift_id_by_value = dict(zip(self.time_shift_values_ms, self.time_shift_token_ids, strict=True))
        self._event_onset_weight_by_id = {
            token_id: sum(action in {LaneAction.TAP, LaneAction.HOLD_START} for action in actions)
            for token_id, actions in self._event_actions_by_id.items()
        }

    @property
    def size(self) -> int:
        return len(self.id_to_token)

    def token_name(self, token_id: int) -> str:
        self._require_token_id(token_id)
        return self.id_to_token[token_id]

    def is_time_shift_token(self, token_id: int) -> bool:
        return token_id in self._time_shift_value_by_id

    def is_event_token(self, token_id: int) -> bool:
        return token_id in self._event_actions_by_id

    def time_shift_token_id(self, delta_ms: int) -> int:
        try:
            return self._time_shift_id_by_value[int(delta_ms)]
        except KeyError as exc:
            raise ValueError(f"TS value outside mapper tuple vocabulary: {delta_ms}") from exc

    def time_shift_value(self, token_id: int) -> int:
        try:
            return self._time_shift_value_by_id[int(token_id)]
        except KeyError as exc:
            raise ValueError(f"not a mapper tuple time-shift token: {token_id}") from exc

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
            raise ValueError(f"time-shift delta has no canonical mapper tuple encoding: {delta_ms}")
        return pieces

    def encode_event(self, lane_actions: Sequence[LaneAction | str]) -> int:
        actions = tuple(coerce_lane_action(action) for action in lane_actions)
        if len(actions) != KEY_COUNT:
            raise ValueError(f"EVENT must contain exactly {KEY_COUNT} lane actions: {actions}")
        if all(action == LaneAction.NONE for action in actions):
            raise ValueError("EVENT token cannot represent an all-NONE lane tuple")
        return self._event_id_by_actions[actions]

    def decode_event(self, token_id: int) -> tuple[LaneAction, ...]:
        try:
            return self._event_actions_by_id[int(token_id)]
        except KeyError as exc:
            raise ValueError(f"not a mapper tuple EVENT token: {token_id}") from exc

    def event_onset_weight(self, token_id: int) -> int:
        if not self.is_event_token(token_id):
            return 0
        return self._event_onset_weight_by_id[int(token_id)]

    def _decode_event_code(self, code: int) -> tuple[LaneAction, ...]:
        actions: list[LaneAction] = []
        value = code
        for _ in range(KEY_COUNT):
            actions.append(CODE_TO_ACTION[value % 4])
            value //= 4
        return tuple(actions)

    def _require_token_id(self, token_id: int) -> None:
        if not 0 <= int(token_id) < self.size:
            raise ValueError(f"token id outside mapper tuple vocabulary: {token_id}")


def coerce_lane_action(action: LaneAction | str) -> LaneAction:
    if isinstance(action, LaneAction):
        return action
    value = getattr(action, "value", action)
    try:
        return LaneAction(str(value))
    except ValueError as exc:
        raise ValueError(f"unsupported mapper tuple lane action: {action}") from exc


def _validate_time_shift_values(values: tuple[int, ...]) -> tuple[int, ...]:
    if not values:
        raise ValueError("mapper tuple time-shift vocabulary cannot be empty")
    normalized = tuple(int(value) for value in values)
    if tuple(sorted(normalized)) != normalized:
        raise ValueError(f"time-shift values must be sorted ascending: {values}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"time-shift values must be unique: {values}")
    for value in normalized:
        if value <= 0 or value % 10 != 0:
            raise ValueError(f"time-shift values must be positive 10ms-grid values: {value}")
    return normalized
