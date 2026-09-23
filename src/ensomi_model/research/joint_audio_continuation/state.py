"""Exact history features and complete-row support without supplied future timing."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..bounded_typed_continuation.contract import ROW_ACTIONS
from ..bounded_typed_continuation.features import RELATIVE_LANES, time_features
from ..oracle_time_continuation.features import TIME_DIM
from ..oracle_time_continuation.replay import ExactReplayState, legal_rows as replay_legal_rows
from ..scoped_style_modeling.dataset import ContractError

BASE_QUERY_DIM = 22 * TIME_DIM + 23
_ROW_INDICES = {actions: index for index, actions in enumerate(ROW_ACTIONS)}


def exact_features(
    replays: Sequence[ExactReplayState], times_ms: Sequence[float],
) -> np.ndarray:
    """Return float32 [batch, 2, BASE_QUERY_DIM] at each candidate clock.

    Ordering matches the R1 query prefix. All four endpoint channels remain
    unavailable; neither future event times nor crop boundaries are inputs.
    Clock queries may equal the last row time, but cannot precede it. Reading
    clocks does not commit rows or alter the replay. Mismatched batch lengths
    and invalid clocks raise ContractError.
    """
    if len(replays) != len(times_ms):
        raise ContractError('Exact features need one query time per replay')
    if not replays:
        return np.empty((0, 2, BASE_QUERY_DIM), dtype=np.float32)
    time_rows, fact_rows = [], []
    for replay, time_ms in zip(replays, times_ms):
        clocks = replay.clocks_at(time_ms)
        for hand, lanes in enumerate(RELATIVE_LANES):
            times = [value for lane in lanes for value in
                     (clocks.ln_age_ms[lane], clocks.lane_attack_ms[lane], clocks.lane_release_ms[lane])]
            times += [value for side in (hand, 1 - hand) for value in
                      (clocks.hand_attack_ms[side], clocks.hand_release_ms[side])]
            times += [clocks.previous_row_ms, clocks.since_first_row_ms, None, None, None, None]
            time_rows.append(times)
            last = ([0.] * 16 if replay.last_row is None else
                    np.eye(4)[np.asarray(replay.last_row.actions)[list(lanes)]].ravel().tolist())
            fact_rows.append([*last, *(replay.occupancy[lane] for lane in lanes),
                              np.log1p(replay.row_count) / 10., np.log1p(replay.note_count) / 10.,
                              not replay.row_count])
    clocks = time_features(time_rows).reshape(len(replays), 2, 22 * TIME_DIM)
    facts = np.asarray(fact_rows, dtype=np.float32).reshape(len(replays), 2, 23)
    return np.concatenate((clocks, facts), axis=-1)


def legal_rows(
    replays: Sequence[ExactReplayState], terminal: Sequence[bool],
) -> np.ndarray:
    """Return bool [batch, 256] in ROW_ACTIONS order, excluding all-EMPTY.

    The canonical replay contract owns simultaneous occupancy transitions.
    Only a true terminal requires every held lane to CLOSE and forbids new
    LN_START. Crop ends receive False. Completed replays, non-boolean flags
    and mismatched batch lengths raise ContractError.
    """
    if len(replays) != len(terminal):
        raise ContractError('Row support needs one terminal flag per replay')
    result = np.zeros((len(replays), len(ROW_ACTIONS)), dtype=np.bool_)
    for index, (replay, is_terminal) in enumerate(zip(replays, terminal)):
        if not isinstance(is_terminal, (bool, np.bool_)):
            raise ContractError('The true song terminal flag must be a boolean')
        actions = replay_legal_rows(replay, is_terminal=bool(is_terminal))
        result[index, [_ROW_INDICES[row] for row in actions]] = True
    return result
