"""Vectorized R0/R1 support with the scalar Schedule predicate as its oracle.

This is the same lane legality, known-end, terminal and future-onset contract
used by atomic commits. It changes the computation of a batch mask, not the
set or order of candidate rows. No floating-point model operation belongs here.
"""
from collections.abc import Sequence

import numpy as np

from ..scoped_style_modeling.dataset import ContractError
from .contract import Arm, ROW_ACTIONS, Schedule

_ACTIONS = np.array(ROW_ACTIONS, dtype=np.int8)
_HEADS = ((_ACTIONS == 1) | (_ACTIONS == 2)).any(-1)
_NONEMPTY = _ACTIONS.any(-1)
for _array in (_ACTIONS, _HEADS, _NONEMPTY):
    _array.flags.writeable = False


def row_supports(states: Sequence[Schedule]):
    """Return [query,256] booleans in ROW_ACTIONS order, including no-event.

    Completed schedules have empty support. O1 predicts a different vocabulary
    and must use its own head/endpoint support. Callers bound the query batch;
    temporary lane masks have shape [query,256,4].
    """
    if any(state.arm not in (Arm.R0, Arm.R1) for state in states):
        raise ContractError('Batched row support is R0/R1-only')
    if not states:
        return np.empty((0, len(ROW_ACTIONS)), dtype=np.bool_)
    occupied = np.array([s.replay.occupancy for s in states], dtype=np.bool_)[:, None, :]
    ends = np.array([[-1 if end is None else end for end in s.known_ends] for s in states], dtype=np.int64)
    indices = np.array([s.index for s in states], dtype=np.int64)
    terminal = np.array([s.index == len(s.timing.times_ms) - 1 for s in states])
    finished = np.array([s.finished for s in states])
    typed = np.array([s.arm == Arm.R1 for s in states])
    roles = np.array([False if s.arm == Arm.R0 or s.finished else s.timing.onsets[s.index] for s in states])
    following = [s._next_onset(s.index) for s in states]
    next_h = np.array([-1 if index is None else index for index in following], dtype=np.int64)
    actions = _ACTIONS[None]

    allowed = np.where(occupied, (actions == 0) | (actions == 3), actions != 3)
    required = np.where(ends == indices[:, None], 3, 0)
    allowed &= (ends < 0)[:, None, :] | (actions == required[:, None, :])
    allowed &= ~terminal[:, None, None] | np.where(occupied, actions == 3, actions != 2)
    valid = allowed.all(-1)
    valid &= np.where(typed[:, None], _HEADS[None] == roles[:, None], _NONEMPTY[None])

    after = np.where(actions == 3, False, (actions == 2) | occupied)
    # A known end must precede H. Unknown holds need an intervening R on which
    # to release; a close at H itself cannot permit a same-lane attack there.
    releasable = ((ends >= 0) & (ends < next_h[:, None])) | ((ends < 0) & (indices[:, None] + 1 < next_h[:, None]))
    room = ~after.all(-1) | (after & releasable[:, None, :]).any(-1)
    valid &= (next_h < 0)[:, None] | room
    valid[finished] = False
    return valid
