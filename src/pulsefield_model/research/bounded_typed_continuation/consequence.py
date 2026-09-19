"""Candidate-row energies from exact action consequences and supplied timing.

Future clocks passively advance the immediate post-action state to the next
required onset. They do not predict intervening actions or expose source LN
ends. Earliest release times are possibilities, not committed future events.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..scoped_style_modeling.dataset import ContractError
from .contract import Arm, ROW_ACTIONS, Schedule
from .features import RELATIVE_LANES, TIME_DIM, time_features
from .temporal import pointwise

LANE_DIM = 5 + 7 * TIME_DIM
TIMING_DIM = 1 + 2 * TIME_DIM
WIDTH = 32


def consequence_features(states: Sequence[Schedule], mode: str):
    """Return [query,lane,action,feature] and [query,timing] arrays.

    Lane fields are action one-hot, post-occupancy, current head/head interval,
    current release/head interval, current release's LN age, three post-action
    clocks at next H (head, release, open LN), and H minus earliest possible
    release for a post-action occupied lane. Nonapplicable clocks are missing.
    Illegal hypothetical lane actions are harmless: the owner masks whole rows
    using the unchanged Schedule support after scoring.
    """
    if mode not in ('actions', 'frontier') or not states or any(s.arm != Arm.R1 or s.finished for s in states):
        raise ContractError('Row consequences require active R1 states and actions or frontier mode')
    count = len(states)
    local = np.zeros((count, 4, 4, LANE_DIM), dtype=np.float32)
    local[..., :4] = np.eye(4, dtype=np.float32)
    timing = np.zeros((count, TIMING_DIM), dtype=np.float32)
    if mode == 'actions':
        return local, timing

    now = np.array([s.time_ms for s in states], dtype=np.float64)[:, None, None]
    starts = np.array([s.replay.open_ln_start_ms for s in states], dtype=np.float64)[..., None]
    attacks = np.array([s.replay.last_lane_attack_ms for s in states], dtype=np.float64)[..., None]
    releases = np.array([s.replay.last_lane_release_ms for s in states], dtype=np.float64)[..., None]
    following, onsets, roles, known = [], [], [], []
    for state in states:
        values, index = state.timing.times_ms, state.index
        next_h = state._next_onset(index)
        following.append(values[index + 1] if index + 1 < len(values) else np.nan)
        onsets.append(values[next_h] if next_h is not None else np.nan)
        roles.append(float(index + 1 < len(values) and state.timing.onsets[index + 1]))
        known.append([np.nan if end is None else values[end] for end in state.known_ends])
    next_r = np.array(following, dtype=np.float64)[:, None, None]
    next_h = np.array(onsets, dtype=np.float64)[:, None, None]
    ends = np.array(known, dtype=np.float64)[..., None]
    actions = np.arange(4)[None, None, :]
    head = (actions == 1) | (actions == 2)
    post_attacks = np.where(head, now, attacks)
    post_releases = np.where(actions == 3, now, releases)
    post_starts = np.where(actions == 3, np.nan, np.where(actions == 2, now, starts))
    occupied = ~np.isnan(post_starts)
    earliest_release = np.where(np.isnan(ends), next_r, ends)
    clocks = np.stack((
        np.where(head, now - attacks, np.nan),
        np.where(head, now - releases, np.nan),
        np.where(actions == 3, now - starts, np.nan),
        next_h - post_attacks,
        next_h - post_releases,
        next_h - post_starts,
        np.where(occupied, next_h - earliest_release, np.nan),
    ), axis=-1)
    local[..., 4] = occupied
    local[..., 5:] = time_features(clocks).reshape(count, 4, 4, 7 * TIME_DIM)
    timing[:, :TIME_DIM] = time_features((next_r - now)[:, 0, 0])
    timing[:, TIME_DIM:2 * TIME_DIM] = time_features((next_h - now)[:, 0, 0])
    timing[:, -1] = roles
    return local, timing


class RowConsequence(nn.Module):
    """A zero-initialized mirror-equivariant residual over the complete row.

    The first affine is exactly factorized by lane position. Only sixteen
    lane/action descriptors are projected per query; their small projections
    are gathered and summed for each of the 256 complete candidate rows.
    """
    def __init__(self, hidden: int, mode: str):
        super().__init__()
        if mode not in ('actions', 'frontier'):
            raise ContractError('Unknown row-consequence mode')
        self.mode = mode
        self.lanes = nn.ModuleList(nn.Linear(LANE_DIM, WIDTH, bias=False) for _ in range(4))
        self.timing = nn.Linear(TIMING_DIM, WIDTH)
        self.context = nn.Linear(hidden, WIDTH, bias=False)
        self.output = nn.Linear(WIDTH, 1, bias=False)
        nn.init.zeros_(self.output.weight)
        relative = torch.tensor(RELATIVE_LANES)
        self.register_buffer('relative', relative, persistent=False)
        self.register_buffer('choices', torch.tensor(ROW_ACTIONS)[:, relative].permute(1, 0, 2), persistent=False)

    def score(self, hands, local, timing):
        hidden = pointwise(self.context, hands) + self.timing(timing)[:, None]
        hidden = hidden[:, :, None, :]
        for position, projection in enumerate(self.lanes):
            projected = pointwise(projection, local[:, self.relative[:, position]])
            indices = self.choices[:, :, position][None, :, :, None].expand(len(hands), -1, -1, WIDTH)
            hidden = hidden + projected.gather(2, indices)
        return pointwise(self.output, F.gelu(hidden)).squeeze(-1).mean(1)

    def forward(self, hands, states):
        local, timing = consequence_features(states, self.mode)
        return self.score(hands, hands.new_tensor(local), hands.new_tensor(timing))
