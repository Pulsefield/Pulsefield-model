"""Exploratory linked-head representation and an untruncated endpoint scorer."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

HANDS = ((0, 1), (3, 2))
SCALES = (50., 100., 250., 500., 1000., 2000., 4000., 16000.)


def time_tensor(rows):
    """Own contiguous float64 times; structured disk rows have a 12-byte stride."""
    return torch.from_numpy(rows['time'].copy())


def linked_rows(rows):
    """Return onset roles, head endpoint labels, and prior object commitments.

    Endpoints belong to supervision. At row i, pending[i] contains only heads
    strictly before i; no simultaneous head's endpoint can enter its features.
    In generated execution these commitments must come from predicted objects.
    """
    times, actions = rows['time'], rows['actions']
    assert len(rows) and np.all(np.diff(times) > 0)
    assert np.all((actions >= 0) & (actions <= 3))
    ends = np.full(actions.shape, -1, dtype=np.int64)
    opened = np.full(4, -1, dtype=np.int64)
    for i, row in enumerate(actions):
        assert np.any(row)
        for c, action in enumerate(row):
            if action in (1, 2):
                assert opened[c] == -1
            if action == 2:
                opened[c] = i
            elif action == 3:
                assert opened[c] >= 0 and opened[c] < i
                ends[opened[c], c] = i
                opened[c] = -1
    assert np.all(opened == -1)
    pending = np.full(actions.shape, -1, dtype=np.int64)
    plan = np.full(4, -1, dtype=np.int64)
    for i, row in enumerate(actions):
        pending[i] = plan
        for c, action in enumerate(row):
            if action == 2:
                assert ends[i, c] > i
                plan[c] = ends[i, c]
            elif action == 3:
                assert plan[c] == i
                plan[c] = -1
    onsets = ((actions == 1) | (actions == 2)).any(axis=1)
    return onsets, ends, pending


def reconstruct(actions, ends):
    """Materialize only head rows and the releases their objects require."""
    result = np.where((actions == 1) | (actions == 2), actions, 0).astype(np.uint8)
    for i, c in np.argwhere(actions == 2):
        end = int(ends[i, c])
        assert i < end < len(actions) and result[end, c] == 0
        result[end, c] = 3
    return result


def age_features(values):
    return [feature for value in values for feature in
            ((0., 0., 0.) if value is None else
             (1., math.log1p(max(0., value) / 1000.), max(0., value) / (max(0., value) + 250.)))]


def context_features(hidden, query, actions, pending_end_times, lane):
    """Mirror-consistent history, chosen head row, and previous planned ends.

    No endpoint label argument exists. The caller owns that label separately.
    """
    hand = 0 if lane < 2 else 1
    order = HANDS[hand] + HANDS[1 - hand]
    values = [float(lane == HANDS[hand][0]), float(lane == HANDS[hand][1])]
    for c in order:
        values.extend(float(int(actions[c]) == a) for a in range(4))
    values += age_features([None if pending_end_times[c] is None else
                            pending_end_times[c] - query.time_ms for c in order])
    clocks = query.clocks
    for field in (clocks.ln_age_ms, clocks.lane_attack_ms, clocks.lane_release_ms):
        values += age_features([field[c] for c in order])
    values += age_features([clocks.previous_row_ms, clocks.since_first_row_ms])
    return torch.cat((hidden[hand].detach().flatten().cpu(),
                      hidden[1 - hand].detach().flatten().cpu(), torch.tensor(values, dtype=torch.float32)))


def candidate_features(times, onsets, index):
    """Every strictly future candidate, with timing and supplied onset roles only."""
    times = torch.as_tensor(times, dtype=torch.float32)
    onsets = torch.as_tensor(onsets, dtype=torch.bool)
    assert 0 <= index < len(times) - 1 and times.ndim == onsets.ndim == 1
    future = times[index + 1:]
    delta = future - times[index]
    assert bool((delta > 0).all())
    rank = torch.arange(1, len(future) + 1, dtype=torch.float32)
    before = times[index:-1]
    after = torch.cat((times[index + 2:], times[-1:]))
    cols = [delta / (delta + scale) for scale in SCALES]
    cols += [torch.log1p(delta / 1000.), torch.asinh(delta / 1000.),
             torch.log1p(rank), 1. / rank, rank / (rank + 8.), rank / (rank + 32.)]
    for gap in (future - before, after - future):
        cols += [gap / (gap + 250.), torch.log1p(gap / 1000.)]
    cols += [onsets[index + 1:].float(), (rank == len(future)).float()]
    value = torch.stack(cols, -1)
    assert value.shape == (len(future), 20) and bool(torch.isfinite(value).all())
    return value


class EndpointHead(nn.Module):
    def __init__(self, context_dim, width=64):
        super().__init__()
        self.context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, 128),
                                     nn.SiLU(), nn.Linear(128, width))
        self.candidate = nn.Sequential(nn.Linear(20, width), nn.SiLU(), nn.Linear(width, width))
        self.prior = nn.Linear(20, 1)
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)
        nn.init.zeros_(self.prior.weight)
        nn.init.zeros_(self.prior.bias)
        self.width = width

    def forward(self, context, candidates, valid, *, use_context):
        query = self.context(context if use_context else torch.zeros_like(context))
        scores = (self.candidate(candidates) * query[:, None]).sum(-1) / math.sqrt(self.width)
        scores = scores + self.prior(candidates).squeeze(-1)
        assert bool(valid.any(-1).all()) and bool(torch.isfinite(scores[valid]).all())
        return scores.masked_fill(~valid, -torch.inf).log_softmax(-1)


def collate(cases, charts):
    candidates = [candidate_features(charts[x['source']]['times'], charts[x['source']]['onsets'], x['index']) for x in cases]
    width = max(len(x) for x in candidates)
    values = torch.zeros(len(cases), width, 20)
    valid = torch.zeros(len(cases), width, dtype=torch.bool)
    target = torch.tensor([x['end_index'] - x['index'] - 1 for x in cases])
    for i, feature in enumerate(candidates):
        assert 0 <= target[i] < len(feature)
        values[i, :len(feature)] = feature
        valid[i, :len(feature)] = True
    return torch.stack([x['context'] for x in cases]), values, valid, target
