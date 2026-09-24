"""Causal row-count composition and exact within-count layout normalization."""
import numpy as np
import torch
from torch import nn

from ..bounded_typed_continuation.contract import ROW_ACTIONS
from ..bounded_typed_continuation.features import TIME_DIM, time_features
from ..bounded_typed_continuation.temporal import FiniteTemporal, TemporalConfig
from ..scoped_style_modeling.dataset import ContractError

ROW_COUNTS = tuple((sum(a in (1, 2) for a in row), row.count(2), row.count(3))
                   for row in ROW_ACTIONS)
COUNT_MARKS = tuple(sorted(set(ROW_COUNTS)))
COUNT_TOKEN_DIM = TIME_DIM + 3
COUNT_STATE_DIM = 5 * TIME_DIM


def count_tokens(times, previous_times, actions):
    """Elapsed row time and count triples, with no column or future-tail data."""
    actions = np.asarray(actions).reshape(-1, 4)
    counts = np.stack((np.isin(actions, (1, 2)).sum(-1), (actions == 2).sum(-1),
                       (actions == 3).sum(-1)), -1) / 4.
    gaps = np.asarray(times, np.float64) - np.asarray(previous_times, np.float64)
    features = np.concatenate((time_features(gaps), counts), -1).astype(np.float32)
    return np.repeat(features[:, None], 2, axis=1)


def count_state(starts_ms, previous_times, times):
    """Read committed LN ages without lane identity or cumulative row totals."""
    now = np.asarray(times, np.float64)
    starts = np.asarray(starts_ms, np.float64).reshape(-1, 4)
    ages = np.sort(now[:, None] - starts, axis=-1)
    elapsed = now - np.asarray(previous_times, np.float64)
    return time_features(np.concatenate((elapsed[:, None], ages), -1)).reshape(len(now), COUNT_STATE_DIM)


class RowCountModel(nn.Module):
    def __init__(self, audio_width, preview_width, expansion):
        super().__init__()
        self.temporal = FiniteTemporal(TemporalConfig(COUNT_TOKEN_DIM, 32, 4, expansion))
        self.readout = nn.Sequential(nn.Linear(audio_width + preview_width + COUNT_STATE_DIM + 32, 128),
                                     nn.GELU(), nn.Linear(128, len(COUNT_MARKS)))
        index = torch.tensor([COUNT_MARKS.index(mark) for mark in ROW_COUNTS])
        self.register_buffer('row_mark', index, persistent=False)
        self.register_buffer('members', torch.arange(len(COUNT_MARKS))[:, None] == index, persistent=False)

    def logits(self, audio, history, preview, state):
        if (history.shape != (*audio.shape[:-1], 2, 32) or
                state.shape != (*audio.shape[:-1], COUNT_STATE_DIM)):
            raise ContractError('Count composition requires its own history and committed LN/row ages')
        return self.readout(torch.cat((audio, history.mean(-2), preview, state), -1))

    def compose(self, row_scores, count_logits, legal):
        """Preserve every legal row while replacing its count-group marginal.

        Inactive groups use finite dummy normalizers to avoid undefined backward
        derivatives of an all-negative-infinity logsumexp. Their mark mass and
        every illegal row remain exactly zero.
        """
        members = legal[..., None, :] & self.members
        active = members.any(-1)
        grouped = row_scores[..., None, :].masked_fill(~members, -torch.inf)
        grouped = torch.where(active[..., None], grouped, torch.zeros_like(grouped))
        normalizer = grouped.logsumexp(-1)
        marks = count_logits.masked_fill(~active, -torch.inf).log_softmax(-1)
        result = row_scores - normalizer[..., self.row_mark] + marks[..., self.row_mark]
        return result.masked_fill(~legal, -torch.inf)
