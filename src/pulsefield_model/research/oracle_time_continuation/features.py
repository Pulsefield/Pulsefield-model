"""History-only time features in canonical self/other, outer/inner coordinates.

Absolute timestamps stay as Python float64 values until subtraction. Missing
clocks have an availability channel and a zero basis, distinct from a known
zero elapsed time. No following gap or source LN endpoint is accepted here.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import torch
from torch import Tensor, nn

from ..scoped_style_modeling.replay import HAND_COLUMNS
from .replay import ExactReplayState

TIME_SCALES_MS = (8., 16., 32., 64., 128., 256., 512., 1024., 2048., 4096.)
TIME_DIM = 3 + 2 * len(TIME_SCALES_MS)


class HistoryStatus(IntEnum):
    BOS = 0
    PRESENT = 1
    TRUNCATED = 2
    PADDING = 3


def clock_features(values: Sequence[float | None], like: Tensor) -> Tensor:
    available = like.new_tensor([x is not None for x in values])
    delta = like.new_tensor([0. if x is None else x for x in values])
    seconds = delta / 1000
    ratio = delta[:, None] / like.new_tensor(TIME_SCALES_MS)
    envelope = torch.rsqrt(1 + ratio.square())
    basis = torch.cat((seconds[:, None], seconds.asinh()[:, None],
                       ratio * envelope, envelope), -1)
    return torch.cat((basis * available[:, None], available[:, None]), -1)


def status_features(status: HistoryStatus, like: Tensor) -> Tensor:
    return like.new_tensor([status == value for value in HistoryStatus])


def relative_lanes(hand: int) -> tuple[int, ...]:
    return HAND_COLUMNS[hand] + HAND_COLUMNS[1 - hand]


def action_features(actions: tuple[int, ...], like: Tensor) -> Tensor:
    return like.new_tensor([[float(actions[lane] == action)
                             for lane in relative_lanes(hand) for action in range(4)]
                            for hand in range(2)])


@dataclass(frozen=True)
class PaceState:
    """The most recent 32 completed positive event gaps, in milliseconds."""

    gaps_ms: tuple[float, ...] = ()

    @property
    def mean_ms(self) -> float | None:
        return sum(self.gaps_ms) / len(self.gaps_ms) if self.gaps_ms else None

    def commit(self, gap_ms: float | None) -> PaceState:
        return self if gap_ms is None else PaceState((self.gaps_ms + (gap_ms,))[-32:])


class HistoryEncoder(nn.Module):
    """Shared hand encoder; both ordered roles and all four lanes remain visible."""

    def __init__(self, hidden: int):
        super().__init__()
        # Four lanes x (LN age, attack, release), two hands x two clocks,
        # previous gap, age of chart, pace mean and pace span.
        self.projection = nn.Sequential(nn.Linear(20 * TIME_DIM + 26, hidden), nn.GELU(),
                                        nn.Linear(hidden, hidden))

    def forward(self, history: ExactReplayState, time_ms: float, pace: PaceState,
                gap_ms: float | None, is_terminal: bool) -> Tensor:
        like = self.projection[0].weight
        clocks = history.clocks_at(time_ms)
        last = action_features(history.last_row.actions, like) if history.last_row else like.new_zeros(2, 16)
        hands = []
        for hand in range(2):
            lanes = relative_lanes(hand)
            times = [value for lane in lanes for value in
                     (clocks.ln_age_ms[lane], clocks.lane_attack_ms[lane], clocks.lane_release_ms[lane])]
            times += [value for side in (hand, 1 - hand) for value in
                      (clocks.hand_attack_ms[side], clocks.hand_release_ms[side])]
            times += [gap_ms, clocks.since_first_row_ms, pace.mean_ms,
                      sum(pace.gaps_ms) if pace.gaps_ms else None]
            status = HistoryStatus.BOS if not history.row_count else HistoryStatus.PRESENT
            facts = like.new_tensor([*(history.occupancy[lane] for lane in lanes),
                                     len(pace.gaps_ms) / 32, is_terminal])
            hands.append(torch.cat((clock_features(times, like).flatten(), last[hand], facts,
                                    status_features(status, like))))
        return self.projection(torch.stack(hands))
