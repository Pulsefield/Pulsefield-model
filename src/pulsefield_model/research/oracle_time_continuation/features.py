"""Physical time and committed action features in relative hand coordinates.

Absolute timestamps stay as Python float64 values until subtraction. Missing
clocks have an availability channel and a zero basis, distinct from a known
zero elapsed time. The separate skeleton encoder accepts only unlabeled times;
the history encoder never accepts a source LN endpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Sequence

import torch
from torch import Tensor, nn

from ..scoped_style_modeling.replay import HAND_COLUMNS
from .replay import ExactReplayState

if TYPE_CHECKING:
    from .engine import PredictionInput

TIME_SCALES_MS = (8., 16., 32., 64., 128., 256., 512., 1024., 2048., 4096.)
TIME_DIM = 3 + 2 * len(TIME_SCALES_MS)
TIME_FEATURE_SCHEMA = "bounded-seconds-asinh-v2"


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
    # A linear chart-age/hold-age channel makes unseen long durations dominate
    # the network. Keep a bounded signed ratio and an invertible logarithmic
    # channel, while exact replay retains the unmodified physical milliseconds.
    bounded_seconds = seconds / (1 + seconds.abs())
    basis = torch.cat((bounded_seconds[:, None], seconds.asinh()[:, None],
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


class SkeletonTimeEncoder(nn.Module):
    """Ordered offsets and successive gaps from a bounded known time skeleton.

    The shared output has no lane identity. Missing positions near the true
    skeleton end use unavailable clocks, never a target-window boundary.
    """

    def __init__(self, hidden: int, rows: int):
        super().__init__()
        self.rows = rows
        self.projection = nn.Sequential(nn.Linear(2 * rows * TIME_DIM, hidden), nn.GELU(),
                                        nn.Linear(hidden, hidden))
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, offsets: Sequence[tuple[float, ...]]) -> Tensor:
        from ..scoped_style_modeling.dataset import ContractError

        clocks = []
        for values in offsets:
            if len(values) > self.rows:
                raise ContractError('Future timing exceeds the configured skeleton context')
            gaps = tuple(b - a for a, b in zip((0., *values), values))
            clocks.extend((*values, *([None] * (self.rows - len(values))),
                           *gaps, *([None] * (self.rows - len(values)))))
        like = self.projection[0].weight
        features = clock_features(clocks, like).reshape(len(offsets), 2 * self.rows * TIME_DIM)
        return self.projection(features)


class ClockReadout(nn.Module):
    """Shared hand-pair unary residual from exact pre-row facts and known times.

    The output bypasses learned history compression without changing its cache.
    It receives no target actions or source LN endpoints. A zero final layer
    preserves the existing distribution when this optional path is introduced.
    """

    def __init__(self, hidden: int, lookahead_rows: int):
        super().__init__()
        self.lookahead_rows = lookahead_rows
        # 18 elapsed clocks, previous actions/occupancy/terminal, then time-only lookahead.
        width = (18 + 2 * lookahead_rows) * TIME_DIM + 21
        self.projection = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 16))
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, queries: Sequence[PredictionInput]) -> Tensor:
        from ..scoped_style_modeling.dataset import ContractError

        like = self.projection[0].weight
        elapsed, facts, future = [], [], []
        for query in queries:
            clocks = query.clocks
            history = query.history
            offsets = query.future_offsets_ms
            if len(offsets) > self.lookahead_rows:
                raise ContractError('Clock readout received excess future timing')
            gaps = tuple(b - a for a, b in zip((0., *offsets), offsets))
            padding = [None] * (self.lookahead_rows - len(offsets))
            future.extend((*offsets, *padding, *gaps, *padding))
            for hand in range(2):
                lanes = relative_lanes(hand)
                elapsed.extend(value for lane in lanes for value in
                               (clocks.ln_age_ms[lane], clocks.lane_attack_ms[lane], clocks.lane_release_ms[lane]))
                elapsed.extend(value for side in (hand, 1 - hand) for value in
                               (clocks.hand_attack_ms[side], clocks.hand_release_ms[side]))
                elapsed.extend((clocks.previous_row_ms, clocks.since_first_row_ms))
                last = ([float(history.last_row.actions[lane] == action) for lane in lanes for action in range(4)]
                        if history.last_row else [0.] * 16)
                facts.append([*last, *(history.occupancy[lane] for lane in lanes), query.is_terminal])
        count = len(queries)
        inputs = torch.cat((clock_features(elapsed, like).reshape(count, 2, 18 * TIME_DIM),
                            like.new_tensor(facts).reshape(count, 2, 21),
                            clock_features(future, like).reshape(count, 1, 2 * self.lookahead_rows * TIME_DIM)
                            .expand(count, 2, -1)), -1)
        return self.projection(inputs)
