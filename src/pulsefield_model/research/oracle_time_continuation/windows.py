"""Song-group balanced training draws, independent of output decoding.

The population contains only charts with a complete 30-note seed and a target
suffix. Starts are represented by ranges, not per-row sampling tables. Source
arrays retain their M0 owner; this module does not implement a disk cache.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, dataclass
import math
import random
from typing import Sequence

from ..scoped_style_modeling.dataset import ContractError
from .data import ContinuationSource, SeedSelection

CONTEXT_STRATA = ((0, 64), (64, 512), (512, None))
CONTEXT_LABELS = ("0-63", "64-511", "512+")


@dataclass(frozen=True)
class WindowSamplingPolicy:
    seed: int = 17
    horizons_s: tuple[float, ...] = (1., 4., 16.)

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ContractError("Window sampling seed must be a nonnegative integer")
        if (len(self.horizons_s) != 3 or
                any(isinstance(h, bool) or not math.isfinite(h) or h <= 0 for h in self.horizons_s) or
                not self.horizons_s[0] < self.horizons_s[1] < self.horizons_s[2]):
            raise ContractError("Window horizons must be three increasing, finite positive seconds")
        object.__setattr__(self, "horizons_s", tuple(float(h) for h in self.horizons_s))


@dataclass(frozen=True)
class TrainingWindow:
    """One draw path; source and horizon metadata must never enter model features.

    Indices are zero-based and stop is exclusive. Probability describes the
    complete group/chart/stratum/start/horizon path, not a merged target interval.
    """

    source: ContinuationSource
    seed: SeedSelection
    start: int
    stop: int
    stratum: int
    horizon_index: int
    horizon_s: float
    path_counts: tuple[int, int, int, int, int]

    def __post_init__(self) -> None:
        if self.seed != self.source.minimum_seed() or not self.seed.eligible:
            raise ContractError("Training window requires the source's eligible complete seed")
        if (type(self.start) is not int or type(self.stop) is not int or
                not self.seed.seed_row_count <= self.start < self.stop <= len(self.source.targets)):
            raise ContractError("Training window must contain a nonempty suffix after its complete seed")
        if (type(self.stratum) is not int or not 0 <= self.stratum < 3 or
                type(self.horizon_index) is not int or not 0 <= self.horizon_index < 3 or
                isinstance(self.horizon_s, bool) or not math.isfinite(self.horizon_s) or self.horizon_s <= 0):
            raise ContractError("Training window has invalid stratum or horizon metadata")
        lower, upper = CONTEXT_STRATA[self.stratum]
        extra = self.start - self.seed.seed_row_count
        if extra < lower or (upper is not None and extra >= upper):
            raise ContractError("Training window stratum differs from its prefix length")
        times = self.source.skeleton.times_ms
        if self.stop != bisect_left(times, times[self.start] + self.horizon_s * 1000, lo=self.start + 1):
            raise ContractError("Training window must include every row in its half-open horizon")
        if len(self.path_counts) != 5 or any(type(count) is not int or count <= 0 for count in self.path_counts):
            raise ContractError("Training window requires five positive draw-path population counts")

    @property
    def probability(self) -> float:
        return 1. / math.prod(self.path_counts)

    @property
    def target_rows(self) -> int:
        return self.stop - self.start

    def record(self) -> dict:
        times = self.source.skeleton.times_ms
        return {
            **asdict(self.source.identity), **asdict(self.seed),
            "context_stratum": CONTEXT_LABELS[self.stratum],
            "extra_history_rows": self.start - self.seed.seed_row_count,
            "start": self.start, "stop": self.stop,
            "horizon_index": self.horizon_index, "horizon_s": self.horizon_s,
            "path_counts": dict(zip(("groups", "charts", "strata", "starts", "horizons"), self.path_counts)),
            "draw_path_probability": self.probability,
            "target_rows": self.target_rows, "target_start_ms": times[self.start],
            "target_last_ms": times[self.stop - 1],
            "target_span_ms": times[self.stop - 1] - times[self.start],
            "horizon_end_ms": times[self.start] + 1000 * self.horizon_s,
            "includes_terminal": self.stop == len(times),
        }


@dataclass(frozen=True)
class EligibleChart:
    source: ContinuationSource
    seed: SeedSelection
    starts: tuple[tuple[int, range], ...]


class WindowSampler:
    """Filter once, then draw uniformly at each level without rejection sampling.

    All supplied identities are checked for duplicate sources and song-group
    split conflicts before selecting the requested split. Input ordering does
    not affect seeded draws. No annotations or target actions select starts.
    """

    def __init__(self, sources: Sequence[ContinuationSource], policy: WindowSamplingPolicy = WindowSamplingPolicy(),
                 *, split: str = "train"):
        if split not in ("train", "validation", "test"):
            raise ContractError("Window population requires a train/validation/test split")
        self.policy = policy
        self.split = split
        self.rng = random.Random(policy.seed)
        group_splits, seen, groups, excluded = {}, set(), {}, []
        for source in sorted(sources, key=lambda item: item.identity.source_sha256):
            identity = source.identity
            if identity.source_sha256 in seen:
                raise ContractError("Window population contains a duplicate source identity")
            seen.add(identity.source_sha256)
            previous = group_splits.setdefault(identity.group_id, identity.split)
            if previous != identity.split:
                raise ContractError("A song group cannot occur in multiple splits")
            if identity.split != split:
                continue
            seed = source.minimum_seed()
            if not seed.eligible:
                excluded.append({**asdict(identity), **asdict(seed)})
                continue
            start, end = seed.seed_row_count, len(source.targets)
            strata = tuple((index, range(start + lower, min(end, start + upper) if upper is not None else end))
                           for index, (lower, upper) in enumerate(CONTEXT_STRATA) if start + lower < end)
            groups.setdefault(identity.group_id, []).append(EligibleChart(source, seed, strata))
        if not groups:
            raise ContractError(f"No eligible {split} charts have a 30-note seed and target suffix")
        self.groups = {key: tuple(groups[key]) for key in sorted(groups)}
        self.charts = {chart.source.identity.source_sha256: chart
                       for charts in self.groups.values() for chart in charts}
        self.excluded = tuple(excluded)

    def window(self, source_sha256: str, start: int, horizon_index: int) -> TrainingWindow:
        """Describe a particular population path without advancing the RNG."""
        if source_sha256 not in self.charts:
            raise ContractError("Requested source is outside the eligible window population")
        chart = self.charts[source_sha256]
        if type(horizon_index) is not int or not 0 <= horizon_index < len(self.policy.horizons_s):
            raise ContractError("Window horizon index is outside the sampling policy")
        selected = next(((index, starts) for index, starts in chart.starts if type(start) is int and start in starts), None)
        if selected is None:
            raise ContractError("Target start must follow the full seed and leave a nonempty suffix")
        stratum, starts = selected
        horizon = self.policy.horizons_s[horizon_index]
        times = chart.source.skeleton.times_ms
        stop = bisect_left(times, times[start] + horizon * 1000, lo=start + 1)
        counts = (len(self.groups), len(self.groups[chart.source.identity.group_id]), len(chart.starts),
                  len(starts), len(self.policy.horizons_s))
        return TrainingWindow(chart.source, chart.seed, start, stop, stratum, horizon_index, horizon, counts)

    def draw(self) -> TrainingWindow:
        group = self.rng.choice(tuple(self.groups))
        chart = self.rng.choice(self.groups[group])
        _, starts = self.rng.choice(chart.starts)
        start = self.rng.choice(starts)
        horizon_index = self.rng.randrange(len(self.policy.horizons_s))
        return self.window(chart.source.identity.source_sha256, start, horizon_index)

    def target_probability(self, window: TrainingWindow) -> float:
        """Sum paths that yield this exact chart/start/stop, including clipped horizons."""
        paths = (self.window(window.source.identity.source_sha256, window.start, index)
                 for index in range(len(self.policy.horizons_s)))
        return sum(path.probability for path in paths if path.stop == window.stop)
