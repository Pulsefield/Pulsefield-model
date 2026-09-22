"""Song-group balanced training draws with optional known-time gap strata.

The population contains only charts with a complete 30-note seed and a target
suffix. Starts are represented by ranges, not per-row sampling tables. Source
handles may use disk storage; this sampler retains no parsed source buffers.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass, replace
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
    windows_per_chart: int = 1
    gap_sampling_probability: float = 0.
    gap_thresholds_ms: tuple[float, ...] = (2000., 8000., 32000.)
    gap_context_rows: int = 32

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ContractError("Window sampling seed must be a nonnegative integer")
        if type(self.windows_per_chart) is not int or not 1 <= self.windows_per_chart <= 8:
            raise ContractError("windows_per_chart must be an integer in [1,8]")
        if (len(self.horizons_s) != 3 or
                any(isinstance(h, bool) or not math.isfinite(h) or h <= 0 for h in self.horizons_s) or
                not self.horizons_s[0] < self.horizons_s[1] < self.horizons_s[2]):
            raise ContractError("Window horizons must be three increasing, finite positive seconds")
        object.__setattr__(self, "horizons_s", tuple(float(h) for h in self.horizons_s))
        if (isinstance(self.gap_sampling_probability, bool) or not math.isfinite(self.gap_sampling_probability) or
                not 0 <= self.gap_sampling_probability <= 1):
            raise ContractError('gap_sampling_probability must be in [0,1]')
        if (len(self.gap_thresholds_ms) != 3 or
                any(isinstance(t, bool) or not math.isfinite(t) or t <= 0 for t in self.gap_thresholds_ms) or
                not self.gap_thresholds_ms[0] < self.gap_thresholds_ms[1] < self.gap_thresholds_ms[2]):
            raise ContractError('gap_thresholds_ms must contain three increasing positive thresholds')
        if type(self.gap_context_rows) is not int or not 1 <= self.gap_context_rows <= 128:
            raise ContractError('gap_context_rows must be an integer in [1,128]')
        object.__setattr__(self, 'gap_thresholds_ms', tuple(float(t) for t in self.gap_thresholds_ms))


@dataclass(frozen=True)
class TrainingWindow:
    """One draw path; source and horizon metadata must never enter model features.

    Indices are zero-based and stop is exclusive. Probability describes the
    complete route/group/chart/stratum/start/horizon path, not a merged target
    interval. The optional gap route changes the training distribution.
    """

    source: ContinuationSource
    seed: SeedSelection
    start: int
    stop: int
    stratum: int
    horizon_index: int
    horizon_s: float
    path_counts: tuple[int, int, int, int, int]
    route_probability: float = 1.
    gap_band: int | None = None
    gap_index: int | None = None

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
        if (isinstance(self.route_probability, bool) or not math.isfinite(self.route_probability) or
                not 0 <= self.route_probability <= 1):
            raise ContractError('Window route probability must be in [0,1]')
        if self.gap_band is not None or self.gap_index is not None:
            if (type(self.gap_band) is not int or not 0 <= self.gap_band < 3 or
                    type(self.gap_index) is not int or not self.start <= self.gap_index < self.stop or
                    self.gap_index >= len(times) - 1):
                raise ContractError('Gap sampling must include its nonterminal boundary in the target window')

    @property
    def probability(self) -> float:
        return self.route_probability / math.prod(self.path_counts)

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
            "path_counts": dict(zip(("groups", "charts", "strata" if self.gap_band is None else "gaps",
                                      "starts", "horizons"), self.path_counts)),
            "draw_path_probability": self.probability,
            "target_rows": self.target_rows, "target_start_ms": times[self.start],
            "target_last_ms": times[self.stop - 1],
            "target_span_ms": times[self.stop - 1] - times[self.start],
            "horizon_end_ms": times[self.start] + 1000 * self.horizon_s,
            "includes_terminal": self.stop == len(times),
            **(dict(sampling_route='base' if self.gap_band is None else 'gap',
                    route_probability=self.route_probability, gap_band=self.gap_band, gap_index=self.gap_index)
               if self.route_probability != 1 or self.gap_band is not None else {}),
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
        self.gaps, self.gap_groups = {}, {}
        if policy.gap_sampling_probability:
            self._index_gaps()

    def _index_gaps(self):
        groups = {}
        for sha, chart in self.charts.items():
            bands = [[], [], []]
            times = chart.source.skeleton.times_ms
            previous = times[chart.seed.seed_row_count]
            for index in range(chart.seed.seed_row_count, len(times) - 1):
                following = times[index + 1]
                band = bisect_right(self.policy.gap_thresholds_ms, following - previous) - 1
                if band >= 0:
                    bands[band].append(index)
                previous = following
            self.gaps[sha] = tuple(tuple(band) for band in bands)
            for band, indices in enumerate(bands):
                if indices:
                    groups.setdefault(band, {}).setdefault(chart.source.identity.group_id, []).append(chart)
        if not groups:
            raise ContractError('No eligible skeleton gaps for the requested sampling mixture')
        self.gap_groups = {band: {group: tuple(charts) for group, charts in sorted(population.items())}
                           for band, population in sorted(groups.items())}

    def gap_starts(self, chart: EligibleChart, index: int) -> range:
        """Starts within the context-row cap whose full horizon includes this boundary."""
        times = chart.source.skeleton.times_ms
        lower = max(chart.seed.seed_row_count, index - self.policy.gap_context_rows + 1,
                    bisect_right(times, times[index] - self.policy.horizons_s[-1] * 1000))
        return range(lower, index + 1)

    def gap_window(self, source_sha256: str, band: int, index: int, start: int) -> TrainingWindow:
        """Describe one known-time stratified draw path without advancing the RNG."""
        if source_sha256 not in self.charts:
            raise ContractError('Requested source is outside the eligible window population')
        chart = self.charts[source_sha256]
        if (type(band) is not int or type(index) is not int or band not in self.gap_groups or
                index not in self.gaps[source_sha256][band]):
            raise ContractError('Requested gap is outside the eligible time stratum')
        starts = self.gap_starts(chart, index)
        if type(start) is not int or start not in starts:
            raise ContractError('Gap target start is outside the bounded context')
        groups = self.gap_groups[band]
        counts = (len(groups), len(groups[chart.source.identity.group_id]), len(self.gaps[source_sha256][band]),
                  len(starts), 1)
        return replace(self.window(source_sha256, start, 2), path_counts=counts,
                       route_probability=self.policy.gap_sampling_probability / len(self.gap_groups),
                       gap_band=band, gap_index=index)

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
        return TrainingWindow(chart.source, chart.seed, start, stop, stratum, horizon_index, horizon, counts,
                              route_probability=1 - self.policy.gap_sampling_probability)

    def draw(self) -> TrainingWindow:
        if self.policy.gap_sampling_probability and self.rng.random() < self.policy.gap_sampling_probability:
            band, chart = self._choose_gap_chart()
            return self._draw_gap_chart(band, chart)
        group = self.rng.choice(tuple(self.groups))
        chart = self.rng.choice(self.groups[group])
        return self._draw_chart(chart)

    def _draw_chart(self, chart) -> TrainingWindow:
        _, starts = self.rng.choice(chart.starts)
        start = self.rng.choice(starts)
        horizon_index = self.rng.randrange(len(self.policy.horizons_s))
        return self.window(chart.source.identity.source_sha256, start, horizon_index)

    def _choose_gap_chart(self):
        band = self.rng.choice(tuple(self.gap_groups))
        group = self.rng.choice(tuple(self.gap_groups[band]))
        return band, self.rng.choice(self.gap_groups[band][group])

    def _draw_gap_chart(self, band, chart):
        index = self.rng.choice(self.gaps[chart.source.identity.source_sha256][band])
        start = self.rng.choice(self.gap_starts(chart, index))
        return self.gap_window(chart.source.identity.source_sha256, band, index, start)

    def draw_batch(self, count: int) -> tuple[TrainingWindow, ...]:
        """Share chart draws while preserving the chosen mixture's expected risk.

        Shared-chart windows are correlated and sorted by start for prefix reuse.
        Route/group/chart/stratum/start/horizon path probabilities remain marginal
        probabilities before sorting, not per-position order-statistic densities
        or a claim of independent draws within the batch.
        """
        if type(count) is not int or count <= 0:
            raise ContractError('A sampled batch requires a positive window count')
        if self.policy.windows_per_chart == 1:
            return tuple(self.draw() for _ in range(count))
        windows = []
        while len(windows) < count:
            if self.policy.gap_sampling_probability and self.rng.random() < self.policy.gap_sampling_probability:
                band, chart = self._choose_gap_chart()
                draw = lambda: self._draw_gap_chart(band, chart)
            else:
                group = self.rng.choice(tuple(self.groups))
                chart = self.rng.choice(self.groups[group])
                draw = lambda: self._draw_chart(chart)
            cohort = [draw() for _ in range(min(self.policy.windows_per_chart, count - len(windows)))]
            windows.extend(sorted(cohort, key=lambda window: window.start))
        return tuple(windows)

    def gap_population(self) -> list[dict]:
        """Describe only the available time strata, outside model inputs."""
        thresholds = self.policy.gap_thresholds_ms
        return [dict(band=band, lower_ms=thresholds[band],
                     upper_ms=thresholds[band + 1] if band + 1 < len(thresholds) else None,
                     groups=len(groups), charts=sum(len(charts) for charts in groups.values()),
                     gaps=sum(len(self.gaps[chart.source.identity.source_sha256][band])
                              for charts in groups.values() for chart in charts))
                for band, groups in self.gap_groups.items()]

    def target_probability(self, window: TrainingWindow) -> float:
        """Sum paths that yield this exact chart/start/stop, including clipped horizons."""
        paths = (self.window(window.source.identity.source_sha256, window.start, index)
                 for index in range(len(self.policy.horizons_s)))
        result = sum(path.probability for path in paths if path.stop == window.stop)
        if self.policy.gap_sampling_probability:
            sha = window.source.identity.source_sha256
            chart = self.charts[sha]
            for band in self.gap_groups:
                for index in self.gaps[sha][band]:
                    if window.start in self.gap_starts(chart, index):
                        path = self.gap_window(sha, band, index, window.start)
                        if path.stop == window.stop:
                            result += path.probability
        return result
