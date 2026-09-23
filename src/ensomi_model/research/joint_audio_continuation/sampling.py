"""Logical event examples with complete waiting-time supervision in bounded queries.

Source event times select training intervals only. Predictor inputs still come
from the query's observed history, exact clocks and audio; no source wait length
or target-selection indicator is added. A logical example can contain several
queries, whose losses must be summed before averaging over logical examples.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import JointChart, JointQuery, query


@dataclass(frozen=True)
class LogicalExample:
    chart: JointChart
    queries: tuple[JointQuery, ...]

    def __post_init__(self):
        if not isinstance(self.queries, tuple) or not self.queries:
            raise ValueError('A logical example requires a nonempty tuple of bounded queries')


def full_wait(chart: JointChart, target_index: int, config) -> LogicalExample:
    """Partition the previous-row-to-target interval without gaps or overlap.

    Every query is at most timing_horizon_ms long. The last query ends at the
    selected target and spans the full horizon when the wait is longer; preceding
    queries are censored survival intervals. BOS starts at -1 ms, preserving a
    target at zero. No row is inserted, and all queries retain the same observed
    source prefix, including open holds with unknown endpoints.
    """
    times = chart.source.rows['time']
    if type(target_index) is not int or not 0 <= target_index < len(times):
        raise ValueError('Full-wait target must index an original source row')
    horizon = config.timing_horizon_ms
    if type(horizon) is not int or horizon <= 0:
        raise ValueError('Full-wait horizon must be a positive integer')
    target = int(times[target_index])
    cursor = int(times[target_index - 1]) if target_index else -1
    tail_start = max(cursor, target - horizon)
    queries = []
    while cursor < tail_start:
        end = min(cursor + horizon, tail_start)
        queries.append(query(chart, cursor, history_limit=config.history_rows, horizon_ms=end - cursor))
        cursor = end
    queries.append(query(chart, cursor, history_limit=config.history_rows, horizon_ms=target - cursor))
    return LogicalExample(chart, tuple(queries))


def sample_example(chart: JointChart, rng, config) -> LogicalExample:
    """Preserve the baseline random draws and optionally complete selected waits.

    With full_wait_supervision=False, every branch returns the original single
    query. When enabled, only BOS and event-prefix choices expand; uniform-time
    and outro choices remain conditional bounded queries. The sampler consumes
    no additional randomness when constructing a complete wait.
    """
    if type(config.full_wait_supervision) is not bool:
        raise ValueError('Full-wait supervision must be an explicit boolean')
    choice = float(rng.random())
    times = chart.source.rows['time']
    target = None
    if choice < .08:
        cursor, target = -1, 0
    elif choice < .78:
        target = int(rng.integers(len(times)))
        cursor = int(times[target - 1]) if target else -1
    elif choice < .95:
        cursor = int(rng.integers(-1, chart.duration_ms))
    else:
        cursor = int(rng.integers(min(int(times[-1]), chart.duration_ms - 1), chart.duration_ms))
    if config.full_wait_supervision and target is not None:
        return full_wait(chart, target, config)
    return LogicalExample(chart, (query(chart, cursor, history_limit=config.history_rows,
                                       horizon_ms=config.timing_horizon_ms),))


def draw_examples(groups, rng, config, count) -> list[LogicalExample]:
    """Sample groups, then their separate arrangements, using the baseline order."""
    examples = []
    for _ in range(count):
        group = groups[int(rng.integers(len(groups)))]
        chart = group[int(rng.integers(len(group)))]
        examples.append(sample_example(chart, rng, config))
    return examples


def full_gap_probes(charts, split, config) -> list[LogicalExample]:
    """Return complete long-gap examples sorted by group, source and target index.

    TRAIN and validation are explicit, separate selections. Other splits are
    rejected. These deterministic coverage/probe examples always include their
    complete waits, independent of the random sampler's supervision flag.
    """
    if split not in ('train', 'validation'):
        raise ValueError('Full-gap probes require an explicit TRAIN or validation split')
    examples = []
    selected = sorted((chart for chart in charts if chart.split == split),
                      key=lambda chart: (chart.group_id, chart.entry['source_sha256']))
    for chart in selected:
        times = chart.source.rows['time']
        gaps = np.diff(np.r_[-1, times])
        for index in np.flatnonzero(gaps > config.timing_horizon_ms):
            examples.append(full_wait(chart, int(index), config))
    return examples


def coverage_examples(charts, config) -> list[LogicalExample]:
    """Select every TRAIN wait longer than the horizon, including late openings."""
    return full_gap_probes(charts, 'train', config)
