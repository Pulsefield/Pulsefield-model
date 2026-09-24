"""Native activity and action diagnostics that retain early and empty outputs.

Counts and timing bands locate cases for review; they do not assign style,
difficulty or BAD labels. A partial chart has only observed holding time through
its fixed-through clock, never invented LN endpoints.
"""
from __future__ import annotations

import math

import numpy as np

from ..oracle_time_continuation.replay import ExactReplayState, commit
from ..scoped_style_modeling.dataset import ContractError


def prefix_stage(state):
    """Classify actual committed head count and occupancy, including BOS."""
    if not state.row_count:
        return 'bos'
    return ('early' if state.note_count < 30 else 'mature') + ('_held' if any(state.occupancy) else '_free')


def native_diagnostics(rows, *, coverage_ms, duration_ms):
    """Summarize validated complete rows through an explicit observation clock.

    First-30 time includes the complete crossing row. TAP interval denominators
    count consecutive same-lane TAP-to-TAP head relations, not all chart heads.
    Release-to-next-head intervals consume each release once and stay separate.
    Empty charts retain their full observed BOS duration and undefined rates.
    """
    if (not math.isfinite(coverage_ms) or not math.isfinite(duration_ms) or
            not 0 <= coverage_ms <= duration_ms):
        raise ContractError('Native diagnostics require 0 <= coverage <= audio duration')
    state = ExactReplayState()
    stages = {name: dict(duration_ms=0., event_rows=0, heads=0) for name in
              ('bos', 'early_free', 'early_held', 'mature_free', 'mature_held')}
    previous_heads, pending_releases = [None] * 4, [None] * 4
    heads, head_rows, ln_heads = [], [], 0
    closed_lengths, tap_gaps, release_gaps = [], [], []
    lane_time = 0.
    cursor = 0.
    first30 = None
    for row in rows:
        if row.time_ms > coverage_ms:
            raise ContractError('A diagnostic row exceeds fixed chart coverage')
        stage = stages[prefix_stage(state)]
        interval = row.time_ms - cursor
        if interval < 0:
            raise ContractError('Diagnostic rows must increase from the audio origin')
        stage['duration_ms'] += interval
        lane_time += interval * sum(state.occupancy)
        count = sum(action in (1, 2) for action in row.actions)
        stage['event_rows'] += 1
        stage['heads'] += count
        if count:
            heads.extend([float(row.time_ms)] * count)
            head_rows.append(float(row.time_ms))
        for lane, action in enumerate(row.actions):
            if action in (1, 2):
                previous = previous_heads[lane]
                if action == 1 and previous is not None and previous[1] == 1:
                    tap_gaps.append(float(row.time_ms - previous[0]))
                if pending_releases[lane] is not None:
                    release_gaps.append(float(row.time_ms - pending_releases[lane]))
                    pending_releases[lane] = None
                previous_heads[lane] = (row.time_ms, action)
                ln_heads += action == 2
            elif action == 3 and state.open_ln_start_ms[lane] is not None:
                closed_lengths.append(float(row.time_ms - state.open_ln_start_ms[lane]))
                pending_releases[lane] = row.time_ms
        state = commit(state, row, is_terminal=row.time_ms == duration_ms)
        if first30 is None and state.note_count >= 30:
            first30 = float(row.time_ms)
        cursor = float(row.time_ms)
    remaining = coverage_ms - cursor
    stages[prefix_stage(state)]['duration_ms'] += remaining
    lane_time += remaining * sum(state.occupancy)
    times = np.asarray(heads)
    max_heads = (int((np.searchsorted(times, times + 1000., side='left') - np.arange(len(times))).max())
                 if len(times) else 0)
    bands = {str(band): sum(gap <= band for gap in tap_gaps) for band in (5, 10, 20, 40, 80)}
    quantiles = (dict(zip(('minimum', 'median', 'p90', 'maximum'),
                          map(float, np.quantile(closed_lengths, [0., .5, .9, 1.])))) if closed_lengths else None)
    gaps = np.diff(np.asarray([0., *head_rows, float(coverage_ms)]))
    return dict(coverage_ms=float(coverage_ms), duration_ms=float(duration_ms), rows=state.row_count,
        heads=state.note_count, head_rows=len(head_rows), ln_heads=ln_heads,
        ln_head_fraction=ln_heads / len(heads) if heads else None,
        first_head_ms=heads[0] if heads else None, last_head_ms=heads[-1] if heads else None,
        first30_head_row_ms=first30, reached30=first30 is not None,
        max_heads_in_sliding_1s=max_heads, longest_observed_head_gap_ms=float(gaps.max()),
        stage_activity=stages, observed_occupied_lane_ms=lane_time,
        occupied_lane_fraction_of_coverage=lane_time / (4 * coverage_ms) if coverage_ms else None,
        closed_ln_count=len(closed_lengths), closed_ln_duration_ms=quantiles, open_lanes=list(state.occupancy),
        eligible_tap_tap_transitions=len(tap_gaps), tap_tap_counts_le_ms=bands,
        tap_tap_rate_per1000_transitions={key: 1000 * count / len(tap_gaps) if tap_gaps else None
                                        for key, count in bands.items()},
        release_to_next_head_count=len(release_gaps),
        minimum_release_to_next_head_ms=min(release_gaps) if release_gaps else None)
