"""Observe each effective control range without erasing boundary obligations.

Inputs are materialized rows from the beginning of the chart, not a cropped
suffix. Holds retain their actual start/end times. Counts describe generated
structure and experienced actions; they are neither local star ratings nor
semantic style assessments.
"""
from dataclasses import asdict

import numpy as np


def _facts(rows):
    notes, releases, heads, masks = [], [], [], []
    pending, last_head, last_kind, last_release = [None]*4, [None]*4, [None]*4, [None]*4
    for row in rows:
        t = row.time_ms
        mask = 0
        for lane, action in enumerate(row.actions):
            if action in (1, 2):
                hh = np.nan if last_head[lane] is None else t-last_head[lane]
                rh = t-last_release[lane] if last_kind[lane] == 2 and last_release[lane] is not None else np.nan
                heads.append((t, lane, action, hh, rh))
                last_head[lane], last_kind[lane] = t, action
                notes.append([t, t if action == 1 else np.inf, lane, action])
                if action == 2:
                    pending[lane] = len(notes)-1
                mask |= 1 << lane
            elif action == 3:
                index = pending[lane]
                if index is None:
                    raise ValueError('Range evaluation needs the prefix containing every released LN head')
                notes[index][1] = t
                releases.append((t, lane, notes[index][0]))
                pending[lane], last_release[lane] = None, t
        if mask:
            masks.append((t, mask))
    return (np.asarray(notes, float).reshape(-1, 4), np.asarray(heads, float).reshape(-1, 5),
            np.asarray(releases, float).reshape(-1, 3), np.asarray(masks, float).reshape(-1, 2))


def _distribution(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return dict(count=len(values), minimum=float(values.min()) if len(values) else None,
                p05=float(np.quantile(values, .05)) if len(values) else None,
                median=float(np.median(values)) if len(values) else None)


def _observe(facts, start, end):
    notes, heads, releases, masks = facts
    h = heads[(start <= heads[:, 0]) & (heads[:, 0] < end)]
    r = releases[(start <= releases[:, 0]) & (releases[:, 0] < end)]
    m = masks[(start <= masks[:, 0]) & (masks[:, 0] < end), 1].astype(int)
    ln = notes[:, 3] == 2
    entering = int((ln & (notes[:, 0] < start) & (notes[:, 1] >= start)).sum())
    leaving = int((ln & (notes[:, 0] < end) & (notes[:, 1] >= end)).sum())
    new_ln = int((h[:, 2] == 2).sum())
    durations = r[:, 0]-r[:, 2]
    seconds = (end-start)/1000
    release_durations = _distribution(durations)
    release_durations.update(fraction_le40=float((durations <= 40).mean()) if len(r) else None,
                             fraction_le80=float((durations <= 80).mean()) if len(r) else None)
    return dict(start_ms=start, end_ms=end, duration_ms=end-start,
        counts=dict(heads=len(h), head_rows=len(m), tap_heads=len(h)-new_ln,
                    ln_heads=new_ln, releases=len(r), actions=len(h)+len(r),
                    entering_LNs=entering, leaving_LNs=leaving,
                    releases_of_entering_LNs=int((r[:, 2] < start).sum())),
        rates_per_second=dict(heads=len(h)/seconds, releases=len(r)/seconds,
                              actions=(len(h)+len(r))/seconds),
        ln_head_fraction=new_ln/len(h) if len(h) else None,
        heads_per_head_row=len(h)/len(m) if len(m) else None,
        same_column_HH_ms=_distribution(h[:, 3]),
        release_to_next_same_column_head_ms=_distribution(h[:, 4]),
        released_LN_duration_ms=release_durations,
        adjacent_head_mask_reuse=dict(pairs=max(0, len(m)-1),
            identical=sum(int(a == b) for a, b in zip(m, m[1:])),
            repeated_columns=sum((int(a) & int(b)).bit_count() for a, b in zip(m, m[1:])),
            following_heads=sum(int(a).bit_count() for a in m[1:])))


def describe_control_ranges(rows, controls, end_ms, *, window_ms=8000):
    """Return one report per resolved half-open range from zero to ``end_ms``.

    Observation windows stay inside their control range; the last may be
    shorter and is explicitly marked. Recovery clocks read the real prefix.
    Release-duration statistics belong to releases inside the observation,
    including incoming holds; LN proportion counts only heads inside it.
    No observation is assigned a fragment star rating or a style judgment.
    """
    facts = _facts(rows)
    result = []
    for index, span in enumerate(controls.resolved_ranges(0, end_ms)):
        summary = _observe(facts, span.start_ms, span.end_ms)
        summary.update(range_index=index, requested=asdict(span), style_assessment='unreviewed')
        windows = []
        for start in range(span.start_ms, span.end_ms, window_ms):
            stop = min(start+window_ms, span.end_ms)
            observation = _observe(facts, start, stop)
            observation['full_window'] = stop-start == window_ms
            windows.append(observation)
        summary['windows'] = windows
        result.append(summary)
    return result
