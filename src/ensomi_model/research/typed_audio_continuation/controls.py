"""Scoped, independently optional semantic conditions shared by both factors.

Stars refer to whole-chart labels during the initial fit. Applying that request
to a smaller scope requests comparable local organization, not a local SR.
Style names/ordinal scales belong to the supplied annotation vocabulary.
"""
from dataclasses import dataclass, field

import numpy as np

from ..bounded_typed_continuation.features import TIME_DIM, time_features

SHARED_SCOPE = 'shared-scope-v1'
PER_FIELD_SCOPE = 'per-field-scope-v1'


@dataclass(frozen=True)
class ControlSpan:
    start_ms: int
    end_ms: int
    stars: float | None = None
    ln_fraction: float | None = None
    style: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ControlSchedule:
    spans: tuple = ()
    style_names: tuple = ()

    @property
    def width(self):
        return self.width_for(SHARED_SCOPE)

    def width_for(self, encoding):
        n = 2+len(self.style_names)
        if encoding == SHARED_SCOPE:
            return 2*n + 2*TIME_DIM
        if encoding == PER_FIELD_SCOPE:
            return 2*n + 2*n*TIME_DIM
        raise ValueError(f'Unknown control encoding: {encoding}')

    def at(self, times, *, encoding=SHARED_SCOPE):
        """Resolve values with either shared or independently owned scope clocks.

        Values and known bits have the same prefix in both encodings. Per-field
        clocks follow the last nonmissing assignment to each attribute. After
        an override expires, the earlier assignment and its original extent
        resume; this does not reset physical history or create a new quota.
        """
        result = np.zeros((len(times), self.width_for(encoding)), np.float32)
        n = 2 + len(self.style_names)
        for j, t in enumerate(times):
            begin = end = None
            clocks = [None]*(2*n)
            for span in self.spans:
                if not span.start_ms <= t < span.end_ms:
                    continue
                unknown = span.style.keys() - set(self.style_names)
                if unknown:
                    raise ValueError(f'Style has no trained vocabulary slot: {unknown}')
                values = [None if span.stars is None else (span.stars-4)/2,
                          None if span.ln_fraction is None else 2*span.ln_fraction-1,
                          *(span.style.get(k) for k in self.style_names)]
                for k, value in enumerate(values):
                    if value is not None:
                        result[j, k] = value
                        result[j, n+k] = 1
                        clocks[2*k:2*k+2] = (t-span.start_ms, span.end_ms-t)
                begin, end = span.start_ms, span.end_ms
            if encoding == SHARED_SCOPE:
                clocks = [None if begin is None else t-begin, None if end is None else end-t]
            result[j, 2*n:] = time_features(clocks).reshape(-1)
        return result

    def resolved_ranges(self, start_ms, end_ms):
        """Partition an observation range at every declared control boundary.

        Values use the same partial-override semantics as ``at``. Boundaries
        remain separate even when adjacent values match: a requested scope is
        an evaluation unit, not an instruction to pool its observations.
        """
        edges = sorted({start_ms, end_ms, *(t for s in self.spans
            for t in (s.start_ms, s.end_ms) if start_ms < t < end_ms)})
        result = []
        for begin, end in zip(edges, edges[1:]):
            stars = fraction = None
            style = {}
            for span in self.spans:
                if span.start_ms <= begin < span.end_ms:
                    unknown = span.style.keys() - set(self.style_names)
                    if unknown:
                        raise ValueError(f'Style has no trained vocabulary slot: {unknown}')
                    if span.stars is not None:
                        stars = span.stars
                    if span.ln_fraction is not None:
                        fraction = span.ln_fraction
                    style.update({k: v for k, v in span.style.items() if v is not None})
            result.append(ControlSpan(begin, end, stars, fraction, style))
        return tuple(result)


def source_schedule(rows, duration, stars, width_ms=16000, offset_ms=0, *, style_names=(),
                    difficulty_trace=None):
    """Separate-chart scoped targets; no union of alternative arrangements.

    An optional complete-source strain trace replaces the whole-chart star
    condition inside each sampled scope with its declared difficulty proxy.
    Its specification must accompany trained checkpoints. Omitting it retains
    the original whole-chart supervision.
    """
    spans = [ControlSpan(0, duration+1, stars=stars)]
    for start in range(-offset_ms, duration+1, width_ms):
        stop = min(duration+1, start+width_ms)
        actions = rows['actions'][(rows['time'] >= max(0, start)) & (rows['time'] < stop)]
        heads = int(np.isin(actions, (1, 2)).sum())
        fraction = float((actions == 2).sum()/heads) if heads else None
        level = None if difficulty_trace is None else difficulty_trace.level(max(0, start), stop)
        spans.append(ControlSpan(max(0, start), stop, stars=level, ln_fraction=fraction))
    return ControlSchedule(tuple(spans), style_names)
