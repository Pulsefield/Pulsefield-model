"""Target-free performed-time, local-pace and redline coordinates.

Neighborhood statistics use complete positive attack gaps wholly inside the
review context. Redlines are queried at the event; normalized crossing gaps
are local coordinates, not integrated beats. Inherited SV points are ignored.
"""
from bisect import bisect_right
from dataclasses import dataclass, fields
from functools import lru_cache
import math
import re
from statistics import median

import torch
from torch.nn.utils.rnn import pad_sequence

from .dataset import ContractError, digest
from .replay import HAND_COLUMNS, time_feature

TIME_DIM = 8
TEMPORAL_ROW_DIM = 18
TEMPORAL_LANE_DIM = 72
TEMPORAL_RELATION_DIM = 24
SPAN_DIM = 9


def parse_redlines(data, expected_sha256, *, last_query_ms=None):
    if digest(data) != expected_sha256:
        raise ContractError('Source SHA-256 mismatch before timing parsing')
    section, points = None, {}
    for number, raw in enumerate(re.split(r'\r\n|\n|\r', data.decode('utf-8-sig')), 1):
        text = raw.strip()
        if text.startswith('['):
            section = text
        elif section == '[TimingPoints]' and text and not text.startswith('//'):
            values = text.split(',')
            try:
                t, length = float(values[0]), float(values[1])
                if last_query_ms is not None and math.isfinite(t) and t > last_query_ms:
                    continue
                red = int(values[6]) if len(values) > 6 else 1
                if red not in (0, 1) or not math.isfinite(t) or not math.isfinite(length):
                    raise ValueError('invalid timing value')
                if red:
                    if length <= 0:
                        raise ValueError('redline beat length must be positive')
                    # Source order resolves multiple redlines at the same time.
                    points[t] = length
            except (ValueError, IndexError) as exc:
                raise ContractError(f'Timing point line {number}: {exc}') from exc
    if not points:
        raise ContractError('Enhanced timing requires at least one source redline')
    return sorted(points.items())


def signed_log(value):
    return math.copysign(math.log1p(abs(value)), value)


class TimeCoordinates:
    def __init__(self, chart, redlines, rate=1.0, radii=(2, 8)):
        if not math.isfinite(rate) or rate <= 0:
            raise ContractError('Playback rate must be finite and positive')
        self.times = [r.time_ms for r in chart.inputs.rows if r.attack_columns]
        self.gaps = [(b-a)/rate for a, b in zip(self.times, self.times[1:])]
        self.redlines = redlines
        self.redtimes = [p[0] for p in redlines]
        self.rate, self.radii = rate, radii
        self.at = lru_cache(maxsize=None)(self.at)

    def at(self, t):
        rank = bisect_right(self.times, t)-1
        gap_index = rank-1
        gap = self.gaps[gap_index] if gap_index >= 0 else None
        previous = self.gaps[gap_index-1] if gap_index >= 1 else None
        references = []
        center = max(0, gap_index)
        for radius in self.radii:
            values = self.gaps[max(0, center-radius):center+radius+1]
            references.append(median(values) if values else None)
        red_index = bisect_right(self.redtimes, t)-1
        beat = self.redlines[red_index][1]/self.rate if red_index >= 0 else None
        return gap, previous, references, beat

    def describe(self, source_ms, t):
        _, _, refs, beat = self.at(t)
        performed = None if source_ms is None else source_ms/self.rate
        values = list(time_feature(performed))
        for reference in (*refs, beat):
            valid = performed is not None and reference is not None
            values.extend((signed_log(performed/reference) if valid else 0.0, valid))
        return values

    def row(self, row):
        gap, previous, _, _ = self.at(row.time_ms)
        return [*self.describe(None if gap is None else gap*self.rate, row.time_ms),
                math.log(gap/previous) if gap is not None and previous is not None else 0.0,
                gap is not None and previous is not None,
                *self.describe(row.elapsed_ms, row.time_ms)]


@dataclass(frozen=True)
class Sidecars:
    lanes: torch.Tensor
    rows: torch.Tensor
    edges: torch.Tensor
    relations: torch.Tensor
    spans: torch.Tensor

    def to(self, device):
        return Sidecars(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})


def span_metadata(chart, dilations, rate):
    rows, scope = chart.inputs.rows, chart.inputs.scope
    radii, radius = [0], 0
    for d in dilations:
        radius += d
        radii.append(radius)
    result = []
    for i, row in enumerate(rows):
        scales = []
        for radius in radii:
            left, right = max(0, i-radius), min(len(rows)-1, i+radius)
            span = (rows[right].time_ms-rows[left].time_ms)/rate
            attacks = sum(bool(r.attack_columns) for r in rows[left:right+1])
            scales.append([time_feature(span)[0], math.log1p(attacks),
                           (row.time_ms-scope.start_ms)/(scope.end_ms-scope.start_ms),
                           i-radius >= 0, i+radius < len(rows),
                           rows[left].time_ms < scope.start_ms, rows[right].time_ms >= scope.end_ms,
                           row.in_scope, row.phase == 'source'])
        result.append(scales)
    return result


def temporal_sidecars(examples, redlines, rates, radii, dilations):
    lanes, rows, spans, edges, relations = [], [], [], [], []
    for example, points, rate in zip(examples, redlines, rates):
        chart = example.chart
        coordinates = TimeCoordinates(chart, points, rate, radii)
        row_values, lane_values = [], []
        for row in chart.inputs.rows:
            row_values.append(coordinates.row(row))
            lane_row = []
            for lane in row.lanes:
                values = []
                for key in ('previous_attack_ms', 'next_attack_ms', 'before_age_ms', 'before_remaining_ms',
                            'before_duration_ms', 'after_age_ms', 'after_remaining_ms', 'after_duration_ms'):
                    values.extend(coordinates.describe(getattr(lane, key), row.time_ms))
                nearest = None
                if lane.ln_close and coordinates.times:
                    j = bisect_right(coordinates.times, row.time_ms)
                    candidates = coordinates.times[max(0, j-1):j+1]
                    nearest = min(candidates, key=lambda t: (abs(t-row.time_ms), t))-row.time_ms
                values.extend(coordinates.describe(nearest, row.time_ms))
                lane_row.append(values)
            lane_values.append([[lane_row[j] for j in columns] for columns in HAND_COLUMNS])
        lanes.append(torch.tensor(lane_values, dtype=torch.float32))
        rows.append(torch.tensor(row_values, dtype=torch.float32))
        spans.append(torch.tensor(span_metadata(chart, dilations, rate), dtype=torch.float32))
        for edge in example.edges:
            t = chart.inputs.rows[edge.query//2].time_ms
            edges.append(coordinates.describe(edge.elapsed_ms, t))
            for relation in edge.relations:
                available = relation.kind in ('ln_identity', 'occupied_role')
                relations.append([v for key in ('duration_ms', 'head_delta_ms', 'close_delta_ms')
                                  for v in coordinates.describe(getattr(relation, key) if available else None, t)])
    return Sidecars(pad_sequence(lanes, batch_first=True), pad_sequence(rows, batch_first=True),
                    torch.tensor(edges, dtype=torch.float32), torch.tensor(relations, dtype=torch.float32),
                    pad_sequence(spans, batch_first=True))
