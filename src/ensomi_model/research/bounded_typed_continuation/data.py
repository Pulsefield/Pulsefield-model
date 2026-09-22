"""Source-label ownership and parameter-independent bounded training windows.

Compact integer snapshots recover complete exact facts at any crop without
neural prefix replay. Source endpoints remain supervision; only O1 prior plans
and typed seed plans are projected into predictor features.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from ..oracle_time_continuation.data import SourceIdentity
from ..oracle_time_continuation.replay import ExactReplayState
from ..oracle_time_continuation.schema import CompleteRow
from ..oracle_time_continuation.storage import ROW_DTYPE, SOURCE_FORMAT, file_digest
from ..scoped_style_modeling.dataset import ContractError
from .contract import Arm, Schedule, Timing
from .features import CONTENT_DIM, QUERY_DIM, TimingView, content_features, query_features


class SourceChart:
    """Validated source rows plus exact integer indexes; never a model argument.

    Construction costs O(source rows), independent of neural parameters. Owners
    should apply a byte/count LRU before loading many charts. No hidden states or
    expanded neural features are cached here.
    """
    def __init__(self, identity: SourceIdentity, rows, *, minimum_seed_notes=30):
        if type(minimum_seed_notes) is not int or minimum_seed_notes <= 0:
            raise ContractError('Minimum seed note count must be a positive integer')
        if not isinstance(rows, np.ndarray) or rows.dtype != ROW_DTYPE:
            raise ContractError('Source chart requires the admitted lossless row-array format')
        rows = np.array(rows, dtype=ROW_DTYPE, copy=True)
        if (rows.ndim != 1 or not len(rows) or np.any(~np.isfinite(rows['time'])) or
                np.any(rows['time'] < 0) or np.any(np.diff(rows['time']) <= 0) or
                np.any(rows['actions'] > 3) or np.any(~rows['actions'].any(-1))):
            raise ContractError('Source rows require increasing finite times and nonempty four-lane actions')
        self.identity, self.rows = identity, rows
        actions, indices = rows['actions'], np.arange(len(rows), dtype=np.int32)
        attack = (actions == 1) | (actions == 2)
        self.note_counts = np.r_[0, np.cumsum(attack.sum(-1))]
        seed_last = np.searchsorted(self.note_counts, minimum_seed_notes, side='left')
        if seed_last >= len(rows):
            raise ContractError('Continuation needs a complete minimum-note seed and a nonempty suffix')
        self.seed_rows = int(seed_last)
        self.endpoints = np.full((len(rows), 4), -1, dtype=np.int32)
        for lane in range(4):
            starts, ends = np.flatnonzero(actions[:, lane] == 2), np.flatnonzero(actions[:, lane] == 3)
            if (len(starts) != len(ends) or np.any(ends <= starts) or
                    np.any(starts[1:] <= ends[:-1])):
                raise ContractError('Source LN starts and releases must pair strictly without overlap')
            self.endpoints[starts, lane] = ends
        self.last_attack = np.maximum.accumulate(np.where(attack, indices[:, None], -1), axis=0)
        self.last_release = np.maximum.accumulate(np.where(actions == 3, indices[:, None], -1), axis=0)
        latest_start = np.maximum.accumulate(np.where(actions == 2, indices[:, None], -1), axis=0)
        self.open_start = np.where(latest_start > self.last_release, latest_start, -1).astype(np.int32)
        occupied_before = np.vstack((np.full((1, 4), -1), self.open_start[:-1])) >= 0
        if np.any(attack & occupied_before) or np.any((actions == 3) & ~occupied_before) or np.any(self.open_start[-1] >= 0):
            raise ContractError('Source actions violate exact occupancy or terminal closure')
        roles = attack.any(-1)
        times = tuple(float(t) for t in rows['time'])
        self.typed = TimingView(Timing(times, tuple(bool(role) for role in roles)))
        self.untyped = TimingView(Timing(times))
        self.onsets = np.flatnonzero(roles & (indices >= self.seed_rows))
        if not len(self.onsets):
            raise ContractError('This onset-normalized experiment requires a post-seed onset')
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)

    @classmethod
    def from_cache(cls, directory: Path, identity: SourceIdentity, *, max_bytes=64 * 1024 ** 2):
        """Read a pinned admitted cache, checking identity, row bytes and digest."""
        directory = Path(directory)
        metadata = json.loads((directory / 'metadata.json').read_text())
        path = directory / 'rows.bin'
        if (metadata['format'] != SOURCE_FORMAT or SourceIdentity(**metadata['identity']) != identity or
                path.stat().st_size != metadata['row_count'] * ROW_DTYPE.itemsize or
                file_digest(path, max_bytes) != metadata['rows_sha256']):
            raise ContractError('Source cache identity, format or row digest differs from the admitted source')
        result = cls(identity, np.fromfile(path, dtype=ROW_DTYPE))
        if result.seed_rows != metadata['seed']['seed_row_count'] or metadata['seed']['ineligible_reason'] is not None:
            raise ContractError('Source cache seed differs from the full 30-note rule')
        return result

    @property
    def array_bytes(self):
        return sum(value.nbytes for value in vars(self).values() if isinstance(value, np.ndarray))

    def view(self, arm):
        if not isinstance(arm, Arm):
            raise ContractError('Source projection requires an explicit task arm')
        return self.untyped if arm == Arm.R0 else self.typed

    def row(self, index):
        row = self.rows[index]
        return CompleteRow(float(row['time']), tuple(int(action) for action in row['actions']))

    def state(self, arm: Arm, index: int):
        if type(index) is not int or not 0 <= index <= len(self.rows):
            raise ContractError('Exact source query is outside the physical row sequence')
        view = self.view(arm)
        if not index:
            return Schedule(arm, view.timing)
        def times(values):
            return tuple(None if i < 0 else float(self.rows['time'][i]) for i in values)
        opened = self.open_start[index - 1]
        replay = ExactReplayState(row_count=index, note_count=int(self.note_counts[index]),
                                  first_time_ms=float(self.rows['time'][0]), last_row=self.row(index - 1),
                                  open_ln_start_ms=times(opened), last_lane_attack_ms=times(self.last_attack[index - 1]),
                                  last_lane_release_ms=times(self.last_release[index - 1]), is_complete=index == len(self.rows))
        ends = tuple(int(self.endpoints[start, c]) if start >= 0 and
                     (arm == Arm.O1 or (arm == Arm.R1 and start < self.seed_rows)) else None
                     for c, start in enumerate(opened))
        return Schedule(arm, view.timing, index, replay, ends)

    def content(self, arm: Arm, start: int, stop: int):
        self.view(arm)
        if not 0 <= start < stop <= len(self.rows):
            raise ContractError('Content crop must be a nonempty physical row interval')
        indices = np.arange(start, stop)
        previous = np.where(indices > 0, self.rows['time'][np.maximum(indices - 1, 0)], np.nan)
        ends = self.endpoints[start:stop]
        visible = (ends >= 0) & ((arm == Arm.O1) | ((arm == Arm.R1) & (indices[:, None] < self.seed_rows)))
        times = np.where(visible, self.rows['time'][np.maximum(ends, 0)], np.nan)
        return content_features([self.row(int(i)) for i in indices], previous, times)


@dataclass(frozen=True)
class SourceInterval:
    """One shared onset-space selection, interpreted identically by all arms."""
    chart: SourceChart
    first_onset: int
    onset_count: int

    def __post_init__(self):
        if (type(self.first_onset) is not int or type(self.onset_count) is not int or
                self.first_onset < 0 or self.onset_count <= 0 or
                self.first_onset + self.onset_count > len(self.chart.onsets)):
            raise ContractError('Source interval must select actual post-seed onset exposures')

    @property
    def start(self):
        return self.chart.seed_rows if self.first_onset == 0 else int(self.chart.onsets[self.first_onset])

    @property
    def stop(self):
        following = self.first_onset + self.onset_count
        return len(self.chart.rows) if following == len(self.chart.onsets) else int(self.chart.onsets[following])


@dataclass
class PreparedBatch:
    raw: np.ndarray
    valid: np.ndarray
    truncated: np.ndarray
    query_features: np.ndarray
    batch_indices: np.ndarray
    positions: np.ndarray
    states: list[Schedule]
    labels: list[tuple[int, ...]]
    endpoints: list[dict[int, int]]
    views: list[TimingView]
    source_onsets: int
    physical_rows: int
    prefix_rows: int
    context_spans_ms: list[float]
    seed_raw: np.ndarray
    seed_valid: np.ndarray
    memory_raw: np.ndarray | None = None
    memory_valid: np.ndarray | None = None
    memory_onsets: np.ndarray | None = None


def prepare_batch(intervals: list[SourceInterval], arm: Arm, receptive_tokens: int, *, full_history=False):
    """Materialize finite raw prefixes and all supervised physical suffix rows.

    The oldest content gap reads one preceding timestamp. No source action older
    than receptive_tokens enters rolling content; exact facts stay complete.
    The original supplied seed is separately projected for optional persistent
    conditioning. Its endpoints follow the same arm-specific visibility rule.
    full_history additionally materializes the true BOS prefix for causal landmark
    memory; it never substitutes a truncated learned state from an earlier update.
    Forced O1 releases still write content but do not create stochastic queries.
    """
    if type(full_history) is not bool:
        raise ContractError('Full-history preparation must be explicitly boolean')
    if not intervals or type(receptive_tokens) is not int or receptive_tokens <= 0:
        raise ContractError('Batch preparation needs intervals and a positive learned token range')
    crops = [(max(0, item.start - receptive_tokens), item.stop) for item in intervals]
    length = max(stop - start for start, stop in crops)
    raw = np.zeros((len(intervals), length, 2, CONTENT_DIM), dtype=np.float32)
    valid = np.zeros((len(intervals), length), dtype=np.bool_)
    truncated = np.zeros(len(intervals), dtype=np.bool_)
    seed_length = max(item.chart.seed_rows for item in intervals)
    seed_raw = np.zeros((len(intervals), seed_length, 2, CONTENT_DIM), dtype=np.float32)
    seed_valid = np.zeros((len(intervals), seed_length), dtype=np.bool_)
    memory_raw = memory_valid = memory_onsets = None
    if full_history:
        memory_length = max(item.stop for item in intervals)
        memory_raw = np.zeros((len(intervals), memory_length, 2, CONTENT_DIM), dtype=np.float32)
        memory_valid = np.zeros((len(intervals), memory_length), dtype=np.bool_)
        memory_onsets = np.zeros_like(memory_valid)
    states, labels, endpoints, views, facts, batches, positions, spans = [], [], [], [], [], [], [], []
    for batch, (item, (first, stop)) in enumerate(zip(intervals, crops)):
        chart, view = item.chart, item.chart.view(arm)
        seed_raw[batch, :chart.seed_rows] = chart.content(arm, 0, chart.seed_rows)
        seed_valid[batch, :chart.seed_rows] = True
        if full_history:
            memory_raw[batch, :stop] = chart.content(arm, 0, stop)
            memory_valid[batch, :stop] = True
            memory_onsets[batch, :stop] = chart.typed.roles[:stop]
        raw[batch, :stop - first] = chart.content(arm, first, stop)
        valid[batch, :stop - first] = True
        truncated[batch] = first > 0
        queried = [i for i in range(item.start, stop) if arm != Arm.O1 or view.timing.onsets[i]]
        current = [chart.state(arm, i) for i in queried]
        states.extend(current)
        facts.append(query_features(current, view))
        for i in queried:
            action = chart.row(i).actions
            labels.append(tuple(a if a in (1, 2) else 0 for a in action) if arm == Arm.O1 else action)
            endpoints.append({c: int(end) for c, end in enumerate(chart.endpoints[i]) if end >= 0} if arm == Arm.O1 else {})
        views.extend([view] * len(queried))
        batches.extend([batch] * len(queried))
        positions.extend(i - first for i in queried)
        spans.append(float(view.times[item.start] - view.times[max(0, item.start - receptive_tokens - 1)]))
    return PreparedBatch(raw, valid, truncated, np.concatenate(facts), np.asarray(batches), np.asarray(positions),
                         states, labels, endpoints, views, sum(i.onset_count for i in intervals),
                         sum(i.stop - i.start for i in intervals), sum(i.start - first for i, (first, _) in zip(intervals, crops)),
                         spans, seed_raw, seed_valid, memory_raw, memory_valid, memory_onsets)


def batch_predictions(model, batch: PreparedBatch):
    """Causal hand vectors and legal action log-probabilities for prepared queries."""
    like = model.temporal.input.weight
    raw = like.new_tensor(batch.raw)
    valid = torch.as_tensor(batch.valid, dtype=torch.bool, device=like.device)
    content = model.temporal(raw, valid)
    before = model.temporal.before(content, valid, truncated_start=torch.as_tensor(batch.truncated, device=like.device))
    batches = torch.as_tensor(batch.batch_indices, dtype=torch.long, device=like.device)
    positions = torch.as_tensor(batch.positions, dtype=torch.long, device=like.device)
    seed = None
    if model.config.seed_context == 'observed':
        seed = model.encode_seed(like.new_tensor(batch.seed_raw),
                                 torch.as_tensor(batch.seed_valid, dtype=torch.bool, device=like.device))[batches]
    memory = None
    if model.long_memory is not None:
        if batch.memory_raw is None or batch.memory_valid is None or batch.memory_onsets is None:
            raise ContractError('Long-memory training requires explicitly prepared full histories')
        bank = model.long_memory.encode(like.new_tensor(batch.memory_raw),
            torch.as_tensor(batch.memory_valid, device=like.device),
            torch.as_tensor(batch.memory_onsets, device=like.device))
        absolute = torch.tensor([s.replay.row_count for s in batch.states], dtype=torch.long, device=like.device)
        memory = bank.query(batches, absolute)
    hands = model.readout(before[batches, positions], like.new_tensor(batch.query_features), seed, memory)
    return hands, model.decision_log_probs(hands, batch.states)


def batch_likelihood(model, batch: PreparedBatch, *, candidate_budget=8192, recompute=True, diagnostics=None,
                     source_kl_weight=0.):
    """Sum every task factor, normalized by actual supervised source onsets.

    Returned head/endpoint sums describe local training factor costs. They are
    not a complete-suffix R1/O1 comparison unless this batch covers that suffix.
    """
    hands, probabilities = batch_predictions(model, batch)
    like = model.temporal.input.weight
    lookup = {choice: i for i, choice in enumerate(model.choices)}
    labels = torch.tensor([lookup[row] for row in batch.labels], dtype=torch.long, device=like.device)
    selected = probabilities.gather(1, labels[:, None]).squeeze(1)
    if not bool(torch.isfinite(selected).all()):
        raise ContractError('Source decision is outside the model support')
    head_nll = -selected.sum()
    endpoint_nll = (model.endpoint_log_probs(hands, batch.states, batch.labels, batch.endpoints, batch.views,
                                           candidate_budget=candidate_budget, recompute=recompute, ledger=diagnostics).sum().neg()
                    if model.pointer is not None else head_nll.new_zeros(()))
    source_kl = head_nll.new_zeros(())
    if source_kl_weight:
        if model.row_consequence is None or any(p.requires_grad for name, p in model.named_parameters()
                                               if not name.startswith('row_consequence.')):
            raise ContractError('Source KL requires the inherited policy to be frozen')
        with torch.no_grad():
            reference = model.decision_log_probs(hands.detach(), batch.states, include_consequence=False)
        legal = torch.isfinite(reference)
        difference = reference.masked_fill(~legal, 0.) - probabilities.masked_fill(~legal, 0.)
        source_kl = (reference.exp() * difference).sum()
        if diagnostics is not None:
            diagnostics['source_kl_sum'] = float(source_kl.detach().cpu())
    if diagnostics is not None:
        with torch.no_grad():
            observed = torch.tensor(batch.labels, device=like.device)
            choices = torch.tensor(model.choices, device=like.device)
            for name, action in (('ln_type', 2), ('tap_type', 1)):
                values = []
                for lane in range(4):
                    at = observed[:, lane] == action
                    values.append(-probabilities[at][:, choices[:, lane] == action].logsumexp(-1).sum())
                diagnostics[name + '_nll_sum'] = float(torch.stack(values).sum().cpu())
                diagnostics[name + '_count'] = int((observed == action).sum().cpu())
            # Score both classes on source-onset lanes where either outcome is
            # feasible. Positive-only likelihood rewards an excessive LN prior.
            for name in ('ln_binary', 'head_type_binary'):
                diagnostics.update({name + suffix: 0 for suffix in
                                    ('_nll_sum', '_brier_sum', '_probability_sum', '_positive_count', '_count')})
            source_onset = ((observed == 1) | (observed == 2)).any(-1)
            for lane in range(4):
                ln = probabilities[:, choices[:, lane] == 2].logsumexp(-1)
                other = probabilities[:, choices[:, lane] != 2].logsumexp(-1)
                tap = probabilities[:, choices[:, lane] == 1].logsumexp(-1)
                truth = observed[:, lane] == 2
                for name, negative, cohort in (
                        ('ln_binary', other, source_onset),
                        ('head_type_binary', tap, (observed[:, lane] == 1) | truth)):
                    at = cohort & torch.isfinite(ln) & torch.isfinite(negative)
                    positive, negative = ln[at], negative[at]
                    normalizer = torch.logaddexp(positive, negative)
                    positive, negative = positive - normalizer, negative - normalizer
                    p, y = positive.exp(), truth[at]
                    diagnostics[name + '_nll_sum'] += float(-torch.where(y, positive, negative).sum().cpu())
                    diagnostics[name + '_brier_sum'] += float((p - y.float()).square().sum().cpu())
                    diagnostics[name + '_probability_sum'] += float(p.sum().cpu())
                    diagnostics[name + '_positive_count'] += int(y.sum().cpu())
                    diagnostics[name + '_count'] += len(p)
            diagnostics.setdefault('endpoint_decisions', 0)
            diagnostics.setdefault('scored_order_factors', 0)
            diagnostics.setdefault('candidate_pairs', 0)
    return (head_nll + endpoint_nll + source_kl_weight * source_kl) / batch.source_onsets, torch.stack((head_nll, endpoint_nll))
