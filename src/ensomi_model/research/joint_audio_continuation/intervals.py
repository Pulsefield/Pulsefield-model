"""Continuous source-clock likelihood with shared causal history computation.

Intervals partition the actual integer audio clock, not detected beats or chart
sections. Every millisecond and full event row is scored once. Full-song audio
context is supplied separately from the bounded local crop and chart labels.
"""
from dataclasses import dataclass, replace
import math

import numpy as np
import torch
from torch.nn import functional as F

from ..bounded_typed_continuation.contract import Arm, ROW_ACTIONS
from ..bounded_typed_continuation.features import CONTENT_DIM
from ..scoped_style_modeling.dataset import ContractError
from .batching import interpolate_audio
from .data import JointChart
from .state import BASE_QUERY_DIM, exact_features, legal_rows


@dataclass(frozen=True)
class IntervalExample:
    chart: JointChart
    index: int
    width_ms: int = 8000

    def __post_init__(self):
        if (type(self.width_ms) is not int or self.width_ms <= 0 or type(self.index) is not int or
                not 0 <= self.index < self.count):
            raise ContractError('Interval index must belong to the positive-width audio partition')

    @property
    def count(self):
        return (self.chart.duration_ms + self.width_ms) // self.width_ms

    @property
    def start_ms(self):
        return self.index * self.width_ms

    @property
    def end_ms(self):
        """Exclusive clock boundary; the actual audio endpoint is included."""
        return min(self.start_ms + self.width_ms, self.chart.duration_ms + 1)

    @property
    def weight_per_second(self):
        """Uniform-interval importance weight for this chart's NLL per second."""
        return 1000 * self.count / (self.chart.duration_ms + 1)

    def identity(self):
        return dict(source_sha256=self.chart.entry['source_sha256'], interval_index=self.index,
                    start_ms=self.start_ms, end_ms=self.end_ms, width_ms=self.width_ms,
                    weight_per_second=self.weight_per_second)


@dataclass(frozen=True)
class IntervalInputs:
    mel: torch.Tensor
    mel_valid: torch.Tensor
    mel_start: torch.Tensor
    frame_count: torch.Tensor
    raw: torch.Tensor
    history_valid: torch.Tensor
    timing_times: torch.Tensor
    timing_history: torch.Tensor
    timing_exact: torch.Tensor
    timing_valid: torch.Tensor
    timing_forced: torch.Tensor
    row_times: torch.Tensor
    row_history: torch.Tensor
    row_exact: torch.Tensor
    row_legal: torch.Tensor
    occupancy: torch.Tensor


@dataclass(frozen=True)
class IntervalTargets:
    timing_event: torch.Tensor
    row_index: torch.Tensor


@dataclass(frozen=True)
class IntervalBatch:
    inputs: IntervalInputs
    targets: IntervalTargets
    weight_per_second: float


@dataclass(frozen=True)
class IntervalScores:
    timing_logits: torch.Tensor
    row_log_probs: torch.Tensor


def _pad_first(values, quantum=128, *, edge=False):
    """Bound runtime shape variety without adding scored rows or audio frames."""
    values = np.asarray(values)
    padding = (-len(values)) % quantum
    if not padding or not len(values):
        return values
    extra = np.repeat(values[-1:], padding, axis=0) if edge else np.zeros((padding, *values.shape[1:]), values.dtype)
    return np.concatenate((values, extra), axis=0)


def collate_interval(example, model_config, device='cpu'):
    """Build one interval with true prefix state and causal per-query indices.

The local audio extent depends only on the sampled interval, never its labels.
The raw sequence includes the necessary prehistory and within-interval rows;
the timing/row indices always select the encoding before their target event.
Neither a crop end nor source exhaustion forces a release.
"""
    chart = example.chart
    source = chart.source
    start, end = example.start_ms, example.end_ms
    times = source.rows['time']
    first = int(np.searchsorted(times, start, side='left'))
    stop = int(np.searchsorted(times, end, side='left'))
    history_start = max(0, first - (2 ** (model_config.history_levels + 1) - 1))
    raw = (source.content(Arm.R0, history_start, stop) if stop > history_start else
           np.empty((0, 2, CONTENT_DIM), np.float32))
    timing_times, timing_states, timing_history = [], [], []
    timing_valid, timing_forced, timing_event = [], [], []
    row_times, row_states, row_history, row_targets = [], [], [], []
    cursor = start - 1
    for index in range(first, stop + 1):
        has_event = index < stop
        target = int(times[index]) if has_event else end - 1
        if has_event and times[index] != target:
            raise ContractError('Interval supervision requires native integer-millisecond source rows')
        if cursor >= target:
            continue
        replay = replace(source.state(Arm.R0, index).replay, is_complete=False)
        bins = np.arange((cursor + 1) // 10, target // 10 + 1, dtype=np.int64)
        clocks = bins * 10 + 9
        native = bins[:, None] * 10 + np.arange(10)
        valid = (native > cursor) & (native <= target)
        forced = valid & (native == chart.duration_ms) & any(replay.occupancy)
        events = valid & (native == target) & has_event
        if np.any(forced & ~events):
            raise ContractError('Source interval leaves an active LN at the true audio terminal')
        timing_times.extend(clocks.tolist())
        timing_states.extend([replay] * len(bins))
        timing_history.extend([index - history_start - 1] * len(bins))
        timing_valid.extend(valid.tolist())
        timing_forced.extend(forced.tolist())
        timing_event.extend(events.tolist())
        if has_event:
            row_times.append(target)
            row_states.append(replay)
            row_history.append(index - history_start - 1)
            row_targets.append(ROW_ACTIONS.index(tuple(map(int, source.rows['actions'][index]))))
        cursor = target
    if sum(map(sum, timing_valid)) != end - start:
        raise ContractError('Interval likelihood does not cover its integer clock exactly once')

    frames, halo = len(chart.mel), model_config.audio_halo_frames
    first_clock = (start // 10) * 10
    last_clock = ((end - 1) // 10) * 10 + 9
    low = np.clip((first_clock - 20) / 10, 0., frames - 1)
    high = np.clip((last_clock - 20) / 10, 0., frames - 1)
    audio_start = int(math.floor(low)) - halo
    audio_stop = min(frames - 1, int(math.floor(high)) + 1) + halo + 1
    mel = np.zeros((1, audio_stop - audio_start, 128), np.float32)
    mel_valid = np.zeros(mel.shape[:2], np.bool_)
    lo, hi = max(0, audio_start), min(frames, audio_stop)
    mel[0, lo - audio_start:hi - audio_start] = chart.mel[lo:hi]
    mel_valid[0, lo - audio_start:hi - audio_start] = True
    terminal = [t == chart.duration_ms for t in row_times]
    support = legal_rows(row_states, terminal)
    if any(not support[i, target] for i, target in enumerate(row_targets)):
        raise ContractError('Interval target row violates its true prefix occupancy')

    # MPS caches first-seen operator shapes. Pad only computation axes; exact
    # replay and target counts remain actual, and every added timing slot is
    # excluded from likelihood. Repeated row queries have no corresponding
    # targets. No padding position is an observed history row or audio frame.
    raw_valid = _pad_first(np.ones(len(raw), np.bool_))
    raw = _pad_first(raw)
    mel = _pad_first(mel[0])[None]
    mel_valid = _pad_first(mel_valid[0])[None]
    timing_exact = _pad_first(exact_features(timing_states, timing_times), edge=True)
    row_exact = _pad_first(exact_features(row_states, row_times), 64, edge=True)
    row_occupancy = _pad_first(np.asarray([s.occupancy for s in row_states], np.bool_).reshape(-1, 4), 64, edge=True)

    def tensor(value, dtype=None):
        return torch.as_tensor(value, dtype=dtype, device=device)

    inputs = IntervalInputs(tensor(mel), tensor(mel_valid), tensor([audio_start]), tensor([frames]),
        tensor(raw[None]), tensor(raw_valid[None]), tensor(_pad_first(timing_times, edge=True)),
        tensor(_pad_first(timing_history, edge=True)), tensor(timing_exact),
        tensor(_pad_first(timing_valid), torch.bool), tensor(_pad_first(timing_forced), torch.bool),
        tensor(_pad_first(row_times, 64, edge=True), torch.long),
        tensor(_pad_first(row_history, 64, edge=True), torch.long), tensor(row_exact),
        tensor(_pad_first(support, 64, edge=True)), tensor(row_occupancy))
    return IntervalBatch(inputs, IntervalTargets(tensor(_pad_first(timing_event), torch.bool), tensor(row_targets, torch.long)),
                         example.weight_per_second)


def encode_interval(model, inputs, coarse=None):
    """Compute shared audio and causal history once, without target tensors."""
    encoded = model.encode_crop(inputs.mel, inputs.mel_valid, inputs.mel_start, inputs.frame_count, coarse)
    history = model.temporal(inputs.raw, inputs.history_valid)[0] if inputs.raw.shape[1] else None
    return encoded, history


def score_encoded_interval(model, inputs, encoded, history):
    """Score query states from already encoded, explicitly conditioned audio."""

    def gather(indices):
        boundary = model.temporal.boundary[0].expand(len(indices), 2, -1)
        if history is None:
            return boundary
        return torch.where((indices >= 0)[:, None, None], history[indices.clamp_min(0)], boundary)

    # Queries have different true prefix states, so use the batch axis for
    # them. The candidate-time axis has length one; no query sees its peers.
    audio = interpolate_audio(encoded, inputs.timing_times[None], inputs.mel_start, inputs.frame_count)[0]
    timing = model.timing_logits(audio[:, None], gather(inputs.timing_history), inputs.timing_exact[:, None])[:, 0]
    if len(inputs.row_times):
        audio = interpolate_audio(encoded, inputs.row_times[None], inputs.mel_start, inputs.frame_count)[0]
        rows = model.row_log_probs(audio, gather(inputs.row_history), inputs.row_exact, inputs.row_legal, inputs.occupancy)
    else:
        rows = timing.new_empty((0, 256))
    return IntervalScores(timing, rows)


def score_interval(model, inputs, coarse=None, *, code=None):
    """Read full-song audio and causal history; latent models require a code."""
    from .intent_model import IntentAudioModel
    encoded, history = encode_interval(model, inputs, coarse)
    if isinstance(model, IntentAudioModel):
        encoded = model.apply_code(encoded, code)
    elif code is not None:
        raise ContractError('This interval model has no persistent intent code')
    return score_encoded_interval(model, inputs, encoded, history)


def interval_losses(scores, batch):
    """Return scalar timing, row and joint NLL sums, before importance weighting."""
    event = batch.targets.timing_event
    valid = batch.inputs.timing_valid & ~batch.inputs.timing_forced
    timing = torch.where(event, F.softplus(-scores.timing_logits), F.softplus(scores.timing_logits))
    timing = timing.masked_select(valid).sum()
    row = (-scores.row_log_probs[torch.arange(len(batch.targets.row_index), device=timing.device),
                                 batch.targets.row_index].sum() if len(batch.targets.row_index) else timing * 0)
    return timing, row, timing + row
