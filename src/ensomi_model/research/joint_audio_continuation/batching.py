"""Align native event queries, finite audio crops and separately owned labels."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ..bounded_typed_continuation.contract import ROW_ACTIONS
from ..bounded_typed_continuation.features import CONTENT_DIM
from ..scoped_style_modeling.dataset import ContractError
from .data import JointChart, JointQuery, MEL_CENTER_ORIGIN_MS, MEL_HOP_MS
from .model import JointAudioModel, JointModelConfig
from .state import BASE_QUERY_DIM, exact_features, legal_rows
from .timing import hazard_nll


@dataclass(frozen=True)
class JointInputs:
    mel: Tensor
    mel_valid: Tensor
    mel_starts: Tensor
    mel_frame_counts: Tensor
    raw: Tensor
    history_valid: Tensor
    truncated: Tensor
    timing_times_ms: Tensor
    timing_exact: Tensor
    timing_valid: Tensor
    timing_forced: Tensor
    row_times_ms: Tensor
    row_exact: Tensor
    row_legal: Tensor
    occupancy: Tensor


@dataclass(frozen=True)
class JointTargets:
    event_index: Tensor
    row_index: Tensor
    row_mask: Tensor


@dataclass(frozen=True)
class JointBatch:
    inputs: JointInputs
    targets: JointTargets


@dataclass(frozen=True)
class JointScores:
    timing_logits: Tensor
    row_log_probs: Tensor


@dataclass(frozen=True)
class JointLosses:
    timing: Tensor
    row: Tensor
    total: Tensor


def interpolate_audio(
    encoded: Tensor, times_ms: Tensor, starts: Tensor | None = None,
    frame_counts: Tensor | None = None,
) -> Tensor:
    """Interpolate encoded [B,F,A] at [B] or [B,Q] times without detaching.

    Global Mel frame i has center 20 + 10*i milliseconds. Times outside real
    frame centers clamp to the nearest real frame. Crops supply their global
    starting frame index, possibly negative, and the complete song frame count.
    Defaults describe a full-song batch. Both interpolation neighbors must be
    inside the supplied encoding; virtual crop frames cannot replace real song
    frames. Input shape, device and coverage violations raise ContractError.
    """
    if (encoded.ndim != 3 or encoded.shape[1] == 0 or not encoded.is_floating_point() or
            times_ms.ndim not in (1, 2) or times_ms.shape[0] != encoded.shape[0] or
            times_ms.device != encoded.device):
        raise ContractError('Audio interpolation needs [B,F,A] encoding and aligned [B] or [B,Q] times')
    batch, frames, width = encoded.shape
    starts = torch.zeros(batch, dtype=torch.long, device=encoded.device) if starts is None else starts
    frame_counts = (torch.full((batch,), frames, dtype=torch.long, device=encoded.device)
                    if frame_counts is None else frame_counts)
    if (starts.shape != (batch,) or frame_counts.shape != (batch,) or starts.dtype != torch.long or
            frame_counts.dtype != torch.long or starts.device != encoded.device or
            frame_counts.device != encoded.device or not bool((frame_counts > 0).all())):
        raise ContractError('Audio interpolation needs one integer crop start and positive song frame count')
    times = times_ms[:, None] if times_ms.ndim == 1 else times_ms
    if not bool(torch.isfinite(times).all()):
        raise ContractError('Audio interpolation times must be finite')
    position = (times.to(encoded.dtype) - MEL_CENTER_ORIGIN_MS) / MEL_HOP_MS
    position = torch.minimum(position.clamp_min(0.), (frame_counts - 1)[:, None])
    low = position.floor().long()
    high = torch.minimum(low + 1, (frame_counts - 1)[:, None])
    weight = (position - low).unsqueeze(-1)
    low, high = low - starts[:, None], high - starts[:, None]
    if bool(((low < 0) | (high >= frames)).any()):
        raise ContractError('Audio crop does not contain both real interpolation neighbors')
    left = encoded.gather(1, low[..., None].expand(-1, -1, width))
    right = encoded.gather(1, high[..., None].expand(-1, -1, width))
    values = left + weight * (right - left)
    return values[:, 0] if times_ms.ndim == 1 else values


def collate(
    queries: Sequence[JointQuery], charts: Sequence[JointChart],
    model_config: JointModelConfig, device: torch.device | str = 'cpu',
) -> JointBatch:
    """Batch aligned chart/query pairs, right-padding physical history and audio.

    Timing bins are absolute 10 ms bins; exact/audio queries occur at each bin's
    final millisecond. Native validity is cursor < t <= horizon, with only true
    audio-end closure forced for an occupied prefix. Each audio crop includes
    both interpolation neighbors for every requested bin plus the encoder halo.
    Crop extent never depends on which native millisecond is the target.

    Row predictors condition on the target time for p(row | time, history,
    audio). Censored rows use the cursor as an unscored placeholder. Event and
    row identities remain in targets and are never passed to score_batch.
    Invalid pairs, history envelopes and illegal target rows raise ContractError.
    """
    if not queries or len(queries) != len(charts):
        raise ContractError('Collation requires a nonempty aligned list of queries and charts')
    history_limit = 2 ** (model_config.history_levels + 1) - 1
    halo = model_config.audio_halo_frames
    batch = len(queries)
    first_bins = np.asarray([(q.cursor_ms + 1) // 10 for q in queries], dtype=np.int64)
    bin_counts = np.asarray([max(1, q.horizon_end_ms // 10 - first + 1)
                             for q, first in zip(queries, first_bins)], dtype=np.int64)
    max_bins = int(bin_counts.max())
    timing_times = np.zeros((batch, max_bins), np.int64)
    timing_exact = np.zeros((batch, max_bins, 2, BASE_QUERY_DIM), np.float32)
    timing_valid = np.zeros((batch, max_bins * 10), np.bool_)
    timing_forced = np.zeros_like(timing_valid)
    raw = np.zeros((batch, history_limit, 2, CONTENT_DIM), np.float32)
    history_valid = np.zeros((batch, history_limit), np.bool_)
    row_times = np.zeros(batch, np.int64)
    event_indices = np.full(batch, -1, np.int64)
    row_indices = np.full(batch, -1, np.int64)
    row_mask = np.asarray([not q.censored for q in queries], np.bool_)
    frame_counts, starts, crops, crop_valid = [], [], [], []
    for index, (q, chart, first, count) in enumerate(zip(queries, charts, first_bins, bin_counts)):
        if (q.cursor_ms < -1 or not q.cursor_ms <= q.horizon_end_ms <= chart.duration_ms or
                q.raw.shape != (len(q.valid), 2, CONTENT_DIM) or len(q.raw) > history_limit or
                q.valid.dtype != np.bool_ or not q.valid.all() or
                chart.mel.ndim != 2 or chart.mel.shape[1] != 128 or len(chart.mel) == 0):
            raise ContractError('Query clocks, real history or canonical Mel exceed their valid envelope')
        bins = np.arange(first, first + count, dtype=np.int64)
        bin_ends = bins * 10 + 9
        timing_times[index, :count] = bin_ends
        timing_times[index, count:] = bin_ends[-1]
        timing_exact[index] = exact_features([q.replay] * max_bins, timing_times[index].tolist())
        native = (bins[:, None] * 10 + np.arange(10)).reshape(-1)
        valid = (native > q.cursor_ms) & (native <= q.horizon_end_ms)
        timing_valid[index, :count * 10] = valid
        timing_forced[index, :count * 10] = valid & (native == chart.duration_ms) & any(q.replay.occupancy)
        raw[index, :len(q.raw)] = q.raw
        history_valid[index, :len(q.raw)] = q.valid
        if q.censored:
            if q.target_time_ms is not None or q.target_actions is not None:
                raise ContractError('A censored interval cannot contain a target row')
            row_times[index] = max(0, q.cursor_ms)
        else:
            if (q.target_time_ms is None or q.target_actions not in ROW_ACTIONS or
                    not q.cursor_ms < q.target_time_ms <= q.horizon_end_ms):
                raise ContractError('Target row must occur inside its observed native interval')
            row_times[index] = q.target_time_ms
            event_indices[index] = q.target_time_ms - first * 10
            row_indices[index] = ROW_ACTIONS.index(q.target_actions)
        frames = len(chart.mel)
        # Include the entire first bin, so choosing a row before its bin-end
        # query cannot change the audio crop or leak its target position.
        first_time = min(first * 10, max(0, q.cursor_ms))
        low = np.clip((first_time - MEL_CENTER_ORIGIN_MS) / MEL_HOP_MS, 0., frames - 1)
        high = np.clip((bin_ends[-1] - MEL_CENTER_ORIGIN_MS) / MEL_HOP_MS, 0., frames - 1)
        start = int(np.floor(low)) - halo
        stop = min(frames - 1, int(np.floor(high)) + 1) + halo + 1
        crop = np.zeros((stop - start, 128), np.float32)
        mask = np.zeros(stop - start, np.bool_)
        lo, hi = max(start, 0), min(stop, frames)
        crop[lo - start:hi - start] = chart.mel[lo:hi]
        mask[lo - start:hi - start] = True
        starts.append(start)
        frame_counts.append(frames)
        crops.append(crop)
        crop_valid.append(mask)
    max_frames = max(map(len, crops))
    mel = np.zeros((batch, max_frames, 128), np.float32)
    mel_valid = np.zeros((batch, max_frames), np.bool_)
    for index, (crop, mask) in enumerate(zip(crops, crop_valid)):
        mel[index, :len(crop)] = crop
        mel_valid[index, :len(mask)] = mask
    replays = [q.replay for q in queries]
    row_exact = exact_features(replays, row_times.tolist())
    row_legal = legal_rows(replays, [int(t) == c.duration_ms for t, c in zip(row_times, charts)])
    if any(scored and not support[target] for scored, support, target in zip(row_mask, row_legal, row_indices)):
        raise ContractError('Target row violates exact occupancy or true terminal closure')

    def tensor(values):
        return torch.as_tensor(values, device=device)

    inputs = JointInputs(tensor(mel), tensor(mel_valid), tensor(np.asarray(starts, np.int64)),
        tensor(np.asarray(frame_counts, np.int64)), tensor(raw), tensor(history_valid),
        tensor(np.asarray([q.truncated for q in queries], np.bool_)), tensor(timing_times),
        tensor(timing_exact), tensor(timing_valid), tensor(timing_forced), tensor(row_times),
        tensor(row_exact), tensor(row_legal), tensor(np.asarray([q.replay.occupancy for q in queries], np.bool_)))
    return JointBatch(inputs, JointTargets(tensor(event_indices), tensor(row_indices), tensor(row_mask)))


def score_batch(model: JointAudioModel, inputs: JointInputs) -> JointScores:
    """Evaluate timing and conditional row distributions without reading labels."""
    if getattr(model.config, 'global_audio', False):
        raise ContractError('Global audio queries require full-song context; use the context query scorer')
    encoded = model.encode_audio(inputs.mel, inputs.mel_valid)
    history = model.encode_history(inputs.raw, inputs.history_valid, inputs.truncated)
    timing_audio = interpolate_audio(encoded, inputs.timing_times_ms, inputs.mel_starts, inputs.mel_frame_counts)
    row_audio = interpolate_audio(encoded, inputs.row_times_ms, inputs.mel_starts, inputs.mel_frame_counts)
    timing = model.timing_logits(timing_audio, history, inputs.timing_exact).flatten(1)
    rows = model.row_log_probs(row_audio, history, inputs.row_exact, inputs.row_legal, inputs.occupancy)
    return JointScores(timing, rows)


def batch_losses(scores: JointScores, batch: JointBatch) -> JointLosses:
    """Return unreduced likelihood terms, with exactly zero row loss when censored."""
    inputs, targets = batch.inputs, batch.targets
    timing = hazard_nll(scores.timing_logits, inputs.timing_valid, targets.event_index, inputs.timing_forced)
    row = torch.zeros_like(timing)
    selected = targets.row_mask
    row[selected] = -scores.row_log_probs[selected, targets.row_index[selected]]
    return JointLosses(timing, row, timing + row)
