"""Joint head/release/row likelihood with the same native clock as generation."""
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch.nn import functional as F

from ..bounded_typed_continuation.contract import Arm
from ..joint_audio_continuation.batching import interpolate_audio
from ..joint_audio_continuation.intervals import IntervalInputs, _pad_first, collate_interval as collate_rows
from ..scoped_style_modeling.dataset import ContractError
from .features import (
    LNProjection, consequences, head_clocks, preview_after, preview_features,
    release_clocks, release_masks, row_support, skeleton_tokens,
)
from .release import conditioned_release_logits
from .counts import count_state, count_tokens
from .spacing import allowed_rows as spaced_rows, check_head_capacity, release_limits


@dataclass(frozen=True)
class ReleaseWaitInputs:
    times: torch.Tensor
    history: torch.Tensor
    clocks: torch.Tensor
    native_indices: torch.Tensor
    destinations: torch.Tensor
    offsets: torch.Tensor


@dataclass(frozen=True)
class PlannedInputs:
    base: IntervalInputs
    head_raw: torch.Tensor
    head_history_valid: torch.Tensor
    head_history: torch.Tensor
    head_clock: torch.Tensor
    skeleton_raw: torch.Tensor
    release_clock: torch.Tensor
    release_valid: torch.Tensor
    release_forced: torch.Tensor
    row_preview: torch.Tensor
    consequence_local: torch.Tensor
    consequence_timing: torch.Tensor
    release_waits: tuple[ReleaseWaitInputs, ...]
    count_raw: torch.Tensor | None = None
    count_clock: torch.Tensor | None = None
    head_valid: torch.Tensor | None = None
    response_allowed: torch.Tensor | None = None


@dataclass(frozen=True)
class PlannedBatch:
    inputs: PlannedInputs
    head_event: torch.Tensor
    release_event: torch.Tensor
    row_index: torch.Tensor
    weight_per_second: float


@dataclass(frozen=True)
class PlannedScores:
    head: torch.Tensor
    release: torch.Tensor
    row: torch.Tensor


def collate_interval(example, config, device='cpu'):
    """Keep teacher head plans distinct from future row/LN materialization."""
    base = collate_rows(example, config, 'cpu')
    x = base.inputs
    source = example.chart.source
    times = source.rows['time']
    roles = np.isin(source.rows['actions'], (1, 2)).any(-1)
    head_indices = np.flatnonzero(roles)
    heads = times[head_indices]
    gap = config.minimum_action_gap_ms
    if gap:
        check_head_capacity(heads, gap)
    first = int(np.searchsorted(times, example.start_ms))
    stop = int(np.searchsorted(times, example.end_ms))
    row_start = max(0, first - (2 ** (config.history_levels + 1) - 1))
    head_start = max(0, int(np.searchsorted(head_indices, first)) - (2 ** (config.head_levels + 1) - 1))
    head_stop = int(np.searchsorted(head_indices, stop))
    head_positions = np.arange(head_start, head_stop)
    previous_heads = [heads[i - 1] if i else None for i in head_positions]
    head_raw = skeleton_tokens(heads[head_positions], previous_heads)
    skeleton_raw = skeleton_tokens(times[row_start:stop],
        [times[i - 1] if i else None for i in range(row_start, stop)], roles[row_start:stop])

    query_indices = x.timing_history.numpy() + row_start + 1
    replays = [replace(source.state(Arm.R0, int(i)).replay, is_complete=False) for i in query_indices]
    previous_skeleton = [times[i - 1] if i else None for i in query_indices]
    observed = [max(example.start_ms - 1, int(t) if t is not None else -1) for t in previous_skeleton]
    ln = [LNProjection(s.open_ln_start_ms, t) for s, t in zip(replays, observed)]
    previews = [preview_after(heads, t, config.lookahead) for t in observed]
    query_head_positions = np.searchsorted(head_indices, query_indices) - 1
    clocks = head_clocks([heads[i] if i >= 0 else None for i in query_head_positions], x.timing_times.numpy())
    r_clocks = release_clocks(ln, previous_skeleton, x.timing_times.numpy(), previews, example.chart.duration_ms)
    native = x.timing_times.numpy()[:, None] - 9 + np.arange(10)[None]
    r_valid, r_forced = release_masks(ln, native, previews, example.chart.duration_ms,
                                    minimum_action_gap_ms=gap)
    r_valid &= x.timing_valid.numpy()
    r_forced &= r_valid
    h_valid = x.timing_valid.numpy().copy()
    if gap:
        earliest_heads = np.asarray([heads[i-3]+gap if i >= 3 else 0 for i in query_head_positions])
        h_valid &= native >= earliest_heads[:, None]

    def tensor(value, dtype=None):
        return torch.as_tensor(value, dtype=dtype, device=device)

    waits = []
    last_audio_query = int(x.timing_times.max())
    if config.condition_full_holds or gap:
        # Each prefix defines a hypothetical unchanged LN state through its
        # deadline. Actual future tails select targets, never the waiting bound.
        for prefix in np.unique(query_indices[r_valid.any(-1)]):
            i = int(np.flatnonzero(query_indices == prefix)[0])
            start = observed[i]
            if gap:
                earliest, deadline = release_limits(ln[i], previews[i], example.chart.duration_ms, gap)
                if previews[i].times_ms and deadline >= previews[i].times_ms[0]:
                    continue
                begin = max(start+1, int(earliest))
            else:
                if not all(t is not None for t in ln[i].starts_ms) or not previews[i].times_ms:
                    continue
                deadline = previews[i].times_ms[0]-1
                begin = start+1
            bins = np.arange(begin // 10, deadline // 10 + 1)
            anchors = bins * 10 + 9
            padded = _pad_first(anchors, edge=True)
            clocks_r = release_clocks([ln[i]] * len(anchors), [previous_skeleton[i]] * len(anchors),
                                     anchors, [previews[i]] * len(anchors), example.chart.duration_ms)
            destinations = np.flatnonzero(((query_indices == prefix)[:, None] & r_valid).reshape(-1))
            waits.append(ReleaseWaitInputs(tensor(padded),
                tensor(np.full(len(padded), int(x.timing_history[i])), torch.long),
                tensor(_pad_first(clocks_r, edge=True)),
                tensor(np.arange(begin, deadline + 1) - bins[0] * 10, torch.long),
                tensor(destinations, torch.long), tensor(native.reshape(-1)[destinations] - begin, torch.long)))
            last_audio_query = max(last_audio_query, int(anchors[-1]))

    if waits:
        # Refill from complete audio: formerly padded crop positions can become
        # real frames when the hypothetical wait extends past the interval.
        frames, audio_start = len(example.chart.mel), int(x.mel_start[0])
        high = np.clip((last_audio_query - 20) / 10, 0., frames - 1)
        audio_stop = min(frames - 1, int(np.floor(high)) + 1) + config.audio_halo_frames + 1
        mel = np.zeros((audio_stop - audio_start, 128), np.float32)
        valid = np.zeros(len(mel), np.bool_)
        low, high = max(0, audio_start), min(frames, audio_stop)
        mel[low - audio_start:high - audio_start] = example.chart.mel[low:high]
        valid[low - audio_start:high - audio_start] = True
        x = replace(x, mel=torch.from_numpy(_pad_first(mel)[None]),
                    mel_valid=torch.from_numpy(_pad_first(valid)[None]))
    target_head = np.asarray([bool(roles[i]) if i < len(roles) else False for i in query_indices])
    head_event = base.targets.timing_event.numpy() & target_head[:, None]
    release_event = base.targets.timing_event.numpy() & ~target_head[:, None]
    if np.any(head_event & ~h_valid):
        raise ContractError('Source head violates the minimum-spacing capacity')
    if np.any(release_event & ~r_valid) or np.any(r_forced & ~release_event):
        raise ContractError('Source release likelihood violates the planned head deadline or LN state')

    row_indices = x.row_history.numpy() + row_start + 1
    row_states = [replace(source.state(Arm.R0, int(i)).replay, is_complete=False) for i in row_indices]
    row_times = x.row_times.numpy()
    row_roles = roles[row_indices]
    row_previews = [preview_after(heads, t, config.lookahead) for t in row_times]
    support = row_support(row_states, row_times, row_roles, row_previews, example.chart.duration_ms)
    if any(not support[i, target] for i, target in enumerate(base.targets.row_index.tolist())):
        raise ContractError('Source row cannot realize its head plan')
    response_allowed = None
    if gap:
        response_allowed = np.stack([spaced_rows(state, int(t), preview, example.chart.duration_ms, gap)
            for state, t, preview in zip(row_states, row_times, row_previews)]) if len(row_states) else np.empty((0,256),bool)
        if any(not response_allowed[i,target] for i,target in enumerate(base.targets.row_index.tolist())):
            raise ContractError('Source row violates action spacing or its continuation support')
    row_preview = preview_features(row_previews, row_times, row_roles, example.chart.duration_ms, config.lookahead)
    local, future = consequences(row_states, row_times, row_previews, example.chart.duration_ms)
    count_raw = count_clock = None
    if config.row_factorization == 'count_layout':
        count_raw = tensor(_pad_first(count_tokens(times[row_start:stop],
            [times[i - 1] if i else None for i in range(row_start, stop)],
            source.rows['actions'][row_start:stop]))[None])
        count_clock = tensor(count_state([s.open_ln_start_ms for s in row_states],
            [times[i - 1] if i else None for i in row_indices], row_times))

    x = replace(x, row_legal=torch.from_numpy(support))
    x = IntervalInputs(**{name: value.to(device) for name, value in vars(x).items()})
    inputs = PlannedInputs(x, tensor(_pad_first(head_raw)[None]),
        tensor(_pad_first(np.ones(len(head_raw), np.bool_))[None]),
        tensor(query_head_positions - head_start, torch.long), tensor(clocks),
        tensor(_pad_first(skeleton_raw)[None]), tensor(r_clocks), tensor(r_valid), tensor(r_forced),
        tensor(row_preview), tensor(local), tensor(future), tuple(waits), count_raw, count_clock,
        tensor(h_valid), None if response_allowed is None else tensor(response_allowed))
    return PlannedBatch(inputs, tensor(head_event), tensor(release_event), base.targets.row_index.to(device),
                        example.weight_per_second)


def _gather(module, encoded, indices):
    boundary = module.boundary[0].expand(len(indices), 2, -1)
    if encoded is None:
        return boundary
    return torch.where((indices >= 0)[:, None, None], encoded[indices.clamp_min(0)], boundary)


def score_interval(model, inputs, coarse, *, profile_index=None):
    x = inputs.base
    raw_audio = model.encode_crop(x.mel, x.mel_valid, x.mel_start, x.frame_count, coarse)
    encoded = model.condition_audio(raw_audio, profile_index)
    downstream = (encoded if model.config.profile_head_rate_downstream else
                  model.condition_audio(raw_audio, profile_index, downstream=True))
    row_history = model.temporal(x.raw, x.history_valid)[0] if x.raw.shape[1] else None
    skeleton_history = (model.skeleton_temporal(inputs.skeleton_raw, x.history_valid)[0]
                        if inputs.skeleton_raw.shape[1] else None)
    head_history = (model.head_temporal(inputs.head_raw, inputs.head_history_valid)[0]
                    if inputs.head_raw.shape[1] else None)
    audio = interpolate_audio(encoded, x.timing_times[None], x.mel_start, x.frame_count)[0]
    h = model.head_logits(audio, _gather(model.head_temporal, head_history, inputs.head_history), inputs.head_clock)
    if not model.config.profile_head_rate_downstream:
        audio = interpolate_audio(downstream, x.timing_times[None], x.mel_start, x.frame_count)[0]
    r = model.release_logits(audio, _gather(model.skeleton_temporal, skeleton_history, x.timing_history),
                             inputs.release_clock)
    for wait in inputs.release_waits:
        audio = interpolate_audio(downstream, wait.times[None], x.mel_start, x.frame_count)[0]
        raw = model.release_logits(audio, _gather(model.skeleton_temporal, skeleton_history, wait.history), wait.clocks)
        conditional = conditioned_release_logits(raw.flatten()[wait.native_indices])
        r = r.flatten().index_copy(0, wait.destinations, conditional[wait.offsets]).reshape_as(r)
    if len(x.row_times):
        audio = interpolate_audio(downstream, x.row_times[None], x.mel_start, x.frame_count)[0]
        counts = {}
        if model.config.row_factorization == 'count_layout':
            past = (model.row_counts.temporal(inputs.count_raw, x.history_valid)[0]
                    if inputs.count_raw.shape[1] else None)
            counts = dict(count_history=_gather(model.row_counts.temporal, past, x.row_history),
                          count_clock=inputs.count_clock)
        if model.config.minimum_action_gap_ms:
            counts['response_allowed'] = inputs.response_allowed
        rows = model.planned_row_log_probs(audio, _gather(model.temporal, row_history, x.row_history),
            x.row_exact, x.row_legal, x.occupancy, inputs.row_preview,
            inputs.consequence_local, inputs.consequence_timing, **counts)
    else:
        rows = h.new_empty((0, 256))
    return PlannedScores(h, r, rows)


def interval_losses(scores, batch):
    """Return head, release, row and joint NLL sums on the actual audio clock."""
    def binary(logits, event, valid):
        active = torch.where(valid, logits, 0.)
        return torch.where(event, F.softplus(-active), F.softplus(active)).masked_select(valid).sum()

    h = binary(scores.head, batch.head_event,
               batch.inputs.base.timing_valid if batch.inputs.head_valid is None else batch.inputs.head_valid)
    r = binary(scores.release, batch.release_event, batch.inputs.release_valid & ~batch.inputs.release_forced)
    row = (-scores.row[torch.arange(len(batch.row_index), device=h.device), batch.row_index].sum()
           if len(batch.row_index) else h * 0)
    return h, r, row, h + r + row
