"""Joint teacher likelihood for the resource plan and its conditional geometry."""
from dataclasses import replace

import numpy as np
import torch

from ..bounded_typed_continuation.contract import Arm
from ..joint_audio_continuation.batching import interpolate_audio
from ..joint_audio_continuation.intervals import collate_interval as collate_base, _pad_first
from ..planned_audio_continuation.features import HeadPreview, consequences
from ..planned_audio_continuation.intervals import _gather
from .program import HEADS, head_tokens, preview, row_support, tokens


def collate(example, model, program, controls, recovery, device):
    base = collate_base(example, model.config)
    x = base.inputs
    times = example.chart.source.rows['time']
    first, stop = np.searchsorted(times, [example.start_ms, example.end_ms])
    start = max(0, first-(2**(model.config.history_levels+1)-1))
    indices = x.timing_history.numpy()+start+1
    row_indices = x.row_history.numpy()+start+1
    duration = example.chart.duration_ms
    nrows = len(base.targets.row_index)
    marks = np.asarray([e[1] for e in program.events])
    raw = tokens(times[start:stop], marks[start:stop], [times[i-1] if i else None for i in range(start, stop)])
    clocks = np.stack([program.states[i].clocks(t, duration, remaining_availability=model.head_stream)
                       for i, t in zip(indices, x.timing_times.numpy())])
    native = x.timing_times.numpy()[:, None]-9+np.arange(10)
    support = np.stack([program.states[i].type_support(t, duration, recovery) for i, t in zip(indices, native)])
    role = np.array([1 if i < len(marks) and HEADS[marks[i]] else 2 for i in indices])
    target = np.where(base.targets.timing_event.numpy(), role[:, None], 0)
    rclocks, mark_support, row_allowed, plans, replays, previews = [], [], [], [], [], []
    for i, now in zip(row_indices, x.row_times.numpy()):
        now = int(now); m = int(marks[i])
        rclocks.append(program.states[i].clocks(now, duration, remaining_availability=model.head_stream))
        mark_support.append(program.states[i].support(now, duration, recovery) & ((HEADS > 0) == (HEADS[m] > 0)))
        replay = replace(example.chart.source.state(Arm.R0, int(i)).replay, is_complete=False)
        replays.append(replay)
        row_allowed.append(row_support(replay, program.bindings[i], now, m, duration, recovery))
        plans.append(preview(program.events[i:i+model.config.lookahead], now, model.config.lookahead, program.bindings[i]))
        ahead = tuple(t for t, m in program.events[i+1:] if HEADS[m])[:model.config.lookahead]
        previews.append(HeadPreview(ahead, len(ahead) < model.config.lookahead))
    if nrows:
        targets = base.targets.row_index.numpy()
        if not all(row_allowed[j][v] and mark_support[j][marks[row_indices[j]]] for j, v in enumerate(targets)):
            raise ValueError('Source row does not realize its typed resource program')
    local, future = consequences(replays, x.row_times.numpy(), previews, duration)
    def tensor(a, dtype=None):
        a = np.asarray(a)
        return torch.as_tensor(a, dtype=torch.float32 if dtype is None and a.dtype.kind == 'f' else dtype, device=device)
    extra = {}
    if model.head_stream:
        hi = np.flatnonzero(HEADS[marks] > 0)
        hfirst = max(0, int(np.searchsorted(hi, first))-model.head_temporal.config.receptive_tokens)
        hstop = int(np.searchsorted(hi, stop))
        positions = hi[hfirst:hstop]
        previous = [times[hi[j-1]] if j else None for j in range(hfirst, hstop)]
        extra = dict(head_raw=tensor(_pad_first(head_tokens(times[positions], marks[positions], previous))[None]),
            head_valid=tensor(_pad_first(np.ones(len(positions), bool))[None]),
            head_indices=tensor(np.searchsorted(hi, indices)-hfirst-1, torch.long),
            head_clocks=tensor(np.stack([program.states[i].clocks(t, duration,
                remaining_availability=True, head_phase=True) for i,t in zip(indices, x.timing_times.numpy())])))
    return dict(base=replace(x, **{k: v.to(device) for k, v in vars(x).items()}),
        raw=tensor(_pad_first(raw)[None]), clocks=tensor(clocks), support=tensor(support),
        control=tensor(controls.at(x.timing_times.numpy())), target=tensor(target, torch.long),
        row_clocks=tensor(rclocks), mark_support=tensor(mark_support, torch.bool),
        row_support=tensor(row_allowed, torch.bool), plan=tensor(plans),
        row_control=tensor(controls.at(x.row_times.numpy())), local=tensor(local), future=tensor(future),
        marks=tensor(marks[row_indices[:nrows]], torch.long), rows=base.targets.row_index.to(device), **extra)


def losses(model, batch, coarse):
    b, x = batch, batch['base']
    body = model.body
    encoded = body.encode_crop(x.mel, x.mel_valid, x.mel_start, x.frame_count, coarse)
    ph = model.plan_temporal(b['raw'], x.history_valid)[0] if b['raw'].shape[1] else None
    audio = interpolate_audio(encoded, x.timing_times[None], x.mel_start, x.frame_count)[0]
    extra = {}
    if model.head_stream:
        hh = model.head_temporal(b['head_raw'], b['head_valid'])[0] if b['head_raw'].shape[1] else None
        extra = dict(head_history=_gather(model.head_temporal, hh, b['head_indices']), head_clocks=b['head_clocks'])
    clock = model.clock_log_probs(audio, _gather(model.plan_temporal, ph, x.timing_history),
        b['clocks'], b['control'], b['support'], **extra)
    timing = -clock.gather(-1, b['target'][..., None]).squeeze(-1)[x.timing_valid].sum()
    n = len(b['rows'])
    if not n:
        return torch.stack((timing, timing*0, timing*0))
    audio = interpolate_audio(encoded, x.row_times[None], x.mel_start, x.frame_count)[0]
    mark = model.mark_log_probs(audio, _gather(model.plan_temporal, ph, x.row_history),
        b['row_clocks'], b['row_control'], b['mark_support'])
    rh = body.temporal(x.raw, x.history_valid)[0]
    row = model.row_log_probs(audio, _gather(body.temporal, rh, x.row_history), x.row_exact,
        b['row_support'], x.occupancy, b['plan'], b['row_control'], b['local'], b['future'])
    return torch.stack((timing, -mark[torch.arange(n), b['marks']].sum(), -row[torch.arange(n), b['rows']].sum()))
