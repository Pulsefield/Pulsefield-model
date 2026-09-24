"""Lazy head lookahead, online releases and exact row commitment from full audio."""
import hashlib
import io
from pathlib import Path
import time

import numpy as np
import torch

from ..bounded_typed_continuation.contract import ROW_ACTIONS
from ..bounded_typed_continuation.features import content_features
from ..joint_audio_continuation.batching import interpolate_audio
from ..joint_audio_continuation.generation import GenerationUpdate, NativeGeneration, _synchronize
from ..joint_audio_continuation.state import exact_features
from ..joint_audio_continuation.timing import sample_hazards
from ..oracle_time_continuation.replay import ExactReplayState, commit
from ..oracle_time_continuation.schema import CompleteRow
from ..scoped_style_modeling.dataset import ContractError
from .features import (
    HeadPreview, LNProjection, consequences, head_clocks, preview_features,
    release_clocks, release_masks, row_support, skeleton_tokens,
)
from .model import PlannedAudioModel, PlannedModelConfig


class _BudgetStop(Exception):
    pass


class HeadPlanner:
    """A pure head-stream sampler; row decisions never change its state or RNG."""
    def __init__(self, model, encoded, duration_ms, seed, chunk_ms=500, guard=None):
        self.model, self.encoded, self.duration_ms = model, encoded, duration_ms
        self.device, self.dtype = encoded.device, encoded.dtype
        self.rng = torch.Generator(device='cpu').manual_seed(seed)
        self.chunk_ms, self.guard = chunk_ms, guard
        self.cursor, self.last_head, self.residual = -1, None, None
        self.cache = model.head_temporal.empty_cache()
        self.queue, self.generated = [], []
        self.bins = 0

    @property
    def finished(self):
        return self.cursor == self.duration_ms

    @torch.no_grad()
    def fill(self, count):
        while len(self.queue) < count and not self.finished:
            if self.guard is not None:
                self.guard()
            end = min(self.cursor + self.chunk_ms, self.duration_ms)
            bins = torch.arange((self.cursor + 1) // 10, end // 10 + 1, device=self.device)
            anchors = bins * 10 + 9
            audio = interpolate_audio(self.encoded, anchors[None])[0]
            clocks = torch.as_tensor(head_clocks([self.last_head] * len(bins), anchors.cpu().numpy()),
                                     dtype=self.dtype, device=self.device)
            history = self.model.head_temporal.read(self.cache)[None].expand(len(bins), -1, -1)
            logits = self.model.head_logits(audio, history, clocks).flatten()
            native = (bins[:, None] * 10 + torch.arange(10, device=self.device)).flatten()
            valid = (native > self.cursor) & (native <= end)
            event, self.residual = sample_hazards(logits, valid, torch.zeros_like(valid), self.rng, self.residual)
            self.bins += len(bins)
            if event is None:
                self.cursor = end
                continue
            self.cursor = int(native[event])
            token = skeleton_tokens([self.cursor], [self.last_head])[0]
            self.cache = self.model.head_temporal.append(self.cache, torch.as_tensor(token, dtype=self.dtype, device=self.device))
            self.last_head, self.residual = self.cursor, None
            self.queue.append(self.cursor)
            self.generated.append(self.cursor)

    def preview(self, now, count):
        future = tuple(t for t in self.queue if t > now)[:count]
        return HeadPreview(future, self.finished and len(future) < count)


@torch.inference_mode()
def rollout(model, mel, duration_ms, *, seed, chunk_ms=500, head_chunk_ms=500,
            max_rows=30000, max_seconds=90., on_update=None, stop_callback=None):
    if (type(duration_ms) is not int or duration_ms < 0 or
            any(type(v) is not int or v <= 0 for v in (chunk_ms, head_chunk_ms, max_rows)) or
            not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ContractError('Planned rollout requires a finite audio clock and positive resource bounds')
    model.eval()
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    _synchronize(device)
    started = time.perf_counter()
    encoded = model.encode_audio(torch.as_tensor(np.array(mel, copy=True), dtype=dtype, device=device)[None])
    _synchronize(device)
    audio_seconds = time.perf_counter() - started
    rows, coverage, latencies = [], [], []
    cursor, residual = -1, None
    replay = ExactReplayState()
    row_cache, skeleton_cache = model.temporal.empty_cache(), model.skeleton_temporal.empty_cache()
    release_rng = torch.Generator(device='cpu').manual_seed(seed ^ 0x4E51)
    row_rng = torch.Generator(device='cpu').manual_seed(seed ^ 0xA301)

    def guard():
        if time.perf_counter() - started >= max_seconds:
            raise _BudgetStop('time_limit')
        if len(rows) >= max_rows:
            raise _BudgetStop('row_limit')
        if stop_callback is not None:
            reason = stop_callback()
            if reason:
                raise _BudgetStop(str(reason))

    planner = HeadPlanner(model, encoded, duration_ms, seed ^ 0x17AB, head_chunk_ms, guard)
    startup = first30_rows = first30_heads = None
    first30_rows_clock = first30_heads_clock = None
    release_bins = deadline_events = terminal_events = 0
    reason = 'completed'

    def tensor(value, dtype_override=None):
        return torch.as_tensor(value, dtype=dtype if dtype_override is None else dtype_override, device=device)

    try:
        while cursor < duration_ms:
            guard()
            tick = time.perf_counter()
            planner.fill(model.config.lookahead + 1)
            next_h = planner.queue[0] if planner.queue else None
            if next_h is None and not planner.finished:
                raise ContractError('A missing head queue is not audio termination')
            event_time, head_role, forced_event = None, False, False
            previous = None if replay.last_row is None else replay.last_row.time_ms
            if not any(replay.occupancy):
                if next_h is None:
                    cursor = duration_ms
                else:
                    event_time, head_role = next_h, True
            else:
                end = min(cursor + chunk_ms, duration_ms if next_h is None else next_h - 1)
                if end > cursor:
                    bins = torch.arange((cursor + 1) // 10, end // 10 + 1, device=device)
                    anchors = bins * 10 + 9
                    native = bins[:, None] * 10 + torch.arange(10, device=device)
                    projection = LNProjection(replay.open_ln_start_ms, cursor)
                    preview = planner.preview(cursor, model.config.lookahead)
                    previews, states = [preview] * len(bins), [projection] * len(bins)
                    clocks = release_clocks(states, [previous] * len(bins), anchors.cpu().numpy(), previews, duration_ms)
                    audio = interpolate_audio(encoded, anchors[None])[0]
                    history = model.skeleton_temporal.read(skeleton_cache)[None].expand(len(bins), -1, -1)
                    logits = model.release_logits(audio, history, tensor(clocks)).flatten()
                    valid, forced = release_masks(states, native.cpu().numpy(), previews, duration_ms)
                    valid &= native.cpu().numpy() <= end
                    forced &= valid
                    event, residual = sample_hazards(logits, tensor(valid, torch.bool).flatten(),
                        tensor(forced, torch.bool).flatten(), release_rng, residual)
                    release_bins += len(bins)
                    if event is not None:
                        event_time = int(native.flatten()[event])
                        forced_event = bool(forced.reshape(-1)[event])
                    else:
                        cursor = end
                elif next_h is not None:
                    if all(replay.occupancy):
                        raise ContractError('Planned head reached an all-held state without a release clock')
                    event_time, head_role = next_h, True
            published = None
            if event_time is not None:
                cursor = event_time
                preview = planner.preview(cursor, model.config.lookahead)
                legal = row_support([replay], [cursor], [head_role], [preview], duration_ms)
                context = preview_features([preview], [cursor], [head_role], duration_ms, model.config.lookahead)
                local, future = consequences([replay], [cursor], [preview], duration_ms)
                audio = interpolate_audio(encoded, torch.tensor([cursor], device=device))
                log_probs = model.planned_row_log_probs(audio, model.temporal.read(row_cache)[None],
                    tensor(exact_features([replay], [cursor])), tensor(legal, torch.bool),
                    tensor([replay.occupancy], torch.bool), tensor(context), tensor(local), tensor(future))[0]
                index = int(torch.multinomial(log_probs.detach().cpu().double().exp(), 1, generator=row_rng))
                published = CompleteRow(cursor, ROW_ACTIONS[index])
                replay = commit(replay, published, is_terminal=cursor == duration_ms)
                row_cache = model.temporal.append(row_cache,
                    tensor(content_features([published], [previous], [[None] * 4])[0]))
                skeleton_cache = model.skeleton_temporal.append(skeleton_cache,
                    tensor(skeleton_tokens([cursor], [previous], [head_role])[0]))
                rows.append(published)
                if head_role:
                    if not planner.queue or planner.queue.pop(0) != cursor:
                        raise ContractError('Committed head differs from the immutable head plan')
                if forced_event:
                    terminal_events += cursor == duration_ms
                    deadline_events += cursor != duration_ms
                residual = None
            _synchronize(device)
            elapsed = time.perf_counter() - started
            latencies.append(time.perf_counter() - tick)
            coverage.append(dict(coverage_ms=cursor, elapsed_seconds=elapsed))
            if startup is None and cursor >= min(8000, duration_ms):
                startup = elapsed
            if first30_rows is None and len(rows) >= 30:
                first30_rows, first30_rows_clock = elapsed, cursor
            if first30_heads is None and replay.note_count >= 30:
                first30_heads, first30_heads_clock = elapsed, cursor
            if on_update is not None:
                on_update(GenerationUpdate(published, cursor, cursor == duration_ms))
    except _BudgetStop as error:
        reason = str(error)
    completed = cursor == duration_ms
    if completed and any(replay.occupancy):
        raise ContractError('Completed planned generation retains an LN')
    return NativeGeneration(tuple(rows), completed, reason, cursor, dict(
        audio_encode_seconds=audio_seconds, generation_seconds=time.perf_counter() - started,
        rows=len(rows), heads=replay.note_count, open_lanes=list(replay.occupancy),
        first30_rows_seconds=first30_rows, first30_rows_through_ms=first30_rows_clock,
        first30_heads_seconds=first30_heads, first30_heads_through_ms=first30_heads_clock,
        startup_seconds=startup, startup_target_ms=min(8000, duration_ms),
        step_p50_ms=float(np.quantile(latencies, .5) * 1000) if latencies else None,
        step_p99_ms=float(np.quantile(latencies, .99) * 1000) if latencies else None,
        scheduler_steps=len(latencies), head_bins_scored=planner.bins, release_bins_scored=release_bins,
        forced_deadline_releases=deadline_events, forced_terminal_releases=terminal_events,
        planned_heads=len(planner.generated), coverage=coverage,
        latency_scope='cached canonical Mel through published coverage; excludes waveform decode and Mel computation'))


def load_model(path, expected_sha256, *, device='cpu'):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ContractError('Planned checkpoint differs from its pinned SHA-256')
    payload = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    if payload.get('format') != 'joint-audio/planned-v1':
        raise ContractError('Expected a planned audio checkpoint')
    model = PlannedAudioModel(PlannedModelConfig(**payload['model_config']))
    model.load_state_dict(payload['model'], strict=True)
    if any(not bool(torch.isfinite(v).all()) for v in model.state_dict().values()) or not bool((model.audio_std > 0).all()):
        raise ContractError('Planned checkpoint has nonfinite tensors or invalid audio normalization')
    return model.to(device).eval(), {k: payload[k] for k in ('source_revision', 'manifest_sha256', 'config')}
