"""Generate a typed score ahead of R1; revise only unpublished control scopes."""
from dataclasses import asdict, dataclass
import time

import numpy as np
import torch

from ..bounded_typed_continuation.contract import ROW_ACTIONS
from ..bounded_typed_continuation.features import content_features
from ..joint_audio_continuation.batching import interpolate_audio
from ..joint_audio_continuation.generation import NativeGeneration
from ..joint_audio_continuation.state import exact_features
from ..joint_audio_continuation.timing import sample_hazards
from ..oracle_time_continuation.replay import ExactReplayState, commit
from ..oracle_time_continuation.schema import CompleteRow
from ..planned_audio_continuation.features import HeadPreview, consequences
from .controls import ControlSchedule
from .allocation import Allocation, ln_episodes
from .program import HEADS, MARKS, Resources, Recovery, bind_row, head_tokens, preview, row_support, tokens
from .proportions import tilt_ln_count


@dataclass
class PlanPoint:
    cursor: int
    state: Resources
    cache: object
    rng: torch.Tensor
    head_cache: object = None
    allocation: Allocation = Allocation()


class TypedSession:
    @torch.inference_mode()
    def __init__(self, model, mel, duration_ms, controls, *, seed=251925, recovery=Recovery(),
                 ln_feedback=None):
        model.eval()
        self.model, self.controls, self.recovery = model, controls, recovery
        self.ln_feedback = ln_feedback
        self.allocation = Allocation()
        self.ln_scopes = ln_episodes(controls) if ln_feedback is not None else ()
        self.device = next(model.parameters()).device
        self.duration = duration_ms
        started = time.perf_counter()
        self.encoded = model.body.encode_audio(self.tensor(np.array(mel, copy=True))[None])
        self.audio_seconds = time.perf_counter()-started
        self.rng = torch.Generator(device='cpu').manual_seed(seed)
        self.row_rng = torch.Generator(device='cpu').manual_seed(seed ^ 0xA301)
        self.state, self.cursor, self.residual = Resources(), -1, None
        self.cache = model.plan_temporal.empty_cache()
        self.head_cache = model.head_temporal.empty_cache() if model.head_stream else None
        self.row_cache = model.body.temporal.empty_cache()
        self.replay, self.bindings = ExactReplayState(), (None,)*4
        self.rows, self.queue = [], []
        self.coverage = -1
        self.published_point = self.point()

    def tensor(self, a, dtype=torch.float32):
        return torch.as_tensor(a, dtype=dtype, device=self.device)

    def point(self):
        return PlanPoint(self.cursor, self.state, self.cache, self.rng.get_state(), self.head_cache,
                         self.allocation)

    def audio(self, times):
        return interpolate_audio(self.encoded, self.tensor(times, torch.long)[None],
            self.tensor([0], torch.long), self.tensor([self.encoded.shape[1]], torch.long))[0]

    @torch.inference_mode()
    def next_event(self):
        while self.cursor < self.duration:
            end = min(self.cursor+500, self.duration)
            anchors = np.arange((self.cursor+1)//10, end//10+1)*10+9
            native = anchors[:, None]-9+np.arange(10)
            valid = (native > self.cursor) & (native <= end)
            support = self.state.type_support(native, self.duration, self.recovery)
            history = self.model.plan_temporal.read(self.cache)[None].expand(len(anchors), -1, -1)
            extra = {}
            if self.model.bounded_clock:
                extra['head_clocks'] = self.tensor(self.state.clocks(anchors, self.duration, head_phase=True))
            if self.model.head_stream:
                extra = dict(head_history=self.model.head_temporal.read(self.head_cache)[None].expand(len(anchors), -1, -1),
                             head_clocks=self.tensor(self.state.clocks(anchors, self.duration,
                                 remaining_availability=True, head_phase=True)))
            lp = self.model.clock_log_probs(self.audio(anchors), history,
                self.tensor(self.state.clocks(anchors, self.duration, remaining_availability=self.model.head_stream)),
                self.tensor(self.controls.at(anchors)), self.tensor(support, torch.bool), **extra)
            hazard = torch.logsumexp(lp[..., 1:], -1)-lp[..., 0]
            index, self.residual = sample_hazards(hazard.flatten(), self.tensor(valid.reshape(-1), torch.bool),
                self.tensor((~support[..., 0]).reshape(-1), torch.bool), self.rng, self.residual)
            if index is None:
                self.cursor = end
                continue
            now = int(native.reshape(-1)[index])
            kind = 1+int(torch.multinomial(lp.reshape(-1, 3)[index, 1:].softmax(-1).cpu(), 1, generator=self.rng))
            allowed = self.state.support(now, self.duration, self.recovery) & ((HEADS > 0) == (kind == 1))
            logp = self.model.mark_log_probs(self.audio([now]), self.model.plan_temporal.read(self.cache)[None],
                self.tensor(self.state.clocks(np.array([now]), self.duration, remaining_availability=self.model.head_stream)),
                self.tensor(self.controls.at([now])), self.tensor(allowed[None], torch.bool))[0]
            if self.ln_feedback is not None:
                span = next((s for s in self.ln_scopes if s.start_ms <= now < s.end_ms), None)
                self.allocation = self.allocation.in_scope(span)
                shift = self.tensor([self.ln_feedback.log_odds_shift(self.allocation)])
                logp = tilt_ln_count(logp[None], self.tensor(allowed[None], torch.bool), shift.sigmoid(), .5)[0]
            mark = int(torch.multinomial(logp.exp().cpu(), 1, generator=self.rng))
            if self.ln_feedback is not None:
                self.allocation = self.allocation.advance(*MARKS[mark][:2])
            previous = self.state.previous
            previous_head = self.state.last_head
            self.state, fresh = self.state.advance(now, mark, self.recovery)
            raw = tokens([now], [mark], [previous])[0]
            self.cache = self.model.plan_temporal.append(self.cache, self.tensor(raw))
            if self.model.head_stream and HEADS[mark]:
                self.head_cache = self.model.head_temporal.append(self.head_cache,
                    self.tensor(head_tokens([now], [mark], [previous_head])[0]))
            self.cursor, self.residual = now, None
            self.queue.append(((now, mark), fresh, self.point()))
            return

    def fill(self):
        while len(self.queue) < self.model.config.lookahead and self.cursor < self.duration:
            self.next_event()

    @torch.inference_mode()
    def materialize(self):
        self.fill()
        if not self.queue:
            return None
        (now, mark), fresh, point = self.queue[0]
        events = [q[0] for q in self.queue]
        support = row_support(self.replay, self.bindings, now, mark, self.duration, self.recovery)
        if not support.any():
            raise RuntimeError(f'Resource program has no column realization at {now}')
        future_h = tuple(t for t, m in events[1:] if HEADS[m])
        local, timing = consequences([self.replay], [now],
            [HeadPreview(future_h, self.cursor == self.duration)], self.duration)
        logp = self.model.row_log_probs(self.audio([now]), self.model.body.temporal.read(self.row_cache)[None],
            self.tensor(exact_features([self.replay], [now])), self.tensor(support[None], torch.bool),
            self.tensor([self.replay.occupancy], torch.bool),
            self.tensor(preview(events, now, self.model.config.lookahead, self.bindings)[None]),
            self.tensor(self.controls.at([now])), self.tensor(local), self.tensor(timing))[0]
        choice = int(torch.multinomial(logp.exp().cpu(), 1, generator=self.row_rng))
        row = CompleteRow(now, ROW_ACTIONS[choice])
        previous = None if self.replay.last_row is None else self.replay.last_row.time_ms
        raw = content_features([row], [previous], [(None,)*4])[0]
        self.row_cache = self.model.body.temporal.append(self.row_cache, self.tensor(raw))
        self.replay = commit(self.replay, row, is_terminal=now == self.duration)
        self.bindings = bind_row(self.bindings, mark, row.actions, fresh)
        self.rows.append(row); self.queue.pop(0)
        self.coverage, self.published_point = now, point
        return row

    def publish_to(self, end_ms, *, minimum_rows=0):
        """Settle an audio-clock interval, optionally waiting for startup rows."""
        while True:
            self.fill()
            if not self.queue:
                self.coverage = self.duration
                break
            if self.queue[0][0][0] > end_ms and len(self.rows) >= minimum_rows:
                self.coverage = max(self.coverage, end_ms)
                break
            self.materialize()
        return self.coverage

    def update_controls(self, span):
        """Invalidate unpublished future planning; retain all published rows/LNs.

        Confirmed empty time before the changed scope remains observed. Its
        residual survival budget is resampled conditionally, not replayed under
        the new law. Existing future events before the scope remain in queue.
        """
        if span.start_ms <= self.coverage:
            raise ValueError('A control update must start after published coverage')
        self.controls = ControlSchedule((*self.controls.spans, span), self.controls.style_names)
        if self.ln_feedback is not None:
            self.ln_scopes = ln_episodes(self.controls)
        old_cursor = self.cursor
        self.queue = [q for q in self.queue if q[0][0] < span.start_ms]
        point = self.queue[-1][2] if self.queue else self.published_point
        self.cursor = max(point.cursor, self.coverage, min(old_cursor, span.start_ms-1))
        self.state, self.cache = point.state, point.cache
        self.head_cache = point.head_cache
        self.allocation = point.allocation
        self.rng.set_state(point.rng)
        self.residual = None


@torch.inference_mode()
def rollout(model, mel, duration_ms, controls, *, seed=251925, max_seconds=120., on_window=None,
            ln_feedback=None):
    started = time.perf_counter()
    session = TypedSession(model, mel, duration_ms, controls, seed=seed, ln_feedback=ln_feedback)
    windows, first30 = [], None
    while session.coverage < duration_ms:
        before = time.perf_counter()
        target = min(duration_ms, max(0, session.coverage)+8000)
        session.publish_to(target, minimum_rows=30 if not session.rows else 0)
        elapsed = time.perf_counter()-started
        if first30 is None and len(session.rows) >= 30:
            first30 = elapsed
        windows.append(dict(coverage_ms=session.coverage, service_seconds=time.perf_counter()-before, rows=len(session.rows)))
        if on_window:
            on_window(session)
        if elapsed > max_seconds or len(session.rows) > 30000:
            break
    completed = session.coverage == duration_ms and not any(session.replay.occupancy)
    metrics = dict(audio_seconds=session.audio_seconds, generation_seconds=time.perf_counter()-started,
        startup_seconds=windows[0]['service_seconds']+session.audio_seconds, first30_rows_seconds=first30,
        windows=windows, row_count=len(session.rows), controls=[vars(s) for s in session.controls.spans],
        ln_feedback=asdict(ln_feedback) if ln_feedback is not None else None)
    return NativeGeneration(tuple(session.rows), completed, 'complete' if completed else 'budget', session.coverage, metrics)
