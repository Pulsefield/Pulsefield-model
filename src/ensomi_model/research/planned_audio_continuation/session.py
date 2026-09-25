"""Branchable native continuation with immutable history caches and owned RNGs.

A fork shares only the frozen model, encoded full audio, immutable replay and
append-only cache values. It owns every queue, row list and random generator.
This is in-memory planning state, not a durable crash-recovery checkpoint.
"""
import copy
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
from .attack_response import short_attack_costs, short_attack_pairs, select_response_row
from .counts import count_state, count_tokens
from .features import (HeadPreview, LNProjection, consequences, preview_features,
                       release_clocks, release_masks, row_support, skeleton_tokens)
from .release import conditioned_release_logits
from .row_constraints import NoRowContinuation, condition_rows
from .spacing import allowed_rows as spaced_rows, check_head_capacity, release_limits


class _BudgetStop(Exception):
    pass


def _fork_rng(rng):
    if rng is None:
        return None
    result = torch.Generator(device='cpu')
    result.set_state(rng.get_state())
    return result


class FixedHeadPlan:
    """Research substitution of generated times; no future row or LN content."""
    def __init__(self, times, duration_ms, guard):
        times = tuple(times)
        if (any(type(t) is not int or not 0 <= t <= duration_ms for t in times) or
                any(a >= b for a, b in zip(times, times[1:]))):
            raise ContractError('Fixed H plan requires increasing native clocks within the audio')
        self.queue, self.generated = list(times), list(times)
        self.finished, self.bins, self.guard = True, 0, guard

    def fill(self, count):
        self.guard()

    def preview(self, now, count):
        future = tuple(t for t in self.queue if t > now)[:count]
        return HeadPreview(future, len(future) < count)

    def pop(self):
        return self.queue.pop(0)


class ContinuationSession:
    """One exact H/R/row trajectory, forkable before publication.

    Parameters and encoded audio must remain unchanged while any fork is live.
    A retry changes release/row RNGs only; the H and arrangement choices remain
    fixed. Observed empty coverage is preserved even when its release-survival
    threshold is resampled using the conditional exponential memoryless law.
    NoRowContinuation preserves prior rows/replay/coverage, but may have consumed
    release randomness or extended H lookahead. Discard that failed proposal;
    retry by forking the saved publication boundary, not the failed session.
    """
    @torch.inference_mode()
    def __init__(self, model, mel, duration_ms, *, seed, planner_factory,
                 chunk_ms=500, head_chunk_ms=500, max_rows=30000, max_seconds=90.,
                 stop_callback=None, correct_short_attacks=False,
                 arrangement_profile=None, head_times=None, row_constraint='none', controls=None):
        if (type(duration_ms) is not int or duration_ms < 0 or
                any(type(v) is not int or v <= 0 for v in (chunk_ms, head_chunk_ms, max_rows)) or
                not np.isfinite(max_seconds) or max_seconds <= 0 or
                type(correct_short_attacks) is not bool):
            raise ContractError('Planned rollout requires a finite audio clock and positive resource bounds')
        model.eval()
        if row_constraint not in ('none', 'current', 'preview'):
            raise ContractError('Row constraint must be none, current or preview')
        if row_constraint != 'none' and correct_short_attacks:
            raise ContractError('Row constraints cannot be combined with response correction')
        if row_constraint == 'preview' and model.config.lookahead < 5:
            raise ContractError('Preview row constraint requires at least five lookahead heads')
        if model.config.minimum_action_gap_ms and (correct_short_attacks or row_constraint != 'none'):
            raise ContractError('Joint action spacing cannot combine with a separate row correction or constraint')
        if model.config.minimum_action_gap_ms and head_times is not None:
            check_head_capacity(head_times, model.config.minimum_action_gap_ms)
        self.model = model
        self.controls = controls
        self.device, self.dtype = next(model.parameters()).device, next(model.parameters()).dtype
        _synchronize(self.device)
        self.started = time.perf_counter()
        encoded, self.arrangement = model.encode_generation(
            torch.as_tensor(np.array(mel, copy=True), dtype=self.dtype, device=self.device)[None],
            seed=seed ^ 0x61F9, code=arrangement_profile)
        profile_index = self.arrangement.get('arrangement_profile')
        self.encoded = model.condition_audio(encoded, profile_index)
        self.downstream_encoded = (self.encoded if model.config.profile_head_rate_downstream else
                                   model.condition_audio(encoded, profile_index, downstream=True))
        _synchronize(self.device)
        self.audio_seconds = time.perf_counter() - self.started
        self.duration_ms, self.chunk_ms = duration_ms, chunk_ms
        self.max_rows, self.max_seconds, self.stop_callback = max_rows, max_seconds, stop_callback
        self.correct_short_attacks = correct_short_attacks
        self.row_constraint, self.constraint_decisions = row_constraint, []
        self.rows, self.response_decisions = [], []
        self.cursor, self.residual = -1, None
        self.replay = ExactReplayState()
        self.row_cache, self.skeleton_cache = model.temporal.empty_cache(), model.skeleton_temporal.empty_cache()
        self.count_cache = (model.row_counts.temporal.empty_cache()
                            if model.config.row_factorization == 'count_layout' else None)
        self.release_rng = torch.Generator(device='cpu').manual_seed(seed ^ 0x4E51)
        self.row_rng = torch.Generator(device='cpu').manual_seed(seed ^ 0xA301)
        self.correction_rng = torch.Generator(device='cpu').manual_seed(seed ^ 0x52C4) if correct_short_attacks else None
        options = {} if controls is None else dict(controls=controls)
        self.planner = (planner_factory(model, self.encoded, duration_ms, seed ^ 0x17AB, head_chunk_ms, self.guard, **options)
                        if head_times is None else FixedHeadPlan(head_times, duration_ms, self.guard))
        self.release_bins = self.deadline_events = self.terminal_events = self.conditioned_waits = 0

    def guard(self):
        if time.perf_counter() - self.started >= self.max_seconds:
            raise _BudgetStop('time_limit')
        if len(self.rows) >= self.max_rows:
            raise _BudgetStop('row_limit')
        if self.stop_callback is not None:
            reason = self.stop_callback()
            if reason:
                raise _BudgetStop(str(reason))

    def tensor(self, value, dtype_override=None):
        return torch.as_tensor(value, dtype=self.dtype if dtype_override is None else dtype_override,
                               device=self.device)

    def fork(self, *, retry_seed=None):
        result = copy.copy(self)
        result.rows, result.response_decisions = list(self.rows), list(self.response_decisions)
        result.constraint_decisions = list(self.constraint_decisions)
        result.release_rng, result.row_rng = _fork_rng(self.release_rng), _fork_rng(self.row_rng)
        result.correction_rng = _fork_rng(self.correction_rng)
        result.planner = copy.copy(self.planner)
        result.planner.queue, result.planner.generated = list(self.planner.queue), list(self.planner.generated)
        if hasattr(self.planner, 'points'):
            result.planner.points = list(self.planner.points)
        result.planner.guard = result.guard
        if hasattr(self.planner, 'rng'):
            result.planner.rng = _fork_rng(self.planner.rng)
        if retry_seed is not None:
            result.release_rng.manual_seed(retry_seed ^ 0x4E51)
            result.row_rng.manual_seed(retry_seed ^ 0xA301)
            if result.correction_rng is not None:
                result.correction_rng.manual_seed(retry_seed ^ 0x52C4)
            result.residual = None
        return result

    @torch.inference_mode()
    def row_options(self, now):
        return {}

    def prefer_rows(self, log_probs, legal, now):
        return log_probs

    def record_row(self, row):
        pass

    def control_at(self, times):
        return ({} if self.controls is None else dict(control=self.tensor(
            self.controls.at(times, encoding=self.model.control_encoding))))

    def release_window(self, preview, profile):
        return release_limits(LNProjection(self.replay.open_ln_start_ms, self.cursor),
                              preview, self.duration_ms, profile)

    @torch.inference_mode()
    def step(self, *, stop_at=None):
        if self.cursor >= self.duration_ms:
            raise ContractError('Cannot advance a completed continuation')
        self.guard()
        stop_at = self.duration_ms if stop_at is None else min(stop_at, self.duration_ms)
        observed_through = self.cursor
        self.planner.fill(self.model.config.lookahead + 1)
        next_h = self.planner.queue[0] if self.planner.queue else None
        if next_h is None and not self.planner.finished:
            raise ContractError('A missing head queue is not audio termination')
        event_time, head_role, forced_event = None, False, False
        previous = None if self.replay.last_row is None else self.replay.last_row.time_ms
        if not any(self.replay.occupancy):
            if next_h is None or next_h > stop_at:
                self.cursor = stop_at
            else:
                event_time, head_role = next_h, True
        else:
            gap = getattr(self.model, 'recovery', self.model.config.minimum_action_gap_ms)
            if gap:
                earliest, deadline = self.release_window(
                    self.planner.preview(self.cursor, self.model.config.lookahead), gap)
                conditioned = next_h is None or deadline < next_h
                if conditioned and earliest > deadline:
                    raise ContractError('Action spacing has no eligible release before its deadline')
            else:
                conditioned = self.model.config.condition_full_holds and all(self.replay.occupancy) and next_h is not None
                deadline = next_h-1 if conditioned else None
            end = (deadline if conditioned else
                   min(self.cursor + self.chunk_ms, stop_at, self.duration_ms if next_h is None else next_h - 1))
            if end > self.cursor:
                bins = torch.arange((self.cursor + 1) // 10, end // 10 + 1, device=self.device)
                anchors = bins * 10 + 9
                native = bins[:, None] * 10 + torch.arange(10, device=self.device)
                projection = LNProjection(self.replay.open_ln_start_ms, self.cursor)
                preview = self.planner.preview(self.cursor, self.model.config.lookahead)
                previews, states = [preview] * len(bins), [projection] * len(bins)
                clocks = release_clocks(states, [previous] * len(bins), anchors.cpu().numpy(), previews, self.duration_ms)
                audio = interpolate_audio(self.downstream_encoded, anchors[None])[0]
                history = self.model.skeleton_temporal.read(self.skeleton_cache)[None].expand(len(bins), -1, -1)
                logits = self.model.release_logits(audio, history, self.tensor(clocks),
                    **self.control_at(anchors.cpu().numpy())).flatten()
                valid, forced = release_masks(states, native.cpu().numpy(), previews, self.duration_ms,
                    minimum_action_gap_ms=gap, windows=[(earliest, deadline)]*len(bins) if gap else None)
                valid &= native.cpu().numpy() <= end
                forced &= valid
                if conditioned:
                    indices = torch.where(self.tensor(valid, torch.bool).flatten())[0]
                    logits = logits.index_copy(0, indices, conditioned_release_logits(logits[indices]))
                    self.conditioned_waits += 1
                valid &= native.cpu().numpy() <= stop_at
                forced &= valid
                event, self.residual = sample_hazards(logits, self.tensor(valid, torch.bool).flatten(),
                    self.tensor(forced, torch.bool).flatten(), self.release_rng, self.residual)
                self.release_bins += len(bins)
                if event is not None:
                    event_time = int(native.flatten()[event])
                    forced_event = bool(forced.reshape(-1)[event])
                else:
                    self.cursor = min(end, stop_at)
            elif next_h is not None and next_h <= stop_at:
                if all(self.replay.occupancy):
                    raise ContractError('Planned head reached an all-held state without a release clock')
                event_time, head_role = next_h, True
            else:
                self.cursor = stop_at
        published = None
        if event_time is not None:
            self.cursor = event_time
            preview = self.planner.preview(self.cursor, self.model.config.lookahead)
            legal = row_support([self.replay], [self.cursor], [head_role], [preview], self.duration_ms)
            context = preview_features([preview], [self.cursor], [head_role], self.duration_ms, self.model.config.lookahead)
            local, future = consequences([self.replay], [self.cursor], [preview], self.duration_ms)
            counts = {}
            if self.model.config.minimum_action_gap_ms:
                response = spaced_rows(self.replay, self.cursor, preview, self.duration_ms,
                                       getattr(self.model, 'recovery', self.model.config.minimum_action_gap_ms))
                if not (response & legal[0]).any():
                    decision = dict(mode='joint_action_spacing', time_ms=self.cursor,
                        minimum_action_gap_ms=self.model.config.minimum_action_gap_ms,
                        last_attacks=list(self.replay.last_lane_attack_ms),
                        last_releases=list(self.replay.last_lane_release_ms),
                        ln_starts=list(self.replay.open_ln_start_ms), future_heads=list(preview.times_ms),
                        preview_complete=preview.complete, observed_through_ms=observed_through)
                    self.constraint_decisions.append(decision)
                    self.cursor = observed_through
                    raise NoRowContinuation(decision)
                counts['response_allowed'] = self.tensor(response[None], torch.bool)
            if self.model.config.row_factorization == 'count_layout':
                counts.update(count_history=self.model.row_counts.temporal.read(self.count_cache)[None],
                    count_clock=self.tensor(count_state([self.replay.open_ln_start_ms], [previous], [self.cursor])))
            audio = interpolate_audio(self.downstream_encoded, torch.tensor([self.cursor], device=self.device))
            log_probs = self.model.planned_row_log_probs(audio, self.model.temporal.read(self.row_cache)[None],
                self.tensor(exact_features([self.replay], [self.cursor])), self.tensor(legal, torch.bool),
                self.tensor([self.replay.occupancy], torch.bool), self.tensor(context), self.tensor(local), self.tensor(future),
                **counts, **self.control_at([self.cursor]), **self.row_options(self.cursor))[0]
            proposal_log_probs = self.prefer_rows(log_probs.detach().cpu().double(), legal[0], self.cursor)
            decision = None
            if self.row_constraint != 'none':
                try:
                    proposal_log_probs, decision = condition_rows(proposal_log_probs, self.replay,
                        self.cursor, preview, self.row_constraint)
                except NoRowContinuation as error:
                    error.decision['observed_through_ms'] = observed_through
                    self.constraint_decisions.append(error.decision)
                    self.cursor = observed_through
                    raise
            index = int(torch.multinomial(proposal_log_probs.exp(), 1, generator=self.row_rng))
            if decision is not None:
                decision['selected_actions'] = list(ROW_ACTIONS[index])
                self.constraint_decisions.append(decision)
            if self.correct_short_attacks:
                costs = short_attack_costs(self.replay, self.cursor, preview)
                minimum = float(costs[legal[0]].min())
                if not np.isfinite(minimum):
                    raise ContractError('Legal row has no relaxed head continuation')
                selected = select_response_row(proposal_log_probs, index, legal[0], costs, self.correction_rng)
                if selected != index or minimum > 0:
                    self.response_decisions.append(dict(time_ms=self.cursor, proposal=list(ROW_ACTIONS[index]),
                        selected=list(ROW_ACTIONS[selected]), proposal_cost=float(costs[index]),
                        selected_cost=minimum, corrected=selected != index,
                        previous_attacks=list(self.replay.last_lane_attack_ms),
                        ln_starts=list(self.replay.open_ln_start_ms), future_heads=list(preview.times_ms),
                        preview_complete=preview.complete))
                index = selected
            published = CompleteRow(self.cursor, ROW_ACTIONS[index])
            self.replay = commit(self.replay, published, is_terminal=self.cursor == self.duration_ms)
            self.row_cache = self.model.temporal.append(self.row_cache,
                self.tensor(content_features([published], [previous], [[None] * 4])[0]))
            self.skeleton_cache = self.model.skeleton_temporal.append(self.skeleton_cache,
                self.tensor(skeleton_tokens([self.cursor], [previous], [head_role])[0]))
            if self.model.config.row_factorization == 'count_layout':
                self.count_cache = self.model.row_counts.temporal.append(self.count_cache,
                    self.tensor(count_tokens([self.cursor], [previous], [published.actions])[0]))
            self.rows.append(published)
            self.record_row(published)
            if head_role:
                if not self.planner.queue or self.planner.pop() != self.cursor:
                    raise ContractError('Committed head differs from the immutable head plan')
            if forced_event:
                self.terminal_events += self.cursor == self.duration_ms
                self.deadline_events += self.cursor != self.duration_ms
            self.residual = None
        if self.cursor == self.duration_ms and any(self.replay.occupancy):
            raise ContractError('Completed planned generation retains an LN')
        return GenerationUpdate(published, self.cursor, self.cursor == self.duration_ms)


class PublicationLog:
    """Measure delivered rows and settled coverage, excluding rejected futures."""
    def __init__(self, started, duration_ms, callback):
        self.started, self.duration_ms, self.callback = started, duration_ms, callback
        self.coverage, self.latencies = [], []
        self.rows = self.heads = 0
        self.startup = self.first30_rows = self.first30_heads = None
        self.first30_rows_clock = self.first30_heads_clock = None

    def publish(self, update, service_seconds):
        if update.row is not None:
            self.rows += 1
            self.heads += sum(a in (1, 2) for a in update.row.actions)
        elapsed = time.perf_counter() - self.started
        self.latencies.append(service_seconds)
        self.coverage.append(dict(coverage_ms=update.coverage_ms, elapsed_seconds=elapsed))
        if self.startup is None and update.coverage_ms >= min(8000, self.duration_ms):
            self.startup = elapsed
        if self.first30_rows is None and self.rows >= 30:
            self.first30_rows, self.first30_rows_clock = elapsed, update.coverage_ms
        if self.first30_heads is None and self.heads >= 30:
            self.first30_heads, self.first30_heads_clock = elapsed, update.coverage_ms
        if self.callback is not None:
            self.callback(update)

    def result(self, session, reason, **extra):
        return NativeGeneration(tuple(session.rows), session.cursor == session.duration_ms, reason, session.cursor, dict(
            audio_encode_seconds=session.audio_seconds, generation_seconds=time.perf_counter()-self.started,
            rows=len(session.rows), heads=session.replay.note_count, open_lanes=list(session.replay.occupancy),
            first30_rows_seconds=self.first30_rows, first30_rows_through_ms=self.first30_rows_clock,
            first30_heads_seconds=self.first30_heads, first30_heads_through_ms=self.first30_heads_clock,
            startup_seconds=self.startup, startup_target_ms=min(8000, self.duration_ms),
            step_p50_ms=float(np.quantile(self.latencies, .5)*1000) if self.latencies else None,
            step_p99_ms=float(np.quantile(self.latencies, .99)*1000) if self.latencies else None,
            scheduler_steps=len(self.latencies), head_bins_scored=session.planner.bins,
            release_bins_scored=session.release_bins, forced_deadline_releases=session.deadline_events,
            forced_terminal_releases=session.terminal_events, conditioned_release_waits=session.conditioned_waits,
            correct_short_attacks=session.correct_short_attacks, short_attack_pairs=short_attack_pairs(session.rows),
            response_decisions=session.response_decisions, **session.arrangement,
            row_constraint=session.row_constraint, constraint_decisions=session.constraint_decisions,
            planned_heads=len(session.planner.generated), coverage=self.coverage,
            minimum_action_gap_ms=session.model.config.minimum_action_gap_ms,
            latency_scope='cached canonical Mel through published coverage; excludes waveform decode and Mel computation',
            **extra))
