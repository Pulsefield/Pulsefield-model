import copy
from dataclasses import replace
from types import SimpleNamespace
import time

import numpy as np
import pytest
import torch

from ensomi_model.research.bounded_typed_continuation.contract import ROW_ACTIONS
from ensomi_model.research.joint_audio_continuation.generation import GenerationUpdate
from ensomi_model.research.oracle_time_continuation.replay import ExactReplayState, commit
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.planned_audio_continuation import buffering, generation
from ensomi_model.research.planned_audio_continuation.model import PlannedAudioModel
from ensomi_model.research.planned_audio_continuation.session import ContinuationSession
from .test_distribution import config


def test_session_forks_own_rngs_queues_and_preserve_shared_cache_values():
    torch.manual_seed(23)
    model = PlannedAudioModel(replace(config(), bounded_head=True, condition_full_holds=True))
    with torch.no_grad():
        model.head_base.weight.zero_()
        model.head_base.bias.fill_(-2.2)
    session = ContinuationSession(model, np.zeros((30, 128), np.float32), 300,
                                  seed=17, planner_factory=generation.HeadPlanner)
    session.step()
    original_rows = tuple(session.rows)
    original_queue = list(session.planner.queue)
    rng = session.row_rng.get_state().clone()
    buffers = [b.clone() for b in session.row_cache.buffers]
    first, second = session.fork(), session.fork()
    assert first.encoded is session.encoded
    for branch in (first, second):
        while branch.cursor < 300:
            branch.step()
    assert first.rows == second.rows
    assert len(first.rows) > len(original_rows) > 0
    assert tuple(session.rows) == original_rows and session.planner.queue == original_queue
    assert torch.equal(session.row_rng.get_state(), rng)
    for observed, expected in zip(session.row_cache.buffers, buffers):
        torch.testing.assert_close(observed, expected, atol=0, rtol=0)
    retry = session.fork(retry_seed=901)
    assert torch.equal(retry.planner.rng.get_state(), session.planner.rng.get_state())
    assert not torch.equal(retry.row_rng.get_state(), session.row_rng.get_state())
    assert retry.residual is None


def tap_model(monkeypatch):
    model = PlannedAudioModel(config())
    calls = []
    original = model.encode_generation

    def encode(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    def scores(audio, history, exact, legal, *rest):
        logits = torch.full((1, 256), -torch.inf)
        logits[0, ROW_ACTIONS.index((1, 0, 0, 0))] = 0.
        assert bool(legal[0, ROW_ACTIONS.index((1, 0, 0, 0))])
        return logits

    monkeypatch.setattr(model, 'encode_generation', encode)
    monkeypatch.setattr(model, 'planned_row_log_probs', scores)
    return model, calls


def test_healthy_buffering_is_row_identical_encodes_once_and_publishes_complete_rows(monkeypatch):
    model, encodes = tap_model(monkeypatch)
    mel = np.zeros((10, 128), np.float32)
    plain = generation.rollout(model, mel, 100, seed=17, head_times=(0, 40, 80))
    updates = []
    buffered = buffering.rollout_buffered(model, mel, 100, seed=17, head_times=(0, 40, 80),
                                         window_ms=35, on_update=updates.append)
    assert len(encodes) == 2
    assert plain.rows == buffered.rows and buffered.completed
    assert [u.row for u in updates if u.row is not None] == list(buffered.rows)
    assert updates[-1].coverage_ms == 100 and updates[-1].completed
    assert all(a.coverage_ms <= b.coverage_ms for a, b in zip(updates, updates[1:]))
    assert buffered.metrics['rejected_proposals'] == 0


def scripted_session(always_bad=False, held_prefix=False):
    """A causal two-branch event process for publication tests, with exact replay.

    Its future violation lies outside the first publication cut, so these tests
    fail if the sampler checks only the current window or leaks a rejected halo.
    Neural cache/RNG fork behavior is tested separately on the real session.
    """
    class Scripted:
        def __init__(self, model, mel, duration_ms, **kwargs):
            self.started, self.duration_ms = time.perf_counter(), duration_ms
            self.device, self.audio_seconds, self.arrangement = torch.device('cpu'), 0., {}
            self.rows, self.response_decisions = [], []
            self.cursor, self.replay, self.safe = -1, ExactReplayState(), False
            self.planner = SimpleNamespace(bins=0, generated=[0, 30, 31] if held_prefix else [0, 10])
            self.release_bins = self.deadline_events = self.terminal_events = self.conditioned_waits = 0
            self.correct_short_attacks = False

        def fork(self, retry_seed=None):
            result = copy.copy(self)
            result.rows = list(self.rows)
            result.safe = self.safe or (retry_seed is not None and not always_bad)
            return result

        def step(self):
            if held_prefix:
                trajectory = [(0, (2, 0, 0, 0)), (30, (0, 1, 0, 0)),
                              (31, (0, 1, 0, 0)), (80, (3, 0, 0, 0))]
            else:
                trajectory = [(0, (1, 0, 0, 0)),
                              (10, (0, 1, 0, 0) if self.safe else (1, 0, 0, 0))]
            if len(self.rows) < len(trajectory):
                t, actions = trajectory[len(self.rows)]
                row = CompleteRow(t, actions)
                self.replay = commit(self.replay, row)
                self.rows.append(row)
                self.cursor = t
            else:
                row = None
                self.cursor = self.duration_ms
            return GenerationUpdate(row, self.cursor, self.cursor == self.duration_ms)

    return Scripted


def test_halo_rejection_restores_boundary_and_never_publishes_rejected_future(monkeypatch):
    monkeypatch.setattr(buffering, 'ContinuationSession', scripted_session())
    events, rejected = [], []

    def reject(metadata, fork):
        assert events == []
        rejected.append((metadata, tuple(fork.rows)))

    result = buffering.rollout_buffered(None, None, 50, seed=17, window_ms=1,
                                       on_update=events.append, on_rejected=reject)
    assert result.completed and len(rejected) == 1
    assert rejected[0][0]['cut_ms'] == 0
    assert rejected[0][0]['pairs'][0]['time_ms'] == 10
    assert rejected[0][1][1].actions == (1, 0, 0, 0)
    assert result.rows[1].actions == (0, 1, 0, 0)
    assert [e.row for e in events if e.row is not None] == list(result.rows)
    assert all(e.row is None or e.row.actions != (3, 0, 0, 0) for e in events)


def test_attempt_exhaustion_keeps_only_published_prefix_with_open_ln(monkeypatch):
    monkeypatch.setattr(buffering, 'ContinuationSession', scripted_session(always_bad=True, held_prefix=True))
    events = []
    result = buffering.rollout_buffered(None, None, 100, seed=17, window_ms=1,
                                       max_attempts=2, on_update=events.append)
    assert not result.completed and result.stop_reason == 'planning_attempt_limit'
    assert result.coverage_ms == 0 and result.rows == (CompleteRow(0, (2, 0, 0, 0)),)
    assert result.metrics['open_lanes'] == [True, False, False, False]
    assert len(events) == 1 and not events[0].completed
    assert result.metrics['rejected_proposals'] == 2


def test_close_pair_entry_halo_and_exact_twenty_ms_have_separate_meanings():
    state = commit(ExactReplayState(), CompleteRow(0, (1, 2, 0, 0)))
    state = commit(state, CompleteRow(10, (0, 3, 0, 0)))
    rows = [CompleteRow(20, (1, 0, 0, 0)), CompleteRow(30, (0, 2, 0, 0))]
    assert not buffering.close_pairs(state, rows, 29, screen_release_heads=True)
    pairs = buffering.close_pairs(state, rows, 30, screen_release_heads=True)
    assert [(p['kind'], p['gap_ms']) for p in pairs] == [('RH', 20)]
    assert not buffering.close_pairs(state, rows, 30, screen_release_heads=False)


def test_budget_stop_discards_unpublished_rows_and_consumer_error_propagates(monkeypatch):
    model, _ = tap_model(monkeypatch)
    mel = np.zeros((10, 128), np.float32)
    events = []
    capped = buffering.rollout_buffered(model, mel, 100, seed=17, head_times=(0, 40, 80),
        stop_callback=lambda: 'external_resource_stop', on_update=events.append)
    assert capped.stop_reason == 'external_resource_stop' and not capped.completed
    assert not capped.rows and capped.coverage_ms == -1 and not events

    calls = []

    def stop_after_one_unpublished_row():
        calls.append(1)
        return 'mid_proposal_stop' if len(calls) >= 3 else None

    capped = buffering.rollout_buffered(model, mel, 100, seed=17, head_times=(0, 40, 80),
        stop_callback=stop_after_one_unpublished_row, on_update=events.append)
    assert not capped.rows and not events and capped.stop_reason == 'mid_proposal_stop'
    assert capped.metrics['evaluated_speculative_rows'] == capped.metrics['max_unpublished_rows'] == 1
    assert capped.metrics['windows'][0]['attempts'][0]['status'] == 'capped'
    assert capped.metrics['windows'][0]['service_seconds'] > 0

    def fail(update):
        events.append(update)
        raise RuntimeError('consumer failed')

    with pytest.raises(RuntimeError, match='consumer failed'):
        buffering.rollout_buffered(model, mel, 100, seed=17, head_times=(0, 40, 80), on_update=fail)
    assert len(events) == 1 and events[0].row == CompleteRow(0, (1, 0, 0, 0))
