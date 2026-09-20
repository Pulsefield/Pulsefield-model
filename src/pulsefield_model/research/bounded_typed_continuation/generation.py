"""Native sampling with exact commitments and raw-state recovery.

Only the supplied timing and seed enter initialization. Durable state contains
exact gameplay facts, a finite raw physical history and CPU RNG state. Optional
persistent conditioning also retains the original raw seed. Landmark memory
retains the complete raw prefix. Learned values are rebuilt under verified
parameters instead of serialized.
"""
from __future__ import annotations

from collections import deque
from bisect import bisect_left
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Mapping, Sequence

import torch

from ..oracle_time_continuation.replay import ExactReplayState
from ..oracle_time_continuation.schema import CompleteRow, checked_time
from ..scoped_style_modeling.dataset import ContractError
from .contract import Arm, HEAD_ACTIONS, ROW_ACTIONS, Schedule, Timing
from .features import TimingView, content_features, query_features
from .support import row_supports


@dataclass(frozen=True)
class RawEvent:
    row: CompleteRow
    previous_time_ms: float | None
    new_end_times: tuple[float | None, ...] = (None, None, None, None)

    def __post_init__(self):
        if self.previous_time_ms is not None and checked_time(self.previous_time_ms) >= self.row.time_ms:
            raise ContractError('Raw event predecessor must strictly precede the physical row')
        if not isinstance(self.new_end_times, tuple) or len(self.new_end_times) != 4:
            raise ContractError('Raw event needs four immutable optional new endpoint plans')
        for action, end in zip(self.row.actions, self.new_end_times):
            if end is not None and (action != 2 or checked_time(end) <= self.row.time_ms):
                raise ContractError('Raw future plan belongs to a newly started positive-duration LN')

    def features(self):
        return content_features([self.row], [self.previous_time_ms], [self.new_end_times])[0]


@dataclass(frozen=True)
class GeneratedStep:
    candidate_index: int
    row: CompleteRow | None
    endpoints: dict[int, int]
    row_or_head_log_prob: float
    endpoint_log_prob: float | None


def model_signature(model):
    return tuple((id(p), p._version, str(p.device), p.dtype) for p in model.parameters())


def model_digest(model):
    """Device-independent identity of configuration and exact parameter bytes."""
    config = asdict(model.config)
    # Preserve the identity of checkpoints predating optional seed conditioning.
    if config['seed_context'] == 'none':
        config.pop('seed_context')
    if config['long_memory'] == 'none':
        for field in ('long_memory', 'memory_hidden', 'memory_stride'):
            config.pop(field)
    if config['head_routing'] == 'none':
        for field in ('head_routing', 'routing_hidden'):
            config.pop(field)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for name, tensor in model.state_dict().items():
        digest.update(json.dumps((name, str(tensor.dtype), tuple(tensor.shape))).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def timing_digest(timing):
    return hashlib.sha256(json.dumps(asdict(timing), sort_keys=True, allow_nan=False).encode()).hexdigest()


def seed_events(arm, timing, rows, crossing_ends):
    """Project complete supplied objects into raw events with permitted plans."""
    state = Schedule.from_seed(arm, timing, rows, crossing_ends)
    plans, pending = [{} for _ in rows], {}
    if arm != Arm.R0:
        for i, row in enumerate(rows):
            for lane, action in enumerate(row.actions):
                if action == 2:
                    pending[lane] = i
                elif action == 3:
                    plans[pending.pop(lane)][lane] = i
        for lane, start in pending.items():
            plans[start][lane] = state.known_ends[lane]
    events = tuple(RawEvent(row, None if i == 0 else rows[i - 1].time_ms,
                            tuple(timing.times_ms[plans[i][c]] if c in plans[i] else None for c in range(4)))
                   for i, row in enumerate(rows))
    return state, events


class Rollout:
    """A single-chart predictor; construction owns no source suffix or labels.

    Raw history must contain exactly the most recent receptive_tokens physical
    rows, or all rows when younger. Its oldest event retains the true preceding
    timestamp. Deterministic candidates consume no random draws. Temperature is
    one and no additional repetition, duration or source-style policy is applied.
    """
    def __init__(self, model, state: Schedule, history: Sequence[RawEvent], *, seed_rows: int,
                 seed_history: Sequence[RawEvent] | None = None,
                 memory_history: Sequence[RawEvent] | None = None):
        if state.arm != model.config.arm:
            raise ContractError('Rollout state and model must use the same arm')
        if type(seed_rows) is not int or not 0 <= seed_rows <= state.replay.row_count:
            raise ContractError('Rollout seed count must belong to the committed physical prefix')
        capacity = model.temporal.config.receptive_tokens
        if len(history) != min(capacity, state.replay.row_count):
            raise ContractError('Recovery needs the complete bounded raw physical history')
        if history:
            if history[-1].row != state.replay.last_row:
                raise ContractError('Raw history must end at the exact committed last row')
            if any(a.row.time_ms != b.previous_time_ms for a, b in zip(history, history[1:])):
                raise ContractError('Raw history must retain consecutive physical predecessors')
            if state.replay.row_count == len(history):
                if history[0].previous_time_ms is not None or history[0].row.time_ms != state.replay.first_time_ms:
                    raise ContractError('A complete raw prefix begins at the true BOS')
            elif history[0].previous_time_ms is None:
                raise ContractError('Truncated raw history must retain its preceding physical timestamp')
            first_row = state.replay.row_count - len(history)
            for i, event in enumerate(history, first_row):
                if (state.arm == Arm.R0 or (state.arm == Arm.R1 and i >= seed_rows)) and any(end is not None for end in event.new_end_times):
                    raise ContractError('Raw history exposes a future endpoint unavailable to this arm')
                if state.arm == Arm.O1 or (state.arm == Arm.R1 and i < seed_rows):
                    if any(action == 2 and end is None for action, end in zip(event.row.actions, event.new_end_times)):
                        raise ContractError('Known historical objects must retain their originally committed endpoints')
        self.model, self.state = model, state
        self.seed_rows = seed_rows
        self.history = deque(history, maxlen=capacity)
        self.seed_history, self.seed_context = (), None
        if model.config.seed_context == 'observed':
            if seed_history is None or len(seed_history) != seed_rows or not seed_rows:
                raise ContractError('Observed recovery requires the complete original raw seed')
            seed_history = tuple(seed_history)
            crossing = {lane: bisect_left(state.timing.times_ms, end)
                        for event in seed_history for lane, end in enumerate(event.new_end_times)
                        if end is not None and end > seed_history[-1].row.time_ms}
            seeded, expected = seed_events(state.arm, state.timing, [event.row for event in seed_history], crossing)
            if (seed_history != expected or seeded.index > state.index or
                    seeded.replay.first_time_ms != state.replay.first_time_ms or
                    seeded.replay.note_count > state.replay.note_count):
                raise ContractError('Persistent raw seed differs from its supplied objects or exact prefix')
            first_row = state.replay.row_count - len(history)
            if any(event != seed_history[i] for i, event in enumerate(history, first_row) if i < seed_rows):
                raise ContractError('Persistent seed and rolling history disagree in their overlapping rows')
            for lane, end in enumerate(seeded.known_ends):
                if end is not None and end >= state.index and (
                        end != state.known_ends[lane] or
                        seeded.replay.open_ln_start_ms[lane] != state.replay.open_ln_start_ms[lane]):
                    raise ContractError('An unexpired original seed commitment is missing from exact state')
            self.seed_history = seed_history
            raw = model.temporal.input.weight.new_tensor(content_features(
                [event.row for event in seed_history], [event.previous_time_ms for event in seed_history],
                [event.new_end_times for event in seed_history]))[None]
            with torch.no_grad():
                self.seed_context = model.encode_seed(raw, torch.ones(raw.shape[:2], dtype=torch.bool, device=raw.device))[0]
        elif seed_history is not None:
            raise ContractError('Only observed seed conditioning retains a persistent raw seed')
        self.memory_history, self.memory_cache = [], None
        if model.long_memory is not None:
            if memory_history is None or len(memory_history) != state.replay.row_count or not seed_rows:
                raise ContractError('Landmark recovery requires the complete raw physical history')
            memory_history = tuple(memory_history)
            if tuple(history) != memory_history[-len(history):]:
                raise ContractError('Long memory and local history disagree')
            original = memory_history[:seed_rows]
            crossing = {lane: bisect_left(state.timing.times_ms, end)
                        for event in original for lane, end in enumerate(event.new_end_times)
                        if end is not None and end > original[-1].row.time_ms}
            replayed, expected = seed_events(state.arm, state.timing, [e.row for e in original], crossing)
            if original != expected or self.seed_history and original != self.seed_history:
                raise ContractError('Long memory differs from the complete original seed')
            pending = iter(memory_history[seed_rows:])
            event = next(pending, None)
            while replayed.index < state.index:
                current = event if event is not None and event.row.time_ms == replayed.time_ms else None
                if current is not None:
                    if current.previous_time_ms != replayed.replay.last_row.time_ms or any(e is not None for e in current.new_end_times):
                        raise ContractError('Long memory has an invalid predecessor or unavailable suffix plan')
                    event = next(pending, None)
                replayed, emitted = replayed.advance(None if current is None else current.row.actions)
                if emitted != (None if current is None else current.row):
                    raise ContractError('Long memory differs from the permitted committed rows')
            if event is not None or replayed != state:
                raise ContractError('Complete long memory and exact state disagree')
            self.memory_history = list(memory_history)
            self.memory_cache = model.long_memory.empty_cache()
            for event in memory_history:
                self.memory_cache = model.long_memory.append(self.memory_cache,
                    model.temporal.input.weight.new_tensor(event.features()), any(a in (1, 2) for a in event.row.actions))
        elif memory_history is not None:
            raise ContractError('A model without long memory cannot retain a full raw memory history')
        self.view = TimingView(state.timing)
        self.signature = model_signature(model)
        self.parameter_digest = model_digest(model)
        self.cache = model.temporal.empty_cache(truncated_start=state.replay.row_count > len(history))
        for event in history:
            self.cache = model.temporal.append(self.cache, model.temporal.input.weight.new_tensor(event.features()))

    @classmethod
    def from_seed(cls, model, timing: Timing, rows: Sequence[CompleteRow], crossing_ends: Mapping[int, int] | None = None):
        state, events = seed_events(model.config.arm, timing, rows, crossing_ends)
        first = max(0, len(rows) - model.temporal.config.receptive_tokens)
        return cls(model, state, events[first:], seed_rows=len(rows),
                   seed_history=events if model.config.seed_context == 'observed' else None,
                   memory_history=events if model.long_memory is not None else None)

    def _check(self):
        if model_signature(self.model) != self.signature:
            raise ContractError('Rollout parameters changed; start or restore under an explicit parameter identity')

    @torch.no_grad()
    def prediction(self):
        """Read the current stochastic decision without changing history or RNG."""
        self._check()
        if self.state.finished:
            raise ContractError('Completed rollout has no prediction query')
        like = self.model.temporal.input.weight
        history = self.model.temporal.read(self.cache)[None]
        seed = None if self.seed_context is None else self.seed_context[None]
        memory = None if self.memory_cache is None else self.model.long_memory.cached_query(self.memory_cache)
        hands = self.model.readout(history, like.new_tensor(query_features([self.state], self.view)), seed, memory)
        return hands[0], self.model.decision_log_probs(hands, [self.state])[0]

    @torch.no_grad()
    def step(self, generator: torch.Generator, *, candidate_budget=8192, score_endpoints=True):
        self._check()
        if self.state.finished:
            raise StopIteration('Candidate schedule is complete')
        if generator.device.type != 'cpu':
            raise ContractError('Rollout draws require a CPU generator for durable device-independent RNG state')
        before = self.state
        ends, selected_logp, endpoint_logp = {}, 0., 0.
        if before.arm == Arm.O1 and not before.timing.onsets[before.index]:
            actions = before.forced_row()
        else:
            choices = HEAD_ACTIONS if before.arm == Arm.O1 else ROW_ACTIONS
            support = before.head_support() if before.arm == Arm.O1 else row_supports([before])[0]
            legal = [i for i, allowed in enumerate(support) if allowed]
            if not legal:
                raise ContractError('Current candidate has no feasible continuation')
            if len(legal) == 1:
                chosen, hands = legal[0], None
            else:
                hands, logp = self.prediction()
                chosen = int(torch.multinomial(logp.cpu().double().exp(), 1, generator=generator))
                selected_logp = float(logp[chosen].cpu())
            actions = choices[chosen]
            if before.arm == Arm.O1:
                heads = actions
                if 2 in heads:
                    if hands is None:
                        hands, _ = self.prediction()
                    ends = self.model.sample_endpoints(hands, before, heads, self.view, generator=generator,
                                                       candidate_budget=candidate_budget)
                    if score_endpoints:
                        endpoint_logp = float(self.model.endpoint_log_probs(hands[None], [before], [heads], [ends],
                                               [self.view], candidate_budget=candidate_budget, recompute=False)[0].cpu())
                    else:
                        endpoint_logp = None
                actions = tuple(3 if end == before.index else head for end, head in zip(before.known_ends, heads))
        after, row = before.advance(actions, ends)
        cache, memory_cache, event = self.cache, self.memory_cache, None
        if row is not None:
            previous = None if before.replay.last_row is None else before.replay.last_row.time_ms
            new_plans = tuple(before.timing.times_ms[end] if action == 2 and end is not None else None
                              for action, end in zip(row.actions, after.known_ends))
            event = RawEvent(row, previous, new_plans)
            cache = self.model.temporal.append(cache, self.model.temporal.input.weight.new_tensor(event.features()))
            if memory_cache is not None:
                memory_cache = self.model.long_memory.append(memory_cache,
                    self.model.temporal.input.weight.new_tensor(event.features()), any(a in (1, 2) for a in row.actions))
        # Exact and learned state advance together only after the full decision
        # and content update succeeded. Callers should checkpoint RNG/state before
        # a resource boundary if they need to retry an interrupted step.
        self.state, self.cache, self.memory_cache = after, cache, memory_cache
        if event is not None:
            self.history.append(event)
            if memory_cache is not None:
                self.memory_history.append(event)
        return GeneratedStep(before.index, row, ends, selected_logp, endpoint_logp)

    def snapshot(self, generator):
        """Return owned raw facts/RNG for a safe torch weights-only checkpoint."""
        self._check()
        if generator.device.type != 'cpu':
            raise ContractError('Durable rollout RNG must use the CPU generator')
        result = dict(format='bounded-typed/rollout-v1', model_config=json.loads(json.dumps(asdict(self.model.config))),
                    parameter_sha256=self.parameter_digest, timing_sha256=timing_digest(self.state.timing),
                    seed_rows=self.seed_rows, index=self.state.index, replay=asdict(self.state.replay),
                    known_ends=list(self.state.known_ends), history=[asdict(event) for event in self.history],
                    rng=generator.get_state().clone())
        if self.model.config.seed_context == 'observed':
            result['seed_history'] = [asdict(event) for event in self.seed_history]
        if self.memory_cache is not None:
            result['memory_history'] = [asdict(event) for event in self.memory_history]
        return result

    @classmethod
    def restore(cls, model, timing, snapshot):
        config = dict(snapshot['model_config'])
        config.setdefault('seed_context', 'none')
        config.setdefault('long_memory', 'none')
        config.setdefault('memory_hidden', 256)
        config.setdefault('memory_stride', 64)
        config.setdefault('head_routing', 'none')
        config.setdefault('routing_hidden', 512)
        if (snapshot['format'] != 'bounded-typed/rollout-v1' or config != asdict(model.config) or
                snapshot['parameter_sha256'] != model_digest(model) or snapshot['timing_sha256'] != timing_digest(timing)):
            raise ContractError('Rollout checkpoint differs from its model parameters, configuration or timing condition')
        replay_values = dict(snapshot['replay'])
        if replay_values['last_row'] is not None:
            replay_values['last_row'] = CompleteRow(**replay_values['last_row'])
        for key in ('open_ln_start_ms', 'last_lane_attack_ms', 'last_lane_release_ms'):
            replay_values[key] = tuple(replay_values[key])
        state = Schedule(model.config.arm, timing, snapshot['index'], ExactReplayState(**replay_values),
                         tuple(snapshot['known_ends']))
        history = [RawEvent(CompleteRow(**value['row']), value['previous_time_ms'], tuple(value['new_end_times']))
                   for value in snapshot['history']]
        seed_history = ([RawEvent(CompleteRow(**value['row']), value['previous_time_ms'], tuple(value['new_end_times']))
                         for value in snapshot['seed_history']] if 'seed_history' in snapshot else None)
        memory_history = ([RawEvent(CompleteRow(**v['row']), v['previous_time_ms'], tuple(v['new_end_times']))
                           for v in snapshot['memory_history']] if 'memory_history' in snapshot else None)
        result = cls(model, state, history, seed_rows=snapshot['seed_rows'], seed_history=seed_history,
                     memory_history=memory_history)
        generator = torch.Generator().set_state(snapshot['rng'].clone())
        return result, generator
