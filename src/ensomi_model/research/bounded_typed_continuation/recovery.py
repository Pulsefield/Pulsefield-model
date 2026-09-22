"""TRAIN-only native-prefix preferences, separate from source action labels.

The pool owns heuristic negative families and their generation provenance.
Inference remains unconstrained beyond the original feasibility support.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from ..scoped_style_modeling.dataset import ContractError
from .condition import GenerationCondition, pinned_bytes
from .contract import Arm, ROW_ACTIONS, Schedule
from .data import PreparedBatch, batch_predictions
from .features import TimingView, content_features, query_features
from .generation import RawEvent, seed_events
from .support import row_supports
from .response import response_preference


def alternative_mask(state, core_mask, blocking_mask):
    """Allow a different attack group or release of a specified blocking hold."""
    if (state.arm != Arm.R1 or type(core_mask) is not int or not 0 < core_mask < 16 or
            type(blocking_mask) is not int or not 0 <= blocking_mask < 16 or core_mask & blocking_mask):
        raise ContractError('Recovery needs R1 and disjoint four-lane core/blocking masks')
    choices = np.asarray(ROW_ACTIONS)
    good = np.zeros(len(choices), dtype=np.bool_)
    if blocking_mask:
        lanes = [c for c in range(4) if blocking_mask & (1 << c)]
        if any(state.replay.open_ln_start_ms[c] is None for c in lanes):
            raise ContractError('Recovery blocking lanes must be occupied in the exact native state')
        good |= (choices[:, lanes] == 3).any(1)
    if state.timing.onsets[state.index]:
        lanes = [c for c in range(4) if core_mask & (1 << c)]
        good |= ~np.isin(choices[:, lanes], (1, 2)).all(1)
    return good


def complement_loss(log_probs, alternatives, family=None):
    """Negative log alternative mass, optionally conditional on a row family."""
    mask = torch.as_tensor(alternatives, dtype=torch.bool, device=log_probs.device)
    legal = torch.isfinite(log_probs)
    conditional = family is not None
    family = legal if family is None else torch.as_tensor(family, dtype=torch.bool, device=log_probs.device)
    if (log_probs.ndim != 2 or mask.shape != log_probs.shape or
            family.shape != log_probs.shape or bool((legal & mask & ~family).any()) or
            not bool((legal & family & mask).any(-1).all()) or not bool((legal & family & ~mask).any(-1).all())):
        raise ContractError('Recovery requires both alternative and negative legal actions at every query')
    numerator = log_probs.masked_fill(~mask, -torch.inf).logsumexp(-1)
    return ((log_probs.masked_fill(~family, -torch.inf).logsumexp(-1) - numerator).mean()
            if conditional else -numerator.mean())


@dataclass
class NativeQuery:
    state: Schedule
    history: tuple[RawEvent, ...]
    seed: tuple[RawEvent, ...]
    alternatives: np.ndarray
    family: np.ndarray | None = None

    def prepare(self, receptive_tokens, *, full_history):
        """Project true committed history; the current placeholder is never read."""
        if len(self.history) != self.state.replay.row_count or not self.seed:
            raise ContractError('Native query requires its complete physical prefix and original seed')
        def features(events):
            return content_features([e.row for e in events], [e.previous_time_ms for e in events],
                                    [e.new_end_times for e in events])
        first = max(0, len(self.history) - receptive_tokens)
        prefix = features(self.history[first:])
        # before() queries the preceding physical token, including at skipped R.
        raw = np.concatenate((prefix, np.zeros_like(prefix[:1])), axis=0)[None]
        seed = features(self.seed)[None]
        memory = features(self.history)[None] if full_history else None
        view = TimingView(self.state.timing)
        return PreparedBatch(raw=raw, valid=np.ones(raw.shape[:2], bool), truncated=np.array([first > 0]),
            query_features=query_features([self.state], view), batch_indices=np.array([0]),
            positions=np.array([len(prefix)]), states=[self.state], labels=[], endpoints=[], views=[view],
            source_onsets=0, physical_rows=0, prefix_rows=len(prefix), context_spans_ms=[],
            seed_raw=seed, seed_valid=np.ones(seed.shape[:2], bool), memory_raw=memory,
            memory_valid=None if memory is None else np.ones(memory.shape[:2], bool),
            memory_onsets=None if memory is None else np.array([[any(a in (1, 2) for a in e.row.actions) for e in self.history]]))


def trajectory_queries(value, expected_condition):
    """Replay the original candidate schedule, retaining empty R decisions."""
    condition = GenerationCondition.from_payload(value['condition'])
    if condition != expected_condition or condition.arm != Arm.R1:
        raise ContractError('Recovery condition differs from its pinned TRAIN timing and complete seed')
    state, seed = seed_events(Arm.R1, condition.timing, condition.seed_rows, condition.crossing)
    if len(value['actions']) != len(condition.timing.times_ms) - state.index:
        raise ContractError('Recovery trajectory must contain every post-seed candidate')
    markers = {q['index']: q for q in value['queries']}
    if len(markers) != len(value['queries']):
        raise ContractError('Recovery query candidate indices must be unique')
    result, history = [], list(seed)
    for actions in value['actions']:
        marker = markers.pop(state.index, None)
        if marker is not None:
            family = None
            if marker.get('kind') == 'response':
                if marker['action'] != actions:
                    raise ContractError('Response preference differs from its native sampled action')
                preference = response_preference(state, actions, marker['threshold_ms'])
                if preference is None:
                    raise ContractError('Response query has no strictly improved alternative')
                good, family, _ = preference
            elif marker.get('kind') in (None, 'core'):
                good = alternative_mask(state, marker['core_mask'], marker['blocking_mask'])
            else:
                raise ContractError('Unknown native recovery preference kind')
            legal = np.asarray(row_supports([state])[0], dtype=bool)
            if not (good & legal).any() or not (~good & legal).any():
                raise ContractError('Recovery query lacks a legal alternative or negative continuation')
            result.append(NativeQuery(state, tuple(history), seed, good, family))
        before = state
        state, row = state.advance(tuple(actions))
        if row is not None:
            history.append(RawEvent(row, before.replay.last_row.time_ms))
    if markers or not state.finished or any(state.replay.occupancy):
        raise ContractError('Recovery queries or generated trajectory are incomplete')
    return result


class RecoveryPool:
    """Pinned machine preferences from TRAIN, sampled independently of CE draws."""
    def __init__(self, path, digest, plan, cache):
        path = Path(path)
        manifest = json.loads(pinned_bytes(path, digest))
        if (manifest['format'] != 'bounded-typed/native-recovery-pool-v1' or
                manifest['status'] != 'ready' or not manifest['runs']):
            raise ContractError('Recovery requires a complete, ready native preference pool')
        groups, seen = {}, set()
        for run in manifest['runs']:
            sha = run['source_sha256']
            if sha in seen or sha not in plan['sources'] or plan['sources'][sha]['identity']['split'] != 'train':
                raise ContractError('Recovery sources must be distinct pinned TRAIN identities')
            seen.add(sha)
            filename = Path(run['path'])
            if filename.name != run['path']:
                raise ContractError('Recovery trajectories must be direct children of their pool')
            value = json.loads(pinned_bytes(path.parent / filename, run['sha256']))
            pin = plan['sources'][sha]
            if (value['identity'] != pin['identity'] or value['source_metadata_sha256'] != pin['metadata_sha256'] or
                    value['source_rows_sha256'] != pin['rows_sha256'] or
                    any(value[k] != manifest[k] for k in ('policy_checkpoint_sha256', 'policy_source_revision', 'seed'))):
                raise ContractError('Recovery generation provenance differs from its source or policy pins')
            chart = cache.interval(dict(source_sha256=sha, first_onset=0, onset_count=1)).chart
            expected = GenerationCondition(Arm.R1, chart.typed.timing,
                tuple(chart.row(i) for i in range(chart.seed_rows)), chart.state(Arm.R1, chart.seed_rows).known_ends)
            queries = trajectory_queries(value, expected)
            if len(queries) != run['queries']:
                raise ContractError('Recovery query count differs from its pinned manifest')
            if queries:
                groups.setdefault(chart.identity.group_id, []).extend(queries)
        self.groups = [groups[key] for key in sorted(groups)]
        if not self.groups or sum(map(len, self.groups)) != manifest['queries']:
            raise ContractError('Recovery pool is empty or its query accounting differs')

    def select(self, update, count, seed):
        # Update-addressed randomness is deterministic across checkpoint recovery.
        rng = np.random.default_rng(np.random.SeedSequence([seed, update]))
        for _ in range(count):
            group = self.groups[int(rng.integers(len(self.groups)))]
            yield group[int(rng.integers(len(group)))]

    def backward(self, model, *, update, count, seed, weight, check):
        total = 0.
        for query in self.select(update, count, seed):
            batch = query.prepare(model.temporal.config.receptive_tokens, full_history=model.long_memory is not None)
            _, log_probs = batch_predictions(model, batch)
            loss = complement_loss(log_probs, query.alternatives[None],
                None if query.family is None else query.family[None])
            if not bool(torch.isfinite(loss)):
                raise ContractError('Recovery objective became nonfinite')
            (loss * (weight / count)).backward()
            total += float(loss.detach().cpu())
            check('native-recovery-backward')
        return dict(recovery_loss_sum=total, recovery_queries=count)
