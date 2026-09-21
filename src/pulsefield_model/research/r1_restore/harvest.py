"""Regenerate pinned TRAIN trajectories and the three historical preference rules."""
from collections import deque
import json
from pathlib import Path

import numpy as np

from ..bounded_typed_continuation.condition import GenerationCondition
from ..bounded_typed_continuation.contract import Arm, Schedule
from ..bounded_typed_continuation.corpus import ChartCache
from ..bounded_typed_continuation.generate_config import GenerateConfig
from ..bounded_typed_continuation.recovery import RecoveryPool, alternative_mask, trajectory_queries
from ..bounded_typed_continuation.response import response_preference
from ..bounded_typed_continuation.support import row_supports
from ..oracle_time_continuation.storage import file_digest
from ..scoped_style_modeling.dataset import ContractError
from ..vacation_training.control import publish_json
from ..vacation_training.stress import run_cases


def condition_for(chart):
    return GenerationCondition(Arm.R1, chart.typed.timing, tuple(chart.row(i) for i in range(chart.seed_rows)),
                               chart.state(Arm.R1, chart.seed_rows).known_ends)


def short_heads(state, action):
    return sum(a in (1, 2) and any(t is not None and state.time_ms - t < 30 for t in
               (state.replay.last_lane_attack_ms[c], state.replay.last_lane_release_ms[c]))
               for c, a in enumerate(action))


def core_queries(chart, condition, actions, kind):
    width, spacing = (32, 8) if kind == 'routing' else (12, 4)
    recent, markers = deque(maxlen=width), []
    state = Schedule.from_seed(Arm.R1, condition.timing, condition.seed_rows, condition.crossing)
    source_heads = np.isin(chart.rows['actions'], (1, 2))
    heads, last_marker = 0, -spacing
    for action in actions:
        if kind == 'routing' and recent and state.time_ms - recent[-1][0] >= 1000:
            recent.clear()
        if len(recent) == width and heads - last_marker >= spacing:
            indices = [r[1] for r in recent]
            source_max = int(source_heads[indices].sum(0).max())
            counts = [sum(bool(mask & (1 << c)) for _, _, mask in recent) for c in range(4)]
            core = sum(1 << c for c in range(4) if counts[c] >= 28) if kind == 'routing' else 0
            held = state.replay.open_ln_start_ms
            blocking = sum(1 << c for c, start in enumerate(held) if start is not None and
                           start <= recent[0][0] and not core & (1 << c))
            accept = bool(core) and source_max <= 24
            source_masks = {sum(1 << c for c, v in enumerate(row) if v) for row in source_heads[indices]}
            if kind == 'release':
                core = 15 ^ blocking
                accept = (sum(t is not None for t in held) == 3 and blocking.bit_count() == 3 and
                          source_max <= 9 and len(source_masks) >= 3 and all(r[2] == core for r in recent))
            if accept:
                good = alternative_mask(state, core, blocking)
                legal = np.asarray(row_supports([state])[0], bool)
                if (good & legal).any() and (~good & legal).any():
                    markers.append(dict(index=state.index, core_mask=core, blocking_mask=blocking,
                        window_start_ms=recent[0][0], source_max_lane_count=source_max,
                        source_distinct_masks=len(source_masks), head_number=heads))
                    last_marker = heads
        if condition.timing.onsets[state.index]:
            recent.append((state.time_ms, state.index, sum(1 << c for c, a in enumerate(action) if a in (1, 2))))
            heads += 1
        state, _ = state.advance(tuple(action))
    if not state.finished or any(state.replay.occupancy):
        raise ContractError('Native trajectory did not finish its exact schedule')
    return markers


def response_queries(chart, condition, actions):
    state = Schedule.from_seed(Arm.R1, condition.timing, condition.seed_rows, condition.crossing)
    first, bad = state.index, []
    for action in actions:
        if short_heads(state, action):
            bad.append(state.index)
        state, _ = state.advance(tuple(action))
    if not state.finished or any(state.replay.occupancy):
        raise ContractError('Response trajectory did not finish its exact schedule')
    source_short = np.zeros(len(condition.timing.times_ms), np.int64)
    source = Schedule.from_seed(Arm.R1, condition.timing, condition.seed_rows, condition.crossing)
    while not source.finished:
        action = chart.row(source.index).actions
        source_short[source.index] = short_heads(source, action)
        source, _ = source.advance(action)
    onsets, inspect = np.flatnonzero(condition.timing.onsets), set()
    for target in bad:
        at = int(np.searchsorted(onsets, target))
        inspect.update(range(max(first, int(onsets[max(0, at - 2)])), target + 1))
    state = Schedule.from_seed(Arm.R1, condition.timing, condition.seed_rows, condition.crossing)
    markers = []
    for action in actions:
        if state.index in inspect:
            preference = response_preference(state, action, 30.)
            future = onsets[onsets > state.index][:2]
            end = int(future[-1]) if len(future) else state.index
            if preference is not None and source_short[state.index:end + 1].sum() == 0:
                markers.append(dict(kind='response', index=state.index, action=list(action), threshold_ms=30.,
                    source_short_heads=0, window_end_index=end, **preference[2]))
        state, _ = state.advance(tuple(action))
    return markers


def preference_markers(chart, condition, actions, kind):
    if kind not in ('routing', 'release', 'response'):
        raise ContractError('Unknown native preference stage')
    if len(actions) != len(condition.timing.times_ms) - len(condition.seed_rows):
        raise ContractError('A native pool must preserve every candidate, including empty R decisions')
    markers = (response_queries(chart, condition, actions) if kind == 'response'
               else core_queries(chart, condition, actions, kind))
    if len(markers) > 32:
        markers = [markers[int(i)] for i in np.linspace(0, len(markers) - 1, 32, dtype=int)]
    value = dict(condition=json.loads(json.dumps(condition.payload())), actions=actions, queries=markers)
    if len(trajectory_queries(value, condition)) != len(markers):
        raise ContractError('Replayed native preferences differ from admission')
    return markers


def collect_pool(config, kind, parent, plan, selection, output, state, save, control):
    """Use durable ordinary generation; only completed charts enter a recovery pool."""
    output.mkdir(parents=True, exist_ok=True)
    pool = output / 'pool'
    pool.mkdir(exist_ok=True)
    state.setdefault('runs', [])
    state.setdefault('generation', {})
    if not selection:
        raise ContractError('Native selection cannot be empty')
    for index, saved in enumerate(state['runs']):
        if (index >= len(selection) or saved['source_sha256'] != selection[index]['source_sha256'] or
                file_digest(pool / saved['path']) != saved['sha256']):
            raise ContractError('Completed native pool payload changed')
    seed = 43 if kind == 'routing' else 17
    minimum_queries, minimum_groups = (64, 8) if kind == 'response' else (16, 4)
    freeze = dict(kind=kind, policy_checkpoint_sha256=parent['checkpoint_sha256'],
        policy_source_revision=parent['source_revision'], seed=seed,
        selection=[r['source_sha256'] for r in selection], plan_sha256=parent['plan_sha256'])
    freeze_path = output / 'freeze.json'
    if freeze_path.exists() and json.loads(freeze_path.read_text()) != freeze:
        raise ContractError('Native collector inputs changed after freezing')
    if not freeze_path.exists():
        publish_json(freeze_path, freeze)
    cache = ChartCache(config.training.source_cache_dir, plan['sources'], max_sources=1, max_bytes=64 * 1024**2)
    generation = GenerateConfig(checkpoint_file=parent['checkpoint_file'], checkpoint_sha256=parent['checkpoint_sha256'],
        device='cpu', cpu_threads=1, resources=config.training.resources,
        footprint_limit_bytes=config.training.footprint_limit_bytes, max_seconds=600.)
    # One source per selected group; changing the selection on resume is forbidden by the freeze.
    for index, row in enumerate(selection):
        if index < len(state['runs']):
            run = state['runs'][index]
            if run['source_sha256'] != row['source_sha256'] or file_digest(pool / run['path']) != run['sha256']:
                raise ContractError('Completed native pool payload changed')
        else:
            if reason := control.boundary():
                return dict(status='paused', reason=reason)
            sha = row['source_sha256']
            pin = plan['sources'][sha]
            if pin['identity']['split'] != 'train' or pin['identity']['group_id'] != row['group_id']:
                raise ContractError('Native collector may use only its pinned TRAIN groups')
            chart = cache.get(sha)
            condition = condition_for(chart)
            condition_path = output / f'{sha}-condition.json'
            condition_sha = publish_json(condition_path, condition.payload())
            case = dict(id=sha, source_sha256=sha, group_id=row['group_id'], split='train', seed=seed,
                        condition_file=str(condition_path), condition_sha256=condition_sha)
            result = run_cases([case], generation, output / 'generation', state['generation'], save, control)
            if result['status'] == 'paused':
                return result
            item = state['generation']['cases'][sha]
            if item['status'] != 'completed':
                raise ContractError('Native generation failed; retain evidence and inspect before another run')
            final_result = item['segments'][-1]
            if final_result['source_revision'] != parent['source_revision']:
                raise ContractError('Native collector implementation differs from its frozen parent source')
            final = Path(final_result['output_dir'])
            if file_digest(final / 'decisions.jsonl') != final_result['decisions_sha256']:
                raise ContractError('Native decisions changed after verified generation')
            actions = []
            for offset, line in enumerate((final / 'decisions.jsonl').read_text().splitlines()):
                decision = json.loads(line)
                if decision['candidate_index'] != len(condition.seed_rows) + offset:
                    raise ContractError('Native decision cursor is discontinuous')
                actions.append([0, 0, 0, 0] if decision['row'] is None else decision['row']['actions'])
            markers = preference_markers(chart, condition, actions, kind)
            value = dict(identity=pin['identity'], source_metadata_sha256=pin['metadata_sha256'],
                source_rows_sha256=pin['rows_sha256'], condition=condition.payload(), actions=actions, queries=markers,
                policy_checkpoint_sha256=parent['checkpoint_sha256'],
                policy_source_revision=parent['source_revision'], seed=seed)
            path = pool / f'{sha}.json'
            digest = publish_json(path, value)
            state['runs'].append(dict(source_sha256=sha, path=path.name, sha256=digest, queries=len(markers)))
            save()
            print(json.dumps(dict(phase=kind + '-harvest', charts=len(state['runs']),
                                  queries=sum(r['queries'] for r in state['runs']))), flush=True)
        queries = sum(r['queries'] for r in state['runs'])
        groups = sum(bool(r['queries']) for r in state['runs'])
        if kind == 'release' and queries >= minimum_queries and groups >= minimum_groups:
            break
    ready = queries >= minimum_queries and groups >= minimum_groups
    manifest = dict(format='bounded-typed/native-recovery-pool-v1',
        status='ready' if ready else 'insufficient-native-failures',
        provenance='Reconstructed TRAIN-only machine preferences; not human labels.',
        policy_checkpoint_sha256=parent['checkpoint_sha256'], policy_source_revision=parent['source_revision'],
        seed=seed, plan_sha256=parent['plan_sha256'], runs=state['runs'], queries=queries,
        selection_sha256=file_digest(freeze_path), groups=groups)
    path = pool / 'manifest.json'
    digest = publish_json(path, manifest)
    if ready:
        RecoveryPool(path, digest, plan, cache)
    return dict(status=manifest['status'], path=str(path), sha256=digest, queries=queries, groups=groups)
