"""Frozen native cases, separate TRAIN/VAL products and replayable diagnostics."""
from dataclasses import replace
import json
from pathlib import Path

from ..bounded_typed_continuation import generate_run
from ..bounded_typed_continuation.condition import GenerationCondition
from ..oracle_time_continuation.quality import source_metrics
from ..oracle_time_continuation.schema import CompleteRow
from ..oracle_time_continuation.storage import file_digest
from ..scoped_style_modeling.dataset import ContractError
from .control import publish_json, read_json


def validate_cases(value):
    if value.get('format') != 'vacation/native-cases-v1' or not value.get('cases'):
        raise ContractError('Stress inputs require nonempty vacation/native-cases-v1 cases')
    ids, groups, sources = set(), {}, {}
    for case in value['cases']:
        name = case['id']
        if (not isinstance(name, str) or not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in name) or
                name in ids or case['split'] not in ('train', 'validation') or not case['group_id']):
            raise ContractError('Native cases need unique safe IDs and explicit TRAIN/VAL group ownership')
        ids.add(name)
        if groups.setdefault(case['group_id'], case['split']) != case['split']:
            raise ContractError('A native song group cannot cross TRAIN and VAL')
        condition = GenerationCondition.from_payload(read_json(case['condition_file'], case['condition_sha256']))
        if type(case['seed']) is not int or not 0 <= case['seed'] < 2**63:
            raise ContractError('Native case seed must be a nonnegative integer below 2**63')
        if (not case.get('source_sha256') or len(case['source_sha256']) != 64 or
                any(c not in '0123456789abcdef' for c in case['source_sha256'])):
            raise ContractError('Native cases require source provenance')
        allocation = (case['group_id'], case['split'])
        if sources.setdefault(case['source_sha256'], allocation) != allocation:
            raise ContractError('A native source cannot change its group or split across cases')
        if not condition.seed_rows:
            raise ContractError('Native conditions require an observed seed')
    return value


def diagnostics(directory):
    def rows():
        with (directory / 'rows.jsonl').open() as stream:
            for line in stream:
                value = json.loads(line)
                yield CompleteRow(value['time_ms'], tuple(value['actions']))
    metrics = source_metrics(rows())
    # Export complete context around the longest identical-row run. This is a
    # physical locator, with no automatic style or playability label.
    best = (0, 0, 0)
    start = 0
    previous = None
    for i, row in enumerate(rows()):
        if row.actions != previous:
            start = i
        if i - start + 1 > best[0]:
            best = (i - start + 1, start, i + 1)
        previous = row.actions
    context = [dict(event_id=i, time_ms=row.time_ms, actions=row.actions)
               for i, row in enumerate(rows()) if max(0, best[1] - 16) <= i < best[2] + 16]
    receipt = dict(organization=metrics, repeated_run_event_range=best[1:], context=context,
                   native_star_rating=None, native_star_reason='no_pinned_calculator_selected',
                   rows_sha256=file_digest(directory / 'rows.jsonl'))
    publish_json(directory / 'diagnostics.json', receipt)
    return receipt


def run_cases(cases, generation, output, state, save_state, control):
    state.setdefault('cases', {})
    for case in cases:
        item = state['cases'].setdefault(case['id'], dict(segments=[], status='pending'))
        if item['status'] in ('completed', 'failed'):
            continue
        if reason := control.boundary():
            return dict(status='paused', reason=reason)
        directory = output / case['split'] / case['id']
        directory.mkdir(parents=True, exist_ok=True)
        if item.get('active_segment'):
            result_path = Path(item['active_segment']) / 'result.json'
            if not result_path.exists():
                item.update(status='failed', error='Interrupted segment has no finalized result; inspect its durable checkpoint')
                save_state()
                continue
            result = json.loads(result_path.read_text())
            item['segments'].append(result)
            item.pop('active_segment')
        while not item['segments'] or item['segments'][-1]['status'] != 'completed':
            if reason := control.boundary():
                return dict(status='paused', reason=reason)
            parent = item['segments'][-1] if item['segments'] else None
            target = directory / f'segment-{len(item["segments"]):05d}'
            cfg = replace(generation, condition_file=case['condition_file'], condition_sha256=case['condition_sha256'],
                          seed=case['seed'], output_dir=str(target), stop_after_candidate=None,
                          presentation_source=case.get('presentation_source', ''),
                          presentation_sha256=case.get('presentation_sha256', ''),
                          resume_from=parent['checkpoint_path'] if parent else None,
                          resume_sha256=parent['checkpoint_sha256'] if parent else '')
            control.storage(2 * cfg.resources.checkpoint_max_bytes)
            item['active_segment'] = str(target)
            save_state()
            try:
                result = generate_run.run_generation(cfg, stop_requested=control.boundary)
            except Exception as error:
                from ..oracle_time_continuation.runtime import ResourceLimit
                if isinstance(error, ResourceLimit):
                    raise
                item.update(status='failed', error=f'{type(error).__name__}: {error}')
                item.pop('active_segment', None)
                save_state()
                break
            item['segments'].append(result)
            item.pop('active_segment')
            save_state()
            if result['status'] == 'paused' and result['pause_reason'] != 'time_limit':
                return dict(status='paused', reason=result['pause_reason'])
        if item['status'] != 'failed':
            final = Path(item['segments'][-1]['output_dir'])
            diagnostics(final)
            item.update(status='completed', split=case['split'], source_sha256=case['source_sha256'],
                        condition_sha256=case['condition_sha256'], checkpoint_sha256=generation.checkpoint_sha256,
                        diagnostics_file=str(final / 'diagnostics.json'),
                        diagnostics_sha256=file_digest(final / 'diagnostics.json'))
            save_state()
    failed = sum(item['status'] == 'failed' for item in state['cases'].values())
    return dict(status='completed_with_anomalies' if failed else 'completed',
                completed=sum(item['status'] == 'completed' for item in state['cases'].values()), failed=failed)
