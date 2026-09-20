"""Serial source, seed, memory and native-correction stages with durable recovery."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import platform
import time

import psutil
import torch

from ..bounded_typed_continuation import train_run
from ..bounded_typed_continuation.condition import GenerationCondition
from ..bounded_typed_continuation.contract import Arm
from ..bounded_typed_continuation.corpus import ChartCache, read_plan
from ..bounded_typed_continuation.generate_config import GenerateConfig
from ..bounded_typed_continuation.model import BoundedModel, ModelConfig
from ..bounded_typed_continuation.smoke_run import source_revision
from ..oracle_time_continuation.runtime import ResourceGuard, ResourceLimit
from ..oracle_time_continuation.storage import file_digest
from ..scoped_style_modeling.dataset import ContractError
from ..vacation_training.control import Control, publish_json, read_json, run_lock
from ..vacation_training.stress import run_cases
from ..vacation_training.teacher import evaluate, validate_evaluation
from .config import STAGES, stage_training
from .harvest import collect_pool, condition_for
from .preparation import prepare_census, prepare_plans


def preflight(config, root, resolved_yaml):
    started, last_check = time.monotonic(), -float('inf')
    guard = ResourceGuard('cpu', config.resources)
    def check():
        nonlocal last_check
        now = time.monotonic()
        if now - started > config.preflight_max_seconds:
            raise ResourceLimit('R1 input preflight exhausted its declared wall budget')
        if now - last_check >= 1:
            guard.check('restore-preflight')
            last_check = now
    revision = source_revision()
    settings = json.loads(json.dumps(asdict(config)))
    settings.pop('mode')
    path = root / 'freeze.json'
    old = json.loads(path.read_text()) if path.exists() else None
    environment = dict(python=platform.python_version(), torch=str(torch.__version__), platform=platform.platform())
    if old and (old['config'] != settings or old['source_revision'] != revision):
        raise ContractError('Restoration source or configuration changed after its freeze')
    if old and old['environment'] != environment:
        raise ContractError('Restoration runtime changed after its execution freeze')
    plan = read_plan(config.base_plan_file, config.base_plan_sha256)
    evaluation = validate_evaluation(read_json(config.evaluation_file, config.evaluation_sha256), plan)
    cache = ChartCache(evaluation['source_cache_dir'], evaluation['sources'], max_sources=1)
    for draw in evaluation['windows']:
        check()
        cache.interval(draw)
    for case in evaluation['native_cases']:
        expected = condition_for(cache.get(case['source_sha256']))
        actual = GenerationCondition.from_payload(read_json(case['condition_file'], case['condition_sha256']))
        if actual != expected:
            raise ContractError('Fixed evaluation timing/seed differs from its source')
    if old:
        for sha, pin in plan['sources'].items():
            check()
            folder = Path(config.training.source_cache_dir) / sha
            if (file_digest(folder / 'rows.bin') != pin['rows_sha256'] or
                    file_digest(folder / 'metadata.json') != pin['metadata_sha256']):
                raise ContractError('TRAIN cache changed after restoration preparation')
        for name in ('census', 'selection'):
            read_json(root / f'{name}.json', old['inputs'][name + '_sha256'])
        for receipt in old['plans'].values():
            read_plan(receipt['path'], receipt['sha256'])
        return old
    inputs = prepare_census(config, plan, root, check)
    plans = prepare_plans(config, plan, root)
    check()
    freeze = dict(format='r1-restore/freeze-v1', source_revision=revision, config=settings,
        inputs=inputs, plans=plans, environment=environment)
    (root / 'resolved.yaml').write_text(resolved_yaml)
    publish_json(path, freeze)
    return freeze


def optimizer_by_name(payload):
    settings = dict(payload['config']['model'])
    settings['arm'] = Arm(settings['arm'])
    with torch.device('meta'):
        model = BoundedModel(ModelConfig(**settings))
    names = [name for name, _ in model.named_parameters()]
    ids = payload['optimizer']['param_groups'][0]['params']
    if len(ids) != len(names):
        raise ContractError('Checkpoint optimizer does not match its parameter names')
    return {name: payload['optimizer']['state'].get(identifier) for name, identifier in zip(names, ids)}


def same_state(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(same_state(v, right[k]) for k, v in left.items())
    return left == right


def audit_frozen_parent(parent, child_path):
    if file_digest(Path(parent['checkpoint_file'])) != parent['checkpoint_sha256']:
        raise ContractError('Parent checkpoint changed during its correction stage')
    old = torch.load(parent['checkpoint_file'], map_location='cpu', weights_only=True)
    child = torch.load(child_path, map_location='cpu', weights_only=True)
    before, after = optimizer_by_name(old), optimizer_by_name(child)
    for name, value in old['model'].items():
        if not torch.equal(value, child['model'][name]):
            raise ContractError(f'Frozen inherited parameter changed: {name}')
    for name, values in before.items():
        if not same_state(values, after[name]):
            raise ContractError(f'Frozen inherited Adam state changed: {name}')
    return dict(inherited_parameters_unchanged=True, inherited_adam_unchanged=True,
                parent_checkpoint_sha256=parent['checkpoint_sha256'], child_checkpoint_sha256=file_digest(child_path))


def train_stage(settings, stage, receipt, parent, pool, output, record, save, control):
    record.setdefault('segments', [])
    if active := record.get('active_segment'):
        result_file = Path(active) / 'result.json'
        if not result_file.exists():
            raise ContractError('Interrupted training lacks a finalized ledger; audit the hard kill before recovery')
        record['segments'].append(dict(path=active, result=json.loads(result_file.read_text())))
        record.pop('active_segment')
        save()
    if record['segments'] and record['segments'][-1]['result']['status'] == 'completed':
        return record['segments'][-1]
    cfg = replace(settings, plan_file=receipt['path'], plan_sha256=receipt['sha256'])
    if pool:
        cfg = replace(cfg, recovery_pool=pool['path'], recovery_sha256=pool['sha256'])
    if record['segments']:
        previous = record['segments'][-1]
        if previous['result']['status'] not in ('paused', 'time-limit'):
            raise ContractError('Failed training segments are not automatically retried')
        if previous['result']['status'] == 'time-limit':
            return previous
        cfg = replace(cfg, resume_from=str(Path(previous['path']) / 'checkpoint.pt'))
    elif parent:
        cfg = replace(cfg, fork_from=parent['checkpoint_file'], fork_sha256=parent['checkpoint_sha256'],
                      fork_source_revision=parent['source_revision'], fork_plan_file=parent['plan_file'])
    target = output / f'segment-{len(record["segments"]):05d}'
    cfg = replace(cfg, output_dir=str(target))
    cfg.validate()
    record['active_segment'] = str(target)
    save()
    result = train_run.run_training(cfg, stop_requested=control.boundary)
    record['segments'].append(dict(path=str(target), result=result))
    record.pop('active_segment')
    save()
    return record['segments'][-1]


def run_restore(config, *, resolved_yaml=''):
    config.validate()
    root = Path(config.output_dir).resolve()
    ledger_path = root / 'ledger.json'
    if config.mode == 'status':
        return json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(status='not_started')
    with run_lock(root):
        freeze = preflight(config, root, resolved_yaml)
        freeze_sha = file_digest(root / 'freeze.json')
        if config.mode == 'preflight':
            return dict(status='preflight_passed', freeze_sha256=freeze_sha,
                        stages={s: p['target_onsets'] for s, p in freeze['plans'].items()})
        if config.mode == 'run':
            if ledger_path.exists():
                raise ContractError('Restoration already started; use mode=resume')
            ledger = dict(format='r1-restore/ledger-v1', freeze_sha256=freeze_sha, started_at=time.time(),
                initial_swap_bytes=psutil.swap_memory().used, status='running', quality_status='unreviewed',
                stages={s: dict(status='pending', harvest={}, evaluation={}) for s in STAGES})
        else:
            ledger = json.loads(ledger_path.read_text())
            if ledger['freeze_sha256'] != freeze_sha:
                raise ContractError('Restoration ledger belongs to another execution freeze')
            if ledger['status'] in ('failed', 'needs_review', 'completed', 'budget_exhausted'):
                return ledger
        def save():
            ledger['updated_at'] = time.time()
            publish_json(ledger_path, ledger)
        def control_for(record, phase, seconds):
            began = record.setdefault(phase + '_started_at', time.time())
            save()
            return Control(config, root, ledger['started_at'], began + seconds, ledger['initial_swap_bytes'])
        def pause(result):
            reason = result.get('reason', result.get('pause_reason', 'incomplete_stage'))
            ledger.update(status='budget_exhausted' if reason in ('stage_time_limit', 'queue_time_limit') else 'paused',
                          reason=reason)
            save()
            return ledger
        save()
        base = read_plan(config.base_plan_file, config.base_plan_sha256)
        selection = read_json(root / 'selection.json', freeze['inputs']['selection_sha256'])['selections']
        evaluation = validate_evaluation(read_json(config.evaluation_file, config.evaluation_sha256), base)
        parent = None
        try:
            for stage in STAGES:
                if source_revision() != freeze['source_revision']:
                    raise ContractError('Committed source changed during the restoration queue')
                record = ledger['stages'][stage]
                if record['status'] == 'completed':
                    parent = record['checkpoint']
                    if file_digest(Path(parent['checkpoint_file'])) != parent['checkpoint_sha256']:
                        raise ContractError('Completed stage checkpoint changed')
                    continue
                ledger.update(status='running', active_stage=stage)
                settings = stage_training(config, stage)
                pool = record.get('pool')
                if stage in ('routing', 'release', 'response') and pool is None:
                    record['status'] = 'collecting_native'
                    seconds = config.release_harvest_max_seconds if stage == 'release' else config.harvest_max_seconds
                    control = control_for(record, 'harvest', seconds)
                    with control.signals():
                        pool = collect_pool(config, stage, parent, base, selection[stage], root / stage / 'native',
                                            record['harvest'], save, control)
                    if pool['status'] == 'paused':
                        return pause(pool)
                    record['pool'] = pool
                    if pool['status'] != 'ready':
                        record['status'] = ledger['status'] = 'needs_review'
                        ledger['reason'] = 'insufficient_native_preferences'
                        save()
                        return ledger
                record['status'] = 'training'
                control = control_for(record, 'training', config.training.max_seconds)
                with control.signals():
                    segment = train_stage(settings, stage, freeze['plans'][stage], parent, pool,
                                          root / stage / 'training', record, save, control)
                result = segment['result']
                if result['source_revision'] != freeze['source_revision']:
                    raise ContractError('Training segment differs from the frozen implementation')
                if result['status'] != 'completed':
                    if result['status'] == 'time-limit':
                        return pause(dict(reason='stage_time_limit'))
                    if result['status'] == 'paused':
                        return pause(result)
                    raise ContractError('Training did not complete its declared stage')
                if result['durable_onset_exposures'] != freeze['plans'][stage]['target_onsets']:
                    raise ContractError('Stage checkpoint is at the wrong source exposure')
                checkpoint = Path(segment['path']) / 'checkpoint.pt'
                if file_digest(checkpoint) != result['checkpoint_sha256']:
                    raise ContractError('Stage checkpoint differs from its durable ledger')
                if parent and settings.trainable != 'all' and 'parent_audit' not in record:
                    record['parent_audit'] = audit_frozen_parent(parent, checkpoint)
                    save()
                record['status'] = 'evaluating'
                control = control_for(record, 'evaluation', config.evaluation_max_seconds)
                readout = root / stage / 'readout'
                readout.mkdir(parents=True, exist_ok=True)
                with control.signals():
                    if 'likelihood_sha256' not in record['evaluation']:
                        report, reason = evaluate(checkpoint, settings, evaluation, control)
                        if reason:
                            return pause(dict(reason=reason))
                        record['evaluation']['likelihood_sha256'] = publish_json(readout / 'likelihood.json', report)
                        save()
                    else:
                        read_json(readout / 'likelihood.json', record['evaluation']['likelihood_sha256'])
                    generation = GenerateConfig(checkpoint_file=str(checkpoint), checkpoint_sha256=result['checkpoint_sha256'],
                        cpu_threads=1, device='cpu', resources=settings.resources,
                        footprint_limit_bytes=settings.footprint_limit_bytes, max_seconds=600.)
                    native = record['evaluation'].setdefault('native', {})
                    outcome = run_cases(evaluation['native_cases'], generation, readout / 'native', native, save, control)
                    if outcome['status'] == 'paused':
                        return pause(outcome)
                    if outcome['failed']:
                        raise ContractError('A fixed native readout failed; inspect before continuing the lineage')
                parent = dict(checkpoint_file=str(checkpoint), checkpoint_sha256=result['checkpoint_sha256'],
                    source_revision=result['source_revision'], plan_file=freeze['plans'][stage]['path'],
                    plan_sha256=freeze['plans'][stage]['sha256'], onset_exposures=result['durable_onset_exposures'])
                record.update(status='completed', checkpoint=parent)
                save()
            ledger.update(status='completed', checkpoint=parent,
                          quality_status='requires_longform_ln_tap_and_local_response_review')
            save()
            publish_json(root / 'summary.json', ledger)
            return ledger
        except ResourceLimit as error:
            ledger.update(status='needs_review', reason='resource_limit', error=str(error))
            save()
            return ledger
        except Exception as error:
            ledger.update(status='failed', error=f'{type(error).__name__}: {error}')
            save()
            raise
