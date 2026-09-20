"""One serial queue with frozen inputs and durable, inspectable stage receipts."""
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import platform
import subprocess
import time

import torch
import psutil

from ..bounded_typed_continuation import generate_run
from ..bounded_typed_continuation.corpus import ChartCache, read_plan
from ..bounded_typed_continuation.condition import GenerationCondition
from ..bounded_typed_continuation.smoke_run import source_revision
from ..oracle_time_continuation.runtime import ResourceGuard, ResourceLimit
from ..oracle_time_continuation.storage import file_digest
from ..scoped_style_modeling.dataset import ContractError
from .audio import run_audio, validate_manifest
from .control import Control, publish_json, read_json, run_lock, tree_bytes
from .stress import run_cases, validate_cases
from .teacher import prepare_plan, run_teacher, validate_evaluation

TERMINAL = {'completed', 'completed_with_anomalies', 'failed', 'budget_exhausted', 'disabled'}


def preflight(config, root, resolved_yaml):
    began = time.time()
    def check_time():
        if time.time() - began >= config.preflight_max_seconds:
            raise ResourceLimit('Input preflight exceeded its wall budget')
    revision = source_revision()
    settings = json.loads(json.dumps(asdict(config)))
    settings.pop('mode')
    freeze_file = root / 'freeze.json'
    old = json.loads(freeze_file.read_text()) if freeze_file.exists() else None
    if old and (old['config'] != settings or old['source_revision'] != revision):
        raise ContractError('Queue configuration or committed implementation differs from its execution freeze')
    inputs = {}
    if config.audio.enabled:
        manifest = validate_manifest(read_json(config.audio.manifest_file, config.audio.manifest_sha256))
        inputs['audio_manifest_sha256'] = config.audio.manifest_sha256
        inputs['audio_assets'] = len(manifest['assets'])
        for utility in ('ffmpeg', 'ffprobe'):
            result = subprocess.run([utility, '-version'], capture_output=True, text=True, timeout=10, check=True)
            inputs[utility] = result.stdout.splitlines()[0]
    if config.teacher.enabled:
        plan = read_plan(config.teacher.base_plan_file, config.teacher.base_plan_sha256)
        evaluation = validate_evaluation(read_json(config.teacher.evaluation_file, config.teacher.evaluation_sha256), plan)
        cache = ChartCache(evaluation['source_cache_dir'], evaluation['sources'], max_sources=1)
        for draw in evaluation['windows']:
            cache.interval(draw)
        for case in evaluation['native_cases']:
            source = cache.get(case['source_sha256'])
            arm = config.teacher.training.model.arm
            expected = GenerationCondition(arm, source.view(arm).timing,
                tuple(source.row(i) for i in range(source.seed_rows)), source.state(arm, source.seed_rows).known_ends)
            actual = GenerationCondition.from_payload(read_json(case['condition_file'], case['condition_sha256']))
            if actual != expected:
                raise ContractError('Native readout condition differs from its pinned source timing/seed')
        # Validate every TRAIN source's bytes without holding the corpus in RAM.
        source_root = Path(config.teacher.training.source_cache_dir)
        for sha, pin in plan['sources'].items():
            check_time()
            if (file_digest(source_root / sha / 'metadata.json') != pin['metadata_sha256'] or
                    file_digest(source_root / sha / 'rows.bin') != pin['rows_sha256']):
                raise ContractError('Training cache changed after the base plan was frozen')
        plan_file = root / 'training-plan.json'
        if old:
            plan_sha = old['inputs']['training_plan_sha256']
            read_plan(plan_file, plan_sha)
        else:
            _, plan_sha = prepare_plan(config.teacher, plan_file)
        inputs.update(training_plan_sha256=plan_sha, base_plan_sha256=config.teacher.base_plan_sha256,
                      evaluation_sha256=config.teacher.evaluation_sha256,
                      train_sources=len(plan['sources']), target_onsets=config.teacher.milestones[-1])
    if config.stress.enabled:
        cases = validate_cases(read_json(config.stress.manifest_file, config.stress.manifest_sha256))
        g = config.stress.generation
        if file_digest(Path(g.checkpoint_file), g.resources.checkpoint_max_bytes) != g.checkpoint_sha256:
            raise ContractError('Stress checkpoint differs from its frozen digest')
        guard = ResourceGuard('cpu', config.resources)
        try:
            model, _ = generate_run._model(replace(g, device='cpu'))
        except Exception as error:
            raise ContractError('Stress checkpoint cannot be loaded by the native generation runner') from error
        for case in cases['cases']:
            condition = GenerationCondition.from_payload(read_json(case['condition_file'], case['condition_sha256']))
            if condition.arm != model.config.arm:
                raise ContractError('Stress condition task arm differs from the pinned model')
        guard.check('stress-model-preflight')
        del model
        gc.collect()
        inputs.update(stress_manifest_sha256=config.stress.manifest_sha256,
                      stress_checkpoint_sha256=g.checkpoint_sha256, stress_cases=len(cases['cases']))
    environment = dict(python=platform.python_version(), torch=str(torch.__version__), platform=platform.platform())
    check_time()
    freeze = dict(format='vacation/freeze-v1', source_revision=revision, config=settings,
                  inputs=inputs, environment=environment)
    if old and freeze != old:
        raise ContractError('Pinned inputs, codec or execution environment changed after preflight')
    if not old:
        publish_json(root / 'run-config.json', settings)
        (root / 'resolved.yaml').write_text(resolved_yaml)
        publish_json(freeze_file, freeze)
    return freeze


def run_queue(config, *, resolved_yaml=''):
    config.validate()
    began = time.time()
    root = Path(config.output_dir).resolve()
    ledger_file = root / 'ledger.json'
    if config.mode == 'status':
        return json.loads(ledger_file.read_text()) if ledger_file.exists() else dict(status='not_started')
    with run_lock(root):
        freeze = preflight(config, root, resolved_yaml)
        freeze_sha = file_digest(root / 'freeze.json')
        if config.mode == 'preflight':
            return dict(status='preflight_passed', freeze_sha256=freeze_sha, **freeze['inputs'])
        if config.mode == 'run':
            if ledger_file.exists():
                raise ContractError('This queue already has a ledger; use mode=resume')
            ledger = dict(format='vacation/ledger-v1', freeze_sha256=freeze_sha, started_at=began,
                          initial_swap_bytes=psutil.swap_memory().used,
                          status='running', stages={name: dict(status='pending', state={})
                          for name in ('audio', 'teacher', 'stress')})
        else:
            ledger = json.loads(ledger_file.read_text())
            if ledger['freeze_sha256'] != freeze_sha:
                raise ContractError('Queue ledger belongs to another execution freeze')

        def save():
            ledger['updated_at'] = time.time()
            publish_json(ledger_file, ledger)

        save()
        for name in ('audio', 'teacher', 'stress'):
            stage = getattr(config, name)
            record = ledger['stages'][name]
            if not stage.enabled:
                record['status'] = 'disabled'
                save()
                continue
            if record['status'] in TERMINAL:
                continue
            record.setdefault('started_at', time.time())
            control = Control(config, root, ledger['started_at'], record['started_at'] + stage.max_seconds,
                              ledger['initial_swap_bytes'])
            record['status'] = 'running'
            ledger['status'] = 'running'
            save()
            try:
                with control.signals():
                    reason = control.boundary()
                    if reason:
                        result = dict(status='paused', reason=reason)
                    elif name == 'audio':
                        result = run_audio(stage, root / name, control)
                    elif name == 'teacher':
                        result = run_teacher(stage, root / name, root / 'training-plan.json',
                            freeze['inputs']['training_plan_sha256'], record['state'], save, control)
                    else:
                        cases = validate_cases(read_json(stage.manifest_file, stage.manifest_sha256))
                        result = run_cases(cases['cases'], stage.generation, root / name, record['state'], save, control)
                if result['status'] == 'paused':
                    reason = result['reason']
                    record['result'] = result
                    if reason in ('stage_time_limit', 'audio_byte_limit'):
                        record['status'] = 'budget_exhausted'
                    else:
                        record['status'] = ledger['status'] = 'paused'
                        save()
                        return ledger
                else:
                    record.update(status=result['status'], result=result)
            except ResourceLimit as error:
                record['status'] = ledger['status'] = 'paused'
                record['result'] = dict(reason='resource_limit', error=str(error))
                save()
                return ledger
            except Exception as error:
                record.update(status='failed', error=f'{type(error).__name__}: {error}')
            save()
        ledger['status'] = ('completed' if all(s['status'] in ('completed', 'disabled') for s in ledger['stages'].values())
                            else 'finished_with_incomplete_stages')
        ledger['product_bytes'] = tree_bytes(root)
        save()
        publish_json(root / 'summary.json', ledger)
        return ledger
