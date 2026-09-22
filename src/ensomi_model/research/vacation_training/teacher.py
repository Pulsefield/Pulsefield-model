"""Ordinary source supervision at fixed exposure cuts, with fixed held-out readouts."""
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path

import torch

from ..bounded_typed_continuation import train_run
from ..bounded_typed_continuation.corpus import ChartCache, SamplingConfig, draw_plan, read_plan
from ..bounded_typed_continuation.generate_config import GenerateConfig
from ..bounded_typed_continuation.model import BoundedModel
from ..oracle_time_continuation.storage import file_digest
from ..oracle_time_continuation.runtime import ResourceGuard
from ..scoped_style_modeling.dataset import ContractError
from .control import publish_json, read_json
from .stress import run_cases, validate_cases


def validate_evaluation(value, plan):
    if value.get('format') != 'vacation/teacher-evaluation-v1' or not value.get('windows'):
        raise ContractError('Teacher readouts require pinned vacation/teacher-evaluation-v1 windows')
    train_groups = {s['identity']['group_id'] for s in plan['sources'].values()}
    splits = set()
    for draw in value['windows']:
        pin = value['sources'][draw['source_sha256']]
        identity = pin['identity']
        split = identity['split']
        splits.add(split)
        if split == 'train':
            if plan['sources'].get(draw['source_sha256']) != pin:
                raise ContractError('TRAIN readouts must retain the frozen training source pins')
        elif split != 'validation' or identity['group_id'] in train_groups:
            raise ContractError('Evaluation sources must be TRAIN or disjoint held-out VAL groups')
        if (type(draw['first_onset']) is not int or type(draw['onset_count']) is not int or
                not 0 <= draw['first_onset'] < pin['onsets'] or not 1 <= draw['onset_count'] <= 256):
            raise ContractError('Readout windows must use bounded source-onset crops')
        if draw['first_onset'] + draw['onset_count'] > pin['onsets']:
            raise ContractError('Readout window exceeds its source')
    if splits != {'train', 'validation'}:
        raise ContractError('Teacher readouts need fixed TRAIN and VAL windows to report their gap')
    cases = validate_cases(dict(format='vacation/native-cases-v1', cases=value.get('native_cases', [])))['cases']
    if any(c['split'] != 'validation' or c['group_id'] in train_groups for c in cases):
        raise ContractError('Checkpoint native readouts must use disjoint VAL groups')
    for case in cases:
        pin = value['sources'].get(case['source_sha256'])
        if pin is None or (pin['identity']['group_id'], pin['identity']['split']) != (case['group_id'], case['split']):
            raise ContractError('Native readout provenance must match a pinned evaluation source')
    return value


def prepare_plan(config, destination):
    base = read_plan(config.base_plan_file, config.base_plan_sha256)
    sampling = SamplingConfig(**{**base['sampling'], 'milestones': tuple(config.milestones)})
    plan = {**base, 'sampling': asdict(sampling), 'draws': draw_plan(base['sources'], sampling)}
    sha = publish_json(destination, plan)
    read_plan(destination, sha)
    return plan, sha


@torch.no_grad()
def evaluate(checkpoint, config, evaluation, control):
    guard = ResourceGuard(config.device, config.resources)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = BoundedModel(config.model).to(config.device).eval()
    model.load_state_dict(payload['model'])
    del payload
    guard.check('evaluation-model-loaded')
    cache = ChartCache(evaluation['source_cache_dir'], evaluation['sources'],
                       max_sources=config.cache_max_sources, max_bytes=config.cache_max_bytes)
    records, totals = [], {}
    for draw in evaluation['windows']:
        if reason := control.boundary():
            return None, reason
        interval = cache.interval(draw)
        def check(phase):
            guard.check('evaluation-' + phase)
            control.storage()
        metrics = train_run.measure(model, [interval], candidate_budget=config.candidate_budget, backward=False,
                                    denominator=interval.onset_count, check=check)
        split = interval.chart.identity.split
        train_run.add_metrics(totals.setdefault(split, {}), metrics)
        records.append(dict(**draw, split=split, metrics=metrics))
    for metrics in totals.values():
        metrics['nll_per_onset'] = (metrics['head_nll_sum'] + metrics['endpoint_nll_sum']) / metrics['source_onsets']
    return dict(cases=records, totals=totals,
                validation_train_nll_gap=totals['validation']['nll_per_onset'] - totals['train']['nll_per_onset']), None


def run_teacher(config, output, plan_file, plan_sha, state, save_state, control):
    plan = read_plan(plan_file, plan_sha)
    evaluation = validate_evaluation(read_json(config.evaluation_file, config.evaluation_sha256), plan)
    state.setdefault('segments', [])
    state.setdefault('evaluations', {})
    output.mkdir(parents=True, exist_ok=True)
    if state.get('active_segment'):
        result_file = Path(state['active_segment']) / 'result.json'
        if not result_file.exists():
            raise ContractError('Interrupted teacher segment lacks a runtime ledger; audit hard-kill time before resuming')
        state['segments'].append(dict(path=state.pop('active_segment'), result=json.loads(result_file.read_text())))
        save_state()
    for milestone in config.milestones:
        while not state['segments'] or state['segments'][-1]['result']['durable_onset_exposures'] < milestone:
            if reason := control.boundary():
                return dict(status='paused', reason=reason)
            parent = state['segments'][-1] if state['segments'] else None
            target = output / f'segment-{len(state["segments"]):05d}'
            settings = replace(config.training, plan_file=str(plan_file), plan_sha256=plan_sha,
                               output_dir=str(target), max_seconds=config.max_seconds,
                               stop_after_checkpoint=milestone,
                               resume_from=str(Path(parent['path']) / 'checkpoint.pt') if parent else None)
            control.storage(3 * settings.resources.checkpoint_max_bytes)
            state['active_segment'] = str(target)
            save_state()
            result = train_run.run_training(settings, stop_requested=control.boundary)
            state['segments'].append(dict(path=str(target), result=result))
            state.pop('active_segment')
            save_state()
            if result['status'] == 'time-limit':
                return dict(status='paused', reason='stage_time_limit')
            if result.get('pause_reason'):
                return dict(status='paused', reason=result['pause_reason'])
        parent = state['segments'][-1]
        checkpoint = Path(parent['path']) / 'checkpoint.pt'
        if file_digest(checkpoint) != parent['result']['checkpoint_sha256']:
            raise ContractError('Teacher checkpoint differs from its durable segment receipt')
        entry = state['evaluations'].setdefault(str(milestone), {})
        directory = output / f'readout-{milestone}'
        directory.mkdir(exist_ok=True)
        if not entry.get('likelihood_sha256'):
            report, reason = evaluate(checkpoint, config.training, evaluation, control)
            if reason:
                return dict(status='paused', reason=reason)
            entry['likelihood_sha256'] = publish_json(directory / 'likelihood.json', report)
            save_state()
            gc.collect()
        generation = GenerateConfig(execution_profile=config.training.execution_profile,
            checkpoint_file=str(checkpoint), checkpoint_sha256=file_digest(checkpoint), device=config.training.device,
            cpu_threads=config.training.cpu_threads, candidate_budget=config.training.candidate_budget,
            footprint_limit_bytes=config.training.footprint_limit_bytes, resources=config.training.resources)
        native = entry.setdefault('native', {})
        result = run_cases(evaluation['native_cases'], generation, directory / 'native', native, save_state, control)
        if result['status'] == 'paused':
            return result
        entry['native_result'] = result
        save_state()
    return dict(status='completed', checkpoint=str(checkpoint), checkpoint_sha256=file_digest(checkpoint),
                onset_exposures=state['segments'][-1]['result']['durable_onset_exposures'])
