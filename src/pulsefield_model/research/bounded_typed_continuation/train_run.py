"""Corpus training with exact exposure cuts and durable optimizer boundaries.

Each invocation owns a fresh segment directory. A resumed segment retains the
parent's logs, verifies its committed prefix and charges its entire measured
runtime, including discarded work. An unmeasured hard-killed parent requires an
external runtime audit before it can support an equal-compute continuation.
"""
from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
import json
from pathlib import Path
import platform
import time

import torch

from ..oracle_time_continuation.runtime import (
    ResourceGuard, ResourceLimit, atomic_checkpoint, log_boundary, verify_boundary, write_record,
)
from ..oracle_time_continuation.storage import file_digest
from ..scoped_style_modeling.dataset import ContractError
from .corpus import ChartCache, Coverage, next_batch, read_plan
from .contract import Arm
from .data import batch_likelihood, prepare_batch
from .model import BoundedModel, ModelConfig
from .memory import footprint_bytes
from .recovery import RecoveryPool
from .smoke_run import save_json, source_revision, synchronize
from .train_config import TrainConfig

RECOVERY_FIELDS = ('recovery_pool', 'recovery_sha256', 'recovery_weight', 'recovery_queries', 'recovery_seed')
FORMAT = 'bounded-typed/corpus-training-v1'
EXECUTION_FIELDS = {'output_dir', 'resume_from', 'stop_after_checkpoint', 'plan_file', 'source_cache_dir',
                    'fork_from', 'fork_sha256', 'fork_source_revision', 'fork_plan_file'}


def training_identity(config):
    result = deepcopy({key: value for key, value in config.items() if key not in EXECUTION_FIELDS})
    if result.get('trainable', 'all') == 'all':
        result.pop('trainable', None)
    if result['model'].get('head_routing', 'none') == 'none':
        for field in ('head_routing', 'routing_hidden'):
            result['model'].pop(field, None)
    if not result.get('recovery_pool'):
        for field in RECOVERY_FIELDS:
            result.pop(field, None)
    if result['model'].get('long_memory', 'none') == 'none':
        for field in ('long_memory', 'memory_hidden', 'memory_stride'):
            result['model'].pop(field, None)
    return result


def measure(model, intervals, *, candidate_budget, backward, denominator, check):
    """Accumulate one microbatch's summed loss over the effective batch onsets."""
    started = time.monotonic()
    batch = prepare_batch(intervals, model.config.arm, model.temporal.config.receptive_tokens,
                          full_history=model.long_memory is not None)
    prepared = time.monotonic()
    diagnostics = {}
    loss, factors = batch_likelihood(model, batch, candidate_budget=candidate_budget,
                                     recompute=backward, diagnostics=diagnostics)
    device = next(model.parameters()).device.type
    synchronize(device)
    forwarded = time.monotonic()
    check('forward')
    if backward:
        (loss * (batch.source_onsets / denominator)).backward()
        synchronize(device)
        check('backward')
    finished = time.monotonic()
    head, endpoint = factors.detach().cpu().tolist()
    result = dict(head_nll_sum=head, endpoint_nll_sum=endpoint, source_onsets=batch.source_onsets,
                physical_rows=batch.physical_rows, prefix_rows=batch.prefix_rows,
                padded_rows=int(batch.valid.size), context_spans_ms=batch.context_spans_ms,
                prepare_seconds=prepared - started, forward_seconds=forwarded - prepared,
                backward_seconds=finished - forwarded, **diagnostics)
    if batch.memory_valid is not None:
        result.update(full_history_rows=int(batch.memory_valid.sum()), full_history_padded_rows=int(batch.memory_valid.size))
    return result


def add_metrics(total, record):
    for key, value in record.items():
        if key != 'context_spans_ms':
            total[key] = total.get(key, 0) + value


def read_resume(path, config, revision, plan):
    path = Path(path)
    parent = path.parent
    result_path = parent / 'result.json'
    if not result_path.is_file():
        raise ContractError('Resume needs a finalized parent runtime ledger; unmeasured hard-kill time requires an audit')
    result = json.loads(result_path.read_text())
    if result['checkpoint_sha256'] != file_digest(path):
        raise ContractError('Resume must use the final durable checkpoint identified by its parent segment')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (payload['format'] != FORMAT or payload['source_revision'] != revision or
            training_identity(payload['config']) != training_identity(config)):
        raise ContractError('Resume source or scientific/resource configuration differs from the checkpoint')
    cursor = payload['cursor']
    exposure = plan['draws'][cursor - 1]['exposure_end'] if cursor else 0
    if payload['source_onset_exposures'] != exposure or result['durable_onset_exposures'] != exposure:
        raise ContractError('Resume cursor, exposure and parent ledger disagree')
    verify_boundary(parent / 'training.jsonl', payload['journal_boundary'])
    charged = result['compute_seconds']
    if not isinstance(charged, (int, float)) or not 0 <= charged < float('inf'):
        raise ContractError('Parent compute ledger must be finite and nonnegative')
    return payload, float(charged), dict(path=str(result_path.resolve()), sha256=file_digest(result_path),
                                         discarded_updates=result['last_completed_update'] - payload['update'])


def read_fork(config, plan):
    """Validate an explicit source transition and an immutable plan extension.

    Optional zero-initialized residuals or a native recovery objective may be
    added. The original plan must have completed, and every old draw and source pin must remain an exact prefix.
    The operator pins the audited parent source; this is not ordinary resume.
    """
    path = Path(config['fork_from'])
    if file_digest(path) != config['fork_sha256']:
        raise ContractError('Fork checkpoint differs from its pinned digest')
    reference = torch.load(path, map_location='cpu', weights_only=True)
    old_plan = read_plan(config['fork_plan_file'], reference['config']['plan_sha256'])
    payload, charged, parent = read_resume(path, reference['config'], config['fork_source_revision'], old_plan)
    result = json.loads((path.parent / 'result.json').read_text())
    if (result['status'] != 'completed' or payload['cursor'] != len(old_plan['draws']) or
            parent['discarded_updates'] or result['source_revision'] != payload['source_revision']):
        raise ContractError('Fork initialization requires a completed, fully durable parent plan')
    old_identity, new_identity = (deepcopy(training_identity(c)) for c in (payload['config'], config))
    added_recovery = not old_identity.get('recovery_pool') and bool(new_identity.get('recovery_pool'))
    if added_recovery:
        for field in RECOVERY_FIELDS:
            new_identity.pop(field)
    added_routing = (old_identity['model'].get('head_routing', 'none') == 'none' and
                     new_identity['model'].get('head_routing') == 'residual')
    if added_routing:
        if old_identity.get('trainable', 'all') != 'all' or new_identity.pop('trainable', None) != 'routing':
            raise ContractError('A head-routing fork must freeze the entire inherited policy')
        for field in ('head_routing', 'routing_hidden'):
            new_identity['model'].pop(field)
    fields = ('endpoint_availability', 'row_consequence', 'seed_context', 'long_memory')
    old_modes = [old_identity['model'].pop(field, 'none') for field in fields]
    new_modes = [new_identity['model'].pop(field, 'none') for field in fields]
    added_memory = old_modes[-1] == 'none' and new_modes[-1] == 'landmarks'
    if added_memory:
        for field in ('memory_hidden', 'memory_stride'):
            new_identity['model'].pop(field)
    old_identity.pop('plan_sha256')
    new_identity.pop('plan_sha256')
    # Objective forks keep the entire architecture; residual extensions append
    # parameters while retaining the existing order and optimizer states.
    preserved = ((added_memory and not added_recovery and old_modes[:-1] == new_modes[:-1]) or
                 ((added_recovery or added_routing) and not added_memory and old_modes == new_modes))
    if (((added_recovery or added_routing) and old_modes != new_modes) or
            (any(mode != 'none' for mode in old_modes) and not preserved) or old_identity != new_identity):
        raise ContractError('Fork scientific configuration may only add optional residuals, frozen-base routing or native recovery and extend its plan')
    base_keys = set(old_plan) - {'sampling', 'draws'}
    if ({key: old_plan[key] for key in base_keys} != {key: plan.get(key) for key in base_keys} or
            set(plan) != set(old_plan) or
            {k: v for k, v in old_plan['sampling'].items() if k != 'milestones'} !=
            {k: v for k, v in plan['sampling'].items() if k != 'milestones'} or
            plan['sampling']['milestones'][:len(old_plan['sampling']['milestones'])] != old_plan['sampling']['milestones'] or
            len(plan['draws']) <= len(old_plan['draws']) or
            plan['draws'][:len(old_plan['draws'])] != old_plan['draws']):
        raise ContractError('Fork plan must preserve every source, sampler setting and old draw prefix')
    parent.update(kind='fork', source_revision=payload['source_revision'], checkpoint_sha256=config['fork_sha256'],
                  plan_sha256=payload['config']['plan_sha256'], endpoint_availability=config['model']['endpoint_availability'],
                  row_consequence=config['model']['row_consequence'], seed_context=config['model']['seed_context'],
                  long_memory=config['model']['long_memory'])
    if added_recovery:
        parent.update(recovery_pool_sha256=config['recovery_sha256'])
    if added_routing:
        parent.update(head_routing=config['model']['head_routing'], trainable=config['trainable'])
    return payload, charged, parent


def restore_fork(model, optimizer, payload):
    """Copy existing parameters and Adam state; appended residual moments start empty."""
    settings = dict(payload['config']['model'])
    settings['arm'] = Arm(settings['arm'])
    previous = BoundedModel(ModelConfig(**settings))
    previous.load_state_dict(payload['model'])
    old_names = [name for name, _ in previous.named_parameters()]
    new_names = [name for name, _ in model.named_parameters()]
    added = new_names[len(old_names):]
    prefixes = ('pointer.availability_residual.', 'row_consequence.', 'seed_residual.', 'long_memory.', 'route_residual.')
    if new_names[:len(old_names)] != old_names or any(not name.startswith(prefixes) for name in added):
        raise ContractError('Fork model must preserve parameter order and append only optional residuals')
    missing, unexpected = model.load_state_dict(payload['model'], strict=False)
    if set(missing) != set(added) or unexpected:
        raise ContractError('Fork model weights do not match its declared residual extension')
    state = deepcopy(payload['optimizer'])
    current_groups = optimizer.state_dict()['param_groups']
    if len(state['param_groups']) != 1 or len(current_groups) != 1:
        raise ContractError('Fork migration requires the single corpus AdamW parameter group')
    old_ids, new_ids = state['param_groups'][0]['params'], current_groups[0]['params']
    if len(old_ids) != len(old_names) or not set(state['state']).issubset(old_ids):
        raise ContractError('Fork optimizer state does not match its parent parameters')
    translation = dict(zip(old_ids, new_ids))
    state['state'] = {translation[key]: value for key, value in state['state'].items()}
    state['param_groups'][0]['params'] = new_ids
    optimizer.load_state_dict(state)


def configure_trainable(model, scope):
    """Keep frozen parameters in Adam for exact inherited-state preservation."""
    if scope == 'routing' and model.route_residual is None:
        raise ContractError('Routing-only updates require the head-routing residual')
    if scope not in ('all', 'routing'):
        raise ContractError('Unknown trainable parameter scope')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(scope == 'all' or name.startswith('route_residual.'))


def run_training(config: TrainConfig, *, resolved_yaml=''):
    config.validate()
    if any(not getattr(config, key) for key in ('plan_file', 'plan_sha256', 'source_cache_dir')):
        raise ContractError('Corpus training requires a pinned shared plan and admitted source cache')
    revision = source_revision()
    flat = json.loads(json.dumps(asdict(config)))
    started = time.monotonic()
    plan = read_plan(config.plan_file, config.plan_sha256)
    if config.stop_after_checkpoint is not None and config.stop_after_checkpoint not in plan['sampling']['milestones']:
        raise ContractError('Requested stopping point is not a shared exposure checkpoint')
    payload, prior_seconds, parent = None, 0., None
    if config.resume_from is not None:
        payload, prior_seconds, parent = read_resume(config.resume_from, flat, revision, plan)
    elif config.fork_from is not None:
        payload, prior_seconds, parent = read_fork(flat, plan)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'run-config.json', flat)
    (output / 'resolved.yaml').write_text(resolved_yaml)
    save_json(output / 'environment.json', dict(source_revision=revision, python=platform.python_version(),
              torch=str(torch.__version__), platform=platform.platform(), machine=platform.machine(), parent=parent))
    torch.set_num_threads(config.cpu_threads)
    cursor = payload['cursor'] if payload else 0
    update = payload['update'] if payload else 0
    exposure = payload['source_onset_exposures'] if payload else 0
    total = dict(payload['metrics']) if payload else {}
    coverage = Coverage(plan['sources'], payload['coverage'] if payload else None)
    cache = ChartCache(config.source_cache_dir, plan['sources'], max_sources=config.cache_max_sources,
                        max_bytes=config.cache_max_bytes)
    recovery = None
    durable_exposure, durable_update, checkpoint_sha = exposure, update, None
    status, failure, result = 'completed', None, None
    with (output / 'resources.jsonl').open('w') as resource_log, (output / 'training.jsonl').open('wb') as journal:
        try:
            guard = ResourceGuard(config.device, config.resources, resource_log)

            def check(phase):
                footprint = footprint_bytes()
                guard.check(phase, update=update, onset_exposures=exposure, footprint_bytes=footprint)
                if footprint is not None and footprint > config.footprint_limit_bytes:
                    raise ResourceLimit(f'{phase}: task footprint={footprint} exceeds {config.footprint_limit_bytes}')
                if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > config.resources.output_max_bytes:
                    raise ContractError('Corpus segment exceeds its output byte budget')

            if config.recovery_pool:
                recovery = RecoveryPool(config.recovery_pool, config.recovery_sha256, plan, cache)
                check('native-recovery-loaded')
            torch.manual_seed(config.model_seed)
            model = BoundedModel(config.model).to(config.device)
            configure_trainable(model, config.trainable)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            if payload:
                if config.fork_from is not None:
                    restore_fork(model, optimizer, payload)
                else:
                    model.load_state_dict(payload['model'])
                    optimizer.load_state_dict(payload['optimizer'])
                torch.set_rng_state(payload['torch_rng'])
                if config.device == 'mps':
                    torch.mps.set_rng_state(payload['device_rng'])
                elif config.device == 'cuda':
                    torch.cuda.set_rng_state(payload['device_rng'])
            model.train()

            def checkpoint(*, milestone=False):
                nonlocal durable_exposure, durable_update, checkpoint_sha
                device_rng = (torch.mps.get_rng_state() if config.device == 'mps' else
                              torch.cuda.get_rng_state() if config.device == 'cuda' else None)
                state = dict(format=FORMAT, source_revision=revision, config=flat, model=model.state_dict(),
                             optimizer=optimizer.state_dict(), cursor=cursor, update=update,
                             source_onset_exposures=exposure, coverage=coverage.snapshot(), metrics=total,
                             torch_rng=torch.get_rng_state(), device_rng=device_rng, journal_boundary=log_boundary(journal))
                atomic_checkpoint(output / 'checkpoint.pt', state, config.resources)
                durable_exposure, durable_update = exposure, update
                checkpoint_sha = file_digest(output / 'checkpoint.pt')
                if milestone:
                    atomic_checkpoint(output / f'exposure-{exposure}.pt', state, config.resources)
                guard.update_boundary(update=update)
                check('checkpoint')

            checkpoint()
            while cursor < len(plan['draws']):
                if exposure == config.stop_after_checkpoint:
                    status = 'paused'
                    break
                if prior_seconds + time.monotonic() - started >= config.max_seconds:
                    status = 'time-limit'
                    checkpoint()
                    break
                check('update-start')
                draws, next_cursor = next_batch(plan, cursor, config.batch_size)
                actual = sum(draw['onset_count'] for draw in draws)
                learning_rate = config.learning_rate * min(1., (exposure + actual) / max(1, config.warmup_onsets))
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate
                optimizer.zero_grad(set_to_none=True)
                record, spans = {}, []
                before_load = time.monotonic()
                for first in range(0, len(draws), config.microbatch_size):
                    intervals = [cache.interval(draw) for draw in draws[first:first + config.microbatch_size]]
                    measured = measure(model, intervals, candidate_budget=config.candidate_budget,
                                       backward=True, denominator=actual, check=check)
                    add_metrics(record, measured)
                    spans.extend(measured['context_spans_ms'])
                    del intervals
                record['load_seconds'] = time.monotonic() - before_load - sum(record[k] for k in
                    ('prepare_seconds', 'forward_seconds', 'backward_seconds'))
                if recovery is not None:
                    before_recovery = time.monotonic()
                    record.update(recovery.backward(model, update=update, count=config.recovery_queries,
                        seed=config.recovery_seed, weight=config.recovery_weight, check=check))
                    record['recovery_seconds'] = time.monotonic() - before_recovery
                before_step = time.monotonic()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                synchronize(config.device)
                record['step_seconds'] = time.monotonic() - before_step
                cursor, exposure, update = next_cursor, draws[-1]['exposure_end'], update + 1
                coverage.commit(draws)
                add_metrics(total, record)
                write_record(journal, dict(**record, context_spans_ms=spans, update=update, cursor=cursor,
                             learning_rate=learning_rate, grad_norm=float(norm.detach().cpu()),
                             total_onset_exposures=exposure, compute_seconds=prior_seconds + time.monotonic() - started,
                             cache=cache.metrics()), config.resources)
                journal.flush()
                milestone = draws[-1]['checkpoint'] is not None
                if update % config.report_every == 0 or milestone:
                    checkpoint(milestone=milestone)
                    print(json.dumps(dict(phase='train', arm=config.model.arm.value, update=update,
                          onset_exposures=exposure, **coverage.metrics(),
                          nll_per_onset=(record['head_nll_sum'] + record['endpoint_nll_sum']) / actual,
                          compute_seconds=prior_seconds + time.monotonic() - started)), flush=True)
            if checkpoint_sha is not None and durable_update != update:
                checkpoint()
        except BaseException as error:
            status, failure = 'stopped', dict(type=type(error).__name__, message=str(error))
            raise
        finally:
            result = dict(status=status, source_revision=revision, arm=config.model.arm.value,
                          source_onset_exposures=exposure, durable_onset_exposures=durable_exposure,
                          last_completed_update=update, durable_update=durable_update,
                          checkpoint_sha256=checkpoint_sha, coverage=coverage.metrics(), metrics=total,
                          segment_seconds=time.monotonic() - started,
                          compute_seconds=prior_seconds + time.monotonic() - started,
                          parent=parent, failure=failure, cache=cache.metrics())
            save_json(output / 'result.json', result)
    return result
