"""Corpus training with exact exposure cuts and durable optimizer boundaries.

Each invocation owns a fresh segment directory. A resumed segment retains the
parent's logs, verifies its committed prefix and charges its entire measured
runtime, including discarded work. An unmeasured hard-killed parent requires an
external runtime audit before it can support an equal-compute continuation.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import torch

from ..oracle_time_continuation.runtime import (
    ResourceGuard, atomic_checkpoint, log_boundary, verify_boundary, write_record,
)
from ..oracle_time_continuation.storage import file_digest
from ..scoped_style_modeling.dataset import ContractError
from .corpus import ChartCache, Coverage, next_batch, read_plan
from .data import batch_likelihood, prepare_batch
from .model import BoundedModel
from .smoke_run import save_json, source_revision, synchronize
from .train_config import TrainConfig

FORMAT = 'bounded-typed/corpus-training-v1'
EXECUTION_FIELDS = {'output_dir', 'resume_from', 'stop_after_checkpoint', 'plan_file', 'source_cache_dir'}


def training_identity(config):
    return {key: value for key, value in config.items() if key not in EXECUTION_FIELDS}


def measure(model, intervals, *, candidate_budget, backward, denominator, check):
    """Accumulate one microbatch's summed loss over the effective batch onsets."""
    started = time.monotonic()
    batch = prepare_batch(intervals, model.config.arm, model.temporal.config.receptive_tokens)
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
    return dict(head_nll_sum=head, endpoint_nll_sum=endpoint, source_onsets=batch.source_onsets,
                physical_rows=batch.physical_rows, prefix_rows=batch.prefix_rows,
                padded_rows=int(batch.valid.size), context_spans_ms=batch.context_spans_ms,
                prepare_seconds=prepared - started, forward_seconds=forwarded - prepared,
                backward_seconds=finished - forwarded, **diagnostics)


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
    durable_exposure, durable_update, checkpoint_sha = exposure, update, None
    status, failure, result = 'completed', None, None
    with (output / 'resources.jsonl').open('w') as resource_log, (output / 'training.jsonl').open('wb') as journal:
        try:
            guard = ResourceGuard(config.device, config.resources, resource_log)

            def check(phase):
                guard.check(phase, update=update, onset_exposures=exposure)
                if sum(p.stat().st_size for p in output.iterdir() if p.is_file()) > config.resources.output_max_bytes:
                    raise ContractError('Corpus segment exceeds its output byte budget')

            torch.manual_seed(config.model_seed)
            model = BoundedModel(config.model).to(config.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            if payload:
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
