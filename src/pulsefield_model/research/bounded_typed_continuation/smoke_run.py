"""Fresh, resource-bounded training checks on pinned TRAIN source intervals.

This runner measures pipeline learning and compute, not generalization or arm
quality. It saves recoverable parameters/optimizer/draw state at report points;
automatic resume and main-screen sampling are separate future entrypoints.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from ..oracle_time_continuation.corpus import catalog_entries, read_split
from ..oracle_time_continuation.data import SourceIdentity
from ..oracle_time_continuation.runtime import ResourceGuard, atomic_checkpoint, write_record
from ..oracle_time_continuation.storage import file_digest
from ..scoped_style_modeling.dataset import ContractError
from .data import SourceChart, SourceInterval, batch_likelihood, prepare_batch
from .model import BoundedModel
from .smoke_config import SmokeConfig


def source_revision():
    root = Path(__file__).resolve().parents[4]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    if git('status', '--porcelain'):
        raise ContractError('Real-data learning checks require a clean committed product source')
    return git('rev-parse', 'HEAD')


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_intervals(config: SmokeConfig, check):
    if not config.interval_sha256 or file_digest(Path(config.interval_manifest), 1024 ** 2) != config.interval_sha256:
        raise ContractError('Interval manifest differs from its pinned digest')
    manifest = json.loads(Path(config.interval_manifest).read_text())
    if manifest.get('format') != 'bounded-typed/source-intervals-v1' or not 1 <= len(manifest.get('entries', [])) <= 16:
        raise ContractError('Learning check requires 1 to 16 explicitly selected source intervals')
    if any(key in manifest and manifest[key] != getattr(config, key) for key in ('catalog_sha256', 'split_sha256')):
        raise ContractError('Interval manifest allocation provenance differs from the pinned configuration')
    requested = [SourceIdentity(**entry['identity']) for entry in manifest['entries']]
    if len({identity.group_id for identity in requested}) != len(requested):
        raise ContractError('Learning-check source groups must be distinct')
    assignments = read_split(config.split_manifest, config.split_sha256)
    catalog = catalog_entries(config.catalog_path, config.catalog_sha256, assignments,
                              [identity.source_sha256 for identity in requested], split='train')
    by_sha = {entry['source_sha256']: entry for entry in catalog}
    intervals = []
    for entry, identity in zip(manifest['entries'], requested):
        admitted = by_sha[identity.source_sha256]
        expected = SourceIdentity(**{key: admitted[key] for key in asdict(identity)})
        if identity != expected or identity.split != 'train':
            raise ContractError('Interval identity differs from the pinned TRAIN catalog')
        chart = SourceChart.from_cache(Path(config.source_cache_dir) / identity.source_sha256, identity)
        if len(chart.rows) != admitted['event_count']:
            raise ContractError('Source row count differs from the pinned catalog')
        interval = SourceInterval(chart, entry['first_onset'], entry['onset_count'])
        if interval.onset_count > 256 or interval.stop - interval.start > 1024:
            raise ContractError('Learning-check interval exceeds 256 onsets or 1024 physical target rows')
        intervals.append(interval)
        check('source-loaded', charts=len(intervals))
    return manifest, intervals


def synchronize(device):
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()


def summarize(records):
    keys = ('head_nll_sum', 'endpoint_nll_sum', 'source_onsets', 'physical_rows', 'prefix_rows',
            'ln_type_nll_sum', 'ln_type_count', 'tap_type_nll_sum', 'tap_type_count',
            'endpoint_decisions', 'scored_order_factors', 'candidate_pairs')
    result = {key: sum(record[key] for record in records) for key in keys}
    result['nll_per_onset'] = (result['head_nll_sum'] + result['endpoint_nll_sum']) / result['source_onsets']
    result['head_nll_per_onset'] = result['head_nll_sum'] / result['source_onsets']
    for name in ('ln_type', 'tap_type'):
        result[name + '_nll'] = (result[name + '_nll_sum'] / result[name + '_count'] if result[name + '_count'] else None)
    result['endpoint_nll_per_ln'] = (result['endpoint_nll_sum'] / result['endpoint_decisions'] if result['endpoint_decisions'] else None)
    return result


def run_smoke(config: SmokeConfig, *, resolved_yaml=''):
    config.validate()
    required = ('interval_manifest', 'interval_sha256', 'catalog_path', 'catalog_sha256',
                'split_manifest', 'split_sha256', 'source_cache_dir')
    if any(not getattr(config, name) for name in required):
        raise ContractError('Learning check requires pinned interval/catalog/split inputs and an admitted source cache')
    revision = source_revision()
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    flat = json.loads(json.dumps(asdict(config)))
    save_json(output / 'run-config.json', flat)
    (output / 'resolved.yaml').write_text(resolved_yaml)
    save_json(output / 'environment.json', dict(source_revision=revision, python=platform.python_version(),
              torch=torch.__version__, platform=platform.platform(), machine=platform.machine()))
    started, update = time.monotonic(), 0
    torch.set_num_threads(config.cpu_threads)
    with (output / 'resources.jsonl').open('w') as resources, (output / 'training.jsonl').open('wb') as journal:
        guard = ResourceGuard(config.device, config.resources, resources)
        def check(phase, **metadata):
            value = guard.check(phase, **metadata)
            if time.monotonic() - started > config.max_seconds:
                raise TimeoutError('Learning check reached its complete-run wall-time bound')
            total_bytes = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
            if total_bytes > config.resources.output_max_bytes:
                raise ContractError('Learning check exceeded its total output byte budget')
            return value
        try:
            manifest, intervals = load_intervals(config, check)
            save_json(output / 'intervals.json', manifest)
            torch.manual_seed(config.model_seed)
            model = BoundedModel(config.model).to(config.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            rng = np.random.default_rng(config.shuffle_seed)
            order, cursor = [], 0
            exposure = 0
            counts = [0] * len(intervals)

            def measure(items, *, backward=False):
                before_prepare = time.monotonic()
                batch = prepare_batch(items, config.model.arm, model.temporal.config.receptive_tokens,
                                      full_history=model.long_memory is not None)
                after_prepare = time.monotonic()
                diagnostics = {}
                loss, factors = batch_likelihood(model, batch, candidate_budget=config.candidate_budget,
                                                 recompute=backward, diagnostics=diagnostics)
                synchronize(config.device)
                after_forward = time.monotonic()
                check('forward', update=update, backward=backward)
                if backward:
                    loss.backward()
                    synchronize(config.device)
                    check('backward', update=update)
                after_backward = time.monotonic()
                head, endpoint = factors.detach().cpu().tolist()
                record = dict(head_nll_sum=head, endpoint_nll_sum=endpoint, source_onsets=batch.source_onsets,
                              physical_rows=batch.physical_rows, prefix_rows=batch.prefix_rows,
                              context_spans_ms=batch.context_spans_ms, prepare_seconds=after_prepare - before_prepare,
                              forward_seconds=after_forward - after_prepare, backward_seconds=after_backward - after_forward,
                              **diagnostics)
                if batch.memory_valid is not None:
                    record.update(full_history_rows=int(batch.memory_valid.sum()), full_history_padded_rows=int(batch.memory_valid.size))
                return record

            @torch.no_grad()
            def evaluate(label):
                model.eval()
                begin = time.monotonic()
                records = [measure([interval]) for interval in intervals]
                result = dict(summary=summarize(records), intervals=records, seconds=time.monotonic() - begin)
                save_json(output / (label + '.json'), result)
                check(label)
                return result

            initial = evaluate('initial')
            print(json.dumps(dict(phase='initial', arm=config.model.arm.value, **initial['summary'])), flush=True)
            training_started = time.monotonic()
            model.train()
            for update in range(1, config.updates + 1):
                check('update-start', update=update)
                chosen = []
                while len(chosen) < config.batch_size:
                    if cursor == len(order):
                        order, cursor = rng.permutation(len(intervals)).tolist(), 0
                    chosen.append(order[cursor])
                    cursor += 1
                optimizer.zero_grad(set_to_none=True)
                learning_rate = config.learning_rate * min(1., update / max(1, config.warmup_updates))
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate
                record = measure([intervals[i] for i in chosen], backward=True)
                before_step = time.monotonic()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                synchronize(config.device)
                exposure += record['source_onsets']
                for i in chosen:
                    counts[i] += 1
                record.update(update=update, selected_intervals=chosen, learning_rate=learning_rate,
                              grad_norm=float(norm.detach().cpu()), step_seconds=time.monotonic() - before_step,
                              total_onset_exposures=exposure, elapsed_seconds=time.monotonic() - started)
                write_record(journal, record, config.resources)
                journal.flush()
                if update % config.report_every == 0 or update == config.updates:
                    payload = dict(format='bounded-typed/learning-check-v1', source_revision=revision, config=flat,
                                   model=model.state_dict(), optimizer=optimizer.state_dict(), update=update,
                                   source_onset_exposures=exposure, interval_draw_counts=counts,
                                   order=order, cursor=cursor, shuffle_state=rng.bit_generator.state,
                                   torch_rng=torch.get_rng_state())
                    atomic_checkpoint(output / 'checkpoint.pt', payload, config.resources)
                    guard.update_boundary(update=update)
                    print(json.dumps(dict(phase='train', arm=config.model.arm.value, update=update,
                          onset_exposures=exposure, nll_per_onset=(record['head_nll_sum'] + record['endpoint_nll_sum']) / record['source_onsets'],
                          elapsed_seconds=record['elapsed_seconds'])), flush=True)
            training_seconds = time.monotonic() - training_started
            final = evaluate('final')
            result = dict(status='completed', source_revision=revision, arm=config.model.arm.value,
                          parameters=sum(p.numel() for p in model.parameters()), updates=update,
                          initial=initial['summary'], final=final['summary'], training_seconds=training_seconds,
                          total_seconds=time.monotonic() - started, source_onset_exposures=exposure,
                          unique_source_onsets=sum(item.onset_count for i, item in enumerate(intervals) if counts[i]),
                          interval_draw_counts=counts, checkpoint_sha256=file_digest(output / 'checkpoint.pt'))
            check('completed')
            save_json(output / 'result.json', result)
            return result
        except BaseException as error:
            save_json(output / 'failure.json', dict(status='stopped', update=update, exception=type(error).__name__,
                                                    reason=str(error), elapsed_seconds=time.monotonic() - started))
            raise
