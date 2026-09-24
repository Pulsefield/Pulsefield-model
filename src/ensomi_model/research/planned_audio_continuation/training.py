"""Bounded joint fitting of a head plan, physical release clock and action rows.

Sampling and importance weights reuse the frozen full-audio interval protocol.
The three factors keep separate sums; no per-event loss reweighting changes
the joint chart-time likelihood.
"""
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch

from ..joint_audio_continuation.context_training import (
    example_from_identity, freeze_protocol, initial_group_receipts, song_context,
)
from ..joint_audio_continuation.data import _json, digest, load_corpus
from ..joint_audio_continuation.generation import _resource_stop
from ..joint_audio_continuation.model import R1_TRANSFER_MODULES
from ..joint_audio_continuation.training import revision, training_normalization
from .intervals import collate_interval, interval_losses, score_interval
from .model import PlannedAudioModel, PlannedModelConfig, initialize_from_r1

INHERITED = (*R1_TRANSFER_MODULES, 'row_consequence')


def optimizer_groups(model, config):
    groups = [dict(name=name, params=[], param_names=[], lr=rate) for name, rate in
              (('r1', config.inherited_learning_rate), ('audio_skeleton', config.learning_rate))]
    for name, parameter in model.named_parameters():
        group = groups[int(name.split('.')[0] not in INHERITED)]
        group['params'].append(parameter)
        group['param_names'].append(name)
    return groups


def backward_update(model, planned_songs, charts, device):
    totals = np.zeros(4, np.float64)
    counts = dict(intervals=0, milliseconds=0, event_rows=0, head_rows=0, release_rows=0, release_clocks=0)
    for records in planned_songs:
        examples = [example_from_identity(r, charts) for r in records]
        if len({e.chart.entry['audio_sha256'] for e in examples}) != 1:
            raise ValueError('Song microbatch has inconsistent complete audio')
        coarse = song_context(model, examples[0].chart, device)
        total = None
        for example in examples:
            batch = collate_interval(example, model.config, device)
            losses = interval_losses(score_interval(model, batch.inputs, coarse), batch)
            weight = example.weight_per_second / (len(planned_songs) * len(examples))
            loss = losses[-1] * weight
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite planned joint likelihood')
            total = loss if total is None else total + loss
            totals += np.asarray([float(v.detach().cpu()) for v in losses]) * weight
            counts['intervals'] += 1
            counts['milliseconds'] += example.end_ms - example.start_ms
            counts['event_rows'] += len(batch.row_index)
            counts['head_rows'] += int(batch.head_event.sum())
            counts['release_rows'] += int(batch.release_event.sum())
            counts['release_clocks'] += int(batch.inputs.release_valid.sum())
        total.backward()
    return totals, counts


@torch.inference_mode()
def evaluate_intervals(model, records, charts, device):
    model.eval()
    groups = {}
    for record in records:
        groups.setdefault(record['source_sha256'], []).append(record)
    results = []
    for sha, group in groups.items():
        coarse = song_context(model, charts[sha], device)
        for record in group:
            example = example_from_identity(record, charts)
            batch = collate_interval(example, model.config, device)
            losses = interval_losses(score_interval(model, batch.inputs, coarse), batch)
            weight = example.weight_per_second if record['panel'] == 'population' else 1000 / (example.end_ms - example.start_ms)
            results.append(dict(**record, event_rows=len(batch.row_index),
                **{name: float(value.cpu()) * weight for name, value in
                   zip(('head_nll_per_second', 'release_nll_per_second', 'row_nll_per_second', 'joint_nll_per_second'), losses)}))
    summary = {panel: dict(intervals=sum(r['panel'] == panel for r in results),
        **{name: float(np.mean([r[name] for r in results if r['panel'] == panel])) for name in
           ('head_nll_per_second', 'release_nll_per_second', 'row_nll_per_second', 'joint_nll_per_second')})
        for panel in ('population', 'bos')}
    return dict(**summary, records=results)


def save_checkpoint(path, model, optimizer, update, config, source, protocol, transfer):
    temporary = path.with_suffix('.tmp')
    torch.save(dict(format='joint-audio/planned-v1', model_config=asdict(model.config), model=model.state_dict(),
        optimizer=optimizer.state_dict(), update=update, config=asdict(config), source_revision=source,
        manifest_sha256=config.manifest_sha256, protocol=protocol, transfer=transfer,
        torch_rng=torch.get_rng_state(), mps_rng=torch.mps.get_rng_state() if config.device == 'mps' else None), temporary)
    temporary.replace(path)


def train(config, *, resolved_yaml=''):
    config.validate()
    source = revision()
    if digest(Path(config.root) / 'manifest.json') != config.manifest_sha256:
        raise ValueError('Planned corpus differs from its pinned manifest')
    if stop := _resource_stop(Path(config.root)):
        raise RuntimeError(f'Planned training cannot start: {stop}')
    started = time.perf_counter()
    directory = Path(config.root) / 'planned-training' / config.run_name
    directory.mkdir(parents=True, exist_ok=False)
    _json(directory / 'config.json', asdict(config))
    (directory / 'resolved.yaml').write_text(resolved_yaml)
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    charts, norm = load_corpus(config.root)
    by = {c.entry['source_sha256']: c for c in charts}
    norm, norm_identity = training_normalization(config, charts, norm)
    protocol, protocol_identity = freeze_protocol(charts, config)
    model = PlannedAudioModel(PlannedModelConfig(bounded_head=config.bounded_head,
        head_bound=config.head_bound, head_decay_ms=config.head_decay_ms))
    transfer = initialize_from_r1(model, config.r1_checkpoint_file, config.r1_checkpoint_sha256)
    model.set_audio_normalization(torch.tensor(norm['mean']), torch.tensor(norm['std']))
    tracked_modules = tuple(name for name in ('head_temporal', 'head_condition', 'timing', 'release_clock',
        'skeleton_temporal', 'row_consequence', 'head_base') if hasattr(model, name))
    initial = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n.split('.')[0] in tracked_modules}
    model.to(config.device)
    groups = optimizer_groups(model, config)
    receipts = initial_group_receipts(groups)
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    validation = protocol['validation']
    if config.validation_songs:
        selected = sorted({r['source_sha256'] for r in validation})[:config.validation_songs]
        validation = [r for r in validation if r['source_sha256'] in selected]
    _json(directory / 'freeze.json', dict(source_revision=source, config=asdict(config), model_config=asdict(model.config),
        **protocol_identity, **norm_identity, transfer=transfer, parameters=model.parameter_counts(),
        optimizer_groups=receipts, validation=validation,
        environment=dict(torch=str(torch.__version__), python=__import__('sys').version,
                         ram_bytes=psutil.virtual_memory().total)))
    last, reason, latest = 0, 'updates_complete', None
    histories, counts = [], {}
    try:
        with (directory / 'updates.jsonl').open('w') as log:
            for update in range(config.updates + 1):
                if update:
                    stop = _resource_stop(Path(config.root))
                    if stop or time.perf_counter() - started >= config.max_seconds:
                        reason = stop or 'wall_clock_limit'
                        break
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    averages, consumed = backward_update(model, protocol['updates'][update - 1], by, config.device)
                    norm_value = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
                    optimizer.step()
                    last = update
                    for name, value in consumed.items():
                        counts[name] = counts.get(name, 0) + value
                    histories.append(averages)
                    if update % 10 == 0 or update == config.updates:
                        means = np.mean(histories[-10:], axis=0)
                        record = dict(update=update, seconds=time.perf_counter() - started, **counts,
                            **{name: float(v) for name, v in zip(
                                ('head_nll_per_second', 'release_nll_per_second', 'row_nll_per_second', 'joint_nll_per_second'), means)},
                            grad_norm=float(norm_value.cpu()), available_bytes=psutil.virtual_memory().available,
                            rss_bytes=psutil.Process().memory_info().rss,
                            mps_driver_bytes=torch.mps.driver_allocated_memory() if config.device == 'mps' else 0)
                        log.write(json.dumps(record, allow_nan=False) + '\n')
                        log.flush()
                        print(json.dumps(record), flush=True)
                if update % config.validation_every == 0 or update == config.updates:
                    latest = evaluate_intervals(model, validation, by, config.device)
                    _json(directory / f'evaluation-{update}.json', dict(update=update, **latest))
                    print(json.dumps(dict(update=update, population=latest['population'], bos=latest['bos'])), flush=True)
                    save_checkpoint(directory / 'last.pt', model, optimizer, update, config, source, protocol_identity, transfer)
    except BaseException as error:
        _json(directory / 'failure.json', dict(update=last, error=repr(error), seconds=time.perf_counter() - started))
        raise
    save_checkpoint(directory / 'last.pt', model, optimizer, last, config, source, protocol_identity, transfer)
    changes = {module: 0. for module in tracked_modules}
    for name, parameter in model.named_parameters():
        if name in initial:
            changes[name.split('.')[0]] += float((parameter.detach().cpu() - initial[name]).square().sum())
    result = dict(status='completed' if last == config.updates else 'bounded_stop', stop_reason=reason, updates=last,
        seconds=time.perf_counter() - started, **counts, source_revision=source,
        checkpoint_sha256=digest(directory / 'last.pt'), **protocol_identity,
        parameter_update_l2={name: value ** .5 for name, value in changes.items()},
        final_evaluation_update=last if last == config.updates else (last // config.validation_every) * config.validation_every,
        population=latest['population'], bos=latest['bos'], selection='Fixed endpoint; likelihood is not playability selection')
    _json(directory / 'result.json', result)
    return result
