"""Bounded joint likelihood fitting with separate timing and action diagnostics."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
import psutil
import torch

from .batching import batch_losses, collate, score_batch
from .data import _json, digest, load_corpus, query, smoke_charts, train_groups
from .model import JointAudioModel, JointModelConfig, initialize_from_r1
from .sampling import LogicalExample, coverage_examples, draw_examples as draw_logical_examples, full_gap_probes


def revision():
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('Experiments require a clean committed source checkout')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def selected_groups(charts, count):
    """Select whole TRAIN groups, taking one original chart per stratum first."""
    groups = train_groups(charts)
    if not groups:
        raise ValueError('Joint training needs at least one TRAIN group')
    if count == 0 or count >= len(groups):
        return groups
    first = [chart.group_id for chart in smoke_charts(charts)]
    order = {group: index for index, group in enumerate(first)}
    return sorted(groups, key=lambda group: (order.get(group[0].group_id, len(order)), group[0].group_id))[:count]


def sample_query(chart, rng, config):
    """Mix event-prefix likelihood with BOS, arbitrary rests and the outro.

    Physical history always ends at the actual last source row before cursor.
    Querying inside a rest observes conditional survival, not a synthetic row.
    This mixture is a training objective, not a population-weighted score.
    """
    choice = float(rng.random())
    times = chart.source.rows['time']
    if choice < .08:
        cursor = -1
    elif choice < .78:
        next_index = int(rng.integers(len(times)))
        cursor = int(times[next_index - 1]) if next_index else -1
    elif choice < .95:
        cursor = int(rng.integers(-1, chart.duration_ms))
    else:
        cursor = int(rng.integers(min(int(times[-1]), chart.duration_ms - 1), chart.duration_ms))
    return query(chart, cursor, history_limit=config.history_rows, horizon_ms=config.timing_horizon_ms)


def draw_examples(groups, rng, config, count):
    examples = []
    for _ in range(count):
        group = groups[int(rng.integers(len(groups)))]
        chart = group[int(rng.integers(len(group)))]
        examples.append((chart, sample_query(chart, rng, config)))
    return examples


def make_batch(examples, model_config, device):
    return collate([item[1] for item in examples], [item[0] for item in examples], model_config, device=device)


def backward_logical(model, examples, config):
    """Accumulate full waiting likelihood, normalized by logical example count.

    Expanded windows are microbatched at batch_size. Optimizer zero/step and
    clipping belong to the caller and happen once per logical batch. A long
    interval contributes all disjoint survival chunks and exactly one row loss.
    """
    flat = [(example.chart, sample) for example in examples for sample in example.queries]
    sums = np.zeros(3, np.float64)
    for start in range(0, len(flat), config.batch_size):
        batch = make_batch(flat[start:start + config.batch_size], model.config, config.device)
        terms = batch_losses(score_batch(model, batch.inputs), batch)
        loss = terms.total.sum() / len(examples)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite joint timing/action objective')
        loss.backward()
        sums += [float(terms.total.sum().detach().cpu()), float(terms.timing.sum().detach().cpu()),
                 float(terms.row.sum().detach().cpu())]
    return sums / len(examples), len(flat)


@torch.inference_mode()
def evaluate(model, examples, config):
    model.eval()
    records = []
    for start in range(0, len(examples), config.batch_size):
        part = examples[start:start + config.batch_size]
        batch = make_batch(part, model.config, config.device)
        terms = batch_losses(score_batch(model, batch.inputs), batch)
        for (chart, sample), timing, row in zip(part, terms.timing.cpu().tolist(), terms.row.cpu().tolist()):
            records.append(dict(source_sha256=chart.entry['source_sha256'], cursor_ms=sample.cursor_ms,
                                event=not sample.censored, timing_nll=timing, row_nll=row))
    events = [record for record in records if record['event']]
    return dict(queries=len(records), events=len(events),
                timing_nll=float(np.mean([r['timing_nll'] for r in records])),
                row_nll=float(np.mean([r['row_nll'] for r in events])) if events else None,
                joint_nll=float(np.mean([r['timing_nll'] + r['row_nll'] for r in records])), records=records)


@torch.inference_mode()
def evaluate_full_gaps(model, examples, config):
    """Score targeted complete waits separately from the fixed common query panel."""
    model.eval()
    records = []
    for example in examples:
        flat = [(example.chart, sample) for sample in example.queries]
        timing = row = 0.
        for start in range(0, len(flat), config.batch_size):
            batch = make_batch(flat[start:start + config.batch_size], model.config, config.device)
            terms = batch_losses(score_batch(model, batch.inputs), batch)
            timing += float(terms.timing.sum().cpu())
            row += float(terms.row.sum().cpu())
        records.append(dict(source_sha256=example.chart.entry['source_sha256'],
            cursor_ms=example.queries[0].cursor_ms, target_time_ms=example.queries[-1].target_time_ms,
            queries=len(example.queries), timing_nll=timing, row_nll=row, joint_nll=timing + row))
    return dict(logical_examples=len(records), queries=sum(r['queries'] for r in records),
        timing_nll=float(np.mean([r['timing_nll'] for r in records])) if records else None,
        row_nll=float(np.mean([r['row_nll'] for r in records])) if records else None,
        joint_nll=float(np.mean([r['joint_nll'] for r in records])) if records else None, records=records)


def save_checkpoint(path, model, optimizer, update, config, source_revision, rng, manifest_sha, transfer):
    temporary = path.with_suffix('.tmp')
    torch.save(dict(format='joint-audio/model-v1', model_config=asdict(model.config), model=model.state_dict(),
                    optimizer=optimizer.state_dict(), update=update, config=asdict(config),
                    source_revision=source_revision, manifest_sha256=manifest_sha, transfer=transfer,
                    rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                    mps_rng=torch.mps.get_rng_state() if config.device == 'mps' else None), temporary)
    temporary.replace(path)


def train(config, *, resolved_yaml=''):
    """Fit a fresh bounded run; persist last/best checkpoints and explicit stop cause.

    Best means lowest fixed development-query joint NLL. Native sampling and
    Lens inspection remain separate quality evidence. No resume or overwrite is
    implicit. The fixed-query option is a memorization/gradient diagnostic only.
    """
    source_revision = revision()
    started = time.perf_counter()
    directory = Path(config.root).resolve() / 'training' / config.run_name
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'resolved.yaml').write_text(resolved_yaml)
    _json(directory / 'config.json', asdict(config))
    manifest_sha = digest(Path(config.root) / 'manifest.json')
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    charts, normalization = load_corpus(config.root)
    groups = selected_groups(charts, config.train_groups)
    model = JointAudioModel(JointModelConfig(audio_width=config.audio_width, audio_levels=config.audio_levels))
    transfer = initialize_from_r1(model, config.r1_checkpoint_file, config.r1_checkpoint_sha256)
    model.set_audio_normalization(torch.tensor(normalization['mean']), torch.tensor(normalization['std']))
    model.to(config.device)
    copied = set(transfer['copied'])
    inherited = [p for name, p in model.named_parameters() if name in copied]
    new = [p for name, p in model.named_parameters() if name not in copied]
    optimizer = torch.optim.AdamW([dict(params=inherited, lr=config.inherited_learning_rate),
                                   dict(params=new, lr=config.learning_rate)], weight_decay=config.weight_decay)
    fixed = draw_examples(groups, np.random.default_rng(config.seed + 1), config,
                          config.fixed_train_queries) if config.fixed_train_queries else []
    train_probe = fixed or draw_examples(groups, np.random.default_rng(config.seed + 2), config, 48)
    validation = [chart for chart in charts if chart.split == 'validation']
    val_rng = np.random.default_rng(config.seed + 3)
    val_probe = [(chart, sample_query(chart, val_rng, config)) for chart in validation for _ in range(4)]
    if not val_probe:
        raise ValueError('A separate validation cohort is required')
    gap_probe = full_gap_probes(charts, 'validation', config)
    coverage = coverage_examples([c for group in groups for c in group], config) if config.coverage_pass else []
    _json(directory / 'freeze.json', dict(source_revision=source_revision, config=asdict(config),
        manifest_sha256=manifest_sha, normalization_sha256=digest(Path(config.root) / 'normalization.json'),
        transfer=transfer, training_groups=[[chart.entry['source_sha256'] for chart in group] for group in groups],
        train_probe=[dict(source_sha256=c.entry['source_sha256'], cursor_ms=q.cursor_ms) for c, q in train_probe],
        validation_probe=[dict(source_sha256=c.entry['source_sha256'], cursor_ms=q.cursor_ms) for c, q in val_probe],
        full_gap_probe=[dict(source_sha256=e.chart.entry['source_sha256'], cursor_ms=e.queries[0].cursor_ms,
                            target_time_ms=e.queries[-1].target_time_ms, queries=len(e.queries)) for e in gap_probe],
        coverage_pass=[dict(source_sha256=e.chart.entry['source_sha256'], cursor_ms=e.queries[0].cursor_ms,
                           target_time_ms=e.queries[-1].target_time_ms, queries=len(e.queries)) for e in coverage],
        selection='fixed-query TRAIN diagnostic' if fixed else 'fixed VAL query joint NLL; not playability',
        query_mixture=dict(bos=.08, event_prefix=.70, absolute_cursor=.17, outro=.05),
        environment=dict(torch=torch.__version__, python=__import__('sys').version, ram_bytes=psutil.virtual_memory().total)))
    best, best_update, last_update, stop_reason = float('inf'), 0, 0, 'updates_complete'
    histories, consumed_queries, coverage_consumed = [], 0, 0
    try:
        with (directory / 'updates.jsonl').open('w') as log:
            for update in range(config.updates + 1):
                if update:
                    if time.perf_counter() - started >= config.max_seconds:
                        stop_reason = 'wall_clock_limit'; break
                    if (Path(config.root) / 'PAUSE').exists():
                        stop_reason = 'pause_file'; break
                    if update % 20 == 1:
                        if psutil.virtual_memory().available < 2 * 1024 ** 3:
                            stop_reason = 'available_ram_below_2GiB'; break
                        if shutil.disk_usage(directory).free < 40 * 1024 ** 3:
                            stop_reason = 'free_disk_below_40GiB'; break
                    model.train()
                    examples = ([LogicalExample(c, (q,)) for c, q in
                                 [fixed[int(rng.integers(len(fixed)))] for _ in range(config.batch_size)]] if fixed
                                else draw_logical_examples(groups, rng, config, config.batch_size))
                    if update <= len(coverage):
                        examples[0] = coverage[update - 1]
                        coverage_consumed += 1
                    optimizer.zero_grad(set_to_none=True)
                    averages, query_count = backward_logical(model, examples, config)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm,
                                                               error_if_nonfinite=True)
                    optimizer.step()
                    last_update = update
                    consumed_queries += query_count
                    histories.append(averages)
                    if update % 20 == 0:
                        averages = np.mean(histories[-20:], axis=0)
                        record = dict(update=update, seconds=time.perf_counter() - started,
                            joint_nll=float(averages[0]), timing_nll=float(averages[1]), row_nll_per_example=float(averages[2]),
                            logical_examples=update * config.batch_size, physical_queries=consumed_queries,
                            coverage_examples=coverage_consumed,
                            grad_norm=float(grad_norm.cpu()), rss_bytes=psutil.Process().memory_info().rss,
                            available_bytes=psutil.virtual_memory().available,
                            mps_driver_bytes=torch.mps.driver_allocated_memory() if config.device == 'mps' else 0)
                        log.write(json.dumps(record, allow_nan=False) + '\n'); log.flush()
                        print(json.dumps(record), flush=True)
                if update % config.validation_every == 0 or update == config.updates:
                    train_metrics = evaluate(model, train_probe, config)
                    val_metrics = evaluate(model, val_probe, config)
                    gap_metrics = evaluate_full_gaps(model, gap_probe, config)
                    metrics = dict(update=update, train=train_metrics, validation=val_metrics, full_gap_validation=gap_metrics)
                    _json(directory / f'evaluation-{update}.json', metrics)
                    print(json.dumps(dict(update=update, train={k:v for k,v in train_metrics.items() if k != 'records'},
                                          validation={k:v for k,v in val_metrics.items() if k != 'records'},
                                          full_gap_validation={k:v for k,v in gap_metrics.items() if k != 'records'})), flush=True)
                    score = (train_metrics if fixed else val_metrics)['joint_nll']
                    if score < best:
                        best, best_update = score, update
                        save_checkpoint(directory / 'best.pt', model, optimizer, update, config, source_revision,
                                        rng, manifest_sha, transfer)
                    save_checkpoint(directory / 'last.pt', model, optimizer, update, config, source_revision,
                                    rng, manifest_sha, transfer)
    except BaseException as error:
        _json(directory / 'failure.json', dict(update=last_update, error=repr(error), seconds=time.perf_counter()-started))
        raise
    save_checkpoint(directory / 'last.pt', model, optimizer, last_update, config, source_revision, rng, manifest_sha, transfer)
    result = dict(status='completed' if last_update == config.updates else 'bounded_stop', stop_reason=stop_reason,
                  updates=last_update, seconds=time.perf_counter() - started, best_update=best_update,
                  best_joint_nll=best, best_checkpoint_sha256=digest(directory / 'best.pt'),
                  last_checkpoint_sha256=digest(directory / 'last.pt'), source_revision=source_revision,
                  logical_examples=last_update * config.batch_size, physical_queries=consumed_queries,
                  coverage_examples=coverage_consumed, coverage_total=len(coverage),
                  selection='TRAIN memorization diagnostic' if fixed else 'VAL development joint NLL')
    _json(directory / 'result.json', result)
    return dict(result_file=str(directory / 'result.json'), **result)
