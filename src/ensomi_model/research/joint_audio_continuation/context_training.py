"""Matched full-audio interval fitting with explicit chart-time weighting.

A frozen group/arrangement/time plan is shared across architecture cells.
Each song microbatch encodes its complete coarse audio once, while exact
interval likelihood reuses causal history and finite local audio crops.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch

from .context_model import ContextAudioModel, ContextModelConfig
from .data import _json, digest, load_corpus, train_groups
from .generation import _resource_stop
from .intervals import IntervalExample, collate_interval, interval_losses, score_interval
from .model import R1_TRANSFER_MODULES, initialize_from_r1
from .training import revision, training_normalization


def make_protocol(charts, config):
    """Return a deterministic architecture-independent exposure/validation plan."""
    groups = train_groups(charts)
    validation = sorted((c for c in charts if c.split == 'validation'), key=lambda c: c.entry['source_sha256'])
    if not groups or not validation:
        raise ValueError('Context training requires distinct TRAIN and validation groups')
    rng = np.random.default_rng(config.sample_seed)
    updates = []
    for _ in range(config.plan_updates):
        songs = []
        for _ in range(config.songs_per_update):
            group = groups[int(rng.integers(len(groups)))]
            if len({c.entry['audio_sha256'] for c in group}) != 1:
                raise ValueError('A shared-song microbatch requires the same paired audio bytes')
            intervals = []
            for _ in range(config.intervals_per_song):
                chart = group[int(rng.integers(len(group)))]
                count = IntervalExample(chart, 0, config.interval_ms).count
                intervals.append(IntervalExample(chart, int(rng.integers(count)), config.interval_ms).identity())
            songs.append(intervals)
        updates.append(songs)
    rng = np.random.default_rng(config.validation_seed)
    probes = []
    for chart in validation:
        count = IntervalExample(chart, 0, config.interval_ms).count
        probes.extend(dict(panel='population', **IntervalExample(chart, int(rng.integers(count)), config.interval_ms).identity())
                      for _ in range(4))
        probes.append(dict(panel='bos', **IntervalExample(chart, 0, config.interval_ms).identity()))
    return dict(format='joint-audio/context-protocol-v1', manifest_sha256=config.manifest_sha256,
        sample_seed=config.sample_seed, validation_seed=config.validation_seed, interval_ms=config.interval_ms,
        plan_updates=config.plan_updates, songs_per_update=config.songs_per_update,
        intervals_per_song=config.intervals_per_song, updates=updates, validation=probes,
        objective='uniform song group, separate arrangement, full-chart NLL per actual integer-clock second')


def freeze_protocol(charts, config):
    """Create the shared plan once or reject a changed sampling/data contract."""
    path = Path(config.root) / f'{config.protocol_name}.json'
    protocol = make_protocol(charts, config)
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError('Existing shared context protocol differs; use a fresh named protocol')
    else:
        with path.open('x') as stream:
            stream.write(json.dumps(protocol, indent=2, allow_nan=False) + '\n')
    return protocol, dict(protocol_file=str(path.resolve()), protocol_sha256=digest(path))


def example_from_identity(record, charts):
    chart = charts[record['source_sha256']]
    example = IntervalExample(chart, record['interval_index'], record['width_ms'])
    if any(record[k] != v for k, v in example.identity().items()):
        raise ValueError('Frozen interval no longer matches its source/audio clock')
    return example


def song_context(model, chart, device):
    """The only coarse-encoder input is complete canonical song Mel."""
    if not model.config.global_audio:
        return None
    frames = len(chart.mel)
    tokens = (frames + 49) // 50
    capacity = 50 * (1 << (tokens - 1).bit_length())
    mel = np.zeros((1, capacity, 128), np.float32)
    mel[0, :frames] = chart.mel
    valid = np.arange(capacity)[None] < frames
    return model.encode_coarse(torch.as_tensor(mel, device=device), torch.as_tensor(valid, device=device))


def backward_update(model, planned_songs, charts, device):
    """Accumulate song microbatches; one optimizer step belongs to the caller."""
    totals = np.zeros(3, np.float64)
    counts = dict(intervals=0, milliseconds=0, event_rows=0, heads=0, timing_bins=0)
    for records in planned_songs:
        examples = [example_from_identity(r, charts) for r in records]
        if len({e.chart.entry['audio_sha256'] for e in examples}) != 1:
            raise ValueError('Planned microbatch does not share the same full audio')
        coarse = song_context(model, examples[0].chart, device)
        total = None
        for example in examples:
            batch = collate_interval(example, model.config, device)
            timing, row, joint = interval_losses(score_interval(model, batch.inputs, coarse), batch)
            weight = example.weight_per_second / (len(planned_songs) * len(examples))
            loss = joint * weight
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite weighted interval likelihood')
            total = loss if total is None else total + loss
            totals += np.asarray([float(v.detach().cpu()) for v in (joint, timing, row)]) * weight
            counts['intervals'] += 1
            counts['milliseconds'] += example.end_ms - example.start_ms
            counts['event_rows'] += len(batch.targets.row_index)
            counts['timing_bins'] += int(batch.inputs.timing_valid.any(-1).sum())
            counts['heads'] += int(sum(sum(a in (1, 2) for a in actions) for actions in
                example.chart.source.rows['actions'][(example.chart.source.rows['time'] >= example.start_ms) &
                                                     (example.chart.source.rows['time'] < example.end_ms)]))
        total.backward()
    return totals, counts


@torch.inference_mode()
def evaluate_intervals(model, records, charts, device, *, context_mode='correct'):
    """Evaluate frozen source intervals, optionally perturbing coarse context.

Population intervals use their importance weights; the separate BOS report
uses actual observed seconds. Local audio is identical across coarse probes.
The shift rotates real encoded coarse tokens by half a song, with no chart input.
"""
    if context_mode not in ('correct', 'zero', 'shift'):
        raise ValueError('Unknown coarse-context diagnostic')
    if context_mode != 'correct' and not model.config.global_audio:
        raise ValueError('Coarse-context diagnostics require a global audio branch')
    model.eval()
    groups = {}
    for record in records:
        groups.setdefault(record['source_sha256'], []).append(record)
    results = []
    for sha, group in groups.items():
        coarse = song_context(model, charts[sha], device)
        if context_mode == 'zero':
            coarse = (torch.zeros_like(coarse[0]), coarse[1])
        elif context_mode == 'shift':
            real_tokens = int(coarse[1][0])
            shifted = coarse[0].clone()
            shifted[:, :real_tokens] = torch.roll(coarse[0][:, :real_tokens], real_tokens // 2, dims=1)
            coarse = (shifted, coarse[1])
        for record in group:
            example = example_from_identity(record, charts)
            batch = collate_interval(example, model.config, device)
            timing, row, joint = interval_losses(score_interval(model, batch.inputs, coarse), batch)
            weight = (example.weight_per_second if record['panel'] == 'population' else
                      1000 / (example.end_ms - example.start_ms))
            results.append(dict(**record, event_rows=len(batch.targets.row_index),
                timing_nll=float(timing.cpu()), row_nll=float(row.cpu()),
                joint_nll_per_second=float(joint.cpu()) * weight))
    summary = {panel: dict(intervals=sum(r['panel'] == panel for r in results),
        joint_nll_per_second=float(np.mean([r['joint_nll_per_second'] for r in results if r['panel'] == panel])))
        for panel in ('population', 'bos')}
    return dict(context_mode=context_mode, **summary, records=results)


def model_identity(model):
    """Hash initial common parameters separately from new architecture modules."""
    h = hashlib.sha256()
    common = ('audio_', 'temporal.', 'exact.', 'fuse.', 'joint.', 'route_residual.', 'release_residual.')
    for name, value in sorted(model.state_dict().items()):
        if name.startswith(common):
            h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def optimizer_groups(model, config):
    """Keep module learning rates fixed even when an R1 stage lacks routing.

    Parameter names are retained in the optimizer checkpoint for auditing;
    whether a tensor was copied is initialization provenance, not its rate.
    """
    groups = [dict(name=name, params=[], param_names=[], lr=rate) for name, rate in
              (('r1', config.inherited_learning_rate), ('audio_timing', config.learning_rate))]
    for name, parameter in model.named_parameters():
        group = groups[int(name.split('.')[0] not in R1_TRANSFER_MODULES)]
        group['params'].append(parameter)
        group['param_names'].append(name)
    return [group for group in groups if group['params']]


def initial_group_receipts(groups):
    receipts = []
    for group in groups:
        identity = hashlib.sha256()
        for name, value in zip(group['param_names'], group['params']):
            identity.update(name.encode())
            identity.update(value.detach().cpu().numpy().tobytes())
        receipts.append(dict(name=group['name'], learning_rate=group['lr'], param_names=group['param_names'],
                             parameters=sum(p.numel() for p in group['params']),
                             initial_parameters_sha256=identity.hexdigest()))
    return receipts


def save_checkpoint(path, model, optimizer, update, config, source, protocol_identity, transfer):
    temporary = path.with_suffix('.tmp')
    torch.save(dict(format='joint-audio/context-v1', model_config=asdict(model.config), model=model.state_dict(),
        optimizer=optimizer.state_dict(), update=update, config=asdict(config), source_revision=source,
        manifest_sha256=config.manifest_sha256, protocol=protocol_identity, transfer=transfer,
        torch_rng=torch.get_rng_state(), mps_rng=torch.mps.get_rng_state() if config.device == 'mps' else None), temporary)
    temporary.replace(path)


def train(config, *, resolved_yaml=''):
    """Fit a fresh fixed endpoint, retaining resources, exposure and stop cause.

No best-checkpoint selection or implicit resume is used. Every cell verifies
the common protocol before learning. Validation is source-conditioned evidence;
native sampling and Lens/player assessment are separate from this entrypoint.
"""
    config.validate()
    source = revision()
    if digest(Path(config.root) / 'manifest.json') != config.manifest_sha256:
        raise ValueError('Context corpus manifest differs from its pinned identity')
    started = time.perf_counter()
    directory = Path(config.root) / 'context-training' / config.run_name
    directory.mkdir(parents=True, exist_ok=False)
    _json(directory / 'config.json', asdict(config))
    (directory / 'resolved.yaml').write_text(resolved_yaml)
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    charts, norm = load_corpus(config.root)
    by = {c.entry['source_sha256']: c for c in charts}
    norm, norm_identity = training_normalization(config, charts, norm)
    protocol, protocol_identity = freeze_protocol(charts, config)
    model = ContextAudioModel(ContextModelConfig(global_audio=config.global_audio, bounded_timing=config.bounded_timing))
    transfer = initialize_from_r1(model, config.r1_checkpoint_file, config.r1_checkpoint_sha256)
    model.set_audio_normalization(torch.tensor(norm['mean']), torch.tensor(norm['std']))
    initial_common_sha = model_identity(model)
    model.to(config.device)
    groups = optimizer_groups(model, config)
    group_receipts = initial_group_receipts(groups)
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    validation = protocol['validation']
    if config.validation_songs:
        selected = sorted({r['source_sha256'] for r in validation})[:config.validation_songs]
        validation = [r for r in validation if r['source_sha256'] in selected]
    _json(directory / 'freeze.json', dict(source_revision=source, config=asdict(config),
        manifest_sha256=config.manifest_sha256, **protocol_identity, **norm_identity, transfer=transfer,
        initial_common_sha256=initial_common_sha, parameters=model.parameter_counts(), validation=validation,
        optimizer_groups=group_receipts,
        environment=dict(torch=torch.__version__, python=__import__('sys').version,
                         ram_bytes=psutil.virtual_memory().total)))
    counts = dict(intervals=0, milliseconds=0, event_rows=0, heads=0, timing_bins=0)
    last, reason, latest = 0, 'updates_complete', None
    histories = []
    try:
        with (directory / 'updates.jsonl').open('w') as log:
            for update in range(config.updates + 1):
                if update:
                    stop = _resource_stop(Path(config.root))
                    if stop or time.perf_counter() - started >= config.max_seconds:
                        reason = stop or 'wall_clock_limit'; break
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    averages, consumed = backward_update(model, protocol['updates'][update - 1], by, config.device)
                    norm_value = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
                    optimizer.step()
                    last = update
                    for name, value in consumed.items():
                        counts[name] += value
                    histories.append(averages)
                    if update % 10 == 0 or update == config.updates:
                        means = np.mean(histories[-10:], axis=0)
                        record = dict(update=update, seconds=time.perf_counter() - started,
                            joint_nll_per_second=float(means[0]), timing_nll_per_second=float(means[1]),
                            row_nll_per_second=float(means[2]), grad_norm=float(norm_value.cpu()), **counts,
                            available_bytes=psutil.virtual_memory().available,
                            rss_bytes=psutil.Process().memory_info().rss,
                            active_mps_bytes=torch.mps.current_allocated_memory() if config.device == 'mps' else 0,
                            mps_driver_bytes=torch.mps.driver_allocated_memory() if config.device == 'mps' else 0)
                        log.write(json.dumps(record, allow_nan=False) + '\n'); log.flush()
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
    result = dict(status='completed' if last == config.updates else 'bounded_stop', stop_reason=reason,
        updates=last, seconds=time.perf_counter() - started, **counts, source_revision=source,
        checkpoint_sha256=digest(directory / 'last.pt'), **protocol_identity,
        final_evaluation_update=last if last == config.updates else (last // config.validation_every) * config.validation_every,
        population=latest['population'], bos=latest['bos'], selection='fixed final update; not a playability selection')
    _json(directory / 'result.json', result)
    return result
