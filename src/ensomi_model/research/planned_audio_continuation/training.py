"""Bounded joint fitting of a head plan, physical release clock and action rows.

Sampling and importance weights reuse the frozen full-audio interval protocol.
The three factors keep separate sums; no per-event loss reweighting changes
the joint chart-time likelihood.
"""
from dataclasses import asdict
import hashlib
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
from .profiles import build_profile_bank, chart_assignment

INHERITED = (*R1_TRANSFER_MODULES, 'row_consequence')
LOSS_NAMES = ('head_nll_per_second', 'release_nll_per_second', 'row_nll_per_second',
              'conditional_nll_per_second', 'profile_nll_per_second', 'joint_nll_per_second')
MATERIALIZER_MODULES = (*INHERITED, 'audio_residual', 'context_condition', 'preview_condition',
                        'skeleton_temporal', 'release_clock', 'row_counts')


def frozen_fingerprint(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def configure_train_scope(model, scope):
    """Materializer-only fitting keeps every shared H/audio/profile tensor fixed."""
    if scope not in ('all', 'materializer'):
        raise ValueError('Training scope must be all or materializer')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(scope == 'all' or name.split('.')[0] in MATERIALIZER_MODULES)
    return dict(scope=scope, frozen_sha256=frozen_fingerprint(model),
                frozen_parameters=[n for n, p in model.named_parameters() if not p.requires_grad])


def optimizer_groups(model, config):
    groups = [dict(name=name, params=[], param_names=[], lr=rate) for name, rate in
              (('r1', config.inherited_learning_rate), ('audio_skeleton', config.learning_rate))]
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = groups[int(name.split('.')[0] not in INHERITED)]
        group['params'].append(parameter)
        group['param_names'].append(name)
    return groups


def _weighted_losses(losses, weight, prior, profile_index, duration_ms, population_divisor):
    """A profile is one chart factor; all factors share the inclusive native clock."""
    conditional = losses[-1] * weight
    profile = (conditional.new_zeros(()) if prior is None else
               -prior[0, profile_index] * (1000 / (duration_ms + 1)) / population_divisor)
    return torch.stack((*(v * weight for v in losses[:3]), conditional, profile, conditional + profile))


def backward_update(model, planned_songs, charts, device, profiles=None):
    totals = np.zeros(len(LOSS_NAMES), np.float64)
    counts = dict(intervals=0, milliseconds=0, event_rows=0, head_rows=0, release_rows=0, release_clocks=0)
    for records in planned_songs:
        examples = [example_from_identity(r, charts) for r in records]
        if len({e.chart.entry['audio_sha256'] for e in examples}) != 1:
            raise ValueError('Song microbatch has inconsistent complete audio')
        coarse = song_context(model, examples[0].chart, device)
        prior = model.profile_log_probs(coarse) if model.config.profile_count else None
        total = None
        for example in examples:
            profile_index = profiles[example.chart.entry['source_sha256']] if prior is not None else None
            batch = collate_interval(example, model.config, device)
            losses = interval_losses(score_interval(model, batch.inputs, coarse, profile_index=profile_index), batch)
            divisor = len(planned_songs) * len(examples)
            values = _weighted_losses(losses, example.weight_per_second / divisor, prior,
                                      profile_index, example.chart.duration_ms, divisor)
            loss = values[-1]
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite planned joint likelihood')
            total = loss if total is None else total + loss
            totals += values.detach().cpu().numpy()
            counts['intervals'] += 1
            counts['milliseconds'] += example.end_ms - example.start_ms
            counts['event_rows'] += len(batch.row_index)
            counts['head_rows'] += int(batch.head_event.sum())
            counts['release_rows'] += int(batch.release_event.sum())
            counts['release_clocks'] += int(batch.inputs.release_valid.sum())
        # A silent interval has no learnable factor when H/audio are frozen.
        if total.requires_grad:
            total.backward()
    return totals, counts


@torch.inference_mode()
def evaluate_intervals(model, records, charts, device, profiles=None):
    model.eval()
    groups = {}
    for record in records:
        groups.setdefault(record['source_sha256'], []).append(record)
    results = []
    for sha, group in groups.items():
        coarse = song_context(model, charts[sha], device)
        prior = model.profile_log_probs(coarse) if model.config.profile_count else None
        profile_index = profiles[sha] if prior is not None else None
        for record in group:
            example = example_from_identity(record, charts)
            batch = collate_interval(example, model.config, device)
            losses = interval_losses(score_interval(model, batch.inputs, coarse, profile_index=profile_index), batch)
            weight = example.weight_per_second if record['panel'] == 'population' else 1000 / (example.end_ms - example.start_ms)
            values = _weighted_losses(losses, weight, prior, profile_index, example.chart.duration_ms, 1)
            results.append(dict(**record, event_rows=len(batch.row_index),
                **{name: float(value.cpu()) for name, value in zip(LOSS_NAMES, values)}, arrangement_profile=profile_index))
    summary = {panel: dict(intervals=sum(r['panel'] == panel for r in results),
        **{name: float(np.mean([r[name] for r in results if r['panel'] == panel])) for name in
           LOSS_NAMES})
        for panel in ('population', 'bos')}
    return dict(**summary, records=results, arrangement_condition='reference_profile' if profiles is not None else 'none')


def save_checkpoint(path, model, optimizer, update, config, source, protocol, transfer):
    temporary = path.with_suffix('.tmp')
    family = 'joint-audio/planned-profile-v1' if model.config.profile_count else 'joint-audio/planned-v1'
    torch.save(dict(format=family, model_config=asdict(model.config), model=model.state_dict(),
        optimizer=optimizer.state_dict(), update=update, config=asdict(config), source_revision=source,
        manifest_sha256=config.manifest_sha256, protocol=protocol, transfer=transfer,
        torch_rng=torch.get_rng_state(), mps_rng=torch.mps.get_rng_state() if config.device == 'mps' else None), temporary)
    temporary.replace(path)


def initialize_from_planned(model, config, norm):
    """Copy pinned weights with fresh optimizer state and optional profile routing.

    A profiled source requires the same bank and preserves its learned prior.
    An unprofiled source can initialize the common tensors of a profiled model.
    Same-bank changes may route density or add the separate count conditional.
    """
    from .generation import load_model
    baseline, metadata = load_model(config.initial_checkpoint_file, config.initial_checkpoint_sha256)
    expected = asdict(model.config)
    if baseline.config.profile_count not in (0, model.config.profile_count):
        raise ValueError('Planned initialization differs in arrangement profile count')
    expected['profile_count'] = baseline.config.profile_count
    expected['profile_head_rate_downstream'] = baseline.config.profile_head_rate_downstream
    expected['row_factorization'] = baseline.config.row_factorization
    expected['minimum_action_gap_ms'] = baseline.config.minimum_action_gap_ms
    if asdict(baseline.config) != expected or metadata['manifest_sha256'] != config.manifest_sha256:
        raise ValueError('Planned initialization differs in architecture or training corpus')
    for name, values in (('audio_mean', norm['mean']), ('audio_std', norm['std'])):
        if not torch.equal(getattr(baseline, name), getattr(baseline, name).new_tensor(values)):
            raise ValueError('Planned initialization uses different audio normalization')
    if baseline.config.profile_count:
        for name in ('profile_values', 'profile_codes', 'profile_mean', 'profile_std'):
            if not torch.equal(getattr(model, name), getattr(baseline, name)):
                raise ValueError('Planned initialization uses a different arrangement profile bank')
    source = baseline.state_dict()
    missing, unexpected = model.load_state_dict(source, strict=False)
    added = {'profile_values', 'profile_codes', 'profile_mean', 'profile_std',
             'profile_condition.weight', 'profile_prior.weight', 'profile_prior.bias'} if (
                 model.config.profile_count and not baseline.config.profile_count) else set()
    if model.config.row_factorization == 'count_layout' and baseline.config.row_factorization == 'flat':
        added.update(n for n in model.state_dict() if n.startswith('row_counts.'))
    if set(missing) != added or unexpected:
        raise ValueError('Warm initialization did not copy exactly the common model')
    return dict(initialization='planned_weights_fresh_optimizer', checkpoint_sha256=config.initial_checkpoint_sha256,
                source_revision=metadata['source_revision'], copied=sorted(source), new=sorted(added),
                profile_head_rate_downstream=dict(source=baseline.config.profile_head_rate_downstream,
                                                  target=model.config.profile_head_rate_downstream),
                row_factorization=dict(source=baseline.config.row_factorization, target=model.config.row_factorization),
                minimum_action_gap_ms=dict(source=baseline.config.minimum_action_gap_ms,
                                           target=model.config.minimum_action_gap_ms))


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
    bank, profiles = None, None
    if config.profile_bank_file is not None:
        if digest(config.profile_bank_file) != config.profile_bank_sha256:
            raise ValueError('Arrangement profile bank differs from its pinned SHA-256')
        bank = json.loads(Path(config.profile_bank_file).read_text())
        if bank != build_profile_bank(charts, bank['count']):
            raise ValueError('Profile bank does not reproduce from the current TRAIN corpus')
        profiles = {sha: chart_assignment(c, bank) for sha, c in by.items()}
        _json(directory / 'profile_bank.json', bank)
    model = PlannedAudioModel(PlannedModelConfig(bounded_head=config.bounded_head,
        head_bound=config.head_bound, head_decay_ms=config.head_decay_ms,
        condition_full_holds=config.condition_full_holds, profile_count=0 if bank is None else bank['count'],
        profile_head_rate_downstream=config.profile_head_rate_downstream, row_factorization=config.row_factorization,
        minimum_action_gap_ms=config.minimum_action_gap_ms))
    if bank is not None:
        model.configure_profiles(bank)
    transfer = (initialize_from_planned(model, config, norm) if config.initial_checkpoint_file is not None else
                initialize_from_r1(model, config.r1_checkpoint_file, config.r1_checkpoint_sha256))
    model.set_audio_normalization(torch.tensor(norm['mean']), torch.tensor(norm['std']))
    scope = configure_train_scope(model, config.train_scope)

    def verify_frozen():
        if frozen_fingerprint(model) != scope['frozen_sha256']:
            raise RuntimeError('Frozen head/audio/profile parameters changed during materializer fitting')

    tracked_modules = tuple(name for name in ('head_temporal', 'head_condition', 'timing', 'release_clock',
        'skeleton_temporal', 'row_consequence', 'head_base', 'profile_condition', 'profile_prior', 'row_counts') if hasattr(model, name))
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
        optimizer_groups=receipts, parameter_scope=scope, validation=validation,
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
                    averages, consumed = backward_update(model, protocol['updates'][update - 1], by, config.device, profiles)
                    norm_value = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
                    optimizer.step()
                    last = update
                    for name, value in consumed.items():
                        counts[name] = counts.get(name, 0) + value
                    histories.append(averages)
                    if update % 10 == 0 or update == config.updates:
                        means = np.mean(histories[-10:], axis=0)
                        record = dict(update=update, seconds=time.perf_counter() - started, **counts,
                            **{name: float(v) for name, v in zip(LOSS_NAMES, means)},
                            grad_norm=float(norm_value.cpu()), available_bytes=psutil.virtual_memory().available,
                            rss_bytes=psutil.Process().memory_info().rss,
                            mps_driver_bytes=torch.mps.driver_allocated_memory() if config.device == 'mps' else 0)
                        log.write(json.dumps(record, allow_nan=False) + '\n')
                        log.flush()
                        print(json.dumps(record), flush=True)
                if update % config.validation_every == 0 or update == config.updates:
                    verify_frozen()
                    latest = evaluate_intervals(model, validation, by, config.device, profiles)
                    _json(directory / f'evaluation-{update}.json', dict(update=update, **latest))
                    print(json.dumps(dict(update=update, population=latest['population'], bos=latest['bos'])), flush=True)
                    save_checkpoint(directory / 'last.pt', model, optimizer, update, config, source, protocol_identity, transfer)
    except BaseException as error:
        _json(directory / 'failure.json', dict(update=last, error=repr(error), seconds=time.perf_counter() - started))
        raise
    verify_frozen()
    save_checkpoint(directory / 'last.pt', model, optimizer, last, config, source, protocol_identity, transfer)
    changes = {module: 0. for module in tracked_modules}
    for name, parameter in model.named_parameters():
        if name in initial:
            changes[name.split('.')[0]] += float((parameter.detach().cpu() - initial[name]).square().sum())
    result = dict(status='completed' if last == config.updates else 'bounded_stop', stop_reason=reason, updates=last,
        seconds=time.perf_counter() - started, **counts, source_revision=source, parameter_scope=scope,
        checkpoint_sha256=digest(directory / 'last.pt'), **protocol_identity,
        parameter_update_l2={name: value ** .5 for name, value in changes.items()},
        final_evaluation_update=last if last == config.updates else (last // config.validation_every) * config.validation_every,
        population=latest['population'], bos=latest['bos'], selection='Fixed endpoint; likelihood is not playability selection')
    _json(directory / 'result.json', result)
    return result
