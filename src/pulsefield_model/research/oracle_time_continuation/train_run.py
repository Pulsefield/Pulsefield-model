"""Disk-backed sequence training with update-boundary durable recovery."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import torch

from ..scoped_style_modeling.dataset import ContractError
from .checkpoint import runtime_identity
from .config import BackboneConfig
from .corpus import admit_entry, catalog_entries, read_split, source_assignments
from .model import CausalBackbone, WEIGHTS_FORMAT
from .runtime import (ResourceGuard, ResourceLimit, align_boundary, atomic_checkpoint, log_boundary,
                      training_checkpoint_tensor_bytes, verify_boundary, write_record)
from .storage import SourceStore, file_digest
from .training import SequenceTrainer
from .training_config import TrainExperimentConfig
from .windows import WindowSampler


def load_training_sources(config: TrainExperimentConfig, *, store=None, progress=lambda **kw: None):
    if config.catalog_path:
        assignments = read_split(config.split_manifest, config.split_sha256)
        entries = catalog_entries(config.catalog_path, config.catalog_sha256, assignments, config.source_sha256)
        store = store or SourceStore(Path(config.cache_dir), config.cache)
        sources = []
        for index, entry in enumerate(entries):
            sources.append(admit_entry(store, entry))
            if (index + 1) % 128 == 0:
                progress(admitted=index + 1, total=len(entries), **store.metrics())
        return tuple(sources)
    assignments = source_assignments(config.split_manifest, config.split_sha256, config.source_sha256, 'train')
    store = store or SourceStore(Path(config.cache_dir), config.cache)
    return tuple(store.admit(Path(config.source_dir) / (sha + '.osu'), sha,
                             group_id=assignments[sha]['group_id'], split='train') for sha in config.source_sha256)


def run_training(config: TrainExperimentConfig, *, resolved_yaml: str) -> dict:
    """Recover the last whole update, discarding any incomplete draw/gradient work.

    A checkpoint precedes the first draw and follows each configured interval
    of complete updates, including the requested final update.
    It owns optimizer, sampler/Torch RNGs, counters and both journal positions.
    Only the requested total update count may change on resume; model, data,
    optimizer and runtime identity must match. A new run can initialize from
    pinned weights while starting fresh optimizer, RNGs and exposure counters;
    this is separate from exact resume. Final weights omit optimizer state.
    """
    config.validate()
    if not config.output_dir:
        raise ContractError('Training requires an output_dir')
    output = Path(config.output_dir)
    if not config.resume and output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ContractError('Training output_dir must be absent or empty')
    output.mkdir(parents=True, exist_ok=True)
    previous_threads = torch.get_num_threads()
    store = SourceStore(Path(config.cache_dir), config.cache)
    try:
        torch.set_num_threads(config.cpu_threads)
        with (output / 'resources.jsonl').open('a') as resource_log:
            guard = ResourceGuard(config.device, config.resources, resource_log)
            sources = load_training_sources(config, store=store,
                                            progress=lambda **kw: guard.check('source-admission', **kw))
            guard.check('source-admission', **store.metrics())
            sampler = WindowSampler(sources, config.windows)
            guard.check('window-population', gap_strata=sampler.gap_population(), **store.metrics())
            torch.manual_seed(config.model_seed)
            model = CausalBackbone(config.model).to(config.device)
            if training_checkpoint_tensor_bytes(model) > config.resources.checkpoint_max_bytes:
                raise ResourceLimit('Model plus AdamW moments exceed checkpoint_max_bytes before training')
            guard.check('model-initialization', checkpoint_tensor_budget=training_checkpoint_tensor_bytes(model))
            initialization = None
            if config.initial_weights and not config.resume:
                path = Path(config.initial_weights)
                if path.stat().st_size > config.resources.checkpoint_max_bytes:
                    raise ContractError('Initial weights exceed configured load size limit')
                if file_digest(path) != config.initial_weights_sha256:
                    raise ContractError('Initial weights differ from their pinned SHA-256')
                weights = torch.load(path, map_location='cpu', weights_only=True)
                if weights['format'] != WEIGHTS_FORMAT or BackboneConfig(**weights['model_config']) != config.model:
                    raise ContractError('Initial weights format or model configuration mismatch')
                model.load_state_dict(weights['model_state_dict'])
                initialization = dict(weights_sha256=config.initial_weights_sha256,
                                      source_updates=weights['updates'])
                del weights
            trainer = SequenceTrainer(model, config.training, config.objective)
            trainer.resource_check = guard.check
            trainer.prepare_target = guard.prepare_target
            trainer.resource_check_every_rows = config.resources.check_every_rows
            identity = {key: value for key, value in asdict(config).items()
                        if key not in ('updates', 'resume', 'output_dir')}
            identity['runtime'] = runtime_identity(config.device)
            prefill_rows = target_rows = 0
            paths = {name: output / (name + '.jsonl') for name in ('windows', 'updates')}
            if config.resume:
                if (output / 'checkpoint.pt').stat().st_size > config.resources.checkpoint_max_bytes:
                    raise ContractError('Training checkpoint exceeds configured load size limit')
                checkpoint = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=True)
                if checkpoint['format'] != 'oracle-time/training-v1' or checkpoint['identity'] != identity:
                    raise ContractError('Training resume configuration or runtime identity mismatch')
                for name, path in paths.items():
                    verify_boundary(path, checkpoint['logs'][name])
                model.load_state_dict(checkpoint['model_state_dict'])
                trainer.optimizer.load_state_dict(checkpoint['optimizer'])
                sampler.rng.setstate(checkpoint['sampler_rng'])
                torch.set_rng_state(checkpoint['torch_rng'])
                if config.device == 'mps':
                    torch.mps.set_rng_state(checkpoint['device_rng'])
                elif config.device == 'cuda':
                    torch.cuda.set_rng_state_all(checkpoint['device_rng'])
                trainer.updates, trainer.clipped_updates = checkpoint['updates'], checkpoint['clipped_updates']
                initialization = checkpoint['initialization']
                prefill_rows, target_rows = checkpoint['prefill_rows'], checkpoint['target_rows']
                for name, path in paths.items():
                    align_boundary(path, checkpoint['logs'][name])
                del checkpoint
            else:
                (output / 'resolved.yaml').write_text(resolved_yaml)
                (output / 'run-config.json').write_text(json.dumps(asdict(config), indent=2) + '\n')
                (output / 'population.json').write_text(json.dumps(dict(
                    split_sha256=config.split_sha256, groups=len(sampler.groups),
                    gap_strata=sampler.gap_population(),
                    eligible_sources=[asdict(chart.source.identity) for chart in sampler.charts.values()],
                    excluded_sources=sampler.excluded), indent=2) + '\n')
            with paths['windows'].open('ab') as draw_log, paths['updates'].open('ab') as update_log:
                def save():
                    payload = dict(format='oracle-time/training-v1', identity=identity,
                                   model_config=asdict(model.config), model_state_dict=model.state_dict(),
                                   initialization=initialization,
                                   optimizer=trainer.optimizer.state_dict(), sampler_rng=sampler.rng.getstate(),
                                   torch_rng=torch.get_rng_state(),
                                   device_rng=torch.mps.get_rng_state() if config.device == 'mps' else
                                              torch.cuda.get_rng_state_all() if config.device == 'cuda' else None,
                                   updates=trainer.updates, clipped_updates=trainer.clipped_updates,
                                   prefill_rows=prefill_rows, target_rows=target_rows,
                                   logs=dict(windows=log_boundary(draw_log), updates=log_boundary(update_log)))
                    size = atomic_checkpoint(output / 'checkpoint.pt', payload, config.resources)
                    guard.check('checkpoint', update=trainer.updates, **size, **store.metrics())
                if not config.resume:
                    save()
                report = None
                while trainer.updates < config.updates:
                    report = trainer.update(sampler.draw_batch(config.training.effective_batch_size))
                    for index, window in enumerate(report.pop('windows')):
                        write_record(draw_log, dict(update=report['update'], window=index,
                                                    sampling_seed=config.windows.seed, **window), config.resources)
                    prefill_rows += report['prefill_rows']
                    target_rows += report['target_rows']
                    report.update(cumulative_prefill_rows=prefill_rows, cumulative_target_rows=target_rows,
                                  learning_rate=trainer.optimizer.param_groups[0]['lr'],
                                  weight_decay=trainer.optimizer.param_groups[0]['weight_decay'])
                    write_record(update_log, report, config.resources)
                    trainer.optimizer.zero_grad(set_to_none=True)
                    guard.update_boundary(update=trainer.updates)
                    if trainer.updates % config.checkpoint_every_updates == 0 or trainer.updates == config.updates:
                        save()
            atomic_checkpoint(output / 'weights.pt', dict(format=WEIGHTS_FORMAT,
                              model_config=asdict(config.model), model_state_dict=model.state_dict(),
                              updates=trainer.updates, initialization=initialization), config.resources)
            guard.check('weights')
            return dict(output_dir=str(output.resolve()), updates=trainer.updates, prefill_rows=prefill_rows,
                        target_rows=target_rows, last_update=report, cache=store.metrics(),
                        initialization=initialization)
    finally:
        store.clear()
        torch.set_num_threads(previous_threads)
