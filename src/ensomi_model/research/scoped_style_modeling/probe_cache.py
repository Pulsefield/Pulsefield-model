"""Checkpoint-bound CPU/disk backbone cache; no target enters the cached state."""
from dataclasses import asdict
import json
from pathlib import Path
from time import monotonic

import torch
from torch.nn.utils.rnn import pad_sequence

from .config import ModelConfig
from .dataset import ContractError, checked_bytes, digest
from .model import initialize_model
from .probe_data import TENSOR_VERSION, input_identity
from .train import memory_snapshot, synchronize, write_json


def load_backbone(config, corpus):
    path = Path(config.checkpoint)
    sha = digest(path.read_bytes())
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['cohort_sha256'] != corpus.summary['parent_cohort_sha256'] or checkpoint['split_sha256'] != corpus.summary['split_sha256']:
        raise ContractError('Frozen checkpoint and parent preparation/split differ')
    if checkpoint['beta'] != 0 or checkpoint['seed'] != 17:
        raise ContractError('Probe B requires the style-only seed 17 checkpoint')
    if checkpoint['config'] != asdict(config.model):
        raise ContractError('Probe B model dimensions must match the frozen checkpoint')
    model = initialize_model(ModelConfig(**checkpoint['config']), 17, auxiliary=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.encoder.eval(), sha


def cache_identity(row, checkpoint_sha):
    return {'checkpoint_sha256': checkpoint_sha, 'input_id': input_identity(row),
            'chart_sha256': row['chart_sha256'], 'playback_rate': row['playback_rate'],
            'tensor_version': TENSOR_VERSION}


def build_cache(config, corpus):
    started = monotonic()
    encoder, checkpoint_sha = load_backbone(config, corpus)
    encoder.to(config.device)
    directory = Path(config.cache_dir)
    directory.mkdir(parents=True, exist_ok=False)
    indices = sorted(set(corpus.train_indices+corpus.indices('human', 'validation')+corpus.indices('machine', 'validation')))
    unique = {}
    for i in indices:
        unique.setdefault(input_identity(corpus.records[i]), i)
    manifest = {'checkpoint_sha256': checkpoint_sha, 'tensor_version': TENSOR_VERSION,
                'cohort_sha256': corpus.summary['cohort_sha256'], 'split_sha256': corpus.summary['split_sha256'],
                'target_policy_sha256': corpus.summary['target_policy_sha256'], 'entries': {}}
    peak = {}
    with torch.inference_mode():
        for count, (identity, index) in enumerate(unique.items(), 1):
            chart, examples, _, _ = corpus.unique_batch([index])
            state = encoder(chart.to(config.device))[0].cpu()
            payload = {'identity': cache_identity(corpus.records[index], checkpoint_sha), 'states': state,
                       'length': int(chart.lengths[0]), 'section_indices': chart.section_indices[0],
                       'section_events': chart.section_events[0],
                       'source_times_ms': [r.time_ms for r in examples[0].chart.inputs.rows]}
            path = directory/f'{identity}.pt'
            torch.save(payload, path)
            manifest['entries'][identity] = {'sha256': digest(path.read_bytes()), 'bytes': path.stat().st_size,
                                            'length': payload['length']}
            for key, value in memory_snapshot(config.device).items():
                peak[key] = max(peak.get(key, 0), value)
            if count % 100 == 0:
                print(f'Cached {count}/{len(unique)} inputs', flush=True)
    synchronize(config.device)
    manifest.update(runtime_seconds=monotonic()-started, bytes=sum(v['bytes'] for v in manifest['entries'].values()),
                    sampled_peak_memory=peak, device=config.device)
    write_json(directory/'manifest.json', manifest)
    return {k: v for k, v in manifest.items() if k != 'entries'} | {'inputs': len(unique)}


class BackboneCache:
    def __init__(self, config, corpus):
        self.directory = Path(config.cache_dir)
        self.manifest = json.loads((self.directory/'manifest.json').read_text())
        self.encoder, self.checkpoint_sha = load_backbone(config, corpus)
        expected = {'checkpoint_sha256': self.checkpoint_sha, 'tensor_version': TENSOR_VERSION,
                    'cohort_sha256': corpus.summary['cohort_sha256'], 'split_sha256': corpus.summary['split_sha256'],
                    'target_policy_sha256': corpus.summary['target_policy_sha256']}
        if any(self.manifest.get(k) != v for k, v in expected.items()):
            raise ContractError('Backbone cache identity differs; use a fresh cache directory')

    def batch(self, records):
        states = []
        for row in records:
            identity = input_identity(row)
            entry = self.manifest['entries'][identity]
            path = self.directory/f'{identity}.pt'
            checked_bytes(path, entry['sha256'])
            data = torch.load(path, weights_only=True, map_location='cpu')
            if data['identity'] != cache_identity(row, self.checkpoint_sha):
                raise ContractError('Cached state identity mismatch')
            if data['states'].shape != (entry['length'], 2, self.encoder.hand.hidden_size*2):
                raise ContractError('Cached state dimensions differ from the encoder')
            if not torch.isfinite(data['states']).all():
                raise ContractError('Cached encoder states are nonfinite')
            states.append(data['states'])
        return pad_sequence(states, batch_first=True)
