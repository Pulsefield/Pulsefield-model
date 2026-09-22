"""Typed generation runner using a pinned source allocation and trained weights."""
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path

import torch

from ..scoped_style_modeling.dataset import ContractError
from .config import BackboneConfig
from .decoding import DecodeSamplingPolicy
from .generation import Rollout
from .export import presentation_header
from .model import CausalBackbone, WEIGHTS_FORMAT
from .runtime import ResourceConfig, ResourceGuard, validate_envelope
from .storage import SourceCacheConfig, SourceStore
from .corpus import admit_entry, catalog_entries, read_split, source_assignments


@dataclass
class GenerateConfig:
    source_dir: str = 'artifacts/scoped-style-modeling/sources'
    split_manifest: str = 'artifacts/scoped-style-modeling/prepare-v1/split-manifest.json'
    split_sha256: str = ''
    source_sha256: str = ''
    catalog_path: str = ''
    catalog_sha256: str = ''
    split: str = 'validation'
    weights: str = ''
    output_dir: str = 'artifacts/oracle-time-continuation/m3-generate'
    cache_dir: str = 'artifacts/oracle-time-continuation/source-cache-v1'
    device: str = 'cpu'
    cpu_threads: int = 1
    seed: int = 17
    resume: bool = False
    checkpoint_every_rows: int = 512
    decode: DecodeSamplingPolicy = field(default_factory=DecodeSamplingPolicy)
    cache: SourceCacheConfig = field(default_factory=SourceCacheConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)

    def validate(self):
        if self.device not in ('cpu', 'mps', 'cuda') or self.split not in ('train', 'validation'):
            raise ContractError('Generation requires cpu/mps/cuda and an explicit train/validation allocation')
        if not self.weights or not self.source_sha256 or not self.output_dir:
            raise ContractError('Generation requires weights, source_sha256 and output_dir')
        if type(self.cpu_threads) is not int or self.cpu_threads <= 0:
            raise ContractError('cpu_threads must be a positive integer')
        if type(self.checkpoint_every_rows) is not int or not 1 <= self.checkpoint_every_rows <= 8192:
            raise ContractError('checkpoint_every_rows must be an integer in [1,8192]')
        if bool(self.catalog_path) != bool(self.catalog_sha256):
            raise ContractError('catalog_path and catalog_sha256 must be supplied together')
        for value in (self.source_sha256, self.split_sha256, *([self.catalog_sha256] if self.catalog_path else [])):
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ContractError('Generation requires lowercase pinned SHA-256 identities')


def run_generation(config: GenerateConfig, *, resolved_yaml: str):
    config.validate()
    previous = torch.get_num_threads()
    store = SourceStore(Path(config.cache_dir), config.cache)
    try:
        torch.set_num_threads(config.cpu_threads)
        ResourceGuard(config.device, config.resources)
        if config.catalog_path:
            entry, = catalog_entries(config.catalog_path, config.catalog_sha256,
                                     read_split(config.split_manifest, config.split_sha256),
                                     [config.source_sha256], split=config.split)
            source = admit_entry(store, entry)
            source_path = Path(entry['path'])
        else:
            assignments = source_assignments(config.split_manifest, config.split_sha256, [config.source_sha256], config.split)
            source_path = Path(config.source_dir) / (config.source_sha256 + '.osu')
            source = store.admit(source_path, config.source_sha256,
                                 group_id=assignments[config.source_sha256]['group_id'], split=config.split)
        seed = source.minimum_seed()
        if not seed.eligible:
            raise ContractError(f'Ineligible generation source: {seed.ineligible_reason}')
        if Path(config.weights).stat().st_size > config.resources.checkpoint_max_bytes:
            raise ContractError('Weights exceed checkpoint load size limit')
        weights = torch.load(config.weights, map_location='cpu', weights_only=True)
        if weights['format'] != WEIGHTS_FORMAT:
            raise ContractError('Generation requires weights-v2-bounded-time; earlier time encodings require their original runtime')
        model_config = BackboneConfig(**weights['model_config'])
        validate_envelope(model_config)
        model = CausalBackbone(model_config).to(config.device)
        model.load_state_dict(weights['model_state_dict'])
        del weights
        with Rollout(model, source.skeleton, Path(config.output_dir), config.decode, seed=config.seed,
                     resources=config.resources, resume=config.resume, export_header=presentation_header(source_path),
                     source_identity=asdict(source.identity), checkpoint_every_rows=config.checkpoint_every_rows) as rollout:
            if not config.resume:
                (Path(config.output_dir) / 'resolved.yaml').write_text(resolved_yaml)
                rollout.prefill(islice(source.targets, seed.seed_row_count))
            timing = rollout.advance()
            return dict(**rollout.finish(), execution=timing, cache=store.metrics())
    finally:
        store.clear()
        torch.set_num_threads(previous)
