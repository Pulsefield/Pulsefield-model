"""Typed configuration for the seed 17 development probes; test is never selected."""
from dataclasses import dataclass, field
import math

from .config import ModelConfig
from .dataset import ContractError

ROOT = 'artifacts/scoped-style-modeling'
SEED_RUN = ROOT + '/overnight-17-29-43-v1/seed-17'


@dataclass
class ProbeConfig:
    stage: str = 'audit'
    prepared_dir: str = ROOT + '/prepare-v1'
    probe_prepared_dir: str = ROOT + '/prepare-probes-v2'
    seed_run_dir: str = SEED_RUN
    checkpoint: str = SEED_RUN + '/style-only/best.pt'
    cache_dir: str = ROOT + '/seed17-backbone-cache-v2'
    output_dir: str = ROOT + '/probe-a-v1'
    localized_trill_path: str | None = None
    device: str = 'mps'
    seed: int = 17
    max_updates: int = 1000
    arm_budget_seconds: float = 900.0
    evaluation_every: int = 100
    throughput_updates: int = 5
    batch_size: int = 16
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    gradient_cap: float = 1.0
    cpu_threads: int = 1
    pace_radii: list[int] = field(default_factory=lambda: [2, 8])
    dilations: list[int] = field(default_factory=lambda: [1, 2, 4])
    model: ModelConfig = field(default_factory=ModelConfig)

    def validate(self):
        self.model.validate()
        if self.stage not in ('prepare', 'audit', 'cache', 'readout', 'pilot'):
            raise ContractError('stage must be prepare, audit, cache, readout, or pilot')
        if self.device not in ('cpu', 'mps', 'cuda'):
            raise ContractError('device must be explicit cpu, mps, or cuda')
        for name in ('max_updates', 'evaluation_every', 'throughput_updates', 'batch_size', 'cpu_threads'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ContractError(f'{name} must be a positive integer')
        if type(self.seed) is not int or self.seed < 0:
            raise ContractError('seed must be a nonnegative integer')
        for name in ('arm_budget_seconds', 'learning_rate', 'gradient_cap', 'weight_decay'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name != 'weight_decay' and value == 0):
                raise ContractError(f'{name} has an invalid value')
        if len(self.pace_radii) != 2 or any(type(x) is not int or x < 1 for x in self.pace_radii):
            raise ContractError('pace_radii requires two positive attack-gap neighborhood radii')
        if not self.dilations or any(type(x) is not int or x < 1 for x in self.dilations):
            raise ContractError('dilations requires positive event-position offsets')
        if self.stage in ('readout', 'pilot'):
            cap = 900 if self.stage == 'readout' else 1800
            if self.max_updates > 1000 or self.arm_budget_seconds > cap:
                raise ContractError(f'{self.stage} caps are 1000 updates and {cap} seconds per arm')
        for name in ('prepared_dir', 'probe_prepared_dir', 'seed_run_dir', 'checkpoint', 'cache_dir', 'output_dir'):
            if not getattr(self, name):
                raise ContractError(f'{name} must be nonempty')
