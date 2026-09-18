"""Typed corpus-training settings; sampling is owned by the pinned plan."""
from dataclasses import dataclass, field
import math

from ..scoped_style_modeling.dataset import ContractError
from .contract import Arm
from .model import ModelConfig
from .smoke_config import SmokeResources


@dataclass
class TrainConfig:
    plan_file: str = ''
    plan_sha256: str = ''
    source_cache_dir: str = ''
    output_dir: str = 'artifacts/bounded-typed-continuation/corpus'
    resume_from: str | None = None
    stop_after_checkpoint: int | None = None
    device: str = 'mps'
    model_seed: int = 171
    cpu_threads: int = 1
    batch_size: int = 4
    microbatch_size: int = 2
    report_every: int = 32
    candidate_budget: int = 8192
    learning_rate: float = .0003
    weight_decay: float = .01
    max_grad_norm: float = 1.
    warmup_onsets: int = 32768
    max_seconds: float = 14400.
    cache_max_sources: int = 64
    cache_max_bytes: int = 256 * 1024 ** 2
    model: ModelConfig = field(default_factory=lambda: ModelConfig(Arm.O1))
    resources: SmokeResources = field(default_factory=SmokeResources)

    def validate(self):
        if self.device not in ('cpu', 'mps', 'cuda'):
            raise ContractError('Training device must be cpu, mps or cuda')
        for name in ('cpu_threads', 'batch_size', 'microbatch_size', 'report_every', 'candidate_budget',
                     'cache_max_sources', 'cache_max_bytes'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ContractError(f'{name} must be a positive integer')
        for name in ('model_seed', 'warmup_onsets'):
            if type(getattr(self, name)) is not int or not 0 <= getattr(self, name) < 2 ** 63:
                raise ContractError(f'{name} must be a nonnegative integer below 2**63')
        if self.stop_after_checkpoint is not None and (type(self.stop_after_checkpoint) is not int or self.stop_after_checkpoint <= 0):
            raise ContractError('stop_after_checkpoint must be null or a positive onset checkpoint')
        for name in ('learning_rate', 'weight_decay', 'max_grad_norm', 'max_seconds'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0 or (name != 'weight_decay' and value == 0):
                raise ContractError(f'{name} must be finite and positive, or nonnegative for weight decay')
        if (self.microbatch_size > min(self.batch_size, 2) or self.batch_size > 8 or
                self.model.hidden > 128 or self.model.levels > 8 or self.model.expansion > 4 or
                self.model.coupling_rank > 16 or self.candidate_budget > 32768 or
                self.cache_max_sources > 128 or self.cache_max_bytes > 512 * 1024 ** 2):
            raise ContractError('Corpus training exceeds its bounded model, batch, candidate or cache envelope')
        if self.plan_sha256 and (len(self.plan_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.plan_sha256)):
            raise ContractError('plan_sha256 must be a lowercase SHA-256 digest')
