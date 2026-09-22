"""Typed settings for a bounded learning check; no Hydra objects reach runtime."""
from dataclasses import asdict, dataclass, field
import math

from ..oracle_time_continuation.runtime import ResourceConfig
from ..scoped_style_modeling.dataset import ContractError
from .contract import Arm
from .model import ModelConfig


@dataclass(frozen=True)
class SmokeResources:
    driver_limit_bytes: int = 6 * 1024 ** 3
    rss_limit_bytes: int = 6 * 1024 ** 3
    allocator_ceiling_bytes: int = 8 * 1024 ** 3
    min_available_bytes: int = 2 * 1024 ** 3
    max_swap_growth_bytes: int = 128 * 1024 ** 2
    checkpoint_max_bytes: int = 128 * 1024 ** 2
    disk_reserve_bytes: int = 1024 ** 3
    output_max_bytes: int = 512 * 1024 ** 2
    release_mps_cache_at_boundaries: bool = True

    def __post_init__(self):
        ResourceConfig(**asdict(self))


@dataclass
class SmokeConfig:
    interval_manifest: str = ''
    interval_sha256: str = ''
    catalog_path: str = ''
    catalog_sha256: str = ''
    split_manifest: str = ''
    split_sha256: str = ''
    source_cache_dir: str = ''
    output_dir: str = 'artifacts/bounded-typed-continuation/smoke'
    device: str = 'mps'
    model_seed: int = 171
    shuffle_seed: int = 271
    cpu_threads: int = 1
    updates: int = 128
    batch_size: int = 2
    report_every: int = 8
    candidate_budget: int = 8192
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    max_grad_norm: float = 1.
    warmup_updates: int = 8
    max_seconds: float = 1800.
    model: ModelConfig = field(default_factory=lambda: ModelConfig(Arm.O1))
    resources: SmokeResources = field(default_factory=SmokeResources)

    def validate(self):
        if self.device not in ('cpu', 'mps', 'cuda'):
            raise ContractError('Smoke device must be cpu, mps or cuda')
        for name in ('cpu_threads', 'updates', 'batch_size', 'report_every', 'candidate_budget'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ContractError(f'{name} must be a positive integer')
        for name in ('model_seed', 'shuffle_seed', 'warmup_updates'):
            if type(getattr(self, name)) is not int or not 0 <= getattr(self, name) < 2 ** 63:
                raise ContractError(f'{name} must be a nonnegative integer below 2**63')
        for name in ('learning_rate', 'weight_decay', 'max_grad_norm', 'max_seconds'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0 or (name != 'weight_decay' and value == 0):
                raise ContractError(f'{name} must be finite and positive, or nonnegative for weight decay')
        if (self.batch_size > 4 or self.model.hidden > 128 or self.model.levels > 8 or
                self.model.expansion > 4 or self.model.coupling_rank > 16 or self.candidate_budget > 32768):
            raise ContractError('Learning check exceeds the supported small-model/batch/candidate envelope')
        for name in ('interval_sha256', 'catalog_sha256', 'split_sha256'):
            value = getattr(self, name)
            if value and (len(value) != 64 or any(c not in '0123456789abcdef' for c in value)):
                raise ContractError(f'{name} must be a lowercase SHA-256 digest')
