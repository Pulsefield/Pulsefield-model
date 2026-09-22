"""Typed settings for the six-stage R1 reconstruction."""
from dataclasses import dataclass, field, replace
import math

from ..bounded_typed_continuation.contract import Arm
from ..bounded_typed_continuation.generate_config import checked_digest
from ..bounded_typed_continuation.model import ModelConfig
from ..bounded_typed_continuation.smoke_config import SmokeResources
from ..bounded_typed_continuation.train_config import TrainConfig
from ..scoped_style_modeling.dataset import ContractError

STAGES = ('base', 'seed', 'memory', 'routing', 'release', 'response')


@dataclass
class RestoreConfig:
    mode: str = 'preflight'
    output_dir: str = ''
    base_plan_file: str = ''
    base_plan_sha256: str = ''
    catalog_file: str = ''
    catalog_root: str = ''
    evaluation_file: str = ''
    evaluation_sha256: str = ''
    milestones: dict[str, list[int]] = field(default_factory=lambda: dict(
        base=[250000, 1000000, 2000000, 4000000, 4125000, 4500000],
        seed=[4625000, 5000000], memory=[5125000, 6000000], routing=[6250000],
        release=[6375000, 6500000], response=[6625000, 6750000]))
    max_seconds: float = 43200.
    preflight_max_seconds: float = 1800.
    harvest_max_seconds: float = 1800.
    release_harvest_max_seconds: float = 3600.
    evaluation_max_seconds: float = 1800.
    max_bytes: int = 20 * 1024**3
    disk_reserve_bytes: int = 100 * 1024**3
    require_ac_power: bool = True
    check_thermal: bool = True
    footprint_limit_bytes: int = 6 * 1024**3
    resources: SmokeResources = field(default_factory=lambda: SmokeResources(
        disk_reserve_bytes=100 * 1024**3, output_max_bytes=20 * 1024**3))
    training: TrainConfig = field(default_factory=lambda: TrainConfig(
        model=ModelConfig(Arm.R1), model_seed=172))

    def validate(self):
        if self.mode not in ('preflight', 'run', 'resume', 'status') or not self.output_dir:
            raise ContractError('R1 restoration needs preflight/run/resume/status and an output_dir')
        if self.mode == 'status':
            return
        for key in ('base_plan', 'evaluation'):
            checked_digest(getattr(self, key + '_sha256'), key)
            if not getattr(self, key + '_file') or not getattr(self, key + '_sha256'):
                raise ContractError('R1 restoration requires pinned plan and evaluation inputs')
        if not self.catalog_file or not self.catalog_root or not self.training.source_cache_dir:
            raise ContractError('R1 restoration requires the catalog, original source root and row cache')
        self.training.validate()
        t = self.training
        if t.plan_file or t.plan_sha256 or t.output_dir != TrainConfig().output_dir:
            raise ContractError('The restoration queue owns training plan and segment output paths')
        if (t.device != 'cpu' or t.cpu_threads != 1 or t.execution_profile != 'small' or t.model.arm != Arm.R1 or
                any(getattr(t.model, k) != 'none' for k in ('seed_context', 'long_memory', 'head_routing',
                                                         'release_routing', 'row_consequence', 'endpoint_availability')) or
                t.trainable != 'all' or t.recovery_pool or t.source_kl_weight or t.resume_from or t.fork_from or
                t.stop_after_checkpoint is not None):
            raise ContractError('Restoration starts from a plain CPU1 R1; the queue owns all extensions and recovery')
        if set(self.milestones) != set(STAGES) or any(not self.milestones[s] for s in STAGES):
            raise ContractError('Restoration needs all six nonempty stage milestone lists')
        points = [n for stage in STAGES for n in self.milestones[stage]]
        if any(type(n) is not int or n <= 0 for n in points) or any(a >= b for a, b in zip(points, points[1:])):
            raise ContractError('Restoration milestones must be strictly increasing across stages')
        for key in ('max_seconds', 'preflight_max_seconds', 'harvest_max_seconds',
                    'release_harvest_max_seconds', 'evaluation_max_seconds'):
            n = getattr(self, key)
            if isinstance(n, bool) or not math.isfinite(n) or n <= 0:
                raise ContractError('Restoration wall budgets must be finite and positive')
        for key in ('require_ac_power', 'check_thermal'):
            if type(getattr(self, key)) is not bool:
                raise ContractError('Restoration power checks must be boolean')
        if (type(self.max_bytes) is not int or self.max_bytes <= 0 or
                type(self.disk_reserve_bytes) is not int or self.disk_reserve_bytes <= 0 or
                not 0 < self.footprint_limit_bytes <= 6 * 1024**3):
            raise ContractError('Restoration requires positive storage budgets and at most 6 GiB footprint')


def stage_training(config, stage):
    """Add only the module belonging to the next historical continuation."""
    index = STAGES.index(stage)
    model = replace(config.training.model, seed_context='observed' if index >= 1 else 'none',
        long_memory='landmarks' if index >= 2 else 'none', head_routing='residual' if index >= 3 else 'none',
        release_routing='residual' if index >= 4 else 'none', row_consequence='frontier2' if index >= 5 else 'none')
    return replace(config.training, model=model,
        trainable={3: 'routing', 4: 'release', 5: 'consequence'}.get(index, 'all'),
        source_kl_weight=1. if index == 5 else 0.)
