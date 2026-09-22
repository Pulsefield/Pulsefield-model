"""Frozen input identities and bounded stage settings for unattended work."""
from dataclasses import dataclass, field
import math

from ..bounded_typed_continuation.train_config import TrainConfig
from ..bounded_typed_continuation.generate_config import GenerateConfig, checked_digest
from ..bounded_typed_continuation.smoke_config import SmokeResources
from ..scoped_style_modeling.dataset import ContractError


@dataclass
class AudioConfig:
    enabled: bool = True
    manifest_file: str = ''
    manifest_sha256: str = ''
    max_seconds: float = 57600.
    max_bytes: int = 80 * 1024**3
    chunk_frames: int = 6000


@dataclass
class TeacherConfig:
    enabled: bool = True
    base_plan_file: str = ''
    base_plan_sha256: str = ''
    evaluation_file: str = ''
    evaluation_sha256: str = ''
    milestones: list[int] = field(default_factory=lambda: [1000000, 5000000, 10000000, 20000000, 30000000])
    max_seconds: float = 158400.
    training: TrainConfig = field(default_factory=TrainConfig)


@dataclass
class StressConfig:
    enabled: bool = True
    manifest_file: str = ''
    manifest_sha256: str = ''
    max_seconds: float = 28800.
    generation: GenerateConfig = field(default_factory=GenerateConfig)


@dataclass
class VacationConfig:
    mode: str = 'preflight'
    output_dir: str = 'artifacts/vacation-training'
    max_seconds: float = 259200.
    preflight_max_seconds: float = 7200.
    max_bytes: int = 100 * 1024**3
    disk_reserve_bytes: int = 100 * 1024**3
    require_ac_power: bool = True
    check_thermal: bool = True
    footprint_limit_bytes: int = 12 * 1024**3
    resources: SmokeResources = field(default_factory=lambda: SmokeResources(
        driver_limit_bytes=8 * 1024**3, rss_limit_bytes=12 * 1024**3,
        min_available_bytes=4 * 1024**3, max_swap_growth_bytes=512 * 1024**2,
        checkpoint_max_bytes=1024**3, disk_reserve_bytes=100 * 1024**3,
        output_max_bytes=100 * 1024**3))
    audio: AudioConfig = field(default_factory=AudioConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    stress: StressConfig = field(default_factory=StressConfig)

    def validate(self):
        if self.mode not in ('preflight', 'run', 'resume', 'status') or not self.output_dir:
            raise ContractError('Vacation mode must be preflight, run, resume or status with an output_dir')
        if self.mode == 'status':
            return
        for name in ('require_ac_power', 'check_thermal'):
            if type(getattr(self, name)) is not bool:
                raise ContractError(f'{name} must be boolean')
        for owner in (self, self.audio, self.teacher, self.stress):
            if isinstance(owner.max_seconds, bool) or not math.isfinite(owner.max_seconds) or owner.max_seconds <= 0:
                raise ContractError('Stage and queue wall budgets must be finite positive seconds')
        if (isinstance(self.preflight_max_seconds, bool) or not math.isfinite(self.preflight_max_seconds) or
                not 0 < self.preflight_max_seconds <= 7200):
            raise ContractError('Preflight wall budget must be positive and at most two hours')
        for value in (self.max_bytes, self.disk_reserve_bytes, self.audio.max_bytes, self.audio.chunk_frames):
            if type(value) is not int or value <= 0:
                raise ContractError('Storage budgets and chunk_frames must be positive integers')
        if type(self.footprint_limit_bytes) is not int or not 0 < self.footprint_limit_bytes <= 12 * 1024**3:
            raise ContractError('Queue footprint limit must be positive and at most 12 GiB')
        if self.audio.chunk_frames > 6000 or self.audio.max_bytes > self.max_bytes:
            raise ContractError('Audio chunks must be at most 6000 frames and fit the queue storage budget')
        for name in ('audio', 'teacher', 'stress'):
            stage = getattr(self, name)
            if type(stage.enabled) is not bool:
                raise ContractError(f'{name}.enabled must be boolean')
            if not stage.enabled:
                continue
            pairs = (('base_plan_file', 'base_plan_sha256'), ('evaluation_file', 'evaluation_sha256')) if name == 'teacher' else (
                ('manifest_file', 'manifest_sha256'),)
            for path, digest in pairs:
                checked_digest(getattr(stage, digest), name + '.' + digest)
                if not getattr(stage, path) or not getattr(stage, digest):
                    raise ContractError(f'{name} requires pinned {path} and {digest}')
        if self.teacher.enabled:
            points = self.teacher.milestones
            if (not points or any(type(n) is not int or n <= 0 for n in points) or
                    any(a >= b for a, b in zip(points, points[1:])) or points[-1] > 30000000):
                raise ContractError('Teacher milestones must increase through at most 30M onset exposures')
            self.teacher.training.validate()
            if (self.teacher.training.resume_from or self.teacher.training.fork_from or self.teacher.training.recovery_pool or
                    self.teacher.training.stop_after_checkpoint is not None):
                raise ContractError('The queue owns teacher resume and milestones; training starts from a clean base')
            if not self.teacher.training.source_cache_dir:
                raise ContractError('Teacher training requires its admitted source_cache_dir')
        if self.stress.enabled:
            self.stress.generation.validate()
            if not self.stress.generation.checkpoint_file or not self.stress.generation.checkpoint_sha256:
                raise ContractError('Stress runs require a fixed checkpoint and SHA-256 selected before execution')
