"""Typed startup settings for sequence training; no Hydra objects reach runtime."""
from dataclasses import dataclass, field
import math

from ..scoped_style_modeling.dataset import ContractError
from .config import BackboneConfig
from .objective import ObjectiveConfig
from .windows import WindowSamplingPolicy
from .storage import SourceCacheConfig
from .runtime import ResourceConfig, validate_envelope


@dataclass(frozen=True)
class TrainingConfig:
    effective_batch_size: int = 2
    microbatch_size: int = 2
    chunk_rows: int = 128
    learning_rate: float = 0.0001
    timing_learning_rate: float | None = None
    clock_readout_learning_rate: float | None = None
    weight_decay: float = 0.01
    max_grad_norm: float = 1.
    parallel_frontiers: bool = False
    reuse_prefixes: bool = False
    warmup_updates: int = 0

    def __post_init__(self) -> None:
        if type(self.parallel_frontiers) is not bool:
            raise ContractError("parallel_frontiers must be boolean")
        if type(self.reuse_prefixes) is not bool:
            raise ContractError("reuse_prefixes must be boolean")
        if type(self.warmup_updates) is not int or self.warmup_updates < 0:
            raise ContractError("warmup_updates must be a nonnegative integer")
        for name in ("effective_batch_size", "microbatch_size", "chunk_rows"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ContractError(f"{name} must be a positive integer")
        if self.microbatch_size > self.effective_batch_size:
            raise ContractError("microbatch_size cannot exceed effective_batch_size")
        if self.chunk_rows > 128:
            raise ContractError("Training chunks support at most 128 rows")
        for name in ("learning_rate", "weight_decay", "max_grad_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ContractError(f"{name} must be finite and nonnegative")
        if self.learning_rate == 0 or self.max_grad_norm == 0:
            raise ContractError("learning_rate and max_grad_norm must be positive")
        for name in ('timing_learning_rate', 'clock_readout_learning_rate'):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value <= 0):
                raise ContractError(f'{name} must be finite and positive when supplied')


@dataclass
class TrainExperimentConfig:
    source_dir: str = "artifacts/scoped-style-modeling/sources"
    split_manifest: str = "artifacts/scoped-style-modeling/prepare-v1/split-manifest.json"
    split_sha256: str = ""
    source_sha256: list[str] = field(default_factory=list)
    catalog_path: str = ""
    catalog_sha256: str = ""
    initial_weights: str = ""
    initial_weights_sha256: str = ""
    output_dir: str = "artifacts/oracle-time-continuation/m3-train"
    cache_dir: str = "artifacts/oracle-time-continuation/source-cache-v1"
    resume: bool = False
    cache: SourceCacheConfig = field(default_factory=SourceCacheConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    device: str = "cpu"
    model_seed: int = 17
    cpu_threads: int = 1
    updates: int = 1
    checkpoint_every_updates: int = 1
    model: BackboneConfig = field(default_factory=BackboneConfig)
    windows: WindowSamplingPolicy = field(default_factory=WindowSamplingPolicy)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        validate_envelope(self.model, self.training.microbatch_size)
        if self.device not in ("cpu", "mps", "cuda"):
            raise ContractError("device must be cpu, mps or cuda")
        for name in ("cpu_threads", "updates", "checkpoint_every_updates"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ContractError(f"{name} must be a positive integer")
        if type(self.model_seed) is not int or not 0 <= self.model_seed < 2 ** 63:
            raise ContractError("model_seed must be an integer in [0, 2**63)")
        if self.training.chunk_rows > self.model.max_chunk:
            raise ContractError("training.chunk_rows cannot exceed model.max_chunk")
        if self.windows.windows_per_chart > self.training.effective_batch_size:
            raise ContractError("windows_per_chart cannot exceed effective_batch_size")
        if self.training.timing_learning_rate is not None and not self.model.time_lookahead_rows:
            raise ContractError('timing_learning_rate requires model.time_lookahead_rows>0')
        if self.training.clock_readout_learning_rate is not None and not self.model.clock_readout_hidden:
            raise ContractError('clock_readout_learning_rate requires model.clock_readout_hidden>0')
        if len(set(self.source_sha256)) != len(self.source_sha256):
            raise ContractError("source_sha256 must contain distinct source identities")
        if bool(self.catalog_path) != bool(self.catalog_sha256):
            raise ContractError("catalog_path and catalog_sha256 must be supplied together")
        if bool(self.initial_weights) != bool(self.initial_weights_sha256):
            raise ContractError("initial_weights and initial_weights_sha256 must be supplied together")
        for value in [v for v in (self.split_sha256, self.catalog_sha256, self.initial_weights_sha256) if v] + self.source_sha256:
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ContractError("Source and split identities must be lowercase SHA-256 digests")
