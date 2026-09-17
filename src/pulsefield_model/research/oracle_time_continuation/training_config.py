"""Typed startup settings for sequence training; no Hydra objects reach runtime."""
from dataclasses import dataclass, field
import math

from ..scoped_style_modeling.dataset import ContractError
from .config import BackboneConfig
from .objective import ObjectiveConfig
from .windows import WindowSamplingPolicy


@dataclass(frozen=True)
class TrainingConfig:
    effective_batch_size: int = 2
    microbatch_size: int = 2
    chunk_rows: int = 128
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    max_grad_norm: float = 1.

    def __post_init__(self) -> None:
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


@dataclass
class TrainExperimentConfig:
    source_dir: str = "artifacts/scoped-style-modeling/sources"
    split_manifest: str = "artifacts/scoped-style-modeling/prepare-v1/split-manifest.json"
    split_sha256: str = ""
    source_sha256: list[str] = field(default_factory=list)
    output_dir: str = "artifacts/oracle-time-continuation/m2-train"
    device: str = "mps"
    model_seed: int = 17
    cpu_threads: int = 1
    updates: int = 1
    model: BackboneConfig = field(default_factory=BackboneConfig)
    windows: WindowSamplingPolicy = field(default_factory=WindowSamplingPolicy)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        if self.device not in ("cpu", "mps", "cuda"):
            raise ContractError("device must be cpu, mps or cuda")
        for name in ("cpu_threads", "updates"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ContractError(f"{name} must be a positive integer")
        if type(self.model_seed) is not int or not 0 <= self.model_seed < 2 ** 63:
            raise ContractError("model_seed must be an integer in [0, 2**63)")
        if self.training.chunk_rows > self.model.max_chunk:
            raise ContractError("training.chunk_rows cannot exceed model.max_chunk")
        if len(set(self.source_sha256)) != len(self.source_sha256):
            raise ContractError("source_sha256 must contain distinct source identities")
        for value in ([self.split_sha256] if self.split_sha256 else []) + self.source_sha256:
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ContractError("Source and split identities must be lowercase SHA-256 digests")
