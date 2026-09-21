"""Typed process configuration for the frozen paired assessment experiment."""
from dataclasses import dataclass, field
import math

from .dataset import ContractError


@dataclass
class ModelConfig:
    lane_dim: int = 16
    hand_hidden: int = 32
    attention_heads: int = 4
    feedforward_dim: int = 128
    row_dim: int = 64
    section_hidden: int = 32
    head_hidden: int = 64
    concept_dim: int = 16
    assessment_dim: int = 8
    mask_dim: int = 8
    selector_hidden: int = 64
    dropout: float = 0.1

    def validate(self):
        for name, value in vars(self).items():
            if name != "dropout" and (type(value) is not int or value < 1):
                raise ContractError(f"{name} must be a positive integer")
        if not 0 <= self.dropout < 1 or (2*self.hand_hidden) % self.attention_heads:
            raise ContractError("Invalid dropout or attention head dimensions")


@dataclass
class TrainConfig:
    prepared_dir: str = "artifacts/scoped-style-modeling/prepare-v1"
    output_dir: str = "artifacts/scoped-style-modeling/paired-17"
    device: str = "mps"
    seeds: list[int] = field(default_factory=lambda: [17])
    evidence_weight: float = 0.1
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    gradient_cap: float = 1.0
    batch_size: int = 16
    max_epochs: int = 30
    patience: int = 5
    max_updates: int | None = None
    arm_budget_seconds: float = 1200.0
    gradient_log_every: int = 100
    memory_log_every: int = 0
    cpu_threads: int = 1
    model: ModelConfig = field(default_factory=ModelConfig)

    def validate(self):
        self.model.validate()
        if not self.prepared_dir or not self.output_dir or self.device not in ("cpu", "mps", "cuda"):
            raise ContractError("Expected nonempty paths and explicit cpu, mps, or cuda device")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(type(s) is not int or s < 0 for s in self.seeds):
            raise ContractError("seeds must be distinct nonnegative integers")
        for name in ("batch_size", "max_epochs", "patience", "gradient_log_every", "cpu_threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ContractError(f"{name} must be a positive integer")
        if self.max_updates is not None and (type(self.max_updates) is not int or self.max_updates < 1):
            raise ContractError("max_updates must be null or a positive integer")
        if type(self.memory_log_every) is not int or self.memory_log_every < 0:
            raise ContractError("memory_log_every must be a nonnegative integer; zero disables sampling")
        for name in ("evidence_weight", "learning_rate", "gradient_cap", "arm_budget_seconds", "weight_decay"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name != "weight_decay" and value == 0):
                raise ContractError(f"{name} must be finite and {'nonnegative' if name == 'weight_decay' else 'positive'}")
