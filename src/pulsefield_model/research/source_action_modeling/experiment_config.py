"""Validated bounds for the local/relation composition experiment."""
from dataclasses import dataclass, field
import math

from ..scoped_style_modeling.dataset import ContractError
from .model import ModelConfig


@dataclass
class CompositionExperimentConfig:
    output_dir: str = "artifacts/source-action-composition/overnight"
    prepared_dir: str = "artifacts/scoped-style-modeling/prepare-v1"
    dataset_dir: str = "artifacts/scoped-style-modeling/dataset-b22a7a4"
    index_path: str = "artifacts/indexes/beatmap_index_4k.parquet"
    dataset_root: str = "dataset"
    source_cache: str = "artifacts/scoped-style-modeling/sources"
    warm_start_dir: str | None = None
    device: str = "mps"
    seeds: list[int] = field(default_factory=lambda: [17, 29])
    train_group_limit: int | None = None
    source_cache_charts: int = 16
    max_source_rows: int = 256
    validation_groups: int = 64
    near_radius: int = 8
    blocks_per_update: int = 8
    max_updates: int = 20000
    min_updates: int = 1000
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    gradient_cap: float = 1.0
    training_seconds_per_seed: float = 14400.0
    finalization_seconds_per_seed: float = 5400.0
    max_seconds: float = 39600.0
    max_driver_bytes: int = 8589934592
    max_rss_bytes: int = 12884901888
    max_output_bytes: int = 4294967296
    checkpoint_every: int = 500
    checkpoint_seconds: float = 300.0
    log_every: int = 25
    evaluation_batch_size: int = 4
    bootstrap_samples: int = 1000
    path_pairs: int = 24
    readout_steps: int = 600
    readout_batch_size: int = 4
    readout_learning_rate: float = 0.001
    structural_probe_ridge: float = 0.1
    cpu_threads: int = 1
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        feedforward_dim=256, decoder_hidden=64, action_dim=16, interaction_dim=16))

    def validate(self):
        self.model.__post_init__()
        if self.warm_start_dir is not None and not self.warm_start_dir.strip():
            raise ContractError("warm_start_dir must be a nonempty checkpoint directory or null")
        if self.model.hand_hidden != 32:
            raise ContractError("Composition banks currently require hand_hidden=32 (64 channels)")
        if self.model.lane_dim != 16:
            raise ContractError("lane_dim is fixed at 16 for the shared baseline initializer")
        if self.model.dropout != 0:
            raise ContractError("Matched operator-order comparison requires dropout=0")
        if self.device not in ("cpu", "mps"):
            raise ContractError("This bounded runner supports explicit cpu or mps")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(type(s) is not int or s < 0 for s in self.seeds):
            raise ContractError("seeds must be distinct nonnegative integers")
        for name, value in vars(self).items():
            if name in ("output_dir", "prepared_dir", "dataset_dir", "index_path", "dataset_root", "source_cache") and not value.strip():
                raise ContractError(f"{name} must be nonempty")
            if type(value) is int and value < 1:
                raise ContractError(f"{name} must be positive")
            if type(value) is float and (not math.isfinite(value) or value < 0 or (name != "weight_decay" and value == 0)):
                raise ContractError(f"{name} must be finite and positive")
        if self.min_updates > self.max_updates or self.max_source_rows < 128:
            raise ContractError("Require min_updates <= max_updates and at least 128 context rows for 64-row targets")
        if len(self.seeds) * self.finalization_seconds_per_seed >= self.max_seconds:
            raise ContractError("Total time must leave room for population and training")
