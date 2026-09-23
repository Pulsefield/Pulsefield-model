"""Bounded experiments on the canonical music Mel and timed-row distribution."""
from dataclasses import dataclass


@dataclass
class JointConfig:
    mode: str = 'prepare'
    root: str = 'artifacts/joint-audio/20260923-v1'
    base_manifest_file: str = 'artifacts/audio-skeleton/20260923-v1/manifest.json'
    catalog_file: str = 'artifacts/oracle-time-review/20260915-adfb1ee/catalog.json'
    catalog_sha256: str = 'e31b7e8f4daa044503ef2b8411bc41727ba371eec804c9462a4608be8112ad28'
    catalog_root: str = '.'
    source_cache_dir: str = 'artifacts/oracle-time-continuation/full-cache-v1'
    max_train_alternatives: int = 2
    r1_checkpoint_file: str = 'artifacts/audio-skeleton/20260923-v1/inputs/r1.pt'
    r1_checkpoint_sha256: str = '4b3ec1561e33d0ebe2756cfe13571ec414fd5bb470b430f0c578545863115f70'
    run_name: str = 'smoke-v1'
    device: str = 'mps'
    seed: int = 230923
    updates: int = 600
    batch_size: int = 8
    learning_rate: float = 0.0003
    inherited_learning_rate: float = 0.00003
    weight_decay: float = 0.01
    max_grad_norm: float = 1.
    max_seconds: float = 900.
    train_groups: int = 6
    fixed_train_queries: int = 0
    full_wait_supervision: bool = False
    coverage_pass: bool = False
    validation_every: int = 100
    history_rows: int = 511
    timing_horizon_ms: int = 4000
    audio_width: int = 96
    audio_levels: int = 6
    generation_split: str = 'validation'
    generation_cases: int = 6
    generation_seed: int = 17
    generation_max_rows: int = 30000
    checkpoint_file: str = ''
    checkpoint_sha256: str = ''
    cpu_threads: int = 1

    def validate(self):
        if self.mode not in ('prepare', 'train', 'generate'):
            raise ValueError('Joint mode must be prepare, train or generate')
        if self.device not in ('cpu', 'mps') or self.generation_split not in ('train', 'validation'):
            raise ValueError('Use CPU/MPS and TRAIN/validation only')
        if '/' in self.run_name or self.run_name in ('', '.', '..'):
            raise ValueError('run_name must be a single directory name')
        for name in ('updates', 'batch_size', 'validation_every', 'history_rows', 'timing_horizon_ms',
                     'audio_width', 'audio_levels', 'generation_cases', 'generation_max_rows', 'cpu_threads'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.history_rows != 511:
            raise ValueError('The inherited eight-level TCN requires the declared 511-row history envelope')
        if self.audio_levels > 8 or self.timing_horizon_ms > 16000 or self.cpu_threads > 10:
            raise ValueError('Configuration exceeds the bounded joint experiment envelope')
        if self.train_groups < 0 or self.max_train_alternatives < 0 or self.fixed_train_queries < 0:
            raise ValueError('Training group and alternative counts cannot be negative')
        if type(self.full_wait_supervision) is not bool or type(self.coverage_pass) is not bool:
            raise ValueError('Waiting-interval and coverage settings must be boolean')
        if self.full_wait_supervision and self.fixed_train_queries:
            raise ValueError('Full waiting-interval supervision requires random queries')
        if self.coverage_pass and (not self.full_wait_supervision or self.fixed_train_queries):
            raise ValueError('Coverage pass requires full waiting-interval supervision and random queries')
        if any(getattr(self, n) <= 0 for n in ('learning_rate', 'inherited_learning_rate', 'max_seconds', 'max_grad_norm')):
            raise ValueError('Learning rates, gradient and time limits must be positive')
        if self.mode == 'generate' and (not self.checkpoint_file or len(self.checkpoint_sha256) != 64):
            raise ValueError('Generation requires a pinned joint checkpoint')
