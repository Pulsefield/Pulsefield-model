"""Typed inputs for bounded skeleton experiments."""
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    mode: str = 'prepare'
    root: str = 'artifacts/audio-skeleton/20260923-v1'
    catalog_file: str = 'artifacts/oracle-time-review/20260915-adfb1ee/catalog.json'
    catalog_sha256: str = 'e31b7e8f4daa044503ef2b8411bc41727ba371eec804c9462a4608be8112ad28'
    catalog_root: str = '.'
    source_cache_dir: str = 'artifacts/oracle-time-continuation/full-cache-v1'
    train_per_stratum: int = 8
    validation_per_stratum: int = 2
    selection_seed: int = 20260923
    checkpoint_file: str = 'artifacts/audio-skeleton/20260923-v1/inputs/r1.pt'
    checkpoint_sha256: str = '4b3ec1561e33d0ebe2756cfe13571ec414fd5bb470b430f0c578545863115f70'
    workers: int = 4
    device: str = 'mps'
    max_seconds: float = 2700.
    beat_model: str = 'final0'
    run_name: str = 'pilot'
    use_beat_features: bool = False
    updates: int = 1200
    batch_size: int = 12
    learning_rate: float = 0.0003
    seed: int = 172
    overfit_charts: int = 0
    validation_every: int = 200
    resume_from: str = ''

    def validate(self):
        if self.mode not in ('prepare', 'sensitivity', 'cache', 'train', 'evaluate'):
            raise ValueError('Unknown audio skeleton experiment mode')
        if self.device not in ('mps', 'cpu') or self.workers not in range(1, 9):
            raise ValueError('Use cpu/mps and one to eight workers')
        if any(getattr(self, name) <= 0 for name in (
                'train_per_stratum', 'validation_per_stratum', 'max_seconds',
                'updates', 'batch_size', 'learning_rate', 'validation_every')):
            raise ValueError('Sampling, training and time bounds must be positive')
        if '/' in self.run_name or self.run_name in ('', '.', '..'):
            raise ValueError('run_name must be a single nonempty directory name')
        if self.overfit_charts < 0:
            raise ValueError('overfit_charts cannot be negative')
