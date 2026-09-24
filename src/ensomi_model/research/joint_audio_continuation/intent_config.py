"""Bounded matched continuation fitting with an optional four-state latent."""
from dataclasses import dataclass
import math


@dataclass
class IntentTrainConfig:
    root: str = 'artifacts/joint-audio/20260924-alias-restored-v1'
    manifest_sha256: str = '4ad9abfd0ae7798e0a85b96a5dbecdaefd2a81f78bf181438407442b964118e1'
    initial_checkpoint_file: str = 'artifacts/joint-audio/20260924-expanded-v1/delivery/context-r1-playtest-v2/model.pt'
    initial_checkpoint_sha256: str = '1e86b79dd1144bca282a01fb798dc304094357bff5b80deeb099982749f5d48c'
    protocol_name: str = 'intent-comparison-v1'
    run_name: str = 'main-k1-v1'
    states: int = 1
    seed: int = 230941
    sample_seed: int = 230942
    validation_seed: int = 230943
    plan_updates: int = 1200
    updates: int = 1200
    songs_per_update: int = 2
    intervals_per_song: int = 2
    interval_ms: int = 8000
    validation_every: int = 300
    validation_songs: int = 0
    learning_rate: float = .0001
    inherited_learning_rate: float = .00001
    intent_learning_rate: float = .0003
    weight_decay: float = .01
    max_grad_norm: float = 1.
    max_seconds: float = 7200.
    device: str = 'mps'
    cpu_threads: int = 1

    def validate(self):
        for name in ('run_name', 'protocol_name'):
            value = getattr(self, name)
            if not value or value in ('.', '..') or '/' in value or '\\' in value:
                raise ValueError(f'{name} must be one directory component')
        for name in ('manifest_sha256', 'initial_checkpoint_sha256'):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError(f'{name} must be a lowercase SHA-256')
        if not self.initial_checkpoint_file or type(self.states) is not int or self.states not in (1, 4):
            raise ValueError('Intent comparison requires a pinned initial checkpoint and states=1 or 4')
        for name in ('plan_updates', 'updates', 'songs_per_update', 'intervals_per_song', 'interval_ms',
                     'validation_every', 'cpu_threads'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('seed', 'sample_seed', 'validation_seed', 'validation_songs'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.updates > self.plan_updates or self.interval_ms > 16000 or self.cpu_threads > 10:
            raise ValueError('Intent run exceeds its frozen plan or resource envelope')
        if self.device not in ('cpu', 'mps'):
            raise ValueError('Intent study uses CPU or explicit MPS')
        for name in ('learning_rate', 'inherited_learning_rate', 'intent_learning_rate', 'max_grad_norm', 'max_seconds'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('weight_decay must be finite and nonnegative')
