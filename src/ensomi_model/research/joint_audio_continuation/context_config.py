"""Typed process settings for the matched audio/history path study."""
from dataclasses import dataclass
import math


@dataclass
class ContextTrainConfig:
    root: str = 'artifacts/joint-audio/20260924-expanded-v1'
    manifest_sha256: str = 'cc60dd39920c626f3be5498ab8ba7aa342dd0956c0a548cc393d85a1b03f5173'
    normalization_file: str = 'artifacts/joint-audio/20260923-v1/normalization.json'
    normalization_sha256: str = '9cf461a0e825f974f0a80a364123c7afedf1683af76e60fe09b0fbe51c2c8287'
    r1_checkpoint_file: str = 'artifacts/audio-skeleton/20260923-v1/inputs/r1.pt'
    r1_checkpoint_sha256: str = '4b3ec1561e33d0ebe2756cfe13571ec414fd5bb470b430f0c578545863115f70'
    protocol_name: str = 'context-paths-v1'
    run_name: str = 'paths-local-fused-v1'
    global_audio: bool = False
    bounded_timing: bool = False
    seed: int = 230928
    sample_seed: int = 230929
    validation_seed: int = 230930
    plan_updates: int = 1200
    updates: int = 1200
    songs_per_update: int = 2
    intervals_per_song: int = 2
    interval_ms: int = 8000
    validation_every: int = 200
    validation_songs: int = 0
    learning_rate: float = .0003
    inherited_learning_rate: float = .00003
    weight_decay: float = .01
    max_grad_norm: float = 1.
    max_seconds: float = 7200.
    device: str = 'mps'
    cpu_threads: int = 1

    def validate(self):
        for name in ('run_name', 'protocol_name'):
            value = getattr(self, name)
            if not value or value in ('.', '..') or '/' in value or '\\' in value:
                raise ValueError(f'{name} must be a single nonempty directory component')
        for name in ('manifest_sha256', 'normalization_sha256', 'r1_checkpoint_sha256'):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError(f'{name} must be a lowercase SHA-256 digest')
        for name in ('plan_updates', 'updates', 'songs_per_update', 'intervals_per_song', 'interval_ms',
                     'validation_every', 'cpu_threads'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('seed', 'sample_seed', 'validation_seed', 'validation_songs'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if self.updates > self.plan_updates or self.interval_ms > 16000 or self.cpu_threads > 10:
            raise ValueError('Run exceeds its frozen plan or bounded interval/CPU envelope')
        if self.device not in ('cpu', 'mps'):
            raise ValueError('Context study supports CPU or explicit MPS execution')
        if type(self.global_audio) is not bool or type(self.bounded_timing) is not bool:
            raise ValueError('Architecture switches must be boolean')
        for name in ('learning_rate', 'inherited_learning_rate', 'max_grad_norm', 'max_seconds'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('weight_decay must be finite and nonnegative')
