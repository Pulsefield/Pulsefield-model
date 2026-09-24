"""Torch-free settings for source-chart-free planned audio inference."""
from dataclasses import dataclass
import math


@dataclass
class AudioInferenceConfig:
    audio_file: str = ''
    checkpoint_file: str = ''
    checkpoint_sha256: str = ''
    output_dir: str = ''
    device: str = 'cpu'
    cpu_threads: int = 1
    seed: int = 17
    chunk_ms: int = 500
    head_chunk_ms: int = 500
    max_rows: int = 30000
    max_seconds: float = 90.
    startup_coverage_ms: int = 8000
    startup_min_rows: int = 30
    correct_short_attacks: bool = False
    arrangement_profile: int | None = None

    def validate(self):
        if any(not isinstance(getattr(self, name), str) or not getattr(self, name).strip()
               for name in ('audio_file', 'checkpoint_file', 'output_dir')):
            raise ValueError('Audio inference requires audio_file, checkpoint_file and a fresh output_dir')
        if (len(self.checkpoint_sha256) != 64 or
                any(c not in '0123456789abcdef' for c in self.checkpoint_sha256)):
            raise ValueError('Audio inference requires a lowercase checkpoint SHA-256')
        if self.device not in ('cpu', 'mps'):
            raise ValueError('Planned audio inference supports cpu or mps')
        for name in ('cpu_threads', 'chunk_ms', 'head_chunk_ms', 'max_rows', 'startup_coverage_ms'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if type(self.startup_min_rows) is not int or self.startup_min_rows < 0:
            raise ValueError('startup_min_rows must be a nonnegative integer')
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError('seed must be an integer in [0, 2**63)')
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError('max_seconds must be finite and positive')
        if type(self.correct_short_attacks) is not bool:
            raise ValueError('correct_short_attacks must be boolean')
        if self.arrangement_profile is not None and (type(self.arrangement_profile) is not int or self.arrangement_profile < 0):
            raise ValueError('arrangement_profile must be null or a nonnegative index')
