"""Typed startup settings for portable native generation and condition preparation."""
from dataclasses import dataclass, field
import math

from ..scoped_style_modeling.dataset import ContractError
from .smoke_config import SmokeResources


def checked_digest(value, name):
    if value and (not isinstance(value, str) or len(value) != 64 or
                  any(c not in '0123456789abcdef' for c in value)):
        raise ContractError(f'{name} must be a full lowercase SHA-256 digest')


@dataclass
class PrepareConditionConfig:
    source_file: str = ''
    source_sha256: str = ''
    output_file: str = ''
    arm: str = 'r1'
    seed_notes: int = 30

    def validate(self):
        checked_digest(self.source_sha256, 'source_sha256')
        if self.arm not in ('r0', 'r1', 'o1'):
            raise ContractError('Condition arm must be r0, r1 or o1')
        if type(self.seed_notes) is not int or self.seed_notes <= 0:
            raise ContractError('seed_notes must be a positive integer')


@dataclass
class GenerateConfig:
    checkpoint_file: str = ''
    checkpoint_sha256: str = ''
    condition_file: str = ''
    condition_sha256: str = ''
    presentation_source: str = ''
    presentation_sha256: str = ''
    output_dir: str = 'artifacts/bounded-typed-continuation/generated'
    resume_from: str | None = None
    resume_sha256: str = ''
    device: str = 'cpu'
    cpu_threads: int = 1
    seed: int = 17
    candidate_budget: int = 8192
    score_endpoints: bool = False
    checkpoint_every_candidates: int = 512
    stop_after_candidate: int | None = None
    max_seconds: float = 3600.
    footprint_limit_bytes: int = 6 * 1024 ** 3
    resources: SmokeResources = field(default_factory=SmokeResources)

    def validate(self):
        for name in ('checkpoint_sha256', 'condition_sha256', 'presentation_sha256', 'resume_sha256'):
            checked_digest(getattr(self, name), name)
        if bool(self.resume_from) != bool(self.resume_sha256):
            raise ContractError('resume_from and resume_sha256 must be supplied together')
        if bool(self.presentation_source) != bool(self.presentation_sha256):
            raise ContractError('presentation_source and presentation_sha256 must be supplied together')
        if self.device not in ('cpu', 'mps', 'cuda'):
            raise ContractError('Generation device must be cpu, mps or cuda')
        for name in ('cpu_threads', 'candidate_budget', 'checkpoint_every_candidates', 'footprint_limit_bytes'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ContractError(f'{name} must be a positive integer')
        if (self.candidate_budget > 32768 or self.checkpoint_every_candidates > 8192 or
                self.footprint_limit_bytes > 8 * 1024 ** 3):
            raise ContractError('Generation exceeds its candidate, checkpoint or memory envelope')
        if type(self.seed) is not int or not 0 <= self.seed < 2 ** 63:
            raise ContractError('seed must be a nonnegative integer below 2**63')
        if type(self.score_endpoints) is not bool:
            raise ContractError('score_endpoints must be boolean')
        if self.stop_after_candidate is not None and (type(self.stop_after_candidate) is not int or self.stop_after_candidate <= 0):
            raise ContractError('stop_after_candidate must be null or a positive candidate cursor')
        if isinstance(self.max_seconds, bool) or not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ContractError('max_seconds must be finite and positive')
