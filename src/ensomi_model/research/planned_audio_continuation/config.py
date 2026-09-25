"""Typed process settings for the planned head/release research prototype."""
from dataclasses import dataclass
import math

from ..joint_audio_continuation.context_config import ContextTrainConfig


@dataclass
class PlannedTrainConfig(ContextTrainConfig):
    root: str = 'artifacts/joint-audio/20260924-alias-restored-v1'
    manifest_sha256: str = '4ad9abfd0ae7798e0a85b96a5dbecdaefd2a81f78bf181438407442b964118e1'
    protocol_name: str = 'r1-lineage-comparison-v1'
    run_name: str = 'planned-main-v1'
    global_audio: bool = True
    bounded_timing: bool = False
    seed: int = 230941
    sample_seed: int = 230942
    validation_seed: int = 230943
    validation_every: int = 300
    max_seconds: float = 3600.
    bounded_head: bool = False
    head_bound: float = 4.
    head_decay_ms: float = 1000.
    condition_full_holds: bool = False
    initial_checkpoint_file: str | None = None
    initial_checkpoint_sha256: str | None = None
    profile_bank_file: str | None = None
    profile_bank_sha256: str | None = None
    profile_head_rate_downstream: bool = True
    row_factorization: str = 'flat'
    train_scope: str = 'all'
    minimum_action_gap_ms: int = 0

    def validate(self):
        super().validate()
        if not self.global_audio or self.bounded_timing:
            raise ValueError('Planned training requires full audio and separate skeleton timing')
        if type(self.bounded_head) is not bool:
            raise ValueError('Bounded head mode must be boolean')
        if type(self.condition_full_holds) is not bool:
            raise ValueError('Full-hold conditioning mode must be boolean')
        if any(not math.isfinite(v) or v <= 0 for v in (self.head_bound, self.head_decay_ms)):
            raise ValueError('Head history bound and decay must be finite and positive')
        for prefix in ('initial_checkpoint', 'profile_bank'):
            path, sha = getattr(self, prefix + '_file'), getattr(self, prefix + '_sha256')
            if path is not None or sha is not None:
                if not path or not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
                    raise ValueError(f'{prefix} requires a path and lowercase SHA-256')
        if self.profile_bank_file is not None and self.initial_checkpoint_file is None:
            raise ValueError('Profile fitting requires a pinned planned initialization checkpoint')
        if type(self.profile_head_rate_downstream) is not bool:
            raise ValueError('Downstream profile head-rate mode must be boolean')
        if not self.profile_head_rate_downstream and self.profile_bank_file is None:
            raise ValueError('Downstream profile routing requires a profile bank')
        if self.row_factorization not in ('flat', 'count_layout'):
            raise ValueError('Row factorization must be flat or count_layout')
        if self.train_scope not in ('all', 'materializer'):
            raise ValueError('Training scope must be all or materializer')
        if self.train_scope == 'materializer' and self.initial_checkpoint_file is None:
            raise ValueError('Materializer-only fitting requires a pinned planned checkpoint')
        if type(self.minimum_action_gap_ms) is not int or self.minimum_action_gap_ms < 0:
            raise ValueError('Minimum action gap must be a nonnegative native-ms integer')
        if self.minimum_action_gap_ms and not self.condition_full_holds:
            raise ValueError('Action spacing requires conditional release waits')
