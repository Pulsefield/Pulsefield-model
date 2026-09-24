"""Typed process settings for the planned head/release research prototype."""
from dataclasses import dataclass

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

    def validate(self):
        super().validate()
        if not self.global_audio or self.bounded_timing:
            raise ValueError('Planned training requires full audio and separate skeleton timing')
