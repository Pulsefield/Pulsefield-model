"""Owned learned carry; exact facts and execution position retain their M0 owner."""
from __future__ import annotations

from dataclasses import dataclass, replace

from torch import Tensor

from .engine import ContinuationState
from .local import LocalState
from .relation import RelationState
from .temporal import TemporalState, TemporalTrace


@dataclass(frozen=True)
class NeuralState:
    execution: ContinuationState
    local: LocalState
    relation: RelationState
    temporal: TemporalState
    signature: tuple
    inference: bool = False

    def detached(self) -> NeuralState:
        """Cut every learned writer graph into owned storage at a TBPTT boundary.

        Exact LN obligations, relation identities, pace and temporal provenance
        remain unchanged. No state points to a prior state or a whole-chart view.
        """
        return replace(self, local=self.local.detached(), relation=self.relation.detached(),
                       temporal=self.temporal.detached())


@dataclass(frozen=True)
class ChunkResult:
    log_probs: Tensor
    legal: Tensor
    state: NeuralState
    temporal_trace: TemporalTrace
    relation_query_ids: tuple[tuple[int, ...], ...]
    relation_content_ids: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class BatchResult:
    """Ragged chunks padded only in output tensors; invalid cells are zero."""

    log_probs: Tensor
    legal: Tensor
    valid: Tensor
    states: tuple[NeuralState, ...]
    chunks: tuple[ChunkResult, ...]
