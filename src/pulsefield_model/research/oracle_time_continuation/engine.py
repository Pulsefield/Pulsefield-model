"""Scheduler/query/commit boundary shared by teacher forcing and continuation.

The M0 data interface stays independent of PyTorch. Learned execution imports
model types lazily and preserves the same exact replay and terminal rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

from ..scoped_style_modeling.dataset import ContractError
from .replay import ExactReplayState, ReplayClocks, commit, legal_rows
from .schema import Actions, CompleteRow, TimeSkeleton, checked_time

if TYPE_CHECKING:
    from .model import CausalBackbone, JointRowDistribution
    from .state import BatchResult, ChunkResult, NeuralState


@dataclass(frozen=True)
class PredictionInput:
    """The model's entire exact input: current time, terminal condition, history.

    Target rows, full skeleton, source identity and execution position stay in
    their owners. Reading this object cannot consume a row or advance a cache.
    """

    time_ms: float
    is_terminal: bool
    history: ExactReplayState

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_ms", checked_time(self.time_ms))
        if type(self.is_terminal) is not bool:
            raise ContractError("The true skeleton terminal flag must be a boolean")
        if self.history.is_complete or (self.history.last_row is not None and
                                        self.time_ms <= self.history.last_row.time_ms):
            raise ContractError("Prediction time must follow an unfinished committed prefix")

    @property
    def clocks(self) -> ReplayClocks:
        return self.history.clocks_at(self.time_ms)

    @property
    def legal_actions(self) -> tuple[Actions, ...]:
        return legal_rows(self.history, is_terminal=self.is_terminal)


@dataclass(frozen=True)
class ContinuationState:
    """Execution state; pass query(), rather than this object, to a predictor."""

    skeleton: TimeSkeleton
    replay: ExactReplayState = ExactReplayState()

    def __post_init__(self) -> None:
        position = self.next_index
        if position > len(self.skeleton.times_ms):
            raise ContractError("Replay position exceeds the skeleton")
        if position and (
            self.replay.first_time_ms != self.skeleton.times_ms[0]
            or self.replay.last_row.time_ms != self.skeleton.times_ms[position - 1]
            or self.replay.is_complete != self.finished
        ):
            raise ContractError("Replay boundary differs from its skeleton position or terminal condition")

    @property
    def next_index(self) -> int:
        return self.replay.row_count

    @property
    def finished(self) -> bool:
        return self.next_index == len(self.skeleton.times_ms)

    def query(self) -> PredictionInput:
        """Return immutable pre-row input, rejecting an exhausted skeleton."""
        if self.finished:
            raise ContractError("The skeleton is exhausted; there is no next row")
        return PredictionInput(self.skeleton.times_ms[self.next_index],
                               self.next_index == len(self.skeleton.times_ms) - 1, self.replay)

    def commit(self, row: CompleteRow) -> ContinuationState:
        """Commit one true or sampled row at exactly the next skeleton time."""
        query = self.query()
        if row.time_ms != query.time_ms:
            raise ContractError("Commit must match the next skeleton time")
        return ContinuationState(self.skeleton, commit(self.replay, row, is_terminal=query.is_terminal))


def prefill(skeleton: TimeSkeleton, prefix: Iterable[CompleteRow]) -> ContinuationState:
    """Replay from the true beginning, including release rows; never reset at a window.

    The prefix must cover consecutive skeleton positions starting at zero. A
    resource or sample horizon has no terminal semantics. An empty prefix is BOS.
    """
    state = ContinuationState(skeleton)
    for row in prefix:
        state = state.commit(row)
    return state


class ContinuationEngine:
    """Atomic online execution plus bounded, time-parallel teacher forcing.

    All learned states belong to this model's fixed parameters and device.
    Inference caches require eval mode with gradients disabled. Training carry
    stores raw layer inputs; call state.detached() at a TBPTT boundary.
    """

    def __init__(self, model: CausalBackbone):
        self.model = model

    def start(self, skeleton: TimeSkeleton, *, inference: bool = False) -> NeuralState:
        from .local import LocalState
        from .relation import RelationState
        from .state import NeuralState
        from .temporal import TemporalState

        state = NeuralState(ContinuationState(skeleton), LocalState(), RelationState(), TemporalState(),
                            self.model.cache_signature(), inference)
        self._check(state)
        return state

    def _check(self, state: NeuralState) -> None:
        import torch

        if state.signature != self.model.cache_signature():
            raise ContractError("Learned state parameters/device/cache version changed; replay the prefix")
        if state.inference and (self.model.training or torch.is_grad_enabled()):
            raise ContractError("Projected inference caches require eval mode with gradients disabled")
        if state.temporal.row_count != state.execution.next_index:
            raise ContractError("Learned memory and exact replay must share the committed row count")

    def predict(self, state: NeuralState) -> JointRowDistribution:
        """Read a pre-row distribution without mutating any exact or learned state."""
        self._check(state)
        return self.model(state.execution.query(), state.local, state.relation, state.temporal)

    def commit(self, state: NeuralState, row: CompleteRow) -> NeuralState:
        """Validate a complete row, construct content privately, then publish new state."""
        from .state import NeuralState

        self._check(state)
        query = state.execution.query()
        execution = state.execution.commit(row)
        content, local, relation = self.model.content_input(query, row, execution.replay, state.local, state.relation)
        temporal = self.model.temporal.commit(content, state.temporal, row.time_ms, inference=state.inference)
        return NeuralState(execution, local, relation, temporal, state.signature, state.inference)

    def prefill(self, skeleton: TimeSkeleton, prefix: Iterable[CompleteRow], *, inference: bool = False) -> NeuralState:
        """Content-only no-grad replay from true BOS; no prefix logits are computed."""
        import torch

        with torch.no_grad():
            state = self.start(skeleton, inference=inference)
            for row in prefix:
                state = self.commit(state, row)
            return state.detached()

    def teacher_force(self, state: NeuralState, rows: Sequence[CompleteRow]) -> ChunkResult:
        """Score a chunk before each target becomes content; leave carry attached.

        Local/relation inputs are constructed in row order; temporal layers run
        in parallel with explicit per-row masks. A chunk end never closes LNs.
        Empty chunks preserve state and contain no queries or padding rows.
        """
        import torch

        from .model import PreRowEncoding
        from .state import ChunkResult, NeuralState
        from .temporal import TemporalTrace

        self._check(state)
        if len(rows) > self.model.config.max_chunk:
            raise ContractError("Teacher-forcing chunk exceeds max_chunk; split without resetting state")
        if not rows:
            like = next(self.model.parameters())
            return ChunkResult(like.new_empty(0, 256), torch.empty(0, 256, dtype=torch.bool, device=like.device),
                               state, TemporalTrace((), (), ()), (), ())
        execution, local, relation = state.execution, state.local, state.relation
        queries, contents, inputs, relation_queries, relation_contents = [], [], [], [], []
        for row in rows:
            query = execution.query()
            following = execution.commit(row)
            inputs.append(query)
            relation_queries.append(relation.visible_ids)
            queries.append(self.model.query_input(query, local, relation))
            content, local, relation = self.model.content_input(query, row, following.replay, local, relation)
            contents.append(content)
            relation_contents.append(relation.visible_ids)
            execution = following
        hidden, temporal, trace = self.model.temporal.chunk(torch.stack(queries), torch.stack(contents), state.temporal,
                                                           tuple(row.time_ms for row in rows), inference=state.inference)
        log_probs, legal = self.model.head(PreRowEncoding(hidden, tuple(inputs)))
        updated = NeuralState(execution, local, relation, temporal, state.signature, state.inference)
        return ChunkResult(log_probs, legal, updated, trace, tuple(relation_queries), tuple(relation_contents))

    def teacher_force_batch(self, states: Sequence[NeuralState], rows: Sequence[Sequence[CompleteRow]]) -> BatchResult:
        """Dispatch ragged per-chart dense chunks, then pad outputs with a validity mask.

        Charts have independent banks and may have different prefix lengths.
        No padded row enters an encoder or advances the corresponding state.
        """
        import torch
        from torch.nn import functional as F

        from .state import BatchResult

        if not states or len(states) != len(rows):
            raise ContractError("A batch requires matching nonempty state and chunk lists")
        chunks = tuple(self.teacher_force(state, chunk) for state, chunk in zip(states, rows))
        width = max(len(chunk) for chunk in rows)
        log_probs = torch.stack([F.pad(chunk.log_probs, (0, 0, 0, width - len(chunk.log_probs))) for chunk in chunks])
        legal = torch.stack([F.pad(chunk.legal, (0, 0, 0, width - len(chunk.legal)), value=False) for chunk in chunks])
        valid = torch.tensor([[index < len(chunk) for index in range(width)] for chunk in rows],
                             dtype=torch.bool, device=log_probs.device)
        return BatchResult(log_probs, legal, valid, tuple(chunk.state for chunk in chunks), chunks)
