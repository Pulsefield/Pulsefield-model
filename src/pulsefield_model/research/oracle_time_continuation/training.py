"""Fixed effective-batch accumulation with content-only prefix replay and TBPTT.

An update owns its sampled windows and replays every prefix with the current
parameters. Each chunk is consumed by one backward and all learned carry is
detached before the next chunk. No learned state survives the optimizer step.
"""
from __future__ import annotations

from itertools import islice
import math
from time import perf_counter
from typing import Sequence

import torch

from ..scoped_style_modeling.dataset import ContractError
from .engine import ContinuationEngine
from .model import CausalBackbone, row_index
from .objective import MARGINAL_NAMES, ObjectiveConfig, sequence_cost
from .state import NeuralState
from .training_config import TrainingConfig
from .windows import TrainingWindow


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_coverage(state: NeuralState) -> dict:
    def coverage(tokens):
        return {"tokens": len(tokens), "source_rows": sum(token.position.count for token in tokens),
                "first_id": tokens[0].position.first_id if tokens else None,
                "last_id": tokens[-1].position.last_id if tokens else None,
                "span_ms": tokens[-1].position.end_ms - tokens[0].position.start_ms if tokens else None}
    return {"recent": coverage(state.temporal.recent), "coarse": coverage(state.temporal.coarse),
            "relation_nodes": len(state.relation.nodes),
            "active_ln_heads": sum(head is not None for head in state.relation.active_heads)}


def target_tensors(states, chunks, device):
    """Build only this chunk's supervision, with pre-row occupancy from exact replay."""
    width = max(map(len, chunks))
    targets = torch.full((len(chunks), width), -1, dtype=torch.long, device=device)
    occupancy = torch.zeros(len(chunks), width, 4, dtype=torch.bool, device=device)
    for index, (state, rows) in enumerate(zip(states, chunks)):
        execution, occupied = state.execution, []
        for row in rows:
            occupied.append(execution.replay.occupancy)
            execution = execution.commit(row)
        if rows:
            targets[index, :len(rows)] = torch.tensor([row_index(row.actions) for row in rows], device=device)
            occupancy[index, :len(rows)] = torch.tensor(occupied, dtype=torch.bool, device=device)
    return targets, occupancy


class SequenceTrainer:
    """Teacher-forced updates on admitted train windows; reports contain no tensors.

    Pre-step failures clear accumulated gradients and do not call optimizer.step.
    This API has no durable recovery; disk checkpoints and restart belong to
    the runtime milestone. Callers must not reuse a partially failed update.
    """

    def __init__(self, model: CausalBackbone, training: TrainingConfig = TrainingConfig(),
                 objective: ObjectiveConfig = ObjectiveConfig()):
        if training.chunk_rows > model.config.max_chunk:
            raise ContractError("Training chunk_rows exceeds the backbone max_chunk")
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise ContractError("Sequence training currently requires FP32 model parameters")
        self.engine = ContinuationEngine(model)
        self.config, self.objective = training, objective
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate,
                                          weight_decay=training.weight_decay)
        self.updates = self.clipped_updates = 0

    def update(self, windows: Sequence[TrainingWindow]) -> dict:
        """Accumulate exactly B_eff windows, then clip once and step once.

        Microbatches may be ragged and targets may exceed chunk_rows. Neither
        active-window counts nor short tail lengths enter the denominator.
        Prefix cost and elapsed time include every row from true chart BOS.
        """
        if len(windows) != self.config.effective_batch_size:
            raise ContractError("An update requires exactly effective_batch_size sampled windows")
        if not torch.is_grad_enabled():
            raise ContractError("Sequence training requires enabled gradients")
        if any(window.source.identity.split != "train" for window in windows):
            raise ContractError("Sequence training only consumes the existing train split")
        model = self.engine.model
        model.train()
        device = next(model.parameters()).device
        synchronize(device)
        started = perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        records, gradient_squared = [], [0., 0., 0.]
        try:
            for first in range(0, len(windows), self.config.microbatch_size):
                batch = windows[first:first + self.config.microbatch_size]
                batch_records, states = [], []
                for window in batch:
                    synchronize(device)
                    prefill_started = perf_counter()
                    state = self.engine.prefill(window.source.skeleton, islice(window.source.targets, window.start))
                    synchronize(device)
                    record = {**window.record(), "prefill_seconds": perf_counter() - prefill_started,
                              "prefix_rows": state.execution.next_index,
                              "prefix_notes": state.execution.replay.note_count,
                              "prefix_elapsed_ms": state.execution.replay.last_row.time_ms - state.execution.replay.first_time_ms,
                              "prefix_coverage": memory_coverage(state), "chunks": 0, "sequence_nll": 0.,
                              "marginal_nll": dict.fromkeys(MARGINAL_NAMES, 0.)}
                    states.append(state)
                    batch_records.append(record)
                for offset in range(0, max(window.target_rows for window in batch), self.config.chunk_rows):
                    chunks = tuple(window.source.targets[min(window.start + offset, window.stop):
                                                          min(window.start + offset + self.config.chunk_rows, window.stop)]
                                   for window in batch)
                    targets, occupancy = target_tensors(states, chunks, device)
                    result = self.engine.teacher_force_batch(states, chunks)
                    cost = sequence_cost(result.log_probs, result.legal, result.valid, targets, occupancy,
                                         effective_batch_size=self.config.effective_batch_size, config=self.objective)
                    cost.loss.backward()
                    nll = cost.nll.detach().sum(1).cpu().tolist()
                    marginals = cost.marginal_nll.detach().sum(1).cpu().tolist()
                    for index, record in enumerate(batch_records):
                        record["chunks"] += bool(chunks[index])
                        record["sequence_nll"] += nll[index]
                        for name, value in zip(MARGINAL_NAMES, marginals[index]):
                            record["marginal_nll"][name] += value
                    for index, value in enumerate(cost.weighted_logit_gradient_squared.cpu().tolist()):
                        gradient_squared[index] += value
                    states = [state.detached() for state in result.states]
                    # Release logits, traces and writer graphs before constructing another chunk.
                    del result, cost, targets, occupancy
                for record, state in zip(batch_records, states):
                    record["nats_per_row"] = record["sequence_nll"] / record["target_rows"]
                    record["final_occupancy"] = state.execution.replay.occupancy
                records.extend(batch_records)
                del states, state
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm,
                                                       error_if_nonfinite=True))
            self.optimizer.step()
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True)
            raise
        self.updates += 1
        self.clipped_updates += norm > self.config.max_grad_norm
        synchronize(device)
        denominator = self.config.effective_batch_size * self.objective.normalization_rows
        nll = sum(record["sequence_nll"] for record in records)
        marginal_nll = {name: sum(record["marginal_nll"][name] for record in records) for name in MARGINAL_NAMES}
        rows = sum(record["target_rows"] for record in records)
        return {"update": self.updates, "effective_batch_size": len(windows), "denominator": denominator,
                "sequence_nll": nll, "sequence_loss": nll / denominator,
                "loss": (nll + self.objective.lambda_struct * sum(marginal_nll.values()) / 3) / denominator,
                "marginal_nll": marginal_nll,
                "weighted_marginal_loss": {name: value * self.objective.lambda_struct / (3 * denominator)
                                           for name, value in marginal_nll.items()},
                "weighted_marginal_logit_gradient_l2": dict(zip(MARGINAL_NAMES, map(math.sqrt, gradient_squared))),
                "target_rows": rows, "nats_per_row": nll / rows,
                "prefill_rows": sum(record["prefix_rows"] for record in records),
                "prefill_seconds": sum(record["prefill_seconds"] for record in records),
                "wall_seconds": perf_counter() - started, "unclipped_gradient_norm": norm,
                "clipped": norm > self.config.max_grad_norm,
                "clipping_frequency": self.clipped_updates / self.updates, "windows": records}
