"""Fixed-window teacher-forced measurements with streamed per-row code lengths."""
from itertools import islice
from time import perf_counter

import torch

from .engine import ContinuationEngine
from .model import row_index
from .runtime import ResourceConfig, write_record
from .training import synchronize


@torch.no_grad()
def evaluate_windows(model, windows, *, progress=None, row_log=None, resources=ResourceConfig()):
    """Score unchanged true prefixes under current weights without optimizer writes.

    Windows retain their pinned group/split identities. Report per-case costs and
    pooled nats/row separately; repeated windows are exposures, not new charts.
    """
    model.eval()
    engine = ContinuationEngine(model, parallel_frontiers=True)
    device = next(model.parameters()).device
    records = []
    started = perf_counter()
    for window in windows:
        state = engine.prefill(window.source.skeleton, islice(window.source.targets, window.start),
                               inference=True, progress=progress)
        nll = 0.
        for first in range(window.start, window.stop, model.config.max_chunk):
            rows = window.source.targets[first:min(first + model.config.max_chunk, window.stop)]
            result = engine.teacher_force(state, rows)
            targets = torch.tensor([row_index(row.actions) for row in rows], device=device)
            costs = (-result.log_probs.gather(1, targets[:, None])[:, 0]).cpu().tolist()
            nll += sum(costs)
            if row_log is not None:
                for offset, cost in enumerate(costs):
                    write_record(row_log, dict(source_sha256=window.source.identity.source_sha256,
                                              event_id=first + offset, nll=cost), resources)
            state = result.state
            del result
            if progress is not None:
                progress('evaluation', row=state.execution.next_index)
        records.append(dict(**window.record(), sequence_nll=nll, nats_per_row=nll / window.target_rows))
        del state
    synchronize(device)
    rows = sum(r['target_rows'] for r in records)
    return dict(cases=records, target_rows=rows, sequence_nll=sum(r['sequence_nll'] for r in records),
                nats_per_row=sum(r['sequence_nll'] for r in records) / rows, seconds=perf_counter() - started)
