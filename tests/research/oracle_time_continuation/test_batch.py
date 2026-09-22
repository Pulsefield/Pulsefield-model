from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from ensomi_model.research.oracle_time_continuation.engine import ContinuationEngine
from ensomi_model.research.oracle_time_continuation.model import row_index
from .test_model import make_engine, mixed_rows, skeleton, assert_learned_equal, randomize_optional_readouts


@pytest.mark.parametrize('prefix', [0, 1, 11, 72])
@pytest.mark.parametrize('inference', [False, True])
def test_parallel_frontiers_match_reference_and_future_rows_are_invisible(prefix, inference):
    torch.set_num_threads(1)
    reference = make_engine()
    reference.model.eval()
    parallel = ContinuationEngine(reference.model, parallel_frontiers=True)
    rows = mixed_rows(100)
    with torch.no_grad():
        a = reference.prefill(skeleton(rows), rows[:prefix], inference=inference, progress_every_rows=17)
        b = parallel.prefill(skeleton(rows), rows[:prefix], inference=inference, progress_every_rows=13)
        assert_learned_equal(a, b, atol=6e-6)
        first = reference.teacher_force(a, rows[prefix:])
        second = parallel.teacher_force(b, rows[prefix:])
        torch.testing.assert_close(first.log_probs, second.log_probs, atol=1e-5, rtol=2e-5)
        assert first.relation_query_ids == second.relation_query_ids
        assert first.relation_content_ids == second.relation_content_ids
        assert first.temporal_trace == second.temporal_trace
        assert_learned_equal(first.state, second.state, atol=8e-6)
        changed = list(rows[prefix:])
        if len(changed) > 2:
            changed[1] = replace(changed[1], actions=(0, 1, 1, 1))
            alternative = parallel.teacher_force(b, changed)
            torch.testing.assert_close(second.log_probs[:2], alternative.log_probs[:2], atol=0, rtol=0)


@pytest.mark.parametrize('time_lookahead_rows,clock_readout_hidden', [(0, 0), (16, 0), (0, 12), (16, 12)])
def test_parallel_chunk_parameter_gradients_match_reference_with_detached_history(time_lookahead_rows, clock_readout_hidden):
    torch.set_num_threads(1)
    reference = make_engine(time_lookahead_rows=time_lookahead_rows, clock_readout_hidden=clock_readout_hidden)
    randomize_optional_readouts(reference)
    parallel = ContinuationEngine(deepcopy(reference.model), parallel_frontiers=True)
    rows = mixed_rows(80)
    for engine in (reference, parallel):
        state = engine.prefill(skeleton(rows), rows[:50])
        result = engine.teacher_force(state, rows[50:])
        targets = torch.tensor([row_index(row.actions) for row in rows[50:]])
        (-result.log_probs.gather(1, targets[:, None]).sum() / len(targets)).backward()
    for (name, left), (_, right) in zip(reference.model.named_parameters(), parallel.model.named_parameters()):
        if left.grad is None or right.grad is None:
            assert left.grad is right.grad is None, name
        else:
            torch.testing.assert_close(left.grad, right.grad, atol=3e-6, rtol=3e-4, msg=name)
