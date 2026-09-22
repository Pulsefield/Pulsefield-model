from copy import deepcopy

import pytest
import torch

from ensomi_model.research.oracle_time_continuation.engine import ContinuationEngine
from ensomi_model.research.oracle_time_continuation.model import row_index
from ensomi_model.research.oracle_time_continuation.widen import widen_temporal
from ensomi_model.research.scoped_style_modeling.dataset import ContractError
from .test_model import make_engine, mixed_rows, skeleton


@pytest.mark.parametrize('noise', [0., .01])
def test_widen_preserves_bos_cached_and_chunk_predictions_and_breaks_gradient_symmetry(noise):
    torch.set_num_threads(1)
    source = make_engine(temporal_expansion=2, temporal_bias_hidden=8, time_lookahead_rows=16).model.eval()
    torch.nn.init.normal_(source.timing.projection[-1].weight, std=.02)
    before = deepcopy(source.state_dict())
    rng = torch.get_rng_state().clone()
    large = widen_temporal(source, split_noise=noise)
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert large.config.temporal_hidden == 2 * source.config.temporal_hidden
    small_engine = ContinuationEngine(source, parallel_frontiers=True)
    large_engine = ContinuationEngine(large, parallel_frontiers=True)
    rows = mixed_rows(80)
    with torch.no_grad():
        for prefix in (0, 1, 18, 51):
            a = small_engine.prefill(skeleton(rows), rows[:prefix], inference=True)
            b = large_engine.prefill(skeleton(rows), rows[:prefix], inference=True)
            torch.testing.assert_close(small_engine.predict(a).log_probs, large_engine.predict(b).log_probs,
                                       atol=8e-6, rtol=3e-5)
            left = small_engine.teacher_force(a, rows[prefix:])
            right = large_engine.teacher_force(b, rows[prefix:])
            assert torch.equal(left.legal, right.legal)
            torch.testing.assert_close(left.log_probs, right.log_probs, atol=8e-6, rtol=3e-5)
    state = large_engine.prefill(skeleton(rows), rows[:51])
    result = large_engine.teacher_force(state, rows[51:])
    ids = torch.tensor([row_index(row.actions) for row in rows[51:]])
    loss = -result.log_probs.gather(1, ids[:, None]).mean()
    loss.backward()
    grad = large.temporal.input_projection.weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    if noise:
        assert not torch.allclose(*grad.chunk(2), atol=1e-8, rtol=1e-5)


def test_widen_rejects_ambiguous_or_unsupported_inputs():
    with pytest.raises(ContractError, match='explicit'):
        widen_temporal(make_engine().model)
    source = make_engine(temporal_expansion=2, temporal_bias_hidden=8).model
    with pytest.raises(ContractError, match='noise'):
        widen_temporal(source, split_noise=float('nan'))
    with pytest.raises(ContractError, match='CPU FP32'):
        widen_temporal(source.double())
