import pytest
import torch

from ensomi_model.research.oracle_time_continuation.engine import ContinuationEngine
from ensomi_model.research.oracle_time_continuation.config import BackboneConfig
from ensomi_model.research.oracle_time_continuation.model import CausalBackbone, row_index
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow, TimeSkeleton


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
@pytest.mark.parametrize('parallel', [False, True])
def test_default_fp32_backbone_max_chunk_backward_and_cached_prediction_on_mps(parallel):
    torch.manual_seed(87)
    model = CausalBackbone(BackboneConfig(time_lookahead_rows=16)).to("mps").eval()
    torch.nn.init.normal_(model.timing.projection[-1].weight, std=.02)
    engine = ContinuationEngine(model, parallel_frontiers=parallel)
    rows = [CompleteRow(0, (2, 0, 0, 0))]
    rows += [CompleteRow(index * 20 + (90000 if index > 100 else 0), (0, 1, index % 2, 0))
             for index in range(1, 159)]
    rows += [CompleteRow(rows[-1].time_ms + 20, (3, 0, 0, 0))]
    skeleton = TimeSkeleton(tuple(row.time_ms for row in rows))
    state = engine.prefill(skeleton, rows[:32])
    with torch.no_grad():
        cached = engine.prefill(skeleton, rows[:32], inference=True)
        torch.testing.assert_close(engine.predict(state).log_probs, engine.predict(cached).log_probs,
                                   atol=3e-5, rtol=3e-5)
        expected = engine.teacher_force(cached, rows[32:])
    result = engine.teacher_force(state, rows[32:])
    torch.testing.assert_close(result.log_probs, expected.log_probs, atol=3e-5, rtol=3e-5)
    targets = torch.tensor([row_index(row.actions) for row in rows[32:]], device="mps")
    loss = -result.log_probs.gather(1, targets[:, None]).sum() / 128
    assert torch.isfinite(loss)
    loss.backward()
    for module in (model.facts, model.local, model.relation, model.temporal, model.head, model.timing):
        gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(value).all() for value in gradients)
        assert sum(value.abs().sum().item() for value in gradients) > 0
    assert result.state.execution.finished
    assert not any(result.state.execution.replay.occupancy)
