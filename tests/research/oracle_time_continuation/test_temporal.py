import pytest
import torch

from pulsefield_model.research.oracle_time_continuation.config import BackboneConfig
from pulsefield_model.research.oracle_time_continuation.temporal import TemporalEncoder, TemporalState


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def expected_keys(n, config, content=False):
    blocks = max(0, n - config.recent) // config.coarse_group
    coarse = [((block - 1) * config.coarse_group + 1, block * config.coarse_group, True)
              for block in range(max(1, blocks - config.coarse_capacity + 1), blocks + 1)]
    fine = [(index, index, False) for index in range(blocks * config.coarse_group + 1, n + 1 + content)]
    return tuple(coarse + fine) or ((0, 0, False),)


@pytest.mark.parametrize("width", (1, 17, 64, 128))
def test_default_archive_births_visibility_fifo_payload_and_no_pending_gap(width):
    torch.manual_seed(13)
    config = BackboneConfig(hidden=8, heads=2, temporal_layers=1)
    encoder = TemporalEncoder(config).eval()
    # Includes the first transition and one full-capacity coarse eviction.
    length = config.recent + (config.coarse_capacity + 1) * config.coarse_group + 2
    values, queries = torch.randn(length, 2, 8), torch.randn(length, 2, 8)
    times = tuple(1e12 + index * .25 for index in range(length))
    state = TemporalState()
    observed = set()
    with torch.no_grad():
        for start in range(0, length, width):
            end = min(length, start + width)
            output, state, trace = encoder.chunk(queries[start:end], values[start:end], state, times[start:end],
                                                 inference=False)
            assert torch.isfinite(output).all()
            assert len(trace.candidates) <= config.recent_capacity + config.coarse_capacity + width + (width + 15) // 16 + 1
            for offset, (query_keys, content_keys) in enumerate(zip(trace.query_ids, trace.content_ids)):
                n = start + offset
                assert query_keys == expected_keys(n, config)
                assert content_keys == expected_keys(n, config, content=True)
                if n in (512, 513, 527, 528, 1552):
                    observed.add(n)
                for position in trace.candidates:
                    if position.is_coarse:
                        block = position.last_id // config.coarse_group
                        assert position.birth == config.recent + block * config.coarse_group
                        assert position.eviction == config.recent + (block + config.coarse_capacity) * config.coarse_group
                        assert position.start_ms == times[position.first_id - 1]
                        assert position.end_ms == times[position.last_id - 1]
            assert tuple(token.position.key for token in state.tokens) == expected_keys(end, config)
            assert len(state.recent) <= 527 and len(state.coarse) <= 64
        assert observed == {512, 513, 527, 528, 1552}
        for token in state.tokens:
            expected = values[token.position.first_id - 1:token.position.last_id].mean(0)
            torch.testing.assert_close(token.inputs[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize('expansion', [1, 3])
def test_two_layer_step_dense_and_cached_inference_have_identical_layer_inputs(expansion):
    torch.manual_seed(44)
    config = BackboneConfig(hidden=12, heads=3, recent=4, coarse_group=3, coarse_capacity=2,
                            temporal_expansion=expansion, temporal_bias_hidden=8 if expansion > 1 else None)
    encoder = TemporalEncoder(config).eval()
    contents, queries = torch.randn(36, 2, 12), torch.randn(36, 2, 12)
    times = tuple(index * 10. if index < 10 else 90000. + index * 10. for index in range(36))
    with torch.no_grad():
        step = TemporalState()
        expected = []
        for content, query, time in zip(contents, queries, times):
            expected.append(encoder.query(query, step, time))
            step = encoder.commit(content, step, time, inference=False)
        for cached in (False, True):
            for width in (1, 7, 17, 36):
                state, outputs = TemporalState(), []
                for start in range(0, 36, width):
                    end = min(36, start + width)
                    result, state, _ = encoder.chunk(queries[start:end], contents[start:end], state, times[start:end],
                                                     inference=cached)
                    outputs.append(result)
                torch.testing.assert_close(torch.cat(outputs), torch.stack(expected), atol=2e-6, rtol=2e-5)
                assert tuple(token.position for token in state.tokens) == tuple(token.position for token in step.tokens)
                for left, right in zip(state.tokens, step.tokens):
                    for a, b in zip(left.inputs, right.inputs):
                        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
                    if cached:
                        for layer, raw in enumerate(left.inputs):
                            expected_kv = encoder.layers[layer].project(raw[None])
                            for a, b in zip(left.projected[layer], expected_kv):
                                torch.testing.assert_close(a, b[:, :, 0], atol=2e-6, rtol=2e-5)


def test_coarse_means_precede_normalization_and_projection():
    torch.manual_seed(16)
    config = BackboneConfig(hidden=8, heads=2, temporal_layers=1, recent=2, coarse_group=2)
    encoder = TemporalEncoder(config).eval()
    values = torch.randn(4, 2, 8) * torch.tensor([.1, 10., 1., 1.])[:, None, None]
    with torch.no_grad():
        state = TemporalState()
        for index, content in enumerate(values):
            state = encoder.commit(content, state, index, inference=True)
        coarse = state.coarse[0]
        torch.testing.assert_close(coarse.inputs[0], values[:2].mean(0), rtol=0, atol=0)
        correct = encoder.layers[0].project(values[:2].mean(0, keepdim=True))
        wrong = encoder.layers[0].project(values[:2])
        for result, expected, projected_first in zip(coarse.projected[0], correct, wrong):
            torch.testing.assert_close(result, expected[:, :, 0], atol=0, rtol=0)
            assert not torch.allclose(result, projected_first.mean(2), atol=.01)


def test_detached_history_trains_current_read_normalization_and_kv_but_not_writer():
    torch.manual_seed(26)
    config = BackboneConfig(hidden=12, heads=3, recent=4, coarse_group=2, coarse_capacity=2)
    encoder = TemporalEncoder(config)
    writer = torch.nn.Linear(5, 12)
    state = TemporalState()
    original_inputs = writer(torch.randn(10, 2, 5))
    original_inputs.retain_grad()
    for index, content in enumerate(original_inputs):
        state = encoder.commit(content, state, index * 100., inference=False)
    state = state.detached()
    assert len(state.tokens) > 1 and state.coarse
    result = encoder.query(torch.randn(2, 12), state, 1050.)
    (result * torch.randn_like(result)).sum().backward()
    assert writer.weight.grad is None and original_inputs.grad is None
    for layer in encoder.layers:
        for parameter in (layer.norm.weight, layer.key.weight, layer.value.weight):
            assert parameter.grad is not None and parameter.grad.norm() > 1e-6
    assert all(value.grad_fn is None for token in state.tokens for value in token.inputs)


def test_current_chunk_writers_get_later_loss_and_never_current_or_earlier_loss():
    torch.manual_seed(59)
    config = BackboneConfig(hidden=8, heads=2, recent=3, coarse_group=2, coarse_capacity=2)
    encoder = TemporalEncoder(config)
    writer = torch.nn.Linear(5, 8)
    contents = writer(torch.randn(9, 2, 5))
    contents.retain_grad()
    queries = torch.randn_like(contents)
    result, _, _ = encoder.chunk(queries, contents, TemporalState(), tuple(float(i) for i in range(9)), inference=False)
    loss = (result[5] * torch.randn_like(result[5])).sum()
    loss.backward()
    assert writer.weight.grad is not None and writer.weight.grad.norm() > 1e-6
    assert contents.grad[:5].norm() > 1e-6
    assert not contents.grad[5:].any()


@pytest.mark.parametrize('expansion', [1, 3])
def test_dense_and_step_gradient_paths_agree_without_tbptt(expansion):
    torch.manual_seed(80)
    config = BackboneConfig(hidden=8, heads=2, recent=3, coarse_group=2, coarse_capacity=2,
                            temporal_expansion=expansion, temporal_bias_hidden=8 if expansion > 1 else None)
    encoder = TemporalEncoder(config)
    queries = torch.randn(12, 2, 8, requires_grad=True)
    contents = torch.randn(12, 2, 8, requires_grad=True)
    weights = torch.randn(12, 2, config.temporal_hidden)
    times = tuple(float(i * 30) for i in range(12))
    dense, _, _ = encoder.chunk(queries, contents, TemporalState(), times, inference=False)
    parameters = (queries, contents, *encoder.parameters())
    # The top layer's content output is not persistent; its FFN can receive
    # query gradients but must not secretly become the next event's key/value.
    dense_grads = torch.autograd.grad((dense * weights).sum(), parameters, allow_unused=True)
    state, outputs = TemporalState(), []
    for query, content, time in zip(queries, contents, times):
        outputs.append(encoder.query(query, state, time))
        state = encoder.commit(content, state, time, inference=False)
    step_grads = torch.autograd.grad((torch.stack(outputs) * weights).sum(), parameters, allow_unused=True)
    for dense_grad, step_grad in zip(dense_grads, step_grads):
        if dense_grad is None or step_grad is None:
            assert dense_grad is step_grad is None
        else:
            torch.testing.assert_close(dense_grad, step_grad, atol=2e-5, rtol=3e-5)
