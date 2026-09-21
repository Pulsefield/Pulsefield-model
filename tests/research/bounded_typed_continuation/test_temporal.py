import copy

import pytest
import torch

from pulsefield_model.research.bounded_typed_continuation.temporal import FiniteTemporal, TemporalConfig
from pulsefield_model.research.scoped_style_modeling.dataset import ContractError


def network(device='cpu', levels=3):
    torch.manual_seed(43)
    return FiniteTemporal(TemporalConfig(input_dim=11, hidden=12, levels=levels, expansion=2)).to(device)


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='requires Apple MPS'))])
def test_dense_cached_and_cropped_histories_agree(device):
    model = network(device)
    raw = torch.randn(1, 47, 2, 11, device=device)
    with torch.no_grad():
        dense = model(raw)
        cache = model.empty_cache()
        for i in range(raw.shape[1]):
            previous = model.read(cache).clone()
            if i:
                torch.testing.assert_close(previous, dense[0, i-1], atol=2e-5, rtol=2e-5)
            cache = model.append(cache, raw[0, i])
            torch.testing.assert_close(model.read(cache), dense[0, i], atol=2e-5, rtol=2e-5)
        width = model.config.receptive_tokens
        cropped = model(raw[:, -width:])
        torch.testing.assert_close(cropped[:, -1], dense[:, -1], atol=2e-5, rtol=2e-5)
        assert all(buffer.numel() == 2*d*2*12 for buffer,d in zip(cache.buffers,model.config.dilations))


def test_actual_dependency_is_finite_and_gradients_reach_context_writers():
    model = network()
    width = model.config.receptive_tokens
    assert width == 15 and TemporalConfig(11).receptive_tokens == 511
    raw = torch.randn(1, width + 7, 2, 11, requires_grad=True)
    out = model(raw)
    out[:, -1].square().sum().backward()
    assert raw.grad[:, :-width].count_nonzero() == 0
    assert raw.grad[:, -width].abs().sum() > 0
    changed = raw.detach().clone()
    changed[:, :-width] = torch.randn_like(changed[:, :-width]) * 100
    torch.testing.assert_close(model(changed)[:, -1], out[:, -1], atol=0, rtol=0)


def test_cropped_recomputation_has_the_same_parameter_gradients():
    a = network()
    b = copy.deepcopy(a)
    raw = torch.randn(2, 39, 2, 11)
    a(raw)[:, -1].square().sum().backward()
    b(raw[:, -a.config.receptive_tokens:])[:, -1].square().sum().backward()
    for pa, pb in zip(a.parameters(), b.parameters()):
        if pa.grad is None:
            assert pb.grad is None
        else:
            torch.testing.assert_close(pa.grad, pb.grad, atol=2e-5, rtol=2e-5)


def test_padding_and_query_alignment_preserve_bos_vs_truncated_and_no_current_action():
    model = network()
    with torch.no_grad():
        model.boundary[0].fill_(1.)
        model.boundary[1].fill_(2.)
    raw = torch.randn(2, 12, 2, 11)
    valid = torch.zeros(2, 12, dtype=torch.bool)
    valid[:, 3:10] = True
    content = model(raw, valid)
    torch.testing.assert_close(content[:, 3:10], model(raw[:, 3:10]), atol=2e-6, rtol=2e-6)
    before = model.before(content, valid, truncated_start=torch.tensor([False, True]))
    assert (before[0, 3] == 1).all() and (before[1, 3] == 2).all()
    torch.testing.assert_close(before[:, 4], content[:, 3])
    changed = raw.clone()
    changed[:, 8:] *= 30
    updated = model.before(model(changed, valid), valid, truncated_start=torch.tensor([False, True]))
    torch.testing.assert_close(updated[:, :9], before[:, :9], atol=0, rtol=0)
    assert content[~valid].count_nonzero() == 0
    valid[:, 5] = False
    with pytest.raises(ContractError, match='internal'):
        model(raw, valid)


def test_hand_exchange_is_equivariant_and_cache_cannot_cross_an_update():
    model = network()
    raw = torch.randn(2, 17, 2, 11)
    torch.testing.assert_close(model(raw.flip(2)), model(raw).flip(2), atol=0, rtol=0)
    cache = model.append(model.empty_cache(), raw[0, 0])
    with torch.no_grad():
        model.input.weight.add_(.01)
    with pytest.raises(ContractError, match='parameter version'):
        model.append(cache, raw[0, 1])
    with pytest.raises(ContractError, match='different model'):
        network().read(cache)


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='requires Apple MPS'))])
def test_full_eight_level_field_has_no_hidden_dependency_beyond_511_tokens(device):
    model = network(device, levels=8)
    raw = torch.randn(1, 550, 2, 11, device=device)
    with torch.no_grad():
        expected = model(raw)[:, -1]
        cropped = model(raw[:, -511:])[:, -1]
        changed = raw.clone()
        changed[:, :-511] *= 100
        torch.testing.assert_close(cropped, expected, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(model(changed)[:, -1], expected, atol=0, rtol=0)
