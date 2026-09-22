from copy import deepcopy
from itertools import product

import numpy as np
import pytest
import torch

from ensomi_model.research.bounded_typed_continuation.contract import Arm, Schedule, Timing
from ensomi_model.research.bounded_typed_continuation.features import CONTENT_DIM, EndpointAvailability, TimingView, query_features
from ensomi_model.research.bounded_typed_continuation.model import BoundedModel, EndpointFactor, EndpointPointer, ModelConfig
from ensomi_model.research.scoped_style_modeling.dataset import ContractError


def small_model(arm=Arm.O1, device='cpu', availability='none'):
    torch.manual_seed(171)
    return BoundedModel(ModelConfig(arm, hidden=12, levels=2, coupling_rank=3,
                                   endpoint_availability=availability)).to(device)


@pytest.mark.parametrize('arm', list(Arm))
def test_joint_actions_normalize_on_exact_support_and_mirror(arm):
    model = small_model(arm)
    timing = Timing((0., 100., 200.), None if arm == Arm.R0 else (True, True, False))
    a, _ = Schedule(arm, timing).advance((2, 0, 1, 0), {0: 2} if arm == Arm.O1 else None)
    b, _ = Schedule(arm, timing).advance((0, 1, 0, 2), {3: 2} if arm == Arm.O1 else None)
    history = torch.randn(1, 2, model.config.hidden)
    histories = torch.cat((history, history.flip(1)))
    exact = torch.from_numpy(query_features([a, b], TimingView(timing)))
    encoded = model.readout(histories, exact)
    actual = model.decision_log_probs(encoded, [a, b])
    torch.testing.assert_close(actual.exp().sum(-1), torch.ones(2))
    reverse = [model.choices.index(row[::-1]) for row in model.choices]
    torch.testing.assert_close(actual[0], actual[1, reverse], atol=2e-6, rtol=2e-6)
    valid = a.head_support() if arm == Arm.O1 else a.row_support()
    assert torch.isfinite(actual[0]).tolist() == list(valid)


@pytest.mark.parametrize('device', ['cpu', 'mps'])
@pytest.mark.parametrize('availability', ['none', 'commitment'])
def test_chunked_recomputed_pointer_matches_dense_values_and_all_gradients(device, availability):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    torch.manual_seed(19)
    dtype = torch.float64 if device == 'cpu' else torch.float32
    pointer = EndpointPointer(8, availability).to(device=device, dtype=dtype)
    if pointer.availability_residual is not None:
        torch.nn.init.normal_(pointer.availability_residual[-1].weight, std=.05)
    dense = deepcopy(pointer)
    view = TimingView(Timing(tuple(float(i * i) for i in range(41)), tuple(i % 2 == 0 for i in range(41))))
    facts = EndpointAvailability((40, 39), 0)
    factors = [EndpointFactor(view, 0, 1, 41, 40, facts), EndpointFactor(view, 2, 3, 31, 11, facts),
               EndpointFactor(view, 30, 31, 32, 31, facts)]
    context = torch.randn(3, 8, device=device, dtype=dtype, requires_grad=True)
    copied = context.detach().clone().requires_grad_()
    actual = pointer.log_prob(context, factors, candidate_budget=7, recompute=True)
    expected = dense.log_prob(copied, factors, candidate_budget=200, recompute=False)
    tolerance = 2e-10 if device == 'cpu' else 2e-5
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    assert abs(float(actual[-1].detach())) < tolerance
    (-actual.sum()).backward()
    (-expected.sum()).backward()
    torch.testing.assert_close(context.grad, copied.grad, atol=tolerance, rtol=tolerance)
    for (name, a), (_, b) in zip(pointer.named_parameters(), dense.named_parameters()):
        if name.startswith('context.'):
            assert a.grad is b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('availability', ['none', 'commitment'])
def test_complete_dependent_endpoint_mixture_normalizes_and_is_mirror_equivariant(availability):
    model = small_model(availability=availability).double()
    if model.pointer.availability_residual is not None:
        torch.nn.init.normal_(model.pointer.availability_residual[-1].weight, std=.1)
    timing = Timing((0., 40., 100., 160.), (True, False, True, False))
    state, view = Schedule(Arm.O1, timing), TimingView(timing)
    group = (2, 2, 2, 2)
    targets = [dict(enumerate(ends)) for ends in product((1, 2, 3), repeat=4) if 1 in ends]
    encoded = torch.randn(1, 2, model.config.hidden, dtype=torch.float64)
    hands = encoded.expand(len(targets), -1, -1)
    actual = model.endpoint_log_probs(hands, [state] * len(targets), [group] * len(targets), targets,
                                      [view] * len(targets), candidate_budget=17)
    assert float(actual.exp().sum().detach()) == pytest.approx(1., abs=1e-12)
    mirrored = [{3 - c: end for c, end in target.items()} for target in targets]
    reverse = model.endpoint_log_probs(hands.flip(1), [state] * len(targets), [group] * len(targets), mirrored,
                                       [view] * len(targets), candidate_budget=11)
    torch.testing.assert_close(actual, reverse, atol=2e-12, rtol=2e-12)
    # Endpoints remain distinct decisions; equal-start LNs are not tied.
    assert any(len(set(target.values())) > 1 for target in targets)


def test_head_type_and_endpoint_gradients_reach_the_common_encoder():
    model = small_model()
    timing = Timing((0., 100., 200., 300., 400.), (True, True, True, False, False))
    state, _ = Schedule(Arm.O1, timing).advance((1, 0, 0, 0))
    view = TimingView(timing)
    history = model.temporal(torch.randn(1, 4, 2, CONTENT_DIM))[:, -1]
    hands = model.readout(history, torch.from_numpy(query_features([state], view)))
    group, target = (2, 2, 0, 0), {0: 3, 1: 4}
    head = model.decision_log_probs(hands, [state])[0, model.choices.index(group)]
    head_gradient = torch.autograd.grad(-head, model.temporal.input.weight, retain_graph=True)[0]
    assert head_gradient.norm() > 1e-6
    endpoint = model.endpoint_log_probs(hands, [state], [group], [target], [view], candidate_budget=2).sum()
    endpoint_gradient = torch.autograd.grad(-endpoint, model.temporal.input.weight, retain_graph=True)[0]
    assert endpoint_gradient.norm() > 1e-6
    (-head - endpoint).backward()
    assert model.joint.unary.weight.grad.norm() > 1e-6
    assert model.pointer.context[0].weight.grad.norm() > 1e-6
    assert model.pointer.candidate[0].weight.grad.norm() > 1e-6


@pytest.mark.parametrize('device', ['cpu', 'mps'])
@pytest.mark.parametrize('availability', ['none', 'commitment'])
def test_endpoint_sampling_keeps_feasibility_and_is_chunk_partition_invariant(device, availability):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    model = small_model(device=device, availability=availability)
    if model.pointer.availability_residual is not None:
        torch.nn.init.normal_(model.pointer.availability_residual[-1].weight, std=.05)
    timing = Timing((0., 40., 100., 160.), (True, False, True, False))
    state, view = Schedule(Arm.O1, timing), TimingView(timing)
    group, hands = (2, 2, 2, 2), torch.randn(2, model.config.hidden, device=device)
    for seed in range(15):
        a = model.sample_endpoints(hands, state, group, view, generator=torch.Generator().manual_seed(seed), candidate_budget=1)
        b = model.sample_endpoints(hands, state, group, view, generator=torch.Generator().manual_seed(seed), candidate_budget=100)
        assert a == b and 1 in a.values()
        state.advance(group, a)
    with pytest.raises(ContractError, match='target'):
        EndpointFactor(view, 0, 1, 4, True)


def test_all_arms_start_with_identical_common_weights_at_a_shared_initialization():
    models = [small_model(arm) for arm in Arm]
    for name in ('temporal', 'exact', 'fuse'):
        parameters = [dict(getattr(model, name).named_parameters()) for model in models]
        for key in parameters[0]:
            torch.testing.assert_close(parameters[0][key], parameters[1][key], atol=0, rtol=0)
            torch.testing.assert_close(parameters[0][key], parameters[2][key], atol=0, rtol=0)


@pytest.mark.parametrize('availability', ['none', 'commitment'])
def test_recomputed_candidate_blocks_do_not_retain_full_future_activation_storage(availability):
    pointer = EndpointPointer(16, availability)
    view = TimingView(Timing(tuple(float(i) for i in range(2001)), (True,) * 2001))
    facts = EndpointAvailability((1700, 800), 0)
    factors = [EndpointFactor(view, 0, 1, 2001, 2000, facts), EndpointFactor(view, 4, 5, 1700, 1000, facts)]

    def saved_bytes(recompute):
        saved = {}
        def pack(tensor):
            storage = tensor.untyped_storage()
            saved[storage.data_ptr()] = storage.nbytes()
            return tensor
        context = torch.randn(2, 16, requires_grad=True)
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            loss = -pointer.log_prob(context, factors, candidate_budget=97, recompute=recompute).sum()
        retained = sum(saved.values())
        loss.backward()
        return retained

    dense, recomputed = saved_bytes(False), saved_bytes(True)
    assert recomputed < dense / 10
