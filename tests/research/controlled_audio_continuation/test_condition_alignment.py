import math

import pytest
import torch

from ensomi_model.research.controlled_audio_continuation.condition_alignment import condition_alignment


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='MPS unavailable'))])
def test_alignment_rewards_condition_use_without_a_shared_likelihood_shortcut(device):
    reference_positive = torch.tensor([-80., -120.], device=device, requires_grad=True)
    reference_negative = torch.tensor([-79., -119.], device=device, requires_grad=True)
    positive = reference_positive.detach().clone().requires_grad_()
    negative = reference_negative.detach().clone().requires_grad_()
    initial = condition_alignment(positive, negative, reference_positive, reference_negative)
    torch.testing.assert_close(initial.cpu(), torch.tensor(2*math.log(2)), atol=1e-6, rtol=1e-6)
    initial.backward()
    assert torch.isfinite(positive.grad).all() and (positive.grad < 0).all()
    assert torch.isfinite(negative.grad).all() and (negative.grad > 0).all()
    assert reference_positive.grad is reference_negative.grad is None

    improved = condition_alignment(positive+2, negative-2, reference_positive, reference_negative)
    assert improved < initial
    # A condition-independent improvement cannot satisfy the contrast. This
    # distinguishes the auxiliary objective from another positive-only NLL.
    for shift in (-10., -2., 2., 10.):
        same = condition_alignment(positive+shift, negative+shift,
                                   reference_positive, reference_negative)
        assert same > initial
