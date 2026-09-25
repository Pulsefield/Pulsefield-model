import torch

from ensomi_model.research.typed_audio_continuation.program import MARKS, HEADS
from ensomi_model.research.typed_audio_continuation.proportions import GROUPS, GROUP_INDEX, condition_ln_count


def test_ln_control_preserves_head_release_mass_and_has_positive_response():
    rng = torch.Generator().manual_seed(193)
    raw = torch.randn(8, len(MARKS), generator=rng, requires_grad=True)
    support = torch.tensor(HEADS)[None].expand(8, -1) > 0
    support[:, ::7] = False
    original = raw.masked_fill(~support, -torch.inf).softmax(-1)
    low = condition_ln_count(raw, support, torch.full((8,), .2)).exp()
    high = condition_ln_count(raw, support, torch.full((8,), .7)).exp()
    groups = torch.tensor(GROUP_INDEX)
    for g in range(len(GROUPS)):
        torch.testing.assert_close(low[:, groups == g].sum(-1), original[:, groups == g].sum(-1))
        torch.testing.assert_close(high[:, groups == g].sum(-1), original[:, groups == g].sum(-1))
    counts = torch.tensor([m[1] for m in MARKS], dtype=raw.dtype)
    assert bool(((high @ counts) > (low @ counts)).all())
    loss = -(high[:, 15].log()).sum()
    loss.backward()
    assert bool(torch.isfinite(raw.grad).all())
