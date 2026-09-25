import torch

from ensomi_model.research.typed_audio_continuation.guidance import guide_head_clock, guide_head_count, StarGuidance
from ensomi_model.research.typed_audio_continuation.program import HEADS, MARKS


def test_head_guidance_keeps_release_no_event_odds_and_forced_support():
    c = torch.tensor([[.97, .02, .01], [.9, 0., .1], [0., 0., 1.]]).log()
    u = torch.tensor([[.97, .01, .02], [.7, 0., .3], [0., 0., 1.]]).log()
    p = guide_head_clock(c, u, 2.).exp()
    torch.testing.assert_close(p.sum(-1), torch.ones(3))
    assert p[0, 1] > c[0, 1].exp()
    torch.testing.assert_close(p[0, 2]/p[0, 0], c[0, 2].exp()/c[0, 0].exp())
    torch.testing.assert_close(p[1:], c[1:].exp())


def test_head_count_guidance_preserves_joint_ln_release_conditionals():
    rng = torch.Generator().manual_seed(217)
    support = torch.tensor([[0 < t+l < 4 and r in (0, 1) for t,l,r in MARKS]])
    c = torch.randn(1, len(MARKS), generator=rng).masked_fill(~support, -torch.inf).log_softmax(-1)
    u = torch.randn(1, len(MARKS), generator=rng).masked_fill(~support, -torch.inf).log_softmax(-1)
    guided = guide_head_count(c, u, 2.)
    torch.testing.assert_close(guided.exp().sum(-1), torch.ones(1))
    assert torch.equal(torch.isfinite(guided), support)
    for count in (1, 2, 3):
        mask = torch.tensor(HEADS == count) & support[0]
        torch.testing.assert_close(guided[:, mask].softmax(-1), c[:, mask].softmax(-1))


def test_only_difficulty_is_omitted_from_the_contrast():
    class Model:
        style_names = ('tech',)
        def row_log_probs(self, *args):
            self.seen.append(args[6].clone())
            return torch.tensor([[.4, .6]]).log()
    model = Model(); model.seen = []
    control = torch.tensor([[.5, -.6, 1., 1., 1., 1., 2., 3.]])
    StarGuidance().scores(model, 'row', None, None, None, None, None, None, control)
    expected = control.clone();expected[:, 0] = expected[:, 3] = 0
    torch.testing.assert_close(model.seen[0], control)
    torch.testing.assert_close(model.seen[1], expected)
