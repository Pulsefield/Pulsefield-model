import math

import torch

from ensomi_model.research.controlled_audio_continuation.allocation import LnAmountFeedback, LnAmountState
from ensomi_model.research.planned_audio_continuation.counts import ROW_COUNTS
from ensomi_model.research.typed_audio_continuation.allocation import ln_episodes
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule, ControlSpan


def test_final_amount_tilt_preserves_head_release_marginals_and_conditional_layout():
    torch.manual_seed(192)
    scores = torch.randn(256, dtype=torch.float64)
    scores[::7] = -torch.inf
    scores = scores.log_softmax(-1)
    policy = LnAmountFeedback()
    changed = policy.scores(scores, LnAmountState(0, .7, -1.3))
    counts = torch.tensor(ROW_COUNTS)
    for h in range(5):
        for r in range(5):
            mask = (counts[:, 0] == h) & (counts[:, 2] == r)
            torch.testing.assert_close(changed.exp()[mask].sum(), scores.exp()[mask].sum())
            for l in range(h+1):
                group = mask & (counts[:, 1] == l) & torch.isfinite(scores)
                if group.any():
                    torch.testing.assert_close(changed[group].log_softmax(-1), scores[group].log_softmax(-1))
    assert torch.equal(torch.isfinite(changed), torch.isfinite(scores))
    assert (changed.exp()*counts[:, 1]).sum() < (scores.exp()*counts[:, 1]).sum()


def test_integral_state_corrects_a_persistent_amount_bias():
    policy = LnAmountFeedback()
    state = LnAmountState(0, .7)
    total = 0.
    for _ in range(800):
        # Expected feedback isolates bias response from finite-sample variance.
        p = 1/(1+math.exp(-(math.log(.9/.1)+state.offset)))
        total += p
        state = policy.advance(state, 1-p, p)
    assert abs(total/800-.7) < .02
    assert abs(p-.7) < .001


def test_projection_does_not_store_saturated_debt_and_scopes_are_independent():
    policy = LnAmountFeedback()
    state = LnAmountState(0, .7)
    for _ in range(1000):
        state = policy.advance(state, 4, 0)
    assert state.offset == 2
    assert policy.advance(state, 0, 4).offset < 2
    assert policy.advance(state, 0, 0) == state
    schedule = ControlSchedule((ControlSpan(0, 1000, ln_fraction=.7),
        ControlSpan(200, 400, stars=3, style={'tech': 1}),
        ControlSpan(500, 700, ln_fraction=.2)), ('tech',))
    episodes = ln_episodes(schedule)
    assert [(s.start_ms, s.end_ms) for s in episodes] == [(0, 500), (500, 700), (700, 1000)]
    assert state.in_scope(episodes[0]) is state
    assert state.in_scope(episodes[1]).offset == 0
    assert state.in_scope(episodes[2]).offset == 0
    unknown = state.in_scope(None)
    scores = torch.randn(256).log_softmax(-1)
    assert policy.scores(scores, unknown) is scores
