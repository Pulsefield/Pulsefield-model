import numpy as np
import torch

from ensomi_model.research.planned_audio_continuation.model import PlannedModelConfig
from ensomi_model.research.typed_audio_continuation.allocation import Allocation, LnFeedback, ln_episodes
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule, ControlSpan
from ensomi_model.research.typed_audio_continuation.generation import TypedSession
from ensomi_model.research.typed_audio_continuation.model import TypedAudioModel
from ensomi_model.research.typed_audio_continuation.program import MARKS
from ensomi_model.research.typed_audio_continuation.response_preference import RecoveryPreference


def test_amount_ownership_ignores_other_fields_and_shadowed_boundaries():
    controls = ControlSchedule((ControlSpan(0, 1000, ln_fraction=.2),
        ControlSpan(400, 500, ln_fraction=.5), ControlSpan(200, 700, ln_fraction=.7),
        ControlSpan(300, 800, stars=5., style={'tech': 1.})), ('tech',))
    spans = ln_episodes(controls)
    assert [(s.start_ms, s.end_ms, s.ln_fraction) for s in spans] == [
        (0, 200, .2), (200, 700, .7), (700, 1000, .2)]
    previous = Allocation(0, .2, 100, 30)
    assert previous.in_scope(spans[0]) is previous
    assert previous.in_scope(spans[2]).heads == 0
    feedback = LnFeedback()
    assert feedback.log_odds_shift(previous) < 0
    assert feedback.log_odds_shift(Allocation(0, .2, 100, 10)) > 0
    assert abs(feedback.log_odds_shift(Allocation(0, .2, 100000, 100000))) <= 1


def test_unpublished_plan_rolls_allocation_back_with_the_matching_prefix():
    torch.manual_seed(451)
    torch.set_num_threads(1)
    model = TypedAudioModel(PlannedModelConfig(history_levels=2, skeleton_levels=2), ln_prior=.2).eval()
    schedule = ControlSchedule((ControlSpan(0, 6001, ln_fraction=.7),))
    session = TypedSession(model, np.zeros((600, 128), np.float32), 6000, schedule,
                           seed=78, ln_feedback=LnFeedback(),
                           recovery_preference=RecoveryPreference(head_pressure=4.))
    session.publish_to(1000)
    rows = tuple(session.rows)
    # Some planned events are beyond the requested change and must be undone.
    change = session.coverage+1
    retained = session.published_point.allocation
    session.update_controls(ControlSpan(change, 4000, ln_fraction=.2))
    assert session.allocation == retained
    assert session.recent_heads == session.published_point.recent_heads
    session.publish_to(4500)
    assert tuple(session.rows[:len(rows)]) == rows
    point = session.published_point.allocation
    # The restored .7 request has a new episode; the overridden interval creates
    # no repayment obligation. Verify actual published plus queued plan counts.
    events = [(r.time_ms, sum(a in (1, 2) for a in r.actions), r.actions.count(2)) for r in session.rows]
    within = [e for e in events if point.start_ms <= e[0] <= session.published_point.cursor]
    assert point.start_ms == 4000
    assert point.heads == sum(e[1] for e in within)
    assert point.ln_heads == sum(e[2] for e in within)
    last = session.queue[-1][2].allocation if session.queue else point
    queued = [MARKS[q[0][1]] for q in session.queue]
    assert last.heads == point.heads+sum(t+l for t,l,_ in queued)
    assert session.published_point.recent_heads == tuple(
        (t, h) for t, h, _ in events if h and session.published_point.cursor-t < 169)
