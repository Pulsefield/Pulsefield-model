import numpy as np
import torch

from ensomi_model.research.planned_audio_continuation.model import PlannedModelConfig
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule, ControlSpan
from ensomi_model.research.typed_audio_continuation.demand import AudioDemand, DemandBalance, DemandCurve, DemandFeedback, pool_audio
from ensomi_model.research.typed_audio_continuation.generation import TypedSession
from ensomi_model.research.typed_audio_continuation.model import TypedAudioModel
from ensomi_model.research.typed_audio_continuation.program import HEADS


def test_discounted_target_and_count_balance_preserve_elapsed_time():
    curve = DemandCurve.from_rates([0, 1000, 2000], [10, 0], 4000)
    expected = 40*(-np.expm1(-.25))
    np.testing.assert_allclose(curve.mass_at([1000, 2000]), [expected, expected*np.exp(-.25)])
    balance = DemandBalance().advance(500, 2, 4000).advance(1000, 3, 4000)
    np.testing.assert_allclose(balance.mass_at(2000, 4000), 2*np.exp(-.375)+3*np.exp(-.25))
    feedback = DemandFeedback()
    assert feedback.shift(DemandBalance(1000, 50), curve, 1000) < 0
    assert feedback.shift(DemandBalance(), curve, 1000) > 0
    assert abs(feedback.shift(DemandBalance(1000, 5000), curve, 1000)) <= 2


def test_future_partial_cell_control_update_does_not_rewrite_past_demand():
    model = AudioDemand(3, ControlSchedule().width, 2)
    audio = torch.ones(503, 3)
    pooled = pool_audio(audio)
    torch.testing.assert_close(pooled, torch.ones(11, 3))
    before = ControlSchedule((ControlSpan(0, 5031, stars=3),))
    after = ControlSchedule((*before.spans, ControlSpan(1763, 3221, stars=5)))
    left = DemandCurve.build(model, audio, before, 5030, 4000)
    right = DemandCurve.build(model, audio, after, 5030, 4000)
    times = np.arange(0, 1764)
    np.testing.assert_allclose(left.mass_at(times), right.mass_at(times))
    assert right.mass_at(3000) > left.mass_at(3000)


def test_native_rollback_restores_the_actual_head_ledger():
    torch.manual_seed(45)
    torch.set_num_threads(1)
    model = TypedAudioModel(PlannedModelConfig(history_levels=2, skeleton_levels=2), ln_prior=.2).eval()
    schedule = ControlSchedule((ControlSpan(0, 6001, stars=3, ln_fraction=.7),))
    demand = AudioDemand(model.config.conditioned_audio_width, schedule.width, 2).eval()
    session = TypedSession(model, np.zeros((600, 128), np.float32), 6000, schedule,
                           seed=78, demand_model=demand)
    session.publish_to(1000)
    rows = tuple(session.rows)
    preserved = session.published_point.demand
    session.update_controls(ControlSpan(session.coverage+1, 4000, stars=5))
    assert session.demand == preserved
    session.publish_to(4500)
    assert tuple(session.rows[:len(rows)]) == rows
    point = session.queue[-1][2] if session.queue else session.published_point
    events = [(r.time_ms, sum(a in (1, 2) for a in r.actions)) for r in session.rows]
    events += [(q[0][0], int(HEADS[q[0][1]])) for q in session.queue]
    expected = sum(h*np.exp(-(point.cursor-t)/4000) for t, h in events)
    np.testing.assert_allclose(point.demand.mass_at(point.cursor, 4000), expected)
