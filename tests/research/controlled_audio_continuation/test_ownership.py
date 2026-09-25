from dataclasses import replace

import numpy as np
import pytest
import torch

from ensomi_model.research.bounded_typed_continuation.contract import ROW_ACTIONS
from ensomi_model.research.controlled_audio_continuation.model import ControlledAudioModel
from ensomi_model.research.controlled_audio_continuation.generation import ControlledSession
from ensomi_model.research.controlled_audio_continuation.onset_rate import OnsetRate
from ensomi_model.research.joint_audio_continuation.intervals import IntervalExample
from ensomi_model.research.planned_audio_continuation.counts import COUNT_MARKS
from ensomi_model.research.planned_audio_continuation.intervals import collate_interval, score_interval, interval_losses
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule, ControlSpan
from ensomi_model.research.typed_audio_continuation.program import Recovery
from ensomi_model.research.typed_audio_continuation.demand import AudioDemand
from planned_audio_continuation.test_distribution import chart, config


def model():
    return ControlledAudioModel(replace(config(), lookahead=16, bounded_head=True,
        condition_full_holds=True, minimum_action_gap_ms=60), style_names=('tech',))


def test_complete_row_loss_calibrates_counts_and_frontier_can_change_their_mass():
    torch.set_num_threads(1)
    net = model()
    for p in net.parameters():
        torch.nn.init.zeros_(p)
    c = chart([0, 50, 100, 150], [(2, 2, 2, 2), (3, 0, 0, 0), (1, 0, 0, 0), (0, 3, 3, 3)], 200)
    controls = ControlSchedule((ControlSpan(0, 201, stars=3, ln_fraction=.7),), net.style_names)
    batch = collate_interval(IntervalExample(c, 0, 201), net.config, recovery=net.recovery)
    x = batch.inputs
    coarse = net.encode_coarse(torch.from_numpy(c.mel)[None])
    scores = score_interval(net, x, coarse, controls=controls)
    total = interval_losses(scores, batch)[-1]
    assert torch.isfinite(total)
    total.backward()
    assert net.composition.readout[-1].bias.grad.abs().sum() > 0
    assert net.head_control.weight.grad is not None
    assert net.row_control.weight.grad is not None

    single, quad = [ROW_ACTIONS.index(a) for a in ((1, 0, 0, 0), (1, 1, 1, 1))]
    allowed = torch.zeros((1, 256), dtype=torch.bool)
    allowed[:, [single, quad]] = True
    args = (torch.zeros(1, net.config.conditioned_audio_width),
            net.temporal.boundary[None], x.base.row_exact[:1], allowed, x.base.occupancy[:1],
            x.row_preview[:1], x.consequence_local[:1], x.consequence_timing[:1])
    condition = torch.tensor(controls.at([0], encoding=net.control_encoding))
    p = net.planned_row_log_probs(*args, control=condition).exp()
    torch.testing.assert_close(p[0, [single, quad]], torch.tensor([.5, .5]))
    net.zero_grad()
    (-net.planned_row_log_probs(*args, control=condition)[0, single]).backward()
    assert net.composition.readout[-1].bias.grad[COUNT_MARKS.index((4, 0, 0))] > .4
    penalty = torch.zeros(1, 256)
    penalty[0, quad] = -4
    net.row_consequence.score = lambda *unused: penalty
    after = net.planned_row_log_probs(*args, control=condition).exp()
    assert after[0, quad] < .02 and after[0, single] > .98


def test_tap_count_and_layout_do_not_enter_skeleton_network_inputs():
    torch.manual_seed(151)
    net = model()
    controls = ControlSchedule((ControlSpan(0, 501, stars=3, ln_fraction=.2),), net.style_names)
    a = chart([0, 100, 250, 400], [(1, 0, 0, 0), (0, 2, 0, 0), (0, 0, 0, 1), (0, 3, 0, 0)], 500)
    b = chart([0, 100, 250, 400], [(0, 1, 1, 1), (0, 2, 0, 0), (1, 0, 1, 0), (0, 3, 0, 0)], 500)
    batches = [collate_interval(IntervalExample(c, 0, 501), net.config, recovery=net.recovery) for c in (a, b)]
    coarse = net.encode_coarse(torch.from_numpy(a.mel)[None])
    sa, sb = [score_interval(net, v.inputs, coarse, controls=controls) for v in batches]
    torch.testing.assert_close(sa.head, sb.head, rtol=0, atol=0)
    torch.testing.assert_close(sa.release, sb.release, rtol=0, atol=0)
    assert not torch.equal(batches[0].inputs.base.raw, batches[1].inputs.base.raw)


def test_release_preference_is_ln_only_but_its_feasible_wait_uses_r1_response():
    torch.manual_seed(26)
    net = model()
    controls = ControlSchedule((ControlSpan(0, 501, stars=3, ln_fraction=.2),), net.style_names)
    actions = [(0, 0, 0, 2), (0, 1, 1, 0), (0, 0, 0, 3), (1, 0, 0, 0), (0, 0, 0, 1)]
    a = chart([0, 364, 366, 383, 417], actions, 500)
    b = chart([0, 364, 366, 383, 417], [actions[0], (0, 1, 0, 0), *actions[2:]], 500)
    ba, bb = [collate_interval(IntervalExample(c, 0, 501), net.config, recovery=net.recovery) for c in (a, b)]
    torch.testing.assert_close(ba.inputs.release_clock, bb.inputs.release_clock, rtol=0, atol=0)
    torch.testing.assert_close(ba.inputs.skeleton_raw, bb.inputs.skeleton_raw, rtol=0, atol=0)
    coarse = net.encode_coarse(torch.from_numpy(a.mel)[None])
    sa, sb = [score_interval(net, v.inputs, coarse, controls=controls) for v in (ba, bb)]
    torch.testing.assert_close(sa.head, sb.head, rtol=0, atol=0)
    assert not torch.equal(sa.release, sb.release)


def test_scoped_control_update_retains_rows_and_finishes_open_holds():
    torch.manual_seed(18)
    net = model().eval()
    controls = ControlSchedule((ControlSpan(0, 4001, stars=3, ln_fraction=.7),), net.style_names)
    session = ControlledSession(net, np.zeros((400, 128), np.float32), 4000, controls,
                                seed=21, recovery_preference=None)
    session.publish_to(1000)
    prefix = tuple(session.rows)
    retained = tuple(t for t in session.planner.queue if t <= 1100)
    session.update_controls(ControlSpan(1001, 3000, stars=5, ln_fraction=.2, style={'tech': 1}))
    assert tuple(session.planner.queue[:len(retained)]) == retained
    session.publish_to(4000)
    assert tuple(session.rows[:len(prefix)]) == prefix
    assert session.coverage == 4000 and not any(session.replay.occupancy)
    assert all(r.time_ms <= 1000 for r in prefix)
    last = [None]*4
    for row in session.rows:
        for lane, action in enumerate(row.actions):
            if action in (1, 2):
                assert last[lane] is None or row.time_ms-last[lane] >= 60
                last[lane] = row.time_ms


@pytest.mark.parametrize('with_activity', [False, True])
def test_object_demand_changes_r1_choices_without_changing_h_times(with_activity):
    torch.set_num_threads(1)
    torch.manual_seed(203)
    net = model().eval()
    controls = ControlSchedule((ControlSpan(0, 2501, stars=3, ln_fraction=.2),), net.style_names)
    demand = AudioDemand(net.config.conditioned_audio_width,
        controls.width_for(net.control_encoding), 2+len(net.style_names), control_encoding=net.control_encoding)
    # An unusually low nominal rate makes the row-policy effect visible while
    # still requiring R1 to realize every H supplied by the unchanged planner.
    with torch.no_grad():
        demand.query[-1].bias[0] = -3
    rate = OnsetRate(net.config.conditioned_audio_width, controls.width_for(net.control_encoding),
                    2+len(net.style_names)).eval() if with_activity else None
    sessions = [ControlledSession(net, np.zeros((250, 128), np.float32), 2500, controls,
        seed=177, row_demand_model=choice, onset_rate_model=rate) for choice in (None, demand)]
    for session in sessions:
        session.publish_to(1000)
        prefix = tuple(session.rows)
        session.update_controls(ControlSpan(1400, 2100, stars=5))
        session.publish_to(2500)
        assert tuple(session.rows[:len(prefix)]) == prefix
        assert not any(session.replay.occupancy)
    heads = lambda s: [r.time_ms for r in s.rows if any(a in (1, 2) for a in r.actions)]
    assert heads(sessions[0]) == heads(sessions[1])
    assert sessions[0].rows != sessions[1].rows
    assert sum(a in (1, 2) for r in sessions[1].rows for a in r.actions) < sum(
        a in (1, 2) for r in sessions[0].rows for a in r.actions)
