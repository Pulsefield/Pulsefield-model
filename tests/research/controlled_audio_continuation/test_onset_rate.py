import numpy as np
import torch

from ensomi_model.research.controlled_audio_continuation.generation import ControlledSession
from ensomi_model.research.controlled_audio_continuation.onset_rate import OnsetRate, onset_prefix
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule, ControlSpan
from controlled_audio_continuation.test_ownership import model
from planned_audio_continuation.test_distribution import chart


def test_onset_targets_do_not_count_chord_members_or_release_only_rows():
    source = chart([0, 100, 200, 300], [(2, 2, 2, 2), (3, 3, 3, 3), (1, 0, 0, 0), (0, 1, 1, 0)], 500)
    np.testing.assert_array_equal(onset_prefix(source.source.rows), [0, 1, 1, 2, 3])


def test_onset_ledger_restores_lookahead_and_forks_without_row_counts():
    torch.manual_seed(305)
    torch.set_num_threads(1)
    net = model().eval()
    controls = ControlSchedule((ControlSpan(0, 6001, stars=3, ln_fraction=.7),), net.style_names)
    rate = OnsetRate(net.config.conditioned_audio_width, controls.width_for(net.control_encoding),
                    2+len(net.style_names)).eval()
    session = ControlledSession(net, np.zeros((600, 128), np.float32), 6000, controls,
                                seed=130, onset_rate_model=rate)
    session.publish_to(1000)
    prefix = tuple(session.rows)
    # Lookahead has sampled Hs beyond publication; only retained Hs may count
    # after revising its suffix. The sum is independent of all row actions.
    assert session.planner.generated[-1] > session.coverage
    session.update_controls(ControlSpan(1400, 4200, stars=5))
    check = lambda planner: sum(np.exp(-(planner.cursor-t)/4000) for t in planner.generated)
    np.testing.assert_allclose(session.planner.onset_balance.mass_at(session.planner.cursor, 4000), check(session.planner))
    fork = session.fork()
    session.publish_to(4500)
    fork.publish_to(4500)
    assert session.rows == fork.rows
    assert tuple(session.rows[:len(prefix)]) == prefix
    for current in (session, fork):
        np.testing.assert_allclose(current.planner.onset_balance.mass_at(current.planner.cursor, 4000), check(current.planner))
    fork.update_controls(ControlSpan(4600, 5800, ln_fraction=.2))
    old = (session.planner.onset_balance, tuple(session.rows))
    fork.publish_to(6000)
    assert old == (session.planner.onset_balance, tuple(session.rows))
    session.publish_to(6000)
    assert not any(session.replay.occupancy) and not any(fork.replay.occupancy)
