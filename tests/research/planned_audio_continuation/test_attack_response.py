from itertools import product

import numpy as np
import torch

from ensomi_model.research.bounded_typed_continuation.contract import ROW_ACTIONS
from ensomi_model.research.oracle_time_continuation.replay import ExactReplayState, commit
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.planned_audio_continuation import generation
from ensomi_model.research.planned_audio_continuation.attack_response import (
    select_response_row, short_attack_costs, short_attack_pairs,
)
from ensomi_model.research.planned_audio_continuation.features import HeadPreview, row_support
from ensomi_model.research.planned_audio_continuation.model import PlannedAudioModel
from .test_distribution import config


def replay(rows):
    state = ExactReplayState()
    for t, actions in rows:
        state = commit(state, CompleteRow(t, actions))
    return state


def test_attack_diagnostic_is_strict_and_counts_tap_and_ln_heads_only():
    rows = [CompleteRow(t, a) for t, a in [
        (0, (2, 1, 0, 0)), (1, (3, 0, 1, 0)), (19, (1, 0, 0, 2)),
        (20, (0, 1, 0, 0)), (21, (0, 0, 2, 3)), (39, (0, 0, 0, 1)),
    ]]
    pairs = short_attack_pairs(rows)
    assert pairs == [dict(lane=0, previous_ms=0, time_ms=19, gap_ms=19, actions=[2, 1])]
    # A release neither counts as an attack nor clears the preceding attack.
    assert not short_attack_pairs(rows[2:])


def test_upstream_chord_commits_an_unavoidable_short_repeat():
    state = replay([(46114, (0, 1, 0, 0)), (46664, (1, 0, 1, 1))])
    preview = HeadPreview((47166, 47182), True)
    costs = short_attack_costs(state, 47163, preview)
    support = row_support([state], [47163], [True], [preview], 48000)[0]
    for i in np.flatnonzero(support):
        assert costs[i] == max(0, sum(a in (1, 2) for a in ROW_ACTIONS[i]) + 2 - 4)
    state = commit(state, CompleteRow(47163, (1, 1, 0, 1)))
    state = commit(state, CompleteRow(47166, (0, 0, 1, 0)))
    preview = HeadPreview((), True)
    support = row_support([state], [47182], [True], [preview], 48000)[0]
    assert short_attack_costs(state, 47182, preview)[support].min() == 1


def test_exact_twenty_is_excluded_and_all_nearby_heads_are_considered():
    state = ExactReplayState()
    triple = ROW_ACTIONS.index((1, 1, 0, 1))
    assert short_attack_costs(state, 100, HeadPreview((103, 120), True))[triple] == 0
    assert short_attack_costs(state, 100, HeadPreview((103, 119), True))[triple] == 1
    single = ROW_ACTIONS.index((0, 1, 0, 0))
    assert short_attack_costs(state, 100, HeadPreview((101, 102, 103, 104), True))[single] == 1
    rested = replay([(80, (1, 1, 1, 1))])
    assert short_attack_costs(rested, 100, HeadPreview((), True))[triple] == 0


def test_finite_response_matches_exhaustive_lane_assignments_and_mirror():
    state = replay([(50, (2, 0, 0, 0)), (84, (0, 1, 0, 0)), (95, (0, 0, 1, 1))])
    now, preview = 100, HeadPreview((101, 105, 119), True)
    support = row_support([state], [now], [True], [preview], 200)[0]
    costs = short_attack_costs(state, now, preview)
    for i in np.flatnonzero(support):
        after = commit(state, CompleteRow(now, ROW_ACTIONS[i]))
        immediate = len(short_attack_pairs([state.last_row, CompleteRow(now, ROW_ACTIONS[i])]))
        # Include the older column-1 clock omitted by that two-row diagnostic.
        immediate += int(ROW_ACTIONS[i][1] in (1, 2) and now - 84 < 20)
        possible_costs = []
        for lanes in product(range(4), repeat=3):
            if after.occupancy[lanes[0]]:  # No earlier native release at H=now+1.
                continue
            clocks = list(after.last_lane_attack_ms)
            count = immediate
            for time_ms, lane in zip(preview.times_ms, lanes):
                count += clocks[lane] is not None and time_ms - clocks[lane] < 20
                clocks[lane] = time_ms
            possible_costs.append(count)
        assert costs[i] == min(possible_costs)
    mirrored = replay([(50, (0, 0, 0, 2)), (84, (0, 0, 1, 0)), (95, (1, 1, 0, 0))])
    mirror_costs = short_attack_costs(mirrored, now, preview)
    np.testing.assert_equal(costs, mirror_costs[[ROW_ACTIONS.index(a[::-1]) for a in ROW_ACTIONS]])


def test_selection_preserves_healthy_sample_and_lexicographic_minimum_changes():
    state = ExactReplayState()
    now, preview = 100, HeadPreview((103, 119), True)
    costs = short_attack_costs(state, now, preview)
    support = row_support([state], [now], [True], [preview], 200)[0]
    log_probs = torch.full((256,), -1000., dtype=torch.float64)
    proposal = ROW_ACTIONS.index((2, 1, 0, 1))
    minimal_edit = ROW_ACTIONS.index((2, 1, 0, 0))
    log_probs[minimal_edit] = -999.
    rng = torch.Generator().manual_seed(12)
    before = rng.get_state()
    assert select_response_row(log_probs, minimal_edit, support, costs, rng) == minimal_edit
    assert torch.equal(before, rng.get_state())
    chosen = select_response_row(log_probs, proposal, support, costs, rng)
    assert costs[chosen] == 0
    actions = ROW_ACTIONS[chosen]
    assert sum(a in (1, 2) for a in actions) == 2 and actions.count(2) == 1
    assert sum(a != b for a, b in zip(actions, ROW_ACTIONS[proposal])) == 1
    assert not torch.equal(before, rng.get_state())


def test_policy_defaults_off_and_healthy_rollout_does_not_change(monkeypatch):
    class FixedPlan:
        def __init__(self, *args):
            self.queue, self.generated, self.finished, self.bins = [0, 3, 19], [0, 3, 19], True, 0

        def fill(self, count):
            pass

        def preview(self, now, count):
            future = tuple(t for t in self.queue if t > now)
            return HeadPreview(future[:count], len(future) < count)

    monkeypatch.setattr(generation, 'HeadPlanner', FixedPlan)
    model = PlannedAudioModel(config())
    calls = []

    def run(policy=None, healthy=False):
        calls.clear()

        def scores(audio, hidden, raw, legal, *rest):
            choice = [(1, 0, 0, 0) if healthy else (1, 1, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0)][len(calls)]
            calls.append(choice)
            logits = torch.full((1, 256), -30.)
            logits[0, ROW_ACTIONS.index(choice)] = 30.
            return logits.masked_fill(~legal, -torch.inf).log_softmax(-1)

        monkeypatch.setattr(model, 'planned_row_log_probs', scores)
        kwargs = {} if policy is None else dict(correct_short_attacks=policy)
        return generation.rollout(model, np.zeros((5, 128), np.float32), 50, seed=17, **kwargs)

    default, disabled, enabled = run(), run(False), run(True)
    assert default.completed and disabled.completed and enabled.completed
    assert default.rows == disabled.rows
    assert len(default.metrics['short_attack_pairs']) == 1
    assert not enabled.metrics['short_attack_pairs']
    assert [r.time_ms for r in enabled.rows] == [0, 3, 19]
    assert enabled.metrics['response_decisions'][0]['time_ms'] == 0
    healthy, corrected = run(False, True), run(True, True)
    assert healthy.rows == corrected.rows
    assert not corrected.metrics['response_decisions']
