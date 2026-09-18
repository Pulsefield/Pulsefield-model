import numpy as np
import pytest
import torch

from pulsefield_model.research.bounded_typed_continuation.contract import Arm, Schedule, Timing
from pulsefield_model.research.bounded_typed_continuation.features import (
    AVAILABILITY_DIM, CONTENT_DIM, QUERY_DIM, TIMING_DIM, EndpointAvailability,
    TimingView, content_features, endpoint_availability, factor_features,
    query_features, time_features, transition_features,
)
from pulsefield_model.research.oracle_time_continuation.features import TIME_DIM, clock_features
from pulsefield_model.research.oracle_time_continuation.schema import CompleteRow
from pulsefield_model.research.scoped_style_modeling.dataset import ContractError


def test_time_basis_preserves_missing_zero_and_large_clock_differences():
    values = [None, 0., .125, -75., 1e9]
    expected = clock_features(values, torch.zeros(1)).numpy()
    np.testing.assert_allclose(time_features(values), expected, atol=2e-6, rtol=2e-6)
    assert not time_features([None]).any() and time_features([0.])[0, -1] == 1.
    row = CompleteRow(2. ** 40 + .125, (2, 0, 0, 0))
    raw = content_features([row], [2. ** 40], [[2. ** 40 + .625, None, None, None]])
    small = content_features([CompleteRow(.125, row.actions)], [0.], [[.625, None, None, None]])
    np.testing.assert_array_equal(raw, small)


def test_online_content_only_exposes_committed_object_plans_and_skips_no_event():
    timing = Timing((0., 100., 200., 500.), (True, False, True, False))
    row_state = Schedule(Arm.R1, timing)
    object_state = Schedule(Arm.O1, timing)
    row_next, row = row_state.advance((2, 0, 0, 0))
    object_next, obj = object_state.advance((2, 0, 0, 0), {0: 3})
    r1 = transition_features(row_state, row_next, row)
    o1 = transition_features(object_state, object_next, obj)
    assert r1.shape == o1.shape == (2, CONTENT_DIM)
    assert not r1[:, 16 + TIME_DIM:].any() and o1[:, 16 + TIME_DIM:].any()
    skipped, absent = object_next.advance(None)
    assert transition_features(object_next, skipped, absent) is None
    after, row = skipped.advance((0, 1, 0, 0))
    token = transition_features(skipped, after, row)
    np.testing.assert_array_equal(token[0, 16:16 + TIME_DIM], time_features([200.])[0])
    with pytest.raises(ContractError, match='newly committed'):
        content_features([row], [0.], [[300., None, None, None]])


def test_complete_exact_plan_and_age_survive_a_long_crop_boundary():
    times = tuple(float(i * 100) for i in range(602))
    timing = Timing(times, (True,) * 601 + (False,))
    state = Schedule(Arm.O1, timing)
    state, _ = state.advance((2, 0, 0, 0), {0: 601})
    for _ in range(600):
        state, _ = state.advance((0, 1, 0, 0))
    features = query_features([state], TimingView(timing))
    assert features.shape == (1, 2, QUERY_DIM)
    np.testing.assert_array_equal(features[0, 0, :TIME_DIM], time_features([60100.])[0])
    np.testing.assert_array_equal(features[0, 0, 18 * TIME_DIM:19 * TIME_DIM], time_features([0.])[0])


def test_mirroring_swaps_feature_hands_and_preserves_pointer_relative_roles():
    timing = Timing((0., 100., 200., 10000.), (True, True, False, False))
    a, row = Schedule(Arm.O1, timing).advance((2, 1, 0, 0), {0: 3})
    b, reverse = Schedule(Arm.O1, timing).advance((0, 0, 1, 2), {3: 3})
    view = TimingView(timing)
    np.testing.assert_array_equal(query_features([a], view)[:, ::-1], query_features([b], view))
    content = content_features([row], [None], [[10000., None, None, None]])
    mirrored = content_features([reverse], [None], [[None, None, None, 10000.]])
    np.testing.assert_array_equal(content[:, ::-1], mirrored)
    group = (0, 2, 2, 2)
    np.testing.assert_array_equal(factor_features(a, group, {1: 2}, 2),
                                  factor_features(b, group[::-1], {2: 2}, 1))


def test_timing_uses_complete_supplied_future_without_action_labels():
    times = (0., 50., 100., 10000., 10100., 50000.)
    typed = TimingView(Timing(times, (True, True, False, True, False, False)))
    untyped = TimingView(Timing(times))
    values = typed.queries([0, 2, 5])
    assert values.shape == (3, TIMING_DIM)
    assert values[0, -6] == 1 and untyped.queries([0])[0, -6] == 0
    assert values[0, -1] == 0 and values[-1, -1] == 1
    assert values[0, -4] == 0 and values[-1, -4] == 1
    # The next long gap is defined by the full supplied timing, even if the
    # caller's target window ends at its second candidate.
    np.testing.assert_array_equal(values[0, 32 * TIME_DIM:33 * TIME_DIM], time_features([100.])[0])
    translated = TimingView(Timing(tuple(t + 2. ** 40 for t in times), typed.timing.onsets))
    np.testing.assert_array_equal(values, translated.queries([0, 2, 5]))
    np.testing.assert_array_equal(typed.candidates(0, 1, 6), translated.candidates(0, 1, 6))
    with pytest.raises(ValueError):
        typed.times[0] = 5.


def test_candidate_availability_matches_literal_future_holds_and_endpoint_inclusion():
    times = (0., 30., 70., 100., 125., 170., 500., 900.)
    roles = (True, True, False, True, True, True, False, False)
    view = TimingView(Timing(times, roles))
    facts = EndpointAvailability((5, 3), 1)
    actual = view.availability(0, 1, len(times), facts)
    expected = []
    onset_indices = [i for i, role in enumerate(roles) if role]
    weights = {b: 1000 / (times[b] - times[a]) for a, b in zip(onset_indices, onset_indices[1:])}
    for end in range(1, len(times)):
        row = []
        for others in range(4):
            bound = max([0] + [i for i in range(1, end + 1)
                               if sum(other >= i for other in facts.other_ends) >= others])
            included = [i for i in onset_indices if 0 < i <= bound]
            row.extend(np.log1p([len(included), sum(weights[i] for i in included), times[bound] / 1000]) / 8)
        expected.append(row + [2 / 3, 1 / 3])
    assert actual.shape == (7, AVAILABILITY_DIM)
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-7)
    # At end index3, H at100ms still has three occupied columns. It cannot
    # restart a lane that releases on that row.
    assert actual[2, 6] == pytest.approx(np.log1p(2) / 8)
    shifted = TimingView(Timing(tuple(t + 2. ** 40 for t in times), roles))
    np.testing.assert_array_equal(actual, shifted.availability(0, 1, len(times), facts))
    np.testing.assert_array_equal(actual, np.concatenate([view.availability(0, a, b, facts)
                                                        for a, b in ((1, 3), (3, 8))]))


def test_partial_endpoint_availability_has_only_prior_factors_and_mirrors():
    timing = Timing((0., 100., 150., 200., 400., 900.), (True, True, False, True, False, False))
    state, _ = Schedule(Arm.O1, timing).advance((2, 0, 0, 0), {0: 5})
    reverse, _ = Schedule(Arm.O1, timing).advance((0, 0, 0, 2), {3: 5})
    heads = (0, 2, 2, 2)
    first = endpoint_availability(state, heads, {}, 1)
    assert first == EndpointAvailability((5,), 2)
    later = endpoint_availability(state, heads, {1: 3}, 2)
    assert later == EndpointAvailability((5, 3), 1)
    assert later == endpoint_availability(reverse, heads[::-1], {2: 3}, 1)
    with pytest.raises(ContractError, match='partial endpoint'):
        TimingView(timing).availability(1, 2, 6, EndpointAvailability((1,), 0))
    with pytest.raises(ContractError, match='partial endpoint'):
        TimingView(Timing(timing.times_ms)).availability(1, 2, 6, first)
