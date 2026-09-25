import numpy as np

from ensomi_model.research.bounded_typed_continuation.features import TIME_DIM, time_features
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule, ControlSpan, PER_FIELD_SCOPE
from ensomi_model.research.typed_audio_continuation.evaluation import describe_control_ranges


def test_partial_controls_resolve_into_separate_restored_ranges():
    schedule = ControlSchedule((ControlSpan(0, 1000, stars=3., ln_fraction=.2),
        ControlSpan(200, 700, style={'tech': 1.}), ControlSpan(400, 600, stars=5.)), ('tech',))
    spans = schedule.resolved_ranges(0, 1000)
    assert [(s.start_ms, s.end_ms, s.stars, s.ln_fraction, s.style) for s in spans] == [
        (0, 200, 3., .2, {}), (200, 400, 3., .2, {'tech': 1.}),
        (400, 600, 5., .2, {'tech': 1.}), (600, 700, 3., .2, {'tech': 1.}),
        (700, 1000, 3., .2, {})]


def test_each_condition_retains_its_scope_through_partial_override_and_restoration():
    schedule = ControlSchedule((ControlSpan(0, 1000, stars=3., ln_fraction=.2),
        ControlSpan(200, 700, style={'tech': 1.}), ControlSpan(400, 600, stars=5.)), ('tech',))
    encoded = schedule.at([300, 500, 650, 750], encoding=PER_FIELD_SCOPE)
    np.testing.assert_array_equal(encoded[:, :6], schedule.at([300, 500, 650, 750])[:, :6])
    clocks = encoded[:, 6:].reshape(4, 3, 2*TIME_DIM)
    expected = [((300, 700), (300, 700), (100, 400)),
                ((100, 100), (500, 500), (300, 200)),
                ((650, 350), (650, 350), (450, 50)),
                ((750, 250), (750, 250), (None, None))]
    np.testing.assert_array_equal(clocks, time_features(expected).reshape(clocks.shape))
    other = ControlSchedule((ControlSpan(250, 350, stars=3., ln_fraction=.2),
                              ControlSpan(200, 700, style={'tech': 1.})), ('tech',))
    # Shared clocks collapse two different difficulty/LN scopes at this point.
    np.testing.assert_array_equal(schedule.at([300]), other.at([300]))
    assert not np.array_equal(encoded[:1], other.at([300], encoding=PER_FIELD_SCOPE))


def test_range_metrics_keep_crossing_holds_without_inventing_boundary_heads():
    rows = [CompleteRow(100, (2, 0, 0, 0)), CompleteRow(200, (0, 1, 0, 0)),
            CompleteRow(300, (3, 0, 0, 0)), CompleteRow(500, (0, 0, 2, 0)),
            CompleteRow(800, (0, 0, 3, 0))]
    schedule = ControlSchedule((ControlSpan(0, 1000, stars=3., ln_fraction=.2),
                                ControlSpan(200, 600, stars=5., ln_fraction=.7)))
    a, b, c = describe_control_ranges(rows, schedule, 1000, window_ms=250)
    assert [r['counts']['heads'] for r in (a, b, c)] == [1, 2, 0]
    assert [r['ln_head_fraction'] for r in (a, b, c)] == [1., .5, None]
    assert b['counts']['entering_LNs'] == b['counts']['leaving_LNs'] == 1
    assert b['counts']['releases_of_entering_LNs'] == 1
    assert b['released_LN_duration_ms']['median'] == 200
    assert c['released_LN_duration_ms']['median'] == 300
    for r in (a, b, c):
        q = r['counts']
        assert q['releases'] == q['ln_heads']+q['entering_LNs']-q['leaving_LNs']
        assert all(r['start_ms'] <= w['start_ms'] < w['end_ms'] <= r['end_ms'] for w in r['windows'])
    assert [(w['start_ms'], w['end_ms'], w['full_window']) for w in b['windows']] == [
        (200, 450, True), (450, 600, False)]
