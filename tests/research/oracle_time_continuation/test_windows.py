from collections import defaultdict
from dataclasses import replace
import math

import pytest

from ensomi_model.research.oracle_time_continuation.windows import WindowSampler, WindowSamplingPolicy
from ensomi_model.research.scoped_style_modeling.dataset import ContractError
from .conftest import admit


def tap_source(rows, *, group="song:a", offset=0, split="train", gap=100):
    source = admit([(i % 4, offset + i * gap, offset + i * gap) for i in range(rows)])
    return replace(source, identity=replace(source.identity, group_id=group, split=split))


def test_population_path_probabilities_follow_group_chart_feasible_stratum_start_horizon_order():
    sources = (tap_source(580), tap_source(50, offset=1), tap_source(95, group="song:b"),
               tap_source(12, group="song:c"), tap_source(30, group="song:c", offset=2))
    sampler = WindowSampler(sources)
    assert len(sampler.groups) == 2 and len(sampler.charts) == 3
    assert {item["ineligible_reason"] for item in sampler.excluded} == {"fewer-than-30-notes", "no-target-suffix"}
    group_mass, chart_mass, stratum_mass = defaultdict(float), defaultdict(float), defaultdict(float)
    for sha, chart in sampler.charts.items():
        for stratum, starts in chart.starts:
            for start in starts:
                for horizon in range(3):
                    window = sampler.window(sha, start, horizon)
                    expected = 1 / (2 * len(sampler.groups[chart.source.identity.group_id]) * len(chart.starts) * len(starts) * 3)
                    assert window.probability == expected
                    group_mass[chart.source.identity.group_id] += window.probability
                    chart_mass[sha] += window.probability
                    stratum_mass[sha, stratum] += window.probability
    assert all(math.isclose(mass, .5) for mass in group_mass.values())
    for sha, chart in sampler.charts.items():
        expected = .5 / len(sampler.groups[chart.source.identity.group_id])
        assert math.isclose(chart_mass[sha], expected)
        assert all(math.isclose(stratum_mass[sha, s], expected / len(chart.starts)) for s, _ in chart.starts)
    long = sources[0].identity.source_sha256
    for start, stratum in ((30, 0), (93, 0), (94, 1), (541, 1), (542, 2)):
        assert sampler.window(long, start, 0).stratum == stratum


def test_half_open_horizons_keep_all_rows_and_all_horizon_paths_at_the_true_end(chord_source):
    source = tap_source(600, gap=30)
    sampler = WindowSampler((source, chord_source))
    sha = source.identity.source_sha256
    assert sampler.window(sha, 30, 0).target_rows == 34
    assert sampler.window(sha, 30, 1).target_rows == 134
    assert sampler.window(sha, 30, 2).target_rows == 534  # exceeds four full Q=128 chunks
    exact = WindowSampler((tap_source(100),))
    key = next(iter(exact.charts))
    assert exact.window(key, 30, 0).stop == 40  # row exactly at start + 1000ms is excluded
    ends = [sampler.window(sha, 599, horizon) for horizon in range(3)]
    assert [window.target_rows for window in ends] == [1, 1, 1]
    assert all(window.record()["includes_terminal"] for window in ends)
    assert math.isclose(sampler.target_probability(ends[0]), 3 * ends[0].probability)
    chord = sampler.window(chord_source.identity.source_sha256, chord_source.minimum_seed().seed_row_count, 0)
    assert chord.seed.seed_note_count == 33
    assert any(all(action not in (1, 2) for action in row.actions)
               for row in chord_source.targets[chord.start:chord.stop])
    with pytest.raises(ContractError, match="every row"):
        replace(sampler.window(sha, 30, 2), stop=158)


def test_draws_are_repeatable_order_independent_and_rng_is_separate_from_torch():
    sources = (tap_source(80), tap_source(560, group="song:b"))
    first = WindowSampler(sources, WindowSamplingPolicy(seed=29))
    second = WindowSampler(sources[::-1], WindowSamplingPolicy(seed=29))
    assert [first.draw().record() for _ in range(60)] == [second.draw().record() for _ in range(60)]
    saved = first.rng.getstate()
    expected = first.draw().record()
    first.rng.setstate(saved)
    assert first.draw().record() == expected


def test_invalid_populations_and_policies_fail_before_sampling():
    source = tap_source(70)
    heldout = tap_source(80, split="validation")
    with pytest.raises(ContractError, match="multiple splits"):
        WindowSampler((source, heldout))
    with pytest.raises(ContractError, match="duplicate"):
        WindowSampler((source, source))
    with pytest.raises(ContractError, match="No eligible"):
        WindowSampler((tap_source(20),))
    with pytest.raises(ContractError, match="No eligible"):
        WindowSampler((heldout,))
    for horizons in ((1.,), (1., 1., 16.), (0., 4., 16.), (1., 4., float("inf")), (1., 4., float("nan"))):
        with pytest.raises(ContractError, match="horizons"):
            WindowSamplingPolicy(horizons_s=horizons)
    sampler = WindowSampler((source,))
    for start in (0, 29, 70, True):
        with pytest.raises(ContractError, match="Target start"):
            sampler.window(source.identity.source_sha256, start, 0)


def test_shared_chart_batches_keep_group_balance_start_order_and_exact_rng_recovery():
    from collections import Counter
    sources = (tap_source(70), tap_source(80, offset=1), tap_source(560, group='song:b'))
    plain = WindowSampler(sources)
    independent = WindowSampler(sources)
    assert plain.draw_batch(8) == tuple(independent.draw() for _ in range(8))
    shared = WindowSampler(sources, WindowSamplingPolicy(windows_per_chart=4))
    groups = Counter()
    for _ in range(1000):
        batch = shared.draw_batch(8)
        for offset in (0, 4):
            cohort = batch[offset:offset + 4]
            assert len({w.source.identity.source_sha256 for w in cohort}) == 1
            assert [w.start for w in cohort] == sorted(w.start for w in cohort)
            assert all(w.probability == shared.window(w.source.identity.source_sha256, w.start, w.horizon_index).probability
                       for w in cohort)
            groups[cohort[0].source.identity.group_id] += 1
    assert .45 < groups['song:a'] / sum(groups.values()) < .55
    saved = shared.rng.getstate()
    expected = shared.draw_batch(7)
    shared.rng.setstate(saved)
    assert shared.draw_batch(7) == expected
    with pytest.raises(ContractError, match='windows_per_chart'):
        WindowSamplingPolicy(windows_per_chart=9)


def gap_source(*, group='song:a', offset=0):
    times = [offset]
    gaps = {19: 32000, 35: 2000, 38: 2000, 44: 8000, 50: 32000, 53: 64000}
    for index in range(59):
        times.append(times[-1] + gaps.get(index, 100))
    source = admit([(i % 4, time, time) for i, time in enumerate(times)])
    return replace(source, identity=replace(source.identity, group_id=group))


def test_gap_mixture_enumerated_path_mass_and_merged_interval_probabilities():
    sampler = WindowSampler((gap_source(), gap_source(offset=1), gap_source(group='song:b', offset=2)),
                            WindowSamplingPolicy(gap_sampling_probability=.25, gap_context_rows=16))
    intervals, representatives, masses = defaultdict(float), {}, defaultdict(float)

    def collect(window, route):
        key = (window.source.identity.source_sha256, window.start, window.stop)
        intervals[key] += window.probability
        representatives[key] = window
        masses[route] += window.probability

    for sha, chart in sampler.charts.items():
        for _, starts in chart.starts:
            for start in starts:
                for horizon in range(3):
                    collect(sampler.window(sha, start, horizon), 'base')
        assert sampler.gaps[sha] == ((35, 38), (44,), (50, 53))  # seed gap is excluded
        for band, indices in enumerate(sampler.gaps[sha]):
            for index in indices:
                for start in sampler.gap_starts(chart, index):
                    window = sampler.gap_window(sha, band, index, start)
                    assert window.start <= index < window.stop and index - start < 16
                    assert window.horizon_index == 2 and window.record()['sampling_route'] == 'gap'
                    collect(window, band)
    assert masses['base'] == pytest.approx(.75)
    assert all(masses[band] == pytest.approx(.25 / 3) for band in range(3))
    assert sum(intervals.values()) == pytest.approx(1)
    for key, mass in intervals.items():
        assert sampler.target_probability(representatives[key]) == pytest.approx(mass)
    assert [band['gaps'] for band in sampler.gap_population()] == [6, 3, 6]


def test_gap_start_horizon_is_half_open_and_never_labels_a_crop_as_terminal():
    source = gap_source()
    sampler = WindowSampler((source,), WindowSamplingPolicy(gap_sampling_probability=1, gap_context_rows=128))
    sha = source.identity.source_sha256
    # The 16-second horizon may not reach backward across the preceding 32-second gap.
    assert list(sampler.gap_starts(sampler.charts[sha], 53)) == [51, 52, 53]
    window = sampler.gap_window(sha, 2, 53, 51)
    assert window.stop == 54 and not window.record()['includes_terminal']
    assert source.skeleton.times_ms[54] - source.skeleton.times_ms[53] == 64000
    assert sampler.window(sha, 51, 2).probability == 0
    exact = WindowSampler((tap_source(70, gap=2000),), WindowSamplingPolicy(gap_sampling_probability=1))
    chart = next(iter(exact.charts.values()))
    assert list(exact.gap_starts(chart, 50)) == list(range(43, 51))


def test_gap_draws_use_only_times_after_seed_and_resume_the_same_correlated_batches():
    from types import SimpleNamespace
    from collections import Counter

    class UnreadableTargets:
        def __len__(self):
            return 60

        def __getitem__(self, index):
            raise AssertionError('Sampling must not read target actions')

    source = gap_source()
    time_only = SimpleNamespace(identity=source.identity, skeleton=source.skeleton,
                                targets=UnreadableTargets(), minimum_seed=source.minimum_seed)
    policy = WindowSamplingPolicy(seed=71, windows_per_chart=4, gap_sampling_probability=.25)
    sampler, reference = WindowSampler((time_only,), policy), WindowSampler((source,), policy)
    routes = Counter()
    for _ in range(200):
        batch = sampler.draw_batch(8)
        assert [w.record() for w in batch] == [w.record() for w in reference.draw_batch(8)]
        for offset in (0, 4):
            cohort = batch[offset:offset + 4]
            assert len({w.gap_band for w in cohort}) == 1
            assert [w.start for w in cohort] == sorted(w.start for w in cohort)
            routes['base' if cohort[0].gap_band is None else 'gap'] += 1
    assert .18 < routes['gap'] / sum(routes.values()) < .32
    state = sampler.rng.getstate()
    expected = [w.record() for w in sampler.draw_batch(7)]
    sampler.rng.setstate(state)
    assert [w.record() for w in sampler.draw_batch(7)] == expected


def test_gap_policy_validation_and_disabled_route_preserve_existing_draws(monkeypatch):
    for overrides in ({'gap_sampling_probability': True}, {'gap_sampling_probability': 1.1},
                      {'gap_sampling_probability': float('nan')}, {'gap_context_rows': 0},
                      {'gap_context_rows': 129}, {'gap_thresholds_ms': (1, 2)},
                      {'gap_thresholds_ms': (1, 2, 2)}, {'gap_thresholds_ms': (1, 2, float('inf'))}):
        with pytest.raises(ContractError):
            WindowSamplingPolicy(**overrides)
    with pytest.raises(ContractError, match='No eligible skeleton gaps'):
        WindowSampler((tap_source(60),), WindowSamplingPolicy(gap_sampling_probability=.25))
    source = gap_source()
    sampler = WindowSampler((source,), WindowSamplingPolicy(gap_sampling_probability=.25))
    for band, index, start in ((True, 44, 40), (1, 35, 30), (2, 53, 50), (1, 44, True)):
        with pytest.raises(ContractError):
            sampler.gap_window(source.identity.source_sha256, band, index, start)
    monkeypatch.setattr(WindowSampler, '_index_gaps', lambda self: pytest.fail('Disabled gap route must not scan times'))
    plain = WindowSampler((source,))
    explicit = WindowSampler((source,), WindowSamplingPolicy(gap_sampling_probability=0, gap_context_rows=1))
    assert [w.record() for w in plain.draw_batch(8)] == [w.record() for w in explicit.draw_batch(8)]
    assert not plain.gaps and not plain.gap_groups
