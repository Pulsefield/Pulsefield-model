from collections import defaultdict
from dataclasses import replace
import math

import pytest

from pulsefield_model.research.oracle_time_continuation.windows import WindowSampler, WindowSamplingPolicy
from pulsefield_model.research.scoped_style_modeling.dataset import ContractError
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
