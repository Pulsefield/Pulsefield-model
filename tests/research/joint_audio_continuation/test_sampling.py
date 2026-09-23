from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ensomi_model.research.bounded_typed_continuation.data import SourceChart
from ensomi_model.research.joint_audio_continuation.data import JointChart, query
from ensomi_model.research.joint_audio_continuation.sampling import (
    coverage_examples, draw_examples, full_gap_probes, full_wait, sample_example,
)
from ensomi_model.research.joint_audio_continuation.timing import hazard_nll
from ensomi_model.research.oracle_time_continuation.data import SourceIdentity
from ensomi_model.research.oracle_time_continuation.storage import ROW_DTYPE


def config(enabled=True):
    return SimpleNamespace(full_wait_supervision=enabled, history_rows=511, timing_horizon_ms=4000)


def chart(*, first=0, group='group', split='train', release=13033, source_sha='a' * 64):
    rows = np.zeros(7, ROW_DTYPE)
    rows['time'] = np.asarray([0, 17, 9033, release, release + 1, 17035, 21035]) + first
    rows['actions'] = [(2, 0, 0, 0), (0, 1, 0, 0), (0, 0, 2, 1), (3, 0, 0, 0),
                       (0, 1, 0, 0), (0, 0, 3, 0), (1, 0, 0, 0)]
    identity = SourceIdentity(source_sha, 'b' * 64, group, split)
    source = SourceChart(identity, rows, minimum_seed_notes=1)
    return JointChart(asdict(identity), source, np.zeros((3000, 128), np.float32), 30000)


@pytest.mark.parametrize('target_index', range(7))
def test_full_wait_partitions_exact_native_interval_and_keeps_one_event(target_index):
    source = chart()
    example = full_wait(source, target_index, config())
    previous = int(source.source.rows['time'][target_index - 1]) if target_index else -1
    target = int(source.source.rows['time'][target_index])
    assert example.chart is source
    assert example.queries[0].cursor_ms == previous
    assert example.queries[-1].horizon_end_ms == target
    assert example.queries[-1].target_index == target_index
    assert example.queries[-1].target_time_ms == target
    assert all(q.censored for q in example.queries[:-1])
    assert not example.queries[-1].censored
    assert all(0 < q.horizon_end_ms - q.cursor_ms <= 4000 for q in example.queries)
    assert all(left.horizon_end_ms == right.cursor_ms for left, right in zip(example.queries, example.queries[1:]))
    assert sum(q.horizon_end_ms - q.cursor_ms for q in example.queries) == target - previous
    assert all(q.source_index == target_index for q in example.queries)
    assert all(q.replay == example.queries[0].replay for q in example.queries)
    if target - previous > 4000:
        assert example.queries[-1].horizon_end_ms - example.queries[-1].cursor_ms == 4000
    if target_index == 0:
        assert [(q.cursor_ms, q.horizon_end_ms, q.target_time_ms) for q in example.queries] == [(-1, 0, 0)]


def test_summed_chunk_likelihood_and_gradients_equal_complete_native_wait():
    example = full_wait(chart(), 2, config())
    assert [(q.cursor_ms, q.horizon_end_ms) for q in example.queries] == [(17, 4017), (4017, 5033), (5033, 9033)]
    length = 9033 - 17
    logits = torch.linspace(-9., -4., length, dtype=torch.float64, requires_grad=True)[None]
    valid = torch.ones_like(logits, dtype=torch.bool)
    forced = torch.zeros_like(valid)
    entire = hazard_nll(logits, valid, torch.tensor([length - 1]), forced)
    parts, start = [], 0
    for q in example.queries:
        stop = start + q.horizon_end_ms - q.cursor_ms
        target = -1 if q.censored else stop - start - 1
        parts.append(hazard_nll(logits[:, start:stop], valid[:, start:stop], torch.tensor([target]), forced[:, start:stop]))
        start = stop
    summed = torch.stack(parts).sum(0)
    torch.testing.assert_close(summed, entire, atol=1e-12, rtol=1e-12)
    a = torch.autograd.grad(summed.sum(), logits, retain_graph=True)[0]
    b = torch.autograd.grad(entire.sum(), logits)[0]
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_full_wait_retains_old_hold_without_revealing_later_endpoint():
    a, b = full_wait(chart(release=13033), 2, config()), full_wait(chart(release=15033), 2, config())
    for left, right in zip(a.queries, b.queries):
        assert left.replay == right.replay
        assert left.replay.open_ln_start_ms[0] == 0.
        assert not left.replay.is_complete
        np.testing.assert_array_equal(left.raw, right.raw)
        assert left.target_time_ms == right.target_time_ms
        assert left.target_actions == right.target_actions
    assert a.queries[0].censored and a.queries[-1].target_actions == (0, 0, 2, 1)


def legacy_draw(groups, rng, count):
    """Frozen random-call sequence and cursor rules of the 09b919c sampler."""
    result = []
    for _ in range(count):
        group = groups[int(rng.integers(len(groups)))]
        source = group[int(rng.integers(len(group)))]
        choice, times = float(rng.random()), source.source.rows['time']
        if choice < .08:
            cursor = -1
        elif choice < .78:
            target = int(rng.integers(len(times)))
            cursor = int(times[target - 1]) if target else -1
        elif choice < .95:
            cursor = int(rng.integers(-1, source.duration_ms))
        else:
            cursor = int(rng.integers(min(int(times[-1]), source.duration_ms - 1), source.duration_ms))
        result.append((source, query(source, cursor)))
    return result


def test_disabled_mode_exactly_preserves_legacy_rng_chart_cursor_and_target():
    groups = [[chart(), chart(first=5000, source_sha='c' * 64)], [chart(group='other', source_sha='d' * 64)]]
    expected_rng = np.random.default_rng(230923)
    actual_rng = np.random.default_rng(230923)
    expected = legacy_draw(groups, expected_rng, 200)
    actual = draw_examples(groups, actual_rng, config(False), 200)
    assert actual_rng.bit_generator.state == expected_rng.bit_generator.state
    for (source, q), example in zip(expected, actual):
        assert example.chart is source and len(example.queries) == 1
        got = example.queries[0]
        assert (got.cursor_ms, got.horizon_end_ms, got.target_time_ms, got.target_actions) == (
            q.cursor_ms, q.horizon_end_ms, q.target_time_ms, q.target_actions)
        assert got.replay == q.replay
        np.testing.assert_array_equal(got.raw, q.raw)


class BranchRng:
    def __init__(self, choice, integer=None):
        self.choice, self.integer, self.calls = choice, integer, []

    def random(self):
        self.calls.append(('random',))
        return self.choice

    def integers(self, *args):
        self.calls.append(('integers', *args))
        assert self.integer is not None
        return self.integer


@pytest.mark.parametrize('choice,integer,expanded', [(.01, None, True), (.3, 2, True),
                                                    (.85, 500, False), (.98, 28000, False)])
def test_enabled_mode_only_expands_selected_bos_and_event_prefix_without_extra_draws(choice, integer, expanded):
    source = chart(first=5000)
    old_rng, new_rng = BranchRng(choice, integer), BranchRng(choice, integer)
    before = sample_example(source, old_rng, config(False))
    after = sample_example(source, new_rng, config(True))
    assert old_rng.calls == new_rng.calls
    assert after.chart is before.chart
    if expanded:
        assert len(after.queries) > 1
        assert before.queries[0].cursor_ms == after.queries[0].cursor_ms
        assert after.queries[-1].target_index == (0 if choice < .08 else integer)
    else:
        assert len(after.queries) == 1
        a, b = before.queries[0], after.queries[0]
        assert (a.cursor_ms, a.horizon_end_ms, a.target_index) == (b.cursor_ms, b.horizon_end_ms, b.target_index)
        assert a.replay == b.replay
        np.testing.assert_array_equal(a.raw, b.raw)


def test_coverage_and_probe_sets_are_separate_deterministic_complete_targets():
    train = chart(first=5000)
    other = chart(group='earlier', source_sha='c' * 64)
    val = chart(first=5000, group='validation', split='validation', source_sha='d' * 64)
    # A TEST placeholder must be filtered before its source is inspected.
    forbidden = replace(val, entry=dict(val.entry, split='test'), source=None)
    inputs = [val, train, forbidden, other]
    covered = coverage_examples(inputs, config(False))
    reversed_coverage = coverage_examples(inputs[::-1], config(False))
    identities = lambda examples: [(e.chart.group_id, e.chart.entry['source_sha256'], e.queries[-1].target_index)
                                   for e in examples]
    expected = []
    for source in (train, other):
        indices = np.flatnonzero(np.diff(np.r_[-1, source.source.rows['time']]) > 4000)
        expected.extend((source.group_id, source.entry['source_sha256'], int(index)) for index in indices)
    assert identities(covered) == identities(reversed_coverage) == sorted(expected)
    assert all(e.chart.split == 'train' and not e.queries[-1].censored for e in covered)
    probes = full_gap_probes(inputs, 'validation', config())
    assert probes and all(e.chart is val and not e.queries[-1].censored for e in probes)
    assert any(e.queries[-1].target_index == 0 for e in covered)
    with pytest.raises(ValueError, match='split'):
        full_gap_probes(inputs, 'test', config())
