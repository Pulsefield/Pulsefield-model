from dataclasses import replace
from itertools import product
import subprocess
import sys

import pytest
import torch

from ensomi_model.research.oracle_time_continuation.config import BackboneConfig
from ensomi_model.research.oracle_time_continuation.engine import ContinuationEngine
from ensomi_model.research.oracle_time_continuation.features import HistoryStatus, clock_features
from ensomi_model.research.oracle_time_continuation.model import CausalBackbone, PreRowEncoding, row_index
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow, TimeSkeleton
from ensomi_model.research.scoped_style_modeling.dataset import ContractError


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_engine(**kwargs):
    torch.manual_seed(72)
    config = BackboneConfig(hidden=16, heads=2, recent=8, coarse_group=3, coarse_capacity=2,
                            attacks_per_lane=3, releases_per_lane=2)
    return ContinuationEngine(CausalBackbone(replace(config, **kwargs)))


def mixed_rows(count=40):
    rows = [CompleteRow(0, (2, 1, 0, 0))]
    for index in range(1, count - 1):
        actions = [0, 0, 0, 0]
        actions[1 + index % 3] = 1
        if index % 5 == 0:
            actions[1] = 1
        rows.append(CompleteRow(index * 17 + (90000 if index > 20 else 0), tuple(actions)))
    rows.append(CompleteRow(rows[-1].time_ms + 200, (3, 0, 0, 0)))
    return tuple(rows)


def skeleton(rows):
    return TimeSkeleton(tuple(row.time_ms for row in rows))


def randomize_optional_readouts(engine):
    if engine.model.timing is not None:
        torch.nn.init.normal_(engine.model.timing.projection[-1].weight, std=.1)
    if engine.model.head.clock_readout is not None:
        torch.nn.init.normal_(engine.model.head.clock_readout.projection[-1].weight, std=.1)


def assert_learned_equal(first, second, *, atol=3e-6):
    assert first.execution == second.execution
    assert first.local.pace == second.local.pace
    for left, right in zip(first.local.latest, second.local.latest):
        assert left.support == right.support
        torch.testing.assert_close(left.value, right.value, atol=atol, rtol=2e-5)
    assert first.relation.visible_ids == second.relation.visible_ids
    assert first.relation.attacks == second.relation.attacks
    assert first.relation.releases == second.relation.releases
    assert first.relation.active_heads == second.relation.active_heads
    for left, right in zip(first.relation.nodes, second.relation.nodes):
        assert left.closed_heads == right.closed_heads
        torch.testing.assert_close(left.payload, right.payload, atol=atol, rtol=2e-5)
    assert tuple(t.position for t in first.temporal.tokens) == tuple(t.position for t in second.temporal.tokens)
    for left, right in zip(first.temporal.tokens, second.temporal.tokens):
        for a, b in zip(left.inputs, right.inputs):
            torch.testing.assert_close(a, b, atol=atol, rtol=2e-5)


@pytest.mark.parametrize('time_lookahead_rows,clock_readout_hidden', [(0, 0), (16, 0), (0, 12), (16, 12)])
def test_step_dense_ragged_batch_and_chunk_partition_agree(chord_source, time_lookahead_rows, clock_readout_hidden):
    engine = make_engine(time_lookahead_rows=time_lookahead_rows, clock_readout_hidden=clock_readout_hidden)
    randomize_optional_readouts(engine)
    rows = mixed_rows()
    prefix = 7
    with torch.no_grad():
        initial = engine.prefill(skeleton(rows), rows[:prefix])
        step, predictions = initial, []
        relation_queries, relation_contents = [], []
        for row in rows[prefix:]:
            predictions.append(engine.predict(step).log_probs)
            relation_queries.append(step.relation.visible_ids)
            step = engine.commit(step, row)
            relation_contents.append(step.relation.visible_ids)
        expected = torch.stack(predictions)
        for width in (1, 7, 17, 64, 128):
            state, outputs = initial, []
            for start in range(prefix, len(rows), width):
                result = engine.teacher_force(state, rows[start:start + width])
                offset = start - prefix
                assert result.relation_query_ids == tuple(relation_queries[offset:offset + width])
                assert result.relation_content_ids == tuple(relation_contents[offset:offset + width])
                state = result.state
                outputs.append(result.log_probs)
            torch.testing.assert_close(torch.cat(outputs), expected, atol=3e-6, rtol=2e-5)
            assert_learned_equal(state, step)
        other = engine.prefill(chord_source.skeleton, chord_source.targets[:2])
        batch = engine.teacher_force_batch((initial, other, initial),
                                           (rows[prefix:], chord_source.targets[2:6], ()))
        torch.testing.assert_close(batch.log_probs[0], expected, atol=3e-6, rtol=2e-5)
        assert batch.valid.sum(1).tolist() == [len(expected), 4, 0]
        assert not batch.legal[~batch.valid].any()
        assert not batch.log_probs[~batch.valid].any()
        assert batch.states[2] is initial
        assert batch.states[1].execution.next_index == 6


@pytest.mark.parametrize('time_lookahead_rows,clock_readout_hidden', [(0, 0), (16, 0), (0, 12), (16, 12)])
def test_current_and_future_targets_cannot_change_pre_row_distribution(time_lookahead_rows, clock_readout_hidden):
    engine = make_engine(time_lookahead_rows=time_lookahead_rows, clock_readout_hidden=clock_readout_hidden)
    randomize_optional_readouts(engine)
    rows = mixed_rows(18)
    changed = list(rows)
    changed[6] = CompleteRow(rows[6].time_ms, (0, 1, 1, 1))
    changed[7] = CompleteRow(rows[7].time_ms, (3, 0, 0, 0))
    changed[-1] = CompleteRow(rows[-1].time_ms, (1, 0, 0, 0))
    with torch.no_grad():
        initial = engine.prefill(skeleton(rows), rows[:3])
        first = engine.teacher_force(initial, rows[3:])
        second = engine.teacher_force(initial, changed[3:])
        torch.testing.assert_close(first.log_probs[:4], second.log_probs[:4], atol=0, rtol=0)
        assert not torch.allclose(first.log_probs[4].exp(), second.log_probs[4].exp())
        assert initial.execution.replay.open_ln_start_ms[0] == 0
        # The action prefix is identical; the optional time-only context changes.
        altered = TimeSkeleton(tuple(t if i <= 3 else t + 1234 for i, t in enumerate(skeleton(rows).times_ms)))
        same_prefix = engine.prefill(altered, rows[:3])
        if time_lookahead_rows:
            assert not torch.allclose(engine.predict(initial).log_probs, engine.predict(same_prefix).log_probs)
        else:
            torch.testing.assert_close(engine.predict(initial).log_probs, engine.predict(same_prefix).log_probs,
                                       atol=0, rtol=0)


def test_predict_is_pure_and_commit_updates_all_history_paths_atomically():
    engine = make_engine()
    rows = mixed_rows()
    with torch.no_grad():
        state = engine.prefill(skeleton(rows), rows[:8])
        original = state.detached()
        first = engine.predict(state)
        torch.testing.assert_close(first.log_probs, engine.predict(state).log_probs, atol=0, rtol=0)
        assert_learned_equal(state, original, atol=0)
        with pytest.raises(ContractError, match="occupancy"):
            engine.commit(state, CompleteRow(rows[8].time_ms, (1, 2, 0, 0)))
        assert_learned_equal(state, original, atol=0)
        chosen = CompleteRow(rows[8].time_ms, (3, 2, 1, 0))
        changed = engine.commit(state, chosen)
        assert changed.execution.replay.occupancy == (False, True, False, False)
        assert changed.local.pace.gaps_ms == state.local.pace.gaps_ms + (17.,)
        assert changed.local.latest[0].row == chosen
        assert changed.relation.active_heads == (None, 9, None, None)
        assert changed.relation.nodes[-1].closed_heads == (1, None, None, None)
        assert changed.temporal.row_count == 9
        assert not torch.equal(changed.temporal.recent[-1].inputs[0], state.temporal.recent[-1].inputs[0])


@pytest.mark.parametrize('time_lookahead_rows,clock_readout_hidden', [(0, 0), (16, 0), (0, 12), (16, 12)])
def test_mirror_equivariance_including_coarse_memory_and_terminal_support(time_lookahead_rows, clock_readout_hidden):
    engine = make_engine(time_lookahead_rows=time_lookahead_rows, clock_readout_hidden=clock_readout_hidden)
    randomize_optional_readouts(engine)
    rows = mixed_rows()
    mirrored = tuple(CompleteRow(row.time_ms, row.actions[::-1]) for row in rows)
    with torch.no_grad():
        first = engine.teacher_force(engine.start(skeleton(rows)), rows)
        second = engine.teacher_force(engine.start(skeleton(rows)), mirrored)
    permutation = torch.tensor([64 * d + 16 * c + 4 * b + a for a, b, c, d in product(range(4), repeat=4)])
    assert torch.equal(first.legal, second.legal[:, permutation])
    torch.testing.assert_close(first.log_probs, second.log_probs[:, permutation], atol=4e-6, rtol=2e-5)
    for a, b in zip(first.state.temporal.tokens, second.state.temporal.tokens):
        for left, right in zip(a.inputs, b.inputs):
            torch.testing.assert_close(left, right.flip(0), atol=4e-6, rtol=2e-5)


def test_zero_time_projection_preserves_compatible_logits_and_receives_training_gradient():
    baseline = make_engine()
    anchored = make_engine(time_lookahead_rows=16)
    for name, value in baseline.model.state_dict().items():
        torch.testing.assert_close(value, anchored.model.state_dict()[name], rtol=0, atol=0)
    loaded = anchored.model.load_state_dict(baseline.model.state_dict(), strict=False)
    assert loaded.missing_keys == ['timing.projection.0.weight', 'timing.projection.0.bias',
                                   'timing.projection.2.weight', 'timing.projection.2.bias']
    assert not loaded.unexpected_keys
    rows = mixed_rows(80)
    prefix = 50
    with torch.no_grad():
        a = baseline.prefill(skeleton(rows), rows[:prefix])
        b = anchored.prefill(skeleton(rows), rows[:prefix])
        torch.testing.assert_close(baseline.predict(a).log_probs, anchored.predict(b).log_probs, atol=0, rtol=0)
    result = anchored.teacher_force(b, rows[prefix:])
    indices = torch.tensor([row_index(row.actions) for row in rows[prefix:]])
    (-result.log_probs.gather(1, indices[:, None]).sum()).backward()
    grad = anchored.model.timing.projection[-1].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_zero_clock_readout_preserves_pretrained_function_and_receives_gradient():
    baseline = make_engine(time_lookahead_rows=16)
    randomize_optional_readouts(baseline)
    candidate = make_engine(time_lookahead_rows=16, clock_readout_hidden=12)
    loaded = candidate.model.load_state_dict(baseline.model.state_dict(), strict=False)
    assert loaded.missing_keys and all(name.startswith('head.clock_readout.') for name in loaded.missing_keys)
    assert not loaded.unexpected_keys
    rows = mixed_rows(80)
    with torch.no_grad():
        a = baseline.prefill(skeleton(rows), rows[:50])
        b = candidate.prefill(skeleton(rows), rows[:50])
        torch.testing.assert_close(baseline.predict(a).log_probs, candidate.predict(b).log_probs, atol=0, rtol=0)
    result = candidate.teacher_force(b, rows[50:])
    targets = torch.tensor([row_index(row.actions) for row in rows[50:]])
    (-result.log_probs.gather(1, targets[:, None]).sum()).backward()
    grad = candidate.model.head.clock_readout.projection[-1].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_clock_readout_uses_physical_intervals_and_future_times_without_absolute_timestamp():
    engine = make_engine(time_lookahead_rows=16, clock_readout_hidden=12)
    randomize_optional_readouts(engine)
    rows = mixed_rows()
    shifted = tuple(replace(row, time_ms=row.time_ms + 1_000_000_000) for row in rows)
    with torch.no_grad():
        state = engine.prefill(skeleton(rows), rows[:10])
        translated = engine.prefill(skeleton(shifted), shifted[:10])
        query = state.execution.query(16)
        other = translated.execution.query(16)
        readout = engine.model.head.clock_readout
        torch.testing.assert_close(readout((query,)), readout((other,)), rtol=0, atol=0)
        altered = (query, replace(query, time_ms=rows[9].time_ms + 250),
                   replace(query, future_offsets_ms=tuple(4 * x for x in query.future_offsets_ms)))
        hidden = torch.zeros(3, 2, engine.model.config.temporal_hidden)
        probabilities, legal = engine.model.head(PreRowEncoding(hidden, altered))
        assert torch.equal(legal[0], legal[1]) and torch.equal(legal[0], legal[2])
        assert not torch.allclose(probabilities[0], probabilities[1])
        assert not torch.allclose(probabilities[0], probabilities[2])


@pytest.mark.parametrize("occupied", list(product((False, True), repeat=4)))
def test_head_masks_match_exact_replay_for_all_occupancies(occupied):
    engine = make_engine()
    with torch.no_grad():
        state = engine.start(TimeSkeleton((0, 10, 20)))
        actions = tuple(2 if held else 0 for held in occupied) if any(occupied) else (1, 0, 0, 0)
        state = engine.commit(state, CompleteRow(0, actions))
        for terminal in (False, True):
            current = state if not terminal else replace(state, execution=replace(state.execution, skeleton=TimeSkeleton((0, 10))))
            distribution = engine.predict(current)
            expected = {row_index(row) for row in current.execution.query().legal_actions}
            assert set(distribution.legal.nonzero().flatten().tolist()) == expected
            torch.testing.assert_close(distribution.log_probs.exp().sum(), torch.tensor(1.))
            assert distribution.log_probs[0].isneginf()
        with pytest.raises(ContractError, match="pre-row"):
            engine.model.head(torch.zeros(1, 2, 16))


def test_bos_short_history_pace_and_float64_time_differences(chord_source):
    engine = make_engine()
    with torch.no_grad():
        state = engine.start(chord_source.skeleton)
        assert not state.local.latest and state.local.pace.mean_ms is None
        like = next(engine.model.parameters())
        features = clock_features((None, 0, .01), like)
        assert not features[0].any() and features[1, -1] == 1
        assert not torch.equal(features[1], features[2])
        for index, row in enumerate(chord_source.targets):
            state = engine.commit(state, row)
            for entry, size in zip(state.local.latest, (3, 7, 15)):
                expected = tuple(range(max(1, index + 2 - size), index + 2))
                assert tuple(identity for identity, _ in entry.support) == expected
                assert entry.count == len(expected)
                assert entry.status == (HistoryStatus.TRUNCATED if expected[0] > 1 else HistoryStatus.PRESENT)
            assert [len(buf) for buf in state.local.buffers] == [min(index + 1, size) for size in (2, 4, 8)]
        times = [float(i * 100) for i in range(32)] + [95100., 95200.]
        rows = [CompleteRow(time, (1, 0, 0, 0)) for time in times]
        state = engine.prefill(skeleton(rows), rows[:32])
        assert state.local.pace.mean_ms == 100
        engine.predict(state)
        assert state.local.pace.mean_ms == 100
        state = engine.commit(state, rows[32])
        assert len(state.local.pace.gaps_ms) == 32 and state.local.pace.mean_ms == 95100 / 32
        origin = 2. ** 40
        small = (CompleteRow(0, (2, 0, 0, 0)), CompleteRow(.125, (0, 1, 0, 0)), CompleteRow(.25, (3, 0, 0, 0)))
        shifted = tuple(CompleteRow(row.time_ms + origin, row.actions) for row in small)
        a = engine.teacher_force(engine.start(skeleton(small)), small)
        b = engine.teacher_force(engine.start(skeleton(shifted)), shifted)
        torch.testing.assert_close(a.log_probs, b.log_probs, atol=0, rtol=0)


def test_prefill_is_content_only_and_active_ln_survives_detachment_and_eviction(monkeypatch):
    engine = make_engine(recent=4, coarse_group=2, coarse_capacity=1)
    rows = mixed_rows(70)
    def fail(*args, **kwargs):
        raise AssertionError("prefill must not compute a query or head")
    with monkeypatch.context() as patch:
        patch.setattr(engine.model, "query_input", fail)
        patch.setattr(engine.model.head, "forward", fail)
        state = engine.prefill(skeleton(rows), rows[:4])
    saved_head = state.relation.nodes[0].payload.clone()
    for start in range(4, 68, 8):
        state = engine.teacher_force(state, rows[start:min(68, start + 8)]).state.detached()
        assert state.execution.replay.open_ln_start_ms[0] == 0
        assert state.relation.active_heads[0] == 1
        assert len(state.relation.nodes) <= engine.model.config.relation_capacity
        assert len(state.temporal.recent) <= engine.model.config.recent_capacity
        assert len(state.temporal.coarse) <= 1
        torch.testing.assert_close(state.relation.nodes[0].payload, saved_head, atol=0, rtol=0)
        tensors = [node.payload for node in state.relation.nodes]
        tensors += [entry.value for buf in state.local.buffers for entry in buf]
        tensors += [entry.value for entry in state.local.latest]
        tensors += [value for token in state.temporal.tokens for value in token.inputs]
        for value in tensors:
            assert not value.requires_grad and value.grad_fn is None and value._base is None
            assert value.untyped_storage().nbytes() == value.numel() * value.element_size()


def test_parameter_and_inference_cache_lifecycle():
    engine = make_engine()
    rows = mixed_rows(15)
    engine.model.eval()
    with torch.no_grad():
        training = engine.prefill(skeleton(rows), rows[:8])
        cached = engine.prefill(skeleton(rows), rows[:8], inference=True)
        torch.testing.assert_close(engine.predict(training).log_probs, engine.predict(cached).log_probs,
                                   atol=2e-6, rtol=2e-5)
        a = engine.teacher_force(training, rows[8:])
        b = engine.teacher_force(cached, rows[8:])
        torch.testing.assert_close(a.log_probs, b.log_probs, atol=2e-6, rtol=2e-5)
        assert_learned_equal(a.state, b.state)
        assert all(token.projected is not None for token in b.state.temporal.tokens)
    with pytest.raises(ContractError, match="gradients disabled"):
        engine.predict(cached)
    with torch.no_grad():
        next(engine.model.parameters()).add_(.001)
        with pytest.raises(ContractError, match="replay the prefix"):
            engine.predict(training)
        with pytest.raises(ContractError, match="replay the prefix"):
            engine.commit(cached, rows[8])


def test_configuration_and_chunk_bounds():
    with pytest.raises(ContractError, match="positive integer"):
        BackboneConfig(recent=True)
    with pytest.raises(ContractError, match="divisible"):
        BackboneConfig(hidden=15)
    with pytest.raises(ContractError, match="128"):
        BackboneConfig(max_chunk=129)
    engine = make_engine(max_chunk=3)
    rows = mixed_rows(6)
    state = engine.start(skeleton(rows))
    with pytest.raises(ContractError, match="max_chunk"):
        engine.teacher_force(state, rows)
    with pytest.raises(ContractError, match="matching"):
        engine.teacher_force_batch((state,), ())
    for actions in ((0, 0, 0, 0), (-1, 4, 0, 1), (1, 0, 0), (True, 0, 0, 0)):
        with pytest.raises(ContractError):
            row_index(actions)


def test_backbone_does_not_import_masked_or_legacy_model_paths():
    script = """
import sys
from ensomi_model.research.oracle_time_continuation.model import CausalBackbone
for module in sys.modules:
    assert not module.startswith(('ensomi_model.models', 'ensomi_model.timing',
                                  'ensomi_model.inference', 'ensomi_model.training')), module
    assert module not in ('ensomi_model.research.source_action_modeling.model',
                          'ensomi_model.research.source_action_modeling.observation',
                          'ensomi_model.research.source_action_modeling.tensors',
                          'ensomi_model.research.source_action_modeling.local_representation'), module
"""
    subprocess.run([sys.executable, "-c", script], check=True)
