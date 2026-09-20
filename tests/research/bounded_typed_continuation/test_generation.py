from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
import torch

from pulsefield_model.research.bounded_typed_continuation.contract import Arm, Schedule, Timing
from pulsefield_model.research.bounded_typed_continuation.generation import RawEvent, Rollout
from pulsefield_model.research.bounded_typed_continuation.data import SourceChart
from pulsefield_model.research.bounded_typed_continuation.features import query_features
from pulsefield_model.research.bounded_typed_continuation.model import BoundedModel, ModelConfig
from pulsefield_model.research.bounded_typed_continuation.verification import verify_complete
from pulsefield_model.research.oracle_time_continuation.data import SourceIdentity, admit_source
from pulsefield_model.research.oracle_time_continuation.export import export_osu
from pulsefield_model.research.oracle_time_continuation.runtime import ResourceConfig
from pulsefield_model.research.oracle_time_continuation.schema import CompleteRow
from pulsefield_model.research.oracle_time_continuation.storage import ROW_DTYPE
from pulsefield_model.research.scoped_style_modeling.dataset import ContractError


def setup(arm, device='cpu', availability='none', consequence='none', seed_context='none', long_memory='none'):
    torch.manual_seed(17)
    model = BoundedModel(ModelConfig(arm, hidden=8, levels=2, coupling_rank=2,
                                   endpoint_availability=availability, row_consequence=consequence,
                                   seed_context=seed_context, long_memory=long_memory,
                                   **({'memory_hidden': 12, 'memory_stride': 4} if long_memory != 'none' else {}))).to(device).eval()
    if model.pointer is not None and model.pointer.availability_residual is not None:
        torch.nn.init.normal_(model.pointer.availability_residual[-1].weight, std=.05)
    if model.row_consequence is not None:
        torch.nn.init.normal_(model.row_consequence.output.weight, std=.05)
    if model.seed_residual is not None:
        torch.nn.init.normal_(model.seed_residual[-1].weight, std=.05)
    if model.long_memory is not None:
        torch.nn.init.normal_(model.long_memory.output.weight, std=.05)
    timing = Timing(tuple(100. * i for i in range(37)), None if arm == Arm.R0 else tuple(i % 5 != 4 for i in range(37)))
    seed = [CompleteRow(0., (2, 2, 0, 0)), CompleteRow(100., (0, 0, 1, 0)), CompleteRow(200., (0, 0, 0, 1))]
    rollout = Rollout.from_seed(model, timing, seed, None if arm == Arm.R0 else {0: 10, 1: 17})
    return model, timing, seed, rollout


@pytest.mark.parametrize('arm,availability,consequence', [(arm, 'none', 'none') for arm in Arm] +
                         [(Arm.O1, 'commitment', 'none'), (Arm.R1, 'none', 'actions'), (Arm.R1, 'none', 'frontier')])
@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_native_rollout_preserves_task_support_and_round_trips_osu(arm, availability, consequence, device, tmp_path):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    model, timing, seed, rollout = setup(arm, device, availability, consequence)
    generator = torch.Generator().manual_seed(71)
    rows, decisions = list(seed), []
    while not rollout.state.finished:
        step = rollout.step(generator, candidate_budget=5)
        decisions.append(step)
        if step.row is not None:
            rows.append(step.row)
        if arm == Arm.R0:
            assert step.row is not None
        else:
            assert bool(step.row and any(a in (1, 2) for a in step.row.actions)) == timing.onsets[step.candidate_index]
    assert not any(rollout.state.replay.occupancy)
    assert rollout.state.replay.row_count == len(rows)
    plans = {(step.candidate_index, lane): end for step in decisions for lane, end in step.endpoints.items()}
    checked = verify_complete(rows, timing, arm, seed, None if arm == Arm.R0 else {0: 10, 1: 17}, plans)
    assert checked['rows'] == len(rows) and checked['heads'] == rollout.state.replay.note_count
    if arm != Arm.R0:
        assert next(row.time_ms for row in rows if row.actions[0] == 3) == 1000.
        assert next(row.time_ms for row in rows if row.actions[1] == 3) == 1700.
    if arm == Arm.O1:
        assert any(step.endpoints for step in decisions)
        for step in decisions:
            for lane, end in step.endpoints.items():
                close = next(row.time_ms for row in rows if row.time_ms > step.row.time_ms and row.actions[lane] == 3)
                assert close == timing.times_ms[end]
    path, destination = tmp_path / 'rows.jsonl', tmp_path / 'generated.osu'
    path.write_text(''.join(json.dumps(dict(event_id=i, time_ms=row.time_ms, actions=row.actions)) + '\n' for i, row in enumerate(rows)))
    report = export_osu(path, destination, [row.time_ms for row in rows], ResourceConfig(output_max_bytes=1024 ** 2))
    data = destination.read_bytes()
    reparsed = admit_source(data, hashlib.sha256(data).hexdigest(), group_id='generated-synthetic', split='train')
    assert reparsed.targets == tuple(rows) and report['notes'] == rollout.state.replay.note_count


@pytest.mark.parametrize('arm,availability,consequence', [(arm, 'none', 'none') for arm in Arm] +
                         [(Arm.O1, 'commitment', 'none'), (Arm.R1, 'none', 'actions'), (Arm.R1, 'none', 'frontier')])
@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_owned_raw_checkpoint_rebuild_preserves_rng_actions_and_plans(arm, availability, consequence, device, tmp_path):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    model, timing, _, rollout = setup(arm, device, availability, consequence)
    generator = torch.Generator().manual_seed(91)
    for _ in range(10):
        rollout.step(generator, candidate_budget=3)
    assert rollout.state.index == 13
    assert len(rollout.history) == model.temporal.config.receptive_tokens
    if arm != Arm.R0:
        assert rollout.state.replay.open_ln_start_ms[1] == 0.
        assert rollout.history[0].row.time_ms > 0.
    snapshot = rollout.snapshot(generator)
    path = tmp_path / 'resume.pt'
    torch.save(snapshot, path)
    restored, copied_rng = Rollout.restore(model, timing, torch.load(path, weights_only=True))
    torch.testing.assert_close(model.temporal.read(rollout.cache), model.temporal.read(restored.cache), atol=0, rtol=0)
    assert restored.state == rollout.state
    while not rollout.state.finished:
        a = rollout.step(generator, candidate_budget=3)
        b = restored.step(copied_rng, candidate_budget=11)
        assert a.row == b.row and a.endpoints == b.endpoints
        assert a.row_or_head_log_prob == pytest.approx(b.row_or_head_log_prob, abs=2e-6)
        assert a.endpoint_log_prob == pytest.approx(b.endpoint_log_prob, abs=2e-6)
    assert restored.state == rollout.state
    assert torch.equal(generator.get_state(), copied_rng.get_state())
    # Snapshot ownership survives continued generation and fresh restoration.
    again, _ = Rollout.restore(model, timing, snapshot)
    assert again.state.index == 13


def test_deterministic_skips_do_not_write_history_or_consume_rng():
    model = BoundedModel(ModelConfig(Arm.O1, hidden=8, levels=2, coupling_rank=2)).eval()
    timing = Timing((0., 50., 100.), (True, False, True))
    rollout = Rollout.from_seed(model, timing, [CompleteRow(0., (1, 0, 0, 0))])
    generator = torch.Generator().manual_seed(5)
    state, cache, random = rollout.state.replay, rollout.cache, generator.get_state().clone()
    step = rollout.step(generator)
    assert step.row is None and rollout.state.index == 2
    assert rollout.state.replay is state and rollout.cache is cache
    assert torch.equal(random, generator.get_state())
    assert step.row_or_head_log_prob == step.endpoint_log_prob == 0.


def test_unused_terminal_candidate_finishes_without_inventing_a_row():
    model = BoundedModel(ModelConfig(Arm.O1, hidden=8, levels=2, coupling_rank=2)).eval()
    timing = Timing((0., 50., 100.), (True, False, False))
    seed = [CompleteRow(0., (1, 0, 0, 0))]
    rollout = Rollout.from_seed(model, timing, seed)
    generator = torch.Generator().manual_seed(5)
    assert rollout.step(generator).row is None and rollout.step(generator).row is None
    assert rollout.state.finished and not rollout.state.replay.is_complete
    restored, _ = Rollout.restore(model, timing, rollout.snapshot(generator))
    assert restored.state.finished
    assert verify_complete(seed, timing, Arm.O1, seed)['skipped_candidates'] == 2
    with pytest.raises(StopIteration):
        restored.step(generator)


@pytest.mark.parametrize('availability', ['none', 'commitment'])
def test_selected_object_log_probability_uses_the_same_training_mixture(availability):
    model, timing, _, rollout = setup(Arm.O1, availability=availability)
    generator = torch.Generator().manual_seed(71)
    checked = 0
    while not rollout.state.finished:
        state = rollout.state
        if timing.onsets[state.index]:
            hands, logp = rollout.prediction()
        step = rollout.step(generator, candidate_budget=2)
        if step.endpoints:
            group = tuple(a if a in (1, 2) else 0 for a in step.row.actions)
            assert step.row_or_head_log_prob == pytest.approx(float(logp[model.choices.index(group)]), abs=1e-7)
            expected = model.endpoint_log_probs(hands[None], [state], [group], [step.endpoints], [rollout.view], candidate_budget=1000)
            assert step.endpoint_log_prob == pytest.approx(float(expected.detach()[0]), abs=2e-6)
            checked += 1
    assert checked > 0


def test_recovery_rejects_missing_history_wrong_conditions_and_stale_weights():
    model, timing, _, rollout = setup(Arm.O1)
    generator = torch.Generator().manual_seed(91)
    saved = rollout.snapshot(generator)
    missing = deepcopy(saved)
    missing['history'].pop(0)
    with pytest.raises(ContractError, match='complete bounded'):
        Rollout.restore(model, timing, missing)
    changed = Timing(timing.times_ms[:-1] + (timing.times_ms[-1] + 1.,), timing.onsets)
    with pytest.raises(ContractError, match='timing'):
        Rollout.restore(model, changed, saved)
    with torch.no_grad():
        model.exact[0].bias.add_(.01)
    with pytest.raises(ContractError, match='parameters changed'):
        rollout.step(generator)
    with pytest.raises(ContractError, match='parameters'):
        Rollout.restore(model, timing, saved)


def test_r1_raw_recovery_cannot_retroactively_reveal_suffix_ln_endpoints():
    model = BoundedModel(ModelConfig(Arm.R1, hidden=8, levels=2, coupling_rank=2))
    timing = Timing((0., 50., 100., 150.), (True, True, False, True))
    state = Schedule.from_seed(Arm.R1, timing, [CompleteRow(0., (1, 0, 0, 0))])
    state, row = state.advance((2, 0, 0, 0))
    history = [RawEvent(CompleteRow(0., (1, 0, 0, 0)), None), RawEvent(row, 0., (100., None, None, None))]
    with pytest.raises(ContractError, match='unavailable'):
        Rollout(model, state, history, seed_rows=1)


def test_independent_verifier_rejects_timing_seed_and_plan_violations():
    timing = Timing((0., 100., 200., 300.), (True, True, False, True))
    seed = [CompleteRow(0., (1, 0, 0, 0))]
    rows = [*seed, CompleteRow(100., (2, 0, 0, 0)), CompleteRow(200., (3, 0, 0, 0)), CompleteRow(300., (0, 1, 0, 0))]
    verify_complete(rows, timing, Arm.O1, seed, suffix_plans={(1, 0): 2})
    with pytest.raises(ContractError, match='early'):
        verify_complete(rows, timing, Arm.O1, seed, suffix_plans={(1, 0): 3})
    with pytest.raises(ContractError, match='endpoint'):
        verify_complete(rows, timing, Arm.O1, seed)
    with pytest.raises(ContractError, match='seed'):
        verify_complete(rows[1:], timing, Arm.O1, seed, suffix_plans={(1, 0): 2})
    with pytest.raises(ContractError, match='required'):
        verify_complete(rows[:-1], timing, Arm.O1, seed, suffix_plans={(1, 0): 2})


def test_production_receptive_range_recovers_with_old_seed_holds_outside_raw_history():
    model = BoundedModel(ModelConfig(Arm.O1)).eval()
    timing = Timing(tuple(float(i * 100) for i in range(560)), (True,) * 559 + (False,))
    seed = [CompleteRow(0., (2, 2, 2, 1))]
    rollout = Rollout.from_seed(model, timing, seed, {0: 559, 1: 559, 2: 559})
    generator = torch.Generator().manual_seed(19)
    for _ in range(535):
        rollout.step(generator)
    assert rollout.state.replay.open_ln_start_ms[:3] == (0., 0., 0.)
    assert len(rollout.history) == 511 and rollout.history[0].row.time_ms > 0.
    restored, random = Rollout.restore(model, timing, rollout.snapshot(generator))
    torch.testing.assert_close(model.temporal.read(rollout.cache), model.temporal.read(restored.cache), atol=0, rtol=0)
    while not rollout.state.finished:
        assert rollout.step(generator).row == restored.step(random).row


@pytest.mark.parametrize('arm,consequence', [(arm, 'none') for arm in Arm] + [(Arm.R1, 'frontier')])
@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_native_teacher_forcing_matches_training_features_states_and_probabilities(arm, consequence, device, monkeypatch):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    rows = np.zeros(40, dtype=ROW_DTYPE)
    rows['time'] = np.arange(40) * 100. + (np.arange(40) >= 25) * 4000.
    rows['actions'][:, 2] = 1
    for lane, start, end in ((0, 0, 31), (1, 6, 11), (3, 17, 23)):
        rows['actions'][start, lane] = 2
        rows['actions'][end, lane] = 3
    source = SourceChart(SourceIdentity('a' * 64, 'b' * 64, 'synthetic', 'train'), rows, minimum_seed_notes=1)
    torch.manual_seed(23)
    model = BoundedModel(ModelConfig(arm, hidden=8, levels=2, coupling_rank=2, row_consequence=consequence)).to(device).eval()
    if model.row_consequence is not None:
        torch.nn.init.normal_(model.row_consequence.output.weight, std=.05)
    timing = source.view(arm).timing
    rollout = Rollout.from_seed(model, timing, [source.row(0)], None if arm == Arm.R0 else {0: 31})
    raw = source.content(arm, 0, len(rows))
    like = model.temporal.input.weight
    valid = torch.ones((1, len(rows)), dtype=torch.bool, device=device)
    content = model.temporal(like.new_tensor(raw[None]), valid)
    before = model.temporal.before(content, valid, truncated_start=torch.zeros(1, dtype=torch.bool, device=device))

    def choose_truth(_probabilities, _count, **_kwargs):
        action = source.row(rollout.state.index).actions
        target = tuple(a if a in (1, 2) else 0 for a in action) if arm == Arm.O1 else action
        return torch.tensor([model.choices.index(target)])

    def endpoint_truth(*_args, **_kwargs):
        return {lane: int(end) for lane, end in enumerate(source.endpoints[rollout.state.index]) if end >= 0}

    # Override only random selections. Native commit, feature construction,
    # cache writes and probability scoring remain on the real generation path.
    monkeypatch.setattr(torch, 'multinomial', choose_truth)
    monkeypatch.setattr(model, 'sample_endpoints', endpoint_truth)
    generator = torch.Generator().manual_seed(19)
    with torch.no_grad():
        for i in range(1, len(rows)):
            indexed = source.state(arm, i)
            assert rollout.state == indexed
            np.testing.assert_array_equal(query_features([rollout.state], rollout.view),
                                          query_features([indexed], source.view(arm)))
            if arm != Arm.O1 or timing.onsets[i]:
                hands = model.readout(before[:, i], like.new_tensor(query_features([indexed], source.view(arm))))
                trained = model.decision_log_probs(hands, [indexed])[0]
                _, online = rollout.prediction()
                torch.testing.assert_close(online, trained, atol=2e-5, rtol=2e-5)
            generated = rollout.step(generator, candidate_budget=7)
            assert generated.row == source.row(i)
            np.testing.assert_array_equal(rollout.history[-1].features(), raw[i])
