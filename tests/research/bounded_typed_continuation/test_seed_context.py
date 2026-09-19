from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json

import numpy as np
import pytest
import torch

from pulsefield_model.research.bounded_typed_continuation.contract import Arm
from pulsefield_model.research.bounded_typed_continuation.data import SourceInterval, batch_likelihood, prepare_batch
from pulsefield_model.research.bounded_typed_continuation.features import query_features
from pulsefield_model.research.bounded_typed_continuation.generation import Rollout, model_digest
from pulsefield_model.research.bounded_typed_continuation.model import BoundedModel, ModelConfig
from pulsefield_model.research.scoped_style_modeling.dataset import ContractError
from .test_data import chart, mixed_chart
from .test_generation import setup


def observed_model(device='cpu'):
    torch.manual_seed(731)
    model = BoundedModel(ModelConfig(Arm.R1, hidden=8, levels=2, coupling_rank=2, seed_context='observed')).to(device)
    torch.nn.init.normal_(model.seed_residual[-1].weight, std=.1)
    return model


def seed_encoding(model, batch):
    like = model.temporal.input.weight
    return model.encode_seed(like.new_tensor(batch.seed_raw), torch.as_tensor(batch.seed_valid, device=like.device))


def test_seed_configuration_and_actual_parameter_increment():
    for arm in (Arm.R0, Arm.O1):
        with pytest.raises(ContractError, match='R1-only'):
            ModelConfig(arm, seed_context='observed')
    with pytest.raises(ContractError, match='Seed context'):
        ModelConfig(Arm.R1, seed_context='unknown')
    base = ModelConfig(Arm.R1)
    sizes = [sum(p.numel() for p in BoundedModel(replace(base, seed_context=mode)).parameters())
             for mode in ('none', 'zero', 'observed')]
    assert sizes[1] == sizes[2] == sizes[0] + 49280


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_original_seed_pooling_ignores_padding_and_preserves_mirror_and_gradients(device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    model = observed_model(device)
    short = mixed_chart()
    long = chart([(1, 0, 0, 0)] * 22, minimum_seed_notes=14)
    batch = prepare_batch([SourceInterval(short, 0, 2), SourceInterval(long, 0, 2)], Arm.R1, 7)
    pooled = seed_encoding(model, batch)
    for i, source in enumerate((short, long)):
        single = prepare_batch([SourceInterval(source, 0, 2)], Arm.R1, 7)
        torch.testing.assert_close(pooled[i], seed_encoding(model, single)[0], atol=2e-6, rtol=2e-6)
    changed = deepcopy(batch)
    changed.seed_raw[~changed.seed_valid] = 1e6
    torch.testing.assert_close(pooled, seed_encoding(model, changed), atol=0, rtol=0)
    # Mirror only swaps the two canonical hand views throughout the shared path.
    like = model.temporal.input.weight
    raw = like.new_tensor(batch.seed_raw)
    valid = torch.as_tensor(batch.seed_valid, device=device)
    mirrored = model.encode_seed(raw.flip(2), valid)
    torch.testing.assert_close(mirrored, pooled.flip(1), atol=0, rtol=0)
    before = torch.randn_like(pooled)
    exact = like.new_tensor(batch.query_features[:2])
    a = model.joint(model.readout(before, exact, pooled))
    b = model.joint(model.readout(before.flip(1), exact.flip(1), mirrored))
    mirror = [model.choices.index(tuple(reversed(row))) for row in model.choices]
    torch.testing.assert_close(a, b[:, mirror], atol=2e-6, rtol=2e-6)
    batch_likelihood(model, batch)[0].backward()
    assert model.temporal.input.weight.grad.norm() > 0
    assert model.seed_residual[0].weight.grad[:, model.config.hidden:].norm() > 0
    assert model.seed_residual[-1].weight.grad.norm() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_seed_readout_rejects_missing_condition_and_control_cannot_consume_one():
    model = observed_model()
    batch = prepare_batch([SourceInterval(mixed_chart(), 0, 1)], Arm.R1, 7)
    pooled = seed_encoding(model, batch)
    before, exact = torch.zeros_like(pooled), torch.from_numpy(batch.query_features[:1])
    with pytest.raises(ContractError, match='original seed'):
        model.readout(before, exact)
    for mode in ('none', 'zero'):
        control = BoundedModel(replace(model.config, seed_context=mode))
        with pytest.raises(ContractError, match='do not consume'):
            control.readout(before, exact, pooled)
    raw = torch.from_numpy(batch.seed_raw)
    for valid in (torch.zeros(raw.shape[:2], dtype=torch.bool), torch.ones(raw.shape[:2])):
        with pytest.raises(ContractError, match='prefix-padded'):
            model.encode_seed(raw, valid)


def test_persistent_condition_uses_original_seed_but_not_ancient_nonseed_actions():
    actions = [(1, 0, 0, 0)] * 60
    actions[16] = (1, 1, 1, 1)  # Overwrite each lane's exact attack clock in all sources.
    seed_changed, old_changed = list(actions), list(actions)
    seed_changed[0] = (0, 1, 0, 0)
    old_changed[2] = (0, 1, 0, 0)
    sources = [chart(a) for a in (actions, seed_changed, old_changed)]
    model = observed_model().double()
    batches = [prepare_batch([SourceInterval(source, 39, 4)], Arm.R1, 7) for source in sources]
    for other in batches[1:]:
        np.testing.assert_array_equal(batches[0].raw, other.raw)
        np.testing.assert_array_equal(batches[0].query_features, other.query_features)
        assert batches[0].states == other.states
    values = [batch_likelihood(model, batch)[0] for batch in batches]
    torch.testing.assert_close(values[0], values[2], atol=0, rtol=0)
    assert abs(float((values[0] - values[1]).detach())) > 1e-7
    # Shared-encoder seed gradients and rolling gradients obey the same crop boundary.
    complete = prepare_batch([SourceInterval(sources[0], 39, 4)], Arm.R1, 100)
    full = batch_likelihood(model, complete)[0]
    torch.testing.assert_close(values[0], full, atol=1e-10, rtol=1e-10)
    for a, b in zip(torch.autograd.grad(values[0], tuple(model.parameters()), allow_unused=True),
                    torch.autograd.grad(full, tuple(model.parameters()), allow_unused=True)):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)


def test_suffix_endpoint_labels_do_not_enter_persistent_seed_or_current_decision():
    a = chart([(1, 0, 0, 0), (2, 2, 0, 0), (0, 0, 1, 0), (3, 0, 0, 0), (0, 3, 0, 0), (1, 0, 0, 0)])
    b = chart([(1, 0, 0, 0), (2, 2, 0, 0), (0, 0, 1, 0), (0, 3, 0, 0), (3, 0, 0, 0), (1, 0, 0, 0)])
    model = observed_model()
    batches = [prepare_batch([SourceInterval(source, 0, 1)], Arm.R1, 7) for source in (a, b)]
    np.testing.assert_array_equal(batches[0].seed_raw, batches[1].seed_raw)
    torch.testing.assert_close(batch_likelihood(model, batches[0])[0], batch_likelihood(model, batches[1])[0], atol=0, rtol=0)


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_persistent_seed_native_dense_teacher_forcing_and_expired_seed_recovery(device, monkeypatch):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    actions = [(1, 0, 0, 0)] * 42
    actions[0], actions[30] = (2, 0, 1, 0), (3, 0, 1, 0)
    actions[1:30] = [(0, 1, 0, 0)] * 29
    source = chart(actions, minimum_seed_notes=5)
    model = observed_model(device).eval()
    timing = source.typed.timing
    seed = [source.row(i) for i in range(source.seed_rows)]
    rollout = Rollout.from_seed(model, timing, seed, {0: 30})
    batch = prepare_batch([SourceInterval(source, 0, len(source.onsets))], Arm.R1, 7)
    like = model.temporal.input.weight
    valid = torch.as_tensor(batch.valid, device=device)
    content = model.temporal(like.new_tensor(batch.raw), valid)
    before = model.temporal.before(content, valid, truncated_start=torch.as_tensor(batch.truncated, device=device))
    encoded_seed = seed_encoding(model, batch)
    torch.testing.assert_close(rollout.seed_context, encoded_seed[0], atol=0, rtol=0)
    def truth(_probabilities, _count, **_kwargs):
        return torch.tensor([model.choices.index(source.row(rollout.state.index).actions)])
    monkeypatch.setattr(torch, 'multinomial', truth)
    rng = torch.Generator().manual_seed(19)
    for j, state in enumerate(batch.states):
        hands = model.readout(before[:, batch.positions[j]], like.new_tensor(query_features([state], source.typed)), encoded_seed)
        trained = model.decision_log_probs(hands, [state])[0]
        torch.testing.assert_close(rollout.prediction()[1], trained, atol=2e-5, rtol=2e-5)
        if state.index == 25:
            assert rollout.history[0].row.time_ms > seed[-1].time_ms
            restored, other_rng = Rollout.restore(model, timing, rollout.snapshot(rng))
            torch.testing.assert_close(restored.seed_context, rollout.seed_context, atol=0, rtol=0)
            torch.testing.assert_close(restored.prediction()[1], rollout.prediction()[1], atol=0, rtol=0)
            assert torch.equal(other_rng.get_state(), rng.get_state())
        rollout.step(rng)
    assert rollout.state.finished


def test_seed_snapshot_validation_and_legacy_default_identity():
    model, timing, _, rollout = setup(Arm.R1, seed_context='observed')
    rng = torch.Generator().manual_seed(19)
    for _ in range(10):
        rollout.step(rng)
    saved = rollout.snapshot(rng)
    assert 'seed_context' not in saved and len(saved['seed_history']) == 3
    bad = deepcopy(saved)
    bad.pop('seed_history')
    with pytest.raises(ContractError, match='complete original'):
        Rollout.restore(model, timing, bad)
    bad = deepcopy(saved)
    bad['seed_history'][0]['new_end_times'] = (1000., 1800., None, None)
    with pytest.raises(ContractError, match='commitment'):
        Rollout.restore(model, timing, bad)
    bad = deepcopy(saved)
    bad['seed_history'][1]['previous_time_ms'] = 1.
    with pytest.raises(ContractError, match='Persistent raw seed'):
        Rollout.restore(model, timing, bad)
    plain, timing, _, original = setup(Arm.R1)
    legacy = original.snapshot(rng)
    legacy['model_config'].pop('seed_context')
    old_config = asdict(plain.config)
    old_config.pop('seed_context')
    digest = hashlib.sha256(json.dumps(old_config, sort_keys=True).encode())
    for name, tensor in plain.state_dict().items():
        digest.update(json.dumps((name, str(tensor.dtype), tuple(tensor.shape))).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    assert legacy['parameter_sha256'] == model_digest(plain) == digest.hexdigest()
    restored, _ = Rollout.restore(plain, timing, legacy)
    assert restored.state == original.state and not restored.seed_history
