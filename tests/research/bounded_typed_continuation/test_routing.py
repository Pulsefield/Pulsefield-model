from dataclasses import replace

import numpy as np
import pytest
import torch

from ensomi_model.research.bounded_typed_continuation import train_run
from ensomi_model.research.bounded_typed_continuation.contract import Arm, Schedule, Timing
from ensomi_model.research.bounded_typed_continuation.data import SourceInterval, batch_likelihood, batch_predictions, prepare_batch
from ensomi_model.research.bounded_typed_continuation.generation import RawEvent, Rollout, seed_events
from ensomi_model.research.bounded_typed_continuation.model import BoundedModel, ModelConfig
from ensomi_model.research.bounded_typed_continuation.recovery import NativeQuery
from ensomi_model.research.bounded_typed_continuation.train_hydra import compose_config
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.scoped_style_modeling.dataset import ContractError
from .test_data import mixed_chart
from .test_fork import write_extension
from .test_recovery import pool_fixture
from .test_train import compare_states, config_fixture


def models(device):
    torch.manual_seed(123)
    config = ModelConfig(Arm.R1, hidden=8, levels=2, coupling_rank=2, seed_context='observed',
                         long_memory='landmarks', memory_hidden=16, memory_stride=2)
    base = BoundedModel(config).to(device)
    route = BoundedModel(replace(config, head_routing='residual', routing_hidden=16)).to(device)
    missing, unexpected = route.load_state_dict(base.state_dict(), strict=False)
    assert missing and all(key.startswith('route_residual.') for key in missing) and not unexpected
    return base, route


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_initial_exactness_conditionals_releases_and_mirror_after_nonzero_routing(device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    base, route = models(device)
    timing = Timing((0., 100., 200., 300., 400.), (True, True, False, True, False))
    left, _ = Schedule(Arm.R1, timing).advance((2, 0, 1, 0))
    right, _ = Schedule(Arm.R1, timing).advance((0, 1, 0, 2))
    release, _ = left.advance((0, 2, 0, 1))
    states = [left, right, release]
    hands = torch.randn(3, 2, 8, device=device)
    hands[1] = hands[0].flip(0)
    expected = base.decision_log_probs(hands, states)
    initial = route.decision_log_probs(hands, states)
    torch.testing.assert_close(initial, expected, atol=0, rtol=0)
    torch.nn.init.normal_(route.route_residual.score[-1].weight, std=1.)
    actual = route.decision_log_probs(hands, states)
    assert not torch.allclose(actual[:2], expected[:2])
    torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)
    masks = route.route_residual.masks
    for i in range(2):
        for mask in range(1, 16):
            at = (masks == mask) & torch.isfinite(expected[i])
            if not at.any():
                continue
            a, b = actual[i, at], expected[i, at]
            torch.testing.assert_close(a - a.logsumexp(0), b - b.logsumexp(0), atol=2e-6, rtol=2e-6)
    mirrored = [route.choices.index(row[::-1]) for row in route.choices]
    torch.testing.assert_close(actual[0], actual[1, mirrored], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual.exp().sum(-1), torch.ones(3, device=device), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_routing_freezes_inherited_gradients_and_native_dense_recovery(device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    base, route = models(device)
    train_run.configure_trainable(route, 'routing')
    source = mixed_chart()
    batch = prepare_batch([SourceInterval(source, 0, len(source.onsets))], Arm.R1, 7, full_history=True)
    loss, _ = batch_likelihood(route, batch)
    loss.backward()
    for name, p in route.named_parameters():
        if name.startswith('route_residual.'):
            assert p.requires_grad and p.grad is not None
        else:
            assert not p.requires_grad and p.grad is None
    assert route.route_residual.score[-1].weight.grad.norm() > 0
    optimizer = torch.optim.AdamW(route.parameters(), lr=.01, weight_decay=.1)
    optimizer.step()
    for name, p in base.state_dict().items():
        torch.testing.assert_close(p, route.state_dict()[name], atol=0, rtol=0)
    timing = Timing(tuple(float(i*100) for i in range(25)), tuple(i%3!=2 for i in range(25)))
    rows = (CompleteRow(0., (1, 0, 0, 0)),)
    rollout = Rollout.from_seed(route, timing, rows, {})
    _, seed = seed_events(Arm.R1, timing, rows, {})
    history = list(seed)
    rng = torch.Generator().manual_seed(41)
    while not rollout.state.finished:
        state = rollout.state
        query = NativeQuery(state, tuple(history), seed, np.zeros(256, bool))
        batch = query.prepare(7, full_history=True)
        dense = batch_predictions(route, batch)[1][0]
        torch.testing.assert_close(dense, rollout.prediction()[1], atol=3e-6, rtol=3e-5)
        if state.index == 13:
            restored, saved_rng = Rollout.restore(route, timing, rollout.snapshot(rng))
        step = rollout.step(rng)
        if state.index >= 13:
            assert restored.step(saved_rng) == step
        if step.row is not None:
            history.append(RawEvent(step.row, state.replay.last_row.time_ms))
    assert torch.equal(rng.get_state(), saved_rng.get_state())


def test_routing_fork_preserves_weights_adam_and_exact_resumed_updates(tmp_path, monkeypatch):
    monkeypatch.setattr(train_run, 'source_revision', lambda: 'e' * 40)
    config = replace(config_fixture(tmp_path), model=models('cpu')[0].config)
    parent_result = train_run.run_training(config)
    parent = torch.load(tmp_path/'whole/checkpoint.pt', weights_only=True)
    pool, pool_sha, _, _ = pool_fixture(config, tmp_path)
    extension, plan_sha = write_extension(config, tmp_path)
    monkeypatch.setattr(train_run, 'source_revision', lambda: 'f' * 40)
    config = replace(config, model=replace(config.model, head_routing='residual', routing_hidden=16), trainable='routing',
        recovery_pool=str(pool), recovery_sha256=pool_sha, plan_file=extension, plan_sha256=plan_sha,
        output_dir=str(tmp_path/'route-whole'), fork_from=str(tmp_path/'whole/checkpoint.pt'),
        fork_sha256=parent_result['checkpoint_sha256'], fork_source_revision='e'*40, fork_plan_file=config.plan_file)
    whole = train_run.run_training(config)
    final = torch.load(tmp_path/'route-whole/checkpoint.pt', weights_only=True)
    for name, value in parent['model'].items():
        compare_states(value, final['model'][name])
    for key, value in parent['optimizer']['state'].items():
        compare_states(value, final['optimizer']['state'][key])
    assert final['model']['route_residual.score.2.weight'].norm() > 0
    assert whole['metrics']['recovery_queries'] > 0
    train_run.run_training(replace(config, output_dir=str(tmp_path/'paused'), stop_after_checkpoint=137))
    train_run.run_training(replace(config, output_dir=str(tmp_path/'resumed'), stop_after_checkpoint=None,
        fork_from=None, fork_sha256='', fork_source_revision='', fork_plan_file='', resume_from=str(tmp_path/'paused/checkpoint.pt')))
    resumed = torch.load(tmp_path/'resumed/checkpoint.pt', weights_only=True)
    for key in ('model','optimizer','torch_rng','coverage','cursor','source_onset_exposures'):
        compare_states(final[key], resumed[key])
    for change in ({'trainable':'all'}, {'model':replace(config.model,memory_hidden=32)}):
        with pytest.raises(ContractError):
            train_run.run_training(replace(config,output_dir=str(tmp_path/'invalid'),**change))


def test_routing_hydra_projection_and_invalid_scope():
    config = compose_config(['model.arm=R1','model.head_routing=residual','model.routing_hidden=384','trainable=routing'])
    assert config.model.routing_hidden == 384 and config.model.head_routing == 'residual' and config.trainable == 'routing'
    assert compose_config().trainable == 'all' and compose_config().model.head_routing == 'none'
    for overrides in (['model.head_routing=residual'], ['trainable=routing'], ['trainable=unknown'],
                      ['model.arm=R1','model.head_routing=unknown'], ['model.routing_hidden=0']):
        with pytest.raises((ContractError, ValueError)):
            compose_config(overrides)
