from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from pulsefield_model.research.bounded_typed_continuation import train_run
from pulsefield_model.research.bounded_typed_continuation.contract import Arm, Schedule, Timing
from pulsefield_model.research.bounded_typed_continuation.corpus import SamplingConfig, create_plan
from pulsefield_model.research.bounded_typed_continuation.data import SourceInterval, batch_likelihood, batch_predictions, prepare_batch
from pulsefield_model.research.bounded_typed_continuation.generation import RawEvent, Rollout, seed_events
from pulsefield_model.research.bounded_typed_continuation.model import BoundedModel, ModelConfig
from pulsefield_model.research.bounded_typed_continuation.recovery import NativeQuery, alternative_mask
from pulsefield_model.research.bounded_typed_continuation.support import row_supports
from pulsefield_model.research.bounded_typed_continuation.train_config import TrainConfig
from pulsefield_model.research.bounded_typed_continuation.train_hydra import compose_config
from pulsefield_model.research.bounded_typed_continuation.smoke_config import SmokeResources
from pulsefield_model.research.oracle_time_continuation.schema import CompleteRow
from pulsefield_model.research.oracle_time_continuation.storage import ROW_DTYPE, file_digest
from pulsefield_model.research.scoped_style_modeling.dataset import ContractError
from .test_corpus import inputs
from .test_data import mixed_chart
from .test_fork import write_extension
from .test_recovery import pool_fixture, save
from .test_train import compare_states


def models(device):
    torch.manual_seed(123)
    config = ModelConfig(Arm.R1, hidden=8, levels=2, coupling_rank=2, seed_context='observed',
        long_memory='landmarks', memory_hidden=16, memory_stride=2, head_routing='residual', routing_hidden=16)
    base = BoundedModel(config).to(device)
    torch.nn.init.normal_(base.route_residual.score[-1].weight, std=.1)
    release = BoundedModel(replace(config, release_routing='residual', release_hidden=16)).to(device)
    missing, unexpected = release.load_state_dict(base.state_dict(), strict=False)
    assert missing and all(key.startswith('release_residual.') for key in missing) and not unexpected
    return base, release


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_release_conditionals_no_hold_identity_mirror_and_initial_exactness(device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    base, release = models(device)
    timing = Timing((0., 100., 200., 300., 400.), (True, True, False, True, False))
    left, _ = Schedule(Arm.R1, timing).advance((2, 0, 1, 0))
    right, _ = Schedule(Arm.R1, timing).advance((0, 1, 0, 2))
    r_only, _ = left.advance((0, 2, 0, 1))
    free, _ = Schedule(Arm.R1, timing).advance((1, 0, 0, 0))
    states = [left, right, r_only, free]
    hands = torch.randn(4, 2, 8, device=device)
    hands[1] = hands[0].flip(0)
    expected = base.decision_log_probs(hands, states)
    torch.testing.assert_close(release.decision_log_probs(hands, states), expected, atol=0, rtol=0)
    torch.nn.init.normal_(release.release_residual.score[-1].weight, std=1.)
    actual = release.decision_log_probs(hands, states)
    assert not torch.allclose(actual[:3], expected[:3])
    torch.testing.assert_close(actual[3], expected[3], atol=0, rtol=0)
    for i in range(3):
        for mask in range(16):
            at = (release.release_residual.masks == mask) & torch.isfinite(expected[i])
            if at.any():
                a, b = actual[i, at], expected[i, at]
                torch.testing.assert_close(a-a.logsumexp(0), b-b.logsumexp(0), atol=3e-6, rtol=3e-6)
    mirrored = [release.choices.index(row[::-1]) for row in release.choices]
    torch.testing.assert_close(actual[0], actual[1, mirrored], atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_release_only_gradients_and_native_dense_raw_recovery(device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    base, release = models(device)
    train_run.configure_trainable(release, 'release')
    source = mixed_chart()
    batch = prepare_batch([SourceInterval(source, 0, len(source.onsets))], Arm.R1, 7, full_history=True)
    batch_likelihood(release, batch)[0].backward()
    assert release.release_residual.score[-1].weight.grad.norm() > 0
    assert all(p.grad is None for n, p in release.named_parameters() if not n.startswith('release_residual.'))
    optimizer = torch.optim.AdamW(release.parameters(), lr=.01, weight_decay=.1)
    optimizer.step()
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, release.state_dict()[name], atol=0, rtol=0)
    timing = Timing(tuple(float(i*100) for i in range(25)), tuple(i%3!=2 for i in range(25)))
    rows = (CompleteRow(0., (1, 0, 0, 0)),)
    rollout = Rollout.from_seed(release, timing, rows, {})
    _, seed = seed_events(Arm.R1, timing, rows, {})
    history = list(seed)
    rng = torch.Generator().manual_seed(41)
    while not rollout.state.finished:
        state = rollout.state
        query = NativeQuery(state, tuple(history), seed, np.zeros(256, bool))
        dense = batch_predictions(release, query.prepare(7, full_history=True))[1][0]
        torch.testing.assert_close(dense, rollout.prediction()[1], atol=3e-6, rtol=3e-5)
        if state.index == 13:
            restored, saved_rng = Rollout.restore(release, timing, rollout.snapshot(rng))
        step = rollout.step(rng)
        if state.index >= 13:
            assert restored.step(saved_rng) == step
        if step.row is not None:
            history.append(RawEvent(step.row, state.replay.last_row.time_ms))


def held_config(tmp_path):
    paths = inputs(tmp_path)
    # Add independently closing holds before freezing source/cache identities.
    # The original fixture is TAP-only and cannot exercise a learned release.
    for i, directory in enumerate(sorted(paths['source_cache_dir'].iterdir())):
        rows = np.fromfile(directory/'rows.bin', dtype=ROW_DTYPE)
        for first in range(30, 90, 10):
            rows['actions'][first, (i+1)%4] = 2
            rows['actions'][first+3, (i+1)%4] = 3
            rows['actions'][first+1, (i+2)%4] = 2
            rows['actions'][first+5, (i+2)%4] = 3
        (directory/'rows.bin').write_bytes(rows.tobytes())
        metadata = json.loads((directory/'metadata.json').read_text())
        metadata['rows_sha256'] = file_digest(directory/'rows.bin')
        (directory/'metadata.json').write_text(json.dumps(metadata))
    plan = create_plan(**paths, output_file=tmp_path/'plan.json',
        sampling=SamplingConfig(seed=17, horizons=(8, 16), milestones=(37, 103)))
    return TrainConfig(plan_file=plan['path'], plan_sha256=plan['sha256'],
        source_cache_dir=str(paths['source_cache_dir']), output_dir=str(tmp_path/'whole'), device='cpu',
        model=models('cpu')[0].config, trainable='routing', batch_size=3, microbatch_size=2,
        report_every=1, learning_rate=.001, warmup_onsets=16, max_seconds=60,
        resources=SmokeResources(min_available_bytes=1024**2))


def test_release_fork_replaces_pool_preserves_parent_and_resumes_exactly(tmp_path, monkeypatch):
    monkeypatch.setattr(train_run, 'source_revision', lambda: 'e'*40)
    config = held_config(tmp_path)
    old_pool, old_sha, plan, cache = pool_fixture(config, tmp_path)
    config = replace(config, recovery_pool=str(old_pool), recovery_sha256=old_sha)
    parent_result = train_run.run_training(config)
    parent = torch.load(tmp_path/'whole/checkpoint.pt', weights_only=True)
    trajectory = json.loads((tmp_path/'trajectory.json').read_text())
    chart = cache.get(trajectory['identity']['source_sha256'])
    markers = []
    for index in range(chart.seed_rows, len(chart.rows)):
        state = chart.state(Arm.R1, index)
        blocking = sum(1 << c for c, occupied in enumerate(state.replay.occupancy) if occupied)
        if not blocking or blocking == 15:
            continue
        core = next(1 << c for c in range(4) if not blocking & (1 << c))
        good = alternative_mask(state, core, blocking)
        legal = np.asarray(row_supports([state])[0], bool)
        if (good & legal).any() and (~good & legal).any():
            markers.append(dict(index=index, core_mask=core, blocking_mask=blocking))
    assert markers
    trajectory['queries'] = markers
    trajectory_sha = save(tmp_path/'held-trajectory.json', trajectory)
    manifest = json.loads(old_pool.read_text())
    manifest['runs'][0].update(path='held-trajectory.json', sha256=trajectory_sha, queries=len(markers))
    manifest['queries'] = len(markers)
    pool_sha = save(tmp_path/'held-pool.json', manifest)
    extension, plan_sha = write_extension(config, tmp_path)
    monkeypatch.setattr(train_run, 'source_revision', lambda: 'f'*40)
    config = replace(config, model=replace(config.model, release_routing='residual', release_hidden=16),
        trainable='release', recovery_pool=str(tmp_path/'held-pool.json'), recovery_sha256=pool_sha,
        plan_file=extension, plan_sha256=plan_sha, output_dir=str(tmp_path/'release-whole'),
        fork_from=str(tmp_path/'whole/checkpoint.pt'), fork_sha256=parent_result['checkpoint_sha256'],
        fork_source_revision='e'*40, fork_plan_file=config.plan_file)
    whole = train_run.run_training(config)
    final = torch.load(tmp_path/'release-whole/checkpoint.pt', weights_only=True)
    for name, value in parent['model'].items():
        compare_states(value, final['model'][name])
    for key, value in parent['optimizer']['state'].items():
        compare_states(value, final['optimizer']['state'][key])
    assert final['model']['release_residual.score.2.weight'].norm() > 0
    assert whole['metrics']['recovery_queries'] > 0
    train_run.run_training(replace(config, output_dir=str(tmp_path/'paused'), stop_after_checkpoint=137))
    train_run.run_training(replace(config, output_dir=str(tmp_path/'resumed'), stop_after_checkpoint=None,
        fork_from=None, fork_sha256='', fork_source_revision='', fork_plan_file='', resume_from=str(tmp_path/'paused/checkpoint.pt')))
    resumed = torch.load(tmp_path/'resumed/checkpoint.pt', weights_only=True)
    for key in ('model', 'optimizer', 'torch_rng', 'coverage', 'cursor', 'source_onset_exposures'):
        compare_states(final[key], resumed[key])
    for change in ({'trainable':'all'}, {'recovery_weight':.5}, {'model':replace(config.model, memory_hidden=32)}):
        with pytest.raises(ContractError):
            train_run.run_training(replace(config, output_dir=str(tmp_path/'invalid'), **change))


def test_release_hydra_projection_and_scope_validation():
    config = compose_config(['model.arm=R1', 'model.release_routing=residual', 'model.release_hidden=384', 'trainable=release'])
    assert config.model.release_hidden == 384 and config.model.release_routing == 'residual' and config.trainable == 'release'
    assert compose_config().model.release_routing == 'none'
    for overrides in (['model.release_routing=residual'], ['trainable=release'],
                      ['model.arm=R1', 'model.release_routing=unknown'], ['model.release_hidden=0']):
        with pytest.raises((ContractError, ValueError)):
            compose_config(overrides)
