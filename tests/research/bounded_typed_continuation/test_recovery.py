from dataclasses import asdict, replace
import hashlib
import json

import numpy as np
import pytest
import torch

from ensomi_model.research.bounded_typed_continuation import train_run
from ensomi_model.research.bounded_typed_continuation.condition import GenerationCondition
from ensomi_model.research.bounded_typed_continuation.contract import Arm, ROW_ACTIONS, Timing
from ensomi_model.research.bounded_typed_continuation.corpus import ChartCache, read_plan
from ensomi_model.research.bounded_typed_continuation.data import batch_predictions
from ensomi_model.research.bounded_typed_continuation.generation import RawEvent, Rollout, seed_events
from ensomi_model.research.bounded_typed_continuation.model import BoundedModel, ModelConfig
from ensomi_model.research.bounded_typed_continuation.recovery import (
    NativeQuery, RecoveryPool, alternative_mask, complement_loss, trajectory_queries,
)
from ensomi_model.research.bounded_typed_continuation.support import row_supports
from ensomi_model.research.bounded_typed_continuation.train_hydra import compose_config
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.scoped_style_modeling.dataset import ContractError
from .test_fork import write_extension
from .test_train import compare_states, config_fixture


def save(path, value):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('memory,device', [('none', 'cpu'), ('landmarks', 'cpu'), ('landmarks', 'mps')])
def test_native_query_dense_predictions_and_current_placeholder_noninterference(memory, device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    torch.manual_seed(72)
    model = BoundedModel(ModelConfig(Arm.R1, hidden=8, levels=2, coupling_rank=2,
        seed_context='observed', long_memory=memory, memory_hidden=16, memory_stride=2)).to(device)
    if model.long_memory is not None:
        torch.nn.init.normal_(model.long_memory.output.weight, std=.1)
    timing = Timing(tuple(float(i * 100) for i in range(30)), tuple(i % 3 != 2 for i in range(30)))
    rows = (CompleteRow(0., (1, 0, 0, 0)),)
    rollout = Rollout.from_seed(model, timing, rows, {})
    _, seed = seed_events(Arm.R1, timing, rows, {})
    history = list(seed)
    rng = torch.Generator().manual_seed(41)
    skipped = 0
    while not rollout.state.finished:
        state = rollout.state
        query = NativeQuery(state, tuple(history), seed, np.zeros(256, bool))
        batch = query.prepare(model.temporal.config.receptive_tokens, full_history=memory != 'none')
        hands, logp = batch_predictions(model, batch)
        native_hands, native_logp = rollout.prediction()
        torch.testing.assert_close(hands[0], native_hands, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(logp[0], native_logp, atol=3e-6, rtol=3e-5)
        batch.raw[0, -1] = 700
        again = batch_predictions(model, batch)[1]
        torch.testing.assert_close(again, logp, atol=0, rtol=0)
        step = rollout.step(rng)
        if step.row is None:
            skipped += 1
        else:
            history.append(RawEvent(step.row, state.replay.last_row.time_ms))
    assert skipped and state.index > len(history) - 1


def test_complement_likelihood_is_stable_and_pushes_mass_to_legal_alternatives():
    logits = torch.tensor([[80., 0., -2., 99.]], requires_grad=True)
    logp = logits.masked_fill(torch.tensor([[False, False, False, True]]), -torch.inf).log_softmax(-1)
    good = np.array([[False, True, True, True]])
    loss = complement_loss(logp, good)
    assert 79 < loss < 81
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0] > .9 and logits.grad[0, 1:3].sum() < -.9 and logits.grad[0, 3] == 0
    for bad in (np.zeros((1, 4), bool), np.ones((1, 4), bool)):
        with pytest.raises(ContractError, match='both alternative'):
            complement_loss(logp, bad)


def test_hold_escape_uses_original_skipped_release_candidates_and_no_suffix_end_plans():
    timing = Timing((0., 100., 200., 300., 400., 500., 600.), (True, True, False, True, False, True, False))
    condition = GenerationCondition(Arm.R1, timing, (CompleteRow(0., (1, 0, 0, 0)),), (None,) * 4)
    actions = [[1, 2, 2, 2], [0, 0, 0, 0], [1, 0, 0, 0], [0, 3, 3, 3], [0, 1, 0, 0], [0, 0, 0, 0]]
    value = dict(condition=json.loads(json.dumps(condition.payload())), actions=actions,
                 queries=[dict(index=4, core_mask=1, blocking_mask=14)])
    query, = trajectory_queries(value, condition)
    assert query.state.index == 4 and len(query.history) == 3
    assert query.state.timing == timing
    assert query.history[1].new_end_times == (None,) * 4
    good = query.alternatives
    assert good[ROW_ACTIONS.index((0, 3, 0, 0))]
    assert not good[ROW_ACTIONS.index((0, 0, 0, 0))]
    value['actions'][2] = [0, 1, 0, 0]
    with pytest.raises(ContractError, match='Illegal row'):
        trajectory_queries(value, condition)


def pool_fixture(config, tmp_path):
    plan = read_plan(config.plan_file, config.plan_sha256)
    cache = ChartCache(config.source_cache_dir, plan['sources'], max_sources=2, max_bytes=1024**2)
    sha, pin = next(iter(plan['sources'].items()))
    chart = cache.interval(dict(source_sha256=sha, first_onset=0, onset_count=1)).chart
    condition = GenerationCondition(Arm.R1, chart.typed.timing, tuple(chart.row(i) for i in range(chart.seed_rows)),
                                    chart.state(Arm.R1, chart.seed_rows).known_ends)
    markers = []
    for i in range(chart.seed_rows, len(chart.rows)):
        state = chart.state(Arm.R1, i)
        good = alternative_mask(state, 1, 0)
        legal = np.asarray(row_supports([state])[0], bool)
        if (good & legal).any() and (~good & legal).any():
            markers.append(dict(index=i, core_mask=1, blocking_mask=0))
    assert markers
    run = dict(identity=pin['identity'], source_metadata_sha256=pin['metadata_sha256'], source_rows_sha256=pin['rows_sha256'],
               condition=condition.payload(), actions=chart.rows['actions'][chart.seed_rows:].tolist(), queries=markers,
               policy_checkpoint_sha256='a' * 64, policy_source_revision='e' * 40, seed=43)
    trajectory = tmp_path / 'trajectory.json'
    digest = save(trajectory, run)
    manifest = dict(format='bounded-typed/native-recovery-pool-v1', status='ready', runs=[dict(source_sha256=sha,
        path=trajectory.name, sha256=digest, queries=len(markers))], queries=len(markers), policy_checkpoint_sha256='a' * 64,
        policy_source_revision='e' * 40, seed=43, plan_sha256=config.plan_sha256)
    path = tmp_path / 'pool.json'
    return path, save(path, manifest), plan, cache


def test_pinned_pool_split_digest_and_deterministic_sampling(tmp_path):
    config = replace(config_fixture(tmp_path), model=ModelConfig(Arm.R1, hidden=8, levels=2, coupling_rank=2))
    path, digest, plan, cache = pool_fixture(config, tmp_path)
    pool = RecoveryPool(path, digest, plan, cache)
    assert [q.state.index for q in pool.select(14, 8, 954)] == [q.state.index for q in pool.select(14, 8, 954)]
    with pytest.raises(ContractError, match='SHA-256'):
        RecoveryPool(path, '0' * 64, plan, cache)
    sha = json.loads(path.read_text())['runs'][0]['source_sha256']
    plan['sources'][sha]['identity']['split'] = 'validation'
    with pytest.raises(ContractError, match='TRAIN'):
        RecoveryPool(path, digest, plan, cache)


def test_same_architecture_objective_fork_runner_consumption_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(train_run, 'source_revision', lambda: 'e' * 40)
    config = replace(config_fixture(tmp_path), model=ModelConfig(Arm.R1, hidden=8, levels=2, coupling_rank=2,
        seed_context='observed', long_memory='landmarks', memory_hidden=16, memory_stride=2))
    parent = train_run.run_training(config)
    path, digest, _, _ = pool_fixture(config, tmp_path)
    extension, plan_digest = write_extension(config, tmp_path)
    monkeypatch.setattr(train_run, 'source_revision', lambda: 'f' * 40)
    fork = replace(config, plan_file=extension, plan_sha256=plan_digest, recovery_pool=str(path), recovery_sha256=digest,
        output_dir=str(tmp_path / 'fork-whole'), fork_from=str(tmp_path / 'whole/checkpoint.pt'),
        fork_sha256=parent['checkpoint_sha256'], fork_source_revision='e' * 40, fork_plan_file=config.plan_file)
    whole = train_run.run_training(fork)
    assert whole['metrics']['recovery_queries'] > 0 and whole['metrics']['recovery_loss_sum'] > 0
    paused = train_run.run_training(replace(fork, output_dir=str(tmp_path / 'fork-paused'), stop_after_checkpoint=137))
    assert paused['status'] == 'paused'
    resumed = train_run.run_training(replace(fork, output_dir=str(tmp_path / 'resumed'), stop_after_checkpoint=None,
        fork_from=None, fork_sha256='', fork_source_revision='', fork_plan_file='', resume_from=str(tmp_path / 'fork-paused/checkpoint.pt')))
    a = torch.load(tmp_path / 'fork-whole/checkpoint.pt', weights_only=True)
    b = torch.load(tmp_path / 'resumed/checkpoint.pt', weights_only=True)
    for key in ('model', 'optimizer', 'torch_rng', 'cursor', 'coverage'):
        compare_states(a[key], b[key])
    assert resumed['metrics']['recovery_queries'] == whole['metrics']['recovery_queries']


def test_hydra_recovery_projection_rejects_unused_and_invalid_settings():
    config = compose_config(['model.arm=R1', 'recovery_pool=pool.json', 'recovery_sha256=' + 'a' * 64,
                             'recovery_weight=.2', 'recovery_queries=3', 'recovery_seed=7'])
    assert (config.recovery_pool, config.recovery_weight, config.recovery_queries, config.recovery_seed) == ('pool.json', .2, 3, 7)
    for overrides in (['recovery_weight=.2'], ['recovery_pool=pool.json'], ['recovery_queries=0'],
                      ['recovery_pool=pool.json', 'recovery_sha256=' + 'a' * 64]):
        with pytest.raises(ContractError):
            compose_config(overrides)
