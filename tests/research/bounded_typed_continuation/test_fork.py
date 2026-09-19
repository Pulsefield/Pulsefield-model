from dataclasses import asdict, replace
import hashlib
import json

import pytest
import torch

from pulsefield_model.research.bounded_typed_continuation import train_run
from pulsefield_model.research.bounded_typed_continuation.contract import Arm
from pulsefield_model.research.bounded_typed_continuation.corpus import SamplingConfig, draw_plan
from pulsefield_model.research.bounded_typed_continuation.data import SourceInterval, batch_likelihood, prepare_batch
from pulsefield_model.research.bounded_typed_continuation.evaluation import suffix_likelihood
from pulsefield_model.research.bounded_typed_continuation.model import BoundedModel, ModelConfig
from pulsefield_model.research.scoped_style_modeling.dataset import ContractError
from .test_data import mixed_chart
from .test_train import compare_states, config_fixture


def write_extension(config, tmp_path, *, seed=None):
    with open(config.plan_file) as stream:
        plan = json.load(stream)
    settings = dict(plan['sampling'])
    settings['milestones'] = (37, 103, 137, 167)
    if seed is not None:
        settings['seed'] = seed
    sampling = SamplingConfig(**settings)
    plan.update(sampling=asdict(sampling), draws=draw_plan(plan['sources'], sampling))
    path = tmp_path / f'extension-{seed}.json'
    path.write_text(json.dumps(plan, sort_keys=True) + '\n')
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def test_explicit_fork_then_strict_resume_matches_uninterrupted_extended_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(train_run, 'source_revision', lambda: 'e' * 40)
    config = config_fixture(tmp_path)
    parent = train_run.run_training(config)
    extension, digest = write_extension(config, tmp_path)
    monkeypatch.setattr(train_run, 'source_revision', lambda: 'f' * 40)
    target = replace(config, plan_file=extension, plan_sha256=digest, output_dir=str(tmp_path / 'extended'))
    whole = train_run.run_training(target)
    fork = replace(target, output_dir=str(tmp_path / 'fork'), stop_after_checkpoint=137,
                   fork_from=str(tmp_path / 'whole/checkpoint.pt'), fork_sha256=parent['checkpoint_sha256'],
                   fork_source_revision='e' * 40, fork_plan_file=config.plan_file)
    first = train_run.run_training(fork)
    assert first['status'] == 'paused' and first['source_onset_exposures'] == 137
    assert first['parent']['kind'] == 'fork' and first['parent']['source_revision'] == 'e' * 40
    assert first['compute_seconds'] >= parent['compute_seconds'] + first['segment_seconds'] - 1e-3
    resumed = replace(target, output_dir=str(tmp_path / 'resumed'), resume_from=str(tmp_path / 'fork/checkpoint.pt'))
    second = train_run.run_training(resumed)
    assert second['coverage'] == whole['coverage'] and second['source_onset_exposures'] == 167
    a = torch.load(tmp_path / 'extended/checkpoint.pt', weights_only=True)
    b = torch.load(tmp_path / 'resumed/checkpoint.pt', weights_only=True)
    for key in ('model', 'optimizer', 'torch_rng', 'device_rng', 'coverage', 'cursor', 'source_onset_exposures'):
        compare_states(a[key], b[key])
    assert json.loads((tmp_path / 'whole/result.json').read_text()) == parent
    bad_plan, bad_digest = write_extension(config, tmp_path, seed=99)
    for changes, pattern in (
        ({'fork_sha256': '0' * 64}, 'pinned digest'),
        ({'fork_source_revision': 'd' * 40}, 'configuration'),
        ({'learning_rate': .002}, 'scientific configuration'),
        ({'plan_file': bad_plan, 'plan_sha256': bad_digest}, 'old draw prefix'),
    ):
        with pytest.raises(ContractError, match=pattern):
            train_run.run_training(replace(fork, output_dir=str(tmp_path / 'rejected'), **changes))
    # A new-source checkpoint is still not an ordinary resume of its ancestor.
    with pytest.raises(ContractError, match='configuration'):
        train_run.run_training(replace(config, output_dir=str(tmp_path / 'not-a-fork'),
                                       resume_from=str(tmp_path / 'whole/checkpoint.pt')))


@pytest.mark.parametrize('arm,field,modes', [(Arm.O1, 'endpoint_availability', ('none', 'zero', 'commitment')),
                                          (Arm.R1, 'row_consequence', ('none', 'actions', 'frontier'))])
def test_zero_residual_fork_preserves_function_existing_adam_state_and_control_initialization(arm, field, modes):
    torch.manual_seed(71)
    settings = ModelConfig(arm, hidden=8, levels=2, coupling_rank=2)
    previous = BoundedModel(settings)
    source = mixed_chart()
    batch = prepare_batch([SourceInterval(source, 0, len(source.onsets))], arm,
                          previous.temporal.config.receptive_tokens)
    original_optimizer = torch.optim.AdamW(previous.parameters(), lr=.0003)
    batch_likelihood(previous, batch, candidate_budget=3)[0].backward()
    original_optimizer.step()
    payload = dict(config={'model': asdict(settings)}, model=previous.state_dict(), optimizer=original_optimizer.state_dict())
    payload['config']['model'].pop('row_consequence')  # Pre-extension checkpoints omit the default field.
    expected = suffix_likelihood(previous, source, candidate_budget=3)
    models = []
    for mode in modes:
        torch.manual_seed(171)
        model = BoundedModel(replace(settings, **{field: mode}))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
        train_run.restore_fork(model, optimizer, payload)
        for key, value in previous.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        restored = optimizer.state_dict()
        compare_states(payload['optimizer']['state'], restored['state'])
        actual = suffix_likelihood(model, source, candidate_budget=3)
        for key in ('head_nll_sum', 'endpoint_nll_sum'):
            assert actual[key] == pytest.approx(expected[key], rel=0, abs=1e-7)
        if mode != 'none':
            assert len(restored['param_groups'][0]['params']) > len(payload['optimizer']['param_groups'][0]['params'])
            loss, _ = batch_likelihood(model, batch, candidate_budget=3)
            loss.backward()
            output = model.pointer.availability_residual[-1] if arm == Arm.O1 else model.row_consequence.output
            assert output.weight.grad.norm() > 1e-8
        models.append(model)
    compare_states(models[1].state_dict(), models[2].state_dict())


@pytest.mark.parametrize('mode', ['actions', 'frontier'])
def test_r1_residual_fork_and_resume_preserve_extended_training_exactly(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(train_run, 'source_revision', lambda: 'e' * 40)
    config = config_fixture(tmp_path)
    config.model = replace(config.model, arm=Arm.R1)
    parent = train_run.run_training(config)
    extension, digest = write_extension(config, tmp_path)
    monkeypatch.setattr(train_run, 'source_revision', lambda: 'f' * 40)
    target = replace(config, model=replace(config.model, row_consequence=mode), plan_file=extension, plan_sha256=digest,
                     fork_from=str(tmp_path / 'whole/checkpoint.pt'), fork_sha256=parent['checkpoint_sha256'],
                     fork_source_revision='e' * 40, fork_plan_file=config.plan_file,
                     output_dir=str(tmp_path / 'extended'))
    whole = train_run.run_training(target)
    first = train_run.run_training(replace(target, output_dir=str(tmp_path / 'first'), stop_after_checkpoint=137))
    assert first['status'] == 'paused' and first['parent']['row_consequence'] == mode
    second = train_run.run_training(replace(target, output_dir=str(tmp_path / 'resumed'),
        fork_from=None, fork_sha256='', fork_source_revision='', fork_plan_file='',
        resume_from=str(tmp_path / 'first/checkpoint.pt')))
    assert second['status'] == whole['status'] == 'completed'
    a = torch.load(tmp_path / 'extended/checkpoint.pt', weights_only=True)
    b = torch.load(tmp_path / 'resumed/checkpoint.pt', weights_only=True)
    for key in ('model', 'optimizer', 'torch_rng', 'device_rng', 'coverage', 'cursor', 'source_onset_exposures'):
        compare_states(a[key], b[key])
    assert a['model']['row_consequence.output.weight'].norm() > 0
    # A feature-mode switch with identical tensor shapes is still a scientific change.
    other = 'actions' if mode == 'frontier' else 'frontier'
    with pytest.raises(ContractError, match='scientific configuration'):
        train_run.run_training(replace(target, output_dir=str(tmp_path / 'reinterpretation'),
            model=replace(target.model, row_consequence=other), fork_from=str(tmp_path / 'extended/checkpoint.pt'),
            fork_sha256=whole['checkpoint_sha256'], fork_source_revision='f' * 40, fork_plan_file=extension))
