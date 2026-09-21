from dataclasses import replace
from importlib.resources import files
import json

import numpy as np
import pytest
import torch

from pulsefield_model.research.bounded_typed_continuation import train_run
from pulsefield_model.research.bounded_typed_continuation.contract import Arm
from pulsefield_model.research.bounded_typed_continuation.corpus import SamplingConfig, create_plan
from pulsefield_model.research.bounded_typed_continuation.data import SourceInterval, batch_likelihood, prepare_batch
from pulsefield_model.research.bounded_typed_continuation.evaluation import suffix_likelihood
from pulsefield_model.research.bounded_typed_continuation.model import BoundedModel, ModelConfig
from pulsefield_model.research.bounded_typed_continuation.train_hydra import compose_config
from pulsefield_model.research.scoped_style_modeling.dataset import ContractError
from .test_corpus import inputs
from .test_data import mixed_chart


def test_hydra_packaged_projection_and_rejected_unknown_or_unused_fields():
    config = compose_config(['model.arm=R1', 'batch_size=3', 'microbatch_size=1', 'stop_after_checkpoint=250000'])
    assert config.model.arm == Arm.R1 and config.batch_size == 3 and config.microbatch_size == 1
    assert config.device == 'cpu' and config.footprint_limit_bytes == 6 * 1024 ** 3
    assert config.stop_after_checkpoint == 250000
    assert config.model.endpoint_availability == 'none' and config.fork_from is None
    typed = compose_config(['model.endpoint_availability=commitment'])
    assert typed.model.endpoint_availability == 'commitment'
    consequence = compose_config(['model.arm=R1', 'model.row_consequence=frontier'])
    assert consequence.model.row_consequence == 'frontier' and config.model.row_consequence == 'none'
    conditioned = compose_config(['model.arm=R1', 'model.seed_context=observed'])
    assert conditioned.model.seed_context == 'observed' and config.model.seed_context == 'none'
    memory = compose_config(['model.arm=R1', 'model.long_memory=landmarks', 'model.memory_hidden=192', 'model.memory_stride=32'])
    assert memory.model.long_memory == 'landmarks' and memory.model.memory_hidden == 192 and memory.model.memory_stride == 32
    assert config.model.long_memory == 'none'
    for overrides in (['model.long_memory=landmarks'], ['model.arm=R1', 'model.long_memory=unknown'],
                      ['model.memory_hidden=0'], ['model.memory_stride=0']):
        with pytest.raises((ContractError, ValueError)):
            compose_config(overrides)
    for overrides in (['model.seed_context=observed'], ['model.arm=R1', 'model.seed_context=unknown']):
        with pytest.raises((ContractError, ValueError)):
            compose_config(overrides)
    assert files('pulsefield_model.configs.hydra').joinpath('bounded_typed_train.yaml').is_file()
    for overrides in (['+unused=1'], ['+model.unused=1'], ['+resources.check_every_rows=1'],
                      ['candidate_budget=0'], ['learning_rate=0'], ['microbatch_size=3'], ['cache_max_sources=129'],
                      ['footprint_limit_bytes=0'], ['model.arm=R1', 'model.endpoint_availability=zero'],
                      ['fork_from=parent.pt'], ['model.endpoint_availability=unknown'],
                      ['model.row_consequence=frontier'], ['model.arm=R1', 'model.row_consequence=unknown']):
        with pytest.raises((ValueError, ContractError)):
            compose_config(overrides)


@pytest.mark.parametrize('arm,consequence', [(arm, 'none') for arm in Arm] + [(Arm.R1, 'frontier')])
@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_unequal_microbatches_have_same_summed_loss_gradients(arm, consequence, device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    torch.manual_seed(42)
    model = BoundedModel(ModelConfig(arm, hidden=8, levels=2, coupling_rank=2, row_consequence=consequence)).to(device)
    if model.row_consequence is not None:
        torch.nn.init.normal_(model.row_consequence.output.weight, std=.05)
    source = mixed_chart()
    intervals = [SourceInterval(source, 0, 1), SourceInterval(source, 1, 3)]
    batch = prepare_batch(intervals, arm, model.temporal.config.receptive_tokens)
    loss, _ = batch_likelihood(model, batch, candidate_budget=3)
    loss.backward()
    expected = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    records = [train_run.measure(model, [item], candidate_budget=3, backward=True, denominator=4,
                               check=lambda _: None) for item in intervals]
    assert sum(record['source_onsets'] for record in records) == 4
    for parameter, gradient in zip(model.parameters(), expected):
        if gradient is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, gradient, atol=3e-6, rtol=5e-4)


@pytest.mark.parametrize('arm,consequence', [(arm, 'none') for arm in Arm] + [(Arm.R1, 'frontier')])
def test_full_suffix_code_length_is_invariant_to_onset_chunking(arm, consequence):
    model = BoundedModel(ModelConfig(arm, hidden=8, levels=2, coupling_rank=2, row_consequence=consequence)).double()
    if model.row_consequence is not None:
        torch.nn.init.normal_(model.row_consequence.output.weight, std=.05)
    source = mixed_chart()
    whole = suffix_likelihood(model, source, chunk_onsets=128, candidate_budget=3)
    chunks = suffix_likelihood(model, source, chunk_onsets=1, candidate_budget=2)
    assert chunks['physical_rows'] == len(source.rows) - source.seed_rows
    assert chunks['source_onsets'] == len(source.onsets)
    for key in ('head_nll_sum', 'endpoint_nll_sum', 'nll_per_onset', 'ln_binary_nll_sum',
                'ln_binary_count', 'ln_binary_positive_count', 'head_type_binary_nll_sum'):
        assert chunks[key] == pytest.approx(whole[key], rel=1e-9, abs=1e-9)
    assert model.training


def config_fixture(tmp_path):
    paths = inputs(tmp_path)
    plan = create_plan(**paths, output_file=tmp_path / 'plan.json',
                       sampling=SamplingConfig(seed=17, horizons=(8, 16), milestones=(37, 103)))
    return compose_config([f'plan_file={plan["path"]}', f'plan_sha256={plan["sha256"]}',
                           f'source_cache_dir={paths["source_cache_dir"]}', f'output_dir={tmp_path / "whole"}',
                           'device=cpu', 'model.hidden=8', 'model.levels=2', 'model.coupling_rank=2',
                           'batch_size=3', 'microbatch_size=2', 'candidate_budget=3', 'report_every=1',
                           'learning_rate=0.001', 'warmup_onsets=16', 'cache_max_sources=1',
                           'resources.min_available_bytes=1048576', 'max_seconds=60'])


def compare_states(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            compare_states(first[key], second[key])
    elif isinstance(first, list):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            compare_states(a, b)
    else:
        assert first == second


@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_exact_milestone_resume_preserves_optimizer_and_exposures(tmp_path, monkeypatch, device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    monkeypatch.setattr(train_run, 'source_revision', lambda: 'e' * 40)
    config = replace(config_fixture(tmp_path), device=device)
    whole = train_run.run_training(config, resolved_yaml='synthetic runtime projection\n')
    assert whole['status'] == 'completed' and whole['source_onset_exposures'] == 103
    first_config = replace(config, output_dir=str(tmp_path / 'first'), stop_after_checkpoint=37)
    first = train_run.run_training(first_config)
    assert first['status'] == 'paused' and first['durable_onset_exposures'] == 37
    resumed_config = replace(config, output_dir=str(tmp_path / 'second'), resume_from=str(tmp_path / 'first/checkpoint.pt'))
    second = train_run.run_training(resumed_config)
    assert second['compute_seconds'] >= first['compute_seconds'] + second['segment_seconds'] - 1e-3
    assert second['coverage'] == whole['coverage'] and second['parent']['discarded_updates'] == 0
    a = torch.load(tmp_path / 'whole/checkpoint.pt', weights_only=True)
    b = torch.load(tmp_path / 'second/checkpoint.pt', weights_only=True)
    for key in ('model', 'optimizer', 'torch_rng', 'device_rng', 'coverage', 'cursor', 'source_onset_exposures'):
        compare_states(a[key], b[key])
    records = [json.loads(line) for line in (tmp_path / 'whole/training.jsonl').read_text().splitlines()]
    assert sum(record['source_onsets'] for record in records) == 103
    assert records[0]['learning_rate'] == .001
    assert whole['cache']['evictions'] > 0
    with pytest.raises(FileExistsError):
        train_run.run_training(config)
    with pytest.raises(ContractError, match='configuration'):
        train_run.run_training(replace(resumed_config, learning_rate=.002, output_dir=str(tmp_path / 'bad')))
    with (tmp_path / 'first/training.jsonl').open('r+b') as stream:
        stream.write(b'X')
    with pytest.raises(ContractError, match='digest'):
        train_run.run_training(replace(resumed_config, output_dir=str(tmp_path / 'corrupt')))


def test_failed_update_is_preserved_in_parent_log_and_charged_on_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(train_run, 'source_revision', lambda: 'e' * 40)
    config = config_fixture(tmp_path)
    original_write = train_run.write_record
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('injected after optimizer before publication')
        return original_write(*args, **kwargs)

    monkeypatch.setattr(train_run, 'write_record', interrupted)
    with pytest.raises(RuntimeError, match='injected'):
        train_run.run_training(config)
    failed = json.loads((tmp_path / 'whole/result.json').read_text())
    assert failed['last_completed_update'] == failed['durable_update'] + 1
    monkeypatch.setattr(train_run, 'write_record', original_write)
    resumed = train_run.run_training(replace(config, output_dir=str(tmp_path / 'resume'),
                                            resume_from=str(tmp_path / 'whole/checkpoint.pt')))
    assert resumed['parent']['discarded_updates'] == 1 and resumed['source_onset_exposures'] == 103
    assert resumed['compute_seconds'] >= failed['compute_seconds'] + resumed['segment_seconds'] - 1e-3
    assert json.loads((tmp_path / 'whole/result.json').read_text()) == failed


def test_binary_diagnostic_counts_negative_examples_and_excludes_forced_types():
    model = BoundedModel(ModelConfig(Arm.O1, hidden=8, levels=2, coupling_rank=2))
    source = mixed_chart()
    record = suffix_likelihood(model, source)
    assert record['ln_binary_count'] > record['ln_binary_positive_count'] > 0
    assert record['ln_binary_nll_sum'] > 0 and record['ln_binary_brier_sum'] > 0
    assert record['head_type_binary_count'] < record['ln_binary_count']
    assert np.isfinite(record['head_type_binary_nll_sum'])


def test_footprint_guard_stops_even_when_resident_set_guard_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(train_run, 'source_revision', lambda: 'e' * 40)
    config = config_fixture(tmp_path)
    monkeypatch.setattr(train_run, 'footprint_bytes', lambda: config.footprint_limit_bytes + 1)
    with pytest.raises(ContractError, match='task footprint'):
        train_run.run_training(config)
    result = json.loads((tmp_path / 'whole/result.json').read_text())
    assert result['status'] == 'stopped' and result['last_completed_update'] == 0
