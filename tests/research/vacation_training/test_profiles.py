from dataclasses import replace

import pytest
import torch

from pulsefield_model.research.bounded_typed_continuation.train_hydra import compose_config
from pulsefield_model.research.bounded_typed_continuation.generate_hydra import compose_config as generate
from pulsefield_model.research.bounded_typed_continuation.model import BoundedModel
from pulsefield_model.research.bounded_typed_continuation import train_run
from pulsefield_model.research.scoped_style_modeling.dataset import ContractError
from pulsefield_model.research.vacation_training.hydra import compose_config as vacation
from bounded_typed_continuation.test_train import compare_states, config_fixture
from bounded_typed_continuation.test_data import mixed_chart
from pulsefield_model.research.bounded_typed_continuation.data import SourceInterval


def test_teacher_profiles_are_explicit_and_do_not_expand_small_defaults():
    cfg = compose_config(config_name='bounded_typed_train_teacher35m')
    assert cfg.execution_profile == 'teacher35m' and cfg.microbatch_size == 1
    assert cfg.model.hidden == 512 and cfg.model.seed_context == 'observed'
    with torch.device('meta'):
        model = BoundedModel(cfg.model)
    assert sum(p.numel() for p in model.parameters()) == 35178768
    assert generate(config_name='bounded_typed_generate_teacher35m').resources.checkpoint_max_bytes == 1024**3
    for overrides in (['model.hidden=512'], ['execution_profile=teacher35m'], ['execution_profile=unknown']):
        with pytest.raises(ContractError):
            compose_config(overrides)
    for overrides in (['microbatch_size=2'], ['model.row_consequence=frontier2'], ['resources.checkpoint_max_bytes=1024']):
        with pytest.raises(ContractError):
            compose_config(overrides, config_name='bounded_typed_train_teacher35m')
    assert vacation(['mode=status']).mode == 'status'
    with pytest.raises((ContractError, ValueError)):
        vacation(['mode=status', '+teacher.unused=1'])
    with pytest.raises(ContractError, match='pinned'):
        vacation()


def test_safe_boundary_pause_then_resume_matches_uninterrupted_optimizer(tmp_path, monkeypatch):
    monkeypatch.setattr(train_run, 'source_revision', lambda: 'e' * 40)
    cfg = config_fixture(tmp_path)
    train_run.run_training(cfg)
    count = 0
    def stop():
        nonlocal count
        count += 1
        return 'pause_file' if count == 2 else None
    first = replace(cfg, output_dir=str(tmp_path / 'pause'))
    result = train_run.run_training(first, stop_requested=stop)
    assert result['pause_reason'] == 'pause_file' and result['durable_update'] == 1
    resumed = replace(cfg, output_dir=str(tmp_path / 'resumed'), resume_from=str(tmp_path / 'pause/checkpoint.pt'))
    train_run.run_training(resumed)
    a = torch.load(tmp_path / 'whole/checkpoint.pt', weights_only=True)
    b = torch.load(tmp_path / 'resumed/checkpoint.pt', weights_only=True)
    for key in ('model', 'optimizer', 'torch_rng', 'cursor', 'source_onset_exposures', 'coverage'):
        compare_states(a[key], b[key])


def test_response_profile_matches_the_recorded_small_candidate_and_trains_all_residuals():
    cfg = compose_config(config_name='bounded_typed_train_r1_response')
    model = BoundedModel(cfg.model)
    assert sum(p.numel() for p in model.parameters()) == 3084432
    assert cfg.execution_profile == 'small' and cfg.model_seed == 172
    queue = vacation(['mode=status'], config_name='vacation_training_r1_response')
    queue.teacher.training.validate()
    assert queue.teacher.training.model == cfg.model
    assert queue.teacher.training.cpu_threads == 1
    assert not queue.audio.enabled and not queue.stress.enabled
    assert queue.teacher.milestones[-1] == 6750000
    teacher = compose_config(config_name='bounded_typed_train_teacher35m')
    assert teacher.model.hidden == 512 and teacher.model.head_routing == teacher.model.release_routing == 'none'
    assert teacher.model.row_consequence == 'none'
    source = mixed_chart()
    interval = SourceInterval(source, 0, len(source.onsets))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    train_run.measure(model, [interval], candidate_budget=cfg.candidate_budget, backward=True,
                      denominator=interval.onset_count, check=lambda _: None)
    outputs = [model.route_residual.score[-1].weight, model.release_residual.score[-1].weight,
               model.row_consequence.output.weight]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0 for p in outputs)
    optimizer.step()
    assert all(torch.count_nonzero(p) > 0 for p in outputs)
