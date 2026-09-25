from dataclasses import replace
from importlib.resources import files
import json

import pytest
import torch

from ensomi_model.research.bounded_typed_continuation.contract import Arm
from ensomi_model.research.bounded_typed_continuation.model import BoundedModel, ModelConfig
from ensomi_model.research.joint_audio_continuation.data import digest, frontend_identity
from ensomi_model.research.planned_audio_continuation.config import PlannedTrainConfig
from ensomi_model.research.planned_audio_continuation.generation import load_model, rollout
from ensomi_model.research.planned_audio_continuation.hydra import compose_config
from ensomi_model.research.planned_audio_continuation.model import PlannedAudioModel, initialize_from_r1
from ensomi_model.research.planned_audio_continuation import training
from .test_distribution import config, source


def test_packaged_settings_reject_flat_timing_and_unused_fields():
    assert files('ensomi_model.configs.hydra').joinpath('planned_audio.yaml').is_file()
    cfg = compose_config(['updates=32', 'validation_every=32', 'validation_songs=6', 'run_name=probe'])
    assert cfg.updates == 32 and cfg.global_audio and not cfg.bounded_timing
    assert cfg.validation_songs == 6 and cfg.run_name == 'probe'
    bounded = compose_config(['bounded_head=true', 'head_bound=3', 'head_decay_ms=400'])
    assert bounded.bounded_head and bounded.head_bound == 3 and bounded.head_decay_ms == 400
    assert compose_config(['condition_full_holds=true']).condition_full_holds
    for override in ('global_audio=false', 'bounded_timing=true'):
        with pytest.raises(ValueError, match='separate skeleton'):
            compose_config([override])
    with pytest.raises(ValueError, match='Unknown'):
        compose_config(['+unused=1'])


def test_frontier2_is_actually_transferred_and_has_the_inherited_learning_rate(tmp_path):
    cfg = config()
    source_model = BoundedModel(ModelConfig(Arm.R1, hidden=cfg.hidden, levels=cfg.history_levels,
        expansion=cfg.expansion, coupling_rank=cfg.coupling_rank, head_routing='residual',
        routing_hidden=cfg.routing_hidden, release_routing='residual', release_hidden=cfg.release_hidden,
        row_consequence='frontier2'))
    torch.nn.init.normal_(source_model.row_consequence.output.weight, std=.1)
    path = tmp_path / 'r1.pt'
    torch.save(dict(config=dict(model=dict(arm='r1', row_consequence='frontier2')), model=source_model.state_dict()), path)
    model = PlannedAudioModel(cfg)
    receipt = initialize_from_r1(model, path, digest(path))
    for name, value in source_model.row_consequence.state_dict().items():
        torch.testing.assert_close(value, model.row_consequence.state_dict()[name], rtol=0, atol=0)
        assert 'row_consequence.' + name in receipt['copied']
        assert 'row_consequence.' + name not in receipt['omitted']
    assert 'row_consequence' not in receipt['omitted_modules']
    groups = training.optimizer_groups(model, PlannedTrainConfig())
    rates = {name: group['lr'] for group in groups for name in group['param_names']}
    assert len(rates) == len(list(model.parameters()))
    assert rates['row_consequence.output.weight'] == 3e-5
    assert rates['head_temporal.input.weight'] == 3e-4
    assert rates['release_clock.0.weight'] == 3e-4


@pytest.mark.parametrize('minimum_gap', [0, 1])
def test_runner_consumes_full_audio_plan_and_restores_the_native_endpoint(tmp_path, monkeypatch, minimum_gap):
    src = source()
    train = replace(src, entry=dict(src.entry, audio_sha256='b' * 64))
    val = replace(src, entry=dict(src.entry, source_sha256='c' * 64, group_id='val',
                                split='validation', audio_sha256='d' * 64))
    charts = [train, val]
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}\n')
    norm = dict(mean=[.7] * 128, std=[1.3] * 128, frontend=frontend_identity(),
                frame_count=len(src.mel), audio_sha256=['b' * 64])
    norm_file = tmp_path / 'normalization.json'
    norm_file.write_text(json.dumps(norm))
    cfg = PlannedTrainConfig(root=str(tmp_path), manifest_sha256=digest(manifest),
        normalization_file=str(norm_file), normalization_sha256=digest(norm_file),
        plan_updates=2, updates=2, songs_per_update=1, intervals_per_song=2,
        interval_ms=7, validation_every=2, device='cpu', max_seconds=60,
        bounded_head=True, head_bound=3, head_decay_ms=400, condition_full_holds=True,
        minimum_action_gap_ms=minimum_gap)
    monkeypatch.setattr(training, 'revision', lambda: 'f' * 40)
    monkeypatch.setattr(training, 'load_corpus', lambda root: (charts, norm))
    monkeypatch.setattr(training, 'initialize_from_r1', lambda *args: {'copied': []})
    monkeypatch.setattr(training, '_resource_stop', lambda root: None)
    lengths = []

    def small_model(settings):
        model = PlannedAudioModel(replace(config(), bounded_head=settings.bounded_head,
            head_bound=settings.head_bound, head_decay_ms=settings.head_decay_ms,
            condition_full_holds=settings.condition_full_holds,
            minimum_action_gap_ms=settings.minimum_action_gap_ms,
            lookahead=9 if settings.minimum_action_gap_ms else 2))
        model.context.register_forward_pre_hook(lambda module, args: lengths.append(int(args[1].sum())))
        return model

    monkeypatch.setattr(training, 'PlannedAudioModel', small_model)
    result = training.train(cfg, resolved_yaml='updates: 2\n')
    assert result['status'] == 'completed' and result['intervals'] == 4
    assert set(lengths) == {len(src.mel)}
    assert all(value > 0 for value in result['parameter_update_l2'].values())
    directory = tmp_path / 'planned-training' / cfg.run_name
    assert (directory / 'resolved.yaml').read_text() == 'updates: 2\n'
    assert json.loads((directory / 'config.json').read_text())['interval_ms'] == 7
    model, metadata = load_model(directory / 'last.pt', result['checkpoint_sha256'])
    torch.testing.assert_close(model.audio_mean, torch.full((128,), .7))
    torch.testing.assert_close(model.audio_std, torch.full((128,), 1.3))
    assert metadata['source_revision'] == 'f' * 40
    assert model.config.bounded_head and model.config.head_bound == 3 and model.config.head_decay_ms == 400
    assert model.config.condition_full_holds
    assert model.config.minimum_action_gap_ms == minimum_gap
    generated = rollout(model, src.mel, src.duration_ms, seed=3, max_seconds=10)
    assert generated.completed and not any(generated.metrics['open_lanes'])
    with pytest.raises(FileExistsError):
        training.train(cfg)
