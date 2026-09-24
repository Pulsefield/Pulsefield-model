from dataclasses import replace
from importlib.resources import files
import json

import numpy as np
import pytest
import torch

from ensomi_model.research.joint_audio_continuation.context_config import ContextTrainConfig
from ensomi_model.research.joint_audio_continuation.context_hydra import compose_config
from ensomi_model.research.joint_audio_continuation.context_model import ContextAudioModel
from ensomi_model.research.joint_audio_continuation import context_training as training
from ensomi_model.research.joint_audio_continuation.data import digest, frontend_identity
from ensomi_model.research.joint_audio_continuation.generation import load_model, rollout
from .test_batching import chart
from .test_context_intervals import context_config


def corpus():
    source = chart([0, 300, 900], [(2, 0, 0, 0), (0, 1, 0, 0), (3, 0, 0, 0)], 1321)
    train = replace(source, entry=dict(source.entry, audio_sha256='b' * 64))
    alternative = replace(train, entry=dict(train.entry, source_sha256='d' * 64))
    val = replace(source, entry=dict(source.entry, source_sha256='c' * 64,
                                    group_id='validation-group', split='validation', audio_sha256='e' * 64))
    return [train, alternative, val]


def test_packaged_context_configuration_and_strict_projection():
    assert files('ensomi_model.configs.hydra').joinpath('joint_audio_context.yaml').is_file()
    cfg = compose_config(['global_audio=true', 'bounded_timing=true', 'updates=2', 'validation_songs=1'])
    assert cfg.global_audio and cfg.bounded_timing and cfg.updates == 2 and cfg.validation_songs == 1
    with pytest.raises(ValueError, match='Unknown'):
        compose_config(['+unused=1'])
    with pytest.raises(ValueError, match='frozen plan'):
        compose_config(['updates=1201'])
    with pytest.raises(ValueError, match='positive integer'):
        compose_config(['songs_per_update=0'])


def test_protocol_pairs_architecture_cells_preserves_alternatives_and_rejects_change(tmp_path):
    cfg = ContextTrainConfig(root=str(tmp_path), plan_updates=20, updates=2, interval_ms=333)
    charts = corpus()
    p, identity = training.freeze_protocol(charts, cfg)
    same, other_identity = training.freeze_protocol(charts, replace(cfg, global_audio=True, bounded_timing=True))
    assert p == same and identity == other_identity
    samples = [r for update in p['updates'] for song in update for r in song]
    assert {r['source_sha256'] for r in samples} == {'a' * 64, 'd' * 64}
    assert all(r['source_sha256'] == 'c' * 64 for r in p['validation'])
    assert len(samples) == 80
    with pytest.raises(ValueError, match='differs'):
        training.freeze_protocol(charts, replace(cfg, interval_ms=500))


def test_real_training_runner_consumes_full_audio_normalizer_plan_and_serializes_native_model(tmp_path, monkeypatch):
    charts = corpus()
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}\n')
    norm = dict(mean=[.7] * 128, std=[1.3] * 128, frontend=frontend_identity(),
                frame_count=len(charts[0].mel), audio_sha256=['b' * 64])
    norm_file = tmp_path / 'normalization.json'
    norm_file.write_text(json.dumps(norm))
    cfg = ContextTrainConfig(root=str(tmp_path), manifest_sha256=digest(manifest),
        normalization_file=str(norm_file), normalization_sha256=digest(norm_file),
        global_audio=True, bounded_timing=True, plan_updates=2, updates=2, songs_per_update=1,
        intervals_per_song=2, interval_ms=500, validation_every=2, device='cpu', max_seconds=60)
    monkeypatch.setattr(training, 'revision', lambda: 'f' * 40)
    monkeypatch.setattr(training, 'load_corpus', lambda root: (charts, norm))
    monkeypatch.setattr(training, 'initialize_from_r1', lambda *args: {'copied': []})
    monkeypatch.setattr(training, '_resource_stop', lambda root: None)
    encoded_lengths = []

    def small_model(config):
        model = ContextAudioModel(context_config(global_audio=config.global_audio, bounded_timing=config.bounded_timing))
        model.context.register_forward_pre_hook(lambda module, args: encoded_lengths.append((args[0].shape[1], int(args[1].sum()))))
        return model

    monkeypatch.setattr(training, 'ContextAudioModel', small_model)
    result = training.train(cfg, resolved_yaml='global_audio: true\nbounded_timing: true\n')
    assert result['status'] == 'completed' and result['intervals'] == 4
    assert encoded_lengths and {real for padded, real in encoded_lengths} == {len(charts[0].mel)}
    assert all(padded >= real for padded, real in encoded_lengths)
    directory = tmp_path / 'context-training' / cfg.run_name
    assert (directory / 'resolved.yaml').read_text().startswith('global_audio: true')
    assert json.loads((directory / 'config.json').read_text())['interval_ms'] == 500
    freeze = json.loads((directory / 'freeze.json').read_text())
    assert freeze['normalization_sha256'] == digest(norm_file)
    model, metadata = load_model(directory / 'last.pt', result['checkpoint_sha256'])
    torch.testing.assert_close(model.audio_mean, torch.full((128,), .7))
    torch.testing.assert_close(model.audio_std, torch.full((128,), 1.3))
    assert metadata['model_config']['global_audio'] and metadata['model_config']['bounded_timing']
    generation = rollout(model, charts[0].mel, charts[0].duration_ms, seed=3, chunk_ms=117, max_seconds=10)
    assert generation.completed and not any(generation.metrics['open_lanes'])
    repartitioned = rollout(model, charts[0].mel, charts[0].duration_ms, seed=3, chunk_ms=500, max_seconds=10)
    assert repartitioned.rows == generation.rows and repartitioned.completed
    with pytest.raises(FileExistsError):
        training.train(cfg)
