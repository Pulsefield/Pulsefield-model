from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import wave

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from ensomi_model.features.audio import load_audio_file
from ensomi_model.features.mel_base import MUSIC_MEL_CACHE_CONFIG, compute_log_mel_10ms
from ensomi_model.research.joint_audio_continuation import generation, hydra
from ensomi_model.research.joint_audio_continuation.config import JointConfig
from ensomi_model.research.joint_audio_continuation.model import JointAudioModel, JointModelConfig
from ensomi_model.research.oracle_time_continuation.data import source_rows
from ensomi_model.research.scoped_style_modeling.dataset import ContractError
from ensomi_model.research.source_action_modeling.actions import parse_source


def fixture_checkpoint(path):
    config = JointModelConfig(hidden=12, audio_width=8, audio_levels=2, history_levels=2,
                              expansion=2, coupling_rank=3, routing_hidden=20, release_hidden=24)
    model = JointAudioModel(config)
    with torch.no_grad():
        model.audio_mean.fill_(3.)
        model.audio_std.fill_(2.)
        model.timing[-1].weight.zero_()
        model.timing[-1].bias.fill_(100.)
        model.joint.unary.weight.zero_()
        model.joint.unary.bias.fill_(-80.)
        model.joint.unary.bias[5] = 80.  # TAP/TAP within each hand.
        model.joint.matrix.weight.zero_()
        model.joint.matrix.bias.zero_()
    torch.save(dict(model_config=asdict(config), model=model.state_dict(), source_revision='a' * 40,
                    manifest_sha256='b' * 64, config=dict(root='/unavailable/training/corpus')), path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def real_ten_ms_wav(path):
    samples = np.rint(12000 * np.sin(2 * np.pi * 1000 * np.arange(240) / 24000)).astype('<i2')
    with wave.open(str(path), 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24000)
        stream.writeframes(samples.tobytes())


def test_new_audio_inference_needs_no_manifest_cache_source_or_seed(tmp_path, monkeypatch):
    checkpoint, audio = tmp_path / 'model.pt', tmp_path / 'new.wav'
    sha = fixture_checkpoint(checkpoint)
    real_ten_ms_wav(audio)
    root = tmp_path / 'fresh-artifacts'
    called = []

    def clean_revision():
        called.append(True)
        return 'c' * 40

    def forbidden(*args, **kwargs):
        raise AssertionError('New audio must not read a corpus or source beatmap')

    monkeypatch.setattr(generation, '_clean_revision', clean_revision)
    monkeypatch.setattr(generation, '_resource_stop', lambda _: None)
    monkeypatch.setattr(generation, 'load_corpus', forbidden)
    monkeypatch.setattr(generation, 'presentation_header', forbidden)
    settings = JointConfig(mode='infer_audio', root=str(root), run_name='new-song',
        checkpoint_file=str(checkpoint), checkpoint_sha256=sha, audio_file=str(audio), device='cpu',
        max_seconds=30., generation_seed=17)
    result = generation.infer_audio(settings, resolved_yaml='mode: infer_audio\n')
    assert called == [True] and result['status'] == 'completed'
    assert result['rows'] == 11 and result['heads'] == 44
    assert not (root / 'manifest.json').exists() and not (root / 'features').exists()
    output = root / 'inference' / 'new-song'
    recipe = json.loads((output / 'recipe.json').read_text())
    report = json.loads((output / 'result.json').read_text())
    assert recipe['audio']['sha256'] == hashlib.sha256(audio.read_bytes()).hexdigest()
    assert recipe['checkpoint_sha256'] == sha and recipe['source_revision'] == 'c' * 40
    assert recipe['decoded_audio']['samples'] == 240 and recipe['decoded_audio']['duration_ms'] == 10
    assert recipe['frontend']['config']['sample_rate'] == 24000
    assert recipe['frontend']['config']['mel_bins'] == 128
    mel = np.load(output / 'mel.npy')
    expected = compute_log_mel_10ms(load_audio_file(audio, 24000), sample_rate=24000,
                                    config=MUSIC_MEL_CACHE_CONFIG)
    assert mel.shape == (1, 128) and float(mel.max() - mel.min()) > 0
    np.testing.assert_array_equal(mel, expected)
    bundled_audio = Path(result['audio_file'])
    assert bundled_audio.read_bytes() == audio.read_bytes()
    osu = Path(result['osu_file'])
    contents = osu.read_bytes()
    assert b'AudioFilename:audio.wav' in contents and b'joint audio prototype' in contents
    assert b'0,500,4,2,0,100,1,0' in contents and b'not an inferred musical beat grid' in contents
    rows = source_rows(parse_source(contents, hashlib.sha256(contents).hexdigest()).objects)
    assert len(rows) == 11 and [row.time_ms for row in rows] == list(range(11))
    assert report['reparse_pass'] and report['first30_heads_through_ms'] == 7
    published = [json.loads(line) for line in Path(result['events_file']).read_text().splitlines()]
    updates = [item for item in published if item['kind'] == 'update']
    assert [u['row']['time_ms'] for u in updates if u['row']] == list(range(11))
    assert all(not u['completed'] for u in updates[:-1]) and updates[-1]['completed']
    assert published[-1]['kind'] == 'stop' and published[-1]['completed']
    assert published[0]['elapsed_seconds'] < published[-1]['elapsed_seconds']
    assert report['profile']['startup_coverage_ms'] == 10
    for key in ('audio_decode_seconds', 'mel_seconds', 'model_load_seconds', 'audio_encode_seconds',
                'startup_from_audio_seconds', 'first30_heads_from_audio_seconds'):
        assert report['profile'][key] >= 0.
    with pytest.raises(FileExistsError):
        generation.infer_audio(settings)


def test_source_free_header_sanitizes_title_without_creating_sections():
    header = generation._source_free_header('song\n[Events]\r\n.wav').decode()
    assert header.count('[General]\n') == 1 and '\n[Events]\n' not in header
    assert 'Title:song [Events] (joint audio prototype)' in header


def test_audio_inference_preserves_clean_source_and_preprocessing_time_guards(tmp_path, monkeypatch):
    settings = JointConfig(mode='infer_audio', root=str(tmp_path / 'runs'), run_name='limited',
        checkpoint_file='unopened.pt', checkpoint_sha256='0' * 64, audio_file='unopened.wav',
        device='cpu', max_seconds=1e-12)
    monkeypatch.setattr(generation, '_clean_revision', lambda: 'd' * 40)
    monkeypatch.setattr(generation, '_resource_stop', lambda _: None)
    result = generation.infer_audio(settings)
    assert result['status'] == 'capped' and result['stop_reason'] == 'time_limit'
    assert result['osu_file'] is None
    report = json.loads(Path(result['result_file']).read_text())
    assert report['phase'] == 'input_verification'

    def dirty_revision():
        raise ContractError('Experiments require a clean committed source checkout')

    monkeypatch.setattr(generation, '_clean_revision', dirty_revision)
    with pytest.raises(ContractError, match='clean committed'):
        generation.infer_audio(replace(settings, run_name='dirty'))
    assert not (Path(settings.root) / 'inference' / 'dirty').exists()


def test_hydra_audio_fields_reach_only_the_audio_runner(tmp_path, monkeypatch, capsys):
    composed = hydra.compose_config(['mode=infer_audio', 'audio_file=new.wav', 'checkpoint_file=small.pt',
        'checkpoint_sha256=' + '0' * 64, 'device=cpu', 'root=' + str(tmp_path)])
    received = []

    def infer(settings, *, resolved_yaml):
        received.append((settings, resolved_yaml))
        return dict(status='fixture')

    monkeypatch.setattr(generation, 'infer_audio', infer)
    hydra.cli.__wrapped__(OmegaConf.structured(composed))
    assert received[0][0] == composed and 'audio_file: new.wav' in received[0][1]
    assert json.loads(capsys.readouterr().out)['status'] == 'fixture'
    for change in (dict(audio_file=''), dict(checkpoint_sha256=''), dict(mode='generate'),
                   dict(full_wait_supervision=True, fixed_train_queries=4)):
        with pytest.raises(ValueError):
            replace(composed, **change).validate()
