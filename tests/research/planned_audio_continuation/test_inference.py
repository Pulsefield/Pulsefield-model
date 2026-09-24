from dataclasses import asdict, replace
from importlib.resources import files
import json
from pathlib import Path
import subprocess
import sys
import wave

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from ensomi_model.features.audio import load_audio_file
from ensomi_model.features.mel_base import MUSIC_MEL_CACHE_CONFIG, compute_log_mel_10ms
from ensomi_model.research.joint_audio_continuation import data
from ensomi_model.research.joint_audio_continuation.generation import save_rollout
from ensomi_model.research.oracle_time_continuation.replay import ExactReplayState, commit
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.planned_audio_continuation import inference, infer_hydra
from ensomi_model.research.planned_audio_continuation.generation import load_model, rollout
from ensomi_model.research.planned_audio_continuation.inference_config import AudioInferenceConfig
from ensomi_model.research.planned_audio_continuation.model import PlannedAudioModel
from .test_distribution import config


def inputs(tmp_path, monkeypatch):
    checkpoint, audio = tmp_path / 'model.pt', tmp_path / 'music.wav'
    torch.manual_seed(75)
    model = PlannedAudioModel(config())
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.audio_mean.fill_(.5)
        model.audio_std.fill_(2.)
        model.timing[-1].bias.fill_(100.)
        model.joint.unary.bias[10] = 80.  # Favor held heads while retaining legal support.
    torch.save(dict(format='joint-audio/planned-v1', model_config=asdict(config()), model=model.state_dict(),
                    source_revision='a' * 40, manifest_sha256='b' * 64,
                    config=dict(root='/missing/training/corpus')), checkpoint)
    samples = np.rint(12000 * np.sin(2 * np.pi * 1000 * np.arange(1440) / 24000)).astype('<i2')
    with wave.open(str(audio), 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24000)
        stream.writeframes(samples.tobytes())
    monkeypatch.setattr(inference, '_clean_revision', lambda: 'c' * 40)
    monkeypatch.setattr(inference, '_resource_stop', lambda path: None)
    return AudioInferenceConfig(audio_file=str(audio), checkpoint_file=str(checkpoint),
        checkpoint_sha256=data.digest(checkpoint), output_dir=str(tmp_path / 'run'), max_seconds=30.,
        seed=23, chunk_ms=7, head_chunk_ms=11, startup_coverage_ms=10, startup_min_rows=2)


def read(path):
    return json.loads(Path(path).read_text())


def test_audio_stream_replays_exactly_without_training_data_and_keeps_native_samples(tmp_path, monkeypatch):
    cfg = inputs(tmp_path, monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError('Source-chart-free inference must not open a corpus')

    monkeypatch.setattr(data, 'load_corpus', forbidden)
    observed = []

    def receive(event):
        # Publication is already readable when an incremental consumer runs.
        assert json.loads((Path(cfg.output_dir) / 'events.jsonl').read_text().splitlines()[-1]) == event
        observed.append(event)

    result = inference.infer_audio(cfg, resolved_yaml='seed: 23\n', on_event=receive)
    assert result['completed'] and result['status'] == 'completed'
    events = [json.loads(x) for x in Path(result['events_file']).read_text().splitlines()]
    assert events == observed and [e['sequence'] for e in events] == list(range(len(events)))
    assert events[0]['format'] == inference.STREAM_FORMAT and events[-1]['kind'] == 'stop'
    state, coverage, ready = ExactReplayState(), -1, False
    updates = [e for e in events if e['kind'] == 'update']
    for e in updates:
        assert e['coverage_ms'] > coverage
        if e['row']:
            row = CompleteRow(e['row']['time_ms'], tuple(e['row']['actions']))
            assert row.time_ms > coverage and row.time_ms <= e['coverage_ms']
            state = commit(state, row, is_terminal=e['completed'])
        assert not ready or e['playback_ready']
        ready = e['playback_ready']
        coverage = e['coverage_ms']
    assert coverage == 60 and not any(state.occupancy) and ready
    assert any(e['playback_ready'] and not e['completed'] for e in updates)
    assert any(2 in e['row']['actions'] for e in updates if e['row'])
    assert all(set(e['row']) == {'time_ms', 'actions'} for e in updates if e['row'])
    assert Path(result['audio_file']).read_bytes() == Path(cfg.audio_file).read_bytes()
    assert read(Path(cfg.output_dir) / 'chart/result.json')['reparse_pass']
    recipe = read(Path(cfg.output_dir) / 'recipe.json')
    assert recipe['checkpoint']['config']['root'] == '/missing/training/corpus'
    assert recipe['decoded_audio']['samples'] == 1440 and recipe['mel']['shape'] == [6, 128]
    assert read(Path(cfg.output_dir) / 'config.json') == asdict(cfg)
    assert (Path(cfg.output_dir) / 'resolved.yaml').read_text() == 'seed: 23\n'
    model, _ = load_model(cfg.checkpoint_file, cfg.checkpoint_sha256)
    mel = compute_log_mel_10ms(load_audio_file(cfg.audio_file, 24000), sample_rate=24000,
                              config=MUSIC_MEL_CACHE_CONFIG)
    direct = rollout(model, mel, 60, seed=cfg.seed, chunk_ms=cfg.chunk_ms, head_chunk_ms=cfg.head_chunk_ms)
    saved = save_rollout(tmp_path / 'direct', direct)
    assert saved['rows_sha256'] == read(Path(result['result_file']))['rows_sha256']
    with pytest.raises(FileExistsError):
        inference.infer_audio(cfg)


def test_row_limit_preserves_open_holds_without_fabricated_completion(tmp_path, monkeypatch):
    cfg = replace(inputs(tmp_path, monkeypatch), max_rows=1)
    result = inference.infer_audio(cfg)
    assert result['status'] == 'capped' and result['stop_reason'] == 'row_limit'
    assert not result['completed'] and result['osu_file'] is None
    saved = read(Path(cfg.output_dir) / 'chart/result.json')
    assert saved['rows'] == 1 and any(saved['open_lanes'])
    events = [json.loads(x) for x in Path(result['events_file']).read_text().splitlines()]
    assert events[-1]['kind'] == 'stop' and not events[-1]['completed']
    assert not any(e['playback_ready'] for e in events if e['kind'] == 'update')


def test_short_song_can_finish_before_startup_quota_and_settings_reach_rollout(tmp_path, monkeypatch):
    cfg = replace(inputs(tmp_path, monkeypatch), startup_min_rows=10000, correct_short_attacks=True)
    original = inference.rollout
    received = []

    def capture(model, mel, duration, **kwargs):
        received.append(kwargs)
        return original(model, mel, duration, **kwargs)

    monkeypatch.setattr(inference, 'rollout', capture)
    result = inference.infer_audio(cfg)
    events = [json.loads(x) for x in Path(result['events_file']).read_text().splitlines()]
    updates = [e for e in events if e['kind'] == 'update']
    assert all(not e['playback_ready'] for e in updates[:-1])
    assert updates[-1]['completed'] and updates[-1]['playback_ready']
    for field in ('seed', 'chunk_ms', 'head_chunk_ms', 'max_rows', 'correct_short_attacks'):
        assert received[0][field] == getattr(cfg, field)
    assert 0 < received[0]['max_seconds'] < cfg.max_seconds
    assert callable(received[0]['stop_callback']) and callable(received[0]['on_update'])


def test_preprocessing_stop_and_consumer_failure_leave_honest_streams(tmp_path, monkeypatch):
    cfg = inputs(tmp_path, monkeypatch)
    result = inference.infer_audio(replace(cfg, max_seconds=1e-12))
    assert result['status'] == 'capped' and result['stop_reason'] == 'time_limit'
    assert result['rows'] == 0 and result['osu_file'] is None
    failed_cfg = replace(cfg, output_dir=str(tmp_path / 'consumer-failure'))

    def consumer(event):
        if event['kind'] == 'update':
            raise RuntimeError('consumer disconnected')

    with pytest.raises(RuntimeError, match='consumer disconnected'):
        inference.infer_audio(failed_cfg, on_event=consumer)
    events = [json.loads(x) for x in (Path(failed_cfg.output_dir) / 'events.jsonl').read_text().splitlines()]
    assert events[-1]['kind'] == 'error' and not events[-1]['completed']
    assert len([e for e in events if e['kind'] == 'update']) == 1
    assert read(Path(failed_cfg.output_dir) / 'failure.json')['rows'] == 1
    assert not (Path(failed_cfg.output_dir) / 'chart/generated.osu').exists()


def test_packaged_inference_projection_and_cli_callback(tmp_path, monkeypatch, capsys):
    assert files('ensomi_model.configs.inference').joinpath('planned_audio.yaml').is_file()
    args = ['audio_file=music.wav', 'checkpoint_file=model.pt', 'checkpoint_sha256=' + 'a'*64,
            'output_dir=' + str(tmp_path), 'seed=99', 'startup_min_rows=12', 'head_chunk_ms=73']
    cfg = infer_hydra.compose_config(args)
    received = []

    def run(settings, *, resolved_yaml, on_event):
        received.append((settings, resolved_yaml))
        on_event(dict(kind='fixture'))

    monkeypatch.setattr(inference, 'infer_audio', run)
    infer_hydra.cli.__wrapped__(OmegaConf.structured(cfg))
    assert received[0][0] == cfg and 'head_chunk_ms: 73' in received[0][1]
    assert json.loads(capsys.readouterr().out) == {'kind': 'fixture'}
    with pytest.raises(ValueError, match='Unknown'):
        infer_hydra.compose_config(args + ['+unused=1'])
    for change in ('device=cuda', 'startup_min_rows=-1', 'max_seconds=0', 'checkpoint_sha256=bad'):
        with pytest.raises(ValueError):
            infer_hydra.compose_config(args + [change])


def test_help_and_config_inspection_do_not_import_torch():
    for argument in ('--help', '--cfg=job'):
        script = f'''
import importlib.abc, runpy, sys
class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Help imported Torch')
sys.meta_path.insert(0, BlockTorch())
sys.argv = ['planned_audio', {argument!r}]
runpy.run_module('ensomi_model.research.planned_audio_continuation.infer_hydra', run_name='__main__')
'''
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert 'startup_coverage_ms' in result.stdout and 'checkpoint_sha256' in result.stdout
