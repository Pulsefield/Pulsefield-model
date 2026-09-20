"""Exercise the declared teacher envelope through the ordinary durable runners."""
from dataclasses import replace
import gc
import json
from pathlib import Path

import torch
import pytest

from pulsefield_model.research.bounded_typed_continuation import train_run, generate_run
from pulsefield_model.research.bounded_typed_continuation.generate_config import GenerateConfig
from pulsefield_model.research.bounded_typed_continuation.train_hydra import compose_config
from pulsefield_model.research.oracle_time_continuation.storage import file_digest
from bounded_typed_continuation.test_train import compare_states
from .test_queue import configuration


@pytest.mark.parametrize('preset,profile_name', [('teacher35m', 'teacher35m'), ('r1_response', 'small')])
def test_declared_profile_cpu_training_resume_and_native_generation(tmp_path, monkeypatch, preset, profile_name):
    for module in (train_run, generate_run):
        monkeypatch.setattr(module, 'source_revision', lambda: 'e' * 40)
    queue = configuration(tmp_path)
    small = queue.teacher.training
    profile = compose_config(config_name='bounded_typed_train_' + preset)
    cfg = replace(profile, plan_file=small.plan_file, plan_sha256=small.plan_sha256,
        source_cache_dir=small.source_cache_dir, output_dir=str(tmp_path / 'whole'), report_every=100,
        resources=replace(profile.resources, min_available_bytes=1024**2, disk_reserve_bytes=1024**2))
    whole = train_run.run_training(cfg)
    assert whole['durable_onset_exposures'] == 103
    paused = train_run.run_training(replace(cfg, output_dir=str(tmp_path / 'first'), stop_after_checkpoint=37))
    assert paused['durable_onset_exposures'] == 37
    resumed = train_run.run_training(replace(cfg, output_dir=str(tmp_path / 'second'),
                                             resume_from=str(tmp_path / 'first/checkpoint.pt')))
    assert resumed['durable_onset_exposures'] == 103
    a = torch.load(tmp_path / 'whole/checkpoint.pt', weights_only=True)
    b = torch.load(tmp_path / 'second/checkpoint.pt', weights_only=True)
    for key in ('model', 'optimizer', 'torch_rng', 'cursor', 'source_onset_exposures', 'coverage'):
        compare_states(a[key], b[key])
    assert (tmp_path / 'second/checkpoint.pt').stat().st_size < cfg.resources.checkpoint_max_bytes
    del a, b
    gc.collect()
    case = json.loads(Path(queue.teacher.evaluation_file).read_text())['native_cases'][0]
    settings = GenerateConfig(execution_profile=profile_name, checkpoint_file=str(tmp_path / 'second/checkpoint.pt'),
        checkpoint_sha256=resumed['checkpoint_sha256'], condition_file=case['condition_file'],
        condition_sha256=case['condition_sha256'], output_dir=str(tmp_path / 'generated'), resources=cfg.resources)
    generated = generate_run.run_generation(settings)
    calls = 0
    def stop():
        nonlocal calls
        calls += 1
        return 'pause_file' if calls == 9 else None
    first = generate_run.run_generation(replace(settings, output_dir=str(tmp_path / 'gen-first')), stop_requested=stop)
    assert first['status'] == 'paused' and first['pause_reason'] == 'pause_file'
    second = generate_run.run_generation(replace(settings, output_dir=str(tmp_path / 'gen-second'),
        resume_from=first['checkpoint_path'], resume_sha256=first['checkpoint_sha256']))
    assert generated['status'] == second['status'] == 'completed'
    assert file_digest(tmp_path / 'generated/rows.jsonl') == file_digest(tmp_path / 'gen-second/rows.jsonl')
    assert file_digest(tmp_path / 'generated/generated.osu') == file_digest(tmp_path / 'gen-second/generated.osu')
