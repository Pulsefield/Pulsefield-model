from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from pulsefield_model.research.bounded_typed_continuation import train_run, generate_run
from pulsefield_model.research.bounded_typed_continuation.condition import GenerationCondition
from pulsefield_model.research.bounded_typed_continuation.contract import Arm
from pulsefield_model.research.bounded_typed_continuation.corpus import ChartCache, read_plan
from pulsefield_model.research.vacation_training import run
from pulsefield_model.research.vacation_training.config import VacationConfig, AudioConfig, TeacherConfig, StressConfig
from pulsefield_model.research.vacation_training.control import Control, publish_json, run_lock
from pulsefield_model.research.oracle_time_continuation.storage import file_digest
from pulsefield_model.research.scoped_style_modeling.dataset import ContractError
from bounded_typed_continuation.test_train import config_fixture


def configuration(tmp_path):
    train = config_fixture(tmp_path)
    train = replace(train, model=replace(train.model, arm=Arm.R1))
    plan = read_plan(train.plan_file, train.plan_sha256)
    sha, pin = next(iter(plan['sources'].items()))
    val_sha = '9' * 64
    val = json.loads(json.dumps(pin))
    val['identity'].update(source_sha256=val_sha, arrangement_sha256='8'*64, group_id='validation-group', split='validation')
    folder = Path(train.source_cache_dir) / val_sha
    shutil.copytree(Path(train.source_cache_dir) / sha, folder)
    metadata = json.loads((folder / 'metadata.json').read_text())
    metadata['identity'] = val['identity']
    publish_json(folder / 'metadata.json', metadata)
    val['metadata_sha256'] = file_digest(folder / 'metadata.json')
    sources = {sha: pin, val_sha: val}
    source = ChartCache(train.source_cache_dir, sources).get(val_sha)
    condition = GenerationCondition(Arm.R1, source.typed.timing,
        tuple(source.row(i) for i in range(source.seed_rows)), source.state(Arm.R1, source.seed_rows).known_ends)
    condition_path = tmp_path / 'condition.json'
    condition_sha = publish_json(condition_path, condition.payload())
    cases = [dict(id='validation-seed17', split='validation', group_id='validation-group', source_sha256=val_sha,
                  condition_file=str(condition_path), condition_sha256=condition_sha, seed=17)]
    evaluation = dict(format='vacation/teacher-evaluation-v1', source_cache_dir=train.source_cache_dir,
        sources=sources, windows=[dict(source_sha256=s, first_onset=0, onset_count=8) for s in sources], native_cases=cases)
    path = tmp_path / 'evaluation.json'
    digest = publish_json(path, evaluation)
    return VacationConfig(output_dir=str(tmp_path / 'queue'), require_ac_power=False, check_thermal=False,
        disk_reserve_bytes=1024**2, resources=replace(train.resources, disk_reserve_bytes=1024**2),
        audio=AudioConfig(enabled=False), stress=StressConfig(enabled=False),
        teacher=TeacherConfig(base_plan_file=train.plan_file, base_plan_sha256=train.plan_sha256,
            evaluation_file=str(path), evaluation_sha256=digest, milestones=[37, 103], max_seconds=60, training=train))


def test_real_multistage_teacher_freeze_pause_resume_readouts_and_no_rerun(tmp_path, monkeypatch):
    for module in (run, train_run, generate_run):
        monkeypatch.setattr(module, 'source_revision', lambda: 'e' * 40)
    cfg = configuration(tmp_path)
    assert run.run_queue(cfg)['status'] == 'preflight_passed'
    root = Path(cfg.output_dir)
    (root / 'PAUSE').touch()
    paused = run.run_queue(replace(cfg, mode='run'))
    assert paused['status'] == 'paused' and not (root / 'teacher').exists()
    (root / 'PAUSE').unlink()
    result = run.run_queue(replace(cfg, mode='resume'))
    assert result['status'] == 'completed'
    teacher = result['stages']['teacher']
    assert teacher['result']['onset_exposures'] == 103
    assert len(teacher['state']['segments']) == 2
    for milestone in (37, 103):
        directory = root / 'teacher' / f'readout-{milestone}'
        report = json.loads((directory / 'likelihood.json').read_text())
        assert set(report['totals']) == {'train', 'validation'}
        generated = directory / 'native/validation/validation-seed17/segment-00000'
        assert (generated / 'generated.osu').is_file()
        assert (generated / 'diagnostics.json').is_file()
    before = teacher['state']
    again = run.run_queue(replace(cfg, mode='resume'))
    assert again['stages']['teacher']['state'] == before
    with pytest.raises(ContractError, match='configuration'):
        run.run_queue(replace(cfg, mode='resume', max_seconds=400))
    assert run.run_queue(VacationConfig(mode='status', output_dir=cfg.output_dir))['status'] == 'completed'


def test_exclusive_run_ownership_and_queue_expiry(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(ContractError, match='Another process'):
            with run_lock(tmp_path):
                pytest.fail('second writer entered')
    config = VacationConfig(max_seconds=1, require_ac_power=False, check_thermal=False)
    assert Control(config, tmp_path, 0, 10000).boundary() == 'queue_time_limit'


def test_cross_split_validation_and_changed_input_are_rejected_before_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'source_revision', lambda: 'e' * 40)
    cfg = configuration(tmp_path)
    manifest = json.loads(Path(cfg.teacher.evaluation_file).read_text())
    val = next(s for s in manifest['sources'].values() if s['identity']['split'] == 'validation')
    val['identity']['group_id'] = 'group-0'
    cfg.teacher.evaluation_sha256 = publish_json(cfg.teacher.evaluation_file, manifest)
    with pytest.raises(ContractError, match='disjoint'):
        run.run_queue(cfg)
    assert not (Path(cfg.output_dir) / 'ledger.json').exists()


def test_stage_budget_does_not_block_independent_next_stage_and_failures_are_not_retried(tmp_path, monkeypatch):
    cfg = configuration(tmp_path)
    cfg.audio.enabled = True
    cfg.stress.enabled = True
    calls = []
    def prepare(*_args):
        publish_json(Path(cfg.output_dir) / 'freeze.json', dict(test=True))
        return dict(inputs={'training_plan_sha256': 'a'*64})
    monkeypatch.setattr(run, 'preflight', prepare)
    monkeypatch.setattr(VacationConfig, 'validate', lambda _: None)
    monkeypatch.setattr(run, 'run_audio', lambda *_: (calls.append('audio') or dict(status='paused', reason='stage_time_limit')))
    def fail(*_args):
        calls.append('teacher')
        raise ContractError('numerical failure')
    monkeypatch.setattr(run, 'run_teacher', fail)
    monkeypatch.setattr(run, 'read_json', lambda *_: {'cases': []})
    monkeypatch.setattr(run, 'validate_cases', lambda value: value)
    monkeypatch.setattr(run, 'run_cases', lambda *_: (calls.append('stress') or dict(status='completed')))
    result = run.run_queue(replace(cfg, mode='run'))
    assert calls == ['audio', 'teacher', 'stress']
    assert result['stages']['audio']['status'] == 'budget_exhausted'
    assert result['stages']['teacher']['status'] == 'failed'
    assert result['status'] == 'finished_with_incomplete_stages'
    run.run_queue(replace(cfg, mode='resume'))
    assert calls == ['audio', 'teacher', 'stress']
