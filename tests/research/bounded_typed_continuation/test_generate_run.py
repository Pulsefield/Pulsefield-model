from dataclasses import asdict, replace
from importlib.resources import files
import hashlib
import json

import pytest
import torch

from ensomi_model.research.bounded_typed_continuation import generate_run
from ensomi_model.research.bounded_typed_continuation.condition import (
    GenerationCondition, condition_from_source, write_source_condition,
)
from ensomi_model.research.bounded_typed_continuation.condition_hydra import compose_config as prepare_config
from ensomi_model.research.bounded_typed_continuation.contract import Arm, Timing
from ensomi_model.research.bounded_typed_continuation.generate_config import GenerateConfig
from ensomi_model.research.bounded_typed_continuation.generate_hydra import compose_config
from ensomi_model.research.bounded_typed_continuation.generation import Rollout
from ensomi_model.research.bounded_typed_continuation.smoke_config import SmokeResources
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.oracle_time_continuation.storage import file_digest
from ensomi_model.research.scoped_style_modeling.dataset import ContractError
from .test_generation import setup


def inputs(tmp_path, arm=Arm.R1, device='cpu', availability='none', seed_context='none', long_memory='none', head_routing='none', release_routing='none'):
    model, timing, seed, _ = setup(arm, device, availability, seed_context=seed_context, long_memory=long_memory,
                                  head_routing=head_routing, release_routing=release_routing)
    checkpoint = tmp_path / 'trained.pt'
    torch.save(dict(format='bounded-typed/corpus-training-v1', source_revision='a' * 40,
                    config=dict(model=json.loads(json.dumps(asdict(model.config)))),
                    model={k: v.cpu() for k, v in model.state_dict().items()}), checkpoint)
    condition = GenerationCondition(arm, timing, tuple(seed),
                                    (None,) * 4 if arm == Arm.R0 else (10, 17, None, None))
    path = tmp_path / 'condition.json'
    path.write_text(json.dumps(condition.payload()) + '\n')
    resources = SmokeResources(min_available_bytes=1024 ** 2, disk_reserve_bytes=1024 ** 2,
                               output_max_bytes=8 * 1024 ** 2)
    config = GenerateConfig(checkpoint_file=str(checkpoint), checkpoint_sha256=file_digest(checkpoint),
        condition_file=str(path), condition_sha256=file_digest(path), output_dir=str(tmp_path / 'whole'),
        device=device, seed=91, candidate_budget=5, checkpoint_every_candidates=7,
        resources=resources, score_endpoints=availability == 'commitment')
    return model, condition, config


@pytest.mark.parametrize('arm,availability,seed_context,long_memory,head_routing,release_routing',
    [(a, 'none', 'none', 'none', 'none', 'none') for a in Arm] +
    [(Arm.O1, 'commitment', 'none', 'none', 'none', 'none'), (Arm.R1, 'none', 'zero', 'none', 'none', 'none'),
     (Arm.R1, 'none', 'observed', 'none', 'none', 'none'), (Arm.R1, 'none', 'observed', 'landmarks', 'none', 'none'),
     (Arm.R1, 'none', 'observed', 'landmarks', 'residual', 'none'),
     (Arm.R1, 'none', 'observed', 'landmarks', 'residual', 'residual')])
@pytest.mark.parametrize('device', ['cpu', 'mps'])
def test_packaged_run_matches_native_and_exact_resume_without_modifying_parent(tmp_path, monkeypatch, arm, availability,
                                                                              seed_context, long_memory, head_routing, release_routing, device):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    monkeypatch.setattr(generate_run, 'source_revision', lambda: 'b' * 40)
    model, condition, config = inputs(tmp_path, arm, device, availability, seed_context, long_memory, head_routing, release_routing)
    before_threads = torch.get_num_threads()
    torch.set_num_threads(config.cpu_threads)
    try:
        native = Rollout.from_seed(model, condition.timing, condition.seed_rows, condition.crossing)
        rng = torch.Generator().manual_seed(config.seed)
        expected_rows = list(condition.seed_rows)
        expected_decisions = []
        while not native.state.finished:
            step = native.step(rng, candidate_budget=config.candidate_budget, score_endpoints=config.score_endpoints)
            expected_decisions.append(json.loads(json.dumps(asdict(step))))
            if step.row is not None:
                expected_rows.append(step.row)
    finally:
        torch.set_num_threads(before_threads)
    whole = generate_run.run_generation(config, resolved_yaml='verified test configuration\n')
    assert whole['status'] == 'completed' and whole['reparse_pass']
    assert torch.get_num_threads() == before_threads
    whole_rows = [json.loads(s) for s in (tmp_path / 'whole/rows.jsonl').read_text().splitlines()]
    assert [CompleteRow(r['time_ms'], tuple(r['actions'])) for r in whole_rows] == expected_rows
    assert [json.loads(s) for s in (tmp_path / 'whole/decisions.jsonl').read_text().splitlines()] == expected_decisions
    assert (tmp_path / 'whole/condition.json').read_bytes() == (tmp_path / 'condition.json').read_bytes()
    first = generate_run.run_generation(replace(config, output_dir=str(tmp_path / 'paused'), stop_after_candidate=13))
    assert first['status'] == 'paused' and first['candidate_index'] == 13
    assert not (tmp_path / 'paused/generated.osu').exists()
    for name in ('rows', 'decisions'):
        with (tmp_path / f'paused/{name}.jsonl').open('ab') as stream:
            stream.write(b'non-durable interrupted tail')
    preserved = {p.name: p.read_bytes() for p in (tmp_path / 'paused').iterdir()}
    resumed = generate_run.run_generation(replace(config, output_dir=str(tmp_path / 'resumed'),
        resume_from=first['checkpoint_path'], resume_sha256=first['checkpoint_sha256']))
    assert resumed['status'] == 'completed'
    assert preserved == {p.name: p.read_bytes() for p in (tmp_path / 'paused').iterdir()}
    for name in ('rows.jsonl', 'decisions.jsonl', 'generated.osu'):
        assert (tmp_path / 'whole' / name).read_bytes() == (tmp_path / 'resumed' / name).read_bytes()
    checkpoint = torch.load(resumed['checkpoint_path'], weights_only=True)
    assert checkpoint['rollout']['replay']['row_count'] == len(expected_rows)
    assert torch.equal(checkpoint['rollout']['rng'], rng.get_state())


def test_hydra_projection_packaging_and_rejected_fields():
    config = compose_config(['seed=73', 'candidate_budget=99', 'checkpoint_every_candidates=17',
                             'stop_after_candidate=25', 'score_endpoints=true', 'max_seconds=5'])
    assert config.seed == 73 and config.candidate_budget == 99 and config.score_endpoints
    assert config.stop_after_candidate == 25 and config.checkpoint_every_candidates == 17 and config.max_seconds == 5
    assert prepare_config(['arm=o1', 'seed_notes=41']).seed_notes == 41
    for name in ('bounded_typed_generate.yaml', 'bounded_typed_condition.yaml'):
        assert files('ensomi_model.configs.hydra').joinpath(name).is_file()
    for overrides in (['+unused=1'], ['+resources.check_every_rows=1'], ['resume_from=parent.pt'],
                      ['presentation_source=map.osu'], ['candidate_budget=0'], ['candidate_budget=32769'],
                      ['cpu_threads=0'], ['stop_after_candidate=0'], ['seed=-1'], ['max_seconds=0'],
                      ['condition_sha256=wrong'], ['device=auto']):
        with pytest.raises((ValueError, ContractError)):
            compose_config(overrides)
    for overrides in (['+unused=1'], ['arm=other'], ['seed_notes=0'], ['source_sha256=bad']):
        with pytest.raises((ValueError, ContractError)):
            prepare_config(overrides)


def test_resume_binds_expired_persistent_seed_objects_to_external_condition(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_run, 'source_revision', lambda: 'b' * 40)
    _, _, config = inputs(tmp_path, seed_context='observed')
    paused = generate_run.run_generation(replace(config, stop_after_candidate=25))
    payload = torch.load(paused['checkpoint_path'], weights_only=True)
    # Both seed LN commitments have expired and their rows left rolling history.
    # The external condition must still own the persistent learned input.
    payload['rollout']['seed_history'][0]['new_end_times'] = (1100., 1700., None, None)
    changed = tmp_path / 'whole/tampered.pt'
    torch.save(payload, changed)
    with pytest.raises(ContractError, match='Persistent seed differs'):
        generate_run.run_generation(replace(config, output_dir=str(tmp_path / 'rejected'),
            resume_from=str(changed), resume_sha256=file_digest(changed)))


def source_bytes(later_lane=2):
    return (f'osu file format v14\n[General]\nMode:3\n[Difficulty]\nCircleSize:4\n[HitObjects]\n'
            f'64,192,0,128,0,500:0:0:0:0:\n192,192,100,1,0,0:0:0:0:\n'
            f'{64+128*later_lane},192,200,1,0,0:0:0:0:\n448,192,600,1,0,0:0:0:0:\n').encode()


@pytest.mark.parametrize('arm', list(Arm))
def test_preparation_keeps_crossing_seed_objects_but_not_suffix_lane_assignments(arm, tmp_path):
    first, second = source_bytes(2), source_bytes(1)
    a = condition_from_source(first, hashlib.sha256(first).hexdigest(), arm, seed_notes=2)
    b = condition_from_source(second, hashlib.sha256(second).hexdigest(), arm, seed_notes=2)
    assert a == b and len(a.seed_rows) == 2
    assert a.crossing == ({} if arm == Arm.R0 else {0: 3})
    assert a.timing.onsets == (None if arm == Arm.R0 else (True, True, True, False, True))
    payload = json.loads(json.dumps(a.payload()))
    assert GenerationCondition.from_payload(payload) == a
    payload['suffix_actions'] = []
    with pytest.raises(ContractError, match='exactly'):
        GenerationCondition.from_payload(payload)
    source = tmp_path / 'source.osu'
    source.write_bytes(first)
    result = write_source_condition(source, file_digest(source), tmp_path / 'condition.json', arm=arm, seed_notes=2)
    assert result['condition_sha256'] == file_digest(tmp_path / 'condition.json')
    with pytest.raises(FileExistsError):
        write_source_condition(source, file_digest(source), tmp_path / 'condition.json', arm=arm, seed_notes=2)
    with pytest.raises(ContractError, match='requested'):
        condition_from_source(first, file_digest(source), arm, seed_notes=30)


def test_interrupted_step_recovers_only_durable_journals(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_run, 'source_revision', lambda: 'c' * 40)
    _, _, config = inputs(tmp_path)
    generate_run.run_generation(config)
    original = generate_run.write_record

    def interrupt(stream, value, resources):
        original(stream, value, resources)
        if stream.name.endswith('decisions.jsonl') and value['candidate_index'] == 9:
            raise OSError('simulated interruption after decision append')

    monkeypatch.setattr(generate_run, 'write_record', interrupt)
    with pytest.raises(OSError, match='interruption'):
        generate_run.run_generation(replace(config, output_dir=str(tmp_path / 'failed')))
    monkeypatch.setattr(generate_run, 'write_record', original)
    failure = json.loads((tmp_path / 'failed/failure.json').read_text())
    assert failure['durable_candidate_index'] == 7
    parent = tmp_path / 'failed/checkpoint.pt'
    resumed = generate_run.run_generation(replace(config, output_dir=str(tmp_path / 'recovered'),
                                                   resume_from=str(parent), resume_sha256=file_digest(parent)))
    assert resumed['status'] == 'completed'
    for name in ('rows.jsonl', 'decisions.jsonl', 'generated.osu'):
        assert (tmp_path / 'whole' / name).read_bytes() == (tmp_path / 'recovered' / name).read_bytes()


def test_resume_rejects_changed_inputs_corrupt_logs_and_snapshot_clocks(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_run, 'source_revision', lambda: 'd' * 40)
    _, _, config = inputs(tmp_path)
    first = generate_run.run_generation(replace(config, stop_after_candidate=13))
    resume = replace(config, output_dir=str(tmp_path / 'resumed'), resume_from=first['checkpoint_path'],
                     resume_sha256=first['checkpoint_sha256'])
    with pytest.raises(ContractError, match='identities'):
        generate_run.run_generation(replace(resume, seed=config.seed + 1))
    assert not (tmp_path / 'resumed').exists()
    payload = torch.load(first['checkpoint_path'], weights_only=True)
    payload['rollout']['replay']['note_count'] += 1
    altered = tmp_path / 'whole/altered.pt'
    torch.save(payload, altered)
    with pytest.raises(ContractError, match='snapshot disagree'):
        generate_run.run_generation(replace(resume, resume_from=str(altered), resume_sha256=file_digest(altered)))
    rows = tmp_path / 'whole/rows.jsonl'
    original = rows.read_bytes()
    rows.write_bytes(b'X' + original[1:])
    with pytest.raises(ContractError, match='digest'):
        generate_run.run_generation(resume)
    assert not (tmp_path / 'resumed').exists()


def test_time_pause_unused_terminal_and_single_resource_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_run, 'source_revision', lambda: 'e' * 40)
    _, _, config = inputs(tmp_path, Arm.O1)
    condition = GenerationCondition(Arm.O1, Timing((0., 50., 100.), (True, False, False)),
                                    (CompleteRow(0., (1, 0, 0, 0)),), (None,) * 4)
    path = tmp_path / 'empty-tail.json'
    path.write_text(json.dumps(condition.payload()))
    config = replace(config, condition_file=str(path), condition_sha256=file_digest(path), max_seconds=1e-9)
    first = generate_run.run_generation(config)
    assert first['status'] == 'paused' and first['pause_reason'] == 'time_limit'
    assert first['candidate_index'] == 1
    guards = []
    original = generate_run.ResourceGuard

    class Guard(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            guards.append(self)

    monkeypatch.setattr(generate_run, 'ResourceGuard', Guard)
    result = generate_run.run_generation(replace(config, output_dir=str(tmp_path / 'resumed'), max_seconds=10,
        resume_from=first['checkpoint_path'], resume_sha256=first['checkpoint_sha256']))
    assert result['status'] == 'completed' and result['mechanical']['skipped_candidates'] == 2
    assert result['rows'] == 1 and len(guards) == 1
