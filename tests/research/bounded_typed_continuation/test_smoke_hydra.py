from dataclasses import asdict
import hashlib
from importlib.resources import files
import json

import numpy as np
import pytest
import torch

from pulsefield_model.research.bounded_typed_continuation.contract import Arm
from pulsefield_model.research.bounded_typed_continuation.data import SourceChart
from pulsefield_model.research.bounded_typed_continuation.smoke_hydra import compose_config
from pulsefield_model.research.bounded_typed_continuation import smoke_run
from pulsefield_model.research.oracle_time_continuation.data import SourceIdentity
from pulsefield_model.research.oracle_time_continuation.storage import ROW_DTYPE, SOURCE_FORMAT, file_digest
from pulsefield_model.research.scoped_style_modeling.dataset import ContractError, canonical_json


def test_packaged_composition_arm_selection_and_complete_projection():
    config = compose_config(['model.arm=R1', 'model.hidden=32', 'model.row_consequence=actions', 'model.seed_context=observed',
                             'candidate_budget=321', 'updates=7'])
    assert config.model.arm == Arm.R1 and config.model.hidden == 32
    assert config.candidate_budget == 321 and config.updates == 7
    assert config.model.row_consequence == 'actions'
    assert config.model.seed_context == 'observed'
    assert files('pulsefield_model.configs.hydra').joinpath('bounded_typed_smoke.yaml').is_file()
    for overrides in (['+unused=1'], ['+model.unused=1'], ['+resources.unused=1'], ['learning_rate=0'],
                      ['candidate_budget=0'], ['model.levels=9'], ['batch_size=5']):
        with pytest.raises((ContractError, ValueError)):
            compose_config(overrides)


def inputs(tmp_path):
    cache = tmp_path / 'cache'
    cache.mkdir()
    catalog, entries, assignments = [], [], {}
    for i in range(2):
        identity = SourceIdentity(('a' if i == 0 else 'c') * 64, ('b' if i == 0 else 'd') * 64, f'group-{i}', 'train')
        rows = np.zeros(45, dtype=ROW_DTYPE)
        rows['time'] = np.arange(45) * 100.
        rows['actions'][:, 0] = 1
        rows['actions'][30] = (2, 0, 0, 0)
        rows['actions'][31] = (0, 1, 0, 0)
        rows['actions'][32] = (3, 0, 0, 0)
        source = SourceChart(identity, rows)
        directory = cache / identity.source_sha256
        directory.mkdir()
        payload = rows.tobytes()
        (directory / 'rows.bin').write_bytes(payload)
        metadata = dict(format=SOURCE_FORMAT, identity=asdict(identity), row_count=len(rows),
                        rows_sha256=hashlib.sha256(payload).hexdigest(),
                        seed=dict(seed_row_count=source.seed_rows, ineligible_reason=None))
        (directory / 'metadata.json').write_text(json.dumps(metadata))
        catalog.append(dict(**asdict(identity), event_count=len(rows), path='synthetic-source.osu'))
        entries.append(dict(identity=asdict(identity), first_onset=0, onset_count=4))
        assignments[identity.source_sha256] = dict(group_id=identity.group_id, split='train')
    split = dict(sources=assignments)
    split['sha256'] = hashlib.sha256(canonical_json(split).encode()).hexdigest()
    for name, value in (('catalog', catalog), ('split', split),
                        ('intervals', dict(format='bounded-typed/source-intervals-v1', entries=entries))):
        (tmp_path / (name + '.json')).write_text(json.dumps(value))
    return [f'interval_manifest={tmp_path / "intervals.json"}', f'interval_sha256={file_digest(tmp_path / "intervals.json")}',
            f'catalog_path={tmp_path / "catalog.json"}', f'catalog_sha256={file_digest(tmp_path / "catalog.json")}',
            f'split_manifest={tmp_path / "split.json"}', f'split_sha256={split["sha256"]}', f'source_cache_dir={cache}']


def test_real_runner_consumes_projected_fields_and_saves_safe_recovery_state(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke_run, 'source_revision', lambda: 'e' * 40)
    config = compose_config(inputs(tmp_path) + [f'output_dir={tmp_path / "run"}', 'device=cpu', 'model.hidden=8',
                             'model.levels=2', 'model.coupling_rank=2', 'updates=2', 'report_every=1',
                             'candidate_budget=3', 'batch_size=2', 'learning_rate=0.003', 'warmup_updates=1',
                             'resources.min_available_bytes=1048576', 'max_seconds=60'])
    result = smoke_run.run_smoke(config, resolved_yaml='projected synthetic test\n')
    assert result['status'] == 'completed' and result['updates'] == 2
    assert result['source_onset_exposures'] == 16 and result['unique_source_onsets'] == 8
    assert result['interval_draw_counts'] == [2, 2]
    assert np.isfinite(result['final']['ln_type_nll']) and result['final']['endpoint_decisions'] == 2
    output = tmp_path / 'run'
    checkpoint = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=True)
    assert checkpoint['update'] == 2 and checkpoint['optimizer']['state']
    assert checkpoint['config']['model']['hidden'] == 8 and checkpoint['config']['candidate_budget'] == 3
    assert checkpoint['config']['learning_rate'] == .003
    records = [json.loads(line) for line in (output / 'training.jsonl').read_text().splitlines()]
    assert len(records) == 2 and records[0]['source_onsets'] == 8
    assert records[-1]['total_onset_exposures'] == 16 and records[-1]['grad_norm'] > 0
    assert not (output / 'failure.json').exists()
    with pytest.raises(FileExistsError):
        smoke_run.run_smoke(config)


def test_manifest_train_identity_and_unique_groups_are_checked(tmp_path):
    config = compose_config(inputs(tmp_path))
    path = tmp_path / 'intervals.json'
    data = json.loads(path.read_text())
    data['entries'][0]['identity']['split'] = 'validation'
    path.write_text(json.dumps(data))
    config.interval_sha256 = file_digest(path)
    with pytest.raises(ContractError, match='TRAIN'):
        smoke_run.load_intervals(config, lambda *_args, **_kwargs: None)
