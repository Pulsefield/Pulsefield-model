import json
from pathlib import Path

import pytest

from ensomi_model.research.oracle_time_continuation.corpus import catalog_entries, read_split
from ensomi_model.research.oracle_time_continuation.data import admit_source
from ensomi_model.research.oracle_time_continuation.storage import file_digest
from ensomi_model.research.oracle_time_continuation.train_hydra import compose_config
from ensomi_model.research.oracle_time_continuation.train_run import load_training_sources, run_training
from ensomi_model.research.scoped_style_modeling.dataset import ContractError
from .test_train_hydra import write_inputs


def fixture(tmp_path):
    sha, manifest_path, manifest = write_inputs(tmp_path)
    path = tmp_path / 'sources' / (sha + '.osu')
    source = admit_source(path.read_bytes(), sha, group_id='song:test', split='train')
    entries = [dict(source_sha256=sha, arrangement_sha256=source.identity.arrangement_sha256,
                    path=str(path), group_id='song:test', split='train', event_count=len(source.targets))]
    catalog = tmp_path / 'catalog.json'
    catalog.write_text(json.dumps(entries))
    config = compose_config([f'catalog_path={catalog}', f'catalog_sha256={file_digest(catalog)}',
                             f'split_manifest={manifest_path}', f'split_sha256={manifest["sha256"]}',
                             f'cache_dir={tmp_path / "cache"}', f'output_dir={tmp_path / "run"}',
                             'model.hidden=8', 'model.heads=2', 'model.recent=4',
                             'model.coarse_group=2', 'model.coarse_capacity=2', 'model.coupling_rank=4'])
    return config, entries


def test_pinned_catalog_reaches_runner_and_preserves_source_identity(tmp_path):
    cfg, entries = fixture(tmp_path)
    assert cfg.source_sha256 == []
    sources = load_training_sources(cfg)
    assert sources[0].row_count == entries[0]['event_count']
    assert sources[0].identity.arrangement_sha256 == entries[0]['arrangement_sha256']
    assert run_training(cfg, resolved_yaml='catalog-test')['updates'] == 1


@pytest.mark.parametrize('fault', ['sha', 'heldout', 'duplicate', 'arrangement', 'groups', 'row-count'])
def test_catalog_rejects_identity_split_and_duplicate_faults(tmp_path, fault):
    cfg, entries = fixture(tmp_path)
    assignments = read_split(cfg.split_manifest, cfg.split_sha256)
    if fault == 'sha':
        cfg.catalog_sha256 = '0' * 64
    elif fault == 'heldout':
        entries[0]['split'] = 'validation'
    elif fault == 'duplicate':
        entries.append(dict(entries[0]))
    elif fault == 'arrangement':
        entries[0]['arrangement_sha256'] = '0' * 64
    elif fault == 'groups':
        entries.append({**entries[0], 'source_sha256': '1' * 64, 'arrangement_sha256': '2' * 64,
                        'split': 'validation'})
    elif fault == 'row-count':
        entries[0]['event_count'] += 1
    path = Path(cfg.catalog_path)
    path.write_text(json.dumps(entries))
    if fault != 'sha':
        cfg.catalog_sha256 = file_digest(path)
    with pytest.raises(ContractError):
        if fault in ('arrangement', 'row-count'):
            load_training_sources(cfg)
        else:
            catalog_entries(path, cfg.catalog_sha256, assignments)


def test_catalog_selection_cannot_read_heldout_payload(tmp_path, monkeypatch):
    cfg, entries = fixture(tmp_path)
    entries.append({**entries[0], 'source_sha256': '1' * 64, 'arrangement_sha256': '2' * 64,
                    'split': 'validation', 'group_id': 'song:heldout', 'path': 'does-not-exist.osu'})
    Path(cfg.catalog_path).write_text(json.dumps(entries))
    cfg.catalog_sha256 = file_digest(Path(cfg.catalog_path))
    assert len(load_training_sources(cfg)) == 1
    cfg.source_sha256 = ['1' * 64]
    with pytest.raises(ContractError, match='train catalog'):
        load_training_sources(cfg)
