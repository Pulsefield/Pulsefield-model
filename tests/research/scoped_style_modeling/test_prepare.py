import json
from pathlib import Path

import pytest

from ensomi_model.research.scoped_style_modeling import prepare
from ensomi_model.research.scoped_style_modeling.dataset import FOUNDATION, digest
from ensomi_model.research.scoped_style_modeling.prepare import PrepareConfig, load_chart, run_preparation


def synthetic_snapshot(tmp_path, monkeypatch):
    data = b'osu file format v14\n[General]\nMode:3\n[Difficulty]\nCircleSize:4\n[HitObjects]\n64,192,100,1,0,0:0:0:0:\n448,192,200,1,0,0:0:0:0:\n'
    sha = digest(data)
    cache = tmp_path/'sources'
    cache.mkdir()
    (cache/f'{sha}.osu').write_bytes(data)
    sources = [{'source_sha256': sha, 'byte_length': len(data), 'beatmap_id': 1, 'beatmap_set_id': 1,
                'note_count': 2, 'source_ref': {'kind': 'osu', 'uri': 'https://osu.ppy.sh/osu/1'}}]
    row = {'record_id': 'machine:1', 'source_sha256': sha, 'cell_id': 'cell-1', 'foundation_id': FOUNDATION,
           'start_ms': 100, 'end_ms': 300, 'tag_id': 'tech', 'playback_rate': 1, 'origin': 'agent-reviewed',
           'presence': 'absent', 'salience': None, 'details': {'review_context': {'start_ms': 0, 'end_ms': 400},
           'evidence': {'note_refs': [{'source_line': 7, 'column': 0, 'kind': 'normal', 'start_ms': 100, 'end_ms': 100}]}}}
    manifest = {'provenance': {}, 'foundations': {FOUNDATION: {'calibration_source_sha256': []}}}
    tables = {'machine': [row], 'human': []}
    monkeypatch.setattr(prepare, 'load_snapshot', lambda *args, **kwargs: (manifest, sources, tables))
    cfg = PrepareConfig(dataset_dir=str(tmp_path/'snapshot'), source_cache=str(cache), output_dir=str(tmp_path/'out'), split_manifest_path=str(tmp_path/'split.json'))
    return cfg, row, sources


def test_end_to_end_same_cohort_frozen_split_target_separation(tmp_path, monkeypatch):
    cfg, row, _ = synthetic_snapshot(tmp_path, monkeypatch)
    result = run_preparation(cfg)
    assert result['data_contract_ready']
    assert result['assessment_cells'] == {'machine': 1}
    assert len(result['support']) == 90
    first = json.loads((Path(cfg.output_dir)/'assessment-cohort.jsonl').read_text())
    chart, edges = load_chart(Path(cfg.output_dir)/'contexts'/f"{first['chart_key']}.json.gz", expected_sha256=first['chart_sha256'])
    assert first['assessment'] == 'absent' and first['evidence_masks'] == [0, 1, 0]
    row['presence'], row['salience'] = 'present', 'prominent'
    row['details']['evidence'] = None
    row['details']['rationale'] = 'Ignore this target-bearing text'
    row['details']['evidence_review'] = {'rationale_reviewed': True}
    cfg.output_dir = str(tmp_path/'second')
    second_result = run_preparation(cfg)
    second = json.loads((Path(cfg.output_dir)/'assessment-cohort.jsonl').read_text())
    assert second_result['split_sha256'] == result['split_sha256']
    assert second_result['assessment_cells'] == result['assessment_cells']
    assert second['chart_key'] == first['chart_key'] and second['chart_sha256'] == first['chart_sha256']
    assert second['evidence_status'] == 'missing' and second['evidence_masks'] is None
    assert load_chart(Path(cfg.output_dir)/'contexts'/f"{second['chart_key']}.json.gz", expected_sha256=second['chart_sha256']) == (chart, edges)
    with pytest.raises(FileExistsError):
        run_preparation(cfg)


def test_bad_evidence_keeps_assessment_but_blocks_ready(tmp_path, monkeypatch):
    cfg, row, _ = synthetic_snapshot(tmp_path, monkeypatch)
    row['details']['evidence']['note_refs'][0]['source_line'] = 999
    result = run_preparation(cfg)
    assert not result['data_contract_ready']
    assert result['invalid_evidence_cells'] == 1
    assert result['assessment_cells'] == {'machine': 1}
    assert result['excluded_assessment_cells'] == 0
    item = json.loads((Path(cfg.output_dir)/'assessment-cohort.jsonl').read_text())
    assert item['evidence_status'] == 'invalid'
    assert len(json.loads((Path(cfg.output_dir)/'evidence-errors.json').read_text())) == 1


def test_missing_source_is_reported_for_same_cohort_not_evidence_mask(tmp_path, monkeypatch):
    cfg, row, _ = synthetic_snapshot(tmp_path, monkeypatch)
    (Path(cfg.source_cache)/f"{row['source_sha256']}.osu").unlink()
    result = run_preparation(cfg)
    assert not result['data_contract_ready']
    assert result['excluded_assessment_cells'] == 1
    assert result['invalid_evidence_cells'] == 0
    assert result['source_errors'][0]['status'] == 'failed'
    assert result['split_sha256']


def test_noncanonical_source_rows_block_ready_with_original_identity(tmp_path, monkeypatch):
    cfg, row, sources = synthetic_snapshot(tmp_path, monkeypatch)
    path = Path(cfg.source_cache)/f"{row['source_sha256']}.osu"
    data = path.read_bytes().replace(b'64,192,100,1,0,0:0:0:0:', b'64,192,100,128,0,200:0:0:0:0:').replace(b'448,192,200', b'64,192,200')
    path.unlink()
    sha = digest(data)
    (Path(cfg.source_cache)/f'{sha}.osu').write_bytes(data)
    row['source_sha256'] = sha
    row['details']['evidence'] = None
    sources[0]['source_sha256'], sources[0]['byte_length'] = sha, len(data)
    result = run_preparation(cfg)
    assert not result['data_contract_ready']
    assert result['row_incompatible_sources'] == 1
    assert result['replayed_source_count'] == 1
    assert result['assessment_cells'] == {'machine': 1}
    report = json.loads((Path(cfg.output_dir)/'row-incompatibilities.json').read_text())
    assert report[0]['source_line_pairs'] == [[7, 8]]
