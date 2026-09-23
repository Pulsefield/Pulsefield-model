from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from ensomi_model.features.mel_base import MUSIC_MEL_CACHE_CONFIG, compute_log_mel_10ms
from ensomi_model.research.bounded_typed_continuation.contract import Arm
from ensomi_model.research.bounded_typed_continuation.data import SourceChart
from ensomi_model.research.joint_audio_continuation import data
from ensomi_model.research.oracle_time_continuation.data import SourceIdentity
from ensomi_model.research.oracle_time_continuation.storage import ROW_DTYPE, SOURCE_FORMAT


def chart(actions, times=None, duration_ms=10000):
    rows = np.zeros(len(actions), dtype=ROW_DTYPE)
    rows['time'] = np.arange(len(actions)) * 100 if times is None else times
    rows['actions'] = actions
    identity = SourceIdentity('a' * 64, 'b' * 64, 'group', 'train')
    source = SourceChart(identity, rows, minimum_seed_notes=1)
    return data.JointChart(asdict(identity), source, np.zeros((1000, 128), np.float32), duration_ms)


def test_canonical_frontend_identity_and_right_padded_clock():
    identity = data.frontend_identity()
    assert identity['config_hash'] == MUSIC_MEL_CACHE_CONFIG.mel_config_hash
    assert identity['config']['sample_rate'] == 24000
    assert identity['config']['mel_bins'] == 128
    assert (identity['frame_hop_ms'], identity['frame_support_ms'], identity['frame_center_origin_ms']) == (10, 40, 20)
    samples = np.zeros(241, np.float32)
    actual = compute_log_mel_10ms(samples, sample_rate=24000, config=MUSIC_MEL_CACHE_CONFIG)
    assert actual.shape == (2, 128)
    assert np.isfinite(actual).all()


def test_bos_and_exact_zero_ms_event_target():
    source = chart([(1, 0, 0, 0), (0, 1, 0, 0)])
    q = data.query(source, -1)
    assert q.source_index == 0 and q.replay.row_count == 0
    assert q.raw.shape[0] == 0 and len(q.valid) == 0
    assert q.target_time_ms == 0 and q.target_actions == (1, 0, 0, 0)
    assert not q.censored and not q.replay.is_complete


def test_empty_interval_retains_exact_held_state_without_manufacturing_release():
    source = chart([(2, 0, 0, 0), (0, 1, 0, 0), (3, 0, 0, 0), (1, 0, 0, 0)],
                   [0, 100, 8000, 9000])
    q = data.query(source, 2000, history_limit=1, horizon_ms=1000)
    assert q.censored and q.target_actions is None and q.target_time_ms is None
    assert q.horizon_end_ms == 3000 and q.source_index == 2
    assert q.replay.open_ln_start_ms == (0., None, None, None)
    assert q.replay.clocks_at(2000).ln_age_ms[0] == 2000
    assert q.truncated and q.predecessor_time_ms == 0 and len(q.raw) == 1
    np.testing.assert_array_equal(q.raw, source.source.content(Arm.R0, 1, 2))
    next_q = data.query(source, 7000, horizon_ms=1000)
    assert next_q.target_time_ms == 8000 and next_q.target_actions == (3, 0, 0, 0)


def test_future_ln_endpoint_changes_do_not_change_predictor_inputs():
    a = chart([(2, 2, 0, 0), (0, 0, 1, 0), (3, 0, 0, 0), (0, 3, 0, 0), (1, 0, 0, 0)])
    b = chart([(2, 2, 0, 0), (0, 0, 1, 0), (0, 3, 0, 0), (3, 0, 0, 0), (1, 0, 0, 0)])
    aq, bq = data.query(a, 100), data.query(b, 100)
    assert aq.replay == bq.replay
    np.testing.assert_array_equal(aq.raw, bq.raw)
    assert aq.target_actions != bq.target_actions
    assert not hasattr(aq, 'timing') and not hasattr(aq, 'known_ends')


def test_exhausted_source_is_censored_until_real_audio_end_without_completion_leak():
    source = chart([(1, 0, 0, 0), (0, 1, 0, 0)])
    assert source.source.state(Arm.R0, 2).replay.is_complete
    for cursor in (100, 5000, 10000):
        q = data.query(source, cursor)
        assert q.censored and not q.replay.is_complete
        assert q.horizon_end_ms == min(cursor + 4000, source.duration_ms)
        assert q.replay.row_count == 2


def fixture_corpus(tmp_path):
    source_root, cache_root = tmp_path / 'sources', tmp_path / 'cache'
    source_root.mkdir()
    catalog = []

    def add(name, group, split, audio_name, last_ms=310):
        audio_file = source_root / audio_name
        if not audio_file.exists():
            with wave.open(str(audio_file), 'wb') as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(24000)
                phase = 0 if audio_name == 'train.wav' else 1
                samples = (np.sin(np.arange(24000) / (10 + phase)) * 10000).astype('<i2')
                handle.writeframes(samples.tobytes())
        path = source_root / f'{name}.osu'
        path.write_text(f'osu file format v14\n[General]\nAudioFilename: {audio_name}\n[Metadata]\nVersion: {name}\n[HitObjects]\n')
        sha = data.digest(path)
        rows = np.zeros(32, ROW_DTYPE)
        rows['time'] = np.rint(np.linspace(0, last_ms, 32))
        rows['actions'][:, 0] = 1
        directory = cache_root / sha
        directory.mkdir(parents=True)
        (directory / 'rows.bin').write_bytes(rows.tobytes())
        identity = dict(source_sha256=sha, arrangement_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),
                        group_id=group, split=split)
        metadata = dict(format=SOURCE_FORMAT, identity=identity, row_count=len(rows),
            rows_sha256=data.digest(directory / 'rows.bin'), seed=dict(seed_row_count=30, ineligible_reason=None))
        (directory / 'metadata.json').write_text(json.dumps(metadata))
        entry = dict(identity, path=path.name, event_count=32)
        catalog.append(entry)
        return entry

    train = add('train', 'train-group', 'train', 'train.wav')
    validation = add('validation', 'validation-group', 'validation', 'validation.wav')
    alternative = add('alternative', 'train-group', 'train', 'train.wav')
    add('different-audio', 'train-group', 'train', 'other.wav')
    # A tempting group match in TEST must never open its nonexistent payload.
    catalog.append(dict(source_sha256='0' * 64, arrangement_sha256='1' * 64,
                        group_id='train-group', split='test', path='missing.osu'))
    base = dict(charts=[data._source_entry(e, source_root, cache_root) for e in (train, validation)])
    return source_root, cache_root, catalog, base, alternative


def test_alternative_selection_retains_exact_audio_group_and_validation(tmp_path):
    source_root, cache_root, catalog, base, alternative = fixture_corpus(tmp_path)
    selected, exclusions = data.select_entries(base, catalog, source_root=source_root, cache_root=cache_root)
    assert [entry['source_sha256'] for entry in selected if entry['split'] == 'validation'] == [base['charts'][1]['source_sha256']]
    assert alternative['source_sha256'] in {entry['source_sha256'] for entry in selected}
    assert len(selected) == 3 and len(exclusions) == 1
    assert 'audio bytes differ' in exclusions[0]['reason']
    assert len({entry['audio_sha256'] for entry in selected if entry['split'] == 'train'}) == 1
    corrupted = dict(base, charts=[dict(base['charts'][0], split='validation'), base['charts'][1]])
    with pytest.raises(ValueError, match='identity'):
        data.select_entries(corrupted, catalog, source_root=source_root, cache_root=cache_root)


def test_preparation_load_and_unique_train_only_normalization(tmp_path, monkeypatch):
    source_root, cache_root, catalog, base, _ = fixture_corpus(tmp_path)
    catalog_file = tmp_path / 'catalog.json'
    catalog_file.write_text(json.dumps(catalog))
    base['config'] = dict(catalog_sha256=data.digest(catalog_file))
    base_file = tmp_path / 'base.json'
    base_file.write_text(json.dumps(base))
    monkeypatch.setattr(data.subprocess, 'check_output', lambda args, **kwargs: '' if 'status' in args else 'revision')
    config = SimpleNamespace(root=str(tmp_path / 'output'), base_manifest_file=str(base_file),
        catalog_file=str(catalog_file), catalog_sha256=data.digest(catalog_file), catalog_root=str(source_root),
        source_cache_dir=str(cache_root), max_train_alternatives=2, max_seconds=60.)
    receipt = data.prepare(config)
    assert receipt['counts'] == {'train': 2, 'validation': 1}
    charts, normalization = data.load_corpus(config.root)
    groups = data.train_groups(charts)
    assert len(groups) == 1 and len(groups[0]) == 2
    train_mel = groups[0][0].mel
    assert normalization['frame_count'] == len(train_mel) == 100
    np.testing.assert_allclose(normalization['mean'], train_mel.astype(np.float64).mean(0))
    np.testing.assert_allclose(normalization['std'], train_mel.astype(np.float64).std(0), atol=1e-6)
    assert all(chart.duration_ms == 1000 for chart in charts)
    assert len(data.smoke_charts(charts)) == 1
    # Loading must reject a modified feature even when its filename is unchanged.
    manifest = json.loads((tmp_path / 'output' / 'manifest.json').read_text())
    feature = next(iter(manifest['assets'].values()))['mel_file']
    with open(feature, 'ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='Mel bytes'):
        data.load_corpus(config.root)


def test_preparation_rejects_rows_past_audio_without_changing_duration(tmp_path, monkeypatch):
    source_root, cache_root, catalog, base, _ = fixture_corpus(tmp_path)
    catalog_file = tmp_path / 'catalog.json'
    catalog_file.write_text(json.dumps(catalog))
    base['config'] = dict(catalog_sha256=data.digest(catalog_file))
    base_file = tmp_path / 'base.json'
    base_file.write_text(json.dumps(base))
    monkeypatch.setattr(data.subprocess, 'check_output', lambda args, **kwargs: '' if 'status' in args else 'revision')
    monkeypatch.setattr(data, 'load_audio_file', lambda *args: np.zeros(2400, np.float32))
    config = SimpleNamespace(root=str(tmp_path / 'output'), base_manifest_file=str(base_file),
        catalog_file=str(catalog_file), catalog_sha256=data.digest(catalog_file), catalog_root=str(source_root),
        source_cache_dir=str(cache_root), max_train_alternatives=0, max_seconds=60.)
    with pytest.raises(ValueError, match='true audio end'):
        data.prepare(config)
    assert not (tmp_path / 'output' / 'manifest.json').exists()
    exclusions = json.loads((tmp_path / 'output' / 'exclusions.json').read_text())
    assert len(exclusions) == 2 and all(item['duration_ms'] == 100 for item in exclusions)
