from dataclasses import asdict
import hashlib
import json

import numpy as np
import pytest
import torch

from ensomi_model.research.bounded_typed_continuation.corpus import (
    ChartCache, Coverage, SamplingConfig, create_plan, draw_plan, next_batch, read_plan,
)
from ensomi_model.research.oracle_time_continuation.data import SourceIdentity
from ensomi_model.research.oracle_time_continuation.storage import ROW_DTYPE, SOURCE_FORMAT, file_digest
from ensomi_model.research.scoped_style_modeling.dataset import ContractError, canonical_json


def inputs(tmp_path):
    catalog, census, assignments = [], [], {}
    cache = tmp_path / 'cache'
    cache.mkdir()
    for i in range(3):
        identity = SourceIdentity(str(i + 1) * 64, str(i + 4) * 64, 'group-0' if i < 2 else 'group-1', 'train')
        rows = np.zeros(100 + i, dtype=ROW_DTYPE)
        rows['time'] = np.arange(len(rows)) * 100.
        rows['actions'][:, i] = 1
        path = cache / identity.source_sha256
        path.mkdir()
        (path / 'rows.bin').write_bytes(rows.tobytes())
        metadata = dict(format=SOURCE_FORMAT, identity=asdict(identity), row_count=len(rows),
                        rows_sha256=file_digest(path / 'rows.bin'), seed=dict(seed_row_count=30, ineligible_reason=None))
        (path / 'metadata.json').write_text(json.dumps(metadata))
        catalog.append(dict(**asdict(identity), event_count=len(rows), path='unused-source.osu'))
        census.append(dict(source_sha256=identity.source_sha256, group_id=identity.group_id, seed_rows=30,
                           suffix_rows=len(rows) - 30, suffix_onsets=len(rows) - 30))
        assignments[identity.source_sha256] = dict(group_id=identity.group_id, split='train')
    split = dict(sources=assignments)
    split['sha256'] = hashlib.sha256(canonical_json(split).encode()).hexdigest()
    (tmp_path / 'catalog.json').write_text(json.dumps(catalog))
    (tmp_path / 'split.json').write_text(json.dumps(split))
    (tmp_path / 'census.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in census))
    return dict(catalog_path=tmp_path / 'catalog.json', catalog_sha256=file_digest(tmp_path / 'catalog.json'),
                split_manifest=tmp_path / 'split.json', split_sha256=split['sha256'],
                census_path=tmp_path / 'census.jsonl', census_sha256=file_digest(tmp_path / 'census.jsonl'),
                source_cache_dir=cache)


def test_pinned_plan_is_deterministic_and_batches_stop_at_exact_onset_checkpoints(tmp_path):
    config = SamplingConfig(seed=7, horizons=(8, 16), milestones=(37, 103, 211))
    paths = inputs(tmp_path)
    first = create_plan(**paths, sampling=config, output_file=tmp_path / 'first.json')
    second = create_plan(**paths, sampling=config, output_file=tmp_path / 'second.json')
    assert first['sha256'] == second['sha256']
    plan = read_plan(tmp_path / 'first.json', first['sha256'])
    cursor, total, checkpoints = 0, 0, []
    while cursor < len(plan['draws']):
        batch, cursor = next_batch(plan, cursor, 4)
        total += sum(row['onset_count'] for row in batch)
        assert total == batch[-1]['exposure_end']
        assert not any(row['checkpoint'] is not None for row in batch[:-1])
        if batch[-1]['checkpoint'] is not None:
            checkpoints.append(total)
    assert checkpoints == [37, 103, 211] and total == 211
    with pytest.raises(FileExistsError):
        create_plan(**paths, sampling=config, output_file=tmp_path / 'first.json')
    plan['draws'][0]['first_onset'] += 1
    altered = tmp_path / 'altered.json'
    altered.write_text(json.dumps(plan))
    with pytest.raises(ContractError, match='deterministic'):
        read_plan(altered, file_digest(altered))


def test_group_uniformity_is_distinct_from_chart_uniformity():
    sources = {str(i): dict(identity=dict(group_id='a' if i < 2 else 'b'), onsets=100) for i in range(3)}
    draws = draw_plan(sources, SamplingConfig(seed=17, horizons=(1,), milestones=(10000,), seed_probability=.2))
    group_b = sum(row['source_sha256'] == '2' for row in draws) / len(draws)
    seed_fraction = sum(row['seed_stratum'] for row in draws) / len(draws)
    assert .47 < group_b < .53 and .18 < seed_fraction < .22
    assert all(row['first_onset'] == 0 for row in draws if row['seed_stratum'])


def test_digest_pinned_lru_evicts_without_invalidating_inflight_source(tmp_path):
    paths = inputs(tmp_path)
    result = create_plan(**paths, sampling=SamplingConfig(horizons=(8,), milestones=(40,)), output_file=tmp_path / 'plan.json')
    plan = read_plan(tmp_path / 'plan.json', result['sha256'])
    cache = ChartCache(paths['source_cache_dir'], plan['sources'], max_sources=1, max_bytes=1024 ** 2)
    shas = sorted(plan['sources'])
    first = cache.get(shas[0])
    assert cache.get(shas[0]) is first
    second = cache.get(shas[1])
    assert second is not first and first.row(30).actions == (1, 0, 0, 0)
    assert cache.metrics()['sources'] == 1 and cache.metrics()['evictions'] == 1
    assert cache.charged_bytes < cache.max_bytes
    metadata = paths['source_cache_dir'] / shas[0] / 'metadata.json'
    metadata.write_text(metadata.read_text() + ' ')
    with pytest.raises(ContractError, match='changed'):
        cache.get(shas[0])


def test_coverage_counts_union_of_actual_intervals_and_survives_safe_checkpoint(tmp_path):
    sources = {'a': dict(onsets=17, identity=dict(group_id='g')), 'b': dict(onsets=4, identity=dict(group_id='g'))}
    coverage = Coverage(sources)
    coverage.commit([dict(source_sha256='a', first_onset=0, onset_count=8),
                     dict(source_sha256='a', first_onset=4, onset_count=8),
                     dict(source_sha256='b', first_onset=3, onset_count=1)])
    assert coverage.metrics() == dict(unique_onsets=13, charts=2, groups=1)
    path = tmp_path / 'coverage.pt'
    torch.save(coverage.snapshot(), path)
    restored = Coverage(sources, torch.load(path, weights_only=True))
    assert restored.metrics() == coverage.metrics()
    restored.commit([dict(source_sha256='a', first_onset=16, onset_count=1)])
    assert restored.metrics()['unique_onsets'] == 14
    with pytest.raises(ContractError, match='outside'):
        restored.commit([dict(source_sha256='a', first_onset=17, onset_count=1)])


@pytest.mark.parametrize('kwargs', [dict(seed_probability=-.1), dict(horizons=(0,)), dict(milestones=(8, 7)), dict(seed=True)])
def test_sampling_rejects_invalid_declared_distributions(kwargs):
    with pytest.raises(ContractError):
        SamplingConfig(**kwargs)
