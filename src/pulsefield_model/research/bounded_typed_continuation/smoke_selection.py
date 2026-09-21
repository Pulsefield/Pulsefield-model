"""Deterministic TRAIN-only selection for pipeline learning, not quality ranking.

The census is an audited source inventory, not predictor input. The resulting
manifest names every source, group, onset interval and mechanical selection fact.
"""
import json
from pathlib import Path
import time

import numpy as np
import psutil

from ..oracle_time_continuation.corpus import catalog_entries, read_split
from ..oracle_time_continuation.data import SourceIdentity
from ..oracle_time_continuation.storage import file_digest
from ..scoped_style_modeling.dataset import ContractError
from .data import SourceChart, SourceInterval


def select_intervals(*, census_path, census_sha256, catalog_path, catalog_sha256,
                     split_manifest, split_sha256, source_cache_dir, output_file, seed=371):
    """Write a fresh 16-group/128-onset manifest under a ten-minute read budget.

    Four TAP, four mixed, four LN-rich, two independent-endpoint and two long-gap
    examples serve a mechanical/learning check. Each source has 640–2400 physical
    rows to bound this first full-support pointer exercise. These are deliberate
    selection biases; this manifest cannot stand in for a population evaluation.
    """
    output_file = Path(output_file)
    if output_file.exists():
        raise FileExistsError(output_file)
    started = time.monotonic()
    if file_digest(Path(census_path), 32 * 1024 ** 2) != census_sha256:
        raise ContractError('TRAIN census differs from its pinned digest')
    assignments = read_split(split_manifest, split_sha256)
    catalog = catalog_entries(catalog_path, catalog_sha256, assignments, split='train')
    by_sha = {entry['source_sha256']: entry for entry in catalog}
    census = [json.loads(line) for line in Path(census_path).read_text().splitlines()]
    if len({row['source_sha256'] for row in census}) != len(census):
        raise ContractError('TRAIN census contains duplicate source rows')
    for row in census:
        expected = by_sha.get(row['source_sha256'])
        if expected is None or expected['group_id'] != row['group_id'] or expected['split'] != 'train':
            raise ContractError('Census source allocation differs from the TRAIN catalog')
    candidates = [row for row in census if row['suffix_onsets'] >= 128 and
                  640 <= row['seed_rows'] + row['suffix_rows'] <= 2400]
    rng = np.random.default_rng(seed)
    ordered = [candidates[int(i)] for i in rng.permutation(len(candidates))]
    entries, used_groups, inspected = [], set(), 0
    for stratum, needed in (('tap', 4), ('mixed', 4), ('ln-rich', 4), ('independent-ends', 2), ('long-gap', 2)):
        selected = 0
        for record in ordered:
            if (time.monotonic() - started > 600 or psutil.Process().memory_info().rss > 3 * 1024 ** 3 or
                    psutil.virtual_memory().available < 2 * 1024 ** 3):
                raise ContractError('Learning-check selection exceeded its time or memory bound')
            fraction = record['suffix_ln_fraction']
            if record['group_id'] in used_groups:
                continue
            if ((stratum == 'tap' and fraction != 0) or
                    (stratum == 'mixed' and not .02 <= fraction < .4) or
                    (stratum == 'ln-rich' and fraction < .4) or
                    (stratum == 'independent-ends' and fraction == 0)):
                continue
            admitted = by_sha[record['source_sha256']]
            identity = SourceIdentity(**{key: admitted[key] for key in ('source_sha256', 'arrangement_sha256', 'group_id', 'split')})
            chart = SourceChart.from_cache(Path(source_cache_dir) / identity.source_sha256, identity)
            inspected += 1
            if len(chart.onsets) != record['suffix_onsets'] or chart.seed_rows != record['seed_rows']:
                raise ContractError('Source/census onset or seed counts disagree')
            endpoint_min = np.where(chart.endpoints >= 0, chart.endpoints, len(chart.rows)).min(-1)
            independent = np.flatnonzero(((chart.endpoints >= 0).sum(-1) >= 2) &
                                         (endpoint_min != chart.endpoints.max(-1)) &
                                         (np.arange(len(chart.rows)) >= chart.seed_rows))
            gaps = np.diff(chart.rows['time'])
            long_gaps = np.flatnonzero((gaps >= 2000.) & (np.arange(len(gaps)) >= chart.seed_rows))
            if stratum == 'independent-ends':
                if not len(independent):
                    continue
                middle = int(np.searchsorted(chart.onsets, independent[len(independent) // 2]))
                first = max(0, min(len(chart.onsets) - 128, middle - 64))
            elif stratum == 'long-gap':
                if not len(long_gaps):
                    continue
                gap = int(long_gaps[np.argmax(gaps[long_gaps])])
                middle = int(np.searchsorted(chart.onsets, gap + 1))
                first = max(0, min(len(chart.onsets) - 128, middle - 64))
            else:
                first = 0 if selected % 2 == 0 else min(len(chart.onsets) - 128, int(np.searchsorted(chart.onsets, 600)))
            interval = SourceInterval(chart, first, 128)
            selected_actions = chart.rows['actions'][interval.start:interval.stop]
            ln_heads = int((selected_actions == 2).sum())
            if stratum in ('mixed', 'ln-rich') and ln_heads < 4:
                continue
            if interval.stop - interval.start > 1024:
                continue
            independent_count = int(((independent >= interval.start) & (independent < interval.stop)).sum())
            selected_gaps = gaps[max(chart.seed_rows, interval.start - 1):interval.stop - 1]
            if stratum == 'independent-ends' and not independent_count:
                continue
            if stratum == 'long-gap' and not np.any(selected_gaps >= 2000.):
                continue
            entries.append(dict(identity=identity.__dict__, first_onset=first, onset_count=128, stratum=stratum,
                                facts=dict(source_rows=len(chart.rows), target_start=interval.start, target_stop=interval.stop,
                                           heads=int(((selected_actions == 1) | (selected_actions == 2)).sum()), ln_heads=ln_heads,
                                           independent_endpoint_rows=independent_count,
                                           largest_gap_ms=float(selected_gaps.max(initial=0.)),
                                           seed_crossing_lns=int((chart.open_start[chart.seed_rows - 1] >= 0).sum()),
                                           crop_crossing_lns=int((chart.open_start[interval.start - 1] >= 0).sum()),
                                           full_512_row_context=interval.start >= 512)))
            used_groups.add(identity.group_id)
            selected += 1
            if selected == needed:
                break
        if selected != needed:
            raise ContractError(f'Could not fill learning-check stratum {stratum}: {selected}/{needed}')
    manifest = dict(format='bounded-typed/source-intervals-v1', seed=seed, catalog_sha256=catalog_sha256,
                    split_sha256=split_sha256, census_sha256=census_sha256, inspected_sources=inspected,
                    entries=entries)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open('x') as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    return dict(path=str(output_file), sha256=file_digest(output_file), groups=len(used_groups),
                inspected_sources=inspected, selection_seconds=time.monotonic() - started,
                facts=[entry['facts'] for entry in entries])
