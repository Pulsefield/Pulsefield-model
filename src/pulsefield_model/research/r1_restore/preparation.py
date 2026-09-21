"""Source-only allocation and deterministic plans for a reconstructed lineage."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from ...osu_core.difficulty import compute_mania_star_rating_20241007, parse_osu_file
from ..bounded_typed_continuation.corpus import SamplingConfig, draw_plan, read_plan
from ..oracle_time_continuation.storage import ROW_DTYPE, file_digest
from ..scoped_style_modeling.dataset import ContractError
from ..vacation_training.control import publish_json, read_json
from .config import STAGES


def distinct_groups(rows, count=None):
    result, seen = [], set()
    for row in rows:
        if row['group_id'] not in seen:
            seen.add(row['group_id'])
            result.append(row)
            if count is not None and len(result) == count:
                break
    if count is not None and len(result) != count:
        raise ContractError('The fixed native selection has too few eligible TRAIN groups')
    return result


def hashed(rows, seed):
    return sorted(rows, key=lambda r: hashlib.sha256(f"{seed}:{r['source_sha256']}".encode()).hexdigest())


def select_sources(census):
    routing = distinct_groups(hashed([r for r in census if 1500 <= r['onsets'] <= 6000 and
                                     r['duration_ms'] >= 180000], 953), 32)
    ordinary = distinct_groups(hashed([r for r in census if 256 <= r['onsets'] <= 3000 and
                                      r['duration_ms'] >= 180000], 957), 32)
    excluded = {r['group_id'] for r in ordinary}
    transition = [r for r in census if r['group_id'] not in excluded and r['transition'] is not None and
                  r['stars'] is not None and 2 <= r['stars'] <= 6 and
                  180000 <= r['duration_ms'] <= 600000 and 256 <= r['onsets'] <= 4000]
    transition.sort(key=lambda r: (-r['transition']['ratio'], -r['transition']['count'], r['source_sha256']))
    release = distinct_groups(transition)
    if len(release) < 4:
        raise ContractError('The fixed transition cohort has too few TRAIN groups')
    random = np.random.default_rng(20260922)
    response, seen = [], set()
    for band in (4, 5):
        for rich in (False, True):
            groups = {}
            for row in census:
                if (row['group_id'] in seen or row['stars'] is None or
                        not band <= row['stars'] < band + 1 + (1e-7 if band == 5 else 0) or
                        not 180000 <= row['duration_ms'] <= 600000 or not 1000 <= row['onsets'] <= 6000 or
                        (row['ln_fraction'] >= .1) != rich):
                    continue
                groups.setdefault(row['group_id'], []).append(row)
            keys = sorted(groups)
            if len(keys) < 8:
                raise ContractError('A response stratum needs eight distinct TRAIN groups')
            for index in random.choice(len(keys), 8, replace=False):
                group = keys[int(index)]
                charts = sorted(groups[group], key=lambda r: r['source_sha256'])
                response.append(charts[int(random.integers(len(charts)))])
                seen.add(group)
    return dict(routing=routing, release=release, response=response,
                excluded_ordinary_release_groups=sorted(excluded))


def transition_witness(times):
    """Match the corrected sparse-to-dense source timing selector."""
    if len(times) < 25:
        return None
    before, after = times[12:-12] - times[:-24], times[24:] - times[12:-12]
    gaps = np.diff(times)
    maximum = np.array([gaps[i:i + 12].max() for i in range(len(times) - 24)])
    ratios = before / after
    good = np.flatnonzero((before >= 6000) & (before <= 30000) & (after <= 3000) &
                         (ratios >= 4) & (maximum <= 2500))
    if not len(good):
        return None
    best = int(good[np.argmax(ratios[good])])
    return dict(ratio=float(ratios[best]), count=len(good), time_ms=float(times[12 + best]))


def prepare_census(config, plan, root, check):
    """Read only admitted TRAIN payloads; all selection identities are frozen before training."""
    catalog = read_json(config.catalog_file, plan['catalog_sha256'], limit=32 * 1024**2)
    by_sha = {row['source_sha256']: row for row in catalog}
    if len(by_sha) != len(catalog):
        raise ContractError('Catalog contains duplicate source identities')
    census = []
    for index, (sha, pin) in enumerate(sorted(plan['sources'].items())):
        check()
        entry = by_sha.get(sha)
        if entry is None or any(entry.get(k) != v for k, v in pin['identity'].items()):
            raise ContractError('Restoration census source differs from its TRAIN allocation')
        folder = Path(config.training.source_cache_dir) / sha
        if (file_digest(folder / 'metadata.json') != pin['metadata_sha256'] or
                file_digest(folder / 'rows.bin') != pin['rows_sha256']):
            raise ContractError('TRAIN cache changed after plan preparation')
        rows = np.fromfile(folder / 'rows.bin', dtype=ROW_DTYPE)
        suffix = rows[pin['seed_rows']:]
        heads = np.isin(suffix['actions'], (1, 2))
        if int(heads.any(1).sum()) != pin['onsets']:
            raise ContractError('Census onset count differs from its plan')
        duration = float(rows[-1]['time'] - rows[0]['time'])
        source = Path(entry['path'])
        source = source if source.is_absolute() else Path(config.catalog_root) / source
        stars = None
        if 180000 <= duration <= 600000 and 256 <= pin['onsets'] <= 6000:
            if file_digest(source) != sha:
                raise ContractError('TRAIN source differs from the admitted catalog')
            parsed = parse_osu_file(source)
            if parsed.mode != 3 or parsed.circle_size != 4:
                raise ContractError('Restoration source must be native four-key mania')
            stars = compute_mania_star_rating_20241007(parsed.hit_objects, 4, 1.)
        times = rows['time'][np.isin(rows['actions'], (1, 2)).any(1)]
        witness = (transition_witness(times) if stars is not None and 2 <= stars <= 6 and pin['onsets'] <= 4000 else None)
        census.append(dict(source_sha256=sha, group_id=pin['identity']['group_id'], onsets=pin['onsets'],
            source_file=str(source.resolve()), duration_ms=duration, stars=stars,
            ln_fraction=float((suffix['actions'] == 2).sum() / heads.sum()), transition=witness))
        if (index + 1) % 500 == 0:
            print(json.dumps(dict(phase='restore-census', charts=index + 1, total=len(plan['sources']))), flush=True)
    selection = select_sources(census)
    census_sha = publish_json(root / 'census.json', dict(format='r1-restore/census-v1', sources=census))
    selection_sha = publish_json(root / 'selection.json', dict(format='r1-restore/selection-v1',
        census_sha256=census_sha, selections=selection,
        historical_identity_restored=False, scope='Reconstructed source-only rules; new parent weights imply new trajectories'))
    return dict(census_sha256=census_sha, selection_sha256=selection_sha)


def prepare_plans(config, base, root):
    points, receipts, previous = [], {}, []
    original = base['sampling']['milestones']
    if config.milestones['base'][:len(original)] != original:
        raise ContractError('The base stage must retain the supplied plan milestones')
    (root / 'plans').mkdir(exist_ok=True)
    for stage in STAGES:
        points.extend(config.milestones[stage])
        sampling = SamplingConfig(**{**base['sampling'], 'milestones': tuple(points)})
        draws = draw_plan(base['sources'], sampling)
        if draws[:len(previous)] != previous:
            raise ContractError('A stage changed the inherited draw prefix')
        plan = {**base, 'sampling': asdict(sampling), 'draws': draws}
        path = root / 'plans' / f'{stage}.json'
        digest = publish_json(path, plan)
        read_plan(path, digest)
        receipts[stage] = dict(path=str(path), sha256=digest, target_onsets=points[-1])
        previous = draws
    return receipts
