"""Rebuild pinned vacation inputs from the surviving September corpus.

Usage: uv run --offline --python 3.10 --extra mps python
       experiments/vacation_rebuild/prepare_inputs.py /absolute/fresh/asset-root

This prepares inputs only. It does not train a model or recover historical draws.
"""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq
from omegaconf import OmegaConf

from pulsefield_model.osu_core.difficulty import compute_mania_star_rating_20241007, parse_osu_file
from pulsefield_model.research.bounded_typed_continuation.condition import write_source_condition
from pulsefield_model.research.bounded_typed_continuation.contract import Arm
from pulsefield_model.research.bounded_typed_continuation.corpus import SamplingConfig, create_plan, read_plan
from pulsefield_model.research.oracle_time_continuation.corpus import catalog_entries, read_split
from pulsefield_model.research.oracle_time_continuation.storage import ROW_DTYPE, file_digest
from pulsefield_model.research.vacation_training.audio_inputs import prepare_audio_inputs
from pulsefield_model.research.vacation_training.control import publish_json
from pulsefield_model.research.vacation_training.teacher import validate_evaluation

REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / 'artifacts/oracle-time-review/20260915-adfb1ee/catalog.json'
CATALOG_SHA = 'e31b7e8f4daa044503ef2b8411bc41727ba371eec804c9462a4608be8112ad28'
SPLIT = REPO / 'artifacts/scoped-style-modeling/prepare-v1/split-manifest.json'
SPLIT_SHA = '15175f45e91cf7299a9a30166731bf38ee7361399b346fb692cf68e76de5992a'
CACHE = REPO / 'artifacts/oracle-time-continuation/full-cache-v1'
INDEX = REPO / 'artifacts/indexes/beatmap_index_4k.parquet'
SELECTION_SEED = 20260920


def ordering(value):
    return hashlib.sha256(f'{SELECTION_SEED}:{value}'.encode()).hexdigest()


def main(root):
    root = root.expanduser().resolve()
    if root.is_relative_to(REPO):
        raise ValueError('Keep the unique asset owner outside the product worktree')
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / 'inputs'
    inputs.mkdir()
    assignments = read_split(SPLIT, SPLIT_SHA)
    entries = [e for split in ('train', 'validation')
               for e in catalog_entries(CATALOG, CATALOG_SHA, assignments, split=split)]
    shutil.copyfile(CATALOG, inputs / 'catalog.json')
    shutil.copyfile(SPLIT, inputs / 'split-manifest.json')
    index = pq.read_table(INDEX, columns=['shard', 'beatmap_path', 'difficulty']).to_pylist()
    stars = {str(Path('dataset') / row['shard'] / row['beatmap_path']): row['difficulty'] for row in index}
    pins, census, candidates, excluded = {}, [], [], []
    for entry in entries:
        sha = entry['source_sha256']
        directory = CACHE / sha
        metadata = json.loads((directory / 'metadata.json').read_text())
        if metadata['identity'] != {key: entry[key] for key in ('source_sha256', 'arrangement_sha256', 'group_id', 'split')}:
            raise ValueError('Cache allocation differs from the catalog')
        if metadata['seed']['ineligible_reason'] is not None:
            excluded.append(dict(source_sha256=sha, reason=metadata['seed']['ineligible_reason']))
            continue
        rows = np.memmap(directory / 'rows.bin', dtype=ROW_DTYPE, mode='r')
        seed_rows = metadata['seed']['seed_row_count']
        actions = rows['actions'][seed_rows:]
        heads = (actions == 1) | (actions == 2)
        onsets = int(heads.any(-1).sum())
        if not onsets:
            raise ValueError('Eligible source has no post-seed onsets')
        pins[sha] = dict(identity=metadata['identity'], rows=len(rows), seed_rows=seed_rows,
                         onsets=onsets, rows_sha256=metadata['rows_sha256'],
                         metadata_sha256=file_digest(directory / 'metadata.json'))
        record = dict(source_sha256=sha, group_id=entry['group_id'], seed_rows=seed_rows,
                      suffix_rows=len(rows) - seed_rows, suffix_onsets=onsets)
        if entry['split'] == 'train':
            census.append(record)
        candidates.append(dict(**entry, onsets=onsets,
            suffix_seconds=float(rows[-1]['time'] - rows[seed_rows - 1]['time']) / 1000,
            suffix_ln_head_fraction=float((actions == 2).sum() / heads.sum()),
            index_difficulty=stars.get(entry['path'])))
        del rows
    assert len(census) == 11563 and len({r['group_id'] for r in census}) == 3169
    assert sum(r['suffix_onsets'] for r in census) == 10735674
    census_file = inputs / 'train-census.jsonl'
    census_file.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in census))
    plan_receipt = create_plan(catalog_path=inputs / 'catalog.json', catalog_sha256=CATALOG_SHA,
        split_manifest=inputs / 'split-manifest.json', split_sha256=SPLIT_SHA,
        census_path=census_file, census_sha256=file_digest(census_file), source_cache_dir=CACHE,
        output_file=inputs / 'base-plan.json', sampling=SamplingConfig())
    plan = read_plan(plan_receipt['path'], plan_receipt['sha256'])
    print(json.dumps(dict(stage='plan', **plan_receipt)), flush=True)

    # These are fixed monitoring cases, not a new claim of independent validation.
    # LN amount supplies coverage strata; it does not assign a semantic LN label.
    selected, used_groups = [], set()
    for split in ('train', 'validation'):
        for lower in (2, 3, 4, 5):
            for ln_rich in (False, True):
                pool = [r for r in candidates if r['split'] == split and
                    r['index_difficulty'] is not None and lower - .5 <= r['index_difficulty'] <= lower + 1.5 and
                    180 <= r['suffix_seconds'] <= 600 and r['onsets'] >= 256 and
                    (r['suffix_ln_head_fraction'] >= .1) == ln_rich]
                pool.sort(key=lambda r: (ordering(r['group_id']), ordering(r['source_sha256'])))
                for row in pool:
                    if row['group_id'] in used_groups:
                        continue
                    source = REPO / row['path']
                    if file_digest(source) != row['source_sha256']:
                        raise ValueError('Selected raw source differs from the catalog')
                    parsed = parse_osu_file(source)
                    rating = compute_mania_star_rating_20241007(parsed.hit_objects, 4, 1.)
                    if lower <= rating < lower + 1 or lower == 5 and rating == 6:
                        selected.append(dict(**row, source_stars=rating, source_star_band=[lower, lower + 1],
                                             ln_amount_stratum='at_least_0.1' if ln_rich else 'below_0.1'))
                        used_groups.add(row['group_id'])
                        break
                else:
                    raise ValueError(f'No fixed source for {split}, {lower} stars, LN-rich={ln_rich}')
    evaluation_pins, windows, cases = {}, [], []
    for index, row in enumerate(selected):
        sha = row['source_sha256']
        evaluation_pins[sha] = pins[sha]
        for start in (0, (row['onsets'] - 256) // 2, row['onsets'] - 256):
            windows.append(dict(source_sha256=sha, first_onset=start, onset_count=256))
        if row['split'] != 'validation':
            continue
        prepared = write_source_condition(REPO / row['path'], sha, inputs / 'conditions' / f'{sha}.json', arm=Arm.R1)
        for seed in (17, 23):
            cases.append(dict(id=f'val-{index:02d}-s{seed}', source_sha256=sha, group_id=row['group_id'],
                split='validation', condition_file=prepared['condition_file'], condition_sha256=prepared['condition_sha256'],
                seed=seed, presentation_source=str(REPO / row['path']), presentation_sha256=sha))
    evaluation = dict(format='vacation/teacher-evaluation-v1', source_cache_dir=str(CACHE),
                      sources=evaluation_pins, windows=windows, native_cases=cases)
    validate_evaluation(evaluation, plan)
    evaluation_file = inputs / 'evaluation.json'
    evaluation_sha = publish_json(evaluation_file, evaluation)
    publish_json(inputs / 'evaluation-selection.json', dict(seed=SELECTION_SEED, selected=selected,
        index_file=str(INDEX), index_sha256=file_digest(INDEX), candidate_proxy='index difficulty within band +/-0.5',
        rating='mania 20241007, native 4K, rate 1, original source object order',
        calculator_sha256=file_digest(REPO / 'src/pulsefield_model/osu_core/difficulty.py'),
        scope='Fixed monitoring; historical validation reuse is not excluded; LN amount is not a semantic judgment'))
    audio = prepare_audio_inputs(plan_file=plan_receipt['path'], plan_sha256=plan_receipt['sha256'],
        catalog_file=inputs / 'catalog.json', catalog_sha256=CATALOG_SHA, catalog_root=REPO,
        source_cache_dir=CACHE, output_file=inputs / 'audio.json')
    config = dict(defaults=['vacation_training_r1_response', '_self_'], mode='preflight', output_dir=str(root / 'run'),
        audio=dict(manifest_file=audio['manifest_file'], manifest_sha256=audio['manifest_sha256']),
        teacher=dict(base_plan_file=plan_receipt['path'], base_plan_sha256=plan_receipt['sha256'],
            evaluation_file=str(evaluation_file), evaluation_sha256=evaluation_sha,
            training=dict(source_cache_dir=str(CACHE), device='cpu', cpu_threads=1)),
        stress=dict(enabled=False))
    config_dir = root / 'config'
    config_dir.mkdir()
    config_file = config_dir / 'r1-response.yaml'
    config_file.write_text('# Fresh R1 response-architecture training; the lost small-model stress checkpoint is unavailable.\n' +
                          OmegaConf.to_yaml(OmegaConf.create(config)))
    teacher_config = {**config, 'defaults': ['vacation_training', '_self_'],
                      'output_dir': str(root / 'teacher35m-run')}
    teacher_file = config_dir / 'teacher35m.yaml'
    teacher_file.write_text('# Separate clean 35M teacher, with audio caching and fresh initialization.\n' +
                           OmegaConf.to_yaml(OmegaConf.create(teacher_config)))
    audio_manifest = json.loads(Path(audio['manifest_file']).read_text())
    quarantine = [a for a in audio_manifest['assets'] if set(a['known_directory_splits']) != {'train'}]
    receipt = dict(format='vacation/rebuilt-inputs-v1',
        source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        driver_sha256=file_digest(Path(__file__)), historical_plan_recovered=False,
        plan=plan_receipt, sampling=asdict(SamplingConfig()), excluded_sources=excluded,
        evaluation_sources=len(evaluation_pins), evaluation_windows=len(windows), native_cases_per_milestone=len(cases),
        evaluation_sha256=evaluation_sha, audio=audio, audio_quarantined_assets=len(quarantine),
        audio_issue_counts=dict(Counter(i['reason'] for i in audio_manifest['issues'])),
        config_file=str(config_file), config_sha256=file_digest(config_file),
        teacher_config_file=str(teacher_file), teacher_config_sha256=file_digest(teacher_file))
    publish_json(root / 'preparation.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 2 or not Path(sys.argv[1]).is_absolute():
        raise SystemExit('Provide one absolute, fresh asset directory outside the worktree')
    main(Path(sys.argv[1]))
