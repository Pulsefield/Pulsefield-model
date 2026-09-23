"""Deterministic source-only selection with original song splits and audio pins."""
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from ..bounded_typed_continuation.condition import condition_from_source
from ..bounded_typed_continuation.contract import Arm
from ..oracle_time_continuation.storage import ROW_DTYPE


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 ** 2), b''):
            value.update(part)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def revision():
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('Experiments require a clean committed source checkout')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def audio_path(path):
    section = None
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        text = line.strip()
        if text == '[HitObjects]':
            break
        if text.startswith('['):
            section = text
        elif section == '[General]' and text.startswith('AudioFilename:'):
            result = (Path(path).parent / text.split(':', 1)[1].strip().replace('\\', '/')).resolve()
            if result.is_file() and result.is_relative_to(Path(path).parent.resolve()):
                return result
    return None


def prepare(config):
    destination = Path(config.root) / 'manifest.json'
    if destination.exists():
        raise ValueError('Preparation needs a fresh manifest destination')
    source_revision = revision()
    if digest(config.catalog_file) != config.catalog_sha256:
        raise ValueError('Catalog digest changed')
    entries = json.loads(Path(config.catalog_file).read_text())
    root, cache = Path(config.catalog_root).resolve(), Path(config.source_cache_dir).resolve()
    pools = defaultdict(list)
    for entry in entries:
        if entry['split'] not in ('train', 'validation'):
            continue
        directory = cache / entry['source_sha256']
        metadata = json.loads((directory / 'metadata.json').read_text())
        if metadata['identity'] != {k: entry[k] for k in ('source_sha256', 'arrangement_sha256', 'group_id', 'split')}:
            raise ValueError('Cache identity differs from the catalog')
        if metadata['seed']['ineligible_reason'] is not None:
            continue
        rows = np.memmap(directory / 'rows.bin', mode='r', dtype=ROW_DTYPE)
        duration = float(rows[-1]['time']) / 1000.
        if not 60 <= duration <= 360:
            continue
        head = np.isin(rows['actions'], (1, 2))
        onsets = int(head.any(-1).sum())
        density = onsets / max(1., float(rows[-1]['time'] - rows[0]['time']) / 1000.)
        ln_fraction = float((rows['actions'] == 2).sum() / head.sum())
        stratum = (0 if density < 4 else 1 if density < 8 else 2, int(ln_fraction >= .1))
        pools[(entry['split'], *stratum)].append(dict(**entry, density=density,
            ln_fraction=ln_fraction, duration_seconds=duration, onsets=onsets,
            rows_file=str(directory / 'rows.bin'), rows_sha256=metadata['rows_sha256'],
            metadata_sha256=digest(directory / 'metadata.json'), stratum=list(stratum)))
    def order(value):
        return hashlib.sha256(f'{config.selection_seed}:{value}'.encode()).hexdigest()
    selected, groups, seen_audio, issues = [], set(), {}, []
    for split, count in (('validation', config.validation_per_stratum), ('train', config.train_per_stratum)):
        for density_band in range(3):
            for ln_rich in range(2):
                found = 0
                for entry in sorted(pools[(split, density_band, ln_rich)],
                                    key=lambda e: (order(e['group_id']), order(e['source_sha256']))):
                    if entry['group_id'] in groups:
                        continue
                    path = root / entry['path']
                    audio = audio_path(path)
                    if audio is None:
                        issues.append(dict(source_sha256=entry['source_sha256'], reason='missing_audio'))
                        continue
                    sha = digest(audio)
                    if sha in seen_audio and seen_audio[sha] != split:
                        issues.append(dict(source_sha256=entry['source_sha256'], reason='cross_split_audio_duplicate'))
                        continue
                    if digest(path) != entry['source_sha256'] or digest(entry['rows_file']) != entry['rows_sha256']:
                        raise ValueError('Selected source or cache digest changed')
                    condition = condition_from_source(path.read_bytes(), entry['source_sha256'], Arm.R1)
                    condition_file = destination.parent / 'conditions' / f"{entry['source_sha256']}.json"
                    write_json(condition_file, condition.payload())
                    selected.append(dict(**entry, source_file=str(path.resolve()), audio_file=str(audio),
                        audio_sha256=sha, condition_file=str(condition_file.resolve()),
                        condition_sha256=digest(condition_file)))
                    groups.add(entry['group_id'])
                    seen_audio[sha] = split
                    found += 1
                    if found == count:
                        break
                if found != count:
                    raise ValueError(f'Insufficient eligible audio in stratum {(split, density_band, ln_rich)}: {found}/{count}')
    result = dict(format='audio-skeleton/corpus-v1', source_revision=source_revision,
                  config=asdict(config), charts=selected, issues=issues,
                  counts=dict(Counter(e['split'] for e in selected)),
                  split_scope='Original train/validation groups; selected exact-audio duplicates excluded; TEST unopened')
    write_json(destination, result)
    return dict(manifest=str(destination), sha256=digest(destination), counts=result['counts'], issues=len(issues))


def read_manifest(root):
    return json.loads((Path(root) / 'manifest.json').read_text())
