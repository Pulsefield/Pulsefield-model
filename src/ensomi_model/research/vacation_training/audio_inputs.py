"""Freeze TRAIN audio content and chart associations without repairing originals."""
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..bounded_typed_continuation.corpus import read_plan
from ..oracle_time_continuation.storage import ROW_DTYPE, file_digest
from ..scoped_style_modeling.dataset import ContractError
from .control import publish_json, read_json


def prepare_audio_inputs(*, plan_file, plan_sha256, catalog_file, catalog_sha256,
                        catalog_root, source_cache_dir, output_file):
    """Hash original audio once per path and retain every TRAIN difficulty link.

    Only admitted TRAIN payloads are opened. Other split metadata can flag a
    shared mapset directory, but this is not a cross-split content census.
    Missing or ambiguous assets remain explicit issues; paths are not guessed.
    """
    destination = Path(output_file)
    if destination.exists():
        raise ContractError('Audio input preparation requires a new output_file')
    plan = read_plan(plan_file, plan_sha256)
    if catalog_sha256 != plan['catalog_sha256']:
        raise ContractError('Audio catalog must match the frozen training allocation')
    catalog = read_json(catalog_file, catalog_sha256)
    by_sha = {e['source_sha256']: e for e in catalog}
    if len(by_sha) != len(catalog):
        raise ContractError('Audio catalog has duplicate source identities')
    root = Path(catalog_root).resolve()
    def source_path(entry):
        path = Path(entry['path'])
        return (path if path.is_absolute() else root / path).resolve()
    directory_splits = defaultdict(set)
    for entry in catalog:
        directory_splits[source_path(entry).parent].add(entry['split'])
    assets, paths, issues = {}, {}, []
    for sha, pin in sorted(plan['sources'].items()):
        entry = by_sha.get(sha)
        if entry is None or any(entry.get(k) != v for k, v in pin['identity'].items()):
            raise ContractError('Audio source allocation differs from the training plan')
        source = source_path(entry)
        if file_digest(source, 64 * 1024**2) != sha:
            raise ContractError('Raw TRAIN source changed after catalog admission')
        filename, section = None, None
        with source.open(encoding='utf-8-sig') as stream:
            for line in stream:
                text = line.strip()
                if text == '[HitObjects]':
                    break
                if text.startswith('['):
                    section = text
                elif section == '[General]' and text.startswith('AudioFilename:'):
                    filename = text.split(':', 1)[1].strip()
        if not filename:
            issues.append(dict(source_sha256=sha, reason='missing_audio_filename'))
            continue
        audio = (source.parent / filename.replace('\\', '/')).resolve()
        if not audio.is_relative_to(source.parent) or not audio.is_file():
            issues.append(dict(source_sha256=sha, reason='missing_or_outside_audio_path', path=str(audio)))
            continue
        if audio not in paths:
            paths[audio] = file_digest(audio, 1024**3)
        audio_sha = paths[audio]
        rows_file = Path(source_cache_dir) / sha / 'rows.bin'
        if file_digest(rows_file) != pin['rows_sha256']:
            raise ContractError('Audio time audit row cache differs from the plan')
        rows = np.memmap(rows_file, mode='r', dtype=ROW_DTYPE)
        association = dict(source_sha256=sha, group_id=pin['identity']['group_id'], split='train',
                           first_ms=float(rows[0]['time']), last_ms=float(rows[-1]['time']))
        del rows
        asset = assets.setdefault(audio_sha, dict(sha256=audio_sha, path=str(audio), aliases=[], sources=[],
                                                 known_directory_splits=[]))
        if str(audio) not in asset['aliases']:
            asset['aliases'].append(str(audio))
        asset['sources'].append(association)
        asset['known_directory_splits'] = sorted(set(asset['known_directory_splits']) | directory_splits[source.parent])
    result = dict(format='vacation/audio-inputs-v1', plan_sha256=plan_sha256, catalog_sha256=catalog_sha256,
                  content_scope='TRAIN payloads only; other splits checked by directory metadata, not content',
                  assets=[assets[key] for key in sorted(assets)], issues=issues)
    digest = publish_json(destination, result)
    return dict(manifest_file=str(destination.resolve()), manifest_sha256=digest,
                assets=len(assets), missing_or_ambiguous=len(issues))
