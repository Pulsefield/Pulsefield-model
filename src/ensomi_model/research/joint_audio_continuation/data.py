"""Pinned paired-chart corpus and source replay for audio-conditioned event queries.

Mel frames use the canonical uncentered music frontend. Source rows are labels;
only observed content and exact past state leave the source owner as inputs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from ...features import audio as audio_frontend, mel_base
from ...features.audio import load_audio_file
from ...features.mel_base import MUSIC_MEL_CACHE_CONFIG, compute_log_mel_10ms
from ..bounded_typed_continuation.contract import Arm
from ..bounded_typed_continuation.data import SourceChart
from ..bounded_typed_continuation.features import CONTENT_DIM
from ..oracle_time_continuation.data import SourceIdentity
from ..oracle_time_continuation.replay import ExactReplayState

MEL_HOP_MS = 10
MEL_SUPPORT_MS = 40
MEL_CENTER_ORIGIN_MS = 20
IDENTITY_KEYS = ('source_sha256', 'arrangement_sha256', 'group_id', 'split')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 ** 2), b''):
            value.update(part)
    return value.hexdigest()


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def frontend_identity():
    """Return the pinned frontend, frame clock and implementation hashes."""
    config = MUSIC_MEL_CACHE_CONFIG
    if (config.sample_rate, config.mel_bins, config.hop_ms, config.n_fft, config.win_length) != (24000, 128, 10, 960, 960):
        raise ValueError('Joint audio corpus requires the canonical 24 kHz music Mel frontend')
    return dict(config={key: value for key, value in asdict(config).items() if key != 'cache_root'},
                config_hash=config.mel_config_hash,
                audio_code_sha256=digest(audio_frontend.__file__),
                mel_code_sha256=digest(mel_base.__file__),
                frame_hop_ms=MEL_HOP_MS, frame_support_ms=MEL_SUPPORT_MS,
                frame_center_origin_ms=MEL_CENTER_ORIGIN_MS,
                frame_count='ceil(decoded_sample_count / hop_samples)',
                padding='right zero padding to cover every uncentered 40 ms window',
                waveform='load_audio_file: mono, speed=1, peak normalized')


def _audio_path(source):
    section = None
    for line in Path(source).read_text(encoding='utf-8-sig').splitlines():
        text = line.strip()
        if text == '[HitObjects]':
            break
        if text.startswith('['):
            section = text
        elif section == '[General]' and text.startswith('AudioFilename:'):
            result = (Path(source).parent / text.split(':', 1)[1].strip().replace('\\', '/')).resolve()
            if result.is_file() and result.is_relative_to(Path(source).parent.resolve()):
                return result
    return None


def load_source_aliases(path, expected_sha256, catalog, catalog_sha256):
    """Read explicit source-file alternatives; never infer pairing from song names.

    The pinned manifest maps catalog TRAIN/VAL source hashes to nonempty lists
    of candidate paths. Each candidate's bytes and path are checked when used.
    Candidates must be under catalog_root; relative paths use that root.
    """
    if digest(path) != expected_sha256:
        raise ValueError('Source aliases differ from their pinned SHA-256')
    manifest = json.loads(Path(path).read_text())
    if (manifest.get('format') != 'joint-audio/source-aliases-v1' or
            manifest.get('catalog_sha256') != catalog_sha256):
        raise ValueError('Source aliases use a different format or catalog identity')
    sources = manifest.get('sources')
    allowed = {e['source_sha256'] for e in catalog if e['split'] in ('train', 'validation')}
    if not isinstance(sources, dict) or not set(sources) <= allowed:
        raise ValueError('Source aliases must name only catalog TRAIN/VAL sources')
    if any(not isinstance(paths, list) or not paths or
           any(not isinstance(p, str) or not p for p in paths) for paths in sources.values()):
        raise ValueError('Each source alias requires a nonempty list of file paths')
    return sources


def _paired_audio(source_file, source_root, expected_sha256, aliases):
    choices = {}
    for source in [source_file, *sorted((source_root / p).resolve() for p in aliases)]:
        if not source.is_relative_to(source_root) or digest(source) != expected_sha256:
            raise ValueError('Paired source alias path or bytes differ from the pinned source')
        audio = _audio_path(source)
        if audio is not None:
            choices.setdefault(digest(audio), (audio, source))
    if len(choices) > 1:
        raise ValueError('Identical source aliases refer to ambiguous audio bytes')
    if not choices:
        raise ValueError('Source has no readable local paired audio')
    return next(iter(choices.values()))


def _source_entry(entry, source_root, cache_root, *, paired_source_files=()):
    source_root, cache_root = Path(source_root).resolve(), Path(cache_root).resolve()
    source_file = (source_root / entry['path']).resolve()
    if not source_file.is_relative_to(source_root) or digest(source_file) != entry['source_sha256']:
        raise ValueError('Source path or bytes differ from the pinned catalog')
    directory = cache_root / entry['source_sha256']
    source = SourceChart.from_cache(directory, SourceIdentity(**{key: entry[key] for key in IDENTITY_KEYS}))
    if np.any(source.rows['time'] != np.rint(source.rows['time'])):
        raise ValueError('Joint event clock requires original integer-millisecond source times')
    audio_file, paired_source = _paired_audio(source_file, source_root, entry['source_sha256'], paired_source_files)
    heads = np.isin(source.rows['actions'], (1, 2))
    density = float(heads.any(-1).sum() / max(1., (source.rows['time'][-1] - source.rows['time'][0]) / 1000.))
    ln_fraction = float((source.rows['actions'] == 2).sum() / heads.sum())
    result = dict(entry, source_file=str(source_file), audio_file=str(audio_file), audio_sha256=digest(audio_file),
                  rows_file=str(directory / 'rows.bin'), rows_sha256=digest(directory / 'rows.bin'),
                  metadata_sha256=digest(directory / 'metadata.json'), density=density, ln_fraction=ln_fraction,
                  stratum=[0 if density < 4 else 1 if density < 8 else 2, int(ln_fraction >= .1)],
                  row_count=len(source.rows), last_row_ms=int(source.rows['time'][-1]))
    if paired_source != source_file:
        result['paired_source_file'] = str(paired_source)
    return result


def select_entries(base, catalog, *, source_root, cache_root, max_train_alternatives=2, source_aliases=None):
    """Verify the base cohort and add separate targets from its TRAIN audio groups.

    Catalog TEST entries are never opened. Alternatives must retain the original
    group and exact audio bytes. Rejected alternatives are returned as exclusions;
    any invalid base chart aborts so the validation cohort cannot silently shrink.
    """
    if type(max_train_alternatives) is not int or max_train_alternatives < 0:
        raise ValueError('Alternative count must be a nonnegative integer')
    source_root, cache_root = Path(source_root).resolve(), Path(cache_root).resolve()
    source_aliases = {} if source_aliases is None else source_aliases
    by_sha = {entry['source_sha256']: entry for entry in catalog if entry['split'] in ('train', 'validation')}
    selected, exclusions, groups, audio_splits = [], [], {}, {}
    for original in base['charts']:
        sha = original['source_sha256']
        if original['split'] not in ('train', 'validation') or sha not in by_sha:
            raise ValueError('Base chart is outside the pinned TRAIN/VAL catalog')
        catalog_entry = by_sha[sha]
        if any(original[key] != catalog_entry[key] for key in (*IDENTITY_KEYS, 'path')):
            raise ValueError('Base identity differs from the pinned catalog')
        entry = _source_entry(catalog_entry, source_root, cache_root, paired_source_files=source_aliases.get(sha, ()))
        for key in ('audio_sha256', 'rows_sha256', 'metadata_sha256'):
            if entry[key] != original[key]:
                raise ValueError(f'Base {key} differs from its pinned manifest')
        group, split, audio = entry['group_id'], entry['split'], entry['audio_sha256']
        if group in groups or (audio in audio_splits and audio_splits[audio] != split):
            raise ValueError('Base cohort repeats a group or leaks audio across splits')
        groups[group], audio_splits[audio] = entry, split
        selected.append(dict(entry, selection='base'))
    chosen = {entry['source_sha256'] for entry in selected}
    candidates = defaultdict(list)
    for entry in catalog:
        if (entry['split'] == 'train' and entry['group_id'] in groups and
                groups[entry['group_id']]['split'] == 'train' and entry['source_sha256'] not in chosen):
            candidates[entry['group_id']].append(entry)
    for group in sorted(candidates):
        accepted = 0
        for candidate in sorted(candidates[group], key=lambda entry: entry['source_sha256']):
            if accepted == max_train_alternatives:
                break
            try:
                entry = _source_entry(candidate, source_root, cache_root,
                                      paired_source_files=source_aliases.get(candidate['source_sha256'], ()))
                if entry['audio_sha256'] != groups[group]['audio_sha256']:
                    raise ValueError('Alternative audio bytes differ from the original TRAIN group')
            except (ValueError, OSError, KeyError) as error:
                exclusions.append(dict(source_sha256=candidate['source_sha256'], reason=str(error)))
                continue
            selected.append(dict(entry, selection='alternative'))
            accepted += 1
    return selected, exclusions


def prepare(config):
    """Write fresh canonical features and a pinned corpus; requires clean Git state.

    Normalization counts every frame of each unique TRAIN audio once. Source rows
    beyond decoded audio are reported and rejected, never accommodated by padding
    the song duration. The 40 ms frontend padding changes features only.
    """
    started = time.monotonic()
    root = Path(config.root).resolve()
    if (root / 'manifest.json').exists():
        raise ValueError('Preparation requires a fresh corpus manifest destination')
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ValueError('Experiments require a clean committed source checkout')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if digest(config.catalog_file) != config.catalog_sha256:
        raise ValueError('Catalog digest differs from its pin')
    base = json.loads(Path(config.base_manifest_file).read_text())
    if base['config']['catalog_sha256'] != config.catalog_sha256:
        raise ValueError('Base cohort and alternative catalog have different pins')
    catalog = json.loads(Path(config.catalog_file).read_text())
    aliases = ({} if config.source_aliases_file is None else
               load_source_aliases(config.source_aliases_file, config.source_aliases_sha256,
                                   catalog, config.catalog_sha256))
    selected, exclusions = select_entries(base, catalog,
        source_root=config.catalog_root, cache_root=config.source_cache_dir,
        max_train_alternatives=config.max_train_alternatives, source_aliases=aliases)
    assets, count = {}, 0
    total = np.zeros(MUSIC_MEL_CACHE_CONFIG.mel_bins, np.float64)
    total_square = np.zeros_like(total)
    (root / 'features').mkdir(parents=True, exist_ok=True)
    for entry in selected:
        sha = entry['audio_sha256']
        if sha in assets:
            continue
        if time.monotonic() - started > config.max_seconds:
            raise TimeoutError('Canonical audio preparation exceeded its wall-clock budget')
        waveform = load_audio_file(entry['audio_file'], MUSIC_MEL_CACHE_CONFIG.sample_rate)
        if not len(waveform) or not np.isfinite(waveform).all():
            raise ValueError('Decoded audio must contain finite nonempty samples')
        mel = compute_log_mel_10ms(waveform, sample_rate=MUSIC_MEL_CACHE_CONFIG.sample_rate,
                                 config=MUSIC_MEL_CACHE_CONFIG)
        path = root / 'features' / f'{sha}.npy'
        np.save(path, mel)
        assets[sha] = dict(audio_file=entry['audio_file'], audio_sha256=sha,
            decoded_sample_count=len(waveform), sample_rate=MUSIC_MEL_CACHE_CONFIG.sample_rate,
            decoded_sha256=hashlib.sha256(waveform.tobytes()).hexdigest(),
            duration_ms=len(waveform) * 1000 // MUSIC_MEL_CACHE_CONFIG.sample_rate,
            mel_file=str(path), mel_sha256=digest(path), mel_shape=list(mel.shape), split=entry['split'])
        if entry['split'] == 'train':
            values = mel.astype(np.float64)
            count += len(values)
            total += values.sum(0)
            total_square += (values * values).sum(0)
    admitted, base_failures = [], []
    for entry in selected:
        asset = assets[entry['audio_sha256']]
        if entry['last_row_ms'] * asset['sample_rate'] > asset['decoded_sample_count'] * 1000:
            failure = dict(source_sha256=entry['source_sha256'], reason='source_rows_beyond_true_audio_end',
                           last_row_ms=entry['last_row_ms'], duration_ms=asset['duration_ms'])
            exclusions.append(failure)
            if entry['selection'] == 'base':
                base_failures.append(failure)
        else:
            admitted.append(entry)
    _json(root / 'exclusions.json', exclusions)
    if base_failures:
        raise ValueError('Base source rows exceed true audio end; see exclusions.json; cohort was not modified')
    if count == 0:
        raise ValueError('Normalization requires at least one TRAIN audio frame')
    mean = total / count
    std = np.sqrt(np.maximum(total_square / count - mean * mean, 1e-12))
    normalization = dict(mean=mean.tolist(), std=std.tolist(), frame_count=count,
        audio_sha256=sorted(sha for sha, asset in assets.items() if asset['split'] == 'train'),
        scope='unique TRAIN audio frames; alternatives do not duplicate normalization weight',
        frontend=frontend_identity())
    _json(root / 'normalization.json', normalization)
    manifest = dict(format='joint-audio/corpus-v1', source_revision=revision,
        base_manifest_file=str(Path(config.base_manifest_file).resolve()),
        base_manifest_sha256=digest(config.base_manifest_file), catalog_sha256=config.catalog_sha256,
        frontend=frontend_identity(), charts=admitted, assets=assets,
        normalization_file=str(root / 'normalization.json'), normalization_sha256=digest(root / 'normalization.json'),
        exclusions_sha256=digest(root / 'exclusions.json'), counts=dict(Counter(e['split'] for e in admitted)),
        split_scope='Original TRAIN groups with separate arrangements; unchanged VAL cohort; TEST unopened')
    if config.source_aliases_file is not None:
        manifest['source_aliases_file'] = str(Path(config.source_aliases_file).resolve())
        manifest['source_aliases_sha256'] = config.source_aliases_sha256
    _json(root / 'manifest.json', manifest)
    return dict(manifest=str(root / 'manifest.json'), sha256=digest(root / 'manifest.json'),
                counts=manifest['counts'], exclusions=len(exclusions), seconds=time.monotonic() - started)


@dataclass(frozen=True)
class JointChart:
    entry: dict
    source: SourceChart
    mel: np.ndarray
    duration_ms: int

    @property
    def group_id(self):
        return self.entry['group_id']

    @property
    def split(self):
        return self.entry['split']


def load_corpus(root):
    """Verify pinned bytes and return charts plus TRAIN-only normalization metadata."""
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['format'] != 'joint-audio/corpus-v1' or manifest['frontend'] != frontend_identity():
        raise ValueError('Corpus format or canonical frontend identity changed')
    if digest(manifest['normalization_file']) != manifest['normalization_sha256']:
        raise ValueError('TRAIN normalization bytes changed')
    if digest(root / 'exclusions.json') != manifest['exclusions_sha256']:
        raise ValueError('Corpus exclusion record changed')
    normalization = json.loads(Path(manifest['normalization_file']).read_text())
    if normalization['frontend'] != manifest['frontend']:
        raise ValueError('Normalization uses a different audio frontend')
    expected_train = sorted({entry['audio_sha256'] for entry in manifest['charts'] if entry['split'] == 'train'})
    if normalization['audio_sha256'] != expected_train:
        raise ValueError('Normalization audio identities differ from TRAIN')
    arrays, charts = {}, []
    for entry in manifest['charts']:
        if entry['split'] not in ('train', 'validation'):
            raise ValueError('Corpus includes a forbidden split')
        sha, asset = entry['audio_sha256'], manifest['assets'][entry['audio_sha256']]
        if sha not in arrays:
            if digest(asset['audio_file']) != sha or digest(asset['mel_file']) != asset['mel_sha256']:
                raise ValueError('Audio or canonical Mel bytes changed')
            arrays[sha] = np.load(asset['mel_file'], mmap_mode='r')
            expected_frames = (asset['decoded_sample_count'] + MUSIC_MEL_CACHE_CONFIG.hop_length - 1) // MUSIC_MEL_CACHE_CONFIG.hop_length
            if (asset['sample_rate'] != MUSIC_MEL_CACHE_CONFIG.sample_rate or arrays[sha].dtype != np.float32 or
                    arrays[sha].shape != (expected_frames, MUSIC_MEL_CACHE_CONFIG.mel_bins)
                    or asset['duration_ms'] != asset['decoded_sample_count'] * 1000 // asset['sample_rate']):
                raise ValueError('Mel array or decoded audio clock differs from its identity')
        directory = Path(entry['rows_file']).parent
        if (digest(entry['audio_file']) != sha or digest(entry['source_file']) != entry['source_sha256'] or
                digest(entry['rows_file']) != entry['rows_sha256'] or
                digest(directory / 'metadata.json') != entry['metadata_sha256']):
            raise ValueError('Source, paired audio or admitted cache bytes changed')
        if 'paired_source_file' in entry:
            paired = Path(entry['paired_source_file'])
            if (digest(paired) != entry['source_sha256'] or
                    _audio_path(paired) != Path(entry['audio_file']).resolve()):
                raise ValueError('Paired source alias bytes or audio reference changed')
        source = SourceChart.from_cache(directory, SourceIdentity(**{key: entry[key] for key in IDENTITY_KEYS}))
        if source.rows['time'][-1] > asset['duration_ms']:
            raise ValueError('Source rows exceed the true audio clock')
        charts.append(JointChart(entry, source, arrays[sha], asset['duration_ms']))
    return charts, normalization


@dataclass(frozen=True)
class JointQuery:
    cursor_ms: int
    horizon_end_ms: int
    source_index: int
    replay: ExactReplayState
    raw: np.ndarray
    valid: np.ndarray
    truncated: bool
    history_start_index: int
    predecessor_time_ms: float | None
    target_index: int | None
    target_time_ms: int | None
    target_actions: tuple[int, ...] | None

    @property
    def censored(self):
        return self.target_index is None


def query(chart: JointChart, cursor_ms: int, *, history_limit=511, horizon_ms=4000):
    """Observe rows at or before cursor; label the next row or a censored interval.

    BOS uses -1 ms so an original event at zero remains a future target. History
    retains its true predecessor interval after truncation. Holds survive empty
    crops with no endpoint disclosure. Source exhaustion never marks completion;
    true audio end is represented only by horizon_end_ms == chart.duration_ms.
    """
    if type(cursor_ms) is not int or not -1 <= cursor_ms <= chart.duration_ms:
        raise ValueError('Query cursor must be an integer in [-1, audio duration]')
    if type(history_limit) is not int or history_limit <= 0 or type(horizon_ms) is not int or horizon_ms <= 0:
        raise ValueError('History limit and query horizon must be positive integers')
    source = chart.source
    stop = int(np.searchsorted(source.rows['time'], cursor_ms, side='right'))
    start = max(0, stop - history_limit)
    replay = replace(source.state(Arm.R0, stop).replay, is_complete=False)
    raw = source.content(Arm.R0, start, stop) if stop > start else np.empty((0, 2, CONTENT_DIM), np.float32)
    horizon_end = min(cursor_ms + horizon_ms, chart.duration_ms)
    target = stop if stop < len(source.rows) and source.rows['time'][stop] <= horizon_end else None
    return JointQuery(cursor_ms, horizon_end, stop, replay, raw, np.ones(len(raw), np.bool_), start > 0,
        start, float(source.rows['time'][start - 1]) if start else None, target,
        int(source.rows['time'][target]) if target is not None else None,
        tuple(int(value) for value in source.rows['actions'][target]) if target is not None else None)


def train_groups(charts):
    """Return TRAIN chart lists by group for uniform group-then-chart sampling."""
    groups = defaultdict(list)
    for chart in charts:
        if chart.split == 'train':
            groups[chart.group_id].append(chart)
    return [sorted(groups[group], key=lambda chart: chart.entry['source_sha256']) for group in sorted(groups)]


def smoke_charts(charts, *, per_stratum=1):
    """Select deterministic original TRAIN charts across density and LN strata."""
    if type(per_stratum) is not int or per_stratum <= 0:
        raise ValueError('Smoke count per stratum must be positive')
    pools = defaultdict(list)
    for chart in charts:
        if chart.split == 'train' and chart.entry.get('selection', 'base') == 'base':
            pools[tuple(chart.entry['stratum'])].append(chart)
    return [chart for key in sorted(pools)
            for chart in sorted(pools[key], key=lambda item: item.entry['source_sha256'])[:per_stratum]]
