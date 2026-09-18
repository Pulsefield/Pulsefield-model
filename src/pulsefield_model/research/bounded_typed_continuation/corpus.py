"""Shared immutable corpus draws and a bounded cache of exact source indexes.

Sampling metadata never enters the predictor. A plan pins row-cache digests and
aligns exposure checkpoints across arms, including variable-length tail draws.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np

from ..oracle_time_continuation.corpus import catalog_entries, read_split
from ..oracle_time_continuation.data import SourceIdentity
from ..oracle_time_continuation.storage import ROW_DTYPE, SOURCE_FORMAT, file_digest
from ..scoped_style_modeling.dataset import ContractError
from .data import SourceChart, SourceInterval

PLAN_FORMAT = 'bounded-typed/corpus-plan-v1'


@dataclass(frozen=True)
class SamplingConfig:
    seed: int = 471
    horizons: tuple[int, ...] = (128, 256)
    seed_probability: float = .125
    milestones: tuple[int, ...] = (250000, 1000000, 2000000)

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2 ** 63:
            raise ContractError('Corpus draw seed must be a nonnegative integer below 2**63')
        if (not self.horizons or any(type(n) is not int or not 1 <= n <= 256 for n in self.horizons) or
                len(set(self.horizons)) != len(self.horizons)):
            raise ContractError('Corpus horizons must be distinct positive onset counts up to 256')
        if (isinstance(self.seed_probability, bool) or not np.isfinite(self.seed_probability) or
                not 0 <= self.seed_probability <= 1):
            raise ContractError('Seed-window probability must be in [0,1]')
        if (not self.milestones or any(type(n) is not int or n <= 0 for n in self.milestones) or
                any(a >= b for a, b in zip(self.milestones, self.milestones[1:]))):
            raise ContractError('Exposure checkpoints must be increasing positive integers')


def draw_plan(sources, config: SamplingConfig):
    """Group-uniform, then chart-uniform draws, with an explicit seed stratum.

    Interior draws choose a valid fixed-size onset window uniformly. The separate
    seed probability deliberately covers deployment's short initial history.
    Short charts and the final draw before each milestone use their actual size.
    This sampler is not uniform over corpus onsets or elapsed song seconds.
    """
    groups = defaultdict(list)
    for sha, source in sources.items():
        if type(source['onsets']) is not int or source['onsets'] <= 0:
            raise ContractError('Eligible corpus sources require post-seed onsets')
        groups[source['identity']['group_id']].append(sha)
    if not groups:
        raise ContractError('Corpus sampling needs eligible source groups')
    members = [sorted(groups[group]) for group in sorted(groups)]
    random = np.random.default_rng(config.seed)
    draws, exposures = [], 0
    for milestone in config.milestones:
        while exposures < milestone:
            group = members[int(random.integers(len(members)))]
            sha = group[int(random.integers(len(group)))]
            available = sources[sha]['onsets']
            requested = int(random.choice(config.horizons))
            horizon = min(available, requested)
            seed = bool(random.random() < config.seed_probability)
            first = 0 if seed else int(random.integers(available - horizon + 1))
            count = min(horizon, milestone - exposures)
            exposures += count
            draws.append(dict(source_sha256=sha, first_onset=first, onset_count=count,
                              requested_horizon=requested, seed_stratum=seed,
                              exposure_end=exposures, checkpoint=exposures if exposures == milestone else None))
    return draws


def create_plan(*, catalog_path, catalog_sha256, split_manifest, split_sha256,
                census_path, census_sha256, source_cache_dir, output_file,
                sampling=SamplingConfig()):
    """Freeze an existing TRAIN census plus admitted row-cache identities.

    Read-only source preparation never changes the allocation or admits held-out
    payloads. The caller must first prepare any missing admitted TRAIN cache.
    """
    if Path(output_file).exists():
        raise FileExistsError(output_file)
    if file_digest(Path(census_path), 32 * 1024 ** 2) != census_sha256:
        raise ContractError('Corpus census differs from its pinned bytes')
    assignments = read_split(split_manifest, split_sha256)
    catalog = {row['source_sha256']: row for row in catalog_entries(catalog_path, catalog_sha256, assignments, split='train')}
    census = [json.loads(line) for line in Path(census_path).read_text().splitlines()]
    sources = {}
    for row in census:
        sha = row['source_sha256']
        expected = catalog.get(sha)
        if sha in sources or expected is None or expected['group_id'] != row['group_id']:
            raise ContractError('Census source identity differs from the pinned TRAIN catalog')
        identity = SourceIdentity(**{key: expected[key] for key in ('source_sha256', 'arrangement_sha256', 'group_id', 'split')})
        directory = Path(source_cache_dir) / sha
        metadata_path, rows_path = directory / 'metadata.json', directory / 'rows.bin'
        metadata = json.loads(metadata_path.read_text())
        if (metadata['format'] != SOURCE_FORMAT or SourceIdentity(**metadata['identity']) != identity or
                metadata['row_count'] != expected['event_count'] or
                metadata['row_count'] != row['seed_rows'] + row['suffix_rows'] or
                metadata['seed']['seed_row_count'] != row['seed_rows'] or
                metadata['seed']['ineligible_reason'] is not None or
                rows_path.stat().st_size != metadata['row_count'] * ROW_DTYPE.itemsize or
                file_digest(rows_path, 64 * 1024 ** 2) != metadata['rows_sha256']):
            raise ContractError('Admitted corpus cache differs from its source, seed, rows or census')
        sources[sha] = dict(identity=asdict(identity), rows=metadata['row_count'], seed_rows=row['seed_rows'],
                            onsets=row['suffix_onsets'], rows_sha256=metadata['rows_sha256'],
                            metadata_sha256=file_digest(metadata_path))
    plan = dict(format=PLAN_FORMAT, catalog_sha256=catalog_sha256, split_sha256=split_sha256,
                census_sha256=census_sha256, sampling=asdict(sampling), sources=sources,
                draws=draw_plan(sources, sampling))
    destination = Path(output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as stream:
        json.dump(plan, stream, sort_keys=True, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
    return dict(path=str(destination), sha256=file_digest(destination), sources=len(sources),
                groups=len({s['identity']['group_id'] for s in sources.values()}),
                draws=len(plan['draws']), onset_exposures=sampling.milestones[-1])


def read_plan(path, expected_sha256):
    if not expected_sha256 or file_digest(Path(path), 32 * 1024 ** 2) != expected_sha256:
        raise ContractError('Corpus plan differs from its pinned bytes')
    plan = json.loads(Path(path).read_text())
    if plan['format'] != PLAN_FORMAT:
        raise ContractError('Unsupported corpus plan format')
    config = SamplingConfig(**plan['sampling'])
    for sha, source in plan['sources'].items():
        identity = SourceIdentity(**source['identity'])
        if identity.source_sha256 != sha or identity.split != 'train':
            raise ContractError('Corpus plans may reference only their pinned TRAIN sources')
    if draw_plan(plan['sources'], config) != plan['draws']:
        raise ContractError('Corpus draws differ from their declared deterministic sampler')
    return plan


def charged_source_bytes(source: SourceChart):
    """Conservative cache charge, including timing arrays and Python time tuples.

    In-flight intervals may retain an evicted chart until their update completes;
    this charge bounds the cache owner, while the runner also bounds batch size
    and observes process RSS. It is not a process-memory measurement.
    """
    total = source.array_bytes + 8192
    for view in (source.typed, source.untyped):
        total += sum(value.nbytes for value in vars(view).values() if isinstance(value, np.ndarray))
        total += sum(value.nbytes for value in view.gap_starts)
        for values in (view.timing.times_ms, view.timing.onsets):
            if values is not None:
                total += sys.getsizeof(values) + sum(sys.getsizeof(value) for value in values)
    return total


class ChartCache:
    """Count/byte LRU of parameter-independent, digest-checked source indexes."""
    def __init__(self, root, sources, *, max_sources=64, max_bytes=256 * 1024 ** 2):
        if any(type(n) is not int or n <= 0 for n in (max_sources, max_bytes)):
            raise ContractError('Source cache count and byte limits must be positive integers')
        self.root, self.sources = Path(root), sources
        self.max_sources, self.max_bytes = max_sources, max_bytes
        self.cache = OrderedDict()
        self.charged_bytes = self.hits = self.misses = self.evictions = 0

    def get(self, sha):
        if sha in self.cache:
            self.hits += 1
            self.cache.move_to_end(sha)
            return self.cache[sha][0]
        metadata = self.sources[sha]
        directory = self.root / sha
        if (file_digest(directory / 'metadata.json') != metadata['metadata_sha256'] or
                file_digest(directory / 'rows.bin', 64 * 1024 ** 2) != metadata['rows_sha256']):
            raise ContractError('Source cache changed after corpus-plan preparation')
        source = SourceChart.from_cache(directory, SourceIdentity(**metadata['identity']))
        if (len(source.rows) != metadata['rows'] or source.seed_rows != metadata['seed_rows'] or
                len(source.onsets) != metadata['onsets']):
            raise ContractError('Exact source indexes disagree with the corpus plan')
        self.misses += 1
        charged = charged_source_bytes(source)
        if charged <= self.max_bytes:
            while self.cache and (len(self.cache) >= self.max_sources or self.charged_bytes + charged > self.max_bytes):
                _, (_, old_bytes) = self.cache.popitem(last=False)
                self.charged_bytes -= old_bytes
                self.evictions += 1
            self.cache[sha] = (source, charged)
            self.charged_bytes += charged
        return source

    def interval(self, draw):
        return SourceInterval(self.get(draw['source_sha256']), draw['first_onset'], draw['onset_count'])

    def metrics(self):
        return dict(sources=len(self.cache), charged_bytes=self.charged_bytes,
                    hits=self.hits, misses=self.misses, evictions=self.evictions)


def next_batch(plan, cursor, batch_size):
    """Return a deterministic batch that never straddles an exposure checkpoint."""
    if type(cursor) is not int or not 0 <= cursor <= len(plan['draws']) or type(batch_size) is not int or batch_size <= 0:
        raise ContractError('Corpus batch cursor/size is invalid')
    result = []
    while cursor < len(plan['draws']) and len(result) < batch_size:
        draw = plan['draws'][cursor]
        result.append(draw)
        cursor += 1
        if draw['checkpoint'] is not None:
            break
    return result, cursor


class Coverage:
    """Exact unique onset exposure accounting, with compact durable bitmaps."""
    def __init__(self, sources, packed=None):
        self.sources, self.masks = sources, {}
        for sha, payload in (packed or {}).items():
            count = sources[sha]['onsets']
            if len(payload) != (count + 7) // 8:
                raise ContractError('Coverage bitmap length differs from its source onset count')
            self.masks[sha] = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), count=count).astype(np.bool_)

    def commit(self, draws):
        for draw in draws:
            sha, first, count = draw['source_sha256'], draw['first_onset'], draw['onset_count']
            if sha not in self.masks:
                self.masks[sha] = np.zeros(self.sources[sha]['onsets'], dtype=np.bool_)
            mask = self.masks[sha]
            if type(first) is not int or type(count) is not int or first < 0 or count <= 0 or first + count > len(mask):
                raise ContractError('Coverage draw lies outside its source onsets')
            mask[first:first + count] = True

    def snapshot(self):
        return {sha: np.packbits(mask).tobytes() for sha, mask in self.masks.items()}

    def metrics(self):
        return dict(unique_onsets=sum(int(mask.sum()) for mask in self.masks.values()),
                    charts=len(self.masks), groups=len({self.sources[sha]['identity']['group_id'] for sha in self.masks}))
