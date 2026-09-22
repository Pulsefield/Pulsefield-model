"""Disk-owned source rows with count/byte LRU limits and bounded read staging."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import os

import numpy as np

from ..scoped_style_modeling.dataset import ContractError
from .data import SeedSelection, SourceIdentity
from .schema import CompleteRow

ROW_DTYPE = np.dtype([('time', '<f8'), ('actions', 'u1', (4,))])
SOURCE_FORMAT = 'oracle-time/source-rows-v1'


def file_digest(path: Path, limit: int | None = None) -> str:
    if limit is not None and path.stat().st_size > limit:
        raise ContractError(f'{path}: exceeds supported byte limit {limit}')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class SourceCacheConfig:
    max_sources: int = 16
    max_bytes: int = 256 * 1024**2
    staging_rows: int = 4096
    source_max_bytes: int = 64 * 1024**2

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) is not int or value <= 0:
                raise ContractError(f'{name} must be a positive integer')
        if self.source_max_bytes > 64 * 1024**2:
            raise ContractError('Source admission supports at most 64 MiB')


class DiskColumn(Sequence):
    def __init__(self, source, column):
        self.source, self.column = source, column

    def __len__(self):
        return self.source.row_count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = self.source.store.row(self.source, index)
        return float(row['time']) if self.column == 'time' else CompleteRow(
            float(row['time']), tuple(int(a) for a in row['actions']))


@dataclass(frozen=True)
class DiskSkeleton:
    """Already admitted scheduler times, accessed without materializing a tuple."""
    times_ms: DiskColumn


class DiskSource:
    def __init__(self, store, directory: Path, metadata: dict):
        self.store, self.directory = store, directory
        self.identity = SourceIdentity(**metadata['identity'])
        self.seed = SeedSelection(**metadata['seed'])
        self.row_count = metadata['row_count']
        self.skeleton = DiskSkeleton(DiskColumn(self, 'time'))
        self.targets = DiskColumn(self, 'row')

    def minimum_seed(self):
        return self.seed


class SourceStore:
    """Source handles hold metadata only; eviction releases every owned row buffer.

    A source larger than max_bytes is read through one staging block. Slices
    materialize only the requested rows; callers must use chunk-sized slices.
    Admission streams through the canonical raw parser and a disk-backed sort
    behind the source byte cap; no full target tuple or event dictionary is built.
    """
    def __init__(self, root: Path, config: SourceCacheConfig = SourceCacheConfig()):
        self.root, self.config = Path(root), config
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache = OrderedDict()
        self.owned_bytes = self.hits = self.misses = self.evictions = 0
        self.staging = None

    def admit(self, path: Path, sha: str, *, group_id: str, split: str) -> DiskSource:
        path = Path(path)
        if file_digest(path, self.config.source_max_bytes) != sha:
            raise ContractError('Source bytes differ from their SHA-256 identity')
        directory = self.root / sha
        if not directory.exists():
            import shutil
            from .publication import staging_directory
            with staging_directory(directory) as temporary:
                if shutil.disk_usage(self.root).free < 512 * 1024**2 + path.stat().st_size * 8:
                    raise ContractError('Insufficient disk space for source admission staging')
                metadata = stream_admission(path, temporary, sha, group_id, split)
                if file_digest(path, self.config.source_max_bytes) != sha:
                    raise ContractError('Source bytes changed during admission')
                with (temporary / 'metadata.json').open('w') as output:
                    json.dump(metadata, output)
                    output.flush()
                    os.fsync(output.fileno())
                os.rename(temporary, directory)
                descriptor = os.open(self.root, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        metadata = json.loads((directory / 'metadata.json').read_text())
        identity = metadata['identity']
        if (metadata['format'] != SOURCE_FORMAT or identity['source_sha256'] != sha or
                identity['group_id'] != group_id or identity['split'] != split or
                (directory / 'rows.bin').stat().st_size != metadata['row_count'] * ROW_DTYPE.itemsize or
                file_digest(directory / 'rows.bin') != metadata['rows_sha256']):
            raise ContractError('Disk source schema, identity, allocation or row digest mismatch')
        return DiskSource(self, directory, metadata)

    def row(self, source: DiskSource, index: int):
        sha = source.identity.source_sha256
        size = source.row_count * ROW_DTYPE.itemsize
        if sha in self.cache:
            self.hits += 1
            self.cache.move_to_end(sha)
            return self.cache[sha][index]
        if size <= self.config.max_bytes:
            self.misses += 1
            while self.cache and (len(self.cache) >= self.config.max_sources or self.owned_bytes + size > self.config.max_bytes):
                _, old = self.cache.popitem(last=False)
                self.owned_bytes -= old.nbytes
                self.evictions += 1
            data = np.fromfile(source.directory / 'rows.bin', dtype=ROW_DTYPE)
            data.flags.writeable = False
            self.cache[sha] = data
            self.owned_bytes += data.nbytes
            return data[index]
        first = index // self.config.staging_rows * self.config.staging_rows
        if self.staging is None or self.staging[:2] != (sha, first):
            self.misses += 1
            count = min(self.config.staging_rows, source.row_count - first)
            with (source.directory / 'rows.bin').open('rb') as stream:
                stream.seek(first * ROW_DTYPE.itemsize)
                data = np.frombuffer(stream.read(count * ROW_DTYPE.itemsize), dtype=ROW_DTYPE)
            if len(data) != count:
                raise ContractError('Disk source was truncated while reading')
            self.staging = (sha, first, data)
        else:
            self.hits += 1
        return self.staging[2][index - first]

    def metrics(self):
        return dict(sources=len(self.cache), owned_bytes=self.owned_bytes,
                    staging_bytes=0 if self.staging is None else self.staging[2].nbytes,
                    hits=self.hits, misses=self.misses, evictions=self.evictions)

    def clear(self):
        self.cache.clear()
        self.staging = None
        self.owned_bytes = 0


def stream_admission(path: Path, directory: Path, sha: str, group_id: str, split: str) -> dict:
    """Use canonical raw parsing, disk sorting and constant-size exact replay.

    Arrangement hashing reproduces the existing sorted canonical-JSON signature.
    No whole-chart object list, event dictionary or targets tuple is retained.
    """
    from itertools import groupby
    import sqlite3
    from ..scoped_style_modeling.dataset import canonical_json
    from ..scoped_style_modeling.replay import iter_source_objects
    from .replay import ExactReplayState, commit
    db = sqlite3.connect(directory / 'admission.sqlite')
    try:
        db.execute('PRAGMA cache_size=-2048')
        db.execute('PRAGMA temp_store=FILE')
        db.execute('CREATE TABLE objects (lane INTEGER, kind TEXT, start REAL, end REAL, line INTEGER, signature TEXT)')
        db.execute('CREATE TABLE events (time REAL, lane INTEGER, action INTEGER)')
        with path.open('r', encoding='utf-8-sig', newline=None) as stream:
            for note in iter_source_objects(stream):
                db.execute('INSERT INTO objects VALUES (?, ?, ?, ?, ?, ?)',
                           (note.column, note.kind, note.start_ms, note.end_ms, note.source_line,
                            canonical_json([note.column, note.kind, note.start_ms, note.end_ms])))
                db.execute('INSERT INTO events VALUES (?, ?, ?)',
                           (note.start_ms, note.column, 2 if note.kind == 'long' else 1))
                if note.kind == 'long':
                    db.execute('INSERT INTO events VALUES (?, ?, ?)', (note.end_ms, note.column, 3))
        db.commit()
        previous = [None] * 4
        for lane, kind, start, end, line in db.execute('SELECT lane,kind,start,end,line FROM objects ORDER BY start,lane,line'):
            old = previous[lane]
            if old is not None and (start <= old[0] or start < old[1] or (old[2] == 'long' and start == old[1])):
                raise ContractError(f'Ambiguous same-lane or V3-incompatible source objects {old[3]}/{line}')
            previous[lane] = (start, end, kind, line)
        h = hashlib.sha256(b'[')
        first = True
        for (signature,) in db.execute('SELECT signature FROM objects ORDER BY lane,kind,start,end'):
            if not first:
                h.update(b',')
            h.update(signature.encode())
            first = False
        h.update(b']')
        identity = SourceIdentity(sha, h.hexdigest(), group_id, split)
        replay, seed = ExactReplayState(), None
        row_hash = hashlib.sha256()
        row_count = db.execute('SELECT COUNT(DISTINCT time) FROM events').fetchone()[0]
        with (directory / 'rows.bin').open('wb') as output:
            events = db.execute('SELECT time,lane,action FROM events ORDER BY time,lane')
            import struct
            for time, group in groupby(events, lambda r: r[0]):
                actions = [0] * 4
                for _, lane, action in group:
                    if actions[lane]:
                        raise ContractError('Source has more than one action in a lane at the same time')
                    actions[lane] = action
                row = CompleteRow(time, tuple(actions))
                replay = commit(replay, row, is_terminal=replay.row_count + 1 == row_count)
                if seed is None and replay.note_count >= 30:
                    seed = SeedSelection(replay.note_count, replay.row_count, time - replay.first_time_ms,
                                         'no-target-suffix' if replay.row_count == row_count else None)
                data = struct.pack('<d4B', time, *actions)
                row_hash.update(data)
                output.write(data)
            output.flush()
            os.fsync(output.fileno())
        if seed is None:
            seed = SeedSelection(replay.note_count, row_count,
                                 0. if not row_count else replay.last_row.time_ms - replay.first_time_ms,
                                 'fewer-than-30-notes')
        return dict(format=SOURCE_FORMAT, identity=asdict(identity), seed=asdict(seed), row_count=row_count,
                    rows_sha256=row_hash.hexdigest())
    finally:
        db.close()
        (directory / 'admission.sqlite').unlink(missing_ok=True)
