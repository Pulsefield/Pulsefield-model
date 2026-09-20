"""Independent streaming row validation and ordered, disk-staged osu! export."""
import json
import os
from pathlib import Path
import sqlite3

from ..scoped_style_modeling.dataset import ContractError
from .runtime import ResourceConfig, ResourceLimit, sync_directory
from .publication import staging_directory


def presentation_header(source_path: Path) -> bytes:
    """Copy playback metadata for export only; no HitObjects enter this header.

    The audio filename and timing/SV points refer to the original mapset. Audio
    is neither read by the model nor bundled. The generated difficulty receives
    a new identity so importing it does not replace the source difficulty.
    """
    sections = {}
    section = None
    size = 0
    allowed = {'General', 'Metadata', 'Difficulty', 'TimingPoints'}
    with Path(source_path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            size += len(line.encode())
            if size > 1024**2:
                raise ContractError('Source playback header exceeds 1 MiB')
            stripped = line.strip()
            if stripped == '[HitObjects]':
                break
            if stripped.startswith('[') and stripped.endswith(']'):
                section = stripped[1:-1]
            elif section in allowed and stripped and not stripped.startswith('//'):
                sections.setdefault(section, []).append(stripped)
    def fields(name):
        return dict(line.split(':', 1) for line in sections.get(name, []) if ':' in line)
    general = {key.strip(): value.strip() for key, value in fields('General').items()
               if key.strip() in ('AudioFilename', 'AudioLeadIn', 'PreviewTime', 'SampleSet', 'StackLeniency')}
    general['Mode'] = '3'
    metadata = {key.strip(): value.strip() for key, value in fields('Metadata').items()}
    metadata.update(Creator='Pulsefield', Version=metadata.get('Version', 'Source') + ' (oracle continuation)',
                    BeatmapID='0', BeatmapSetID='-1')
    difficulty = {key.strip(): value.strip() for key, value in fields('Difficulty').items()}
    difficulty['CircleSize'] = '4'
    lines = ['osu file format v14']
    for name, values in (('General', general), ('Metadata', metadata), ('Difficulty', difficulty)):
        lines += [f'[{name}]', *(f'{key}:{value}' for key, value in values.items())]
    lines += ['[TimingPoints]', *sections.get('TimingPoints', ['0,500,4,2,0,100,1,0']), '[HitObjects]']
    return ('\n'.join(lines) + '\n').encode()


def verify_rows(path: Path, times, *, require_complete=True, limit_bytes=None):
    open_heads = [None] * 4
    count = notes = closes = 0
    with Path(path).open('rb') as stream:
        while limit_bytes is None or stream.tell() < limit_bytes:
            line = stream.readline(-1 if limit_bytes is None else limit_bytes - stream.tell())
            if not line:
                break
            if not line.endswith(b"\n"):
                raise ContractError("Export row boundary is not a complete persisted line")
            value = json.loads(line)
            time, actions = value['time_ms'], value['actions']
            if (value['event_id'] != count or count >= len(times) or time != times[count] or
                    len(actions) != 4 or not any(actions)):
                raise ContractError('Export row identity, time, action width or nonempty rule failed')
            for lane, action in enumerate(actions):
                if type(action) is not int or action not in (0, 1, 2, 3):
                    raise ContractError('Invalid exported lane action')
                if action in (1, 2):
                    if open_heads[lane] is not None:
                        raise ContractError('Export attacks an occupied lane')
                    notes += 1
                    if action == 2:
                        open_heads[lane] = time
                elif action == 3:
                    if open_heads[lane] is None or time <= open_heads[lane]:
                        raise ContractError('Export closes a missing or nonpositive LN')
                    open_heads[lane] = None
                    closes += 1
            count += 1
    if require_complete and (count != len(times) or any(t is not None for t in open_heads)):
        raise ContractError('Export is incomplete or leaves open long notes')
    return dict(rows=count, notes=notes, ln_closes=closes, open_heads=open_heads)


def export_osu(rows_path: Path, destination: Path, times, config: ResourceConfig = ResourceConfig(), *, header=None):
    """Stream rows to disk sort storage; retain only four pending LN heads in RAM.

    Times remain exact; no endpoint is copied from a source suffix or invented.
    A complete independently verified row file is required before publication.
    """
    import shutil
    report = verify_rows(rows_path, times)
    with staging_directory(destination) as temporary:
        if shutil.disk_usage(destination.parent).free < config.disk_reserve_bytes + 2 * config.output_max_bytes:
            raise ResourceLimit('Insufficient disk space for export staging and reserve')
        db = sqlite3.connect(temporary / 'notes.sqlite')
        try:
            page_size = db.execute('PRAGMA page_size').fetchone()[0]
            max_pages = config.output_max_bytes // page_size
            if max_pages < 2 or db.execute(f'PRAGMA max_page_count={max_pages}').fetchone()[0] > max_pages:
                raise ResourceLimit('Export staging cannot fit within output_max_bytes')
            db.execute('PRAGMA cache_size=-2048')
            db.execute('CREATE TABLE notes (time REAL, lane INTEGER, end REAL, PRIMARY KEY(time,lane)) WITHOUT ROWID')
            heads = [None] * 4
            with rows_path.open('rb') as stream:
                for line in stream:
                    value = json.loads(line)
                    time = value['time_ms']
                    for lane, action in enumerate(value['actions']):
                        if action == 1:
                            db.execute('INSERT INTO notes VALUES (?, ?, ?)', (time, lane, time))
                        elif action == 2:
                            heads[lane] = time
                        elif action == 3:
                            db.execute('INSERT INTO notes VALUES (?, ?, ?)', (heads[lane], lane, time))
                            heads[lane] = None
            db.commit()
            if (temporary / 'notes.sqlite').stat().st_size > config.output_max_bytes:
                raise ResourceLimit('Export staging exceeds output_max_bytes')
            with (temporary / 'chart.osu').open('wb') as output:
                def write(data):
                    if output.tell() + len(data) > config.output_max_bytes:
                        raise ResourceLimit('Export exceeds output_max_bytes')
                    output.write(data)
                write(header if header is not None else b'osu file format v14\n[General]\nMode:3\n[Metadata]\nTitle:Oracle-time continuation\nArtist:Pulsefield\nCreator:Pulsefield\nVersion:Generated\n[Difficulty]\nCircleSize:4\n[TimingPoints]\n0,500,4,2,0,100,1,0\n[HitObjects]\n')
                for start, lane, end in db.execute('SELECT time,lane,end FROM notes ORDER BY time,lane'):
                    kind = 128 if end > start else 1
                    tail = f'{end:.17g}:0:0:0:0:' if kind == 128 else '0:0:0:0:'
                    write(f'{64 + 128 * lane},192,{start:.17g},{kind},0,{tail}\n'.encode())
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary / 'chart.osu', destination)
            sync_directory(destination.parent)
        except sqlite3.OperationalError as exc:
            # Python 3.10 does not expose sqlite_errorcode on these exceptions.
            if str(exc) == 'database or disk is full':
                raise ResourceLimit('Export staging reached its SQLite page or disk limit') from exc
            raise
        finally:
            db.close()
    return report
