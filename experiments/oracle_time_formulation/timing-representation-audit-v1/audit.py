"""Measure onset-skeleton and LN-endpoint representation costs on TRAIN only."""
import heapq
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('artifacts/oracle-time-continuation/m3-20260917')
sys.path.insert(0, str(ROOT / 'clock-readout-v1'))
from common import RUNTIME_SHA, save, verify_runtime
from pulsefield_model.research.oracle_time_continuation.corpus import admit_entry, catalog_entries, read_split
from pulsefield_model.research.oracle_time_continuation.runtime import ResourceConfig, ResourceGuard
from pulsefield_model.research.oracle_time_continuation.storage import ROW_DTYPE, SourceStore, file_digest

OUT = Path(__file__).parent / 'results'
RANK_EDGES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
DURATION_EDGES = [50, 100, 200, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]


def main():
    started = time.perf_counter()
    def deadline(*_):
        raise TimeoutError('10-minute representation audit bound')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(600)
    verify_runtime()
    OUT.mkdir()
    resource_log = (OUT/'resources.jsonl').open('x')
    guard = ResourceGuard('cpu', ResourceConfig(rss_limit_bytes=2*1024**3,
                          min_available_bytes=2*1024**3), resource_log)
    catalog = Path('artifacts/oracle-time-review/20260915-adfb1ee/catalog.json')
    catalog_sha = 'e31b7e8f4daa044503ef2b8411bc41727ba371eec804c9462a4608be8112ad28'
    split = Path('artifacts/scoped-style-modeling/prepare-v1/split-manifest.json')
    split_sha = '15175f45e91cf7299a9a30166731bf38ee7361399b346fb692cf68e76de5992a'
    entries = catalog_entries(catalog, catalog_sha, read_split(split, split_sha), split='train')
    assert len(entries) == 11564 and all(e['split'] == 'train' for e in entries)
    save(OUT/'manifest.json', dict(script_sha256=file_digest(Path(__file__)), runtime_sha256=RUNTIME_SHA,
        catalog_sha256=catalog_sha, split_sha256=split_sha, charts=len(entries), split='train',
        rank_upper_edges=RANK_EDGES, duration_upper_edges_ms=DURATION_EDGES,
        bounds=dict(seconds=600, rss_bytes=2*1024**3, available_bytes=2*1024**3, output_bytes=128*1024**2)))
    store = SourceStore(Path('artifacts/oracle-time-continuation/full-cache-v1'))
    rank_hist = np.zeros(len(RANK_EDGES)+1, dtype=np.int64)
    duration_hist = np.zeros(len(DURATION_EDGES)+1, dtype=np.int64)
    totals = dict(charts=0, eligible_charts=0, event_rows=0, onset_rows=0, release_only_rows=0,
                  notes=0, lns=0, releases_at_onset=0, multi_ln_start_rows=0,
                  multi_ln_start_rows_with_different_ends=0)
    longest_rank, longest_duration = [], []
    with (OUT/'per-chart.jsonl').open('x') as stream:
        for chart_index, entry in enumerate(entries):
            source = admit_entry(store, entry)
            rows = np.fromfile(source.directory/'rows.bin', dtype=ROW_DTYPE)
            assert len(rows) == source.row_count
            actions, times = rows['actions'], rows['time']
            assert np.all(np.diff(times) > 0) and np.all(actions.any(axis=1))
            attacks = ((actions == 1) | (actions == 2)).sum(axis=1)
            heads = (actions == 2).sum(axis=1)
            onset = attacks > 0
            ln_count, onsets_at_end = 0, 0
            by_start = {}
            chart_rank_hist = np.zeros_like(rank_hist)
            chart_duration_hist = np.zeros_like(duration_hist)
            for lane in range(4):
                starts = np.flatnonzero(actions[:, lane] == 2)
                ends = np.flatnonzero(actions[:, lane] == 3)
                assert len(starts) == len(ends)
                assert np.all(starts < ends) and np.all(ends[:-1] < starts[1:])
                ranks = ends - starts
                durations = times[ends] - times[starts]
                chart_rank_hist += np.bincount(np.searchsorted(RANK_EDGES, ranks, side='left'), minlength=len(rank_hist))
                chart_duration_hist += np.bincount(np.searchsorted(DURATION_EDGES, durations, side='left'), minlength=len(duration_hist))
                ln_count += len(starts)
                onsets_at_end += int(onset[ends].sum())
                for start, end in zip(starts[heads[starts] > 1], ends[heads[starts] > 1]):
                    by_start.setdefault(int(start), []).append(int(end))
                for values, heap in [(ranks, longest_rank), (durations, longest_duration)]:
                    if len(values):
                        i = int(values.argmax())
                        item = (float(values[i]), entry['source_sha256'], lane,
                                float(times[starts[i]]), float(times[ends[i]]))
                        heapq.heappush(heap, item)
                        if len(heap) > 12:
                            heapq.heappop(heap)
            record = dict(source_sha256=entry['source_sha256'], group_id=entry['group_id'],
                event_rows=len(rows), onset_rows=int(onset.sum()), release_only_rows=int((~onset).sum()),
                notes=int(attacks.sum()), lns=ln_count, releases_at_onset=onsets_at_end,
                multi_ln_start_rows=len(by_start),
                multi_ln_start_rows_with_different_ends=sum(len(set(v)) > 1 for v in by_start.values()),
                eligible=source.minimum_seed().eligible,
                endpoint_rank_histogram=chart_rank_hist.tolist(), duration_histogram=chart_duration_hist.tolist())
            for key in totals:
                totals[key] += (1 if key == 'charts' else int(record['eligible']) if key == 'eligible_charts' else record[key])
            rank_hist += chart_rank_hist
            duration_hist += chart_duration_hist
            stream.write(json.dumps(record)+'\n')
            if chart_index % 128 == 0:
                guard.check('chart-complete', charts=chart_index+1)
                assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) < 128*1024**2
            if (chart_index+1) % 1000 == 0:
                print(json.dumps(dict(charts=chart_index+1, seconds=time.perf_counter()-started)), flush=True)
    assert totals['event_rows'] == totals['onset_rows'] + totals['release_only_rows']
    assert int(rank_hist.sum()) == int(duration_hist.sum()) == totals['lns']
    result = dict(status='complete', totals=totals, endpoint_rank_upper_edges=RANK_EDGES,
        endpoint_rank_histogram=rank_hist.tolist(), duration_upper_edges_ms=DURATION_EDGES,
        duration_histogram=duration_hist.tolist(),
        largest_per_chart_lane_endpoint_rank_examples=sorted(longest_rank, reverse=True),
        largest_per_chart_lane_duration_examples=sorted(longest_duration, reverse=True),
        per_chart_sha256=file_digest(OUT/'per-chart.jsonl'), seconds=time.perf_counter()-started,
        resources=guard.check('complete'),
        interpretation='Population description of a possible onset-conditioned note-object model. '
            'Endpoint ranks count later positions in the original event union. Source LN ends are targets, '
            'not proposed runtime oracle inputs. This audit does not test model learning, generation quality, '
            'or alignment of different charts in one split group. No validation/test payload is opened.')
    save(OUT/'readout.json', result)
    signal.alarm(0)
    print(json.dumps(dict(status='complete', **totals, seconds=result['seconds'])), flush=True)


if __name__ == '__main__':
    main()
