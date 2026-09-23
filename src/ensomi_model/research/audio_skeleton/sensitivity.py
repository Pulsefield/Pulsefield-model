"""Paired native R1 rollout under additional non-head timing opportunities."""
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import psutil
import torch

from ..bounded_typed_continuation.condition import GenerationCondition
from ..bounded_typed_continuation.contract import Arm, Timing
from ..bounded_typed_continuation.generation import Rollout
from ..bounded_typed_continuation.model import BoundedModel, ModelConfig
from ..bounded_typed_continuation.verification import verify_complete
from ..oracle_time_continuation.data import source_rows
from ..oracle_time_continuation.export import export_osu, presentation_header
from ..oracle_time_continuation.runtime import ResourceConfig
from ..source_action_modeling.actions import parse_source
from .corpus import digest, read_manifest, revision, write_json


def augment(condition, every):
    """Preserve every original event, H, physical seed and crossing endpoint time."""
    if every == 0:
        return condition
    if every < 1:
        raise ValueError('Candidate insertion period must be positive')
    times = condition.timing.times_ms
    added = []
    eligible = 0
    for i in range(len(condition.seed_rows) - 1, len(times) - 1):
        if times[i + 1] - times[i] >= 20:
            if eligible % every == 0:
                added.append((times[i] + times[i + 1]) / 2)
            eligible += 1
    combined = tuple(sorted((*times, *added)))
    heads = {t for t, h in zip(times, condition.timing.onsets) if h}
    ends = tuple(None if i is None else bisect_left(combined, times[i]) for i in condition.crossing_ends)
    return GenerationCondition(Arm.R1, Timing(combined, tuple(t in heads for t in combined)),
                               condition.seed_rows, ends)


def diagnostics(rows, seed_end):
    starts, previous_head, previous_release = [None] * 4, [None] * 4, [None] * 4
    durations = []
    heads = ln_heads = releases_only = crowded = onsets = short20 = short40 = 0
    for row in rows:
        t = row.time_ms
        suffix = t > seed_end
        has_head = any(a in (1, 2) for a in row.actions)
        if suffix:
            onsets += has_head
            crowded += has_head and sum(s is not None for s in starts) >= 2
            releases_only += not has_head
        for lane, a in enumerate(row.actions):
            if a in (1, 2):
                if suffix:
                    heads += 1
                    ln_heads += a == 2
                    gaps = [t - p for p in (previous_head[lane], previous_release[lane]) if p is not None]
                    short20 += bool(gaps) and min(gaps) < 20
                    short40 += bool(gaps) and min(gaps) < 40
                previous_head[lane] = t
                if a == 2:
                    starts[lane] = t
            elif a == 3:
                if starts[lane] is None:
                    raise ValueError('Diagnostic replay found an unmatched release')
                if starts[lane] > seed_end:
                    durations.append(t - starts[lane])
                starts[lane] = None
                previous_release[lane] = t
    if any(t is not None for t in starts):
        raise ValueError('Diagnostic replay found an unclosed hold')
    return dict(heads=heads, ln_heads=ln_heads, release_only_rows=releases_only,
                held_two_plus_onset_fraction=crowded / max(1, onsets),
                below20_per1000heads=1000 * short20 / max(1, heads),
                below40_per1000heads=1000 * short40 / max(1, heads),
                ln_fraction=ln_heads / max(1, heads),
                ln_duration_median_ms=float(np.median(durations)) if durations else None,
                ln_duration_p90_ms=float(np.quantile(durations, .9)) if durations else None)


def load_model(path, sha):
    if digest(path) != sha:
        raise ValueError('R1 checkpoint digest changed')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    settings = dict(payload['config']['model'])
    settings['arm'] = Arm(settings['arm'])
    model = BoundedModel(ModelConfig(**settings))
    model.load_state_dict(payload['model'], strict=True)
    return model.eval()


def run_case(task):
    config, entry, every, seed, deadline, source_revision = task
    torch.set_num_threads(1)
    name = f"{entry['source_sha256'][:12]}-r{every}-s{seed}"
    directory = Path(config.root) / 'sensitivity' / name
    directory.mkdir(parents=True, exist_ok=False)
    if digest(entry['condition_file']) != entry['condition_sha256']:
        raise ValueError('Condition digest changed')
    condition = augment(GenerationCondition.from_payload(json.loads(Path(entry['condition_file']).read_text())), every)
    write_json(directory / 'condition.json', condition.payload())
    model = load_model(config.checkpoint_file, config.checkpoint_sha256)
    started = time.perf_counter()
    rollout = Rollout.from_seed(model, condition.timing, condition.seed_rows, condition.crossing)
    prefill = time.perf_counter() - started
    rows = list(condition.seed_rows)
    latencies = []
    rng = torch.Generator().manual_seed(seed)
    while not rollout.state.finished:
        if rollout.state.index % 128 == 0:
            if time.time() > deadline or (Path(config.root) / 'PAUSE').exists():
                raise RuntimeError('Sensitivity time/pause boundary reached')
            if psutil.virtual_memory().available < 2 * 1024 ** 3:
                raise RuntimeError('Available memory below 2 GiB')
        before = time.perf_counter()
        step = rollout.step(rng)
        latencies.append(time.perf_counter() - before)
        if step.row is not None:
            rows.append(step.row)
    elapsed = time.perf_counter() - started
    verified = verify_complete(rows, condition.timing, Arm.R1, condition.seed_rows, condition.crossing)
    with (directory / 'rows.jsonl').open('w') as stream:
        for i, row in enumerate(rows):
            stream.write(json.dumps(dict(event_id=i, **asdict(row))) + '\n')
    export_osu(directory / 'rows.jsonl', directory / 'generated.osu', [r.time_ms for r in rows],
               ResourceConfig(disk_reserve_bytes=40 * 1024 ** 3, output_max_bytes=64 * 1024 ** 2),
               header=presentation_header(Path(entry['source_file'])))
    contents = (directory / 'generated.osu').read_bytes()
    if tuple(source_rows(parse_source(contents, digest(directory / 'generated.osu')).objects)) != tuple(rows):
        raise ValueError('Export/reparse changed generated rows')
    result = dict(id=name, source_revision=source_revision, source_sha256=entry['source_sha256'],
        stratum=entry['stratum'], seed=seed, insertion_period=every,
        checkpoint_sha256=config.checkpoint_sha256, condition_sha256=digest(directory / 'condition.json'),
        candidates=len(condition.timing.times_ms), mechanics=verified, reparse_pass=True,
        generation_seconds=elapsed, prefill_seconds=prefill,
        step_p50_ms=float(np.quantile(latencies, .5) * 1000),
        step_p99_ms=float(np.quantile(latencies, .99) * 1000),
        diagnostics=diagnostics(rows, condition.seed_rows[-1].time_ms))
    write_json(directory / 'result.json', result)
    return result


def run(config):
    source_revision = revision()
    output = Path(config.root) / 'sensitivity'
    output.mkdir(exist_ok=False)
    manifest = read_manifest(config.root)
    entries = [e for e in manifest['charts'] if e['split'] == 'validation']
    deadline = time.time() + config.max_seconds
    tasks = [(config, e, period, seed, deadline, source_revision)
             for e in entries for period in (0, 4, 1) for seed in (17, 23)]
    write_json(output / 'freeze.json', dict(source_revision=source_revision, config=asdict(config),
        manifest_sha256=digest(Path(config.root) / 'manifest.json'), cases=len(tasks), seeds=[17, 23],
        insertion_periods=[0, 4, 1]))
    results, failures = [], []
    with ProcessPoolExecutor(max_workers=config.workers, mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = {executor.submit(run_case, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(json.dumps(dict(completed=len(results), cases=len(tasks), id=result['id'],
                                      seconds=result['generation_seconds'])), flush=True)
            except Exception as error:
                failures.append(dict(source=task[1]['source_sha256'], period=task[2], seed=task[3], error=repr(error)))
            write_json(output / 'progress.json', dict(completed=len(results), cases=len(tasks), failures=failures))
    paired = []
    by_key = {(r['source_sha256'], r['seed'], r['insertion_period']): r for r in results}
    for r in results:
        if not r['insertion_period']:
            continue
        baseline = by_key.get((r['source_sha256'], r['seed'], 0))
        if baseline:
            a, b = baseline['diagnostics'], r['diagnostics']
            paired.append(dict(source=r['source_sha256'], seed=r['seed'], period=r['insertion_period'],
                duration_ratio=b['ln_duration_median_ms'] / a['ln_duration_median_ms']
                    if a['ln_duration_median_ms'] and b['ln_duration_median_ms'] else None,
                ln_fraction_delta=b['ln_fraction'] - a['ln_fraction'],
                release_rows_delta=b['release_only_rows'] - a['release_only_rows'],
                short40_delta=b['below40_per1000heads'] - a['below40_per1000heads']))
    result = dict(status='failed' if failures else 'completed', source_revision=source_revision,
                  results=results, paired=paired, failures=failures)
    write_json(output / 'summary.json', result)
    return dict(status=result['status'], outputs=len(results), failures=failures, summary=str(output / 'summary.json'))
