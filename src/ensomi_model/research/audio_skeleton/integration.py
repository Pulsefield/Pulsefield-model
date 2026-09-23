"""Native R1 evaluation with predicted release opportunities or full timing."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from ..bounded_typed_continuation.condition import GenerationCondition
from ..bounded_typed_continuation.contract import Arm, Timing
from .corpus import digest, read_manifest, revision, write_json
from .data import pick_events
from .sensitivity import run_case


def predicted_condition(original, heads, release_only, terminal_ms):
    """Retain only the supplied physical seed and its known future obligations."""
    boundary = original.seed_rows[-1].time_ms
    if terminal_ms <= boundary:
        raise ValueError('Predicted schedule needs a true horizon after the seed')
    head_times = {r.time_ms for r in original.seed_rows if any(a in (1, 2) for a in r.actions)}
    head_times.update(float(t) for t in heads if boundary < t <= terminal_ms)
    known = [None if i is None else original.timing.times_ms[i] for i in original.crossing_ends]
    if any(t is not None and t > terminal_ms for t in known):
        raise ValueError('Predicted horizon excludes a committed seed endpoint')
    times = {r.time_ms for r in original.seed_rows} | head_times | {float(terminal_ms)}
    times.update(float(t) for t in release_only if boundary < t <= terminal_ms)
    times.update(t for t in known if t is not None)
    times = tuple(sorted(times))
    lookup = {t: i for i, t in enumerate(times)}
    crossing = tuple(None if t is None else lookup[t] for t in known)
    return GenerationCondition(Arm.R1, Timing(times, tuple(t in head_times for t in times)),
                               original.seed_rows, crossing)


def run(config):
    source_revision = revision()
    root = Path(config.root)
    trained = root / 'training' / config.run_name
    result = json.loads((trained / 'result.json').read_text())
    freeze = json.loads((trained / 'freeze.json').read_text())
    if digest(root / 'manifest.json') != freeze['manifest_sha256']:
        raise ValueError('Integration corpus differs from model training')
    if digest(trained / 'best.pt') != result['checkpoint_sha256']:
        raise ValueError('Selected skeleton model checkpoint changed')
    directory = root / 'integration' / config.run_name
    directory.mkdir(parents=True, exist_ok=False)
    features = json.loads((root / 'features/index.json').read_text())
    entries = {e['source_sha256']: e for e in read_manifest(root)['charts']}
    tasks, failures = [], []
    deadline = time.time() + config.max_seconds
    for sha in freeze['assessment']:
        entry = entries[sha]
        original = GenerationCondition.from_payload(json.loads(Path(entry['condition_file']).read_text()))
        for control in ('supplied_density', 'train_default_density'):
            prediction = np.load(trained / f'{sha}-{control}.npy')
            h, r = [pick_events(prediction[:, k * 2:k * 2 + 2],
                                  prediction[:, 4 + k * 2:6 + k * 2], result['thresholds'][k]) for k in range(2)]
            for variant in ('release_only', 'full_timing'):
                key = f'{variant}-{control}'
                heads = h if variant == 'full_timing' else [t for t, yes in zip(original.timing.times_ms, original.timing.onsets) if yes]
                terminal = (features['assets'][entry['audio_sha256']]['duration_ms'] if variant == 'full_timing'
                            else original.timing.times_ms[-1])
                try:
                    condition = predicted_condition(original, heads, r, terminal)
                except ValueError as error:
                    failures.append(dict(source=sha, variant=key, stage='condition', error=repr(error)))
                    continue
                path = directory / 'conditions' / f'{sha}-{key}.json'
                write_json(path, condition.payload())
                own = replace(config, root=str(directory / key))
                case = dict(entry, condition_file=str(path.resolve()), condition_sha256=digest(path), variant=key)
                for seed in (17, 23):
                    tasks.append((own, case, 0, seed, deadline, source_revision))
    write_json(directory / 'freeze.json', dict(source_revision=source_revision, config=asdict(config),
        skeleton_checkpoint_sha256=result['checkpoint_sha256'], manifest_sha256=freeze['manifest_sha256'],
        thresholds=result['thresholds'], cases=len(tasks),
        scope='Original playable seed retained. Release-only arm retains original H and terminal; full arm uses audio duration.'))
    outputs = []
    with ProcessPoolExecutor(max_workers=config.workers, mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = {executor.submit(run_case, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                out = future.result()
                outputs.append(dict(out, variant=task[1]['variant']))
                print(json.dumps(dict(completed=len(outputs), cases=len(tasks), variant=task[1]['variant'], id=out['id'])), flush=True)
            except Exception as error:
                failures.append(dict(source=task[1]['source_sha256'], variant=task[1]['variant'], seed=task[3], error=repr(error)))
            write_json(directory / 'progress.json', dict(completed=len(outputs), cases=len(tasks), failures=failures))
    summary = dict(status='failed' if failures else 'completed', source_revision=source_revision,
                   results=outputs, failures=failures)
    write_json(directory / 'summary.json', summary)
    return dict(status=summary['status'], outputs=len(outputs), failures=failures, summary=str(directory / 'summary.json'))
