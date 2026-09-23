"""Native joint time/row generation from audio and an empty chart prefix.

An exponential survival threshold persists across scheduler chunks. Chunks do
not become timing inputs or release deadlines; only the decoded song terminal
can force closure. Source charts supply presentation metadata after generation.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import psutil
import torch

from ..audio_skeleton.sensitivity import diagnostics
from ..bounded_typed_continuation.contract import ROW_ACTIONS
from ..bounded_typed_continuation.features import content_features
from ..oracle_time_continuation.data import source_rows
from ..oracle_time_continuation.export import export_osu, presentation_header, verify_rows
from ..oracle_time_continuation.replay import ExactReplayState, commit
from ..oracle_time_continuation.runtime import ResourceConfig, ResourceLimit
from ..oracle_time_continuation.schema import CompleteRow
from ..scoped_style_modeling.dataset import ContractError
from ..source_action_modeling.actions import parse_source
from .batching import interpolate_audio
from .data import digest, load_corpus
from .model import JointAudioModel, JointModelConfig
from .state import exact_features, legal_rows
from .timing import sample_hazards


@dataclass(frozen=True)
class NativeGeneration:
    rows: tuple[CompleteRow, ...]
    completed: bool
    stop_reason: str
    coverage_ms: int
    metrics: dict


def _synchronize(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


@torch.no_grad()
def rollout(model: JointAudioModel, mel: np.ndarray, duration_ms: int, *, seed=17,
            chunk_ms=4000, max_rows=30000, max_seconds=900., stop_callback=None):
    """Generate native integer-ms rows; return incomplete output on a resource cap.

Mel is the complete canonical song representation. It is encoded once, with
no source rows or seed. Startup coverage includes audio encoding and requires
fixed decisions through min(8000, duration_ms), including empty intervals.
The time budget is checked between scheduler steps; a cap never fabricates LN
    endpoints. An optional callback is checked every twenty scheduler steps and
    returns a stop-reason string or None. It does not participate in sampling.
    Reported inference latency excludes decoding audio and computing Mel.
    """
    if (type(duration_ms) is not int or duration_ms < 0 or type(chunk_ms) is not int or chunk_ms <= 0 or
            type(max_rows) is not int or max_rows <= 0 or not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ContractError('Native generation requires a nonnegative duration and positive resource limits')
    if mel.ndim != 2 or mel.shape[1] != 128 or len(mel) == 0 or not np.isfinite(mel).all():
        raise ContractError('Native generation requires finite complete [frames,128] Mel audio')
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    model.eval()
    _synchronize(device)
    started = time.perf_counter()
    encoded = model.encode_audio(torch.as_tensor(np.array(mel, copy=True), dtype=dtype, device=device)[None])
    _synchronize(device)
    audio_seconds = time.perf_counter() - started
    replay, cache = ExactReplayState(), model.temporal.empty_cache()
    rng = torch.Generator(device='cpu').manual_seed(seed)
    cursor, residual, rows = -1, None, []
    latencies, coverage = [], []
    startup_seconds, startup_target = None, min(8000, duration_ms)
    stop_reason, bins_scored, clocks_scored, forced_terminal = 'completed', 0, 0, 0
    while cursor < duration_ms:
        if stop_callback is not None and len(latencies) % 20 == 0:
            reason = stop_callback()
            if reason:
                stop_reason = str(reason)
                break
        if time.perf_counter() - started >= max_seconds:
            stop_reason = 'time_limit'
            break
        if len(rows) >= max_rows:
            stop_reason = 'row_limit'
            break
        tick = time.perf_counter()
        end = min(cursor + chunk_ms, duration_ms)
        bins = torch.arange((cursor + 1) // 10, end // 10 + 1, device=device, dtype=torch.long)
        bin_times = bins * 10 + 9
        audio = interpolate_audio(encoded, bin_times[None])
        exact = torch.as_tensor(exact_features([replay] * len(bins), bin_times.cpu().tolist()),
                                dtype=dtype, device=device)[None]
        history = model.temporal.read(cache)[None]
        logits = model.timing_logits(audio, history, exact).reshape(-1)
        times = (bins[:, None] * 10 + torch.arange(10, device=device)[None]).reshape(-1)
        valid = (times > cursor) & (times <= end)
        forced = (times == duration_ms) & any(replay.occupancy)
        event, residual = sample_hazards(logits, valid, forced, rng, residual)
        bins_scored += len(bins)
        clocks_scored += int(valid.sum())
        if event is None:
            cursor = end
        else:
            cursor = int(times[event])
            terminal = cursor == duration_ms
            forced_terminal += terminal and any(replay.occupancy)
            row_audio = interpolate_audio(encoded, torch.tensor([cursor], device=device))
            row_exact = torch.as_tensor(exact_features([replay], [cursor]), dtype=dtype, device=device)
            legal = torch.as_tensor(legal_rows([replay], [terminal]), device=device)
            occupancy = torch.tensor([replay.occupancy], dtype=torch.bool, device=device)
            probs = model.row_log_probs(row_audio, history, row_exact, legal, occupancy)[0].exp().cpu()
            selected = int(torch.multinomial(probs, 1, generator=rng))
            row = CompleteRow(cursor, ROW_ACTIONS[selected])
            previous = None if replay.last_row is None else replay.last_row.time_ms
            replay = commit(replay, row, is_terminal=terminal)
            raw = content_features([row], [previous], [[None] * 4])[0]
            cache = model.temporal.append(cache, torch.as_tensor(raw, dtype=dtype, device=device))
            rows.append(row)
            residual = None
        _synchronize(device)
        now = time.perf_counter()
        latencies.append(now - tick)
        coverage.append(dict(coverage_ms=cursor, elapsed_seconds=now - started))
        if startup_seconds is None and cursor >= startup_target:
            startup_seconds = now - started
    completed = cursor == duration_ms
    if completed and any(replay.occupancy):
        raise ContractError('Completed native generation left an open long note')
    elapsed = time.perf_counter() - started
    return NativeGeneration(tuple(rows), completed, stop_reason, cursor,
        dict(audio_encode_seconds=audio_seconds, generation_seconds=elapsed,
             startup_target_ms=startup_target, startup_seconds=startup_seconds,
             latency_scope='cached Mel through fixed chart coverage; excludes waveform decode and Mel computation',
             step_p50_ms=float(np.quantile(latencies, .5) * 1000) if latencies else None,
             step_p99_ms=float(np.quantile(latencies, .99) * 1000) if latencies else None,
             scheduler_steps=len(latencies), timing_bins_scored=bins_scored, timing_clocks_scored=clocks_scored,
             forced_terminal_events=forced_terminal, rows=len(rows), heads=replay.note_count,
             open_lanes=list(replay.occupancy), coverage=coverage))


def load_model(path, expected_sha256, *, device='cpu'):
    """Verify pinned checkpoint bytes and restore model plus provenance metadata."""
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ContractError('Joint checkpoint differs from its pinned SHA-256')
    payload = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    required = {'model_config', 'model', 'source_revision', 'manifest_sha256', 'config'}
    if not isinstance(payload, dict) or not required <= set(payload):
        raise ContractError('Joint checkpoint lacks model configuration or training provenance')
    model = JointAudioModel(JointModelConfig(**payload['model_config']))
    model.load_state_dict(payload['model'], strict=True)
    if (any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()) or
            not bool((model.audio_std > 0).all())):
        raise ContractError('Joint checkpoint contains nonfinite parameters or invalid audio scales')
    metadata = {key: payload[key] for key in ('model_config', 'source_revision', 'manifest_sha256', 'config')}
    return model.to(device).eval(), metadata


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def save_rollout(directory, result: NativeGeneration, *, source_file=None, audio_file=None):
    """Persist rows, independently verify them and export only complete charts."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    row_file = directory / 'rows.jsonl'
    with row_file.open('w') as stream:
        for index, row in enumerate(result.rows):
            stream.write(json.dumps(dict(event_id=index, **asdict(row))) + '\n')
    times = [row.time_ms for row in result.rows]
    mechanics = verify_rows(row_file, times, require_complete=result.completed)
    info = dict(completed=result.completed, stop_reason=result.stop_reason,
                coverage_ms=result.coverage_ms, mechanics=mechanics, rows_sha256=digest(row_file),
                **result.metrics)
    if result.completed:
        header = None if source_file is None else presentation_header(Path(source_file))
        if header is not None:
            header = header.replace(b' (oracle continuation)', b' (joint audio native)')
        if audio_file is not None:
            original = Path(audio_file)
            # Copy into the case directory so the exported chart is playable
            # independently of the source mapset's relative audio location.
            destination = directory / ('audio' + original.suffix.lower())
            shutil.copyfile(original, destination)
            if header is not None:
                lines = header.decode('utf-8').splitlines()
                lines = [f'AudioFilename:{destination.name}' if line.startswith('AudioFilename:') else line
                         for line in lines]
                header = ('\n'.join(lines) + '\n').encode('utf-8')
            info['audio_file'] = str(destination.resolve())
        osu = directory / 'generated.osu'
        export_osu(row_file, osu, times,
                   ResourceConfig(output_max_bytes=64 * 1024 ** 2), header=header)
        parsed = parse_source(osu.read_bytes(), digest(osu))
        if source_rows(parsed.objects) != result.rows:
            raise ContractError('Export/reparse changed joint generated rows')
        info.update(osu_file=str(osu.resolve()), osu_sha256=digest(osu), reparse_pass=True,
                    diagnostics=diagnostics(result.rows, -1))
    _write_json(directory / 'result.json', info)
    return info


def _select_cases(charts, split, count):
    pools = defaultdict(list)
    for chart in charts:
        if chart.split == split and chart.entry.get('selection', 'base') == 'base':
            pools[tuple(chart.entry['stratum'])].append(chart)
    for group in pools.values():
        group.sort(key=lambda chart: chart.entry['source_sha256'])
    selected, depth = [], 0
    while len(selected) < count and any(depth < len(pool) for pool in pools.values()):
        for stratum in sorted(pools):
            if depth < len(pools[stratum]) and len(selected) < count:
                selected.append(pools[stratum][depth])
        depth += 1
    return selected


def generate(config, *, resolved_yaml=''):
    """Run a pinned, fresh native cohort and persist complete or capped outcomes."""
    config.validate()
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ContractError('Experiments require a clean committed source checkout')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    started = time.perf_counter()
    torch.set_num_threads(config.cpu_threads)
    root = Path(config.root).resolve()

    def resource_stop():
        if (root / 'PAUSE').exists():
            return 'pause_file'
        if psutil.virtual_memory().available < 2 * 1024 ** 3:
            return 'available_memory_below_2_gib'
        if shutil.disk_usage(root).free < 40 * 1024 ** 3:
            return 'free_disk_below_40_gib'
        return None

    if reason := resource_stop():
        raise ResourceLimit(f'Joint generation stopped before initialization: {reason}')
    manifest_sha256 = digest(root / 'manifest.json')
    model, checkpoint = load_model(config.checkpoint_file, config.checkpoint_sha256, device=config.device)
    if checkpoint['manifest_sha256'] != manifest_sha256:
        raise ContractError('Generation corpus differs from the training checkpoint manifest')
    charts, _ = load_corpus(root)
    cases = _select_cases(charts, config.generation_split, config.generation_cases)
    if not cases:
        raise ContractError('Generation cohort has no cases in the requested split')
    initialization_seconds = time.perf_counter() - started
    output = root / 'generation' / config.run_name
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.yaml').write_text(resolved_yaml)
    _write_json(output / 'freeze.json', dict(source_revision=revision, config=asdict(config),
        checkpoint_sha256=config.checkpoint_sha256, checkpoint=checkpoint, manifest_sha256=manifest_sha256,
        source_sha256=[chart.entry['source_sha256'] for chart in cases], initialization_seconds=initialization_seconds))
    results = []
    for index, chart in enumerate(cases):
        remaining = config.max_seconds - (time.perf_counter() - started)
        if remaining <= 0:
            break
        seed = config.generation_seed + index
        native = rollout(model, chart.mel, chart.duration_ms, seed=seed,
                         chunk_ms=config.timing_horizon_ms, max_rows=config.generation_max_rows,
                         max_seconds=remaining, stop_callback=resource_stop)
        info = save_rollout(output / chart.entry['source_sha256'][:12], native,
                            source_file=chart.entry['source_file'], audio_file=chart.entry['audio_file'])
        info.update(source_sha256=chart.entry['source_sha256'], group_id=chart.group_id,
                    split=chart.split, stratum=chart.entry['stratum'], seed=seed,
                    duration_ms=chart.duration_ms)
        _write_json(output / chart.entry['source_sha256'][:12] / 'result.json', info)
        results.append(info)
        if not native.completed:
            break
    report = dict(source_revision=revision, checkpoint_sha256=config.checkpoint_sha256,
        manifest_sha256=manifest_sha256, device=config.device, cpu_threads=config.cpu_threads,
        initialization_seconds=initialization_seconds, seconds=time.perf_counter() - started,
        requested_cases=len(cases), completed_cases=sum(case['completed'] for case in results),
        status='completed' if len(results) == len(cases) and all(case['completed'] for case in results) else 'capped',
        output=str(output), cases=results)
    _write_json(output / 'result.json', report)
    return dict(status=report['status'], result_file=str(output / 'result.json'),
                completed_cases=report['completed_cases'], requested_cases=report['requested_cases'],
                initialization_seconds=initialization_seconds, seconds=report['seconds'],
                cases=[dict(source_sha256=case['source_sha256'], completed=case['completed'],
                            rows=case['rows'], startup_seconds=case['startup_seconds'],
                            step_p99_ms=case['step_p99_ms']) for case in results])
