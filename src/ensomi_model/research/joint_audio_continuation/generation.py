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
from .head_spacing import sample_row


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
            chunk_ms=4000, max_rows=30000, max_seconds=900., stop_callback=None,
            head_spacing_ms=0.):
    """Generate native integer-ms rows; return incomplete output on a resource cap.

    Mel is the complete canonical song representation. It is encoded once, with
    no source rows or seed. Startup coverage includes audio encoding and requires
    fixed decisions through min(8000, duration_ms), including empty intervals.
    The time budget is checked between scheduler steps; a cap never fabricates LN
    endpoints. An optional callback is checked every twenty scheduler steps and
    returns a stop-reason string or None. It does not participate in sampling.
    Reported inference latency excludes decoding audio and computing Mel.
    The optional head-spacing prior thins proposed rows. Its rejected proposals
    advance fixed-through time but not replay/history. When enabled, max_rows
    also bounds total proposals, including rejected ones.
    """
    if (type(duration_ms) is not int or duration_ms < 0 or type(chunk_ms) is not int or chunk_ms <= 0 or
            type(max_rows) is not int or max_rows <= 0 or not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ContractError('Native generation requires a nonnegative duration and positive resource limits')
    if mel.ndim != 2 or mel.shape[1] != 128 or len(mel) == 0 or not np.isfinite(mel).all():
        raise ContractError('Native generation requires finite complete [frames,128] Mel audio')
    if not math.isfinite(head_spacing_ms) or head_spacing_ms < 0:
        raise ContractError('Head-spacing scale must be finite and nonnegative')
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    model.eval()
    _synchronize(device)
    started = time.perf_counter()
    encoded = model.encode_audio(torch.as_tensor(np.array(mel, copy=True), dtype=dtype, device=device)[None])
    _synchronize(device)
    audio_seconds = time.perf_counter() - started
    replay, cache = ExactReplayState(), model.temporal.empty_cache()
    rng = torch.Generator(device='cpu').manual_seed(seed)
    acceptance_rng = torch.Generator(device='cpu').manual_seed(seed ^ 0x5A17)
    cursor, residual, rows = -1, None, []
    latencies, coverage = [], []
    startup_seconds, startup_target = None, min(8000, duration_ms)
    first30_seconds = first30_time_ms = None
    stop_reason, bins_scored, clocks_scored, forced_terminal = 'completed', 0, 0, 0
    proposals, rejected = 0, []
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
        if head_spacing_ms and proposals >= max_rows:
            stop_reason = 'proposal_limit'
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
            log_probs = model.row_log_probs(row_audio, history, row_exact, legal, occupancy)[0]
            actions, accepted, factor = sample_row(log_probs, replay, cursor, rng, acceptance_rng,
                head_spacing_ms, forced_terminal=terminal and any(replay.occupancy))
            proposals += 1
            if accepted:
                row = CompleteRow(cursor, actions)
                previous = None if replay.last_row is None else replay.last_row.time_ms
                replay = commit(replay, row, is_terminal=terminal)
                raw = content_features([row], [previous], [[None] * 4])[0]
                cache = model.temporal.append(cache, torch.as_tensor(raw, dtype=dtype, device=device))
                rows.append(row)
            else:
                rejected.append(dict(time_ms=cursor, actions=actions, log_acceptance=factor,
                                     committed_rows=replay.row_count,
                                     last_lane_head_ms=replay.last_lane_attack_ms))
            residual = None
        _synchronize(device)
        now = time.perf_counter()
        latencies.append(now - tick)
        coverage.append(dict(coverage_ms=cursor, elapsed_seconds=now - started))
        if startup_seconds is None and cursor >= startup_target:
            startup_seconds = now - started
        if first30_seconds is None and replay.note_count >= 30:
            first30_seconds, first30_time_ms = now - started, cursor
    completed = cursor == duration_ms
    if completed and any(replay.occupancy):
        raise ContractError('Completed native generation left an open long note')
    elapsed = time.perf_counter() - started
    return NativeGeneration(tuple(rows), completed, stop_reason, cursor,
        dict(audio_encode_seconds=audio_seconds, generation_seconds=elapsed,
             startup_target_ms=startup_target, startup_seconds=startup_seconds,
             first30_heads_seconds=first30_seconds, first30_heads_through_ms=first30_time_ms,
             latency_scope='cached Mel through fixed chart coverage; excludes waveform decode and Mel computation',
             step_p50_ms=float(np.quantile(latencies, .5) * 1000) if latencies else None,
             step_p99_ms=float(np.quantile(latencies, .99) * 1000) if latencies else None,
             scheduler_steps=len(latencies), timing_bins_scored=bins_scored, timing_clocks_scored=clocks_scored,
             head_spacing_ms=head_spacing_ms, proposed_rows=proposals, rejected_rows=rejected,
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
    if payload.get('format') == 'joint-audio/context-v1':
        from .context_model import ContextAudioModel, ContextModelConfig
        model = ContextAudioModel(ContextModelConfig(**payload['model_config']))
    else:
        model = JointAudioModel(JointModelConfig(**payload['model_config']))
    model.load_state_dict(payload['model'], strict=True)
    if (any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()) or
            not bool((model.audio_std > 0).all())):
        raise ContractError('Joint checkpoint contains nonfinite parameters or invalid audio scales')
    metadata = {key: payload[key] for key in ('model_config', 'source_revision', 'manifest_sha256', 'config')}
    return model.to(device).eval(), metadata


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _source_free_header(audio_file=None):
    title = 'Generated audio' if audio_file is None else Path(audio_file).stem
    title = ''.join(' ' if ord(character) < 32 else character for character in title).strip() or 'Generated audio'
    lines = ['osu file format v14', '[General]', 'Mode:3', '[Metadata]',
             f'Title:{title} (joint audio prototype)', 'Artist:Unknown', 'Creator:Ensomi',
             'Version:Prototype 4K', 'BeatmapID:0', 'BeatmapSetID:-1', '[Difficulty]',
             'HPDrainRate:5', 'CircleSize:4', 'OverallDifficulty:5', 'ApproachRate:5',
             'SliderMultiplier:1.4', 'SliderTickRate:1', '[TimingPoints]',
             '// Constant 120 BPM for editor and scroll only; not an inferred musical beat grid.',
             '0,500,4,2,0,100,1,0', '[HitObjects]']
    return ('\n'.join(lines) + '\n').encode('utf-8')


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
        header = _source_free_header(audio_file) if source_file is None else presentation_header(Path(source_file))
        if source_file is not None:
            header = header.replace(b' (oracle continuation)', b' (joint audio native)')
        if audio_file is not None:
            original = Path(audio_file)
            # Copy into the case directory so the exported chart is playable
            # independently of the source mapset's relative audio location.
            destination = directory / ('audio' + original.suffix.lower())
            shutil.copyfile(original, destination)
            lines = header.decode('utf-8').splitlines()
            if any(line.startswith('AudioFilename:') for line in lines):
                lines = [f'AudioFilename:{destination.name}' if line.startswith('AudioFilename:') else line
                         for line in lines]
            else:
                lines.insert(lines.index('[General]') + 1, f'AudioFilename:{destination.name}')
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


def _clean_revision():
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise ContractError('Experiments require a clean committed source checkout')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def _resource_stop(root):
    if (root / 'PAUSE').exists():
        return 'pause_file'
    if psutil.virtual_memory().available < 2 * 1024 ** 3:
        return 'available_memory_below_2_gib'
    if shutil.disk_usage(root).free < 40 * 1024 ** 3:
        return 'free_disk_below_40_gib'
    return None


def generate(config, *, resolved_yaml=''):
    """Run a pinned, fresh native cohort and persist complete or capped outcomes."""
    config.validate()
    revision = _clean_revision()
    started = time.perf_counter()
    torch.set_num_threads(config.cpu_threads)
    root = Path(config.root).resolve()

    def resource_stop():
        return _resource_stop(root)

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
                         max_seconds=remaining, stop_callback=resource_stop, head_spacing_ms=config.head_spacing_ms)
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


def infer_audio(config, *, resolved_yaml=''):
    """Generate a prototype chart from new audio without opening a chart corpus.

    The checkpoint supplies audio normalization. Canonical decoding and Mel
    computation use the complete input audio; no seed, source chart or beat grid
    is supplied. The fresh run records input/model hashes, frontend identity and
    per-stage latency after imports and Git validation. Resource/time stops are
    checked between preprocessing stages and by the native scheduler. Incomplete
    runs retain available rows without exporting invented hold endpoints.
    """
    from ...features.audio import load_audio_file
    from ...features.mel_base import MUSIC_MEL_CACHE_CONFIG, compute_log_mel_10ms
    from .data import frontend_identity

    config.validate()
    if config.mode != 'infer_audio':
        raise ContractError('New-audio inference requires mode=infer_audio')
    revision = _clean_revision()
    torch.set_num_threads(config.cpu_threads)
    root = Path(config.root).resolve()
    output = root / 'inference' / config.run_name
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.yaml').write_text(resolved_yaml)
    started = time.perf_counter()
    phase, profile = 'initialization', {}
    recipe = dict(format='joint-audio/new-audio-inference-v1', source_revision=revision,
        config=asdict(config), checkpoint_file=str(Path(config.checkpoint_file).resolve()),
        checkpoint_sha256=config.checkpoint_sha256, frontend=frontend_identity(),
        normalization='checkpoint audio_mean/audio_std buffers; no statistics fitted to input audio',
        initialization='BOS; no source beatmap, seed or supplied event times',
        presentation='constant 120 BPM and constant SV for editor/scroll only; no inferred musical beat grid',
        environment=dict(device=config.device, cpu_threads=config.cpu_threads, torch=str(torch.__version__)),
        latency_scope='after module imports and Git validation; includes input verification and artifact writes')

    def check(next_phase):
        nonlocal phase
        phase = next_phase
        reason = _resource_stop(root)
        if reason is None and time.perf_counter() - started >= config.max_seconds:
            reason = 'time_limit'
        if reason:
            raise ResourceLimit(reason)

    try:
        check('input_verification')
        tick = time.perf_counter()
        audio_file = Path(config.audio_file).resolve()
        audio_sha256 = digest(audio_file)
        recipe['audio'] = dict(path=str(audio_file), sha256=audio_sha256, bytes=audio_file.stat().st_size)
        profile['input_verification_seconds'] = time.perf_counter() - tick

        check('model_load')
        tick = time.perf_counter()
        model, checkpoint = load_model(config.checkpoint_file, config.checkpoint_sha256, device=config.device)
        _synchronize(next(model.parameters()).device)
        profile['model_load_seconds'] = time.perf_counter() - tick
        recipe['checkpoint'] = checkpoint
        recipe['parameters'] = model.parameter_counts()

        check('audio_decode')
        tick = time.perf_counter()
        waveform = load_audio_file(audio_file, MUSIC_MEL_CACHE_CONFIG.sample_rate)
        profile['audio_decode_seconds'] = time.perf_counter() - tick
        if not len(waveform) or not np.isfinite(waveform).all():
            raise ContractError('Audio inference requires a nonempty finite decoded waveform')
        if digest(audio_file) != audio_sha256:
            raise ContractError('Input audio changed during decoding')
        duration_ms = len(waveform) * 1000 // MUSIC_MEL_CACHE_CONFIG.sample_rate
        recipe['decoded_audio'] = dict(samples=len(waveform), sample_rate=MUSIC_MEL_CACHE_CONFIG.sample_rate,
            duration_ms=duration_ms, sha256=hashlib.sha256(waveform.tobytes()).hexdigest())

        check('mel_computation')
        tick = time.perf_counter()
        mel = compute_log_mel_10ms(waveform, sample_rate=MUSIC_MEL_CACHE_CONFIG.sample_rate,
                                   config=MUSIC_MEL_CACHE_CONFIG)
        profile['mel_seconds'] = time.perf_counter() - tick
        if mel.ndim != 2 or mel.shape[1] != 128 or not len(mel) or not np.isfinite(mel).all():
            raise ContractError('Canonical audio frontend produced invalid Mel frames')
        mel_file = output / 'mel.npy'
        np.save(mel_file, mel)
        recipe['mel'] = dict(path=str(mel_file), sha256=digest(mel_file), shape=list(mel.shape))
        _write_json(output / 'recipe.json', recipe)

        check('native_generation')
        before_rollout = time.perf_counter() - started
        native = rollout(model, mel, duration_ms, seed=config.generation_seed,
            chunk_ms=config.timing_horizon_ms, max_rows=config.generation_max_rows,
            max_seconds=config.max_seconds - before_rollout, stop_callback=lambda: _resource_stop(root),
            head_spacing_ms=config.head_spacing_ms)
        profile['audio_encode_seconds'] = native.metrics['audio_encode_seconds']
        profile['native_generation_seconds'] = native.metrics['generation_seconds']
        profile['startup_coverage_ms'] = native.metrics['startup_target_ms']
        profile['startup_from_audio_seconds'] = (None if native.metrics['startup_seconds'] is None else
                                                 before_rollout + native.metrics['startup_seconds'])
        profile['first30_heads_from_audio_seconds'] = (None if native.metrics['first30_heads_seconds'] is None else
                                                       before_rollout + native.metrics['first30_heads_seconds'])
        phase = 'export'
        tick = time.perf_counter()
        info = save_rollout(output / 'chart', native, audio_file=audio_file)
        profile['export_seconds'] = time.perf_counter() - tick
        if native.completed and digest(info['audio_file']) != audio_sha256:
            raise ContractError('Bundled audio differs from the decoded input')
        report = dict(status='completed' if native.completed else 'capped', source_revision=revision,
            audio_sha256=audio_sha256, checkpoint_sha256=config.checkpoint_sha256,
            duration_ms=duration_ms, recipe_file=str(output / 'recipe.json'), profile=profile,
            seconds=time.perf_counter() - started, **info)
    except ResourceLimit as error:
        report = dict(status='capped', completed=False, stop_reason=str(error), phase=phase,
                      source_revision=revision, profile=profile, seconds=time.perf_counter()-started,
                      recipe_file=str(output / 'recipe.json'))
        _write_json(output / 'recipe.json', recipe)
    except BaseException as error:
        _write_json(output / 'recipe.json', recipe)
        _write_json(output / 'failure.json', dict(phase=phase, error=repr(error),
                                                 seconds=time.perf_counter()-started))
        raise
    _write_json(output / 'result.json', report)
    return dict(status=report['status'], result_file=str(output / 'result.json'),
                recipe_file=str(output / 'recipe.json'), osu_file=report.get('osu_file'),
                audio_file=report.get('audio_file'), rows=report.get('rows', 0), heads=report.get('heads', 0),
                profile=profile, step_p99_ms=report.get('step_p99_ms'),
                stop_reason=report.get('stop_reason'), seconds=report['seconds'])
