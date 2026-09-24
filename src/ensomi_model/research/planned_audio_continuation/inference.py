"""Full audio to incremental rows, settled coverage and a verified playtest export.

The callback and flushed JSONL file share the same ordered records. They expose
committed rows with open LN heads, never guessed future endpoints. Slow callbacks
apply backpressure without changing model RNG. This is a synchronous local
producer; playback, network transport and crash resumption belong to consumers.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from ...features.audio import load_audio_file
from ...features.mel_base import MUSIC_MEL_CACHE_CONFIG, compute_log_mel_10ms
from ..joint_audio_continuation.data import digest, frontend_identity
from ..joint_audio_continuation.generation import _clean_revision, _resource_stop, _synchronize, save_rollout
from ..scoped_style_modeling.dataset import ContractError
from .buffering import rollout_buffered
from .generation import load_model, rollout
from .inference_config import AudioInferenceConfig

STREAM_FORMAT = 'planned-audio/stream-v1'


class _PreprocessingStop(Exception):
    pass


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def infer_audio(config: AudioInferenceConfig, *, resolved_yaml='', on_event=None):
    """Generate from full audio and BOS into a fresh directory; return run metadata.

    Requires a clean Git checkout and pinned planned checkpoint, not its training
    corpus. The callback receives independent JSON-compatible records after each
    has been flushed to events.jsonl. Callback errors propagate and preserve the
    written prefix with an error record. There is no restart/resume promise.

    Time/resource limits are checked between preprocessing stages and scheduler
    steps; a single operation, callback or final export may exceed the time bound.
    A capped run keeps its prefix and open LNs, without exporting fabricated tails.
    Readiness requires the configured row count and coverage, or true completion
    for shorter charts. It does not certify musical quality or future throughput.
    With screen_unpublished, callbacks see only windows accepted by the bounded
    joint-continuation screen. Attempt exhaustion preserves the published prefix.
    """
    config.validate()
    revision = _clean_revision()
    torch.set_num_threads(config.cpu_threads)
    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'resolved.yaml').write_text(resolved_yaml)
    _write(output / 'config.json', asdict(config))
    started = time.perf_counter()
    phase, profile = 'initialization', {}
    duration_ms, ready_seconds = None, None
    row_count, coverage, completed, sequence = 0, -1, False, 0
    events_file = output / 'events.jsonl'
    recipe = dict(format=STREAM_FORMAT, source_revision=revision, config=asdict(config),
        checkpoint_sha256=config.checkpoint_sha256, frontend=frontend_identity(),
        environment=dict(device=config.device, cpu_threads=config.cpu_threads, torch=str(torch.__version__)),
        initialization='BOS, with complete audio and checkpoint normalization; no source chart, seed or redlines',
        latency_scope='after imports, configuration, Git validation and output setup; includes model load and fresh audio preprocessing')

    def stop_reason():
        return _resource_stop(output) or ('time_limit' if time.perf_counter() - started >= config.max_seconds else None)

    def check(next_phase):
        nonlocal phase
        phase = next_phase
        if reason := stop_reason():
            raise _PreprocessingStop(reason)

    with events_file.open('x') as stream:
        def emit(kind, *, notify=True, **fields):
            nonlocal sequence
            event = dict(kind=kind, sequence=sequence, elapsed_seconds=time.perf_counter()-started, **fields)
            serialized = json.dumps(event, allow_nan=False)
            stream.write(serialized + '\n')
            stream.flush()
            sequence += 1
            if notify and on_event is not None:
                on_event(json.loads(serialized))

        def record_failure(error):
            _write(output / 'recipe.json', recipe)
            _write(output / 'failure.json', dict(phase=phase, error=repr(error), coverage_ms=coverage,
                completed=completed, rows=row_count, seconds=time.perf_counter()-started))
            emit('error', notify=False, phase=phase, error=repr(error), coverage_ms=coverage, completed=completed)

        try:
            emit('start', format=STREAM_FORMAT, key_count=4, lane_index_base=0,
                 actions=['EMPTY', 'TAP', 'LN_START', 'LN_CLOSE'], source_revision=revision,
                 checkpoint_sha256=config.checkpoint_sha256,
                 startup_coverage_ms=config.startup_coverage_ms, startup_min_rows=config.startup_min_rows)
            check('input_verification')
            tick = time.perf_counter()
            audio_file = Path(config.audio_file).resolve()
            audio_sha = digest(audio_file)
            recipe['audio'] = dict(path=str(audio_file), sha256=audio_sha, bytes=audio_file.stat().st_size)
            profile['input_verification_seconds'] = time.perf_counter() - tick

            check('model_load')
            tick = time.perf_counter()
            model, metadata = load_model(config.checkpoint_file, config.checkpoint_sha256, device=config.device)
            if config.arrangement_profile is not None and config.arrangement_profile >= model.config.profile_count:
                raise ContractError('Requested arrangement_profile is absent from this checkpoint')
            _synchronize(next(model.parameters()).device)
            profile['model_load_seconds'] = time.perf_counter() - tick
            recipe['checkpoint'] = metadata

            check('audio_decode')
            tick = time.perf_counter()
            waveform = load_audio_file(audio_file, MUSIC_MEL_CACHE_CONFIG.sample_rate)
            profile['audio_decode_seconds'] = time.perf_counter() - tick
            if not len(waveform) or not np.isfinite(waveform).all():
                raise ContractError('Audio inference requires a nonempty finite waveform')
            if digest(audio_file) != audio_sha:
                raise ContractError('Audio bytes changed during decoding')
            duration_ms = len(waveform) * 1000 // MUSIC_MEL_CACHE_CONFIG.sample_rate
            recipe['decoded_audio'] = dict(samples=len(waveform), sample_rate=MUSIC_MEL_CACHE_CONFIG.sample_rate,
                duration_ms=duration_ms, sha256=hashlib.sha256(waveform.tobytes()).hexdigest())

            check('mel_computation')
            tick = time.perf_counter()
            mel = compute_log_mel_10ms(waveform, sample_rate=MUSIC_MEL_CACHE_CONFIG.sample_rate,
                                      config=MUSIC_MEL_CACHE_CONFIG)
            profile['mel_seconds'] = time.perf_counter() - tick
            if mel.ndim != 2 or mel.shape[1] != 128 or not len(mel) or not np.isfinite(mel).all():
                raise ContractError('Canonical Mel must have nonempty finite 128-bin frames')
            recipe['mel'] = dict(shape=list(mel.shape), array_sha256=hashlib.sha256(mel.tobytes()).hexdigest())
            _write(output / 'recipe.json', recipe)
            emit('audio_ready', duration_ms=duration_ms, audio_sha256=audio_sha, profile=dict(profile))

            check('native_generation')
            before_rollout = time.perf_counter() - started

            def publish(update):
                nonlocal row_count, coverage, completed, ready_seconds
                row_count += update.row is not None
                coverage, completed = update.coverage_ms, update.completed
                ready = completed or (coverage >= min(config.startup_coverage_ms, duration_ms) and
                                      row_count >= config.startup_min_rows)
                if ready and ready_seconds is None:
                    ready_seconds = time.perf_counter() - started
                emit('update', **asdict(update), row_count=row_count, playback_ready=ready)

            options = dict(seed=config.seed, chunk_ms=config.chunk_ms,
                head_chunk_ms=config.head_chunk_ms, max_rows=config.max_rows,
                max_seconds=config.max_seconds-before_rollout, stop_callback=stop_reason,
                on_update=publish, arrangement_profile=config.arrangement_profile)
            if config.screen_unpublished:
                native = rollout_buffered(model, mel, duration_ms, **options)
            else:
                native = rollout(model, mel, duration_ms,
                                 correct_short_attacks=config.correct_short_attacks, **options)
            profile.update(audio_encode_seconds=native.metrics['audio_encode_seconds'],
                generation_seconds=native.metrics['generation_seconds'],
                playback_ready_seconds=ready_seconds,
                first30_rows_seconds=None if native.metrics['first30_rows_seconds'] is None else
                    before_rollout+native.metrics['first30_rows_seconds'],
                first8_seconds=None if native.metrics['startup_seconds'] is None else
                    before_rollout+native.metrics['startup_seconds'])
            phase = 'export'
            tick = time.perf_counter()
            saved = save_rollout(output / 'chart', native, audio_file=audio_file)
            profile['export_seconds'] = time.perf_counter()-tick
            if native.completed and digest(saved['audio_file']) != audio_sha:
                raise ContractError('Exported audio differs from the decoded input')
            report = dict(status='completed' if native.completed else 'capped', duration_ms=duration_ms,
                          audio_sha256=audio_sha, **saved)
        except _PreprocessingStop as error:
            report = dict(status='capped', completed=completed, stop_reason=str(error), coverage_ms=coverage,
                          rows=row_count, duration_ms=duration_ms, phase=phase)
        except BaseException as error:
            record_failure(error)
            raise
        try:
            emit('stop', completed=report['completed'], stop_reason=report['stop_reason'],
                 coverage_ms=report['coverage_ms'], rows=report['rows'], status=report['status'],
                 result_file=str(output / 'result.json'))
        except BaseException as error:
            record_failure(error)
            raise
    _write(output / 'recipe.json', recipe)
    report.update(source_revision=revision, checkpoint_sha256=config.checkpoint_sha256, profile=profile,
                  recipe_file=str(output / 'recipe.json'), events_file=str(events_file),
                  events_sha256=digest(events_file), seconds=time.perf_counter()-started)
    _write(output / 'result.json', report)
    return dict(status=report['status'], completed=report['completed'], stop_reason=report['stop_reason'],
        result_file=str(output / 'result.json'), events_file=str(events_file),
        osu_file=report.get('osu_file'), audio_file=report.get('audio_file'),
        rows=report['rows'], profile=profile, seconds=report['seconds'])
