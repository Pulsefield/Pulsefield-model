"""Content-addressed, time-exact log-mel shards from streaming FFmpeg decode.

Frames start on a single global 240-sample grid. Chunk boundaries retain the
784 overlapping samples, so a chunk never creates padding or a second clock.
Resume verifies published shards and decodes from the beginning before skipping
their samples; codec seek approximations cannot shift the remaining frame grid.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import torch
from torchaudio.functional import melscale_fbanks

from ..oracle_time_continuation.publication import staging_directory
from ..oracle_time_continuation.runtime import sync_directory
from ..oracle_time_continuation.storage import file_digest
from ..scoped_style_modeling.dataset import ContractError
from .control import publish_json, read_json, tree_bytes

SPEC = dict(format='vacation/log-mel-v1', sample_rate=24000, hop_samples=240,
            fft_samples=1024, window='hann-periodic', center=False, mel_bins=128,
            mel_scale='htk', mel_norm='slaney', f_min=30., f_max=12000.,
            power=2, log='log10', floor=1e-10, dtype='float16', frame_origin_samples=512)


def mel_frames(samples):
    wave = torch.from_numpy(np.asarray(samples, dtype=np.float32).copy())
    if len(wave) < 1024 or not bool(torch.isfinite(wave).all()):
        raise ContractError('Mel extraction needs at least 1024 finite samples')
    spectrum = torch.stft(wave, n_fft=1024, hop_length=240, window=torch.hann_window(1024),
                          center=False, return_complex=True).abs().square()
    filters = melscale_fbanks(513, 30., 12000., 128, 24000, norm='slaney', mel_scale='htk')
    value = (spectrum.T @ filters).clamp_min(1e-10).log10().numpy().astype('<f2')
    if not np.isfinite(value).all():
        raise ContractError('Audio features contain nonfinite values')
    return value


def frame_chunks(stream, chunk_frames):
    """Yield (global first frame, samples), retaining overlap across short reads."""
    target = ((chunk_frames - 1) * 240 + 1024) * 4
    pending, first = bytearray(), 0
    while True:
        while len(pending) < target:
            block = stream.read(target - len(pending))
            if not block:
                break
            pending.extend(block)
        if len(pending) % 4:
            raise ContractError('Decoder ended with an incomplete float32 sample')
        frames = max(0, (len(pending) // 4 - 1024) // 240 + 1)
        if not frames:
            return
        count = (frames - 1) * 240 + 1024
        yield first, np.frombuffer(bytes(pending[:count * 4]), dtype='<f4')
        first += frames
        del pending[:frames * 240 * 4]
        if frames < chunk_frames:
            return


def validate_manifest(value):
    if value.get('format') != 'vacation/audio-inputs-v1' or not isinstance(value.get('assets'), list):
        raise ContractError('Audio inputs require vacation/audio-inputs-v1 assets')
    seen = set()
    for asset in value['assets']:
        sha = asset['sha256']
        if (len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha) or sha in seen or
                not asset.get('path') or not asset.get('sources')):
            raise ContractError('Audio assets need unique content hashes, paths and source associations')
        seen.add(sha)
        for source in asset['sources']:
            if source['split'] not in ('train', 'validation', 'test') or not source['group_id']:
                raise ContractError('Audio source associations require known split and song group')
            if not 0 <= source['first_ms'] <= source['last_ms']:
                raise ContractError('Audio association has an invalid chart time range')
    return value


def probe(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries',
        'stream=sample_rate,channels,duration:format=duration', '-of', 'json', str(path)],
        capture_output=True, text=True, timeout=30, check=True)
    try:
        value = json.loads(result.stdout)
        stream = value['streams'][0]
        duration = float(stream.get('duration') or value['format']['duration'])
        rate, channels = int(stream['sample_rate']), int(stream['channels'])
    except (ValueError, KeyError, IndexError) as error:
        raise ContractError('Audio has no usable container duration/rate/channel metadata') from error
    if not np.isfinite(duration) or duration <= 0:
        raise ContractError('Audio duration must be finite and positive')
    return dict(duration_seconds=duration, source_sample_rate=rate, channels=channels)


def cache_asset(asset, directory, config, control):
    directory.mkdir(parents=True, exist_ok=True)
    path = Path(asset['path'])
    if file_digest(path, 1024**3) != asset['sha256']:
        raise ContractError('Audio content changed after its input manifest was frozen')
    metadata_file = directory / 'index.json'
    metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else dict(
        audio_sha256=asset['sha256'], spec=SPEC, shards=[], complete=False, probe=probe(path))
    if metadata['audio_sha256'] != asset['sha256'] or metadata['spec'] != SPEC:
        raise ContractError('Cached audio identity or feature schema changed')
    expected = 0
    for shard in metadata['shards']:
        if shard['first_frame'] != expected or Path(shard['file']).name != shard['file']:
            raise ContractError('Audio shards have a gap, overlap or invalid path')
        if file_digest(directory / shard['file']) != shard['sha256']:
            raise ContractError('Published audio shard digest differs')
        expected += shard['frames']
    if metadata['complete']:
        return metadata, None
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path), '-map', '0:a:0',
            '-ac', '1', '-ar', '24000', '-f', 'f32le', 'pipe:1'], stdout=subprocess.PIPE, stderr=errors)
        class CountedReader:
            size = 0
            def read(self, count):
                block = process.stdout.read(count)
                self.size += len(block)
                return block
        decoded = CountedReader()
        try:
            for first, samples in frame_chunks(decoded, config.chunk_frames):
                if reason := control.boundary():
                    return metadata, reason
                count = (len(samples) - 1024) // 240 + 1
                if first < expected:
                    if first + count > expected:
                        raise ContractError('Resume chunk size differs from the published audio prefix')
                    continue
                values = mel_frames(samples)
                destination = directory / f'frames-{first:09d}.npy'
                with staging_directory(destination) as stage:
                    control.storage(values.nbytes + 4096)
                    if tree_bytes(directory.parent) + values.nbytes + 4096 > config.max_bytes:
                        return metadata, 'audio_byte_limit'
                    temporary = stage / 'frames.npy'
                    with temporary.open('wb') as output:
                        np.save(output, values, allow_pickle=False)
                        output.flush()
                        os.fsync(output.fileno())
                    os.replace(temporary, destination)
                    sync_directory(directory)
                metadata['shards'].append(dict(file=destination.name, first_frame=first, frames=len(values),
                                               sha256=file_digest(destination), bytes=destination.stat().st_size))
                publish_json(metadata_file, metadata)
                print(json.dumps(dict(stage='audio', sha256=asset['sha256'], frames=first + len(values))), flush=True)
            if process.wait(timeout=30):
                errors.seek(0)
                raise ContractError('FFmpeg decode failed: ' + errors.read(4096).decode(errors='replace'))
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    if file_digest(path) != asset['sha256'] or not metadata['shards']:
        raise ContractError('Audio changed during decoding or yielded no complete feature frame')
    metadata['complete'] = True
    metadata['frames'] = sum(s['frames'] for s in metadata['shards'])
    metadata['decoded_samples'] = decoded.size // 4
    metadata['decoded_duration_ms'] = 1000 * metadata['decoded_samples'] / 24000
    metadata['unframed_tail_samples'] = metadata['decoded_samples'] - ((metadata['frames'] - 1) * 240 + 1024)
    metadata['container_duration_difference_ms'] = metadata['decoded_duration_ms'] - 1000 * metadata['probe']['duration_seconds']
    metadata['time_axis_issues'] = [dict(source_sha256=s['source_sha256'], reason='chart_after_audio',
        chart_last_ms=s['last_ms'], audio_duration_ms=metadata['decoded_duration_ms'])
        for s in asset['sources'] if s['last_ms'] > metadata['decoded_duration_ms']]
    publish_json(metadata_file, metadata)
    return metadata, None


def run_audio(config, output, control):
    manifest = validate_manifest(read_json(config.manifest_file, config.manifest_sha256))
    output.mkdir(parents=True, exist_ok=True)
    report_file = output / 'summary.json'
    report = json.loads(report_file.read_text()) if report_file.exists() else dict(
        spec=SPEC, manifest_sha256=config.manifest_sha256, assets={}, input_issues=manifest.get('issues', []),
        music_encoder='not_selected')
    if report['manifest_sha256'] != config.manifest_sha256:
        raise ContractError('Audio input manifest differs from its receipt')
    publish_json(report_file, report)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for asset in manifest['assets']:
            sha = asset['sha256']
            if reason := control.boundary():
                return dict(status='paused', reason=reason)
            if ({s['split'] for s in asset['sources']} != {'train'} or
                    set(asset.get('known_directory_splits', ['train'])) != {'train'}):
                report['assets'][sha] = dict(status='quarantined', reason='non_train_or_cross_split', sources=asset['sources'])
            elif report['assets'].get(sha, {}).get('status') == 'failed':
                continue  # A failure is retained; resume is not an automatic retry policy.
            else:
                try:
                    metadata, reason = cache_asset(asset, output / sha, config, control)
                    report['assets'][sha] = dict(status='completed' if metadata['complete'] else 'partial',
                        index_file=str(output / sha / 'index.json'), sources=asset['sources'])
                    if reason:
                        publish_json(report_file, report)
                        return dict(status='paused', reason=reason)
                except (ContractError, subprocess.SubprocessError, OSError) as error:
                    # Storage pressure is a queue stop, not a corrupt-audio classification.
                    from ..oracle_time_continuation.runtime import ResourceLimit
                    if isinstance(error, ResourceLimit):
                        raise
                    report['assets'][sha] = dict(status='failed', error=str(error), sources=asset['sources'])
            publish_json(report_file, report)
    finally:
        torch.set_num_threads(previous)
    counts = {name: sum(a['status'] == name for a in report['assets'].values())
              for name in ('completed', 'failed', 'quarantined')}
    return dict(status='completed' if not counts['failed'] and not counts['quarantined'] and not report['input_issues']
                else 'completed_with_anomalies',
                **counts, summary_sha256=file_digest(report_file))
