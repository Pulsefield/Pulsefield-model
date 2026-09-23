"""Pinned audio features with explicit 10 ms and BeatThis 20 ms clocks."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import psutil
import torch
from torchaudio.functional import melscale_fbanks

from .corpus import digest, read_manifest, revision, write_json

MEL_SPEC = dict(sample_rate=24000, n_fft=1024, hop=240, bins=128,
                center=True, pad='constant', origin_ms=0., power=2,
                mel='htk-slaney', min_hz=30., max_hz=12000., log='log10-floor-1e-10')


def decode(path, rate):
    result = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path),
        '-map', '0:a:0', '-ac', '1', '-ar', str(rate), '-f', 'f32le', 'pipe:1'],
        capture_output=True, check=True)
    wave = np.frombuffer(result.stdout, dtype='<f4').copy()
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError('Audio decode was empty or nonfinite')
    return wave


def mel(wave):
    spectrum = torch.stft(torch.from_numpy(wave), n_fft=1024, hop_length=240,
        window=torch.hann_window(1024), center=True, pad_mode='constant', return_complex=True).abs().square()
    filters = melscale_fbanks(513, 30., 12000., 128, 24000, norm='slaney', mel_scale='htk')
    return (spectrum.T @ filters).clamp_min(1e-10).log10().numpy().astype(np.float16)


def save_array(path, values):
    path = Path(path)
    with path.with_suffix('.tmp').open('wb') as stream:
        np.save(stream, values, allow_pickle=False)
    path.with_suffix('.tmp').replace(path)


def cache(config):
    from beat_this.inference import load_model, split_piece
    from beat_this.preprocessing import LogMelSpect

    source_revision = revision()
    manifest = read_manifest(config.root)
    directory = Path(config.root) / 'features'
    directory.mkdir(exist_ok=True)
    index_file = directory / 'index.json'
    manifest_sha = digest(Path(config.root) / 'manifest.json')
    index = json.loads(index_file.read_text()) if index_file.exists() else dict(
        format='audio-skeleton/features-v1', manifest_sha256=manifest_sha,
        mel_spec=MEL_SPEC, assets={}, runs=[])
    if index['manifest_sha256'] != manifest_sha or index['mel_spec'] != MEL_SPEC:
        raise ValueError('Feature cache schema or manifest changed')
    index['runs'].append(dict(source_revision=source_revision, config=asdict(config)))
    torch.set_num_threads(2)
    model, frontend, captured = None, None, []
    if config.use_beat_features:
        model = load_model(config.beat_model, device=config.device)
        frontend = LogMelSpect(device='cpu')
        checkpoint = Path(torch.hub.get_dir()) / 'checkpoints' / f'beat_this-{config.beat_model}.ckpt'
        model_sha = digest(checkpoint)
        model.task_heads.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    started = time.perf_counter()
    for entry in manifest['charts']:
        if time.perf_counter() - started > config.max_seconds or (Path(config.root) / 'PAUSE').exists():
            write_json(index_file, index)
            return dict(status='paused', completed=len(index['assets']))
        if psutil.virtual_memory().available < 2 * 1024 ** 3 or shutil.disk_usage(directory).free < 40 * 1024 ** 3:
            raise RuntimeError('Feature extraction reached its memory/disk reserve')
        sha = entry['audio_sha256']
        if digest(entry['audio_file']) != sha:
            raise ValueError('Selected audio bytes changed')
        record = index['assets'].setdefault(sha, {})
        now = time.perf_counter()
        if 'mel_file' not in record:
            wave = decode(entry['audio_file'], 24000)
            values = mel(wave)
            path = directory / f'{sha}-mel.npy'
            save_array(path, values)
            record.update(mel_file=str(path.resolve()), mel_sha256=digest(path),
                          frames=len(values), duration_ms=len(wave) / 24,
                          mel_seconds=time.perf_counter() - now)
        elif digest(record['mel_file']) != record['mel_sha256']:
            raise ValueError('Cached mel digest changed')
        if config.use_beat_features and config.beat_model not in record:
            wave = decode(entry['audio_file'], 22050)
            with torch.inference_mode():
                spect = frontend(torch.from_numpy(wave))
                chunks, starts = split_piece(spect, 1500, border_size=6)
                embedding = None
                assigned = np.zeros(len(spect), dtype=bool)
                for chunk, start in zip(chunks, starts):
                    captured.clear()
                    predictions = model(chunk[None].to(config.device))
                    features = torch.cat((captured[0][0], predictions['beat'][0, :, None],
                                          predictions['downbeat'][0, :, None]), -1)[6:-6].cpu().numpy()
                    first, stop = int(start + 6), int(start + 6 + len(features))
                    if embedding is None:
                        embedding = np.empty((len(spect), features.shape[-1]), dtype=np.float16)
                    new = ~assigned[first:stop]
                    embedding[first:stop][new] = features[new]
                    assigned[first:stop] = True
                if not assigned.all() or not np.isfinite(embedding).all():
                    raise ValueError('BeatThis aggregation has missing/nonfinite feature frames')
            path = directory / f'{sha}-{config.beat_model}.npy'
            save_array(path, embedding)
            record[config.beat_model] = dict(file=str(path.resolve()), sha256=digest(path),
                checkpoint_sha256=model_sha, frames=len(embedding), width=embedding.shape[1],
                frame_ms=20., origin_ms=0., seconds=time.perf_counter() - now)
            if config.device == 'mps':
                torch.mps.empty_cache()
        elif config.use_beat_features:
            saved = record[config.beat_model]
            if saved['checkpoint_sha256'] != model_sha or digest(saved['file']) != saved['sha256']:
                raise ValueError('Cached BeatThis feature/model identity changed')
        write_json(index_file, index)
        print(json.dumps(dict(stage='features', completed=len(index['assets']),
                              audio=sha[:12], seconds=time.perf_counter() - now)), flush=True)
    return dict(status='completed', assets=len(index['assets']), seconds=time.perf_counter() - started,
                index_file=str(index_file), sha256=digest(index_file))
