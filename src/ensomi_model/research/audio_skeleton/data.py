"""Chart event targets, coarse controls and aligned audio windows."""
import json
from pathlib import Path

import numpy as np

from ..oracle_time_continuation.storage import ROW_DTYPE
from .corpus import digest, read_manifest

FRAME_MS = 10.
SLOTS = 2
WINDOW = 1600
HALO = 400


def event_targets(times, frames):
    positions = np.rint(np.asarray(times) / FRAME_MS).astype(int)
    if np.any(positions < 0) or np.any(positions >= frames):
        raise ValueError('Chart events extend outside the decoded audio frame clock')
    labels, offsets = np.zeros((frames, SLOTS), np.float32), np.zeros((frames, SLOTS), np.float32)
    counts = np.zeros(frames, dtype=int)
    for t, position in zip(times, positions):
        slot = counts[position]
        if slot == SLOTS:
            raise ValueError('More same-role events in a frame than the declared slot capacity')
        labels[position, slot] = 1
        offsets[position, slot] = t / FRAME_MS - position
        counts[position] += 1
    return labels, offsets


def load_charts(config):
    manifest = read_manifest(config.root)
    index = json.loads((Path(config.root) / 'features/index.json').read_text())
    if index['manifest_sha256'] != digest(Path(config.root) / 'manifest.json'):
        raise ValueError('Audio feature manifest differs from chart selection')
    charts = []
    for entry in manifest['charts']:
        asset = index['assets'][entry['audio_sha256']]
        if digest(entry['rows_file']) != entry['rows_sha256'] or digest(asset['mel_file']) != asset['mel_sha256']:
            raise ValueError('Source rows or audio features changed')
        rows = np.memmap(entry['rows_file'], mode='r', dtype=ROW_DTYPE)
        head = np.isin(rows['actions'], (1, 2)).any(-1)
        times = [rows['time'][head].copy(), rows['time'][~head].copy()]
        mel = np.load(asset['mel_file'], mmap_mode='r')
        targets = [event_targets(t, len(mel)) for t in times]
        controls = np.log1p([len(t) / (asset['duration_ms'] / 1000.) for t in times]).astype(np.float32)
        beats = None
        if config.use_beat_features:
            beat = asset[config.beat_model]
            if digest(beat['file']) != beat['sha256']:
                raise ValueError('BeatThis cached features changed')
            beats = np.load(beat['file'], mmap_mode='r')
        charts.append(dict(entry=entry, mel=mel, beats=beats, times=times, controls=controls,
            labels=np.concatenate([t[0] for t in targets], -1), offsets=np.concatenate([t[1] for t in targets], -1)))
    return charts


def window(chart, start, *, length=WINDOW, controls=None, beat_width=514):
    positions = np.arange(start, start + length)
    valid = (positions >= 0) & (positions < len(chart['mel']))
    safe = np.clip(positions, 0, len(chart['mel']) - 1)
    mel = np.asarray(chart['mel'][safe], dtype=np.float32)
    mel[~valid] = -10.
    beats = np.zeros((length, beat_width), dtype=np.float32)
    if chart['beats'] is not None:
        p = np.clip(positions / 2., 0, len(chart['beats']) - 1)
        lower = np.floor(p).astype(int)
        upper = np.minimum(lower + 1, len(chart['beats']) - 1)
        fraction = (p - lower).astype(np.float32)[:, None]
        beats = chart['beats'][lower].astype(np.float32) * (1 - fraction) + chart['beats'][upper].astype(np.float32) * fraction
        beats[~valid] = 0.
    labels, offsets = chart['labels'][safe].copy(), chart['offsets'][safe].copy()
    mask = valid.copy()
    mask[:HALO] = mask[-HALO:] = False
    return mel, beats, chart['controls'] if controls is None else controls, labels, offsets, mask


def match_events(reference, predicted, tolerance):
    """Maximum-cardinality ordered matching; each timestamp is used once."""
    reference, predicted = np.sort(reference), np.sort(predicted)
    i = j = matched = 0
    while i < len(reference) and j < len(predicted):
        if abs(reference[i] - predicted[j]) <= tolerance:
            matched += 1
            i += 1
            j += 1
        elif predicted[j] < reference[i]:
            j += 1
        else:
            i += 1
    precision = matched / max(1, len(predicted))
    recall = matched / max(1, len(reference))
    total = len(reference) + len(predicted)
    return dict(f1=2 * matched / total if total else 1., precision=precision,
                recall=recall, matches=matched, reference=len(reference), predicted=len(predicted))


def pick_events(probabilities, offsets, threshold):
    p = np.asarray(probabilities)
    if p.ndim == 2:
        # Additional same-frame events require the primary slot's local peak.
        first = p[:, 0]
        peaks = (first >= threshold) & (first > np.r_[-np.inf, first[:-1]]) & (first >= np.r_[first[1:], -np.inf])
        frame, slot = np.nonzero(peaks[:, None] & (p >= threshold))
        return np.unique(np.maximum(0., (frame + offsets[frame, slot]) * FRAME_MS))
    left, right = np.r_[-np.inf, p[:-1]], np.r_[p[1:], -np.inf]
    selected = np.flatnonzero((p >= threshold) & (p > left) & (p >= right))
    return np.maximum(0., (selected + offsets[selected]) * FRAME_MS)
