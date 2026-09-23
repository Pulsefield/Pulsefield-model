"""Bounded paired pilots with separate calibration and assessment song groups."""
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import psutil
import torch
from torch.nn import functional as F

from .corpus import digest, revision, write_json
from .data import HALO, WINDOW, load_charts, match_events, pick_events, window
from .model import SkeletonModel


def save_checkpoint(path, model, optimizer, update, config, source_revision, rng, manifest_sha):
    temporary = path.with_suffix('.tmp')
    torch.save(dict(format='audio-skeleton/model-v1', model=model.state_dict(), optimizer=optimizer.state_dict(),
        update=update, config=asdict(config), source_revision=source_revision,
        manifest_sha256=manifest_sha, rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
        mps_rng=torch.mps.get_rng_state() if config.device == 'mps' else None), temporary)
    temporary.replace(path)


def loss_terms(logits, offsets, labels, targets, mask):
    weights = logits.new_tensor([12., 48., 24., 96.])
    bce = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weights, reduction='none')
    event = (bce * mask[..., None]).sum() / (mask.sum() * 4).clamp_min(1)
    positive = labels * mask[..., None]
    offset = (F.smooth_l1_loss(offsets, targets, reduction='none') * positive).sum() / positive.sum().clamp_min(1)
    return event + offset, event, offset


@torch.inference_mode()
def predict(model, chart, device, controls=None):
    model.eval()
    outputs = np.empty((len(chart['mel']), 8), dtype=np.float32)
    stride = WINDOW - 2 * HALO
    for first in range(0, len(outputs), stride):
        values = window(chart, first - HALO, controls=controls, beat_width=model.beat_width)
        x, b, c = [torch.from_numpy(v[None]).to(device) for v in values[:3]]
        logits, offsets = model(x, b, c)
        count = min(stride, len(outputs) - first)
        result = torch.cat((logits.sigmoid(), offsets), -1)[0, HALO:HALO + count].cpu().numpy()
        outputs[first:first + count] = result
    return outputs


def score_predictions(charts, predictions, thresholds):
    records = []
    for chart, output in zip(charts, predictions):
        roles = []
        for role in range(2):
            sl = slice(role * 2, role * 2 + 2)
            events = pick_events(output[:, sl], output[:, 4:][:, sl], thresholds[role])
            roles.append({str(tol): match_events(chart['times'][role], events, tol) for tol in (10, 20, 40, 70)})
        records.append(dict(source_sha256=chart['entry']['source_sha256'], roles=roles))
    means, empty = {}, {}
    for k, name in enumerate(('head', 'release_only')):
        positive = [r for r in records if r['roles'][k]['20']['reference'] > 0]
        negative = [r for r in records if r['roles'][k]['20']['reference'] == 0]
        means[name] = {str(t): float(np.mean([r['roles'][k][str(t)]['f1'] for r in positive]))
                      if positive else None for t in (10, 20, 40, 70)}
        empty[name] = dict(charts=len(negative), predicted_events=sum(r['roles'][k]['20']['predicted'] for r in negative))
    return dict(macro_f1=means, empty_reference=empty, charts=records)


def calibrate(charts, predictions):
    thresholds = []
    for role in range(2):
        scores = []
        for threshold in (.1, .2, .3, .4, .5, .6, .7, .8, .9):
            sl = slice(role * 2, role * 2 + 2)
            values = [match_events(c['times'][role], pick_events(p[:, sl], p[:, 4:][:, sl], threshold), 20)['f1']
                      for c, p in zip(charts, predictions) if len(c['times'][role])]
            scores.append((float(np.mean(values)) if values else 0., threshold))
        thresholds.append(max(scores)[1])
    return thresholds, score_predictions(charts, predictions, thresholds)


def run(config, *, resolved_yaml=''):
    if config.mode == 'evaluate':
        return evaluate(config)
    source_revision = revision()
    directory = Path(config.root) / 'training' / config.run_name
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'resolved.yaml').write_text(resolved_yaml)
    manifest_sha = digest(Path(config.root) / 'manifest.json')
    feature_sha = digest(Path(config.root) / 'features/index.json')
    torch.set_num_threads(2)
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    charts = load_charts(config)
    train = [c for c in charts if c['entry']['split'] == 'train']
    validation = [c for c in charts if c['entry']['split'] == 'validation']
    # Selection orders each two-chart validation stratum together. One group is
    # calibration; the other is read only after checkpoint selection completes.
    calibration, assessment = validation[::2], validation[1::2]
    if config.overfit_charts:
        selected = np.linspace(0, len(train) - 1, min(len(train), config.overfit_charts)).astype(int)
        train = [train[i] for i in selected]
        calibration = train
    beat_width = next((c['beats'].shape[1] for c in charts if c['beats'] is not None), 514)
    model = SkeletonModel(use_beat_features=config.use_beat_features, beat_width=beat_width).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=.01)
    initial_update = 0
    if config.resume_from:
        state = torch.load(config.resume_from, map_location=config.device, weights_only=True)
        if state['manifest_sha256'] != manifest_sha:
            raise ValueError('Resume corpus differs')
        for key in ('use_beat_features', 'beat_model', 'batch_size', 'seed', 'overfit_charts', 'learning_rate'):
            if state['config'][key] != getattr(config, key):
                raise ValueError(f'Resume changed {key}')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        initial_update = state['update']
        rng.bit_generator.state = state['rng']
        torch.set_rng_state(state['torch_rng'].cpu())
        if config.device == 'mps':
            torch.mps.set_rng_state(state['mps_rng'].cpu())
    default_controls = np.median(np.stack([c['controls'] for c in train]), axis=0)
    write_json(directory / 'freeze.json', dict(source_revision=source_revision, config=asdict(config),
        manifest_sha256=manifest_sha, features_sha256=feature_sha,
        parameters=sum(p.numel() for p in model.parameters()), default_controls=default_controls.tolist(),
        conditioning='log1p chart-wide H and release-only rows per audio second; supplied proxy, not inferred difficulty',
        calibration=[c['entry']['source_sha256'] for c in calibration],
        assessment=[c['entry']['source_sha256'] for c in assessment]))
    started, best, best_thresholds = time.perf_counter(), -1., [.5, .5]
    losses, last_update = [], initial_update
    with (directory / 'updates.jsonl').open('w') as log:
        for update in range(initial_update + 1, config.updates + 1):
            if time.perf_counter() - started > config.max_seconds or (Path(config.root) / 'PAUSE').exists():
                break
            if update % 20 == 0 and psutil.virtual_memory().available < 2 * 1024 ** 3:
                break
            model.train()
            batch = []
            for _ in range(config.batch_size):
                chart = train[int(rng.integers(len(train)))]
                center = int(rng.integers(len(chart['mel'])))
                batch.append(window(chart, center - WINDOW // 2, beat_width=beat_width))
            tensors = [torch.from_numpy(np.stack([b[i] for b in batch])).to(config.device) for i in range(6)]
            optimizer.zero_grad(set_to_none=True)
            logits, offsets = model(*tensors[:3])
            loss, event, offset = loss_terms(logits, offsets, *tensors[3:])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite skeleton training objective')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            last_update = update
            losses.append(float(loss.detach().cpu()))
            if update % 20 == 0:
                record = dict(update=update, seconds=time.perf_counter() - started,
                    loss=float(np.mean(losses[-20:])), event_loss=float(event.detach().cpu()),
                    offset_loss=float(offset.detach().cpu()), rss_bytes=psutil.Process().memory_info().rss,
                    mps_driver_bytes=torch.mps.driver_allocated_memory() if config.device == 'mps' else 0)
                log.write(json.dumps(record) + '\n'); log.flush()
                print(json.dumps(record), flush=True)
            if update % config.validation_every == 0 or update == config.updates:
                predictions = [predict(model, c, config.device) for c in calibration]
                thresholds, metrics = calibrate(calibration, predictions)
                score = np.mean([metrics['macro_f1'][role]['20'] for role in ('head', 'release_only')
                                 if metrics['macro_f1'][role]['20'] is not None])
                write_json(directory / f'calibration-{update}.json', dict(update=update, thresholds=thresholds, **metrics))
                print(json.dumps(dict(update=update, calibration=metrics['macro_f1'], thresholds=thresholds)), flush=True)
                if score > best:
                    best, best_thresholds = float(score), thresholds
                    save_checkpoint(directory / 'best.pt', model, optimizer, update, config, source_revision, rng, manifest_sha)
                    write_json(directory / 'best.json', dict(update=update, score=best, thresholds=thresholds))
                save_checkpoint(directory / 'last.pt', model, optimizer, update, config, source_revision, rng, manifest_sha)
    save_checkpoint(directory / 'last.pt', model, optimizer, last_update, config, source_revision, rng, manifest_sha)
    if not (directory / 'best.pt').exists():
        return dict(status='paused_before_validation', updates=last_update, directory=str(directory))
    state = torch.load(directory / 'best.pt', map_location=config.device, weights_only=True)
    model.load_state_dict(state['model'])
    records = {}
    if not config.overfit_charts:
        for name, controls in (('supplied_density', None), ('train_default_density', default_controls)):
            predictions = [predict(model, c, config.device, controls=controls) for c in assessment]
            records[name] = score_predictions(assessment, predictions, best_thresholds)
            for chart, values in zip(assessment, predictions):
                np.save(directory / f"{chart['entry']['source_sha256']}-{name}.npy", values)
    result = dict(status='completed' if last_update == config.updates else 'bounded_stop', updates=last_update,
                  seconds=time.perf_counter() - started, best_update=state['update'],
                  best_calibration_score=best, thresholds=best_thresholds, assessment=records,
                  checkpoint_sha256=digest(directory / 'best.pt'), source_revision=source_revision)
    write_json(directory / 'result.json', result)
    return dict(status=result['status'], updates=last_update, seconds=result['seconds'],
                result_file=str(directory / 'result.json'), assessment={k:v['macro_f1'] for k,v in records.items()})


def evaluate(config):
    raise ValueError('Use the training run assessment outputs; downstream R1 evaluation is a separate experiment')
