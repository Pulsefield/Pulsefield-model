"""Exact finite-state expectations over shared source-clock interval encodings."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch

from .context_model import ContextAudioModel
from .context_training import freeze_protocol, example_from_identity, song_context
from .data import _json, digest, load_corpus
from .generation import load_model, _resource_stop
from .intent_model import IntentAudioModel, IntentModelConfig, arrangement_summary
from .intervals import collate_interval, encode_interval, score_encoded_interval, interval_losses
from .training import revision

INTENT_MODULES = ('intent.', 'intent_prior.', 'intent_posterior.')
INHERITED_MODULES = ('temporal.', 'exact.', 'fuse.', 'joint.', 'route_residual.', 'release_residual.')


def common_identity(model):
    value = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if not name.startswith(INTENT_MODULES):
            value.update(name.encode()); value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def initialize(config):
    """Warm-start identical shared weights; K=1 has no extra decoder parameters."""
    base, metadata = load_model(config.initial_checkpoint_file, config.initial_checkpoint_sha256)
    if type(base) is not ContextAudioModel or not base.config.global_audio or not base.config.bounded_timing:
        raise ValueError('Intent comparison starts from the pinned global/bounded context model')
    identity = common_identity(base)
    if config.states == 1:
        return base, identity
    model = IntentAudioModel(IntentModelConfig(**asdict(base.config)))
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    if unexpected or any(not name.startswith(INTENT_MODULES) for name in missing):
        raise ValueError('Intent initialization did not preserve every shared tensor')
    if common_identity(model) != identity:
        raise ValueError('Common initial weights differ across intent arms')
    return model, identity


def state_terms(model, coarse, summary):
    if isinstance(model, IntentAudioModel):
        log_q = model.posterior_log_probs(summary[None])[0]
        log_p = model.prior_log_probs(coarse)[0]
        q = log_q.exp()
        return q, (q * (log_q - log_p)).sum(), log_q, log_p
    like = coarse[0]
    zero = like.new_zeros(1)
    return like.new_ones(1), zero[0], zero, zero


def component_losses(model, batch, coarse):
    """Return [states, timing/row/joint] NLL, sharing all backbone computation."""
    encoded, history = encode_interval(model, batch.inputs, coarse)
    states = model.config.intent_states if isinstance(model, IntentAudioModel) else 1
    return torch.stack([torch.stack(interval_losses(score_encoded_interval(model, batch.inputs,
        model.apply_code(encoded, code) if states > 1 else encoded, history), batch)) for code in range(states)])


def objective(components, q, kl, example):
    """Unbiased negative whole-chart ELBO per second for a sampled interval."""
    reconstruction = (q[:, None] * components).sum(0) * example.weight_per_second
    kl_per_second = kl * (1000 / (example.chart.duration_ms + 1))
    return reconstruction[-1] + kl_per_second, reconstruction, kl_per_second


def backward_update(model, songs, charts, summaries, device):
    totals = np.zeros(5, np.float64)
    counts = dict(intervals=0, milliseconds=0, event_rows=0, heads=0, timing_bins=0)
    for records in songs:
        examples = [example_from_identity(r, charts) for r in records]
        if len({e.chart.entry['audio_sha256'] for e in examples}) != 1:
            raise ValueError('Intent microbatch must share exact full audio')
        coarse = song_context(model, examples[0].chart, device)
        loss = None
        for ex in examples:
            batch = collate_interval(ex, model.config, device)
            q, kl, _, _ = state_terms(model, coarse, summaries[ex.chart.entry['source_sha256']])
            total, recon, kl_s = objective(component_losses(model, batch, coarse), q, kl, ex)
            scaled = total / (len(songs) * len(examples))
            if not bool(torch.isfinite(scaled)):
                raise RuntimeError('Nonfinite intent ELBO')
            loss = scaled if loss is None else loss + scaled
            totals += np.asarray([float(v.detach().cpu()) for v in
                (total, recon[0], recon[1], kl_s, -(q*q.clamp_min(1e-30).log()).sum())]) / (len(songs)*len(examples))
            counts['intervals'] += 1
            counts['milliseconds'] += ex.end_ms-ex.start_ms
            counts['event_rows'] += len(batch.targets.row_index)
            counts['timing_bins'] += int(batch.inputs.timing_valid.any(-1).sum())
            rows = ex.chart.source.rows
            counts['heads'] += int(np.isin(rows['actions'][(rows['time']>=ex.start_ms)&(rows['time']<ex.end_ms)],(1,2)).sum())
        loss.backward()
    return totals, counts


@torch.inference_mode()
def evaluate(model, records, charts, summaries, device):
    """Fixed source intervals; posterior reconstruction is not native quality.

    Population weighting estimates the negative whole-chart ELBO per second.
    BOS reports local reconstruction per second plus the separately named
    chart-normalized KL. No independent per-window mixture is formed.
    """
    model.eval(); grouped = {}
    for r in records: grouped.setdefault(r['source_sha256'], []).append(r)
    results, codes = [], []
    for sha, group in grouped.items():
        chart = charts[sha]; coarse = song_context(model, chart, device)
        q, kl, log_q, log_p = state_terms(model, coarse, summaries[sha])
        codes.append(dict(source_sha256=sha, q=q.cpu().tolist(), prior=log_p.exp().cpu().tolist(),
                          kl_nats=float(kl.cpu()), posterior_entropy=float(-(q*log_q).sum().cpu())))
        for r in group:
            ex = example_from_identity(r, charts); batch = collate_interval(ex, model.config, device)
            components = component_losses(model, batch, coarse)
            total,recon,kl_s = objective(components,q,kl,ex)
            weight = ex.weight_per_second if r['panel']=='population' else 1000/(ex.end_ms-ex.start_ms)
            expected = float((q*components[:,-1]).sum().cpu())*weight
            shifted = float((torch.roll(q,1)*components[:,-1]).sum().cpu())*weight
            results.append(dict(**r, event_rows=len(batch.targets.row_index),
                reconstruction_per_second=expected, chart_kl_per_second=float(kl_s.cpu()),
                negative_elbo_per_second=float(total.cpu()) if r['panel']=='population' else None,
                permuted_posterior_reconstruction_per_second=shifted,
                component_joint_nll=components[:,-1].cpu().tolist()))
    population=[r for r in results if r['panel']=='population'];bos=[r for r in results if r['panel']=='bos']
    qs=np.asarray([r['q'] for r in codes]);mean=qs.mean(0)
    return dict(population=dict(intervals=len(population),negative_elbo_per_second=float(np.mean([r['negative_elbo_per_second'] for r in population]))),
        bos=dict(intervals=len(bos),reconstruction_per_second=float(np.mean([r['reconstruction_per_second'] for r in bos]))),
        posterior_information_nats=float(-(mean*np.log(np.maximum(mean,1e-30))).sum()-np.mean([r['posterior_entropy'] for r in codes])),
        codes=codes,records=results)


def save_checkpoint(path, model, optimizer, update, config, source, protocol_identity):
    temporary=path.with_suffix('.tmp')
    torch.save(dict(format='joint-audio/intent-v1' if isinstance(model,IntentAudioModel) else 'joint-audio/context-v1',
        model_config=asdict(model.config),model=model.state_dict(),optimizer=optimizer.state_dict(),update=update,
        config=asdict(config),source_revision=source,manifest_sha256=config.manifest_sha256,
        protocol=protocol_identity,initial_checkpoint_sha256=config.initial_checkpoint_sha256,
        torch_rng=torch.get_rng_state()),temporary)
    temporary.replace(path)


def train(config, *, resolved_yaml=''):
    config.validate(); source=revision(); started=time.perf_counter()
    root=Path(config.root)
    if digest(root/'manifest.json')!=config.manifest_sha256:
        raise ValueError('Intent corpus differs from pinned manifest')
    directory=root/'intent-training'/config.run_name; directory.mkdir(parents=True,exist_ok=False)
    _json(directory/'config.json',asdict(config)); (directory/'resolved.yaml').write_text(resolved_yaml)
    torch.set_num_threads(config.cpu_threads); torch.manual_seed(config.seed)
    cs,_=load_corpus(root); charts={c.entry['source_sha256']:c for c in cs}
    summaries={sha:torch.as_tensor(arrangement_summary(c),device=config.device) for sha,c in charts.items()}
    protocol,identity=freeze_protocol(cs,config)
    model,common=initialize(config); model.to(config.device)
    groups=[dict(params=[],lr=rate) for rate in (config.inherited_learning_rate,config.learning_rate,config.intent_learning_rate)]
    for name,p in model.named_parameters():
        groups[2 if name.startswith(INTENT_MODULES) else 0 if name.startswith(INHERITED_MODULES) else 1]['params'].append(p)
    optimizer=torch.optim.AdamW([g for g in groups if g['params']],weight_decay=config.weight_decay)
    validation=protocol['validation']
    if config.validation_songs:
        selected=sorted({r['source_sha256'] for r in validation})[:config.validation_songs]
        validation=[r for r in validation if r['source_sha256'] in selected]
    _json(directory/'freeze.json',dict(source_revision=source,config=asdict(config),**identity,
        common_initial_sha256=common,parameters=model.parameter_counts(),validation=validation,
        normalization='unchanged initial checkpoint buffers',environment=dict(torch=str(torch.__version__),
        python=__import__('sys').version,ram_bytes=psutil.virtual_memory().total)))
    counts=dict(intervals=0,milliseconds=0,event_rows=0,heads=0,timing_bins=0)
    history=[];last=0;reason='updates_complete';latest=None;eval_update=0
    try:
        with (directory/'updates.jsonl').open('w') as log:
            for update in range(config.updates+1):
                if update:
                    stop=_resource_stop(root)
                    if stop or time.perf_counter()-started>=config.max_seconds:
                        reason=stop or 'wall_clock_limit'; break
                    model.train();optimizer.zero_grad(set_to_none=True)
                    values,consumed=backward_update(model,protocol['updates'][update-1],charts,summaries,config.device)
                    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),config.max_grad_norm,error_if_nonfinite=True)
                    optimizer.step();last=update;history.append(values)
                    for name,value in consumed.items():counts[name]+=value
                    if update%10==0 or update==config.updates:
                        avg=np.mean(history[-10:],axis=0)
                        record=dict(update=update,seconds=time.perf_counter()-started,negative_elbo_per_second=float(avg[0]),
                            timing_reconstruction_per_second=float(avg[1]),row_reconstruction_per_second=float(avg[2]),
                            chart_kl_per_second=float(avg[3]),posterior_entropy=float(avg[4]),grad_norm=float(grad.cpu()),
                            **counts,available_bytes=psutil.virtual_memory().available,rss_bytes=psutil.Process().memory_info().rss,
                            active_mps_bytes=torch.mps.current_allocated_memory() if config.device=='mps' else 0,
                            mps_driver_bytes=torch.mps.driver_allocated_memory() if config.device=='mps' else 0)
                        if isinstance(model,IntentAudioModel):record['code_offset_norm']=float(model.intent.weight.detach().norm().cpu())
                        log.write(json.dumps(record,allow_nan=False)+'\n');log.flush();print(json.dumps(record),flush=True)
                if update%config.validation_every==0 or update==config.updates:
                    latest=evaluate(model,validation,charts,summaries,config.device);eval_update=update
                    _json(directory/f'evaluation-{update}.json',dict(update=update,**latest))
                    print(dict(update=update,population=latest['population'],bos=latest['bos'],posterior_information_nats=latest['posterior_information_nats']),flush=True)
                    save_checkpoint(directory/'last.pt',model,optimizer,update,config,source,identity)
    except BaseException as error:
        _json(directory/'failure.json',dict(update=last,error=repr(error),seconds=time.perf_counter()-started));raise
    save_checkpoint(directory/'last.pt',model,optimizer,last,config,source,identity)
    result=dict(status='completed' if last==config.updates else 'bounded_stop',stop_reason=reason,updates=last,
        seconds=time.perf_counter()-started,**counts,source_revision=source,checkpoint_sha256=digest(directory/'last.pt'),
        **identity,final_evaluation_update=eval_update,population=latest['population'],bos=latest['bos'],
        posterior_information_nats=latest['posterior_information_nats'],selection='fixed endpoint; posterior ELBO is not native quality')
    _json(directory/'result.json',result);return result
