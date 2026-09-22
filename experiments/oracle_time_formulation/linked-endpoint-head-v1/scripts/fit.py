"""Paired generic-prior versus linked-history endpoint fitting on frozen features."""
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from core import EndpointHead, collate
from extract import OWNER, OUT as FEATURES, SEED, save
from ensomi_model.research.oracle_time_continuation.runtime import ResourceConfig, ResourceGuard
from ensomi_model.research.oracle_time_continuation.storage import file_digest

OUT = OWNER/'fit'


def evaluate(model,cases,charts,use_context):
    records=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(cases),32):
            batch=cases[start:start+32]
            context,values,valid,target=collate(batch,charts)
            logp=model(context,values,valid,use_context=use_context)
            prediction=logp.argmax(-1)
            nll=-logp[torch.arange(len(batch)),target]
            for case,pred,loss in zip(batch,prediction.tolist(),nll.tolist()):
                chart=charts[case['source']]
                predicted=case['index']+1+pred
                records.append(dict(source=case['source'],group=case['group'],index=case['index'],lane=case['lane'],
                    end_index=case['end_index'],predicted_end_index=predicted,nll=loss,
                    correct=predicted==case['end_index'],error_ms=abs(float(chart['times'][predicted]-chart['times'][case['end_index']]))))
    groups={g:[r for r in records if r['group']==g] for g in sorted({r['group'] for r in records})}
    group_summary={g:dict(nll=float(np.mean([r['nll'] for r in rows])),
                          accuracy=float(np.mean([r['correct'] for r in rows])),
                          mean_error_ms=float(np.mean([r['error_ms'] for r in rows])),cases=len(rows))
                   for g,rows in groups.items()}
    return dict(group_mean_nll=float(np.mean([r['nll'] for r in group_summary.values()])),
                group_mean_accuracy=float(np.mean([r['accuracy'] for r in group_summary.values()])),
                per_head_nll=float(np.mean([r['nll'] for r in records])),
                per_head_accuracy=float(np.mean([r['correct'] for r in records])),
                mean_error_ms=float(np.mean([r['error_ms'] for r in records])),
                median_error_ms=float(np.median([r['error_ms'] for r in records])),
                groups=group_summary,records=records)


def main():
    OUT.mkdir()
    started=time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    resources=(OUT/'resources.jsonl').open('x')
    guard=ResourceGuard('cpu',ResourceConfig(rss_limit_bytes=6*1024**3,min_available_bytes=2*1024**3),resources)
    manifest=json.loads((FEATURES/'manifest.json').read_text())
    assert manifest['status']=='complete'
    assert file_digest(FEATURES/'selection.json')==manifest['selection_sha256']
    charts={}; train=[]; validation=[]
    for item in manifest['files']:
        path=FEATURES/item['file']
        assert file_digest(path)==item['sha256']
        chart=torch.load(path,map_location='cpu',weights_only=True)
        assert item['split']==chart['split'] and item['cases']==len(chart['cases'])
        charts[chart['source']]=chart
        (train if chart['split']=='train' else validation).extend(chart['cases'])
    assert len({c['group'] for c in train})==128 and len({c['group'] for c in validation})==32
    assert not {c['group'] for c in train}&{c['group'] for c in validation}
    generic=EndpointHead(manifest['context_dim'])
    models={'prior':generic,'context':copy.deepcopy(generic)}
    optimizers={name:torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01) for name,model in models.items()}
    rng=random.Random(SEED)
    draws=[[rng.randrange(len(train)) for _ in range(32)] for _ in range(400)]
    save(OUT/'manifest.json',dict(feature_manifest_sha256=file_digest(FEATURES/'manifest.json'),seed=SEED,
         train_cases=len(train),validation_cases=len(validation),updates=400,batch=32,lr=.001,weight_decay=.01,clip=1.,
         parameters=sum(p.numel() for p in generic.parameters()),context_dim=manifest['context_dim'],
         shared_draws=draws,code_sha256={p.name:file_digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}))
    history=[]
    with (OUT/'updates.jsonl').open('x') as stream:
        for step,indices in enumerate(draws,1):
            batch=[train[i] for i in indices]
            context,values,valid,target=collate(batch,charts)
            losses={}
            for arm,model in models.items():
                model.train(); optimizers[arm].zero_grad(set_to_none=True)
                logp=model(context,values,valid,use_context=arm=='context')
                loss=-logp[torch.arange(len(batch)),target].mean()
                assert bool(torch.isfinite(loss))
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                assert bool(torch.isfinite(norm))
                optimizers[arm].step()
                losses[arm]=float(loss.detach())
            record=dict(step=step,loss=losses,seconds=time.perf_counter()-started)
            stream.write(json.dumps(record)+'\n'); stream.flush()
            history.append(record)
            if step%25==0:
                print(json.dumps(record),flush=True)
                guard.check('fit',step=step)
            assert time.perf_counter()-started<1200
    results={}
    for arm,model in models.items():
        results[arm]=evaluate(model,validation,charts,arm=='context')
        save(OUT/f'{arm}-validation.json',results[arm])
        with (OUT/f'{arm}.pt').open('xb') as stream:
            torch.save(dict(model=model.state_dict(),context_dim=manifest['context_dim'],arm=arm,
                            feature_manifest_sha256=file_digest(FEATURES/'manifest.json')),stream)
        guard.check('evaluated',arm=arm)
    groups=sorted(results['prior']['groups'])
    deltas=np.array([results['context']['groups'][g]['nll']-results['prior']['groups'][g]['nll'] for g in groups])
    boot=np.random.default_rng(SEED).choice(deltas,(5000,len(groups)),replace=True).mean(-1)
    interval=np.quantile(boot,[.025,.975]).tolist()
    accuracy_delta=results['context']['group_mean_accuracy']-results['prior']['group_mean_accuracy']
    summary=dict(status='complete',seconds=time.perf_counter()-started,
        metrics={arm:{k:v for k,v in data.items() if k not in ('groups','records')} for arm,data in results.items()},
        group_nll_delta=float(deltas.mean()),group_nll_delta_ci95=interval,groups_improved=int((deltas<0).sum()),
        group_accuracy_delta=accuracy_delta,bootstrap_replicates=5000,bootstrap_seed=SEED,
        criterion_met=bool(deltas.mean()<=-.05 and interval[1]<0 and accuracy_delta>=-.05),
        feature_manifest_sha256=file_digest(FEATURES/'manifest.json'),
        evidence_sha256={p.name:file_digest(p) for p in OUT.iterdir() if p.suffix in ('.pt','.json')},
        limitation='Teacher-forced conditional endpoint feasibility on selected LN-rich charts; not free-generation quality.')
    save(OUT/'readout.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
