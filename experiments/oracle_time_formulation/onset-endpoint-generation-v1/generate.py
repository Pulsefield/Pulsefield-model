"""Exploratory onset-required continuation with generated endpoint obligations."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path('artifacts/oracle-time-continuation/m3-20260917')
OWNER=ROOT/'onset-endpoint-generation-v1'
HEAD_OWNER=ROOT/'linked-endpoint-head-v1'
sys.path.insert(0,str(HEAD_OWNER/'scripts'))
from core import EndpointHead, candidate_features, context_features, linked_rows
from endpoint_feasibility import conditional_branch, requires_early_release
from schedule import Schedule
from ensomi_model.research.oracle_time_continuation.config import BackboneConfig
from ensomi_model.research.oracle_time_continuation.corpus import admit_entry, catalog_entries, read_split
from ensomi_model.research.oracle_time_continuation.decoding import DecodeSamplingPolicy, sample_row
from ensomi_model.research.oracle_time_continuation.engine import ContinuationEngine, PredictionInput
from ensomi_model.research.oracle_time_continuation.export import export_osu, presentation_header
from ensomi_model.research.oracle_time_continuation.model import CausalBackbone, JointRowDistribution, PreRowEncoding
from ensomi_model.research.oracle_time_continuation.quality import ChartMetrics, source_metrics
from ensomi_model.research.oracle_time_continuation.replay import commit
from ensomi_model.research.oracle_time_continuation.runtime import ResourceConfig, ResourceGuard
from ensomi_model.research.oracle_time_continuation.schema import CompleteRow
from ensomi_model.research.oracle_time_continuation.storage import ROW_DTYPE, SourceStore, file_digest
from ensomi_model.research.scoped_style_modeling.replay import parse_source

WEIGHTS=ROOT/'clock-readout-v1/clock-u100/weights.pt'
WEIGHTS_SHA='29a2a29e8dd901b72447f24884336adf78349264bd7d61846976d7e4f6856641'
ENDPOINT_SHA={'prior':'31ce4e2821ebd42aaf3edd7df029fd3782626e7faa639580828c2065a5ebd128',
              'context':'7a5d8c391b30ce60eea15faf57a8f7d53ed9ae59a90128d83261ddd85f7a2297'}


def save(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def main():
    arm=sys.argv[1];assert arm in ENDPOINT_SHA
    out=ROOT/f'quality-onset-endpoint-{arm}-v1';out.mkdir()
    started=time.perf_counter();torch.set_num_threads(1)
    runtime=ROOT/'skeleton-time-runtime-v8'
    assert file_digest(runtime/'sha256.json')=='09469489b1d5b0d3f9e92890fa0ae6f58c3a9ecc2bf78cad977a865a3d16790e'
    for name,sha in json.loads((runtime/'sha256.json').read_text()).items():assert file_digest(runtime/name)==sha
    from ensomi_model.research.oracle_time_continuation import model as module
    assert Path(module.__file__).resolve().is_relative_to(runtime.resolve())
    assert file_digest(WEIGHTS)==WEIGHTS_SHA
    endpoint_path=HEAD_OWNER/'fit-u1600'/f'{arm}.pt'
    assert file_digest(endpoint_path)==ENDPOINT_SHA[arm]
    payload=torch.load(WEIGHTS,map_location='cpu',weights_only=True)
    model=CausalBackbone(BackboneConfig(**payload['model_config'])).eval();model.load_state_dict(payload['model_state_dict']);del payload
    payload=torch.load(endpoint_path,map_location='cpu',weights_only=True)
    endpoint=EndpointHead(payload['context_dim']).eval();endpoint.load_state_dict(payload['model']);del payload
    signature=model.cache_signature()
    split_sha='15175f45e91cf7299a9a30166731bf38ee7361399b346fb692cf68e76de5992a'
    catalog=Path('artifacts/oracle-time-review/20260915-adfb1ee/catalog.json')
    entries=catalog_entries(catalog,'e31b7e8f4daa044503ef2b8411bc41727ba371eec804c9462a4608be8112ad28',
        read_split('artifacts/scoped-style-modeling/prepare-v1/split-manifest.json',split_sha),split='validation')
    entries=[e for e in entries if e['source_sha256'].startswith(('85058a','871955','ecc496'))];assert len(entries)==3
    store=SourceStore(ROOT/'quality-source-cache')
    config=ResourceConfig(rss_limit_bytes=6*1024**3,min_available_bytes=2*1024**3,output_max_bytes=2*1024**3)
    resources=(out/'resources.jsonl').open('x');guard=ResourceGuard('cpu',config,resources)
    save(out/'manifest.json',dict(task='Required onsets and optional endpoint candidates; prefix LN objects include full endpoints',
        sources=[e['source_sha256'] for e in entries],arm=arm,weights_sha256=WEIGHTS_SHA,endpoint_sha256=ENDPOINT_SHA[arm],
        row_seed=17,endpoint_seed=100020,row_temperature=.85,endpoint_temperature=1.,top_p=1.,
        runtime_sha256=file_digest(runtime/'sha256.json'),code_sha256={str(p):file_digest(p) for p in
        [Path(__file__),OWNER/'schedule.py',HEAD_OWNER/'scripts/core.py',HEAD_OWNER/'scripts/endpoint_feasibility.py']}))
    reports=[]
    with torch.no_grad(),(out/'completed-reports.jsonl').open('x') as completed:
        for entry in entries:
            sha=entry['source_sha256'];source=admit_entry(store,entry)
            raw=np.fromfile(source.directory/'rows.bin',dtype=ROW_DTYPE)
            times=tuple(float(t) for t in raw['time']);onsets,end_labels,_=linked_rows(raw)
            seed_rows=source.seed.seed_row_count
            seed_objects={i:{int(c):int(end_labels[i,c]) for c in np.flatnonzero(raw['actions'][i]==2)} for i in range(seed_rows)}
            del end_labels
            schedule=Schedule(tuple(bool(x) for x in onsets))
            folder=out/sha[:12]/'p1.0-s17';folder.mkdir(parents=True)
            save(folder/'seed-object-plans.json',dict(seed_rows=seed_rows,plans=seed_objects))
            engine=ContinuationEngine(model,parallel_frontiers=True)
            neural=engine.prefill(source.skeleton,source.targets[:seed_rows],inference=True)
            replay,local,relation,temporal=neural.execution.replay,neural.local,neural.relation,neural.temporal
            row_rng=torch.Generator().manual_seed(17);end_rng=torch.Generator().manual_seed(100020)
            row_path=folder/'rows.jsonl';material_times=[];metrics=ChartMetrics();omitted=0;sampled_heads=0;ln_heads=0;conditioned_rows=0
            with row_path.open('x') as journal,(folder/'endpoint-decisions.jsonl').open('x') as endpoint_journal:
                def record(i,actions,kind,sampling=None):
                    value=dict(event_id=len(material_times),candidate_index=i,time_ms=times[i],actions=actions,decision_kind=kind,sampling=sampling)
                    journal.write(json.dumps(value)+'\n');material_times.append(times[i]);metrics.consume(times[i],actions,sampling)
                for i in range(seed_rows):
                    actions=tuple(int(a) for a in raw['actions'][i]);schedule,actual=schedule.commit(actions,seed_objects[i]);assert actual==actions
                    record(i,actions,'seed')
                while not schedule.finished:
                    i=schedule.index;t=times[i]
                    assert tuple(e is not None for e in schedule.ends)==replay.occupancy
                    query=PredictionInput(t,i==len(times)-1,replay,tuple(times[j]-t for j in range(i+1,min(len(times),i+17))))
                    sampling=None;selected={}
                    if onsets[i]:
                        frontier=model.query_input(query,local,relation);hidden=model.temporal.query(frontier,temporal,t)
                        logp,legal=model.head(PreRowEncoding(hidden[None],(query,)))
                        allowed=torch.tensor([schedule.row_possible(tuple(a)) for a in model.head.rows.tolist()]) & legal[0]
                        assert bool(allowed.any()), 'No feasible required-onset row'
                        distribution=JointRowDistribution(logp[0].masked_fill(~allowed,-torch.inf).log_softmax(-1),allowed)
                        row,sampling=sample_row(distribution,t,DecodeSamplingPolicy(temperature=.85),row_rng)
                        actions=row.actions;lanes=[c for c,a in enumerate(actions) if a==2]
                        if lanes:
                            plans=tuple(None if e is None else times[e] for e in schedule.ends)
                            contexts=torch.stack([context_features(hidden,query,actions,plans,c) for c in lanes])
                            candidates=candidate_features(times,onsets,i).unsqueeze(0).expand(len(lanes),-1,-1)
                            duration_logp=endpoint(contexts,candidates,torch.ones(candidates.shape[:2],dtype=torch.bool),use_context=arm=='context').double().log_softmax(-1)
                            remaining=tuple(None if e==i else e for e in schedule.ends)
                            must_early=requires_early_release(remaining,set(lanes),schedule.next_onset)
                            early=torch.tensor([schedule.next_onset is not None and j<schedule.next_onset for j in range(i+1,len(times))])
                            satisfied=not must_early;conditioned_rows+=int(must_early)
                            for position,c in enumerate(lanes):
                                branch=conditional_branch(duration_logp,early,position,satisfied)
                                offset=int(torch.multinomial(branch.exp(),1,generator=end_rng))
                                selected[c]=i+1+offset;satisfied=satisfied or bool(early[offset])
                                endpoint_journal.write(json.dumps(dict(candidate_index=i,lane=c,end_index=selected[c],end_ms=times[selected[c]],
                                    raw_log_probability=float(duration_logp[position,offset]),conditional_log_probability=float(branch[offset]),joint_constraint=must_early))+'\n')
                            assert satisfied
                        sampled_heads+=sum(a in (1,2) for a in actions);ln_heads+=len(lanes);kind='head'
                    else:
                        actions=schedule.forced_actions();kind='release'
                    schedule,actual=schedule.commit(actions,selected)
                    if actual is None:
                        omitted+=1
                    else:
                        row=CompleteRow(t,actual)
                        following=commit(replay,row,is_terminal=i==len(times)-1)
                        content,local,relation=model.content_input(query,row,following,local,relation)
                        temporal=model.temporal.commit(content,temporal,t,inference=True)
                        replay=following;record(i,actual,kind,sampling)
                    if i%64==0:
                        journal.flush();endpoint_journal.flush();guard.check('generate',arm=arm,source=sha[:12],candidate=i)
                        assert model.cache_signature()==signature and temporal.row_count==replay.row_count
                        assert time.perf_counter()-started<1800
                journal.flush()
            assert not any(replay.occupancy) and not any(e is not None for e in schedule.ends)
            output=folder/'generated.osu';export=export_osu(row_path,output,material_times,config,header=presentation_header(Path(entry['path'])))
            data=output.read_bytes();generated=parse_source(data,file_digest(output))
            assert {n.start_ms for n in generated.objects}=={t for t,yes in zip(times,onsets) if yes}
            assert all(n.end_ms in times for n in generated.objects)
            assert export['rows']==len(material_times) and export['notes']==len(generated.objects)
            report=dict(source=entry,source_metrics=source_metrics(source.targets),seed=17,top_p=1.,
                result=dict(organization=metrics.report(),generated_sha256=file_digest(output),rows_sha256=file_digest(row_path),
                    omitted_candidates=omitted,required_onsets=int(onsets.sum()),sampled_heads=sampled_heads,predicted_lns=ln_heads,
                    joint_conditioned_rows=conditioned_rows,export=export,seed_rows=seed_rows,seed_objects_include_endpoints=True))
            reports.append(report);completed.write(json.dumps(report)+'\n');completed.flush()
            print(json.dumps(dict(arm=arm,source=sha[:12],omitted=omitted,predicted_lns=ln_heads,organization=metrics.report())),flush=True)
            guard.check('chart-complete',source=sha[:12])
    save(out/'readout.json',dict(arm=arm,weights_sha256=WEIGHTS_SHA,endpoint_sha256=ENDPOINT_SHA[arm],reports=reports,seconds=time.perf_counter()-started,
        limitation='Changed conditional task and object seed; no whole-chart quality pass.'))
    print(json.dumps(dict(status='complete',arm=arm,seconds=time.perf_counter()-started,readout_sha256=file_digest(out/'readout.json'))),flush=True)


if __name__=='__main__':main()
