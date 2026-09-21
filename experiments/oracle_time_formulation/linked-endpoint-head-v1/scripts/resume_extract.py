"""Frozen causal features for a declared linked-endpoint supervision probe."""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from core import context_features, linked_rows, reconstruct, time_tensor
from pulsefield_model.research.oracle_time_continuation.config import BackboneConfig
from pulsefield_model.research.oracle_time_continuation.corpus import admit_entry, catalog_entries, read_split
from pulsefield_model.research.oracle_time_continuation.engine import ContinuationEngine
from pulsefield_model.research.oracle_time_continuation.model import CausalBackbone
from pulsefield_model.research.oracle_time_continuation.runtime import ResourceConfig, ResourceGuard
from pulsefield_model.research.oracle_time_continuation.storage import ROW_DTYPE, SourceStore, file_digest

ROOT = Path('artifacts/oracle-time-continuation/m3-20260917')
OWNER = ROOT / 'linked-endpoint-head-v1'
OUT = OWNER / 'features-rest'
RUNTIME = ROOT / 'skeleton-time-runtime-v8'
SEED = 20260920
WEIGHTS = ROOT / 'clock-readout-v1/clock-u100/weights.pt'
WEIGHTS_SHA = '29a2a29e8dd901b72447f24884336adf78349264bd7d61846976d7e4f6856641'
CATALOG = Path('artifacts/oracle-time-review/20260915-adfb1ee/catalog.json')


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def rank(value):
    return hashlib.sha256(f'{SEED}/{value}'.encode()).hexdigest()


def main():
    OUT.mkdir()
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    assert file_digest(RUNTIME/'sha256.json') == '09469489b1d5b0d3f9e92890fa0ae6f58c3a9ecc2bf78cad977a865a3d16790e'
    for path, sha in json.loads((RUNTIME/'sha256.json').read_text()).items():
        assert file_digest(RUNTIME/path) == sha
    from pulsefield_model.research.oracle_time_continuation import model as model_module
    assert Path(model_module.__file__).resolve().is_relative_to(RUNTIME.resolve())
    assert file_digest(WEIGHTS) == WEIGHTS_SHA
    assert file_digest(CATALOG) == 'e31b7e8f4daa044503ef2b8411bc41727ba371eec804c9462a4608be8112ad28'
    split_sha = '15175f45e91cf7299a9a30166731bf38ee7361399b346fb692cf68e76de5992a'
    assignments = read_split('artifacts/scoped-style-modeling/prepare-v1/split-manifest.json',split_sha)
    entries = [entry for split in ('train','validation') for entry in catalog_entries(
        CATALOG,file_digest(CATALOG),assignments,split=split)]
    store = SourceStore(Path('artifacts/oracle-time-continuation/full-cache-v1'))
    resources = (OUT/'resources.jsonl').open('x')
    guard = ResourceGuard('cpu', ResourceConfig(rss_limit_bytes=6*1024**3,min_available_bytes=2*1024**3),resources)
    resume_path=OWNER/'resume-input.json'
    resume=json.loads(resume_path.read_text())
    original_selection=OWNER/'features/selection.json'
    assert file_digest(original_selection)==resume['selection_sha256']
    selected=json.loads(original_selection.read_text())['sources']
    by_sha={e['source_sha256']:e for e in entries}
    assert all(item['entry']==by_sha[item['entry']['source_sha256']] for item in selected)
    retained={item['source']:item for item in resume['files']}
    assert 0<len(retained)<len(selected)==160
    for item in retained.values():
        path=OWNER/item['path']
        assert file_digest(path)==item['sha256']
        chart=torch.load(path,map_location='cpu',weights_only=True)
        selection=next(x for x in selected if x['entry']['source_sha256']==item['source'])
        assert chart['source']==item['source'] and chart['split']==item['split'] and chart['group']==item['group']
        assert [(c['index'],c['lane']) for c in chart['cases']]==[tuple(t) for t in selection['targets']]
        assert len(chart['cases'])==item['cases'] and len(chart['times'])==item['rows']
        assert all(torch.isfinite(c['context']).all() and len(c['context'])==1096 for c in chart['cases'])
    save(OUT/'resume-provenance.json',dict(resume_input_sha256=file_digest(resume_path),
        original_selection_sha256=file_digest(original_selection),retained_charts=len(retained),
        procedure='Identical frozen full-prefix computation for missing charts; retained features are hash-verified.'))
    save(OUT/'selection.json',dict(seed=SEED,sources=selected,weights_sha256=WEIGHTS_SHA,
         runtime_sha256=file_digest(RUNTIME/'sha256.json'),catalog_sha256=file_digest(CATALOG),split_sha256=split_sha,
         code_sha256={p.name:file_digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
         task='Known chosen LN head, all future event-union endpoints; onset roles supplied; earlier endpoint decisions teacher-forced.'))
    payload = torch.load(WEIGHTS,map_location='cpu',weights_only=True)
    model = CausalBackbone(BackboneConfig(**payload['model_config'])).eval()
    model.load_state_dict(payload['model_state_dict'])
    del payload
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    engine = ContinuationEngine(model,parallel_frontiers=True)
    captured = []
    handle = model.head.register_forward_pre_hook(lambda module,args:captured.append(args[0]))
    manifest = []
    with torch.no_grad():
        for ordinal,item in enumerate(selected):
            entry = item['entry']
            sha=entry['source_sha256']
            if sha in retained:
                prior=retained[sha]
                manifest.append(dict(file='../'+prior['path'],sha256=prior['sha256'],split=prior['split'],
                    cases=prior['cases'],rows=prior['rows'],group=prior['group'],reused=True))
                print(json.dumps(dict(ordinal=ordinal+1,source=sha[:12],reused=True,seconds=time.perf_counter()-started)),flush=True)
                continue
            source = admit_entry(store,entry)
            rows = np.fromfile(source.directory/'rows.bin',dtype=ROW_DTYPE)
            onsets,ends,pending = linked_rows(rows)
            assert np.array_equal(reconstruct(rows['actions'],ends),rows['actions'])
            wanted = {}
            for i,c in item['targets']:
                wanted.setdefault(i,[]).append(c)
            cases = []
            state = engine.start(source.skeleton,inference=True)
            stop = max(wanted)+1
            for start in range(0,stop,model.config.max_chunk):
                end = min(stop,start+model.config.max_chunk)
                captured.clear()
                result = engine.teacher_force(state,source.targets[start:end])
                state = result.state.detached()
                assert len(captured)==1 and len(captured[0].inputs)==end-start
                pre = captured[0]
                for i in range(start,end):
                    if i not in wanted:
                        continue
                    prior = tuple(None if x<0 else float(rows['time'][x]) for x in pending[i])
                    for c in wanted[i]:
                        assert rows['actions'][i,c]==2 and prior[c] is None
                        cases.append(dict(source=entry['source_sha256'],group=entry['group_id'],
                            split=entry['split'],index=i,lane=c,end_index=int(ends[i,c]),
                            context=context_features(pre.hidden[i-start],pre.inputs[i-start],rows['actions'][i],prior,c)))
                captured.clear()
                guard.check('extract',source=entry['source_sha256'][:12],row=end)
                assert time.perf_counter()-started < 1800
            assert len(cases)==len(item['targets'])
            path=OUT/(entry['source_sha256']+'.pt')
            with path.open('xb') as stream:
                torch.save(dict(source=entry['source_sha256'],group=entry['group_id'],split=entry['split'],
                           times=time_tensor(rows),onsets=torch.tensor(onsets),cases=cases),stream)
            manifest.append(dict(file=path.name,sha256=file_digest(path),split=entry['split'],cases=len(cases),
                                 rows=source.row_count,group=entry['group_id']))
            print(json.dumps(dict(ordinal=ordinal+1,source=entry['source_sha256'][:12],split=entry['split'],
                                  cases=len(cases),prefix_rows=stop,seconds=time.perf_counter()-started)),flush=True)
    handle.remove()
    save(OUT/'manifest.json',dict(status='complete',files=manifest,selection_sha256=file_digest(OUT/'selection.json'),
         context_dim=len(cases[0]['context']),seconds=time.perf_counter()-started,weights_sha256=file_digest(WEIGHTS)))
    assert file_digest(WEIGHTS)==WEIGHTS_SHA
    print(json.dumps(dict(status='complete',charts=len(manifest),cases=sum(m['cases'] for m in manifest),
                         manifest_sha256=file_digest(OUT/'manifest.json'),seconds=time.perf_counter()-started)),flush=True)


if __name__=='__main__':
    main()
