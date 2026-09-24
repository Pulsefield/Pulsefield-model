from dataclasses import asdict, replace
import copy
from importlib.resources import files
import json

import numpy as np
import pytest
import torch

from ensomi_model.research.joint_audio_continuation.context_model import ContextAudioModel
from ensomi_model.research.joint_audio_continuation.context_training import song_context
from ensomi_model.research.joint_audio_continuation.batching import collate
from ensomi_model.research.joint_audio_continuation.context_evaluation import score_context_queries
from ensomi_model.research.joint_audio_continuation.data import query, digest
from ensomi_model.research.joint_audio_continuation.generation import load_model, rollout
from ensomi_model.research.joint_audio_continuation.intent_config import IntentTrainConfig
from ensomi_model.research.joint_audio_continuation.intent_hydra import compose_config
from ensomi_model.research.joint_audio_continuation.intent_model import (
    IntentAudioModel, IntentModelConfig, arrangement_summary,
)
from ensomi_model.research.joint_audio_continuation import intent_training as training
from ensomi_model.research.joint_audio_continuation.intervals import (
    IntervalExample, collate_interval, interval_losses, score_interval,
)
from ensomi_model.research.scoped_style_modeling.dataset import ContractError
from .test_context_intervals import context_config
from .test_context_training import corpus
from .test_batching import chart
from .test_audio_inference import real_ten_ms_wav


def models():
    base = ContextAudioModel(context_config(global_audio=True, bounded_timing=True))
    intent = IntentAudioModel(IntentModelConfig(**asdict(base.config)))
    intent.load_state_dict(base.state_dict(), strict=False)
    return base, intent


def test_summary_uses_whole_target_without_labels_and_observes_native_durations():
    source = chart([0,10,50,200],[(2,0,0,0),(0,1,0,0),(3,0,0,0),(0,0,1,0)],300)
    actual = arrangement_summary(source)
    np.testing.assert_allclose(actual,[np.log1p(3000/301),1/3,1,0,0,0,50/(4*301),np.log1p(.05)],rtol=1e-6)
    changed = chart([0,10,150,200],[(2,0,0,0),(0,1,0,0),(3,0,0,0),(0,0,1,0)],300)
    assert not np.array_equal(actual,arrangement_summary(changed))


def test_zero_offsets_match_original_decoder_for_every_code_and_missing_code_is_rejected():
    torch.manual_seed(56)
    base, model = models()
    assert training.common_identity(base)==training.common_identity(model)
    source=corpus()[0];ex=IntervalExample(source,0,500);batch=collate_interval(ex,base.config)
    coarse=song_context(base,source,'cpu')
    reference=score_interval(base,batch.inputs,coarse)
    for code in range(4):
        actual=score_interval(model,batch.inputs,coarse,code=code)
        torch.testing.assert_close(actual.timing_logits,reference.timing_logits,atol=0,rtol=0)
        torch.testing.assert_close(actual.row_log_probs,reference.row_log_probs,atol=0,rtol=0)
    with pytest.raises(ContractError,match='Intent code'):
        score_interval(model,batch.inputs,coarse)
    with pytest.raises(ContractError,match='choose a persistent code'):
        model.encode_audio(torch.from_numpy(source.mel)[None])
    with pytest.raises(ContractError,match='no persistent intent'):
        rollout(base,source.mel,source.duration_ms,intent_code=2)


def test_enumerated_objective_and_shared_backbone_gradients_match_independent_code_scores():
    torch.manual_seed(73)
    _,model=models()
    with torch.no_grad():
        model.intent.weight.normal_(std=.15)
        model.context_condition.weight.normal_(std=.05)
        model.context_base.weight.normal_(std=.05)
        model.intent_prior[-1].weight.normal_(std=.03)
    independent=copy.deepcopy(model)
    source=corpus()[0];ex=IntervalExample(source,0,500);batch=collate_interval(ex,model.config)
    summary=torch.from_numpy(arrangement_summary(source))
    coarse=song_context(model,source,'cpu');q,kl,_,_=training.state_terms(model,coarse,summary)
    components=training.component_losses(model,batch,coarse)
    actual,_,kl_s=training.objective(components,q,kl,ex)
    assert kl_s.item()==pytest.approx(kl.item()*1000/(source.duration_ms+1))
    actual.backward()
    other_coarse=song_context(independent,source,'cpu')
    oq,okl,_,_=training.state_terms(independent,other_coarse,summary)
    losses=torch.stack([interval_losses(score_interval(independent,batch.inputs,other_coarse,code=z),batch)[2] for z in range(4)])
    expected=(oq*losses).sum()*ex.weight_per_second+okl*1000/(source.duration_ms+1)
    expected.backward()
    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-6)
    for (name,a),(other,b) in zip(model.named_parameters(),independent.named_parameters()):
        assert name==other
        if a.grad is None or b.grad is None:assert a.grad is None and b.grad is None
        else:torch.testing.assert_close(a.grad,b.grad,atol=2e-4,rtol=2e-4)
    for module in (model.intent,model.intent_prior,model.intent_posterior,model.audio_input):
        assert sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)>0


def test_fixed_code_does_not_read_reference_future_or_renormalize_on_scheduler_edges():
    torch.manual_seed(84);_,model=models();model.eval()
    a=chart([0,10,100,200],[(2,0,0,0),(0,1,0,0),(3,0,0,0),(0,0,1,0)],300)
    b=chart([0,10,150,200],[(2,0,0,0),(0,1,0,0),(3,0,0,0),(0,0,1,0)],300)
    coarse=song_context(model,a,'cpu')
    # Endpoints differ, but no event occurs in the same 40 ms observation window.
    ba=collate([query(a,10,history_limit=7,horizon_ms=40)],[a],model.config)
    bb=collate([query(b,10,history_limit=7,horizon_ms=40)],[b],model.config)
    x=score_context_queries(model,ba.inputs,coarse,code=2);y=score_context_queries(model,bb.inputs,coarse,code=2)
    torch.testing.assert_close(x.timing_logits,y.timing_logits,atol=0,rtol=0)
    torch.testing.assert_close(x.row_log_probs,y.row_log_probs,atol=0,rtol=0)
    calls=[]
    hook=model.context.register_forward_hook(lambda *args:calls.append(True))
    first=rollout(model,a.mel,a.duration_ms,seed=71,chunk_ms=37,intent_code=2)
    assert len(calls)==1
    second=rollout(model,a.mel,a.duration_ms,seed=71,chunk_ms=500,intent_code=2)
    assert first.rows==second.rows and first.completed and second.completed
    assert first.metrics['intent_code']==2 and first.metrics['intent_selection']=='explicit_code'
    sampled=rollout(model,a.mel,a.duration_ms,seed=71,chunk_ms=37)
    assert sampled.metrics['intent_selection']=='audio_prior' and sampled.metrics['intent_prior']==[.25]*4
    assert len(calls)==3;hook.remove()


def test_intent_hydra_projects_states_and_rejects_unknown_or_invalid_settings():
    assert files('ensomi_model.configs.hydra').joinpath('joint_audio_intent.yaml').is_file()
    cfg=compose_config(['states=4','updates=2','validation_songs=1','intent_learning_rate=0.0007'])
    assert (cfg.states,cfg.updates,cfg.validation_songs,cfg.intent_learning_rate)==(4,2,1,.0007)
    with pytest.raises(ValueError,match='Unknown'):compose_config(['+unused=1'])
    with pytest.raises(ValueError,match='states=1 or 4'):compose_config(['states=2'])
    with pytest.raises(ValueError,match='frozen plan'):compose_config(['updates=1201'])


def test_source_free_runner_consumes_explicit_code_and_never_reads_recognition_targets(tmp_path,monkeypatch):
    from ensomi_model.research.joint_audio_continuation import generation
    from ensomi_model.research.joint_audio_continuation.hydra import compose_config as inference_config
    _,model=models();path=tmp_path/'intent.pt';audio=tmp_path/'new.wav';real_ten_ms_wav(audio)
    torch.save(dict(format='joint-audio/intent-v1',model_config=asdict(model.config),model=model.state_dict(),
        source_revision='e'*40,manifest_sha256='d'*64,config={}),path)
    def forbidden(*args,**kwargs):raise AssertionError('Source-free inference cannot read target statistics')
    monkeypatch.setattr(generation,'_clean_revision',lambda:'a'*40)
    monkeypatch.setattr(generation,'_resource_stop',lambda root:None)
    monkeypatch.setattr(generation,'load_corpus',forbidden)
    monkeypatch.setattr(IntentAudioModel,'posterior_log_probs',forbidden)
    cfg=inference_config(['mode=infer_audio','device=cpu','root='+str(tmp_path/'outputs'),
        'audio_file='+str(audio),'checkpoint_file='+str(path),'checkpoint_sha256='+digest(path),'intent_code=2'])
    result=generation.infer_audio(cfg)
    receipt=json.loads(open(result['result_file']).read())
    assert receipt['intent_code']==2 and receipt['intent_selection']=='explicit_code'
    assert receipt['intent_prior']==[.25]*4 and result['status']=='completed'
    for bad in ('intent_code=4','intent_code=-1'):
        with pytest.raises(ValueError,match='intent_code'):
            inference_config(['mode=generate','checkpoint_file=unused.pt','checkpoint_sha256='+'a'*64,bad])


@pytest.mark.parametrize('states',[1,4])
def test_training_entrypoint_pins_common_weights_full_audio_plan_and_native_checkpoint(tmp_path,monkeypatch,states):
    cs=corpus();base,_=models();initial=tmp_path/'initial.pt'
    torch.save(dict(format='joint-audio/context-v1',model_config=asdict(base.config),model=base.state_dict(),
                    source_revision='f'*40,manifest_sha256='e'*64,config={}),initial)
    manifest=tmp_path/'manifest.json';manifest.write_text('{}\n')
    cfg=IntentTrainConfig(root=str(tmp_path),manifest_sha256=digest(manifest),initial_checkpoint_file=str(initial),
        initial_checkpoint_sha256=digest(initial),states=states,run_name=f'k{states}',plan_updates=2,updates=2,
        interval_ms=500,validation_every=2,songs_per_update=1,intervals_per_song=2,device='cpu',max_seconds=60)
    monkeypatch.setattr(training,'revision',lambda:'a'*40)
    monkeypatch.setattr(training,'load_corpus',lambda root:(cs,{}))
    monkeypatch.setattr(training,'_resource_stop',lambda root:None)
    result=training.train(cfg,resolved_yaml=f'states: {states}\n')
    assert result['status']=='completed' and result['intervals']==4
    directory=tmp_path/'intent-training'/cfg.run_name
    freeze=json.loads((directory/'freeze.json').read_text())
    assert freeze['common_initial_sha256']==training.common_identity(base)
    assert (directory/'resolved.yaml').read_text()==f'states: {states}\n'
    model,metadata=load_model(directory/'last.pt',result['checkpoint_sha256'])
    assert isinstance(model,IntentAudioModel)==(states==4)
    torch.testing.assert_close(model.audio_mean,base.audio_mean,atol=0,rtol=0)
    assert rollout(model,cs[0].mel,cs[0].duration_ms,seed=13,chunk_ms=117).completed
    with pytest.raises(FileExistsError):training.train(cfg)
