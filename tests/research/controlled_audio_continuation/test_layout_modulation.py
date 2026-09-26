"""Conditional placement capacity without changing the skeleton or row contract."""
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from ensomi_model.research.bounded_typed_continuation.contract import ROW_ACTIONS
from ensomi_model.research.controlled_audio_continuation.generation import ControlledSession
from ensomi_model.research.controlled_audio_continuation.model import ControlledAudioModel, load_model
from ensomi_model.research.controlled_audio_continuation.sampling import replay_row_scores
from ensomi_model.research.joint_audio_continuation.intervals import IntervalExample
from ensomi_model.research.planned_audio_continuation.intervals import collate_interval, score_interval
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule, ControlSpan
from planned_audio_continuation.test_distribution import chart, config
from controlled_audio_continuation.test_sampling import source


DEVICES = ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='MPS unavailable'))]


def model(enabled):
    torch.set_num_threads(1)
    return ControlledAudioModel(replace(config(), lookahead=16, bounded_head=True,
        condition_full_holds=True, minimum_action_gap_ms=60),
        style_names=('trill-organization',), layout_modulation=enabled).eval()


def controls(net, value=1):
    return ControlSchedule((ControlSpan(0,1001,stars=3,ln_fraction=.4),
                            ControlSpan(200,800,style={'trill-organization':value})),net.style_names)


@pytest.mark.parametrize('device', DEVICES)
def test_identity_initialization_preserves_probabilities_and_strict_checkpoint_loading(tmp_path, device):
    torch.manual_seed(235)
    base, added = model(False), model(True)
    receipt = added.load_state_dict(base.state_dict(),strict=False)
    assert receipt.missing_keys == ['layout_modulation.weight'] and not receipt.unexpected_keys
    base.to(device);added.to(device)
    c = source()
    encoded = base.encode_audio(torch.from_numpy(c.mel)[None].to(device))
    batch = collate_interval(IntervalExample(c,0,1001),base.config,device,recovery=base.recovery)
    values = [score_interval(n,batch.inputs,None,controls=controls(n),encoded_full=encoded)
              for n in (base,added)]
    for a,b in zip(vars(values[0]).values(),vars(values[1]).values()):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    for net,legacy in ((base,True),(added,False)):
        options=net.probability_options()
        if legacy:del options['layout_modulation']
        path=tmp_path/('old.pt' if legacy else 'modulated.pt')
        torch.save(dict(format='controlled-audio/v1',model_config=asdict(net.config),
            probability_options=options,model=net.state_dict()),path)
        restored=load_model(path,device=device)
        assert (restored.layout_modulation is not None) == (not legacy)
        for name,value in net.state_dict().items():
            torch.testing.assert_close(value,restored.state_dict()[name],atol=0,rtol=0)


@pytest.mark.parametrize('device', DEVICES)
def test_control_history_interaction_is_learnable_and_mirror_equivariant(device):
    torch.manual_seed(610)
    net=model(True).to(device)
    with torch.no_grad():
        net.row_control.weight.normal_(std=.1)
    c=chart([0,100,200,300,550,700,900],
        [(1,1,0,0),(0,0,1,1),(1,1,0,0),(0,0,1,1),
         (1,1,0,0),(0,0,1,1),(1,1,0,0)],1000)
    encoded=net.encode_audio(torch.from_numpy(c.mel)[None].to(device))
    batch=collate_interval(IntervalExample(c,0,1001),net.config,device,recovery=net.recovery)
    captured=[]
    handle=net.joint.register_forward_hook(lambda module,args,value:captured.append(value))
    def run(value,inputs=batch.inputs):
        return score_interval(net,inputs,None,controls=controls(net,value),encoded_full=encoded)
    before=[run(value) for value in (-1,1)]
    left,right=[ROW_ACTIONS.index(row) for row in ((1,1,0,0),(0,0,1,1))]
    def contrast(a,b):
        return (b[:,left]-b[:,right])-(a[:,left]-a[:,right])
    delta=contrast(*captured)
    torch.testing.assert_close(delta,torch.zeros_like(delta),atol=3e-6,rtol=0)
    gradient=torch.autograd.grad(delta.sum(),net.layout_modulation.weight)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>1e-4
    with torch.no_grad():net.layout_modulation.weight.normal_(std=.1)
    captured.clear()
    after=[run(value) for value in (-1,1)]
    assert contrast(*captured).abs().max()>1e-4
    for old,new in zip(before,after):
        torch.testing.assert_close(old.head,new.head,atol=0,rtol=0)
        torch.testing.assert_close(old.release,new.release,atol=0,rtol=0)
    reflected=chart(c.source.rows['time'],c.source.rows['actions'][:,::-1],c.duration_ms)
    mb=collate_interval(IntervalExample(reflected,0,1001),net.config,device,recovery=net.recovery)
    mirrored=run(1,mb.inputs)
    reverse=[ROW_ACTIONS.index(row[::-1]) for row in ROW_ACTIONS]
    torch.testing.assert_close(after[1].row,mirrored.row[:,reverse],atol=2e-5,rtol=2e-5)
    handle.remove()


def test_nonzero_modulation_native_cache_matches_scoring_across_control_scopes():
    torch.manual_seed(920)
    net=model(True)
    with torch.no_grad():
        net.layout_modulation.weight.normal_(std=.08)
        net.row_control.weight.normal_(std=.04)
    mel=np.random.default_rng(17).normal(size=(100,128)).astype(np.float32)
    cc=controls(net)
    sampled=[]
    class Recorder(ControlledSession):
        def prefer_rows(self,*args):
            q=super().prefer_rows(*args);sampled.append(q.clone());return q
    with torch.inference_mode():
        session=Recorder(net,mel,1000,cc,seed=67,
            head_times=(0,100,200,300,450,550,650,750,900))
        for end in (333,600,777,1000):session.publish_to(end)
    generated=replace(chart([r.time_ms for r in session.rows],
        [r.actions for r in session.rows],1000),mel=mel)
    encoded=net.encode_audio(torch.from_numpy(mel)[None])
    rescored=[]
    for i in range(IntervalExample(generated,0,137).count):
        example=IntervalExample(generated,i,137)
        batch=collate_interval(example,net.config,recovery=net.recovery)
        raw=score_interval(net,batch.inputs,None,controls=cc,encoded_full=encoded)
        rescored.append(replay_row_scores(raw.row[:len(batch.row_index)],example,cc))
    q=torch.cat(rescored)
    torch.testing.assert_close(q,torch.stack(sampled),atol=3e-5,rtol=3e-6)
    target=torch.tensor([ROW_ACTIONS.index(row.actions) for row in session.rows])
    loss=-q[torch.arange(len(q)),target].sum()
    gradient=torch.autograd.grad(loss,net.layout_modulation.weight)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
