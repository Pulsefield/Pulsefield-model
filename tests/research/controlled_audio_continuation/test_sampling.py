from dataclasses import replace

import numpy as np
import pytest
import torch

from ensomi_model.research.bounded_typed_continuation.contract import ROW_ACTIONS
from ensomi_model.research.controlled_audio_continuation.generation import ControlledSession
from ensomi_model.research.controlled_audio_continuation.sampling import replay_row_scores
from ensomi_model.research.joint_audio_continuation.intervals import IntervalExample
from ensomi_model.research.planned_audio_continuation.intervals import collate_interval, score_interval
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule, ControlSpan
from controlled_audio_continuation.test_hold_audio import setup
from planned_audio_continuation.test_distribution import chart


def schedule(model):
    return ControlSchedule((ControlSpan(0,1001,stars=3,ln_fraction=.8),
        ControlSpan(350,650,ln_fraction=.2),
        ControlSpan(500,800,style={'ln-coordination':1})),model.style_names)


def source():
    # The first LN crosses both amount-control boundaries and several scoring
    # partitions. The second crosses the restored amount episode into EOF.
    return chart([0,100,200,300,550,700,900,1000],
        [(2,1,0,0),(0,0,1,0),(0,0,0,1),(0,1,0,0),
         (0,2,1,0),(0,0,0,1),(3,0,1,0),(0,3,0,0)],1000)


def scores(net,c,width,device='cpu'):
    encoded=net.encode_audio(torch.from_numpy(c.mel)[None].to(device))
    total=[]
    for i in range(IntervalExample(c,0,width).count):
        example=IntervalExample(c,i,width)
        batch=collate_interval(example,net.config,device,recovery=net.recovery)
        raw=score_interval(net,batch.inputs,None,controls=schedule(net),encoded_full=encoded)
        q=replay_row_scores(raw.row[:len(batch.row_index)],example,schedule(net))
        assert q.device.type=='cpu' and q.dtype==torch.float64
        total.append(-q[torch.arange(len(q)),batch.row_index.cpu()].sum())
    return torch.stack(total).sum()


def test_full_prefix_sampling_scores_and_gradients_survive_partition_and_control_changes():
    torch.manual_seed(921)
    net=setup(0)
    c=source()
    a=scores(net,c,1001)
    ga=torch.autograd.grad(a,net.composition.readout[-1].weight)[0]
    assert torch.isfinite(ga).all() and ga.abs().sum()>0
    b=scores(net,c,137)
    gb=torch.autograd.grad(b,net.composition.readout[-1].weight)[0]
    torch.testing.assert_close(a,b,atol=3e-5,rtol=3e-6)
    torch.testing.assert_close(ga,gb,atol=3e-5,rtol=3e-6)


def test_replayed_probabilities_match_the_distribution_actually_sampled():
    torch.manual_seed(751)
    net=setup(0)
    mel=np.random.default_rng(47).normal(size=(100,128)).astype(np.float32)
    recorded=[]
    class Recorder(ControlledSession):
        def prefer_rows(self,*args):
            q=super().prefer_rows(*args)
            recorded.append(q.clone())
            return q
    with torch.inference_mode():
        session=Recorder(net,mel,1000,schedule(net),seed=87,
                         head_times=(0,100,200,300,450,550,650,750,900))
        for end in (333,600,777,1000):session.publish_to(end)
    c=replace(chart([r.time_ms for r in session.rows],[r.actions for r in session.rows],1000),mel=mel)
    encoded=net.encode_audio(torch.from_numpy(mel)[None])
    actual=[]
    for i in range(IntervalExample(c,0,137).count):
        example=IntervalExample(c,i,137)
        batch=collate_interval(example,net.config,recovery=net.recovery)
        raw=score_interval(net,batch.inputs,None,controls=schedule(net),encoded_full=encoded)
        actual.append(replay_row_scores(raw.row[:len(batch.row_index)],example,schedule(net)))
    replayed=torch.cat(actual)
    torch.testing.assert_close(replayed,torch.stack(recorded),atol=3e-5,rtol=3e-6)
    torch.testing.assert_close(replayed.logsumexp(-1),torch.zeros(len(replayed),dtype=torch.float64),
                               atol=2e-6,rtol=0)
    # This loss uses each sampled action on its own prefix, not the source's
    # next action after changing history.
    loss=-replayed[torch.arange(len(replayed)),torch.tensor([
        ROW_ACTIONS.index(row.actions) for row in session.rows])].sum()
    assert torch.isfinite(loss)


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS unavailable')
def test_sampling_feedback_cpu_arithmetic_backpropagates_to_mps_weights():
    torch.manual_seed(271)
    cpu=setup(0)
    mps=setup(0).to('mps');mps.load_state_dict(cpu.state_dict())
    values=[]
    for device,net in (('cpu',cpu),('mps',mps)):
        loss=scores(net,source(),1001,device)
        gradient=torch.autograd.grad(loss,net.composition.readout[-1].weight)[0]
        assert torch.isfinite(loss) and torch.isfinite(gradient).all()
        values.append((loss.detach().cpu(),gradient.detach().cpu()))
    for a,b in zip(*values):torch.testing.assert_close(a,b,atol=3e-4,rtol=3e-4)
