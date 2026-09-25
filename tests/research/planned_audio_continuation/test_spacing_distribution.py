from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from ensomi_model.research.bounded_typed_continuation.contract import ROW_ACTIONS
from ensomi_model.research.joint_audio_continuation.intervals import IntervalExample
from ensomi_model.research.oracle_time_continuation.replay import ExactReplayState
from ensomi_model.research.planned_audio_continuation import buffering, generation
from ensomi_model.research.planned_audio_continuation.hydra import compose_config
from ensomi_model.research.planned_audio_continuation.intervals import collate_interval, interval_losses, score_interval
from ensomi_model.research.planned_audio_continuation.model import PlannedAudioModel
from ensomi_model.research.planned_audio_continuation.release import conditioned_release_logits
from ensomi_model.research.scoped_style_modeling.dataset import ContractError
from .test_distribution import chart, config


def settings(**kwargs):
    return replace(config(), lookahead=9, minimum_action_gap_ms=21, condition_full_holds=True, **kwargs)


def held_source():
    return chart([0,35,80,100,110],
        [(2,2,2,2),(3,0,0,0),(2,0,0,0),(0,3,3,3),(3,0,0,0)],110)


def test_conditional_likelihood_respects_minimum_duration_and_true_release_deadlines():
    model = PlannedAudioModel(settings())
    for p in model.parameters():
        torch.nn.init.zeros_(p)
    with torch.no_grad():
        model.timing[-1].bias.fill_(math.log(.2/.8))
        model.release_clock[-1].bias.fill_(math.log(.3/.7))
    c = held_source()
    batch = collate_interval(IntervalExample(c,0,200),model.config)
    scores = score_interval(model,batch.inputs,model.encode_coarse(torch.from_numpy(c.mel)[None]))
    losses = interval_losses(scores,batch)
    # First all-held wait is conditioned on R in [21,59]. Following R=35,
    # partial occupancy has no deadline before H=80 and survives 36..79.
    # After H=80, terminal-bound waits are [81,110], then [101,110].
    expected_r = (-math.log(.3*.7**14/(1-.7**39))-44*math.log(.7)
                  -math.log(.3*.7**19/(1-.7**30))-math.log(.3*.7**9/(1-.7**10)))
    torch.testing.assert_close(losses[1],losses[1].new_tensor(expected_r),atol=1e-5,rtol=1e-6)
    assert len(batch.inputs.release_waits)==3
    # A release of the new column-0 LN at t=100 would be only 20 ms old.
    i = batch.inputs.base.row_times.tolist().index(100)
    assert not batch.inputs.response_allowed[i,ROW_ACTIONS.index((3,3,3,3))]
    assert batch.inputs.response_allowed[i,ROW_ACTIONS.index((0,3,3,3))]
    losses[-1].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_spacing_likelihood_partition_and_gradients_agree():
    torch.manual_seed(914)
    c = held_source(); model = PlannedAudioModel(settings()).eval()
    def evaluate(width):
        coarse = model.encode_coarse(torch.from_numpy(c.mel)[None]); total = None
        for i in range(IntervalExample(c,0,width).count):
            b = collate_interval(IntervalExample(c,i,width),model.config)
            values = torch.stack(interval_losses(score_interval(model,b.inputs,coarse),b))
            total = values if total is None else total+values
        gradient = torch.autograd.grad(total[-1],model.release_clock[-1].bias)[0]
        return total.detach(),gradient
    a,ga=evaluate(200); b,gb=evaluate(17)
    torch.testing.assert_close(a,b,atol=3e-5,rtol=3e-6)
    torch.testing.assert_close(ga,gb,atol=3e-5,rtol=3e-6)


def test_complete_row_conditioning_changes_count_marginals_after_composition():
    torch.manual_seed(915)
    model=PlannedAudioModel(settings(history_levels=4,row_factorization='count_layout'))
    c=chart([0,5],[(1,0,0,0),(0,1,0,0)],100)
    b=collate_interval(IntervalExample(c,0,200),model.config)
    coarse=model.encode_coarse(torch.from_numpy(c.mel)[None])
    constrained=score_interval(model,b.inputs,coarse).row
    model.config=replace(model.config,minimum_action_gap_ms=0)
    raw=score_interval(model,b.inputs,coarse).row
    expected=raw.masked_fill(~b.inputs.response_allowed,-torch.inf).log_softmax(-1)
    torch.testing.assert_close(constrained,expected)
    counts=torch.tensor([sum(a in (1,2) for a in row) for row in ROW_ACTIONS])
    assert raw[0,counts==4].exp().sum()>0
    assert constrained[0,counts==4].exp().sum()==0
    assert constrained[0,counts==1].exp().sum()>raw[0,counts==1].exp().sum()


def test_tap_layout_does_not_enter_spacing_head_or_release_law():
    a=chart([0,30,60,90,150],[(0,2,0,0),(1,0,0,0),(0,0,1,0),(0,0,0,1),(0,3,0,0)],200)
    b=chart([0,30,60,90,150],[(0,2,0,0),(0,0,0,1),(1,0,0,0),(0,0,1,0),(0,3,0,0)],200)
    model=PlannedAudioModel(settings()).eval()
    coarse=model.encode_coarse(torch.from_numpy(a.mel)[None])
    batches=[collate_interval(IntervalExample(c,0,300),model.config) for c in (a,b)]
    scores=[score_interval(model,c.inputs,coarse) for c in batches]
    for name in ('head','release'):
        torch.testing.assert_close(getattr(scores[0],name),getattr(scores[1],name),rtol=0,atol=0)
    for name in ('head_valid','release_valid','release_forced'):
        assert torch.equal(getattr(batches[0].inputs,name),getattr(batches[1].inputs,name))


@pytest.mark.parametrize('factor',['flat','count_layout'])
def test_native_and_teacher_row_laws_match_and_chunking_preserves_generation(factor):
    torch.manual_seed(916)
    model=PlannedAudioModel(settings(history_levels=4,row_factorization=factor)).eval()
    with torch.no_grad():
        model.timing[-1].bias.fill_(math.log(.15/.85))
        model.release_clock[-1].bias.fill_(math.log(.08/.92))
    mel=np.random.default_rng(33).normal(size=(30,128)).astype(np.float32)
    original=model.planned_row_log_probs; records=[]
    def track(*a,**kw):
        value=original(*a,**kw);records.append(value.detach().clone());return value
    model.planned_row_log_probs=track
    a=generation.rollout(model,mel,300,seed=19,chunk_ms=23,head_chunk_ms=31)
    model.planned_row_log_probs=original
    b=generation.rollout(model,mel,300,seed=19,chunk_ms=17,head_chunk_ms=7)
    assert a.completed and a.rows==b.rows
    assert not buffering.close_pairs(ExactReplayState(),a.rows,300,
        screen_release_heads=True,minimum_action_gap_ms=21)
    source=chart([r.time_ms for r in a.rows],[r.actions for r in a.rows],300)
    source=replace(source,mel=mel)
    batch=collate_interval(IntervalExample(source,0,400),model.config)
    with torch.no_grad():
        values=score_interval(model,batch.inputs,model.encode_coarse(torch.from_numpy(mel)[None]))
    assert len(batch.row_index)==len(a.rows)
    torch.testing.assert_close(values.row[:len(batch.row_index)],torch.cat(records),atol=2e-5,rtol=2e-6)
    updates=[]
    buffered=buffering.rollout_buffered(model,mel,300,seed=19,window_ms=33,on_update=updates.append)
    assert buffered.completed and buffered.rows==a.rows and buffered.metrics['rejected_proposals']==0
    assert [u.row for u in updates if u.row is not None]==list(a.rows)
    assert updates[-1].completed


def test_reference_incompatibility_and_invalid_fixed_head_plans_are_explicit():
    model=PlannedAudioModel(settings())
    bad=chart([0,20,50],[(2,0,0,0),(3,0,0,0),(1,0,0,0)],100)
    with pytest.raises(ContractError,match='Source release likelihood'):
        collate_interval(IntervalExample(bad,0,200),model.config)
    with pytest.raises(ContractError,match='four-column'):
        generation.rollout(model,bad.mel,100,seed=1,head_times=(0,1,2,3,20))
    with pytest.raises(ContractError,match='cannot combine'):
        generation.rollout(model,bad.mel,100,seed=1,row_constraint='current')
    assert compose_config(['minimum_action_gap_ms=21','condition_full_holds=true']).minimum_action_gap_ms==21
    with pytest.raises(ValueError,match='conditional release waits'):
        compose_config(['minimum_action_gap_ms=21'])
    with pytest.raises(ValueError,match='Minimum action gap'):
        compose_config(['minimum_action_gap_ms=-1'])


def test_native_release_process_frees_columns_before_a_dense_cluster_and_preserves_forks():
    from ensomi_model.research.planned_audio_continuation.session import ContinuationSession

    torch.manual_seed(917)
    model=PlannedAudioModel(settings()).eval()
    with torch.no_grad():
        model.release_clock[-1].weight.zero_()
        model.release_clock[-1].bias.fill_(-1000.)
    original=model.planned_row_log_probs
    def favor_holds(*args,**kwargs):
        scores=original(*args,**kwargs)
        bias=scores.new_zeros(256);bias[ROW_ACTIONS.index((2,2,2,2))]=30
        return (scores+bias).log_softmax(-1)
    model.planned_row_log_probs=favor_holds
    mel=np.zeros((21,128),np.float32); heads=(0,80,81,82,83,200)
    session=ContinuationSession(model,mel,200,seed=18,planner_factory=generation.HeadPlanner,head_times=heads)
    session.step()
    assert session.rows[0].actions==(2,2,2,2)
    fork=session.fork(); prefix=tuple(session.rows); replay=session.replay
    while fork.cursor<200: fork.step()
    assert tuple(session.rows)==prefix and session.replay is replay and session.cursor==0
    assert fork.conditioned_waits>0
    assert min(r.time_ms for r in fork.rows if any(a==3 for a in r.actions))>=21
    assert not buffering.close_pairs(ExactReplayState(),fork.rows,200,
        screen_release_heads=True,minimum_action_gap_ms=21)
    generated=generation.rollout(model,mel,200,seed=18,head_times=heads,chunk_ms=7)
    assert generated.completed and tuple(fork.rows)==generated.rows
    assert [r.time_ms for r in generated.rows if any(a in (1,2) for a in r.actions)]==list(heads)
    assert not any(generated.metrics['open_lanes'])


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS unavailable')
def test_spacing_mps_likelihood_and_gradients_match_cpu():
    torch.manual_seed(918)
    cpu=PlannedAudioModel(settings()); mps=PlannedAudioModel(settings()).to('mps')
    mps.load_state_dict(cpu.state_dict()); c=held_source(); results=[]
    for device,model in (('cpu',cpu),('mps',mps)):
        batch=collate_interval(IntervalExample(c,0,200),model.config,device)
        coarse=model.encode_coarse(torch.from_numpy(c.mel)[None].to(device))
        losses=torch.stack(interval_losses(score_interval(model,batch.inputs,coarse),batch))
        losses[-1].backward()
        gradients={n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.grad is not None}
        assert all(torch.isfinite(v).all() for v in gradients.values())
        results.append((losses.detach().cpu(),gradients))
    torch.testing.assert_close(results[0][0],results[1][0],atol=2e-4,rtol=1e-5)
    assert results[0][1].keys()==results[1][1].keys()
    for name,a in results[0][1].items():
        torch.testing.assert_close(a,results[1][1][name],atol=2e-4,rtol=2e-4)
