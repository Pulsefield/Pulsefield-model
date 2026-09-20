from dataclasses import replace
from functools import lru_cache
import json

import numpy as np
import pytest
import torch

from pulsefield_model.research.bounded_typed_continuation import train_run
from pulsefield_model.research.bounded_typed_continuation.consequence import TIMING_DIM, consequence_features
from pulsefield_model.research.bounded_typed_continuation.condition import GenerationCondition
from pulsefield_model.research.bounded_typed_continuation.contract import Arm, ROW_ACTIONS, Schedule, Timing
from pulsefield_model.research.bounded_typed_continuation.data import SourceInterval, batch_likelihood, prepare_batch
from pulsefield_model.research.bounded_typed_continuation.features import time_features
from pulsefield_model.research.bounded_typed_continuation.model import BoundedModel, ModelConfig
from pulsefield_model.research.bounded_typed_continuation.recovery import complement_loss, trajectory_queries
from pulsefield_model.research.bounded_typed_continuation.response import response_costs, response_preference
from pulsefield_model.research.bounded_typed_continuation.support import row_supports
from pulsefield_model.research.bounded_typed_continuation.train_hydra import compose_config
from pulsefield_model.research.oracle_time_continuation.schema import CompleteRow
from pulsefield_model.research.scoped_style_modeling.dataset import ContractError
from .test_data import mixed_chart
from .test_fork import write_extension
from .test_recovery import pool_fixture
from .test_release_routing import held_config
from .test_train import compare_states


def current_cost(state,actions,threshold):
    return sum(action in (1,2) and any(t is not None and state.time_ms-t<threshold for t in
                (state.replay.last_lane_attack_ms[c],state.replay.last_lane_release_ms[c]))
               for c,action in enumerate(actions))


def exhaustive(state,threshold):
    @lru_cache(maxsize=None)
    def best(state,remaining):
        if not remaining or state.finished:return 0
        result=99
        for index in np.flatnonzero(row_supports([state])[0]):
            actions=ROW_ACTIONS[index]
            cost=current_cost(state,actions,threshold)
            child,_=state.advance(actions)
            value=cost+best(child,remaining-int(state.timing.onsets[state.index]))
            result=min(result,value)
            if result==0:break
        return result
    return best(state,2)



def test_two_onset_response_matches_exhaustive_legal_futures_and_mirrors():
    fixtures=[
        ((0.,50.,100.,117.,154.),(True,True,True,True,True),[(1,0,0,0),(2,2,0,0)],{}),
        ((0.,50.,100.,117.,154.),(True,True,False,True,True),[(1,0,0,0),(2,2,2,2)],{}),
        ((0.,50.,100.,120.,160.),(True,True,False,True,True),[(1,0,0,0),(2,2,2,2)],{}),
        ((0.,50.,100.,140.,180.),(True,True,False,True,True),[(1,0,0,0),(2,2,2,2)],{}),
        ((0.,50.,100.,110.,125.,160.),(True,True,True,False,True,True),[(1,0,0,0),(2,2,0,0)],{}),
        ((0.,50.,100.,117.,154.),(True,True,True,True,True),[(2,2,1,0),(0,0,0,1)],{0:3,1:4}),
        ((0.,50.,100.,117.,154.),(True,True,False,True,True),[(2,2,1,0),(0,0,0,1)],{0:2,1:3}),
    ]
    reverse = [ROW_ACTIONS.index(a[::-1]) for a in ROW_ACTIONS]
    preferences = 0
    for times, roles, rows, crossing in fixtures:
        states = []
        for mirrored in (False, True):
            timing = Timing(times, roles)
            actions = [a[::-1] if mirrored else a for a in rows]
            ends = {3-c if mirrored else c: end for c, end in crossing.items()}
            if crossing:
                state = Schedule.from_seed(Arm.R1, timing, [CompleteRow(t,a) for t,a in zip(times,actions)], ends)
            else:
                state = Schedule(Arm.R1, timing)
                for a in actions:
                    state, _ = state.advance(a)
            states.append(state)
        for threshold in (20., 30., 40.):
            state, mirror = states
            costs, legal = response_costs(state, threshold)
            mirrored_costs, mirrored_legal = response_costs(mirror, threshold)
            np.testing.assert_array_equal(costs, mirrored_costs[reverse])
            np.testing.assert_array_equal(legal, mirrored_legal[reverse])
            indices = np.flatnonzero(legal)
            for index in indices[np.linspace(0,len(indices)-1,min(16,len(indices)),dtype=int)]:
                action = ROW_ACTIONS[index]
                child, _ = state.advance(action)
                assert costs[index] == current_cost(state, action, threshold) + exhaustive(child, threshold)
                pref = response_preference(state, action, threshold)
                if pref is None:
                    continue
                preferences += 1
                good, family, info = pref
                reflected = response_preference(mirror, action[::-1], threshold)
                for original, counterpart in zip((good, family), reflected[:2]):
                    np.testing.assert_array_equal(original, counterpart[reverse])
                assert info == reflected[2] and costs[good].max() < costs[index]
                hc = np.isin(ROW_ACTIONS, (1,2)).sum(-1)
                lc = (np.array(ROW_ACTIONS)==2).sum(-1)
                if (legal & (costs < costs[index]) & (hc==hc[index]) & (lc==lc[index])).any():
                    assert (hc[good]==hc[index]).all() and (lc[good]==lc[index]).all()
    assert preferences > 0


def test_conditioned_preference_cannot_escape_to_other_compositions():
    logits = torch.tensor([[80., 0., -2., 99.]], dtype=torch.float64, requires_grad=True)
    good = np.array([[False, True, False, False]])
    family = np.array([[True, True, False, False]])
    loss = complement_loss(logits.log_softmax(-1), good, family)
    loss.backward()
    assert 79 < loss < 81 and logits.grad[0,0] > .99 and logits.grad[0,1] < -.99
    torch.testing.assert_close(logits.grad[0,2:], torch.zeros(2,dtype=torch.float64), atol=1e-14, rtol=0)
    changed = logits.detach().clone();changed[0,2:] += 1000
    torch.testing.assert_close(complement_loss(changed.log_softmax(-1),good,family), loss)
    with pytest.raises(ContractError):
        complement_loss(logits.log_softmax(-1), good, np.array([[True,False,True,False]]))


def test_second_frontier_is_supplied_timing_and_keeps_existing_feature_layout():
    states = [Schedule(Arm.R1, Timing((0.,79.,82.,end,200.),(True,False,True,True,False))) for end in (100.,180.)]
    local, future = consequence_features(states,'frontier2')
    old_local, old_future = consequence_features(states,'frontier')
    np.testing.assert_array_equal(local,old_local)
    np.testing.assert_array_equal(future[:,:TIMING_DIM],old_future)
    np.testing.assert_array_equal(future[:,TIMING_DIM:],time_features([100.,180.]))
    for s, row in zip(states,future):
        shifted = Schedule(Arm.R1,Timing(tuple(t+2**40 for t in s.timing.times_ms),s.timing.onsets))
        np.testing.assert_array_equal(consequence_features([shifted],'frontier2')[1][0],row)


def test_response_trajectory_checks_sampled_action_and_keeps_skipped_release():
    timing = Timing((0.,50.,90.,117.,140.,200.),(True,True,False,True,True,False))
    condition = GenerationCondition(Arm.R1,timing,(CompleteRow(0.,(1,0,0,0)),),(None,)*4)
    actions = [[2,2,2,1],[0,0,0,0],[3,0,0,1],[1,3,3,0],[0,0,0,0]]
    state = Schedule.from_seed(Arm.R1,timing,condition.seed_rows,condition.crossing)
    state,_ = state.advance(tuple(actions[0]))
    # Waiting until117 leaves only23ms to recover before the second H at140.
    pref = response_preference(state,actions[1],30.)
    assert pref is not None
    value = dict(condition=json.loads(json.dumps(condition.payload())),actions=actions,queries=[dict(kind='response',index=2,action=actions[1],threshold_ms=30.)])
    query, = trajectory_queries(value,condition)
    np.testing.assert_array_equal(query.alternatives,pref[0]);np.testing.assert_array_equal(query.family,pref[1])
    value['queries'][0]['action'] = [3,3,3,3]
    with pytest.raises(ContractError,match='sampled action'):
        trajectory_queries(value,condition)


@pytest.mark.parametrize('device',['cpu','mps'])
def test_source_kl_anchor_freezes_parent_and_preserves_likelihood_reporting(device):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    model = BoundedModel(ModelConfig(Arm.R1,hidden=8,levels=2,coupling_rank=2,row_consequence='frontier2')).to(device)
    train_run.configure_trainable(model,'consequence')
    source=mixed_chart();batch=prepare_batch([SourceInterval(source,0,len(source.onsets))],Arm.R1,7)
    plain, factors = batch_likelihood(model,batch)
    anchored, again = batch_likelihood(model,batch,source_kl_weight=1.)
    torch.testing.assert_close(plain,anchored,atol=0,rtol=0)
    torch.testing.assert_close(factors,again,atol=0,rtol=0)
    torch.nn.init.normal_(model.row_consequence.output.weight,std=.5)
    diagnostics={}
    plain,factors=batch_likelihood(model,batch)
    anchored,again=batch_likelihood(model,batch,source_kl_weight=1.,diagnostics=diagnostics)
    assert anchored>plain and diagnostics['source_kl_sum']>0
    torch.testing.assert_close(factors,again,atol=0,rtol=0)
    (anchored-plain).backward()
    assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('row_consequence.'))
    assert model.row_consequence.output.weight.grad.norm()>0


def test_response_fork_remaps_interleaved_adam_consumes_kl_and_resumes(tmp_path,monkeypatch):
    monkeypatch.setattr(train_run,'source_revision',lambda:'e'*40)
    config=held_config(tmp_path)
    config=replace(config,trainable='all',model=replace(config.model,release_routing='residual',release_hidden=16))
    pool,sha,_,_=pool_fixture(config,tmp_path)
    config=replace(config,recovery_pool=str(pool),recovery_sha256=sha)
    parent_result=train_run.run_training(config)
    parent=torch.load(tmp_path/'whole/checkpoint.pt',weights_only=True)
    extension,plan_sha=write_extension(config,tmp_path)
    monkeypatch.setattr(train_run,'source_revision',lambda:'f'*40)
    fork=replace(config,trainable='consequence',source_kl_weight=1.,model=replace(config.model,row_consequence='frontier2'),plan_file=extension,plan_sha256=plan_sha,output_dir=str(tmp_path/'response'),fork_from=str(tmp_path/'whole/checkpoint.pt'),fork_sha256=parent_result['checkpoint_sha256'],fork_source_revision='e'*40,fork_plan_file=config.plan_file)
    final_result=train_run.run_training(fork)
    final=torch.load(tmp_path/'response/checkpoint.pt',weights_only=True)
    old_names=list(dict(BoundedModel(config.model).named_parameters()))
    new_names=list(dict(BoundedModel(fork.model).named_parameters()))
    assert new_names[:len(old_names)]!=old_names
    for name,value in parent['model'].items():compare_states(value,final['model'][name])
    for old_id,name in enumerate(old_names):
        if old_id in parent['optimizer']['state']:compare_states(parent['optimizer']['state'][old_id],final['optimizer']['state'][new_names.index(name)])
    assert final_result['metrics']['source_kl_sum']>0 and final['model']['row_consequence.output.weight'].norm()>0
    train_run.run_training(replace(fork,output_dir=str(tmp_path/'paused'),stop_after_checkpoint=137))
    train_run.run_training(replace(fork,output_dir=str(tmp_path/'resumed'),fork_from=None,fork_sha256='',fork_source_revision='',fork_plan_file='',resume_from=str(tmp_path/'paused/checkpoint.pt')))
    resumed=torch.load(tmp_path/'resumed/checkpoint.pt',weights_only=True)
    for key in ('model','optimizer','torch_rng','coverage','cursor','source_onset_exposures'):compare_states(final[key],resumed[key])
    for change in ({'trainable':'all'},{'recovery_weight':.5},{'model':replace(fork.model,seed_context='zero')}):
        with pytest.raises(ContractError):train_run.run_training(replace(fork,output_dir=str(tmp_path/'invalid'),**change))


def test_response_hydra_projection_and_dormant_training_identity():
    config=compose_config(['model.arm=R1','model.row_consequence=frontier2','trainable=consequence','source_kl_weight=1.'])
    assert config.source_kl_weight==1. and config.trainable=='consequence' and config.model.row_consequence=='frontier2'
    for args in (['source_kl_weight=1.'],['trainable=consequence'],['source_kl_weight=-1.']):
        with pytest.raises(ContractError):compose_config(args)
    old={'model':{'row_consequence':'none'},'trainable':'all'}
    assert train_run.training_identity(old)==train_run.training_identity(dict(old,source_kl_weight=0.))
