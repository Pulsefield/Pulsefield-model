from dataclasses import asdict

import pytest
import torch

from ensomi_model.osu_core.difficulty import RawHitObject
from ensomi_model.research.controlled_audio_continuation.difficulty_tuning import (
    difficulty_columns, folded_state_dict, frozen_digest, tune_difficulty,
)
from ensomi_model.research.controlled_audio_continuation.model import load_model
from ensomi_model.research.controlled_audio_continuation.outcomes import scoped_difficulty
from ensomi_model.research.joint_audio_continuation.intervals import IntervalExample
from ensomi_model.research.planned_audio_continuation.intervals import collate_interval, score_interval
from ensomi_model.research.typed_audio_continuation.controls import ControlSchedule, ControlSpan
from controlled_audio_continuation.test_hold_audio import setup
from controlled_audio_continuation.test_sampling import source


@pytest.mark.parametrize('device',['cpu','mps'])
def test_only_difficulty_columns_change_and_fold_to_the_standard_checkpoint(device,tmp_path):
    if device=='mps' and not torch.backends.mps.is_available():pytest.skip('MPS unavailable')
    torch.manual_seed(517)
    model=setup(0).to(device);c=source()
    # The checkpoint's composition output is trained; the tiny fixture starts
    # that layer at zero, which would block gradients to its input columns.
    with torch.no_grad():model.composition.readout[-1].weight.normal_(std=.05)
    batch=collate_interval(IntervalExample(c,0,1001),model.config,device,recovery=model.recovery)
    with torch.no_grad():encoded=model.encode_audio(torch.from_numpy(c.mel)[None].to(device))
    def predict(stars):
        control=ControlSchedule((ControlSpan(0,1001,stars=stars,ln_fraction=.8),),model.style_names)
        return score_interval(model,batch.inputs,None,controls=control,encoded_full=encoded).row
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    initial=predict(3).detach();unknown=predict(None).detach()
    digest=frozen_digest(model)
    columns=tune_difficulty(model)
    initial_folded=folded_state_dict(model)
    assert all(torch.equal(value.cpu(),initial_folded[name]) for name,value in before.items())
    tolerance=0 if device=='cpu' else 3e-6
    torch.testing.assert_close(initial,predict(3),atol=tolerance,rtol=tolerance)
    trainable=[p for p in model.parameters() if p.requires_grad]
    assert len(trainable)==2 and sum(p.numel() for p in trainable)==(12+128)*48
    optimizer=torch.optim.AdamW(trainable,lr=.01,weight_decay=.1)
    q=predict(3)
    loss=-q[torch.arange(len(batch.row_index),device=device),batch.row_index].sum()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in trainable)
    optimizer.step()
    assert frozen_digest(model)==digest
    torch.testing.assert_close(unknown,predict(None),atol=tolerance,rtol=tolerance)
    state=folded_state_dict(model)
    assert state.keys()==before.keys()
    for name,original in before.items():
        changed=state[name]!=original.cpu()
        if name.removesuffix('.weight') in columns:
            mask=torch.zeros_like(changed);mask[:,columns[name.removesuffix('.weight')]]=True
            assert not changed[~mask].any() and changed[mask].any()
        else:assert not changed.any()
    path=tmp_path/f'{device}.pt'
    torch.save(dict(format='controlled-audio/v1',model_config=asdict(model.config),
        probability_options=model.probability_options(),model=state),path)
    restored=load_model(path,device=device)
    with torch.no_grad():model.row_control.parametrizations.weight[0].values.add_(.01)
    assert all(torch.equal(v.cpu(),state[k]) for k,v in restored.state_dict().items())
    assert difficulty_columns(restored)==columns


def test_scoped_outcome_keeps_real_crossing_tails_but_excludes_later_heads():
    objects=[RawHitObject(0,0,3),RawHitObject(100,100,1),RawHitObject(200,900,0),
        RawHitObject(500,500,1),RawHitObject(600,1000,2),RawHitObject(850,1300,3),
        RawHitObject(1200,1200,0)]
    result=scoped_difficulty(objects,400,800)
    assert result.score_end_ms==1001
    changed=scoped_difficulty(objects[:5]+[RawHitObject(850,1700,3)],400,800)
    assert changed==result
    earlier=list(objects);earlier[2]=RawHitObject(200,450,0)
    assert scoped_difficulty(earlier,400,800).level!=result.level
    assert scoped_difficulty([],400,800).score_end_ms==800
