from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

import numpy as np
import pytest
import torch

from pulsefield_model.research.bounded_typed_continuation.contract import Arm
from pulsefield_model.research.bounded_typed_continuation.data import SourceInterval, batch_likelihood, prepare_batch
from pulsefield_model.research.bounded_typed_continuation.features import query_features
from pulsefield_model.research.bounded_typed_continuation.generation import Rollout, model_digest
from pulsefield_model.research.bounded_typed_continuation.model import BoundedModel, ModelConfig
from pulsefield_model.research.scoped_style_modeling.dataset import ContractError
from .test_data import chart


def memory_model(device='cpu'):
    torch.manual_seed(821)
    model = BoundedModel(ModelConfig(Arm.R1, hidden=8, levels=2, coupling_rank=2,
        seed_context='observed', long_memory='landmarks', memory_hidden=12, memory_stride=4)).to(device)
    torch.nn.init.normal_(model.long_memory.output.weight, std=.1)
    return model


def encode(model, batch):
    like = model.temporal.input.weight
    valid = torch.as_tensor(batch.valid, device=like.device)
    before = model.temporal.before(model.temporal(like.new_tensor(batch.raw), valid), valid,
                                   truncated_start=torch.as_tensor(batch.truncated, device=like.device))
    seed = model.encode_seed(like.new_tensor(batch.seed_raw), torch.as_tensor(batch.seed_valid, device=like.device))
    bank = model.long_memory.encode(like.new_tensor(batch.memory_raw),
        torch.as_tensor(batch.memory_valid, device=like.device), torch.as_tensor(batch.memory_onsets, device=like.device))
    return before, seed, bank


def test_configuration_and_default_parameter_digest_remain_compatible():
    for arm in (Arm.R0, Arm.O1):
        with pytest.raises(ContractError, match='R1-only'):
            ModelConfig(arm, long_memory='landmarks')
    for settings in ({'long_memory':'unknown'}, {'memory_hidden':0}, {'memory_stride':False}):
        with pytest.raises(ContractError):
            ModelConfig(Arm.R1, **settings)
    model = BoundedModel(ModelConfig(Arm.R1, hidden=8, levels=2, coupling_rank=2))
    old = asdict(model.config)
    for k in ('seed_context','long_memory','memory_hidden','memory_stride','head_routing','routing_hidden'):
        old.pop(k)
    h = hashlib.sha256(json.dumps(old, sort_keys=True).encode())
    for name, tensor in model.state_dict().items():
        h.update(json.dumps((name, str(tensor.dtype), tuple(tensor.shape))).encode())
        h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    assert h.hexdigest() == model_digest(model)


@pytest.mark.parametrize('device', ['cpu','mps'])
def test_full_history_causal_dense_native_and_raw_recovery(device, monkeypatch):
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    cycle = [(2,2,0,0),(0,0,1,0),(3,0,0,0),(1,0,2,0),(0,3,0,0),
             (2,0,0,1),(0,0,3,0),(3,0,0,0),(1,0,0,0)]
    source = chart(cycle*7, minimum_seed_notes=5)
    model = memory_model(device).eval()
    batch = prepare_batch([SourceInterval(source,0,len(source.onsets))], Arm.R1,7,full_history=True)
    like = model.temporal.input.weight
    before, seed, bank = encode(model,batch)
    initial = source.state(Arm.R1,source.seed_rows)
    rollout = Rollout.from_seed(model,source.typed.timing,[source.row(i) for i in range(source.seed_rows)],
                               {c:e for c,e in enumerate(initial.known_ends) if e is not None})
    def truth(_probabilities,_count,**_kwargs):
        return torch.tensor([model.choices.index(source.row(rollout.state.index).actions)])
    monkeypatch.setattr(torch,'multinomial',truth)
    rng = torch.Generator().manual_seed(13)
    for j,state in enumerate(batch.states):
        request=bank.query(torch.zeros(1,dtype=torch.long,device=device),
                           torch.tensor([state.replay.row_count],device=device))
        hands=model.readout(before[:,batch.positions[j]],like.new_tensor(query_features([state],source.typed)),seed,request)
        expected=model.decision_log_probs(hands,[state])[0]
        torch.testing.assert_close(expected,rollout.prediction()[1],atol=3e-5,rtol=3e-5)
        if j==35:
            saved=rollout.snapshot(rng)
            assert len(saved['memory_history'])>len(saved['history'])
            restored,other=Rollout.restore(model,source.typed.timing,saved)
            torch.testing.assert_close(restored.prediction()[1],rollout.prediction()[1],atol=0,rtol=0)
            assert torch.equal(other.get_state(),rng.get_state())
            bad=deepcopy(saved);bad['memory_history'].pop(0)
            with pytest.raises(ContractError,match='complete raw'):
                Rollout.restore(model,source.typed.timing,bad)
        rollout.step(rng)
    assert rollout.state.finished and len(rollout.memory_history)==len(source.rows)
    loss,_=batch_likelihood(model,batch)
    loss.backward()
    assert model.long_memory.encoder.weight_ih_l0.grad.norm()>0
    assert model.long_memory.output.weight.grad.norm()>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_ancient_content_changes_memory_without_changing_local_or_exact_features():
    actions=[(0,1,0,0)]*48
    actions[0]=(1,0,0,0);actions[8]=(1,1,1,1)
    changed=list(actions);changed[2]=(0,0,1,0)
    sources=[chart(a) for a in (actions,changed)]
    model=memory_model().double()
    batches=[prepare_batch([SourceInterval(s,30,4)],Arm.R1,7,full_history=True) for s in sources]
    for field in ('raw','query_features','seed_raw'):
        np.testing.assert_array_equal(getattr(batches[0],field),getattr(batches[1],field))
    a,b=[batch_likelihood(model,b)[0] for b in batches]
    assert abs(float((a-b).detach()))>1e-8
    full=prepare_batch([SourceInterval(sources[0],30,4)],Arm.R1,100,full_history=True)
    torch.testing.assert_close(a,batch_likelihood(model,full)[0],atol=1e-10,rtol=1e-10)


@pytest.mark.parametrize('device',['cpu','mps'])
def test_landmark_attention_never_reads_current_or_future_labels_and_preserves_mirror(device):
    if device=='mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    model=memory_model(device)
    source=chart([(1,0,0,0)]*24)
    batch=prepare_batch([SourceInterval(source,0,len(source.onsets))],Arm.R1,7,full_history=True)
    like=model.temporal.input.weight
    raw=like.new_tensor(batch.memory_raw);valid=torch.as_tensor(batch.memory_valid,device=device)
    onsets=torch.as_tensor(batch.memory_onsets,device=device)
    owners=torch.zeros(1,dtype=torch.long,device=device);rows=torch.tensor([11],device=device)
    hands=torch.randn(1,2,8,device=device)
    original=model.long_memory.encode(raw,valid,onsets)
    changed=raw.clone();changed[:,11:]=torch.randn_like(changed[:,11:])*3
    future=model.long_memory.encode(changed,valid,onsets)
    a=model.long_memory.read(hands,original.query(owners,rows))
    b=model.long_memory.read(hands,future.query(owners,rows))
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    mirrored=model.long_memory.encode(raw.flip(2),valid,onsets)
    other=model.long_memory.read(hands.flip(1),mirrored.query(owners,rows))
    torch.testing.assert_close(a.flip(1),other,atol=2e-6,rtol=2e-6)
    # No completed landmark exists before the fourth head; the residual is zero.
    early=original.query(owners,torch.tensor([3],device=device))
    torch.testing.assert_close(model.long_memory.read(hands,early),torch.zeros_like(hands),atol=0,rtol=0)


def test_long_memory_cannot_silently_fall_back_to_a_local_training_crop():
    model=memory_model()
    source=chart([(1,0,0,0)]*24)
    batch=prepare_batch([SourceInterval(source,12,4)],Arm.R1,7)
    with pytest.raises(ContractError,match='full histories'):
        batch_likelihood(model,batch)


def test_full_prefix_memory_microbatch_gradients_match_padded_batch():
    from pulsefield_model.research.bounded_typed_continuation.train_run import measure
    model=memory_model().double()
    source=chart([(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)]*15)
    intervals=[SourceInterval(source,8,3),SourceInterval(source,40,5)]
    batch=prepare_batch(intervals,Arm.R1,7,full_history=True)
    batch_likelihood(model,batch)[0].backward()
    expected=[None if p.grad is None else p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    for interval in intervals:
        measure(model,[interval],candidate_budget=8,backward=True,denominator=8,check=lambda _:None)
    for p,grad in zip(model.parameters(),expected):
        if grad is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad,grad,atol=1e-10,rtol=1e-8)
