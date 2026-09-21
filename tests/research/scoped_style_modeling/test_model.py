from dataclasses import replace

import pytest
import torch

from pulsefield_model.research.scoped_style_modeling.dataset import ContractError, Interval, NoteRef
from pulsefield_model.research.scoped_style_modeling.model import ModelConfig, initialize_model, losses
from pulsefield_model.research.scoped_style_modeling.relations import prepare_relations
from pulsefield_model.research.scoped_style_modeling.replay import prepare_chart, mirror_objects, evidence_masks
from pulsefield_model.research.scoped_style_modeling.tensors import Example, collate, exact_history

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture(mirror=False, empty=False, long=False):
    objects = (NoteRef(10, 0, "long", 0, 900), NoteRef(11, 1, "long", 20, 100),
               NoteRef(12, 2, "normal", 100, 100), NoteRef(13, 3, "long", 180, 250),
               NoteRef(14, 2, "normal", 400, 400), NoteRef(15, 3, "normal", 480, 480))
    if long:
        objects += tuple(NoteRef(100+i, i % 4, "normal", 1000+i*100, 1000+i*100) for i in range(8))
    if mirror:
        objects = mirror_objects(objects)
    scope = Interval(100, 1900 if long else 400)
    chart = prepare_chart(objects if not empty else (), scope, Interval(50, 2000 if long else 500))
    masks = evidence_masks(chart, () if empty else (objects[0], objects[2], objects[3]))
    return Example(chart, prepare_relations(chart), 0, 1, masks)


def test_forward_masks_normalization_and_empty_missing():
    item = fixture()
    batch = collate([item, replace(item, masks=None), replace(item, masks=(0,)*len(item.masks)), fixture(empty=True)])
    model = initialize_model(ModelConfig(), 17, auxiliary=True).eval()
    result = losses(model, batch, 0.1)
    aux = result.auxiliary
    assert torch.isfinite(result.total)
    torch.testing.assert_close(result.logits.softmax(-1).sum(-1), torch.ones(4))
    torch.testing.assert_close(aux.log_probs.exp().sum(-1), torch.ones_like(batch.selector.times))
    assert (aux.log_probs[~batch.selector.valid_masks] == -torch.inf).all()
    assert aux.eligible.tolist() == [True, False, True, False]
    assert aux.normalized_nll[2] > 0
    assert aux.normalized_nll[[1, 3]].tolist() == [0, 0]
    forced = batch.selector.valid_masks.sum(-1) == 1
    assert (aux.log_probs[:, :, 0][forced] == 0).all()
    assert batch.chart.section_events[-1].sum() == 0
    with pytest.raises(ContractError, match="Impossible"):
        collate([replace(item, masks=(15,)*len(item.masks))])


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("device", DEVICES)
def test_padded_batch_matches_single_valid_states_assessment_and_selector(empty, device):
    item = fixture(empty=empty)
    single, padded = collate([item]), collate([item, fixture(long=True)])
    single, padded = single.to(device), padded.to(device)
    model = initialize_model(ModelConfig(), 17, auxiliary=True).to(device).eval()
    with torch.no_grad():
        hs, hp = model.encoder(single.chart), model.encoder(padded.chart)
        torch.testing.assert_close(hs[0], hp[0, :hs.shape[1]], atol=2e-6, rtol=2e-5)
        assert hp[0, hs.shape[1]:].count_nonzero() == 0
        rs, rp = losses(model, single, 0.1), losses(model, padded, 0.1)
        torch.testing.assert_close(rs.logits[0], rp.logits[0], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(rs.auxiliary.log_probs[0], rp.auxiliary.log_probs[0, :single.selector.indices.shape[1]], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(rs.auxiliary.final_states[0], rp.auxiliary.final_states[0], atol=2e-6, rtol=2e-5)


def test_target_history_and_selector_removal_cannot_change_assessment():
    item = fixture()
    batch = collate([item])
    changed = collate([replace(item, assessment=2, masks=(0,)*len(item.masks))])
    model = initialize_model(ModelConfig(), 17, auxiliary=True).eval()
    h = model.encoder(batch.chart)
    expected = model.assessment(batch.chart, batch.concepts)
    first = losses(model, batch, 0.1)
    second = losses(model, changed, 0.1)
    assert torch.equal(h, model.encoder(changed.chart))
    assert torch.equal(first.logits, second.logits)
    assert not torch.equal(first.auxiliary.final_states, second.auxiliary.final_states)
    history = replace(changed.selector, history=torch.randn_like(changed.selector.history))
    model.selector(h, changed.concepts, changed.assessments, history)
    assert torch.equal(expected, model.assessment(batch.chart, batch.concepts))
    model.selector = None
    assert torch.equal(expected, model.assessment(batch.chart, batch.concepts))


@pytest.mark.parametrize("device", DEVICES)
def test_mirror_assessment_per_step_distribution_and_sequence_score(device):
    a, b = collate([fixture()]).to(device), collate([fixture(mirror=True)]).to(device)
    model = initialize_model(ModelConfig(), 17, auxiliary=True).to(device).eval()
    ha, hb = model.encoder(a.chart), model.encoder(b.chart)
    torch.testing.assert_close(ha.flip(2), hb, atol=2e-6, rtol=2e-5)
    x, y = losses(model, a, 0.1), losses(model, b, 0.1)
    torch.testing.assert_close(x.logits, y.logits, atol=2e-6, rtol=2e-5)
    mapping = [sum(((m >> i) & 1) << (3-i) for i in range(4)) for m in range(16)]
    torch.testing.assert_close(x.auxiliary.log_probs, y.auxiliary.log_probs[:, :, mapping], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(x.auxiliary.sequence_nll, y.auxiliary.sequence_nll)
    torch.testing.assert_close(x.auxiliary.final_states.flip(1), y.auxiliary.final_states)


def test_gradients_reach_only_encoder_and_own_branch():
    batch = collate([fixture()])
    model = initialize_model(ModelConfig(), 17, auxiliary=True)
    result = losses(model, batch, 0.1)
    result.assessment.backward(retain_graph=True)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.selector.parameters())
    model.zero_grad(set_to_none=True)
    result.evidence.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.selector.parameters())
    assert all(p.grad is None for p in model.assessor.parameters())


@pytest.mark.parametrize("device", DEVICES)
def test_beta_zero_update_matches_assessment_only_with_dropout(device):
    config = ModelConfig()
    base = initialize_model(config, 17, auxiliary=False).to(device)
    aux = initialize_model(config, 17, auxiliary=True).to(device)
    batch = collate([fixture(), fixture(long=True)]).to(device)
    outputs, gradients = [], []
    for model in (base, aux):
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        torch.manual_seed(91)
        result = losses(model, batch, 0)
        outputs.append(result.logits.detach().cpu())
        result.total.backward()
        gradients.append({name: p.grad.detach().cpu().clone() for name, p in model.named_parameters() if p.grad is not None})
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    torch.testing.assert_close(outputs[0], outputs[1], atol=2e-6, rtol=2e-5)
    for name, gradient in gradients[0].items():
        torch.testing.assert_close(gradient, gradients[1][name], atol=2e-7, rtol=2e-5)
    for name, parameter in base.named_parameters():
        other = dict(aux.named_parameters())[name]
        if device == "cpu":
            assert torch.equal(parameter, other), name
        else:
            # Repeated identical MPS baselines differ by ~2e-6 after AdamW,
            # including with deterministic algorithms enabled. Constrain
            # forward/gradient agreement above before allowing update roundoff.
            torch.testing.assert_close(parameter, other, atol=5e-6, rtol=1e-5, msg=name)
    assert all(p.grad is None for p in aux.selector.parameters())


def test_initialization_preserves_caller_rng_and_matching_components():
    state = torch.random.get_rng_state().clone()
    base = initialize_model(ModelConfig(), 17, auxiliary=False)
    aux = initialize_model(ModelConfig(), 17, auxiliary=True)
    assert torch.equal(state, torch.random.get_rng_state())
    for name, parameter in base.named_parameters():
        assert torch.equal(parameter, dict(aux.named_parameters())[name])


def test_exact_history_boundary_and_release():
    item = fixture()
    history = torch.tensor(exact_history(item.chart, item.masks))
    assert history[0].count_nonzero() == 0
    # Entering selection changes active-hold status, without a synthetic attack.
    assert history[1, 0, 0, :2].tolist() == [0, 0]
    assert history[1, 0, 0, 4:].tolist() == [1, 1]
    release = next(i for i, d in enumerate(item.chart.decisions) if item.chart.inputs.rows[d.encoder_index].time_ms == 250)
    assert history[release, 1, 0, 4:].tolist() == [1, 1]


def test_simultaneous_memory_update_uses_prior_other_hand():
    model = initialize_model(ModelConfig(), 17, auxiliary=True)
    selector = model.selector
    c, prior = torch.randn(2, 2, 64), torch.randn(2, 2, 64)
    chosen, condition, dt = torch.tensor([3, 12]), torch.randn(2, 24), torch.rand(2)
    captured = []
    hook = selector.memory.register_forward_pre_hook(lambda module, args: captured.append(args))
    selector.advance(c, prior, chosen, condition, dt)
    hook.remove()
    inputs, hidden = captured[0]
    torch.testing.assert_close(hidden, prior.flatten(0, 1))
    torch.testing.assert_close(inputs[:, 64:128], prior.flip(1).flatten(0, 1))


def test_tiny_slice_overfit_both_declared_objectives():
    batch = collate([fixture(), replace(fixture(long=True), concept=1, assessment=2),
                     replace(fixture(empty=True), concept=2, assessment=0)])
    model = initialize_model(ModelConfig(dropout=0), 17, auxiliary=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    initial = losses(model, batch, 0.1)
    initial_a, initial_e = initial.assessment.item(), initial.evidence.item()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        result = losses(model, batch, 0.1)
        result.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
    final = losses(model, batch, 0.1)
    assert final.assessment < initial_a*0.1
    assert final.evidence < initial_e*0.2
