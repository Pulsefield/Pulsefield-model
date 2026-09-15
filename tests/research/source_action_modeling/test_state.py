from dataclasses import replace
from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from pulsefield_model.research.scoped_style_modeling.dataset import ContractError, Interval, NoteRef
from pulsefield_model.research.scoped_style_modeling.replay import prepare_chart
from pulsefield_model.research.source_action_modeling.checkpoint import load_snapshot, save_snapshot
from pulsefield_model.research.source_action_modeling.diagnostics import capture_response, finish_response
from pulsefield_model.research.source_action_modeling.model import ModelConfig, initialize_model
from pulsefield_model.research.source_action_modeling.observation import EventBlock, observe
from pulsefield_model.research.source_action_modeling.sampling import BlockSampler, TrainingContext
from pulsefield_model.research.source_action_modeling.tensors import collate
from .conftest import example, fixture_chart


def contexts():
    return [TrainingContext("group-a", "a", fixture_chart()), TrainingContext("group-b", "b", fixture_chart(long=True))]


def test_sampling_reproducible_label_free_feasible_scales_and_position():
    sampler = BlockSampler(contexts())
    sampler.draw(3)
    state = sampler.state_dict()
    samples, records = sampler.draw(50)
    other = BlockSampler(contexts()[::-1], seed=999)
    other.load_state_dict(state)
    again, repeat = other.draw(50)
    assert samples == again and records == repeat
    assert {r["rows"] for r in records} == {4, 16}
    assert {r["group_id"] for r in records} == {"group-a", "group-b"}
    assert other.position == 53
    for record in records:
        context = next(c for c in sampler.contexts if c.key == record["context_key"])
        expected = 1 / (2 * len([s for s in (4, 16, 64) if s <= context.event_count]) *
                        (context.event_count - record["rows"] + 1))
        assert record["sampling_probability"] == expected
    assert all(r["rows"] <= (9 if r["context_key"] == "a" else 34) for r in records)
    with pytest.raises(ContractError, match="population"):
        BlockSampler(contexts()[:1]).load_state_dict(state)
    changed = [replace(contexts()[0], chart=fixture_chart(changed=True)), contexts()[1]]
    # Same skeleton and identities give exactly the same event-block draws even
    # when targets change. Real source changes require a new identity upstream.
    base_records = BlockSampler(contexts()).draw(20)[1]
    changed_records = BlockSampler(changed).draw(20)[1]
    for a, b in zip(base_records, changed_records):
        assert {k: v for k, v in a.items() if k != "attack_group_span"} == {
            k: v for k, v in b.items() if k != "attack_group_span"}


def test_64_row_scale_retains_release_only_targets_and_trains():
    objects = tuple(NoteRef(i + 1, i % 4, "long", i * 200, i * 200 + 100) for i in range(64))
    chart = prepare_chart(objects, Interval(0, 12800), Interval(0, 12800))
    samples, records = BlockSampler([TrainingContext("group", "128-event-context", chart)]).draw(30)
    assert {r["rows"] for r in records} == {4, 16, 64}
    item = observe(chart, EventBlock(1, 64), entering_occupancy=(False,) * 4)
    assert item.targets[0] == (3, 0, 0, 0) and item.attack_group_span == 32
    model = initialize_model()
    output = model(collate([item]))
    output.loss.backward()
    assert output.row_nll.shape == (1, 64) and torch.isfinite(output.row_nll).all()


def test_sampling_probability_accounts_for_unequal_context_counts_per_group():
    population = [TrainingContext("a", "short", fixture_chart()), TrainingContext("a", "long", fixture_chart(long=True)),
                  TrainingContext("b", "other", fixture_chart())]
    records = BlockSampler(population).draw(40)[1]
    assert {r["context_key"] for r in records} == {"short", "long", "other"}
    for record in records:
        count = 34 if record["context_key"] == "long" else 9
        contexts_in_group = 2 if record["group_id"] == "a" else 1
        scales = 2 if count == 34 else 1
        assert record["sampling_probability"] == 1 / (2 * contexts_in_group * scales * (count - record["rows"] + 1))


@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []))
def test_snapshot_reproduces_probabilities_rng_sampling_and_next_adamw_update(tmp_path, device):
    model = initialize_model(ModelConfig(dropout=0.1)).to(device)
    sampler = BlockSampler(contexts())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    def update(model, optimizer, scheduler, sampler):
        draws = (random.random(), np.random.random(), torch.rand(2), torch.rand(2, device=device).cpu())
        items, record = sampler.draw(2)
        optimizer.zero_grad(set_to_none=True)
        loss = model(collate(items).to(device)).loss
        loss.backward()
        gradients = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()}
        optimizer.step()
        scheduler.step()
        return draws, record, float(loss.detach()), gradients

    update(model, optimizer, scheduler, sampler)
    path = tmp_path / "snapshot.pt"
    save_snapshot(path, model, optimizer, sampler, update=1, scheduler=scheduler)
    saved_state = {n: v.detach().clone() for n, v in model.state_dict().items()}
    saved_optimizer = deepcopy(optimizer.state_dict())
    with pytest.raises(FileExistsError):
        save_snapshot(path, model, optimizer, sampler, update=1, scheduler=scheduler)
    model.eval()
    fixed = collate([example()]).to(device)
    with torch.no_grad():
        expected = model(fixed).log_probs.clone()
    model.train()
    first = update(model, optimizer, scheduler, sampler)
    final = {n: p.detach().clone() for n, p in model.named_parameters()}
    restored = initialize_model(ModelConfig(dropout=0.1), seed=99).to(device)
    new_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.1)
    new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=1, gamma=0.9)
    new_sampler = BlockSampler(contexts(), seed=99)
    assert load_snapshot(path, restored, new_optimizer, new_sampler, scheduler=new_scheduler) == 1
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in saved_state.items())
    loaded_optimizer = new_optimizer.state_dict()
    assert loaded_optimizer["param_groups"] == saved_optimizer["param_groups"]
    for key, state in saved_optimizer["state"].items():
        for name, value in state.items():
            assert torch.equal(value.cpu(), loaded_optimizer["state"][key][name].cpu())
    restored.eval()
    with torch.no_grad():
        # Distinct MPS modules with identical weights also differ by 1 ULP
        # without any checkpoint I/O. State equality remains exact above.
        torch.testing.assert_close(restored(fixed).log_probs, expected,
                                   rtol=3e-6 if device == "mps" else 0, atol=2e-6 if device == "mps" else 0)
    restored.train()
    second = update(restored, new_optimizer, new_scheduler, new_sampler)
    assert first[0][:2] == second[0][:2] and first[1] == second[1]
    assert torch.equal(first[0][2], second[0][2]) and torch.equal(first[0][3], second[0][3])
    assert first[2] == pytest.approx(second[2], abs=3e-6)
    assert scheduler.state_dict() == new_scheduler.state_dict()
    for name, gradient in first[3].items():
        torch.testing.assert_close(second[3][name], gradient, atol=2e-7 if device == "mps" else 0,
                                   rtol=3e-5 if device == "mps" else 0)
    # Near-zero softmax-bias gradients differ by ~1e-10 on MPS; AdamW's
    # epsilon can amplify that to several e-6 in unidentifiable parameters.
    # Check every resumed update against an independent CPU optimizer using
    # the original in-memory moments and the observed accelerator gradients.
    oracle = initialize_model(ModelConfig(dropout=0.1))
    oracle.load_state_dict(saved_state)
    oracle_optimizer = torch.optim.AdamW(oracle.parameters())
    oracle_optimizer.load_state_dict(saved_optimizer)
    for name, parameter in oracle.named_parameters():
        parameter.grad = second[3][name]
    oracle_optimizer.step()
    for n, p in restored.named_parameters():
        if device == "cpu":
            torch.testing.assert_close(p, final[n], atol=0, rtol=0)
        torch.testing.assert_close(p.detach().cpu(), dict(oracle.named_parameters())[n], atol=2e-7, rtol=3e-6, msg=n)
    model.eval()
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(fixed).log_probs, model(fixed).log_probs, atol=3e-6, rtol=3e-6)
    with pytest.raises(ContractError, match="scheduler"):
        load_snapshot(path, restored, new_optimizer, new_sampler)


def test_actual_update_diagnostic_has_disjoint_ownership_and_checked_residual():
    batch = collate([example()])
    model = initialize_model(ModelConfig(dropout=0.2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    before = capture_response(model, batch)
    assert model.training and all(p.grad is None for p in model.parameters())
    unchanged = finish_response(before, model, batch)
    assert unchanged["actual_change"] == 0 and unchanged["linearization_residual"] == 0
    model(batch).loss.backward()
    optimizer.step()
    result = finish_response(before, model, batch)
    assert model.training
    assert set(result["gradient_dot_displacement"]) == {"encoder", "decoder"}
    assert all(v > 0 for v in result["parameter_displacement_l2"].values())
    assert result["linearized_change"] == sum(result["gradient_dot_displacement"].values())
    assert abs(result["linearization_residual"]) < 0.05 * abs(result["actual_change"])
    with pytest.raises(ContractError, match="identical"):
        finish_response(before, model, collate([example(changed=True)]))


@pytest.mark.parametrize("mismatch", ["schema", "decoder", "loss"])
def test_snapshot_rejects_old_decoder_or_risk_before_mutating_state(tmp_path, mismatch):
    model = initialize_model()
    sampler = BlockSampler(contexts())
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "snapshot.pt"
    save_snapshot(path, model, optimizer, sampler, update=0)
    payload = torch.load(path, weights_only=False)
    if mismatch == "schema":
        payload["schema"] = 2
    else:
        payload["model_policy"][mismatch] = "incompatible"
    torch.save(payload, path)
    state, rng, sampling = deepcopy(model.state_dict()), torch.random.get_rng_state(), sampler.state_dict()
    with pytest.raises(ContractError, match="contract"):
        load_snapshot(path, model, optimizer, sampler)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())
    assert torch.equal(rng, torch.random.get_rng_state()) and sampling == sampler.state_dict()
