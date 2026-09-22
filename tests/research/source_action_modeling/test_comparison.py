from copy import deepcopy
from dataclasses import fields, replace

import pytest
import torch

from ensomi_model.research.scoped_style_modeling.dataset import ContractError, Interval, NoteRef
from ensomi_model.research.scoped_style_modeling.replay import prepare_chart
from ensomi_model.research.source_action_modeling.checkpoint import load_snapshot, save_snapshot
from ensomi_model.research.source_action_modeling.comparison import evaluate_structure, paired_batches, structural_report, train_paired_step
from ensomi_model.research.source_action_modeling.diagnostics import capture_response, finish_response
from ensomi_model.research.source_action_modeling.model import ModelConfig, initialize_comparison
from ensomi_model.research.source_action_modeling.observation import EventBlock, ViewPolicy, observe, paired_views, visible_states
from ensomi_model.research.source_action_modeling.sampling import PairedBlockSampler, TrainingContext, VIEWS
from ensomi_model.research.source_action_modeling.semantic_probe import initialize_readout
from ensomi_model.research.source_action_modeling.tensors import collate, observation_relations
from .conftest import example, fixture_chart

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (["cuda"] if torch.cuda.is_available() else [])


def contexts():
    chart = prepare_chart(tuple(NoteRef(i + 1, i % 4, "normal", i * 100, i * 100) for i in range(80)),
                          Interval(0, 8000), Interval(0, 8000))
    return [TrainingContext("a", "short", fixture_chart()), TrainingContext("b", "long", chart)]


def test_paired_views_hide_extra_context_share_only_permitted_summaries_and_legality():
    policy = ViewPolicy(0)
    views = paired_views(example(start=5), policy)
    batches = paired_batches([views])
    assert views["near"].observation.rows == views["coarse"].observation.rows
    assert not views["near"].observation.summaries
    assert views["detailed"].observation.summaries == views["coarse"].observation.summaries
    assert (batches["near"].queries.entering_occupancy == -1).all()
    assert visible_states(views["detailed"].observation)[2] != visible_states(views["near"].observation)[2]
    for view, batch in batches.items():
        assert torch.equal(batch.targets, batches["near"].targets)
        assert torch.equal(batch.queries.entering_occupancy, batches["near"].queries.entering_occupancy)
        assert torch.equal(batch.observation.times_ms, batches["near"].observation.times_ms)
    for view in ("near", "coarse"):
        kinds = {k for _, _, kinds in observation_relations(views[view].observation) for k, *_ in kinds}
        assert not kinds & {"ln_identity", "recurrence", "attack_1", "attack_2"}
    changed = paired_views(example(changed=True), policy)
    original = paired_views(example(), policy)
    for view in VIEWS:
        a, b = collate([original[view]]), collate([changed[view]])
        for f in fields(a.observation):
            assert torch.equal(getattr(a.observation, f.name), getattr(b.observation, f.name)), (view, f.name)
    bad = dict(views)
    bad["detailed"] = replace(bad["detailed"], query_entering_occupancy=(False,) * 4)
    with pytest.raises(ContractError, match="prefix"):
        paired_batches([bad])


def test_coarse_forgets_far_order_but_retains_equal_counts_and_known_occupation():
    def draw(order):
        chart = prepare_chart(tuple(NoteRef(i + 1, lane, "normal", i * 100, i * 100) for i, lane in enumerate(order)),
                              Interval(0, 1000), Interval(0, 1000))
        return paired_views(observe(chart, EventBlock(4, 2), entering_occupancy=(False,) * 4), ViewPolicy(0))
    a, b = draw((0, 1, 0, 1, 2, 3, 2, 3)), draw((0, 0, 1, 1, 2, 3, 2, 3))
    assert a["detailed"].observation != b["detailed"].observation
    assert a["coarse"].observation == b["coarse"].observation
    assert a["near"].observation == b["near"].observation


@pytest.mark.parametrize("device", DEVICES)
def test_common_updates_and_structural_raw_losses_scales_durations_and_group_uncertainty(device):
    models = {name: model.to(device) for name, model in initialize_comparison().items()}
    optimizers = {name: torch.optim.AdamW(m.parameters(), lr=1e-4) for name, m in models.items()}
    sampler = PairedBlockSampler(contexts(), policy=ViewPolicy(1))
    record = train_paired_step(models, optimizers, sampler, blocks=2)
    assert set(record["view_sha256"]) == set(VIEWS)
    assert all(r["views"] == list(VIEWS) for r in record["blocks"])
    assert all(torch.isfinite(torch.tensor(r["loss"])) and r["update_seconds"] > 0 for r in record["models"].values())
    paired, records = sampler.draw(6)
    result = evaluate_structure(models, paired, records, batch_size=2, bootstrap_samples=40)
    assert all(m.training for m in models.values())
    assert len(result["paired_detailed_mean_row_nll"]) == 3
    assert len(result["paired_detailed_sequence_nll"]) == 3
    for name, report in result["models"].items():
        assert report["max_retained_bank_bytes"] > 0
        for row in report["blocks"]:
            assert row["near_minus_detailed_mean_row_nll"] == row["near_mean_row_nll"] - row["detailed_mean_row_nll"]
            assert row["near_minus_detailed_sequence_nll"] == row["near_sequence_nll"] - row["detailed_sequence_nll"]
            assert row["detailed_sequence_nll"] == sum(row["detailed_row_nll"])
            assert row["detailed_sequence_nll"] / row["rows"] == row["detailed_mean_row_nll"]
            assert row["coarse_minus_detailed_first_nll"] == row["coarse_first_nll"] - row["detailed_first_nll"]
            assert len(row["detailed_row_nll"]) == row["rows"]
        assert set(report["by_scale"]) == {"4", "16", "64"}
        first = report["by_decoder_position"]["0"]["detailed_nll"]
        assert first["group_means"] == report["overall"]["detailed_first_nll"]["group_means"]
    assert result["models"]["reference_h"]["batch_sha256"] == result["models"]["composed_all"]["batch_sha256"]
    mismatched = [dict(r, view_policy={**r["view_policy"], "near_radius": 0}) for r in records]
    with pytest.raises(ContractError, match="recorded visibility"):
        evaluate_structure(models, paired, mismatched)
    rows = [{"group_id": "a", "rows": 4, "duration_ms": 100, "near_mean_row_nll": 1}] * 9
    rows += [{"group_id": "b", "rows": 4, "duration_ms": 100, "near_mean_row_nll": 5}]
    estimate = structural_report(rows, bootstrap_samples=100)["overall"]["near_mean_row_nll"]
    assert estimate["mean"] == 3 and estimate["group_bootstrap_95"] is not None


def test_bank_likelihood_diagnostic_includes_reader_and_checks_policy():
    model = initialize_comparison()["composed_all"]
    batch = collate([example()])
    before = capture_response(model, batch)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    model(batch).loss.backward()
    optimizer.step()
    result = finish_response(before, model, batch)
    assert set(result["gradient_dot_displacement"]) == {"encoder", "reader", "decoder"}
    assert abs(result["linearization_residual"]) < .05 * abs(result["actual_change"])
    model.access = "H"
    with pytest.raises(ContractError, match="policy"):
        finish_response(before, model, batch)


def test_extended_snapshot_resumes_next_paired_draw_dropout_update_and_fixed_readout(tmp_path):
    config = ModelConfig(dropout=.1)
    model = initialize_comparison(config)["composed_all"]
    sampler = PairedBlockSampler(contexts(), policy=ViewPolicy(0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, .9)
    readout = initialize_readout("all")
    readout.fitted = True
    def update(m, opt, sch, draws):
        pairs, records = draws.draw(2)
        batch = collate([p[v] for p in pairs for v in VIEWS])
        opt.zero_grad(set_to_none=True)
        output = m(batch)
        output.loss.backward()
        opt.step()
        sch.step()
        return records, output.log_probs.detach().clone()
    update(model, optimizer, scheduler, sampler)
    path = tmp_path / "bank.pt"
    saved = deepcopy(model.state_dict())
    save_snapshot(path, model, optimizer, sampler, update=1, scheduler=scheduler, readout=readout)
    expected_records, expected_probs = update(model, optimizer, scheduler, sampler)
    restored = initialize_comparison(config, seed=99)["composed_all"]
    new_optimizer = torch.optim.AdamW(restored.parameters(), lr=.1)
    new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, 1, .5)
    new_sampler = PairedBlockSampler(contexts(), seed=99, policy=ViewPolicy(0))
    new_head = initialize_readout("all", seed=99)
    assert load_snapshot(path, restored, new_optimizer, new_sampler, scheduler=new_scheduler, readout=new_head) == 1
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in saved.items())
    assert new_head.fitted and all(torch.equal(v, new_head.state_dict()[k]) for k, v in readout.state_dict().items())
    actual_records, actual_probs = update(restored, new_optimizer, new_scheduler, new_sampler)
    assert actual_records == expected_records
    torch.testing.assert_close(actual_probs, expected_probs, atol=0, rtol=0)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    assert new_scheduler.state_dict() == scheduler.state_dict()
    wrong_views = PairedBlockSampler(contexts(), policy=ViewPolicy(1))
    with pytest.raises(ContractError, match="view policy"):
        load_snapshot(path, restored, new_optimizer, wrong_views, scheduler=new_scheduler, readout=new_head)
    with pytest.raises(ContractError, match="readout"):
        load_snapshot(path, restored, new_optimizer, new_sampler, scheduler=new_scheduler)
    restored.access = "H"
    with pytest.raises(ContractError, match="contract"):
        load_snapshot(path, restored, new_optimizer, new_sampler, scheduler=new_scheduler, readout=new_head)
