from dataclasses import replace

import pytest
import torch

from ensomi_model.research.scoped_style_modeling.dataset import Interval, NoteRef
from ensomi_model.research.scoped_style_modeling.replay import prepare_chart
from ensomi_model.research.source_action_modeling.model import initialize_comparison
from ensomi_model.research.source_action_modeling.observation import observe_complete
from ensomi_model.research.source_action_modeling.representation import LEVELS, LOCAL_RADII, RepresentationBank, support_report
from ensomi_model.research.source_action_modeling.semantic_probe import initialize_readout
from ensomi_model.research.source_action_modeling.tensors import collate, collate_observations
from .conftest import example, fixture_chart

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_shared_initialization_all_levels_mirror_padding_and_both_heads(device):
    torch.manual_seed(12)
    rng = torch.random.get_rng_state()
    models = initialize_comparison()
    assert torch.equal(rng, torch.random.get_rng_state())
    h, full = models["composed_h"], models["composed_all"]
    assert sum(p.numel() for p in h.parameters()) == sum(p.numel() for p in full.parameters())
    assert all(torch.equal(value, full.state_dict()[key]) for key, value in h.state_dict().items())
    for model in models.values():
        for module in ("reader", "decoder"):
            assert all(torch.equal(v, getattr(h, module).state_dict()[k]) for k, v in getattr(model, module).state_dict().items())
    batch = collate([example(), example(long=True)]).to(device)
    mirror = collate([example(mirror=True), example(long=True, mirror=True)]).to(device)
    single = collate([example()]).to(device)
    for name, model in models.items():
        model.to(device).eval()
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, torch.nn.LayerNorm):
                    module.bias.fill_(.37)
        head = initialize_readout(model.access).to(device).eval()
        with torch.no_grad():
            a, b, one = (model.encoder(x.observation) for x in (batch, mirror, single))
            assert a.levels == (("H",) if name == "reference_h" else LEVELS)
            length = int(single.observation.lengths[0])
            for x, y, z in zip(a.states, b.states, one.states):
                torch.testing.assert_close(x.flip(2), y, atol=4e-6, rtol=1e-5)
                torch.testing.assert_close(x[0, :length], z[0], atol=3e-6, rtol=1e-5)
                assert not x[0, length:].count_nonzero()
            concepts, indices = torch.tensor([0, 4], device=device), torch.arange(2, device=device)
            torch.testing.assert_close(head(a, concepts, indices), head(b, concepts, indices), atol=3e-6, rtol=1e-5)
            torch.testing.assert_close(head(a, concepts, indices)[0], head(one, concepts[:1], indices[:1])[0], atol=3e-6, rtol=1e-5)
            out, mirrored = model(batch), model(mirror)
            permutation = torch.arange(256, device=device).reshape(16, 16).T.flatten()
            torch.testing.assert_close(out.log_probs[:, :, permutation], mirrored.log_probs, atol=5e-6, rtol=1e-5)
            torch.testing.assert_close(out.final_states.flip(1), mirrored.final_states, atol=3e-6, rtol=1e-5)
            torch.testing.assert_close(out.log_probs[0], model(single).log_probs[0], atol=4e-6, rtol=1e-5)


def test_intrinsic_and_local_states_exclude_nonlocal_facts_and_respect_exact_support():
    obs = observe_complete(fixture_chart(long=True))
    tensor = collate_observations([obs])
    encoder = initialize_comparison()["composed_all"].encoder.eval()
    a = encoder(tensor)
    lanes = tensor.lanes.clone()
    lanes[..., 4:] = torch.randn_like(lanes[..., 4:])
    changed_facts = encoder(replace(tensor, lanes=lanes, summaries=torch.randn_like(tensor.summaries)))
    for i in range(5):
        torch.testing.assert_close(a.states[i], changed_facts.states[i], atol=0, rtol=0)
    assert not torch.allclose(a.contextual, changed_facts.contextual)
    center = 18
    for level, radius in LOCAL_RADII.items():
        # Keep the complete supplied skeleton fixed while changing a farther action.
        outside = center + radius + 1
        rows = list(obs.rows)
        assert rows[outside].phase == "source"
        lane = 2 if rows[outside].actions[1] else 1
        rows[outside] = replace(rows[outside], actions=tuple(int(i == lane) for i in range(4)))
        b = encoder(collate_observations([replace(obs, rows=tuple(rows))]))
        index = LEVELS.index(level)
        torch.testing.assert_close(a.states[index][:, center], b.states[index][:, center], atol=0, rtol=0)


def test_order_composition_and_release_boundary_support_reporting():
    def chart(order):
        return prepare_chart(tuple(NoteRef(i + 1, lane, "normal", i * 100, i * 100) for i, lane in enumerate(order)),
                             Interval(0, 1000), Interval(0, 1000))
    charts = [chart((0, 1, 0, 1, 2, 3)), chart((0, 0, 1, 1, 2, 3))]
    encoder = initialize_comparison()["composed_all"].encoder.eval()
    banks = [encoder(collate_observations([observe_complete(c)])) for c in charts]
    assert not torch.allclose(banks[0].states[1][:, 2:5], banks[1].states[1][:, 2:5])
    complete = fixture_chart(long=True)
    report = support_report(example(long=True).observation, complete_chart=complete)
    for level, radius in LOCAL_RADII.items():
        assert report[level][18]["timeline_rows"] == 2 * radius + 1
    release = prepare_chart((NoteRef(1, 0, "long", 0, 100),), Interval(0, 200), Interval(0, 200))
    release_report = support_report(observe_complete(release), complete_chart=release)
    assert any(row["source_events"] > row["attack_group_span"] for row in release_report["L3"])
    assert all(row["attack_group_span"] is None for row in support_report(example().observation)["U"])
    assert report["H"][0]["timeline_rows"] == len(complete.inputs.rows)


@pytest.mark.parametrize("access", ["H", "all"])
@pytest.mark.parametrize("device", DEVICES)
def test_direct_reads_reach_visible_early_states_and_composition_carries_prediction_gradients(access, device):
    model = initialize_comparison()["composed_all" if access == "all" else "composed_h"].to(device)
    batch = collate([example()]).to(device)
    bank = model.encoder(batch.observation)
    # Independent leaves sever composition, so their gradients measure direct reads.
    leaves = tuple(s.detach().requires_grad_() for s in bank.states)
    direct = RepresentationBank(bank.levels, leaves, bank.observation)
    model.decoder(model.reader(direct, batch.queries, access=access), batch.queries, batch.targets).loss.backward()
    visible = batch.observation.lanes[..., 3].bool().all((-1, -2))
    for state in leaves[:-1]:
        if access == "all":
            assert state.grad is not None and state.grad[visible].abs().sum() > 0
        else:
            assert state.grad is None
    assert leaves[-1].grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    model(batch).loss.backward()
    for component in (model.encoder.source, *model.encoder.local, model.encoder.relations, model.encoder.hand):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in component.parameters())
        assert any(p.grad.abs().sum() > 0 for p in component.parameters())


def test_action_reader_never_reads_teacher_forced_targets_and_empty_section_head_is_finite():
    model = initialize_comparison()["composed_all"].eval()
    a, b = collate([example()]), collate([example(changed=True)])
    assert not torch.equal(a.targets, b.targets)
    x, y = model.encoder(a.observation), model.encoder(b.observation)
    torch.testing.assert_close(model.reader(x, a.queries, access="all"), model.reader(y, b.queries, access="all"), atol=0, rtol=0)
    empty = prepare_chart((), Interval(0, 100), Interval(0, 100))
    bank = model.encoder(collate_observations([observe_complete(empty)]))
    logits = initialize_readout("all")(bank, torch.arange(5), torch.zeros(5, dtype=torch.long))
    assert logits.shape == (5, 3) and torch.isfinite(logits).all()
