from copy import deepcopy
from dataclasses import fields, replace

import pytest
import torch

from ensomi_model.research.scoped_style_modeling.dataset import ContractError, Interval, NoteRef
from ensomi_model.research.scoped_style_modeling.replay import prepare_chart
from ensomi_model.research.source_action_modeling.checkpoint import load_snapshot, save_snapshot
from ensomi_model.research.source_action_modeling.comparison import evaluate_structure
from ensomi_model.research.source_action_modeling.local_representation import (
    ConditionedLocalBlock, LocalRepresentationConfig, local_support_report, source_packet)
from ensomi_model.research.source_action_modeling.model import ModelConfig, initialize_comparison
from ensomi_model.research.source_action_modeling.observation import ObservedRow, observe_complete, paired_views
from ensomi_model.research.source_action_modeling.representation_experiments import (
    initialize_representation_comparison, representation_arms, train_representation_step)
from ensomi_model.research.source_action_modeling.sampling import PairedBlockSampler, TrainingContext
from ensomi_model.research.source_action_modeling.semantic_probe import initialize_readout
from ensomi_model.research.source_action_modeling.tensors import collate, collate_observations
from ensomi_model.research.source_action_modeling.time_basis import event_geometry, time_basis, time_basis_dim
from .conftest import example, fixture_chart

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (["cuda"] if torch.cuda.is_available() else [])


def test_physical_basis_has_smooth_finite_gradients_signed_direction_and_unaliased_tail():
    times = torch.tensor([-1e7, -125., -8., 0., 8., 125., 1e7], requires_grad=True)
    basis = time_basis(times, "smooth")
    assert basis.shape == (7, time_basis_dim("smooth"))
    torch.testing.assert_close(basis[:, :2], time_basis(times, "scalar"))
    torch.testing.assert_close(basis[:, :12], -basis.flip(0)[:, :12])
    torch.testing.assert_close(basis[:, 12:], basis.flip(0)[:, 12:])
    assert basis[4, 2] - basis[3, 2] > .7
    assert basis[-1, 0] == 10000 and basis[-1, 1] > basis[-2, 1]
    gradient = torch.autograd.grad(basis.sum(), times)[0]
    assert torch.isfinite(basis).all() and torch.isfinite(gradient).all()
    with pytest.raises(ContractError, match="basis"):
        time_basis(times, "unknown")


def test_event_geometry_ratio_scale_invariance_availability_and_release_events():
    observation = collate_observations([observe_complete(fixture_chart(long=True))])
    original = event_geometry(observation)
    faster = event_geometry(replace(observation, times_ms=observation.times_ms / 2))
    torch.testing.assert_close(original.row_time("smooth")[..., -4:], faster.row_time("smooth")[..., -4:])
    for offset in (-2, -1, 0, 1, 2):
        torch.testing.assert_close(original.pair_time(offset, "smooth")[..., -4:],
                                   faster.pair_time(offset, "smooth")[..., -4:], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(original.row_time("smooth")[..., :-4], faster.row_time("smooth")[..., :-4])
    source = observation.rows[..., 5].bool()
    assert original.valid.sum() == source.sum()
    release = prepare_chart((NoteRef(1, 0, "long", 0, 125),), Interval(0, 200), Interval(0, 200))
    release_tensor = collate_observations([observe_complete(release)])
    release_geometry = event_geometry(release_tensor)
    actions = release_geometry.gather(release_tensor.lanes)
    assert (release_geometry.valid & actions[..., 2].any((-1, -2)) & ~actions[..., :2].any((-1, -2, -3))).any()
    assert not original.gap_available[0, 0, 0]
    assert not original.row_time("smooth")[~original.valid].count_nonzero()


def test_comparison_keeps_baseline_exact_and_new_module_initialization_independent_of_arms():
    torch.manual_seed(221)
    rng = torch.random.get_rng_state()
    arms = representation_arms()
    models = initialize_representation_comparison(arms)
    assert torch.equal(rng, torch.random.get_rng_state())
    baseline = initialize_comparison()["composed_all"]
    assert models["baseline"].state_dict().keys() == baseline.state_dict().keys()
    assert all(torch.equal(v, models["baseline"].state_dict()[k]) for k, v in baseline.state_dict().items())
    batch = collate([example()])
    torch.testing.assert_close(models["baseline"](batch).log_probs, baseline(batch).log_probs, atol=0, rtol=0)
    single = initialize_representation_comparison({"combined": arms["combined"]})["combined"]
    for name, value in models["combined"].state_dict().items():
        assert torch.equal(value, single.state_dict()[name])
        for model in models.values():
            state = model.state_dict()
            if name in state and state[name].shape == value.shape:
                assert torch.equal(value, state[name]), name


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("arm", ["event", "scalar", "smooth", "row", "time", "row_time", "combined", "state"])
def test_interventions_preserve_mirror_padding_hidden_targets_and_gradients(device, arm):
    model = initialize_representation_comparison({arm: representation_arms()[arm]})[arm].to(device).eval()
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.bias.fill_(.37)
    batch = collate([example(), example(long=True)]).to(device)
    mirrored = collate([example(mirror=True), example(long=True, mirror=True)]).to(device)
    single = collate([example()]).to(device)
    with torch.no_grad():
        a, b, one = [model.encoder(x.observation) for x in (batch, mirrored, single)]
        length = int(single.observation.lengths[0])
        for x, y, z in zip(a.states, b.states, one.states):
            torch.testing.assert_close(x.flip(2), y, atol=6e-6, rtol=2e-5)
            torch.testing.assert_close(x[0, :length], z[0], atol=6e-6, rtol=2e-5)
            assert not x[0, length:].count_nonzero()
        permutation = torch.arange(256, device=device).reshape(16, 16).T.flatten()
        torch.testing.assert_close(model(batch).log_probs[..., permutation], model(mirrored).log_probs,
                                   atol=8e-6, rtol=2e-5)
        head = initialize_readout("all").to(device)
        concepts, indices = torch.tensor([0, 4], device=device), torch.arange(2, device=device)
        torch.testing.assert_close(head(a, concepts, indices), head(b, concepts, indices), atol=6e-6, rtol=2e-5)
        for view in ("near", "detailed", "coarse"):
            observations = [collate([paired_views(example(changed=changed))[view]]).to(device).observation
                            for changed in (False, True)]
            assert all(torch.equal(getattr(observations[0], f.name), getattr(observations[1], f.name))
                       for f in fields(observations[0]))
            x, y = [model.encoder(obs) for obs in observations]
            for left, right in zip(x.states, y.states):
                # Identical MPS inputs can differ by a few ULPs in scatter/GRU reductions.
                torch.testing.assert_close(left, right, atol=2e-6 if device == "mps" else 0,
                                           rtol=2e-5 if device == "mps" else 0)
    model(batch).loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    for component in ("source_time", "row_interaction"):
        module = getattr(model.encoder, component)
        if module is not None:
            assert sum(p.grad.abs().sum() for p in module.parameters()) > 0


def test_source_event_local_states_ignore_inserted_synthetic_markers():
    obs = observe_complete(fixture_chart(long=True))
    inserted = ObservedRow(325., "boundary", (), (0, 0, 0, 0))
    changed = replace(obs, rows=tuple(sorted((*obs.rows, inserted), key=lambda r: r.time_ms)))
    a, b = collate_observations([obs]), collate_observations([changed])
    for name, model in initialize_representation_comparison(representation_arms()).items():
        left, right = model.encoder(a), model.encoder(b)
        differences = []
        for x, y in zip(left.states[:4], right.states[:4]):
            x, y = x[a.rows[..., 5].bool()], y[b.rows[..., 5].bool()]
            if name != "baseline":
                torch.testing.assert_close(x, y, atol=0, rtol=0)
            differences.append(not torch.allclose(x, y))
        if name == "baseline":
            assert any(differences)


def test_action_time_and_state_supports_are_separate_and_raw_access_bypasses_u():
    obs = observe_complete(fixture_chart(long=True))
    tensor = collate_observations([obs])
    arms = representation_arms()
    models = initialize_representation_comparison({name: arms[name] for name in ("combined", "state")})
    lanes = tensor.lanes.clone()
    lanes[..., 4:] = torch.randn_like(lanes[..., 4:])
    changed = replace(tensor, lanes=lanes, summaries=torch.randn_like(tensor.summaries))
    for name, model in models.items():
        a, b = model.encoder(tensor), model.encoder(changed)
        torch.testing.assert_close(a.states[0], b.states[0], atol=0, rtol=0)
        for x, y in zip(a.states[1:4], b.states[1:4]):
            if name == "combined":
                torch.testing.assert_close(x, y, atol=0, rtol=0)
            else:
                assert not torch.allclose(x, y)
    report = local_support_report(obs, arms["combined"])
    source = [i for i, r in enumerate(obs.rows) if r.phase == "source"]
    center = source[18]
    encoder = models["combined"].encoder
    original = encoder(tensor)
    for level, radius in zip(("U", "L1", "L2", "L3"), (0, 1, 3, 7)):
        support = report[level][center]
        assert support["action_row_indices"] == source[18 - radius:19 + radius]
        assert support["action_source_events"] == 2 * radius + 1
        assert support["action_duration_ms"] == obs.rows[source[18 + radius]].time_ms - obs.rows[source[18 - radius]].time_ms
        assert support["time_row_indices"] == source[17 - radius:20 + radius]
        outside = source[19 + radius]
        changed_lanes = tensor.lanes.clone()
        changed_lanes[:, outside, :, :, :4] = 1 - changed_lanes[:, outside, :, :, :4]
        bank = encoder(replace(tensor, lanes=changed_lanes))
        index = ("U", "L1", "L2", "L3").index(level)
        torch.testing.assert_close(original.states[index][:, center], bank.states[index][:, center], atol=0, rtol=0)
        changed_rows = list(obs.rows)
        outside_time = source[20 + radius]
        changed_rows[outside_time] = replace(changed_rows[outside_time], time_ms=changed_rows[outside_time].time_ms + 7)
        shifted = encoder(collate_observations([replace(obs, rows=tuple(changed_rows))]))
        torch.testing.assert_close(original.states[index][:, center], shifted.states[index][:, center], atol=0, rtol=0)
    conditioned = local_support_report(obs, arms["state"])
    assert conditioned["L1"][center]["state_condition_row_indices"] == list(range(center))
    assert conditioned["L1"][center]["entering_occupancy_condition"]
    assert not conditioned["U"][center]["state_condition_row_indices"]

    config = LocalRepresentationConfig(local_operator="event", source_skip=True)
    geometry = event_geometry(tensor)
    packet = source_packet(tensor, geometry, config)
    packet = replace(packet, row_facts=geometry.gather(packet.row_facts), row_metadata=geometry.gather(packet.row_metadata),
                     state_before=geometry.gather(packet.state_before))
    block = ConditionedLocalBlock(1, 0, config)
    x = torch.zeros(*geometry.valid.shape, 2, 64)
    before = block(x, geometry.valid, packet=packet, geometry=geometry)
    after = block(x, geometry.valid, packet=replace(packet, row_facts=torch.zeros_like(packet.row_facts)), geometry=geometry)
    assert not torch.allclose(before, after)


@pytest.mark.parametrize("conditioning", ["time_action", "time", "action", "constant"])
def test_pair_time_and_relation_conditioning_act_when_content_and_row_time_paths_are_fixed(conditioning):
    tensor = collate([example(long=True)]).observation
    config = replace(representation_arms()["time"], kernel_condition=conditioning)
    geometry = event_geometry(tensor)
    faster = event_geometry(replace(tensor, times_ms=tensor.times_ms / 2))
    packet = source_packet(tensor, geometry, config)
    packet = replace(packet, row_facts=geometry.gather(packet.row_facts), row_metadata=geometry.gather(packet.row_metadata),
                     state_before=geometry.gather(packet.state_before))
    block = ConditionedLocalBlock(1, 0, config)
    x = torch.randn(*geometry.valid.shape, 2, 64)
    a = block(x, geometry.valid, packet=packet, geometry=geometry)
    b = block(x, geometry.valid, packet=packet, geometry=faster)
    c = block(x, geometry.valid, packet=replace(packet, row_facts=packet.row_facts.flip(-1)), geometry=geometry)
    assert torch.allclose(a, b) == (conditioning in ("action", "constant"))
    assert torch.allclose(a, c) == (conditioning in ("time", "constant"))


@pytest.mark.parametrize("device", DEVICES)
def test_empty_source_scope_remains_finite_with_time_and_state_conditions(device):
    model = initialize_representation_comparison({"state": representation_arms()["state"]})["state"].to(device)
    empty = observe_complete(prepare_chart((), Interval(0, 100), Interval(0, 100)))
    bank = model.encoder(collate_observations([empty]).to(device))
    for x in bank.states:
        assert torch.isfinite(x).all()
    for x in bank.states[1:4]:
        torch.testing.assert_close(x, bank.states[0], atol=0, rtol=0)
    logits = initialize_readout("all").to(device)(bank, torch.arange(5, device=device), torch.zeros(5, dtype=torch.long, device=device))
    assert torch.isfinite(logits).all()


def test_paired_updates_evaluation_and_snapshot_resume_include_representation_policy(tmp_path):
    arms = representation_arms()
    chosen = {name: arms[name] for name in ("smooth", "combined")}
    config = ModelConfig(dropout=.1)
    models = initialize_representation_comparison(chosen, config)
    contexts = [TrainingContext("a", "one", fixture_chart()), TrainingContext("b", "two", fixture_chart(long=True))]
    sampler = PairedBlockSampler(contexts)
    optimizers = {name: torch.optim.AdamW(m.parameters(), lr=1e-4) for name, m in models.items()}
    result = train_representation_step(models, optimizers, sampler, blocks=1)
    assert set(result["models"]) == set(chosen)
    paired, records = sampler.draw(2)
    report = evaluate_structure(models, paired, records, batch_size=1, bootstrap_samples=10)
    assert len(report["paired_detailed_mean_row_nll"]) == 1
    for name in chosen:
        assert report["models"][name]["policy"] == models[name].policy_identity
    fixed = collate([example()])
    model, optimizer = models["combined"], optimizers["combined"]
    path = tmp_path / "representation.pt"
    save_snapshot(path, model, optimizer, sampler, update=1)
    def update(m, opt):
        opt.zero_grad(set_to_none=True)
        out = m(fixed)
        out.loss.backward()
        opt.step()
        return out.log_probs.detach()
    expected = update(model, optimizer)
    restored = initialize_representation_comparison({"combined": chosen["combined"]}, config, seed=91)["combined"]
    new_optimizer = torch.optim.AdamW(restored.parameters())
    assert load_snapshot(path, restored, new_optimizer, PairedBlockSampler(contexts)) == 1
    torch.testing.assert_close(update(restored, new_optimizer), expected, atol=0, rtol=0)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())
    wrong = initialize_representation_comparison({"combined": replace(chosen["combined"], kernel_condition="action")}, config)["combined"]
    untouched = deepcopy(wrong.state_dict())
    with pytest.raises(ContractError, match="contract"):
        load_snapshot(path, wrong, torch.optim.AdamW(wrong.parameters()), PairedBlockSampler(contexts))
    assert all(torch.equal(v, wrong.state_dict()[k]) for k, v in untouched.items())


def test_invalid_representation_policies_are_rejected():
    for options in ({"time_basis": "bpm"}, {"local_operator": "attention"}, {"row_interaction": 1},
                    {"kernel_condition": "invalid"},
                    {"local_operator": "time"}):
        with pytest.raises(ContractError):
            LocalRepresentationConfig(**options)
