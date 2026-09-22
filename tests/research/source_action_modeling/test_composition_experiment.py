from importlib.resources import files
import json

import pytest
import torch

from ensomi_model.research.source_action_modeling.checkpoint import load_snapshot, save_snapshot
from ensomi_model.research.source_action_modeling.composition import initialize_composition, ORDERS
from ensomi_model.research.source_action_modeling.comparison import train_paired_step
from ensomi_model.research.source_action_modeling.experiment_hydra import compose_config
from ensomi_model.research.source_action_modeling.representation_experiments import initialize_representation_comparison, representation_arms
from ensomi_model.research.source_action_modeling.sampling import TrainingContext, PairedBlockSampler
from ensomi_model.research.source_action_modeling.structural_probe import structural_cases, evaluate_structural_reuse
from ensomi_model.research.source_action_modeling.tensors import collate
from .conftest import example, fixture_chart

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_order_is_only_parameter_matched_intervention_and_preserves_mirror_padding(device):
    rng = torch.random.get_rng_state()
    models = initialize_composition()
    assert torch.equal(rng, torch.random.get_rng_state())
    left, right = models.values()
    assert all(torch.equal(v, right.state_dict()[k]) for k, v in left.state_dict().items())
    baseline = initialize_representation_comparison({"combined": representation_arms()["combined"]})["combined"].to(device).eval()
    batch = collate([example(), example(long=True)]).to(device)
    permutation = torch.arange(256, device=device).reshape(16, 16).T.flatten()
    values = {}
    for name, model in models.items():
        model.to(device).eval()
        trace = []
        handles = [module.register_forward_hook(lambda m, args, out, label=label: trace.append(label))
                   for label, module in [("L1", model.encoder.local[0]), ("L2", model.encoder.local[1]),
                                         ("L3", model.encoder.local[2]), ("R", model.encoder.relations)]]
        values[name] = model(batch).log_probs
        assert trace == list(ORDERS[name])
        for h in handles:
            h.remove()
        mirrored = model(collate([example(mirror=True), example(long=True, mirror=True)]).to(device)).log_probs
        torch.testing.assert_close(values[name][..., permutation], mirrored, atol=8e-6, rtol=3e-5)
        bank = model.encoder(batch.observation)
        length = int(collate([example()]).observation.lengths[0])
        single = model.encoder(collate([example()]).to(device).observation)
        for state, alone in zip(bank.states, single.states):
            assert not state[0, length:].count_nonzero()
            torch.testing.assert_close(state[0, :length], alone[0], atol=8e-6, rtol=3e-5)
        model(batch).loss.backward()
        for module in (*model.encoder.local, model.encoder.relations):
            assert sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None) > 0
    # Repeated forwards of the same MPS model differ by up to 9.54e-7 in
    # sparse reductions. CPU execution and every initialization tensor are exact.
    tolerance = 2e-6 if device == "mps" else 0
    torch.testing.assert_close(values["serial"], baseline(batch).log_probs, atol=tolerance, rtol=0)
    assert not torch.allclose(values["serial"].nan_to_num(), values["interleaved"].nan_to_num())


def test_paired_order_snapshots_continue_exactly(tmp_path):
    contexts = [TrainingContext("g", "k", fixture_chart(long=True))]
    models = initialize_composition()
    optimizers = {n: torch.optim.AdamW(m.parameters(), lr=0.001) for n, m in models.items()}
    sampler = PairedBlockSampler(contexts, seed=17)
    train_paired_step(models, optimizers, sampler, blocks=1, configurations=tuple(ORDERS))
    for n, m in models.items():
        save_snapshot(tmp_path / f"{n}.pt", m, optimizers[n], sampler, update=1)
    expected = train_paired_step(models, optimizers, sampler, blocks=1, configurations=tuple(ORDERS))
    continued = initialize_composition()
    new_optimizers = {n: torch.optim.AdamW(m.parameters(), lr=0.001) for n, m in continued.items()}
    next_sampler = PairedBlockSampler(contexts, seed=17)
    for n, m in continued.items():
        assert load_snapshot(tmp_path / f"{n}.pt", m, new_optimizers[n], next_sampler) == 1
    actual = train_paired_step(continued, new_optimizers, next_sampler, blocks=1, configurations=tuple(ORDERS))
    assert actual["blocks"] == expected["blocks"]
    for n, m in models.items():
        assert all(torch.equal(v, continued[n].state_dict()[k]) for k, v in m.state_dict().items())


def test_packaged_config_projects_and_rejects_ineffective_or_unknown_fields():
    assert files("ensomi_model.configs.hydra").joinpath("source_action_composition.yaml").is_file()
    config = compose_config()
    assert config.seeds == [17, 29] and config.model.decoder_hidden == 64
    assert config.max_seconds == 39600 and config.model.hand_hidden == 32
    assert config.train_group_limit is None and config.source_cache_charts == 16
    assert config.warm_start_dir is None
    assert compose_config(["warm_start_dir=/tmp/source-action-old"]).warm_start_dir == "/tmp/source-action-old"
    with pytest.raises(Exception, match="warm_start_dir"):
        compose_config(["warm_start_dir="])
    assert compose_config(["train_group_limit=32"]).train_group_limit == 32
    for override in ("+unused=true", "+model.unused=1", "model.hand_hidden=64", "model.lane_dim=32", "model.dropout=0.1",
                     "seeds=[17,17]", "min_updates=999999", "max_source_rows=64", "train_group_limit=0",
                     "source_cache_charts=0", "+local_train_groups=2048", "+local_validation_groups=128"):
        with pytest.raises(Exception):
            compose_config([override])


def test_factorial_holdout_preserves_histograms_and_does_not_invent_semantics():
    cases = structural_cases()
    for c in cases:
        assert len(c["lanes"]) == 64
        assert [c["lanes"].count(i) for i in range(4)] == [16] * 4
    fit = [c for c in cases if c["split"] == "fit"]
    holdout = [c for c in cases if c["split"] == "combination_holdout"]
    assert {tuple(c["bits"]) for c in fit}.isdisjoint({tuple(c["bits"]) for c in holdout})
    for axis in range(3):
        assert {c["bits"][axis] for c in fit} == {0, 1}
    model = initialize_composition()["interleaved"]
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    report = evaluate_structural_reuse(model, batch_size=32)
    assert not report["factorization_claim"] and not report["human_style_labels"]
    assert set(report["stages"]) == {"U", "S1", "S2", "S3", "S4", "H", "orderless_source"}
    assert report["stages"]["orderless_source"]["counterfactuals"]["run_expansion"]["facts"]["adjacent_lane_recurrence"]["direction_accuracy"] == 0
    assert all(torch.equal(p, before[n]) and p.grad is None for n, p in model.named_parameters())


@pytest.mark.parametrize("failure", [FileNotFoundError, KeyboardInterrupt])
def test_runner_records_failure_and_rejects_output_collision(tmp_path, monkeypatch, failure):
    from ensomi_model.research.source_action_modeling import experiment
    config = compose_config(["device=cpu", f"output_dir={tmp_path / 'run'}"])
    def missing_population(*args, **kwargs):
        raise failure("Population interrupted")
    monkeypatch.setattr(experiment, "load_population", missing_population)
    previous_threads = torch.get_num_threads()
    with pytest.raises(failure, match="Population interrupted"):
        experiment.run_experiment(config, resolved_yaml="fixture")
    assert torch.get_num_threads() == previous_threads
    report = json.loads((tmp_path / "run/report.json").read_text())
    assert report["status"] == "incomplete" and not report["seeds"]
    assert (tmp_path / "run/runner.json").is_file()
    with pytest.raises(FileExistsError):
        experiment.run_experiment(config, resolved_yaml="fixture")
