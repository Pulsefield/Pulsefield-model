"""Operator capacity and paired-run contracts; these are not chart-benefit tests."""
from importlib.resources import files
from types import SimpleNamespace

import pytest
import torch

from pulsefield_model.research.scoped_style_modeling.config import ModelConfig as AttentionConfig
from pulsefield_model.research.scoped_style_modeling.model import RelationAttention
from pulsefield_model.research.source_action_modeling.checkpoint import load_snapshot, save_snapshot
from pulsefield_model.research.source_action_modeling.comparison import train_paired_step
from pulsefield_model.research.source_action_modeling.composition import initialize_composition
from pulsefield_model.research.source_action_modeling.experiment_hydra import compose_config
from pulsefield_model.research.source_action_modeling.relation_matching import initialize_relation_matching
from pulsefield_model.research.source_action_modeling.sampling import PairedBlockSampler, TrainingContext
from pulsefield_model.research.source_action_modeling.tensors import collate
from .conftest import example, fixture_chart

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_zero_relation_key_recovers_actual_attention_outputs_and_common_gradients(device):
    config = AttentionConfig(hand_hidden=4, attention_heads=2, dropout=0)
    torch.manual_seed(23)
    baseline = RelationAttention(config, edge_dim=3, relation_dim=5).to(device)
    rng = torch.random.get_rng_state()
    torch.manual_seed(23)
    candidate = RelationAttention(config, edge_dim=3, relation_dim=5, matching="query").to(device)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(value, candidate.state_dict()[key]) for key, value in baseline.state_dict().items())
    assert not candidate.relation_key.count_nonzero()
    query = torch.arange(12, device=device).repeat_interleave(3)
    neighbor = (query + torch.arange(3, device=device).repeat(12)) % 12
    chart = SimpleNamespace(edge_index=torch.stack((query, neighbor)),
        edge_features=torch.randn(36, 3, device=device), relation_edges=torch.arange(36, device=device).repeat(2),
        relation_features=torch.randn(72, 5, device=device))
    source = torch.randn(2, 3, 2, 8, device=device)
    extra = torch.randn(36, 8, device=device)
    direction = torch.randn_like(source)
    outputs, gradients = [], []
    for model in (baseline, candidate):
        x, descriptor = source.detach().clone().requires_grad_(), extra.detach().clone().requires_grad_()
        output = model(x, chart, descriptor)
        (output * direction).sum().backward()
        outputs.append(output)
        gradients.append((x.grad, descriptor.grad))
    tolerance = 1e-5 if device == "mps" else 0
    torch.testing.assert_close(outputs[0], outputs[1], atol=tolerance, rtol=0)
    for a, b in zip(gradients[0], gradients[1]):
        torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
    for name, parameter in baseline.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(candidate.named_parameters())[name].grad,
                                   atol=tolerance, rtol=tolerance)
    assert candidate.relation_key.grad.isfinite().all() and candidate.relation_key.grad.abs().sum() > 0
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=3e-4)
    optimizer.step()
    assert candidate.relation_key.count_nonzero()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("heads", [1, 2])
def test_actual_attention_relation_preference_reverses_with_query_at_fixed_neighbor_keys(device, heads):
    config = AttentionConfig(hand_hidden=2, attention_heads=heads, dropout=0)
    candidate = RelationAttention(config, edge_dim=1, relation_dim=1, matching="query").to(device)
    with torch.no_grad():
        for p in candidate.parameters():
            p.zero_()
        width = 4 // heads
        # Equal nonzero keys isolate relative preference from content matching.
        candidate.qkv.bias[4:8] = 1
        for h in range(heads):
            candidate.qkv.weight[h * width, 0] = 1
            candidate.qkv.weight[8 + h * width, 1] = 1
            candidate.qkv.weight[8 + h * width + 1, 2] = 1
            candidate.relation_key[h * width, 0] = 2
        candidate.relation[0].weight[0, 0] = 1
        candidate.relation[3].weight[0, 0] = 1
        candidate.bias.weight[:, 0] = 0.25
    baseline = RelationAttention(config, edge_dim=1, relation_dim=1).to(device)
    baseline.load_state_dict({k: v for k, v in candidate.state_dict().items() if k != "relation_key"})
    chart = SimpleNamespace(edge_index=torch.tensor([[0, 0, 1, 2], [1, 2, 1, 2]], device=device),
        edge_features=torch.zeros(4, 1, device=device), relation_edges=torch.tensor([0, 1], device=device),
        relation_features=torch.tensor([[1.], [-1.]], device=device))
    source = torch.tensor([[1., 0, 0, 0], [0, 1., 0, 0], [0, 0, 1., 0]], device=device)
    messages = {}
    for name, model in (("additive", baseline), ("query", candidate)):
        captured = []
        # Read the actual aggregated values before residual/norm can confound
        # the query change. The two neighbor values encode their probabilities.
        handle = model.output.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
        for sign in (1, -1):
            x = source.clone()
            x[0, 0] = sign
            qkv = model.qkv(x).reshape(3, 3, heads, 4 // heads)
            assert torch.equal(qkv[1, 1], qkv[2, 1])
            model(x, chart)
        handle.remove()
        messages[name] = [m[0].reshape(heads, 4 // heads)[:, :2] for m in captured]
    for probabilities in messages["additive"]:
        assert (probabilities[:, 0] > probabilities[:, 1]).all()
    # The common content-score shift can round away one FP32 bit before softmax.
    torch.testing.assert_close(*messages["additive"], atol=1e-7, rtol=0)
    assert (messages["query"][0][:, 0] > messages["query"][0][:, 1]).all()
    assert (messages["query"][1][:, 0] < messages["query"][1][:, 1]).all()


@pytest.mark.parametrize("device", DEVICES)
def test_comparison_matches_existing_four_action_baseline_and_preserves_mirror_padding(device):
    rng = torch.random.get_rng_state()
    models = initialize_relation_matching()
    assert torch.equal(rng, torch.random.get_rng_state())
    old = initialize_composition()["serial"].to(device).eval()
    left, right = models.values()
    assert all(torch.equal(v, right.state_dict()[k]) and torch.equal(v.cpu(), old.state_dict()[k].cpu())
               for k, v in left.state_dict().items())
    assert set(right.state_dict()) - set(left.state_dict()) == {"encoder.relations.relation_key"}
    assert sum(p.numel() for p in right.parameters()) - sum(p.numel() for p in left.parameters()) == 4096
    batch = collate([example(), example(long=True)]).to(device)
    tolerance = 3e-6 if device == "mps" else 0
    with torch.no_grad():
        expected = old(batch).log_probs
        for model in models.values():
            model.to(device).eval()
            torch.testing.assert_close(expected, model(batch).log_probs, atol=tolerance, rtol=0)
        right.encoder.relations.relation_key.normal_(std=0.2)
        output = right(batch).log_probs
        permutation = torch.arange(256, device=device).reshape(16, 16).T.flatten()
        mirrored = right(collate([example(mirror=True), example(long=True, mirror=True)]).to(device)).log_probs
        torch.testing.assert_close(output[..., permutation], mirrored, atol=8e-6, rtol=3e-5)
        bank = right.encoder(batch.observation)
        single = right.encoder(collate([example()]).to(device).observation)
        length = single.states[0].shape[1]
        for state, alone in zip(bank.states, single.states):
            assert not state[0, length:].count_nonzero()
            torch.testing.assert_close(state[0, :length], alone[0], atol=8e-6, rtol=3e-5)
    fresh = right.initialize_untrained(17)
    assert fresh.policy_identity == right.policy_identity
    assert not fresh.encoder.relations.relation_key.count_nonzero()


def test_paired_matching_snapshot_continues_same_draws_and_updates(tmp_path):
    contexts = [TrainingContext("g", "k", fixture_chart(long=True))]
    models = initialize_relation_matching()
    optimizers = {n: torch.optim.AdamW(m.parameters(), lr=3e-4) for n, m in models.items()}
    sampler = PairedBlockSampler(contexts, seed=17)
    result = train_paired_step(models, optimizers, sampler, blocks=1, configurations=tuple(models))
    assert result["models"]["additive"]["loss"] == result["models"]["query"]["loss"]
    for n, m in models.items():
        save_snapshot(tmp_path / f"{n}.pt", m, optimizers[n], sampler, update=1)
    expected = train_paired_step(models, optimizers, sampler, blocks=1, configurations=tuple(models))
    continued = initialize_relation_matching()
    opts = {n: torch.optim.AdamW(m.parameters(), lr=3e-4) for n, m in continued.items()}
    restored = PairedBlockSampler(contexts, seed=17)
    for n, m in continued.items():
        assert load_snapshot(tmp_path / f"{n}.pt", m, opts[n], restored) == 1
    actual = train_paired_step(continued, opts, restored, blocks=1, configurations=tuple(continued))
    assert actual["blocks"] == expected["blocks"] and actual["view_sha256"] == expected["view_sha256"]
    for n, m in models.items():
        assert all(torch.equal(v, continued[n].state_dict()[k]) for k, v in m.state_dict().items())
    with pytest.raises(Exception, match="contract mismatch"):
        load_snapshot(tmp_path / "query.pt", continued["additive"], opts["additive"], restored)


def test_packaged_matching_config_is_bounded_and_reaches_initializer():
    name = "source_action_relation_matching"
    assert files("pulsefield_model.configs.hydra").joinpath(name + ".yaml").is_file()
    config = compose_config(config_name=name)
    assert config.comparison == "relation_matching" and config.backbone_schedule == "serial"
    assert config.seeds == [17] and config.min_updates == config.max_updates == 300
    assert config.train_group_limit is None and config.model.dropout == 0 and config.warm_start_dir is None
    models = initialize_relation_matching(config.model, config.seeds[0], backbone_schedule=config.backbone_schedule)
    assert {m.encoder.arm for m in models.values()} == {"serial"}
    assert models["query"].encoder.relations.relation_key.shape == (64, 64)
    for override in ("comparison=unknown", "backbone_schedule=null", "backbone_schedule=unknown",
                     "warm_start_dir=/tmp/weights", "+relation_matching=true", "+model.dtype=float16"):
        with pytest.raises(Exception):
            compose_config([override], config_name=name)
    with pytest.raises(Exception, match="backbone_schedule"):
        compose_config(["backbone_schedule=serial"])


def test_runner_consumes_comparison_schedule_and_records_common_initialization(tmp_path, monkeypatch):
    import json
    from pulsefield_model.research.source_action_modeling import experiment
    config = compose_config(["device=cpu", "backbone_schedule=interleaved", f"output_dir={tmp_path / 'run'}"],
                            config_name="source_action_relation_matching")
    monkeypatch.setattr(experiment, "load_population", lambda *a, **kw: (None, None, None, {"counts": {}}))
    monkeypatch.setattr(experiment, "export_allocation", lambda *a, **kw: None)
    monkeypatch.setattr(experiment, "fixed_validation", lambda *a, **kw: ([], []))
    monkeypatch.setattr(experiment, "FullCorpusSampler", lambda *a, **kw: None)
    def inspect_models(models, *args, **kwargs):
        assert set(models) == {"additive", "query"}
        assert {m.encoder.arm for m in models.values()} == {"interleaved"}
        assert models["additive"].encoder.relations.relation_key is None
        assert not models["query"].encoder.relations.relation_key.count_nonzero()
        raise RuntimeError("Runner selection inspected")
    monkeypatch.setattr(experiment, "evaluate_structure", inspect_models)
    with pytest.raises(RuntimeError, match="Runner selection inspected"):
        experiment.run_experiment(config, resolved_yaml="fixture")
    report = json.loads((tmp_path / "run/report.json").read_text())
    initialization = report["seeds"]["17"]["initialization"]
    assert initialization["common_tensors_equal"] and initialization["relation_key_zero"]
    assert initialization["parameter_dtypes"] == {"additive": ["torch.float32"], "query": ["torch.float32"]}
    manifest = json.loads((tmp_path / "run/source/manifest.json").read_text())
    assert "scoped_style_modeling/model.py" in manifest
