from copy import deepcopy

import pytest
import torch

from ensomi_model.research.scoped_style_modeling.dataset import ContractError
from ensomi_model.research.source_action_modeling.checkpoint import (
    load_snapshot, save_snapshot, warm_start_snapshot)
from ensomi_model.research.source_action_modeling.composition import initialize_composition
from ensomi_model.research.source_action_modeling.experiment_hydra import compose_config
from ensomi_model.research.source_action_modeling.model import initialize_model, initialize_comparison
from ensomi_model.research.source_action_modeling.sampling import BlockSampler, TrainingContext
from ensomi_model.research.source_action_modeling.tensors import collate
from .conftest import example, fixture_chart


def legacy_snapshot(path, model):
    """The schema-3 bilinear tensor layout, with identifiable non-prefix embeddings."""
    sampler = BlockSampler([TrainingContext("g", "c", fixture_chart())])
    save_snapshot(path, model, torch.optim.AdamW(model.parameters()), sampler, update=123)
    payload = torch.load(path, weights_only=False)
    payload["schema"] = 3
    payload["input_contract"] = "source-action-visibility-v2"
    payload["model_policy"]["decoder"] = "joint-row/context-bilinear-hand-transpose-v1"
    alphabet = (0, 1, 2, 4, 5, 6)
    payload["model"]["decoder.actions"] = torch.tensor([[[a,b],[c,d]] for a in alphabet for b in alphabet
                                                        for c in alphabet for d in alphabet])
    payload["model"]["decoder.hand_tokens"] = torch.tensor([[i // 36, i % 36] for i in range(1296)])
    payload["model"]["decoder.action.weight"] = torch.arange(36 * model.config.action_dim).reshape(
        36, model.config.action_dim).float() / 100
    torch.save(payload, path)
    return payload


@pytest.mark.parametrize("family", ["stage1", "reference_h", "composed_all", "serial", "interleaved"])
def test_six_action_weights_map_by_semantics_and_retrain_with_fresh_state(tmp_path, family):
    def make(seed):
        if family == "stage1":
            return initialize_model(seed=seed)
        if family in ("serial", "interleaved"):
            return initialize_composition(compose_config().model, seed=seed)[family]
        return initialize_comparison(seed=seed)[family]
    path = tmp_path / "six.pt"
    saved = legacy_snapshot(path, make(17))
    original_bytes = path.read_bytes()
    model = make(29).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sampler = BlockSampler([TrainingContext("g", "c", fixture_chart())])
    rng, sampling = torch.random.get_rng_state(), sampler.state_dict()
    with pytest.raises(ContractError, match="warm_start_snapshot"):
        load_snapshot(path, model, optimizer, sampler)
    report = warm_start_snapshot(path, model)
    indices = [0,1,2,3,6,7,8,9,12,13,14,15,18,19,20,21]
    assert report["hand_embedding_indices"] == indices and report["source_update"] == 123
    assert report["training_state_restored"] is False
    assert path.read_bytes() == original_bytes and not model.training
    assert not optimizer.state and sampler.state_dict() == sampling
    assert torch.equal(rng, torch.random.get_rng_state())
    for name, value in model.named_parameters():
        expected = saved["model"][name]
        if name == "decoder.action.weight":
            expected = expected[indices]
        assert torch.equal(value, expected), name
    assert model.decoder.actions.shape == (256,2,2)
    assert set(model.decoder.actions.flatten().tolist()) == {0,1,2,3}
    output = model(collate([example()]))
    assert output.log_probs.shape[-1] == 256 and torch.isfinite(output.loss)
    output.loss.backward()
    optimizer.step()
    new = tmp_path / "four.pt"
    save_snapshot(new, model, optimizer, sampler, update=1)
    restored = make(43)
    assert load_snapshot(new, restored, torch.optim.AdamW(restored.parameters()), sampler) == 1
    assert all(torch.equal(v, restored.state_dict()[k]) for k,v in model.state_dict().items())
    another = make(47)
    assert warm_start_snapshot(new, another)["policy"] == "four-action-weights-v1"
    assert all(torch.equal(v, another.state_dict()[k]) for k,v in model.state_dict().items())


@pytest.mark.parametrize("bad", ["static-decoder", "policy", "shape", "buffer", "embedding", "keys", "schema"])
def test_incompatible_migration_fails_before_any_parameter_or_rng_mutation(tmp_path, bad):
    path = tmp_path / "old.pt"
    model = initialize_model()
    payload = legacy_snapshot(path, model)
    if bad == "static-decoder":
        payload["model_policy"]["decoder"] = "static-gram"
    elif bad == "policy":
        payload["model_policy"]["loss"] = "different-risk"
    elif bad == "shape":
        payload["model"]["decoder.interaction.weight"] = torch.zeros(1,1)
    elif bad == "buffer":
        payload["model"]["decoder.actions"][1,0,0] = 99
    elif bad == "embedding":
        payload["model"]["decoder.action.weight"] = torch.zeros(35,model.config.action_dim)
    elif bad == "keys":
        del payload["model"]["decoder.interaction.bias"]
    else:
        payload["schema"] = 2
    torch.save(payload,path)
    initial,rng=deepcopy(model.state_dict()),torch.random.get_rng_state()
    with pytest.raises(ContractError,match="Warm-start"):
        warm_start_snapshot(path,model)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in initial.items())
    assert torch.equal(rng,torch.random.get_rng_state())


def test_runner_consumes_warm_start_before_evaluation_and_training(tmp_path, monkeypatch):
    from ensomi_model.research.source_action_modeling import experiment
    config=compose_config(["device=cpu","seeds=[17]",f"output_dir={tmp_path/'new'}",f"warm_start_dir={tmp_path/'old'}"])
    directory=tmp_path/'old/seed-17'
    directory.mkdir(parents=True)
    sources=initialize_composition(config.model)
    for name,model in sources.items():
        with torch.no_grad():
            next(model.encoder.parameters()).add_(1)
        legacy_snapshot(directory/f'{name}-latest.pt',model)
    contexts=[TrainingContext('g','c',fixture_chart())]
    class ReachedEvaluation(Exception):pass
    monkeypatch.setattr(experiment,'load_population',lambda *a,**k:(None,contexts,None,{'counts':{}}))
    monkeypatch.setattr(experiment,'export_allocation',lambda *a,**k:None)
    monkeypatch.setattr(experiment,'FullCorpusSampler',lambda *a,**k:object())
    def evaluate(models,*a,**k):
        for name,model in models.items():
            assert torch.equal(next(model.encoder.parameters()),next(sources[name].encoder.parameters()))
        raise ReachedEvaluation()
    monkeypatch.setattr(experiment,'evaluate_structure',evaluate)
    with pytest.raises(ReachedEvaluation):
        experiment.run_experiment(config,resolved_yaml='test')
    import json
    report=json.loads((tmp_path/'new/report.json').read_text())
    seed=report['seeds']['17']
    assert seed['updates']==0 and seed['initialization']['policy']=='warm-start'
    assert all(v['source_update']==123 for v in seed['initialization']['weights'].values())
    assert report['status']=='incomplete'
