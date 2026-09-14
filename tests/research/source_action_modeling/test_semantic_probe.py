from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest
import torch

from pulsefield_model.research.scoped_style_modeling.dataset import ASSESSMENTS, CONCEPTS, REVISION, ContractError
from pulsefield_model.research.scoped_style_modeling.probe_data import input_identity
from pulsefield_model.research.source_action_modeling.comparison import pretraining_contexts
from pulsefield_model.research.source_action_modeling.diagnostics import capture_semantic_response, finish_semantic_response
from pulsefield_model.research.source_action_modeling.model import initialize_comparison
from pulsefield_model.research.source_action_modeling.sampling import SPLIT_SHA256
from pulsefield_model.research.source_action_modeling.semantic_probe import (SemanticCorpus, evaluate_readout, fit_matched_probes,
                                                                           fit_readout, human_targets, initialize_readout)
from pulsefield_model.research.source_action_modeling.tensors import collate
from .conftest import example, fixture_chart

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) + (["cuda"] if torch.cuda.is_available() else [])


def corpus_fixture():
    chart = fixture_chart()
    records = []
    for split in ("train", "validation"):
        for i, concept in enumerate(CONCEPTS):
            identity = f"{split}-{concept}"
            records.append({"cell_id": identity, "record_ids": [identity], "concept": concept, "assessment": ASSESSMENTS[i % 3],
                            "scope": asdict(chart.inputs.scope), "context": asdict(chart.inputs.context), "playback_rate": 1,
                            "source_sha256": split, "chart_key": split, "chart_sha256": split, "group_id": split, "split": split,
                            "layer": "human", "provenance_json": json.dumps([{"record_id": identity, "human_confidence": "high"}])})
    corpus = SimpleNamespace(records=records, summary={"revision": REVISION, "split_sha256": SPLIT_SHA256},
                             chart=lambda key, sha: (chart, ()))
    return corpus


def test_explicit_human_selection_blocks_fallback_conflicts_and_low_confidence_tech():
    corpus = corpus_fixture()
    records = corpus.records
    machine = {**records[0], "cell_id": "machine", "source_sha256": "machine", "layer": "machine"}
    low = {**records[3], "cell_id": "low-tech", "source_sha256": "low-tech", "record_ids": ["low"],
           "provenance_json": json.dumps([{"record_id": "low", "human_confidence": "low"}])}
    records.extend((machine, low))
    issues = [{"layer": "human", "kind": "conflicting-cell", "cell_id": records[0]["cell_id"]}]
    assert human_targets(records, issues, split="train") == [1, 2, 3, 4]
    with pytest.raises(ContractError, match="only human"):
        human_targets(records, issues, split="test")
    adapter = SemanticCorpus(corpus, issues)
    with pytest.raises(ContractError, match="eligible human"):
        adapter.batch([10])
    records[1]["playback_rate"] = 1.2
    with pytest.raises(ContractError, match="only 1x"):
        SemanticCorpus(corpus, issues)


@pytest.mark.parametrize("device", DEVICES)
def test_complete_original_inputs_are_deduplicated_and_frozen_fit_preserves_encoder(device):
    corpus = SemanticCorpus(corpus_fixture(), [])
    batch = corpus.batch(corpus.train_indices)
    assert len(batch.observation.lengths) == 1
    assert batch.input_indices.tolist() == [0] * 5
    assert batch.observation.lanes[..., 3].all()
    models = initialize_comparison()
    a, b = initialize_readout("H"), initialize_readout("all")
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    model = models["composed_all"].to(device).train()
    initial = deepcopy(model.state_dict())
    # Existing gradients belong to the predictor's caller and must also survive.
    model(collate([example()]).to(device)).loss.backward()
    gradients = {n: p.grad.clone() for n, p in model.encoder.named_parameters()}
    head, fit = fit_readout(model, corpus, steps=3, batch_size=5, learning_rate=1e-3)
    assert head.fitted and len(fit["losses"]) == 3 and model.encoder.training
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in initial.items())
    assert all(torch.equal(p.grad, gradients[n]) for n, p in model.encoder.named_parameters())
    assert all(corpus.records[i]["layer"] == "human" and corpus.records[i]["split"] == "train" for i in fit["sampled_indices"])
    result = evaluate_readout(model, head, corpus, batch_size=2)
    assert len(result["predictions"]) == 5 and result["metrics"]["macro_nll"] is not None
    assert "ranking" in result["metrics"] and "joint_stream_jack" in result["metrics"]
    assert all(r["layer"] == "human" for r in result["predictions"])
    matched = fit_matched_probes(model, corpus, encoder_seed=17, readout_seed=29, steps=1, batch_size=5, learning_rate=1e-3)
    assert matched["trained"]["fit"]["sampled_indices"] == matched["untrained"]["fit"]["sampled_indices"]
    assert matched["trained"]["readout"].access == matched["untrained"]["readout"].access == "all"
    assert len(matched["paired"]["cells"]) == 5


def test_fitted_semantic_margin_linearizes_encoder_updates_and_rejects_head_drift():
    corpus = SemanticCorpus(corpus_fixture(), [])
    model = initialize_comparison()["composed_all"]
    fixed = corpus.batch(corpus.validation_indices)
    head = initialize_readout("all")
    with pytest.raises(ContractError, match="fitted"):
        capture_semantic_response(model, head, fixed)
    head, _ = fit_readout(model, corpus, steps=1, batch_size=5, learning_rate=1e-3)
    before = capture_semantic_response(model, head, fixed)
    assert model.encoder.training
    assert finish_semantic_response(before, model, head, fixed)["actual_change"] == 0
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    model(collate([example()])).loss.backward()
    optimizer.step()
    result = finish_semantic_response(before, model, head, fixed)
    assert set(result["gradient_dot_displacement"]) == {"encoder"}
    assert abs(result["linearization_residual"]) < 2e-7 + .05 * abs(result["actual_change"])
    with torch.no_grad():
        next(head.parameters()).add_(.1)
    with pytest.raises(ContractError, match="readout changed"):
        finish_semantic_response(before, model, head, fixed)


def test_pretraining_deduplicates_concepts_and_rejects_heldout_and_declared_related_variants():
    corpus = corpus_fixture()
    keys = {input_identity(r) for r in corpus.records if r["split"] == "train"}
    contexts, manifest = pretraining_contexts(corpus, input_keys=keys)
    assert len(contexts) == len(manifest["inputs"]) == 1
    assert manifest["dataset_revision"] == REVISION
    assert manifest["excluded_sources"] == ["validation"]
    assert contexts[0].chart.inputs.scope == fixture_chart().inputs.scope
    with pytest.raises(ContractError, match="excluded"):
        pretraining_contexts(corpus, input_keys=keys, excluded_groups=("train",))
    with pytest.raises(ContractError, match="excluded"):
        pretraining_contexts(corpus, input_keys={input_identity(r) for r in corpus.records})
    corpus.summary["split_sha256"] = "wrong"
    with pytest.raises(ContractError, match="pinned"):
        pretraining_contexts(corpus, input_keys=keys)
