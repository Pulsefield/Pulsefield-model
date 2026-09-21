from dataclasses import asdict
import gzip
import json
import subprocess
import sys

import pytest
import torch

from pulsefield_model.research.scoped_style_modeling.config import TrainConfig
from pulsefield_model.research.scoped_style_modeling.corpus import PreparedCorpus, sampled_epoch
from pulsefield_model.research.scoped_style_modeling.dataset import (
    ASSESSMENTS, CONCEPTS, REVISION, MANIFEST_SHA256, METHOD, SPECIFICATION_SHA256, ContractError, canonical_json, digest,
)
from pulsefield_model.research.scoped_style_modeling.metrics import assessment_report, record_metrics
from pulsefield_model.research.scoped_style_modeling.model import initialize_model
from pulsefield_model.research.scoped_style_modeling.prepare import _chart_payload
from pulsefield_model.research.scoped_style_modeling.replay import selected_objects
from pulsefield_model.research.scoped_style_modeling.train import run_training, write_json
from pulsefield_model.research.scoped_style_modeling.train_hydra import compose_config
from test_model import fixture


def corpus_fixture(tmp_path):
    root = tmp_path/"prepared"
    (root/"contexts").mkdir(parents=True)
    item = fixture()
    data = canonical_json(_chart_payload(item.chart, item.edges)).encode()
    sha = digest(data)
    with gzip.open(root/"contexts/chart.json.gz", "wb") as stream:
        stream.write(data)
    assignments = {f"source-{s}": {"group_id": f"group-{s}", "split": s} for s in ("train", "validation", "test")}
    split = {"sources": assignments}
    split["sha256"] = digest(canonical_json(split).encode())
    write_json(root/"split-manifest.json", split)
    records = []
    for layer, partition in (("machine", "train"), ("machine", "validation"), ("human", "validation"), ("human", "test")):
        for i, concept in enumerate(CONCEPTS):
            records.append({"cell_id": f"{partition}:{concept}", "record_ids": [f"{layer}:{partition}:{concept}"],
                "layer": layer, "concept": concept, "assessment": ASSESSMENTS[i % 3],
                "origins": ["agent-reviewed" if layer == "machine" else "human-confirmed"], "provenance_json": "[]",
                "source_sha256": f"source-{partition}", **assignments[f"source-{partition}"],
                "chart_key": "chart", "chart_sha256": sha,
                "evidence": [asdict(n) for n in selected_objects(item.chart, item.masks)],
                "evidence_status": "available", "evidence_masks": item.masks})
    payload = "".join(canonical_json(row)+"\n" for row in records).encode()
    (root/"assessment-cohort.jsonl").write_bytes(payload)
    write_json(root/"summary.json", {"data_contract_ready": True, "revision": REVISION,
        "manifest_sha256": MANIFEST_SHA256, "method": METHOD,
        "specification_sha256": SPECIFICATION_SHA256, "split_sha256": split["sha256"],
        "cohort_sha256": digest(payload), "support": [], "rare_slices": []})
    return root


def test_metrics_decompose_and_keep_missed_positives():
    logits = [[10, 0, 1], [0, 2, 0], [0, 0, 2], [2, 0, 0]]
    labels = [2, 1, 2, 0]
    values = record_metrics(logits, labels)
    assert values[0]["predicted"] == 0
    assert values[0]["strength_nll"] is not None
    for label, row in zip(labels, values):
        assert row["nll"] == pytest.approx(row["presence_nll"]+(row["strength_nll"] if label else 0))
    rows = [{**v, "layer": "machine", "concept": CONCEPTS[0], "group_id": "a" if i < 3 else "b",
             "assessment": ASSESSMENTS[labels[i]]} for i, v in enumerate(values)]
    report = assessment_report(rows)
    assert report["macro_nll"] == pytest.approx((sum(v["nll"] for v in values[:3])/3+values[3]["nll"])/2)
    assert report["concepts"][CONCEPTS[0]]["strength"]["balanced_accuracy"] == 1
    assert report["concepts"][CONCEPTS[1]]["strength_nll"] is None
    with pytest.raises(ContractError, match="separate"):
        assessment_report([*rows, {**rows[0], "layer": "human"}])
    rare = assessment_report(rows[:1])["concepts"][CONCEPTS[0]]
    assert rare["strength"]["balanced_accuracy"] is None
    assert rare["strength_nll"] is not None


def test_extreme_logits_have_finite_conditional_nll():
    values = record_metrics([[10000, -10000, -9999]], [2])[0]
    assert values["strength_nll"] == pytest.approx(0.3132616875)
    assert values["nll"] == pytest.approx(values["presence_nll"]+values["strength_nll"])


def test_frozen_corpus_sampler_and_contract_failures(tmp_path):
    root = corpus_fixture(tmp_path)
    corpus = PreparedCorpus(root)
    first = sampled_epoch(corpus.records, 17, 0)
    assert first == sampled_epoch(corpus.records, 17, 0)
    assert all(corpus.records[i]["split"] == "train" and corpus.records[i]["layer"] == "machine" for i in first)
    changed = [{**r, "evidence_masks": None, "evidence_status": "missing"} for r in corpus.records]
    assert first == sampled_epoch(changed, 17, 0)
    assert corpus.batch(first).assessments.shape == (5,)
    (root/"assessment-cohort.jsonl").write_text("corrupt")
    with pytest.raises(ContractError, match="SHA-256"):
        PreparedCorpus(root)


def test_old_preparation_spec_metadata_stays_unavailable(tmp_path):
    root = corpus_fixture(tmp_path)
    summary = json.loads((root/"summary.json").read_text())
    del summary["specification_sha256"]
    write_json(root/"summary.json", summary)
    assert PreparedCorpus(root).summary.get("specification_sha256") is None
    summary["method"] = "wrong-method"
    write_json(root/"summary.json", summary)
    with pytest.raises(ContractError, match="frozen study"):
        PreparedCorpus(root)


def test_invalid_evidence_identities_fail_before_availability_mask(tmp_path):
    root = corpus_fixture(tmp_path)
    corpus = PreparedCorpus(root)
    corpus.records[0]["evidence"][0]["source_line"] = 999999
    corpus.records[0]["evidence_status"] = "missing"
    with pytest.raises(ContractError, match="identity/membership"):
        corpus.example(0)


def test_paired_training_end_to_end_and_selector_free_checkpoint(tmp_path):
    root = corpus_fixture(tmp_path)
    config = TrainConfig(prepared_dir=str(root), output_dir=str(tmp_path/"run"), device="cpu",
                         max_epochs=2, max_updates=1, batch_size=3, memory_log_every=1)
    summary = run_training(config, resolved_yaml="device: cpu\n")
    assert summary["test_evaluated"] is False
    arms = summary["pairs"][0]["arms"]
    assert [a["updates"] for a in arms.values()] == [1, 1]
    assert [a["draws"] for a in arms.values()] == [3, 3]
    assert all(a["stop_reason"] == "max_updates" for a in arms.values())
    assert len(json.loads((tmp_path/"run/seed-17/validation-history.json").read_text())) == 2
    assert (tmp_path/"run/resolved-hydra.yaml").read_text() == "device: cpu\n"
    memory = [json.loads(line) for line in (tmp_path/"run/seed-17/memory.jsonl").read_text().splitlines()]
    assert [row["phase"] for row in memory] == ["initialized", "paired_update", "validation_complete", "diagnostics_complete"]
    assert memory[1]["paired_batches"] == 1
    assert memory[1]["arm_updates"] == {"style-only": 1, "style+evidence": 1}
    assert all(row["process_peak_rss_bytes"] > 0 for row in memory)
    for name in arms:
        directory = tmp_path/"run/seed-17"/name
        inference = torch.load(directory/"assessment.pt", weights_only=True)
        model = initialize_model(config.model, 99, auxiliary=False)
        model.load_state_dict(inference["model"], strict=True)
        model.eval()
        batch = PreparedCorpus(root).batch([5])
        recorded = json.loads((directory/"machine-validation.json").read_text())["predictions"][0]["logits"]
        torch.testing.assert_close(model.assessment(batch.chart, batch.concepts)[0], torch.tensor(recorded), atol=2e-6, rtol=2e-5)
    with pytest.raises(FileExistsError):
        run_training(config)


def test_wall_clock_stops_both_at_same_update(tmp_path):
    root = corpus_fixture(tmp_path)
    result = run_training(TrainConfig(prepared_dir=str(root), output_dir=str(tmp_path/"budget"), device="cpu",
                                     max_epochs=2, batch_size=3, arm_budget_seconds=0.00001))
    pair = result["pairs"][0]
    assert pair["wall_clock_truncated"]
    assert [a["updates"] for a in pair["arms"].values()] == [1, 1]
    assert all(a["budget_overshoot_seconds"] > 0 for a in pair["arms"].values())
    assert not (tmp_path/"budget/seed-17/memory.jsonl").exists()


def test_checkpoint_and_early_stop_use_assessment_not_evidence(tmp_path, monkeypatch):
    from pulsefield_model.research.scoped_style_modeling import train
    root = corpus_fixture(tmp_path)
    original = train.evaluate
    calls = {}

    def competing_metrics(model, corpus, indices, *args, **kwargs):
        rows, report = original(model, corpus, indices, *args, **kwargs)
        if corpus.records[indices[0]]["layer"] == "machine":
            count = calls.get(id(model), 0)
            calls[id(model)] = count+1
            report["macro_nll"] = [1.0, 0.8, 0.9, 0.8][count]
            report["evidence"] = {"macro_normalized_nll": 10-count}
        return rows, report

    monkeypatch.setattr(train, "evaluate", competing_metrics)
    result = run_training(TrainConfig(prepared_dir=str(root), output_dir=str(tmp_path/"selection"),
                                     device="cpu", batch_size=5, max_epochs=5, patience=1))
    for name, arm in result["pairs"][0]["arms"].items():
        assert arm["updates"] == 3 and arm["best_epoch"] == 2
        assert arm["stop_reason"] == "early_stopping"
        checkpoint = torch.load(tmp_path/"selection/seed-17"/name/"best.pt", weights_only=True)
        assert checkpoint["epoch"] == 2 and checkpoint["validation_macro_nll"] == 0.8


def test_hydra_projection_rejects_unknowns_and_invalid_values():
    assert compose_config() == TrainConfig()
    cfg = compose_config(["device=cpu", "seeds=[29,43]", "model.hand_hidden=16", "max_updates=2"])
    assert cfg.device == "cpu" and cfg.seeds == [29, 43] and cfg.model.hand_hidden == 16 and cfg.max_updates == 2
    for override in ("+ignored=1", "+model.ignored=1", "batch_size=0", "model.attention_heads=3", "evidence_weight=-1", "memory_log_every=-1"):
        with pytest.raises(Exception):
            compose_config([override])


def test_packaged_training_config_available():
    from importlib.resources import files
    preset = files("pulsefield_model.configs.hydra").joinpath("scoped_style_train.yaml")
    assert "scoped_style_train_schema" in preset.read_text()


def test_overnight_preset_preserves_paired_protocol():
    config = compose_config(config_name="scoped_style_overnight")
    assert config.seeds == [17, 29, 43]
    assert config.arm_budget_seconds*len(config.seeds)*2 == 8*3600
    assert config.memory_log_every == 20
    assert config.batch_size == 16 and config.max_epochs == 30 and config.patience == 5
    assert config.max_updates is None and config.model == TrainConfig().model
    assert config.evidence_weight == 0.1 and config.learning_rate == 0.0003
    assert compose_config(["memory_log_every=0", "arm_budget_seconds=3600"], config_name="scoped_style_overnight").memory_log_every == 0


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_memory_snapshot_device_counters_do_not_clear_cache(device, monkeypatch):
    from pulsefield_model.research.scoped_style_modeling import train
    synchronized = []
    monkeypatch.setattr(train, "synchronize", lambda name: synchronized.append(name))
    backend = getattr(torch, device)
    monkeypatch.setattr(backend, "empty_cache", lambda: pytest.fail("Telemetry must not change allocator behavior"))
    active = "current_allocated_memory" if device == "mps" else "memory_allocated"
    reserved = "driver_allocated_memory" if device == "mps" else "memory_reserved"
    monkeypatch.setattr(backend, active, lambda: 1024)
    monkeypatch.setattr(backend, reserved, lambda: 8192)
    sample = train.memory_snapshot(device)
    assert synchronized == [device]
    assert sample[f"{device}_active_bytes"] == 1024
    assert sample["mps_driver_bytes" if device == "mps" else "cuda_reserved_bytes"] == 8192
    assert sample["process_peak_rss_bytes"] > 0


def test_training_help_has_no_torch_or_legacy_import():
    script = '''
import sys
class Block:
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith(('pulsefield_model.training', 'pulsefield_model.inference')):
            raise RuntimeError(fullname)
sys.meta_path.insert(0, Block())
from pulsefield_model.research.scoped_style_modeling.train_hydra import cli
sys.argv = ['train_hydra', '--help']
cli()
'''
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "evidence_weight" in result.stdout
