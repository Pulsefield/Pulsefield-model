from dataclasses import asdict
from importlib.resources import files
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from pulsefield_model.research.oracle_time_continuation import train_run
from pulsefield_model.research.oracle_time_continuation.train_hydra import compose_config
from pulsefield_model.research.scoped_style_modeling.dataset import ContractError, canonical_json, digest
from .conftest import source_bytes


def write_inputs(tmp_path):
    data = source_bytes([(i % 4, i * 100, i * 100) for i in range(70)])
    sha = digest(data)
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / (sha + ".osu")).write_bytes(data)
    manifest = {"sources": {sha: {"group_id": "song:test", "split": "train"}}}
    manifest["sha256"] = digest(canonical_json(manifest).encode())
    path = tmp_path / "split.json"
    path.write_text(json.dumps(manifest))
    return sha, path, manifest


def test_packaged_defaults_typed_overrides_and_unknown_key_rejection():
    assert files("pulsefield_model.configs.hydra").joinpath("oracle_time_train.yaml").is_file()
    config = compose_config()
    assert config.training.chunk_rows == config.model.max_chunk == 128
    assert config.windows.horizons_s == (1., 4., 16.)
    assert config.objective.normalization_rows == 128 and config.objective.lambda_struct == 0
    for overrides in (["+unexpected=1"], ["+training.unexpected=1"], ["+model.unexpected=1"],
                      ["+windows.unexpected=1"], ["+objective.unexpected=1"]):
        with pytest.raises(ContractError, match="Unknown"):
            compose_config(overrides)
    for overrides in (["training.chunk_rows=129"], ["model.max_chunk=4", "training.chunk_rows=5"],
                      ["objective.lambda_struct=-1"], ["windows.horizons_s=[1,4,4]"], ["updates=0"]):
        with pytest.raises((ContractError, ValueError)):
            compose_config(overrides)


def test_all_projected_settings_reach_runner_and_logs_are_reproducible(tmp_path, monkeypatch):
    sha, manifest_path, manifest = write_inputs(tmp_path)
    config = compose_config([
        f"source_dir={tmp_path / 'sources'}", f"split_manifest={manifest_path}",
        f"split_sha256={manifest['sha256']}", f"source_sha256=[{sha}]", f"output_dir={tmp_path / 'run'}",
        "device=cpu", "model_seed=29", "cpu_threads=1", "updates=2", "windows.seed=71",
        "windows.horizons_s=[0.2,0.4,0.8]", "model.hidden=8", "model.heads=2", "model.recent=4",
        "model.coarse_group=2", "model.coarse_capacity=2", "model.coupling_rank=4",
        "training.effective_batch_size=3", "training.microbatch_size=2", "training.chunk_rows=3",
        "training.learning_rate=0.0007", "training.weight_decay=0.03", "training.max_grad_norm=0.4",
        "objective.lambda_struct=0.5", "objective.normalization_rows=64",
    ])
    received, original = [], train_run.SequenceTrainer

    def trainer(model, training, objective):
        result = original(model, training, objective)
        received.append((model.config, training, objective, result.optimizer.param_groups[0]["lr"],
                         result.optimizer.param_groups[0]["weight_decay"], torch.get_num_threads()))
        return result

    monkeypatch.setattr(train_run, "SequenceTrainer", trainer)
    report = train_run.run_training(config, resolved_yaml="test: resolved\n")
    assert received == [(config.model, config.training, config.objective, .0007, .03, 1)]
    assert report["updates"] == 2 and report["target_rows"] > 0
    output = Path(report["output_dir"])
    draws = [json.loads(line) for line in (output / "windows.jsonl").read_text().splitlines()]
    updates = [json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()]
    assert len(draws) == 6 and len(updates) == 2
    assert all(update["denominator"] == 3 * 64 for update in updates)
    assert report["target_rows"] == sum(draw["target_rows"] for draw in draws)
    assert report["prefill_rows"] == sum(draw["prefix_rows"] for draw in draws)
    assert all(draw["sampling_seed"] == 71 and draw["horizon_s"] in (.2, .4, .8) for draw in draws)
    assert all(draw["prefix_coverage"]["recent"]["source_rows"] <= 5 for draw in draws)
    assert all(update["prefill_seconds"] <= update["wall_seconds"] for update in updates)
    assert (output / "resolved.yaml").read_text() == "test: resolved\n"
    assert json.loads((output / "run-config.json").read_text()) == json.loads(json.dumps(asdict(config)))
    weights = torch.load(output / "weights.pt", weights_only=True)
    assert weights["model_config"] == asdict(config.model) and weights["updates"] == 2
    assert "optimizer" not in weights and "sampler" not in weights
    with pytest.raises(ContractError, match="absent or empty"):
        train_run.run_training(config, resolved_yaml="")
    config.output_dir = str(tmp_path / "repeat")
    repeat = train_run.run_training(config, resolved_yaml="test: resolved\n")
    repeated = torch.load(Path(repeat["output_dir"]) / "weights.pt", weights_only=True)
    for key, value in weights["model_state_dict"].items():
        torch.testing.assert_close(value, repeated["model_state_dict"][key], rtol=0, atol=0)


def test_split_digest_and_heldout_assignment_are_checked_before_source_payloads(tmp_path, monkeypatch):
    sha, path, manifest = write_inputs(tmp_path)
    config = compose_config([f"source_dir={tmp_path / 'sources'}", f"split_manifest={path}",
                             f"split_sha256={manifest['sha256']}", f"source_sha256=[{sha}]"])
    manifest["sources"][sha]["split"] = "test"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ContractError, match="pinned SHA"):
        train_run.load_training_sources(config)
    manifest["sha256"] = digest(canonical_json({"sources": manifest["sources"]}).encode())
    config.split_sha256 = manifest["sha256"]
    path.write_text(json.dumps(manifest))

    def no_payload(path):
        raise AssertionError("held-out payload must not be read")

    monkeypatch.setattr(Path, "read_bytes", no_payload)
    with pytest.raises(ContractError, match="train split"):
        train_run.load_training_sources(config)


def test_cli_inspection_and_legacy_flags_and_runtime_import_boundary():
    entry = [sys.executable, "-m", "pulsefield_model.research.oracle_time_continuation.train_hydra"]
    inspection = subprocess.run(entry + ["--cfg", "job"], capture_output=True, text=True, check=True)
    assert "normalization_rows: 128.0" in inspection.stdout
    for arguments in (["--learning-rate", "0.1"], ["--config-name"]):
        result = subprocess.run(entry + arguments, capture_output=True, text=True)
        assert result.returncode != 0 and "error" in result.stderr
    code = """
import sys
from pulsefield_model.research.oracle_time_continuation.training import SequenceTrainer
assert 'hydra' not in sys.modules
assert 'omegaconf' not in sys.modules
assert not any(name.startswith(('pulsefield_model.models', 'pulsefield_model.training', 'pulsefield_model.inference'))
               for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)
