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
    assert config.windows.gap_sampling_probability == 0
    assert config.windows.gap_thresholds_ms == (2000., 8000., 32000.)
    assert config.objective.normalization_rows == 128 and config.objective.lambda_struct == 0
    assert config.model.clock_readout_hidden == 0 and config.training.clock_readout_learning_rate is None
    for overrides in (["+unexpected=1"], ["+training.unexpected=1"], ["+model.unexpected=1"],
                      ["+windows.unexpected=1"], ["+objective.unexpected=1"]):
        with pytest.raises(ContractError, match="Unknown"):
            compose_config(overrides)
    for overrides in (["training.chunk_rows=129"], ["model.max_chunk=4", "training.chunk_rows=5"],
                      ["objective.lambda_struct=-1"], ["windows.horizons_s=[1,4,4]"], ["updates=0"],
                      ['training.timing_learning_rate=0.01'],
                      ['training.clock_readout_learning_rate=0.001'], ['model.clock_readout_hidden=-1'],
                      ['model.clock_readout_hidden=129'],
                      ['model.clock_readout_hidden=16', 'training.clock_readout_learning_rate=0'],
                      ['windows.gap_sampling_probability=1.1'], ['windows.gap_context_rows=129'],
                      ['windows.gap_thresholds_ms=[2000,1000,32000]'],
                      ['model.time_lookahead_rows=16', 'training.timing_learning_rate=0']):
        with pytest.raises((ContractError, ValueError)):
            compose_config(overrides)


def test_mac_profile_concentrates_capacity_in_temporal_and_bounds_the_live_chunk():
    config = compose_config(config_name='oracle_time_train_mac')
    assert config.model.hidden == 128 and config.model.temporal_hidden == 512
    assert config.model.temporal_layers == 6 and config.model.temporal_bias_hidden == 64
    assert config.training.microbatch_size == 1
    assert config.training.chunk_rows == config.model.max_chunk == 64
    assert config.windows.windows_per_chart == 4 and config.training.effective_batch_size == 8
    assert config.training.reuse_prefixes and config.training.warmup_updates == 20
    assert config.training.learning_rate == .00003 and config.training.weight_decay == .01
    assert config.objective.lambda_struct == .3
    assert config.checkpoint_every_updates == 25 and config.resources.checkpoint_max_bytes == 512 * 1024**2
    for overrides in (['training.microbatch_size=2'], ['model.max_chunk=128'], ['model.temporal_expansion=9'],
                      ['model.temporal_bias_hidden=128']):
        with pytest.raises(ContractError, match='envelope'):
            compose_config(overrides, config_name='oracle_time_train_mac')


def test_large_mac_profile_budgets_the_measured_temporal_and_checkpoint_capacity():
    config = compose_config(config_name='oracle_time_train_mac_large')
    assert config.model.temporal_hidden == 1024 and config.model.hidden == 128
    assert config.training.microbatch_size == 1 and config.model.max_chunk == 64
    assert config.resources.checkpoint_max_bytes == 1024**3
    assert config.resources.rss_limit_bytes == 8 * 1024**3
    assert config.checkpoint_every_updates == 100


def test_all_projected_settings_reach_runner_and_logs_are_reproducible(tmp_path, monkeypatch):
    sha, manifest_path, manifest = write_inputs(tmp_path)
    config = compose_config([
        f"source_dir={tmp_path / 'sources'}", f"split_manifest={manifest_path}",
        f"split_sha256={manifest['sha256']}", f"source_sha256=[{sha}]", f"output_dir={tmp_path / 'run'}",
        "device=cpu", "model_seed=29", "cpu_threads=1", "updates=2", "windows.seed=71",
        "windows.horizons_s=[0.2,0.4,0.8]", "model.hidden=8", "model.heads=2", "model.recent=4",
        "windows.windows_per_chart=3", "training.reuse_prefixes=true",
        "windows.gap_sampling_probability=0.5", "windows.gap_thresholds_ms=[100,200,400]",
        "windows.gap_context_rows=7",
        "model.coarse_group=2", "model.coarse_capacity=2", "model.coupling_rank=4",
        "model.time_lookahead_rows=16", "training.timing_learning_rate=0.001",
        "model.clock_readout_hidden=12", "training.clock_readout_learning_rate=0.002",
        "training.effective_batch_size=3", "training.microbatch_size=2", "training.chunk_rows=3",
        "training.learning_rate=0.0007", "training.weight_decay=0.03", "training.max_grad_norm=0.4",
        "objective.lambda_struct=0.5", "objective.normalization_rows=64",
    ])
    received, clock_gradients, original = [], [], train_run.SequenceTrainer

    def trainer(model, training, objective):
        result = original(model, training, objective)
        received.append((model.config, training, objective, result.optimizer.param_groups[0]["lr"],
                         result.optimizer.param_groups[0]["weight_decay"], torch.get_num_threads()))
        update = result.update
        def measured_update(windows):
            report = update(windows)
            clock_gradients.append([float(layer.weight.grad.norm())
                                    for layer in (model.head.clock_readout.projection[0],
                                                  model.head.clock_readout.projection[-1])])
            return report
        result.update = measured_update
        return result

    monkeypatch.setattr(train_run, "SequenceTrainer", trainer)
    report = train_run.run_training(config, resolved_yaml="test: resolved\n")
    assert received == [(config.model, config.training, config.objective, .0007, .03, 1)]
    assert clock_gradients[0][0] == 0 and clock_gradients[0][1] > 0
    assert all(value > 0 for value in clock_gradients[1])
    assert report["updates"] == 2 and report["target_rows"] > 0
    output = Path(report["output_dir"])
    draws = [json.loads(line) for line in (output / "windows.jsonl").read_text().splitlines()]
    updates = [json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()]
    assert len(draws) == 6 and len(updates) == 2
    assert all(update["denominator"] == 3 * 64 for update in updates)
    assert report["target_rows"] == sum(draw["target_rows"] for draw in draws)
    assert report["prefill_rows"] == sum(draw["prefix_rows"] for draw in draws)
    assert all(draw["sampling_seed"] == 71 and draw["horizon_s"] in (.2, .4, .8) for draw in draws)
    assert {draw['sampling_route'] for draw in draws} == {'base', 'gap'}
    assert all(draw['gap_index'] - draw['start'] < 7 for draw in draws if draw['sampling_route'] == 'gap')
    assert json.loads((output / 'population.json').read_text())['gap_strata'] == [
        dict(band=0, lower_ms=100., upper_ms=200., groups=1, charts=1, gaps=39)]
    assert all(draw["prefix_coverage"]["recent"]["source_rows"] <= 5 for draw in draws)
    assert all(update["prefill_seconds"] <= update["wall_seconds"] for update in updates)
    assert all(update["computed_prefill_rows"] < update["prefill_rows"] for update in updates)
    assert all(update['learning_rates'] == {'backbone': .0007, 'timing': .001, 'clock_readout': .002}
               for update in updates)
    assert (output / "resolved.yaml").read_text() == "test: resolved\n"
    assert json.loads((output / "run-config.json").read_text()) == json.loads(json.dumps(asdict(config)))
    weights = torch.load(output / "weights.pt", weights_only=True)
    assert weights["model_config"] == asdict(config.model) and weights["updates"] == 2
    assert "optimizer" not in weights and "sampler" not in weights
    assert weights['model_state_dict']['head.clock_readout.projection.2.weight'].abs().sum() > 0
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
