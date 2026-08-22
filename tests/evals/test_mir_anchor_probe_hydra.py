from __future__ import annotations

import json
from dataclasses import asdict
from importlib import resources
from pathlib import Path

import pytest
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

from pulsefield_model.evals.mir_anchor_probe_hydra import MIRAnchorExperimentConfig
from pulsefield_model.evals.mir_anchor_probe_hydra import compose_mir_anchor_probe_config
from pulsefield_model.evals.mir_anchor_probe_hydra import mir_anchor_probe_config_from_hydra
from pulsefield_model.evals.mir_anchor_probe_hydra import probe_run_config_from_sections
from pulsefield_model.evals.mir_anchor_probe_hydra import run_mir_anchor_stage


def test_default_config_composes_from_packaged_resource() -> None:
    config = compose_mir_anchor_probe_config()

    assert isinstance(config, MIRAnchorExperimentConfig)
    assert config.stage == "prepare"
    assert config.risk.max_gap_ms == 2_000
    assert config.data.train_audio_count == 750
    assert config.data.validation_audio_count == 150
    assert config.data.test_audio_count == 150
    assert config.teacher.mel_hop_ms == 5
    assert config.teacher.tempogram_hop_ms == 20
    assert config.probe.seeds == [0, 1, 2]
    assert config.probe.train_choice_sets_per_batch == 32
    assert config.probe.encoder_chunk_frames == 8_192
    assert config.probe.encoder_max_fast_frames == 147_456
    assert resources.files("pulsefield_model.configs.hydra").joinpath("mir_anchor_probe.yaml").is_file()


def test_overrides_reach_the_typed_runtime_config() -> None:
    config = compose_mir_anchor_probe_config(
        (
            "stage=probe",
            "risk.controls_per_case=8",
            "risk.max_gap_ms=1500",
            "data.train_audio_count=70",
            "data.validation_audio_count=15",
            "data.test_audio_count=15",
            "probe.history_rows=16",
            "probe.train_choice_sets_per_batch=17",
            "probe.encoder_chunk_frames=4096",
            "probe.encoder_max_fast_frames=131072",
            "probe.seeds=[7,8]",
            "teacher.tempo_bins=48",
        ),
    )

    assert config.stage == "probe"
    assert config.risk.controls_per_case == 8
    assert config.risk.max_gap_ms == 1_500
    assert config.data.train_audio_count == 70
    assert config.data.validation_audio_count == 15
    assert config.data.test_audio_count == 15
    assert config.probe.history_rows == 16
    assert config.probe.train_choice_sets_per_batch == 17
    assert config.probe.encoder_chunk_frames == 4_096
    assert config.probe.encoder_max_fast_frames == 131_072
    assert config.probe.seeds == [7, 8]
    assert config.teacher.tempo_bins == 48


def test_unknown_keys_and_invalid_semantics_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown probe keys"):
        compose_mir_anchor_probe_config(("+probe.unused=true",))

    source = OmegaConf.structured(MIRAnchorExperimentConfig)
    source.stage = "train"
    with pytest.raises(ValueError, match="prepare, extract, probe"):
        mir_anchor_probe_config_from_hydra(source)


def test_teacher_validation_runs_during_composition() -> None:
    with pytest.raises((ConfigCompositionException, ValueError), match="positive multiple"):
        compose_mir_anchor_probe_config(("teacher.tempogram_hop_ms=12",))
    with pytest.raises(ValueError, match="exactly one scale"):
        compose_mir_anchor_probe_config(("teacher.tempogram_window_seconds=[4.0,8.0]",))
    with pytest.raises(ValueError, match="positive and finite"):
        compose_mir_anchor_probe_config(("probe.learning_rate=inf",))
    with pytest.raises(ValueError, match="data.test_audio_count must be a positive integer"):
        compose_mir_anchor_probe_config(("data.test_audio_count=0",))


def test_prepare_stage_projects_exact_split_counts(tmp_path: Path, monkeypatch) -> None:
    from pulsefield_model.evals import mir_anchor_probe

    calls = []

    def fake_prepare(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(mir_anchor_probe, "prepare_mir_anchor_manifest", fake_prepare)
    config = compose_mir_anchor_probe_config(
        (
            "stage=prepare",
            "data.train_audio_count=7",
            "data.validation_audio_count=2",
            "data.test_audio_count=1",
            f"output.dir={tmp_path.as_posix()}",
        ),
    )

    run_mir_anchor_stage(config)

    assert len(calls) == 1
    assert calls[0]["audio_count"] == 10
    assert calls[0]["split_counts"] == {"train": 7, "validation": 2, "test": 1}


def test_hydra_and_direct_runtime_defaults_match() -> None:
    from pulsefield_model.evals.mir_anchor_probe import MIRProbeRunConfig

    hydra_config = compose_mir_anchor_probe_config()
    runtime = MIRProbeRunConfig()

    projected = probe_run_config_from_sections(hydra_config, seed=hydra_config.probe.seeds[0])

    assert asdict(projected) == asdict(runtime)


def test_probe_stage_projects_every_run_field_and_separates_seeds(tmp_path: Path, monkeypatch) -> None:
    from pulsefield_model.evals import mir_anchor_probe

    calls = []
    aggregate_calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        run_config = kwargs["run_config"]
        output_dir = Path(kwargs["output_dir"])
        return mir_anchor_probe.MIRProbeRunReport(
            seed=run_config.seed,
            parameter_count=123,
            best_epoch=2,
            best_validation_nll=0.5,
            effective_audio_counts={"train": 1, "validation": 1, "test": 1},
            effective_case_counts={"train": 2, "validation": 2, "test": 2},
            state_path=output_dir / "probe_state.pt",
            per_audio_path=output_dir / "per_audio.parquet",
            summary_path=output_dir / "summary.json",
        )

    monkeypatch.setattr(mir_anchor_probe, "run_mir_anchor_probe", fake_run)

    def fake_aggregate(reports, **kwargs):
        aggregate_calls.append((reports, kwargs))
        return {
            "seeds": [report.seed for report in reports],
            "seed_selection": False,
        }

    monkeypatch.setattr(mir_anchor_probe, "aggregate_mir_probe_runs", fake_aggregate)
    config = compose_mir_anchor_probe_config(
        (
            "stage=probe",
            "probe.seeds=[7,8]",
            "probe.history_rows=16",
            "probe.train_choice_sets_per_batch=17",
            "probe.eval_choice_sets_per_batch=99",
            "probe.encoder_chunk_frames=4096",
            "probe.encoder_max_fast_frames=131072",
            "risk.controls_per_case=8",
            f"output.dir={tmp_path.as_posix()}",
        ),
    )

    reports = run_mir_anchor_stage(config)

    assert tuple(report.seed for report in reports) == (7, 8)
    assert [call["run_config"].seed for call in calls] == [7, 8]
    assert all(call["run_config"].choice_seed == 1_337 for call in calls)
    assert all(call["run_config"].history_rows == 16 for call in calls)
    assert all(call["run_config"].train_choice_sets_per_batch == 17 for call in calls)
    assert all(call["run_config"].eval_choice_sets_per_batch == 99 for call in calls)
    assert all(call["run_config"].encoder_chunk_frames == 4_096 for call in calls)
    assert all(call["run_config"].encoder_max_fast_frames == 131_072 for call in calls)
    assert all(call["run_config"].controls_per_case == 8 for call in calls)
    assert all(call["feature_config"].mel_hop_ms == 5 for call in calls)
    assert len(aggregate_calls) == 1
    aggregate_reports, aggregate_kwargs = aggregate_calls[0]
    assert tuple(report.seed for report in aggregate_reports) == (7, 8)
    assert aggregate_kwargs == {
        "output_path": tmp_path / "multi_seed_inference.json",
        "split": "test",
        "bootstrap_seed": 1_337,
    }
    report = json.loads((tmp_path / "multi_seed_report.json").read_text(encoding="utf-8"))
    assert report["inference"] == {"seeds": [7, 8], "seed_selection": False}
