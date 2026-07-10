from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("torch")

from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError, ConfigKeyError

from pulsefield_model.training import mapper_v2, mapper_v2_1
from pulsefield_model.training.hydra_config import (
    TrainingExperimentConfig,
    compose_training_experiment_config,
    default_hydra_config_dir,
    training_experiment_config_to_legacy_dict,
    validate_training_experiment_config,
)
from pulsefield_model.training.mapper_training_hydra import _call_kwargs


@pytest.mark.parametrize(
    ("mapper_preset", "legacy_path", "legacy_loader"),
    (
        (
            "v2_tuple_d384_l4_phase_b",
            "configs/training/stage2_mapper_v2_phase_b_global_mps.yaml",
            mapper_v2.load_run_config,
        ),
        (
            "v2_tuple_d768_l8_phase_b",
            "configs/training/stage2_mapper_v2_phase_b_global_d768_l8_mps.yaml",
            mapper_v2.load_run_config,
        ),
        (
            "v2_1_sparse_d384_l4_phase_b",
            "configs/training/stage2_mapper_v2_1_phase_b_sparse_global_mps.yaml",
            mapper_v2_1.load_run_config,
        ),
    ),
)
def test_mapper_training_hydra_composes_legacy_equivalent_config(
    mapper_preset: str,
    legacy_path: str,
    legacy_loader: object,
) -> None:
    config = compose_training_experiment_config(overrides=[f"training/mapper={mapper_preset}"])

    structured = validate_training_experiment_config(config)
    assert isinstance(OmegaConf.to_object(structured), TrainingExperimentConfig)
    assert training_experiment_config_to_legacy_dict(config) == legacy_loader(legacy_path)


@pytest.mark.parametrize(
    "override",
    (
        "+unexpected_section=1",
        "+data.unexpected_path=artifacts/nope",
        "+runtime.unexpected_device_flag=true",
        "+model.unexpected_width=12",
        "+control_model.unexpected_width=12",
        "+loss.unexpected_weight=0.5",
    ),
)
def test_mapper_training_hydra_rejects_unknown_keys(override: str) -> None:
    with pytest.raises((ConfigAttributeError, ConfigKeyError, ValueError)):
        compose_training_experiment_config(
            overrides=[
                "training/mapper=v2_1_sparse_d384_l4_phase_b",
                override,
            ],
        )


def test_default_mapper_training_hydra_config_uses_v2_tuple_preset() -> None:
    config = compose_training_experiment_config()
    legacy = training_experiment_config_to_legacy_dict(config)

    assert legacy["run_name"] == "stage2_mapper_v2_phase_b_global_d768_l8_b1"
    assert legacy == mapper_v2.load_run_config(
        "configs/training/stage2_mapper_v2_phase_b_global_d768_l8_mps.yaml",
    )
    assert "eval_index_path" not in legacy
    assert "init_from_mapper_checkpoint" not in legacy
    assert "dataset_progress" not in legacy
    assert (default_hydra_config_dir() / "training" / "mapper").is_dir()


def test_mapper_training_hydra_schema_allows_plain_optional_overrides() -> None:
    config = compose_training_experiment_config(
        overrides=[
            "training/mapper=v2_tuple_d384_l4_phase_b",
            "output.resume_from=artifacts/example.pt",
            "runtime.dataset_progress=true",
        ],
    )

    legacy = training_experiment_config_to_legacy_dict(config)

    assert legacy["resume_from"] == "artifacts/example.pt"
    assert legacy["dataset_progress"] is True


@pytest.mark.parametrize(
    "override",
    (
        "data.length_bucketed_batches=true",
        "data.length_bucket_size_multiplier=16",
        "output.init_from_mapper_checkpoint=artifacts/example.pt",
    ),
)
def test_v21_hydra_schema_rejects_fields_its_runner_cannot_consume(override: str) -> None:
    with pytest.raises(ValueError, match="v2_1 mapper does not support training config field"):
        compose_training_experiment_config(
            overrides=[
                "training/mapper=v2_1_sparse_d384_l4_phase_b",
                override,
            ],
        )


def test_v2_hydra_schema_allows_v2_specific_fields() -> None:
    config = compose_training_experiment_config(
        overrides=[
            "training/mapper=v2_tuple_d384_l4_phase_b",
            "data.length_bucketed_batches=false",
            "output.init_from_mapper_checkpoint=artifacts/example.pt",
        ],
    )

    legacy = training_experiment_config_to_legacy_dict(config)

    assert legacy["length_bucketed_batches"] is False
    assert legacy["init_from_mapper_checkpoint"] == "artifacts/example.pt"


def test_runner_projection_rejects_non_null_unconsumed_fields() -> None:
    def runner(*, accepted: int) -> None:
        del accepted

    with pytest.raises(ValueError, match="runner cannot consume config field.*unconsumed"):
        _call_kwargs(runner, {"accepted": 1, "unconsumed": 2})


def test_mapper_training_hydra_dry_run_does_not_write_cwd_log(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pulsefield_model.training.mapper_training_hydra",
            "--dry-run",
            "training/mapper=v2_1_sparse_d384_l4_phase_b",
            f"output.output_dir={tmp_path / 'out'}",
            "output.resume_from=null",
        ],
        cwd=tmp_path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert "mapper_training_hydra_dry_run" in result.stdout
    assert not (tmp_path / "mapper_training_hydra.log").exists()


@pytest.mark.parametrize(
    ("module", "args", "expected"),
    (
        ("pulsefield_model.training.mapper_v2", ("--hydra-help",), "Hydra ("),
        ("pulsefield_model.training.mapper_v2", ("--help",), "training/mapper"),
        ("pulsefield_model.training.mapper_v2", ("--cfg", "job"), "stage2_mapper_v2_phase_b_global_d384_l4_b2"),
        ("pulsefield_model.training.mapper_v2", ("-c", "job"), "stage2_mapper_v2_phase_b_global_d384_l4_b2"),
        ("pulsefield_model.training.mapper_v2_1", ("--hydra-help",), "Hydra ("),
        ("pulsefield_model.training.mapper_v2_1", ("--help",), "training/mapper"),
        (
            "pulsefield_model.training.mapper_v2_1",
            ("--cfg", "job"),
            "stage2_mapper_v2_1_phase_b_sparse_global_d384_l4_b2",
        ),
    ),
)
def test_mapper_alias_supports_hydra_control_flags(
    module: str,
    args: tuple[str, ...],
    expected: str,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            module,
            *args,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert expected in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    ("args", "expected"),
    (
        (("--package",), "--package/-p"),
        (("-p",), "--package/-p"),
        (("--config-path",), "--config-path/-cp"),
        (("-cp",), "--config-path/-cp"),
        (("--config-name",), "--config-name/-cn"),
        (("-cn",), "--config-name/-cn"),
        (("--config-dir",), "--config-dir/-cd"),
        (("-cd",), "--config-dir/-cd"),
        (("--cfg",), "--cfg/-c"),
        (("-c",), "--cfg/-c"),
    ),
)
def test_mapper_alias_rejects_missing_hydra_flag_values_before_preset_append(
    tmp_path: Path,
    args: tuple[str, ...],
    expected: str,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pulsefield_model.training.mapper_v2",
            *args,
        ],
        cwd=tmp_path,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 2
    assert "usage: mapper training Hydra entrypoint" in result.stderr
    assert f"argument {expected}: expected one argument" in result.stderr
    assert "Traceback" not in result.stderr
    assert "mapper_tuple_control_teacher_cache_precompute" not in result.stdout + result.stderr
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize(
    ("args", "expected"),
    (
        (("--dry-run", "--max-steps", "1"), "run.max_steps=1"),
        (("--resume-from", "artifacts/example.pt"), "output.resume_from=artifacts/example.pt"),
        (("--config", "configs/training/example.yaml"), "training/mapper=<preset>"),
        (
            ("--init-from-control-checkpoint", "artifacts/control.pt"),
            "output.init_from_control_checkpoint=artifacts/control.pt",
        ),
        (
            ("--init-from-mapper-checkpoint", "artifacts/mapper.pt"),
            "output.init_from_mapper_checkpoint=artifacts/mapper.pt",
        ),
        (("--no-include-full-song-context",), "data.include_full_song_context=false"),
        (("--no-dataset-progress",), "runtime.dataset_progress=false"),
        (("--no-length-bucketed-batches",), "data.length_bucketed_batches=false"),
        (("--no-skip-first-eval-pass",), "run.skip_first_eval_pass=false"),
    ),
)
def test_mapper_v2_alias_rejects_deprecated_legacy_training_flags(
    args: tuple[str, ...],
    expected: str,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pulsefield_model.training.mapper_v2",
            *args,
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 2
    assert "Deprecated legacy training flag" in result.stderr
    assert expected in result.stderr
    assert "unrecognized arguments" not in result.stderr
    assert "Traceback" not in result.stderr


def test_mapper_alias_rejects_mapper_group_override_without_traceback() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pulsefield_model.training.mapper_v2",
            "training/mapper=v2_1_sparse_d384_l4_phase_b",
            "--dry-run",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 2
    assert "mapper preset aliases are fixed to training/mapper=v2_tuple_d384_l4_phase_b" in result.stderr
    assert "pulsefield_model.training.mapper_training_hydra" in result.stderr
    assert "Traceback" not in result.stderr


def test_mapper_training_hydra_imports_stay_out_of_legacy_training_modules() -> None:
    restricted_paths = (
        "src/pulsefield_model/training/mapper_v2.py",
        "src/pulsefield_model/training/mapper_v2_1.py",
        "src/pulsefield_model/training/mapper_runner.py",
        "src/pulsefield_model/training/mapper_common.py",
    )

    for path in restricted_paths:
        text = Path(path).read_text(encoding="utf-8")
        assert "import hydra" not in text, path
        assert "from hydra" not in text, path
        assert "omegaconf" not in text.lower(), path
