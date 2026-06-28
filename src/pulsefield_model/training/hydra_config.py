from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


@dataclass
class ExperimentSection:
    name: str = ""
    mapper: str = ""
    phase: str = ""
    preset: str = ""
    legacy_config_path: str | None = None


@dataclass
class RunSection:
    run_name: str = ""
    max_steps: int | None = None
    eval_every: int | None = None
    save_every: int | None = None
    log_every: int | None = None
    mps_cleanup_every: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    weight_decay: float | None = None
    skip_first_eval_pass: bool | None = None


@dataclass
class DataSection:
    dataset_root: str | None = None
    index_path: str | None = None
    eval_index_path: str | None = None
    control_v3_timeseries_path: str | None = None
    eval_fraction: float | None = None
    eval_size: int | None = None
    final_train_eval_size: int | None = None
    mapper_record_cache_path: str | None = None
    include_full_song_context: bool | None = None
    length_bucketed_batches: bool | None = None
    length_bucket_size_multiplier: int | None = None
    control_teacher_cache_dir: str | None = None
    precompute_control_teacher_cache: bool | None = None
    precompute_control_teacher_cache_only: bool | None = None
    control_teacher_precompute_batch_size: int | None = None
    require_control_teacher_cache: bool | None = None
    control_teacher_cache_overwrite: bool | None = None


@dataclass
class RuntimeSection:
    seed: int | None = None
    device: str | None = None
    num_workers: int | None = None
    max_cached_maps: int | None = None
    dataset_progress: bool | None = None


@dataclass
class OutputSection:
    output_dir: str | None = None
    init_from_control_checkpoint: str | None = None
    init_from_mapper_checkpoint: str | None = None
    resume_from: str | None = None


@dataclass
class TrainingExperimentConfig:
    dry_run: bool = False
    experiment: ExperimentSection = field(default_factory=ExperimentSection)
    run: RunSection = field(default_factory=RunSection)
    data: DataSection = field(default_factory=DataSection)
    runtime: RuntimeSection = field(default_factory=RuntimeSection)
    output: OutputSection = field(default_factory=OutputSection)
    control_model: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    loss: dict[str, Any] = field(default_factory=dict)


_TRAINING_EXPERIMENT_SCHEMA_NAME = "training_experiment_schema"

_MAPPER_KIND_BY_EXPERIMENT_VALUE = {
    "v2": "v2",
    "v2_tuple": "v2",
    "mapper_v2": "v2",
    "v2_1": "v2_1",
    "v2_1_sparse": "v2_1",
    "mapper_v2_1": "v2_1",
}

_LEGACY_SECTION_FIELDS = {
    "run": (
        "run_name",
        "max_steps",
        "eval_every",
        "save_every",
        "log_every",
        "mps_cleanup_every",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "skip_first_eval_pass",
    ),
    "data": (
        "dataset_root",
        "index_path",
        "eval_index_path",
        "control_v3_timeseries_path",
        "eval_fraction",
        "eval_size",
        "final_train_eval_size",
        "mapper_record_cache_path",
        "include_full_song_context",
        "length_bucketed_batches",
        "length_bucket_size_multiplier",
        "control_teacher_cache_dir",
        "precompute_control_teacher_cache",
        "precompute_control_teacher_cache_only",
        "control_teacher_precompute_batch_size",
        "require_control_teacher_cache",
        "control_teacher_cache_overwrite",
    ),
    "runtime": (
        "seed",
        "device",
        "num_workers",
        "max_cached_maps",
        "dataset_progress",
    ),
    "output": (
        "output_dir",
        "init_from_control_checkpoint",
        "init_from_mapper_checkpoint",
        "resume_from",
    ),
}
_TRAINING_EXPERIMENT_TOP_LEVEL_FIELDS = frozenset(
    (
        "dry_run",
        "experiment",
        "run",
        "data",
        "runtime",
        "output",
        "control_model",
        "model",
        "loss",
    )
)
_EXPERIMENT_FIELDS = frozenset(("name", "mapper", "phase", "preset", "legacy_config_path"))


def register_training_experiment_config() -> None:
    from hydra.core.config_store import ConfigStore

    ConfigStore.instance().store(
        name=_TRAINING_EXPERIMENT_SCHEMA_NAME,
        node=TrainingExperimentConfig,
    )


register_training_experiment_config()


def default_hydra_config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "hydra"


def compose_training_experiment_config(
    *,
    config_name: str = "mapper_training",
    overrides: Sequence[str] = (),
    config_dir: Path | None = None,
) -> Any:
    from hydra import compose, initialize_config_dir

    register_training_experiment_config()
    resolved_config_dir = default_hydra_config_dir() if config_dir is None else Path(config_dir).resolve()
    with initialize_config_dir(version_base=None, config_dir=resolved_config_dir.as_posix()):
        config = compose(config_name=config_name, overrides=list(overrides))
    validate_training_experiment_config(config)
    return config


def validate_training_experiment_config(config: Any) -> Any:
    from omegaconf import OmegaConf

    structured = OmegaConf.merge(OmegaConf.structured(TrainingExperimentConfig), config)
    _validate_known_training_experiment_keys(structured)
    mapper_kind = mapper_training_kind(structured)
    if structured.experiment.phase != "B":
        raise ValueError(f"unsupported mapper training phase: {structured.experiment.phase!r}")
    _validate_model_sections(structured, mapper_kind=mapper_kind)
    return structured


def mapper_training_kind(config: Any) -> str:
    from omegaconf import OmegaConf

    container = OmegaConf.to_container(config, resolve=True)
    if not isinstance(container, Mapping):
        raise ValueError("training experiment config must be a mapping")
    experiment = container.get("experiment")
    if not isinstance(experiment, Mapping):
        raise ValueError("training experiment config requires an experiment section")
    raw_mapper = experiment.get("mapper")
    if not isinstance(raw_mapper, str) or not raw_mapper:
        raise ValueError("training experiment config requires experiment.mapper")
    try:
        return _MAPPER_KIND_BY_EXPERIMENT_VALUE[raw_mapper]
    except KeyError as exc:
        raise ValueError(f"unknown mapper training experiment mapper: {raw_mapper!r}") from exc


def training_experiment_config_to_legacy_dict(config: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    validate_training_experiment_config(config)
    container = OmegaConf.to_container(config, resolve=True)
    if not isinstance(container, Mapping):
        raise ValueError("training experiment config must be a mapping")

    legacy: dict[str, Any] = {}
    explicit_preset_fields = _explicit_preset_legacy_fields(container)
    for section_name, allowed_fields in _LEGACY_SECTION_FIELDS.items():
        section = container.get(section_name, {})
        if section is None:
            continue
        if not isinstance(section, Mapping):
            raise ValueError(f"training experiment {section_name} section must be a mapping")
        unknown = sorted(set(section) - set(allowed_fields))
        if unknown:
            raise ValueError(f"unknown training experiment {section_name} keys: {unknown}")
        for key, value in section.items():
            if value is None and key not in explicit_preset_fields.get(section_name, set()):
                continue
            legacy[key] = value

    for section_name in ("model", "control_model", "loss"):
        section = container.get(section_name, {})
        if section is None:
            section = {}
        if not isinstance(section, Mapping):
            raise ValueError(f"training experiment {section_name} section must be a mapping")
        legacy[section_name] = dict(section)

    return legacy


def write_training_experiment_config_artifacts(config: Any, *, output_dir: Path) -> None:
    from omegaconf import OmegaConf

    resolved = validate_training_experiment_config(config)
    legacy = training_experiment_config_to_legacy_dict(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=resolved, f=output_dir / "hydra_resolved_config.yaml")
    (output_dir / "legacy_run_config.yaml").write_text(
        yaml.safe_dump(legacy, sort_keys=False),
        encoding="utf-8",
    )


def _validate_model_sections(config: Any, *, mapper_kind: str) -> None:
    from omegaconf import OmegaConf

    from pulsefield_model.models.control import ControlDemoGlobalEncoderConfig
    from pulsefield_model.models.mapper.v2 import MapperV2Config
    from pulsefield_model.models.mapper.v2_1 import MapperV21Config, MapperV21LossConfig
    from pulsefield_model.training.mapper_common import MapperTuplePhaseBLossConfig

    model_config_type: type[Any]
    loss_config_type: type[Any]
    if mapper_kind == "v2":
        model_config_type = MapperV2Config
        loss_config_type = MapperTuplePhaseBLossConfig
    elif mapper_kind == "v2_1":
        model_config_type = MapperV21Config
        loss_config_type = MapperV21LossConfig
    else:
        raise ValueError(f"unknown mapper training kind: {mapper_kind!r}")

    OmegaConf.merge(OmegaConf.structured(ControlDemoGlobalEncoderConfig), config.control_model)
    OmegaConf.merge(OmegaConf.structured(model_config_type), config.model)
    OmegaConf.merge(OmegaConf.structured(loss_config_type), config.loss)


def _validate_known_training_experiment_keys(config: Any) -> None:
    from omegaconf import OmegaConf

    container = OmegaConf.to_container(config, resolve=True)
    if not isinstance(container, Mapping):
        raise ValueError("training experiment config must be a mapping")

    unknown = sorted(set(container) - _TRAINING_EXPERIMENT_TOP_LEVEL_FIELDS)
    if unknown:
        raise ValueError(f"unknown training experiment keys: {unknown}")

    experiment = container.get("experiment", {})
    if not isinstance(experiment, Mapping):
        raise ValueError("training experiment section must be a mapping")
    unknown_experiment = sorted(set(experiment) - _EXPERIMENT_FIELDS)
    if unknown_experiment:
        raise ValueError(f"unknown training experiment experiment keys: {unknown_experiment}")

    for section_name, allowed_fields in _LEGACY_SECTION_FIELDS.items():
        section = container.get(section_name, {})
        if section is None:
            continue
        if not isinstance(section, Mapping):
            raise ValueError(f"training experiment {section_name} section must be a mapping")
        unknown_section = sorted(set(section) - set(allowed_fields))
        if unknown_section:
            raise ValueError(f"unknown training experiment {section_name} keys: {unknown_section}")


def _explicit_preset_legacy_fields(container: Mapping[str, Any]) -> dict[str, set[str]]:
    experiment = container.get("experiment")
    if not isinstance(experiment, Mapping):
        return {}
    preset = experiment.get("preset")
    if not isinstance(preset, str) or not preset:
        return {}

    preset_path = default_hydra_config_dir() / "training" / "mapper" / f"{preset}.yaml"
    try:
        loaded = yaml.safe_load(preset_path.read_text(encoding="utf-8"))
    except OSError:
        return {}
    if not isinstance(loaded, Mapping):
        return {}

    fields_by_section: dict[str, set[str]] = {}
    for section_name in _LEGACY_SECTION_FIELDS:
        section = loaded.get(section_name)
        if isinstance(section, Mapping):
            fields_by_section[section_name] = {str(key) for key in section}
    return fields_by_section
