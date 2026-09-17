"""Packaged Hydra process boundary for oracle-time sequence training."""
from dataclasses import fields
import json

from hydra import compose, initialize_config_module, main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from ..scoped_style_modeling.dataset import ContractError
from .config import BackboneConfig
from .objective import ObjectiveConfig
from .training_config import TrainExperimentConfig, TrainingConfig
from .windows import WindowSamplingPolicy

CONFIG_TYPES = {"model": BackboneConfig, "windows": WindowSamplingPolicy,
                "objective": ObjectiveConfig, "training": TrainingConfig}


def _schema():
    schema = OmegaConf.structured(TrainExperimentConfig)
    for name in CONFIG_TYPES:
        OmegaConf.set_readonly(schema[name], False)
    return schema


ConfigStore.instance().store(name="oracle_time_train_schema", node=_schema())


def project_config(config: DictConfig) -> TrainExperimentConfig:
    values = OmegaConf.to_container(config, resolve=True)
    for part, cls in [(values, TrainExperimentConfig)] + [(values[name], cls) for name, cls in CONFIG_TYPES.items()]:
        unknown = set(part) - {field.name for field in fields(cls)}
        if unknown:
            raise ContractError(f"Unknown oracle-time training fields: {sorted(unknown)}")
    result = OmegaConf.to_object(OmegaConf.merge(_schema(), config))
    result.validate()
    return result


def compose_config(overrides: list[str] | None = None) -> TrainExperimentConfig:
    with initialize_config_module(version_base="1.3", config_module="pulsefield_model.configs.hydra"):
        return project_config(compose(config_name="oracle_time_train", overrides=overrides or []))


@main(version_base="1.3", config_path="../../configs/hydra", config_name="oracle_time_train")
def cli(config: DictConfig) -> None:
    from .train_run import run_training
    result = run_training(project_config(config), resolved_yaml=OmegaConf.to_yaml(config, resolve=True))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    cli()
