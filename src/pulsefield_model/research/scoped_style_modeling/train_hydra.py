"""Torch-free Hydra process boundary for paired scoped-style training."""
from dataclasses import fields
import json

from hydra import compose, initialize_config_module, main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .config import TrainConfig, ModelConfig
from .dataset import ContractError

ConfigStore.instance().store(name="scoped_style_train_schema", node=TrainConfig)


def project_config(config: DictConfig) -> TrainConfig:
    values = OmegaConf.to_container(config, resolve=True)
    unknown = set(values)-{f.name for f in fields(TrainConfig)}
    if unknown:
        raise ContractError(f"Unknown training fields: {sorted(unknown)}")
    unknown_model = set(values["model"])-{f.name for f in fields(ModelConfig)}
    if unknown_model:
        raise ContractError(f"Unknown model fields: {sorted(unknown_model)}")
    result = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(TrainConfig), config))
    result.validate()
    return result


def compose_config(overrides: list[str] | None = None, *, config_name: str = "scoped_style_train") -> TrainConfig:
    with initialize_config_module(version_base="1.3", config_module="pulsefield_model.configs.hydra"):
        return project_config(compose(config_name=config_name, overrides=overrides or []))


@main(version_base="1.3", config_path="../../configs/hydra", config_name="scoped_style_train")
def cli(config: DictConfig) -> None:
    from .train import run_training
    print(json.dumps(run_training(project_config(config), resolved_yaml=OmegaConf.to_yaml(config, resolve=True)), indent=2))


if __name__ == "__main__":
    cli()
