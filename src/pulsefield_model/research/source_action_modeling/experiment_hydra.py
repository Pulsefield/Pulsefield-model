"""Packaged Hydra entrypoint for the bounded source-action composition study."""
import json
from dataclasses import fields

from hydra import compose, initialize_config_module, main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from ..scoped_style_modeling.dataset import ContractError
from .experiment_config import CompositionExperimentConfig
from .model import ModelConfig

def _schema():
    schema = OmegaConf.structured(CompositionExperimentConfig)
    # Runtime ModelConfig is immutable; startup composition must still allow
    # typed overrides before reconstructing that frozen dataclass.
    OmegaConf.set_readonly(schema.model, False)
    return schema


ConfigStore.instance().store(name="source_action_composition_schema", node=_schema())


def project_config(config: DictConfig):
    values = OmegaConf.to_container(config, resolve=True)
    for part, cls in ((values, CompositionExperimentConfig), (values["model"], ModelConfig)):
        unknown = set(part) - {f.name for f in fields(cls)}
        if unknown:
            raise ContractError(f"Unknown composition configuration fields: {sorted(unknown)}")
    result = OmegaConf.to_object(OmegaConf.merge(_schema(), config))
    result.validate()
    return result


def compose_config(overrides=None, *, config_name="source_action_composition"):
    with initialize_config_module(version_base="1.3", config_module="pulsefield_model.configs.hydra"):
        return project_config(compose(config_name=config_name, overrides=overrides or []))


@main(version_base="1.3", config_path="../../configs/hydra", config_name="source_action_composition")
def cli(config: DictConfig):
    from .experiment import run_experiment
    print(json.dumps(run_experiment(project_config(config), resolved_yaml=OmegaConf.to_yaml(config, resolve=True)), indent=2))


if __name__ == "__main__":
    cli()
