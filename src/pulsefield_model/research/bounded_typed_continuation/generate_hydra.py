"""Packaged Hydra boundary for native generation from external conditions."""
from dataclasses import fields
import json

from hydra import compose, initialize_config_module, main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from ..scoped_style_modeling.dataset import ContractError
from .generate_config import GenerateConfig
from .smoke_config import SmokeResources


def schema():
    result = OmegaConf.structured(GenerateConfig)
    OmegaConf.set_readonly(result.resources, False)
    return result


ConfigStore.instance().store(name='bounded_typed_generate_schema', node=schema())


def project_config(config):
    values = OmegaConf.to_container(config, resolve=True)
    for value, cls in ((values, GenerateConfig), (values['resources'], SmokeResources)):
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ContractError(f'Unknown bounded generation fields: {sorted(unknown)}')
    result = OmegaConf.to_object(OmegaConf.merge(schema(), config))
    result.validate()
    return result


def compose_config(overrides=None):
    with initialize_config_module(version_base='1.3', config_module='pulsefield_model.configs.hydra'):
        return project_config(compose(config_name='bounded_typed_generate', overrides=overrides or []))


@main(version_base='1.3', config_path='../../configs/hydra', config_name='bounded_typed_generate')
def cli(config: DictConfig):
    from .generate_run import run_generation
    print(json.dumps(run_generation(project_config(config), resolved_yaml=OmegaConf.to_yaml(config, resolve=True)), indent=2))


if __name__ == '__main__':
    cli()
