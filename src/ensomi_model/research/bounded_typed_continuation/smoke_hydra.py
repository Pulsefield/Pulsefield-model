"""Packaged Hydra boundary for the bounded three-arm learning check."""
from dataclasses import fields
import json

from hydra import compose, initialize_config_module, main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from ..scoped_style_modeling.dataset import ContractError
from .model import ModelConfig
from .smoke_config import SmokeConfig, SmokeResources


def schema():
    result = OmegaConf.structured(SmokeConfig)
    for part in ('model', 'resources'):
        OmegaConf.set_readonly(result[part], False)
    return result


ConfigStore.instance().store(name='bounded_typed_smoke_schema', node=schema())


def project_config(config: DictConfig):
    values = OmegaConf.to_container(config, resolve=True)
    for part, cls in ((values, SmokeConfig), (values['model'], ModelConfig), (values['resources'], SmokeResources)):
        unknown = set(part) - {field.name for field in fields(cls)}
        if unknown:
            raise ContractError(f'Unknown bounded learning-check fields: {sorted(unknown)}')
    result = OmegaConf.to_object(OmegaConf.merge(schema(), config))
    result.validate()
    return result


def compose_config(overrides=None):
    with initialize_config_module(version_base='1.3', config_module='ensomi_model.configs.hydra'):
        return project_config(compose(config_name='bounded_typed_smoke', overrides=overrides or []))


@main(version_base='1.3', config_path='../../configs/hydra', config_name='bounded_typed_smoke')
def cli(config: DictConfig):
    from .smoke_run import run_smoke
    result = run_smoke(project_config(config), resolved_yaml=OmegaConf.to_yaml(config, resolve=True))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    cli()
