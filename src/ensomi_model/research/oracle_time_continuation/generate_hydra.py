"""Packaged Hydra boundary for durable oracle-time generation."""
from dataclasses import fields
import json

from hydra import compose, initialize_config_module, main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from ..scoped_style_modeling.dataset import ContractError
from .decoding import DecodeSamplingPolicy
from .generate_run import GenerateConfig
from .runtime import ResourceConfig
from .storage import SourceCacheConfig

CONFIG_TYPES = {'decode': DecodeSamplingPolicy, 'cache': SourceCacheConfig, 'resources': ResourceConfig}


def _schema():
    schema = OmegaConf.structured(GenerateConfig)
    for name in CONFIG_TYPES:
        OmegaConf.set_readonly(schema[name], False)
    return schema


ConfigStore.instance().store(name='oracle_time_generate_schema', node=_schema())


def project_config(config):
    values = OmegaConf.to_container(config, resolve=True)
    for part, cls in [(values, GenerateConfig)] + [(values[name], cls) for name, cls in CONFIG_TYPES.items()]:
        unknown = set(part) - {field.name for field in fields(cls)}
        if unknown:
            raise ContractError(f'Unknown oracle-time generation fields: {sorted(unknown)}')
    return OmegaConf.to_object(OmegaConf.merge(_schema(), config))


def compose_config(overrides=None, *, config_name='oracle_time_generate'):
    with initialize_config_module(version_base='1.3', config_module='ensomi_model.configs.hydra'):
        return project_config(compose(config_name=config_name, overrides=overrides or []))


@main(version_base='1.3', config_path='../../configs/hydra', config_name='oracle_time_generate')
def cli(config: DictConfig):
    from .generate_run import run_generation
    print(json.dumps(run_generation(project_config(config), resolved_yaml=OmegaConf.to_yaml(config, resolve=True)), indent=2))


if __name__ == '__main__':
    cli()
