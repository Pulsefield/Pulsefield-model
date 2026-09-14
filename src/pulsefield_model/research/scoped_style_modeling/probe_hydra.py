"""Torch-free Hydra entrypoint for scoped-style probes."""
from dataclasses import fields
import json

from hydra import compose, initialize_config_module, main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .config import ModelConfig
from .dataset import ContractError
from .probe_config import ProbeConfig

ConfigStore.instance().store(name='scoped_style_probe_schema', node=ProbeConfig)


def project_config(config: DictConfig) -> ProbeConfig:
    values = OmegaConf.to_container(config, resolve=True)
    for values_part, cls in ((values, ProbeConfig), (values['model'], ModelConfig)):
        unknown = set(values_part) - {f.name for f in fields(cls)}
        if unknown:
            raise ContractError(f'Unknown probe configuration fields: {sorted(unknown)}')
    result = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(ProbeConfig), config))
    result.validate()
    return result


def compose_config(overrides=None, *, config_name='scoped_style_probe'):
    with initialize_config_module(version_base='1.3', config_module='pulsefield_model.configs.hydra'):
        return project_config(compose(config_name=config_name, overrides=overrides or []))


@main(version_base='1.3', config_path='../../configs/hydra', config_name='scoped_style_probe')
def cli(config: DictConfig):
    from .probes import run_probe
    print(json.dumps(run_probe(project_config(config), resolved_yaml=OmegaConf.to_yaml(config, resolve=True)), indent=2))


if __name__ == '__main__':
    cli()
