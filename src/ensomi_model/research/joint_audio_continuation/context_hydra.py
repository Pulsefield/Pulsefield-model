"""Packaged configuration boundary for full-audio interval experiments."""
from dataclasses import fields
import json

from hydra import compose, initialize_config_module, main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .context_config import ContextTrainConfig

ConfigStore.instance().store(name='joint_audio_context_schema', node=ContextTrainConfig)


def project_config(config):
    values = OmegaConf.to_container(config, resolve=True)
    if set(values) != {f.name for f in fields(ContextTrainConfig)}:
        raise ValueError('Unknown or missing context experiment setting')
    result = ContextTrainConfig(**values)
    result.validate()
    return result


def compose_config(overrides=None):
    with initialize_config_module(version_base='1.3', config_module='ensomi_model.configs.hydra'):
        return project_config(compose(config_name='joint_audio_context', overrides=overrides or []))


@main(version_base='1.3', config_path='../../configs/hydra', config_name='joint_audio_context')
def cli(config: DictConfig):
    from .context_training import train
    result = train(project_config(config), resolved_yaml=OmegaConf.to_yaml(config, resolve=True))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    cli()
