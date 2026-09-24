"""Packaged process boundary for matched persistent-intent experiments."""
from dataclasses import fields
import json

from hydra import compose, initialize_config_module, main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .intent_config import IntentTrainConfig

ConfigStore.instance().store(name='joint_audio_intent_schema', node=IntentTrainConfig)


def project_config(config):
    values = OmegaConf.to_container(config, resolve=True)
    if set(values) != {f.name for f in fields(IntentTrainConfig)}:
        raise ValueError('Unknown or missing intent experiment setting')
    result = IntentTrainConfig(**values)
    result.validate()
    return result


def compose_config(overrides=None):
    with initialize_config_module(version_base='1.3', config_module='ensomi_model.configs.hydra'):
        return project_config(compose(config_name='joint_audio_intent', overrides=overrides or []))


@main(version_base='1.3', config_path='../../configs/hydra', config_name='joint_audio_intent')
def cli(config: DictConfig):
    from .intent_training import train
    result = train(project_config(config), resolved_yaml=OmegaConf.to_yaml(config, resolve=True))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    cli()
