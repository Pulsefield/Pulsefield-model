"""Packaged process boundary for joint head, release and row fitting."""
from dataclasses import fields
import json

from hydra import compose, initialize_config_module, main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .config import PlannedTrainConfig

ConfigStore.instance().store(name='planned_audio_schema', node=PlannedTrainConfig)


def project_config(config):
    values = OmegaConf.to_container(config, resolve=True)
    if set(values) != {f.name for f in fields(PlannedTrainConfig)}:
        raise ValueError('Unknown or missing planned audio setting')
    result = PlannedTrainConfig(**values)
    result.validate()
    return result


def compose_config(overrides=None):
    with initialize_config_module(version_base='1.3', config_module='ensomi_model.configs.hydra'):
        return project_config(compose(config_name='planned_audio', overrides=overrides or []))


@main(version_base='1.3', config_path='../../configs/hydra', config_name='planned_audio')
def cli(config: DictConfig):
    from .training import train
    result = train(project_config(config), resolved_yaml=OmegaConf.to_yaml(config, resolve=True))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    cli()
