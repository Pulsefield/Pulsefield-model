"""Packaged configuration boundary for joint timed-row experiments."""
from dataclasses import fields
import json

from hydra import compose, initialize_config_module, main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .config import JointConfig

ConfigStore.instance().store(name='joint_audio_schema', node=JointConfig)


def project_config(config):
    values = OmegaConf.to_container(config, resolve=True)
    if set(values) != {f.name for f in fields(JointConfig)}:
        raise ValueError('Unknown or missing joint experiment setting')
    result = JointConfig(**values)
    result.validate()
    return result


def compose_config(overrides=None):
    with initialize_config_module(version_base='1.3', config_module='ensomi_model.configs.hydra'):
        return project_config(compose(config_name='joint_audio', overrides=overrides or []))


@main(version_base='1.3', config_path='../../configs/hydra', config_name='joint_audio')
def cli(config: DictConfig):
    settings = project_config(config)
    if settings.mode == 'prepare':
        from .data import prepare
        result = prepare(settings)
    elif settings.mode == 'train':
        from .training import train
        result = train(settings, resolved_yaml=OmegaConf.to_yaml(config, resolve=True))
    elif settings.mode == 'generate':
        from .generation import generate
        result = generate(settings, resolved_yaml=OmegaConf.to_yaml(config, resolve=True))
    else:
        from .generation import infer_audio
        result = infer_audio(settings, resolved_yaml=OmegaConf.to_yaml(config, resolve=True))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    cli()
