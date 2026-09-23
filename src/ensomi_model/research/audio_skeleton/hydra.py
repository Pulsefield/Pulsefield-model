"""Packaged configuration boundary for skeleton experiments."""
from dataclasses import fields
import json

from hydra import compose, initialize_config_module, main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .config import ExperimentConfig

ConfigStore.instance().store(name='audio_skeleton_schema', node=ExperimentConfig)


def project_config(config):
    values = OmegaConf.to_container(config, resolve=True)
    if set(values) != {f.name for f in fields(ExperimentConfig)}:
        raise ValueError('Unknown or missing skeleton configuration field')
    result = ExperimentConfig(**values)
    result.validate()
    return result


def compose_config(overrides=None):
    with initialize_config_module(version_base='1.3', config_module='ensomi_model.configs.hydra'):
        return project_config(compose(config_name='audio_skeleton', overrides=overrides or []))


@main(version_base='1.3', config_path='../../configs/hydra', config_name='audio_skeleton')
def cli(config: DictConfig):
    settings = project_config(config)
    if settings.mode == 'prepare':
        from .corpus import prepare
        result = prepare(settings)
    elif settings.mode == 'sensitivity':
        from .sensitivity import run
        result = run(settings)
    elif settings.mode == 'cache':
        from .audio import cache
        result = cache(settings)
    else:
        from .training import run
        result = run(settings, resolved_yaml=OmegaConf.to_yaml(config, resolve=True))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    cli()
