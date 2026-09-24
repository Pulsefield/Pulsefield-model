"""Packaged, Torch-free CLI boundary; stdout is the ordered JSONL event stream."""
from dataclasses import fields
import json

from hydra import compose, initialize_config_module, main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .inference_config import AudioInferenceConfig

ConfigStore.instance().store(name='planned_audio_inference_schema', node=AudioInferenceConfig)


def project_config(config):
    values = OmegaConf.to_container(config, resolve=True)
    if set(values) != {f.name for f in fields(AudioInferenceConfig)}:
        raise ValueError('Unknown or missing planned audio inference setting')
    result = AudioInferenceConfig(**values)
    result.validate()
    return result


def compose_config(overrides=None):
    with initialize_config_module(version_base='1.3', config_module='ensomi_model.configs.inference'):
        return project_config(compose(config_name='planned_audio', overrides=overrides or []))


@main(version_base='1.3', config_path='../../configs/inference', config_name='planned_audio')
def cli(config: DictConfig):
    from .inference import infer_audio
    infer_audio(project_config(config), resolved_yaml=OmegaConf.to_yaml(config, resolve=True),
                on_event=lambda event: print(json.dumps(event, allow_nan=False), flush=True))


if __name__ == '__main__':
    cli()
