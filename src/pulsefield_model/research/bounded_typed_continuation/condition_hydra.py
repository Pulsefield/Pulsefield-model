"""Prepare a timing/seed condition from a verified source without a corpus lookup."""
from dataclasses import fields
import json

from hydra import compose, initialize_config_module, main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from ..scoped_style_modeling.dataset import ContractError
from .generate_config import PrepareConditionConfig

ConfigStore.instance().store(name='bounded_typed_condition_schema', node=PrepareConditionConfig)


def project_config(config):
    unknown = set(config) - {f.name for f in fields(PrepareConditionConfig)}
    if unknown:
        raise ContractError(f'Unknown condition preparation fields: {sorted(unknown)}')
    result = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(PrepareConditionConfig), config))
    result.validate()
    return result


def compose_config(overrides=None):
    with initialize_config_module(version_base='1.3', config_module='pulsefield_model.configs.hydra'):
        return project_config(compose(config_name='bounded_typed_condition', overrides=overrides or []))


@main(version_base='1.3', config_path='../../configs/hydra', config_name='bounded_typed_condition')
def cli(config: DictConfig):
    from .condition import write_source_condition
    from .contract import Arm
    settings = project_config(config)
    if any(not getattr(settings, key) for key in ('source_file', 'source_sha256', 'output_file')):
        raise ContractError('Preparation requires source_file, source_sha256 and a fresh output_file')
    print(json.dumps(write_source_condition(settings.source_file, settings.source_sha256, settings.output_file,
                                            arm=Arm(settings.arm), seed_notes=settings.seed_notes), indent=2))


if __name__ == '__main__':
    cli()
