"""Packaged configuration boundary for the serial vacation worker."""
from dataclasses import fields, is_dataclass
import json

from hydra import compose, initialize_config_module, main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from ..scoped_style_modeling.dataset import ContractError
from .config import VacationConfig


def schema():
    value = OmegaConf.structured(VacationConfig)
    def unlock(node):
        if isinstance(node, DictConfig):
            OmegaConf.set_readonly(node, False)
            for child in node.values():
                unlock(child)
    unlock(value)
    return value


ConfigStore.instance().store(name='vacation_training_schema', node=schema())


def project_config(config):
    merged = OmegaConf.merge(schema(), config)
    value = OmegaConf.to_object(merged)
    def check(node, raw):
        if is_dataclass(node):
            extra = set(raw) - {f.name for f in fields(node)}
            if extra:
                raise ContractError(f'Unknown vacation fields: {sorted(extra)}')
            for f in fields(node):
                check(getattr(node, f.name), raw[f.name])
    check(value, OmegaConf.to_container(merged, resolve=True))
    value.validate()
    return value


def compose_config(overrides=None):
    with initialize_config_module(version_base='1.3', config_module='pulsefield_model.configs.hydra'):
        return project_config(compose(config_name='vacation_training', overrides=overrides or []))


@main(version_base='1.3', config_path='../../configs/hydra', config_name='vacation_training')
def cli(config: DictConfig):
    from .run import run_queue
    projected = project_config(config)
    result = run_queue(projected, resolved_yaml=OmegaConf.to_yaml(config, resolve=True))
    print(json.dumps(result, indent=2))
    if projected.mode in ('run', 'resume') and result['status'] == 'finished_with_incomplete_stages':
        raise SystemExit(2)


if __name__ == '__main__':
    cli()
