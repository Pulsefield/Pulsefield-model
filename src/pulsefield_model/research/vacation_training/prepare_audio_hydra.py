"""Packaged boundary for the immutable TRAIN audio inventory."""
from dataclasses import dataclass, fields
import json

from hydra import main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from ..scoped_style_modeling.dataset import ContractError


@dataclass
class PrepareAudioConfig:
    plan_file: str = ''
    plan_sha256: str = ''
    catalog_file: str = ''
    catalog_sha256: str = ''
    catalog_root: str = ''
    source_cache_dir: str = ''
    output_file: str = ''


ConfigStore.instance().store(name='vacation_audio_inputs_schema', node=PrepareAudioConfig)


@main(version_base='1.3', config_path='../../configs/hydra', config_name='vacation_audio_inputs')
def cli(config: DictConfig):
    from .audio_inputs import prepare_audio_inputs
    values = OmegaConf.to_container(config, resolve=True)
    if set(values) != {f.name for f in fields(PrepareAudioConfig)} or not all(values.values()):
        raise ContractError('Audio preparation requires every declared path/digest and rejects unknown fields')
    print(json.dumps(prepare_audio_inputs(**values), indent=2))


if __name__ == '__main__':
    cli()
