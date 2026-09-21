"""Hydra process boundary for the frozen scoped-style data preparation preset."""
from __future__ import annotations

from dataclasses import fields
import json

from hydra import compose, initialize_config_module, main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from .dataset import ContractError
from .prepare import PrepareConfig, run_preparation

ConfigStore.instance().store(name="scoped_style_prepare_schema", node=PrepareConfig)


def project_config(config: DictConfig) -> PrepareConfig:
    values = OmegaConf.to_container(config, resolve=True)
    unknown = set(values) - {f.name for f in fields(PrepareConfig)}
    if unknown:
        raise ContractError(f"Unknown preparation fields: {sorted(unknown)}")
    result = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(PrepareConfig), config))
    if result.workers < 1 or result.timeout_seconds <= 0:
        raise ContractError("workers and timeout_seconds must be positive")
    for key in ("dataset_dir", "source_cache", "output_dir", "split_manifest_path"):
        if not getattr(result, key):
            raise ContractError(f"{key} must be a nonempty path")
    return result


def compose_config(overrides: list[str] | None = None) -> PrepareConfig:
    with initialize_config_module(version_base="1.3", config_module="pulsefield_model.configs.hydra"):
        return project_config(compose(config_name="scoped_style_prepare", overrides=overrides or []))


@main(version_base="1.3", config_path="../../configs/hydra", config_name="scoped_style_prepare")
def cli(config: DictConfig) -> None:
    summary = run_preparation(project_config(config))
    print(json.dumps({k: summary[k] for k in ("data_contract_ready", "verified_source_count", "replayed_source_count", "assessment_cells", "split_sha256")}, indent=2))
    if not summary["data_contract_ready"]:
        raise ContractError("Data contracts failed; inspect summary.json and the source/evidence error reports")


if __name__ == "__main__":
    cli()
