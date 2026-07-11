from __future__ import annotations

import asyncio
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra import main as hydra_main
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from pulsefield_model.inference.config import (
    InferenceServiceConfig,
    inference_service_config_from_mapping,
    project_to_ws_endpoint_config,
)


CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs" / "inference"
_REGISTERED = False


def register_inference_config_store() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    ConfigStore.instance().store(
        group="schema/inference",
        name="service",
        node=InferenceServiceConfig,
        package="_global_",
    )
    _REGISTERED = True


register_inference_config_store()


@hydra_main(version_base="1.3", config_path="../configs/inference", config_name="service")
def _hydra_main(config: DictConfig) -> None:
    run_ws_server_from_config(inference_service_config_from_hydra(config))


def compose_inference_service_config(overrides: Sequence[str] | None = None) -> InferenceServiceConfig:
    with initialize_config_dir(version_base="1.3", config_dir=CONFIGS_DIR.resolve().as_posix()):
        cfg = compose(config_name="service", overrides=list(overrides or ()))
    return inference_service_config_from_hydra(cfg)


def inference_service_config_from_hydra(config: DictConfig) -> InferenceServiceConfig:
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, Mapping):
        raise TypeError(f"inference config must resolve to a mapping, got {type(resolved).__name__}")
    return inference_service_config_from_mapping(resolved)


def run_ws_server_from_config(config: InferenceServiceConfig) -> int:
    from pulsefield_model.inference.ws_endpoint import InferenceEndpoint
    from pulsefield_model.inference.ws_server import serve_forever

    asyncio.run(serve_forever(InferenceEndpoint(config=project_to_ws_endpoint_config(config))))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    with _patched_argv([sys.argv[0], *args]):
        _hydra_main()
    return 0


@contextmanager
def _patched_argv(argv: Sequence[str]) -> object:
    original = sys.argv
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = original


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIGS_DIR",
    "compose_inference_service_config",
    "inference_service_config_from_hydra",
    "main",
    "register_inference_config_store",
    "run_ws_server_from_config",
]
