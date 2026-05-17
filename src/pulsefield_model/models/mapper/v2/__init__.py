"""Mapper v2 global-context model."""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "MapperV2Config": "model",
    "MapperV2ForwardOutput": "model",
    "MapperV2Model": "model",
    "MapperV2ModelOutput": "model",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
