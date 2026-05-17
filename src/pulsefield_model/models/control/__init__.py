"""Control model components."""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "CONTEXT_LENGTH_FRAMES": "context",
    "TARGET_OFFSET_IN_CONTEXT": "context",
    "TARGET_WINDOW_LENGTH_FRAMES": "context",
    "prepare_control_context_batch": "context",
    "ControlEncoder": "encoder",
    "ControlEncoderConfig": "encoder",
    "ControlEncoderOutput": "encoder",
    "ControlLossConfig": "loss",
    "ControlLossOutput": "loss",
    "ControlModelLoss": "loss",
    "ControlDemoEncoder": "demo",
    "ControlDemoEncoderConfig": "demo",
    "ControlDemoEncoderOutput": "demo",
    "ControlDemoLossConfig": "demo_loss",
    "ControlDemoLossOutput": "demo_loss",
    "ControlDemoModelLoss": "demo_loss",
    "ControlDemoGlobalEncoder": "demo_global",
    "ControlDemoGlobalEncoderConfig": "demo_global",
    "ControlDemoGlobalEncoderOutput": "demo_global",
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
