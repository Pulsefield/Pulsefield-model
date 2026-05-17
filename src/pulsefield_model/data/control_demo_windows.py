from __future__ import annotations

from typing import Any, Sequence

import torch

from pulsefield_model.data.control_windows import collate_control_context_windows
from pulsefield_model.features.control_v3_targets import MODEL_FEATURE_NAMES


CONTROL_DEMO_VALUE_FEATURE_NAMES = ("density_level",)
CONTROL_DEMO_CONFIDENCE_FEATURE_NAMES = ("density_confidence",)
CONTROL_DEMO_TARGET_FEATURE_NAMES = CONTROL_DEMO_VALUE_FEATURE_NAMES + CONTROL_DEMO_CONFIDENCE_FEATURE_NAMES
CONTROL_DEMO_TARGET_CHANNELS = len(CONTROL_DEMO_TARGET_FEATURE_NAMES)
DENSITY_LEVEL_TARGET_INDEX = 0
DENSITY_CONFIDENCE_TARGET_INDEX = 1

_CONTROL_DEMO_SOURCE_INDEXES = tuple(MODEL_FEATURE_NAMES.index(name) for name in CONTROL_DEMO_TARGET_FEATURE_NAMES)


def extract_control_demo_target(control_v3_target: torch.Tensor) -> torch.Tensor:
    if not isinstance(control_v3_target, torch.Tensor):
        raise ValueError("control_v3_target must be a torch.Tensor")
    if control_v3_target.ndim < 2:
        raise ValueError("control_v3_target must have at least frame and channel dimensions")
    if int(control_v3_target.shape[-1]) != len(MODEL_FEATURE_NAMES):
        raise ValueError(f"control_v3_target must have {len(MODEL_FEATURE_NAMES)} channels")
    return control_v3_target[..., list(_CONTROL_DEMO_SOURCE_INDEXES)].contiguous()


def collate_control_demo_context_windows(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    batch = collate_control_context_windows(samples)
    batch["control_demo_target"] = extract_control_demo_target(batch.pop("control_v3_target"))
    batch.pop("ln_change_n_eff_target", None)
    return batch
