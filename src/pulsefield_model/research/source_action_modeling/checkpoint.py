"""Resumable snapshots for trusted local source-action runs, with a versioned contract."""
from dataclasses import asdict
from pathlib import Path
import random

import numpy as np
import torch

from ..scoped_style_modeling.dataset import ContractError
from .observation import INPUT_CONTRACT


def save_snapshot(path: Path, model, optimizer, sampler, *, update: int, scheduler=None):
    """Exclusively create a snapshot; retain optimizer, sampler and all used RNGs.

    These files use Python serialization and must only be loaded from trusted
    local runs. New input contracts cannot reuse complete-chart checkpoints.
    """
    device = next(model.parameters()).device.type
    rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.random.get_rng_state()}
    if device == "mps":
        rng["mps"] = torch.mps.get_rng_state()
    if device == "cuda":
        rng["cuda"] = torch.cuda.get_rng_state_all()
    payload = {"schema": 1, "input_contract": INPUT_CONTRACT, "model_config": asdict(model.config),
               "encoder_type": f"{type(model.encoder).__module__}.{type(model.encoder).__qualname__}",
               "optimizer_type": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
               "scheduler_type": None if scheduler is None else f"{type(scheduler).__module__}.{type(scheduler).__qualname__}",
               "device_type": device, "model": model.state_dict(), "training": model.training,
               "optimizer": optimizer.state_dict(), "sampler": sampler.state_dict(), "rng": rng,
               "scheduler": None if scheduler is None else scheduler.state_dict(), "update": update}
    with Path(path).open("xb") as stream:
        torch.save(payload, stream)


def load_snapshot(path: Path, model, optimizer, sampler, *, scheduler=None) -> int:
    """Restore on the same device family, including next-sample and next-update state."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected_encoder = f"{type(model.encoder).__module__}.{type(model.encoder).__qualname__}"
    if payload.get("schema") != 1 or payload.get("input_contract") != INPUT_CONTRACT or (
        payload.get("model_config") != asdict(model.config) or payload.get("encoder_type") != expected_encoder
    ):
        raise ContractError("Snapshot model or partial-observation contract mismatch")
    if payload["device_type"] != next(model.parameters()).device.type:
        raise ContractError("Exact RNG restoration requires the same device family")
    if (payload["scheduler"] is None) != (scheduler is None):
        raise ContractError("Snapshot scheduler presence differs from the supplied scheduler")
    if payload.get("optimizer_type") != f"{type(optimizer).__module__}.{type(optimizer).__qualname__}" or (
        payload.get("scheduler_type") != (None if scheduler is None else f"{type(scheduler).__module__}.{type(scheduler).__qualname__}")
    ):
        raise ContractError("Snapshot optimizer or scheduler type mismatch")
    sampler.load_state_dict(payload["sampler"])
    model.load_state_dict(payload["model"])
    model.train(payload["training"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    rng = payload["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.random.set_rng_state(rng["torch"])
    if "mps" in rng:
        torch.mps.set_rng_state(rng["mps"])
    if "cuda" in rng:
        torch.cuda.set_rng_state_all(rng["cuda"])
    return payload["update"]
