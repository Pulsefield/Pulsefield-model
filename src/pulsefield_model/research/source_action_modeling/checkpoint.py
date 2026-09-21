"""Resumable snapshots for trusted local source-action runs, with a versioned contract."""
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
import random

import numpy as np
import torch

from ..scoped_style_modeling.dataset import ContractError, digest
from .model import DECODER_POLICY
from .observation import INPUT_CONTRACT

SNAPSHOT_SCHEMA = 4
SIX_ACTION_INPUT_CONTRACT = "source-action-visibility-v2"
SIX_ACTION_DECODER_POLICY = "joint-row/context-bilinear-hand-transpose-v1"


def warm_start_snapshot(path: Path, model) -> dict:
    """Load weights from a compatible trusted-local four- or six-action snapshot.

    Six-action hand embeddings are selected by action meaning; all other learned
    tensors are copied exactly and action lookup buffers are rebuilt. Architecture,
    access policy and model dimensions must match. Contract and tensor checks run
    before modifying the model. Optimizer, scheduler, sampler, RNG, gradients,
    training mode and fitted concept reader are not restored. This starts a new
    training/evaluation run; it is not exact continuation of the saved update.
    """
    path = Path(path)
    raw = path.read_bytes()
    payload = torch.load(BytesIO(raw), map_location="cpu", weights_only=False)
    old = (payload.get("schema"), payload.get("input_contract")) == (3, SIX_ACTION_INPUT_CONTRACT)
    current = (payload.get("schema"), payload.get("input_contract")) == (SNAPSHOT_SCHEMA, INPUT_CONTRACT)
    policy = dict(model.policy_identity)
    if old:
        policy["decoder"] = SIX_ACTION_DECODER_POLICY
    if not (old or current) or policy.get("decoder") not in (DECODER_POLICY, SIX_ACTION_DECODER_POLICY) or (
        payload.get("model_config") != asdict(model.config) or payload.get("model_policy") != policy or
        payload.get("encoder_type") != f"{type(model.encoder).__module__}.{type(model.encoder).__qualname__}"
    ):
        raise ContractError("Warm-start snapshot architecture, access or action contract mismatch")
    saved, expected = payload["model"], model.state_dict()
    if set(saved) != set(expected):
        raise ContractError("Warm-start snapshot tensor keys differ from the model")
    # Old lane enumeration was (EMPTY, TAP, START, CLOSE, CLOSE+TAP, CLOSE+START).
    # CLOSE had bit value 4 but ordinal 3. Select pairs, not the first 16 rows.
    indices = torch.tensor([6 * outer + inner for outer in range(4) for inner in range(4)])
    if old:
        alphabet = (0, 1, 2, 4, 5, 6)
        buffers = {
            "decoder.actions": torch.tensor([[[a, b], [c, d]] for a in alphabet for b in alphabet
                                              for c in alphabet for d in alphabet]),
            "decoder.hand_tokens": torch.tensor([[i // 36, i % 36] for i in range(1296)]),
        }
    else:
        buffers = {k: expected[k].cpu() for k in ("decoder.actions", "decoder.hand_tokens")}
    candidate = {}
    for name, target in expected.items():
        value = saved[name]
        if not isinstance(value, torch.Tensor) or value.dtype != target.dtype:
            raise ContractError(f"Warm-start tensor dtype mismatch: {name}")
        if name in buffers:
            if not torch.equal(value, buffers[name]):
                raise ContractError(f"Warm-start action lookup mismatch: {name}")
            value = target
        elif old and name == "decoder.action.weight":
            if value.shape != (36, model.config.action_dim):
                raise ContractError("Warm-start six-action embedding shape mismatch")
            value = value.index_select(0, indices)
        if value.shape != target.shape:
            raise ContractError(f"Warm-start tensor shape mismatch: {name}")
        candidate[name] = value
    model.load_state_dict(candidate)
    return {"path": str(path.resolve()), "sha256": digest(raw), "source_schema": payload["schema"],
            "source_update": payload["update"], "source_model_policy": payload["model_policy"],
            "policy": "six-to-four-action-weights-v1" if old else "four-action-weights-v1",
            "hand_embedding_indices": indices.tolist() if old else list(range(16)),
            "training_state_restored": False}


def save_snapshot(path: Path, model, optimizer, sampler, *, update: int, scheduler=None, readout=None):
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
    payload = {"schema": SNAPSHOT_SCHEMA, "input_contract": INPUT_CONTRACT, "model_config": asdict(model.config),
               "model_policy": getattr(model, "policy_identity", "stage1-position-gather-v1"),
               "encoder_type": f"{type(model.encoder).__module__}.{type(model.encoder).__qualname__}",
               "optimizer_type": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
               "scheduler_type": None if scheduler is None else f"{type(scheduler).__module__}.{type(scheduler).__qualname__}",
               "device_type": device, "model": model.state_dict(), "training": model.training,
               "optimizer": optimizer.state_dict(), "sampler": sampler.state_dict(), "rng": rng,
               "scheduler": None if scheduler is None else scheduler.state_dict(), "update": update,
               "readout": None if readout is None else {"policy": readout.policy_identity,
                   "state": readout.state_dict(), "fitted": readout.fitted, "training": readout.training}}
    with Path(path).open("xb") as stream:
        torch.save(payload, stream)


def load_snapshot(path: Path, model, optimizer, sampler, *, scheduler=None, readout=None) -> int:
    """Restore on the same device family, including next-sample and next-update state."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected_encoder = f"{type(model.encoder).__module__}.{type(model.encoder).__qualname__}"
    if payload.get("schema") != SNAPSHOT_SCHEMA or payload.get("input_contract") != INPUT_CONTRACT or (
        payload.get("model_config") != asdict(model.config) or payload.get("encoder_type") != expected_encoder or
        payload.get("model_policy") != getattr(model, "policy_identity", "stage1-position-gather-v1")
    ):
        raise ContractError("Snapshot model or partial-observation contract mismatch; use warm_start_snapshot for six-action weights")
    if payload["device_type"] != next(model.parameters()).device.type:
        raise ContractError("Exact RNG restoration requires the same device family")
    if (payload["scheduler"] is None) != (scheduler is None):
        raise ContractError("Snapshot scheduler presence differs from the supplied scheduler")
    if payload.get("optimizer_type") != f"{type(optimizer).__module__}.{type(optimizer).__qualname__}" or (
        payload.get("scheduler_type") != (None if scheduler is None else f"{type(scheduler).__module__}.{type(scheduler).__qualname__}")
    ):
        raise ContractError("Snapshot optimizer or scheduler type mismatch")
    saved_readout = payload.get("readout")
    if (saved_readout is None) != (readout is None) or (
        readout is not None and saved_readout["policy"] != readout.policy_identity
    ):
        raise ContractError("Snapshot semantic readout presence or policy mismatch")
    sampler.load_state_dict(payload["sampler"])
    model.load_state_dict(payload["model"])
    model.train(payload["training"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if readout is not None:
        readout.load_state_dict(saved_readout["state"])
        readout.fitted = saved_readout["fitted"]
        readout.train(saved_readout["training"])
    rng = payload["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.random.set_rng_state(rng["torch"])
    if "mps" in rng:
        torch.mps.set_rng_state(rng["mps"])
    if "cuda" in rng:
        torch.cuda.set_rng_state_all(rng["cuda"])
    return payload["update"]
