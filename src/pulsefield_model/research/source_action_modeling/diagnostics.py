"""Actual update displacement and a fixed block-likelihood linearization."""
from dataclasses import dataclass, fields
from hashlib import sha256

import torch

from ..scoped_style_modeling.dataset import ContractError
from .tensors import BlockBatch


def batch_identity(batch: BlockBatch) -> str:
    identity = sha256()
    tensors = [*(getattr(batch.observation, f.name) for f in fields(batch.observation)),
               *(getattr(batch.queries, f.name) for f in fields(batch.queries)), batch.targets]
    for tensor in tensors:
        identity.update(str((tensor.shape, tensor.dtype)).encode())
        identity.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return identity.hexdigest()


@dataclass(frozen=True)
class ResponseBefore:
    batch_identity: str
    value: float
    parameters: dict
    gradients: dict
    buffers: dict


def capture_response(model, fixed_batch: BlockBatch) -> ResponseBefore:
    """Capture eval-mode mean block log likelihood without changing .grad or mode."""
    training = model.training
    model.eval()
    try:
        named = dict(model.named_parameters())
        if any(name.split(".", 1)[0] not in ("encoder", "decoder") for name in named):
            raise ContractError("Diagnostic parameters require disjoint encoder/decoder ownership")
        response = -model(fixed_batch).loss
        gradients = torch.autograd.grad(response, tuple(named.values()), allow_unused=True)
        return ResponseBefore(batch_identity(fixed_batch), float(response.detach()),
                              {n: p.detach().cpu().clone() for n, p in named.items()},
                              {n: torch.zeros_like(p, device="cpu") if g is None else g.detach().cpu().clone()
                               for (n, p), g in zip(named.items(), gradients)},
                              {n: b.detach().cpu().clone() for n, b in model.named_buffers()})
    finally:
        model.train(training)


def finish_response(before: ResponseBefore, model, fixed_batch: BlockBatch) -> dict:
    """Compare the same scalar response at the actual before/after parameters.

    Contributions count each parameter once by module ownership. They are local
    linearization terms, not causal attribution to examples, actions or slots.
    """
    if before.batch_identity != batch_identity(fixed_batch):
        raise ContractError("Update diagnostics require identical fixed inputs and targets")
    buffers = dict(model.named_buffers())
    if buffers.keys() != before.buffers.keys() or any(
        not torch.equal(before.buffers[n], buffers[n].detach().cpu()) for n in buffers
    ):
        raise ContractError("Diagnostic non-parameter state changed")
    named = dict(model.named_parameters())
    if named.keys() != before.parameters.keys():
        raise ContractError("Diagnostic parameter ownership changed")
    contributions, squared = {"encoder": 0.0, "decoder": 0.0}, {"encoder": 0.0, "decoder": 0.0}
    for name, parameter in named.items():
        group = name.split(".", 1)[0]
        delta = parameter.detach().cpu().double() - before.parameters[name].double()
        contributions[group] += float((before.gradients[name].double() * delta).sum())
        squared[group] += float(delta.square().sum())
    training = model.training
    model.eval()
    try:
        with torch.no_grad():
            after = float(-model(fixed_batch).loss)
    finally:
        model.train(training)
    actual, predicted = after - before.value, sum(contributions.values())
    return {"before_log_likelihood": before.value, "after_log_likelihood": after, "actual_change": actual,
            "gradient_dot_displacement": contributions, "linearized_change": predicted,
            "linearization_residual": actual - predicted,
            "parameter_displacement_l2": {g: v ** 0.5 for g, v in squared.items()},
            "fixed_batch_sha256": before.batch_identity}
