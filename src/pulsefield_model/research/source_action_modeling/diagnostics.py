"""Actual update displacement and a fixed block-likelihood linearization."""
from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256

import torch

from ..scoped_style_modeling.dataset import ContractError
from .tensors import BlockBatch


def batch_identity(batch) -> str:
    """Hash nested tensor dataclasses, including input/query/response metadata."""
    identity = sha256()
    def visit(value):
        if is_dataclass(value):
            for field in fields(value):
                identity.update(field.name.encode())
                visit(getattr(value, field.name))
        else:
            identity.update(str((value.shape, value.dtype)).encode())
            identity.update(value.detach().cpu().contiguous().numpy().tobytes())
    visit(batch)
    return identity.hexdigest()


@dataclass(frozen=True)
class ResponseBefore:
    batch_identity: str
    value: float
    parameters: dict
    gradients: dict
    buffers: dict
    policy: str


def capture_response(model, fixed_batch: BlockBatch) -> ResponseBefore:
    """Capture negative equal-block mean row NLL without changing .grad or mode."""
    training = model.training
    model.eval()
    try:
        named = dict(model.named_parameters())
        if any(name.split(".", 1)[0] not in ("encoder", "reader", "decoder") for name in named):
            raise ContractError("Diagnostic parameters require disjoint encoder/reader/decoder ownership")
        response = -model(fixed_batch).loss
        gradients = torch.autograd.grad(response, tuple(named.values()), allow_unused=True)
        return ResponseBefore(batch_identity(fixed_batch), float(response.detach()),
                              {n: p.detach().cpu().clone() for n, p in named.items()},
                              {n: torch.zeros_like(p, device="cpu") if g is None else g.detach().cpu().clone()
                               for (n, p), g in zip(named.items(), gradients)},
                              {n: b.detach().cpu().clone() for n, b in model.named_buffers()},
                              str(getattr(model, "policy_identity", "stage1-position-gather-v1")))
    finally:
        model.train(training)


def finish_response(before: ResponseBefore, model, fixed_batch: BlockBatch) -> dict:
    """Compare the same scalar response at the actual before/after parameters.

    Contributions count each parameter once by module ownership. They are local
    linearization terms, not causal attribution to examples, actions or slots.
    """
    if before.batch_identity != batch_identity(fixed_batch):
        raise ContractError("Update diagnostics require identical fixed inputs and targets")
    if before.policy != str(getattr(model, "policy_identity", "stage1-position-gather-v1")):
        raise ContractError("Diagnostic model or reader policy changed")
    buffers = dict(model.named_buffers())
    if buffers.keys() != before.buffers.keys() or any(
        not torch.equal(before.buffers[n], buffers[n].detach().cpu()) for n in buffers
    ):
        raise ContractError("Diagnostic non-parameter state changed")
    named = dict(model.named_parameters())
    if named.keys() != before.parameters.keys():
        raise ContractError("Diagnostic parameter ownership changed")
    groups = sorted({n.split(".", 1)[0] for n in named})
    contributions, squared = dict.fromkeys(groups, 0.0), dict.fromkeys(groups, 0.0)
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


@dataclass(frozen=True)
class SemanticResponseBefore:
    response: ResponseBefore
    readout_state: dict
    readout_policy: dict


class _SemanticResponse(torch.nn.Module):
    """Reuse the update linearization with only the encoder as a parameter owner."""
    def __init__(self, model, readout):
        super().__init__()
        self.encoder = model.encoder
        # The fitted readout is a fixed response function, not an update owner.
        object.__setattr__(self, "readout", readout)
        self.policy_identity = {"model": model.policy_identity, "response": "positive-minus-negative-presence-margin-v1",
                                "readout": readout.policy_identity}

    def forward(self, batch):
        from types import SimpleNamespace
        logits = self.readout(self.encoder(batch.observation), batch.concepts, batch.input_indices)
        margin = logits[:, 1:].logsumexp(-1) - logits[:, 0]
        positive = batch.targets > 0
        if not positive.any() or positive.all():
            raise ContractError("Semantic response requires fixed positive and negative human targets")
        response = margin[positive].mean() - margin[~positive].mean()
        return SimpleNamespace(loss=-response)


def capture_semantic_response(model, readout, fixed_batch) -> SemanticResponseBefore:
    """Capture a fitted presence-margin response for a short encoder-update window.

    Use a SemanticCorpus batch of original human scopes. The readout remains
    fixed; its own parameters are neither attributed nor updated here.
    """
    if not readout.fitted or model.access != readout.access:
        raise ContractError("Semantic diagnostics require a fitted readout with matching access")
    encoder_mode = model.encoder.training
    response = _SemanticResponse(model, readout)
    try:
        before = capture_response(response, fixed_batch)
    finally:
        model.encoder.train(encoder_mode)
    return SemanticResponseBefore(before, {n: v.detach().cpu().clone() for n, v in readout.state_dict().items()},
                                  readout.policy_identity)


def finish_semantic_response(before: SemanticResponseBefore, model, readout, fixed_batch) -> dict:
    if not readout.fitted or readout.policy_identity != before.readout_policy or any(
        n not in before.readout_state or not torch.equal(v.detach().cpu(), before.readout_state[n])
        for n, v in readout.state_dict().items()
    ) or readout.state_dict().keys() != before.readout_state.keys():
        raise ContractError("Fitted semantic readout changed during the encoder-update window")
    encoder_mode = model.encoder.training
    try:
        result = finish_response(before.response, _SemanticResponse(model, readout), fixed_batch)
    finally:
        model.encoder.train(encoder_mode)
    result["before_presence_margin"] = result.pop("before_log_likelihood")
    result["after_presence_margin"] = result.pop("after_log_likelihood")
    result["response"] = "positive-minus-negative-presence-margin"
    return result
