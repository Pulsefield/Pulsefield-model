"""First-release hazards conditioned on an event within a finite legal wait."""
import torch

from ..scoped_style_modeling.dataset import ContractError


def conditioned_release_logits(raw):
    """Normalize one complete, unpadded wait; the caller forces its last clock.

    For raw hazards r, the conditional hazard odds at u are exp(raw[u]) / Z,
    where Z is the raw probability of an event strictly after u by the deadline.
    Suffix probabilities use log cumulative hazard to retain rare-event shapes
    without subtracting nearly equal survival probabilities. Below -40 the
    log-softplus and log-event-probability asymptotes agree to floating precision.

    Inputs must be finite float32/64 logits in chronological order, covering
    every feasible native clock. The final returned logit is a finite zero
    placeholder: it is excluded from binary loss and marked forced in sampling.
    Its raw logit still affects every preceding conditional hazard and gradient.
    """
    if (raw.ndim != 1 or not len(raw) or raw.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(raw).all())):
        raise ContractError('Conditional release needs a nonempty finite float32/64 wait')
    # logaddexp retains tiny positive rates on MPS, where softplus can round
    # them to zero before the outer log and produce nonfinite gradients.
    log_rate = torch.where(raw < -40, raw,
                           torch.logaddexp(raw.clamp_min(-40), torch.zeros_like(raw)).log())
    log_mass = torch.logcumsumexp(log_rate.flip(0), 0).flip(0)
    # Clamp only the unused/asymptotic arithmetic to keep both branches and
    # their gradients finite, including logits far below exp's underflow range.
    log_event = torch.where(log_mass < -40, log_mass,
                           (-torch.expm1(-log_mass.clamp(-40, 20).exp())).log())
    return torch.cat((raw[:-1] - log_event[1:], raw[-1:] * 0))
