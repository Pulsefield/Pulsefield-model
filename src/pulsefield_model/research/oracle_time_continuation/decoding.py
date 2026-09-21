"""Joint-row sampling on legal support, with tie-preserving nucleus truncation."""
from dataclasses import dataclass
import math

import torch

from ..scoped_style_modeling.dataset import ContractError
from .model import JointRowDistribution
from .schema import CompleteRow


@dataclass(frozen=True)
class DecodeSamplingPolicy:
    temperature: float = 1.
    top_p: float = 1.
    beta: float = 0.
    mode: str = 'sample'
    version: str = 'legal-joint-tied-nucleus-v1'

    def __post_init__(self):
        if (isinstance(self.temperature, bool) or not math.isfinite(self.temperature) or self.temperature <= 0 or
                isinstance(self.top_p, bool) or not math.isfinite(self.top_p) or not 0 < self.top_p <= 1):
            raise ContractError('Decode temperature must be positive and top_p must be in (0,1]')
        if self.beta != 0:
            raise ContractError('No history prior is fitted; decode beta must be zero')
        if self.mode not in ('sample', 'greedy') or self.version != 'legal-joint-tied-nucleus-v1':
            raise ContractError('Unsupported decode mode or policy version')


def policy_probabilities(distribution: JointRowDistribution, policy: DecodeSamplingPolicy):
    # MPS rejects a combined device/dtype conversion to float64. Move first;
    # the higher-precision policy arithmetic belongs exclusively to the CPU.
    raw = distribution.log_probs.detach().cpu().to(dtype=torch.float64)
    legal = distribution.legal.detach().cpu()
    if raw.shape != (256,) or legal.shape != (256,) or not bool(legal.any()):
        raise ContractError('Decode requires one nonempty joint-row support')
    if bool(legal[0]) or not bool(torch.isfinite(raw[legal]).all()):
        raise ContractError('Legal decode scores must be finite and exclude all-empty')
    logits = (raw / policy.temperature).masked_fill(~legal, -torch.inf)
    if not bool(torch.isfinite(logits[legal]).all()):
        raise ContractError('Decode temperature produced nonfinite legal scores')
    probabilities = logits.softmax(-1)
    if policy.mode == 'greedy':
        keep = legal & (logits == logits.max())
    elif policy.top_p < 1:
        values = probabilities[legal].sort(descending=True).values
        cutoff_index = min(int(torch.searchsorted(values.cumsum(0), policy.top_p)), len(values) - 1)
        keep = legal & (probabilities >= values[cutoff_index])
    else:
        keep = legal
    probabilities = probabilities * keep
    probabilities /= probabilities.sum()
    return raw, probabilities, keep


def sample_row(distribution: JointRowDistribution, time_ms: float, policy: DecodeSamplingPolicy,
               rng: torch.Generator):
    raw, probabilities, keep = policy_probabilities(distribution, policy)
    # CPU categorical sampling makes the RNG independent of accelerator streams.
    index = int(torch.multinomial(probabilities, 1, generator=rng))
    actions = tuple((index // factor) % 4 for factor in (64, 16, 4, 1))
    rows = torch.arange(256)[:, None] // torch.tensor((64, 16, 4, 1))[None] % 4
    closes = rows == 3
    return CompleteRow(time_ms, actions), dict(
        raw_model_log_probability=float(raw[index]), decode_log_probability=float(probabilities[index].log()),
        legal_candidates=int(distribution.legal.sum()), retained_candidates=int(keep.sum()),
        raw_close_probability=(raw.exp()[:, None] * closes).sum(0).tolist(),
        policy_close_probability=(probabilities[:, None] * closes).sum(0).tolist())
