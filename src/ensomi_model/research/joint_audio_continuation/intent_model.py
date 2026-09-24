"""One categorical arrangement choice shared by every event in a song.

The decoder receives only a selected code through its full-song context port.
Reference summaries belong to the training recognition network, never inference.
The four states have no predefined style or difficulty names.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..scoped_style_modeling.dataset import ContractError
from .context_model import ContextAudioModel, ContextModelConfig


@dataclass(frozen=True)
class IntentModelConfig(ContextModelConfig):
    intent_states: int = 4
    intent_hidden: int = 32

    def __post_init__(self):
        super().__post_init__()
        if (not self.global_audio or type(self.intent_states) is not int or self.intent_states != 4 or
                type(self.intent_hidden) is not int or self.intent_hidden <= 0):
            raise ContractError('Persistent intent requires full audio, four states and positive hidden width')


def arrangement_summary(chart):
    """Eight deterministic whole-target descriptors for recognition only.

    They are log head rate, LN/head fraction, four attack-row chord fractions,
    occupied lane-time fraction and log median LN duration in seconds. No chart
    labels or beat-grid annotations are needed. Empty LN sets contribute zero.
    """
    rows = chart.source.rows
    counts = np.isin(rows['actions'], (1, 2)).sum(-1)
    heads, head_rows = int(counts.sum()), int((counts > 0).sum())
    lengths = []
    for lane in range(4):
        starts = rows['time'][rows['actions'][:, lane] == 2]
        ends = rows['time'][rows['actions'][:, lane] == 3]
        if len(starts) != len(ends) or np.any(ends <= starts):
            raise ContractError('Recognition summary requires complete valid LN pairs')
        lengths.extend((ends - starts).tolist())
    duration = chart.duration_ms + 1
    if not heads or not head_rows or duration <= 0 or rows['time'][-1] > chart.duration_ms:
        raise ContractError('Recognition summary requires a valid complete chart/audio clock')
    return np.asarray([np.log1p(heads * 1000 / duration), len(lengths) / heads,
        *(float((counts == size).sum()) / head_rows for size in (1, 2, 3, 4)),
        sum(lengths) / (4 * duration), np.log1p(np.median(lengths) / 1000) if lengths else 0], np.float32)


class IntentAudioModel(ContextAudioModel):
    def __init__(self, config: IntentModelConfig):
        super().__init__(config)
        self.intent = nn.Embedding(config.intent_states, config.context_width)
        self.intent_prior = nn.Sequential(nn.Linear(config.context_width, config.intent_hidden), nn.GELU(),
                                          nn.Linear(config.intent_hidden, config.intent_states))
        # Restrict recognition to global target descriptors. Its output remains
        # a distribution over four codes, not a continuous target side channel.
        self.intent_posterior = nn.Sequential(nn.Linear(8, config.intent_hidden), nn.GELU(),
                                              nn.Linear(config.intent_hidden, config.intent_states))
        nn.init.zeros_(self.intent.weight)
        nn.init.zeros_(self.intent_prior[-1].weight)
        nn.init.zeros_(self.intent_prior[-1].bias)

    def prior_log_probs(self, coarse):
        """Audio-only prior from real complete-song context tokens."""
        values, counts = coarse
        valid = torch.arange(values.shape[1], device=values.device)[None] < counts[:, None]
        pooled = (values * valid[..., None]).sum(1) / counts[:, None]
        return self.intent_prior(pooled).log_softmax(-1)

    def posterior_log_probs(self, summary):
        if summary.ndim != 2 or summary.shape[-1] != 8 or not bool(torch.isfinite(summary).all()):
            raise ContractError('Recognition requires finite [batch,8] whole-chart descriptors')
        return self.intent_posterior(summary).log_softmax(-1)

    def apply_code(self, encoded, code):
        """Add one centered categorical offset to the shared audio-context port.

        The constant offset reaches the complete-row, timing-base and bounded
        history-residual paths. No mask, physical action or event clock changes.
        """
        if type(code) is not int or not 0 <= code < self.config.intent_states:
            raise ContractError('Intent code must be one integer in [0,4)')
        if encoded.shape[-1] != self.config.conditioned_audio_width:
            raise ContractError('Intent conditioning requires the full local/global audio width')
        offset = self.intent.weight[code] - self.intent.weight.mean(0)
        return torch.cat((encoded[..., :self.config.audio_width],
                          encoded[..., self.config.audio_width:] + offset), -1)

    def encode_audio(self, mel, valid=None):
        raise ContractError('Intent generation must choose a persistent code with encode_generation')

    def encode_generation(self, mel, *, seed, code=None):
        """Encode once and choose one audio-prior code, independent of event RNG."""
        if len(mel) != 1:
            raise ContractError('Native intent generation requires one complete song')
        valid = torch.ones(mel.shape[:2], dtype=torch.bool, device=mel.device)
        coarse = self.encode_coarse(mel, valid)
        prior = self.prior_log_probs(coarse)[0].exp()
        sampled = code is None
        if sampled:
            rng = torch.Generator(device='cpu').manual_seed(seed)
            code = int(torch.multinomial(prior.detach().cpu(), 1, generator=rng))
        encoded = self.encode_crop(mel, valid, torch.zeros(1, dtype=torch.long, device=mel.device),
                                   valid.sum(-1), coarse)
        return self.apply_code(encoded, code), dict(intent_code=code, intent_seed=seed,
            intent_selection='audio_prior' if sampled else 'explicit_code',
            intent_prior=prior.detach().cpu().tolist())
