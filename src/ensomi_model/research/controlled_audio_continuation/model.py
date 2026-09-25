"""Timing-only skeleton factors and a scoped, complete-row R1 policy.

Counts are an internal factor of R1, never part of its skeleton input. The
frontier score is normalized across complete rows after count/layout composition.
"""
from dataclasses import asdict, replace
import math

import torch
from torch import nn

from ..planned_audio_continuation.counts import COUNT_MARKS, ROW_COUNTS
from ..planned_audio_continuation.model import PlannedAudioModel, PlannedModelConfig
from ..typed_audio_continuation.controls import ControlSchedule, PER_FIELD_SCOPE
from ..typed_audio_continuation.program import Recovery


class RowComposition(nn.Module):
    """R1's count-family distribution, conditioned on its own full context."""
    def __init__(self, audio_width, context_width, preview_width, control_width, hidden=128):
        super().__init__()
        self.readout = nn.Sequential(nn.Linear(audio_width+context_width+preview_width+control_width, hidden),
                                     nn.GELU(), nn.Linear(hidden, len(COUNT_MARKS)))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        index = torch.tensor([COUNT_MARKS.index(mark) for mark in ROW_COUNTS])
        self.register_buffer('row_mark', index, persistent=False)
        self.register_buffer('members', torch.arange(len(COUNT_MARKS))[:, None] == index, persistent=False)
        groups = tuple(sorted({(h, r) for h, l, r in COUNT_MARKS}))
        group = torch.tensor([groups.index((h, r)) for h, l, r in COUNT_MARKS])
        self.register_buffer('group', group, persistent=False)
        self.register_buffer('group_members', torch.arange(len(groups))[:, None] == group, persistent=False)
        self.register_buffer('longs', torch.tensor([l for h, l, r in COUNT_MARKS]), persistent=False)

    def logits(self, audio, context, preview, control):
        return self.readout(torch.cat((audio, context.mean(-2), preview, control), -1))

    def count_log_probs(self, raw, actual, active, shift):
        """Keep actual-control head/release mass; tilt local LN preferences."""
        members = active[:, None] & self.group_members[None]
        exists = members.any(-1)
        def group_sum(values):
            masked = values[:, None].masked_fill(~members, -torch.inf)
            return torch.where(exists[..., None], masked, 0.).logsumexp(-1)
        mass = group_sum(actual.masked_fill(~active, -torch.inf).log_softmax(-1))
        tilted = raw+shift[:, None]*self.longs
        return (mass[:, self.group]+tilted-group_sum(tilted)[:, self.group]).masked_fill(~active, -torch.inf)

    def compose(self, layout, count_log_probs, legal):
        members = legal[:, None] & self.members[None]
        active = members.any(-1)
        scores = layout[:, None].masked_fill(~members, -torch.inf)
        normalizers = torch.where(active[..., None], scores, 0.).logsumexp(-1)
        return (layout-normalizers[:, self.row_mark]+count_log_probs[:, self.row_mark]).masked_fill(~legal, -torch.inf)


class ControlledAudioModel(PlannedAudioModel):
    control_encoding = PER_FIELD_SCOPE

    def __init__(self, config, *, style_names=(), ln_reference=.17, recovery=Recovery(60, 50, 50)):
        super().__init__(config)
        self.style_names = tuple(style_names)
        self.ln_reference, self.recovery = ln_reference, recovery
        width = ControlSchedule(style_names=self.style_names).width_for(self.control_encoding)
        self.head_control = nn.Linear(width, config.conditioned_audio_width, bias=False)
        self.release_control = nn.Linear(width, config.conditioned_audio_width, bias=False)
        self.row_control = nn.Linear(width, config.hidden, bias=False)
        for module in (self.head_control, self.release_control, self.row_control):
            nn.init.zeros_(module.weight)
        self.composition = RowComposition(config.conditioned_audio_width, config.hidden,
            self.preview_condition.in_features, width)

    def probability_options(self):
        return dict(style_names=list(self.style_names), ln_reference=self.ln_reference,
                    recovery=asdict(self.recovery))

    def head_logits(self, audio, history, clocks, *, control):
        return super().head_logits(audio+self.head_control(control), history, clocks)

    def release_logits(self, audio, history, clocks, *, control):
        return super().release_logits(audio+self.release_control(control), history, clocks)

    def planned_row_log_probs(self, audio, history, exact, legal, occupancy, preview, local, timing,
                              *, control, response_allowed=None, ln_shift=0.):
        allowed = legal if response_allowed is None else legal & response_allowed
        base = self.condition(history, exact, audio)+self.preview_condition(preview).unsqueeze(-2)
        hands = base+self.row_control(control).unsqueeze(-2)
        layout = self.joint(hands)
        layout = layout+torch.where(self.has_head[None], self.route_residual(hands,
            torch.ones(len(audio), dtype=torch.bool, device=audio.device)), 0.)
        layout = layout+self.release_residual(hands, occupancy.any(-1))
        actual = self.composition.logits(audio, base, preview, control)
        known = control[:, 3+len(self.style_names)] > 0
        reference = control.clone()
        reference[:, 1] = torch.where(known, 2*self.ln_reference-1, control[:, 1])
        raw = self.composition.logits(audio, base, preview, reference)
        rho = ((control[:, 1]+1)/2).clamp(.0001, .9999)
        shift = torch.where(known, torch.logit(rho)-math.log(self.ln_reference/(1-self.ln_reference)), 0.)+ln_shift
        active = (allowed[:, None] & self.composition.members[None]).any(-1)
        counts = self.composition.count_log_probs(raw, actual, active, shift)
        scores = self.composition.compose(layout, counts, allowed)
        # This comparison must survive count-group normalization. In particular,
        # a costly four-key row cannot escape its frontier cost as a singleton.
        scores = scores+self.row_consequence.score(hands, local, timing)
        return scores.masked_fill(~allowed, -torch.inf).log_softmax(-1)


def warm_model(path, *, device='cpu', recovery=Recovery(60, 50, 50)):
    """Retain compatible audio/R1 tensors; initialize the new timing/count law.

    The old typed count predictor and typed preview are deliberately not copied:
    they condition on decisions that no longer belong to the skeleton.
    """
    saved = torch.load(path, map_location='cpu', weights_only=False)
    config = replace(PlannedModelConfig(**saved['model_config']), bounded_head=True,
        condition_full_holds=True, minimum_action_gap_ms=recovery.hh, profile_count=0, row_factorization='flat')
    options = saved['probability_options']
    model = ControlledAudioModel(config, style_names=options['style_names'],
                                 ln_reference=options['ln_prior'], recovery=recovery)
    unused_profile = {'profile_values', 'profile_codes', 'profile_mean', 'profile_std'}
    retained = {k.removeprefix('body.'): v for k, v in saved['model'].items()
                if k.startswith('body.') and k.removeprefix('body.') not in unused_profile}
    receipt = model.load_state_dict(retained, strict=False)
    if receipt.unexpected_keys:
        raise ValueError(f'Unconsumed row/audio initialization: {receipt.unexpected_keys}')
    with torch.no_grad():
        model.head_base.weight.copy_(saved['model']['clock_audio.weight'][::2])
        model.head_base.bias.copy_(saved['model']['clock_audio.bias'][::2])
        n = 2*(2+len(model.style_names))
        model.row_control.weight[:, :n].copy_(saved['model']['row_control.weight'][:, :n])
    copied = sorted(retained)
    report = dict(copied=copied, initialized=sorted(receipt.missing_keys),
        omitted_unused_profile_buffers=sorted(k for k in unused_profile if 'body.'+k in saved['model']),
        projection='Direct head-audio affine and control value/known-bit columns copied; scope clocks and typed plan are not copied.',
        parameters=sum(p.numel() for p in model.parameters()))
    return model.to(device), report


def load_model(path, *, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved.get('format') != 'controlled-audio/v1':
        raise ValueError('Expected a controlled-audio/v1 checkpoint')
    options = dict(saved['probability_options'])
    options['recovery'] = Recovery(**options['recovery'])
    model = ControlledAudioModel(PlannedModelConfig(**saved['model_config']), **options)
    model.load_state_dict(saved['model'])
    return model.to(device).eval()
