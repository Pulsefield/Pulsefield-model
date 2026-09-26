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
from ..joint_audio_continuation.batching import interpolate_audio
from ..scoped_style_modeling.dataset import ContractError
from ..typed_audio_continuation.controls import ControlSchedule, PER_FIELD_SCOPE
from ..typed_audio_continuation.program import Recovery
from .composition_prior import CompositionPrior
from .hold_audio import HoldAudioCues


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

    def prior_log_probs(self, prior, allowed):
        exists = (allowed[:, None] & self.group_members[:, self.row_mark][None]).any(-1)
        return prior.masked_fill(~exists, -torch.inf).log_softmax(-1)

    def count_log_probs(self, raw, actual, active, shift, *, prior=None, history_bound=None):
        """Keep actual-control head/release mass; tilt local LN preferences."""
        members = active[:, None] & self.group_members[None]
        exists = members.any(-1)
        def group_sum(values):
            masked = values[:, None].masked_fill(~members, -torch.inf)
            return torch.where(exists[..., None], masked, 0.).logsumexp(-1)
        if prior is None:
            mass = group_sum(actual.masked_fill(~active, -torch.inf).log_softmax(-1))
        else:
            full = group_sum(actual)
            center = full.masked_fill(~exists, 0.).sum(-1)/exists.sum(-1)
            residual = history_bound*torch.tanh((full-center[:, None])/history_bound)
            mass = (prior+residual).masked_fill(~exists, -torch.inf).log_softmax(-1)
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

    def __init__(self, config, *, style_names=(), ln_reference=.17, recovery=Recovery(60, 50, 50),
                 count_history_bound=None, hold_audio_width=0):
        super().__init__(config)
        self.style_names = tuple(style_names)
        self.ln_reference, self.recovery = ln_reference, recovery
        if count_history_bound is not None and (not math.isfinite(count_history_bound) or count_history_bound <= 0):
            raise ValueError('Composition history bound must be finite and positive')
        self.count_history_bound = count_history_bound
        width = ControlSchedule(style_names=self.style_names).width_for(self.control_encoding)
        self.head_control = nn.Linear(width, config.conditioned_audio_width, bias=False)
        self.release_control = nn.Linear(width, config.conditioned_audio_width, bias=False)
        self.row_control = nn.Linear(width, config.hidden, bias=False)
        for module in (self.head_control, self.release_control, self.row_control):
            nn.init.zeros_(module.weight)
        self.composition = RowComposition(config.conditioned_audio_width, config.hidden,
            self.preview_condition.in_features, width)
        self.count_prior = (None if count_history_bound is None else CompositionPrior(
            config.conditioned_audio_width, self.preview_condition.in_features, width,
            len(self.composition.group_members)))
        if type(hold_audio_width) is not int or hold_audio_width < 0:
            raise ValueError('Active-LN audio width must be a nonnegative integer')
        self.hold_audio_width = hold_audio_width
        self.hold_cues = (HoldAudioCues(config.conditioned_audio_width, hold_audio_width, config.hidden)
                         if hold_audio_width else None)

    @property
    def requires_full_audio_queries(self):
        return self.hold_cues is not None

    def hold_audio_options(self, encoded, starts, times, *, audio_starts=None, frame_counts=None):
        if self.hold_cues is None:
            return {}
        if (starts is None or starts.shape != (len(times), 4) or starts.dtype != torch.long or
                len(encoded) != 1 or bool(((starts >= 0) & (starts > times[:, None])).any())):
            raise ContractError('LN audio retrieval requires committed native starts for one complete song')
        active = starts >= 0
        origins = interpolate_audio(encoded, starts.clamp_min(0).reshape(1, -1),
                                    audio_starts, frame_counts).reshape(len(times), 4, encoded.shape[-1])
        age = torch.asinh((times[:, None]-starts).to(encoded.dtype)/1000)*active
        holds = torch.cat((origins*active[..., None], age[..., None],
                           active.to(encoded.dtype)[..., None]), -1)
        return dict(hold_audio=holds)

    def probability_options(self):
        return dict(style_names=list(self.style_names), ln_reference=self.ln_reference,
                    recovery=asdict(self.recovery), count_history_bound=self.count_history_bound,
                    hold_audio_width=self.hold_audio_width)

    def head_logits(self, audio, history, clocks, *, control):
        return super().head_logits(audio+self.head_control(control), history, clocks)

    def release_logits(self, audio, history, clocks, *, control, hold_audio=None):
        context = None if self.hold_cues is None else self.hold_cues.release_values(audio, hold_audio)
        return super().release_logits(audio+self.release_control(control), history, clocks,
                                       context_addition=context)

    def planned_row_log_probs(self, audio, history, exact, legal, occupancy, preview, local, timing,
                              *, control, response_allowed=None, ln_shift=0., hold_audio=None):
        allowed = legal if response_allowed is None else legal & response_allowed
        base = self.condition(history, exact, audio)+self.preview_condition(preview).unsqueeze(-2)
        if self.hold_cues is not None:
            base = base+self.hold_cues.row_values(audio, hold_audio)
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
        prior = None if self.count_prior is None else self.count_prior(audio, exact, preview, control)
        counts = self.composition.count_log_probs(raw, actual, active, shift,
                                                 prior=prior, history_bound=self.count_history_bound)
        scores = self.composition.compose(layout, counts, allowed)
        # This comparison must survive count-group normalization. In particular,
        # a costly four-key row cannot escape its frontier cost as a singleton.
        consequence_context = hands
        if self.count_prior is not None:
            # Chart history and exact replay still inform the response preference;
            # music conditions arrangement through the prior and layout paths.
            consequence_context = (self.condition(history, exact, torch.zeros_like(audio))
                + self.preview_condition(preview).unsqueeze(-2) + self.row_control(control).unsqueeze(-2))
        scores = scores+self.row_consequence.score(consequence_context, local, timing)
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
