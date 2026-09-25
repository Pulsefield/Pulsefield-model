"""One resource score planner and an R1-derived column arranger."""
import math
import copy

import torch
from torch import nn
from torch.nn import functional as F

from ..bounded_typed_continuation.temporal import FiniteTemporal, TemporalConfig, pointwise
from ..planned_audio_continuation.model import PlannedAudioModel, PlannedModelConfig
from .controls import ControlSchedule
from .program import CLOCK_DIM, MARKS, TOKEN_DIM, TIME_DIM


class TypedAudioModel(nn.Module):
    def __init__(self, body_config, *, style_names=(), plan_hidden=64, plan_levels=5,
                 ln_base=False, head_stream=False, bounded_clock=False, ln_prior=None):
        super().__init__()
        self.config = body_config
        self.style_names = tuple(style_names)
        self.ln_base = ln_base
        if ln_prior is not None and not 0 < ln_prior < 1:
            raise ValueError('LN reference fraction must be strictly inside (0,1)')
        self.ln_prior = ln_prior
        self.head_stream = False
        self.bounded_clock = False
        self.body = PlannedAudioModel(body_config)
        self.plan_temporal = FiniteTemporal(TemporalConfig(TOKEN_DIM, plan_hidden, plan_levels, 4))
        width = ControlSchedule(style_names=self.style_names).width
        self.plan_query = nn.Sequential(nn.Linear(plan_hidden+CLOCK_DIM+width+body_config.conditioned_audio_width, 128),
                                        nn.GELU(), nn.Linear(128, 128), nn.GELU())
        self.clock = nn.Linear(128, 20)
        self.clock_audio = nn.Linear(body_config.conditioned_audio_width, 20)
        self.mark = nn.Linear(128, len(MARKS))
        nn.init.normal_(self.clock.weight, std=.001)
        with torch.no_grad():
            self.clock.bias.zero_()
            self.clock_audio.weight.zero_()
            self.clock_audio.bias.copy_(torch.tensor([math.log(.006), math.log(.001)]*10))
        self.plan_preview = nn.Linear(body_config.lookahead*TOKEN_DIM+20, body_config.hidden, bias=False)
        self.row_control = nn.Linear(width, body_config.hidden, bias=False)
        nn.init.zeros_(self.plan_preview.weight)
        nn.init.zeros_(self.row_control.weight)
        # The previous H/R/profile factors are not part of this probability law.
        for name in ('head_temporal', 'skeleton_temporal', 'head_condition', 'release_clock',
                     'head_base', 'timing', 'context_timing', 'profile_condition', 'profile_prior',
                     'preview_condition'):
            if hasattr(self.body, name):
                delattr(self.body, name)
        if head_stream:
            self.enable_head_stream()
        if bounded_clock:
            self.enable_bounded_clock()

    def enable_head_stream(self):
        """Initialize head ownership from the currently loaded shared weights."""
        if self.bounded_clock:
            raise ValueError('Bounded shared and head-owned clocks are separate experiments')
        self.head_temporal = copy.deepcopy(self.plan_temporal)
        self.head_query = copy.deepcopy(self.plan_query)
        self.head_clock = nn.Linear(128, 10).to(self.clock.weight.device)
        with torch.no_grad():
            self.head_clock.weight.copy_(self.clock.weight[::2])
            self.head_clock.bias.copy_(self.clock.bias[::2])
        self.head_stream = True

    def enable_bounded_clock(self):
        """Keep controls and real LN obligations outside fading history."""
        if self.head_stream:
            raise ValueError('Bounded shared and head-owned clocks are separate experiments')
        width = ControlSchedule(style_names=self.style_names).width
        self.clock_base = nn.Sequential(nn.Linear(self.config.conditioned_audio_width+width+4*TIME_DIM+4, 128),
                                        nn.GELU(), nn.Linear(128, 20)).to(self.clock.weight.device)
        nn.init.zeros_(self.clock_base[-1].weight)
        nn.init.zeros_(self.clock_base[-1].bias)
        self.bounded_clock = True

    def clock_parts(self, audio, history, clocks, control, head_clocks):
        """Base, bounded modulation and H/R gates on the bin query clock.

        Only audio, declared controls, active-LN ages and occupation enter the
        base. Exact physical support remains external and never fades.
        """
        holds = torch.cat((clocks[..., :4*TIME_DIM], clocks[..., -4:]), -1)
        base = self.clock_audio(audio) + self.clock_base(torch.cat((audio, control, holds), -1))
        def gate(values):
            age = 1000*values[..., 8*TIME_DIM+1].sinh().clamp_min(0)
            present = values[..., 9*TIME_DIM-1]
            return present*torch.exp(-age/self.config.head_decay_ms)
        gates = torch.stack((gate(head_clocks), gate(clocks)), -1).repeat(1, 10)
        raw = self.clock(self.plan_values(audio, history, clocks, control))
        bound = self.config.head_bound
        return base, bound*gates*torch.tanh(raw/bound), gates

    def plan_values(self, audio, history, clocks, control):
        return pointwise(self.plan_query, torch.cat((audio, history.mean(-2), clocks, control), -1))

    def clock_log_probs(self, audio, history, clocks, control, support, *, head_history=None, head_clocks=None):
        if self.bounded_clock:
            base, modulation, _ = self.clock_parts(audio, history, clocks, control, head_clocks)
            logits = (base+modulation).reshape(-1, 10, 2)
        else:
            logits = (self.clock_audio(audio) + self.clock(self.plan_values(audio, history, clocks, control))).reshape(-1, 10, 2)
        if self.head_stream:
            head = self.head_query(torch.cat((audio, head_history.mean(-2), head_clocks, control), -1))
            h = self.clock_audio(audio)[:, ::2] + self.head_clock(head)
            r = logits[..., 1]
            log_h = F.logsigmoid(h).masked_fill(~support[..., 1], -torch.inf)
            log_no_h = torch.where(support[..., 1], F.logsigmoid(-h), 0.)
            # At true EOS an outstanding LN forces an event, not a new head.
            forced_release = ~support[..., 0]
            log_r = torch.where(forced_release, 0., F.logsigmoid(r)).masked_fill(~support[..., 2], -torch.inf)
            log_no_r = torch.where(support[..., 2], F.logsigmoid(-r), 0.).masked_fill(forced_release, -torch.inf)
            return torch.stack((log_no_h+log_no_r, log_h, log_no_h+log_r), -1)
        logits = torch.cat((torch.zeros_like(logits[..., :1]), logits), -1)
        return logits.masked_fill(~support, -torch.inf).log_softmax(-1)

    def mark_log_probs(self, audio, history, clocks, control, support):
        if not self.ln_base and self.ln_prior is None:
            return self.mark(self.plan_values(audio, history, clocks, control)).masked_fill(~support, -torch.inf).log_softmax(-1)
        from .proportions import condition_ln_count, tilt_ln_count
        known_index = 3 + len(self.style_names)
        residual_control = control.clone()
        if self.ln_prior is None:
            residual_control[..., 1] = 0.
            residual_control[..., known_index] = 0.
        else:
            # Unknown amount and a known reference amount are different
            # conditions. Retain presence, but remove the varying amount from
            # local preferences so the odds shift owns its direct effect.
            residual_control[..., 1] = torch.where(control[..., known_index] > 0,
                                                   2*self.ln_prior-1, 0.)
        raw = self.mark(self.plan_values(audio, history, clocks, residual_control))
        controlled = (condition_ln_count(raw, support, (control[:, 1]+1)/2) if self.ln_prior is None else
                      tilt_ln_count(raw, support, (control[:, 1]+1)/2, self.ln_prior))
        original = raw.masked_fill(~support, -torch.inf).log_softmax(-1)
        return torch.where(control[:, known_index, None] > 0, controlled, original)

    def probability_options(self):
        """Constructor options actually consumed by the typed probability law."""
        return dict(style_names=list(self.style_names), plan_hidden=self.plan_temporal.config.hidden,
                    plan_levels=self.plan_temporal.config.levels, ln_base=self.ln_base,
                    head_stream=self.head_stream, bounded_clock=self.bounded_clock, ln_prior=self.ln_prior)

    def row_log_probs(self, audio, history, exact, support, occupancy, plan, control, local, timing):
        body = self.body
        hands = body.condition(history, exact, audio)
        hands = hands + (self.plan_preview(plan) + self.row_control(control)).unsqueeze(-2)
        scores = body.joint(hands) + body.row_consequence.score(hands, local, timing)
        scores = scores + torch.where(body.has_head[None],
            body.route_residual(hands, torch.ones(len(audio), dtype=torch.bool, device=audio.device)), 0.)
        scores = scores + body.release_residual(hands, occupancy.any(-1))
        return scores.masked_fill(~support, -torch.inf).log_softmax(-1)


def warm_model(checkpoint, device='cpu', *, style_names=()):
    old = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = TypedAudioModel(PlannedModelConfig(**old['model_config']), style_names=style_names)
    model.body.load_state_dict({k: v for k, v in old['model'].items() if k in model.body.state_dict()}, strict=True)
    if 'head_base.weight' in old['model']:
        with torch.no_grad():
            model.clock_audio.weight[::2].copy_(old['model']['head_base.weight'])
            model.clock_audio.bias[::2].copy_(old['model']['head_base.bias'])
    return model.to(device)
