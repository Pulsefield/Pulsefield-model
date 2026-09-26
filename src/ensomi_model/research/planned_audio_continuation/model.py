"""Separate skeleton histories, direct row audio and R1 candidate consequences."""
from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path

import torch
from torch import nn

from ..bounded_typed_continuation.consequence import RowConsequence
from ..bounded_typed_continuation.features import TIME_DIM
from ..bounded_typed_continuation.temporal import FiniteTemporal, TemporalConfig, pointwise
from ..joint_audio_continuation.context_model import ContextAudioModel, ContextModelConfig
from ..joint_audio_continuation.model import initialize_from_r1 as initialize_rows
from ..scoped_style_modeling.dataset import ContractError
from .features import RELEASE_CLOCK_DIM
from .counts import RowCountModel


@dataclass(frozen=True)
class PlannedModelConfig(ContextModelConfig):
    global_audio: bool = True
    head_hidden: int = 64
    head_levels: int = 6
    skeleton_hidden: int = 64
    skeleton_levels: int = 5
    lookahead: int = 16
    bounded_head: bool = False
    head_bound: float = 4.
    head_decay_ms: float = 1000.
    condition_full_holds: bool = False
    profile_count: int = 0
    profile_head_rate_downstream: bool = True
    row_factorization: str = 'flat'
    minimum_action_gap_ms: int = 0

    def __post_init__(self):
        super().__post_init__()
        if not self.global_audio or self.bounded_timing:
            raise ContractError('Planned model requires full audio and excludes the coupled timing residual')
        if any(type(v) is not int or v <= 0 for v in
               (self.head_hidden, self.head_levels, self.skeleton_hidden, self.skeleton_levels, self.lookahead)):
            raise ContractError('Skeleton dimensions and lookahead must be positive integers')
        if self.head_levels > 8 or self.skeleton_levels > self.history_levels or self.lookahead < 2:
            raise ContractError('Skeleton context exceeds its bounded history or lacks two-head preview')
        if type(self.bounded_head) is not bool:
            raise ContractError('Bounded head mode must be boolean')
        if type(self.condition_full_holds) is not bool:
            raise ContractError('Full-hold conditioning mode must be boolean')
        if type(self.profile_count) is not int or self.profile_count < 0:
            raise ContractError('Profile count must be a nonnegative integer')
        if type(self.profile_head_rate_downstream) is not bool:
            raise ContractError('Downstream profile head-rate mode must be boolean')
        if not self.profile_head_rate_downstream and not self.profile_count:
            raise ContractError('Downstream profile routing requires arrangement profiles')
        if self.row_factorization not in ('flat', 'count_layout'):
            raise ContractError('Row factorization must be flat or count_layout')
        if self.row_factorization == 'count_layout' and self.history_levels < 4:
            raise ContractError('Count composition requires at least 31 rows of history')
        if type(self.minimum_action_gap_ms) is not int or self.minimum_action_gap_ms < 0:
            raise ContractError('Minimum action gap must be a nonnegative native-ms integer')
        if self.minimum_action_gap_ms and self.lookahead < 9:
            raise ContractError('Action spacing requires at least nine lookahead heads')
        if self.minimum_action_gap_ms and not self.condition_full_holds:
            raise ContractError('Action spacing requires conditional release waits')
        if any(not math.isfinite(v) or v <= 0 for v in (self.head_bound, self.head_decay_ms)):
            raise ContractError('Head history bound and decay must be finite and positive')


class PlannedAudioModel(ContextAudioModel):
    """Reuse full-audio and row modules, with a distinct probability interface.

    The inherited timing MLP and global timing projection now score the head
    stream from its own history. They never receive the row TCN or exact replay.
    Old flat-event entrypoints reject calls rather than silently omit the plan.
    """
    def __init__(self, config: PlannedModelConfig):
        super().__init__(config)
        self.head_temporal = FiniteTemporal(TemporalConfig(TIME_DIM, config.head_hidden, config.head_levels, config.expansion))
        self.skeleton_temporal = FiniteTemporal(TemporalConfig(TIME_DIM + 2, config.skeleton_hidden,
                                                              config.skeleton_levels, config.expansion))
        self.head_condition = nn.Sequential(nn.Linear(config.head_hidden + 2 * TIME_DIM, config.hidden), nn.GELU())
        self.release_clock = nn.Sequential(
            nn.Linear(config.skeleton_hidden + RELEASE_CLOCK_DIM + config.conditioned_audio_width, config.hidden),
            nn.GELU(), nn.Linear(config.hidden, 10))
        nn.init.normal_(self.release_clock[-1].weight, std=.001)
        nn.init.constant_(self.release_clock[-1].bias, math.log(.002 / .998))
        self.preview_condition = nn.Linear((config.lookahead + 1) * TIME_DIM + 2, config.hidden, bias=False)
        nn.init.zeros_(self.preview_condition.weight)
        self.row_consequence = RowConsequence(config.hidden, 'frontier2')
        if config.bounded_head:
            # Construct after all shared modules so the baseline parameter draws
            # remain paired. This branch is a rate; the old timing MLP becomes
            # a centered historical correction rather than another rate.
            self.head_base = nn.Linear(config.conditioned_audio_width, 10)
            nn.init.normal_(self.head_base.weight, std=.001)
            nn.init.constant_(self.head_base.bias, math.log(.006 / .994))
            nn.init.zeros_(self.timing[-1].bias)
        if config.profile_count:
            self.profile_condition = nn.Linear(3, config.conditioned_audio_width, bias=False)
            self.profile_prior = nn.Linear(config.context_width, config.profile_count)
            nn.init.zeros_(self.profile_condition.weight)
            nn.init.zeros_(self.profile_prior.weight)
            nn.init.zeros_(self.profile_prior.bias)
            self.register_buffer('profile_values', torch.zeros(config.profile_count, 3))
            self.register_buffer('profile_codes', torch.zeros(config.profile_count, 3))
            self.register_buffer('profile_mean', torch.zeros(3))
            self.register_buffer('profile_std', torch.ones(3))
        if config.row_factorization == 'count_layout':
            self.row_counts = RowCountModel(config.conditioned_audio_width,
                (config.lookahead + 1) * TIME_DIM + 2, config.expansion)

    @torch.no_grad()
    def configure_profiles(self, bank):
        from .profiles import PROFILE_FORMAT, normalized
        if bank['format'] != PROFILE_FORMAT or bank['count'] != self.config.profile_count:
            raise ContractError('Profile bank differs from the model configuration')
        for buffer, value in ((self.profile_values, bank['profiles']), (self.profile_mean, bank['mean']),
                              (self.profile_std, bank['std']), (self.profile_codes, normalized(bank['profiles'], bank))):
            buffer.copy_(torch.as_tensor(value, dtype=buffer.dtype, device=buffer.device))
        masses = self.profile_prior.bias.new_tensor(bank['masses'])
        if not bool((masses > 0).all()) or not bool((self.profile_std > 0).all()):
            raise ContractError('Profile masses and scales must be positive')
        self.profile_prior.bias.copy_(masses.log())

    def profile_log_probs(self, coarse):
        """One prior over arrangements, reading only real full-song audio tokens."""
        if not self.config.profile_count:
            raise ContractError('This checkpoint has no arrangement profile prior')
        values, counts = coarse
        valid = torch.arange(values.shape[1], device=values.device)[None] < counts[:, None]
        pooled = (values * valid[..., None]).sum(1) / counts[:, None]
        return self.profile_prior(pooled).log_softmax(-1)

    def condition_audio(self, encoded, profile_index=None, *, downstream=False):
        """Condition unmodified audio for H or for the release/row factors.

        Optional routing removes only the standardized density request from
        downstream inputs. Actual H preview and other profile fields remain.
        """
        if not self.config.profile_count and profile_index is None:
            return encoded
        if type(profile_index) is not int or not 0 <= profile_index < self.config.profile_count:
            raise ContractError('A profiled query requires an in-range arrangement profile index')
        code = self.profile_codes[profile_index]
        if downstream and not self.config.profile_head_rate_downstream:
            code = code * code.new_tensor((0., 1., 1.))
        return encoded + self.profile_condition(code)

    def encode_generation(self, mel, *, seed, code=None):
        """Return unconditioned audio and a profile choice without event-RNG use.

        Callers form H and downstream views with condition_audio so excluded
        attributes are never added and then subtracted in floating point.
        """
        if not self.config.profile_count:
            if code is not None:
                raise ContractError('This checkpoint has no arrangement profiles')
            return self.encode_audio(mel), {}
        if len(mel) != 1:
            raise ContractError('Native profile selection requires one complete song')
        valid = torch.ones(mel.shape[:2], dtype=torch.bool, device=mel.device)
        coarse = self.encode_coarse(mel, valid)
        prior = self.profile_log_probs(coarse)[0].exp()
        sampled = code is None
        if sampled:
            rng = torch.Generator(device='cpu').manual_seed(seed)
            code = int(torch.multinomial(prior.detach().cpu().double(), 1, generator=rng))
        encoded = self.encode_crop(mel, valid, torch.zeros(1, dtype=torch.long, device=mel.device),
                                   valid.sum(-1), coarse)
        return encoded, dict(arrangement_profile=code, arrangement_values=self.profile_values[code].tolist(),
            arrangement_seed=seed, arrangement_selection='audio_prior' if sampled else 'requested',
            arrangement_prior=prior.detach().cpu().tolist())

    def timing_logits(self, *args, **kwargs):
        raise ContractError('Planned model requires separate head and release probability queries')

    @property
    def requires_full_audio_queries(self):
        return False

    def hold_audio_options(self, encoded, starts, times, *, audio_starts=None, frame_counts=None):
        """Optional LN-origin observations for release and row factors, never H."""
        return {}

    def row_log_probs(self, *args, **kwargs):
        raise ContractError('Planned rows require head preview and candidate consequences')

    def _head_proposal(self, audio, history, clocks):
        if (audio.shape[:-1] != history.shape[:-2] or history.shape[-2:] != (2, self.config.head_hidden) or
                clocks.shape != (*audio.shape[:-1], 2 * TIME_DIM)):
            raise ContractError('Head queries require their own history and head-only clocks')
        value = pointwise(self.head_condition, torch.cat((history.mean(-2), clocks), -1))
        value = pointwise(self.timing[0], torch.cat((value, audio[..., :self.config.audio_width]), -1))
        value = value + pointwise(self.context_timing, audio[..., self.config.audio_width:])
        return pointwise(self.timing[2], self.timing[1](value))

    def head_parts(self, audio, history, clocks):
        """Return an audio base, bounded historical residual and recency gate.

        The first clock is elapsed time since the last H at the canonical bin
        query time. Its availability bit distinguishes BOS from a long wait.
        A long wait removes the historical veto, never forces a new head.
        """
        if not self.config.bounded_head:
            raise ContractError('Head decomposition requires bounded head mode')
        raw = self._head_proposal(audio, history, clocks)
        base = pointwise(self.head_base, audio)
        age = clocks[..., :TIME_DIM]
        elapsed_ms = 1000 * age[..., 1].sinh().clamp_min(0)
        gate = torch.exp(-elapsed_ms / self.config.head_decay_ms) * age[..., -1]
        residual = self.config.head_bound * gate[..., None] * raw.tanh()
        return base, residual, gate

    def head_logits(self, audio, history, clocks):
        if not self.config.bounded_head:
            return self._head_proposal(audio, history, clocks)
        base, residual, _ = self.head_parts(audio, history, clocks)
        return base + residual

    def release_logits(self, audio, history, clocks):
        if (audio.shape[:-1] != history.shape[:-2] or
                history.shape[-2:] != (2, self.config.skeleton_hidden) or
                clocks.shape != (*audio.shape[:-1], 2, RELEASE_CLOCK_DIM)):
            raise ContractError('Release queries require skeleton history and LN-only clocks')
        paired_audio = audio.unsqueeze(-2).expand(*history.shape[:-1], audio.shape[-1])
        return pointwise(self.release_clock, torch.cat((history, clocks, paired_audio), -1)).mean(-2)

    def planned_row_log_probs(self, audio, history, exact, legal, occupancy, preview, local, timing,
                              *, count_history=None, count_clock=None, response_allowed=None):
        if legal.dtype != torch.bool or not bool(legal.any(-1).all()):
            raise ContractError('Planned row support must contain a legal complete row')
        hands = self.condition(history, exact, audio) + self.preview_condition(preview).unsqueeze(-2)
        scores = self.joint(hands) + self.row_consequence.score(hands, local, timing)
        routed = self.route_residual(hands, torch.ones(len(audio), dtype=torch.bool, device=audio.device))
        scores = scores + torch.where(self.has_head[None], routed, torch.zeros_like(routed))
        scores = scores + self.release_residual(hands, occupancy.any(-1))
        if self.config.row_factorization == 'count_layout':
            if count_history is None or count_clock is None:
                raise ContractError('Count/layout rows require a separate count history and LN projection')
            marks = self.row_counts.logits(audio, count_history, preview, count_clock)
            result = self.row_counts.compose(scores, marks, legal)
        else:
            result = scores.masked_fill(~legal, -torch.inf).log_softmax(-1)
        if self.config.minimum_action_gap_ms:
            if (response_allowed is None or response_allowed.dtype != torch.bool or
                    response_allowed.shape != legal.shape or not bool((response_allowed & legal).any(-1).all())):
                raise ContractError('Action spacing requires nonempty complete-row response support')
            result = result.masked_fill(~response_allowed, -torch.inf).log_softmax(-1)
        elif response_allowed is not None:
            raise ContractError('Unconfigured action-spacing support would be unused')
        return result


@torch.no_grad()
def initialize_from_r1(model, path, expected_sha256):
    """Restore compatible row and frontier2 weights; report every omission.

    Native release opportunities and audio/plan conditioning differ from R1.
    Copying these tensors does not assert equality of the resulting policies.
    """
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ContractError('R1 checkpoint differs from the pinned SHA-256')
    payload = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    if payload.get('config', {}).get('model', {}).get('row_consequence') != 'frontier2':
        raise ContractError('Planned initialization requires the R1 frontier2 checkpoint')
    receipt = initialize_rows(model, path, expected_sha256)
    names = ['row_consequence.' + name for name in model.row_consequence.state_dict()]
    source = payload['model']
    model.row_consequence.load_state_dict({name.removeprefix('row_consequence.'): source[name] for name in names}, strict=True)
    if any(not bool(torch.isfinite(source[name]).all()) for name in names):
        raise ContractError('R1 consequence weights must be finite')
    count = sum(source[name].numel() for name in names)
    receipt['copied'] = sorted([*receipt['copied'], *names])
    receipt['omitted'] = sorted(set(receipt['omitted']) - set(names))
    receipt['omitted_modules'] = sorted({n.split('.')[0] for n in receipt['omitted']})
    receipt['copied_parameters'] += count
    receipt['omitted_source_parameters'] -= count
    receipt['consequence_opportunities'] = 'Every native millisecond; earliest possibility is now+1, not a tail prediction'
    return receipt
