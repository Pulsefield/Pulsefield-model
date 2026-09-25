"""Lazy head lookahead, online releases and exact row commitment from full audio."""
import hashlib
import io
from pathlib import Path
import time

import torch

from ..joint_audio_continuation.batching import interpolate_audio
from ..joint_audio_continuation.generation import _synchronize
from ..joint_audio_continuation.timing import sample_hazards
from ..scoped_style_modeling.dataset import ContractError
from .features import HeadPreview, head_clocks, skeleton_tokens
from .model import PlannedAudioModel, PlannedModelConfig
from .session import ContinuationSession, PublicationLog, _BudgetStop
from .row_constraints import NoRowContinuation
from .spacing import next_head_earliest


class HeadPlanner:
    """A pure head-stream sampler; row decisions never change its state or RNG."""
    def __init__(self, model, encoded, duration_ms, seed, chunk_ms=500, guard=None):
        self.model, self.encoded, self.duration_ms = model, encoded, duration_ms
        self.device, self.dtype = encoded.device, encoded.dtype
        self.rng = torch.Generator(device='cpu').manual_seed(seed)
        self.chunk_ms, self.guard = chunk_ms, guard
        self.cursor, self.last_head, self.residual = -1, None, None
        self.cache = model.head_temporal.empty_cache()
        self.queue, self.generated = [], []
        self.bins = 0

    @property
    def finished(self):
        return self.cursor == self.duration_ms

    @torch.no_grad()
    def fill(self, count):
        while len(self.queue) < count and not self.finished:
            if self.guard is not None:
                self.guard()
            end = min(self.cursor + self.chunk_ms, self.duration_ms)
            bins = torch.arange((self.cursor + 1) // 10, end // 10 + 1, device=self.device)
            anchors = bins * 10 + 9
            audio = interpolate_audio(self.encoded, anchors[None])[0]
            clocks = torch.as_tensor(head_clocks([self.last_head] * len(bins), anchors.cpu().numpy()),
                                     dtype=self.dtype, device=self.device)
            history = self.model.head_temporal.read(self.cache)[None].expand(len(bins), -1, -1)
            logits = self.model.head_logits(audio, history, clocks).flatten()
            native = (bins[:, None] * 10 + torch.arange(10, device=self.device)).flatten()
            valid = (native > self.cursor) & (native <= end)
            if self.model.config.minimum_action_gap_ms:
                valid &= native >= next_head_earliest(self.generated, self.model.config.minimum_action_gap_ms)
            event, self.residual = sample_hazards(logits, valid, torch.zeros_like(valid), self.rng, self.residual)
            self.bins += len(bins)
            if event is None:
                self.cursor = end
                continue
            self.cursor = int(native[event])
            token = skeleton_tokens([self.cursor], [self.last_head])[0]
            self.cache = self.model.head_temporal.append(self.cache, torch.as_tensor(token, dtype=self.dtype, device=self.device))
            self.last_head, self.residual = self.cursor, None
            self.queue.append(self.cursor)
            self.generated.append(self.cursor)

    def preview(self, now, count):
        future = tuple(t for t in self.queue if t > now)[:count]
        return HeadPreview(future, self.finished and len(future) < count)


@torch.inference_mode()
def rollout(model, mel, duration_ms, *, seed, chunk_ms=500, head_chunk_ms=500,
            max_rows=30000, max_seconds=90., on_update=None, stop_callback=None,
            correct_short_attacks=False, arrangement_profile=None, head_times=None,
            row_constraint='none'):
    """Generate one trajectory from full audio and BOS, publishing each step.

    A resource cap returns the materialized prefix with open holds preserved.
    Consumer exceptions propagate. Optional fixed native H times support
    research comparisons; they contain no future row actions or LN endpoints.
    Optional row constraints condition the complete row law on HH/RH checks;
    empty support returns row_constraint_empty without advancing coverage past
    the unmaterialized event. They cannot combine with response correction.
    """
    session = ContinuationSession(model, mel, duration_ms, seed=seed, planner_factory=HeadPlanner,
        chunk_ms=chunk_ms, head_chunk_ms=head_chunk_ms, max_rows=max_rows, max_seconds=max_seconds,
        stop_callback=stop_callback, correct_short_attacks=correct_short_attacks,
        arrangement_profile=arrangement_profile, head_times=head_times, row_constraint=row_constraint)
    publication = PublicationLog(session.started, duration_ms, on_update)
    reason = 'completed'
    try:
        while session.cursor < duration_ms:
            tick = time.perf_counter()
            update = session.step()
            _synchronize(session.device)
            publication.publish(update, time.perf_counter()-tick)
    except (_BudgetStop, NoRowContinuation) as error:
        reason = str(error)
    return publication.result(session, reason)


def load_model(path, expected_sha256, *, device='cpu'):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ContractError('Planned checkpoint differs from its pinned SHA-256')
    payload = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    if payload.get('format') not in ('joint-audio/planned-v1', 'joint-audio/planned-profile-v1'):
        raise ContractError('Expected a planned audio checkpoint')
    model = PlannedAudioModel(PlannedModelConfig(**payload['model_config']))
    if bool(model.config.profile_count) != (payload['format'] == 'joint-audio/planned-profile-v1'):
        raise ContractError('Checkpoint family differs from its arrangement configuration')
    model.load_state_dict(payload['model'], strict=True)
    if any(not bool(torch.isfinite(v).all()) for v in model.state_dict().values()) or not bool((model.audio_std > 0).all()):
        raise ContractError('Planned checkpoint has nonfinite tensors or invalid audio normalization')
    if model.config.profile_count and not bool((model.profile_std > 0).all()):
        raise ContractError('Profile checkpoint has invalid normalization')
    return model.to(device).eval(), {k: payload[k] for k in ('source_revision', 'manifest_sha256', 'config')}
