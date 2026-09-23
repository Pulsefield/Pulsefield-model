"""Local next-event distillation on actual generated histories.

Targets include every native-time/complete-row outcome and right censoring.
Source futures are discarded before collation; only the teacher supplies the
continuation distribution. This objective adds no inference parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from ..bounded_typed_continuation.contract import ROW_ACTIONS
from ..scoped_style_modeling.dataset import ContractError
from .batching import JointInputs, collate, interpolate_audio
from .data import query
from .state import exact_features, legal_rows


@dataclass(frozen=True)
class NativeWindow:
    inputs: JointInputs
    times_ms: Tensor
    exact: Tensor
    legal: Tensor
    log_acceptance: Tensor


@dataclass(frozen=True)
class WindowScores:
    timing_logits: Tensor
    row_log_probs: Tensor


@dataclass(frozen=True)
class WindowTarget:
    event: Tensor
    censor: Tensor


def native_window(examples, model_config, *, horizon_ms=80, scale_ms=27., device='cpu'):
    """Read fixed physical prefixes and score every time in a bounded window.

    Each example is (chart, cursor_ms). The chart may own a sampled trajectory;
    its later rows are never labels for this objective. A window does not commit
    hypothetical future events. Padded clocks use legal unscored placeholders.
    """
    if not math.isfinite(scale_ms) or scale_ms < 0:
        raise ContractError('Native teacher head scale must be finite and nonnegative')
    if not examples or any(cursor >= chart.duration_ms for chart, cursor in examples):
        raise ContractError('Native windows need prefixes before the true audio end')
    charts = [chart for chart, _ in examples]
    queries = [replace(query(chart, cursor, horizon_ms=horizon_ms,
                             history_limit=2 ** (model_config.history_levels + 1) - 1),
                       target_index=None, target_time_ms=None, target_actions=None)
               for chart, cursor in examples]
    inputs = collate(queries, charts, model_config, device=device).inputs
    times = (inputs.timing_times_ms.cpu().numpy()[..., None] - 9 + np.arange(10)).reshape(len(examples), -1)
    times = np.maximum(times, np.asarray([q.cursor_ms + 1 for q in queries])[:, None])
    times = np.minimum(times, np.asarray([q.horizon_end_ms for q in queries])[:, None])
    count = times.shape[1]
    replays = [q.replay for q in queries for _ in range(count)]
    terminal = (times == np.asarray([chart.duration_ms for chart in charts])[:, None]).ravel()
    exact = exact_features(replays, times.ravel().tolist()).reshape(len(examples), count, 2, -1)
    legal = legal_rows(replays, terminal).reshape(len(examples), count, 256)
    lane_logs = np.zeros((*times.shape, 4), np.float32)
    if scale_ms:
        for b, q in enumerate(queries):
            for lane, previous in enumerate(q.replay.last_lane_attack_ms):
                if previous is not None:
                    age = times[b] - previous
                    if np.any(age <= 0):
                        raise ContractError('Native future clocks must follow every observed head')
                    lane_logs[b, :, lane] = 4 * np.minimum(0., np.log(age / scale_ms))
    heads = np.isin(np.asarray(ROW_ACTIONS), (1, 2)).astype(np.float32)
    factors = lane_logs @ heads.T
    return NativeWindow(inputs, torch.as_tensor(times, device=device),
                        torch.as_tensor(exact, device=device), torch.as_tensor(legal, device=device),
                        torch.as_tensor(factors, device=device))


def score_window(model, window: NativeWindow):
    """Share one audio/history encoding across all native-clock row queries."""
    inputs = window.inputs
    encoded = model.encode_audio(inputs.mel, inputs.mel_valid)
    history = model.encode_history(inputs.raw, inputs.history_valid, inputs.truncated)
    timing_audio = interpolate_audio(encoded, inputs.timing_times_ms, inputs.mel_starts, inputs.mel_frame_counts)
    timing = model.timing_logits(timing_audio, history, inputs.timing_exact).flatten(1)
    row_audio = interpolate_audio(encoded, window.times_ms, inputs.mel_starts, inputs.mel_frame_counts)
    batch, count = window.times_ms.shape
    rows = model.row_log_probs(row_audio.flatten(0, 1),
        history[:, None].expand(-1, count, -1, -1).flatten(0, 1), window.exact.flatten(0, 1),
        window.legal.flatten(0, 1), inputs.occupancy[:, None].expand(-1, count, -1).flatten(0, 1))
    return WindowScores(timing, rows.reshape(batch, count, -1))


def _log1mexp(nonpositive):
    """Stable log(1-exp(x)), including log zero at x=0."""
    return torch.where(nonpositive < -math.log(2), torch.log1p(-nonpositive.exp()),
                       torch.log(-torch.expm1(nonpositive)))


def next_event_log_mass(scores: WindowScores, valid: Tensor, forced: Tensor,
                        log_acceptance: Tensor | None = None):
    """Return log mass for each first event and the no-event window outcome.

    Thinning changes both event hazard and row probabilities. At an occupied
    true terminal, the event is mandatory and its rows are renormalized.
    """
    timing, rows = scores.timing_logits, scores.row_log_probs
    if (timing.ndim != 2 or rows.shape[:2] != timing.shape or rows.ndim != 3 or
            valid.shape != timing.shape or forced.shape != timing.shape or
            valid.dtype != torch.bool or forced.dtype != torch.bool or bool((forced & ~valid).any())):
        raise ContractError('Window scores require aligned hazards, row laws and valid terminal flags')
    log_hazard, log_no = F.logsigmoid(timing), F.logsigmoid(-timing)
    if log_acceptance is not None:
        if (log_acceptance.shape != rows.shape or not bool(torch.isfinite(log_acceptance).all()) or
                bool((log_acceptance > 0).any())):
            raise ContractError('Marked thinning requires aligned finite nonpositive log factors')
        weighted = rows + log_acceptance
        log_z = weighted.logsumexp(-1)
        # Roundoff in normalized row scores can produce a tiny positive log Z.
        log_hazard = log_hazard + log_z.clamp_max(0.)
        log_no = _log1mexp(log_hazard)
        rows = weighted - log_z[..., None]
    log_hazard = torch.where(forced, torch.zeros_like(log_hazard), log_hazard)
    log_no = torch.where(forced, torch.full_like(log_no, -torch.inf), log_no)
    log_hazard = log_hazard.masked_fill(~valid, -torch.inf)
    log_no = log_no.masked_fill(~valid, 0.)
    cumulative = log_no.cumsum(-1)
    survival = torch.cat((torch.zeros_like(cumulative[:, :1]), cumulative[:, :-1]), -1)
    return survival[..., None] + log_hazard[..., None] + rows, cumulative[:, -1]


@torch.no_grad()
def teacher_target(scores, window: NativeWindow):
    """Freeze the corrected teacher law; no sampled row becomes a target label."""
    event, censor = next_event_log_mass(scores, window.inputs.timing_valid,
                                       window.inputs.timing_forced, window.log_acceptance)
    return WindowTarget(event.exp().detach(), censor.exp().detach())


def window_kl(scores, window: NativeWindow, target: WindowTarget):
    """KL(teacher || student) over native-time/row outcomes plus censor mass."""
    event, censor = next_event_log_mass(scores, window.inputs.timing_valid, window.inputs.timing_forced)
    if (target.event.shape != event.shape or target.censor.shape != censor.shape or
            target.event.requires_grad or target.censor.requires_grad):
        raise ContractError('Native KL requires aligned detached teacher probabilities')

    def term(probability, log_student):
        positive = probability > 0
        safe_student = log_student.masked_fill(~positive, 0.)
        log_teacher = probability.clamp_min(torch.finfo(probability.dtype).tiny).log()
        return probability * (log_teacher - safe_student)

    return term(target.event, event).sum((-2, -1)) + term(target.censor, censor)
