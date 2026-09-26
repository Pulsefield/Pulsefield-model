"""Replayable row preferences for deployment and generated-trajectory learning."""
import numpy as np
import torch

from ..bounded_typed_continuation.contract import Arm
from ..scoped_style_modeling.dataset import ContractError
from ..typed_audio_continuation.allocation import ln_episodes
from ..typed_audio_continuation.program import ACTIONS, Resources
from ..typed_audio_continuation.response_preference import RecoveryPreference
from .allocation import LnAmountFeedback, LnAmountState


def recovery_cost(replay, recent_heads, now, duration_ms, controls, preference):
    """Compare complete rows using the deployed empirical recovery preference."""
    control = controls.at([now])
    known = control[0, 2+len(controls.style_names)] > 0
    stars = 2*control[0, 0]+4 if known else np.nan
    cost = preference.row_cost(replay, now, stars)
    if now != duration_ms:
        ages = now-np.asarray(replay.open_ln_start_ms, float)
        cost += (ACTIONS == 3) @ preference.cost(ages, stars, 'hr')
    counts = np.isin(ACTIONS, (1, 2)).sum(-1)
    return cost+preference.head_cost(Resources(starts=replay.open_ln_start_ms),
                                       recent_heads, now, counts, stars)


def advance_recent_heads(recent_heads, row, preference):
    if preference is None:
        return ()
    horizon = max(preference.hh_ms)
    result = tuple((t, h) for t, h in recent_heads if row.time_ms-t < horizon)
    heads = row.actions.count(1)+row.actions.count(2)
    return result+((row.time_ms, heads),) if heads else result


def replay_row_scores(log_probs, example, controls, *, ln_feedback=LnAmountFeedback(),
                      recovery_preference=RecoveryPreference(head_pressure=4.)):
    """Apply the deployed preferences on the trajectory's actual complete prefix.

    Input scores are normalized neural row probabilities for this interval in
    source-row order. The source owner may contain a generated trajectory; no
    source next-action labels are substituted into its history. Prefix rows
    before the interval reconstruct LN feedback and recovery state but incur no
    loss. Physical support must already be applied by the model/collator.

    Return differentiable CPU float64 log probabilities, matching the native
    sampler's arithmetic. This supports the default LN/recovery policy only;
    optional rate, object-demand or response-projection policies require their
    own probability reconstruction.
    """
    source = example.chart.source
    times = source.rows['time']
    first, stop = np.searchsorted(times, (example.start_ms, example.end_ms))
    if log_probs.shape != (stop-first, 256):
        raise ContractError('Sampling scores must match every row in the requested interval')
    values = log_probs.cpu().double()
    spans = ln_episodes(controls)
    allocation, recent = LnAmountState(), ()
    output = []
    for i in range(int(stop)):
        row = source.row(i)
        span = next((s for s in spans if s.start_ms <= row.time_ms < s.end_ms), None)
        allocation = allocation.in_scope(span)
        if i >= first:
            scores = values[i-first]
            if recovery_preference is not None:
                cost = recovery_cost(source.state(Arm.R0, i).replay, recent, row.time_ms,
                    example.chart.duration_ms, controls, recovery_preference)
                scores = (scores-torch.from_numpy(cost)).log_softmax(-1)
            output.append(ln_feedback.scores(scores, allocation) if ln_feedback else scores)
        if ln_feedback is not None:
            allocation = ln_feedback.advance(allocation, row.actions.count(1), row.actions.count(2))
        recent = advance_recent_heads(recent, row, recovery_preference)
    return torch.stack(output) if output else values
