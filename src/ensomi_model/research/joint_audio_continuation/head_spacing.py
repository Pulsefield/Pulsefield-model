"""Optional marked-event thinning from same-lane head ages.

This is an experimental decoder prior, not a physical limit or a beat grid.
It multiplies each proposed head by min(1, (age / scale)**4), using time since
the preceding head rather than time since a release. Rejected rows must never
enter replay/history; a true occupied terminal instead renormalizes legal rows.
"""
from __future__ import annotations

import math

import torch

from ..bounded_typed_continuation.contract import ROW_ACTIONS
from ..scoped_style_modeling.dataset import ContractError


def log_acceptance(replay, time_ms, actions, scale_ms):
    """Return the row's log acceptance; closes and first heads contribute zero."""
    if not math.isfinite(scale_ms) or scale_ms < 0:
        raise ContractError('Head-spacing scale must be finite and nonnegative')
    if scale_ms == 0:
        return 0.
    result = 0.
    for lane, action in enumerate(actions):
        previous = replay.last_lane_attack_ms[lane]
        if action not in (1, 2) or previous is None:
            continue
        age = time_ms - previous
        if not math.isfinite(age) or age <= 0:
            raise ContractError('A proposed head must follow its previous same-lane head')
        if age < scale_ms:
            result += 4 * (math.log(age) - math.log(scale_ms))
    return result


def sample_row(log_probs, replay, time_ms, generator, acceptance_generator,
               scale_ms=0., *, forced_terminal=False):
    """Return (proposed action tuple, accepted, log acceptance).

    Sampling uses the original RNG for time/row proposals and a separate CPU
    generator for acceptance. Scale zero and unit-acceptance rows consume no
    acceptance randomness. At a forced terminal, sample the reweighted legal
    distribution and accept unconditionally so required holds cannot be stranded.
    """
    if log_probs.shape != (256,) or generator.device.type != 'cpu' or acceptance_generator.device.type != 'cpu':
        raise ContractError('Row sampling needs 256 log probabilities and CPU generators')
    logs = log_probs.detach()
    if forced_terminal and scale_ms > 0:
        factors = torch.tensor([log_acceptance(replay, time_ms, row, scale_ms)
                                for row in ROW_ACTIONS], dtype=torch.float64)
        probabilities = (logs.cpu().double() + factors).softmax(-1)
    else:
        probabilities = logs.exp().cpu()
    selected = int(torch.multinomial(probabilities, 1, generator=generator))
    actions = ROW_ACTIONS[selected]
    factor = log_acceptance(replay, time_ms, actions, scale_ms)
    if forced_terminal or factor == 0:
        return actions, True, factor
    uniform = float(torch.rand((), dtype=torch.float64, generator=acceptance_generator))
    accepted = uniform == 0 or math.log(uniform) <= factor
    return actions, accepted, factor
