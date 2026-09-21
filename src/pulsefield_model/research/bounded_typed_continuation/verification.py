"""Independent full-output checks against timing, seed and chosen object plans.

This verifier does not reuse scheduler feasibility masks or a model cache.
Output parsing/export are separate checks; source suffix actions are not inputs.
"""
from bisect import bisect_left

from ..scoped_style_modeling.dataset import ContractError
from .contract import Arm


def verify_complete(rows, timing, arm, seed, crossing_ends=None, suffix_plans=None):
    """Validate a complete materialized chart, including an unused terminal R.

    Typed seeds receive exactly their crossing LN endpoints. O1 also supplies
    the sampled (start candidate, lane) -> end candidate decisions for every
    suffix-born LN. R0/R1 must not supply suffix plans. No source suffix appears
    in this interface.
    """
    crossing = {} if crossing_ends is None else dict(crossing_ends)
    plans = {} if suffix_plans is None else dict(suffix_plans)
    if not isinstance(arm, Arm) or ((arm == Arm.R0) != (timing.onsets is None)):
        raise ContractError('Output verifier requires timing roles matching the task arm')
    if (arm != Arm.O1 and plans) or (arm == Arm.R0 and crossing):
        raise ContractError('Output verifier received future plans unavailable to this task')
    if not seed and crossing:
        raise ContractError('An empty seed cannot carry crossing LN endpoints')
    opened = [None] * 4
    expected = [None] * 4
    previous = -1
    row_count = heads = ln_heads = onset_rows = 0
    seen_plans = set()
    boundary_checked = not seed
    for row in rows:
        candidate = bisect_left(timing.times_ms, row.time_ms)
        if candidate <= previous or candidate == len(timing.times_ms) or timing.times_ms[candidate] != row.time_ms:
            raise ContractError('Generated rows must be an increasing subset of the supplied candidates')
        if row_count < len(seed) and (candidate != row_count or row != seed[row_count]):
            raise ContractError('Generated output changed or skipped a supplied seed row')
        if not boundary_checked and row_count == len(seed):
            required = {lane for lane, start in enumerate(opened) if start is not None} if arm != Arm.R0 else set()
            if set(crossing) != required:
                raise ContractError('Typed seed crossing endpoints differ from its active LN lanes')
            for lane, end in crossing.items():
                if type(lane) is not int or type(end) is not int or not len(seed) <= end < len(timing.times_ms):
                    raise ContractError('Seed endpoint is not a pending candidate obligation')
                expected[lane] = end
            boundary_checked = True
        has_head = any(action in (1, 2) for action in row.actions)
        if arm != Arm.R0 and has_head != timing.onsets[candidate]:
            raise ContractError('Generated head occurrence disagrees with H')
        onset_rows += has_head
        for lane, action in enumerate(row.actions):
            if expected[lane] is not None and expected[lane] <= candidate and (candidate != expected[lane] or action != 3):
                raise ContractError('Generated output missed a committed LN release')
            if action in (1, 2):
                if opened[lane] is not None:
                    raise ContractError('Generated output attacks an occupied lane')
                heads += 1
                if action == 2:
                    ln_heads += 1
                    opened[lane] = candidate
                    if arm == Arm.O1 and candidate >= len(seed):
                        key = (candidate, lane)
                        end = plans.get(key)
                        if type(end) is not int or not candidate < end < len(timing.times_ms):
                            raise ContractError('Generated O1 LN lacks a strictly future chosen endpoint')
                        expected[lane] = end
                        seen_plans.add(key)
            elif action == 3:
                if opened[lane] is None or opened[lane] >= candidate:
                    raise ContractError('Generated output releases a missing or nonpositive LN')
                if expected[lane] is not None and expected[lane] != candidate:
                    raise ContractError('Generated output released a committed LN early')
                opened[lane] = expected[lane] = None
        previous, row_count = candidate, row_count + 1
    if row_count < len(seed) or any(start is not None for start in opened):
        raise ContractError('Generated output is incomplete or leaves an active LN')
    if not boundary_checked and crossing:
        raise ContractError('Finished output cannot retain a seed crossing commitment')
    if (arm == Arm.R0 and row_count != len(timing.times_ms)) or (arm != Arm.R0 and onset_rows != sum(timing.onsets)):
        raise ContractError('Generated output omitted a required event/onset')
    if seen_plans != set(plans):
        raise ContractError('Output has extra or missing suffix object decisions')
    return dict(rows=row_count, heads=heads, ln_heads=ln_heads, onset_rows=onset_rows,
                skipped_candidates=len(timing.times_ms) - row_count)
