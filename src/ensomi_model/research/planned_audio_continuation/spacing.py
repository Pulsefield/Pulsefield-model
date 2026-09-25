"""A shared minimum-spacing condition for heads, LN releases and row choices.

This is one explicit response constraint, not a complete playability evaluator.
The head and release helpers read only their declared skeleton/LN inputs. Row
viability may inspect full replay and conditions the final complete-row law.
"""
import numpy as np

from ..bounded_typed_continuation.contract import ROW_ACTIONS
from ..scoped_style_modeling.dataset import ContractError
from .features import LNProjection

_ACTIONS = np.asarray(ROW_ACTIONS)
_HEADS = np.isin(_ACTIONS, (1, 2))


def recovery_values(profile):
    """HH, release-to-head and head-to-release intervals for one session.

    An integer retains the original uniform-spacing policy. A profile supplies
    distinct intervals; it changes decoding support, not V3 chart legality.
    """
    return (profile, profile, profile) if isinstance(profile, int) else (profile.hh, profile.rh, profile.hr)


def next_head_earliest(head_times, gap_ms):
    """Four columns can realize at most four distinct H rows in a gap-wide window."""
    return head_times[-4] + recovery_values(gap_ms)[0] if len(head_times) >= 4 else 0


def check_head_capacity(head_times, gap_ms):
    gap_ms = recovery_values(gap_ms)[0]
    if any(b-a < gap_ms for a, b in zip(head_times, head_times[4:])):
        raise ContractError('Head plan exceeds four-column minimum-spacing capacity')


def release_limits(state, preview, duration_ms, gap_ms):
    """Return earliest eligible release and a necessary release deadline.

    A deadline equal to the next H may be satisfied by releasing in that H row.
    A deadline strictly before it requires a release-only event. With no next H,
    the true audio end is the deadline. None/None means there is no open hold.
    Ages and the observation clock are the only materialized-row input here.
    """
    if not isinstance(state, LNProjection):
        raise ContractError('Release spacing requires an LN-only projection')
    starts = [t for t in state.starts_ms if t is not None]
    if not starts:
        return None, None
    heads = preview.times_ms
    if any(t <= state.observed_through_ms or t > duration_ms for t in heads):
        raise ContractError('Release spacing preview must follow the observation clock')
    if not preview.complete and len(heads) < 4:
        raise ContractError('Release spacing requires four future heads or a complete preview')
    hh, rh, hr = recovery_values(gap_ms)
    earliest = max(state.observed_through_ms+1, min(starts)+hr)
    free = 4-len(starts)
    critical = next((heads[i] for i in range(free, len(heads))
                     if free == 0 or heads[i]-heads[i-free] < hh), None)
    deadline = min(duration_ms, critical-rh) if critical is not None else duration_ms
    return earliest, deadline


def allowed_rows(state, now, preview, duration_ms, gap_ms, *, actions=None):
    """Admit current-clean rows whose post-state has a spaced future realization.

    The caller intersects physical row support and ensures the H-capacity law.
    Future realization may release existing holds and place one TAP per H.
    For each held column, earliest release is max(now+1, start+HR); its next
    attack must also meet RH and HH. Free-column clocks are exact.

    Original availability restrictions expire within max(HH, HR+RH, 1+RH).
    Beyond that, the four-H capacity law suffices. The required head count
    follows from that horizon and HH; alternatively the preview must reach its
    end or declare completion. Open holds must also be closable
    by the true audio end. This tests existence, not learned future probability.
    """
    hh, rh, hr = recovery_values(gap_ms)
    if any(type(v) is not int or v < 1 for v in (hh, rh, hr)):
        raise ContractError('Action spacing requires a positive native-ms gap')
    if any(t <= now or t > duration_ms for t in preview.times_ms):
        raise ContractError('Row spacing preview must contain strictly future H times')
    reach = max(hh, hr+rh, 1+rh)
    horizon = min(duration_ms, now+reach)
    sufficient_heads = 4*((reach+hh-1)//hh)+1
    if (not preview.complete and len(preview.times_ms) < sufficient_heads and
            (not preview.times_ms or preview.times_ms[-1] < horizon)):
        raise ContractError('Row spacing requires a complete recovery horizon or sufficient future heads')
    actions = _ACTIONS if actions is None else np.asarray(actions).reshape(-1, 4)
    heads = _HEADS if actions is _ACTIONS else np.isin(actions, (1, 2))
    attacks = np.asarray([t if t is not None else -np.inf for t in state.last_lane_attack_ms])
    releases = np.asarray([t if t is not None else -np.inf for t in state.last_lane_release_ms])
    starts = np.asarray(state.open_ln_start_ms, np.float64)
    bad_head = heads & ((now-attacks < hh) | (now-releases < rh))
    bad_release = (actions == 3) & (now-starts < hr)
    allowed = ~(bad_head | bad_release).any(-1)
    post_attack = np.where(heads, now, attacks)
    post_release = np.where(actions == 3, now, releases)
    post_start = np.where(actions == 3, np.nan, np.where(actions == 2, now, starts))
    occupied = ~np.isnan(post_start)
    earliest_release = np.maximum(now+1, post_start+hr)
    allowed &= (~occupied | (earliest_release <= duration_ms)).all(-1)
    available_at = np.where(occupied, np.maximum(earliest_release+rh, post_attack+hh),
                            np.maximum(post_attack+hh, post_release+rh))
    index = np.arange(len(actions))
    for time_ms in preview.times_ms:
        if time_ms > horizon:
            break
        eligible = available_at <= time_ms
        allowed &= eligible.any(-1)
        # All already-eligible columns are equivalent for later TAP feasibility.
        # Replacing any one with time+gap preserves the remaining future options.
        lane = eligible.argmax(-1)
        available_at[index, lane] = time_ms+hh
    return allowed
