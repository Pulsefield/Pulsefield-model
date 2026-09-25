"""Validate a joint unpublished future before advancing the publication clock.

This research policy conditions each proposed window on finite close-pair
screens. It is not the global whole-chart constrained posterior. The RH screen
is an explicit experimental preference, distinct from the confirmed HH rule.
First proposals reuse a previously screened halo, so they are not independent
base-model draws conditional on published rows alone.
"""
import hashlib
import time

import torch

from ..joint_audio_continuation.generation import GenerationUpdate, _synchronize
from ..scoped_style_modeling.dataset import ContractError
from .generation import HeadPlanner
from .session import ContinuationSession, PublicationLog, _BudgetStop
from .row_constraints import NoRowContinuation


def close_pairs(prefix, rows, through_ms, *, screen_release_heads):
    """Inspect a continuation against actual prior lane clocks, including entry.

    Only the first attack after a release forms an RH pair. Exactly 20 ms is
    excluded from HH and included in the optional RH screen. Later speculative
    rows do not enter a bounded halo's predicate.
    """
    heads = list(prefix.last_lane_attack_ms)
    releases = [r if r is not None and (h is None or r > h) else None
                for r, h in zip(prefix.last_lane_release_ms, heads)]
    result = []
    for row in rows:
        t = int(row.time_ms)
        if t > through_ms:
            break
        for lane, action in enumerate(row.actions):
            if action in (1, 2):
                for kind, previous in (('HH', heads[lane]), ('RH', releases[lane])):
                    qualifies = (previous is not None and
                                 (t - previous < 20 if kind == 'HH' else
                                  screen_release_heads and t - previous <= 20))
                    if qualifies:
                        result.append(dict(kind=kind, lane=lane, previous_ms=previous,
                                           time_ms=t, gap_ms=t-previous))
                heads[lane], releases[lane] = t, None
            elif action == 3:
                releases[lane] = t
    return result


def _retry_seed(seed, window, attempt):
    value = f'planned-window-v1:{seed}:{window}:{attempt}'.encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], 'little') % (2 ** 63)


@torch.inference_mode()
def rollout_buffered(model, mel, duration_ms, *, seed, window_ms=8000, max_attempts=4,
                     screen_release_heads=True, chunk_ms=500, head_chunk_ms=500,
                     max_rows=30000, max_seconds=90., on_update=None, stop_callback=None,
                     arrangement_profile=None, head_times=None, on_rejected=None,
                     row_constraint='none'):
    """Publish only a screened window; keep the last published prefix on a cap.

    Every first proposal restores the accepted boundary's exact RNG/cache state.
    Retries draw new R/row randomness and preserve the H/profile choices. A
    20-ms sampled halo tests interactions with the publication boundary; only
    the earlier window is published. The next first proposal reproduces that
    halo. No rejected row or speculative coverage reaches ``on_update``.

    ``on_rejected`` is a research observer receiving metadata and an independent
    in-memory fork for offline inspection. It must not mutate shared weights or
    audio; its work counts toward the time budget. Observer/consumer exceptions
    propagate. Exhaustion returns an incomplete result with original open holds.
    The packaged inference entrypoint selects this policy only when requested.

    The Python-only row_constraint option conditions each complete-row draw.
    Empty allowed support rejects that unpublished proposal and consumes one
    attempt, rather than publishing the event or changing the R/H clocks.
    Its joint HH/RH predicate requires screen_release_heads=True.
    """
    if (any(type(v) is not int or v <= 0 for v in (window_ms, max_attempts)) or
            type(screen_release_heads) is not bool):
        raise ContractError('Buffered generation requires positive window/attempt bounds and a boolean RH screen')
    if row_constraint != 'none' and not screen_release_heads:
        raise ContractError('Joint row constraints require the RH publication screen')
    accepted = ContinuationSession(model, mel, duration_ms, seed=seed, planner_factory=HeadPlanner,
        chunk_ms=chunk_ms, head_chunk_ms=head_chunk_ms, max_rows=max_rows, max_seconds=max_seconds,
        stop_callback=stop_callback, arrangement_profile=arrangement_profile, head_times=head_times,
        row_constraint=row_constraint)
    publication = PublicationLog(accepted.started, duration_ms, on_update)
    windows, reason = [], 'completed'
    evaluated_rows = max_unpublished = 0
    try:
        while accepted.cursor < duration_ms:
            tick = time.perf_counter()
            begin = accepted
            count = len(begin.rows)
            target = min(duration_ms, begin.cursor + window_ms)
            record = dict(start_ms=begin.cursor, requested_through_ms=target, attempts=[])
            windows.append(record)
            chosen = None
            for attempt in range(max_attempts):
                trial = begin.fork(retry_seed=None if attempt == 0 else
                                   _retry_seed(seed, len(windows)-1, attempt))
                observation = dict(attempt=attempt, status='running', accepted=False, pairs=None)
                record['attempts'].append(observation)
                try:
                    while trial.cursor < target:
                        trial.step()
                    cut = trial.fork()
                    through = min(duration_ms, cut.cursor + 20)
                    observation.update(cut_ms=cut.cursor, checked_through_ms=through)
                    while trial.cursor < through:
                        trial.step()
                    _synchronize(trial.device)
                    pairs = close_pairs(begin.replay, trial.rows[count:], through,
                                        screen_release_heads=screen_release_heads)
                    observation.update(pairs=pairs, accepted=not pairs,
                                       status='rejected' if pairs else 'accepted')
                except _BudgetStop as error:
                    observation.update(status='capped', stop_reason=str(error))
                    raise
                except NoRowContinuation as error:
                    observation.update(status='rejected', constraint_failure=error.decision)
                finally:
                    proposed_rows = len(trial.rows) - count
                    evaluated_rows += proposed_rows
                    max_unpublished = max(max_unpublished, proposed_rows)
                    observation.update(speculative_coverage_ms=trial.cursor, proposed_rows=proposed_rows)
                if observation['accepted']:
                    chosen = cut
                    break
                if on_rejected is not None:
                    on_rejected(dict(window=len(windows)-1, **observation), trial.fork())
            record['service_seconds'] = time.perf_counter() - tick
            if chosen is None:
                reason = 'planning_attempt_limit'
                break
            accepted = chosen
            delivered = accepted.rows[count:]
            for i, row in enumerate(delivered):
                publication.publish(GenerationUpdate(row, int(row.time_ms), row.time_ms == duration_ms),
                                    record['service_seconds'] if i == 0 else 0.)
            if not delivered or delivered[-1].time_ms < accepted.cursor:
                publication.publish(GenerationUpdate(None, accepted.cursor, accepted.cursor == duration_ms),
                                    0. if delivered else record['service_seconds'])
    except _BudgetStop as error:
        reason = str(error)
        if windows:
            windows[-1]['service_seconds'] = time.perf_counter() - tick
    return publication.result(accepted, reason, publication_policy='unpublished-continuation-screen-v1',
        window_ms=window_ms, max_attempts=max_attempts, screen_release_heads=screen_release_heads,
        windows=windows, evaluated_speculative_rows=evaluated_rows, max_unpublished_rows=max_unpublished,
        rejected_proposals=sum(a['status'] == 'rejected' for w in windows for a in w['attempts']),
        step_latency_scope='Window validation service is charged once at batch delivery; use windows for sampling cost')
