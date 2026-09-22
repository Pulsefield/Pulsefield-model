"""Offline correctness checks against raw source intervals, without a model.

The source-side oracle uses endpoint bisection rather than the engine's action
transitions. Its future-aware facts stay here and never become prediction input.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import asdict

from ..scoped_style_modeling.dataset import ContractError
from ..source_action_modeling.actions import parse_source
from .data import admit_source
from .engine import ContinuationState, prefill


def _check_facts(state, lane_objects, time_ms, *, inclusive):
    bisect = bisect_right if inclusive else bisect_left
    heads, attacks, releases = [], [], []
    note_count = 0
    for starts, ends, holds, hold_ends in lane_objects:
        count = bisect(starts, time_ms)
        note_count += count
        attacks.append(starts[count - 1] if count else None)
        closes = bisect(ends, time_ms)
        releases.append(ends[closes - 1] if closes else None)
        head_index = bisect(holds, time_ms) - 1
        occupied = head_index >= 0 and (hold_ends[head_index] > time_ms if inclusive else
                                        hold_ends[head_index] >= time_ms)
        heads.append(holds[head_index] if occupied else None)
    expected = {"open_ln_start_ms": tuple(heads), "last_lane_attack_ms": tuple(attacks),
                "last_lane_release_ms": tuple(releases), "note_count": note_count}
    for name, value in expected.items():
        if getattr(state, name) != value:
            phase = "post" if inclusive else "pre"
            raise ContractError(f"Raw source {phase}-row {name} differs at {time_ms} ms")


def verify_source(data: bytes, expected_sha256: str, *, group_id: str, split: str) -> dict:
    """Check every source pre/post-state and selected full-prefix replay boundaries.

    Reads only caller-supplied bytes, retains a constant number of state samples,
    and returns a JSON-compatible report. Invalid source/replay raises rather
    than producing a success record. Ineligible short charts are still replayed
    and reported with their seed exclusion reason.
    """
    source = admit_source(data, expected_sha256, group_id=group_id, split=split)
    original = parse_source(data, expected_sha256)
    lanes = []
    for lane in range(4):
        objects = [note for note in original.objects if note.column == lane]
        holds = [note for note in objects if note.kind == "long"]
        lanes.append((tuple(note.start_ms for note in objects), tuple(note.end_ms for note in holds),
                      tuple(note.start_ms for note in holds), tuple(note.end_ms for note in holds)))
    seed = source.minimum_seed()
    boundaries = {0, len(source.targets)}
    if seed.eligible:
        boundaries.update((seed.seed_row_count, max(seed.seed_row_count, len(source.targets) // 2)))
    state = ContinuationState(source.skeleton)
    snapshots = {0: state}
    for index, row in enumerate(source.targets):
        query = state.query()
        if query != state.query() or query.time_ms != row.time_ms:
            raise ContractError("Repeated prediction query changed state or source time")
        if query.is_terminal != (index == len(source.targets) - 1):
            raise ContractError("Terminal flag differs from the true source endpoint")
        _check_facts(query.history, lanes, row.time_ms, inclusive=False)
        state = state.commit(row)
        _check_facts(state.replay, lanes, row.time_ms, inclusive=True)
        if state.next_index in boundaries:
            snapshots[state.next_index] = state
    for boundary, expected in snapshots.items():
        if prefill(source.skeleton, source.targets[:boundary]) != expected:
            raise ContractError(f"Full-prefix replay differs at row boundary {boundary}")
        if seed.eligible and seed.seed_row_count <= boundary < len(source.targets):
            if source.prefix_state(boundary) != expected:
                raise ContractError(f"Continuation seed replay differs at row boundary {boundary}")
    return {
        "schema": "oracle-time-exact-replay-verification-v1", "status": "passed",
        "identity": asdict(source.identity), "source_objects": len(original.objects),
        "source_rows": len(source.targets), "verified_pre_post_pairs": len(source.targets),
        "long_notes": sum(note.kind == "long" for note in original.objects),
        "release_only_rows": sum(all(a in (0, 3) for a in row.actions) for row in source.targets),
        "seed": asdict(seed), "verified_prefix_boundaries": sorted(snapshots),
        "terminal_occupancy": list(state.replay.occupancy),
    }
