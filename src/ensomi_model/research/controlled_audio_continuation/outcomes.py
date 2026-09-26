"""Offline scoped control outcomes with explicit future-tail dependencies."""
from dataclasses import dataclass
import math

from ..typed_audio_continuation.difficulty_targets import ScopeStrainTrace


@dataclass(frozen=True)
class ScopedDifficulty:
    level: float
    score_end_ms: int


def scoped_difficulty(objects, start_ms, end_ms):
    """Read a half-open native-ms scope after its relevant LN ends are resolved.

    Keep every object from the full prefix whose head is before end_ms, with
    its real tail, including tails beyond the scope. This retains incoming
    strain and makes unrelated later heads irrelevant by construction. It is
    an offline strain proxy, not a causal player state or an official fragment
    rating. Future tails must never be supplied to the generator as inputs.

    score_end_ms is an exclusive boundary after which actions cannot change
    these selected objects or the resulting scalar. A conditional row-score
    gradient for this outcome needs no later action terms. Other outcomes,
    including a whole-song LN request, have their own dependency horizons.
    """
    if (type(start_ms) is not int or type(end_ms) is not int or
            not 0 <= start_ms < end_ms):
        raise ValueError('A scoped outcome requires a positive native-ms interval')
    prefix = [o for o in objects if o.start_time < end_ms]
    boundary = max([end_ms, *(math.floor(o.end_time)+1 for o in prefix)])
    return ScopedDifficulty(ScopeStrainTrace.from_objects(prefix).level(start_ms, end_ms), boundary)
