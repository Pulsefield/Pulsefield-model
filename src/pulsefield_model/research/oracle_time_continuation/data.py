"""Verified source/target ownership and the complete-row 30-note seed rule."""
from __future__ import annotations

from dataclasses import dataclass

from ..scoped_style_modeling.dataset import ContractError
from ..source_action_modeling.actions import ATTACK_ACTIONS, LN_CLOSE, LN_START, TAP, parse_source
from .engine import ContinuationState, prefill
from .schema import CompleteRow, TimeSkeleton

MIN_SEED_NOTES = 30


@dataclass(frozen=True)
class SourceIdentity:
    """Provenance supplied by the existing song-group allocation, outside model input."""

    source_sha256: str
    arrangement_sha256: str
    group_id: str
    split: str

    def __post_init__(self) -> None:
        for value in (self.source_sha256, self.arrangement_sha256):
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ContractError("Source identity requires lowercase SHA-256 digests")
        if not isinstance(self.group_id, str) or not self.group_id.strip() or self.split not in ("train", "validation", "test"):
            raise ContractError("Source identity requires its existing song group and train/validation/test split")


@dataclass(frozen=True)
class SeedSelection:
    """Count original hit objects (TAP/LN_START), retaining the whole threshold row.

    An ineligible chart retains its observed counts and an explicit reason; it
    never receives a smaller minimum. Elapsed time runs from the first source row.
    """

    seed_note_count: int
    seed_row_count: int
    seed_elapsed_ms: float
    ineligible_reason: str | None

    @property
    def eligible(self) -> bool:
        return self.ineligible_reason is None


@dataclass(frozen=True)
class ContinuationSource:
    """Supervision owner, including all future actions; never a predictor argument.

    Source identities and original LN endpoints are admitted here. Targets are
    immutable complete rows and the skeleton is their attack/release union.
    Construction validates complete source replay without retaining its states.
    """

    identity: SourceIdentity
    skeleton: TimeSkeleton
    targets: tuple[CompleteRow, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.targets, tuple) or tuple(row.time_ms for row in self.targets) != self.skeleton.times_ms:
            raise ContractError("Source targets must exactly cover the time skeleton")
        prefill(self.skeleton, self.targets)

    def minimum_seed(self) -> SeedSelection:
        notes, rows = 0, 0
        for row in self.targets:
            notes += sum(action in ATTACK_ACTIONS for action in row.actions)
            rows += 1
            if notes >= MIN_SEED_NOTES:
                break
        elapsed = self.targets[rows - 1].time_ms - self.targets[0].time_ms if rows else 0.0
        reason = ("fewer-than-30-notes" if notes < MIN_SEED_NOTES else
                  "no-target-suffix" if rows == len(self.targets) else None)
        return SeedSelection(notes, rows, elapsed, reason)

    def prefix_state(self, target_start: int | None = None) -> ContinuationState:
        """Replay every preceding row using the 30-note minimum, including open LNs.

        A later target start extends the original prefix; it cannot initialize
        from a cropped window. Short charts and empty suffixes raise ContractError.
        """
        seed = self.minimum_seed()
        if not seed.eligible:
            raise ContractError(f"Chart is ineligible for continuation: {seed.ineligible_reason}")
        start = seed.seed_row_count if target_start is None else target_start
        if type(start) is not int or not seed.seed_row_count <= start < len(self.targets):
            raise ContractError("Target start must follow the full 30-note seed and leave a suffix")
        return prefill(self.skeleton, self.targets[:start])


def admit_source(data: bytes, expected_sha256: str, *, group_id: str, split: str) -> ContinuationSource:
    """Verify original 4K bytes and build exact rows, without scope markers or retiming.

    Reuses V3 raw admission only. Prepared-chart features and future LN pairing
    never enter prediction; the scheduler separately exposes bounded skeleton times.
    Nonfinite/negative times and incompatible single-lane coincidences fail.
    Group and split must come from the caller's existing song-group allocation.
    """
    source = parse_source(data, expected_sha256)
    events: dict[float, list[int]] = {}
    for note in source.objects:
        events.setdefault(note.start_ms, [0] * 4)[note.column] = LN_START if note.kind == "long" else TAP
        if note.kind == "long":
            events.setdefault(note.end_ms, [0] * 4)[note.column] = LN_CLOSE
    rows = tuple(CompleteRow(time, tuple(actions)) for time, actions in sorted(events.items()))
    identity = SourceIdentity(source.source_sha256, source.arrangement_sha256, group_id, split)
    return ContinuationSource(identity, TimeSkeleton(tuple(row.time_ms for row in rows)), rows)
