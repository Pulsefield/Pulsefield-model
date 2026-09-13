"""Exact source actions, half-open scopes, and target-free model inputs.

Source milliseconds and physical one-based line numbers are never rounded.
This module does not import the legacy tokenizer or replay implementation.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
import math
import re

from .dataset import ContractError, Interval, NoteRef, canonical_json, digest

# Serialized columns -> (hand, outer/inner). Mirror exchanges hands only.
HAND_COLUMNS = ((0, 1), (3, 2))
MIRROR_COLUMNS = (3, 2, 1, 0)


def hand_role(column: int) -> tuple[int, int]:
    if column not in range(4):
        raise ContractError(f"Expected zero-based 4K column: {column}")
    return (0, column) if column < 2 else (1, 3 - column)


def relative_role(column: int, query_hand: int) -> int:
    hand, role = hand_role(column)
    return 2 * (hand != query_hand) + role


def mirror_objects(objects: tuple[NoteRef, ...]) -> tuple[NoteRef, ...]:
    return tuple(replace(n, column=MIRROR_COLUMNS[n.column]) for n in objects)


def time_feature(milliseconds: float | None) -> tuple[float, bool]:
    """Return signed log1p seconds plus availability, preserving a real zero."""
    if milliseconds is None:
        return 0.0, False
    return math.copysign(math.log1p(abs(milliseconds) / 1000), milliseconds), True


@dataclass(frozen=True)
class SourceChart:
    source_sha256: str
    objects: tuple[NoteRef, ...]
    arrangement_sha256: str
    row_incompatibilities: tuple[tuple[int, int], ...]


def parse_source(data: bytes, expected_sha256: str) -> SourceChart:
    """Verify bytes, then parse original 4K objects and reject ambiguous replay.

    Same-lane close/head coincidences retain both exact actions and are reported
    as V3-row incompatibilities. They must not be retimed into invented rows.
    Overlapping holds, duplicate heads, zero-length holds and unsupported object
    kinds fail because their source-to-candidate/lane mapping is ambiguous.
    """
    if digest(data) != expected_sha256:
        raise ContractError("Source SHA-256 mismatch before source-line resolution")
    section = None
    mode = keys = None
    objects = []
    for line_number, raw in enumerate(re.split(r"\r\n|\n|\r", data.decode("utf-8-sig")), 1):
        text = raw.strip()
        if text.startswith("[") and text.endswith("]"):
            section = text
            continue
        if not text or text.startswith("//"):
            continue
        if section == "[General]" and text.startswith("Mode:"):
            mode = int(text.split(":", 1)[1])
        elif section == "[Difficulty]" and text.startswith("CircleSize:"):
            keys = float(text.split(":", 1)[1])
        elif section == "[HitObjects]":
            parts = text.split(",")
            try:
                x, start, flags = float(parts[0]), float(parts[2]), int(parts[3])
                if not math.isfinite(x) or not 0 <= x <= 512:
                    raise ValueError("x must be within 0..512")
                column = min(3, int(x * 4 // 512))
                kind_bits = flags & (1 | 2 | 8 | 128)
                if kind_bits == 128:
                    kind, end = "long", float(parts[5].split(":", 1)[0])
                elif kind_bits == 1:
                    kind, end = "normal", start
                else:
                    raise ValueError(f"unsupported hit-object type {flags}")
                objects.append(NoteRef(line_number, column, kind, start, end))
            except (IndexError, ValueError) as exc:
                raise ContractError(f"source line {line_number}: {exc}") from exc
    if mode != 3 or keys != 4:
        raise ContractError(f"Expected mania Mode:3 and CircleSize:4, found {mode}/{keys}")
    ordered = tuple(sorted(objects, key=lambda n: (n.start_ms, n.column, n.source_line)))
    collisions = []
    for lane in range(4):
        previous = None
        for note in (n for n in ordered if n.column == lane):
            if previous:
                if note.start_ms == previous.start_ms or note.start_ms < previous.end_ms:
                    raise ContractError(f"Ambiguous same-lane source objects {previous.source_line}/{note.source_line}")
                if previous.kind == "long" and note.start_ms == previous.end_ms:
                    collisions.append((previous.source_line, note.source_line))
            previous = note
    signature = sorted((n.column, n.kind, n.start_ms, n.end_ms) for n in ordered)
    return SourceChart(expected_sha256, ordered, digest(canonical_json(signature).encode()), tuple(collisions))


@dataclass(frozen=True)
class LaneFacts:
    tap: bool
    ln_start: bool
    ln_close: bool
    occupied_before: bool
    occupied_after: bool
    previous_attack_ms: float | None
    next_attack_ms: float | None
    before_age_ms: float | None
    before_remaining_ms: float | None
    before_duration_ms: float | None
    after_age_ms: float | None
    after_remaining_ms: float | None
    after_duration_ms: float | None
    before_head_visible: bool
    before_close_visible: bool
    after_head_visible: bool
    after_close_visible: bool


@dataclass(frozen=True)
class Row:
    time_ms: float
    phase: str
    markers: tuple[str, ...]
    lanes: tuple[LaneFacts, ...]
    in_scope: bool
    elapsed_ms: float
    scope_relative_ms: float
    context_relative_ms: float

    @property
    def attack_columns(self) -> tuple[int, ...]:
        return tuple(j for j, f in enumerate(self.lanes) if f.tap or f.ln_start)

    @property
    def action_columns(self) -> tuple[int, ...]:
        return tuple(j for j, f in enumerate(self.lanes) if f.tap or f.ln_start or f.ln_close)


@dataclass(frozen=True)
class ChartInputs:
    """Only chart-derived data. No arbitrary source IDs, concept or target values."""
    scope: Interval
    context: Interval
    rows: tuple[Row, ...]
    section_indices: tuple[int, ...]
    event_indices: tuple[int, ...]
    readout_elapsed_ms: tuple[float, ...]


@dataclass(frozen=True)
class Decision:
    encoder_index: int
    candidates: tuple[NoteRef | None, ...]

    @property
    def eligible_mask(self) -> int:
        # Bits follow source columns 0..3; canonical hand masks use hand_mask().
        return sum(1 << j for j, n in enumerate(self.candidates) if n is not None)

    @property
    def valid_masks(self) -> tuple[int, ...]:
        eligible = self.eligible_mask
        return tuple(mask for mask in range(16) if mask & ~eligible == 0)


def hand_mask(mask: int, hand: int) -> int:
    return sum(((mask >> column) & 1) << role for role, column in enumerate(HAND_COLUMNS[hand]))


@dataclass(frozen=True)
class PreparedChart:
    inputs: ChartInputs
    visible_objects: tuple[NoteRef, ...]
    decisions: tuple[Decision, ...]


def prepare_chart(objects: tuple[NoteRef, ...], scope: Interval, context: Interval) -> PreparedChart:
    """Replay complete Q, preserving outside endpoints only for visible holds.

    Boundary markers are before the source actions at their timestamp. An LN
    closing exactly at a boundary is still occupied at that marker, but cannot
    be selected as an entering LN. No source event at Q.end is encoded.
    """
    if not context.start_ms <= scope.start_ms < scope.end_ms <= context.end_ms:
        raise ContractError("Section must be inside declared review context")
    visible = tuple(n for n in objects if context.contains(n.start_ms) or
                    (n.kind == "long" and n.start_ms < context.start_ms <= n.end_ms))
    # Inclusive close-at-Q.start is required for exact before-phase occupation.
    times = {n.start_ms for n in visible if context.contains(n.start_ms)}
    times.update(n.end_ms for n in visible if n.kind == "long" and context.contains(n.end_ms))
    markers: dict[float, list[str]] = {}
    for t, name in ((context.start_ms, "context_start"), (context.end_ms, "context_end"),
                    (scope.start_ms, "section_start"), (scope.end_ms, "section_end")):
        markers.setdefault(t, []).append(name)
    positions = sorted([(t, "source") for t in times] + [(t, "boundary") for t in markers])
    lane_objects = [tuple(n for n in visible if n.column == j) for j in range(4)]
    attacks = [tuple(n.start_ms for n in lane_objects[j] if context.contains(n.start_ms)) for j in range(4)]
    rows = []
    for t, phase in positions:
        lane_facts = []
        for lane in range(4):
            notes = lane_objects[lane]
            head = next((n for n in notes if n.start_ms == t), None) if phase == "source" else None
            close = next((n for n in notes if n.kind == "long" and n.end_ms == t), None) if phase == "source" else None
            before = next((n for n in notes if n.kind == "long" and n.start_ms < t <= n.end_ms), None)
            after = next((n for n in notes if n.kind == "long" and n.start_ms <= t < n.end_ms), None) if phase == "source" else before
            lane_times = attacks[lane]
            prev_index = bisect_left(lane_times, t) - 1
            next_index = bisect_right(lane_times, t) if phase == "source" else bisect_left(lane_times, t)

            def hold_facts(n: NoteRef | None) -> tuple:
                return (None, None, None) if n is None else (t - n.start_ms, n.end_ms - t, n.end_ms - n.start_ms)

            lane_facts.append(LaneFacts(
                head is not None and head.kind == "normal", head is not None and head.kind == "long", close is not None,
                before is not None, after is not None,
                t - lane_times[prev_index] if prev_index >= 0 else None,
                lane_times[next_index] - t if next_index < len(lane_times) else None,
                *hold_facts(before), *hold_facts(after),
                before is not None and context.contains(before.start_ms), before is not None and context.contains(before.end_ms),
                after is not None and context.contains(after.start_ms), after is not None and context.contains(after.end_ms)))
        rows.append(Row(t, phase, tuple(sorted(markers[t])) if phase == "boundary" else (), tuple(lane_facts),
                        phase == "source" and scope.contains(t), t - rows[-1].time_ms if rows else 0,
                        t - scope.start_ms, t - context.start_ms))
    event_indices = tuple(i for i, row in enumerate(rows) if row.in_scope)
    section_indices = tuple(i for i, row in enumerate(rows) if row.in_scope or
                            "section_start" in row.markers or "section_end" in row.markers)
    elapsed = tuple(0 if k == 0 else rows[i].time_ms - rows[section_indices[k-1]].time_ms for k, i in enumerate(section_indices))
    boundary_index = next(i for i, r in enumerate(rows) if "section_start" in r.markers)
    eligible = tuple(n for n in visible if scope.contains(n.start_ms) or
                     (n.kind == "long" and n.start_ms < scope.start_ms < n.end_ms))
    decisions = []
    for i in (boundary_index, *event_indices):
        row = rows[i]
        candidates = []
        for lane in range(4):
            matches = [n for n in eligible if n.column == lane and
                       (n.start_ms < scope.start_ms if i == boundary_index else n.start_ms == row.time_ms)]
            if len(matches) > 1:
                raise ContractError(f"Multiple candidate objects in lane {lane} at {row.time_ms}/{row.phase}")
            candidates.append(matches[0] if matches else None)
        decisions.append(Decision(i, tuple(candidates)))
    mapped = [n for decision in decisions for n in decision.candidates if n is not None]
    if len(mapped) != len(set(mapped)) or set(mapped) != set(eligible):
        raise ContractError("Every eligible source object must have exactly one selection decision")
    return PreparedChart(ChartInputs(scope, context, tuple(rows), section_indices, event_indices, elapsed), visible, tuple(decisions))


def evidence_masks(prepared: PreparedChart, evidence: tuple[NoteRef, ...] | None) -> tuple[int, ...] | None:
    """Map exact references to targets; missing is None, explicit empty is all zero.

    Source-line, lane, kind and both endpoints must all match. Unknown, duplicate
    or out-of-scope references raise instead of becoming an availability mask.
    """
    if evidence is None:
        return None
    eligible = {n for d in prepared.decisions for n in d.candidates if n is not None}
    if len(set(evidence)) != len(evidence) or not set(evidence) <= eligible:
        raise ContractError(f"Evidence identity/membership failed: {sorted(set(evidence) - eligible)}")
    selected = set(evidence)
    return tuple(sum(1 << j for j, n in enumerate(d.candidates) if n in selected) for d in prepared.decisions)


def selected_objects(prepared: PreparedChart, masks: tuple[int, ...]) -> tuple[NoteRef, ...]:
    """Decode legal simultaneous masks to original identities, including boundary LNs."""
    if len(masks) != len(prepared.decisions):
        raise ContractError("Selection mask count differs from decision timeline")
    result = []
    for d, mask in zip(prepared.decisions, masks):
        if type(mask) is not int or mask not in d.valid_masks:
            raise ContractError(f"Impossible selection mask {mask} for availability {d.eligible_mask}")
        result.extend(n for j, n in enumerate(d.candidates) if mask & (1 << j))
    return tuple(sorted(result))
