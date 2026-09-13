"""Combine exact source relations once per hand-node pair for attention."""
from __future__ import annotations

from dataclasses import dataclass

from .replay import PreparedChart, hand_role, relative_role


@dataclass(frozen=True, order=True)
class Relation:
    kind: str
    # Roles are self outer/inner, other outer/inner relative to the query hand.
    query_role: int = -1
    neighbor_role: int = -1
    intervening_attack_rows: int = -1
    occupied_before: bool = False
    occupied_after: bool = False
    duration_ms: float = 0
    head_delta_ms: float = 0
    close_delta_ms: float = 0
    endpoint: str = ""


@dataclass(frozen=True)
class Edge:
    query: int
    neighbor: int
    elapsed_ms: float
    query_phase: str
    neighbor_phase: str
    query_attack_group: int
    neighbor_attack_group: int
    role_intersection: int
    query_actions: tuple[int, ...]
    neighbor_actions: tuple[int, ...]
    query_occupied_before: int
    query_occupied_after: int
    neighbor_occupied_before: int
    neighbor_occupied_after: int
    relations: tuple[Relation, ...]


def prepare_relations(prepared: PreparedChart) -> tuple[Edge, ...]:
    """Build all six channels plus self edges from immutable, complete source facts.

    Outside-context endpoints never acquire nodes. Multiple channels and role
    pairs are retained on one edge, so each neighbor occurs once in attention.
    Node index is 2 * row_index + hand; no source-line identity is embedded.
    """
    rows = prepared.inputs.rows
    pairs: dict[tuple[int, int], set[Relation]] = {}

    def add(q: int, n: int, relation: Relation) -> None:
        pairs.setdefault((q, n), set()).add(relation)

    def connect_rows(a: int, b: int, kind: str) -> None:
        for h in range(2):
            for other in range(2):
                add(2*a+h, 2*b+other, Relation(kind))
                add(2*b+other, 2*a+h, Relation(kind))

    for i in range(len(rows)):
        for h in range(2):
            add(2*i+h, 2*i+h, Relation("self"))
            add(2*i+h, 2*i+1-h, Relation("simultaneous"))
        if i:
            connect_rows(i-1, i, "event_succession")
    # Boundary markers receive timeline neighbors without replacing the
    # original adjacency between consecutive source events.
    source_rows = [i for i, row in enumerate(rows) if row.phase == "source"]
    for a, b in zip(source_rows, source_rows[1:]):
        connect_rows(a, b, "event_succession")
    attack_rows = [i for i, row in enumerate(rows) if row.attack_columns]
    attack_rank = {index: rank for rank, index in enumerate(attack_rows)}
    for offset in (1, 2):
        for a, b in zip(attack_rows, attack_rows[offset:]):
            connect_rows(a, b, f"attack_succession_{offset}")
    for lane in range(4):
        indices = [i for i in attack_rows if lane in rows[i].attack_columns]
        h, _ = hand_role(lane)
        for a, b in zip(indices, indices[1:]):
            relation = Relation("same_lane_recurrence", relative_role(lane, h), relative_role(lane, h), attack_rank[b]-attack_rank[a]-1)
            add(2*a+h, 2*b+h, relation)
            add(2*b+h, 2*a+h, relation)
    source_indices = {row.time_ms: i for i, row in enumerate(rows) if row.phase == "source"}
    holds = [n for n in prepared.visible_objects if n.kind == "long"]
    for hold in holds:
        h, role = hand_role(hold.column)
        head, close = source_indices.get(hold.start_ms), source_indices.get(hold.end_ms)
        if head is not None and close is not None:
            for a, b, endpoint in ((head, close, "close"), (close, head, "head")):
                add(2*a+h, 2*b+h, Relation("ln_identity", role, role, duration_ms=hold.end_ms-hold.start_ms,
                    head_delta_ms=hold.start_ms-rows[a].time_ms, close_delta_ms=hold.end_ms-rows[a].time_ms, endpoint=endpoint))
        for i, row in enumerate(rows):
            if row.phase != "source" or not row.action_columns:
                continue
            before = hold.start_ms < row.time_ms <= hold.end_ms
            after = hold.start_ms <= row.time_ms < hold.end_ms
            if not (before or after):
                continue
            for lane in row.action_columns:
                acting_hand, _ = hand_role(lane)
                for target, endpoint in ((head, "head"), (close, "close")):
                    if target is not None:
                        add(2*i+acting_hand, 2*target+h, Relation("occupied_role", relative_role(lane, acting_hand),
                            relative_role(hold.column, acting_hand), occupied_before=before, occupied_after=after,
                            duration_ms=hold.end_ms-hold.start_ms, head_delta_ms=hold.start_ms-row.time_ms,
                            close_delta_ms=hold.end_ms-row.time_ms, endpoint=endpoint))
    def row_attributes(index: int, hand: int) -> tuple:
        row = rows[index]
        group = before = after = 0
        actions = [0]*4
        for lane, facts in enumerate(row.lanes):
            role = relative_role(lane, hand)
            group |= int(facts.tap or facts.ln_start) << role
            before |= int(facts.occupied_before) << role
            after |= int(facts.occupied_after) << role
            actions[role] = int(facts.tap) | (int(facts.ln_start) << 1) | (int(facts.ln_close) << 2)
        return group, tuple(actions), before, after

    attributes = {(i, h): row_attributes(i, h) for i in range(len(rows)) for h in range(2)}
    result = []
    for (q, n), facts in sorted(pairs.items()):
        qi, hand = divmod(q, 2)
        ni = n//2
        qg, qa, qb, qafter = attributes[qi, hand]
        ng, na, nb, nafter = attributes[ni, hand]
        result.append(Edge(q, n, rows[ni].time_ms-rows[qi].time_ms, rows[qi].phase, rows[ni].phase,
                           qg, ng, qg & ng, qa, na, qb, qafter, nb, nafter, tuple(sorted(facts))))
    return tuple(result)
