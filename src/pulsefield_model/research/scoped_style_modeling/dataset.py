"""Pinned publication adapter and deterministic source-group splitting.

Metadata and supervision remain here; replay accepts only source objects and
scope/context intervals. No rationale or delivery complement becomes a feature.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from urllib.request import urlopen

DATASET = "sed-i/mania-pattern-annotations"
REVISION = "b22a7a443783e05fee4db4b1d22b8e573ad448ae"
MANIFEST_SHA256 = "92ecf080737ec2e38a2508b0730a672e09edd45d48f46a2d5777cbb70cefcbcf"
METHOD = "method-5ebd91cd0db19242f14bf5d4fc96b328c792b185ec4ab2bc785ca2e13a4d056c"
FOUNDATION = "f-15fa68913bdb2bf3"
SPECIFICATION_SHA256 = "197ae4c5de62d4f7207200c6892562650a47e3dbc0ea80640ac86776e7ea9bdd"
CONCEPTS = ("jack-organization", "stream-organization", "trill-organization", "tech", "ln-coordination")
ASSESSMENTS = ("absent", "supporting", "prominent")
TABLES = ("data/sources.parquet", "data/human.parquet", f"data/agent/{METHOD}.parquet")


class ContractError(ValueError):
    """An identity or representation contract failed; never mask this as missing."""


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def checked_bytes(path: Path, expected: str) -> bytes:
    data = path.read_bytes()
    actual = digest(data)
    if actual != expected:
        raise ContractError(f"{path}: SHA-256 {actual}, expected {expected}")
    return data


def fetch_verified(url: str, path: Path, expected: str, timeout: float = 30) -> bytes:
    """Reuse verified bytes or fetch once; mismatches never replace a cached file."""
    if path.exists():
        return checked_bytes(path, expected)
    with urlopen(url, timeout=timeout) as response:
        data = response.read()
    if digest(data) != expected:
        raise ContractError(f"{url}: SHA-256 {digest(data)}, expected {expected}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)
    return data


def load_snapshot(root: Path, *, download: bool = False) -> tuple[dict, list[dict], dict[str, list[dict]]]:
    """Validate the pinned manifest and table bytes before decoding Parquet."""
    import pyarrow.parquet as pq

    def obtain(name: str, expected: str) -> bytes:
        if download:
            url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{name}"
            return fetch_verified(url, root / name, expected)
        return checked_bytes(root / name, expected)

    manifest = json.loads(obtain("manifest.json", MANIFEST_SHA256))
    if (manifest["contract"], manifest["version"], manifest["release_id"]) != ("beatmap-lens-annotations", 4, "v3"):
        raise ContractError("Expected publication schema v4, release v3")
    decoded = {}
    for name in TABLES:
        metadata = manifest["files"][name]
        obtain(name, metadata["sha256"])
        rows = pq.read_table(root / name).to_pylist()
        if len(rows) != metadata["rows"]:
            raise ContractError(f"{name}: row count differs from publication manifest")
        decoded[name] = rows
    return manifest, decoded[TABLES[0]], {"human": decoded[TABLES[1]], "machine": decoded[TABLES[2]]}


@dataclass(frozen=True)
class Interval:
    start_ms: float
    end_ms: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.start_ms) and math.isfinite(self.end_ms) and self.start_ms < self.end_ms):
            raise ContractError(f"Invalid half-open interval {self}")

    def contains(self, time_ms: float) -> bool:
        return self.start_ms <= time_ms < self.end_ms


@dataclass(frozen=True, order=True)
class NoteRef:
    source_line: int
    column: int
    kind: str
    start_ms: float
    end_ms: float

    def __post_init__(self) -> None:
        if type(self.source_line) is not int or self.source_line < 1 or type(self.column) is not int or not 0 <= self.column < 4:
            raise ContractError(f"Invalid source-line/lane identity: {self}")
        if not all(math.isfinite(v) for v in (self.start_ms, self.end_ms)):
            raise ContractError(f"Nonfinite object time: {self}")
        if not ((self.kind == "normal" and self.end_ms == self.start_ms) or
                (self.kind == "long" and self.end_ms > self.start_ms)):
            raise ContractError(f"Invalid object kind/endpoints: {self}")


@dataclass(frozen=True)
class Record:
    record_ids: tuple[str, ...]
    cell_id: str
    source_sha256: str
    scope: Interval
    context: Interval
    concept: str
    assessment: str
    layer: str
    origins: tuple[str, ...]
    playback_rate: float
    foundation_id: str
    evidence: tuple[NoteRef, ...] | None
    provenance_json: str

    @property
    def exact_cell(self) -> tuple:
        return (self.source_sha256, self.scope.start_ms, self.scope.end_ms, self.concept, self.playback_rate)


def adapt_records(tables: dict[str, list[dict]], sources: list[dict]) -> tuple[tuple[Record, ...], list[dict]]:
    """Collapse agreeing cells within human/machine layers; report all exclusions.

    Conflicts have no supervised target. Agreeing duplicates use the lowest
    record ID's context/evidence without unioning witnesses; all provenance stays.
    Invalid evidence is an error, not an availability state. Null references and
    explicit empty references retain distinct values (None versus ()).
    """
    source_ids = {s["source_sha256"] for s in sources}
    if len(source_ids) != len(sources):
        raise ContractError("Duplicate source identity in source table")
    cells: dict[tuple, list[dict]] = {}
    cell_ids: dict[str, tuple] = {}
    exact_ids: dict[tuple, str] = {}
    record_ids: set[str] = set()
    issues = []
    for layer, rows in sorted(tables.items()):
        if layer not in ("human", "machine"):
            raise ContractError(f"Unsupported judgment layer {layer}")
        for row in rows:
            rid = row["record_id"]
            if rid in record_ids:
                raise ContractError(f"Duplicate record ID {rid}")
            record_ids.add(rid)
            if row["source_sha256"] not in source_ids or row["foundation_id"] != FOUNDATION:
                raise ContractError(f"{rid}: unknown source or Foundation")
            if row["tag_id"] not in CONCEPTS or row["playback_rate"] != 1:
                raise ContractError(f"{rid}: expected frozen five concepts at 1x")
            origin = row["origin"]
            if (layer == "machine" and origin != "agent-reviewed") or (layer == "human" and origin not in ("human-direct", "human-confirmed", "human-modified")):
                raise ContractError(f"{rid}: origin {origin} does not match layer {layer}")
            cell = (row["source_sha256"], row["start_ms"], row["end_ms"], row["tag_id"], row["playback_rate"])
            cid = row["cell_id"]
            if cell_ids.setdefault(cid, cell) != cell or exact_ids.setdefault(cell, cid) != cid:
                raise ContractError(f"{rid}: cell ID is not a bijection with the exact cell")
            cells.setdefault((layer, cell), []).append(row)
    records = []
    for (layer, _), rows in sorted(cells.items()):
        rows.sort(key=lambda r: r["record_id"])
        row = rows[0]
        labels = set()
        for member in rows:
            presence, salience = member["presence"], member["salience"]
            if presence == "present" and salience in ASSESSMENTS[1:]:
                labels.add(salience)
            elif presence in ("absent", "unresolved", "unreviewed") and salience is None:
                labels.add(presence)
            else:
                raise ContractError(f"{member['record_id']}: invalid presence/salience")
        ids = tuple(r["record_id"] for r in rows)
        if len(labels) != 1:
            issues.append({"kind": "conflicting-cell", "cell_id": row["cell_id"], "layer": layer, "record_ids": ids, "assessments": sorted(labels)})
            continue
        label = next(iter(labels))
        if label not in ASSESSMENTS:
            issues.append({"kind": "unsupervised-cell", "cell_id": row["cell_id"], "layer": layer, "record_ids": ids, "status": label})
            continue
        scope = Interval(row["start_ms"], row["end_ms"])
        context_value = row["details"]["review_context"]
        if context_value is None:
            raise ContractError(f"{ids}: missing declared review_context")
        context = Interval(**context_value)
        if not context.start_ms <= scope.start_ms < scope.end_ms <= context.end_ms:
            raise ContractError(f"{ids}: scope is outside review_context")
        value = row["details"].get("evidence")
        refs = value.get("note_refs") if value is not None else None
        evidence = None if refs is None else tuple(sorted(NoteRef(**ref) for ref in refs))
        if evidence is not None and len(evidence) != len(set(evidence)):
            raise ContractError(f"{ids}: duplicate evidence object")
        provenance = []
        for member in rows:
            details = member["details"]
            provenance.append({key: member.get(key) for key in (
                "record_id", "origin", "observation_id", "observation_sha256", "decision_id", "handoff_id", "claim_id", "provenance_id", "human_confidence", "auxiliary_evidence_status")})
            provenance[-1].update({key: details.get(key) for key in ("evidence_review", "proposal_changes", "human_revision")})
        records.append(Record(ids, row["cell_id"], row["source_sha256"], scope, context,
                              row["tag_id"], label, layer, tuple(r["origin"] for r in rows),
                              row["playback_rate"], row["foundation_id"], evidence, canonical_json(provenance)))
        if len(rows) > 1:
            issues.append({"kind": "agreeing-duplicate", "cell_id": row["cell_id"], "layer": layer,
                           "record_ids": ids, "representative_record_id": ids[0]})
    return tuple(records), issues


def split_manifest(sources: list[dict], *, known_groups: dict[str, list[str]] | None = None,
                   arrangement_hashes: dict[str, str] | None = None) -> dict:
    """Seed-0 80/10/10 grouping by source, known version/set, and supplied identities.

    Positive beatmap-set IDs conservatively group available shared-song metadata;
    they do not establish global song deduplication. Explicit song/audio/duplicate
    groups may join sources. Calibration exposure does not join split groups.
    """
    ids = sorted(s["source_sha256"] for s in sources)
    parent = dict(zip(ids, ids))

    def root(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def join(members: list[str]) -> None:
        if any(m not in parent for m in members):
            raise ContractError("Known split group references an unpublished source")
        if members:
            roots = sorted({root(m) for m in members})
            for r in roots:
                parent[r] = roots[0]

    links: dict[str, list[str]] = {}
    for s in sources:
        for field in ("beatmap_id", "beatmap_set_id"):
            if s.get(field) is not None and s[field] > 0:
                links.setdefault(f"{field}:{s[field]}", []).append(s["source_sha256"])
    for identity, members in (known_groups or {}).items():
        links[f"known:{identity}"] = sorted(set(members))
    for source, signature in (arrangement_hashes or {}).items():
        links.setdefault(f"arrangement:{signature}", []).append(source)
    for members in links.values():
        join(members)
    groups: dict[str, list[str]] = {}
    for source in ids:
        groups.setdefault(root(source), []).append(source)
    assignments = {}
    for group_id, members in sorted(groups.items()):
        # Integer comparison avoids float rounding at the interval endpoints.
        bucket = int(digest(f"0:{group_id}".encode()), 16)
        split = "train" if bucket * 10 < 8 * 2**256 else "validation" if bucket * 10 < 9 * 2**256 else "test"
        for source in members:
            assignments[source] = {"group_id": group_id, "split": split}
    body = {"version": 1, "dataset": DATASET, "revision": REVISION, "split_seed": 0,
            "algorithm": "sha256(utf8('0:' + lexicographically-smallest-source-sha256)); intervals 0:.8:.9:1",
            "grouping_links": {key: sorted(set(value)) for key, value in sorted(links.items()) if len(set(value)) > 1},
            "groups": groups, "sources": assignments}
    return {**body, "sha256": digest(canonical_json(body).encode())}


def freeze_manifest(path: Path, manifest: dict) -> None:
    """Write once, or verify exact equality; an existing split is never redrawn."""
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ContractError(f"{path}: frozen manifest differs; use an explicitly revised manifest path")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        output.write(canonical_json(manifest) + "\n")
