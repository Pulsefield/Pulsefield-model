"""Auditable data preparation, with frozen grouping and separate target artifacts."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, fields
import gzip
import json
from pathlib import Path
import sys
from time import monotonic

from .dataset import (
    ASSESSMENTS, CONCEPTS, DATASET, MANIFEST_SHA256, METHOD, REVISION, SPECIFICATION_SHA256,
    ContractError, adapt_records, canonical_json, digest, freeze_manifest, load_snapshot, split_manifest,
)
from .recovery import recover_sources
from .relations import prepare_relations
from .replay import evidence_masks, parse_source, prepare_chart, selected_objects


@dataclass
class PrepareConfig:
    dataset_dir: str = "artifacts/scoped-style-modeling/dataset-b22a7a4"
    source_cache: str = "artifacts/scoped-style-modeling/sources"
    output_dir: str = "artifacts/scoped-style-modeling/prepare-v1"
    split_manifest_path: str = "artifacts/scoped-style-modeling/split-v1.json"
    known_groups_path: str | None = None
    download: bool = False
    workers: int = 4
    timeout_seconds: float = 30


def _json(path: Path, value: object) -> None:
    path.write_text(canonical_json(value) + "\n")


def _chart_payload(prepared, edges) -> dict:
    # Repeated field names live once in each file; source identities are sidecars.
    from .replay import LaneFacts, Row
    from .relations import Edge, Relation
    row_fields = tuple(f.name for f in fields(Row))
    edge_fields = tuple(f.name for f in fields(Edge))
    lane_fields = tuple(f.name for f in fields(LaneFacts))
    relation_fields = tuple(f.name for f in fields(Relation))
    rows = [[getattr(row, key) if key != "lanes" else
             [[getattr(lane, name) for name in lane_fields] for lane in row.lanes]
             for key in row_fields] for row in prepared.inputs.rows]
    graph = [[getattr(edge, key) if key != "relations" else
              [[getattr(relation, name) for name in relation_fields] for relation in edge.relations]
              for key in edge_fields] for edge in edges]
    return {"schema_version": 1, "scope": asdict(prepared.inputs.scope), "context": asdict(prepared.inputs.context),
            "row_fields": row_fields, "lane_fields": lane_fields, "rows": rows,
            "edge_fields": edge_fields, "relation_fields": relation_fields, "edges": graph,
            "section_indices": prepared.inputs.section_indices, "event_indices": prepared.inputs.event_indices,
            "readout_elapsed_ms": prepared.inputs.readout_elapsed_ms,
            "visible_objects": [asdict(n) for n in prepared.visible_objects],
            "decisions": [{"encoder_index": d.encoder_index, "source_lines": [n.source_line if n else None for n in d.candidates],
                           "eligible_mask": d.eligible_mask, "valid_masks": d.valid_masks} for d in prepared.decisions]}


def load_chart(path: Path, *, expected_sha256: str):
    """Verify the cohort's chart hash, then load immutable inputs and identity sidecars."""
    from .dataset import Interval, NoteRef
    from .replay import ChartInputs, Decision, LaneFacts, PreparedChart, Row
    from .relations import Edge, Relation
    with gzip.open(path, "rb") as stream:
        payload = stream.read()
    if digest(payload) != expected_sha256:
        raise ContractError(f"{path}: prepared chart SHA-256 differs from cohort")
    data = json.loads(payload)
    if data["schema_version"] != 1:
        raise ContractError("Unsupported prepared-chart schema")
    rows = []
    for values in data["rows"]:
        row = dict(zip(data["row_fields"], values))
        row["lanes"] = tuple(LaneFacts(**dict(zip(data["lane_fields"], v))) for v in row["lanes"])
        row["markers"] = tuple(row["markers"])
        rows.append(Row(**row))
    objects = tuple(NoteRef(**n) for n in data["visible_objects"])
    by_line = {n.source_line: n for n in objects}
    decisions = tuple(Decision(d["encoder_index"], tuple(by_line[line] if line is not None else None for line in d["source_lines"])) for d in data["decisions"])
    inputs = ChartInputs(Interval(**data["scope"]), Interval(**data["context"]), tuple(rows),
                         tuple(data["section_indices"]), tuple(data["event_indices"]), tuple(data["readout_elapsed_ms"]))
    edges = []
    for values in data["edges"]:
        edge = dict(zip(data["edge_fields"], values))
        edge["relations"] = tuple(Relation(**dict(zip(data["relation_fields"], r))) for r in edge["relations"])
        edge["query_actions"] = tuple(edge["query_actions"])
        edge["neighbor_actions"] = tuple(edge["neighbor_actions"])
        edges.append(Edge(**edge))
    return PreparedChart(inputs, objects, decisions), tuple(edges)


def run_preparation(config: PrepareConfig) -> dict:
    """Prepare every resolved cell and report every recovery/contract failure.

    Output directories must be fresh. Failed evidence never removes an otherwise
    valid assessment: its explicit invalid status blocks readiness. Missing
    evidence retains the assessment with no auxiliary target. Source/replay
    failures are listed as excluded assessments for both arms, and block readiness
    until investigated. Existing split assignments must match exactly.
    """
    if config.workers < 1 or config.timeout_seconds <= 0:
        raise ContractError("workers and timeout_seconds must be positive")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _json(output / "config.json", asdict(config))
    started = monotonic()
    manifest, sources, tables = load_snapshot(Path(config.dataset_dir), download=config.download)
    records, issues = adapt_records(tables, sources)
    recovery = recover_sources(sources, Path(config.source_cache), download=config.download,
                               workers=config.workers, timeout=config.timeout_seconds)
    _json(output / "source-recovery.json", recovery)
    charts, source_errors, row_incompatibilities = {}, [], []
    source_metadata = {s["source_sha256"]: s for s in sources}
    for result in recovery:
        source = result["source_sha256"]
        if result["status"] != "verified":
            source_errors.append(result)
            continue
        try:
            chart = parse_source(Path(result["path"]).read_bytes(), source)
            if len(chart.objects) != source_metadata[source]["note_count"]:
                raise ContractError("Parsed object count differs from source table")
            charts[source] = chart
            if chart.row_incompatibilities:
                row_incompatibilities.append({"source_sha256": source, "kind": "same-lane-close-head",
                                              "source_line_pairs": chart.row_incompatibilities})
        except (ValueError, UnicodeError) as exc:
            source_errors.append({"source_sha256": source, "status": "invalid-replay", "error": str(exc)})
    known_groups = json.loads(Path(config.known_groups_path).read_text()) if config.known_groups_path else None
    split = split_manifest(sources, known_groups=known_groups,
                           arrangement_hashes={key: value.arrangement_sha256 for key, value in charts.items()})
    freeze_manifest(Path(config.split_manifest_path), split)
    _json(output / "split-manifest.json", split)
    _json(output / "adapter-issues.json", issues)
    _json(output / "row-incompatibilities.json", row_incompatibilities)
    contexts_dir = output / "contexts"
    contexts_dir.mkdir()
    context_keys = {}
    context_hashes = {}
    cohort = []
    excluded = []
    errors = []
    support: dict[tuple, dict] = {}
    rare = {}
    rare_class_support = {}
    exposure_sources = set()
    exposure_records = 0
    exposure_by_split = Counter()
    prepared_count = 0
    for record in records:
        assignment = split["sources"][record.source_sha256]
        base = {"cell_id": record.cell_id, "record_ids": record.record_ids, "layer": record.layer, **assignment}
        if record.source_sha256 not in charts:
            excluded.append({**base, "source_sha256": record.source_sha256, "reason": "source-recovery-or-replay-failed"})
            continue
        key = digest(canonical_json([record.source_sha256, asdict(record.scope), asdict(record.context)]).encode())
        # Scopes of one source are consecutive in the adapter output; only keep
        # this source's prepared contexts in memory while emitting all records.
        if context_keys and next(iter(context_keys.values()))[0] != record.source_sha256:
            context_keys.clear()
        if key not in context_keys:
            prepared = prepare_chart(charts[record.source_sha256].objects, record.scope, record.context)
            if key not in context_hashes:
                edges = prepare_relations(prepared)
                payload = canonical_json(_chart_payload(prepared, edges)).encode()
                context_hashes[key] = digest(payload)
                with (contexts_dir / f"{key}.json.gz").open("wb") as raw:
                    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as stream:
                        stream.write(payload)
                prepared_count += 1
                if prepared_count % 100 == 0:
                    print(f"Prepared {prepared_count} distinct source/scope/context inputs", flush=True)
            context_keys[key] = (record.source_sha256, prepared)
        prepared = context_keys[key][1]
        try:
            masks = evidence_masks(prepared, record.evidence)
            if masks is not None and selected_objects(prepared, masks) != tuple(sorted(record.evidence)):
                raise ContractError("Evidence object/mask round-trip mismatch")
            evidence_status = "missing" if masks is None else "empty" if not any(masks) else "available"
        except ContractError as exc:
            masks = None
            evidence_status = "invalid"
            errors.append({**base, "error": str(exc)})
        item = {**asdict(record), **assignment, "chart_key": key, "chart_sha256": context_hashes[key],
                "evidence_status": evidence_status, "evidence_masks": masks,
                "auxiliary_decisions": sum(d.eligible_mask != 0 for d in prepared.decisions)}
        cohort.append(item)
        bucket = (record.layer, assignment["split"], record.concept, record.assessment)
        slot = support.setdefault(bucket, {"records": 0, "groups": set(), "sources": set(), "evidence": Counter()})
        slot["records"] += 1
        slot["groups"].add(assignment["group_id"])
        slot["sources"].add(record.source_sha256)
        slot["evidence"][evidence_status] += 1
        flags = {"entering_ln": prepared.decisions[0].eligible_mask != 0,
                 "selected_entering_ln": bool(masks and masks[0]),
                 "partial_attack_group": False, "distributed_witness": False}
        if masks is not None:
            runs = 0
            previous_selected = False
            for d, mask in zip(prepared.decisions[1:], masks[1:]):
                if d.eligible_mask:
                    runs += bool(mask) and not previous_selected
                    previous_selected = bool(mask)
                    flags["partial_attack_group"] |= mask not in (0, d.eligible_mask)
            flags["distributed_witness"] = runs > 1
        for name, present in flags.items():
            if present:
                rare_key = (record.layer, assignment["split"], name)
                slot_rare = rare.setdefault(rare_key, {"records": 0, "groups": set()})
                slot_rare["records"] += 1
                slot_rare["groups"].add(assignment["group_id"])
                class_key = (*bucket, name)
                class_slot = rare_class_support.setdefault(class_key, {"records": 0, "groups": set()})
                class_slot["records"] += 1
                class_slot["groups"].add(assignment["group_id"])
        refs = []
        for p in json.loads(record.provenance_json):
            refs.extend(manifest["provenance"].get(p.get("provenance_id"), {}).get("human_evidence_refs", []))
        if refs:
            exposure_records += 1
            exposure_by_split[f"{record.layer}/{assignment['split']}"] += 1
            exposure_sources.update(ref["source_sha256"] for ref in refs)
    with (output / "assessment-cohort.jsonl").open("w") as stream:
        for item in cohort:
            stream.write(canonical_json(item) + "\n")
    _json(output / "excluded-assessments.json", excluded)
    _json(output / "evidence-errors.json", errors)
    # Emit all 90 support cells, including zeros, so absent strength classes show.
    support_rows = []
    for layer in ("machine", "human"):
        for partition in ("train", "validation", "test"):
            for concept in CONCEPTS:
                for assessment in ASSESSMENTS:
                    slot = support.get((layer, partition, concept, assessment), {"records": 0, "groups": set(), "sources": set(), "evidence": {}})
                    support_rows.append({"layer": layer, "split": partition, "concept": concept, "assessment": assessment,
                                         "records": slot["records"], "groups": len(slot["groups"]), "sources": len(slot["sources"]),
                                         "evidence": dict(slot["evidence"])})
    # A V3 single-action row cannot represent these coincidences. Preserve their
    # exact source facts in artifacts, but do not silently certify a training run.
    ready = not (source_errors or errors or row_incompatibilities or any(i["kind"] == "conflicting-cell" for i in issues))
    slice_names = ("entering_ln", "selected_entering_ln", "partial_attack_group", "distributed_witness")
    rare_rows = []
    for layer in ("machine", "human"):
        for partition in ("train", "validation", "test"):
            for name in slice_names:
                slot = rare.get((layer, partition, name), {"records": 0, "groups": set()})
                rare_rows.append({"layer": layer, "split": partition, "slice": name,
                                  "records": slot["records"], "groups": len(slot["groups"])})
    summary = {"schema_version": 1, "dataset": DATASET, "revision": REVISION, "manifest_sha256": MANIFEST_SHA256,
               "specification_sha256": SPECIFICATION_SHA256,
               "method": METHOD, "split_sha256": split["sha256"], "data_contract_ready": ready,
               "source_count": len(sources), "verified_source_count": sum(r["status"] == "verified" for r in recovery),
               "replayed_source_count": len(charts), "source_errors": source_errors,
               "row_incompatible_sources": len(row_incompatibilities),
               "input_records": {k: len(v) for k, v in tables.items()},
               "assessment_cells": dict(Counter(r["layer"] for r in cohort)),
               "excluded_assessment_cells": len(excluded), "invalid_evidence_cells": len(errors),
               "duplicate_records_collapsed": sum(len(i["record_ids"])-1 for i in issues if i["kind"] == "agreeing-duplicate"),
               "context_count": len(context_hashes), "source_group_count": len(split["groups"]),
               "group_counts": dict(Counter(split["sources"][members[0]]["split"] for members in split["groups"].values())),
               "support": support_rows,
               "missing_heldout_classes": [r for r in support_rows if r["split"] != "train" and r["records"] == 0],
               "rare_slices": rare_rows,
               "rare_class_support": [{"layer": k[0], "split": k[1], "concept": k[2], "assessment": k[3],
                                       "slice": k[4], "records": v["records"], "groups": len(v["groups"])}
                                      for k, v in sorted(rare_class_support.items())],
               "exposure": {"records_with_published_human_refs": exposure_records, "records_by_layer_split": dict(exposure_by_split),
                            "referenced_source_count": len(exposure_sources), "referenced_sources": sorted(exposure_sources),
                            "referenced_source_splits": dict(Counter(split["sources"][source]["split"] if source in split["sources"] else "unpublished" for source in exposure_sources)),
                            "calibration_sources": manifest["foundations"][records[0].foundation_id]["calibration_source_sha256"] if records else []},
               "cohort_sha256": digest((output / "assessment-cohort.jsonl").read_bytes()),
               "runtime_seconds": monotonic() - started, "command": sys.argv}
    _json(output / "summary.json", summary)
    return summary
