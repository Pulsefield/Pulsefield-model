"""Verified prepared cohorts and deterministic concept/group/record sampling."""
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import random

from .dataset import (ASSESSMENTS, CONCEPTS, REVISION, MANIFEST_SHA256, METHOD, SPECIFICATION_SHA256, ContractError,
                      NoteRef, canonical_json, checked_bytes, digest)
from .prepare import load_chart
from .replay import evidence_masks
from .tensors import Example, collate


class PreparedCorpus:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.summary = json.loads((self.root/"summary.json").read_text())
        if self.summary["data_contract_ready"] is not True:
            raise ContractError("Prepared data_contract_ready=false; repair reported data errors before training")
        if (self.summary["revision"] != REVISION or self.summary["manifest_sha256"] != MANIFEST_SHA256 or
                self.summary["method"] != METHOD or
                self.summary.get("specification_sha256") not in (None, SPECIFICATION_SHA256)):
            raise ContractError("Prepared dataset/specification differs from frozen study")
        split = json.loads((self.root/"split-manifest.json").read_text())
        actual = digest(canonical_json({k: v for k, v in split.items() if k != "sha256"}).encode())
        if actual != split["sha256"] or actual != self.summary["split_sha256"]:
            raise ContractError("Frozen split manifest hash mismatch")
        payload = checked_bytes(self.root/"assessment-cohort.jsonl", self.summary["cohort_sha256"])
        self.records = [json.loads(line) for line in payload.splitlines()]
        seen = set()
        for row in self.records:
            key = (row["layer"], row["cell_id"])
            if key in seen:
                raise ContractError(f"Duplicate evaluation cell {key}")
            seen.add(key)
            assignment = split["sources"][row["source_sha256"]]
            if any(row[k] != assignment[k] for k in ("split", "group_id")):
                raise ContractError("Cohort assignment differs from frozen split")
            if row["evidence_status"] not in ("missing", "empty", "available"):
                raise ContractError("Invalid evidence is a data error, not unavailable supervision")
            if row["concept"] not in CONCEPTS or row["assessment"] not in ASSESSMENTS:
                raise ContractError("Cohort has an unsupported concept or unresolved assessment")
        # Bound cached source graphs per corpus, rather than retaining every context.
        self.chart = lru_cache(maxsize=32)(self._chart)

    def _chart(self, key, sha):
        return load_chart(self.root/"contexts"/f"{key}.json.gz", expected_sha256=sha)

    def example(self, index):
        row = self.records[index]
        chart, edges = self.chart(row["chart_key"], row["chart_sha256"])
        evidence = None if row["evidence"] is None else tuple(NoteRef(**n) for n in row["evidence"])
        masks = evidence_masks(chart, evidence)
        stored = None if row["evidence_masks"] is None else tuple(row["evidence_masks"])
        status = "missing" if masks is None else "available" if any(masks) else "empty"
        if masks != stored or status != row["evidence_status"]:
            raise ContractError(f"{row['cell_id']}: evidence identities, status and masks disagree")
        return Example(chart, edges, CONCEPTS.index(row["concept"]), ASSESSMENTS.index(row["assessment"]), masks)

    def indices(self, layer, split):
        return [i for i, row in enumerate(self.records) if row["layer"] == layer and row["split"] == split]

    def batch(self, indices):
        return collate([self.example(i) for i in indices])


def sampled_epoch(records, seed, epoch):
    """Draw N training records with uniform concept, group, then record selection."""
    groups = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(records):
        if row["layer"] == "machine" and row["split"] == "train":
            groups[row["concept"]][row["group_id"]].append(index)
    if not groups:
        raise ContractError("No machine training records")
    if set(groups) != set(CONCEPTS):
        raise ContractError("Training requires support for every frozen concept")
    rng = random.Random(f"scoped-style:{seed}:epoch:{epoch}")
    group_ids = {concept: sorted(groups[concept]) for concept in CONCEPTS}
    count = sum(len(indices) for concept in groups.values() for indices in concept.values())
    result = []
    for _ in range(count):
        concept = rng.choice(CONCEPTS)
        group = rng.choice(group_ids[concept])
        result.append(rng.choice(groups[concept][group]))
    return result
