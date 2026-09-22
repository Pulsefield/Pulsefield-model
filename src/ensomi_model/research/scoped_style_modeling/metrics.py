"""Assessment validation with concept/group weighting and separate class tasks."""
from __future__ import annotations

from collections import defaultdict
import math

import torch

from .dataset import ASSESSMENTS, CONCEPTS, ContractError


def record_metrics(logits, labels):
    """Stable per-record decomposition, including reference positives predicted absent."""
    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if not torch.isfinite(logits).all():
        raise ContractError("Nonfinite assessment logits")
    logp = logits.log_softmax(-1)
    positive = torch.logsumexp(logp[:, 1:], -1)
    strength = logits[:, 1:].log_softmax(-1)
    nll = -logp.gather(1, labels[:, None]).squeeze(1)
    presence = -torch.where(labels > 0, positive, logp[:, 0])
    strength_nll = -strength.gather(1, (labels-1).clamp_min(0)[:, None]).squeeze(1)
    return [{"nll": float(nll[i]), "presence_nll": float(presence[i]),
             "strength_nll": float(strength_nll[i]) if labels[i] > 0 else None,
             "predicted": int(logits[i].argmax()), "presence_predicted": int(positive[i] >= math.log(0.5)),
             "strength_predicted": int(strength[i, 1] >= math.log(0.5))} for i in range(len(labels))]


def group_mean(rows, field):
    groups = defaultdict(list)
    for row in rows:
        if row[field] is not None:
            groups[row["group_id"]].append(row[field])
    return sum(sum(v)/len(v) for v in groups.values())/len(groups) if groups else None


def macro_mean(rows, field):
    values = [group_mean([r for r in rows if r["concept"] == concept], field) for concept in CONCEPTS]
    present = [v for v in values if v is not None]
    return sum(present)/len(present) if present else None


def _support(rows):
    return {"records": len(rows), "groups": len({r["group_id"] for r in rows})}


def _balanced(rows, task):
    recalls = []
    for label in (0, 1):
        members = []
        for row in rows:
            assessment = ASSESSMENTS.index(row["assessment"])
            if task == "strength" and assessment == 0:
                continue
            target = int(assessment > 0) if task == "presence" else assessment-1
            if target == label:
                members.append({**row, "correct": int(row[f"{task}_predicted"] == target)})
        recalls.append(group_mean(members, "correct"))
    return {"recall": recalls, "balanced_accuracy": sum(recalls)/2 if all(v is not None for v in recalls) else None}


def assessment_report(rows):
    """Report one origin layer only; cross-layer aggregates are rejected."""
    if len({r["layer"] for r in rows}) > 1:
        raise ContractError("Machine and human assessment metrics must remain separate")
    concepts = {}
    for concept in CONCEPTS:
        members = [r for r in rows if r["concept"] == concept]
        confusion = [[0]*3 for _ in range(3)]
        for row in members:
            confusion[ASSESSMENTS.index(row["assessment"])][row["predicted"]] += 1
        concepts[concept] = {**_support(members), "nll": group_mean(members, "nll"),
            "presence_nll": group_mean(members, "presence_nll"), "strength_nll": group_mean(members, "strength_nll"),
            "presence": _balanced(members, "presence"), "strength": _balanced(members, "strength"),
            "confusion_counts": confusion,
            "confusion_rates": [[v/sum(row) for v in row] if sum(row) else None for row in confusion],
            "classes": {label: {**_support(subset := [r for r in members if r["assessment"] == label]),
                                  "nll": group_mean(subset, "nll"),
                                  "presence_nll": group_mean(subset, "presence_nll"),
                                  "strength_nll": group_mean(subset, "strength_nll")} for label in ASSESSMENTS}}
    return {**_support(rows), "macro_nll": macro_mean(rows, "nll"),
            "macro_presence_nll": macro_mean(rows, "presence_nll"),
            "macro_strength_nll": macro_mean(rows, "strength_nll"), "concepts": concepts}


def evidence_report(rows):
    eligible = [r for r in rows if r["evidence_eligible"]]
    return {"conditioning": "reference assessment and teacher-forced reference masks",
            "eligible": _support(eligible), "total": _support(rows),
            "macro_normalized_nll": macro_mean(eligible, "evidence_nll"),
            "macro_sequence_nll": macro_mean(eligible, "sequence_nll"),
            "by_assessment": {label: {**_support(subset := [r for r in eligible if r["assessment"] == label]),
                                      "macro_normalized_nll": macro_mean(subset, "evidence_nll")}
                              for label in ASSESSMENTS}}
