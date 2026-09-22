"""Frozen linear readouts of exact synthetic relations, with factorial holdouts.

Programmed action statistics are diagnostic targets, never human style labels.
No synthetic example or target updates the source-action encoder or decoder.
"""
from itertools import product
import math

import torch

from ..scoped_style_modeling.dataset import Interval, NoteRef, canonical_json, digest
from ..scoped_style_modeling.replay import prepare_chart
from .observation import declared_entering_occupancy, observe_complete
from .tensors import collate_observations
from .representation import valid_rows

FACTS = ("adjacent_lane_recurrence", "two_step_return_without_repeat", "lag16_lane_match", "adjacent_hand_transition")
AXES = ("run_expansion", "motif_reordering", "lane_to_hand_assignment")


def structural_cases():
    """64 tap rows, equal lane histograms and constant gaps within every case.

    Fit on even-parity factor combinations at 72/120/200 ms. Odd parity at those
    gaps holds out combinations; 90/160 ms holds out pace for all combinations.
    Counterfactual pairs at held-out paces toggle one generator setting. Actual
    changes in all four statistics are reported, including coupled changes.
    """
    result = []
    for bits, gap, phase, mirror in product(list(product((0, 1), repeat=3)), (72, 120, 200, 90, 160), range(4), (False, True)):
        runs, reorder, assignment = bits
        expansion = 1 + runs
        symbols = []
        motif = (0, 1, 0, 1, 2, 3, 2, 3)
        for cycle in range(8 // expansion):
            permutation = (0, 2, 1, 3) if reorder and cycle % 2 else (0, 1, 2, 3)
            symbols.extend(permutation[lane] for lane in motif)
        lanes = [lane for lane in symbols for _ in range(expansion)]
        mapping = (0, 2, 1, 3) if assignment else (0, 1, 2, 3)
        lanes = [mapping[lane] for lane in lanes]
        lanes = lanes[phase:] + lanes[:phase]
        if mirror:
            lanes = [3 - lane for lane in lanes]
        facts = [sum(a == b for a, b in zip(lanes, lanes[1:])) / 63,
                 sum(a == c and a != b for a, b, c in zip(lanes, lanes[1:], lanes[2:])) / 62,
                 sum(a == b for a, b in zip(lanes, lanes[16:])) / 48,
                 sum(a // 2 != b // 2 for a, b in zip(lanes, lanes[1:])) / 63]
        split = "pace_holdout" if gap in (90, 160) else "fit" if sum(bits) % 2 == 0 else "combination_holdout"
        objects = tuple(NoteRef(i + 1, lane, "normal", i * gap, i * gap) for i, lane in enumerate(lanes))
        interval = Interval(0, len(lanes) * gap)
        chart = prepare_chart(objects, interval, interval)
        identity = {"bits": bits, "gap_ms": gap, "phase": phase, "mirror": mirror}
        result.append({**identity, "id": digest(canonical_json(identity).encode()), "split": split,
                       "lanes": lanes, "facts": facts,
                       "observation": observe_complete(chart, entering_occupancy=declared_entering_occupancy(chart))})
    return result


def _features(model, cases, batch_size, guard):
    matrices = {}
    device = next(model.parameters()).device
    mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(cases), batch_size):
                if guard:
                    guard()
                obs = collate_observations([c["observation"] for c in cases[start:start + batch_size]]).to(device)
                bank = model.encoder(obs)
                anchors = valid_rows(obs) & obs.rows[..., 5].bool()
                mask = anchors[:, :, None, None]
                denominator = anchors.sum(1)[:, None] * 2
                for level, state in zip(bank.levels, bank.states):
                    mean = (state * mask).sum((1, 2)) / denominator
                    second = (state.square() * mask).sum((1, 2)) / denominator
                    matrices.setdefault(level, []).append(torch.cat((mean, second), -1).cpu().double())
    finally:
        model.train(mode)
    matrices = {name: torch.cat(values) for name, values in matrices.items()}
    # Identical orderless statistics within each pace; this control cannot read
    # action order. The linear probe receives the same 64-row extent.
    matrices["orderless_source"] = torch.tensor([[c["lanes"].count(i) / 64 for i in range(4)] +
        [c["gap_ms"] / 1000, math.log(c["gap_ms"]), 64] for c in cases], dtype=torch.float64)
    return matrices


def evaluate_structural_reuse(model, *, batch_size=8, ridge=0.1, guard=None):
    """Fit CPU ridge probes on frozen stage moments, evaluate exact held-out facts.

    Ridge and features are fixed before labels are read. Each stage has its own
    readout. Delta errors and unchanged-target leakage use fresh-pace pairs;
    they establish readout response, not a causal neural subspace alignment.
    """
    if batch_size < 1 or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("Structural probes require positive batch size and ridge")
    cases = structural_cases()
    target = torch.tensor([c["facts"] for c in cases], dtype=torch.float64)
    fit = torch.tensor([c["split"] == "fit" for c in cases])
    matrices = _features(model, cases, batch_size, guard)
    pairs = []
    by_identity = {(tuple(c["bits"]), c["gap_ms"], c["phase"], c["mirror"]): i for i, c in enumerate(cases)}
    for i, case in enumerate(cases):
        if case["split"] != "pace_holdout":
            continue
        for axis, bit in enumerate(case["bits"]):
            if bit:
                continue
            changed = list(case["bits"])
            changed[axis] = 1
            j = by_identity[tuple(changed), case["gap_ms"], case["phase"], case["mirror"]]
            pairs.append((axis, i, j))
    reports = {}
    for name, x in matrices.items():
        mean, scale = x[fit].mean(0), x[fit].std(0).clamp_min(1e-6)
        x = torch.cat(((x - mean) / scale, torch.ones((len(x), 1), dtype=x.dtype)), 1)
        penalty = torch.eye(x.shape[1], dtype=x.dtype) * ridge
        penalty[-1, -1] = 0
        weights = torch.linalg.solve(x[fit].T @ x[fit] / int(fit.sum()) + penalty,
                                     x[fit].T @ target[fit] / int(fit.sum()))
        predicted = x @ weights
        split_report = {}
        for split in ("fit", "combination_holdout", "pace_holdout"):
            selected = torch.tensor([c["split"] == split for c in cases])
            mse = (predicted[selected] - target[selected]).square().mean(0)
            prior = (target[selected] - target[fit].mean(0)).square().mean(0)
            split_report[split] = {"cases": int(selected.sum()), "mse": dict(zip(FACTS, mse.tolist())),
                "training_mean_prior_mse": dict(zip(FACTS, prior.tolist()))}
        changes = {}
        for axis, label in enumerate(AXES):
            indices = [(i, j) for a, i, j in pairs if a == axis]
            truth = torch.stack([target[j] - target[i] for i, j in indices])
            prediction = torch.stack([predicted[j] - predicted[i] for i, j in indices])
            changes[label] = {"pairs": len(indices), "facts": {}}
            for k, fact in enumerate(FACTS):
                changed = truth[:, k].abs() > 1e-9
                changes[label]["facts"][fact] = {
                    "true_mean_abs_delta": float(truth[:, k].abs().mean()),
                    "delta_mse": float((truth[:, k] - prediction[:, k]).square().mean()),
                    "changed_pairs": int(changed.sum()),
                    "direction_accuracy": float((truth[changed, k] * prediction[changed, k] > 0).double().mean()) if changed.any() else None,
                    "unchanged_mean_abs_predicted_delta": float(prediction[~changed, k].abs().mean()) if (~changed).any() else None}
        reports[name] = {"features": x.shape[1] - 1, "splits": split_report, "counterfactuals": changes}
    manifest = [{k: v for k, v in c.items() if k != "observation"} for c in cases]
    return {"policy": "factorial-taps/frozen-ridge/64-rows-v1", "facts": FACTS, "ridge": ridge,
            "manifest_sha256": digest(canonical_json(manifest).encode()), "cases": manifest,
            "factorization_claim": False, "human_style_labels": False, "stages": reports}
