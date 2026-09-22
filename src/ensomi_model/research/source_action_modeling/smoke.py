"""Offline, bounded Stage 1 wiring check on eight pinned training contexts."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import platform
import random
import subprocess
from time import monotonic

import numpy as np
import torch

from ..scoped_style_modeling.dataset import (ContractError, Interval, REVISION, MANIFEST_SHA256,
                                           canonical_json, checked_bytes, digest)
from ..scoped_style_modeling.replay import prepare_chart
from .actions import parse_source
from .checkpoint import save_snapshot
from .diagnostics import capture_response, finish_response
from .model import initialize_model
from .observation import INPUT_CONTRACT, EventBlock, declared_entering_occupancy, observe
from .sampling import BlockSampler, SAMPLING_POLICY, SPLIT_SHA256, TrainingContext
from .tensors import collate

MAX_CONTEXTS, MAX_ROWS, MAX_UPDATES, MAX_SECONDS = 8, 128, 20, 120


def select_contexts(records: list[dict], assignments: dict) -> list[dict]:
    """Select one context per first eight sorted training groups, ignoring labels."""
    candidates = {}
    for row in records:
        assignment = assignments[row["source_sha256"]]
        if any(row[k] != assignment[k] for k in ("split", "group_id")):
            raise ContractError("Context assignment differs from the pinned split")
        if assignment["split"] != "train":
            continue
        item = {k: row[k] for k in ("group_id", "chart_key", "source_sha256", "scope", "context")}
        group = item["group_id"]
        if group not in candidates or item["chart_key"] < candidates[group]["chart_key"]:
            candidates[group] = item
    selected = [candidates[g] for g in sorted(candidates)[:MAX_CONTEXTS]]
    if len(selected) != MAX_CONTEXTS:
        raise ContractError("The bounded real-input check requires eight distinct training groups")
    return selected


def load_contexts(prepared_dir: Path, source_cache: Path):
    """Verify pinned metadata and original local bytes; never download or read held-out charts.

    Each selected review context supplies its first at most 128 real event
    positions. The capped interval is both task scope and context. Original
    annotation scope/context identities remain in the diagnostic sidecar only.
    """
    summary = json.loads((prepared_dir / "summary.json").read_text())
    if (summary["revision"], summary["manifest_sha256"], summary["split_sha256"], summary["data_contract_ready"]) != (
        REVISION, MANIFEST_SHA256, SPLIT_SHA256, True
    ):
        raise ContractError("Prepared dataset revision, manifest, split or readiness mismatch")
    split = json.loads((prepared_dir / "split-manifest.json").read_text())
    actual = digest(canonical_json({k: v for k, v in split.items() if k != "sha256"}).encode())
    if actual != SPLIT_SHA256 or split["sha256"] != actual:
        raise ContractError("Pinned split SHA-256 mismatch")
    payload = checked_bytes(prepared_dir / "assessment-cohort.jsonl", summary["cohort_sha256"])
    selected = select_contexts([json.loads(line) for line in payload.splitlines()], split["sources"])
    contexts, identities = [], []
    for item in selected:
        source = parse_source((source_cache / f"{item['source_sha256']}.osu").read_bytes(), item["source_sha256"])
        expected_key = digest(canonical_json([item["source_sha256"], item["scope"], item["context"]]).encode())
        if expected_key != item["chart_key"]:
            raise ContractError("Prepared source/scope/context identity mismatch")
        original = Interval(**item["context"])
        times = sorted({t for n in source.objects for t in (n.start_ms, n.end_ms) if original.contains(t)})
        if len(times) < 4:
            raise ContractError("Selected training context has fewer than four real source rows")
        used = Interval(original.start_ms, times[MAX_ROWS] if len(times) > MAX_ROWS else original.end_ms)
        chart = prepare_chart(source.objects, used, used)
        key = digest(canonical_json([item["chart_key"], asdict(used), INPUT_CONTRACT]).encode())
        context = TrainingContext(item["group_id"], key, chart)
        if not 4 <= context.event_count <= MAX_ROWS:
            raise ContractError("Bounded source conversion violated the event-row limit")
        contexts.append(context)
        identities.append({**item, "task_context_key": key, "task_scope": asdict(used), "task_context": asdict(used),
                           "source_event_rows": context.event_count, "context_prefix_capped": len(times) > MAX_ROWS})
    return contexts, identities


def _metrics(model, batch):
    mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = model(batch)
        later = batch.queries.steps.clone()
        later[:, 0] = False
        return {"mean_block_row_nll": float(output.loss), "first_row_nll": float(output.row_nll[:, 0].mean()),
                "later_row_nll": float(output.row_nll[later].mean()),
                "per_block_row_nll": output.mean_row_nll.cpu().tolist(),
                "per_block_sequence_nll": output.sequence_nll.cpu().tolist()}
    finally:
        model.train(mode)


def run_smoke(prepared_dir: Path, source_cache: Path, output_dir: Path, *, device: str = "mps") -> dict:
    """Create a fresh run, stopping at 20 updates or 120 seconds including diagnostics.

    Fixed seed 17, at most eight blocks per update, no validation/test inputs,
    no downloads and no resume path. Contract failures stop immediately and are
    recorded in report.json. A time-limited incomplete report is wiring evidence
    only; it must not be presented as successful bounded verification.
    """
    started = monotonic()
    deadline = started + MAX_SECONDS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    if device not in ("cpu", "mps", "cuda"):
        raise ContractError("An explicit cpu, mps or cuda device is required")
    report = {"status": "running", "input_contract": INPUT_CONTRACT, "dataset_revision": REVISION,
              "split_sha256": SPLIT_SHA256, "seed": 17, "device": device, "sampling_policy": SAMPLING_POLICY,
              "bounds": {"contexts": MAX_CONTEXTS, "source_event_rows": MAX_ROWS, "updates": MAX_UPDATES,
                         "seconds_including_diagnostics": MAX_SECONDS, "logical_batch_size": 8},
              "updates": [], "diagnostics": [], "exposure": [],
              "environment": {"python": platform.python_version(), "torch": torch.__version__}}

    def write_report():
        report["elapsed_seconds"] = monotonic() - started
        (output_dir / "report.json").write_text(canonical_json(report) + "\n")

    def check_time():
        if monotonic() >= deadline:
            raise TimeoutError("Stage 1 real-input time bound reached")

    previous_threads = torch.get_num_threads()
    try:
        report["product_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        report["product_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        contexts, identities = load_contexts(Path(prepared_dir), Path(source_cache))
        (output_dir / "contexts.json").write_text(canonical_json(identities) + "\n")
        check_time()
        torch.set_num_threads(1)
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        model = initialize_model().to(device)
        report["model_config"] = asdict(model.config)
        report["parameters"] = {name: sum(p.numel() for p in module.parameters())
                                for name, module in (("encoder", model.encoder), ("decoder", model.decoder))}
        sampler = BlockSampler(contexts, seed=17)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
        report["optimizer"] = {"name": "AdamW", "learning_rate": 0.0003, "weight_decay": 0.0001, "gradient_cap": 1.0}
        fixed_items, fixed_records = BlockSampler(contexts, seed=17).draw(8)
        fixed = collate(fixed_items).to(device)
        report["fixed_blocks"] = fixed_records
        diagnostic = collate([observe(contexts[0].chart, EventBlock(0, 4),
                                     entering_occupancy=declared_entering_occupancy(contexts[0].chart))]).to(device)
        initial_parameters = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
        before_metrics = monotonic()
        report["initial"] = _metrics(model, fixed)
        metrics_cost = monotonic() - before_metrics
        step_cost = 0.0
        for update in range(1, MAX_UPDATES + 1):
            check_time()
            # Reserve a measured final evaluation and snapshot margin before
            # admitting another step. No additional update is a throughput probe.
            if deadline - monotonic() < max(2.0, 1.5 * step_cost + 1.5 * metrics_cost + 1.0):
                report["stop_reason"] = "time_admission_bound"
                break
            step_started = monotonic()
            items, exposure = sampler.draw(8)
            batch = collate(items).to(device)
            before = capture_response(model, diagnostic) if update in (1, 2, 3) else None
            check_time()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            check_time()
            prediction.loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            check_time()
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ContractError("Nonfinite parameter after optimizer update")
            report["updates"].append({"update": update, "loss": float(prediction.loss.detach()),
                                      "gradient_norm_before_clip": float(norm)})
            report["exposure"].extend(exposure)
            if before is not None:
                check_time()
                report["diagnostics"].append({"update": update, **finish_response(before, model, diagnostic)})
            step_cost = monotonic() - step_started
            write_report()
        else:
            report["stop_reason"] = "update_bound"
        check_time()
        report["final"] = _metrics(model, fixed)
        report["parameter_displacement_l2"] = {}
        for group in ("encoder", "decoder"):
            squared = sum(float((p.detach().cpu().double() - initial_parameters[n].double()).square().sum())
                          for n, p in model.named_parameters() if n.startswith(group + "."))
            report["parameter_displacement_l2"][group] = squared ** 0.5
        check_time()
        save_snapshot(output_dir / "final.pt", model, optimizer, sampler, update=len(report["updates"]))
        check_time()
        report["status"] = "completed" if len(report["diagnostics"]) == 3 else "incomplete"
        write_report()
        return report
    except TimeoutError as exc:
        report.update(status="incomplete", stop_reason="time_bound", error=str(exc))
        write_report()
        return report
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_report()
        raise
    finally:
        torch.set_num_threads(previous_threads)
