"""Paired updates and assessment-selected checkpoints; never opens test charts.

Each arm receives the same sampled batches and shared-component random stream.
Wall-clock exhaustion stops at a paired update barrier, with overshoot reported.
Early stopping remains per arm under the same assessment validation rule.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import resource
import shutil
import subprocess
import sys
from time import monotonic

import torch

from .config import TrainConfig
from .corpus import PreparedCorpus, sampled_epoch
from .dataset import SPECIFICATION_SHA256, ContractError, canonical_json, digest
from .metrics import assessment_report, evidence_report, record_metrics
from .model import initialize_model, losses


def write_json(path, value):
    Path(path).write_text(canonical_json(value)+"\n")


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def memory_snapshot(device):
    """Return process-wide counters after synchronization, in bytes.

    Process peak RSS is a high-water mark and cannot diagnose live retention.
    MPS driver/CUDA reserved memory includes cached capacity; these overlapping
    counters must not be summed with RSS or active device allocation.
    """
    synchronize(device)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"process_peak_rss_bytes": peak_rss if sys.platform == "darwin" else peak_rss*1024}
    if device == "mps":
        result.update(mps_active_bytes=torch.mps.current_allocated_memory(),
                      mps_driver_bytes=torch.mps.driver_allocated_memory())
    elif device == "cuda":
        result.update(cuda_active_bytes=torch.cuda.memory_allocated(),
                      cuda_reserved_bytes=torch.cuda.memory_reserved())
    return result


def log_memory(path, config, *, phase, epoch, paired_batches, arms):
    if not config.memory_log_every:
        return
    with Path(path).open("a") as stream:
        stream.write(canonical_json({"phase": phase, "epoch": epoch, "paired_batches": paired_batches,
                                    "arm_updates": {name: arm["updates"] for name, arm in arms.items()},
                                    **memory_snapshot(config.device)})+"\n")


def seed_update(seed, update):
    value = int(digest(f"scoped-style:{seed}:update:{update}".encode())[:15], 16)
    torch.manual_seed(value)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(value)


def _gradient_norm(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    squares = [g.detach().square().sum() for g in gradients if g is not None]
    return torch.stack(squares).sum().sqrt() if squares else loss.new_zeros(())


def update(model, optimizer, batch, beta, config, *, log_gradients=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    result = losses(model, batch, beta)
    if not torch.isfinite(result.total):
        raise ContractError("Nonfinite training loss; paired run stopped")
    logs = {"loss": result.total.item(), "assessment_nll": result.assessment.item(),
            "evidence_nll": result.evidence.item() if beta else None}
    if log_gradients:
        parameters = list(model.encoder.parameters())
        primary = _gradient_norm(result.assessment, parameters)
        auxiliary = _gradient_norm(result.evidence, parameters) if beta else primary.new_zeros(())
        logs.update(encoder_assessment_gradient_norm=primary.item(), encoder_evidence_gradient_norm=auxiliary.item(),
                    encoder_weighted_evidence_gradient_norm=(beta*auxiliary).item())
    result.total.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_cap, error_if_nonfinite=True)
    optimizer.step()
    logs["unclipped_gradient_norm"] = norm.item()
    return logs


@torch.no_grad()
def evaluate(model, corpus, indices, batch_size, device, *, auxiliary=False):
    model.eval()
    rows = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start+batch_size]
        if any(corpus.records[i]["split"] != "validation" for i in selected):
            raise ContractError("Training evaluation accepts validation records only")
        batch = corpus.batch(selected).to(device)
        h = model.encoder(batch.chart)
        logits = model.assessor(h, batch.chart, batch.concepts)
        diagnostics = record_metrics(logits.cpu(), batch.assessments.cpu())
        evidence = model.selector(h, batch.concepts, batch.assessments, batch.selector) if auxiliary else None
        for position, index in enumerate(selected):
            record = corpus.records[index]
            row = {k: record[k] for k in ("cell_id", "record_ids", "group_id", "source_sha256", "concept",
                                         "assessment", "layer", "origins", "provenance_json", "evidence_status")}
            row.update(diagnostics[position], logits=logits[position].cpu().tolist())
            if evidence is not None:
                eligible = bool(evidence.eligible[position])
                row.update(evidence_eligible=eligible,
                           evidence_nll=float(evidence.normalized_nll[position]) if eligible else None,
                           sequence_nll=float(evidence.sequence_nll[position]) if eligible else None)
            rows.append(row)
    report = assessment_report(rows)
    if auxiliary:
        report["evidence"] = evidence_report(rows)
    return rows, report


def source_record(output):
    """Save the executable research sources, frozen spec, config, tests and lock."""
    root = Path(__file__).resolve().parents[4]
    files = [*Path(__file__).parent.glob("*.py"),
             * (root/"tests/research/scoped_style_modeling").glob("*.py"),
             *(root/"src/pulsefield_model/configs/hydra").glob("scoped_style_*.yaml"),
             root/"docs/research/scoped_style_witness_generation.md", root/"pyproject.toml", root/"uv.lock"]
    hashes = {}
    for source in sorted(files):
        relative = source.relative_to(root)
        target = output/"source"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[str(relative)] = digest(source.read_bytes())
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if hashes["docs/research/scoped_style_witness_generation.md"] != SPECIFICATION_SHA256:
        raise ContractError("Study document differs from the frozen implementation specification")
    return {"git_revision": revision, "source_files": hashes,
            "source_sha256": digest(canonical_json(hashes).encode()), "torch_version": torch.__version__,
            "python_version": sys.version, "command": sys.argv}


def run_training(config: TrainConfig, *, resolved_yaml: str | None = None):
    config.validate()
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise ContractError("Requested MPS device is unavailable")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ContractError("Requested CUDA device is unavailable")
    torch.set_num_threads(config.cpu_threads)
    corpus = PreparedCorpus(Path(config.prepared_dir))
    validation = corpus.indices("machine", "validation")
    if not validation:
        raise ContractError("Assessment validation cohort is empty")
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"config.json", asdict(config))
    shutil.copyfile(corpus.root/"split-manifest.json", output/"split-manifest.json")
    shutil.copyfile(corpus.root/"summary.json", output/"preparation-summary.json")
    if resolved_yaml is not None:
        (output/"resolved-hydra.yaml").write_text(resolved_yaml)
    provenance = {**source_record(output), "cohort_sha256": corpus.summary["cohort_sha256"],
                  "split_sha256": corpus.summary["split_sha256"], "dataset_revision": corpus.summary["revision"],
                  "specification_sha256": SPECIFICATION_SHA256,
                  "preparation_specification_sha256": corpus.summary.get("specification_sha256"), "device": config.device,
                  "arm_budget_seconds": config.arm_budget_seconds, "max_updates": config.max_updates,
                  "assessment_support": corpus.summary["support"], "rare_slices": corpus.summary["rare_slices"]}
    write_json(output/"run.json", provenance)
    pairs = []
    try:
        for seed in config.seeds:
            pairs.append(_train_pair(config, corpus, validation, output, seed))
    except Exception as exc:
        write_json(output/"failure.json", {"type": type(exc).__name__, "message": str(exc), "completed_pairs": pairs})
        raise
    report = {"pairs": pairs, "test_evaluated": False,
              "interpretation": "Development validation only; no held-out test comparison or evidence-benefit claim."}
    write_json(output/"summary.json", report)
    return report


def _train_pair(config, corpus, validation, output, seed):
    directory = output/f"seed-{seed}"
    directory.mkdir()
    arms = {}
    for name, beta in (("style-only", 0.0), ("style+evidence", config.evidence_weight)):
        model = initialize_model(config.model, seed, auxiliary=bool(beta)).to(config.device)
        arms[name] = {"model": model, "optimizer": torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                      weight_decay=config.weight_decay), "beta": beta, "updates": 0, "draws": 0,
                      "seconds": 0.0, "best": float("inf"), "stale": 0, "active": True,
                      "best_epoch": None, "stop_reason": None}
        (directory/name).mkdir()
        write_json(directory/name/"run.json", {"seed": seed, "beta": beta,
                   "training_parameters": sum(p.numel() for p in model.parameters()),
                   "inference_parameters": sum(p.numel() for module in (model.encoder, model.assessor) for p in module.parameters())})
    streams, history = [], []
    paired_batches = 0
    log_memory(directory/"memory.jsonl", config, phase="initialized", epoch=0, paired_batches=paired_batches, arms=arms)
    reason = "max_epochs"
    stop = False
    for epoch in range(config.max_epochs):
        order = sampled_epoch(corpus.records, seed, epoch)
        streams.append({"epoch": epoch+1, "indices": order,
                        "cell_ids": [corpus.records[i]["cell_id"] for i in order]})
        write_json(directory/"minibatch-stream.json", streams)
        for start in range(0, len(order), config.batch_size):
            active = [arm for arm in arms.values() if arm["active"]]
            if any(arm["seconds"] >= config.arm_budget_seconds for arm in active):
                reason, stop = "wall_clock", True
                break
            if config.max_updates is not None and any(arm["updates"] >= config.max_updates for arm in active):
                reason, stop = "max_updates", True
                break
            selected = order[start:start+config.batch_size]
            started = monotonic()
            batch = corpus.batch(selected).to(config.device)
            synchronize(config.device)
            common_seconds = monotonic()-started
            for name, arm in arms.items():
                if not arm["active"]:
                    continue
                seed_update(seed, arm["updates"])
                started = monotonic()
                logs = update(arm["model"], arm["optimizer"], batch, arm["beta"], config,
                              log_gradients=arm["updates"] % config.gradient_log_every == 0)
                synchronize(config.device)
                arm["seconds"] += common_seconds+monotonic()-started
                arm["updates"] += 1
                arm["draws"] += len(selected)
                with (directory/name/"updates.jsonl").open("a") as stream:
                    stream.write(canonical_json({"epoch": epoch+1, "update": arm["updates"], "draws": arm["draws"],
                                                "charged_seconds": arm["seconds"], **logs})+"\n")
            paired_batches += 1
            if config.memory_log_every and paired_batches % config.memory_log_every == 0:
                log_memory(directory/"memory.jsonl", config, phase="paired_update", epoch=epoch+1,
                           paired_batches=paired_batches, arms=arms)
            if start == 0 or (start//config.batch_size+1) % 20 == 0:
                print(f"seed={seed} epoch={epoch+1} paired_batch={start//config.batch_size+1} "
                      + " ".join(f"{name}:updates={a['updates']},seconds={a['seconds']:.1f}" for name, a in arms.items()), flush=True)
        active = [a for a in arms.values() if a["active"]]
        if any(a["seconds"] >= config.arm_budget_seconds for a in active):
            reason, stop = "wall_clock", True
        elif config.max_updates is not None and any(a["updates"] >= config.max_updates for a in active):
            reason, stop = "max_updates", True
        for name, arm in arms.items():
            if not arm["active"]:
                continue
            started = monotonic()
            predictions, report = evaluate(arm["model"], corpus, validation, config.batch_size, config.device)
            synchronize(config.device)
            arm["seconds"] += monotonic()-started
            metric = report["macro_nll"]
            if metric is None:
                raise ContractError("Validation assessment macro-NLL unavailable")
            improved = metric < arm["best"]
            arm["stale"] = 0 if improved else arm["stale"]+1
            if improved:
                arm["best"], arm["best_epoch"] = metric, epoch+1
                torch.save({"model": arm["model"].state_dict(), "config": asdict(config.model), "beta": arm["beta"],
                            "seed": seed, "epoch": epoch+1, "updates": arm["updates"], "validation_macro_nll": metric,
                            "cohort_sha256": corpus.summary["cohort_sha256"], "split_sha256": corpus.summary["split_sha256"]},
                           directory/name/"best.pt")
                write_json(directory/name/"best-validation.json", {"assessment": report, "predictions": predictions})
            history.append({"arm": name, "epoch": epoch+1, "updates": arm["updates"],
                            "validation": report, "improved": improved, "charged_seconds": arm["seconds"]})
            if arm["stale"] >= config.patience:
                arm["active"], arm["stop_reason"] = False, "early_stopping"
            print(f"seed={seed} arm={name} epoch={epoch+1} validation_macro_nll={metric:.6f} best_epoch={arm['best_epoch']}", flush=True)
        write_json(directory/"validation-history.json", history)
        log_memory(directory/"memory.jsonl", config, phase="validation_complete", epoch=epoch+1,
                   paired_batches=paired_batches, arms=arms)
        if any(a["seconds"] >= config.arm_budget_seconds for a in arms.values() if a["active"]):
            reason, stop = "wall_clock", True
        if stop or not any(a["active"] for a in arms.values()):
            break
    reports = {}
    for name, arm in arms.items():
        # Complete validation can cross the budget on the final allowed epoch.
        if arm["seconds"] >= config.arm_budget_seconds and arm["stop_reason"] is None:
            reason = "wall_clock"
        checkpoint = torch.load(directory/name/"best.pt", map_location=config.device, weights_only=True)
        arm["model"].load_state_dict(checkpoint["model"])
        diagnostics_started = monotonic()
        layers = {}
        for layer in ("machine", "human"):
            predictions, report = evaluate(arm["model"], corpus, corpus.indices(layer, "validation"), config.batch_size,
                                           config.device, auxiliary=bool(arm["beta"]))
            layers[layer] = report
            write_json(directory/name/f"{layer}-validation.json", {"metrics": report, "predictions": predictions})
        synchronize(config.device)
        inference = {k: v.cpu() for k, v in arm["model"].state_dict().items() if not k.startswith("selector.")}
        torch.save({"model": inference, "config": asdict(config.model), "seed": seed,
                    "split_sha256": corpus.summary["split_sha256"], "cohort_sha256": corpus.summary["cohort_sha256"]},
                   directory/name/"assessment.pt")
        reports[name] = {"updates": arm["updates"], "draws": arm["draws"], "charged_seconds": arm["seconds"],
                         "diagnostic_seconds": monotonic()-diagnostics_started,
                         "budget_overshoot_seconds": max(0, arm["seconds"]-config.arm_budget_seconds),
                         "stop_reason": arm["stop_reason"] or reason, "best_epoch": arm["best_epoch"],
                         "validation": layers}
    pair = {"seed": seed, "arms": reports,
            "wall_clock_truncated": any(r["stop_reason"] == "wall_clock" for r in reports.values()),
            "validation_delta": reports["style-only"]["validation"]["machine"]["macro_nll"]-
                                reports["style+evidence"]["validation"]["machine"]["macro_nll"]}
    write_json(directory/"summary.json", pair)
    log_memory(directory/"memory.jsonl", config, phase="diagnostics_complete", epoch=epoch+1,
               paired_batches=paired_batches, arms=arms)
    return pair
