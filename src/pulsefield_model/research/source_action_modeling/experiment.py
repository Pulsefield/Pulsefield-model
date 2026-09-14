"""Complete bounded composition comparison on verified local source actions.

Fresh destinations only. Training arms share an endpoint per seed; evaluations
never select checkpoints. Checkpoints and a source diff preserve exploratory
work, but an uncommitted run is not clean-revision research evidence.
"""
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import gc
import platform
import random
import resource
import subprocess
from time import monotonic

import numpy as np
import torch

from ..scoped_style_modeling.dataset import ContractError, canonical_json, digest
from ..scoped_style_modeling.probe_metrics import paired_changes
from .checkpoint import save_snapshot
from .comparison import evaluate_structure, train_paired_step, parameter_counts
from .composition import initialize_composition, ORDERS
from .consistency import evaluate_path_consistency, prefix_path_pair
from .experiment_config import CompositionExperimentConfig
from .pilot import fixed_validation
from .full_corpus import load_population, export_allocation, FullCorpusSampler
from .observation import ViewPolicy
from .semantic_probe import fit_readout, evaluate_readout
from .structural_probe import evaluate_structural_reuse


def _rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if platform.system() == "Darwin" else peak * 1024


def run_experiment(config: CompositionExperimentConfig, *, resolved_yaml: str) -> dict:
    """Train every declared seed, then finish all three evaluation families.

    A per-seed time endpoint is valid after min_updates, with both arms at the
    same update. Exceeding a global/resource bound writes an incomplete report
    and raises. No test payloads, downloads, validation tuning or automatic
    resume are used. Source and resolved configuration are saved before data.
    """
    config.validate()
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise ContractError("The selected MPS device is unavailable")
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    started = monotonic()
    report = {"status": "running", "stage": "setup", "seeds": {}, "config": asdict(config),
              "started_utc": datetime.now(timezone.utc).isoformat(), "max_observed_driver_bytes": 0,
              "environment": {"torch": torch.__version__, "python": platform.python_version(),
                              "platform": platform.platform(), "device": config.device}}
    previous_threads = torch.get_num_threads()

    def write(path, value):
        path = root / path
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(canonical_json(value) + "\n")
        temp.replace(path)

    def guard():
        report["elapsed_seconds"] = monotonic() - started
        report["process_peak_rss_bytes"] = _rss_bytes()
        if config.device == "mps":
            report["max_observed_driver_bytes"] = max(report["max_observed_driver_bytes"], torch.mps.driver_allocated_memory())
        if report["elapsed_seconds"] >= config.max_seconds:
            raise TimeoutError("Global experiment wall-clock bound exhausted")
        if report["max_observed_driver_bytes"] > config.max_driver_bytes or _rss_bytes() > config.max_rss_bytes:
            raise MemoryError("Experiment MPS-driver or process-RSS bound exceeded")

    def progress(stage, **fields):
        report.update(stage=stage, elapsed_seconds=monotonic() - started, **fields)
        write("report.json", report)
        print(canonical_json({"stage": stage, "elapsed_seconds": report["elapsed_seconds"], **fields}), flush=True)
        if sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) > config.max_output_bytes:
            raise OSError("Experiment output storage bound exceeded")

    def snapshot(directory, name, model, optimizer, sampler, update, *, endpoint=False):
        path = directory / f"{name}-{'endpoint' if endpoint else 'latest'}.pt"
        temporary = path.with_suffix(".partial")
        save_snapshot(temporary, model, optimizer, sampler, update=update)
        temporary.replace(path)

    try:
        torch.set_num_threads(config.cpu_threads)
        if config.device == "mps":
            fraction = min(1.0, config.max_driver_bytes / torch.mps.recommended_max_memory())
            torch.mps.set_per_process_memory_fraction(fraction)
        (root / "resolved.yaml").write_text(resolved_yaml)
        write("runner.json", asdict(config))
        report["product_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        report["product_dirty"] = bool(status.strip())
        (root / "source-status.txt").write_text(status)
        (root / "source.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD", "--binary"]))
        # Git diff omits new files. Preserve module copies and hashes alongside
        # the resolved configuration saved above.
        sources = Path(__file__).parent
        source_dir = root / "source"
        source_dir.mkdir()
        hashes = {}
        for path in sorted(sources.glob("*.py")):
            payload = path.read_bytes()
            (source_dir / path.name).write_bytes(payload)
            hashes[path.name] = digest(payload)
        report["source_sha256"] = digest(canonical_json(hashes).encode())
        write("source/manifest.json", hashes)
        progress("population")
        training, validation, semantic, population = load_population(
            Path(config.prepared_dir), Path(config.dataset_dir), train_group_limit=config.train_group_limit,
            validation_groups=config.validation_groups, max_source_rows=config.max_source_rows,
            source_cache=Path(config.source_cache), source_cache_charts=config.source_cache_charts,
            index_path=Path(config.index_path), dataset_root=Path(config.dataset_root), guard=guard)
        export_allocation(root / "allocation", population, training,
                          protocol={"id": "source-action-composition-v2", **asdict(config)})
        write("population.json", population)
        report["population_counts"] = population["counts"]
        report["population_sha256"] = digest((root / "population.json").read_bytes())
        policy = ViewPolicy(config.near_radius)
        paired, records = fixed_validation(validation, policy)
        write("validation-blocks.json", records)
        report["validation_manifest_sha256"] = digest((root / "validation-blocks.json").read_bytes())
        paths = [prefix_path_pair(replace(p["detailed"],
                    observation=replace(p["detailed"].observation, summaries=()), query_entering_occupancy=None),
                    prefix_rows=max(1, len(p["detailed"].targets) // 2)) for p in paired[:config.path_pairs]]
        guard()
        for seed_index, seed in enumerate(config.seeds):
            seed_started = monotonic()
            directory = root / f"seed-{seed}"
            directory.mkdir()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            models = {n: m.to(config.device) for n, m in initialize_composition(config.model, seed).items()}
            optimizers = {n: torch.optim.AdamW(m.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
                          for n, m in models.items()}
            sampler = FullCorpusSampler(training, seed=seed, policy=policy)
            seed_report = {"updates": 0, "models": {n: {"parameters": parameter_counts(m), "policy": m.policy_identity}
                                                     for n, m in models.items()}}
            report["seeds"][str(seed)] = seed_report
            progress("initial-structure", seed=seed)
            write(f"seed-{seed}/structure-initial.json", evaluate_structure(models, paired, records,
                  batch_size=config.evaluation_batch_size, bootstrap_samples=config.bootstrap_samples, on_batch=guard))
            training_started = monotonic()
            checkpoint_time = training_started
            remaining_seeds = len(config.seeds) - seed_index
            remaining = config.max_seconds - (monotonic() - started)
            budget = min(config.training_seconds_per_seed,
                         remaining / remaining_seeds - config.finalization_seconds_per_seed)
            if budget <= 0:
                raise TimeoutError("Insufficient time for training and reserved finalization")
            seed_report.update(training_budget_seconds=budget, training_stop_reason="update-bound")
            with (directory / "updates.jsonl").open("x") as stream:
                for update in range(1, config.max_updates + 1):
                    guard()
                    if monotonic() - training_started >= budget:
                        seed_report["training_stop_reason"] = "common-time-bound"
                        break
                    result = train_paired_step(models, optimizers, sampler, blocks=config.blocks_per_update,
                                               gradient_cap=config.gradient_cap, configurations=tuple(ORDERS))
                    stream.write(canonical_json({"update": update, **result}) + "\n")
                    stream.flush()
                    seed_report["updates"] = update
                    seed_report["coverage"] = sampler.coverage()
                    if update == 1 or update % config.log_every == 0:
                        progress("pretraining", seed=seed, update=update,
                                 losses={n: r["loss"] for n, r in result["models"].items()})
                    if update % config.checkpoint_every == 0 or monotonic() - checkpoint_time >= config.checkpoint_seconds:
                        for name, model in models.items():
                            snapshot(directory, name, model, optimizers[name], sampler, update)
                        checkpoint_time = monotonic()
                        progress("checkpoint", seed=seed, update=update)
            seed_report["training_seconds"] = monotonic() - training_started
            seed_report["coverage"] = sampler.coverage()
            write(f"seed-{seed}/coverage.json", {**sampler.coverage(), "seen_sources": sorted(sampler.seen)})
            for name, model in models.items():
                snapshot(directory, name, model, optimizers[name], sampler, seed_report["updates"], endpoint=True)
            if seed_report["updates"] < config.min_updates:
                raise ContractError("Common endpoint did not reach min_updates")
            progress("final-structure", seed=seed)
            structure = evaluate_structure(models, paired, records, batch_size=config.evaluation_batch_size,
                                            bootstrap_samples=config.bootstrap_samples, on_batch=guard)
            write(f"seed-{seed}/structure-final.json", structure)
            seed_report["paired_gain_nats"] = structure["paired_detailed_mean_row_nll"]["serial_minus_interleaved"]
            human_order = {}
            for name, model in models.items():
                guard()
                progress("path-consistency", seed=seed, arm=name)
                write(f"seed-{seed}/{name}-paths.json", evaluate_path_consistency(model, paths,
                      batch_size=config.evaluation_batch_size, on_batch=guard))
                predictions = {}
                for condition in ("trained", "untrained"):
                    candidate = model if condition == "trained" else model.initialize_untrained(seed).to(config.device)
                    progress("structural-reuse", seed=seed, arm=name, condition=condition)
                    write(f"seed-{seed}/{name}-{condition}-relations.json", evaluate_structural_reuse(candidate,
                          batch_size=config.evaluation_batch_size, ridge=config.structural_probe_ridge, guard=guard))
                    progress("human-readout", seed=seed, arm=name, condition=condition)
                    def on_step(step, head, optimizer):
                        guard()
                        if step % config.log_every == 0:
                            progress("human-readout", seed=seed, arm=name, condition=condition, readout_step=step)
                    head, fit = fit_readout(candidate, semantic, steps=config.readout_steps,
                        batch_size=config.readout_batch_size, learning_rate=config.readout_learning_rate,
                        seed=seed, on_step=on_step)
                    guard()
                    evaluated = evaluate_readout(candidate, head, semantic, batch_size=config.readout_batch_size, on_batch=guard)
                    write(f"seed-{seed}/{name}-{condition}-human.json", {"fit": fit, **evaluated})
                    predictions[condition] = evaluated["predictions"]
                    if condition == "trained":
                        human_order[name] = evaluated["predictions"]
                    seed_report["models"][name][condition] = {"human_macro_nll": evaluated["metrics"]["macro_nll"],
                        "human_concept_nll": {c: r["nll"] for c, r in evaluated["metrics"]["concepts"].items()}}
                    torch.save({"state": head.state_dict(), "policy": head.policy_identity, "fit": fit},
                               directory / f"{name}-{condition}-readout.pt")
                    if condition == "untrained":
                        candidate.cpu()
                    del candidate, head
                    gc.collect()
                    if config.device == "mps":
                        torch.mps.empty_cache()
                write(f"seed-{seed}/{name}-human-reuse.json", paired_changes(predictions["untrained"], predictions["trained"]))
            write(f"seed-{seed}/human-order-comparison.json", paired_changes(human_order["serial"], human_order["interleaved"]))
            seed_report["elapsed_seconds"] = monotonic() - seed_started
            del models, optimizers, model
            gc.collect()
            if config.device == "mps":
                torch.mps.empty_cache()
            guard()
            progress("seed-completed", seed=seed)
        report["status"] = "completed"
        report["interpretation"] = "Prediction and frozen reuse measurements; semantic factorization and player demand are not established."
        progress("completed")
        return report
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="incomplete", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=monotonic() - started)
        write("report.json", report)
        raise
    finally:
        torch.set_num_threads(previous_threads)
        if config.device == "mps":
            torch.mps.set_per_process_memory_fraction(1.0)
