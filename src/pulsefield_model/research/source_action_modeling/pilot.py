"""Bounded source-action access pilot on the pinned local annotation cohort.

This runner fixes exposure and endpoint selection, excludes known cross-split
song variants, and records paired structure, frozen reuse and update responses.
It is a Python API, with no downloads or resume/overwrite mode.
"""
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import random
import resource
import subprocess
from time import monotonic
import unicodedata

import numpy as np
import torch

from ..scoped_style_modeling.corpus import PreparedCorpus
from ..scoped_style_modeling.dataset import (CONCEPTS, ContractError, adapt_records,
    canonical_json, digest, load_snapshot)
from ..scoped_style_modeling.probe_data import input_identity, support
from ..scoped_style_modeling.probe_metrics import prediction_rows, views, paired_changes
from .checkpoint import save_snapshot
from .comparison import pretraining_contexts, evaluate_structure, train_paired_step
from .diagnostics import (capture_response, finish_response, capture_semantic_response,
                          finish_semantic_response)
from .model import initialize_comparison
from .local_corpus import load_local_corpus
from .observation import EventBlock, ViewPolicy, declared_entering_occupancy, observe, paired_views
from .sampling import TrainingContext, PairedBlockSampler, VIEWS
from .semantic_probe import SemanticCorpus, fit_readout, evaluate_readout
from .tensors import collate

PROTOCOL = {
    "id": "source-action-access-pilot-v1", "seed": 17, "updates": 3000,
    "blocks_per_update": 4, "near_radius": 8, "max_source_rows": 256,
    "learning_rate": 0.0003, "weight_decay": 0.0001, "gradient_cap": 1.0,
    "readout_steps": 400, "readout_batch_size": 8, "readout_learning_rate": 0.001,
    "evaluation_batch_size": 8, "bootstrap_samples": 2000,
    "likelihood_updates": [1, 100, 1000, 3000], "diagnostic_suffix_updates": 3,
    "max_seconds": 9000, "max_training_seconds": 6000, "finalization_reserve_seconds": 2400,
    "checkpoint_updates": 100, "checkpoint_seconds": 300,
    "local_train_groups": 1024, "local_validation_groups": 128,
    "max_driver_bytes": 8 * 1024**3, "max_output_bytes": 1024**3,
    "primary_gain_nats": 0.02, "semantic_macro_regression_nats": 0.05,
    "semantic_concept_regression_nats": 0.10, "semantic_reuse_gain_nats": 0.02,
}


def song_exclusions(sources, assignments):
    """Exclude whole training groups matching held-out normalized artist/title.

    NFKC, case folding and whitespace folding are the only normalization. This
    catches exact metadata variants and does not claim audio deduplication.
    """
    songs = defaultdict(list)
    for source in sources:
        key = tuple(" ".join(unicodedata.normalize("NFKC", source[k] or "").casefold().split())
                    for k in ("artist", "title"))
        if all(key):
            songs[key].append(source["source_sha256"])
    overlaps = [members for members in songs.values()
                if len({assignments[s]["split"] for s in members}) > 1]
    excluded = sorted({assignments[s]["group_id"] for members in overlaps for s in members
                       if assignments[s]["split"] == "train"})
    return excluded, {"method": "NFKC/casefold/whitespace artist+title",
                      "cross_split_source_sets": sorted(sorted(s) for s in overlaps),
                      "excluded_training_groups": excluded, "audio_deduplication": "not-established"}


def fixed_validation(contexts, policy, *, seed=17):
    """Two distinct starts per feasible scale/group, with one hash-selected context.

    Selection uses only group/input identities and supplied event counts. Every
    retained group contributes at least one block; target actions are not read
    to select the context, scale or start.
    """
    groups = defaultdict(list)
    for context in contexts:
        groups[context.group_id].append(context)
    paired, records = [], []
    for group, candidates in sorted(groups.items()):
        for size in (4, 16, 64):
            eligible = [c for c in candidates if c.event_count >= size]
            if not eligible:
                continue
            key = lambda c: digest(f"validation:{seed}:{group}:{size}:{c.key}".encode())
            context = min(eligible, key=key)
            rng = random.Random(f"validation:{seed}:{group}:{size}:starts")
            starts = sorted(rng.sample(range(context.event_count - size + 1),
                                      min(2, context.event_count - size + 1)))
            for start in starts:
                example = observe(context.chart, EventBlock(start, size),
                                  entering_occupancy=declared_entering_occupancy(context.chart))
                observation = example.observation
                paired.append(paired_views(example, policy))
                records.append({"group_id": group, "context_key": context.key,
                    "event_start": start, "rows": size, "attack_group_span": example.attack_group_span,
                    "duration_ms": observation.rows[observation.target_indices[-1]].time_ms
                                   - observation.rows[observation.target_indices[0]].time_ms,
                    "views": list(VIEWS), "view_policy": asdict(policy)})
    return paired, records


def load_population(prepared_dir, dataset_dir, *, allocation_dir=None):
    """Verify local inputs and preserve original review scopes and split assignments.

    Prediction uses all remaining 4..256-row training contexts. Human probes use
    every eligible original scope, including scopes beyond the prediction cap.
    Test chart payloads are never opened.
    """
    corpus = PreparedCorpus(Path(prepared_dir))
    manifest, sources, tables = load_snapshot(Path(dataset_dir))
    _, issues = adapt_records(tables, sources)
    if any(value for key, value in manifest["exclusions"]["human"].items() if key != "source-excluded"):
        raise ContractError("Human publication exclusions require an exact-cell adapter")
    assignments = json.loads((corpus.root / "split-manifest.json").read_text())["sources"]
    local, excluded, local_report = load_local_corpus(sources, assignments,
        train_groups=PROTOCOL["local_train_groups"], validation_groups=PROTOCOL["local_validation_groups"])
    _, song_audit = song_exclusions(sources, assignments)
    song_audit["excluded_training_groups"] = excluded
    song_audit["additional_grouping"] = "Transitive published group, beatmap ID and set ID links through the local index"
    selected, validation, rejected = set(), [], []
    identities = {input_identity(r): r for r in corpus.records if r["split"] in ("train", "validation")}
    for key, row in sorted(identities.items()):
        if row["split"] == "train" and row["group_id"] in excluded:
            rejected.append({"input_id": key, "reason": "heldout-song-metadata-match"})
            continue
        chart, _ = corpus.chart(row["chart_key"], row["chart_sha256"])
        context = TrainingContext(row["group_id"], key, chart)
        if not 4 <= context.event_count <= PROTOCOL["max_source_rows"]:
            rejected.append({"input_id": key, "reason": "source-event-count", "rows": context.event_count})
        elif row["split"] == "train":
            selected.add(key)
        else:
            validation.append(context)
    training, population = pretraining_contexts(corpus, input_keys=selected, excluded_groups=excluded)
    semantic = SemanticCorpus(corpus, issues)
    semantic.train_indices = [i for i in semantic.train_indices if corpus.records[i]["group_id"] not in excluded]
    semantic.eligible = set(semantic.train_indices + semantic.validation_indices)
    if set(corpus.records[i]["concept"] for i in semantic.train_indices) != set(CONCEPTS):
        raise ContractError("Song exclusions removed training support for a concept")
    if {c.group_id for c in training} & {c.group_id for c in validation}:
        raise ContractError("Prediction populations overlap")
    # Materialize original semantic inputs now, so a missing chart cannot turn
    # the final probe into an unrecorded subset after pretraining has finished.
    for i in semantic.train_indices + semantic.validation_indices:
        row = corpus.records[i]
        corpus.chart(row["chart_key"], row["chart_sha256"])
    population.update(song_audit=song_audit, excluded_inputs=rejected,
        semantic_support=support(corpus.records, sorted(semantic.eligible)),
        semantic_train_indices=semantic.train_indices, semantic_validation_indices=semantic.validation_indices,
        validation_inputs=[{"input_id": c.key, "group_id": c.group_id, "rows": c.event_count,
                            **{k: identities[c.key][k] for k in ("source_sha256", "scope", "context", "chart_key", "chart_sha256")}}
                           for c in validation],
        counts={"training_contexts": len(training), "training_groups": len({c.group_id for c in training}),
                "validation_contexts": len(validation), "validation_groups": len({c.group_id for c in validation})},
        cohort_sha256=corpus.summary["cohort_sha256"], test_chart_payloads_opened=False)
    for context in local["train"]:
        training.append(context)
    validation.extend(local["validation"])
    population["local_corpus"] = local_report
    population["counts"].update(total_training_contexts=len(training), total_training_groups=len({c.group_id for c in training}),
        total_validation_contexts=len(validation), total_validation_groups=len({c.group_id for c in validation}))
    if allocation_dir is not None:
        export_allocation(Path(allocation_dir), population)
    return training, validation, semantic, population


def export_allocation(directory, population):
    """Write immutable Parquet source assignments and exact reusable window rows."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    directory.mkdir(parents=True, exist_ok=False)
    local = population["local_corpus"]
    assignments = local.pop("source_assignments")
    pq.write_table(pa.Table.from_pylist(assignments), directory / "source-assignments.parquet", compression="zstd")
    windows = []
    def row(item, split, origin, path=None):
        return {"input_id": item["input_id"], "group_id": item["group_id"], "split": split,
                "origin": origin, "source_sha256": item["source_sha256"], "source_path": path,
                "scope_start_ms": item["scope"]["start_ms"], "scope_end_ms": item["scope"]["end_ms"],
                "context_start_ms": item["context"]["start_ms"], "context_end_ms": item["context"]["end_ms"],
                "playback_rate": 1.0, "chart_key": item.get("chart_key"), "chart_sha256": item.get("chart_sha256"),
                "source_event_rows": item.get("event_count", item.get("rows"))}
    for split, key in (("train", "inputs"), ("validation", "validation_inputs")):
        windows.extend(row(item, split, "annotation-context") for item in population[key])
    for item in local["selected"]:
        windows.extend(row(window, item["split"], "local-corpus", item["path"]) for window in item["windows"])
    pq.write_table(pa.Table.from_pylist(windows), directory / "windows.parquet", compression="zstd")
    manifest = {"schema": "source-action-allocation-v1", "protocol": PROTOCOL,
                "annotation_revision": population["dataset_revision"], "annotation_split_sha256": population["split_sha256"],
                "local_index_sha256": local["index_sha256"], "counts": population["counts"],
                "source_assignments_rows": len(assignments), "window_rows": len(windows),
                "files": {p.name: digest(p.read_bytes()) for p in sorted(directory.iterdir())}}
    (directory / "manifest.json").write_text(canonical_json(manifest) + "\n")


def _fine_contributions(before, model):
    contributions = defaultdict(float)
    for name, parameter in model.named_parameters():
        parts = name.split(".")
        group = ".".join(parts[:3] if name.startswith(("encoder.local.", "encoder.reference.")) else parts[:2])
        delta = parameter.detach().cpu().double() - before.parameters[name].double()
        contributions[group] += float((before.gradients[name].double() * delta).sum())
    return dict(contributions)


def _training_predictions(model, head, semantic, indices, batch_size):
    predictions = []
    mode = model.encoder.training
    model.encoder.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                selected = indices[start:start + batch_size]
                batch = semantic.batch(selected).to(next(model.parameters()).device)
                logits = head(model.encoder(batch.observation), batch.concepts, batch.input_indices)
                predictions.extend(prediction_rows([semantic.records[i] for i in selected], logits.cpu()))
    finally:
        model.encoder.train(mode)
    return {"metrics": views(predictions), "predictions": predictions}


def _diagnostic_cells(semantic):
    chosen = {}
    for concept in CONCEPTS:
        selected = []
        for positive in (False, True):
            candidates = sorted((i for i in semantic.train_indices if semantic.records[i]["concept"] == concept
                                 and (semantic.records[i]["assessment"] != "absent") == positive),
                                key=lambda i: semantic.records[i]["cell_id"])
            groups = set()
            for i in candidates:
                group = semantic.records[i]["group_id"]
                if group not in groups:
                    selected.append(i)
                    groups.add(group)
                if len(groups) == 2:
                    break
        if not any(semantic.records[i]["assessment"] == "absent" for i in selected) or not any(
            semantic.records[i]["assessment"] != "absent" for i in selected
        ):
            raise ContractError("Every semantic diagnostic needs positive and negative training targets")
        chosen[concept] = selected
    return chosen


def run_pilot(output_dir: Path, *, device: str = "mps") -> dict:
    """Run one seed-17 pilot; write partial evidence on failure or exhausted bounds.

    Requires clean Git state and the fixed local inputs. The common endpoint
    is saved before probes and a separately identified three-update diagnostic
    suffix. Stop at 3,000 updates or the declared common time bound. No
    validation-dependent checkpoint or budget selection is performed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = monotonic()
    report = {"status": "running", "protocol": PROTOCOL, "device": device, "updates": 0,
              "diagnostics": [], "started_utc": datetime.now(timezone.utc).isoformat(),
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "platform": platform.platform(), "machine": platform.machine()},
              "maximum_observed_driver_bytes": 0}

    def write(name, value):
        (output_dir / name).write_text(canonical_json(value) + "\n")

    def progress(stage, **values):
        report.update(stage=stage, elapsed_seconds=monotonic() - started, **values)
        report["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if device == "mps":
            report["maximum_observed_driver_bytes"] = max(report["maximum_observed_driver_bytes"],
                                                          torch.mps.driver_allocated_memory())
        write("report.json", report)
        print(canonical_json({k: report[k] for k in ("stage", "updates", "elapsed_seconds")}), flush=True)

    def guard():
        if monotonic() - started > PROTOCOL["max_seconds"]:
            raise TimeoutError("Pilot wall-clock bound exhausted")
        if device == "mps" and torch.mps.driver_allocated_memory() > PROTOCOL["max_driver_bytes"]:
            raise MemoryError("Pilot MPS driver allocation bound exceeded")
        if sum(p.stat().st_size for p in output_dir.rglob("*") if p.is_file()) > PROTOCOL["max_output_bytes"]:
            raise OSError("Pilot output storage bound exceeded")

    previous_threads = torch.get_num_threads()
    try:
        if device != "mps" or not torch.backends.mps.is_available():
            raise ContractError("This fixed pilot requires the selected MPS device")
        report["product_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        report["product_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        if report["product_dirty"]:
            raise ContractError("Commit the intervention and runner before executing the pilot")
        torch.set_num_threads(1)
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        progress("population")
        training, validation, semantic, population = load_population(
            Path("artifacts/scoped-style-modeling/prepare-v1"),
            Path("artifacts/scoped-style-modeling/dataset-b22a7a4"), allocation_dir=output_dir / "dataset")
        write("population.json", population)
        report["population_sha256"] = digest((output_dir / "population.json").read_bytes())
        report["population_counts"] = population["counts"]
        policy = ViewPolicy(PROTOCOL["near_radius"])
        sampler = PairedBlockSampler(training, seed=17, policy=policy)
        paired, records = fixed_validation(validation, policy)
        write("validation-blocks.json", records)
        report["validation_blocks_sha256"] = digest((output_dir / "validation-blocks.json").read_bytes())
        models = {name: model.to(device) for name, model in initialize_comparison(seed=17).items()}
        optimizers = {name: torch.optim.AdamW(model.parameters(), lr=PROTOCOL["learning_rate"],
                                            weight_decay=PROTOCOL["weight_decay"]) for name, model in models.items()}
        fixed_pairs, fixed_records = PairedBlockSampler(training, seed=17017, policy=policy).draw(2)
        fixed = collate([p["detailed"] for p in fixed_pairs]).to(device)
        write("diagnostic-blocks.json", fixed_records)
        guard()
        progress("initial-structure")
        structure = evaluate_structure(models, paired, records, batch_size=PROTOCOL["evaluation_batch_size"],
                                       bootstrap_samples=PROTOCOL["bootstrap_samples"])
        write("structure-initial.json", structure)
        guard()
        training_started = monotonic()
        checkpoint_time = monotonic()
        report["training_stop_reason"] = "update-bound"
        with (output_dir / "updates.jsonl").open("x") as stream:
            for update in range(1, PROTOCOL["updates"] + 1):
                guard()
                if (monotonic() - training_started >= PROTOCOL["max_training_seconds"] or
                    monotonic() - started >= PROTOCOL["max_seconds"] - PROTOCOL["finalization_reserve_seconds"]):
                    report["training_stop_reason"] = "common-time-bound"
                    break
                before = {name: capture_response(model, fixed) for name, model in models.items()} if (
                    update in PROTOCOL["likelihood_updates"]) else {}
                result = train_paired_step(models, optimizers, sampler, blocks=PROTOCOL["blocks_per_update"],
                                          gradient_cap=PROTOCOL["gradient_cap"])
                stream.write(canonical_json({"update": update, **result}) + "\n")
                stream.flush()
                report["updates"] = update
                for name, capture in before.items():
                    report["diagnostics"].append({"update": update, "model": name, "response": "block-log-likelihood",
                        **finish_response(capture, models[name], fixed),
                        "fine_module_contributions": _fine_contributions(capture, models[name])})
                if update == 1 or update % 25 == 0:
                    progress("pretraining", last_training_losses={n: r["loss"] for n, r in result["models"].items()})
                if update % PROTOCOL["checkpoint_updates"] == 0 or monotonic() - checkpoint_time >= PROTOCOL["checkpoint_seconds"]:
                    for name, model in models.items():
                        save_snapshot(output_dir / f"{name}-update-{update:05d}.pt", model, optimizers[name], sampler, update=update)
                    checkpoint_time = monotonic()
                    progress("checkpoint")
        report["training_seconds"] = monotonic() - training_started
        endpoint = report["updates"]
        if endpoint < 100:
            raise ContractError("Fewer than 100 paired updates cannot complete this pilot")
        for name, model in models.items():
            save_snapshot(output_dir / f"{name}-endpoint.pt", model, optimizers[name], sampler, update=endpoint)
        guard()
        progress("final-structure")
        structure = evaluate_structure(models, paired, records, batch_size=PROTOCOL["evaluation_batch_size"],
                                       bootstrap_samples=PROTOCOL["bootstrap_samples"])
        write("structure-final.json", structure)
        report["primary"] = structure["paired_detailed_nll"]["composed_h_minus_composed_all"]
        heads, semantic_summary = {}, {}
        untrained = initialize_comparison(seed=17)
        for name, model in models.items():
            predictions, fitted = {}, {}
            for condition, candidate in (("trained", model), ("untrained", untrained[name].to(device))):
                guard()
                progress(f"probe-{name}-{condition}")
                def on_step(step, head, optimizer):
                    guard()
                    if step % 100 == 0:
                        with (output_dir / f"probe-{name}-{condition}-{step:04d}.pt").open("xb") as stream:
                            torch.save({"readout": head.state_dict(), "optimizer": optimizer.state_dict(),
                                "step": step, "seed": 17, "protocol": PROTOCOL, "endpoint_update": endpoint,
                                "model": name, "condition": condition, "population_sha256": report["population_sha256"]}, stream)
                        progress(f"probe-{name}-{condition}-{step}")
                head, fit = fit_readout(candidate, semantic, steps=PROTOCOL["readout_steps"],
                    batch_size=PROTOCOL["readout_batch_size"], learning_rate=PROTOCOL["readout_learning_rate"], seed=17,
                    on_step=on_step)
                evaluated = evaluate_readout(candidate, head, semantic, batch_size=PROTOCOL["readout_batch_size"])
                fitted[condition] = fit
                predictions[condition] = evaluated["predictions"]
                training_result = _training_predictions(candidate, head, semantic, semantic.train_indices,
                                                        PROTOCOL["readout_batch_size"])
                write(f"semantic-{name}-{condition}.json", {"fit": fit, **evaluated, "training": training_result})
                semantic_summary.setdefault(name, {})[condition] = {"validation_macro_nll": evaluated["metrics"]["macro_nll"],
                    "validation_concept_nll": {c: r["nll"] for c, r in evaluated["metrics"]["concepts"].items()},
                    "training_macro_nll": training_result["metrics"]["macro_nll"]}
                with (output_dir / f"readout-{name}-{condition}.pt").open("xb") as stream:
                    torch.save({"state": head.state_dict(), "policy": head.policy_identity, "fitted": head.fitted}, stream)
                if condition == "trained":
                    heads[name] = head
                else:
                    candidate.cpu()
            if fitted["trained"]["sampled_indices"] != fitted["untrained"]["sampled_indices"]:
                raise ContractError("Matched frozen probe exposures differ")
            write(f"semantic-{name}-paired.json", paired_changes(predictions["untrained"], predictions["trained"]))
        report["semantic"] = semantic_summary
        guard()
        progress("semantic-update-diagnostics")
        cells = _diagnostic_cells(semantic)
        write("diagnostic-human-cells.json", cells)
        semantic_before, semantic_batches = {}, {}
        for concept, indices in cells.items():
            semantic_batches[concept] = semantic.batch(indices).to(device)
            for name, model in models.items():
                semantic_before[name, concept] = capture_semantic_response(model, heads[name], semantic_batches[concept])
        likelihood_before = {name: capture_response(model, fixed) for name, model in models.items()}
        for suffix in range(1, PROTOCOL["diagnostic_suffix_updates"] + 1):
            guard()
            result = train_paired_step(models, optimizers, sampler, blocks=PROTOCOL["blocks_per_update"])
            write(f"diagnostic-update-{suffix}.json", result)
            if suffix in (1, 3):
                for name, model in models.items():
                    report["diagnostics"].append({"update_window": [endpoint, endpoint + suffix], "model": name,
                        "response": "block-log-likelihood", **finish_response(likelihood_before[name], model, fixed),
                        "fine_module_contributions": _fine_contributions(likelihood_before[name], model)})
                    for concept, batch in semantic_batches.items():
                        capture = semantic_before[name, concept]
                        report["diagnostics"].append({"update_window": [endpoint, endpoint + suffix], "model": name,
                            "concept": concept, **finish_semantic_response(capture, model, heads[name], batch),
                            "fine_module_contributions": _fine_contributions(capture.response,
                                torch.nn.ModuleDict({"encoder": model.encoder}))})
        for name, model in models.items():
            save_snapshot(output_dir / f"{name}-diagnostic-suffix.pt", model, optimizers[name], sampler,
                          update=endpoint + PROTOCOL["diagnostic_suffix_updates"], readout=heads[name])
        report["status"], report["stop_reason"] = "completed", "fixed-update-and-probe-bounds"
        progress("completed")
        return report
    except Exception as exc:
        report.update(status="incomplete" if isinstance(exc, TimeoutError) else "failed",
                      error=f"{type(exc).__name__}: {exc}", stop_reason="guard-or-execution-error")
        progress("stopped")
        raise
    finally:
        torch.set_num_threads(previous_threads)
