"""Paired source-action updates, group uncertainty and fixed-population evaluation.

These Python interfaces do not choose a research budget or success threshold.
Population, update horizon, practical gain and semantic guards belong in a
bounded Experiment Card before a research run.
"""
from collections import defaultdict
from dataclasses import asdict, replace
import math
import random
from time import perf_counter

import torch

from ..scoped_style_modeling.dataset import REVISION, ContractError
from ..scoped_style_modeling.probe_data import input_identity
from .diagnostics import batch_identity
from .model import CONFIGURATIONS, LOSS_POLICY
from .observation import BlockExample, ViewPolicy, paired_views, visible_states
from .sampling import SPLIT_SHA256, VIEWS, TrainingContext
from .tensors import collate


def pretraining_contexts(corpus, *, input_keys: set[str], excluded_sources=(), excluded_groups=()):
    """Load an explicitly selected, deduplicated source/scope/context/rate population.

    All held-out source/group identities in the verified cohort are excluded,
    along with caller-declared related variants. This does not establish complete
    song/audio deduplication. Concept records only locate original inputs.
    """
    if (corpus.summary["revision"], corpus.summary["split_sha256"]) != (REVISION, SPLIT_SHA256):
        raise ContractError("Pretraining requires the pinned annotation revision and split")
    blocked_sources, blocked_groups = set(excluded_sources), set(excluded_groups)
    for row in corpus.records:
        if row["split"] != "train":
            blocked_sources.add(row["source_sha256"])
            blocked_groups.add(row["group_id"])
    selected = {}
    for row in corpus.records:
        key = input_identity(row)
        if key not in input_keys:
            continue
        if row["split"] != "train" or row["source_sha256"] in blocked_sources or row["group_id"] in blocked_groups:
            raise ContractError("Selected pretraining input overlaps an excluded source or group")
        if row["playback_rate"] != 1:
            raise ContractError("The frozen original-input adapter supports only recorded 1x inputs")
        identity = {k: row[k] for k in ("source_sha256", "scope", "context", "playback_rate", "group_id", "chart_key", "chart_sha256")}
        if key in selected and selected[key] != identity:
            raise ContractError("Duplicate input identity has inconsistent source or group metadata")
        selected[key] = identity
    if not input_keys or set(selected) != set(input_keys):
        raise ContractError("Every selected pretraining input must exist in the training cohort")
    contexts, manifest = [], []
    for key, item in sorted(selected.items()):
        chart, _ = corpus.chart(item["chart_key"], item["chart_sha256"])
        if (asdict(chart.inputs.scope), asdict(chart.inputs.context)) != (item["scope"], item["context"]):
            raise ContractError("Pretraining input scope/context differs from the selected source")
        context = TrainingContext(item["group_id"], key, chart)
        if context.event_count < 4:
            raise ContractError("Selected pretraining context has fewer than four source events")
        contexts.append(context)
        manifest.append({"input_id": key, **item, "event_count": context.event_count})
    return contexts, {"dataset_revision": REVISION, "split_sha256": SPLIT_SHA256, "inputs": manifest,
                       "excluded_sources": sorted(blocked_sources), "excluded_groups": sorted(blocked_groups),
                       "song_audio_deduplication": "not-established"}


def paired_batches(paired: list[dict[str, BlockExample]]):
    """Validate equal target/skeleton/prefix conditions before any model update."""
    if not paired or any(set(p) != set(VIEWS) for p in paired):
        raise ContractError("Every paired block requires near, detailed and coarse views")
    for item in paired:
        near = item["near"]
        a = near.observation
        if a.summaries or a.rows != item["coarse"].observation.rows:
            raise ContractError("Near/coarse must share visibility; near has no far summaries")
        if near.query_entering_occupancy is not None and near.query_entering_occupancy != visible_states(a)[2]:
            raise ContractError("Common decoder entry must come from the near view's permitted prefix")
        for view in VIEWS:
            other = item[view]
            b = other.observation
            if (a.scope, a.context, a.entering_occupancy, a.target_indices,
                [(r.time_ms, r.phase, r.markers) for r in a.rows], near.targets) != (
                b.scope, b.context, b.entering_occupancy, b.target_indices,
                [(r.time_ms, r.phase, r.markers) for r in b.rows], other.targets
            ):
                raise ContractError("Paired target values or supplied skeleton differ")
        if item["detailed"].observation.summaries != item["coarse"].observation.summaries:
            raise ContractError("Detailed and coarse must receive identical far summaries")
    batches = {view: collate([p[view] for p in paired]) for view in VIEWS}
    near = batches["near"]
    for batch in batches.values():
        if any(not torch.equal(getattr(near.queries, name), getattr(batch.queries, name))
               for name in ("indices", "steps", "entering_occupancy")):
            raise ContractError("Paired decoder query/prefix conditions differ")
    return batches


def _synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def parameter_counts(model) -> dict:
    return {name: sum(p.numel() for p in getattr(model, name).parameters()) for name in ("encoder", "reader", "decoder")}


def train_paired_step(models, optimizers, sampler, *, blocks: int, gradient_cap: float = 1.0,
                      configurations: tuple[str, ...] = CONFIGURATIONS) -> dict:
    """One common target/view exposure and one optimizer step per configuration.

    All three views participate equally in each update. Models must use the same
    device and optimizer settings; caller-owned schedulers advance outside this
    function. Returned manifests include exact tensor/target identities.
    """
    if not configurations or len(set(configurations)) != len(configurations) or (
        set(models) != set(configurations) or set(optimizers) != set(models)
    ):
        raise ContractError("The comparison requires exactly its named models and their optimizers")
    if not math.isfinite(gradient_cap) or gradient_cap <= 0:
        raise ContractError("Gradient cap must be positive and finite")
    devices = {next(model.parameters()).device for model in models.values()}
    configs = [model.config for model in models.values()]
    settings = [{k: v for k, v in opt.param_groups[0].items() if k != "params"} for opt in optimizers.values()]
    if len(devices) != 1 or any(c != configs[0] for c in configs) or any(s != settings[0] for s in settings) or (
        len({type(opt) for opt in optimizers.values()}) != 1
    ):
        raise ContractError("Paired model dimensions, device or optimizer settings differ")
    for name, optimizer in optimizers.items():
        if len(optimizer.param_groups) != 1 or {id(p) for p in optimizer.param_groups[0]["params"]} != {
            id(p) for p in models[name].parameters()
        } or models[name].configuration != name:
            raise ContractError("Each comparison optimizer must own exactly its named model")
    paired, records = sampler.draw(blocks)
    separate = paired_batches(paired)
    # Each block has the same three-view multiplicity, preserving block weighting.
    batch = collate([p[view] for p in paired for view in VIEWS]).to(next(iter(devices)))
    result = {"loss_policy": LOSS_POLICY, "sampling_policy": sampler.sampling_policy, "view_weight": 1 / len(VIEWS),
              "blocks": records, "view_sha256": {view: batch_identity(b) for view, b in separate.items()}, "models": {}}
    for name in configurations:
        model, optimizer = models[name], optimizers[name]
        model.train()
        device = next(model.parameters()).device
        _synchronize(device)
        started = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        output.loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_cap, error_if_nonfinite=True)
        optimizer.step()
        _synchronize(device)
        result["models"][name] = {"loss": float(output.loss.detach()), "gradient_norm": float(norm),
                                  "update_seconds": perf_counter() - started, "parameters": parameter_counts(model)}
    return result


def _estimate(rows, field, *, seed, bootstrap_samples):
    groups = defaultdict(list)
    for row in rows:
        if row.get(field) is not None:
            groups[row["group_id"]].append(row[field])
    means = {g: sum(v) / len(v) for g, v in sorted(groups.items())}
    values = list(means.values())
    interval = None
    if len(values) >= 2 and bootstrap_samples:
        rng = random.Random(seed)
        draws = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(bootstrap_samples))
        interval = [draws[int((len(draws) - 1) * q)] for q in (.025, .975)]
    return {"mean": sum(values) / len(values) if values else None, "group_means": means, "groups": len(values),
            "group_bootstrap_95": interval}


def structural_report(rows: list[dict], *, seed: int = 17, bootstrap_samples: int = 1000) -> dict:
    """Group-macro raw NLL and within-block context contrasts, by scale and duration.

    Uncertainty resamples source-group means of already paired differences. A
    single group has no interval; absent scales and later rows remain unavailable.
    """
    if type(bootstrap_samples) is not int or bootstrap_samples < 0:
        raise ContractError("Bootstrap samples must be a nonnegative integer")
    responses = ("sequence_nll", "mean_row_nll", "first_nll", "later_nll")
    fields = [f"{view}_{response}" for view in VIEWS for response in responses]
    fields += [f"{view}_minus_detailed_{response}" for view in ("near", "coarse") for response in responses]
    def summarize(items):
        return {"blocks": len(items), **{f: _estimate(items, f, seed=seed, bootstrap_samples=bootstrap_samples) for f in fields}}
    durations = (("under-250ms", 0, 250), ("250ms-1s", 250, 1000), ("1-4s", 1000, 4000), ("4s-plus", 4000, math.inf))
    positions = defaultdict(list)
    for row in rows:
        for position in range(row["rows"]):
            item = {"group_id": row["group_id"]}
            for view in VIEWS:
                losses = row.get(f"{view}_row_nll")
                if losses is not None:
                    item[f"{view}_nll"] = losses[position]
            for view in ("near", "coarse"):
                if f"{view}_nll" in item and "detailed_nll" in item:
                    item[f"{view}_minus_detailed_nll"] = item[f"{view}_nll"] - item["detailed_nll"]
            positions[position].append(item)
    position_fields = [f"{v}_nll" for v in VIEWS] + [f"{v}_minus_detailed_nll" for v in ("near", "coarse")]
    return {"aggregation": "mean-within-group/mean-over-represented-groups", "nll_unit": "nats",
            "overall": summarize(rows), "by_scale": {str(size): summarize([r for r in rows if r["rows"] == size])
                                                       for size in (4, 16, 64)},
            "by_duration": {name: summarize([r for r in rows if lo <= r["duration_ms"] < hi]) for name, lo, hi in durations},
            "by_decoder_position": {str(position): {f: _estimate(items, f, seed=seed, bootstrap_samples=0)
                                                    for f in position_fields}
                                    for position, items in sorted(positions.items())},
            "blocks": rows}


def evaluate_structure(models, paired, records, *, batch_size: int = 8, bootstrap_samples: int = 1000, seed: int = 17,
                       on_batch=None):
    """Evaluate a fixed paired manifest in bounded batches; restore model modes.

    Optional on_batch() may interrupt evaluation by raising, with mode restoration.
    """
    if len(paired) != len(records) or not paired or type(batch_size) is not int or batch_size < 1:
        raise ContractError("Structural evaluation requires aligned nonempty blocks/records and a positive batch size")
    for item, record in zip(paired, records):
        obs = item["near"].observation
        duration = obs.rows[obs.target_indices[-1]].time_ms - obs.rows[obs.target_indices[0]].time_ms
        if record["rows"] != len(obs.target_indices) or record["duration_ms"] != duration or record["views"] != list(VIEWS):
            raise ContractError("Structural reporting manifest differs from evaluated target blocks")
        detailed = item["detailed"]
        original = replace(detailed, observation=replace(detailed.observation, summaries=()), query_entering_occupancy=None)
        if paired_views(original, ViewPolicy(**record["view_policy"])) != item:
            raise ContractError("Evaluated views differ from the recorded visibility/summary policy")
    result = {}
    for name, model in models.items():
        rows = [dict(r) for r in records]
        mode, device = model.training, next(model.parameters()).device
        model.eval()
        _synchronize(device)
        started = perf_counter()
        identities, bank_bytes = [], 0
        try:
            with torch.no_grad():
                for start in range(0, len(paired), batch_size):
                    if on_batch is not None:
                        on_batch()
                    batches = paired_batches(paired[start:start + batch_size])
                    identities.append({view: batch_identity(b) for view, b in batches.items()})
                    for view, cpu_batch in batches.items():
                        batch = cpu_batch.to(device)
                        bank = model.encoder(batch.observation)
                        bank_bytes = max(bank_bytes, sum(s.numel() * s.element_size() for s in bank.states))
                        output = model.decoder(model.reader(bank, batch.queries, access=model.access), batch.queries, batch.targets)
                        for i, losses in enumerate(output.row_nll.cpu().tolist()):
                            record = rows[start + i]
                            losses = losses[:int(batch.queries.steps[i].sum())]
                            record.update({f"{view}_sequence_nll": sum(losses), f"{view}_mean_row_nll": sum(losses) / len(losses),
                                           f"{view}_first_nll": losses[0],
                                           f"{view}_later_nll": sum(losses[1:]) / (len(losses) - 1) if len(losses) > 1 else None,
                                           f"{view}_row_nll": losses})
            _synchronize(device)
        finally:
            model.train(mode)
        for row in rows:
            for view in ("near", "coarse"):
                for response in ("sequence_nll", "mean_row_nll", "first_nll", "later_nll"):
                    a, b = row[f"{view}_{response}"], row[f"detailed_{response}"]
                    row[f"{view}_minus_detailed_{response}"] = None if a is None else a - b
        result[name] = {"policy": model.policy_identity, "parameters": parameter_counts(model),
                         "evaluation_seconds": perf_counter() - started, "max_retained_bank_bytes": bank_bytes,
                         "batch_sha256": identities, **structural_report(rows, seed=seed, bootstrap_samples=bootstrap_samples)}
    contrasts = {response: {} for response in ("sequence_nll", "mean_row_nll")}
    names = list(models)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            for response in contrasts:
                changes = [{"group_id": a["group_id"], "gain": a[f"detailed_{response}"] - b[f"detailed_{response}"]}
                           for a, b in zip(result[left]["blocks"], result[right]["blocks"])]
                contrasts[response][f"{left}_minus_{right}"] = _estimate(changes, "gain", seed=seed, bootstrap_samples=bootstrap_samples)
    return {"schema": "source-action-structure-v2", "models": result,
            **{f"paired_detailed_{response}": values for response, values in contrasts.items()},
            "uncertainty": "paired source-group bootstrap",
            "bootstrap_samples": bootstrap_samples, "seed": seed}
