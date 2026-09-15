"""All eligible song groups and beatmaps, with bounded on-demand source windows.

The index locates original files and split metadata only. Human sections retain
their own scopes in the frozen readout; they do not add pretraining sampling units.
"""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
import json
import random

import pyarrow as pa
import pyarrow.parquet as pq

from ..scoped_style_modeling.corpus import PreparedCorpus
from ..scoped_style_modeling.dataset import (CONCEPTS, REVISION, ContractError, Interval,
    adapt_records, canonical_json, digest, load_snapshot)
from ..scoped_style_modeling.probe_data import support
from ..scoped_style_modeling.replay import prepare_chart
from .actions import ACTION_SCHEMA, parse_source
from .local_corpus import grouping, header_metadata, normalized
from .observation import (BLOCK_SIZES, EventBlock, ViewPolicy, declared_entering_occupancy,
                          observe, observe_complete, paired_views)
from .sampling import SPLIT_SHA256, VIEWS, TrainingContext
from .semantic_probe import SemanticCorpus

SAMPLING_POLICY = "uniform-song/beatmap/source-window/feasible-{4,16,64}/event-start-v2"
WINDOW_POLICY = "source-event-window-v2"


@dataclass(frozen=True)
class SourceEntry:
    group_id: str
    split: str
    source_sha256: str
    arrangement_sha256: str
    path: str
    event_count: int


def _times(source):
    return tuple(sorted({t for obj in source.objects for t in (obj.start_ms, obj.end_ms)}))


class SourceCatalog:
    """Keep identities resident and only a bounded LRU of verified source objects."""
    def __init__(self, entries, *, max_source_rows=256, source_cache_charts=16):
        if any(type(v) is not int or v < 4 for v in (max_source_rows,)) or (
            type(source_cache_charts) is not int or source_cache_charts < 1
        ):
            raise ContractError("Source windows require at least four rows and a positive cache capacity")
        self.entries = sorted(entries, key=lambda e: (e.group_id, e.source_sha256))
        self.by_sha = {e.source_sha256: e for e in self.entries}
        if not self.entries or len(self.by_sha) != len(self.entries):
            raise ContractError("Source catalog requires distinct, nonempty source identities")
        if any(e.split not in ("train", "validation") or e.event_count < 4 for e in self.entries):
            raise ContractError("Catalog entries must be eligible training or validation sources")
        self.max_source_rows = max_source_rows
        self.groups = {s: defaultdict(list) for s in ("train", "validation")}
        for entry in self.entries:
            self.groups[entry.split][entry.group_id].append(entry)
        if set(self.groups["train"]) & set(self.groups["validation"]):
            raise ContractError("Source catalog groups overlap across training and validation")
        self.signature = digest(canonical_json({"sources": [asdict(e) for e in self.entries],
            "window_policy": WINDOW_POLICY, "max_source_rows": max_source_rows, "action_schema": ACTION_SCHEMA}).encode())
        self.source = lru_cache(maxsize=source_cache_charts)(self._source)

    def _source(self, sha):
        entry = self.by_sha[sha]
        source = parse_source(Path(entry.path).read_bytes(), sha)
        times = _times(source)
        if (source.arrangement_sha256, len(times)) != (entry.arrangement_sha256, entry.event_count):
            raise ContractError("Cached source differs from the verified catalog")
        return source, times

    def window_count(self, entry):
        return max(1, entry.event_count - self.max_source_rows + 1)

    def window(self, entry, start):
        if type(start) is not int or not 0 <= start < self.window_count(entry):
            raise ContractError("Source window start is outside the catalog range")
        source, times = self.source(entry.source_sha256)
        rows = min(entry.event_count, self.max_source_rows)
        end = times[start + rows] if start + rows < len(times) else times[-1] + 1
        interval = Interval(times[start], end)
        key = digest(canonical_json([entry.source_sha256, asdict(interval), WINDOW_POLICY]).encode())
        # Full original objects preserve an LN entering this bounded interval.
        chart = prepare_chart(source.objects, interval, interval)
        context = TrainingContext(entry.group_id, key, chart)
        if context.event_count != rows:
            raise ContractError("Source window changed event count during replay")
        return context


def _component_split(component):
    splits = component["annotation_splits"]
    return "test" if "test" in splits else "validation" if "validation" in splits else "train" if splits else component["split"]


def build_catalog(sources, assignments, *, index_path, dataset_root, source_cache,
                  max_source_rows=256, source_cache_charts=16, train_group_limit=None, guard=None):
    """Verify every candidate beatmap; reserve whole connected held-out groups.

    Validation arrangements are checked before training to exclude exact overlap.
    Test members contribute metadata only; audio/content deduplication of unseen
    test payloads is not asserted. Invalid and duplicate candidates remain in the
    allocation with explicit reasons. A limit is only for reduced software runs.
    """
    index_path, dataset_root, source_cache = map(Path, (index_path, dataset_root, source_cache))
    rows = pq.read_table(index_path).to_pylist()
    components = grouping(sources, rows, assignments)
    excluded = {assignments[s["source_sha256"]]["group_id"] for c in components
        if _component_split(c) != "train" for s in c["sources"]
        if assignments[s["source_sha256"]]["split"] == "train"}
    training = [c for c in components if _component_split(c) == "train"]
    selected_groups = {c["identity"] for c in training[:train_group_limit]}
    allocations, entries, seen_sha, seen_arrangements = [], [], {}, {}
    report = {"index_path": str(index_path), "index_sha256": digest(index_path.read_bytes()),
        "index_rows": len(rows), "index_components": len(components),
        "candidate_groups": dict(Counter(_component_split(c) for c in components)),
        "split_policy": "transitive published-group/set/beatmap/normalized-song; heldout wins; novel local-split-v1",
        "train_group_limit": train_group_limit, "test_source_payloads_opened": False,
        "audio_deduplication": "not-established"}
    # Process validation first; no training arrangement may also be used there.
    for component in sorted(components, key=lambda c: (_component_split(c) != "validation", c["identity"])):
        split, group = _component_split(component), "song:" + component["identity"]
        members = [("annotation-source", s, source_cache / (s["source_sha256"] + ".osu")) for s in component["sources"]]
        members.extend(("local-index", r, dataset_root / str(r["shard"]) / r["beatmap_path"])
                       for r in sorted(component["rows"], key=lambda r: (str(r["shard"]), r["beatmap_path"])))
        for origin, member, path in members:
            published = assignments.get(member.get("source_sha256"), {})
            record = {"origin": origin, "source_path": str(path), "source_sha256": member.get("source_sha256"),
                "group_id": group, "split": split, "published_group_id": published.get("group_id"),
                "published_split": published.get("split"), "annotation_splits": component["annotation_splits"],
                "status": "heldout-no-read" if split == "test" else "outside-training-limit",
                "rejection": None, "arrangement_sha256": None, "source_event_rows": None,
                "canonical_source_sha256": None}
            allocations.append(record)
            if split == "test" or (split == "train" and component["identity"] not in selected_groups):
                continue
            if guard is not None:
                guard()
            try:
                data = path.read_bytes()
                sha = digest(data)
                if origin == "annotation-source" and (sha != member["source_sha256"] or len(data) != member["byte_length"]):
                    raise ContractError("Original annotation source bytes differ from the publication")
                header = header_metadata(data)
                if any(normalized(header.get(k)) != normalized(member.get(field)) for k, field in
                       (("Artist", "artist"), ("Title", "title"))):
                    raise ContractError("Original metadata differs from the source catalog")
                for key, field in (("BeatmapID", "beatmap_id"), ("BeatmapSetID", "beatmap_set_id")):
                    actual, indexed = header.get(key), member.get(field)
                    if actual and indexed is not None and int(actual) > 0 and int(indexed) > 0 and int(actual) != int(indexed):
                        raise ContractError("Original positive beatmap/set ID differs from the source catalog")
                record["source_sha256"] = sha
                if sha in assignments and assignments[sha]["split"] != split:
                    raise ContractError("Original source belongs to a different published split")
                source = parse_source(data, sha)
                count = len(_times(source))
                record.update(arrangement_sha256=source.arrangement_sha256, source_event_rows=count)
                if origin == "annotation-source" and len(source.objects) != member["note_count"]:
                    raise ContractError("Original annotation object count differs from the publication")
                if count < 4:
                    raise ContractError("Source has fewer than four source events")
                previous = seen_sha.get(sha) or seen_arrangements.get(source.arrangement_sha256)
                if previous:
                    record.update(status="duplicate", canonical_source_sha256=previous.source_sha256,
                                  rejection="exact-source-or-arrangement-duplicate:" + previous.split)
                    continue
                entry = SourceEntry(group, split, sha, source.arrangement_sha256, str(path), count)
                entries.append(entry)
                seen_sha[sha] = seen_arrangements[source.arrangement_sha256] = entry
                record["status"] = "eligible"
                if len(entries) % 256 == 0:
                    print(f"Verified full-corpus beatmaps: {len(entries)}", flush=True)
            except (ContractError, OSError, UnicodeError, ValueError) as exc:
                record.update(status="rejected", rejection=f"{type(exc).__name__}: {exc}")
    catalog = SourceCatalog(entries, max_source_rows=max_source_rows, source_cache_charts=source_cache_charts)
    if not catalog.groups["train"] or not catalog.groups["validation"]:
        raise ContractError("Full corpus requires eligible training and validation groups")
    report.update(excluded_annotation_training_groups=sorted(excluded),
        allocation_status_counts=dict(Counter(r["status"] for r in allocations)),
        eligible_groups={s: len(g) for s, g in catalog.groups.items()},
        eligible_beatmaps=dict(Counter(e.split for e in entries)),
        empty_training_groups=sorted("song:" + g for g in selected_groups if "song:" + g not in catalog.groups["train"]),
        source_assignments=allocations, catalog_sha256=catalog.signature)
    return catalog, sorted(excluded), report


class FullCorpusSampler:
    """Uniform song, beatmap, source window, feasible target size and target start.

    Only identities and event counts select exposures. Cache eviction never
    consumes RNG. Coverage is cumulative per seed and travels with checkpoints.
    """
    sampling_policy = SAMPLING_POLICY

    def __init__(self, catalog, seed=17, *, policy=ViewPolicy()):
        self.catalog, self.policy = catalog, policy
        self.groups = catalog.groups["train"]
        self.group_ids = sorted(self.groups)
        if not self.group_ids:
            raise ContractError("Sampler requires training sources")
        self.rng, self.position, self.seen = random.Random(seed), 0, set()

    def draw(self, count=1):
        if type(count) is not int or count < 1:
            raise ContractError("Sample count must be a positive integer")
        paired, records = [], []
        for _ in range(count):
            group = self.rng.choice(self.group_ids)
            entry = self.rng.choice(self.groups[group])
            windows = self.catalog.window_count(entry)
            window_start = self.rng.randrange(windows)
            context = self.catalog.window(entry, window_start)
            scales = [s for s in BLOCK_SIZES if s <= context.event_count]
            size = self.rng.choice(scales)
            start = self.rng.randrange(context.event_count - size + 1)
            example = observe(context.chart, EventBlock(start, size),
                              entering_occupancy=declared_entering_occupancy(context.chart))
            paired.append(paired_views(example, self.policy))
            obs = example.observation
            records.append({"position": self.position, "group_id": group, "source_sha256": entry.source_sha256,
                "source_event_rows": entry.event_count, "source_window_start": window_start,
                "context_key": context.key, "context_rows": context.event_count,
                "scope": asdict(context.chart.inputs.scope), "event_start": start, "rows": size,
                "attack_group_span": example.attack_group_span, "views": list(VIEWS), "view_policy": asdict(self.policy),
                "sampling_probability": 1 / (len(self.groups) * len(self.groups[group]) * windows *
                                             len(scales) * (context.event_count - size + 1)),
                "duration_ms": obs.rows[obs.target_indices[-1]].time_ms - obs.rows[obs.target_indices[0]].time_ms})
            self.position += 1
            self.seen.add(entry.source_sha256)
        return paired, records

    def coverage(self):
        return {"draws": self.position, "seen_groups": len({self.catalog.by_sha[s].group_id for s in self.seen}),
                "total_groups": len(self.groups), "seen_beatmaps": len(self.seen),
                "total_beatmaps": sum(len(v) for v in self.groups.values())}

    def state_dict(self):
        return {"policy": self.sampling_policy, "population": self.catalog.signature,
                "view_policy": asdict(self.policy), "position": self.position, "rng": self.rng.getstate(),
                "seen_sources": sorted(self.seen)}

    def load_state_dict(self, state):
        if (state["policy"], state["population"], state["view_policy"]) != (
            self.sampling_policy, self.catalog.signature, asdict(self.policy)
        ):
            raise ContractError("Sampler policy or source population differs from snapshot")
        seen = set(state["seen_sources"])
        allowed = {e.source_sha256 for g in self.groups.values() for e in g}
        if type(state["position"]) is not int or state["position"] < len(seen) or not seen <= allowed:
            raise ContractError("Invalid snapshot sampling coverage")
        self.rng.setstate(state["rng"])
        self.position, self.seen = state["position"], seen


def validation_contexts(catalog, groups):
    """Fix two windows from one identity-selected beatmap per validation song."""
    available = sorted(catalog.groups["validation"], key=lambda g: digest(f"composition-validation-v2:{g}".encode()))
    if len(available) < groups:
        raise ContractError("Insufficient groups for the fixed validation population")
    contexts, manifest = [], []
    for group in available[:groups]:
        entry = min(catalog.groups["validation"][group], key=lambda e: digest(f"validation-chart-v2:{e.source_sha256}".encode()))
        windows = catalog.window_count(entry)
        rng = random.Random("validation-window-v2:" + entry.source_sha256)
        for start in sorted(rng.sample(range(windows), min(2, windows))):
            context = catalog.window(entry, start)
            contexts.append(context)
            manifest.append({"input_id": context.key, **asdict(entry), "source_event_start": start,
                "rows": context.event_count, "scope": asdict(context.chart.inputs.scope)})
    return contexts, manifest


def load_population(prepared_dir, dataset_dir, *, index_path, dataset_root, source_cache,
                    validation_groups, max_source_rows=256, source_cache_charts=16, train_group_limit=None, guard=None):
    corpus = PreparedCorpus(Path(prepared_dir))
    manifest, sources, tables = load_snapshot(Path(dataset_dir))
    _, issues = adapt_records(tables, sources)
    if any(value for key, value in manifest["exclusions"]["human"].items() if key != "source-excluded"):
        raise ContractError("Human publication exclusions require an exact-cell adapter")
    assignments = json.loads((corpus.root / "split-manifest.json").read_text())["sources"]
    catalog, excluded, local = build_catalog(sources, assignments, index_path=index_path,
        dataset_root=dataset_root, source_cache=source_cache, max_source_rows=max_source_rows,
        source_cache_charts=source_cache_charts, train_group_limit=train_group_limit, guard=guard)
    semantic = SemanticCorpus(corpus, issues)
    semantic.train_indices = [i for i in semantic.train_indices if corpus.records[i]["group_id"] not in excluded]
    semantic.eligible = set(semantic.train_indices + semantic.validation_indices)
    if {corpus.records[i]["concept"] for i in semantic.train_indices} != set(CONCEPTS):
        raise ContractError("Song exclusions removed training support for a concept")
    for i in sorted(semantic.eligible):
        if guard is not None:
            guard()
        row = corpus.records[i]
        chart, _ = corpus.chart(row["chart_key"], row["chart_sha256"])
        observe_complete(chart, entering_occupancy=declared_entering_occupancy(chart))
    corpus.chart.cache_clear()
    validation, windows = validation_contexts(catalog, validation_groups)
    population = {"dataset_revision": REVISION, "split_sha256": SPLIT_SHA256,
        "cohort_sha256": corpus.summary["cohort_sha256"], "local_corpus": local,
        "sampling_policy": SAMPLING_POLICY, "max_source_rows": max_source_rows,
        "semantic_support": support(corpus.records, sorted(semantic.eligible)),
        "semantic_train_indices": semantic.train_indices, "semantic_validation_indices": semantic.validation_indices,
        "annotation_sections_in_pretraining": "original sources participate as beatmaps; no extra section sampling weight",
        "validation_inputs": windows, "test_chart_payloads_opened": False,
        "counts": {"training_groups": len(catalog.groups["train"]),
                   "training_beatmaps": sum(e.split == "train" for e in catalog.entries),
                   "possible_training_windows": sum(catalog.window_count(e) for e in catalog.entries if e.split == "train"),
                   "validation_candidate_groups": len(catalog.groups["validation"]),
                   "validation_groups": validation_groups, "validation_contexts": len(validation),
                   "semantic_training_cells": len(semantic.train_indices),
                   "semantic_validation_cells": len(semantic.validation_indices)}}
    return catalog, validation, semantic, population


def export_allocation(directory, population, catalog, *, protocol):
    directory.mkdir(parents=True, exist_ok=False)
    assignments = population["local_corpus"].pop("source_assignments")
    tables = {"source-assignments": assignments,
        "eligible-sources": [{**asdict(e), "context_rows": min(e.event_count, catalog.max_source_rows),
            "window_start_count": catalog.window_count(e)} for e in catalog.entries],
        "validation-windows": population["validation_inputs"]}
    for name, records in tables.items():
        pq.write_table(pa.Table.from_pylist(records), directory / (name + ".parquet"), compression="zstd")
    manifest = {"schema": "source-action-full-corpus-allocation-v2", "protocol": protocol,
        "catalog_sha256": catalog.signature, "annotation_revision": population["dataset_revision"],
        "annotation_split_sha256": population["split_sha256"], "counts": population["counts"],
        "local_index_sha256": population["local_corpus"]["index_sha256"],
        "rows": {k: len(v) for k, v in tables.items()},
        "realized_training_windows": "seed-*/updates.jsonl; checkpoint sampler records cumulative source coverage",
        "files": {p.name: digest(p.read_bytes()) for p in sorted(directory.iterdir())}}
    (directory / "manifest.json").write_text(canonical_json(manifest) + "\n")
