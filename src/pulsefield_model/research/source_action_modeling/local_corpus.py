"""Source-exact windows from a bounded, song-grouped slice of local 4K beatmaps.

The local Parquet index locates raw files and grouping metadata. Legacy timing,
difficulty and training features do not enter source-action inputs.
"""
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import random
import unicodedata

import pyarrow.parquet as pq

from ..scoped_style_modeling.dataset import ContractError, Interval, canonical_json, digest
from ..scoped_style_modeling.replay import parse_source, prepare_chart
from .sampling import TrainingContext


def normalized(value):
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


def grouping(sources, rows, assignments):
    """Join set, beatmap, published-group and normalized artist/title identities.

    An annotated component inherits every published split it touches. New
    components receive a fixed hash split, independent of action values.
    """
    entries = list(sources) + list(rows)
    parent = list(range(len(entries)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owner = {}
    for i, row in enumerate(entries):
        tokens = []
        for field in ("beatmap_set_id", "beatmap_id"):
            value = row.get(field)
            if value is not None and str(value).lstrip("-").isdigit() and int(value) > 0:
                tokens.append((field, str(int(value))))
        song = tuple(normalized(row.get(k)) for k in ("artist", "title"))
        if all(song):
            tokens.append(("song", *song))
        if i < len(sources):
            tokens.append(("published-group", assignments[row["source_sha256"]]["group_id"]))
        for token in tokens:
            if token in owner:
                a, b = root(i), root(owner[token])
                parent[max(a, b)] = min(a, b)
            else:
                owner[token] = i
    components = defaultdict(lambda: {"sources": [], "rows": []})
    for i, row in enumerate(entries):
        components[root(i)]["sources" if i < len(sources) else "rows"].append(row)
    result = []
    for component in components.values():
        keys = sorted(["source:" + s["source_sha256"] for s in component["sources"]]
                      + [f"path:{r['shard']}/{r['beatmap_path']}" for r in component["rows"]])
        identity = digest(canonical_json(keys).encode())
        splits = sorted({assignments[s["source_sha256"]]["split"] for s in component["sources"]})
        bucket = int(digest(("source-action-local-split-v1:" + identity).encode())[:16], 16) % 10
        result.append({**component, "identity": identity, "annotation_splits": splits,
                       "split": "train" if bucket < 8 else "validation" if bucket == 8 else "test"})
    return sorted(result, key=lambda c: c["identity"])


def header_metadata(data):
    section, result = "", {}
    for raw in data.decode("utf-8-sig").splitlines():
        line = raw.strip()
        if line.startswith("["):
            section = line
        elif section == "[Metadata]" and ":" in line:
            key, value = line.split(":", 1)
            result[key] = value.strip()
        elif section == "[HitObjects]":
            break
    return result


def source_windows(source, group, *, rows=128, count=2):
    """Select bounded source-event intervals without consulting action labels."""
    times = sorted({t for obj in source.objects for t in (obj.start_ms, obj.end_ms)})
    if len(times) < rows:
        raise ContractError(f"Source has fewer than {rows} source events")
    starts = random.Random("source-action-windows-v1:" + source.source_sha256).sample(
        range(len(times) - rows + 1), min(count, len(times) - rows + 1))
    contexts, manifest = [], []
    for start in sorted(starts):
        end = times[start + rows] if start + rows < len(times) else times[-1] + 1
        interval = Interval(times[start], end)
        key = digest(canonical_json([source.source_sha256, asdict(interval), "local-source-event-window-v1"]).encode())
        chart = prepare_chart(source.objects, interval, interval)
        context = TrainingContext("local:" + group, key, chart)
        if context.event_count != rows:
            raise ContractError("Source window changed event count during replay")
        contexts.append(context)
        manifest.append({"input_id": key, "group_id": context.group_id, "source_sha256": source.source_sha256,
                         "arrangement_sha256": source.arrangement_sha256, "scope": asdict(interval),
                         "context": asdict(interval), "source_event_start": start, "rows": rows})
    return contexts, manifest


def load_local_corpus(sources, assignments, *, index_path=Path("artifacts/indexes/beatmap_index_4k.parquet"),
                      dataset_root=Path("dataset"), train_groups=1024, validation_groups=128):
    """Load novel song components; reject invalid replay with recorded reasons.

    All components touching annotations are excluded from the added corpus.
    Annotation training groups connected to held-out annotations are returned
    for exclusion from both pretraining and human fitting. Local test files are
    not opened. Selection scans at most twice each requested group count.
    """
    payload = Path(index_path).read_bytes()
    rows = pq.read_table(index_path).to_pylist()
    components = grouping(sources, rows, assignments)
    exclusions = sorted({assignments[s["source_sha256"]]["group_id"] for c in components
        if any(split != "train" for split in c["annotation_splits"])
        for s in c["sources"] if assignments[s["source_sha256"]]["split"] == "train"})
    candidates = [c for c in components if not c["sources"]]
    report = {"index_path": str(index_path), "index_sha256": digest(payload), "index_rows": len(rows),
        "index_components": len(components), "novel_components": len(candidates),
        "split_policy": "source-action-local-split-v1; hash bucket 0..7 train, 8 validation, 9 test",
        "excluded_annotation_training_groups": exclusions,
        "candidate_groups": {s: sum(c["split"] == s for c in candidates) for s in ("train", "validation", "test")},
        "selected": [], "rejected": [], "test_source_payloads_opened": False,
        "audio_deduplication": "not-established"}
    result = {"train": [], "validation": []}
    seen_arrangements = set()
    for split, bound in (("train", train_groups), ("validation", validation_groups)):
        retained = 0
        for component in [c for c in candidates if c["split"] == split][:2 * bound]:
            # One chart per song component prevents large difficulty sets from
            # receiving more chart exposure merely because they contain variants.
            row = min(component["rows"], key=lambda r: digest(f"chart-v1:{r['shard']}/{r['beatmap_path']}".encode()))
            path = Path(dataset_root) / str(row["shard"]) / row["beatmap_path"]
            identity = {"component": component["identity"], "split": split, "path": str(path)}
            try:
                data = path.read_bytes()
                header = header_metadata(data)
                if any(normalized(header.get(k)) != normalized(row.get(field)) for k, field in
                       (("Artist", "artist"), ("Title", "title"))):
                    raise ContractError("Original metadata differs from the local index")
                for key, field in (("BeatmapID", "beatmap_id"), ("BeatmapSetID", "beatmap_set_id")):
                    actual, indexed = header.get(key), row.get(field)
                    if actual and indexed is not None and int(actual) > 0 and int(indexed) > 0 and int(actual) != int(indexed):
                        raise ContractError("Original positive beatmap/set ID differs from the local index")
                sha = digest(data)
                if sha in assignments:
                    raise ContractError("Novel component contains a published annotation source")
                source = parse_source(data, sha)
                if source.arrangement_sha256 in seen_arrangements:
                    raise ContractError("Selected local source duplicates an already selected exact arrangement")
                contexts, windows = source_windows(source, component["identity"])
            except (ContractError, OSError, UnicodeError, ValueError) as exc:
                report["rejected"].append({**identity, "error": f"{type(exc).__name__}: {exc}"})
                continue
            result[split].extend(contexts)
            seen_arrangements.add(source.arrangement_sha256)
            report["selected"].append({**identity, "source_sha256": sha, "windows": windows})
            retained += 1
            if retained % 128 == 0:
                print(f"Verified local {split} song groups: {retained}/{bound}", flush=True)
            if retained == bound:
                break
        if retained != bound:
            raise ContractError(f"Only {retained}/{bound} valid novel {split} groups within the declared scan bound")
    selected = {r["path"]: r for r in report["selected"]}
    rejected = {r["path"]: r["error"] for r in report["rejected"]}
    allocations = []
    for component in components:
        for origin, members in (("annotation", component["sources"]), ("local-index", component["rows"])):
            for member in members:
                sha = member.get("source_sha256")
                path = str(Path(dataset_root) / str(member["shard"]) / member["beatmap_path"]) if origin == "local-index" else None
                chosen = selected.get(path)
                published = assignments.get(sha, {})
                allocations.append({"origin": origin, "source_path": path,
                    "source_sha256": sha or (chosen["source_sha256"] if chosen else None),
                    "component_id": component["identity"], "published_group_id": published.get("group_id"),
                    "published_split": published.get("split"), "annotation_splits": component["annotation_splits"],
                    "local_split": component["split"] if not component["sources"] else "annotation-reserved",
                    "selected_local_source": chosen is not None,
                    "annotation_training_excluded": published.get("split") == "train" and published.get("group_id") in exclusions,
                    "rejection": rejected.get(path), "artist": member.get("artist"), "title": member.get("title"),
                    "beatmap_id": str(member.get("beatmap_id")), "beatmap_set_id": str(member.get("beatmap_set_id"))})
    report["source_assignments"] = allocations
    return result, exclusions, report
