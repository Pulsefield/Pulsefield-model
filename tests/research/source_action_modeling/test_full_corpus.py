from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from pulsefield_model.research.scoped_style_modeling.dataset import ContractError, digest
from pulsefield_model.research.scoped_style_modeling.replay import parse_source
from pulsefield_model.research.source_action_modeling.checkpoint import load_snapshot, save_snapshot
from pulsefield_model.research.source_action_modeling.comparison import train_paired_step
from pulsefield_model.research.source_action_modeling.composition import ORDERS, initialize_composition
from pulsefield_model.research.source_action_modeling.full_corpus import (FullCorpusSampler, SourceCatalog,
    SAMPLING_POLICY, build_catalog, validation_contexts)
from pulsefield_model.research.source_action_modeling.observation import EventBlock, declared_entering_occupancy, observe


def source_bytes(title, beatmap, set_id, *, count=80, gap=100, hold=False):
    header = (f"osu file format v14\n[General]\nMode:3\n[Difficulty]\nCircleSize:4\n"
              f"[Metadata]\nArtist:Artist\nTitle:{title}\nBeatmapID:{beatmap}\nBeatmapSetID:{set_id}\n[HitObjects]\n")
    notes = ("64,192,0,128,0,1000:0:0:0:0:\n" if hold else "")
    for i in range(count):
        lane = 1 + i % 3 if hold else i % 4
        notes += f"{64 + lane * 128},192,{(i + 1) * gap},1,0,0:0:0:0:\n"
    return (header + notes).encode()


@pytest.fixture
def source_fixture(tmp_path):
    raw, cache = tmp_path / "raw", tmp_path / "sources"
    (raw / "s").mkdir(parents=True)
    cache.mkdir()
    sources, assignments, rows = [], {}, []

    def add(title, bid, sid, split, *, indexed=True, annotated=True, count=80, gap=100, hold=False):
        data = source_bytes(title, bid, sid, count=count, gap=gap, hold=hold)
        sha = digest(data)
        if annotated:
            sources.append({"artist": "Artist", "title": title, "beatmap_id": bid, "beatmap_set_id": sid,
                "source_sha256": sha, "byte_length": len(data), "note_count": count + int(hold)})
            assignments[sha] = {"group_id": f"published-{sid}-{split}", "split": split}
            (cache / (sha + ".osu")).write_bytes(data)
        if indexed:
            path = f"{bid}.osu"
            rows.append({"shard": "s", "beatmap_path": path, "artist": "Artist", "title": title,
                         "beatmap_id": bid, "beatmap_set_id": sid})
            (raw / "s" / path).write_bytes(data)
        return sha

    a = add("A", 11, 1, "train", gap=100, hold=True)
    b = add("A", 12, 1, "train", annotated=False, gap=101)
    c = add("C", 31, 3, "train", indexed=False, count=7, gap=200)
    add("V", 21, 2, "validation", gap=150)
    # An exact arrangement in a different training group is rejected against V.
    add("Duplicate", 41, 4, "train", gap=150)
    add("Mixed", 51, 5, "train", gap=110)
    add("Mixed", 52, 5, "test", gap=120)
    test_paths = {raw / "s/51.osu", raw / "s/52.osu"}
    test_paths.update(cache / (s["source_sha256"] + ".osu") for s in sources if s["title"] == "Mixed")
    # One malformed beatmap must not remove its valid siblings.
    rows.append({"shard": "s", "beatmap_path": "bad.osu", "artist": "Artist", "title": "A",
                 "beatmap_id": 13, "beatmap_set_id": 1})
    (raw / "s/bad.osu").write_text("bad source")
    index = tmp_path / "index.parquet"
    pq.write_table(pa.Table.from_pylist(rows), index)
    return {"sources": sources, "assignments": assignments, "index_path": index, "dataset_root": raw,
            "source_cache": cache}, (a, b, c), test_paths


def test_full_population_all_beatmaps_cache_only_sources_and_heldout_isolation(source_fixture, monkeypatch):
    kwargs, expected, forbidden = source_fixture
    read_bytes = Path.read_bytes
    def checked(path):
        assert path not in forbidden, "Test source payload was opened"
        return read_bytes(path)
    monkeypatch.setattr(Path, "read_bytes", checked)
    catalog, excluded, report = build_catalog(**kwargs, max_source_rows=16, source_cache_charts=2)
    assert {e.source_sha256 for e in catalog.entries if e.split == "train"} == set(expected)
    assert len(catalog.groups["train"]) == 2
    assert excluded == ["published-5-train"]
    assert report["allocation_status_counts"]["rejected"] == 1
    assert any(r["rejection"] == "exact-source-or-arrangement-duplicate:validation" for r in report["source_assignments"])
    assert all(r["status"] == "heldout-no-read" for r in report["source_assignments"] if r["split"] == "test")
    assert len(report["empty_training_groups"]) == 1
    contexts, manifest = validation_contexts(catalog, 1)
    assert len(contexts) == len(manifest) == 2 and all(c.event_count == 16 for c in contexts)


def test_dynamic_sampling_has_equal_song_weight_replay_and_bounded_cache(source_fixture):
    kwargs, expected, _ = source_fixture
    catalog, _, _ = build_catalog(**kwargs, max_source_rows=16, source_cache_charts=2)
    sampler = FullCorpusSampler(catalog, seed=17)
    _, records = sampler.draw(40)
    assert {r["source_sha256"] for r in records} == set(expected)
    assert len({r["source_window_start"] for r in records}) > 2
    for record in records:
        e = catalog.by_sha[record["source_sha256"]]
        rows = min(e.event_count, 16)
        scales = [n for n in (4, 16, 64) if n <= rows]
        probability = 1 / (2 * len(catalog.groups["train"][e.group_id]) * catalog.window_count(e) *
                           len(scales) * (rows - record["rows"] + 1))
        assert record["sampling_probability"] == probability
        assert record["context_rows"] <= 16
    assert sampler.coverage() == {"draws": 40, "seen_groups": 2, "total_groups": 2, "seen_beatmaps": 3, "total_beatmaps": 3}
    assert catalog.source.cache_info().currsize == 2
    state = sampler.state_dict()
    replay_catalog = SourceCatalog(catalog.entries, max_source_rows=16, source_cache_charts=1)
    replay = FullCorpusSampler(replay_catalog, seed=99)
    replay.load_state_dict(state)
    assert sampler.draw(5)[1] == replay.draw(5)[1]
    wrong = FullCorpusSampler(SourceCatalog(catalog.entries, max_source_rows=32))
    with pytest.raises(ContractError, match="population"):
        wrong.load_state_dict(state)


def test_dynamic_window_preserves_entering_long_note_and_detects_source_mutation(source_fixture):
    kwargs, expected, _ = source_fixture
    catalog, _, _ = build_catalog(**kwargs, max_source_rows=16, source_cache_charts=1)
    entry = catalog.by_sha[expected[0]]
    context = catalog.window(entry, 5)
    assert context.event_count == 16
    assert declared_entering_occupancy(context.chart) == (True, False, False, False)
    assert context.chart.inputs.scope.start_ms == 500
    other = catalog.by_sha[expected[1]]
    catalog.window(other, 0)
    Path(entry.path).write_bytes(Path(entry.path).read_bytes() + b"\n")
    with pytest.raises(ContractError, match="SHA-256"):
        catalog.window(entry, 5)


@pytest.mark.parametrize("head", [b"64,192,1000,1,0,0:0:0:0:\n", b"64,192,1000,128,0,1100:0:0:0:0:\n"])
def test_catalog_rejects_simultaneous_close_and_new_attack(source_fixture, head):
    kwargs, _, _ = source_fixture
    data = source_bytes("A", 12, 1, hold=True) + head
    sha = digest(data)
    assert parse_source(data, sha).row_incompatibilities
    (kwargs["dataset_root"] / "s/12.osu").write_bytes(data)
    catalog, _, report = build_catalog(**kwargs, max_source_rows=16)
    assert sha not in catalog.by_sha
    rejection = next(r for r in report["source_assignments"] if r["source_sha256"] == sha)
    assert rejection["status"] == "rejected" and "SourceActionSchemaError" in rejection["rejection"]


def test_full_sampler_checkpoint_continues_identical_updates(source_fixture, tmp_path):
    kwargs, _, _ = source_fixture
    catalog, _, _ = build_catalog(**kwargs, max_source_rows=16)
    sampler = FullCorpusSampler(catalog)
    models = initialize_composition()
    optimizers = {n: torch.optim.AdamW(m.parameters(), lr=0.001) for n, m in models.items()}
    first = train_paired_step(models, optimizers, sampler, blocks=1, configurations=tuple(ORDERS))
    assert first["sampling_policy"] == SAMPLING_POLICY
    for name, model in models.items():
        save_snapshot(tmp_path / (name + ".pt"), model, optimizers[name], sampler, update=1)
    expected = train_paired_step(models, optimizers, sampler, blocks=1, configurations=tuple(ORDERS))
    fresh = initialize_composition()
    fresh_optimizers = {n: torch.optim.AdamW(m.parameters(), lr=0.001) for n, m in fresh.items()}
    continued = FullCorpusSampler(catalog)
    for name, model in fresh.items():
        assert load_snapshot(tmp_path / (name + ".pt"), model, fresh_optimizers[name], continued) == 1
    actual = train_paired_step(fresh, fresh_optimizers, continued, blocks=1, configurations=tuple(ORDERS))
    assert actual["blocks"] == expected["blocks"] and continued.coverage() == sampler.coverage()
    for name, model in models.items():
        assert all(torch.equal(value, fresh[name].state_dict()[key]) for key, value in model.state_dict().items())
