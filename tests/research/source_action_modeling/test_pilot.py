from dataclasses import replace

import pytest
import torch

from pulsefield_model.research.scoped_style_modeling.dataset import ContractError, NoteRef
from pulsefield_model.research.scoped_style_modeling.replay import SourceChart
from pulsefield_model.research.source_action_modeling.local_corpus import grouping, source_windows
from pulsefield_model.research.source_action_modeling.pilot import fixed_validation, _fine_contributions, export_allocation
from pulsefield_model.research.source_action_modeling.observation import ViewPolicy
from pulsefield_model.research.source_action_modeling.model import initialize_comparison
from pulsefield_model.research.source_action_modeling.diagnostics import capture_response, finish_response
from pulsefield_model.research.source_action_modeling.semantic_probe import SemanticCorpus, fit_readout
from pulsefield_model.research.source_action_modeling.tensors import collate
from .conftest import example
from .test_semantic_probe import corpus_fixture


def test_song_components_join_transitive_ids_and_metadata_without_changing_published_split():
    sources = [dict(source_sha256="a", artist="Artist A", title="One", beatmap_set_id=1),
               dict(source_sha256="b", artist="Artist B", title="Two", beatmap_set_id=2)]
    rows = [dict(shard="0", beatmap_path="bridge.osu", artist="  ARTIST B ", title="Two", beatmap_set_id=1),
            dict(shard="0", beatmap_path="new.osu", artist="New", title="Song", beatmap_set_id=3)]
    assignments = {"a": {"group_id": "a", "split": "train"}, "b": {"group_id": "b", "split": "validation"}}
    components = grouping(sources, rows, assignments)
    linked = next(c for c in components if c["sources"])
    assert linked["annotation_splits"] == ["train", "validation"]
    assert len(linked["sources"]) == 2 and len(linked["rows"]) == 1
    assert assignments["a"]["split"] == "train"
    reversed_components = grouping(sources[::-1], rows[::-1], assignments)
    assert {c["identity"]: c["split"] for c in components} == {c["identity"]: c["split"] for c in reversed_components}


def test_source_windows_and_validation_selection_depend_on_positions_not_action_values():
    objects = tuple(NoteRef(i + 1, i % 4, "normal", i * 100, i * 100) for i in range(300))
    source = SourceChart("x", objects, "arrangement", ())
    contexts, manifest = source_windows(source, "group")
    changed = replace(source, objects=tuple(replace(o, column=3-o.column) for o in objects))
    other, other_manifest = source_windows(changed, "group")
    assert manifest == other_manifest
    assert len(contexts) == 2 and all(c.event_count == 128 for c in contexts)
    pairs, records = fixed_validation(contexts, ViewPolicy(8))
    other_pairs, other_records = fixed_validation(other, ViewPolicy(8))
    assert records == other_records and len(records) == 6
    assert {r["rows"] for r in records} == {4, 16, 64}
    assert any(a["detailed"].targets != b["detailed"].targets for a, b in zip(pairs, other_pairs))


def test_fine_update_contributions_sum_to_existing_directional_response():
    model = initialize_comparison()["composed_all"]
    fixed = collate([example()])
    before = capture_response(model, fixed)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    model(fixed).loss.backward()
    optimizer.step()
    result = finish_response(before, model, fixed)
    fine = _fine_contributions(before, model)
    assert sum(fine.values()) == pytest.approx(result["linearized_change"], abs=1e-12)
    assert all(f"encoder.local.{i}" in fine for i in range(3))


def test_probe_progress_can_stop_at_an_update_and_restores_encoder_mode():
    corpus = SemanticCorpus(corpus_fixture(), [])
    model = initialize_comparison()["composed_h"]
    model.encoder.train()
    calls = []
    def stop(step, head, optimizer):
        calls.append(step)
        assert optimizer.state and not head.fitted
        raise ContractError("bounded stop")
    with pytest.raises(ContractError, match="bounded stop"):
        fit_readout(model, corpus, steps=3, batch_size=5, learning_rate=.001, on_step=stop)
    assert calls == [1] and model.encoder.training


def test_parquet_allocation_preserves_window_identity_and_file_hashes(tmp_path):
    import json
    import pyarrow.parquet as pq
    from pulsefield_model.research.scoped_style_modeling.dataset import digest
    item = {"input_id": "input", "group_id": "group", "source_sha256": "source", "chart_key": "chart",
            "chart_sha256": "chart-bytes", "scope": {"start_ms": 10., "end_ms": 100.},
            "context": {"start_ms": 0., "end_ms": 120.}, "event_count": 8}
    population = {"dataset_revision": "revision", "split_sha256": "split", "counts": {"training_contexts": 1},
        "inputs": [item], "validation_inputs": [], "local_corpus": {"index_sha256": "index", "selected": [],
        "source_assignments": [{"source_sha256": "source", "published_split": "train"}]}}
    output = tmp_path / "dataset"
    export_allocation(output, population)
    windows = pq.read_table(output / "windows.parquet").to_pylist()
    assert windows[0]["input_id"] == "input" and windows[0]["scope_start_ms"] == 10.
    assert windows[0]["context_start_ms"] == 0. and windows[0]["chart_sha256"] == "chart-bytes"
    manifest = json.loads((output / "manifest.json").read_text())
    assert all(digest((output / name).read_bytes()) == sha for name, sha in manifest["files"].items())
    with pytest.raises(FileExistsError):
        export_allocation(output, population)
