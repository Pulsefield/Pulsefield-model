from dataclasses import asdict, fields
from io import BytesIO
import json
import subprocess
import sys

import pytest
from omegaconf.errors import ConfigKeyError
from hydra.errors import ConfigCompositionException

from pulsefield_model.research.scoped_style_modeling.dataset import (
    FOUNDATION, ContractError, adapt_records, digest, fetch_verified, freeze_manifest, load_snapshot, split_manifest,
)
from pulsefield_model.research.scoped_style_modeling.prepare import PrepareConfig
from pulsefield_model.research.scoped_style_modeling.prepare_hydra import compose_config
from pulsefield_model.research.scoped_style_modeling.recovery import recover_sources
from pulsefield_model.research.scoped_style_modeling.replay import ChartInputs, LaneFacts, Row


def source(n, **kwargs):
    return {"source_sha256": digest(str(n).encode()), "beatmap_id": n, "beatmap_set_id": n, **kwargs}


def record(n=1, **kwargs):
    value = {"record_id": f"human:{n}", "cell_id": "cell-1", "source_sha256": source(1)["source_sha256"],
             "foundation_id": FOUNDATION, "start_ms": 100, "end_ms": 200, "tag_id": "tech",
             "playback_rate": 1, "origin": "human-confirmed", "presence": "present", "salience": "supporting",
             "details": {"review_context": {"start_ms": 0, "end_ms": 300}, "evidence": {"note_refs": []}},
             "human_confidence": "low"}
    value.update(kwargs)
    return value


def test_origin_layers_duplicates_and_confirmation_metadata():
    human = [record(2), record(1)]
    machine = [record(3, origin="agent-reviewed", human_confidence=None)]
    result, issues = adapt_records({"machine": machine, "human": human}, [source(1)])
    assert len(result) == 2
    assert result[0].layer == "human"
    assert result[0].record_ids == ("human:1", "human:2")
    assert result[0].evidence == ()
    provenance = json.loads(result[0].provenance_json)
    assert provenance[0]["origin"] == "human-confirmed"
    assert provenance[0]["human_confidence"] == "low"
    assert issues[0]["kind"] == "agreeing-duplicate"
    assert len(result[1].record_ids) == 1


def test_conflicts_unresolved_unreviewed_are_not_absent():
    rows = [record(1), record(2, presence="absent", salience=None),
            record(3, cell_id="cell-2", start_ms=101, presence="unresolved", salience=None),
            record(4, cell_id="cell-3", start_ms=102, presence="unreviewed", salience=None)]
    records, issues = adapt_records({"human": rows}, [source(1)])
    assert records == ()
    assert [i["kind"] for i in issues] == ["conflicting-cell", "unsupervised-cell", "unsupervised-cell"]


def test_missing_evidence_keeps_same_assessment_and_explicit_empty_is_available():
    empty = record()
    missing = record()
    missing["details"]["evidence"] = None
    a, _ = adapt_records({"human": [empty]}, [source(1)])
    b, _ = adapt_records({"human": [missing]}, [source(1)])
    assert a[0].exact_cell == b[0].exact_cell
    assert a[0].assessment == b[0].assessment
    assert a[0].evidence == () and b[0].evidence is None


@pytest.mark.parametrize("change", [
    {"playback_rate": 1.5}, {"foundation_id": "wrong"}, {"tag_id": "density"},
    {"presence": "absent", "salience": "prominent"}, {"source_sha256": "changed"},
    {"origin": "agent-reviewed"},
])
def test_adapter_rejects_contract_changes(change):
    with pytest.raises(ContractError):
        adapt_records({"human": [record(**change)]}, [source(1)])


def test_invalid_context_and_reference_do_not_become_missing():
    value = record()
    value["details"]["review_context"]["end_ms"] = 150
    with pytest.raises(ContractError, match="outside"):
        adapt_records({"human": [value]}, [source(1)])
    value = record()
    value["details"]["evidence"]["note_refs"] = [{"source_line": 0, "column": 0, "kind": "normal", "start_ms": 100, "end_ms": 100}]
    with pytest.raises(ContractError, match="identity"):
        adapt_records({"human": [value]}, [source(1)])


def test_cell_ids_are_exact_bijection():
    with pytest.raises(ContractError, match="bijection"):
        adapt_records({"human": [record(1), record(2, start_ms=101)]}, [source(1)])
    with pytest.raises(ContractError, match="bijection"):
        adapt_records({"human": [record(1), record(2, cell_id="other")]}, [source(1)])


def test_split_groups_known_versions_song_audio_and_duplicate_arrangements(tmp_path):
    sources = [source(1), source(2, beatmap_set_id=1), source(3, beatmap_id=1), source(4), source(5), source(6)]
    ids = [s["source_sha256"] for s in sources]
    options = {"known_groups": {"audio:shared": ids[2:4]}, "arrangement_hashes": {ids[3]: "same", ids[4]: "same"}}
    manifest = split_manifest(sources, **options)
    assert manifest == split_manifest(sources[::-1], **options)
    assert len(manifest["groups"]) == 2
    assert len({manifest["sources"][s]["group_id"] for s in ids[:5]}) == 1
    assert len({manifest["sources"][s]["split"] for s in ids[:5]}) == 1
    path = tmp_path/'split.json'
    freeze_manifest(path, manifest)
    freeze_manifest(path, manifest)
    before = path.read_bytes()
    with pytest.raises(ContractError, match="frozen manifest differs"):
        freeze_manifest(path, split_manifest(sources))
    assert path.read_bytes() == before
    with pytest.raises(ContractError, match="unpublished"):
        split_manifest(sources, known_groups={"invalid": ["unknown"]})


def test_split_is_seed_zero_hash_threshold_and_has_no_label_input():
    sources = [source(i, beatmap_id=None, beatmap_set_id=None) for i in range(100)]
    manifest = split_manifest(sources)
    for item in sources:
        sha = item["source_sha256"]
        x = int(digest(f"0:{sha}".encode()), 16) / 2**256
        expected = "train" if x < .8 else "validation" if x < .9 else "test"
        assert manifest["sources"][sha]["split"] == expected
    assert {v["split"] for v in manifest["sources"].values()} == {"train", "validation", "test"}


def test_chart_input_types_have_no_target_or_provenance_channels():
    forbidden = {"assessment", "evidence", "concept", "record_ids", "source_line", "origin", "provenance_json", "rationale", "context_note_refs", "cell_id"}
    for cls in (ChartInputs, Row, LaneFacts):
        assert not ({f.name for f in fields(cls)} & forbidden)


def test_recovery_mismatch_and_offline_absence_are_explicit(tmp_path):
    data = b"original source bytes"
    sha = digest(data)
    row = {"source_sha256": sha, "byte_length": len(data), "source_ref": {"kind": "osu", "uri": "https://osu.ppy.sh/osu/1"}}
    assert recover_sources([row], tmp_path, download=False)[0]["status"] == "failed"
    path = tmp_path/f'{sha}.osu'
    path.write_bytes(b"new chart")
    status = recover_sources([row], tmp_path, download=False)[0]
    assert status["status"] == "failed" and "SHA-256" in status["error"]
    assert path.read_bytes() == b"new chart"
    path.write_bytes(data)
    assert recover_sources([row], tmp_path, download=False)[0]["status"] == "verified"


def test_dataset_manifest_must_match_pin_before_parquet_decode(tmp_path):
    (tmp_path/'manifest.json').write_text('{}')
    with pytest.raises(ContractError, match="SHA-256"):
        load_snapshot(tmp_path)


def test_download_checks_identity_before_persisting(tmp_path, monkeypatch):
    from pulsefield_model.research.scoped_style_modeling import dataset
    path = tmp_path/'source.osu'
    monkeypatch.setattr(dataset, 'urlopen', lambda *args, **kwargs: BytesIO(b'new revision'))
    with pytest.raises(ContractError, match="SHA-256"):
        fetch_verified('https://example.org/source.osu', path, digest(b'original'))
    assert not path.exists()
    monkeypatch.setattr(dataset, 'urlopen', lambda *args, **kwargs: BytesIO(b'original'))
    assert fetch_verified('https://example.org/source.osu', path, digest(b'original')) == b'original'
    assert path.read_bytes() == b'original'


def test_hydra_defaults_projection_and_unknown_fields(tmp_path):
    cfg = compose_config()
    assert cfg == PrepareConfig()
    changed = compose_config([f'output_dir={tmp_path}', 'workers=2', 'timeout_seconds=7', 'download=true', 'known_groups_path=identities.json'])
    assert changed.output_dir == str(tmp_path)
    assert changed.workers == 2 and changed.timeout_seconds == 7 and changed.download
    assert changed.known_groups_path == 'identities.json'
    with pytest.raises((ContractError, ConfigCompositionException, ConfigKeyError)):
        compose_config(['+unexpected=true'])
    with pytest.raises(ContractError):
        compose_config(['workers=0'])
    assert asdict(changed).keys() == asdict(cfg).keys()


def test_help_keeps_torch_and_legacy_runtime_unloaded():
    script = '''
import importlib.abc
import runpy
import sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith(("pulsefield_model.models", "pulsefield_model.training", "pulsefield_model.osu_core")):
            raise AssertionError("Research data preparation imported a model/legacy runtime: " + fullname)
sys.meta_path.insert(0, BlockRuntime())
sys.argv = ["prepare_hydra", "--help"]
runpy.run_module("pulsefield_model.research.scoped_style_modeling.prepare_hydra", run_name="__main__")
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'split_manifest_path:' in result.stdout
