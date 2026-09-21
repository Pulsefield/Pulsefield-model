from copy import deepcopy
import json

import pytest

from pulsefield_model.research.scoped_style_modeling.dataset import ContractError
from pulsefield_model.research.source_action_modeling.smoke import load_contexts, run_smoke, select_contexts


def records():
    rows, assignments = [], {}
    for i in range(12):
        source, group = f"source-{i:02}", f"group-{i:02}"
        split = "train" if i < 9 else "validation" if i == 9 else "test"
        assignments[source] = {"group_id": group, "split": split}
        for suffix in ("b", "a"):
            rows.append({"source_sha256": source, "group_id": group, "split": split, "chart_key": f"{source}-{suffix}",
                         "scope": {"start_ms": 0, "end_ms": 100}, "context": {"start_ms": 0, "end_ms": 200},
                         "concept": "anything", "assessment": "anything"})
    return rows, assignments


def test_selects_exactly_eight_distinct_training_contexts_without_labels():
    rows, assignments = records()
    selected = select_contexts(rows, assignments)
    changed = deepcopy(rows[::-1])
    for row in changed:
        row.pop("concept")
        row["assessment"] = "different"
    assert select_contexts(changed, assignments) == selected
    assert len({r["group_id"] for r in selected}) == 8
    assert all(r["chart_key"].endswith("-a") for r in selected)
    assert all(assignments[r["source_sha256"]]["split"] == "train" for r in selected)
    rows[0]["split"] = "test"
    with pytest.raises(ContractError, match="assignment"):
        select_contexts(rows, assignments)


def test_mismatched_assets_stop_and_missing_assets_are_reported_without_network(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("The Stage 1 check must remain offline")
    monkeypatch.setattr("urllib.request.urlopen", no_network)
    root = tmp_path / "prepared"
    root.mkdir()
    summary = {"revision": "wrong", "manifest_sha256": "wrong", "split_sha256": "wrong", "data_contract_ready": True}
    (root / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ContractError, match="mismatch"):
        load_contexts(root, tmp_path)
    output = tmp_path / "run"
    with pytest.raises(FileNotFoundError):
        run_smoke(tmp_path / "absent", tmp_path, output, device="cpu")
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed" and report["updates"] == []
    with pytest.raises(FileExistsError):
        run_smoke(root, tmp_path, output, device="cpu")
