"""Verify the portable source identities and recompute the included summaries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "source-manifest.json").read_text())
    for item in manifest["files"]:
        path = (root / item["path"]).resolve()
        assert path.is_relative_to(root)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], item["path"]
    data = json.loads((root / "evidence.json").read_text())
    population = data["population_audit"]["totals"]
    assert population["onset_rows"] + population["release_only_rows"] == population["event_rows"]
    assert sum(data["population_audit"]["endpoint_rank_histogram"]) == population["lns"]
    assert sum(data["population_audit"]["duration_histogram"]) == population["lns"]

    recomputed = {}
    for updates, record in data["endpoint_fits"].items():
        summary, groups = record["summary"], record["groups"]
        identities = sorted(groups["prior"])
        assert identities == sorted(groups["context"]) and len(identities) == 32
        for arm in ("prior", "context"):
            values = groups[arm]
            count = sum(values[key]["cases"] for key in identities)
            assert count == 1794
            for metric, output in (("nll", "group_mean_nll"), ("accuracy", "group_mean_accuracy")):
                np.testing.assert_allclose(
                    np.mean([values[key][metric] for key in identities]),
                    summary["metrics"][arm][output], rtol=0, atol=1e-12,
                )
            np.testing.assert_allclose(
                sum(values[key]["nll"] * values[key]["cases"] for key in identities) / count,
                summary["metrics"][arm]["per_head_nll"], rtol=0, atol=1e-12,
            )
        assert all(groups["prior"][key]["cases"] == groups["context"][key]["cases"] for key in identities)
        deltas = np.array([groups["context"][key]["nll"] - groups["prior"][key]["nll"] for key in identities])
        bootstrap = np.random.default_rng(summary["bootstrap_seed"]).choice(
            deltas, (summary["bootstrap_replicates"], len(deltas)), replace=True,
        ).mean(-1)
        interval = np.quantile(bootstrap, [.025, .975])
        np.testing.assert_allclose(deltas.mean(), summary["group_nll_delta"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(interval, summary["group_nll_delta_ci95"], rtol=0, atol=1e-12)
        assert int((deltas < 0).sum()) == summary["groups_improved"]
        accuracy_delta = summary["metrics"]["context"]["group_mean_accuracy"] - summary["metrics"]["prior"]["group_mean_accuracy"]
        np.testing.assert_allclose(accuracy_delta, summary["group_accuracy_delta"], rtol=0, atol=1e-12)
        assert bool(deltas.mean() <= -.05 and interval[1] < 0 and accuracy_delta >= -.05) == summary["criterion_met"]
        recomputed[updates] = {"delta": float(deltas.mean()), "ci95": interval.tolist()}

    old = {item["source_sha256"]: item for item in data["old_clock_seed17"]}
    for records in data["typed_onset_generation"].values():
        assert len(records) == 3
        for item in records:
            assert item["source_metrics"]["rows"] - item["omitted_candidates"] == item["organization"]["rows"]
            assert item["source_metrics"]["rows"] == old[item["source_sha256"]]["organization"]["rows"]
            assert item["required_onsets"] <= item["organization"]["rows"]
            assert item["joint_conditioned_rows"] == 0
            witness = item["minimum_four_attack_witness"]
            notes = witness["notes"]
            assert len(notes) == 4 and len({note["column"] for note in notes}) == 1
            assert notes[-1]["startMs"] - notes[0]["startMs"] == witness["span_ms"]
    print(json.dumps({"verified_source_files": len(manifest["files"]), "endpoint_bootstrap": recomputed}, indent=2))


if __name__ == "__main__":
    main()
