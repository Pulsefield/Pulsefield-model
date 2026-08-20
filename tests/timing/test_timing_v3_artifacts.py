from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pulsefield_model.timing.evaluation.artifacts import (
    audit_timing_v3_baseline_artifacts,
)


def test_audit_reconciles_inference_and_comparator_denominators(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.jsonl"
    _write_jsonl(
        inventory,
        [
            {"maps": [{}, {}]},
            {"maps": [{}]},
        ],
    )
    baseline = tmp_path / "baseline.jsonl"
    _write_jsonl(
        baseline,
        [
            _row("a", ok=True, comparisons=(True, False), stage=None),
            _row("b", ok=False, comparisons=(False,), stage="compare"),
        ],
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "source": {
                    "inventory_sha256": _sha256(inventory),
                    "unique_audio_count": 2,
                    "map_reference_count": 3,
                },
                "results": {
                    "result_count": 2,
                    "successful_audio_count": 1,
                    "failed_audio_count": 1,
                    "fit_success_audio_count": 2,
                    "fit_failure_audio_count": 0,
                    "comparison_attempted_audio_count": 2,
                    "comparison_eligible_audio_count": 1,
                    "comparator_unavailable_audio_count": 1,
                    "paired_comparison_count": 1,
                    "comparison_failure_count": 2,
                },
                "failures": {
                    "failed_audio_count": 1,
                    "fit_failure_audio_count": 0,
                    "comparator_unavailable_audio_count": 1,
                    "comparison_failure_count": 2,
                    "stage_counts": {"compare": 1},
                },
            }
        ),
        encoding="utf-8",
    )

    audit = audit_timing_v3_baseline_artifacts(
        baseline,
        summary,
        inventory_jsonl_path=inventory,
    )

    assert audit.ok
    assert audit.fit_success_audio_count == 2
    assert audit.comparison_eligible_audio_count == 1
    assert audit.comparator_unavailable_audio_count == 1
    assert json.loads(json.dumps(audit.to_dict(), allow_nan=False))["ok"] is True
    assert audit.require_ok() is audit


def test_audit_rejects_stale_comparison_eligible_count(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    _write_jsonl(
        baseline,
        [
            _row("a", ok=True, comparisons=(True,), stage=None),
            _row("b", ok=False, comparisons=(False,), stage="compare"),
        ],
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "source": {"unique_audio_count": 2},
                "results": {
                    "result_count": 2,
                    "successful_audio_count": 1,
                    "failed_audio_count": 1,
                    "fit_success_audio_count": 2,
                    "fit_failure_audio_count": 0,
                    "comparison_attempted_audio_count": 2,
                    "comparison_eligible_audio_count": 2,
                    "comparator_unavailable_audio_count": 1,
                    "paired_comparison_count": 1,
                    "comparison_failure_count": 1,
                },
                "failures": {
                    "failed_audio_count": 1,
                    "fit_failure_audio_count": 0,
                    "comparator_unavailable_audio_count": 1,
                    "comparison_failure_count": 1,
                    "stage_counts": {"compare": 1},
                },
            }
        ),
        encoding="utf-8",
    )

    audit = audit_timing_v3_baseline_artifacts(baseline, summary)

    assert not audit.ok
    assert any("comparison_eligible_audio_count=2, recomputed=1" in error for error in audit.errors)
    with pytest.raises(ValueError, match="comparison_eligible_audio_count"):
        audit.require_ok()


def test_audit_reports_duplicate_audio_and_resume_keys(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    duplicate = _row("a", ok=True, comparisons=(True,), stage=None)
    _write_jsonl(baseline, [duplicate, duplicate])
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"source": {}, "results": {}, "failures": {}}), encoding="utf-8")

    audit = audit_timing_v3_baseline_artifacts(baseline, summary)

    assert any("audio_key values must be unique" in error for error in audit.errors)
    assert any("resume fingerprints must be unique" in error for error in audit.errors)


def _row(
    audio_key: str,
    *,
    ok: bool,
    comparisons: tuple[bool, ...],
    stage: str | None,
) -> dict[str, object]:
    return {
        "audio_key": audio_key,
        "resume": {"fingerprint": f"resume-{audio_key}"},
        "ok": ok,
        "fit": {"predicted_segments": []},
        "comparisons": [{"ok": value} for value in comparisons],
        "failure_stage": stage,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
