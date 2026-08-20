from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pytest

from pulsefield_model.timing.evaluation import checkpoint_divergence as checkpoint_eval


def test_failed_checkpoint_is_not_comparable_divergence() -> None:
    selected = [_selected_item()]
    results_by_checkpoint = {
        "final0": [_ok_result("final0", phase_error_ms=50.0)],
        "final1": [_failed_result("final1")],
        "final2": [_ok_result("final2", phase_error_ms=20.0)],
    }

    report = checkpoint_eval.compute_divergence_report(
        selected=selected,
        results_by_checkpoint=results_by_checkpoint,
        checkpoints=["final0", "final1", "final2"],
    )

    assert report["attempted_all_checkpoint_count"] == 1
    assert report["complete_map_count"] == 0
    assert report["failed_or_missing_map_count"] == 1
    assert report["material_phase_divergence_count"] == 0
    assert report["material_phase_divergence_rate"] is None


def test_selection_summary_records_underfilled_strata() -> None:
    selected = [_selected_item(selection_slice="multi_bpm")]
    summary = checkpoint_eval.selection_summary(
        selected,
        metadata={
            "requested_mixed_maps": 2,
            "requested_multi_bpm_maps": 2,
            "selected_mixed_maps": 0,
            "selected_multi_bpm_maps": 1,
            "underfilled_mixed_maps": 2,
            "underfilled_multi_bpm_maps": 1,
        },
    )

    assert summary["selected"] == 1
    assert summary["underfilled_mixed_maps"] == 2
    assert summary["underfilled_multi_bpm_maps"] == 1


def test_resume_retries_failed_rows_and_rejects_mismatch() -> None:
    args = argparse.Namespace(
        index=Path("index.parquet"),
        dataset_root=Path("dataset"),
        mixed_maps=1,
        multi_bpm_maps=1,
        max_duration_seconds=130.0,
        sample_seed="seed",
        device="mps",
        float16=True,
        divergence_phase_threshold_ms=8.0,
        high_error_phase_threshold_ms=45.0,
    )
    payload = checkpoint_eval.initial_payload(
        args=args,
        selected=[_selected_item()],
        checkpoints=["final0"],
        selection_metadata={},
    )
    existing_payload = copy.deepcopy(payload)
    existing_payload["results"]["final0"] = [_failed_result("final0")]

    merged = checkpoint_eval.merge_existing_results(payload, existing_payload)

    assert merged["results"]["final0"] == []

    mismatched_payload = copy.deepcopy(existing_payload)
    mismatched_payload["experiment"]["runtime"]["float16"] = False
    with pytest.raises(ValueError, match="mismatched fields"):
        checkpoint_eval.merge_existing_results(payload, mismatched_payload)


def _selected_item(*, selection_slice: str = "multi_bpm") -> dict[str, object]:
    return {
        "selection_key": "set:map:audio.mp3",
        "selection_slice": selection_slice,
        "beatmap_set_id": 1,
        "beatmap_id": 2,
        "title": "Song",
        "artist": "Artist",
        "creator": "Creator",
        "version": "4K",
        "audio_path": "audio.mp3",
        "beatmap_path": "map.osu",
        "duration_seconds": 60.0,
        "oracle_segment_count": 1,
        "oracle_unique_bpms": [120.0],
        "is_multi_bpm": selection_slice == "multi_bpm",
    }


def _ok_result(checkpoint_path: str, *, phase_error_ms: float) -> dict[str, object]:
    return {
        **_selected_item(),
        "ok": True,
        "checkpoint_path": checkpoint_path,
        "prediction_checkpoint_path": checkpoint_path,
        "prediction_seconds": 1.0,
        "fit_seconds": 0.1,
        "total_seconds": 1.1,
        "frame_count": 100,
        "score": 0.5,
        "diagnostics": {},
        "predicted_segments": [
            {
                "offset_ms": 0.0,
                "beat_length_ms": 500.0,
                "bpm": 120.0,
                "meter": 4,
            }
        ],
        "metrics": {
            "frame_count": 100,
            "beat_pulse_mae": 0.0,
            "local_bpm_mae": 0.0,
            "local_bpm_alias_mae": 0.0,
            "mean_phase_error_beats": 0.0,
            "max_phase_error_beats": 0.0,
            "mean_phase_error_ms": phase_error_ms,
            "max_phase_error_ms": phase_error_ms,
        },
    }


def _failed_result(checkpoint_path: str) -> dict[str, object]:
    return {
        **_selected_item(),
        "ok": False,
        "checkpoint_path": checkpoint_path,
        "error": "RuntimeError: test failure",
        "total_seconds": 0.1,
    }
