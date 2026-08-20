from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pulsefield_model.timing.evaluation.highres_boundary_v2 import (
    FEATURE_NAMES,
    FROZEN_RATIOS,
    HighresBoundaryV2Error,
    ROUTE_SCHEMA,
    TRUTH_SCHEMA,
    RunnerConfig,
    _candidate_seconds_from_grid,
    _extract_candidate_features,
    feature_names,
    preflight_run,
    self_check,
)


def test_highres_boundary_v2_feature_contract_is_exact_and_nonleaky() -> None:
    payload = self_check()

    assert payload["feature_count"] == 100
    assert payload["mlp_hidden_units"] == [96, 48]
    assert payload["candidate_margin_seconds"] == 8.0
    assert payload["candidate_hop_seconds"] == 0.1
    assert feature_names() == FEATURE_NAMES


def test_highres_boundary_v2_candidate_grid_uses_full_eight_second_margin() -> None:
    sample_times = np.arange(0, 401, dtype=np.float64) * 0.05

    candidates = _candidate_seconds_from_grid(
        sample_times,
        margin_seconds=8.0,
        candidate_hop_seconds=0.1,
    )

    assert candidates[0] == pytest.approx(8.0)
    assert candidates[-1] == pytest.approx(12.0)
    assert np.allclose(np.diff(candidates), 0.1)


def test_highres_boundary_v2_feature_matrix_shape_and_finiteness() -> None:
    sample_times = np.arange(0, 401, dtype=np.float64) * 0.05
    candidates = _candidate_seconds_from_grid(
        sample_times,
        margin_seconds=8.0,
        candidate_hop_seconds=0.1,
    )
    base = np.sin(sample_times, dtype=np.float64).astype(np.float32)
    signals = np.stack([base + np.float32(index) for index in range(10)], axis=1)

    features = _extract_candidate_features(
        sample_times=sample_times,
        signals=signals,
        candidates_seconds=candidates,
        sample_hop_seconds=0.05,
    )

    assert features.shape == (41, 100)
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))


def test_highres_boundary_v2_preflight_blocks_small_train_jump_slice(tmp_path: Path) -> None:
    train_routes = tmp_path / "train_routes.jsonl"
    holdout_routes = tmp_path / "holdout_routes.jsonl"
    train_truth = tmp_path / "train_truth.jsonl"
    _write_jsonl(
        train_routes,
        [
            {
                "schema": ROUTE_SCHEMA,
                "route_id": f"r_{index}",
                "audio_path": f"/tmp/r_{index}.wav",
            }
            for index in range(4)
        ],
    )
    _write_jsonl(
        holdout_routes,
        [
            {
                "schema": ROUTE_SCHEMA,
                "route_id": f"h_{index}",
                "audio_path": f"/tmp/h_{index}.wav",
            }
            for index in range(4)
        ],
    )
    _write_jsonl(
        train_truth,
        [
            {
                "schema": TRUTH_SCHEMA,
                "route_id": f"r_{index}",
                "split": "train",
                "transform": "jump",
                "target_rate_ratio": FROZEN_RATIOS[index % len(FROZEN_RATIOS)],
                "seam_output_seconds": 10.0,
                "source_key": f"source_{index}",
            }
            for index in range(4)
        ],
    )
    config = RunnerConfig(
        train_routes_jsonl=train_routes,
        holdout_routes_jsonl=holdout_routes,
        train_truth_jsonl=train_truth,
        min_train_jump_routes=128,
    )

    with pytest.raises(HighresBoundaryV2Error, match="at least 128 train jump routes"):
        preflight_run(config)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")

