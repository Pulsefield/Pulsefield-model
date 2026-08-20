from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pulsefield_model.timing.evaluation.exp013_pilot import (
    EXP013_FROZEN_INFERENCE_SCHEMA,
    EXP013_PILOT_RESULT_SCHEMA,
    run_exp013_pilot,
)
from pulsefield_model.timing.evaluation.exp014_pilot import (
    EXP014_FROZEN_INFERENCE_SCHEMA,
    EXP014_INFERENCE_FAMILY,
    EXP014_PILOT_RESULT_SCHEMA,
    EXP014_PILOT_SUMMARY_SCHEMA,
    run_exp014_pilot,
)
from pulsefield_model.timing.v3.analytic_curve import (
    ConstantTempoSection,
    PhaseContinuousTimingCurve,
)
from pulsefield_model.timing.v3.tempo_track import (
    TempoTrackDiagnostics,
    TempoTrackProductionSelection,
    TempoTrackResult,
    TimingCandidateDiagnostic,
)


def test_exp013_filter_selects_high_medium_nonambiguous_stable_and_jump(
    tmp_path: Path,
) -> None:
    pilot = tmp_path / "pilot.jsonl"
    _write_jsonl(
        pilot,
        [
            _pilot_row("stable-high", stratum="stable", confidence="high"),
            _pilot_row("jump-medium", stratum="jump_candidate", confidence="medium"),
            _pilot_row("stable-low", stratum="stable", confidence="low"),
            _pilot_row(
                "stable-ambiguous",
                stratum="stable",
                confidence="high",
                ambiguous=True,
            ),
            _pilot_row("dense-high", stratum="dense", confidence="high"),
        ],
    )

    summary = run_exp013_pilot(
        pilot_jsonl_path=pilot,
        output_jsonl_path=tmp_path / "out.jsonl",
        summary_json_path=tmp_path / "summary.json",
        explicit_cache_audio_keys=["jump-medium"],
        **_accepted_dependencies(),
    )

    rows = _read_jsonl(tmp_path / "out.jsonl")
    assert summary["row_count"] == 1
    assert rows[0]["cache_audio_key"] == "jump-medium"
    assert rows[0]["stratum"] == "jump_candidate"
    assert rows[0]["product_status"] == "v3_accepted"


def test_exp014_runner_uses_exp014_schema_and_inference_family(tmp_path: Path) -> None:
    pilot = tmp_path / "pilot.jsonl"
    _write_jsonl(pilot, [_pilot_row("accepted", stratum="stable")])

    summary = run_exp014_pilot(
        pilot_jsonl_path=pilot,
        output_jsonl_path=tmp_path / "out.jsonl",
        summary_json_path=tmp_path / "summary.json",
        **_accepted_dependencies(),
    )

    row = _read_jsonl(tmp_path / "out.jsonl")[0]
    assert summary["schema"] == EXP014_PILOT_SUMMARY_SCHEMA
    assert row["schema"] == EXP014_PILOT_RESULT_SCHEMA
    assert row["frozen_inference"]["schema"] == EXP014_FROZEN_INFERENCE_SCHEMA
    assert row["frozen_inference"]["inference_family"] == EXP014_INFERENCE_FAMILY


def test_frozen_payload_sha_is_computed_before_weak_oracle_and_hides_label_data(
    tmp_path: Path,
) -> None:
    pilot = tmp_path / "pilot.jsonl"
    _write_jsonl(
        pilot,
        [_pilot_row("accepted", stratum="stable", beatmap_path="/oracle/path.osu")],
    )
    events: list[str] = []

    def frozen_observer(payload: dict[str, Any], frozen_sha256: str) -> None:
        dumped = json.dumps(payload, sort_keys=True)
        assert payload["schema"] == EXP013_FROZEN_INFERENCE_SCHEMA
        assert frozen_sha256
        assert "representative_redline_grid" not in dumped
        assert "/oracle/path.osu" not in dumped
        events.append("frozen")

    def weak_oracle_loader(path: Path) -> object:
        assert events == ["frozen"]
        assert path == Path("/oracle/path.osu")
        events.append("oracle")
        return object()

    dependencies = _accepted_dependencies()
    dependencies["weak_oracle_loader"] = weak_oracle_loader
    run_exp013_pilot(
        pilot_jsonl_path=pilot,
        output_jsonl_path=tmp_path / "out.jsonl",
        summary_json_path=tmp_path / "summary.json",
        frozen_payload_observer=frozen_observer,
        **dependencies,
    )

    assert events == ["frozen", "oracle"]


def test_v2_fallback_does_not_load_or_score_weak_oracle(tmp_path: Path) -> None:
    pilot = tmp_path / "pilot.jsonl"
    _write_jsonl(pilot, [_pilot_row("fallback", stratum="jump_candidate")])

    def fail_oracle(path: Path) -> object:
        raise AssertionError(f"oracle should not load for fallback: {path}")

    def fail_metrics(*args: object, **kwargs: object) -> object:
        raise AssertionError("metrics should not run for fallback")

    dependencies = _base_dependencies()
    dependencies["candidate_generator"] = (
        lambda prediction, *, audio_evidence: _tempo_result("v2_fallback")
    )
    dependencies["weak_oracle_loader"] = fail_oracle
    dependencies["metric_evaluator"] = fail_metrics
    run_exp013_pilot(
        pilot_jsonl_path=pilot,
        output_jsonl_path=tmp_path / "out.jsonl",
        summary_json_path=tmp_path / "summary.json",
        **dependencies,
    )

    row = _read_jsonl(tmp_path / "out.jsonl")[0]
    assert row["product_status"] == "v2_fallback"
    assert row["selected_candidate_index"] is None
    assert row["weak_oracle_evaluation"]["selected_metrics"] is None
    assert (
        row["weak_oracle_evaluation"]["unavailable_reason"]
        == "product_status_v2_fallback"
    )


def test_weak_oracle_failure_does_not_rewrite_accepted_inference_to_hard_failure(
    tmp_path: Path,
) -> None:
    pilot = tmp_path / "pilot.jsonl"
    _write_jsonl(pilot, [_pilot_row("accepted", stratum="stable")])

    def failing_oracle(path: Path) -> object:
        raise RuntimeError("oracle unavailable")

    dependencies = _accepted_dependencies()
    dependencies["weak_oracle_loader"] = failing_oracle
    summary = run_exp013_pilot(
        pilot_jsonl_path=pilot,
        output_jsonl_path=tmp_path / "out.jsonl",
        summary_json_path=tmp_path / "summary.json",
        **dependencies,
    )

    row = _read_jsonl(tmp_path / "out.jsonl")[0]
    assert row["product_status"] == "v3_accepted"
    assert row["failure_stage"] is None
    assert row["weak_oracle_evaluation"]["available"] is False
    assert (
        row["weak_oracle_evaluation"]["unavailable_reason"]
        == "weak_oracle_evaluation_error"
    )
    assert "oracle unavailable" in row["weak_oracle_evaluation"]["error"]
    assert summary["status_counts"] == {"v3_accepted": 1}


def test_row_exception_is_isolated_as_hard_failure(tmp_path: Path) -> None:
    pilot = tmp_path / "pilot.jsonl"
    _write_jsonl(
        pilot,
        [
            _pilot_row("bad", stratum="stable"),
            _pilot_row("good", stratum="stable"),
        ],
    )

    def cache_loader(cache_audio_key: str) -> _FakePrediction:
        return _FakePrediction(cache_audio_key)

    def generator(prediction: _FakePrediction, *, audio_evidence: object) -> TempoTrackResult:
        if prediction.cache_audio_key == "bad":
            raise RuntimeError("synthetic row failure")
        return _tempo_result("v3_accepted")

    dependencies = _base_dependencies()
    dependencies["beatthis_cache_loader"] = cache_loader
    dependencies["candidate_generator"] = generator
    summary = run_exp013_pilot(
        pilot_jsonl_path=pilot,
        output_jsonl_path=tmp_path / "out.jsonl",
        summary_json_path=tmp_path / "summary.json",
        **dependencies,
    )

    rows = _read_jsonl(tmp_path / "out.jsonl")
    assert [row["product_status"] for row in rows] == ["hard_failure", "v3_accepted"]
    assert rows[0]["failure_stage"] == "inference"
    assert "synthetic row failure" in rows[0]["error"]
    assert summary["status_counts"] == {"hard_failure": 1, "v3_accepted": 1}


def test_baseline_v2_is_parsed_only_after_frozen_sha_and_fallback_is_not_scored(
    tmp_path: Path,
) -> None:
    pilot = tmp_path / "pilot.jsonl"
    baseline = tmp_path / "baseline.jsonl"
    _write_jsonl(
        pilot,
        [
            _pilot_row("accepted", stratum="stable"),
            _pilot_row("fallback", stratum="jump_candidate"),
        ],
    )
    events: list[str] = []
    baseline_snapshot_sha: str | None = None

    def generator(prediction: _FakePrediction, *, audio_evidence: object) -> TempoTrackResult:
        if prediction.cache_audio_key == "fallback":
            return _tempo_result("v2_fallback")
        return _tempo_result("v3_accepted")

    def frozen_observer(payload: dict[str, Any], frozen_sha256: str) -> None:
        nonlocal baseline_snapshot_sha
        if not baseline.exists():
            _write_jsonl(
                baseline,
                [
                    _baseline_row("accepted"),
                    _baseline_row("fallback"),
                ],
            )
            baseline_snapshot_sha = hashlib.sha256(baseline.read_bytes()).hexdigest()
        elif payload["cache_audio_key"] == "fallback":
            _write_jsonl(baseline, [_baseline_row("mutated-after-snapshot")])
        events.append(f"frozen:{payload['cache_audio_key']}")

    dependencies = _base_dependencies()
    dependencies["beatthis_cache_loader"] = lambda key: _FakePrediction(key)
    dependencies["candidate_generator"] = generator
    summary = run_exp013_pilot(
        pilot_jsonl_path=pilot,
        baseline_v2_jsonl_path=baseline,
        output_jsonl_path=tmp_path / "out.jsonl",
        summary_json_path=tmp_path / "summary.json",
        frozen_payload_observer=frozen_observer,
        **dependencies,
    )

    rows = _read_jsonl(tmp_path / "out.jsonl")
    accepted_eval = rows[0]["weak_oracle_evaluation"]
    fallback_eval = rows[1]["weak_oracle_evaluation"]
    assert events == ["frozen:accepted", "frozen:fallback"]
    assert accepted_eval["baseline_v2"]["available"] is True
    assert accepted_eval["baseline_v2"]["ok"] is True
    assert accepted_eval["baseline_v2"]["status"] == "ok"
    assert accepted_eval["baseline_v2"]["metrics_available"] is True
    assert accepted_eval["baseline_v2"]["metrics"]["weak_oracle_phase_p90_ms"] == 12.0
    assert fallback_eval["baseline_v2"]["available"] is True
    assert fallback_eval["baseline_v2"]["ok"] is True
    assert fallback_eval["baseline_v2"]["metrics"] is None
    assert summary["input_baseline_v2_jsonl_sha256"] == baseline_snapshot_sha
    assert summary["fallback_rate"] == 0.5
    assert summary["status_counts_by_stratum"] == {
        "jump_candidate": {"v2_fallback": 1},
        "stable": {"v3_accepted": 1},
    }


def test_pilot_summary_sha_uses_initial_parse_snapshot(tmp_path: Path) -> None:
    pilot = tmp_path / "pilot.jsonl"
    _write_jsonl(pilot, [_pilot_row("initial", stratum="stable")])
    pilot_snapshot_sha = hashlib.sha256(pilot.read_bytes()).hexdigest()

    def frozen_observer(payload: dict[str, Any], frozen_sha256: str) -> None:
        assert payload["cache_audio_key"] == "initial"
        _write_jsonl(pilot, [_pilot_row("mutated", stratum="stable")])

    summary = run_exp013_pilot(
        pilot_jsonl_path=pilot,
        output_jsonl_path=tmp_path / "out.jsonl",
        summary_json_path=tmp_path / "summary.json",
        frozen_payload_observer=frozen_observer,
        **_accepted_dependencies(),
    )

    rows = _read_jsonl(tmp_path / "out.jsonl")
    assert rows[0]["cache_audio_key"] == "initial"
    assert summary["input_pilot_jsonl_sha256"] == pilot_snapshot_sha
    assert hashlib.sha256(pilot.read_bytes()).hexdigest() != pilot_snapshot_sha
    assert summary["output_jsonl_sha256"] == hashlib.sha256(
        (tmp_path / "out.jsonl").read_bytes()
    ).hexdigest()


def test_baseline_failure_status_uses_real_ok_failure_schema(tmp_path: Path) -> None:
    pilot = tmp_path / "pilot.jsonl"
    baseline = tmp_path / "baseline.jsonl"
    _write_jsonl(pilot, [_pilot_row("failed-baseline", stratum="jump_candidate")])
    _write_jsonl(
        baseline,
        [
            {
                "audio_key": "failed-baseline",
                "ok": False,
                "failure_stage": "cache",
                "error_type": "FileNotFoundError",
                "error": "missing cache",
                "fit": None,
            }
        ],
    )

    dependencies = _base_dependencies()
    dependencies["candidate_generator"] = (
        lambda prediction, *, audio_evidence: _tempo_result("v2_fallback")
    )
    run_exp013_pilot(
        pilot_jsonl_path=pilot,
        baseline_v2_jsonl_path=baseline,
        output_jsonl_path=tmp_path / "out.jsonl",
        summary_json_path=tmp_path / "summary.json",
        **dependencies,
    )

    row = _read_jsonl(tmp_path / "out.jsonl")[0]
    baseline_payload = row["weak_oracle_evaluation"]["baseline_v2"]
    assert baseline_payload["available"] is True
    assert baseline_payload["ok"] is False
    assert baseline_payload["status"] == "failed"
    assert baseline_payload["failure_stage"] == "cache"
    assert baseline_payload["error_type"] == "FileNotFoundError"
    assert baseline_payload["metrics"] is None


def test_summary_reports_absolute_endpoint_drift_without_hiding_signed_value(
    tmp_path: Path,
) -> None:
    pilot = tmp_path / "pilot.jsonl"
    _write_jsonl(
        pilot,
        [
            _pilot_row("negative-drift", stratum="stable"),
            _pilot_row("positive-drift", stratum="stable"),
        ],
    )
    endpoint_drifts = iter([-100.0, 20.0])

    class DriftMetrics:
        def to_dict(self) -> dict[str, float]:
            return _metrics_payload(endpoint_relative_drift_ms=next(endpoint_drifts))

    dependencies = _accepted_dependencies()
    dependencies["metric_evaluator"] = lambda *args, **kwargs: DriftMetrics()
    summary = run_exp013_pilot(
        pilot_jsonl_path=pilot,
        output_jsonl_path=tmp_path / "out.jsonl",
        summary_json_path=tmp_path / "summary.json",
        **dependencies,
    )

    metrics = summary["accepted_weak_metrics"]
    assert metrics["weak_oracle_endpoint_relative_drift_ms"]["mean"] == -40.0
    assert metrics["weak_oracle_endpoint_abs_relative_drift_ms"]["mean"] == 60.0
    assert metrics["weak_oracle_endpoint_abs_relative_drift_ms"]["max"] == 100.0


@pytest.mark.parametrize(
    ("status", "selection_patch", "expected_error"),
    (
        (
            "v2_fallback",
            {"selected_candidate_index": 0},
            "v2_fallback selection cannot carry selected candidate",
        ),
        (
            "v3_accepted",
            {"selected_fingerprint_sha256": "wrong-fingerprint"},
            "v3_accepted selected fingerprint mismatch",
        ),
    ),
)
def test_runner_rejects_invalid_production_selection_invariants(
    tmp_path: Path,
    status: str,
    selection_patch: dict[str, object],
    expected_error: str,
) -> None:
    pilot = tmp_path / "pilot.jsonl"
    _write_jsonl(pilot, [_pilot_row("bad-selection", stratum="stable")])

    def generator(prediction: _FakePrediction, *, audio_evidence: object) -> TempoTrackResult:
        result = _tempo_result(status)
        assert result.production_selection is not None
        return replace(
            result,
            production_selection=replace(
                result.production_selection,
                **selection_patch,
            ),
        )

    dependencies = _base_dependencies()
    dependencies["candidate_generator"] = generator
    summary = run_exp013_pilot(
        pilot_jsonl_path=pilot,
        output_jsonl_path=tmp_path / "out.jsonl",
        summary_json_path=tmp_path / "summary.json",
        **dependencies,
    )

    row = _read_jsonl(tmp_path / "out.jsonl")[0]
    assert row["product_status"] == "hard_failure"
    assert row["failure_stage"] == "inference"
    assert expected_error in row["error"]
    assert summary["status_counts"] == {"hard_failure": 1}


def test_outputs_are_stable_json_serializable(tmp_path: Path) -> None:
    pilot = tmp_path / "pilot.jsonl"
    _write_jsonl(pilot, [_pilot_row("serializable", stratum="stable")])

    run_exp013_pilot(
        pilot_jsonl_path=pilot,
        output_jsonl_path=tmp_path / "out.jsonl",
        summary_json_path=tmp_path / "summary.json",
        **_accepted_dependencies(),
    )

    row_payload = json.loads((tmp_path / "out.jsonl").read_text().splitlines()[0])
    summary_payload = json.loads((tmp_path / "summary.json").read_text())
    assert row_payload["schema"] == EXP013_PILOT_RESULT_SCHEMA
    assert row_payload["frozen_inference_sha256"]
    assert row_payload["selected_curve"]["sections"]
    assert summary_payload["accepted_weak_metrics"]["weak_oracle_phase_p90_ms"][
        "mean"
    ] == 12.0


@pytest.mark.parametrize("collision", ("pilot", "baseline"))
def test_runner_rejects_output_collision_with_inputs(
    tmp_path: Path,
    collision: str,
) -> None:
    pilot = tmp_path / "pilot.jsonl"
    baseline = tmp_path / "baseline.jsonl"
    _write_jsonl(pilot, [_pilot_row("collision", stratum="stable")])
    _write_jsonl(baseline, [_baseline_row("collision")])
    output = pilot if collision == "pilot" else baseline

    with pytest.raises(ValueError, match="must differ"):
        run_exp013_pilot(
            pilot_jsonl_path=pilot,
            baseline_v2_jsonl_path=baseline,
            output_jsonl_path=output,
            summary_json_path=tmp_path / "summary.json",
            **_accepted_dependencies(),
        )

    assert json.loads(pilot.read_text().splitlines()[0])["source"][
        "cache_audio_key"
    ] == "collision"
    assert json.loads(baseline.read_text().splitlines()[0])["audio_key"] == "collision"


def test_selected_malformed_row_isolated_as_input_schema_hard_failure(
    tmp_path: Path,
) -> None:
    pilot = tmp_path / "pilot.jsonl"
    malformed = _pilot_row("malformed", stratum="stable")
    malformed.pop("resolved_audio_path")
    _write_jsonl(
        pilot,
        [malformed, _pilot_row("good", stratum="stable")],
    )

    summary = run_exp013_pilot(
        pilot_jsonl_path=pilot,
        output_jsonl_path=tmp_path / "out.jsonl",
        summary_json_path=tmp_path / "summary.json",
        **_accepted_dependencies(),
    )

    rows = _read_jsonl(tmp_path / "out.jsonl")
    assert [row["product_status"] for row in rows] == [
        "hard_failure",
        "v3_accepted",
    ]
    assert rows[0]["failure_stage"] == "input_schema"
    assert "resolved_audio_path" in rows[0]["error"]
    assert summary["hard_failure_count"] == 1


def _accepted_dependencies() -> dict[str, object]:
    dependencies = _base_dependencies()
    dependencies["candidate_generator"] = (
        lambda prediction, *, audio_evidence: _tempo_result("v3_accepted")
    )
    return dependencies


def _base_dependencies() -> dict[str, object]:
    return {
        "beatthis_cache_loader": lambda key: _FakePrediction(key),
        "beatthis_cache_path_resolver": lambda key: Path(f"/cache/{key}.npz"),
        "mel_loader": lambda audio_path, *, repo_root: (
            np.zeros((24, 80), dtype=np.float32),
            "fixture_mel",
            None,
        ),
        "raw_evidence_extractor": lambda mel, *, audio_duration_seconds: object(),
        "weak_oracle_loader": lambda path: object(),
        "metric_evaluator": lambda *args, **kwargs: _FakeMetrics(),
    }


def _tempo_result(status: str) -> TempoTrackResult:
    curve = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(ConstantTempoSection(0, 32, 120.0),),
    )
    selected = 0 if status == "v3_accepted" else None
    return TempoTrackResult(
        observations=(),
        candidates=(curve,),
        candidate_diagnostics=(
            TimingCandidateDiagnostic(
                fingerprint_sha256=curve.fingerprint_sha256,
                curve_class=curve.curve_class,
                source="fixture",
                generation_score=1.0,
            ),
        ),
        diagnostics=TempoTrackDiagnostics(
            version="fixture",
            beat_peak_count=0,
            raw_boundary_count=0,
            pair_seed_count=0,
            shared_start_beat=0,
            shared_end_beat=32,
            primary_origin_time_ms=0.0,
            primary_bpm=120.0,
            candidate_count=1,
        ),
        production_selection=TempoTrackProductionSelection(
            status=status,  # type: ignore[arg-type]
            selected_candidate_index=selected,
            selected_fingerprint_sha256=None if selected is None else curve.fingerprint_sha256,
            lane="constant" if selected is not None else "fallback",
            fallback_reason=None if selected is not None else "fixture_fallback",
            raw_run=None,
            eligible_candidate_indices=(0,) if selected is not None else (),
            raw_self_rank_by_candidate=((0, 1),),
            beatthis_aba_rank_by_candidate=(),
        ),
    )


def _pilot_row(
    cache_audio_key: str,
    *,
    stratum: str,
    confidence: str = "high",
    ambiguous: bool = False,
    beatmap_path: str = "/fixture/oracle.osu",
) -> dict[str, object]:
    return {
        "resolved_audio_path": f"/fixture/audio/{cache_audio_key}.mp3",
        "source": {
            "cache_audio_key": cache_audio_key,
            "cache_duration_seconds": 20.0,
        },
        "label": {
            "stratum": stratum,
            "confidence": confidence,
            "ambiguous": ambiguous,
        },
        "representative_redline_grid": {
            "beatmap_path": beatmap_path,
            "agreement_rate": 1.0,
            "evidence_class": stratum,
        },
    }


def _baseline_row(cache_audio_key: str) -> dict[str, object]:
    return {
        "audio_key": cache_audio_key,
        "status": "ok",
        "fit": {
            "predicted_segments": [
                {"offset_ms": 0.0, "beat_length_ms": 500.0, "meter": 4},
            ],
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class _FakePrediction:
    def __init__(self, cache_audio_key: str) -> None:
        self.cache_audio_key = cache_audio_key


class _FakeMetrics:
    def to_dict(self) -> dict[str, float]:
        return _metrics_payload()


def _metrics_payload(
    *,
    endpoint_relative_drift_ms: float = 2.0,
) -> dict[str, float]:
    return {
        "weak_oracle_phase_mean_ms": 6.0,
        "weak_oracle_phase_p50_ms": 7.0,
        "weak_oracle_phase_p90_ms": 12.0,
        "weak_oracle_phase_max_ms": 18.0,
        "weak_oracle_local_bpm_alias_mae": 0.5,
        "weak_oracle_endpoint_relative_drift_ms": endpoint_relative_drift_ms,
        "weak_oracle_max_abs_prefix_relative_drift_ms": 3.0,
    }
