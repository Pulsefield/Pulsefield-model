from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from pulsefield_model.timing.evaluation import exp004_protocol, exp004_runner
from pulsefield_model.timing.evaluation import exp004_weak_evidence as weak_evidence
from pulsefield_model.timing.evaluation.exp004_weak_evidence import (
    BASELINE_RESULT_SCHEMA,
    RESULT_SCHEMA,
    SUMMARY_SCHEMA,
    SYNTHETIC_FIXTURE_STAGE,
    main as weak_evidence_main,
    run_exp004_weak_evidence,
)
from pulsefield_model.timing.schema import FittedTimingGrid, TimingSegment
from pulsefield_model.timing.v3.schema import ConstantTimingSection, TimingV3Grid


def _grid(*segments: tuple[float, float]) -> FittedTimingGrid:
    return FittedTimingGrid(
        tuple(
            TimingSegment(offset_ms=offset_ms, beat_length_ms=beat_length_ms)
            for offset_ms, beat_length_ms in segments
        )
    )


def _v3_constant(*, bpm: float = 120.0, coverage_end_ms: float = 1000.0) -> TimingV3Grid:
    beat_length = 60000.0 / bpm
    beat_count = int(round(coverage_end_ms / beat_length))
    return TimingV3Grid(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(ConstantTimingSection(start_beat=0, end_beat=beat_count, bpm=bpm),),
        coverage_start_ms=0.0,
        coverage_end_ms=coverage_end_ms,
    )


def _v3_constant_with_first_boundary(
    *,
    first_boundary_ms: float,
    bpm: float = 120.0,
    coverage_end_ms: float = 1000.0,
) -> TimingV3Grid:
    beat_length = 60000.0 / bpm
    beat_count = int(round((coverage_end_ms - first_boundary_ms) / beat_length)) + 1
    return TimingV3Grid(
        origin_beat=1,
        origin_time_ms=first_boundary_ms + beat_length,
        sections=(ConstantTimingSection(start_beat=0, end_beat=beat_count, bpm=bpm),),
        coverage_start_ms=0.0,
        coverage_end_ms=coverage_end_ms,
    )


def _v3_jump() -> TimingV3Grid:
    return TimingV3Grid(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTimingSection(start_beat=0, end_beat=4, bpm=120.0),
            ConstantTimingSection(start_beat=4, end_beat=9, bpm=150.0),
        ),
        coverage_start_ms=0.0,
        coverage_end_ms=4000.0,
    )


def test_common_difficulty_audio_first_and_multi_map_weighting(tmp_path: Path) -> None:
    inputs = _write_weak_inputs(
        tmp_path,
        [
            _spec(
                "a",
                comparators=[_comparator("a-easy", _grid((0.0, 500.0))), _comparator("a-hard", _grid((0.0, 500.0)))],
            ),
            _spec(
                "b",
                comparators=[_comparator("b-map", _grid((250.0, 500.0)))],
            ),
        ],
    )

    summary = _run(inputs)

    assert summary["schema"] == SUMMARY_SCHEMA
    assert summary["denominators"]["comparison_eligible_count"] == 2
    assert summary["methods"]["CJ3"]["audio_count"] == 2
    assert summary["audio_aggregates"]["CJ3_alias_phase_mean_ms"]["mean"] == pytest.approx(125.0)
    assert summary["difficulty_aggregates"]["CJ3_alias_phase_mean_ms"]["mean"] == pytest.approx(250.0 / 3.0)
    assert summary["primary"]["mean_phase_ratio"]["value"] == pytest.approx(1.0)


def test_pure_safety_fallback_and_comparator_unavailable_denominators(tmp_path: Path) -> None:
    inputs = _write_weak_inputs(
        tmp_path,
        [
            _spec("pure", comparators=[_comparator("pure-map", _grid((0.0, 500.0)))]),
            _spec(
                "fallback-with-comparator",
                cj3_grid=None,
                selected_source="current_v2",
                selected_used_fallback=True,
                comparators=[_comparator("fallback-map", _grid((0.0, 500.0)))],
            ),
            _spec(
                "fallback-no-comparator",
                cj3_grid=None,
                selected_source="current_v2",
                selected_used_fallback=True,
                baseline_present=False,
            ),
        ],
    )

    summary = _run(inputs)
    rows = _read_jsonl(inputs["output"])

    assert summary["denominators"]["projection_evaluable_count"] == 3
    assert summary["denominators"]["comparison_eligible_count"] == 2
    assert summary["denominators"]["pure_CJ3_phase_count"] == 1
    assert summary["denominators"]["selected_safety_phase_count"] == 2
    assert summary["denominators"]["selected_fallback_count"] == 2
    assert summary["denominators"]["selected_fallback_rate"] == pytest.approx(2.0 / 3.0)
    no_comparator = rows[2]["denominator_row"]
    assert no_comparator["selected_used_fallback"] is True
    assert no_comparator["comparison_eligible"] is False
    assert no_comparator["selected_safety_phase_scored"] is False
    assert no_comparator["current_v2_phase_matched"] is False


def test_cj3_success_with_current_v2_fit_failure_stays_out_of_paired_phase_denominators(
    tmp_path: Path,
) -> None:
    inputs = _write_weak_inputs(
        tmp_path,
        [
            _spec(
                "cj3-only",
                current_v2_available=False,
                comparators=[_comparator("cj3-only-map", _grid((0.0, 500.0)))],
            )
        ],
    )

    summary = _run(inputs)
    row = _read_jsonl(inputs["output"])[0]

    assert row["methods"]["current_v2"]["comparison_matched"] is False
    assert row["methods"]["CJ3"]["comparison_matched"] is True
    assert row["methods"]["selected_CJ3_or_current_v2_fallback"]["comparison_matched"] is True
    assert row["denominator_row"]["current_v2_phase_matched"] is False
    assert row["denominator_row"]["pure_cj3_phase_matched"] is False
    assert row["denominator_row"]["selected_safety_phase_scored"] is False
    assert summary["denominators"]["comparison_eligible_count"] == 1
    assert summary["denominators"]["pure_CJ3_phase_count"] == 0
    assert summary["denominators"]["selected_safety_phase_count"] == 0


def test_zero_phase_ratios_use_frozen_both_zero_convention(tmp_path: Path) -> None:
    inputs = _write_weak_inputs(
        tmp_path,
        [_spec("zero", comparators=[_comparator("zero-map", _grid((0.0, 500.0)))])],
    )

    summary = _run(inputs)

    assert summary["primary"]["mean_phase_ratio"] == {
        "defined": True,
        "reason": "both_zero_convention",
        "value": 1.0,
    }
    assert summary["primary"]["p90_phase_ratio"] == {
        "defined": True,
        "reason": "both_zero_convention",
        "value": 1.0,
    }


def test_raw_alias_and_endpoint_left_limit_metrics_are_preserved(tmp_path: Path) -> None:
    inputs = _write_weak_inputs(
        tmp_path,
        [
            _spec(
                "alias",
                coverage_end_ms=1000.0,
                current_v2=_grid((0.0, 250.0)),
                cj3_grid=_v3_constant(bpm=240.0, coverage_end_ms=1000.0),
                comparators=[_comparator("alias-map", _grid((0.0, 500.0)))],
            ),
            _spec(
                "endpoint",
                coverage_end_ms=2000.0,
                current_v2=_grid((0.0, 501.0)),
                cj3_grid=_v3_constant(bpm=60000.0 / 501.0, coverage_end_ms=2004.0),
                comparators=[_comparator("endpoint-map", _grid((0.0, 500.0), (2000.0, 250.0)))],
            ),
        ],
    )

    _run(inputs)
    rows = _read_jsonl(inputs["output"])
    alias_metrics = rows[0]["methods"]["CJ3"]["difficulty_metrics"][0]
    endpoint_metrics = rows[1]["methods"]["current_v2"]["difficulty_metrics"][0]["raw"]

    assert alias_metrics["raw"]["local_bpm_mae"] == pytest.approx(120.0)
    assert alias_metrics["bpm-80-160"]["local_bpm_mae"] == pytest.approx(0.0)
    expected_endpoint_beats = 2000.0 / 501.0 - 4.0
    assert endpoint_metrics["endpoint_relative_drift_ms"] == pytest.approx(expected_endpoint_beats * 500.0)


def test_weak_boundary_consensus_uses_audio_first_difficulty_support(tmp_path: Path) -> None:
    inputs = _write_weak_inputs(
        tmp_path,
        [
            _spec(
                "jump",
                coverage_end_ms=4000.0,
                current_v2=_grid((0.0, 500.0)),
                cj3_grid=_v3_jump(),
                comparators=[
                    _comparator("easy", _grid((0.0, 500.0), (2005.0, 400.0))),
                    _comparator("hard", _grid((0.0, 500.0), (1990.0, 400.0))),
                    _comparator("stable", _grid((0.0, 500.0))),
                ],
            )
        ],
    )

    _run(inputs)
    row = _read_jsonl(inputs["output"])[0]

    boundary = row["methods"]["CJ3"]["weak_boundary_audio"]
    assert boundary["weak_consensus_supported_boundary_count"] == 1
    assert boundary["weak_consensus_boundaries"][0]["matched_valid_difficulty_count"] == 2


def test_resume_reuse_stale_baseline_and_stale_metrics_rows(tmp_path: Path) -> None:
    inputs = _write_weak_inputs(
        tmp_path,
        [_spec("resume", comparators=[_comparator("resume-map", _grid((0.0, 500.0)))])],
    )
    first = _run(inputs)
    assert first["resume"]["processed"] == 1

    second = _run(inputs)
    assert second["resume"]["reused_success"] == 1

    rows = _read_jsonl(inputs["output"])
    rows[0]["resume"]["fingerprint"] = "0" * 64
    _write_jsonl(inputs["output"], rows)
    stale_metric = _run(inputs)
    assert stale_metric["resume"]["recomputed_stale"] == 1

    baseline_rows = _read_jsonl(inputs["baseline"])
    baseline_rows[0]["comparisons"][0]["oracle_segments"][0]["offset_ms"] = 10.0
    _write_jsonl(inputs["baseline"], baseline_rows)
    stale_baseline = _run(inputs)
    assert stale_baseline["resume"]["recomputed_stale"] == 1


def test_projection_summary_sha_mismatch_and_duplicate_joins_fail_closed(tmp_path: Path) -> None:
    inputs = _write_weak_inputs(
        tmp_path,
        [_spec("dup-a", comparators=[]), _spec("dup-b", comparators=[])],
    )
    summary = _read_json(inputs["projection_summary"])
    summary["output"]["sha256"] = "0" * 64
    _write_json(inputs["projection_summary"], summary)
    with pytest.raises(ValueError, match="projection JSONL SHA"):
        _run(inputs)

    inputs = _write_weak_inputs(tmp_path / "baseline-dup", [_spec("a"), _spec("b")])
    baseline_rows = _read_jsonl(inputs["baseline"])
    baseline_rows[1]["audio_key"] = baseline_rows[0]["audio_key"]
    _write_jsonl(inputs["baseline"], baseline_rows)
    with pytest.raises(ValueError, match="duplicate baseline audio key"):
        _run(inputs)

    inputs = _write_weak_inputs(tmp_path / "projection-dup", [_spec("a"), _spec("b")])
    projection_rows = _read_jsonl(inputs["projection"])
    projection_rows[1]["identity"]["cache_audio_key"] = projection_rows[0]["identity"]["cache_audio_key"]
    _refingerprint_projection_row(projection_rows[1])
    _write_jsonl(inputs["projection"], projection_rows)
    projection_summary = _read_json(inputs["projection_summary"])
    projection_summary["output"]["sha256"] = _file_sha256(inputs["projection"])
    _write_json(inputs["projection_summary"], projection_summary)
    with pytest.raises(ValueError, match="duplicate projection audio key"):
        _run(inputs)


def test_malformed_oracle_is_isolated_without_dropping_valid_siblings(tmp_path: Path) -> None:
    inputs = _write_weak_inputs(
        tmp_path,
        [
            _spec(
                "malformed",
                comparators=[
                    _comparator("valid", _grid((0.0, 500.0))),
                    {"ok": True, "difficulty_key": "bad", "beatmap_path": "bad.osu", "oracle_segments": []},
                ],
            )
        ],
    )

    summary = _run(inputs)
    row = _read_jsonl(inputs["output"])[0]

    assert summary["denominators"]["comparison_eligible_count"] == 1
    assert row["baseline"]["invalid_comparator_count"] == 1
    assert row["methods"]["CJ3"]["difficulty_count"] == 1
    assert "malformed_or_unavailable_stored_comparator" in row["reasons"]


def test_finite_json_atomic_outputs_and_lock_fail_closed(tmp_path: Path) -> None:
    inputs = _write_weak_inputs(
        tmp_path,
        [_spec("finite", comparators=[_comparator("finite-map", _grid((0.0, 500.0)))])],
    )
    lock_path = inputs["output"].with_name(f".{inputs['output'].name}.lock")
    lock_path.write_text("owned", encoding="utf-8")
    with pytest.raises(RuntimeError, match="locked"):
        _run(inputs)
    lock_path.unlink()

    summary = _run(inputs)
    rows = _read_jsonl(inputs["output"])

    json.dumps(summary, allow_nan=False)
    json.dumps(rows, allow_nan=False)
    assert not list(tmp_path.glob("*.tmp"))
    assert not lock_path.exists()
    assert rows[0]["schema"] == RESULT_SCHEMA
    assert isinstance(rows[0]["result_fingerprint"], str)
    assert len(rows[0]["result_fingerprint"]) == 64


def test_result_fingerprint_tamper_and_projection_identity_stale_recompute(tmp_path: Path) -> None:
    inputs = _write_weak_inputs(
        tmp_path,
        [_spec("fingerprint", comparators=[_comparator("fingerprint-map", _grid((0.0, 500.0)))])],
    )
    _run(inputs)

    rows = _read_jsonl(inputs["output"])
    rows[0]["identity"]["audio_group_key"] = "tampered-group"
    _write_jsonl(inputs["output"], rows)

    summary = _run(inputs)
    repaired = _read_jsonl(inputs["output"])[0]

    assert summary["resume"]["recomputed_stale"] == 1
    assert repaired["identity"]["audio_group_key"] == "group-fingerprint"
    assert repaired["result_fingerprint"] == rows[0]["result_fingerprint"]


def test_projection_summary_path_stage_and_hard_guard_contracts(tmp_path: Path) -> None:
    inputs = _write_weak_inputs(tmp_path, [_spec("path-copy")])
    summary = _read_json(inputs["projection_summary"])
    copied_projection = tmp_path / "copy.jsonl"
    copied_projection.write_text(inputs["projection"].read_text(encoding="utf-8"), encoding="utf-8")
    summary["output"]["path"] = copied_projection.as_posix()
    _write_json(inputs["projection_summary"], summary)
    with pytest.raises(ValueError, match="output.path"):
        _run(inputs)

    formal = _write_weak_inputs(tmp_path / "formal", [_spec("formal")], stage="repair80")
    with pytest.raises(ValueError, match="repair80 projection JSONL must contain exactly 80 rows"):
        _run(formal)

    hard_guard = _write_weak_inputs(
        tmp_path / "hard-guard",
        [_spec("hard-guard")],
        projection_hard_guards_ok=False,
    )
    hard_guard_summary = _run(hard_guard)
    assert hard_guard_summary["decision"] == "kill"
    assert hard_guard_summary["next_action"] == "stop_projection_hard_guard_failure"
    assert hard_guard_summary["stage_gates"]["gates"]["projection_hard_guards"]["status"] == "kill"


def test_legacy_schema_and_mixed_oracle_segment_payloads_are_rejected_locally(tmp_path: Path) -> None:
    inputs = _write_weak_inputs(tmp_path / "legacy", [_spec("legacy")])
    baseline_rows = _read_jsonl(inputs["baseline"])
    baseline_rows[0]["schema"] = "pulsefield_model.timing_v3_exp004_v2_baseline_result_v1"
    _write_jsonl(inputs["baseline"], baseline_rows)
    with pytest.raises(ValueError, match="unsupported schema"):
        _run(inputs)

    inputs = _write_weak_inputs(
        tmp_path / "mixed-segment",
        [
            _spec(
                "mixed-segment",
                comparators=[
                    _comparator("valid", _grid((0.0, 500.0))),
                    {
                        "ok": True,
                        "difficulty_key": "mixed",
                        "beatmap_path": "mixed.osu",
                        "oracle_segments": [
                            {"offset_ms": 0.0, "beat_length_ms": 500.0, "meter": 4},
                            "not-a-segment",
                        ],
                    },
                    {
                        "ok": True,
                        "difficulty_key": "bad-meter",
                        "beatmap_path": "bad-meter.osu",
                        "oracle_segments": [{"offset_ms": 0.0, "beat_length_ms": 500.0, "meter": 0}],
                    },
                ],
            )
        ],
    )
    _run(inputs)
    row = _read_jsonl(inputs["output"])[0]
    assert row["baseline"]["valid_comparator_count"] == 1
    assert row["baseline"]["invalid_comparator_count"] == 2
    assert row["methods"]["CJ3"]["difficulty_count"] == 1


def test_stage_gates_long_zero_stable_kill_and_no_path_denominator(tmp_path: Path) -> None:
    long_zero = _write_weak_inputs(
        tmp_path / "long-zero",
        [_spec("ordinary", comparators=[_comparator("ordinary-map", _grid((0.0, 500.0)))])],
    )
    long_zero_summary = _run(long_zero)
    assert (
        long_zero_summary["stage_gates"]["gates"]["long_max_prefix_drift_mean_ratio"]["status"]
        == "not_applicable"
    )

    stable_specs = [
        _spec(
            f"stable-{index}",
            coverage_end_ms=4000.0,
            current_v2=_grid((5.0, 500.0)),
            cj3_grid=_v3_constant_with_first_boundary(first_boundary_ms=-100.0, coverage_end_ms=4000.0),
            comparators=[_comparator(f"stable-{index}-map", _grid((0.0, 500.0)))],
            strata={"pilot_stratum": "stable"},
        )
        for index in range(5)
    ]
    stable_summary = _run(_write_weak_inputs(tmp_path / "stable-kill", stable_specs))
    assert stable_summary["stage_gates"]["gates"]["stable_mean_phase_ratio"]["status"] == "kill"

    no_path = _write_weak_inputs(
        tmp_path / "no-path",
        [
            _spec(
                "no-path",
                cj3_grid=None,
                selected_source=None,
                projection_evaluable=False,
                candidate_status="failed",
                candidate_reason="candidate_extraction_failure",
                baseline_present=False,
            )
        ],
    )
    no_path_summary = _run(no_path)
    gate = no_path_summary["stage_gates"]["gates"]["no_path_plus_candidate_extraction_failure_rate"]
    assert gate["numerator"] == 1
    assert gate["denominator"] == 1
    assert no_path_summary["denominator_policy"]["no_path_plus_candidate_extraction_failure_rate"]["denominator"] == "cache_valid_count"


def test_strata_and_cj1_cj2_jump_ablation_gates(tmp_path: Path) -> None:
    specs = []
    for index in range(15):
        specs.append(
            _spec(
                f"jump-{index}",
                coverage_end_ms=4000.0,
                current_v2=_grid((5.0, 500.0)),
                cj3_grid=_v3_jump(),
                variant_grids={
                    "CJ1": _v3_constant_with_first_boundary(first_boundary_ms=-5.0, coverage_end_ms=4000.0),
                    "CJ2": _v3_constant_with_first_boundary(first_boundary_ms=-5.0, coverage_end_ms=4000.0),
                },
                comparators=[_comparator(f"jump-{index}-map", _grid((0.0, 500.0), (2000.0, 400.0)))],
                strata={
                    "pilot_stratum": "jump_candidate",
                    "label_confidence": 0.9,
                    "source_long_track": index == 0,
                },
            )
        )
    inputs = _write_weak_inputs(tmp_path, specs)
    summary = _run(inputs)
    rows = _read_jsonl(inputs["output"])

    assert rows[0]["strata"]["primary_stratum"] == "jump"
    assert rows[0]["strata"]["beatthis_evidence_tercile"] == "unavailable"
    assert rows[0]["strata"]["predicted_jump_bin"] == "1"
    assert "jump" in summary["strata"]["by_primary_stratum"]
    assert summary["strata"]["by_primary_stratum"]["jump"]["paired_audio_count"] == 15
    assert summary["stage_gates"]["gates"]["jump_ablation_CJ3_vs_CJ1"]["denominator"]["sample_count"] == 15
    assert summary["stage_gates"]["gates"]["jump_ablation_CJ3_vs_CJ2"]["denominator"]["sample_count"] == 15


def test_cli_main_writes_summary(tmp_path: Path) -> None:
    inputs = _write_weak_inputs(
        tmp_path,
        [_spec("cli", comparators=[_comparator("cli-map", _grid((0.0, 500.0)))])],
    )

    assert weak_evidence_main(
        [
            "--projection-jsonl",
            inputs["projection"].as_posix(),
            "--projection-summary",
            inputs["projection_summary"].as_posix(),
            "--baseline-jsonl",
            inputs["baseline"].as_posix(),
            "--output-jsonl",
            inputs["output"].as_posix(),
            "--summary-json",
            inputs["summary"].as_posix(),
        ]
    ) == 0
    assert _read_json(inputs["summary"])["schema"] == SUMMARY_SCHEMA


def test_projection_row_fingerprint_and_formal_ready_fail_closed(tmp_path: Path) -> None:
    inputs = _write_weak_inputs(tmp_path / "projection-tamper", [_spec("tamper")])
    rows = _read_jsonl(inputs["projection"])
    rows[0]["row_complete"] = False
    _write_jsonl(inputs["projection"], rows)
    projection_summary = _read_json(inputs["projection_summary"])
    projection_summary["output"]["sha256"] = _file_sha256(inputs["projection"])
    _write_json(inputs["projection_summary"], projection_summary)
    with pytest.raises(ValueError, match="result_fingerprint mismatch"):
        _run(inputs)

    rows[0]["result_fingerprint"] = ""
    _refingerprint_projection_row(rows[0])
    _write_jsonl(inputs["projection"], rows)
    projection_summary["output"]["sha256"] = _file_sha256(inputs["projection"])
    _write_json(inputs["projection_summary"], projection_summary)
    with pytest.raises(ValueError, match="incomplete"):
        _run(inputs)

    formal_specs = [_spec(f"formal-ready-{index}") for index in range(100)]
    formal = _write_weak_inputs(
        tmp_path / "formal-ready",
        formal_specs,
        stage="holdout100",
        formal_execution_ready=False,
    )
    with pytest.raises(ValueError, match="formal_execution_ready"):
        _run(formal)

    prior_mismatch = _write_weak_inputs(
        tmp_path / "prior-mismatch",
        formal_specs,
        stage="holdout100",
        formal_execution_ready=True,
        prior_baseline_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="prior baseline_jsonl_sha256"):
        _run(prior_mismatch)


def test_formal_empty_primary_is_ambiguous_not_pass(tmp_path: Path) -> None:
    inputs = _write_weak_inputs(
        tmp_path,
        [_spec(f"no-comparator-{index}", comparators=[], runtime_seconds=1.0) for index in range(100)],
        stage="holdout100",
        formal_execution_ready=True,
    )

    summary = _run(inputs)

    gate = summary["stage_gates"]["gates"]["phase_denominator_available"]
    assert gate["status"] == "ambiguous"
    assert gate["decision_gate"] is True
    assert summary["decision"] == "ambiguous"


def test_formal_projection_summary_row_consistency_tamper_fail_closed(tmp_path: Path) -> None:
    def _formal_inputs(root: Path) -> dict[str, Path]:
        specs = [
            _spec(
                "formal-fallback",
                cj3_grid=None,
                selected_source="current_v2",
                selected_used_fallback=True,
                runtime_seconds=1.0,
            ),
            *[_spec(f"formal-{index}", runtime_seconds=1.0) for index in range(99)],
        ]
        return _write_weak_inputs(root, specs, stage="holdout100", formal_execution_ready=True)

    cases = [
        (("source", "identity_rows", "ordered_cache_audio_keys_sha256"), "0" * 64, "identity_rows.ordered_cache_audio_keys_sha256"),
        (("results", "successful_count"), 99, "results.successful_count"),
        (("denominators", "selected_fallback_count"), 0, "denominators.selected_fallback_count"),
        (("hard_guards", "ok"), False, "hard_guards.ok"),
        (
            ("source", "prior_stage", "stage_constraints"),
            {
                "schema": exp004_protocol.STAGE_CONSTRAINT_SCHEMA,
                "stage": "holdout100",
                "quota_degraded": True,
                "degraded_quotas": ["jump"],
                "broad_underfilled": False,
            },
            "prior stage_constraints",
        ),
    ]
    for index, (path, value, match) in enumerate(cases):
        inputs = _formal_inputs(tmp_path / f"case-{index}")
        projection_summary = _read_json(inputs["projection_summary"])
        target = projection_summary
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        _write_json(inputs["projection_summary"], projection_summary)
        with pytest.raises(ValueError, match=match):
            _run(inputs)


def test_jump_component_kill_does_not_override_combined_or_gate(tmp_path: Path) -> None:
    specs = [
        _spec(
            f"jump-or-{index}",
            coverage_end_ms=4000.0,
            current_v2=_grid((5.0, 501.0)),
            cj3_grid=_v3_constant_with_first_boundary(first_boundary_ms=-100.0, coverage_end_ms=4000.0),
            comparators=[_comparator(f"jump-or-{index}-map", _grid((0.0, 500.0)))],
            strata={"pilot_stratum": "jump_candidate"},
        )
        for index in range(15)
    ]

    summary = _run(_write_weak_inputs(tmp_path, specs))
    gates = summary["stage_gates"]["gates"]

    assert gates["jump_mean_phase_ratio"]["status"] == "kill"
    assert gates["jump_mean_phase_ratio"]["decision_gate"] is False
    assert gates["jump_endpoint_drift_mean_ratio"]["status"] == "pass"
    assert gates["jump_combined_mean_or_drift"]["status"] == "pass"


def test_quota_zero_denominators_make_decision_ambiguous(tmp_path: Path) -> None:
    inputs = _write_weak_inputs(
        tmp_path,
        [_spec("quota-zero", comparators=[_comparator("quota-zero-map", _grid((0.0, 500.0)))])],
        stage_constraints={"quota_degraded": True, "broad_underfilled": True},
    )

    summary = _run(inputs)
    gate = summary["stage_gates"]["gates"]["quota_degraded_minimum_denominator"]

    assert gate["status"] == "ambiguous"
    assert gate["numerator"] == {"jump_count": 0, "long_count": 0}
    assert summary["decision"] == "ambiguous"


def test_stable_n4_is_not_applicable_and_ramp_audit_is_bucketed(tmp_path: Path) -> None:
    stable_specs = [
        _spec(
            f"stable-n4-{index}",
            comparators=[_comparator(f"stable-n4-{index}-map", _grid((0.0, 500.0)))],
            strata={"pilot_stratum": "stable"},
        )
        for index in range(4)
    ]
    stable_summary = _run(_write_weak_inputs(tmp_path / "stable-n4", stable_specs))
    assert stable_summary["stage_gates"]["gates"]["stable_mean_phase_ratio"]["status"] == "not_applicable"

    ramp = _write_weak_inputs(
        tmp_path / "ramp",
        [_spec("ramp", comparators=[_comparator("ramp-map", _grid((0.0, 500.0)))], strata={"pilot_stratum": "ramp_audit"})],
    )
    _run(ramp)
    row = _read_jsonl(ramp["output"])[0]
    assert row["strata"]["primary_stratum"] == "ramp_audit"
    assert row["strata"]["ramp"] is True


def test_exact_no_path_reasons_are_counted_without_substring_matches(tmp_path: Path) -> None:
    specs = [
        _spec(
            "global-no-path",
            cj3_grid=None,
            selected_source="current_v2",
            selected_used_fallback=True,
            variant_reasons={"CJ3": "no_global_constant_jump_path"},
        ),
        _spec(
            "no-origin",
            cj3_grid=None,
            selected_source="current_v2",
            selected_used_fallback=True,
            variant_reasons={"CJ3": "no_origin_candidate"},
        ),
        _spec(
            "substring-only",
            cj3_grid=None,
            selected_source="current_v2",
            selected_used_fallback=True,
            variant_reasons={"CJ3": "synthetic_no_path"},
        ),
        _spec(
            "candidate-failed",
            cj3_grid=None,
            selected_source=None,
            projection_evaluable=False,
            candidate_status="failed",
            candidate_reason="candidate_extraction_failure",
        ),
    ]

    summary = _run(_write_weak_inputs(tmp_path, specs))
    gate = summary["stage_gates"]["gates"]["no_path_plus_candidate_extraction_failure_rate"]

    assert gate["numerator"] == 3
    assert gate["denominator"] == 4


def test_cj2_headline_delta_under_one_percent_is_ambiguous(tmp_path: Path) -> None:
    specs = [
        _spec(
            f"cj2-close-{index}",
            coverage_end_ms=4000.0,
            current_v2=_grid((5.0, 500.0)),
            cj3_grid=_v3_constant_with_first_boundary(first_boundary_ms=-99.5, coverage_end_ms=4000.0),
            variant_grids={"CJ2": _v3_constant_with_first_boundary(first_boundary_ms=-100.0, coverage_end_ms=4000.0)},
            comparators=[_comparator(f"cj2-close-{index}-map", _grid((0.0, 500.0)))],
            strata={"pilot_stratum": "jump_candidate"},
        )
        for index in range(15)
    ]

    summary = _run(_write_weak_inputs(tmp_path, specs))
    gate = summary["stage_gates"]["gates"]["jump_ablation_CJ3_vs_CJ2"]

    assert gate["status"] == "ambiguous"
    assert gate["reason"] == "CJ3_CJ2_headline_delta_under_1_percent"


def test_evaluator_source_rehash_detects_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = weak_evidence._evaluator_source_payload()
    original_file_sha256 = weak_evidence._file_sha256
    metrics_path = Path(payload["metrics_path"]).resolve(strict=False)

    def _tampered_file_sha256(path: Path) -> str:
        if Path(path).resolve(strict=False) == metrics_path:
            return "0" * 64
        return original_file_sha256(path)

    monkeypatch.setattr(weak_evidence, "_file_sha256", _tampered_file_sha256)
    with pytest.raises(RuntimeError, match="metrics_path changed"):
        weak_evidence._require_evaluator_source_unchanged(payload)


def test_projection_jsonl_sha_uses_parsed_bytes_not_late_file_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_weak_inputs(
        tmp_path,
        [_spec("parsed-bytes", comparators=[_comparator("parsed-bytes-map", _grid((0.0, 500.0)))])],
    )
    pinned_sha = _read_json(inputs["projection_summary"])["output"]["sha256"]
    rows = _read_jsonl(inputs["projection"])
    rows[0]["runtime"]["total_seconds"] = 99.0
    _refingerprint_projection_row(rows[0])
    _write_jsonl(inputs["projection"], rows)

    original_file_sha256 = weak_evidence._file_sha256
    projection_path = inputs["projection"].resolve(strict=False)

    def _spoofed_file_sha256(path: Path) -> str:
        if Path(path).resolve(strict=False) == projection_path:
            return pinned_sha
        return original_file_sha256(path)

    monkeypatch.setattr(weak_evidence, "_file_sha256", _spoofed_file_sha256)
    with pytest.raises(ValueError, match="projection JSONL SHA"):
        _run(inputs)


def test_projection_summary_sha_uses_parsed_bytes_not_late_file_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_weak_inputs(
        tmp_path,
        [_spec("summary-parsed-bytes", comparators=[_comparator("summary-parsed-bytes-map", _grid((0.0, 500.0)))])],
    )
    expected_sha = hashlib.sha256(inputs["projection_summary"].read_bytes()).hexdigest()
    summary_path = inputs["projection_summary"].resolve(strict=False)
    original_file_sha256 = weak_evidence._file_sha256

    def _spoofed_file_sha256(path: Path) -> str:
        if Path(path).resolve(strict=False) == summary_path:
            return "0" * 64
        return original_file_sha256(path)

    monkeypatch.setattr(weak_evidence, "_file_sha256", _spoofed_file_sha256)
    source, _rows = weak_evidence._load_projection_inputs(inputs["projection"], inputs["projection_summary"])

    assert source.summary_sha256 == expected_sha


def test_projection_atomic_swap_after_parse_is_caught_by_final_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_weak_inputs(
        tmp_path,
        [_spec("atomic-swap", comparators=[_comparator("atomic-swap-map", _grid((0.0, 500.0)))])],
    )
    original_evaluate = weak_evidence._evaluate_projection_row
    replacement = inputs["projection"].with_suffix(".replacement.jsonl")
    replacement.write_text('{"changed":true}\n', encoding="utf-8")
    swapped = False

    def _swap_once(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal swapped
        if not swapped:
            os.replace(replacement, inputs["projection"])
            swapped = True
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(weak_evidence, "_evaluate_projection_row", _swap_once)
    with pytest.raises(RuntimeError, match="projection JSONL changed"):
        _run(inputs)


def test_projection_summary_in_place_mutation_after_parse_is_caught_by_final_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_weak_inputs(
        tmp_path,
        [_spec("summary-in-place", comparators=[_comparator("summary-in-place-map", _grid((0.0, 500.0)))])],
    )
    original_evaluate = weak_evidence._evaluate_projection_row
    mutated = False

    def _mutate_once(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal mutated
        if not mutated:
            inputs["projection_summary"].write_text(
                inputs["projection_summary"].read_text(encoding="utf-8") + " ",
                encoding="utf-8",
            )
            mutated = True
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(weak_evidence, "_evaluate_projection_row", _mutate_once)
    with pytest.raises(RuntimeError, match="projection summary changed"):
        _run(inputs)


def test_baseline_in_place_mutation_after_parse_is_caught_by_final_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_weak_inputs(
        tmp_path,
        [_spec("in-place", comparators=[_comparator("in-place-map", _grid((0.0, 500.0)))])],
    )
    original_evaluate = weak_evidence._evaluate_projection_row
    mutated = False

    def _mutate_once(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal mutated
        if not mutated:
            inputs["baseline"].write_text(inputs["baseline"].read_text(encoding="utf-8") + " ", encoding="utf-8")
            mutated = True
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(weak_evidence, "_evaluate_projection_row", _mutate_once)
    with pytest.raises(RuntimeError, match="baseline JSONL changed"):
        _run(inputs)


def _spec(
    key: str,
    *,
    coverage_end_ms: float = 1000.0,
    current_v2: FittedTimingGrid | None = None,
    current_v2_available: bool = True,
    cj3_grid: TimingV3Grid | None = None,
    variant_grids: dict[str, TimingV3Grid | None] | None = None,
    variant_reasons: dict[str, str | None] | None = None,
    selected_source: str | None = "CJ3",
    selected_used_fallback: bool = False,
    projection_evaluable: bool | None = None,
    cache_valid: bool = True,
    candidate_status: str = "accepted",
    candidate_reason: str | None = None,
    comparators: list[dict[str, Any]] | None = None,
    baseline_present: bool = True,
    strata: dict[str, Any] | None = None,
    runtime_seconds: float | None = None,
) -> dict[str, Any]:
    current_v2 = _grid((0.0, 500.0)) if current_v2 is None else current_v2
    if cj3_grid is None and selected_source == "CJ3":
        cj3_grid = _v3_constant(coverage_end_ms=coverage_end_ms)
    if projection_evaluable is None:
        projection_evaluable = selected_source is not None
    return {
        "key": key,
        "coverage_end_ms": coverage_end_ms,
        "current_v2": current_v2,
        "current_v2_available": current_v2_available,
        "cj3_grid": cj3_grid,
        "variant_grids": dict(variant_grids or {}),
        "variant_reasons": dict(variant_reasons or {}),
        "selected_source": selected_source,
        "selected_used_fallback": selected_used_fallback,
        "projection_evaluable": projection_evaluable,
        "cache_valid": cache_valid,
        "candidate_status": candidate_status,
        "candidate_reason": candidate_reason,
        "comparators": list(comparators or []),
        "baseline_present": baseline_present,
        "strata": dict(strata or {}),
        "runtime_seconds": runtime_seconds,
    }


def _comparator(key: str, grid: FittedTimingGrid) -> dict[str, Any]:
    return {
        "ok": True,
        "difficulty_key": key,
        "beatmap_path": f"{key}.osu",
        "oracle_segments": _segments(grid),
    }


def _write_weak_inputs(
    root: Path,
    specs: list[dict[str, Any]],
    *,
    stage: str = SYNTHETIC_FIXTURE_STAGE,
    projection_hard_guards_ok: bool = True,
    formal_execution_ready: bool = False,
    integration_blockers: list[str] | None = None,
    stage_constraints: dict[str, Any] | None = None,
    prior_baseline_sha256: str | None = None,
) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    projection_path = root / "projection.jsonl"
    projection_summary_path = root / "projection.summary.json"
    baseline_path = root / "baseline.jsonl"
    output_path = root / "weak.jsonl"
    summary_path = root / "weak.summary.json"
    provenance = {
        "behavior_fingerprint": "a" * 64,
        "config_fingerprint": "b" * 64,
        "run_fingerprint": "c" * 64,
    }
    projection_rows = [
        _projection_row(index, spec, provenance=provenance, stage=stage)
        for index, spec in enumerate(specs)
    ]
    _write_jsonl(projection_path, projection_rows)
    baseline_rows = [
        _baseline_row(index, spec)
        for index, spec in enumerate(specs)
        if spec["baseline_present"]
    ]
    _write_jsonl(baseline_path, baseline_rows)
    stage_audio_count = exp004_runner.STAGE_AUDIO_COUNTS.get(stage, len(specs))
    baseline_sha = _file_sha256(baseline_path)
    prior_sha = prior_baseline_sha256 if prior_baseline_sha256 is not None else baseline_sha
    ordered_keys = [str(spec["key"]) for spec in specs]
    key_set_hash = _json_sha(sorted(set(ordered_keys)))
    ordered_key_hash = exp004_protocol.ordered_cache_audio_keys_sha256(ordered_keys)
    identity_rows = [
        {
            "schema": exp004_protocol.IDENTITY_ROW_SCHEMA,
            "stage": stage,
            "cache_audio_key": str(spec["key"]),
            "audio_group_key": f"group-{spec['key']}",
        }
        for spec in specs
    ]
    fallback_reason_counts: dict[str, int] = {}
    for spec in specs:
        if spec["selected_used_fallback"]:
            reason = "synthetic_fallback"
            fallback_reason_counts[reason] = fallback_reason_counts.get(reason, 0) + 1
    projection_evaluable_count = sum(bool(spec["projection_evaluable"]) for spec in specs)
    selected_fallback_count = sum(bool(spec["selected_used_fallback"]) for spec in specs)
    fallback_rate = selected_fallback_count / projection_evaluable_count if projection_evaluable_count else None
    projection_summary = {
        "schema": exp004_runner.SUMMARY_SCHEMA,
        "experiment": "timing_v3_experiment_004",
        "stage": stage,
        "provenance": provenance,
        "source": {
            "stage_audio_count": stage_audio_count,
            "identity_rows": {
                "row_count": len(specs),
                "cache_audio_keys_sha256": key_set_hash,
                "ordered_cache_audio_keys_sha256": ordered_key_hash,
                "ordered_identity_rows_sha256": _json_sha(identity_rows),
            },
            "selection_manifest": {
                "row_count": len(specs),
                "cache_audio_keys_sha256": key_set_hash,
                "ordered_cache_audio_keys_sha256": ordered_key_hash,
                "stage_constraints": dict(stage_constraints or {"schema": exp004_protocol.STAGE_CONSTRAINT_SCHEMA, "stage": stage, "quota_degraded": False, "degraded_quotas": [], "broad_underfilled": False}),
            },
            "prior_stage": {
                "stage": "repair80",
                "behavior_fingerprint": provenance["behavior_fingerprint"],
                "config_fingerprint": provenance["config_fingerprint"],
                "weak_output_jsonl_sha256": "d" * 64,
                "projection_jsonl_sha256": "e" * 64,
                "projection_summary_sha256": "f" * 64,
                "baseline_jsonl_sha256": prior_sha,
                "evaluator_sha256": "1" * 64,
                "stage_gates_sha256": "2" * 64,
                "protocol_binding_sha256": "3" * 64,
                "stage_constraints": dict(stage_constraints or {"schema": exp004_protocol.STAGE_CONSTRAINT_SCHEMA, "stage": stage, "quota_degraded": False, "degraded_quotas": [], "broad_underfilled": False}),
            },
        },
        "output": {"path": projection_path.as_posix(), "sha256": _file_sha256(projection_path), "row_count": len(specs)},
        "results": {
            "result_count": len(specs),
            "successful_count": projection_evaluable_count,
            "failed_count": len(specs) - projection_evaluable_count,
        },
        "denominators": {
            "stage_audio_count": stage_audio_count,
            "cache_valid_count": sum(bool(spec["cache_valid"]) for spec in specs),
            "projection_evaluable_count": projection_evaluable_count,
            "selected_fallback_count": selected_fallback_count,
            "selected_fallback_rate": fallback_rate,
            "fallback_reason_counts": dict(sorted(fallback_reason_counts.items())),
        },
        "hard_guards": {
            "ok": projection_hard_guards_ok,
            "violations": [] if projection_hard_guards_ok else [{"reason": "synthetic_projection_hard_guard"}],
        },
        "integration": {
            "formal_execution_ready": formal_execution_ready,
            "blockers": list(integration_blockers or []),
        },
    }
    _write_json(projection_summary_path, projection_summary)
    return {
        "projection": projection_path,
        "projection_summary": projection_summary_path,
        "baseline": baseline_path,
        "output": output_path,
        "summary": summary_path,
    }


def _projection_row(index: int, spec: dict[str, Any], *, provenance: dict[str, str], stage: str) -> dict[str, Any]:
    key = str(spec["key"])
    current_v2 = spec["current_v2"]
    assert isinstance(current_v2, FittedTimingGrid)
    cj3_grid = spec["cj3_grid"]
    variants: dict[str, Any] = {}
    variant_grids = spec["variant_grids"]
    variant_reasons = spec["variant_reasons"]
    for variant in ("CJ0", "CJ1", "CJ2", "CJ3"):
        if variant in variant_grids:
            grid = variant_grids[variant]
        else:
            grid = cj3_grid if variant == "CJ3" else _v3_constant(coverage_end_ms=float(spec["coverage_end_ms"]))
        variants[variant] = _variant_payload(grid, reason=variant_reasons.get(variant))
    selected_source = spec["selected_source"]
    projection_evaluable = bool(spec["projection_evaluable"])
    selected_grid = current_v2 if selected_source == "current_v2" else cj3_grid if selected_source == "CJ3" else None
    payload = {
        "schema": exp004_runner.RESULT_SCHEMA,
        "experiment": "timing_v3_experiment_004",
        "stage": stage,
        "row_index": index,
        "row_complete": True,
        "ok": projection_evaluable,
        "identity": {"cache_audio_key": key, "audio_group_key": f"group-{key}"},
        "provenance": dict(provenance),
        "resume": {"schema": exp004_runner.RESUME_SCHEMA, "fingerprint": _sha(f"resume-{key}")},
        "cache": {"status": "valid", "coverage_start_ms": 0.0, "coverage_end_ms": float(spec["coverage_end_ms"])},
        "current_v2": (
            {
                "status": "accepted",
                "reason": None,
                "grid": _fitted_payload(current_v2),
                "grid_fingerprint": _json_sha(_fitted_payload(current_v2)),
            }
            if spec["current_v2_available"]
            else {
                "status": "failed",
                "reason": "fit_failure",
                "grid": None,
                "grid_fingerprint": None,
            }
        ),
        "candidate_extraction": {"status": spec["candidate_status"], "reason": spec["candidate_reason"]},
        "variants": variants,
        "selection": {
            "method": "selected_CJ3_or_current_v2_fallback_v1",
            "source": selected_source,
            "used_fallback": bool(spec["selected_used_fallback"]),
            "fallback_reason": "synthetic_fallback" if spec["selected_used_fallback"] else None,
            "grid_fingerprint": _json_sha(selected_grid.to_dict() if isinstance(selected_grid, TimingV3Grid) else _fitted_payload(selected_grid)) if selected_grid is not None else None,
        },
        "projection_flags": {
            "cache_valid": bool(spec["cache_valid"]),
            "projection_evaluable": projection_evaluable,
            "pure_cj3_grid_produced": cj3_grid is not None,
            "selected_used_fallback": bool(spec["selected_used_fallback"]),
            "hard_guard_violation": False,
            "hard_guard_reasons": [],
        },
        "runtime": {"total_seconds": float(spec["runtime_seconds"] if spec["runtime_seconds"] is not None else index + 1)},
    }
    payload["result_fingerprint"] = _json_sha(payload)
    return payload


def _variant_payload(grid: TimingV3Grid | None, *, reason: str | None = None) -> dict[str, Any]:
    if grid is None:
        return {
            "status": "failed",
            "reason": reason or "synthetic_no_path",
            "grid": None,
            "grid_fingerprint": None,
        }
    return {
        "status": "accepted",
        "reason": None,
        "grid": grid.to_dict(),
        "grid_fingerprint": _json_sha(grid.to_dict()),
    }


def _baseline_row(index: int, spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": BASELINE_RESULT_SCHEMA,
        "ok": True,
        "audio_key": spec["key"],
        "row_index": index,
        "evaluation_strata": spec["strata"],
        "fit": {"predicted_segments": _segments(spec["current_v2"])},
        "comparisons": spec["comparators"],
    }


def _fitted_payload(grid: FittedTimingGrid | None) -> dict[str, Any]:
    assert grid is not None
    return {"schema": "pulsefield_model.timing_fitted_grid_v1", "segments": _segments(grid)}


def _segments(grid: FittedTimingGrid) -> list[dict[str, Any]]:
    return [
        {"offset_ms": segment.offset_ms, "beat_length_ms": segment.beat_length_ms, "meter": segment.meter}
        for segment in grid.segments
    ]


def _run(inputs: dict[str, Path]) -> dict[str, Any]:
    return run_exp004_weak_evidence(
        projection_jsonl_path=inputs["projection"],
        projection_summary_path=inputs["projection_summary"],
        baseline_jsonl_path=inputs["baseline"],
        output_jsonl_path=inputs["output"],
        summary_json_path=inputs["summary"],
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) for row in rows)
    path.write_text(data + ("\n" if data else ""), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _json_sha(payload: Any) -> str:
    return _sha(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _refingerprint_projection_row(row: dict[str, Any]) -> None:
    row.pop("result_fingerprint", None)
    row["result_fingerprint"] = _json_sha(row)


def _file_sha256(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(text: str) -> str:
    return __import__("hashlib").sha256(text.encode("utf-8")).hexdigest()
