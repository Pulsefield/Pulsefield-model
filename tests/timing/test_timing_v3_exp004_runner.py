from __future__ import annotations

import ast
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pulsefield_model.timing.evaluation import exp004_runner as runner
from pulsefield_model.timing.providers.beatthis_cache import (
    BeatThisFramePredictionCacheConfig,
    BeatThisFramePredictionCacheError,
    beatthis_frame_prediction_cache_path,
    save_beatthis_frame_prediction_cache,
)
from pulsefield_model.timing.schema import FittedTimingGrid, FrameTimingPrediction, TimingSegment
from pulsefield_model.timing.v3.global_constant_jump import (
    BOUNDARY_CANDIDATE_SCORE_VERSION,
    CANDIDATE_CONTRACT_VERSION,
    GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON_SHA256,
    GLOBAL_CONSTANT_JUMP_VARIANTS,
    PULSE_CORRELATION_VERSION,
    GlobalConstantJumpCandidateDiagnostics,
    GlobalConstantJumpCandidateSet,
    GlobalConstantJumpDiagnostics,
    GlobalConstantJumpResult,
)
from pulsefield_model.timing.v3.schema import ConstantTimingSection, TimingV3Grid


@dataclass(frozen=True)
class _V2Diagnostics:
    marker: int


def test_runner_loads_and_fits_each_cache_once_with_one_restricted_candidate_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_stage_inputs(tmp_path, stage="repair80")
    calls = _install_fakes(monkeypatch, inputs)

    summary = _run(inputs)

    assert len(calls["cache"]) == 80
    assert len(calls["v2"]) == 80
    assert len(calls["extract"]) == 80
    assert len(calls["variants"]) == 80
    assert all(call == tuple(GLOBAL_CONSTANT_JUMP_VARIANTS) for call in calls["variants"])
    assert summary["schema"] == runner.SUMMARY_SCHEMA
    assert summary["denominators"] == {
        "stage_audio_count": 80,
        "cache_valid_count": 80,
        "projection_evaluable_count": 80,
        "selected_fallback_count": 0,
        "selected_fallback_rate": 0.0,
        "fallback_reason_counts": {},
    }
    assert summary["oracle_comparison"]["status"] == runner.UNDEFINED_PROJECTION_ONLY
    assert summary["oracle_comparison"]["comparison_eligible_count"] is None
    assert summary["integration"] == {
        "formal_execution_ready": True,
        "blockers": list(runner.INTEGRATION_BLOCKERS),
    }
    assert summary["source"]["selection_manifest"]["stage_constraints"] == {
        "schema": runner.exp004_protocol.STAGE_CONSTRAINT_SCHEMA,
        "stage": "repair80",
        "quota_degraded": False,
        "degraded_quotas": [],
        "broad_underfilled": False,
    }
    behavior_sources = {
        item["relative_path"]: item["sha256"]
        for item in summary["provenance"]["behavior"]["source_files"]
    }
    for relative_path in (
        "docs/research/timing_v3_experiment_004_global_constant_jump.md",
        "docs/research/timing_v3_experiment_004_protocol_clarification_001.md",
    ):
        assert behavior_sources[relative_path] == runner._file_sha256(
            runner._repo_root() / relative_path
        )

    rows = _read_jsonl(inputs["output"])
    assert [row["identity"]["cache_audio_key"] for row in rows] == inputs["keys"]
    assert all(row["current_v2"]["status"] == "accepted" for row in rows)
    assert all(tuple(row["variants"]) == tuple(GLOBAL_CONSTANT_JUMP_VARIANTS) for row in rows)
    assert all(row["selection"]["source"] == "CJ3" for row in rows)
    assert all(row["candidate_extraction"]["status"] == "accepted" for row in rows)
    json.dumps(rows, allow_nan=False)


def test_projection_and_fallback_truth_table_keeps_both_fail_non_evaluable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_stage_inputs(tmp_path, stage="repair80")
    _install_fakes(
        monkeypatch,
        inputs,
        cj3_fail={0, 1},
        v2_fail={1},
        invalid_cj3={2},
    )

    summary = _run(inputs)
    rows = _read_jsonl(inputs["output"])

    assert rows[0]["selection"]["source"] == "current_v2"
    assert rows[0]["projection_flags"]["selected_used_fallback"] is True
    assert rows[0]["projection_flags"]["pure_cj3_grid_produced"] is False
    assert rows[1]["selection"]["source"] is None
    assert rows[1]["projection_flags"]["projection_evaluable"] is False
    assert rows[1]["ok"] is False
    assert rows[2]["variants"]["CJ3"]["reason"] == "schema_or_serialization_failure"
    assert rows[2]["selection"]["source"] == "current_v2"
    assert summary["denominators"]["projection_evaluable_count"] == 79
    assert summary["denominators"]["selected_fallback_count"] == 2
    assert summary["denominators"]["selected_fallback_rate"] == pytest.approx(2 / 79)
    assert summary["hard_guards"]["ok"] is False
    assert summary["hard_guards"]["violations"] == [
        {"reason": "CJ3_schema_or_serialization_failure", "row_indices": [2]}
    ]


def test_missing_corrupt_and_mutated_cache_are_distinct_and_never_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_stage_inputs(tmp_path, stage="repair80")
    inputs["cache_paths"][0].unlink()
    calls = _install_fakes(monkeypatch, inputs, corrupt_cache={1, 3}, mutate_cache={2, 3})

    with pytest.raises(RuntimeError, match="declared cache file changed"):
        _run(inputs)
    rows = _read_jsonl(inputs["output"])

    assert rows[0]["cache"]["status"] == "missing"
    assert rows[1]["cache"]["status"] == "invalid"
    assert rows[2]["cache"]["status"] == "mutated_during_load"
    assert rows[3]["cache"]["status"] == "mutated_during_load"
    assert rows[0]["variants"]["CJ3"]["reason"] == "cache_missing"
    assert rows[1]["variants"]["CJ3"]["reason"] == "cache_invalid"
    assert rows[2]["variants"]["CJ3"]["reason"] == "cache_mutated_during_load"
    assert rows[0]["current_v2"]["status"] == "not_run"
    assert rows[1]["current_v2"]["status"] == "not_run"
    assert rows[2]["current_v2"]["status"] == "not_run"
    assert len(calls["cache"]) == 80
    assert len(calls["v2"]) == 76
    assert not inputs["summary"].exists()
    row_violations = {
        row["row_index"]: row["projection_flags"]["hard_guard_reasons"]
        for row in rows[:4]
    }
    assert row_violations == {
        0: ["cache_missing"],
        1: ["cache_invalid"],
        2: ["cache_mutated_during_load"],
        3: ["cache_mutated_during_load"],
    }
    assert rows[1]["cache"]["identity_after_load"] == rows[1]["cache"]["identity_before_load"]


def test_resume_reuses_exact_rows_and_invalidates_one_changed_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_stage_inputs(tmp_path, stage="repair80")
    first_calls = _install_fakes(monkeypatch, inputs)
    _run(inputs)
    assert len(first_calls["cache"]) == 80

    second_calls = _install_fakes(monkeypatch, inputs)
    second = _run(inputs)
    assert second["resume"]["reused_success"] == 80
    assert second["resume"]["processed"] == 0
    assert second_calls["cache"] == []
    assert second_calls["v2"] == []

    inputs["cache_paths"][0].write_bytes(b"changed-cache-bytes")
    third_calls = _install_fakes(monkeypatch, inputs)
    third = _run(inputs)
    assert third["resume"]["recomputed_stale"] == 1
    assert third["resume"]["reused_success"] == 79
    assert third["resume"]["processed"] == 1
    assert third_calls["cache"] == [inputs["keys"][0]]


@pytest.mark.parametrize("damage", ["duplicate", "wrong_schema", "truncated", "tampered"])
def test_resume_rejects_duplicate_wrong_schema_and_truncated_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    inputs = _write_stage_inputs(tmp_path, stage="repair80")
    _install_fakes(monkeypatch, inputs)
    _run(inputs)
    rows = _read_jsonl(inputs["output"])
    if damage == "duplicate":
        inputs["output"].write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in [*rows, rows[0]]) + "\n",
            encoding="utf-8",
        )
    elif damage == "wrong_schema":
        rows[0]["schema"] = "wrong"
        _write_jsonl(inputs["output"], rows)
    elif damage == "tampered":
        rows[0]["selection"]["source"] = None
        _write_jsonl(inputs["output"], rows)
    else:
        inputs["output"].write_text('{"schema":', encoding="utf-8")

    with pytest.raises(ValueError):
        _run(inputs)


def test_retry_failures_recomputes_only_non_evaluable_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_stage_inputs(tmp_path, stage="repair80")
    _install_fakes(monkeypatch, inputs, cj3_fail={0}, v2_fail={0})
    _run(inputs)

    calls = _install_fakes(monkeypatch, inputs, cj3_fail={0}, v2_fail={0})
    summary = _run(inputs, retry_failures=True)

    assert summary["resume"]["retried_failure"] == 1
    assert summary["resume"]["reused_success"] == 79
    assert calls["cache"] == [inputs["keys"][0]]


def test_unrelated_git_dirty_provenance_does_not_invalidate_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_stage_inputs(tmp_path, stage="repair80")
    _install_fakes(monkeypatch, inputs)
    monkeypatch.setattr(
        runner,
        "_git_provenance",
        lambda: {"commit_sha": "a" * 40, "dirty_files": ["?? unrelated-a"]},
    )
    _run(inputs)

    calls = _install_fakes(monkeypatch, inputs)
    monkeypatch.setattr(
        runner,
        "_git_provenance",
        lambda: {"commit_sha": "b" * 40, "dirty_files": ["?? unrelated-b"]},
    )
    summary = _run(inputs)

    assert summary["resume"]["reused_success"] == 80
    assert calls["cache"] == []


def test_formal_stage_requires_matching_completed_prior_summary(tmp_path: Path) -> None:
    inputs = _write_stage_inputs(tmp_path, stage="holdout100")
    with pytest.raises(ValueError, match="requires the repair80 weak summary"):
        _run(inputs)

    cache_config = _cache_config(inputs)
    behavior = runner._behavior_payload(runner._behavior_source_identities())
    behavior_fingerprint = runner._stable_json_sha256(behavior)
    config_fingerprint = runner._stable_json_sha256(runner._config_payload(cache_config))
    prior = _prior_weak_summary_payload(
        behavior_fingerprint="0" * 64,
        config_fingerprint=config_fingerprint,
    )
    _write_json(inputs["prior"], prior)
    with pytest.raises(ValueError, match="behavior fingerprint"):
        _run(inputs, prior_stage_summary_path=inputs["prior"])

    prior = _prior_weak_summary_payload(
        behavior_fingerprint=behavior_fingerprint,
        config_fingerprint="2" * 64,
    )
    _write_json(inputs["prior"], prior)
    with pytest.raises(ValueError, match="config fingerprint"):
        _run(inputs, prior_stage_summary_path=inputs["prior"])

    prior = _prior_weak_summary_payload(
        behavior_fingerprint=behavior_fingerprint,
        config_fingerprint=config_fingerprint,
    )
    _write_json(inputs["prior"], prior)
    accepted = runner._load_and_validate_prior_summary(
        inputs["prior"],
        stage="holdout100",
        behavior_fingerprint=behavior_fingerprint,
        config_fingerprint=config_fingerprint,
    )
    assert accepted is not None
    assert accepted["decision"] == "debug_only"

    prior["stage_gates"]["gates"]["jump_mean_phase_ratio"]["decision_gate"] = True
    _write_json(inputs["prior"], prior)
    with pytest.raises(ValueError, match="decision relevance"):
        runner._load_and_validate_prior_summary(
            inputs["prior"],
            stage="holdout100",
            behavior_fingerprint=behavior_fingerprint,
            config_fingerprint=config_fingerprint,
        )

    prior = _prior_weak_summary_payload(
        behavior_fingerprint=behavior_fingerprint,
        config_fingerprint=config_fingerprint,
    )
    prior["stage_gates"]["gates"]["projection_hard_guards"].update(
        {"value": False, "status": "kill", "reason": "tampered"}
    )
    _write_json(inputs["prior"], prior)
    with pytest.raises(ValueError, match="hard-guard gate disagrees"):
        runner._load_and_validate_prior_summary(
            inputs["prior"],
            stage="holdout100",
            behavior_fingerprint=behavior_fingerprint,
            config_fingerprint=config_fingerprint,
        )

    holdout_prior = _prior_weak_summary_payload(
        behavior_fingerprint=behavior_fingerprint,
        config_fingerprint=config_fingerprint,
        stage="holdout100",
    )
    _write_json(inputs["prior"], holdout_prior)
    assert runner._load_and_validate_prior_summary(
        inputs["prior"],
        stage="broad500",
        behavior_fingerprint=behavior_fingerprint,
        config_fingerprint=config_fingerprint,
    ) is not None
    holdout_prior["stage_gates"]["gates"]["all_mean_phase_ratio"].update(
        {"value": 1.2, "status": "kill", "reason": "tampered"}
    )
    _write_json(inputs["prior"], holdout_prior)
    with pytest.raises(ValueError, match="do not imply"):
        runner._load_and_validate_prior_summary(
            inputs["prior"],
            stage="broad500",
            behavior_fingerprint=behavior_fingerprint,
            config_fingerprint=config_fingerprint,
        )

    prior = _prior_weak_summary_payload(
        behavior_fingerprint=behavior_fingerprint,
        config_fingerprint=config_fingerprint,
    )
    prior["decision"] = "pass"
    prior["protocol_binding"]["decision"] = "pass"
    _write_json(inputs["prior"], prior)
    with pytest.raises(ValueError, match="does not authorize"):
        runner._load_and_validate_prior_summary(
            inputs["prior"],
            stage="holdout100",
            behavior_fingerprint=behavior_fingerprint,
            config_fingerprint=config_fingerprint,
        )


def test_prior_summary_load_rejects_mid_read_swap_before_sha_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_stage_inputs(tmp_path, stage="holdout100")
    cache_config = _cache_config(inputs)
    behavior = runner._behavior_payload(runner._behavior_source_identities())
    behavior_fingerprint = runner._stable_json_sha256(behavior)
    config_fingerprint = runner._stable_json_sha256(runner._config_payload(cache_config))
    prior = _prior_weak_summary_payload(
        behavior_fingerprint=behavior_fingerprint,
        config_fingerprint=config_fingerprint,
    )
    _write_json(inputs["prior"], prior)

    original_signature = runner._file_stat_signature
    calls = 0

    def mutating_signature(path: Path) -> tuple[int, int, int, int]:
        nonlocal calls
        if path == inputs["prior"]:
            calls += 1
            if calls == 2:
                inputs["prior"].write_bytes(b'{"schema":"attacker-swapped-after-parse"}\n')
        return original_signature(path)

    monkeypatch.setattr(runner, "_file_stat_signature", mutating_signature)

    with pytest.raises(RuntimeError, match="file changed while reading"):
        runner._load_and_validate_prior_summary(
            inputs["prior"],
            stage="holdout100",
            behavior_fingerprint=behavior_fingerprint,
            config_fingerprint=config_fingerprint,
        )


def test_lock_alias_worker_bounds_and_nonfinite_guards_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_stage_inputs(tmp_path, stage="repair80")
    _install_fakes(monkeypatch, inputs)

    with pytest.raises(ValueError, match="workers must be an integer"):
        _run(inputs, workers=0)
    with pytest.raises(ValueError, match="workers must be an integer"):
        _run(inputs, workers=runner.MAX_PROJECTION_WORKERS + 1)
    with pytest.raises(ValueError, match="aliases"):
        _run(inputs, output_jsonl_path=inputs["identity"])
    with pytest.raises(ValueError, match="declared cache path"):
        _run(inputs, output_jsonl_path=inputs["cache_paths"][0])
    with pytest.raises(ValueError, match="non-finite"):
        runner._write_json_atomic(tmp_path / "bad.json", {"bad": float("nan")})

    lock_path = inputs["output"].with_name(f".{inputs['output'].name}.lock")
    lock_path.write_text("owned", encoding="utf-8")
    with pytest.raises(RuntimeError, match="locked"):
        _run(inputs)


def test_spawn_process_workers_are_picklable_timeout_safe_and_manifest_ordered(
    tmp_path: Path,
) -> None:
    cache_config = BeatThisFramePredictionCacheConfig(cache_root=tmp_path / "cache")
    run_provenance = {
        "schema": "synthetic-provenance-v1",
        "run_fingerprint": "a" * 64,
        "behavior_fingerprint": "b" * 64,
        "config_fingerprint": "c" * 64,
    }
    work_items = []
    for index in range(2):
        cache_key = f"spawn-cache-{index}"
        signal_values = np.zeros(32, dtype=np.float32)
        signal_values[8] = 1.0
        signal_values[24] = 1.0
        save_beatthis_frame_prediction_cache(
            FrameTimingPrediction(
                provider="beat-this",
                checkpoint_path=cache_config.checkpoint_path,
                source_path=None,
                beat_prob=signal_values,
                downbeat_prob=np.zeros_like(signal_values),
                frame_rate_hz=cache_config.frame_rate_hz,
            ),
            cache_key,
            cache_config,
        )
        identity = runner._IdentityRow(
            row_index=index,
            cache_audio_key=cache_key,
            audio_group_key=f"spawn-group-{index}",
            stage="repair80",
            payload_sha256=_sha(f"identity-{index}"),
        )
        entry = runner._SelectionEntry(
            row_index=index,
            cache_audio_key=cache_key,
            audio_group_key=f"spawn-group-{index}",
            payload_sha256=_sha(f"entry-{index}"),
        )
        work_items.append(
            runner._ProjectionWorkItem(
                identity=identity,
                entry=entry,
                cache_identity=runner._cache_file_identity(
                    beatthis_frame_prediction_cache_path(cache_key, cache_config)
                ),
                resume={"schema": runner.RESUME_SCHEMA, "fingerprint": _sha(f"resume-{index}")},
            )
        )

    rows = list(
        runner._evaluate_projection_work_items(
            work_items,
            workers=2,
            cache_config=cache_config,
            run_provenance=run_provenance,
        )
    )

    assert [row["row_index"] for row in rows] == [0, 1]
    assert [row["identity"]["cache_audio_key"] for row in rows] == [
        "spawn-cache-0",
        "spawn-cache-1",
    ]
    assert all(row["cache"]["status"] == "valid" for row in rows)
    assert all(row["row_complete"] is True for row in rows)


def test_unique_atomic_temporary_does_not_collide_with_legacy_tmp_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_stage_inputs(tmp_path, stage="repair80")
    _install_fakes(monkeypatch, inputs)
    collision = inputs["output"].with_name(inputs["output"].name + ".tmp")
    collision.write_text("sentinel", encoding="utf-8")

    _run(inputs)

    assert collision.read_text(encoding="utf-8") == "sentinel"
    assert list(tmp_path.glob(f".{inputs['output'].name}.*.tmp")) == []


def test_per_audio_timeout_is_enforced_without_silently_running_fitters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_stage_inputs(tmp_path, stage="repair80")
    calls = _install_fakes(monkeypatch, inputs)
    original_loader = runner.load_beatthis_frame_prediction_cache

    def slow_loader(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.05)
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(runner, "load_beatthis_frame_prediction_cache", slow_loader)
    monkeypatch.setattr(runner, "PER_AUDIO_TIMEOUT_SECONDS", 0.001)

    summary = _run(inputs)
    rows = _read_jsonl(inputs["output"])

    assert summary["runtime"]["per_audio_timeout_seconds"] == 0.001
    assert summary["denominators"]["projection_evaluable_count"] == 0
    assert all(row["cache"]["status"] == "timeout" for row in rows)
    assert calls["v2"] == []
    assert calls["extract"] == []


@pytest.mark.parametrize(
    ("slow_stage", "expected_active_stage", "projection_evaluable_count", "fallback_count"),
    [
        ("current_v2", "current_v2", 79, 0),
        ("candidate_extraction", "candidate_extraction", 80, 1),
    ],
)
def test_timeout_attribution_preserves_completed_pre_variant_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    slow_stage: str,
    expected_active_stage: str,
    projection_evaluable_count: int,
    fallback_count: int,
) -> None:
    inputs = _write_stage_inputs(tmp_path, stage="repair80")
    _install_fakes(monkeypatch, inputs)
    monkeypatch.setattr(runner, "PER_AUDIO_TIMEOUT_SECONDS", 0.05)
    observed_active_stages = _observe_timeout_active_stages(monkeypatch)

    if slow_stage == "current_v2":
        original_grid_fitter = runner.GridFitter

        class SlowFirstGridFitter:
            def __init__(self) -> None:
                self._delegate = original_grid_fitter()

            def fit(self, prediction: FrameTimingPrediction) -> Any:
                if _prediction_index(prediction) == 0:
                    time.sleep(0.2)
                return self._delegate.fit(prediction)

        monkeypatch.setattr(runner, "GridFitter", SlowFirstGridFitter)
    else:
        original_extract = runner.extract_global_constant_jump_candidates

        def slow_first_extract(prediction: FrameTimingPrediction) -> Any:
            if _prediction_index(prediction) == 0:
                time.sleep(0.2)
            return original_extract(prediction)

        monkeypatch.setattr(runner, "extract_global_constant_jump_candidates", slow_first_extract)

    summary = _run(inputs)
    row = _read_jsonl(inputs["output"])[0]

    assert observed_active_stages == [expected_active_stage]
    assert row["cache"]["status"] == "valid"
    if slow_stage == "current_v2":
        assert row["current_v2"]["reason"] == "timeout"
        assert row["candidate_extraction"]["reason"] == "not_run_after_current_v2_timeout"
        assert row["selection"]["source"] is None
    else:
        assert row["current_v2"]["status"] == "accepted"
        assert row["candidate_extraction"]["reason"] == "timeout"
        assert row["selection"]["source"] == "current_v2"
    assert all(row["variants"][variant]["reason"] == "timeout" for variant in GLOBAL_CONSTANT_JUMP_VARIANTS)
    assert summary["denominators"]["projection_evaluable_count"] == projection_evaluable_count
    assert summary["denominators"]["selected_fallback_count"] == fallback_count
    assert summary["hard_guards"] == {
        "ok": False,
        "violations": [
            {"reason": f"{variant}_timeout", "row_indices": [0]}
            for variant in GLOBAL_CONSTANT_JUMP_VARIANTS
        ],
    }


@pytest.mark.parametrize("target_variant", GLOBAL_CONSTANT_JUMP_VARIANTS)
def test_each_variant_fit_timeout_preserves_completed_variants_and_marks_only_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_variant: str,
) -> None:
    inputs = _write_stage_inputs(tmp_path, stage="repair80")
    _install_fakes(monkeypatch, inputs)
    monkeypatch.setattr(runner, "PER_AUDIO_TIMEOUT_SECONDS", 0.05)
    observed_active_stages = _install_targeted_variant_timeout(
        monkeypatch,
        target_variant=target_variant,
        stage="fit",
    )

    summary = _run(inputs)
    row = _read_jsonl(inputs["output"])[0]

    assert observed_active_stages == [f"variant_fit:{target_variant}"]
    _assert_timeout_variant_tail(row, target_variant=target_variant)
    assert row["current_v2"]["status"] == "accepted"
    assert row["candidate_extraction"]["status"] == "accepted"
    assert row["selection"]["source"] == "current_v2"
    assert summary["denominators"]["projection_evaluable_count"] == 80
    assert summary["denominators"]["selected_fallback_count"] == 1
    assert summary["denominators"]["fallback_reason_counts"] == {"timeout": 1}
    _assert_partial_timeout_summary(summary, target_variant=target_variant)


@pytest.mark.parametrize("target_variant", GLOBAL_CONSTANT_JUMP_VARIANTS)
def test_each_variant_serialization_timeout_preserves_completed_variants_and_marks_only_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_variant: str,
) -> None:
    inputs = _write_stage_inputs(tmp_path, stage="repair80")
    _install_fakes(monkeypatch, inputs)
    monkeypatch.setattr(runner, "PER_AUDIO_TIMEOUT_SECONDS", 0.05)
    observed_active_stages = _install_targeted_variant_timeout(
        monkeypatch,
        target_variant=target_variant,
        stage="serialize",
    )

    summary = _run(inputs)
    row = _read_jsonl(inputs["output"])[0]

    assert observed_active_stages == [f"variant_serialize:{target_variant}"]
    _assert_timeout_variant_tail(row, target_variant=target_variant)
    assert row["current_v2"]["status"] == "accepted"
    assert row["candidate_extraction"]["status"] == "accepted"
    assert row["selection"]["source"] == "current_v2"
    assert summary["denominators"]["projection_evaluable_count"] == 80
    assert summary["denominators"]["selected_fallback_count"] == 1
    assert summary["denominators"]["fallback_reason_counts"] == {"timeout": 1}
    _assert_partial_timeout_summary(summary, target_variant=target_variant)


@pytest.mark.parametrize("failure_stage", ["fit", "serialize"])
def test_non_timeout_variant_exception_preserves_prefix_and_marks_current_and_tail_distinctly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    inputs = _write_stage_inputs(tmp_path, stage="repair80")
    _install_fakes(monkeypatch, inputs)
    target_variant = "CJ2"
    if failure_stage == "fit":
        original_iterator = runner.iter_global_constant_jump_variants

        def failing_iterator(prediction: FrameTimingPrediction, **kwargs: Any) -> Any:
            for variant, result in original_iterator(prediction, **kwargs):
                if _prediction_index(prediction) == 0 and variant == target_variant:
                    raise ValueError("synthetic current variant fit failure")
                yield variant, result

        monkeypatch.setattr(runner, "iter_global_constant_jump_variants", failing_iterator)
        current_reason = "variant_fit_failure"
        tail_reason = "not_run_after_variant_fit_failure"
        hard_guard_reason = "CJ2_variant_fit_failure"
    else:
        original_serialize = runner._serialize_variant_result
        candidate_zero_fingerprint = hashlib.sha256(b"candidate-0").hexdigest()

        def failing_serialize(result: Any, **kwargs: Any) -> Any:
            if (
                result.variant == target_variant
                and kwargs["expected_candidate_fingerprint"] == candidate_zero_fingerprint
            ):
                raise ValueError("synthetic current variant serialization failure")
            return original_serialize(result, **kwargs)

        monkeypatch.setattr(runner, "_serialize_variant_result", failing_serialize)
        current_reason = "schema_or_serialization_failure"
        tail_reason = "not_run_after_schema_or_serialization_failure"
        hard_guard_reason = "CJ2_schema_or_serialization_failure"

    summary = _run(inputs)
    row = _read_jsonl(inputs["output"])[0]

    assert row["variants"]["CJ0"]["status"] == "accepted"
    assert row["variants"]["CJ1"]["status"] == "accepted"
    assert row["variants"]["CJ2"]["status"] == "failed"
    assert row["variants"]["CJ2"]["reason"] == current_reason
    assert row["variants"]["CJ3"]["status"] == "not_run"
    assert row["variants"]["CJ3"]["reason"] == tail_reason
    assert row["selection"]["source"] == "current_v2"
    assert row["projection_flags"]["hard_guard_reasons"] == [hard_guard_reason]
    assert summary["denominators"]["projection_evaluable_count"] == 80
    assert summary["denominators"]["selected_fallback_count"] == 1
    assert summary["hard_guards"]["violations"] == [
        {"reason": hard_guard_reason, "row_indices": [0]}
    ]


def test_partial_variant_timeout_row_is_resume_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_stage_inputs(tmp_path, stage="repair80")
    _install_fakes(monkeypatch, inputs)
    monkeypatch.setattr(runner, "PER_AUDIO_TIMEOUT_SECONDS", 0.05)
    _install_targeted_variant_timeout(monkeypatch, target_variant="CJ2", stage="fit")
    first_summary = _run(inputs)
    first_output = inputs["output"].read_bytes()

    calls = _install_fakes(monkeypatch, inputs)
    second_summary = _run(inputs)

    assert first_summary["hard_guards"]["ok"] is False
    assert second_summary["resume"]["reused_success"] == 80
    assert second_summary["resume"]["processed"] == 0
    assert calls["cache"] == []
    assert inputs["output"].read_bytes() == first_output
    row = _read_jsonl(inputs["output"])[0]
    assert row["variants"]["CJ0"]["status"] == "accepted"
    assert row["variants"]["CJ1"]["status"] == "accepted"
    assert row["variants"]["CJ2"]["reason"] == "timeout"
    assert row["variants"]["CJ3"]["reason"] == "timeout"


@pytest.mark.parametrize("mutating_stage", ["current_v2", "candidate_extraction", "variant_fit"])
def test_cache_mutation_during_projection_path_marks_row_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutating_stage: str,
) -> None:
    inputs = _write_stage_inputs(tmp_path, stage="repair80")
    _install_fakes(monkeypatch, inputs, mutate_during={0: mutating_stage})

    with pytest.raises(RuntimeError, match="declared cache file changed"):
        _run(inputs)
    rows = _read_jsonl(inputs["output"])
    row = rows[0]

    assert row["cache"]["status"] == "mutated_during_execution"
    assert row["current_v2"]["status"] == "not_run"
    assert row["current_v2"]["reason"] == "cache_mutated_during_execution"
    assert row["candidate_extraction"]["status"] == "not_run"
    assert row["variants"]["CJ3"]["reason"] == "cache_mutated_during_execution"
    assert row["selection"]["source"] is None
    assert row["projection_flags"]["projection_evaluable"] is False
    assert row["projection_flags"]["hard_guard_reasons"] == ["cache_mutated_during_execution"]
    assert not inputs["summary"].exists()


def test_final_cache_rescan_rejects_mutated_reused_row_without_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_stage_inputs(tmp_path, stage="repair80")
    _install_fakes(monkeypatch, inputs)
    _run(inputs)

    calls = _install_fakes(monkeypatch, inputs)
    second_summary = tmp_path / "second-summary.json"
    original_require_sources = runner._require_sources_unchanged

    def mutate_reused_cache(source_identities: Any) -> None:
        inputs["cache_paths"][0].write_bytes(b"mutated-after-reuse")
        original_require_sources(source_identities)

    monkeypatch.setattr(runner, "_require_sources_unchanged", mutate_reused_cache)

    with pytest.raises(RuntimeError, match="declared cache file changed"):
        _run(inputs, summary_json_path=second_summary)

    assert calls["cache"] == []
    assert not second_summary.exists()


@pytest.mark.parametrize("slow_stage", ["current_v2", "candidate_extraction", "variant_fit", "serialization"])
def test_timeout_propagates_through_every_broad_row_stage_handler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    slow_stage: str,
) -> None:
    inputs = _write_stage_inputs(tmp_path, stage="repair80")
    _install_fakes(monkeypatch, inputs)
    monkeypatch.setattr(runner, "PER_AUDIO_TIMEOUT_SECONDS", 0.02)

    if slow_stage == "current_v2":
        class SlowGridFitter:
            def fit(self, _prediction: FrameTimingPrediction) -> Any:
                time.sleep(0.05)
                raise AssertionError("timer failed to interrupt v2")

        monkeypatch.setattr(runner, "GridFitter", SlowGridFitter)
    elif slow_stage == "candidate_extraction":
        original = runner.extract_global_constant_jump_candidates

        def slow_extract(prediction: FrameTimingPrediction) -> Any:
            time.sleep(0.05)
            return original(prediction)

        monkeypatch.setattr(runner, "extract_global_constant_jump_candidates", slow_extract)
    elif slow_stage == "variant_fit":
        original = runner.iter_global_constant_jump_variants

        def slow_variants(*args: Any, **kwargs: Any) -> Any:
            for item in original(*args, **kwargs):
                time.sleep(0.05)
                yield item

        monkeypatch.setattr(runner, "iter_global_constant_jump_variants", slow_variants)
    else:
        original = runner._serialize_variant_result

        def slow_serialize(*args: Any, **kwargs: Any) -> Any:
            time.sleep(0.05)
            return original(*args, **kwargs)

        monkeypatch.setattr(runner, "_serialize_variant_result", slow_serialize)

    summary = _run(inputs)
    rows = _read_jsonl(inputs["output"])

    assert summary["hard_guards"]["ok"] is False
    assert all(row["variants"]["CJ3"]["reason"] == "timeout" for row in rows)
    if slow_stage == "current_v2":
        assert all(row["projection_flags"]["projection_evaluable"] is False for row in rows)
    else:
        assert all(row["selection"]["source"] == "current_v2" for row in rows)


def test_projection_module_has_no_forbidden_evidence_or_transport_imports_and_path_trap_is_idle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = Path(runner.__file__ or "")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            imports.add(module_name)
            imports.update(f"{module_name}.{alias.name}" for alias in node.names)
    assert not any(
        token in imported
        for imported in imports
        for token in ("oracle", "evidence", "labels", "requests", "urllib", "socket")
    )

    inputs = _write_stage_inputs(tmp_path, stage="repair80")
    _install_fakes(monkeypatch, inputs)
    original_open = Path.open

    def trapping_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.suffix == ".osu":
            raise AssertionError("projection runner attempted to open a forbidden comparator path")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", trapping_open)
    summary = _run(inputs)
    assert summary["results"]["result_count"] == 80


def test_identity_manifest_order_duplicates_and_stage_counts_fail_before_cache_access(tmp_path: Path) -> None:
    inputs = _write_stage_inputs(tmp_path, stage="repair80")
    rows = _read_jsonl(inputs["identity"])
    rows[0], rows[1] = rows[1], rows[0]
    _write_jsonl(inputs["identity"], rows)
    with pytest.raises(ValueError, match="mismatch"):
        _run(inputs)

    inputs = _write_stage_inputs(tmp_path / "duplicate", stage="repair80")
    rows = _read_jsonl(inputs["identity"])
    rows[1]["cache_audio_key"] = rows[0]["cache_audio_key"]
    _write_jsonl(inputs["identity"], rows)
    with pytest.raises(ValueError, match="duplicate identity cache_audio_key"):
        _run(inputs)

    inputs = _write_stage_inputs(tmp_path / "short", stage="repair80")
    rows = _read_jsonl(inputs["identity"])
    _write_jsonl(inputs["identity"], rows[:-1])
    with pytest.raises(ValueError, match="exactly 80"):
        _run(inputs)


@pytest.mark.parametrize(
    ("stage", "expected_count"),
    [("repair80", 80), ("holdout100", 100), ("broad500", 500), ("full5050", 5050)],
)
def test_all_frozen_stage_counts_reconcile_ordered_compact_inputs(
    tmp_path: Path,
    stage: str,
    expected_count: int,
) -> None:
    inputs = _write_stage_inputs(tmp_path, stage=stage, create_cache_files=False)

    identities, _identity_source = runner._load_identity_rows(
        inputs["identity"],
        expected_stage=stage,
    )
    _manifest, entries, _selection_source = runner._load_selection_manifest(
        inputs["selection"],
        expected_stage=stage,
    )
    runner._reconcile_identities(identities, entries, expected_stage=stage)

    assert runner.STAGE_AUDIO_COUNTS[stage] == expected_count
    assert len(identities) == expected_count
    assert len(entries) == expected_count


def _prior_weak_summary_payload(
    *,
    behavior_fingerprint: str,
    config_fingerprint: str,
    stage: str = "repair80",
) -> dict[str, Any]:
    count = runner.STAGE_AUDIO_COUNTS[stage]
    decision, next_action = runner._PRIOR_WEAK_DECISION[stage]
    output = {
        "path": f"/synthetic/{stage}-weak.jsonl",
        "sha256": "d" * 64,
        "row_count": count,
    }
    denominators = {
        "stage_audio_count": count,
        "cache_valid_count": count,
        "projection_evaluable_count": count,
        "comparison_eligible_count": count,
        "pure_CJ3_phase_count": count,
        "pure_CJ3_phase_coverage": 1.0,
        "selected_safety_phase_count": count,
        "selected_fallback_count": 0,
        "selected_fallback_rate": 0.0,
    }
    gates = {
        gate_name: {
            "value": True if gate_name == "projection_hard_guards" else 1.0,
            "numerator": 1,
            "denominator": 1,
            "status": "pass",
            "threshold": "synthetic frozen gate",
            "reason": None,
            "decision_gate": gate_name
            not in {
                "jump_mean_phase_ratio",
                "jump_endpoint_drift_mean_ratio",
                *({"phase_denominator_available"} if stage == "repair80" else set()),
            },
        }
        for gate_name in runner._REQUIRED_WEAK_STAGE_GATES
    }
    stage_gates = {
        "schema": runner.WEAK_STAGE_GATES_SCHEMA,
        "stage": stage,
        "sample_counts": {"all": count},
        "denominator_notes": {},
        "gates": gates,
    }
    hard_guards = {"ok": True, "violations": [], "source": "projection_summary"}
    evaluator = {
        "schema": "pulsefield_model.timing_v3_exp004_weak_evaluator_source_v1",
        "evaluator_path": "/synthetic/exp004_weak_evidence.py",
        "evaluator_sha256": "e" * 64,
        "metrics_path": "/synthetic/exp004_metrics.py",
        "metrics_sha256": "f" * 64,
        "canonical_bpm_binding": {
            "canonicalization": "bpm-80-160",
            "function_module": "synthetic",
            "function_qualname": "synthetic",
            "source_path": "/synthetic/canonicalization.py",
            "source_sha256": "1" * 64,
        },
    }
    source_projection = {
        "stage": stage,
        "behavior_fingerprint": behavior_fingerprint,
        "config_fingerprint": config_fingerprint,
        "run_fingerprint": "a" * 64,
        "projection_jsonl_sha256": "b" * 64,
        "projection_summary_sha256": "c" * 64,
        "projection_summary_hard_guards_ok": True,
        "projection_summary_formal_execution_ready": True,
    }
    binding = {
        "schema": runner.WEAK_PROTOCOL_BINDING_SCHEMA,
        "source_projection": source_projection,
        "baseline": {
            "path": "/synthetic/baseline.jsonl",
            "sha256": "2" * 64,
            "row_count": 5050,
            "schema": "pulsefield_model.timing_v3_cache_backed_v2_baseline_result_v2",
        },
        "evaluator": evaluator,
        "output": output,
        "denominators": denominators,
        "stage_gates": stage_gates,
        "decision": decision,
        "next_action": next_action,
        "hard_guards": hard_guards,
    }
    return {
        "schema": runner.WEAK_SUMMARY_SCHEMA,
        "experiment": "timing_v3_experiment_004",
        "stage": stage,
        "source": {
            "projection_jsonl": {"path": "/synthetic/projection.jsonl", "sha256": "b" * 64},
            "projection_summary": {
                "path": "/synthetic/projection.summary.json",
                "sha256": "c" * 64,
                "schema": runner.SUMMARY_SCHEMA,
                "stage": stage,
                "behavior_fingerprint": behavior_fingerprint,
                "config_fingerprint": config_fingerprint,
                "run_fingerprint": "a" * 64,
                "output_path": "/synthetic/projection.jsonl",
                "output_sha256": "b" * 64,
                "output_row_count": count,
                "hard_guards_ok": True,
                "formal_execution_ready": True,
            },
            "baseline_jsonl": {"path": "/synthetic/baseline.jsonl", "sha256": "2" * 64},
            "evaluator": evaluator,
        },
        "output": output,
        "denominators": denominators,
        "stage_gates": stage_gates,
        "hard_guards": hard_guards,
        "decision": decision,
        "next_action": next_action,
        "protocol_binding": binding,
    }


def _write_stage_inputs(
    root: Path,
    *,
    stage: str,
    create_cache_files: bool = True,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    count = runner.STAGE_AUDIO_COUNTS[stage]
    keys = [f"cache-key-{index:04d}" for index in range(count)]
    groups = [f"audio-group-{index:04d}" for index in range(count)]
    identity_path = root / "identities.jsonl"
    selection_path = root / "selection.json"
    output_path = root / "projection.jsonl"
    summary_path = root / "summary.json"
    prior_path = root / "prior.json"
    cache_root = root / "cache"
    upstream_path = root / "synthetic-upstream.json"
    upstream_payload = {
        "schema": "pulsefield_model.timing_v3_exp004_synthetic_upstream_v1",
        "stage": stage,
        "ordered_cache_audio_keys": keys,
    }
    _write_json(upstream_path, upstream_payload)
    upstream_source = runner.exp004_protocol.build_exp004_upstream_source(
        source_schema=str(upstream_payload["schema"]),
        source_path=upstream_path,
        source_fingerprint_sha256=_json_sha256(upstream_payload),
        row_count=count,
        ordered_cache_audio_keys_sha256=(
            runner.exp004_protocol.ordered_cache_audio_keys_sha256(keys)
        ),
    )
    runner.exp004_protocol.build_exp004_execution_inputs(
        stage=stage,
        ordered_identities=[
            {
                "cache_audio_key": cache_key,
                "audio_group_key": audio_group_key,
                "resolved_audio_path": (root / f"audio-{index:04d}.mp3").as_posix(),
            }
            for index, (cache_key, audio_group_key) in enumerate(
                zip(keys, groups, strict=True)
            )
        ],
        upstream_source=upstream_source,
        identity_rows_jsonl_path=identity_path,
        execution_selection_manifest_path=selection_path,
    )

    cache_config = BeatThisFramePredictionCacheConfig(cache_root=cache_root)
    cache_paths = [beatthis_frame_prediction_cache_path(key, cache_config) for key in keys]
    if create_cache_files:
        for index, path in enumerate(cache_paths):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"synthetic-cache-{index}".encode("utf-8"))

    return {
        "root": root,
        "stage": stage,
        "keys": keys,
        "groups": groups,
        "identity": identity_path,
        "selection": selection_path,
        "output": output_path,
        "summary": summary_path,
        "prior": prior_path,
        "cache_root": cache_root,
        "cache_paths": cache_paths,
        "upstream": upstream_path,
    }


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    inputs: dict[str, Any],
    *,
    cj3_fail: set[int] | None = None,
    v2_fail: set[int] | None = None,
    invalid_cj3: set[int] | None = None,
    corrupt_cache: set[int] | None = None,
    mutate_cache: set[int] | None = None,
    mutate_during: dict[int, str] | None = None,
) -> dict[str, list[Any]]:
    cj3_fail = set(cj3_fail or ())
    v2_fail = set(v2_fail or ())
    invalid_cj3 = set(invalid_cj3 or ())
    corrupt_cache = set(corrupt_cache or ())
    mutate_cache = set(mutate_cache or ())
    mutate_during = dict(mutate_during or {})
    index_by_key = {key: index for index, key in enumerate(inputs["keys"])}
    calls: dict[str, list[Any]] = {"cache": [], "v2": [], "extract": [], "variants": []}

    def mutate_if_requested(index: int, stage: str) -> None:
        if mutate_during.get(index) == stage:
            inputs["cache_paths"][index].write_bytes(f"mutated-during-{stage}".encode("utf-8"))

    def load_cache(cache_audio_key: str, _config: BeatThisFramePredictionCacheConfig) -> FrameTimingPrediction | None:
        index = index_by_key[cache_audio_key]
        calls["cache"].append(cache_audio_key)
        if index in mutate_cache:
            inputs["cache_paths"][index].write_bytes(b"mutated-during-load")
        if index in corrupt_cache:
            raise BeatThisFramePredictionCacheError("synthetic corrupt cache")
        if not inputs["cache_paths"][index].exists():
            return None
        beat = np.zeros(500, dtype=np.float32)
        beat[0] = np.float32(index / 10_000.0)
        return FrameTimingPrediction(
            provider="beat_this",
            checkpoint_path="final0",
            source_path=f"forbidden/source/{index}.mp3",
            beat_prob=beat,
            downbeat_prob=np.zeros_like(beat),
            frame_rate_hz=50.0,
        )

    class FakeGridFitter:
        def fit(self, prediction: FrameTimingPrediction) -> Any:
            index = _prediction_index(prediction)
            calls["v2"].append(index)
            mutate_if_requested(index, "current_v2")
            if index in v2_fail:
                raise ValueError("synthetic v2 failure")
            return SimpleNamespace(
                grid=FittedTimingGrid((TimingSegment(offset_ms=0.0, beat_length_ms=500.0),)),
                score=1.0,
                diagnostics=_V2Diagnostics(marker=index),
            )

    def extract(prediction: FrameTimingPrediction) -> GlobalConstantJumpCandidateSet:
        assert prediction.source_path is None
        assert prediction.beat_prob.flags.writeable is False
        index = _prediction_index(prediction)
        calls["extract"].append(index)
        mutate_if_requested(index, "candidate_extraction")
        fingerprint = hashlib.sha256(f"candidate-{index}".encode()).hexdigest()
        diagnostics = GlobalConstantJumpCandidateDiagnostics(
            candidate_contract_version=CANDIDATE_CONTRACT_VERSION,
            constants_json_sha256=GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON_SHA256,
            pulse_correlation_version=PULSE_CORRELATION_VERSION,
            boundary_candidate_score_version=BOUNDARY_CANDIDATE_SCORE_VERSION,
            frame_count=500,
            frame_rate_hz=50.0,
            coverage_start_ms=0.0,
            coverage_end_ms=10_000.0,
            min_period_frames=3,
            max_period_frames=150,
            beat_peak_count=0,
            downbeat_peak_count=0,
            tempo_candidate_count=1,
            origin_candidate_count=1,
            boundary_candidate_count=0,
            input_signal_sha256=hashlib.sha256(f"signal-{index}".encode()).hexdigest(),
            candidate_fingerprint=fingerprint,
        )
        return GlobalConstantJumpCandidateSet((), (), (), (), (), diagnostics)

    def iter_variants(
        prediction: FrameTimingPrediction,
        *,
        variants: Any,
        attempt_cap: int,
        candidate_set: GlobalConstantJumpCandidateSet,
    ) -> Any:
        assert prediction.source_path is None
        assert attempt_cap > 0
        index = _prediction_index(prediction)
        calls["variants"].append(tuple(variants))
        mutate_if_requested(index, "variant_fit")
        for variant in variants:
            failed = variant == "CJ3" and index in cj3_fail
            grid: Any = None if failed else _valid_v3_grid()
            if variant == "CJ3" and index in invalid_cj3:
                grid = object()
            reason = "synthetic_no_path" if failed else None
            diagnostics = _variant_diagnostics(
                variant,
                candidate_fingerprint=candidate_set.diagnostics.candidate_fingerprint,
                grid=grid if isinstance(grid, TimingV3Grid) else None,
                reason=reason,
            )
            yield (
                variant,
                GlobalConstantJumpResult(
                    variant=variant,
                    grid=grid,
                    diagnostics=diagnostics,
                    reason=reason,
                ),
            )

    monkeypatch.setattr(runner, "load_beatthis_frame_prediction_cache", load_cache)
    monkeypatch.setattr(runner, "GridFitter", FakeGridFitter)
    monkeypatch.setattr(runner, "extract_global_constant_jump_candidates", extract)
    monkeypatch.setattr(runner, "iter_global_constant_jump_variants", iter_variants)
    return calls


def _variant_diagnostics(
    variant: str,
    *,
    candidate_fingerprint: str,
    grid: TimingV3Grid | None,
    reason: str | None,
) -> GlobalConstantJumpDiagnostics:
    grid_fingerprint = _json_sha256(grid.to_dict()) if grid is not None else None
    return GlobalConstantJumpDiagnostics(
        variant=variant,
        candidate_contract_version=CANDIDATE_CONTRACT_VERSION,
        constants_json_sha256=GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON_SHA256,
        coverage_start_ms=0.0,
        coverage_end_ms=10_000.0,
        frame_count=500,
        frame_rate_hz=50.0,
        min_period_frames=3,
        max_period_frames=150,
        beat_peak_count=0,
        downbeat_peak_count=0,
        tempo_candidate_count=1,
        origin_candidate_count=1,
        boundary_candidate_count=0,
        section_attempt_count=1,
        edge_count_cache_size=1,
        section_score_cache_size=1,
        beam_pruned_state_count=0,
        selected_section_count=1 if grid is not None else 0,
        selected_origin_time_ms=0.0 if grid is not None else None,
        selected_downbeat_phase=0 if grid is not None else None,
        objective=0.0 if grid is not None else None,
        duration_objective=0.0 if grid is not None else None,
        transition_objective=0.0 if grid is not None else None,
        alias_switch_count=0,
        max_boundary_displacement_ms=0.0,
        fallback_reason=reason,
        input_signal_sha256="a" * 64,
        candidate_fingerprint=candidate_fingerprint,
        replay_fingerprint="b" * 64,
        grid_fingerprint=grid_fingerprint,
    )


def _valid_v3_grid() -> TimingV3Grid:
    return TimingV3Grid(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(ConstantTimingSection(start_beat=0, end_beat=20, bpm=120.0),),
        coverage_start_ms=0.0,
        coverage_end_ms=10_000.0,
    )


def _observe_timeout_active_stages(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    observed: list[str] = []
    original_apply_timeout = runner._apply_timeout

    def observing_apply_timeout(state: dict[str, Any]) -> None:
        observed.append(str(state["active_stage"]))
        original_apply_timeout(state)

    monkeypatch.setattr(runner, "_apply_timeout", observing_apply_timeout)
    return observed


def _install_targeted_variant_timeout(
    monkeypatch: pytest.MonkeyPatch,
    *,
    target_variant: str,
    stage: str,
) -> list[str]:
    observed = _observe_timeout_active_stages(monkeypatch)
    if stage == "fit":
        original_iterator = runner.iter_global_constant_jump_variants

        def slow_iterator(prediction: FrameTimingPrediction, **kwargs: Any) -> Any:
            for variant, result in original_iterator(prediction, **kwargs):
                if _prediction_index(prediction) == 0 and variant == target_variant:
                    time.sleep(0.2)
                yield variant, result

        monkeypatch.setattr(runner, "iter_global_constant_jump_variants", slow_iterator)
    elif stage == "serialize":
        original_serialize = runner._serialize_variant_result
        candidate_zero_fingerprint = hashlib.sha256(b"candidate-0").hexdigest()

        def slow_serialize(result: Any, **kwargs: Any) -> Any:
            if (
                result.variant == target_variant
                and kwargs["expected_candidate_fingerprint"] == candidate_zero_fingerprint
            ):
                time.sleep(0.2)
            return original_serialize(result, **kwargs)

        monkeypatch.setattr(runner, "_serialize_variant_result", slow_serialize)
    else:
        raise AssertionError(f"unsupported synthetic timeout stage: {stage}")
    return observed


def _assert_timeout_variant_tail(row: dict[str, Any], *, target_variant: str) -> None:
    target_index = GLOBAL_CONSTANT_JUMP_VARIANTS.index(target_variant)
    for index, variant in enumerate(GLOBAL_CONSTANT_JUMP_VARIANTS):
        payload = row["variants"][variant]
        if index < target_index:
            assert payload["status"] == "accepted"
            assert payload["reason"] is None
        else:
            assert payload["status"] == "not_run"
            assert payload["reason"] == "timeout"


def _assert_partial_timeout_summary(summary: dict[str, Any], *, target_variant: str) -> None:
    target_index = GLOBAL_CONSTANT_JUMP_VARIANTS.index(target_variant)
    for index, variant in enumerate(GLOBAL_CONSTANT_JUMP_VARIANTS):
        if index < target_index:
            assert summary["results"]["variant_status_counts"][variant] == {"accepted": 80}
            assert summary["results"]["variant_reason_counts"][variant] == {}
        else:
            assert summary["results"]["variant_status_counts"][variant] == {
                "accepted": 79,
                "not_run": 1,
            }
            assert summary["results"]["variant_reason_counts"][variant] == {"timeout": 1}
    assert summary["hard_guards"] == {
        "ok": False,
        "violations": [
            {"reason": f"{variant}_timeout", "row_indices": [0]}
            for variant in GLOBAL_CONSTANT_JUMP_VARIANTS[target_index:]
        ],
    }


def _prediction_index(prediction: FrameTimingPrediction) -> int:
    return int(round(float(prediction.beat_prob[0]) * 10_000.0))


def _cache_config(inputs: dict[str, Any]) -> BeatThisFramePredictionCacheConfig:
    return BeatThisFramePredictionCacheConfig(cache_root=inputs["cache_root"])


def _run(inputs: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    kwargs = {
        "stage": inputs["stage"],
        "identity_rows_jsonl_path": inputs["identity"],
        "selection_manifest_path": inputs["selection"],
        "prior_stage_summary_path": None,
        "cache_root": inputs["cache_root"],
        "output_jsonl_path": inputs["output"],
        "summary_json_path": inputs["summary"],
        "checkpoint_every": runner.STAGE_AUDIO_COUNTS[inputs["stage"]],
    }
    kwargs.update(overrides)
    return runner.run_exp004_projection(**kwargs)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
