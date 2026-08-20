from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from pulsefield_model.timing.evaluation import exp007_artifacts as artifacts
from pulsefield_model.timing.evaluation import exp007_protocol as protocol
from pulsefield_model.timing.evaluation.exp007_metrics import (
    canonical_sha256,
    rate_value,
    ratio_value,
    stats_value,
)
from pulsefield_model.timing.evaluation.exp007_weak_evidence import (
    BoundaryEvidence,
    BoundarySummary,
    ComparatorAvailability,
    DriftSummary,
    ObjectGridSummary,
    PhaseSummary,
    PredictionRowRef,
    Repair80MetricRow,
    WeakMetricRow,
    evaluate_repair80,
    evaluate_schedule_weak_veto,
    make_repair80_summary,
    make_repair80_summary_from_rows,
    make_schedule_weak_failure_record,
    make_schedule_weak_hard_failure_outcome,
    make_schedule_weak_success_outcome,
    make_schedule_weak_veto_summary,
    make_weak_row,
    schedule_weak_resume_action,
    validate_repair80_summary,
    validate_repair80_summary_authoritatively,
    validate_schedule_weak_failure_record,
    validate_schedule_weak_outcome,
    validate_schedule_weak_veto_summary,
    validate_weak_row,
    validate_weak_resume_prefix,
)
from pulsefield_model.timing.v3 import global_constant_jump as candidate_source


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _boundary(f1: float = 0.8, *, valid_change: bool = True) -> BoundaryEvidence:
    return BoundaryEvidence(True, 1, valid_change, ratio_value(f1, 1.0))


def _weak(
    index: int,
    *,
    stage: str = "schedule16",
    audio_group_key: str | None = None,
    arm: str = "S64",
    current_phase: float = 10.0,
    pure_phase: float = 10.0,
    current_drift: float = 10.0,
    pure_drift: float = 10.0,
    current_f1: float = 0.8,
    pure_f1: float = 0.8,
    comparator_state: str = "available",
    candidate_status: str = "accepted",
    baseline_status: str = "accepted",
) -> WeakMetricRow:
    candidate_accepted = candidate_status == "accepted"
    baseline_accepted = baseline_status == "accepted"
    selected_available = candidate_accepted or baseline_accepted
    comparator_available = comparator_state == "available"
    current_matched = comparator_available and baseline_accepted
    pure_matched = comparator_available and candidate_accepted and baseline_accepted
    selected_matched = comparator_available and selected_available and baseline_accepted
    phase_current = stats_value([current_phase]) if current_matched else None
    drift_current = stats_value([current_drift]) if current_matched else None
    phase_pure = stats_value([pure_phase]) if pure_matched else None
    drift_pure = stats_value([pure_drift]) if pure_matched else None
    phase_product = stats_value([pure_phase if candidate_accepted else current_phase]) if selected_matched else None
    drift_product = stats_value([pure_drift if candidate_accepted else current_drift]) if selected_matched else None
    unavailable = BoundaryEvidence.unavailable()
    current_boundary = _boundary(current_f1) if current_matched else unavailable
    pure_boundary = _boundary(pure_f1) if pure_matched else unavailable
    selected_boundary = _boundary(pure_f1 if candidate_accepted else current_f1) if selected_matched else unavailable
    return WeakMetricRow(
        stage=stage,  # type: ignore[arg-type]
        row_index=index,
        cache_audio_key=f"audio-{index:03d}",
        audio_group_key=audio_group_key or f"group-{index:03d}",
        prediction_row_sha256=_sha(f"prediction-{stage}-{index}"),
        weak_row_payload_sha256=_sha(f"weak-{stage}-{index}"),
        schedule_arm=arm,
        comparator_state=comparator_state,  # type: ignore[arg-type]
        candidate_status=candidate_status,  # type: ignore[arg-type]
        baseline_status=baseline_status,  # type: ignore[arg-type]
        selected_status="accepted" if selected_available else "unavailable",
        product_grid_available=selected_available,
        current_v2_phase_matched=current_matched,
        pure_exp006_phase_matched=pure_matched,
        selected_safety_phase_matched=selected_matched,
        current_v2_phase_ms=phase_current,
        pure_exp006_phase_ms=phase_pure,
        product_phase_ms=phase_product,
        current_v2_alias_max_prefix_ms=drift_current,
        pure_exp006_alias_max_prefix_ms=drift_pure,
        product_alias_max_prefix_ms=drift_product,
        current_v2_boundary=current_boundary,
        pure_exp006_boundary=pure_boundary,
        selected_boundary=selected_boundary,
    )


def _selected(rows: list[WeakMetricRow]) -> list[PredictionRowRef]:
    return [
        PredictionRowRef(
            row.row_index,
            row.cache_audio_key,
            row.prediction_row_sha256,
            row.schedule_arm,
        )
        for row in rows
    ]


def _schedule_deps() -> dict[str, str]:
    return {
        "four_arm_stage_summary_sha256": _sha("four-arm"),
        "candidate_global_manifest_sha256": _sha("global"),
        "source_closure_fingerprint_sha256": _sha("source"),
        "source_selection_sha256": _sha("selection"),
    }


def _schedule_summary_bundle() -> tuple[
    list[WeakMetricRow],
    object,
    dict[str, object],
    dict[str, str],
]:
    rows = [_weak(index) for index in range(16)]
    evaluation = evaluate_schedule_weak_veto(
        rows,
        selected_rows=_selected(rows),
        selected_schedule_arm="S64",
    )
    deps = _schedule_deps()
    summary = make_schedule_weak_veto_summary(
        evaluation,
        schedule_arm="S64",
        **deps,
    )
    return rows, evaluation, summary, deps


def _available_comparator() -> ComparatorAvailability:
    return ComparatorAvailability(
        "available",
        valid_difficulty_count=1,
        invalid_difficulty_count=0,
        reason=None,
        comparator_payloads_sha256=_sha("comparator"),
    )


def _boundary_summary(
    *,
    tp: int = 2,
    fp: int = 1,
    fn: int = 0,
    weak_consensus_supported_count: int = 1,
) -> BoundarySummary:
    return BoundarySummary(
        eligible=True,
        valid_difficulty_count=1,
        tp=tp,
        fp=fp,
        fn=fn,
        f1=ratio_value(2 * tp, 2 * tp + fp + fn),
        matched_error_ms=stats_value([0.0] * tp),
        weak_consensus_supported_count=weak_consensus_supported_count,
    )


def _object_grid_summary() -> ObjectGridSummary:
    return ObjectGridSummary(
        eligible=True,
        object_count=2,
        start_residual_ms=stats_value([1.0, 2.0]),
        end_residual_ms=stats_value([1.0, 2.0]),
        inlier_count=2,
        inlier_rate=rate_value(2, 2),
    )


def _persisted_weak_row(
    row: WeakMetricRow,
    deps: dict[str, str],
    *,
    stage: str | None = None,
    current_matched: bool | None = None,
    pure_matched: bool | None = None,
    selected_matched: bool | None = None,
) -> dict[str, object]:
    current_resolved = (
        row.current_v2_phase_matched
        if current_matched is None
        else current_matched
    )
    pure_resolved = (
        row.pure_exp006_phase_matched if pure_matched is None else pure_matched
    )
    selected_resolved = (
        row.selected_safety_phase_matched
        if selected_matched is None
        else selected_matched
    )
    phase = PhaseSummary(
        current_v2_ms=row.current_v2_phase_ms if current_resolved else None,
        pure_exp006_ms=row.pure_exp006_phase_ms if pure_resolved else None,
        product_ms=row.product_phase_ms if selected_resolved else None,
    )
    drift = DriftSummary(
        current_v2_alias_max_prefix_ms=(
            row.current_v2_alias_max_prefix_ms if current_resolved else None
        ),
        pure_exp006_alias_max_prefix_ms=(
            row.pure_exp006_alias_max_prefix_ms if pure_resolved else None
        ),
        product_alias_max_prefix_ms=(
            row.product_alias_max_prefix_ms if selected_resolved else None
        ),
    )
    return make_weak_row(
        stage=stage or row.stage,
        schedule_arm=row.schedule_arm,
        row_index=row.row_index,
        cache_audio_key=row.cache_audio_key,
        audio_group_key=row.audio_group_key,
        prediction_row_sha256=row.prediction_row_sha256,
        four_arm_stage_summary_sha256=deps["four_arm_stage_summary_sha256"],
        candidate_global_manifest_sha256=deps["candidate_global_manifest_sha256"],
        source_selection_sha256=deps["source_selection_sha256"],
        comparator_availability=_available_comparator(),
        current_v2_phase_matched=current_resolved,
        pure_exp006_phase_matched=pure_resolved,
        selected_safety_phase_matched=selected_resolved,
        phase_metrics_summary=phase,
        drift_metrics_summary=drift,
        current_v2_boundary_summary=(
            _boundary_summary() if current_resolved else BoundarySummary.ineligible()
        ),
        pure_exp006_boundary_summary=(
            _boundary_summary() if pure_resolved else BoundarySummary.ineligible()
        ),
        selected_boundary_summary=(
            _boundary_summary() if selected_resolved else BoundarySummary.ineligible()
        ),
        object_grid_summary=(
            _object_grid_summary()
            if selected_resolved
            else ObjectGridSummary.ineligible()
        ),
    )


def _completed_row_refs(rows: list[Repair80MetricRow]) -> list[dict[str, object]]:
    return [
        protocol.make_completed_row_ref(
            row_index=row.weak.row_index,
            cache_audio_key=row.weak.cache_audio_key,
            identity_payload_sha256=_sha(f"identity-{row.weak.row_index}"),
            row_payload_sha256=row.weak.prediction_row_sha256,
            candidate_reference_entry_payload_sha256=_sha(
                f"reference-{row.weak.row_index}"
            ),
        )
        for row in rows
    ]


def _completed_row_refs_from_manifest(
    prediction_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, object]]:
    return [
        protocol.make_completed_row_ref(
            row_index=row["row_index"],
            cache_audio_key=row["cache_audio_key"],
            identity_payload_sha256=row["identity_payload_sha256"],
            row_payload_sha256=row["row_payload_sha256"],
            candidate_reference_entry_payload_sha256=entry["entry_payload_sha256"],
        )
        for row, entry in zip(prediction_rows, manifest["entries"], strict=True)
    ]


def _weak_row_refs_from_rows(
    weak_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        protocol.validate_weak_row_ref(
            {
                "row_index": row["row_index"],
                "cache_audio_key": row["cache_audio_key"],
                "prediction_row_sha256": row["prediction_row_sha256"],
                "weak_row_payload_sha256": row["weak_row_payload_sha256"],
            }
        )
        for row in weak_rows
    ]


def _rehash_weak_row_payload(row: dict[str, Any]) -> None:
    row["deterministic_projection_sha256"] = protocol.canonical_json_sha256(
        {
            key: row[key]
            for key in protocol.WEAK_ROW_FIELDS
            if key not in {
                "deterministic_projection_sha256",
                "weak_row_payload_sha256",
            }
        }
    )
    row["weak_row_payload_sha256"] = protocol.payload_hash(
        row,
        "weak_row_payload_sha256",
    )


def _repair_identity_label_sources() -> dict[str, Any]:
    label_rows = []
    for index in range(80):
        if index < 5:
            label = "stable"
            is_long = True
        elif index < 20:
            label = "jump_candidate"
            is_long = False
        else:
            label = "dense"
            is_long = False
        label_row = {
            "cache_audio_key": f"audio-{index:03d}",
            "audio_group_key": f"group-{index:03d}",
            "label_stratum": label,
            "source_long_track": is_long,
            "duration_ms": 120_000 + index,
            "source": {
                "cache_audio_key": f"audio-{index:03d}",
                "long_track": is_long,
            },
            "label": {"stratum": label},
        }
        label_rows.append(label_row)
    identity_rows = [
        protocol.make_identity(
            stage=protocol.EXP007_REPAIR_STAGE,
            row_index=index,
            source_row_index=10_000 + index,
            cache_audio_key=label_row["cache_audio_key"],
            audio_group_key=label_row["audio_group_key"],
            label_stratum=label_row["label_stratum"],
            source_long_track=label_row["source_long_track"],
            duration_ms=label_row["duration_ms"],
            label_source_sha256=protocol.canonical_json_sha256(label_row),
        )
        for index, label_row in enumerate(label_rows)
    ]
    identity_artifact, identity_source = _source_artifact_and_ref(
        protocol.SOURCE_REPAIR80_IDENTITY_ARTIFACT_SCHEMA,
        identity_rows,
    )
    label_artifact, label_source = _source_artifact_and_ref(
        protocol.SOURCE_LABELS_ARTIFACT_SCHEMA,
        label_rows,
    )
    return {
        "repair80_identity_rows": identity_rows,
        "repair80_label_rows": label_rows,
        "repair80_identity_source_artifact": protocol.canonical_json_bytes(
            identity_artifact
        ),
        "repair80_label_source_artifact": protocol.canonical_json_bytes(
            label_artifact
        ),
        "identity_source": identity_source,
        "label_source": label_source,
    }


def _source_artifact_and_ref(
    artifact_schema: str,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = {
        "schema": artifact_schema,
        "rows": [dict(row) for row in rows],
    }
    source = protocol.make_source_ref(
        artifact_schema=artifact_schema,
        sha256=protocol.canonical_json_sha256(artifact),
        row_count=len(rows),
        ordered_rows_sha256=protocol.canonical_json_sha256(artifact["rows"]),
    )
    return artifact, source


def _repair_source_kwargs(source_context: dict[str, Any]) -> dict[str, Any]:
    return {
        "repair80_identity_source_artifact": source_context[
            "repair80_identity_source_artifact"
        ],
        "repair80_label_source_artifact": source_context[
            "repair80_label_source_artifact"
        ],
        "repair80_identity_rows": source_context["repair80_identity_rows"],
        "repair80_label_rows": source_context["repair80_label_rows"],
    }


def _repair_summary_bundle(
    tmp_path: Path,
    *,
    product_branches: dict[int, tuple[bool, bool]] | None = None,
) -> tuple[
    dict[str, object],
    object,
    dict[str, object],
    dict[str, object],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, object]],
    list[Repair80MetricRow],
    dict[str, Any],
]:
    _, _, weak_summary, deps = _schedule_summary_bundle()
    weak_outcome = make_schedule_weak_success_outcome(weak_summary)
    source_context = _repair_identity_label_sources()
    binding = protocol.make_repair80_input_binding(
        identity_source=source_context["identity_source"],
        label_source=source_context["label_source"],
        four_arm_stage_summary_sha256=deps["four_arm_stage_summary_sha256"],
        candidate_global_manifest_sha256=deps["candidate_global_manifest_sha256"],
        source_selection_sha256=deps["source_selection_sha256"],
        schedule_weak_veto_outcome_sha256=protocol.object_complete_sha256(
            weak_outcome
        ),
    )
    rows, prediction_rows, weak_rows, manifest = _repair_authority_inputs(
        tmp_path,
        binding=binding,
        deps=deps,
        repair80_identity_rows=source_context["repair80_identity_rows"],
        product_branches=product_branches,
    )
    evaluation = evaluate_repair80(
        rows,
        selected_schedule_arm="S64",
        worker_lifetime_rss_bytes=[1, 1, 1, 1],
    )
    summary = make_repair80_summary(
        evaluation,
        schedule_arm="S64",
        repair80_input_binding=binding,
        schedule_weak_veto_outcome=weak_outcome,
        run_config_fingerprint_sha256=_sha("repair-run-config"),
        candidate_reference_manifest=manifest,
        artifact_root=tmp_path,
        row_refs=_completed_row_refs_from_manifest(prediction_rows, manifest),
        weak_row_refs=_weak_row_refs_from_rows(weak_rows),
        aggregate_wall_seconds=10.0,
        worker_lifetime_rss_bytes=[1, 1, 1, 1],
    )
    return (
        summary,
        evaluation,
        binding,
        weak_outcome,
        manifest,
        prediction_rows,
        weak_rows,
        rows,
        source_context,
    )


def test_schedule_winner_only_veto_passes_and_hashes_exact_pair_order() -> None:
    rows = [_weak(index) for index in range(16)]

    result = evaluate_schedule_weak_veto(
        rows,
        selected_rows=_selected(rows),
        selected_schedule_arm="S64",
    )

    assert result.decision == "pass"
    assert result.action == "authorize_repair80"
    assert result.denominators.phase_common.count == 16
    assert result.denominators.weak_change_boundary_audio.count == 16
    assert result.gates.pure_mean_phase_ratio.value == 1.0
    expected_pairs = [
        {
            "row_index": row.row_index,
            "cache_audio_key": row.cache_audio_key,
            "row_payload_sha256": row.prediction_row_sha256,
            "prediction_row_sha256": row.prediction_row_sha256,
            "weak_row_payload_sha256": row.weak_row_payload_sha256,
        }
        for row in rows
    ]
    assert result.row_weak_pairs_sha256 == canonical_sha256(expected_pairs)
    assert result.to_dict()["gates"]["pure_phase_coverage"]["value"] == 1.0


def test_schedule_veto_never_accepts_runner_up_swapped_or_duplicate_group_rows() -> None:
    rows = [_weak(index) for index in range(16)]
    selected = _selected(rows)
    with pytest.raises(ValueError, match="runner-up"):
        evaluate_schedule_weak_veto(
            [replace(rows[0], schedule_arm="S30"), *rows[1:]],
            selected_rows=selected,
            selected_schedule_arm="S64",
        )
    with pytest.raises(ValueError, match="pairing mismatch"):
        evaluate_schedule_weak_veto(
            [replace(rows[0], prediction_row_sha256=_sha("stale")), *rows[1:]],
            selected_rows=selected,
            selected_schedule_arm="S64",
        )
    with pytest.raises(ValueError, match="one-to-one"):
        evaluate_schedule_weak_veto(
            [rows[0], replace(rows[1], audio_group_key=rows[0].audio_group_key), *rows[2:]],
            selected_rows=selected,
            selected_schedule_arm="S64",
        )


def test_schedule_ratio_infinity_is_negative_and_comparator_conflict_is_ambiguous() -> None:
    infinite_rows = [_weak(index, current_phase=0.0, pure_phase=1.0) for index in range(16)]
    negative = evaluate_schedule_weak_veto(
        infinite_rows,
        selected_rows=_selected(infinite_rows),
        selected_schedule_arm="S64",
    )
    assert negative.gates.pure_mean_phase_ratio.state == "positive_infinity"
    assert negative.decision == "negative"

    conflict_rows = [_weak(index) for index in range(16)]
    conflict_rows[15] = _weak(15, comparator_state="conflicting")
    ambiguous = evaluate_schedule_weak_veto(
        conflict_rows,
        selected_rows=_selected(conflict_rows),
        selected_schedule_arm="S64",
    )
    assert ambiguous.denominators.comparator_conflicting_audio.count == 1
    assert ambiguous.decision == "ambiguous"


@pytest.mark.parametrize(
    ("current_f1", "pure_f1", "expected"),
    [
        (0.100001, 0.0, "negative"),
        (0.10, 0.0, "ambiguous"),
        (0.050001, 0.0, "ambiguous"),
        (0.05, 0.0, "pass"),
    ],
)
def test_schedule_boundary_delta_exact_edges_and_near_zero_absolute_f1(
    current_f1: float, pure_f1: float, expected: str
) -> None:
    rows = [
        _weak(index, current_f1=current_f1, pure_f1=pure_f1)
        for index in range(16)
    ]
    result = evaluate_schedule_weak_veto(
        rows,
        selected_rows=_selected(rows),
        selected_schedule_arm="S64",
    )
    assert result.gate_decisions["boundary_f1_delta"] == expected

    near_zero = [_weak(index, current_f1=0.001, pure_f1=0.001) for index in range(16)]
    near_zero_result = evaluate_schedule_weak_veto(
        near_zero,
        selected_rows=_selected(near_zero),
        selected_schedule_arm="S64",
    )
    assert near_zero_result.gates.current_v2_boundary_f1_mean == pytest.approx(0.001)
    assert near_zero_result.gate_decisions["boundary_f1_delta"] == "pass"


def _repair_rows(
    *,
    fallback_count: int = 0,
    duplicate_groups: bool = False,
    stable_excess: int = 0,
) -> list[Repair80MetricRow]:
    rows: list[Repair80MetricRow] = []
    fallback_indexes = set(range(30, 30 + fallback_count))
    for index in range(80):
        if index < 5:
            label = "stable"
            is_long = True
        elif index < 20:
            label = "jump_candidate"
            is_long = False
        else:
            label = "dense"
            is_long = False
        fallback = index in fallback_indexes
        weak = _weak(
            index,
            stage="repair80",
            audio_group_key="shared-group" if duplicate_groups else None,
            current_drift=10.0,
            pure_drift=8.0,
            candidate_status="tagged_fallback" if fallback else "accepted",
        )
        rows.append(
            Repair80MetricRow(
                weak=weak,
                label_stratum=label,  # type: ignore[arg-type]
                source_long_track=is_long,
                cache_valid=True,
                projection_evaluable=True,
                fallback_reason=(
                    "local_frontier_resource_cap_exceeded" if fallback else None
                ),
                audio_arm_seconds=10.0,
                overlap_p90_ms=10.0 if not fallback else None,
                candidate_section_count=(
                    1 + (stable_excess if index == 0 else 0)
                    if not fallback
                    else None
                ),
                current_v2_segment_count=1,
                seam_ms=0.0 if not fallback else None,
            )
        )
    return rows


def _repair_authority_inputs(
    root: Path,
    *,
    binding: dict[str, object],
    deps: dict[str, str],
    repair80_identity_rows: list[dict[str, Any]],
    product_branches: dict[int, tuple[bool, bool]] | None = None,
) -> tuple[
    list[Repair80MetricRow],
    list[dict[str, Any]],
    list[dict[str, object]],
    dict[str, Any],
]:
    metric_rows: list[Repair80MetricRow] = []
    prediction_rows: list[dict[str, Any]] = []
    weak_rows: list[dict[str, object]] = []
    for index in range(80):
        identity = protocol.validate_identity(repair80_identity_rows[index])
        row, candidate = _repair_prediction_row_and_candidate(
            index,
            input_sha=binding["binding_fingerprint_sha256"],
            source_sha=deps["source_closure_fingerprint_sha256"],
            identity_payload_sha256=identity["identity_payload_sha256"],
        )
        candidate_accepted, baseline_accepted = (
            product_branches.get(index, (True, True))
            if product_branches is not None
            else (True, True)
        )
        fallback_reason = "no_origin_candidate"
        row = _prediction_row_with_product_branch(
            row,
            candidate_accepted=candidate_accepted,
            baseline_accepted=baseline_accepted,
            fallback_reason=fallback_reason,
        )
        artifacts.publish_candidate_reference_row_bundle(
            root=root,
            bundle=artifacts.make_candidate_reference_row_bundle(
                stage=protocol.EXP007_REPAIR_STAGE,
                schedule_arm="S64",
                row=row,
                candidate_payload=candidate,
                input_signal_sha256=_input_sha(index),
            ),
        )
        weak = replace(
            _weak(
                index,
                stage="repair80",
                current_drift=10.0,
                pure_drift=8.0,
                candidate_status=(
                    "accepted" if candidate_accepted else "tagged_fallback"
                ),
                baseline_status=(
                    "accepted" if baseline_accepted else "unavailable"
                ),
            ),
            prediction_row_sha256=row["row_payload_sha256"],
        )
        weak_payload = _persisted_weak_row(weak, deps, stage="repair80")
        weak = replace(
            weak,
            weak_row_payload_sha256=weak_payload["weak_row_payload_sha256"],
        )
        metric_rows.append(
            Repair80MetricRow(
                weak=weak,
                label_stratum=identity["label_stratum"],  # type: ignore[arg-type]
                source_long_track=identity["source_long_track"],
                cache_valid=True,
                projection_evaluable=True,
                fallback_reason=None if candidate_accepted else fallback_reason,
                audio_arm_seconds=0.0,
                overlap_p90_ms=0.0 if candidate_accepted else None,
                candidate_section_count=1 if candidate_accepted else None,
                current_v2_segment_count=1 if baseline_accepted else None,
                seam_ms=0.0 if candidate_accepted else None,
            )
        )
        prediction_rows.append(row)
        weak_rows.append(weak_payload)
    manifest = artifacts.build_candidate_reference_manifest(
        root=root,
        stage=protocol.EXP007_REPAIR_STAGE,
        input_manifest_sha256=binding["binding_fingerprint_sha256"],
        source_closure_fingerprint_sha256=deps["source_closure_fingerprint_sha256"],
        reference_arm="S64",
    )
    return metric_rows, prediction_rows, weak_rows, manifest


def _repair_prediction_row_and_candidate(
    index: int,
    *,
    input_sha: str,
    source_sha: str,
    identity_payload_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = _candidate_payload(index)
    serialized = artifacts.serialize_candidate_payload(candidate)
    row = protocol.minimal_row_result(
        stage=protocol.EXP007_REPAIR_STAGE,
        schedule_arm="S64",
        row_index=index,
        cache_audio_key=f"audio-{index:03d}",
        audio_group_key=f"group-{index:03d}",
        identity_payload_sha256=identity_payload_sha256,
        source_closure_fingerprint_sha256=source_sha,
        run_config_fingerprint_sha256=_sha("repair-run-config"),
        selector_manifest_sha256=_sha("repair-selector"),
        input_manifest_sha256=input_sha,
        candidate_payload_schema=serialized.schema,
        candidate_payload_byte_count=serialized.byte_count,
        candidate_payload_field_set_sha256=serialized.field_set_sha256,
        candidate_payload_sha256=serialized.payload_sha256,
        candidate_fingerprint=serialized.candidate_fingerprint,
    )
    return row, candidate


def _prediction_row_with_product_branch(
    row: dict[str, Any],
    *,
    candidate_accepted: bool,
    baseline_accepted: bool,
    fallback_reason: str,
) -> dict[str, Any]:
    if candidate_accepted and baseline_accepted:
        return row
    candidate_base = row["methods"]["candidate"]
    baseline_base = row["methods"]["baseline"]
    if candidate_accepted:
        candidate_method = candidate_base
    else:
        candidate_method = protocol.make_method_result(
            method_kind="candidate",
            status="tagged_fallback",
            reason=fallback_reason,
            fallback_kind=fallback_reason,
        )
    if baseline_accepted:
        baseline_method = baseline_base
    else:
        baseline_method = protocol.make_method_result(
            method_kind="baseline",
            status="unavailable",
            reason="prediction_too_short",
        )
    if candidate_accepted:
        selected_method = protocol.make_method_result(
            method_kind="selected",
            status="accepted",
            grid=candidate_base["grid"],
            grid_summary=candidate_base["grid_summary"],
        )
    elif baseline_accepted:
        selected_method = protocol.make_method_result(
            method_kind="selected",
            status="accepted",
            reason=fallback_reason,
            fallback_kind=fallback_reason,
            grid=baseline_base["grid"],
            grid_summary=baseline_base["grid_summary"],
        )
    else:
        selected_method = protocol.make_method_result(
            method_kind="selected",
            status="unavailable",
            reason="candidate_fallback_and_baseline_unavailable",
            fallback_kind=fallback_reason,
        )
    diagnostics = _diagnostics_for_product_branch(
        row,
        candidate_accepted=candidate_accepted,
        fallback_reason=fallback_reason,
    )
    return protocol.make_row_result(
        stage=row["stage"],
        schedule_arm=row["schedule_arm"],
        row_index=row["row_index"],
        cache_audio_key=row["cache_audio_key"],
        audio_group_key=row["audio_group_key"],
        identity_payload_sha256=row["identity_payload_sha256"],
        cache_identity=row["cache_identity"],
        source_closure_fingerprint_sha256=row["source_closure_fingerprint_sha256"],
        run_config_fingerprint_sha256=row["run_config_fingerprint_sha256"],
        selector_manifest_sha256=row["selector_manifest_sha256"],
        input_manifest_sha256=row["input_manifest_sha256"],
        resume=row["resume"],
        restricted_prediction=row["restricted_prediction"],
        candidate_payload_schema=row["candidate_payload_schema"],
        candidate_payload_byte_count=row["candidate_payload_byte_count"],
        candidate_payload_field_set_sha256=row[
            "candidate_payload_field_set_sha256"
        ],
        candidate_payload_sha256=row["candidate_payload_sha256"],
        candidate_fingerprint=row["candidate_fingerprint"],
        methods={
            "candidate": candidate_method,
            "baseline": baseline_method,
            "selected": selected_method,
        },
        denominator_flags=protocol.make_denominator_flags(
            cache_valid=True,
            projection_evaluable=True,
            candidate_accepted=candidate_accepted,
            candidate_tagged_fallback=not candidate_accepted,
            baseline_accepted=baseline_accepted,
            product_grid_available=candidate_accepted or baseline_accepted,
            overlap_available=candidate_accepted,
            current_v2_phase_matched=False,
            pure_exp006_phase_matched=False,
            selected_safety_phase_matched=False,
        ),
        diagnostics_summary=diagnostics,
        runtime=row["runtime"],
        rss=row["rss"],
        hard_guards=row["hard_guards"],
    )


def _diagnostics_for_product_branch(
    row: dict[str, Any],
    *,
    candidate_accepted: bool,
    fallback_reason: str,
) -> dict[str, Any]:
    original = row["diagnostics_summary"]
    index = row["row_index"]
    if candidate_accepted:
        return original
    overlap = protocol.make_overlap_summary(
        record_count=1,
        available_record_count=0,
        unavailable_record_count=1,
        comparable_beat_count=0,
        p90_ms=None,
        p90_beats=None,
        residual_vector_sha256=None,
        records_sha256=protocol.canonical_json_sha256(
            {"row_index": index, "records": "synthetic-fallback"}
        ),
    )
    return protocol.make_bounded_diagnostics_summary(
        schedule_arm=row["schedule_arm"],
        result_reason=fallback_reason,
        selected_section_count=None,
        block_count=original["block_count"],
        candidate_fingerprint=row["candidate_fingerprint"],
        grid_fingerprint=protocol.canonical_json_sha256(
            {"row_index": index, "grid": "candidate-unavailable"}
        ),
        replay_fingerprint=original["replay_fingerprint"],
        transition_cache_size=original["transition_cache_size"],
        actual_scored_edge_count=original["actual_scored_edge_count"],
        selected_terminal_objective=None,
        runner_up_terminal_objective=None,
        selected_runner_up_margin=None,
        block_resource_records_sha256=original["block_resource_records_sha256"],
        class_coverage_records_sha256=original["class_coverage_records_sha256"],
        overlap=overlap,
    )


def _prediction_row_with_identity_payload(
    row: dict[str, Any],
    identity_payload_sha256: str,
) -> dict[str, Any]:
    return protocol.make_row_result(
        stage=row["stage"],
        schedule_arm=row["schedule_arm"],
        row_index=row["row_index"],
        cache_audio_key=row["cache_audio_key"],
        audio_group_key=row["audio_group_key"],
        identity_payload_sha256=identity_payload_sha256,
        cache_identity=row["cache_identity"],
        source_closure_fingerprint_sha256=row["source_closure_fingerprint_sha256"],
        run_config_fingerprint_sha256=row["run_config_fingerprint_sha256"],
        selector_manifest_sha256=row["selector_manifest_sha256"],
        input_manifest_sha256=row["input_manifest_sha256"],
        resume=row["resume"],
        restricted_prediction=row["restricted_prediction"],
        candidate_payload_schema=row["candidate_payload_schema"],
        candidate_payload_byte_count=row["candidate_payload_byte_count"],
        candidate_payload_field_set_sha256=row[
            "candidate_payload_field_set_sha256"
        ],
        candidate_payload_sha256=row["candidate_payload_sha256"],
        candidate_fingerprint=row["candidate_fingerprint"],
        methods=row["methods"],
        denominator_flags=row["denominator_flags"],
        diagnostics_summary=row["diagnostics_summary"],
        runtime=row["runtime"],
        rss=row["rss"],
        hard_guards=row["hard_guards"],
    )


def test_repair80_uses_identity_denominators_allows_duplicate_groups_and_overlapping_long() -> None:
    result = evaluate_repair80(
        _repair_rows(duplicate_groups=True),
        selected_schedule_arm="S64",
        worker_lifetime_rss_bytes=[100, 200, 300, 400],
    )

    assert result.decision == "pass"
    assert result.action == "write_result_and_next_no_data_card"
    assert result.denominators.stage_audio_count == 80
    assert result.denominators.stage_audio.count == 80
    assert result.denominators.stable_pure_paired.count == 5
    assert result.denominators.long_pure_paired.count == 5
    assert result.denominators.jump_pure_paired.count == 15
    assert result.gates.jump_alias_drift_mean_ratio.value == pytest.approx(0.8)
    assert set(result.gates.to_dict()) == {
        "candidate_fallback_rate",
        "selected_product_fallback_rate",
        "no_origin_or_path_rate",
        "runtime_seconds",
        "worker_rss_bytes",
        "overlap_ms",
        "stable_section_excess",
        "pure_mean_phase_ratio",
        "pure_p90_phase_ratio",
        "pure_phase_coverage",
        "current_v2_phase_mean_ms",
        "pure_exp006_phase_mean_ms",
        "current_v2_phase_p90_ms",
        "pure_exp006_phase_p90_ms",
        "stable_phase_mean_ratio",
        "stable_phase_p90_ratio",
        "jump_phase_mean_ratio",
        "current_v2_jump_alias_drift_mean_ms",
        "pure_exp006_jump_alias_drift_mean_ms",
        "jump_alias_drift_mean_ratio",
        "current_v2_long_alias_drift_mean_ms",
        "pure_exp006_long_alias_drift_mean_ms",
        "current_v2_long_alias_drift_p90_ms",
        "pure_exp006_long_alias_drift_p90_ms",
        "long_alias_drift_mean_ratio",
        "long_alias_drift_p90_ratio",
        "current_v2_boundary_f1_mean",
        "pure_exp006_boundary_f1_mean",
        "selected_boundary_f1_mean",
        "pure_minus_v2_boundary_f1_delta",
        "every_row_under_180_seconds",
        "seam_zero",
        "section_cap_valid",
        "replay_schema_source_cache_integrity",
    }
    assert result.gates.current_v2_phase_mean_ms == 10.0
    assert result.gates.pure_exp006_phase_p90_ms == 10.0
    assert result.gates.current_v2_jump_alias_drift_mean_ms == 10.0
    assert result.gates.pure_exp006_jump_alias_drift_mean_ms == 8.0
    assert result.gates.current_v2_long_alias_drift_p90_ms == 10.0
    assert result.gates.pure_exp006_long_alias_drift_p90_ms == 8.0


def test_repair80_rejects_duplicate_cache_key_but_never_deduplicates_audio_group() -> None:
    rows = _repair_rows(duplicate_groups=True)
    rows[1] = replace(rows[1], weak=replace(rows[1].weak, cache_audio_key=rows[0].cache_audio_key))
    with pytest.raises(ValueError, match="cache_audio_key"):
        evaluate_repair80(
            rows,
            selected_schedule_arm="S64",
            worker_lifetime_rss_bytes=[1, 1, 1, 1],
        )


@pytest.mark.parametrize(
    ("fallback_count", "expected"),
    [(4, "pass"), (8, "ambiguous"), (9, "negative")],
)
def test_repair80_candidate_fallback_rate_uses_candidate_status_not_selected_product(
    fallback_count: int, expected: str
) -> None:
    result = evaluate_repair80(
        _repair_rows(fallback_count=fallback_count),
        selected_schedule_arm="S64",
        worker_lifetime_rss_bytes=[1, 1, 1, 1],
    )
    assert result.denominators.candidate_fallback_audio.count == fallback_count
    assert result.denominators.selected_product_fallback_audio.count == fallback_count
    assert result.gate_decisions["candidate_fallback_rate"] == expected
    assert result.decision == expected


def test_repair80_authoritative_four_product_branches_allow_baseline_unavailable(
    tmp_path: Path,
) -> None:
    product_branches = {
        index: (
            index < 40,
            index < 20 or 40 <= index < 60,
        )
        for index in range(80)
    }
    (
        summary,
        evaluation,
        binding,
        weak_outcome,
        manifest,
        prediction_rows,
        weak_rows,
        repair_rows,
        repair_source_context,
    ) = _repair_summary_bundle(tmp_path, product_branches=product_branches)

    validate_repair80_summary_authoritatively(
        summary,
        repair_metric_rows=repair_rows,
        repair80_input_binding=binding,
        **_repair_source_kwargs(repair_source_context),
        schedule_weak_veto_outcome=weak_outcome,
        candidate_reference_manifest=manifest,
        artifact_root=tmp_path,
        prediction_rows=prediction_rows,
        weak_rows=weak_rows,
        aggregate_wall_seconds=10.0,
        worker_lifetime_rss_bytes=[1, 1, 1, 1],
    )

    assert evaluation.denominators.candidate_accepted_audio.count == 40
    assert evaluation.denominators.candidate_fallback_audio.count == 40
    assert evaluation.denominators.baseline_accepted_audio.count == 40
    assert evaluation.denominators.product_grid_available_audio.count == 60
    assert evaluation.denominators.selected_product_fallback_audio.count == 20
    assert evaluation.denominators.current_v2_phase_matched.count == 40
    assert evaluation.denominators.pure_exp006_phase_matched.count == 20
    assert evaluation.denominators.selected_safety_phase_matched.count == 40
    assert evaluation.denominators.overlap_available_audio.count == 40
    assert summary["decision"] == "negative"

    accepted_without_baseline = repair_rows[20]
    assert accepted_without_baseline.candidate_section_count == 1
    assert accepted_without_baseline.current_v2_segment_count is None
    assert accepted_without_baseline.seam_ms == 0.0
    assert accepted_without_baseline.weak.selected_status == "accepted"
    assert accepted_without_baseline.weak.product_grid_available is True
    assert accepted_without_baseline.weak.current_v2_phase_matched is False
    assert accepted_without_baseline.weak.selected_safety_phase_matched is False

    fallback_with_baseline = repair_rows[40]
    assert fallback_with_baseline.candidate_section_count is None
    assert fallback_with_baseline.current_v2_segment_count == 1
    assert fallback_with_baseline.seam_ms is None
    assert fallback_with_baseline.weak.selected_status == "accepted"
    assert fallback_with_baseline.weak.product_grid_available is True
    assert fallback_with_baseline.weak.selected_safety_phase_matched is True

    fallback_without_baseline = repair_rows[60]
    assert fallback_without_baseline.candidate_section_count is None
    assert fallback_without_baseline.current_v2_segment_count is None
    assert fallback_without_baseline.seam_ms is None
    assert fallback_without_baseline.weak.selected_status == "unavailable"
    assert fallback_without_baseline.weak.product_grid_available is False
    assert fallback_without_baseline.weak.selected_safety_phase_matched is False

    assert prediction_rows[20]["methods"]["baseline"]["grid_summary"] is None
    assert prediction_rows[20]["methods"]["selected"]["grid_summary"]["grid_kind"] == "timing_v3"
    assert prediction_rows[40]["methods"]["candidate"]["grid_summary"] is None
    assert prediction_rows[40]["methods"]["selected"]["grid_summary"]["grid_kind"] == "current_v2"
    assert prediction_rows[60]["methods"]["selected"]["grid_summary"] is None


def test_repair80_stable_section_excess_over_one_is_negative() -> None:
    result = evaluate_repair80(
        _repair_rows(stable_excess=2),
        selected_schedule_arm="S64",
        worker_lifetime_rss_bytes=[1, 1, 1, 1],
    )
    assert result.gates.stable_section_excess.maximum == 2
    assert result.gate_decisions["stable_section_excess"] == "negative"
    assert result.action == "stop_negative"


def test_weak_resume_prefix_failure_record_and_outcome_hashes_fail_closed() -> None:
    rows = [_weak(index) for index in range(16)]
    selected = _selected(rows)
    completed = tuple(row.ref() for row in rows[:3])
    plan = validate_weak_resume_prefix(selected, completed)
    assert len(plan.completed_prefix) == 3
    assert plan.pending[0].row_index == 3

    with pytest.raises(ValueError, match="stale, swapped, gapped"):
        validate_weak_resume_prefix(selected, [rows[1].ref()])

    deps = _schedule_deps()
    failure = make_schedule_weak_failure_record(
        schedule_arm="S64",
        four_arm_stage_summary_sha256=deps["four_arm_stage_summary_sha256"],
        candidate_global_manifest_sha256=deps["candidate_global_manifest_sha256"],
        source_selection_sha256=deps["source_selection_sha256"],
        source_closure_fingerprint_sha256=deps["source_closure_fingerprint_sha256"],
        selected_rows=selected,
        completed=completed,
        failure_kind="metrics_failure",
        failure_stage="metrics",
        causing_row_index=3,
    )
    assert failure["completed_prefix_count"] == 3
    assert failure["pending_count"] == 13
    assert failure["causing_cache_audio_key"] == rows[3].cache_audio_key
    validate_schedule_weak_failure_record(failure, selected_rows=selected)

    hard = make_schedule_weak_hard_failure_outcome(failure)
    assert validate_schedule_weak_outcome(hard) == "hard_failure"
    assert schedule_weak_resume_action(hard, selected_rows=selected, completed=()) == "reuse_hard_failure"

    evaluation = evaluate_schedule_weak_veto(
        rows,
        selected_rows=selected,
        selected_schedule_arm="S64",
    )
    summary = make_schedule_weak_veto_summary(evaluation, schedule_arm="S64", **deps)
    success = make_schedule_weak_success_outcome(summary)
    assert validate_schedule_weak_outcome(success) == "success"
    assert schedule_weak_resume_action(success, selected_rows=selected, completed=()) == "reuse_success"
    assert schedule_weak_resume_action(None, selected_rows=selected, completed=completed) == "continue_prefix"

    tampered = dict(success)
    tampered["summary"] = {"decision": "negative", "action": "stop_negative"}
    with pytest.raises(ValueError, match="ScheduleWeakVetoSummary|fields"):
        validate_schedule_weak_outcome(tampered)

    with pytest.raises(ValueError, match="kind/stage"):
        make_schedule_weak_failure_record(
            schedule_arm="S64",
            four_arm_stage_summary_sha256=deps["four_arm_stage_summary_sha256"],
            candidate_global_manifest_sha256=deps["candidate_global_manifest_sha256"],
            source_selection_sha256=deps["source_selection_sha256"],
            source_closure_fingerprint_sha256=deps["source_closure_fingerprint_sha256"],
            selected_rows=selected,
            completed=completed,
            failure_kind="metrics_failure",
            failure_stage="schema",
        )


def test_weak_resume_success_outcome_rejects_stale_selected_refs() -> None:
    rows = [_weak(index) for index in range(16)]
    selected = _selected(rows)
    deps = _schedule_deps()
    evaluation = evaluate_schedule_weak_veto(
        rows,
        selected_rows=selected,
        selected_schedule_arm="S64",
    )
    summary = make_schedule_weak_veto_summary(evaluation, schedule_arm="S64", **deps)
    success = make_schedule_weak_success_outcome(summary)

    changed = list(selected)
    changed[0] = PredictionRowRef(
        0,
        changed[0].cache_audio_key,
        _sha("changed-selected-row"),
        changed[0].schedule_arm,
    )
    with pytest.raises(ValueError, match="stale|mismatched"):
        schedule_weak_resume_action(success, selected_rows=changed, completed=())

    changed_arm = list(selected)
    changed_arm[0] = PredictionRowRef(
        0,
        changed_arm[0].cache_audio_key,
        changed_arm[0].prediction_row_sha256,
        "S30",
    )
    with pytest.raises(ValueError, match="schedule arm mismatch"):
        schedule_weak_resume_action(success, selected_rows=changed_arm, completed=())

    swapped = [selected[1], selected[0], *selected[2:]]
    with pytest.raises(ValueError, match="contiguous|ordered"):
        schedule_weak_resume_action(success, selected_rows=swapped, completed=())


def test_weak_resume_hard_failure_outcome_rejects_stale_selected_refs() -> None:
    rows = [_weak(index) for index in range(16)]
    selected = _selected(rows)
    completed = tuple(row.ref() for row in rows[:3])
    deps = _schedule_deps()
    failure = make_schedule_weak_failure_record(
        schedule_arm="S64",
        four_arm_stage_summary_sha256=deps["four_arm_stage_summary_sha256"],
        candidate_global_manifest_sha256=deps["candidate_global_manifest_sha256"],
        source_selection_sha256=deps["source_selection_sha256"],
        source_closure_fingerprint_sha256=deps["source_closure_fingerprint_sha256"],
        selected_rows=selected,
        completed=completed,
        failure_kind="metrics_failure",
        failure_stage="metrics",
        causing_row_index=3,
    )
    hard = make_schedule_weak_hard_failure_outcome(failure)

    changed_pending = list(selected)
    changed_pending[3] = PredictionRowRef(
        3,
        "changed-audio-003",
        changed_pending[3].prediction_row_sha256,
        changed_pending[3].schedule_arm,
    )
    with pytest.raises(ValueError, match="pending suffix|mismatch"):
        schedule_weak_resume_action(hard, selected_rows=changed_pending, completed=())

    changed_arm = list(selected)
    changed_arm[3] = PredictionRowRef(
        3,
        changed_arm[3].cache_audio_key,
        changed_arm[3].prediction_row_sha256,
        "S30",
    )
    with pytest.raises(ValueError, match="schedule arm mismatch"):
        schedule_weak_resume_action(hard, selected_rows=changed_arm, completed=())

    swapped = [selected[1], selected[0], *selected[2:]]
    with pytest.raises(ValueError, match="contiguous|ordered"):
        schedule_weak_resume_action(hard, selected_rows=swapped, completed=())


def test_schedule_success_outcome_rejects_arbitrary_summary_dict() -> None:
    with pytest.raises(ValueError, match="ScheduleWeakVetoSummary|fields"):
        make_schedule_weak_success_outcome(
            {"decision": "pass", "action": "authorize_repair80"}
        )


def test_boundary_summary_rejects_tp_fp_fn_f1_mismatch() -> None:
    with pytest.raises(ValueError, match="F1"):
        BoundarySummary(
            eligible=True,
            valid_difficulty_count=1,
            tp=1,
            fp=1,
            fn=0,
            f1=ratio_value(1.0, 1.0),
            matched_error_ms=stats_value([0.0]),
            weak_consensus_supported_count=1,
        )


def test_weak_row_rejects_null_matrix_and_boundary_without_grid() -> None:
    row = _weak(0)
    deps = _schedule_deps()
    sample = stats_value([10.0])

    with pytest.raises(ValueError, match="null matrix"):
        make_weak_row(
            stage="schedule16",
            schedule_arm="S64",
            row_index=row.row_index,
            cache_audio_key=row.cache_audio_key,
            audio_group_key=row.audio_group_key,
            prediction_row_sha256=row.prediction_row_sha256,
            four_arm_stage_summary_sha256=deps["four_arm_stage_summary_sha256"],
            candidate_global_manifest_sha256=deps["candidate_global_manifest_sha256"],
            source_selection_sha256=deps["source_selection_sha256"],
            comparator_availability=_available_comparator(),
            current_v2_phase_matched=False,
            pure_exp006_phase_matched=False,
            selected_safety_phase_matched=False,
            phase_metrics_summary=PhaseSummary(sample, None, None),
            drift_metrics_summary=DriftSummary(None, None, None),
            current_v2_boundary_summary=BoundarySummary.ineligible(),
            pure_exp006_boundary_summary=BoundarySummary.ineligible(),
            selected_boundary_summary=BoundarySummary.ineligible(),
            object_grid_summary=ObjectGridSummary.ineligible(),
        )

    with pytest.raises(ValueError, match="boundary requires comparator and grid"):
        make_weak_row(
            stage="schedule16",
            schedule_arm="S64",
            row_index=row.row_index,
            cache_audio_key=row.cache_audio_key,
            audio_group_key=row.audio_group_key,
            prediction_row_sha256=row.prediction_row_sha256,
            four_arm_stage_summary_sha256=deps["four_arm_stage_summary_sha256"],
            candidate_global_manifest_sha256=deps["candidate_global_manifest_sha256"],
            source_selection_sha256=deps["source_selection_sha256"],
            comparator_availability=_available_comparator(),
            current_v2_phase_matched=False,
            pure_exp006_phase_matched=False,
            selected_safety_phase_matched=False,
            phase_metrics_summary=PhaseSummary(None, None, None),
            drift_metrics_summary=DriftSummary(None, None, None),
            current_v2_boundary_summary=_boundary_summary(),
            pure_exp006_boundary_summary=BoundarySummary.ineligible(),
            selected_boundary_summary=BoundarySummary.ineligible(),
            object_grid_summary=ObjectGridSummary.ineligible(),
        )


def test_schedule_summary_rejects_swapped_and_cross_stage_weak_rows() -> None:
    deps = _schedule_deps()
    base_rows = [_weak(index) for index in range(16)]
    weak_rows = [_persisted_weak_row(row, deps) for row in base_rows]
    rows = [
        replace(
            row,
            weak_row_payload_sha256=weak_row["weak_row_payload_sha256"],
        )
        for row, weak_row in zip(base_rows, weak_rows, strict=True)
    ]
    evaluation = evaluate_schedule_weak_veto(
        rows,
        selected_rows=_selected(rows),
        selected_schedule_arm="S64",
    )
    summary = make_schedule_weak_veto_summary(evaluation, schedule_arm="S64", **deps)
    validate_schedule_weak_veto_summary(
        summary,
        evaluation=evaluation,
        weak_rows=weak_rows,
    )

    swapped = [weak_rows[1], weak_rows[0], *weak_rows[2:]]
    with pytest.raises(ValueError, match="mismatch"):
        validate_schedule_weak_veto_summary(summary, weak_rows=swapped)

    cross_stage = [
        _persisted_weak_row(rows[0], deps, stage="repair80"),
        *weak_rows[1:],
    ]
    with pytest.raises(ValueError, match="cross-stage"):
        validate_schedule_weak_veto_summary(summary, weak_rows=cross_stage)


def test_schedule_summary_rejects_decision_action_cross_pair() -> None:
    _, evaluation, summary, _ = _schedule_summary_bundle()
    validate_schedule_weak_veto_summary(summary, evaluation=evaluation)

    tampered = dict(summary, action="stop_negative")
    tampered["summary_fingerprint_sha256"] = protocol.payload_hash(
        tampered,
        "summary_fingerprint_sha256",
    )
    with pytest.raises(ValueError, match="decision/action"):
        validate_schedule_weak_veto_summary(tampered)


def test_repair80_summary_binds_dependencies_and_decision_action_pairs(
    tmp_path: Path,
) -> None:
    (
        summary,
        evaluation,
        binding,
        weak_outcome,
        manifest,
        prediction_rows,
        weak_rows,
        repair_rows,
        repair_source_context,
    ) = _repair_summary_bundle(tmp_path)
    validate_repair80_summary(
        summary,
        evaluation=evaluation,
        repair80_input_binding=binding,
        schedule_weak_veto_outcome=weak_outcome,
        candidate_reference_manifest=manifest,
        artifact_root=tmp_path,
        weak_rows=weak_rows,
    )
    validate_repair80_summary_authoritatively(
        summary,
        repair_metric_rows=repair_rows,
        repair80_input_binding=binding,
        **_repair_source_kwargs(repair_source_context),
        schedule_weak_veto_outcome=weak_outcome,
        candidate_reference_manifest=manifest,
        artifact_root=tmp_path,
        prediction_rows=prediction_rows,
        weak_rows=weak_rows,
        aggregate_wall_seconds=10.0,
        worker_lifetime_rss_bytes=[1, 1, 1, 1],
    )
    assert len(summary["gates"]) == 34
    assert set(summary["gates"]) == protocol.REPAIR80_GATES_FIELDS
    assert summary["gates"] == protocol.validate_repair80_gates(
        evaluation.gates.to_dict()
    )
    assert summary["gates"]["selected_product_fallback_rate"] == {
        "numerator": 0.0,
        "denominator": 80.0,
        "value": 0.0,
    }
    assert summary["gates"]["stable_phase_mean_ratio"]["state"] == "finite"
    assert summary["gates"]["jump_alias_drift_mean_ratio"]["state"] == "finite"
    assert "hard_guards_passed" not in summary["gates"]
    assert "stable_mean_phase_ratio" not in summary["gates"]

    mismatched_dependency = dict(
        summary,
        source_selection_sha256=_sha("different-selection"),
    )
    mismatched_dependency["summary_fingerprint_sha256"] = protocol.payload_hash(
        mismatched_dependency,
        "summary_fingerprint_sha256",
    )
    with pytest.raises(ValueError, match="source_selection_sha256 binding mismatch"):
        validate_repair80_summary(
            mismatched_dependency,
            evaluation=evaluation,
            repair80_input_binding=binding,
            schedule_weak_veto_outcome=weak_outcome,
            candidate_reference_manifest=manifest,
            artifact_root=tmp_path,
        )

    bad_action = dict(summary, action="stop_negative")
    bad_action["summary_fingerprint_sha256"] = protocol.payload_hash(
        bad_action,
        "summary_fingerprint_sha256",
    )
    with pytest.raises(ValueError, match="decision/action"):
        validate_repair80_summary(
            bad_action,
            candidate_reference_manifest=manifest,
            artifact_root=tmp_path,
        )


def test_repair80_summary_requires_manifest_object_and_reopens_bundles(
    tmp_path: Path,
) -> None:
    (
        summary,
        evaluation,
        binding,
        weak_outcome,
        manifest,
        prediction_rows,
        weak_rows,
        repair_rows,
        repair_source_context,
    ) = _repair_summary_bundle(tmp_path)

    other_root = tmp_path / "missing-bundles"
    other_root.mkdir()
    with pytest.raises(Exception, match="bundle|path|exist|No such file"):
        validate_repair80_summary(
            summary,
            evaluation=evaluation,
            repair80_input_binding=binding,
            schedule_weak_veto_outcome=weak_outcome,
            candidate_reference_manifest=manifest,
            artifact_root=other_root,
        )

    swapped = copy.deepcopy(manifest)
    swapped["entries"][0], swapped["entries"][1] = (
        swapped["entries"][1],
        swapped["entries"][0],
    )
    swapped["ordered_entries_sha256"] = protocol.canonical_json_sha256(
        swapped["entries"]
    )
    swapped["manifest_fingerprint_sha256"] = protocol.payload_hash(
        swapped,
        "manifest_fingerprint_sha256",
    )
    with pytest.raises(Exception, match="row order|manifest"):
        validate_repair80_summary(
            summary,
            evaluation=evaluation,
            repair80_input_binding=binding,
            schedule_weak_veto_outcome=weak_outcome,
            candidate_reference_manifest=swapped,
            artifact_root=tmp_path,
        )

    fake_sha = dict(
        summary,
        candidate_reference_manifest_sha256=_sha("caller-forged-reference-sha"),
    )
    fake_sha["summary_fingerprint_sha256"] = protocol.payload_hash(
        fake_sha,
        "summary_fingerprint_sha256",
    )
    with pytest.raises(ValueError, match="manifest SHA mismatch"):
        validate_repair80_summary(
            fake_sha,
            evaluation=evaluation,
            repair80_input_binding=binding,
            schedule_weak_veto_outcome=weak_outcome,
            candidate_reference_manifest=manifest,
            artifact_root=tmp_path,
        )

    forged = dict(
        summary,
        candidate_reference_manifest_sha256=protocol.object_complete_sha256(manifest),
    )
    forged["summary_fingerprint_sha256"] = protocol.payload_hash(
        forged,
        "summary_fingerprint_sha256",
    )
    validate_repair80_summary_authoritatively(
        forged,
        repair_metric_rows=repair_rows,
        repair80_input_binding=binding,
        **_repair_source_kwargs(repair_source_context),
        schedule_weak_veto_outcome=weak_outcome,
        candidate_reference_manifest=manifest,
        artifact_root=tmp_path,
        prediction_rows=prediction_rows,
        weak_rows=weak_rows,
        aggregate_wall_seconds=10.0,
        worker_lifetime_rss_bytes=[1, 1, 1, 1],
    )


def test_repair80_authoritative_summary_rejects_shape_valid_metric_tamper(
    tmp_path: Path,
) -> None:
    (
        summary,
        evaluation,
        binding,
        weak_outcome,
        manifest,
        prediction_rows,
        weak_rows,
        repair_rows,
        repair_source_context,
    ) = _repair_summary_bundle(tmp_path)
    tampered = copy.deepcopy(summary)
    tampered["gates"]["runtime_seconds"] = dict(
        tampered["gates"]["runtime_seconds"],
        p90=20.0,
        maximum=20.0,
    )
    tampered["summary_fingerprint_sha256"] = protocol.payload_hash(
        tampered,
        "summary_fingerprint_sha256",
    )

    with pytest.raises(ValueError, match="gates|authoritative recomputation"):
        validate_repair80_summary_authoritatively(
            tampered,
            repair_metric_rows=repair_rows,
            repair80_input_binding=binding,
            **_repair_source_kwargs(repair_source_context),
            schedule_weak_veto_outcome=weak_outcome,
            candidate_reference_manifest=manifest,
            artifact_root=tmp_path,
            prediction_rows=prediction_rows,
            weak_rows=weak_rows,
            aggregate_wall_seconds=10.0,
            worker_lifetime_rss_bytes=[1, 1, 1, 1],
        )


def test_repair80_authoritative_summary_rejects_metric_row_cross_bind_tamper(
    tmp_path: Path,
) -> None:
    (
        summary,
        _evaluation,
        binding,
        weak_outcome,
        manifest,
        prediction_rows,
        weak_rows,
        repair_rows,
        repair_source_context,
    ) = _repair_summary_bundle(tmp_path)
    changed_metrics = [
        replace(repair_rows[0], audio_arm_seconds=1.0),
        *repair_rows[1:],
    ]

    with pytest.raises(ValueError, match="audio runtime mismatch"):
        validate_repair80_summary_authoritatively(
            summary,
            repair_metric_rows=changed_metrics,
            repair80_input_binding=binding,
            **_repair_source_kwargs(repair_source_context),
            schedule_weak_veto_outcome=weak_outcome,
            candidate_reference_manifest=manifest,
            artifact_root=tmp_path,
            prediction_rows=prediction_rows,
            weak_rows=weak_rows,
            aggregate_wall_seconds=10.0,
            worker_lifetime_rss_bytes=[1, 1, 1, 1],
        )


def test_repair80_authoritative_summary_binds_identity_sources(
    tmp_path: Path,
) -> None:
    (
        summary,
        _evaluation,
        binding,
        weak_outcome,
        manifest,
        prediction_rows,
        weak_rows,
        repair_rows,
        repair_source_context,
    ) = _repair_summary_bundle(tmp_path)

    class_tamper = [
        replace(repair_rows[0], label_stratum="dense"),
        *repair_rows[1:],
    ]
    with pytest.raises(ValueError, match="metric/identity label_stratum mismatch"):
        validate_repair80_summary_authoritatively(
            summary,
            repair_metric_rows=class_tamper,
            repair80_input_binding=binding,
            **_repair_source_kwargs(repair_source_context),
            schedule_weak_veto_outcome=weak_outcome,
            candidate_reference_manifest=manifest,
            artifact_root=tmp_path,
            prediction_rows=prediction_rows,
            weak_rows=weak_rows,
            aggregate_wall_seconds=10.0,
            worker_lifetime_rss_bytes=[1, 1, 1, 1],
        )

    long_tamper = [
        *repair_rows[:5],
        replace(repair_rows[5], source_long_track=True),
        *repair_rows[6:],
    ]
    with pytest.raises(ValueError, match="metric/identity source_long_track mismatch"):
        validate_repair80_summary_authoritatively(
            summary,
            repair_metric_rows=long_tamper,
            repair80_input_binding=binding,
            **_repair_source_kwargs(repair_source_context),
            schedule_weak_veto_outcome=weak_outcome,
            candidate_reference_manifest=manifest,
            artifact_root=tmp_path,
            prediction_rows=prediction_rows,
            weak_rows=weak_rows,
            aggregate_wall_seconds=10.0,
            worker_lifetime_rss_bytes=[1, 1, 1, 1],
        )

    swapped_identity_context = dict(repair_source_context)
    swapped_identity_rows = list(repair_source_context["repair80_identity_rows"])
    swapped_identity_rows[0], swapped_identity_rows[1] = (
        swapped_identity_rows[1],
        swapped_identity_rows[0],
    )
    swapped_identity_context["repair80_identity_rows"] = swapped_identity_rows
    with pytest.raises(ValueError, match="do not match source artifact rows"):
        validate_repair80_summary_authoritatively(
            summary,
            repair_metric_rows=repair_rows,
            repair80_input_binding=binding,
            **_repair_source_kwargs(swapped_identity_context),
            schedule_weak_veto_outcome=weak_outcome,
            candidate_reference_manifest=manifest,
            artifact_root=tmp_path,
            prediction_rows=prediction_rows,
            weak_rows=weak_rows,
            aggregate_wall_seconds=10.0,
            worker_lifetime_rss_bytes=[1, 1, 1, 1],
        )

    bad_artifact_context = dict(repair_source_context)
    bad_identity_artifact = {
        "schema": protocol.SOURCE_REPAIR80_IDENTITY_ARTIFACT_SCHEMA,
        "rows": copy.deepcopy(repair_source_context["repair80_identity_rows"]),
    }
    bad_identity_artifact["rows"][0]["source_row_index"] += 1
    bad_artifact_context["repair80_identity_source_artifact"] = (
        protocol.canonical_json_bytes(bad_identity_artifact)
    )
    with pytest.raises(ValueError, match="source_repair80_identity.sha256 mismatch"):
        validate_repair80_summary_authoritatively(
            summary,
            repair_metric_rows=repair_rows,
            repair80_input_binding=binding,
            **_repair_source_kwargs(bad_artifact_context),
            schedule_weak_veto_outcome=weak_outcome,
            candidate_reference_manifest=manifest,
            artifact_root=tmp_path,
            prediction_rows=prediction_rows,
            weak_rows=weak_rows,
            aggregate_wall_seconds=10.0,
            worker_lifetime_rss_bytes=[1, 1, 1, 1],
        )

    changed_predictions = list(prediction_rows)
    changed_predictions[0] = _prediction_row_with_identity_payload(
        prediction_rows[0],
        _sha("forged-repair80-identity-payload"),
    )
    with pytest.raises(
        ValueError,
        match="identity/prediction identity_payload_sha256 mismatch",
    ):
        validate_repair80_summary_authoritatively(
            summary,
            repair_metric_rows=repair_rows,
            repair80_input_binding=binding,
            **_repair_source_kwargs(repair_source_context),
            schedule_weak_veto_outcome=weak_outcome,
            candidate_reference_manifest=manifest,
            artifact_root=tmp_path,
            prediction_rows=changed_predictions,
            weak_rows=weak_rows,
            aggregate_wall_seconds=10.0,
            worker_lifetime_rss_bytes=[1, 1, 1, 1],
        )


@pytest.mark.parametrize(
    ("metric_boundary_attr", "weak_boundary_field"),
    [
        ("current_v2_boundary", "current_v2_boundary_summary"),
        ("pure_exp006_boundary", "pure_exp006_boundary_summary"),
        ("selected_boundary", "selected_boundary_summary"),
    ],
)
def test_repair80_authoritative_summary_rejects_any_boundary_summary_mismatch(
    tmp_path: Path,
    metric_boundary_attr: str,
    weak_boundary_field: str,
) -> None:
    (
        summary,
        _evaluation,
        binding,
        weak_outcome,
        manifest,
        prediction_rows,
        weak_rows,
        repair_rows,
        repair_source_context,
    ) = _repair_summary_bundle(tmp_path)
    changed_weak_rows = copy.deepcopy(weak_rows)
    changed_weak_rows[0][weak_boundary_field] = _boundary_summary(
        tp=0,
        fp=1,
        fn=0,
    ).to_dict()
    _rehash_weak_row_payload(changed_weak_rows[0])
    changed_metric_weak = replace(
        repair_rows[0].weak,
        weak_row_payload_sha256=changed_weak_rows[0]["weak_row_payload_sha256"],
    )
    changed_metrics = [
        replace(repair_rows[0], weak=changed_metric_weak),
        *repair_rows[1:],
    ]

    with pytest.raises(ValueError, match=f"{metric_boundary_attr} boundary"):
        make_repair80_summary_from_rows(
            schedule_arm="S64",
            repair_metric_rows=changed_metrics,
            repair80_input_binding=binding,
            **_repair_source_kwargs(repair_source_context),
            schedule_weak_veto_outcome=weak_outcome,
            run_config_fingerprint_sha256=summary["run_config_fingerprint_sha256"],
            candidate_reference_manifest=manifest,
            artifact_root=tmp_path,
            prediction_rows=prediction_rows,
            weak_rows=changed_weak_rows,
            aggregate_wall_seconds=10.0,
            worker_lifetime_rss_bytes=[1, 1, 1, 1],
        )


def test_repair80_authoritative_summary_rejects_boundary_valid_weak_count_mismatch(
    tmp_path: Path,
) -> None:
    (
        summary,
        _evaluation,
        binding,
        weak_outcome,
        manifest,
        prediction_rows,
        weak_rows,
        repair_rows,
        repair_source_context,
    ) = _repair_summary_bundle(tmp_path)
    changed_weak_rows = copy.deepcopy(weak_rows)
    changed_weak_rows[0]["current_v2_boundary_summary"] = _boundary_summary(
        weak_consensus_supported_count=0,
    ).to_dict()
    _rehash_weak_row_payload(changed_weak_rows[0])
    changed_metric_weak = replace(
        repair_rows[0].weak,
        weak_row_payload_sha256=changed_weak_rows[0]["weak_row_payload_sha256"],
    )
    changed_metrics = [
        replace(repair_rows[0], weak=changed_metric_weak),
        *repair_rows[1:],
    ]

    with pytest.raises(ValueError, match="current_v2_boundary boundary"):
        make_repair80_summary_from_rows(
            schedule_arm="S64",
            repair_metric_rows=changed_metrics,
            repair80_input_binding=binding,
            **_repair_source_kwargs(repair_source_context),
            schedule_weak_veto_outcome=weak_outcome,
            run_config_fingerprint_sha256=summary["run_config_fingerprint_sha256"],
            candidate_reference_manifest=manifest,
            artifact_root=tmp_path,
            prediction_rows=prediction_rows,
            weak_rows=changed_weak_rows,
            aggregate_wall_seconds=10.0,
            worker_lifetime_rss_bytes=[1, 1, 1, 1],
        )


def test_repair80_authoritative_summary_rejects_row_and_weak_mismatch(
    tmp_path: Path,
) -> None:
    (
        summary,
        evaluation,
        binding,
        weak_outcome,
        manifest,
        prediction_rows,
        weak_rows,
        repair_rows,
        repair_source_context,
    ) = _repair_summary_bundle(tmp_path)

    changed_rows = [dict(row) for row in prediction_rows]
    changed_rows[0] = dict(
        changed_rows[0],
        row_payload_sha256=_sha("forged-row-payload"),
    )
    with pytest.raises(ValueError, match="reference bundle|RowResult|hash"):
        validate_repair80_summary_authoritatively(
            summary,
            repair_metric_rows=repair_rows,
            repair80_input_binding=binding,
            **_repair_source_kwargs(repair_source_context),
            schedule_weak_veto_outcome=weak_outcome,
            candidate_reference_manifest=manifest,
            artifact_root=tmp_path,
            prediction_rows=changed_rows,
            weak_rows=weak_rows,
            aggregate_wall_seconds=10.0,
            worker_lifetime_rss_bytes=[1, 1, 1, 1],
        )

    changed_weak = copy.deepcopy(weak_rows)
    changed_weak[0]["prediction_row_sha256"] = _sha("forged-weak-prediction")
    changed_weak[0]["deterministic_projection_sha256"] = protocol.canonical_json_sha256(
        {
            key: changed_weak[0][key]
            for key in protocol.WEAK_ROW_FIELDS
            if key not in {
                "deterministic_projection_sha256",
                "weak_row_payload_sha256",
            }
        }
    )
    changed_weak[0]["weak_row_payload_sha256"] = protocol.payload_hash(
        changed_weak[0],
        "weak_row_payload_sha256",
    )
    with pytest.raises(ValueError, match="prediction row SHA|weak prediction"):
        validate_repair80_summary_authoritatively(
            summary,
            repair_metric_rows=repair_rows,
            repair80_input_binding=binding,
            **_repair_source_kwargs(repair_source_context),
            schedule_weak_veto_outcome=weak_outcome,
            candidate_reference_manifest=manifest,
            artifact_root=tmp_path,
            prediction_rows=prediction_rows,
            weak_rows=changed_weak,
            aggregate_wall_seconds=10.0,
            worker_lifetime_rss_bytes=[1, 1, 1, 1],
        )


def test_repair80_metric_row_rejects_noncanonical_fallback_reason() -> None:
    weak = _weak(
        0,
        stage="repair80",
        candidate_status="tagged_fallback",
    )
    with pytest.raises(ValueError, match="fallback reason"):
        Repair80MetricRow(
            weak=weak,
            label_stratum="dense",
            source_long_track=False,
            cache_valid=True,
            projection_evaluable=True,
            fallback_reason="other_typed_fallback",
            audio_arm_seconds=10.0,
            overlap_p90_ms=None,
            candidate_section_count=None,
            current_v2_segment_count=1,
            seam_ms=None,
        )


class _FrameCountSignal:
    def __init__(self, frame_count: int) -> None:
        self.shape = (frame_count,)


def _candidate_payload(index: int) -> dict[str, Any]:
    input_sha = _input_sha(index)
    frame_rate_hz = candidate_source.GLOBAL_CONSTANT_JUMP_CONSTANTS.expected_frame_rate_hz
    frame_count = 4000 + index
    (
        _,
        frame_rate_hz,
        coverage_start_ms,
        coverage_end_ms,
        min_period_frames,
        max_period_frames,
    ) = candidate_source._prediction_geometry(  # noqa: SLF001
        _FrameCountSignal(frame_count),
        frame_rate_hz,
        candidate_source.GLOBAL_CONSTANT_JUMP_CONSTANTS,
    )
    peak_time_ms = float(index * 500)
    payload: dict[str, Any] = {
        "schema": artifacts.CANDIDATE_PAYLOAD_SCHEMA,
        "beat_peaks": [
            {
                "frame_index": index,
                "refined_frame": float(index) + 0.25,
                "time_ms": peak_time_ms,
                "confidence": 0.9,
            }
        ],
        "downbeat_peaks": [
            {
                "frame_index": index + 1,
                "refined_frame": float(index) + 1.25,
                "time_ms": peak_time_ms + 250.0,
                "confidence": 0.8,
            }
        ],
        "tempo_candidates": [
            {"bpm": 120.0, "source": "autocorrelation", "score": 1.0}
        ],
        "origin_candidates": [
            {"anchor_id": 0, "time_ms": peak_time_ms, "bpm": 120.0, "score": 0.5}
        ],
        "boundary_candidates": [
            {
                "anchor_id": 0,
                "time_ms": peak_time_ms,
                "source_peak_index": 0,
                "source_peak_time_ms": peak_time_ms,
                "source_peak_confidence": 0.9,
                "rank_score": 1.0,
                "evidence_mode": "ordinary",
                "left_period_ms": 500.0,
                "right_period_ms": 500.0,
                "ordinary_score": 1.0,
                "super_score": None,
                "downbeat_bonus": 0.0,
                "nearest_downbeat_distance_ms": 0.0,
            }
        ],
        "diagnostics": {
            "candidate_contract_version": candidate_source.CANDIDATE_CONTRACT_VERSION,
            "constants_json_sha256": (
                candidate_source.GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON_SHA256
            ),
            "pulse_correlation_version": candidate_source.PULSE_CORRELATION_VERSION,
            "boundary_candidate_score_version": (
                candidate_source.BOUNDARY_CANDIDATE_SCORE_VERSION
            ),
            "frame_count": frame_count,
            "frame_rate_hz": frame_rate_hz,
            "coverage_start_ms": coverage_start_ms,
            "coverage_end_ms": coverage_end_ms,
            "min_period_frames": min_period_frames,
            "max_period_frames": max_period_frames,
            "beat_peak_count": 1,
            "downbeat_peak_count": 1,
            "tempo_candidate_count": 1,
            "origin_candidate_count": 1,
            "boundary_candidate_count": 1,
            "input_signal_sha256": input_sha,
            "candidate_fingerprint": _sha("placeholder"),
        },
    }
    payload["diagnostics"]["candidate_fingerprint"] = _candidate_fingerprint(payload)
    return payload


def _candidate_fingerprint(payload: dict[str, Any]) -> str:
    return candidate_source._candidate_fingerprint(  # noqa: SLF001
        tempo_candidates=tuple(
            candidate_source.TempoCandidate(**candidate)
            for candidate in payload["tempo_candidates"]
        ),
        origin_candidates=tuple(
            candidate_source.OriginCandidate(**candidate)
            for candidate in payload["origin_candidates"]
        ),
        boundary_candidates=tuple(
            candidate_source.BoundaryCandidate(**candidate)
            for candidate in payload["boundary_candidates"]
        ),
        beat_peaks=tuple(
            candidate_source.MaterializedPeak(**peak)
            for peak in payload["beat_peaks"]
        ),
        downbeat_peaks=tuple(
            candidate_source.MaterializedPeak(**peak)
            for peak in payload["downbeat_peaks"]
        ),
        input_signal_sha256=payload["diagnostics"]["input_signal_sha256"],
    )


def _input_sha(index: int) -> str:
    return protocol.canonical_json_sha256(
        {"row_index": index, "signal": "synthetic"}
    )
