from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

from pulsefield_model.timing.evaluation import exp007_protocol as protocol
from pulsefield_model.timing.evaluation.exp007_metrics import (
    AudioSetBinding,
    Decision,
    RateValue,
    RatioValue,
    StatsValue,
    arithmetic_mean,
    audio_set_binding,
    classify_boundary_delta,
    classify_lower_rate,
    classify_upper_ratio,
    classify_upper_value,
    combine_decisions,
    linear_p90,
    rate_value,
    ratio_value,
    stats_value,
    undefined_ratio,
)


EXP007_EXPERIMENT_ID = "timing_v3_experiment_007"
SCHEDULE_STAGE = "schedule16"
REPAIR_STAGE = "repair80"
SCHEDULE_ROW_COUNT = 16
REPAIR_ROW_COUNT = 80
WORKER_RSS_CAP_BYTES = 4_294_967_296
ROW_TIMEOUT_SECONDS = 180.0

ComparatorState = Literal["available", "unavailable", "conflicting"]
CandidateStatus = Literal["accepted", "tagged_fallback"]
MethodStatus = Literal["accepted", "unavailable"]
CandidateFallbackReason = Literal[
    "no_origin_candidate",
    "no_local_frontier_path",
    "local_frontier_resource_cap_exceeded",
]


def _require_nonempty(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_optional_sha256(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    return _require_sha256(value, name)


def _require_index(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _present_count(value: int | None, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is unavailable for this reducer branch")
    return _require_index(value, name)


def _finite_nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and nonnegative")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _require_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be bool")
    return value


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _ratio_finite_value(value: RatioValue) -> float | None:
    if value.state in {"finite", "both_zero"}:
        assert value.value is not None
        return value.value
    return None


def _stats_present(value: StatsValue | None) -> bool:
    return value is not None and value.count > 0


def _coerce_stats_value(value: Any, name: str) -> StatsValue:
    if isinstance(value, StatsValue):
        return value
    payload = _require_mapping(value, name)
    protocol.validate_exact_fields(
        payload,
        frozenset({"count", "mean", "p50", "p90", "maximum"}),
        name,
    )
    return StatsValue(
        count=_require_index(payload.get("count"), f"{name}.count"),
        mean=payload.get("mean"),
        p50=payload.get("p50"),
        p90=payload.get("p90"),
        maximum=payload.get("maximum"),
    )


def _coerce_optional_stats_value(value: Any, name: str) -> StatsValue | None:
    if value is None:
        return None
    return _coerce_stats_value(value, name)


def _coerce_ratio_value(value: Any, name: str) -> RatioValue:
    if isinstance(value, RatioValue):
        return value
    payload = _require_mapping(value, name)
    protocol.validate_exact_fields(
        payload,
        frozenset({"state", "numerator", "denominator", "value"}),
        name,
    )
    return RatioValue(
        state=payload.get("state"),
        numerator=payload.get("numerator"),
        denominator=payload.get("denominator"),
        value=payload.get("value"),
    )


def _coerce_rate_value(value: Any, name: str) -> RateValue:
    if isinstance(value, RateValue):
        return value
    payload = _require_mapping(value, name)
    protocol.validate_exact_fields(
        payload,
        frozenset({"numerator", "denominator", "value"}),
        name,
    )
    return RateValue(
        numerator=_require_index(payload.get("numerator"), f"{name}.numerator"),
        denominator=_require_index(payload.get("denominator"), f"{name}.denominator"),
        value=payload.get("value"),
    )


@dataclass(frozen=True)
class BoundaryEvidence:
    eligible: bool
    valid_difficulty_count: int
    valid_weak_change: bool
    f1: RatioValue

    def __post_init__(self) -> None:
        if not isinstance(self.eligible, bool) or not isinstance(self.valid_weak_change, bool):
            raise ValueError("boundary booleans must be bool")
        count = _require_index(self.valid_difficulty_count, "valid_difficulty_count")
        if not self.eligible:
            if count != 0 or self.valid_weak_change or self.f1.state != "undefined":
                raise ValueError("ineligible boundary evidence must use zero/false/undefined")
        elif count <= 0:
            raise ValueError("eligible boundary evidence requires a valid difficulty")
        if self.valid_weak_change and not self.eligible:
            raise ValueError("valid weak change requires eligible boundary evidence")

    @classmethod
    def unavailable(cls) -> BoundaryEvidence:
        return cls(False, 0, False, undefined_ratio())


@dataclass(frozen=True)
class ComparatorAvailability:
    state: ComparatorState
    valid_difficulty_count: int
    invalid_difficulty_count: int
    reason: str | None
    comparator_payloads_sha256: str | None

    def __post_init__(self) -> None:
        if self.state not in {"available", "unavailable", "conflicting"}:
            raise ValueError("ComparatorAvailability.state is invalid")
        _require_index(
            self.valid_difficulty_count,
            "ComparatorAvailability.valid_difficulty_count",
        )
        _require_index(
            self.invalid_difficulty_count,
            "ComparatorAvailability.invalid_difficulty_count",
        )
        if self.state == "available":
            if self.valid_difficulty_count <= 0:
                raise ValueError("available comparator requires valid difficulties")
            if self.reason is not None:
                raise ValueError("available comparator reason must be null")
            _require_sha256(
                self.comparator_payloads_sha256 or "",
                "ComparatorAvailability.comparator_payloads_sha256",
            )
        else:
            _require_nonempty(
                self.reason or "",
                "ComparatorAvailability.reason",
            )
            _require_optional_sha256(
                self.comparator_payloads_sha256,
                "ComparatorAvailability.comparator_payloads_sha256",
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhaseSummary:
    current_v2_ms: StatsValue | None
    product_ms: StatsValue | None
    pure_exp006_ms: StatsValue | None

    def __post_init__(self) -> None:
        for name in ("current_v2_ms", "product_ms", "pure_exp006_ms"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, StatsValue):
                raise ValueError(f"PhaseSummary.{name} must be StatsValue or null")

    def to_dict(self) -> dict[str, Any]:
        return _nested_dataclass_dict(self)


@dataclass(frozen=True)
class DriftSummary:
    current_v2_alias_max_prefix_ms: StatsValue | None
    product_alias_max_prefix_ms: StatsValue | None
    pure_exp006_alias_max_prefix_ms: StatsValue | None

    def __post_init__(self) -> None:
        for name in (
            "current_v2_alias_max_prefix_ms",
            "product_alias_max_prefix_ms",
            "pure_exp006_alias_max_prefix_ms",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, StatsValue):
                raise ValueError(f"DriftSummary.{name} must be StatsValue or null")

    def to_dict(self) -> dict[str, Any]:
        return _nested_dataclass_dict(self)


@dataclass(frozen=True)
class BoundarySummary:
    eligible: bool
    valid_difficulty_count: int
    tp: int
    fp: int
    fn: int
    f1: RatioValue
    matched_error_ms: StatsValue
    weak_consensus_supported_count: int

    def __post_init__(self) -> None:
        _require_bool(self.eligible, "BoundarySummary.eligible")
        valid_count = _require_index(
            self.valid_difficulty_count,
            "BoundarySummary.valid_difficulty_count",
        )
        tp = _require_index(self.tp, "BoundarySummary.tp")
        fp = _require_index(self.fp, "BoundarySummary.fp")
        fn = _require_index(self.fn, "BoundarySummary.fn")
        consensus = _require_index(
            self.weak_consensus_supported_count,
            "BoundarySummary.weak_consensus_supported_count",
        )
        if consensus > valid_count:
            raise ValueError("BoundarySummary consensus count cannot exceed valid count")
        if not isinstance(self.f1, RatioValue):
            raise ValueError("BoundarySummary.f1 must be RatioValue")
        if not isinstance(self.matched_error_ms, StatsValue):
            raise ValueError("BoundarySummary.matched_error_ms must be StatsValue")
        if not self.eligible:
            if (
                valid_count,
                tp,
                fp,
                fn,
                consensus,
                self.matched_error_ms.count,
            ) != (0, 0, 0, 0, 0, 0):
                raise ValueError("ineligible BoundarySummary must use zero counts")
            if self.f1 != undefined_ratio():
                raise ValueError("ineligible BoundarySummary requires undefined F1")
            return
        if valid_count <= 0:
            raise ValueError("eligible BoundarySummary requires valid difficulties")
        expected_f1 = ratio_value(2 * tp, 2 * tp + fp + fn)
        if self.f1 != expected_f1:
            raise ValueError("BoundarySummary F1 does not match TP/FP/FN counts")
        if self.matched_error_ms.count != tp:
            raise ValueError("BoundarySummary matched_error count must equal TP count")

    @classmethod
    def ineligible(cls) -> BoundarySummary:
        return cls(False, 0, 0, 0, 0, undefined_ratio(), stats_value([]), 0)

    def to_dict(self) -> dict[str, Any]:
        return _nested_dataclass_dict(self)


@dataclass(frozen=True)
class ObjectGridSummary:
    eligible: bool
    object_count: int
    start_residual_ms: StatsValue
    end_residual_ms: StatsValue
    inlier_count: int
    inlier_rate: RateValue | None

    def __post_init__(self) -> None:
        _require_bool(self.eligible, "ObjectGridSummary.eligible")
        object_count = _require_index(
            self.object_count,
            "ObjectGridSummary.object_count",
        )
        inlier_count = _require_index(
            self.inlier_count,
            "ObjectGridSummary.inlier_count",
        )
        if inlier_count > object_count:
            raise ValueError("ObjectGridSummary inlier_count cannot exceed object_count")
        if not isinstance(self.start_residual_ms, StatsValue) or not isinstance(
            self.end_residual_ms,
            StatsValue,
        ):
            raise ValueError("ObjectGridSummary residuals must be StatsValue")
        if not self.eligible:
            if (
                object_count,
                inlier_count,
                self.start_residual_ms.count,
                self.end_residual_ms.count,
            ) != (0, 0, 0, 0):
                raise ValueError("ineligible ObjectGridSummary must use zero counts")
            if self.inlier_rate is not None:
                raise ValueError("ineligible ObjectGridSummary inlier_rate must be null")
            return
        if object_count <= 0:
            raise ValueError("eligible ObjectGridSummary requires objects")
        if self.start_residual_ms.count != object_count:
            raise ValueError("ObjectGridSummary start residual count mismatch")
        if self.end_residual_ms.count != object_count:
            raise ValueError("ObjectGridSummary end residual count mismatch")
        if self.inlier_rate != rate_value(inlier_count, object_count):
            raise ValueError("ObjectGridSummary inlier_rate mismatch")

    @classmethod
    def ineligible(cls) -> ObjectGridSummary:
        return cls(False, 0, stats_value([]), stats_value([]), 0, None)

    def to_dict(self) -> dict[str, Any]:
        return _nested_dataclass_dict(self)


@dataclass(frozen=True)
class WeakRow:
    schema: str
    experiment_id: str
    stage: Literal["schedule16", "repair80"]
    schema_descriptor_sha256: str
    schedule_arm: str
    row_index: int
    cache_audio_key: str
    audio_group_key: str
    prediction_row_sha256: str
    four_arm_stage_summary_sha256: str
    candidate_global_manifest_sha256: str
    source_selection_sha256: str
    comparator_availability: ComparatorAvailability
    current_v2_phase_matched: bool
    pure_exp006_phase_matched: bool
    selected_safety_phase_matched: bool
    phase_metrics_summary: PhaseSummary
    drift_metrics_summary: DriftSummary
    current_v2_boundary_summary: BoundarySummary
    pure_exp006_boundary_summary: BoundarySummary
    selected_boundary_summary: BoundarySummary
    object_grid_summary: ObjectGridSummary
    deterministic_projection_sha256: str
    weak_row_payload_sha256: str

    def __post_init__(self) -> None:
        validate_weak_row(self.to_dict())

    def ref(self) -> WeakRowRef:
        return WeakRowRef(
            self.row_index,
            self.cache_audio_key,
            self.prediction_row_sha256,
            self.weak_row_payload_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        return _nested_dataclass_dict(self)


@dataclass(frozen=True)
class PredictionRowRef:
    row_index: int
    cache_audio_key: str
    prediction_row_sha256: str
    schedule_arm: str

    def __post_init__(self) -> None:
        _require_index(self.row_index, "row_index")
        _require_nonempty(self.cache_audio_key, "cache_audio_key")
        _require_sha256(self.prediction_row_sha256, "prediction_row_sha256")
        _require_nonempty(self.schedule_arm, "schedule_arm")


@dataclass(frozen=True)
class WeakRowRef:
    row_index: int
    cache_audio_key: str
    prediction_row_sha256: str
    weak_row_payload_sha256: str

    def __post_init__(self) -> None:
        _require_index(self.row_index, "row_index")
        _require_nonempty(self.cache_audio_key, "cache_audio_key")
        _require_sha256(self.prediction_row_sha256, "prediction_row_sha256")
        _require_sha256(self.weak_row_payload_sha256, "weak_row_payload_sha256")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WeakPendingRowRef:
    row_index: int
    cache_audio_key: str
    prediction_row_sha256: str

    def __post_init__(self) -> None:
        _require_index(self.row_index, "row_index")
        _require_nonempty(self.cache_audio_key, "cache_audio_key")
        _require_sha256(self.prediction_row_sha256, "prediction_row_sha256")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_comparator_availability(payload: Mapping[str, Any]) -> dict[str, Any]:
    payload = _require_mapping(payload, "ComparatorAvailability")
    protocol.validate_exact_fields(
        payload,
        protocol.COMPARATOR_AVAILABILITY_FIELDS,
        "ComparatorAvailability",
    )
    state = payload.get("state")
    availability = ComparatorAvailability(
        state=state,  # type: ignore[arg-type]
        valid_difficulty_count=_require_index(
            payload.get("valid_difficulty_count"),
            "ComparatorAvailability.valid_difficulty_count",
        ),
        invalid_difficulty_count=_require_index(
            payload.get("invalid_difficulty_count"),
            "ComparatorAvailability.invalid_difficulty_count",
        ),
        reason=payload.get("reason"),
        comparator_payloads_sha256=payload.get("comparator_payloads_sha256"),
    )
    return availability.to_dict()


def validate_phase_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    payload = _require_mapping(payload, "PhaseSummary")
    protocol.validate_exact_fields(payload, protocol.PHASE_SUMMARY_FIELDS, "PhaseSummary")
    summary = PhaseSummary(
        current_v2_ms=_coerce_optional_stats_value(
            payload.get("current_v2_ms"),
            "PhaseSummary.current_v2_ms",
        ),
        product_ms=_coerce_optional_stats_value(
            payload.get("product_ms"),
            "PhaseSummary.product_ms",
        ),
        pure_exp006_ms=_coerce_optional_stats_value(
            payload.get("pure_exp006_ms"),
            "PhaseSummary.pure_exp006_ms",
        ),
    )
    return summary.to_dict()


def validate_drift_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    payload = _require_mapping(payload, "DriftSummary")
    protocol.validate_exact_fields(payload, protocol.DRIFT_SUMMARY_FIELDS, "DriftSummary")
    summary = DriftSummary(
        current_v2_alias_max_prefix_ms=_coerce_optional_stats_value(
            payload.get("current_v2_alias_max_prefix_ms"),
            "DriftSummary.current_v2_alias_max_prefix_ms",
        ),
        product_alias_max_prefix_ms=_coerce_optional_stats_value(
            payload.get("product_alias_max_prefix_ms"),
            "DriftSummary.product_alias_max_prefix_ms",
        ),
        pure_exp006_alias_max_prefix_ms=_coerce_optional_stats_value(
            payload.get("pure_exp006_alias_max_prefix_ms"),
            "DriftSummary.pure_exp006_alias_max_prefix_ms",
        ),
    )
    return summary.to_dict()


def validate_boundary_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    payload = _require_mapping(payload, "BoundarySummary")
    protocol.validate_exact_fields(
        payload,
        protocol.BOUNDARY_SUMMARY_FIELDS,
        "BoundarySummary",
    )
    summary = BoundarySummary(
        eligible=_require_bool(payload.get("eligible"), "BoundarySummary.eligible"),
        valid_difficulty_count=_require_index(
            payload.get("valid_difficulty_count"),
            "BoundarySummary.valid_difficulty_count",
        ),
        tp=_require_index(payload.get("tp"), "BoundarySummary.tp"),
        fp=_require_index(payload.get("fp"), "BoundarySummary.fp"),
        fn=_require_index(payload.get("fn"), "BoundarySummary.fn"),
        f1=_coerce_ratio_value(payload.get("f1"), "BoundarySummary.f1"),
        matched_error_ms=_coerce_stats_value(
            payload.get("matched_error_ms"),
            "BoundarySummary.matched_error_ms",
        ),
        weak_consensus_supported_count=_require_index(
            payload.get("weak_consensus_supported_count"),
            "BoundarySummary.weak_consensus_supported_count",
        ),
    )
    return summary.to_dict()


def validate_object_grid_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    payload = _require_mapping(payload, "ObjectGridSummary")
    protocol.validate_exact_fields(
        payload,
        protocol.OBJECT_GRID_SUMMARY_FIELDS,
        "ObjectGridSummary",
    )
    inlier_rate_payload = payload.get("inlier_rate")
    summary = ObjectGridSummary(
        eligible=_require_bool(payload.get("eligible"), "ObjectGridSummary.eligible"),
        object_count=_require_index(
            payload.get("object_count"),
            "ObjectGridSummary.object_count",
        ),
        start_residual_ms=_coerce_stats_value(
            payload.get("start_residual_ms"),
            "ObjectGridSummary.start_residual_ms",
        ),
        end_residual_ms=_coerce_stats_value(
            payload.get("end_residual_ms"),
            "ObjectGridSummary.end_residual_ms",
        ),
        inlier_count=_require_index(
            payload.get("inlier_count"),
            "ObjectGridSummary.inlier_count",
        ),
        inlier_rate=(
            None
            if inlier_rate_payload is None
            else _coerce_rate_value(
                inlier_rate_payload,
                "ObjectGridSummary.inlier_rate",
            )
        ),
    )
    return summary.to_dict()


def make_weak_row(
    *,
    stage: str,
    schedule_arm: str,
    row_index: int,
    cache_audio_key: str,
    audio_group_key: str,
    prediction_row_sha256: str,
    four_arm_stage_summary_sha256: str,
    candidate_global_manifest_sha256: str,
    source_selection_sha256: str,
    comparator_availability: ComparatorAvailability | Mapping[str, Any],
    current_v2_phase_matched: bool,
    pure_exp006_phase_matched: bool,
    selected_safety_phase_matched: bool,
    phase_metrics_summary: PhaseSummary | Mapping[str, Any],
    drift_metrics_summary: DriftSummary | Mapping[str, Any],
    current_v2_boundary_summary: BoundarySummary | Mapping[str, Any],
    pure_exp006_boundary_summary: BoundarySummary | Mapping[str, Any],
    selected_boundary_summary: BoundarySummary | Mapping[str, Any],
    object_grid_summary: ObjectGridSummary | Mapping[str, Any],
    schema_descriptor_sha256: str | None = None,
) -> dict[str, Any]:
    descriptor = schema_descriptor_sha256 or protocol.schema_descriptor_sha256(
        protocol.WEAK_ROW_SCHEMA
    )
    if descriptor != protocol.schema_descriptor_sha256(protocol.WEAK_ROW_SCHEMA):
        raise ValueError("WeakRow schema descriptor must match protocol registry")
    base = {
        "schema": protocol.WEAK_ROW_SCHEMA,
        "experiment_id": EXP007_EXPERIMENT_ID,
        "stage": stage,
        "schema_descriptor_sha256": descriptor,
        "schedule_arm": schedule_arm,
        "row_index": row_index,
        "cache_audio_key": cache_audio_key,
        "audio_group_key": audio_group_key,
        "prediction_row_sha256": prediction_row_sha256,
        "four_arm_stage_summary_sha256": four_arm_stage_summary_sha256,
        "candidate_global_manifest_sha256": candidate_global_manifest_sha256,
        "source_selection_sha256": source_selection_sha256,
        "comparator_availability": (
            comparator_availability.to_dict()
            if isinstance(comparator_availability, ComparatorAvailability)
            else dict(comparator_availability)
        ),
        "current_v2_phase_matched": current_v2_phase_matched,
        "pure_exp006_phase_matched": pure_exp006_phase_matched,
        "selected_safety_phase_matched": selected_safety_phase_matched,
        "phase_metrics_summary": (
            phase_metrics_summary.to_dict()
            if isinstance(phase_metrics_summary, PhaseSummary)
            else dict(phase_metrics_summary)
        ),
        "drift_metrics_summary": (
            drift_metrics_summary.to_dict()
            if isinstance(drift_metrics_summary, DriftSummary)
            else dict(drift_metrics_summary)
        ),
        "current_v2_boundary_summary": (
            current_v2_boundary_summary.to_dict()
            if isinstance(current_v2_boundary_summary, BoundarySummary)
            else dict(current_v2_boundary_summary)
        ),
        "pure_exp006_boundary_summary": (
            pure_exp006_boundary_summary.to_dict()
            if isinstance(pure_exp006_boundary_summary, BoundarySummary)
            else dict(pure_exp006_boundary_summary)
        ),
        "selected_boundary_summary": (
            selected_boundary_summary.to_dict()
            if isinstance(selected_boundary_summary, BoundarySummary)
            else dict(selected_boundary_summary)
        ),
        "object_grid_summary": (
            object_grid_summary.to_dict()
            if isinstance(object_grid_summary, ObjectGridSummary)
            else dict(object_grid_summary)
        ),
    }
    projection_sha = protocol.canonical_json_sha256(base)
    payload = {
        **base,
        "deterministic_projection_sha256": projection_sha,
    }
    payload["weak_row_payload_sha256"] = protocol.payload_hash(
        payload,
        "weak_row_payload_sha256",
    )
    return validate_weak_row(payload)


def validate_weak_row(
    payload: Mapping[str, Any],
    *,
    expected_stage: str | None = None,
    expected_schedule_arm: str | None = None,
    expected_four_arm_stage_summary_sha256: str | None = None,
    expected_candidate_global_manifest_sha256: str | None = None,
    expected_source_selection_sha256: str | None = None,
    paired_prediction_row_sha256: str | None = None,
) -> dict[str, Any]:
    payload = _require_mapping(payload, "WeakRow")
    protocol.validate_exact_fields(payload, protocol.WEAK_ROW_FIELDS, "WeakRow")
    if payload.get("schema") != protocol.WEAK_ROW_SCHEMA:
        raise ValueError("WeakRow schema is invalid")
    if payload.get("experiment_id") != EXP007_EXPERIMENT_ID:
        raise ValueError("WeakRow experiment_id is invalid")
    stage = payload.get("stage")
    if stage not in {SCHEDULE_STAGE, REPAIR_STAGE}:
        raise ValueError("WeakRow stage is invalid")
    if expected_stage is not None and stage != expected_stage:
        raise ValueError("WeakRow cross-stage reference mismatch")
    descriptor = _require_sha256(
        payload.get("schema_descriptor_sha256"),
        "WeakRow.schema_descriptor_sha256",
    )
    if descriptor != protocol.schema_descriptor_sha256(protocol.WEAK_ROW_SCHEMA):
        raise ValueError("WeakRow schema descriptor must match protocol registry")
    schedule_arm = _require_nonempty(payload.get("schedule_arm"), "WeakRow.schedule_arm")
    if expected_schedule_arm is not None and schedule_arm != expected_schedule_arm:
        raise ValueError("WeakRow schedule arm mismatch")
    row_index = _require_index(payload.get("row_index"), "WeakRow.row_index")
    cache_audio_key = _require_nonempty(
        payload.get("cache_audio_key"),
        "WeakRow.cache_audio_key",
    )
    audio_group_key = _require_nonempty(
        payload.get("audio_group_key"),
        "WeakRow.audio_group_key",
    )
    prediction_sha = _require_sha256(
        payload.get("prediction_row_sha256"),
        "WeakRow.prediction_row_sha256",
    )
    if paired_prediction_row_sha256 is not None and prediction_sha != paired_prediction_row_sha256:
        raise ValueError("WeakRow prediction row SHA mismatch")
    for field_name, expected in (
        (
            "four_arm_stage_summary_sha256",
            expected_four_arm_stage_summary_sha256,
        ),
        (
            "candidate_global_manifest_sha256",
            expected_candidate_global_manifest_sha256,
        ),
        ("source_selection_sha256", expected_source_selection_sha256),
    ):
        value = _require_sha256(payload.get(field_name), f"WeakRow.{field_name}")
        if expected is not None and value != expected:
            raise ValueError(f"WeakRow {field_name} dependency mismatch")
    comparator = validate_comparator_availability(payload.get("comparator_availability"))
    current_matched = _require_bool(
        payload.get("current_v2_phase_matched"),
        "WeakRow.current_v2_phase_matched",
    )
    pure_matched = _require_bool(
        payload.get("pure_exp006_phase_matched"),
        "WeakRow.pure_exp006_phase_matched",
    )
    selected_matched = _require_bool(
        payload.get("selected_safety_phase_matched"),
        "WeakRow.selected_safety_phase_matched",
    )
    if comparator["state"] != "available" and (
        current_matched or pure_matched or selected_matched
    ):
        raise ValueError("WeakRow unavailable/conflicting comparator requires false flags")
    if pure_matched and not current_matched:
        raise ValueError("WeakRow pure phase match requires current-v2 match")
    if selected_matched and not current_matched:
        raise ValueError("WeakRow selected phase match requires current-v2 match")
    phase = validate_phase_summary(payload.get("phase_metrics_summary"))
    drift = validate_drift_summary(payload.get("drift_metrics_summary"))
    _validate_weak_row_null_matrix(
        current_matched=current_matched,
        pure_matched=pure_matched,
        selected_matched=selected_matched,
        phase=phase,
        drift=drift,
    )
    current_boundary = validate_boundary_summary(
        payload.get("current_v2_boundary_summary")
    )
    pure_boundary = validate_boundary_summary(payload.get("pure_exp006_boundary_summary"))
    selected_boundary = validate_boundary_summary(
        payload.get("selected_boundary_summary")
    )
    _validate_weak_row_boundary_branch(
        comparator_state=comparator["state"],
        current_matched=current_matched,
        pure_matched=pure_matched,
        selected_matched=selected_matched,
        current_boundary=current_boundary,
        pure_boundary=pure_boundary,
        selected_boundary=selected_boundary,
    )
    object_grid = validate_object_grid_summary(payload.get("object_grid_summary"))
    if object_grid["eligible"] and (
        comparator["state"] != "available" or not selected_matched
    ):
        raise ValueError("WeakRow object grid requires comparator and selected grid")
    projection_base = {
        key: payload[key]
        for key in protocol.WEAK_ROW_FIELDS
        if key not in {"deterministic_projection_sha256", "weak_row_payload_sha256"}
    }
    projection_sha = _require_sha256(
        payload.get("deterministic_projection_sha256"),
        "WeakRow.deterministic_projection_sha256",
    )
    if projection_sha != protocol.canonical_json_sha256(projection_base):
        raise ValueError("WeakRow deterministic projection hash mismatch")
    result = {
        "schema": protocol.WEAK_ROW_SCHEMA,
        "experiment_id": EXP007_EXPERIMENT_ID,
        "stage": stage,
        "schema_descriptor_sha256": descriptor,
        "schedule_arm": schedule_arm,
        "row_index": row_index,
        "cache_audio_key": cache_audio_key,
        "audio_group_key": audio_group_key,
        "prediction_row_sha256": prediction_sha,
        "four_arm_stage_summary_sha256": payload["four_arm_stage_summary_sha256"],
        "candidate_global_manifest_sha256": payload[
            "candidate_global_manifest_sha256"
        ],
        "source_selection_sha256": payload["source_selection_sha256"],
        "comparator_availability": comparator,
        "current_v2_phase_matched": current_matched,
        "pure_exp006_phase_matched": pure_matched,
        "selected_safety_phase_matched": selected_matched,
        "phase_metrics_summary": phase,
        "drift_metrics_summary": drift,
        "current_v2_boundary_summary": current_boundary,
        "pure_exp006_boundary_summary": pure_boundary,
        "selected_boundary_summary": selected_boundary,
        "object_grid_summary": object_grid,
        "deterministic_projection_sha256": projection_sha,
        "weak_row_payload_sha256": _require_sha256(
            payload.get("weak_row_payload_sha256"),
            "WeakRow.weak_row_payload_sha256",
        ),
    }
    if result["weak_row_payload_sha256"] != protocol.payload_hash(
        result,
        "weak_row_payload_sha256",
    ):
        raise ValueError("WeakRow payload hash mismatch")
    return result


def weak_row_ref_from_row(payload: Mapping[str, Any]) -> WeakRowRef:
    row = validate_weak_row(payload)
    return WeakRowRef(
        row["row_index"],
        row["cache_audio_key"],
        row["prediction_row_sha256"],
        row["weak_row_payload_sha256"],
    )


def _validate_weak_row_null_matrix(
    *,
    current_matched: bool,
    pure_matched: bool,
    selected_matched: bool,
    phase: Mapping[str, Any],
    drift: Mapping[str, Any],
) -> None:
    checks = (
        (
            current_matched,
            phase.get("current_v2_ms"),
            drift.get("current_v2_alias_max_prefix_ms"),
            "current_v2",
        ),
        (
            pure_matched,
            phase.get("pure_exp006_ms"),
            drift.get("pure_exp006_alias_max_prefix_ms"),
            "pure_exp006",
        ),
        (
            selected_matched,
            phase.get("product_ms"),
            drift.get("product_alias_max_prefix_ms"),
            "product",
        ),
    )
    for matched, phase_value, drift_value, name in checks:
        if matched:
            if not _stats_present(_coerce_stats_value(phase_value, f"{name}.phase")):
                raise ValueError(f"WeakRow {name} phase StatsValue must be nonempty")
            if not _stats_present(_coerce_stats_value(drift_value, f"{name}.drift")):
                raise ValueError(f"WeakRow {name} drift StatsValue must be nonempty")
        elif phase_value is not None or drift_value is not None:
            raise ValueError(f"WeakRow {name} violates the exact null matrix")


def _validate_weak_row_boundary_branch(
    *,
    comparator_state: str,
    current_matched: bool,
    pure_matched: bool,
    selected_matched: bool,
    current_boundary: Mapping[str, Any],
    pure_boundary: Mapping[str, Any],
    selected_boundary: Mapping[str, Any],
) -> None:
    for name, matched, boundary in (
        ("current_v2", current_matched, current_boundary),
        ("pure_exp006", pure_matched, pure_boundary),
        ("selected", selected_matched, selected_boundary),
    ):
        if boundary["eligible"] and (comparator_state != "available" or not matched):
            raise ValueError(f"WeakRow {name} boundary requires comparator and grid")


@dataclass(frozen=True)
class WeakMetricRow:
    stage: Literal["schedule16", "repair80"]
    row_index: int
    cache_audio_key: str
    audio_group_key: str
    prediction_row_sha256: str
    weak_row_payload_sha256: str
    schedule_arm: str
    comparator_state: ComparatorState
    candidate_status: CandidateStatus
    baseline_status: MethodStatus
    selected_status: MethodStatus
    product_grid_available: bool
    current_v2_phase_matched: bool
    pure_exp006_phase_matched: bool
    selected_safety_phase_matched: bool
    current_v2_phase_ms: StatsValue | None
    pure_exp006_phase_ms: StatsValue | None
    product_phase_ms: StatsValue | None
    current_v2_alias_max_prefix_ms: StatsValue | None
    pure_exp006_alias_max_prefix_ms: StatsValue | None
    product_alias_max_prefix_ms: StatsValue | None
    current_v2_boundary: BoundaryEvidence
    pure_exp006_boundary: BoundaryEvidence
    selected_boundary: BoundaryEvidence

    def __post_init__(self) -> None:
        if self.stage not in {SCHEDULE_STAGE, REPAIR_STAGE}:
            raise ValueError("weak row stage must be schedule16 or repair80")
        _require_index(self.row_index, "row_index")
        _require_nonempty(self.cache_audio_key, "cache_audio_key")
        _require_nonempty(self.audio_group_key, "audio_group_key")
        _require_sha256(self.prediction_row_sha256, "prediction_row_sha256")
        _require_sha256(self.weak_row_payload_sha256, "weak_row_payload_sha256")
        _require_nonempty(self.schedule_arm, "schedule_arm")
        if self.comparator_state not in {"available", "unavailable", "conflicting"}:
            raise ValueError("invalid comparator state")
        if self.candidate_status not in {"accepted", "tagged_fallback"}:
            raise ValueError("invalid candidate status")
        if self.baseline_status not in {"accepted", "unavailable"}:
            raise ValueError("invalid baseline status")
        if self.selected_status not in {"accepted", "unavailable"}:
            raise ValueError("invalid selected status")
        for name in (
            "product_grid_available",
            "current_v2_phase_matched",
            "pure_exp006_phase_matched",
            "selected_safety_phase_matched",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")
        expected_selected = (
            self.candidate_status == "accepted" or self.baseline_status == "accepted"
        )
        if (self.selected_status == "accepted") != expected_selected:
            raise ValueError("selected status violates the four-branch product truth table")
        if self.product_grid_available != (self.selected_status == "accepted"):
            raise ValueError("product grid availability must match selected status")
        if self.pure_exp006_phase_matched and (
            not self.current_v2_phase_matched or self.candidate_status != "accepted"
        ):
            raise ValueError("pure phase match requires accepted candidate and current-v2 match")
        if self.selected_safety_phase_matched and (
            not self.current_v2_phase_matched or not self.product_grid_available
        ):
            raise ValueError("selected phase match requires product grid and current-v2 match")
        if self.current_v2_phase_matched and self.baseline_status != "accepted":
            raise ValueError("current-v2 phase match requires accepted baseline")
        if self.comparator_state != "available" and any(
            (
                self.current_v2_phase_matched,
                self.pure_exp006_phase_matched,
                self.selected_safety_phase_matched,
            )
        ):
            raise ValueError("unusable comparator requires all phase flags false")
        matrix = (
            (
                self.current_v2_phase_matched,
                self.current_v2_phase_ms,
                self.current_v2_alias_max_prefix_ms,
                "current_v2",
            ),
            (
                self.pure_exp006_phase_matched,
                self.pure_exp006_phase_ms,
                self.pure_exp006_alias_max_prefix_ms,
                "pure_exp006",
            ),
            (
                self.selected_safety_phase_matched,
                self.product_phase_ms,
                self.product_alias_max_prefix_ms,
                "product",
            ),
        )
        for matched, phase, drift, name in matrix:
            if matched != (_stats_present(phase) and _stats_present(drift)):
                raise ValueError(f"{name} phase/drift fields violate the exact null matrix")
        boundaries = (
            (self.current_v2_boundary, self.baseline_status == "accepted"),
            (self.pure_exp006_boundary, self.candidate_status == "accepted"),
            (self.selected_boundary, self.selected_status == "accepted"),
        )
        for boundary, grid_available in boundaries:
            if boundary.eligible and (
                self.comparator_state != "available" or not grid_available
            ):
                raise ValueError("eligible boundary evidence requires comparator and named grid")
            if self.comparator_state != "available" and boundary.eligible:
                raise ValueError("unusable comparator makes every boundary ineligible")

    def ref(self) -> WeakRowRef:
        return WeakRowRef(
            self.row_index,
            self.cache_audio_key,
            self.prediction_row_sha256,
            self.weak_row_payload_sha256,
        )


@dataclass(frozen=True)
class ScheduleWeakDenominators:
    stage_audio_count: int
    stage_audio: AudioSetBinding
    comparator_available_audio: AudioSetBinding
    comparator_unavailable_audio: AudioSetBinding
    comparator_conflicting_audio: AudioSetBinding
    current_v2_phase_matched: AudioSetBinding
    pure_exp006_phase_matched: AudioSetBinding
    selected_safety_phase_matched: AudioSetBinding
    phase_common: AudioSetBinding
    alias_drift_common: AudioSetBinding
    weak_change_boundary_audio: AudioSetBinding

    def to_dict(self) -> dict[str, Any]:
        return _nested_dataclass_dict(self)


CoverageValue = RateValue | RatioValue


@dataclass(frozen=True)
class ScheduleWeakGates:
    pure_mean_phase_ratio: RatioValue
    pure_p90_phase_ratio: RatioValue
    pure_phase_coverage: CoverageValue
    current_v2_phase_mean_ms: float | None
    pure_exp006_phase_mean_ms: float | None
    current_v2_phase_p90_ms: float | None
    pure_exp006_phase_p90_ms: float | None
    current_v2_alias_drift_mean_ms: float | None
    pure_exp006_alias_drift_mean_ms: float | None
    current_v2_alias_drift_p90_ms: float | None
    pure_exp006_alias_drift_p90_ms: float | None
    alias_max_prefix_drift_mean_ratio: RatioValue
    alias_max_prefix_drift_p90_ratio: RatioValue
    current_v2_boundary_f1_mean: float | None
    pure_exp006_boundary_f1_mean: float | None
    selected_boundary_f1_mean: float | None
    pure_minus_v2_boundary_f1_delta: float | None

    def to_dict(self) -> dict[str, Any]:
        return _nested_dataclass_dict(self)


@dataclass(frozen=True)
class ScheduleWeakEvaluation:
    denominators: ScheduleWeakDenominators
    gates: ScheduleWeakGates
    gate_decisions: Mapping[str, Decision]
    decision: Decision
    action: Literal["authorize_repair80", "stop_ambiguous", "stop_negative"]
    weak_row_refs: tuple[WeakRowRef, ...]
    selected_row_refs_sha256: str
    row_weak_pairs_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return _nested_dataclass_dict(self)


def evaluate_schedule_weak_veto(
    rows: Sequence[WeakMetricRow],
    *,
    selected_rows: Sequence[PredictionRowRef],
    selected_schedule_arm: str,
) -> ScheduleWeakEvaluation:
    _validate_winner_pairing(
        rows,
        selected_rows=selected_rows,
        selected_schedule_arm=selected_schedule_arm,
        stage=SCHEDULE_STAGE,
        expected_count=SCHEDULE_ROW_COUNT,
        require_unique_audio_group=True,
    )
    by_key = {row.cache_audio_key: row for row in rows}
    keys = tuple(ref.cache_audio_key for ref in selected_rows)

    def selected(predicate: Any) -> tuple[str, ...]:
        return tuple(key for key in keys if predicate(by_key[key]))

    available = selected(lambda row: row.comparator_state == "available")
    unavailable = selected(lambda row: row.comparator_state == "unavailable")
    conflicting = selected(lambda row: row.comparator_state == "conflicting")
    current = selected(lambda row: row.current_v2_phase_matched)
    pure = selected(lambda row: row.pure_exp006_phase_matched)
    product = selected(lambda row: row.selected_safety_phase_matched)
    phase_common = selected(
        lambda row: row.candidate_status == "accepted"
        and row.baseline_status == "accepted"
        and row.comparator_state == "available"
        and row.current_v2_phase_matched
        and row.pure_exp006_phase_matched
    )
    drift_common = selected(
        lambda row: row.cache_audio_key in phase_common
        and _stats_present(row.current_v2_alias_max_prefix_ms)
        and _stats_present(row.pure_exp006_alias_max_prefix_ms)
    )
    boundary_common = selected(
        lambda row: row.candidate_status == "accepted"
        and row.baseline_status == "accepted"
        and row.comparator_state == "available"
        and row.current_v2_boundary.valid_weak_change
        and row.pure_exp006_boundary.valid_weak_change
        and _ratio_finite_value(row.current_v2_boundary.f1) is not None
        and _ratio_finite_value(row.pure_exp006_boundary.f1) is not None
    )
    denominators = ScheduleWeakDenominators(
        stage_audio_count=SCHEDULE_ROW_COUNT,
        stage_audio=audio_set_binding(keys),
        comparator_available_audio=audio_set_binding(available),
        comparator_unavailable_audio=audio_set_binding(unavailable),
        comparator_conflicting_audio=audio_set_binding(conflicting),
        current_v2_phase_matched=audio_set_binding(current),
        pure_exp006_phase_matched=audio_set_binding(pure),
        selected_safety_phase_matched=audio_set_binding(product),
        phase_common=audio_set_binding(phase_common),
        alias_drift_common=audio_set_binding(drift_common),
        weak_change_boundary_audio=audio_set_binding(boundary_common),
    )
    gates = _schedule_gates(by_key, phase_common, drift_common, boundary_common, current, pure)
    decisions: dict[str, Decision] = {
        "current_v2_phase_minimum": "pass" if len(current) >= 8 else "ambiguous",
        "pure_exp006_phase_minimum": "pass" if len(pure) >= 8 else "ambiguous",
        "selected_safety_phase_minimum": "pass" if len(product) >= 8 else "ambiguous",
        "phase_common_minimum": "pass" if len(phase_common) >= 8 else "ambiguous",
        "alias_drift_common_minimum": "pass" if len(drift_common) >= 8 else "ambiguous",
        "boundary_common_minimum": "pass" if len(boundary_common) >= 5 else "ambiguous",
        "comparator_conflict": "ambiguous" if conflicting else "pass",
        "pure_mean_phase_ratio": classify_upper_ratio(
            gates.pure_mean_phase_ratio, pass_max=1.05, ambiguous_max=1.10
        ),
        "pure_p90_phase_ratio": classify_upper_ratio(
            gates.pure_p90_phase_ratio, pass_max=1.10, ambiguous_max=1.15
        ),
        "pure_phase_coverage": classify_lower_rate(
            gates.pure_phase_coverage, pass_min=0.95, ambiguous_min=0.90
        ),
        "alias_drift_mean_ratio": classify_upper_ratio(
            gates.alias_max_prefix_drift_mean_ratio,
            pass_max=1.15,
            ambiguous_max=1.30,
        ),
        "alias_drift_p90_ratio": classify_upper_ratio(
            gates.alias_max_prefix_drift_p90_ratio,
            pass_max=1.15,
            ambiguous_max=1.30,
        ),
        "boundary_f1_delta": classify_boundary_delta(
            gates.pure_minus_v2_boundary_f1_delta
        ),
    }
    decision = combine_decisions(decisions.values())
    action = {
        "pass": "authorize_repair80",
        "ambiguous": "stop_ambiguous",
        "negative": "stop_negative",
    }[decision]
    weak_refs = tuple(row.ref() for row in rows)
    selected_hash_payload = [
        {
            "row_index": ref.row_index,
            "cache_audio_key": ref.cache_audio_key,
            "prediction_row_sha256": ref.prediction_row_sha256,
        }
        for ref in selected_rows
    ]
    pair_hash_payload = [
        {
            "row_index": row.row_index,
            "cache_audio_key": row.cache_audio_key,
            "row_payload_sha256": row.prediction_row_sha256,
            "prediction_row_sha256": row.prediction_row_sha256,
            "weak_row_payload_sha256": row.weak_row_payload_sha256,
        }
        for row in rows
    ]
    return ScheduleWeakEvaluation(
        denominators=denominators,
        gates=gates,
        gate_decisions=decisions,
        decision=decision,
        action=action,
        weak_row_refs=weak_refs,
        selected_row_refs_sha256=protocol.canonical_json_sha256(selected_hash_payload),
        row_weak_pairs_sha256=protocol.canonical_json_sha256(pair_hash_payload),
    )


def _schedule_gates(
    by_key: Mapping[str, WeakMetricRow],
    phase_common: Sequence[str],
    drift_common: Sequence[str],
    boundary_common: Sequence[str],
    current: Sequence[str],
    pure: Sequence[str],
) -> ScheduleWeakGates:
    if len(phase_common) >= 8:
        current_mean = arithmetic_mean(
            by_key[key].current_v2_phase_ms.mean for key in phase_common  # type: ignore[union-attr]
        )
        pure_mean = arithmetic_mean(
            by_key[key].pure_exp006_phase_ms.mean for key in phase_common  # type: ignore[union-attr]
        )
        current_p90 = linear_p90(
            by_key[key].current_v2_phase_ms.p90 for key in phase_common  # type: ignore[union-attr]
        )
        pure_p90 = linear_p90(
            by_key[key].pure_exp006_phase_ms.p90 for key in phase_common  # type: ignore[union-attr]
        )
        mean_ratio = ratio_value(pure_mean, current_mean)
        p90_ratio = ratio_value(pure_p90, current_p90)
    else:
        current_mean = pure_mean = current_p90 = pure_p90 = None
        mean_ratio = p90_ratio = undefined_ratio()
    if len(drift_common) >= 8:
        current_drift_mean = arithmetic_mean(
            by_key[key].current_v2_alias_max_prefix_ms.mean for key in drift_common  # type: ignore[union-attr]
        )
        pure_drift_mean = arithmetic_mean(
            by_key[key].pure_exp006_alias_max_prefix_ms.mean for key in drift_common  # type: ignore[union-attr]
        )
        current_drift_p90 = linear_p90(
            by_key[key].current_v2_alias_max_prefix_ms.p90 for key in drift_common  # type: ignore[union-attr]
        )
        pure_drift_p90 = linear_p90(
            by_key[key].pure_exp006_alias_max_prefix_ms.p90 for key in drift_common  # type: ignore[union-attr]
        )
        drift_mean_ratio = ratio_value(pure_drift_mean, current_drift_mean)
        drift_p90_ratio = ratio_value(pure_drift_p90, current_drift_p90)
    else:
        current_drift_mean = pure_drift_mean = current_drift_p90 = pure_drift_p90 = None
        drift_mean_ratio = drift_p90_ratio = undefined_ratio()
    if len(boundary_common) >= 5:
        current_boundary = arithmetic_mean(
            _ratio_finite_value(by_key[key].current_v2_boundary.f1)  # type: ignore[arg-type]
            for key in boundary_common
        )
        pure_boundary = arithmetic_mean(
            _ratio_finite_value(by_key[key].pure_exp006_boundary.f1)  # type: ignore[arg-type]
            for key in boundary_common
        )
        selected_values = [
            _ratio_finite_value(by_key[key].selected_boundary.f1)
            for key in boundary_common
        ]
        selected_boundary = (
            arithmetic_mean(value for value in selected_values if value is not None)
            if all(value is not None for value in selected_values)
            else None
        )
        boundary_delta = pure_boundary - current_boundary
    else:
        current_boundary = pure_boundary = selected_boundary = boundary_delta = None
    coverage: RateValue | RatioValue
    coverage = rate_value(len(pure), len(current)) if current else undefined_ratio()
    return ScheduleWeakGates(
        pure_mean_phase_ratio=mean_ratio,
        pure_p90_phase_ratio=p90_ratio,
        pure_phase_coverage=coverage,
        current_v2_phase_mean_ms=current_mean,
        pure_exp006_phase_mean_ms=pure_mean,
        current_v2_phase_p90_ms=current_p90,
        pure_exp006_phase_p90_ms=pure_p90,
        current_v2_alias_drift_mean_ms=current_drift_mean,
        pure_exp006_alias_drift_mean_ms=pure_drift_mean,
        current_v2_alias_drift_p90_ms=current_drift_p90,
        pure_exp006_alias_drift_p90_ms=pure_drift_p90,
        alias_max_prefix_drift_mean_ratio=drift_mean_ratio,
        alias_max_prefix_drift_p90_ratio=drift_p90_ratio,
        current_v2_boundary_f1_mean=current_boundary,
        pure_exp006_boundary_f1_mean=pure_boundary,
        selected_boundary_f1_mean=selected_boundary,
        pure_minus_v2_boundary_f1_delta=boundary_delta,
    )


def make_schedule_weak_veto_summary(
    evaluation: ScheduleWeakEvaluation,
    *,
    schedule_arm: str,
    four_arm_stage_summary_sha256: str,
    candidate_global_manifest_sha256: str,
    source_closure_fingerprint_sha256: str,
    source_selection_sha256: str,
) -> dict[str, Any]:
    refs = _weak_ref_dicts(evaluation.weak_row_refs)
    denominators = protocol.validate_schedule_weak_denominators(
        evaluation.denominators.to_dict()
    )
    gates = protocol.validate_schedule_weak_gates(evaluation.gates.to_dict())
    payload = {
        "schema": protocol.SCHEDULE_WEAK_VETO_SUMMARY_SCHEMA,
        "experiment_id": EXP007_EXPERIMENT_ID,
        "stage": SCHEDULE_STAGE,
        "schema_descriptor_sha256": protocol.schema_descriptor_sha256(
            protocol.SCHEDULE_WEAK_VETO_SUMMARY_SCHEMA
        ),
        "schedule_arm": schedule_arm,
        "four_arm_stage_summary_sha256": four_arm_stage_summary_sha256,
        "candidate_global_manifest_sha256": candidate_global_manifest_sha256,
        "source_closure_fingerprint_sha256": source_closure_fingerprint_sha256,
        "source_selection_sha256": source_selection_sha256,
        "selected_row_refs_sha256": protocol.canonical_json_sha256(
            _selected_ref_hash_payload(refs)
        ),
        "row_weak_pairs_sha256": protocol.canonical_json_sha256(
            _row_weak_pair_hash_payload(refs)
        ),
        "weak_row_count": len(refs),
        "weak_row_refs": refs,
        "weak_payloads_sha256": protocol.canonical_json_sha256(refs),
        "denominators": denominators,
        "gates": gates,
        "decision": evaluation.decision,
        "action": evaluation.action,
    }
    summary = protocol.validate_schedule_weak_veto_summary(
        protocol.with_payload_hash(payload, "summary_fingerprint_sha256")
    )
    if summary["selected_row_refs_sha256"] != evaluation.selected_row_refs_sha256:
        raise ValueError("ScheduleWeakVetoSummary selected refs mismatch evaluation")
    if summary["row_weak_pairs_sha256"] != evaluation.row_weak_pairs_sha256:
        raise ValueError("ScheduleWeakVetoSummary row weak pairs mismatch evaluation")
    return summary


def validate_schedule_weak_veto_summary(
    payload: Mapping[str, Any],
    *,
    evaluation: ScheduleWeakEvaluation | None = None,
    weak_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    summary = protocol.validate_schedule_weak_veto_summary(payload)
    if evaluation is not None:
        expected = make_schedule_weak_veto_summary(
            evaluation,
            schedule_arm=summary["schedule_arm"],
            four_arm_stage_summary_sha256=summary["four_arm_stage_summary_sha256"],
            candidate_global_manifest_sha256=summary[
                "candidate_global_manifest_sha256"
            ],
            source_closure_fingerprint_sha256=summary[
                "source_closure_fingerprint_sha256"
            ],
            source_selection_sha256=summary["source_selection_sha256"],
        )
        if summary != expected:
            raise ValueError("ScheduleWeakVetoSummary does not match evaluation")
    if weak_rows is not None:
        _validate_weak_rows_against_refs(
            weak_rows,
            summary["weak_row_refs"],
            expected_stage=SCHEDULE_STAGE,
            expected_schedule_arm=summary["schedule_arm"],
            expected_four_arm_stage_summary_sha256=summary[
                "four_arm_stage_summary_sha256"
            ],
            expected_candidate_global_manifest_sha256=summary[
                "candidate_global_manifest_sha256"
            ],
            expected_source_selection_sha256=summary["source_selection_sha256"],
            context="ScheduleWeakVetoSummary",
        )
    return summary


def _weak_ref_dicts(refs: Sequence[WeakRowRef | Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        protocol.validate_weak_row_ref(
            ref.to_dict() if isinstance(ref, WeakRowRef) else ref
        )
        for ref in refs
    ]


def _selected_ref_hash_payload(refs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "row_index": ref["row_index"],
            "cache_audio_key": ref["cache_audio_key"],
            "prediction_row_sha256": ref["prediction_row_sha256"],
        }
        for ref in refs
    ]


def _row_weak_pair_hash_payload(refs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "row_index": ref["row_index"],
            "cache_audio_key": ref["cache_audio_key"],
            "row_payload_sha256": ref["prediction_row_sha256"],
            "prediction_row_sha256": ref["prediction_row_sha256"],
            "weak_row_payload_sha256": ref["weak_row_payload_sha256"],
        }
        for ref in refs
    ]


def _validate_weak_rows_against_refs(
    weak_rows: Sequence[Mapping[str, Any]],
    refs: Sequence[Mapping[str, Any]],
    *,
    expected_stage: str,
    expected_schedule_arm: str,
    expected_four_arm_stage_summary_sha256: str,
    expected_candidate_global_manifest_sha256: str,
    expected_source_selection_sha256: str,
    context: str,
) -> None:
    if len(weak_rows) != len(refs):
        raise ValueError(f"{context} weak row/ref count mismatch")
    for index, (weak_row, ref) in enumerate(zip(weak_rows, refs, strict=True)):
        row = validate_weak_row(
            weak_row,
            expected_stage=expected_stage,
            expected_schedule_arm=expected_schedule_arm,
            expected_four_arm_stage_summary_sha256=expected_four_arm_stage_summary_sha256,
            expected_candidate_global_manifest_sha256=expected_candidate_global_manifest_sha256,
            expected_source_selection_sha256=expected_source_selection_sha256,
            paired_prediction_row_sha256=ref["prediction_row_sha256"],
        )
        actual_ref = {
            "row_index": row["row_index"],
            "cache_audio_key": row["cache_audio_key"],
            "prediction_row_sha256": row["prediction_row_sha256"],
            "weak_row_payload_sha256": row["weak_row_payload_sha256"],
        }
        if actual_ref != ref:
            raise ValueError(f"{context} weak row ref mismatch at index {index}")


@dataclass(frozen=True)
class Repair80MetricRow:
    weak: WeakMetricRow
    label_stratum: Literal["stable", "jump_candidate", "dense", "ramp_candidate", "ambiguous"]
    source_long_track: bool
    cache_valid: bool
    projection_evaluable: bool
    fallback_reason: str | None
    audio_arm_seconds: float
    overlap_p90_ms: float | None
    candidate_section_count: int | None
    current_v2_segment_count: int | None
    seam_ms: float | None
    replay_schema_source_cache_integrity: bool = True

    def __post_init__(self) -> None:
        if self.weak.stage != REPAIR_STAGE:
            raise ValueError("repair metric row requires stage=repair80")
        if self.label_stratum not in {
            "stable",
            "jump_candidate",
            "dense",
            "ramp_candidate",
            "ambiguous",
        }:
            raise ValueError("invalid label stratum")
        for name in (
            "source_long_track",
            "cache_valid",
            "projection_evaluable",
            "replay_schema_source_cache_integrity",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")
        if self.projection_evaluable and not self.cache_valid:
            raise ValueError("projection evaluable requires valid cache")
        if self.weak.candidate_status == "accepted" and self.fallback_reason is not None:
            raise ValueError("accepted candidate cannot have a fallback reason")
        if self.weak.candidate_status == "accepted":
            _require_index(self.candidate_section_count, "candidate_section_count")
            if (
                self.seam_ms is None
                or isinstance(self.seam_ms, bool)
                or not math.isfinite(float(self.seam_ms))
            ):
                raise ValueError("seam_ms must be finite for an accepted candidate")
            if float(self.seam_ms) < 0:
                raise ValueError("seam_ms must be nonnegative")
        else:
            if self.fallback_reason not in protocol.CANDIDATE_FALLBACK_REASONS:
                raise ValueError("invalid candidate fallback reason")
            if self.candidate_section_count is not None:
                raise ValueError("candidate section count requires accepted candidate")
            if self.seam_ms is not None:
                raise ValueError("seam_ms requires accepted candidate")
        if self.weak.baseline_status == "accepted":
            _require_index(self.current_v2_segment_count, "current_v2_segment_count")
        elif self.current_v2_segment_count is not None:
            raise ValueError("current-v2 segment count requires accepted baseline")
        _finite_nonnegative(self.audio_arm_seconds, "audio_arm_seconds")
        if self.overlap_p90_ms is not None:
            if self.weak.candidate_status != "accepted":
                raise ValueError("overlap p90 requires accepted candidate")
            _finite_nonnegative(self.overlap_p90_ms, "overlap_p90_ms")

    @property
    def cache_audio_key(self) -> str:
        return self.weak.cache_audio_key


@dataclass(frozen=True)
class Repair80Denominators:
    stage_audio_count: int
    stage_audio: AudioSetBinding
    cache_valid_audio: AudioSetBinding
    projection_evaluable_audio: AudioSetBinding
    candidate_accepted_audio: AudioSetBinding
    candidate_fallback_audio: AudioSetBinding
    selected_product_fallback_audio: AudioSetBinding
    baseline_accepted_audio: AudioSetBinding
    product_grid_available_audio: AudioSetBinding
    no_origin_or_path_audio: AudioSetBinding
    resource_cap_fallback_audio: AudioSetBinding
    overlap_available_audio: AudioSetBinding
    current_v2_phase_matched: AudioSetBinding
    pure_exp006_phase_matched: AudioSetBinding
    selected_safety_phase_matched: AudioSetBinding
    phase_common: AudioSetBinding
    stable_pure_paired: AudioSetBinding
    jump_pure_paired: AudioSetBinding
    long_pure_paired: AudioSetBinding
    jump_alias_drift_common: AudioSetBinding
    long_alias_drift_common: AudioSetBinding
    repair_boundary_common: AudioSetBinding

    def to_dict(self) -> dict[str, Any]:
        return _nested_dataclass_dict(self)


@dataclass(frozen=True)
class Repair80Gates:
    candidate_fallback_rate: RateValue
    selected_product_fallback_rate: RateValue
    no_origin_or_path_rate: RateValue
    runtime_seconds: StatsValue
    worker_rss_bytes: StatsValue
    overlap_ms: StatsValue
    stable_section_excess: StatsValue
    pure_mean_phase_ratio: RatioValue
    pure_p90_phase_ratio: RatioValue
    pure_phase_coverage: CoverageValue
    current_v2_phase_mean_ms: float | None
    pure_exp006_phase_mean_ms: float | None
    current_v2_phase_p90_ms: float | None
    pure_exp006_phase_p90_ms: float | None
    stable_phase_mean_ratio: RatioValue
    stable_phase_p90_ratio: RatioValue
    jump_phase_mean_ratio: RatioValue
    current_v2_jump_alias_drift_mean_ms: float | None
    pure_exp006_jump_alias_drift_mean_ms: float | None
    jump_alias_drift_mean_ratio: RatioValue
    current_v2_long_alias_drift_mean_ms: float | None
    pure_exp006_long_alias_drift_mean_ms: float | None
    current_v2_long_alias_drift_p90_ms: float | None
    pure_exp006_long_alias_drift_p90_ms: float | None
    long_alias_drift_mean_ratio: RatioValue
    long_alias_drift_p90_ratio: RatioValue
    current_v2_boundary_f1_mean: float | None
    pure_exp006_boundary_f1_mean: float | None
    selected_boundary_f1_mean: float | None
    pure_minus_v2_boundary_f1_delta: float | None
    every_row_under_180_seconds: bool
    seam_zero: bool
    section_cap_valid: bool
    replay_schema_source_cache_integrity: bool

    def to_dict(self) -> dict[str, Any]:
        return _nested_dataclass_dict(self)


@dataclass(frozen=True)
class Repair80Evaluation:
    denominators: Repair80Denominators
    gates: Repair80Gates
    gate_decisions: Mapping[str, Decision]
    decision: Decision
    action: Literal[
        "write_result_and_next_no_data_card", "stop_ambiguous", "stop_negative"
    ]

    def to_dict(self) -> dict[str, Any]:
        return _nested_dataclass_dict(self)


def evaluate_repair80(
    rows: Sequence[Repair80MetricRow],
    *,
    selected_schedule_arm: str,
    worker_lifetime_rss_bytes: Sequence[int],
) -> Repair80Evaluation:
    if len(rows) != REPAIR_ROW_COUNT:
        raise ValueError("repair80 requires exactly 80 identity rows")
    if tuple(row.weak.row_index for row in rows) != tuple(range(REPAIR_ROW_COUNT)):
        raise ValueError("repair80 rows must be in exact contiguous identity order")
    keys = tuple(row.cache_audio_key for row in rows)
    if len(set(keys)) != REPAIR_ROW_COUNT:
        raise ValueError("repair80 cache_audio_key values must be unique")
    if any(row.weak.schedule_arm != selected_schedule_arm for row in rows):
        raise ValueError("repair80 is selected-winner-only; runner-up rows are forbidden")
    if len(worker_lifetime_rss_bytes) != 4:
        raise ValueError("repair80 requires exactly four worker RSS values")
    rss_values = tuple(_require_index(value, "worker RSS") for value in worker_lifetime_rss_bytes)
    if any(value > WORKER_RSS_CAP_BYTES for value in rss_values):
        raise ValueError("successful repair80 cannot exceed the worker RSS hard cap")
    for row in rows:
        if not row.cache_valid or not row.projection_evaluable:
            raise ValueError("successful repair80 requires valid/evaluable rows")
        if row.audio_arm_seconds >= ROW_TIMEOUT_SECONDS:
            raise ValueError("successful repair80 requires every row below 180 seconds")
        if row.seam_ms is not None and row.seam_ms != 0.0:
            raise ValueError("successful repair80 requires exact zero candidate seam")
        if (
            row.candidate_section_count is not None
            and row.candidate_section_count > 20
        ):
            raise ValueError("successful repair80 requires candidate section_count <= 20")
        if not row.replay_schema_source_cache_integrity:
            raise ValueError("successful repair80 requires replay/schema/source/cache integrity")
    by_key = {row.cache_audio_key: row for row in rows}

    def members(predicate: Any) -> tuple[str, ...]:
        return tuple(key for key in keys if predicate(by_key[key]))

    accepted = members(lambda row: row.weak.candidate_status == "accepted")
    fallback = members(lambda row: row.weak.candidate_status == "tagged_fallback")
    baseline = members(lambda row: row.weak.baseline_status == "accepted")
    product_available = members(lambda row: row.weak.product_grid_available)
    selected_product_fallback = members(
        lambda row: row.weak.candidate_status == "tagged_fallback"
        and row.weak.baseline_status == "accepted"
        and row.weak.selected_status == "accepted"
    )
    no_origin = members(
        lambda row: row.weak.candidate_status == "tagged_fallback"
        and row.fallback_reason in {"no_origin_candidate", "no_local_frontier_path"}
    )
    resource = members(
        lambda row: row.weak.candidate_status == "tagged_fallback"
        and row.fallback_reason == "local_frontier_resource_cap_exceeded"
    )
    overlap = members(
        lambda row: row.weak.candidate_status == "accepted" and row.overlap_p90_ms is not None
    )
    current = members(
        lambda row: row.weak.comparator_state == "available"
        and row.weak.baseline_status == "accepted"
        and row.weak.current_v2_phase_matched
    )
    pure = members(
        lambda row: row.weak.comparator_state == "available"
        and row.weak.candidate_status == "accepted"
        and row.weak.baseline_status == "accepted"
        and row.weak.current_v2_phase_matched
        and row.weak.pure_exp006_phase_matched
    )
    product = members(
        lambda row: row.weak.comparator_state == "available"
        and row.weak.product_grid_available
        and row.weak.baseline_status == "accepted"
        and row.weak.current_v2_phase_matched
        and row.weak.selected_safety_phase_matched
    )
    phase_common = pure
    stable = tuple(key for key in pure if by_key[key].label_stratum == "stable")
    jump = tuple(key for key in pure if by_key[key].label_stratum == "jump_candidate")
    long = tuple(key for key in pure if by_key[key].source_long_track)
    jump_drift = tuple(
        key
        for key in jump
        if _stats_present(by_key[key].weak.current_v2_alias_max_prefix_ms)
        and _stats_present(by_key[key].weak.pure_exp006_alias_max_prefix_ms)
    )
    long_drift = tuple(
        key
        for key in long
        if _stats_present(by_key[key].weak.current_v2_alias_max_prefix_ms)
        and _stats_present(by_key[key].weak.pure_exp006_alias_max_prefix_ms)
    )
    boundary = tuple(
        key
        for key in jump
        if by_key[key].weak.current_v2_boundary.valid_weak_change
        and by_key[key].weak.pure_exp006_boundary.valid_weak_change
        and _ratio_finite_value(by_key[key].weak.current_v2_boundary.f1) is not None
        and _ratio_finite_value(by_key[key].weak.pure_exp006_boundary.f1) is not None
    )
    denominators = Repair80Denominators(
        stage_audio_count=REPAIR_ROW_COUNT,
        stage_audio=audio_set_binding(keys),
        cache_valid_audio=audio_set_binding(keys),
        projection_evaluable_audio=audio_set_binding(keys),
        candidate_accepted_audio=audio_set_binding(accepted),
        candidate_fallback_audio=audio_set_binding(fallback),
        selected_product_fallback_audio=audio_set_binding(selected_product_fallback),
        baseline_accepted_audio=audio_set_binding(baseline),
        product_grid_available_audio=audio_set_binding(product_available),
        no_origin_or_path_audio=audio_set_binding(no_origin),
        resource_cap_fallback_audio=audio_set_binding(resource),
        overlap_available_audio=audio_set_binding(overlap),
        current_v2_phase_matched=audio_set_binding(current),
        pure_exp006_phase_matched=audio_set_binding(pure),
        selected_safety_phase_matched=audio_set_binding(product),
        phase_common=audio_set_binding(phase_common),
        stable_pure_paired=audio_set_binding(stable),
        jump_pure_paired=audio_set_binding(jump),
        long_pure_paired=audio_set_binding(long),
        jump_alias_drift_common=audio_set_binding(jump_drift),
        long_alias_drift_common=audio_set_binding(long_drift),
        repair_boundary_common=audio_set_binding(boundary),
    )
    gates = _repair_gates(
        by_key,
        keys=keys,
        fallback=fallback,
        selected_product_fallback=selected_product_fallback,
        no_origin=no_origin,
        overlap=overlap,
        current=current,
        pure=pure,
        stable=stable,
        jump=jump,
        long=long,
        jump_drift=jump_drift,
        long_drift=long_drift,
        boundary=boundary,
        rss_values=rss_values,
    )
    decisions: dict[str, Decision] = {
        "pure_phase_minimum": "pass" if len(pure) >= 40 else "ambiguous",
        "selected_safety_minimum": "pass" if len(product) >= 40 else "ambiguous",
        "stable_minimum": "pass" if len(stable) >= 5 else "ambiguous",
        "jump_minimum": "pass" if len(jump) >= 15 else "ambiguous",
        "long_minimum": "pass" if len(long) >= 5 else "ambiguous",
        "overlap_minimum": "pass" if len(overlap) >= 20 else "ambiguous",
        "boundary_minimum": "pass" if len(boundary) >= 15 else "ambiguous",
        "pure_mean_phase_ratio": classify_upper_ratio(
            gates.pure_mean_phase_ratio, pass_max=1.05, ambiguous_max=1.10
        ),
        "pure_p90_phase_ratio": classify_upper_ratio(
            gates.pure_p90_phase_ratio, pass_max=1.10, ambiguous_max=1.15
        ),
        "pure_phase_coverage": classify_lower_rate(
            gates.pure_phase_coverage, pass_min=0.95, ambiguous_min=0.90
        ),
        "stable_mean_phase_ratio": classify_upper_ratio(
            gates.stable_phase_mean_ratio, pass_max=1.10, ambiguous_max=1.20
        ),
        "stable_p90_phase_ratio": classify_upper_ratio(
            gates.stable_phase_p90_ratio, pass_max=1.10, ambiguous_max=1.20
        ),
        "jump_phase_mean_ratio": classify_upper_ratio(
            gates.jump_phase_mean_ratio, pass_max=1.05, ambiguous_max=1.15
        ),
        "jump_alias_drift_mean_ratio": classify_upper_ratio(
            gates.jump_alias_drift_mean_ratio, pass_max=0.90, ambiguous_max=1.15
        ),
        "long_alias_drift_mean_ratio": classify_upper_ratio(
            gates.long_alias_drift_mean_ratio, pass_max=1.15, ambiguous_max=1.30
        ),
        "long_alias_drift_p90_ratio": classify_upper_ratio(
            gates.long_alias_drift_p90_ratio, pass_max=1.15, ambiguous_max=1.30
        ),
        "boundary_f1_delta": classify_boundary_delta(
            gates.pure_minus_v2_boundary_f1_delta
        ),
        "candidate_fallback_rate": classify_upper_value(
            gates.candidate_fallback_rate.value, pass_max=0.05, ambiguous_max=0.10
        ),
        "no_origin_or_path_rate": classify_upper_value(
            gates.no_origin_or_path_rate.value, pass_max=0.03, ambiguous_max=0.05
        ),
        "runtime_p90": classify_upper_value(
            gates.runtime_seconds.p90, pass_max=30.0, ambiguous_max=60.0
        ),
        "overlap_p90": (
            classify_upper_value(gates.overlap_ms.p90, pass_max=45.0, ambiguous_max=90.0)
            if len(overlap) >= 20
            else "ambiguous"
        ),
        "stable_section_excess": (
            "negative" if (gates.stable_section_excess.maximum or 0.0) > 1.0 else "pass"
        ),
    }
    decision = combine_decisions(decisions.values())
    action = {
        "pass": "write_result_and_next_no_data_card",
        "ambiguous": "stop_ambiguous",
        "negative": "stop_negative",
    }[decision]
    return Repair80Evaluation(denominators, gates, decisions, decision, action)


def _repair_gates(
    by_key: Mapping[str, Repair80MetricRow],
    *,
    keys: Sequence[str],
    fallback: Sequence[str],
    selected_product_fallback: Sequence[str],
    no_origin: Sequence[str],
    overlap: Sequence[str],
    current: Sequence[str],
    pure: Sequence[str],
    stable: Sequence[str],
    jump: Sequence[str],
    long: Sequence[str],
    jump_drift: Sequence[str],
    long_drift: Sequence[str],
    boundary: Sequence[str],
    rss_values: Sequence[int],
) -> Repair80Gates:
    (
        pure_mean,
        pure_p90,
        current_phase_mean,
        pure_phase_mean,
        current_phase_p90,
        pure_phase_p90,
    ) = _phase_reducers(by_key, pure, minimum=40)
    stable_mean, stable_p90, _, _, _, _ = _phase_reducers(
        by_key, stable, minimum=5
    )
    jump_mean, _, _, _, _, _ = _phase_reducers(by_key, jump, minimum=15)
    (
        jump_drift_mean,
        _,
        current_jump_drift_mean,
        pure_jump_drift_mean,
        _,
        _,
    ) = _drift_reducers(by_key, jump_drift, minimum=15)
    (
        long_drift_mean,
        long_drift_p90,
        current_long_drift_mean,
        pure_long_drift_mean,
        current_long_drift_p90,
        pure_long_drift_p90,
    ) = _drift_reducers(by_key, long_drift, minimum=5)
    if len(boundary) >= 15:
        current_boundary = arithmetic_mean(
            _ratio_finite_value(by_key[key].weak.current_v2_boundary.f1)  # type: ignore[arg-type]
            for key in boundary
        )
        pure_boundary = arithmetic_mean(
            _ratio_finite_value(by_key[key].weak.pure_exp006_boundary.f1)  # type: ignore[arg-type]
            for key in boundary
        )
        selected_values = [
            _ratio_finite_value(by_key[key].weak.selected_boundary.f1) for key in boundary
        ]
        if not all(value is not None for value in selected_values):
            raise ValueError(
                "repair boundary common requires finite selected boundary F1 reporting"
            )
        selected_boundary = arithmetic_mean(
            value for value in selected_values if value is not None
        )
        boundary_delta = pure_boundary - current_boundary
    else:
        current_boundary = pure_boundary = selected_boundary = boundary_delta = None
    coverage: CoverageValue = (
        rate_value(len(pure), len(current)) if current else undefined_ratio()
    )
    return Repair80Gates(
        candidate_fallback_rate=rate_value(len(fallback), REPAIR_ROW_COUNT),
        selected_product_fallback_rate=rate_value(
            len(selected_product_fallback), REPAIR_ROW_COUNT
        ),
        no_origin_or_path_rate=rate_value(len(no_origin), REPAIR_ROW_COUNT),
        runtime_seconds=stats_value(by_key[key].audio_arm_seconds for key in keys),
        worker_rss_bytes=stats_value(rss_values),
        overlap_ms=stats_value(by_key[key].overlap_p90_ms for key in overlap),  # type: ignore[arg-type]
        stable_section_excess=stats_value(
            max(
                0,
                _present_count(
                    by_key[key].candidate_section_count,
                    "candidate_section_count",
                )
                - _present_count(
                    by_key[key].current_v2_segment_count,
                    "current_v2_segment_count",
                ),
            )
            for key in stable
        ),
        pure_mean_phase_ratio=pure_mean,
        pure_p90_phase_ratio=pure_p90,
        pure_phase_coverage=coverage,
        current_v2_phase_mean_ms=current_phase_mean,
        pure_exp006_phase_mean_ms=pure_phase_mean,
        current_v2_phase_p90_ms=current_phase_p90,
        pure_exp006_phase_p90_ms=pure_phase_p90,
        stable_phase_mean_ratio=stable_mean,
        stable_phase_p90_ratio=stable_p90,
        jump_phase_mean_ratio=jump_mean,
        current_v2_jump_alias_drift_mean_ms=current_jump_drift_mean,
        pure_exp006_jump_alias_drift_mean_ms=pure_jump_drift_mean,
        jump_alias_drift_mean_ratio=jump_drift_mean,
        current_v2_long_alias_drift_mean_ms=current_long_drift_mean,
        pure_exp006_long_alias_drift_mean_ms=pure_long_drift_mean,
        current_v2_long_alias_drift_p90_ms=current_long_drift_p90,
        pure_exp006_long_alias_drift_p90_ms=pure_long_drift_p90,
        long_alias_drift_mean_ratio=long_drift_mean,
        long_alias_drift_p90_ratio=long_drift_p90,
        current_v2_boundary_f1_mean=current_boundary,
        pure_exp006_boundary_f1_mean=pure_boundary,
        selected_boundary_f1_mean=selected_boundary,
        pure_minus_v2_boundary_f1_delta=boundary_delta,
        every_row_under_180_seconds=True,
        seam_zero=True,
        section_cap_valid=True,
        replay_schema_source_cache_integrity=True,
    )


def make_repair80_summary(
    evaluation: Repair80Evaluation,
    *,
    schedule_arm: str,
    repair80_input_binding: Mapping[str, Any],
    schedule_weak_veto_outcome: Mapping[str, Any],
    run_config_fingerprint_sha256: str,
    candidate_reference_manifest: Mapping[str, Any],
    artifact_root: Any,
    row_refs: Sequence[Mapping[str, Any]],
    weak_row_refs: Sequence[WeakRowRef | Mapping[str, Any]],
    aggregate_wall_seconds: float,
    worker_lifetime_rss_bytes: Sequence[int],
) -> dict[str, Any]:
    binding = protocol.validate_repair80_input_binding(repair80_input_binding)
    weak_outcome = protocol.validate_schedule_weak_veto_outcome(
        schedule_weak_veto_outcome
    )
    weak_summary = _require_pass_weak_outcome(weak_outcome)
    if schedule_arm != weak_summary["schedule_arm"]:
        raise ValueError("Repair80Summary schedule arm must match weak winner")
    rows = _completed_ref_dicts(row_refs)
    weak_refs = _weak_ref_dicts(weak_row_refs)
    reference_manifest = _validate_repair80_reference_manifest(
        candidate_reference_manifest,
        artifact_root=artifact_root,
        binding=binding,
        schedule_arm=schedule_arm,
        source_closure_fingerprint_sha256=weak_summary[
            "source_closure_fingerprint_sha256"
        ],
        row_refs=rows,
    )
    reference_manifest_sha256 = protocol.object_complete_sha256(reference_manifest)
    denominators = protocol.validate_repair80_denominators(
        evaluation.denominators.to_dict()
    )
    gates = _repair80_protocol_gates(evaluation.gates)
    rss_values = tuple(_require_index(value, "worker RSS") for value in worker_lifetime_rss_bytes)
    if protocol.validate_stats_value(stats_value(rss_values).to_dict()) != gates["worker_rss_bytes"]:
        raise ValueError("Repair80Summary RSS summary does not match evaluation")
    runtime_summary = protocol.validate_runtime_summary(
        {
            "row_seconds": evaluation.gates.runtime_seconds.to_dict(),
            "aggregate_wall_seconds": aggregate_wall_seconds,
        }
    )
    rss_summary = protocol.validate_rss_summary(
        {
            "worker_count": 4,
            "worker_lifetime_bytes": list(rss_values),
            "arm_max_worker_bytes": max(rss_values),
        }
    )
    payload = {
        "schema": protocol.REPAIR80_SUMMARY_SCHEMA,
        "experiment_id": EXP007_EXPERIMENT_ID,
        "stage": REPAIR_STAGE,
        "schema_descriptor_sha256": protocol.schema_descriptor_sha256(
            protocol.REPAIR80_SUMMARY_SCHEMA
        ),
        "schedule_arm": schedule_arm,
        "four_arm_stage_summary_sha256": binding["four_arm_stage_summary_sha256"],
        "candidate_global_manifest_sha256": binding[
            "candidate_global_manifest_sha256"
        ],
        "source_selection_sha256": binding["source_selection_sha256"],
        "schedule_weak_veto_outcome_sha256": protocol.object_complete_sha256(
            weak_outcome
        ),
        "run_config_fingerprint_sha256": run_config_fingerprint_sha256,
        "source_closure_fingerprint_sha256": weak_summary[
            "source_closure_fingerprint_sha256"
        ],
        "repair_input_binding_sha256": binding["binding_fingerprint_sha256"],
        "repair_identity_source": binding["identity_source"],
        "repair_label_source": binding["label_source"],
        "candidate_reference_manifest_sha256": reference_manifest_sha256,
        "row_count": len(rows),
        "row_refs": rows,
        "row_payloads_sha256": protocol.canonical_json_sha256(rows),
        "weak_row_count": len(weak_refs),
        "weak_row_refs": weak_refs,
        "weak_payloads_sha256": protocol.canonical_json_sha256(weak_refs),
        "row_weak_pairs_sha256": protocol.canonical_json_sha256(
            _repair_row_weak_pair_hash_payload(rows, weak_refs)
        ),
        "denominators": denominators,
        "gates": gates,
        "decision": evaluation.decision,
        "action": evaluation.action,
        "runtime_summary": runtime_summary,
        "rss_summary": rss_summary,
    }
    return validate_repair80_summary(
        protocol.with_payload_hash(payload, "summary_fingerprint_sha256"),
        evaluation=evaluation,
        repair80_input_binding=binding,
        schedule_weak_veto_outcome=weak_outcome,
        candidate_reference_manifest=reference_manifest,
        artifact_root=artifact_root,
    )


def validate_repair80_summary(
    payload: Mapping[str, Any],
    *,
    candidate_reference_manifest: Mapping[str, Any],
    artifact_root: Any,
    evaluation: Repair80Evaluation | None = None,
    repair80_input_binding: Mapping[str, Any] | None = None,
    schedule_weak_veto_outcome: Mapping[str, Any] | None = None,
    weak_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    summary = protocol.validate_repair80_summary(payload)
    if evaluation is not None:
        if summary["denominators"] != protocol.validate_repair80_denominators(
            evaluation.denominators.to_dict()
        ):
            raise ValueError("Repair80Summary denominators do not match evaluation")
        if summary["gates"] != _repair80_protocol_gates(evaluation.gates):
            raise ValueError("Repair80Summary gates do not match evaluation")
        if summary["decision"] != evaluation.decision or summary["action"] != evaluation.action:
            raise ValueError("Repair80Summary decision/action mismatch evaluation")
    if repair80_input_binding is not None:
        binding = protocol.validate_repair80_input_binding(repair80_input_binding)
        if summary["repair_input_binding_sha256"] != binding["binding_fingerprint_sha256"]:
            raise ValueError("Repair80Summary repair input binding SHA mismatch")
        for summary_name, binding_name in (
            ("four_arm_stage_summary_sha256", "four_arm_stage_summary_sha256"),
            ("candidate_global_manifest_sha256", "candidate_global_manifest_sha256"),
            ("source_selection_sha256", "source_selection_sha256"),
            (
                "schedule_weak_veto_outcome_sha256",
                "schedule_weak_veto_outcome_sha256",
            ),
        ):
            if summary[summary_name] != binding[binding_name]:
                raise ValueError(f"Repair80Summary {summary_name} binding mismatch")
        if summary["repair_identity_source"] != binding["identity_source"]:
            raise ValueError("Repair80Summary identity source mismatch")
        if summary["repair_label_source"] != binding["label_source"]:
            raise ValueError("Repair80Summary label source mismatch")
    if schedule_weak_veto_outcome is not None:
        weak_outcome = protocol.validate_schedule_weak_veto_outcome(
            schedule_weak_veto_outcome
        )
        weak_summary = _require_pass_weak_outcome(weak_outcome)
        if summary["schedule_weak_veto_outcome_sha256"] != protocol.object_complete_sha256(
            weak_outcome
        ):
            raise ValueError("Repair80Summary weak outcome SHA mismatch")
        for field_name in (
            "four_arm_stage_summary_sha256",
            "candidate_global_manifest_sha256",
            "source_selection_sha256",
            "source_closure_fingerprint_sha256",
        ):
            if summary[field_name] != weak_summary[field_name]:
                raise ValueError(f"Repair80Summary weak {field_name} mismatch")
        if summary["schedule_arm"] != weak_summary["schedule_arm"]:
            raise ValueError("Repair80Summary weak schedule arm mismatch")
    reference_binding = (
        protocol.validate_repair80_input_binding(repair80_input_binding)
        if repair80_input_binding is not None
        else None
    )
    reference_source_sha = summary["source_closure_fingerprint_sha256"]
    _validate_repair80_reference_manifest(
        candidate_reference_manifest,
        artifact_root=artifact_root,
        binding=reference_binding,
        schedule_arm=summary["schedule_arm"],
        source_closure_fingerprint_sha256=reference_source_sha,
        row_refs=summary["row_refs"],
        expected_manifest_sha256=summary["candidate_reference_manifest_sha256"],
    )
    if weak_rows is not None:
        _validate_weak_rows_against_refs(
            weak_rows,
            summary["weak_row_refs"],
            expected_stage=REPAIR_STAGE,
            expected_schedule_arm=summary["schedule_arm"],
            expected_four_arm_stage_summary_sha256=summary[
                "four_arm_stage_summary_sha256"
            ],
            expected_candidate_global_manifest_sha256=summary[
                "candidate_global_manifest_sha256"
            ],
            expected_source_selection_sha256=summary["source_selection_sha256"],
            context="Repair80Summary",
        )
    return summary


def make_repair80_summary_from_rows(
    *,
    schedule_arm: str,
    repair_metric_rows: Sequence[Repair80MetricRow],
    repair80_input_binding: Mapping[str, Any],
    repair80_identity_source_artifact: bytes,
    repair80_label_source_artifact: bytes,
    repair80_identity_rows: Sequence[Mapping[str, Any]],
    repair80_label_rows: Sequence[Mapping[str, Any]],
    schedule_weak_veto_outcome: Mapping[str, Any],
    run_config_fingerprint_sha256: str,
    candidate_reference_manifest: Mapping[str, Any],
    artifact_root: Any,
    prediction_rows: Sequence[Mapping[str, Any]],
    weak_rows: Sequence[Mapping[str, Any]],
    aggregate_wall_seconds: float,
    worker_lifetime_rss_bytes: Sequence[int],
) -> dict[str, Any]:
    binding = protocol.validate_repair80_input_binding(repair80_input_binding)
    weak_outcome = protocol.validate_schedule_weak_veto_outcome(
        schedule_weak_veto_outcome
    )
    weak_summary = _require_pass_weak_outcome(weak_outcome)
    if schedule_arm != weak_summary["schedule_arm"]:
        raise ValueError("Repair80Summary schedule arm must match weak winner")
    validated_identities = protocol.validate_repair80_identity_sources_for_execution(
        repair80_identity_source_artifact=repair80_identity_source_artifact,
        repair80_label_source_artifact=repair80_label_source_artifact,
        repair80_identity_rows=repair80_identity_rows,
        repair80_label_rows=repair80_label_rows,
        identity_source=binding["identity_source"],
        label_source=binding["label_source"],
    )
    reference_manifest = _validate_repair80_reference_manifest(
        candidate_reference_manifest,
        artifact_root=artifact_root,
        binding=binding,
        schedule_arm=schedule_arm,
        source_closure_fingerprint_sha256=weak_summary[
            "source_closure_fingerprint_sha256"
        ],
        row_refs=None,
    )
    _validate_repair_metric_rows_against_prediction_and_weak_rows(
        repair_metric_rows,
        prediction_rows=prediction_rows,
        weak_rows=weak_rows,
        schedule_arm=schedule_arm,
        source_closure_fingerprint_sha256=weak_summary[
            "source_closure_fingerprint_sha256"
        ],
        run_config_fingerprint_sha256=run_config_fingerprint_sha256,
        binding=binding,
        validated_identities=validated_identities,
    )
    evaluation = evaluate_repair80(
        repair_metric_rows,
        selected_schedule_arm=schedule_arm,
        worker_lifetime_rss_bytes=worker_lifetime_rss_bytes,
    )
    rows = _completed_refs_from_prediction_rows(
        prediction_rows,
        reference_manifest=reference_manifest,
        artifact_root=artifact_root,
        binding=binding,
        schedule_arm=schedule_arm,
        source_closure_fingerprint_sha256=weak_summary[
            "source_closure_fingerprint_sha256"
        ],
        run_config_fingerprint_sha256=run_config_fingerprint_sha256,
    )
    weak_refs = [
        protocol.validate_weak_row_ref(weak_row_ref_from_row(row).to_dict())
        for row in weak_rows
    ]
    return make_repair80_summary(
        evaluation,
        schedule_arm=schedule_arm,
        repair80_input_binding=binding,
        schedule_weak_veto_outcome=weak_outcome,
        run_config_fingerprint_sha256=run_config_fingerprint_sha256,
        candidate_reference_manifest=reference_manifest,
        artifact_root=artifact_root,
        row_refs=rows,
        weak_row_refs=weak_refs,
        aggregate_wall_seconds=aggregate_wall_seconds,
        worker_lifetime_rss_bytes=worker_lifetime_rss_bytes,
    )


def validate_repair80_summary_authoritatively(
    payload: Mapping[str, Any],
    *,
    repair_metric_rows: Sequence[Repair80MetricRow],
    repair80_input_binding: Mapping[str, Any],
    repair80_identity_source_artifact: bytes,
    repair80_label_source_artifact: bytes,
    repair80_identity_rows: Sequence[Mapping[str, Any]],
    repair80_label_rows: Sequence[Mapping[str, Any]],
    schedule_weak_veto_outcome: Mapping[str, Any],
    candidate_reference_manifest: Mapping[str, Any],
    artifact_root: Any,
    prediction_rows: Sequence[Mapping[str, Any]],
    weak_rows: Sequence[Mapping[str, Any]],
    aggregate_wall_seconds: float,
    worker_lifetime_rss_bytes: Sequence[int],
) -> dict[str, Any]:
    shape = protocol.validate_repair80_summary(payload)
    expected = make_repair80_summary_from_rows(
        schedule_arm=shape["schedule_arm"],
        repair_metric_rows=repair_metric_rows,
        repair80_input_binding=repair80_input_binding,
        repair80_identity_source_artifact=repair80_identity_source_artifact,
        repair80_label_source_artifact=repair80_label_source_artifact,
        repair80_identity_rows=repair80_identity_rows,
        repair80_label_rows=repair80_label_rows,
        schedule_weak_veto_outcome=schedule_weak_veto_outcome,
        run_config_fingerprint_sha256=shape["run_config_fingerprint_sha256"],
        candidate_reference_manifest=candidate_reference_manifest,
        artifact_root=artifact_root,
        prediction_rows=prediction_rows,
        weak_rows=weak_rows,
        aggregate_wall_seconds=aggregate_wall_seconds,
        worker_lifetime_rss_bytes=worker_lifetime_rss_bytes,
    )
    summary = validate_repair80_summary(
        payload,
        repair80_input_binding=repair80_input_binding,
        schedule_weak_veto_outcome=schedule_weak_veto_outcome,
        candidate_reference_manifest=candidate_reference_manifest,
        artifact_root=artifact_root,
        weak_rows=weak_rows,
    )
    if summary != expected:
        raise ValueError("Repair80Summary does not match authoritative recomputation")
    return summary


def _completed_ref_dicts(refs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [protocol.validate_completed_row_ref(ref) for ref in refs]


def _validate_repair80_reference_manifest(
    candidate_reference_manifest: Mapping[str, Any],
    *,
    artifact_root: Any,
    binding: Mapping[str, Any] | None,
    schedule_arm: str,
    source_closure_fingerprint_sha256: str,
    row_refs: Sequence[Mapping[str, Any]] | None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    artifacts = _artifacts_module()
    manifest = artifacts.validate_candidate_reference_manifest(
        candidate_reference_manifest,
        root=artifact_root,
    )
    if manifest["stage"] != REPAIR_STAGE:
        raise ValueError("Repair80Summary candidate reference manifest stage mismatch")
    if manifest["reference_arm"] != schedule_arm:
        raise ValueError("Repair80Summary candidate reference manifest arm mismatch")
    if manifest["row_count"] != REPAIR_ROW_COUNT:
        raise ValueError("Repair80Summary candidate reference manifest row count mismatch")
    if manifest["source_closure_fingerprint_sha256"] != source_closure_fingerprint_sha256:
        raise ValueError("Repair80Summary candidate reference manifest source mismatch")
    if binding is not None:
        expected_input_sha = binding["binding_fingerprint_sha256"]
        if manifest["input_manifest_sha256"] != expected_input_sha:
            raise ValueError("Repair80Summary candidate reference manifest input mismatch")
    actual_sha = protocol.object_complete_sha256(manifest)
    if expected_manifest_sha256 is not None and expected_manifest_sha256 != actual_sha:
        raise ValueError("Repair80Summary candidate reference manifest SHA mismatch")
    if row_refs is not None:
        _validate_repair80_reference_entries_against_row_refs(manifest, row_refs)
    return manifest


def _validate_repair80_reference_entries_against_row_refs(
    reference_manifest: Mapping[str, Any],
    row_refs: Sequence[Mapping[str, Any]],
) -> None:
    refs = [protocol.validate_completed_row_ref(ref) for ref in row_refs]
    if len(refs) != REPAIR_ROW_COUNT:
        raise ValueError("Repair80Summary row refs must contain exactly 80 rows")
    entries = reference_manifest["entries"]
    if len(entries) != len(refs):
        raise ValueError("Repair80Summary manifest row count mismatch")
    for expected_index, (entry, row_ref) in enumerate(zip(entries, refs, strict=True)):
        if entry["row_index"] != expected_index or row_ref["row_index"] != expected_index:
            raise ValueError("Repair80Summary manifest row order mismatch")
        if entry["cache_audio_key"] != row_ref["cache_audio_key"]:
            raise ValueError("Repair80Summary manifest cache key mismatch")
        if (
            row_ref["candidate_reference_entry_payload_sha256"]
            != entry["entry_payload_sha256"]
        ):
            raise ValueError("Repair80Summary manifest reference entry SHA mismatch")


def _completed_refs_from_prediction_rows(
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    reference_manifest: Mapping[str, Any],
    artifact_root: Any,
    binding: Mapping[str, Any],
    schedule_arm: str,
    source_closure_fingerprint_sha256: str,
    run_config_fingerprint_sha256: str,
) -> list[dict[str, Any]]:
    if len(prediction_rows) != REPAIR_ROW_COUNT:
        raise ValueError("Repair80Summary authoritative rows must contain 80 rows")
    artifacts = _artifacts_module()
    rows = [protocol.validate_row_result(row) for row in prediction_rows]
    refs: list[dict[str, Any]] = []
    for expected_index, (row, entry) in enumerate(
        zip(rows, reference_manifest["entries"], strict=True)
    ):
        if row["row_index"] != expected_index:
            raise ValueError("Repair80Summary prediction rows must be ordered")
        if row["stage"] != REPAIR_STAGE or row["schedule_arm"] != schedule_arm:
            raise ValueError("Repair80Summary prediction row stage/arm mismatch")
        if row["input_manifest_sha256"] != binding["binding_fingerprint_sha256"]:
            raise ValueError("Repair80Summary prediction row input mismatch")
        if row["source_closure_fingerprint_sha256"] != source_closure_fingerprint_sha256:
            raise ValueError("Repair80Summary prediction row source mismatch")
        if row["run_config_fingerprint_sha256"] != run_config_fingerprint_sha256:
            raise ValueError("Repair80Summary prediction row run config mismatch")
        bundle = artifacts.read_candidate_reference_row_bundle(
            artifact_root,
            entry["bundle_relative_path"],
        )
        if row != bundle["row"]:
            raise ValueError("Repair80Summary prediction row does not match reference bundle")
        refs.append(
            protocol.make_completed_row_ref(
                row_index=row["row_index"],
                cache_audio_key=row["cache_audio_key"],
                identity_payload_sha256=row["identity_payload_sha256"],
                row_payload_sha256=row["row_payload_sha256"],
                candidate_reference_entry_payload_sha256=entry[
                    "entry_payload_sha256"
                ],
            )
        )
    return refs


def _validate_repair_metric_rows_against_prediction_and_weak_rows(
    repair_metric_rows: Sequence[Repair80MetricRow],
    *,
    prediction_rows: Sequence[Mapping[str, Any]],
    weak_rows: Sequence[Mapping[str, Any]],
    schedule_arm: str,
    source_closure_fingerprint_sha256: str,
    run_config_fingerprint_sha256: str,
    binding: Mapping[str, Any],
    validated_identities: Sequence[Mapping[str, Any]],
) -> None:
    if (
        len(repair_metric_rows) != REPAIR_ROW_COUNT
        or len(prediction_rows) != REPAIR_ROW_COUNT
        or len(weak_rows) != REPAIR_ROW_COUNT
        or len(validated_identities) != REPAIR_ROW_COUNT
    ):
        raise ValueError("Repair80Summary authoritative inputs must contain 80 rows")
    validated_prediction_rows = [
        protocol.validate_row_result(row) for row in prediction_rows
    ]
    validated_weak_rows = [
        validate_weak_row(
            row,
            expected_stage=REPAIR_STAGE,
            expected_schedule_arm=schedule_arm,
            expected_four_arm_stage_summary_sha256=binding[
                "four_arm_stage_summary_sha256"
            ],
            expected_candidate_global_manifest_sha256=binding[
                "candidate_global_manifest_sha256"
            ],
            expected_source_selection_sha256=binding["source_selection_sha256"],
        )
        for row in weak_rows
    ]
    identities = [protocol.validate_identity(row) for row in validated_identities]
    for expected_index, (metric, prediction, weak, identity) in enumerate(
        zip(
            repair_metric_rows,
            validated_prediction_rows,
            validated_weak_rows,
            identities,
            strict=True,
        )
    ):
        if not isinstance(metric, Repair80MetricRow):
            raise ValueError("Repair80Summary repair metric rows must be Repair80MetricRow")
        if metric.weak.stage != REPAIR_STAGE or metric.weak.schedule_arm != schedule_arm:
            raise ValueError("Repair80Summary metric weak row stage/arm mismatch")
        if (
            metric.weak.row_index != expected_index
            or prediction["row_index"] != expected_index
            or weak["row_index"] != expected_index
            or identity["row_index"] != expected_index
        ):
            raise ValueError("Repair80Summary authoritative row order mismatch")
        if prediction["stage"] != REPAIR_STAGE or prediction["schedule_arm"] != schedule_arm:
            raise ValueError("Repair80Summary prediction row stage/arm mismatch")
        if prediction["input_manifest_sha256"] != binding["binding_fingerprint_sha256"]:
            raise ValueError("Repair80Summary prediction row input mismatch")
        if prediction["source_closure_fingerprint_sha256"] != source_closure_fingerprint_sha256:
            raise ValueError("Repair80Summary prediction row source mismatch")
        if prediction["run_config_fingerprint_sha256"] != run_config_fingerprint_sha256:
            raise ValueError("Repair80Summary prediction row run config mismatch")
        _validate_identity_terminal_binding(
            metric,
            prediction=prediction,
            identity=identity,
        )
        _validate_metric_row_identity(metric, prediction=prediction, weak=weak)
        _validate_metric_row_candidate_flags(metric, prediction=prediction)
        _validate_metric_row_weak_payload(metric, weak=weak)
        _validate_metric_row_runtime_and_guards(metric, prediction=prediction)


def _validate_metric_row_identity(
    metric: Repair80MetricRow,
    *,
    prediction: Mapping[str, Any],
    weak: Mapping[str, Any],
) -> None:
    for field_name in ("cache_audio_key", "audio_group_key"):
        if getattr(metric.weak, field_name) != prediction[field_name]:
            raise ValueError(f"Repair80Summary metric/prediction {field_name} mismatch")
        if getattr(metric.weak, field_name) != weak[field_name]:
            raise ValueError(f"Repair80Summary metric/weak {field_name} mismatch")
    if metric.weak.prediction_row_sha256 != prediction["row_payload_sha256"]:
        raise ValueError("Repair80Summary metric prediction row SHA mismatch")
    if weak["prediction_row_sha256"] != prediction["row_payload_sha256"]:
        raise ValueError("Repair80Summary weak prediction row SHA mismatch")
    if metric.weak.weak_row_payload_sha256 != weak["weak_row_payload_sha256"]:
        raise ValueError("Repair80Summary metric weak row SHA mismatch")


def _validate_identity_terminal_binding(
    metric: Repair80MetricRow,
    *,
    prediction: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    for field_name in ("cache_audio_key", "audio_group_key", "identity_payload_sha256"):
        if identity[field_name] != prediction[field_name]:
            raise ValueError(
                f"Repair80Summary identity/prediction {field_name} mismatch"
            )
    if metric.label_stratum != identity["label_stratum"]:
        raise ValueError("Repair80Summary metric/identity label_stratum mismatch")
    if metric.source_long_track is not identity["source_long_track"]:
        raise ValueError("Repair80Summary metric/identity source_long_track mismatch")


def _validate_metric_row_candidate_flags(
    metric: Repair80MetricRow,
    *,
    prediction: Mapping[str, Any],
) -> None:
    flags = prediction["denominator_flags"]
    candidate_method = prediction["methods"]["candidate"]
    baseline_method = prediction["methods"]["baseline"]
    selected_method = prediction["methods"]["selected"]
    expected_candidate_status = (
        "accepted" if flags["candidate_accepted"] else "tagged_fallback"
    )
    expected_baseline_status = (
        "accepted" if flags["baseline_accepted"] else "unavailable"
    )
    expected_selected_status = (
        "accepted" if flags["product_grid_available"] else "unavailable"
    )
    if metric.weak.candidate_status != expected_candidate_status:
        raise ValueError("Repair80Summary metric candidate status mismatch")
    if metric.weak.baseline_status != expected_baseline_status:
        raise ValueError("Repair80Summary metric baseline status mismatch")
    if metric.weak.selected_status != expected_selected_status:
        raise ValueError("Repair80Summary metric selected status mismatch")
    if metric.cache_valid != prediction["cache_identity"]["exists"]:
        raise ValueError("Repair80Summary metric cache validity mismatch")
    if metric.cache_valid != flags["cache_valid"]:
        raise ValueError("Repair80Summary metric cache flag mismatch")
    if metric.projection_evaluable != flags["projection_evaluable"]:
        raise ValueError("Repair80Summary metric projection flag mismatch")
    if metric.weak.product_grid_available != flags["product_grid_available"]:
        raise ValueError("Repair80Summary metric product-grid flag mismatch")
    if metric.overlap_p90_ms is not None and not flags["overlap_available"]:
        raise ValueError("Repair80Summary metric overlap flag mismatch")
    if metric.overlap_p90_ms is None and flags["overlap_available"]:
        raise ValueError("Repair80Summary metric overlap availability mismatch")
    overlap = prediction["diagnostics_summary"]["overlap"]
    if metric.overlap_p90_ms != overlap["p90_ms"]:
        raise ValueError("Repair80Summary metric overlap p90 mismatch")
    if candidate_method["status"] != metric.weak.candidate_status:
        raise ValueError("Repair80Summary candidate method status mismatch")
    if baseline_method["status"] != metric.weak.baseline_status:
        raise ValueError("Repair80Summary baseline method status mismatch")
    if selected_method["status"] != metric.weak.selected_status:
        raise ValueError("Repair80Summary selected method status mismatch")
    if metric.fallback_reason != candidate_method["reason"]:
        raise ValueError("Repair80Summary fallback reason mismatch")


def _validate_metric_row_weak_payload(
    metric: Repair80MetricRow,
    *,
    weak: Mapping[str, Any],
) -> None:
    comparator_state = weak["comparator_availability"]["state"]
    if metric.weak.comparator_state != comparator_state:
        raise ValueError("Repair80Summary comparator state mismatch")
    for name in (
        "current_v2_phase_matched",
        "pure_exp006_phase_matched",
        "selected_safety_phase_matched",
    ):
        if getattr(metric.weak, name) != weak[name]:
            raise ValueError(f"Repair80Summary weak {name} mismatch")
    if _stats_dict(metric.weak.current_v2_phase_ms) != weak["phase_metrics_summary"]["current_v2_ms"]:
        raise ValueError("Repair80Summary current-v2 phase metric mismatch")
    if _stats_dict(metric.weak.pure_exp006_phase_ms) != weak["phase_metrics_summary"]["pure_exp006_ms"]:
        raise ValueError("Repair80Summary pure Exp006 phase metric mismatch")
    if _stats_dict(metric.weak.product_phase_ms) != weak["phase_metrics_summary"]["product_ms"]:
        raise ValueError("Repair80Summary product phase metric mismatch")
    if (
        _stats_dict(metric.weak.current_v2_alias_max_prefix_ms)
        != weak["drift_metrics_summary"]["current_v2_alias_max_prefix_ms"]
    ):
        raise ValueError("Repair80Summary current-v2 drift metric mismatch")
    if (
        _stats_dict(metric.weak.pure_exp006_alias_max_prefix_ms)
        != weak["drift_metrics_summary"]["pure_exp006_alias_max_prefix_ms"]
    ):
        raise ValueError("Repair80Summary pure Exp006 drift metric mismatch")
    if (
        _stats_dict(metric.weak.product_alias_max_prefix_ms)
        != weak["drift_metrics_summary"]["product_alias_max_prefix_ms"]
    ):
        raise ValueError("Repair80Summary product drift metric mismatch")
    for metric_name, weak_name in (
        ("current_v2_boundary", "current_v2_boundary_summary"),
        ("pure_exp006_boundary", "pure_exp006_boundary_summary"),
        ("selected_boundary", "selected_boundary_summary"),
    ):
        expected = _boundary_evidence_from_summary(
            weak[weak_name],
            context=f"Repair80Summary.{weak_name}",
        )
        if getattr(metric.weak, metric_name) != expected:
            raise ValueError(f"Repair80Summary {metric_name} boundary metric mismatch")


def _validate_metric_row_runtime_and_guards(
    metric: Repair80MetricRow,
    *,
    prediction: Mapping[str, Any],
) -> None:
    runtime = prediction["runtime"]
    if metric.audio_arm_seconds != runtime["audio_arm_seconds"]:
        raise ValueError("Repair80Summary audio runtime mismatch")
    diagnostics = prediction["diagnostics_summary"]
    candidate_summary = prediction["methods"]["candidate"]["grid_summary"]
    baseline_summary = prediction["methods"]["baseline"]["grid_summary"]
    selected_summary = prediction["methods"]["selected"]["grid_summary"]
    expected_candidate_section_count = (
        candidate_summary["section_count"] if candidate_summary is not None else None
    )
    if metric.candidate_section_count != expected_candidate_section_count:
        raise ValueError("Repair80Summary candidate section count mismatch")
    if diagnostics["selected_section_count"] != expected_candidate_section_count:
        raise ValueError("Repair80Summary diagnostics selected section count mismatch")
    expected_current_v2_segment_count = (
        baseline_summary["section_count"] if baseline_summary is not None else None
    )
    if metric.current_v2_segment_count != expected_current_v2_segment_count:
        raise ValueError("Repair80Summary current-v2 segment count mismatch")
    expected_seam_ms = (
        candidate_summary["maximum_seam_discontinuity_ms"]
        if candidate_summary is not None
        else None
    )
    if metric.seam_ms != expected_seam_ms:
        raise ValueError("Repair80Summary seam mismatch")
    hard_guards = prediction["hard_guards"]
    expected_integrity = (
        hard_guards["cache_unchanged"]
        and hard_guards["source_unchanged"]
        and hard_guards["schema_valid"]
        and hard_guards["row_within_byte_cap"]
    )
    if metric.replay_schema_source_cache_integrity != expected_integrity:
        raise ValueError("Repair80Summary integrity guard mismatch")


def _stats_dict(value: StatsValue | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return value.to_dict()


def _boundary_evidence_from_summary(
    payload: Mapping[str, Any],
    *,
    context: str,
) -> BoundaryEvidence:
    summary = validate_boundary_summary(payload)
    f1_payload = _coerce_ratio_value(summary["f1"], f"{context}.f1")
    if f1_payload.state == "undefined":
        f1 = undefined_ratio()
    else:
        if f1_payload.value is None:
            raise ValueError(f"{context}.f1 must be finite or undefined")
        f1 = ratio_value(f1_payload.value, 1.0)
    return BoundaryEvidence(
        eligible=summary["eligible"],
        valid_difficulty_count=summary["valid_difficulty_count"],
        valid_weak_change=summary["weak_consensus_supported_count"] > 0,
        f1=f1,
    )


def _artifacts_module() -> Any:
    from pulsefield_model.timing.evaluation import exp007_artifacts as artifacts

    return artifacts


def _repair80_protocol_gates(gates: Repair80Gates) -> dict[str, Any]:
    return protocol.validate_repair80_gates(gates.to_dict())


def _repair_row_weak_pair_hash_payload(
    row_refs: Sequence[Mapping[str, Any]],
    weak_refs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(row_refs) != len(weak_refs):
        raise ValueError("row/weak ref count mismatch")
    pairs: list[dict[str, Any]] = []
    for row_ref, weak_ref in zip(row_refs, weak_refs, strict=True):
        if row_ref["row_index"] != weak_ref["row_index"]:
            raise ValueError("row/weak ref index mismatch")
        if row_ref["cache_audio_key"] != weak_ref["cache_audio_key"]:
            raise ValueError("row/weak ref cache key mismatch")
        if row_ref["row_payload_sha256"] != weak_ref["prediction_row_sha256"]:
            raise ValueError("row/weak ref prediction SHA mismatch")
        pairs.append(
            {
                "row_index": row_ref["row_index"],
                "cache_audio_key": row_ref["cache_audio_key"],
                "row_payload_sha256": row_ref["row_payload_sha256"],
                "prediction_row_sha256": weak_ref["prediction_row_sha256"],
                "weak_row_payload_sha256": weak_ref["weak_row_payload_sha256"],
            }
        )
    return pairs


def _require_pass_weak_outcome(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    weak_outcome = protocol.validate_schedule_weak_veto_outcome(payload)
    if weak_outcome["status"] != "success":
        raise ValueError("repair80 requires weak success outcome")
    weak_summary = weak_outcome["summary"]
    if weak_summary["decision"] != "pass" or weak_summary["action"] != "authorize_repair80":
        raise ValueError("repair80 requires weak pass outcome")
    return weak_summary


def _phase_reducers(
    by_key: Mapping[str, Repair80MetricRow], keys: Sequence[str], *, minimum: int
) -> tuple[
    RatioValue,
    RatioValue,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    if len(keys) < minimum:
        return undefined_ratio(), undefined_ratio(), None, None, None, None
    current_mean = arithmetic_mean(
        by_key[key].weak.current_v2_phase_ms.mean for key in keys  # type: ignore[union-attr]
    )
    pure_mean = arithmetic_mean(
        by_key[key].weak.pure_exp006_phase_ms.mean for key in keys  # type: ignore[union-attr]
    )
    current_p90 = linear_p90(
        by_key[key].weak.current_v2_phase_ms.p90 for key in keys  # type: ignore[union-attr]
    )
    pure_p90 = linear_p90(
        by_key[key].weak.pure_exp006_phase_ms.p90 for key in keys  # type: ignore[union-attr]
    )
    return (
        ratio_value(pure_mean, current_mean),
        ratio_value(pure_p90, current_p90),
        current_mean,
        pure_mean,
        current_p90,
        pure_p90,
    )


def _drift_reducers(
    by_key: Mapping[str, Repair80MetricRow], keys: Sequence[str], *, minimum: int
) -> tuple[
    RatioValue,
    RatioValue,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    if len(keys) < minimum:
        return undefined_ratio(), undefined_ratio(), None, None, None, None
    current_mean = arithmetic_mean(
        by_key[key].weak.current_v2_alias_max_prefix_ms.mean for key in keys  # type: ignore[union-attr]
    )
    pure_mean = arithmetic_mean(
        by_key[key].weak.pure_exp006_alias_max_prefix_ms.mean for key in keys  # type: ignore[union-attr]
    )
    current_p90 = linear_p90(
        by_key[key].weak.current_v2_alias_max_prefix_ms.p90 for key in keys  # type: ignore[union-attr]
    )
    pure_p90 = linear_p90(
        by_key[key].weak.pure_exp006_alias_max_prefix_ms.p90 for key in keys  # type: ignore[union-attr]
    )
    return (
        ratio_value(pure_mean, current_mean),
        ratio_value(pure_p90, current_p90),
        current_mean,
        pure_mean,
        current_p90,
        pure_p90,
    )


@dataclass(frozen=True)
class WeakResumePlan:
    completed_prefix: tuple[WeakRowRef, ...]
    pending: tuple[WeakPendingRowRef, ...]


def validate_weak_resume_prefix(
    selected_rows: Sequence[PredictionRowRef], completed: Sequence[WeakRowRef]
) -> WeakResumePlan:
    if len(selected_rows) != SCHEDULE_ROW_COUNT:
        raise ValueError("schedule weak resume requires exactly 16 selected rows")
    if len(completed) > len(selected_rows):
        raise ValueError("weak prefix is longer than selected rows")
    for index, selected in enumerate(selected_rows):
        if selected.row_index != index:
            raise ValueError("selected row refs must be contiguous and ordered")
    for index, weak_ref in enumerate(completed):
        selected = selected_rows[index]
        if (
            weak_ref.row_index != index
            or weak_ref.cache_audio_key != selected.cache_audio_key
            or weak_ref.prediction_row_sha256 != selected.prediction_row_sha256
        ):
            raise ValueError("weak prefix is stale, swapped, gapped, or mismatched")
    pending = tuple(
        WeakPendingRowRef(ref.row_index, ref.cache_audio_key, ref.prediction_row_sha256)
        for ref in selected_rows[len(completed) :]
    )
    return WeakResumePlan(tuple(completed), pending)


def make_schedule_weak_failure_record(
    *,
    schema_descriptor_sha256: str | None = None,
    schedule_arm: str,
    four_arm_stage_summary_sha256: str,
    candidate_global_manifest_sha256: str,
    source_selection_sha256: str,
    source_closure_fingerprint_sha256: str,
    selected_rows: Sequence[PredictionRowRef],
    completed: Sequence[WeakRowRef],
    failure_kind: Literal[
        "weak_input_failure",
        "comparator_failure",
        "metrics_failure",
        "schema_failure",
        "publication_failure",
    ],
    failure_stage: Literal["weak_input", "comparator", "metrics", "schema", "publication"],
    causing_row_index: int | None = None,
) -> dict[str, Any]:
    stage_by_kind = {
        "weak_input_failure": "weak_input",
        "comparator_failure": "comparator",
        "metrics_failure": "metrics",
        "schema_failure": "schema",
        "publication_failure": "publication",
    }
    if failure_kind not in stage_by_kind:
        raise ValueError("weak failure kind is invalid")
    expected_stage = stage_by_kind[failure_kind]
    if failure_stage != expected_stage:
        raise ValueError("weak failure kind/stage mismatch")
    descriptor = schema_descriptor_sha256 or protocol.schema_descriptor_sha256(
        protocol.SCHEDULE_WEAK_FAILURE_SCHEMA
    )
    if descriptor != protocol.schema_descriptor_sha256(protocol.SCHEDULE_WEAK_FAILURE_SCHEMA):
        raise ValueError("weak failure schema descriptor must match protocol registry")
    for name, value in (
        ("schema_descriptor_sha256", descriptor),
        ("four_arm_stage_summary_sha256", four_arm_stage_summary_sha256),
        ("candidate_global_manifest_sha256", candidate_global_manifest_sha256),
        ("source_selection_sha256", source_selection_sha256),
        ("source_closure_fingerprint_sha256", source_closure_fingerprint_sha256),
    ):
        _require_sha256(value, name)
    _require_nonempty(schedule_arm, "schedule_arm")
    plan = validate_weak_resume_prefix(selected_rows, completed)
    if causing_row_index is not None:
        _require_index(causing_row_index, "causing_row_index")
        if causing_row_index != len(completed):
            raise ValueError("causing row must be the first pending weak row")
        causing_key = selected_rows[causing_row_index].cache_audio_key
    else:
        causing_key = None
    completed_payload = [ref.to_dict() for ref in plan.completed_prefix]
    pending_payload = [ref.to_dict() for ref in plan.pending]
    payload: dict[str, Any] = {
        "schema": "pulsefield_model.timing_v3_exp007_schedule_weak_failure_v1",
        "experiment_id": EXP007_EXPERIMENT_ID,
        "stage": SCHEDULE_STAGE,
        "schema_descriptor_sha256": descriptor,
        "schedule_arm": schedule_arm,
        "four_arm_stage_summary_sha256": four_arm_stage_summary_sha256,
        "candidate_global_manifest_sha256": candidate_global_manifest_sha256,
        "source_selection_sha256": source_selection_sha256,
        "source_closure_fingerprint_sha256": source_closure_fingerprint_sha256,
        "expected_row_count": SCHEDULE_ROW_COUNT,
        "failure_kind": failure_kind,
        "failure_stage": failure_stage,
        "causing_row_index": causing_row_index,
        "causing_cache_audio_key": causing_key,
        "completed_prefix_count": len(completed_payload),
        "completed_prefix": completed_payload,
        "completed_prefix_sha256": protocol.canonical_json_sha256(completed_payload),
        "pending_count": len(pending_payload),
        "pending": pending_payload,
        "pending_sha256": protocol.canonical_json_sha256(pending_payload),
        "failure_deterministic_fingerprint_sha256": None,
        "full_payload_sha256": None,
    }
    deterministic = dict(payload)
    deterministic.pop("failure_deterministic_fingerprint_sha256")
    deterministic.pop("full_payload_sha256")
    payload["failure_deterministic_fingerprint_sha256"] = protocol.canonical_json_sha256(
        deterministic
    )
    full = dict(payload)
    full.pop("full_payload_sha256")
    payload["full_payload_sha256"] = protocol.canonical_json_sha256(full)
    return validate_schedule_weak_failure_record(payload, selected_rows=selected_rows)


def validate_schedule_weak_failure_record(
    payload: Mapping[str, Any],
    *,
    selected_rows: Sequence[PredictionRowRef] | None = None,
) -> dict[str, Any]:
    failure = protocol.validate_schedule_weak_failure_record(payload)
    stage_by_kind = {
        "weak_input_failure": "weak_input",
        "comparator_failure": "comparator",
        "metrics_failure": "metrics",
        "schema_failure": "schema",
        "publication_failure": "publication",
    }
    if failure["failure_stage"] != stage_by_kind[failure["failure_kind"]]:
        raise ValueError("weak failure kind/stage mismatch")
    completed = tuple(
        WeakRowRef(
            ref["row_index"],
            ref["cache_audio_key"],
            ref["prediction_row_sha256"],
            ref["weak_row_payload_sha256"],
        )
        for ref in failure["completed_prefix"]
    )
    pending = tuple(
        WeakPendingRowRef(
            ref["row_index"],
            ref["cache_audio_key"],
            ref["prediction_row_sha256"],
        )
        for ref in failure["pending"]
    )
    if selected_rows is not None:
        plan = validate_weak_resume_prefix(selected_rows, completed)
        if [ref.to_dict() for ref in plan.pending] != [ref.to_dict() for ref in pending]:
            raise ValueError("weak failure pending suffix mismatch")
    else:
        if tuple(ref.row_index for ref in completed) != tuple(range(len(completed))):
            raise ValueError("weak failure completed prefix is not contiguous")
        if tuple(ref.row_index for ref in pending) != tuple(
            range(len(completed), SCHEDULE_ROW_COUNT)
        ):
            raise ValueError("weak failure pending suffix is not contiguous")
    causing_index = failure["causing_row_index"]
    causing_key = failure["causing_cache_audio_key"]
    if causing_index is not None:
        if causing_index != len(completed):
            raise ValueError("weak failure causing row must be first pending")
        if not pending or pending[0].cache_audio_key != causing_key:
            raise ValueError("weak failure causing key mismatch")
    return failure


def make_schedule_weak_success_outcome(
    summary: Mapping[str, Any], *, schema_descriptor_sha256: str | None = None
) -> dict[str, Any]:
    descriptor = schema_descriptor_sha256 or protocol.schema_descriptor_sha256(
        protocol.SCHEDULE_WEAK_SUCCESS_SCHEMA
    )
    if descriptor != protocol.schema_descriptor_sha256(protocol.SCHEDULE_WEAK_SUCCESS_SCHEMA):
        raise ValueError("weak success schema descriptor must match protocol registry")
    summary_payload = validate_schedule_weak_veto_summary(summary)
    payload: dict[str, Any] = {
        "schema": protocol.SCHEDULE_WEAK_SUCCESS_SCHEMA,
        "experiment_id": EXP007_EXPERIMENT_ID,
        "stage": SCHEDULE_STAGE,
        "schema_descriptor_sha256": descriptor,
        "status": "success",
        "summary": summary_payload,
        "summary_payload_sha256": protocol.object_complete_sha256(summary_payload),
    }
    return protocol.validate_schedule_weak_veto_outcome(
        protocol.with_payload_hash(payload, "outcome_fingerprint_sha256")
    )


def make_schedule_weak_hard_failure_outcome(
    failure: Mapping[str, Any], *, schema_descriptor_sha256: str | None = None
) -> dict[str, Any]:
    descriptor = schema_descriptor_sha256 or protocol.schema_descriptor_sha256(
        protocol.SCHEDULE_WEAK_HARD_FAILURE_SCHEMA
    )
    if descriptor != protocol.schema_descriptor_sha256(protocol.SCHEDULE_WEAK_HARD_FAILURE_SCHEMA):
        raise ValueError("weak hard-failure schema descriptor must match protocol registry")
    failure_payload = validate_schedule_weak_failure_record(failure)
    payload: dict[str, Any] = {
        "schema": protocol.SCHEDULE_WEAK_HARD_FAILURE_SCHEMA,
        "experiment_id": EXP007_EXPERIMENT_ID,
        "stage": SCHEDULE_STAGE,
        "schema_descriptor_sha256": descriptor,
        "status": "hard_failure",
        "failure": failure_payload,
        "failure_payload_sha256": protocol.object_complete_sha256(failure_payload),
    }
    return protocol.validate_schedule_weak_veto_outcome(
        protocol.with_payload_hash(payload, "outcome_fingerprint_sha256")
    )


def validate_schedule_weak_outcome(payload: Mapping[str, Any]) -> Literal["success", "hard_failure"]:
    outcome = protocol.validate_schedule_weak_veto_outcome(payload)
    if outcome["status"] == "success":
        validate_schedule_weak_veto_summary(outcome["summary"])
        return "success"
    validate_schedule_weak_failure_record(outcome["failure"])
    return "hard_failure"


def schedule_weak_resume_action(
    outcome: Mapping[str, Any] | None,
    *,
    selected_rows: Sequence[PredictionRowRef],
    completed: Sequence[WeakRowRef],
) -> Literal["reuse_success", "reuse_hard_failure", "continue_prefix"]:
    if outcome is not None:
        validated = protocol.validate_schedule_weak_veto_outcome(outcome)
        if validated["status"] == "success":
            summary = validate_schedule_weak_veto_summary(validated["summary"])
            _validate_selected_rows_schedule_arm(
                selected_rows,
                summary["schedule_arm"],
                context="weak success outcome",
            )
            completed_refs = tuple(
                WeakRowRef(
                    ref["row_index"],
                    ref["cache_audio_key"],
                    ref["prediction_row_sha256"],
                    ref["weak_row_payload_sha256"],
                )
                for ref in summary["weak_row_refs"]
            )
            plan = validate_weak_resume_prefix(selected_rows, completed_refs)
            if plan.pending:
                raise ValueError("weak success outcome does not cover selected rows")
            return "reuse_success"
        failure = validate_schedule_weak_failure_record(
            validated["failure"],
            selected_rows=selected_rows,
        )
        _validate_selected_rows_schedule_arm(
            selected_rows,
            failure["schedule_arm"],
            context="weak hard-failure outcome",
        )
        return "reuse_hard_failure"
    validate_weak_resume_prefix(selected_rows, completed)
    return "continue_prefix"


def _validate_selected_rows_schedule_arm(
    selected_rows: Sequence[PredictionRowRef],
    expected_schedule_arm: str,
    *,
    context: str,
) -> None:
    _require_nonempty(expected_schedule_arm, f"{context}.schedule_arm")
    for selected in selected_rows:
        if selected.schedule_arm != expected_schedule_arm:
            raise ValueError(f"{context} selected schedule arm mismatch")


def _validate_winner_pairing(
    rows: Sequence[WeakMetricRow],
    *,
    selected_rows: Sequence[PredictionRowRef],
    selected_schedule_arm: str,
    stage: str,
    expected_count: int,
    require_unique_audio_group: bool,
) -> None:
    _require_nonempty(selected_schedule_arm, "selected_schedule_arm")
    if len(rows) != expected_count or len(selected_rows) != expected_count:
        raise ValueError(f"{stage} requires exactly {expected_count} winner rows")
    if tuple(row.row_index for row in rows) != tuple(range(expected_count)):
        raise ValueError("weak rows must be in contiguous identity order")
    if tuple(ref.row_index for ref in selected_rows) != tuple(range(expected_count)):
        raise ValueError("selected row refs must be in contiguous identity order")
    keys = [row.cache_audio_key for row in rows]
    if len(set(keys)) != expected_count:
        raise ValueError("cache_audio_key values must be unique")
    if require_unique_audio_group and len({row.audio_group_key for row in rows}) != expected_count:
        raise ValueError("schedule16 requires one-to-one audio group/key mapping")
    for row, selected in zip(rows, selected_rows):
        if row.stage != stage:
            raise ValueError("cross-stage weak row is forbidden")
        if row.schedule_arm != selected_schedule_arm or selected.schedule_arm != selected_schedule_arm:
            raise ValueError("winner-only evaluation forbids runner-up rows")
        if (
            row.row_index != selected.row_index
            or row.cache_audio_key != selected.cache_audio_key
            or row.prediction_row_sha256 != selected.prediction_row_sha256
        ):
            raise ValueError("weak row and selected prediction pairing mismatch")


def _nested_dataclass_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {
            field: _nested_dataclass_dict(getattr(value, field))
            for field in value.__dataclass_fields__
        }
    if isinstance(value, Mapping):
        return {key: _nested_dataclass_dict(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_nested_dataclass_dict(item) for item in value]
    if isinstance(value, list):
        return [_nested_dataclass_dict(item) for item in value]
    return value
