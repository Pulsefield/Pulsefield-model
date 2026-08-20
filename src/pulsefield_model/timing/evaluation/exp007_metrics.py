from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence


Decision = Literal["pass", "ambiguous", "negative"]
ScheduleArm = Literal["S30", "S60", "S90", "S64"]

SCHEDULE_ARMS: tuple[ScheduleArm, ...] = ("S30", "S60", "S90", "S64")
SCHEDULE_TIE_RANK: Mapping[ScheduleArm, int] = {
    "S64": 0,
    "S90": 1,
    "S60": 2,
    "S30": 3,
}
SOURCE_ELIMINATION_REASON_ORDER = (
    "candidate_fallback_guard",
    "no_origin_or_path_guard",
    "runtime_nonfinite",
    "runtime_p90_guard",
    "row_timeout_guard",
    "rss_nonfinite",
    "rss_cap_guard",
    "seam_guard",
    "section_cap_guard",
    "row_consistency_guard",
    "overlap_common_minimum",
    "section_common_minimum",
    "overlap_e1_guard",
)
SOURCE_STAGE_AUDIO_COUNT = 16
SOURCE_OVERLAP_COMMON_MINIMUM = 5
SOURCE_SECTION_COMMON_MINIMUM = 8
SOURCE_FALLBACK_COUNT_MAXIMUM = 1
SOURCE_NO_ORIGIN_OR_PATH_COUNT_MAXIMUM = 0
SOURCE_RUNTIME_P90_MAXIMUM_SECONDS = 60.0
SOURCE_ROW_TIMEOUT_SECONDS = 180.0
SOURCE_RSS_CAP_BYTES = 4_294_967_296
SOURCE_ROW_JSON_BYTE_CAP = 1_048_576
SOURCE_SECTION_CAP = 20


def canonical_json_bytes(value: Any) -> bytes:
    """Return the Exp007 canonical JSON encoding for a mathematical payload."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _finite_nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class RatioValue:
    state: Literal["finite", "both_zero", "positive_infinity", "undefined"]
    numerator: float | None
    denominator: float | None
    value: float | None

    def __post_init__(self) -> None:
        if self.state == "undefined":
            if (self.numerator, self.denominator, self.value) != (None, None, None):
                raise ValueError("undefined ratio requires three null values")
            return
        if self.numerator is None or self.denominator is None:
            raise ValueError("defined ratio states require numeric operands")
        numerator = _finite_nonnegative(self.numerator, "numerator")
        denominator = _finite_nonnegative(self.denominator, "denominator")
        object.__setattr__(self, "numerator", numerator)
        object.__setattr__(self, "denominator", denominator)
        if self.state == "finite":
            if denominator <= 0.0:
                raise ValueError("finite ratio requires a positive denominator")
            expected = numerator / denominator
            if self.value is None or not math.isfinite(float(self.value)):
                raise ValueError("finite ratio requires a finite value")
            if float(self.value) != expected:
                raise ValueError("finite ratio value is inconsistent with operands")
            object.__setattr__(self, "value", expected)
            return
        if self.state == "both_zero":
            if numerator != 0.0 or denominator != 0.0 or self.value != 1.0:
                raise ValueError("both_zero ratio must be 0/0 with value 1.0")
            return
        if self.state == "positive_infinity":
            if numerator <= 0.0 or denominator != 0.0 or self.value is not None:
                raise ValueError("positive_infinity requires positive/zero and null value")
            return
        raise ValueError(f"unknown ratio state: {self.state!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ratio_value(numerator: float | None, denominator: float | None) -> RatioValue:
    if numerator is None or denominator is None:
        if numerator is not None or denominator is not None:
            raise ValueError("undefined ratio requires both operands to be null")
        return RatioValue("undefined", None, None, None)
    numerator = _finite_nonnegative(numerator, "numerator")
    denominator = _finite_nonnegative(denominator, "denominator")
    if denominator > 0.0:
        return RatioValue("finite", numerator, denominator, numerator / denominator)
    if numerator == 0.0:
        return RatioValue("both_zero", 0.0, 0.0, 1.0)
    return RatioValue("positive_infinity", numerator, 0.0, None)


def undefined_ratio() -> RatioValue:
    return ratio_value(None, None)


@dataclass(frozen=True)
class RateValue:
    numerator: int
    denominator: int
    value: float

    def __post_init__(self) -> None:
        numerator = _nonnegative_int(self.numerator, "numerator")
        denominator = _nonnegative_int(self.denominator, "denominator")
        if denominator <= 0 or numerator > denominator:
            raise ValueError("rate requires 0 <= numerator <= denominator and denominator > 0")
        expected = numerator / denominator
        if isinstance(self.value, bool) or not math.isfinite(float(self.value)):
            raise ValueError("rate value must be finite")
        if float(self.value) != expected:
            raise ValueError("rate value is inconsistent with operands")
        object.__setattr__(self, "value", expected)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rate_value(numerator: int, denominator: int) -> RateValue:
    numerator = _nonnegative_int(numerator, "numerator")
    denominator = _nonnegative_int(denominator, "denominator")
    if denominator <= 0 or numerator > denominator:
        raise ValueError("rate requires 0 <= numerator <= denominator and denominator > 0")
    return RateValue(numerator, denominator, numerator / denominator)


def linear_percentile(values: Sequence[float] | Iterable[float], percentile: float) -> float:
    if isinstance(percentile, bool):
        raise ValueError("percentile must be finite and in [0, 100]")
    percentile = float(percentile)
    if not math.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be finite and in [0, 100]")
    ordered = sorted(_finite_nonnegative(value, "sample") for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one sample")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def linear_p90(values: Sequence[float] | Iterable[float]) -> float:
    return linear_percentile(values, 90.0)


@dataclass(frozen=True)
class StatsValue:
    count: int
    mean: float | None
    p50: float | None
    p90: float | None
    maximum: float | None

    def __post_init__(self) -> None:
        count = _nonnegative_int(self.count, "count")
        values = (self.mean, self.p50, self.p90, self.maximum)
        if count == 0:
            if values != (None, None, None, None):
                raise ValueError("empty stats require null reducers")
            return
        if any(value is None for value in values):
            raise ValueError("nonempty stats require every reducer")
        checked = tuple(
            _finite_nonnegative(float(value), name)
            for name, value in zip(("mean", "p50", "p90", "maximum"), values)
        )
        if checked[3] < max(checked[0], checked[1], checked[2]):
            raise ValueError("maximum cannot be below another reducer")
        object.__setattr__(self, "mean", checked[0])
        object.__setattr__(self, "p50", checked[1])
        object.__setattr__(self, "p90", checked[2])
        object.__setattr__(self, "maximum", checked[3])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stats_value(values: Sequence[float] | Iterable[float]) -> StatsValue:
    checked = tuple(_finite_nonnegative(value, "sample") for value in values)
    if not checked:
        return StatsValue(0, None, None, None, None)
    return StatsValue(
        count=len(checked),
        mean=float(statistics.fmean(checked)),
        p50=linear_percentile(checked, 50.0),
        p90=linear_p90(checked),
        maximum=max(checked),
    )


@dataclass(frozen=True)
class AudioSetBinding:
    count: int
    sorted_cache_audio_keys_sha256: str

    def __post_init__(self) -> None:
        _nonnegative_int(self.count, "count")
        if (
            not isinstance(self.sorted_cache_audio_keys_sha256, str)
            or len(self.sorted_cache_audio_keys_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.sorted_cache_audio_keys_sha256)
        ):
            raise ValueError("sorted_cache_audio_keys_sha256 must be lowercase SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audio_set_binding(keys: Iterable[str]) -> AudioSetBinding:
    ordered = sorted(keys)
    if any(not isinstance(key, str) or not key for key in ordered):
        raise ValueError("audio keys must be nonempty strings")
    if len(set(ordered)) != len(ordered):
        raise ValueError("audio keys must be unique")
    return AudioSetBinding(len(ordered), canonical_sha256(ordered))


def arithmetic_mean(values: Sequence[float] | Iterable[float]) -> float:
    checked = tuple(_finite_nonnegative(value, "sample") for value in values)
    if not checked:
        raise ValueError("mean requires at least one sample")
    return float(statistics.fmean(checked))


def classify_upper_ratio(
    ratio: RatioValue,
    *,
    pass_max: float,
    ambiguous_max: float,
) -> Decision:
    pass_max = _finite_nonnegative(pass_max, "pass_max")
    ambiguous_max = _finite_nonnegative(ambiguous_max, "ambiguous_max")
    if ambiguous_max < pass_max:
        raise ValueError("ambiguous_max must be >= pass_max")
    if ratio.state == "undefined":
        return "ambiguous"
    if ratio.state == "positive_infinity":
        return "negative"
    assert ratio.value is not None
    if ratio.value <= pass_max:
        return "pass"
    if ratio.value <= ambiguous_max:
        return "ambiguous"
    return "negative"


def classify_upper_value(value: float | None, *, pass_max: float, ambiguous_max: float) -> Decision:
    if value is None:
        return "ambiguous"
    value = _finite_nonnegative(value, "value")
    pass_max = _finite_nonnegative(pass_max, "pass_max")
    ambiguous_max = _finite_nonnegative(ambiguous_max, "ambiguous_max")
    if ambiguous_max < pass_max:
        raise ValueError("ambiguous_max must be >= pass_max")
    if value <= pass_max:
        return "pass"
    if value <= ambiguous_max:
        return "ambiguous"
    return "negative"


def classify_lower_rate(rate: RateValue | RatioValue, *, pass_min: float, ambiguous_min: float) -> Decision:
    pass_min = _finite_nonnegative(pass_min, "pass_min")
    ambiguous_min = _finite_nonnegative(ambiguous_min, "ambiguous_min")
    if pass_min < ambiguous_min:
        raise ValueError("pass_min must be >= ambiguous_min")
    if isinstance(rate, RatioValue):
        if rate.state == "undefined":
            return "ambiguous"
        if rate.state == "positive_infinity":
            raise ValueError("coverage cannot be positive infinity")
        assert rate.value is not None
        value = rate.value
    else:
        value = rate.value
    if value >= pass_min:
        return "pass"
    if value >= ambiguous_min:
        return "ambiguous"
    return "negative"


def classify_boundary_delta(value: float | None) -> Decision:
    if value is None:
        return "ambiguous"
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError("boundary delta must be finite")
    value = float(value)
    if value >= -0.05:
        return "pass"
    if value >= -0.10:
        return "ambiguous"
    return "negative"


def combine_decisions(values: Iterable[Decision]) -> Decision:
    decisions = tuple(values)
    if any(value not in {"pass", "ambiguous", "negative"} for value in decisions):
        raise ValueError("unknown decision")
    if "negative" in decisions:
        return "negative"
    if "ambiguous" in decisions:
        return "ambiguous"
    return "pass"


@dataclass(frozen=True)
class SourceMetricRow:
    """Source-only projection of one validated schedule16 RowResult.

    This object intentionally contains no labels, weak-comparator values, or
    `.osu` evidence.  The persisted RowResult remains the source of truth; this
    is only the immutable reducer input projection.
    """

    schedule_arm: ScheduleArm
    row_index: int
    cache_audio_key: str
    audio_group_key: str
    cache_valid: bool
    projection_evaluable: bool
    candidate_status: Literal["accepted", "tagged_fallback"]
    candidate_fallback_reason: str | None
    baseline_status: Literal["accepted", "unavailable"]
    selected_status: Literal["accepted", "unavailable"]
    candidate_section_count: int | None
    current_v2_segment_count: int | None
    current_v2_projection_sha256: str | None
    candidate_seam_ms: float | None
    overlap_p90_ms: float | None
    audio_arm_seconds: float
    row_json_bytes: int
    replay_schema_source_cache_candidate_v2_consistent: bool

    def __post_init__(self) -> None:
        if self.schedule_arm not in SCHEDULE_ARMS:
            raise ValueError("invalid schedule arm")
        _nonnegative_int(self.row_index, "row_index")
        for name in ("cache_audio_key", "audio_group_key"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        for name in (
            "cache_valid",
            "projection_evaluable",
            "replay_schema_source_cache_candidate_v2_consistent",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")
        if self.projection_evaluable and not self.cache_valid:
            raise ValueError("projection evaluable requires a valid cache")
        if self.candidate_status not in {"accepted", "tagged_fallback"}:
            raise ValueError("invalid candidate status")
        if self.baseline_status not in {"accepted", "unavailable"}:
            raise ValueError("invalid baseline status")
        if self.selected_status not in {"accepted", "unavailable"}:
            raise ValueError("invalid selected status")
        if self.candidate_status == "accepted":
            if self.candidate_fallback_reason is not None:
                raise ValueError("accepted candidate cannot have a fallback reason")
            if self.candidate_section_count is None or self.candidate_seam_ms is None:
                raise ValueError("accepted candidate requires section and seam values")
            _nonnegative_int(self.candidate_section_count, "candidate_section_count")
            _finite_nonnegative(self.candidate_seam_ms, "candidate_seam_ms")
        else:
            if self.candidate_fallback_reason not in {
                "no_origin_candidate",
                "no_local_frontier_path",
                "local_frontier_resource_cap_exceeded",
            }:
                raise ValueError("invalid candidate fallback reason")
            if self.candidate_section_count is not None or self.candidate_seam_ms is not None:
                raise ValueError("fallback candidate cannot have grid values")
            if self.overlap_p90_ms is not None:
                raise ValueError("fallback candidate cannot have overlap")
        if self.baseline_status == "accepted":
            if self.current_v2_segment_count is None:
                raise ValueError("accepted baseline requires segment count")
            _nonnegative_int(self.current_v2_segment_count, "current_v2_segment_count")
            _require_digest(
                self.current_v2_projection_sha256,
                "current_v2_projection_sha256",
            )
        elif (
            self.current_v2_segment_count is not None
            or self.current_v2_projection_sha256 is not None
        ):
            raise ValueError("unavailable baseline cannot have projection values")
        expected_selected = (
            self.candidate_status == "accepted" or self.baseline_status == "accepted"
        )
        if (self.selected_status == "accepted") != expected_selected:
            raise ValueError("selected status violates the four-branch truth table")
        if self.overlap_p90_ms is not None:
            if self.candidate_status != "accepted":
                raise ValueError("overlap requires an accepted candidate")
            _finite_nonnegative(self.overlap_p90_ms, "overlap_p90_ms")
        _finite_nonnegative(self.audio_arm_seconds, "audio_arm_seconds")
        _nonnegative_int(self.row_json_bytes, "row_json_bytes")


@dataclass(frozen=True)
class RuntimeSummary:
    row_seconds: StatsValue
    aggregate_wall_seconds: float

    def __post_init__(self) -> None:
        _finite_nonnegative(self.aggregate_wall_seconds, "aggregate_wall_seconds")


@dataclass(frozen=True)
class RssSummary:
    worker_count: int
    worker_lifetime_bytes: tuple[int, ...]
    arm_max_worker_bytes: int

    def __post_init__(self) -> None:
        if self.worker_count != 4 or len(self.worker_lifetime_bytes) != 4:
            raise ValueError("RSS summary requires exactly four workers")
        checked = tuple(
            _nonnegative_int(value, "worker_lifetime_bytes")
            for value in self.worker_lifetime_bytes
        )
        expected = max(checked)
        if self.arm_max_worker_bytes != expected:
            raise ValueError("arm_max_worker_bytes mismatch")
        object.__setattr__(self, "worker_lifetime_bytes", checked)


@dataclass(frozen=True)
class SourceArmDenominators:
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


@dataclass(frozen=True)
class SourceArmGates:
    candidate_fallback_rate: RateValue
    selected_product_fallback_rate: RateValue
    no_origin_or_path_rate: RateValue
    runtime_seconds: StatsValue
    worker_rss_bytes: StatsValue
    candidate_seam_ms: StatsValue
    candidate_section_count: StatsValue
    row_json_bytes: StatsValue
    every_row_under_180_seconds: bool
    seam_zero: bool
    section_cap_valid: bool
    row_byte_cap_valid: bool
    replay_schema_source_cache_candidate_v2_consistent: bool


@dataclass(frozen=True)
class SourceArmEvaluation:
    schedule_arm: ScheduleArm
    rows: tuple[SourceMetricRow, ...]
    denominators: SourceArmDenominators
    gates: SourceArmGates
    runtime_summary: RuntimeSummary
    rss_summary: RssSummary


@dataclass(frozen=True)
class SourceArmOrderValues:
    schedule_arm: ScheduleArm
    e0_eligible: bool
    e1_eligible: bool
    elimination_reasons: tuple[str, ...]
    candidate_fallback_count: int
    no_origin_or_path_count: int
    p90_overlap_ms: float | None
    section_inflation_violation_count: int | None
    p90_section_excess: float | None
    p90_runtime: float
    max_worker_rss: int
    tie_rank: int
    order_tuple_sha256: str | None


@dataclass(frozen=True)
class SourceScheduleSelection:
    source_decision: Literal["positive", "ambiguous", "negative"]
    selected_schedule_arm: ScheduleArm | None
    overlap_common: AudioSetBinding
    section_common: AudioSetBinding
    arm_order_values: tuple[SourceArmOrderValues, ...]


def evaluate_source_arm(
    rows: Sequence[SourceMetricRow],
    *,
    schedule_arm: ScheduleArm,
    worker_lifetime_rss_bytes: Sequence[int],
    aggregate_wall_seconds: float,
) -> SourceArmEvaluation:
    if schedule_arm not in SCHEDULE_ARMS:
        raise ValueError("invalid schedule arm")
    if len(rows) != SOURCE_STAGE_AUDIO_COUNT:
        raise ValueError("source arm requires exactly 16 rows")
    if tuple(row.row_index for row in rows) != tuple(range(SOURCE_STAGE_AUDIO_COUNT)):
        raise ValueError("source rows must be in contiguous row order")
    if any(row.schedule_arm != schedule_arm for row in rows):
        raise ValueError("source row arm mismatch")
    group_keys = tuple(row.audio_group_key for row in rows)
    cache_keys = tuple(row.cache_audio_key for row in rows)
    if len(set(cache_keys)) != len(cache_keys):
        raise ValueError("source cache_audio_key values must be unique")
    if len(set(group_keys)) != len(group_keys):
        raise ValueError("schedule16 audio_group_key values must be unique")

    # Schedule16 source truth is grouped by the unique audio_group_key.  The
    # cache key remains the cross-arm identity/audit key, but it is not the
    # source-summary denominator identity (repair80 deliberately differs).
    def keys_where(predicate: Any) -> tuple[str, ...]:
        return tuple(row.audio_group_key for row in rows if predicate(row))

    accepted = keys_where(lambda row: row.candidate_status == "accepted")
    fallback = keys_where(lambda row: row.candidate_status == "tagged_fallback")
    baseline = keys_where(lambda row: row.baseline_status == "accepted")
    product = keys_where(lambda row: row.selected_status == "accepted")
    selected_fallback = keys_where(
        lambda row: row.candidate_status == "tagged_fallback"
        and row.baseline_status == "accepted"
    )
    no_origin = keys_where(
        lambda row: row.candidate_fallback_reason
        in {"no_origin_candidate", "no_local_frontier_path"}
    )
    resource = keys_where(
        lambda row: row.candidate_fallback_reason
        == "local_frontier_resource_cap_exceeded"
    )
    overlap = keys_where(lambda row: row.overlap_p90_ms is not None)
    denominators = SourceArmDenominators(
        stage_audio_count=SOURCE_STAGE_AUDIO_COUNT,
        stage_audio=audio_set_binding(group_keys),
        cache_valid_audio=audio_set_binding(
            keys_where(lambda row: row.cache_valid)
        ),
        projection_evaluable_audio=audio_set_binding(
            keys_where(lambda row: row.projection_evaluable)
        ),
        candidate_accepted_audio=audio_set_binding(accepted),
        candidate_fallback_audio=audio_set_binding(fallback),
        selected_product_fallback_audio=audio_set_binding(selected_fallback),
        baseline_accepted_audio=audio_set_binding(baseline),
        product_grid_available_audio=audio_set_binding(product),
        no_origin_or_path_audio=audio_set_binding(no_origin),
        resource_cap_fallback_audio=audio_set_binding(resource),
        overlap_available_audio=audio_set_binding(overlap),
    )
    runtimes = tuple(row.audio_arm_seconds for row in rows)
    rss = tuple(_nonnegative_int(value, "worker RSS") for value in worker_lifetime_rss_bytes)
    if len(rss) != 4:
        raise ValueError("source arm requires exactly four worker RSS values")
    seam_values = tuple(
        row.candidate_seam_ms
        for row in rows
        if row.candidate_seam_ms is not None
    )
    section_values = tuple(
        row.candidate_section_count
        for row in rows
        if row.candidate_section_count is not None
    )
    gates = SourceArmGates(
        candidate_fallback_rate=rate_value(len(fallback), SOURCE_STAGE_AUDIO_COUNT),
        selected_product_fallback_rate=rate_value(
            len(selected_fallback), SOURCE_STAGE_AUDIO_COUNT
        ),
        no_origin_or_path_rate=rate_value(len(no_origin), SOURCE_STAGE_AUDIO_COUNT),
        runtime_seconds=stats_value(runtimes),
        worker_rss_bytes=stats_value(rss),
        candidate_seam_ms=stats_value(seam_values),
        candidate_section_count=stats_value(section_values),
        row_json_bytes=stats_value(tuple(row.row_json_bytes for row in rows)),
        every_row_under_180_seconds=all(
            value < SOURCE_ROW_TIMEOUT_SECONDS for value in runtimes
        ),
        seam_zero=all(value == 0.0 for value in seam_values),
        section_cap_valid=all(value <= SOURCE_SECTION_CAP for value in section_values),
        row_byte_cap_valid=all(
            row.row_json_bytes < SOURCE_ROW_JSON_BYTE_CAP for row in rows
        ),
        replay_schema_source_cache_candidate_v2_consistent=all(
            row.replay_schema_source_cache_candidate_v2_consistent for row in rows
        ),
    )
    return SourceArmEvaluation(
        schedule_arm=schedule_arm,
        rows=tuple(rows),
        denominators=denominators,
        gates=gates,
        runtime_summary=RuntimeSummary(stats_value(runtimes), aggregate_wall_seconds),
        rss_summary=RssSummary(4, rss, max(rss)),
    )


def select_source_schedule(
    arms: Mapping[ScheduleArm, SourceArmEvaluation],
) -> SourceScheduleSelection:
    if set(arms) != set(SCHEDULE_ARMS):
        raise ValueError("source selection requires exactly the four frozen arms")
    ordered_arms = tuple(arms[arm] for arm in SCHEDULE_ARMS)
    reference_groups = tuple(row.audio_group_key for row in ordered_arms[0].rows)
    reference_keys = tuple(row.cache_audio_key for row in ordered_arms[0].rows)
    for arm in ordered_arms:
        if arm.schedule_arm not in SCHEDULE_ARMS:
            raise ValueError("source arm identity is invalid")
        if tuple(row.audio_group_key for row in arm.rows) != reference_groups:
            raise ValueError("cross-arm audio_group_key order mismatch")
        if tuple(row.cache_audio_key for row in arm.rows) != reference_keys:
            raise ValueError("cross-arm cache_audio_key order mismatch")

    preliminary_reasons: dict[ScheduleArm, list[str]] = {}
    e0_arms: list[SourceArmEvaluation] = []
    for arm in ordered_arms:
        reasons = _source_e0_reasons(arm)
        preliminary_reasons[arm.schedule_arm] = reasons
        if not reasons:
            e0_arms.append(arm)

    overlap_keys: tuple[str, ...] = ()
    section_keys: tuple[str, ...] = ()
    common_minima_hold = False
    if len(e0_arms) >= 2:
        overlap_keys = tuple(
            group_key
            for index, group_key in enumerate(reference_groups)
            if all(
                arm.rows[index].candidate_status == "accepted"
                and arm.rows[index].overlap_p90_ms is not None
                for arm in e0_arms
            )
        )
        section_keys = tuple(
            group_key
            for index, group_key in enumerate(reference_groups)
            if all(arm.rows[index].candidate_status == "accepted" for arm in e0_arms)
            and _current_v2_identical(tuple(arm.rows[index] for arm in e0_arms))
        )
        common_minima_hold = (
            len(overlap_keys) >= SOURCE_OVERLAP_COMMON_MINIMUM
            and len(section_keys) >= SOURCE_SECTION_COMMON_MINIMUM
        )

    order_values: list[SourceArmOrderValues] = []
    eligible_order_tuples: list[tuple[tuple[Any, ...], ScheduleArm]] = []
    for arm in ordered_arms:
        reasons = list(preliminary_reasons[arm.schedule_arm])
        e0 = not reasons
        p90_overlap: float | None = None
        section_violations: int | None = None
        p90_section_excess: float | None = None
        e1 = False
        order_sha: str | None = None
        if e0 and len(e0_arms) >= 2:
            if len(overlap_keys) < SOURCE_OVERLAP_COMMON_MINIMUM:
                reasons.append("overlap_common_minimum")
            if len(section_keys) < SOURCE_SECTION_COMMON_MINIMUM:
                reasons.append("section_common_minimum")
            if common_minima_hold:
                by_key = {row.audio_group_key: row for row in arm.rows}
                p90_overlap = linear_p90(
                    by_key[key].overlap_p90_ms for key in overlap_keys  # type: ignore[arg-type]
                )
                excess = tuple(
                    max(
                        0,
                        by_key[key].candidate_section_count
                        - by_key[key].current_v2_segment_count,
                    )
                    for key in section_keys
                )
                section_violations = sum(value > 1 for value in excess)
                p90_section_excess = linear_p90(excess)
                if p90_overlap > 90.0:
                    reasons.append("overlap_e1_guard")
                else:
                    e1 = True
                    order_tuple = (
                        arm.denominators.candidate_fallback_audio.count,
                        arm.denominators.no_origin_or_path_audio.count,
                        p90_overlap,
                        section_violations,
                        p90_section_excess,
                        SCHEDULE_TIE_RANK[arm.schedule_arm],
                    )
                    order_sha = canonical_sha256(order_tuple)
                    eligible_order_tuples.append((order_tuple, arm.schedule_arm))
        reasons = [
            reason for reason in SOURCE_ELIMINATION_REASON_ORDER if reason in reasons
        ]
        order_values.append(
            SourceArmOrderValues(
                schedule_arm=arm.schedule_arm,
                e0_eligible=e0,
                e1_eligible=e1,
                elimination_reasons=tuple(reasons),
                candidate_fallback_count=arm.denominators.candidate_fallback_audio.count,
                no_origin_or_path_count=arm.denominators.no_origin_or_path_audio.count,
                p90_overlap_ms=p90_overlap,
                section_inflation_violation_count=section_violations,
                p90_section_excess=p90_section_excess,
                p90_runtime=arm.gates.runtime_seconds.p90,  # type: ignore[arg-type]
                max_worker_rss=arm.rss_summary.arm_max_worker_bytes,
                tie_rank=SCHEDULE_TIE_RANK[arm.schedule_arm],
                order_tuple_sha256=order_sha,
            )
        )

    if len(e0_arms) < 2 or not common_minima_hold:
        decision: Literal["positive", "ambiguous", "negative"] = "ambiguous"
        selected_arm: ScheduleArm | None = None
    elif not eligible_order_tuples:
        decision = "negative"
        selected_arm = None
    else:
        decision = "positive"
        selected_arm = min(eligible_order_tuples)[1]
    return SourceScheduleSelection(
        source_decision=decision,
        selected_schedule_arm=selected_arm,
        overlap_common=audio_set_binding(overlap_keys),
        section_common=audio_set_binding(section_keys),
        arm_order_values=tuple(order_values),
    )


def _source_e0_reasons(arm: SourceArmEvaluation) -> list[str]:
    reasons: list[str] = []
    denominators = arm.denominators
    gates = arm.gates
    if denominators.candidate_fallback_audio.count > SOURCE_FALLBACK_COUNT_MAXIMUM:
        reasons.append("candidate_fallback_guard")
    if (
        denominators.no_origin_or_path_audio.count
        > SOURCE_NO_ORIGIN_OR_PATH_COUNT_MAXIMUM
    ):
        reasons.append("no_origin_or_path_guard")
    if gates.runtime_seconds.p90 is None:
        reasons.append("runtime_nonfinite")
    elif gates.runtime_seconds.p90 > SOURCE_RUNTIME_P90_MAXIMUM_SECONDS:
        reasons.append("runtime_p90_guard")
    if not gates.every_row_under_180_seconds:
        reasons.append("row_timeout_guard")
    if gates.worker_rss_bytes.maximum is None:
        reasons.append("rss_nonfinite")
    elif gates.worker_rss_bytes.maximum > SOURCE_RSS_CAP_BYTES:
        reasons.append("rss_cap_guard")
    if not gates.seam_zero:
        reasons.append("seam_guard")
    if not gates.section_cap_valid:
        reasons.append("section_cap_guard")
    if not (
        gates.row_byte_cap_valid
        and gates.replay_schema_source_cache_candidate_v2_consistent
        and denominators.cache_valid_audio.count == SOURCE_STAGE_AUDIO_COUNT
        and denominators.projection_evaluable_audio.count == SOURCE_STAGE_AUDIO_COUNT
    ):
        reasons.append("row_consistency_guard")
    return reasons


def _current_v2_identical(rows: Sequence[SourceMetricRow]) -> bool:
    if any(row.baseline_status != "accepted" for row in rows):
        return False
    projections = {row.current_v2_projection_sha256 for row in rows}
    segment_counts = {row.current_v2_segment_count for row in rows}
    return len(projections) == 1 and len(segment_counts) == 1


def _require_digest(value: str | None, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value
