from __future__ import annotations

import json

import pytest

from pulsefield_model.timing.evaluation.exp007_metrics import (
    AudioSetBinding,
    RateValue,
    RatioValue,
    StatsValue,
    SourceMetricRow,
    audio_set_binding,
    canonical_json_bytes,
    canonical_sha256,
    classify_boundary_delta,
    classify_lower_rate,
    classify_upper_ratio,
    combine_decisions,
    evaluate_source_arm,
    linear_p90,
    rate_value,
    ratio_value,
    select_source_schedule,
    stats_value,
    undefined_ratio,
)


def test_canonical_json_and_audio_set_hash_use_sorted_unique_key_preimage() -> None:
    assert canonical_json_bytes({"é": 1, "a": 2}) == b'{"a":2,"\\u00e9":1}'
    binding = audio_set_binding(["z", "a"])

    assert binding == AudioSetBinding(2, canonical_sha256(["a", "z"]))
    assert json.loads(canonical_json_bytes(binding.to_dict()))["count"] == 2
    with pytest.raises(ValueError, match="unique"):
        audio_set_binding(["a", "a"])


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [
        (2.0, 4.0, RatioValue("finite", 2.0, 4.0, 0.5)),
        (0.0, 0.0, RatioValue("both_zero", 0.0, 0.0, 1.0)),
        (1.0, 0.0, RatioValue("positive_infinity", 1.0, 0.0, None)),
        (None, None, RatioValue("undefined", None, None, None)),
    ],
)
def test_ratio_value_serializes_every_frozen_state(
    numerator: float | None,
    denominator: float | None,
    expected: RatioValue,
) -> None:
    assert ratio_value(numerator, denominator) == expected
    assert "Infinity" not in canonical_json_bytes(expected.to_dict()).decode()


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: ratio_value(None, 1.0),
        lambda: ratio_value(-1.0, 1.0),
        lambda: RatioValue("finite", 1.0, 2.0, 0.4),
        lambda: RatioValue("both_zero", 0.0, 0.0, None),
        lambda: RatioValue("positive_infinity", 1.0, 0.0, float("inf")),
    ],
)
def test_ratio_value_rejects_malformed_or_nonfinite_states(constructor: object) -> None:
    with pytest.raises(ValueError):
        constructor()  # type: ignore[operator]


def test_rate_value_is_integer_exact_and_rejects_bool_or_zero_denominator() -> None:
    assert rate_value(3, 4) == RateValue(3, 4, 0.75)
    with pytest.raises(ValueError):
        rate_value(True, 4)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        rate_value(0, 0)
    with pytest.raises(ValueError):
        RateValue(1, 2, 0.6)


def test_stats_and_linear_p90_are_exact_and_do_not_drop_bad_samples() -> None:
    values = [0.0, 10.0, 20.0, 30.0]
    assert linear_p90(values) == pytest.approx(27.0)
    assert stats_value(values) == StatsValue(
        count=4,
        mean=15.0,
        p50=15.0,
        p90=27.0,
        maximum=30.0,
    )
    assert stats_value([]) == StatsValue(0, None, None, None, None)
    with pytest.raises(ValueError):
        stats_value([1.0, float("nan")])
    with pytest.raises(ValueError):
        stats_value([1.0, -0.1])


def test_upper_ratio_lower_coverage_and_precedence_match_frozen_bands() -> None:
    assert classify_upper_ratio(ratio_value(1.05, 1.0), pass_max=1.05, ambiguous_max=1.10) == "pass"
    assert classify_upper_ratio(ratio_value(1.10, 1.0), pass_max=1.05, ambiguous_max=1.10) == "ambiguous"
    assert classify_upper_ratio(ratio_value(1.1001, 1.0), pass_max=1.05, ambiguous_max=1.10) == "negative"
    assert classify_upper_ratio(ratio_value(1.0, 0.0), pass_max=1.05, ambiguous_max=1.10) == "negative"
    assert classify_upper_ratio(undefined_ratio(), pass_max=1.05, ambiguous_max=1.10) == "ambiguous"

    assert classify_lower_rate(rate_value(95, 100), pass_min=0.95, ambiguous_min=0.90) == "pass"
    assert classify_lower_rate(rate_value(90, 100), pass_min=0.95, ambiguous_min=0.90) == "ambiguous"
    assert classify_lower_rate(rate_value(89, 100), pass_min=0.95, ambiguous_min=0.90) == "negative"
    assert combine_decisions(["pass", "ambiguous", "negative"]) == "negative"


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        (-0.100001, "negative"),
        (-0.10, "ambiguous"),
        (-0.050001, "ambiguous"),
        (-0.05, "pass"),
        (0.0, "pass"),
        (None, "ambiguous"),
    ],
)
def test_boundary_delta_edges(delta: float | None, expected: str) -> None:
    assert classify_boundary_delta(delta) == expected


def test_source_only_schedule_selection_recomputes_common_sets_and_total_order() -> None:
    evaluations = {
        arm: evaluate_source_arm(
            _source_rows(arm, overlap_ms={"S30": 30.0, "S60": 20.0, "S90": 10.0, "S64": 10.0}[arm]),
            schedule_arm=arm,
            worker_lifetime_rss_bytes=(100, 200, 300, 400),
            aggregate_wall_seconds=12.0,
        )
        for arm in ("S30", "S60", "S90", "S64")
    }

    result = select_source_schedule(evaluations)

    assert result.source_decision == "positive"
    assert result.selected_schedule_arm == "S64"
    assert result.overlap_common.count == 16
    assert result.section_common.count == 16
    assert result.overlap_common == audio_set_binding(
        [f"group-{index:02d}" for index in range(16)]
    )
    assert tuple(item.schedule_arm for item in result.arm_order_values) == (
        "S30",
        "S60",
        "S90",
        "S64",
    )
    assert all(item.e0_eligible and item.e1_eligible for item in result.arm_order_values)
    assert all(item.order_tuple_sha256 is not None for item in result.arm_order_values)


def test_source_selection_is_ambiguous_when_fewer_than_two_arms_pass_e0() -> None:
    evaluations = {}
    for arm in ("S30", "S60", "S90", "S64"):
        rows = _source_rows(arm, overlap_ms=10.0)
        if arm != "S64":
            rows = tuple(
                row
                if index >= 2
                else SourceMetricRow(
                    **{
                        **row.__dict__,
                        "candidate_status": "tagged_fallback",
                        "candidate_fallback_reason": "local_frontier_resource_cap_exceeded",
                        "candidate_section_count": None,
                        "candidate_seam_ms": None,
                        "overlap_p90_ms": None,
                    }
                )
                for index, row in enumerate(rows)
            )
        evaluations[arm] = evaluate_source_arm(
            rows,
            schedule_arm=arm,
            worker_lifetime_rss_bytes=(100, 200, 300, 400),
            aggregate_wall_seconds=12.0,
        )

    result = select_source_schedule(evaluations)

    assert result.source_decision == "ambiguous"
    assert result.selected_schedule_arm is None
    assert sum(item.e0_eligible for item in result.arm_order_values) == 1


def test_source_section_common_requires_identical_current_v2_projection() -> None:
    evaluations = {
        arm: evaluate_source_arm(
            _source_rows(
                arm,
                overlap_ms=10.0,
                projection_salt="mismatch" if arm == "S64" else "same",
            ),
            schedule_arm=arm,
            worker_lifetime_rss_bytes=(100, 200, 300, 400),
            aggregate_wall_seconds=12.0,
        )
        for arm in ("S30", "S60", "S90", "S64")
    }

    result = select_source_schedule(evaluations)

    assert result.source_decision == "ambiguous"
    assert result.section_common.count == 0
    assert all(
        "section_common_minimum" in item.elimination_reasons
        for item in result.arm_order_values
    )


def _source_rows(
    arm: str,
    *,
    overlap_ms: float,
    projection_salt: str = "same",
) -> tuple[SourceMetricRow, ...]:
    return tuple(
        SourceMetricRow(
            schedule_arm=arm,  # type: ignore[arg-type]
            row_index=index,
            cache_audio_key=f"audio-{index:02d}",
            audio_group_key=f"group-{index:02d}",
            cache_valid=True,
            projection_evaluable=True,
            candidate_status="accepted",
            candidate_fallback_reason=None,
            baseline_status="accepted",
            selected_status="accepted",
            candidate_section_count=2,
            current_v2_segment_count=1,
            current_v2_projection_sha256=canonical_sha256(
                [projection_salt, index]
            ),
            candidate_seam_ms=0.0,
            overlap_p90_ms=overlap_ms,
            audio_arm_seconds=5.0,
            row_json_bytes=1000,
            replay_schema_source_cache_candidate_v2_consistent=True,
        )
        for index in range(16)
    )
