from __future__ import annotations

import hashlib
import json
import math

import pytest

from pulsefield_model.timing.v3.analytic_curve import (
    ANALYTIC_CURVE_VERSION,
    ConstantTempoSection,
    LinearTimeRampSection,
    PhaseContinuousTimingCurve,
    solve_linear_time_ramp_elapsed_seconds,
)


def test_constant_curve_roundtrips_beat_and_time() -> None:
    curve = PhaseContinuousTimingCurve(
        origin_beat=-2,
        origin_time_ms=250.0,
        sections=(ConstantTempoSection(-2, 6, 120.0),),
    )

    assert curve.curve_class == "constant"
    assert curve.start_beat == -2
    assert curve.end_beat == 6
    assert curve.start_time_ms == 250.0
    assert curve.end_time_ms == pytest.approx(4250.0)
    assert curve.time_at_beat(0.0) == pytest.approx(1250.0)
    assert curve.beat_at_time(1250.0) == pytest.approx(0.0)
    assert curve.bpm_at_time(curve.end_time_ms) == 120.0


@pytest.mark.parametrize(
    ("start_bpm", "end_bpm"),
    [(60.0, 180.0), (240.0, 90.0)],
)
def test_ramp_matches_closed_form_and_both_inverses(
    start_bpm: float,
    end_bpm: float,
) -> None:
    section = LinearTimeRampSection(0, 20, start_bpm, end_bpm)
    curve = PhaseContinuousTimingCurve(0, 1000.0, (section,))
    duration = 120.0 * 20.0 / (start_bpm + end_bpm)
    acceleration = (end_bpm - start_bpm) / duration

    assert section.duration_seconds == pytest.approx(duration)
    assert section.acceleration_bpm_per_second == pytest.approx(acceleration)
    assert curve.end_time_ms == pytest.approx(1000.0 + 1000.0 * duration)
    assert curve.curve_class == "ramp"

    for sample in range(1001):
        elapsed = duration * sample / 1000.0
        expected_beat = (start_bpm * elapsed + 0.5 * acceleration * elapsed**2) / 60.0
        time_ms = 1000.0 + elapsed * 1000.0
        beat = curve.beat_at_time(time_ms)
        assert beat == pytest.approx(expected_beat, abs=1e-11)
        assert curve.bpm_at_time(time_ms) == pytest.approx(
            start_bpm + acceleration * elapsed,
            abs=1e-11,
        )
        assert curve.time_at_beat(beat) == pytest.approx(time_ms, abs=1e-7)

    for sample in range(1001):
        beat = 20.0 * sample / 1000.0
        time_ms = curve.time_at_beat(beat)
        assert curve.beat_at_time(time_ms) == pytest.approx(beat, abs=1e-10)


def test_decreasing_ramp_chooses_root_inside_section() -> None:
    section = LinearTimeRampSection(0, 20, 240.0, 60.0)
    acceleration = section.acceleration_bpm_per_second
    beat_offset = 10.0

    stable = solve_linear_time_ramp_elapsed_seconds(
        beat_offset,
        start_bpm=section.start_bpm,
        acceleration_bpm_per_second=acceleration,
    )
    discriminant = section.start_bpm**2 + 120.0 * acceleration * beat_offset
    root_one = (-section.start_bpm + math.sqrt(discriminant)) / acceleration
    root_two = (-section.start_bpm - math.sqrt(discriminant)) / acceleration

    assert stable == pytest.approx(root_one)
    assert 0.0 <= stable <= section.duration_seconds
    assert root_two > section.duration_seconds


def test_zero_acceleration_inverse_is_exact_constant_formula() -> None:
    for beat_offset in (0.0, 0.125, 1.0, 7.25, 20.0):
        actual = solve_linear_time_ramp_elapsed_seconds(
            beat_offset,
            start_bpm=137.0,
            acceleration_bpm_per_second=0.0,
        )
        assert actual == 60.0 * beat_offset / 137.0


def test_tiny_nonzero_ramps_retain_ramp_identity_and_roundtrip() -> None:
    for end_bpm in (math.nextafter(120.0, math.inf), math.nextafter(120.0, -math.inf)):
        curve = PhaseContinuousTimingCurve(
            0,
            -25.0,
            (LinearTimeRampSection(0, 64, 120.0, end_bpm),),
        )
        restored = PhaseContinuousTimingCurve.from_canonical_bytes(curve.canonical_bytes())

        assert curve.curve_class == "ramp"
        assert restored.curve_class == "ramp"
        assert restored.sections[0].end_bpm == end_bpm
        assert restored.fingerprint_sha256 == curve.fingerprint_sha256
        assert restored.time_at_beat(64.0) == pytest.approx(curve.time_at_beat(64.0), abs=1e-6)


def test_equal_endpoint_ramp_is_rejected_with_constant_instruction() -> None:
    with pytest.raises(ValueError, match="use ConstantTempoSection"):
        LinearTimeRampSection(0, 8, 120.0, 120.0)


def test_seams_share_derived_time_and_queries_select_right_hand_section() -> None:
    curve = PhaseContinuousTimingCurve(
        0,
        100.0,
        (
            ConstantTempoSection(0, 4, 120.0),
            LinearTimeRampSection(4, 12, 120.0, 180.0),
            LinearTimeRampSection(12, 20, 180.0, 90.0),
            ConstantTempoSection(20, 24, 150.0),
        ),
    )

    assert curve.curve_class == "ramp"
    assert [report.beat for report in curve.seam_reports] == [4, 12, 20]
    assert [report.tempo_continuous for report in curve.seam_reports] == [True, True, False]
    assert all(report.phase_discontinuity_ms == 0.0 for report in curve.seam_reports)

    for section_index in range(1, len(curve.sections)):
        seam_beat = curve.sections[section_index].start_beat
        seam_time = curve.boundary_times_ms[section_index]
        assert curve.time_at_beat(seam_beat) == seam_time
        assert curve.beat_at_time(seam_time) == float(seam_beat)
        assert curve.section_at_beat(seam_beat) is curve.sections[section_index]
        assert curve.section_at_time(seam_time) is curve.sections[section_index]
        assert curve.bpm_at_time(seam_time) == curve.sections[section_index].start_bpm

    assert curve.section_at_beat(curve.end_beat) is curve.sections[-1]
    assert curve.section_at_time(curve.end_time_ms) is curve.sections[-1]


def test_two_different_constant_sections_are_a_phase_continuous_jump() -> None:
    curve = PhaseContinuousTimingCurve(
        0,
        0.0,
        (
            ConstantTempoSection(0, 4, 120.0),
            ConstantTempoSection(4, 12, 180.0),
        ),
    )

    assert curve.curve_class == "jump"
    assert curve.boundary_times_ms == pytest.approx((0.0, 2000.0, 2000.0 + 8000.0 / 3.0))
    assert curve.beat_at_time(2000.0) == 4.0
    assert curve.bpm_at_time(2000.0) == 180.0
    assert curve.seam_reports[0].phase_discontinuity_ms == 0.0


def test_same_bpm_constant_neighbors_are_noncanonical() -> None:
    with pytest.raises(ValueError, match="same-BPM.*noncanonical"):
        PhaseContinuousTimingCurve(
            0,
            0.0,
            (
                ConstantTempoSection(0, 4, 120.0),
                ConstantTempoSection(4, 8, 120.0),
            ),
        )


def test_canonical_serialization_has_exact_shape_and_fingerprint() -> None:
    curve = PhaseContinuousTimingCurve(
        -1,
        -0.125,
        (
            ConstantTempoSection(-1, 3, 120.25),
            LinearTimeRampSection(3, 8, 120.25, 177.75),
        ),
    )

    expected_payload = {
        "version": ANALYTIC_CURVE_VERSION,
        "origin_beat": -1,
        "origin_time_ms": "-0.125",
        "sections": [
            {
                "type": "constant",
                "start_beat": -1,
                "end_beat": 3,
                "bpm": "120.25",
            },
            {
                "type": "linear_bpm_time",
                "start_beat": 3,
                "end_beat": 8,
                "start_bpm": "120.25",
                "end_bpm": "177.75",
            },
        ],
    }
    expected_bytes = json.dumps(
        expected_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")

    assert curve.to_dict() == expected_payload
    assert curve.canonical_bytes() == expected_bytes
    assert curve.fingerprint_sha256 == hashlib.sha256(expected_bytes).hexdigest()
    assert curve.canonical_fingerprint_sha256 == curve.fingerprint_sha256

    restored = PhaseContinuousTimingCurve.from_canonical_bytes(expected_bytes)
    assert restored == curve
    assert restored.canonical_bytes() == expected_bytes


@pytest.mark.parametrize(
    "mutator",
    [
        lambda raw: raw + b"\n",
        lambda raw: raw.replace(b'"origin_beat":-1', b'"origin_beat": -1'),
        lambda raw: raw.replace(b'"-0.125"', b'"-0.1250"'),
    ],
)
def test_noncanonical_json_bytes_are_rejected(mutator) -> None:  # type: ignore[no-untyped-def]
    curve = PhaseContinuousTimingCurve(
        -1,
        -0.125,
        (ConstantTempoSection(-1, 3, 120.25),),
    )
    with pytest.raises(ValueError, match="canonical"):
        PhaseContinuousTimingCurve.from_canonical_bytes(mutator(curve.canonical_bytes()))


def test_duplicate_json_keys_and_extra_fields_are_rejected() -> None:
    duplicate = (
        b'{"origin_beat":0,"origin_beat":0,"origin_time_ms":"0",'
        b'"sections":[{"bpm":"120","end_beat":4,"start_beat":0,'
        b'"type":"constant"}],"version":"pulsefield_model.timing_v3_analytic_curve_v1"}'
    )
    with pytest.raises(ValueError, match="strict JSON"):
        PhaseContinuousTimingCurve.from_canonical_bytes(duplicate)

    curve = PhaseContinuousTimingCurve(0, 0.0, (ConstantTempoSection(0, 4, 120.0),))
    payload = curve.to_dict()
    payload["extra"] = True
    with pytest.raises(ValueError, match="exactly keys"):
        PhaseContinuousTimingCurve.from_dict(payload)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True, "120"])
def test_invalid_numeric_inputs_fail_explicitly(value: object) -> None:
    with pytest.raises(ValueError):
        ConstantTempoSection(0, 4, value)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PhaseContinuousTimingCurve(0, value, (ConstantTempoSection(0, 4, 120.0),))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ConstantTempoSection(0, 0, 120.0),
        lambda: LinearTimeRampSection(2, 1, 120.0, 130.0),
        lambda: ConstantTempoSection(0, 4, 19.999),
        lambda: LinearTimeRampSection(0, 4, 120.0, 1000.001),
        lambda: PhaseContinuousTimingCurve(1, 0.0, (ConstantTempoSection(0, 4, 120.0),)),
        lambda: PhaseContinuousTimingCurve(
            0,
            0.0,
            (ConstantTempoSection(0, 4, 120.0), ConstantTempoSection(5, 8, 140.0)),
        ),
    ],
)
def test_invalid_sections_and_curve_topology_fail(factory) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        factory()


def test_queries_outside_closed_domain_fail_and_endpoints_are_allowed() -> None:
    curve = PhaseContinuousTimingCurve(
        3,
        500.0,
        (LinearTimeRampSection(3, 11, 90.0, 180.0),),
    )

    assert curve.time_at_beat(3.0) == 500.0
    assert curve.time_at_beat(11.0) == curve.end_time_ms
    assert curve.beat_at_time(500.0) == 3.0
    assert curve.beat_at_time(curve.end_time_ms) == 11.0

    with pytest.raises(ValueError, match="outside"):
        curve.time_at_beat(math.nextafter(3.0, -math.inf))
    with pytest.raises(ValueError, match="outside"):
        curve.time_at_beat(math.nextafter(11.0, math.inf))
    with pytest.raises(ValueError, match="outside"):
        curve.beat_at_time(math.nextafter(500.0, -math.inf))
    with pytest.raises(ValueError, match="outside"):
        curve.beat_at_time(math.nextafter(curve.end_time_ms, math.inf))


def test_from_dict_requires_canonical_float_strings() -> None:
    curve = PhaseContinuousTimingCurve(0, 0.0, (ConstantTempoSection(0, 4, 120.0),))
    payload = curve.to_dict()
    payload["origin_time_ms"] = 0.0
    with pytest.raises(ValueError, match="canonical float string"):
        PhaseContinuousTimingCurve.from_dict(payload)

    payload = curve.to_dict()
    payload["sections"][0]["bpm"] = "120.0"  # type: ignore[index]
    with pytest.raises(ValueError, match="canonical finite float string"):
        PhaseContinuousTimingCurve.from_dict(payload)
