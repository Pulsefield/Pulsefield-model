from __future__ import annotations

import json
import math

import pytest

from pulsefield_model.timing.v3.schema import (
    TIMING_V3_GRID_SCHEMA,
    ConstantTimingSection,
    TimingV3Grid,
    roundtrip_seam_tolerance_ms,
)


def test_rejects_empty_sections() -> None:
    with pytest.raises(ValueError, match="sections must be non-empty"):
        TimingV3Grid(
            origin_beat=0,
            origin_time_ms=0.0,
            sections=(),
            coverage_start_ms=0.0,
            coverage_end_ms=1.0,
        )


def test_rejects_reversed_and_noncontiguous_sections() -> None:
    with pytest.raises(ValueError, match="end_beat must be greater"):
        ConstantTimingSection(start_beat=4, end_beat=4, bpm=120.0)

    with pytest.raises(ValueError, match="contiguous"):
        TimingV3Grid(
            origin_beat=0,
            origin_time_ms=0.0,
            sections=(
                ConstantTimingSection(start_beat=0, end_beat=4, bpm=120.0),
                ConstantTimingSection(start_beat=5, end_beat=8, bpm=120.0),
            ),
            coverage_start_ms=0.0,
            coverage_end_ms=3000.0,
        )


@pytest.mark.parametrize("bpm", [0.0, -120.0, math.inf, math.nan, 19.999, 1000.001])
def test_rejects_nonfinite_or_out_of_guard_bpm(bpm: float) -> None:
    with pytest.raises(ValueError, match="bpm"):
        ConstantTimingSection(start_beat=0, end_beat=4, bpm=bpm)


def test_rejects_mismatched_origin_and_uncovered_cache_interval() -> None:
    section = ConstantTimingSection(start_beat=1, end_beat=5, bpm=120.0)

    with pytest.raises(ValueError, match="origin_beat must be inside"):
        TimingV3Grid(
            origin_beat=0,
            origin_time_ms=0.0,
            sections=(section,),
            coverage_start_ms=0.0,
            coverage_end_ms=1000.0,
        )

    with pytest.raises(ValueError, match="coverage_start_ms is before"):
        TimingV3Grid(
            origin_beat=0,
            origin_time_ms=10.0,
            sections=(ConstantTimingSection(start_beat=0, end_beat=4, bpm=120.0),),
            coverage_start_ms=0.0,
            coverage_end_ms=1000.0,
        )

    with pytest.raises(ValueError, match="coverage_end_ms is after"):
        TimingV3Grid(
            origin_beat=0,
            origin_time_ms=0.0,
            sections=(ConstantTimingSection(start_beat=0, end_beat=4, bpm=120.0),),
            coverage_start_ms=0.0,
            coverage_end_ms=2000.0011,
        )

    with pytest.raises(ValueError, match="coverage_end_ms must be greater"):
        TimingV3Grid(
            origin_beat=0,
            origin_time_ms=0.0,
            sections=(ConstantTimingSection(start_beat=0, end_beat=4, bpm=120.0),),
            coverage_start_ms=1000.0,
            coverage_end_ms=1000.0,
        )


def test_constant_grid_derives_times_lookup_and_v2_adapter() -> None:
    grid = TimingV3Grid(
        origin_beat=0,
        origin_time_ms=1000.0,
        sections=(ConstantTimingSection(start_beat=0, end_beat=4, bpm=120.0),),
        coverage_start_ms=1000.0,
        coverage_end_ms=3000.0,
    )

    assert grid.section_start_time_ms(0) == pytest.approx(1000.0)
    assert grid.section_end_time_ms(0) == pytest.approx(3000.0)
    assert grid.end_time_ms == pytest.approx(3000.0)
    assert grid.grid_end_time_ms == pytest.approx(3000.0)
    assert grid.time_at_beat(2) == pytest.approx(2000.0)
    assert grid.time_at_beat(4) == pytest.approx(3000.0)
    assert grid.beat_at_time(2250.0) == pytest.approx(2.5)
    assert grid.section_at_beat(3.999) == grid.sections[0]
    assert grid.section_at_time(2999.999) == grid.sections[0]
    assert grid.to_human_lines() == ("beat [0,4) bpm 120",)

    fitted = grid.to_fitted_timing_grid()
    assert len(fitted.segments) == 1
    assert fitted.segments[0].offset_ms == pytest.approx(1000.0)
    assert fitted.segments[0].beat_length_ms == pytest.approx(500.0)


def test_on_lattice_jump_shares_one_boundary_time() -> None:
    grid = TimingV3Grid(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTimingSection(start_beat=0, end_beat=4, bpm=120.0),
            ConstantTimingSection(start_beat=4, end_beat=8, bpm=240.0),
        ),
        coverage_start_ms=0.0,
        coverage_end_ms=3000.0,
    )

    assert grid.section_end_time_ms(0) == pytest.approx(2000.0)
    assert grid.section_start_time_ms(1) == pytest.approx(grid.section_end_time_ms(0))
    assert grid.section_at_beat(4) == grid.sections[1]
    assert grid.section_at_time(2000.0) == grid.sections[1]
    assert grid.time_at_beat(6) == pytest.approx(2500.0)
    assert grid.beat_at_time(2500.0) == pytest.approx(6.0)
    assert grid.to_human_text() == "beat [0,4) bpm 120\nbeat [4,8) bpm 240"


def test_positive_and_negative_origins_are_supported() -> None:
    negative_origin = TimingV3Grid(
        origin_beat=-2,
        origin_time_ms=-1000.0,
        sections=(ConstantTimingSection(start_beat=-2, end_beat=4, bpm=120.0),),
        coverage_start_ms=0.0,
        coverage_end_ms=2000.0,
    )
    assert negative_origin.time_at_beat(0) == pytest.approx(0.0)
    assert negative_origin.beat_at_time(0.0) == pytest.approx(0.0)

    positive_origin = TimingV3Grid(
        origin_beat=3,
        origin_time_ms=250.0,
        sections=(ConstantTimingSection(start_beat=3, end_beat=7, bpm=120.0),),
        coverage_start_ms=250.0,
        coverage_end_ms=2250.0,
    )
    assert positive_origin.time_at_beat(3) == pytest.approx(250.0)
    assert positive_origin.beat_at_time(1250.0) == pytest.approx(5.0)


def test_source_anchor_origin_may_be_inside_backward_extended_first_section() -> None:
    grid = TimingV3Grid(
        origin_beat=0,
        origin_time_ms=250.0,
        sections=(ConstantTimingSection(start_beat=-1, end_beat=4, bpm=120.0),),
        coverage_start_ms=0.0,
        coverage_end_ms=2000.0,
    )

    assert grid.start_beat == -1
    assert grid.start_time_ms == pytest.approx(-250.0)
    assert grid.time_at_beat(0) == pytest.approx(250.0)
    assert grid.beat_at_time(250.0) == pytest.approx(0.0)

    restored = TimingV3Grid.from_dict(json.loads(json.dumps(grid.to_dict())))
    assert restored.origin_beat == 0
    assert restored.origin_time_ms == pytest.approx(250.0)
    assert restored.start_time_ms == pytest.approx(-250.0)
    assert restored.time_at_beat(0) == pytest.approx(250.0)


@pytest.mark.parametrize("bpm,beat_count", [(20.0, 1453), (1000.0, 72634)])
def test_json_roundtrip_preserves_long_coverage_and_boundary_seams(bpm: float, beat_count: int) -> None:
    grid = TimingV3Grid(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(ConstantTimingSection(start_beat=0, end_beat=beat_count, bpm=bpm),),
        coverage_start_ms=0.0,
        coverage_end_ms=4_358_000.0,
    )

    payload = grid.to_dict()
    assert payload["schema"] == TIMING_V3_GRID_SCHEMA
    assert set(payload["sections"][0]) == {"start_beat", "end_beat", "bpm"}

    restored = TimingV3Grid.from_dict(json.loads(json.dumps(payload)))

    for original_time_ms, restored_time_ms in zip(grid.boundary_times_ms, restored.boundary_times_ms):
        assert abs(restored_time_ms - original_time_ms) <= roundtrip_seam_tolerance_ms(original_time_ms)


def test_from_dict_rejects_wrong_schema_version() -> None:
    payload = TimingV3Grid(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(ConstantTimingSection(start_beat=0, end_beat=4, bpm=120.0),),
        coverage_start_ms=0.0,
        coverage_end_ms=2000.0,
    ).to_dict()

    bad_schema = dict(payload, schema="pulsefield_model.timing_v3_grid_v2")
    with pytest.raises(ValueError, match="schema"):
        TimingV3Grid.from_dict(bad_schema)

    bad_version = dict(payload, version=2)
    with pytest.raises(ValueError, match="version"):
        TimingV3Grid.from_dict(bad_version)

    for malformed_version in (True, 1.0):
        with pytest.raises(ValueError, match="version must be an integer"):
            TimingV3Grid.from_dict(dict(payload, version=malformed_version))
