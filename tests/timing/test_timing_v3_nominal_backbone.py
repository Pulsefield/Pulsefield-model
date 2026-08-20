from __future__ import annotations

from typing import Literal

from pulsefield_model.timing.v3 import tempo_track as tt


def _observation(
    source: Literal["beatthis", "raw_audio"],
    *,
    center_ms: float,
    bpm: float,
) -> tt.LocalTempoObservation:
    return tt.LocalTempoObservation(
        source=source,
        center_time_ms=center_ms,
        window_start_ms=center_ms - 1500.0,
        window_end_ms=center_ms + 1500.0,
        bpm=bpm,
        strength=1.0,
    )


def test_nominal_backbone_uses_one_alias_family_for_persistent_step() -> None:
    left_centers = tuple(2000.0 + 500.0 * index for index in range(32))
    right_centers = tuple(22_000.0 + 500.0 * index for index in range(32))
    beatthis = tuple(
        _observation("beatthis", center_ms=center, bpm=77.5)
        for center in left_centers
    ) + tuple(
        _observation("beatthis", center_ms=center, bpm=90.5)
        for center in right_centers
    )
    raw = tuple(
        _observation("raw_audio", center_ms=center, bpm=155.0)
        for center in left_centers
    ) + tuple(
        _observation("raw_audio", center_ms=center, bpm=181.0)
        for center in right_centers
    )

    assert tt._nominal_backbone_bpms(
        beatthis,
        raw,
        boundary_times_ms=(20_000.0,),
        duration_ms=40_000.0,
        config=tt.DEFAULT_TEMPO_TRACK_CONFIG,
    ) == (155.0, 181.0)
    assert tt._nominal_backbone_has_minimum_total_delta((155.0, 181.0))
    assert not tt._nominal_backbone_has_minimum_total_delta((86.0, 97.0))
    assert tt._nominal_backbone_has_minimum_total_delta(
        (182.0, 184.0, 187.0, 182.0, 184.0, 186.0)
    )


def test_nominal_backbone_requires_raw_count_and_four_positive_edge_gains() -> None:
    curve = tt._nominal_backbone_curve(
        (120.0, 150.0),
        (20_000.0,),
        phase_ms=0.0,
        duration_ms=40_000.0,
    )
    assert curve is not None
    left_centers = tuple(2000.0 + 500.0 * index for index in range(32))
    right_centers = tuple(22_000.0 + 500.0 * index for index in range(32))
    raw = tuple(
        _observation("raw_audio", center_ms=center, bpm=120.0)
        for center in left_centers
    ) + tuple(
        _observation("raw_audio", center_ms=center, bpm=150.0)
        for center in right_centers
    )
    beatthis = tuple(
        _observation("beatthis", center_ms=center, bpm=120.0)
        for center in left_centers
    ) + tuple(
        _observation("beatthis", center_ms=center, bpm=150.0)
        for center in right_centers
    )

    counts = tt._nominal_backbone_raw_support_counts(
        curve,
        raw=raw,
        config=tt.DEFAULT_TEMPO_TRACK_CONFIG,
    )
    gains = tt._nominal_backbone_edge_gains(
        curve,
        edge_index=0,
        beatthis=beatthis,
        raw=raw,
        config=tt.DEFAULT_TEMPO_TRACK_CONFIG,
    )

    assert min(counts) >= tt._NOMINAL_BACKBONE_MINIMUM_RAW_SUPPORT
    assert min(gains) > 0.0
    sparse_raw = tuple(value for value in raw if value.center_time_ms >= 14_000.0)
    sparse_counts = tt._nominal_backbone_raw_support_counts(
        curve,
        raw=sparse_raw,
        config=tt.DEFAULT_TEMPO_TRACK_CONFIG,
    )
    assert sparse_counts[0] < tt._NOMINAL_BACKBONE_MINIMUM_RAW_SUPPORT
