from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from pulsefield_model.timing.canonicalization import (
    TIMING_CANONICALIZATION_BPM_80_160,
    canonicalize_timing_grid,
)
from pulsefield_model.timing.grid_fitting.alias import (
    _alias_bpm_mae,
)
from pulsefield_model.timing.rendering.dense_timing_v2 import (
    DEFAULT_DENSE_TIMING_V2_CONFIG,
    DenseTimingV2Config,
    active_timing_arrays,
    dense_timing_v2_frame_times,
    render_dense_timing_v2,
)
from pulsefield_model.timing.schema import FittedTimingGrid, TimingSegment


_BEAT_PULSE_CHANNEL = 0
_PHASE_SIN_CHANNEL = 1
_PHASE_COS_CHANNEL = 2
_LOCAL_BPM_CHANNEL = 3


@dataclass(frozen=True)
class TimingGridStructuralComparison:
    canonical_oracle_segment_count: int
    predicted_segment_count: int
    abs_canonical_segment_count_delta: int
    predicted_unique_bpm_count: int
    predicted_tempo_family_switch_count: int
    redundant_oracle_segment_count: int
    predicted_alias_switch_count: int | None = None


@dataclass(frozen=True)
class TimingGridComparison:
    frame_count: int
    beat_pulse_mae: float
    local_bpm_mae: float
    local_bpm_alias_mae: float
    mean_phase_error_beats: float
    max_phase_error_beats: float
    mean_phase_error_ms: float
    max_phase_error_ms: float
    canonical_oracle_segment_count: int
    predicted_segment_count: int
    abs_canonical_segment_count_delta: int
    predicted_unique_bpm_count: int
    predicted_tempo_family_switch_count: int
    redundant_oracle_segment_count: int
    predicted_alias_switch_count: int | None = None


def compare_timing_grids(
    predicted_grid: FittedTimingGrid,
    oracle_grid: FittedTimingGrid,
    *,
    frame_count: int,
    input_start_ms: float = 0.0,
    config: DenseTimingV2Config = DEFAULT_DENSE_TIMING_V2_CONFIG,
    predicted_alias_switch_count: int | None = None,
) -> TimingGridComparison:
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count!r}")

    predicted_track = render_dense_timing_v2(
        predicted_grid,
        input_start_ms=input_start_ms,
        frame_count=frame_count,
        config=config,
    )
    oracle_track = render_dense_timing_v2(
        oracle_grid,
        input_start_ms=input_start_ms,
        frame_count=frame_count,
        config=config,
    )

    frame_times_ms = dense_timing_v2_frame_times(
        input_start_ms,
        frame_count=frame_count,
        config=config,
    )
    _, oracle_beat_lengths_ms = active_timing_arrays(oracle_grid, frame_times_ms)

    phase_error_beats = _phase_error_beats(predicted_track, oracle_track)
    phase_error_ms = phase_error_beats * oracle_beat_lengths_ms
    structural_comparison = compare_timing_grid_structure(
        predicted_grid,
        oracle_grid,
        predicted_alias_switch_count=predicted_alias_switch_count,
    )

    return TimingGridComparison(
        frame_count=frame_count,
        beat_pulse_mae=_mean_absolute_channel_delta(
            predicted_track,
            oracle_track,
            _BEAT_PULSE_CHANNEL,
        ),
        local_bpm_mae=_mean_absolute_channel_delta(
            predicted_track,
            oracle_track,
            _LOCAL_BPM_CHANNEL,
        ),
        local_bpm_alias_mae=_alias_bpm_mae(
            predicted_track[:, _LOCAL_BPM_CHANNEL].astype(np.float64),
            oracle_track[:, _LOCAL_BPM_CHANNEL].astype(np.float64),
        ),
        mean_phase_error_beats=float(np.mean(phase_error_beats)),
        max_phase_error_beats=float(np.max(phase_error_beats)),
        mean_phase_error_ms=float(np.mean(phase_error_ms)),
        max_phase_error_ms=float(np.max(phase_error_ms)),
        canonical_oracle_segment_count=structural_comparison.canonical_oracle_segment_count,
        predicted_segment_count=structural_comparison.predicted_segment_count,
        abs_canonical_segment_count_delta=structural_comparison.abs_canonical_segment_count_delta,
        predicted_unique_bpm_count=structural_comparison.predicted_unique_bpm_count,
        predicted_tempo_family_switch_count=structural_comparison.predicted_tempo_family_switch_count,
        redundant_oracle_segment_count=structural_comparison.redundant_oracle_segment_count,
        predicted_alias_switch_count=structural_comparison.predicted_alias_switch_count,
    )


def compare_timing_grid_structure(
    predicted_grid: FittedTimingGrid,
    oracle_grid: FittedTimingGrid,
    *,
    canonicalization: str = TIMING_CANONICALIZATION_BPM_80_160,
    bpm_tolerance: float = 1e-3,
    predicted_alias_switch_count: int | None = None,
) -> TimingGridStructuralComparison:
    if bpm_tolerance < 0.0:
        raise ValueError(f"bpm_tolerance must be non-negative, got {bpm_tolerance!r}")
    if predicted_alias_switch_count is not None and predicted_alias_switch_count < 0:
        raise ValueError(
            f"predicted_alias_switch_count must be non-negative, got {predicted_alias_switch_count!r}"
        )

    canonical_oracle_regions = _canonical_tempo_regions(
        oracle_grid,
        canonicalization=canonicalization,
        bpm_tolerance=bpm_tolerance,
    )
    canonical_oracle_segment_count = len(canonical_oracle_regions)
    predicted_segment_count = len(predicted_grid.segments)

    return TimingGridStructuralComparison(
        canonical_oracle_segment_count=canonical_oracle_segment_count,
        predicted_segment_count=predicted_segment_count,
        abs_canonical_segment_count_delta=abs(
            predicted_segment_count - canonical_oracle_segment_count
        ),
        predicted_unique_bpm_count=_unique_bpm_count(
            (segment.local_bpm for segment in predicted_grid.segments),
            bpm_tolerance=bpm_tolerance,
        ),
        predicted_tempo_family_switch_count=_tempo_family_switch_count(
            predicted_grid,
            canonicalization=canonicalization,
            bpm_tolerance=bpm_tolerance,
        ),
        redundant_oracle_segment_count=len(oracle_grid.segments) - canonical_oracle_segment_count,
        predicted_alias_switch_count=predicted_alias_switch_count,
    )


def _phase_error_beats(
    predicted_track: np.ndarray,
    oracle_track: np.ndarray,
) -> np.ndarray:
    angle_delta = np.arctan2(
        predicted_track[:, _PHASE_SIN_CHANNEL] * oracle_track[:, _PHASE_COS_CHANNEL]
        - predicted_track[:, _PHASE_COS_CHANNEL] * oracle_track[:, _PHASE_SIN_CHANNEL],
        predicted_track[:, _PHASE_COS_CHANNEL] * oracle_track[:, _PHASE_COS_CHANNEL]
        + predicted_track[:, _PHASE_SIN_CHANNEL] * oracle_track[:, _PHASE_SIN_CHANNEL],
    )
    return np.abs(angle_delta) / (2.0 * np.pi)


def _mean_absolute_channel_delta(
    predicted_track: np.ndarray,
    oracle_track: np.ndarray,
    channel_index: int,
) -> float:
    return float(np.mean(np.abs(predicted_track[:, channel_index] - oracle_track[:, channel_index])))


def _canonical_tempo_regions(
    grid: FittedTimingGrid,
    *,
    canonicalization: str,
    bpm_tolerance: float,
) -> tuple[TimingSegment, ...]:
    canonical_grid = canonicalize_timing_grid(grid, canonicalization=canonicalization)
    regions: list[TimingSegment] = []
    for segment in canonical_grid.segments:
        if regions and _bpms_match(regions[-1].local_bpm, segment.local_bpm, bpm_tolerance=bpm_tolerance):
            continue
        regions.append(segment)
    return tuple(regions)


def _tempo_family_switch_count(
    grid: FittedTimingGrid,
    *,
    canonicalization: str,
    bpm_tolerance: float,
) -> int:
    regions = _canonical_tempo_regions(
        grid,
        canonicalization=canonicalization,
        bpm_tolerance=bpm_tolerance,
    )
    return max(0, len(regions) - 1)


def _unique_bpm_count(bpms: Iterable[float], *, bpm_tolerance: float) -> int:
    unique_bpms: list[float] = []
    for bpm in bpms:
        bpm = float(bpm)
        if not any(_bpms_match(bpm, unique_bpm, bpm_tolerance=bpm_tolerance) for unique_bpm in unique_bpms):
            unique_bpms.append(bpm)
    return len(unique_bpms)


def _bpms_match(left: float, right: float, *, bpm_tolerance: float) -> bool:
    return abs(left - right) <= bpm_tolerance
