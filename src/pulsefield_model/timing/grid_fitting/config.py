from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

import numpy as np

from pulsefield_model.timing.canonicalization import DEFAULT_TIMING_CANONICALIZATION
from pulsefield_model.timing.canonicalization import require_timing_canonicalization


@dataclass(frozen=True)
class GridFitterConfig:
    min_bpm: float = 20.0
    max_bpm: float = 1000.0
    bpm_step: float = 0.5
    offset_step_ms: float = 20.0
    pulse_width_ms: float = 40.0
    double_tempo_score_ratio_threshold: float = 0.95
    max_segments: int = 16
    min_segment_duration_ms: float = 8000.0
    split_step_ms: float = 4000.0
    split_score_improvement_threshold: float = 0.02
    split_phase_change_threshold_ms: float = 10.0
    super_timing_split_candidates: bool = True
    split_relative_interval_change_threshold: float = 0.025
    split_downbeat_signal_weight: float = 0.35
    autocorrelation_candidate_count: int = 16
    bpm_search_window_ratio: float = 0.08
    bpm_search_window_min_bpm: float = 2.0
    max_grid_candidates_per_segment: int = 1000
    max_split_candidates_per_segment: int = 4
    initial_batch_split_candidate_count: int = 16
    initial_batch_split_min_candidate_count: int = 8
    initial_batch_split_max_parent_score: float = 0.75
    long_prediction_duration_seconds: float = 600.0
    long_max_segments: int = 20
    long_max_grid_candidates_per_segment: int = 1000
    long_max_split_candidates_per_segment: int = 5
    long_downbeat_refine_candidate_count: int = 20
    downbeat_period_beats: int = 4
    downbeat_tie_score_margin: float = 0.05
    downbeat_split_score_bonus: float = 0.5
    downbeat_refine_candidate_count: int = 16
    merge_similar_segments: bool = True
    merge_bpm_tolerance: float = 1.5
    merge_relative_bpm_tolerance: float = 0.005
    merge_phase_tolerance_ms: float = 35.0
    merge_many_similar_min_segments: int = 4
    merge_many_similar_bpm_tolerance: float = 2.0
    merge_alias_min_segments: int = 4
    merge_alias_requires_downbeat_signal: bool = True
    merge_alias_bpm_tolerance: float = 2.0
    merge_alias_phase_tolerance_ms: float = 60.0
    merge_alias_max_fit_score: float = 0.92
    canonicalization: str = DEFAULT_TIMING_CANONICALIZATION
    canonicalize_tempo_aliases: bool = True
    alias_tempo_multipliers: tuple[float, ...] = (0.25, 1.0 / 3.0, 0.5, 1.0, 2.0, 3.0, 4.0)
    alias_score_tie_margin: float = 0.03
    alias_score_ratio_threshold: float = 0.97
    alias_preferred_min_bpm: float = 80.0
    alias_preferred_max_bpm: float = 240.0
    alias_preferred_band_bonus: float = 0.015
    alias_current_tempo_bonus: float = 0.04
    alias_downbeat_score_weight: float = 0.02
    alias_continuity_penalty: float = 0.02
    alias_semantic_promotion_in_band_min_bpm: float = 86.0
    alias_semantic_promotion_current_max_bpm: float = 100.0
    alias_semantic_promotion_score_ratio_threshold: float = 0.65
    alias_semantic_promotion_low_bpm_max_fit_score: float = 0.78
    alias_semantic_promotion_strong_score_ratio_threshold: float = 0.78
    alias_semantic_promotion_low_confidence_max_fit_score: float = 0.70
    alias_semantic_promotion_low_confidence_score_ratio_threshold: float = 0.60
    alias_semantic_promotion_low_confidence_max_candidate_bpm: float = 185.0
    alias_semantic_promotion_low_bpm_max_segments: int = 4
    alias_semantic_promotion_bonus: float = 0.35
    alias_collapse_score_ratio_threshold: float = 0.78
    alias_demotion_dropped_support_ratio_threshold: float = 0.35
    alias_promotion_inserted_support_ratio_threshold: float = 0.35
    alias_beat_match_tolerance_ms: float = 45.0

    def __post_init__(self) -> None:
        _require_positive_finite(self, _POSITIVE_FINITE_FIELDS)
        _require_nonnegative_finite(self, _NONNEGATIVE_FINITE_FIELDS)
        _require_positive(self, _POSITIVE_COUNT_FIELDS)
        _require_nonnegative(self, _NONNEGATIVE_COUNT_FIELDS)
        require_timing_canonicalization(self.canonicalization)
        _require_alias_tempo_multipliers(self.alias_tempo_multipliers)

        if not np.isfinite(self.max_bpm) or self.max_bpm <= self.min_bpm:
            raise ValueError(f"max_bpm must be finite and greater than min_bpm, got {self.max_bpm!r}")
        if self.alias_preferred_max_bpm <= self.alias_preferred_min_bpm:
            raise ValueError(
                "alias_preferred_max_bpm must be greater than alias_preferred_min_bpm, "
                f"got {self.alias_preferred_max_bpm!r} <= {self.alias_preferred_min_bpm!r}",
            )


_POSITIVE_FINITE_FIELDS: Final[tuple[str, ...]] = (
    "min_bpm",
    "bpm_step",
    "offset_step_ms",
    "pulse_width_ms",
    "min_segment_duration_ms",
    "split_step_ms",
)

_NONNEGATIVE_FINITE_FIELDS: Final[tuple[str, ...]] = (
    "double_tempo_score_ratio_threshold",
    "split_score_improvement_threshold",
    "split_phase_change_threshold_ms",
    "split_relative_interval_change_threshold",
    "split_downbeat_signal_weight",
    "bpm_search_window_ratio",
    "bpm_search_window_min_bpm",
    "initial_batch_split_max_parent_score",
    "long_prediction_duration_seconds",
    "downbeat_tie_score_margin",
    "downbeat_split_score_bonus",
    "merge_bpm_tolerance",
    "merge_relative_bpm_tolerance",
    "merge_phase_tolerance_ms",
    "merge_many_similar_bpm_tolerance",
    "merge_alias_bpm_tolerance",
    "merge_alias_phase_tolerance_ms",
    "merge_alias_max_fit_score",
    "alias_score_tie_margin",
    "alias_score_ratio_threshold",
    "alias_preferred_min_bpm",
    "alias_preferred_max_bpm",
    "alias_preferred_band_bonus",
    "alias_current_tempo_bonus",
    "alias_downbeat_score_weight",
    "alias_continuity_penalty",
    "alias_semantic_promotion_in_band_min_bpm",
    "alias_semantic_promotion_current_max_bpm",
    "alias_semantic_promotion_score_ratio_threshold",
    "alias_semantic_promotion_low_bpm_max_fit_score",
    "alias_semantic_promotion_strong_score_ratio_threshold",
    "alias_semantic_promotion_low_confidence_max_fit_score",
    "alias_semantic_promotion_low_confidence_score_ratio_threshold",
    "alias_semantic_promotion_low_confidence_max_candidate_bpm",
    "alias_semantic_promotion_bonus",
    "alias_collapse_score_ratio_threshold",
    "alias_demotion_dropped_support_ratio_threshold",
    "alias_promotion_inserted_support_ratio_threshold",
    "alias_beat_match_tolerance_ms",
)

_POSITIVE_COUNT_FIELDS: Final[tuple[str, ...]] = (
    "max_segments",
    "autocorrelation_candidate_count",
    "max_grid_candidates_per_segment",
    "max_split_candidates_per_segment",
    "long_max_segments",
    "long_max_grid_candidates_per_segment",
    "long_max_split_candidates_per_segment",
    "long_downbeat_refine_candidate_count",
    "downbeat_period_beats",
    "downbeat_refine_candidate_count",
    "merge_many_similar_min_segments",
    "merge_alias_min_segments",
    "alias_semantic_promotion_low_bpm_max_segments",
)

_NONNEGATIVE_COUNT_FIELDS: Final[tuple[str, ...]] = (
    "initial_batch_split_candidate_count",
    "initial_batch_split_min_candidate_count",
)


def _require_positive_finite(config: GridFitterConfig, fields: tuple[str, ...]) -> None:
    for field in fields:
        value = getattr(config, field)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{field} must be positive and finite, got {value!r}")


def _require_nonnegative_finite(config: GridFitterConfig, fields: tuple[str, ...]) -> None:
    for field in fields:
        value = getattr(config, field)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{field} must be non-negative and finite, got {value!r}")


def _require_positive(config: GridFitterConfig, fields: tuple[str, ...]) -> None:
    for field in fields:
        value = getattr(config, field)
        if value <= 0:
            raise ValueError(f"{field} must be positive, got {value!r}")


def _require_nonnegative(config: GridFitterConfig, fields: tuple[str, ...]) -> None:
    for field in fields:
        value = getattr(config, field)
        if value < 0:
            raise ValueError(f"{field} must be non-negative, got {value!r}")


def _require_alias_tempo_multipliers(multipliers: tuple[float, ...]) -> None:
    if not multipliers:
        raise ValueError("alias_tempo_multipliers must be non-empty")
    if 1.0 not in multipliers:
        raise ValueError("alias_tempo_multipliers must include 1.0")
    for multiplier in multipliers:
        if not np.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError(f"alias_tempo_multipliers must be positive and finite, got {multiplier!r}")


def _effective_config_for_prediction(
    frame_count: int,
    *,
    frame_rate_hz: float,
    config: GridFitterConfig,
) -> GridFitterConfig:
    duration_seconds = float(frame_count) / frame_rate_hz
    if duration_seconds < config.long_prediction_duration_seconds:
        return config
    return replace(
        config,
        max_segments=max(config.max_segments, config.long_max_segments),
        max_grid_candidates_per_segment=max(
            config.max_grid_candidates_per_segment,
            config.long_max_grid_candidates_per_segment,
        ),
        max_split_candidates_per_segment=max(
            config.max_split_candidates_per_segment,
            config.long_max_split_candidates_per_segment,
        ),
        downbeat_refine_candidate_count=max(
            config.downbeat_refine_candidate_count,
            config.long_downbeat_refine_candidate_count,
        ),
    )
