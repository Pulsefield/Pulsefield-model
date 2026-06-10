from __future__ import annotations

from dataclasses import dataclass

from pulsefield_model.timing.ramp_detection import TimingRampDetection
from pulsefield_model.timing.schema import FittedTimingGrid


@dataclass(frozen=True)
class TimingFitDiagnostics:
    fit_score: float
    selected_period_frames: float
    selected_offset_frames: float
    selected_bpm: float
    candidate_count: int
    half_tempo_score: float
    double_tempo_score: float
    raw_selected_bpm: float
    raw_score: float
    tempo_multiplier: float
    segment_alias_switch_count: int
    tempo_multiplier_distribution: dict[str, int]
    alias_candidate_count: int = 0
    ramp_detection: TimingRampDetection | None = None


@dataclass(frozen=True)
class TimingFitResult:
    grid: FittedTimingGrid
    score: float
    diagnostics: TimingFitDiagnostics


@dataclass(frozen=True)
class _SegmentFit:
    start_frame: int
    end_frame: int
    score: float
    beat_length_ms: float
    offset_ms: float
    half_tempo_score: float
    double_tempo_score: float
    raw_bpm: float
    raw_score: float
    tempo_multiplier: float
    candidate_count: int

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def bpm(self) -> float:
        return 60000.0 / self.beat_length_ms


@dataclass(frozen=True)
class _SplitCandidate:
    frame: int
    score: float


@dataclass(frozen=True)
class _ChangeTimeCandidate:
    time_ms: float
    score: float


@dataclass(frozen=True)
class _EvaluatedSplit:
    segment_index: int
    candidate: _SplitCandidate
    left_fit: _SegmentFit
    right_fit: _SegmentFit
    improvement: float


@dataclass(frozen=True)
class _GridCandidate:
    score: float
    bpm: float
    beat_length_ms: float
    offset_ms: float
