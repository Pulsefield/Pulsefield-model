from __future__ import annotations

import math
from typing import Literal

import numpy as np

from pulsefield_model.timing.v3.analytic_curve import (
    ConstantTempoSection,
    PhaseContinuousTimingCurve,
)
from pulsefield_model.timing.v3.audio_evidence import (
    CandidateRawAudioScore,
    RawAudioEvidence,
    RawAudioEvidenceRanking,
)
from pulsefield_model.timing.v3.tempo_track import (
    _BaseHypothesis,
    _collapsed_constant_counterfactual,
    _section_matches_raw_run,
    _select_exp014_production_candidate,
    LocalTempoObservation,
    TempoTrackConfig,
    TempoTrackRawRunDiagnostic,
)


def test_exp018_shifted_closed_aba_middle_run_uses_edge_tolerance() -> None:
    curve = _closed_aba()

    assert _matches_middle(curve, raw_run=_shifted_middle_run(curve))


def test_exp018_shifted_closed_aba_still_rejects_beyond_edge_tolerance() -> None:
    curve = _closed_aba()

    assert not _matches_middle(
        curve,
        raw_run=_shifted_middle_run(curve, start_offset_ms=-4000.1),
    )
    assert not _matches_middle(
        curve,
        raw_run=_shifted_middle_run(curve, end_offset_ms=4000.1),
    )


def test_exp018_closed_aba_keeps_direction_tempo_ratio_and_overlap_guards() -> None:
    curve = _closed_aba()

    assert not _matches_middle(
        curve,
        raw_run=_shifted_middle_run(curve, direction="up"),
    )
    assert not _matches_middle(
        curve,
        raw_run=_shifted_middle_run(curve, median_bpm=120.0),
    )
    assert not _matches_middle(
        curve,
        raw_run=_raw_run(
            direction="down",
            start_time_ms=10_000.0,
            end_time_ms=14_000.0,
            median_bpm=150.0,
        ),
    )


def test_exp018_persistent_tail_stays_on_tight_anchor_tolerance() -> None:
    curve = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 60, 180.0),
            ConstantTempoSection(60, 210, 150.0),
        ),
    )
    tail = curve.sections[1]
    assert isinstance(tail, ConstantTempoSection)
    section_start_ms = curve.time_at_beat(float(tail.start_beat))

    assert not _section_matches_raw_run(
        curve,
        tail,
        section_index=1,
        raw_run=_raw_run(
            direction="down",
            start_time_ms=section_start_ms + 2500.0,
            end_time_ms=section_start_ms + 3500.0,
            median_bpm=150.0,
        ),
        raw_observations=(),
        primary=_primary(),
        audio_duration_ms=curve.end_time_ms,
        config=TempoTrackConfig(),
    )


def test_exp018_alias_consistent_but_unequal_outer_bpm_stays_strict() -> None:
    curve = _closed_aba(first_bpm=90.0, first_beats=30, last_bpm=180.0)

    assert not _matches_middle(curve, raw_run=_shifted_middle_run(curve))


def test_exp018_non_aba_and_long_aba_topologies_stay_strict() -> None:
    multi_step = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 60, 180.0),
            ConstantTempoSection(60, 75, 150.0),
            ConstantTempoSection(75, 85, 165.0),
            ConstantTempoSection(85, 180, 180.0),
        ),
    )
    long_aba = _closed_aba(middle_beats=25)

    assert not _matches_middle(multi_step, raw_run=_shifted_middle_run(multi_step))
    assert not _matches_middle(long_aba, raw_run=_shifted_middle_run(long_aba))


def test_exp018_selector_adds_only_shifted_closed_aba_and_keeps_gain_ranking() -> None:
    config = TempoTrackConfig()
    constant = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(ConstantTempoSection(0, 180, 180.0),),
    )
    strict_aba = _closed_aba(first_beats=66, middle_beats=5)
    shifted_aba = _closed_aba()
    persistent = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 60, 180.0),
            ConstantTempoSection(60, 210, 150.0),
        ),
    )
    candidates = (constant, strict_aba, shifted_aba, persistent)
    duration_ms = max(candidate.end_time_ms for candidate in candidates)
    collapsed = tuple(
        _collapsed_constant_counterfactual(
            candidate,
            duration_ms=duration_ms,
            base_bpm=180.0,
            config=config,
        )
        for candidate in candidates
    )

    selection = _select_exp014_production_candidate(
        candidates,
        raw_self_ranking=_ranking(candidates, raw_scores=(0.40, 0.70, 0.90, 1.00)),
        collapsed_raw_self_ranking=_ranking(
            collapsed,
            raw_scores=(0.40, 0.20, 0.20, 0.10),
        ),
        collapsed_counterfactuals=collapsed,
        observations=_raw_run_observations(),
        primary=_primary(),
        beat_signal=np.zeros(int(math.ceil(duration_ms * 50.0 / 1000.0))),
        frame_rate_hz=50.0,
        config=config,
    )

    gain_by_index = {
        index: raw_gain
        for index, raw_gain, _, _ in selection.paired_raw_gain_by_candidate
    }
    assert selection.status == "v3_accepted"
    assert selection.lane == "paired_jump"
    assert selection.eligible_candidate_indices == (1, 2)
    assert gain_by_index[3] > gain_by_index[2] > gain_by_index[1]
    assert selection.selected_candidate_index == 2


def _closed_aba(
    *,
    first_bpm: float = 180.0,
    first_beats: int = 60,
    middle_beats: int = 15,
    middle_bpm: float = 150.0,
    last_bpm: float = 180.0,
) -> PhaseContinuousTimingCurve:
    middle_start = first_beats
    middle_end = middle_start + middle_beats
    return PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, middle_start, first_bpm),
            ConstantTempoSection(middle_start, middle_end, middle_bpm),
            ConstantTempoSection(middle_end, 180, last_bpm),
        ),
    )


def _matches_middle(
    curve: PhaseContinuousTimingCurve,
    *,
    raw_run: TempoTrackRawRunDiagnostic,
) -> bool:
    section = curve.sections[1]
    assert isinstance(section, ConstantTempoSection)
    return _section_matches_raw_run(
        curve,
        section,
        section_index=1,
        raw_run=raw_run,
        raw_observations=(),
        primary=_primary(),
        audio_duration_ms=curve.end_time_ms,
        config=TempoTrackConfig(),
    )


def _shifted_middle_run(
    curve: PhaseContinuousTimingCurve,
    *,
    start_offset_ms: float = 2500.0,
    end_offset_ms: float = -2500.0,
    direction: Literal["up", "down"] = "down",
    median_bpm: float = 150.0,
) -> TempoTrackRawRunDiagnostic:
    middle = curve.sections[1]
    assert isinstance(middle, ConstantTempoSection)
    return _raw_run(
        direction=direction,
        start_time_ms=curve.time_at_beat(float(middle.start_beat)) + start_offset_ms,
        end_time_ms=curve.time_at_beat(float(middle.end_beat)) + end_offset_ms,
        median_bpm=median_bpm,
    )


def _raw_run(
    *,
    direction: Literal["up", "down"],
    start_time_ms: float,
    end_time_ms: float,
    median_bpm: float,
) -> TempoTrackRawRunDiagnostic:
    return TempoTrackRawRunDiagnostic(
        direction=direction,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        expanded_start_time_ms=start_time_ms - 1000.0,
        expanded_end_time_ms=end_time_ms + 1000.0,
        median_bpm=median_bpm,
        weighted_median_delta_bpm=100.0 if direction == "up" else -100.0,
        observation_count=3,
        summed_strength=1.0,
    )


def _raw_run_observations() -> tuple[LocalTempoObservation, ...]:
    return tuple(
        LocalTempoObservation(
            source="raw_audio",
            center_time_ms=center_time_ms,
            window_start_ms=center_time_ms - 3000.0,
            window_end_ms=center_time_ms + 3000.0,
            bpm=150.0,
            strength=0.5,
        )
        for center_time_ms in (22_500.0, 23_000.0, 23_500.0)
    )


def _ranking(
    candidates: tuple[PhaseContinuousTimingCurve, ...],
    *,
    raw_scores: tuple[float, ...],
) -> RawAudioEvidenceRanking:
    scores = tuple(
        CandidateRawAudioScore(
            candidate_index=index,
            fingerprint_sha256=candidate.fingerprint_sha256,
            raw_score=raw_score,
            mean_beat_support=raw_score,
            mean_half_beat_support=0.0,
            window_contrast_p10=raw_score,
            retained_beat_count=max(16, candidate.end_beat - candidate.start_beat),
            complete_window_count=1,
            unavailable_reason=None,
            candidate_domain_beat_count=max(
                16,
                candidate.end_beat - candidate.start_beat,
            ),
            complete_window_start_beats=(candidate.start_beat,),
        )
        for index, (candidate, raw_score) in enumerate(zip(candidates, raw_scores))
    )
    return RawAudioEvidenceRanking(
        evidence=_evidence(),
        candidate_scores=scores,
        ranked_scores=tuple(
            sorted(
                scores,
                key=lambda score: (-score.raw_score, score.fingerprint_sha256),
            )
        ),
        common_beat_indices=(),
        complete_window_start_beats=(),
        unavailable_reason=None,
    )


def _evidence() -> RawAudioEvidence:
    centers = np.asarray((0.01, 0.02, 0.03), dtype=np.float64)
    return RawAudioEvidence(
        frame_center_seconds=centers,
        band_flux=np.zeros((centers.size, 4), dtype=np.float32),
        band_percentile95=(1.0, 1.0, 1.0, 1.0),
        input_frame_count=centers.size,
        valid_frame_count=centers.size,
        audio_duration_seconds=0.04,
    )


def _primary() -> _BaseHypothesis:
    return _BaseHypothesis(origin_time_ms=0.0, bpm=180.0, score=1.0)
