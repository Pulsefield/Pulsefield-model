from __future__ import annotations

import math

import numpy as np

from pulsefield_model.timing.schema import FrameTimingPrediction
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
    _CurveProposal,
    _collapsed_constant_counterfactual,
    _collapsed_counterfactuals_for_proposals,
    _raw_run_jump_proposals,
    _score_collapsed_counterfactuals_independent,
    _select_exp014_production_candidate,
    generate_timing_candidates,
    LocalTempoObservation,
    tempo_track_result_to_dict,
    TempoTrackConfig,
)


def test_exp014_collapsed_constant_uses_primary_full_duration_domain() -> None:
    candidate = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 10, 120.0),
            ConstantTempoSection(10, 40, 180.0),
        ),
    )

    collapsed = _collapsed_constant_counterfactual(
        candidate,
        duration_ms=candidate.end_time_ms,
        base_bpm=120.0,
        config=TempoTrackConfig(),
    )

    assert collapsed.curve_class == "constant"
    assert collapsed.sections[0].bpm == 120.0
    assert collapsed.end_beat == 30
    assert collapsed.end_beat != candidate.end_beat
    assert collapsed.end_time_ms >= candidate.end_time_ms


def test_exp014_collapsed_counterfactuals_use_candidate_alias_base() -> None:
    candidate = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 18, 90.0),
            ConstantTempoSection(18, 30, 120.0),
        ),
    )
    proposal = _CurveProposal(
        curve=candidate,
        source="raw_run_persistent_a_to_b_start",
        score=1.0,
        collapse_bpm=90.0,
    )

    collapsed = _collapsed_counterfactuals_for_proposals(
        (proposal,),
        duration_ms=60_000.0,
        config=TempoTrackConfig(),
    )[0]
    wrong_primary_alias = _collapsed_constant_counterfactual(
        candidate,
        duration_ms=60_000.0,
        base_bpm=180.0,
        config=TempoTrackConfig(),
    )

    assert collapsed.sections[0].bpm == 90.0
    assert collapsed.end_beat == 90
    assert collapsed.fingerprint_sha256 != wrong_primary_alias.fingerprint_sha256


def test_exp014_duplicate_collapsed_fingerprints_score_independently() -> None:
    first = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 16, 180.0),
            ConstantTempoSection(16, 24, 150.0),
            ConstantTempoSection(24, 96, 180.0),
        ),
    )
    second = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 20, 180.0),
            ConstantTempoSection(20, 28, 150.0),
            ConstantTempoSection(28, 96, 180.0),
        ),
    )
    duration_ms = max(first.end_time_ms, second.end_time_ms)
    collapsed = tuple(
        _collapsed_constant_counterfactual(
            candidate,
            duration_ms=duration_ms,
            base_bpm=180.0,
            config=TempoTrackConfig(),
        )
        for candidate in (first, second)
    )

    assert collapsed[0].fingerprint_sha256 == collapsed[1].fingerprint_sha256
    ranking = _score_collapsed_counterfactuals_independent(
        _raw_evidence_from_curve(collapsed[0], duration_ms=duration_ms),
        collapsed,
    )

    assert [score.candidate_index for score in ranking.candidate_scores] == [0, 1]
    assert len(ranking.ranked_scores) == 2


def test_exp014_raw_run_generation_includes_persistent_and_long_aba() -> None:
    config = TempoTrackConfig()
    observations = _raw_run_observations(
        centers_ms=tuple(float(value) for value in range(20_000, 51_000, 1000)),
        bpm=150.0,
    )

    proposals = _raw_run_jump_proposals(
        np.zeros(5000, dtype=np.float64),
        frame_rate_hz=50.0,
        duration_ms=80_000.0,
        bases=(_BaseHypothesis(origin_time_ms=0.0, bpm=180.0, score=1.0),),
        observations=observations,
        config=config,
    )

    assert any(proposal.source.startswith("raw_run_persistent") for proposal in proposals)
    assert any(
        proposal.source == "raw_run_aba"
        and len(proposal.curve.sections) == 3
        and proposal.curve.sections[1].duration_seconds >= 25.0
        for proposal in proposals
    )
    assert all(proposal.curve.end_time_ms >= 80_000.0 for proposal in proposals)


def test_exp014_persistent_a_to_b_raw_run_need_not_reach_audio_end() -> None:
    config = TempoTrackConfig()
    duration_ms = 80_000.0
    constant = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(ConstantTempoSection(0, 240, 180.0),),
    )
    jump = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 60, 180.0),
            ConstantTempoSection(60, 210, 150.0),
        ),
    )
    collapsed = (
        constant,
        _collapsed_constant_counterfactual(
            jump,
            duration_ms=duration_ms,
            base_bpm=180.0,
            config=config,
        ),
    )

    selection = _select_exp014_production_candidate(
        (constant, jump),
        raw_self_ranking=_ranking((constant, jump), raw_scores=(0.25, 1.0)),
        collapsed_raw_self_ranking=_ranking(collapsed, raw_scores=(0.25, 0.1)),
        collapsed_counterfactuals=collapsed,
        observations=_raw_run_observations(
            centers_ms=tuple(float(value) for value in range(20_000, 51_000, 1000)),
            bpm=150.0,
        ),
        primary=_BaseHypothesis(origin_time_ms=0.0, bpm=180.0, score=1.0),
        beat_signal=np.zeros(int(duration_ms * 50.0 / 1000.0), dtype=np.float64),
        frame_rate_hz=50.0,
        config=config,
    )

    assert selection.status == "v3_accepted"
    assert selection.lane == "paired_jump"
    assert selection.selected_candidate_index == 1


def test_exp014_mid_song_raw_run_does_not_support_false_b_to_a_from_start() -> None:
    config = TempoTrackConfig()
    duration_ms = 80_000.0
    constant = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(ConstantTempoSection(0, 240, 180.0),),
    )
    false_b_to_a = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 60, 150.0),
            ConstantTempoSection(60, 228, 180.0),
        ),
    )
    collapsed = (
        constant,
        _collapsed_constant_counterfactual(
            false_b_to_a,
            duration_ms=duration_ms,
            base_bpm=180.0,
            config=config,
        ),
    )

    selection = _select_exp014_production_candidate(
        (constant, false_b_to_a),
        raw_self_ranking=_ranking((constant, false_b_to_a), raw_scores=(0.25, 1.0)),
        collapsed_raw_self_ranking=_ranking(collapsed, raw_scores=(0.25, 0.1)),
        collapsed_counterfactuals=collapsed,
        observations=_raw_run_observations(
            centers_ms=(10_000.0, 11_000.0, 12_000.0),
            bpm=180.0,
        )
        + _raw_run_observations(
            centers_ms=(20_000.0, 21_000.0, 22_000.0, 23_000.0, 24_000.0),
            bpm=150.0,
        ),
        primary=_BaseHypothesis(origin_time_ms=0.0, bpm=180.0, score=1.0),
        beat_signal=np.zeros(int(duration_ms * 50.0 / 1000.0), dtype=np.float64),
        frame_rate_hz=50.0,
        config=config,
    )

    assert selection.status == "v3_accepted"
    assert selection.lane == "constant"
    assert selection.selected_candidate_index == 0


def test_exp014_raw_run_generation_includes_adjacent_chain() -> None:
    config = TempoTrackConfig()
    observations = _raw_run_observations(
        centers_ms=(10_000.0, 11_000.0, 12_000.0),
        bpm=170.0,
    ) + _raw_run_observations(
        centers_ms=(16_000.0, 17_000.0, 18_000.0),
        bpm=145.0,
    )

    proposals = _raw_run_jump_proposals(
        np.zeros(5000, dtype=np.float64),
        frame_rate_hz=50.0,
        duration_ms=80_000.0,
        bases=(_BaseHypothesis(origin_time_ms=0.0, bpm=200.0, score=1.0),),
        observations=observations,
        config=config,
    )

    chains = [proposal for proposal in proposals if proposal.source == "raw_run_chain_3_sections"]
    assert chains
    assert all(2 <= len(proposal.curve.sections) <= 4 for proposal in chains)


def test_exp014_positive_gain_without_raw_run_selects_constant() -> None:
    config = TempoTrackConfig()
    constant = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(ConstantTempoSection(0, 96, 180.0),),
    )
    jump = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 32, 180.0),
            ConstantTempoSection(32, 40, 150.0),
            ConstantTempoSection(40, 96, 180.0),
        ),
    )
    collapsed = (
        constant,
        _collapsed_constant_counterfactual(
            jump,
            duration_ms=jump.end_time_ms,
            base_bpm=180.0,
            config=config,
        ),
    )

    selection = _select_exp014_production_candidate(
        (constant, jump),
        raw_self_ranking=_ranking((constant, jump), raw_scores=(0.25, 1.0)),
        collapsed_raw_self_ranking=_ranking(collapsed, raw_scores=(0.25, 0.1)),
        collapsed_counterfactuals=collapsed,
        observations=(),
        primary=_BaseHypothesis(origin_time_ms=0.0, bpm=180.0, score=1.0),
        beat_signal=np.zeros(5000, dtype=np.float64),
        frame_rate_hz=50.0,
        config=config,
    )

    assert selection.status == "v3_accepted"
    assert selection.lane == "constant"
    assert selection.selected_candidate_index == 0
    assert selection.paired_raw_gain_by_candidate == ((1, 0.9, 0.1, collapsed[1].fingerprint_sha256),)


def test_exp014_diagnostic_reports_internal_jump_candidate_pruning() -> None:
    truth = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=150.0,
        sections=(
            ConstantTempoSection(0, 40, 175.0),
            ConstantTempoSection(40, 52, 143.0),
            ConstantTempoSection(52, 130, 175.0),
        ),
    )
    config = TempoTrackConfig(maximum_jump_candidates=1, maximum_candidates=64)
    duration_ms = truth.end_time_ms + 1000.0

    result = generate_timing_candidates(
        _prediction_from_curve(truth, duration_ms=duration_ms),
        audio_evidence=_raw_evidence_from_curve(truth, duration_ms=duration_ms),
        config=config,
    )
    payload = tempo_track_result_to_dict(result, include_observations=False)

    assert result.diagnostics.candidate_count < config.maximum_candidates
    assert result.diagnostics.candidate_cap_pruning_reason is not None
    assert "maximum_jump_candidates_1" in result.diagnostics.candidate_cap_pruning_reason
    assert (
        payload["diagnostics"]["candidate_cap_pruning_reason"]
        == result.diagnostics.candidate_cap_pruning_reason
    )


def _raw_run_observations(
    *,
    centers_ms: tuple[float, ...],
    bpm: float,
) -> tuple[LocalTempoObservation, ...]:
    return tuple(
        LocalTempoObservation(
            source="raw_audio",
            center_time_ms=center,
            window_start_ms=center - 3000.0,
            window_end_ms=center + 3000.0,
            bpm=bpm,
            strength=0.5,
        )
        for center in centers_ms
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
            candidate_domain_beat_count=max(16, candidate.end_beat - candidate.start_beat),
            complete_window_start_beats=(candidate.start_beat,),
        )
        for index, (candidate, raw_score) in enumerate(zip(candidates, raw_scores))
    )
    return RawAudioEvidenceRanking(
        evidence=_raw_evidence_from_curve(candidates[0], duration_ms=40_000.0),
        candidate_scores=scores,
        ranked_scores=tuple(
            sorted(scores, key=lambda score: (-score.raw_score, score.fingerprint_sha256))
        ),
        common_beat_indices=(),
        complete_window_start_beats=(),
        unavailable_reason=None,
    )


def _raw_evidence_from_curve(
    curve: PhaseContinuousTimingCurve,
    *,
    duration_ms: float,
) -> RawAudioEvidence:
    centers = np.arange(0.0125, duration_ms / 1000.0, 0.01, dtype=np.float64)
    flux = np.zeros((centers.size, 4), dtype=np.float32)
    for beat in range(curve.start_beat, curve.end_beat):
        time_seconds = curve.time_at_beat(float(beat)) / 1000.0
        index = int(np.argmin(np.abs(centers - time_seconds)))
        flux[index, :] = np.float32(1.0)
    return RawAudioEvidence(
        frame_center_seconds=centers,
        band_flux=flux,
        band_percentile95=(1.0, 1.0, 1.0, 1.0),
        input_frame_count=centers.size,
        valid_frame_count=centers.size,
        audio_duration_seconds=duration_ms / 1000.0,
    )


def _prediction_from_curve(
    curve: PhaseContinuousTimingCurve,
    *,
    duration_ms: float,
    frame_rate_hz: float = 50.0,
) -> FrameTimingPrediction:
    frame_count = int(math.ceil(duration_ms * frame_rate_hz / 1000.0))
    beat_prob = np.zeros(frame_count, dtype=np.float32)
    downbeat_prob = np.zeros(frame_count, dtype=np.float32)
    for beat in range(curve.start_beat, curve.end_beat):
        time_ms = curve.time_at_beat(float(beat))
        _write_pulse(beat_prob, time_ms=time_ms, frame_rate_hz=frame_rate_hz)
        if beat % 4 == 0:
            _write_pulse(downbeat_prob, time_ms=time_ms, frame_rate_hz=frame_rate_hz)
    return FrameTimingPrediction(
        provider="cached-beatthis",
        beat_prob=beat_prob,
        downbeat_prob=downbeat_prob,
        frame_rate_hz=frame_rate_hz,
    )


def _write_pulse(
    signal: np.ndarray,
    *,
    time_ms: float,
    frame_rate_hz: float,
) -> None:
    frame = int(round(time_ms * frame_rate_hz / 1000.0))
    if 0 <= frame < signal.size:
        signal[frame] = np.float32(1.0)
    for neighbor in (frame - 1, frame + 1):
        if 0 <= neighbor < signal.size:
            signal[neighbor] = max(signal[neighbor], np.float32(0.1))
