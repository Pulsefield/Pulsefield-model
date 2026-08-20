from __future__ import annotations

import json
import math

import numpy as np

from pulsefield_model.timing.schema import FrameTimingPrediction
from pulsefield_model.timing.v3.analytic_curve import (
    ConstantTempoSection,
    LinearTimeRampSection,
    PhaseContinuousTimingCurve,
)
from pulsefield_model.timing.v3.audio_evidence import RawAudioEvidence
from pulsefield_model.timing.v3.audio_evidence import (
    CandidateRawAudioScore,
    RawAudioEvidenceRanking,
)
from pulsefield_model.timing.v3.tempo_track import (
    TEMPO_TRACK_RESULT_DUMP_SCHEMA_VERSION,
    TEMPO_TRACK_VERSION,
    TempoTrackConfig,
    _dominant_raw_tempo_run,
    _eligible_jump_aba_delta,
    _BaseHypothesis,
    _select_exp013_production_candidate,
    _snap_raw_observation_bpm_to_primary,
    _weighted_median,
    generate_timing_candidates,
    LocalTempoObservation,
    tempo_track_result_to_dict,
)


def test_constant_200_candidate_is_generated_deterministically() -> None:
    truth = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=180.0,
        sections=(ConstantTempoSection(0, 200, 200.0),),
    )
    prediction = _prediction_from_curve(truth, duration_ms=60_000.0)

    first = generate_timing_candidates(prediction)
    second = generate_timing_candidates(prediction)

    assert any(
        candidate.curve_class == "constant"
        and abs(candidate.sections[0].bpm - 200.0) <= 1.0
        for candidate in first.candidates
    )
    assert tuple(candidate.fingerprint_sha256 for candidate in first.candidates) == tuple(
        candidate.fingerprint_sha256 for candidate in second.candidates
    )
    assert len(first.candidates) <= 64
    assert all(candidate.start_beat == 0 for candidate in first.candidates)
    assert all(candidate.end_time_ms >= 60_000.0 for candidate in first.candidates)
    assert first.production_selection.status == "v2_fallback"
    assert first.production_selection.fallback_reason == "raw_audio_evidence_not_supplied"
    assert first.selected_candidate is None


def test_exp013_constant_lane_selects_direct_constant_with_raw_evidence() -> None:
    truth = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=180.0,
        sections=(ConstantTempoSection(0, 200, 200.0),),
    )
    duration_ms = 60_000.0

    result = generate_timing_candidates(
        _prediction_from_curve(truth, duration_ms=duration_ms),
        audio_evidence=_raw_evidence_from_curve(truth, duration_ms=duration_ms),
    )

    assert result.production_selection.status == "v3_accepted"
    assert result.production_selection.lane == "constant"
    assert result.production_selection.raw_run is None
    assert result.selected_candidate is not None
    assert result.selected_candidate.curve_class == "constant"
    assert abs(result.selected_candidate.sections[0].bpm - 200.0) <= 1.0


def test_short_aba_excursion_is_proposed_from_raw_local_tempo() -> None:
    truth = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=150.0,
        sections=(
            ConstantTempoSection(0, 40, 175.0),
            ConstantTempoSection(40, 50, 143.0),
            ConstantTempoSection(50, 100, 175.0),
        ),
    )
    duration_ms = truth.end_time_ms + 1000.0
    prediction = _prediction_from_curve(truth, duration_ms=duration_ms)
    evidence = _raw_evidence_from_curve(truth, duration_ms=duration_ms)

    result = generate_timing_candidates(prediction, audio_evidence=evidence)

    raw_local = [
        observation
        for observation in result.observations
        if observation.source == "raw_audio"
        and truth.boundary_times_ms[1] - 2000.0
        <= observation.center_time_ms
        <= truth.boundary_times_ms[2] + 2000.0
    ]
    assert raw_local
    assert min(abs(observation.bpm - 143.0) for observation in raw_local) <= 8.0

    matching = []
    for candidate in result.candidates:
        if candidate.curve_class != "jump" or len(candidate.sections) != 3:
            continue
        middle = candidate.sections[1]
        if not isinstance(middle, ConstantTempoSection):
            continue
        if abs(middle.bpm - 143.0) <= 8.0 and 2.0 <= middle.duration_seconds <= 8.0:
            matching.append(candidate)
    assert matching
    assert all(report.phase_discontinuity_ms == 0.0 for report in matching[0].seam_reports)
    assert result.raw_self_ranking is not None
    assert result.raw_self_ranking is not None
    assert result.production_selection.status == "v3_accepted"
    # Raw tempo evidence may propose the excursion, but it cannot use the same
    # raw stream to promote itself without an independent BeatThis boundary.
    # This synthetic curve also produces an independently anchored proposal,
    # so the jump remains selectable through that source.
    assert result.production_selection.lane == "paired_jump"
    assert result.production_selection.raw_run is not None
    assert result.production_selection.raw_run.direction == "down"
    selected_index = result.production_selection.selected_candidate_index
    assert selected_index is not None
    assert not result.candidate_diagnostics[selected_index].source.startswith("raw_run_")
    assert result.selected_candidate is not None
    assert result.selected_candidate.curve_class == "jump"


def test_time_linear_120_to_180_ramp_candidate_is_generated() -> None:
    truth = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=200.0,
        sections=(LinearTimeRampSection(0, 150, 120.0, 180.0),),
    )
    prediction = _prediction_from_curve(truth, duration_ms=truth.end_time_ms + 200.0)

    result = generate_timing_candidates(prediction)

    ramps = [candidate for candidate in result.candidates if candidate.curve_class == "ramp"]
    assert ramps
    assert any(
        isinstance(candidate.sections[0], LinearTimeRampSection)
        and candidate.sections[0].end_bpm - candidate.sections[0].start_bpm >= 35.0
        and abs(candidate.sections[0].start_bpm - 120.0) <= 20.0
        and abs(candidate.sections[0].end_bpm - 180.0) <= 20.0
        for candidate in ramps
    )
    for candidate in ramps:
        for report in candidate.seam_reports:
            assert report.phase_discontinuity_ms == 0.0
    if result.production_selection.status == "v3_accepted":
        assert result.selected_candidate is not None
        assert result.selected_candidate.curve_class != "ramp"
    assert not any(
        result.candidates[index].curve_class == "ramp"
        for index in result.production_selection.eligible_candidate_indices
    )


def test_exp013_raw_run_uses_small_rational_snap_and_weighted_median() -> None:
    config = TempoTrackConfig()

    assert (
        _snap_raw_observation_bpm_to_primary(87.48, primary_bpm=175.0, config=config)
        == 175.0
    )
    assert (
        _snap_raw_observation_bpm_to_primary(143.0, primary_bpm=175.0, config=config)
        == 143.0
    )
    assert _weighted_median(
        np.asarray((140.0, 180.0, 220.0), dtype=np.float64),
        np.asarray((0.2, 0.7, 0.1), dtype=np.float64),
    ) == 180.0


def test_exp013_raw_run_detection_requires_three_close_same_direction_points() -> None:
    config = TempoTrackConfig()
    close_run = tuple(
        LocalTempoObservation(
            source="raw_audio",
            center_time_ms=center_time_ms,
            window_start_ms=center_time_ms - 3000.0,
            window_end_ms=center_time_ms + 3000.0,
            bpm=143.0,
            strength=0.08,
        )
        for center_time_ms in (10_000.0, 11_000.0, 12_000.0)
    )
    gapped = (
        close_run[0],
        close_run[1],
        LocalTempoObservation(
            source="raw_audio",
            center_time_ms=12_600.1,
            window_start_ms=9600.1,
            window_end_ms=15_600.1,
            bpm=143.0,
            strength=0.08,
        ),
    )

    detected = _dominant_raw_tempo_run(close_run, base_bpm=175.0, config=config)

    assert detected is not None
    assert detected.direction == "down"
    assert detected.observation_count == 3
    assert detected.expanded_start_time_ms == 9000.0
    assert detected.expanded_end_time_ms == 13_000.0
    assert _dominant_raw_tempo_run(gapped, base_bpm=175.0, config=config) is None


def test_exp013_raw_run_without_compatible_jump_falls_back_with_null_selection() -> None:
    constant = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=150.0,
        sections=(ConstantTempoSection(0, 100, 175.0),),
    )
    selection = _select_exp013_production_candidate(
        (constant,),
        raw_self_ranking=_synthetic_raw_self_ranking(
            (constant,),
            raw_scores=(1.0,),
        ),
        observations=_synthetic_raw_run_observations(),
        primary=_Primary(bpm=175.0),
        beat_signal=np.zeros(2400, dtype=np.float64),
        frame_rate_hz=50.0,
        config=TempoTrackConfig(),
    )

    assert selection.status == "v2_fallback"
    assert selection.selected_candidate_index is None
    assert selection.selected_fingerprint_sha256 is None
    assert selection.fallback_reason == "no_production_eligible_jump_for_raw_run"
    assert selection.raw_run is not None
    assert selection.raw_self_rank_by_candidate == ()
    assert selection.beatthis_aba_rank_by_candidate == ()


def test_exp013_raw_self_rank_is_lane_relative_not_global() -> None:
    eligible_a = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=150.0,
        sections=(
            ConstantTempoSection(0, 40, 175.0),
            ConstantTempoSection(40, 50, 143.0),
            ConstantTempoSection(50, 100, 175.0),
        ),
    )
    eligible_b = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=150.0,
        sections=(
            ConstantTempoSection(0, 42, 175.0),
            ConstantTempoSection(42, 52, 143.0),
            ConstantTempoSection(52, 100, 175.0),
        ),
    )
    high_raw_ineligible = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=150.0,
        sections=(ConstantTempoSection(0, 100, 175.0),),
    )
    config = TempoTrackConfig()
    observations = _synthetic_raw_run_observations()
    beat_signal = np.zeros(2400, dtype=np.float64)

    without_ineligible = _select_exp013_production_candidate(
        (eligible_a, eligible_b),
        raw_self_ranking=_synthetic_raw_self_ranking(
            (eligible_a, eligible_b),
            raw_scores=(0.80, 0.70),
        ),
        observations=observations,
        primary=_Primary(bpm=175.0),
        beat_signal=beat_signal,
        frame_rate_hz=50.0,
        config=config,
    )
    with_ineligible = _select_exp013_production_candidate(
        (high_raw_ineligible, eligible_a, eligible_b),
        raw_self_ranking=_synthetic_raw_self_ranking(
            (high_raw_ineligible, eligible_a, eligible_b),
            raw_scores=(10.0, 0.80, 0.70),
        ),
        observations=observations,
        primary=_Primary(bpm=175.0),
        beat_signal=beat_signal,
        frame_rate_hz=50.0,
        config=config,
    )

    assert without_ineligible.status == "v3_accepted"
    assert with_ineligible.status == "v3_accepted"
    assert without_ineligible.raw_self_rank_by_candidate == ((0, 1), (1, 2))
    assert with_ineligible.raw_self_rank_by_candidate == ((1, 1), (2, 2))
    assert 0 not in with_ineligible.eligible_candidate_indices
    assert with_ineligible.selected_candidate_index in (1, 2)


def test_exp013_middle_run_tempo_ratio_uses_small_rational_alias_canonicalization() -> None:
    config = TempoTrackConfig()
    compatible = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 40, 200.0),
            ConstantTempoSection(40, 48, 100.0),
            ConstantTempoSection(48, 96, 200.0),
        ),
    )
    raw_run = _raw_run_diagnostic(direction="down", median_bpm=200.0)

    assert _eligible_jump_aba_delta(
        compatible,
        raw_run=raw_run,
        primary=_BaseHypothesis(origin_time_ms=0.0, bpm=200.0, score=1.0),
        beat_signal=np.zeros(4000, dtype=np.float64),
        frame_rate_hz=50.0,
        config=config,
    ) is not None


def test_exp013_middle_run_tempo_ratio_rejects_true_incompatibility() -> None:
    config = TempoTrackConfig()
    incompatible = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 40, 200.0),
            ConstantTempoSection(40, 48, 123.0),
            ConstantTempoSection(48, 96, 200.0),
        ),
    )
    raw_run = _raw_run_diagnostic(direction="down", median_bpm=200.0)

    assert _eligible_jump_aba_delta(
        incompatible,
        raw_run=raw_run,
        primary=_BaseHypothesis(origin_time_ms=0.0, bpm=200.0, score=1.0),
        beat_signal=np.zeros(4000, dtype=np.float64),
        frame_rate_hz=50.0,
        config=config,
    ) is None


def test_result_dump_joins_curve_generation_and_raw_score() -> None:
    truth = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=180.0,
        sections=(ConstantTempoSection(0, 80, 200.0),),
    )
    duration_ms = truth.end_time_ms + 200.0
    result = generate_timing_candidates(
        _prediction_from_curve(truth, duration_ms=duration_ms),
        audio_evidence=_raw_evidence_from_curve(truth, duration_ms=duration_ms),
    )

    payload = tempo_track_result_to_dict(result)

    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    assert payload["schema"] == TEMPO_TRACK_RESULT_DUMP_SCHEMA_VERSION
    assert payload["tempo_track_version"] == TEMPO_TRACK_VERSION
    assert payload["selected_candidate_index"] == (
        result.production_selection.selected_candidate_index
    )
    assert payload["production_selection"]["selected_candidate_index"] == (
        result.production_selection.selected_candidate_index
    )
    assert payload["production_selection"]["selection_rank_scope"] == "eligible_lane"
    assert payload["diagnostics"]["candidate_count"] == len(result.candidates)
    assert len(payload["candidates"]) == len(result.candidates)
    first = payload["candidates"][0]
    assert first["candidate_index"] == 0
    assert first["fingerprint_sha256"] == result.candidates[0].fingerprint_sha256
    assert first["curve"]["sections"]
    assert first["raw_audio_self_score"]["raw_score"] is not None
    assert "observations" in payload

    compact = tempo_track_result_to_dict(result, include_observations=False)
    assert "observations" not in compact


def test_result_dump_serializes_fallback_null_selection() -> None:
    result = generate_timing_candidates(
        _prediction_from_curve(
            PhaseContinuousTimingCurve(
                origin_beat=0,
                origin_time_ms=180.0,
                sections=(ConstantTempoSection(0, 80, 200.0),),
            ),
            duration_ms=24_000.0,
        )
    )

    payload = tempo_track_result_to_dict(result)

    assert payload["selected_candidate_index"] is None
    assert payload["selected_fingerprint_sha256"] is None
    assert payload["production_selection"]["status"] == "v2_fallback"
    assert payload["production_selection"]["selected_candidate_index"] is None
    assert payload["production_selection"]["selected_fingerprint_sha256"] is None
    assert not any(candidate["selected"] for candidate in payload["candidates"])


def _prediction_from_curve(
    curve: PhaseContinuousTimingCurve,
    *,
    duration_ms: float,
    corrupt_interval: tuple[float, float, float] | None = None,
    frame_rate_hz: float = 50.0,
) -> FrameTimingPrediction:
    frame_count = int(math.ceil(duration_ms * frame_rate_hz / 1000.0))
    beat_prob = np.zeros(frame_count, dtype=np.float32)
    downbeat_prob = np.zeros(frame_count, dtype=np.float32)
    for beat in range(curve.start_beat, curve.end_beat):
        time_ms = curve.time_at_beat(float(beat))
        if corrupt_interval is not None and corrupt_interval[0] <= time_ms <= corrupt_interval[1]:
            continue
        _write_pulse(beat_prob, time_ms=time_ms, frame_rate_hz=frame_rate_hz)
        if beat % 4 == 0:
            _write_pulse(downbeat_prob, time_ms=time_ms, frame_rate_hz=frame_rate_hz)
    if corrupt_interval is not None:
        start_ms, end_ms, corrupt_bpm = corrupt_interval
        time_ms = start_ms
        while time_ms <= end_ms:
            _write_pulse(beat_prob, time_ms=time_ms, frame_rate_hz=frame_rate_hz)
            time_ms += 60000.0 / corrupt_bpm
    return FrameTimingPrediction(
        provider="cached-beatthis",
        beat_prob=beat_prob,
        downbeat_prob=downbeat_prob,
        frame_rate_hz=frame_rate_hz,
    )


class _Primary:
    def __init__(self, *, bpm: float) -> None:
        self.bpm = bpm


def _raw_run_diagnostic(
    *,
    direction: str,
    median_bpm: float,
) -> object:
    from pulsefield_model.timing.v3.tempo_track import TempoTrackRawRunDiagnostic

    return TempoTrackRawRunDiagnostic(
        direction=direction,
        start_time_ms=12_000.0,
        end_time_ms=16_000.0,
        expanded_start_time_ms=11_000.0,
        expanded_end_time_ms=17_000.0,
        median_bpm=median_bpm,
        weighted_median_delta_bpm=-100.0 if direction == "down" else 100.0,
        observation_count=3,
        summed_strength=1.0,
    )


def _synthetic_raw_run_observations() -> tuple[LocalTempoObservation, ...]:
    return tuple(
        LocalTempoObservation(
            source="raw_audio",
            center_time_ms=center_time_ms,
            window_start_ms=center_time_ms - 3000.0,
            window_end_ms=center_time_ms + 3000.0,
            bpm=143.0,
            strength=0.08,
        )
        for center_time_ms in (14_000.0, 15_000.0, 16_000.0)
    )


def _synthetic_raw_self_ranking(
    candidates: tuple[PhaseContinuousTimingCurve, ...],
    *,
    raw_scores: tuple[float, ...],
) -> RawAudioEvidenceRanking:
    if len(candidates) != len(raw_scores):
        raise ValueError("candidates and raw_scores must be parallel")
    evidence = _raw_evidence_from_curve(candidates[0], duration_ms=40_000.0)
    scores = tuple(
        CandidateRawAudioScore(
            candidate_index=index,
            fingerprint_sha256=candidate.fingerprint_sha256,
            raw_score=raw_score,
            mean_beat_support=raw_score,
            mean_half_beat_support=0.0,
            window_contrast_p10=raw_score,
            retained_beat_count=candidate.end_beat - candidate.start_beat,
            complete_window_count=1,
            unavailable_reason=None,
            candidate_domain_beat_count=candidate.end_beat - candidate.start_beat,
            complete_window_start_beats=(candidate.start_beat,),
        )
        for index, (candidate, raw_score) in enumerate(zip(candidates, raw_scores))
    )
    ranked = tuple(
        sorted(scores, key=lambda score: (-score.raw_score, score.fingerprint_sha256))
    )
    return RawAudioEvidenceRanking(
        evidence=evidence,
        candidate_scores=scores,
        ranked_scores=ranked,
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
        if index + 1 < flux.shape[0]:
            flux[index + 1, :] = np.float32(0.25)
    return RawAudioEvidence(
        frame_center_seconds=centers,
        band_flux=flux,
        band_percentile95=(1.0, 1.0, 1.0, 1.0),
        input_frame_count=centers.size,
        valid_frame_count=centers.size,
        audio_duration_seconds=duration_ms / 1000.0,
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
