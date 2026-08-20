from __future__ import annotations

import numpy as np

from pulsefield_model.timing.v3.analytic_curve import (
    ConstantTempoSection,
    LinearTimeRampSection,
    PhaseContinuousTimingCurve,
)
from pulsefield_model.timing.v3.audio_evidence import (
    RawAudioEvidence,
    score_raw_audio_evidence_independent,
)
from pulsefield_model.timing.v3.tempo_track import (
    _LOCALIZED_BACKBONE_RAMP_SOURCE,
    _CurveProposal,
    _LocalizedBackboneSelection,
    _LocalizedRampChoice,
    _LocalizedRampRun,
    _PersistentStepSpec,
    _bounded_proposals,
    _infer_localized_persistent_step,
    _localized_backbone_ramp_choice,
    _localized_persistent_step_curve,
    _localized_ramp_fast_lane_selection,
    _localized_ramp_observation_gate,
    _localized_ramp_runs,
    _splice_localized_ramp,
    LocalTempoObservation,
    TempoTrackConfig,
    TempoTrackProductionSelection,
)


def _observation(source: str, second: float, bpm: float) -> LocalTempoObservation:
    return LocalTempoObservation(
        source=source,
        center_time_ms=second * 1000.0,
        window_start_ms=(second - 3.0) * 1000.0,
        window_end_ms=(second + 3.0) * 1000.0,
        bpm=bpm,
        strength=0.9,
    )


def _run() -> _LocalizedRampRun:
    return _LocalizedRampRun(
        observation_start_ms=119_000.0,
        observation_end_ms=125_000.0,
        fitted_start_bpm=173.5,
        fitted_end_bpm=225.2,
        r_squared=0.94,
        mean_strength=0.58,
        point_count=7,
        generation_score=28.5,
    )


def _evidence_for_curve(
    curve: PhaseContinuousTimingCurve,
    *,
    duration_seconds: float,
) -> RawAudioEvidence:
    centers = np.arange(0.005, duration_seconds, 0.01, dtype=np.float64)
    flux = np.zeros((centers.size, 4), dtype=np.float32)
    for beat in range(curve.start_beat, curve.end_beat):
        time_seconds = curve.time_at_beat(float(beat)) / 1000.0
        if not 0.0 <= time_seconds < duration_seconds:
            continue
        index = int(np.argmin(np.abs(centers - time_seconds)))
        flux[max(0, index - 1) : min(centers.size, index + 2)] = 2.0
    return RawAudioEvidence(
        frame_center_seconds=centers,
        band_flux=flux,
        band_percentile95=(1.0, 1.0, 1.0, 1.0),
        input_frame_count=centers.size,
        valid_frame_count=centers.size,
        audio_duration_seconds=duration_seconds,
    )


def test_localized_mode_regions_recover_persistent_120_to_180_step() -> None:
    config = TempoTrackConfig()
    observations = tuple(
        _observation("beatthis", float(second), 120.0 if second < 75 else 180.0)
        for second in range(119)
    )

    assert _infer_localized_persistent_step(
        _run(), observations, config=config
    ) == _PersistentStepSpec(
        left_bpm=120.0,
        right_bpm=180.0,
        transition_target_ms=80_000.0,
    )


def test_localized_ramp_splice_is_continuous_and_wins_both_controls() -> None:
    config = TempoTrackConfig()
    backbone = _localized_persistent_step_curve(
        _PersistentStepSpec(120.0, 180.0, 80_000.0),
        phase_ms=240.0,
        duration_ms=160_000.0,
    )
    triplet = _splice_localized_ramp(
        backbone,
        _run(),
        ramp_start_target_ms=120_273.0,
        ramp_end_target_ms=127_604.0,
        duration_ms=160_000.0,
        endpoint_delta_scale=0.6,
    )

    assert len(triplet.ramp.sections) == 4
    assert isinstance(triplet.ramp.sections[2], LinearTimeRampSection)
    assert all(
        report.phase_discontinuity_ms == 0.0
        for report in triplet.ramp.seam_reports
    )

    samples = tuple(float(second) for second in range(117, 131))
    raw = tuple(
        _observation(
            "raw_audio",
            second,
            triplet.ramp.bpm_at_time(second * 1000.0),
        )
        for second in samples
    )
    beatthis = tuple(
        _observation(
            "beatthis",
            second,
            triplet.ramp.bpm_at_time(second * 1000.0),
        )
        for second in samples
    )
    gate = _localized_ramp_observation_gate(
        triplet,
        raw_observations=raw,
        beat_observations=beatthis,
        config=config,
    )

    assert gate.eligible
    assert gate.raw_step_gain > 0.0
    assert gate.raw_no_ramp_gain > 0.0
    assert gate.beatthis_step_gain > 0.0
    assert gate.beatthis_no_ramp_gain > 0.0


def test_localized_ramp_fast_lane_selects_only_the_pre_gated_choice() -> None:
    config = TempoTrackConfig()
    backbone_curve = _localized_persistent_step_curve(
        _PersistentStepSpec(120.0, 180.0, 80_000.0),
        phase_ms=240.0,
        duration_ms=160_000.0,
    )
    triplet = _splice_localized_ramp(
        backbone_curve,
        _run(),
        ramp_start_target_ms=120_273.0,
        ramp_end_target_ms=127_604.0,
        duration_ms=160_000.0,
        endpoint_delta_scale=0.6,
    )
    proposal = _CurveProposal(
        curve=triplet.ramp,
        source=_LOCALIZED_BACKBONE_RAMP_SOURCE,
        score=1.0,
        collapse_bpm=180.0,
    )
    choice = _LocalizedRampChoice(
        proposal=proposal,
        backbone=_LocalizedBackboneSelection(
            curve=backbone_curve,
            raw_rank=1,
            beatthis_rank=1,
            constant_left_raw_gain=1.0,
            constant_left_beatthis_gain=1.0,
            constant_right_raw_gain=1.0,
            constant_right_beatthis_gain=1.0,
        ),
        gate=_localized_ramp_observation_gate(
            triplet,
            raw_observations=tuple(
                _observation(
                    "raw_audio",
                    second,
                    triplet.ramp.bpm_at_time(second * 1000.0),
                )
                for second in range(117, 131)
            ),
            beat_observations=tuple(
                _observation(
                    "beatthis",
                    second,
                    triplet.ramp.bpm_at_time(second * 1000.0),
                )
                for second in range(117, 131)
            ),
            config=config,
        ),
        run=_run(),
        endpoint_delta_scale=0.6,
    )
    raw_ranking = score_raw_audio_evidence_independent(
        _evidence_for_curve(triplet.ramp, duration_seconds=160.0),
        (triplet.ramp,),
    )
    fallback = TempoTrackProductionSelection(
        status="v2_fallback",
        selected_candidate_index=None,
        selected_fingerprint_sha256=None,
        lane="fallback",
        fallback_reason="test",
        raw_run=None,
        eligible_candidate_indices=(),
        raw_self_rank_by_candidate=(),
        beatthis_aba_rank_by_candidate=(),
    )

    selected = _localized_ramp_fast_lane_selection(
        (proposal,),
        choice=choice,
        raw_self_ranking=raw_ranking,
        fallback=fallback,
        config=config,
    )
    rejected = _localized_ramp_fast_lane_selection(
        (proposal,),
        choice=None,
        raw_self_ranking=raw_ranking,
        fallback=fallback,
        config=config,
    )

    assert selected.lane == "localized_ramp"
    assert selected.selected_candidate_index == 0
    assert rejected is fallback


def test_localized_ramp_has_a_reserved_ramp_slot_and_abrupt_steps_do_not_trigger() -> None:
    config = TempoTrackConfig(maximum_candidates=3)
    constant = _CurveProposal(
        curve=PhaseContinuousTimingCurve(
            origin_beat=0,
            origin_time_ms=0.0,
            sections=(ConstantTempoSection(0, 100, 120.0),),
        ),
        source="global_constant",
        score=1.0,
        collapse_bpm=120.0,
    )
    jump = _CurveProposal(
        curve=PhaseContinuousTimingCurve(
            origin_beat=0,
            origin_time_ms=0.0,
            sections=(
                ConstantTempoSection(0, 50, 120.0),
                ConstantTempoSection(50, 100, 150.0),
            ),
        ),
        source="paired_unmerged_boundary",
        score=1.0,
        collapse_bpm=120.0,
    )
    global_ramp = _CurveProposal(
        curve=PhaseContinuousTimingCurve(
            origin_beat=0,
            origin_time_ms=0.0,
            sections=(LinearTimeRampSection(0, 100, 120.0, 140.0),),
        ),
        source="linear_tempo_beatthis",
        score=100.0,
        collapse_bpm=120.0,
    )
    localized_ramp = _CurveProposal(
        curve=PhaseContinuousTimingCurve(
            origin_beat=0,
            origin_time_ms=0.0,
            sections=(LinearTimeRampSection(0, 100, 120.0, 150.0),),
        ),
        source=_LOCALIZED_BACKBONE_RAMP_SOURCE,
        score=1.0,
        collapse_bpm=120.0,
    )

    retained = _bounded_proposals(
        (constant,), (jump,), (global_ramp, localized_ramp), config=config
    )
    abrupt = tuple(
        _observation("beatthis", float(index), 147.0 if index < 7 else 123.0)
        for index in range(14)
    )

    assert tuple(proposal.source for proposal in retained) == (
        "global_constant",
        "paired_unmerged_boundary",
        _LOCALIZED_BACKBONE_RAMP_SOURCE,
    )
    assert not _localized_ramp_runs(abrupt, config=TempoTrackConfig())
