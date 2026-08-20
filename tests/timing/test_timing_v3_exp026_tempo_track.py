from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from pulsefield_model.timing.schema import FrameTimingPrediction
from pulsefield_model.timing.v3 import tempo_track as tempo_track_module
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
    _BoundarySeed,
    _collapsed_constant_counterfactual,
    _CurveProposal,
    _EarlyHalfPrimaryPrefixStepChoice,
    _early_half_primary_prefix_fast_lane_selection,
    _early_half_primary_prefix_step_choices,
    _EARLY_HALF_PRIMARY_PREFIX_STEP_SOURCE,
    _generalized_support_delta,
    _jump_proposals,
    _observation_side_segment_proposals,
    _observation_side_segment_seeds,
    _PairSeed,
    _RegionalTempoMode,
    _regional_closed_aba_mode_triples,
    _regional_mode_pairs,
    _score_localized_backbones,
    _select_exp014_production_candidate,
    _SHORT_WINDOW_MONOTONE_CHAIN_SOURCE,
    _is_short_window_monotone_chain_jump,
    _short_window_monotone_chain_proposal,
    _sparse_middle_beat_counts,
    generate_timing_candidates,
    LocalTempoObservation,
    TempoTrackConfig,
    TempoTrackProductionSelection,
)


def test_exp026_sparse_middle_beat_beam_keeps_regional_direct_and_caps_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TempoTrackConfig()
    base_bpm = 180.0
    middle_bpm = 143.0
    first_beat = 60
    middle_beats = 10
    first_time_ms = first_beat * 60000.0 / base_bpm
    second_time_ms = first_time_ms + middle_beats * 60000.0 / middle_bpm
    duration_ms = 65_000.0
    truth = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, first_beat, base_bpm),
            ConstantTempoSection(first_beat, first_beat + middle_beats, middle_bpm),
            ConstantTempoSection(first_beat + middle_beats, 195, base_bpm),
        ),
    )
    pair = _PairSeed(
        left_time_ms=first_time_ms,
        right_time_ms=second_time_ms,
        rank_score=10.0,
        source="paired_unmerged_boundary",
    )
    observations = tuple(
        _observation(source, center_time_ms, bpm)
        for source in ("beatthis", "raw_audio")
        for center_time_ms, bpm in (
            (first_time_ms + 1500.0, middle_bpm),
            (first_time_ms + 2500.0, middle_bpm),
            (first_time_ms + 2000.0, 120.0),
        )
    )
    support_calls: dict[int, int] = {}
    original_support = tempo_track_module._aba_support_delta

    def counted_support(*args: object, first_beat: int, **kwargs: object) -> float:
        support_calls[first_beat] = support_calls.get(first_beat, 0) + 1
        return original_support(*args, first_beat=first_beat, **kwargs)

    monkeypatch.setattr(tempo_track_module, "_aba_support_delta", counted_support)
    batch = _jump_proposals(
        _beat_signal_from_curve(truth, duration_ms=duration_ms),
        frame_rate_hz=50.0,
        duration_ms=duration_ms,
        shared_end_beat=195,
        bases=(_BaseHypothesis(origin_time_ms=0.0, bpm=base_bpm, score=1.0),),
        observations=observations,
        pair_seeds=(pair,),
        config=config,
    )

    direct = _find_sections(
        batch.proposals,
        (base_bpm, middle_bpm, base_bpm),
        tolerance_bpm=1e-9,
    )
    assert direct is not None
    assert support_calls
    assert max(support_calls.values()) <= 6
    assert sum(support_calls.values()) <= 18


def test_exp026_sparse_middle_beat_beam_has_bounded_nominal_fallback() -> None:
    config = TempoTrackConfig()
    counts = _sparse_middle_beat_counts(
        pair=_PairSeed(
            left_time_ms=12_000.0,
            right_time_ms=16_000.0,
            rank_score=1.0,
            source="paired_unmerged_boundary",
        ),
        observations=(),
        base=_BaseHypothesis(origin_time_ms=0.0, bpm=180.0, score=1.0),
        first_beat=36,
        left_time_ms=12_000.0,
        middle_duration_ms=4_000.0,
        shared_end_beat=200,
        config=config,
    )

    assert 0 < len(counts) <= 6
    implied_bpms = tuple(60000.0 * count / 4000.0 for count in counts)
    assert all(abs(bpm - 5.0 * round(bpm / 5.0)) < 1e-9 for bpm in implied_bpms)
    assert all(abs(bpm - 180.0) >= config.minimum_jump_bpm for bpm in implied_bpms)


def test_exp026_side_segment_uses_local_octave_alias_above_global_max() -> None:
    config = TempoTrackConfig()
    duration_ms = 50_000.0
    boundary_time_ms = 20_000.0
    truth = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 66, 199.0),
            ConstantTempoSection(66, 206, 278.0),
        ),
    )

    proposals = _observation_side_segment_proposals(
        _beat_signal_from_curve(truth, duration_ms=duration_ms),
        frame_rate_hz=50.0,
        duration_ms=duration_ms,
        global_candidates=_global_candidates_from_curve(truth),
        bases=(_BaseHypothesis(origin_time_ms=0.0, bpm=199.0, score=1.0),),
        observations=(
            _observation("raw_audio", boundary_time_ms - 1000.0, 99.5),
            _observation("beatthis", boundary_time_ms - 900.0, 199.0),
            _observation("raw_audio", boundary_time_ms + 1000.0, 139.0),
            _observation("beatthis", boundary_time_ms + 900.0, 139.0),
        ),
        boundary_seeds=(_BoundarySeed(time_ms=boundary_time_ms, rank_score=10.0),),
        pair_seeds=(),
        config=config,
    )

    direct = _find_sections(proposals, (199.0, 278.0))
    assert direct is not None
    assert len(proposals) <= 6
    assert abs(direct.boundary_times_ms[1] - boundary_time_ms) <= 1000.0
    assert all(report.phase_discontinuity_ms == 0.0 for report in direct.seam_reports)


def test_exp026_side_segment_adds_bounded_mode_change_midpoint_seed() -> None:
    config = TempoTrackConfig()
    transition_time_ms = 43_653.0
    duration_ms = 70_000.0
    truth = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 90, 125.0),
            ConstantTempoSection(90, 151, 140.0),
        ),
    )
    observations = tuple(
        _observation(source, center, bpm)
        for source in ("beatthis", "raw_audio")
        for center, bpm in (
            (34_000.0, 125.1),
            (39_000.0, 125.1),
            (43_000.0, 125.1),
            (44_000.0, 140.1),
            (49_000.0, 140.1),
            (54_000.0, 140.1),
        )
    )

    seeds = _observation_side_segment_seeds(
        observations,
        boundary_seeds=(),
        config=config,
    )
    proposals = _observation_side_segment_proposals(
        _beat_signal_from_curve(truth, duration_ms=duration_ms),
        frame_rate_hz=50.0,
        duration_ms=duration_ms,
        global_candidates=_global_candidates_from_curve(truth),
        # The global hypothesis follows the dominant ending section.  The
        # observed left section must still be allowed to propose 125 -> 140.
        bases=(_BaseHypothesis(origin_time_ms=0.0, bpm=140.0, score=1.0),),
        observations=observations,
        boundary_seeds=(),
        pair_seeds=(),
        config=config,
    )

    assert any(seed.source == "mode_change_midpoint" for seed in seeds)
    direct = _find_sections(proposals, (125.0, 140.1), tolerance_bpm=0.25)
    assert direct is not None
    assert abs(direct.boundary_times_ms[1] - transition_time_ms) <= 1000.0


def test_exp026_lower_rank_mode_change_region_uses_previous_hop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TempoTrackConfig()
    regions = tuple(
        (10_000.0 + 20_000.0 * index, 100.0 - index)
        for index in range(9)
    )
    monkeypatch.setattr(
        tempo_track_module,
        "_observation_mode_change_regions",
        lambda *_args, **_kwargs: regions,
    )

    seeds = _observation_side_segment_seeds(
        (),
        boundary_seeds=(),
        config=config,
    )

    lower_rank_score = regions[-1][1]
    lower_rank_times = tuple(
        seed.time_ms for seed in seeds if seed.rank_score == lower_rank_score
    )
    assert lower_rank_times == (
        regions[-1][0] - 1000.0 * config.local_hop_seconds,
    )


def test_exp026_regional_lane_ignores_pair_seed_triples() -> None:
    config = TempoTrackConfig()
    first_time_ms = 20_000.0
    second_time_ms = first_time_ms + 16 * 60000.0 / 170.0
    duration_ms = 70_000.0
    truth = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 50, 150.0),
            ConstantTempoSection(50, 66, 170.0),
            ConstantTempoSection(66, 160, 120.0),
        ),
    )
    observations = tuple(
        _observation(source, center, bpm)
        for source in ("beatthis", "raw_audio")
        for center, bpm in (
            (first_time_ms - 1500.0, 150.0),
            (first_time_ms + 1500.0, 170.0),
            (first_time_ms + 3500.0, 170.0),
            (second_time_ms + 1500.0, 120.0),
        )
    )

    proposals = _observation_side_segment_proposals(
        _beat_signal_from_curve(truth, duration_ms=duration_ms),
        frame_rate_hz=50.0,
        duration_ms=duration_ms,
        global_candidates=_global_candidates_from_curve(truth),
        bases=(_BaseHypothesis(origin_time_ms=0.0, bpm=150.0, score=1.0),),
        observations=observations,
        boundary_seeds=(_BoundarySeed(time_ms=first_time_ms, rank_score=20.0),),
        pair_seeds=(
            _PairSeed(
                left_time_ms=first_time_ms,
                right_time_ms=second_time_ms,
                rank_score=20.0,
                source="paired_unmerged_boundary",
            ),
        ),
        config=config,
    )

    assert proposals
    assert all(len(proposal.curve.sections) == 2 for proposal in proposals)
    assert _find_sections(proposals, (150.0, 170.0, 120.0)) is None


def test_exp026_regional_lane_keeps_strict_closed_aba_pair() -> None:
    config = TempoTrackConfig()
    first_beat = 93
    middle_beats = 33
    first_time_ms = 20_000.0
    origin_time_ms = first_time_ms - first_beat * 60000.0 / 278.0
    second_time_ms = first_time_ms + middle_beats * 60000.0 / 199.0
    duration_ms = 60_000.0
    truth = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=origin_time_ms,
        sections=(
            ConstantTempoSection(0, first_beat, 278.0),
            ConstantTempoSection(first_beat, first_beat + middle_beats, 199.0),
            ConstantTempoSection(first_beat + middle_beats, 265, 278.0),
        ),
    )
    observations = tuple(
        _observation(source, center, bpm)
        for source in ("beatthis", "raw_audio")
        for center, bpm in (
            (first_time_ms - 4000.0, 139.0),
            (first_time_ms + 4000.0, 99.5),
            (second_time_ms - 4000.0, 99.5),
            (second_time_ms + 4000.0, 139.0),
        )
    )
    pair = _PairSeed(
        left_time_ms=first_time_ms,
        right_time_ms=second_time_ms,
        rank_score=20.0,
        source="paired_unmerged_boundary",
    )

    proposals = _observation_side_segment_proposals(
        _beat_signal_from_curve(truth, duration_ms=duration_ms),
        frame_rate_hz=50.0,
        duration_ms=duration_ms,
        global_candidates=_global_candidates_from_curve(truth),
        bases=(_BaseHypothesis(origin_time_ms=origin_time_ms, bpm=139.0, score=1.0),),
        observations=observations,
        boundary_seeds=(
            _BoundarySeed(time_ms=first_time_ms, rank_score=10.0),
            _BoundarySeed(time_ms=second_time_ms, rank_score=10.0),
        ),
        pair_seeds=(pair,),
        config=config,
    )

    direct = _find_sections(proposals, (278.0, 199.0, 278.0), tolerance_bpm=1.0)
    assert direct is not None
    assert len(proposals) <= 8
    assert all(report.phase_discontinuity_ms == 0.0 for report in direct.seam_reports)


def test_exp026_regional_pairs_reject_mixed_alias_multipliers() -> None:
    config = TempoTrackConfig()

    pairs = _regional_mode_pairs(
        (
            _RegionalTempoMode(
                source="raw_audio",
                bpm=299.66,
                support=1.0,
                alias_multiplier=2.0,
            ),
            _RegionalTempoMode(
                source="raw_audio",
                bpm=149.83,
                support=0.5,
                alias_multiplier=1.0,
            ),
        ),
        (
            _RegionalTempoMode(
                source="raw_audio",
                bpm=74.80,
                support=10.0,
                alias_multiplier=0.5,
            ),
            _RegionalTempoMode(
                source="beatthis",
                bpm=170.20,
                support=0.5,
                alias_multiplier=1.0,
            ),
        ),
        max_pairs=4,
        config=config,
    )

    assert pairs
    assert all(
        left.alias_multiplier == right.alias_multiplier
        for left, right, _ in pairs
    )
    assert all(
        not (math.isclose(left.bpm, 299.66) and math.isclose(right.bpm, 74.80))
        for left, right, _ in pairs
    )
    assert pairs[0][0].bpm == 149.83
    assert pairs[0][1].bpm == 170.20


def test_exp026_regional_pairs_reject_four_to_three_subdivision_alias() -> None:
    pairs = _regional_mode_pairs(
        (
            _RegionalTempoMode(
                source="beatthis",
                bpm=200.0,
                support=1.0,
                alias_multiplier=1.0,
            ),
        ),
        (
            _RegionalTempoMode(
                source="beatthis",
                bpm=800.0 / 3.0,
                support=1.0,
                alias_multiplier=1.0,
            ),
        ),
        max_pairs=4,
        config=TempoTrackConfig(),
    )

    assert pairs == ()


def test_exp026_closed_aba_rejects_three_to_two_subdivision_alias() -> None:
    triples = _regional_closed_aba_mode_triples(
        (
            _RegionalTempoMode(
                source="raw_audio",
                bpm=67.0,
                support=1.0,
                alias_multiplier=1.0,
            ),
        ),
        (
            _RegionalTempoMode(
                source="raw_audio",
                bpm=100.0,
                support=1.0,
                alias_multiplier=1.0,
            ),
        ),
        (
            _RegionalTempoMode(
                source="raw_audio",
                bpm=67.0,
                support=1.0,
                alias_multiplier=1.0,
            ),
        ),
        max_triples=4,
        config=TempoTrackConfig(),
    )

    assert triples == ()


def test_exp026_side_segment_seed_and_candidate_work_are_bounded() -> None:
    config = TempoTrackConfig()
    observations = tuple(
        _observation(
            "raw_audio" if index % 2 else "beatthis",
            center_time_ms=float(index * 3000),
            bpm=120.0 if index % 2 else 170.0,
        )
        for index in range(80)
    )
    boundary_seeds = tuple(
        _BoundarySeed(time_ms=float(index * 1000), rank_score=100.0 - index)
        for index in range(80)
    )
    truth = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 80, 120.0),
            ConstantTempoSection(80, 260, 170.0),
        ),
    )

    seeds = _observation_side_segment_seeds(
        observations,
        boundary_seeds=boundary_seeds,
        config=config,
    )
    proposals = _observation_side_segment_proposals(
        _beat_signal_from_curve(truth, duration_ms=100_000.0),
        frame_rate_hz=50.0,
        duration_ms=100_000.0,
        global_candidates=_global_candidates_from_curve(truth),
        bases=(_BaseHypothesis(origin_time_ms=0.0, bpm=120.0, score=1.0),),
        observations=observations,
        boundary_seeds=boundary_seeds,
        pair_seeds=(),
        config=config,
    )

    assert len(seeds) <= 32
    assert len(proposals) <= 6
    assert all(len(proposal.curve.sections) == 2 for proposal in proposals)


def test_localized_backbone_prescore_respects_raw_scorer_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TempoTrackConfig()
    curves = tuple(
        PhaseContinuousTimingCurve(
            origin_beat=0,
            origin_time_ms=float(index),
            sections=(
                ConstantTempoSection(0, 64, 120.0),
                ConstantTempoSection(64, 160, 150.0),
            ),
        )
        for index in range(75)
    )
    seen_lengths: list[int] = []
    seen_first_fingerprints: list[str] = []

    def fake_score(
        evidence: RawAudioEvidence,
        candidates: tuple[PhaseContinuousTimingCurve, ...],
    ) -> RawAudioEvidenceRanking:
        seen_lengths.append(len(candidates))
        seen_first_fingerprints.append(candidates[0].fingerprint_sha256)
        return _ranking(
            candidates,
            raw_scores=tuple(
                2.0 if index == 0 else 1.0 for index in range(len(candidates))
            ),
        )

    monkeypatch.setattr(
        tempo_track_module,
        "score_raw_audio_evidence_independent",
        fake_score,
    )
    monkeypatch.setattr(
        tempo_track_module,
        "_physical_curve_support_score",
        lambda *_args, curve, **_kwargs: 1.0
        if curve.curve_class == "jump"
        else 0.0,
    )

    _score_localized_backbones(
        curves,
        evidence=_empty_evidence(),
        beat_signal=np.zeros(1, dtype=np.float64),
        frame_rate_hz=50.0,
        duration_ms=80_000.0,
        config=config,
    )

    assert seen_lengths == [64, 2]
    assert seen_first_fingerprints[0] == curves[0].fingerprint_sha256


def test_exp026_source_aware_selector_blocks_raw_run_self_promotion() -> None:
    config = TempoTrackConfig()
    constant = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(ConstantTempoSection(0, 140, 150.0),),
    )
    raw_run_jump = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 50, 150.0),
            ConstantTempoSection(50, 150, 170.0),
        ),
    )
    anchored_jump = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 60, 150.0),
            ConstantTempoSection(60, 160, 170.0),
        ),
    )
    candidates = (constant, raw_run_jump, anchored_jump)
    duration_ms = max(candidate.end_time_ms for candidate in candidates)
    collapsed = tuple(
        _collapsed_constant_counterfactual(
            candidate,
            duration_ms=duration_ms,
            base_bpm=150.0,
            config=config,
        )
        for candidate in candidates
    )

    selection = _select_exp014_production_candidate(
        candidates,
        raw_self_ranking=_ranking(candidates, raw_scores=(0.4, 2.0, 1.5)),
        collapsed_raw_self_ranking=_ranking(collapsed, raw_scores=(0.4, 0.1, 0.2)),
        collapsed_counterfactuals=collapsed,
        observations=tuple(
            _observation(source, center, bpm)
            for source in ("beatthis", "raw_audio")
            for center, bpm in ((20_000.0, 150.0), (28_000.0, 170.0))
        ),
        primary=_BaseHypothesis(origin_time_ms=0.0, bpm=150.0, score=1.0),
        beat_signal=_beat_signal_from_curve(anchored_jump, duration_ms=duration_ms),
        frame_rate_hz=50.0,
        candidate_sources=(
            "global_constant",
            "raw_run_persistent_a_to_b_start",
            "observation_side_segment",
        ),
        config=config,
    )

    assert selection.status == "v3_accepted"
    assert selection.lane == "paired_jump"
    assert selection.eligible_candidate_indices == (2,)
    assert selection.selected_candidate_index == 2


def test_short_window_monotone_chain_generation_is_bounded_and_dual_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TempoTrackConfig()
    centers_ms = np.arange(123_000.0, 141_000.0, 500.0)
    bpms = np.interp(
        centers_ms,
        (126_000.0, 137_500.0),
        (128.0, 145.0),
    )
    observations = tuple(
        LocalTempoObservation(
            source=source,
            center_time_ms=float(center_ms),
            window_start_ms=float(center_ms - 1500.0),
            window_end_ms=float(center_ms + 1500.0),
            bpm=float(bpm),
            strength=1.0,
        )
        for source in ("beatthis", "raw_audio")
        for center_ms, bpm in zip(centers_ms, bpms, strict=True)
    )
    monkeypatch.setattr(
        tempo_track_module,
        "_generalized_support_delta",
        lambda *_args, **_kwargs: 1.0,
    )

    proposal = _short_window_monotone_chain_proposal(
        signal=np.zeros(12_000, dtype=np.float64),
        frame_rate_hz=50.0,
        duration_ms=240_000.0,
        bases=(_BaseHypothesis(origin_time_ms=225.0, bpm=128.0, score=1.0),),
        observations=observations,
        config=config,
    )

    assert proposal is not None
    assert proposal.source == _SHORT_WINDOW_MONOTONE_CHAIN_SOURCE
    assert 4 <= len(proposal.curve.sections) <= 6
    assert _is_short_window_monotone_chain_jump(proposal.curve)
    assert all(
        abs(float(section.start_bpm) - round(float(section.start_bpm))) < 1e-9
        for section in proposal.curve.sections
    )


def test_short_window_monotone_chain_rejects_three_section_takeover() -> None:
    three_section = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 60, 128.0),
            ConstantTempoSection(60, 64, 136.0),
            ConstantTempoSection(64, 200, 145.0),
        ),
    )

    assert not _is_short_window_monotone_chain_jump(three_section)


def test_short_window_monotone_chain_rejects_small_total_drift() -> None:
    small_drift = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 60, 100.0),
            ConstantTempoSection(60, 64, 99.0),
            ConstantTempoSection(64, 68, 97.0),
            ConstantTempoSection(68, 72, 91.0),
            ConstantTempoSection(72, 120, 90.0),
        ),
    )

    assert not _is_short_window_monotone_chain_jump(small_drift)


def test_short_window_monotone_chain_selector_is_source_specific_and_positive_gain_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TempoTrackConfig()
    constant = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(ConstantTempoSection(0, 200, 128.0),),
    )
    chain = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 60, 128.0),
            ConstantTempoSection(60, 64, 131.0),
            ConstantTempoSection(64, 68, 135.0),
            ConstantTempoSection(68, 72, 139.0),
            ConstantTempoSection(72, 76, 143.0),
            ConstantTempoSection(76, 200, 145.0),
        ),
    )
    candidates = (constant, chain)
    duration_ms = max(candidate.end_time_ms for candidate in candidates)
    collapsed = tuple(
        _collapsed_constant_counterfactual(
            candidate,
            duration_ms=duration_ms,
            base_bpm=128.0,
            config=config,
        )
        for candidate in candidates
    )
    monkeypatch.setattr(
        tempo_track_module,
        "_generalized_support_delta",
        lambda *_args, **_kwargs: 1.0,
    )

    selected = _select_exp014_production_candidate(
        candidates,
        raw_self_ranking=_ranking(candidates, raw_scores=(0.4, 1.5)),
        collapsed_raw_self_ranking=_ranking(collapsed, raw_scores=(0.4, 0.2)),
        collapsed_counterfactuals=collapsed,
        observations=(),
        primary=_BaseHypothesis(origin_time_ms=0.0, bpm=128.0, score=1.0),
        beat_signal=np.zeros(1, dtype=np.float64),
        frame_rate_hz=50.0,
        candidate_sources=("global_constant", _SHORT_WINDOW_MONOTONE_CHAIN_SOURCE),
        config=config,
    )
    old_source = _select_exp014_production_candidate(
        candidates,
        raw_self_ranking=_ranking(candidates, raw_scores=(0.4, 1.5)),
        collapsed_raw_self_ranking=_ranking(collapsed, raw_scores=(0.4, 0.2)),
        collapsed_counterfactuals=collapsed,
        observations=(),
        primary=_BaseHypothesis(origin_time_ms=0.0, bpm=128.0, score=1.0),
        beat_signal=np.zeros(1, dtype=np.float64),
        frame_rate_hz=50.0,
        candidate_sources=("global_constant", "observation_side_segment"),
        config=config,
    )
    negative_raw_gain = _select_exp014_production_candidate(
        candidates,
        raw_self_ranking=_ranking(candidates, raw_scores=(0.4, 0.1)),
        collapsed_raw_self_ranking=_ranking(collapsed, raw_scores=(0.4, 0.2)),
        collapsed_counterfactuals=collapsed,
        observations=(),
        primary=_BaseHypothesis(origin_time_ms=0.0, bpm=128.0, score=1.0),
        beat_signal=np.zeros(1, dtype=np.float64),
        frame_rate_hz=50.0,
        candidate_sources=("global_constant", _SHORT_WINDOW_MONOTONE_CHAIN_SOURCE),
        config=config,
    )
    monkeypatch.setattr(
        tempo_track_module,
        "_generalized_support_delta",
        lambda *_args, **_kwargs: -1.0,
    )
    negative_beatthis_gain = _select_exp014_production_candidate(
        candidates,
        raw_self_ranking=_ranking(candidates, raw_scores=(0.4, 1.5)),
        collapsed_raw_self_ranking=_ranking(collapsed, raw_scores=(0.4, 0.2)),
        collapsed_counterfactuals=collapsed,
        observations=(),
        primary=_BaseHypothesis(origin_time_ms=0.0, bpm=128.0, score=1.0),
        beat_signal=np.zeros(1, dtype=np.float64),
        frame_rate_hz=50.0,
        candidate_sources=("global_constant", _SHORT_WINDOW_MONOTONE_CHAIN_SOURCE),
        config=config,
    )

    assert _is_short_window_monotone_chain_jump(chain)
    assert selected.lane == "paired_jump"
    assert selected.selected_candidate_index == 1
    assert old_source.lane == "constant"
    assert old_source.selected_candidate_index == 0
    assert negative_raw_gain.selected_candidate_index == 0
    assert negative_beatthis_gain.selected_candidate_index == 0


def test_early_half_primary_prefix_step_uses_high_rank_pair_and_prefix_support() -> None:
    config = TempoTrackConfig()
    duration_ms = 50_000.0
    low_bpm = 114.0
    primary_bpm = 230.0
    boundary_beat = 41
    origin_time_ms = 100.0
    boundary_time_ms = origin_time_ms + boundary_beat * 60000.0 / low_bpm
    truth = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=origin_time_ms,
        sections=(
            ConstantTempoSection(0, boundary_beat, low_bpm),
            ConstantTempoSection(boundary_beat, 150, primary_bpm),
        ),
    )
    observations = tuple(
        _observation(source, center, low_bpm)
        for source in ("beatthis", "raw_audio")
        for center in (8000.0, 10_000.0, 12_000.0, 14_000.0, 16_000.0)
    )

    choices = _early_half_primary_prefix_step_choices(
        beat_signal=_beat_signal_from_curve(truth, duration_ms=duration_ms),
        frame_rate_hz=50.0,
        duration_ms=duration_ms,
        global_candidates=_global_candidates_from_curve(truth),
        primary=_BaseHypothesis(origin_time_ms=0.0, bpm=primary_bpm, score=1.0),
        boundary_seeds=(
            _BoundarySeed(time_ms=17_000.0, rank_score=420.0),
            _BoundarySeed(time_ms=boundary_time_ms + 40.0, rank_score=410.0),
        ),
        short_observations=observations,
        config=config,
    )

    assert len(choices) == 1
    choice = choices[0]
    curve = choice.proposal.curve
    assert choice.proposal.source == _EARLY_HALF_PRIMARY_PREFIX_STEP_SOURCE
    assert choice.proposal.collapse_bpm == primary_bpm
    assert len(curve.sections) == 2
    assert curve.sections[0].bpm == pytest.approx(low_bpm, abs=0.5)
    assert curve.sections[1].bpm == primary_bpm
    assert abs(curve.boundary_times_ms[1] - boundary_time_ms) <= 100.0
    assert choice.prefix_observation_count == len(observations)


def test_early_half_primary_prefix_fast_lane_challenges_only_constant() -> None:
    config = TempoTrackConfig()
    constant = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(ConstantTempoSection(0, 192, 230.0),),
    )
    step = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=100.0,
        sections=(
            ConstantTempoSection(0, 41, 114.0),
            ConstantTempoSection(41, 150, 230.0),
        ),
    )
    proposals = (
        _CurveProposal(
            curve=constant,
            source="global_constant",
            score=1.0,
            collapse_bpm=230.0,
        ),
        _CurveProposal(
            curve=step,
            source=_EARLY_HALF_PRIMARY_PREFIX_STEP_SOURCE,
            score=1.0,
            collapse_bpm=230.0,
        ),
    )
    collapsed = (
        constant,
        _collapsed_constant_counterfactual(
            step,
            duration_ms=max(constant.end_time_ms, step.end_time_ms),
            base_bpm=230.0,
            config=config,
        ),
    )
    choice = _EarlyHalfPrimaryPrefixStepChoice(
        proposal=proposals[1],
        left_boundary_time_ms=17_000.0,
        right_boundary_time_ms=21_700.0,
        boundary_time_ms=step.boundary_times_ms[1],
        left_rank_score=420.0,
        right_rank_score=410.0,
        beatthis_support_delta=0.0,
        prefix_observation_count=10,
    )
    fallback = TempoTrackProductionSelection(
        status="v3_accepted",
        selected_candidate_index=0,
        selected_fingerprint_sha256=constant.fingerprint_sha256,
        lane="constant",
        fallback_reason=None,
        raw_run=None,
        eligible_candidate_indices=(0,),
        raw_self_rank_by_candidate=((0, 1),),
        beatthis_aba_rank_by_candidate=(),
    )

    selected = _early_half_primary_prefix_fast_lane_selection(
        proposals,
        choices=(choice,),
        raw_self_ranking=_ranking((constant, step), raw_scores=(0.4, 0.50)),
        collapsed_raw_self_ranking=_ranking(collapsed, raw_scores=(0.4, 0.49)),
        collapsed_counterfactuals=collapsed,
        fallback=fallback,
        config=config,
    )
    already_jump = _early_half_primary_prefix_fast_lane_selection(
        proposals,
        choices=(choice,),
        raw_self_ranking=_ranking((constant, step), raw_scores=(0.4, 0.50)),
        collapsed_raw_self_ranking=_ranking(collapsed, raw_scores=(0.4, 0.49)),
        collapsed_counterfactuals=collapsed,
        fallback=TempoTrackProductionSelection(
            status="v3_accepted",
            selected_candidate_index=1,
            selected_fingerprint_sha256=step.fingerprint_sha256,
            lane="paired_jump",
            fallback_reason=None,
            raw_run=None,
            eligible_candidate_indices=(1,),
            raw_self_rank_by_candidate=((1, 1),),
            beatthis_aba_rank_by_candidate=((1, 1),),
        ),
        config=config,
    )

    assert selected.lane == "early_half_primary_prefix_step"
    assert selected.selected_candidate_index == 1
    assert selected.paired_raw_gain_by_candidate[0][1] == pytest.approx(0.01)
    assert already_jump.lane == "paired_jump"
    assert already_jump.selected_candidate_index == 1


def test_exp026_side_segment_selector_requires_four_way_regional_support() -> None:
    config = TempoTrackConfig()
    constant = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(ConstantTempoSection(0, 140, 150.0),),
    )
    jump = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 50, 150.0),
            ConstantTempoSection(50, 150, 170.0),
        ),
    )
    candidates = (constant, jump)
    duration_ms = max(candidate.end_time_ms for candidate in candidates)
    collapsed = tuple(
        _collapsed_constant_counterfactual(
            candidate,
            duration_ms=duration_ms,
            base_bpm=150.0,
            config=config,
        )
        for candidate in candidates
    )
    beat_signal = _beat_signal_from_curve(jump, duration_ms=duration_ms)
    assert _generalized_support_delta(
        beat_signal,
        frame_rate_hz=50.0,
        candidate=jump,
        collapsed=collapsed[1],
        radius_ms=config.beat_event_radius_ms,
    ) > 0.0

    selection = _select_exp014_production_candidate(
        candidates,
        raw_self_ranking=_ranking(candidates, raw_scores=(0.4, 1.5)),
        collapsed_raw_self_ranking=_ranking(collapsed, raw_scores=(0.4, 0.2)),
        collapsed_counterfactuals=collapsed,
        observations=(
            _observation("raw_audio", 16_000.0, 150.0),
            _observation("raw_audio", 24_000.0, 170.0),
            _observation("beatthis", 24_000.0, 170.0),
        ),
        primary=_BaseHypothesis(origin_time_ms=0.0, bpm=150.0, score=1.0),
        beat_signal=beat_signal,
        frame_rate_hz=50.0,
        candidate_sources=("global_constant", "observation_side_segment"),
        config=config,
    )

    assert selection.status == "v3_accepted"
    assert selection.lane == "constant"
    assert selection.eligible_candidate_indices == (0,)
    assert selection.selected_candidate_index == 0


def test_exp026_side_segment_two_section_requires_one_eighth_first_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TempoTrackConfig()
    constant = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(ConstantTempoSection(0, 110, 150.0),),
    )
    target_first_ratio = 0.1206
    first_duration_seconds = 10 * 60.0 / 150.0
    second_duration_seconds = (
        first_duration_seconds * (1.0 - target_first_ratio) / target_first_ratio
    )
    rejected_first = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 10, 150.0),
            ConstantTempoSection(10, 110, 100 * 60.0 / second_duration_seconds),
        ),
    )
    allowed_short_terminal = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 100, 150.0),
            ConstantTempoSection(100, 110, 120.0),
        ),
    )

    def select(jump: PhaseContinuousTimingCurve):
        candidates = (constant, jump)
        duration_ms = max(candidate.end_time_ms for candidate in candidates)
        collapsed = tuple(
            _collapsed_constant_counterfactual(
                candidate,
                duration_ms=duration_ms,
                base_bpm=150.0,
                config=config,
            )
            for candidate in candidates
        )
        return _select_exp014_production_candidate(
            candidates,
            raw_self_ranking=_ranking(candidates, raw_scores=(0.4, 1.5)),
            collapsed_raw_self_ranking=_ranking(collapsed, raw_scores=(0.4, 0.2)),
            collapsed_counterfactuals=collapsed,
            observations=(),
            primary=_BaseHypothesis(origin_time_ms=0.0, bpm=150.0, score=1.0),
            beat_signal=np.zeros(1, dtype=np.float64),
            frame_rate_hz=50.0,
            candidate_sources=("global_constant", "observation_side_segment"),
            config=config,
        )

    monkeypatch.setattr(
        tempo_track_module,
        "_generalized_support_delta",
        lambda *_args, **_kwargs: 1.0,
    )
    monkeypatch.setattr(
        tempo_track_module,
        "_observation_side_single_boundary_has_cross_source_support",
        lambda *_args, **_kwargs: True,
    )
    below_threshold = select(rejected_first)
    terminal_short = select(allowed_short_terminal)

    rejected_ratio = (
        rejected_first.sections[0].duration_seconds
        / sum(section.duration_seconds for section in rejected_first.sections)
    )
    terminal_ratio = (
        allowed_short_terminal.sections[1].duration_seconds
        / sum(section.duration_seconds for section in allowed_short_terminal.sections)
    )

    assert rejected_ratio == pytest.approx(0.1206)
    assert terminal_ratio == pytest.approx(1.0 / 9.0)
    assert below_threshold.lane == "constant"
    assert below_threshold.selected_candidate_index == 0
    assert terminal_short.lane == "paired_jump"
    assert terminal_short.selected_candidate_index == 1


def test_exp026_side_segment_closed_aba_requires_middle_and_right_mode_support() -> None:
    config = TempoTrackConfig()
    constant = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(ConstantTempoSection(0, 183, 180.0),),
    )
    aba = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 60, 180.0),
            ConstantTempoSection(60, 75, 150.0),
            ConstantTempoSection(75, 180, 180.0),
        ),
    )
    candidates = (constant, aba)
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

    unsupported = _select_exp014_production_candidate(
        candidates,
        raw_self_ranking=_ranking(candidates, raw_scores=(0.4, 1.5)),
        collapsed_raw_self_ranking=_ranking(collapsed, raw_scores=(0.4, 0.2)),
        collapsed_counterfactuals=collapsed,
        observations=(),
        primary=_BaseHypothesis(origin_time_ms=0.0, bpm=180.0, score=1.0),
        beat_signal=_beat_signal_from_curve(aba, duration_ms=duration_ms),
        frame_rate_hz=50.0,
        candidate_sources=("global_constant", "observation_side_segment"),
        config=config,
    )

    middle_support = (_observation("beatthis", 23_000.0, 150.0),)
    missing_raw_right = _select_exp014_production_candidate(
        candidates,
        raw_self_ranking=_ranking(candidates, raw_scores=(0.4, 1.5)),
        collapsed_raw_self_ranking=_ranking(collapsed, raw_scores=(0.4, 0.2)),
        collapsed_counterfactuals=collapsed,
        observations=middle_support + (_observation("beatthis", 35_000.0, 180.0),),
        primary=_BaseHypothesis(origin_time_ms=0.0, bpm=180.0, score=1.0),
        beat_signal=_beat_signal_from_curve(aba, duration_ms=duration_ms),
        frame_rate_hz=50.0,
        candidate_sources=("global_constant", "observation_side_segment"),
        config=config,
    )
    missing_beatthis_right = _select_exp014_production_candidate(
        candidates,
        raw_self_ranking=_ranking(candidates, raw_scores=(0.4, 1.5)),
        collapsed_raw_self_ranking=_ranking(collapsed, raw_scores=(0.4, 0.2)),
        collapsed_counterfactuals=collapsed,
        observations=middle_support + (_observation("raw_audio", 35_000.0, 180.0),),
        primary=_BaseHypothesis(origin_time_ms=0.0, bpm=180.0, score=1.0),
        beat_signal=_beat_signal_from_curve(aba, duration_ms=duration_ms),
        frame_rate_hz=50.0,
        candidate_sources=("global_constant", "observation_side_segment"),
        config=config,
    )
    outside_raw_tolerance = _select_exp014_production_candidate(
        candidates,
        raw_self_ranking=_ranking(candidates, raw_scores=(0.4, 1.5)),
        collapsed_raw_self_ranking=_ranking(collapsed, raw_scores=(0.4, 0.2)),
        collapsed_counterfactuals=collapsed,
        observations=middle_support
        + (
            _observation("beatthis", 35_000.0, 180.0),
            _observation("raw_audio", 35_000.0, 182.0),
        ),
        primary=_BaseHypothesis(origin_time_ms=0.0, bpm=180.0, score=1.0),
        beat_signal=_beat_signal_from_curve(aba, duration_ms=duration_ms),
        frame_rate_hz=50.0,
        candidate_sources=("global_constant", "observation_side_segment"),
        config=config,
    )
    right_support = (
        _observation("beatthis", 35_000.0, 180.0),
        _observation("raw_audio", 35_000.0, 180.0),
    )
    supported = _select_exp014_production_candidate(
        candidates,
        raw_self_ranking=_ranking(candidates, raw_scores=(0.4, 1.5)),
        collapsed_raw_self_ranking=_ranking(collapsed, raw_scores=(0.4, 0.2)),
        collapsed_counterfactuals=collapsed,
        observations=middle_support + right_support,
        primary=_BaseHypothesis(origin_time_ms=0.0, bpm=180.0, score=1.0),
        beat_signal=_beat_signal_from_curve(aba, duration_ms=duration_ms),
        frame_rate_hz=50.0,
        candidate_sources=("global_constant", "observation_side_segment"),
        config=config,
    )
    below_raw_gain_floor = _select_exp014_production_candidate(
        candidates,
        raw_self_ranking=_ranking(candidates, raw_scores=(0.4, 0.204)),
        collapsed_raw_self_ranking=_ranking(collapsed, raw_scores=(0.4, 0.2)),
        collapsed_counterfactuals=collapsed,
        observations=middle_support + right_support,
        primary=_BaseHypothesis(origin_time_ms=0.0, bpm=180.0, score=1.0),
        beat_signal=_beat_signal_from_curve(aba, duration_ms=duration_ms),
        frame_rate_hz=50.0,
        candidate_sources=("global_constant", "observation_side_segment"),
        config=config,
    )

    assert unsupported.status == "v3_accepted"
    assert unsupported.lane == "constant"
    assert unsupported.eligible_candidate_indices == (0,)
    assert unsupported.selected_candidate_index == 0
    assert missing_raw_right.lane == "constant"
    assert missing_raw_right.selected_candidate_index == 0
    assert missing_beatthis_right.lane == "constant"
    assert missing_beatthis_right.selected_candidate_index == 0
    assert outside_raw_tolerance.lane == "constant"
    assert outside_raw_tolerance.selected_candidate_index == 0
    assert supported.status == "v3_accepted"
    assert supported.lane == "paired_jump"
    assert supported.eligible_candidate_indices == (1,)
    assert supported.selected_candidate_index == 1
    assert below_raw_gain_floor.lane == "constant"
    assert below_raw_gain_floor.selected_candidate_index == 0


def test_exp026_side_segment_octave_bucket_uses_absolute_raw_representative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = TempoTrackConfig()
    constant = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(ConstantTempoSection(0, 480, 180.0),),
    )
    favored_double = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 60, 180.0),
            ConstantTempoSection(60, 160, 166.0),
        ),
    )
    competing_double = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 57, 180.0),
            ConstantTempoSection(57, 160, 166.0),
        ),
    )
    raw_preferred_direct = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 30, 90.0),
            ConstantTempoSection(30, 80, 83.0),
        ),
    )
    candidates = (
        constant,
        favored_double,
        competing_double,
        raw_preferred_direct,
    )
    duration_ms = max(candidate.end_time_ms for candidate in candidates)
    collapsed = tuple(
        _collapsed_constant_counterfactual(
            candidate,
            duration_ms=duration_ms,
            base_bpm=float(candidate.sections[0].bpm),
            config=config,
        )
        for candidate in candidates
    )
    beatthis_delta_by_fingerprint = {
        favored_double.fingerprint_sha256: 2.0,
        competing_double.fingerprint_sha256: 3.0,
        raw_preferred_direct.fingerprint_sha256: 1.0,
    }
    monkeypatch.setattr(
        tempo_track_module,
        "_generalized_support_delta",
        lambda _signal, *, candidate, **_kwargs: beatthis_delta_by_fingerprint[
            candidate.fingerprint_sha256
        ],
    )

    selection = _select_exp014_production_candidate(
        candidates,
        raw_self_ranking=_ranking(candidates, raw_scores=(0.4, 2.0, 1.5, 3.0)),
        collapsed_raw_self_ranking=_ranking(
            collapsed,
            raw_scores=(0.4, 0.2, 0.2, 2.8),
        ),
        collapsed_counterfactuals=collapsed,
        observations=tuple(
            _observation(source, center, bpm)
            for source in ("beatthis", "raw_audio")
            for center, bpm in ((15_000.0, 90.0), (23_000.0, 83.0))
        ),
        primary=_BaseHypothesis(origin_time_ms=0.0, bpm=180.0, score=1.0),
        beat_signal=np.zeros(1, dtype=np.float64),
        frame_rate_hz=50.0,
        candidate_sources=(
            "global_constant",
            "observation_side_segment",
            "observation_side_segment",
            "observation_side_segment",
        ),
        candidate_generation_scores=(0.0, 3.0, 2.8, 1.0),
        config=config,
    )

    assert selection.status == "v3_accepted"
    assert selection.eligible_candidate_indices == (1, 2, 3)
    assert selection.selected_candidate_index == 3


def test_exp026_observed_pair_dominates_virtual_when_raw_and_generation_agree() -> None:
    paired = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 160, 175.0),
            ConstantTempoSection(160, 170, 144.0),
            ConstantTempoSection(170, 200, 175.0),
        ),
    )
    virtual = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 165, 175.0),
            ConstantTempoSection(165, 176, 170.0),
            ConstantTempoSection(176, 200, 175.0),
        ),
    )
    candidates = (paired, virtual)
    positive = {
        0: (0, 0.0124, 0.5, "paired", 0.0013, None),
        1: (1, 0.0120, 0.5, "virtual", 0.0076, None),
    }

    assert tempo_track_module._prefer_observed_pair_over_virtual_right(
        1,
        borda_buckets=((0, (0,)), (1, (1,))),
        candidates=candidates,
        sources=("paired_unmerged_boundary", "virtual_right_beatthis"),
        generation_scores=(1.72, 1.65),
        positive_structure_by_index=positive,
    ) == 0
    assert tempo_track_module._prefer_observed_pair_over_virtual_right(
        1,
        borda_buckets=((0, (0,)), (1, (1,))),
        candidates=candidates,
        sources=("paired_unmerged_boundary", "virtual_right_beatthis"),
        generation_scores=(1.60, 1.65),
        positive_structure_by_index=positive,
    ) == 1


def test_exp026_octave_bucket_does_not_use_nominal_prior_for_jump() -> None:
    config = TempoTrackConfig()
    constant = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(ConstantTempoSection(0, 200, 125.0),),
    )
    nominal = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 100, 125.0),
            ConstantTempoSection(100, 200, 140.0),
        ),
    )
    doubled = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 200, 250.0),
            ConstantTempoSection(200, 400, 280.0),
        ),
    )
    candidates = (constant, nominal, doubled)
    duration_ms = max(candidate.end_time_ms for candidate in candidates)
    collapsed = tuple(
        _collapsed_constant_counterfactual(
            candidate,
            duration_ms=duration_ms,
            base_bpm=float(candidate.sections[0].bpm),
            config=config,
        )
        for candidate in candidates
    )
    beatthis_delta_by_fingerprint = {
        nominal.fingerprint_sha256: 0.48,
        doubled.fingerprint_sha256: 0.31,
    }
    original_delta = tempo_track_module._generalized_support_delta
    try:
        tempo_track_module._generalized_support_delta = (
            lambda _signal, *, candidate, **_kwargs: beatthis_delta_by_fingerprint[
                candidate.fingerprint_sha256
            ]
        )
        selection = _select_exp014_production_candidate(
            candidates,
            raw_self_ranking=_ranking(candidates, raw_scores=(0.4, 0.47, 0.70)),
            collapsed_raw_self_ranking=_ranking(
                collapsed,
                raw_scores=(0.4, 0.50, 0.50),
            ),
            collapsed_counterfactuals=collapsed,
            observations=tuple(
                _observation(source, center, bpm)
                for source in ("beatthis", "raw_audio")
                for center, bpm in ((40_000.0, 125.0), (56_000.0, 140.0))
            ),
            primary=_BaseHypothesis(origin_time_ms=0.0, bpm=125.0, score=1.0),
            beat_signal=np.zeros(1, dtype=np.float64),
            frame_rate_hz=50.0,
            candidate_sources=(
                "global_constant",
                "observation_side_segment",
                "observation_side_segment",
            ),
            candidate_generation_scores=(0.0, 1.0, 2.0),
            config=config,
        )
    finally:
        tempo_track_module._generalized_support_delta = original_delta

    assert selection.eligible_candidate_indices == (1, 2)
    assert selection.selected_candidate_index == 2


def test_exp026_anchored_jump_needs_positive_beatthis_support() -> None:
    config = TempoTrackConfig()
    constant = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(ConstantTempoSection(0, 140, 150.0),),
    )
    jump = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 60, 150.0),
            ConstantTempoSection(60, 160, 170.0),
        ),
    )
    candidates = (constant, jump)
    duration_ms = max(candidate.end_time_ms for candidate in candidates)
    collapsed = tuple(
        _collapsed_constant_counterfactual(
            candidate,
            duration_ms=duration_ms,
            base_bpm=150.0,
            config=config,
        )
        for candidate in candidates
    )

    selection = _select_exp014_production_candidate(
        candidates,
        raw_self_ranking=_ranking(candidates, raw_scores=(0.4, 1.5)),
        collapsed_raw_self_ranking=_ranking(collapsed, raw_scores=(0.4, 0.2)),
        collapsed_counterfactuals=collapsed,
        observations=(),
        primary=_BaseHypothesis(origin_time_ms=0.0, bpm=150.0, score=1.0),
        beat_signal=np.zeros(int(math.ceil(duration_ms * 50.0 / 1000.0))),
        frame_rate_hz=50.0,
        candidate_sources=("global_constant", "observation_side_segment"),
        config=config,
    )

    assert selection.status == "v3_accepted"
    assert selection.lane == "constant"
    assert selection.selected_candidate_index == 0


def test_exp026_existing_paired_source_keeps_legacy_beatthis_delta_semantics() -> None:
    config = TempoTrackConfig()
    constant = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(ConstantTempoSection(0, 183, 180.0),),
    )
    paired = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 60, 180.0),
            ConstantTempoSection(60, 75, 150.0),
            ConstantTempoSection(75, 180, 180.0),
        ),
    )
    candidates = (constant, paired)
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
    beat_signal = _beat_signal_from_curve(collapsed[1], duration_ms=duration_ms)
    beatthis_delta = _generalized_support_delta(
        beat_signal,
        frame_rate_hz=50.0,
        candidate=paired,
        collapsed=collapsed[1],
        radius_ms=config.beat_event_radius_ms,
    )

    selection = _select_exp014_production_candidate(
        candidates,
        raw_self_ranking=_ranking(candidates, raw_scores=(0.4, 1.5)),
        collapsed_raw_self_ranking=_ranking(collapsed, raw_scores=(0.4, 0.2)),
        collapsed_counterfactuals=collapsed,
        observations=tuple(
            _observation("raw_audio", center, 150.0)
            for center in (22_500.0, 23_000.0, 23_500.0)
        ),
        primary=_BaseHypothesis(origin_time_ms=0.0, bpm=180.0, score=1.0),
        beat_signal=beat_signal,
        frame_rate_hz=50.0,
        candidate_sources=("global_constant", "paired_unmerged_boundary"),
        config=config,
    )

    gain_by_index = {
        index: raw_gain
        for index, raw_gain, _, _ in selection.paired_raw_gain_by_candidate
    }
    assert gain_by_index[1] > 0.0
    assert beatthis_delta <= 0.0
    assert selection.status == "v3_accepted"
    assert selection.lane == "paired_jump"
    assert selection.eligible_candidate_indices == (1,)
    assert selection.selected_candidate_index == 1


def test_exp026_constant_fixture_preserves_constant_selection_and_caps() -> None:
    truth = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=120.0,
        sections=(ConstantTempoSection(0, 160, 150.0),),
    )
    duration_ms = truth.end_time_ms + 500.0
    prediction = _prediction_from_curve(truth, duration_ms=duration_ms)
    evidence = _raw_evidence_from_curve(truth, duration_ms=duration_ms)

    first = generate_timing_candidates(prediction, audio_evidence=evidence)
    second = generate_timing_candidates(prediction, audio_evidence=evidence)

    assert first.production_selection.status == "v3_accepted"
    assert first.production_selection.lane == "constant"
    assert first.selected_candidate is not None
    assert first.selected_candidate.curve_class == "constant"
    assert len(first.candidates) <= 64
    assert sum(candidate.curve_class == "jump" for candidate in first.candidates) <= 44
    assert tuple(candidate.fingerprint_sha256 for candidate in first.candidates) == tuple(
        candidate.fingerprint_sha256 for candidate in second.candidates
    )


def _find_sections(
    proposals: tuple[object, ...],
    bpms: tuple[float, ...],
    *,
    tolerance_bpm: float = 0.5,
) -> PhaseContinuousTimingCurve | None:
    for proposal in proposals:
        curve = proposal.curve
        if len(curve.sections) != len(bpms):
            continue
        if all(
            isinstance(section, ConstantTempoSection)
            and abs(section.bpm - expected_bpm) <= tolerance_bpm
            for section, expected_bpm in zip(curve.sections, bpms)
        ):
            return curve
    return None


def _observation(source: str, center_time_ms: float, bpm: float) -> LocalTempoObservation:
    return LocalTempoObservation(
        source=source,
        center_time_ms=center_time_ms,
        window_start_ms=center_time_ms - 3000.0,
        window_end_ms=center_time_ms + 3000.0,
        bpm=bpm,
        strength=0.9,
    )


def _global_candidates_from_curve(curve: PhaseContinuousTimingCurve) -> object:
    return SimpleNamespace(
        beat_peaks=tuple(
            SimpleNamespace(
                time_ms=curve.time_at_beat(float(beat)),
                confidence=1.0,
            )
            for beat in range(curve.start_beat, min(curve.end_beat, 192))
        )
    )


def _beat_signal_from_curve(
    curve: PhaseContinuousTimingCurve,
    *,
    duration_ms: float,
    frame_rate_hz: float = 50.0,
) -> np.ndarray:
    frame_count = int(math.ceil(duration_ms * frame_rate_hz / 1000.0))
    signal = np.zeros(frame_count, dtype=np.float64)
    for beat in range(curve.start_beat, curve.end_beat):
        _write_pulse(signal, time_ms=curve.time_at_beat(float(beat)), frame_rate_hz=frame_rate_hz)
    return signal


def _prediction_from_curve(
    curve: PhaseContinuousTimingCurve,
    *,
    duration_ms: float,
    frame_rate_hz: float = 50.0,
) -> FrameTimingPrediction:
    beat_prob = _beat_signal_from_curve(
        curve,
        duration_ms=duration_ms,
        frame_rate_hz=frame_rate_hz,
    ).astype(np.float32)
    downbeat_prob = np.zeros_like(beat_prob)
    for beat in range(curve.start_beat, curve.end_beat, 4):
        _write_pulse(
            downbeat_prob,
            time_ms=curve.time_at_beat(float(beat)),
            frame_rate_hz=frame_rate_hz,
        )
    return FrameTimingPrediction(
        provider="cached-beatthis",
        beat_prob=beat_prob,
        downbeat_prob=downbeat_prob,
        frame_rate_hz=frame_rate_hz,
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
        evidence=_empty_evidence(),
        candidate_scores=scores,
        ranked_scores=tuple(
            sorted(scores, key=lambda score: (-score.raw_score, score.fingerprint_sha256))
        ),
        common_beat_indices=(),
        complete_window_start_beats=(),
        unavailable_reason=None,
    )


def _empty_evidence() -> RawAudioEvidence:
    centers = np.asarray((0.01, 0.02, 0.03), dtype=np.float64)
    return RawAudioEvidence(
        frame_center_seconds=centers,
        band_flux=np.zeros((centers.size, 4), dtype=np.float32),
        band_percentile95=(1.0, 1.0, 1.0, 1.0),
        input_frame_count=centers.size,
        valid_frame_count=centers.size,
        audio_duration_seconds=0.04,
    )


def _write_pulse(
    signal: np.ndarray,
    *,
    time_ms: float,
    frame_rate_hz: float,
) -> None:
    frame = int(round(time_ms * frame_rate_hz / 1000.0))
    if 0 <= frame < signal.size:
        signal[frame] = 1.0
    for neighbor in (frame - 1, frame + 1):
        if 0 <= neighbor < signal.size:
            signal[neighbor] = max(signal[neighbor], 0.1)
