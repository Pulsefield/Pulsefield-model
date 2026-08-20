from __future__ import annotations

import math

from pulsefield_model.timing.v3.analytic_curve import (
    ConstantTempoSection,
    PhaseContinuousTimingCurve,
)
from pulsefield_model.timing.v3.tempo_track import (
    TEMPO_TRACK_RESULT_DUMP_SCHEMA_VERSION,
    TEMPO_TRACK_VERSION,
    _CurveProposal,
    _deduped_jump_retention_order,
    _dominates_short_aba_pareto_candidate,
    _jump_retention_family,
    _retain_jump_proposals_by_family,
    _short_aba_first_pareto_front,
    _short_aba_support_retention_order_key,
    TempoTrackConfig,
)


def test_exp021_versions_bind_pareto_short_aba_retention() -> None:
    assert TEMPO_TRACK_VERSION == "pulsefield_model.timing_v3_tempo_track_exp021_v1"
    assert (
        TEMPO_TRACK_RESULT_DUMP_SCHEMA_VERSION
        == "pulsefield_model.timing_v3_tempo_track_result_dump_exp021_v1"
    )


def test_exp021_first_front_preserves_three_source_evidence_tradeoffs() -> None:
    score_extreme = _short_aba(0, score=100.0, support=0.0)
    compromise = _short_aba(1, score=90.0, support=10.0)
    support_extreme = _short_aba(2, score=0.0, support=20.0)
    dominated = tuple(
        _short_aba(10 + index, score=80.0 - index, support=9.0 - index)
        for index in range(20)
    )

    retained = _retain_jump_proposals_by_family(
        (score_extreme, compromise, support_extreme) + dominated,
        config=TempoTrackConfig(maximum_jump_candidates=14),
    )

    retained_fingerprints = _fingerprints(retained.proposals)
    assert len(retained.proposals) == 14
    assert score_extreme.curve.fingerprint_sha256 in retained_fingerprints
    assert compromise.curve.fingerprint_sha256 in retained_fingerprints
    assert support_extreme.curve.fingerprint_sha256 in retained_fingerprints
    assert _fingerprints(retained.proposals[:3]) == _fingerprints(
        (support_extreme, compromise, score_extreme)
    )


def test_exp021_exact_dominance_ties_and_tiny_tradeoffs() -> None:
    equal_a = _short_aba(0, score=5.0, support=7.0)
    equal_b = _short_aba(1, score=5.0, support=7.0)
    lower_support = _short_aba(2, score=5.0, support=6.0)
    tiny_score = _short_aba(3, score=5.0 + 1e-12, support=7.0)
    tiny_support = _short_aba(4, score=5.0, support=7.0 + 1e-12)

    assert _dominates_short_aba_pareto_candidate(equal_a, lower_support)
    assert not _dominates_short_aba_pareto_candidate(equal_a, equal_b)
    assert not _dominates_short_aba_pareto_candidate(tiny_score, tiny_support)
    assert not _dominates_short_aba_pareto_candidate(tiny_support, tiny_score)

    front = _short_aba_first_pareto_front(
        (equal_a, equal_b, lower_support, tiny_score, tiny_support)
    )
    assert set(_fingerprints(front)) == set(
        _fingerprints((tiny_score, tiny_support))
    )


def test_exp021_fast_skyline_matches_naive_exact_dominance() -> None:
    proposals = tuple(
        _short_aba(
            index,
            score=float((index * 17) % 13),
            support=float((index * 29) % 11),
        )
        for index in range(80)
    )
    expected = tuple(
        proposal
        for proposal in proposals
        if not any(
            other is not proposal
            and _dominates_short_aba_pareto_candidate(other, proposal)
            for other in proposals
        )
    )

    actual = _short_aba_first_pareto_front(tuple(reversed(proposals)))

    assert set(_fingerprints(actual)) == set(_fingerprints(expected))


def test_exp021_nonfinite_objectives_do_not_enter_front() -> None:
    finite = _short_aba(0, score=10.0, support=10.0)
    nonfinite = (
        _short_aba(1, score=math.nan, support=100.0),
        _short_aba(2, score=math.inf, support=99.0),
        _short_aba(3, score=-math.inf, support=98.0),
        _short_aba(4, score=9.0, support=None),
        _short_aba(5, score=9.0, support=math.nan),
        _short_aba(6, score=9.0, support=math.inf),
        _short_aba(7, score=9.0, support=-math.inf),
    )

    front = _short_aba_first_pareto_front((finite,) + nonfinite)
    ordered = _retain_jump_proposals_by_family(
        (finite,) + nonfinite,
        config=TempoTrackConfig(maximum_jump_candidates=8),
    ).proposals

    assert _fingerprints(front) == _fingerprints((finite,))
    assert ordered[0].curve.fingerprint_sha256 == finite.curve.fingerprint_sha256
    assert set(_fingerprints(ordered)) == set(_fingerprints((finite,) + nonfinite))


def test_exp021_front_overflow_keeps_quota_and_existing_support_key() -> None:
    frontier = tuple(
        _short_aba(index, score=100.0 - index, support=float(index))
        for index in range(20)
    )
    expected = tuple(
        sorted(frontier, key=_short_aba_support_retention_order_key)[:14]
    )

    retained = _retain_jump_proposals_by_family(
        frontier,
        config=TempoTrackConfig(maximum_jump_candidates=14),
    )

    assert len(retained.proposals) == 14
    assert _fingerprints(retained.proposals) == _fingerprints(expected)


def test_exp021_pre_family_dedupe_keeps_high_blended_representative() -> None:
    high_blended = _short_aba(0, score=100.0, support=-100.0)
    high_support_duplicate = _CurveProposal(
        curve=high_blended.curve,
        source="virtual_right_raw_audio",
        score=1.0,
        collapse_bpm=high_blended.collapse_bpm,
        aba_support_delta=100.0,
    )
    support_extreme = _short_aba(1, score=0.0, support=10.0)

    deduped = _deduped_jump_retention_order(
        (high_support_duplicate, support_extreme, high_blended)
    )
    retained = _retain_jump_proposals_by_family(
        (high_support_duplicate, support_extreme, high_blended),
        config=TempoTrackConfig(maximum_jump_candidates=2),
    )

    representative = next(
        proposal
        for proposal in deduped
        if proposal.curve.fingerprint_sha256
        == high_blended.curve.fingerprint_sha256
    )
    retained_representative = next(
        proposal
        for proposal in retained.proposals
        if proposal.curve.fingerprint_sha256
        == high_blended.curve.fingerprint_sha256
    )
    assert representative is high_blended
    assert retained_representative is high_blended
    assert retained_representative.aba_support_delta == -100.0


def test_exp021_non_short_order_global_backfill_and_caps_are_isolated() -> None:
    config = TempoTrackConfig(maximum_jump_candidates=24)
    short = tuple(
        _short_aba(index, score=100.0 - index, support=float(index))
        for index in range(18)
    )
    persistent = tuple(_persistent(index, score=80.0 - index) for index in range(4))
    overflow = tuple(_overflow(index, score=60.0 - index) for index in range(12))
    proposals = short + persistent + overflow
    global_order = _deduped_jump_retention_order(proposals)

    retained = _retain_jump_proposals_by_family(proposals, config=config)

    retained_persistent = tuple(
        proposal
        for proposal in retained.proposals
        if _jump_retention_family(proposal, config=config) == "persistent"
    )
    expected_persistent = tuple(
        proposal
        for proposal in global_order
        if _jump_retention_family(proposal, config=config) == "persistent"
    )
    assert _fingerprints(retained_persistent) == _fingerprints(expected_persistent)
    assert len(retained.proposals) == config.maximum_jump_candidates
    assert config.maximum_jump_candidates <= config.maximum_candidates


def _short_aba(
    index: int,
    *,
    score: float,
    support: float | None,
) -> _CurveProposal:
    base_bpm = 180.0 + 0.01 * index
    middle_bpm = 143.0 + 0.01 * index
    curve = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=float(index),
        sections=(
            ConstantTempoSection(0, 60 + index, base_bpm),
            ConstantTempoSection(60 + index, 70 + index, middle_bpm),
            ConstantTempoSection(70 + index, 180 + index, base_bpm),
        ),
    )
    return _CurveProposal(
        curve=curve,
        source="paired_unmerged_boundary",
        score=score,
        collapse_bpm=base_bpm,
        aba_support_delta=support,
    )


def _persistent(index: int, *, score: float) -> _CurveProposal:
    base_bpm = 180.0 + 0.01 * index
    curve = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=float(index),
        sections=(
            ConstantTempoSection(0, 40 + index, base_bpm),
            ConstantTempoSection(40 + index, 220 + index, 150.0 + 0.01 * index),
        ),
    )
    return _CurveProposal(
        curve=curve,
        source="raw_run_persistent_a_to_b_start",
        score=score,
        collapse_bpm=base_bpm,
    )


def _overflow(index: int, *, score: float) -> _CurveProposal:
    base_bpm = 180.0 + 0.01 * index
    curve = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=float(index),
        sections=(
            ConstantTempoSection(0, 40 + index, base_bpm),
            ConstantTempoSection(40 + index, 50 + index, 150.0 + 0.01 * index),
            ConstantTempoSection(50 + index, 220 + index, base_bpm),
        ),
    )
    return _CurveProposal(
        curve=curve,
        source="raw_run_aba",
        score=score,
        collapse_bpm=base_bpm,
    )


def _fingerprints(proposals: tuple[_CurveProposal, ...]) -> tuple[str, ...]:
    return tuple(proposal.curve.fingerprint_sha256 for proposal in proposals)
