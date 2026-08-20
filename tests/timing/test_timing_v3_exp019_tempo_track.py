from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np

from pulsefield_model.timing.v3.analytic_curve import (
    ConstantTempoSection,
    LinearTimeRampSection,
    PhaseContinuousTimingCurve,
)
from pulsefield_model.timing.v3.tempo_track import (
    TEMPO_TRACK_RESULT_DUMP_SCHEMA_VERSION,
    TEMPO_TRACK_VERSION,
    _BaseHypothesis,
    _bounded_proposals,
    _CurveProposal,
    _deduped_jump_retention_order,
    _is_phase1_piecewise_constant_jump,
    _JumpRetentionFamily,
    _jump_proposals,
    _jump_retention_family,
    _PairSeed,
    _retain_jump_proposals_by_family,
    TempoTrackConfig,
    TempoTrackDiagnostics,
    TempoTrackResult,
    TimingCandidateDiagnostic,
    tempo_track_result_to_dict,
)


def test_exp019_short_reserved_slots_prefer_pure_support_before_blended_score() -> None:
    target = _short_aba(99, score=-1.0, support=100.0)
    high_blended = tuple(
        _short_aba(index, score=1000.0 - index, support=float(index))
        for index in range(15)
    )
    proposals = high_blended + (target,)

    retained = _retain_jump_proposals_by_family(
        proposals,
        config=TempoTrackConfig(maximum_jump_candidates=14),
    )
    retained_fingerprints = _fingerprints(retained.proposals)

    assert len(retained.proposals) == 14
    assert target.curve.fingerprint_sha256 in retained_fingerprints
    assert high_blended[0].curve.fingerprint_sha256 not in retained_fingerprints
    assert high_blended[1].curve.fingerprint_sha256 not in retained_fingerprints


def test_exp019_short_reserved_key_handles_none_and_ties_deterministically() -> None:
    proposals = (
        _short_aba(0, score=1.0, support=3.0),
        _short_aba(1, score=100.0, support=2.0),
        _short_aba(2, score=50.0, support=2.0),
        _short_aba(3, score=1000.0, support=None),
        _short_aba(4, score=99.0, support=1.0),
        _short_aba(5, score=100.0, support=2.0, source="virtual_right_raw_audio"),
    )

    retained = _retain_jump_proposals_by_family(
        proposals,
        config=TempoTrackConfig(maximum_jump_candidates=len(proposals)),
    )

    assert _fingerprints(retained.proposals) == _fingerprints(
        (
            proposals[0],
            proposals[1],
            proposals[5],
            proposals[2],
            proposals[4],
            proposals[3],
        )
    )


def test_exp019_missing_support_cannot_displace_finite_support_in_reserved_pass() -> None:
    finite = tuple(
        _short_aba(index, score=float(index), support=1.0 + index)
        for index in range(14)
    )
    unsupported_high_score = _short_aba(90, score=10_000.0, support=None)

    retained = _retain_jump_proposals_by_family(
        finite + (unsupported_high_score,),
        config=TempoTrackConfig(maximum_jump_candidates=14),
    )

    assert unsupported_high_score.curve.fingerprint_sha256 not in _fingerprints(
        retained.proposals
    )
    assert _fingerprints(retained.proposals) == _fingerprints(
        tuple(reversed(finite))
    )


def test_exp019_generated_support_state_is_short_pair_virtual_only() -> None:
    config = TempoTrackConfig()
    batch = _jump_proposals(
        np.zeros(4000, dtype=np.float64),
        frame_rate_hz=50.0,
        duration_ms=80_000.0,
        shared_end_beat=200,
        bases=(_BaseHypothesis(origin_time_ms=0.0, bpm=180.0, score=1.0),),
        observations=(),
        pair_seeds=(
            _PairSeed(
                left_time_ms=12_000.0,
                right_time_ms=16_000.0,
                rank_score=1.0,
                source="paired_unmerged_boundary",
            ),
            _PairSeed(
                left_time_ms=12_000.0,
                right_time_ms=22_000.0,
                rank_score=100.0,
                source="virtual_right_raw_audio",
                preferred_bpm=120.0,
            ),
        ),
        config=config,
    )

    short_pair_or_virtual = tuple(
        proposal
        for proposal in batch.proposals
        if _jump_retention_family(proposal, config=config)
        == "short_aba_paired_boundary"
    )
    long_pair_or_virtual = tuple(
        proposal
        for proposal in batch.proposals
        if _jump_retention_family(proposal, config=config) == "long_aba"
    )
    non_short = (
        _constant(0, score=1.0),
        _long_aba(0, score=1.0, source="paired_unmerged_boundary"),
        _long_aba(1, score=1.0, source="virtual_right_raw_audio"),
        _long_aba(2, score=1.0, source="raw_run_aba"),
        _overflow(0, score=1.0),
        _ramp_like_diagnostic(0, score=1.0),
    )

    assert short_pair_or_virtual
    assert all(
        proposal.aba_support_delta is not None
        and math.isfinite(proposal.aba_support_delta)
        for proposal in short_pair_or_virtual
    )
    assert long_pair_or_virtual
    assert all(proposal.aba_support_delta is None for proposal in long_pair_or_virtual)
    assert all(proposal.aba_support_delta is None for proposal in non_short)


def test_exp019_non_short_reserved_order_and_global_backfill_ignore_support() -> None:
    config = TempoTrackConfig(maximum_jump_candidates=32)
    proposals = (
        tuple(_short_aba(index, score=10.0 - index, support=float(index)) for index in range(4))
        + tuple(
            _persistent(index, score=100.0 - index, support=1000.0 + index)
            for index in range(4)
        )
        + tuple(_long_aba(index, score=80.0 - index, support=900.0 + index) for index in range(3))
        + tuple(_multi_step(index, score=60.0 - index, support=800.0 + index) for index in range(2))
        + tuple(_overflow(index, score=40.0 - index, support=700.0 + index) for index in range(12))
    )

    retained = _retain_jump_proposals_by_family(proposals, config=config)
    global_order = _deduped_jump_retention_order(proposals)
    reserved_expected = _reserved_first_pass_expected(global_order, config=config)
    backfill_expected = []
    reserved_fingerprints = set(_fingerprints(tuple(reserved_expected)))
    for proposal in global_order:
        if proposal.curve.fingerprint_sha256 in reserved_fingerprints:
            continue
        backfill_expected.append(proposal)
        if len(reserved_expected) + len(backfill_expected) >= config.maximum_jump_candidates:
            break

    assert _fingerprints(retained.proposals[: len(reserved_expected)]) == _fingerprints(
        tuple(reserved_expected)
    )
    assert _fingerprints(retained.proposals[len(reserved_expected) :]) == _fingerprints(
        tuple(backfill_expected)
    )


def test_exp019_dedupe_determinism_and_caps_remain_unchanged() -> None:
    config = TempoTrackConfig()
    duplicate_source = _short_aba(777, score=300.0, support=0.5)
    duplicate_lower = _CurveProposal(
        curve=duplicate_source.curve,
        source="paired_unmerged_boundary",
        score=1.0,
        collapse_bpm=duplicate_source.collapse_bpm,
        aba_support_delta=999.0,
    )
    proposals = (
        tuple(_short_aba(index, score=100.0 - index, support=float(index)) for index in range(20))
        + tuple(_persistent(index, score=90.0 - index) for index in range(3))
        + tuple(_long_aba(index, score=80.0 - index) for index in range(2))
        + tuple(_multi_step(index, score=70.0 - index) for index in range(1))
        + tuple(_overflow(index, score=60.0 - index) for index in range(30))
        + (duplicate_source, duplicate_lower)
    )
    constants = tuple(_constant(index, score=1000.0 - index) for index in range(12))
    ramps = tuple(_ramp_like_diagnostic(index, score=10.0 - index) for index in range(8))

    first = _retain_jump_proposals_by_family(proposals, config=config)
    second = _retain_jump_proposals_by_family(proposals, config=config)
    bounded = _bounded_proposals(constants, first.proposals, ramps, config=config)

    assert _fingerprints(first.proposals) == _fingerprints(second.proposals)
    assert _family_counts(first.proposals, config=config) == _family_counts(
        second.proposals,
        config=config,
    )
    assert (
        sum(
            proposal.curve.fingerprint_sha256
            == duplicate_source.curve.fingerprint_sha256
            for proposal in first.proposals
        )
        == 1
    )
    assert config.maximum_jump_candidates == 44
    assert config.maximum_ramp_candidates == 8
    assert config.maximum_candidates == 64
    assert len(first.proposals) <= 44
    assert len(bounded) <= 64
    assert all(
        proposal.curve.fingerprint_sha256 in _fingerprints(bounded)
        for proposal in constants
    )
    assert sum(proposal.curve.curve_class == "ramp" for proposal in bounded) <= 8
    assert not _is_phase1_piecewise_constant_jump(ramps[0].curve)


def test_exp019_result_dump_provenance_excludes_support_state() -> None:
    proposal = _short_aba(0, score=1.0, support=2.5)
    result = TempoTrackResult(
        observations=(),
        candidates=(proposal.curve,),
        candidate_diagnostics=(
            TimingCandidateDiagnostic(
                fingerprint_sha256=proposal.curve.fingerprint_sha256,
                curve_class=proposal.curve.curve_class,
                source=proposal.source,
                generation_score=proposal.score,
            ),
        ),
        diagnostics=TempoTrackDiagnostics(
            version=TEMPO_TRACK_VERSION,
            beat_peak_count=0,
            raw_boundary_count=0,
            pair_seed_count=0,
            shared_start_beat=0,
            shared_end_beat=proposal.curve.end_beat,
            primary_origin_time_ms=proposal.curve.origin_time_ms,
            primary_bpm=proposal.collapse_bpm,
            candidate_count=1,
        ),
    )

    payload = tempo_track_result_to_dict(result)

    assert payload["schema"] == TEMPO_TRACK_RESULT_DUMP_SCHEMA_VERSION
    assert payload["tempo_track_version"] == TEMPO_TRACK_VERSION
    assert payload["candidates"][0]["generation_score"] == proposal.score
    assert "aba_support_delta" not in json.dumps(payload, sort_keys=True)


def _reserved_first_pass_expected(
    global_order: tuple[_CurveProposal, ...],
    *,
    config: TempoTrackConfig,
) -> tuple[_CurveProposal, ...]:
    short = tuple(
        sorted(
            (
                proposal
                for proposal in global_order
                if _jump_retention_family(proposal, config=config)
                == "short_aba_paired_boundary"
            ),
            key=_expected_short_support_key,
        )
    )[:14]
    retained: list[_CurveProposal] = list(short)
    retained_fingerprints = set(_fingerprints(tuple(retained)))
    quotas: tuple[tuple[_JumpRetentionFamily, int], ...] = (
        ("persistent", 10),
        ("long_aba", 8),
        ("multi_step", 6),
        ("overflow", 6),
    )
    for family, quota in quotas:
        family_retained = 0
        for proposal in global_order:
            if _jump_retention_family(proposal, config=config) != family:
                continue
            if proposal.curve.fingerprint_sha256 in retained_fingerprints:
                continue
            retained.append(proposal)
            retained_fingerprints.add(proposal.curve.fingerprint_sha256)
            family_retained += 1
            if family_retained >= quota:
                break
    return tuple(retained)


def _expected_short_support_key(
    proposal: _CurveProposal,
) -> tuple[int, float, float, str, str, tuple[float, ...]]:
    support = proposal.aba_support_delta
    finite = support is not None and math.isfinite(support)
    return (
        0 if finite else 1,
        -float(support) if finite else 0.0,
        -proposal.score,
        proposal.source,
        proposal.curve.fingerprint_sha256,
        proposal.curve.boundary_times_ms,
    )


def _family_counts(
    proposals: tuple[_CurveProposal, ...],
    *,
    config: TempoTrackConfig,
) -> Counter[str]:
    return Counter(_jump_retention_family(proposal, config=config) for proposal in proposals)


def _fingerprints(proposals: tuple[_CurveProposal, ...]) -> tuple[str, ...]:
    return tuple(proposal.curve.fingerprint_sha256 for proposal in proposals)


def _constant(index: int, *, score: float) -> _CurveProposal:
    bpm = 180.0 + 0.25 * index
    curve = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=float(index),
        sections=(ConstantTempoSection(0, 120 + index, bpm),),
    )
    return _CurveProposal(curve=curve, source="global_constant", score=score, collapse_bpm=bpm)


def _short_aba(
    index: int,
    *,
    score: float,
    support: float | None,
    source: str = "paired_unmerged_boundary",
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
        source=source,
        score=score,
        collapse_bpm=base_bpm,
        aba_support_delta=support,
    )


def _persistent(
    index: int,
    *,
    score: float,
    support: float | None = None,
) -> _CurveProposal:
    base_bpm = 180.0 + 0.01 * index
    tail_bpm = 150.0 + 0.01 * index
    curve = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=float(index),
        sections=(
            ConstantTempoSection(0, 40 + index, base_bpm),
            ConstantTempoSection(40 + index, 220 + index, tail_bpm),
        ),
    )
    return _CurveProposal(
        curve=curve,
        source="raw_run_persistent_a_to_b_start",
        score=score,
        collapse_bpm=base_bpm,
        aba_support_delta=support,
    )


def _long_aba(
    index: int,
    *,
    score: float,
    source: str = "raw_run_aba",
    support: float | None = None,
) -> _CurveProposal:
    base_bpm = 180.0 + 0.01 * index
    middle_bpm = 150.0 + 0.01 * index
    curve = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=float(index),
        sections=(
            ConstantTempoSection(0, 50 + index, base_bpm),
            ConstantTempoSection(50 + index, 90 + index, middle_bpm),
            ConstantTempoSection(90 + index, 220 + index, base_bpm),
        ),
    )
    return _CurveProposal(
        curve=curve,
        source=source,
        score=score,
        collapse_bpm=base_bpm,
        aba_support_delta=support,
    )


def _multi_step(
    index: int,
    *,
    score: float,
    support: float | None = None,
) -> _CurveProposal:
    base_bpm = 180.0 + 0.01 * index
    curve = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=float(index),
        sections=(
            ConstantTempoSection(0, 40 + index, base_bpm),
            ConstantTempoSection(40 + index, 70 + index, 150.0 + 0.01 * index),
            ConstantTempoSection(70 + index, 100 + index, 165.0 + 0.01 * index),
            ConstantTempoSection(100 + index, 220 + index, base_bpm),
        ),
    )
    return _CurveProposal(
        curve=curve,
        source="raw_run_chain_4_sections",
        score=score,
        collapse_bpm=base_bpm,
        aba_support_delta=support,
    )


def _overflow(
    index: int,
    *,
    score: float,
    support: float | None = None,
) -> _CurveProposal:
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
        aba_support_delta=support,
    )


def _ramp_like_diagnostic(index: int, *, score: float) -> _CurveProposal:
    curve = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=float(index),
        sections=(LinearTimeRampSection(0, 120 + index, 120.0, 180.0),),
    )
    return _CurveProposal(
        curve=curve,
        source="linear_tempo_raw_audio",
        score=score,
        collapse_bpm=150.0,
    )
