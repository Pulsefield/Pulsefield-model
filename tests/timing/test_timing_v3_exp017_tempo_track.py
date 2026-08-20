from __future__ import annotations

from collections import Counter

from pulsefield_model.timing.v3.analytic_curve import (
    ConstantTempoSection,
    PhaseContinuousTimingCurve,
)
from pulsefield_model.timing.v3.tempo_track import (
    _bounded_proposals,
    _CurveProposal,
    _deduped_jump_retention_order,
    _jump_retention_family,
    _retain_jump_proposals_by_family,
    TempoTrackConfig,
)


def test_exp017_family_classification_uses_frozen_taxonomy() -> None:
    config = TempoTrackConfig()

    assert (
        _jump_retention_family(_short_aba(0, score=1.0), config=config)
        == "short_aba_paired_boundary"
    )
    assert _jump_retention_family(_persistent(0, score=1.0), config=config) == "persistent"
    assert _jump_retention_family(_long_aba(0, score=1.0), config=config) == "long_aba"
    assert _jump_retention_family(_multi_step(0, score=1.0), config=config) == "multi_step"
    assert _jump_retention_family(_overflow(0, score=1.0), config=config) == "overflow"


def test_exp017_family_duration_bounds_ignore_widened_generation_config() -> None:
    widened = TempoTrackConfig(
        minimum_excursion_seconds=1.0,
        maximum_excursion_seconds=90.0,
    )
    below_short = _aba_with_duration(
        duration_seconds=1.999,
        source="paired_unmerged_boundary",
    )
    above_long = _aba_with_duration(
        duration_seconds=60.001,
        source="raw_run_aba",
    )

    assert _jump_retention_family(below_short, config=widened) == "overflow"
    assert _jump_retention_family(above_long, config=widened) == "overflow"


def test_exp017_short_aba_survives_lower_global_rank_under_family_pressure() -> None:
    pressure = tuple(_persistent(index, score=200.0 - index) for index in range(20))
    pressure += tuple(_long_aba(index, score=170.0 - index) for index in range(20))
    pressure += tuple(_multi_step(index, score=140.0 - index) for index in range(10))
    pressure += tuple(_overflow(index, score=120.0 - index) for index in range(10))
    target = _short_aba(99, score=-1.0)
    proposals = pressure + (target,)

    global_order = _deduped_jump_retention_order(proposals)
    global_rank = [
        proposal.curve.fingerprint_sha256 for proposal in global_order
    ].index(target.curve.fingerprint_sha256) + 1
    retained = _retain_jump_proposals_by_family(proposals, config=TempoTrackConfig())
    retained_fingerprints = {proposal.curve.fingerprint_sha256 for proposal in retained.proposals}

    assert global_rank > TempoTrackConfig().maximum_jump_candidates
    assert len(retained.proposals) == TempoTrackConfig().maximum_jump_candidates
    assert target.curve.fingerprint_sha256 in retained_fingerprints
    assert _family_counts(retained.proposals)["short_aba_paired_boundary"] >= 1


def test_exp017_family_quotas_backfill_and_duplicate_dedupe_are_deterministic() -> None:
    duplicate_source = _short_aba(777, score=300.0)
    duplicate_lower = _CurveProposal(
        curve=duplicate_source.curve,
        source="raw_run_aba",
        score=1.0,
        collapse_bpm=duplicate_source.collapse_bpm,
    )
    proposals = (
        tuple(_short_aba(index, score=100.0 - index) for index in range(20))
        + tuple(_persistent(index, score=90.0 - index) for index in range(3))
        + tuple(_long_aba(index, score=80.0 - index) for index in range(2))
        + tuple(_multi_step(index, score=70.0 - index) for index in range(1))
        + tuple(_overflow(index, score=60.0 - index) for index in range(30))
        + (duplicate_source, duplicate_lower)
    )

    config = TempoTrackConfig()
    first = _retain_jump_proposals_by_family(proposals, config=config)
    second = _retain_jump_proposals_by_family(proposals, config=config)
    first_families = tuple(
        _jump_retention_family(proposal, config=config) for proposal in first.proposals
    )

    assert tuple(proposal.curve.fingerprint_sha256 for proposal in first.proposals) == tuple(
        proposal.curve.fingerprint_sha256 for proposal in second.proposals
    )
    assert _family_counts(first.proposals) == _family_counts(second.proposals)
    assert first_families[:14] == ("short_aba_paired_boundary",) * 14
    assert first_families[14:17] == ("persistent",) * 3
    assert first_families[17:19] == ("long_aba",) * 2
    assert first_families[19:20] == ("multi_step",)
    assert first_families[20:26] == ("overflow",) * 6
    assert len(first.proposals) == 44
    assert (
        sum(
            proposal.curve.fingerprint_sha256
            == duplicate_source.curve.fingerprint_sha256
            for proposal in first.proposals
        )
        == 1
    )


def test_exp017_caps_constants_ramps_and_jump_pruning_diagnostics() -> None:
    config = TempoTrackConfig()
    constants = tuple(_constant(index, score=1000.0 - index) for index in range(12))
    jumps = _retain_jump_proposals_by_family(
        tuple(_persistent(index, score=500.0 - index) for index in range(60)),
        config=config,
    )
    ramps = tuple(_ramp_like_diagnostic(index, score=10.0 - index) for index in range(8))

    retained = _bounded_proposals(constants, jumps.proposals, ramps, config=config)

    assert config.maximum_candidates <= 64
    assert config.maximum_jump_candidates <= 44
    assert config.maximum_ramp_candidates <= 8
    assert jumps.pruning_reason == "maximum_jump_candidates_44"
    assert len(jumps.proposals) == 44
    assert len(retained) == 64
    assert all(
        proposal.curve.fingerprint_sha256
        in {candidate.curve.fingerprint_sha256 for candidate in retained}
        for proposal in constants
    )
    assert sum(candidate.curve.curve_class == "ramp" for candidate in retained) == 8


def _family_counts(proposals: tuple[_CurveProposal, ...]) -> Counter[str]:
    config = TempoTrackConfig()
    return Counter(_jump_retention_family(proposal, config=config) for proposal in proposals)


def _constant(index: int, *, score: float) -> _CurveProposal:
    bpm = 180.0 + 0.25 * index
    curve = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=float(index),
        sections=(ConstantTempoSection(0, 120 + index, bpm),),
    )
    return _CurveProposal(curve=curve, source="global_constant", score=score, collapse_bpm=bpm)


def _short_aba(index: int, *, score: float) -> _CurveProposal:
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
    )


def _persistent(index: int, *, score: float) -> _CurveProposal:
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
    )


def _long_aba(index: int, *, score: float) -> _CurveProposal:
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
    return _CurveProposal(curve=curve, source="raw_run_aba", score=score, collapse_bpm=base_bpm)


def _multi_step(index: int, *, score: float) -> _CurveProposal:
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
    return _CurveProposal(curve=curve, source="raw_run_aba", score=score, collapse_bpm=base_bpm)


def _aba_with_duration(*, duration_seconds: float, source: str) -> _CurveProposal:
    base_bpm = 180.0
    middle_beats = 30
    middle_bpm = 60.0 * middle_beats / duration_seconds
    curve = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(
            ConstantTempoSection(0, 40, base_bpm),
            ConstantTempoSection(40, 40 + middle_beats, middle_bpm),
            ConstantTempoSection(40 + middle_beats, 180, base_bpm),
        ),
    )
    return _CurveProposal(
        curve=curve,
        source=source,
        score=1.0,
        collapse_bpm=base_bpm,
    )


def _ramp_like_diagnostic(index: int, *, score: float) -> _CurveProposal:
    from pulsefield_model.timing.v3.analytic_curve import LinearTimeRampSection

    curve = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=float(index),
        sections=(LinearTimeRampSection(0, 120 + index, 120.0, 180.0),),
    )
    return _CurveProposal(curve=curve, source="linear_tempo_raw_audio", score=score, collapse_bpm=150.0)
