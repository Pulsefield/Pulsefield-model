from __future__ import annotations

import json
import math
import time
from dataclasses import FrozenInstanceError, asdict, replace

import numpy as np
import pytest

from pulsefield_model.timing.schema import FrameTimingPrediction
from pulsefield_model.timing.v3.global_constant_jump import (
    BOUNDARY_CANDIDATE_SCORE_VERSION,
    CANDIDATE_CONTRACT_VERSION,
    GLOBAL_CONSTANT_JUMP_CONSTANTS,
    GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON,
    GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON_SHA256,
    PULSE_CORRELATION_VERSION,
    REASON_EDGE_ATTEMPT_CAP_EXCEEDED,
    REASON_NO_GLOBAL_CONSTANT_JUMP_PATH,
    REASON_NO_ORIGIN_CANDIDATE,
    VARIANT_CJ0,
    VARIANT_CJ1,
    VARIANT_CJ2,
    VARIANT_CJ3,
    BoundaryCandidate,
    MaterializedPeak,
    boundary_candidate_score_v1,
    extract_global_constant_jump_candidates,
    fit_global_constant_jump,
    fit_global_constant_jump_variants,
    iter_global_constant_jump_variants,
    materialize_global_constant_jump_peaks,
    pulse_correlation_v1,
)
from pulsefield_model.timing.v3.schema import TimingV3Grid


def test_frozen_constants_are_serialized_and_realized_period_bounds_are_reported() -> None:
    payload = json.loads(GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON)

    assert payload["candidate_contract_version"] == CANDIDATE_CONTRACT_VERSION
    assert payload["hard_bpm_guard"] == [20.0, 1000.0]
    assert payload["beam_width"] == 64
    assert payload["default_edge_attempt_cap"] == 120000
    assert len(GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON_SHA256) == 64

    prediction = _constant_prediction(duration_ms=10_000.0, bpm=120.0)
    candidates = extract_global_constant_jump_candidates(prediction)

    assert candidates.diagnostics.constants_json_sha256 == GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON_SHA256
    assert candidates.diagnostics.min_period_frames == 3
    assert candidates.diagnostics.max_period_frames == 150
    json.dumps(asdict(candidates.diagnostics), allow_nan=False)


def test_peak_materialization_uses_threshold_asymmetric_peak_and_parabolic_refinement() -> None:
    signal = np.asarray([0.0, 0.4, 1.0, 0.2, 0.0], dtype=np.float64)

    peaks = materialize_global_constant_jump_peaks(signal, frame_rate_hz=50.0)

    assert len(peaks) == 1
    peak = peaks[0]
    assert peak.frame_index == 2
    assert peak.refined_frame == pytest.approx(1.9285714285714286)
    assert peak.time_ms == pytest.approx(38.57142857142857)
    assert peak.confidence == pytest.approx(1.0)


@pytest.mark.parametrize("peak_index", [0, 1, 2, 10])
def test_boundary_evidence_is_absent_when_ordinary_and_super_windows_are_missing(peak_index: int) -> None:
    peaks = tuple(
        MaterializedPeak(frame_index=index, refined_frame=float(index), time_ms=500.0 * index, confidence=1.0)
        for index in range(11)
    )

    candidate = boundary_candidate_score_v1(peaks, (), peak_index)

    assert candidate is None


def test_boundary_exact_ordinary_super_tie_selects_ordinary_and_carries_winner_periods(monkeypatch: pytest.MonkeyPatch) -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    peaks = tuple(
        MaterializedPeak(frame_index=index, refined_frame=float(index), time_ms=500.0 * index, confidence=0.8)
        for index in range(9)
    )

    monkeypatch.setattr(module, "_ordinary_boundary_evidence", lambda peak_times, peak_index: (3.0, 500.0, 600.0))
    monkeypatch.setattr(module, "_super_boundary_evidence", lambda peak_times, peak_confidences, peak_index: (3.0, 450.0, 650.0))

    candidate = boundary_candidate_score_v1(peaks, (), 4)

    assert isinstance(candidate, BoundaryCandidate)
    assert candidate.evidence_mode == "ordinary"
    assert candidate.ordinary_score == pytest.approx(3.0)
    assert candidate.super_score == pytest.approx(3.0)
    assert candidate.left_period_ms == pytest.approx(500.0)
    assert candidate.right_period_ms == pytest.approx(600.0)


def test_empty_beat_signal_falls_back_before_reading_any_oracle_source() -> None:
    prediction = _metadata_trapping_prediction(frame_count=500, frame_rate_hz=50.0)
    prediction.arm_metadata_traps()

    result = fit_global_constant_jump(prediction, variant=VARIANT_CJ3)

    assert not result.ok
    assert result.reason == REASON_NO_ORIGIN_CANDIDATE
    assert result.diagnostics.fallback_reason == REASON_NO_ORIGIN_CANDIDATE
    json.dumps(asdict(result.diagnostics), allow_nan=False)


def test_cj0_constant_path_uses_frozen_tempo_not_terminal_end_as_a_beat() -> None:
    prediction = _constant_prediction(duration_ms=20_120.0, bpm=120.0)
    candidates = extract_global_constant_jump_candidates(prediction)

    result = fit_global_constant_jump(prediction, variant=VARIANT_CJ0)

    assert result.ok
    assert result.reason is None
    assert isinstance(result.grid, TimingV3Grid)
    assert len(result.grid.sections) == 1
    section = result.grid.sections[0]
    assert any(section.bpm == pytest.approx(candidate.bpm) for candidate in candidates.tempo_candidates)
    assert section.bpm == pytest.approx(120.0, abs=0.5)
    assert result.grid.coverage_end_ms == pytest.approx(20_120.0)
    assert result.grid.end_time_ms >= result.grid.coverage_end_ms
    assert result.grid.end_time_ms != pytest.approx(result.grid.coverage_end_ms)
    assert result.diagnostics.selected_section_count == 1

    restored = TimingV3Grid.from_dict(json.loads(json.dumps(result.grid.to_dict())))
    assert restored.boundary_times_ms == pytest.approx(result.grid.boundary_times_ms)
    json.dumps(asdict(result.diagnostics), allow_nan=False)


def test_cj1_finds_a_synthetic_on_lattice_jump_with_beat_support_only() -> None:
    prediction = _jump_prediction(first_bpm=120.0, second_bpm=150.0, boundary_ms=12_000.0, duration_ms=24_000.0)
    candidates = extract_global_constant_jump_candidates(prediction)

    result = fit_global_constant_jump(prediction, variant=VARIANT_CJ1)

    assert result.ok
    assert result.grid is not None
    assert len(result.grid.sections) >= 2
    assert candidates.boundary_candidates[0].time_ms == pytest.approx(11_500.0)
    boundary_times = result.grid.boundary_times_ms[1:-1]
    assert any(abs(time_ms - 11_500.0) <= 60.0 for time_ms in boundary_times)
    assert result.diagnostics.section_attempt_count <= GLOBAL_CONSTANT_JUMP_CONSTANTS.default_edge_attempt_cap
    json.dumps(asdict(result.diagnostics), allow_nan=False)


def test_downbeat_free_cj3_reduces_to_cj2_for_the_same_cache() -> None:
    prediction = _jump_prediction(
        first_bpm=120.0,
        second_bpm=150.0,
        boundary_ms=12_000.0,
        duration_ms=24_000.0,
        include_downbeats=False,
    )

    cj2 = fit_global_constant_jump(prediction, variant=VARIANT_CJ2)
    cj3 = fit_global_constant_jump(prediction, variant=VARIANT_CJ3)

    assert cj2.ok
    assert cj3.ok
    assert cj2.grid is not None
    assert cj3.grid is not None
    assert cj3.diagnostics.selected_downbeat_phase is None
    assert cj3.grid.to_dict() == cj2.grid.to_dict()
    assert cj3.diagnostics.objective == pytest.approx(cj2.diagnostics.objective)


def test_downbeat_template_propagates_global_modulo_four_by_absolute_beat_count() -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    frame_rate_hz = 50.0
    frame_count = int(16_000.0 * frame_rate_hz / 1000.0)
    frame_times_ms = np.arange(frame_count, dtype=np.float64) * (1000.0 / frame_rate_hz)
    downbeat_prob = np.zeros(frame_count, dtype=np.float64)
    boundary_ms = 9_000.0
    second_period_ms = 400.0
    boundary_beat = 18
    for beat in range(20, 36, 4):
        _write_pulse(
            downbeat_prob,
            time_ms=boundary_ms + (beat - boundary_beat) * second_period_ms,
            frame_rate_hz=frame_rate_hz,
        )

    global_phase = module._downbeat_pulse_correlation_interval(
        downbeat_prob,
        frame_times_ms,
        tau_ms=boundary_ms,
        beat_at_tau=boundary_beat,
        bpm=150.0,
        global_downbeat_phase=0,
        start_ms=boundary_ms,
        end_ms=16_000.0,
        pulse_width_ms=GLOBAL_CONSTANT_JUMP_CONSTANTS.pulse_width_ms,
    )
    local_reset_phase = module._downbeat_pulse_correlation_interval(
        downbeat_prob,
        frame_times_ms,
        tau_ms=boundary_ms,
        beat_at_tau=0,
        bpm=150.0,
        global_downbeat_phase=0,
        start_ms=boundary_ms,
        end_ms=16_000.0,
        pulse_width_ms=GLOBAL_CONSTANT_JUMP_CONSTANTS.pulse_width_ms,
    )

    assert global_phase.valid
    assert local_reset_phase.valid
    assert global_phase.correlation > local_reset_phase.correlation


def test_downbeat_correlation_is_exactly_canonical_when_beat_and_phase_shift_by_four() -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    frame_times_ms = 123_456_789_000.0 + np.arange(2048, dtype=np.float64) * 20.0
    signal = (
        0.4
        + 0.3 * np.sin(np.arange(frame_times_ms.shape[0], dtype=np.float64) * 0.073)
        + 0.2 * np.cos(np.arange(frame_times_ms.shape[0], dtype=np.float64) * 0.197)
    )

    base = module._downbeat_pulse_correlation_interval(
        signal,
        frame_times_ms,
        tau_ms=123_456_000_000.0,
        beat_at_tau=3,
        bpm=137.5,
        global_downbeat_phase=2,
        start_ms=float(frame_times_ms[10]),
        end_ms=float(frame_times_ms[-10]),
        pulse_width_ms=GLOBAL_CONSTANT_JUMP_CONSTANTS.pulse_width_ms,
    )
    shifted = module._downbeat_pulse_correlation_interval(
        signal,
        frame_times_ms,
        tau_ms=123_456_000_000.0,
        beat_at_tau=7,
        bpm=137.5,
        global_downbeat_phase=2,
        start_ms=float(frame_times_ms[10]),
        end_ms=float(frame_times_ms[-10]),
        pulse_width_ms=GLOBAL_CONSTANT_JUMP_CONSTANTS.pulse_width_ms,
    )

    assert base.valid
    assert shifted == base


def test_attempt_cap_exceeded_is_a_tagged_fallback() -> None:
    prediction = _constant_prediction(duration_ms=20_000.0, bpm=120.0)

    result = fit_global_constant_jump(prediction, variant=VARIANT_CJ0, attempt_cap=1)

    assert not result.ok
    assert result.reason == REASON_EDGE_ATTEMPT_CAP_EXCEEDED
    assert result.diagnostics.section_attempt_count == 1
    json.dumps(asdict(result.diagnostics), allow_nan=False)

    with pytest.raises(ValueError, match="attempt_cap"):
        fit_global_constant_jump(prediction, variant=VARIANT_CJ0, attempt_cap=1.5)  # type: ignore[arg-type]


def test_result_and_diagnostics_are_immutable() -> None:
    result = fit_global_constant_jump(_constant_prediction(duration_ms=10_000.0, bpm=120.0), variant=VARIANT_CJ0)

    with pytest.raises(FrozenInstanceError):
        result.reason = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.diagnostics.variant = VARIANT_CJ1  # type: ignore[misc]


def test_public_variants_api_reuses_one_candidate_set_and_preserves_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    prediction = _metadata_trapping_constant_prediction(duration_ms=10_000.0, bpm=120.0)
    original_extract = module.extract_global_constant_jump_candidates
    calls = 0

    def recording_extract(seen_prediction: FrameTimingPrediction):
        nonlocal calls
        calls += 1
        return original_extract(seen_prediction)

    monkeypatch.setattr(module, "extract_global_constant_jump_candidates", recording_extract)
    prediction.arm_metadata_traps()

    results = module.fit_global_constant_jump_variants(prediction)

    assert calls == 1
    assert tuple(results) == (VARIANT_CJ0, VARIANT_CJ1, VARIANT_CJ2, VARIANT_CJ3)
    fingerprints = {result.diagnostics.candidate_fingerprint for result in results.values()}
    assert len(fingerprints) == 1

    candidate_set = original_extract(_constant_prediction(duration_ms=10_000.0, bpm=120.0))
    reused = fit_global_constant_jump(prediction, variant=VARIANT_CJ0, candidate_set=candidate_set)
    assert reused.diagnostics.candidate_fingerprint == candidate_set.diagnostics.candidate_fingerprint

    bad_diagnostics = replace(candidate_set.diagnostics, frame_count=candidate_set.diagnostics.frame_count + 1)
    bad_candidate_set = replace(candidate_set, diagnostics=bad_diagnostics)
    with pytest.raises(ValueError, match="frame_count"):
        fit_global_constant_jump(prediction, variant=VARIANT_CJ0, candidate_set=bad_candidate_set)

    different_signal = _constant_prediction(duration_ms=10_000.0, bpm=121.0)
    with pytest.raises(ValueError, match="input_signal_sha256"):
        fit_global_constant_jump(different_signal, variant=VARIANT_CJ0, candidate_set=candidate_set)


def test_extracted_candidate_diagnostics_are_the_candidate_reuse_trust_boundary() -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    prediction = _metadata_trapping_constant_prediction(duration_ms=10_000.0, bpm=120.0)
    prediction.arm_metadata_traps()

    candidates = extract_global_constant_jump_candidates(prediction)
    expected_geometry = module._prediction_geometry(
        np.asarray(prediction.beat_prob, dtype=np.float64),
        prediction.frame_rate_hz,
        GLOBAL_CONSTANT_JUMP_CONSTANTS,
    )
    result = fit_global_constant_jump(prediction, variant=VARIANT_CJ0, candidate_set=candidates)

    assert candidates.diagnostics.pulse_correlation_version == PULSE_CORRELATION_VERSION
    assert candidates.diagnostics.boundary_candidate_score_version == BOUNDARY_CANDIDATE_SCORE_VERSION
    assert candidates.diagnostics.min_period_frames == expected_geometry[4]
    assert candidates.diagnostics.max_period_frames == expected_geometry[5]
    assert result.diagnostics.input_signal_sha256 == candidates.diagnostics.input_signal_sha256
    assert result.diagnostics.candidate_fingerprint == candidates.diagnostics.candidate_fingerprint


def test_extracted_boundary_candidates_respect_reuse_authenticity_constraints() -> None:
    prediction = _jump_prediction(first_bpm=120.0, second_bpm=150.0, boundary_ms=12_000.0, duration_ms=24_000.0)
    candidates = extract_global_constant_jump_candidates(prediction)
    assert candidates.boundary_candidates

    duration_s = (candidates.diagnostics.coverage_end_ms - candidates.diagnostics.coverage_start_ms) / 1000.0
    boundary_cap = min(
        GLOBAL_CONSTANT_JUMP_CONSTANTS.max_interior_boundary_candidates,
        max(16, int(math.ceil(duration_s / 4.0))),
    )
    assert len(candidates.boundary_candidates) <= boundary_cap

    previous_time_ms = -math.inf
    for boundary in candidates.boundary_candidates:
        assert candidates.diagnostics.coverage_start_ms <= boundary.time_ms < candidates.diagnostics.coverage_end_ms
        assert boundary.time_ms - previous_time_ms > GLOBAL_CONSTANT_JUMP_CONSTANTS.boundary_merge_ms
        previous_time_ms = boundary.time_ms


@pytest.mark.parametrize(
    ("field", "replacement_value", "message"),
    [
        ("min_period_frames", None, "min_period_frames"),
        ("max_period_frames", None, "max_period_frames"),
        ("pulse_correlation_version", f"{PULSE_CORRELATION_VERSION}-old", "pulse_correlation_version"),
        (
            "boundary_candidate_score_version",
            f"{BOUNDARY_CANDIDATE_SCORE_VERSION}-old",
            "boundary_candidate_score_version",
        ),
    ],
)
def test_candidate_set_reuse_rejects_stale_candidate_diagnostic_provenance(
    field: str,
    replacement_value: object,
    message: str,
) -> None:
    prediction = _constant_prediction(duration_ms=10_000.0, bpm=120.0)
    candidate_set = extract_global_constant_jump_candidates(prediction)
    if field == "min_period_frames":
        replacement_value = candidate_set.diagnostics.min_period_frames + 1
    elif field == "max_period_frames":
        replacement_value = candidate_set.diagnostics.max_period_frames - 1
    tampered = replace(
        candidate_set,
        diagnostics=replace(candidate_set.diagnostics, **{field: replacement_value}),
    )

    with pytest.raises(ValueError, match=message):
        fit_global_constant_jump(prediction, variant=VARIANT_CJ0, candidate_set=tampered)


def test_candidate_set_reuse_rejects_duration_dependent_boundary_candidate_cap() -> None:
    prediction = _constant_prediction(duration_ms=40_000.0, bpm=120.0)
    beat_peaks = materialize_global_constant_jump_peaks(prediction.beat_prob, frame_rate_hz=prediction.frame_rate_hz)
    boundary_candidates = tuple(
        _boundary_candidate_from_peak(beat_peaks, peak_index=4 + index, anchor_id=index)
        for index in range(17)
    )
    candidate_set = _candidate_set_for_prediction(
        prediction,
        beat_peaks=beat_peaks,
        downbeat_peaks=(),
        tempo_candidates=(),
        origin_candidates=(),
        boundary_candidates=boundary_candidates,
    )

    with pytest.raises(ValueError, match="boundary candidate cap"):
        fit_global_constant_jump(prediction, variant=VARIANT_CJ0, candidate_set=candidate_set)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda boundaries, diagnostics: (
                replace(boundaries[0], time_ms=diagnostics.coverage_end_ms),
            )
            + boundaries[1:],
            "coverage",
        ),
        (
            lambda boundaries, diagnostics: (
                boundaries[0],
                replace(boundaries[1], time_ms=boundaries[0].time_ms + 1.0),
            )
            + boundaries[2:],
            "merge spacing",
        ),
    ],
)
def test_candidate_set_reuse_rejects_boundary_authenticity_tampering(mutator, message: str) -> None:
    prediction = _constant_prediction(duration_ms=40_000.0, bpm=120.0)
    beat_peaks = materialize_global_constant_jump_peaks(prediction.beat_prob, frame_rate_hz=prediction.frame_rate_hz)
    boundary_candidates = tuple(
        _boundary_candidate_from_peak(beat_peaks, peak_index=4 + index * 17, anchor_id=index)
        for index in range(3)
    )
    tampered_boundaries = mutator(boundary_candidates, extract_global_constant_jump_candidates(prediction).diagnostics)
    candidate_set = _candidate_set_for_prediction(
        prediction,
        beat_peaks=beat_peaks,
        downbeat_peaks=(),
        tempo_candidates=(),
        origin_candidates=(),
        boundary_candidates=tampered_boundaries,
    )

    with pytest.raises(ValueError, match=message):
        fit_global_constant_jump(prediction, variant=VARIANT_CJ0, candidate_set=candidate_set)


@pytest.mark.parametrize("variant", [VARIANT_CJ1, VARIANT_CJ2])
def test_interior_section_score_cache_for_cj1_cj2_uses_geometry_key(variant: str) -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    context, candidates = _constant_scoring_context(duration_ms=10_000.0, bpm=120.0)
    left_anchor = module._origin_anchors(candidates)[0]
    right_anchor = module._Anchor(anchor_id=1, kind="boundary", time_ms=9_000.0, rank_score=1.0)

    score = module._section_score(
        left_anchor,
        right_anchor=right_anchor,
        left_beat_at_anchor=0,
        beat_count=18,
        bpm=120.0,
        downbeat_phase=None,
        context=context,
        variant=variant,
    )
    attempts_after_first = context.attempt_count
    equivalent = module._section_score(
        left_anchor,
        right_anchor=right_anchor,
        left_beat_at_anchor=37,
        beat_count=18,
        bpm=120.0,
        downbeat_phase=None,
        context=context,
        variant=variant,
    )

    assert score.valid
    assert equivalent == score
    assert context.attempt_count == attempts_after_first
    assert len(context.section_score_cache) == 1


@pytest.mark.parametrize("variant", [VARIANT_CJ0, VARIANT_CJ3])
def test_downbeat_section_score_cache_uses_effective_residue_classes(variant: str) -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    context, candidates = _constant_scoring_context(duration_ms=10_000.0, bpm=120.0)
    left_anchor = module._origin_anchors(candidates)[0]
    right_anchor = module._Anchor(anchor_id=1, kind="boundary", time_ms=9_000.0, rank_score=1.0)

    score = module._section_score(
        left_anchor,
        right_anchor=right_anchor,
        left_beat_at_anchor=0,
        beat_count=18,
        bpm=120.0,
        downbeat_phase=0,
        context=context,
        variant=variant,
    )
    attempts_after_first = context.attempt_count
    equivalent = module._section_score(
        left_anchor,
        right_anchor=right_anchor,
        left_beat_at_anchor=4,
        beat_count=18,
        bpm=120.0,
        downbeat_phase=0,
        context=context,
        variant=variant,
    )
    attempts_after_equivalent = context.attempt_count
    different_residue = module._section_score(
        left_anchor,
        right_anchor=right_anchor,
        left_beat_at_anchor=1,
        beat_count=18,
        bpm=120.0,
        downbeat_phase=0,
        context=context,
        variant=variant,
    )

    assert score.valid
    assert equivalent == score
    assert different_residue.valid
    assert context.attempt_count == attempts_after_first + 1
    assert attempts_after_equivalent == attempts_after_first
    assert len(context.section_score_cache) == 2


def test_section_score_cache_key_matches_frozen_protocol_shape() -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    left_anchor = module._Anchor(anchor_id=7, kind="origin", time_ms=0.0, rank_score=1.0)
    right_anchor = module._Anchor(anchor_id=11, kind="boundary", time_ms=9_000.0, rank_score=1.0)

    assert module._section_score_cache_key(
        left_anchor,
        right_anchor=None,
        left_beat_at_anchor=3,
        beat_count=20,
        bpm=120.0000004,
        downbeat_phase=None,
        variant=VARIANT_CJ1,
    ) == (7, -1, 20, VARIANT_CJ1, 120.0)
    assert module._section_score_cache_key(
        left_anchor,
        right_anchor=None,
        left_beat_at_anchor=3,
        beat_count=20,
        bpm=120.0000004,
        downbeat_phase=2,
        variant=VARIANT_CJ3,
    ) == (7, -1, 20, VARIANT_CJ3, 120.0, 1)
    assert module._section_score_cache_key(
        left_anchor,
        right_anchor=right_anchor,
        left_beat_at_anchor=37,
        beat_count=18,
        bpm=120.0,
        downbeat_phase=None,
        variant=VARIANT_CJ2,
    ) == (7, 11, 18, VARIANT_CJ2)
    assert module._section_score_cache_key(
        left_anchor,
        right_anchor=right_anchor,
        left_beat_at_anchor=37,
        beat_count=18,
        bpm=120.0,
        downbeat_phase=2,
        variant=VARIANT_CJ3,
    ) == (7, 11, 18, VARIANT_CJ3, 3)


def test_terminal_downbeat_cache_separates_residue_none_and_attempt_cap_boundary() -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    context, candidates = _constant_scoring_context(duration_ms=12_000.0, bpm=120.0)
    left_anchor = module._origin_anchors(candidates)[0]

    score = module._section_score(
        left_anchor,
        right_anchor=None,
        left_beat_at_anchor=0,
        beat_count=24,
        bpm=120.0,
        downbeat_phase=0,
        context=context,
        variant=VARIANT_CJ3,
    )
    attempts_after_first = context.attempt_count
    equivalent = module._section_score(
        left_anchor,
        right_anchor=None,
        left_beat_at_anchor=4,
        beat_count=24,
        bpm=120.0,
        downbeat_phase=0,
        context=context,
        variant=VARIANT_CJ3,
    )
    different_residue = module._section_score(
        left_anchor,
        right_anchor=None,
        left_beat_at_anchor=1,
        beat_count=24,
        bpm=120.0,
        downbeat_phase=0,
        context=context,
        variant=VARIANT_CJ3,
    )
    no_phase = module._section_score(
        left_anchor,
        right_anchor=None,
        left_beat_at_anchor=0,
        beat_count=24,
        bpm=120.0,
        downbeat_phase=None,
        context=context,
        variant=VARIANT_CJ3,
    )

    assert score.valid
    assert equivalent == score
    assert different_residue.valid
    assert no_phase.valid
    assert attempts_after_first == 1
    assert context.attempt_count == 3
    assert len(context.terminal_score_cache) == 3

    cap_context, cap_candidates = _constant_scoring_context(duration_ms=12_000.0, bpm=120.0)
    object.__setattr__(cap_context, "attempt_count", GLOBAL_CONSTANT_JUMP_CONSTANTS.default_edge_attempt_cap - 1)
    cap_left_anchor = module._origin_anchors(cap_candidates)[0]
    right_anchor = module._Anchor(anchor_id=1, kind="boundary", time_ms=9_000.0, rank_score=1.0)

    boundary_score = module._section_score(
        cap_left_anchor,
        right_anchor=right_anchor,
        left_beat_at_anchor=0,
        beat_count=18,
        bpm=120.0,
        downbeat_phase=None,
        context=cap_context,
        variant=VARIANT_CJ1,
    )
    assert boundary_score.valid
    assert cap_context.attempt_count == GLOBAL_CONSTANT_JUMP_CONSTANTS.default_edge_attempt_cap
    with pytest.raises(module._AttemptCapExceeded):
        module._section_score(
            cap_left_anchor,
            right_anchor=right_anchor,
            left_beat_at_anchor=0,
            beat_count=19,
            bpm=126.6666666667,
            downbeat_phase=None,
            context=cap_context,
            variant=VARIANT_CJ1,
        )


def test_terminal_section_score_cache_key_includes_bpm_for_equal_beat_counts() -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    prediction = _constant_prediction(duration_ms=10_000.0, bpm=120.0, downbeat_phase=None)
    candidates = _single_origin_candidate_set(prediction, origin_time_ms=0.0, bpm=120.0)
    context = module._SearchContext(
        prediction=prediction,
        candidates=candidates,
        constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
        attempt_cap=GLOBAL_CONSTANT_JUMP_CONSTANTS.default_edge_attempt_cap,
    )
    anchor = module._origin_anchors(candidates)[0]

    score_120 = module._section_score(
        anchor,
        right_anchor=None,
        left_beat_at_anchor=0,
        beat_count=20,
        bpm=120.0,
        downbeat_phase=None,
        context=context,
        variant=VARIANT_CJ1,
    )
    score_150 = module._section_score(
        anchor,
        right_anchor=None,
        left_beat_at_anchor=0,
        beat_count=20,
        bpm=150.0,
        downbeat_phase=None,
        context=context,
        variant=VARIANT_CJ1,
    )

    assert len(context.terminal_score_cache) == 2
    assert score_120.cost != pytest.approx(score_150.cost)


@pytest.mark.parametrize(
    ("bpm", "tempo_bpms"),
    [
        (120.0, (80.0, 100.0, 120.0, 180.0)),
        (121.25, (240.0, 60.0, 119.5, 119.5, 123.0, 500.0)),
        (20.0, (20.0, 1000.0, 240.0)),
        (999.0, (20.0, 333.0, 1000.0)),
        (137.777777, (80.0, 90.0, 160.0, 180.0, 240.0)),
    ],
)
def test_ordered_nearest_tempo_distance_matches_bruteforce(bpm: float, tempo_bpms: tuple[float, ...]) -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    ordered = tuple(sorted(tempo for tempo in tempo_bpms if tempo > 0.0))
    expected = min(abs(math.log2(bpm / tempo)) for tempo in tempo_bpms if tempo > 0.0)

    assert module._nearest_tempo_distance_from_ordered(bpm, ordered) == expected
    assert module._nearest_tempo_distance(bpm, tempo_bpms) == expected


def test_ordered_nearest_tempo_distance_matches_legacy_neighbor_scan_10k() -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    tempo_values = tuple(float(20 + ((index * 37) % 980)) for index in range(257))
    ordered = tuple(sorted(tempo for tempo in tempo_values if math.isfinite(tempo) and tempo > 0.0))
    for index in range(10_000):
        bpm = 20.0 + ((index * 1_234_567) % 980_000) / 1000.0
        insertion_index = module.bisect_left(ordered, bpm)
        nearest_candidates: list[float] = []
        if insertion_index < len(ordered):
            nearest_candidates.append(ordered[insertion_index])
        if insertion_index > 0:
            nearest_candidates.append(ordered[insertion_index - 1])
        expected = float(min(abs(math.log2(bpm / tempo_bpm)) for tempo_bpm in nearest_candidates))

        assert module._nearest_tempo_distance_from_ordered(bpm, ordered) == expected


def test_beat_count_candidates_collect_unique_counts_before_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    prediction = _constant_prediction(duration_ms=20_000.0, bpm=120.0)
    beat_peaks = materialize_global_constant_jump_peaks(prediction.beat_prob, frame_rate_hz=prediction.frame_rate_hz)
    candidate_set = _candidate_set_for_prediction(
        prediction,
        beat_peaks=beat_peaks,
        downbeat_peaks=(),
        tempo_candidates=(
            module.TempoCandidate(bpm=119.9, source="test", score=1.0),
            module.TempoCandidate(bpm=120.0, source="test", score=1.0),
            module.TempoCandidate(bpm=120.1, source="test", score=1.0),
        ),
        origin_candidates=(module.OriginCandidate(anchor_id=0, time_ms=0.0, bpm=120.0, score=1.0),),
        boundary_candidates=(),
    )
    context = module._SearchContext(
        prediction=prediction,
        candidates=candidate_set,
        constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
        attempt_cap=GLOBAL_CONSTANT_JUMP_CONSTANTS.default_edge_attempt_cap,
    )
    left_anchor = module._origin_anchors(candidate_set)[0]
    right_anchor = module._Anchor(anchor_id=1, kind="boundary", time_ms=10_000.0, rank_score=1.0)
    support_calls: list[float] = []

    def fake_support(**kwargs):
        support_calls.append(kwargs["bpm"])
        return 0.25

    monkeypatch.setattr(module, "_beat_peak_support_for_lattice_count_ranking", fake_support)

    candidates = module._beat_count_candidates(left_anchor, right_anchor, context)
    replay = module._beat_count_candidates(left_anchor, right_anchor, context)

    assert {candidate.count for candidate in candidates} == {19, 20, 21}
    assert replay == candidates
    assert len(support_calls) == 3
    assert len(context.count_cache) == 1
    assert context.edge_count_cache_size == 1


def test_lazy_beat_count_support_ranking_matches_legacy_exhaustive(monkeypatch: pytest.MonkeyPatch) -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    context, _ = _constant_scoring_context(duration_ms=20_000.0, bpm=120.0)
    left_anchor = module._Anchor(anchor_id=0, kind="origin", time_ms=0.0, rank_score=1.0)
    right_anchor = module._Anchor(anchor_id=1, kind="boundary", time_ms=10_000.0, rank_score=1.0)
    delta_ms = right_anchor.time_ms - left_anchor.time_ms

    for seed in range(25):
        counts = sorted({1 + ((seed * 17 + index * 19) % 96) for index in range(48)})
        distances = {count: float(((count * (seed + 3)) % 9) / 10.0) for count in counts}
        supports = {count: float(((count * 37 + seed * 11) % 101) / 100.0) for count in counts}
        support_calls: list[int] = []

        def fake_support(**kwargs):
            count = int(round(kwargs["bpm"] * delta_ms / 60000.0))
            support_calls.append(count)
            return supports[count]

        monkeypatch.setattr(module, "_beat_peak_support_for_lattice_count_ranking", fake_support)
        distance_ranked = sorted(
            (distances[count], count, 60000.0 * count / delta_ms)
            for count in counts
        )
        expected = sorted(
            ((count, distances[count], supports[count]) for count in counts),
            key=lambda item: (item[1], -item[2], item[0], 60000.0 * item[0] / delta_ms),
        )[: GLOBAL_CONSTANT_JUMP_CONSTANTS.max_beat_count_candidates_per_edge]

        actual = module._rank_beat_count_candidates_with_lazy_support(
            distance_ranked,
            left_anchor=left_anchor,
            right_anchor=right_anchor,
            context=context,
        )

        assert actual == expected
        needed_distances: set[float] = set()
        covered = 0
        for distance in sorted({distances[count] for count in counts}):
            needed_distances.add(distance)
            covered += sum(1 for count in counts if distances[count] == distance)
            if covered >= GLOBAL_CONSTANT_JUMP_CONSTANTS.max_beat_count_candidates_per_edge:
                break
        assert set(support_calls) == {count for count in counts if distances[count] in needed_distances}


def test_lazy_beat_count_support_evaluates_full_equal_distance_cutoff_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    context, _ = _constant_scoring_context(duration_ms=20_000.0, bpm=120.0)
    left_anchor = module._Anchor(anchor_id=0, kind="origin", time_ms=0.0, rank_score=1.0)
    right_anchor = module._Anchor(anchor_id=1, kind="boundary", time_ms=10_000.0, rank_score=1.0)
    delta_ms = right_anchor.time_ms - left_anchor.time_ms
    counts = tuple(range(1, 26))
    distances = {
        count: 0.1 if count <= 10 else 0.2 if count <= 20 else 0.3
        for count in counts
    }
    supports = {count: float((100 - count) / 100.0) for count in counts}
    support_calls: list[int] = []

    def fake_support(**kwargs):
        count = int(round(kwargs["bpm"] * delta_ms / 60000.0))
        support_calls.append(count)
        return supports[count]

    monkeypatch.setattr(module, "_beat_peak_support_for_lattice_count_ranking", fake_support)
    distance_ranked = sorted(
        (distances[count], count, 60000.0 * count / delta_ms)
        for count in counts
    )

    actual = module._rank_beat_count_candidates_with_lazy_support(
        distance_ranked,
        left_anchor=left_anchor,
        right_anchor=right_anchor,
        context=context,
    )
    expected = sorted(
        ((count, distances[count], supports[count]) for count in counts),
        key=lambda item: (item[1], -item[2], item[0], 60000.0 * item[0] / delta_ms),
    )[: GLOBAL_CONSTANT_JUMP_CONSTANTS.max_beat_count_candidates_per_edge]

    assert actual == expected
    assert support_calls == list(range(1, 21))


def test_beat_count_candidates_with_empty_tempo_pool_returns_empty_without_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    prediction = _constant_prediction(duration_ms=20_000.0, bpm=120.0, downbeat_phase=None)
    candidate_set = _candidate_set_for_prediction(
        prediction,
        beat_peaks=(),
        downbeat_peaks=(),
        tempo_candidates=(),
        origin_candidates=(module.OriginCandidate(anchor_id=0, time_ms=0.0, bpm=120.0, score=1.0),),
        boundary_candidates=(),
    )
    context = module._SearchContext(
        prediction=prediction,
        candidates=candidate_set,
        constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
        attempt_cap=GLOBAL_CONSTANT_JUMP_CONSTANTS.default_edge_attempt_cap,
    )
    left_anchor = module._origin_anchors(candidate_set)[0]
    right_anchor = module._Anchor(anchor_id=1, kind="boundary", time_ms=10_000.0, rank_score=1.0)

    def forbidden_support(**kwargs):
        raise AssertionError("support should not be evaluated without tempo-derived counts")

    monkeypatch.setattr(module, "_beat_peak_support_for_lattice_count_ranking", forbidden_support)

    assert module._beat_count_candidates(left_anchor, right_anchor, context) == ()
    assert module._beat_count_candidates(left_anchor, right_anchor, context) == ()
    assert context.edge_count_cache_size == 1


def test_lazy_beat_count_support_preserves_signed_zero_and_exact_tie_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    context, _ = _constant_scoring_context(duration_ms=20_000.0, bpm=120.0)
    left_anchor = module._Anchor(anchor_id=0, kind="origin", time_ms=0.0, rank_score=1.0)
    right_anchor = module._Anchor(anchor_id=1, kind="boundary", time_ms=10_000.0, rank_score=1.0)
    delta_ms = right_anchor.time_ms - left_anchor.time_ms
    counts = tuple(range(1, 22))
    distances = {count: (-0.0 if count % 2 else 0.0) for count in counts[:20]}
    distances[21] = 0.25
    support_calls: list[int] = []

    def tied_support(**kwargs):
        count = int(round(kwargs["bpm"] * delta_ms / 60000.0))
        support_calls.append(count)
        return 0.5

    monkeypatch.setattr(module, "_beat_peak_support_for_lattice_count_ranking", tied_support)
    distance_ranked = sorted(
        (distances[count], count, 60000.0 * count / delta_ms)
        for count in counts
    )

    actual = module._rank_beat_count_candidates_with_lazy_support(
        distance_ranked,
        left_anchor=left_anchor,
        right_anchor=right_anchor,
        context=context,
    )
    expected = sorted(
        ((count, distances[count], 0.5) for count in counts),
        key=lambda item: (item[1], -item[2], item[0], 60000.0 * item[0] / delta_ms),
    )[: GLOBAL_CONSTANT_JUMP_CONSTANTS.max_beat_count_candidates_per_edge]

    assert actual == expected
    assert support_calls == list(range(1, 21))
    assert math.copysign(1.0, actual[0][1]) == -1.0


def test_peak_recall_precision_fast_path_matches_reference_random() -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    rng = np.random.default_rng(904)
    for _ in range(2_000):
        tau_ms = float(rng.uniform(-50_000.0, 50_000.0))
        bpm = float(rng.uniform(20.0, 1000.0))
        period_ms = 60000.0 / bpm
        start_ms = tau_ms + float(rng.uniform(-20.0, 20.0)) * period_ms
        end_ms = start_ms + float(rng.uniform(0.0, 200.0)) * period_ms
        peak_count = int(rng.integers(0, 50))
        peaks = np.sort(
            rng.uniform(start_ms - 2.0 * period_ms, end_ms + 2.0 * period_ms, size=peak_count).astype(
                np.float64
            )
        )

        expected = module._peak_recall_precision_cost(
            left_anchor_time_ms=tau_ms,
            bpm=bpm,
            start_ms=start_ms,
            end_ms=end_ms,
            beat_peak_times_ms=peaks,
            constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
        )
        actual = module._peak_recall_precision_cost_fast(
            left_anchor_time_ms=tau_ms,
            bpm=bpm,
            start_ms=start_ms,
            end_ms=end_ms,
            beat_peak_times_ms=peaks,
            constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
        )

        assert actual == expected


def test_peak_recall_precision_fast_path_matches_reference_edges_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    bpm = 120.0
    period_ms = 60000.0 / bpm
    widened = replace(
        GLOBAL_CONSTANT_JUMP_CONSTANTS,
        peak_grid_tolerance_ms=1000.0,
        peak_grid_tolerance_beat_fraction=0.6,
    )
    edge_cases = (
        (0.0, 0.0, np.nextafter(4.0 * period_ms, math.inf), np.asarray([0.0, 0.5 * period_ms])),
        (
            np.nextafter(0.0, -math.inf),
            np.nextafter(period_ms, math.inf),
            np.nextafter(4.0 * period_ms, 0.0),
            np.asarray(
                [
                    np.nextafter(period_ms, math.inf),
                    np.nextafter(2.0 * period_ms, -math.inf),
                    np.nextafter(4.0 * period_ms, -math.inf),
                    4.0 * period_ms,
                ],
                dtype=np.float64,
            ),
        ),
        (0.0, 0.0, 4.0 * period_ms, np.asarray([0.5 * period_ms, 0.5 * period_ms])),
        (10.0, 10.0, 10.0, np.asarray([10.0], dtype=np.float64)),
        (10.0, 10.0, 1000.0, np.asarray((), dtype=np.float64)),
    )
    for tau_ms, start_ms, end_ms, peaks in edge_cases:
        expected = module._peak_recall_precision_cost(
            left_anchor_time_ms=tau_ms,
            bpm=bpm,
            start_ms=start_ms,
            end_ms=end_ms,
            beat_peak_times_ms=np.sort(peaks.astype(np.float64)),
            constants=widened,
        )
        actual = module._peak_recall_precision_cost_fast(
            left_anchor_time_ms=tau_ms,
            bpm=bpm,
            start_ms=start_ms,
            end_ms=end_ms,
            beat_peak_times_ms=np.sort(peaks.astype(np.float64)),
            constants=widened,
        )
        assert actual == expected

    fallback_calls = []

    def fake_reference(**kwargs):
        fallback_calls.append(kwargs)
        return 0.375

    monkeypatch.setattr(module, "_peak_recall_precision_cost", fake_reference)
    assert module._peak_recall_precision_cost_fast(
        left_anchor_time_ms=-1.0e20,
        bpm=bpm,
        start_ms=0.0,
        end_ms=period_ms,
        beat_peak_times_ms=np.asarray((), dtype=np.float64),
        constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
    ) == 0.375
    assert len(fallback_calls) == 1


def test_peak_recall_precision_normal_vectorized_path_matches_scalar_ulps_duplicates_and_large_indexes() -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    bpm = 120.0
    period_ms = 60000.0 / bpm
    tolerance_ms = min(
        GLOBAL_CONSTANT_JUMP_CONSTANTS.peak_grid_tolerance_ms,
        GLOBAL_CONSTANT_JUMP_CONSTANTS.peak_grid_tolerance_beat_fraction * period_ms,
    )
    large_index = 2**40
    large_tau_ms = -np.float64(large_index) * period_ms
    cases = (
        (
            0.0,
            np.nextafter(0.0, math.inf),
            np.nextafter(4.0 * period_ms, 0.0),
            np.asarray(
                [
                    0.0,
                    np.nextafter(period_ms, math.inf),
                    period_ms + tolerance_ms,
                    period_ms + tolerance_ms,
                    2.5 * period_ms,
                    np.nextafter(4.0 * period_ms, 0.0),
                ],
                dtype=np.float64,
            ),
        ),
        (
            large_tau_ms,
            0.0,
            4.0 * period_ms,
            np.asarray(
                [
                    0.0,
                    np.nextafter(period_ms, math.inf),
                    np.nextafter(3.0 * period_ms, -math.inf),
                    4.0 * period_ms,
                ],
                dtype=np.float64,
            ),
        ),
    )

    for tau_ms, start_ms, end_ms, peaks in cases:
        peak_times = np.sort(peaks)
        expected_reference = module._peak_recall_precision_cost(
            left_anchor_time_ms=tau_ms,
            bpm=bpm,
            start_ms=start_ms,
            end_ms=end_ms,
            beat_peak_times_ms=peak_times,
            constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
        )
        expected_scalar = module._peak_recall_precision_cost_fast_scalar(
            left_anchor_time_ms=tau_ms,
            bpm=bpm,
            start_ms=start_ms,
            end_ms=end_ms,
            beat_peak_times_ms=peak_times,
            constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
        )
        actual = module._peak_recall_precision_cost_fast(
            left_anchor_time_ms=tau_ms,
            bpm=bpm,
            start_ms=start_ms,
            end_ms=end_ms,
            beat_peak_times_ms=peak_times,
            constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
        )

        assert expected_scalar == expected_reference
        assert actual == expected_scalar


def test_generic_widened_lattice_tolerance_stays_on_scalar_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    widened = replace(
        GLOBAL_CONSTANT_JUMP_CONSTANTS,
        peak_grid_tolerance_ms=1000.0,
        peak_grid_tolerance_beat_fraction=0.6,
    )
    calls = []

    def fake_peak_scalar(**kwargs):
        calls.append(("peak", kwargs["constants"]))
        return 0.375

    def fake_support_scalar(**kwargs):
        calls.append(("support", kwargs["constants"]))
        return 0.625

    monkeypatch.setattr(module, "_peak_recall_precision_cost_fast_scalar", fake_peak_scalar)
    monkeypatch.setattr(module, "_beat_peak_support_for_lattice_count_ranking_scalar", fake_support_scalar)

    assert (
        module._peak_recall_precision_cost_fast(
            left_anchor_time_ms=0.0,
            bpm=120.0,
            start_ms=0.0,
            end_ms=2000.0,
            beat_peak_times_ms=np.asarray([250.0, 750.0], dtype=np.float64),
            constants=widened,
        )
        == 0.375
    )
    assert (
        module._beat_peak_support_for_lattice_count_ranking(
            left_anchor_time_ms=0.0,
            bpm=120.0,
            start_ms=0.0,
            end_ms=2000.0,
            beat_peak_times_ms=np.asarray([250.0, 750.0], dtype=np.float64),
            constants=widened,
        )
        == 0.625
    )
    assert calls == [("peak", widened), ("support", widened)]


def test_lattice_large_index_fallback_preserves_original_fast_path_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    bpm = 120.0
    period_ms = 60000.0 / bpm
    large_index = 2**53 + 128
    tau_ms = -np.float64(large_index) * period_ms
    peaks = np.asarray([0.0, period_ms, 2.0 * period_ms, 3.0 * period_ms, 4.0 * period_ms], dtype=np.float64)
    kwargs = dict(
        left_anchor_time_ms=tau_ms,
        bpm=bpm,
        start_ms=0.0,
        end_ms=4.0 * period_ms,
        beat_peak_times_ms=peaks,
        constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
    )

    expected_peak_cost = module._peak_recall_precision_cost(**kwargs)
    expected_support = module._beat_peak_support_for_lattice_count_ranking_scalar(**kwargs)
    original_support_scalar = module._beat_peak_support_for_lattice_count_ranking_scalar
    support_scalar_calls = []

    def recording_support_scalar(**kwargs):
        support_scalar_calls.append(kwargs)
        return original_support_scalar(**kwargs)

    monkeypatch.setattr(module, "_beat_peak_support_for_lattice_count_ranking_scalar", recording_support_scalar)

    assert module._peak_recall_precision_cost_fast(**kwargs) == expected_peak_cost
    assert module._beat_peak_support_for_lattice_count_ranking(**kwargs) == expected_support
    assert len(support_scalar_calls) == 1


def test_count_ranking_extreme_index_fallback_matches_original_scalar_not_materialized_reference() -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    bpm = 1000.0
    period_ms = 60000.0 / bpm
    kwargs = dict(
        left_anchor_time_ms=-float(2**60) * period_ms,
        bpm=bpm,
        start_ms=0.0,
        end_ms=15_360.0,
        beat_peak_times_ms=np.asarray([0.0], dtype=np.float64),
        constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
    )

    expected_scalar = module._beat_peak_support_for_lattice_count_ranking_scalar(**kwargs)
    materialized_reference = module._beat_peak_support_for_lattice(**kwargs)

    assert expected_scalar == 0.023255813953488413
    assert materialized_reference == 1.0
    assert module._beat_peak_support_for_lattice_count_ranking(**kwargs) == expected_scalar


def test_count_ranking_lattice_support_fast_path_matches_reference_random() -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    rng = np.random.default_rng(13)
    for _ in range(2_000):
        bpm = float(rng.uniform(40.0, 900.0))
        period_ms = 60000.0 / bpm
        tau_ms = float(rng.uniform(-2000.0, 2000.0))
        start_offset = float(rng.uniform(-3.0, 8.0))
        span_beats = float(rng.uniform(0.0, 160.0))
        start_ms = tau_ms + start_offset * period_ms + float(rng.uniform(-30.0, 30.0))
        end_ms = start_ms + span_beats * period_ms + float(rng.uniform(-10.0, 30.0))
        peak_count = int(rng.integers(0, 36))
        peak_times = np.sort(
            rng.uniform(
                min(start_ms, end_ms) - 2.0 * period_ms,
                max(start_ms, end_ms) + 2.0 * period_ms,
                size=peak_count,
            ).astype(np.float64)
        )

        expected = module._beat_peak_support_for_lattice(
            left_anchor_time_ms=tau_ms,
            bpm=bpm,
            start_ms=start_ms,
            end_ms=end_ms,
            beat_peak_times_ms=peak_times,
            constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
        )
        actual = module._beat_peak_support_for_lattice_count_ranking(
            left_anchor_time_ms=tau_ms,
            bpm=bpm,
            start_ms=start_ms,
            end_ms=end_ms,
            beat_peak_times_ms=peak_times,
            constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
        )

        assert actual == expected


def test_count_ranking_lattice_support_fast_path_matches_reference_edge_cases() -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    cases = []
    tau_ms = -0.0
    bpm = 120.0
    period_ms = 60000.0 / bpm
    cases.append((tau_ms, bpm, -0.0, np.nextafter(period_ms * 4.0, math.inf), np.asarray([-0.0], dtype=np.float64)))
    cases.append((0.0, bpm, np.nextafter(period_ms, math.inf), period_ms * 4.0, np.asarray([period_ms], dtype=np.float64)))
    cases.append((0.0, bpm, 0.0, np.nextafter(period_ms * 3.0, 0.0), np.asarray([period_ms * 3.0], dtype=np.float64)))
    cases.append((10.0, 333.0, 10.0, 10.0, np.asarray([10.0], dtype=np.float64)))
    cases.append((10.0, 333.0, 10.0, 1000.0, np.asarray((), dtype=np.float64)))
    cases.append(
        (
            np.nextafter(0.0, -math.inf),
            250.0,
            np.nextafter(0.0, math.inf),
            np.nextafter(60000.0 / 250.0 * 8.0, -math.inf),
            np.asarray(
                [
                    np.nextafter(60000.0 / 250.0, math.inf),
                    np.nextafter(60000.0 / 250.0 * 2.0, -math.inf),
                    np.nextafter(60000.0 / 250.0 * 7.0, math.inf),
                ],
                dtype=np.float64,
            ),
        )
    )

    for tau_ms, bpm, start_ms, end_ms, peaks in cases:
        expected = module._beat_peak_support_for_lattice(
            left_anchor_time_ms=tau_ms,
            bpm=bpm,
            start_ms=start_ms,
            end_ms=end_ms,
            beat_peak_times_ms=peaks,
            constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
        )
        actual = module._beat_peak_support_for_lattice_count_ranking(
            left_anchor_time_ms=tau_ms,
            bpm=bpm,
            start_ms=start_ms,
            end_ms=end_ms,
            beat_peak_times_ms=peaks,
            constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
        )

        assert actual == expected


def test_count_ranking_normal_vectorized_path_matches_scalar_ulps_duplicates_and_large_indexes() -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    bpm = 120.0
    period_ms = 60000.0 / bpm
    tolerance_ms = min(
        GLOBAL_CONSTANT_JUMP_CONSTANTS.peak_grid_tolerance_ms,
        GLOBAL_CONSTANT_JUMP_CONSTANTS.peak_grid_tolerance_beat_fraction * period_ms,
    )
    large_index = 2**40
    large_tau_ms = -np.float64(large_index) * period_ms
    cases = (
        (
            0.0,
            np.nextafter(0.0, math.inf),
            np.nextafter(4.0 * period_ms, 0.0),
            np.asarray(
                [
                    0.0,
                    np.nextafter(period_ms, math.inf),
                    period_ms + tolerance_ms,
                    period_ms + tolerance_ms,
                    2.5 * period_ms,
                    np.nextafter(4.0 * period_ms, 0.0),
                ],
                dtype=np.float64,
            ),
        ),
        (
            large_tau_ms,
            0.0,
            4.0 * period_ms,
            np.asarray(
                [
                    0.0,
                    np.nextafter(period_ms, math.inf),
                    np.nextafter(3.0 * period_ms, -math.inf),
                    4.0 * period_ms,
                ],
                dtype=np.float64,
            ),
        ),
    )

    for tau_ms, start_ms, end_ms, peaks in cases:
        peak_times = np.sort(peaks)
        expected_reference = module._beat_peak_support_for_lattice(
            left_anchor_time_ms=tau_ms,
            bpm=bpm,
            start_ms=start_ms,
            end_ms=end_ms,
            beat_peak_times_ms=peak_times,
            constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
        )
        expected_scalar = module._beat_peak_support_for_lattice_count_ranking_scalar(
            left_anchor_time_ms=tau_ms,
            bpm=bpm,
            start_ms=start_ms,
            end_ms=end_ms,
            beat_peak_times_ms=peak_times,
            constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
        )
        actual = module._beat_peak_support_for_lattice_count_ranking(
            left_anchor_time_ms=tau_ms,
            bpm=bpm,
            start_ms=start_ms,
            end_ms=end_ms,
            beat_peak_times_ms=peak_times,
            constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
        )

        assert expected_scalar == expected_reference
        assert actual == expected_scalar


def test_count_ranking_lattice_support_dedupes_multiple_peaks_and_can_match_adjacent_grid_times() -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    constants = replace(
        GLOBAL_CONSTANT_JUMP_CONSTANTS,
        peak_grid_tolerance_ms=1000.0,
        peak_grid_tolerance_beat_fraction=0.6,
    )
    bpm = 120.0
    period_ms = 60000.0 / bpm
    peaks = np.sort(
        np.asarray(
            [
                period_ms * 0.5,
                period_ms * 0.5,
                period_ms * 2.5,
                np.nextafter(period_ms * 2.5, math.inf),
            ],
            dtype=np.float64,
        )
    )

    expected = module._beat_peak_support_for_lattice(
        left_anchor_time_ms=0.0,
        bpm=bpm,
        start_ms=0.0,
        end_ms=period_ms * 4.0,
        beat_peak_times_ms=peaks,
        constants=constants,
    )
    actual = module._beat_peak_support_for_lattice_count_ranking(
        left_anchor_time_ms=0.0,
        bpm=bpm,
        start_ms=0.0,
        end_ms=period_ms * 4.0,
        beat_peak_times_ms=peaks,
        constants=constants,
    )

    assert actual == expected
    assert actual == 1.0


def test_count_ranking_uses_fast_lattice_support_for_long_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    context, _ = _constant_scoring_context(duration_ms=20_000.0, bpm=120.0)
    left_anchor = module._Anchor(anchor_id=0, kind="origin", time_ms=0.0, rank_score=1.0)
    right_anchor = module._Anchor(anchor_id=1, kind="boundary", time_ms=3_600_000.0, rank_score=1.0)
    peaks = np.asarray([0.0, 500.0, 999_500.0, 2_000_000.0, 3_599_500.0], dtype=np.float64)
    context.beat_peak_times_ms = peaks
    expected_support = module._beat_peak_support_for_lattice(
        left_anchor_time_ms=0.0,
        bpm=120.0,
        start_ms=0.0,
        end_ms=right_anchor.time_ms,
        beat_peak_times_ms=peaks,
        constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
    )

    def forbidden_reference(**kwargs):
        raise AssertionError("count ranking should use the fast lattice support path")

    monkeypatch.setattr(module, "_beat_peak_support_for_lattice", forbidden_reference)
    ranked = module._rank_beat_count_candidates_with_lazy_support(
        [(0.0, 7200, 120.0)],
        left_anchor=left_anchor,
        right_anchor=right_anchor,
        context=context,
    )

    assert ranked == [(7200, 0.0, expected_support)]


def test_lattice_vectorized_micro_performance_beats_scalar_reference() -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    bpm = 120.0
    period_ms = 60000.0 / bpm
    grid_count = 50_000
    grid_times = np.arange(grid_count, dtype=np.float64) * period_ms
    peak_times = np.sort(
        np.concatenate(
            [
                grid_times + np.where(np.arange(grid_count) % 2 == 0, 1.0, -1.0),
                grid_times[::10] + 2.0,
                np.asarray([-period_ms, grid_count * period_ms], dtype=np.float64),
            ]
        ).astype(np.float64)
    )
    kwargs = dict(
        left_anchor_time_ms=0.0,
        bpm=bpm,
        start_ms=0.0,
        end_ms=grid_count * period_ms,
        beat_peak_times_ms=peak_times,
        constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
    )

    support_warmup = module._beat_peak_support_for_lattice_count_ranking(**kwargs)
    recall_warmup = module._peak_recall_precision_cost_fast(**kwargs)

    started = time.perf_counter()
    support_scalar = module._beat_peak_support_for_lattice_count_ranking_scalar(**kwargs)
    support_scalar_seconds = time.perf_counter() - started
    started = time.perf_counter()
    support_fast = module._beat_peak_support_for_lattice_count_ranking(**kwargs)
    support_fast_seconds = time.perf_counter() - started

    started = time.perf_counter()
    recall_scalar = module._peak_recall_precision_cost_fast_scalar(**kwargs)
    recall_scalar_seconds = time.perf_counter() - started
    started = time.perf_counter()
    recall_fast = module._peak_recall_precision_cost_fast(**kwargs)
    recall_fast_seconds = time.perf_counter() - started

    assert support_warmup == support_scalar
    assert recall_warmup == recall_scalar
    assert support_fast == support_scalar
    assert recall_fast == recall_scalar
    assert support_fast_seconds * 3.0 < support_scalar_seconds
    assert recall_fast_seconds * 3.0 < recall_scalar_seconds


def test_iter_variants_matches_independent_fits_and_mapping_api() -> None:
    prediction = _jump_prediction(first_bpm=120.0, second_bpm=150.0, boundary_ms=12_000.0, duration_ms=24_000.0)
    candidates = extract_global_constant_jump_candidates(prediction)
    variants = (VARIANT_CJ0, VARIANT_CJ1, VARIANT_CJ2, VARIANT_CJ3)

    independent = {
        variant: fit_global_constant_jump(prediction, variant=variant, candidate_set=candidates)
        for variant in variants
    }
    iterated = dict(iter_global_constant_jump_variants(prediction, variants=variants, candidate_set=candidates))
    mapped = fit_global_constant_jump_variants(prediction, variants=variants, candidate_set=candidates)

    assert tuple(iterated) == variants
    assert all(result.variant == variant for variant, result in iterated.items())
    assert tuple(mapped) == variants
    for variant in variants:
        _assert_public_result_payload_exact(iterated[variant], independent[variant])
        _assert_public_result_payload_exact(mapped[variant], independent[variant])


def test_iter_variants_keeps_attempt_cap_independent_per_variant() -> None:
    prediction = _constant_prediction(duration_ms=20_000.0, bpm=120.0)
    candidates = extract_global_constant_jump_candidates(prediction)

    results = dict(
        iter_global_constant_jump_variants(
            prediction,
            variants=(VARIANT_CJ0, VARIANT_CJ1),
            attempt_cap=1,
            candidate_set=candidates,
        )
    )

    assert tuple(results) == (VARIANT_CJ0, VARIANT_CJ1)
    assert [result.reason for result in results.values()] == [
        REASON_EDGE_ATTEMPT_CAP_EXCEEDED,
        REASON_EDGE_ATTEMPT_CAP_EXCEEDED,
    ]
    assert [result.diagnostics.section_attempt_count for result in results.values()] == [1, 1]


def test_fallback_replay_fingerprint_binds_search_trace_and_cap_rejected_key() -> None:
    prediction = _constant_prediction(duration_ms=20_000.0, bpm=120.0)
    candidates = extract_global_constant_jump_candidates(prediction)

    cap1 = fit_global_constant_jump(prediction, variant=VARIANT_CJ0, attempt_cap=1, candidate_set=candidates)
    cap1_replay = fit_global_constant_jump(prediction, variant=VARIANT_CJ0, attempt_cap=1, candidate_set=candidates)
    cap2 = fit_global_constant_jump(prediction, variant=VARIANT_CJ0, attempt_cap=2, candidate_set=candidates)

    assert cap1.reason == REASON_EDGE_ATTEMPT_CAP_EXCEEDED
    assert cap1_replay.reason == REASON_EDGE_ATTEMPT_CAP_EXCEEDED
    assert cap2.reason == REASON_EDGE_ATTEMPT_CAP_EXCEEDED
    assert cap1.diagnostics.section_attempt_count == 1
    assert cap1_replay.diagnostics.section_attempt_count == 1
    assert cap2.diagnostics.section_attempt_count == 2
    assert cap1.diagnostics.replay_fingerprint == cap1_replay.diagnostics.replay_fingerprint
    assert cap1.diagnostics.replay_fingerprint != cap2.diagnostics.replay_fingerprint


def test_shifted_origin_prefix_is_scored_as_its_own_closed_partition() -> None:
    prediction = _body_only_prediction(duration_ms=10_000.0, bpm=120.0, origin_time_ms=1000.0)
    candidates = _single_origin_candidate_set(prediction, origin_time_ms=1000.0, bpm=120.0)

    result = fit_global_constant_jump(prediction, variant=VARIANT_CJ1, candidate_set=candidates)

    assert not result.ok
    assert result.reason == REASON_NO_GLOBAL_CONSTANT_JUMP_PATH


def test_candidate_set_reuse_rejects_tamper_stale_counts_nonfinite_and_caps() -> None:
    prediction = _jump_prediction(first_bpm=120.0, second_bpm=150.0, boundary_ms=12_000.0, duration_ms=24_000.0)
    candidate_set = extract_global_constant_jump_candidates(prediction)
    assert candidate_set.boundary_candidates

    stale_count = replace(
        candidate_set,
        diagnostics=replace(
            candidate_set.diagnostics,
            boundary_candidate_count=candidate_set.diagnostics.boundary_candidate_count + 1,
        ),
    )
    with pytest.raises(ValueError, match="boundary_candidate_count"):
        fit_global_constant_jump(prediction, candidate_set=stale_count)

    nonfinite_tempo = replace(
        candidate_set,
        tempo_candidates=(replace(candidate_set.tempo_candidates[0], score=math.inf),)
        + candidate_set.tempo_candidates[1:],
    )
    with pytest.raises(ValueError, match="tempo_candidates"):
        fit_global_constant_jump(prediction, candidate_set=nonfinite_tempo)

    tampered_boundary = replace(
        candidate_set,
        boundary_candidates=(replace(candidate_set.boundary_candidates[0], rank_score=999.0),)
        + candidate_set.boundary_candidates[1:],
    )
    with pytest.raises(ValueError, match="candidate_fingerprint"):
        fit_global_constant_jump(prediction, candidate_set=tampered_boundary)

    stale_source_boundary = replace(
        candidate_set,
        boundary_candidates=(replace(candidate_set.boundary_candidates[0], source_peak_time_ms=-1.0),)
        + candidate_set.boundary_candidates[1:],
    )
    with pytest.raises(ValueError, match="source peak"):
        fit_global_constant_jump(prediction, candidate_set=stale_source_boundary)

    too_many_tempos = replace(
        candidate_set,
        tempo_candidates=tuple(
            replace(candidate_set.tempo_candidates[0], bpm=20.0 + index)
            for index in range(GLOBAL_CONSTANT_JUMP_CONSTANTS.max_tempo_candidates_retained + 1)
        ),
        diagnostics=replace(
            candidate_set.diagnostics,
            tempo_candidate_count=GLOBAL_CONSTANT_JUMP_CONSTANTS.max_tempo_candidates_retained + 1,
        ),
    )
    with pytest.raises(ValueError, match="tempo candidate cap"):
        fit_global_constant_jump(prediction, candidate_set=too_many_tempos)


def test_boundary_bin_cap_never_exceeds_hard_cap_when_populated_bins_exceed_cap() -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    raw_candidates = tuple(
        module._RawBoundaryCandidate(
            time_ms=float(index * 60_000.0),
            source_peak_index=index,
            source_peak_time_ms=float(index * 60_000.0),
            source_peak_confidence=1.0,
            rank_score=float(1000 - index),
            evidence_mode="ordinary",
            left_period_ms=500.0,
            right_period_ms=500.0,
            ordinary_score=float(1000 - index),
            super_score=None,
            downbeat_bonus=0.0,
            nearest_downbeat_distance_ms=None,
        )
        for index in range(250)
    )

    retained = module._cap_boundary_candidates_by_bins(
        raw_candidates,
        cap=GLOBAL_CONSTANT_JUMP_CONSTANTS.max_interior_boundary_candidates,
        coverage_start_ms=0.0,
        constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
    )

    assert len(retained) == GLOBAL_CONSTANT_JUMP_CONSTANTS.max_interior_boundary_candidates
    assert {candidate.source_peak_index for candidate in retained} == set(range(192))


def test_extreme_frame_rate_rejects_nonfinite_derived_period_bounds() -> None:
    prediction = FrameTimingPrediction(
        provider="cached-beatthis",
        beat_prob=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        downbeat_prob=np.zeros(3, dtype=np.float32),
        frame_rate_hz=1e308,
    )

    with pytest.raises(ValueError, match="period frame bounds"):
        extract_global_constant_jump_candidates(prediction)
    with pytest.raises(ValueError, match="period frame bounds"):
        fit_global_constant_jump(prediction, variant=VARIANT_CJ1)


def test_boundary_support_uses_max_supported_peak_within_tolerance_not_nearest() -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    support = module._support_near_time(
        100.0,
        np.asarray([95.0, 140.0], dtype=np.float64),
        np.asarray([0.1, 1.0], dtype=np.float64),
        tolerance_ms=60.0,
    )

    assert support == pytest.approx(1.0 * (1.0 - 40.0 / 60.0))


def test_grid_times_are_exact_half_open_at_ulp_boundaries() -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    assert module._grid_times_in_interval(0.0, 100.0, 0.0, 100.0).tolist() == [0.0]
    assert module._grid_times_in_interval(0.0, 100.0, 0.0, math.nextafter(100.0, math.inf)).tolist() == [0.0, 100.0]
    assert module._grid_times_in_interval(0.0, 100.0, math.nextafter(100.0, -math.inf), 200.0).tolist() == [100.0]


def test_nonempty_zero_norm_downbeat_interval_is_invalid_but_empty_interval_is_zero() -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    prediction = _constant_prediction(duration_ms=5_000.0, bpm=120.0, downbeat_phase=None)
    downbeat_prob = np.zeros(prediction.frame_count, dtype=np.float32)
    downbeat_prob[0] = 1.0
    prediction = FrameTimingPrediction(
        provider="cached-beatthis",
        beat_prob=prediction.beat_prob,
        downbeat_prob=downbeat_prob,
        frame_rate_hz=prediction.frame_rate_hz,
    )
    candidates = _single_origin_candidate_set(prediction, origin_time_ms=1000.0, bpm=120.0)
    context = module._SearchContext(
        prediction=prediction,
        candidates=candidates,
        constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
        attempt_cap=GLOBAL_CONSTANT_JUMP_CONSTANTS.default_edge_attempt_cap,
    )

    invalid = module._section_interval_score(
        left_anchor_time_ms=1000.0,
        left_beat_at_anchor=0,
        beat_count=4,
        bpm=120.0,
        downbeat_phase=0,
        context=context,
        variant=VARIANT_CJ3,
        start_ms=1000.0,
        end_ms=3000.0,
    )
    empty = module._section_interval_score(
        left_anchor_time_ms=1000.0,
        left_beat_at_anchor=0,
        beat_count=4,
        bpm=120.0,
        downbeat_phase=0,
        context=context,
        variant=VARIANT_CJ3,
        start_ms=1000.0,
        end_ms=1000.0,
    )

    assert not invalid.valid
    assert empty.valid
    assert empty.cost == 0.0


def test_exact_origin_and_boundary_merge_distances_are_inclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    raw = (
        module._RawBoundaryCandidate(0.0, 0, 0.0, 1.0, 1.0, "ordinary", 500.0, 500.0, 1.0, None, 0.0, None),
        module._RawBoundaryCandidate(8000.0, 1, 8000.0, 1.0, 2.0, "ordinary", 500.0, 500.0, 2.0, None, 0.0, None),
    )
    assert module._merge_boundary_candidates(raw, GLOBAL_CONSTANT_JUMP_CONSTANTS) == (raw[1],)

    beat_signal = np.ones(100, dtype=np.float64)

    def fake_correlation(signal, frame_times_ms, *, tau_ms, bpm, start_ms, end_ms, pulse_width_ms=40.0):
        if bpm == 120.0:
            return 1.0 if tau_ms == 0.0 else 0.0
        return 0.9 if tau_ms == 20.0 else 0.0

    monkeypatch.setattr(module, "pulse_correlation_v1", fake_correlation)
    origins = module._origin_candidates(
        beat_signal,
        (
            module.TempoCandidate(bpm=120.0, source="test", score=1.0),
            module.TempoCandidate(bpm=125.0, source="test", score=1.0),
        ),
        frame_rate_hz=50.0,
        coverage_start_ms=0.0,
        coverage_end_ms=2000.0,
        constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
    )

    assert len(origins) == 1
    assert origins[0].time_ms == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tau_ms": math.nan, "bpm": 120.0, "start_ms": 0.0, "end_ms": 20.0},
        {"tau_ms": 0.0, "bpm": 0.0, "start_ms": 0.0, "end_ms": 20.0},
        {"tau_ms": 0.0, "bpm": 120.0, "start_ms": 20.0, "end_ms": 0.0},
    ],
)
def test_public_pulse_correlation_validates_finite_inputs(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        pulse_correlation_v1(
            np.asarray([0.0, 1.0], dtype=np.float64),
            np.asarray([0.0, 20.0], dtype=np.float64),
            **kwargs,
        )

    assert pulse_correlation_v1(
        np.asarray([0.0, 1.0], dtype=np.float64),
        np.asarray([0.0, 20.0], dtype=np.float64),
        tau_ms=0.0,
        bpm=120.0,
        start_ms=0.0,
        end_ms=0.0,
    ) == 0.0


def test_outgoing_boundary_anchors_cache_reuses_exact_sorted_result() -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    class TrappingBoundaryAnchors(tuple):
        def __new__(cls, values):
            obj = super().__new__(cls, values)
            obj.iteration_count = 0
            return obj

        def __iter__(self):
            self.iteration_count += 1
            if self.iteration_count > 1:
                raise AssertionError("cached outgoing anchors should not rescan boundary anchors")
            return super().__iter__()

    context, _ = _constant_scoring_context(duration_ms=40_000.0, bpm=120.0)
    origin_anchor = module._Anchor(anchor_id=7, kind="origin", time_ms=1_000.0, rank_score=1.0)
    state = module._initial_state(origin_anchor, downbeat_phase=None)
    boundary_anchors = TrappingBoundaryAnchors(
        (
            module._Anchor(anchor_id=11, kind="boundary", time_ms=9_000.0, rank_score=0.1),
            module._Anchor(anchor_id=12, kind="boundary", time_ms=20_000.0, rank_score=0.8),
            module._Anchor(anchor_id=13, kind="boundary", time_ms=20_000.0, rank_score=0.9),
            module._Anchor(anchor_id=14, kind="boundary", time_ms=30_000.0, rank_score=0.4),
        )
    )

    first = module._outgoing_boundary_anchors(state, boundary_anchors, context)
    second = module._outgoing_boundary_anchors(replace(state, section_count=3), boundary_anchors, context)

    assert first is second
    assert tuple(anchor.anchor_id for anchor in first) == (11, 13, 12, 14)
    assert boundary_anchors.iteration_count == 1


def test_beam_expands_frozen_buckets_in_anchor_section_lexicographic_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    prediction = _constant_prediction(duration_ms=60_000.0, bpm=120.0)
    beat_peaks = materialize_global_constant_jump_peaks(prediction.beat_prob, frame_rate_hz=prediction.frame_rate_hz)
    boundary_candidates = tuple(
        _boundary_candidate_from_peak(beat_peaks, peak_index=20 + index * 20, anchor_id=index)
        for index in range(4)
    )
    candidates = _candidate_set_for_prediction(
        prediction,
        beat_peaks=beat_peaks,
        downbeat_peaks=(),
        tempo_candidates=(module.TempoCandidate(bpm=120.0, source="test", score=1.0),),
        origin_candidates=(module.OriginCandidate(anchor_id=0, time_ms=0.0, bpm=120.0, score=1.0),),
        boundary_candidates=boundary_candidates,
    )
    context = module._SearchContext(
        prediction=prediction,
        candidates=candidates,
        constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
        attempt_cap=GLOBAL_CONSTANT_JUMP_CONSTANTS.default_edge_attempt_cap,
    )
    context.interior_edge_score_bundle_enabled = False
    boundaries = module._boundary_anchors(candidates)
    boundary_by_id = {anchor.anchor_id: anchor for anchor in boundaries}
    expansion_order: list[tuple[int, int]] = []

    def fake_terminal_completions(state, candidates, context, *, variant):
        expansion_order.append((state.anchor.anchor_id, state.section_count))
        return ()

    def fake_outgoing(state, boundary_anchors, context):
        if state.anchor.kind == "origin":
            return (boundary_by_id[1], boundary_by_id[4])
        if state.anchor.anchor_id == 1:
            return (boundary_by_id[3],)
        return ()

    def fake_count_candidates(left_anchor, right_anchor, context):
        return (module._BeatCountCandidate(count=16, bpm=120.0, rank=0, tempo_distance=0.0, beat_support=1.0),)

    def fake_advance_state_into_bucket(states_by_bucket, state, right_anchor, count_candidate, *, context, variant):
        next_state = module._State(
            anchor=right_anchor,
            section_count=state.section_count + 1,
            beat_at_anchor=state.beat_at_anchor + count_candidate.count,
            prev_bpm=count_candidate.bpm,
            prev_alias_family=120.0,
            downbeat_phase=state.downbeat_phase,
            origin_time_ms=state.origin_time_ms,
            duration_objective=state.duration_objective,
            transition_objective=state.transition_objective,
            objective=state.objective,
            alias_switch_count=state.alias_switch_count,
            max_boundary_displacement_ms=state.max_boundary_displacement_ms,
            sections=state.sections,
            edge_tuples=state.edge_tuples + ((state.anchor.anchor_id, right_anchor.anchor_id, state.section_count),),
            replay_key=state.replay_key + ((right_anchor.anchor_id, state.section_count),),
        )
        module._insert_state_into_beam_bucket(
            states_by_bucket,
            (right_anchor.anchor_id, next_state.section_count),
            next_state,
            context,
        )

    monkeypatch.setattr(module, "_terminal_completions", fake_terminal_completions)
    monkeypatch.setattr(module, "_outgoing_boundary_anchors", fake_outgoing)
    monkeypatch.setattr(module, "_beat_count_candidates", fake_count_candidates)
    monkeypatch.setattr(module, "_advance_state_into_beam_bucket", fake_advance_state_into_bucket)

    assert module._assemble_beam(candidates, context, variant=VARIANT_CJ1) is None
    assert expansion_order == [(0, 0), (1, 1), (3, 2), (4, 1)]


def test_bounded_bucket_insertion_matches_stable_sorted_top64_and_counts_pruned() -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    context, _ = _constant_scoring_context(duration_ms=10_000.0, bpm=120.0)
    anchor = module._Anchor(anchor_id=0, kind="origin", time_ms=0.0, rank_score=1.0)
    base = module._initial_state(anchor, downbeat_phase=None)
    states = tuple(
        replace(
            base,
            objective=float((index * 37) % 97),
            alias_switch_count=index % 3,
            prev_bpm=80.0 + index,
            replay_key=(index,),
        )
        for index in range(96)
    )
    bucket_key = (0, 0)
    buckets: dict[tuple[int, int], list[tuple[tuple[object, ...], object]]] = {}
    reference: list[object] = []
    reference_pruned = 0

    for state in states:
        module._insert_state_into_beam_bucket(buckets, bucket_key, state, context)
        reference.append(state)
        reference.sort(key=module._state_order_key)
        if len(reference) > GLOBAL_CONSTANT_JUMP_CONSTANTS.beam_width:
            del reference[GLOBAL_CONSTANT_JUMP_CONSTANTS.beam_width :]
            reference_pruned += 1

    retained = [state for _, state in buckets[bucket_key]]
    assert retained == reference
    assert context.beam_pruned_state_count == reference_pruned

    equal_context, _ = _constant_scoring_context(duration_ms=10_000.0, bpm=120.0)
    equal_bucket: dict[tuple[int, int], list[tuple[tuple[object, ...], object]]] = {}
    equal_states = tuple(replace(base, prev_bpm=80.0 + index) for index in range(65))
    for state in equal_states:
        module._insert_state_into_beam_bucket(equal_bucket, bucket_key, state, equal_context)

    equal_retained = [state for _, state in equal_bucket[bucket_key]]
    assert equal_retained == list(equal_states[: GLOBAL_CONSTANT_JUMP_CONSTANTS.beam_width])
    assert equal_context.beam_pruned_state_count == 1


def test_pending_bucket_insertion_rejects_without_materializing(monkeypatch: pytest.MonkeyPatch) -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    context, _ = _constant_scoring_context(duration_ms=10_000.0, bpm=120.0)
    anchor = module._Anchor(anchor_id=0, kind="origin", time_ms=0.0, rank_score=1.0)
    right_anchor = module._Anchor(anchor_id=1, kind="boundary", time_ms=10_000.0, rank_score=1.0)
    base = module._initial_state(anchor, downbeat_phase=None)
    bucket_key = (right_anchor.anchor_id, 1)
    buckets: dict[tuple[int, int], list] = {}
    for index in range(GLOBAL_CONSTANT_JUMP_CONSTANTS.beam_width):
        retained = replace(
            base,
            anchor=right_anchor,
            section_count=1,
            objective=float(index),
            edge_tuples=((0, 1, index + 1, 120.0),),
            replay_key=base.replay_key + ((right_anchor.anchor_id, index, (0, 1, index + 1, 120.0)),),
        )
        module._insert_state_into_beam_bucket(buckets, bucket_key, retained, context)

    def forbidden_materialize(pending_state, context):
        raise AssertionError("discarded pending candidate was materialized")

    monkeypatch.setattr(module, "_materialize_pending_interior_state", forbidden_materialize)
    edge_tuple = (0, 1, 999, 120.0)
    replay_item = (right_anchor.anchor_id, 999, edge_tuple)
    rejected = module._PendingInteriorState(
        parent=base,
        anchor=right_anchor,
        section_count=1,
        beat_at_anchor=999,
        bpm=120.0,
        first_start_beat=0,
        duration_objective=999.0,
        transition_objective=0.0,
        objective=999.0,
        alias_switch_count=0,
        max_boundary_displacement_ms=0.0,
        edge_tuple=edge_tuple,
        replay_item=replay_item,
        order_key=module._pending_state_order_key(
            base,
            section_count=1,
            objective=999.0,
            alias_switch_count=0,
            max_boundary_displacement_ms=0.0,
            edge_tuple=edge_tuple,
            replay_item=replay_item,
        ),
    )

    module._insert_pending_state_into_beam_bucket(buckets, bucket_key, rejected, context)

    assert context.beam_pruned_state_count == 1
    assert [state.objective for _, state in buckets[bucket_key]] == [float(index) for index in range(64)]


@pytest.mark.parametrize("tie_mode", [False, True])
def test_pre_admission_interior_insert_matches_legacy_append_sort_and_pruned_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tie_mode: bool,
) -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    prediction = _constant_prediction(duration_ms=90_000.0, bpm=120.0, downbeat_phase=None)
    candidate_set = _candidate_set_for_prediction(
        prediction,
        beat_peaks=(),
        downbeat_peaks=(),
        tempo_candidates=(module.TempoCandidate(bpm=120.0, source="test", score=1.0),),
        origin_candidates=(module.OriginCandidate(anchor_id=0, time_ms=0.0, bpm=120.0, score=1.0),),
        boundary_candidates=(),
    )

    def cheap_section_score(
        left_anchor,
        *,
        right_anchor,
        left_beat_at_anchor,
        beat_count,
        bpm,
        downbeat_phase,
        context,
        variant,
    ):
        cache = context.terminal_score_cache if right_anchor is None else context.section_score_cache
        cache_key = module._section_score_cache_key(
            left_anchor,
            right_anchor=right_anchor,
            left_beat_at_anchor=left_beat_at_anchor,
            beat_count=beat_count,
            bpm=bpm,
            downbeat_phase=downbeat_phase,
            variant=variant,
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        context.record_search_trace_event("score_cache_miss", cache_key)
        context.check_attempt(cache_key)
        cost = 0.0 if tie_mode else float(((int(beat_count) * 37) % 101) / 100.0)
        score = module._SectionScore(
            valid=True,
            cost=cost,
            beat_support_cost=0.0,
            peak_recall_precision_cost=0.0,
            downbeat_phase_cost=0.0,
            bpm_prior_cost=0.0,
            beat_count_prior_cost=0.0,
            section_duration_cost=0.0,
        )
        cache[cache_key] = score
        return score

    def zero_transition_score(state, *, right_bpm, context, variant):
        cache_key = (state.prev_bpm, float(right_bpm), state.anchor.time_ms, variant)
        cached = context.transition_score_cache.get(cache_key)
        if cached is not None:
            return cached
        score = module._TransitionScore(
            cost=0.0,
            alias_switch_cost=0.0,
            alias_switch_increment=0,
            jump_size_cost=0.0,
            boundary_support_cost=0.0,
        )
        context.transition_score_cache[cache_key] = score
        return score

    monkeypatch.setattr(module, "_section_score", cheap_section_score)
    monkeypatch.setattr(module, "_transition_score", zero_transition_score)

    right_anchor = module._Anchor(anchor_id=1, kind="boundary", time_ms=30_000.0, rank_score=1.0)
    count_candidates = tuple(
        module._BeatCountCandidate(
            count=index + 1,
            bpm=80.0 + ((index * 17) % 320),
            rank=index,
            tempo_distance=0.0,
            beat_support=1.0,
        )
        for index in range(160)
    )
    optimized_context = module._SearchContext(
        prediction=prediction,
        candidates=candidate_set,
        constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
        attempt_cap=GLOBAL_CONSTANT_JUMP_CONSTANTS.default_edge_attempt_cap,
    )
    reference_context = module._SearchContext(
        prediction=prediction,
        candidates=candidate_set,
        constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
        attempt_cap=GLOBAL_CONSTANT_JUMP_CONSTANTS.default_edge_attempt_cap,
    )
    optimized_state = module._initial_state(module._origin_anchors(candidate_set)[0], downbeat_phase=None)
    reference_state = module._initial_state(module._origin_anchors(candidate_set)[0], downbeat_phase=None)
    optimized_buckets: dict[tuple[int, int], list] = {}
    reference_bucket: list = []
    reference_pruned = 0

    for count_candidate in count_candidates:
        module._advance_state_into_beam_bucket(
            optimized_buckets,
            optimized_state,
            right_anchor,
            count_candidate,
            context=optimized_context,
            variant=VARIANT_CJ1,
        )
        next_state = module._advance_state(
            reference_state,
            right_anchor,
            count_candidate,
            context=reference_context,
            variant=VARIANT_CJ1,
        )
        if next_state is not None:
            reference_bucket.append(next_state)
            reference_bucket.sort(key=module._state_order_key)
            if len(reference_bucket) > GLOBAL_CONSTANT_JUMP_CONSTANTS.beam_width:
                del reference_bucket[GLOBAL_CONSTANT_JUMP_CONSTANTS.beam_width :]
                reference_pruned += 1

    optimized_retained = [state for _, state in optimized_buckets[(right_anchor.anchor_id, 1)]]
    assert optimized_retained == reference_bucket
    assert optimized_context.beam_pruned_state_count == reference_pruned
    assert optimized_context.beam_pruned_state_count > 0
    assert optimized_context.attempt_count == reference_context.attempt_count
    assert len(optimized_context.section_score_cache) == len(reference_context.section_score_cache)
    assert [state.replay_key for state in optimized_retained] == [state.replay_key for state in reference_bucket]


def test_pre_admission_primitive_key_matches_materialized_state_order_randomized() -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    context, _ = _constant_scoring_context(duration_ms=10_000.0, bpm=120.0)
    left_anchor = module._Anchor(anchor_id=0, kind="origin", time_ms=0.0, rank_score=1.0)
    right_anchor = module._Anchor(anchor_id=1, kind="boundary", time_ms=10_000.0, rank_score=1.0)

    def edge_prefix(seed: int, length: int) -> tuple[tuple[int, int, int, float], ...]:
        return tuple((seed, index, 16 + index, 80.0 + seed + index) for index in range(length))

    def replay_prefix(seed: int, length: int) -> tuple[tuple[int, int, tuple[int, int, int, float]], ...]:
        return tuple(
            (seed, index, (seed, index, 16 + index, 80.0 + seed + index))
            for index in range(length)
        )

    def state_with_key(
        *,
        seed: int,
        objective: float,
        section_count: int,
        alias_switch_count: int,
        max_boundary_displacement_ms: float,
        origin_time_ms: float,
        edge_tuples: tuple[tuple[int, int, int, float], ...],
        replay_key: tuple[tuple[int, int, tuple[int, int, int, float]], ...],
    ):
        return module._State(
            anchor=left_anchor,
            section_count=section_count,
            beat_at_anchor=section_count * 16,
            prev_bpm=120.0 + seed,
            prev_alias_family=120.0,
            downbeat_phase=None,
            origin_time_ms=origin_time_ms,
            duration_objective=objective,
            transition_objective=0.0,
            objective=objective,
            alias_switch_count=alias_switch_count,
            max_boundary_displacement_ms=max_boundary_displacement_ms,
            sections=(),
            edge_tuples=edge_tuples,
            replay_key=replay_key,
        )

    for seed in range(300):
        parent_edge_len = seed % 5
        parent_replay_len = (seed * 2) % 5
        parent = state_with_key(
            seed=seed,
            objective=float((seed % 7) / 10.0),
            section_count=parent_edge_len,
            alias_switch_count=seed % 4,
            max_boundary_displacement_ms=float((seed % 11) * 0.5),
            origin_time_ms=float((seed % 13) * 10.0),
            edge_tuples=edge_prefix(seed, parent_edge_len),
            replay_key=replay_prefix(seed, parent_replay_len),
        )
        edge_tuple = (seed + 1, seed + 2, 20 + (seed % 9), 90.0 + (seed % 17))
        replay_item = (seed + 3, seed % 16, edge_tuple)
        candidate_kwargs = {
            "parent": parent,
            "anchor": right_anchor,
            "section_count": 1 + (seed % 6),
            "beat_at_anchor": 32 + seed,
            "bpm": 100.0 + (seed % 19),
            "first_start_beat": seed % 3,
            "duration_objective": float((seed % 17) / 8.0),
            "transition_objective": float(((seed * 3) % 19) / 9.0),
            "objective": float((seed % 17) / 8.0 + ((seed * 3) % 19) / 9.0),
            "alias_switch_count": seed % 5,
            "max_boundary_displacement_ms": float((seed % 23) * 0.25),
            "edge_tuple": edge_tuple,
            "replay_item": replay_item,
            "context": context,
        }
        candidate_state = module._materialize_interior_state(**candidate_kwargs)
        selector = seed % 8
        if selector == 0:
            right_state = replace(candidate_state)
        else:
            right_state = state_with_key(
                seed=seed + 1000,
                objective=(
                    candidate_state.objective + (0.5 if selector == 1 else 0.0)
                ),
                section_count=(
                    candidate_state.section_count + (1 if selector == 2 else 0)
                ),
                alias_switch_count=(
                    candidate_state.alias_switch_count + (1 if selector == 3 else 0)
                ),
                max_boundary_displacement_ms=(
                    candidate_state.max_boundary_displacement_ms + (0.5 if selector == 4 else 0.0)
                ),
                origin_time_ms=(
                    candidate_state.origin_time_ms + (10.0 if selector == 5 else 0.0)
                ),
                edge_tuples=(
                    candidate_state.edge_tuples[:-1]
                    if selector == 6 and len(candidate_state.edge_tuples) > 1
                    else candidate_state.edge_tuples + ((9999, seed, 1, 222.0),)
                    if selector == 6
                    else candidate_state.edge_tuples
                ),
                replay_key=(
                    candidate_state.replay_key[:-1]
                    if selector == 7 and len(candidate_state.replay_key) > 1
                    else candidate_state.replay_key + ((9999, seed, (9999, seed, 1, 222.0)),)
                    if selector == 7
                    else candidate_state.replay_key
                ),
            )
        right_key_parts = module._state_order_key_parts(right_state)
        expected = module._state_order_key(candidate_state) < module._state_order_key(right_state)
        actual = module._interior_state_key_less_than(
            parent=parent,
            objective=candidate_state.objective,
            section_count=candidate_state.section_count,
            alias_switch_count=candidate_state.alias_switch_count,
            max_boundary_displacement_ms=candidate_state.max_boundary_displacement_ms,
            edge_tuple=edge_tuple,
            replay_item=replay_item,
            right_key=right_key_parts,
        )

        assert math.isfinite(candidate_state.objective)
        assert math.isfinite(candidate_state.max_boundary_displacement_ms)
        assert math.isfinite(candidate_state.origin_time_ms)
        assert actual == expected
        assert (
            module._pending_state_order_key(
                parent,
                section_count=candidate_state.section_count,
                objective=candidate_state.objective,
                alias_switch_count=candidate_state.alias_switch_count,
                max_boundary_displacement_ms=candidate_state.max_boundary_displacement_ms,
                edge_tuple=edge_tuple,
                replay_item=replay_item,
            )
            < right_key_parts
        ) == expected
        if selector == 0:
            assert not actual
            assert module._state_order_key(candidate_state) == module._state_order_key(right_state)


def test_pre_admission_bisect_positions_and_equal_full_bucket_rejects_stably(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    prediction = _constant_prediction(duration_ms=20_000.0, bpm=120.0, downbeat_phase=None)
    candidate_set = _candidate_set_for_prediction(
        prediction,
        beat_peaks=(),
        downbeat_peaks=(),
        tempo_candidates=(module.TempoCandidate(bpm=120.0, source="test", score=1.0),),
        origin_candidates=(module.OriginCandidate(anchor_id=0, time_ms=0.0, bpm=120.0, score=1.0),),
        boundary_candidates=(),
    )
    context = module._SearchContext(
        prediction=prediction,
        candidates=candidate_set,
        constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
        attempt_cap=GLOBAL_CONSTANT_JUMP_CONSTANTS.default_edge_attempt_cap,
    )
    parent = module._initial_state(module._origin_anchors(candidate_set)[0], downbeat_phase=None)
    right_anchor = module._Anchor(anchor_id=1, kind="boundary", time_ms=10_000.0, rank_score=1.0)
    bucket_key = (right_anchor.anchor_id, 1)
    buckets: dict[tuple[int, int], list] = {}
    for index in range(GLOBAL_CONSTANT_JUMP_CONSTANTS.beam_width):
        state = replace(
            parent,
            anchor=right_anchor,
            section_count=1,
            beat_at_anchor=16 + index,
            prev_bpm=120.0,
            prev_alias_family=120.0,
            objective=float(index + 1),
            edge_tuples=((0, 1, index, 120.0),),
            replay_key=((1, index, (0, 1, index, 120.0)),),
        )
        module._insert_state_into_beam_bucket(buckets, bucket_key, state, context)

    def position_for(objective: float) -> int:
        edge_tuple = (0, 1, 999, 120.0)
        replay_item = (1, 999, edge_tuple)
        return module._beam_bucket_bisect_right_interior(
            buckets[bucket_key],
            parent=parent,
            objective=objective,
            section_count=1,
            alias_switch_count=0,
            max_boundary_displacement_ms=0.0,
            edge_tuple=edge_tuple,
            replay_item=replay_item,
        )

    assert position_for(0.5) == 0
    assert position_for(63.5) == 63
    assert position_for(65.0) == 64

    equal_context = module._SearchContext(
        prediction=prediction,
        candidates=candidate_set,
        constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
        attempt_cap=GLOBAL_CONSTANT_JUMP_CONSTANTS.default_edge_attempt_cap,
    )
    equal_buckets: dict[tuple[int, int], list] = {}
    edge_tuple = (0, 500, 20, 120.0)
    replay_item = (right_anchor.anchor_id, 0, edge_tuple)
    equal_state = module._materialize_interior_state(
        parent=parent,
        anchor=right_anchor,
        section_count=1,
        beat_at_anchor=20,
        bpm=120.0,
        first_start_beat=0,
        duration_objective=0.0,
        transition_objective=0.0,
        objective=0.0,
        alias_switch_count=0,
        max_boundary_displacement_ms=0.0,
        edge_tuple=edge_tuple,
        replay_item=replay_item,
        context=equal_context,
    )
    for _ in range(GLOBAL_CONSTANT_JUMP_CONSTANTS.beam_width):
        module._insert_state_into_beam_bucket(equal_buckets, bucket_key, equal_state, equal_context)
    before_order = [id(state) for _, state in equal_buckets[bucket_key]]

    def zero_section_score(*args, **kwargs):
        return module._SectionScore(
            valid=True,
            cost=0.0,
            beat_support_cost=0.0,
            peak_recall_precision_cost=0.0,
            downbeat_phase_cost=0.0,
            bpm_prior_cost=0.0,
            beat_count_prior_cost=0.0,
            section_duration_cost=0.0,
        )

    def zero_transition_score(*args, **kwargs):
        return module._TransitionScore(
            cost=0.0,
            alias_switch_cost=0.0,
            alias_switch_increment=0,
            jump_size_cost=0.0,
            boundary_support_cost=0.0,
        )

    monkeypatch.setattr(module, "_section_score", zero_section_score)
    monkeypatch.setattr(module, "_transition_score", zero_transition_score)
    module._advance_state_into_beam_bucket(
        equal_buckets,
        parent,
        right_anchor,
        module._BeatCountCandidate(count=20, bpm=120.0, rank=0, tempo_distance=0.0, beat_support=1.0),
        context=equal_context,
        variant=VARIANT_CJ1,
    )

    assert [id(state) for _, state in equal_buckets[bucket_key]] == before_order
    assert equal_context.beam_pruned_state_count == 1
    assert len(equal_buckets[bucket_key]) == GLOBAL_CONSTANT_JUMP_CONSTANTS.beam_width


def test_small_dense_reference_assembler_matches_optimized_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    prediction = _jump_prediction(first_bpm=120.0, second_bpm=150.0, boundary_ms=12_000.0, duration_ms=36_000.0)
    candidates = extract_global_constant_jump_candidates(prediction)
    optimized = fit_global_constant_jump(prediction, variant=VARIANT_CJ2, candidate_set=candidates)

    monkeypatch.setattr(module, "_assemble_beam", _legacy_reference_assemble_beam)
    legacy = fit_global_constant_jump(prediction, variant=VARIANT_CJ2, candidate_set=candidates)

    assert legacy.reason == optimized.reason
    assert legacy.diagnostics.section_attempt_count == optimized.diagnostics.section_attempt_count
    assert legacy.diagnostics.edge_count_cache_size == optimized.diagnostics.edge_count_cache_size
    assert legacy.diagnostics.beam_pruned_state_count == optimized.diagnostics.beam_pruned_state_count
    assert legacy.diagnostics.replay_fingerprint == optimized.diagnostics.replay_fingerprint
    assert legacy.diagnostics.objective == optimized.diagnostics.objective
    assert (None if legacy.grid is None else legacy.grid.to_dict()) == (
        None if optimized.grid is None else optimized.grid.to_dict()
    )


def test_pruned_dense_reference_assembler_matches_optimized_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    prediction = _constant_prediction(duration_ms=60_000.0, bpm=120.0, downbeat_phase=None)
    boundary_time_ms = 10_000.0
    beat_peaks = (
        MaterializedPeak(
            frame_index=int(round(boundary_time_ms * prediction.frame_rate_hz / 1000.0)),
            refined_frame=boundary_time_ms * prediction.frame_rate_hz / 1000.0,
            time_ms=boundary_time_ms,
            confidence=1.0,
        ),
    )
    boundary = BoundaryCandidate(
        anchor_id=0,
        time_ms=boundary_time_ms,
        source_peak_index=0,
        source_peak_time_ms=boundary_time_ms,
        source_peak_confidence=1.0,
        rank_score=1.0,
        evidence_mode="ordinary",
        left_period_ms=500.0,
        right_period_ms=500.0,
        ordinary_score=1.0,
        super_score=None,
        downbeat_bonus=0.0,
        nearest_downbeat_distance_ms=None,
    )
    candidates = _candidate_set_for_prediction(
        prediction,
        beat_peaks=beat_peaks,
        downbeat_peaks=(),
        tempo_candidates=tuple(
            module.TempoCandidate(bpm=110.0 + index, source="stress", score=1.0)
            for index in range(4)
        ),
        origin_candidates=tuple(
            module.OriginCandidate(anchor_id=index, time_ms=float(index * 20.0), bpm=120.0, score=1.0)
            for index in range(8)
        ),
        boundary_candidates=(boundary,),
    )

    def cheap_section_score(
        left_anchor,
        *,
        right_anchor,
        left_beat_at_anchor,
        beat_count,
        bpm,
        downbeat_phase,
        context,
        variant,
    ):
        cache = context.terminal_score_cache if right_anchor is None else context.section_score_cache
        cache_key = module._section_score_cache_key(
            left_anchor,
            right_anchor=right_anchor,
            left_beat_at_anchor=left_beat_at_anchor,
            beat_count=beat_count,
            bpm=bpm,
            downbeat_phase=downbeat_phase,
            variant=variant,
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        context.record_search_trace_event("score_cache_miss", cache_key)
        context.check_attempt(cache_key)
        right_id = -1 if right_anchor is None else right_anchor.anchor_id
        cost = ((left_anchor.anchor_id * 101 + right_id * 17 + int(beat_count) * 7 + int(round(bpm * 10.0))) % 997) / 997.0
        score = module._SectionScore(
            valid=True,
            cost=float(cost),
            beat_support_cost=0.0,
            peak_recall_precision_cost=0.0,
            downbeat_phase_cost=0.0,
            bpm_prior_cost=0.0,
            beat_count_prior_cost=0.0,
            section_duration_cost=0.0,
        )
        cache[cache_key] = score
        return score

    def cheap_transition_score(state, *, right_bpm, context, variant):
        cache_key = (state.prev_bpm, float(right_bpm), state.anchor.time_ms, variant)
        cached = context.transition_score_cache.get(cache_key)
        if cached is not None:
            return cached
        cost = 0.0 if state.prev_bpm is None else abs(float(right_bpm) - float(state.prev_bpm)) / 1000.0
        score = module._TransitionScore(
            cost=float(cost),
            alias_switch_cost=0.0,
            alias_switch_increment=0,
            jump_size_cost=0.0,
            boundary_support_cost=0.0,
        )
        context.transition_score_cache[cache_key] = score
        return score

    def dense_counts(left_anchor, right_anchor, context):
        context.count_cache_visited_keys.add((left_anchor.anchor_id, right_anchor.anchor_id))
        return tuple(
            module._BeatCountCandidate(
                count=16 + rank,
                bpm=96.0 + rank * 1.25,
                rank=rank,
                tempo_distance=0.0,
                beat_support=1.0,
            )
            for rank in range(GLOBAL_CONSTANT_JUMP_CONSTANTS.max_beat_count_candidates_per_edge)
        )

    monkeypatch.setattr(module, "_section_score", cheap_section_score)
    monkeypatch.setattr(module, "_transition_score", cheap_transition_score)
    monkeypatch.setattr(module, "_beat_count_candidates", dense_counts)

    optimized = fit_global_constant_jump(prediction, variant=VARIANT_CJ1, candidate_set=candidates)

    monkeypatch.setattr(module, "_assemble_beam", _legacy_reference_assemble_beam)
    legacy = fit_global_constant_jump(prediction, variant=VARIANT_CJ1, candidate_set=candidates)

    assert optimized.diagnostics.beam_pruned_state_count > 0
    assert legacy.reason == optimized.reason
    assert legacy.diagnostics.section_attempt_count == optimized.diagnostics.section_attempt_count
    assert legacy.diagnostics.edge_count_cache_size == optimized.diagnostics.edge_count_cache_size
    assert legacy.diagnostics.section_score_cache_size == optimized.diagnostics.section_score_cache_size
    assert legacy.diagnostics.beam_pruned_state_count == optimized.diagnostics.beam_pruned_state_count
    assert legacy.diagnostics.replay_fingerprint == optimized.diagnostics.replay_fingerprint
    assert legacy.diagnostics.objective == optimized.diagnostics.objective
    assert (None if legacy.grid is None else legacy.grid.to_dict()) == (
        None if optimized.grid is None else optimized.grid.to_dict()
    )


@pytest.mark.parametrize("variant", [VARIANT_CJ1, VARIANT_CJ2, VARIANT_CJ3])
def test_interior_score_bundle_enabled_matches_disabled_public_result(
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    prediction = _jump_prediction(
        first_bpm=120.0,
        second_bpm=150.0,
        boundary_ms=12_000.0,
        duration_ms=24_000.0,
    )
    candidates = extract_global_constant_jump_candidates(prediction)

    enabled = _fit_with_forced_bundle_enabled(
        monkeypatch,
        module,
        prediction,
        candidates,
        variant=variant,
        enabled=True,
    )
    disabled = _fit_with_forced_bundle_enabled(
        monkeypatch,
        module,
        prediction,
        candidates,
        variant=variant,
        enabled=False,
    )

    _assert_public_result_payload_exact(enabled, disabled)


def test_interior_score_bundle_key_reuses_only_frozen_geometry_classes() -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    context, candidates = _constant_scoring_context(duration_ms=20_000.0, bpm=120.0)
    del context
    origin = module._origin_anchors(candidates)[0]
    base = module._initial_state(origin, downbeat_phase=0)

    for variant in (VARIANT_CJ1, VARIANT_CJ2):
        assert module._interior_edge_score_bundle_key(base, variant) == module._interior_edge_score_bundle_key(
            replace(base, beat_at_anchor=37),
            variant,
        )

    cj3_base = module._interior_edge_score_bundle_key(base, VARIANT_CJ3)
    assert cj3_base == module._interior_edge_score_bundle_key(replace(base, beat_at_anchor=4), VARIANT_CJ3)
    assert cj3_base != module._interior_edge_score_bundle_key(replace(base, beat_at_anchor=1), VARIANT_CJ3)

    residue_keys = {
        module._interior_edge_score_bundle_key(replace(base, beat_at_anchor=beat), VARIANT_CJ3)
        for beat in range(4)
    }
    none_key = module._interior_edge_score_bundle_key(replace(base, downbeat_phase=None), VARIANT_CJ3)

    assert len(residue_keys) == 4
    assert none_key not in residue_keys
    assert len(residue_keys | {none_key}) == 5


def test_cj1_monotone_rejection_skips_only_strictly_worse_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    context, candidates = _constant_scoring_context(duration_ms=20_000.0, bpm=120.0)
    parent = module._initial_state(module._origin_anchors(candidates)[0], downbeat_phase=None)
    right_anchor = module._Anchor(anchor_id=1, kind="boundary", time_ms=10_000.0, rank_score=1.0)
    count_candidate = module._BeatCountCandidate(
        count=20,
        bpm=120.0,
        rank=0,
        tempo_distance=0.0,
        beat_support=1.0,
    )
    score = module._SectionScore(
        valid=True,
        cost=0.0,
        beat_support_cost=0.0,
        peak_recall_precision_cost=0.0,
        downbeat_phase_cost=0.0,
        bpm_prior_cost=0.0,
        beat_count_prior_cost=0.0,
        section_duration_cost=0.0,
    )
    context.interior_edge_score_bundle_cache[module._interior_edge_score_bundle_key(parent, VARIANT_CJ1)] = (
        (right_anchor, count_candidate, score),
    )
    bucket_key = (right_anchor.anchor_id, 1)
    buckets: dict[tuple[int, int], list] = {}
    for index in range(GLOBAL_CONSTANT_JUMP_CONSTANTS.beam_width):
        retained = replace(
            parent,
            anchor=right_anchor,
            section_count=1,
            objective=float(index),
            duration_objective=float(index),
            edge_tuples=((0, 500, index + 1, 120.0),),
            replay_key=((right_anchor.anchor_id, index, (0, 500, index + 1, 120.0)),),
        )
        module._insert_state_into_beam_bucket(buckets, bucket_key, retained, context)

    transition_calls = []

    def zero_transition(state, *, right_bpm, context, variant):
        transition_calls.append(state.objective)
        return module._TransitionScore(
            cost=0.0,
            alias_switch_cost=0.0,
            alias_switch_increment=0,
            jump_size_cost=0.0,
            boundary_support_cost=0.0,
        )

    monkeypatch.setattr(module, "_transition_score", zero_transition)
    rejected_objectives: dict[int, float] = {}
    for objective in (100.0, 100.0, 101.0):
        state = replace(parent, objective=objective, duration_objective=objective)
        module._advance_state_edges_into_beam_bucket(
            buckets,
            state,
            (right_anchor,),
            context=context,
            variant=VARIANT_CJ1,
            cj1_rejected_objectives=rejected_objectives,
        )

    assert transition_calls == [100.0, 100.0]
    assert rejected_objectives == {0: 100.0}
    assert context.beam_pruned_state_count == 3
    assert [state.objective for _, state in buckets[bucket_key]] == [float(index) for index in range(64)]

def test_interior_score_bundle_reuses_invalid_and_empty_edges_without_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    empty_context, empty_candidates = _constant_scoring_context(duration_ms=20_000.0, bpm=120.0)
    empty_state = module._initial_state(module._origin_anchors(empty_candidates)[0], downbeat_phase=None)
    outgoing_calls = 0

    def empty_outgoing(state, boundary_anchors, context):
        nonlocal outgoing_calls
        outgoing_calls += 1
        return ()

    def forbidden_transition(*args, **kwargs):
        raise AssertionError("empty or invalid bundle entries must not transition")

    monkeypatch.setattr(module, "_outgoing_boundary_anchors", empty_outgoing)
    monkeypatch.setattr(module, "_transition_score", forbidden_transition)
    module._advance_state_edges_into_beam_bucket(
        {},
        empty_state,
        (),
        context=empty_context,
        variant=VARIANT_CJ1,
    )
    assert outgoing_calls == 1
    assert len(empty_context.interior_edge_score_bundle_cache) == 1

    def forbidden_outgoing(*args, **kwargs):
        raise AssertionError("empty bundle should be reused before outgoing scan")

    monkeypatch.setattr(module, "_outgoing_boundary_anchors", forbidden_outgoing)
    module._advance_state_edges_into_beam_bucket(
        {},
        empty_state,
        (),
        context=empty_context,
        variant=VARIANT_CJ1,
    )
    assert outgoing_calls == 1

    invalid_context, invalid_candidates = _constant_scoring_context(duration_ms=20_000.0, bpm=120.0)
    invalid_state = module._initial_state(module._origin_anchors(invalid_candidates)[0], downbeat_phase=None)
    right_anchor = module._Anchor(anchor_id=1, kind="boundary", time_ms=10_000.0, rank_score=1.0)
    invalid_score_calls = 0

    monkeypatch.setattr(module, "_outgoing_boundary_anchors", lambda state, boundary_anchors, context: (right_anchor,))
    monkeypatch.setattr(
        module,
        "_beat_count_candidates",
        lambda left_anchor, right_anchor, context: (
            module._BeatCountCandidate(count=20, bpm=120.0, rank=0, tempo_distance=0.0, beat_support=1.0),
        ),
    )

    def invalid_score(*args, **kwargs):
        nonlocal invalid_score_calls
        invalid_score_calls += 1
        return module._invalid_section_score()

    monkeypatch.setattr(module, "_section_score", invalid_score)
    invalid_buckets: dict[tuple[int, int], list] = {}
    module._advance_state_edges_into_beam_bucket(
        invalid_buckets,
        invalid_state,
        (right_anchor,),
        context=invalid_context,
        variant=VARIANT_CJ1,
    )
    assert invalid_score_calls == 1
    assert invalid_buckets == {}
    assert len(invalid_context.interior_edge_score_bundle_cache) == 1

    def forbidden_score(*args, **kwargs):
        raise AssertionError("invalid score bundle should be reused before rescoring")

    monkeypatch.setattr(module, "_section_score", forbidden_score)
    module._advance_state_edges_into_beam_bucket(
        invalid_buckets,
        invalid_state,
        (right_anchor,),
        context=invalid_context,
        variant=VARIANT_CJ1,
    )
    assert invalid_score_calls == 1
    assert invalid_buckets == {}


def test_interior_score_bundle_attempt_cap_mid_edge_matches_disabled_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    prediction = _constant_prediction(duration_ms=20_000.0, bpm=120.0, downbeat_phase=None)
    beat_peaks = materialize_global_constant_jump_peaks(prediction.beat_prob, frame_rate_hz=prediction.frame_rate_hz)
    boundary = _boundary_candidate_from_peak(beat_peaks, peak_index=20, anchor_id=0)
    candidates = _candidate_set_for_prediction(
        prediction,
        beat_peaks=beat_peaks,
        downbeat_peaks=(),
        tempo_candidates=(module.TempoCandidate(bpm=120.0, source="test", score=1.0),),
        origin_candidates=(module.OriginCandidate(anchor_id=0, time_ms=0.0, bpm=120.0, score=1.0),),
        boundary_candidates=(boundary,),
    )

    def two_interior_counts(left_anchor, right_anchor, context):
        context.count_cache_visited_keys.add((left_anchor.anchor_id, right_anchor.anchor_id))
        return (
            module._BeatCountCandidate(count=20, bpm=120.0, rank=0, tempo_distance=0.0, beat_support=1.0),
            module._BeatCountCandidate(count=21, bpm=126.0, rank=1, tempo_distance=0.0, beat_support=1.0),
        )

    monkeypatch.setattr(module, "_beat_count_candidates", two_interior_counts)
    enabled = _fit_with_forced_bundle_enabled(
        monkeypatch,
        module,
        prediction,
        candidates,
        variant=VARIANT_CJ1,
        enabled=True,
        attempt_cap=2,
    )
    disabled = _fit_with_forced_bundle_enabled(
        monkeypatch,
        module,
        prediction,
        candidates,
        variant=VARIANT_CJ1,
        enabled=False,
        attempt_cap=2,
    )

    assert enabled.reason == REASON_EDGE_ATTEMPT_CAP_EXCEEDED
    assert disabled.reason == REASON_EDGE_ATTEMPT_CAP_EXCEEDED
    assert enabled.diagnostics.section_attempt_count == 2
    assert disabled.diagnostics.section_attempt_count == 2
    assert enabled.diagnostics.beam_pruned_state_count == disabled.diagnostics.beam_pruned_state_count == 0
    assert enabled.diagnostics.replay_fingerprint == disabled.diagnostics.replay_fingerprint
    _assert_public_result_payload_exact(enabled, disabled)


@pytest.mark.parametrize(
    ("variant", "downbeat_phase", "constant_only", "interior"),
    [
        (VARIANT_CJ1, None, True, False),
        (VARIANT_CJ2, None, False, True),
        (VARIANT_CJ3, 2, True, False),
        (VARIANT_CJ0, 1, False, True),
    ],
)
def test_best_terminal_completion_matches_exhaustive_reference(
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    downbeat_phase: int | None,
    constant_only: bool,
    interior: bool,
) -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    prediction = _constant_prediction(duration_ms=40_000.0, bpm=120.0, downbeat_phase=downbeat_phase)
    beat_peaks = materialize_global_constant_jump_peaks(prediction.beat_prob, frame_rate_hz=prediction.frame_rate_hz)
    boundary = _boundary_candidate_from_peak(beat_peaks, peak_index=20, anchor_id=0)
    candidates = _candidate_set_for_prediction(
        prediction,
        beat_peaks=beat_peaks,
        downbeat_peaks=materialize_global_constant_jump_peaks(
            prediction.downbeat_prob,
            frame_rate_hz=prediction.frame_rate_hz,
        ),
        tempo_candidates=(
            module.TempoCandidate(bpm=100.0, source="tie", score=1.0),
            module.TempoCandidate(bpm=120.0, source="tie", score=1.0),
            module.TempoCandidate(bpm=140.0, source="tie", score=1.0),
        ),
        origin_candidates=(module.OriginCandidate(anchor_id=0, time_ms=0.0, bpm=120.0, score=1.0),),
        boundary_candidates=(boundary,),
    )

    def zero_section_score(
        left_anchor,
        *,
        right_anchor,
        left_beat_at_anchor,
        beat_count,
        bpm,
        downbeat_phase,
        context,
        variant,
    ):
        cache = context.terminal_score_cache if right_anchor is None else context.section_score_cache
        cache_key = module._section_score_cache_key(
            left_anchor,
            right_anchor=right_anchor,
            left_beat_at_anchor=left_beat_at_anchor,
            beat_count=beat_count,
            bpm=bpm,
            downbeat_phase=downbeat_phase,
            variant=variant,
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        context.record_search_trace_event("score_cache_miss", cache_key)
        context.check_attempt(cache_key)
        score = module._SectionScore(
            valid=True,
            cost=0.0,
            beat_support_cost=0.0,
            peak_recall_precision_cost=0.0,
            downbeat_phase_cost=0.0,
            bpm_prior_cost=0.0,
            beat_count_prior_cost=0.0,
            section_duration_cost=0.0,
        )
        cache[cache_key] = score
        return score

    def zero_transition_score(state, *, right_bpm, context, variant):
        cache_key = (state.prev_bpm, float(right_bpm), state.anchor.time_ms, variant)
        cached = context.transition_score_cache.get(cache_key)
        if cached is not None:
            return cached
        score = module._TransitionScore(
            cost=0.0,
            alias_switch_cost=0.0,
            alias_switch_increment=0,
            jump_size_cost=0.0,
            boundary_support_cost=0.0,
        )
        context.transition_score_cache[cache_key] = score
        return score

    monkeypatch.setattr(module, "_section_score", zero_section_score)
    monkeypatch.setattr(module, "_transition_score", zero_transition_score)

    def make_state():
        context = module._SearchContext(
            prediction=prediction,
            candidates=candidates,
            constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
            attempt_cap=GLOBAL_CONSTANT_JUMP_CONSTANTS.default_edge_attempt_cap,
        )
        state = module._initial_state(module._origin_anchors(candidates)[0], downbeat_phase=downbeat_phase)
        if interior:
            state = module._advance_state(
                state,
                module._boundary_anchors(candidates)[0],
                module._BeatCountCandidate(count=20, bpm=120.0, rank=0, tempo_distance=0.0, beat_support=1.0),
                context=context,
                variant=variant,
            )
            assert state is not None
        return context, state

    actual_context, actual_state = make_state()
    actual = module._best_terminal_completion(
        actual_state,
        candidates,
        actual_context,
        variant=variant,
        constant_only=constant_only,
    )

    reference_context, reference_state = make_state()
    terminal_bpms = module._terminal_bpms(
        reference_state,
        candidates,
        reference_context,
        constant_only=constant_only,
    )
    if not constant_only:
        assert len(terminal_bpms) > 1
    reference_candidates = tuple(
        completed
        for rank, bpm in enumerate(terminal_bpms)
        if (
            completed := module._complete_terminal_state(
                reference_state,
                bpm=bpm,
                terminal_rank=rank,
                context=reference_context,
                variant=variant,
            )
        )
        is not None
    )
    expected = min(reference_candidates, key=module._complete_state_order_key)

    assert actual == expected


def test_lazy_beam_stress_uses_dense_candidates_through_public_tagged_path() -> None:
    from pulsefield_model.timing.v3 import global_constant_jump as module

    prediction = _stress_prediction()
    candidates = _stress_candidates(prediction)
    assert float(np.linalg.norm(prediction.beat_prob - float(np.mean(prediction.beat_prob)))) > 0.0
    assert len(candidates.origin_candidates) == GLOBAL_CONSTANT_JUMP_CONSTANTS.max_origin_candidates
    assert len(candidates.tempo_candidates) == GLOBAL_CONSTANT_JUMP_CONSTANTS.max_tempo_candidates_retained
    assert len(candidates.boundary_candidates) == GLOBAL_CONSTANT_JUMP_CONSTANTS.max_interior_boundary_candidates
    assert GLOBAL_CONSTANT_JUMP_CONSTANTS.max_section_count == 20

    boundary_anchors = module._boundary_anchors(candidates)
    context = module._SearchContext(
        prediction=prediction,
        candidates=candidates,
        constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
        attempt_cap=GLOBAL_CONSTANT_JUMP_CONSTANTS.default_edge_attempt_cap,
    )
    retained_counts = module._beat_count_candidates(
        module._origin_anchors(candidates)[0],
        boundary_anchors[0],
        context,
    )
    assert len(retained_counts) == GLOBAL_CONSTANT_JUMP_CONSTANTS.max_beat_count_candidates_per_edge

    started = time.perf_counter()
    result = fit_global_constant_jump(prediction, variant=VARIANT_CJ1, candidate_set=candidates)
    runtime_seconds = time.perf_counter() - started
    replay = fit_global_constant_jump(prediction, variant=VARIANT_CJ1, candidate_set=candidates)

    assert result.ok or result.reason == REASON_EDGE_ATTEMPT_CAP_EXCEEDED
    assert result.diagnostics.section_attempt_count <= GLOBAL_CONSTANT_JUMP_CONSTANTS.default_edge_attempt_cap
    assert result.diagnostics.selected_section_count <= GLOBAL_CONSTANT_JUMP_CONSTANTS.max_section_count
    assert result.diagnostics.edge_count_cache_size >= GLOBAL_CONSTANT_JUMP_CONSTANTS.max_origin_candidates
    _assert_public_result_payload_exact(result, replay)
    assert runtime_seconds < 60.0


class _MetadataTrappingPrediction(FrameTimingPrediction):
    def arm_metadata_traps(self) -> None:
        object.__setattr__(self, "_metadata_traps_armed", True)

    def __getattribute__(self, name: str) -> object:
        if name in {
            "source_path",
            "checkpoint_path",
            "provider",
            "metadata",
            "title",
            "artist",
            "api_bpm",
        }:
            armed = object.__getattribute__(self, "__dict__").get("_metadata_traps_armed", False)
            if armed:
                raise AssertionError(f"unexpected metadata access: {name}")
        return super().__getattribute__(name)


def _metadata_trapping_prediction(*, frame_count: int, frame_rate_hz: float) -> _MetadataTrappingPrediction:
    return _MetadataTrappingPrediction(
        provider="cached-beatthis",
        checkpoint_path="checkpoint.pt",
        source_path="map.osu",
        beat_prob=np.zeros(frame_count, dtype=np.float32),
        downbeat_prob=np.zeros(frame_count, dtype=np.float32),
        frame_rate_hz=frame_rate_hz,
    )


def _metadata_trapping_constant_prediction(
    *,
    duration_ms: float,
    bpm: float,
    frame_rate_hz: float = 50.0,
) -> _MetadataTrappingPrediction:
    base = _constant_prediction(duration_ms=duration_ms, bpm=bpm, frame_rate_hz=frame_rate_hz)
    return _MetadataTrappingPrediction(
        provider="cached-beatthis",
        checkpoint_path="checkpoint.pt",
        source_path="metadata-trap.osu",
        beat_prob=base.beat_prob,
        downbeat_prob=base.downbeat_prob,
        frame_rate_hz=base.frame_rate_hz,
    )


def _constant_prediction(
    *,
    duration_ms: float,
    bpm: float,
    frame_rate_hz: float = 50.0,
    downbeat_phase: int | None = 0,
) -> FrameTimingPrediction:
    frame_count = int(round(duration_ms * frame_rate_hz / 1000.0))
    beat_prob = np.zeros(frame_count, dtype=np.float32)
    downbeat_prob = np.zeros(frame_count, dtype=np.float32)
    period_ms = 60000.0 / bpm
    beat = 0
    time_ms = 0.0
    while time_ms < duration_ms:
        _write_pulse(beat_prob, time_ms=time_ms, frame_rate_hz=frame_rate_hz)
        if downbeat_phase is not None and beat % 4 == downbeat_phase:
            _write_pulse(downbeat_prob, time_ms=time_ms, frame_rate_hz=frame_rate_hz)
        beat += 1
        time_ms += period_ms
    return FrameTimingPrediction(
        provider="cached-beatthis",
        checkpoint_path="checkpoint.pt",
        source_path="song.osu",
        beat_prob=beat_prob,
        downbeat_prob=downbeat_prob,
        frame_rate_hz=frame_rate_hz,
    )


def _jump_prediction(
    *,
    first_bpm: float,
    second_bpm: float,
    boundary_ms: float,
    duration_ms: float,
    frame_rate_hz: float = 50.0,
    include_downbeats: bool = True,
    downbeat_phase: int = 0,
) -> FrameTimingPrediction:
    frame_count = int(round(duration_ms * frame_rate_hz / 1000.0))
    beat_prob = np.zeros(frame_count, dtype=np.float32)
    downbeat_prob = np.zeros(frame_count, dtype=np.float32)
    beat = 0
    time_ms = 0.0
    first_period_ms = 60000.0 / first_bpm
    while time_ms <= boundary_ms + 1e-9:
        _write_pulse(beat_prob, time_ms=time_ms, frame_rate_hz=frame_rate_hz)
        if include_downbeats and beat % 4 == downbeat_phase:
            _write_pulse(downbeat_prob, time_ms=time_ms, frame_rate_hz=frame_rate_hz)
        beat += 1
        time_ms += first_period_ms
    second_period_ms = 60000.0 / second_bpm
    time_ms = boundary_ms + second_period_ms
    while time_ms < duration_ms:
        _write_pulse(beat_prob, time_ms=time_ms, frame_rate_hz=frame_rate_hz)
        if include_downbeats and beat % 4 == downbeat_phase:
            _write_pulse(downbeat_prob, time_ms=time_ms, frame_rate_hz=frame_rate_hz)
        beat += 1
        time_ms += second_period_ms
    return FrameTimingPrediction(
        provider="cached-beatthis",
        checkpoint_path="checkpoint.pt",
        source_path="jump.osu",
        beat_prob=beat_prob,
        downbeat_prob=downbeat_prob,
        frame_rate_hz=frame_rate_hz,
    )


def _body_only_prediction(
    *,
    duration_ms: float,
    bpm: float,
    origin_time_ms: float,
    frame_rate_hz: float = 50.0,
) -> FrameTimingPrediction:
    frame_count = int(round(duration_ms * frame_rate_hz / 1000.0))
    beat_prob = np.zeros(frame_count, dtype=np.float32)
    period_ms = 60000.0 / bpm
    time_ms = origin_time_ms
    while time_ms < duration_ms:
        _write_pulse(beat_prob, time_ms=time_ms, frame_rate_hz=frame_rate_hz)
        time_ms += period_ms
    origin_frame = int(round(origin_time_ms * frame_rate_hz / 1000.0))
    beat_prob[:origin_frame] = 0.0
    return FrameTimingPrediction(
        provider="cached-beatthis",
        checkpoint_path="checkpoint.pt",
        source_path="body-only.osu",
        beat_prob=beat_prob,
        downbeat_prob=np.zeros(frame_count, dtype=np.float32),
        frame_rate_hz=frame_rate_hz,
    )


def _write_pulse(
    signal: np.ndarray,
    *,
    time_ms: float,
    frame_rate_hz: float,
) -> None:
    frame = int(round(time_ms * frame_rate_hz / 1000.0))
    if 0 <= frame < signal.shape[0]:
        signal[frame] = max(signal[frame], np.float32(1.0))
    if 0 <= frame - 1 < signal.shape[0]:
        signal[frame - 1] = max(signal[frame - 1], np.float32(0.1))
    if 0 <= frame + 1 < signal.shape[0]:
        signal[frame + 1] = max(signal[frame + 1], np.float32(0.1))


def _boundary_candidate_from_peak(
    beat_peaks: tuple[MaterializedPeak, ...],
    *,
    peak_index: int,
    anchor_id: int,
) -> BoundaryCandidate:
    peak = beat_peaks[peak_index]
    return BoundaryCandidate(
        anchor_id=anchor_id,
        time_ms=peak.time_ms,
        source_peak_index=peak_index,
        source_peak_time_ms=peak.time_ms,
        source_peak_confidence=peak.confidence,
        rank_score=1.0,
        evidence_mode="ordinary",
        left_period_ms=500.0,
        right_period_ms=500.0,
        ordinary_score=1.0,
        super_score=None,
        downbeat_bonus=0.0,
        nearest_downbeat_distance_ms=None,
    )


def _constant_scoring_context(
    *,
    duration_ms: float,
    bpm: float,
):
    from pulsefield_model.timing.v3 import global_constant_jump as module

    prediction = _constant_prediction(duration_ms=duration_ms, bpm=bpm)
    candidates = _single_origin_candidate_set(prediction, origin_time_ms=0.0, bpm=bpm)
    context = module._SearchContext(
        prediction=prediction,
        candidates=candidates,
        constants=GLOBAL_CONSTANT_JUMP_CONSTANTS,
        attempt_cap=GLOBAL_CONSTANT_JUMP_CONSTANTS.default_edge_attempt_cap,
    )
    return context, candidates


def _assert_public_result_payload_exact(left, right) -> None:
    assert left.reason == right.reason
    assert left.diagnostics.section_attempt_count == right.diagnostics.section_attempt_count
    assert left.diagnostics.edge_count_cache_size == right.diagnostics.edge_count_cache_size
    assert left.diagnostics.section_score_cache_size == right.diagnostics.section_score_cache_size
    assert left.diagnostics.beam_pruned_state_count == right.diagnostics.beam_pruned_state_count
    assert left.diagnostics.selected_section_count == right.diagnostics.selected_section_count
    assert left.diagnostics.selected_origin_time_ms == right.diagnostics.selected_origin_time_ms
    assert left.diagnostics.selected_downbeat_phase == right.diagnostics.selected_downbeat_phase
    assert left.diagnostics.objective == right.diagnostics.objective
    assert left.diagnostics.duration_objective == right.diagnostics.duration_objective
    assert left.diagnostics.transition_objective == right.diagnostics.transition_objective
    assert left.diagnostics.alias_switch_count == right.diagnostics.alias_switch_count
    assert left.diagnostics.max_boundary_displacement_ms == right.diagnostics.max_boundary_displacement_ms
    assert left.diagnostics.replay_fingerprint == right.diagnostics.replay_fingerprint
    assert left.diagnostics.grid_fingerprint == right.diagnostics.grid_fingerprint
    assert (None if left.grid is None else left.grid.to_dict()) == (
        None if right.grid is None else right.grid.to_dict()
    )


def _fit_with_forced_bundle_enabled(
    monkeypatch: pytest.MonkeyPatch,
    module,
    prediction: FrameTimingPrediction,
    candidates,
    *,
    variant: str,
    enabled: bool,
    attempt_cap: int = GLOBAL_CONSTANT_JUMP_CONSTANTS.default_edge_attempt_cap,
):
    original_init = module._SearchContext.__init__

    def init_with_bundle_flag(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.interior_edge_score_bundle_enabled = enabled

    with monkeypatch.context() as scoped:
        scoped.setattr(module._SearchContext, "__init__", init_with_bundle_flag)
        return module.fit_global_constant_jump(
            prediction,
            variant=variant,
            attempt_cap=attempt_cap,
            candidate_set=candidates,
        )


def _legacy_reference_assemble_beam(candidates, context, *, variant):
    from pulsefield_model.timing.v3 import global_constant_jump as module

    origin_anchors = module._origin_anchors(candidates)
    boundary_anchors = module._boundary_anchors(candidates)
    states_by_bucket: dict[tuple[int, int], list] = {}
    for origin_anchor in origin_anchors:
        for downbeat_phase in module._origin_downbeat_phases(context, variant):
            state = module._initial_state(origin_anchor, downbeat_phase=downbeat_phase)
            states_by_bucket.setdefault((origin_anchor.anchor_id, 0), []).append(state)
            states_by_bucket[(origin_anchor.anchor_id, 0)].sort(key=module._state_order_key)

    complete_states: list = []
    anchor_order = tuple(sorted(origin_anchors + boundary_anchors, key=lambda anchor: anchor.anchor_id))
    for anchor in anchor_order:
        for section_count in range(context.constants.max_section_count):
            current_states = tuple(
                sorted(
                    states_by_bucket.pop((anchor.anchor_id, section_count), ()),
                    key=module._state_order_key,
                )
            )
            for state in current_states:
                for completed in module._terminal_completions(state, candidates, context, variant=variant):
                    complete_states.append(completed)
                if state.section_count + 1 >= context.constants.max_section_count:
                    continue
                for right_anchor in module._outgoing_boundary_anchors(state, boundary_anchors, context):
                    for count_candidate in module._beat_count_candidates(state.anchor, right_anchor, context):
                        next_state = module._advance_state(
                            state,
                            right_anchor,
                            count_candidate,
                            context=context,
                            variant=variant,
                        )
                        if next_state is not None:
                            bucket_key = (right_anchor.anchor_id, next_state.section_count)
                            bucket = states_by_bucket.setdefault(bucket_key, [])
                            bucket.append(next_state)
                            bucket.sort(key=module._state_order_key)
                            if len(bucket) > context.constants.beam_width:
                                del bucket[context.constants.beam_width :]
                                context.beam_pruned_state_count += 1
    if not complete_states:
        return None
    return min(complete_states, key=module._complete_state_order_key)


def _single_origin_candidate_set(
    prediction: FrameTimingPrediction,
    *,
    origin_time_ms: float,
    bpm: float,
):
    from pulsefield_model.timing.v3 import global_constant_jump as module

    beat_peaks = materialize_global_constant_jump_peaks(prediction.beat_prob, frame_rate_hz=prediction.frame_rate_hz)
    return _candidate_set_for_prediction(
        prediction,
        beat_peaks=beat_peaks,
        downbeat_peaks=(),
        tempo_candidates=(module.TempoCandidate(bpm=bpm, source="test", score=1.0),),
        origin_candidates=(module.OriginCandidate(anchor_id=0, time_ms=origin_time_ms, bpm=bpm, score=1.0),),
        boundary_candidates=(),
    )


def _candidate_set_for_prediction(
    prediction: FrameTimingPrediction,
    *,
    beat_peaks: tuple[MaterializedPeak, ...],
    downbeat_peaks: tuple[MaterializedPeak, ...],
    tempo_candidates,
    origin_candidates,
    boundary_candidates,
):
    from pulsefield_model.timing.v3 import global_constant_jump as module

    input_signal_sha256 = module._input_signal_sha256(
        np.asarray(prediction.beat_prob, dtype=np.float64),
        np.asarray(prediction.downbeat_prob, dtype=np.float64),
    )
    candidate_fingerprint = module._candidate_fingerprint(
        tempo_candidates=tuple(tempo_candidates),
        origin_candidates=tuple(origin_candidates),
        boundary_candidates=tuple(boundary_candidates),
        beat_peaks=tuple(beat_peaks),
        downbeat_peaks=tuple(downbeat_peaks),
        input_signal_sha256=input_signal_sha256,
    )
    from pulsefield_model.timing.v3.global_constant_jump import (
        GlobalConstantJumpCandidateDiagnostics,
        GlobalConstantJumpCandidateSet,
    )

    frame_count, frame_rate_hz, coverage_start_ms, coverage_end_ms, min_period_frames, max_period_frames = (
        module._prediction_geometry(
            np.asarray(prediction.beat_prob, dtype=np.float64),
            prediction.frame_rate_hz,
            GLOBAL_CONSTANT_JUMP_CONSTANTS,
        )
    )
    diagnostics = GlobalConstantJumpCandidateDiagnostics(
        candidate_contract_version=CANDIDATE_CONTRACT_VERSION,
        constants_json_sha256=GLOBAL_CONSTANT_JUMP_CONSTANTS_JSON_SHA256,
        pulse_correlation_version=PULSE_CORRELATION_VERSION,
        boundary_candidate_score_version=BOUNDARY_CANDIDATE_SCORE_VERSION,
        frame_count=frame_count,
        frame_rate_hz=frame_rate_hz,
        coverage_start_ms=coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
        min_period_frames=min_period_frames,
        max_period_frames=max_period_frames,
        beat_peak_count=len(beat_peaks),
        downbeat_peak_count=len(downbeat_peaks),
        tempo_candidate_count=len(tempo_candidates),
        origin_candidate_count=len(origin_candidates),
        boundary_candidate_count=len(boundary_candidates),
        input_signal_sha256=input_signal_sha256,
        candidate_fingerprint=candidate_fingerprint,
    )
    return GlobalConstantJumpCandidateSet(
        beat_peaks=tuple(beat_peaks),
        downbeat_peaks=tuple(downbeat_peaks),
        tempo_candidates=tuple(tempo_candidates),
        origin_candidates=tuple(origin_candidates),
        boundary_candidates=tuple(boundary_candidates),
        diagnostics=diagnostics,
    )


def _stress_candidates(prediction: FrameTimingPrediction):
    from pulsefield_model.timing.v3 import global_constant_jump as module

    boundary_spacing_ms = GLOBAL_CONSTANT_JUMP_CONSTANTS.boundary_merge_ms + 100.0
    boundary_times_ms = tuple(
        10_000.0 + index * boundary_spacing_ms
        for index in range(GLOBAL_CONSTANT_JUMP_CONSTANTS.max_interior_boundary_candidates)
    )
    beat_peaks = tuple(
        MaterializedPeak(
            frame_index=int(round(time_ms * prediction.frame_rate_hz / 1000.0)),
            refined_frame=time_ms * prediction.frame_rate_hz / 1000.0,
            time_ms=time_ms,
            confidence=1.0,
        )
        for time_ms in boundary_times_ms
    )
    tempo_candidates = tuple(
        module.TempoCandidate(bpm=80.0 + index * 2.5, source="stress", score=1.0)
        for index in range(GLOBAL_CONSTANT_JUMP_CONSTANTS.max_tempo_candidates_retained)
    )
    origin_candidates = tuple(
        module.OriginCandidate(anchor_id=index, time_ms=float(index * 20.0), bpm=120.0, score=1.0 - index * 0.001)
        for index in range(16)
    )
    boundary_candidates = tuple(
        BoundaryCandidate(
            anchor_id=index,
            time_ms=boundary_times_ms[index],
            source_peak_index=index,
            source_peak_time_ms=boundary_times_ms[index],
            source_peak_confidence=1.0,
            rank_score=1.0,
            evidence_mode="ordinary",
            left_period_ms=500.0,
            right_period_ms=500.0,
            ordinary_score=1.0,
            super_score=None,
            downbeat_bonus=0.0,
            nearest_downbeat_distance_ms=None,
        )
        for index in range(192)
    )
    return _candidate_set_for_prediction(
        prediction,
        beat_peaks=beat_peaks,
        downbeat_peaks=(),
        tempo_candidates=tempo_candidates,
        origin_candidates=origin_candidates,
        boundary_candidates=boundary_candidates,
    )


def _stress_prediction() -> FrameTimingPrediction:
    frame_rate_hz = 2.0
    duration_ms = 1_610_000.0
    frame_count = int(duration_ms * frame_rate_hz / 1000.0)
    frames = np.arange(frame_count, dtype=np.float64)
    beat_prob = (0.35 + 0.25 * np.sin(frames * 0.071) + 0.15 * np.cos(frames * 0.173)).astype(np.float32)
    beat_prob = np.clip(beat_prob, 0.0, 1.0)
    return FrameTimingPrediction(
        provider="cached-beatthis",
        checkpoint_path="checkpoint.pt",
        source_path="stress.osu",
        beat_prob=beat_prob,
        downbeat_prob=np.zeros(frame_count, dtype=np.float32),
        frame_rate_hz=frame_rate_hz,
    )
