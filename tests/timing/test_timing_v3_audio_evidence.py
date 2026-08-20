from __future__ import annotations

import math

import numpy as np
import pytest

from pulsefield_model.timing.v3.analytic_curve import (
    ConstantTempoSection,
    LinearTimeRampSection,
    PhaseContinuousTimingCurve,
)
from pulsefield_model.timing.v3.audio_evidence import (
    DEFAULT_RAW_AUDIO_EVIDENCE_CONFIG,
    RAW_AUDIO_EVIDENCE_UNAVAILABLE_FEWER_THAN_16_COMMON_BEATS,
    RAW_AUDIO_EVIDENCE_UNAVAILABLE_NO_COMPLETE_WINDOW,
    RAW_AUDIO_EVIDENCE_UNAVAILABLE_RETAINED_DOMAIN_RATIO_BELOW_90,
    RawAudioEvidence,
    RawAudioEvidenceConfig,
    _inclusive_event_windows,
    extract_audio_evidence,
    extract_raw_audio_evidence,
    score_raw_audio_evidence,
    score_raw_audio_evidence_independent,
    score_raw_audio_evidence_scalar_reference,
    score_timing_candidates,
)


def _constant_curve(
    *,
    start_beat: int = 0,
    end_beat: int = 20,
    bpm: float = 120.0,
    origin_time_ms: float = 1000.0,
) -> PhaseContinuousTimingCurve:
    return PhaseContinuousTimingCurve(
        start_beat,
        origin_time_ms,
        (ConstantTempoSection(start_beat, end_beat, bpm),),
    )


def _canonical_evidence(
    *,
    duration_seconds: float = 12.0,
    peak_times_seconds: tuple[float, ...] = (),
    peak_value: float = 1.0,
) -> RawAudioEvidence:
    config = DEFAULT_RAW_AUDIO_EVIDENCE_CONFIG
    frame_count = math.floor(
        (
            duration_seconds * config.sample_rate
            - config.win_length
        )
        / config.hop_length
    ) + 1
    frame_indices = np.arange(frame_count, dtype=np.float64)
    centers = (
        frame_indices * config.hop_length + config.win_length / 2.0
    ) / config.sample_rate
    flux = np.zeros((frame_count, 4), dtype=np.float32)
    for peak_time in peak_times_seconds:
        frame_index = int(np.argmin(np.abs(centers - peak_time)))
        flux[frame_index] = np.float32(peak_value)
    return RawAudioEvidence(
        frame_center_seconds=centers,
        band_flux=flux,
        band_percentile95=(1.0, 1.0, 1.0, 1.0),
        input_frame_count=frame_count,
        valid_frame_count=frame_count,
        audio_duration_seconds=duration_seconds,
    )


def test_extracts_exact_four_band_positive_flux_and_excludes_padded_tail() -> None:
    mel = np.zeros((6, 80), dtype=np.float32)
    mel[0] = np.linspace(-2.0, 2.0, 80, dtype=np.float32)
    mel[1] = mel[0] + np.arange(80, dtype=np.float32) / np.float32(10.0)
    mel[2] = mel[1] - np.float32(0.25)
    mel[3] = mel[2] + np.arange(80, 0, -1, dtype=np.float32) / np.float32(20.0)
    mel[4:] = np.float32(1_000_000.0)

    # Exactly four frames have complete 400-sample windows.  The final two mel
    # rows represent compute_log_mel_10ms right padding and must not influence
    # either the flux or its whole-track percentile.
    duration_seconds = 880.0 / 16_000.0
    result = extract_raw_audio_evidence(
        mel,
        audio_duration_seconds=duration_seconds,
    )

    valid = mel[:4]
    differences = np.zeros_like(valid)
    differences[1:] = np.maximum(valid[1:] - valid[:-1], np.float32(0.0))
    expected = np.empty((4, 4), dtype=np.float32)
    expected_percentiles: list[float] = []
    for band_index, (start, end) in enumerate(((0, 10), (10, 25), (25, 45), (45, 80))):
        top_count = math.ceil((end - start) / 4)
        strengths: list[np.float32] = []
        for row in differences[:, start:end]:
            ordered_indices = np.argsort(-row, kind="stable")
            strength = np.float32(
                sum(float(row[index]) for index in ordered_indices[:top_count])
                / top_count
            )
            strengths.append(strength)
        strength_array = np.asarray(strengths, dtype=np.float32)
        percentile = float(np.percentile(strength_array, 95.0, method="linear"))
        expected_percentiles.append(percentile)
        if not np.any(strength_array):
            expected[:, band_index] = 0.0
        else:
            expected[:, band_index] = np.clip(
                strength_array.astype(np.float64) / (percentile + 1e-6),
                0.0,
                4.0,
            ).astype(np.float32)

    assert result.input_frame_count == 6
    assert result.valid_frame_count == 4
    assert result.frame_center_seconds.dtype == np.float64
    assert result.band_flux.dtype == np.float32
    assert result.frame_center_seconds == pytest.approx((0.0125, 0.0225, 0.0325, 0.0425))
    np.testing.assert_array_equal(result.band_flux, expected)
    assert result.band_percentile95 == pytest.approx(expected_percentiles)
    assert not result.frame_center_seconds.flags.writeable
    assert not result.band_flux.flags.writeable


def test_first_frame_is_zero_and_an_all_zero_band_remains_zero() -> None:
    mel = np.ones((3, 80), dtype=np.float32)
    mel[0, :10] = np.float32(100.0)
    mel[1, :10] = np.float32(101.0)
    result = extract_audio_evidence(mel, audio_duration_seconds=0.045)

    np.testing.assert_array_equal(result.band_flux[0], np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(result.band_flux[:, 1:], np.zeros((3, 3), dtype=np.float32))
    assert result.band_percentile95[1:] == (0.0, 0.0, 0.0)


def test_raw_audio_evidence_defensively_copies_canonical_input_views() -> None:
    center_base = np.arange(6, dtype=np.float64)
    flux_base = np.zeros((6, 4), dtype=np.float32)
    center_view = center_base[1:5]
    flux_view = flux_base[1:5]
    expected_centers = center_view.copy()
    expected_flux = flux_view.copy()

    result = RawAudioEvidence(
        frame_center_seconds=center_view,
        band_flux=flux_view,
        band_percentile95=(0.0, 0.0, 0.0, 0.0),
        input_frame_count=4,
        valid_frame_count=4,
        audio_duration_seconds=4.0,
    )
    center_base[2] = 999.0
    flux_base[2] = np.float32(3.0)

    np.testing.assert_array_equal(result.frame_center_seconds, expected_centers)
    np.testing.assert_array_equal(result.band_flux, expected_flux)
    assert not result.frame_center_seconds.flags.writeable
    assert not result.band_flux.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        result.frame_center_seconds[0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        result.band_flux[0, 0] = np.float32(1.0)


@pytest.mark.parametrize(
    "centers",
    (
        np.asarray((-0.01, 0.10), dtype=np.float64),
        np.asarray((0.00, 1.01), dtype=np.float64),
    ),
)
def test_raw_audio_evidence_rejects_frame_centers_outside_audio_duration(
    centers: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="audio_duration_seconds"):
        RawAudioEvidence(
            frame_center_seconds=centers,
            band_flux=np.zeros((2, 4), dtype=np.float32),
            band_percentile95=(0.0, 0.0, 0.0, 0.0),
            input_frame_count=2,
            valid_frame_count=2,
            audio_duration_seconds=1.0,
        )


@pytest.mark.parametrize(
    ("mel", "error", "match"),
    [
        (np.zeros((8, 80), dtype=np.float64), TypeError, "dtype float32"),
        (np.zeros((8, 79), dtype=np.float32), ValueError, "shape"),
        (np.full((8, 80), np.nan, dtype=np.float32), ValueError, "finite"),
    ],
)
def test_extractor_rejects_noncanonical_mel_input(
    mel: np.ndarray,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        extract_raw_audio_evidence(mel, audio_duration_seconds=1.0)


def test_frozen_config_cannot_be_tuned_at_call_time() -> None:
    mel = np.zeros((4, 80), dtype=np.float32)
    changed = RawAudioEvidenceConfig(event_window_seconds=0.051)
    with pytest.raises(ValueError, match="config is frozen"):
        extract_raw_audio_evidence(
            mel,
            audio_duration_seconds=1.0,
            config=changed,
        )


def test_event_windows_include_literal_float64_boundaries_and_exclude_one_ulp_outside() -> None:
    radius = np.float64(0.050)
    lower = -radius
    upper = radius
    centers = np.asarray(
        (
            np.nextafter(lower, -np.inf),
            lower,
            np.float64(0.0),
            upper,
            np.nextafter(upper, np.inf),
        ),
        dtype=np.float64,
    )
    starts, ends = _inclusive_event_windows(
        centers,
        np.asarray((0.0,), dtype=np.float64),
        radius_seconds=float(radius),
    )

    assert starts.tolist() == [1]
    assert ends.tolist() == [4]
    included = centers[starts[0] : ends[0]]
    assert included.tolist() == [lower, 0.0, upper]


def test_exact_score_reducer_uses_beat_halfbeat_and_one_16_beat_window() -> None:
    truth = _constant_curve()
    beat_times = tuple(truth.time_at_beat(beat) / 1000.0 for beat in range(20))
    evidence = _canonical_evidence(peak_times_seconds=beat_times)

    result = score_raw_audio_evidence(evidence, (truth,))
    score = result.candidate_scores[0]

    assert result.unavailable_reason is None
    assert result.common_beat_indices == tuple(range(20))
    assert result.complete_window_start_beats == (0,)
    assert score.mean_beat_support == 1.0
    assert score.mean_half_beat_support == 0.0
    assert score.window_contrast_p10 == 1.0
    assert score.raw_score == 1.25
    assert score.retained_beat_count == 20
    assert score.complete_window_count == 1
    assert result.selected_fingerprint_sha256 == truth.fingerprint_sha256


def test_raw_audio_score_selects_phase_aligned_curve_without_any_prior() -> None:
    truth = _constant_curve()
    offset_decoy = _constant_curve(origin_time_ms=1160.0)
    mel = np.zeros((1200, 80), dtype=np.float32)
    centers = (np.arange(1200, dtype=np.float64) * 160.0 + 200.0) / 16_000.0
    for beat in range(20):
        event_time = truth.time_at_beat(beat) / 1000.0
        frame_index = int(np.argmin(np.abs(centers - event_time)))
        mel[frame_index] = np.float32(2.0)

    result = score_timing_candidates(
        mel,
        (offset_decoy, truth),
        audio_duration_seconds=12.0,
    )

    assert result.unavailable_reason is None
    assert result.selected_candidate_index == 1
    assert result.selected_fingerprint_sha256 == truth.fingerprint_sha256
    assert result.candidate_scores[1].raw_score > result.candidate_scores[0].raw_score  # type: ignore[operator]


def test_all_candidates_use_identical_common_retained_denominator() -> None:
    first = _constant_curve(origin_time_ms=1000.0)
    second = _constant_curve(origin_time_ms=1100.0)
    full = _canonical_evidence(duration_seconds=12.0)
    centers = full.frame_center_seconds.copy()
    keep = ~(
        ((centers >= 1.20) & (centers <= 1.30))
        | ((centers >= 10.70) & (centers <= 10.80))
    )
    evidence = RawAudioEvidence(
        frame_center_seconds=centers[keep],
        band_flux=full.band_flux[keep].copy(),
        band_percentile95=full.band_percentile95,
        input_frame_count=int(np.count_nonzero(keep)),
        valid_frame_count=int(np.count_nonzero(keep)),
        audio_duration_seconds=12.0,
    )

    result = score_raw_audio_evidence(evidence, (first, second))

    assert result.unavailable_reason is None
    assert result.candidate_scores[0].retained_beat_count == len(result.common_beat_indices)
    assert result.candidate_scores[1].retained_beat_count == len(result.common_beat_indices)
    assert result.candidate_scores[0].retained_beat_count == result.candidate_scores[1].retained_beat_count


def test_independent_score_does_not_censor_direct_tempo_with_half_time_alias() -> None:
    direct = _constant_curve(end_beat=32, bpm=120.0, origin_time_ms=1000.0)
    half_time = _constant_curve(end_beat=16, bpm=60.0, origin_time_ms=1000.0)
    beat_times = tuple(direct.time_at_beat(beat) / 1000.0 for beat in range(32))
    evidence = _canonical_evidence(
        duration_seconds=17.5,
        peak_times_seconds=beat_times,
    )

    with pytest.raises(ValueError, match="same integer beat domain"):
        score_raw_audio_evidence(evidence, (direct, half_time))

    result = score_raw_audio_evidence_independent(evidence, (direct, half_time))
    direct_score, half_time_score = result.candidate_scores

    assert result.unavailable_reason is None
    assert result.common_beat_indices == ()
    assert result.complete_window_start_beats == ()
    assert result.selected_candidate_index == 0
    assert result.ranked_scores == (direct_score, half_time_score)
    assert direct_score.retained_beat_count == 32
    assert direct_score.candidate_domain_beat_count == 32
    assert direct_score.retained_domain_ratio == 1.0
    assert direct_score.complete_window_start_beats == (0, 16)
    assert half_time_score.retained_beat_count == 16
    assert half_time_score.candidate_domain_beat_count == 16
    assert half_time_score.retained_domain_ratio == 1.0
    assert half_time_score.complete_window_start_beats == (0,)
    assert direct_score.raw_score > half_time_score.raw_score  # type: ignore[operator]


def test_independent_score_marks_coverage_unavailable_per_candidate() -> None:
    available = _constant_curve(end_beat=17, bpm=120.0, origin_time_ms=1000.0)
    clipped = _constant_curve(end_beat=20, bpm=120.0, origin_time_ms=1000.0)
    beat_times = tuple(available.time_at_beat(beat) / 1000.0 for beat in range(17))
    evidence = _canonical_evidence(
        duration_seconds=9.35,
        peak_times_seconds=beat_times,
    )

    result = score_raw_audio_evidence_independent(evidence, (available, clipped))
    available_score, clipped_score = result.candidate_scores

    assert result.unavailable_reason is None
    assert result.ranked_scores == (available_score,)
    assert available_score.available
    assert available_score.retained_beat_count == 17
    assert available_score.candidate_domain_beat_count == 17
    assert available_score.retained_domain_ratio == 1.0
    assert clipped_score.raw_score is None
    assert clipped_score.retained_beat_count == 17
    assert clipped_score.candidate_domain_beat_count == 20
    assert clipped_score.retained_domain_ratio == pytest.approx(0.85)
    assert (
        clipped_score.unavailable_reason
        == RAW_AUDIO_EVIDENCE_UNAVAILABLE_RETAINED_DOMAIN_RATIO_BELOW_90
    )

    unavailable = score_raw_audio_evidence_independent(evidence, (clipped,))
    assert unavailable.ranked_scores == ()
    assert (
        unavailable.unavailable_reason
        == RAW_AUDIO_EVIDENCE_UNAVAILABLE_RETAINED_DOMAIN_RATIO_BELOW_90
    )


def test_independent_all_unavailable_reason_uses_frozen_priority() -> None:
    fewer = _constant_curve(end_beat=12, bpm=120.0, origin_time_ms=1000.0)
    coverage = _constant_curve(end_beat=20, bpm=120.0, origin_time_ms=1000.0)
    beat_times = tuple(coverage.time_at_beat(beat) / 1000.0 for beat in range(17))
    evidence = _canonical_evidence(
        duration_seconds=9.35,
        peak_times_seconds=beat_times,
    )

    result = score_raw_audio_evidence_independent(evidence, (fewer, coverage))
    fewer_score, coverage_score = result.candidate_scores

    assert result.ranked_scores == ()
    assert (
        result.unavailable_reason
        == RAW_AUDIO_EVIDENCE_UNAVAILABLE_RETAINED_DOMAIN_RATIO_BELOW_90
    )
    assert (
        fewer_score.unavailable_reason
        == RAW_AUDIO_EVIDENCE_UNAVAILABLE_FEWER_THAN_16_COMMON_BEATS
    )
    assert (
        coverage_score.unavailable_reason
        == RAW_AUDIO_EVIDENCE_UNAVAILABLE_RETAINED_DOMAIN_RATIO_BELOW_90
    )


def test_independent_score_uses_per_candidate_complete_16_beat_windows() -> None:
    curve = _constant_curve(end_beat=32, bpm=60.0, origin_time_ms=0.0)
    centers = np.asarray(
        sorted(
            time
            for beat in range(32)
            for time in (float(beat), beat + 0.5)
            if time not in (8.0, 24.0)
        ),
        dtype=np.float64,
    )
    flux = np.zeros((centers.size, 4), dtype=np.float32)
    evidence = RawAudioEvidence(
        frame_center_seconds=centers,
        band_flux=flux,
        band_percentile95=(0.0, 0.0, 0.0, 0.0),
        input_frame_count=centers.size,
        valid_frame_count=centers.size,
        audio_duration_seconds=32.0,
    )

    result = score_raw_audio_evidence_independent(evidence, (curve,))
    score = result.candidate_scores[0]

    assert result.unavailable_reason == RAW_AUDIO_EVIDENCE_UNAVAILABLE_NO_COMPLETE_WINDOW
    assert score.retained_beat_count == 30
    assert score.candidate_domain_beat_count == 32
    assert score.retained_domain_ratio == pytest.approx(30.0 / 32.0)
    assert score.complete_window_count == 0
    assert score.complete_window_start_beats == ()
    assert score.unavailable_reason == RAW_AUDIO_EVIDENCE_UNAVAILABLE_NO_COMPLETE_WINDOW


def test_fewer_than_16_common_beats_returns_one_shared_unavailable_reason() -> None:
    first = _constant_curve(end_beat=12)
    second = _constant_curve(end_beat=12, origin_time_ms=1030.0)
    evidence = _canonical_evidence(duration_seconds=12.0)
    result = score_raw_audio_evidence(evidence, (first, second))

    assert result.unavailable_reason == RAW_AUDIO_EVIDENCE_UNAVAILABLE_FEWER_THAN_16_COMMON_BEATS
    assert result.ranked_scores == ()
    assert result.selected_fingerprint_sha256 is None
    assert len(result.common_beat_indices) == 12
    assert {score.unavailable_reason for score in result.candidate_scores} == {
        RAW_AUDIO_EVIDENCE_UNAVAILABLE_FEWER_THAN_16_COMMON_BEATS
    }
    assert all(score.raw_score is None for score in result.candidate_scores)


def test_missing_beats_can_leave_16_or_more_common_beats_but_no_complete_window() -> None:
    curve = _constant_curve(end_beat=32, bpm=60.0, origin_time_ms=0.0)
    centers = np.asarray(
        sorted(
            time
            for beat in range(32)
            for time in (float(beat), beat + 0.5)
            if time not in (8.0, 24.0)
        ),
        dtype=np.float64,
    )
    flux = np.zeros((centers.size, 4), dtype=np.float32)
    evidence = RawAudioEvidence(
        frame_center_seconds=centers,
        band_flux=flux,
        band_percentile95=(0.0, 0.0, 0.0, 0.0),
        input_frame_count=centers.size,
        valid_frame_count=centers.size,
        audio_duration_seconds=32.0,
    )

    result = score_raw_audio_evidence(evidence, (curve,))

    assert len(result.common_beat_indices) == 30
    assert 8 not in result.common_beat_indices
    assert 24 not in result.common_beat_indices
    assert result.unavailable_reason == RAW_AUDIO_EVIDENCE_UNAVAILABLE_NO_COMPLETE_WINDOW
    assert result.candidate_scores[0].raw_score is None


def test_vectorized_and_scalar_reference_scores_are_identical_within_contract() -> None:
    candidates = (
        _constant_curve(origin_time_ms=970.0),
        _constant_curve(origin_time_ms=1000.0),
        PhaseContinuousTimingCurve(
            0,
            1000.0,
            (LinearTimeRampSection(0, 20, 90.0, 150.0),),
        ),
    )
    rng = np.random.Generator(np.random.PCG64(12345))
    evidence = RawAudioEvidence(
        frame_center_seconds=(
            np.arange(1200, dtype=np.float64) * 160.0 + 200.0
        )
        / 16_000.0,
        band_flux=rng.uniform(0.0, 4.0, size=(1200, 4)).astype(np.float32),
        band_percentile95=(1.0, 1.0, 1.0, 1.0),
        input_frame_count=1200,
        valid_frame_count=1200,
        audio_duration_seconds=12.01,
    )

    vectorized = score_raw_audio_evidence(evidence, candidates)
    scalar = score_raw_audio_evidence_scalar_reference(evidence, candidates)

    assert vectorized.selected_fingerprint_sha256 == scalar.selected_fingerprint_sha256
    assert vectorized.common_beat_indices == scalar.common_beat_indices
    for actual, reference in zip(vectorized.candidate_scores, scalar.candidate_scores):
        assert actual.fingerprint_sha256 == reference.fingerprint_sha256
        assert abs(actual.raw_score - reference.raw_score) <= 1e-12  # type: ignore[operator]
        assert abs(actual.mean_beat_support - reference.mean_beat_support) <= 1e-12  # type: ignore[operator]
        assert abs(actual.mean_half_beat_support - reference.mean_half_beat_support) <= 1e-12  # type: ignore[operator]
        assert abs(actual.window_contrast_p10 - reference.window_contrast_p10) <= 1e-12  # type: ignore[operator]


def test_equal_scores_rank_by_ascending_canonical_fingerprint() -> None:
    first = _constant_curve(origin_time_ms=1000.0)
    second = _constant_curve(origin_time_ms=1030.0)
    evidence = _canonical_evidence()
    candidates = tuple(sorted((first, second), key=lambda curve: curve.fingerprint_sha256, reverse=True))

    result = score_raw_audio_evidence(evidence, candidates)

    assert all(score.raw_score == 0.0 for score in result.candidate_scores)
    assert [score.fingerprint_sha256 for score in result.ranked_scores] == sorted(
        (first.fingerprint_sha256, second.fingerprint_sha256)
    )


def test_candidate_tuple_domain_count_and_fingerprint_are_validated() -> None:
    evidence = _canonical_evidence()
    curve = _constant_curve()

    with pytest.raises(TypeError, match="ordered tuple"):
        score_raw_audio_evidence(evidence, [curve])  # type: ignore[arg-type]

    mismatch = _constant_curve(end_beat=21)
    with pytest.raises(ValueError, match="same integer beat domain"):
        score_raw_audio_evidence(evidence, (curve, mismatch))

    with pytest.raises(ValueError, match="fingerprints must be unique"):
        score_raw_audio_evidence(evidence, (curve, curve))

    too_many = tuple(
        _constant_curve(origin_time_ms=1000.0 + index)
        for index in range(65)
    )
    with pytest.raises(ValueError, match="more than 64"):
        score_raw_audio_evidence(evidence, too_many)


def test_candidate_times_must_be_finite_and_strictly_increasing() -> None:
    evidence = _canonical_evidence()

    class BadCandidate:
        start_beat = 0
        end_beat = 20
        fingerprint_sha256 = "a" * 64

        def time_at_beat(self, beat: float) -> float:
            return float("nan") if beat == 7.0 else beat * 500.0

    with pytest.raises(ValueError, match="time_at_beat"):
        score_raw_audio_evidence(evidence, (BadCandidate(),))
