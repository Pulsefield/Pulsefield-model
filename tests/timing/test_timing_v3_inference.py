from __future__ import annotations

import numpy as np

from pulsefield_model.timing.grid_fitting.types import (
    TimingFitDiagnostics,
    TimingFitResult,
)
from pulsefield_model.timing.schema import (
    FittedTimingGrid,
    FrameTimingPrediction,
    TimingSegment,
)
from pulsefield_model.timing.v3.analytic_curve import (
    ConstantTempoSection,
    PhaseContinuousTimingCurve,
)
from pulsefield_model.timing.v3.audio_evidence import RawAudioEvidence
from pulsefield_model.timing.v3.inference import (
    TimingEvidenceBundle,
    run_timing_v3_shadow,
    unpack_packed_mel_20ms_to_log_mel_10ms,
)
from pulsefield_model.timing.v3.tempo_track import (
    TempoTrackDiagnostics,
    TempoTrackProductionSelection,
    TempoTrackResult,
    TimingCandidateDiagnostic,
)


def test_v2_mode_keeps_v2_live_and_never_calls_candidate_generator() -> None:
    fit = _v2_fit()

    def unexpected_generator(*args: object, **kwargs: object) -> TempoTrackResult:
        del args, kwargs
        raise AssertionError("v2 mode must not call Timing v3")

    outcome = run_timing_v3_shadow(
        _evidence(),
        v2_fallback_fit=fit,
        mode="v2",
        candidate_generator=unexpected_generator,
    )

    assert outcome.v2_fallback_fit is fit
    assert outcome.live_timing_grid is fit.grid
    assert outcome.shadow_result is None
    assert outcome.telemetry.status == "disabled"
    assert outcome.selected_curve_canonical_bytes is None


def test_shadow_result_is_observable_and_curve_bytes_round_trip() -> None:
    fit = _v2_fit()
    result = _accepted_tempo_result()
    calls: list[tuple[FrameTimingPrediction, RawAudioEvidence | None]] = []

    def candidate_generator(
        prediction: FrameTimingPrediction,
        *,
        audio_evidence: RawAudioEvidence | None = None,
    ) -> TempoTrackResult:
        calls.append((prediction, audio_evidence))
        return result

    clock_values = iter((10.0, 10.125))
    outcome = run_timing_v3_shadow(
        _evidence(with_raw_audio=True),
        v2_fallback_fit=fit,
        mode="v3_shadow",
        candidate_generator=candidate_generator,
        clock=lambda: next(clock_values),
    )

    assert calls[0][0] is not None
    assert isinstance(calls[0][1], RawAudioEvidence)
    assert outcome.live_timing_grid is fit.grid
    assert outcome.shadow_result is result
    assert outcome.telemetry.status == "completed"
    assert outcome.telemetry.selection_status == "v3_accepted"
    assert outcome.telemetry.elapsed_ms == 125.0
    payload = outcome.selected_curve_canonical_bytes
    assert payload is not None
    restored = PhaseContinuousTimingCurve.from_canonical_bytes(payload)
    assert restored.canonical_bytes() == payload
    observable = outcome.to_observable_dict()
    assert observable["live_timing"] == "v2"
    assert observable["selected_shadow_curve"] == result.selected_candidate.to_dict()


def test_shadow_duration_limit_preserves_v2_but_programmer_failure_raises() -> None:
    fit = _v2_fit()
    generator_calls = 0

    def failing_generator(*args: object, **kwargs: object) -> TempoTrackResult:
        nonlocal generator_calls
        del args, kwargs
        generator_calls += 1
        raise RuntimeError("fixture failure")

    skipped = run_timing_v3_shadow(
        _evidence(duration_seconds=600.001),
        v2_fallback_fit=fit,
        mode="v3_shadow",
        max_supported_audio_duration_seconds=600.0,
        candidate_generator=failing_generator,
    )
    with np.testing.assert_raises_regex(RuntimeError, "fixture failure"):
        run_timing_v3_shadow(
            _evidence(),
            v2_fallback_fit=fit,
            mode="v3_shadow",
            candidate_generator=failing_generator,
            clock=iter((1.0, 1.01)).__next__,
        )

    assert generator_calls == 1
    assert skipped.v2_fallback_fit is fit
    assert skipped.telemetry.status == "skipped_duration"


def test_packed_mel_unpack_restores_interleaved_10ms_frames() -> None:
    packed = np.arange(2 * 160, dtype=np.float32).reshape(2, 160)

    unpacked = unpack_packed_mel_20ms_to_log_mel_10ms(packed)

    assert unpacked.shape == (4, 80)
    assert np.array_equal(unpacked[0], packed[0, :80])
    assert np.array_equal(unpacked[1], packed[0, 80:])
    assert np.array_equal(unpacked[2], packed[1, :80])
    assert np.array_equal(unpacked[3], packed[1, 80:])


def _evidence(
    *,
    duration_seconds: float = 4.0,
    with_raw_audio: bool = False,
) -> TimingEvidenceBundle:
    prediction = FrameTimingPrediction(
        provider="fixture",
        checkpoint_path="fixture-checkpoint",
        source_path="fixture.wav",
        beat_prob=np.full(200, 0.5, dtype=np.float32),
        downbeat_prob=np.full(200, 0.25, dtype=np.float32),
        frame_rate_hz=50.0,
    )
    raw = (
        np.zeros((400, 80), dtype=np.float32)
        if with_raw_audio
        else None
    )
    return TimingEvidenceBundle(
        beatthis_frame_probabilities=prediction,
        audio_duration_seconds=duration_seconds,
        raw_audio_log_mel_10ms=raw,
    )


def _v2_fit() -> TimingFitResult:
    grid = FittedTimingGrid((TimingSegment(offset_ms=0.0, beat_length_ms=500.0),))
    return TimingFitResult(
        grid=grid,
        score=1.0,
        diagnostics=TimingFitDiagnostics(
            fit_score=1.0,
            selected_period_frames=25.0,
            selected_offset_frames=0.0,
            selected_bpm=120.0,
            candidate_count=1,
            half_tempo_score=0.0,
            double_tempo_score=0.0,
            raw_selected_bpm=120.0,
            raw_score=1.0,
            tempo_multiplier=1.0,
            segment_alias_switch_count=0,
            tempo_multiplier_distribution={"1": 1},
        ),
    )


def _accepted_tempo_result() -> TempoTrackResult:
    curve = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(ConstantTempoSection(0, 8, 120.0),),
    )
    return TempoTrackResult(
        observations=(),
        candidates=(curve,),
        candidate_diagnostics=(
            TimingCandidateDiagnostic(
                fingerprint_sha256=curve.fingerprint_sha256,
                curve_class=curve.curve_class,
                source="fixture",
                generation_score=1.0,
            ),
        ),
        diagnostics=TempoTrackDiagnostics(
            version="fixture",
            beat_peak_count=0,
            raw_boundary_count=0,
            pair_seed_count=0,
            shared_start_beat=0,
            shared_end_beat=8,
            primary_origin_time_ms=0.0,
            primary_bpm=120.0,
            candidate_count=1,
        ),
        production_selection=TempoTrackProductionSelection(
            status="v3_accepted",
            selected_candidate_index=0,
            selected_fingerprint_sha256=curve.fingerprint_sha256,
            lane="constant",
            fallback_reason=None,
            raw_run=None,
            eligible_candidate_indices=(0,),
            raw_self_rank_by_candidate=((0, 1),),
            beatthis_aba_rank_by_candidate=(),
        ),
    )
