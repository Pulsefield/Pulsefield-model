from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pulsefield_model.timing.grid_fitting.types import TimingFitDiagnostics, TimingFitResult
from pulsefield_model.timing.schema import FittedTimingGrid, FrameTimingPrediction, TimingSegment
from pulsefield_model.timing.v3.fitter import TimingV3FitResult, TimingV3Fitter, TimingV3FitterConfig
from pulsefield_model.timing.v3.projection import (
    PROJECTION_METHOD_PRESERVE_ANCHORS,
    PROJECTION_METHOD_PRESERVE_BPM,
    REASON_RELATIVE_BPM_ADJUSTMENT_EXCEEDED,
)


class _RecordingV2Fitter:
    def __init__(self, fit_result: TimingFitResult) -> None:
        self.fit_result = fit_result
        self.calls: list[FrameTimingPrediction] = []

    def fit(self, prediction: FrameTimingPrediction) -> TimingFitResult:
        self.calls.append(prediction)
        return self.fit_result


class _MetadataTrappingPrediction(FrameTimingPrediction):
    def arm_metadata_traps(self) -> None:
        object.__setattr__(self, "_metadata_traps_armed", True)

    def __getattribute__(self, name: str) -> object:
        if name in {"source_path", "checkpoint_path", "provider", "metadata", "title"}:
            armed = object.__getattribute__(self, "__dict__").get("_metadata_traps_armed", False)
            if armed:
                raise AssertionError(f"unexpected metadata access: {name}")
        return super().__getattribute__(name)


def test_fit_calls_v2_once_with_same_prediction_and_preserves_v2_fit() -> None:
    prediction = _prediction(frame_count=100, frame_rate_hz=50.0)
    v2_fit = _fit_result(_grid((0.0, 500.0)))
    v2_fitter = _RecordingV2Fitter(v2_fit)

    result = TimingV3Fitter(v2_fitter=v2_fitter).fit(prediction)

    assert v2_fitter.calls == [prediction]
    assert v2_fitter.calls[0] is prediction
    assert result.v2_fit is v2_fit
    assert result.v2_fit.grid is v2_fit.grid
    assert result.fallback_v2 is False
    assert result.reason is None


def test_fit_derives_exact_cache_coverage_for_selected_and_control_projection() -> None:
    prediction = _prediction(frame_count=7, frame_rate_hz=3.0)
    expected_coverage_end_ms = 1000.0 * prediction.frame_count / prediction.frame_rate_hz

    result = TimingV3Fitter(
        v2_fitter=_RecordingV2Fitter(_fit_result(_grid((0.0, 500.0)))),
    ).fit(prediction)

    assert result.coverage_start_ms == 0.0
    assert result.coverage_end_ms == expected_coverage_end_ms
    assert result.selected_projection.diagnostics.coverage_start_ms == 0.0
    assert result.selected_projection.diagnostics.coverage_end_ms == expected_coverage_end_ms
    assert result.control_projection.diagnostics.coverage_start_ms == 0.0
    assert result.control_projection.diagnostics.coverage_end_ms == expected_coverage_end_ms
    assert result.selected_v3_grid is not None
    assert result.selected_v3_grid.coverage_start_ms == 0.0
    assert result.selected_v3_grid.coverage_end_ms == expected_coverage_end_ms


def test_family_b_is_selected_and_family_a_is_emitted_as_control() -> None:
    prediction = _prediction(frame_count=300, frame_rate_hz=50.0)
    v2_fit = _fit_result(_grid((0.0, 500.0), (2100.0, 400.0)))

    result = TimingV3Fitter(v2_fitter=_RecordingV2Fitter(v2_fit)).fit(prediction)

    assert result.selected_projection.method == PROJECTION_METHOD_PRESERVE_ANCHORS
    assert result.control_projection.method == PROJECTION_METHOD_PRESERVE_BPM
    assert result.fallback_v2 is False
    assert result.selected_v3_grid is result.selected_projection.grid
    selected_diagnostic = result.selected_projection.diagnostics.boundary_diagnostics[0]
    control_diagnostic = result.control_projection.diagnostics.boundary_diagnostics[0]
    assert selected_diagnostic.projected_right_anchor_ms == pytest.approx(2100.0)
    assert selected_diagnostic.relative_bpm_adjustment == pytest.approx(-100.0 / 2100.0)
    assert control_diagnostic.projected_right_anchor_ms == pytest.approx(2000.0)
    assert control_diagnostic.relative_bpm_adjustment == pytest.approx(0.0)


def test_family_b_projection_failure_is_explicit_v2_fallback_with_control_projection() -> None:
    prediction = _prediction(frame_count=300, frame_rate_hz=50.0)
    v2_fit = _fit_result(_grid((0.0, 500.0), (526.5, 400.0)))

    result = TimingV3Fitter(v2_fitter=_RecordingV2Fitter(v2_fit)).fit(prediction)

    assert result.v2_fit is v2_fit
    assert result.fallback_v2 is True
    assert result.reason == REASON_RELATIVE_BPM_ADJUSTMENT_EXCEEDED
    assert result.selected_projection.reason == REASON_RELATIVE_BPM_ADJUSTMENT_EXCEEDED
    assert result.selected_projection.grid is None
    assert result.selected_v3_grid is None
    assert result.control_projection.ok
    assert result.control_projection.method == PROJECTION_METHOD_PRESERVE_BPM


def test_fit_api_consumes_one_prediction_and_does_not_read_prediction_metadata() -> None:
    signature = inspect.signature(TimingV3Fitter.fit)
    assert list(signature.parameters) == ["self", "prediction"]

    prediction = _metadata_trapping_prediction(frame_count=100, frame_rate_hz=50.0)
    v2_fitter = _RecordingV2Fitter(_fit_result(_grid((0.0, 500.0))))
    prediction.arm_metadata_traps()

    result = TimingV3Fitter(v2_fitter=v2_fitter).fit(prediction)

    assert v2_fitter.calls == [prediction]
    assert result.selected_projection.ok


def test_result_and_config_are_immutable_typed_dataclasses() -> None:
    config = TimingV3FitterConfig()
    with pytest.raises(FrozenInstanceError):
        config.min_bpm = 40.0  # type: ignore[misc]

    result = TimingV3Fitter(
        v2_fitter=_RecordingV2Fitter(_fit_result(_grid((0.0, 500.0)))),
    ).fit(
        _prediction(frame_count=100, frame_rate_hz=50.0),
    )
    assert isinstance(result, TimingV3FitResult)
    with pytest.raises(FrozenInstanceError):
        result.fallback_v2 = True  # type: ignore[misc]


def _prediction(*, frame_count: int, frame_rate_hz: float) -> FrameTimingPrediction:
    return FrameTimingPrediction(
        provider="cached-beatthis",
        checkpoint_path="checkpoint.pt",
        source_path="song.osu",
        beat_prob=np.linspace(0.0, 1.0, frame_count, dtype=np.float32),
        downbeat_prob=np.zeros(frame_count, dtype=np.float32),
        frame_rate_hz=frame_rate_hz,
    )


def _metadata_trapping_prediction(*, frame_count: int, frame_rate_hz: float) -> _MetadataTrappingPrediction:
    return _MetadataTrappingPrediction(
        provider="cached-beatthis",
        checkpoint_path="checkpoint.pt",
        source_path="map-title.osu",
        beat_prob=np.linspace(0.0, 1.0, frame_count, dtype=np.float32),
        downbeat_prob=np.zeros(frame_count, dtype=np.float32),
        frame_rate_hz=frame_rate_hz,
    )


def _grid(*segments: tuple[float, float]) -> FittedTimingGrid:
    return FittedTimingGrid(
        tuple(
            TimingSegment(offset_ms=offset_ms, beat_length_ms=beat_length_ms)
            for offset_ms, beat_length_ms in segments
        )
    )


def _fit_result(grid: FittedTimingGrid) -> TimingFitResult:
    return TimingFitResult(
        grid=grid,
        score=0.9,
        diagnostics=TimingFitDiagnostics(
            fit_score=0.9,
            selected_period_frames=25.0,
            selected_offset_frames=0.0,
            selected_bpm=120.0,
            candidate_count=1,
            half_tempo_score=0.1,
            double_tempo_score=0.2,
            raw_selected_bpm=120.0,
            raw_score=0.9,
            tempo_multiplier=1.0,
            segment_alias_switch_count=0,
            tempo_multiplier_distribution={"1.0": 1},
        ),
    )
