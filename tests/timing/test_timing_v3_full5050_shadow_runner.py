from __future__ import annotations

import json
from itertools import count
from pathlib import Path

import numpy as np
import pytest

from pulsefield_model.timing.evaluation.full5050_shadow_runner import (
    FULL5050_SHADOW_PLAN_SCHEMA,
    Full5050LocatorRow,
    Full5050ShadowPipeline,
    Full5050ShadowRunnerConfig,
    MissingBeatThisCacheError,
    compose_full5050_shadow_inference_config,
    load_full5050_locator_rows,
    main,
    run_full5050_shadow,
    run_full5050_shadow_row,
)
from pulsefield_model.timing.grid_fitting.types import (
    TimingFitDiagnostics,
    TimingFitResult,
)
from pulsefield_model.timing.providers.beatthis_cache import BeatThisFramePredictionCacheConfig
from pulsefield_model.timing.schema import (
    FittedTimingGrid,
    FrameTimingPrediction,
    TimingSegment,
)
from pulsefield_model.timing.v3.analytic_curve import (
    ConstantTempoSection,
    PhaseContinuousTimingCurve,
)
from pulsefield_model.timing.v3.inference import TimingV3Outcome, TimingV3Telemetry
from pulsefield_model.timing.v3.tempo_track import (
    TempoTrackDiagnostics,
    TempoTrackProductionSelection,
    TempoTrackResult,
    TimingCandidateDiagnostic,
)


def test_locator_loader_projects_only_allowed_manifest_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "labels.jsonl"
    _write_manifest(
        manifest,
        [
            {
                "resolved_audio_path": "/audio/a.mp3",
                "source": {
                    "cache_audio_key": "cache-a",
                    "cache_duration_seconds": 12.25,
                    "cache_status": "valid",
                },
                "label": {"must_not_be_read": True},
                "maps": [{"red_timing": "not ground truth"}],
                "metadata_bpm_evidence": {"bpm": "diagnostic only"},
                "representative_redline_grid": {"invalid": object().__class__.__name__},
            },
            {
                "resolved_audio_path": "/audio/b.mp3",
                "source": {
                    "cache_audio_key": None,
                    "cache_duration_seconds": 3.0,
                    "cache_status": "valid",
                },
                "label": None,
            },
        ],
    )

    rows = load_full5050_locator_rows(manifest, expected_row_count=2)

    assert rows == (
        Full5050LocatorRow(
            row_index=0,
            resolved_audio_path=Path("/audio/a.mp3"),
            beatthis_audio_cache_key="cache-a",
            duration_seconds=12.25,
            input_status="valid",
        ),
        Full5050LocatorRow(
            row_index=1,
            resolved_audio_path=Path("/audio/b.mp3"),
            beatthis_audio_cache_key=None,
            duration_seconds=3.0,
            input_status="valid",
        ),
    )
    assert rows[0].row_id == "full5050:0"
    assert rows[0].audio_length_ms == 12_250


def test_locator_loader_requires_exact_expected_count(tmp_path: Path) -> None:
    manifest = tmp_path / "labels.jsonl"
    _write_manifest(manifest, [_manifest_row("/audio/a.mp3", "cache-a")])

    with pytest.raises(ValueError, match="exactly 2 rows"):
        load_full5050_locator_rows(manifest, expected_row_count=2)


def test_runner_hydra_composition_forces_packaged_v3_shadow() -> None:
    config = compose_full5050_shadow_inference_config(
        ["timing.max_supported_audio_duration_seconds=480.0"],
    )

    assert config.timing.mode == "v3_shadow"
    assert config.timing.max_supported_audio_duration_seconds == 480.0

    with pytest.raises(ValueError, match="requires timing.mode=v3_shadow"):
        compose_full5050_shadow_inference_config(["timing=mock_default"])


def test_main_defaults_to_plan_only_without_writing_results(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest = tmp_path / "labels.jsonl"
    output = tmp_path / "results.jsonl"
    _write_manifest(manifest, [_manifest_row("/audio/a.mp3", "cache-a")])

    exit_code = main(
        [
            "--labels-path",
            str(manifest),
            "--output-jsonl",
            str(output),
            "--expected-row-count",
            "1",
        ],
    )

    assert exit_code == 0
    assert not output.exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == FULL5050_SHADOW_PLAN_SCHEMA
    assert payload["action"] == "plan_only"
    assert payload["total_rows"] == 1
    assert payload["timing_mode"] == "v3_shadow"


def test_shadow_row_success_records_stage_timings_curve_class_and_no_hash_payload() -> None:
    row = Full5050LocatorRow(
        row_index=4,
        resolved_audio_path=Path("/audio/a.mp3"),
        beatthis_audio_cache_key="cache-a",
        duration_seconds=2.0,
        input_status="valid",
    )

    result = run_full5050_shadow_row(row, _fake_pipeline())

    assert result["status"] == "completed"
    assert result["reason"] is None
    assert result["beatthis"]["source"] == "cache"
    assert result["timings_ms"].keys() == {"provider", "mel", "v2", "v3", "total"}
    assert result["v3"]["curve_class"] == "constant"
    assert result["v3"]["canonical_curve_roundtrip"] is True
    payload_text = json.dumps(result, sort_keys=True)
    assert "sha256" not in payload_text.lower()
    assert "fingerprint" not in payload_text.lower()


def test_shadow_row_missing_cache_is_failed_row_not_successful_fallback() -> None:
    row = Full5050LocatorRow(
        row_index=4,
        resolved_audio_path=Path("/audio/a.mp3"),
        beatthis_audio_cache_key="missing",
        duration_seconds=2.0,
        input_status="valid",
    )
    pipeline = _fake_pipeline(cache_loader=lambda _key, _config: None)

    result = run_full5050_shadow_row(row, pipeline)

    assert result["status"] == "failed"
    assert result["reason"] == "provider_failed"
    assert result["error"]["type"] == "MissingBeatThisCacheError"
    assert "fallback is disabled" in result["error"]["message"]
    assert result["beatthis"]["source"] is None


def test_full5050_shadow_run_resumes_final_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "labels.jsonl"
    output = tmp_path / "results.jsonl"
    _write_manifest(
        manifest,
        [
            _manifest_row("/audio/a.mp3", "cache-a"),
            _manifest_row("/audio/b.mp3", "cache-b"),
        ],
    )
    output.write_text(
        json.dumps(
            {
                "schema": "existing",
                "row_index": 0,
                "status": "completed",
            },
        )
        + "\n",
        encoding="utf-8",
    )
    config = Full5050ShadowRunnerConfig(
        labels_path=manifest,
        output_jsonl=output,
        expected_row_count=2,
    )
    inference_config = compose_full5050_shadow_inference_config()

    summary = run_full5050_shadow(
        config,
        inference_config=inference_config,
        pipeline=_fake_pipeline(),
    )

    assert summary["total_rows"] == 2
    assert summary["resumed_rows"] == 1
    assert summary["attempted_rows"] == 1
    assert summary["completed_rows"] == 2
    assert summary["recorded_rows"] == 2
    assert len(output.read_text(encoding="utf-8").strip().splitlines()) == 2


def _manifest_row(audio_path: str, cache_key: str | None) -> dict[str, object]:
    return {
        "resolved_audio_path": audio_path,
        "source": {
            "cache_audio_key": cache_key,
            "cache_duration_seconds": 2.0,
            "cache_status": "valid",
        },
        "label": {"ignored": True},
        "maps": [{"ignored": True}],
        "representative_redline_grid": {"ignored": True},
    }


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fake_pipeline(
    *,
    cache_loader=None,
) -> Full5050ShadowPipeline:
    tick = count().__next__
    return Full5050ShadowPipeline(
        timing_mode="v3_shadow",
        max_supported_audio_duration_seconds=600.0,
        beatthis_cache_config=BeatThisFramePredictionCacheConfig(),
        allow_beatthis_provider_fallback=False,
        grid_fitter=_FakeGridFitter(),
        mel_loader=lambda _path, *, audio_cache_key=None: np.zeros((2, 160), dtype=np.float32),
        beatthis_cache_loader=cache_loader or (lambda _key, _config: _prediction()),
        timing_v3_facade=lambda evidence, *, v2_fallback_fit, mode, max_supported_audio_duration_seconds: _outcome(
            v2_fallback_fit,
            mode=mode,
        ),
        clock=tick,
    )


class _FakeGridFitter:
    def fit(self, prediction: FrameTimingPrediction) -> TimingFitResult:
        del prediction
        return _v2_fit()


def _prediction() -> FrameTimingPrediction:
    return FrameTimingPrediction(
        provider="beat-this",
        checkpoint_path="final0",
        source_path="/audio/a.mp3",
        beat_prob=np.full(100, 0.5, dtype=np.float32),
        downbeat_prob=np.full(100, 0.25, dtype=np.float32),
        frame_rate_hz=50.0,
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


def _outcome(
    fit: TimingFitResult,
    *,
    mode: str,
) -> TimingV3Outcome:
    curve = PhaseContinuousTimingCurve(
        origin_beat=0,
        origin_time_ms=0.0,
        sections=(ConstantTempoSection(0, 8, 120.0),),
    )
    result = TempoTrackResult(
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
    return TimingV3Outcome(
        mode=mode,  # type: ignore[arg-type]
        v2_fallback_fit=fit,
        shadow_result=result,
        telemetry=TimingV3Telemetry(
            mode=mode,  # type: ignore[arg-type]
            status="completed",
            elapsed_ms=1.0,
            candidate_count=1,
            selection_status="v3_accepted",
            fallback_reason=None,
            selected_fingerprint_sha256=curve.fingerprint_sha256,
        ),
    )
