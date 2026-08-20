from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from pulsefield_model.inference.session_runtime import SessionRuntime, SessionRuntimeConfig
from pulsefield_model.timing.grid_fitting.types import TimingFitDiagnostics, TimingFitResult
from pulsefield_model.timing.schema import FittedTimingGrid, FrameTimingPrediction, TimingSegment
from pulsefield_model.timing.v3.analytic_curve import (
    ConstantTempoSection,
    PhaseContinuousTimingCurve,
)
from pulsefield_model.timing.v3.inference import (
    TimingEvidenceBundle,
    TimingV3Outcome,
    TimingV3Telemetry,
    run_timing_v3_shadow,
)
from pulsefield_model.timing.v3.tempo_track import (
    TempoTrackDiagnostics,
    TempoTrackProductionSelection,
    TempoTrackResult,
    TimingCandidateDiagnostic,
)


class SessionRuntimeTests(unittest.TestCase):
    def test_auto_session_device_inherits_model_runtime_device(self) -> None:
        runtime = SessionRuntime(
            session_id="s1",
            model_runtime=_fake_model_runtime(_FakeTimingProvider(_prediction())),
            config=SessionRuntimeConfig(device="auto", minimum_frame_count=4),
            mel_loader=_fake_mel_loader(np.zeros((5, 160), dtype=np.float32)),
            grid_fitter=_FakeGridFitter(),
        )

        self.assertEqual(runtime.device, torch.device("cpu"))

    def test_prepare_audio_caches_mel_dense_timing_and_padding_mask(self) -> None:
        audio_path = Path("song.wav")
        mel = np.arange(3 * 160, dtype=np.float32).reshape(3, 160)
        prediction = _prediction(frame_count=512, source_path=audio_path)
        provider = _FakeTimingProvider(prediction)
        fitter = _FakeGridFitter()
        control_model = _FakeControlModel(control_dim=6)
        runtime = SessionRuntime(
            session_id="s1",
            model_runtime=_fake_model_runtime(provider, control_model=control_model),
            config=SessionRuntimeConfig(minimum_frame_count=4),
            mel_loader=_fake_mel_loader(mel),
            grid_fitter=fitter,
        )

        cache = runtime.prepare_audio(audio_path, audio_length_ms=1_234)

        self.assertIs(runtime.audio_cache, cache)
        self.assertEqual(cache.session_id, "s1")
        self.assertEqual(cache.audio_path, audio_path)
        self.assertEqual(cache.audio_length_ms, 1_234)
        self.assertEqual(cache.audio_length_source, "provided")
        self.assertEqual(cache.source_frame_count, 3)
        self.assertEqual(cache.padded_frame_count, 400)
        self.assertEqual(tuple(cache.full_mel.shape), (1, 400, 160))
        self.assertEqual(tuple(cache.full_dense_timing_v2.shape), (1, 400, 4))
        self.assertEqual(tuple(cache.padding_mask.shape), (1, 400))
        self.assertEqual(cache.padding_mask[:, :5].tolist(), [[False, False, False, True, True]])
        self.assertEqual(cache.frame_count_tensor.tolist(), [400])
        self.assertEqual(cache.source_frame_count_tensor.tolist(), [3])
        self.assertTrue(torch.equal(cache.full_mel[0, :3], torch.as_tensor(mel)))
        self.assertTrue(torch.equal(cache.full_mel[0, 3:], torch.zeros(397, 160)))
        self.assertTrue(torch.isfinite(cache.full_dense_timing_v2).all())
        self.assertIs(cache.beatthis_prediction, prediction)
        self.assertIs(cache.timing_fit_result, fitter.result)
        self.assertIs(cache.timing_grid, fitter.result.grid)
        self.assertIs(cache.timing_v3_outcome.v2_fallback_fit, fitter.result)
        self.assertEqual(cache.timing_v3_outcome.mode, "v2")
        self.assertEqual(cache.timing_v3_outcome.telemetry.status, "disabled")
        self.assertEqual(provider.paths, [audio_path])
        self.assertEqual(fitter.predictions, [prediction])
        self.assertIsNotNone(runtime.control_cache)
        assert runtime.control_cache is not None
        self.assertEqual(runtime.control_cache.start_ms, 0)
        self.assertEqual(runtime.control_cache.target_start_frame, 0)
        self.assertEqual(runtime.control_cache.control_slice_start_frames.tolist(), [[0, 100, 200, 300]])
        self.assertEqual(tuple(runtime.control_cache.control_memory_8s.shape), (1, 400, 6))
        self.assertEqual(tuple(runtime.control_cache.density_teacher_8s.shape), (1, 400, 1))
        self.assertIn("density_teacher_8s", runtime.control_cache.as_model_batch())
        self.assertEqual(control_model.calls[0]["target_start_frame"], [0, 100, 200, 300])
        self.assertFalse(control_model.calls[0]["grad_enabled"])
        self.assertTrue(control_model.calls[0]["inference_mode"])

        batch = cache.as_model_batch()
        self.assertIs(batch["full_mel"], cache.full_mel)
        self.assertIs(batch["full_dense_timing_v2"], cache.full_dense_timing_v2)
        self.assertIs(batch["padding_mask"], cache.padding_mask)
        self.assertIs(batch["frame_count"], cache.frame_count_tensor)
        self.assertIs(batch["source_frame_count"], cache.source_frame_count_tensor)

    def test_prepare_audio_rejects_missing_audio_length(self) -> None:
        runtime = SessionRuntime(
            session_id="s1",
            model_runtime=_fake_model_runtime(_FakeTimingProvider(_prediction())),
            config=SessionRuntimeConfig(minimum_frame_count=4),
            mel_loader=_fake_mel_loader(np.zeros((5, 160), dtype=np.float32)),
            grid_fitter=_FakeGridFitter(),
        )

        with self.assertRaisesRegex(TypeError, "audio_length_ms"):
            runtime.prepare_audio("song.wav", audio_length_ms=None)

    def test_reset_audio_cache_drops_cached_tensors(self) -> None:
        runtime = SessionRuntime(
            session_id="s1",
            model_runtime=_fake_model_runtime(_FakeTimingProvider(_prediction())),
            config=SessionRuntimeConfig(minimum_frame_count=4),
            mel_loader=_fake_mel_loader(np.zeros((5, 160), dtype=np.float32)),
            grid_fitter=_FakeGridFitter(),
        )
        runtime.prepare_audio("song.wav", audio_length_ms=100)

        runtime.reset_audio_cache()

        self.assertIsNone(runtime.audio_cache)
        self.assertIsNone(runtime.control_cache)

    def test_prepare_audio_accepts_start_ms_for_initial_control_window(self) -> None:
        control_model = _FakeControlModel(control_dim=3)
        runtime = SessionRuntime(
            session_id="s1",
            model_runtime=_fake_model_runtime(_FakeTimingProvider(_prediction()), control_model=control_model),
            config=SessionRuntimeConfig(minimum_frame_count=4),
            mel_loader=_fake_mel_loader(np.zeros((450, 160), dtype=np.float32)),
            grid_fitter=_FakeGridFitter(),
        )

        cache = runtime.prepare_audio("song.wav", audio_length_ms=9_000, start_ms=2_000)

        self.assertEqual(cache.padded_frame_count, 500)
        self.assertEqual(cache.padding_mask[:, 448:500].tolist()[0][:5], [False, False, True, True, True])
        self.assertIsNotNone(runtime.control_cache)
        assert runtime.control_cache is not None
        self.assertEqual(runtime.control_cache.start_ms, 2_000)
        self.assertEqual(runtime.control_cache.target_start_frame, 100)
        self.assertEqual(runtime.control_cache.control_slice_start_frames.tolist(), [[100, 200, 300, 400]])
        self.assertEqual(control_model.calls[0]["target_start_frame"], [100, 200, 300, 400])

    def test_prepare_control_reuses_audio_cache_without_rerunning_timing(self) -> None:
        prediction = _prediction()
        provider = _FakeTimingProvider(prediction)
        fitter = _FakeGridFitter()
        control_model = _FakeControlModel(control_dim=3)
        runtime = SessionRuntime(
            session_id="s1",
            model_runtime=_fake_model_runtime(provider, control_model=control_model),
            config=SessionRuntimeConfig(minimum_frame_count=4),
            mel_loader=_fake_mel_loader(np.zeros((500, 160), dtype=np.float32)),
            grid_fitter=fitter,
        )
        runtime.prepare_audio("song.wav", audio_length_ms=10_000)
        first_audio_cache = runtime.audio_cache

        control_cache = runtime.prepare_control(start_ms=1_000)

        self.assertIs(runtime.audio_cache, first_audio_cache)
        self.assertIs(runtime.control_cache, control_cache)
        self.assertEqual(provider.paths, [Path("song.wav")])
        self.assertEqual(fitter.predictions, [prediction])
        self.assertEqual(len(control_model.calls), 2)
        self.assertEqual(control_cache.control_slice_start_frames.tolist(), [[50, 150, 250, 350]])

    def test_prepare_control_batch_computes_multiple_windows(self) -> None:
        control_model = _FakeControlModel(control_dim=3)
        runtime = SessionRuntime(
            session_id="s1",
            model_runtime=_fake_model_runtime(_FakeTimingProvider(_prediction()), control_model=control_model),
            config=SessionRuntimeConfig(minimum_frame_count=4, max_control_batch_size=12),
            mel_loader=_fake_mel_loader(np.zeros((900, 160), dtype=np.float32)),
            grid_fitter=_FakeGridFitter(),
        )
        runtime.prepare_audio("song.wav", audio_length_ms=18_000)

        batch_cache = runtime.prepare_control_batch(start_ms_values=(0, 8_000, 16_000))

        self.assertIs(runtime.control_batch_cache, batch_cache)
        self.assertIsNone(runtime.control_cache)
        self.assertEqual(batch_cache.start_ms_values, (0, 8_000, 16_000))
        self.assertEqual(batch_cache.target_start_frames, (0, 400, 800))
        self.assertEqual(
            batch_cache.control_slice_start_frames.tolist(),
            [[0, 100, 200, 300], [400, 500, 600, 700], [800, 900, 1000, 1100]],
        )
        self.assertEqual(tuple(batch_cache.control_memory_8s.shape), (3, 400, 3))
        self.assertEqual(tuple(batch_cache.density_teacher_8s.shape), (3, 400, 1))
        self.assertEqual(float(batch_cache.control_memory_8s[1, 0, 0].item()), 400.0)
        self.assertEqual(float(batch_cache.control_memory_8s[1, 100, 0].item()), 500.0)
        self.assertEqual(float(batch_cache.control_memory_8s[2, 300, 0].item()), 1100.0)
        self.assertEqual(float(batch_cache.density_teacher_8s[1, 0, 0].item()), 400.0)
        self.assertEqual(float(batch_cache.density_teacher_8s[2, 300, 0].item()), 1100.0)
        self.assertEqual(
            control_model.calls[-1]["target_start_frame"],
            [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100],
        )

    def test_prepare_control_batch_rejects_more_than_max(self) -> None:
        runtime = SessionRuntime(
            session_id="s1",
            model_runtime=_fake_model_runtime(_FakeTimingProvider(_prediction())),
            config=SessionRuntimeConfig(minimum_frame_count=4, max_control_batch_size=2),
            mel_loader=_fake_mel_loader(np.zeros((900, 160), dtype=np.float32)),
            grid_fitter=_FakeGridFitter(),
        )
        runtime.prepare_audio("song.wav", audio_length_ms=18_000)

        with self.assertRaisesRegex(ValueError, "<= 2"):
            runtime.prepare_control_batch(start_ms_values=(0, 8_000, 16_000))

    def test_prepare_full_control_batches_whole_song(self) -> None:
        control_model = _FakeControlModel(control_dim=2)
        runtime = SessionRuntime(
            session_id="s1",
            model_runtime=_fake_model_runtime(_FakeTimingProvider(_prediction()), control_model=control_model),
            config=SessionRuntimeConfig(minimum_frame_count=4, max_control_batch_size=2),
            mel_loader=_fake_mel_loader(np.zeros((850, 160), dtype=np.float32)),
            grid_fitter=_FakeGridFitter(),
        )
        runtime.prepare_audio("song.wav", audio_length_ms=18_000)

        full_cache = runtime.prepare_full_control()

        self.assertIs(runtime.full_control_cache, full_cache)
        self.assertEqual(full_cache.max_batch_size, 2)
        self.assertEqual(full_cache.start_ms_values, (0, 8_000, 16_000))
        self.assertEqual(full_cache.target_start_frames, (0, 400, 800))
        self.assertEqual(tuple(full_cache.control_memory_8s.shape), (3, 400, 2))
        self.assertEqual(tuple(full_cache.density_teacher_8s.shape), (3, 400, 1))
        self.assertEqual(float(full_cache.control_memory_8s[2, 0, 0].item()), 800.0)
        self.assertEqual(float(full_cache.control_memory_8s[2, 300, 0].item()), 1100.0)
        self.assertEqual(float(full_cache.density_teacher_8s[2, 300, 0].item()), 1100.0)
        self.assertEqual(len(control_model.calls), 3)
        self.assertEqual(control_model.calls[-2]["target_start_frame"], [0, 100, 200, 300, 400, 500, 600, 700])
        self.assertEqual(control_model.calls[-1]["target_start_frame"], [800, 900, 1000, 1100])

    def test_prepare_mapper_window_caches_projected_control_and_global_context(self) -> None:
        control_model = _FakeControlModel(control_dim=3)
        mapper_model = _FakeMapperModel(control_dim=3, d_model=5)
        runtime = SessionRuntime(
            session_id="s1",
            model_runtime=_fake_model_runtime(
                _FakeTimingProvider(_prediction()),
                control_model=control_model,
                mapper_model=mapper_model,
            ),
            config=SessionRuntimeConfig(minimum_frame_count=4),
            mel_loader=_fake_mel_loader(np.zeros((500, 160), dtype=np.float32)),
            grid_fitter=_FakeGridFitter(),
        )
        runtime.prepare_audio("song.wav", audio_length_ms=10_000)

        window_cache = runtime.prepare_mapper_window(start_ms=0, include_control_attention_kv_cache=True)
        reused = runtime.prepare_mapper_window(start_ms=0, include_control_attention_kv_cache=True)

        self.assertIs(runtime.mapper_window_cache, window_cache)
        self.assertIs(reused, window_cache)
        self.assertEqual(window_cache.start_ms, 0)
        self.assertEqual(window_cache.end_ms, 8_000)
        self.assertEqual(window_cache.target_start_frame, 0)
        self.assertEqual(tuple(window_cache.projected_control_memory_8s.shape), (1, 400, 5))
        self.assertEqual(tuple(window_cache.density_feature_8s.shape), (1, 400, 1))
        self.assertEqual(tuple(window_cache.global_memory.shape), (1, 7, 5))
        self.assertEqual(tuple(window_cache.global_memory_padding_mask.shape), (1, 7))
        self.assertEqual(tuple(window_cache.global_position_features.shape), (1, 4))
        self.assertIsNotNone(window_cache.global_attention_kv_cache)
        assert window_cache.global_attention_kv_cache is not None
        self.assertEqual(len(window_cache.global_attention_kv_cache), 2)
        self.assertEqual(tuple(window_cache.global_attention_kv_cache[0][0].shape), (1, 1, 7, 5))
        self.assertEqual(tuple(window_cache.global_attention_kv_cache[0][1].shape), (1, 1, 7, 5))
        self.assertIsNotNone(window_cache.control_attention_kv_cache)
        assert window_cache.control_attention_kv_cache is not None
        self.assertEqual(len(window_cache.control_attention_kv_cache), 2)
        self.assertEqual(tuple(window_cache.control_attention_kv_cache[0][0].shape), (1, 1, 400, 5))
        self.assertEqual(tuple(window_cache.control_attention_kv_cache[0][1].shape), (1, 1, 400, 5))
        self.assertEqual(len(mapper_model.global_calls), 1)
        self.assertEqual(len(mapper_model.kv_calls), 1)
        self.assertEqual(len(mapper_model.control_kv_calls), 1)
        self.assertFalse(mapper_model.global_calls[0]["grad_enabled"])
        self.assertTrue(mapper_model.global_calls[0]["inference_mode"])
        self.assertFalse(mapper_model.kv_calls[0]["grad_enabled"])
        self.assertTrue(mapper_model.kv_calls[0]["inference_mode"])
        self.assertFalse(mapper_model.control_kv_calls[0]["grad_enabled"])
        self.assertTrue(mapper_model.control_kv_calls[0]["inference_mode"])
        self.assertEqual(mapper_model.global_calls[0]["target_start_frame"], [0])

        batch = window_cache.as_model_batch()
        self.assertIn("projected_control_memory_8s", batch)
        self.assertIn("density_teacher_8s", batch)
        self.assertIn("control_attention_kv_cache", batch)
        self.assertIn("global_memory", batch)
        self.assertIn("global_attention_kv_cache", batch)
        self.assertNotIn("control_memory_8s", batch)

        runtime.prepare_control(start_ms=1_000)
        self.assertIsNone(runtime.mapper_window_cache)

    def test_prepare_audio_consumes_v3_shadow_without_replacing_live_v2_grid(self) -> None:
        mel = np.arange(3 * 160, dtype=np.float32).reshape(3, 160)
        fitter = _FakeGridFitter()
        captured: list[TimingEvidenceBundle] = []

        def shadow_facade(
            evidence: TimingEvidenceBundle,
            *,
            v2_fallback_fit: TimingFitResult,
            mode: str,
            max_supported_audio_duration_seconds: float,
        ):
            captured.append(evidence)
            return run_timing_v3_shadow(
                evidence,
                v2_fallback_fit=v2_fallback_fit,
                mode=mode,  # type: ignore[arg-type]
                max_supported_audio_duration_seconds=max_supported_audio_duration_seconds,
                candidate_generator=lambda prediction, *, audio_evidence=None: _accepted_tempo_result(),
            )

        runtime = SessionRuntime(
            session_id="s1",
            model_runtime=_fake_model_runtime(_FakeTimingProvider(_prediction(frame_count=200))),
            config=SessionRuntimeConfig(
                minimum_frame_count=4,
                timing_mode="v3_shadow",
                timing_max_supported_audio_duration_seconds=600.0,
            ),
            mel_loader=_fake_mel_loader(mel),
            grid_fitter=fitter,
            timing_v3_facade=shadow_facade,
        )

        cache = runtime.prepare_audio("song.wav", audio_length_ms=4_000)

        self.assertEqual(len(captured), 1)
        evidence = captured[0]
        self.assertIs(evidence.beatthis_frame_probabilities, cache.beatthis_prediction)
        self.assertIsNotNone(evidence.raw_audio_log_mel_10ms)
        assert evidence.raw_audio_log_mel_10ms is not None
        self.assertEqual(evidence.raw_audio_log_mel_10ms.shape, (6, 80))
        self.assertTrue(np.array_equal(evidence.raw_audio_log_mel_10ms[0], mel[0, :80]))
        self.assertTrue(np.array_equal(evidence.raw_audio_log_mel_10ms[1], mel[0, 80:]))
        self.assertIs(cache.timing_grid, fitter.result.grid)
        self.assertIs(cache.timing_v3_outcome.live_timing_grid, fitter.result.grid)
        self.assertEqual(cache.timing_v3_outcome.telemetry.status, "completed")
        self.assertIsNotNone(cache.timing_v3_outcome.selected_curve_canonical_bytes)
        self.assertEqual(cache.timing_v3_outcome.to_observable_dict()["live_timing"], "v2")

    def test_prepare_audio_rejects_failed_v3_shadow_outcome(self) -> None:
        fitter = _FakeGridFitter()

        def failed_facade(
            evidence: TimingEvidenceBundle,
            *,
            v2_fallback_fit: TimingFitResult,
            mode: str,
            max_supported_audio_duration_seconds: float,
        ) -> TimingV3Outcome:
            del evidence, max_supported_audio_duration_seconds
            return TimingV3Outcome(
                mode=mode,  # type: ignore[arg-type]
                v2_fallback_fit=v2_fallback_fit,
                telemetry=TimingV3Telemetry(
                    mode=mode,  # type: ignore[arg-type]
                    status="failed",
                    elapsed_ms=1.0,
                    candidate_count=0,
                    selection_status=None,
                    fallback_reason="timing_v3_shadow_failed",
                    selected_fingerprint_sha256=None,
                    error_type="RuntimeError",
                    error_message="fixture failure",
                ),
            )

        runtime = SessionRuntime(
            session_id="s1",
            model_runtime=_fake_model_runtime(_FakeTimingProvider(_prediction(frame_count=200))),
            config=SessionRuntimeConfig(
                minimum_frame_count=4,
                timing_mode="v3_shadow",
            ),
            mel_loader=_fake_mel_loader(np.zeros((3, 160), dtype=np.float32)),
            grid_fitter=fitter,
            timing_v3_facade=failed_facade,
        )

        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            runtime.prepare_audio("song.wav", audio_length_ms=4_000)
        self.assertIsNone(runtime.audio_cache)

    def test_prepare_mapper_window_can_skip_control_attention_cache(self) -> None:
        mapper_model = _FakeMapperModel(control_dim=3, d_model=5)
        mapper_model.control_attention_kv_cache = None  # type: ignore[method-assign]
        runtime = SessionRuntime(
            session_id="s1",
            model_runtime=_fake_model_runtime(
                _FakeTimingProvider(_prediction()),
                control_model=_FakeControlModel(control_dim=3),
                mapper_model=mapper_model,
            ),
            config=SessionRuntimeConfig(minimum_frame_count=4),
            mel_loader=_fake_mel_loader(np.zeros((500, 160), dtype=np.float32)),
            grid_fitter=_FakeGridFitter(),
        )
        runtime.prepare_audio("song.wav", audio_length_ms=10_000)

        window_cache = runtime.prepare_mapper_window(start_ms=0, include_control_attention_kv_cache=False)

        self.assertIsNone(window_cache.control_attention_kv_cache)
        self.assertNotIn("control_attention_kv_cache", window_cache.as_model_batch())

    def test_prepare_control_requires_audio_cache(self) -> None:
        runtime = SessionRuntime(
            session_id="s1",
            model_runtime=_fake_model_runtime(_FakeTimingProvider(_prediction())),
            config=SessionRuntimeConfig(minimum_frame_count=4),
            mel_loader=_fake_mel_loader(np.zeros((5, 160), dtype=np.float32)),
            grid_fitter=_FakeGridFitter(),
        )

        with self.assertRaisesRegex(RuntimeError, "prepare_audio"):
            runtime.prepare_control()

    def test_prepare_control_rejects_unaligned_start_ms(self) -> None:
        runtime = SessionRuntime(
            session_id="s1",
            model_runtime=_fake_model_runtime(_FakeTimingProvider(_prediction())),
            config=SessionRuntimeConfig(minimum_frame_count=4),
            mel_loader=_fake_mel_loader(np.zeros((5, 160), dtype=np.float32)),
            grid_fitter=_FakeGridFitter(),
        )
        runtime.prepare_audio("song.wav", audio_length_ms=100)

        with self.assertRaisesRegex(ValueError, "divisible"):
            runtime.prepare_control(start_ms=21)

    def test_prepare_audio_rejects_invalid_mel_shape(self) -> None:
        runtime = SessionRuntime(
            session_id="s1",
            model_runtime=_fake_model_runtime(_FakeTimingProvider(_prediction())),
            config=SessionRuntimeConfig(minimum_frame_count=4),
            mel_loader=_fake_mel_loader(np.zeros((5, 159), dtype=np.float32)),
            grid_fitter=_FakeGridFitter(),
        )

        with self.assertRaisesRegex(ValueError, "packed_mel"):
            runtime.prepare_audio("song.wav", audio_length_ms=100)


def _prediction(*, frame_count: int = 512, source_path: Path | str = "song.wav") -> FrameTimingPrediction:
    frame_indexes = np.arange(frame_count, dtype=np.float32)
    beat_prob = (np.sin(frame_indexes / 8.0) * 0.25 + 0.5).astype(np.float32)
    downbeat_prob = (np.cos(frame_indexes / 32.0) * 0.25 + 0.5).astype(np.float32)
    return FrameTimingPrediction(
        provider="fake",
        checkpoint_path="fake-checkpoint",
        source_path=Path(source_path).as_posix(),
        beat_prob=beat_prob,
        downbeat_prob=downbeat_prob,
        frame_rate_hz=50.0,
    )


def _fake_mel_loader(mel: np.ndarray):
    def load(audio_path: str | Path) -> np.ndarray:
        del audio_path
        return mel

    return load


def _fake_model_runtime(
    provider: _FakeTimingProvider,
    *,
    control_model: torch.nn.Module | None = None,
    mapper_model: torch.nn.Module | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        device=torch.device("cpu"),
        beatthis_provider=provider,
        control_model=_FakeControlModel() if control_model is None else control_model,
        mapper_model=_FakeMapperModel() if mapper_model is None else mapper_model,
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


class _FakeTimingProvider:
    def __init__(self, prediction: FrameTimingPrediction) -> None:
        self.prediction = prediction
        self.paths: list[Path] = []

    def predict_file(self, audio_path: str | Path) -> FrameTimingPrediction:
        self.paths.append(Path(audio_path))
        return self.prediction


class _FakeControlModel(torch.nn.Module):
    def __init__(self, control_dim: int = 4) -> None:
        super().__init__()
        self.control_dim = int(control_dim)
        self.calls: list[dict[str, object]] = []

    def forward(
        self,
        *,
        context_mel: torch.Tensor,
        context_dense_timing_v2: torch.Tensor,
        normalized_difficulty: torch.Tensor,
        context_padding_mask: torch.Tensor,
        target_start_frame: torch.Tensor | None = None,
        **kwargs: object,
    ) -> SimpleNamespace:
        del context_dense_timing_v2, normalized_difficulty, context_padding_mask, kwargs
        batch_size = int(context_mel.shape[0])
        frames = int(context_mel.shape[1])
        if target_start_frame is None:
            start_values = torch.zeros(batch_size, dtype=torch.float32, device=context_mel.device)
            recorded_starts: list[int] | None = None
        else:
            start_values = target_start_frame.to(device=context_mel.device, dtype=torch.float32).reshape(batch_size)
            recorded_starts = [int(value) for value in target_start_frame.detach().cpu().reshape(-1).tolist()]
        self.calls.append(
            {
                "target_start_frame": recorded_starts,
                "grad_enabled": torch.is_grad_enabled(),
                "inference_mode": torch.is_inference_mode_enabled(),
            },
        )
        control_memory = torch.zeros(
            batch_size,
            frames,
            self.control_dim,
            dtype=context_mel.dtype,
            device=context_mel.device,
        )
        control_memory[:, :, 0] = start_values.reshape(batch_size, 1)
        value_pred = start_values.reshape(batch_size, 1, 1).expand(batch_size, 100, 1).to(dtype=context_mel.dtype)
        return SimpleNamespace(control_memory=control_memory, value_pred=value_pred)


class _FakeMapperModel(torch.nn.Module):
    def __init__(self, *, control_dim: int = 4, d_model: int = 4) -> None:
        super().__init__()
        self.control_projection = torch.nn.Linear(control_dim, d_model, bias=False)
        with torch.no_grad():
            self.control_projection.weight.zero_()
            for index in range(min(control_dim, d_model)):
                self.control_projection.weight[index, index] = 1.0
        self.d_model = int(d_model)
        self.global_calls: list[dict[str, object]] = []
        self.kv_calls: list[dict[str, object]] = []
        self.control_kv_calls: list[dict[str, object]] = []

    def _global_context_memory(
        self,
        *,
        batch: dict[str, torch.Tensor],
        device: torch.device,
        batch_size: int,
        write_start_ms: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del write_start_ms
        target_start_frame = batch["target_start_frame"]
        self.global_calls.append(
            {
                "target_start_frame": [int(value) for value in target_start_frame.detach().cpu().reshape(-1).tolist()],
                "grad_enabled": torch.is_grad_enabled(),
                "inference_mode": torch.is_inference_mode_enabled(),
            },
        )
        memory = torch.ones(batch_size, 7, self.d_model, dtype=torch.float32, device=device)
        padding_mask = torch.zeros(batch_size, 7, dtype=torch.bool, device=device)
        position_features = torch.tensor([[0.0, 0.5, 1.0, 1.5]], dtype=torch.float32, device=device).expand(
            batch_size,
            -1,
        )
        return memory, padding_mask, position_features

    def global_attention_kv_cache(self, global_memory: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        self.kv_calls.append(
            {
                "grad_enabled": torch.is_grad_enabled(),
                "inference_mode": torch.is_inference_mode_enabled(),
            },
        )
        key = global_memory.reshape(global_memory.shape[0], 1, global_memory.shape[1], global_memory.shape[2])
        value = key + 1.0
        return ((key, value), (key + 2.0, value + 2.0))

    def control_attention_kv_cache(
        self,
        projected_control_memory_8s: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        self.control_kv_calls.append(
            {
                "grad_enabled": torch.is_grad_enabled(),
                "inference_mode": torch.is_inference_mode_enabled(),
            },
        )
        key = projected_control_memory_8s.reshape(
            projected_control_memory_8s.shape[0],
            1,
            projected_control_memory_8s.shape[1],
            projected_control_memory_8s.shape[2],
        )
        value = key + 1.0
        return ((key, value), (key + 2.0, value + 2.0))


class _FakeGridFitter:
    def __init__(self) -> None:
        self.predictions: list[FrameTimingPrediction] = []
        self.result = TimingFitResult(
            grid=FittedTimingGrid((TimingSegment(offset_ms=0.0, beat_length_ms=500.0, meter=4),)),
            score=0.99,
            diagnostics=TimingFitDiagnostics(
                fit_score=0.99,
                selected_period_frames=25.0,
                selected_offset_frames=0.0,
                selected_bpm=120.0,
                candidate_count=1,
                half_tempo_score=0.0,
                double_tempo_score=0.0,
                raw_selected_bpm=120.0,
                raw_score=0.99,
                tempo_multiplier=1.0,
                segment_alias_switch_count=0,
                tempo_multiplier_distribution={"1.0": 1},
            ),
        )

    def fit(self, prediction: FrameTimingPrediction) -> TimingFitResult:
        self.predictions.append(prediction)
        return self.result


if __name__ == "__main__":
    unittest.main()
