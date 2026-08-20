from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from pulsefield_model.features.mel import load_full_song_packed_mel_20ms
from pulsefield_model.inference.config import (
    InferenceServiceConfig,
    project_to_ws_endpoint_config,
)
from pulsefield_model.inference.hydra_entry import compose_inference_service_config
from pulsefield_model.inference.model_bundles.timing_mock import (
    _grid_fitter_config_for_canonicalization,
)
from pulsefield_model.timing.grid_fitting import GridFitter, TimingFitResult
from pulsefield_model.timing.providers.beatthis import BeatThisTimingProvider
from pulsefield_model.timing.providers.beatthis_cache import (
    BeatThisFramePredictionCacheConfig,
    load_beatthis_frame_prediction_cache,
)
from pulsefield_model.timing.schema import FrameTimingPrediction
from pulsefield_model.timing.v3.inference import (
    TIMING_MODE_V3_SHADOW,
    TimingEvidenceBundle,
    TimingInferenceMode,
    TimingV3Facade,
    TimingV3Outcome,
    run_timing_v3_shadow,
    unpack_packed_mel_20ms_to_log_mel_10ms,
)


FULL5050_SHADOW_RESULT_SCHEMA = "pulsefield_model.timing_v3_full5050_shadow_result_v1"
FULL5050_SHADOW_SUMMARY_SCHEMA = "pulsefield_model.timing_v3_full5050_shadow_summary_v1"
FULL5050_SHADOW_PLAN_SCHEMA = "pulsefield_model.timing_v3_full5050_shadow_plan_v1"
DEFAULT_FULL5050_LABELS_PATH = Path("artifacts/reports/timing/timing_v3_labels_v1.jsonl")
DEFAULT_FULL5050_OUTPUT_JSONL = Path("artifacts/reports/timing/timing_v3_full5050_shadow_results.jsonl")
DEFAULT_EXPECTED_FULL5050_ROW_COUNT = 5050

_FINAL_ROW_STATUSES = frozenset({"completed", "failed", "skipped_duration"})
_TIMING_STAGE_NAMES = ("provider", "mel", "v2", "v3", "total")


class TimingPredictionProvider(Protocol):
    def predict_file(self, audio_path: str | Path) -> FrameTimingPrediction:
        ...


class TimingGridFitter(Protocol):
    def fit(self, prediction: FrameTimingPrediction) -> TimingFitResult:
        ...


PackedMelLoader = Callable[..., NDArray[np.float32]]
BeatThisCacheLoader = Callable[
    [str, BeatThisFramePredictionCacheConfig],
    FrameTimingPrediction | None,
]
Clock = Callable[[], float]


class Full5050ShadowRunnerError(RuntimeError):
    pass


class MissingBeatThisCacheError(Full5050ShadowRunnerError):
    pass


@dataclass(frozen=True)
class Full5050LocatorRow:
    """Allowed locator-only projection from the full5050 manifest.

    The input manifest contains mapper/redline/label fields. This projection
    intentionally keeps only row identity, audio path, BeatThis cache key,
    duration, and source cache status.
    """

    row_index: int
    resolved_audio_path: Path
    beatthis_audio_cache_key: str | None
    duration_seconds: float
    input_status: str | None

    @property
    def row_id(self) -> str:
        return f"full5050:{self.row_index}"

    @property
    def audio_length_ms(self) -> int:
        return int(round(1000.0 * self.duration_seconds))


@dataclass(frozen=True)
class Full5050ShadowRunnerConfig:
    labels_path: Path = DEFAULT_FULL5050_LABELS_PATH
    output_jsonl: Path = DEFAULT_FULL5050_OUTPUT_JSONL
    summary_json: Path | None = None
    expected_row_count: int = DEFAULT_EXPECTED_FULL5050_ROW_COUNT
    allow_beatthis_provider_fallback: bool = False
    retry_failed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels_path", Path(self.labels_path))
        object.__setattr__(self, "output_jsonl", Path(self.output_jsonl))
        if self.summary_json is not None:
            object.__setattr__(self, "summary_json", Path(self.summary_json))
        if isinstance(self.expected_row_count, bool) or not isinstance(self.expected_row_count, int):
            raise TypeError("expected_row_count must be an integer")
        if self.expected_row_count <= 0:
            raise ValueError("expected_row_count must be positive")
        if not isinstance(self.allow_beatthis_provider_fallback, bool):
            raise TypeError("allow_beatthis_provider_fallback must be a boolean")
        if not isinstance(self.retry_failed, bool):
            raise TypeError("retry_failed must be a boolean")


@dataclass
class Full5050ShadowPipeline:
    timing_mode: TimingInferenceMode
    max_supported_audio_duration_seconds: float
    beatthis_cache_config: BeatThisFramePredictionCacheConfig
    allow_beatthis_provider_fallback: bool
    grid_fitter: TimingGridFitter
    mel_loader: PackedMelLoader = load_full_song_packed_mel_20ms
    beatthis_cache_loader: BeatThisCacheLoader = load_beatthis_frame_prediction_cache
    beatthis_provider_factory: Callable[[], TimingPredictionProvider] | None = None
    timing_v3_facade: TimingV3Facade = run_timing_v3_shadow
    clock: Clock = time.perf_counter
    _beatthis_provider: TimingPredictionProvider | None = field(default=None, init=False, repr=False)

    def beatthis_provider(self) -> TimingPredictionProvider:
        if self.beatthis_provider_factory is None:
            raise MissingBeatThisCacheError("BeatThis provider fallback is disabled")
        if self._beatthis_provider is None:
            self._beatthis_provider = self.beatthis_provider_factory()
        return self._beatthis_provider


def compose_full5050_shadow_inference_config(
    overrides: Sequence[str] | None = None,
) -> InferenceServiceConfig:
    resolved_overrides = ["timing=v3_shadow", *(overrides or ())]
    config = compose_inference_service_config(resolved_overrides)
    if config.timing.mode != TIMING_MODE_V3_SHADOW:
        raise ValueError("full5050 Timing-v3 shadow runner requires timing.mode=v3_shadow")
    project_to_ws_endpoint_config(config)
    return config


def build_full5050_shadow_pipeline(
    config: InferenceServiceConfig,
    *,
    allow_beatthis_provider_fallback: bool = False,
) -> Full5050ShadowPipeline:
    endpoint_config = project_to_ws_endpoint_config(config)
    if endpoint_config.timing_mode != TIMING_MODE_V3_SHADOW:
        raise ValueError("full5050 Timing-v3 shadow runner requires timing_mode=v3_shadow")
    cache_config = BeatThisFramePredictionCacheConfig(
        checkpoint_path=str(endpoint_config.beatthis_checkpoint),
        float16=bool(endpoint_config.beatthis_float16),
    )
    return Full5050ShadowPipeline(
        timing_mode=endpoint_config.timing_mode,  # type: ignore[arg-type]
        max_supported_audio_duration_seconds=float(
            endpoint_config.timing_max_supported_audio_duration_seconds,
        ),
        beatthis_cache_config=cache_config,
        allow_beatthis_provider_fallback=bool(allow_beatthis_provider_fallback),
        grid_fitter=GridFitter(
            _grid_fitter_config_for_canonicalization(endpoint_config.canonicalization),
        ),
        beatthis_provider_factory=lambda: BeatThisTimingProvider(
            checkpoint_path=str(endpoint_config.beatthis_checkpoint),
            device=str(endpoint_config.beatthis_device or "cpu"),
            float16=bool(endpoint_config.beatthis_float16),
        ),
    )


def load_full5050_locator_rows(
    labels_path: str | Path = DEFAULT_FULL5050_LABELS_PATH,
    *,
    expected_row_count: int = DEFAULT_EXPECTED_FULL5050_ROW_COUNT,
) -> tuple[Full5050LocatorRow, ...]:
    path = Path(labels_path)
    if not path.is_file():
        raise FileNotFoundError(f"full5050 labels file not found: {path}")
    if isinstance(expected_row_count, bool) or not isinstance(expected_row_count, int):
        raise TypeError("expected_row_count must be an integer")
    if expected_row_count <= 0:
        raise ValueError("expected_row_count must be positive")

    rows: list[Full5050LocatorRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ValueError(f"full5050 row {row_index} must be a JSON object")
            rows.append(_locator_row_from_mapping(raw, row_index=row_index))

    if len(rows) != expected_row_count:
        raise ValueError(
            f"full5050 manifest must contain exactly {expected_row_count} rows, got {len(rows)}",
        )
    return tuple(rows)


def run_full5050_shadow(
    config: Full5050ShadowRunnerConfig,
    *,
    inference_config: InferenceServiceConfig,
    pipeline: Full5050ShadowPipeline | None = None,
) -> dict[str, object]:
    rows = load_full5050_locator_rows(
        config.labels_path,
        expected_row_count=config.expected_row_count,
    )
    active_pipeline = pipeline or build_full5050_shadow_pipeline(
        inference_config,
        allow_beatthis_provider_fallback=config.allow_beatthis_provider_fallback,
    )
    existing_statuses = _read_existing_result_statuses(
        config.output_jsonl,
    )
    unexpected_indexes = sorted(set(existing_statuses) - set(range(len(rows))))
    if unexpected_indexes:
        raise ValueError(
            f"existing results contain out-of-range row indexes: {unexpected_indexes[:8]}",
        )
    completed = {
        row_index
        for row_index, status in existing_statuses.items()
        if status in _FINAL_ROW_STATUSES
        and not (status == "failed" and config.retry_failed)
    }
    started = active_pipeline.clock()
    counts = {"completed": 0, "failed": 0, "skipped_duration": 0, "resumed": 0}
    config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with config.output_jsonl.open("a", encoding="utf-8") as output:
        for row in rows:
            if row.row_index in completed:
                counts["resumed"] += 1
                continue
            result = run_full5050_shadow_row(row, active_pipeline)
            status = str(result["status"])
            if status in counts:
                counts[status] += 1
            output.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
            output.flush()

    final_statuses = _read_existing_result_statuses(config.output_jsonl)
    unexpected_indexes = sorted(set(final_statuses) - set(range(len(rows))))
    if unexpected_indexes:
        raise ValueError(
            f"results contain out-of-range row indexes: {unexpected_indexes[:8]}",
        )
    summary = {
        "schema": FULL5050_SHADOW_SUMMARY_SCHEMA,
        "labels_path": config.labels_path.as_posix(),
        "output_jsonl": config.output_jsonl.as_posix(),
        "timing_mode": active_pipeline.timing_mode,
        "expected_rows": config.expected_row_count,
        "total_rows": len(rows),
        "completed_rows": sum(status == "completed" for status in final_statuses.values()),
        "failed_rows": sum(status == "failed" for status in final_statuses.values()),
        "skipped_duration_rows": sum(
            status == "skipped_duration" for status in final_statuses.values()
        ),
        "recorded_rows": len(final_statuses),
        "resumed_rows": counts["resumed"],
        "attempted_rows": len(rows) - counts["resumed"],
        "elapsed_ms": _elapsed_ms(started, active_pipeline.clock),
    }
    if config.summary_json is not None:
        config.summary_json.parent.mkdir(parents=True, exist_ok=True)
        config.summary_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def run_full5050_shadow_row(
    row: Full5050LocatorRow,
    pipeline: Full5050ShadowPipeline,
) -> dict[str, object]:
    timings: dict[str, float | None] = {stage: None for stage in _TIMING_STAGE_NAMES}
    total_started = pipeline.clock()
    stage = "provider"
    beatthis_source: str | None = None
    prediction: FrameTimingPrediction | None = None
    outcome: TimingV3Outcome | None = None
    try:
        if row.input_status is not None and row.input_status != "valid":
            raise ValueError(f"input cache status is not valid: {row.input_status}")

        stage_started = pipeline.clock()
        prediction, beatthis_source = _load_prediction(row, pipeline)
        timings["provider"] = _elapsed_ms(stage_started, pipeline.clock)

        stage = "mel"
        stage_started = pipeline.clock()
        packed_mel = _load_packed_mel(row, pipeline)
        raw_audio_log_mel_10ms = unpack_packed_mel_20ms_to_log_mel_10ms(packed_mel)
        timings["mel"] = _elapsed_ms(stage_started, pipeline.clock)

        stage = "v2"
        stage_started = pipeline.clock()
        fit_result = pipeline.grid_fitter.fit(prediction)
        timings["v2"] = _elapsed_ms(stage_started, pipeline.clock)

        stage = "v3"
        stage_started = pipeline.clock()
        outcome = pipeline.timing_v3_facade(
            TimingEvidenceBundle(
                beatthis_frame_probabilities=prediction,
                audio_duration_seconds=row.duration_seconds,
                raw_audio_log_mel_10ms=raw_audio_log_mel_10ms,
            ),
            v2_fallback_fit=fit_result,
            mode=pipeline.timing_mode,
            max_supported_audio_duration_seconds=pipeline.max_supported_audio_duration_seconds,
        )
        if not isinstance(outcome, TimingV3Outcome):
            raise TypeError("timing_v3_facade must return TimingV3Outcome")
        if outcome.v2_fallback_fit is not fit_result:
            raise ValueError("Timing-v3 outcome must retain the current v2 fit as fallback")
        if outcome.telemetry.status not in {"completed", "skipped_duration"}:
            raise RuntimeError(
                f"unexpected Timing-v3 shadow status: {outcome.telemetry.status}",
            )
        timings["v3"] = _elapsed_ms(stage_started, pipeline.clock)
        timings["total"] = _elapsed_ms(total_started, pipeline.clock)
        return _success_row_payload(
            row,
            prediction=prediction,
            beatthis_source=beatthis_source,
            outcome=outcome,
            timings=timings,
        )
    except Exception as exc:  # noqa: BLE001 - row-level failure is the resumable unit.
        timings["total"] = _elapsed_ms(total_started, pipeline.clock)
        return _failure_row_payload(
            row,
            stage=stage,
            beatthis_source=beatthis_source,
            prediction=prediction,
            outcome=outcome,
            timings=timings,
            error=exc,
        )


def plan_full5050_shadow_run(
    config: Full5050ShadowRunnerConfig,
    *,
    inference_config: InferenceServiceConfig,
) -> dict[str, object]:
    rows = load_full5050_locator_rows(
        config.labels_path,
        expected_row_count=config.expected_row_count,
    )
    endpoint_config = project_to_ws_endpoint_config(inference_config)
    return {
        "schema": FULL5050_SHADOW_PLAN_SCHEMA,
        "action": "plan_only",
        "labels_path": config.labels_path.as_posix(),
        "output_jsonl": config.output_jsonl.as_posix(),
        "timing_mode": endpoint_config.timing_mode,
        "max_supported_audio_duration_seconds": (
            endpoint_config.timing_max_supported_audio_duration_seconds
        ),
        "allow_beatthis_provider_fallback": config.allow_beatthis_provider_fallback,
        "expected_rows": config.expected_row_count,
        "total_rows": len(rows),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    runner_config = Full5050ShadowRunnerConfig(
        labels_path=args.labels_path,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        expected_row_count=args.expected_row_count,
        allow_beatthis_provider_fallback=args.allow_beatthis_provider_fallback,
        retry_failed=args.retry_failed,
    )
    inference_config = compose_full5050_shadow_inference_config(args.hydra_override)
    if not args.run:
        print(
            json.dumps(
                plan_full5050_shadow_run(
                    runner_config,
                    inference_config=inference_config,
                ),
                indent=2,
                sort_keys=True,
            ),
        )
        return 0
    summary = run_full5050_shadow(
        runner_config,
        inference_config=inference_config,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _locator_row_from_mapping(raw: Mapping[str, object], *, row_index: int) -> Full5050LocatorRow:
    source = raw.get("source")
    if not isinstance(source, Mapping):
        raise ValueError(f"full5050 row {row_index} source must be an object")

    audio_path_raw = raw.get("resolved_audio_path")
    if not isinstance(audio_path_raw, str) or not audio_path_raw:
        raise ValueError(f"full5050 row {row_index} resolved_audio_path must be a non-empty string")

    cache_key_raw = source.get("cache_audio_key")
    if cache_key_raw is not None and not isinstance(cache_key_raw, str):
        raise ValueError(f"full5050 row {row_index} source.cache_audio_key must be a string or null")

    duration_raw = source.get("cache_duration_seconds")
    if isinstance(duration_raw, bool) or not isinstance(duration_raw, (int, float)):
        raise ValueError(f"full5050 row {row_index} source.cache_duration_seconds must be numeric")
    duration_seconds = float(duration_raw)
    if not np.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError(f"full5050 row {row_index} duration must be positive and finite")

    status_raw = source.get("cache_status")
    if status_raw is not None and not isinstance(status_raw, str):
        raise ValueError(f"full5050 row {row_index} source.cache_status must be a string or null")

    return Full5050LocatorRow(
        row_index=row_index,
        resolved_audio_path=Path(audio_path_raw),
        beatthis_audio_cache_key=cache_key_raw,
        duration_seconds=duration_seconds,
        input_status=status_raw,
    )


def _load_prediction(
    row: Full5050LocatorRow,
    pipeline: Full5050ShadowPipeline,
) -> tuple[FrameTimingPrediction, str]:
    if row.beatthis_audio_cache_key is not None:
        prediction = pipeline.beatthis_cache_loader(
            row.beatthis_audio_cache_key,
            pipeline.beatthis_cache_config,
        )
        if prediction is not None:
            return prediction, "cache"
    if not pipeline.allow_beatthis_provider_fallback:
        raise MissingBeatThisCacheError(
            f"BeatThis cache is missing for {row.row_id}; provider fallback is disabled",
        )
    with_provider = pipeline.beatthis_provider().predict_file(row.resolved_audio_path)
    return with_provider, "provider"


def _load_packed_mel(
    row: Full5050LocatorRow,
    pipeline: Full5050ShadowPipeline,
) -> NDArray[np.float32]:
    packed = pipeline.mel_loader(
        row.resolved_audio_path,
        audio_cache_key=row.beatthis_audio_cache_key,
    )
    array = np.asarray(packed, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 160:
        raise ValueError(f"packed mel must have shape [frames, 160], got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("packed mel must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float32)


def _success_row_payload(
    row: Full5050LocatorRow,
    *,
    prediction: FrameTimingPrediction,
    beatthis_source: str,
    outcome: TimingV3Outcome,
    timings: Mapping[str, float | None],
) -> dict[str, object]:
    telemetry = outcome.telemetry
    selected_curve = outcome.selected_shadow_curve
    if telemetry.status == "completed":
        row_status = "completed"
        reason = telemetry.fallback_reason
    elif telemetry.status == "skipped_duration":
        row_status = "skipped_duration"
        reason = telemetry.fallback_reason
    else:
        row_status = telemetry.status
        reason = telemetry.fallback_reason
    return {
        **_base_row_payload(row),
        "status": row_status,
        "reason": reason,
        "beatthis": {
            "source": beatthis_source,
            "provider": prediction.provider,
            "checkpoint_path": prediction.checkpoint_path,
            "frame_count": prediction.frame_count,
            "frame_rate_hz": float(prediction.frame_rate_hz),
        },
        "timings_ms": _timings_payload(timings),
        "v3": {
            "telemetry_status": telemetry.status,
            "selection_status": telemetry.selection_status,
            "fallback_reason": telemetry.fallback_reason,
            "candidate_count": int(telemetry.candidate_count),
            "curve_class": None if selected_curve is None else selected_curve.curve_class,
            "canonical_curve_roundtrip": outcome.selected_curve_canonical_bytes is not None,
        },
    }


def _failure_row_payload(
    row: Full5050LocatorRow,
    *,
    stage: str,
    beatthis_source: str | None,
    prediction: FrameTimingPrediction | None,
    outcome: TimingV3Outcome | None,
    timings: Mapping[str, float | None],
    error: Exception,
) -> dict[str, object]:
    selected_curve = None if outcome is None else outcome.selected_shadow_curve
    return {
        **_base_row_payload(row),
        "status": "failed",
        "reason": f"{stage}_failed",
        "beatthis": {
            "source": beatthis_source,
            "provider": None if prediction is None else prediction.provider,
            "checkpoint_path": None if prediction is None else prediction.checkpoint_path,
            "frame_count": None if prediction is None else prediction.frame_count,
            "frame_rate_hz": None if prediction is None else float(prediction.frame_rate_hz),
        },
        "timings_ms": _timings_payload(timings),
        "v3": {
            "telemetry_status": None if outcome is None else outcome.telemetry.status,
            "selection_status": None if outcome is None else outcome.telemetry.selection_status,
            "fallback_reason": None if outcome is None else outcome.telemetry.fallback_reason,
            "candidate_count": None if outcome is None else int(outcome.telemetry.candidate_count),
            "curve_class": None if selected_curve is None else selected_curve.curve_class,
            "canonical_curve_roundtrip": (
                False if outcome is None else outcome.selected_curve_canonical_bytes is not None
            ),
        },
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def _base_row_payload(row: Full5050LocatorRow) -> dict[str, object]:
    return {
        "schema": FULL5050_SHADOW_RESULT_SCHEMA,
        "row_index": row.row_index,
        "row_id": row.row_id,
        "resolved_audio_path": row.resolved_audio_path.as_posix(),
        "duration_seconds": row.duration_seconds,
        "audio_length_ms": row.audio_length_ms,
        "input_status": row.input_status,
    }


def _timings_payload(timings: Mapping[str, float | None]) -> dict[str, float | None]:
    return {stage: timings.get(stage) for stage in _TIMING_STAGE_NAMES}


def _read_existing_result_statuses(output_jsonl: Path) -> dict[int, str]:
    if not output_jsonl.exists():
        return {}
    statuses: dict[int, str] = {}
    with output_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"existing result line {line_number} must be a JSON object")
            row_index = payload.get("row_index")
            status = payload.get("status")
            if isinstance(row_index, bool) or not isinstance(row_index, int):
                raise ValueError(f"existing result line {line_number} row_index must be an integer")
            if status not in _FINAL_ROW_STATUSES:
                continue
            statuses[row_index] = status
    return statuses


def _elapsed_ms(started: float, clock: Clock) -> float:
    return max(0.0, 1000.0 * (float(clock()) - float(started)))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or run the exact real5050 Timing-v3 production-shadow batch. "
            "The default is plan-only; pass --run to write per-row results."
        ),
    )
    parser.add_argument("--run", action="store_true", help="execute rows and write JSONL results")
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_FULL5050_LABELS_PATH)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_FULL5050_OUTPUT_JSONL)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--expected-row-count", type=int, default=DEFAULT_EXPECTED_FULL5050_ROW_COUNT)
    parser.add_argument(
        "--allow-beatthis-provider-fallback",
        action="store_true",
        help="run BeatThis provider when an explicit frame-prediction cache entry is missing",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="rerun rows already recorded with status=failed in the output JSONL",
    )
    parser.add_argument(
        "--hydra-override",
        action="append",
        default=[],
        help="additional packaged inference Hydra override, e.g. runtime.beatthis_device=mps",
    )
    return parser.parse_args(argv)


__all__ = [
    "DEFAULT_EXPECTED_FULL5050_ROW_COUNT",
    "DEFAULT_FULL5050_LABELS_PATH",
    "DEFAULT_FULL5050_OUTPUT_JSONL",
    "FULL5050_SHADOW_PLAN_SCHEMA",
    "FULL5050_SHADOW_RESULT_SCHEMA",
    "FULL5050_SHADOW_SUMMARY_SCHEMA",
    "Full5050LocatorRow",
    "Full5050ShadowPipeline",
    "Full5050ShadowRunnerConfig",
    "MissingBeatThisCacheError",
    "build_full5050_shadow_pipeline",
    "compose_full5050_shadow_inference_config",
    "load_full5050_locator_rows",
    "main",
    "plan_full5050_shadow_run",
    "run_full5050_shadow",
    "run_full5050_shadow_row",
]


if __name__ == "__main__":
    raise SystemExit(main())
