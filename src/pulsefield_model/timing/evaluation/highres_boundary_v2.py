from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from pulsefield_model.features.mel_base import load_or_create_log_mel_cache
from pulsefield_model.timing.providers.beatthis import BeatThisTimingProvider
from pulsefield_model.timing.providers.beatthis_cache import (
    BeatThisFramePredictionCacheConfig,
    beatthis_audio_cache_key,
    load_beatthis_frame_prediction_cache,
    save_beatthis_frame_prediction_cache,
)
from pulsefield_model.timing.schema import FrameTimingPrediction
from pulsefield_model.timing.v3.audio_evidence import extract_raw_audio_evidence


EXPERIMENT_NAME = "timing_v3_exp029_highres_boundary_v2"
FEATURE_CONTRACT_ID = "timing_v3_highres_boundary_v2_100_local_relative_v1"
ROUTE_SCHEMA = "pulsefield_model.timing_v3_stable256_causal_opaque_v2"
TRUTH_SCHEMA = "pulsefield_model.timing_v3_stable256_causal_truth_v2"
PREDICTION_SCHEMA = "pulsefield_model.timing_v3_highres_boundary_v2_predictions_v1"
EVAL_SCHEMA = "pulsefield_model.timing_v3_highres_boundary_v2_eval_v1"

DEFAULT_CORPUS_ROOT = Path("artifacts/local/timing_v3/stable256_causal_v2")
DEFAULT_OUTPUT_ROOT = Path("artifacts/local/timing_v3/highres_boundary_v2")
DEFAULT_TRAIN_ROUTES = DEFAULT_CORPUS_ROOT / "opaque/train_routes.jsonl"
DEFAULT_HOLDOUT_ROUTES = DEFAULT_CORPUS_ROOT / "opaque/holdout_routes.jsonl"
DEFAULT_TRAIN_TRUTH = DEFAULT_CORPUS_ROOT / "truth/train_truth.jsonl"
DEFAULT_HOLDOUT_TRUTH = DEFAULT_CORPUS_ROOT / "truth/holdout_truth.jsonl.sealed"

SIGNAL_NAMES: tuple[str, ...] = (
    "beat",
    "downbeat",
    "abs_grad_beat",
    "abs_grad_downbeat",
    "raw_sum",
    "raw_max",
    "raw_low",
    "raw_mid",
    "raw_high",
    "abs_grad_raw_sum",
)
DERIVED_FEATURE_NAMES: tuple[str, ...] = (
    "center_mean",
    "center_max",
    "center_std",
    "near_mean_delta",
    "mid_mean_delta",
    "far_mean_delta",
    "near_abs_mean_delta",
    "mid_abs_mean_delta",
    "near_peak_delta",
    "mid_peak_delta",
)
FEATURE_NAMES: tuple[str, ...] = tuple(
    f"{signal}:{derived}"
    for signal in SIGNAL_NAMES
    for derived in DERIVED_FEATURE_NAMES
)
FORBIDDEN_FEATURE_TOKENS: frozenset[str] = frozenset(
    {
        "candidate",
        "duration",
        "id",
        "index",
        "label",
        "metadata",
        "osu",
        "path",
        "position",
        "ratio",
        "redline",
        "route",
        "seam",
        "seconds",
        "source",
        "split",
        "time",
        "truth",
    }
)
FROZEN_RATIOS: tuple[float, ...] = (0.8, 0.875, 1.125, 1.25)
EXPECTED_ROUTE_FIELDS = frozenset({"schema", "route_id", "audio_path"})


class HighresBoundaryV2Error(ValueError):
    pass


@dataclass(frozen=True)
class RouteRow:
    route_id: str
    audio_path: Path


@dataclass(frozen=True)
class TruthRow:
    route_id: str
    split: str
    transform: str
    ratio: float
    seam_seconds: float | None
    source_key: str


@dataclass(frozen=True)
class RunnerConfig:
    train_routes_jsonl: Path = DEFAULT_TRAIN_ROUTES
    holdout_routes_jsonl: Path = DEFAULT_HOLDOUT_ROUTES
    train_truth_jsonl: Path = DEFAULT_TRAIN_TRUTH
    holdout_truth_jsonl: Path = DEFAULT_HOLDOUT_TRUTH
    output_root: Path = DEFAULT_OUTPUT_ROOT
    route_schema: str = ROUTE_SCHEMA
    truth_schema: str = TRUTH_SCHEMA
    min_train_jump_routes: int = 128
    min_holdout_jump_routes: int = 128
    candidate_margin_seconds: float = 8.0
    candidate_hop_seconds: float = 0.1
    feature_sample_hop_seconds: float = 0.05
    train_epochs: int = 450
    train_batch_routes: int = 16
    seed: int = 20260816
    learning_rate: float = 1.5e-3
    weight_decay: float = 1.0e-3
    hidden_units: tuple[int, int] = (96, 48)
    beatthis_device: str = "auto"
    train_device: str = "auto"
    beatthis_cache_config: BeatThisFramePredictionCacheConfig = (
        BeatThisFramePredictionCacheConfig()
    )

    @property
    def feature_cache_root(self) -> Path:
        return self.output_root / "features_100_local_relative_v1"

    @property
    def model_path(self) -> Path:
        return self.output_root / "model.pt"

    @property
    def train_summary_path(self) -> Path:
        return self.output_root / "train_summary.json"

    @property
    def frozen_predictions_jsonl(self) -> Path:
        return self.output_root / "holdout_predictions.frozen.jsonl"

    @property
    def frozen_predictions_summary_json(self) -> Path:
        return self.output_root / "holdout_predictions.frozen.summary.json"

    @property
    def eval_summary_json(self) -> Path:
        return self.output_root / "holdout_eval.json"


@dataclass(frozen=True)
class RouteFeatureBundle:
    route_id: str
    candidates_seconds: NDArray[np.float64]
    features: NDArray[np.float32]


@dataclass(frozen=True)
class TrainExample:
    route_id: str
    candidates_seconds: NDArray[np.float64]
    features: NDArray[np.float32]
    target_index: int
    target_seconds: float


def feature_names() -> tuple[str, ...]:
    _validate_feature_contract()
    return FEATURE_NAMES


def self_check() -> dict[str, Any]:
    _validate_feature_contract()
    config = RunnerConfig()
    payload = {
        "experiment": EXPERIMENT_NAME,
        "feature_contract": FEATURE_CONTRACT_ID,
        "feature_count": len(FEATURE_NAMES),
        "signal_count": len(SIGNAL_NAMES),
        "derived_feature_count": len(DERIVED_FEATURE_NAMES),
        "candidate_margin_seconds": config.candidate_margin_seconds,
        "candidate_hop_seconds": config.candidate_hop_seconds,
        "feature_sample_hop_seconds": config.feature_sample_hop_seconds,
        "mlp_input_dim": len(FEATURE_NAMES),
        "mlp_hidden_units": list(config.hidden_units),
        "mlp_output_dim": 1,
        "train_epochs": config.train_epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "forbidden_feature_tokens": sorted(FORBIDDEN_FEATURE_TOKENS),
        "status": "pass",
    }
    return payload


def plan_run(config: RunnerConfig) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_NAME,
        "mode": "plan",
        "feature_contract": FEATURE_CONTRACT_ID,
        "train_routes_jsonl": config.train_routes_jsonl.as_posix(),
        "holdout_routes_jsonl": config.holdout_routes_jsonl.as_posix(),
        "train_truth_jsonl": config.train_truth_jsonl.as_posix(),
        "holdout_truth_jsonl": config.holdout_truth_jsonl.as_posix(),
        "output_root": config.output_root.as_posix(),
        "feature_cache_root": config.feature_cache_root.as_posix(),
        "model_path": config.model_path.as_posix(),
        "frozen_predictions_jsonl": config.frozen_predictions_jsonl.as_posix(),
        "guard": {
            "train_uses_holdout_truth": False,
            "predicts_all_holdout_routes_before_eval": True,
            "production_candidates_used": False,
            "absolute_time_features_used": False,
            "route_path_source_duration_ratio_label_features_used": False,
        },
        "run_command": (
            "uv run --extra mps python -m "
            "pulsefield_model.timing.evaluation.highres_boundary_v2 --run"
        ),
        "eval_command_after_truth_unsealed": (
            "uv run --extra mps python -m "
            "pulsefield_model.timing.evaluation.highres_boundary_v2 --evaluate"
        ),
    }


def preflight_run(config: RunnerConfig) -> dict[str, Any]:
    train_routes = load_route_rows(config.train_routes_jsonl, route_schema=config.route_schema)
    holdout_routes = load_route_rows(
        config.holdout_routes_jsonl,
        route_schema=config.route_schema,
    )
    train_truth = load_truth_rows(config.train_truth_jsonl, truth_schema=config.truth_schema)
    train_truth_by_id = _truth_by_route_id(train_truth)
    _require_route_truth_match(train_routes, train_truth_by_id, split_name="train")
    train_jumps = _jump_truth_rows(train_truth)
    if len(train_jumps) < config.min_train_jump_routes:
        raise HighresBoundaryV2Error(
            f"need at least {config.min_train_jump_routes} train jump routes, got {len(train_jumps)}"
        )
    train_jump_ratios = _ratio_counts(train_jumps)
    _require_ratio_buckets(train_jump_ratios, split_name="train")
    train_sources = {row.source_key for row in train_truth}
    return {
        "experiment": EXPERIMENT_NAME,
        "mode": "preflight",
        "train_route_count": len(train_routes),
        "holdout_route_count": len(holdout_routes),
        "train_truth_count": len(train_truth),
        "train_jump_count": len(train_jumps),
        "train_source_count": len(train_sources),
        "train_jump_ratio_counts": train_jump_ratios,
        "holdout_truth_opened": False,
        "status": "pass",
    }


def run_experiment(config: RunnerConfig, *, reuse_frozen_predictions: bool = False) -> dict[str, Any]:
    started_at = time.monotonic()
    preflight = preflight_run(config)
    train_routes = load_route_rows(config.train_routes_jsonl, route_schema=config.route_schema)
    holdout_routes = load_route_rows(
        config.holdout_routes_jsonl,
        route_schema=config.route_schema,
    )
    train_truth = load_truth_rows(config.train_truth_jsonl, truth_schema=config.truth_schema)
    train_truth_by_id = _truth_by_route_id(train_truth)
    train_route_by_id = {row.route_id: row for row in train_routes}
    train_jump_truth = _jump_truth_rows(train_truth)
    train_jump_ids = [row.route_id for row in train_jump_truth]

    config.output_root.mkdir(parents=True, exist_ok=True)
    beatthis_provider = _beatthis_provider(config)
    train_bundles: list[RouteFeatureBundle] = []
    for index, route_id in enumerate(train_jump_ids, start=1):
        bundle = load_or_extract_route_features(
            train_route_by_id[route_id],
            config=config,
            beatthis_provider=beatthis_provider,
        )
        train_bundles.append(bundle)
        if index % 32 == 0 or index == len(train_jump_ids):
            print(
                json.dumps(
                    {
                        "stage": "extract_train_jump",
                        "done": index,
                        "total": len(train_jump_ids),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    examples = [
        _train_example_from_bundle(bundle, train_truth_by_id[bundle.route_id])
        for bundle in train_bundles
    ]
    model, train_summary = train_boundary_model(examples, config=config)
    _save_model(model, train_summary, config=config)

    holdout_predictions_path = config.frozen_predictions_jsonl
    if holdout_predictions_path.exists():
        if not reuse_frozen_predictions:
            raise HighresBoundaryV2Error(
                f"frozen predictions already exist: {holdout_predictions_path}. "
                "Pass --reuse-frozen-predictions to keep them for evaluation; this runner will not overwrite."
            )
        predictions = load_frozen_predictions(holdout_predictions_path)
    else:
        predictions = predict_holdout_routes(
            holdout_routes,
            model,
            config=config,
            beatthis_provider=beatthis_provider,
        )
        write_frozen_predictions(
            predictions,
            output_jsonl=holdout_predictions_path,
            summary_json=config.frozen_predictions_summary_json,
            expected_route_count=len(holdout_routes),
        )

    elapsed = time.monotonic() - started_at
    summary = {
        "experiment": EXPERIMENT_NAME,
        "mode": "run",
        "preflight": preflight,
        "train": train_summary,
        "holdout_prediction_count": len(predictions),
        "frozen_predictions_jsonl": holdout_predictions_path.as_posix(),
        "model_path": config.model_path.as_posix(),
        "holdout_truth_opened": False,
        "elapsed_seconds": elapsed,
        "status": "predictions_frozen",
    }
    _atomic_write_json(config.output_root / "run_summary.json", summary, mode=0o644)
    return summary


def evaluate_frozen_holdout(config: RunnerConfig) -> dict[str, Any]:
    train_truth = load_truth_rows(config.train_truth_jsonl, truth_schema=config.truth_schema)
    holdout_truth = load_truth_rows(config.holdout_truth_jsonl, truth_schema=config.truth_schema)
    holdout_routes = load_route_rows(
        config.holdout_routes_jsonl,
        route_schema=config.route_schema,
    )
    predictions = load_frozen_predictions(config.frozen_predictions_jsonl)
    holdout_route_ids = [row.route_id for row in holdout_routes]
    if set(predictions) != set(holdout_route_ids):
        raise HighresBoundaryV2Error("frozen predictions must cover exactly all opaque holdout routes")
    if [route_id for route_id in holdout_route_ids if route_id not in predictions]:
        raise HighresBoundaryV2Error("frozen predictions are missing holdout route rows")

    train_sources = {row.source_key for row in train_truth}
    holdout_sources = {row.source_key for row in holdout_truth}
    overlap = train_sources & holdout_sources
    if overlap:
        raise HighresBoundaryV2Error(f"train/holdout source leak: {len(overlap)} overlapping sources")

    holdout_jumps = _jump_truth_rows(holdout_truth)
    if len(holdout_jumps) < config.min_holdout_jump_routes:
        raise HighresBoundaryV2Error(
            f"need at least {config.min_holdout_jump_routes} holdout jump routes, got {len(holdout_jumps)}"
        )
    ratio_counts = _ratio_counts(holdout_jumps)
    _require_ratio_buckets(ratio_counts, split_name="holdout")

    rows = []
    errors = []
    for truth in holdout_jumps:
        if truth.seam_seconds is None:
            raise HighresBoundaryV2Error(f"holdout jump route {truth.route_id} has no seam label")
        prediction = predictions[truth.route_id]
        predicted_seconds = _required_float(
            prediction.get("selected_candidate_seconds"),
            f"prediction {truth.route_id} selected_candidate_seconds",
        )
        error_seconds = abs(predicted_seconds - truth.seam_seconds)
        errors.append(error_seconds)
        rows.append(
            {
                "route_id": truth.route_id,
                "ratio": truth.ratio,
                "truth_seam_seconds": truth.seam_seconds,
                "predicted_seconds": predicted_seconds,
                "absolute_error_seconds": error_seconds,
                "pass_1000ms": error_seconds <= 1.0,
            }
        )

    errors_array = np.asarray(errors, dtype=np.float64)
    per_ratio = {}
    for ratio in FROZEN_RATIOS:
        ratio_errors = np.asarray(
            [row["absolute_error_seconds"] for row in rows if _same_ratio(float(row["ratio"]), ratio)],
            dtype=np.float64,
        )
        per_ratio[_ratio_key(ratio)] = _metric_payload(ratio_errors)

    summary = {
        "schema": EVAL_SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "holdout_truth_opened_after_predictions_frozen": True,
        "prediction_file": config.frozen_predictions_jsonl.as_posix(),
        "holdout_jump_count": len(holdout_jumps),
        "holdout_source_count": len(holdout_sources),
        "train_source_count": len(train_sources),
        "source_disjoint": True,
        "primary_gate": {
            "required_pass_count_1000ms": 103,
            "observed_pass_count_1000ms": int(np.count_nonzero(errors_array <= 1.0)),
            "route_count": int(errors_array.size),
            "status": "pass" if int(np.count_nonzero(errors_array <= 1.0)) >= 103 else "fail",
        },
        "metrics": _metric_payload(errors_array),
        "per_ratio": per_ratio,
        "rows": rows,
    }
    _atomic_write_json(config.eval_summary_json, summary, mode=0o644)
    return summary


def load_route_rows(path: Path, *, route_schema: str) -> list[RouteRow]:
    rows: list[RouteRow] = []
    for line_number, payload in _read_jsonl(path):
        keys = frozenset(payload)
        if keys != EXPECTED_ROUTE_FIELDS:
            raise HighresBoundaryV2Error(
                f"route row {line_number} fields must be {sorted(EXPECTED_ROUTE_FIELDS)}, got {sorted(keys)}"
            )
        schema = payload.get("schema")
        if schema != route_schema:
            raise HighresBoundaryV2Error(
                f"route row {line_number} schema {schema!r} does not match {route_schema!r}"
            )
        route_id = _required_string(payload.get("route_id"), f"route row {line_number} route_id")
        audio_path = Path(_required_string(payload.get("audio_path"), f"route row {line_number} audio_path"))
        rows.append(RouteRow(route_id=route_id, audio_path=audio_path))
    _require_unique([row.route_id for row in rows], label=f"{path} route_id")
    return rows


def load_truth_rows(path: Path, *, truth_schema: str) -> list[TruthRow]:
    rows: list[TruthRow] = []
    for line_number, payload in _read_jsonl(path):
        schema = payload.get("schema")
        if schema != truth_schema:
            raise HighresBoundaryV2Error(
                f"truth row {line_number} schema {schema!r} does not match {truth_schema!r}"
            )
        route_id = _required_string(payload.get("route_id"), f"truth row {line_number} route_id")
        split = _required_string(payload.get("split"), f"truth row {line_number} split")
        transform = _required_string(
            payload.get("transform", payload.get("transform_class")),
            f"truth row {line_number} transform",
        )
        ratio_value = payload.get("target_rate_ratio", payload.get("ratio"))
        ratio = _required_float(ratio_value, f"truth row {line_number} target_rate_ratio")
        seam_value = payload.get("seam_output_seconds", payload.get("seam_seconds"))
        seam_seconds = None if seam_value is None else _required_float(seam_value, f"truth row {line_number} seam")
        source_key = _required_string(payload.get("source_key"), f"truth row {line_number} source_key")
        rows.append(
            TruthRow(
                route_id=route_id,
                split=split,
                transform=transform,
                ratio=ratio,
                seam_seconds=seam_seconds,
                source_key=source_key,
            )
        )
    _require_unique([row.route_id for row in rows], label=f"{path} route_id")
    return rows


def load_or_extract_route_features(
    route: RouteRow,
    *,
    config: RunnerConfig,
    beatthis_provider: BeatThisTimingProvider,
) -> RouteFeatureBundle:
    cache_path = _feature_cache_path(config, route.route_id)
    if cache_path.exists():
        return _load_feature_bundle(cache_path, expected_route_id=route.route_id)
    bundle = extract_route_features(route, config=config, beatthis_provider=beatthis_provider)
    _save_feature_bundle(bundle, cache_path=cache_path, config=config)
    return bundle


def extract_route_features(
    route: RouteRow,
    *,
    config: RunnerConfig,
    beatthis_provider: BeatThisTimingProvider,
) -> RouteFeatureBundle:
    audio, sample_rate = _load_audio(route.audio_path)
    duration = float(audio.shape[0]) / float(sample_rate)
    if duration <= 2.0 * config.candidate_margin_seconds:
        raise HighresBoundaryV2Error(
            f"route {route.route_id} duration is too short for margin {config.candidate_margin_seconds}s"
        )
    audio_cache_key = beatthis_audio_cache_key(route.audio_path)
    beatthis_prediction = _load_or_create_beatthis_prediction(
        route.audio_path,
        audio_cache_key=audio_cache_key,
        config=config,
        beatthis_provider=beatthis_provider,
    )
    log_mel = load_or_create_log_mel_cache(
        audio,
        sample_rate=sample_rate,
        audio_cache_key=audio_cache_key,
    )
    raw_evidence = extract_raw_audio_evidence(log_mel, audio_duration_seconds=duration)
    sample_times, signals = _build_signal_grid(
        beatthis_prediction,
        raw_evidence_band_flux=raw_evidence.band_flux,
        raw_evidence_times=raw_evidence.frame_center_seconds,
        duration_seconds=duration,
        sample_hop_seconds=config.feature_sample_hop_seconds,
    )
    candidates_seconds = _candidate_seconds_from_grid(
        sample_times,
        margin_seconds=config.candidate_margin_seconds,
        candidate_hop_seconds=config.candidate_hop_seconds,
    )
    features = _extract_candidate_features(
        sample_times=sample_times,
        signals=signals,
        candidates_seconds=candidates_seconds,
        sample_hop_seconds=config.feature_sample_hop_seconds,
    )
    return RouteFeatureBundle(
        route_id=route.route_id,
        candidates_seconds=candidates_seconds,
        features=features,
    )


def train_boundary_model(
    examples: Sequence[TrainExample],
    *,
    config: RunnerConfig,
) -> tuple[Any, dict[str, Any]]:
    if len(examples) < config.min_train_jump_routes:
        raise HighresBoundaryV2Error(
            f"need at least {config.min_train_jump_routes} train examples, got {len(examples)}"
        )
    import torch
    import torch.nn.functional as F

    device = _torch_device(config.train_device)
    torch.manual_seed(config.seed)
    model = _BoundaryMLP(input_dim=len(FEATURE_NAMES), hidden_units=config.hidden_units).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(config.seed)
    order = np.arange(len(examples), dtype=np.int64)
    epoch_losses: list[float] = []
    for epoch in range(1, config.train_epochs + 1):
        rng.shuffle(order)
        batch_losses: list[float] = []
        for batch_indices in _batches(order, config.train_batch_routes):
            batch = [examples[int(index)] for index in batch_indices]
            x_np, mask_np, targets_np = _pad_batch(batch)
            x = torch.from_numpy(x_np).to(device)
            mask = torch.from_numpy(mask_np).to(device)
            targets = torch.from_numpy(targets_np).to(device)
            logits = model(x.reshape(-1, x.shape[-1])).reshape(x.shape[0], x.shape[1])
            logits = logits.masked_fill(~mask, -1.0e9)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))
        epoch_loss = float(np.mean(batch_losses, dtype=np.float64))
        epoch_losses.append(epoch_loss)
        if epoch == 1 or epoch % 50 == 0 or epoch == config.train_epochs:
            print(
                json.dumps(
                    {
                        "stage": "train",
                        "epoch": epoch,
                        "epochs": config.train_epochs,
                        "loss": epoch_loss,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    train_errors = _prediction_errors_for_examples(model, examples, device=device)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "feature_contract": FEATURE_CONTRACT_ID,
        "route_count": len(examples),
        "candidate_count_min": int(min(example.features.shape[0] for example in examples)),
        "candidate_count_max": int(max(example.features.shape[0] for example in examples)),
        "candidate_count_mean": float(np.mean([example.features.shape[0] for example in examples])),
        "epochs": config.train_epochs,
        "batch_routes": config.train_batch_routes,
        "seed": config.seed,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "hidden_units": list(config.hidden_units),
        "final_loss": epoch_losses[-1],
        "train_metrics": _metric_payload(np.asarray(train_errors, dtype=np.float64)),
    }
    _atomic_write_json(config.train_summary_path, summary, mode=0o644)
    return model, summary


def predict_holdout_routes(
    holdout_routes: Sequence[RouteRow],
    model: Any,
    *,
    config: RunnerConfig,
    beatthis_provider: BeatThisTimingProvider,
) -> list[dict[str, Any]]:
    device = _model_device(model)
    predictions: list[dict[str, Any]] = []
    for index, route in enumerate(holdout_routes, start=1):
        bundle = load_or_extract_route_features(
            route,
            config=config,
            beatthis_provider=beatthis_provider,
        )
        candidate_index, score = _predict_bundle(model, bundle, device=device)
        predictions.append(
            {
                "schema": PREDICTION_SCHEMA,
                "experiment": EXPERIMENT_NAME,
                "feature_contract": FEATURE_CONTRACT_ID,
                "route_id": route.route_id,
                "candidate_count": int(bundle.candidates_seconds.shape[0]),
                "selected_candidate_index": int(candidate_index),
                "selected_candidate_seconds": float(bundle.candidates_seconds[candidate_index]),
                "selected_score": float(score),
            }
        )
        if index % 64 == 0 or index == len(holdout_routes):
            print(
                json.dumps(
                    {"stage": "predict_holdout", "done": index, "total": len(holdout_routes)},
                    sort_keys=True,
                ),
                flush=True,
            )
    return predictions


def write_frozen_predictions(
    predictions: Sequence[Mapping[str, Any]],
    *,
    output_jsonl: Path,
    summary_json: Path,
    expected_route_count: int,
) -> None:
    if output_jsonl.exists():
        raise HighresBoundaryV2Error(f"refusing to overwrite frozen predictions: {output_jsonl}")
    if len(predictions) != expected_route_count:
        raise HighresBoundaryV2Error(
            f"expected {expected_route_count} holdout predictions, got {len(predictions)}"
        )
    _require_unique(
        [_required_string(row.get("route_id"), "prediction route_id") for row in predictions],
        label="prediction route_id",
    )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_jsonl.parent,
            prefix=f".{output_jsonl.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            for prediction in predictions:
                handle.write(json.dumps(dict(prediction), sort_keys=True, separators=(",", ":")))
                handle.write("\n")
        os.chmod(tmp_path, 0o444)
        os.replace(tmp_path, output_jsonl)
        os.chmod(output_jsonl, 0o444)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise

    summary = {
        "schema": PREDICTION_SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "prediction_count": len(predictions),
        "output_jsonl": output_jsonl.as_posix(),
        "holdout_truth_opened": False,
        "frozen": True,
    }
    _atomic_write_json(summary_json, summary, mode=0o444)


def load_frozen_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for line_number, payload in _read_jsonl(path):
        schema = payload.get("schema")
        if schema != PREDICTION_SCHEMA:
            raise HighresBoundaryV2Error(
                f"prediction line {line_number} schema {schema!r} does not match {PREDICTION_SCHEMA!r}"
            )
        route_id = _required_string(payload.get("route_id"), f"prediction line {line_number} route_id")
        if route_id in predictions:
            raise HighresBoundaryV2Error(f"duplicate prediction route_id {route_id!r}")
        predictions[route_id] = dict(payload)
    return predictions


def _validate_feature_contract() -> None:
    if len(SIGNAL_NAMES) != 10:
        raise AssertionError(f"expected 10 signals, got {len(SIGNAL_NAMES)}")
    if len(DERIVED_FEATURE_NAMES) != 10:
        raise AssertionError(f"expected 10 derived features, got {len(DERIVED_FEATURE_NAMES)}")
    if len(FEATURE_NAMES) != 100:
        raise AssertionError(f"expected 100 features, got {len(FEATURE_NAMES)}")
    seen = set()
    for name in FEATURE_NAMES:
        if name in seen:
            raise AssertionError(f"duplicate feature name: {name}")
        seen.add(name)
        tokens = set(name.replace(":", "_").split("_"))
        forbidden = tokens & FORBIDDEN_FEATURE_TOKENS
        if forbidden:
            raise AssertionError(f"feature name {name!r} contains forbidden tokens {sorted(forbidden)}")


def _feature_cache_path(config: RunnerConfig, route_id: str) -> Path:
    return config.feature_cache_root / f"{route_id}.npz"


def _save_feature_bundle(
    bundle: RouteFeatureBundle,
    *,
    cache_path: Path,
    config: RunnerConfig,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment": EXPERIMENT_NAME,
        "feature_contract": FEATURE_CONTRACT_ID,
        "route_id": bundle.route_id,
        "feature_names": list(FEATURE_NAMES),
        "candidate_margin_seconds": config.candidate_margin_seconds,
        "candidate_hop_seconds": config.candidate_hop_seconds,
        "feature_sample_hop_seconds": config.feature_sample_hop_seconds,
    }
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            np.savez_compressed(
                handle,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
                candidates_seconds=np.asarray(bundle.candidates_seconds, dtype=np.float64),
                features=np.asarray(bundle.features, dtype=np.float32),
            )
        os.replace(tmp_path, cache_path)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _load_feature_bundle(cache_path: Path, *, expected_route_id: str) -> RouteFeatureBundle:
    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
            if metadata.get("feature_contract") != FEATURE_CONTRACT_ID:
                raise HighresBoundaryV2Error(f"stale feature cache contract: {cache_path}")
            if metadata.get("route_id") != expected_route_id:
                raise HighresBoundaryV2Error(f"feature cache route mismatch: {cache_path}")
            if tuple(metadata.get("feature_names", ())) != FEATURE_NAMES:
                raise HighresBoundaryV2Error(f"feature cache names mismatch: {cache_path}")
            candidates_seconds = np.asarray(payload["candidates_seconds"], dtype=np.float64)
            features = np.asarray(payload["features"], dtype=np.float32)
    except HighresBoundaryV2Error:
        raise
    except Exception as exc:
        raise HighresBoundaryV2Error(f"invalid feature cache {cache_path}: {exc}") from exc
    _validate_feature_matrix(features, candidates_seconds)
    return RouteFeatureBundle(
        route_id=expected_route_id,
        candidates_seconds=candidates_seconds,
        features=features,
    )


def _load_audio(audio_path: Path) -> tuple[NDArray[np.float32], int]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required; run with the mps or cuda optional extra") from exc
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 2:
        array = np.mean(array, axis=1, dtype=np.float32)
    if array.ndim != 1:
        raise HighresBoundaryV2Error(f"audio must be mono or channel-major 2-D: {audio_path}")
    if int(sample_rate) != 16_000:
        raise HighresBoundaryV2Error(f"expected 16kHz rendered corpus WAV, got {sample_rate}: {audio_path}")
    if array.size == 0:
        raise HighresBoundaryV2Error(f"empty audio: {audio_path}")
    return np.ascontiguousarray(array, dtype=np.float32), int(sample_rate)


def _load_or_create_beatthis_prediction(
    audio_path: Path,
    *,
    audio_cache_key: str,
    config: RunnerConfig,
    beatthis_provider: BeatThisTimingProvider,
) -> FrameTimingPrediction:
    prediction = load_beatthis_frame_prediction_cache(
        audio_cache_key,
        config.beatthis_cache_config,
    )
    if prediction is not None:
        return prediction
    prediction = beatthis_provider.predict_file(audio_path)
    save_beatthis_frame_prediction_cache(
        prediction,
        audio_cache_key,
        config.beatthis_cache_config,
    )
    return prediction


def _build_signal_grid(
    prediction: FrameTimingPrediction,
    *,
    raw_evidence_band_flux: NDArray[np.float32],
    raw_evidence_times: NDArray[np.float64],
    duration_seconds: float,
    sample_hop_seconds: float,
) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
    if duration_seconds <= 0.0:
        raise HighresBoundaryV2Error("duration must be positive")
    frame_count = int(math.floor(duration_seconds / sample_hop_seconds)) + 1
    sample_times = np.arange(frame_count, dtype=np.float64) * np.float64(sample_hop_seconds)

    beat = np.asarray(prediction.beat_prob, dtype=np.float32)
    downbeat = np.asarray(prediction.downbeat_prob, dtype=np.float32)
    beat_times = np.arange(beat.shape[0], dtype=np.float64) / np.float64(prediction.frame_rate_hz)

    raw_flux = np.asarray(raw_evidence_band_flux, dtype=np.float32)
    if raw_flux.ndim != 2 or raw_flux.shape[1] != 4:
        raise HighresBoundaryV2Error(f"expected raw flux shape [frames,4], got {raw_flux.shape}")
    raw_sum = np.sum(raw_flux, axis=1, dtype=np.float32)
    raw_max = np.max(raw_flux, axis=1).astype(np.float32, copy=False)
    raw_low = raw_flux[:, 0]
    raw_mid = np.mean(raw_flux[:, 1:3], axis=1, dtype=np.float32)
    raw_high = raw_flux[:, 3]

    native_signals: list[tuple[NDArray[np.float64], NDArray[np.float32]]] = [
        (beat_times, beat),
        (beat_times, downbeat),
        (beat_times, _abs_gradient(beat)),
        (beat_times, _abs_gradient(downbeat)),
        (raw_evidence_times, raw_sum),
        (raw_evidence_times, raw_max),
        (raw_evidence_times, raw_low),
        (raw_evidence_times, raw_mid),
        (raw_evidence_times, raw_high),
        (raw_evidence_times, _abs_gradient(raw_sum)),
    ]
    resampled = []
    for times, values in native_signals:
        normalized = _robust_normalize(values)
        resampled.append(_interp_signal(times, normalized, sample_times))
    signals = np.stack(resampled, axis=1).astype(np.float32, copy=False)
    if signals.shape[1] != len(SIGNAL_NAMES):
        raise AssertionError("signal count mismatch")
    return sample_times, signals


def _candidate_seconds_from_grid(
    sample_times: NDArray[np.float64],
    *,
    margin_seconds: float,
    candidate_hop_seconds: float,
) -> NDArray[np.float64]:
    if sample_times.size == 0:
        raise HighresBoundaryV2Error("empty feature sample grid")
    min_center = math.ceil((float(sample_times[0]) + margin_seconds) / candidate_hop_seconds) * candidate_hop_seconds
    max_center = math.floor((float(sample_times[-1]) - margin_seconds) / candidate_hop_seconds) * candidate_hop_seconds
    if max_center < min_center:
        raise HighresBoundaryV2Error("no candidates after applying feature margin")
    count = int(math.floor((max_center - min_center) / candidate_hop_seconds + 0.5)) + 1
    candidates = min_center + np.arange(count, dtype=np.float64) * np.float64(candidate_hop_seconds)
    return np.round(candidates, decimals=10).astype(np.float64, copy=False)


def _extract_candidate_features(
    *,
    sample_times: NDArray[np.float64],
    signals: NDArray[np.float32],
    candidates_seconds: NDArray[np.float64],
    sample_hop_seconds: float,
) -> NDArray[np.float32]:
    if signals.ndim != 2 or signals.shape[1] != len(SIGNAL_NAMES):
        raise HighresBoundaryV2Error(f"expected signal matrix [frames,{len(SIGNAL_NAMES)}], got {signals.shape}")
    candidate_indices = np.rint(candidates_seconds / sample_hop_seconds).astype(np.int64)
    if not np.allclose(sample_times[candidate_indices], candidates_seconds, atol=1e-8):
        raise HighresBoundaryV2Error("candidate grid is not aligned to feature sample grid")
    frame_offsets = {
        "center": (-5, 5),
        "near_left": (-20, -2),
        "near_right": (2, 20),
        "mid_left": (-80, -20),
        "mid_right": (20, 80),
        "far_left": (-160, -80),
        "far_right": (80, 160),
    }
    all_features = np.empty((candidates_seconds.shape[0], len(FEATURE_NAMES)), dtype=np.float32)
    output_column = 0
    for signal_index in range(signals.shape[1]):
        values = np.asarray(signals[:, signal_index], dtype=np.float32)
        center_mean, center_max, center_std = _window_mean_max_std(
            values,
            candidate_indices,
            *frame_offsets["center"],
        )
        near_left_mean, near_left_max, _ = _window_mean_max_std(
            values,
            candidate_indices,
            *frame_offsets["near_left"],
        )
        near_right_mean, near_right_max, _ = _window_mean_max_std(
            values,
            candidate_indices,
            *frame_offsets["near_right"],
        )
        mid_left_mean, mid_left_max, _ = _window_mean_max_std(
            values,
            candidate_indices,
            *frame_offsets["mid_left"],
        )
        mid_right_mean, mid_right_max, _ = _window_mean_max_std(
            values,
            candidate_indices,
            *frame_offsets["mid_right"],
        )
        far_left_mean, _, _ = _window_mean_max_std(
            values,
            candidate_indices,
            *frame_offsets["far_left"],
        )
        far_right_mean, _, _ = _window_mean_max_std(
            values,
            candidate_indices,
            *frame_offsets["far_right"],
        )
        near_delta = near_right_mean - near_left_mean
        mid_delta = mid_right_mean - mid_left_mean
        far_delta = far_right_mean - far_left_mean
        columns = (
            center_mean,
            center_max,
            center_std,
            near_delta,
            mid_delta,
            far_delta,
            np.abs(near_delta),
            np.abs(mid_delta),
            near_right_max - near_left_max,
            mid_right_max - mid_left_max,
        )
        for column in columns:
            all_features[:, output_column] = column.astype(np.float32, copy=False)
            output_column += 1
    _validate_feature_matrix(all_features, candidates_seconds)
    return all_features


def _window_mean_max_std(
    values: NDArray[np.float32],
    candidate_indices: NDArray[np.int64],
    start_offset: int,
    end_offset: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    if end_offset < start_offset:
        raise ValueError("end_offset must be >= start_offset")
    starts = candidate_indices + int(start_offset)
    ends = candidate_indices + int(end_offset)
    if starts.min(initial=0) < 0 or ends.max(initial=-1) >= values.shape[0]:
        raise HighresBoundaryV2Error("candidate window exceeds available signal grid")
    prefix = np.concatenate(([0.0], np.cumsum(values.astype(np.float64), dtype=np.float64)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(np.square(values.astype(np.float64)), dtype=np.float64)))
    counts = (ends - starts + 1).astype(np.float64)
    sums = prefix[ends + 1] - prefix[starts]
    sums_sq = prefix_sq[ends + 1] - prefix_sq[starts]
    means = sums / counts
    variances = np.maximum(sums_sq / counts - np.square(means), 0.0)
    max_values = np.asarray([np.max(values[start : end + 1]) for start, end in zip(starts, ends)], dtype=np.float32)
    return (
        means.astype(np.float32),
        max_values,
        np.sqrt(variances).astype(np.float32),
    )


def _train_example_from_bundle(bundle: RouteFeatureBundle, truth: TruthRow) -> TrainExample:
    if truth.transform != "jump":
        raise HighresBoundaryV2Error(f"train example {truth.route_id} is not a jump transform")
    if truth.seam_seconds is None:
        raise HighresBoundaryV2Error(f"train jump route {truth.route_id} has no seam label")
    target_index = int(np.argmin(np.abs(bundle.candidates_seconds - truth.seam_seconds)))
    return TrainExample(
        route_id=bundle.route_id,
        candidates_seconds=bundle.candidates_seconds,
        features=bundle.features,
        target_index=target_index,
        target_seconds=truth.seam_seconds,
    )


def _validate_feature_matrix(features: NDArray[np.float32], candidates_seconds: NDArray[np.float64]) -> None:
    if features.dtype != np.float32:
        raise HighresBoundaryV2Error("features must be float32")
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise HighresBoundaryV2Error(f"features must have shape [candidates,{len(FEATURE_NAMES)}], got {features.shape}")
    if candidates_seconds.dtype != np.float64:
        raise HighresBoundaryV2Error("candidates_seconds must be float64")
    if candidates_seconds.ndim != 1 or candidates_seconds.shape[0] != features.shape[0]:
        raise HighresBoundaryV2Error("candidate seconds must be a 1-D array matching feature rows")
    if not np.all(np.isfinite(features)):
        raise HighresBoundaryV2Error("features contain non-finite values")
    if not np.all(np.isfinite(candidates_seconds)):
        raise HighresBoundaryV2Error("candidate seconds contain non-finite values")


def _abs_gradient(values: NDArray[np.float32]) -> NDArray[np.float32]:
    if values.size == 0:
        return np.empty((0,), dtype=np.float32)
    if values.size == 1:
        return np.zeros_like(values, dtype=np.float32)
    return np.abs(np.gradient(values.astype(np.float32))).astype(np.float32)


def _robust_normalize(values: NDArray[np.float32]) -> NDArray[np.float32]:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return array
    median = float(np.median(array))
    p10 = float(np.percentile(array, 10.0, method="linear"))
    p90 = float(np.percentile(array, 90.0, method="linear"))
    scale = max(p90 - p10, 1.0e-6)
    return ((array.astype(np.float64) - median) / scale).astype(np.float32)


def _interp_signal(
    times: NDArray[np.float64],
    values: NDArray[np.float32],
    sample_times: NDArray[np.float64],
) -> NDArray[np.float32]:
    if values.size == 0 or times.size == 0:
        return np.zeros_like(sample_times, dtype=np.float32)
    if values.shape[0] != times.shape[0]:
        raise HighresBoundaryV2Error("signal values and times length mismatch")
    return np.interp(
        sample_times,
        times.astype(np.float64, copy=False),
        values.astype(np.float32, copy=False),
        left=float(values[0]),
        right=float(values[-1]),
    ).astype(np.float32)


class _BoundaryMLP:  # concrete class is swapped after torch import in __new__.
    def __new__(cls, *, input_dim: int, hidden_units: tuple[int, int]) -> Any:
        import torch.nn as nn

        class BoundaryMLPImpl(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_units[0]),
                    nn.ReLU(),
                    nn.Linear(hidden_units[0], hidden_units[1]),
                    nn.ReLU(),
                    nn.Linear(hidden_units[1], 1),
                )

            def forward(self, x: Any) -> Any:
                return self.net(x).squeeze(-1)

        return BoundaryMLPImpl()


def _pad_batch(
    examples: Sequence[TrainExample],
) -> tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.int64]]:
    max_candidates = max(example.features.shape[0] for example in examples)
    x = np.zeros((len(examples), max_candidates, len(FEATURE_NAMES)), dtype=np.float32)
    mask = np.zeros((len(examples), max_candidates), dtype=np.bool_)
    targets = np.empty((len(examples),), dtype=np.int64)
    for row_index, example in enumerate(examples):
        count = example.features.shape[0]
        x[row_index, :count, :] = example.features
        mask[row_index, :count] = True
        targets[row_index] = example.target_index
    return x, mask, targets


def _batches(values: NDArray[np.int64], batch_size: int) -> Iterable[NDArray[np.int64]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, values.shape[0], batch_size):
        yield values[start : start + batch_size]


def _prediction_errors_for_examples(model: Any, examples: Sequence[TrainExample], *, device: Any) -> list[float]:
    errors = []
    for example in examples:
        bundle = RouteFeatureBundle(
            route_id=example.route_id,
            candidates_seconds=example.candidates_seconds,
            features=example.features,
        )
        selected_index, _ = _predict_bundle(model, bundle, device=device)
        errors.append(abs(float(example.candidates_seconds[selected_index]) - example.target_seconds))
    return errors


def _predict_bundle(model: Any, bundle: RouteFeatureBundle, *, device: Any) -> tuple[int, float]:
    import torch

    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(bundle.features).to(device)
        logits = model(x)
        index = int(torch.argmax(logits).detach().cpu().item())
        score = float(logits[index].detach().cpu().item())
    return index, score


def _save_model(model: Any, train_summary: Mapping[str, Any], *, config: RunnerConfig) -> None:
    import torch

    config.output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": EXPERIMENT_NAME,
        "feature_contract": FEATURE_CONTRACT_ID,
        "feature_names": list(FEATURE_NAMES),
        "input_dim": len(FEATURE_NAMES),
        "hidden_units": list(config.hidden_units),
        "state_dict": model.state_dict(),
        "train_summary": dict(train_summary),
    }
    tmp_path = config.model_path.with_suffix(".pt.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, config.model_path)


def _torch_device(requested: str) -> Any:
    import torch

    if requested == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def _model_device(model: Any) -> Any:
    return next(model.parameters()).device


def _beatthis_provider(config: RunnerConfig) -> BeatThisTimingProvider:
    device = config.beatthis_device
    if device == "auto":
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        except Exception:
            device = "cpu"
    return BeatThisTimingProvider(
        checkpoint_path=config.beatthis_cache_config.checkpoint_path,
        device=device,
        float16=config.beatthis_cache_config.float16,
    )


def _metric_payload(errors_seconds: NDArray[np.float64]) -> dict[str, Any]:
    if errors_seconds.size == 0:
        return {
            "count": 0,
            "pass_count_100ms": 0,
            "pass_count_250ms": 0,
            "pass_count_500ms": 0,
            "pass_count_1000ms": 0,
            "pass_rate_1000ms": None,
            "median_error_ms": None,
            "p90_error_ms": None,
        }
    return {
        "count": int(errors_seconds.size),
        "pass_count_100ms": int(np.count_nonzero(errors_seconds <= 0.1)),
        "pass_count_250ms": int(np.count_nonzero(errors_seconds <= 0.25)),
        "pass_count_500ms": int(np.count_nonzero(errors_seconds <= 0.5)),
        "pass_count_1000ms": int(np.count_nonzero(errors_seconds <= 1.0)),
        "pass_rate_1000ms": float(np.mean(errors_seconds <= 1.0)),
        "median_error_ms": float(np.median(errors_seconds) * 1000.0),
        "p90_error_ms": float(np.percentile(errors_seconds, 90.0, method="linear") * 1000.0),
    }


def _truth_by_route_id(rows: Sequence[TruthRow]) -> dict[str, TruthRow]:
    return {row.route_id: row for row in rows}


def _require_route_truth_match(
    routes: Sequence[RouteRow],
    truth_by_id: Mapping[str, TruthRow],
    *,
    split_name: str,
) -> None:
    route_ids = {row.route_id for row in routes}
    truth_ids = set(truth_by_id)
    if route_ids != truth_ids:
        raise HighresBoundaryV2Error(
            f"{split_name} route/truth IDs differ: routes={len(route_ids)} truth={len(truth_ids)}"
        )


def _jump_truth_rows(rows: Sequence[TruthRow]) -> list[TruthRow]:
    jumps = [row for row in rows if row.transform == "jump"]
    for row in jumps:
        if row.seam_seconds is None:
            raise HighresBoundaryV2Error(f"jump route {row.route_id} has no seam label")
    return jumps


def _ratio_counts(rows: Sequence[TruthRow]) -> dict[str, int]:
    counts = {_ratio_key(ratio): 0 for ratio in FROZEN_RATIOS}
    for row in rows:
        matched = False
        for ratio in FROZEN_RATIOS:
            if _same_ratio(row.ratio, ratio):
                counts[_ratio_key(ratio)] += 1
                matched = True
                break
        if not matched:
            raise HighresBoundaryV2Error(f"unexpected jump ratio {row.ratio!r} for route {row.route_id}")
    return counts


def _require_ratio_buckets(counts: Mapping[str, int], *, split_name: str) -> None:
    missing = [key for key, value in counts.items() if int(value) <= 0]
    if missing:
        raise HighresBoundaryV2Error(f"{split_name} is missing frozen ratio buckets: {missing}")


def _ratio_key(ratio: float) -> str:
    return f"{ratio:g}"


def _same_ratio(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1.0e-9


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise HighresBoundaryV2Error(f"{path} line {line_number} must be a JSON object")
            yield line_number, payload


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise HighresBoundaryV2Error(f"{field_name} must be a non-empty string")
    return value


def _required_float(value: object, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise HighresBoundaryV2Error(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise HighresBoundaryV2Error(f"{field_name} must be finite")
    return result


def _require_unique(values: Sequence[str], *, label: str) -> None:
    seen = set()
    for value in values:
        if value in seen:
            raise HighresBoundaryV2Error(f"duplicate {label}: {value!r}")
        seen.add(value)


def _atomic_write_json(path: Path, payload: Mapping[str, Any], *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
        os.chmod(path, mode)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real stable256 high-resolution Timing v3 seam-boundary experiment.",
    )
    parser.add_argument("--self-check", action="store_true", help="run static contract checks")
    parser.add_argument("--preflight", action="store_true", help="validate readable train/opaque route contract")
    parser.add_argument("--run", action="store_true", help="extract/train/freeze all holdout predictions")
    parser.add_argument("--evaluate", action="store_true", help="evaluate already frozen predictions after truth unseal")
    parser.add_argument("--train-routes", type=Path, default=DEFAULT_TRAIN_ROUTES)
    parser.add_argument("--holdout-routes", type=Path, default=DEFAULT_HOLDOUT_ROUTES)
    parser.add_argument("--train-truth", type=Path, default=DEFAULT_TRAIN_TRUTH)
    parser.add_argument("--holdout-truth", type=Path, default=DEFAULT_HOLDOUT_TRUTH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--route-schema", default=ROUTE_SCHEMA)
    parser.add_argument("--truth-schema", default=TRUTH_SCHEMA)
    parser.add_argument("--beatthis-device", default="auto")
    parser.add_argument("--train-device", default="auto")
    parser.add_argument("--reuse-frozen-predictions", action="store_true")
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> RunnerConfig:
    return RunnerConfig(
        train_routes_jsonl=args.train_routes,
        holdout_routes_jsonl=args.holdout_routes,
        train_truth_jsonl=args.train_truth,
        holdout_truth_jsonl=args.holdout_truth,
        output_root=args.output_root,
        route_schema=args.route_schema,
        truth_schema=args.truth_schema,
        beatthis_device=args.beatthis_device,
        train_device=args.train_device,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = _config_from_args(args)
    if args.self_check:
        print(json.dumps(self_check(), indent=2, sort_keys=True))
        return 0
    if args.run:
        summary = run_experiment(config, reuse_frozen_predictions=args.reuse_frozen_predictions)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.evaluate:
        summary = evaluate_frozen_holdout(config)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["primary_gate"]["status"] == "pass" else 2
    if args.preflight:
        print(json.dumps(preflight_run(config), indent=2, sort_keys=True))
        return 0
    print(json.dumps(plan_run(config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
