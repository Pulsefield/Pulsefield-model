from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping as MappingABC, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np

from pulsefield_model.features.audio import load_audio_file
from pulsefield_model.features.mel import stage2_log_mel_cache_path
from pulsefield_model.features.mel_base import DEFAULT_MEL_CACHE_CONFIG, compute_log_mel_10ms
from pulsefield_model.timing.evaluation.curve_metrics import (
    WeakOracleClass,
    evaluate_curve_against_weak_oracle,
)
from pulsefield_model.timing.providers.oracle import oracle_timing_grid_from_beatmap
from pulsefield_model.timing.schema import FittedTimingGrid, TimingSegment
from pulsefield_model.timing.v3.analytic_curve import PhaseContinuousTimingCurve
from pulsefield_model.timing.v3.audio_evidence import AnalyticTimingCandidate, score_timing_candidates
from pulsefield_model.timing.v3.schema import TimingV3Grid


REAL_AUDIO_PILOT_RESULT_SCHEMA = "pulsefield_model.timing_v3_real_audio_pilot_result_v1"
REAL_AUDIO_PILOT_SUMMARY_SCHEMA = "pulsefield_model.timing_v3_real_audio_pilot_summary_v1"
INFERENCE_FAMILY = "beatthis_candidates_plus_deterministic_raw_audio_v1"
WEAK_ORACLE_POLICY = "post_selection_representative_redline_v1"
DEFAULT_VARIANTS = ("CJ0", "CJ1", "CJ2", "CJ3")


class CandidateGenerator(Protocol):
    def __call__(
        self,
        pilot_row: Mapping[str, Any],
        projection_row: Mapping[str, Any],
        baseline_row: Mapping[str, Any] | None,
    ) -> Sequence[AnalyticTimingCandidate]: ...


WeakOracleLoader = Callable[[Path], FittedTimingGrid]


@dataclass(frozen=True)
class _GridCandidate:
    grid: TimingV3Grid
    source: str
    fingerprint_sha256: str
    score_start_beat: int | None = None
    score_end_beat: int | None = None

    @property
    def start_beat(self) -> int:
        return self.grid.start_beat if self.score_start_beat is None else self.score_start_beat

    @property
    def end_beat(self) -> int:
        return self.grid.end_beat if self.score_end_beat is None else self.score_end_beat

    def time_at_beat(self, beat: float) -> float:
        return self.grid.time_at_beat(beat)


@dataclass(frozen=True)
class FrozenInferenceSelection:
    inference_family: str
    candidate_fingerprints: tuple[str, ...]
    candidate_sources: tuple[str, ...]
    selected_fingerprint_sha256: str | None
    selected_candidate_index: int | None
    unavailable_reason: str | None
    raw_audio_candidate_scores: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_real_audio_pilot(
    *,
    pilot_jsonl_path: str | Path,
    projection_jsonl_path: str | Path,
    baseline_jsonl_path: str | Path,
    output_jsonl_path: str | Path,
    summary_json_path: str | Path,
    repo_root: str | Path | None = None,
    limit: int | None = None,
    candidate_generator: CandidateGenerator | None = None,
    weak_oracle_loader: WeakOracleLoader = oracle_timing_grid_from_beatmap,
) -> dict[str, Any]:
    """Run an inference-first real-audio candidate-ranking pilot.

    Candidate generation and raw-audio ranking complete before the label row's
    representative beatmap path is read or any `.osu` weak oracle is loaded.
    The frozen inference payload is immutable input to the evaluation phase.
    """

    pilot_path = Path(pilot_jsonl_path)
    projection_path = Path(projection_jsonl_path)
    baseline_path = Path(baseline_jsonl_path)
    output_path = Path(output_jsonl_path)
    summary_path = Path(summary_json_path)
    root = Path(repo_root or Path.cwd()).expanduser().resolve(strict=False)
    if output_path.resolve(strict=False) == summary_path.resolve(strict=False):
        raise ValueError("output_jsonl_path and summary_json_path must differ")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    pilot_rows = _read_jsonl(pilot_path)
    if limit is not None:
        pilot_rows = pilot_rows[:limit]
    projections = _index_rows(
        _read_jsonl(projection_path),
        key_getter=lambda row: _require_string(_nested(row, "identity", "cache_audio_key"), "projection cache_audio_key"),
        source_name="projection",
    )
    baselines = _index_rows(
        _read_jsonl(baseline_path),
        key_getter=lambda row: _require_string(row.get("audio_key"), "baseline audio_key"),
        source_name="baseline",
    )
    generator = candidate_generator or projection_cj_candidates

    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for row_index, pilot_row in enumerate(pilot_rows):
        row_started = time.perf_counter()
        cache_audio_key = _require_string(
            _nested(pilot_row, "source", "cache_audio_key"),
            "pilot source.cache_audio_key",
        )
        projection_row = projections.get(cache_audio_key)
        if projection_row is None:
            results.append(
                _failure_row(
                    row_index=row_index,
                    cache_audio_key=cache_audio_key,
                    pilot_row=pilot_row,
                    failure_stage="candidate_join",
                    error="projection row is unavailable",
                    row_started=row_started,
                )
            )
            continue
        baseline_row = baselines.get(cache_audio_key)
        try:
            result = _run_pilot_row(
                row_index=row_index,
                pilot_row=pilot_row,
                projection_row=projection_row,
                baseline_row=baseline_row,
                repo_root=root,
                candidate_generator=generator,
                weak_oracle_loader=weak_oracle_loader,
                row_started=row_started,
            )
        except Exception as exc:
            result = _failure_row(
                row_index=row_index,
                cache_audio_key=cache_audio_key,
                pilot_row=pilot_row,
                failure_stage="execution",
                error=f"{type(exc).__name__}: {exc}",
                row_started=row_started,
            )
        results.append(result)

    _write_jsonl_atomic(output_path, results)
    summary = _summary_payload(
        results,
        pilot_path=pilot_path,
        projection_path=projection_path,
        baseline_path=baseline_path,
        output_path=output_path,
        total_seconds=time.perf_counter() - started,
    )
    _write_json_atomic(summary_path, summary)
    return summary


def projection_cj_candidates(
    pilot_row: Mapping[str, Any],
    projection_row: Mapping[str, Any],
    baseline_row: Mapping[str, Any] | None,
) -> tuple[_GridCandidate, ...]:
    """Load only accepted stored CJ grids; labels and `.osu` are not consulted."""

    del pilot_row, baseline_row
    variants = projection_row.get("variants")
    if not isinstance(variants, MappingABC):
        raise ValueError("projection variants must be a mapping")
    candidates: list[_GridCandidate] = []
    seen: set[str] = set()
    for name in DEFAULT_VARIANTS:
        payload = variants.get(name)
        if not isinstance(payload, MappingABC) or payload.get("status") != "accepted":
            continue
        grid = TimingV3Grid.from_dict(_require_mapping(payload.get("grid"), f"variants.{name}.grid"))
        fingerprint = _require_sha256(
            payload.get("grid_fingerprint") or _stable_json_sha256(grid.to_dict()),
            f"variants.{name}.grid_fingerprint",
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        candidates.append(_GridCandidate(grid=grid, source=name, fingerprint_sha256=fingerprint))
    if not candidates:
        raise ValueError("projection row contains no accepted CJ candidate")
    # CJ variants may use different musical aliases, so their absolute integer
    # labels and total beat counts need not match.  Raw evidence only needs a
    # common ordered event denominator.  Score the exact intersection of their
    # valid integer domains while preserving each candidate's own time mapping
    # and persisted fingerprint.
    common_start = max(candidate.grid.start_beat for candidate in candidates)
    common_end = min(candidate.grid.end_beat for candidate in candidates)
    if common_end - common_start < 16:
        raise ValueError("accepted CJ candidates share fewer than 16 integer beats")
    return tuple(
        _GridCandidate(
            grid=candidate.grid,
            source=candidate.source,
            fingerprint_sha256=candidate.fingerprint_sha256,
            score_start_beat=common_start,
            score_end_beat=common_end,
        )
        for candidate in candidates
    )


def _run_pilot_row(
    *,
    row_index: int,
    pilot_row: Mapping[str, Any],
    projection_row: Mapping[str, Any],
    baseline_row: Mapping[str, Any] | None,
    repo_root: Path,
    candidate_generator: CandidateGenerator,
    weak_oracle_loader: WeakOracleLoader,
    row_started: float,
) -> dict[str, Any]:
    cache_audio_key = _require_string(
        _nested(pilot_row, "source", "cache_audio_key"),
        "pilot source.cache_audio_key",
    )
    audio_path = Path(
        _require_string(pilot_row.get("resolved_audio_path"), "pilot resolved_audio_path")
    )
    duration_seconds = _positive_float(
        _nested(pilot_row, "source", "cache_duration_seconds"),
        "pilot source.cache_duration_seconds",
    )

    # Inference phase. Do not access representative_redline_grid or maps here.
    candidate_started = time.perf_counter()
    candidates = tuple(
        candidate_generator(
            _inference_safe_pilot_row(pilot_row),
            projection_row,
            _inference_safe_baseline_row(baseline_row),
        )
    )
    _require_candidates(candidates)
    candidate_seconds = time.perf_counter() - candidate_started
    mel_started = time.perf_counter()
    mel_10ms, mel_source, mel_path = _load_log_mel_10ms(
        audio_path,
        repo_root=repo_root,
    )
    mel_seconds = time.perf_counter() - mel_started
    score_started = time.perf_counter()
    ranking = score_timing_candidates(
        mel_10ms,
        candidates,
        audio_duration_seconds=duration_seconds,
    )
    score_seconds = time.perf_counter() - score_started
    frozen_selection = FrozenInferenceSelection(
        inference_family=INFERENCE_FAMILY,
        candidate_fingerprints=tuple(candidate.fingerprint_sha256 for candidate in candidates),
        candidate_sources=tuple(
            str(getattr(candidate, "source", type(candidate).__name__)) for candidate in candidates
        ),
        selected_fingerprint_sha256=ranking.selected_fingerprint_sha256,
        selected_candidate_index=ranking.selected_candidate_index,
        unavailable_reason=ranking.unavailable_reason,
        raw_audio_candidate_scores=tuple(asdict(score) for score in ranking.candidate_scores),
    )
    inference_payload = frozen_selection.to_dict()
    inference_sha256 = _stable_json_sha256(inference_payload)
    inference_finished = time.perf_counter()

    selected_candidate = (
        None
        if frozen_selection.selected_candidate_index is None
        else candidates[frozen_selection.selected_candidate_index]
    )

    # Evaluation phase begins only after the inference fingerprint is frozen.
    evaluation_started = time.perf_counter()
    weak_oracle_payload: dict[str, Any]
    if selected_candidate is None:
        weak_oracle_payload = {
            "available": False,
            "unavailable_reason": "raw_audio_selection_unavailable",
            "metrics": None,
            "weak_oracle_ramp_accuracy": None,
        }
    else:
        try:
            weak_oracle_payload = _evaluate_selected_post_selection(
                selected_candidate=selected_candidate,
                pilot_row=pilot_row,
                projection_row=projection_row,
                baseline_row=baseline_row,
                duration_seconds=duration_seconds,
                weak_oracle_loader=weak_oracle_loader,
                frozen_inference_sha256=inference_sha256,
            )
        except Exception as exc:
            weak_oracle_payload = {
                "available": False,
                "unavailable_reason": "weak_oracle_evaluation_error",
                "error": f"{type(exc).__name__}: {exc}",
                "metrics": None,
                "weak_oracle_ramp_accuracy": None,
                "frozen_inference_sha256": inference_sha256,
            }
    evaluation_seconds = time.perf_counter() - evaluation_started

    return {
        "schema": REAL_AUDIO_PILOT_RESULT_SCHEMA,
        "row_index": row_index,
        "ok": True,
        "identity": {
            "cache_audio_key": cache_audio_key,
            "audio_group_index": pilot_row.get("audio_group_index"),
            "audio_group_key": pilot_row.get("audio_group_key"),
            "resolved_audio_path": audio_path.as_posix(),
        },
        "strata": {
            "pilot_stratum": pilot_row.get("pilot_stratum"),
            "label_stratum": _nested(pilot_row, "label", "stratum"),
            "label_confidence": _nested(pilot_row, "label", "confidence"),
            "label_ambiguous": _nested(pilot_row, "label", "ambiguous"),
            "duration_seconds": duration_seconds,
            "duration_le_600_seconds": duration_seconds <= 600.0,
        },
        "inference": {
            **inference_payload,
            "frozen_before_weak_oracle_load": True,
            "frozen_payload_sha256": inference_sha256,
            "mel_source": mel_source,
            "mel_path": mel_path.as_posix() if mel_path is not None else None,
        },
        "weak_oracle_evaluation": weak_oracle_payload,
        "runtime": {
            "candidate_seconds": candidate_seconds,
            "mel_load_or_compute_seconds": mel_seconds,
            "raw_audio_score_seconds": score_seconds,
            "inference_total_seconds": inference_finished - row_started,
            "weak_oracle_evaluation_seconds": evaluation_seconds,
            "total_seconds": time.perf_counter() - row_started,
        },
        "error": None,
    }


def _evaluate_selected_post_selection(
    *,
    selected_candidate: AnalyticTimingCandidate,
    pilot_row: Mapping[str, Any],
    projection_row: Mapping[str, Any],
    baseline_row: Mapping[str, Any] | None,
    duration_seconds: float,
    weak_oracle_loader: WeakOracleLoader,
    frozen_inference_sha256: str,
) -> dict[str, Any]:
    # This is the first function allowed to inspect evaluation-only label data.
    representative = pilot_row.get("representative_redline_grid")
    if not isinstance(representative, MappingABC):
        return {
            "available": False,
            "unavailable_reason": "representative_redline_grid_unavailable",
            "metrics": None,
            "weak_oracle_ramp_accuracy": None,
            "frozen_inference_sha256": frozen_inference_sha256,
        }
    beatmap_path_value = representative.get("beatmap_path")
    if not isinstance(beatmap_path_value, str) or not beatmap_path_value:
        return {
            "available": False,
            "unavailable_reason": "representative_beatmap_path_unavailable",
            "metrics": None,
            "weak_oracle_ramp_accuracy": None,
            "frozen_inference_sha256": frozen_inference_sha256,
        }
    weak_oracle_class = _weak_oracle_class(pilot_row, representative)
    weak_oracle = weak_oracle_loader(Path(beatmap_path_value))
    predicted = _unwrap_candidate(selected_candidate)
    coverage_start_ms = _coverage_start_ms(predicted, projection_row)
    coverage_end_ms = min(
        duration_seconds * 1000.0,
        _coverage_end_ms(predicted, projection_row),
    )
    selected_metrics = evaluate_curve_against_weak_oracle(
        predicted,
        weak_oracle,
        weak_oracle_class=weak_oracle_class,
        coverage_start_ms=coverage_start_ms,
        coverage_end_ms=coverage_end_ms,
    )

    baseline_metrics = None
    baseline_metrics_error = None
    if baseline_row is not None:
        baseline_grid = _baseline_grid(baseline_row)
        if baseline_grid is not None:
            try:
                baseline_metrics = evaluate_curve_against_weak_oracle(
                    baseline_grid,
                    weak_oracle,
                    weak_oracle_class=weak_oracle_class,
                    coverage_start_ms=coverage_start_ms,
                    coverage_end_ms=coverage_end_ms,
                ).to_dict()
            except ValueError as exc:
                # A stored v2 grid may begin after the v3 evaluation coverage;
                # comparator absence must not discard the already-valid
                # selected-candidate metrics.
                baseline_metrics_error = f"{type(exc).__name__}: {exc}"

    return {
        "available": True,
        "policy": WEAK_ORACLE_POLICY,
        "frozen_inference_sha256": frozen_inference_sha256,
        "weak_oracle_class": weak_oracle_class,
        "weak_oracle_beatmap_path": beatmap_path_value,
        "weak_oracle_agreement_rate": representative.get("agreement_rate"),
        "weak_oracle_ramp_accuracy": None,
        "selected_metrics": selected_metrics.to_dict(),
        "baseline_v2_metrics": baseline_metrics,
        "baseline_v2_metrics_error": baseline_metrics_error,
    }


def _load_log_mel_10ms(
    audio_path: Path,
    *,
    repo_root: Path,
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], str, Path | None]:
    resolved_audio = audio_path.expanduser().resolve(strict=False)
    try:
        audio_cache_key = resolved_audio.relative_to(repo_root).as_posix()
    except ValueError:
        audio_cache_key = resolved_audio.as_posix()
    cache_path = stage2_log_mel_cache_path(
        resolved_audio,
        audio_cache_key=audio_cache_key,
    )
    if cache_path.exists():
        return np.load(cache_path).astype(np.float32, copy=False), "existing_10ms_mel_cache", cache_path
    waveform = load_audio_file(
        resolved_audio,
        sample_rate=DEFAULT_MEL_CACHE_CONFIG.sample_rate,
    )
    mel = compute_log_mel_10ms(
        waveform,
        sample_rate=DEFAULT_MEL_CACHE_CONFIG.sample_rate,
    )
    return mel, "decoded_in_memory", None


def _unwrap_candidate(
    candidate: AnalyticTimingCandidate,
) -> TimingV3Grid | PhaseContinuousTimingCurve:
    if isinstance(candidate, _GridCandidate):
        return candidate.grid
    if isinstance(candidate, (TimingV3Grid, PhaseContinuousTimingCurve)):
        return candidate
    grid = getattr(candidate, "grid", None)
    if isinstance(grid, (TimingV3Grid, PhaseContinuousTimingCurve)):
        return grid
    raise TypeError("selected candidate must expose a TimingV3Grid or PhaseContinuousTimingCurve")


def _baseline_grid(row: Mapping[str, Any]) -> FittedTimingGrid | None:
    fit = row.get("fit")
    if not isinstance(fit, MappingABC):
        return None
    segments = fit.get("predicted_segments")
    if not isinstance(segments, Sequence) or isinstance(segments, (str, bytes)) or not segments:
        return None
    return FittedTimingGrid(
        tuple(
            TimingSegment(
                offset_ms=float(_require_mapping(segment, "baseline segment")["offset_ms"]),
                beat_length_ms=float(_require_mapping(segment, "baseline segment")["beat_length_ms"]),
                meter=int(_require_mapping(segment, "baseline segment").get("meter", 4)),
            )
            for segment in segments
        )
    )


def _weak_oracle_class(
    pilot_row: Mapping[str, Any],
    representative: Mapping[str, Any],
) -> WeakOracleClass:
    stratum = _nested(pilot_row, "label", "stratum") or pilot_row.get("pilot_stratum")
    evidence_class = representative.get("evidence_class")
    if stratum in ("ramp_candidate", "ramp") or evidence_class == "ramp_candidate":
        return "ramp_like"
    if stratum in ("jump_candidate", "jump") or evidence_class == "jump_candidate":
        return "jump"
    return "constant"


def _coverage_start_ms(
    predicted: TimingV3Grid | PhaseContinuousTimingCurve,
    projection_row: Mapping[str, Any],
) -> float:
    if isinstance(predicted, TimingV3Grid):
        return max(0.0, predicted.coverage_start_ms, predicted.start_time_ms)
    return max(
        0.0,
        predicted.start_time_ms,
        float(_nested(projection_row, "cache", "coverage_start_ms") or 0.0),
    )


def _coverage_end_ms(
    predicted: TimingV3Grid | PhaseContinuousTimingCurve,
    projection_row: Mapping[str, Any],
) -> float:
    projection_end = _nested(projection_row, "cache", "coverage_end_ms")
    declared_end_ms = (
        predicted.coverage_end_ms
        if isinstance(predicted, TimingV3Grid)
        else predicted.end_time_ms
    )
    end_ms = min(
        declared_end_ms,
        float(projection_end) if projection_end is not None else predicted.end_time_ms,
    )
    # Phase sampling is half-open.  Keep the mathematical endpoint outside
    # the requested interval so strict closed-domain curve queries remain
    # robust to small persisted coverage/boundary rounding differences.
    return float(np.nextafter(end_ms, -math.inf))


def _summary_payload(
    results: Sequence[Mapping[str, Any]],
    *,
    pilot_path: Path,
    projection_path: Path,
    baseline_path: Path,
    output_path: Path,
    total_seconds: float,
) -> dict[str, Any]:
    ok_rows = [row for row in results if row.get("ok")]
    available_rows = [
        row
        for row in ok_rows
        if bool(_nested(row, "weak_oracle_evaluation", "available"))
    ]
    runtime_rows = [row for row in ok_rows if isinstance(row.get("runtime"), MappingABC)]
    exact_counts = Counter()
    exact_denominators = Counter()
    for row in available_rows:
        if not _exact_accuracy_eligible(row):
            continue
        metrics = _nested(row, "weak_oracle_evaluation", "selected_metrics")
        if not isinstance(metrics, MappingABC):
            continue
        weak_class = metrics.get("weak_oracle_class")
        if weak_class == "constant":
            exact_denominators["constant"] += 1
            exact_counts["constant"] += int(bool(metrics.get("weak_oracle_constant_exact_hit")))
        elif weak_class == "jump":
            exact_denominators["jump"] += 1
            exact_counts["jump"] += int(bool(metrics.get("weak_oracle_jump_exact_hit")))

    return {
        "schema": REAL_AUDIO_PILOT_SUMMARY_SCHEMA,
        "inference_family": INFERENCE_FAMILY,
        "weak_oracle_policy": WEAK_ORACLE_POLICY,
        "source": {
            "pilot_jsonl_path": pilot_path.as_posix(),
            "pilot_jsonl_sha256": _file_sha256(pilot_path),
            "projection_jsonl_path": projection_path.as_posix(),
            "projection_jsonl_sha256": _file_sha256(projection_path),
            "baseline_jsonl_path": baseline_path.as_posix(),
            "baseline_jsonl_sha256": _file_sha256(baseline_path),
        },
        "output": {
            "output_jsonl_path": output_path.as_posix(),
            "row_count": len(results),
            "ok_count": len(ok_rows),
            "failed_count": len(results) - len(ok_rows),
            "weak_oracle_available_count": len(available_rows),
        },
        "weak_oracle_exact_accuracy": {
            "constant": _ratio_payload(exact_counts["constant"], exact_denominators["constant"]),
            "jump": _ratio_payload(exact_counts["jump"], exact_denominators["jump"]),
            "ramp": None,
            "ramp_reason": "ramp_like labels are low-confidence diagnostics, not timing truth",
        },
        "runtime": {
            "run_total_seconds": total_seconds,
            "inference_total_seconds": _stats(
                [float(_nested(row, "runtime", "inference_total_seconds")) for row in runtime_rows]
            ),
            "raw_audio_score_seconds": _stats(
                [float(_nested(row, "runtime", "raw_audio_score_seconds")) for row in runtime_rows]
            ),
            "total_seconds": _stats(
                [float(_nested(row, "runtime", "total_seconds")) for row in runtime_rows]
            ),
        },
    }


def _failure_row(
    *,
    row_index: int,
    cache_audio_key: str,
    pilot_row: Mapping[str, Any],
    failure_stage: str,
    error: str,
    row_started: float,
) -> dict[str, Any]:
    return {
        "schema": REAL_AUDIO_PILOT_RESULT_SCHEMA,
        "row_index": row_index,
        "ok": False,
        "identity": {
            "cache_audio_key": cache_audio_key,
            "audio_group_index": pilot_row.get("audio_group_index"),
            "audio_group_key": pilot_row.get("audio_group_key"),
            "resolved_audio_path": pilot_row.get("resolved_audio_path"),
        },
        "failure_stage": failure_stage,
        "error": error,
        "weak_oracle_evaluation": {
            "available": False,
            "metrics": None,
            "weak_oracle_ramp_accuracy": None,
        },
        "runtime": {"total_seconds": time.perf_counter() - row_started},
    }


def _inference_safe_pilot_row(pilot_row: Mapping[str, Any]) -> dict[str, Any]:
    """Return only source identity/duration fields; no labels, maps, or redlines."""

    source = pilot_row.get("source")
    safe_source = dict(source) if isinstance(source, MappingABC) else {}
    return {
        "audio_group_index": pilot_row.get("audio_group_index"),
        "audio_group_key": pilot_row.get("audio_group_key"),
        "resolved_audio_path": pilot_row.get("resolved_audio_path"),
        "source": safe_source,
    }


def _inference_safe_baseline_row(
    baseline_row: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Strip `.osu` paths, comparisons, paired metrics, and evaluation strata."""

    if baseline_row is None:
        return None
    return {
        key: baseline_row.get(key)
        for key in ("audio_key", "audio_path", "prediction", "fit", "runtime")
    }


def _exact_accuracy_eligible(row: Mapping[str, Any]) -> bool:
    return (
        _nested(row, "strata", "label_ambiguous") is False
        and _nested(row, "strata", "label_confidence") in ("high", "medium")
    )


def _require_candidates(candidates: Sequence[AnalyticTimingCandidate]) -> None:
    if not candidates:
        raise ValueError("candidate generator returned no candidates")
    fingerprints = []
    for index, candidate in enumerate(candidates):
        fingerprints.append(_require_sha256(candidate.fingerprint_sha256, f"candidate[{index}].fingerprint"))
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("candidate fingerprints must be unique")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def _index_rows(
    rows: Sequence[dict[str, Any]],
    *,
    key_getter: Callable[[Mapping[str, Any]], str],
    source_name: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = key_getter(row)
        if key in indexed:
            raise ValueError(f"duplicate {source_name} key: {key!r}")
        indexed[key] = row
    return indexed


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    os.replace(tmp, path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "p50": None, "p90": None, "max": None}
    return {
        "count": int(finite.size),
        "p50": float(np.percentile(finite, 50.0)),
        "p90": float(np.percentile(finite, 90.0)),
        "max": float(np.max(finite)),
    }


def _ratio_payload(numerator: int, denominator: int) -> dict[str, float | int | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": None if denominator == 0 else numerator / denominator,
    }


def _stable_json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nested(payload: Mapping[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, MappingABC):
            return None
        value = value.get(key)
    return value


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, MappingABC):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: object, name: str) -> str:
    result = _require_string(value, name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return result


def _positive_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank stored Timing v3 CJ candidates with deterministic raw-audio evidence, then evaluate post hoc.",
    )
    parser.add_argument("--pilot-jsonl", required=True, type=Path)
    parser.add_argument("--projection-jsonl", required=True, type=Path)
    parser.add_argument("--baseline-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    summary = run_real_audio_pilot(
        pilot_jsonl_path=args.pilot_jsonl,
        projection_jsonl_path=args.projection_jsonl,
        baseline_jsonl_path=args.baseline_jsonl,
        output_jsonl_path=args.output_jsonl,
        summary_json_path=args.summary_json,
        repo_root=args.repo_root,
        limit=args.limit,
    )
    if args.json:
        print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))
    else:
        output = summary["output"]
        print(
            "Timing v3 real-audio pilot: "
            f"{output['ok_count']}/{output['row_count']} rows ok, "
            f"weak-oracle available={output['weak_oracle_available_count']}, "
            "ramp_accuracy=null"
        )
    return 1 if int(summary["output"]["failed_count"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
