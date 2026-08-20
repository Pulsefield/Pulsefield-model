from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from pulsefield_model.features.mel import load_full_song_packed_mel_20ms
from pulsefield_model.timing.evaluation.full5050_shadow_runner import (
    DEFAULT_EXPECTED_FULL5050_ROW_COUNT,
    DEFAULT_FULL5050_LABELS_PATH,
    Full5050LocatorRow,
    load_full5050_locator_rows,
)
from pulsefield_model.timing.providers.beatthis_cache import (
    BeatThisFramePredictionCacheConfig,
    load_beatthis_frame_prediction_cache,
)
from pulsefield_model.timing.schema import FrameTimingPrediction
from pulsefield_model.timing.v3.audio_evidence import extract_raw_audio_evidence
from pulsefield_model.timing.v3.inference import unpack_packed_mel_20ms_to_log_mel_10ms


FULL5050_AUDIO_CONSENSUS_PLAN_SCHEMA = (
    "pulsefield_model.timing_v3_full5050_audio_consensus_plan_v1"
)
FULL5050_AUDIO_CONSENSUS_RESULT_SCHEMA = (
    "pulsefield_model.timing_v3_full5050_audio_consensus_result_v1"
)
FULL5050_AUDIO_CONSENSUS_SUMMARY_SCHEMA = (
    "pulsefield_model.timing_v3_full5050_audio_consensus_summary_v1"
)
FULL5050_AUDIO_CONSENSUS_SELECTION_SCHEMA = (
    "pulsefield_model.timing_v3_full5050_audio_consensus_selection_v1"
)
FULL5050_AUDIO_CHANGE_SELECTION_SCHEMA = (
    "pulsefield_model.timing_v3_full5050_audio_change_selection_v1"
)
FULL5050_BEATNET_EVENTS_SCHEMA = "pulsefield_model.timing_v3_full5050_beatnet_events_v1"

DEFAULT_FULL5050_AUDIO_CONSENSUS_OUTPUT_JSONL = Path(
    "artifacts/reports/timing/timing_v3_full5050_audio_consensus_results.jsonl",
)
DEFAULT_FULL5050_BEATNET_EVENTS_JSONL = Path(
    "artifacts/reports/timing/timing_v3_full5050_beatnet_model3_dbn_events.jsonl",
)
DEFAULT_FULL5050_AUDIO_CONSENSUS_SELECTED_JSONL = Path(
    "artifacts/reports/timing/timing_v3_full5050_audio_consensus_constant_sources_256.jsonl",
)
DEFAULT_FULL5050_AUDIO_CHANGE_SELECTED_JSONL = Path(
    "artifacts/reports/timing/timing_v3_full5050_audio_consensus_natural_change_sources_100.jsonl",
)
DEFAULT_TRAIN_SOURCE_COUNT = 128
DEFAULT_HOLDOUT_SOURCE_COUNT = 128
DEFAULT_CHANGE_TRAIN_SOURCE_COUNT = 50
DEFAULT_CHANGE_HOLDOUT_SOURCE_COUNT = 50

BEATTHIS_VIEW_NAME = "beatthis_probabilities"
RAW_FLUX_VIEW_NAME = "raw_log_mel_flux"
BEATNET_VIEW_NAME = "beatnet_model3_dbn"
CONSENSUS_VIEW_NAMES = (BEATTHIS_VIEW_NAME, RAW_FLUX_VIEW_NAME, BEATNET_VIEW_NAME)

_FINAL_ROW_STATUSES = frozenset({"completed", "failed", "skipped_duration"})
_TIMING_STAGE_NAMES = ("beatthis", "raw_flux", "beatnet", "screen", "total")


def _require_positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _require_nonnegative_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return number


def _require_probability(value: object, name: str) -> float:
    number = _require_nonnegative_finite(value, name)
    if number > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return number


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


class BeatNetBeatTimeExtractor(Protocol):
    def extract_beat_times(self, audio_path: str | Path) -> Sequence[float]:
        ...


BeatThisCacheLoader = Callable[
    [str, BeatThisFramePredictionCacheConfig],
    FrameTimingPrediction | None,
]
PackedMelLoader = Callable[..., NDArray[np.float32]]
Clock = Callable[[], float]


class Full5050AudioConsensusError(RuntimeError):
    pass


class MissingBeatThisCacheError(Full5050AudioConsensusError):
    pass


class MissingBeatNetEventsError(Full5050AudioConsensusError):
    pass


class ConstantSourceSelectionError(Full5050AudioConsensusError):
    pass


@dataclass(frozen=True)
class AudioConsensusScreenConfig:
    """Frozen audio-only stability thresholds for the full5050 source screen."""

    min_duration_seconds: float = 90.0
    max_duration_seconds: float = 600.0
    window_seconds: float = 30.0
    hop_seconds: float = 15.0
    min_complete_windows: int = 5
    tempo_min_bpm: float = 45.0
    tempo_max_bpm: float = 240.0
    octave_family_min_bpm: float = 90.0
    octave_family_max_bpm: float = 180.0
    family_tolerance_octaves: float = 0.06
    global_min_confidence: float = 0.07
    window_min_confidence: float = 0.04
    min_confident_window_ratio: float = 0.80
    persistent_run_windows: int = 3
    beatnet_impulse_frame_rate_hz: float = 100.0

    def __post_init__(self) -> None:
        _require_positive_finite(self.min_duration_seconds, "min_duration_seconds")
        _require_positive_finite(self.max_duration_seconds, "max_duration_seconds")
        if self.min_duration_seconds > self.max_duration_seconds:
            raise ValueError("min_duration_seconds cannot exceed max_duration_seconds")
        _require_positive_finite(self.window_seconds, "window_seconds")
        _require_positive_finite(self.hop_seconds, "hop_seconds")
        _require_positive_finite(self.tempo_min_bpm, "tempo_min_bpm")
        _require_positive_finite(self.tempo_max_bpm, "tempo_max_bpm")
        if self.tempo_min_bpm >= self.tempo_max_bpm:
            raise ValueError("tempo_min_bpm must be less than tempo_max_bpm")
        _require_positive_finite(self.octave_family_min_bpm, "octave_family_min_bpm")
        _require_positive_finite(self.octave_family_max_bpm, "octave_family_max_bpm")
        if self.octave_family_min_bpm >= self.octave_family_max_bpm:
            raise ValueError(
                "octave_family_min_bpm must be less than octave_family_max_bpm",
            )
        if self.octave_family_min_bpm < self.tempo_min_bpm:
            raise ValueError("octave_family_min_bpm cannot be below tempo_min_bpm")
        if self.octave_family_max_bpm > self.tempo_max_bpm:
            raise ValueError("octave_family_max_bpm cannot exceed tempo_max_bpm")
        _require_nonnegative_finite(self.family_tolerance_octaves, "family_tolerance_octaves")
        _require_nonnegative_finite(self.global_min_confidence, "global_min_confidence")
        _require_nonnegative_finite(self.window_min_confidence, "window_min_confidence")
        _require_probability(self.min_confident_window_ratio, "min_confident_window_ratio")
        _require_positive_int(self.min_complete_windows, "min_complete_windows")
        _require_positive_int(self.persistent_run_windows, "persistent_run_windows")
        _require_positive_finite(
            self.beatnet_impulse_frame_rate_hz,
            "beatnet_impulse_frame_rate_hz",
        )


@dataclass(frozen=True)
class TempoFamilyEstimate:
    direct_bpm: float | None
    octave_family_bpm: float | None
    confidence: float
    lag_seconds: float | None
    score: float | None
    status: str


@dataclass(frozen=True)
class PersistentChangeEvidence:
    boundary_start_seconds: float
    boundary_end_seconds: float
    left_direct_bpm: float
    right_direct_bpm: float
    left_octave_family_bpm: float
    right_octave_family_bpm: float
    signed_ratio_octaves: float
    confidence: float
    left_window_count: int
    right_window_count: int


@dataclass(frozen=True)
class ViewStabilityResult:
    view_name: str
    status: str
    stable: bool
    reason: str | None
    global_estimate: TempoFamilyEstimate
    window_estimates: tuple[TempoFamilyEstimate, ...]
    window_count: int
    confident_window_count: int
    confident_window_ratio: float
    max_persistent_mismatch_run: int
    max_low_confidence_run: int
    median_window_direct_bpm: float | None
    median_window_family_bpm: float | None
    persistent_change: PersistentChangeEvidence | None = None


@dataclass(frozen=True)
class AudioConsensusResult:
    accepted_constant_source: bool
    reason: str | None
    global_octave_family_bpm: float | None
    max_cross_view_family_distance_octaves: float | None
    confidence_score: float
    confidence_floor: float
    views: Mapping[str, ViewStabilityResult]


@dataclass(frozen=True)
class NaturalChangeConsensusResult:
    accepted_natural_change_source: bool
    reason: str | None
    boundary_start_seconds: float | None
    boundary_end_seconds: float | None
    signed_ratio_octaves: float | None
    family_ratio: float | None
    max_cross_view_ratio_distance_octaves: float | None
    confidence_score: float
    confidence_floor: float
    views: Mapping[str, ViewStabilityResult]


@dataclass(frozen=True)
class Full5050AudioConsensusRunnerConfig:
    labels_path: Path = DEFAULT_FULL5050_LABELS_PATH
    output_jsonl: Path = DEFAULT_FULL5050_AUDIO_CONSENSUS_OUTPUT_JSONL
    beatnet_events_jsonl: Path | None = DEFAULT_FULL5050_BEATNET_EVENTS_JSONL
    summary_json: Path | None = None
    expected_row_count: int = DEFAULT_EXPECTED_FULL5050_ROW_COUNT
    retry_failed: bool = False
    screen: AudioConsensusScreenConfig = AudioConsensusScreenConfig()

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels_path", Path(self.labels_path))
        object.__setattr__(self, "output_jsonl", Path(self.output_jsonl))
        if self.beatnet_events_jsonl is not None:
            object.__setattr__(self, "beatnet_events_jsonl", Path(self.beatnet_events_jsonl))
        if self.summary_json is not None:
            object.__setattr__(self, "summary_json", Path(self.summary_json))
        _require_positive_int(self.expected_row_count, "expected_row_count")
        if not isinstance(self.retry_failed, bool):
            raise TypeError("retry_failed must be a boolean")
        if not isinstance(self.screen, AudioConsensusScreenConfig):
            raise TypeError("screen must be AudioConsensusScreenConfig")


@dataclass
class Full5050AudioConsensusPipeline:
    beatnet_extractor: BeatNetBeatTimeExtractor
    beatthis_cache_config: BeatThisFramePredictionCacheConfig = field(
        default_factory=BeatThisFramePredictionCacheConfig,
    )
    beatthis_cache_loader: BeatThisCacheLoader = load_beatthis_frame_prediction_cache
    mel_loader: PackedMelLoader = load_full_song_packed_mel_20ms
    clock: Clock = time.perf_counter


class BeatNetModel3DBNExtractor:
    """Lazy optional adapter for the official BeatNet model=3 offline DBN path."""

    def __init__(self, *, device: str = "cpu") -> None:
        self.device = str(device)
        self._estimator: object | None = None

    def extract_beat_times(self, audio_path: str | Path) -> Sequence[float]:
        estimator = self._get_estimator()
        output = estimator.process(Path(audio_path).as_posix())
        array = np.asarray(output, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] < 1:
            raise ValueError(f"BeatNet output must have shape [beats, columns], got {array.shape}")
        beat_times = array[:, 0]
        if beat_times.size and not np.all(np.diff(beat_times) > 0.0):
            beat_times = np.unique(beat_times)
        return beat_times.tolist()

    def _get_estimator(self) -> object:
        if self._estimator is None:
            try:
                from BeatNet.BeatNet import BeatNet  # type: ignore[import-not-found]
            except Exception as exc:  # pragma: no cover - optional external package.
                raise RuntimeError(
                    "BeatNet is required for beatnet_model3_dbn extraction; "
                    "install BeatNet or inject a BeatNetBeatTimeExtractor",
                ) from exc
            self._estimator = BeatNet(
                3,
                mode="offline",
                inference_model="DBN",
                plot=[],
                thread=False,
                device=self.device,
            )
        return self._estimator


@dataclass(frozen=True)
class PrecomputedBeatNetEventRow:
    row_index: int
    row_id: str
    resolved_audio_path: Path
    duration_seconds: float
    status: str
    beat_times_seconds: tuple[float, ...]
    error_type: str | None = None
    error_message: str | None = None


class PrecomputedBeatNetEventStore:
    def __init__(
        self,
        event_rows: Mapping[int, PrecomputedBeatNetEventRow],
        *,
        expected_row_count: int,
    ) -> None:
        _require_positive_int(expected_row_count, "expected_row_count")
        self._event_rows = dict(event_rows)
        self.expected_row_count = expected_row_count

    def extract_beat_times(self, audio_path: str | Path) -> Sequence[float]:
        del audio_path
        raise RuntimeError("PrecomputedBeatNetEventStore requires row-index lookup")

    def beat_times_for_row(self, row: Full5050LocatorRow) -> tuple[float, ...]:
        event = self._event_rows.get(row.row_index)
        if event is None:
            raise MissingBeatNetEventsError(f"BeatNet events missing for {row.row_id}")
        _validate_precomputed_event_matches_row(event, row)
        if event.status == "completed":
            if not event.beat_times_seconds:
                raise MissingBeatNetEventsError(f"BeatNet events are empty for {row.row_id}")
            return event.beat_times_seconds
        raise MissingBeatNetEventsError(
            f"BeatNet precompute {event.status} for {row.row_id}: "
            f"{event.error_type or 'unknown'} {event.error_message or ''}".strip(),
        )


def load_precomputed_beatnet_event_store(
    beatnet_events_jsonl: str | Path,
    *,
    locator_rows: Sequence[Full5050LocatorRow],
    expected_row_count: int = DEFAULT_EXPECTED_FULL5050_ROW_COUNT,
) -> PrecomputedBeatNetEventStore:
    path = Path(beatnet_events_jsonl)
    if not path.is_file():
        raise FileNotFoundError(f"BeatNet events JSONL not found: {path}")
    _require_positive_int(expected_row_count, "expected_row_count")
    if len(locator_rows) != expected_row_count:
        raise ValueError(
            f"locator_rows must contain exactly {expected_row_count} rows, got {len(locator_rows)}",
        )
    rows_by_index = {row.row_index: row for row in locator_rows}
    event_rows: dict[int, PrecomputedBeatNetEventRow] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"BeatNet events line {line_number} must be a JSON object")
            event = _precomputed_beatnet_event_from_mapping(payload, line_number=line_number)
            if event.row_index in event_rows:
                raise ValueError(f"duplicate BeatNet events row_index {event.row_index}")
            locator_row = rows_by_index.get(event.row_index)
            if locator_row is None:
                raise ValueError(f"BeatNet events row_index {event.row_index} is not in locator rows")
            _validate_precomputed_event_matches_row(event, locator_row)
            event_rows[event.row_index] = event
    missing = sorted(set(rows_by_index) - set(event_rows))
    if missing:
        preview = ",".join(str(index) for index in missing[:10])
        raise ValueError(f"BeatNet events JSONL is incomplete; missing row indexes: {preview}")
    if len(event_rows) != expected_row_count:
        raise ValueError(
            f"BeatNet events JSONL must contain exactly {expected_row_count} rows, got {len(event_rows)}",
        )
    return PrecomputedBeatNetEventStore(event_rows, expected_row_count=expected_row_count)


def screen_audio_consensus(
    views: Mapping[str, ViewStabilityResult],
    *,
    config: AudioConsensusScreenConfig = AudioConsensusScreenConfig(),
) -> AudioConsensusResult:
    missing = [name for name in CONSENSUS_VIEW_NAMES if name not in views]
    if missing:
        return AudioConsensusResult(
            accepted_constant_source=False,
            reason=f"missing_views:{','.join(missing)}",
            global_octave_family_bpm=None,
            max_cross_view_family_distance_octaves=None,
            confidence_score=0.0,
            confidence_floor=0.0,
            views=dict(views),
        )

    ordered_views = tuple(views[name] for name in CONSENSUS_VIEW_NAMES)
    for view in ordered_views:
        if not view.stable:
            return AudioConsensusResult(
                accepted_constant_source=False,
                reason=f"{view.view_name}:{view.reason or view.status}",
                global_octave_family_bpm=None,
                max_cross_view_family_distance_octaves=None,
                confidence_score=0.0,
                confidence_floor=min(_view_confidence_floor(v) for v in ordered_views),
                views={name: views[name] for name in CONSENSUS_VIEW_NAMES},
            )

    families = tuple(view.global_estimate.octave_family_bpm for view in ordered_views)
    if any(family is None for family in families):
        return AudioConsensusResult(
            accepted_constant_source=False,
            reason="missing_global_family",
            global_octave_family_bpm=None,
            max_cross_view_family_distance_octaves=None,
            confidence_score=0.0,
            confidence_floor=min(_view_confidence_floor(v) for v in ordered_views),
            views={name: views[name] for name in CONSENSUS_VIEW_NAMES},
        )
    family_values = tuple(float(family) for family in families if family is not None)
    pairwise_distances = [
        octave_family_distance_octaves(left, right)
        for index, left in enumerate(family_values)
        for right in family_values[index + 1 :]
    ]
    max_distance = max(pairwise_distances, default=0.0)
    if max_distance > config.family_tolerance_octaves:
        return AudioConsensusResult(
            accepted_constant_source=False,
            reason="cross_view_family_mismatch",
            global_octave_family_bpm=None,
            max_cross_view_family_distance_octaves=max_distance,
            confidence_score=0.0,
            confidence_floor=min(_view_confidence_floor(v) for v in ordered_views),
            views={name: views[name] for name in CONSENSUS_VIEW_NAMES},
        )

    confidence_floor = min(_view_confidence_floor(view) for view in ordered_views)
    confidence_score = confidence_floor / (1.0 + max_distance)
    return AudioConsensusResult(
        accepted_constant_source=True,
        reason=None,
        global_octave_family_bpm=float(np.median(np.asarray(family_values, dtype=np.float64))),
        max_cross_view_family_distance_octaves=max_distance,
        confidence_score=float(confidence_score),
        confidence_floor=float(confidence_floor),
        views={name: views[name] for name in CONSENSUS_VIEW_NAMES},
    )


def screen_natural_change_consensus(
    views: Mapping[str, ViewStabilityResult],
    *,
    config: AudioConsensusScreenConfig = AudioConsensusScreenConfig(),
) -> NaturalChangeConsensusResult:
    missing = [name for name in CONSENSUS_VIEW_NAMES if name not in views]
    if missing:
        return _natural_change_reject(
            views,
            reason=f"missing_views:{','.join(missing)}",
        )

    ordered_views = tuple(views[name] for name in CONSENSUS_VIEW_NAMES)
    changes: list[PersistentChangeEvidence] = []
    for view in ordered_views:
        if view.persistent_change is None:
            return _natural_change_reject(
                {name: views[name] for name in CONSENSUS_VIEW_NAMES},
                reason=f"{view.view_name}:no_persistent_family_change",
            )
        changes.append(view.persistent_change)

    boundary_start = max(change.boundary_start_seconds for change in changes)
    boundary_end = min(change.boundary_end_seconds for change in changes)
    if boundary_start > boundary_end:
        return _natural_change_reject(
            {name: views[name] for name in CONSENSUS_VIEW_NAMES},
            reason="boundary_intersection_empty",
            confidence_floor=min(change.confidence for change in changes),
        )

    signed_ratios = tuple(change.signed_ratio_octaves for change in changes)
    ratio_distances = [
        abs(left - right)
        for index, left in enumerate(signed_ratios)
        for right in signed_ratios[index + 1 :]
    ]
    max_ratio_distance = max(ratio_distances, default=0.0)
    if max_ratio_distance > config.family_tolerance_octaves:
        return _natural_change_reject(
            {name: views[name] for name in CONSENSUS_VIEW_NAMES},
            reason="cross_view_ratio_mismatch",
            confidence_floor=min(change.confidence for change in changes),
            max_ratio_distance=max_ratio_distance,
        )

    confidence_floor = min(change.confidence for change in changes)
    signed_ratio = float(np.median(np.asarray(signed_ratios, dtype=np.float64)))
    confidence_score = confidence_floor / (1.0 + max_ratio_distance)
    return NaturalChangeConsensusResult(
        accepted_natural_change_source=True,
        reason=None,
        boundary_start_seconds=float(boundary_start),
        boundary_end_seconds=float(boundary_end),
        signed_ratio_octaves=signed_ratio,
        family_ratio=float(2.0**signed_ratio),
        max_cross_view_ratio_distance_octaves=float(max_ratio_distance),
        confidence_score=float(confidence_score),
        confidence_floor=float(confidence_floor),
        views={name: views[name] for name in CONSENSUS_VIEW_NAMES},
    )


def evaluate_view_stability(
    view_name: str,
    salience: object,
    *,
    frame_rate_hz: float,
    duration_seconds: float | None = None,
    config: AudioConsensusScreenConfig = AudioConsensusScreenConfig(),
) -> ViewStabilityResult:
    signal = _require_salience_vector(salience, name=view_name)
    frame_rate = _require_positive_finite(frame_rate_hz, "frame_rate_hz")
    duration = (
        signal.shape[0] / frame_rate
        if duration_seconds is None
        else _require_positive_finite(duration_seconds, "duration_seconds")
    )
    if duration < config.min_duration_seconds:
        return _view_result(
            view_name,
            status="skipped_duration",
            stable=False,
            reason="duration_below_minimum",
        )
    if duration > config.max_duration_seconds:
        return _view_result(
            view_name,
            status="skipped_duration",
            stable=False,
            reason="duration_above_maximum",
        )

    global_estimate = estimate_tempo_family_from_salience(
        signal,
        frame_rate_hz=frame_rate,
        config=config,
    )
    if global_estimate.status != "ok":
        return _view_result(
            view_name,
            status="failed",
            stable=False,
            reason=f"global_{global_estimate.status}",
            global_estimate=global_estimate,
        )
    if global_estimate.confidence < config.global_min_confidence:
        return _view_result(
            view_name,
            status="unstable",
            stable=False,
            reason="global_confidence_below_threshold",
            global_estimate=global_estimate,
        )

    windows = _window_slices(
        frame_count=signal.shape[0],
        frame_rate_hz=frame_rate,
        window_seconds=config.window_seconds,
        hop_seconds=config.hop_seconds,
    )
    if len(windows) < config.min_complete_windows:
        return _view_result(
            view_name,
            status="unstable",
            stable=False,
            reason="insufficient_complete_windows",
            global_estimate=global_estimate,
            window_estimates=(),
        )

    window_estimates = tuple(
        estimate_tempo_family_from_salience(signal[start:end], frame_rate_hz=frame_rate, config=config)
        for start, end in windows
    )
    global_direct_bpm = global_estimate.direct_bpm
    confident_flags: list[bool] = []
    mismatch_flags: list[bool] = []
    window_direct_bpms: list[float] = []
    window_families: list[float] = []
    for estimate in window_estimates:
        confident = (
            estimate.status == "ok"
            and estimate.direct_bpm is not None
            and estimate.confidence >= config.window_min_confidence
        )
        confident_flags.append(confident)
        if confident:
            window_direct_bpms.append(float(estimate.direct_bpm))
        if confident and estimate.octave_family_bpm is not None:
            window_families.append(float(estimate.octave_family_bpm))
        mismatch_flags.append(
            confident
            and global_direct_bpm is not None
            and estimate.direct_bpm is not None
            and direct_tempo_distance_octaves(
                float(global_direct_bpm),
                float(estimate.direct_bpm),
            )
            > config.family_tolerance_octaves
        )

    confident_count = sum(1 for flag in confident_flags if flag)
    confident_ratio = confident_count / len(window_estimates)
    max_low_conf_run = _max_true_run([not flag for flag in confident_flags])
    max_mismatch_run = _max_true_run(mismatch_flags)
    persistent_change = _detect_persistent_change(
        window_estimates,
        windows=windows,
        frame_rate_hz=frame_rate,
        config=config,
    )

    stable = True
    reason = None
    if confident_ratio < config.min_confident_window_ratio:
        stable = False
        reason = "confident_window_ratio_below_threshold"
    elif max_low_conf_run >= config.persistent_run_windows:
        stable = False
        reason = "persistent_low_confidence_windows"
    elif persistent_change is not None or max_mismatch_run >= config.persistent_run_windows:
        stable = False
        reason = "persistent_family_change"

    return ViewStabilityResult(
        view_name=view_name,
        status="stable" if stable else "unstable",
        stable=stable,
        reason=reason,
        global_estimate=global_estimate,
        window_estimates=window_estimates,
        window_count=len(window_estimates),
        confident_window_count=confident_count,
        confident_window_ratio=float(confident_ratio),
        max_persistent_mismatch_run=max_mismatch_run,
        max_low_confidence_run=max_low_conf_run,
        median_window_direct_bpm=None
        if not window_direct_bpms
        else float(np.median(np.asarray(window_direct_bpms, dtype=np.float64))),
        median_window_family_bpm=None
        if not window_families
        else float(np.median(np.asarray(window_families, dtype=np.float64))),
        persistent_change=persistent_change,
    )


def estimate_tempo_family_from_salience(
    salience: object,
    *,
    frame_rate_hz: float,
    config: AudioConsensusScreenConfig = AudioConsensusScreenConfig(),
) -> TempoFamilyEstimate:
    signal = _require_salience_vector(salience, name="salience")
    frame_rate = _require_positive_finite(frame_rate_hz, "frame_rate_hz")
    if signal.size < 3:
        return TempoFamilyEstimate(None, None, 0.0, None, None, "insufficient_frames")
    centered = signal.astype(np.float64, copy=False)
    centered = centered - float(np.mean(centered))
    std = float(np.std(centered))
    if not math.isfinite(std) or std <= 1e-12:
        return TempoFamilyEstimate(None, None, 0.0, None, None, "flat_salience")
    centered = centered / std

    min_lag = max(1, int(math.floor(frame_rate * 60.0 / config.tempo_max_bpm)))
    max_lag = min(
        signal.size - 2,
        int(math.ceil(frame_rate * 60.0 / config.tempo_min_bpm)),
    )
    if min_lag > max_lag:
        return TempoFamilyEstimate(None, None, 0.0, None, None, "insufficient_lag_domain")

    correlations = np.empty(max_lag - min_lag + 1, dtype=np.float64)
    lags = np.arange(min_lag, max_lag + 1, dtype=np.int64)
    for output_index, lag in enumerate(lags):
        left = centered[:-lag]
        right = centered[lag:]
        denominator = math.sqrt(float(np.dot(left, left)) * float(np.dot(right, right)))
        correlations[output_index] = 0.0 if denominator <= 1e-12 else float(np.dot(left, right)) / denominator

    if not np.all(np.isfinite(correlations)):
        return TempoFamilyEstimate(None, None, 0.0, None, None, "nonfinite_autocorrelation")
    best_index = int(np.argmax(correlations))
    best_lag = int(lags[best_index])
    best_score = float(correlations[best_index])
    median_score = float(np.median(correlations))
    confidence = max(0.0, best_score - median_score)
    direct_bpm = 60.0 * frame_rate / float(best_lag)
    family_bpm = octave_family_bpm(direct_bpm, config=config)
    return TempoFamilyEstimate(
        direct_bpm=float(direct_bpm),
        octave_family_bpm=family_bpm,
        confidence=float(confidence),
        lag_seconds=float(best_lag / frame_rate),
        score=best_score,
        status="ok",
    )


def octave_family_bpm(
    bpm: float,
    *,
    config: AudioConsensusScreenConfig = AudioConsensusScreenConfig(),
) -> float:
    value = _require_positive_finite(bpm, "bpm")
    while value < config.octave_family_min_bpm:
        value *= 2.0
    while value >= config.octave_family_max_bpm:
        value *= 0.5
    return float(value)


def octave_family_distance_octaves(left_bpm: float, right_bpm: float) -> float:
    left = _require_positive_finite(left_bpm, "left_bpm")
    right = _require_positive_finite(right_bpm, "right_bpm")
    return abs(math.log2(left / right))


def direct_tempo_distance_octaves(left_bpm: float, right_bpm: float) -> float:
    left = _require_positive_finite(left_bpm, "left_bpm")
    right = _require_positive_finite(right_bpm, "right_bpm")
    return abs(math.log2(left / right))


def beat_times_to_impulse_salience(
    beat_times_seconds: object,
    *,
    duration_seconds: float,
    frame_rate_hz: float,
) -> NDArray[np.float32]:
    times = np.asarray(beat_times_seconds, dtype=np.float64)
    if times.ndim != 1:
        raise ValueError("beat_times_seconds must be a 1-D vector")
    if not np.all(np.isfinite(times)):
        raise ValueError("beat_times_seconds must contain only finite values")
    duration = _require_positive_finite(duration_seconds, "duration_seconds")
    frame_rate = _require_positive_finite(frame_rate_hz, "frame_rate_hz")
    frame_count = max(1, int(math.ceil(duration * frame_rate)))
    salience = np.zeros(frame_count, dtype=np.float32)
    if times.size == 0:
        return salience
    in_bounds = times[(times >= 0.0) & (times < duration)]
    if in_bounds.size == 0:
        return salience
    indices = np.rint(in_bounds * frame_rate).astype(np.int64)
    indices = np.clip(indices, 0, frame_count - 1)
    np.add.at(salience, indices, np.float32(1.0))
    return salience


def run_full5050_audio_consensus(
    config: Full5050AudioConsensusRunnerConfig,
    *,
    pipeline: Full5050AudioConsensusPipeline,
) -> dict[str, object]:
    rows = load_full5050_locator_rows(
        config.labels_path,
        expected_row_count=config.expected_row_count,
    )
    if config.beatnet_events_jsonl is None:
        raise MissingBeatNetEventsError(
            "full5050 audio consensus run requires a complete precomputed BeatNet events JSONL",
        )
    active_pipeline = Full5050AudioConsensusPipeline(
        beatnet_extractor=load_precomputed_beatnet_event_store(
            config.beatnet_events_jsonl,
            locator_rows=rows,
            expected_row_count=config.expected_row_count,
        ),
        beatthis_cache_config=pipeline.beatthis_cache_config,
        beatthis_cache_loader=pipeline.beatthis_cache_loader,
        mel_loader=pipeline.mel_loader,
        clock=pipeline.clock,
    )
    completed = _read_existing_result_row_indexes(
        config.output_jsonl,
        retry_failed=config.retry_failed,
    )
    started = active_pipeline.clock()
    resumed_rows = 0
    attempted_rows = 0
    config.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with config.output_jsonl.open("a", encoding="utf-8") as output:
        for row in rows:
            if row.row_index in completed:
                resumed_rows += 1
                continue
            attempted_rows += 1
            result = run_full5050_audio_consensus_row(
                row,
                pipeline=active_pipeline,
                config=config.screen,
            )
            output.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
            output.flush()

    full_counts = _summarize_audio_consensus_result_file(
        config.output_jsonl,
        expected_row_count=config.expected_row_count,
    )
    summary = {
        "schema": FULL5050_AUDIO_CONSENSUS_SUMMARY_SCHEMA,
        "labels_path": config.labels_path.as_posix(),
        "output_jsonl": config.output_jsonl.as_posix(),
        "expected_rows": config.expected_row_count,
        "total_rows": len(rows),
        "completed_rows": full_counts["completed"],
        "failed_rows": full_counts["failed"],
        "skipped_duration_rows": full_counts["skipped_duration"],
        "accepted_rows": full_counts["accepted_constant"],
        "accepted_natural_change_rows": full_counts["accepted_natural_change"],
        "resumed_rows": resumed_rows,
        "attempted_rows": attempted_rows,
        "elapsed_ms": _elapsed_ms(started, active_pipeline.clock),
        "screen": _screen_config_payload(config.screen),
        "beatnet_events_jsonl": None
        if config.beatnet_events_jsonl is None
        else config.beatnet_events_jsonl.as_posix(),
    }
    if config.summary_json is not None:
        config.summary_json.parent.mkdir(parents=True, exist_ok=True)
        config.summary_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def run_full5050_audio_consensus_row(
    row: Full5050LocatorRow,
    *,
    pipeline: Full5050AudioConsensusPipeline,
    config: AudioConsensusScreenConfig = AudioConsensusScreenConfig(),
) -> dict[str, object]:
    timings: dict[str, float | None] = {stage: None for stage in _TIMING_STAGE_NAMES}
    started = pipeline.clock()
    stage = "beatthis"
    try:
        if row.input_status is not None and row.input_status != "valid":
            raise ValueError(f"input cache status is not valid: {row.input_status}")
        if row.duration_seconds < config.min_duration_seconds:
            timings["total"] = _elapsed_ms(started, pipeline.clock)
            return _duration_skip_payload(row, timings=timings, reason="duration_below_minimum")
        if row.duration_seconds > config.max_duration_seconds:
            timings["total"] = _elapsed_ms(started, pipeline.clock)
            return _duration_skip_payload(row, timings=timings, reason="duration_above_maximum")

        stage_started = pipeline.clock()
        beatthis_prediction = _load_beatthis_prediction(row, pipeline)
        beatthis_view = evaluate_view_stability(
            BEATTHIS_VIEW_NAME,
            beatthis_prediction.beat_prob,
            frame_rate_hz=beatthis_prediction.frame_rate_hz,
            duration_seconds=row.duration_seconds,
            config=config,
        )
        timings["beatthis"] = _elapsed_ms(stage_started, pipeline.clock)

        stage = "raw_flux"
        stage_started = pipeline.clock()
        packed_mel = _load_packed_mel(row, pipeline)
        raw_log_mel_10ms = unpack_packed_mel_20ms_to_log_mel_10ms(packed_mel)
        raw_evidence = extract_raw_audio_evidence(
            raw_log_mel_10ms,
            audio_duration_seconds=row.duration_seconds,
        )
        raw_flux_signal = np.max(raw_evidence.band_flux, axis=1).astype(np.float32, copy=False)
        raw_view = evaluate_view_stability(
            RAW_FLUX_VIEW_NAME,
            raw_flux_signal,
            frame_rate_hz=100.0,
            duration_seconds=row.duration_seconds,
            config=config,
        )
        timings["raw_flux"] = _elapsed_ms(stage_started, pipeline.clock)

        stage = "beatnet"
        stage_started = pipeline.clock()
        beatnet_times = _load_beatnet_times(row, pipeline)
        beatnet_signal = beat_times_to_impulse_salience(
            beatnet_times,
            duration_seconds=row.duration_seconds,
            frame_rate_hz=config.beatnet_impulse_frame_rate_hz,
        )
        beatnet_view = evaluate_view_stability(
            BEATNET_VIEW_NAME,
            beatnet_signal,
            frame_rate_hz=config.beatnet_impulse_frame_rate_hz,
            duration_seconds=row.duration_seconds,
            config=config,
        )
        timings["beatnet"] = _elapsed_ms(stage_started, pipeline.clock)

        stage = "screen"
        stage_started = pipeline.clock()
        consensus = screen_audio_consensus(
            {
                BEATTHIS_VIEW_NAME: beatthis_view,
                RAW_FLUX_VIEW_NAME: raw_view,
                BEATNET_VIEW_NAME: beatnet_view,
            },
            config=config,
        )
        natural_change_consensus = screen_natural_change_consensus(
            {
                BEATTHIS_VIEW_NAME: beatthis_view,
                RAW_FLUX_VIEW_NAME: raw_view,
                BEATNET_VIEW_NAME: beatnet_view,
            },
            config=config,
        )
        timings["screen"] = _elapsed_ms(stage_started, pipeline.clock)
        timings["total"] = _elapsed_ms(started, pipeline.clock)
        return _completed_row_payload(
            row,
            beatthis_prediction=beatthis_prediction,
            raw_evidence_valid_frames=raw_evidence.valid_frame_count,
            beatnet_beat_count=len(tuple(beatnet_times)),
            consensus=consensus,
            natural_change_consensus=natural_change_consensus,
            timings=timings,
        )
    except Exception as exc:  # noqa: BLE001 - one row is the resumable failure unit.
        timings["total"] = _elapsed_ms(started, pipeline.clock)
        return _failure_row_payload(row, stage=stage, timings=timings, error=exc)


def plan_full5050_audio_consensus_run(
    config: Full5050AudioConsensusRunnerConfig,
) -> dict[str, object]:
    rows = load_full5050_locator_rows(
        config.labels_path,
        expected_row_count=config.expected_row_count,
    )
    return {
        "schema": FULL5050_AUDIO_CONSENSUS_PLAN_SCHEMA,
        "action": "plan_only",
        "labels_path": config.labels_path.as_posix(),
        "output_jsonl": config.output_jsonl.as_posix(),
        "beatnet_events_jsonl": None
        if config.beatnet_events_jsonl is None
        else config.beatnet_events_jsonl.as_posix(),
        "expected_rows": config.expected_row_count,
        "total_rows": len(rows),
        "views": list(CONSENSUS_VIEW_NAMES),
        "screen": _screen_config_payload(config.screen),
        "notes": {
            "input_scope": "locator_only",
            "mapper_redlines": "not_read_not_ground_truth",
            "same_waveform_views": "correlated_estimators_not_independent_recordings",
            "beatnet": "model_3_offline_dbn_required_when_run",
            "beatnet_precompute": "full_run_requires_complete_locator_bound_events_jsonl",
            "constant_source_gate": "at_least_256_then_128_train_128_untouched_holdout",
            "natural_change_gate": "at_least_100_then_50_train_50_untouched_holdout",
        },
    }


def select_constant_sources_from_results(
    result_jsonl: str | Path,
    *,
    output_jsonl: str | Path = DEFAULT_FULL5050_AUDIO_CONSENSUS_SELECTED_JSONL,
    train_count: int = DEFAULT_TRAIN_SOURCE_COUNT,
    holdout_count: int = DEFAULT_HOLDOUT_SOURCE_COUNT,
) -> dict[str, object]:
    _require_positive_int(train_count, "train_count")
    _require_positive_int(holdout_count, "holdout_count")
    required_count = train_count + holdout_count
    selected_by_source: dict[str, dict[str, object]] = {}
    rows_seen = 0
    accepted_rows = 0
    with Path(result_jsonl).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"result line {line_number} must be a JSON object")
            rows_seen += 1
            consensus = payload.get("consensus")
            if not isinstance(consensus, Mapping):
                continue
            if consensus.get("accepted_constant_source") is not True:
                continue
            accepted_rows += 1
            source_key = payload.get("resolved_audio_path")
            if not isinstance(source_key, str) or not source_key:
                raise ValueError(f"result line {line_number} resolved_audio_path must be non-empty")
            candidate = _selection_payload(payload)
            existing = selected_by_source.get(source_key)
            if existing is None or _selection_sort_key(candidate) < _selection_sort_key(existing):
                selected_by_source[source_key] = candidate

    ranked = sorted(selected_by_source.values(), key=_selection_sort_key)
    if len(ranked) < required_count:
        return {
            "schema": FULL5050_AUDIO_CONSENSUS_SELECTION_SCHEMA,
            "status": "kill",
            "reason": "fewer_than_required_constant_sources",
            "required_sources": required_count,
            "train_sources": train_count,
            "untouched_holdout_sources": holdout_count,
            "selected_sources": len(ranked),
            "accepted_rows": accepted_rows,
            "rows_seen": rows_seen,
            "output_jsonl": Path(output_jsonl).as_posix(),
        }

    selected = [
        {
            **row,
            "split": "train" if index < train_count else "untouched_holdout",
            "split_index": index if index < train_count else index - train_count,
        }
        for index, row in enumerate(ranked[:required_count])
    ]
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in selected),
        encoding="utf-8",
    )
    return {
        "schema": FULL5050_AUDIO_CONSENSUS_SELECTION_SCHEMA,
        "status": "selected",
        "required_sources": required_count,
        "train_sources": train_count,
        "untouched_holdout_sources": holdout_count,
        "selected_sources": len(selected),
        "available_sources": len(ranked),
        "accepted_rows": accepted_rows,
        "rows_seen": rows_seen,
        "output_jsonl": output_path.as_posix(),
    }


def select_natural_change_sources_from_results(
    result_jsonl: str | Path,
    *,
    output_jsonl: str | Path = DEFAULT_FULL5050_AUDIO_CHANGE_SELECTED_JSONL,
    train_count: int = DEFAULT_CHANGE_TRAIN_SOURCE_COUNT,
    holdout_count: int = DEFAULT_CHANGE_HOLDOUT_SOURCE_COUNT,
) -> dict[str, object]:
    _require_positive_int(train_count, "train_count")
    _require_positive_int(holdout_count, "holdout_count")
    required_count = train_count + holdout_count
    selected_by_source: dict[str, dict[str, object]] = {}
    rows_seen = 0
    accepted_rows = 0
    with Path(result_jsonl).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"result line {line_number} must be a JSON object")
            rows_seen += 1
            consensus = payload.get("natural_change_consensus")
            if not isinstance(consensus, Mapping):
                continue
            if consensus.get("accepted_natural_change_source") is not True:
                continue
            accepted_rows += 1
            source_key = payload.get("resolved_audio_path")
            if not isinstance(source_key, str) or not source_key:
                raise ValueError(f"result line {line_number} resolved_audio_path must be non-empty")
            candidate = _change_selection_payload(payload)
            existing = selected_by_source.get(source_key)
            if existing is None or _selection_sort_key(candidate) < _selection_sort_key(existing):
                selected_by_source[source_key] = candidate

    ranked = sorted(selected_by_source.values(), key=_selection_sort_key)
    if len(ranked) < required_count:
        return {
            "schema": FULL5050_AUDIO_CHANGE_SELECTION_SCHEMA,
            "status": "report_only",
            "reason": "fewer_than_required_natural_change_sources",
            "required_sources": required_count,
            "train_sources": train_count,
            "untouched_holdout_sources": holdout_count,
            "selected_sources": len(ranked),
            "accepted_rows": accepted_rows,
            "rows_seen": rows_seen,
            "output_jsonl": Path(output_jsonl).as_posix(),
        }

    selected = [
        {
            **row,
            "split": "train" if index < train_count else "untouched_holdout",
            "split_index": index if index < train_count else index - train_count,
        }
        for index, row in enumerate(ranked[:required_count])
    ]
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in selected),
        encoding="utf-8",
    )
    return {
        "schema": FULL5050_AUDIO_CHANGE_SELECTION_SCHEMA,
        "status": "selected",
        "required_sources": required_count,
        "train_sources": train_count,
        "untouched_holdout_sources": holdout_count,
        "selected_sources": len(selected),
        "available_sources": len(ranked),
        "accepted_rows": accepted_rows,
        "rows_seen": rows_seen,
        "output_jsonl": output_path.as_posix(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    screen = AudioConsensusScreenConfig(
        min_duration_seconds=args.min_duration_seconds,
        max_duration_seconds=args.max_duration_seconds,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
        min_complete_windows=args.min_complete_windows,
        tempo_min_bpm=args.tempo_min_bpm,
        tempo_max_bpm=args.tempo_max_bpm,
        octave_family_min_bpm=args.octave_family_min_bpm,
        octave_family_max_bpm=args.octave_family_max_bpm,
        family_tolerance_octaves=args.family_tolerance_octaves,
        global_min_confidence=args.global_min_confidence,
        window_min_confidence=args.window_min_confidence,
        min_confident_window_ratio=args.min_confident_window_ratio,
        persistent_run_windows=args.persistent_run_windows,
        beatnet_impulse_frame_rate_hz=args.beatnet_impulse_frame_rate_hz,
    )
    runner_config = Full5050AudioConsensusRunnerConfig(
        labels_path=args.labels_path,
        output_jsonl=args.output_jsonl,
        beatnet_events_jsonl=args.beatnet_events_jsonl,
        summary_json=args.summary_json,
        expected_row_count=args.expected_row_count,
        retry_failed=args.retry_failed,
        screen=screen,
    )

    if args.select and args.select_change:
        raise ValueError("--select and --select-change are mutually exclusive")

    if args.select:
        summary = select_constant_sources_from_results(
            args.output_jsonl,
            output_jsonl=args.selected_output_jsonl,
            train_count=args.train_source_count,
            holdout_count=args.holdout_source_count,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["status"] == "selected" else 2

    if args.select_change:
        summary = select_natural_change_sources_from_results(
            args.output_jsonl,
            output_jsonl=args.change_selected_output_jsonl,
            train_count=args.change_train_source_count,
            holdout_count=args.change_holdout_source_count,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if not args.run:
        print(json.dumps(plan_full5050_audio_consensus_run(runner_config), indent=2, sort_keys=True))
        return 0

    pipeline = Full5050AudioConsensusPipeline(
        beatnet_extractor=BeatNetModel3DBNExtractor(device=args.beatnet_device),
    )
    summary = run_full5050_audio_consensus(runner_config, pipeline=pipeline)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _load_beatthis_prediction(
    row: Full5050LocatorRow,
    pipeline: Full5050AudioConsensusPipeline,
) -> FrameTimingPrediction:
    if row.beatthis_audio_cache_key is None:
        raise MissingBeatThisCacheError(f"BeatThis cache key is missing for {row.row_id}")
    prediction = pipeline.beatthis_cache_loader(
        row.beatthis_audio_cache_key,
        pipeline.beatthis_cache_config,
    )
    if prediction is None:
        raise MissingBeatThisCacheError(f"BeatThis cache is missing for {row.row_id}")
    return prediction


def _precomputed_beatnet_event_from_mapping(
    payload: Mapping[str, object],
    *,
    line_number: int,
) -> PrecomputedBeatNetEventRow:
    schema = payload.get("schema")
    if schema != FULL5050_BEATNET_EVENTS_SCHEMA:
        raise ValueError(f"BeatNet events line {line_number} has unexpected schema {schema!r}")
    row_index = payload.get("row_index")
    if isinstance(row_index, bool) or not isinstance(row_index, int):
        raise ValueError(f"BeatNet events line {line_number} row_index must be an integer")
    row_id = payload.get("row_id")
    if not isinstance(row_id, str) or not row_id:
        raise ValueError(f"BeatNet events line {line_number} row_id must be non-empty")
    path_raw = payload.get("resolved_audio_path")
    if not isinstance(path_raw, str) or not path_raw:
        raise ValueError(
            f"BeatNet events line {line_number} resolved_audio_path must be non-empty",
        )
    duration_raw = payload.get("duration_seconds")
    if isinstance(duration_raw, bool) or not isinstance(duration_raw, (int, float)):
        raise ValueError(f"BeatNet events line {line_number} duration_seconds must be numeric")
    duration_seconds = float(duration_raw)
    if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError(
            f"BeatNet events line {line_number} duration_seconds must be positive and finite",
        )
    status = payload.get("status")
    if status not in {"completed", "failed", "skipped_duration"}:
        raise ValueError(
            f"BeatNet events line {line_number} status must be completed, failed, or skipped_duration",
        )
    beat_times_raw = payload.get("beat_times_seconds")
    if beat_times_raw is None:
        beat_times_raw = ()
    if not isinstance(beat_times_raw, Sequence) or isinstance(beat_times_raw, (str, bytes)):
        raise ValueError(
            f"BeatNet events line {line_number} beat_times_seconds must be a sequence",
        )
    beat_times = tuple(float(value) for value in beat_times_raw)
    if not all(math.isfinite(value) for value in beat_times):
        raise ValueError(f"BeatNet events line {line_number} beat times must be finite")
    if any(value < 0.0 for value in beat_times):
        raise ValueError(f"BeatNet events line {line_number} beat times must be non-negative")
    if any(value > duration_seconds for value in beat_times):
        raise ValueError(
            f"BeatNet events line {line_number} beat times must be inside duration_seconds",
        )
    if any(right <= left for left, right in zip(beat_times, beat_times[1:])):
        raise ValueError(
            f"BeatNet events line {line_number} beat times must be strictly increasing",
        )
    error = payload.get("error")
    error_type = None
    error_message = None
    if status == "completed":
        if not beat_times:
            raise ValueError(f"BeatNet events line {line_number} completed row has empty beat times")
        if error is not None:
            raise ValueError(f"BeatNet events line {line_number} completed row cannot include error")
    else:
        if beat_times:
            raise ValueError(
                f"BeatNet events line {line_number} non-completed row cannot include beat times",
            )
        if not isinstance(error, Mapping):
            raise ValueError(
                f"BeatNet events line {line_number} non-completed row requires error object",
            )
        raw_type = error.get("type")
        raw_message = error.get("message")
        if not isinstance(raw_type, str) or not raw_type:
            raise ValueError(f"BeatNet events line {line_number} error.type must be non-empty")
        if not isinstance(raw_message, str):
            raise ValueError(f"BeatNet events line {line_number} error.message must be a string")
        error_type = raw_type
        error_message = raw_message
    return PrecomputedBeatNetEventRow(
        row_index=row_index,
        row_id=row_id,
        resolved_audio_path=Path(path_raw),
        duration_seconds=duration_seconds,
        status=str(status),
        beat_times_seconds=beat_times,
        error_type=error_type,
        error_message=error_message,
    )


def _validate_precomputed_event_matches_row(
    event: PrecomputedBeatNetEventRow,
    row: Full5050LocatorRow,
) -> None:
    if event.row_index != row.row_index:
        raise ValueError(f"BeatNet event row_index mismatch for {row.row_id}")
    if event.row_id != row.row_id:
        raise ValueError(f"BeatNet event row_id mismatch for {row.row_id}")
    if event.resolved_audio_path != row.resolved_audio_path:
        raise ValueError(f"BeatNet event audio path mismatch for {row.row_id}")
    if abs(event.duration_seconds - row.duration_seconds) > 1e-6:
        raise ValueError(f"BeatNet event duration mismatch for {row.row_id}")


def _load_packed_mel(
    row: Full5050LocatorRow,
    pipeline: Full5050AudioConsensusPipeline,
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


def _load_beatnet_times(
    row: Full5050LocatorRow,
    pipeline: Full5050AudioConsensusPipeline,
) -> tuple[float, ...]:
    extractor = pipeline.beatnet_extractor
    if isinstance(extractor, PrecomputedBeatNetEventStore):
        return extractor.beat_times_for_row(row)
    times = tuple(float(value) for value in extractor.extract_beat_times(row.resolved_audio_path))
    if not times:
        raise MissingBeatNetEventsError(f"BeatNet event stream is empty for {row.row_id}")
    if not all(math.isfinite(value) for value in times):
        raise ValueError("BeatNet event times must be finite")
    if any(value < 0.0 for value in times):
        raise ValueError("BeatNet event times must be non-negative")
    if any(value > row.duration_seconds for value in times):
        raise ValueError("BeatNet event times must be inside row duration")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("BeatNet event times must be strictly increasing")
    return times


def _completed_row_payload(
    row: Full5050LocatorRow,
    *,
    beatthis_prediction: FrameTimingPrediction,
    raw_evidence_valid_frames: int,
    beatnet_beat_count: int,
    consensus: AudioConsensusResult,
    natural_change_consensus: NaturalChangeConsensusResult,
    timings: Mapping[str, float | None],
) -> dict[str, object]:
    return {
        **_base_row_payload(row),
        "status": "completed",
        "reason": None,
        "beatthis": {
            "source": "cache",
            "provider": beatthis_prediction.provider,
            "checkpoint_path": beatthis_prediction.checkpoint_path,
            "frame_count": beatthis_prediction.frame_count,
            "frame_rate_hz": float(beatthis_prediction.frame_rate_hz),
        },
        "raw_flux": {
            "source": "packed_log_mel_20ms_inverse_unpack",
            "valid_frame_count": int(raw_evidence_valid_frames),
            "frame_rate_hz": 100.0,
        },
        "beatnet": {
            "source": "BeatNet model=3 mode=offline inference_model=DBN",
            "beat_count": int(beatnet_beat_count),
        },
        "timings_ms": _timings_payload(timings),
        "views": {
            name: _view_payload(view) for name, view in consensus.views.items()
        },
        "consensus": _consensus_payload(consensus),
        "natural_change_consensus": _natural_change_payload(natural_change_consensus),
    }


def _duration_skip_payload(
    row: Full5050LocatorRow,
    *,
    timings: Mapping[str, float | None],
    reason: str,
) -> dict[str, object]:
    return {
        **_base_row_payload(row),
        "status": "skipped_duration",
        "reason": reason,
        "timings_ms": _timings_payload(timings),
        "views": {},
        "consensus": {
            "accepted_constant_source": False,
            "reason": reason,
            "global_octave_family_bpm": None,
            "max_cross_view_family_distance_octaves": None,
            "confidence_score": 0.0,
            "confidence_floor": 0.0,
        },
        "natural_change_consensus": {
            "accepted_natural_change_source": False,
            "reason": reason,
            "boundary_start_seconds": None,
            "boundary_end_seconds": None,
            "signed_ratio_octaves": None,
            "family_ratio": None,
            "max_cross_view_ratio_distance_octaves": None,
            "confidence_score": 0.0,
            "confidence_floor": 0.0,
        },
    }


def _failure_row_payload(
    row: Full5050LocatorRow,
    *,
    stage: str,
    timings: Mapping[str, float | None],
    error: Exception,
) -> dict[str, object]:
    return {
        **_base_row_payload(row),
        "status": "failed",
        "reason": f"{stage}_failed",
        "timings_ms": _timings_payload(timings),
        "views": {},
        "consensus": {
            "accepted_constant_source": False,
            "reason": f"{stage}_failed",
            "global_octave_family_bpm": None,
            "max_cross_view_family_distance_octaves": None,
            "confidence_score": 0.0,
            "confidence_floor": 0.0,
        },
        "natural_change_consensus": {
            "accepted_natural_change_source": False,
            "reason": f"{stage}_failed",
            "boundary_start_seconds": None,
            "boundary_end_seconds": None,
            "signed_ratio_octaves": None,
            "family_ratio": None,
            "max_cross_view_ratio_distance_octaves": None,
            "confidence_score": 0.0,
            "confidence_floor": 0.0,
        },
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def _base_row_payload(row: Full5050LocatorRow) -> dict[str, object]:
    return {
        "schema": FULL5050_AUDIO_CONSENSUS_RESULT_SCHEMA,
        "row_index": row.row_index,
        "row_id": row.row_id,
        "resolved_audio_path": row.resolved_audio_path.as_posix(),
        "duration_seconds": row.duration_seconds,
        "audio_length_ms": row.audio_length_ms,
        "input_status": row.input_status,
    }


def _consensus_payload(consensus: AudioConsensusResult) -> dict[str, object]:
    return {
        "accepted_constant_source": consensus.accepted_constant_source,
        "reason": consensus.reason,
        "global_octave_family_bpm": consensus.global_octave_family_bpm,
        "max_cross_view_family_distance_octaves": (
            consensus.max_cross_view_family_distance_octaves
        ),
        "confidence_score": consensus.confidence_score,
        "confidence_floor": consensus.confidence_floor,
    }


def _natural_change_payload(consensus: NaturalChangeConsensusResult) -> dict[str, object]:
    return {
        "accepted_natural_change_source": consensus.accepted_natural_change_source,
        "reason": consensus.reason,
        "boundary_start_seconds": consensus.boundary_start_seconds,
        "boundary_end_seconds": consensus.boundary_end_seconds,
        "signed_ratio_octaves": consensus.signed_ratio_octaves,
        "family_ratio": consensus.family_ratio,
        "max_cross_view_ratio_distance_octaves": (
            consensus.max_cross_view_ratio_distance_octaves
        ),
        "confidence_score": consensus.confidence_score,
        "confidence_floor": consensus.confidence_floor,
    }


def _view_payload(view: ViewStabilityResult) -> dict[str, object]:
    return {
        "status": view.status,
        "stable": view.stable,
        "reason": view.reason,
        "global": _estimate_payload(view.global_estimate),
        "window_count": view.window_count,
        "confident_window_count": view.confident_window_count,
        "confident_window_ratio": view.confident_window_ratio,
        "max_persistent_mismatch_run": view.max_persistent_mismatch_run,
        "max_low_confidence_run": view.max_low_confidence_run,
        "median_window_direct_bpm": view.median_window_direct_bpm,
        "median_window_family_bpm": view.median_window_family_bpm,
        "persistent_change": None
        if view.persistent_change is None
        else _persistent_change_payload(view.persistent_change),
        "windows": [_estimate_payload(estimate) for estimate in view.window_estimates],
    }


def _persistent_change_payload(change: PersistentChangeEvidence) -> dict[str, object]:
    return {
        "boundary_start_seconds": change.boundary_start_seconds,
        "boundary_end_seconds": change.boundary_end_seconds,
        "left_direct_bpm": change.left_direct_bpm,
        "right_direct_bpm": change.right_direct_bpm,
        "left_octave_family_bpm": change.left_octave_family_bpm,
        "right_octave_family_bpm": change.right_octave_family_bpm,
        "signed_ratio_octaves": change.signed_ratio_octaves,
        "family_ratio": float(2.0**change.signed_ratio_octaves),
        "confidence": change.confidence,
        "left_window_count": change.left_window_count,
        "right_window_count": change.right_window_count,
    }


def _estimate_payload(estimate: TempoFamilyEstimate) -> dict[str, object]:
    return {
        "status": estimate.status,
        "direct_bpm": estimate.direct_bpm,
        "octave_family_bpm": estimate.octave_family_bpm,
        "confidence": estimate.confidence,
        "lag_seconds": estimate.lag_seconds,
        "score": estimate.score,
    }


def _selection_payload(payload: Mapping[str, object]) -> dict[str, object]:
    consensus = payload["consensus"]
    if not isinstance(consensus, Mapping):
        raise ValueError("consensus must be an object")
    row_index = payload.get("row_index")
    if isinstance(row_index, bool) or not isinstance(row_index, int):
        raise ValueError("row_index must be an integer")
    duration = payload.get("duration_seconds")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise ValueError("duration_seconds must be numeric")
    source = payload.get("resolved_audio_path")
    if not isinstance(source, str) or not source:
        raise ValueError("resolved_audio_path must be a non-empty string")
    confidence = consensus.get("confidence_score")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("consensus.confidence_score must be numeric")
    family = consensus.get("global_octave_family_bpm")
    if family is not None and (isinstance(family, bool) or not isinstance(family, (int, float))):
        raise ValueError("consensus.global_octave_family_bpm must be numeric or null")
    return {
        "schema": FULL5050_AUDIO_CONSENSUS_SELECTION_SCHEMA,
        "row_index": row_index,
        "source_key": source,
        "resolved_audio_path": source,
        "duration_seconds": float(duration),
        "global_octave_family_bpm": None if family is None else float(family),
        "confidence_score": float(confidence),
        "confidence_floor": float(consensus.get("confidence_floor", 0.0)),
        "max_cross_view_family_distance_octaves": consensus.get(
            "max_cross_view_family_distance_octaves",
        ),
    }


def _change_selection_payload(payload: Mapping[str, object]) -> dict[str, object]:
    consensus = payload["natural_change_consensus"]
    if not isinstance(consensus, Mapping):
        raise ValueError("natural_change_consensus must be an object")
    row_index = payload.get("row_index")
    if isinstance(row_index, bool) or not isinstance(row_index, int):
        raise ValueError("row_index must be an integer")
    duration = payload.get("duration_seconds")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise ValueError("duration_seconds must be numeric")
    source = payload.get("resolved_audio_path")
    if not isinstance(source, str) or not source:
        raise ValueError("resolved_audio_path must be a non-empty string")
    confidence = consensus.get("confidence_score")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("natural_change_consensus.confidence_score must be numeric")
    return {
        "schema": FULL5050_AUDIO_CHANGE_SELECTION_SCHEMA,
        "row_index": row_index,
        "source_key": source,
        "resolved_audio_path": source,
        "duration_seconds": float(duration),
        "boundary_start_seconds": consensus.get("boundary_start_seconds"),
        "boundary_end_seconds": consensus.get("boundary_end_seconds"),
        "signed_ratio_octaves": consensus.get("signed_ratio_octaves"),
        "family_ratio": consensus.get("family_ratio"),
        "confidence_score": float(confidence),
        "confidence_floor": float(consensus.get("confidence_floor", 0.0)),
        "max_cross_view_ratio_distance_octaves": consensus.get(
            "max_cross_view_ratio_distance_octaves",
        ),
    }


def _selection_sort_key(payload: Mapping[str, object]) -> tuple[float, int]:
    confidence = payload.get("confidence_score")
    row_index = payload.get("row_index")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("selection confidence_score must be numeric")
    if isinstance(row_index, bool) or not isinstance(row_index, int):
        raise ValueError("selection row_index must be an integer")
    return (-float(confidence), int(row_index))


def _view_confidence_floor(view: ViewStabilityResult) -> float:
    window_confidences = [
        estimate.confidence
        for estimate in view.window_estimates
        if estimate.status == "ok" and estimate.confidence >= 0.0
    ]
    if not window_confidences:
        return 0.0
    return float(min(view.global_estimate.confidence, float(np.median(window_confidences))))


def _natural_change_reject(
    views: Mapping[str, ViewStabilityResult],
    *,
    reason: str,
    confidence_floor: float = 0.0,
    max_ratio_distance: float | None = None,
) -> NaturalChangeConsensusResult:
    return NaturalChangeConsensusResult(
        accepted_natural_change_source=False,
        reason=reason,
        boundary_start_seconds=None,
        boundary_end_seconds=None,
        signed_ratio_octaves=None,
        family_ratio=None,
        max_cross_view_ratio_distance_octaves=max_ratio_distance,
        confidence_score=0.0,
        confidence_floor=float(confidence_floor),
        views=dict(views),
    )


def _view_result(
    view_name: str,
    *,
    status: str,
    stable: bool,
    reason: str | None,
    global_estimate: TempoFamilyEstimate | None = None,
    window_estimates: tuple[TempoFamilyEstimate, ...] = (),
) -> ViewStabilityResult:
    return ViewStabilityResult(
        view_name=view_name,
        status=status,
        stable=stable,
        reason=reason,
        global_estimate=global_estimate or TempoFamilyEstimate(None, None, 0.0, None, None, status),
        window_estimates=window_estimates,
        window_count=len(window_estimates),
        confident_window_count=0,
        confident_window_ratio=0.0,
        max_persistent_mismatch_run=0,
        max_low_confidence_run=0,
        median_window_direct_bpm=None,
        median_window_family_bpm=None,
    )


def _detect_persistent_change(
    window_estimates: tuple[TempoFamilyEstimate, ...],
    *,
    windows: Sequence[tuple[int, int]],
    frame_rate_hz: float,
    config: AudioConsensusScreenConfig,
) -> PersistentChangeEvidence | None:
    run = config.persistent_run_windows
    if len(window_estimates) < 2 * run or len(windows) != len(window_estimates):
        return None

    best: PersistentChangeEvidence | None = None
    best_key: tuple[float, float] | None = None
    for split in range(run, len(window_estimates) - run + 1):
        left_block = window_estimates[split - run : split]
        right_block = window_estimates[split : split + run]
        if not _all_confident(left_block, config=config):
            continue
        if not _all_confident(right_block, config=config):
            continue

        left_direct = float(
            np.median(
                np.asarray(
                    [estimate.direct_bpm for estimate in left_block],
                    dtype=np.float64,
                ),
            ),
        )
        right_direct = float(
            np.median(
                np.asarray(
                    [estimate.direct_bpm for estimate in right_block],
                    dtype=np.float64,
                ),
            ),
        )
        left_spread = max(
            direct_tempo_distance_octaves(left_direct, float(estimate.direct_bpm))
            for estimate in left_block
            if estimate.direct_bpm is not None
        )
        right_spread = max(
            direct_tempo_distance_octaves(right_direct, float(estimate.direct_bpm))
            for estimate in right_block
            if estimate.direct_bpm is not None
        )
        if left_spread > config.family_tolerance_octaves:
            continue
        if right_spread > config.family_tolerance_octaves:
            continue

        signed_ratio = math.log2(right_direct / left_direct)
        magnitude = abs(signed_ratio)
        if magnitude <= config.family_tolerance_octaves:
            continue
        confidence = min(
            min(estimate.confidence for estimate in left_block),
            min(estimate.confidence for estimate in right_block),
        )
        boundary_start = windows[split - 1][0] / frame_rate_hz
        boundary_end = windows[split][1] / frame_rate_hz
        change = PersistentChangeEvidence(
            boundary_start_seconds=float(boundary_start),
            boundary_end_seconds=float(boundary_end),
            left_direct_bpm=left_direct,
            right_direct_bpm=right_direct,
            left_octave_family_bpm=octave_family_bpm(left_direct, config=config),
            right_octave_family_bpm=octave_family_bpm(right_direct, config=config),
            signed_ratio_octaves=float(signed_ratio),
            confidence=float(confidence),
            left_window_count=run,
            right_window_count=run,
        )
        key = (float(confidence), float(magnitude))
        if best_key is None or key > best_key:
            best = change
            best_key = key
    return best


def _all_confident(
    estimates: Sequence[TempoFamilyEstimate],
    *,
    config: AudioConsensusScreenConfig,
) -> bool:
    return all(
        estimate.status == "ok"
        and estimate.direct_bpm is not None
        and estimate.confidence >= config.window_min_confidence
        for estimate in estimates
    )


def _window_slices(
    *,
    frame_count: int,
    frame_rate_hz: float,
    window_seconds: float,
    hop_seconds: float,
) -> tuple[tuple[int, int], ...]:
    window_frames = max(1, int(round(window_seconds * frame_rate_hz)))
    hop_frames = max(1, int(round(hop_seconds * frame_rate_hz)))
    if frame_count < window_frames:
        return ()
    starts = list(range(0, frame_count - window_frames + 1, hop_frames))
    last_start = frame_count - window_frames
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return tuple((start, start + window_frames) for start in starts)


def _require_salience_vector(value: object, *, name: str) -> NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return np.ascontiguousarray(array, dtype=np.float32)


def _read_existing_result_row_indexes(
    output_jsonl: Path,
    *,
    retry_failed: bool,
) -> set[int]:
    if not output_jsonl.exists():
        return set()
    completed: set[int] = set()
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
            if status == "failed" and retry_failed:
                continue
            completed.add(row_index)
    return completed


def _summarize_audio_consensus_result_file(
    output_jsonl: Path,
    *,
    expected_row_count: int,
) -> dict[str, int]:
    _require_positive_int(expected_row_count, "expected_row_count")
    final_by_row: dict[int, Mapping[str, object]] = {}
    with output_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"result line {line_number} must be a JSON object")
            row_index = payload.get("row_index")
            if isinstance(row_index, bool) or not isinstance(row_index, int):
                raise ValueError(f"result line {line_number} row_index must be an integer")
            if row_index < 0 or row_index >= expected_row_count:
                raise ValueError(
                    f"result line {line_number} row_index {row_index} is outside "
                    f"[0, {expected_row_count})",
                )
            status = payload.get("status")
            if status not in _FINAL_ROW_STATUSES:
                continue
            final_by_row[row_index] = payload
    missing = sorted(set(range(expected_row_count)) - set(final_by_row))
    if missing:
        preview = ",".join(str(index) for index in missing[:10])
        raise ValueError(f"result JSONL is incomplete; missing row indexes: {preview}")
    counts = {
        "completed": 0,
        "failed": 0,
        "skipped_duration": 0,
        "accepted_constant": 0,
        "accepted_natural_change": 0,
    }
    for payload in final_by_row.values():
        status = payload.get("status")
        if status == "completed":
            counts["completed"] += 1
        elif status == "failed":
            counts["failed"] += 1
        elif status == "skipped_duration":
            counts["skipped_duration"] += 1
        consensus = payload.get("consensus")
        if isinstance(consensus, Mapping) and consensus.get("accepted_constant_source") is True:
            counts["accepted_constant"] += 1
        natural_change = payload.get("natural_change_consensus")
        if (
            isinstance(natural_change, Mapping)
            and natural_change.get("accepted_natural_change_source") is True
        ):
            counts["accepted_natural_change"] += 1
    return counts


def _timings_payload(timings: Mapping[str, float | None]) -> dict[str, float | None]:
    return {stage: timings.get(stage) for stage in _TIMING_STAGE_NAMES}


def _screen_config_payload(config: AudioConsensusScreenConfig) -> dict[str, object]:
    return {
        "min_duration_seconds": config.min_duration_seconds,
        "max_duration_seconds": config.max_duration_seconds,
        "window_seconds": config.window_seconds,
        "hop_seconds": config.hop_seconds,
        "min_complete_windows": config.min_complete_windows,
        "tempo_min_bpm": config.tempo_min_bpm,
        "tempo_max_bpm": config.tempo_max_bpm,
        "octave_family_min_bpm": config.octave_family_min_bpm,
        "octave_family_max_bpm": config.octave_family_max_bpm,
        "family_tolerance_octaves": config.family_tolerance_octaves,
        "global_min_confidence": config.global_min_confidence,
        "window_min_confidence": config.window_min_confidence,
        "min_confident_window_ratio": config.min_confident_window_ratio,
        "persistent_run_windows": config.persistent_run_windows,
        "beatnet_impulse_frame_rate_hz": config.beatnet_impulse_frame_rate_hz,
    }


def _elapsed_ms(started: float, clock: Clock) -> float:
    return max(0.0, 1000.0 * (float(clock()) - float(started)))


def _max_true_run(flags: Sequence[bool]) -> int:
    best = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or run the full5050 audio-only multi-estimator constant-source screen. "
            "Default is plan-only. Pass --run to extract BeatThis/raw/BeatNet views; "
            "pass --select after a completed run to write the constant train/holdout cohort; "
            "pass --select-change to write the natural-change train/holdout cohort."
        ),
    )
    parser.add_argument("--run", action="store_true", help="execute rows and append JSONL results")
    parser.add_argument("--select", action="store_true", help="select constant sources from output JSONL")
    parser.add_argument(
        "--select-change",
        action="store_true",
        help="select natural persistent-change sources from output JSONL",
    )
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_FULL5050_LABELS_PATH)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_FULL5050_AUDIO_CONSENSUS_OUTPUT_JSONL)
    parser.add_argument("--beatnet-events-jsonl", type=Path, default=DEFAULT_FULL5050_BEATNET_EVENTS_JSONL)
    parser.add_argument(
        "--selected-output-jsonl",
        type=Path,
        default=DEFAULT_FULL5050_AUDIO_CONSENSUS_SELECTED_JSONL,
    )
    parser.add_argument(
        "--change-selected-output-jsonl",
        type=Path,
        default=DEFAULT_FULL5050_AUDIO_CHANGE_SELECTED_JSONL,
    )
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--expected-row-count", type=int, default=DEFAULT_EXPECTED_FULL5050_ROW_COUNT)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--train-source-count", type=int, default=DEFAULT_TRAIN_SOURCE_COUNT)
    parser.add_argument("--holdout-source-count", type=int, default=DEFAULT_HOLDOUT_SOURCE_COUNT)
    parser.add_argument("--change-train-source-count", type=int, default=DEFAULT_CHANGE_TRAIN_SOURCE_COUNT)
    parser.add_argument(
        "--change-holdout-source-count",
        type=int,
        default=DEFAULT_CHANGE_HOLDOUT_SOURCE_COUNT,
    )
    parser.add_argument("--beatnet-device", default="cpu")
    parser.add_argument("--min-duration-seconds", type=float, default=90.0)
    parser.add_argument("--max-duration-seconds", type=float, default=600.0)
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--hop-seconds", type=float, default=15.0)
    parser.add_argument("--min-complete-windows", type=int, default=5)
    parser.add_argument("--tempo-min-bpm", type=float, default=45.0)
    parser.add_argument("--tempo-max-bpm", type=float, default=240.0)
    parser.add_argument("--octave-family-min-bpm", type=float, default=90.0)
    parser.add_argument("--octave-family-max-bpm", type=float, default=180.0)
    parser.add_argument("--family-tolerance-octaves", type=float, default=0.06)
    parser.add_argument("--global-min-confidence", type=float, default=0.07)
    parser.add_argument("--window-min-confidence", type=float, default=0.04)
    parser.add_argument("--min-confident-window-ratio", type=float, default=0.80)
    parser.add_argument("--persistent-run-windows", type=int, default=3)
    parser.add_argument("--beatnet-impulse-frame-rate-hz", type=float, default=100.0)
    return parser.parse_args(argv)


__all__ = [
    "BEATNET_VIEW_NAME",
    "BEATTHIS_VIEW_NAME",
    "CONSENSUS_VIEW_NAMES",
    "DEFAULT_FULL5050_AUDIO_CONSENSUS_OUTPUT_JSONL",
    "DEFAULT_FULL5050_AUDIO_CONSENSUS_SELECTED_JSONL",
    "DEFAULT_FULL5050_AUDIO_CHANGE_SELECTED_JSONL",
    "DEFAULT_FULL5050_BEATNET_EVENTS_JSONL",
    "DEFAULT_CHANGE_HOLDOUT_SOURCE_COUNT",
    "DEFAULT_CHANGE_TRAIN_SOURCE_COUNT",
    "DEFAULT_HOLDOUT_SOURCE_COUNT",
    "DEFAULT_TRAIN_SOURCE_COUNT",
    "FULL5050_AUDIO_CHANGE_SELECTION_SCHEMA",
    "FULL5050_BEATNET_EVENTS_SCHEMA",
    "FULL5050_AUDIO_CONSENSUS_PLAN_SCHEMA",
    "FULL5050_AUDIO_CONSENSUS_RESULT_SCHEMA",
    "FULL5050_AUDIO_CONSENSUS_SELECTION_SCHEMA",
    "FULL5050_AUDIO_CONSENSUS_SUMMARY_SCHEMA",
    "RAW_FLUX_VIEW_NAME",
    "AudioConsensusResult",
    "AudioConsensusScreenConfig",
    "BeatNetBeatTimeExtractor",
    "BeatNetModel3DBNExtractor",
    "ConstantSourceSelectionError",
    "Full5050AudioConsensusError",
    "Full5050AudioConsensusPipeline",
    "Full5050AudioConsensusRunnerConfig",
    "MissingBeatThisCacheError",
    "MissingBeatNetEventsError",
    "NaturalChangeConsensusResult",
    "PersistentChangeEvidence",
    "PrecomputedBeatNetEventRow",
    "PrecomputedBeatNetEventStore",
    "TempoFamilyEstimate",
    "ViewStabilityResult",
    "beat_times_to_impulse_salience",
    "direct_tempo_distance_octaves",
    "evaluate_view_stability",
    "estimate_tempo_family_from_salience",
    "main",
    "load_precomputed_beatnet_event_store",
    "octave_family_bpm",
    "octave_family_distance_octaves",
    "plan_full5050_audio_consensus_run",
    "run_full5050_audio_consensus",
    "run_full5050_audio_consensus_row",
    "screen_audio_consensus",
    "screen_natural_change_consensus",
    "select_constant_sources_from_results",
    "select_natural_change_sources_from_results",
]


if __name__ == "__main__":
    raise SystemExit(main())
