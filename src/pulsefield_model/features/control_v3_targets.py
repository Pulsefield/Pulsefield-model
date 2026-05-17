from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pulsefield_model.features.control import (
    mania_hit_objects_to_control_hits,
    red_timing_points_to_beat_length_fn,
)
from pulsefield_model.features.control_v3 import (
    CONFIDENCE_FEATURE_NAMES,
    MODEL_FEATURE_NAMES,
    VALUE_FEATURE_NAMES,
    FeatureConfigV3,
    extract_control_features,
)

from pulsefield_model.osu_core.hitobjects import parse_mania_hit_objects
from pulsefield_model.osu_core.timing import require_red_timing_points


TARGET_WINDOW_SECONDS = 2.0
SOURCE_GRID_STEP_SECONDS = 0.1
SOURCE_GRID_TOLERANCE_SECONDS = 1e-4
TARGET_FRAME_STEP_SECONDS = 0.02
TARGET_FRAME_CENTER_OFFSET_SECONDS = 0.01
TARGET_FRAME_COUNT = 100
TARGET_DIM = len(MODEL_FEATURE_NAMES)
target_dim = TARGET_DIM

TIME_COLUMN = "time_s"
CONTROL_CONFIDENCE_FEATURE_NAME = "control_confidence"
LN_CHANGE_N_EFF_FEATURE_NAME = "ln_change_n_eff"
CONTROL_V3_METADATA_COLUMNS = [
    "filtered_index",
    "source_index",
    "beatmap_id",
    "beatmap_set_id",
    "difficulty",
    TIME_COLUMN,
]
DEFAULT_TIMESERIES_PATH = Path(
    "artifacts/features/control_v3_timeseries_4k_no_timing_anomalies_2to6_dense_local_bpm_norm_unique_le3.parquet"
)


@dataclass(frozen=True)
class ControlV3TargetWindow:
    target: NDArray[np.float32]
    confidence: NDArray[np.float32] | None = None
    metadata: dict[str, Any] | None = None


def compute_control_v3_full_map_features(
    beatmap_path: str | Path,
    *,
    cfg: FeatureConfigV3 | None = None,
) -> pd.DataFrame:
    beatmap_path = Path(beatmap_path)
    hitobjects = parse_mania_hit_objects(beatmap_path, expected_key_count=4)
    timing_points = require_red_timing_points(beatmap_path)
    hits = mania_hit_objects_to_control_hits(hitobjects)
    beat_length_at = red_timing_points_to_beat_length_fn(timing_points)
    duration_s = max((hit.end if hit.end is not None else hit.start for hit in hits), default=0.0)
    out = extract_control_features(
        hits,
        beat_length_at=beat_length_at,
        cfg=cfg or FeatureConfigV3(),
        start_time=0.0,
        end_time=duration_s,
        return_debug=True,
    )

    time_s = np.asarray(out["time"], dtype=np.float32)
    ln_change_n_eff = np.asarray(out["debug"][LN_CHANGE_N_EFF_FEATURE_NAME], dtype=np.float32)
    if ln_change_n_eff.shape != time_s.shape:
        raise ValueError(f"{LN_CHANGE_N_EFF_FEATURE_NAME} must match the control_v3 time grid")
    if not np.all(np.isfinite(ln_change_n_eff)):
        raise ValueError(f"{LN_CHANGE_N_EFF_FEATURE_NAME} contains non-finite values")

    frame = pd.DataFrame({TIME_COLUMN: time_s})
    for name in MODEL_FEATURE_NAMES:
        frame[name] = np.asarray(out["features"][name], dtype=np.float32)
    frame[LN_CHANGE_N_EFF_FEATURE_NAME] = ln_change_n_eff
    frame.attrs["beatmap_path"] = beatmap_path.as_posix()
    validate_control_v3_timeseries(frame)
    return frame


def load_control_v3_timeseries_rows(
    timeseries_path: str | Path = DEFAULT_TIMESERIES_PATH,
    *,
    beatmap_id: int | None = None,
    filtered_index: int | None = None,
    source_index: int | None = None,
    include_ln_change_n_eff: bool = False,
) -> pd.DataFrame | None:
    selector = _one_map_selector(
        beatmap_id=beatmap_id,
        filtered_index=filtered_index,
        source_index=source_index,
    )
    path = Path(timeseries_path)
    if not path.exists():
        return None

    column, value = selector
    frame = pd.read_parquet(
        path,
        columns=_timeseries_columns(include_ln_change_n_eff=include_ln_change_n_eff),
        filters=[(column, "==", value)],
    )
    if frame.empty:
        return None

    frame = frame.sort_values(TIME_COLUMN, kind="mergesort").reset_index(drop=True)
    validate_control_v3_timeseries(frame)
    return frame


def slice_control_v3_target_window(
    rows: pd.DataFrame,
    window_start_s: float,
    *,
    return_confidence: bool = False,
    return_metadata: bool = False,
) -> NDArray[np.float32] | ControlV3TargetWindow:
    validate_control_v3_timeseries(rows)
    frame_times = control_v3_target_frame_times(window_start_s)
    source_times = rows[TIME_COLUMN].to_numpy(dtype=np.float64, copy=False)
    source_features = rows[list(MODEL_FEATURE_NAMES)].to_numpy(dtype=np.float32, copy=False)

    target = np.empty((TARGET_FRAME_COUNT, TARGET_DIM), dtype=np.float32)
    for column_index in range(TARGET_DIM):
        target[:, column_index] = np.interp(
            frame_times,
            source_times,
            source_features[:, column_index],
            left=0.0,
            right=0.0,
        ).astype(np.float32)

    if not np.all(np.isfinite(target)):
        raise ValueError("control_v3 target window contains non-finite values")
    if not (return_confidence or return_metadata):
        return target

    confidence = None
    if return_confidence:
        confidence_index = MODEL_FEATURE_NAMES.index(CONTROL_CONFIDENCE_FEATURE_NAME)
        confidence = target[:, confidence_index].astype(np.float32, copy=True)

    metadata = None
    if return_metadata:
        metadata = {
            "window_start_s": float(window_start_s),
            "window_end_s": float(window_start_s + TARGET_WINDOW_SECONDS),
            "frame_times_s": frame_times.astype(np.float32),
            "feature_names": tuple(MODEL_FEATURE_NAMES),
            "value_feature_names": tuple(VALUE_FEATURE_NAMES),
            "confidence_feature_names": tuple(CONFIDENCE_FEATURE_NAMES),
        }

    return ControlV3TargetWindow(target=target, confidence=confidence, metadata=metadata)


def slice_ln_change_n_eff_target_window(rows: pd.DataFrame, window_start_s: float) -> NDArray[np.float32]:
    validate_control_v3_timeseries(rows)
    if LN_CHANGE_N_EFF_FEATURE_NAME not in rows.columns:
        raise ValueError(f"control_v3 timeseries is missing column: {LN_CHANGE_N_EFF_FEATURE_NAME}")

    frame_times = control_v3_target_frame_times(window_start_s)
    source_times = rows[TIME_COLUMN].to_numpy(dtype=np.float64, copy=False)
    source_values = rows[LN_CHANGE_N_EFF_FEATURE_NAME].to_numpy(dtype=np.float32, copy=False)
    if source_values.ndim != 1 or source_values.shape[0] != source_times.shape[0]:
        raise ValueError(f"{LN_CHANGE_N_EFF_FEATURE_NAME} must be a one-dimensional timeseries")
    if not np.all(np.isfinite(source_values)):
        raise ValueError(f"{LN_CHANGE_N_EFF_FEATURE_NAME} contains non-finite values")

    return np.interp(
        frame_times,
        source_times,
        source_values,
        left=0.0,
        right=0.0,
    ).astype(np.float32)


def control_v3_target_frame_times(window_start_s: float) -> NDArray[np.float64]:
    if not np.isfinite(window_start_s):
        raise ValueError(f"window_start_s must be finite, got {window_start_s!r}")
    scaled = window_start_s / TARGET_FRAME_STEP_SECONDS
    if abs(scaled - round(scaled)) > 1e-6:
        raise ValueError(f"window_start_s must align to the 20ms frame grid: {window_start_s!r}")

    frame_indexes = np.arange(TARGET_FRAME_COUNT, dtype=np.float64)
    return window_start_s + TARGET_FRAME_CENTER_OFFSET_SECONDS + frame_indexes * TARGET_FRAME_STEP_SECONDS


def validate_control_v3_timeseries(rows: pd.DataFrame) -> None:
    missing_columns = [
        column for column in (TIME_COLUMN, *MODEL_FEATURE_NAMES)
        if column not in rows.columns
    ]
    if missing_columns:
        raise ValueError(f"control_v3 timeseries is missing column(s): {missing_columns}")

    time_s = rows[TIME_COLUMN].to_numpy(dtype=np.float64, copy=False)
    if time_s.ndim != 1:
        raise ValueError("control_v3 time column must be one-dimensional")
    if time_s.size == 0:
        raise ValueError("control_v3 timeseries must contain at least one row")
    if not np.all(np.isfinite(time_s)):
        raise ValueError("control_v3 time column contains non-finite values")
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError("control_v3 time column must be strictly increasing")
    if abs(float(time_s[0])) > SOURCE_GRID_TOLERANCE_SECONDS:
        raise ValueError("control_v3 time column must start at 0.0s")
    if time_s.size > 1:
        deltas = np.diff(time_s)
        if not np.all(np.abs(deltas - SOURCE_GRID_STEP_SECONDS) <= SOURCE_GRID_TOLERANCE_SECONDS):
            raise ValueError("control_v3 time column must use the 0.1s source grid")

    features = rows[list(MODEL_FEATURE_NAMES)].to_numpy(dtype=np.float32, copy=False)
    if features.ndim != 2 or features.shape[1] != TARGET_DIM:
        raise ValueError(f"control_v3 features must have shape [T,{TARGET_DIM}], got {features.shape}")
    if not np.all(np.isfinite(features)):
        raise ValueError("control_v3 features contain non-finite values")
    confidence = rows[list(CONFIDENCE_FEATURE_NAMES)].to_numpy(dtype=np.float32, copy=False)
    if np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError("control_v3 confidence columns must be in [0, 1]")


def _one_map_selector(
    *,
    beatmap_id: int | None,
    filtered_index: int | None,
    source_index: int | None,
) -> tuple[str, int]:
    selectors = [
        ("beatmap_id", beatmap_id),
        ("filtered_index", filtered_index),
        ("source_index", source_index),
    ]
    selected = [(name, value) for name, value in selectors if value is not None]
    if len(selected) != 1:
        raise ValueError("exactly one of beatmap_id, filtered_index, or source_index is required")
    name, value = selected[0]
    return name, _validate_selector_value(name, value)


def _validate_selector_value(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer id, got bool")
    if isinstance(value, (int, np.integer)):
        integer = int(value)
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{name} must be an integer id, got {value!r}")
        integer = int(value)
    else:
        raise TypeError(f"{name} must be an integer id, got {type(value).__name__}")
    if integer < 0:
        raise ValueError(f"{name} must be non-negative, got {integer}")
    return integer


def _timeseries_columns(*, include_ln_change_n_eff: bool = False) -> list[str]:
    columns = [*CONTROL_V3_METADATA_COLUMNS, *MODEL_FEATURE_NAMES]
    if include_ln_change_n_eff:
        columns.append(LN_CHANGE_N_EFF_FEATURE_NAME)
    return list(dict.fromkeys(columns))
