from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _frame_step_ms(frame_times_ms: NDArray[np.float64]) -> float:
    if frame_times_ms.shape[0] < 2:
        return 20.0
    return float(np.median(np.diff(frame_times_ms)))


def _frame_rate_hz_from_times(frame_times_ms: NDArray[np.float64]) -> float:
    frame_step_ms = _frame_step_ms(frame_times_ms)
    if frame_step_ms <= 0.0:
        return 50.0
    return 1000.0 / frame_step_ms
