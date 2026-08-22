from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.floating]


@dataclass(frozen=True)
class ChoiceMetrics:
    nll: float
    top1: float
    mean_reciprocal_rank: float
    pairwise_concordance: float
    mean_case_probability: float


@dataclass(frozen=True)
class BootstrapInterval:
    lower: float
    upper: float
    confidence: float
    samples: int


def choice_metrics(log_supports: FloatArray, *, case_index: int = 0) -> ChoiceMetrics:
    values = np.asarray(log_supports, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("log_supports must have shape [cases, alternatives>=2]")
    if not 0 <= case_index < values.shape[1]:
        raise ValueError(f"case_index outside alternative range: {case_index}")
    if not np.all(np.isfinite(values)):
        raise ValueError("log_supports must be finite")

    log_normalizer = _logsumexp(values, axis=1)
    case_log_probability = values[:, case_index] - log_normalizer
    case_scores = values[:, case_index : case_index + 1]
    greater = np.sum(values > case_scores, axis=1)
    equal_controls = np.sum(values == case_scores, axis=1) - 1
    top1_credit = np.where(greater == 0, 1.0 / (equal_controls + 1), 0.0)
    reciprocal_rank = np.asarray(
        [
            np.mean(1.0 / np.arange(greater_count + 1, greater_count + tied_count + 1))
            for greater_count, tied_count in zip(greater, equal_controls + 1)
        ],
        dtype=np.float64,
    )
    controls = np.delete(values, case_index, axis=1)
    concordance = np.mean(
        (case_scores > controls).astype(np.float64) + 0.5 * (case_scores == controls),
        axis=1,
    )
    return ChoiceMetrics(
        nll=float(-np.mean(case_log_probability)),
        top1=float(np.mean(top1_credit)),
        mean_reciprocal_rank=float(np.mean(reciprocal_rank)),
        pairwise_concordance=float(np.mean(concordance)),
        mean_case_probability=float(np.mean(np.exp(case_log_probability))),
    )


def paired_cluster_bootstrap(
    per_audio_deltas: FloatArray,
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 1337,
) -> BootstrapInterval:
    deltas = _validate_audio_deltas(per_audio_deltas)
    if samples <= 0:
        raise ValueError(f"samples must be positive, got {samples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be within (0,1), got {confidence}")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, deltas.size, size=(samples, deltas.size))
    means = np.mean(deltas[indexes], axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, (tail, 1.0 - tail))
    return BootstrapInterval(
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
        samples=samples,
    )


def paired_sign_flip_pvalue(
    per_audio_deltas: FloatArray,
    *,
    samples: int = 10_000,
    seed: int = 1337,
) -> float:
    """One-sided paired sign-flip p-value for a positive mean effect."""

    deltas = _validate_audio_deltas(per_audio_deltas)
    if samples <= 0:
        raise ValueError(f"samples must be positive, got {samples}")
    observed = float(np.mean(deltas))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray((-1.0, 1.0)), size=(samples, deltas.size))
    null_means = np.mean(signs * deltas, axis=1)
    return float((1 + np.count_nonzero(null_means >= observed)) / (samples + 1))


def exact_group_shapley(
    coalition_values: Mapping[frozenset[str], float],
    *,
    groups: Sequence[str] = ("N", "T", "P"),
) -> dict[str, float]:
    """Compute exact Shapley allocations for a complete coalition table."""

    ordered_groups = tuple(groups)
    if not ordered_groups or len(set(ordered_groups)) != len(ordered_groups):
        raise ValueError("groups must be non-empty and unique")
    expected = {
        frozenset(group for index, group in enumerate(ordered_groups) if mask & (1 << index))
        for mask in range(1 << len(ordered_groups))
    }
    missing = expected - coalition_values.keys()
    extra = coalition_values.keys() - expected
    if missing or extra:
        raise ValueError(f"coalition table mismatch: missing={_format_sets(missing)}, extra={_format_sets(extra)}")

    count = len(ordered_groups)
    allocations: dict[str, float] = {}
    for group in ordered_groups:
        value = 0.0
        for coalition in expected:
            if group in coalition:
                continue
            size = len(coalition)
            weight = math.factorial(size) * math.factorial(count - size - 1) / math.factorial(count)
            value += weight * (
                float(coalition_values[coalition | {group}]) - float(coalition_values[coalition])
            )
        allocations[group] = value
    return allocations


def _logsumexp(values: npt.NDArray[np.float64], *, axis: int) -> npt.NDArray[np.float64]:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(np.sum(np.exp(values - maximum), axis=axis))


def _validate_audio_deltas(values: FloatArray) -> npt.NDArray[np.float64]:
    deltas = np.asarray(values, dtype=np.float64)
    if deltas.ndim != 1 or deltas.size == 0 or not np.all(np.isfinite(deltas)):
        raise ValueError("per_audio_deltas must be a non-empty finite one-dimensional array")
    return deltas


def _format_sets(values: set[frozenset[str]]) -> list[list[str]]:
    return sorted((sorted(value) for value in values), key=lambda value: (len(value), value))
