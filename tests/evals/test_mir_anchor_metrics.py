from __future__ import annotations

import numpy as np
import pytest

from pulsefield_model.evals.mir_anchor_metrics import choice_metrics
from pulsefield_model.evals.mir_anchor_metrics import exact_group_shapley
from pulsefield_model.evals.mir_anchor_metrics import paired_cluster_bootstrap
from pulsefield_model.evals.mir_anchor_metrics import paired_sign_flip_pvalue


def test_choice_metrics_use_case_column_zero() -> None:
    metrics = choice_metrics(
        np.asarray(
            [
                [3.0, 1.0, 0.0],
                [0.0, 2.0, 1.0],
            ],
        ),
    )

    assert metrics.top1 == 0.5
    assert metrics.mean_reciprocal_rank == pytest.approx(2.0 / 3.0)
    assert metrics.pairwise_concordance == pytest.approx(0.5)
    assert 0.0 < metrics.mean_case_probability < 1.0
    assert metrics.nll > 0.0


def test_choice_metrics_give_chance_credit_to_ties() -> None:
    metrics = choice_metrics(np.zeros((2, 4)))

    assert metrics.top1 == pytest.approx(0.25)
    assert metrics.mean_reciprocal_rank == pytest.approx((1.0 + 0.5 + 1.0 / 3.0 + 0.25) / 4.0)
    assert metrics.pairwise_concordance == pytest.approx(0.5)
    assert metrics.mean_case_probability == pytest.approx(0.25)


def test_bootstrap_and_sign_flip_are_seeded() -> None:
    deltas = np.asarray([0.1, 0.2, 0.3, 0.4])
    first = paired_cluster_bootstrap(deltas, samples=1000, seed=9)
    second = paired_cluster_bootstrap(deltas, samples=1000, seed=9)

    assert first == second
    assert first.lower > 0.0
    assert paired_sign_flip_pvalue(deltas, samples=1000, seed=4) == paired_sign_flip_pvalue(
        deltas,
        samples=1000,
        seed=4,
    )


def test_exact_shapley_recovers_additive_group_values() -> None:
    contributions = {"N": 1.0, "T": 2.0, "P": 4.0}
    table = {
        frozenset(group for index, group in enumerate(("N", "T", "P")) if mask & (1 << index)): sum(
            contribution
            for group, contribution in contributions.items()
            if group in {name for index, name in enumerate(("N", "T", "P")) if mask & (1 << index)}
        )
        for mask in range(8)
    }

    assert exact_group_shapley(table) == pytest.approx(contributions)


def test_exact_shapley_requires_every_coalition() -> None:
    with pytest.raises(ValueError, match="coalition table mismatch"):
        exact_group_shapley({frozenset(): 0.0})
