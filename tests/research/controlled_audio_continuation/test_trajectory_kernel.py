"""Physical invariants and an exact finite-policy gradient, without chart labels."""
from itertools import product

import numpy as np
import pytest

from ensomi_model.research.controlled_audio_continuation.trajectory_kernel import (
    CELL_FIELDS, distribution_objective, interval_cells, kernel_matrix, squared_distance,
)
from ensomi_model.research.oracle_time_continuation.storage import ROW_DTYPE


def rows(entries):
    result = np.zeros(len(entries), dtype=ROW_DTYPE)
    result['time'] = [r[0] for r in entries]
    result['actions'] = [r[1] for r in entries]
    return result


def taps(masks):
    return rows([(i*100, tuple(int(bool(mask & 1 << k)) for k in range(4)))
                 for i, mask in enumerate(masks)])


def test_blocks_preserve_group_relations_and_physical_symmetries():
    alternate = taps([3, 12]*12)
    repeat = taps([3, 3, 12, 12]*6)
    a = interval_cells(alternate, 0, 2300).values
    b = interval_cells(repeat, 0, 2300).values
    assert squared_distance(kernel_matrix([a, a])) == pytest.approx(0, abs=1e-12)
    assert squared_distance(kernel_matrix([a, a[:, ::-1]])) == pytest.approx(0, abs=1e-12)
    assert squared_distance(kernel_matrix([a, b])) > .001
    shifted = alternate.copy()
    shifted['time'] += 90000
    np.testing.assert_allclose(a, interval_cells(shifted, 90000, 92300).values)
    assert np.isin(alternate['actions'], (1, 2)).sum() == np.isin(repeat['actions'], (1, 2)).sum()


def test_ln_state_and_interior_release_do_not_read_future_endpoints():
    trace = rows([(0, (2, 0, 0, 0)), (100, (0, 1, 0, 0)),
                  (150, (3, 0, 0, 0)), (200, (0, 0, 1, 0)), (300, (1, 0, 0, 0))])
    cells = interval_cells(trace, 0, 200)
    assert cells.times_ms.tolist() == [0, 100]
    f = {name: i for i, name in enumerate(CELL_FIELDS)}
    assert cells.values[0, 0, f['occupied_after']] == 1
    assert cells.values[0, 0, f['interior_release']] == 0
    assert cells.values[1, 0, f['occupied_before']] == 1
    assert cells.values[1, 0, f['ln_age']] == .5
    assert cells.values[1, 0, f['interior_release']] == 1
    assert cells.values[1, 0, f['release_fraction']] == .5
    altered = trace.copy()
    altered['actions'][-1] = (0, 0, 0, 1)
    np.testing.assert_array_equal(cells.values, interval_cells(altered, 0, 200).values)
    # The same LN crossing the right boundary remains occupied, without a tail
    # invented at that boundary. A true release at the right H belongs outside.
    continued = rows([(0, (2, 0, 0, 0)), (100, (0, 1, 0, 0)),
                      (200, (3, 0, 1, 0)), (300, (1, 0, 0, 0))])
    c = interval_cells(continued, 0, 200).values
    assert c[1, 0, f['interior_release']] == 0
    assert squared_distance(kernel_matrix([cells.values, c])) > .001


def test_u_statistic_gradient_matches_expected_distribution_distance():
    # Enumerate independent Bernoulli-policy trajectories. A self-product or a
    # baseline that depends on the current draw changes this exact derivative.
    embeddings = np.array([[0., 1.], [1., .2], [.3, .8]])
    gram = embeddings@embeddings.T
    p = .37
    for count in (3, 4):
        expected_value, derivative = 0., 0.
        for draws in product((0, 1), repeat=count):
            probability = np.prod([p if i else 1-p for i in draws])
            selected = [*draws, 2]
            value, coefficients = distribution_objective(gram[np.ix_(selected, selected)])
            expected_value += probability*value
            derivative += probability*sum(c*(1/p if i else -1/(1-p))
                                           for c, i in zip(coefficients, draws))
        mean = (1-p)*embeddings[0]+p*embeddings[1]
        expected = np.square(mean-embeddings[2]).sum()
        exact_gradient = 2*(mean-embeddings[2])@(embeddings[1]-embeddings[0])
        assert expected_value == pytest.approx(expected)
        assert derivative == pytest.approx(exact_gradient)
