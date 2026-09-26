"""Short physical continuation comparisons, independent of neural history.

An H-to-H cell retains actions, committed lane state and releases inside that
interval. It never resolves a hold beyond the next H. Kernels compare local
block distributions, not aligned next-row labels or canonical player costs.
"""
from dataclasses import dataclass

import numpy as np


CELL_FIELDS = (
    'idle', 'tap', 'ln_start', 'ln_release', 'occupied_before', 'occupied_after',
    'ln_age', 'attack_age', 'attack_known', 'release_age', 'release_known',
    'interior_release', 'release_fraction', 'interval_length',
)
BLOCK_LENGTHS = (1, 2, 4, 8)
BANDWIDTHS = (.125, .25, .5)


@dataclass(frozen=True)
class TrajectoryCells:
    times_ms: np.ndarray
    values: np.ndarray


def interval_cells(rows, start_ms, end_ms):
    """Represent complete H intervals in [start_ms,end_ms) from admitted rows.

    Input is the lossless structured row array used by SourceChart. Its complete
    committed prefix supplies state. A cell's right H may equal end_ms, but its
    action and any later events are excluded. No fabricated closure is needed
    for an LN continuing beyond a cell or the scored scope.
    """
    if not np.isfinite([start_ms, end_ms]).all() or end_ms <= start_ms:
        raise ValueError('Trajectory scope requires increasing finite boundaries')
    times = np.asarray(rows['time'], dtype=np.float64)
    actions = np.asarray(rows['actions'])
    heads = np.isin(actions, (1, 2))
    positions = np.flatnonzero(heads.any(-1))
    left, right = positions[:-1], positions[1:]
    inside = (times[left] >= start_ms) & (times[right] <= end_ms)
    left, right = left[inside], right[inside]
    if not len(left):
        return TrajectoryCells(np.empty(0), np.empty((0, 4, len(CELL_FIELDS))))
    indices = np.arange(len(rows))[:, None]
    previous = np.maximum(left-1, 0)

    def prior(mask):
        latest = np.maximum.accumulate(np.where(mask, indices, -1), axis=0)
        return np.where(left[:, None] > 0, latest[previous], -1)

    attack, release, origin = prior(heads), prior(actions == 3), prior(actions == 2)
    occupied = origin > release
    now, gap = times[left, None], (times[right]-times[left])[:, None]

    def age(past):
        elapsed = now-times[np.maximum(past, 0)]
        return np.where(past >= 0, elapsed/(elapsed+gap), 0.)

    present = actions[left]
    after = np.where(present == 3, False, np.where(present == 2, True, occupied))
    following_release = np.minimum.accumulate(
        np.where(actions == 3, indices, len(rows))[::-1], axis=0)[::-1]
    tail_index = following_release[left+1]
    interior = after & (tail_index < right[:, None])
    tail = times[np.minimum(tail_index, len(rows)-1)]
    fraction = np.where(interior, (tail-now)/gap, 0.)
    scalar = np.stack((occupied, after, age(np.where(occupied, origin, -1)),
        age(attack), attack >= 0, age(release), release >= 0,
        interior, fraction, np.broadcast_to(gap/(gap+250.), present.shape)), -1)
    values = np.concatenate((np.eye(4)[present], scalar), -1)
    return TrajectoryCells(times[left].copy(), values)


def _blocks(cells, length):
    value = np.asarray(cells, dtype=np.float64)
    return np.stack([value[i:i+length] for i in range(len(value)-length+1)])


def _rbf_mean(x, y, bandwidths):
    x, y = x.reshape(len(x), -1), y.reshape(len(y), -1)
    distance = (np.square(x).sum(-1)[:, None]+np.square(y).sum(-1)[None]
                - 2*x@y.T).clip(min=0)/x.shape[-1]
    return float(np.mean([np.exp(-distance/(2*width**2)).mean() for width in bandwidths]))


def kernel_matrix(trajectories, *, lengths=BLOCK_LENGTHS, bandwidths=BANDWIDTHS):
    """Return a PSD Gram matrix of empirical, reflection-invariant embeddings.

    Each item is [H intervals,4,CELL_FIELDS]. Average equally across supported
    block lengths and bandwidths. All trajectories use the same supported scales;
    an empty trajectory is rejected. The reflection acts on an entire block,
    preserving within-block column relationships.
    """
    if (not trajectories or any(len(v) == 0 for v in trajectories) or
            any(np.asarray(v).shape[1:] != (4, len(CELL_FIELDS)) for v in trajectories)):
        raise ValueError('Kernels require nonempty four-column trajectory cells')
    if not bandwidths or any(not np.isfinite(w) or w <= 0 for w in bandwidths):
        raise ValueError('Kernel bandwidths must be finite and positive')
    scales = [n for n in lengths if 0 < n <= min(map(len, trajectories))]
    if not scales:
        raise ValueError('No block length fits every supplied trajectory')
    matrix = np.zeros((len(trajectories), len(trajectories)), dtype=np.float64)
    for length in scales:
        blocks = [_blocks(v, length) for v in trajectories]
        for i, x in enumerate(blocks):
            for j in range(i, len(blocks)):
                y = blocks[j]
                value = .5*(_rbf_mean(x, y, bandwidths)+_rbf_mean(x, y[:, :, ::-1], bandwidths))
                matrix[i, j] += value/len(scales)
                if i != j:
                    matrix[j, i] += value/len(scales)
    return matrix


def squared_distance(matrix, first=0, second=1):
    """Squared distance between two empirical trajectory embeddings."""
    return float(matrix[first, first]+matrix[second, second]-2*matrix[first, second])


def distribution_objective(matrix):
    """Unbiased rollout U-statistic and detached score-function coefficients.

    The last Gram item is the fixed empirical reference; preceding items are
    independent generated trajectories under one condition and initial state.
    Return (estimate, coefficients). The estimate may be negative. For policy
    learning use sum(coefficients[i] * log_probability(trajectory_i)); coefficients
    are data, not differentiable functions of the policy. Each variance-reduction
    baseline uses only the other trajectories. At least three draws are needed.

    Excluding self-products targets the expected generated embedding rather than
    also penalizing variation between otherwise valid generated charts.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    count = len(matrix)-1
    if count < 3 or matrix.shape != (count+1, count+1) or not np.isfinite(matrix).all():
        raise ValueError('Distribution matching requires three draws and one reference')
    generated, reference = matrix[:count, :count], matrix[:count, count]
    estimate = ((generated.sum()-np.trace(generated))/(count*(count-1))
                - 2*reference.mean()+matrix[count, count])
    coefficients = np.zeros(count)
    for i in range(count):
        other = np.arange(count) != i
        cross = generated[np.ix_(other, other)]
        baseline = ((cross.sum()-np.trace(cross))/((count-1)*(count-2))
                    - reference[other].mean())
        coefficients[i] = 2/count*(generated[i, other].mean()-reference[i]-baseline)
    return float(estimate), coefficients
