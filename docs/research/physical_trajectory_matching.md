# Comparing physical continuations with short interval distributions

The [generated-history diagnosis](balanced_condition_alignment.md) found that
improved factual condition ranking did not stabilize R1's own continuations.
The research-only
[trajectory kernel](../../src/ensomi_model/research/controlled_audio_continuation/trajectory_kernel.py)
provides a compact training-signal candidate for comparing those continuations
with genuine chart structure. It is neither a semantic style classifier nor the
canonical player-response cost.

## Physical representation

One cell spans consecutive H times. For each of four columns it records the
action at the first H, occupancy before and after that action, committed LN age,
decaying recency of the last attack/release, and any release
strictly inside the interval. An interior release has a presence flag and its
fractional native-time position. With no new heads between consecutive H events,
a column can release at most once there.

An ongoing LN stays ongoing at the right boundary; its future endpoint is not
invented. A release at the right H belongs to the next cell. Only cells fully
inside the requested scope are compared. Prior committed actions may supply
state, but actions outside the right boundary do not enter the representation.

Active LN age uses `elapsed / (elapsed + H_gap)`. Inactive attack/release recency
uses `H_gap / (elapsed + H_gap)`, zero with no predecessor. Thus ancient and absent
inactive events converge instead of preserving a global "has ever released LN"
bit that can overwhelm a local pure-tap comparison. The interval length uses
`H_gap / (H_gap + 250 ms)`. These bounded coordinates preserve native timing;
they are not beat quantization or physiological limits. Consecutive blocks of
1, 2, 4 and 8 cells retain progressively longer action and occupancy relations.
The representation imposes no vocabulary of trill, jack or Tech templates.

## Kernel comparison and independent rollouts

For each block length, a Gaussian kernel compares mean squared coordinate
distance at bandwidths .125, .25 and .5. Average equally across those bandwidths
and supported lengths. Averaging the kernel over reflection of the entire
four-column block preserves left/right symmetry while retaining its internal
relationships. Averaging over blocks gives a kernel between trajectories.

This adapts the distribution-comparison primitive of
[maximum mean discrepancy](https://www.jmlr.org/papers/v13/gretton12a.html).
[Generative Moment Matching Networks](https://proceedings.mlr.press/v37/li15.html)
use MMD for generative learning; the proposed choreography use instead requires
score-function gradients for discrete, stateful R/R1 trajectories. No generator
fit is established by implementing the comparison.

Let $K_{ij}$ be the kernel between independent generated trajectories $i,j$,
$K_{iE}$ between trajectory $i$ and the fixed empirical reference, and $K_{EE}$
the reference self-kernel. For $m$ draws, estimate the squared distance of the
expected generated embedding from the empirical reference by

$$
\widehat J = \frac{1}{m(m-1)}\sum_{i\ne j}K_{ij}
             -\frac{2}{m}\sum_i K_{iE}+K_{EE}.
$$

The estimate may be negative. It excludes a generated trajectory's self-product:
including that term would also penalize variation between generated charts.
It does not require every generated chart to reproduce one source arrangement.
The finite empirical reference and chosen features still limit the distribution
being matched; no population or universal quality guarantee follows.

For a policy-gradient surrogate, the part depending on draw $i$ has coefficient

$$
c_i = \frac{2}{m}\left(\frac{1}{m-1}\sum_{j\ne i}K_{ij}-K_{iE}-b_i\right),
$$

where $b_i$ uses only other draws: their mean off-diagonal generated kernel minus
their mean reference kernel. At least three draws are required. Treat these
coefficients as detached observations when multiplying each trajectory's log
probability. Sharing the full joint estimate as an ordinary leave-one-out reward
without accounting for cross-draw dependencies can give the wrong gradient.

The [focused tests](../../tests/research/controlled_audio_continuation/test_trajectory_kernel.py)
check exact gradients by enumerating a finite policy, physical LN censoring,
reflection/time-shift invariance, and sensitivity to group repetition at identical
H times and head counts. These establish the calculation, not chart quality.

A learning use must condition comparisons on music, timing and applicable scoped
controls, preserve each trajectory's own replay, and retain broad genuine-source
learning. Kernel improvement requires independent difficulty/control and Lens
checks. Changing the representation or bandwidth changes what is being matched.
