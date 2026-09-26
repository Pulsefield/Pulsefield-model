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

## Paired R/R1 learning result

A 128-update comparison added this trajectory objective to continued factual
learning. It lengthened some generated LNs, but worsened the reserved style
comparison and native difficulty control. Neither endpoint replaces the selected
core model.

Both arms started from the aligned-1200 candidate of the
[balanced condition study](balanced_condition_alignment.md). Audio and H stayed
frozen; R and R1 trained without architectural changes. Each update used the
same three genuine source examples in both arms: an actor target and two broad
source-exposure examples. Existing difficulty-condition alignment remained on
the broad paired examples. All 128 target identities and 384 positive source
identities matched between arms.

Actor targets came from 64 ranked population scopes, sixteen per whole-star bin
2–3 through 5–6, and 38 prominent human-assessed cells. Actual local difficulties
were in 2–6, with at least eight complete H intervals. Remaining prominent cells
were 17 stream, 5 trill, 8 jack, 5 LN coordination and 3 Tech. Population and human
targets alternated; bins/concepts cycled with seeded selection inside each.
The existing qualification audio exclusions remained in force.

For the trajectory arm, each target had three independent continuations from the
same genuine prefix, followed by up to sixteen seconds of generated burn-in and
the scored target scope. The actual generated remainder of the song supplied
complete endpoints for storage; it did not enter the target kernel or the policy
gradient. Likelihood covered the burn-in and target, including both R timing and
deployed R1 row probabilities. No source next-row label was attached to a changed
generated prefix. The kernel surrogate had weight 20 alongside the common source
objective. All 384 sampled training continuations completed.

The qualification used three seeds per condition: 60 difficulty continuations,
24 reserved style charts, 18 LN charts and twelve native charts. The style metric
below is the independent-rollout U-statistic against each source's empirical
block distribution; lower is closer, not necessarily more playable.

| Reserved style scope | Continued factual learning | Plus trajectory objective |
| --- | ---: | ---: |
| Tech | .03113 | .03962 |
| Jack organization | .03155 | .04348 |
| Stream organization | .01350 | .02838 |
| Trill organization | .12839 | .14988 |
| Mean | .05114 | .06534 |

The required .005 absolute and 15% relative improvement both failed. Mean scoped
difficulty error rose from .80687 to .85922, within its .15 regression allowance,
and the global LN-error regression guard passed. Restored difficulty failed its
.25 allowance: singles/low regressed by .344, Shippaisaku/low by .374 and
Shippaisaku/high by .278. These restored ranges were evaluated separately from
the overridden requests.

Lens rendered 23 structure contexts into 86 pages. Focused review viewed fourteen
matched seed-zero pages and read relevant action tables; it was not a full visual
review of every output or a human playtest. Source human labels stayed source-only.

Miraie showed a concrete partial benefit. Across the three LN guard draws,
median hold duration increased from 107/101.5/99 ms to 183/183/183 ms, matching
the source passage's median. Head counts per H moved from 2.773/2.091/2.364 to
2.045/2.091/2.000, closer to its 2.0. The renders showed fewer wide sets of short
LNs and more sustained holds. However, only 13/42, 9/35 and 16/41 generated tails
fell on H, compared with 28/28 in this particular source passage. Matching a
duration median did not recover its articulation.

Starry Jet also gained sustained holds, with more overlap and some collective
releases, without a clear general improvement in the source's moving LN roles.
Shippaisaku's H-coincident tail counts fell from 68/86/77 to 57/69/67; its holds
spanned more later H events. Longer holds alone were therefore insufficient.
These comparisons do not define an at-H requirement for all LN styles.

The trill episode had fewer exact repeated groups, but lost a stable dominant
exchange. The two most common complementary pairs covered .750/.917/.708 of
its H rows in the continued arm and .542/.542/.542 in the trajectory arm. Both
versions mixed or repeated groups; fewer repetitions alone did not establish
better trill organization. Jack seed zero retained recurrent chords in both
arms. Tech and stream retained supplied timing but did not establish a broader
quality improvement. Some guards' global difficulty requests exceeded their
local source readouts, so exact local density matching was not required.

Native H streams matched exactly between arms in all six paired settings.
Nevertheless, all three static difficulty-3 charts became harder:

| Audio | Continued | Trajectory |
| --- | ---: | ---: |
| Zenithfall | 4.136 | 4.474 |
| Hysteric | 3.862 | 4.396 |
| Take | 4.295 | 4.886 |

The difficulty-4.5/LN-.6 override on 64000–96000 ms had closer LN fractions in
the trajectory arm: .621/.627/.631 versus .674/.683/.741. This local improvement
coexisted with worse difficulty outside it and cannot be summarized as overall
control success. Cached-Mel first-30-row times were .224–.438 s, maximum measured
eight-second publication service .391 s, and full generation 3.57–8.66 s. Model
loading, waveform decoding and Mel extraction are excluded.

This objective can move physical continuation behavior, but this fit did not
improve the overall system. Sparse conditional coverage, strong noisy trajectory
gradients and the representation's short horizon remain possible contributors.
An independent [readout audit](row_condition_interactions.md) also identifies an
algebraic limit on main-path condition/history interactions. The fit does not
prove that this limit alone caused its failures; it supplies a reason to test a
small conditional interaction before scaling the same objective.

## Execution identities

The kernel implementation used source
`3cf169cda06c09b3487f12f6dcdacf81b4fc74da` and artifact owner
`20260926-trajectory-kernel-r1-v1`. The initial checkpoint SHA-256 was
`67ef81fbb9fb2ecfc5ca19d8fc6767040b846e1ba8c41d4b8d61067abc08ee26`.
Terminal continued and trajectory checkpoints were respectively
`97a16453bf489c1a4e656a9ebd676028af896b42add5242f45b08dadeba70b8a` and
`c36eb6d94a8d98cdb7484db04870ecdc456c3fbb264e4a8bc6adb32b434d3e8d`.

Preparation and actor draws used seed 261510; generated actor seeds were
261511 + 1000 × update + draw index. The same frozen canonical Mel, ranked corpus
and human cohort as the balanced study supplied inputs. Fits used PyTorch
2.11/MPS with CPU generation, one CPU thread per process on the 24 GiB Mac M5.
AdamW rates were 3e-5 for inherited weights and 3e-4 for composition, control and
preview weights, with .0001 weight decay and norm clipping at 1.

The continued fit took 151.824 s and the trajectory fit 1241.324 s. Their sampled
footprints peaked at 5.157 and 9.093 GiB, and MPS driver allocations at 2.134 and
3.454 GiB; these overlapping ledgers must not be added. Frozen audio/H tensors
were unchanged. The maximum sampled/rescored row log-probability difference was
$2.51\times10^{-5}$ and total R log-probability difference $8.44\times10^{-5}$.
R1 received nonzero trajectory gradients in all 128 updates and R in 113; some
tap-only continuations offered no stochastic R decision. Four-update integration
fits were discarded before the main comparison.
