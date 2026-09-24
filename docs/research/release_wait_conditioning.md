# First-release conditioning before a required head

A known future H and four occupied lanes require at least one intervening
release. The default law in the [planned prototype](planned_audio_continuation.md)
moves surviving probability to the last available release clock. That preserves
legal completion but can concentrate substantial probability one millisecond
before the head. The optional `condition_full_holds` law conditions the raw
waiting distribution on a feasible release without imposing a minimum gap or
forbidding four-lane holds. Native quality under this law requires evaluation.

## Located event and exact reproduction

The bounded head checkpoint
47d41844afc673788cb640c67a1d5e41b2b9ca75ae679298c10072200925bce6 generates
Airborne Robots at seed 33. At 39901 ms it starts LNs in columns 0 and 3 while
columns 1 and 2 are held. The next planned H is 40211 ms. A release-only row
at 40210 ms closes columns 0, 1 and 2; column 1 receives a TAP at 40211 ms.
Column 3 stays held until 40549 ms.

The full 38200–41900 ms episode, entering hold, actions, articulation and both
time-view pages were inspected with Lens. This is a specific release/re-press
concern, not a rule against short LNs, close events in different columns, or
four-lane LN passages generally.

A sampler observer reproduces the complete saved chart exactly, without
changing parameters or advancing the real RNG. It captures the raw release
logits for 39902–40210 ms and the existing release draw. Before the first
release, no H or new LN can occur, so advancing the fixed committed state
along this no-event branch uses no future actual release action.

## Deadline atom versus conditional event probability

Let the current committed time be $s$, the next H be $h$, and $d=h-1$.
Write the raw release hazard as $r_u$ and its first-event probability as

$$
q(u)=r_u\prod_{v=s+1}^{u-1}(1-r_v).
$$

The deadline-atom implementation replaces $r_d$ with one. Therefore its event mass
at $d$ is the entire preceding survival probability

$$
q_{\mathrm{atom}}(d)=\prod_{v=s+1}^{d-1}(1-r_v).
$$

The alternative keeps the raw event shape and conditions on a release before H:

$$
Z=\sum_{u=s+1}^{d}q(u),\qquad
q_{\mathrm{feasible}}(u)=q(u)/Z.
$$

It retains every feasible native timestamp, including $d$. It neither chooses
an LN endpoint at the head nor specifies which columns the release row closes.
After the first release, at least one lane remains free until H because no new
head can occur in between. The ordinary process resumes with the updated state.

| Fixed-state quantity | Value |
| --- | ---: |
| Raw probability of a release before H | 65.6448% |
| Last-clock mass with the deadline atom | 34.4447% |
| Last-clock mass after feasible conditioning | 0.1364% |
| Reduction of last-clock mass | 252.5 times |
| Original release-to-H gap | 1 ms |
| Conditioned gap at the same uniform quantile | 63 ms |

The shared uniform quantile is 0.920926. Under the alternative law it selects
40148 ms. This is a distribution comparison at one fixed state, not a generated
replacement chart or a claim that subsequent row choices are now good.

## Relationship to frontier2

The row consequence feature sees the earliest possible release at 39902 ms,
giving 309 ms of possible slack before H. That is an exact possibility under
native support; it does not describe the probability of realizing the slack.

At the 39901 ms row, the head mask fixes columns 0/3 and the release mask is
empty. Among the four tap/LN-kind variants, the existing consequence residual
reduces the probability of starting two LNs from 71.04% to 65.45%. The actual
all-held row has 5.85% probability in the full legal-row distribution.
Thus the consequence module is active and distinguishes kinds; it does not
account for the release process's large boundary atom.

This exposes two approximation choices: candidate consequences currently use
earliest availability, while the release scheduler converts residual survival
into a last-clock event. A future consequence model may use predicted release
availability in addition to its structural bound. First, the waiting-law
normalization can be tested without inventing a calibrated gameplay penalty.

## Training and inference requirements

The conditional law is implemented in interval training and native sampling.
Existing checkpoints were fitted under the deadline-atom law; enabling the flag
on those fixed weights is an intervention on the generated distribution, not
evidence of a trained conditional-law endpoint.

For a full-occupancy wait, training must include the same normalizer used at
inference. If an interval ends before the first release, it scores conditional
survival within the feasible wait; a crop boundary cannot become a release
deadline. Partitioned interval likelihood must equal the unsplit law.

The normalizer needs raw release predictions through $d$ under the hypothetical
no-release state. This may extend beyond a source interval's actual first
release or local audio crop. Complete audio and the chosen head plan permit
those hypothetical queries in both modes. Future actual occupancy and source
LN endpoints do not. Local audio must include the required horizon and halo.

The shared probability primitive computes conditional hazard odds as

$$
\frac{r_u^*}{1-r_u^*}
=\frac{\exp(\operatorname{logit}(r_u))}
{1-\prod_{v=u+1}^{d}(1-r_v)},\qquad u<d.
$$

Suffix probabilities are accumulated in log space. The final hazard is one,
while its raw value still influences earlier hazards through normalization.
Tests compare probabilities and gradients to normalized first-event masses,
including very small raw probabilities. The implementation uses `logaddexp`
for the integrated hazard because the tested MPS `softplus` operation rounds
very small positive outputs to zero before the subsequent logarithm.

The head process has no analogous requirement to produce an event during every
rest. This conditioning applies when a release is required by the known head
plan, rather than acting as a general event-rate floor. It does not by itself
guarantee comfortable play: a very short feasible interval or a learned event
shape concentrated near H can still produce difficult release/head relations.

## Frozen-weight native comparison

Source `12d80eb3ae488dbe8c13083975d3e60e702f4ef2` implements the conditional law.
The unchanged bounded checkpoint was generated with both settings on Who,
Death Piano, Prom Queen and Good Luck at seeds 17/19, and Airborne at seed 33.
The default setting reproduced every row of all nine saved baseline outputs.
All conditional outputs completed and independently reparsed; every planned H
timestamp remained identical. Seven complete charts were unchanged.

Across the nine charts, same-column release-to-next-head gaps at most 20 ms
decreased from one to zero; head-to-head gaps at most 20 ms remained zero.
These are diagnostic counts, not legality constraints or complete BAD labels.
Airborne realizes the predicted 40148 ms release and 63 ms gap. Its following
head becomes a 73 ms LN, and later row choices differ. The LN fraction changes
from 14.90% to 13.82%. Good Luck seed 17 moves its first full-occupancy release
from 199984 to 199980 ms and two later tails by one millisecond, preserving
its long right-hand holds and their joint exit.

Lens inspection covered the fixed comparison scopes and both changed
full-occupancy episodes, including entering holds and Good Luck's exit.
Fourteen new pages were viewed, with 36 pages verified byte-identical to the
earlier inspected evidence. The specific one-millisecond re-press is removed;
the later Airborne fixed window becomes all taps instead of containing one LN.
Thus improved release mechanics do not imply identical musical organization
or preserved style in every passage. Sparse piano generation and expressive
coverage remain unresolved. There is no listening or player-verdict claim.

Cached-Mel first-30-row latency was 0.155–0.258 s and eight-second readiness
0.153–0.334 s. The paired run took 30.98 s on one CPU thread. These figures omit
waveform/Mel preprocessing and do not establish simultaneous-load performance.

The training corpus contains 1377 full-occupancy waits before a known H across
108 of its 615 training arrangements. Their first-release-to-H gaps range from
39 to 1872 ms, with median 97 ms. The frozen 1200-update exposure plan encounters
787 such interval segments in 215 updates, including ten in the first 32
updates. They cover 107210 scored milliseconds and require 205480 hypothetical
milliseconds for normalization; the longest queried wait is 7107 ms. This
establishes available supervision and bounded observed query cost, without
turning the observed minimum gap into a constraint.

## Evidence

Source: 88e56832287f492fdca81ea7194b90753526cc96.
Local owner:
artifacts/joint-audio/20260924-head-recovery-v1/release-law-audit.
Freeze SHA-256:
1291fd1e66787885ddbef91c8f9d501f1141520db8bc68016240b384bbe4fe12.
Result SHA-256:
d9141ebcf077625dfd884941fac7ea70548ffbbb70e5390ed9f6e7a5d8e7a53b.
The diagnostic took 4.84 s on one CPU thread and is exploratory. The larger
playability goal, piano activity, expressive coverage and validated release
behavior remain open.

The frozen native comparison and exposure audit are owned by
artifacts/joint-audio/20260924-feasible-release-v1. Comparison result SHA-256:
e992e57c53e848abffc9a6c1b16c20e8ccd55c2047bec95c31fce3be9eaad1c9.
Lens review:
9ad0f8b47d917db39c1d0ab522c0cf2aaec088a1b759d862352eac9f3fd2ba74.
Exposure audit:
8ec4c6ba78dd4e4a49472411fce0f35567496ee135e82f0a7b8d4ef9338304eb.
This is an exploratory comparison of changed inference with frozen weights;
training under the conditional law remains a separate question.
