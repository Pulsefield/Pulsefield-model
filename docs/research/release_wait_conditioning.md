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
