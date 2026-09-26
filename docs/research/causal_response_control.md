# Causal response control in audio-conditioned R1

An explicit response of the generated chart reduced low-difficulty overfill.
High-LN generation and short-scope proportion control remained inaccurate.
This study separates the learned
arrangement preference, a declared response approximation, and the rule that
uses that response to modify sampling.

The experiment uses the [controlled audio continuation model](controlled_audio_continuation.md).
H timing supplies head-bearing timestamps; R1 owns cardinality, TAP/LN choices,
release identities and columns. Full audio conditions both timing and R1.
The response policies below are local research prototypes, not default runtime
behavior or definitions of the [V3 gameplay frontier](../formulation/gameplay-state.md).

## Why another likelihood fit was insufficient

At fixed ranked source H times, a mostly single-note 2.999-star chart became
4.852 stars with the 2,500-update model. R1 added 498 heads to the same 783 onsets.
A matched 1,000-update comparison tested ordinary continuation against a small
audio/timing/control count prior with bounded history modulation. Both arms
used identical 2,000 accepted source intervals from 1,399 TRAIN charts; audio
and H/release timing weights stayed unchanged.

Across three seeds, ordinary continuation produced 4.635–4.828 stars and the
prior produced 4.333–4.482. The latter improvement was insufficient. On the
first seed, the prior alone expected about 1.50 heads per H on both genuine
source and generated histories. Its history correction lowered that to 1.34
on source history and raised it to 1.64 on generated history. The subsequent
consequence comparison changed those means only slightly.

The source-conditional distribution and its generated-history behavior both
needed attention. A bound on one logit residual did not constrain the resulting
chart's demand. The detailed matched evidence and checkpoint identities are in
the [R1 source-timing analysis](controlled_audio_continuation.md#r1-errors-with-reference-onset-times).

## A deliberately limited response state

The prototype stores four press traces $u_j$ and an overall trace $v$, initially
zero. Over $\Delta$ elapsed seconds,

$$
u_j(t+\Delta)=0.125^\Delta u_j(t),
\qquad v(t+\Delta)=0.30^\Delta v(t).
$$

A TAP or LN head in column $j$ adds two to $u_j$ and one to $v$. Release-only
rows add nothing. For a candidate complete row $a$, the response is

$$
R(H;a)=0.18\left(v^+ + \max_j u_j^+\right).
$$

The decay constants and attack increments come from the repository's mania
strain calculation. Zero initialization and taking the maximum over all columns
are explicit differences. This is an attack-only response: it excludes occupied
hold duration, release execution and LN coordination. Its decay during an open
hold is not a statement that the player is resting. It is neither official
local stars nor a sufficient representation of canonical gameplay demand.

The state reads committed actions and elapsed time. It has no audio input or
future endpoint input, survives control changes, and can be replayed exactly.
Native fork/control-update tests, an independent sum over source presses,
split-time advancement and mirrored candidate scores checked that implementation.
These checks establish reproducibility, not gameplay semantics.

## Reference calibration and probability projection

Calibration selected 64 ranked TRAIN charts: eight in each combination of a
whole-chart one-star band from 2 to 6 and LN fraction at most .1 or at least .5.
Selection used seed 261220, distinct song groups and a deterministic source-hash
ordering; it excluded the three source-timing diagnostic charts.

Sixteen-second source scopes supplied the existing full-prefix strain label.
For scopes whose label was between 2 and 6, calibration retained 400-ms peaks of
the causal response and weighted cells by duration divided by song duration.
Observations were binned by their scoped label. Each bin used the larger 95th
percentile of response-minus-label from the two LN strata, floored at zero.
Margins at 2.5/3.5/4.5/5.5 stars were .2940/.2786/.2820/.2405, with linear
interpolation and endpoint clamping. Only five TAP and four LN charts contributed
to the highest scoped bin; this is a development reference, not a precise
population quantile estimate.

A first policy subtracted a fixed quadratic excess-response cost from R1
logits. It reduced the fixed-H singles result from 4.852 to 4.036 stars but
failed its absolute-error bound. A finite penalty competes with learned logit
scale and supplies no final probability constraint.

The second policy instead projects the actual R1 distribution. Let $p(a)$ include
its neural consequence comparison, empirical recovery preference and scoped LN
amount feedback. For requested difficulty $D$, define

$$
B=\{a:R(H;a)>D+m(D)\}.
$$

If supported candidates exist outside $B$ and $p(B)>.05$, choose

$$
q(a)=
\begin{cases}
.05\,p(a)/p(B),&a\in B,\\
.95\,p(a)/(1-p(B)),&a\notin B.
\end{cases}
$$

Otherwise leave an already compliant distribution unchanged. This is the
minimum of $\mathrm{KL}(q\Vert p)$ subject to $q(B)\le .05$. It preserves odds
within each set and gives no supported row zero mass solely through this
preference. The .05 budget is a policy choice associated with the reference
tail; a marginal corpus quantile does not prove that this conditional constraint
is a quality guarantee.

When all supported rows exceed the threshold, the request is locally infeasible
under this response. The prototype uses the minimum-response tier as the recovery
set and records the failure to satisfy the original threshold. This does not
erase committed work, skip an H, or claim that the threshold was met.

Projection follows LN feedback so the bound applies to the distribution actually
sampled. It can change LN mass through correlations with placement. The integral
amount controller then observes the actual selected row; the two operations do
not commute and do not enforce two exact constraints simultaneously. Release-only
choices have identical attack responses and are unchanged by this partition.

## Generated-chart results

The generator weights stayed fixed at checkpoint SHA-256
`0f1ddfa5b351988fca04246ec080be106855f49b50fb493370c2d5afee1febb8`,
with the original projected LN amount policy. No new generator fit, onset-rate
readout or mean-head predictor was used. Executable product source was
`94c69684bf82e6621412a4d8a92fa0afb9e8ff72`; the sampling subclass and generated
artifacts belong to `20260926-causal-attack-work-v1`.

For the fixed-H singles chart, seeds 261101/261104/261105 gave 3.556/3.476/3.554
stars after projection, versus 4.852/4.660/4.833 without it. Mean absolute error
fell from 1.783 to .530. The slower chord case changed 2.907 to 2.874 stars, and
the LN source changed 3.492 to 3.386. Their source values were 3.103 and 3.005.
All supplied H timestamps were preserved.

Full audio generation then used fixed seeds 251702/251703/251706 for the following
development audios. Every case started from BOS without a source chart or seed.

| Audio | Requested stars / LN fraction | Baseline stars | Projection stars |
| --- | --- | ---: | ---: |
| Operation: Zenithfall | 3 / .2 | 4.334 | 3.650 |
| Hysteric | 3 / .2 | 3.314 | 3.392 |
| Take | 3 / .2 | 4.520 | 3.528 |
| Operation: Zenithfall | 3 / .7 | 5.026 | 4.897 |
| Hysteric | 3 / .7 | 3.856 | 3.748 |
| Take | 3 / .7 | 4.447 | 3.755 |

All H traces remained identical to their matching baseline. At requested LN
fraction .2, mean star error improved from 1.056 to .523, while mean LN-fraction
error changed from .0017 to .0039. At .7, star error improved from 1.443 to 1.134;
the .310 improvement missed the .35 expansion bound, and Zenithfall retained a
large absolute error. Hysteric at 5/.7 remained 4.533 stars; Take at 6/.2 changed
6.192 to 5.933. These are small, repeatedly used development cases, not a
population estimate or a final playable-model qualification.

Lens images and action/articulation tables show mostly single-note flow returning
in the fast fixed-H crop, changing chords in slower passages, and retained
overlapping holds. They also expose unresolved LN behavior. The high-LN Take
output has median hold duration 100 ms and 28.0% of holds at most 80 ms. Zenithfall's
dense high-LN crop interleaves rapid presses, short releases and longer overlapping
holds. Matching an amount or reducing stars does not establish good articulation.

The low-LN native runs took 3.32–6.14 seconds on one CPU thread on the test Mac.
Shared full-audio encoding plus first eight seconds of coverage took .262–.392
seconds from cached Mel; the slowest eight-second service window was .287 seconds.
These measurements exclude cold audio decoding/Mel extraction and do not measure
time to the first thirty rows.

## Scoped controls and limits of the response

Three further runs started at 3/.2 and applied 5/.7 over $[105000,137000)$ ms,
requested after 96 seconds of coverage. Published prefixes stayed unchanged;
open holds and response history crossed the boundaries. Each effective range
was evaluated independently with the full-prefix scoped strain readout.

| Audio | Before: requested 3 | Override: requested 5 | Restored: requested 3 |
| --- | ---: | ---: | ---: |
| Operation: Zenithfall | 2.397 | 3.787 | 3.837 |
| Hysteric | 3.244 | 4.255 | 3.313 |
| Take | 3.455 | 4.797 | 2.366 |

No range's difficulty-error regression exceeded .35 against the corresponding
baseline, but the absolute control mismatch remains visible. The Take restored
range lasts 7.237 seconds and has 30 heads, including 12 new LNs: fraction .40
for requested .20. Its baseline had 34 heads and fraction .294. This range was
reported separately rather than hidden inside a whole-song average or claimed
to have precise LN control.

Two diagnostics distinguish response blindness from timing constraints. First,
replacing every LN with a TAP **only in the difficulty calculation**, while
keeping all head times and columns, changes the high-LN Zenithfall rating from
4.897 to 3.779. The corresponding Hysteric/Take values become 3.270/3.374.
This is a counterfactual of the rating formula, not a proposed chart edit or a
percentage attribution of player difficulty. It demonstrates a quantity omitted
by the attack-only proxy while leaving a substantial Zenithfall head burden.

Second, an H-only lower bound can be computed without knowing any R1 choices.
Let $U_H$ accumulate two per H with decay .125, and $V_H$ accumulate one per H
with decay .30. Since every H needs a head and a per-column maximum is at least
the mean,

$$
R(H;a)\ge .18\left(V_H+U_H/4\right).
$$

In high-LN Zenithfall this lower bound exceeds the reference at 57 H timestamps
between 293.865 and 317.744 seconds. Its actual generated state has 109 locally
infeasible queries. Those 57 timings already prevent satisfying this particular
response threshold regardless of R1 cardinality or placement. Zero lower-bound
violations elsewhere do not prove feasibility: column recovery and hold state
can still restrict the available rows. These counts are not a division of total
model fault between skeleton and R1.

## Design implications

An explicit response state and a probability constraint address different
problems. The state must preserve the gameplay distinctions relevant to its
declared response; projection only enforces a condition on that response. A
stronger projection cannot recover omitted hold/release semantics.

The H-only bound provides a timing-owned signal: it uses the skeleton's own
history and the invariant that an H bears at least one head. It needs no actual
R1 chord counts or TAP history. Any use in timing generation must define a
consistent waiting-time law, including survival, termination and invariance to
query chunking; an unexplained timestamp veto is insufficient.

LN occupancy is necessary for legality, but an arranger also needs to retain
relationships between a hold's musical origin and possible releases. A possible
small shared representation could retrieve audio at actual active LN starts
and combine it with ages and current/future audio. Such inputs are recoverable
from full audio and the permitted LN state; they need not carry R1 hidden states
or preselected endpoints into the skeleton. This remains a hypothesis, separate
from the observed limits of the attack response.

Future learning on generated histories must use valid targets for those histories.
Projected soft distributions or measured outcomes of their own continuations
can supply limited control supervision. Source next-row labels cannot simply be
attached to a changed prefix. Source imitation, musical fit, style organization
and LN articulation still need independent qualification. The attack-only policy
is not adopted as a general replacement for that work.
