# Screening unpublished joint continuations

The planned audio generator can test its own near future before publishing it.
This optional research decoder samples release timing and row materialization
jointly, preserving generated H timestamps. It addresses a specific gap in
candidate evaluation: the earliest physically possible LN release does not
predict which lanes the composed model will actually free.

The implementation is available as `buffering.rollout_buffered`. The packaged
audio-file CLI still uses the original single-trajectory decoder. No checkpoint
weights or training objective change, and this policy has not established
musical quality or production readiness.

## State and information

`generation.rollout` and the buffered decoder share `ContinuationSession`.
A session owns the exact replay, H planner, row/skeleton history caches, release
survival threshold and event RNGs. It encodes complete audio once. In-memory
forks share frozen audio/model tensors and immutable replay/cache values, while
owning their queues, row lists and random generators. Advancing a fork must not
mutate the source session. These objects are not crash-recovery checkpoints.

The original information contract remains: row materialization reads audio,
future H preview and causal row history; release timing reads audio, its own
skeleton history and the minimal LN projection. Simulation supplies generated
future releases and rows, never future reference endpoints. Fixed H times can
be supplied for research component comparisons without supplying row content.

## Proposal, check and publication

A proposal advances from the last published state until a completed scheduler
step reaches the requested window duration, initially 8000 ms. The actual cut
can exceed that duration because an event or empty-time advance crosses it.
A further 20-ms halo is sampled. The check includes all generated heads through
that halo and the preceding committed lane clocks.

The current experimental screen requires zero strict same-column successive
attacks below 20 ms, counting TAP and LN heads. Its optional RH screen also
requires zero release-to-next-head gaps at most 20 ms. Exactly 20 ms enters RH
but not HH. Only the first attack after a release forms an RH pair. The RH
criterion is a corpus-calibrated experimental preference, not an independently
confirmed universal bad-pattern label. No minimum LN duration, generic onset
gap or restriction on cross-column timing is added.

The first proposal uses the original RNG states. A failed proposal causes a
retry from the same published state with independent R/row randomness; the H
and arrangement RNGs are retained. At observed empty coverage, retrying draws
a fresh exponential survival threshold conditional on the observed absence
of an event. An accepted boundary retains its exact cache/RNG state, so the
next window's first proposal reproduces the checked halo.

The first passing proposal supplies the published prefix. The halo itself is
not published yet. Consumer callbacks receive only accepted rows and settled
coverage in time order; an LN head can remain open. Rejected speculative rows
cannot escape through that callback. An optional research rejection observer
receives a separate in-memory fork for offline inspection, with shared weights
and audio read-only.

There are initially four attempts per window, including the first proposal.
On attempt, time or row exhaustion, the decoder returns the last published
prefix and an explicit incomplete reason. It preserves open LNs and does not
export invented endpoints. Consumer or observer exceptions propagate. A single
model operation or callback can exceed a wall-time bound; checks occur between
scheduler operations. Diagnostic counters include interrupted speculative work.

## Distribution and approximations

For a committed state $s$, let $q(W,L\mid s,A,H,z)$ be the existing model's joint
law for a proposed window $W$ and its halo $L$. The indicator $G(W,L;s)$ denotes
the configured close-pair screen. With unlimited independent attempts, the
accepted proposal would follow

$$
q_{\mathrm{accept}}(W,L\mid s,A,H,z)
\propto q(W,L\mid s,A,H,z)G(W,L;s).
$$

This conditional law is a reference for independent base-model proposals.
The implementation's first proposal instead reuses randomness for a halo that
was screened in the preceding window; subsequent retries draw fresh randomness.
That first proposal therefore also depends on previous planning work, beyond
the published rows alone. Do not identify the implemented policy with this
idealized conditional law, or estimate its normalizer from a mixture of first
and retried proposals.

The finite attempt budget can return an incomplete prefix. Successive windows
also have prefix-dependent normalizers, so this is not an exact sampler of the
whole-song model conditioned on every future constraint. The original NLL does
not evaluate the changed sampling policy. Passing the narrow indicator does
not establish musical interpretation, difficulty or player comfort.

[NeuroLogic A*esque decoding](https://arxiv.org/abs/2112.08726) is an analogue for
using future constraint satisfaction in autoregressive generation.
[The Alive Particle Filter](https://arxiv.org/abs/1304.0151) studies indicator
potentials and variable sampling cost. This prototype uses bounded window
rejection, without their search or particle estimators.
[Twisted SMC](https://arxiv.org/abs/2404.17546) provides a related learned
future-potential construction if an amortized proposal becomes necessary.
These are mechanism analogies, not evidence that the beatmap policy works.

## Evaluation and operational limits

Evaluation must pair unchanged and buffered decoding with identical audio,
weights, arrangement choice and H times. Original healthy trajectories should
remain row-identical when every first proposal passes. Report complete charts,
strict HH/RH counts, note/chord/LN organization, accepted and rejected windows,
attempt exhaustion and publication latency. Lens inspection must include entry
holds and independently released subsets rather than treating an aggregate
LN-head fraction as coordination.

`windows` records every attempt, cut, checked clock, close pair and validation
service time. `evaluated_speculative_rows` counts repeated and interrupted work;
`max_unpublished_rows` measures the largest single proposal retained for checking.
The inherited per-update timing fields describe batched publication, not the
cost of an individual speculative model step. Use window service and total
sampler time for throughput, and actual callback timing for startup.

Keeping H fixed deliberately exposes intrinsically infeasible plans and
limited proposal support. More retries cannot manufacture missing probability
mass. A failure must remain visible; this research decoder has no production
quality fallback. Improving predictive consequences, learning a future-potential
proposal or changing the joint event representation are separate interventions.

## Measured failure removal and composition change

At implementation `452abc689055e2660bda5f62989cf70c95b0db8b`, a paired comparison
uses the shared-profile checkpoint
`abc27f1d192869419e42729a6b9fcdfd1c507fd672e12c9a9c7c868a5082a2ef`.
The 15 cases comprise nine automatic-profile outputs, one unchanged clean
control and five fixed-H component crosses from the
[profile-path study](shared_arrangement_profiles.md#separating-the-two-condition-paths).
Every unbuffered output reproduces its saved baseline rows exactly. No model
weights are fitted or changed.

All 15 screened outputs complete and independently reparse with identical H
times. Strict HH pairs fall from three to zero; RH<=20 pairs fall from twelve
to zero. Across 337 accepted windows, 349 proposals are evaluated, including
12 rejected proposals. The largest window uses three attempts under the cap
of four. Seven cases require no rejection and remain entirely row-identical.

On Apple M5 with one CPU thread, total sampler time is 34.690 seconds for the
baseline and 35.953 seconds for buffering, a 3.64% increase. Maximum measured
validation-window service is .537 seconds; maximum 30-row/eight-second
readiness is .323 seconds from cached canonical Mel. This excludes imports,
checkpoint loading, waveform decoding and Mel computation. Rejected-future
completions for inspection are separate offline work. Maximum unpublished
proposal size is 136 rows. These measurements do not establish behavior under
OS contention, client transport or other songs.

An inspected Prom Queen episode demonstrates the release dependency directly.
The rejected trajectory keeps three columns held across H times 105943,
105945 and 105949 ms, creating 2- and 4-ms repeated attacks in the sole free
column. The accepted continuation closes the already-started column-2/3 holds
at 105429 ms. The same three H times then use columns 1, 0 and 3, while a new
column-2 LN remains held. Later rows retain overlapping holds, independent
releases and repeated double grips. The earlier published LN heads are unchanged;
their tails had not been promised, so this remains consistent with incremental
LN publication.

The comparison nevertheless fails its composition guard in two LN-heavy
component crosses. The guard limits absolute LN-head-fraction change to .10.

| Case, fixed H0/downstream profile 2 | Baseline LN-head fraction | Buffered fraction | Held lane-time fraction, baseline to buffered |
| --- | ---: | ---: | ---: |
| Prom Queen, seed 19 | .5726 | .3995 | .3779 to .2904 |
| Airborne, seed 33 | .4375 | .3021 | .2267 to .1589 |

Both still contain substantial independently articulated LN material in their
inspected fixed scopes. The count change is therefore not a complete loss of
LN capability, but the requested arrangement is not preserved within the
declared bound. All per-case note counts remain between .940 and 1.069 times
baseline, and the other numeric guards pass.

In Prom Queen, LN starts inside the two resampled windows increase from 57 to
64. The remaining suffix outside those windows changes from 288 to 110 LN
starts; the already-published prefix is identical. For Airborne, the resampled
window changes from 46 to 4 LN starts, while the remaining suffix changes from
1195 to 810. Changes therefore extend beyond the locally screened windows.
Changed history and continued RNG state are both live explanations; these
counts do not isolate their contributions or prove a systematic population
bias toward taps.

The completed Lens inspection covers 52 declared scopes: 31 newly inspected
scopes with 109 time pages and complete action/articulation tables, plus 21
scopes reused through exact action, entry-hold and articulation identity. The
scope selection includes fixed musical contexts, every resampled publication
window and every rejected proposal. It is not whole-song visual review.

The accepted scopes retain dense cross-column H bursts, broad chords,
overlapping holds and independently released subsets. Composition changes are
visible as well as measurable. In Airborne's LN-heavy edited window, a mostly
continuous LN relay becomes primarily single/double taps before LNs return.
Prom Queen's edited windows remain LN-rich, but its later fixed scope changes
from three occupied columns with repeated free-column taps to more available
columns and different overlapping holds. Neither LN presence nor aggregate
LN fraction captures this organizational difference.

A supplementary Airborne inspection finds an isolated column-0 LN from
82889 to 82890 ms, well before audio end and without full occupation. The
1-ms object passes both close-pair screens. Its musical and player meaning
is unresolved; it demonstrates an uncovered duration/articulation question,
not evidence for a universal LN-duration floor.

This is bounded evidence of physical-failure removal at low measured sampling
cost, with two failed composition guards and retained expressive mechanisms
in inspected scopes. No listening or player test establishes musical quality.
The policy remains a research option and is not enabled in the packaged CLI.

Local evidence owner: `artifacts/joint-audio/20260925-unpublished-continuation-v1`.
Result SHA-256:
`3fff511d78ef4d0fd3400f33853d2e79219a69dbd034e67bee9d1263b5cb2ca9`.
Completed review SHA-256:
`7d41b477474a0ee0c0a95a367675fe51d33b743794be9568700ec675332c7267`.
The supplementary duration scope adds one inspected time page to the 109 pages
in the declared plan. A rendered page alone is not evidence of inspection.
