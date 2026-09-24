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
