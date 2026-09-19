# Agent Note: Batch exact R0/R1 support without changing probabilities

Note ID: 2026-09-20-bounded-support-throughput
Status: proposed
Kind: implementation
Created: 2026-09-20
Updated: 2026-09-20
Product revision: 4a98c0230549baa19ad860c3a13caf301e0f1f71
Scope: Behavior-preserving vectorization of R0/R1 candidate support for bounded training and native generation
Related: 2026-09-20-r1-action-consequence

## Observed bottleneck

The action-consequence comparison runs under a frozen source. During its serial
training, a separate read-only CPU1 probe profiles the original4M R1 checkpoint
on the first two already pinned extension draws:451 queried rows,384 supplied
onsets and raw shape[2,747,2,131]. Under cProfile the complete forward takes1.095s;
451 calls to `Schedule.row_support` account for .923s, with115456 scalar
`row_possible` calls. Their repeated validation and state construction dominate
this profile. cProfile disproportionately charges Python work and a separate
CPU1 trainer is running, so these are attribution timings, not production speed.

The parent checkpoint SHA is
`ed4ad7dcec30fb2c6f13ee39908bb34c45b06b41799cdf30cfe410d96efdac28`;
shared extension plan SHA is
`767138e58c5288593be284b29a9333544e048468efb8d3f3aae2e42ece4c0fc6`.
Raw probes belong to the existing action-consequence artifact owner:
`support-profile-v1/` and `support-prototype-v1/`. No running trainer, product
source, model weights, optimizer or support rule was modified by those probes.

## Prototype evidence and scope

An artifact-only NumPy batch translation of lane legality, seed obligations,
terminal closure, required H and future-onset room agrees with the scalar oracle
on4244 R0/R1 states, or1086464 candidate booleans. These include the first eight
real draw intervals and generated valid prefixes with no-events and true ends.
On the first real microbatch, CPU FP32 loss and every parameter gradient are
bit-identical when only the support computation is replaced.

Three unprofiled paired forward measurements are .588621/.597983/.598170s for
the scalar path and .174294/.159727/.155938s for the vectorized prototype, a
3.744x median ratio. Corresponding mask-only times are roughly .43–.44s versus
.0183–.0184s. These tiny measurements overlap a separate training process and
do not establish sustained training speed or MPS performance. Prototype report
SHA is `11201a6cc9ea5c98ed12e0c32a495c1b866ea70d55a7ebe75cd56dcd6249b7d5`;
prototype driver SHA is
`2416ad5c62d3ce571fc1791ed60e625f3d78968b363665f496d90a41c5752031`.

The selected implementation keeps `Schedule.row_possible` and scalar support
as the independent exact oracle, adds one batched equivalent for R0/R1, and
routes the central decision scorer through it. O1 head/endpoint support remains
owned by its existing implementation. This is a runtime optimization: no new
model input, mask restriction, sample filtering, objective, parameter or RNG
operation belongs in the change. The user authorizes autonomous implementation
and meaningful commits; the Note remains proposed and is not accepted by an
engineering result.

The isolated `codex/bounded-support-batch` product worktree starts clean at the
source above. The ongoing action-consequence training stays on its original
worktree/source. Do not change that experiment's execution identity or evaluate
its models on another source without an explicit recorded runtime transition.

## Verification and reconsideration

Require exhaustive small combinations of occupancy, known-end obligations,
current/next-H roles, intervening candidates and true-end states against all256
scalar choices. Include R0, R1, no-event states and empty batches, and reject
unsupported O1 calls. Check CPU/MPS full likelihood and all gradients, nonzero
row-consequence modes, native generated decisions/recovery and checkpoint
compatibility. Preserve action ordering and all exact source/seed conditions.

Use real pinned draws for an end-to-end forward/backward/Adam comparison from
identical weights and moments. Require exact CPU updated tensors and fixed-seed
native decisions; any discrepancy blocks integration. Measure memory and warm
runtime separately from equivalence, without conflating cProfile attribution
with unprofiled throughput. Bounded engineering probes use at most120seconds
each, CPU1,6GiB footprint and128MiB maximum swap growth; tiny MPS checks use the
existing test resource envelope. No trained quality candidate is selected here.

Publish only the scoped implementation and verified runtime contract after the
focused owners pass. Final model quality and difficulty/LN results remain owned
by their learning/evaluation experiments.

## Implementation and verification result

The scoped optimization is committed at clean product revision
`9894761e8608ced818e94578a95649f5334b9d7b`, a descendant of the baseline above.
`support.py` owns the batch Boolean implementation. Both the central model scorer
and the native generator's deterministic-choice precheck use it; scalar
`Schedule.row_possible`/`row_support` remain unchanged as the independent oracle.
Model parameters, configs, checkpoint formats, candidate ordering and RNG calls
are unchanged. No generated quality result is asserted by this commit.

The new owner tests exhaust7696 typed-state combinations across occupancy, every
pending known endpoint and all current/future H-role combinations on a small
schedule, plus R0 and sampled generated prefixes, skips and complete schedules.
CPU/MPS likelihoods and all gradients agree with the scalar path. Native
R1 none/actions/frontier modes preserve complete decision records, probabilities,
restored raw histories and final CPU RNG state against scalar scoring and scalar
deterministic prechecks. Initial16 owner tests pass in23.05seconds. The selected
model/data/fork/training/consequence/generation owners pass120 tests in89.07seconds.
After routing the native precheck through batching, the invalidated support/
generation/generation-run selection passes71 tests in39.90seconds. Counts overlap;
these are selected local checks, not a full-repository suite. All commands use
`uv run --offline --python 3.10 --extra mps --group dev pytest -q`.

A real CPU1 AdamW comparison uses eight identical pinned TRAIN draws,1536 onset
exposures and two updates from identical4M weights, moments and RNG. Model tensors,
optimizer tensors/steps, RNG, gradient norms and all non-timing training metrics
are bit-identical after both updates. Scalar update times are2.156403/1.452965s;
batched times are1.145490/.846089s, a1.812x summed-time ratio. Observed footprint
maxima are762365440/810206768bytes and swap growth is zero. The second path retains
the first path's copied result state for equality checks, so that footprint
difference is not a causal memory-cost estimate. A separate CPU1 training process
also overlaps the probe; production throughput remains unmeasured.

The new worktree's artifact owner is
`artifacts/bounded-typed-continuation/support-batch-20260920-v1/`.
`cpu-update-parity-v1/report.json` SHA:
`68246aacb3c020b7e84d197021b2b233f6f177772bba11da09d375b0f9376956`.
No probe-trained checkpoint is retained or selected. The final outgoing source,
all affected callers, untracked owners, Note exclusion and `git diff --check`
were inspected before committing. Product and notes commits remain local.

The action-consequence learning comparison still runs on4a98c02 in its original
worktree. Using9894761 for its evaluation would be a separately recorded common,
behavior-preserving runtime transition, with an unchanged4a98c02 training-source
pin. Do not conflate source provenance or reuse an old runtime's recovery segment.

## Real native compatibility

On the clean9894761 runtime, the original4M checkpoint regenerates two fixed
prior outputs: source20/seed17 (LN-rich context containing the3ms release/head
case) and source26/seed23 (1050.064-second continuation). Both complete exports,
row journals and decision journals have exactly the same byte digests as the
frozen original outputs. Both reparse successfully. The probe uses a90-second
per-generation and120-second per-case bound, with the original conditions,
seed, presentation metadata and CPU1 native sampling. No resampling or quality
selection occurs. Report `native-parity-v1/report.json` SHA:
`71495ffdd6ccca4775f08a9c95899c86cee5fe30802af12b8dc3d49876bd0521`.

This is compatibility evidence, not a new quality result or an isolated speed
benchmark. The action-consequence Card revision3 records adoption of this common
evaluation runtime while leaving its ongoing training source unchanged. This
operational use does not change either Note's proposed lifecycle status.
