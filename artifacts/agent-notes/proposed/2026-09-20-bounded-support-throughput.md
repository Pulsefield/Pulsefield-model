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
