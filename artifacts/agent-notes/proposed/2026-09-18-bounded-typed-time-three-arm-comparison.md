# Agent Note: Bounded typed-time three-arm comparison

Note ID: 2026-09-18-bounded-typed-time-three-arm-comparison
Status: proposed
Kind: research
Created: 2026-09-18
Updated: 2026-09-18
Product revision: 2dc036579853c84815a2ddfa0b1b15b55a4ac0ff on codex/bounded-typed-continuation; published review base 89d5379f150cba9d1684822a44166765d38f644f
Scope: Common finite-context training and generation for original event rows, typed event rows and complete note objects
Related: 2026-09-17-oracle-time-mac-training-and-playable-continuation

## Question and authority

Can a native complete-object factorization learn more playable organization than
a typed row factorization with the same external information, under matched
data exposure and practical Mac compute? Preserve the original event-row task
as a third arm to keep insufficient training of the original contract falsifiable.
The user continues to authorize implementation, model/plan changes and bounded
experimentation toward playable audio-free continuation. No exact Note/Card
acceptance or quality completion is inferred.

The user supplied an external expert response with SHA-256
`07fcb2f496a24d116ffe8e4f20ff0c01346e525d16daf5e9823ed209aaa33e6a`.
Its recommendation is to build a common bounded, batch-trainable baseline,
train all three arms from scratch, and compare typed rows against native objects
with the same H/R/object-seed information. It rejects interpreting the frozen
old actor's TAP-heavy pilot as a native-object test. It also warns that full
prefix execution without cross-boundary writer gradients is not evidence of
learned long-term organization. The earlier owner records new type replay and
TRAIN census results consistent with keeping these questions separate.

## Three conditional tasks

- R0: source event-union R, physical-row seed, every R point nonempty; no H
  roles or future seed endpoints enter prediction.
- R1: R, required onsets H, complete-object seed and true endpoint of time;
  mandatory heads on H, no new heads elsewhere, optional no-event decisions,
  and rowwise release decisions for suffix-born LNs. Seed LN endpoints are
  fixed obligations; suffix LN endpoints remain undecided until a release.
- O1: same external information as R1; jointly select four-lane onset heads
  and then each new LN's complete endpoint. All previous generated/teacher-forced
  object plans enter subsequent head choices explicitly. Releases materialize
  deterministically; unused candidates never become neural history rows.

All arms preserve exact gameplay facts and legality. R1/O1 have the same complete
chart sample space when viability masks are correctly defined. Compare complete
suffix code length per source onset, including all endpoint factors, any order
mixture and feasibility normalization. Window-local row NLL and endpoint-head NLL
are not interchangeable. R0 comparison changes conditioning and support together.
Report seed-boundary-open-LN and seed-boundary-closed subsets separately.

## Common encoder and finite dependency contract

Start with a small shared-hand temporal convolutional encoder and clocks directly
at the joint readout. A causal dilation stack offers an explicitly finite raw
dependency boundary and dense teacher-forced computation. This is an implementation
choice to test, not a claim that convolution is universally preferable to attention.
The closest generic analogue is Bai/Kolter/Koltun's TCN study
(https://arxiv.org/abs/1803.01271); its tasks do not establish mania quality.
Transformer-XL (https://arxiv.org/abs/1901.02860) instead supplies a segment-
recurrence contract, not proof that full-BOS learned replay is required.

Eight single causal width-three layers with dilations1,2,4,8,16,32,64,128 give
511 materialized content rows of receptive field. A token's preceding physical
gap requires one extra boundary timestamp, yielding a declared512-row raw
history envelope. Current exact clocks/plans are separate complete facts;
raw history older than this envelope must not influence the learned path when
those facts and external timing conditions are held fixed. Cropped recomputation,
full sequential execution and cached generation must agree. A zero-length real
history is BOS; missing older learned history at a later time is TRUNCATED.
Do not introduce recursive learned history through a supposedly bounded cache.

Current exact state includes true LN starts, known/unknown planned endpoints,
last attack/release clocks, counts and last physical row. O1 knows all previous
object plans; R1 knows only seed commitments; R0 knows no future endpoints.
Tokens may encode already committed object endpoints, never current/future labels.
Keep raw state/cache parameter-independent across updates. Re-encode the bounded
raw prefix with current parameters and allow gradients through its finite
receptive field, rather than detaching every64rows as in the old trainer.

Known timing inputs include recent candidates and typed roles where permitted,
next supplied long-gap descriptors and multiscale candidate/onset counts. They
must use full external timing, not training-window boundaries. No audio, target
counts, chart identifiers or annotation labels enter the predictor.

## Endpoint dependence and normalization

Current-row LN endpoints may depend on earlier endpoint factors. Use two mirror-
reversed lane orders and marginalize their equal mixture for likelihood; sample
an order from that mixture for generation. Shared-hand weights alone do not
guarantee endpoint-order symmetry. Test the whole joint distribution under mirroring.

Each factor is a normalized distribution over choices admitting at least one
legal completion, including a lane free strictly before the next required H.
This defines a locally masked autoregressive joint model. It is explicitly
different from conditioning an arbitrary unmasked autoregressive joint on
feasibility, whose normalizer can require combinatorial enumeration. Use the
same factor masks and normalizers in training and generation. Do not reuse the
previous independent-endpoint conditioning formula for dependent endpoints.

Keep every strictly future R candidate. Candidate scoring must be chunked with
exact logsumexp and a bounded backward strategy; never cap rare targets at16/128.
Candidate-budget microbatches and recomputation must be checked against dense
values and gradients before real training. No minimum attack interval, maximum
LN duration, source-end fallback or repetition penalty is part of these arms.

## Planned evidence and budgets

First pass mechanical and model checks, then a16-TRAIN-chart learning check
covering taps, independent multiple-LN ends, long gaps and LN state across crop
boundaries. Show distinct head/type and endpoint gradients and learning; do not
pass a model because endpoint loss alone decreases. Real-source and synthetic
long-lived anchors may both be needed because observed TRAIN LN spans fit within
128 event positions, shorter than the chosen512-row learned context.

The main exploratory comparison uses matched source intervals and initial seeds,
with near-equal parameter counts, plain full-model NLL normalized by actual source
onset exposures, and no structural marginal auxiliary losses. Planned checkpoints
are250k/1M/2M onset exposures and at most four training hours per arm, whichever
comes first. Final commands, source/split selections, initialization, optimizer,
sampler and resource guards will be pinned in a runnable Card after the small
learning/resource check; these main runs are not launched by this Note.

Use separate equal-exposure and equal-compute ledgers. Count physical rows,
onsets, endpoint factors, unique target coverage, group coverage and recovery
costs. A longer-context512→1024 comparison belongs only after the main screen,
using the most promising arm and matched fine-tuning budget.

Generation should include the existing regression cases plus separately sampled
12ordinary/12stress groups and three seeds, with mechanical invariants all passing.
Report recurrence burden, chord distributions, LN births/durations/concurrent
occupancy/independent actions, gap crossing and diversity by stratum. Machine
Foundation checks do not replace a new human blinded comparison. A proposed
60% preference point estimate with group-level evidence over chance is a future
continuation criterion, not an assumed success threshold already evaluated.
Independent confirmation needs a second training initialization and unused groups.

## Immediate implementation gate

Implementation proceeds from the clean published snapshot in the new product
worktree; only the existing diagnostic report is copied as a behavior-neutral
documentation update. Establish the shared exact-state/scheduler contracts and
unit checks first, then the finite encoder and pointer likelihood. Commit a clean
source before any model-backed real-data experiment, and pin that commit in its
Card. Synthetic unit checks have a ten-minute envelope per invocation and run
with explicit mps dependencies on this Mac. No larger model/data run is implied
until its exact resource and input manifest exists.

Evaluation: REFINE. Note/Card acceptance: none. This owner tracks the new
comparison; the earlier owner retains historical M3 and endpoint-probe evidence.

## Implementation result: contracts, probability and bounded source projection

Clean product commits `ee83ad0b3d028aed1fd22e32e70b51bae98bfa23` and
`2dc036579853c84815a2ddfa0b1b15b55a4ac0ff` implement the exact scheduler, finite
temporal encoder, permitted features, joint row/head distribution, dependent
endpoint mixture and teacher-forced bounded windows. No new real-data model
training or generation has run. These commits remain local; the published expert
review snapshot is unchanged.

The default R0/R1 models each contain 2,281,104 parameters; O1 contains 2,355,835.
All common encoder/exact-readout/fusion weights initialize identically for a shared
seed. O1's extra parameters implement endpoint scoring. Raw content uses committed
actions, the preceding physical gap and permitted new plans. Exact query features
contain complete clocks, occupancy, counts and known remaining durations, with
R/H-only lookahead derived from the complete external condition.

The full pointer likelihood packs factor/candidate pairs under a candidate budget,
normalizes with chunked logsumexp, and recomputes features/activations during backward.
CPU/MPS tests match dense values and every participating gradient. A saved-tensor
test confirms more than a tenfold reduction in retained tensor storage for its
synthetic 3,695-candidate-pair fixture. This is a correctness/storage mechanism
check, not measured real-corpus throughput or peak system memory.

51 bounded-model tests and 40 unchanged canonical replay tests pass together
(`91 passed in 5.62s`, Python3.10/Torch2.11 with explicit MPS dependencies). Checks
cover full 511-token dense/cache/crop equivalence, parameter gradients, exact
index/replay equivalence, source-target support, mirrored joint probabilities,
normalization over every feasible endpoint assignment on a small fixture,
streamed sampling and unchanged draws under different chunk partitions.
R1 inputs remain identical when only unseen source LN pairings change. An O1
current endpoint label cannot affect its own head decision. Source windows retain
real terminal semantics and endpoints beyond the target interval. Tests run on
CPU and available MPS where applicable; this is not a full repository test run.

The next gate remains an explicitly bounded 16-TRAIN-chart mechanical/learning
check, with recoverable source, a pinned interval manifest and separate head/type
learning criteria. Passing these unit checks does not establish learned LN use,
training efficiency, generated organization or playability. Evaluation: REFINE;
Note/Card acceptance: none; lifecycle remains proposed.
