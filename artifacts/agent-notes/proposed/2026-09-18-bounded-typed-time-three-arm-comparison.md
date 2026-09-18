# Agent Note: Bounded typed-time three-arm comparison

Note ID: 2026-09-18-bounded-typed-time-three-arm-comparison
Status: proposed
Kind: research
Created: 2026-09-18
Updated: 2026-09-19
Product revision: 15d27db0b0723d7f606b1429c901d26c1f61e5ac on codex/bounded-typed-continuation; initial corpus source 1693d62ffaca04b2a6127d8e3a72d1988adf441f; generation source 21475e65d773b7e7199accf0750de584d9f10ce9; short-fit trained source a178bcfe2badaaea47ae9abce02f2494b8ff9643; published review base 89d5379f150cba9d1684822a44166765d38f644f
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

## Experiment Card: bounded-typed-16-train-learning-v1

Revision: 1. Owning Note: 2026-09-18-bounded-typed-time-three-arm-comparison.
Acceptance: none. Execution uses the user's standing authorization to implement
and run bounded model/formulation experiments; this remains exploratory.

Question: does each native arm learn actual source action/type choices on a small
TRAIN slice, while O1 also learns its complete endpoint distribution, with a
bounded practical Mac memory/compute footprint? A pass permits designing the main
comparison; it does not establish generalization, preferred factorization or
generated quality. The closest encoder analogue and task differences are recorded
above. Full object planning remains the selected hypothesis; typed rows and the
original task remain live alternatives rather than rejected branches.

Clean source: `a178bcfe2badaaea47ae9abce02f2494b8ff9643`. This adds the packaged
Hydra learning-check boundary, pinned source selector, parameter/draw/optimizer
checkpoint and type diagnostics to the preceding implementation. The selected
bounded test suite passes 54 tests, including CPU/MPS model execution and a small
synthetic run through the actual runner. No real-data learning preceded this Card.

Baseline: each arm's fresh initialization and initial teacher-forced score on its
own selected intervals. Initial numerical values are intentionally unmeasured
until the recorded run; this is a paired within-arm learning check, not a numerical
R1/O1 likelihood comparison on local windows. Intervention: 128 updates of native
full likelihood, without auxiliary structural losses or decoding penalties.

Inputs:

- Existing TRAIN catalog byte SHA-256
  `e31b7e8f4daa044503ef2b8411bc41727ba371eec804c9462a4608be8112ad28`,
  `artifacts/oracle-time-review/20260915-adfb1ee/catalog.json`.
- Existing split canonical SHA-256
  `15175f45e91cf7299a9a30166731bf38ee7361399b346fb692cf68e76de5992a`,
  `artifacts/scoped-style-modeling/prepare-v1/split-manifest.json`.
- TRAIN census byte SHA-256
  `7a6380fd1efab72ffaaa60250a8b2a92869675699d9d108c7dc87708bb5d0fd9`,
  the earlier `seed-type-population-v1/results/per-chart.jsonl`.
- Fresh selection manifest
  `artifacts/bounded-typed-continuation/smoke-20260918-v1/intervals.json`,
  SHA-256 `fe6090618154f2026e34ce5d432ddc2368d692cd50f0fc28d563a42e308e8fda`.
  Selector seed371 inspected33sources and selected16distinctTRAINgroups in0.161s.
  Four TAP, four mixed, four LN-rich, two independent-endpoint and two long-gap
  strata each contribute128onsets per chart, totaling2048distinctonsets. Source
  lengths range673–2288rows. Seven intervals use a full512-row raw envelope;
  examples include seed/crop-crossing holds, multiple LNs with different ends,
  and4336/2709ms gaps. These are selection facts, not behavior required of a
  generated arrangement. The source cache is the existing admitted
  `artifacts/oracle-time-continuation/full-cache-v1` in the original worktree.

Procedure: run
`uv run --offline --python 3.10 --extra mps --group dev python -m pulsefield_model.research.bounded_typed_continuation.smoke_hydra`
once for each `model.arm=O1`, `R1`, `R0`, serially on the Mac. Supply the exact
interval, catalog and split paths/digests above, the original admitted source
cache, and fresh sibling output directories `o1`, `r1`, `r0` under the manifest
owner. Runtime config records resolved absolute input paths; the note stores
repository-relative provenance. Every other value stays at packaged
`bounded_typed_smoke.yaml`: modelseed171, shuffleseed271, 128AdamWupdates,
two128-onsetintervals/update, learningrate0.001, decay0.01, clip1.0, eight-update
linearwarmup, width128/eightdilationlevels/expansion4/rank16, candidatebudget8192,
oneCPUthread, report/checkpoint everyeightupdates. Expected supervision per arm is
32768source-onsetexposures; a complete16-drawcycle visits every interval once.

Bounds: each complete run, including loading and initial/final evaluation, has
1800wallseconds. At most three serial arms (90minutes outer total), MPS6GiBdriver
guard/8GiBhardallocator, RSS6GiB, availablememory≥2GiB, swapgrowth≤128MiB,
checkpoint≤128MiB, totaloutput≤512MiB/arm, diskreserve1GiB. No downloads/network
are needed. Existing outputs are never overwritten; automatic resume is disabled.
Stop on time/resource guards, nonfinite loss/gradient, a source outside support,
identity/digest failure or any exact-state/normalization failure. A failure keeps
the last durable checkpoint and failure record; changing the procedure requires
a new Card revision/fresh output, not silent continuation.

Learning gate, evaluated on the same2048sourceonsets before/after:
total native factor NLL per onset decreases by at least20%; row/head NLL per onset
decreases by at least10%; mean negative log probability of the observed LN_START
lane action decreases by at least10%; observed TAP lane NLL does not increase.
O1 additionally requires mean joint endpoint NLL per born LN to decrease by at
least10%. All mechanical checks and finite-gradient guards must pass. Report each
arm separately; a singleton deterministic factor has zero cost. These thresholds
are an engineering fit gate, not significance estimates or a claim that the
conditional finite-context data can be memorized to zero loss.

Record source/uniqueonset exposures, repeated draws, physical/prefix rows, context
span seconds, endpoint decisions, both order-factor counts, full candidate pairs,
preparation/forward/backward/optimizer times, evaluation and checkpoint costs,
and resource journal. Forward includes prefix encoding and candidate construction;
backward includes candidate recomputation. No learned prefix is reused across an
optimizer update. The fresh-initialization and final scores are deterministic
teacher-forced diagnostics on this deliberately chosen TRAIN slice. A positive
result supports pipeline learning only; failure calls for debugging or revising
optimization before a main screen. CPU/unit correctness alone cannot substitute
for this head/type learning result. Generated Foundation/human judgment remains
outstanding regardless of the result.

## Result Log: 16-TRAIN learning check, revision 1

All three serial arms completed their exact 128 updates with clean source
`a178bcfe2badaaea47ae9abce02f2494b8ff9643`. Acceptance: none; exploratory execution
under the standing user authorization. Each arm saw32768onsets over2048distinct
onsets, with every selected interval drawn16times. Draw order and common runtime
configuration were verified identical. All support/finite-gradient/resource checks
passed, but **all three fail the predeclared LN-positive type-learning gate**.

| Arm | Native factor NLL/onset, initial → final | Row/head NLL/onset | LN-positive lane NLL | TAP-positive lane NLL |
| --- | --- | --- | --- | --- |
| O1 | 6.953991 → 2.592007 | 4.166273 → 2.194355 | 1.030073 → 1.449072 | 1.099474 → 0.801876 |
| R1 | 4.409525 → 2.441500 | same as total | 1.118710 → 1.339264 | 0.985903 → 0.723925 |
| R0 | 4.691517 → 2.794169 | same as total | 1.128351 → 1.642088 | 1.004750 → 0.732523 |

O1 endpoint NLL per born LN falls6.895225→0.983565. Its total/head/endpoint gates
pass; its LN-positive NLL rises40.68%. R1 and R0 LN-positive NLL rise19.72% and
45.53%. These local training-window likelihoods do not rank the representations;
the factor costs and external conditions differ. No generated quality claim is
made. The trained LN-type positive metric worsens in all three arms, so the object
pointer alone is not a sufficient explanation of that measured result.

Training wall times: O1 100.070s, R1 65.250s, R0 64.710s. Complete run times including
loading and initial/final evaluation:112.186s/72.297s/70.997s. Each arm encoded
37328target physical rows and65776prefix rows; dense padding raises executed row
positions to139078. O1 scored13248LNdecisions,26496orderfactors and21913664future
candidate pairs, with recomputation during backward. Its preparation/forward/
backward/optimizer totals were1.802/29.688/51.361/2.536s; remaining train time
includes guards, checkpoints and logging. These costs are measured on this small
640–2400-row selection, not a whole-corpus throughput estimate.

Sampled RSS peaks:2785607680/2826223616/2681520128bytes. Sampled MPS driver peaks:
1024983040/822050816/822050816bytes. All swap-growth maxima arezero and minimum
available memory remains above6.33GB. The seven complete512-row context envelopes
span59.029–87.300s, median69.851s, from the oldest required predecessor timestamp
to the first target query. Short-seed contexts are excluded from that summary;
these seven deliberately selected examples do not estimate corpus-wide density.

Checkpoint SHA-256, in arm order:

- O1 `75276c37ef92d7af2a5825ab52d4d22280ee19455376839513a4de1ac33b9aa4`
- R1 `d4c6ee5b7492a4d5a422263052d51cf73efe75bf5d34a3998282eb82a3579945`
- R0 `1e33ce1297bd1548da0a59ca70ad99e848184c70976b8280b075a733181b9edb`

Readout: `smoke-20260918-v1/learning-readout.json`, SHA-256
`b76d1e742887eb22bb6e244229d3925e63e829584780917b80efd2479ee38594`.
Readout script SHA-256
`25257b41c8a10175af25bf629c569a276d8a302ae3ab7bee66da1803bb3a3cc6`.
Both live under `artifacts/bounded-typed-continuation/`. The selected intervals
contain3369trueheads, including828LNs, and148rows with independent multiple-LN
endpoints. Five have open seed LNs; three have an open LN at the target boundary.

Evaluation: REFINE. Keep the failed gate unchanged. Before another training screen,
separate inadequate minority-type discrimination from a possible measurement
problem: positive-only NLL against random initialization can worsen when a model
corrects an initially excessive LN prior. That possibility does not erase the
observed failure or establish successful type learning. No main-screen run is
authorized by a claimed pass, and no such pass is recorded.

## Experiment Card: bounded-typed-type-audit-v1

Revision:1. Acceptance:none. This is a post-hoc, read-only diagnostic on the same
three initializations/final checkpoints, not a revised success claim for the
completed learning Card. Source and 16-interval manifest remain the identities
above. Frozen diagnostic script `smoke-20260918-v1/type_diagnostic.py` SHA-256
`d342fd8399ad57fd9feab5098c01ba352951cdd41322e12f2f93751269e9ffb1`.

Question: did LN discrimination/calibration improve despite worse positive-only
NLL, or did likelihood gains primarily come from unrelated factors and prior
adjustment? Reconstruct the initial parameters from the saved model seed and exact
source; reproduce initial/final row/head NLL within2e-5nats/onset before interpreting
new metrics. No optimization, endpoint resampling, free generation or data
expansion occurs.

Measure both proper binary log loss/Brier and tie-aware AUROC/average precision:
(1) LN versus all other actions on structurally LN-capable lanes at source H;
(2) LN versus TAP, conditional on a true source head in that lane, where both
types are feasible. The second is a diagnostic conditional score, not oracle
head-location input to generation. Compare with each same-cohort fitted constant
prevalence baseline; retain per-interval results to distinguish pooled style
separation from within-chart discrimination. R0/R1/O1 eligible cohorts can differ
because supports differ, so do not rank arms by unqualified pooled scores.

Interpretation: improvement over initialization alone can be prior calibration.
Better log loss/Brier than the fitted prevalence baseline plus AUROC above0.5
supports some type discrimination on this TRAIN slice; inspect within-interval
results before attributing it to local pattern learning. Constant-like ranking
with prior-only score gains points toward insufficient discrimination. Neither
outcome passes the original gate or establishes generated LN mode coverage.

Execute `uv run --offline --python 3.10 --extra mps python artifacts/bounded-typed-continuation/smoke-20260918-v1/type_diagnostic.py`
from the same clean product worktree. Fresh output is
`smoke-20260918-v1/type-diagnostic-v1/`; no overwrite/resume. Bound300wallseconds,
oneCPUthread, MPS6GiBdriver/8GiBallocator, RSS6GiB, available≥2GiB, swapgrowth≤128MiB,
output64MiB. Stop on identity/reproduction/numerical/resource failure. Record all
results and limits as post-hoc; no threshold or baseline is changed silently.

## Result Log: post-hoc LN type discrimination audit

The read-only diagnostic completed in9.290s on the same clean source and exact
three checkpoints. Readout SHA-256:
`1f56d656134486e325ab4b4cc8d2baee80015c267e3a9fc1403dcc923b8c1d43`,
`smoke-20260918-v1/type-diagnostic-v1/readout.json`. Initial/final head-loss
reproduction absolute total errors are at most0.000160nats over2048onsets, well
within the stated2e-5nats/onset bound. No parameters, targets or original learning
gate were changed. Acceptance:none; this remains an exploratory diagnostic.

All three arms happen to share7482structurallyLN-capable lane/onset positions on
this slice, with828positives (11.0666%). Initial mean LN probabilities are
34.253%/30.829%/31.438% forO1/R1/R0; final means are8.979%/10.198%/8.335%. Their
initial positive-only probability was therefore not a calibrated prevalence
baseline. Lower positive-only probability alone cannot distinguish improved
classification from failure to learn a minority type.

| Arm | Candidate LN binary NLL, initial → final | Candidate LN AUROC | Candidate LN AP, final |
| --- | --- | --- | --- |
| O1 | 0.484454 → 0.223365 | 0.621507 → 0.907881 | 0.557146 |
| R1 | 0.448692 → 0.225827 | 0.730855 → 0.902541 | 0.512226 |
| R0 | 0.459171 → 0.245046 | 0.639678 → 0.889859 | 0.461355 |

The fitted constant-prevalence baseline hasNLL0.347905 andBrier0.098419. Final
Brier scores are0.068546/0.070380/0.076401. Thus gains exceed merely fitting a
global LN prior on these observed histories. They do not establish generation
calibration; all three final mean LN probabilities remain below the observed rate.

Conditional TAP-versus-LN diagnostics use3367true source heads for which both
types are feasible, including828LNs; two structurallyforcedTAPheads are excluded.
This diagnostic never supplies current head locations to generation.

| Arm | Conditional type NLL, initial → final | Type AUROC, final | Median within-interval type AUROC, initial → final |
| --- | --- | --- | --- |
| O1 | 0.697922 → 0.333056 | 0.912251 | 0.531766 → 0.790624 |
| R1 | 0.635853 → 0.319357 | 0.913326 | 0.616157 → 0.817674 |
| R0 | 0.647677 → 0.349612 | 0.904172 | 0.576711 → 0.786009 |

The global constant type prior hasNLL0.557804. A stronger diagnostic giving each
interval its own fitted true LN fraction has weightedNLL0.371819; all three final
scores are lower. Within-interval AUROC improves in12/12,11/12and10/12intervals
containing both types. O1 final within-interval AUROCs range0.594–0.939. These
observations show local type discrimination as well as pooled style separation
on this small TRAIN slice. They neither rank generated arrangements nor establish
unseen-group performance.

Interpretation: the initial positive-only gate was a poor stand-alone detector
of type learning against an overpredicting random initialization. The original
Card remains failed by its stated rule; this post-hoc audit is not a retroactive
pass. However, describing the native models as having learned only endpoints
would contradict the complete binary/within-interval evidence. Subsequent
learning checks should prospectively use proper all-class scores plus within-
interval discrimination, retaining positive rates and free-generation collapse
diagnostics separately.

Next implementation priority is native sampled generation with finite-cache
recovery, skipped-candidate handling, endpoint obligations and export/reparse
checks. Then fix the main screen's metric contract and sample manifest before
training. No coarse LN-intent oracle, positive-class reward, duration truncation
or decoder repetition penalty is justified by this diagnostic. Evaluation:
REFINE; quality and independent confirmation remain outstanding.

## Implementation result: native generation and raw recovery

Clean source `21475e65d773b7e7199accf0750de584d9f10ce9` adds a native sampler and
an independent completed-output verifier. Source `6a2a5ae241c9c7e04ecb60d218f4ce0abae4a657`
is the intervening behavior-neutral research report. Model/training parameters
remain those of the learning-check source; no new optimizer run occurred.

`Rollout.from_seed` takes only the permitted R/H/seed condition. It uses native
normalized distributions at temperature1, CPU RNG and full-support endpoint
sampling. Deterministic candidates consume no RNG; absent events never enter
physical or learned history. Scored endpoint probabilities marginalize both
orders, while an explicitly unscored endpoint is marked unavailable.

Checkpoints retain exact state, commitments, recent raw physical events with
their predecessor timestamps, and RNG. They exclude learned buffers. Recovery
checks exact parameter/configuration/timing identities and rebuilds the bounded
cache using the same online kernels. Tests verify unchanged rows/plans/RNG and
cache values under CPU/MPS interruption, reject stale parameters and missing raw
context, retain a seed hold beyond the learned range, and finish on an unused
terminal candidate. Full default-width/eight-level CPU recovery is also tested.
Independent output checking covers seed fidelity, R/H coverage, occupancy, all
chosen O1 suffix endpoints and final closure; generated osu! bytes reparse to the
same physical rows. All73bounded tests pass in6.46s, including19newgeneration tests.

## Experiment Card: bounded-typed-native-generation-v1

Revision:1. Acceptance:none. This is a bounded TRAIN free-running diagnostic
under the user's standing experimental authority, not the main quality screen or
an independent generalization result. Original learning-gate failure remains
recorded. Question: do the current native distributions execute/recover correctly
and retain basic LN/attack modes under their own histories after the short fit?

Fixed source: `21475e65d773b7e7199accf0750de584d9f10ce9`. Use all16TRAINsources in
the same pinned `fe6090…e8fda` manifest, now from their actual30-note seed through
the true end of R, rather than only the supervised128-onset intervals. Use the
exactO1/R1/R0 checkpoints and SHA-256 identities recorded above, oneCPUthread,
CPUexecution and generationseed17 for every arm/source. No optimizer update,
temperature change, candidate cap, duration rule, repetition penalty, external
LN-intent label or source-suffix fallback is introduced. Typed arms receive only
the prescribed complete-object seed; rowR0 receives no future seed endpoints.

Frozen script: `artifacts/bounded-typed-continuation/smoke-20260918-v1/generate_pilot.py`,
SHA-256 `e80af7a68514231af2f87429dec1e4abbc0b30a32be3daf80a6a31f9700bd997`.
Execute `uv run --offline --python 3.10 --extra mps python artifacts/bounded-typed-continuation/smoke-20260918-v1/generate_pilot.py`.
Its clean-source assertion, pinned checkpoint digests and TRAIN catalog allocation
are checked before use. Scoring of sampled endpoint likelihoods is disabled for
this generation-cost measurement, explicitly leaving those log probabilities
unavailable; sampling still scores/normalizes the complete support. A read-only
module hook counts actual candidate pairs and neural queries without retaining
tensors or changing outputs.

Outputs: fresh `smoke-20260918-v1/native-generation-v1/`,48charts, per-source
condition, physical rows, candidate decisions, bounded raw checkpoints and osu!
exports; no overwrite/resume. Every512candidates and completion save raw state
plus durable journal boundaries. Atcandidate600of the first chart in each arm,
restore the raw checkpoint, require bit-identical current cache values and every
internal convolution buffer plus identical exact state/RNG, then continue.
Check every output independently and require exact export/reparse row equality.

Bounds:1800seconds for the complete48-chart run, oneCPUthread, RSS6GiB, available
memory≥2GiB, swapgrowth≤128MiB, totaloutput512MiB, checkpoint128MiB, diskreserve1GiB.
No network/download. Resource checks occur every64candidates and after each chart.
Stop on any mechanical/cache/identity/numerical/resource failure; preserve partial
outputs for diagnosis without declaring completion.

Report source and generated suffix heads/LN fraction, head-count histogram,
adjacent same-lane attacks below40ms and longest associated run, concurrent-hold
time, independently released holds, attacks while another hold continues, LN
duration quantiles, skipped candidates, total generation/recovery time, actual
candidate pairs and neural queries. The40ms locator is not a playability cutoff;
overlap and independent-release counts do not by themselves certify LN coordination.
Do not require per-chart source LN ratios or style copying. Low LN coverage,
sustained recurrence burden or mode collapse routes subsequent learning/quality
investigation; this diagnostic alone cannot pass a playability gate. Any rendered
inspection must use the Foundation/current-human judgment scope, not substitute
these counts for it. Evaluation remains pending until outputs are checked.

## Result Log: native generation revision 1 export-path failure

Revision1 stopped on its first chart after native generation, candidate600raw
recovery and independent completed-output verification. Playback-header loading
raised FileNotFoundError because catalog source paths are relative to the original
product worktree, while the new implementation worktree has no dataset directory.
The failure is before export/reparse and before any completed chart result; it is
not a completed48-chart diagnostic. The last resource guard was atcandidate832,
1.783s after startup. Preserve the failed output and original script unchanged.
First generated physical-row SHA-256:
`36c05654e69cb19ebfef7cdd6b18856c1ec3a221d1486d8e19b59f775cd4e081`.

Read-only preflight resolves all16selected raw source paths against the original
catalog-owning worktree and confirms each exists and matches its source SHA-256.
No source cache, model parameters or sampler were changed. Evaluation: REFINE;
the next revision repairs external playback-file resolution.

## Experiment Card revision 2: bounded-typed-native-generation-v1

Acceptance:none. All question, input/model/source identities, sampling policy,
comparisons, resource bounds, metrics and stop conditions from revision1 remain
fixed. The sole implementation repair resolves relative playback source paths
against the catalog-owning original worktree, and preflights every selected source
byte digest before sampling. The fresh output owner becomes
`smoke-20260918-v1/native-generation-v2/`; failed revision1 remains untouched.

Frozen script: `smoke-20260918-v1/generate_pilot_v2.py`, SHA-256
`5b6f4286fa82d7ff9bee76284268fa761d0a099d127200f8ff6156aa174ea8c4`.
Run `uv run --offline --python 3.10 --extra mps python artifacts/bounded-typed-continuation/smoke-20260918-v1/generate_pilot_v2.py`.
Additionally require the repeated first O1 physical-row file to be byte-identical
to revision1's completed generation before exporting it. This verifies that the
I/O repair did not change the previously generated trajectory. No optimization,
new seed, new sample selection or relaxation of mechanical checks is introduced.

## Result Log: native generation revision 2

The48-chart run completed in86.470s. All16conditions per arm passed independent
R/H/seed/occupancy/endpoint/terminal verification and exact osu! reparse equality.
The first O1 row file matches the failed attempt byte-for-byte. Candidate600raw
restoration in the first chart of each arm preserves exact state, RNG, current
learned content and all convolution buffers bit-for-bit; recovery times are
0.318/0.305/0.304s. No optimizer change or sampling intervention occurred.

Readout: `smoke-20260918-v1/native-generation-v2/readout.json`, SHA-256
`17ed6ca391e680ee99a19298db8873260974eaba56edbd9df4368893999c5956`.
All generated source-condition/row/decision/checkpoint/osu! files live in that
owner. The process exited successfully; no generation job remains running.

| Arm | New suffix LN heads / all suffix heads | Per-chart LN fraction min / median / max | Generation seconds |
| --- | --- | --- | --- |
| O1 | 675 / 23294 = 2.8977% | 0.6221% / 2.2184% / 6.6667% | 23.838 |
| R1 | 5555 / 26010 = 21.3572% | 8.4277% / 17.9342% / 46.7553% | 29.997 |
| R0 | 4158 / 27188 = 15.2935% | 6.9661% / 16.3581% / 29.2237% | 30.509 |

O1 samples504368futurecandidatepairs; its low generation cost is coupled to a
low LN birth rate, so this is not an equal-work pointer cost comparison. Neural
queries are14507/15363/16306. O1/R1 skip1669/1060unusedcandidates, whileR0 emits
allR by contract. Sampled CPU RSS peaks at309084160bytes, minimum available memory
is7980351488bytes and swap growth remainszero.

Adjacent same-lane attack pairs below40ms total57/103/127 forO1/R1/R0. The dense
source00126e732bc4 accounts for51/93/108 of those pairs, with longest rapid runs
4/7/4attacks respectively. These locators do not establish Jack or playability.
Suffix time with at least two held LNs is26335/387537/265344ms; independent-release
row counts are116/1247/919, and attacks while another hold continues are327/3410/2456.
These are factual action relationships, not LN-coordination verdicts.

Interpretation: under this short fit and one generation seed, O1 exhibits much
lower LN use across the selected collection, despite good teacher-forced type
discrimination. R1 retains a wider LN-use range. This warrants treating typed
rows as a serious practical candidate; it does not prove O1 intrinsically fails,
that source LN ratios must be copied, or that R1's denser LN arrangements are more
playable. Most complete suffix positions were outside the128-onset learning slice,
and neither full-corpus training nor independent quality evaluation has occurred.

To check an alternative explanation, product commit
`b544ea4d02dea5bf175991ec633045e53b9e20d6` adds six CPU/MPS parity cases without
changing runtime/model behavior. They override only random selections to force
source choices through the actual native generation commit path. Indexed exact
state, permitted raw content and query features match the training owner; online
pre-decision probabilities match dense teacher forcing across an old hold, context
expiry and a long gap. All25generation tests pass in5.17s. This rejects the tested
train/inference feature or cache projection mismatches, not all possible bugs or
the statistical effects of generated-history drift.

Next work: inspect sustained generated organization with time-proportional images
and Foundation/current-human contrasts, then define the main paired training
screen and prospectively repair the type-learning measurement. Keep inadequate
exposure, generated-history drift and factorization/optimization differences as
live explanations. No decoder repair, coarse LN oracle or object-model victory
is inferred. Evaluation: REFINE; visual semantic judgments remain unreviewed.

Calibration preparation: frozen Foundation file bytes still match
`b1aea3cbdfe9102e1657d01acfae3f36729467d0a8675b6272ba4f0b17c743ab`, and all eight
canonical workflow documents match the earlier typed-quality review's recorded
hashes. The find/get MCP tools are not exposed in this session. Current canonical
Python example extraction expects `humanComment`, which some observations in the
older feedback projection omit; that attempted extraction stopped with KeyError.
Use the current canonical reader/projection or an explicit missing-comment adapter
before semantic comparison. Do not promote inherited agent rationale into human
comments. No human record, Foundation or pin was changed.

## Calibration repair and scoped visual results

The canonical workflow reader plus Inspector comment ownership resolves the
older offline projection's missing `humanComment`. The artifact adapter reads
eight copied workflow documents through `WorkflowDirectoryV2.read`,
`effectiveHumanObservationsV2` and `readAgentReviewsV2`; it takes an actual human
decision rationale or direct human claim rationale, leaving missing comments
unrecorded. All eight original documents retain their prior hashes. No annotation,
Foundation, product reader or human confidence is changed. The canonical Python
example extractor now returns 37 effective observations.

Adapter SHA: `1dd637711675366c559ce92c8b6a4d55052e3ae78f7a747bc0aa067807e53c9a`.
Human projection SHA: `244b0596cf88f9ba989a85fab5cb6d570c11f6616fae571e37a27f3fe88d7252`.
Artifact owner is the existing `native-generation-v2/inspection-v1/` under the
learning-check root. Foundation identity remains
`15fa68913bdb2bf395a189df7ab433f6d5b126fc35c46c1dbc8e607ce2182e97`.

All 24 rendered pages in `review.json` were visually inspected at rate one:
six human reference pages, ten LN-rich context pages and eight dense context
pages. Review SHA:
`7063b8d246fe3d635632bd55dc8a08c46d3fcd2aa53bd8bc858baaa72f6a91d9`.
The positive high-confidence human example `human-03f7e300cf02f58f3dcbba66`
marks LN coordination prominent at 167706–169445 ms; its comment identifies
repeated dense multi-lane occupancy with misaligned starts/releases. The negative
high-confidence example `human-9f2c08a0592fa13087c98d5e`, 313004–319158 ms, has
synchronized long holding without independent interior events. It has no
substantive human comment. These calibrate the difference between co-holding
and independent control; neither overlap nor a duration cutoff defines the tag.

On source `bd120339738c6231ab4093f4d19869f9675a2b5eb6053904d47893da7c6f33e8`,
core 95864–103864 ms with two seconds of entry/exit context, O1 has 64 TAP and
3 LN heads. Columns 1/2 hold together at 100872–100984 and column 0 holds alone
at 102998–103110: machine LN-coordination absent/high. R1 has 19 TAP and 51 LN
heads plus two entering holds: present/prominent/high. A witness is column 2
held 100312–100984 while columns 1, 3 and 0 independently release and re-enter;
column 0 taps at 100984 while column 3 remains held to 101096. Independent roles
recur through the core. The scope was centered on its LN-rich supervised window.

On dense source `00126e732bc47a8451ad779c6f82a476482ed4e9d72467d3866cbe91c259e8f6`,
core 158311–164311 ms with two seconds of entry/exit context, O1 has 181 TAP and
1 LN heads: LN coordination absent/high. R1 has 170 TAP and 35 LN heads with a
shorter independent LN episode after dense TAP activity: present/supporting/medium.
The post-hoc rapid-run locator finds seven column-zero taps at 161205, 161238,
161274, 161310, 161345, 161382 and 161417 ms with other-column activity. This is
a localized burden, not a global Jack or whole-chart unplayability verdict.

The same dense core has 105 supplied onsets and 113 source heads, compared with
182 O1 and 205 R1 generated heads. Its maximum one-second H count is 30; all
16 supervised windows have maxima at most 15. Broader data coverage is therefore
a concrete discriminating next step, not an established fix. These two selected
scopes, one generation seed and short-fit checkpoints cannot establish an arm
preference. Other tags, R0 images, larger organization, new human blind comparisons
and independent confirmation remain outstanding. Evaluation: REFINE.

## Corpus-training implementation and input preparation

Clean source `411c8c29abad50a05cf3ceb90f20a20d93a321ed` adds the shared draw plan,
bounded source-index LRU, typed Hydra corpus entrypoint, microbatch accumulation,
safe model/optimizer/RNG/coverage recovery, proper two-class LN diagnostics and
complete-suffix likelihood partitioning. Runtime model distributions are unchanged.
CPU/MPS tests compare unequal-microbatch gradients and bit-exact uninterrupted
versus resumed training; failed publications retain a recoverable boundary and
charge discarded work. All 102 selected tests and 21 package subtests pass in
14.63 s; Hydra `--cfg job` succeeds. No CUDA claim or repository-wide test claim.

Plan preparation consumes the previously pinned TRAIN census/cache, catalog
`e31b7e8f4daa044503ef2b8411bc41727ba371eec804c9462a4608be8112ad28`, allocation
`15175f45e91cf7299a9a30166731bf38ee7361399b346fb692cf68e76de5992a`, and census
`7a6380fd1efab72ffaaa60250a8b2a92869675699d9d108c7dc87708bb5d0fd9`.
It verifies every existing admitted TRAIN row/metadata digest without changing
the allocation. No validation/test payload is used. Preparation takes 3.060 s.
The frozen plan is `artifacts/bounded-typed-continuation/corpus-20260918-v1/plan.json`,
SHA `a6e727d0bbfa414e7d20c742a18d84e8786f155b91629d285e7abf04348cd82f`:
11,563 eligible charts, 3,169 groups, 10,573 draws, 2,000,000 exact onset exposures.

## Experiment Card: bounded-typed-corpus-feasibility-v1

Revision: 1. Acceptance: none. Execution is exploratory under the user's standing
authority for formulation changes and bounded training; no lifecycle acceptance
or quality pass is inferred. This card fixes corpus training and its first
resource/learning readout. A separately pinned development generation/likelihood
screen is required to rank quality and decide between R1/O1.

Question: can the native three-arm setup reach 250,000 common onset exposures
over broader TRAIN coverage on this Mac, with complete endpoint support, finite
gradients and bounded resources? A pass permits continuing the same plan to
1M/2M, not adopting an arm. A failure identifies a concrete resource or learning
problem to fix before extending. The closest baseline is the 16-chart learning
check: 32,768 exposures over 2,048 unique onsets, with native full generation
revealing low O1 LN usage and an unseen dense regime. The encoder/clock/likelihood
family is unchanged. Corpus coverage, window lengths, optimizer schedule and
effective batch change jointly; this is not a causal ablation of one of those.

Fixed source: `411c8c29abad50a05cf3ceb90f20a20d93a321ed`. Use the exact plan SHA
above for O1, R1 and R0, each from scratch with model seed 171. Draw seed 471;
uniform group then uniform chart, horizon 128/256 equiprobable, 12.5% seed-window
stratum, otherwise uniform valid window. Milestones shorten only their final
draw. Actual selected source onsets normalize every joint factor. Default model
128/8/4/rank16 and full-support endpoint budget 8192 with backward recomputation.
AdamW 0.0003, weight decay 0.01, gradient clipping 1, warmup through 32,768 onset
exposures, effective batch 4 intervals, microbatch 2. No auxiliary loss, quality
mask, decoder penalty, audio or LN-intent condition.

Environment: Python 3.10.20, Torch 2.11.0, NumPy 1.26.4, Apple M5/24 GiB,
MPS FP32, one CPU thread. Run arms sequentially O1/R1/R0, never concurrently.
Fresh outputs are `corpus-20260918-v1/o1-250k`, `r1-250k`, `r0-250k` under the
bounded-typed artifact root. Preserve failures and use fresh segments for any
verified recovery. No automatic resume of a parent lacking a finalized runtime
ledger; checkpoint/model/data identities must match. Recovery time and discarded
work count against cumulative compute. The sampler/source plan is never edited.

Command per arm, with `<ARM>`/`<arm>` substituted as above:

```sh
uv run --offline --python 3.10 --extra mps --group dev python -m pulsefield_model.research.bounded_typed_continuation.train_hydra \
  plan_file=artifacts/bounded-typed-continuation/corpus-20260918-v1/plan.json \
  plan_sha256=a6e727d0bbfa414e7d20c742a18d84e8786f155b91629d285e7abf04348cd82f \
  source_cache_dir=../Pulsefield-model/artifacts/oracle-time-continuation/full-cache-v1 \
  output_dir=artifacts/bounded-typed-continuation/corpus-20260918-v1/<arm>-250k \
  model.arm=<ARM> stop_after_checkpoint=250000
```

Resource bounds: 14,400 cumulative training seconds per arm through the whole
2M plan, checked before each optimizer update; report any final update/publication
overrun explicitly. MPS driver/RSS each 6 GiB, allocator ceiling 8 GiB, available
memory at least 2 GiB, swap growth at most 128 MiB. Cache owner at most 64 sources
and 256 MiB conservative charge, with guarded process RSS for live microbatches.
Checkpoint at most 128 MiB, each segment output at most 512 MiB, disk reserve
1 GiB. Check resource pressure after forward/backward and every update start;
checkpoint every 32 updates and every milestone, releasing idle MPS cache there.
Stop immediately on support/nonfinite-gradient/resource failures, preserve the
last durable checkpoint and record the complete failed segment duration.

Primary feasibility gate: all three arms reach exactly 250,000 common onset
exposures within their compute/resource bounds, with zero invalid source targets,
nonfinite updates or silent support truncations, and at least 100,000 unique
source onsets per arm. Report curves for all-class LN NLL/Brier/prevalence and
conditional TAP/LN NLL, complete local factor cost, actual physical/prefix/padded
rows, endpoint factors/candidate pairs and context duration. These training
diagnostics do not replace paired complete-suffix validation or generated quality.
The earlier positive-only random-baseline gate remains failed as originally
declared; this card does not retroactively rename that result a pass.

Positive result: broader training is operationally feasible; inspect fixed
development conditions before deciding further resource allocation. Negative:
preserve and diagnose the exact failure. Ambiguous: finite training costs improve
while generated LN/organization or burden remains weak, or curves still improve
sharply at the budget boundary. None of these proves an intrinsic representation
advantage. One initialization and previously used validation remain development
evidence; second initialization/unused groups and human blind comparisons are
still required for independent confirmation.

## Corpus feasibility result: first O1 resource stop

The revision-1 command starts O1, then stops before starting R1/R0 because its
swap-growth guard fires. It completes 40 updates / 30,295 onset exposures in
102.380 s; the last durable boundary is update 32 / 24,384 exposures. Its
checkpoint SHA is `374ac4c85dfad39b78b9f9dcf71a519b46a4deac6a142338502992589dd5bc01`.
Eight completed updates / 5,911 exposures plus the partial next update are not
recoverable as optimizer progress and must not be counted twice on resume.
Completed coverage is 30,167 unique onsets / 158 charts / 156 groups; checkpoint
coverage is 24,256 / 126 / 124. The feasibility gate has failed in this attempt.

At the stopping backward check, global swap increases 149,487,616 bytes, exceeding
134,217,728; available memory is 6,020,661,248 bytes. Across the sampled run,
maximum active MPS memory is 539,189,504 bytes, driver 1,822,408,704 bytes and
process RSS 3,070,427,136 bytes. At the update-32 cache-release boundary active
memory is 120,041,216 and driver 738,181,120 bytes. RSS subsequently falls while
swap rises. These overlapping counters cannot be added, and global swap does not
identify a process owner. Physical footprint was not recorded. This does not yet
establish an oversized live tensor, allocator leak or graph-cache mechanism.

The condition selector was written and executed after training had stopped
(source timestamps 1789723123/1789723159 versus stop 1789722991); it did not
overlap that run. It pins 24 distinct validation groups: 12 group-uniform ordinary
sources, then two highest-ranked remaining groups for each of LN fraction,
independent endpoint-row count, one-second onset density, chord fraction, longest
onset gap and adjacent eight-second density change. Eligibility is at least 128
suffix onsets and 16 post-seed seconds: 1,632 charts / 435 groups. Selection seed
571, prospectively fixed generation seeds 17/19/23, and uniformly drawn 16-second
core intervals with optional enclosing 64-second contexts. These are development
conditions, not new independent confirmation or source-matching targets.
Condition manifest SHA:
`f0ead07f41111c0413cae9b3f27b3a20c2abbedc2efea1b94f5c2b2707b808eb`;
selector SHA `af8abee08a65ce07370b8fdd5009531a8b71a117cc4b3f0589133d4cd855209d`.
Preparation takes 4.330 s. No generated validation result has been seen.

## Experiment Card: bounded-typed-memory-attribution-v1

Revision: 1. Acceptance: none. Bounded exploratory measurement under standing
user authority, following the repository MPS investigation guide. Fixed source
`411c8c29abad50a05cf3ceb90f20a20d93a321ed`, the exact O1 checkpoint and plan above,
same MPS model/FP32/candidate budget, one CPU thread. No optimizer step or saved
parameter change; backward is only a measured workload.

Compare fresh processes: `fixed-all` repeats the first two draws after the
checkpoint; `variable-all` consumes the next 128 draws in consecutive pairs;
`variable-head` uses those same varying pairs but omits endpoint likelihood and
its backward graph. All perform up to 64 microbatches, clearing gradients before
each, with no intermediate idle-cache release. Thus fixed/variable distinguishes
repetition from new input shapes/content; all/head removes endpoint work. It does
not isolate every shape from semantic content. At every eight steps record macOS
`vmmap -summary` physical footprint alongside synchronized RSS/active/driver/swap
counters. Afterward drop model/data owners, collect garbage and empty MPS cache.

Primary observations are repeated phase-aligned footprint/counter trajectories,
fixed versus variable slope and cleanup response. A variable-only persistent
increase motivates a shape/cache investigation; shared growth motivates lifetime
or host/allocator investigation. Endpoint-only differences narrow ownership but
are not by themselves proof of a backend graph cache. Resource guards are the
unchanged 6 GiB RSS/driver, 8 GiB allocator, 2 GiB available and 128 MiB swap growth.
Each process is capped at 180 measured seconds, checked each step; stop on the
first violation and retain its result. These are diagnostic costs, not exposure
credited to the main comparison. No guard is loosened and no recovery is launched
until this diagnostic is interpreted.

Run `uv run --offline --python 3.10 --extra mps --group dev python
artifacts/bounded-typed-continuation/corpus-20260918-v1/memory_probe.py <mode>`
for the three named modes sequentially, preserving a failed mode before the next
independent control. Script SHA:
`24714e2626b200068d8e9c64359a9d59b118ea50da117e37e8ba8d6a08803500`.
Fresh outputs: `corpus-20260918-v1/memory-probe-v1/<mode>/`. Baseline physical
footprint is unavailable; therefore compare within this instrumented probe and
do not invent an aligned footprint for the stopped main run. Evaluation pending.

## Memory attribution results and execution-device control

All three 64-microbatch probes complete without crossing their recorded guards.
Fixed inputs take 24.629 s: physical footprint changes from 853.9 MiB at step 8
to 865.8 MiB at step 64 and 605.8 MiB after cleanup. Variable inputs take 79.825 s:
footprint grows 3.0/4.2/5.3/6.3/7.3/8.2/8.8/9.6 GiB at steps 8 through 64;
8.5 GiB remains after dropping model/data, garbage collection and idle MPS cache
release. Variable head-only inputs take 38.195 s and grow from 1.6 to 5.4 GiB;
4.3 GiB remains after cleanup. Figures retain `vmmap`'s rounded precision.

The large persistent growth depends on changing inputs, and both the common
encoder/readout path and endpoint path contribute. Nearly constant post-backward
active MPS storage (about 62–67 MB) does not explain the footprint. These controls
support investigating shape-related runtime retention, but do not identify the
exact backend graph/operator cache. They also expose a guard gap: RSS/driver and
global swap bounds can pass while physical footprint becomes too large. Further
Mac training needs physical-footprint observation, not a larger swap allowance.

Memory-attribution Card revision 2 adds two fresh execution-device controls:
same variable-all source draws, checkpoint weights, 64 backward microbatches and
no optimizer, now CPU with one or four threads. Other guards and the 180-second
bound remain unchanged; no model/training objective changes. Compare complete
runtime and footprint against MPS variable-all to decide whether CPU execution
is a simpler practical route before changing GPU operator geometry. New script
`corpus-20260918-v1/cpu_memory_probe.py` is a device/thread-only derivative of the
recorded original; command substitutes it with modes `cpu-1` then `cpu-4`.
Outputs remain distinct under `memory-probe-v1/`. Acceptance remains none.

## CPU control results and corpus Card revision 2

The CPU script SHA is
`5bb0587d018bc90b31bccc0f15411adb89efb0cc33ca0383c187b9b53bba0a9d`.
Both 64-step variable-all controls finish with zero swap growth. One thread
takes 28.275 s, final footprint 733.1 MiB and peak 863.5 MiB; four threads take
25.946 s, final 721.1 MiB and peak 857.2 MiB. Footprints stabilize after the cold
portion. Both retain roughly their final host allocation after cleanup, but that
small retained pool does not show the MPS trajectory's continuing growth.
Minimum available memory is above 10.48 GB in both CPU runs. Result SHAs:

- fixed MPS: `37a2f1dc7be31b5fb97c7678598323403f689823df1ebc1e312d4838b0b700d4`;
- varying MPS: `f54554b96f570a624f8bc5ffce8629e561277d4b63ddd7ffcd0d24c332d003a4`;
- varying head-only MPS: `35984df8dcb89ba104b47bf1d7d3b6db617ae733a0674d79c08ad5fcb777d5b3`;
- CPU one thread: `d28d936b8c9b069f16d42b2d86dd766138ce3333b3b29898515f20a000f943c0`;
- CPU four threads: `8a0de5034bfbe5703feb04cd10ac0c42737545d15eeb1c0472d4b3d3e22a517c`.

Select one-thread CPU for the common training comparison: about 2.8 times faster
than the variable-input MPS control, stable sub-GiB footprint in this workload,
and only a small measured difference from four threads. This decision avoids
changing model geometry before testing adequate training. It does not establish
long-run training memory or universal backend superiority. Keep the probability
heads, finite context, candidate support and sampling plan unchanged.

Clean product source `1693d62ffaca04b2a6127d8e3a72d1988adf441f` adds direct Mach
`TASK_VM_INFO_REV1.phys_footprint` measurement and a 6 GiB guard, and changes the
corpus preset's execution default to CPU. A Darwin test compares the byte counter
with independent `vmmap`; an injected counter verifies stop behavior even when
RSS passes. All 17 affected training/memory tests pass in 8.35 s, including
CPU/MPS exact recovery, and Hydra configuration inspection succeeds. The earlier
102-test run still describes the preceding implementation scope; no new full-suite
claim is made.

The `bounded-typed-corpus-feasibility-v1` Card is now revision 2, proposed with
acceptance none. Replace revision 1's source with the clean source above; replace
MPS execution with CPU/one thread for every arm and add the 6 GiB Mac footprint
guard. Fresh destinations are `o1-cpu-250k`, `r1-cpu-250k`, `r0-cpu-250k` in the
same artifact root. All other data, initialization, optimizer, checkpoints,
sampling, scientific thresholds and resource fields are unchanged. Run the same
command with `device=cpu` and these new outputs, sequential O1/R1/R0. Every arm
starts from scratch; none imports the failed MPS optimizer or its exposures.
Preserve that attempt's 102.380 s and the five diagnostic runtimes as separate
research costs, and include them explicitly in any overall cost claim. This is
a new matched execution comparison, not a free recovery of the failed attempt.
The 14,400-second cap still applies to each CPU arm's cumulative segments, while
equal-compute interpretation must also disclose the discarded development cost.
Evaluation remains pending; no representation or playability result follows from
the execution-device choice.

## Experiment Card: bounded-typed-development-screen-250k-v1

Revision: 1. Acceptance: none. Exploratory development evaluation under standing
user authority, before seeing any generated outputs from the CPU corpus run.
Question: after broader native training, does O1 still show the short-fit LN-use
deficit, and how do R1/O1 compare in full-suffix likelihood, burden and organization
under the same external conditions? R0 remains a different-conditioning practical
reference. None is selected from the expert's prior preference alone.

Fixed source `1693d62ffaca04b2a6127d8e3a72d1988adf441f`; require all three
`*-cpu-250k` runs paused at exactly 250,000 actual source-onset exposures. Verify
each complete final checkpoint against its finalized result digest before loading.
Use the previously frozen 24-condition manifest
`f0ead07f41111c0413cae9b3f27b3a20c2abbedc2efea1b94f5c2b2707b808eb`, including
ordinary/stress strata, fixed 16-second cores and optional 64-second contexts.
These validation groups may have appeared in earlier development; no unused-group
or independent confirmation claim. No model update or annotation mutation.

First score each complete source suffix with 128-onset computation chunks,
including every endpoint, normalizer and the full actual source-onset denominator.
Then generate its complete suffix natively from the minimum seed at temperature
one, for CPU RNG seeds 17/19/23. R1/O1 share R/H/complete-object seed; R0 receives
only its permitted R/physical seed. Use full future endpoint support, candidate
budget 8192, no quality masks or penalties. O1 endpoint log probabilities may
remain explicitly unscored during sampling; this does not change the sampler.
Verify all outputs independently and export/reparse exact physical rows. First
source/seed17 for each arm must rebuild its raw cache after candidate 512 and
match exact state, RNG and learned buffers before continuing.

Required mechanical gate: all 216 generated charts and 72 complete-suffix scores
finish, with zero occupancy/coverage/endpoint/terminal/export failures. One failure
stops the run rather than being averaged into quality. Report ordinary and stress
groups separately: macro and pooled source-onset NLL, paired R1/O1 group differences,
proper binary LN scores, chord distributions, LN births/durations, held concurrency,
independent release/attack relations, rapid same-lane pair frequency and run length,
and cross-seed variation. Below-40ms counts are descriptive locators, not new
playability or semantic thresholds. Source metrics describe conditions and do not
require generated arrangements to reproduce their LN/chord proportions.

No likelihood-only winner or synthetic semantic score is a quality pass. Retained
LN variety with lower localized burden is a promising development signal; reduced
burden through broad LN/organization loss is negative. Mixed or uncertain results
remain unranked. The fixed cores and a smaller sustained-context subset must then
receive time-proportional Foundation/human-calibrated inspection. New blinded human
comparisons and another initialization/unused groups remain independent follow-ups.

Execute sequentially after the three training arms stop; CPU one thread, same
6 GiB footprint/RSS, 2 GiB available, 128 MiB swap-growth guard. Bound the complete
screen to 3,600 seconds and 2 GiB outputs, individual checkpoints to 128 MiB, with
1 GiB disk reserve. Check resources every scoring chunk and 128 generation
candidates; retain raw checkpoints every 512 candidates and complete chart results.
Fresh output `corpus-20260918-v1/development-screen-250k-v1/`; stop on any guard and
preserve incomplete evidence. No automatic overwrite or quality-selective retry.

Command:
`uv run --offline --python 3.10 --extra mps --group dev python artifacts/bounded-typed-continuation/corpus-20260918-v1/development_screen.py`.
Script SHA `30989d8da79e16e87b1f69b00fed693aeff8a84feef7a5bd5e1793f31640d199`;
its reused descriptive helper is the pinned earlier generation script SHA
`5b6f4286fa82d7ff9bee76284268fa761d0a099d127200f8ff6156aa174ea8c4`.
The script passes syntax compilation; actual execution and checkpoint identities
remain pending. Evaluation/Decision pending.

## Corpus feasibility revision-2 result: all arms at 250k

All three CPU arms pause successfully at exactly 250,000 source-onset exposures
after 333 optimizer updates. Shared draw/cursor, physical/prefix/padded-row ledgers,
context spans and exact coverage bitmaps match; configurations differ only in arm
and output directory. Each covers 243,346 distinct onsets, 1,227 charts and 1,081
groups. All loss/gradient/support/resource checks pass. The bounded feasibility
gate is met; generated quality and arm choice remain unresolved.

| Arm | Complete segment seconds | Peak task footprint bytes | Checkpoint SHA-256 |
| --- | --- | --- | --- |
| O1 | 276.129 | 918095456 | `f3e3b264fa6eacf9f3a5949afa0f1b6c81bad3571ae07cc852a29a371b98f366` |
| R1 | 188.842 | 923387536 | `7c83a4920aed8d07aebae4241ca6067d066dd0ecf7fa3f309ffa6da9f804dcbf` |
| R0 | 176.337 | 924599952 | `a3b88f7f211bb1642e0acb56833e496027c86ee3eb826f5d43e87cf46b617671` |

Swap growth is zero throughout. Each processes 262,671 physical target rows,
338,544 prefix rows and 755,854 padded rows. O1 supervises 64,523 LN endpoints,
129,046 order factors and 91,619,349 full-support candidate pairs with recomputed
backward. The CPU run's first update-32 local cost agrees with the earlier MPS
value within 0.0000002 nats/onset; no model geometry was changed to obtain the
runtime/memory improvement.

First-32 to last-32 update summaries indicate learning but use different source
windows: local O1 factor cost 4.713912 to 2.397143, R1 3.502996 to 2.425545, R0
3.583767 to 2.547641 nats/onset. These are not full-suffix cross-arm scores.
Proper LN binary NLL moves 0.294279 to 0.180289 for O1, 0.308116 to 0.185841 for
R1 and 0.310573 to 0.186890 for R0. O1's final-window conditional TAP/LN NLL is
0.322663, with predicted LN mass 19.081% against observed 18.377%; analogous R1
NLL is 0.334705 and R0 0.330376. Feasible diagnostic cohorts can differ slightly
because each factorization has different already-known obligations; do not treat
them as a common joint code-length comparison.

The 1,330 selected intervals now include target densities up to 38 H/second,
versus 15 in the small check. Only 20 intervals reach at least 24 H/second, so
the dense regime remains rare. Among intervals using the full 512-row history
envelope, median span is 111.000 / 64.579 / 46.315 / 39.163 seconds for target
peak-density bins [0,8), [8,16), [16,24), [24,infinity), with 16/232/101/14
intervals respectively. These are paired training-window statistics, not a
corpus-time distribution or proof of learned multi-scale organization.

Readout `corpus-20260918-v1/training-readout-250k.json`, SHA
`1b39b734e38565ce0d927b9c776791da186faba15b553d4644228a504044fdc4`, verifies
all checkpoint/ledger/config identities and records the discarded MPS attempt
and diagnostic runtime separately. The development-screen Card's three checkpoint
inputs are now the exact identities in the table. Proceed with its frozen
24-group/three-seed conditions. Evaluation: REFINE; no generated-quality verdict.

### Development-screen descriptive and visual readout procedure

Before the screen completes, freeze its descriptive reader and initial visual
subset. `screen_readout.py` SHA
`f009fe9bf9af496a2ccbf85846b88e6d35f03b89f5447be6a36479cc1bdfdeed` requires all
216 generations / 72 likelihoods. It reports ordinary/stress/all groups separately,
paired O1-minus-R1 macro NLL with 10,000 group bootstrap resamples at seed 971,
pooled NLL, type scores and descriptive generated metrics. Seed outcomes remain
nested within source groups. No semantic or playability label is inferred from
these counts, and ordinary/stress pooling is not a population weighting claim.

First visual subset: condition indices 12 (first LN-rich source) and 16 (first
dense source), O1/R1 at seed 17. Use their prospectively frozen 16-second cores
with two seconds of entering/exiting context, clipped only at the true chart
end. This is a bounded stress inspection, not the complete quality benchmark or
a method-blinded human vote. Inspect every rendered page and concrete action
relationships before assigning any machine hypotheses; keep other labels and
unseen scopes unreviewed. `render_screen.py` SHA
`4c7abd521911b7af41276a54c53b5a096904f978b2c92aeb378e9d490701fa06` uses the
canonical beatmap-lens renderer/action views and records all source/image hashes.
Output `corpus-20260918-v1/inspection-250k-v1/` must be fresh. Both scripts pass
syntax compilation and require a complete successful generation screen.

The eight canonical human workflow documents, frozen Foundation bytes, Inspector
comment projection and domain owner still match the previous 37-observation read.
Calibration preflight SHA
`eaded64781ce530b8a60723dc183ff241c6d296b178ad00c87f4c3fe48de7b2e` records that
verification. Reuse the actually inspected high-confidence LN positive and
negative above; no cached machine rationale becomes a human comment. Rendering
and these measurements do not change gold or Foundation.

## Development-screen result at 250k

The complete screen finishes in 688.168 s: all 216 generated charts pass independent
coverage/occupancy/endpoints/closure and export/reparse checks, and all 72 full-suffix
scores finish. The three actual raw-cache recoveries match state/RNG/buffers.
Peak observed task footprint is 228,508,608 bytes, with zero swap growth.
Descriptive readout SHA:
`943808c0b27fa6c8c32f99eca5422f57b1f077781f7731cd7b4a61e8e2830a11`.

Ordinary-group macro suffix NLL is O1 2.189516 versus R1 2.196794; the paired
O1-minus-R1 difference is -0.007278, group-bootstrap 95% interval [-0.041318,0.025698].
Stress-group macro NLL is O1 2.485951 versus R1 2.459914, difference 0.026037,
interval [-0.042325,0.092383]. Combined macro difference is 0.009380,
interval [-0.028848,0.047057]. These observations do not establish either an
advantage or equivalence. Pooled full-suffix NLL is 2.291069/2.268465 for O1/R1.
R0's different-condition macro/pooled costs are 2.456342/2.384288, not a same-task
likelihood ranking against typed arms.

Across 72 outputs per arm, new LN fractions are O1 3,316/127,933 = 2.592%, R1
16,071/167,725 = 9.582%, R0 6,666/157,030 = 4.245%. Ordinary/stress O1 fractions
are 3.442%/1.893%, versus R1 8.774%/10.250%. O1's per-chart range is 0.465–10.423%,
R1 3.635–35.459%. Below-40ms same-lane pairs per thousand generated heads are
1.298/3.076/3.324 for O1/R1/R0. O1's lower recurrence count accompanies more
single-note TAP output, so it is not a playability improvement by itself. Full-
suffix source-fitting similarity has not removed the free-running mode difference.

All 32 generated context pages and four additional human-comparison pages were
actually inspected. Render manifest SHA
`42e6e90ab7846c6655a509f2895a313b87b1df2ac1706c737876814e67357f67`;
machine review SHA `507bca0ba79a4c338752b2066e69f9b6c95b70346907e78a01e7c4265b98be9f`.
In the LN-rich `0447fb187bc3` core 16958.524–32958.524 ms, O1 has 214 TAP and no
LN heads; R1 has 225 TAP / 82 LN heads and three entering holds. Original source
has 259 LN heads at the same 197 onsets. O1 LN coordination is absent/high.
R1 has repeated definite independent-control passages interleaved with TAP/chord
runs: present/supporting, medium confidence at the whole 16-second scope. A
witness is column 3 held 17005–17530 while other columns exchange shorter holds
and column 2 taps at 17380/17455. At 29605 columns 1/2/3 start holds ending
29980/29755/29830, followed by other-lane re-entry before the final closure.

In dense `e6b273f7877d`, core 101905.910–117905.910 ms, source has 382 heads
(two LNs) at 293 onsets. O1 has 323 TAP-only heads; R1 has 381 TAP / 18 LN heads.
The two outputs have one/two below-40ms same-lane pairs, longest run two in both,
so this inspected core has no prolonged rapid run by that locator. Conditions
differ from the earlier short-fit dense failure; this does not prove that fixed
regression case was repaired. O1 LN coordination is absent/high. R1 has a short
independent figure: column 0 holds 112931–113101, columns 2/3 start at 113044 and
end at 113158/113215, with independent intervening taps. Machine judgment is
present/supporting with medium confidence; isolated single-anchor holds are not
positive evidence by themselves. No whole-chart preference follows.

The additional opened High/supporting human example is
`human-2208bdfda699add6503ca166`, source `98357fbf0c617bae8d1783950fe0c3a1b97c5cf9366ebce920f9870a8424d4c9`,
scope 87509–93156 with context 86097–94568 ms. It places an independent LN island
inside a larger TAP section, with eight LN and 86 TAP heads. It has no substantive
human comment. The generated dense island is briefer, limiting confidence; no
percentage/duration threshold is inferred. All other tags, ordinary images,
larger 64-second organization, other seeds and R0 images remain unreviewed.

On the first LN-rich source, teacher-forced conditional TAP/LN probability at
true head locations averages 75.969% LN for O1 and 77.120% for R1, while all
observed heads are LN. Native whole-chart LN fractions for seeds 17/19/23 are
2.167/2.327/2.743% for O1 and 30.188/21.647/22.058% for R1. These are different
cohorts/conditioning states and cannot be subtracted as calibration error. They
motivate measuring the complete head-policy mass on aligned source/generated
histories before claiming that endpoints alone learned or that context is too
short. Evaluation: REFINE; no arm selected and no quality completion.

## Experiment Card: bounded-typed-policy-drift-250k-v1

Revision 1; proposed, acceptance none, exploratory under standing user authority.
Fixed source `1693d62ffaca04b2a6127d8e3a72d1988adf441f`, exact 250k O1/R1 checkpoints,
condition index 12 and seed 17. No model update or generation change. Compute
expected LN-head count and expected total-head count by summing each native legal
joint distribution. Score all source-history H queries in finite batches and
instrument a native free-running replay at the same R/H points. Require that the
first head distributions agree within 0.00002 in both expected counts and that
all sampled rows exactly reproduce the completed screen outputs. A failure stops
the diagnostic and reopens implementation consistency before further training.

Report ratios of summed expectations in 32-onset blocks, alongside physical-row
counts and timestamps. This distinguishes a policy shift under generated history
from mere unusual realized LN samples, and locates whether it begins before raw
context eviction. It does not uniquely separate endpoint errors, lane allocation,
training insufficiency or latent style retention. No counterfactual source action
is forced onto an incompatible generated occupancy state.

CPU one thread, at most 120 seconds, existing 6 GiB footprint/RSS, 2 GiB available,
128 MiB swap-growth bounds. Fresh output `corpus-20260918-v1/policy-drift-250k-v1/`.
Run `uv run --offline --python 3.10 --extra mps --group dev python
artifacts/bounded-typed-continuation/corpus-20260918-v1/policy_drift.py`;
script SHA `3a4a5dcbbfe20d97dee3148a6a690749ced7bcb17926d52b8d7ce7d26adf8d88`.
Syntax check passes; measurement and interpretation pending.

## Policy-drift result

The probe completes in 6.484 s, SHA
`b10c1131b5381812372211dd4026714544d3d5c8ec7479a3d97e6caa26b6a50c`.
Both native histories exactly reproduce their saved seed-17 output rows. The
first O1 query has expected total heads 1 and expected LN heads 0.9328866 on both
source and generated paths. R1's first expected LN count is 0.4285475 versus
0.4285476; the required tolerance passes. This is a within-model consistency
check, not a claim that the two models have the same first distribution.

| 32-onset block start | O1 source-history LN mass | O1 generated-history LN mass | R1 source-history LN mass | R1 generated-history LN mass |
| --- | --- | --- | --- | --- |
| 0 | 67.427% | 20.177% | 72.315% | 64.110% |
| 32 | 68.009% | 5.164% | 74.763% | 62.207% |
| 64 | 77.079% | 2.125% | 75.498% | 50.770% |
| 96 | 75.873% | 1.344% | 75.906% | 66.145% |

Mass is summed expected LN heads divided by summed expected total heads, over
every legal joint choice; it is not probability conditional on a true source
head location and not a realized LN fraction. The rows align the same supplied
R/H times. O1 generated physical-row counts at these block starts are 30,63,96,128.
Its first seed action leaves the 511-token raw context only at suffix onset 477;
R1 reaches that point at onset 418. O1's policy shift therefore begins while its
entire seed still fits in the raw receptive field. Literal context eviction is
not a sufficient explanation for this case. This does not prove that learned
use of longer history is adequate or identify the exact feedback mechanism.

R1 also deviates from source-history predictions and later visits low-LN phases,
but its trajectory retains substantial LN mass through several later episodes.
Expected-policy mass shows the difference is not merely an unusually low realized
sample count. Endpoint errors, lane choices, finite-data learning and retention
of style remain live contributors; none is isolated by this observational trace.
`policy-mass.png` and `.svg` visualize the complete block trajectories. Increasing
the raw context alone is not the next intervention. Evaluation: REFINE.

## Experiment Card: bounded-typed-corpus-exposure-v1

Revision 1, proposed, acceptance none. Continue the authorized paired training
trajectory to its predeclared 1,000,000-onset checkpoint, keeping source, model,
optimizer, data draws and probabilities unchanged. Question: does additional
native exposure improve closed-loop LN/organization coverage as well as source
likelihood, or does the 250k policy gap persist? The sole scientific intervention
is additional supervised exposure along the existing frozen plan; no coarse LN
intent, source-future action labels, decoder penalties or longer context are added.

Fixed clean source `1693d62ffaca04b2a6127d8e3a72d1988adf441f`. Resume exact O1/R1/R0
250k checkpoints and SHAs recorded above, using the same plan SHA
`a6e727d0bbfa414e7d20c742a18d84e8786f155b91629d285e7abf04348cd82f`. Run CPU one
thread sequentially O1/R1/R0. Each fresh segment is named `<arm>-cpu-1000k` under
the same corpus root; parent `*-cpu-250k` outputs remain unchanged. Verify parent
journal boundaries, configurations and measured compute before continuing. The
four-hour per-arm cap includes all CPU segments; record the separate discarded
MPS/development costs rather than presenting them as free recovered exposure.
All existing footprint/RSS/swap/available-memory/checkpoint/output guards persist.

Command per arm:

```sh
uv run --offline --python 3.10 --extra mps --group dev python -m pulsefield_model.research.bounded_typed_continuation.train_hydra \
  plan_file=artifacts/bounded-typed-continuation/corpus-20260918-v1/plan.json \
  plan_sha256=a6e727d0bbfa414e7d20c742a18d84e8786f155b91629d285e7abf04348cd82f \
  source_cache_dir=../Pulsefield-model/artifacts/oracle-time-continuation/full-cache-v1 \
  resume_from=artifacts/bounded-typed-continuation/corpus-20260918-v1/<arm>-cpu-250k/checkpoint.pt \
  output_dir=artifacts/bounded-typed-continuation/corpus-20260918-v1/<arm>-cpu-1000k \
  model.arm=<ARM> device=cpu stop_after_checkpoint=1000000
```

Operational gate: all arms reach exactly 1M exposures, with identical paired
draw/coverage accounting and zero support/nonfinite/resource failures; stop and
retain a failed segment rather than silently changing the plan. Repeat the same
24 development groups / three seeds / complete-suffix scoring and fixed visual
cores at the new checkpoint, pinning the stage-specific driver before execution.
No validation-selection changes are permitted in that learning-curve comparison.
Report paired changes in macro/pooled NLL and mode/burden diagnostics, plus the
same history-policy trace when it changes the interpretation.

Continuing to the planned 2M checkpoint remains reasonable if fitting is still
improving and no mechanical/resource regression occurs. A decrease in NLL without
LN/organization improvement is not a quality pass; persistent degeneration requires
reconsidering learning/conditioning/factorization rather than declaring success
from lower burden. The original quality criteria still need sustained structure,
independent LN control where present, larger-context inspection, another model
initialization and independent confirmation. This card selects no winner.

## Paired exposure result: one million

All three arms reach exactly 1,000,000 onset exposures / 1,320 cumulative updates.
Each has 917,737 distinct onsets, 3,933 charts and 2,576 groups; paired interval,
row/prefix/padding/context and coverage ledgers match. Recovery discards no updates.
The resumed segments take 815.205/555.947/521.552 s for O1/R1/R0; cumulative CPU
training time is 1091.334/744.789/697.889 s. All remain below their four-hour caps,
with zero swap growth and segment peak task footprints 890095224/884262448/874366512
bytes. No support/nonfinite/mechanical training failure occurs.

Checkpoint identities:

- O1: `1ac6589506a03d155a4bf652cf3726a3c644fe03dfccb01455dd3cd3439e4439`;
- R1: `f9f6f22258c79477a0478b47f4fc298e25a7e8c71d9f3dcb6502e066bb682909`;
- R0: `181a9554d9b233bc9d79149198e69d43a28bdc4cda8aad4e1cefd26f1fb38533`.

Final 32-update local factor costs are 2.194639/2.195729/2.306777 nats/onset,
versus 2.404629/2.425383/2.522637 in the first 32 updates of these resumed segments.
Those are different windows, not held-out arm comparisons. O1 head cost is
1.925255 and endpoint cost 0.882677 per LN in the final window. Conditional TAP/LN
NLL is 0.291686/0.303477/0.310617. Peak target density still reaches 38 H/s;
91 of 5,275 selected intervals reach at least 24 H/s. Readout SHA
`10d9143a4ab0f7a5fe38edd5af2e01be40c1eb9404e36afbde2557036043bcd4`, script SHA
`ae2dc6b772c42b316f703a65a669ed162e721df4ba9f4ff3991ce72f86c5a549`.
Learning/resource gates remain feasible; native generation evidence is pending.

## Development-screen and policy-probe Card revisions for 1M

The development-screen Card is revision 2, proposed, acceptance none. Retain the
same clean source, 24 conditions, ordinary/stress weighting, seeds 17/19/23,
full-suffix score, mechanics, fixed 16-second cores, limits and interpretation.
Only replace its checkpoint inputs by the exact 1M identities above and use fresh
`development-screen-1000k-v1/` / `inspection-1000k-v1/` outputs. The stage-specific
drivers are textually identical except milestone and path substitutions:

- `development_screen_1000k.py`: `5d3b5212971d1119f864fad2d431e72c5f0fdde23ecbce97ae79d358c0af00f6`;
- `screen_readout_1000k.py`: `e6797e3820784e2f5faadac9d3b1438eca3c7df36d56fa55f0c50c263a0448d5`;
- `render_screen_1000k.py`: `61875ea0ee95df3a06e506933c9bc958c8b38dae98dd21ba43dd7542aaeb571b`.

Likewise policy-drift Card revision 2 repeats the same source/seed/expectation
definition and reproduction checks at 1M, after its generation screen succeeds.
Use `policy_drift_1000k.py`, SHA
`0a8fa84c3d82667491a439402784256943702e369b6153ad88357aa1dc992f8e`, fresh output
`policy-drift-1000k-v1/`, unchanged 120-second and memory limits. All drivers pass
syntax compilation. Invoke each through the same `uv run --offline --python 3.10
--extra mps --group dev python` prefix; no source/model edit or new condition
selection is introduced. Quality decisions remain pending actual outputs.

## One-million development result and diagnostic correction

The 1M screen completes all 216 generations and 72 suffix scores in 696.709 s,
with every mechanical/export/reparse check passing. Peak footprint is 231736280
bytes; swap growth remains zero. Readout SHA
`ea86a71c1b047a996367943b1db230e5134b1b11dc6eb9a75841fcd787399389`.
O1/R1 pooled full-suffix NLL improves from 2.291069/2.268465 at 250k to
2.051118/2.054385. Ordinary macro scores are 2.006055/2.005809 and stress macro
scores 2.245701/2.271089. Overall paired macro difference is -0.012571 with
group-bootstrap 95% interval [-0.041833,0.013609]; no stable arm winner.

O1's LN share rises to 27763/201873 = 13.753%, versus R1 18617/230635 = 8.072%
and R0 14317/252012 = 5.681%. O1's chart range is now 0.321–74.412%, so its earlier
broad lack of high-LN outputs is not invariant to training exposure. However,
all three generate many more four-key attack groups: O1 17684, R1 19354, R0
19047, versus 306 in the matched sources counted once per generation seed.
At 250k the corresponding generated counts were 67/507/129. Fewer below-40ms
pairs therefore cannot establish a lower overall recurrence burden.

A post-hoc factual locator finds consecutive identical nonempty attack groups,
retaining intervening release-only rows and reporting actual gaps without a
semantic threshold. Readout SHA
`4150db9021392bc1e92acb6fd18d12ca465036c0d46a3f5aab68ff7c65531685`.
Examples include 67 O1 quad attacks at 76419–82787 ms (gaps 70–142 ms) on ordinary
source `27ac9470c9a4`, seed 23, and 59 R1 quads at 105794–116340 ms (181–182 ms)
on ordinary source `c1798e61528c`, seed 17. Both source charts have no suffix quad
attack rows. These are locators for sustained repetition and mode concentration,
not universal difficulty verdicts or automatic semantic labels. The 1M fixed
core images are rendered but remain visually unreviewed at this record.

The original 1M policy probe stops at its alignment assertion after exactly
reproducing the O1 rows. Inspection of `Rollout.step` identifies a measurement
omission: legal-singleton H decisions execute without invoking the neural query
hook. They were missing from the captured probability sequence. This does not
invalidate generated rows or the normalized deterministic probability of one.
The original output is retained with a failure record; R1 was not attempted.

Policy-drift Card revision 3 repairs only this readout: for a missing H callback,
require exactly one feasible decision and add its exact head/LN expectation.
Retain source, checkpoints, source/seed, limits and native reproduction checks.
Use fresh `policy-drift-1000k-v2/`, script `policy_drift_1000k_v2.py`, SHA
`9ead461be2ebdfef90abdca451a0723530c30f47dbb46b3df183cb8908e73c24`.
Syntax compilation passes; no model or sampler changes. Acceptance remains none.

## Corpus-exposure Card revision 2: complete the two-million checkpoint

Continue all three arms from their exact 1M checkpoints to the existing plan's
2M endpoint on unchanged source `1693d62ffaca04b2a6127d8e3a72d1988adf441f`.
The held-out fitting curve is still improving materially, resource limits remain
comfortable, and generation changes with exposure; neither early O1 rejection
nor quality completion is supported. Four-key repetition is an explicit quality
concern to carry into the final screen, not something to hide with a decoder rule.

This is corpus-exposure Card revision 2, proposed, acceptance none. All protected
scientific, sampling, optimizer, device, memory and cumulative four-hour-per-arm
fields remain unchanged. Resume `<arm>-cpu-1000k/checkpoint.pt` with its pinned
digest above; use fresh `<arm>-cpu-2000k`, set `stop_after_checkpoint=2000000`,
and execute sequential O1/R1/R0 with the same command prefix/plan/cache settings.
Parents remain immutable. The plan ends at 2M, so successful runner status is
`completed`, not `paused`. Preserve any failure and stop the sequential chain.

At 2M, repeat the fixed development conditions and inspect LN control, sustained
repetition and larger organization. Lower NLL, more LNs or recognizable Jack
structure cannot alone meet the playable-quality goal. If generation remains
concentrated in undesirable recurring states, reconsider the common learning or
conditioning setup; do not redefine success as mechanical legality. Source fitting
and generated organization remain separate decisions. No implementation change
or new external oracle condition is introduced by this continuation.


## Paired exposure result: two million

All three arms completed the immutable plan at exactly 2,000,000 source-onset
exposures and 2,645 cumulative updates. Each covers 1,697,938 distinct onsets,
6,315 charts and 3,078 groups. Interval, physical-row, prefix, padding, context
and coverage ledgers match; configurations match after removing arm and segment
paths. No parent updates were discarded, and no resource/support/nonfinite
failure occurred. Cumulative CPU training seconds are O1 2174.555, R1 1501.429
and R0 1408.085. The final segments take 1083.221/756.640/710.195 seconds.
Peak segment footprints are 872203872/884426360/880690784 bytes with zero swap
growth. These are training costs; evaluation and earlier MPS diagnostics remain
separate. All costs stay below the declared four-hour-per-arm training cap.

Checkpoint identities:

- O1: `759a4e6212e1c47b0039a457d9617bdfa957a7989e31f3715b6f70ff0b18d3b1`;
- R1: `302fae523cf1fe36afff463fe28f20739a2eb40cab5064d0941a588d8f70c564`;
- R0: `fbe7a80936d089779a5cae48e48c3c76f36ef2062b70027c963f2a21cf9ae2d1`;

The last 32-update local factor costs are 2.041436/2.025580/2.130897 nats/onset.
These are training-window summaries, not complete-suffix rankings. O1 endpoint
cost is 0.807911 nats/LN; conditional TAP/LN NLL is 0.275195/0.275957/0.289180.
All 10,573 draws are accounted for; 165 reach at least 24 H/s and the maximum is
38 H/s. Median full 512-row-envelope duration is 103.268/65.625/46.312/36.671
seconds in peak-density bins below 8, 8–16, 16–24, and at least 24 H/s. This
summarizes sampled training windows, not a uniform corpus-time distribution.
Training readout SHA `9b25bbe8453eb2ed92b97d69b24f8977c666bb092112f16b873a2e68bffe4dd4`.

## Development-screen Card revision 3: fixed two-million comparison

Proposed, acceptance none. Repeat the same 24 development groups, seeds
17/19/23, full-suffix normalized NLL, mechanical/export/reparse checks, resource
bounds, fixed 16-second visual cores and interpretation from revision 2. The only
scientific input change is the exact 2M checkpoints above. Source remains clean
`1693d62ffaca04b2a6127d8e3a72d1988adf441f`; conditions SHA remains
`f0ead07f41111c0413cae9b3f27b3a20c2abbedc2efea1b94f5c2b2707b808eb`.
Use fresh `development-screen-2000k-v1/` and `inspection-2000k-v1/`. The runner
status guard changes from paused to completed because this is the plan endpoint.
No model, sampler, candidate support or validation selection changes are made.

Stage-specific artifact driver SHAs:

- `training_readout_2000k.py`: `54010b96622f33510acf9948314ad4c12a2963361985109d022fb1ccb0d3ef7d`;
- `development_screen_2000k.py`: `61c1a330b507b35416919d3ec7cf2504112f50313530773fdc92918257d2c2d2`;
- `screen_readout_2000k.py`: `73a327f6249d7a22d21be7fc0a38c644802a00a960c358320a19594aa7e537cb`;
- `render_screen_2000k.py`: `8fc5836ffe09a11d20514243172dc3ad30d406da489e664f60c6a351ad63eb1a`;
- `policy_drift_2000k.py`: `1da2bdc6897eb5b2bcb49887b377b9237656242cc683144eb038a0bada670f85`;

All drivers pass syntax compilation. Run the development driver, then its
readout/rendering and policy driver through `uv run --offline --python 3.10
--extra mps --group dev python`, on CPU one thread. Screen limit remains one
hour, 6 GiB physical footprint, existing RSS/swap/available-memory guards and
2 GiB output cap. Preserve failures and never overwrite previous outputs.

Policy-drift Card revision 4 uses the same source index 12, seed 17, exact
expectation definition and native-row reproduction checks at 2M. It inherits
the deterministic-singleton H capture correction of revision 3 and writes
fresh `policy-drift-2000k-v1/`; its 120-second/resource limits remain unchanged.
Its checkpoint substitution is prospective; no result is implied.

This is a development learning-curve comparison. Neither lower NLL nor more LN
heads establishes sustained playable organization. Inspect repeated four-key
attacks and independent LN-control episodes explicitly. Independent initialization,
unused-group confirmation and broader organization remain outstanding.


## One-million visual review and repaired policy readout

All 32 fixed-core images have now been inspected at 1x: eight context pages for
each O1/R1 output on the two preselected stress conditions. The same frozen
Foundation and three previously inspected High human LN comparisons apply.
Eight canonical workflow documents and the Foundation file were revalidated
unchanged. No human labels or comments were edited. Scoped machine review SHA
`37e1e3c87ad1e40a65dcbe7d99633d360d2059b55750b8d2da10366ef56d1aa7`, under
`inspection-1000k-v1/review.json`.

For the LN-rich core 16958.524–32958.524 ms, O1 contains 151 LN heads and 76 taps;
independent staggered holds/releases repeatedly organize the core. The machine
assessment is LN coordination present/prominent, High. For example column0
holds 20605–21280 while columns1/2 release and re-enter independently and column3
starts at20755; the pattern of different roles recurs later. R1 contains 24 LN
heads and337 taps, predominantly TAP/chord motion with a definite independent
control island around28330–28930. Assessment is present/supporting, Medium at
the complete16s scope. The presence judgment is clearer than its salience.

For the dense core 101905.910–117905.910 ms, O1 has206 LN heads/129 taps, with
repeated independent control through much of the core: present/prominent, High.
R1 has only3 LN heads/481 taps: a synchronized columns2/3 pair106510–106567 and
one column0 anchor107987–108669. No independent multi-LN organization is present:
absent, High. These are recognizable LN structures, not a universal playability
pass. In particular O1's dense LN control may impose substantial burden. Other
tags, ordinary cores, R0, other seeds and larger64s organization remain unreviewed
by this fixed-core assessment. The early250k O1 absence therefore does not persist.

The corrected policy probe completes in6.871 seconds, reproducing both native
outputs exactly and matching the first source/generated-history query. Readout
`policy-drift-1000k-v2/readout.json`, SHA
`a9e1b7a978f5dab5510f3b837dea7b3246061ed7814d345f4cc9b24a236d5c3f`.
For successive32-onset blocks starting0/32/64/96, O1 source-history expected LN
mass is84.686/89.538/96.136/95.873%; generated-history mass is75.874/78.971/54.560/
39.242%. R1 source values are83.163/85.139/89.955/88.963%, generated values71.239/
36.307/25.471/1.837%. The definition remains expected LN heads divided by expected
total heads over every legal joint choice. O1 retains substantially more LN mass
than at250k on this source/seed, whereas R1 now drifts more. This is one conditional
trajectory, not a causal or universal arm advantage.

All ten post-hoc longest-quad context pages were also inspected. O1's67 consecutive
quad attacks span6368ms with70–142ms gaps; R1's59 span10546ms with181–182ms gaps.
Both have Jack organization present/prominent, High: repeated full-group attacks
organize the complete core and remain dominant in adjacent context. Neither has
LN coordination. The R1 final partial context page covers only47ms, not a2500ms
rest. This confirms sustained recurrence but does not make every such episode
unplayable. Their burdens differ materially; the broader concern is excessive
concentration across generated outputs, not the existence of a valid Jack label.
Other semantic dimensions remain unreviewed. Evaluation remains REFINE.


The quad inspection record is `quad-inspection-1000k-v1/review.json`, SHA
`aaa952b67d6f7bf266c2af2ee5fe40f5b238b286cfd5879b73c9cc6fae9c9061`.
It uses the canonical tag ID `jack-organization` and preserves all ten inspected
image hashes, scopes, playback rate and generated-chart identities.

## Experiment Card: bounded-typed-quad-policy-v1

Revision1, proposed, acceptance none. Behavior-neutral diagnostic extending the
existing expected-policy readout: determine whether the long quad runs reflect
concentrated learned probabilities under generated history or primarily unusual
sample outcomes. This can change the next research question toward conditional
policy calibration and state-distribution drift; it cannot by itself establish
exposure bias, insufficient context or missing style as the unique cause.

Use clean source `1693d62ffaca04b2a6127d8e3a72d1988adf441f`, the same pinned screen
conditions and existing250k/1M/2M checkpoints. Select the two already inspected
1M longest-quad locators before reading their2M outputs: source `27ac9470c9a40a6c7f968ae5516fea21c544a26305edd2ab27feea521152844f`, seed23,
core[76419,82788)ms; source `c1798e61528cf4a5b618efa5d2bb9effe0b23a72ea7a2db58f2046d731d5395b`, seed17,
core[105794,116341)ms. Cross both O1/R1 arms and all three checkpoints on these
fixed sources/seeds:12 native replays. This is post-hoc development diagnosis,
not an independent quality comparison or an equal-compute selection.

At every suffix H, marginalize the same normalized legal choice distribution to
expected head/LN counts and probabilities of1/2/3/4 heads. Include deterministic
singleton choices exactly. Compare source-history and generated-history means
for the full suffix, fixed locator core and consecutive32-onset blocks. Include
actual quad fractions separately. Require complete reproduced rows to equal the
previous native output exactly, aligned H indices, and initial expectations to
match within2e-5 before histories diverge. Stop on any mismatch or resource failure.

Interpretation: high generated-history quad probability in a realized quad run,
with low source-history probability at identical external times, supports policy
concentration under divergent histories, not merely rare draws. If expected mass
is low despite realized repetition, reassess sampling variance and measurement.
If source and generated mass are both high, source-conditioned calibration/model
fit remains a leading issue. Results may be mixed across checkpoints and sources;
no threshold automatically labels an output unplayable and no decoder change is
part of this probe. The closest analogue is the existing paired LN-mass diagnostic;
there is no novelty claim or new scientific intervention.

Driver `quad_policy_probe.py`, SHA
`61cc3b0de40d37077f045481e1dbe8bb6e3ea9ce4dde78dccd0e586f7f723bc5`, passes
syntax compilation. Run through `uv run --offline --python 3.10 --extra mps --group
dev python` after the2M screen completes. CPU one thread; ten-minute total bound,
6GiB physical footprint and existing RSS/swap/available-memory limits. Fresh
`quad-policy-probe-v1/`, preserve failure, no overwrite/resume. This writes only
local derivative diagnostics and leaves source/model/sampler and checkpoints
unchanged. Human/Foundation records remain read-only. Evaluation pending.


## Two-million development evaluation

All 216 generations and 72 complete-suffix scores finish in 705.152 seconds. All
mechanical/export/reparse checks pass, including the three selected exact cache
recoveries. Peak physical footprint is 236864520 bytes and swap growth remains
zero. The readout is `development-readout-2000k.json`, SHA
`262979f1deb1069f0ff9b401f5c2d83b55b79f0de4051b770b57d6ac64e828f3`.
All comparisons below use the unchanged 24 development conditions and three seeds;
no validation or decoding changes were introduced between checkpoints.

| Measure | O1 | R1 | R0 |
| --- | --- | --- | --- |
| Complete-suffix pooled NLL, nats/source onset | 1.906802 | 1.915834 | 2.031677 |
| Group-macro NLL | 1.968009 | 1.987622 | 2.129202 |
| Generated LN heads / all heads | 60430/163160 =37.037% | 35910/129871 =27.651% | 59689/126164 =47.311% |
| Quad attack rows | 223 | 104 | 193 |
| Same-lane attack pairs below 40 ms | 498 | 28 | 30 |
| Such pairs per 1000 heads | 3.052 | 0.216 | 0.238 |
| Longest such attack run | 8 | 2 | 2 |

R0 NLL remains a different-condition task and is not a same-task likelihood
ranking. O1 minus R1 paired macro NLL is -0.019613, with group-bootstrap 95%
interval [-0.049635,0.011185]. The ordinary subset has difference -0.034013,
interval [-0.061996,-0.006458]; the stress subset difference is-0.005213,
interval [-0.058856,0.048380]. No overall stable arm winner or independent
confirmation follows from this development sample.

Quad concentration at 1M is transient on this trajectory: counts fall from
17684/19354/19047 to 223/104/193 without decoder penalties or support changes.
All three models also broaden LN use. The early low-LN and intermediate quad
outcomes cannot be treated as invariant representation failures. However,
O1 rapid recurrence rises, concentrated in stress outputs, and must remain a
quality concern. The figure `learning-curve-v1/learning-curve.png` and its SVG
show source fitting, LN share, quad share and rapid-pair rate across the three
checkpoints. Its provenance pins all input readouts; it is descriptive, not a
composite quality score.

### Fixed-core visual interpretation at 2M

All 32 canonical images were inspected at 1x. Manifest SHA
`4e692038af60fc4acf178472b5ea53429a4423698267f5a174916b57cbf54fdc`, scoped
machine review SHA `c1558cf25d6331fbd233e7ff4909cac3a007f03a42c9e0ff4184d328e65a73fc`.
The frozen Foundation and previously inspected High human positive/supporting/
negative LN cases remain the calibration. No new human labels are claimed.

In the fixed LN-rich 16 s core, O1 has 161 LN heads/80 taps and R1 has 231 LN heads/
38 taps. Both now show sustained independent start/release exchanges:
LN coordination present/prominent, High. O1 interleaves short TAP-led breaks;
R1 more continuously exchanges short holds. For O1, column 1 holds 21505–22030
while column 2 releases 21805 and re-enters 21880, column 3 taps then starts 21955,
and other lanes take distinct control roles. For R1, column 3 holds 18355–18805
while columns 0/1 release 18505 and re-enter 18580, and column 2 releases 18655 and
re-enters 18730. These are arrangement relations, not positivity inferred from
LN fraction. R1's 1M local weakness has changed with further training.

In the dense 16 s core, O1 has 468 taps and four isolated LNs, while R1 has 301 taps
and no LN. O1's four holds never overlap. Both are LN-coordination absent, High.
O1 mainly uses moving TAP/chord groups and some fixed-pair exchanges; R1 mainly
moves single-column TAPs. Other tags are not formally assigned. A TAP-heavy
adaptation is not itself failure, and matching the source LN ratio is not the
criterion. Ordinary cores, R0 images, other seeds and 64 s organization remain
unreviewed by this fixed-core record.

The repeated LN policy probe completes in 7.118 seconds and exactly reproduces
both 2M outputs. SHA `783e13acb5c6d866499cbd311bc43dea498213a9e3e0c74197071eebbd9eeda3`.
In the first four 32-onset blocks, O1 generated-history expected LN mass is
62.548/71.720/88.362/75.626%, R1 is 82.489/59.115/60.501/52.992%. At block 1344,
O1 retains 79.294% and R1 retains 56.677%. This is stronger retention than the
respective early failure trajectories, but remains one source/seed diagnostic.

### Quad policy concentration changes with learning

The prospective 12-replay diagnostic completes in 37.831 seconds; every native
row reproduction, H alignment and first-query equality check passes. Readout
`quad-policy-probe-v1/readout.json`, SHA
`1e37777dfd7fe101637ac453e1f61bc037848ffda9064ae871949ba23420cb53`.

At source `27ac9470c9a4`, seed 23, the O1 core mean probability of a quad is
0.0153% at 250k, 87.315% at 1M and 0% at 2M under its own generated histories. At 1M its
source-history quad probability is only 1.452%, while all 67 realized generated
attacks are quads. At 2M this generated core is instead LN-heavy, with 84.377%
expected LN-head mass. At source `c1798e61528c`, seed 17, R1 generated-history
quad probability changes 0.949%→93.409%→0%; its 1M source-history value is 1.507%.
All 59 realized 1M attacks are quads. O1 also has 86.859% quad probability on this
second 1M core, compared with 1.527% under source history.

Thus these 1M runs reflect concentrated conditional policies under generated
histories, not merely improbable samples from a consistently diffuse policy.
The concentration largely disappears at 2M on the paired trajectories. Because
histories differ, this does not isolate the cause among fitting/calibration,
representation, style retention and feedback. The post-hoc locators are not
independent confirmation. Evaluation: REFINE.

### Remaining O1 burden: commitments can leave only one available lane

A deterministic readout replays all generated physical rows at all three
checkpoints. Before each suffix attack it counts occupied lanes, preserving the
rule that a lane releasing at that instant still cannot restart. It verifies its
rapid-pair counts against every existing generation summary. Output
`rapid-occupancy-readout-v1.json`, SHA
`c7f2886ebb54a5f5d3e1e9c2ddd4ee7778e28f38fe11aad9379c7c3632acafd8`.
At 2M,439 of 498 O1 pairs below 40 ms occur with only one lane available for a new
head, versus 8 of 28 for R1 and 0 of 30 for R0. At 250k O1 had 0 of 166 such constrained
pairs; at 1M it had 75 of 370. The R0 occupancy count is descriptive: R0 does not
receive a mandatory-H condition. For O1, previous endpoint commitments make the
other lanes unavailable; a head at that H is therefore forced onto the sole
free lane. This is a sufficient factual explanation of head placement at that
step, not proof of why the earlier LN decisions were learned.

The largest 2M O1 run is on source
`4f8b228d1b19985500cfbd43c4a166a3708c81cc9e57addcaaba8e3308aab2ca`, seed 23.
Column 2 attacks 18170/18205/18239/18273/18308/18342/18376/18411 ms: eight attacks
over 241 ms, with 34–35 ms gaps. Earlier generated LNs occupy column 1 at 18033–18891,
column 3 at 18067–18411 and column 0 at 18136–18411. All three block a head for the
whole burst, including 18411 under the no-close/restart rule. The same supplied
H permits R1 and source to distribute attacks across columns without such holds.
This is a generated allocation problem, not an unavoidable property of R/H.

All six canonical context images for O1/R1/source were inspected; the last page
covers 742 ms. Manifest SHA `2b52b1a2322a910d1bb11023574fe1b1c6bf682f7ac268e46e1f714fd1789302`,
factual review SHA `a05d463e27f78cae084a914d7c52796037b4ec6ee18ae9c4088c93b0dc710f0b`.
No new semantic label or universal playability cutoff is assigned by this probe.
The rapid-pair locator selects an adverse case; the population readout supplies
its separate frequency evidence.

### Next research decision

Do not select O1 from lower NLL or more LN output. R1 remains cheaper and avoids
most observed rapid recurrence at 2M; R0 also improves markedly. Further common
training remains a live explanation because fitting and native behavior are still
changing. Another initialization, unused groups and larger-context ordinary
inspection are still needed before a stable model choice.

For a targeted O1 branch, inspect how an endpoint candidate exposes the future
availability implied by prior commitments. Source audit shows that the current
query reads exact remaining LN times and near/far timing information, but each
candidate vector encodes duration, adjacent gaps, distance/rank to the endpoint
and its role. It does not explicitly encode the number and spacing of mandatory
H that would have only one free lane under that candidate and already committed
plans. The network could infer some of this from its inputs; the audit does not
prove a missing-information impossibility.

A candidate-conditioned availability feature derived only from supplied R/H and
already chosen plans is a distinct, testable access-path change. It introduces
no new source LN oracle, no minimum-interval rule and no repetition penalty.
It should be compared against matched additional training of the current O1,
with the same LN/mode and burden checks; any model/plan/source change requires
a new proposed Card and explicit checkpoint-migration accounting. Do not silently
extend the immutable 2M plan or relabel completed checkpoints. Coarse external
LN-intent conditioning is not currently the first intervention: LN production
and independent coordination have already emerged, while safe allocation of
future capacity is a concrete remaining problem. Direction remains REFINE;
no new training or representation change has been executed from this section.


## Experiment Card: bounded-typed-endpoint-availability-v1

Revision 1, proposed, acceptance none. Standing implementation/experiment authority
covers this scoped exploratory comparison; no adoption or publication is implied.
The preceding goal turn made progress: paired 2M training, complete development
evaluation and occupancy diagnostics changed the next research question. Current
product and note worktrees were revalidated clean before designing this change.

**Question and baseline.** Does exposing candidate-specific future lane availability
reduce O1's generated rapid recurrence beyond additional training or endpoint
network capacity, without suppressing LN organization? Clean baseline source
`1693d62ffaca04b2a6127d8e3a72d1988adf441f`; initialize every branch from its final
O1 2M checkpoint SHA
`759a4e6212e1c47b0039a457d9617bdfa957a7989e31f3715b6f70ff0b18d3b1`.
The fixed 24-group/three-seed development screen has 498 below-40ms same-lane pairs,
3.052219 pairs/1000 heads pooled and 2.593285 after equal group/seed weighting;
439 pairs have only one free lane. Complete-suffix pooled NLL is 1.906802 and
LN share is 37.037%. These are exploratory development baselines, not independent
confirmation or a universal difficulty scale.

**Branches and intervention.** P0 continues the original endpoint network. C0 adds
a small zero-initialized residual endpoint scorer reading context and ordinary
candidate features, with the new availability inputs fixed to zero. C1 uses the
identical residual scorer and supplies the derived availability features. C0/C1
start from identical added weights; the residual final layer is zero, so all
three begin with the parent's conditional probabilities. New parameter moments
start empty, while all existing optimizer moments/steps, RNG, coverage, exposure
and metrics are preserved. This separates extra capacity from the access path.
No head-network, finite-context, likelihood, legal support, decoding or external
condition change is allowed.

For a proposed endpoint, compute from supplied R/H, previous LN plans and earlier
within-row factors the future interval during which the target LN plus at least
0/1/2/3 other *known* holds remain occupied. For each interval expose transformed
H count, summed inverse H gaps and elapsed duration, plus known-other and pending
current-LN counts. Count H at an LN endpoint as blocked on that lane because same-
time release/restart is excluded. Pending within-row endpoints remain unknown;
never substitute their future teacher-forced labels. These are partial-plan facts,
not a prediction of all future occupancy. The transforms introduce no gap cutoff,
maximum duration, penalty, forbidden pattern or truncated endpoint support.
The closest implementation analogue is the current exact-clock/plan readout;
this is a derived-feature access experiment, not a novelty claim.

**Matched data and execution.** Extend the deterministic corpus plan into a fresh
artifact with milestones 250k/1M/2M/2.25M/2.5M. Require identical sources, data pins,
sampling settings and the entire old draw prefix. Never modify the existing plan.
The new source, extended-plan SHA and exact driver/config identities must be
recorded before training. Add an explicit audited fork initialization path, distinct
from strict same-source resume, that verifies the pinned old checkpoint, finalized
runtime ledger, old plan and prefix relationship. Charge the parent's full 2174.555
seconds and any necessary migration cost to each branch. Ordinary resume retains
its strict source/configuration identity; no relabeled checkpoint workaround.

Run sequential P0/C0/C1 on CPU one thread, with the unchanged batch4/microbatch2,
AdamW, LR0.0003, clipping, candidate8192 and cache settings. First stop at2.25M;
continue to2.5M only if mechanics/resources/finiteness pass and the comparison is
still informative. The original four-hour cumulative per-branch cap, 6GiB footprint,
RSS/swap/available-memory, fresh-output, disk and checkpoint guards remain. Fresh
artifact owner `artifacts/bounded-typed-continuation/availability-20260919-v1/`.
No accelerator training is planned. Targeted CPU/MPS numeric tests remain allowed.

**Verification before execution.** Compare availability features against brute-force
future-H occupancy, including an endpoint exactly on H, translated large timestamps,
unknown later factors and mirror reversal. Verify full-support normalized likelihood,
chunked/dense values and gradients, mirror equivariance with a nonzero new scorer,
sampler feasibility/partition invariance, and bounded recomputed activation storage.
Verify initial old/new function equality, existing optimizer preservation, identical
C0/C1 initialization, successful fork continuation, exact ordinary resume, and rejection
of changed source/plan prefixes, digests or undeclared scientific settings. Keep
canonical Hydra schemas/presets complete and reject unused options.

**Readout and decision.** Repeat the same fixed 24 development conditions and seeds
17/19/23 after the paired checkpoint. Primary burden diagnostic is the group-balanced
mean below-40ms same-lane pair rate per1000 generated heads, averaging seeds within
group. Report pooled counts and one-free-lane pairs separately. A promising access-
path result needs at least25% lower primary rate than both P0 and C0, with a paired
group-bootstrap interval supporting a reduction. If a control rate is zero, there
is no identifiable reduction to claim. Mechanical/export/reparse errors must be zero;
complete-suffix pooled NLL may be at most0.05 nats/onset worse than the better control.
Generated pooled LN share must retain at least75% of each control's share, and fixed
LN-rich core inspection must retain clear independent control. These are experimental
guards, not semantic label thresholds. Check mode concentration, LN duration, chord
counts and the known three-hold burst alongside broader ordinary/64s organization.
A reduced burden obtained through LN or organization collapse is a failed result.

If all branches improve similarly, prioritize further training and retain the simpler
network. C0 matching C1 supports capacity/optimization rather than the new facts. A
C1 benefit with failing quality guards calls for refinement. Wide uncertainty is
inconclusive, and another initialization/unused groups remain required for a stable
choice. This exploratory Card cannot establish overall playable-quality completion.
Implementation and execution results are pending at this revision.


## Endpoint-availability implementation and Card revision 2

Clean intervention source `15d27db0b0723d7f606b1429c901d26c1f61e5ac` implements the
three modes and audited training fork. The baseline-to-intervention diff changes
only endpoint feature/scoring access, fork/configuration plumbing, their tests and
the owning research guide. The original task support, head network, data selection,
finite context, normalized likelihood and sampler rules remain intact. Product
code is locally committed; no remote publication occurred.

Fourteen availability inputs encode the four occupancy intervals, continuous
inverse-gap mass and elapsed duration, plus known/pending counts. C0/C1 both add a
residual MLP of width64 at default hidden128, with a zero final layer. Input
features use only known earlier endpoints. Source-level tests exercise a nonzero
residual so normalization/mirror checks are not vacuous initialization checks.

Selected checks: feature/model24 tests passed in3.58s; train/fork/Hydra20 tests
passed in10.26s; generation/data/corpus/memory/package54 tests and21 package
subtests passed in6.82s. These include CPU/MPS gradients and native recovery,
exact extended-plan fork/resume equivalence, existing Adam-state preservation,
initial functional equality and rejection of mismatched scientific settings,
digests and draw prefixes. The packaged `--cfg job` path displays the new mode
and all four fork fields; it is configuration inspection, not a training run.
`git diff --check` passed. No CUDA or full-repository claim is made.

The new immutable plan is `availability-20260919-v1/plan.json`, SHA
`ab4e01d1efd78162f0bc6e1e612428ab879a2a7d9e36d3e1173079607fd2bef1`.
It contains13213 draws over the same11563 TRAIN sources, with the full10573-draw
old prefix verified exactly. Milestones are250k/1M/2M/2.25M/2.5M. The original
plan and checkpoints remain byte-identical.

Card revision2 pins that clean source/plan and adds the concrete initialization
preflight. All scientific thresholds, comparisons and guards from revision1
remain unchanged; acceptance remains none. Driver `initialize_compare.py`, SHA
`5cc9b2867bf704f72c697997a2780ea8426ec57e8a76d98b9404ce395a9e669c`, passes syntax
compilation. Invoke it with `uv run --offline --python 3.10 --extra mps --group dev
python artifacts/bounded-typed-continuation/availability-20260919-v1/initialize_compare.py`.
It composes the canonical Hydra settings for P0/C0/C1, supplies the pinned parent
checkpoint/source/plan, and stops immediately at2M in fresh `p0-init`, `c0-init`,
`c1-init` segments. These initialize without optimizer updates and preserve all
old parameters, moments, RNG, counters and coverage. Parent compute is charged.

Then verify exact state copying, identical C0/C1 initial parameter bytes, and
native generation on fixed screen indices0/16/23, seed17. Each initial variant
must reproduce the parent's complete physical rows and mechanical checks; full-
suffix head and endpoint NLL sums must agree within1e-5 nats. The three-minute
verification bound and6GiB/resource guards are separate from training; report
verification cost separately. On any mismatch, retain outputs and stop before
training. Do not overwrite initialization artifacts. Successful preflight makes
the concrete paired continuation reviewable; no quality benefit is yet claimed.


## Availability initialization result and first paired continuation

The initialization/preflight succeeds in43.881 seconds. Every original parameter,
Adam moment/step, RNG value, coverage bitmap and cumulative metric is preserved.
C0/C1 added parameter bytes are identical and all initial functions agree. Each
variant exactly reproduces the prior complete generated rows on the three fixed
sources (621/4402/2042 rows), with mechanical verification and complete-suffix
head/endpoint sums within1e-5 nats. Readout SHA
`895b8cee87a2ee0d31e38be2b374313663b1d7e9b43930c1f68be9e90633885d`.
P0 has2355835 parameters; C0/C1 each have2371260, an addition of15425 parameters.
No optimizer update occurs in initialization.

Initial checkpoint SHAs under `availability-20260919-v1/`:

- `p0-init/checkpoint.pt`: `83b67714e4eaf36300616e3250ea3883c6363bcb51543bac85980c4783cfdb4f`;
- `c0-init/checkpoint.pt`: `1ead63e28fede485765e33feaefc134974d86d694c676028629f16e50e3d2355`;
- `c1-init/checkpoint.pt`: `43adb7a1a65de36b7fc11c04e62b33c91bde11ca32367a21e1b269d2ed7e362f`.

Initialization takes0.741/0.434/0.465 seconds, charged with the parent training in
the new ledgers. The separate real-chart verification cost is development evidence,
not supervised exposure. No source or checkpoint repair was needed.

Card revision2's first continuation uses `train_comparison_2250k.py`, SHA
`6248bd7dfe1ba9b0006c19f26fd433dff83bc478df262cbaa542bd50b59c102b`.
It checks the successful preflight digest and exact initial checkpoint SHAs, then
runs P0/C0/C1 sequentially through canonical Hydra composition with
`stop_after_checkpoint=2250000`. Fresh segments are `p0-2250k`, `c0-2250k`,
`c1-2250k`; fork fields are cleared and strict resume is used. Source, extended
plan, optimizer, batch, candidate and resource settings stay pinned above.
The driver verifies matched draw/row/prefix/padding/endpoint ledgers and coverage.
Invoke with the same explicit Python3.10/MPS-dependency/dev prefix as the preflight.
Preserve any failure and stop the chain. Training is authorized exploratory work;
acceptance remains none and the quality result remains pending.


## Availability Card revision 3: fixed 2.25M readout drivers

Proposed, acceptance none. After all three paired continuations complete, run the
unchanged24-condition/three-seed full-suffix likelihood and native generation
screen on the new source and exact finalized2.25M checkpoint digests. The model
task remains O1 in every branch; `arm` in this comparison's readout names P0/C0/C1.
This revision pins measurement drivers without changing intervention, selection,
thresholds, resource bounds or interpretation.

- `development_screen_2250k.py`: `b3a456635e34c444bdc36b4105bd97744c1a2bb8313cb105f320f4d880108a1a`;
- `screen_readout_2250k.py`: `401b57e84d88b921a1ca4c8c89d6520d5cd74af4e8a72d0bb763989e6703b7a9`;
- `render_screen_2250k.py`: `9e116cd7078a867d3ef47c69902c6868c96c39bc16575541d697520f3788b458`.

All syntax-compile. Invoke through the same `uv run --offline --python 3.10
--extra mps --group dev python` prefix after checking the completed training
readout and checkpoint identities. Fresh outputs are
`availability-20260919-v1/development-screen-2250k-v1/`,
`development-readout-2250k.json` and `inspection-2250k-v1/`.
Screen limit stays one hour,6GiB footprint and2GiB output; existing resource
and exact-recovery/export/reparse checks remain. The rendering uses the same
fixed LN-rich/dense cores and seed17 for all three branches, with canonical
images and action listings. Later ordinary/64s inspection remains separate.

The primary rate averages output rates over three seeds, then equally over source
groups. For C1 versus each control, report its point ratio and paired bootstrap
interval for the *difference* in group rates; the latter remains defined for
zero-rate groups. A reduction gate requires a positive control mean, ratio<=0.75,
and a95% percentile interval wholly below zero. Bootstrap units are groups with
seeds nested inside, seed1971,10000 resamples. Separate pooled counts, one-free-
lane pairs, LN shares/quantiles, chord histograms and complete-suffix fitting
remain in the readout. Numeric guard success never substitutes for calibrated
LN-control/mode/organization inspection. Scientific conclusions remain pending.


## Availability paired training result at2.25M

P0/C0/C1 each reaches2,250,000 cumulative onset exposures and2974 updates; the
new segment contributes250,000 onsets and329 updates. Draw/row/prefix/padding/
endpoint-factor ledgers and coverage agree exactly:1,873,845 distinct onsets,
6,735 charts and3,108 groups. No recovery discards or training failure occurs.
Segment runtimes are269.571/314.910/334.380 seconds, cumulative charged training
2444.867/2489.899/2509.400 seconds. This is an equal-exposure comparison; added
endpoint computation is included in each separate compute ledger. The sequential
CPU order is fixed P0/C0/C1 and cache/order effects remain a timing confounder.

Training readout SHA `2cfb3b7e01e6591dd0be37bba84721e1995f468f819209597fb6b791cb8a036e`.
Finalized checkpoints under the corresponding `<arm>-2250k/checkpoint.pt`:

- P0: `a4769bdc1498e178e24dd40050612ffcf67e0076f6879b61ac086a9964686f95`;
- C0: `acd44e22d15266f926fc107c804ac92c3ff02034dbbc0afa5b10d9a0f2119c2b`;
- C1: `4572e55888d8498ec541222e2c30b68fa5963edf98e045e9c19761653ce82bbe`.

All checkpoint digests were revalidated against the completed paired report.
The predeclared fixed development screen can now run on these exact inputs;
source stays `15d27db0b0723d7f606b1429c901d26c1f61e5ac`. No source/model or selection
change follows from local training losses. Quality and control comparisons remain
pending generated evidence.


Resource follow-up confirms P0/C0/C1 segment peak footprints878282336/708609656/
746079912 bytes with zero swap growth. All three checkpoint digests were checked
again. Both residual networks received329 Adam steps on every added parameter;
final-layer weight norms are0.081116/0.084970. Small local loss differences cannot
be explained by an uninitialized or entirely untrained residual branch. An initial
ad-hoc resource summary omitted the optional startup-record footprint field and
raised KeyError; its corrected read skips records without that field. Training
and fixed-screen execution were unaffected.

## Additional Stream calibration for broader organization review

The unchanged eight canonical workflow documents and frozen Foundation were
revalidated before using the existing effective-human packet. Three High examples
were opened through the canonical example reader and all12 context pages inspected:

- `human-2af74649b91a0eb016f7c6d2`, source `871955cefaa2`, scope105169–109343ms:
  human Stream present/prominent. Changing single/pair groups sustain motion while
  mixing taps and short LNs; lead columns and chord placements vary.
- `human-559a6f83cb9345097c3c46d7`, source `5b69e9a82e2b`, scope113773–120440ms:
  human Stream present/supporting. Changing held groups and interleaved attacks
  coexist with more separated chord/hold figures. The final context page covers
  only1ms, outside the assigned core, and cannot establish a long static hold.
- `human-b8e95528659f86b0d58df1b7`, source `ecc49676c356`, scope148072–153786ms:
  human Stream absent. Recurrent quad attacks dominate, with inserted single/pair
  attacks; continuous activity alone does not establish flowing organization.

The descriptive readings above are machine interpretations of the inspected
sources, not new human rationales. The public reader retains one confidence-update
comment and no substantive pattern explanations for the other two examples. Labels, comments,
Foundation and canonical documents remain unchanged. This calibration does not
assign generated Stream tags or prove64s organization.

Artifact owner `availability-20260919-v1/human-stream-calibration-v1/`, manifest SHA
`7f939c2d2833a7d5cbc253b4189667192b2e30ed49a518e1802fab1fceb8e078`.
Review SHA `ec689620cf5665b0e9139624ad08c14593adc8eb2b071a005bfff4a6097de78d`.


The calibration record writer initially compared the public example projection
with the full provenance packet and stopped on that unequal shape. The corrected
check compares the seven shared identity/judgment/scope fields and preserves the
public reader's comment policy. All three public examples and all12 image hashes
are verified; no dataset or human record was changed.


## Endpoint-availability result at 2.25M: no demonstrated improvement

The fixed screen completes all 216 generations and 72 suffix scores in 647.692 s.
Every mechanical/export/reparse check and the three selected exact recoveries
passes. Peak physical footprint is 231703512 bytes; swap growth is zero. Readout
`availability-20260919-v1/development-readout-2250k.json`, SHA
`d7502657f151ba72c32e6f09fb3455a2062becc472353078605e55b77ee8be77`.

| Measure, all 24 development groups | P0 original | C0 capacity control | C1 capacity + availability |
| --- | --- | --- | --- |
| Primary group-balanced rapid-pair rate per 1000 heads | 0.267819 | 0.278082 | 0.278270 |
| Pooled rapid-pair rate per 1000 heads | 0.401099 | 0.410048 | 0.410235 |
| Rapid pairs / those with one free lane | 53 / 1 | 54 / 5 | 54 / 5 |
| Longest below-40ms attack run | 3 | 3 | 3 |
| LN heads / all heads | 13560/132137 =10.262% | 13041/131692 =9.903% | 12974/131632 =9.856% |
| Complete-suffix pooled NLL | 1.913932 | 1.913899 | 1.913886 |

C1's primary-rate ratio is 1.039025 against P0 and 1.000675 against C0. The paired
95% group-bootstrap intervals for its rate differences are [-0.014629,0.040239]
and [-0.00000754,0.00052488], respectively. Both reduction gates fail. C1 passes
the numeric relative LN-share and NLL guards, but has no demonstrated burden
advantage. C0/C1 produce exactly identical complete physical rows on 65 of 72
source/seed cases. P0/C0 agree on 35 and P0/C1 on 34. The near-null effect is not
an untrained-parameter artifact: both added networks received 329 Adam updates.

The larger change is shared with the original model's continued training. O1's
2M parent had 37.037% LN share, 498 rapid pairs and 439 one-free-lane rapid pairs;
P0 at 2.25M has 10.262%, 53 and 1. Complete-suffix NLL slightly worsens from
1.906802 to 1.913932. This is not evidence that candidate availability solved
allocation. Reduced burden accompanies a substantial change in LN use and the
matched LN organization described below. It also does not prove that lower LN
use makes every output worse: alternative TAP-heavy arrangements are allowed.

### Fixed-core verification and quality guard

For each of the two preselected cores, P0/C0/C1 have identical whole exported
chart bytes and image bytes. All 16 unique P0 context images were visually
inspected; 32 corresponding duplicate image paths were hash-verified rather
than separately counted as visual inspection. Manifest SHA
`99cdc4fd876362a0b453e8b512ed1737c47b038cacaae168294a894d07c6f2af`;
scoped review SHA `efdfb9f5898996bc1b6f9b59c7e9350ec2e705a2a427985a5100c22fef7926dd`.
Foundation and the previously inspected High human LN comparisons are unchanged.

The LN-rich core has only four LN heads and 237 taps: column3 at17155–17380,
synchronized columns2/3 at18580–18655, and column0 at19630–19705. The pair has no
independent interior event; the other holds are isolated. LN coordination is
absent, High, for all three variants. The dense core has 305 taps, no LN heads
and no entering holds: absent, High. These assessments use event relationships,
not a LN-percentage threshold. Other semantic labels remain unreviewed here.
The matched LN-rich core's 2M independent coordination is not retained, so the
semantic retention guard does not pass, despite the relative numeric LN guard.
This limited scope cannot establish overall playability or method preference.

A simple read-only check of the last 32 training updates gives observed conditional
LN rates on feasible true head locations of 20.240% at 2M and 19.365% at 2.25M.
The much larger generated-share change cannot be identified with this local label
fraction alone. Optimization, incomplete fitting and generated-history feedback
remain live alternatives; no unique cause is established by these comparisons.
`comparison-figure-v2/comparison.png` and SVG visualize the shared checkpoint change
and near-null feature effect, with exact input identities in their provenance.

### Decision and continuation state

Evaluation: REFINE. Do not promote the new availability scorer or spend the
reserved additional 250k on C0/C1 without a new reason. The first declared 2.25M
checkpoint is a valid stopping point for this branch comparison; the immutable
2.5M plan and all checkpoints remain available, but no 2.5M run has begun. The
new feature remains opt-in with `none` as the default. This finding is bounded
to the tested 250k continuation and does not prove such features can never help.

Prioritize common training/rollout stability and the still-competitive simpler R1.
Checkpoint sensitivity, longer matched training, learning-rate policy and weight
averaging are candidate discriminating questions; none has been executed or
established as the remedy by this result. Before any new run, choose a bounded
comparison and pin its source/configuration and parent accounting. Another model
initialization, unused-group confirmation, ordinary and larger-context organization
review remain required for a stable quality conclusion. The original goal remains
active and unfulfilled; no note lifecycle acceptance or implementation transition
is implied.

All processes from this comparison are terminal: initialization, P0/C0/C1 training,
216-generation evaluation and rendering have completed. Product source remains
clean at `15d27db0b0723d7f606b1429c901d26c1f61e5ac`. The source and evidence commits
are local; no remote push occurred during this experiment.
