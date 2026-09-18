# Agent Note: Bounded typed-time three-arm comparison

Note ID: 2026-09-18-bounded-typed-time-three-arm-comparison
Status: proposed
Kind: research
Created: 2026-09-18
Updated: 2026-09-18
Product revision: 1693d62ffaca04b2a6127d8e3a72d1988adf441f on codex/bounded-typed-continuation; generation source 21475e65d773b7e7199accf0750de584d9f10ce9; short-fit trained source a178bcfe2badaaea47ae9abce02f2494b8ff9643; published review base 89d5379f150cba9d1684822a44166765d38f644f
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
