# Agent Note: Bounded typed-time three-arm comparison

Note ID: 2026-09-18-bounded-typed-time-three-arm-comparison
Status: proposed
Kind: research
Created: 2026-09-18
Updated: 2026-09-18
Product revision: 21475e65d773b7e7199accf0750de584d9f10ce9 on codex/bounded-typed-continuation; trained source a178bcfe2badaaea47ae9abce02f2494b8ff9643; published review base 89d5379f150cba9d1684822a44166765d38f644f
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
