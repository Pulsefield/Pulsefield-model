# Agent Note: Can candidate action consequences improve R1 occupancy decisions?

Note ID: 2026-09-20-r1-action-consequence
Status: proposed
Kind: research
Created: 2026-09-20
Updated: 2026-09-20
Product revision: ca28511cd69ccd0566bdb42be52e4651fb236369
Scope: R1 candidate-row residual, matched continuation controls, native burden and LN retention on the existing 28-group development cohort
Related: 2026-09-20-r1-difficulty-ln-longform; 2026-09-19-bounded-typed-r1-burden-mechanism; 2026-09-19-r1-additional-exposure; 2026-09-20-bounded-support-throughput

## Question and new diagnostic evidence

The 4M model learns independent LN organization but still assigns substantial
probability to actions that leave a required close-spaced attack only one lane.
Two frozen generated histories assign .268926/.183571 mass to predecessor rows
forcing23/37ms head repetitions. This is a model-learning problem worth testing
before introducing a new sampling policy. Lower loss alone did not remove it.

A read-only extension checks latest-release-to-head intervals on all84 primary
outputs of the fixed28-group evaluation. There are153 below40ms cases, with a
group/seed-mean rate .962834 per1000 supplied suffix onsets. The union with the
14 head-to-head cases is167 heads, rate1.044819. The threshold is a diagnostic
locator, not a universal playability definition or proposed decoder constraint.
The baseline derivative hashes are respectively
`36669e548f261a3b4abfe259c3ed66bb6108908b174a35fa81567b23787b6b95`
and `4903f879c3cf19a6a4c2216517cc3f9ef55623af64a023af3d98772e992c869b`.

The smallest case is source20/generation17, column1, release133350ms followed by
LN start133353ms. All four LNs start at133271ms, probability .0487574, and the
only intervening candidate is133350ms. Every feasible release selection then
forces a3ms release-to-head interval at the required133353ms onset. The native
model probability of such a reattack at that next state is numerically one.
The original source instead uses TAP0/LN3 at133271, releases an older column2
LN at133350, and starts column1 at133353; that lane last released at133113.
This source contrast changes the whole history, so it is not causal isolation.

The read-only diagnostic replays eleven generated and eleven source positions;
every generated chosen log probability exactly equals its original journal.
Both complete12-second contexts were inspected through all ten canonical pages,
plus the generated risk page at full resolution. The generated core129351.5–
137351.5ms contains prominent independent LN starts/releases followed by a
clear TAP passage; the3ms restart remains a localized quality concern. The
source keeps LN organization more persistently across that transition. This is
an agent judgment calibrated against the frozen prominent-LN gold example
human-03f7e300cf02f58f3dcbba66, not a new human label. Other quality dimensions
remain unreviewed. Report SHA:
`af78bc47bf2a5d09ec1c089fd580031388ef6cc011e9f35cf0e278664775394f`.
All new diagnostic artifacts belong to
`artifacts/bounded-typed-continuation/action-consequence-20260920-v1/`.

## Experiment Card: r1-action-consequence-v1

Card ID: r1-action-consequence-v1. Revision: 3. Accepted revision: none.
The user authorizes autonomous bounded implementation, training, evaluation and
meaningful local commits toward the active goal. This is an exploratory proposed
Card; execution does not accept the Note or adopt a final model.

### Hypothesis and analogues

Candidate-conditioned access to exact post-action facts makes coupled occupancy
costs easier to learn than inferring every consequence from one shared query
vector. The question is whether it improves native generation beyond continued
baseline learning and a nonlinear candidate-action readout of equal parameter
count. Both controls matter: more training and a higher-order output head can
each explain improvements without consequence features.

The closest local analogue is the zero-initialized O1 endpoint-availability
residual in `model.py`: an optional candidate-dependent score addition preserves
the original function and Adam state at a fork. R1 must retain unknown suffix LN
ends. The broader analogue is learning an energy of a structured candidate
([Belanger and McCallum,2016](https://proceedings.mlr.press/v48/belanger16.html)).
Here all256 discrete rows are enumerable and use exact normalized likelihood,
unlike that paper's iterative label optimization. This is a conventional feature
and readout adaptation; no representation/objective novelty is claimed.

Larger models, a longer temporal context, temperature changes and hand-written
burden penalties are separate branches. The current traces specifically expose
exact action consequences available without those changes. If this smaller
intervention fails against the controls, revise the mechanism rather than
silently expanding it during this comparison.

### Fixed baseline and intervention

Start from the clean product revision above in the isolated
`codex/r1-action-consequence` worktree. Parent: initialization172 at4M, checkpoint
SHA `ed4ad7dcec30fb2c6f13ee39908bb34c45b06b41799cdf30cfe410d96efdac28`,
execution source `50dda55040f51a7afc9a13994b762f953fe3064d`, owned by
`r1-exposure-20260919-v1/seed172-4000k/checkpoint.pt`. Parent plan SHA
`5621297fcf8988d0aa3ae2f35f134af94bba2d21ce68e224dda1d70fb160538c`.
Preserve all source pins, original draws, sampler settings and completed
milestones; append4.125M and4.5M milestones and their deterministic draws.
The shared full TRAIN population is11563 charts/3169 groups. TEST stays unread.

Three modes are trained serially from the same parent and extended draw plan:
`none` continues the original model; `actions` adds the residual with only
candidate-action one-hots; `frontier` supplies the additional exact features.
The latter two have identical parameter shapes and initialization. Zero-valued
feature columns in the control have inactive input weights, so equal total
parameter count is not a claim of equal effective state-feature capacity.

For each physical lane and each of its four hypothetical actions, encode its
one-hot action, post-action occupied flag, and these seven signed time bases:

1. For a current TAP/LN head, time since its preceding same-lane head.
2. For a current TAP/LN head, time since its preceding same-lane release.
3. For a current release, elapsed age of the LN being released.
4. At the next strictly future required onset H, elapsed time since the
   post-action latest head.
5. At that H, elapsed time since the post-action latest release.
6. At that H, age of the post-action open LN if retained without another action.
7. For a post-action occupied lane, H minus its earliest possible future release
   time: its supplied seed endpoint when known, otherwise the next R candidate.

The last four are timing-based views of the candidate's immediate state. They
do not assert that intervening actions are known or will retain that state.
Missing predecessors/future events use the existing unavailable time basis.
The timing block adds next-R gap, next-H gap and next-R onset role. Existing
float64 timestamp subtraction and23-component smooth time bases are reused;
no duration cutoff, source suffix label or newly learned lookahead policy enters.

A width32 MLP energy sums four lane-position affine projections, a timing
projection and the existing hand context projection before GELU and a scalar
output. Canonical relative lane ordering and averaging the two hands preserve
mirror equivariance. The scalar output starts at zero. Factorize the first
affine over16 lane/action feature vectors and gather for256 candidates, rather
than materializing all dense candidate descriptors. At hidden128 this adds
26912 parameters. `actions` zeros every field except action one-hots. The
existing exact support mask and native temperature-one sampling are unchanged.

Necessary runtime/config changes are limited to the optional model field,
candidate feature/energy owner, central decision score path, explicit fork
migration, packaged Hydra projection, relevant tests and self-contained docs.
Generation recovery remains raw-history based and uses the same scorer.

### Procedure and resource bounds

First verify feature/replay agreement, missing clocks, true end, seed obligations,
time-shift invariance, no future-label path, exact support, mirror symmetry,
factorized/dense values and all gradients on CPU/MPS, zero-residual/Adam parity,
matched initialization, nonzero-residual native/dense/recovery parity, and strict
fork/resume equivalence. Use existing small tests, then a real CPU1 batch probe
on the first eight extension draws: at most120seconds per mode, no saved trained
candidate, footprint6GiB and no more128MiB swap growth. Stop for nonfinite loss,
parity failure, memory guard or frontier update time exceeding four times the
continued baseline. Fix only implementation bugs before freezing a clean source
revision; a scientific change requires a new Card revision.

Freeze source OID, exact shared plan, per-mode resolved configs and driver
digests before learning. Use Python3.10, Torch2.11, NumPy1.26 on the M5 Mac,
`uv run --offline --python 3.10 --extra mps`, CPU1. Keep AdamW learning rate.0003,
weight decay.01, batch4/microbatch2, global gradient clip1, consumed warmup32768,
model seed172 and all other scientific/resource parent settings. Copy existing
weights, optimizer moments, RNG, exposure and coverage; new parameter moments
start empty. Preserve the exact training data boundary and count added exposure.

Run each mode to4.125M first in a fresh directory, inspect finite learning and
resource receipts, then strictly resume each to4.5M in another fresh directory.
Intermediate loss differences are observations, not candidate-selection rules.
All modes receive500k added onsets unless a recorded engineering guard stops the
experiment. Training has the inherited14400-second cumulative per-mode bound
(including parent charge),6GiB footprint/RSS,128MiB swap-growth bound, and512MiB
per-segment outputs. Do not overwrite artifacts; only validated durable resume
is allowed. No network or external compute is needed.

Training stays pinned to `4a98c0230549baa19ad860c3a13caf301e0f1f71`.
Evaluate every mode on the common clean runtime
`9894761e8608ced818e94578a95649f5334b9d7b`; its only behavioral implementation
change batches the same exact support predicate. The equality evidence and
runtime transition are recorded below. It introduces no new input or support
restriction and changes no trained checkpoint.

Evaluate final modes on exactly the existing28 distinct groups, generation seeds
17/19/23, full-suffix NLL and native exports:252 outputs and84 suffix scores.
This now constitutes development data, not unseen confirmation. Reuse the pinned
condition manifest SHA
`7c757dbe90fcdaecb667af38e6b17b98a192595542d6aa62fcb742ea9e543bc7`.
Freeze a separate evaluator plan before executing. CPU1, at most10800seconds,
6GiB footprint/RSS,128MiB swap growth and4GiB complete experiment artifacts apply.
Each generation must complete and reparse exactly. Preserve every outcome,
including failures. Export star ratings remain separate from source ratings.

### Metrics, guards and interpretation

Primary: union of same-lane head-to-head or latest-release-to-head intervals
below40ms, counted once per suffix head, divided by supplied suffix onsets and
multiplied by1000. Average three generation seeds per group, then28 groups.
Report both components, raw counts, minima, run lengths and all per-group values.
The unchanged4M baseline is167 events/rate1.044819 on84 outputs; it is descriptive,
while the paired4.5M `none` and `actions` are the causal controls. Bootstrap whole
groups10000 times with seed1207 for paired90% confidence intervals.

A promising consequence-feature result requires at least25% lower mean union
rate than each matched control and an upper paired90% interval below zero for
each comparison. Each individual component may increase by at most the larger
of10% or .025 per1000 onsets relative to continued baseline. Require per-source-
band NLL no worse than either control plus.05, all mechanical receipts valid,
and mean LN-head fraction, independent-LN-start count and taps-during-long-holds
count at least90% of continued baseline, with source onsets as fixed denominator
for counts. These scalar guards do not replace semantic inspection.

Replay the eight preceding queries and failing query from each frozen generated
3ms/23ms/37ms case as mechanistic secondary probes. Report full legal probability
mass on rows that make the next one or two required onsets unavoidably violate
the combined diagnostic, allowing earliest permitted releases and single future
TAPs. These maximize future availability under the equal per-lane40ms diagnostic:
extra heads and new holds cannot improve it. Record when every legal current row
already forces the future burden; that position cannot discriminate a change in
model probabilities. Preserve such states and compare earlier avoidable choices.
Never hard-mask or score-penalize this set during native generation.

Qualitatively compare all three modes on the12 fixed LN cores (source indices
12–14,16–18,20–22,24–26, seed17), the first ordinary source in each band at early/
middle/late scopes (indices0/3/6/9, seed17), and the four longest late scopes
(indices15/19/23/27, seed17). Add each mode's worst combined-burden case, retaining
the selection rule. Mask identities until the28 prospective triple comparisons
and added risk judgments are sealed. Use the unchanged Beatmap Lens Foundation, renderer and confirmed
LN/stream examples. A scope review must distinguish independent LN coordination,
synchronized/isolated holds and organic TAP interplay. No prominent LN core may
lose independent organization without recording a failed retention guard.

Positive permits only a fresh bounded confirmation design; one initialization,
reused groups and the unresolved actual5–6-star coverage cannot complete the
overall goal. A lower NLL without native burden improvement, a gain shared by
the action-only control, collapsed LN participation, or shifted head-to-release
failure is negative or ambiguous for this mechanism. No `SUPPORTED` recommendation
is available without accepted-Card evidence. Any result is recorded as exploratory
with a REFINE decision and its precise limits.

## Implementation and execution freeze

The intervention is committed at clean product revision
`4a98c0230549baa19ad860c3a13caf301e0f1f71`. The baseline-to-intervention diff
contains only the declared model/config/fork changes, their tests and curated
runtime documentation. It does not change data sampling, support, timing inputs,
optimizer settings, inference temperature or parent weights. No Agent Note is
tracked in product history. The original dirty product worktree is untouched.

Selected local checks pass on CPU and available MPS. Initial consequence tests:
10 passed in3.62seconds. Fork/generation/Hydra/training checks:63 passed in42.02
seconds. After adding the current R1 crop, chunking and gradient checks, the
consequence/data/model/training/generation-run/package selection passes76 tests
and22 package-import subtests in47.19seconds. These counts overlap and must not
be summed. The final source change after these checks is docstring spacing only.
Commands use `uv run --offline --python 3.10 --extra mps --group dev pytest -q`
with the named owning files under `tests/research/bounded_typed_continuation/`
and `tests/test_package_layout.py`. The outgoing substantive diff and all direct
callers were inspected; `git diff --check` is clean. This is selected evidence,
not a full-repository test claim.

The real CPU1 probe covers eight identical extension draws and two updates per
mode. Baseline has2281104 parameters; both residual modes have2308016.

| Mode | Summed update seconds | Maximum footprint bytes | Maximum RSS bytes | Swap growth |
| --- | --- | --- | --- | --- |
| none | 3.370936 | 720471504 | 784826368 | 0 |
| actions | 4.167996 | 956991144 | 1021296640 | 0 |
| frontier | 4.169554 | 976668304 | 1040990208 | 0 |

The frontier update-time ratio is1.237, below the declared4x feasibility bound.
This tiny probe establishes engineering feasibility, not a stable throughput
estimate or generation quality. The first `none-v1` probe overlapped pytest; it
is preserved and excluded from this comparison. The cited `none-v2`, `actions-v1`
and `frontier-v1` probes ran serially after the test process terminated. No
candidate checkpoint is retained from these engineering updates.

The shared extended plan has23820 draws, preserving all21179 parent draws.
Its SHA is `767138e58c5288593be284b29a9333544e048468efb8d3f3aae2e42ece4c0fc6`.
The per-mode initial configs and exact six serial commands are pinned by
`execution-plan.json`, SHA
`0036fd35d1a15b09b3e719588dc448b9413663a38532846ed6232fdb5b74033d`.
Each command runs the artifact-owned `run_training.py` with one mode and
`4125k` or `4500k`. The driver verifies a clean source, its own digest, the plan
and config digests, and every initial-mode durable boundary before final
continuation. Fresh segment directories and inherited resource checks apply.
The implementation is ready for the explicitly authorized exploratory run;
training and generated-quality conclusions remain pending.

## Secondary diagnostic correction: Card revision2

Before evaluating any newly trained candidate, the release-aware feasibility
probe shows that the old head-only predecessor query is already too late in both
the23ms and37ms examples. At75089 and180304ms, every legal row forces a combined
head/release burden at the next required onset. Releasing a different lane there
would merely replace a short head/head interval with a short release/head
interval. These states cannot measure improvement in native decision quality.

One earlier query remains discriminating. At75065ms, mass forcing a burden by the
next two required onsets is .684850, while one-onset forcing mass is .205154.
The observed LN2 row has probability .257281. At180266ms, two-onset forcing mass
is .143707, one-onset mass .000153841, and the observed LN2 row has probability
.142955. At133271ms, both horizons give .0487574 for the all-four-LN choice.
The next133350ms state is already unavoidably forced. All six frozen chosen
log probabilities reproduce the original journals exactly. Artifact
`baseline-feasibility-probe.json` SHA:
`fff90f89b800d8aa95c3fb7a6714dc991c57092a0ac882d5db7da37c332c27fe`.

Revision2 changes only the mechanistic secondary procedure, expanding its fixed
history positions and making the one/two-onset distinction explicit. It changes
no primary metric, threshold, model feature, TRAIN draw, learning setting, native
sampling or semantic scope. The already running matched training continues under
the preserved revision1 execution freeze; the evaluator will identify revision2.
This is a declared diagnostic refinement based solely on the original4M outputs,
not selection on a trained candidate's evaluation outcome.

The readout explicitly supplies clocks at one future H. Its existing hand context
also receives the original16-candidate timing lookahead. This experiment does
not provide a learned multi-step planner. Failure against the controls would
reject this particular access/learning choice, not all candidate-conditioned
models or all longer-horizon formulations.

## Result Log: initial125k-per-mode engineering boundary

Owning Note/Card: this proposed Note, r1-action-consequence-v1 revision1 training
freeze; revision2 only refines the forthcoming secondary diagnostic. Accepted
revision: none. Baseline and clean intervention source remain the pinned
ca28511/4a98c02 full OIDs above. Python3.10.20/Torch2.11, CPU1 and model seed172
are unchanged. All three initial forks complete162 updates and exactly125000
added TRAIN onsets, then stop durably at the prescribed4.125M boundary.

| Mode | Segment seconds | Mean TRAIN NLL/onset | Maximum footprint bytes | Residual output norm |
| --- | --- | --- | --- | --- |
| none | 251.733575 | 1.93557888 | 869697144 | absent |
| actions | 309.533403 | 1.93558104 | 1170180064 | .00527125 |
| frontier | 318.361474 | 1.93559269 | 1169950688 | .00847939 |

All saved parameter tensors, losses and gradient norms are finite. Every update's
cursor, exposure, batch onset count and source coverage agree across modes.
Coverage is3007405 unique onsets,8893 charts and3167 groups. Swap growth is zero.
New output weights are nonzero, so the residual path receives optimization; its
small current norm and nearly identical teacher losses do not demonstrate any
native-quality improvement. Equal added data exposure is the comparison; elapsed
time includes small read-only artifact checks and is not an isolated throughput
benchmark.

| Mode | Durable checkpoint SHA-256 |
| --- | --- |
| none | `a8350d05b2d2185e0a8eaf92b0e62bbe222b44bd946c7243f0300f6ff537cc5d` |
| actions | `675162d716d87950f82ad1b849fafb1455cb671d629f66d8806724141d5f3c84` |
| frontier | `7a33241144e47932f6e9e32bd24068bd6643e15ee26ea2cbf42b298d2198196d` |

`initial-training-audit.json` SHA is
`13313289eae9ce7679ce80d84e6006f9cb4d35247e98dd085b4f4a5e2bb01054`.
All engineering guards pass. Strict final continuation starts with `none-4500k`;
the actions/frontier final segments remain pending until it terminates, preserving
the declared serial execution. All initial sessions have terminated and been
reaped. No final-mode checkpoint has yet been selected or evaluated.

The artifact-owned `evaluate.py` now implements all252 native outputs,84 suffix
scores, the paired group bootstrap and the primary/component/LN-retention guards.
Its new union counter reproduces the independently derived14 head,153 release
and167 union cases on all84 frozen original outputs, including the1.044819221932
mean rate. A synthetic overlap case verifies that one head meeting both conditions
is counted once. `evaluation-metric-check.json` SHA:
`1852aae4b866a1418599766d4937b7baaea1d83792c92fe96ab3b797f7051c34`.

The two-onset feasibility diagnostic passes synthetic examples distinguishing an
immediate forced restart from an avoidable two-onset occupancy trap. The masked
renderer resolves all28 exact prospective scopes and adds each mode's minimum-
interval case, with deterministic anonymous A/B/C assignment and a separate
private mapping. These drivers compile; their complete execution remains pending.
Freeze the evaluation-plan with their exact digests and final checkpoint hashes
after the three4.5M segments complete. Keep the private mapping unread until the
scope judgments are sealed. Training/evaluation outcomes and the overall goal
remain unresolved.

## Evaluation logic and original-source spacing reference

The secondary feasibility predicates agree with an independent exhaustive search
over legal future action rows for942 one/two-onset queries on small typed timing
fixtures, including fixed seed endpoints, unknown LN ends and intervening release
candidates. The search permits future TAP/LN chords and release subsets; it does
not reuse the greedy readiness computation. Synthetic comparison records also
verify that identical candidates fail the improvement threshold, while a positive
arithmetic control is rejected after either an NLL regression or LN collapse.
This tests evaluator logic only. `evaluation-logic-check.json` SHA:
`7af1dfb9417a6437bc0a6e315b473c361a19f5b635c57fbff4b4bd93b218ac2f`.

On the original28 source maps, the same suffix diagnostic counts zero short
head/head intervals and six release/head intervals. All six are31ms and occur
in source21. Four restart an outer-lane pair after94ms between heads; two are
single TAP restarts after125ms between heads. At those heads zero or one other
lane is held. This is descriptive context, not an automatic negative label or a
claim that every original source is expert-approved. The mean union rate is
.112309074573 per1000 supplied onsets, versus1.044819221932 for the84 fixed4M
outputs after averaging three seeds per source: approximately9.30x. The unequal
raw sample counts must not be compared without that normalization.

`source-spacing-reference.json` SHA is
`52a19219221969033211934a328b5fbd82f39d6b716a7594adb31d9260837c0b`.
The original cases reinforce why the40ms counter is a diagnostic proxy rather
than a universal quality boundary. No hard gap limit or source-label change is
introduced. The complete episode, LN articulation, concurrent occupancy and
surrounding TAP organization remain necessary for semantic judgment.

## Common evaluation runtime: Card revision3

All three learning trajectories remain on the immutable4a98c02 implementation,
same checkpoints, draws and configurations. The scalar exact-support enumeration
was independently identified as a large host-side cost. Its batched equivalent
is committed and tested at9894761; implementation and evidence belong to
2026-09-20-bounded-support-throughput.

Besides exhaustive Boolean and CPU/MPS likelihood/gradient/native tests, two
real CPU Adam updates preserve every parameter, optimizer moment/step and RNG
bit-exactly. A real generation check reproduces source20/seed17 and
source26/seed23 from the original4M checkpoint on9894761. Complete exported osu!,
row journals and decision journals are byte-identical to their frozen prior
counterparts. The second case covers the1050.064-second continuation. Thus no
quality gain is inferred from these repeats; they establish runtime compatibility.
Native compatibility report SHA:
`71495ffdd6ccca4775f08a9c95899c86cee5fe30802af12b8dc3d49876bd0521`.

Revision3 changes only the common evaluation runtime source. The evaluator now
records training-source4a98c02 and evaluation-source9894761 separately. All primary
metrics, thresholds, native sampling, model features, exposure pairing, selected
groups/seeds and semantic scopes remain unchanged. The original evaluator and
probe drivers are preserved as `evaluate-before-batched-runtime.py` and
`probe-before-batched-runtime.py`. AST equality confirms the primary burden and
comparison functions are unchanged. No newly trained candidate has yet been
evaluated. This declared behavior-neutral transition is authorized by the
user's broad optimization request; the Card remains proposed.

Run evaluation/probing/rendering from the clean `codex/bounded-support-batch`
worktree, invoking the drivers in the original action-consequence artifact owner.
Do not move or modify the active training worktree or reinterpret an old-source
recovery segment. `runtime-transition.json` SHA:
`f2c7b3be3f1350864a0839c4ffa6ea9273a06b1c847e27ce03d74f953eee8056`.
Current evaluator/probe/renderer SHA values are
`4420a84e658238fb6f1c09b89f16b0f0489e083c7187a58f3273eeaf516fe1c4`,
`6c05a51b84c5fadb447a202c5d9dd6b5ddfcf9ad0299f08290b98ce4be4e1ba6`
and `5094f858bc21a826c65d510b4c1e22c0c68ff0dbec9c6cbf9876c0c329e61952`.
The final evaluation-plan still awaits the frontier4.5M checkpoint; no evaluation
or masked semantic judgment has started.

## Result Log: complete matched4.5M training and evaluator freeze

All six training segments terminate at their declared boundaries on clean
4a98c0230549baa19ad860c3a13caf301e0f1f71. Each mode adds500000 onsets in661 updates
and consumes the identical draw/cursor sequence. Final coverage is3206045 unique
onsets,9178 charts and3168 groups for every mode. Every final parameter, loss
and gradient norm is finite; checkpoint journal boundaries and parent result
digests verify. No update is discarded. All original execution sessions are
terminal and reaped.

| Mode | Added measured seconds | Added-exposure mean TRAIN NLL | Peak footprint bytes | Final residual output norm |
| --- | --- | --- | --- | --- |
| none | 997.954619 | 1.92328633 | 869697144 | absent |
| actions | 1257.840447 | 1.92329320 | 1170180064 | .00859360 |
| frontier | 1272.945745 | 1.92322362 | 1169950688 | .06719391 |

Swap growth is zero throughout. Measured cumulative charges including the shared
parent are4108.352111/4368.237939/4383.343237seconds. Added exposure and scientific
settings match; wall time is neither matched compute nor an isolated benchmark.
The small teacher-loss differences and learned residual weights do not decide
the native-quality question.

| Mode | Final checkpoint SHA-256 |
| --- | --- |
| none | `2c452a954b57f0b412a0806dead0adffe11b227a22955f4fb9920d0fdf03d288` |
| actions | `8c04110f56736507afcc3df8554c402ed1ca46a11c756d56a111c6d927781317` |
| frontier | `72c96c58175fcf52ba654f6ccb2a565b8254e9f911d50db69539bac6f8081126` |

`final-training-audit.json` SHA:
`39308a13ad18fc639f92f93425e879099dc9b9528e1437b7f76d9c30788950ee`.
The complete evaluator plan is now frozen at SHA
`f78d68bddb4de3c104df21ea9dcfc738c53d4a1866d4e69e23af80232374c741`.
It pins all final checkpoints, revision3, source9894761, original4a98c02 training
provenance, all driver digests,28 groups, seeds17/19/23,252 outputs and84 suffix
scores. Execute the secondary frozen-history probe and full evaluator under that
plan; then render and judge the masked scopes before revealing identities.
Generated quality, all regression guards and the overall goal remain pending.

## Secondary fixed-history result

The revision3 diagnostic completes108 queries in12.534seconds: nine positions
around each of three fixed histories, scored by original4M and all three4.5M
candidates. Every original4M chosen log probability still equals its old journal
exactly under the common batched runtime. Report SHA:
`513fdbbf5eaae7c619c3488eeb84bfff05d57a016626d46ccab89ee55760caf4`.

| Avoidable predecessor | 4M two-onset forcing mass | 4.5M none | 4.5M actions | 4.5M frontier |
| --- | --- | --- | --- | --- |
| 75065ms before23ms case | .684850 | .674005 | .673943 | .666298 |
| 180266ms before37ms case | .143707 | .426304 | .426138 | .431860 |
| 133271ms before3ms case | .048757 | .015656 | .015652 | .013830 |

The feature branch makes small, mixed changes relative to continued learning.
Additional exposure improves the all-four-LN example but increases the second
example's conditional forcing mass substantially in every mode. These are fixed
old generated histories; the probability of visiting those histories can change.
Therefore this table establishes neither native population improvement nor
regression. It does not select a checkpoint or alter the predefined comparison.
The full252-output evaluator is running unchanged on9894761. Its outcome and
the masked semantic evaluation remain pending.


## Full native comparison: primary gate not met

The revision3 evaluation completes all252 full continuations and84 suffix scores
on the common9894761 runtime in2464.104808seconds. Every native export reparses
exactly and every mechanical receipt passes. Independent audit verifies all
osu!, action-row and decision digests, unique identities and finite scores.
Peak RSS is669925376bytes, footprint543622632bytes and swap growth zero.

| Comparison | Paired mean rate difference | Relative reduction | Paired90% interval |
| --- | --- | --- | --- |
| frontier minus none | -.595222751 | 9.1826% | [-1.672207428, .265617081] |
| frontier minus actions | -.905202972 | 13.3274% | [-2.633005490, .285602155] |

Both primary comparisons fail the predefined25% reduction and negative upper
interval requirements. All numeric regression guards pass. Mean union rates per
1000 supplied suffix H are6.482074836/6.792055057/5.886852085 for
none/actions/frontier; raw counts are1269/1351/1140. Head/head components are
21/21/23 and release/head components1248/1330/1118; one frontier event qualifies
for both and is counted only once in the union. Minima are12/12/3ms. This is a
failed quantitative primary, not evidence that the feature design is effective.

All modes increase LN participation substantially relative to the original4M
checkpoint. Equal-group/seed mean LN fractions are.489053111/.483170115/.530931723;
4M was.180447819. Pooled note-weighted LN fractions differ from these macro means:
4M was.133641613 and continued none is.564090348. These aggregations must not be
interchanged. The three models have five distinct groups with some actual5–6-star
outputs each, but ranges extend from about1.74 to7.02/7.02/7.24 stars. This is
neither stable requested difficulty nor proof of playable high-difficulty quality.

Below40ms release/head timing is a diagnostic locator. A release followed by a
head need not have the same burden as two heads, and higher LN participation
changes its opportunity count. Six31ms release/head events also occur in the
source references. Exact action, surrounding held lanes and longer organization
must determine whether a located event is a material concern. The counter is not
a calibrated demand metric and will not become a decoder rule on this evidence.

Readout SHA: `b7e825369e6fe8065184406868bcf5f07c68593a24a841b192577d4b9cef1ac1`.
Comparison SHA: `0bcfa82343203b8af1d19d7ffb0123ffaf6bf18c408e73d894d5318f53938231`.
Independent audit SHA: `704f76774e792eba186addb6a345bbcb405a920913afc7996e5b28e5be00d823`.

The complete anonymous packet contains31 triple comparisons and717 canonical
pages, manifest SHA
`a599407b10d002d1e14443d3fc0316ac076584289d2de6edefa53ae5d9de7d0d`.
The private label map remains unread. Review has begun with ln-12 only; its three
scope judgments are sealed in `masked-judgments-v1/ln-12.json`, SHA
`cd5684e82a8a0ff9764fd8a03ddf84db1b1282665cd0f6aa6080165ff019e9eb`.
They retain independent LN organization with different TAP/hold tradeoffs, not a
quality winner. All other scopes remain unreviewed. Finish the declared masked
review before revealing labels or making a semantic retention conclusion.
The Card stays proposed, accepted revision none; overall goal remains active.


## Masked semantic judgments sealed

All31 declared triple comparisons are now inspected:28 prospective scopes and
three risk selections,93 variant readings and717 complete canonical time panels.
Exact action records support the cited hold endpoints, handoffs and local timing
concerns. Two critical risk pages were also inspected at original1100x900
resolution. The review distinguishes LN presence, ordinal salience, confidence,
TAP/stream organization, scoped quality concerns and unreviewed tags. It is one
agent's gold-calibrated interpretation, not human playtesting or new gold.
Previously visible numeric summaries can make some variants recognizable.

The private model mapping remains unread at this seal. Judgment manifest
`masked-judgments-v1/manifest.json` SHA:
`695e283beca32767f09c3e613b14689e140bc2d43f028853885bee6bc39294cf`.
It pins every individual judgment and the renderer manifest. All action, montage
and canonical-page digests were reverified. The immutable recorder SHA is
`cb965ea948905eb669678383a5c2b67ec7c1ba570845d8c593eb819bfce694d8`.

Risk-01 and risk-03 have identical full action payloads for each anonymous label
once only the masked source identifier is removed. Both were inspected, but
must be deduplicated as one risk context in interpretation. They are not two
independent successes or failures. The other risk context contains a3ms
same-lane release/head restart under continuing holds. Strong LN organization
can coexist with such localized exact-action concerns.

Commit this seal before reading the private map. Next aggregate the declared LN
retention comparison and scoped tradeoffs, preserving uncertainties and duplicate
risk provenance. No overall quality or successful primary result is claimed.
