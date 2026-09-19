# Agent Note: Can candidate action consequences improve R1 occupancy decisions?

Note ID: 2026-09-20-r1-action-consequence
Status: proposed
Kind: research
Created: 2026-09-20
Updated: 2026-09-20
Product revision: ca28511cd69ccd0566bdb42be52e4651fb236369
Scope: R1 candidate-row residual, matched continuation controls, native burden and LN retention on the existing 28-group development cohort
Related: 2026-09-20-r1-difficulty-ln-longform; 2026-09-19-bounded-typed-r1-burden-mechanism; 2026-09-19-r1-additional-exposure

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

Card ID: r1-action-consequence-v1. Revision: 1. Accepted revision: none.
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

Replay frozen generated histories from the3ms/23ms/37ms cases as mechanistic
secondary probes. Report full legal probability mass on predecessor rows that
force the next required onset into the combined diagnostic, including the
intervening-release case; never hard-mask this set during generation.

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
