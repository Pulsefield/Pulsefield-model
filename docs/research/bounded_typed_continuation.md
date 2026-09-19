# Bounded typed-time continuation

This experiment compares three audio-free continuation tasks using a common
finite-context encoder and complete exact gameplay state. Its purpose is to
separate inadequate learning of the original task, the benefit of typed timing,
and the benefit of deciding complete note objects at their onsets. It remains
a research baseline; generated playability must be evaluated independently of
likelihood and mechanical correctness.

The implementation provides exact execution contracts in
[`contract.py`](../../src/pulsefield_model/research/bounded_typed_continuation/contract.py)
and a dense/cached finite content encoder in
[`temporal.py`](../../src/pulsefield_model/research/bounded_typed_continuation/temporal.py).
[`features.py`](../../src/pulsefield_model/research/bounded_typed_continuation/features.py)
owns permitted exact-state and timing inputs;
[`model.py`](../../src/pulsefield_model/research/bounded_typed_continuation/model.py)
implements joint decisions and dependent endpoint likelihoods and sampling.
[`data.py`](../../src/pulsefield_model/research/bounded_typed_continuation/data.py)
projects source labels into bounded, batched teacher-forced training windows.
[`generation.py`](../../src/pulsefield_model/research/bounded_typed_continuation/generation.py)
samples the native tasks and rebuilds bounded caches from raw checkpoints;
[`verification.py`](../../src/pulsefield_model/research/bounded_typed_continuation/verification.py)
independently checks completed outputs against the external condition and plans.
The packaged `bounded_typed_smoke` configuration provides a bounded learning-check
runner. A 16-chart learning check and paired R1 corpus trajectories through four
million source-onset exposures have run. Additional R1 training improves held-out
likelihood and the inspected local burden failures while retaining some independent
LN organization. Long-form quality, difficulty consistency and the complete range
of LN/tap expression still require broader confirmation.

## Conditions and prediction tasks

R is a strictly increasing sequence of raw source event-union times. Typed
tasks also receive H, the subset requiring at least one note head. The true
terminal time is the last R candidate. None of the arms receives suffix head
columns, head counts, TAP/LN labels, or source LN pairings as external input.
H and the construction of R are oracle information; this stage does not infer
timing from audio or test a general negative-candidate grid.

| Arm | Seed and timing information | Stochastic decisions |
| --- | --- | --- |
| R0 | Physical-row seed and untyped R | A nonempty four-lane action row at every candidate |
| R1 | Complete-object seed, R and H | Required head rows on H; releases or no event elsewhere |
| O1 | Same external conditions as R1 | A four-lane onset group, then a separate endpoint for each new LN |

The minimum production seed contains the complete row reaching at least 30
original note heads. Typed seeds include the endpoints of at most four LNs
still open at that boundary. This information is shared by R1 and O1. R0 does
not receive those future endpoints. The exact execution primitive also accepts
short synthetic seeds for mechanical tests; it does not choose the production
seed size.

R0/R1 lane actions are EMPTY, TAP, LN_START and LN_CLOSE. EMPTY holds an occupied
lane. O1 onset decisions are NO_HEAD, TAP and LN, with at least one head on H.
Each new O1 LN immediately chooses its own strictly future R endpoint. No same-
lane overlap or equal-time close/restart is allowed. Exact closes at the real
terminal are mandatory; a training boundary is never a terminal.

R1 knows only seed LN commitments. Endpoints for its own new LNs remain unknown
until it chooses a release. O1 knows every prior endpoint decision. A known end
cannot be changed to free a lane; an unknown rowwise LN can be released at an
available candidate. Support excludes any decision that leaves no possible
way to make at least one lane free strictly before the next required onset.

At non-onset candidates O1 executes due releases, or creates no event. R1 may
choose release subsets for unknown LNs while executing fixed seed releases;
all EMPTY is legal when it leaves future onset feasibility intact. When all
lanes are already closed, a non-onset R1 candidate has only the no-event choice.
Neither mechanism establishes learned abstention over an independently
constructed candidate grid: original R contains no all-empty source rows.

Candidate position and materialized-row position are separate. No-event advances
the candidate cursor without changing exact physical clocks or writing learned
content. A schedule may finish on an unused terminal candidate after its last
physical action; completion still requires every LN closed.

## Exact state and bounded learned context

Exact state retains real LN starts, current occupancy, last attack/release clocks,
first and last physical times, counts and the last row. Known endpoints remain
separate from unknown endpoints of open R1/R0 LNs. Crossing a crop boundary does
not reset any of these facts. Current state and prior known plans must directly
reach the head/type predictor, not only a legality mask or an endpoint head.

The common starting encoder uses shared hand coordinates and one causal width-
three convolution at each dilation 1, 2, 4, 8, 16, 32, 64, 128. Its content dependency is
exactly `1 + 2 * sum(dilations) = 511` materialized tokens. Encoding the preceding
physical gap of the oldest relevant token needs one additional timestamp, so
the raw history envelope is 512 physical rows. Current complete exact facts and
external time conditions are separate inputs to the prediction query.

This is a concrete finite-context implementation choice, motivated by the
auditable dependency boundary and batched computation of temporal convolution.
The [TCN study](https://arxiv.org/abs/1803.01271) provides a generic sequence-model
analogue, not evidence of mania quality. [Transformer-XL](https://arxiv.org/abs/1901.02860)
uses segment recurrence and therefore represents a different context contract.

Raw content tokens represent committed actions and permitted previously chosen
object endpoints. Current actions are predicted before their content is written.
Only R1 seed objects and O1 objects may expose their already committed future
ends; newly born R1 LN ends are unknown. Full source labels cannot be inserted
into R1 raw history retrospectively before their physical release.

Dense training re-encodes raw bounded history with current parameters. Gradients
may flow through that finite prefix; there is no automatic 64-row detach. No
learned carry crosses an optimizer update. Cached inference is tied to one model
and parameter version, and must match dense execution and cropped recomputation.
Durable recovery can rebuild learned state from raw history and exact facts.

BOS and TRUNCATED are distinct boundaries. Padding is outside a contiguous
materialized sequence; it cannot stand in for skipped timing candidates.
Tests hold exact facts and external timing fixed while changing action tokens
older than the declared field. Such changes must not affect the learned output.
The tests also check parameter gradients under crop recomputation, so matching
forward logits alone is insufficient.

Known timing features may look ahead without an action-causal mask. All arms
can read R; only typed arms can read H. Use full supplied timing for nearby
offsets, future gap descriptors and multiscale counts, independent of training
window endpoints. Report how 512 physical rows translate into seconds across
density strata before interpreting this as musical or phrase-scale context.

## Object endpoint probability

The object probability is the head-group probability times a joint
endpoint probability. Endpoints are decided sequentially within a row; each
factor can read earlier factors' already chosen endpoints. Use an equal mixture
of lane order 0, 1, 2, 3 and its mirror 3, 2, 1, 0, and marginalize that mixture for
likelihood. Sampling an order and following its factors samples the same model.
The full mirrored probability needs a test; shared hand weights alone are not
enough.

Every factor is normalized over endpoints admitting at least one completion.
If another lane is already free, another chosen end is early enough, or a
remaining factor can make a lane free, the current factor retains every future
R candidate. Otherwise it must choose a candidate strictly before the next H.
The legal candidate set is a contiguous half-open R-index interval, exposed by
`endpoint_bounds`; it has no arbitrary 16/128-event cap.

This is a normalized autoregressive joint defined by local completion masks.
It is not an arbitrary unmasked joint subsequently conditioned on feasibility.
That latter normalizer can be combinatorial when endpoint factors depend on
earlier endpoint choices. Training and generation must use the former same
factor masks and normalizers; the independent-endpoint conditioning formula
from the earlier frozen probe does not apply to dependent factors.

Full future candidate scoring requires exact chunked logsumexp and a bounded
backward strategy. The pointer packs ragged factor/candidate pairs into a fixed
candidate budget, then recomputes each block's features and activations during
backward. Only factor contexts and the small logsumexp reduction graph persist.
Tests compare likelihoods and all gradients with dense computation on CPU/MPS,
and compare retained tensor storage with and without recomputation. Rare endpoints
beyond a target window remain supervised; the window end does not truncate support.

### Candidate-conditioned availability experiment

`model.endpoint_availability` selects `none` (the original pointer), `zero` or
`commitment`; the latter two are O1-only. Both add the same small residual scorer
over encoded context and ordinary candidate features. `zero` fixes its additional
availability inputs to zero as a capacity control. `commitment` supplies derived
partial-plan facts. The final residual layer starts at zero, preserving the
original conditional probabilities when existing weights are copied.

For each candidate endpoint, the new inputs describe the future interval during
which that LN and at least zero, one, two or three other known LNs remain held.
Each interval supplies `log1p(value)/8` transforms of its H count, summed inverse
H gaps in reciprocal seconds, and elapsed seconds. Two further inputs count known
other holds and unassigned current LN factors, each divided by three. Occupancy
includes an H exactly at a release because same-time close/restart is forbidden.
The first H has no preceding gap; later gap weights use the previous supplied H.

These fourteen inputs use R/H, earlier object commitments and earlier factors of
the current endpoint order. Later within-row endpoints remain unknown. They do
not forecast objects that will be generated afterward, reveal source suffix LN
labels, constrain durations, or impose a burden penalty. Full endpoint support,
normalization and the mirror-order mixture remain unchanged. Compare the two
residual modes with continued training of `none` before attributing any gain to
the new access path. The feature's generation-quality benefit is not established.

## Source projection and leakage checks

Source admission retains the existing lossless event-row format and verifies
its digest. Compact integer indexes recover exact lane clocks and LN starts at
any target position without replaying neural state from BOS. No learned values
survive an optimizer update. These indexes and all original endpoint labels stay
in the supervision owner, outside the predictor API.

An interval selects actual post-seed source onsets. It includes intervening
release-only rows and extends to immediately before the following unselected
onset, or to the real chart end. The first interval also includes any releases
between the seed and first suffix onset. The same interval selection applies to
all arms. O1 forced releases contribute content but no stochastic likelihood.
Loss sums every relevant factor and divides by the actual selected onset count.

Training tokens use the preceding physical gap and new plans allowed by their
arm. Current query features contain exact lane/hand clocks, occupancy, previous
actions, counts and known remaining hold durations. Timing-only features include
16 upcoming candidates, their gaps/roles, counts within 0.25/1/4/16/64 seconds,
and the next gaps of at least 2/8/32 seconds. Only typed arms expose H. Timestamp
differences are computed in float64 on CPU before conversion to model dtype.

Leakage tests change suffix LN pairings while preserving R/H and the observed
prefix: R1 content/query features must stay identical. O1 may read a prior plan,
but changing its current endpoint label cannot change the current head decision.
Other tests compare indexed exact state against full replay and compare bounded
and BOS loss/parameter gradients with an LN older than the learned context.

## Native generation and recovery

`Rollout.from_seed` accepts the model, R/H timing, complete physical seed rows and
the permitted crossing seed endpoints. It accepts no source suffix labels. Typed
seed objects already closed inside the prefix are paired from that prefix; open
ones use the explicit endpoint condition. Native sampling uses temperature one
and the trained feasible distributions. Deterministic candidates consume no RNG.
Skipped candidates change neither the physical clocks nor the learned history.

The CPU generator owns all random draws, including endpoint order selection and
full-support Gumbel sampling. A scored O1 decision records the head probability
and marginalized endpoint probability; requesting unscored endpoints explicitly
returns an unavailable endpoint log probability, not a substitute path score.

A durable rollout snapshot retains exact facts, known obligations, CPU RNG and
the most recent 511 raw physical events. Each raw event includes its predecessor
timestamp and permitted plans chosen when that event was committed. No learned
buffers are serialized. Restoration checks timing/configuration/parameter-byte
identities and replays those bounded raw events using the same online kernels.
This includes the extra timestamp needed for the 512-row raw dependency envelope.
Parameter updates invalidate a live rollout. A failed step leaves its exact and
learned state uncommitted, but callers must restore RNG from a durable boundary
before retrying if sampling had begun.

Synthetic CPU/MPS tests compare uninterrupted and restored rows, plans, RNG and
probabilities. A full default-width, eight-level CPU test also restores after the
511-token range has discarded three still-held seed heads. Separate full-output
verification checks R/H coverage, unchanged seed, occupancy, terminal closure and
every sampled O1 endpoint without reusing scheduler feasibility masks. Exported
osu! files are reparsed into exact rows in the tests. These checks establish
mechanical behavior on the fixtures. Further CPU/MPS tests force source choices
through the actual native commit path: exact facts, raw features and pre-decision
probabilities match dense teacher-forced training across held LNs, context expiry
and a long gap. This rules out those tested train/inference projection mismatches;
it does not establish stability on generated histories.

### Portable condition and generation commands

The generator accepts a standalone JSON condition and a digest-pinned model
checkpoint. It needs no corpus catalog, split manifest or source suffix actions.
The optional source preparation command derives the supplied timing and seed
from a strictly admitted native 4K osu! file. By default it takes complete rows
through the thirtieth note head, including every head on that final seed row.
The source must contain that many heads and a remaining candidate suffix.

The condition format is `bounded-typed/condition-v1`. `timing.times_ms` contains
strictly increasing finite candidate times. `timing.onsets` is a same-length
boolean H mask for R1/O1 and null for R0. `seed_rows` must be a nonempty, complete
physical prefix of those candidates. Each row contains `time_ms` and four
`actions`: 0 EMPTY, 1 TAP, 2 LN_START, 3 LN_CLOSE. `crossing_ends` has four
lane-ordered entries. Each typed-arm seed LN still held at the boundary requires
its zero-based release candidate index; other entries are null. All four entries
are null for R0. The condition has no suffix lanes, note types or suffix-born LN
endpoints. Unknown fields and inconsistent seed/role/endpoint assignments fail
validation. Condition/source inputs are limited to 64 MiB; a condition can contain
at most 250,000 timing candidates.

For example, this valid R1 condition starts with an LN on lane zero, requires
its release at candidate one and a new head somewhere at candidate two:

```json
{
  "format": "bounded-typed/condition-v1",
  "arm": "r1",
  "timing": {"times_ms": [0, 100, 200, 300], "onsets": [true, false, true, false]},
  "seed_rows": [{"time_ms": 0, "actions": [2, 0, 0, 0]}],
  "crossing_ends": [1, null, null, null]
}
```

These commands use the Mac dependency extra even when generation runs on CPU.
Replace paths and uppercase digest placeholders with the actual files and their
full lowercase SHA-256 values. Preparation prints the resulting condition digest.
Each preparation output file and generation output directory must be fresh.

```sh
uv run --python 3.10 --extra mps python -m pulsefield_model.research.bounded_typed_continuation.condition_hydra \
  source_file=/path/to/source.osu source_sha256=SOURCE_SHA256 \
  output_file=artifacts/bounded-typed-continuation/condition.json arm=r1 seed_notes=30

uv run --python 3.10 --extra mps python -m pulsefield_model.research.bounded_typed_continuation.generate_hydra \
  checkpoint_file=/path/to/checkpoint.pt checkpoint_sha256=CHECKPOINT_SHA256 \
  condition_file=artifacts/bounded-typed-continuation/condition.json condition_sha256=CONDITION_SHA256 \
  output_dir=artifacts/bounded-typed-continuation/generated-example device=cpu cpu_threads=1 seed=17
```

Supported model formats are `bounded-typed/corpus-training-v1` and
`bounded-typed/learning-check-v1`; their arm must match the condition. The runner
loads the saved architecture and strict parameter state, within the supported
128-hidden/eight-level/four-expansion/16-coupling-rank envelope. It retains native
temperature-one sampling. `candidate_budget` bounds endpoint scoring work,
not endpoint support; `score_endpoints=true` also computes marginalized endpoint
log probabilities. CPU generation with one thread is the default. `--help` and
`--cfg job` expose the packaged Hydra settings; `--cfg job` is inspection rather
than semantic validation. Execution requires a clean, committed package checkout.

The optional pair `presentation_source=/path/to/source.osu` and
`presentation_sha256=SOURCE_SHA256` copies only playback metadata, timing/SV
points and the audio filename into the exported header. It assigns new beatmap
identifiers. These header values never enter prediction. Audio is neither read
nor bundled; playback needs the matching original audio/mapset. Omitting the
pair produces a generic header suitable for structural inspection.

Each run writes its resolved/typed settings, input condition, model/environment
provenance, physical rows, decisions, resource observations and `checkpoint.pt`.
Completed runs independently verify the full output, export `generated.osu`,
strictly reparse it into the same physical rows and write `result.json` with file
digests. `max_seconds` or `stop_after_candidate` pauses at a consistent candidate
boundary; a paused run has a checkpoint and result but no partial osu! export.
The stopping cursor is absolute and zero-based: a cursor of 512 means candidates
0 through 511 have been processed. Time is checked between generation steps;
model loading, an individual step and final export can exceed the nominal limit.

To recover, repeat the generation command with a new `output_dir` and add
`resume_from=/path/to/parent/checkpoint.pt` plus `resume_sha256=RESUME_SHA256`, using
the parent's reported checkpoint digest. The model, condition, presentation,
source revision, device, CPU thread count, sampling seed, candidate budget and
endpoint-scoring setting must match. Paths can change when file bytes do not.
Resource limits, checkpoint cadence, time limit and stopping cursor may change;
the stopping cursor cannot precede the restored cursor. Use the same library and
hardware environment for exact numerical recovery.

Recovery verifies journal prefix digests, physically replays their decisions and
checks the saved exact state and bounded raw history. It then rebuilds only the
finite learned context. Durable prefixes are copied into the new output
directory; incomplete tails are excluded and the parent is untouched. A failure
propagates and records `failure.json` when the runner owns the output directory.
It leaves the last complete checkpoint available instead of saving a potentially
partially written step. CPU thread settings are restored on exit.

### Quality coverage required beyond mechanical verification

Playable continuation must remain coherent through a complete suffix, across
multiple source groups, generation seeds and difficulty levels. Independent 4K
coverage in the 2★–6★ range is required alongside harder maps. Record source and
generated star ratings separately with the calculator revision, mods and clock
rate fixed. A source's star band does not certify the generated chart's difficulty,
and matching star ratings do not establish human-like organization. Existing
ordinary/stress selection alone does not establish this difficulty coverage.

Inspect early, middle and late sections plus transitions, long gaps and the full
chart's development. Evaluate long LNs, complex independent LN control, short and
fragmented LN articulation, and their integration with taps. Durations need
physical-time and beat context; short LNs are not intrinsically defects, and long
or dense simultaneous holds do not by themselves establish complex organization.
Track degeneration, difficulty drift and variation across seeds without selecting
only successful excerpts. Use the Foundation and confirmed human examples for
multi-scale judgments. Mechanical validity, total LN proportion and a few strong
local sections cannot substitute for these quality and stability requirements.

### First native full-suffix diagnostic

At source `21475e65d773b7e7199accf0750de584d9f10ce9`, the three short-fit checkpoints
each continued all 16 TRAIN charts from their actual minimum-note seed through the
true end, using generation seed 17 and temperature one. All 48 outputs pass the
independent task/endpoint verifier and exact osu! export/reparse checks. Each arm's
first chart also restores after candidate 600 with identical RNG, exact state,
current content and all convolution buffers, then continues normally.

| Arm | New suffix LN heads / all suffix heads | LN fraction across individual outputs | Total generation time |
| --- | --- | --- | --- |
| O1 | 675 / 23,294 = 2.90% | 0.62–6.67% | 23.838 s |
| R1 | 5,555 / 26,010 = 21.36% | 8.43–46.76% | 29.997 s |
| R0 | 4,158 / 27,188 = 15.29% | 6.97–29.22% | 30.509 s |

This CPU run takes 86.470 seconds including loading, export and verification;
sampled RSS peaks at 309,084,160 bytes with no swap growth. Bounded raw recovery
takes about 0.30–0.32 seconds per checked chart. O1 samples 504,368 full-support
candidate pairs; its shorter generation time is coupled to its much lower LN
birth rate and is not an equal-work endpoint-cost benchmark. R1/O1 skip 1,060/1,669
unused candidates; R0 emits every candidate as required by its different task.

The free-running LN-use gap is substantial despite similar teacher-forced type
discrimination. It warrants investigation of native closed-loop behavior and
adequate training, rather than treating a good endpoint or type score as a quality
pass. Different LN ratios from a source are allowed; this one-seed TRAIN diagnostic
does not prove universal collapse or rank playability. The selected visual scopes
below provide limited Foundation-based inspection; new human comparisons and
independent group/initialization confirmation remain outstanding. Rapid-pair and overlap counts are descriptive
locators, not semantic labels or playability thresholds.

Readout SHA-256:
`17ed6ca391e680ee99a19298db8873260974eaba56edbd9df4368893999c5956`, under
`artifacts/bounded-typed-continuation/smoke-20260918-v1/native-generation-v2/`.
An earlier attempt stopped during playback-header resolution after the first
chart; resolving catalog-relative paths against their original worktree repairs
export. The repeated first generated row file is byte-identical to that attempt.

### Selected visual scopes and human calibration

The frozen Foundation and current human examples distinguish independent LN
control from simultaneous holding alone. A high-confidence positive at
167706–169445 ms on source `ece7388da533` contains repeated staggered starts and
releases across held lanes. A high-confidence negative at 313004–319158 ms on
source `713ef90e11c7` includes synchronized long holds without independent
interior actions. The latter has no substantive human comment; its label must
not be supplemented with an inherited agent rationale. Canonical workflow
projection yields 37 observations from eight unchanged source documents.

Two generated source scopes were inspected at playback rate one, including all
24 rendered context pages across the human references and O1/R1 outputs. These
are machine judgments calibrated to human examples, not new human labels or a
blinded preference test.

| Selected scope | O1 observation | R1 observation |
| --- | --- | --- |
| LN-rich source `bd120339738c`, 95864–103864 ms | 64 TAP and 3 LN heads; two LNs start/end together and the third is isolated. LN coordination absent, high confidence. | 19 TAP and 51 LN heads plus two entering holds. Repeated independent holding, attack and release roles characterize the core. LN coordination present/prominent, high confidence. |
| Dense source `00126e732bc4`, 158311–164311 ms | 181 TAP and 1 LN heads; dense chord re-attacks persist. LN coordination absent, high confidence. | 170 TAP and 35 LN heads; an independent LN episode follows dense TAP activity. LN coordination present/supporting, medium confidence. |

The LN-rich scope was selected around the center of its supervised interval.
The dense scope was located post hoc by R1's longest rapid same-lane attack run:
seven column-zero taps at 161205, 161238, 161274, 161310, 161345, 161382 and
161417 ms, with other-column activity. This establishes a localized burden,
not a global Jack label or universal playability cutoff.

The dense core has 105 supplied onsets and 113 source heads, versus 182 O1 and
205 R1 generated heads. Its maximum count in one second is 30 onsets; the maximum
over the 16 supervised learning-check intervals is 15. Broader training can test
whether chord size adapts to this density, but the coverage gap does not prove
that more training will fix it. Other labels, R0 images, larger organization and
player playability remain unreviewed. Review SHA-256:
`7063b8d246fe3d635632bd55dc8a08c46d3fcd2aa53bd8bc858baaa72f6a91d9`, in
`native-generation-v2/inspection-v1/review.json` under the readout owner above.

## Comparison and evaluation plan

### Initial 16-chart learning result

Clean source `a178bcfe2badaaea47ae9abce02f2494b8ff9643` was tested on 16 distinct
TRAIN groups, with 128 selected onsets per group. Each arm started from model seed
171 and used the same shuffle seed 271, 128 AdamW updates and two intervals per
update: 32,768 exposures over 2,048 distinct source onsets. The slice contains 828
LN and 2,541 TAP heads, independent multiple-LN endpoints, long gaps and crossing
holds. It deliberately tests pipeline learning, not population quality.

| Arm | Total factor NLL/onset, initial → final | Training time | Parameter count |
| --- | --- | --- | --- |
| R0 | 4.691517 → 2.794169 | 64.710 s | 2,281,104 |
| R1 | 4.409525 → 2.441500 | 65.250 s | 2,281,104 |
| O1 | 6.953991 → 2.592007 | 100.070 s | 2,355,835 |

These local-window costs show within-arm fitting progress; they do not rank the
three representations. O1's head NLL falls 4.166273→2.194355 and endpoint NLL per
born LN falls 6.895225→0.983565. All three satisfy support and finite-gradient checks.
Sampled RSS remains below 2.64 GiB and MPS driver memory below 0.96 GiB, with no
swap growth. O1 evaluates 21,913,664 candidate pairs during training, with backward
recomputation. Total run time including loading and initial/final evaluation is
112.186 s for O1, 72.297 s for R1 and 70.997 s for R0. This throughput applies only
to the selected 673–2288-row source charts. The seven intervals using the full
512-row raw envelope span 59.029–87.300 seconds, median 69.851 seconds; a corpus-wide
context-span distribution remains unmeasured.

The initial type gate required negative log probability on true LN_START lanes to
improve against random initialization. All three fail that rule: O1 1.030→1.449,
R1 1.119→1.339 and R0 1.128→1.642. A subsequent diagnostic identifies a limitation
of that positive-only comparison. LN prevalence among feasible lane/onset positions
is 11.07%, while initial mean LN probabilities are 30.83–34.25%. Correcting that
excessive prior can lower probabilities on positives while improving classification.

A read-only audit reproduces initial/final head losses and scores both classes.
O1's full binary LN NLL improves 0.484454→0.223365, versus 0.347905 for a fitted
constant prevalence. Its LN AUROC improves 0.621507→0.907881. Conditional TAP/LN
NLL at true head locations improves 0.697922→0.333056; even a baseline allowed each
interval's true LN fraction has weighted NLL 0.371819. Within-interval type AUROC
improves in all 12 intervals containing both types, with median 0.532→0.791.
R1/R0 also improve proper binary scores and within-interval discrimination.

The original gate remains failed, and this post-hoc diagnostic does not establish
a quality pass. It does show that describing these models as learning only
endpoints would omit evidence of type discrimination. Subsequent checks should
predefine proper all-class scores and within-interval discrimination, while
tracking positive rates and free-generation mode collapse separately. Native
sampled generation, independent groups and Foundation/human quality judgment
remain necessary.

Artifact identities: interval manifest SHA-256
`fe6090618154f2026e34ce5d432ddc2368d692cd50f0fc28d563a42e308e8fda`;
paired learning readout
`b76d1e742887eb22bb6e244229d3925e63e829584780917b80efd2479ee38594`;
post-hoc type readout
`1f56d656134486e325ab4b4cc8d2baee80015c267e3a9fc1403dcc923b8c1d43`.
They are local generated evidence under
`artifacts/bounded-typed-continuation/smoke-20260918-v1/`; the observations above
remain interpretable without those files. No annotation or human gold was changed.

### Learning-check execution and main comparison

The learning-check entrypoint is
`python -m pulsefield_model.research.bounded_typed_continuation.smoke_hydra`.
Use explicit `mps` dependencies on this Mac. Pin `interval_manifest`/`interval_sha256`,
`catalog_path`/`catalog_sha256`, `split_manifest`/`split_sha256`, `source_cache_dir`
and a fresh `output_dir` through Hydra overrides. Select `model.arm=R0`, `R1` or
`O1`; packaged defaults select O1. The runner requires clean committed source,
checks TRAIN identities before loading rows, writes resolved and projected config,
and refuses an existing output directory. Its report-point checkpoint stores
parameters, optimizer, source exposure and draw/RNG state. Automatic resume is not
implemented in this small check.

`smoke_selection.select_intervals` can prepare a deterministic mechanical slice
from a pinned TRAIN census and catalog: 16 distinct groups, 128 target onsets each,
with TAP, mixed, LN-rich, independent-endpoint and long-gap examples. Source lengths
are restricted to 640–2400 physical rows for this first pointer exercise. The
manifest records every source and interval, crossing-hold counts and selection
facts. This deliberately selected slice does not estimate population quality.

The default check uses 128 AdamW updates, two intervals per update, a shared shuffle
seed and actual-onset normalization. It separately records observed TAP/LN lane
probabilities, complete row/head costs and endpoint costs; no auxiliary term uses
those diagnostics. Preparation, dense prefix/target forward, candidate scoring,
backward, optimizer steps, checkpointing and evaluation are recorded or included
in elapsed-time totals. The complete run has an 1800-second limit and explicit
driver/RSS, available-memory, swap-growth and output guards. These settings bound
a pipeline check; corpus training is configured separately below.

Use roughly matched, few-million-parameter common encoders, identical source
intervals and training initialization seeds, with all arms trained from scratch.
Use complete likelihood factors normalized by actual supervised source onset
count. Structural marginal auxiliary losses are omitted from this comparison.

R1 and O1 condition on the same external information and describe the same
complete chart space. Compare total complete-suffix code length, including head
choices, endpoint factors, order mixtures and feasibility normalizers, divided
by the same source onset count. Row-local mean NLL and endpoint-only mean NLL
remain incomparable. R0 is a different-conditioning practical comparison.

After mechanical/model checks, a 16-TRAIN-chart learning check must show actual
head/type learning as well as endpoint learning. It covers TAPs, simultaneous
LNs with different ends, long gaps and exact state at crop boundaries. Synthetic
long-lived anchors supplement real examples where observed corpus LN spans
are shorter than the full learned context.

The initial main screen uses 250k/1M/2M source-onset exposure checkpoints with
at most four training hours per arm, whichever comes first. Pin exact runnable
source, data intervals, optimizer and resource limits after the learning/resource
check. Report equal-exposure and equal-compute results separately, including
unique coverage, repeated exposures, physical rows, endpoint factors and recovery
costs. Prefix preparation and candidate scoring belong in the compute ledger.

### Shared corpus plan and resumable training

`corpus.create_plan` freezes a pinned TRAIN census, catalog/allocation and admitted
row-cache digests. It draws a group uniformly, then a chart uniformly within that
group. Target horizons are 128 or 256 source onsets with equal probability. An
explicit 12.5% seed-window stratum covers short deployment histories; other draws
choose a valid full-length window uniformly. Short sources use their actual
length. Draws are shortened at 250k/1M/2M exposures so every arm reaches exactly
the same boundaries. This distribution is not uniform over onsets or song time.

The corpus entrypoint is
`python -m pulsefield_model.research.bounded_typed_continuation.train_hydra`.
Supply `plan_file`, `plan_sha256`, `source_cache_dir` and a fresh `output_dir`.
Packaged defaults use CPU execution with one thread, four intervals per optimizer update, microbatches of two,
AdamW at 0.0003, weight decay 0.01, clipping at one and a linear warmup through
32,768 actual onset exposures. Gradients divide by the effective batch's actual
onset count even when its microbatches differ in length. Full-support endpoint
blocks retain the candidate budget and backward recomputation described above.
The source-index LRU is bounded at 64 charts and 256 MiB of conservative charges;
active microbatches can retain evicted charts, so process memory remains guarded.
On macOS, the runner also reads `TASK_VM_INFO_REV1.phys_footprint` directly and
enforces `footprint_limit_bytes` (default 6 GiB). Footprint and RSS overlap and
must not be added. Footprint is unavailable on other platforms; existing RSS,
device, available-memory and swap checks still apply there.

Each segment writes resolved/projected configuration, resource and training logs,
and safe-loadable model/optimizer/RNG/coverage checkpoints. `stop_after_checkpoint`
can pause at a declared plan checkpoint. `resume_from` continues the final durable
checkpoint in a fresh segment directory; it requires identical source and
scientific/resource settings, verifies the parent journal prefix, and preserves
the parent artifacts. No learned history is retained. Coverage counts the exact
union of supervised source onsets rather than the number of draws.

`fork_from`, `fork_sha256`, `fork_source_revision` and `fork_plan_file` provide a
separate, explicit initialization path for an audited source transition. All four
are required together and cannot be combined with `resume_from`. The parent must
have completed its plan with a finalized runtime ledger and no discarded updates.
The new plan must preserve every source pin, sampling setting, prior milestone
and old draw, then append further draws. Existing plans and parent outputs are
never edited. Scientific settings may change only the endpoint-availability mode;
the parent must use `none`. Ordinary resume retains exact source/config identity.

Fork initialization copies all existing weights, AdamW moments/steps and RNG,
retains cumulative exposure/coverage/metrics, and charges the parent's entire
measured compute time. Only residual parameters may be appended; their optimizer
state starts empty. Subsequent segments use ordinary resume with the fork fields
cleared. Each result links its immediate parent ledger, retaining the source and
checkpoint transition without relabeling the parent checkpoint. The runner's
source pin is an operator-declared audit boundary, not proof that arbitrary source
changes preserve behavior.

The four-hour compute limit includes plan loading, source preparation, forward,
backward, checkpointing and verified recovery. It is checked before each update;
an in-flight update and final publication can cross the deadline and their actual
duration is reported. A caught failure records its complete segment duration,
including work beyond its last checkpoint, which is charged on resume. An
unmeasured hard-killed parent cannot silently contribute a zero-cost recovery;
the runner requires a finalized parent runtime ledger.

LN diagnostics score both outcomes on feasible source-onset lanes, and conditional
TAP/LN at observed head locations where both types are feasible. They do not add
an auxiliary training loss. `evaluation.suffix_likelihood` sums all factors across
bounded chunks through the true chart end, paying each O1 endpoint exactly once
even when it falls beyond its chunk. This supplies a common complete-suffix R1/O1
score. CPU/MPS tests check microbatch gradients and exact pause/resume trajectories;
chunking tests check complete-suffix likelihood and diagnostics.

### Variable-input MPS memory and CPU execution

The first corpus attempt at source `411c8c29abad50a05cf3ceb90f20a20d93a321ed`
stops after 40 O1 updates / 30,295 onset exposures: global swap growth reaches
149,487,616 bytes against a 128 MiB guard. The last durable checkpoint contains
24,384 exposures. Total measured runtime is 102.380 s. Peak observed active MPS
memory is 0.50 GiB, driver memory 1.70 GiB and RSS 2.86 GiB; these counters do not
explain the process's complete footprint. R1/R0 do not start in that attempt.

A read-only gradient workload starts from that checkpoint and compares 64
microbatches on the same next 128 plan draws, without optimizer updates. Each
process uses full-support endpoint scoring and one CPU thread unless specified.
`vmmap -summary` measures physical footprint every eight steps. Cleanup drops
model/data references, collects garbage and releases unused MPS allocator memory.

| Workload | Time | Final footprint | After cleanup |
| --- | --- | --- | --- |
| MPS, fixed pair repeated | 24.629 s | 865.8 MiB | 605.8 MiB |
| MPS, varying pairs | 79.825 s | 9.6 GiB | 8.5 GiB |
| MPS, varying pairs, head likelihood only | 38.195 s | 5.4 GiB | 4.3 GiB |
| CPU, varying pairs, one thread | 28.275 s | 733.1 MiB | 733.1 MiB |
| CPU, varying pairs, four threads | 25.946 s | 721.1 MiB | 721.1 MiB |

The fixed-input MPS footprint changes only 11.9 MiB from step 8 to step 64;
varying-input footprint continues growing while post-backward active tensor
storage stays near 62–67 MB. Both the common path and endpoint path contribute.
This supports shape-related runtime retention, without establishing ownership by
a specific backend cache. RSS can fall while footprint grows, so an RSS-only
limit is insufficient. The direct Mach counter is checked against independent
`vmmap` output in a Darwin test, and a runner test verifies its stop behavior.

CPU is approximately 2.8 times faster than MPS for this varying-input workload
and avoids its observed footprint growth. Four threads are only 8.2% faster than
one in this single measurement. The corpus default therefore uses one CPU thread
before changing model or pointer geometry. This preserves the probability model
and leaves MPS optimization optional. These are bounded backward workloads; they do not measure full optimizer training
or generated quality. The CPU corpus trajectory below supplies separate training
evidence. Neither a backend-wide leak nor universal CPU superiority is claimed.

Evidence lives under
`artifacts/bounded-typed-continuation/corpus-20260918-v1/memory-probe-v1/`.
The varying MPS result SHA is
`f54554b96f570a624f8bc5ffce8629e561277d4b63ddd7ffcd0d24c332d003a4`;
one-thread CPU result SHA is
`d28d936b8c9b069f16d42b2d86dd766138ce3333b3b29898515f20a000f943c0`.

### Paired corpus learning through two million exposures

At source `1693d62ffaca04b2a6127d8e3a72d1988adf441f`, all arms complete the same
2,000,000 source-onset draws with model seed 171. Each covers 1,697,938 distinct
onsets, 6,315 charts and 3,078 TRAIN groups. Cumulative CPU training takes
2174.555/1501.429/1408.085 seconds for O1/R1/R0, with no swap growth or resource
failure. These measurements support Mac feasibility for this bounded setup.

The fixed development screen contains twelve ordinary and twelve stress groups,
three generation seeds per group and complete-suffix likelihoods. At each of
250k/1M/2M exposures all 216 generations pass mechanical/export/reparse checks.
O1/R1 pooled NLL falls from 2.291069/2.268465 to 1.906802/1.915834 nats per source
onset. At 2M the paired group-macro difference is -0.019613, with a group-bootstrap
95% interval [-0.049635, 0.011185]. This development result does not select a stable
winner. R0 conditions on different information, so its likelihood is not the same
conditional comparison.

Generation changes substantially with exposure. O1/R1/R0 quad attack counts rise
to 17684/19354/19047 at 1M, then fall to 223/104/193 at 2M with unchanged decoding.
Their 2M LN-head shares are 37.037%/27.651%/47.311%. In two preselected stress cores,
Foundation-calibrated machine inspection finds sustained independent LN control
for both O1 and R1 in the LN-rich case; both produce predominantly TAP arrangements
in the dense case. Different source LN proportions are allowed. Those scoped
judgments do not establish whole-chart playability or replace human preference.

A remaining O1 failure concerns allocation: 439 of its 498 same-lane attack pairs
below 40 ms occur when prior LN commitments leave only one available lane. A
selected case has eight attacks on that lane over 241 ms, while three earlier LNs
block the other lanes. The same timing permits distributed attacks in R1 and the
source. R1 has 28 such rapid pairs overall and R0 has 30. The 40 ms diagnostic is
a locator, not a semantic or universal playability threshold. It motivates testing
candidate-conditioned availability against further training and a capacity control.

The corpus plan SHA is
`a6e727d0bbfa414e7d20c742a18d84e8786f155b91629d285e7abf04348cd82f`.
The 2M development readout SHA is
`262979f1deb1069f0ff9b401f5c2d83b55b79f0de4051b770b57d6ac64e828f3`;
the occupancy readout SHA is
`c7f2886ebb54a5f5d3e1e9c2ddd4ee7778e28f38fe11aad9379c7c3632acafd8`.
Local evidence is under `artifacts/bounded-typed-continuation/corpus-20260918-v1/`;
the findings above do not require those generated files to interpret their scope.

Generation retains existing regression cases and adds separately sampled ordinary
and stress groups, with multiple seeds. Mechanical failures cannot be averaged
away. Judge recurrence burden, chord and LN mode coverage, independent hold/attack
relationships, transitions and cross-seed variation. Different reasonable source-
conditioned arrangements are allowed; neither copying source LN ratios nor merely
reducing the shortest attack interval is a success criterion. Foundation-based
machine inspection and current human calibration do not replace new blinded
human comparisons. Independent confirmation needs a second training initialization
and groups unused during model selection.

### Additional R1 exposure and the LN tradeoff

Two R1 initializations, 171 and 172, continue from two to four million source-onset
exposures at source `50dda55040f51a7afc9a13994b762f953fe3064d`. They retain the same
width-128, eight-level encoder, 511-token context, exact state, R/H conditions,
group/chart sampler, optimizer moments and native temperature-one decoding.
The intervention is additional training, with no type penalty, duration floor,
endpoint oracle or sampling filter. Both trajectories use the same draws and
cover 2,940,384 distinct onsets in 8,799 TRAIN charts and 3,167 groups at 4M.

The comparison reuses 24 development VAL groups with three generation seeds per
model. Likelihood is computed over each complete suffix and normalized by source
onsets. The table reports the unweighted mean of 24 per-chart values; uncertainty
comes from 2,000 paired group-bootstrap resamples and does not estimate variation
across model initializations.

| Initialization | Group-mean NLL/onset, 2M → 4M | Relative decrease | 95% interval for mean difference | Generated same-lane pairs below 40 ms, 2M → 4M |
| --- | --- | --- | --- | --- |
| 171 | 2.128667 → 2.028668 | 4.698% | [−0.151322, −0.057800] | 10 → 3 |
| 172 | 2.073766 → 2.001587 | 3.481% | [−0.099802, −0.043235] | 43 → 2 |

All 144 new outputs pass independent mechanics and exact osu! export/reparse.
Both models improve group-mean NLL in each source-star band [2,3), [3,4), [4,5)
and [5,6]. Eighteen groups occupy those bands; they were originally selected by
ordinary/stress descriptors rather than balanced difficulty sampling. Source and
generated stars are separate whole-chart measurements using the pinned local
20241007 calculator, native 4K, no mods and rate 1. A source's rating does not
establish the output's rating or playability. The 40-ms pair count locates possible
burden defects and is not a universal physical limit or a style label.

The improvement has an expressive tradeoff. Generated suffix LN-head fractions
fall from 25.342% to 11.857% for 171 and from 53.857% to 10.597% for 172. All
72 outputs per 4M model contain suffix-born LNs, but outputs containing an LN of
at least two seconds fall from 48 to 21 and from 34 to 23. Observed source LN
actions also receive worse mean negative log probability despite the lower total
NLL. Neither a lower LN share nor a physical duration quantile establishes better
or worse organization; this discrepancy requires inspection of the actual roles.

Six 16-second cores and two 64-second contexts per initialization were compared
with their 2M counterparts at generation seed 17. Machine judgments were frozen
before exposure-stage labels were revealed. Complete entering/exit context was
inspected across 416 canonical time-proportional pages, with exact action and
endpoint witnesses. The frozen Beatmap Lens V2 Foundation and six High-confidence
human Stream/LN examples supplied calibration; these examples are not human
judgments of the generated charts. Prior familiarity with some 2M outputs limits
the masking.

All sixteen 4M scopes were judged locally plausible. Both wider scopes per model
retain recognizable development, ordinary scopes retain definite moving
organization, and the LN-rich core retains definite independent LN control in
both models. Twelve comparisons are ties; three have a limited preference for
4M because it avoids a localized compressed repeat, and one prefers the stronger
independent LN development at 2M. In that last 171 core, coordination remains
present at 4M but changes from prominent to supporting. The 172 LN-rich core
retains prominent independent coordination. The marginal LN-coordination label
in the 172 4M dense core remains unresolved and is not needed to establish the
separate LN-rich result.

The three burden-unresolved 2M scopes contain isolated 26/35-ms same-lane events;
the corresponding 4M scopes avoid them while preserving plausible motion or
held articulation. This is evidence of improvement at those observed locations,
not proof that all generated histories avoid occupancy traps. Synchronized holds,
sequential short-LN handoffs and a single anchor with taps are distinguished from
independent multi-LN control. All can be useful articulation without receiving
the same Foundation label.

On the Apple M5 with 24 GiB memory, Python 3.10.20 and Torch 2.11.0, each extra
2M-exposure segment takes about 25 minutes using one CPU thread. The larger
sampled process footprint is 929 MB, with no swap growth. These are measured
operating points, not accelerator throughput claims. The complete 144-output
evaluation takes 420 seconds under its stated resource bounds.

The comparison's prospective numerical and scoped qualitative guards pass for
both initializations. It supports further investigation of 4M R1 as a practical
candidate while preserving the more expressive 2M alternatives. Reused groups,
single-seed semantic scopes and incomplete difficulty/LN-form coverage prevent
an overall quality claim. Additional exposure beyond 4M has not been evaluated
by this comparison.

Evidence lives under
`artifacts/bounded-typed-continuation/r1-exposure-20260919-v1/`:

- Evaluation readout SHA:
  `ab80ec8e351a3d537d1b387c37cdf41af7d76f9e9aa18544c7ac1102477007de`.
- Sealed masked judgments SHA:
  `a2023a2c8f6579d989fba1f684eb609d9d04470a4971d5be0c70b9b5bf83a6be`.
- Revealed comparison SHA:
  `960d1c228eae09a120d20155102398da9fb00b6a79e111a3b5960b0cca37122a`.

This is exploratory evidence without an accepted research Card. It does not
establish the V3 architecture, human preference or final model adoption.
