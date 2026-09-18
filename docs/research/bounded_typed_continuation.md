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
The packaged `bounded_typed_smoke` configuration provides a bounded learning-check
runner. The corpus training and generated-quality comparison have not run.

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

## Comparison and evaluation plan

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
a pipeline check; the main screen remains separately planned below.

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

The planned main screen uses 250k/1M/2M source-onset exposure checkpoints with
at most four training hours per arm, whichever comes first. Pin exact runnable
source, data intervals, optimizer and resource limits after the learning/resource
check. Report equal-exposure and equal-compute results separately, including
unique coverage, repeated exposures, physical rows, endpoint factors and recovery
costs. Prefix preparation and candidate scoring belong in the compute ledger.

Generation retains existing regression cases and adds separately sampled ordinary
and stress groups, with multiple seeds. Mechanical failures cannot be averaged
away. Judge recurrence burden, chord and LN mode coverage, independent hold/attack
relationships, transitions and cross-seed variation. Different reasonable source-
conditioned arrangements are allowed; neither copying source LN ratios nor merely
reducing the shortest attack interval is a success criterion. Foundation-based
machine inspection and current human calibration do not replace new blinded
human comparisons. Independent confirmation needs a second training initialization
and groups unused during model selection.
