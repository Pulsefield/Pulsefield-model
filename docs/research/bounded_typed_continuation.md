# Bounded typed-time continuation

This experiment compares three audio-free continuation tasks using a common
finite-context encoder and complete exact gameplay state. Its purpose is to
separate inadequate learning of the original task, the benefit of typed timing,
and the benefit of deciding complete note objects at their onsets. It remains
a research baseline; generated playability must be evaluated independently of
likelihood and mechanical correctness.

The implementation provides exact execution contracts in
[`contract.py`](../../src/ensomi_model/research/bounded_typed_continuation/contract.py)
and a dense/cached finite content encoder in
[`temporal.py`](../../src/ensomi_model/research/bounded_typed_continuation/temporal.py).
[`features.py`](../../src/ensomi_model/research/bounded_typed_continuation/features.py)
owns permitted exact-state and timing inputs;
[`model.py`](../../src/ensomi_model/research/bounded_typed_continuation/model.py)
implements joint decisions and dependent endpoint likelihoods and sampling.
[`data.py`](../../src/ensomi_model/research/bounded_typed_continuation/data.py)
projects source labels into bounded, batched teacher-forced training windows.
[`generation.py`](../../src/ensomi_model/research/bounded_typed_continuation/generation.py)
samples the native tasks and rebuilds bounded caches from raw checkpoints;
[`verification.py`](../../src/ensomi_model/research/bounded_typed_continuation/verification.py)
independently checks completed outputs against the external condition and plans.
The packaged `bounded_typed_smoke` configuration provides a bounded learning-check
runner. A 16-chart learning check and paired R1 corpus trajectories through four
million source-onset exposures have run. Additional R1 training improves held-out
likelihood and the inspected local burden failures while retaining some independent
LN organization. A matched continuation to 4.5M tests candidate action-consequence
features but does not meet its native-generation improvement criterion. A matched
continuation to 5M adds persistent original-seed conditioning: it modestly reduces
type-proportion error but misses its declared improvement threshold and retains
local execution concerns. Frozen-base head, release and row-response corrections
through 6.75M subsequently improve the inspected long-form behavior and sharp
rearticulation burden while retaining independent LN/TAP organization. Difficulty
consistency and the complete range of generated playability remain unresolved.

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

R0/R1 prediction computes the same mask for a batch of queries in
[`support.py`](../../src/ensomi_model/research/bounded_typed_continuation/support.py).
It evaluates lane legality, fixed seed ends, terminal closure, onset roles and
future availability with Boolean arrays in the original candidate order.
`Schedule.row_possible` remains the scalar commit validator and independent
test oracle. This batching adds no model input, duration limit or sampling rule;
its temporary lane masks scale as queries × 256 × 4.

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

## Native-prefix recovery training

Optional `recovery_pool` and `recovery_sha256` settings add R1 training on pinned,
complete model-generated TRAIN trajectories. The loader checks source identity,
cache pins, original timing and complete seed, generation provenance, trajectory
legality and query counts. Empty R candidates remain in the replay schedule.
No source suffix lane/type or endpoint enters predictor features. Original seed
and complete generated prefix are re-encoded under the current weights.

Core-recovery queries specify a repeated core of lanes and optional blocking held
lanes. At H, alternatives omit at least one core lane or release a specified
blocking lane; at R, alternatives release a blocking lane. Both alternative
and negative families must contain legal actions. The objective is negative log
probability mass of all legal alternatives, computed with logsumexp. It does
not select a unique correct action or change native sampling support.

Response queries instead compare the sampled row with legal alternatives using
[`response.py`](../../src/ensomi_model/research/bounded_typed_continuation/response.py).
The cost counts heads with a same-lane head or release gap below a declared
threshold, over the current row and the next two supplied H. Future actions use
an optimistic minimum: one TAP per H and earliest possible unknown-LN releases;
original seed endpoints remain fixed. This is a machine preference, not an
equal-style future or a calibrated difficulty score.

Preferred alternatives first minimize changes in head count, then LN-start
count, then cost and lane-action Hamming distance, retaining ties. Their loss
is conditional on the union of sampled and preferred composition families.
Moving probability outside that union does not directly improve the loss.
The loader recomputes these sets from the exact native state and verifies the
recorded sampled action. Source contrast used during pool admission never enters
the predictor. Thresholds affect training preferences only, not feasibility.

`recovery_weight` (default 0.25) multiplies the mean loss over
`recovery_queries` (default 2) per optimizer update. Normal source-onset CE
continues with its original denominator. Queries are sampled uniformly by group,
then within group, using `recovery_seed` (default 954) and the absolute update
number; strict resume reproduces the selections. Recovery query/loss/time
counters are separate from ordinary source exposure and coverage. An explicit
same-architecture fork may introduce the objective while preserving model,
Adam and RNG state and extending the immutable source plan.

The negative preferences are machine heuristics, not human semantic annotations.
They require a separate pool-generation procedure and quality evaluation.
Repeated attacks and long anchors can be intentional: avoiding all repetition,
maximizing diversity or reducing LN quantity is not the training-quality goal.
A full-parameter 250k-onset continuation with 80 native queries from eight TRAIN
groups improves unforced repetition on 16 fixed development generations: maximum
consecutive required head rows containing one lane falls from 152 to 11, and
below-40ms diagnostic events from 69 to 3. No below-10ms event appears. These
counts do not establish playability. In two inspected LN contexts, required
onsets with at least two held lanes fall from 35 and 19 to zero; independent
release rows fall from 22 to zero and from 36 to 3. Canonical Beatmap Lens
inspection confirms substantial loss of overlapping LN/TAP organization.

The candidate fails the structural-retention guard despite removing the earlier
quadruple-repeat and three-held-lane witnesses. A bounded outer-column Trill is
not automatically a failure, and reduced LN quantity is not by itself a quality
judgment. The concern is the lost press/hold/release relationship in the inspected
passages. Additional ordinary CE fitting is bundled with recovery training, so
the comparison does not isolate their causal contributions. The full-parameter
candidate is not a validated replacement for the earlier model.

### Frozen-base head routing

`model.head_routing=residual` adds an R1-only head-mask scorer. A shared MLP of
width `model.routing_hidden` (default 512) reads both inherited hand vectors.
Averaging direct and mirrored scores preserves lane-mirror equivariance. The
last projection starts at zero, so adding the module initially leaves native
probabilities and sampling unchanged.

Each legal full action receives a score shift determined only by its binary
head mask: `score(A) = base_score(A) + route_score(head_mask(A))`. With the
inherited policy frozen, normalization within any fixed legal head-mask family
therefore preserves `P(A | head_mask, state)` for the same supplied condition
and committed history. This retains conditional TAP/LN
kinds and simultaneous release choices while permitting different lane groups
and chord sizes. R-only candidates receive exactly zero shift.

`trainable=routing` freezes every inherited parameter, including seed and memory
encoding. The existing parameters remain in AdamW with absent gradients, keeping
their weights and optimizer moments unchanged; only the new scorer updates.
An explicit fork can add the module, this trainable scope and a pinned native
recovery objective while preserving the original source-plan prefix. Normal
source CE and recovery complement loss both train the new scorer. Ordinary
resume requires the same trainable scope and configuration.

This is a conditional preservation guarantee, not a guarantee of identical LN
trajectories: changed head masks change future states. On 16 complete development
generations, a 250k-onset routing-only continuation reduces the maximum consecutive
required head rows containing a fixed lane from 152 to 22. Below-40ms diagnostic
events remain essentially unchanged, 69 versus 68. In one inspected LN passage,
overlapping holds and independent releases remain close to the frozen parent;
another becomes mostly TAP at one seed and retains brief LN passages at another.
Historic quadruple-repeat collapse improves, but these mixed results do not
establish playable long-form generation.

#### Fresh long-chart validation

The same checkpoint was evaluated on eight previously unused validation song
groups, with two sources in each 2–3, 3–4, 4–5 and 5–6 star band and two generation
seeds per source. Selection used source-only metadata, required at least 180
seconds after the seed, and included one independent-LN-rich source per band.
All 76 prior development groups were excluded, with exclusions checked across
both catalog and original annotation-allocation group identities. Source suffixes
span 187–302 seconds. This is a development screen, not a final test set or a
requested-difficulty experiment.

All 16 native temperature-one generations complete and pass independent mechanics
and exact export/reparse in 185 seconds on one CPU thread of the M5 Air. No
below-10ms same-lane attack/release-to-attack diagnostic appears. Below-40ms events
total 242, concentrated in the two highest source bands; the source references
total 26 when counted once per generation seed. A 5.86-star source produces
6.84- and 7.21-star outputs. Counts and star ratings describe execution burden;
they do not supply semantic playability labels.

Canonical Beatmap Lens inspection identifies a remaining held-state failure.
On the 2.20-star TAP source, one generated continuation holds three fixed lanes
while all 26 required attack rows from 154651 to 163818 ms use lane 1. The holds
begin at 147985, 149318 and 153318 ms and persist through an increase in onset
density around 160318 ms. Slow Jack organization can be intentional; the concern
here is a static allocation persisting into the denser passage without developing
the press/hold/release relationship. The source uses changing columns. Existing
human slow-Jack and LN examples calibrate the distinction, but do not label the
generated chart.

Recomputing those native states with the original external schedule gives 15
legal complete actions per query, all with the same head mask. The head-routing
residual therefore cannot change their conditional release probabilities. The
chance of releasing any held lane starts at 0.814 but falls to 0.022–0.042 during
161651–163651 ms. Lanes 0 and 3 finally release at 163818 ms. This directly
identifies a limitation of head-mask reweighting; it does not establish a general
LN-duration cap or imply that three simultaneous holds are undesirable.

Four source-selected LN contexts retain short-LN flow, varying lengths and some
staggered independent releases, with clearer mixed LN/TAP organization in the
4.02-star case. The dense 5.86-star case becomes substantially more LN-heavy.
The held-state failure prevents readiness; phase and seed contexts not inspected
after that failure remain unreviewed. No human annotation was created or changed.

These results use product `8cf31e8d177fab28060ce92be4f5e92f8f8585f3` and routing
checkpoint SHA-256 `dff1727bcb634666992dc4c836ee507c02f31917b1a32f2f1f1d5329eaf5f080`.
Local evidence owners are `routing-recovery-20260920-v1/` and
`routing-fresh-20260920-v1/` under `artifacts/bounded-typed-continuation/`.
The fresh readout SHA-256 is
`fae14c90363c793ed7895e9d2edd159a862afa878722829eb365ea3c1d1d2350`;
its scoped agent review is `14f2507beca22a75060009812b5e0c49e6290de50b22b98ca6e22221c2a7c531`.
Generated files are local evidence and may be absent in a fresh clone.

### Frozen-base release routing

`model.release_routing=residual` adds an optional R1 release-mask scorer with
independent `model.release_hidden` width, default 512. A shared MLP scores the
16 binary sets of closing lanes; averaging direct and mirrored hand orders
preserves mirror equivariance. Its last projection starts at zero. The default
128-wide backbone gains 139,776 parameters, bringing the seed/memory/head-routing
model to 3,056,784 parameters. Native support and temperature remain unchanged.

Actions with the same close mask receive the same score correction. With the
inherited policy frozen, this preserves `P(action | close_mask, state)` for a
fixed committed prefix, including conditional head routing and TAP/LN kinds.
It can change release choices at both H and R candidates. Queries with no held
lane receive exactly zero correction. These properties do not guarantee
unchanged later LN organization after a different release has been sampled.

`trainable=release` updates only the new scorer and keeps all inherited model
tensors and Adam states unchanged. An explicit fork can append this module to a
completed parent and replace its pinned recovery-pool path and digest. Existing
scalar recovery settings, optimizer settings and source-plan prefix must remain
unchanged. Ordinary resume requires the same model, training scope and pool.
Defaults preserve earlier checkpoint identities. The module is an experimental
candidate; its native quality and retention of independent LN/TAP relationships
require complete-chart evaluation after training.

#### Release recovery on complete charts

A release-only continuation uses 72 machine recovery queries from four TRAIN
song groups, alongside 250,000 additional ordinary source onsets. The native
queries identify persistent three-held-lane states whose remaining lane carries
repeated attacks while the corresponding source uses varied routing. These are
machine preferences, not human semantic labels. Collection selected sources
with sparse-to-dense timing transitions and stopped at its first adequate
103-group prefix; it does not estimate corpus-wide failure prevalence.

Training from routing 6.25M to release 6.5M takes 148 seconds on one CPU thread
of the 24 GiB M5 Air. Sampled peak RSS is 1.85 GB and no swap growth is observed.
All inherited model and Adam tensors remain bit-identical. The real-data
preflight also checks exact initial native draws and conditional preservation
after an optimizer update; its largest within-release log-probability difference
is $9.54\times10^{-7}$, consistent with float32 rounding.

The fixed comparison contains 32 complete continuations from 16 validation song
groups, with seeds 17 and 23. It combines the earlier long-chart cohort and the
fresh cohort described above, including continuations longer than 14 minutes.
All outputs pass independent mechanics and exact export/reparse. Generation
takes 293 seconds on CPU1, with sampled peak RSS 734 MB and no swap growth.

Maximum consecutive required attack rows under the same three unchanged holds
falls from 26 to 4. The earlier 26-row confinement through a density increase
becomes a passage with changing held lanes, releases and multi-column TAP/chord
flow. The maximum consecutive attack rows containing one fixed lane falls from
26 to 14; the remaining 14-row witness is a bounded 2.17-second Jack with varied
entry and exit. The largest exact head-word cycle is a 31-row, 5.25-second
left/right-pair Trill. Its context establishes a bounded exchange, not prolonged
allocation collapse. Neither Jack nor Trill presence is itself a defect.

Matched canonical Beatmap Lens views retain independent LN/TAP relationships in
both selected LN-rich passages and both seeds. In Luster (`ed29fd1a8c2d`,
220097–228097 ms), independent release-row counts change from 34/31 to 37/37.
In Gloomy Flash (`f2e23b5e0b78`, 52881–60881 ms), they change from 36/32 to 28/47.
The views show staggered starts and releases while other lanes
remain held, varied hold lengths and intervening TAP flow. These relations,
calibrated against current High human LN and Stream examples, establish the
retention observation; the counts alone do not. Inspected early/middle/late
contexts and long-chart late excerpts retain changes of organization.

This candidate passes the scoped long-form recovery and LN-retention development
screen. Overall playability and 2–6-star consistency remain unestablished.
Below-40ms same-lane attack or release-to-attack diagnostics increase from 310
to 366 over 90,678 required onsets, with no below-10ms event. The largest inspected
two-second cluster has 20 release-to-head gaps of 37–38 ms, versus 17 baseline
diagnostic events in that scope. It is rhythmic short-LN rearticulation rather
than a new sustained allocation collapse, but its execution cost still matters.
Generated star ratings span 2.137–6.717; the 5.863-star source produces 6.649 and
6.717 stars. No requested-difficulty calibration or player trial is implied.

The review covers 27 generated contexts selected by fixed phase locations and
whole-chart failure locators. It is an unblinded agent development review, not
an inspection of every second or new human annotation. Human labels, confidence
and comments were checked against six unchanged canonical source documents.
The frozen slow-Jack calibration has no recorded confidence; it is not counted
as a High-confidence judgment.

Implementation and training source:
`1e5f5da8d7e1fe54646934e4d512ff8e113bef93`. Final checkpoint SHA-256:
`0a9c87afa43caa9d3647fe3315d79b3a048cc71261571c599c85cf49361dd1ec`.
Local outputs belong to `release-recovery-20260920-v1/` under
`artifacts/bounded-typed-continuation/`; its complete readout is
`d354a939829e0877ccbfa7815bfb312521b91c93692eca8f88c6033651aff0d5`
and scoped review is
`2b7be2eb28d855f558e16cc676476f13ad4839662d91de326352273a22fa0bf6`.
These local assets may be absent in a fresh clone. The portable generation
command below reads the saved architecture directly from the checkpoint.

### Native response recovery result (6.75M)

A frozen-base `frontier2` row residual reduces sharp rearticulation in a paired
48-output development comparison while retaining the inspected long-form and
independent LN/TAP organization. It adds 27,648 parameters to release6.5M, for
3,084,432 total. The inherited policy and its Adam state remain unchanged.
Inference still uses native temperature 1 and the original feasibility support.

The training pool contains 92 machine preferences from nine of 32 source-selected
TRAIN groups. Admission requires a strictly better current-plus-two-H response
bound and no below-30ms head in the corresponding source interval. Of the 92
queries, 81 retain both sampled head and LN-start counts; two are R-only queries.
All nine contributing source/generated contexts were inspected. The preferences
are separate from human annotations and do not expose source future actions to
the predictor. A single 250,000-onset continuation uses source CE, a unit-weight
source KL anchor and two native queries per update at weight 0.25. It finishes
328 updates in 235 seconds on CPU1, with sampled peak RSS 1.38 GB and no swap
growth. Initial native predictions and draws match the parent exactly.

The comparison covers 24 VAL musical groups at seeds 17 and 23, including the
16 outputs of the release candidate's unused-group confirmation. All 48 complete
outputs pass mechanics and exact osu! export/reparse over 120,416 required suffix
onsets. Generation takes 461 seconds on CPU1, with sampled peak RSS 613 MB,
footprint 912 MB and no swap growth. The independent physical-row recount agrees
with the exact-state replay at every reported threshold.

| Diagnostic | Release6.5M | Response6.75M |
| --- | ---: | ---: |
| Heads below 20ms since same-lane head or release | 45 | 13 |
| Heads below 30ms | 160 | 70 |
| Heads below 40ms | 442 | 178 |
| Below-30ms heads in states with no recovered lane | 71 | 13 |
| Below-30ms excess beyond the same-cardinality minimum | 77 | 52 |
| Longest unchanged three-hold allocation, in H | 4 | 2 |
| Longest consecutive fixed-lane attack run, in H | 14 | 18 |

No paired chart increases its below-20ms or below-30ms count. Five increase at
40ms; their inspected events have 30–38ms gaps within varied local organization.
They remain execution-cost concerns rather than evidence of a sustained allocation
collapse. No below-10ms head occurs. These thresholds are diagnostic, not universal
playability boundaries. Total generated heads increase from 166,272 to 177,531;
the improvement does not come from removing required onsets or reducing all
attack activity.

Fifty candidate contexts and eleven matched parent contexts were inspected with
the frozen Lens Foundation and current human LN/Stream contrasts. The 17.50-minute
continuation retains changing organization at early, middle and late scopes,
including both seeds around 882–890 seconds; the 14.20-minute continuation does
so around 713–721 seconds. The largest 18-H lane run lasts 2.357 seconds within
moving chord-Jack figures. Head-mask cycles of 76 H over 5.357 seconds and 71 H
over 5.468 seconds form bounded traversals and left/right-pair alternation with
distinct entry and exit. Repetition counts alone do not classify them as collapse.

Luster's independent close-row counts change from 37/37 to 34/30 in the two
220097–228097ms retention scopes. Gloomy Flash's counts change from 28/47 to
23/16 at 52881–60881ms. The actual views retain staggered starts and releases,
varied hold lengths and TAP participation; they also show reduced concurrent
hold pressure and more TAP activity in Gloomy Flash. Four source-selected LN
cores in the confirmation cohort retain independent organization. LN count alone
would not establish this retention.

The working development candidate is response6.75M; overall disposition remains
REFINE. Generated stars span 2.142–6.128. The 5.863-star source's two outputs change
from 6.649/6.717 to 6.011/5.758, but a 5.089-star source rises from 5.463 to 6.128
at seed 17, and a 3.143-star source gives 4.235/4.384. Source-star matching is not
itself playability, but these shifts limit confidence in difficulty consistency.
This is unblinded development evidence, not a player trial, an inspection of every
second, or final independent acceptance. The bundle of model and objective changes
does not isolate the contribution of each component.

Training and evaluation source:
`9da67258d76c8359964d81aafa33ed177c2de6fd`. Checkpoint SHA-256:
`195f1b0109696302addc1aa62bf896ca399d2a1656c15ba4fe27b4a47c8171ce`.
Local outputs belong to `row-response-recovery-20260920-v1/` under
`artifacts/bounded-typed-continuation/`. Complete comparison SHA-256:
`9d364a3a994f9928c43e8d9d5156742bc180407e46c365c27698487f1c401586`;
scoped semantic review:
`781e908bbe028a309c4d0d4e519b3f7aa629bfac97d34baab8d1f1de43c304c5`.

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
the rolling raw history envelope is 512 physical rows. Current complete exact
facts, external time conditions, optional persistent seed conditioning and
optional full-history landmark memory are separate prediction inputs.

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
older than the declared field. Such changes must not affect the learned output
unless they belong to an explicitly enabled persistent seed or full-history
landmark memory below.
The tests also check parameter gradients under crop recomputation, so matching
forward logits alone is insufficient.

Known timing features may look ahead without an action-causal mask. All arms
can read R; only typed arms can read H. Use full supplied timing for nearby
offsets, future gap descriptors and multiscale counts, independent of training
window endpoints. Report how 512 physical rows translate into seconds across
density strata before interpreting this as musical or phrase-scale context.

### R1 persistent original-seed conditioning

`model.seed_context` selects `none`, `zero` or `observed`; the latter two are
R1-only. `observed` re-encodes only the supplied complete seed with the shared
temporal module. A masked mean over all seed-token outputs gives two canonical
hand vectors. Each token contributes even when a supplied seed exceeds the local
receptive field. The readout concatenates each current hand vector with its seed
vector, applies a shared `Linear(2h,h), GELU, Linear(h,h,bias=False)` residual,
and adds it to that current vector before joint decision scoring. The final
projection starts at zero, preserving the parent model's distribution.

`zero` uses the same residual with a zero seed vector; `none` preserves the
original readout. The two residual modes add 49,280 parameters at hidden 128.
This is a matched information/capacity comparison; zeroed inputs do not provide
equal effective capacity. Both retain the same likelihood objective, support and sampler.
The representation introduces no source suffix action or endpoint, desired
difficulty or target LN fraction. Supplied seed endpoints remain permitted.

Training re-encodes the original raw seed under current weights for each batch,
with gradients through the shared temporal module. Inference computes its vector
once under fixed parameters. When landmark memory is disabled, only original
seed facts persist in addition to the rolling history; unrelated older generated
actions remain outside the learned field. This global-condition adaptation is motivated by the prefix-state
diagnostic below. Its generation-quality benefit has not been established, and
an introductory seed need not represent later chart structure.

### R1 full-history landmark memory

`model.long_memory=landmarks` enables an R1-only learned history branch. Its
shared-hand GRU reads every committed physical row from the true beginning,
using the same action/gap/permitted-plan features as local content. It stores a
landmark after every `model.memory_stride` head rows (default 64). These are
bookkeeping intervals, not asserted musical phrases. Memory width is controlled
by `model.memory_hidden` (default 256), independently of the local TCN width.

The current hand representation attends to all earlier landmark keys and values.
A zero-initialized output projection adds the resulting context before joint
row scoring. Landmarks written by the current target or a future row are masked
before softmax. Queries before the first completed landmark receive a zero
residual. Hand encoders and projections are shared, preserving mirror symmetry.
The default-width memory adds 446,848 parameters; with observed seed conditioning
the model has 2,777,232 parameters. This changes access to past organization,
without providing a future source plan, quantity request, minimum gap or
repetition rule. A 256-wide memory candidate continued from the observed-seed
5M checkpoint to 6M and was evaluated on 16 complete development generations
(eight fixed sources, two seeds). It reduced below-40ms diagnostic counts from
152 to 69, but produced a passage with 143 quadruple TAP rows among 152 head
rows over about 18 seconds, with no held lanes forcing the routing. Another
passage retained a single-column stream under three long holds. The candidate
fails the long-form playability gate; more past memory and fitting did not
prevent collapse. This comparison does not isolate architecture from the
additional training.

Training explicitly prepares full physical prefixes for this branch and
recomputes them with current weights, including gradients through older history.
Local TCN computation remains cropped to its finite field. No parameter-dependent
memory survives an optimizer update. Inference updates the GRU only when a
physical row is emitted and writes keys/values only at completed landmark
boundaries; empty timing candidates do not become content. The active local
history continues to cover recent detail between landmarks.

Recovery retains the complete raw committed history and replays it under verified
parameters, rebuilding the GRU and landmark bank. The constructor verifies that
history against the exact schedule, original seed, permitted endpoint visibility
and rolling history; the packaged runner also binds it to the complete committed
row/decision journal. Thus this optional mode has history storage and recovery
work that grow with generated duration, and full-prefix training work that grows
with crop position. Resource limits still apply. The default `none` mode retains
its earlier parameter identity and bounded-history behavior.

### R1 candidate action consequences

`model.row_consequence` selects `none`, `actions`, `frontier` or `frontier2`;
enabled modes are R1-only. Each adds a width-32 residual energy for each complete
candidate row. `actions` and `frontier` have 26,912 parameters at hidden 128.
A zero output layer preserves the
existing conditional distribution when the original weights are copied.
`actions` supplies candidate-action one-hots. `frontier` additionally supplies
the exact post-action occupied flag and seven per-lane time bases:

- Current head since the preceding same-lane head and release.
- Current release's LN age.
- Post-action head, release and open-LN clocks, passively advanced to the next
  strictly future required onset H.
- H minus the earliest possible release of a post-action occupied lane: its
  supplied seed endpoint if known, otherwise the next R candidate.

The shared timing block contains next-R and next-H gaps and the next-R onset
role. Missing or inapplicable clocks use the existing unavailable encoding.
`frontier2` appends the gap to the second strictly future H to this timing block,
retaining the existing lane fields and earlier timing positions.
All timestamp differences are computed in float64 before float32 conversion.
The future views describe the candidate's immediate state; intervening actions
remain unknown. A possible release is not a committed endpoint. These features
use only exact prefix facts, candidate actions and supplied R/H.

[`consequence.py`](../../src/ensomi_model/research/bounded_typed_continuation/consequence.py)
factorizes the first affine into four relative-lane projections, a timing
projection and the encoded hand context. Sixteen lane/action descriptors are
projected once and gathered for 256 complete candidates. GELU and a shared scalar
projection combine their effects; averaging the two canonical hand views keeps
mirror equivariance. This equals the dense concatenated-feature affine while
avoiding its large candidate-feature tensor. The residual joins the common
decision scorer before the unchanged exact support mask, so teacher likelihood,
native sampling and raw-history recovery consume the same energy.

`trainable=consequence` freezes all inherited parameters and trains only this
residual. Optional `source_kl_weight` adds KL from the frozen parent distribution
to the candidate on the same source states. The parent distribution bypasses
only the row residual, retaining inherited seed, memory, head and release
scorers. Source CE and KL use the source-onset denominator; reported likelihood
factors remain pure CE, with KL recorded separately. The default zero weight
preserves earlier training identities. Nonzero KL requires consequence-only
training.

The optional readout supports matched comparisons against continued `none`
learning and the equally sized `actions` branch. Equal parameter count does not
imply equal effective input capacity: the action-only control zeros all exact
consequence and timing fields. The matched 4.5M comparison below does not establish
a generation-quality benefit from these features.

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
With observed seed conditioning, the snapshot additionally retains the complete
original raw seed, including supplied endpoints of objects that have since ended.
Restoration rebuilds its learned vector under the verified model. The packaged
runner checks this seed against the external condition as well as checking the
rolling state against journals. Without landmark memory, storage depends on
supplied seed length and fixed rolling capacity, not generated duration. Landmark
memory additionally retains the complete raw history, as described above.
Default-disabled models retain
their pre-extension parameter digest and can read older unconditioned snapshots.
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
uv run --python 3.10 --extra mps python -m ensomi_model.research.bounded_typed_continuation.condition_hydra \
  source_file=/path/to/source.osu source_sha256=SOURCE_SHA256 \
  output_file=artifacts/bounded-typed-continuation/condition.json arm=r1 seed_notes=30

uv run --python 3.10 --extra mps python -m ensomi_model.research.bounded_typed_continuation.generate_hydra \
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
`python -m ensomi_model.research.bounded_typed_continuation.smoke_hydra`.
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

The [complete 5M TRAIN distribution census](r1_training_distribution.md)
measures difficulty, LN amount, action clocks and exact consumed windows. It
finds 80.08% of exposures within 2–6 source stars and 1.39% above 6; the sampler
does not impose those bands. Original-seed LN amount often differs from later
phases, and raw human records cover 149 of the 11,563 eligible charts.

The corpus entrypoint is
`python -m ensomi_model.research.bounded_typed_continuation.train_hydra`.
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
never edited. Scientific settings may only add endpoint-availability, R1
row-consequence or R1 seed-context residuals to an original model with all three
modes set to `none`. An explicit landmark-memory extension may also preserve
existing unchanged residuals while appending only the memory module; changing or
removing their modes is rejected. A native-recovery objective fork instead keeps
the entire architecture unchanged and adds its pinned training pool. A head-routing
fork instead appends its scorer and selects routing-only training, optionally
adding the recovery objective in the same transition. Parameter
order and inherited Adam state remain checked.
Release and `frontier2` response forks freeze the inherited policy and may
replace the pinned recovery pool while retaining its scalar settings. A response
fork may add source KL, but must preserve every other residual mode. Adam state
is mapped by parameter name so a row residual inserted before existing optional
modules cannot shift their optimizer identities.
Ordinary resume retains exact source/config identity.

Fork initialization copies all existing weights, AdamW moments/steps and RNG,
retains cumulative exposure/coverage/metrics, and charges the parent's entire
measured compute time. Only residual parameters may be added; their optimizer
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

### Difficulty and long-form coverage at 4M

A further source-only selection excludes both earlier 24-group screens and
chooses 28 distinct VAL groups, seven in each source band [2,3), [3,4), [4,5)
and [5,6]. Each band has three group-uniform ordinary sources and four sources
selected for independent LN starts, short-LN/tap adjacency, taps inside long LNs
and continuation length. These mechanical descriptors locate contrasting cases;
they are not semantic labels. Source ratings use the same pinned calculator and
playback conditions as above. This collection is stratified rather than a
population prevalence sample.

Both fixed 4M models generate every complete suffix under three random seeds:
168 outputs and 56 full-suffix likelihoods. All outputs pass mechanics and exact
export/reparse, including six continuations of approximately 17.5 minutes. The
complete evaluation takes 2,056 seconds on the same one-thread CPU setup, with
sampled maximum process footprint 464 MB and no swap growth. Read-only inspection
and rendering overlap some execution; this is an operating measurement rather
than an isolated throughput benchmark.

| Source-star band | 171 group-mean NLL | 172 group-mean NLL | 172 pairs below 40 ms per 1,000 supplied suffix onsets |
| --- | --- | --- | --- |
| [2,3) | 2.191116 | 2.177359 | 0 |
| [3,4) | 2.233952 | 2.209003 | 0 |
| [4,5) | 2.411728 | 2.397454 | 0.209691 |
| [5,6] | 1.972021 | 1.993644 | 0.118249 |

Initialization 172 passes the declared coarse numerical screens, but the
collection does not pass the broader quality requirements. Its actual generated
star bands contain 13/13/9/2 distinct groups respectively; a group can contribute
to multiple bands. The last band falls short of the required three groups.
Several 5-star source conditions produce 4-star outputs, so input coverage cannot
be used as output coverage.

More importantly, 172 still produces fourteen below-40-ms same-column pairs in
nine outputs. Two context-inspected cases concentrate a 23-ms or 37-ms repeat on
the only available lane while three others are held. Their source counterparts
distribute the same onset sequence across lanes. Reconstructing the original
generated histories reproduces all 44 inspected chosen log probabilities exactly.
Immediately before those two repeats, the model assigns the choices that force
them probabilities 0.268926 and 0.183571. Alternatives that avoid forcing this
head repetition remain in the existing support. The failure therefore is not
explained solely by tiny sampling tails or
unavoidable timing. Source-history controls change the entire history and do not
isolate whether representation, learning coverage or feedback causes the error.

The surrounding generated passages contain real independent LN organization and
tap/hold contrasts. Separate preselected views also show plausible phase and
seed variation in the 2-star examples and prominent independent LN control in a
4.25-star output. These positives do not resolve the localized burden concerns.
Fourteen of sixty prospective primary scopes have been reviewed; the rest remain
unreviewed. The failed output-coverage guard and inspected burden concerns are
already sufficient to withhold an overall quality claim. Complete mechanical
generation of a long chart is not evidence that its entire structure is good.

These findings motivate the candidate action-consequence comparison below.
Actual generated difficulty and independent LN/tap organization remain separate
requirements; neither is established by resampling until a favorable output appears.

The evidence owner is
`artifacts/bounded-typed-continuation/difficulty-ln-longform-20260920-v1/`.
Complete readout SHA:
`30b7e2a7484d84695c7ebe16ce66da1ed1a31ac92998af746738991ccd410ee3`;
independent digest/resource audit SHA:
`f313678fa2896255decac00bbb5b674070c80a9a3c31d6e6ebfd3777d972958d`;
frozen-history probability report SHA:
`ddb9312beda065057300ed9257dad1888a3cb54eec8f8f8153ff2175ff63bc65`.

### Matched R1 action-consequence comparison at 4.5M

Three continuations of initialization 172 start from the same 4M checkpoint and
add 500,000 onsets with identical draws, optimizer settings and inherited Adam
state. `none` preserves the original model; `actions` adds the candidate-action
residual; `frontier` adds the exact consequence features described above. The
two residuals start at zero and have identical parameter shapes. All final runs
reach 3,206,045 unique training onsets across 9,178 charts and 3,168 groups.

The comparison reuses the 28 development groups, three generation seeds and
unchanged native temperature-one sampling. All 252 full outputs pass mechanical
verification and exact export/reparse. All 84 full-suffix likelihoods are finite.
Evaluation takes 2,464 seconds on one M5 CPU thread, with sampled footprint
544 MB and no swap growth. Batched support evaluation changes implementation
cost, not legal candidates or model inputs; separate native replay checks produce
byte-identical outputs and decision journals against the scalar runtime.

The primary diagnostic counts a suffix head once if its preceding same-lane head
or latest LN release is less than 40 ms earlier. Rates use supplied suffix onsets
as denominator, then average seeds within groups and groups equally. This locates
potential action problems; release/head and head/head intervals need not impose
equal burden, and the threshold is not a universal playability boundary.

| Mode | Union rate per 1,000 supplied onsets | Head/head events | Release/head events | Group/seed-mean LN fraction |
| --- | --- | --- | --- | --- |
| none | 6.482075 | 21 | 1,248 | 0.489053 |
| actions | 6.792055 | 21 | 1,330 | 0.483170 |
| frontier | 5.886852 | 23 | 1,118 | 0.530932 |

One `frontier` event meets both interval conditions and counts once in the union.
Its relative reductions are 9.18% against `none` and 13.33% against `actions`,
below the predefined 25% requirement. Paired whole-group 90% bootstrap intervals
for rate differences are [-1.672207, 0.265617] and [-2.633005, 0.285602]. Both
include zero. Numeric regression guards pass, but the primary criterion fails.
The original 4M checkpoint has union rate 1.044819 and group/seed-mean LN fraction
0.180448 on these same conditions. The large shared change after further learning
requires investigation; it cannot be attributed to the consequence residual.

A complete masked agent review uses Beatmap Lens Foundation V2, confirmed human
examples, full endpoints and canonical time-proportional views. It covers 12 LN
cores, 12 ordinary early/middle/late scopes, four long-chart late scopes and three
selected risks. Two risk selections identify the same context. Judgments are
sealed before model identities are revealed, but earlier numeric summaries allow
partial recognition; these are neither blind human preferences nor new gold.

`frontier` retains independent organization in all eight LN cores judged prominent
for `none`; one changes to supporting strength. It also introduces localized
coupled-restart concerns in two LN cores. At a selected risk it avoids the controls'
12 ms restart, while a different source/seed produces its own 3 ms release/head
restart under continuing holds. The four long-chart late scopes remain locally
organized, including near twelve minutes, but style can differ markedly between
branches: a TAP-led passage may become a sustained LN body. Conversely, one
six-minute late scope changes from prominent LN coordination in `none` to absent
in `frontier`. These scoped positives do not establish whole-chart stability or
organic LN/TAP diversity throughout.

Each mode now has five distinct groups with some actual 5–6-star outputs, but
generated ranges extend from about 1.74 to 7.02/7.02/7.24 stars. Coverage therefore
does not establish requested difficulty control. The result supports revising the
learning and conditioning investigation, not adopting a final model or turning
the diagnostic threshold into a decoding rule.

Evidence owner: `artifacts/bounded-typed-continuation/action-consequence-20260920-v1/`.
Native readout SHA:
`b7e825369e6fe8065184406868bcf5f07c68593a24a841b192577d4b9cef1ac1`;
sealed semantic manifest SHA:
`695e283beca32767f09c3e613b14689e140bc2d43f028853885bee6bc39294cf`;
revealed semantic readout SHA:
`e4ec43256cacd6a6e02f5a4c54acbd262a4ddff09b4f8e7f3e37ca68312fb652`.
This is exploratory evidence without an accepted research Card.

### R1 prefix-state recovery diagnostic

The 4M-to-4.5M change is much larger under native generation than under source
history. On the same 28 groups, mean conditional LN probability on source-head
lanes where both TAP and LN are legal increases from 0.262043 to 0.281138;
the observed positive fraction is 0.266066. Binary Brier score improves from
0.087358 to 0.086733. Native group/seed-mean LN fraction increases from 0.180448
to 0.489053. The conditioning and denominators differ, so this contrast does
not by itself isolate a feedback mechanism.

A frozen-checkpoint diagnostic replaces generated history at the middle of
eight reused development charts with the corresponding source history. Six
charts were selected for whole-chart LN excess and two are LN-rich contrasts.
Three generated prefixes per chart each receive a paired 512-onset continuation
with fresh, matching initial RNG states. Both paths retain the original timing,
terminal, weights, support and temperature-one sampler. Only the original seed
has supplied future endpoints; a source-prefix LN born later retains an unknown
end until its release. The intervention changes the whole prefix state,
including learned history, occupancy, clocks and counts.

The primary diagnostic is absolute generated-versus-source LN-head-fraction
error over the first 128 required onsets, averaging three seeds within each
chart and eight charts equally. Source-prefix replacement reduces this error
from 0.350853 to 0.116417, a 66.82% reduction. The paired difference is -0.234436
with a whole-group 90% bootstrap interval of [-0.441586, -0.059069]. Over the
final 128-onset bin, errors are 0.467922 and 0.228744. All 48 windows complete;
an independent audit verifies their action journals, exact states, digests and
statistics. Execution takes 168 seconds on one M5 CPU thread, with sampled
footprint 226 MB and no swap growth.

Recovery is heterogeneous. One mostly TAP reference changes from about 96%
generated LN heads to about 2% in the first bin after source replacement.
Another TAP-only reference recovers initially, but one replaced-prefix seed
later reaches 61% LN heads. A mixed chart stays LN-heavy under both prefixes;
the two LN-rich contrasts do not improve uniformly. The final bin can still
begin with up to 127 tokens from the initial learned history, and exact global
facts can persist longer. These observations establish dependence on prefix
state as a whole, without uniquely identifying learned-memory drift or a
missing global style condition.

Source-prefix replacement is a diagnostic unavailable during ordinary inference.
Matching a source type proportion is neither a semantic quality judgment nor
evidence of playability. Persistent encoding of the original supplied seed,
explicit desired structure, and optimization changes remain distinct hypotheses;
the diagnostic does not select a proven remedy.

Evidence owner: `artifacts/bounded-typed-continuation/prefix-recovery-20260920-v1/`.
Complete report SHA:
`d7fb165b2c316d0298233fa5379944f6441abce41a22bc2f86923cd714d8f9eb`;
independent audit SHA:
`abffcaa5ba30d266fb865b23ec225d98d289023e8814e7947eb881aa8a8bbc62`.
This selected-cohort diagnostic is exploratory evidence without an accepted
research Card.

### Matched persistent original-seed comparison at 5M

Persistent access to the complete supplied seed changes native type balance,
but does not establish stable playable generation. The original 4.5M R1 model
continues for another 500,000 source-onset exposures under three conditions:
the unchanged model (`none`), an additional residual receiving a zero seed vector
(`zero`), and the same residual receiving the learned original-seed representation
(`observed`). The mechanism is defined under
[persistent original-seed conditioning](#r1-persistent-original-seed-conditioning).
The residual adds 49,280 parameters to the 2,281,104-parameter model. Its initial
output is exactly zero; copied parameters, Adam state and initial likelihoods
are preserved. Equal parameter count does not imply equal effective capacity.

All three arms receive the same 661 additional updates, draw plan and optimizer
settings. Final cumulative coverage is 3,459,305 unique onsets across 9,533 charts
and 3,169 groups. Only the final 5M checkpoint is evaluated for quality. The
comparison generates all 28 reused development conditions with three seeds per
arm at temperature one, and scores each complete source suffix. The supplied
seed, R/H, support, objective and sampling rule are unchanged; no source suffix
type proportions or style labels enter the model.

For each complete suffix, divide required onsets into four consecutive bins of
nearly equal onset count. Compute absolute generated-versus-source LN-head-fraction
error in each bin, then average bins and seeds within each group and the 28 groups
equally. These bins are not equal-duration time windows. Source proportions are a
diagnostic reference, not the uniquely correct continuation of a condition.
Generated summary means use the same group/seed weighting. Source-suffix NLL is
averaged over groups after normalization by each suffix's required-onset count.
The interval numerator counts individual heads; its denominator is the number
of required onsets, which can each contain several heads.

| Mode | Four-bin proportion MAE | Mean NLL per required onset | Mean generated LN-head fraction | Short-interval union per 1,000 required onsets |
| --- | ---: | ---: | ---: | ---: |
| none | 0.156525 | 2.151510 | 0.339605 | 2.990635 |
| zero | 0.154749 | 2.151372 | 0.321759 | 2.876967 |
| observed | 0.133155 | 2.156257 | 0.301196 | 2.954803 |

Observed reduces proportion error by 14.93% relative to none and 13.95% relative
to zero. Both miss the declared 20% threshold. The observed-minus-control paired
whole-group 90% bootstrap intervals are [-0.058117, 0.005821] and
[-0.038156, -0.004962], respectively; the comparison against none also fails the
requirement that the interval lie below zero. Both likelihood and short-interval
regression guards pass. All 252 outputs pass mechanical verification and exact
osu! export/reparse; all 84 suffix scores are finite. The minimum release-to-head
interval remains 12 ms in every arm. Raw release/head counts are 528, 518 and 525;
the corresponding head/head counts are 31, 29 and 15. A below-40-ms event is a
screening observation, not a universal playability rejection rule.

Beatmap Lens V2 and confirmed human examples calibrate 31 anonymous scoped
comparisons: 12 LN cores, 12 ordinary early/middle/late scopes, four long-chart
late scopes and three branch-specific worst-interval contexts. All 717 canonical
time panels are inspected through complete montages, with exact action records
and selected full-resolution risk pages. Judgments are sealed before model
identities are revealed. Two risk cases select the same generated context, so
there are 30 unique contexts and 90 unique variant/context judgments. These are
agent judgments, not new human gold. Numerical outcomes were visible before
masking, which permits partial recognition of variants.

Observed retains independent LN presence in all eight cores where a concurrent
control has prominent coordination, passing the declared presence-only guard.
Two of those cores weaken to supporting organization. Another core with a
supporting none-arm episode loses that episode entirely. In the long-chart late
scopes, none's two prominent LN bodies become supporting in observed; the other
two scopes have no independent LN organization in any arm. Local plausibility
therefore does not establish preservation of the full arrangement.

Specific execution concerns remain. In one ordinary late scope, observed creates
a cluster of 37–38 ms release/head repetitions under other held lanes, while
both controls have one isolated 38 ms event. In the selected high-risk condition,
observed holds three lanes across a required onset and forces a 24 ms restart on
the only free lane; it later restarts another lane after 12 ms. The zero control
has a forced 12 ms restart under three holds and further 12 ms repetitions.
Elsewhere, zero retains one column across 48 consecutive attack rows over
6.47 seconds. These are distinct failure mechanisms; neither LN proportion nor
the shortest interval alone measures them. Recognizable LN coordination does not
excuse the execution burden.

Actual generated star ranges are 1.650–8.030 for none, 1.671–8.105 for zero and
1.567–7.882 for observed. The experiment therefore does not establish consistent
2–6-star output. Persistent seed information has a limited signal relative to
the equal-size zero control, but the primary failure, weakened structure and new
local concern prevent a generation-quality improvement claim. A short intro may
also be an inadequate condition for later form. This result does not rule out
training from scratch or explicit desired structure as different interventions.

All three continuations together take 1,696 seconds and evaluation takes
2,341 seconds on one M5 CPU thread with PyTorch 2.11. Peak training RSS is
1.10 GB and evaluation RSS is 615 MB; swap growth is zero. The experiment owner,
including checkpoints and inspection panels, occupies about 639 MiB. This
establishes local feasibility for this parameter scale and procedure, not an
isolated throughput benchmark. One training seed, one continuation boundary,
reused development groups and fixed-seed semantic scopes limit generalization;
TEST remains unread.

Evidence owner: `artifacts/bounded-typed-continuation/persistent-seed-20260920-v1/`.
Runtime revision: `a64ac0a0ed5b1ec6b7af5ab3b571cd122b30a5c4`.
Complete numerical readout SHA:
`b5b0188095158e0c3356f9cf7b10ef3b96595ad3d478a80568486bdaf8a57d1e`;
sealed semantic manifest SHA:
`e4bbcb249cdf3088fbbebd3189cab0731e59e8aa893cccd4319d2c289af84090`;
revealed semantic readout SHA:
`7aced070426f434dda5175b5c6e819b4bbd14def1c316db06c2c8112ba2f5eac`.
This is exploratory evidence without an accepted research Card.

### Native short intervals: inherited state and current choice

The complete 5M population contains both unavoidable current-state clock events
and avoidable action choices. A read-only diagnostic replays all 252 generated
charts and the 28 source references against the original R1 conditions. Only
seed endpoints are known to the schedule. At each required suffix onset, a lane
is eligible for a head only if it is closed before the row; a lane released on
that same row remains ineligible.

For a strict threshold $\tau$, an eligible lane is called recovered here when
both its latest-head and latest-release gaps are at least $\tau$; a missing
predecessor imposes no restriction. This is a clock predicate, not a calibrated
gameplay-demand definition. If no eligible lane is recovered, at least one short
head is unavoidable at that state. If $k$ heads were chosen and $r$ recovered
lanes are available, retaining that head count needs at least $\max(0,k-r)$ short
heads. Any observed excess above that bound has an alternative at the same head
count. The replay explicitly verifies alternative TAP rows with the original
release decisions through `Schedule.row_possible`.

The following disjoint counts use $\tau=40$ ms. Each generated arm has 84 charts
and 171,723 required suffix onsets; sources are counted once, with 57,241 onsets.

| Population | Short heads | Heads at states with no recovered lane | Additional minimum from chosen head count | Excess above that minimum |
| --- | ---: | ---: | ---: | ---: |
| Source | 6 | 0 | 0 | 6 |
| none | 558 | 263 | 75 | 220 |
| zero | 547 | 264 | 86 | 197 |
| observed | 540 | 258 | 80 | 202 |

About 47–48% of generated short heads occur after the history has already removed
every recovered option. This category counts all chosen heads at those states;
the corresponding required-onset counts are 254/257/238. A further 13–16% are
required to retain the chosen head
count, and 36–39% exceed that cardinality minimum. The source's six events are
31 ms release/head transitions on four onsets, all with recovered alternatives.
Their presence in source charts reinforces that this predicate is a diagnostic,
not a universal rejection rule. At 20 ms, source count is zero; generated counts
are 7/10/8, including 2/4/2 heads forced by the current state. All 10/20/30/40 ms
partitions and macro rates are retained, and the 40 ms counts exactly recover
the preceding evaluation for every output.

This decomposition supports investigating both earlier occupancy/release
planning and immediate action selection. It does not identify a unique neural
cause or prove that substituting a recovered lane improves a complete chart.
The verified alternatives use TAPs and can remove LN structure; they are bounds
on available actions, not generated repairs or equal-style substitutes. Additional
intent conditioning, larger capacity, training-state exposure and decoding
remain separate interventions requiring their own native comparisons.

Exact replay takes 55.2 seconds on one M5 CPU thread, peaking at 251 MB RSS and
200 MB footprint with no swap growth. A separate implementation recounts all
280 physical-row sequences without the schedule or classifier helper and
verifies every partition, group/seed aggregation and witness-file digest.
Evidence owner: `artifacts/bounded-typed-continuation/state-choice-audit-20260920-v1/`.
Runtime revision: `67107af6c196de45dee311358237befd96026531`.
Readout SHA: `8a7a3fce2458e880f6f35b49b4e5a63f8d08f919e6d313f91c1f534503dc95d0`;
independent audit SHA:
`72a7a64304688913bb9387a0c0342fd663d136f93f362102572871a504b8f1bc`.
